import os
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import ray
import torch
import vllm
import json
import math
import trafilatura
import traceback
import torch.nn.functional as F
import time
from json import JSONDecoder
from loguru import logger
from blingfire import text_to_sentences_and_offsets
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from models.attr_predictor import AttrPredictor
from models.utils_html import get_tables, clean_html
from utils.cragapi_wrapper import CRAG

logger.level("DEBUG")
CRAG_MOCK_API_URL = os.getenv("CRAG_MOCK_API_URL", "http://localhost:8000")


#### CONFIG PARAMETERS ---

# Define the number of context sentences to consider for generating an answer.
NUM_CONTEXT_SENTENCES = 10
# Set the maximum length for each context sentence (in characters).
MAX_CONTEXT_SENTENCE_LENGTH = 200

# Batch size you wish the evaluators will use to call the `batch_generate_answer` function
AICROWD_SUBMISSION_BATCH_SIZE = 8  # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.

# VLLM Parameters
VLLM_TENSOR_PARALLEL_SIZE = 4  # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.
VLLM_GPU_MEMORY_UTILIZATION = 0.9  # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.

# Sentence Transformer Parameters
SENTENTENCE_TRANSFORMER_BATCH_SIZE = 32  # TUNE THIS VARIABLE depending on the size of your embedding model and GPU mem available

# ---- Query prompt before calculating query embeddings
# ---- Need further investigation to see if this is useful
# QUERY_PROMPT_BGE = "Represent this sentence for searching relevant passages: "
QUERY_PROMPT = ""

#### CONFIG PARAMETERS END---


Entity_Extract_TEMPLATE = """
You are given a Query and Query Time. Do the following: 

1) Determine the domain the query is about. The domain should be one of the following: "finance", "sports", "music", "movie", "encyclopedia". If none of the domain applies, use "other". Use "domain" as the key in the result json. 
2) Extract structured information from the query. Include different keys into the result json depending on the domains, amd put them DIRECTLY in the result json. Here are the rules:

For `encyclopedia` and `other` queries, these are possible keys:
-  `main_entity`: extract the main entity of the query. 

For `finance` queries, these are possible keys:
- `market_identifier`: stock identifiers including individual company names, stock symbols.
- `metric`: financial metrics that the query is asking about. This must be one of the following: `price`, `dividend`, `P/E ratio`, `EPS`, `marketCap`, and `other`.
- `datetime`: time frame that query asks about. When datetime is not explicitly mentioned, use `Query Time` as default. 

For `movie` queries, these are possible keys:
- `movie_name`: name of the movie
- `movie_aspect`: if the query is about a movie, which movie aspect the query asks. This must be one of the following: `budget`, `genres`, `original_language`, `original_title`, `release_date`, `revenue`, `title`, `cast`, `crew`, `rating`, `length`.
- `person`: person name related to moves
- `person_aspect`: if the query is about a person, which person aspect the query asks. This must be one of the following: `acted_movies`, `directed_movies`, `oscar_awards`, `birthday`.
- `year`: if the query is about movies released in a specific year, extract the year

For `music` queries, these are possible keys:
- `artist_name`: name of the artist
- `artist_aspect`: if the query is about an artist, extract the aspect of the artist. This must be one of the following: `member`, `birth place`, `birth date`, `lifespan`, `artist work`, `grammy award count`, `grammy award date`.
- `song_name`: name of the song
- `song_aspect`: if the query is about a song, extract the aspect of the song. This must be one of the following: `auther`, `grammy award count`, `release country`, `release date`.

For `sports` queries, these are possible keys:
- `sport_type`: one of `basketball`, `soccer`, `other`
- `tournament`: such as NBA, World Cup, Olympic.
- `team`: teams that user interested in.
- `datetime`: time frame that user interested in. When datetime is not explicitly mentioned, use `Query Time` as default. 

Return the results in a FLAT json. 

*NEVER include ANY EXPLANATION or NOTE in the output, ONLY OUTPUT JSON*  
"""


def extract_json_objects(text, decoder=JSONDecoder()):
    """Find JSON objects in text, and yield the decoded JSON data"""
    pos = 0
    results = []
    while True:
        match = text.find("{", pos)
        if match == -1:
            break
        try:
            result, index = decoder.raw_decode(text[match:])
            results.append(result)
            pos = match + index
        except ValueError:
            pos = match + 1
    return results


START_WORDS = [
    "who",
    "what",
    "when",
    "where",
    "why",
    "how",
    "is",
    "can",
    "does",
    "do",
    "did",
    "will",
    "would",
    "could",
    "should",
    "are",
    "was",
    "were",
    "has",
    "have",
    "had",
    "which",
    "whom",
    "whose",
]


def is_question(sentence: str):
    return sentence.lower().endswith("?") or sentence.lower().startswith(
        tuple(START_WORDS)
    )


class ChunkExtractor:
    def __init__(self, max_context_sentence_length):
        self.max_context_sentence_length = max_context_sentence_length

    @ray.remote
    def _extract_chunks(self, interaction_id, html_source, html_snippet):
        """
        Extracts and returns chunks from given HTML source.

        Note: This function is for demonstration purposes only.
        We are treating an independent sentence as a chunk here,
        but you could choose to chunk your text more cleverly than this.

        Parameters:
            interaction_id (str): Interaction ID that this HTML source belongs to.
            html_source (str): HTML content from which to extract text.

        Returns:
            Tuple[str, List[str]]: A tuple containing the interaction ID and a list of sentences extracted from the HTML content.
        """
        # Parse the HTML content using BeautifulSoup
        text = trafilatura.extract(html_source, favor_precision=True, deduplicate=True)
        if text is None:
            soup = BeautifulSoup(html_source, "lxml")
            text = soup.get_text(
                " ", strip=True
            )  # Use space as a separator, strip whitespaces
        text.replace("\n", " ").replace("\t", " ").replace("\r", " ")
        text = text.strip()
        text = " ".join(text.split())

        if not text:
            # Return a list with empty string when no text is extracted
            return interaction_id, [""]

        # Extract offsets of sentences from the text
        _, offsets = text_to_sentences_and_offsets(text)

        # Initialize a list to store sentences
        chunks = []

        # Iterate through the list of offsets and extract sentences
        for start, end in offsets:
            # Extract the sentence and limit its length
            sentence = text[start:end]
            while len(sentence) > self.max_context_sentence_length:
                chunks.append(sentence[: self.max_context_sentence_length])
                sentence = sentence[self.max_context_sentence_length :]
            else:
                if is_question(sentence):
                    chunks.append(
                        "{} {}".format(
                            sentence, text[end : end + self.max_context_sentence_length]
                        )
                    )
                else:
                    chunks.append(sentence)

        return interaction_id, chunks

    def extract_chunks(self, batch_interaction_ids, batch_search_results):
        """
        Extracts chunks from given batch search results using parallel processing with Ray.

        Parameters:
            batch_interaction_ids (List[str]): List of interaction IDs.
            batch_search_results (List[List[Dict]]): List of search results batches, each containing HTML text.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing an array of chunks and an array of corresponding interaction IDs.
        """
        # Setup parallel chunk extraction using ray remote
        ray_response_refs = [
            self._extract_chunks.remote(
                self,
                interaction_id=batch_interaction_ids[idx],
                html_source=html_text["page_result"],
                html_snippet=html_text["page_snippet"],
            )
            for idx, search_results in enumerate(batch_search_results)
            for html_text in search_results
        ]

        # Wait until all sentence extractions are complete
        # and collect chunks for every interaction_id separately
        chunk_dictionary = defaultdict(list)

        for response_ref in ray_response_refs:
            interaction_id, _chunks = ray.get(
                response_ref
            )  # Blocking call until parallel execution is complete
            chunk_dictionary[interaction_id].extend(_chunks)

        # Flatten chunks and keep a map of corresponding interaction_ids
        chunks, chunk_interaction_ids = self._flatten_chunks(chunk_dictionary)

        return chunks, chunk_interaction_ids

    def _flatten_chunks(self, chunk_dictionary):
        """
        Flattens the chunk dictionary into separate lists for chunks and their corresponding interaction IDs.

        Parameters:
            chunk_dictionary (defaultdict): Dictionary with interaction IDs as keys and lists of chunks as values.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing an array of chunks and an array of corresponding interaction IDs.
        """
        chunks = []
        chunk_interaction_ids = []

        for interaction_id, _chunks in chunk_dictionary.items():
            # De-duplicate chunks within the scope of an interaction ID
            unique_chunks = list(set(_chunks))
            chunks.extend(unique_chunks)
            chunk_interaction_ids.extend([interaction_id] * len(unique_chunks))

        # Convert to numpy arrays for convenient slicing/masking operations later
        chunks = np.array(chunks)
        chunk_interaction_ids = np.array(chunk_interaction_ids)

        return chunks, chunk_interaction_ids


class ChunkTableExtractor:
    def __init__(self, max_context_sentence_length):
        self.max_context_sentence_length = max_context_sentence_length

    @ray.remote
    def _extract_chunks(
        self, interaction_id, html_source, html_snippet, html_name, include_tables
    ):
        """
        Extracts and returns chunks from given HTML source.

        Note: This function is for demonstration purposes only.
        We are treating an independent sentence as a chunk here,
        but you could choose to chunk your text more cleverly than this.

        Parameters:
            interaction_id (str): Interaction ID that this HTML source belongs to.
            html_source (str): HTML content from which to extract text.

        Returns:
            Tuple[str, List[str]]: A tuple containing the interaction ID and a list of sentences extracted from the HTML content.
        """
        # Parse the HTML content using BeautifulSoup
        text = trafilatura.extract(
            html_source,
            favor_precision=True,
            deduplicate=True,
            include_tables=include_tables,
        )
        if text is None:
            soup = BeautifulSoup(html_source, "lxml")
            text = soup.get_text(
                " ", strip=True
            )  # Use space as a separator, strip whitespaces
        text.replace("\n", " ").replace("\t", " ").replace("\r", " ")
        text = text.strip()
        text = " ".join(text.split())

        if not text:
            # Return a list with empty string when no text is extracted
            return interaction_id, [""], [""]

        # Extract offsets of sentences from the text
        _, offsets = text_to_sentences_and_offsets(text)

        # Initialize a list to store sentences
        chunks = []

        # Iterate through the list of offsets and extract sentences
        for start, end in offsets:
            # Extract the sentence and limit its length
            sentence = text[start:end]
            while len(sentence) > self.max_context_sentence_length:
                chunks.append(sentence[: self.max_context_sentence_length])
                sentence = sentence[self.max_context_sentence_length :]
            else:
                if is_question(sentence):
                    chunks.append(
                        "{} {}".format(
                            sentence, text[end : end + self.max_context_sentence_length]
                        )
                    )
                else:
                    chunks.append(sentence)

        final_chunks = [html_snippet]
        curr_sent = []
        for sent in chunks:
            if len(curr_sent) < 3:
                curr_sent.append(sent)
            else:
                final_chunks.append(" ".join(curr_sent))
                curr_sent = [sent]
        if len(curr_sent) > 0:
            final_chunks.append(" ".join(curr_sent))

        tables = get_tables(clean_html(html_source), html_name)

        return interaction_id, final_chunks, tables

    def extract_chunks(
        self, batch_interaction_ids, batch_search_results, include_tables
    ):
        """
        Extracts chunks from given batch search results using parallel processing with Ray.

        Parameters:
            batch_interaction_ids (List[str]): List of interaction IDs.
            batch_search_results (List[List[Dict]]): List of search results batches, each containing HTML text.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing an array of chunks and an array of corresponding interaction IDs.
        """
        # Setup parallel chunk extraction using ray remote
        ray_response_refs = [
            self._extract_chunks.remote(
                self,
                interaction_id=batch_interaction_ids[idx],
                html_source=html_text["page_result"],
                html_snippet=html_text["page_snippet"],
                html_name=html_text["page_name"],
                include_tables=include_tables,
            )
            for idx, search_results in enumerate(batch_search_results)
            for html_text in search_results
        ]

        # Wait until all sentence extractions are complete
        # and collect chunks for every interaction_id separately
        chunk_dictionary = defaultdict(list)
        table_dictionary = defaultdict(list)

        for response_ref in ray_response_refs:
            interaction_id, _chunks, _tables = ray.get(
                response_ref
            )  # Blocking call until parallel execution is complete
            chunk_dictionary[interaction_id].extend(_chunks)
            table_dictionary[interaction_id].extend(_tables)

        # Flatten chunks and keep a map of corresponding interaction_ids
        chunks, chunk_interaction_ids = self._flatten_chunks(chunk_dictionary)
        tables, table_interaction_ids = self._flatten_chunks(table_dictionary)

        return chunks, tables, chunk_interaction_ids, table_interaction_ids

    def _flatten_chunks(self, chunk_dictionary):
        """
        Flattens the chunk dictionary into separate lists for chunks and their corresponding interaction IDs.

        Parameters:
            chunk_dictionary (defaultdict): Dictionary with interaction IDs as keys and lists of chunks as values.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing an array of chunks and an array of corresponding interaction IDs.
        """
        chunks = []
        chunk_interaction_ids = []

        for interaction_id, _chunks in chunk_dictionary.items():
            # De-duplicate chunks within the scope of an interaction ID
            unique_chunks = list(set(_chunks))
            chunks.extend(unique_chunks)
            chunk_interaction_ids.extend([interaction_id] * len(unique_chunks))

        # Convert to numpy arrays for convenient slicing/masking operations later
        chunks = np.array(chunks)
        chunk_interaction_ids = np.array(chunk_interaction_ids)

        return chunks, chunk_interaction_ids


class CalcAgent:
    def __init__(self, max_table_length=6000):
        self.max_table_length = max_table_length
        self.expression_sample_num = 5

    def format_expression_prompts(
        self,
        batch_queries: list[str],
        batch_retrieval_results: list[list[str]],
        batch_tables: list[list[str]],
        tokenizer: AutoTokenizer,
    ) -> list[str]:
        system_prompt = """You are provided with a question and various references. Your task is to generate a possible useful expression that is needed to answer the question. Here are the rules:
1. The expression **MUST** be a valid Python expression.
2. The expression **MUST** be useful to answer the question.
3. If you think no expression is needed, you **MUST** answer with empty string.
4. The output should be succinct, you **MUST** do reasoning in your heart without outputing the reasoning.
5. You **MUST NOT** output any other words except the valid Python expression.
6. You **MUST NOT** output the expression that need the user to input anything.
"""
        formatted_prompts = []
        for _idx, query in enumerate(batch_queries):
            retrieval_results = batch_retrieval_results[_idx]
            related_tables = batch_tables[_idx]
            user_message = ""
            references = ""
            if len(retrieval_results) > 0:
                references += "# References \n"
                # Format the top sentences as references in the model's prompt template.
                for _snippet_idx, snippet in enumerate(retrieval_results):
                    references += f"- {snippet.strip()}\n"
            user_message += f"{references}\n------\n\n"

            if len(related_tables) > 0:
                table_references = ""
                user_message += "## Table references \n"
                for idx, table in enumerate(related_tables):
                    table_references += f"### Table {idx + 1}: \n"
                    table_references += f"{table}\n"
                table_references = table_references[: self.max_table_length]
                user_message += f"{table_references}\n------\n\n"
            user_message += "**Remember your rules**:"
            user_message += """
1. The expression **MUST** be a valid Python expression.
2. The expression **MUST** be useful to answer the question.
3. If you think no expression is needed, you **MUST** answer with empty string.
4. The output should be succinct, you **MUST** do reasoning in your heart without outputing the reasoning.
5. You **MUST NOT** output any other words except the valid Python expression.
6. You **MUST NOT** output the expression that need the user to input anything.
"""
            user_message += f"Question: {query}\n"
            user_message += f"Using the references listed above and based on the question, generate a valid Python expression for me: \n"

            formatted_prompts.append(
                tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        return formatted_prompts

    def get_calculation_results(
        self,
        llm: vllm.LLM,
        batch_queries: list[str],
        batch_retrieval_results: list[list[str]],
        batch_tables: list[list[str]],
    ) -> list[list[str]]:
        expression_prompts = self.format_expression_prompts(
            batch_queries, batch_retrieval_results, batch_tables, llm.get_tokenizer()
        )
        generated_expression_responses = llm.generate(
            expression_prompts,
            vllm.SamplingParams(
                n=self.expression_sample_num,  # Number of output sequences to return for each prompt.
                top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
                temperature=1.0,  # Randomness of the sampling
                skip_special_tokens=True,  # Whether to skip special tokens in the output.
                max_tokens=50,  # Maximum number of tokens to generate per output sequence.
            ),
            use_tqdm=False,  # you might consider setting this to True during local development
        )
        generated_expressions = []
        for response in generated_expression_responses:
            curr_expressions: list[str] = []
            curr_results: list[str] = []
            for _idx in range(self.expression_sample_num):
                try:
                    # FIXME: May have bugs in _idx?
                    curr_expression = response.outputs[_idx].text.strip().rstrip()
                except IndexError as e:
                    logger.error("Got index error: {}".format(traceback.format_exc()))
                    logger.error("IDX: {}".format(_idx))
                    logger.error("Response: {}".format(response))
                    logger.error("Outputs num: {}".format(len(response.outputs)))
                    continue
                if curr_expression in curr_expressions:
                    continue
                if "input(" in curr_expression:
                    continue
                if "print(" in curr_expression:
                    continue
                try:
                    exec_result = eval(curr_expression)
                    if type(exec_result) in [list, set, dict]:
                        raise Exception(
                            "Expression should not return a list, set or dict"
                        )
                    exec_result = str(exec_result)
                    curr_expressions.append(curr_expression)
                    curr_results.append(exec_result)
                except:
                    exec_result = None
            curr_generated_expressions = []
            for exp, res in zip(curr_expressions, curr_results):
                # logger.info(f"Expression: {exp}, Result: {res}")
                if res == "":
                    continue
                repr_res = repr(res)
                if repr_res.replace("'", '"') == exp.replace("'", '"'):
                    continue
                if res == exp:
                    continue
                curr_generated_expressions.append(
                    json.dumps({"Expression": exp, "Result": res})
                )
            # logger.info(f"Generated expressions: {curr_generated_expressions}")
            generated_expressions.append(curr_generated_expressions)
        return generated_expressions


class CustomSentenceEmbeddingModel:
    def __init__(self, model_path: str, max_length: int):
        self.device = torch.device("cuda:0")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        self.model = self.model.half().to(self.device)
        self.max_length = max_length

    @torch.inference_mode()
    def encode(
        self,
        sentences: list[str],
        normalize_embeddings: bool,
        batch_size: int,
        prompt: str = None,
    ):
        if isinstance(sentences, np.ndarray):
            sentences = sentences.tolist()
        if prompt:
            sentences = [f"{prompt}{sentence}" for sentence in sentences]
        with torch.autocast(device_type=self.device.type, dtype=torch.float16):
            all_embeddings = []
            while len(sentences) > batch_size:
                curr_batch = sentences[:batch_size]
                sentences = sentences[batch_size:]
                batch_dict = self.tokenizer(
                    curr_batch,
                    max_length=self.max_length,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.device)
                outputs = self.model(**batch_dict)
                embeddings = outputs.last_hidden_state[:, 0]
                if normalize_embeddings:
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings)
            else:
                batch_dict = self.tokenizer(
                    sentences,
                    max_length=self.max_length,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.device)
                outputs = self.model(**batch_dict)
                embeddings = outputs.last_hidden_state[:, 0]
                if normalize_embeddings:
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings)
            embeddings = torch.cat(all_embeddings, dim=0).cpu().detach().numpy()
        return embeddings


class KGRAGModel:
    """
    An example RAGModel for the KDDCup 2024 Meta CRAG Challenge
    which includes all the key components of a RAG lifecycle.
    """

    def __init__(
        self,
        batch_size=AICROWD_SUBMISSION_BATCH_SIZE,
        nprocs=VLLM_TENSOR_PARALLEL_SIZE,
        sentence_transformer_bsz=SENTENTENCE_TRANSFORMER_BATCH_SIZE,
        local_large_files_path=None,
        model_size="70B",
        custom_local_model_path=None,
        quantization=None,
        vllm_gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
        idk_attrs={"static_or_dynamic": ["dynamic"]},
        attr_combine_method="union",
        attr_predictor_method="few-shot",
        return_additional_info: bool = False,
        sentence_model_name: str = "sentence-t5-large",
        use_table_info: bool = True,
        max_table_length: int = 4000,
        num_context_sentences=NUM_CONTEXT_SENTENCES,
        max_context_sentence_length=MAX_CONTEXT_SENTENCE_LENGTH,
        log_while_inference=False,
        custom_sentence_embedding=False,
        query_prompt=QUERY_PROMPT,
        include_table_in_text=False,
        add_direct_answers=True,
        invalid_question_keys=["none", "never"],
    ):
        self.batch_size = batch_size
        self.nprocs = nprocs
        self.sentence_transformer_bsz = sentence_transformer_bsz
        self.local_large_files_path = local_large_files_path
        self.model_size = model_size
        self.custom_local_model_path = custom_local_model_path
        self.quantization = quantization
        self.vllm_gpu_memory_utilization = vllm_gpu_memory_utilization
        if self.model_size == "70B" and quantization is None:
            self.quantization = "gptq"
        assert self.model_size in ["8B", "70B"]
        self.idk_attrs = idk_attrs
        self.attr_combine_method = attr_combine_method
        assert self.attr_combine_method in ["union", "intersection"]
        self.return_additional_info = return_additional_info
        self.sentence_model_name = sentence_model_name
        self.use_table_info = use_table_info
        self.max_table_length = max_table_length
        self.num_context_sentences = num_context_sentences
        self.max_context_sentence_length = max_context_sentence_length
        self.log_while_inference = log_while_inference
        self.custom_sentence_embedding = custom_sentence_embedding
        self.query_prompt = query_prompt
        self.include_table_in_text = include_table_in_text
        self.add_direct_answers = add_direct_answers
        self.invalid_question_keys = invalid_question_keys
        self.initialize_models()
        self.attr_predictor = None
        if self.idk_attrs is not None:
            assert attr_predictor_method is not None
            for attr_name in self.idk_attrs.keys():
                assert attr_name in ["domain", "question_type", "static_or_dynamic"]
            self.attr_predictor = AttrPredictor(attr_predictor_method, self.llm)
        if self.use_table_info:
            self.chunk_extractor = ChunkTableExtractor(self.max_context_sentence_length)
        else:
            self.chunk_extractor = ChunkExtractor(self.max_context_sentence_length)

        self.log_direct_answer_times = 10
        self.log_false_premise_times = math.inf

        self.calculator = CalcAgent(self.max_table_length)

        # print all configs
        logger.info("Configs:")
        logger.info(
            json.dumps(
                {
                    "self.batch_size": self.batch_size,
                    "self.nprocs": self.nprocs,
                    "self.sentence_transformer_bsz": self.sentence_transformer_bsz,
                    "self.local_large_files_path": self.local_large_files_path,
                    "self.model_size": self.model_size,
                    "self.custom_local_model_path": self.custom_local_model_path,
                    "self.quantization": self.quantization,
                    "self.vllm_gpu_memory_utilization": self.vllm_gpu_memory_utilization,
                    "self.idk_attrs": self.idk_attrs,
                    "self.return_additional_info": self.return_additional_info,
                    "self.sentence_model_name": self.sentence_model_name,
                    "self.use_table_info": self.use_table_info,
                    "self.max_table_length": self.max_table_length,
                    "self.num_context_sentences": self.num_context_sentences,
                    "self.max_context_sentence_length": self.max_context_sentence_length,
                    "self.log_while_inference": self.log_while_inference,
                    "self.custom_sentence_embedding": self.custom_sentence_embedding,
                    "self.query_prompt": self.query_prompt,
                    "self.include_table_in_text": self.include_table_in_text,
                    "self.add_direct_answers": self.add_direct_answers,
                    "self.invalid_question_keys": self.invalid_question_keys,
                },
                indent=2,
            )
        )

    def initialize_models(self):
        # Initialize Meta Llama 3 - 8B Instruct Model
        if self.model_size == "8B":
            self.model_name = f"large-files/ckpts/Meta-Llama-3-8B-Instruct/"
        else:
            self.model_name = f"large-files/ckpts/Llama3-70b-instruct-gptq-new/"
        if self.local_large_files_path is not None:
            self.model_name = os.path.join(self.local_large_files_path, self.model_name)
        if self.custom_local_model_path is not None:
            self.model_name = self.custom_local_model_path
        logger.info(self.model_name)
        logger.info(self.quantization)

        if not os.path.exists(self.model_name):
            raise Exception(
                f"""
            The evaluators expect the model weights to be checked into the repository,
            but we could not find the model weights at {self.model_name}
            
            Please follow the instructions in the docs below to download and check in the model weights.
            
            https://gitlab.aicrowd.com/aicrowd/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/meta-comphrehensive-rag-benchmark-starter-kit/-/blob/master/docs/dataset.md
            """
            )

        # Initialize the model with vllm
        self.llm = vllm.LLM(
            self.model_name,
            tensor_parallel_size=self.nprocs,
            quantization=self.quantization,
            gpu_memory_utilization=self.vllm_gpu_memory_utilization,
            trust_remote_code=True,
            dtype="half",  # note: bfloat16 is not supported on nvidia-T4 GPUs
            enforce_eager=True,
            disable_custom_all_reduce=True,
        )
        self.tokenizer = self.llm.get_tokenizer()

        # Load a sentence transformer model optimized for sentence embeddings, using CUDA if available.
        sentence_model_path = (
            f"large-files/ckpts/sentence-transformers/{self.sentence_model_name}"
            if self.local_large_files_path is None
            else os.path.join(
                self.local_large_files_path,
                f"large-files/ckpts/sentence-transformers/{self.sentence_model_name}",
            )
        )
        logger.info("Sentence model path: {}".format(sentence_model_path))
        if self.custom_sentence_embedding:
            self.sentence_model = CustomSentenceEmbeddingModel(
                sentence_model_path, max_length=self.max_context_sentence_length
            )
        else:
            self.sentence_model = SentenceTransformer(
                sentence_model_path, device=torch.device("cpu"), trust_remote_code=True
            )
            self.sentence_model = self.sentence_model.half().cuda(0)

    def calculate_embeddings(self, sentences, prompt: str = None):
        """
        Compute normalized embeddings for a list of sentences using a sentence encoding model.

        This function leverages multiprocessing to encode the sentences, which can enhance the
        processing speed on multi-core machines.

        Args:
            sentences (List[str]): A list of sentences for which embeddings are to be computed.

        Returns:
            np.ndarray: An array of normalized embeddings for the given sentences.

        """
        embeddings = self.sentence_model.encode(
            sentences=sentences,
            normalize_embeddings=True,
            batch_size=self.sentence_transformer_bsz,
            prompt=prompt,
        )
        # Note: There is an opportunity to parallelize the embedding generation across 4 GPUs
        #       but sentence_model.encode_multi_process seems to interefere with Ray
        #       on the evaluation servers.
        #       todo: this can also be done in a Ray native approach.
        #
        return embeddings

    def get_batch_size(self) -> int:
        """
        Determines the batch size that is used by the evaluator when calling the `batch_generate_answer` function.

        The evaluation timeouts linearly scale with the batch size.
            i.e.: time out for the `batch_generate_answer` call = batch_size * per_sample_timeout


        Returns:
            int: The batch size, an integer between 1 and 16. It can be dynamic
                 across different batch_generate_answer calls, or stay a static value.
        """
        return self.batch_size

    def format_prompts_for_entity_extraction(self, queries, query_times):
        formatted_prompts = []
        for _idx, query in enumerate(queries):
            query_time = query_times[_idx]
            user_message = ""
            user_message += f"Query: {query}\n"
            user_message += f"Query Time: {query_time}\n"

            formatted_prompts.append(
                self.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": Entity_Extract_TEMPLATE},
                        {"role": "user", "content": user_message},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        return formatted_prompts

    def extract_entity(self, queries, query_times) -> list[dict]:
        formatted_prompts = self.format_prompts_for_entity_extraction(
            queries, query_times
        )
        responses = self.llm.generate(
            formatted_prompts,
            vllm.SamplingParams(
                n=1,  # Number of output sequences to return for each prompt.
                top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
                temperature=0.1,  # Randomness of the sampling
                skip_special_tokens=True,  # Whether to skip special tokens in the output.
                max_tokens=512,  # Maximum number of tokens to generate per output sequence.
            ),
            use_tqdm=False,  # you might consider setting this to True during local development
        )

        entities = []
        for response in responses:
            res = response.outputs[0].text
            try:
                res = json.loads(res)
            except:
                res = extract_json_objects(res)
            entities.append(res)
        return entities

    def get_kg_results(self, entities) -> list[list[str]]:
        # examples for "open" (encyclopedia),  "movie" or "other" domains
        api = CRAG(server=CRAG_MOCK_API_URL)
        batch_kg_results = []
        for entity in entities:
            kg_results = []
            res = ""
            if "domain" in entity.keys():
                domain = entity["domain"]
                if domain in ["encyclopedia", "other"]:
                    if "main_entity" in entity.keys():
                        try:
                            top_entity_name = api.open_search_entity_by_name(
                                entity["main_entity"]
                            )["result"][0]
                            res = api.open_get_entity(top_entity_name)["result"]
                            kg_results.append({top_entity_name: res})
                        except Exception as e:
                            logger.warning(f"Error in open_get_entity: {e}")
                            pass
                if domain == "movie":
                    if (
                        "movie_name" in entity.keys()
                        and entity["movie_name"] is not None
                    ):
                        if isinstance(entity["movie_name"], str):
                            movie_names = entity["movie_name"].split(",")
                        else:
                            movie_names = entity["movie_name"]
                        for movie_name in movie_names:
                            try:
                                res = api.movie_get_movie_info(movie_name)["result"][0]
                                res = res[entity["movie_aspect"]]
                                kg_results.append(
                                    {movie_name + "_" + entity["movie_aspect"]: res}
                                )
                            except Exception as e:
                                logger.warning(f"Error in movie_get_movie_info: {e}")
                                pass
                    if "person" in entity.keys() and entity["person"] is not None:
                        if isinstance(entity["person"], str):
                            person_list = entity["person"].split(",")
                        else:
                            person_list = entity["person"]
                        for person in person_list:
                            try:
                                res = api.movie_get_person_info(person)["result"][0]
                                aspect = entity["person_aspect"]
                                if aspect in ["oscar_awards", "birthday"]:
                                    res = res[aspect]
                                    kg_results.append({person + "_" + aspect: res})
                                if aspect in ["acted_movies", "directed_movies"]:
                                    movie_info = []
                                    for movie_id in res[aspect]:
                                        movie_info.append(
                                            api.movie_get_movie_info_by_id(movie_id)
                                        )
                                    kg_results.append(
                                        {person + "_" + aspect: movie_info}
                                    )
                            except Exception as e:
                                logger.warning(f"Error in movie_get_person_info: {e}")
                                pass
                    if "year" in entity.keys() and entity["year"] is not None:
                        if isinstance(entity["year"], str) or isinstance(
                            entity["year"], int
                        ):
                            years = str(entity["year"]).split(",")
                        else:
                            years = entity["year"]
                        for year in years:
                            try:
                                res = api.movie_get_year_info(year)["result"]
                                all_movies = []
                                oscar_movies = []
                                for movie_id in res["movie_list"]:
                                    all_movies.append(
                                        api.movie_get_movie_info_by_id(movie_id)[
                                            "result"
                                        ]["title"]
                                    )
                                for movie_id in res["oscar_awards"]:
                                    oscar_movies.append(
                                        api.movie_get_movie_info_by_id(movie_id)[
                                            "result"
                                        ]["title"]
                                    )
                                kg_results.append({year + "_all_movies": all_movies})
                                kg_results.append(
                                    {year + "_oscar_movies": oscar_movies}
                                )
                            except Exception as e:
                                logger.warning(f"Error in movie_get_year_info: {e}")
                                pass
            batch_kg_results.append([str(res) for res in kg_results])
        return batch_kg_results

    def batch_generate_answer(self, batch: Dict[str, Any]) -> List[str]:
        """
        Generates answers for a batch of queries using associated (pre-cached) search results and query times.

        Parameters:
            batch (Dict[str, Any]): A dictionary containing a batch of input queries with the following keys:
                - 'interaction_id;  (List[str]): List of interaction_ids for the associated queries
                - 'query' (List[str]): List of user queries.
                - 'search_results' (List[List[Dict]]): List of search result lists, each corresponding
                                                      to a query. Please refer to the following link for
                                                      more details about the individual search objects:
                                                      https://gitlab.aicrowd.com/aicrowd/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/meta-comphrehensive-rag-benchmark-starter-kit/-/blob/master/docs/dataset.md#search-results-detail
                - 'query_time' (List[str]): List of timestamps (represented as a string), each corresponding to when a query was made.

        Returns:
            List[str]: A list of plain text responses for each query in the batch. Each response is limited to 75 tokens.
            If the generated response exceeds 75 tokens, it will be truncated to fit within this limit.

        Notes:
        - If the correct answer is uncertain, it's preferable to respond with "I don't know" to avoid
          the penalty for hallucination.
        - Response Time: Ensure that your model processes and responds to each query within 30 seconds.
          Failing to adhere to this time constraint **will** result in a timeout during evaluation.
        """
        batch_interaction_ids = batch["interaction_id"]
        queries = batch["query"]
        batch_search_results = batch["search_results"]
        query_times = batch["query_time"]
        curr_start_time = time.perf_counter()

        # Chunk all search results using ChunkExtractor
        if self.use_table_info:
            chunks, tables, chunk_interaction_ids, table_interaction_ids = (
                self.chunk_extractor.extract_chunks(
                    batch_interaction_ids,
                    batch_search_results,
                    include_tables=self.include_table_in_text,
                )
            )
        else:
            chunks, chunk_interaction_ids = self.chunk_extractor.extract_chunks(
                batch_interaction_ids, batch_search_results
            )
            tables = []

        # Calculate all chunk embeddings
        chunk_embeddings = self.calculate_embeddings(chunks)

        # Calculate embeddings for queries
        query_embeddings = self.calculate_embeddings(
            queries,
            prompt=self.query_prompt,
        )

        # Retrieve top matches for the whole batch
        batch_retrieval_results = []
        batch_tables = []
        for _idx, interaction_id in enumerate(batch_interaction_ids):
            query = queries[_idx]
            query_time = query_times[_idx]
            query_embedding = query_embeddings[_idx]

            # Identify chunks that belong to this interaction_id
            relevant_chunks_mask = chunk_interaction_ids == interaction_id

            # Filter out the said chunks and corresponding embeddings
            relevant_chunks = chunks[relevant_chunks_mask]
            if len(tables) > 0:
                relavant_tables_mask = table_interaction_ids == interaction_id
                relevant_tables = tables[relavant_tables_mask]
            else:
                relevant_tables = []
            relevant_chunks_embeddings = chunk_embeddings[relevant_chunks_mask]

            # Calculate cosine similarity between query and chunk embeddings,
            cosine_scores = (relevant_chunks_embeddings * query_embedding).sum(1)

            # and retrieve top-N results.
            retrieval_results = relevant_chunks[
                (-cosine_scores).argsort()[: self.num_context_sentences]
            ]

            # You might also choose to skip the steps above and
            # use a vectorDB directly.
            batch_retrieval_results.append(retrieval_results)
            batch_tables.append(relevant_tables)
        logger.info(
            "Time for HTML processing: {:.2f}s".format(
                time.perf_counter() - curr_start_time
            )
        )
        curr_start_time = time.perf_counter()

        answers = [None for _ in queries]
        reasoning_prompts = [None for _ in queries]
        all_reasoning_outputs = [None for _ in queries]
        pred_idxs = list(range(len(queries)))
        all_predicted_attrs = {}
        if self.idk_attrs:
            idk_idxs = []
            for attr_name in self.idk_attrs.keys():
                curr_idk_idxs = set()
                predicted_attrs = self.attr_predictor.predict_attr(batch, attr_name)
                all_predicted_attrs[attr_name] = predicted_attrs
                for idx, pred_attr in enumerate(predicted_attrs):
                    if pred_attr in self.idk_attrs[attr_name]:
                        # answers[idx] = "I don't know."
                        curr_idk_idxs.add(idx)
                idk_idxs.append(curr_idk_idxs)
            # get the intersection of idx_idxs
            if self.attr_combine_method == "union":
                idk_idxs = set.union(*idk_idxs)
            else:
                idk_idxs = set.intersection(*idk_idxs)
            for idk_idx in idk_idxs:
                answers[idk_idx] = "I don't know."
                pred_idxs.remove(idk_idx)
        logger.info(
            "Time for attribute prediction: {:.2f}s".format(
                time.perf_counter() - curr_start_time
            )
        )
        curr_start_time = time.perf_counter()

        queries = [queries[idx] for idx in pred_idxs]
        query_times = [query_times[idx] for idx in pred_idxs]
        batch_retrieval_results = [batch_retrieval_results[idx] for idx in pred_idxs]
        batch_tables = [batch_tables[idx] for idx in pred_idxs]

        calculation_results: list[list[str]] = self.calculator.get_calculation_results(
            self.llm, queries, batch_retrieval_results, batch_tables
        )
        logger.info(
            "Time for calculation_results: {:.2f}s".format(
                time.perf_counter() - curr_start_time
            )
        )
        curr_start_time = time.perf_counter()

        if self.add_direct_answers:
            direct_answers = self.get_direct_answers(queries)
        else:
            direct_answers = [None for _ in queries]
        logger.info(
            "Time for direct_answers: {:.2f}s".format(
                time.perf_counter() - curr_start_time
            )
        )
        curr_start_time = time.perf_counter()

        entities = self.extract_entity(queries, query_times)
        logger.info(
            "Time for entity extraction: {:.2f}s".format(
                time.perf_counter() - curr_start_time
            )
        )
        curr_start_time = time.perf_counter()
        logger.debug("Entities: {}".format(json.dumps(entities)))
        batch_kg_results = self.get_kg_results(entities)
        logger.info(
            "Time for KG retrieval: {:.2f}s".format(
                time.perf_counter() - curr_start_time
            )
        )
        curr_start_time = time.perf_counter()

        reasoning_formatted_prompts = self.format_reasoning_prompts(
            queries,
            query_times,
            batch_retrieval_results,
            batch_tables,
            calculation_results,
            direct_answers,
            batch_kg_results,
        )
        for _idx, _reasoning_prompt in zip(pred_idxs, reasoning_formatted_prompts):
            reasoning_prompts[_idx] = _reasoning_prompt
        if len(pred_idxs) == 0:
            curr_sample_num = 1
        else:
            curr_sample_num = max(int(4 / (len(pred_idxs))), 1)

        reasoning_responses = self.llm.generate(
            reasoning_formatted_prompts,
            vllm.SamplingParams(
                n=curr_sample_num,  # Number of output sequences to return for each prompt.
                top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
                temperature=0.5,  # Randomness of the sampling
                skip_special_tokens=True,  # Whether to skip special tokens in the output.
                max_tokens=512,  # Maximum number of tokens to generate per output sequence.
            ),
            use_tqdm=False,  # you might consider setting this to True during local development
        )
        reasoning_answers = []
        pred_to_pop = []
        parse_succ = 0
        parse_fail = 0
        for idx, response in enumerate(reasoning_responses):
            assert answers[pred_idxs[idx]] is None, (
                answers,
                answers[pred_idxs[idx]],
                pred_idxs,
                idx,
                response,
            )
            all_ret_answers = [o.text.strip().rstrip() for o in response.outputs]
            all_reasoning_outputs[pred_idxs[idx]] = all_ret_answers
            is_idk = False
            for tmp_ret_answer in all_ret_answers:
                if tmp_ret_answer == "" or "i don't know" in tmp_ret_answer.lower():
                    tmp_ret_answer = "I don't know."
                    answers[pred_idxs[idx]] = tmp_ret_answer
                    pred_to_pop.append(idx)
                    is_idk = True
                    break
            if is_idk:
                all_reasoning_outputs[pred_idxs[idx]].append(
                    {"Parse success": True, "Parsed result": "i don't know"}
                )
                continue
            is_false_premise = False
            for tmp_ret_answer in all_ret_answers:
                if "invalid question" in tmp_ret_answer.lower() or (
                    tmp_ret_answer.lower() in self.invalid_question_keys
                ):
                    tmp_ret_answer = "invalid question"
                    answers[pred_idxs[idx]] = tmp_ret_answer
                    pred_to_pop.append(idx)
                    is_false_premise = True
                    break
            if is_false_premise:
                all_reasoning_outputs[pred_idxs[idx]].append(
                    {"Parse success": True, "Parsed result": "invalid question"}
                )
                continue
            ret_answer = response.outputs[0].text.strip().rstrip()
            try:
                ret_answer_lower = ret_answer.lower()
                assert "## answer:" in ret_answer_lower
                curr_final_answer = (
                    ret_answer_lower.split("## answer:")[1].strip().rstrip()
                )
                # assert curr_final_answer.endswith("===end===")
                curr_final_answer = (
                    curr_final_answer.split("===end===")[0].strip().rstrip()
                )

                assert "## false premise:" in curr_final_answer
                curr_final_answer, false_premise = curr_final_answer.split(
                    "## false premise:"
                )
                curr_final_answer = curr_final_answer.strip().rstrip()
                false_premise = false_premise.strip().rstrip().lower()
                if self.log_false_premise_times:
                    self.log_false_premise_times -= 1
                    logger.debug(
                        json.dumps(
                            {
                                "log times left": self.log_false_premise_times,
                                "query": queries[idx],
                                "raw ret_answer": ret_answer,
                                "curr_final_answer": curr_final_answer,
                                "false_premise": false_premise,
                            },
                            indent=2,
                        )
                    )
                if false_premise in ["yes", "true", "1"]:
                    curr_final_answer = "invalid question"

                assert len(curr_final_answer) < 300
                if "invalid question" in curr_final_answer.lower() or (
                    curr_final_answer.lower() in self.invalid_question_keys
                ):
                    curr_final_answer = "invalid question"
                answers[pred_idxs[idx]] = curr_final_answer.strip()
                pred_to_pop.append(idx)
                parse_succ += 1
                all_reasoning_outputs[pred_idxs[idx]].append(
                    {"Parse success": True, "Parsed result": curr_final_answer.strip()}
                )
                continue
            except:
                logger.info(
                    "Failed to parse str. Fallback to use summarize the reasoning."
                )
                logger.info(traceback.format_exc())
                logger.info(ret_answer)
                parse_fail += 1
                all_reasoning_outputs[pred_idxs[idx]].append(
                    {"Parse success": False, "Parsed result": ""}
                )
                reasoning_answers.append(ret_answer)
        logger.info(
            "Curr batch:\n\tparse successfully: {}\n\tparse failed: {}".format(
                parse_succ, parse_fail
            )
        )
        logger.info(
            "Time for reasoning: {:.2f}s".format(time.perf_counter() - curr_start_time)
        )
        curr_start_time = time.perf_counter()
        pred_idxs = [
            pred_idxs[idx] for idx in range(len(pred_idxs)) if idx not in pred_to_pop
        ]
        queries = [
            queries[idx] for idx in range(len(queries)) if idx not in pred_to_pop
        ]
        query_times = [
            query_times[idx]
            for idx in range(len(query_times))
            if idx not in pred_to_pop
        ]
        batch_retrieval_results = [
            batch_retrieval_results[idx]
            for idx in range(len(batch_retrieval_results))
            if idx not in pred_to_pop
        ]
        batch_tables = [
            batch_tables[idx]
            for idx in range(len(batch_tables))
            if idx not in pred_to_pop
        ]
        calculation_results = [
            calculation_results[idx]
            for idx in range(len(calculation_results))
            if idx not in pred_to_pop
        ]

        # Prepare formatted prompts from the LLM
        formatted_prompts = self.format_prompts(
            queries,
            reasoning_answers,
        )
        # Generate responses via vllm
        responses = self.llm.generate(
            formatted_prompts,
            vllm.SamplingParams(
                n=1,  # Number of output sequences to return for each prompt.
                top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
                temperature=0.1,  # Randomness of the sampling
                skip_special_tokens=True,  # Whether to skip special tokens in the output.
                max_tokens=50,  # Maximum number of tokens to generate per output sequence.
            ),
            use_tqdm=False,  # you might consider setting this to True during local development
        )
        # Aggregate answers into List[str]
        prompt_lens = []
        for idx, response in enumerate(responses):
            assert answers[pred_idxs[idx]] is None, (
                answers,
                answers[pred_idxs[idx]],
                pred_idxs,
                idx,
            )
            ret_answer = response.outputs[0].text.strip().rstrip()
            prompt_lens.append(len(response.prompt_token_ids))
            if ret_answer == "":
                ret_answer = "I don't know."
            if "i don't know" in ret_answer.lower():
                ret_answer = "I don't know."
            if "i do not know" in ret_answer.lower():
                ret_answer = "I don't know."
            if "invalid question" in ret_answer.lower() or (
                ret_answer.lower() in self.invalid_question_keys
            ):
                ret_answer = "invalid question"
            answers[pred_idxs[idx]] = ret_answer.strip()

        if self.log_while_inference:
            logger.info("Batch queries:")
            logger.info(json.dumps(queries, indent=2))
            logger.info("Batch answers:")
            logger.info(json.dumps(answers, indent=2))
            logger.info(f"Prompt lengths: {prompt_lens}")
            logger.info("Predicted attributes:")
            logger.info(json.dumps(all_predicted_attrs, indent=2))

        formatted_prompts_final = []
        prompt_lens_final = []
        for idx in range(len(answers)):
            if idx in pred_idxs:
                prompt_lens_final.append(prompt_lens.pop(0))
                formatted_prompts_final.append(formatted_prompts.pop(0))
            else:
                prompt_lens_final.append(0)
                formatted_prompts_final.append("")
        logger.info(
            "Time for final generation: {:.2f}s".format(
                time.perf_counter() - curr_start_time
            )
        )

        if self.return_additional_info:
            return answers, {
                "all_predicted_attrs": all_predicted_attrs,
                "prompt_lens": prompt_lens_final,
                "prompts": formatted_prompts_final,
                "generated_expressions": calculation_results,
                "reasoning_prompts": reasoning_prompts,
                "reasoning_outputs": all_reasoning_outputs,
            }
        else:
            return answers

    def get_direct_answers(self, queries) -> list[str]:
        system_prompt = """You are provided with a question.
Your task is to answer the question with your reasoning process.
If you can't answer it directly based on your knowledge, respond with 'I don't know'.
If you think the premise of the question is wrong, for example, the question asks information about a person's husband, but you are sure that the person doesn't have one, you should answer with "Invalid question" without any other words.
You **MUST** think if the question has a false premise, then think the final answer.
You **MUST** generate the reasoning process before the answer. You **MUST** generate your output with the following format:

===START===
## Reasoning:
- Does it have a false premise?
<YOUR REASONING>
- What is the final answer?
<YOUR REASONING>
------
## Answer:
<YOUR FINAL ANSWER>
===END===

**IMPORTANT RULES**:
- If you can't answer it directly based on your knowledge, respond with 'I don't know'.
- Your generation **MUST** starts with "===START===" and ends with "===END===".
- `<YOUR FINAL ANSWER>` should be succinct, and use as few words as possible.
- `<YOUR REASONING>` should be a detailed reasoning process that explains how you arrived at your answer.
- If you think the premise of the question is wrong, for example, the question asks information about a person's husband, but you are sure that the person doesn't have one, you should answer with "Invalid question" without any other words.
Let's think step by step now!"""
        formatted_prompts = []
        for _idx, query in enumerate(queries):
            user_message = query
            formatted_prompts.append(
                self.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        direct_reasoning_responses = self.llm.generate(
            formatted_prompts,
            vllm.SamplingParams(
                n=1,  # Number of output sequences to return for each prompt.
                top_p=0.01,  # Float that controls the cumulative probability of the top tokens to consider.
                temperature=0.5,  # Randomness of the sampling
                skip_special_tokens=True,  # Whether to skip special tokens in the output.
                max_tokens=512,  # Maximum number of tokens to generate per output sequence.
            ),
            use_tqdm=False,  # you might consider setting this to True during local development
        )
        direct_answers = []
        for idx, response in enumerate(direct_reasoning_responses):
            response_text = response.outputs[0].text.strip().rstrip()
            if response_text == "" or "i don't know" in response_text.lower():
                response_text = "I don't know."
                direct_answers.append(response_text)
                continue
            if "invalid question" in response_text.lower() or (
                response_text.lower() in self.invalid_question_keys
            ):
                response_text = "invalid question"
                direct_answers.append(response_text)
                continue
            try:
                response_text_lower = response_text.lower()
                assert "## answer:" in response_text_lower
                curr_final_answer = (
                    response_text_lower.split("## answer:")[1].strip().rstrip()
                )
                # assert curr_final_answer.endswith("===end===")
                curr_final_answer = (
                    curr_final_answer.split("===end===")[0].strip().rstrip()
                )
                if curr_final_answer == "":
                    direct_answers.append("i don't know")
                    continue
                if "invalid question" in curr_final_answer.lower() or (
                    curr_final_answer.lower() in self.invalid_question_keys
                ):
                    direct_answers.append("invalid question")
                    continue
                direct_answers.append(curr_final_answer.strip())
            except:
                direct_answers.append("i don't know")
        assert len(direct_answers) == len(queries)
        if self.log_direct_answer_times > 0:
            self.log_direct_answer_times -= 1
            logger.debug(
                f"Example direct answer {self.log_direct_answer_times}: {direct_answers[0]}"
            )
        return direct_answers

    def format_reasoning_prompts(
        self,
        queries,
        query_times,
        batch_retrieval_results: list[list[str]] = [],
        batch_table_results: list[list[str]] = [],
        batch_calculation_results: list[list[str]] = [],
        direct_answers: list[str] = [],
        batch_kg_results: list[list[str]] = [],
    ):
        assert (
            len(queries)
            == len(query_times)
            == len(batch_retrieval_results)
            == len(batch_table_results)
            == len(batch_calculation_results)
            == len(direct_answers)
            == len(batch_kg_results)
        )
        system_prompt = """You are provided with a question and various references.
Your task is to answer the question with your reasoning process.
There are also some calculation results from another agent, which may be useful for you.
There is an answer from another agent which may be useful. It may have hallucination. You need to judge whether to trust it by yourself.
If the references do not contain the necessary information to answer the question and you can't answer it directly based on your knowledge, respond with 'I don't know'.
If you think the premise of the question is wrong, for example, the question asks information about a person's husband, but you are sure that the person doesn't have one, you should answer with "Invalid question" without any other words.
You **MUST** think if the question has a false premise, then think the final answer.
You **MUST** generate the reasoning process before the answer. You **MUST** generate your output with the following format:

===START===
## Reasoning:
- Does it have a false premise?
<YOUR REASONING>
- What is the final answer?
<YOUR REASONING>
- Can you answer it based on current knowledge?
<YOUR REASONING>
------
## Answer:
<YOUR FINAL ANSWER>
## False Premise:
<HAS_FALSE_PREMISE_OR_NOT>
===END===

**IMPORTANT RULES**:
- If the references do not contain the necessary information to answer the question and you can't answer it directly based on your knowledge, respond with 'I don't know'.
- Your generation **MUST** starts with "===START===" and ends with "===END===".
- `<YOUR FINAL ANSWER>` should be succinct, and use as few words as possible.
- `<YOUR REASONING>` should be a detailed reasoning process that explains how you arrived at your answer.
- `<HAS_FALSE_PREMISE_OR_NOT>` should be "yes" if the premise is wrong and the question is invalid, and "no" otherwise. It can **ONLY** be chosen from these two options.
- If you think the premise of the question is wrong, for example, the question asks information about a person's husband, but you are sure that the person doesn't have one, you should answer with "Invalid question" without any other words.
Let's think step by step now!"""
        formatted_prompts = []
        for _idx, query in enumerate(queries):
            query_time = query_times[_idx]
            retrieval_results = batch_retrieval_results[_idx]
            related_tables = batch_table_results[_idx]
            generated_expressions = batch_calculation_results[_idx]
            direct_answer = direct_answers[_idx]
            kg_results = batch_kg_results[_idx]
            user_message = ""
            references = ""
            if len(retrieval_results) > 0:
                references += "# References \n"
                # Format the top sentences as references in the model's prompt template.
                for _snippet_idx, snippet in enumerate(retrieval_results):
                    references += f"- {snippet.strip()}\n"
            user_message += f"{references}\n------\n\n"
            if len(kg_results) > 0:
                kg_references = ""
                user_message += "## Knowledge Graph references \n"
                for idx, kg_result in enumerate(kg_results):
                    kg_references += f"### KG Ref {idx + 1}: \n"
                    kg_references += f"{kg_result}\n"
                kg_references = kg_references[:1000]
                user_message += f"{kg_references}\n------\n\n"
                logger.debug("Currect KG references:\n{}".format(kg_references))
            if len(related_tables) > 0:
                table_references = ""
                user_message += "## Table references \n"
                for idx, table in enumerate(related_tables):
                    table_references += f"### Table {idx + 1}: \n"
                    table_references += f"{table}\n"
                table_references = table_references[: self.max_table_length]
                user_message += f"{table_references}\n------\n\n"
            if len(generated_expressions) > 0:
                expression_references = ""
                user_message += "## Possible useful calculation results \n"
                for idx, expression in enumerate(generated_expressions):
                    expression_references += f"### Calculation {idx + 1}: \n"
                    expression_references += f"{expression}\n"
                user_message += f"{expression_references}\n------\n\n"
            if direct_answer is not None:
                user_message += (
                    f"# An answer from another agent:\n{direct_answer}\n------\n\n"
                )
            user_message += """**Remember your IMPORTANT RULES**:
- If the references do not contain the necessary information to answer the question and you can't answer it directly based on your knowledge, respond with 'I don't know'.
- Your generation **MUST** starts with "===START===" and ends with "===END===".
- `<YOUR FINAL ANSWER>` should be succinct, and use as few words as possible.
- `<YOUR REASONING>` should be a detailed reasoning process that explains how you arrived at your answer.
- `<HAS_FALSE_PREMISE_OR_NOT>` should be "yes" if the question is invalid, and "no" otherwise. It can **ONLY** be chosen from these two options.
- If you think the premise of the question is wrong, for example, the question asks information about a person's husband, but you are sure that the person doesn't have one, you should answer with "Invalid question" without any other words.
"""
            user_message += (
                f"Using the references listed above, answer the following question: \n"
            )
            user_message += f"Current Time: {query_time}\n"
            user_message += f"Question: {query}\n"
            user_message += "Let's think step by step now!\n"
            formatted_prompts.append(
                self.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        return formatted_prompts

    def format_prompts(
        self,
        queries,
        reasoning_answers: list[str] = [],
    ):
        """
        Formats queries, corresponding query_times and retrieval results using the chat_template of the model.

        Parameters:
        - queries (List[str]): A list of queries to be formatted into prompts.
        - query_times (List[str]): A list of query_time strings corresponding to each query.
        - batch_retrieval_results (List[str])
        """
        assert len(queries) == len(reasoning_answers)
        system_prompt = "You are provided with a question and a reasoning process from another agent. Your task is to summarize the reasoning process and finally answer the question succinctly, using the fewest words possible."
        formatted_prompts = []

        for _idx, query in enumerate(queries):
            reasoning_process = reasoning_answers[_idx]

            user_message = ""
            user_message += f"Question: {query}\n"
            user_message += (
                f"# Useful Reasoning Process: \n{reasoning_process}\n-----\n\n"
            )

            user_message += f"Using the reasoning process above, answer the question."

            formatted_prompts.append(
                self.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

        return formatted_prompts
