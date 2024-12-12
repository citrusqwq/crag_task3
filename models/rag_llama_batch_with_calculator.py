import os
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import ray
import torch
import vllm
import json
import trafilatura
import traceback
import torch.nn.functional as F
from loguru import logger
from blingfire import text_to_sentences_and_offsets
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from models.attr_predictor import AttrPredictor
from models.utils_html import get_tables, clean_html

# os.system("pip list")
# os.system("nvidia-smi")
logger.level("DEBUG")

######################################################################################################
######################################################################################################
###
### IMPORTANT !!!
### Before submitting, please follow the instructions in the docs below to download and check in :
### the model weighs.
###
###  https://gitlab.aicrowd.com/aicrowd/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/meta-comphrehensive-rag-benchmark-starter-kit/-/blob/master/docs/download_baseline_model_weights.md
###
### And please pay special attention to the comments that start with "TUNE THIS VARIABLE"
###                        as they depend on your model and the available GPU resources.
###
### DISCLAIMER: This baseline has NOT been tuned for performance
###             or efficiency, and is provided as is for demonstration.
######################################################################################################


# Load the environment variable that specifies the URL of the MockAPI. This URL is essential
# for accessing the correct API endpoint in Task 2 and Task 3. The value of this environment variable
# may vary across different evaluation settings, emphasizing the importance of dynamically obtaining
# the API URL to ensure accurate endpoint communication.

# Please refer to https://gitlab.aicrowd.com/aicrowd/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/crag-mock-api
# for more information on the MockAPI.
#
# **Note**: This environment variable will not be available for Task 1 evaluations.
CRAG_MOCK_API_URL = os.getenv("CRAG_MOCK_API_URL", "http://localhost:8000")


#### CONFIG PARAMETERS ---

# Define the number of context sentences to consider for generating an answer.
NUM_CONTEXT_SENTENCES = 16
# Set the maximum length for each context sentence (in characters).
MAX_CONTEXT_SENTENCE_LENGTH = 500

# Batch size you wish the evaluators will use to call the `batch_generate_answer` function
AICROWD_SUBMISSION_BATCH_SIZE = 4  # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.

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

        tables = get_tables(clean_html(html_source), html_name)

        return interaction_id, chunks, tables

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
        self.expression_sample_num = 20

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
                # logger.debug(f"Expression: {exp}, Result: {res}")
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
            # logger.debug(f"Generated expressions: {curr_generated_expressions}")
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


class BatchRAGModelWithCalculator:
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
        max_table_length: int = 6000,
        num_context_sentences=NUM_CONTEXT_SENTENCES,
        max_context_sentence_length=MAX_CONTEXT_SENTENCE_LENGTH,
        log_while_inference=False,
        custom_sentence_embedding=False,
        query_prompt=QUERY_PROMPT,
        include_table_in_text=True,
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

        answers = [None for _ in queries]
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

        calculation_results: list[list[str]] = self.calculator.get_calculation_results(
            self.llm, queries, batch_retrieval_results, batch_tables
        )

        # Prepare formatted prompts from the LLM
        formatted_prompts = self.format_prompts(
            queries,
            query_times,
            batch_retrieval_results,
            batch_tables,
            calculation_results,
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
                # Note: We are using 50 max new tokens instead of 75,
                # because the 75 max token limit for the competition is checked using the Llama2 tokenizer.
                # Llama3 instead uses a different tokenizer with a larger vocabulary
                # This allows the Llama3 tokenizer to represent the same content more efficiently,
                # while using fewer tokens.
            ),
            use_tqdm=False,  # you might consider setting this to True during local development
        )

        # Aggregate answers into List[str]
        # answers = []
        prompt_lens = []
        for idx, response in enumerate(responses):
            ret_answer = response.outputs[0].text.strip().rstrip()
            prompt_lens.append(len(response.prompt_token_ids))
            if ret_answer == "":
                ret_answer = "I don't know."
            if "i don't know" in ret_answer.lower():
                ret_answer = "I don't know."
            if "i do not know" in ret_answer.lower():
                ret_answer = "I don't know."
            if answers[idx] is None:
                answers[idx] = ret_answer

        if self.log_while_inference:
            logger.info("Batch queries:")
            logger.info(json.dumps(queries, indent=2))
            logger.info("Batch answers:")
            logger.info(json.dumps(answers, indent=2))
            logger.info(f"Prompt lengths: {prompt_lens}")
            logger.info("Predicted attributes:")
            logger.info(json.dumps(all_predicted_attrs, indent=2))

        if self.return_additional_info:
            return answers, {
                "all_predicted_attrs": all_predicted_attrs,
                "prompt_lens": prompt_lens,
                "prompts": formatted_prompts,
                "generated_expressions": calculation_results,
            }
        else:
            return answers

    def format_prompts(
        self,
        queries,
        query_times,
        batch_retrieval_results: list[list[str]] = [],
        batch_table_results: list[list[str]] = [],
        batch_calculation_results: list[list[str]] = [],
    ):
        """
        Formats queries, corresponding query_times and retrieval results using the chat_template of the model.

        Parameters:
        - queries (List[str]): A list of queries to be formatted into prompts.
        - query_times (List[str]): A list of query_time strings corresponding to each query.
        - batch_retrieval_results (List[str])
        """
        system_prompt = "You are provided with a question and various references. Your task is to answer the question succinctly, using the fewest words possible. There are also some calculation results from another agent, which may be useful for you. If the references do not contain the necessary information to answer the question and you can't answer it directly based on your knowledge, respond with 'I don't know'. There is no need to explain the reasoning behind your answers."
        formatted_prompts = []

        for _idx, query in enumerate(queries):
            query_time = query_times[_idx]
            retrieval_results = batch_retrieval_results[_idx]
            related_tables = batch_table_results[_idx]
            generated_expressions = batch_calculation_results[_idx]

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

            if len(generated_expressions) > 0:
                expression_references = ""
                user_message += "## Possible useful calculation results \n"
                for idx, expression in enumerate(generated_expressions):
                    expression_references += f"### Calculation {idx + 1}: \n"
                    expression_references += f"{expression}\n"
                user_message += f"{expression_references}\n------\n\n"

            user_message += "**Remember your rule**: If the references do not contain the necessary information to answer the question and you can't answer it directly based on your knowledge, respond with 'I don't know'. There is no need to explain the reasoning behind your answers. \n"
            user_message += (
                f"Using the references listed above, answer the following question: \n"
            )
            user_message += f"Current Time: {query_time}\n"
            user_message += f"Question: {query}\n"

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
