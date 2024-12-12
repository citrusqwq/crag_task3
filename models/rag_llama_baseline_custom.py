import os
from typing import Dict, List

import numpy as np
import torch
import json
from loguru import logger
from blingfire import text_to_sentences_and_offsets
from bs4 import BeautifulSoup
from models.utils import (
    trim_predictions_to_max_token_length,
    process_search_res,
    process_search_res_v_1_3,
    EXAMPLE_TEMPLATE,
    EXAMPLE_TEMPLATE_WO_DOMAIN,
    ALL_DOMAINS,
)
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

CRAG_MOCK_API_URL = os.getenv("CRAG_MOCK_API_URL", "http://localhost:8000")


class CustomRAGModel:
    def __init__(self, model="llama3-8b-base", debug=False):
        """
        Initialize the RAGModel with necessary models and configurations.

        This constructor sets up the environment by loading sentence transformers for embedding generation,
        a large language model for generating responses, and tokenizer for text processing. It also initializes
        model parameters and templates for generating answers.
        input:
            model: in ["llama3-8b-base", "llama3-8b-chat", "llama3-8b-groq"]
        """
        # Load a sentence transformer model optimized for sentence embeddings, using CUDA if available.
        self.sentence_model = SentenceTransformer(
            "large-files/ckpts/sentence-transformers/all-MiniLM-L6-v2", device="cuda"
        )

        # Configuration for model quantization to improve performance, using 4-bit precision.
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )

        ########## LLM setting ##########
        assert model in [
            "llama3-8b-base",
            "llama3-8b-chat",
            "llama3-8b-groq",
            "llama3-70b-chat",
        ]
        if model == "llama3-8b-base":
            # Specify the large language model to be used.
            model_name = "large-files/ckpts/Meta-Llama-3-8B"

            # Load the tokenizer for the specified model.
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            # Load the large language model with the specified quantization configuration.
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                quantization_config=bnb_config,
            )

            # Initialize a text generation pipeline with the loaded model and tokenizer.
            self.generation_pipe = pipeline(
                task="text-generation",
                model=self.llm,
                tokenizer=self.tokenizer,
                return_full_text=False,
                pad_token_id=self.llm.config.eos_token_id,
                max_new_tokens=32,
                do_sample=False,
            )
            self.prompt_template = """
### Question
{query}

### References 
{references}

### Answer
"""
        elif model in ["llama3-8b-chat", "llama3-70b-chat"]:
            model_name = (
                "large-files/ckpts/Meta-Llama-3-8B-Instruct"
                if model == "llama3-8b-chat"
                else "/home1/rag-challenge/rag-challenge/models/Meta-Llama-3-70B-Instruct"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                quantization_config=bnb_config,
            )
            terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            ]
            self.chat_pipeline = pipeline(
                task="conversational",
                model=self.llm,
                tokenizer=self.tokenizer,
                pad_token_id=self.llm.config.eos_token_id,
                do_sample=False,
                eos_token_id=terminators,
            )

            def generation_pipe(prompt):
                conversation = self.chat_pipeline(
                    [
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    max_new_tokens=32,
                )
                return [{"generated_text": conversation.generated_responses[0]}]

            self.generation_pipe = generation_pipe
            self.prompt_template = """You are given a quesition and references which may or may not help answer the question. Your goal is to answer the question in as few words as possible.
### Question
{query}

### References 
{references}

### Answer
"""
        elif model == "llama3-8b-groq":
            from groq import Groq
            from api_key import groq_api_key
            import httpx

            proxy_url = "http://127.0.0.1:7890"
            self.client = Groq(
                api_key=groq_api_key,
                http_client=httpx.Client(proxy=proxy_url),
            )

            def generation_pipe(prompt):
                return_value = ""
                while return_value == "":
                    try:
                        chat_completion = self.client.chat.completions.create(
                            messages=[
                                {
                                    "role": "user",
                                    "content": prompt,
                                }
                            ],
                            model="llama3-8b-8192",
                            temperature=0.0,
                        )
                        return_value = chat_completion.choices[0].message.content
                        break
                    except:
                        continue
                return [{"generated_text": return_value}]

            self.generation_pipe = generation_pipe
            self.prompt_template = """You are given a quesition and references which may or may not help answer the question. Your goal is to answer the question in as few words as possible.
### Question
{query}

### References 
{references}

### Answer
"""

        self.debug = debug
        with open("models/few-shots/domain_classification_15_shot.txt", "r") as f:
            self.domain_classification_few_shot = f.read()
        self.few_shots = {}
        for domain in ALL_DOMAINS:
            with open(f"models/few-shots/{domain}_few_shot.txt", "r") as f:
                self.few_shots[domain] = f.read()
        with open("models/few-shots/common_few_shot.txt", "r") as f:
            self.common_few_shots = f.read()
        with open("models/few-shots/common_few_shot_v1.3.txt", "r") as f:
            self.common_few_shots_v1_3 = f.read()
        # self.generate_answer = self.generate_answer_with_domain_pred
        os.system("nvidia-smi")

    def predict_domain(self, query) -> str:
        prompt = (
            self.domain_classification_few_shot
            + f"\n### Question: {query}\n### Domain: "
        )
        output = self.generation_pipe(
            prompt,
            max_new_tokens=10,
        )
        output = output[0]["generated_text"]
        pred_domain = output.split("\n")[0].strip().rstrip()
        if pred_domain not in ALL_DOMAINS:
            pred_domain = "open"
        return pred_domain

    def debug_print(self, *args, **kwargs):
        if self.debug:
            print(*args, **kwargs)

    def generate_answer(
        self, query: str, search_results: List[Dict], query_time: str
    ) -> str:
        return self.generate_answer_ablation_search_res_process(
            # return self.generate_answer_add_sentence_ctx(
            # return self.generate_answer_add_sentence_ctx_trafila(
            query,
            search_results,
            query_time,
        )

    def generate_answer_with_domain_pred(
        self, query: str, search_results: List[Dict], query_time: str
    ) -> str:
        """
        Generate an answer based on the provided query and a list of pre-cached search results.

        Parameters:
        - query (str): The user's question.
        - search_results (List[Dict]): A list containing the search result objects,
        as described here:
          https://gitlab.aicrowd.com/aicrowd/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/meta-comphrehensive-rag-benchmark-starter-kit/-/blob/master/docs/dataset.md#search-results-detail
        - query_time (str): The time at which the query was made, represented as a string.

        Returns:
        - str: A text response that answers the query. Limited to 75 tokens.

        This method processes the search results to extract relevant sentences, generates embeddings for them,
        and selects the top context sentences based on cosine similarity to the query embedding. It then formats
        this information into a prompt for the language model, which generates an answer that is then trimmed to
        meet the token limit.
        """
        top_sentences, *_ = process_search_res(
            self.sentence_model, query, search_results
        )
        references = "\n".join([f"<DOC>{s}</DOC>" for s in top_sentences])
        references = "\n" + references
        predicted_domain = self.predict_domain(query)
        new_example_question = EXAMPLE_TEMPLATE.format(
            query=query,
            domain=predicted_domain,
            references=references,
        )
        final_prompt = f"{self.few_shots[predicted_domain]}{new_example_question}"

        # Generate an answer using the formatted prompt.
        result = self.generation_pipe(final_prompt)
        result: str = result[0]["generated_text"]

        answer = result.split("\n")[0].strip().rstrip()
        print("Answer: ", answer)
        if len(answer) == 0:
            answer = "I don't know"
        # print("raw answer: ", answer)
        try:
            answer = json.loads(answer)
            if not (answer["useful"] and answer["confidence"]):
                answer = "I don't know"
            answer = answer["generation"]
        except:
            answer = "I don't know"

        # Trim the prediction to a maximum of 75 tokens to meet the submission requirements.
        trimmed_answer = trim_predictions_to_max_token_length(answer)

        return trimmed_answer, top_sentences

    def generate_answer_without_domain_pred_v_1_3(
        self, query: str, search_results: List[Dict], query_time: str
    ) -> str:
        top_sentences, *_ = process_search_res_v_1_3(
            self.sentence_model, query, search_results
        )
        references = "\n".join([f"<DOC>{s}</DOC>" for s in top_sentences])
        references = "\n" + references
        new_example_question = EXAMPLE_TEMPLATE_WO_DOMAIN.format(
            query=query,
            references=references,
        )
        final_prompt = f"{self.common_few_shots_v1_3}{new_example_question}"

        # Generate an answer using the formatted prompt.
        result = self.generation_pipe(final_prompt)
        result: str = result[0]["generated_text"]

        answer = result.split("\n")[0].strip().rstrip()
        if len(answer) == 0:
            answer = "I don't know"
        try:
            answer = json.loads(answer)
            if not (answer["useful"] and answer["confidence"]):
                answer = "I don't know"
            answer = answer["generation"]
        except:
            answer = "I don't know"

        # Trim the prediction to a maximum of 75 tokens to meet the submission requirements.
        trimmed_answer = trim_predictions_to_max_token_length(answer)

        return trimmed_answer, top_sentences

    def generate_answer_without_domain_pred(
        self, query: str, search_results: List[Dict], query_time: str
    ) -> str:
        """
        Generate an answer based on the provided query and a list of pre-cached search results.

        Parameters:
        - query (str): The user's question.
        - search_results (List[Dict]): A list containing the search result objects,
        as described here:
          https://gitlab.aicrowd.com/aicrowd/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/meta-comphrehensive-rag-benchmark-starter-kit/-/blob/master/docs/dataset.md#search-results-detail
        - query_time (str): The time at which the query was made, represented as a string.

        Returns:
        - str: A text response that answers the query. Limited to 75 tokens.

        This method processes the search results to extract relevant sentences, generates embeddings for them,
        and selects the top context sentences based on cosine similarity to the query embedding. It then formats
        this information into a prompt for the language model, which generates an answer that is then trimmed to
        meet the token limit.
        """
        top_sentences, *_ = process_search_res(
            self.sentence_model, query, search_results
        )
        references = "\n".join([f"<DOC>{s}</DOC>" for s in top_sentences])
        references = "\n" + references
        # print(json.dumps({
        #     "query": query,
        #     "references": top_sentences,
        # }, indent=2))
        new_example_question = EXAMPLE_TEMPLATE_WO_DOMAIN.format(
            query=query,
            references=references,
        )
        final_prompt = f"{self.common_few_shots}{new_example_question}"

        # Generate an answer using the formatted prompt.
        result = self.generation_pipe(final_prompt)
        result: str = result[0]["generated_text"]

        answer = result.split("\n")[0].strip().rstrip()
        if len(answer) == 0:
            answer = "I don't know"
        # print("raw answer: ", answer)
        try:
            answer = json.loads(answer)
            if not (answer["useful"] and answer["confidence"]):
                answer = "I don't know"
            answer = answer["generation"]
        except:
            answer = "I don't know"

        # Trim the prediction to a maximum of 75 tokens to meet the submission requirements.
        trimmed_answer = trim_predictions_to_max_token_length(answer)

        return trimmed_answer, top_sentences

    # TODO: 上面三个generate函数没有适配chat的prompt模板
    def generate_answer_ablation_search_res_process(
        self, query: str, search_results: List[Dict], query_time: str
    ) -> str:
        top_sentences, *_ = process_search_res(
            self.sentence_model, query, search_results, remove_ques=True
        )
        references = ""
        for snippet in top_sentences:
            references += "<DOC>\n" + snippet + "\n</DOC>\n"

        final_prompt = self.prompt_template.format(query=query, references=references)
        result = self.generation_pipe(final_prompt)
        result: str = result[0]["generated_text"]
        answer = result.split("\n")[0].strip().rstrip()
        trimmed_answer = trim_predictions_to_max_token_length(answer)
        return trimmed_answer, top_sentences

    def generate_answer_add_sentence_ctx(
        self, query: str, search_results: List[Dict], query_time: str
    ) -> str:
        top_sentences, *_ = process_search_res(
            # logger.info("正在使用trafilatura")
            # top_sentences, *_ = process_search_res_lcw_trafilatura(
            self.sentence_model,
            query,
            search_results,
            extend_sents=True,
        )
        references = ""
        for snippet in top_sentences:
            references += "<DOC>\n" + snippet + "\n</DOC>\n"
        final_prompt = self.prompt_template.format(query=query, references=references)
        result = self.generation_pipe(final_prompt)
        result: str = result[0]["generated_text"]
        answer = result.split("\n")[0].strip().rstrip()
        trimmed_answer = trim_predictions_to_max_token_length(answer)
        return trimmed_answer, top_sentences
