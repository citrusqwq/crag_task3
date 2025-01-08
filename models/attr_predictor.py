import os
import json
import argparse
import pickle
import numpy as np
import torch
import time

# import vllm
import math
import random
import Levenshtein
from loguru import logger
from tqdm import tqdm
from typing import Dict, List

# from sentence_transformers import SentenceTransformer
# from transformers import (
#    AutoModelForCausalLM,
#    AutoTokenizer,
#    BitsAndBytesConfig,
#    pipeline,
#    TextGenerationPipeline,
#    PreTrainedTokenizer,
#    PreTrainedTokenizerFast,
# )
from loguru import logger
from openai import OpenAI


# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

"""
def format_prompt(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    system_prompt: str,
    user_prompt_template: str,
    queries: list[str],
) -> list[str]:
    formatted_prompts = []
    for _idx, query in enumerate(queries):
        formatted_prompts.append(
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": user_prompt_template.format(query=query),
                    },
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return formatted_prompts
"""


def format_prompt(
    system_prompt: str, user_prompt_template: str, queries: list[str]
) -> list[list[dict]]:
    formatted_prompts = []
    for query in queries:
        formatted_prompts.append(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt_template.format(query=query)},
            ]
        )
    return formatted_prompts


def predict_domain(llm: OpenAI, batch, sample_num: int = 5):
    queries = batch["query"]
    system_prompt = """You are provided with a question. Your task is to answer the domain of this question. You **MUST** choose from the following domains: ["finance", "music", "movie", "sports", "open"]. You **MUST** give the domain succinctly, using the fewest words possible."""
    user_prompt = """Here is the question: {query}\nWhat is the domain of this question? Remember your rule: You **MUST** choose from the following domains: ["finance", "music", "movie", "sports", "open"]."""
    formatted_prompts = format_prompt(system_prompt, user_prompt, queries)
    """
    responses = llm.generate(
        formatted_prompts,
        vllm.SamplingParams(
            n=sample_num,  # Number of output sequences to return for each prompt.
            top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
            temperature=0.1,  # Randomness of the sampling
            skip_special_tokens=True,  # Whether to skip special tokens in the output.
            max_tokens=10,  # Maximum number of tokens to generate per output sequence.
        ),
        use_tqdm=False,
    )
    """
    sample_num = 1
    responses = []
    for prompt in formatted_prompts:
        response = llm.chat.completions.create(
            model="llama3.3:70b",
            messages=prompt,
            n=sample_num,  # Generate multiple samples
            temperature=0.1,  # Low randomness
            top_p=0.9,  # Control cumulative probability
            max_tokens=10,  # Limit output length
        )
        responses.append(response)

    answers = []
    for response in responses:
        # curr_samples = [
        #    #response.outputs[idx].text.strip().rstrip() for idx in range(sample_num)
        #    response["choices"][idx]["message"]["content"].strip() for idx in range(sample_num)
        # ]
        curr_samples = [
            response.choices[idx].message.content.strip() for idx in range(sample_num)
        ]
        curr_cnt = {}
        for sample in curr_samples:
            curr_cnt.setdefault(sample, 0)
            curr_cnt[sample] += 1
        curr_ans = max(curr_cnt, key=curr_cnt.get)
        curr_final_ans = None
        all_keys = ["finance", "music", "movie", "sports", "open"]
        for curr_key in all_keys:
            if curr_key in curr_ans:
                curr_final_ans = curr_key
                break
        if curr_final_ans is None:
            # choose the close distance
            min_dist = math.inf
            for curr_key in all_keys:
                curr_dist = Levenshtein.distance(curr_key, curr_ans)
                if curr_dist < min_dist:
                    min_dist = curr_dist
                    curr_final_ans = curr_key
        answers.append(curr_final_ans)
    return answers


def predict_question_type(
    llm: OpenAI, batch, sample_num: int = 5, few_shots: list = None
):
    queries = batch["query"]
    system_prompt = """You will be provided with a question. Your task is to identify whether this question is a simple question or a complex question. A simple question is that you can answer directly or just need a little additional outside information. A complex question is that needs complex reasoning and analyzing from a lot of outside information. You **MUST** choose from one of the following choices: ["simple", "complex"]. You **MUST** give the question type succinctly, using the fewest words possible."""
    if few_shots is not None:
        system_prompt += "\nHere are some examples:\n"
        for example in few_shots:
            system_prompt += (
                "------\n### Question: {}\n### Question Type: {}\n\n".format(
                    example["query"], example["question_type"]
                )
            )
    user_prompt = """Here is the question: {query}\nRemember your rule: You **MUST** choose from the following choices: ["simple", "complex"].\nWhat is the question type of this question?"""
    formatted_prompts = format_prompt(system_prompt, user_prompt, queries)
    """
    responses = llm.generate(
        formatted_prompts,
        vllm.SamplingParams(
            n=sample_num,  # Number of output sequences to return for each prompt.
            top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
            temperature=0.1,  # Randomness of the sampling
            skip_special_tokens=True,  # Whether to skip special tokens in the output.
            max_tokens=10,  # Maximum number of tokens to generate per output sequence.
        ),
        use_tqdm=False,
    )
    """
    sample_num = 1
    responses = []
    for prompt in formatted_prompts:
        response = llm.chat.completions.create(
            model="llama3.3:70b",
            messages=prompt,
            n=sample_num,  # Generate multiple samples
            temperature=0.1,  # Low randomness
            top_p=0.9,  # Control cumulative probability
            max_tokens=10,  # Limit output length
        )
        responses.append(response)

    answers = []
    for response in responses:
        # curr_samples = [
        #    #response.outputs[idx].text.strip().rstrip() for idx in range(sample_num)
        #    response["choices"][idx]["message"]["content"].strip() for idx in range(sample_num)
        # ]
        curr_samples = [
            response.choices[idx].message.content.strip() for idx in range(sample_num)
        ]
        curr_cnt = {}
        for sample in curr_samples:
            curr_cnt.setdefault(sample, 0)
            curr_cnt[sample] += 1
        curr_ans = max(curr_cnt, key=curr_cnt.get)
        curr_final_ans = None
        all_keys = ["simple", "complex"]
        for curr_key in all_keys:
            if curr_key in curr_ans:
                curr_final_ans = curr_key
                break
        if curr_final_ans is None:
            # choose the close distance
            min_dist = math.inf
            for curr_key in all_keys:
                curr_dist = Levenshtein.distance(curr_key, curr_ans)
                if curr_dist < min_dist:
                    min_dist = curr_dist
                    curr_final_ans = curr_key
        answers.append(curr_final_ans)
    return answers


def predict_static_or_dynamic(
    llm: OpenAI, batch, sample_num: int = 5, few_shots: list = None
):
    queries = batch["query"]
    system_prompt = """You will be provided with a question. Your task is to identify whether this question is a static question or a dynamic question. A static question is that the answer is fixed and will not change over time. A dynamic question is that the answer will change over time or needs time information. You **MUST** choose from one of the following choices: ["static", "dynamic"]. You **MUST** give the question type succinctly, using the fewest words possible."""
    if few_shots is not None:
        system_prompt += "\nHere are some examples:\n"
        for example in few_shots:
            system_prompt += (
                "------\n### Question: {}\n### Static or Dynamic: {}\n\n".format(
                    example["query"], example["static_or_dynamic"]
                )
            )
    user_prompt = """Here is the question: {query}\nRemember your rule: You **MUST** choose from the following choices: ["static", "dynamic"].\nWhat is the static or dynamic of this question?"""
    formatted_prompts = format_prompt(system_prompt, user_prompt, queries)
    """
    responses = llm.generate(
        formatted_prompts,
        vllm.SamplingParams(
            n=sample_num,  # Number of output sequences to return for each prompt.
            top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
            temperature=0.1,  # Randomness of the sampling
            skip_special_tokens=True,  # Whether to skip special tokens in the output.
            max_tokens=10,  # Maximum number of tokens to generate per output sequence.
        ),
        use_tqdm=False,
    )
    """
    sample_num = 1
    responses = []
    for prompt in formatted_prompts:
        print(prompt)
        response = llm.chat.completions.create(
            model="llama3.3:70b",
            messages=prompt,
            n=sample_num,  # Generate multiple samples
            temperature=0.1,  # Low randomness
            top_p=0.9,  # Control cumulative probability
            max_tokens=10,  # Limit output length
        )
        responses.append(response)
        print(response)
    answers = []
    for response in responses:
        # curr_samples = [
        #    #response.outputs[idx].text.strip().rstrip() for idx in range(sample_num)
        #    response["choices"][idx]["message"]["content"].strip() for idx in range(sample_num)
        # ]
        curr_samples = [
            response.choices[idx].message.content.strip() for idx in range(sample_num)
        ]
        curr_final_ans = "static"
        for sample in curr_samples:
            if "dynamic" in sample:
                curr_final_ans = "dynamic"
                break
        answers.append(curr_final_ans)
    return answers


def get_few_shots(dataset_items: list, num_shots: int, key: str):
    few_shots = []
    few_shots_file_path = f"models/few-shots/{key}_few_shots.jsonl"
    if os.path.exists(few_shots_file_path):
        logger.info(f"Found saved few shots in {few_shots_file_path}")
        with open(few_shots_file_path, "r") as f:
            for line in f:
                few_shots.append(json.loads(line))
        return few_shots
    logger.info(f"No such file: {few_shots_file_path}, generating few shots...")
    all_labels_items = {}
    for item in dataset_items:
        all_labels_items.setdefault(item[key], [])
        all_labels_items[item[key]].append(item)
    for label in all_labels_items:
        few_shots.extend(random.sample(all_labels_items[label], num_shots))
    random.shuffle(few_shots)
    return few_shots


class AttrPredictor:
    def __init__(self, method: str, vllm_model: OpenAI = None):
        assert method in ["svm", "few-shot"]
        if method == "svm":
            raise NotImplementedError
        self.method = method
        self.vllm_model = vllm_model
        self.valid_dataset_items = []
        self.public_test_dataset_items = []
        with open("large-files/dataset_v3_no_search_results.jsonl", "r") as f:
            for line in tqdm(f, desc="Loading dataset", ncols=100):
                curr_data = json.loads(line)
                assert curr_data["question_type"] in [
                    "simple",
                    "simple_w_condition",
                    "comparison",
                    "aggregation",
                    "set",
                    "false_premise",
                    "post-processing",
                    "multi-hop",
                ]
                if curr_data["question_type"] in ["simple", "simple_w_condition"]:
                    curr_data["question_type"] = "simple"
                else:
                    curr_data["question_type"] = "complex"

                assert curr_data["static_or_dynamic"] in [
                    "static",
                    "slow-changing",
                    "fast-changing",
                    "real-time",
                ]
                if curr_data["static_or_dynamic"] in ["static", "slow-changing"]:
                    curr_data["static_or_dynamic"] = "static"
                else:
                    curr_data["static_or_dynamic"] = "dynamic"

                if curr_data["split"] == 1:
                    self.public_test_dataset_items.append(curr_data)
                else:
                    self.valid_dataset_items.append(curr_data)

        self.question_type_few_shots = get_few_shots(
            self.valid_dataset_items, 5, "question_type"
        )
        self.static_dynamic_few_shots = get_few_shots(
            self.valid_dataset_items, 10, "static_or_dynamic"
        )
        logger.info(
            "Static or dynamic few shots: {}".format(
                json.dumps(self.static_dynamic_few_shots, indent=2)
            )
        )

    def predict_attr(self, batch: list, attr_name: str) -> list[str]:
        assert attr_name in ["domain", "question_type", "static_or_dynamic"]
        if attr_name == "domain":
            return predict_domain(self.vllm_model, batch)
        elif attr_name == "question_type":
            return predict_question_type(
                self.vllm_model, batch, few_shots=self.question_type_few_shots
            )
        elif attr_name == "static_or_dynamic":
            return predict_static_or_dynamic(
                self.vllm_model, batch, few_shots=self.static_dynamic_few_shots
            )
        else:
            raise NotImplementedError
