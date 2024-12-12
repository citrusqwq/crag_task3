#!/usr/bin/env python

import os
import time
import numpy as np

from transformers import LlamaTokenizerFast
from sentence_transformers import SentenceTransformer
from blingfire import text_to_sentences_and_offsets
from bs4 import BeautifulSoup
from loguru import logger

tokenizer_path = os.path.join(os.path.dirname(__file__), "..", "tokenizer")
tokenizer = LlamaTokenizerFast.from_pretrained(tokenizer_path)
ALL_DOMAINS = ["sports", "open", "music", "movie", "finance"]
EXAMPLE_TEMPLATE = """
### Question: {query}
### Domain: {domain}
### References: {references}
### Answer:"""
EXAMPLE_TEMPLATE_WO_DOMAIN = """
### Question: {query}
### References: {references}
### Answer:"""


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


def trim_predictions_to_max_token_length(prediction):
    """Trims prediction output to 75 tokens"""
    max_token_length = 75
    tokenized_prediction = tokenizer.encode(prediction)
    trimmed_tokenized_prediction = tokenized_prediction[1 : max_token_length + 1]
    trimmed_prediction = tokenizer.decode(trimmed_tokenized_prediction)
    return trimmed_prediction


def process_search_res(
    sentence_model: SentenceTransformer,
    curr_query: str,
    curr_search_results: list[str],
    num_context: int = 8,
    max_ctx_sentence_length: int = 200,
    remove_ques: bool = False,
    cut_long_sents: bool = True,
    remove_duplicates: bool = False,
    extend_sents: bool = False,
    extend_sents_len: int = 300,
) -> tuple[list[str], list[str], float, float, float]:
    start_time = time.perf_counter()
    all_sentences = []
    all_sentences_ctx = []
    for html_text in curr_search_results:
        # Parse the HTML content to extract text.
        soup = BeautifulSoup(html_text["page_result"], features="html.parser")
        text = soup.get_text().replace("\n", " ")
        text = text.replace("\r", " ")
        text = " ".join(text.split())
        if len(text) > 0:
            offsets = text_to_sentences_and_offsets(text)[1]
            for ofs in offsets:
                if extend_sents:
                    start_idx = max(0, ofs[0] - extend_sents_len)
                    end_idx = min(len(text), ofs[1] + extend_sents_len)
                else:
                    start_idx = ofs[0]
                    end_idx = ofs[1]
                sentence = text[ofs[0] : ofs[1]]
                if remove_ques and is_question(sentence):
                    continue
                if cut_long_sents:
                    curr_sents = []
                    last_ofs_0 = ofs[0]
                    while len(sentence) > 0:
                        curr_sents.append(sentence[:max_ctx_sentence_length])
                        sentence = sentence[max_ctx_sentence_length:]
                        all_sentences_ctx.append(
                            text[
                                max(start_idx, last_ofs_0 - extend_sents_len) : min(
                                    last_ofs_0
                                    + max_ctx_sentence_length
                                    + extend_sents_len,
                                    end_idx,
                                )
                            ]
                        )
                        last_ofs_0 += max_ctx_sentence_length
                    all_sentences.extend(curr_sents)
                else:
                    all_sentences.append(sentence[:max_ctx_sentence_length])
                    end_idx = max(ofs[0] + max_ctx_sentence_length, end_idx)
                    all_sentences_ctx.append(text[start_idx:end_idx])
        else:
            all_sentences.append("")
            if extend_sents:
                all_sentences_ctx.append("")
    if remove_duplicates:
        all_sentences = list(set(all_sentences))
    time_to_parse = time.perf_counter() - start_time
    all_embeddings = sentence_model.encode(all_sentences, normalize_embeddings=True)
    query_embedding = sentence_model.encode(curr_query, normalize_embeddings=True)[
        None, :
    ]
    cosine_scores = (all_embeddings * query_embedding).sum(1)
    if extend_sents:
        top_sentences = np.array(all_sentences_ctx)[
            (-cosine_scores).argsort()[:num_context]
        ]
        logger.debug("#" * 150)
        random_chosen_idx = np.random.randint(0, len(all_sentences) - 1)
        logger.debug(all_sentences[random_chosen_idx])
        logger.debug(all_sentences_ctx[random_chosen_idx])
        assert len(all_sentences) == len(all_sentences_ctx), (
            len(all_sentences),
            len(all_sentences_ctx),
        )
    else:
        top_sentences = np.array(all_sentences)[
            (-cosine_scores).argsort()[:num_context]
        ]
    top_sentences = top_sentences.tolist()
    time_to_calculate = time.perf_counter() - start_time - time_to_parse
    return (
        top_sentences,
        all_sentences,
        time.perf_counter() - start_time,
        time_to_parse,
        time_to_calculate,
    )


def process_search_res_v_1_3(
    sentence_model: SentenceTransformer,
    curr_query: str,
    curr_search_results: list[str],
) -> tuple[list[str], list[str], float, float, float]:
    start_time = time.perf_counter()
    all_sentences = []
    num_context = 4
    max_ctx_sentence_length = 200
    for html_text in curr_search_results:
        # Parse the HTML content to extract text.
        soup = BeautifulSoup(html_text["page_result"], features="html.parser")
        text = soup.get_text().replace("\n", " ")
        text = text.replace("\r", " ")
        text = " ".join(text.split())
        if len(text) > 0:
            offsets = text_to_sentences_and_offsets(text)[1]
            for ofs in offsets:
                sentence = text[ofs[0] : ofs[1]]
                if is_question(sentence):
                    continue
                all_sentences.append(sentence[:max_ctx_sentence_length])
        else:
            all_sentences.append("")
    all_sentences = list(set(all_sentences))
    time_to_parse = time.perf_counter() - start_time
    all_embeddings = sentence_model.encode(all_sentences, normalize_embeddings=True)
    query_embedding = sentence_model.encode(curr_query, normalize_embeddings=True)[
        None, :
    ]
    cosine_scores = (all_embeddings * query_embedding).sum(1)
    top_sentences = np.array(all_sentences)[(-cosine_scores).argsort()[:num_context]]
    top_sentences = top_sentences.tolist()
    time_to_calculate = time.perf_counter() - start_time - time_to_parse
    return (
        top_sentences,
        all_sentences,
        time.perf_counter() - start_time,
        time_to_parse,
        time_to_calculate,
    )
