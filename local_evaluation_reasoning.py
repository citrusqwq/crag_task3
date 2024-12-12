import bz2
import json
import os
import math
import argparse
import random
import torch
import numpy as np
import traceback
from datetime import datetime

from loguru import logger
from openai import APIConnectionError, OpenAI, RateLimitError
from prompts.templates import IN_CONTEXT_EXAMPLES, INSTRUCTIONS
from tqdm.auto import tqdm
from transformers import LlamaTokenizerFast
from api_key import deepseek_api_key, openai_api_key

tokenizer = LlamaTokenizerFast.from_pretrained("tokenizer")
LOG_DIR = None


def load_json_file(file_path):
    """Load and return the content of a JSON file."""
    logger.info(f"Loading JSON from {file_path}")
    with open(file_path) as f:
        return json.load(f)


def get_system_message():
    """Returns the system message containing instructions and in context examples."""
    return INSTRUCTIONS + IN_CONTEXT_EXAMPLES


def attempt_api_call(client: OpenAI, model_name: str, messages, max_retries=10):
    """Attempt an API call with retries upon encountering specific errors."""
    # todo: add default response when all efforts fail
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                response_format={"type": "json_object"},
                seed=args.seed,
                temperature=0.0,
                max_tokens=15,
            )
            return response.choices[0].message.content
        except (APIConnectionError, RateLimitError) as e:
            traceback.print_exc()
            logger.warning(
                f"API call failed with {e} on attempt {attempt + 1}, retrying..."
            )
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            break
    return None


def log_response(messages, response, output_directory="api_responses"):
    """Save the response from the API to a file."""
    os.makedirs(output_directory, exist_ok=True)
    file_name = datetime.now().strftime("%d-%m-%Y-%H-%M-%S.json")
    file_path = os.path.join(output_directory, file_name)
    with open(file_path, "w") as f:
        json.dump({"messages": messages, "response": response}, f)


def parse_response(resp: str):
    """Pass auto-eval output from the evaluator."""
    try:
        resp = resp.lower()
        model_resp = json.loads(resp)
        answer = -1
        if "accuracy" in model_resp and (
            (model_resp["accuracy"] is True)
            or (
                isinstance(model_resp["accuracy"], str)
                and model_resp["accuracy"].lower() == "true"
            )
        ):
            answer = 1
        else:
            raise ValueError(f"Could not parse answer from response: {model_resp}")

        return answer
    except:
        resp = resp.lower()
        resp = resp.split("\n")[0].strip().rstrip().split()[-1]
        if resp.lower() in ["yes", "true", "correct", "1"]:
            return 1
        return -1


def trim_predictions_to_max_token_length(prediction):
    """Trims prediction output to 75 tokens"""
    max_token_length = 75
    tokenized_prediction = tokenizer.encode(prediction)
    trimmed_tokenized_prediction = tokenized_prediction[1 : max_token_length + 1]
    trimmed_prediction = tokenizer.decode(trimmed_tokenized_prediction)
    return trimmed_prediction


def load_data_in_batches(dataset_items, batch_size):
    """
    Generator function that reads data from a compressed file and yields batches of data.
    Each batch is a dictionary containing lists of interaction_ids, queries, search results, query times, and answers.

    Args:
    dataset_path (str): Path to the dataset file.
    batch_size (int): Number of data items in each batch.

    Yields:
    dict: A batch of data.
    """

    def initialize_batch():
        """Helper function to create an empty batch."""
        return {
            "interaction_id": [],
            "query": [],
            "search_results": [],
            "query_time": [],
            "answer": [],
            "domain": [],
            "question_type": [],
            "static_or_dynamic": [],
        }

    batch = initialize_batch()
    for data_item in dataset_items:
        for key in batch:
            batch[key].append(data_item[key])
        if len(batch["query"]) == batch_size:
            yield batch
            batch = initialize_batch()
    if batch["query"]:
        yield batch


def generate_predictions(dataset_path, participant_model):
    """
    Processes batches of data from a dataset to generate predictions using a model.

    Args:
    dataset_path (str): Path to the dataset.
    participant_model (object): UserModel that provides `get_batch_size()` and `batch_generate_answer()` interfaces.

    Returns:
    tuple: A tuple containing lists of queries, ground truths, and predictions.
    """
    queries, ground_truths, predictions = [], [], []
    domains, question_types, static_or_dynamics = [], [], []
    all_reasonings = []
    batch_size = participant_model.get_batch_size()

    dataset_items = []
    if not args.use_public_test:
        with bz2.open(dataset_path, "rt") as bz2_file:
            for line in tqdm(bz2_file, desc="Loading dataset"):
                dataset_items.append(json.loads(line))
    else:
        with open(dataset_path, "r") as f:
            for line_no, line in enumerate(f, 1):
                pass
        with open(dataset_path, "r") as f:
            for line in tqdm(f, desc="Loading dataset", total=line_no):
                curr_data = json.loads(line)
                if curr_data["split"] == 1 and random.random() < args.test_ratio:
                    dataset_items.append(curr_data)
    batch_num = math.ceil(len(dataset_items) / batch_size)
    logger.info(f"Number of data items: {len(dataset_items)}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Number of batches: {batch_num}")
    batch_data_generator = load_data_in_batches(dataset_items, batch_size)
    for batch in tqdm(
        batch_data_generator, desc="Generating predictions", total=batch_num
    ):
        batch_ground_truths = batch.pop(
            "answer"
        )  # Remove answers from batch and store them
        batch_predictions, batch_reasoning = participant_model.batch_generate_answer(batch)
        queries.extend(batch["query"])
        domains.extend(batch["domain"])
        question_types.extend(batch["question_type"])
        static_or_dynamics.extend(batch["static_or_dynamic"])
        ground_truths.extend(batch_ground_truths)
        predictions.extend(batch_predictions)
        all_reasonings.extend(batch_reasoning)
    final_predictions = []
    for query, ground_truth, prediction, domain, ques_type, static_or_dynamic, reasoning in zip(
        queries, ground_truths, predictions, domains, question_types, static_or_dynamics, all_reasonings
    ):
        # trim prediction to 75 tokens
        prediction = trim_predictions_to_max_token_length(prediction)
        final_predictions.append(
            {
                "query": query,
                "ground_truth": str(ground_truth).strip().lower(),
                "prediction": str(prediction).strip().lower(),
                "domain": domain,
                "question_type": ques_type,
                "static_or_dynamic": static_or_dynamic,
                "reasoning_output": reasoning,
            }
        )
    return final_predictions


def old_generate_predictions(dataset_path, participant_model, args):
    predictions = []
    if not args.use_public_test:
        with bz2.open(dataset_path, "rt") as bz2_file:
            for line in tqdm(bz2_file, desc="Generating Predictions"):
                data = json.loads(line)
                query = data["query"]
                web_search_results = data["search_results"]
                query_time = data["query_time"]
                prediction, top_sentences = participant_model.generate_answer(
                    query, web_search_results, query_time
                )
                # trim prediction to 75 tokens
                prediction = trim_predictions_to_max_token_length(prediction)
                predictions.append(
                    {
                        "query": query,
                        "top_sentences": top_sentences,
                        "ground_truth": str(data["answer"]).strip().lower(),
                        "prediction": str(prediction).strip().lower(),
                        "domain": data["domain"],
                        "question_type": data["question_type"],
                        "static_or_dynamic": data["static_or_dynamic"],
                    }
                )
    else:
        eval_set = []
        with open(dataset_path, "r") as f:
            for line in tqdm(f, desc="Generating evaluation set"):
                line = line.strip()
                curr_data = json.loads(line)
                if curr_data["split"] == 1 and random.random() < args.test_ratio:
                    eval_set.append(curr_data)
        for data_item in tqdm(eval_set, "Generating Predictions"):
            query = data_item["query"]
            web_search_results = data_item["search_results"]
            query_time = data_item["query_time"]
            prediction, top_sentences = participant_model.generate_answer(
                query, web_search_results, query_time
            )
            # trim prediction to 75 tokens
            prediction = trim_predictions_to_max_token_length(prediction)
            predictions.append(
                {
                    "query": query,
                    "top_sentences": top_sentences,
                    "ground_truth": str(data_item["answer"]).strip().lower(),
                    "prediction": str(prediction).strip().lower(),
                    "domain": data_item["domain"],
                    "question_type": data_item["question_type"],
                    "static_or_dynamic": data_item["static_or_dynamic"],
                }
            )

    return predictions


def evaluate_predictions(predictions, evaluation_model_name, openai_client: OpenAI):
    n_miss, n_correct, n_correct_exact = 0, 0, 0
    correct_by_domain = {}
    correct_by_ques_type = {}
    correct_by_static_or_dynamic = {}
    system_message = get_system_message()
    logger.info(json.dumps(predictions, indent=2))
    if LOG_DIR:
        with open(os.path.join(LOG_DIR, "predictions.json"), "w") as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
        evaluation_log_fp = open(os.path.join(LOG_DIR, "evaluation.log"), "w")
    else:
        evaluation_log_fp = None

    eval_bar = tqdm(
        predictions, total=len(predictions), desc="Evaluating Predictions (0/0/0.00%)"
    )
    curr_cnt = 0
    for prediction_dict in eval_bar:
        curr_cnt += 1
        query, ground_truth, prediction = (
            prediction_dict["query"],
            prediction_dict["ground_truth"],
            prediction_dict["prediction"],
        )
        domain, ques_type = prediction_dict["domain"], prediction_dict["question_type"]
        static_or_dynamic = prediction_dict["static_or_dynamic"]
        # [ratio, correct, total]
        correct_by_domain.setdefault(domain, [0, 0, 0])
        correct_by_ques_type.setdefault(ques_type, [0, 0, 0])
        correct_by_static_or_dynamic.setdefault(static_or_dynamic, [0, 0, 0])
        correct_by_domain[domain][2] += 1
        correct_by_ques_type[ques_type][2] += 1
        correct_by_static_or_dynamic[static_or_dynamic][2] += 1

        messages = [
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": f"Question: {query}\n Ground truth: {ground_truth}\n Prediction: {prediction}\n",
            },
        ]
        # if prediction == "i don't know" or prediction == "i don't know.":
        if "i don't know" in prediction.lower():
            n_miss += 1
            prediction_dict["eval_res"] = "miss"
            continue
        if prediction.lower() == ground_truth.lower():
            n_correct_exact += 1
            n_correct += 1
            prediction_dict["eval_res"] = "correct_exact"
            correct_by_domain[domain][1] += 1
            correct_by_ques_type[ques_type][1] += 1
            correct_by_static_or_dynamic[static_or_dynamic][1] += 1
            eval_bar.set_description(
                f"Evaluating Predictions ({n_correct}/{curr_cnt}/{n_correct / curr_cnt:.2%})"
            )
            continue

        response = attempt_api_call(openai_client, evaluation_model_name, messages)
        logger.info(f"Raw response:\n{response}")
        if response:
            log_response(messages, response)
            eval_res = parse_response(response)
            logger.info(f"Parsed response: {eval_res}")
            if eval_res == 1:
                n_correct += 1
                correct_by_domain[domain][1] += 1
                correct_by_ques_type[ques_type][1] += 1
                correct_by_static_or_dynamic[static_or_dynamic][1] += 1
        else:
            eval_res = None
            logger.info("No parsed response")
        prediction_dict["eval_res"] = "correct" if eval_res == 1 else "hallucination"
        if evaluation_log_fp:
            evaluation_log_fp.write(
                "#" * 150
                + f"\nQuery: {query}\n"
                + f"Ground truth: {ground_truth}\n"
                + f"Prediction: {prediction}\n"
                + f"Eval response: {response}\n"
                + f"Parsed response: {eval_res}\n\n"
            )
            evaluation_log_fp.flush()
        eval_bar.set_description(
            f"Evaluating Predictions ({n_correct}/{curr_cnt}/{n_correct / curr_cnt:.2%})"
        )
    # re-save predictions with scores
    if LOG_DIR:
        with open(os.path.join(LOG_DIR, "predictions.json"), "w") as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
    n = len(predictions)
    assert curr_cnt == n
    for domain, (_, d_correct, d_total) in correct_by_domain.items():
        correct_by_domain[domain][0] = d_correct / d_total
    for ques_type, (_, q_correct, q_total) in correct_by_ques_type.items():
        correct_by_ques_type[ques_type][0] = q_correct / q_total
    for static_or_dynamic, (
        _,
        s_correct,
        s_total,
    ) in correct_by_static_or_dynamic.items():
        correct_by_static_or_dynamic[static_or_dynamic][0] = s_correct / s_total
    results = {
        "score": (2 * n_correct + n_miss) / n - 1,
        "exact_accuracy": n_correct_exact / n,
        "accuracy": n_correct / n,
        "hallucination": (n - n_correct - n_miss) / n,
        "missing": n_miss / n,
        "n_miss": n_miss,
        "n_correct": n_correct,
        "n_correct_exact": n_correct_exact,
        "total": n,
        "details": {
            "Acc by domain": correct_by_domain,
            "Acc by ques type": correct_by_ques_type,
            "Acc by static or dynamic": correct_by_static_or_dynamic,
        },
    }
    logger.info(json.dumps(results, indent=2))
    evaluation_log_fp.close()
    return results


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    return seed


if __name__ == "__main__":
    from models.user_config import UserModel

    parser = argparse.ArgumentParser()
    parser.add_argument("--use-public-test", action="store_true")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", type=str)
    parser.add_argument("--local-large-files-path", type=str)
    args = parser.parse_args()
    if args.log_dir:
        LOG_DIR = args.log_dir
        os.makedirs(LOG_DIR, exist_ok=True)
        logger.add(os.path.join(LOG_DIR, "main_script.log"), level="DEBUG", mode="w")
    logger.info(json.dumps(vars(args), indent=2))
    setup_seed(args.seed)
    if args.use_public_test:
        DATASET_PATH = "../data-explore/crag_task_1_dev_v3_release.jsonl"
    else:
        DATASET_PATH = "example_data/dev_data.jsonl.bz2"

    # EVALUATION_MODEL_NAME = os.getenv("EVALUATION_MODEL_NAME", "gpt-3.5-turbo-0125")
    # openai_client = OpenAI(api_key=openai_api_key)

    EVALUATION_MODEL_NAME = "deepseek-chat"
    openai_client = OpenAI(
        base_url="https://api.deepseek.com",
        api_key=deepseek_api_key,
    )

    # Generate predictions
    # participant_model = UserModel(debug=True)
    participant_model = UserModel(
        nprocs=torch.cuda.device_count(),
        local_large_files_path=args.local_large_files_path,
        return_reasoning=True,
    )
    predictions = generate_predictions(DATASET_PATH, participant_model)

    evaluation_results = evaluate_predictions(
        predictions, EVALUATION_MODEL_NAME, openai_client
    )
