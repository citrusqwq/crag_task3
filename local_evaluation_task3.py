import bz2
import json
import os
import math
import argparse
import random
import torch
import numpy as np
import traceback
import time
from datetime import datetime

from loguru import logger
from openai import APIConnectionError, OpenAI, RateLimitError
from prompts.templates import IN_CONTEXT_EXAMPLES, INSTRUCTIONS
from tqdm.auto import tqdm

# from transformers import LlamaTokenizerFast

# from api_key import deepseek_api_key, openai_api_key


# tokenizer = LlamaTokenizerFast.from_pretrained("tokenizer")
# LOG_DIR = None
LOG_DIR = "./Log"


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


'''
def trim_predictions_to_max_token_length(prediction):
    """Trims prediction output to 75 tokens"""
    max_token_length = 75
    tokenized_prediction = tokenizer.encode(prediction)
    trimmed_tokenized_prediction = tokenized_prediction[1 : max_token_length + 1]
    trimmed_prediction = tokenizer.decode(trimmed_tokenized_prediction)
    return trimmed_prediction
'''


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


def generate_predictions(dataset_path, participant_model, case_idxs: list[int]):
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
    batch_size = participant_model.get_batch_size()
    all_prompt_lens: list[int] = []
    all_prompts: list[str] = []
    all_reasoning_prompts: list[str] = []
    all_reasoning_outputs: list[list[str | dict]] = []

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
    if case_idxs:
        dataset_items = [dataset_items[idx] for idx in case_idxs]
    batch_num = math.ceil(len(dataset_items) / batch_size)
    logger.info(f"Number of data items: {len(dataset_items)}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Number of batches: {batch_num}")
    batch_data_generator = load_data_in_batches(dataset_items, batch_size)
    all_predicted_attrs: list[dict[str, str]] = []
    for batch in tqdm(
        batch_data_generator, desc="Generating predictions", total=batch_num
    ):
        batch_ground_truths = batch.pop(
            "answer"
        )  # Remove answers from batch and store them
        batch_predictions, batch_additional_info = (
            participant_model.batch_generate_answer(batch)
        )
        if "all_predicted_attrs" in batch_additional_info:
            batch_predicted_attrs: dict[str, list[str]] = batch_additional_info[
                "all_predicted_attrs"
            ]
            curr_pred_attrs_list = [{} for _ in range(len(batch["query"]))]
            for attr_name in batch_predicted_attrs:
                for idx in range(len(batch_predicted_attrs[attr_name])):
                    if attr_name not in curr_pred_attrs_list[idx]:
                        curr_pred_attrs_list[idx][attr_name] = batch_predicted_attrs[
                            attr_name
                        ][idx]
                    else:
                        raise ValueError(
                            f"Attribute {attr_name} already exists for index {idx}"
                        )
            all_predicted_attrs.extend(curr_pred_attrs_list)
            # logger.debug(len(all_predicted_attrs))
        if "prompt_lens" in batch_additional_info:
            all_prompt_lens.extend(batch_additional_info["prompt_lens"])
        if "prompts" in batch_additional_info:
            all_prompts.extend(batch_additional_info["prompts"])
        if "reasoning_prompts" in batch_additional_info:
            all_reasoning_prompts.extend(batch_additional_info["reasoning_prompts"])
        if "reasoning_outputs" in batch_additional_info:
            all_reasoning_outputs.extend(batch_additional_info["reasoning_outputs"])
        queries.extend(batch["query"])
        domains.extend(batch["domain"])
        question_types.extend(batch["question_type"])
        static_or_dynamics.extend(batch["static_or_dynamic"])
        ground_truths.extend(batch_ground_truths)
        predictions.extend(batch_predictions)
    final_predictions = []
    assert (
        len(queries)
        == len(ground_truths)
        == len(predictions)
        == len(domains)
        == len(question_types)
        == len(static_or_dynamics)
    )
    if len(all_predicted_attrs) > 0:
        assert len(all_predicted_attrs) == len(queries), (
            len(all_predicted_attrs),
            len(queries),
        )
    if len(all_prompts) > 0:
        assert len(all_prompts) == len(queries), (len(all_prompts), len(queries))
    for idx in range(len(all_predicted_attrs)):
        with open(os.path.join(LOG_DIR, "all_predicted_attrs.jsonl"), "w") as f:
            f.write(json.dumps(all_predicted_attrs[idx]) + "\n")
    curr_idx = 0
    for query, ground_truth, prediction, domain, ques_type, static_or_dynamic in zip(
        queries, ground_truths, predictions, domains, question_types, static_or_dynamics
    ):
        # trim prediction to 75 tokens
        # prediction = trim_predictions_to_max_token_length(prediction)
        if len(all_predicted_attrs) > 0:
            pred_attrs = all_predicted_attrs[curr_idx]
        else:
            pred_attrs = {}
        final_predictions.append(
            {
                "query": query,
                "ground_truth": str(ground_truth).strip().lower(),
                "prediction": str(prediction).strip().lower(),
                "domain": domain,
                "question_type": ques_type,
                "static_or_dynamic": static_or_dynamic,
                "pred_attrs": pred_attrs,
                "prompt_len": all_prompt_lens[curr_idx] if all_prompt_lens else "N/A",
            }
        )
        curr_idx += 1
    logger.info(f"Max prompt length: {max(all_prompt_lens)}")
    logger.info(f"Min prompt length: {min(all_prompt_lens)}")
    sorted_all_prompt_lens = sorted(all_prompt_lens)
    for i in range(1, 10):
        logger.info(
            f"{i * 10}%: {sorted_all_prompt_lens[len(sorted_all_prompt_lens) * i // 10]}"
        )
    # save all prompt lens
    with open(os.path.join(LOG_DIR, "all_prompt_lens.txt"), "w") as f:
        for prompt_len in all_prompt_lens:
            f.write(f"{prompt_len}\n")
    with open(os.path.join(LOG_DIR, "all_prompts.txt"), "w") as f:
        for query, prompt in zip(queries, all_prompts):
            f.write("=" * 50 + "\n")
            f.write(f"Query:\n{query}\n")
            f.write("*" * 50 + "\n")
            f.write(f"Prompt:\n{prompt}\n")
            f.write("=" * 50 + "\n")
    with open(os.path.join(LOG_DIR, "all_reasoning.txt"), "w") as f:
        for query, reasoning_prompt, reasoning_output in zip(
            queries, all_reasoning_prompts, all_reasoning_outputs
        ):
            if reasoning_prompt is None:
                assert reasoning_output is None
            if reasoning_output is None:
                assert reasoning_prompt is None

            f.write("=" * 50 + "\n")
            f.write(f"Query:\n{query}\n")
            f.write("*" * 50 + "\n")
            f.write(f"Reasoning prompt:\n{reasoning_prompt}\n")
            f.write("*" * 50 + "\n")
            if reasoning_output is None:
                f.write(f"Reasoning output (sample_num=None):\n")
                f.write("None\n")
            else:
                f.write(f"Reasoning output (sample_num={len(reasoning_output)-1}):\n")
                for output in reasoning_output:
                    if isinstance(output, dict):
                        f.write(json.dumps(output, indent=2) + "\n")
                    else:
                        f.write(f"{output}\n")
                f.write("=" * 50 + "\n")

    for idx, prediction in enumerate(final_predictions):
        prediction["idx"] = idx
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
                # prediction = trim_predictions_to_max_token_length(prediction)
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
            # prediction = trim_predictions_to_max_token_length(prediction)
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
        correct_by_domain.setdefault(domain, [0, 0, 0, 0])
        correct_by_ques_type.setdefault(ques_type, [0, 0, 0, 0])
        correct_by_static_or_dynamic.setdefault(static_or_dynamic, [0, 0, 0, 0])
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
            correct_by_domain[domain][3] += 1
            correct_by_ques_type[ques_type][3] += 1
            correct_by_static_or_dynamic[static_or_dynamic][3] += 1
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
    for domain, (_, d_correct, d_total, d_missing) in correct_by_domain.items():
        correct_by_domain[domain][0] = d_correct / d_total
        correct_by_domain[domain].append(d_correct / (d_total - d_missing + 1e-6))
    for ques_type, (_, q_correct, q_total, q_missing) in correct_by_ques_type.items():
        correct_by_ques_type[ques_type][0] = q_correct / q_total
        correct_by_ques_type[ques_type].append(q_correct / (q_total - q_missing + 1e-6))
    for static_or_dynamic, (
        _,
        s_correct,
        s_total,
        s_missing,
    ) in correct_by_static_or_dynamic.items():
        correct_by_static_or_dynamic[static_or_dynamic][0] = s_correct / s_total
        correct_by_static_or_dynamic[static_or_dynamic].append(
            s_correct / (s_total - s_missing + 1e-6)
        )
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


def evaluate_attr_predictions(predictions: list[dict]):
    acc_by_values = {}
    for pred in predictions:
        pred_attrs = pred["pred_attrs"]
        for attr_name in pred_attrs:
            curr_pred = pred_attrs[attr_name]
            if attr_name not in acc_by_values:
                acc_by_values[attr_name] = {}
            if attr_name == "domain":
                curr_domain = pred["domain"]
                curr_gt = curr_domain
            elif attr_name == "question_type":
                curr_ques_type = pred["question_type"]
                if curr_ques_type in ["simple", "simple_w_condition"]:
                    curr_ques_type = "simple"
                else:
                    curr_ques_type = "complex"
                curr_gt = curr_ques_type
            elif attr_name == "static_or_dynamic":
                curr_static_or_dynamic = pred["static_or_dynamic"]
                if curr_static_or_dynamic in ["static", "slow-changing"]:
                    curr_static_or_dynamic = "static"
                else:
                    curr_static_or_dynamic = "dynamic"
                curr_gt = curr_static_or_dynamic
            else:
                raise ValueError(f"Unknown attribute name: {attr_name}")
            if curr_gt not in acc_by_values[attr_name]:
                acc_by_values[attr_name][curr_gt] = [0, 0]
            if curr_pred == curr_gt:
                acc_by_values[attr_name][curr_gt][0] += 1
            acc_by_values[attr_name][curr_gt][1] += 1
    for attr_name in acc_by_values:
        name_correct = 0
        name_total = 0
        for value in acc_by_values[attr_name]:
            name_correct += acc_by_values[attr_name][value][0]
            name_total += acc_by_values[attr_name][value][1]
            acc_by_values[attr_name][value].append(
                acc_by_values[attr_name][value][0] / acc_by_values[attr_name][value][1]
            )
        acc_by_values[attr_name]["total"] = [
            name_correct,
            name_total,
            name_correct / name_total,
        ]
    logger.info(json.dumps(acc_by_values, indent=2))
    with open(os.path.join(LOG_DIR, "attr_predictions_res.json"), "w") as f:
        json.dump(acc_by_values, f, indent=2)


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    return seed


def save_few_shots(few_shots, key):
    with open(os.path.join(LOG_DIR, f"{key}_few_shots.jsonl"), "w") as f:
        for shot in few_shots:
            f.write(json.dumps(shot) + "\n")


if __name__ == "__main__":
    from models.user_config import UserModel

    parser = argparse.ArgumentParser()
    parser.add_argument("--use-public-test", action="store_true")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--model_size", type=str, choices=["8B", "70B"], default="70B")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", type=str)
    parser.add_argument("--local-large-files-path", type=str)
    parser.add_argument("--custom_local_model_path", type=str)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.6)
    parser.add_argument("--sentence_model_name", type=str, default="sentence-t5-large")
    parser.add_argument("--no_use_table_info", action="store_true")
    parser.add_argument("--max_table_length", type=int, default=4000)
    parser.add_argument("--num_context_sentences", type=int, default=10)
    parser.add_argument("--max_context_sentence_length", type=int, default=200)
    parser.add_argument("--case_idxs", type=int, nargs="+")
    parser.add_argument(
        "--idk_attrs",
        type=str,
        default='{"static_or_dynamic": ["dynamic"]}',
    )
    parser.add_argument("--include_table_in_text", action="store_true")
    parser.add_argument(
        "--query_prompt",
        type=str,
        default="",
    )
    parser.add_argument(
        "--invalid_question_keys", type=str, default='["none", "never"]'
    )
    args = parser.parse_args()
    args.idk_attrs = json.loads(args.idk_attrs)
    args.invalid_question_keys = json.loads(args.invalid_question_keys)
    if args.log_dir:
        LOG_DIR = args.log_dir
        os.makedirs(LOG_DIR, exist_ok=True)
        logger.add(os.path.join(LOG_DIR, "main_script.log"), level="DEBUG", mode="w")
        os.system(
            "git rev-parse HEAD > {}".format(os.path.join(LOG_DIR, "commit_id.txt"))
        )
        os.system("git status >> {}".format(os.path.join(LOG_DIR, "commit_id.txt")))
        os.system("git diff >> {}".format(os.path.join(LOG_DIR, "commit_id.txt")))
    logger.info(json.dumps(vars(args), indent=2))
    setup_seed(args.seed)
    if args.use_public_test:
        # DATASET_PATH = "../data-explore/crag_task_3_dev_v3/crag_task_3_dev_v3_full.jsonl"
        DATASET_PATH = "../crag_task_3_dev_v4/crag_task_3_dev_v4_1.jsonl"
        # DATASET_PATH = "./dev_data.jsonl"
    else:
        raise ValueError("Only public test is supported for now.")

    # EVALUATION_MODEL_NAME = os.getenv("EVALUATION_MODEL_NAME", "gpt-3.5-turbo-0125")
    # openai_client = OpenAI(api_key=openai_api_key)

    EVALUATION_MODEL_NAME = os.getenv("EVALUATION_MODEL_NAME", "gemma2:27b")

    # EVALUATION_MODEL_NAME = "deepseek-chat"
    # openai_client = OpenAI(
    #    base_url="https://api.deepseek.com",
    #    api_key=deepseek_api_key,
    # )

    # Generate predictions
    # participant_model = UserModel(debug=True)
    # FIXME: No options should be the default setting of the model
    #        `include_table_in_text` should be changed.
    participant_model = UserModel(
        model_size=args.model_size,
        batch_size=args.batch_size,
        nprocs=torch.cuda.device_count(),
        local_large_files_path=args.local_large_files_path,
        custom_local_model_path=args.custom_local_model_path,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        return_additional_info=True,
        sentence_model_name=args.sentence_model_name,
        use_table_info=not args.no_use_table_info,
        idk_attrs=args.idk_attrs,
        max_table_length=args.max_table_length,
        num_context_sentences=args.num_context_sentences,
        max_context_sentence_length=args.max_context_sentence_length,
        log_while_inference=False,
        query_prompt=args.query_prompt,
        include_table_in_text=args.include_table_in_text,
        invalid_question_keys=args.invalid_question_keys,
    )
    if participant_model.attr_predictor is not None:
        save_few_shots(
            participant_model.attr_predictor.question_type_few_shots, "question_type"
        )
        save_few_shots(
            participant_model.attr_predictor.static_dynamic_few_shots,
            "static_or_dynamic",
        )
    logger.info(f"Using model: {participant_model}")
    START_TIME = time.perf_counter()
    predictions = generate_predictions(DATASET_PATH, participant_model, args.case_idxs)
    evaluate_attr_predictions(predictions)
    # format time
    elapsed_time = time.perf_counter() - START_TIME
    formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    logger.info("Prediction time: {}".format(formatted_time))

    openai_client = OpenAI(
        base_url=os.getenv("INTERWEB_HOST", "https://interweb.l3s.uni-hannover.de"),
        api_key=os.getenv("INTERWEB_APIKEY"),
    )

    evaluation_results = evaluate_predictions(
        predictions, EVALUATION_MODEL_NAME, openai_client
    )
