import json
import argparse
import os
from pathlib import Path
from tqdm import tqdm
from loguru import logger

OUTPUT_PATH = None


def main():
    global OUTPUT_PATH
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-file", type=str, required=True)
    parser.add_argument("--new-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logger.add(os.path.join(args.output_dir, "analyze_results.log"), mode="w")
    logger.info(json.dumps(vars(args), indent=4))
    OUTPUT_PATH = Path(args.output_dir)
    with open(args.old_file, "r") as f:
        old_dpredictions = json.load(f)
    with open(args.new_file, "r") as f:
        new_predictions = json.load(f)
    assert len(old_dpredictions) == len(new_predictions)
    static_or_dynamic_diff = []
    removed_miss = []
    added_miss = []
    from_hallucination_to_correct = []
    from_correct_to_hallucination = []
    curr_idx = 0
    for old_item, new_item in tqdm(
        zip(old_dpredictions, new_predictions),
        total=len(old_dpredictions),
        ascii=True,
    ):
        assert old_item["query"] == new_item["query"], (old_item, new_item)
        assert curr_idx == old_item["idx"], (curr_idx, old_item["idx"])
        assert curr_idx == new_item["idx"], (curr_idx, new_item["idx"])
        if (
            old_item["pred_attrs"]["static_or_dynamic"]
            != new_item["pred_attrs"]["static_or_dynamic"]
        ):
            static_or_dynamic_diff.append((old_item["idx"]))
        if (
            old_item["eval_res"] == "hallucination"
            and new_item["eval_res"] == "correct"
        ):
            from_hallucination_to_correct.append((old_item["idx"]))
        if (
            old_item["eval_res"] == "correct"
            and new_item["eval_res"] == "hallucination"
        ):
            from_correct_to_hallucination.append((old_item["idx"]))
        if (
            old_item["eval_res"] == "miss"
            and new_item["eval_res"] != "miss"
        ):
            removed_miss.append((old_item["idx"]))
        if (
            old_item["eval_res"] != "miss"
            and new_item["eval_res"] == "miss"
        ):
            added_miss.append((old_item["idx"]))
        curr_idx += 1
    logger.info(f"static_or_dynamic_diff num: {len(static_or_dynamic_diff)}")
    logger.info(f"from_hallucination_to_correct num: {len(from_hallucination_to_correct)}")
    logger.info(f"from_correct_to_hallucination num: {len(from_correct_to_hallucination)}")
    logger.info(f"removed_miss num: {len(removed_miss)}")
    logger.info(f"added_miss num: {len(added_miss)}")
    with open(OUTPUT_PATH / "static_or_dynamic_diff.txt", "w") as f:
        f.write("\n".join(map(str, static_or_dynamic_diff)))
        f.write("\n\n" + " ".join(map(str, static_or_dynamic_diff)))
    with open(OUTPUT_PATH / "from_hallucination_to_correct.txt", "w") as f:
        f.write("\n".join(map(str, from_hallucination_to_correct)))
        f.write("\n\n" + " ".join(map(str, from_hallucination_to_correct)))
    with open(OUTPUT_PATH / "from_correct_to_hallucination.txt", "w") as f:
        f.write("\n".join(map(str, from_correct_to_hallucination)))
        f.write("\n\n" + " ".join(map(str, from_correct_to_hallucination)))
    with open(OUTPUT_PATH / "removed_miss.txt", "w") as f:
        f.write("\n".join(map(str, removed_miss)))
        f.write("\n\n" + " ".join(map(str, removed_miss)))
    with open(OUTPUT_PATH / "added_miss.txt", "w") as f:
        f.write("\n".join(map(str, added_miss)))
        f.write("\n\n" + " ".join(map(str, added_miss)))


if __name__ == "__main__":
    main()
