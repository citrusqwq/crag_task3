from models.utils_html import *
import pandas as pd
import json
import argparse
import pickle
import os
import os.path as osp
from tqdm import tqdm
from pprint import pprint
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        default="../data-explore/crag_task_1_dev_v3_release.jsonl",
        help="Path to the JSONL file",
    )
    parser.add_argument("--outdir", default="local_eval_logs/debug-figs/")
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    data = []
    with open(args.file, "r") as f:
        for line in tqdm(f, ncols=100):
            line = line.strip()
            json_obj = json.loads(line)
            data.append(json_obj)

    all_cnt = 0
    have_table_cnt = 0
    table_lens = []
    table_nums = []
    tables = []
    for data_item in tqdm(data, desc="Processing tables..."):
        curr_query = data_item["query"]
        curr_test_data_webpages = data_item["search_results"]
        curr_tables = []
        for web_page in curr_test_data_webpages:
            page_html = web_page["page_result"]
            page_html = clean_html(page_html)
            curr_tables.extend(get_tables(page_html))
        tables.append(curr_tables)
        all_cnt += 1
        table_nums.append(len(curr_tables))
        table_lens.append(sum([len(table) for table in curr_tables]))
        if len(curr_tables) > 0:
            have_table_cnt += 1

    print(f"all_cnt: {all_cnt}, have_table_cnt: {have_table_cnt}")
    # Draw the table numbers histogram
    print("Drawing the table numbers histogram...")
    plt.hist(table_nums, bins=50)
    plt.xlabel("Table Numbers")
    plt.ylabel("Frequency")
    plt.title("Table Numbers Histogram")
    plt.savefig(osp.join(args.outdir, "table_nums_hist.png"))
    plt.cla()
    # Draw the table lens histogram
    print("Drawing the table lens histogram...")
    plt.hist(table_lens, bins=50)
    plt.xlabel("Table Lens")
    plt.ylabel("Frequency")
    plt.title("Table Lens Histogram")
    plt.savefig(osp.join(args.outdir, "table_lens_hist.png"))
    plt.cla()
    with open(osp.join(args.outdir, "tables.pkl"), "wb") as f:
        pickle.dump(tables, f)


if __name__ == "__main__":
    main()
