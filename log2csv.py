import json
import argparse
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert log file to csv file')
    parser.add_argument('--log_dir', type=str, help='log dir to convert, should contain predictions.json')
    parser.add_argument('--save_file', type=str, help='csv file to save', default=None)
    args = parser.parse_args()

    with open(args.log_dir + '/predictions.json', 'r') as f:
        predictions = json.load(f)

    df = pd.DataFrame(predictions)
    df = df[['query', 'ground_truth', 'top_sentences', 'prediction', 'eval_res']]
    if not args.save_file:
        args.save_file = args.log_dir + '/predictions.csv'
    df.to_csv(args.save_file, index=False)
    