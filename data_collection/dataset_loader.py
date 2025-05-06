import pandas as pd
import json
from argparse import ArgumentParser

def generate_dataset(input_csv, output_path):
    # Load the input CSV file into a pandas DataFrame
    df = pd.read_csv(input_csv)

    # Open the output file in write mode
    with open(output_path, 'w') as f:
        # Iterate over each row in the DataFrame and convert to JSONL format
        for _, row in df.iterrows():
            article = {
                'text': row['text'],
                'summary': row['summary'],
                'stock': row['stock'],
                'url': row['url'],
                'timestamp': row['timestamp']
            }
            # Write each article as a JSON object per line
            f.write(json.dumps(article) + '\n')

# ───────────── Command-Line Interface ─────────────
if __name__ == "__main__":
    parser = ArgumentParser(description='Convert full article CSV to JSONL')
    parser.add_argument('--input_csv', required=True, help='input CSV file')  # Path to input CSV
    parser.add_argument('--output', default='./data/raw_articles.jsonl', help='Output path')  # Output JSONL path
    args = parser.parse_args()

    # Run the dataset generator
    generate_dataset(args.input_csv, args.output)