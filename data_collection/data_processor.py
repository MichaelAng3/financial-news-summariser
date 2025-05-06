import json
from argparse import ArgumentParser
from datetime import datetime

class DataProcessor:
    def clean_text(self, text):
        # remove whitespace
        return text.strip()

    def process_file(self, input_path, output_path):
        processed = []

        # Open and read the input JSONL file line by line
        with open(input_path, 'r') as f:
            for line in f:
                article = json.loads(line)

                # Skip entries missing required fields
                if 'text' not in article or 'summary' not in article:
                    continue

                # Format and clean each article before saving
                processed.append({
                    'text': self.clean_text(article['text']),
                    'summary': self.clean_text(article['summary']),
                    'source_url': article.get('url', 'N/A'),
                    'stock': article.get('stock', 'UNKNOWN'),
                    # Use current timestamp if missing
                    'timestamp': article.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                })

        # Write the processed entries back to a new JSONL file
        with open(output_path, 'w') as f:
            for item in processed:
                f.write(json.dumps(item) + '\n')

# ───────────── Command-Line Interface ─────────────
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--input', required=True)   # Path to raw input file
    parser.add_argument('--output', required=True)  # Path to save cleaned output
    args = parser.parse_args()

    # Run the processor
    processor = DataProcessor()
    processor.process_file(args.input, args.output)