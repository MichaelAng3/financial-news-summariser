from transformers import pipeline
import json
import torch

class BaselineSummarizer:
    def __init__(self):
        # Initialize a text summarization pipeline using the Flan-T5 model
        self.summarizer = pipeline(
            "summarization",
            model="google/flan-t5-base",
            device=0 if torch.cuda.is_available() else -1  # Use GPU if available
        )

    def summarize(self, text, max_length=150):
        # Generate a summary for the given text
        try:
            return self.summarizer(
                text,
                max_length=max_length,
                min_length=30,
                do_sample=False,
                truncation=True   # Truncate input if too long
            )[0]['summary_text']
        except Exception as e:
            print(f"Error: {e}")
            return "Summary generation failed"

def process_batch(input_path, output_path):
    summarizer = BaselineSummarizer()

    # Open input file for reading and output file for writing
    with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in fin:
            article = json.loads(line)

            # Skip entries without a 'text' field
            if 'text' not in article:
                continue

            # Generate baseline summary and add it to the article
            article['baseline_summary'] = summarizer.summarize(article['text'])

            # Write the updated article to the output file
            fout.write(json.dumps(article) + '\n')

# ───────────── CLI Entry Point ─────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help="Input JSONL file path")
    parser.add_argument('--output', required=True, help="Output JSONL file path")
    args = parser.parse_args()

    process_batch(args.input, args.output)
