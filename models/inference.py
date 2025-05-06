import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel, PeftConfig
from argparse import ArgumentParser

class FineTunedInference:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load PEFT config and base model
        self.config = PeftConfig.from_pretrained(model_path)
        self.base_model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.base_model_name_or_path
        )

        # Load fine-tuned LoRA adapter weights into base model
        self.model = PeftModel.from_pretrained(
            self.base_model,
            model_path
        ).to(self.device)

        # Load tokenizer associated with base model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model_name_or_path
        )

    def summarize(self, text, max_length=150):
        # Tokenize input text for summarization
        inputs = self.tokenizer(
            f"summarize: {text}",
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)

        # Generate summary using beam search
        outputs = self.model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=max_length,
            num_beams=4,
            early_stopping=True
        )

        # Decode the generated tokens into text
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    parser = ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to processed data')  # Input JSONL file
    parser.add_argument('--model_path', required=True, help='Path to fine-tuned model')  # LoRA model path
    parser.add_argument('--output', required=True, help='Output file path')  # Output file with summaries
    args = parser.parse_args()

    # Initialize inference model
    inferencer = FineTunedInference(args.model_path)

    # Read input file, generate summaries and save output
    with open(args.input, 'r') as fin, open(args.output, 'w') as fout:
        for line in fin:
            article = json.loads(line)

            # Generate summary using fine-tuned model
            ft_summary = inferencer.summarize(article['text'])

            # Add generated summary and timestamp (if missing)
            article['ft_summary'] = ft_summary
            if 'timestamp' not in article:
                from datetime import datetime
                article['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Write updated article to output
            fout.write(json.dumps(article) + '\n')

# ───────────── CLI Entry Point ─────────────
if __name__ == "__main__":
    main()