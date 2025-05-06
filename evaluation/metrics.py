from rouge_score import rouge_scorer
from transformers import pipeline
import numpy as np
import json
import torch
import argparse
from tqdm import tqdm
from bert_score import score as bert_score

class Evaluator:
    def __init__(self):
        # Initialize ROUGE-L scorer and NLI-based zero-shot classifier for factuality
        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.nli_checker = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=0 if torch.cuda.is_available() else -1
        )

    def _truncate(self, text, max_chars):
        # Truncate text at word boundary within max_chars
        return text[:max_chars].rsplit(' ', 1)[0] if text else ""

    def calculate_rouge(self, reference, candidate):
        # Compute ROUGE-L score between truncated reference and candidate
        try:
            ref_trunc = self._truncate(reference, 512)
            cand_trunc = self._truncate(candidate, 150)
            return self.scorer.score(ref_trunc, cand_trunc)['rougeL'].fmeasure
        except:
            return 0.0

    def calculate_bertscore(self, reference, candidate):
        # Compute BERTScore between reference and candidate
        try:
            P, R, F1 = bert_score(
                [candidate], [reference],
                lang="en", model_type="bert-base-uncased",
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            return F1[0].item()
        except Exception:
            return 0.0

    def check_factuality_batch(self, premises, hypotheses):
        # Use NLI model to check factuality (entailment) for summary against source
        scores = []
        try:
            results = self.nli_checker(
                sequences=hypotheses,
                candidate_labels=["entailment", "neutral", "contradiction"],
                hypothesis_template="This is supported by the text: '{}'"
            )
            for result in results:
                score = result['scores'][result['labels'].index('entailment')]
                scores.append(score)
        except Exception as e:
            scores = [0.0 for _ in hypotheses]
        return scores

def evaluate_batch(data_path):
    evaluator = Evaluator()
    metrics = {
        'rouge_baseline': [],
        'rouge_finetuned': [],
        'factual_baseline': [],
        'factual_finetuned': [],
        'bertscore_baseline': [],
        'bertscore_finetuned': []
    }

    # Load JSONL data
    try:
        with open(data_path, 'r') as f:
            all_lines = [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: File {data_path} not found")
        return None

    # Lists for batch factuality checking
    baseline_premises, baseline_hypotheses = [], []
    finetuned_premises, finetuned_hypotheses = [], []

    for article in tqdm(all_lines, desc="Collecting metrics"):
        try:
            # Ensure all required fields are present
            required_fields = ['text', 'baseline_summary', 'ft_summary']
            if not all(field in article for field in required_fields):
                continue

            ref = article['text']
            base_sum = article['baseline_summary']
            ft_sum = article['ft_summary']

            # ROUGE-L
            metrics['rouge_baseline'].append(evaluator.calculate_rouge(ref, base_sum))
            metrics['rouge_finetuned'].append(evaluator.calculate_rouge(ref, ft_sum))

            # BERTScore
            metrics['bertscore_baseline'].append(evaluator.calculate_bertscore(ref, base_sum))
            metrics['bertscore_finetuned'].append(evaluator.calculate_bertscore(ref, ft_sum))

            # Prepare for factuality scoring
            baseline_premises.append(ref)
            baseline_hypotheses.append(base_sum)
            finetuned_premises.append(ref)
            finetuned_hypotheses.append(ft_sum)

        except Exception as e:
            tqdm.write(f"Skipping article due to error: {str(e)}")

    # Factuality (entailment) scores
    metrics['factual_baseline'] = evaluator.check_factuality_batch(
        baseline_premises, baseline_hypotheses
    )
    metrics['factual_finetuned'] = evaluator.check_factuality_batch(
        finetuned_premises, finetuned_hypotheses
    )

    # Return averaged metrics
    return {
        'rouge_baseline': np.mean(metrics['rouge_baseline']),
        'rouge_finetuned': np.mean(metrics['rouge_finetuned']),
        'factual_baseline': np.mean(metrics['factual_baseline']),
        'factual_finetuned': np.mean(metrics['factual_finetuned']),
        'bertscore_baseline': np.mean(metrics['bertscore_baseline']),
        'bertscore_finetuned': np.mean(metrics['bertscore_finetuned']),
    }

def save_hallucinations(data_path, output_path='hallucinations.json', threshold=0.5):
    evaluator = Evaluator()
    hallucinations = []

    # Check each summary for low factuality (entailment score < threshold)
    with open(data_path, 'r') as f:
        for line in tqdm(f, desc="Detecting Hallucinations"):
            article = json.loads(line)
            try:
                text = article['text']
                for key in ['baseline_summary', 'ft_summary']:
                    if key not in article:
                        continue
                    summary = article[key]
                    entail_score = evaluator.check_factuality_batch([text], [summary])[0]
                    if entail_score < threshold:
                        hallucinations.append({
                            'type': key,
                            'text': text,
                            'summary': summary,
                            'entailment_score': entail_score
                        })
            except Exception:
                continue

    # Save detected hallucinations to file
    with open(output_path, 'w') as fout:
        json.dump(hallucinations, fout, indent=2)

# ───────────── CLI Entry Point ─────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate summarization performance')
    parser.add_argument('--data', required=True, help='Path to JSONL file containing summaries')
    args = parser.parse_args()

    results = evaluate_batch(args.data)

    if results:
        # Display evaluation metrics in a table
        print("\nEvaluation Results")
        print(f"| {'Metric':<20} | {'Baseline':<10} | {'Fine-tuned':<10} |")
        print(f"| {'-'*20} | {'-'*10} | {'-'*10} |")
        print(f"| {'ROUGE-L Score':<20} | {results['rouge_baseline']:>10.3f} | {results['rouge_finetuned']:>10.3f} |")
        print(f"| {'Factuality Score':<20} | {results['factual_baseline']:>10.2%} | {results['factual_finetuned']:>10.2%} |")
        print(f"| {'BERTScore':<20} | {results['bertscore_baseline']:>10.3f} | {results['bertscore_finetuned']:>10.3f} |")

        # Save results to file
        with open('evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
    else:
        print("Evaluation failed due to errors")

    save_hallucinations(args.data)
    print("Saved hallucinated summaries to hallucinations.json")
