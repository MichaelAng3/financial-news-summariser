#!/bin/bash

set -e

echo ">> Step 1: Load Raw Articles"
python data_collection/dataset_loader.py --input_csv ./data/Financial_News_Dataset.csv --output ./data/raw_articles.jsonl

echo ">> Step 2: Generate Summaries from Teacher"
python data_collection/data_processor.py --input ./data/raw_articles.jsonl --output ./data/processed_data.jsonl

echo ">> Step 3: Run Baseline Summarizer"
python models/baseline_summarizer.py --input ./data/processed_data.jsonl --output ./data/baseline_results.jsonl

echo ">> Step 4: Fine-Tune Model"
python models/finetune_model.py --train_data ./data/processed_data.jsonl --output_dir ./models/finetuned_model --per_device_train_batch_size 8

echo ">> Step 5: Inference from Fine-tuned Model"
python models/inference.py --input ./data/baseline_results.jsonl --model_path ./models/finetuned_model --output ./data/final_summaries.jsonl

echo ">> Step 6: Run Evaluation"
python evaluation/metrics.py --data ./data/final_summaries.jsonl
