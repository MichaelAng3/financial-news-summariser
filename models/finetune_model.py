from transformers import (
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
)
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType
import torch
from datasets import load_dataset
import os
import argparse

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Base model to fine-tune
MODEL_NAME = "google/flan-t5-base"

# Configuration for LoRA (parameter-efficient fine-tuning)
LORA_CONFIG = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q", "v", "k", "o", "wi", "wo"]
)

def preprocess_function(examples, tok):
    # Clean and filter short summaries
    pairs = [(t.strip(), s.strip()) for t, s in
             zip(examples["text"], examples["summary"])
             if s and len(s.split()) >= 6]

    if not pairs:
        return {}

    # Tokenize input texts and summaries
    texts, summaries = zip(*pairs)
    model_inputs = tok(list(texts), max_length=512, truncation=True)

    with tok.as_target_tokenizer():
        model_inputs["labels"] = tok(list(summaries),
                                     max_length=150,
                                     truncation=True).input_ids
    return model_inputs

def fine_tune(train_path, output_dir, per_device_train_batch_size, num_train_epochs=5):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.config.use_cache = False  # Disable for compatibility with checkpointing

    # Apply LoRA to enable parameter-efficient fine-tuning
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()  # Print which parameters are being trained

    # Load and split dataset into train/validation
    raw_dataset = load_dataset('json', data_files={"data": train_path})["data"]
    split_dataset = raw_dataset.train_test_split(test_size=0.1, seed=42)

    tokenized_train = split_dataset["train"].map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=split_dataset["train"].column_names,
    )
    tokenized_eval = split_dataset["test"].map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=split_dataset["test"].column_names,
    )

    # Set up training configuration
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        num_train_epochs=num_train_epochs,
        max_grad_norm=1.0,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=False,
        gradient_checkpointing=False,
        report_to="none",
        seed=42,
    )

    # Define how batches are collated (handles padding, etc.)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        pad_to_multiple_of=8
    )

    # Initialize HuggingFace trainer and begin fine-tuning
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    # Save the fine-tuned model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

# ───────────── CLI Entry Point ─────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', required=True, help="Path to training JSONL file")
    parser.add_argument('--output_dir', default="./finetuned_model", help="Directory to save model")
    parser.add_argument('--per_device_train_batch_size', type=int, default=8, help="Batch size per device")
    parser.add_argument('--num_train_epochs', type=int, default=5, help="Number of training epochs")
    args = parser.parse_args()

    fine_tune(args.train_data, args.output_dir, args.per_device_train_batch_size, args.num_train_epochs)
