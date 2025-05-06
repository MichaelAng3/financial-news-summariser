# Financial‑News Summariser
This project monitors and summarises financial news articles using a fine-tuned language model based on **FLAN-T5**. It offers a complete ML pipeline — from dataset preparation to evaluation — and presents the results in a user-friendly **Streamlit dashboard**.

## Features
- **Summarisation Pipeline**: using a fine-tuned FLAN-T5 model with **LoRA** for parameter-efficient training.
- **Evaluation Metrics**: Includes ROUGE-L, BERTScore and factual  scoring via zero-shot entailment.
- **Hallucination Detection**: Identifies factually unsupported summaries using entailment confidence thresholds.
- **Baseline vs Fine-tuned**: comparison of summaries.
- **Interactive Web Dashboard**: with filters for date and stock tickers.
- **Dockerized Setup**: using Docker and Docker Compose for portability and reproducibility.

## Docker Setup

### Install Docker
Follow the instructions for your OS:
- Windows: https://docs.docker.com/desktop/setup/install/windows-install/
- Linux: https://docs.docker.com/desktop/setup/install/linux/
- Mac: https://docs.docker.com/desktop/setup/install/mac-install/

### Build and Run the Container
From the root directory:
```bash
sudo docker compose up --build -d
```

Check if it's running
```bash
docker ps
```

Open the Dashboard
```bash
http://localhost:8501
```


## Manual Pipeline Execution
```bash
bash run_pipeline.sh
```

This performs the following steps:

1. Load raw financial articles from CSV.

2. Clean and structure data into JSONL format.

3. Generate baseline summaries using FLAN-T5.

4. Fine-tune FLAN-T5 with LoRA on processed data.

5. Generate summaries from the fine-tuned model.

6. Evaluate with ROUGE, BERTScore and factuality scoring.

7. Detect hallucinated summaries based on low entailment scores.

8. Save results

## Local Development
Install Requirements
```bash
pip install -r requirements.txt
```

Run Streamlit Dashboard
```bash
streamlit run app/dashboard.py
```

## Requirements
See requirements.txt for full dependency list.