# ---- build image -------------------------------------------------
    FROM python:3.9

    # speed up installs & runtime
    ENV DEBIAN_FRONTEND=noninteractive \
        PYTHONDONTWRITEBYTECODE=1 \
        PYTHONUNBUFFERED=1
    
    WORKDIR /app
    
    # system deps for torch + tokenizers
    RUN apt-get update && apt-get install -y --no-install-recommends \
            git build-essential libffi-dev \
            && rm -rf /var/lib/apt/lists/*
    
    # requirements first (better caching)
    COPY requirements.txt .
    RUN pip install --timeout 300 --no-cache-dir -r requirements.txt
    
    # project code
    COPY . .
    
    # Streamlit uses port 8501
    EXPOSE 8501
    
    # default command: run dashboard
    CMD ["streamlit", "run", "app/dashboard.py", "--server.port", "8501", "--server.headless", "true"]