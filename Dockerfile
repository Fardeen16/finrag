# syntax=docker/dockerfile:1

# --- frontend ---------------------------------------------------------------
FROM node:22-alpine AS frontend
WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci || npm install
COPY frontend/ ./
RUN npm run build

# --- backend ----------------------------------------------------------------
# Hugging Face Docker Spaces run as UID 1000. Install system packages as root,
# then drop privileges before baking weights or starting uvicorn.
FROM python:3.12-slim AS runtime

RUN useradd -m -u 1000 user

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/user/.cache/huggingface \
    EMBEDDING_PROVIDER=local \
    AGENT_MODE=live \
    PORT=8080 \
    PATH=/home/user/.local/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# CPU torch first so sentence-transformers does not pull a CUDA wheel.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY --chown=user:user backend ./backend
COPY --chown=user:user data/alphabet_financials_structured.csv data/alphabet_financials_structured.csv
COPY --from=frontend --chown=user:user /ui/dist ./frontend/dist

USER user

# Bake model weights so the first Spaces request does not download HF models.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

EXPOSE 8080
CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
