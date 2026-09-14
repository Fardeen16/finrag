# syntax=docker/dockerfile:1
# Queue worker for MiniLM rerank. Official cu128 base so torch matches 12.8/13.0 hosts.
FROM runpod/pytorch:1.0.3-cu1281-torch280-ubuntu2404

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    HF_HUB_CACHE=/models \
    TRANSFORMERS_CACHE=/models \
    SENTENCE_TRANSFORMERS_HOME=/models \
    MODEL_NAMES=cross-encoder/ms-marco-MiniLM-L-6-v2 \
    HF_HUB_OFFLINE=0

COPY requirements.txt /app/requirements.txt
# Base image cryptography is Debian-managed; ignore RECORD so the RunPod SDK installs.
RUN pip install --no-cache-dir --break-system-packages --ignore-installed cryptography \
    && pip install --no-cache-dir --break-system-packages -r /app/requirements.txt \
    && python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

COPY handler.py test_input.json /app/

# Bake MiniLM so the first serverless job does not download weights under the 40s watchdog.
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu')"

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

CMD ["python", "-u", "/app/handler.py"]
