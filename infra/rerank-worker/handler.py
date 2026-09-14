"""Queue worker: MiniLM CrossEncoder. Load at import so the first job is inference only."""

from __future__ import annotations

import os
from typing import Any

os.environ.setdefault("HF_HOME", "/models")
os.environ.setdefault("HF_HUB_CACHE", "/models")
os.environ.setdefault("TRANSFORMERS_CACHE", "/models")
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", "/models")

import runpod  # noqa: E402
import torch  # noqa: E402
from sentence_transformers import CrossEncoder  # noqa: E402

MODEL_ID = os.environ.get("MODEL_NAMES", "cross-encoder/ms-marco-MiniLM-L-6-v2")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_MODEL = CrossEncoder(MODEL_ID, device=DEVICE, trust_remote_code=False)


def _as_docs(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item if isinstance(item, str) else str(item or "") for item in value]
    return []


def handler(job: dict[str, Any]) -> dict[str, Any]:
    payload = job.get("input") or {}
    query = payload.get("query")
    docs = _as_docs(payload.get("docs"))
    if not isinstance(query, str) or not query.strip() or not docs:
        return {"error": "Need string query and a non-empty docs list"}

    scores = _MODEL.predict([[query, doc] for doc in docs], convert_to_numpy=True)
    return {
        "model": MODEL_ID,
        "scores": [float(score) for score in scores],
        "usage": {"total": len(docs)},
    }


runpod.serverless.start({"handler": handler})
