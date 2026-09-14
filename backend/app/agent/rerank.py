"""CrossEncoder rerank without PyTorch on the web host.

Default: FastEmbed ONNX MiniLM (~80 MB), the same architecture as
`cross-encoder/ms-marco-MiniLM-L-6-v2`. Optional: a RunPod Infinity
endpoint via `RERANK_BASE_URL` if one is healthy.
"""

from __future__ import annotations

import asyncio
import logging
import time
from functools import lru_cache
from typing import Any, List, Sequence

import httpx

from ..config import get_settings

logger = logging.getLogger(__name__)

LOCAL_RERANKER_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


def reranker_enabled() -> bool:
    return get_settings().enable_reranker


def rerank_backend() -> str:
    settings = get_settings()
    if not settings.enable_reranker:
        return "off"
    if settings.rerank_base_url:
        return settings.rerank_base_url
    return f"local-onnx:{LOCAL_RERANKER_MODEL}"


def _endpoint_root(url: str) -> str:
    root = url.strip().rstrip("/")
    for suffix in ("/runsync", "/run", "/openai/v1"):
        if root.endswith(suffix):
            return root[: -len(suffix)]
    return root


def _api_key() -> str:
    settings = get_settings()
    return (settings.rerank_api_key or settings.vllm_api_key or "").strip()


def scores_from_output(output: Any, n_docs: int) -> List[float]:
    """Map an Infinity / RunPod rerank payload to one score per input doc."""
    if isinstance(output, dict) and "output" in output and "results" not in output and "scores" not in output:
        output = output["output"]

    results = None
    if isinstance(output, dict):
        if isinstance(output.get("scores"), list) and len(output["scores"]) == n_docs:
            return [float(score) for score in output["scores"]]
        results = output.get("results") or output.get("data") or output.get("docs")
    elif isinstance(output, list):
        results = output

    if not isinstance(results, list) or not results:
        raise ValueError(f"Unexpected rerank payload: {type(output).__name__}")

    scores = [0.0] * n_docs
    for position, item in enumerate(results):
        if isinstance(item, (int, float)):
            if position < n_docs:
                scores[position] = float(item)
            continue
        if not isinstance(item, dict):
            continue
        index = item.get("index", position)
        try:
            index = int(index)
        except (TypeError, ValueError):
            index = position
        if index < 0 or index >= n_docs:
            continue
        raw = item.get("relevance_score", item.get("score", item.get("relevanceScore")))
        if raw is None:
            continue
        scores[index] = float(raw)
    return scores


@lru_cache(maxsize=1)
def local_cross_encoder():
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(model_name=LOCAL_RERANKER_MODEL)


def warmup_local_reranker() -> None:
    local_cross_encoder()


def _local_scores(query: str, documents: Sequence[str]) -> List[float]:
    return [float(score) for score in local_cross_encoder().rerank(query, list(documents))]


async def _remote_scores(query: str, docs: List[str]) -> List[float]:
    settings = get_settings()
    root = _endpoint_root(settings.rerank_base_url or "")
    if not root:
        raise RuntimeError("RERANK_BASE_URL is not set")

    payload = {
        "input": {
            "model": settings.reranker_model,
            "query": query,
            "docs": docs,
            "return_docs": False,
        }
    }
    headers = {
        "Authorization": f"Bearer {_api_key()}",
        "Content-Type": "application/json",
        "Accept-Encoding": "identity",
    }
    timeout = httpx.Timeout(settings.rerank_timeout_s, connect=30.0)
    deadline = time.monotonic() + settings.rerank_timeout_s

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(f"{root}/runsync", json=payload, headers=headers)
        response.raise_for_status()
        body = response.json()
        status = str(body.get("status") or "")
        job_id = body.get("id")
        output = body.get("output")

        while status in {"IN_QUEUE", "IN_PROGRESS", ""} and job_id:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Rerank job {job_id} timed out ({status})")
            await asyncio.sleep(min(2.0, remaining))
            polled = await client.get(f"{root}/status/{job_id}", headers=headers)
            polled.raise_for_status()
            body = polled.json()
            status = str(body.get("status") or "")
            output = body.get("output", output)

    if status.upper() not in {"", "COMPLETED"}:
        error = body.get("error") or output or body
        raise RuntimeError(f"Rerank job {status}: {error}")
    if output is None:
        raise RuntimeError(f"Rerank returned no output: {body!r}"[:800])
    return scores_from_output(output, len(docs))


async def rerank_documents(query: str, documents: Sequence[str]) -> List[float]:
    """Return a relevance score per document, same order as `documents`."""
    docs = [doc if isinstance(doc, str) else str(doc or "") for doc in documents]
    if not docs:
        return []

    settings = get_settings()
    if settings.rerank_base_url:
        return await _remote_scores(query, docs)
    return await asyncio.to_thread(_local_scores, query, docs)
