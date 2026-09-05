"""Vector-only retrieval check. Avoids the reranker and LLM rewrite, which
together with the embedded Qdrant client have been crashing this process
(0xC0000005) on Windows.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.agent.embeddings import get_embeddings  # noqa: E402
from backend.app.agent.resources import get_qdrant  # noqa: E402
from backend.app.config import get_settings  # noqa: E402

QUERIES = [
    "What are the main competitive risks Alphabet identifies in AI and cloud?",
    "antitrust lawsuits Google Search advertising",
    "cybersecurity risk management and incidents",
]


def main() -> None:
    settings = get_settings()
    embedder = get_embeddings()
    client = get_qdrant()
    print(f"collection {settings.collection_name}: "
          f"{client.get_collection(settings.collection_name).points_count} points\n")

    for query in QUERIES:
        print(f"Q: {query}")
        hits = client.query_points(
            collection_name=settings.collection_name,
            query=embedder.embed_query(query),
            limit=5,
            with_payload=True,
        ).points
        for rank, hit in enumerate(hits, 1):
            payload = hit.payload or {}
            section = payload.get("section") or "(none)"
            summary = " ".join(str(payload.get("summary") or payload.get("content") or "").split())
            print(f"  {rank}. {hit.score:.4f}  [{section}]  {summary[:140]}")
        print()


if __name__ == "__main__":
    main()
