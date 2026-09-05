"""Is the CrossEncoder rerank helping or hurting?

The agent's Auditor rejected the Librarian's 10-K output as irrelevant while a
direct vector query on the same topic returned excellent hits. That points at
something between the two: the LLM query rewrite, or the rerank.

Prints raw vector order against reranked order so the two are directly
comparable. Relevant to the README's unverified "30% Top-K precision
improvement" claim.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.agent.embeddings import get_embeddings  # noqa: E402
from backend.app.agent.resources import get_qdrant, get_reranker  # noqa: E402
from backend.app.agent.tools import optimize_query  # noqa: E402
from backend.app.config import get_settings  # noqa: E402

QUERY = "What are the main competitive risks Alphabet identifies in AI and cloud?"


def snippet(payload, width=110):
    text = payload.get("summary") or payload.get("content") or ""
    section = payload.get("section") or "?"
    return f"[{section}] " + " ".join(str(text).split())[:width]


async def main():
    settings = get_settings()
    embedder = get_embeddings()
    client = get_qdrant()

    rewritten = await optimize_query(QUERY)
    print(f"original:  {QUERY}")
    print(f"rewritten: {rewritten}\n")

    for label, query_text in (("ORIGINAL", QUERY), ("REWRITTEN", rewritten)):
        hits = client.query_points(
            collection_name=settings.collection_name,
            query=embedder.embed_query(query_text),
            limit=settings.retrieval_candidates,
            with_payload=True,
        ).points

        print(f"=== {label} ===")
        print("  vector top 5:")
        for rank, hit in enumerate(hits[:5], 1):
            print(f"    {rank}. {hit.score:.4f}  {snippet(hit.payload)}")

        pairs = [[query_text, h.payload.get("content", "")] for h in hits]
        scores = get_reranker().predict(pairs)
        ranked = sorted(zip(hits, scores), key=lambda p: p[1], reverse=True)

        print("  reranked top 5:")
        for rank, (hit, score) in enumerate(ranked[:5], 1):
            moved = hits.index(hit) + 1
            print(f"    {rank}. {score:+.3f}  (was #{moved})  {snippet(hit.payload)}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
