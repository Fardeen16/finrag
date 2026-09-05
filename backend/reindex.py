"""Rebuild the Qdrant collection with the configured embedding provider.

    python backend/reindex.py --dry-run      # report what would change
    python backend/reindex.py                # rebuild
    python backend/reindex.py --query "..."  # rebuild then sanity-check retrieval

Why this exists: the original index holds 768-dim text-embedding-004 vectors, so
moving to a local model is not a config flip. Vectors from a different model are
not comparable to those, and Qdrant will not complain — it returns nearest
neighbours in the wrong space, so retrieval silently degrades to noise.

Replaces the Qdrant half of `populate_db.py`, which had an API key hardcoded at
line 27 and embedded via the Gemini batch endpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from qdrant_client.models import Distance, PointStruct, VectorParams

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.agent.embeddings import get_embeddings, embedding_dimension  # noqa: E402
from backend.app.agent.resources import get_qdrant  # noqa: E402
from backend.app.config import PROJECT_ROOT, get_settings  # noqa: E402

CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks_v2.json"
BATCH_SIZE = 32


def embedding_text(chunk: Dict[str, Any]) -> str:
    """Text actually sent to the embedding model.

    Keeps `populate_db.py`'s summary + keywords + content recipe but adds the
    hypothetical questions, which the enrichment step already generated and the
    old pipeline then ignored. They tend to phrase the chunk the way a user
    would ask for it, which is exactly what helps a query match.
    """
    parts = []
    if summary := chunk.get("summary"):
        parts.append(f"Summary: {summary}")
    if keywords := chunk.get("keywords"):
        joined = ", ".join(keywords) if isinstance(keywords, list) else str(keywords)
        parts.append(f"Keywords: {joined}")
    if questions := chunk.get("hypothetical_questions"):
        joined = " ".join(questions) if isinstance(questions, list) else str(questions)
        parts.append(f"Answers questions like: {joined}")
    if table_summary := chunk.get("table_summary"):
        parts.append(f"Table: {table_summary}")
    parts.append(f"Content: {str(chunk.get('content', ''))[:1500]}")
    return "\n".join(parts)


def load_chunks() -> List[Dict[str, Any]]:
    if not CHUNKS_PATH.exists():
        raise SystemExit(f"No enriched chunks at {CHUNKS_PATH}. Run the enrichment step first.")
    with CHUNKS_PATH.open(encoding="utf-8") as handle:
        chunks = json.load(handle)
    if not isinstance(chunks, list) or not chunks:
        raise SystemExit(f"{CHUNKS_PATH} did not contain a non-empty list.")
    return chunks


def describe_existing(client, collection: str) -> str:
    try:
        info = client.get_collection(collection)
        vectors = info.config.params.vectors
        size = vectors.size if isinstance(vectors, VectorParams) else "?"
        return f"{info.points_count} points, {size}-dim"
    except Exception:
        return "does not exist"


def reset_collection(client, dimension: int):
    """Drop and recreate the collection at `dimension`, returning a usable client.

    In embedded (path) mode, delete_collection followed by create_collection does
    not reliably resize the on-disk vector array — upserting then fails with
    "could not broadcast input array from shape (384,) into shape (768,)". The
    reliable reset there is to close the client and remove the storage directory.
    Safe to do because the store is fully rebuildable from chunks_v2.json.
    """
    settings = get_settings()
    vectors_config = VectorParams(size=dimension, distance=Distance.COSINE)

    if settings.qdrant_url:
        if client.collection_exists(settings.collection_name):
            client.delete_collection(settings.collection_name)
        client.create_collection(settings.collection_name, vectors_config=vectors_config)
        return client

    import shutil

    client.close()
    get_qdrant.cache_clear()
    if settings.qdrant_path.exists():
        shutil.rmtree(settings.qdrant_path)
        print(f"  wiped {settings.qdrant_path}")
    settings.qdrant_path.mkdir(parents=True, exist_ok=True)

    fresh = get_qdrant()
    fresh.create_collection(settings.collection_name, vectors_config=vectors_config)

    actual = fresh.get_collection(settings.collection_name).config.params.vectors.size
    if actual != dimension:
        raise SystemExit(f"Collection created at {actual}-dim, expected {dimension}")
    return fresh


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Report without writing")
    parser.add_argument("--query", help="Run a retrieval check after rebuilding")
    args = parser.parse_args()

    settings = get_settings()
    chunks = load_chunks()
    client = get_qdrant()

    print("Re-index plan")
    print(f"  chunks          {len(chunks)} from {CHUNKS_PATH.name}")
    print(f"  provider        {settings.embedding_provider}")
    if settings.embedding_provider == "local":
        print(f"  model           {settings.local_embedding_model}")
    print(f"  target          {settings.qdrant_url or settings.qdrant_path}")
    print(f"  collection      {settings.collection_name} ({describe_existing(client, settings.collection_name)})")

    print("\nLoading embedding model...")
    embedder = get_embeddings()
    dimension = embedding_dimension()
    print(f"  new vectors     {dimension}-dim")

    if args.dry_run:
        print("\nDry run: nothing written.")
        return 0

    print(f"\nRecreating collection '{settings.collection_name}'...")
    client = reset_collection(client, dimension)

    print(f"Embedding {len(chunks)} chunks...")
    texts = [embedding_text(chunk) for chunk in chunks]
    vectors: List[List[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        vectors.extend(embedder.embed_documents(batch))
        print(f"  {min(start + BATCH_SIZE, len(texts))}/{len(texts)}")

    if len(vectors) != len(chunks):
        raise SystemExit(f"Got {len(vectors)} vectors for {len(chunks)} chunks; aborting.")

    points = [
        PointStruct(
            id=index,
            vector=vector,
            payload={
                "source": chunk.get("source"),
                "content": chunk.get("content"),
                "summary": chunk.get("summary"),
                "keywords": chunk.get("keywords"),
                "is_table": chunk.get("is_table", False),
                "section": chunk.get("section"),
                "enriched": chunk.get("enriched", False),
                # Written now so enabling per-user filtering later is a query
                # change rather than another full re-index.
                "tenant": settings.public_tenant,
            },
        )
        for index, (chunk, vector) in enumerate(zip(chunks, vectors))
    ]

    print("Upserting...")
    client.upsert(collection_name=settings.collection_name, points=points, wait=True)
    count = client.get_collection(settings.collection_name).points_count
    print(f"Done: {count} points, {dimension}-dim, cosine.")

    if args.query:
        print(f"\nRetrieval check: {args.query!r}")
        hits = client.query_points(
            collection_name=settings.collection_name,
            query=embedder.embed_query(args.query),
            limit=3,
            with_payload=True,
        ).points
        for rank, hit in enumerate(hits, start=1):
            snippet = str(hit.payload.get("summary") or hit.payload.get("content"))
            snippet = " ".join(snippet.split())[:150]
            print(f"  {rank}. score={hit.score:.4f}  {snippet}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
