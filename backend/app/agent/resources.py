"""Lazily constructed shared resources.

The original `agent_tools.py` opened Qdrant, loaded a CrossEncoder and called
`setup_data_stores()` — which calls `exit()` on missing files — at module import
time. Importing it from a web server would kill the worker on a bad path and
hold the embedded Qdrant lock for the process lifetime. Everything here is
built on first use instead, and raises rather than exiting.
"""

from __future__ import annotations

from functools import lru_cache

import qdrant_client
from qdrant_client.models import Distance, VectorParams

from ..config import get_settings


@lru_cache(maxsize=1)
def get_qdrant() -> qdrant_client.QdrantClient:
    settings = get_settings()
    if settings.qdrant_url:
        return qdrant_client.QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            check_compatibility=False,
        )
    if not settings.qdrant_path.exists():
        raise RuntimeError(
            f"No Qdrant store at {settings.qdrant_path}. Run the ingestion scripts "
            f"or set QDRANT_URL to a server."
        )
    return qdrant_client.QdrantClient(path=str(settings.qdrant_path))


def collection_dimension() -> int:
    settings = get_settings()
    info = get_qdrant().get_collection(settings.collection_name)
    params = info.config.params
    vectors = params.vectors
    if isinstance(vectors, VectorParams):
        return int(vectors.size)
    if isinstance(vectors, dict):  # named vectors
        return int(next(iter(vectors.values())).size)
    raise RuntimeError(f"Cannot determine vector size for '{settings.collection_name}'")


def collection_distance() -> Distance | None:
    settings = get_settings()
    vectors = get_qdrant().get_collection(settings.collection_name).config.params.vectors
    if isinstance(vectors, VectorParams):
        return vectors.distance
    return None


@lru_cache(maxsize=1)
def get_sql_database():
    import sqlite3

    import pandas as pd
    from langchain_community.utilities import SQLDatabase

    from ..config import PROJECT_ROOT

    settings = get_settings()
    if not settings.sqlite_path.exists():
        csv_path = PROJECT_ROOT / "data" / "alphabet_financials_structured.csv"
        if not csv_path.exists():
            raise RuntimeError(
                f"No SQLite database at {settings.sqlite_path} and no CSV at {csv_path}"
            )
        settings.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.read_csv(csv_path)
        with sqlite3.connect(settings.sqlite_path) as conn:
            frame.to_sql(settings.table_name, conn, if_exists="replace", index=False)

    return SQLDatabase.from_uri(
        f"sqlite:///{settings.sqlite_path}",
        include_tables=[settings.table_name],
    )
