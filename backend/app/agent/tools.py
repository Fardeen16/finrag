"""The four specialist tools, ported to the local model.

Behaviour matches `agent_tools.py`; only the model behind each call changed.
Print statements are gone — node progress now travels over the event stream.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from typing import Any, Dict, List

import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..config import get_settings
from .embeddings import assert_dimension_matches, get_embeddings
from .llm import chat_model, structured_model
from .rerank import rerank_documents, reranker_enabled
from .resources import collection_dimension, get_qdrant, get_sql_database

logger = logging.getLogger(__name__)

_QUERY_REWRITE = (
    'Rewrite the following query for a semantic search system. Respond with ONLY '
    'the rewritten query, no preamble.\nQuery: "{query}"\nRewritten:'
)


async def optimize_query(query: str) -> str:
    """Rewrite a user query for retrieval.

    Falls back to the original query on failure: a bad rewrite should degrade
    retrieval, not fail the request.
    """
    try:
        response = await chat_model(temperature=0.0, max_tokens=128).ainvoke(
            _QUERY_REWRITE.format(query=query)
        )
        rewritten = (response.content or "").strip().strip('"')
        return rewritten or query
    except Exception:
        return query


@tool
async def librarian_rag_tool(query: str) -> List[Dict[str, Any]]:
    """Retrieves deep, contextual information from Alphabet's financial filings."""
    settings = get_settings()
    await asyncio.to_thread(get_embeddings)
    await asyncio.to_thread(lambda: assert_dimension_matches(collection_dimension()))
    embedding = await asyncio.to_thread(get_embeddings().embed_query, query)

    use_rerank = reranker_enabled()
    limit = settings.retrieval_candidates if use_rerank else settings.retrieval_top_k
    hits = await asyncio.to_thread(
        lambda: get_qdrant().query_points(
            collection_name=settings.collection_name,
            query=embedding,
            limit=limit,
            with_payload=True,
        ).points
    )
    if not hits:
        return []

    scored: List[tuple[Any, float | None]]
    if use_rerank:
        docs = [
            str((hit.payload or {}).get("content") or (hit.payload or {}).get("summary") or "")
            for hit in hits
        ]
        try:
            scores = await rerank_documents(query, docs)
            scored = sorted(zip(hits, scores), key=lambda item: item[1], reverse=True)
        except Exception:
            logger.exception("Remote rerank failed; using cosine order")
            scored = [
                (hit, float(hit.score) if hit.score is not None else None)
                for hit in hits
            ]
    else:
        scored = [
            (hit, float(hit.score) if hit.score is not None else None)
            for hit in hits
        ]

    return [
        {
            "source": (hit.payload or {}).get("source"),
            "content": (hit.payload or {}).get("content"),
            "summary": (hit.payload or {}).get("summary"),
            "rerank_score": score,
        }
        for hit, score in scored[: settings.retrieval_top_k]
    ]


class _SqlQuery(BaseModel):
    sql: str = Field(description="A single read-only SQLite SELECT statement.")


_SQL_PROMPT = """Write one SQLite SELECT query answering the question.

Schema:
{schema}

Rules:
- SELECT only. No INSERT, UPDATE, DELETE, DROP or PRAGMA.
- Use only the columns shown above.
- SQLite syntax: no YEAR(), no TOP. Use LIMIT.
- Return the query only.

Question: {question}
"""

# Only a bare SELECT/WITH is allowed through. The question text reaches the model
# that writes this SQL, so the generated statement is untrusted input: a prompt
# injection that talks the model into a DROP must not reach the database.
_FORBIDDEN_SQL = (
    "insert", "update", "delete", "drop", "alter", "create",
    "attach", "detach", "pragma", "replace", "vacuum",
)


def _run_select(sql: str) -> str:
    settings = get_settings()
    stripped = sql.strip().rstrip(";").strip()
    lowered = stripped.lower()

    if not (lowered.startswith("select") or lowered.startswith("with")):
        return f"Refused: only SELECT queries are allowed, got: {stripped[:120]}"
    if any(word in lowered.split() for word in _FORBIDDEN_SQL):
        return f"Refused: statement contains a write operation: {stripped[:120]}"
    if ";" in stripped:
        return "Refused: multiple statements are not allowed."

    with sqlite3.connect(f"file:{settings.sqlite_path}?mode=ro", uri=True) as conn:
        df = pd.read_sql_query(stripped, conn)

    if df.empty:
        return f"Query returned no rows.\nSQL: {stripped}"
    return f"SQL: {stripped}\n\n{df.to_string(index=False, max_rows=40)}"


@tool
async def analyst_sql_tool(query: str) -> str:
    """Answers specific questions about revenue, net income, etc., from a SQL database."""
    # Replaces langchain's `create_sql_agent`, which was an agent loop costing
    # several LLM round trips and was the slowest node in the graph. One
    # constrained-decoding call plus a direct execution is faster, and it drops a
    # dependency on the legacy langchain-community agent stack.
    try:
        schema = await asyncio.to_thread(get_sql_database().get_table_info)
    except Exception as exc:
        return f"Could not read database schema: {type(exc).__name__}: {exc}"

    try:
        generated = await structured_model(_SqlQuery).ainvoke(
            _SQL_PROMPT.format(schema=schema, question=query)
        )
    except Exception as exc:
        return f"Could not generate SQL: {type(exc).__name__}: {exc}"

    try:
        return await asyncio.to_thread(_run_select, generated.sql)
    except Exception as exc:
        return f"SQL failed: {type(exc).__name__}: {exc}\nSQL: {generated.sql}"


def _trend_sync(_: str) -> str:
    settings = get_settings()
    with sqlite3.connect(settings.sqlite_path) as conn:
        df = pd.read_sql_query(
            f"SELECT * FROM {settings.table_name} ORDER BY year, quarter", conn
        )
    if df.empty:
        return "No rows in the financial summary table."

    df["period"] = df["year"].astype(str) + "-" + df["quarter"].astype(str)
    df = df.set_index("period")
    metric = "revenue_usd_billions"

    qoq = df[metric].pct_change()
    yoy = df[metric].pct_change(4)
    return (
        f"Analysis of {metric} from {df.index[0]} to {df.index[-1]}: "
        f"${df[metric].iloc[0]:.1f}B -> ${df[metric].iloc[-1]:.1f}B. "
        f"Latest QoQ {qoq.iloc[-1]:.1%}, latest YoY {yoy.iloc[-1]:.1%}."
    )


@tool
async def analyst_trend_tool(query: str) -> str:
    """Analyzes financial trends like QoQ or YoY growth from the SQL database."""
    return await asyncio.to_thread(_trend_sync, query)


@tool
async def scout_web_search_tool(query: str) -> str:
    """Searches the web for real-time information like stock prices or recent news."""
    from langchain_community.tools import DuckDuckGoSearchRun

    try:
        return await DuckDuckGoSearchRun().arun(query)
    except Exception as exc:
        return f"Web search failed: {type(exc).__name__}: {exc}"


ALL_TOOLS = [
    librarian_rag_tool,
    analyst_sql_tool,
    analyst_trend_tool,
    scout_web_search_tool,
]
TOOL_MAP = {t.name: t for t in ALL_TOOLS}
