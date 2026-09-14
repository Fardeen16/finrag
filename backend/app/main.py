"""FastAPI surface for the agent.

`AGENT_MODE=mock` (default) replays canned traces so the UI can be developed
without a GPU. `AGENT_MODE=live` runs the LangGraph supervisor. The event
contract is the same either way; only the generator behind StreamingResponse
changes.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import get_settings
from .events import ChatRequest
from .mock_agent import replay
from .stream import live_stream
from .agent.rerank import rerank_backend

SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _warmup() -> None:
    """Load embedding + reranker weights and open stores before the first query.

    The first Librarian call otherwise spends ~15s downloading/loading models,
    which on Fargate is paid by whoever happens to ask first.
    """
    from .agent.embeddings import get_embeddings
    from .agent.resources import get_qdrant, get_sql_database

    get_embeddings()
    get_qdrant()
    get_sql_database()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings = get_settings()
    if settings.agent_mode == "live" and settings.warmup_on_start:
        await asyncio.to_thread(_warmup)
    yield


app = FastAPI(title="FinRAG API", version="0.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health() -> dict[str, object]:
    settings = get_settings()
    return {
        "status": "ok",
        "agent": settings.agent_mode,
        "vllm": settings.vllm_base_url,
        "embeddings": settings.embedding_provider,
        "qdrant": str(settings.qdrant_url or settings.qdrant_path),
        "rerank": rerank_backend(),
    }


@app.post("/api/chat/stream")
async def chat_stream(
    body: ChatRequest,
    speed: float = Query(1.0, gt=0, le=50, description="Mock replay speed. Ignored in live mode."),
) -> StreamingResponse:
    """Stream agent state and answer tokens as Server-Sent Events.

    POST rather than GET because the query goes in the body and, once auth
    lands, the JWT goes in an Authorization header. That rules out `EventSource`
    on the client — it is GET-only and cannot set headers.
    """
    settings = get_settings()

    async def stream() -> AsyncIterator[bytes]:
        frames = (
            live_stream(body.query, body.conversation_id)
            if settings.agent_mode == "live"
            else replay(body.query, body.conversation_id, speed)
        )
        async for frame in frames:
            yield frame.encode("utf-8")

    return StreamingResponse(stream(), media_type="text/event-stream", headers=SSE_HEADERS)


_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if _dist.is_dir():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="spa")
