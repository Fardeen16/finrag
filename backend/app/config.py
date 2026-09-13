"""Environment-driven configuration.

Replaces the hardcoded API key in `agent_tools.py`. Nothing in the new code
path reads a literal credential.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Inference -----------------------------------------------------------
    # Points at local vLLM in development and at the RunPod endpoint in
    # deployment. Nothing else changes between the two.
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_api_key: str = "EMPTY"  # vLLM ignores it, but the client requires one.
    vllm_model: str = "Qwen/Qwen2.5-7B-Instruct-AWQ"
    # First token after a RunPod cold start is often 2–3 minutes. The LangChain
    # stream-chunk default of 120s aborts that wait and the 300s agent ceiling
    # then fires while the Planner is still on its first call.
    llm_timeout_s: float = 240.0
    # The openai client retries with backoff by default, which turns one dead
    # request into minutes of silence. Keep it low and let the agent's own
    # wall-clock ceiling be the real backstop.
    llm_max_retries: int = 1

    # --- Embeddings ----------------------------------------------------------
    # 'google' matches the vectors already in Qdrant (text-embedding-004, 768d).
    # 'local' is where this is going, but flipping it requires a full re-index —
    # vectors from different models are not comparable, so a mismatched
    # collection returns plausible-looking nonsense rather than an error.
    embedding_provider: Literal["google", "local"] = "local"
    google_api_key: Optional[str] = None
    local_embedding_model: str = "BAAI/bge-small-en-v1.5"

    # --- Stores --------------------------------------------------------------
    # Embedded (path) mode is single-process and lock-protected: fine for a CLI,
    # unusable behind a multi-worker server or on ephemeral container storage.
    # Set qdrant_url to move to server mode.
    qdrant_url: Optional[str] = None
    qdrant_api_key: Optional[str] = None
    qdrant_path: Path = PROJECT_ROOT / "data" / "qdrant_db"
    collection_name: str = "financial_docs_alphabet"

    sqlite_path: Path = PROJECT_ROOT / "data" / "financials.db"
    table_name: str = "financials_summary"

    @field_validator("qdrant_url", "qdrant_api_key", mode="before")
    @classmethod
    def _strip_secrets(cls, value: object) -> object:
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value

    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # CrossEncoder needs PyTorch. Off by default so Librarian fits in 512 MB.
    enable_reranker: bool = False
    retrieval_candidates: int = 20
    retrieval_top_k: int = 5

    # Tenant tag written into every Qdrant payload. Shared reference filings are
    # owned by this pseudo-tenant; per-user uploads will carry their own user_id.
    # Present from the first index so enabling isolation never needs a re-index.
    public_tenant: str = "public"

    # --- Agent ceilings ------------------------------------------------------
    # Auditor-driven replans, hard capped. Without this the router replans
    # forever whenever the model cannot satisfy the Auditor.
    max_replans: int = 2
    audit_pass_score: int = 3
    # Plans longer than this are truncated, which is what makes the superstep
    # budget below a bounded quantity rather than a guess.
    max_plan_steps: int = 3

    # recursion_limit is a superstep budget, not a loop guard — max_replans is
    # the loop guard. Worst case is:
    #   gatekeeper + (max_replans + 1) * (planner + max_plan_steps * 2) + synthesize
    # which is 1 + 3 * (1 + 10) + 1 = 35 at the defaults. 50 leaves headroom.
    # Setting it near the worst case is what produced GraphRecursionError before.
    recursion_limit: int = 50
    agent_timeout_s: float = 600.0

    # 'mock' replays canned traces so the UI can be developed without a GPU.
    # 'live' runs the LangGraph supervisor against vLLM + Qdrant.
    agent_mode: Literal["mock", "live"] = "mock"
    # FastEmbed is small enough to load at boot on 512 MB.
    warmup_on_start: bool = True

    def require_google_key(self) -> str:
        if not self.google_api_key:
            raise RuntimeError(
                "embedding_provider='google' needs GOOGLE_API_KEY set. Either export it "
                "or switch EMBEDDING_PROVIDER=local (which requires re-indexing Qdrant)."
            )
        return self.google_api_key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
