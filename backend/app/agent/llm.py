"""Chat model construction, pointed at vLLM.

Structured output uses `method="json_schema"` rather than the library default.
Both work against vLLM 0.28, but json_schema routes through the OpenAI
`response_format` field, which vLLM honours with its constrained-decoding
backend (xgrammar). Measured 5/5 schema-valid Plan generations that way.

Do NOT reach for `extra_body={"guided_json": ...}`. In vLLM 0.28 that key is
silently ignored — the request succeeds and returns unconstrained prose, which
is far worse than an error because it looks like a model quality problem.
"""

from __future__ import annotations

from typing import Type, TypeVar

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from ..config import get_settings

TModel = TypeVar("TModel", bound=BaseModel)


def chat_model(
    temperature: float = 0.0,
    max_tokens: int | None = None,
    *,
    streaming: bool = True,
) -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(
        base_url=settings.vllm_base_url,
        api_key=settings.vllm_api_key,
        model=settings.vllm_model,
        temperature=temperature,
        max_completion_tokens=max_tokens,
        timeout=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,
        # Default 120s kills the first chunk of a cold RunPod worker.
        stream_chunk_timeout=settings.llm_timeout_s,
        disable_streaming=not streaming,
        # httpx2's gzip/brotli decoder crashes on RunPod responses
        # (`process() takes no keyword arguments`) and langchain reports that
        # as a generic "Connection error."
        default_headers={"Accept-Encoding": "identity"},
    )


def structured_model(schema: Type[TModel], temperature: float = 0.0):
    """A model constrained to emit `schema`.

    Qwen2.5-7B-AWQ is a much weaker instruction follower than the Gemini model
    this replaced, so constrained decoding is load-bearing here rather than a
    nicety: the Gatekeeper, Planner and Auditor all fail hard on invalid JSON.
    """
    return chat_model(temperature=temperature, streaming=False).with_structured_output(
        schema, method="json_schema"
    )
