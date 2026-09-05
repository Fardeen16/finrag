"""Translate LangGraph `astream_events` into the SSE event contract.

The first version of this adapter *predicted* the next node after each update.
After `verify` that guess is often wrong — the router and the adapter can
disagree about whether to replan, run another tool, or synthesise. Each wrong
guess emitted a `node_start synthesizer` the UI never closed, so the trace
showed "Synthesizer attempt 8 / running" while the graph was still on tools.

`astream_events` reports the node LangGraph actually entered, so a start is
only emitted for a real visit.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, AsyncIterator, Dict, Optional

from .agent.graph import get_graph, run_config
from .config import get_settings
from .events import (
    Audit,
    Clarification,
    Error,
    Heartbeat,
    NodeEnd,
    NodeName,
    NodeStart,
    Plan,
    PlanStep,
    RunEnd,
    RunStart,
    Token,
    ToolEnd,
    ToolStart,
    sse,
)

GRAPH_TO_UI = {
    "gatekeeper": NodeName.gatekeeper,
    "planner": NodeName.planner,
    "execute_tool": NodeName.tool_executor,
    "verify": NodeName.auditor,
    "synthesize": NodeName.synthesizer,
}

PING_S = 5.0


def _preview(output: Any, width: int = 240) -> str:
    text = output if isinstance(output, str) else str(output)
    return " ".join(text.split())[:width]


def _node_name(event: Dict[str, Any]) -> Optional[str]:
    """Return a supervisor node name, or None for nested LLM/tool events."""
    name = event.get("name")
    node = (event.get("metadata") or {}).get("langgraph_node")
    if name in GRAPH_TO_UI and node == name:
        return name
    return None


async def live_stream(query: str, conversation_id: Optional[str] = None) -> AsyncIterator[str]:
    """Yield SSE frames for one live supervisor run."""
    settings = get_settings()
    seq = 0
    started = time.perf_counter()
    last_mark = started
    deadline = started + settings.agent_timeout_s
    node_timings_ms: Dict[str, int] = {}
    attempts: Dict[str, int] = {}
    llm_calls = 0
    output_tokens = 0
    streamed_tokens = False
    replan_count = 0
    run_id = f"run_{uuid.uuid4().hex[:12]}"
    conversation_id = conversation_id or f"conv_{uuid.uuid4().hex[:12]}"

    def emit(event) -> str:
        nonlocal seq
        seq += 1
        event.seq = seq
        return sse(event)

    yield emit(RunStart(seq=0, run_id=run_id, conversation_id=conversation_id, query=query))

    iterator = get_graph().astream_events(
        {"original_request": query, "replan_count": 0},
        run_config(),
        version="v2",
    ).__aiter__()

    # wait_for() cancels its awaitable on timeout. Cancelling anext() closes
    # the async generator, so a RunPod call that takes >PING_S would abort the
    # whole graph. Shield the next event and reuse the same task across pings.
    pending: Optional[asyncio.Task] = None

    try:
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                if pending is not None:
                    pending.cancel()
                yield emit(
                    Error(
                        seq=0,
                        code="timeout",
                        message=f"Agent exceeded its {settings.agent_timeout_s:.0f}s ceiling.",
                        retryable=True,
                    )
                )
                return
            if pending is None:
                pending = asyncio.ensure_future(anext(iterator))
            try:
                event = await asyncio.wait_for(
                    asyncio.shield(pending), timeout=min(PING_S, remaining)
                )
                pending = None
            except asyncio.TimeoutError:
                if time.perf_counter() >= deadline:
                    pending.cancel()
                    yield emit(
                        Error(
                            seq=0,
                            code="timeout",
                            message=f"Agent exceeded its {settings.agent_timeout_s:.0f}s ceiling.",
                            retryable=True,
                        )
                    )
                    return
                yield emit(Heartbeat(seq=0))
                continue
            except StopAsyncIteration:
                pending = None
                break

            kind = event.get("event")
            graph_node = _node_name(event)

            if (
                kind == "on_chat_model_stream"
                and (event.get("metadata") or {}).get("langgraph_node") == "synthesize"
            ):
                chunk = (event.get("data") or {}).get("chunk")
                text = getattr(chunk, "content", None) or ""
                if isinstance(text, list):
                    text = "".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in text
                    )
                if text:
                    streamed_tokens = True
                    output_tokens += 1
                    yield emit(Token(seq=0, text=text))
                continue

            if graph_node is None:
                continue

            ui_node = GRAPH_TO_UI[graph_node]

            if kind == "on_chain_start":
                attempts[graph_node] = attempts.get(graph_node, 0) + 1
                last_mark = time.perf_counter()
                yield emit(
                    NodeStart(seq=0, node=ui_node, attempt=attempts[graph_node])
                )
                continue

            if kind != "on_chain_end":
                continue

            now = time.perf_counter()
            duration_ms = int((now - last_mark) * 1000)
            node_timings_ms[ui_node.value] = (
                node_timings_ms.get(ui_node.value, 0) + duration_ms
            )
            payload = (event.get("data") or {}).get("output") or {}
            if not isinstance(payload, dict):
                payload = {}

            yield emit(
                NodeEnd(
                    seq=0,
                    node=ui_node,
                    attempt=attempts.get(graph_node, 1),
                    duration_ms=duration_ms,
                )
            )

            if graph_node == "gatekeeper":
                llm_calls += 1
                question = payload.get("clarification_question")
                if question:
                    yield emit(Clarification(seq=0, question=question))
                scoped = payload.get("final_response")
                if scoped:
                    streamed_tokens = True
                    output_tokens = len(str(scoped).split())
                    yield emit(Token(seq=0, text=str(scoped)))

            elif graph_node == "planner":
                llm_calls += 1
                replan_count = int(payload.get("replan_count") or replan_count)
                steps = [
                    PlanStep(
                        tool_name=s["tool_name"],
                        tool_input=s.get("tool_input") or "",
                    )
                    for s in payload.get("plan") or []
                    if isinstance(s, dict) and s.get("tool_name") != "FINISH"
                ]
                yield emit(
                    Plan(seq=0, attempt=attempts.get(graph_node, 1), steps=steps)
                )

            elif graph_node == "execute_tool":
                steps = payload.get("intermediate_steps") or []
                if steps:
                    last = steps[-1]
                    output = last.get("tool_output")
                    failed = isinstance(output, str) and (
                        output.startswith("Tool raised")
                        or output.startswith("Unknown tool")
                    )
                    yield emit(
                        ToolStart(
                            seq=0,
                            tool_name=last.get("tool_name") or "unknown",
                            tool_input=last.get("tool_input") or "",
                        )
                    )
                    yield emit(
                        ToolEnd(
                            seq=0,
                            tool_name=last.get("tool_name") or "unknown",
                            duration_ms=duration_ms,
                            ok=not failed,
                            preview=_preview(output),
                            result_count=len(output) if isinstance(output, list) else None,
                            error=(str(output)[:200] if failed else None),
                        )
                    )

            elif graph_node == "verify":
                llm_calls += 1
                history = payload.get("verification_history") or []
                if history:
                    audit = history[-1]
                    score = int(audit.get("confidence_score") or 1)
                    action = (
                        "replan" if score < settings.audit_pass_score else "accept"
                    )
                    yield emit(
                        Audit(
                            seq=0,
                            attempt=attempts.get(graph_node, 1),
                            confidence_score=score,
                            is_relevant=bool(audit.get("is_relevant")),
                            reasoning=str(audit.get("reasoning") or ""),
                            action=action,
                        )
                    )

            elif graph_node == "synthesize":
                llm_calls += 1
                if not streamed_tokens:
                    answer = payload.get("final_response") or ""
                    output_tokens = len(str(answer).split())
                    if answer:
                        yield emit(Token(seq=0, text=str(answer)))

    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - the client needs the error on the stream
        yield emit(
            Error(seq=0, code=type(exc).__name__, message=str(exc)[:400], retryable=True)
        )
        return

    yield emit(
        RunEnd(
            seq=0,
            run_id=run_id,
            duration_ms=int((time.perf_counter() - started) * 1000),
            node_timings_ms=node_timings_ms,
            llm_calls=llm_calls,
            output_tokens=output_tokens,
            replans=replan_count,
        )
    )
