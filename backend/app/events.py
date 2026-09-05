"""Wire format for the agent event stream.

This module is the contract between the LangGraph supervisor and any client.
The frontend mirrors these shapes in `frontend/src/lib/events.ts` — change one,
change the other.

Two rules keep the stream usable:

1. Every event carries `seq`. SSE gives no ordering guarantee across reconnects,
   and a client that buffers out-of-order node transitions will render the graph
   walk wrong. `seq` is monotonic within a run.
2. `type` is duplicated in the JSON payload as well as the SSE `event:` field.
   The redundancy costs a few bytes and makes a raw `curl` transcript
   self-describing.
"""

from __future__ import annotations

import json
import time
from enum import Enum
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field


class NodeName(str, Enum):
    """Nodes in the supervisor graph, matching `agent_tools.py`."""

    gatekeeper = "gatekeeper"
    planner = "planner"
    tool_executor = "tool_executor"
    auditor = "auditor"
    synthesizer = "synthesizer"


class ToolName(str, Enum):
    librarian = "librarian_rag_tool"
    sql_analyst = "analyst_sql_tool"
    trend_analyst = "analyst_trend_tool"
    scout = "scout_web_search_tool"


class EventType(str, Enum):
    run_start = "run_start"
    node_start = "node_start"
    node_end = "node_end"
    plan = "plan"
    tool_start = "tool_start"
    tool_end = "tool_end"
    audit = "audit"
    clarification = "clarification"
    token = "token"
    run_end = "run_end"
    error = "error"
    heartbeat = "heartbeat"


class _Base(BaseModel):
    seq: int
    ts: float = Field(default_factory=time.time)


class RunStart(_Base):
    type: Literal[EventType.run_start] = EventType.run_start
    run_id: str
    conversation_id: str
    query: str


class NodeStart(_Base):
    type: Literal[EventType.node_start] = EventType.node_start
    node: NodeName
    # Incremented each time a node is re-entered, so the UI can show
    # "Planner (attempt 2)" after the Auditor forces a replan.
    attempt: int = 1


class NodeEnd(_Base):
    type: Literal[EventType.node_end] = EventType.node_end
    node: NodeName
    attempt: int = 1
    duration_ms: int


class PlanStep(BaseModel):
    tool_name: str
    tool_input: str


class Plan(_Base):
    """Emitted by the planner so the UI can show intent before any tool runs."""

    type: Literal[EventType.plan] = EventType.plan
    attempt: int = 1
    steps: list[PlanStep]


class ToolStart(_Base):
    type: Literal[EventType.tool_start] = EventType.tool_start
    tool_name: str
    tool_input: str


class ToolEnd(_Base):
    type: Literal[EventType.tool_end] = EventType.tool_end
    tool_name: str
    duration_ms: int
    ok: bool = True
    # Never the full tool output. Retrieval hits can be tens of kilobytes and
    # the browser does not need them to render a trace.
    preview: Optional[str] = None
    result_count: Optional[int] = None
    error: Optional[str] = None


class Audit(_Base):
    type: Literal[EventType.audit] = EventType.audit
    attempt: int = 1
    confidence_score: int = Field(ge=1, le=5)
    is_relevant: bool
    reasoning: str
    # Mirrors the `< 3` threshold in the router.
    action: Literal["accept", "replan"]


class Clarification(_Base):
    """Terminal event when the Gatekeeper judges the request too vague."""

    type: Literal[EventType.clarification] = EventType.clarification
    question: str


class Token(_Base):
    type: Literal[EventType.token] = EventType.token
    text: str


class RunEnd(_Base):
    type: Literal[EventType.run_end] = EventType.run_end
    run_id: str
    duration_ms: int
    # Per-node wall time, keyed by node name. The README calls for per-node
    # latency instrumentation from the start; this is where it surfaces.
    node_timings_ms: dict[str, int] = Field(default_factory=dict)
    llm_calls: int = 0
    output_tokens: int = 0
    replans: int = 0


class Error(_Base):
    type: Literal[EventType.error] = EventType.error
    code: str
    message: str
    retryable: bool = False


class Heartbeat(_Base):
    """Named SSE event so browsers treat the stream as active during long LLM waits.

    SSE comments (`: ping`) are dropped by some clients and do not reset
    fetch timeouts; Chrome then reports `Connection error` around five minutes.
    """

    type: Literal[EventType.heartbeat] = EventType.heartbeat


AgentEvent = Union[
    RunStart,
    NodeStart,
    NodeEnd,
    Plan,
    ToolStart,
    ToolEnd,
    Audit,
    Clarification,
    Token,
    RunEnd,
    Error,
    Heartbeat,
]


def sse(event: AgentEvent) -> str:
    """Serialise one event as an SSE frame.

    `json.dumps` on `model_dump(mode="json")` rather than `model_dump_json()`
    so enum members land as their string values.
    """
    payload: dict[str, Any] = event.model_dump(mode="json")
    return f"event: {payload['type']}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n"


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    conversation_id: Optional[str] = None
