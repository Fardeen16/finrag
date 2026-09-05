"""Canned replays of a supervisor run, for building the frontend without a GPU.

Timings default to the real thing rather than something snappy. A live run is
15-30s because one query fans out into Gatekeeper -> Planner -> tools ->
Auditor -> Synthesizer, and a UI that feels fine at 2s can be unusable at 25s.
Pass `speed` to compress the replay once the layout is settled.

Replaced wholesale by `astream_events` over the real graph later; the event
shapes it emits are the ones in `events.py`.
"""

from __future__ import annotations

import asyncio
import re
import uuid
from collections import defaultdict
from typing import AsyncIterator

from .events import (
    Audit,
    Clarification,
    Error,
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

# Wall times observed per node, in seconds. The SQL analyst is slow because
# `create_sql_agent` is itself an agent loop: schema introspection, query
# drafting, then execution, so several LLM round trips.
GATEKEEPER_S = 1.2
PLANNER_S = 2.6
AUDITOR_S = 2.1
TOOL_S = {
    "librarian_rag_tool": 2.4,  # query rewrite + embed + Qdrant + CrossEncoder
    "analyst_sql_tool": 4.3,
    "analyst_trend_tool": 0.4,  # pure pandas, no LLM
    "scout_web_search_tool": 2.7,
}

# 40 tok/s single-stream, per the benchmark table in the README.
SECONDS_PER_TOKEN = 1 / 40

ACTIONABLE = re.compile(
    r"\b(revenue|income|profit|margin|trend|growth|risk|competiti\w*|segment|"
    r"cloud|capex|10-?k|filing|quarter|q[1-4]|fy|yoy|qoq|year|compare|"
    r"guidance|operating|expense|headcount|buyback|dividend)\b",
    re.I,
)
QUANTITATIVE = re.compile(r"\b(revenue|income|margin|trend|growth|yoy|qoq|quarter|q[1-4]|capex|expense)\b", re.I)
QUALITATIVE = re.compile(r"\b(risk|competiti\w*|10-?k|filing|strategy|regulat\w*|moat|threat)\b", re.I)


class _Run:
    """Sequence numbering, timing accumulation and speed scaling for one run."""

    def __init__(self, query: str, conversation_id: str, speed: float) -> None:
        self.query = query
        self.conversation_id = conversation_id
        self.run_id = f"run_{uuid.uuid4().hex[:12]}"
        self.speed = max(speed, 0.05)
        self._seq = 0
        # Reported latencies are nominal, not measured: they describe what the
        # run would cost at speed=1. Measured time would collapse under a speed
        # multiplier and give the UI unrealistic numbers to lay out against.
        self.nominal_ms = 0
        self.node_timings_ms: dict[str, int] = defaultdict(int)
        self.llm_calls = 0
        self.output_tokens = 0
        self.replans = 0

    def seq(self) -> int:
        self._seq += 1
        return self._seq

    async def sleep(self, seconds: float) -> None:
        self.nominal_ms += int(seconds * 1000)
        await asyncio.sleep(seconds / self.speed)

    def node_start(self, node: NodeName, attempt: int = 1) -> str:
        return sse(NodeStart(seq=self.seq(), node=node, attempt=attempt))

    def node_end(self, node: NodeName, seconds: float, attempt: int = 1) -> str:
        ms = int(seconds * 1000)
        self.node_timings_ms[node.value] += ms
        return sse(NodeEnd(seq=self.seq(), node=node, attempt=attempt, duration_ms=ms))


def _pick_scenario(query: str) -> str:
    if not ACTIONABLE.search(query) or len(query.split()) < 4:
        return "clarification"
    if QUANTITATIVE.search(query) and QUALITATIVE.search(query):
        return "replan"
    return "simple"


TOOL_RESULTS = {
    "analyst_sql_tool": (
        "FY2024 total revenue: $350.0B (FY2023: $307.4B).",
        1,
    ),
    "analyst_trend_tool": (
        "revenue_usd_billions 2023-Q1 to 2024-Q4: $69.8B -> $96.5B. "
        "Recent QoQ 5.4%, recent YoY 11.8%.",
        8,
    ),
    "librarian_rag_tool": (
        "Item 1A Risk Factors - intense competition in AI, search and cloud; "
        "rapid model commoditisation; regulatory pressure in the EU.",
        5,
    ),
    "scout_web_search_tool": (
        "GOOGL last close $178.42, +1.2% on the session.",
        3,
    ),
}

ANSWER_REPLAN = (
    "Alphabet's revenue grew from $307.4B in FY2023 to $350.0B in FY2024, a 13.9% increase, "
    "with quarterly revenue rising from $69.8B in Q1 2023 to $96.5B in Q4 2024. Growth has been "
    "steady rather than accelerating: recent QoQ growth of 5.4% against YoY of 11.8% suggests the "
    "annual rate is being sustained by consistent sequential gains, not a single outlier quarter.\n\n"
    "The 10-K's competitive risk disclosures read as a direct commentary on that trajectory. "
    "Management flags intense competition across search, cloud and AI, and specifically the risk "
    "that model capability becomes commoditised. A data-grounded hypothesis: the revenue line has "
    "so far absorbed those pressures because Cloud is compounding off a smaller base while Search "
    "monetisation holds, but the margin story is where competition would show up first. Rising "
    "capital expenditure on AI infrastructure alongside only moderate sequential revenue growth "
    "is the pattern to watch.\n\n"
    "Caveat: this reads growth and risk disclosure side by side. The filing does not attribute "
    "any revenue movement to competitive dynamics, so the causal link is inference, not disclosure."
)

ANSWER_SIMPLE = (
    "Alphabet reported total revenue of $350.0B for FY2024, up from $307.4B in FY2023 — "
    "growth of 13.9% year over year. Quarterly revenue reached $96.5B in Q4 2024, the highest "
    "in the series, continuing an unbroken sequence of sequential increases through the period.\n\n"
    "Source: Alphabet Inc. FY2024 10-K, consolidated statements of income."
)


async def _stream_tokens(run: _Run, text: str) -> AsyncIterator[str]:
    """Emit `text` in word-ish chunks at the measured single-stream rate."""
    chunks = re.findall(r"\S+\s*", text)
    for chunk in chunks:
        await run.sleep(SECONDS_PER_TOKEN)
        run.output_tokens += 1
        yield sse(Token(seq=run.seq(), text=chunk))


async def _tool_call(run: _Run, tool_name: str, tool_input: str) -> AsyncIterator[str]:
    seconds = TOOL_S.get(tool_name, 1.5)
    preview, count = TOOL_RESULTS.get(tool_name, ("(no result)", 0))
    yield sse(ToolStart(seq=run.seq(), tool_name=tool_name, tool_input=tool_input))
    await run.sleep(seconds)
    # Timing is recorded by the caller's `node_end` for tool_executor; adding it
    # here as well would double count.
    yield sse(
        ToolEnd(
            seq=run.seq(),
            tool_name=tool_name,
            duration_ms=int(seconds * 1000),
            ok=True,
            preview=preview,
            result_count=count,
        )
    )


async def replay(query: str, conversation_id: str | None = None, speed: float = 1.0) -> AsyncIterator[str]:
    run = _Run(query, conversation_id or f"conv_{uuid.uuid4().hex[:8]}", speed)
    scenario = _pick_scenario(query)

    try:
        yield sse(
            RunStart(
                seq=run.seq(),
                run_id=run.run_id,
                conversation_id=run.conversation_id,
                query=query,
            )
        )

        # --- Gatekeeper ------------------------------------------------------
        yield run.node_start(NodeName.gatekeeper)
        await run.sleep(GATEKEEPER_S)
        run.llm_calls += 1
        yield run.node_end(NodeName.gatekeeper, GATEKEEPER_S)

        if scenario == "clarification":
            yield sse(
                Clarification(
                    seq=run.seq(),
                    question=(
                        "That's a little broad for me to answer well. Are you asking about "
                        "revenue and margin trends, the risk factors in the latest 10-K, or "
                        "current market performance?"
                    ),
                )
            )
            yield sse(
                RunEnd(
                    seq=run.seq(),
                    run_id=run.run_id,
                    duration_ms=run.nominal_ms,
                    node_timings_ms=dict(run.node_timings_ms),
                    llm_calls=run.llm_calls,
                    output_tokens=run.output_tokens,
                    replans=run.replans,
                )
            )
            return

        if scenario == "replan":
            # Attempt 1: the Planner reaches for SQL alone, which cannot speak to
            # the qualitative half of the question. The Auditor catches it.
            yield run.node_start(NodeName.planner, attempt=1)
            await run.sleep(PLANNER_S)
            run.llm_calls += 1
            yield sse(
                Plan(
                    seq=run.seq(),
                    attempt=1,
                    steps=[
                        PlanStep(tool_name="analyst_sql_tool", tool_input="total revenue FY2023 and FY2024"),
                        PlanStep(tool_name="FINISH", tool_input=""),
                    ],
                )
            )
            yield run.node_end(NodeName.planner, PLANNER_S, attempt=1)

            yield run.node_start(NodeName.tool_executor, attempt=1)
            async for frame in _tool_call(run, "analyst_sql_tool", "total revenue FY2023 and FY2024"):
                yield frame
            yield run.node_end(NodeName.tool_executor, TOOL_S["analyst_sql_tool"], attempt=1)

            yield run.node_start(NodeName.auditor, attempt=1)
            await run.sleep(AUDITOR_S)
            run.llm_calls += 1
            yield sse(
                Audit(
                    seq=run.seq(),
                    attempt=1,
                    confidence_score=2,
                    is_relevant=True,
                    reasoning=(
                        "The revenue figures are accurate but address only half the request. "
                        "The user also asked about competitive risks in the 10-K, and no "
                        "retrieval over the filing was performed."
                    ),
                    action="replan",
                )
            )
            yield run.node_end(NodeName.auditor, AUDITOR_S, attempt=1)
            run.replans += 1

            # Attempt 2: replan with the Auditor's reasoning fed back in.
            yield run.node_start(NodeName.planner, attempt=2)
            await run.sleep(PLANNER_S)
            run.llm_calls += 1
            yield sse(
                Plan(
                    seq=run.seq(),
                    attempt=2,
                    steps=[
                        PlanStep(tool_name="analyst_trend_tool", tool_input="revenue trend last two years"),
                        PlanStep(tool_name="librarian_rag_tool", tool_input="competitive risks in AI and cloud"),
                        PlanStep(tool_name="FINISH", tool_input=""),
                    ],
                )
            )
            yield run.node_end(NodeName.planner, PLANNER_S, attempt=2)

            yield run.node_start(NodeName.tool_executor, attempt=2)
            for name, tool_input in (
                ("analyst_trend_tool", "revenue trend last two years"),
                ("librarian_rag_tool", "competitive risks in AI and cloud"),
            ):
                async for frame in _tool_call(run, name, tool_input):
                    yield frame
            yield run.node_end(
                NodeName.tool_executor,
                TOOL_S["analyst_trend_tool"] + TOOL_S["librarian_rag_tool"],
                attempt=2,
            )

            yield run.node_start(NodeName.auditor, attempt=2)
            await run.sleep(AUDITOR_S)
            run.llm_calls += 1
            yield sse(
                Audit(
                    seq=run.seq(),
                    attempt=2,
                    confidence_score=4,
                    is_relevant=True,
                    reasoning=(
                        "Both halves of the request are now covered: quantitative trend data "
                        "and the relevant risk-factor passages. Sufficient to synthesise."
                    ),
                    action="accept",
                )
            )
            yield run.node_end(NodeName.auditor, AUDITOR_S, attempt=2)
            answer = ANSWER_REPLAN
        else:
            yield run.node_start(NodeName.planner)
            await run.sleep(PLANNER_S)
            run.llm_calls += 1
            yield sse(
                Plan(
                    seq=run.seq(),
                    steps=[
                        PlanStep(tool_name="analyst_sql_tool", tool_input=query),
                        PlanStep(tool_name="FINISH", tool_input=""),
                    ],
                )
            )
            yield run.node_end(NodeName.planner, PLANNER_S)

            yield run.node_start(NodeName.tool_executor)
            async for frame in _tool_call(run, "analyst_sql_tool", query):
                yield frame
            yield run.node_end(NodeName.tool_executor, TOOL_S["analyst_sql_tool"])

            yield run.node_start(NodeName.auditor)
            await run.sleep(AUDITOR_S)
            run.llm_calls += 1
            yield sse(
                Audit(
                    seq=run.seq(),
                    confidence_score=4,
                    is_relevant=True,
                    reasoning="Tool output directly answers the request with figures from the filing.",
                    action="accept",
                )
            )
            yield run.node_end(NodeName.auditor, AUDITOR_S)
            answer = ANSWER_SIMPLE

        # --- Synthesizer -----------------------------------------------------
        yield run.node_start(NodeName.synthesizer)
        run.llm_calls += 1
        tokens_before = run.output_tokens
        async for frame in _stream_tokens(run, answer):
            yield frame
        yield run.node_end(
            NodeName.synthesizer,
            (run.output_tokens - tokens_before) * SECONDS_PER_TOKEN,
        )

        yield sse(
            RunEnd(
                seq=run.seq(),
                run_id=run.run_id,
                duration_ms=run.nominal_ms,
                node_timings_ms=dict(run.node_timings_ms),
                llm_calls=run.llm_calls,
                output_tokens=run.output_tokens,
                replans=run.replans,
            )
        )

    except asyncio.CancelledError:
        # Client hung up. Nothing to clean up in the mock, but the real graph
        # will need to cancel in-flight vLLM requests here.
        raise
    except Exception as exc:  # pragma: no cover - defensive
        yield sse(
            Error(
                seq=run.seq(),
                code="mock_failure",
                message=str(exc),
                retryable=True,
            )
        )
