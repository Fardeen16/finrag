"""The LangGraph supervisor: Gatekeeper -> Planner -> Tools -> Auditor -> Synthesizer.

Ported from `agent_tools.py` with two bug fixes and one prompt change.

Bug 1 — the original router cleared failed work with `state['intermediate_steps'] = []`
inside a conditional-edge function. Conditional edges only choose a route; their
mutations are discarded, so rejected tool output stayed in state and reached the
Synthesizer. Clearing now happens in the Planner, which can actually write state.

Bug 2 — nothing capped Auditor-driven replanning. If the model could not satisfy
the Auditor, planner -> tools -> auditor cycled until LangGraph raised
GraphRecursionError. `replan_count` is now checked against `max_replans`, and on
exhaustion the graph synthesises from what it has instead of looping.

Prompt change — the Gatekeeper prompt that worked on Gemini makes Qwen2.5-7B far
too eager to ask for clarification; it rejected "What was Alphabet revenue in
FY2024?" as ambiguous. It now gets worked examples and a default-to-specific rule.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from ..config import get_settings
from .llm import chat_model, structured_model
from .tools import ALL_TOOLS, TOOL_MAP


class AgentState(TypedDict, total=False):
    original_request: str
    clarification_question: Optional[str]
    plan: List[Dict[str, Any]]
    intermediate_steps: List[Dict[str, Any]]
    verification_history: List[Dict[str, Any]]
    replan_count: int
    final_response: str


# --- Structured output schemas ----------------------------------------------


class GatekeeperDecision(BaseModel):
    is_specific: bool = Field(description="True if the request is actionable as written.")
    clarification_question: Optional[str] = Field(
        default=None, description="Question to ask the user. Null when is_specific is true."
    )


class PlanStep(BaseModel):
    tool_name: str = Field(description="Exact tool name, or FINISH for the final step.")
    tool_input: str = Field(description="Complete input string for the tool.")


class Plan(BaseModel):
    steps: List[PlanStep] = Field(description="Ordered tool calls. The last must be FINISH.")


class VerificationResult(BaseModel):
    confidence_score: int = Field(ge=1, le=5, description="1-5 confidence in the tool output.")
    is_relevant: bool
    reasoning: str


# Demo corpus is the FY2024 Alphabet 10-K only. Other years are refused
# before any LLM call so they do not burn GPU or invent numbers from the
# leftover 2022–2023 SQL rows.
DEMO_SCOPE_YEAR = 2024
DEMO_SCOPE_MESSAGE = (
    "This is a demo project with limited data. Only Alphabet's FY2024 10-K "
    "is in scope, so I cannot answer questions about other years. Ask about "
    "2024, for example: \"What was total revenue in FY2024?\""
)
_YEAR_RE = re.compile(r"\b(?:fy\s*)?(20\d{2})\b", re.IGNORECASE)
_MULTI_YEAR_RE = re.compile(
    r"\b(?:last|past|previous|prior)\s+"
    r"(?:\d+|two|three|four|five)\s+years?\b"
    r"|\bover the years\b",
    re.IGNORECASE,
)


def demo_scope_refusal(request: str) -> Optional[str]:
    """Return the demo refusal if the request names a year other than 2024."""
    years = {int(match.group(1)) for match in _YEAR_RE.finditer(request)}
    if any(year != DEMO_SCOPE_YEAR for year in years) or _MULTI_YEAR_RE.search(request):
        return DEMO_SCOPE_MESSAGE
    return None


_SPECIFIC_RE = re.compile(
    r"revenue|margin|risk|10-?k|income|cloud|search|capex|eps|profit|"
    r"filing|segment|advertising|youtube|net income",
    re.IGNORECASE,
)


def looks_actionable(request: str) -> bool:
    """True when the request already names a metric or topic, so skip the LLM Gatekeeper."""
    return bool(_SPECIFIC_RE.search(request))


# --- Prompts ----------------------------------------------------------------

GATEKEEPER_PROMPT = """You decide whether a financial research request is actionable.

Default to SPECIFIC. Only ask for clarification when the request names no metric,
no topic and no time frame at all.

SPECIFIC examples:
- "What was total revenue in FY2024?" (names a metric and a period)
- "Analyze the revenue trend and the competitive risks in the 10-K" (multi-part is still specific)
- "Compare cloud growth to search" (names segments)

AMBIGUOUS examples:
- "How is the company?" (no metric, no topic, no period)
- "Tell me about it."

Request: "{request}"
"""

AUDITOR_PROMPT = """Audit whether this tool output answers the user's request.

Score 1-5 on whether the output, on its own, is sufficient to answer the request:
  1-2 = misses the request or covers only part of a multi-part question
  3   = adequate
  4-5 = fully covers the request

User request: {request}
Tool called: {tool_name}
Tool output: {tool_output}
"""

SYNTHESIZER_PROMPT = """You are an expert financial analyst. Answer the user's request
using only the context below. Where you infer a causal link, label it explicitly as a
hypothesis rather than stating it as fact. If the context does not support part of the
request, say so.

User request:
{request}

Context:
---
{context}
---

Answer:"""


def _planner_prompt(request: str, feedback: str) -> str:
    descriptions = "\n".join(
        f"- {t.name}: {(t.description or '').strip()}" for t in ALL_TOOLS
    )
    return f"""Create a short plan to answer the request using the tools below.
Use tool names exactly as written. The final step must always be FINISH with an empty input.

Use the fewest tools that can answer. One tool is enough for a single metric or a
single qualitative topic. Never call every tool. Never add a tool "just in case".
At most two tools before FINISH unless the user asked two distinct questions.

Tools:
{descriptions}
- FINISH: signals the plan is complete.

{feedback}
User request: {request}
"""


# --- Nodes ------------------------------------------------------------------


async def gatekeeper_node(state: AgentState) -> Dict[str, Any]:
    request = state["original_request"]
    scoped = demo_scope_refusal(request)
    if scoped:
        return {"clarification_question": None, "final_response": scoped}
    if looks_actionable(request):
        return {"clarification_question": None}
    try:
        decision = await structured_model(GatekeeperDecision).ainvoke(
            GATEKEEPER_PROMPT.format(request=request)
        )
    except Exception:
        # A Gatekeeper failure should not block the request; let the Planner try.
        return {"clarification_question": None}

    if decision.is_specific:
        return {"clarification_question": None}
    return {
        "clarification_question": decision.clarification_question
        or "Could you be more specific about which metric or period you mean?"
    }


async def planner_node(state: AgentState) -> Dict[str, Any]:
    settings = get_settings()
    request = state["original_request"]
    history = state.get("verification_history") or []
    steps = state.get("intermediate_steps") or []

    is_replan = bool(history) and history[-1]["confidence_score"] < settings.audit_pass_score
    feedback = ""
    if is_replan:
        audit = history[-1]
        failed_tool = steps[-1]["tool_name"] if steps else "unknown"
        feedback = (
            "Previous attempt feedback:\n"
            f"The tool '{failed_tool}' was called but the auditor scored it "
            f"{audit['confidence_score']}/5, reasoning: '{audit['reasoning']}'\n"
            "Produce a DIFFERENT plan that covers what was missed. Do not repeat the "
            "same first step.\n"
        )

    plan = await structured_model(Plan).ainvoke(_planner_prompt(request, feedback))
    steps_out = [step.model_dump() for step in plan.steps]

    # Bound the plan so the superstep budget stays a known quantity.
    working = [s for s in steps_out if s["tool_name"] != "FINISH"][: settings.max_plan_steps]
    steps_out = working + [{"tool_name": "FINISH", "tool_input": ""}]

    updates: Dict[str, Any] = {"plan": steps_out}
    if is_replan:
        # Discard rejected work so the Synthesizer never grounds on it, and reset
        # the audit trail so the router judges the new attempt on its own merits.
        updates["intermediate_steps"] = []
        updates["verification_history"] = []
        updates["replan_count"] = state.get("replan_count", 0) + 1
    return updates


async def tool_executor_node(state: AgentState) -> Dict[str, Any]:
    plan = state.get("plan") or []
    if not plan:
        return {}

    step, remaining = plan[0], plan[1:]
    if step["tool_name"] == "FINISH":
        return {"plan": remaining}

    tool = TOOL_MAP.get(step["tool_name"])
    if tool is None:
        # Hallucinated tool name. Record it as a failed step so the Auditor sees
        # the problem and the Planner gets a chance to correct itself.
        output: Any = f"Unknown tool '{step['tool_name']}'. Valid tools: {sorted(TOOL_MAP)}"
    else:
        try:
            output = await tool.ainvoke({"query": step["tool_input"]})
        except Exception as exc:
            output = f"Tool raised {type(exc).__name__}: {exc}"

    completed = {
        "tool_name": step["tool_name"],
        "tool_input": step["tool_input"],
        "tool_output": output,
    }
    return {
        "intermediate_steps": (state.get("intermediate_steps") or []) + [completed],
        "plan": remaining,
    }


async def auditor_node(state: AgentState) -> Dict[str, Any]:
    steps = state.get("intermediate_steps") or []
    if not steps:
        return {}

    last = steps[-1]
    try:
        result = await structured_model(VerificationResult).ainvoke(
            AUDITOR_PROMPT.format(
                request=state["original_request"],
                tool_name=last["tool_name"],
                # Truncated: retrieval output can be tens of KB and the model has
                # a 4096-token context window.
                tool_output=json.dumps(last["tool_output"], default=str)[:2500],
            )
        )
        audit = result.model_dump()
    except Exception as exc:
        # Treat an unreadable audit as a pass. Failing closed here would replan
        # on infrastructure errors, which is the wrong response.
        audit = {
            "confidence_score": get_settings().audit_pass_score,
            "is_relevant": True,
            "reasoning": f"Audit unavailable ({type(exc).__name__}); proceeding.",
        }

    return {"verification_history": (state.get("verification_history") or []) + [audit]}


async def synthesizer_node(state: AgentState) -> Dict[str, Any]:
    steps = state.get("intermediate_steps") or []
    if not steps:
        return {
            "final_response": (
                "I could not gather enough grounded information to answer that. "
                "Try naming a specific metric and period."
            )
        }

    context = "\n\n".join(
        f"## Tool: {s['tool_name']}\nOutput: {json.dumps(s['tool_output'], default=str)[:3000]}"
        for s in steps
    )
    response = await chat_model(temperature=0.2, max_tokens=1024).ainvoke(
        SYNTHESIZER_PROMPT.format(request=state["original_request"], context=context)
    )
    return {"final_response": response.content}


# --- Routing ----------------------------------------------------------------


def route_after_audit(state: AgentState) -> str:
    settings = get_settings()
    history = state.get("verification_history") or []

    plan = state.get("plan") or []
    remaining = [s for s in plan if s["tool_name"] != "FINISH"]
    audit_failed = bool(history) and history[-1]["confidence_score"] < settings.audit_pass_score

    if audit_failed:
        if state.get("replan_count", 0) < settings.max_replans:
            return "planner"
        # Replan budget spent. Keep going only if untried steps remain — they may
        # yet succeed. Otherwise synthesise from imperfect evidence, which beats
        # burning supersteps re-auditing failures until the limit trips.
        return "execute_tool" if remaining else "synthesize"

    return "execute_tool" if remaining else "synthesize"


def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("gatekeeper", gatekeeper_node)
    builder.add_node("planner", planner_node)
    builder.add_node("execute_tool", tool_executor_node)
    builder.add_node("verify", auditor_node)
    builder.add_node("synthesize", synthesizer_node)

    builder.set_entry_point("gatekeeper")
    builder.add_conditional_edges(
        "gatekeeper",
        lambda s: END if (s.get("clarification_question") or s.get("final_response")) else "planner",
        {END: END, "planner": "planner"},
    )
    builder.add_edge("planner", "execute_tool")
    builder.add_edge("execute_tool", "verify")
    builder.add_conditional_edges(
        "verify",
        route_after_audit,
        {"planner": "planner", "execute_tool": "execute_tool", "synthesize": "synthesize"},
    )
    builder.add_edge("synthesize", END)
    return builder.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_config() -> Dict[str, Any]:
    return {"recursion_limit": get_settings().recursion_limit}


async def run_agent(query: str) -> AgentState:
    """Run to completion under a hard wall-clock ceiling.

    The recursion limit bounds graph steps but not time — a single slow tool can
    still hang the request, so both ceilings are needed.
    """
    settings = get_settings()
    initial: AgentState = {"original_request": query, "replan_count": 0}
    return await asyncio.wait_for(
        get_graph().ainvoke(initial, run_config()),
        timeout=settings.agent_timeout_s,
    )
