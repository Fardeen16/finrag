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

AUDITOR_PROMPT = """Audit this single tool call. Be lenient.

Score 1 only when the tool crashed, refused, or returned an execution error.
Any real payload — including "no rows", partial coverage, or data for a
different year than the user asked — is at least a 3. Other tools or the
Synthesizer can fill gaps. Do not fail a tool for being incomplete on its own.

User request: {request}
Tool called: {tool_name}
Tool output: {tool_output}
"""

SYNTHESIZER_PROMPT = """You are an expert financial analyst. Write a complete
answer to the user's request using every tool output below.

Rules:
- Combine the successful tools into one answer. Do not ignore a tool that
  returned useful text.
- If one tool found nothing or only covered part of the question, say so
  briefly and answer from the other tools.
- Where you infer a causal link, label it as a hypothesis.
- If the context does not support part of the request, say so.

User request:
{request}

Context:
---
{context}
---

Answer:"""

_TOOL_ERROR_MARKERS = (
    "tool raised",
    "unknown tool",
    "could not read database",
    "could not generate sql",
    "sql failed",
    "web search failed",
    "refused:",
)


def _tool_output_is_error(output: Any) -> bool:
    text = output if isinstance(output, str) else json.dumps(output, default=str)
    lowered = text.lower()
    return any(marker in lowered for marker in _TOOL_ERROR_MARKERS)


def _lenient_audit(raw: Dict[str, Any], tool_output: Any) -> Dict[str, Any]:
    """Force 1 only on real tool errors; otherwise never fail the pass threshold."""
    settings = get_settings()
    audit = dict(raw)
    if _tool_output_is_error(tool_output):
        audit["confidence_score"] = 1
        audit["is_relevant"] = False
        return audit
    score = int(audit.get("confidence_score") or settings.audit_pass_score)
    audit["confidence_score"] = max(score, settings.audit_pass_score)
    audit["is_relevant"] = True
    return audit


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
        kept = [s["tool_name"] for s in steps[:-1]]
        kept_note = (
            f"Keep using the already-successful tools ({', '.join(kept)}). "
            if kept
            else ""
        )
        feedback = (
            "Previous attempt feedback:\n"
            f"The tool '{failed_tool}' errored (auditor {audit['confidence_score']}/5): "
            f"'{audit['reasoning']}'\n"
            f"{kept_note}"
            "Produce a DIFFERENT next step that covers what was missed. "
            "Do not repeat the failed tool as the first step.\n"
        )

    plan = await structured_model(Plan).ainvoke(_planner_prompt(request, feedback))
    steps_out = [step.model_dump() for step in plan.steps]

    # Bound the plan so the superstep budget stays a known quantity.
    working = [s for s in steps_out if s["tool_name"] != "FINISH"][: settings.max_plan_steps]
    steps_out = working + [{"tool_name": "FINISH", "tool_input": ""}]

    updates: Dict[str, Any] = {"plan": steps_out}
    if is_replan:
        # Drop only the failed last step. Successful earlier tools stay so the
        # Synthesizer can still write a collective answer.
        updates["intermediate_steps"] = steps[:-1]
        updates["verification_history"] = history[:-1]
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
        audit = _lenient_audit(result.model_dump(), last["tool_output"])
    except Exception as exc:
        # Treat an unreadable audit as a pass. Failing closed here would replan
        # on infrastructure errors, which is the wrong response.
        audit = {
            "confidence_score": get_settings().audit_pass_score,
            "is_relevant": True,
            "reasoning": f"Audit unavailable ({type(exc).__name__}); proceeding.",
        }

    return {"verification_history": (state.get("verification_history") or []) + [audit]}


def _message_text(response: Any) -> str:
    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text") or part.get("content") or ""))
            else:
                parts.append(str(getattr(part, "text", "") or ""))
        return "".join(parts).strip()
    return str(content or "").strip()


def _readable_tool_excerpt(step: Dict[str, Any]) -> str:
    name = step.get("tool_name") or "tool"
    output = step.get("tool_output")
    if isinstance(output, str):
        return f"{name}: {output[:500]}"
    if isinstance(output, list):
        snippets = []
        for hit in output[:3]:
            if not isinstance(hit, dict):
                continue
            snippet = (hit.get("summary") or hit.get("content") or "")[:240]
            if snippet:
                snippets.append(snippet)
        return f"{name}: " + " | ".join(snippets) if snippets else f"{name}: no passages"
    return f"{name}: {json.dumps(output, default=str)[:400]}"


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
    # Do not use str.format — tool JSON braces can blow up the template.
    prompt = SYNTHESIZER_PROMPT.replace("{request}", state["original_request"]).replace(
        "{context}", context
    )
    text = ""
    try:
        # Non-streaming: a single completion. Streaming was passing
        # stream_chunk_timeout into ChatOpenAI and raising TypeError in ~4ms.
        response = await chat_model(
            temperature=0.2, max_tokens=1024, streaming=False
        ).ainvoke(prompt)
        text = _message_text(response)
    except Exception:
        text = ""

    if not text:
        text = (
            "Here is what the tools found (I could not format a full narrative):\n\n"
            + "\n\n".join(_readable_tool_excerpt(step) for step in steps)
        )
    return {"final_response": text}


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
