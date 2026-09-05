"""Run the migrated agent against vLLM from the command line.

    python backend/run_agent.py "What was total revenue in FY2024?"
    python backend/run_agent.py --check          # config + connectivity only
    python backend/run_agent.py --nodes "..."    # per-node timing

Exists so the migration can be verified without the HTTP layer in the way.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.agent.graph import get_graph, run_config  # noqa: E402
from backend.app.config import get_settings  # noqa: E402


async def check() -> int:
    settings = get_settings()
    print("Configuration")
    print(f"  vLLM            {settings.vllm_base_url}")
    print(f"  model           {settings.vllm_model}")
    print(f"  embeddings      {settings.embedding_provider}")
    print(f"  qdrant          {settings.qdrant_url or settings.qdrant_path}")
    print(f"  collection      {settings.collection_name}")
    print(f"  max_replans     {settings.max_replans}")
    print(f"  recursion_limit {settings.recursion_limit}")

    failures = 0

    print("\nConnectivity")
    try:
        from backend.app.agent.llm import chat_model

        t0 = time.perf_counter()
        reply = await chat_model(max_tokens=8).ainvoke("Reply with: ok")
        print(f"  [OK]   vLLM chat ({time.perf_counter() - t0:.2f}s) -> {reply.content!r}")
    except Exception as exc:
        print(f"  [FAIL] vLLM chat: {type(exc).__name__}: {exc}")
        failures += 1

    try:
        from backend.app.agent.resources import collection_dimension

        dim = collection_dimension()
        print(f"  [OK]   Qdrant collection, {dim}-dim vectors")
    except Exception as exc:
        print(f"  [FAIL] Qdrant: {type(exc).__name__}: {exc}")
        failures += 1
        dim = None

    try:
        from backend.app.agent.embeddings import embedding_dimension

        emb_dim = embedding_dimension()
        if dim is not None and emb_dim != dim:
            print(f"  [FAIL] embeddings are {emb_dim}-dim but collection is {dim}-dim")
            failures += 1
        else:
            print(f"  [OK]   embeddings, {emb_dim}-dim")
    except Exception as exc:
        print(f"  [FAIL] embeddings: {type(exc).__name__}: {exc}")
        failures += 1

    try:
        from backend.app.agent.resources import get_sql_database

        tables = get_sql_database().get_usable_table_names()
        print(f"  [OK]   SQLite tables: {tables}")
    except Exception as exc:
        print(f"  [FAIL] SQLite: {type(exc).__name__}: {exc}")
        failures += 1

    print("\n" + ("all checks passed" if failures == 0 else f"{failures} check(s) failed"))
    return 1 if failures else 0


async def stream_nodes(query: str) -> int:
    """Walk the graph, printing each node as it completes with its wall time."""
    settings = get_settings()
    print(f"Query: {query}\n")

    state = {"original_request": query, "replan_count": 0}
    started = time.perf_counter()
    last = started
    timings: dict[str, float] = {}
    final = None

    try:
        async for update in get_graph().astream(state, run_config()):
            now = time.perf_counter()
            for node, payload in update.items():
                elapsed = now - last
                timings[node] = timings.get(node, 0.0) + elapsed
                print(f"  {node:<14} {elapsed:>6.2f}s", end="")

                if node == "planner" and payload.get("plan"):
                    names = [s["tool_name"] for s in payload["plan"]]
                    replans = payload.get("replan_count")
                    suffix = f"  (replan #{replans})" if replans else ""
                    print(f"  plan={names}{suffix}")
                elif node == "execute_tool" and payload.get("intermediate_steps"):
                    step = payload["intermediate_steps"][-1]
                    preview = str(step["tool_output"]).replace("\n", " ")[:90]
                    print(f"  {step['tool_name']} -> {preview}")
                elif node == "verify" and payload.get("verification_history"):
                    audit = payload["verification_history"][-1]
                    verdict = (
                        "REPLAN"
                        if audit["confidence_score"] < settings.audit_pass_score
                        else "accept"
                    )
                    print(f"  score={audit['confidence_score']}/5 {verdict}")
                elif node == "gatekeeper":
                    q = payload.get("clarification_question")
                    print(f"  {'CLARIFY: ' + q[:70] if q else 'specific'}")
                else:
                    print()

                if payload.get("final_response"):
                    final = payload["final_response"]
                if payload.get("clarification_question"):
                    final = "[clarification] " + payload["clarification_question"]
                last = now
    except asyncio.TimeoutError:
        print("\nAgent exceeded its wall-clock ceiling.")
        return 1

    total = time.perf_counter() - started
    print(f"\n{'-' * 70}\n{final}\n{'-' * 70}")
    print(f"total {total:.2f}s")
    for node, seconds in sorted(timings.items(), key=lambda kv: -kv[1]):
        print(f"  {node:<14} {seconds:>6.2f}s  {seconds / total * 100:>4.0f}%")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", help="Question to ask the agent")
    parser.add_argument("--check", action="store_true", help="Config and connectivity only")
    args = parser.parse_args()

    if args.check or not args.query:
        return asyncio.run(check())
    return asyncio.run(stream_nodes(args.query))


if __name__ == "__main__":
    raise SystemExit(main())
