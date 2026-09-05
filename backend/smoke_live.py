"""Smoke-test the live SSE adapter against local vLLM. Prints event types only."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.stream import live_stream  # noqa: E402


async def main() -> int:
    query = "What was Alphabet total revenue in FY2024?"
    print(f"query: {query}\n")
    async for frame in live_stream(query):
        if frame.startswith(":"):
            print("  ping")
            continue
        for line in frame.splitlines():
            if line.startswith("data: "):
                payload = json.loads(line[6:])
                kind = payload.get("type")
                extra = ""
                if kind == "node_start":
                    extra = f" {payload.get('node')}"
                elif kind == "node_end":
                    extra = f" {payload.get('node')} {payload.get('duration_ms')}ms"
                elif kind == "plan":
                    extra = " " + ",".join(s["tool_name"] for s in payload.get("steps", []))
                elif kind == "tool_start":
                    extra = f" {payload.get('tool_name')}"
                elif kind == "audit":
                    extra = f" {payload.get('confidence_score')}/5 {payload.get('action')}"
                elif kind == "token":
                    extra = f" {payload.get('text')[:40]!r}"
                elif kind == "error":
                    extra = f" {payload.get('code')}: {payload.get('message')}"
                elif kind == "run_end":
                    extra = f" {payload.get('duration_ms')}ms replans={payload.get('replans')}"
                print(f"  {kind}{extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
