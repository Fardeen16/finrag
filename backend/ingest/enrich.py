"""Add search metadata to a chunk corpus, using Qwen on vLLM.

Runs as a separate process from `pipeline.py` on purpose. Doing both in one
process crashed the interpreter (0xC0000005) as soon as enrichment started: the
lxml/unstructured object graph for a 2 MB filing stays alive while async HTTP
work begins, and the combination is unstable on this platform. Splitting the
stages also means a failed enrichment run never costs the parse again.

Enrichment is best-effort by design. `enrich_chunk.py`, which this replaces,
dropped any chunk whose API call failed:

    if enrichment_data:
        all_enriched_chunks.append(...)

Here a failure leaves the chunk in place with `enriched: false` and an
`enrich_error` note, so content is never lost to a transient error. Progress is
flushed to disk periodically for the same reason.

    python backend/ingest/enrich.py --limit 8         # sample
    python backend/ingest/enrich.py                   # all unenriched chunks
    python backend/ingest/enrich.py --redo            # re-enrich everything
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

CORPUS = Path(__file__).resolve().parents[2] / "data" / "chunks_v2.json"
SAVE_EVERY = 25


class ChunkMetadata(BaseModel):
    """Enrichment fields.

    Deliberately flat, with no optional members. Qwen2.5-7B follows a small
    all-required schema far more reliably under constrained decoding, and a
    table's summary belongs in `summary` anyway.
    """

    summary: str = Field(description="One or two sentences describing the content.")
    keywords: List[str] = Field(description="Five to seven key topics or entities.")
    hypothetical_questions: List[str] = Field(
        description="Three to five questions this content answers."
    )


# The summary is embedded alongside the content, so it must read as a statement
# about Alphabet, not about the document. Qwen's default register is
# meta-commentary ("The provided excerpt does not contain...") which pollutes the
# vector with document-description language that no user query resembles.
_NO_META = """Write about Alphabet's business itself, never about this text.
Never begin with or include phrases like "The excerpt", "This chunk", "This
table", "The provided", "The document" or "This section". State the substance
directly. Add no facts that are not present. If the content is only
administrative boilerplate, say what it covers in one short clause."""

PROSE_PROMPT = f"""You are indexing Alphabet's 10-K for a financial analyst's search tool.

{_NO_META}

Phrase the questions the way an analyst would actually type them.

Content:
---
{{content}}
---"""

TABLE_PROMPT = f"""You are indexing a TABLE from Alphabet's 10-K for a financial
analyst's search tool.

{_NO_META}

Name the line items and periods, and state any trend in the figures, for
example: "Google Cloud revenue grew from $33.1B in 2023 to $43.2B in 2024."
Do not invent figures.

Content:
---
{{content}}
---"""


async def enrich_one(record: Dict[str, Any], model, attempts: int) -> bool:
    """Fill enrichment fields in place. Returns success; never raises."""
    body = str(record.get("content") or record.get("content_html") or "")
    if len(body.strip()) < 40:
        record["enrich_error"] = "content too short to enrich"
        return False

    template = TABLE_PROMPT if record.get("is_table") else PROSE_PROMPT
    prompt = template.format(content=body[:4000])

    for attempt in range(1, attempts + 1):
        try:
            result: ChunkMetadata = await model.ainvoke(prompt)
            record["summary"] = result.summary.strip()
            record["keywords"] = [k.strip() for k in result.keywords if k.strip()]
            record["hypothetical_questions"] = [
                q.strip() for q in result.hypothetical_questions if q.strip()
            ]
            record["enriched"] = True
            record.pop("enrich_error", None)
            return True
        except Exception as exc:  # noqa: BLE001 - enrichment is best-effort
            if attempt == attempts:
                record["enrich_error"] = f"{type(exc).__name__}: {exc}"[:200]
                return False
            # Jittered backoff so a burst of failures does not retry in lockstep.
            await asyncio.sleep(min(2**attempt, 8) + random.random())
    return False


async def run(
    records: List[Dict[str, Any]],
    targets: List[Dict[str, Any]],
    path: Path,
    concurrency: int,
    attempts: int,
) -> None:
    from backend.app.agent.llm import structured_model

    model = structured_model(ChunkMetadata, temperature=0.1)
    semaphore = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    total = len(targets)
    done = 0
    failed = 0

    def save() -> None:
        path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    async def worker(record: Dict[str, Any]) -> None:
        nonlocal done, failed
        async with semaphore:
            ok = await enrich_one(record, model, attempts)
        async with lock:
            done += 1
            if not ok:
                failed += 1
            if done % SAVE_EVERY == 0 or done == total:
                save()
                print(f"  {done}/{total} done, {failed} failed (saved)", flush=True)

    await asyncio.gather(*(worker(record) for record in targets))
    save()
    print(f"\nEnriched {total - failed}/{total}.")
    if failed:
        print(f"{failed} chunks kept with content but no metadata (enriched=false)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--limit", type=int, help="Enrich at most N chunks")
    parser.add_argument("--section", help="Only chunks from this section")
    parser.add_argument("--redo", action="store_true", help="Re-enrich already-done chunks")
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--attempts", type=int, default=3)
    args = parser.parse_args()

    records = json.loads(args.corpus.read_text(encoding="utf-8"))
    targets = records if args.redo else [r for r in records if not r.get("enriched")]
    if args.section:
        targets = [r for r in targets if r.get("section") == args.section]
    if args.limit:
        targets = targets[: args.limit]

    already = sum(1 for r in records if r.get("enriched"))
    print(f"corpus      {args.corpus.name}: {len(records)} chunks, {already} already enriched")
    print(f"to enrich   {len(targets)} (concurrency {args.concurrency}, {args.attempts} attempts)")
    if not targets:
        print("Nothing to do.")
        return 0

    asyncio.run(run(records, targets, args.corpus, args.concurrency, args.attempts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
