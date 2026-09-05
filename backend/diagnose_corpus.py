"""How much of the 10-K actually made it into the index?

Retrieval for risk-factor questions kept returning business-description content,
which suggests the chunking dropped sections rather than that the embeddings are
weak. Compares indexed text against the source filing.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHUNKS = ROOT / "data" / "enriched_chunks.json"
SOURCE = ROOT / "data" / "alphabet_10k_submission.txt"

TERMS = [
    "risk factors",
    "item 1a",
    "competition",
    "antitrust",
    "adversely affect",
    "could harm",
]


def main() -> None:
    chunks = json.loads(CHUNKS.read_text(encoding="utf-8"))
    indexed = " ".join(str(c.get("content", "")) for c in chunks)
    source = SOURCE.read_text(encoding="utf-8", errors="ignore")

    print(f"source file    {SOURCE.stat().st_size:,} bytes")
    print(f"source text    {len(source):,} chars")
    print(f"indexed text   {len(indexed):,} chars across {len(chunks)} chunks")
    print(f"coverage       {len(indexed) / len(source) * 100:.1f}% of source")
    print(f"avg chunk      {len(indexed) // len(chunks):,} chars")

    lower_indexed = indexed.lower()
    lower_source = source.lower()
    print(f"\n{'term':<20} {'indexed':>8} {'source':>8}")
    for term in TERMS:
        print(f"{term:<20} {lower_indexed.count(term):>8} {lower_source.count(term):>8}")

    sources = {c.get("source") for c in chunks}
    print(f"\ndistinct payload sources: {sources}")
    tables = sum(1 for c in chunks if c.get("is_table"))
    print(f"chunks flagged as tables: {tables}/{len(chunks)}")


if __name__ == "__main__":
    main()
