"""Compare an indexed corpus against the filing's actual readable text.

    python backend/measure_corpus.py                          # old corpus
    python backend/measure_corpus.py --chunks data/chunks_v2.json
    python backend/measure_corpus.py --chunks ... --gap        # sample what is missing

An earlier version of this measurement divided indexed characters by raw file
size, which is ~86% XBRL markup, and reported 7.1% coverage. Visible text is the
honest denominator.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.ingest.extract import (  # noqa: E402
    DEFAULT_SOURCE,
    find_sections,
    load_soup,
    strip_hidden,
    visible_text,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TERMS = ["risk factors", "adversely affect", "antitrust", "competition", "could harm"]


def normalise(text: str) -> str:
    return " ".join(text.lower().split())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=PROJECT_ROOT / "data" / "enriched_chunks.json")
    parser.add_argument("--gap", action="store_true", help="Sample lines absent from the corpus")
    args = parser.parse_args()

    print("Extracting visible text...")
    text = visible_text(strip_hidden(load_soup()))
    raw_size = DEFAULT_SOURCE.stat().st_size

    print(f"\nraw file            {raw_size:,} chars")
    print(f"visible text        {len(text):,} chars ({len(text) / raw_size * 100:.1f}% of raw)")

    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    contents = [str(c.get("content") or "") for c in chunks]
    indexed = " ".join(contents)
    tables = sum(1 for c in chunks if c.get("is_table"))

    print(f"\ncorpus              {args.chunks.name}")
    print(f"  chunks            {len(chunks):,} ({tables} tables, {len(chunks) - tables} prose)")
    print(f"  content chars     {len(indexed):,}")
    print(f"  coverage          {len(indexed) / len(text) * 100:.1f}% of visible text")

    lowered_text = normalise(text)
    lowered_indexed = normalise(indexed)
    print(f"\n{'term':<20} {'corpus':>8} {'filing':>8} {'kept':>7}")
    for term in TERMS:
        found = lowered_indexed.count(term)
        total = lowered_text.count(term)
        pct = f"{found / total * 100:.0f}%" if total else "n/a"
        print(f"{term:<20} {found:>8} {total:>8} {pct:>7}")

    if not args.gap:
        return

    # Which sentences of the filing never made it into any chunk? Compare on
    # normalised sentence text so whitespace and case differences do not count
    # as losses.
    print("\nGap analysis")
    sentences = [s.strip() for s in re.split(r"\n+", text) if len(s.strip()) > 25]
    missing = [s for s in sentences if normalise(s)[:60] not in lowered_indexed]
    print(f"  lines >25 chars   {len(sentences):,}")
    print(f"  absent from corpus{len(missing):,} ({len(missing) / len(sentences) * 100:.1f}%)")

    # Characterise the misses rather than eyeballing them one by one.
    buckets: Counter[str] = Counter()
    for line in missing:
        stripped = line.strip()
        if re.fullmatch(r"[\d\s.,$%()\-–—]+", stripped):
            buckets["numeric / table fragment"] += 1
        elif re.match(r"^(item|part)\s", stripped, re.I):
            buckets["item or part heading"] += 1
        elif len(stripped) < 60:
            buckets["short label or cell"] += 1
        elif stripped.count("$") > 2:
            buckets["financial figures"] += 1
        else:
            buckets["prose"] += 1
    print("\n  composition of missing lines")
    for name, count in buckets.most_common():
        print(f"    {name:<26} {count:>6}")

    prose_missing = [
        line
        for line in missing
        if len(line) >= 60
        and line.count("$") <= 2
        and not re.fullmatch(r"[\d\s.,$%()\-–—]+", line.strip())
        and not re.match(r"^(item|part)\s", line.strip(), re.I)
    ]
    print(f"\n  sample missing prose ({len(prose_missing)} lines)")
    for line in prose_missing[:12]:
        print(f"    - {' '.join(line.split())[:130]}")

    sections = sorted(find_sections(text).items(), key=lambda kv: kv[1])
    print("\n  item headings located (offsets should ascend; they do not,")
    print("  so section labels in the corpus are approximate)")
    for name, start in sections:
        print(f"    {name:<26} {start:>9,}")


if __name__ == "__main__":
    main()
