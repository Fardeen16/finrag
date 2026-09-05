"""Parse and chunk the filing into a corpus. No LLM calls.

Stage one of two; `enrich.py` adds search metadata afterwards. They are separate
processes because running both in one crashed the interpreter (0xC0000005) when
async HTTP work began while the lxml object graph was still alive.

Replaces the parsing half of `enrich_chunk.py`, which lost about a third of the
filing's prose and half its coverage:

* `partition_html` never descends into `<ix:continuation>`, the inline-XBRL tag
  that continues a text-block fact. The Antitrust Matters and cybersecurity
  discussions live inside it. `clean_html` now unwraps those tags, taking
  coverage of readable text from 68% to 99%.
* Table chunks embed flattened text, not `text_as_html`. The old pipeline stored
  raw HTML as embedded content for 72 of 138 chunks, so those vectors encoded
  markup. HTML is kept in `content_html` for display instead.
* Every chunk is kept. Enrichment is an overlay applied later, never a filter.

    python backend/ingest/pipeline.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from unstructured.chunking.title import chunk_by_title
from unstructured.partition.html import partition_html

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.ingest.extract import (  # noqa: E402
    DEFAULT_SOURCE,
    ITEM_PATTERNS,
    clean_html,
)

OUTPUT_PATH = Path(__file__).resolve().parents[2] / "data" / "chunks_v2.json"
SOURCE_LABEL = "10-K/alphabet_10k_submission.txt"

MAX_CHARACTERS = 2048
COMBINE_UNDER = 256
NEW_AFTER = 1800

# `text_as_html` alone flags 114 of 200 chunks as tables, because iXBRL wraps
# narrative prose in <table> for page layout. The two populations separate
# cleanly when measured: layout wrappers have exactly 2 rows and a digit ratio
# near 0.001, while real financial tables have 19-46 rows and ratios of
# 0.15-0.21. Requiring both signals keeps the statements and drops the wrappers.
TABLE_MIN_ROWS = 4
TABLE_MIN_DIGIT_RATIO = 0.02

# A section heading is a short standalone line. Cross-references such as "see
# Item 1A Risk Factors for further detail" match the same patterns but sit inside
# long sentences, so length is what separates the two.
MAX_HEADING_CHARS = 120


def digit_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(char.isdigit() for char in text) / len(text)


def is_real_table(html: Optional[str], text: str) -> bool:
    if not html:
        return False
    rows = html.lower().count("<tr")
    return rows >= TABLE_MIN_ROWS and digit_ratio(text) >= TABLE_MIN_DIGIT_RATIO


def heading_section(text: str) -> Optional[str]:
    """The item this text is a heading for, or None if it is not a heading."""
    stripped = " ".join(text.split())
    if not stripped or len(stripped) > MAX_HEADING_CHARS:
        return None
    lowered = stripped.lower()
    for key, pattern in ITEM_PATTERNS:
        if re.match(pattern, lowered):
            return key
    return None


def tag_elements(elements) -> Dict[str, Optional[str]]:
    """Map element id -> 10-K section, by walking document order.

    This replaces searching for each chunk's text inside the flat filing. That
    search kept failing because unstructured collapses whitespace, so chunks
    silently inherited the previous chunk's label and Item 1A never appeared at
    all. Elements arrive in document order, so tracking the current heading as we
    pass it needs no matching and cannot drift.

    The table of contents lists every item before the body begins, which would
    otherwise consume all ten headings immediately. It is skipped by requiring
    that a heading not be immediately followed by another heading.
    """
    labels: Dict[str, Optional[str]] = {}
    current: Optional[str] = None
    for index, element in enumerate(elements):
        section = heading_section(element.text or "")
        if section is not None:
            following = elements[index + 1].text if index + 1 < len(elements) else ""
            in_toc = heading_section(following or "") is not None
            if not in_toc:
                current = section
        labels[str(element.id)] = current
    return labels


def build_chunks() -> List[Dict[str, Any]]:
    print(f"Reading {DEFAULT_SOURCE.name}...")
    cleaned = clean_html(DEFAULT_SOURCE, verbose=True)
    print(f"  cleaned html    {len(cleaned):,} chars")

    print("Partitioning...")
    elements = partition_html(text=cleaned, infer_table_structure=True)
    print(f"  elements        {len(elements):,}")

    labels = tag_elements(elements)
    tagged = sum(1 for value in labels.values() if value)
    print(f"  tagged          {tagged:,}/{len(elements):,} elements with a section")

    print("Chunking...")
    chunks = chunk_by_title(
        elements,
        max_characters=MAX_CHARACTERS,
        combine_text_under_n_chars=COMBINE_UNDER,
        new_after_n_chars=NEW_AFTER,
        include_orig_elements=True,
    )
    print(f"  chunks          {len(chunks):,}")

    records: List[Dict[str, Any]] = []
    fallback: Optional[str] = None
    resolved = 0
    for index, chunk in enumerate(chunks):
        meta = chunk.metadata.to_dict()
        html = meta.get("text_as_html")
        text = (chunk.text or "").strip()
        is_table = is_real_table(html, text)

        if not text and not html:
            continue

        # Take the section from the chunk's constituent elements. Carry the last
        # known section forward when a chunk has none, which mirrors how a reader
        # would attribute unlabelled prose.
        section: Optional[str] = None
        for original in getattr(chunk.metadata, "orig_elements", None) or []:
            section = labels.get(str(original.id))
            if section:
                break
        if section:
            resolved += 1
            fallback = section
        else:
            section = fallback

        records.append(
            {
                "id": index,
                "source": SOURCE_LABEL,
                "section": section,
                "is_table": is_table,
                # Embedded content is always text. HTML is display-only.
                "content": text or "",
                "content_html": html if is_table else None,
                "enriched": False,
                "summary": None,
                "keywords": [],
                "hypothetical_questions": [],
            }
        )

    print(f"  kept            {len(records):,} non-empty chunks")
    print(f"  sectioned       {resolved:,} chunks resolved directly from elements")
    return records


def report(records: List[Dict[str, Any]]) -> None:
    tables = sum(1 for r in records if r["is_table"])
    enriched = sum(1 for r in records if r["enriched"])
    chars = sum(len(r["content"]) for r in records)
    print("\nCorpus")
    print(f"  chunks          {len(records):,}  ({tables} tables, {len(records) - tables} prose)")
    print(f"  enriched        {enriched:,}")
    print(f"  content chars   {chars:,}")

    by_section: Dict[str, int] = {}
    for record in records:
        key = record["section"] or "(unlabelled)"
        by_section[key] = by_section.get(key, 0) + 1
    print("\nChunks per section")
    for name, count in sorted(by_section.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<26} {count:>5}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    records = build_chunks()
    report(records)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
