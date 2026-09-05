"""Locate the stage that loses content: partitioning or chunking.

Chunking splits, it does not drop, so if element text already falls short of
visible text the loss is in `partition_html`. Also traces specific missing terms
back to their surrounding prose so the gap can be characterised rather than
guessed at.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from unstructured.chunking.title import chunk_by_title
from unstructured.partition.html import partition_html

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.ingest.extract import (  # noqa: E402
    DEFAULT_SOURCE,
    clean_html,
    load_soup,
    strip_hidden,
    visible_text,
)

TERM = "antitrust"


def main() -> None:
    plain = visible_text(strip_hidden(load_soup()))
    cleaned = clean_html(DEFAULT_SOURCE)

    elements = partition_html(text=cleaned, infer_table_structure=True)
    element_chars = sum(len((el.text or "")) for el in elements)

    chunks = chunk_by_title(
        elements, max_characters=2048, combine_text_under_n_chars=256, new_after_n_chars=1800
    )
    chunk_chars = sum(len((c.text or "")) for c in chunks)

    print("Stage-by-stage character counts")
    print(f"  visible text      {len(plain):,}")
    print(f"  after partition   {element_chars:,}  ({element_chars / len(plain) * 100:.1f}%)")
    print(f"  after chunking    {chunk_chars:,}  ({chunk_chars / len(plain) * 100:.1f}%)")

    by_type: dict[str, int] = {}
    for el in elements:
        name = type(el).__name__
        by_type[name] = by_type.get(name, 0) + 1
    print("\nElement types from partition")
    for name, count in sorted(by_type.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<24} {count:>5}")

    # Trace the term through each stage.
    element_text = " ".join((el.text or "") for el in elements).lower()
    chunk_text = " ".join((c.text or "") for c in chunks).lower()
    print(f"\n'{TERM}' occurrences")
    print(f"  visible text      {plain.lower().count(TERM)}")
    print(f"  after partition   {element_text.count(TERM)}")
    print(f"  after chunking    {chunk_text.count(TERM)}")

    print(f"\nContexts in visible text (first 8)")
    for match in list(re.finditer(TERM, plain, re.I))[:8]:
        start = max(0, match.start() - 90)
        snippet = " ".join(plain[start : match.end() + 90].split())
        present = "kept" if snippet[:50].lower() in element_text else "LOST"
        print(f"  [{present}] ...{snippet}...")


if __name__ == "__main__":
    main()
