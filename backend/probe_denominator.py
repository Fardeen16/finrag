"""Is 'visible text' an honest denominator, and where does the lost prose live?

Two things to rule out before treating the 32% gap as real content loss:

1. That `strip_hidden` actually removes the iXBRL hidden fact store. If the tag
   names do not match what the parser produced, the denominator silently
   includes thousands of duplicated facts and every coverage number is wrong.
2. That the missing prose is not sitting inside <table> elements, where
   `partition_html` would route it to `text_as_html` instead of `.text`.
"""

from __future__ import annotations

import sys
from pathlib import Path

from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.ingest.extract import DEFAULT_SOURCE, load_soup, strip_hidden, visible_text  # noqa: E402

PROBE = "Antitrust Matters"


def main() -> None:
    soup = load_soup()
    before = visible_text(soup)
    print(f"visible text before strip_hidden   {len(before):,}")

    # What hidden-ish tags does the parser actually expose?
    print("\ntag names matching hidden/header:")
    for name in ("ix:header", "header", "ix:hidden", "hidden", "ix:references", "ix:resources"):
        found = soup.find_all(name)
        if found:
            chars = sum(len(t.get_text()) for t in found)
            print(f"  {name:<16} {len(found):>4} tags, {chars:>10,} chars of text")

    soup2 = strip_hidden(load_soup())
    after = visible_text(soup2)
    print(f"\nvisible text after strip_hidden    {len(after):,}")
    print(f"removed                            {len(before) - len(after):,}")

    # Where does the lost passage live in the DOM?
    raw = DEFAULT_SOURCE.read_text(encoding="utf-8", errors="ignore")
    index = raw.find(PROBE)
    print(f"\n'{PROBE}' found in raw at offset {index:,}")
    if index < 0:
        return

    fresh = BeautifulSoup(raw, "lxml")
    node = fresh.find(string=lambda s: s and PROBE in s)
    if node is None:
        print("  could not locate the string as a text node")
        return

    ancestors = [p.name for p in node.parents if p.name]
    print(f"  ancestor chain: {' < '.join(ancestors[:12])}")
    print(f"  inside a <table>? {'table' in ancestors}")

    table = next((p for p in node.parents if p.name == "table"), None)
    if table is not None:
        text = " ".join(table.get_text(" ").split())
        print(f"  enclosing table holds {len(text):,} chars of text")
        print(f"  first 300: {text[:300]}")


if __name__ == "__main__":
    main()
