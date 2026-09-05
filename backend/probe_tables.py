"""How many chunks flagged is_table are genuinely tabular?

`text_as_html` presence is the current test, and it flags 114 of 200 chunks.
That is implausible for a 10-K: iXBRL uses <table> for page layout, so narrative
prose ends up wrapped in one. Digit density separates the two cleanly enough to
pick a threshold from the distribution rather than by guessing.

Reads the saved corpus rather than re-partitioning — partitioning this filing
segfaults intermittently, and the saved records already carry text and HTML.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_CHUNKS = Path(__file__).resolve().parents[1] / "data" / "chunks_v2.json"


def digit_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(char.isdigit() for char in text) / len(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    args = parser.parse_args()

    rows = json.loads(args.chunks.read_text(encoding="utf-8"))
    flagged = []
    for record in rows:
        if not record.get("is_table"):
            continue
        text = str(record.get("content") or "")
        html = str(record.get("content_html") or "")
        flagged.append(
            {
                "id": record["id"],
                "section": record.get("section"),
                "ratio": digit_ratio(text),
                "rows": html.lower().count("<tr"),
                "chars": len(text),
                "head": " ".join(text.split())[:66],
            }
        )

    print(f"{len(flagged)} of {len(rows)} chunks flagged is_table\n")
    buckets = [(0.0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 1.01)]
    for low, high in buckets:
        group = [f for f in flagged if low <= f["ratio"] < high]
        print(f"  digit ratio {low:.2f}-{high:.2f}  {len(group):>4} chunks")

    print("\nlowest digit density (prose wrapped in a layout table)")
    for item in sorted(flagged, key=lambda f: f["ratio"])[:8]:
        print(f"  id={item['id']:>3} r={item['ratio']:.3f} tr={item['rows']:>3} {item['head']}")

    print("\nhighest digit density (real financial tables)")
    for item in sorted(flagged, key=lambda f: -f["ratio"])[:8]:
        print(f"  id={item['id']:>3} r={item['ratio']:.3f} tr={item['rows']:>3} {item['head']}")

    for threshold in (0.02, 0.03, 0.05):
        kept = sum(1 for f in flagged if f["ratio"] >= threshold)
        print(f"\nthreshold {threshold:.2f} -> {kept} tables, {len(rows) - kept} prose")


if __name__ == "__main__":
    main()
