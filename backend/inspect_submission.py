"""What is actually inside data/alphabet_10k_submission.txt?

The ingestion scripts run partition_html over this whole file. It is an SEC
full-submission dissemination file: several SGML-wrapped documents including
exhibits and XBRL, not a single 10-K. This prints the document inventory so the
right one can be extracted.
"""

from __future__ import annotations

import re
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "data" / "alphabet_10k_submission.txt"

DOC_RE = re.compile(r"<DOCUMENT>(.*?)</DOCUMENT>", re.S | re.I)
FIELD_RE = {
    name: re.compile(rf"<{name}>(.*?)\s*$", re.I | re.M)
    for name in ("TYPE", "SEQUENCE", "FILENAME", "DESCRIPTION")
}


def main() -> None:
    raw = SOURCE.read_text(encoding="utf-8", errors="ignore")
    print(f"file: {SOURCE.name}  {len(raw):,} chars\n")

    docs = DOC_RE.findall(raw)
    if not docs:
        print("No <DOCUMENT> wrappers found — file may be a bare HTML document.")
        print(f"starts with: {raw[:300]!r}")
        return

    print(f"{len(docs)} documents inside\n")
    print(f"{'#':>3} {'TYPE':<14} {'SEQ':>4} {'chars':>12}  FILENAME")
    for index, body in enumerate(docs, start=1):
        fields = {}
        for name, pattern in FIELD_RE.items():
            match = pattern.search(body)
            fields[name] = match.group(1).strip() if match else ""
        print(
            f"{index:>3} {fields['TYPE'][:14]:<14} {fields['SEQUENCE']:>4} "
            f"{len(body):>12,}  {fields['FILENAME'][:40]}"
        )

    # Where does the risk-factor language actually live?
    print("\nRisk-factor markers per document:")
    for index, body in enumerate(docs, start=1):
        lowered = body.lower()
        counts = {
            term: lowered.count(term)
            for term in ("risk factors", "adversely affect", "antitrust", "competition")
        }
        if any(counts.values()):
            type_match = FIELD_RE["TYPE"].search(body)
            doc_type = type_match.group(1).strip() if type_match else "?"
            print(f"  doc {index} ({doc_type}): {counts}")


if __name__ == "__main__":
    main()
