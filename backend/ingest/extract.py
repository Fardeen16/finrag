"""Extract readable 10-K text from the inline-XBRL filing.

The source is a single iXBRL document (Workiva-generated), not an SGML
submission wrapper. That matters for two reasons:

1. Most of its 2.5 MB is markup and XBRL facts, so measuring index coverage
   against raw file size badly understates it. `visible_text()` gives the honest
   denominator.
2. `<ix:header>` holds the hidden fact store — thousands of tagged values plus,
   often, entire narrative sections duplicated inside TextBlock facts. Feeding it
   to a chunker produces duplicate and non-prose chunks, so it is stripped.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

from bs4 import BeautifulSoup, Comment, XMLParsedAsHTMLWarning

# The filing opens with an XML declaration but is HTML in practice; the HTML
# parser is what handles its malformed markup correctly.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "alphabet_10k_submission.txt"

# Canonical 10-K part/item headings, used to locate sections in the flat text.
ITEM_PATTERNS: List[Tuple[str, str]] = [
    ("item_1_business", r"item\s*1\.?\s*business"),
    ("item_1a_risk_factors", r"item\s*1a\.?\s*risk\s*factors"),
    ("item_1b_unresolved", r"item\s*1b\.?\s*unresolved\s*staff\s*comments"),
    ("item_1c_cybersecurity", r"item\s*1c\.?\s*cybersecurity"),
    ("item_2_properties", r"item\s*2\.?\s*properties"),
    ("item_3_legal", r"item\s*3\.?\s*legal\s*proceedings"),
    ("item_5_market", r"item\s*5\.?\s*market\s*for"),
    ("item_7_mdna", r"item\s*7\.?\s*management.s\s*discussion"),
    ("item_7a_market_risk", r"item\s*7a\.?\s*quantitative"),
    ("item_8_financial", r"item\s*8\.?\s*financial\s*statements"),
    ("item_9a_controls", r"item\s*9a\.?\s*controls"),
]


def load_soup(path: Path = DEFAULT_SOURCE) -> BeautifulSoup:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    # lxml handles the iXBRL namespaces and malformed markup far faster than
    # html.parser at this size.
    return BeautifulSoup(raw, "lxml")


# Non-rendered iXBRL infrastructure: the fact store, plus the context and unit
# definitions in ix:resources. None of it is human-readable content, and leaving
# ix:resources in place inflated the coverage denominator by ~28k chars.
XBRL_PLUMBING = ("ix:header", "header", "ix:hidden", "hidden", "ix:resources", "ix:references")

# Rendered iXBRL tags. Their contents are part of the visible document, but
# `partition_html` does not descend into namespaced elements it does not know,
# so they must be unwrapped rather than kept. ix:continuation is the important
# one: it continues a text-block fact, and the legal-contingencies notes live
# inside it.
XBRL_INLINE_PREFIX = "ix:"


def strip_hidden(soup: BeautifulSoup) -> BeautifulSoup:
    """Remove the XBRL fact store, scripts, styles and comments."""
    for tag_name in ("script", "style", "head"):
        for tag in soup.find_all(tag_name):
            tag.decompose()

    for name in XBRL_PLUMBING:
        for tag in soup.find_all(name):
            tag.decompose()

    for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()

    # Elements explicitly hidden from rendering are XBRL plumbing, not content.
    for tag in soup.find_all(style=re.compile(r"display\s*:\s*none", re.I)):
        tag.decompose()

    return soup


def visible_text(soup: BeautifulSoup) -> str:
    text = soup.get_text("\n")
    text = re.sub(r"[ \t\u00a0]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def unwrap_inline_xbrl(soup: BeautifulSoup) -> int:
    """Replace remaining ix:* tags with their children. Returns the count.

    Without this, `partition_html` silently skips everything nested inside
    ix:continuation — about a third of the filing's prose, including the
    Antitrust Matters and cybersecurity discussions.
    """
    removed = 0
    for tag in soup.find_all(lambda t: t.name and t.name.startswith(XBRL_INLINE_PREFIX)):
        tag.unwrap()
        removed += 1
    return removed


def clean_html(path: Path = DEFAULT_SOURCE, verbose: bool = False) -> str:
    """HTML with XBRL plumbing removed, suitable for `partition_html`.

    Tables are preserved — they carry the financial statements — so the chunker
    can still detect and structure them.
    """
    soup = strip_hidden(load_soup(path))
    unwrapped = unwrap_inline_xbrl(soup)
    if verbose:
        print(f"  unwrapped       {unwrapped:,} inline iXBRL tags")
    body = soup.body or soup
    return str(body)


def _greedy_pass(lowered: str, start: int) -> List[Tuple[str, int]]:
    """Match items in canonical order, each after the previous one."""
    found: List[Tuple[str, int]] = []
    cursor = start
    for key, pattern in ITEM_PATTERNS:
        match = re.compile(pattern).search(lowered, cursor)
        if match:
            found.append((key, match.start()))
            cursor = match.end()
    return found


def find_sections(text: str) -> Dict[str, int]:
    """Character offsets of each 10-K item heading, in document order.

    Neither "first match" nor "last match" works here. The table of contents
    lists every item in the same order as the body, so first-match returns the
    TOC; cross-references like "see Item 1A Risk Factors" appear throughout, so
    last-match returns those and yields offsets that run backwards.

    Instead, sweep for ordered runs of headings and keep the run that spans the
    most text. The real sections are the only run covering the whole filing —
    the TOC is packed into a couple of thousand characters.
    """
    lowered = text.lower()
    best: List[Tuple[str, int]] = []
    best_span = -1
    start = 0

    while start < len(lowered):
        run = _greedy_pass(lowered, start)
        if len(run) < 2:
            break
        span = run[-1][1] - run[0][1]
        if span > best_span:
            best_span, best = span, run
        start = run[0][1] + 1

    return dict(best)
