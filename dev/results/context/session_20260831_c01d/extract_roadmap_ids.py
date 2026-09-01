#!/usr/bin/env python3
"""Extract task IDs from docs/VYVAR_ROADMAP.md.

IDs are hyphenated tokens inside **bold**, plus a small extras list of
unbolded IDs that live in NEXT SESSION headings. Naive ALLCAPS is not used.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
ROADMAP = REPO / "docs" / "VYVAR_ROADMAP.md"
HERE = Path(__file__).resolve().parent

BOLD_HYPHEN = re.compile(r"\*\*([A-Z][A-Z0-9]*(?:-[A-Z0-9*]+)+)\*\*")
BOLD_TODO_N = re.compile(r"\*\*(TODO-[0-9]+[a-z0-9]*)\*\*")
BOLD_TODO_BARE = re.compile(r"\*\*(TODO-[A-Z][A-Z0-9]*)\*\*")
HEADING_ID = re.compile(
    r"^## (?:DONE|CLOSED|OPEN|QUEUED|SUPERSEDED|IN-FLIGHT|NEXT SESSION)[^\n]*?\b([A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+)\b",
    re.M,
)

# Unbolded IDs that the stacked NEXT SESSION headings used as task names.
EXTRAS = (
    "EDGE-ANNULUS-01",
    "SEL-GHOST-01",
    "APERTURE-01",
    "APERTURE-01b",
    "APERTURE-01c",
    "APERTURE-01d",
    "FRAME-QC-PARITY",
    "FRAME-QC-PARITY-01",
    "REG-520-01",
    "EPSF-VALID-02",
    "DOCS-SYNC-517",
    "ERA-03",
    "ARCHIVE-CLEANUP",
)


def extract_ids(text: str) -> dict[str, int]:
    found: dict[str, int] = {}
    for rx in (BOLD_HYPHEN, BOLD_TODO_N, BOLD_TODO_BARE):
        for m in rx.finditer(text):
            tok = m.group(1).rstrip("*")
            if tok.endswith("-"):
                continue
            found[tok] = found.get(tok, 0) + 1
    for m in HEADING_ID.finditer(text):
        tok = m.group(1)
        found[tok] = found.get(tok, 0) + 1
    for tok in EXTRAS:
        if re.search(rf"\b{re.escape(tok)}\b", text):
            found[tok] = found.get(tok, 0) + 1
    return found


def main() -> None:
    out_name = "roadmap_ids_before.json"
    if len(sys.argv) > 1:
        out_name = sys.argv[1]
    out = HERE / out_name
    text = ROADMAP.read_text(encoding="utf-8")
    found = extract_ids(text)
    ids = sorted(found)
    payload = {
        "n": len(ids),
        "ids": ids,
        "counts": {k: found[k] for k in ids},
        "source": "docs/VYVAR_ROADMAP.md",
    }
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"n={len(ids)} wrote {out}")


if __name__ == "__main__":
    main()
