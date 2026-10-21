#!/usr/bin/env python3
"""Repair 38 cp1252-undefined (0x9d) bytes in VYVAR_AUDIT_2026_CLOSURE.md at 4a3e855."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "VYVAR_AUDIT_2026_CLOSURE.md"
LOG: list[str] = []


def repair_text(text: str) -> str:
    """Apply ordered repairs; log each substitution class."""
    rules: list[tuple[str, str, str]] = [
        (r"2026 \ufffd Wave", "2026 - Wave", "L1 title em dash"),
        (r"7\ufffd7", "7x7", "multiply box"),
        (r"target phot \? VYVAR", "target phot vs VYVAR", "L37 header"),
        (r"\| \ufffd \| \ufffd \|", "| - | - |", "L37 table cells"),
        (r"VYVAR\ufffdphotutils", "VYVAR-photutils", "L39 dash"),
        (r"\?3 mmag", "~3 mmag", "approx 3 mmag"),
        (r"7\.6\ufffd7\.8", "7.6-7.8", "L39 range"),
        (r"0\.5\ufffd15", "0.5-15", "radius range"),
        (r"5\.0\ufffd5\.75", "5.0-5.75", "EE px range"),
        (r"3\.3\ufffd3\.5", "3.3-3.5", "FWHM range"),
        (r"81\ufffd86%", "81-86%", "EE pct range"),
        (r"67\ufffd73%", "67-73%", "EE pct range"),
        (r"2\.5\ufffd r_Kron", "2.5x r_Kron", "Kron multiplier"),
        (r"factor \ufffd mean", "factor x mean", "AIJ factor"),
        (r"6 \ufffd median", "6 x median", "VaST multiplier"),
        (r"VYVAR\ufffds", "VYVAR's", "possessive"),
        (r"medium\ufffdhigh", "medium-high", "compound"),
        (r"2\ufffd underquote", "2x underquote", "factor 2x"),
        (r"\(\ufffd3\.1\)", "(S3.1)", "section ref"),
        (r"per \ufffd2", "per S2", "section ref"),
        (r"Fixed r\?\?", "Fixed r90", "r90 literal corruption"),
        (r"FITS\*\* \ufffd", "FITS** -", "em dash after FITS"),
        (r"435 \ufffd", "435 -", "em dash"),
        (r"A-1 \? draft", "A-1 - draft", "em dash"),
        (r" \ufffd ", " - ", "generic em dash"),
    ]
    out = text
    for pat, repl, note in rules:
        new = re.sub(pat, repl, out)
        if new != out:
            LOG.append(f"{note}: applied")
            out = new
    return out


def main() -> int:
    raw = subprocess.check_output(["git", "cat-file", "-p", "4a3e855:docs/VYVAR_AUDIT_2026_CLOSURE.md"])
    n9d = raw.count(bytes([0x9D]))
    text = raw.decode("cp1252", "replace")
    out = repair_text(text)
    if "\ufffd" in out:
        for i, line in enumerate(out.splitlines(), 1):
            if "\ufffd" in line:
                raise SystemExit(f"U+FFFD remains at line {i}: {line[:100]!r}")
    for c in out:
        if ord(c) > 127:
            raise SystemExit(f"non-ASCII U+{ord(c):04X} at position")
    OUT.write_text(out if out.endswith("\n") else out + "\n", encoding="ascii", newline="\n")
    (ROOT / "tmp" / "closure_repair_log.txt").write_text(
        f"0x9d bytes in source: {n9d}\nrepairs: {len(LOG)}\n" + "\n".join(LOG),
        encoding="ascii",
    )
    print(f"wrote {OUT}; source 0x9d={n9d}; repairs={len(LOG)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
