"""FITS-KEYS-2026-10: every FITS keyword VYVAR writes is a standard keyword of at most 8 characters
(no HIERARCH cards). The renamed keywords (docs/VYVAR_FITS_KEYWORDS.md) are not read under their
old names: old drafts are processed again."""

from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src_py"

# a VYVAR keyword literal anywhere in the code ("VY_..." / "VYxxxx")
VY_LITERAL = re.compile(r"[\"'](VY(?!VAR_)_?[A-Z0-9_]+)[\"']")
# an upper-case keyword assigned into a header: hdr["KEY"] = ...
HEADER_WRITE = re.compile(r"\b\w*(?:hdr|header|hd)\w*\[[\"']([A-Z0-9_-]+)[\"']\]\s*=")
OLD_NAMES = ("VY_SKYPASS", "VY_CALDATASUM", "VY_CALSTAGE", "VY_CDSTAT", "VY_DKRSMP", "VY_DKRSMP_SRC",
             "VY_RDNOIS", "VY_CHANNEL", "VY_BAYERMASK", "VY_OSC_BIN", "VY_OSCSRC", "VY_OSCWCS", "VY_OSCREG",
             "VY_FWHM_GAUSS", "VY_FWHM_GAUSSIAN", "VY_NSTARS", "VY_NDETECT", "VY_PLATESCALE", "VY_REFFILE",
             "VY_MSFRAME", "VY_SRCFILE", "VY_SRCFRAME", "VY_SRCPATH", "VY_COSMNPX", "VY_TARGRA", "VY_TARGDEC")


def _sources() -> list[tuple[str, str]]:
    return [(p.name, p.read_text(encoding="utf-8")) for p in sorted(SRC.glob("*.py"))]


def test_vyvar_keywords_have_at_most_8_characters() -> None:
    long_keys = sorted({f"{n}:{k}" for n, s in _sources() for k in VY_LITERAL.findall(s) if len(k) > 8})
    assert long_keys == []


def test_header_writes_use_standard_keywords() -> None:
    long_keys = sorted({f"{n}:{k}" for n, s in _sources() for k in HEADER_WRITE.findall(s) if len(k) > 8})
    assert long_keys == []


def test_old_keyword_names_are_gone() -> None:
    pat = re.compile(r"\b(" + "|".join(OLD_NAMES) + r")\b")
    hits = sorted({f"{n}:{m}" for n, s in _sources() for m in pat.findall(s)})
    assert hits == []
