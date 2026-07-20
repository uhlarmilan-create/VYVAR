#!/usr/bin/env python3
"""ASCII-migrate tracked text files (ENCODING-POLICY).

Decode heuristic: UTF-8, else cp1252 (undefined bytes -> U+FFFD via replace).
Transliterate via an explicit mapping table. Unmapped non-ASCII => STOP (no write).
Idempotent. --check reports without writing.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

EXTS = {
    ".md",
    ".py",
    ".json",
    ".txt",
    ".cfg",
    ".toml",
    ".yml",
    ".yaml",
    ".ps1",
    ".sh",
}
SPECIAL_NAMES = {".gitignore", ".gitattributes", "LICENSE"}

# Explicit transliteration only -- no guessing beyond this table.
CHAR_MAP: dict[str, str] = {
    # dashes / minus
    "\u2014": "-",  # em dash
    "\u2013": "-",  # en dash
    "\u2012": "-",  # figure dash
    "\u2011": "-",  # non-breaking hyphen
    "\u2010": "-",  # hyphen
    "\u2212": "-",  # minus sign
    "\u00ad": "",  # soft hyphen
    # quotes -- curly doubles become SINGLE quotes so Python "..." strings
    # that contained typographic quotes do not break when ASCII-folded.
    "\u2018": "'",
    "\u2019": "'",
    "\u201a": "'",
    "\u201b": "'",
    "\u201c": "'",
    "\u201d": "'",
    "\u201e": "'",
    "\u201f": "'",
    "\u2032": "'",  # prime
    "\u2033": " arcsec",  # double prime (plate scale etc.)
    # ellipsis / spaces / signs
    "\u2026": "...",
    "\u00a0": " ",
    "\u00d7": "x",
    "\u00b1": "+-",
    "\u00b7": ".",  # middle dot
    "\u00b2": "^2",
    "\u00b3": "^3",
    "\u00b9": "^1",
    "\u00b0": " deg",
    "\u00b5": "u",  # micro
    "\u2248": "~",
    "\u2264": "<=",
    "\u2265": ">=",
    "\u2260": "!=",
    "\u2261": "==",
    "\u226a": "<<",
    "\u226b": ">>",
    "\u2192": "->",
    "\u2190": "<-",
    "\u2194": "<->",
    "\u21d2": "=>",
    "\u00a9": "(c)",
    "\u00a7": "S",  # section sign (marker)
    "\u2020": "+",  # dagger used as footnote marker
    "\u221a": "sqrt",
    "\u2229": "intersect",
    "\u2208": "in",
    "\u2295": "(+)",
    # Greek (science docs)
    "\u0394": "Delta",
    "\u03a3": "Sigma",
    "\u03c3": "sigma",
    "\u03c7": "chi",
    # box drawing / blocks -> ASCII stand-ins
    "\u2500": "-",
    "\u2550": "=",
    "\u2588": "#",
    # check / cross / warning (UI/docs)
    "\u2713": "[OK]",
    "\u2705": "[OK]",
    "\u274c": "[X]",
    "\u26a0": "!",
    "\u26a1": "!",
    "\u23f3": "...",
    "\ufe0f": "",  # variation selector
    # emoji / dingbats used as section markers in UI + a few scripts/docs
    "\U0001f52d": "[telescope]",
    "\U0001f50d": "[search]",
    "\U0001f50e": "[search]",
    "\U0001f52c": "[microscope]",
    "\U0001f4cb": "[clipboard]",
    "\U0001f4be": "[save]",
    "\U0001f4c2": "[folder]",
    "\U0001f4c4": "[page]",
    "\U0001f4c8": "[chart]",
    "\U0001f4ca": "[chart]",
    "\U0001f4d0": "[tri]",
    "\U0001f4e5": "[inbox]",
    "\U0001f4f7": "[camera]",
    "\U0001f504": "[refresh]",
    "\U0001f5fa": "[map]",
    "\U0001f680": "[rocket]",
    "\U0001f31f": "[star]",
    "\U0001f3af": "[target]",
    "\U0001f914": "[think]",
    "\U0001f9ea": "[test]",
    "\U0001f534": "[red]",
    "\U0001f535": "[blue]",
    "\U0001f7e0": "[orange]",
    "\U0001f7e1": "[yellow]",
    "\U0001f7e2": "[green]",
    "\U0001f7e3": "[purple]",
    "\U0001f6a8": "[alert]",
    "\u2753": "?",
    "\u2728": "*",
    "\u2717": "x",
    "\u2795": "+",
    "\u2605": "*",
    "\u25b6": ">",
    "\u25cf": "*",
    "\u26aa": "o",
    "\u2139": "i",
    "\u2022": "*",
    "\u2021": "++",
    "\u2191": "^",
    "\u2193": "v",
    "\u2197": "/",
    "\u23f8": "||",
    "\u2615": "[coffee]",
    # more Greek / math
    "\u03b1": "alpha",
    "\u03b2": "beta",
    "\u03b3": "gamma",
    "\u03b4": "delta",
    "\u03b5": "epsilon",
    "\u03b7": "eta",
    "\u03c0": "pi",
    "\u03c1": "rho",
    "\u03c9": "omega",
    "\u2209": "not-in",
    "\u2227": "and",
    "\u2273": ">=",
    "\u2287": "supset",
    "\u2079": "^9",
    "\u2082": "_2",
    "\u00bd": "1/2",
    "\u00b4": "'",
    "\u00ab": "<<",
    "\u00bb": ">>",
    "\u00a6": "|",
    "\u00e2": "a",
    "\u00f1": "n",
    "\u00c3": "A",
    "\u20ac": "EUR",
    # extra Central-European letters seen in citations / ledgers
    "\u0155": "r",
    "\u0106": "C",
    "\u0107": "c",
    "\u0141": "L",
    "\u0142": "l",
    "\u0143": "N",
    "\u0144": "n",
    "\u0150": "O",
    "\u0151": "o",
    "\u015a": "S",
    "\u015b": "s",
    "\u0170": "U",
    "\u0171": "u",
    "\u0179": "Z",
    "\u017a": "z",
    # replacement / BOM
    "\ufffd": "-",
    "\ufeff": "",
    # superscript minus (e.g. 10^-3 style leftovers)
    "\u207b": "-",
}

# Project convention: Czech/Slovak text is ASCII-without-diacritics.
_CZECH_FOLD = {
    "\u00e1": "a",
    "\u00c1": "A",
    "\u00e4": "a",
    "\u00c4": "A",
    "\u010d": "c",
    "\u010c": "C",
    "\u010f": "d",
    "\u010e": "D",
    "\u00e9": "e",
    "\u00c9": "E",
    "\u011b": "e",
    "\u011a": "E",
    "\u00ed": "i",
    "\u00cd": "I",
    "\u013a": "l",
    "\u0139": "L",
    "\u013e": "l",
    "\u013d": "L",
    "\u0148": "n",
    "\u0147": "N",
    "\u00f3": "o",
    "\u00d3": "O",
    "\u00f4": "o",
    "\u00d4": "O",
    "\u0159": "r",
    "\u0158": "R",
    "\u0161": "s",
    "\u0160": "S",
    "\u0165": "t",
    "\u0164": "T",
    "\u00fa": "u",
    "\u00da": "U",
    "\u016f": "u",
    "\u016e": "U",
    "\u00fd": "y",
    "\u00dd": "Y",
    "\u017e": "z",
    "\u017d": "Z",
}
CHAR_MAP.update(_CZECH_FOLD)


def _tracked_text_files(repo_root: Path) -> list[Path]:
    out = subprocess.check_output(
        ["git", "ls-files", "-z"],
        cwd=repo_root,
    )
    paths: list[Path] = []
    for raw in out.split(b"\0"):
        if not raw:
            continue
        rel = raw.decode("utf-8", errors="surrogateescape")
        if rel.startswith("Archive/") or rel.startswith("Archive\\"):
            continue
        p = Path(rel)
        if p.suffix.lower() in EXTS or p.name in SPECIAL_NAMES:
            paths.append(p)
    return paths


def decode_text(data: bytes) -> tuple[str, str]:
    """Return (text, codec_used)."""
    try:
        return data.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        # cp1252 leaves 0x81/0x8D/0x8F/0x90/0x9D undefined -> U+FFFD
        return data.decode("cp1252", errors="replace"), "cp1252"


def transmute(text: str) -> tuple[str | None, set[str]]:
    """Return (ascii_text or None if STOP, unmapped_chars)."""
    unmapped: set[str] = set()
    parts: list[str] = []
    for ch in text:
        o = ord(ch)
        if o < 0x80:
            parts.append(ch)
            continue
        if ch in CHAR_MAP:
            parts.append(CHAR_MAP[ch])
            continue
        unmapped.add(ch)
    if unmapped:
        return None, unmapped
    return "".join(parts), set()


def _py_syntax_ok(source: str, rel: Path) -> bool:
    if rel.suffix.lower() != ".py":
        return True
    try:
        compile(source, rel.as_posix(), "exec")
        return True
    except SyntaxError:
        return False


def process_file(
    repo_root: Path, rel: Path, *, check_only: bool
) -> tuple[str, int, set[str]]:
    """
    Returns (status, n_replaced, unmapped).
    status: 'ascii' | 'rewritten' | 'would_rewrite' | 'stop' | 'error'
    """
    path = repo_root / rel
    try:
        data = path.read_bytes()
    except OSError as exc:
        return f"error:{exc}", 0, set()
    if not any(b >= 0x80 for b in data):
        return "ascii", 0, set()
    text, _codec = decode_text(data)
    n_non = sum(1 for ch in text if ord(ch) >= 0x80)
    new, unmapped = transmute(text)
    if unmapped:
        return "stop", n_non, unmapped
    assert new is not None
    # Refuse to write a .py file that would not compile (quote folding hazard).
    if not _py_syntax_ok(new, rel):
        return "stop", n_non, {"<syntax-after-fold>"}
    new_bytes = new.encode("ascii")
    if new_bytes == data:
        return "ascii", 0, set()
    if check_only:
        return "would_rewrite", n_non, set()
    path.write_bytes(new_bytes)
    return "rewritten", n_non, set()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check",
        action="store_true",
        help="Report only; do not write files",
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Repo root (default: git rev-parse --show-toplevel)",
    )
    args = ap.parse_args(argv)
    if args.root is None:
        root = Path(
            subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"], text=True
            ).strip()
        )
    else:
        root = args.root.resolve()

    rewritten: list[tuple[str, int]] = []
    would: list[tuple[str, int]] = []
    stop: list[tuple[str, list[str]]] = []
    errors: list[str] = []

    for rel in _tracked_text_files(root):
        status, n, unmapped = process_file(root, rel, check_only=args.check)
        if status == "rewritten":
            rewritten.append((rel.as_posix(), n))
        elif status == "would_rewrite":
            would.append((rel.as_posix(), n))
        elif status == "stop":
            codes = sorted(f"U+{ord(c):04X}" for c in unmapped)
            stop.append((rel.as_posix(), codes))
        elif status.startswith("error"):
            errors.append(f"{rel.as_posix()}: {status}")

    mode = "CHECK" if args.check else "WRITE"
    print(f"ascii_migrate [{mode}] root={root}")
    target = would if args.check else rewritten
    print(f"migrated_or_would={len(target)}")
    for path, n in sorted(target):
        print(f"  {path}  nonascii_chars={n}")
    print(f"stop={len(stop)}")
    for path, codes in sorted(stop):
        print(f"  {path}  unmapped={','.join(codes)}")
    if errors:
        print(f"errors={len(errors)}")
        for e in errors:
            print(f"  {e}")

    # Non-zero if STOP or errors (caller / CI can decide). Check mode also
    # returns 1 when work remains (would_rewrite or stop).
    if errors or stop:
        return 2
    if args.check and would:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
