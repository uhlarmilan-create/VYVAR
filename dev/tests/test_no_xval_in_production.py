"""Invariant: SEP/xval harness modules stay offline - not imported by production code."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

_SKIP_DIR_NAMES = frozenset(
    {
        "dev",
        "tests",
        "scripts",
        "sandbox",
        "tmp",
        "Archive",
        ".venv",
        "__pycache__",
        "orchestrator",
        ".git",
        ".cursor",
    }
)

_SKIP_FILE_PREFIXES = ("xval_",)
_SKIP_FILE_NAMES = frozenset({"validate_lc_crossval.py"})

_IMPORT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^\s*from\s+xval_run\b"), "xval_run"),
    (re.compile(r"^\s*import\s+xval_run\b"), "xval_run"),
    (re.compile(r"^\s*from\s+xval_harness_core\b"), "xval_harness_core"),
    (re.compile(r"^\s*import\s+xval_harness_core\b"), "xval_harness_core"),
    (
        re.compile(r"^\s*from\s+xval_harness_core\s+import\b.*\bassign_sep_confidence\b"),
        "assign_sep_confidence",
    ),
    (
        re.compile(r"^\s*import\s+xval_harness_core\s+.*\bassign_sep_confidence\b"),
        "assign_sep_confidence",
    ),
)


def _is_production_py(path: Path) -> bool:
    if path.suffix != ".py":
        return False
    if path.name in _SKIP_FILE_NAMES:
        return False
    if any(path.name.startswith(p) for p in _SKIP_FILE_PREFIXES):
        return False
    rel = path.relative_to(ROOT)
    if any(part in _SKIP_DIR_NAMES for part in rel.parts):
        return False
    return True


def _production_py_files() -> list[Path]:
    return sorted(p for p in ROOT.rglob("*.py") if _is_production_py(p))


def _forbidden_imports_in_file(path: Path) -> list[tuple[int, str, str]]:
    hits: list[tuple[int, str, str]] = []
    text = path.read_text(encoding="utf-8")
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        for pattern, symbol in _IMPORT_PATTERNS:
            if pattern.search(line):
                hits.append((lineno, symbol, line.rstrip()))
    return hits


def test_no_xval_imports_in_production_modules() -> None:
    """Production must not import xval_run, xval_harness_core, or assign_sep_confidence."""
    violations: list[str] = []
    for path in _production_py_files():
        for lineno, symbol, line in _forbidden_imports_in_file(path):
            rel = path.relative_to(ROOT).as_posix()
            violations.append(f"{rel}:{lineno} imports {symbol}: {line}")
    assert not violations, "SEP/xval must stay offline-only:\n" + "\n".join(violations)
