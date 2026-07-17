"""Guard: docs/ holds living documentation only.

Milan's rule (DOCS-CLEANUP, 2026-07-17): docs/ contains ONLY documentation of how
the project is designed, set up, and operated -- living state/process/decision docs
plus specs. Audit/investigation/result artifacts belong in dev/results/. No
subdirectories, no CURSOR_* working documents, markdown only.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS = REPO_ROOT / "docs"


def test_docs_dir_exists() -> None:
    assert DOCS.is_dir(), f"docs/ not found at {DOCS}"


def test_docs_has_no_subdirectories() -> None:
    subdirs = sorted(p.name for p in DOCS.iterdir() if p.is_dir())
    assert not subdirs, (
        "docs/ must contain no subdirectories (figures/samples/results belong in "
        f"dev/results/); found: {subdirs}"
    )


def test_docs_contains_only_markdown() -> None:
    non_md = sorted(
        p.name for p in DOCS.iterdir() if p.is_file() and p.suffix.lower() != ".md"
    )
    assert not non_md, f"docs/ must contain only *.md files; found non-markdown: {non_md}"


def test_docs_has_no_cursor_prefixed_files() -> None:
    offenders = sorted(
        p.name for p in DOCS.iterdir() if p.is_file() and p.name.startswith("CURSOR_")
    )
    assert not offenders, (
        f"CURSOR_* working documents belong in dev/results/, not docs/; found: {offenders}"
    )
