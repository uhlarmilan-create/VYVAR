"""Guard: docs/ holds living documentation only.

Milan's rule (DOCS-CLEANUP, 2026-07-17): docs/ contains ONLY documentation of how
the project is designed, set up, and operated -- living state/process/decision docs
plus specs. Audit/investigation/result artifacts belong in dev/results/. No
subdirectories, no CURSOR_* working documents.

Allowed file types: Markdown (*.md) plus PDF guides (*.pdf) that a committed builder
under dev/tools/docs_pdf/ regenerates (DOCS-PDF, 2026-07-18).
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


_ALLOWED_SUFFIXES = {".md", ".pdf"}


def test_docs_contains_only_allowed_types() -> None:
    # Markdown for living docs; PDF for the CZ guides that a committed builder under
    # dev/tools/docs_pdf/ regenerates (DOCS-PDF). Anything else is a stray artifact.
    disallowed = sorted(
        p.name
        for p in DOCS.iterdir()
        if p.is_file() and p.suffix.lower() not in _ALLOWED_SUFFIXES
    )
    assert not disallowed, (
        "docs/ may contain only *.md and *.pdf files; found: " f"{disallowed}"
    )


def test_docs_pdfs_have_a_committed_builder() -> None:
    # Every PDF in docs/ must be regenerable from a committed builder, so binaries are
    # never orphaned. Presence of the builder dir + a script per PDF is the contract.
    builders_dir = REPO_ROOT / "dev" / "tools" / "docs_pdf"
    pdfs = sorted(p.name for p in DOCS.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")
    if not pdfs:
        return
    assert builders_dir.is_dir(), (
        f"docs/ has PDFs {pdfs} but no builder dir at {builders_dir}"
    )
    builder_scripts = list(builders_dir.glob("build_*.py"))
    assert builder_scripts, (
        f"docs/ has PDFs {pdfs} but no build_*.py under {builders_dir}"
    )


def test_docs_has_no_cursor_prefixed_files() -> None:
    offenders = sorted(
        p.name for p in DOCS.iterdir() if p.is_file() and p.name.startswith("CURSOR_")
    )
    assert not offenders, (
        f"CURSOR_* working documents belong in dev/results/, not docs/; found: {offenders}"
    )
