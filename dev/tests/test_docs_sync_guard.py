# -*- coding: ascii -*-
"""DOCS-SYNC machine guard: FLOW facts vs config/code, no SPECs in docs/.

If these go red, documentation drifted under a code change: update the FLOW
builder prose + flow_doc_facts.py, regenerate docs/VYVAR_FLOW_CZ.pdf, and
keep *_SPEC.md under dev/results/specs/ only.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS = REPO_ROOT / "docs"
FACTS_DIR = REPO_ROOT / "dev" / "tools" / "docs_pdf"


def _ensure_paths() -> None:
    for p in (REPO_ROOT / "src_py", FACTS_DIR):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


def test_no_spec_files_in_docs() -> None:
    offenders = sorted(p.name for p in DOCS.glob("*SPEC*.md"))
    assert offenders == [], (
        "*_SPEC.md files belong in dev/results/specs/, not docs/; found: "
        f"{offenders}"
    )


def test_flow_doc_config_facts() -> None:
    _ensure_paths()
    from config import load_config_json  # noqa: PLC0415
    from flow_doc_facts import DOC_CONFIG_FACTS  # noqa: PLC0415

    cfg = load_config_json(REPO_ROOT)
    msg = (
        "config default changed under FLOW doc: update builder prose + "
        "flow_doc_facts.py, regenerate docs/VYVAR_FLOW_CZ.pdf"
    )
    for key, expected in DOC_CONFIG_FACTS.items():
        assert key in cfg, f"{msg} (missing key {key!r})"
        assert cfg[key] == expected, (
            f"{msg} ({key!r}: config={cfg[key]!r} facts={expected!r})"
        )


def test_flow_doc_functions_exist() -> None:
    _ensure_paths()
    from flow_doc_facts import DOC_FUNCTIONS  # noqa: PLC0415

    for rel, symbol in DOC_FUNCTIONS:
        path = REPO_ROOT / rel
        assert path.is_file(), f"FLOW-doc module missing: {rel}"
        text = path.read_text(encoding="utf-8")
        ok = f"def {symbol}" in text or f"class {symbol}" in text
        assert ok, (
            f"FLOW-doc symbol {symbol!r} not found as def/class in {rel}; "
            "update builder prose + flow_doc_facts.py, regenerate "
            "docs/VYVAR_FLOW_CZ.pdf"
        )


def test_flow_pdf_exists() -> None:
    # v3 full-depth edition is ~131 kB / 36 pp; v2 placeholder was ~21 kB.
    # Threshold 100 kB catches accidental revert to the short edition.
    pdf = DOCS / "VYVAR_FLOW_CZ.pdf"
    assert pdf.is_file(), f"missing {pdf}"
    size = pdf.stat().st_size
    assert size > 100_000, (
        f"docs/VYVAR_FLOW_CZ.pdf too small ({size} bytes); regenerate via "
        "python dev/tools/docs_pdf/build_flow_doc.py"
    )
