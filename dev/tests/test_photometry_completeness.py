"""Photometry completeness gate (truncation-as-success guard)."""
from __future__ import annotations

from pathlib import Path

import pytest

from night_run import audit_photometry_completeness

_ROOT = Path(__file__).resolve().parents[2]
_DRAFT_385 = _ROOT / "Archive" / "Drafts" / "draft_000385"
_DRAFT_386 = _ROOT / "Archive" / "Drafts" / "draft_000386"


@pytest.mark.skipif(not (_DRAFT_385 / "platesolve").is_dir(), reason="draft_000385 not present")
def test_truncated_draft_385_fails_completeness_gate():
    """draft_385 R was 69/373 - must not pass the gate."""
    out = _DRAFT_385 / "platesolve" / "R_20_2" / "photometry"
    audit = audit_photometry_completeness(out)
    assert audit["n_active_targets"] == 373
    assert audit["n_summary_rows"] == 69
    assert audit["ok"] is False


@pytest.mark.skipif(not (_DRAFT_386 / "platesolve").is_dir(), reason="draft_000386 not present")
def test_full_draft_386_passes_completeness_gate():
    for setup in ("B_20_2", "V_20_2", "R_20_2", "L_20_2"):
        out = _DRAFT_386 / "platesolve" / setup / "photometry"
        audit = audit_photometry_completeness(out)
        assert audit["ok"] is True, f"{setup}: {audit}"
