"""Tests for scripts/provenance_guard.py (PROVENANCE-GUARD)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.provenance_guard import (
    ProvenanceGuardError,
    assert_stamped,
    is_stamped,
    stamp_output_meta,
)

LEDGER_PATH = Path(__file__).resolve().parents[1] / "validation" / "VYVAR_VALIDATION_LEDGER.json"


def test_refuses_missing_provenance(tmp_path: Path) -> None:
    phot = tmp_path / "photometry"
    phot.mkdir()
    (phot / "pipeline_meta.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ProvenanceGuardError, match="PROVENANCE-GUARD REFUSE"):
        assert_stamped(phot, draft_id=426, setup="i_70_4", allow_unstamped=False)


def test_allow_unstamped_override(tmp_path: Path) -> None:
    phot = tmp_path / "photometry"
    phot.mkdir()
    (phot / "pipeline_meta.json").write_text("{}", encoding="utf-8")
    res = assert_stamped(phot, draft_id=426, setup="i_70_4", allow_unstamped=True)
    assert res["provenance_unstamped"] is True
    assert res["stamped"] is False


def test_accepts_stamped_meta(tmp_path: Path) -> None:
    phot = tmp_path / "photometry"
    phot.mkdir()
    meta = {
        "provenance": {
            "git_hash": "c8d6e8037978cbdc77911eba45c6a65ee2c1920b",
            "stamped_at_utc": "2026-07-13T12:00:00+00:00",
        }
    }
    (phot / "pipeline_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    assert is_stamped(phot)
    res = assert_stamped(phot, draft_id=426, setup="i_70_4")
    assert res["stamped"] is True
    assert res["provenance_unstamped"] is False


def test_stamp_output_meta_flags_unstamped() -> None:
    out = stamp_output_meta({"x": 1}, {"stamped": False, "provenance_unstamped": True, "git_hash": None})
    assert out["provenance_unstamped"] is True
    assert out["provenance_guard"]["provenance_unstamped"] is True


def test_ledger_includes_vl_provenance() -> None:
    ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    ids = {it["id"] for it in ledger["items"]}
    assert "VL-PROVENANCE" in ids
