"""Tests for epsf_frame_accounting and INV-PSF-FRAME-01 (EPSF-VALID-02 F2)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src_py"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from epsf_frame_accounting import (  # noqa: E402
    DEFAULT_ZERO_OK_FRAME_FRACTION_FAIL,
    enforce_inv_psf_frame_01,
    summarize_epsf_frame_job,
)
from invariants_runtime import InvariantViolation  # noqa: E402


def _rec(name: str, n_ok: int, *, exc: str | None = None) -> dict:
    return {
        "frame_name": name,
        "frame_index": 0,
        "n_fit": 10,
        "n_ok": n_ok,
        "exception_class": exc,
        "exception_message": "boom" if exc else None,
        "traceback_tail": "tail" if exc else None,
    }


def test_exception_message_recorded_in_summary() -> None:
    summary = summarize_epsf_frame_job([_rec("f0", 0, exc="RuntimeError")])
    rec = summary["per_frame_records"][0]
    assert rec["exception_class"] == "RuntimeError"
    assert rec["exception_message"] == "boom"


def test_invariant_trips_above_20_percent() -> None:
    records = [_rec(f"f{i}", 0, exc="ValueError") for i in range(5)]
    records.extend([_rec(f"ok{i}", 5) for i in range(5)])
    summary = summarize_epsf_frame_job(records)
    assert summary["frames_with_zero_ok_fraction"] == 0.5
    with pytest.raises(InvariantViolation):
        enforce_inv_psf_frame_01(summary, fail_fraction=DEFAULT_ZERO_OK_FRAME_FRACTION_FAIL)


def test_invariant_passes_at_or_below_20_percent() -> None:
    records = [_rec("bad", 0, exc="ValueError")]
    records.extend([_rec(f"ok{i}", 5) for i in range(9)])
    summary = summarize_epsf_frame_job(records)
    assert summary["frames_with_zero_ok_fraction"] == 0.1
    policy = enforce_inv_psf_frame_01(summary, fail_fraction=DEFAULT_ZERO_OK_FRAME_FRACTION_FAIL)
    assert policy == "WARN"
