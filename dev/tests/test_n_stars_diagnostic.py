# -*- coding: ascii -*-
"""FRAME-QC-PARITY-02C: n_stars diagnostic (D-NSTARS-DIAG-01). Synthetic only."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src_py"
sys.path.insert(0, str(SRC))

import pipeline  # noqa: E402
import pipeline_constants  # noqa: E402
from pipeline_calibrate import emit_n_stars_diagnostic, plan_n_stars_diagnostic  # noqa: E402
from pipeline_constants import N_STARS_DIAG_K  # noqa: E402


def _row(src: str, n: float, status: str = "ok") -> dict:
    return {
        "src": src,
        "dst": src,
        "status": status,
        "n_stars_detected": n,
    }


def test_n_stars_diag_k_importable_from_leaf_and_facade() -> None:
    assert N_STARS_DIAG_K == 5.0
    assert pipeline_constants.N_STARS_DIAG_K == 5.0
    assert pipeline.N_STARS_DIAG_K == 5.0
    assert pipeline.N_STARS_DIAG_K is pipeline_constants.N_STARS_DIAG_K


def test_plan_warns_both_sides_keeps_inliers() -> None:
    rows = [
        _row(f"BO_CVn_Light_{i:03d}.fits", n)
        for i, n in enumerate(
            [90, 92, 94, 96, 98, 100, 102, 104, 106, 108],
            start=1,
        )
    ]
    rows.append(_row("BO_CVn_Light_050.fits", 50.0))
    rows.append(_row("BO_CVn_Light_200.fits", 200.0))
    rows.append(_row("BO_CVn_Light_099.fits", 10.0, status="rejected"))
    df = pd.DataFrame(rows)
    plan = plan_n_stars_diagnostic(df)
    assert plan["bounds_undefined"] is False
    assert plan["k"] == 5.0
    frames = {rec["frame"]: rec for rec in plan["frames"]}
    assert "Light_050" in frames and frames["Light_050"]["side"] == "low"
    assert "Light_200" in frames and frames["Light_200"]["side"] == "high"
    assert plan["n_low"] == 1
    assert plan["n_high"] == 1
    inliers = {f"Light_{i:03d}" for i in range(1, 11)}
    assert inliers.isdisjoint(frames)
    assert "Light_099" not in frames


def _frameqc_texts(caplog) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if "[FRAME-QC] n_stars diagnostic:" in r.getMessage()
    ]


def test_plan_mad_zero_emits_summary_only(caplog) -> None:
    rows = [
        _row(f"BO_CVn_Light_{i:03d}.fits", 98.0) for i in range(1, 8)
    ]
    df = pd.DataFrame(rows)
    plan = plan_n_stars_diagnostic(df)
    assert plan["bounds_undefined"] is True
    assert plan["frames"] == []
    assert plan["median"] == 98.0
    with caplog.at_level(logging.WARNING, logger="pipeline"):
        emit_n_stars_diagnostic(df)
    texts = _frameqc_texts(caplog)
    assert any(
        "MAD=0; bounds undefined; no per-frame warnings" in t for t in texts
    )
    assert not any("outside" in t for t in texts)


def test_emit_per_frame_and_summary(caplog) -> None:
    rows = [
        _row(f"BO_CVn_Light_{i:03d}.fits", n)
        for i, n in enumerate(
            [90, 92, 94, 96, 98, 100, 102, 104, 106, 108],
            start=1,
        )
    ]
    rows.append(_row("BO_CVn_Light_050.fits", 50.0))
    rows.append(_row("BO_CVn_Light_200.fits", 200.0))
    df = pd.DataFrame(rows)
    with caplog.at_level(logging.WARNING, logger="pipeline"):
        emit_n_stars_diagnostic(df)
    texts = _frameqc_texts(caplog)
    per = [t for t in texts if "outside" in t and "frame kept" in t]
    assert len(per) == 2
    assert any("Light_050" in t and "n=50" in t for t in per)
    assert any("Light_200" in t and "n=200" in t for t in per)
    assert any("n_low=1 n_high=1" in t for t in texts)
