"""Fix 3 (Round 1): completeness gate distinguishes honest-unmeasurable from truncation.

The gate must NOT fail a run where the only missing targets are below the achieved
detection depth (honest RED / undetected / too faint), but MUST still fail genuine
silent truncation (a cut-short process that drops measurable, bright, on-frame targets).
"""
from __future__ import annotations

import pandas as pd

from night_run import audit_photometry_completeness


def _write(tmp_path, active_rows, summary_ids):
    at = pd.DataFrame(active_rows)
    at.to_csv(tmp_path / "active_targets.csv", index=False)
    sm = pd.DataFrame({"catalog_id": list(summary_ids), "lc_rms": [0.05] * len(summary_ids)})
    sm.to_csv(tmp_path / "photometry_summary.csv", index=False)
    return tmp_path


def test_honest_unmeasurable_passes(tmp_path):
    """draft_413 g-like: 19 measured (10.9-13.8 mag), 3 missing are far below depth (16-17)."""
    measured = [{"catalog_id": f"M{i}", "mag": 10.9 + i * 0.15} for i in range(19)]  # up to ~13.6
    faint_missing = [
        {"catalog_id": "F1", "mag": 16.64},
        {"catalog_id": "F2", "mag": 16.92},
        {"catalog_id": "F3", "mag": 17.29},
    ]
    _write(tmp_path, measured + faint_missing, [r["catalog_id"] for r in measured])

    out = audit_photometry_completeness(tmp_path)
    assert out["n_active_targets"] == 22
    assert out["n_summary_rows"] == 19
    assert out["n_unmeasurable_missing"] == 3
    assert out["n_measurable_active"] == 19
    assert out["measurable_ratio"] == 1.0
    assert out["ok"] is True
    # Raw ratio would have failed the old gate.
    assert out["ratio"] < 0.90


def test_truncation_still_fails(tmp_path):
    """draft_383-like silent truncation: many missing targets are bright (measurable)."""
    measured = [{"catalog_id": f"M{i}", "mag": 11.0 + i * 0.02} for i in range(69)]  # up to ~12.4
    # 304 missing, spanning bright-to-mid (all <= achieved depth -> measurable misses).
    missing = [{"catalog_id": f"X{i}", "mag": 11.5 + (i % 10) * 0.05} for i in range(304)]
    _write(tmp_path, measured + missing, [r["catalog_id"] for r in measured])

    out = audit_photometry_completeness(tmp_path)
    assert out["n_active_targets"] == 373
    assert out["n_summary_rows"] == 69
    assert out["n_measurable_missing"] == 304
    assert out["measurable_ratio"] < 0.90
    assert out["ok"] is False


def test_nothing_measured_fails(tmp_path):
    """No depth can be asserted -> every miss is measurable -> fail (no false success)."""
    active = [{"catalog_id": f"M{i}", "mag": 12.0 + i * 0.1} for i in range(10)]
    _write(tmp_path, active, [])  # zero summary rows
    out = audit_photometry_completeness(tmp_path)
    assert out["n_summary_rows"] == 0
    assert out["ok"] is False


def test_mixed_bright_and_faint_missing_fails(tmp_path):
    """If a missing target is brighter than achieved depth, the run must still fail."""
    measured = [{"catalog_id": f"M{i}", "mag": 11.0 + i * 0.2} for i in range(18)]  # up to ~14.4
    missing = [
        {"catalog_id": "FAINT", "mag": 17.0},   # honest unmeasurable
        {"catalog_id": "BRIGHT", "mag": 12.5},  # measurable miss -> truncation signal
    ]
    _write(tmp_path, measured + missing, [r["catalog_id"] for r in measured])
    out = audit_photometry_completeness(tmp_path)
    assert out["n_unmeasurable_missing"] == 1
    assert out["n_measurable_missing"] == 1
    # 18 / (18+1) = 94.7% >= 90% -> still ok here; tighten with explicit min_ratio.
    out2 = audit_photometry_completeness(tmp_path, min_ratio=0.99)
    assert out2["ok"] is False
