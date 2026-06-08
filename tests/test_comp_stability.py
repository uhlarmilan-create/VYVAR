"""Tests for check_comparison_stability slope filter."""

from __future__ import annotations

import numpy as np

from photometry_core import check_comparison_stability


def _make_flat_ensemble(n_comp: int, n_frames: int, *, trend_cid: str | None, slope_mag_per_day: float) -> tuple[dict, dict]:
    """Several stable comps + one optional trending comp (linear in BJD days)."""
    comp_lc: dict[str, np.ndarray] = {}
    comp_bjd: dict[str, np.ndarray] = {}
    bjd = np.linspace(2459000.0, 2459000.5, n_frames)
    rng = np.random.default_rng(0)
    for i in range(n_comp):
        cid = trend_cid if (trend_cid is not None and i == n_comp - 1) else f"stable_{i}"
        noise = 0.002 * rng.standard_normal(n_frames)
        if cid == trend_cid:
            mag = 12.0 + slope_mag_per_day * (bjd - bjd[0]) + noise
        else:
            mag = 12.0 + noise
        comp_lc[cid] = mag.astype(float)
        comp_bjd[cid] = bjd.copy()
    return comp_lc, comp_bjd


def test_slope_filter_excludes_trending_comp():
    """+6 mmag/hr slope comp excluded at threshold 5.0."""
    # 6 mmag/hr = 0.006 mag/hr = 0.144 mag per BJD day
    comp_lc, comp_bjd = _make_flat_ensemble(5, 30, trend_cid="trend", slope_mag_per_day=0.144)
    q = check_comparison_stability(
        comp_lc,
        comp_bjd=comp_bjd,
        max_comp_slope_mmag_hr=5.0,
        outlier_sigma=99.0,
    )
    assert q["trend"]["quality"] in ("excluded", "suspect")
    assert q["trend"].get("slope_mmag_hr", 0) > 5.0


def test_slope_filter_keeps_stable_comp():
    """+1 mmag/hr slope comp passes threshold 5.0."""
    comp_lc, comp_bjd = _make_flat_ensemble(5, 30, trend_cid="trend", slope_mag_per_day=0.024)
    q = check_comparison_stability(
        comp_lc,
        comp_bjd=comp_bjd,
        max_comp_slope_mmag_hr=5.0,
        outlier_sigma=99.0,
    )
    assert q["trend"]["quality"] == "good"


def test_slope_filter_disabled_at_high_threshold():
    """+6 mmag/hr comp passes when threshold=99.0."""
    comp_lc, comp_bjd = _make_flat_ensemble(5, 30, trend_cid="trend", slope_mag_per_day=0.144)
    q = check_comparison_stability(
        comp_lc,
        comp_bjd=comp_bjd,
        max_comp_slope_mmag_hr=99.0,
        outlier_sigma=99.0,
    )
    assert q["trend"]["quality"] == "good"


def test_slope_filter_requires_min_20_frames():
    """15 frames: slope filter skipped even with huge trend."""
    comp_lc, comp_bjd = _make_flat_ensemble(5, 15, trend_cid="trend", slope_mag_per_day=0.05)
    q = check_comparison_stability(
        comp_lc,
        comp_bjd=comp_bjd,
        max_comp_slope_mmag_hr=5.0,
        outlier_sigma=99.0,
    )
    assert q["trend"]["quality"] == "good"
    assert "slope_mmag_hr" not in q["trend"]
