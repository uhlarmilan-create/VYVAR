"""
tests/test_alg_functions.py
ALG-2/3/4/5 function audit tests — created 2026-05-21 overnight

Run with: python -m pytest tests/test_alg_functions.py -v
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from photometry_core import (
    check_comparison_stability,
    democratic_detrend_lc,
    ensemble_normalize,
    pytics_iterative_weights,
    savgol_detrend_lc,
    temporal_bin_comp_lc,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normal_flags(n: int) -> list[str]:
    return ["normal"] * n


def _make_comp_lc(
    n_comp: int,
    n_frames: int,
    *,
    noise: float = 0.01,
    shared_trend: np.ndarray | None = None,
    variable_idx: int | None = None,
    variable_extra: float = 0.05,
    seed: int = 42,
) -> tuple[dict[str, np.ndarray], list[str]]:
    rng = np.random.default_rng(seed)
    cids = [f"comp_{i}" for i in range(n_comp)]
    comp_lc: dict[str, np.ndarray] = {}
    for i, cid in enumerate(cids):
        arr = 12.0 + noise * rng.standard_normal(n_frames)
        if shared_trend is not None:
            arr = arr + shared_trend
        if variable_idx is not None and i == variable_idx:
            arr = arr + variable_extra * rng.standard_normal(n_frames)
        comp_lc[cid] = arr.astype(float)
    return comp_lc, cids


def _comp_quality_all_good(cids: list[str]) -> dict[str, dict]:
    return {cid: {"quality": "good", "rms_p2p": 0.005} for cid in cids}


def _ensemble_scatter(comp_lc: dict[str, np.ndarray], cids: list[str]) -> float:
    matrix = np.column_stack([comp_lc[cid] for cid in cids])
    ensemble = np.nanmedian(matrix, axis=1)
    residuals = matrix - ensemble[:, np.newaxis]
    return float(np.nanmean(np.nanstd(residuals, axis=0, ddof=1)))


def _broeg_weights(rms_map: dict[str, float], cids: list[str]) -> np.ndarray:
    rms = np.array([float(rms_map[cid]) for cid in cids], dtype=float)
    rms = np.where(rms > 0, rms, 1.0)
    w = 1.0 / (rms**2)
    return w / w.sum()


# ---------------------------------------------------------------------------
# ALG-2: Savitzky-Golay detrending
# ---------------------------------------------------------------------------

def test_alg2_sg_removes_trend():
    """SG detrend should remove a linear trend while preserving signal."""
    n = 60
    bjd = np.linspace(2459000.0, 2459000.5, n)
    trend = np.linspace(0.0, 0.4, n)
    signal = 0.04 * np.sin(np.linspace(0, 6 * np.pi, n))
    mag = 12.0 + trend + signal
    flags = _normal_flags(n)

    out = savgol_detrend_lc(
        mag, bjd, flags, window_frac=0.35, polyorder=2, min_points=10, enabled=True
    )

    trend_rms_before = float(np.nanstd(trend))
    residual = out - (12.0 + signal)
    trend_rms_after = float(np.nanstd(residual))
    signal_rms = float(np.nanstd(out - 12.0))

    assert trend_rms_after < 0.5 * trend_rms_before, (
        f"trend not reduced enough: before={trend_rms_before:.4f} after={trend_rms_after:.4f}"
    )
    assert abs(signal_rms - float(np.nanstd(signal))) < 0.15, (
        f"signal distorted: signal_rms={signal_rms:.4f} expected~{np.nanstd(signal):.4f}"
    )


def test_alg2_sg_short_window():
    """SG with window > n_points / too few normals should not crash."""
    n = 8
    bjd = np.linspace(2459000.0, 2459000.1, n)
    mag = 12.0 + np.linspace(0, 0.2, n)
    flags = _normal_flags(n)

    out = savgol_detrend_lc(
        mag, bjd, flags, window_frac=0.9, polyorder=2, min_points=10, enabled=True
    )
    assert out.shape == mag.shape
    assert np.all(np.isfinite(out) | np.isnan(out))


def test_alg2_sg_all_nan():
    """All-NaN input should return all-NaN without exception."""
    n = 30
    mag = np.full(n, np.nan)
    bjd = np.linspace(2459000.0, 2459000.3, n)
    flags = _normal_flags(n)

    out = savgol_detrend_lc(mag, bjd, flags, enabled=True)
    assert out.shape == mag.shape
    assert np.all(np.isnan(out))


def test_alg2_sg_disabled():
    """When enabled=False, output == input."""
    n = 40
    mag = 12.0 + 0.01 * np.arange(n)
    bjd = np.linspace(2459000.0, 2459000.4, n)
    flags = _normal_flags(n)

    out = savgol_detrend_lc(mag, bjd, flags, enabled=False)
    np.testing.assert_array_equal(out, mag)


# ---------------------------------------------------------------------------
# ALG-3: Temporal binning
# ---------------------------------------------------------------------------

def test_alg3_binning_reduces_noise():
    """Binned LC should have lower comp-ensemble scatter than unbinned."""
    n_frames = 45
    comp_lc, cids = _make_comp_lc(5, n_frames, noise=0.03, seed=1)
    all_frames = pd.DataFrame({"catalog_id": cids * n_frames, "bjd": np.tile(np.arange(n_frames), 5)})

    scatter_before = _ensemble_scatter(comp_lc, cids)
    binned = temporal_bin_comp_lc(
        comp_lc, _comp_quality_all_good(cids), all_frames, window=5, enabled=True
    )
    scatter_after = _ensemble_scatter(binned, cids)

    assert scatter_after < scatter_before, (
        f"binning did not reduce scatter: before={scatter_before:.5f} after={scatter_after:.5f}"
    )
    ratio = scatter_after / scatter_before
    assert ratio < 0.95, f"expected meaningful reduction, ratio={ratio:.3f}"


def test_alg3_single_frame_bins():
    """Very short series / window<3 should equal original (passthrough)."""
    comp_lc, cids = _make_comp_lc(4, 3, noise=0.01)
    out = temporal_bin_comp_lc(
        comp_lc, _comp_quality_all_good(cids), pd.DataFrame(), window=1, enabled=True
    )
    for cid in cids:
        np.testing.assert_array_equal(out[cid], comp_lc[cid])


def test_alg3_preserves_signal():
    """Binning should preserve the mean magnitude per comp."""
    n_frames = 40
    comp_lc, cids = _make_comp_lc(5, n_frames, noise=0.02, seed=7)
    out = temporal_bin_comp_lc(
        comp_lc, _comp_quality_all_good(cids), pd.DataFrame(), window=7, enabled=True
    )
    for cid in cids:
        assert abs(float(np.nanmean(out[cid])) - float(np.nanmean(comp_lc[cid]))) < 0.02


def test_alg3_disabled():
    """When enabled=False, output == input."""
    comp_lc, cids = _make_comp_lc(5, 20)
    out = temporal_bin_comp_lc(comp_lc, _comp_quality_all_good(cids), pd.DataFrame(), enabled=False)
    for cid in cids:
        np.testing.assert_array_equal(out[cid], comp_lc[cid])


def test_alg3_fewer_than_three_comps_passthrough():
    """<3 comps → no binning (graceful passthrough)."""
    comp_lc, cids = _make_comp_lc(2, 20)
    out = temporal_bin_comp_lc(comp_lc, {}, pd.DataFrame(), window=5, enabled=True)
    for cid in cids:
        np.testing.assert_array_equal(out[cid], comp_lc[cid])


# ---------------------------------------------------------------------------
# ALG-4: Democratic Detrender
# ---------------------------------------------------------------------------

def test_alg4_democratic_removes_common_trend():
    """Democratic detrender should remove systematic airmass-like trend."""
    n = 50
    bjd = np.linspace(2459000.0, 2459000.5, n)
    airmass = 1.0 + 0.5 * np.linspace(0, 1, n)
    trend = 0.15 * (airmass - np.nanmedian(airmass))
    mag = 12.0 + trend + 0.005 * np.random.default_rng(0).standard_normal(n)
    flags = _normal_flags(n)

    mag_dem, _ = democratic_detrend_lc(
        mag, bjd, airmass, flags, window_frac=0.4, min_points=10, enabled=True
    )
    rms_before = float(np.nanstd(mag - np.nanmedian(mag)))
    rms_after = float(np.nanstd(mag_dem - np.nanmedian(mag_dem)))
    assert rms_after < 0.7 * rms_before, (
        f"trend not reduced: rms_before={rms_before:.5f} rms_after={rms_after:.5f}"
    )


def test_alg4_does_not_remove_real_signal():
    """Unique sinusoidal signal on target should be largely preserved."""
    n = 80
    bjd = np.linspace(2459000.0, 2459001.0, n)
    airmass = np.full(n, 1.2)
    unique = 0.08 * np.sin(np.linspace(0, 10 * np.pi, n))
    mag = 12.0 + unique
    flags = _normal_flags(n)

    mag_dem, _ = democratic_detrend_lc(mag, bjd, airmass, flags, window_frac=0.25, enabled=True)
    mag_dem_c = mag_dem - float(np.nanmedian(mag_dem))
    corr = float(np.corrcoef(unique, mag_dem_c)[0, 1])
    assert corr > 0.75, f"unique signal correlation too low: {corr:.3f}"


def test_alg4_minimum_points_passthrough():
    """With fewer than min_points normal frames, should fall back gracefully."""
    n = 12
    mag = 12.0 + 0.1 * np.arange(n)
    bjd = np.linspace(2459000.0, 2459000.1, n)
    flags = ["normal"] * 5 + ["saturated"] * 7

    mag_dem, err_inf = democratic_detrend_lc(
        mag, bjd, np.full(n, 1.1), flags, min_points=10, enabled=True
    )
    np.testing.assert_array_equal(mag_dem, mag)
    np.testing.assert_array_equal(err_inf, np.zeros(n))


def test_alg4_disabled():
    """When enabled=False, output == input and zero inflation."""
    n = 30
    mag = 12.0 + 0.02 * np.sin(np.linspace(0, 4, n))
    bjd = np.linspace(2459000.0, 2459000.3, n)
    flags = _normal_flags(n)

    mag_dem, err_inf = democratic_detrend_lc(
        mag, bjd, np.full(n, 1.2), flags, enabled=False
    )
    np.testing.assert_array_equal(mag_dem, mag)
    np.testing.assert_array_equal(err_inf, np.zeros(n))


# ---------------------------------------------------------------------------
# ALG-5: PyTICS iterative comp weights
# ---------------------------------------------------------------------------

def test_alg5_pytics_downweights_variable_comp():
    """PyTICS should assign higher RMS (lower weight) to a variable comp star."""
    n_frames = 50
    comp_lc, cids = _make_comp_lc(
        5, n_frames, noise=0.008, variable_idx=2, variable_extra=0.06, seed=3
    )
    comp_quality = _comp_quality_all_good(cids)
    rms_init = {cid: 0.01 for cid in cids}
    w_before = _broeg_weights(rms_init, cids)

    rms_updated = pytics_iterative_weights(
        comp_lc, comp_quality, rms_init, n_iter=5, enabled=True
    )
    w_after = _broeg_weights(rms_updated, cids)

    var_idx = 2
    median_other = float(np.median([w_after[i] for i in range(len(cids)) if i != var_idx]))
    assert w_after[var_idx] < 0.5 * median_other, (
        f"variable comp weight not reduced: w_var={w_after[var_idx]:.5f} "
        f"median_other={median_other:.5f} w_before_var={w_before[var_idx]:.5f}"
    )
    assert rms_updated[cids[var_idx]] > rms_init[cids[var_idx]] * 1.1


def test_alg5_pytics_stable_comps_equal_weights():
    """All stable comps with identical LC should get similar weights after PyTICS."""
    n_frames = 40
    rng = np.random.default_rng(11)
    base = (12.0 + 0.007 * rng.standard_normal(n_frames)).astype(float)
    cids = [f"comp_{i}" for i in range(5)]
    comp_lc = {cid: base.copy() for cid in cids}
    comp_quality = _comp_quality_all_good(cids)
    rms_init = {cid: 0.01 for cid in cids}
    rms_updated = pytics_iterative_weights(
        comp_lc, comp_quality, rms_init, n_iter=5, enabled=True
    )
    w = _broeg_weights(rms_updated, cids)
    assert float(np.std(w) / np.mean(w)) < 0.15, f"weights too spread: {w}"


def test_alg5_pytics_single_comp():
    """Fewer than 3 comps should not crash PyTICS."""
    comp_lc, cids = _make_comp_lc(2, 20)
    rms_init = {cid: 0.01 for cid in cids}
    out = pytics_iterative_weights(comp_lc, _comp_quality_all_good(cids), rms_init, enabled=True)
    assert out == rms_init


def test_alg5_disabled():
    """When enabled=False, comp_rms_map unchanged (Broeg weights unchanged)."""
    comp_lc, cids = _make_comp_lc(5, 30, variable_idx=1, variable_extra=0.1)
    rms_init = {cid: 0.01 + 0.002 * i for i, cid in enumerate(cids)}
    out = pytics_iterative_weights(
        comp_lc, _comp_quality_all_good(cids), rms_init, enabled=False
    )
    assert out == rms_init


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

def test_alg_pipeline_integration():
    """Run ALG-3 → stability → ALG-5 → ensemble → ALG-2/4 (disabled) without crash."""
    n = 55
    bjd = np.linspace(2459000.0, 2459000.6, n)
    shared = 0.03 * np.linspace(0, 1, n)
    comp_lc, cids = _make_comp_lc(6, n, noise=0.012, shared_trend=shared, seed=99)
    target_cid = "target_1"
    target_lc = comp_lc[cids[0]] + 0.05 * np.sin(np.linspace(0, 5 * np.pi, n))

    comp_lc_b = temporal_bin_comp_lc(
        comp_lc, {}, pd.DataFrame({"catalog_id": [], "bjd": []}), window=0, enabled=True
    )
    comp_quality = check_comparison_stability(
        comp_lc_b, comp_rms_map={cid: 0.01 for cid in cids}, n_comp_min=3
    )
    rms_map = pytics_iterative_weights(
        comp_lc_b, comp_quality, {cid: 0.01 for cid in cids}, enabled=True
    )
    cat_mag = {cid: 12.0 for cid in cids}
    mag_calib, delta_mag, scatter = ensemble_normalize(
        target_lc,
        comp_lc_b,
        cat_mag,
        comp_quality,
        comp_rms_map=rms_map,
        n_comp_min=3,
        n_comp_max=6,
    )
    assert mag_calib.shape == (n,)
    assert delta_mag.shape == (n,)
    assert scatter.shape == (n,)
    assert np.isfinite(mag_calib).sum() >= n - 2

    flags = _normal_flags(n)
    mag_sg = savgol_detrend_lc(mag_calib, bjd, flags, enabled=False)
    mag_dem, err_inf = democratic_detrend_lc(
        mag_sg, bjd, np.linspace(1.0, 1.5, n), flags, enabled=False
    )
    assert mag_dem.shape == (n,)
    assert err_inf.shape == (n,)


def test_alg_all_disabled_passthrough():
    """With ALG-3/5 disabled, ensemble path unchanged vs binned+pytics path on same comps."""
    n = 40
    comp_lc, cids = _make_comp_lc(5, n, seed=21)
    target_lc = comp_lc[cids[0]].copy()
    comp_quality = _comp_quality_all_good(cids)
    rms_flat = {cid: 0.01 for cid in cids}

    mag_off, d_off, _ = ensemble_normalize(
        target_lc, comp_lc, {cid: 12.0 for cid in cids}, comp_quality, comp_rms_map=rms_flat
    )
    comp_same = temporal_bin_comp_lc(comp_lc, comp_quality, pd.DataFrame(), enabled=False)
    rms_same = pytics_iterative_weights(comp_lc, comp_quality, rms_flat, enabled=False)
    mag_on, d_on, _ = ensemble_normalize(
        target_lc, comp_same, {cid: 12.0 for cid in cids}, comp_quality, comp_rms_map=rms_same
    )
    np.testing.assert_allclose(mag_off, mag_on, rtol=0, atol=1e-12)
    np.testing.assert_allclose(d_off, d_on, rtol=0, atol=1e-12)


def test_alg4_not_applied_to_mag_calib_in_phase2a_by_default():
    """Document: democratic output is side column only when enabled; default off."""
    from config import AppConfig

    cfg = AppConfig()
    assert cfg.democratic_detrend_enabled is False
    assert cfg.savgol_detrend_enabled is False
    assert cfg.temporal_binning_enabled is False
    assert cfg.pytics_enabled is True


# ---------------------------------------------------------------------------
# Optional: draft 342 real data smoke (skip if archive missing)
# ---------------------------------------------------------------------------

_DRAFT342_LC = Path(
    r"C:\ASTRO\python\VYVAR\Archive\Drafts\draft_000342"
    r"\platesolve\NoFilter_60_2\photometry\lightcurves"
)


@pytest.mark.skipif(not _DRAFT342_LC.is_dir(), reason="draft_342 lightcurves not on disk")
def test_draft342_lc_columns_smoke():
    """One real LC from draft_342: columns and finite mag_calib."""
    lcs = sorted(_DRAFT342_LC.glob("lightcurve_*.csv"))
    assert len(lcs) >= 1
    df = pd.read_csv(lcs[0])
    assert "mag_calib" in df.columns
    assert "mag_calib_ct" in df.columns
    assert np.isfinite(df["mag_calib"].to_numpy()).sum() >= 5
