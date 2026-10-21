"""ENSEMBLE-BROEG-2026-10: ensemble_normalize uses Broeg (2005) weights, no rejection."""
from __future__ import annotations

import numpy as np

from photometry_core import ensemble_normalize
from photometry_lightcurve import broeg_weights


def test_ensemble_normalize_no_per_frame_zp_rejection():
    """All admitted comps contribute every frame; a jumpy comp only loses weight."""
    n = 20
    rng = np.random.default_rng(0)
    cids = [f"c{i}" for i in range(5)]
    trend = 0.02 * np.linspace(0, 1, n)
    comp_lc = {c: 12.0 + trend + 0.003 * rng.normal(size=n) for c in cids}
    outlier = comp_lc["c4"].copy()
    outlier[::2] += 0.15
    comp_lc["c4"] = outlier
    target = 11.5 + trend + 0.003 * rng.normal(size=n)
    cat = {c: 12.0 for c in cids}
    quality = {c: {"quality": "good"} for c in cids}

    w_out: dict[str, float] = {}
    mag, _delta, scatter = ensemble_normalize(
        target, comp_lc, cat, quality, n_comp_min=3, weights_out=w_out
    )
    assert np.isfinite(mag).sum() == n
    assert set(w_out) == set(cids)  # membership unchanged
    assert w_out["c4"] < 0.05 * min(w_out[c] for c in cids[:4])

    # Manual: normalised comps, Broeg weights, absolute level from catalogue
    m = np.vstack([comp_lc[c] for c in cids])
    med = np.median(m, axis=1)
    w, _ = broeg_weights(m - med[:, None])
    Z = (w[:, None] * (m - med[:, None])).sum(axis=0) / w.sum()
    C = float(np.sum(w * (12.0 - med)) / w.sum())
    np.testing.assert_allclose(mag, target - Z + C, rtol=0, atol=1e-12)
    assert np.all(np.isfinite(scatter)) and np.all(scatter > 0)


def test_ensemble_normalize_three_comps_stable():
    """Identical noiseless comps: equal weights, exact differential result."""
    n = 10
    cids = ["a", "b", "c"]
    comp_lc = {c: 12.0 + 0.01 * i + np.linspace(0, 0.02, n) for i, c in enumerate(cids)}
    target = np.linspace(11.0, 11.02, n)
    cat = {c: 12.0 for c in cids}
    quality = {c: {"quality": "good"} for c in cids}
    mag, _, _ = ensemble_normalize(target, comp_lc, cat, quality, n_comp_min=3)
    # differential: target - comps' common trend + mean catalogue offset (-0.01)
    np.testing.assert_allclose(mag, np.full(n, 11.0 - 0.01), rtol=0, atol=1e-9)


def test_missing_comp_does_not_shift_level():
    n = 40
    rng = np.random.default_rng(3)
    clouds = 0.3 * np.sin(np.linspace(0, 4, n))
    comp_lc = {f"c{i}": 10.0 + 0.7 * i + clouds + 0.002 * rng.normal(size=n) for i in range(4)}
    comp_lc["c0"][10:20] = np.nan
    target = 11.0 + clouds + 0.002 * rng.normal(size=n)
    cat = {f"c{i}": 12.0 + 0.7 * i for i in range(4)}
    mag, _, _ = ensemble_normalize(target, comp_lc, cat, {c: {} for c in comp_lc}, n_comp_min=3)
    assert abs(np.nanmean(mag[10:20]) - np.nanmean(mag[20:30])) < 0.005
    assert abs(np.nanmean(mag) - 13.0) < 0.005


def test_too_few_comps_on_frame_gives_nan():
    n = 10
    comp_lc = {f"c{i}": np.full(n, 12.0) + 0.001 * i * np.arange(n) for i in range(3)}
    comp_lc["c1"][4] = np.nan
    mag, _, _ = ensemble_normalize(np.full(n, 11.0), comp_lc, {c: 12.0 for c in comp_lc},
                                   {c: {} for c in comp_lc}, n_comp_min=3)
    assert not np.isfinite(mag[4]) and np.isfinite(mag[3])
