"""Unit test: ensemble_normalize ZP path has no per-frame rejection."""
from __future__ import annotations

import numpy as np

from photometry_core import ensemble_normalize


def test_ensemble_normalize_no_per_frame_zp_rejection():
    """All admitted comps contribute every frame (INV-COMP-MEMBERSHIP).

    With N>=4 an injected outlier must still enter the Broeg-weighted ZP
    (formerly rejected by the removed 3xMAD clip).
    """
    n = 20
    rng = np.random.default_rng(0)
    cids = [f"c{i}" for i in range(5)]
    # Shared extinction trend + small noise
    trend = 0.02 * np.linspace(0, 1, n)
    comp_lc = {c: 12.0 + trend + 0.003 * rng.normal(size=n) for c in cids}
    # Inject a large ZP outlier on comp c4 for half the frames
    outlier = comp_lc["c4"].copy()
    outlier[::2] += 0.15
    comp_lc["c4"] = outlier
    target = 11.5 + trend + 0.003 * rng.normal(size=n)
    cat = {c: 12.0 for c in cids}
    quality = {c: {"quality": "good", "rms_p2p": 0.01} for c in cids}
    rms = {c: 0.01 for c in cids}

    mag, delta, scatter = ensemble_normalize(
        target,
        comp_lc,
        cat,
        quality,
        comp_rms_map=rms,
        n_comp_min=3,
        n_comp_max=5,
    )
    assert np.isfinite(mag).sum() == n

    # Manual Broeg ZP with ALL 5 comps (no rejection)
    zp_all = np.zeros(n)
    for i in range(n):
        zs = np.array([cat[c] - comp_lc[c][i] for c in cids])
        w = np.array([1.0 / (rms[c] ** 2) for c in cids])
        zp_all[i] = float(np.sum(w * zs) / np.sum(w))
    mag_all = target + zp_all
    np.testing.assert_allclose(mag, mag_all, rtol=0, atol=1e-12)

    # And it must DIFFER from the 4-comp (c4 dropped) solution on outlier frames
    zp_drop = np.zeros(n)
    keep = [c for c in cids if c != "c4"]
    for i in range(n):
        zs = np.array([cat[c] - comp_lc[c][i] for c in keep])
        w = np.array([1.0 / (rms[c] ** 2) for c in keep])
        zp_drop[i] = float(np.sum(w * zs) / np.sum(w))
    mag_drop = target + zp_drop
    assert float(np.max(np.abs(mag - mag_drop))) > 0.01


def test_ensemble_normalize_three_comps_stable():
    """N=3 path (draft 435) is Broeg-weighted mean; clip gate never applied."""
    n = 10
    cids = ["a", "b", "c"]
    comp_lc = {c: np.full(n, 12.0 + 0.01 * i) for i, c in enumerate(cids)}
    for c in cids:
        comp_lc[c] = comp_lc[c] + np.linspace(0, 0.02, n)
    target = np.linspace(11.0, 11.02, n)
    cat = {c: 12.0 for c in cids}
    quality = {c: {"quality": "good", "rms_p2p": 0.01} for c in cids}
    rms = {"a": 0.01, "b": 0.02, "c": 0.015}
    mag, _, _ = ensemble_normalize(
        target, comp_lc, cat, quality, comp_rms_map=rms, n_comp_min=3, n_comp_max=3
    )
    zp = np.zeros(n)
    for i in range(n):
        zs = np.array([cat[c] - comp_lc[c][i] for c in cids])
        w = np.array([1.0 / (rms[c] ** 2) for c in cids])
        zp[i] = float(np.sum(w * zs) / np.sum(w))
    np.testing.assert_allclose(mag, target + zp, rtol=0, atol=1e-12)
