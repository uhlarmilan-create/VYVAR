"""P-10 bundle: gradient-immune DAO noise estimator (sigma_pp)."""

from __future__ import annotations

import numpy as np
from astropy.stats import sigma_clipped_stats

from pipeline import _dao_noise_sigma_adu, _pixel_noise_sigma_pp_adu


def test_sigma_pp_stable_under_doubled_linear_gradient() -> None:
    rng = np.random.default_rng(0)
    h, w = 512, 512
    noise = rng.normal(0, 46.0, (h, w)).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    g1 = (1000.0 + 0.08 * xx.astype(np.float32) + 0.05 * yy.astype(np.float32) + noise).astype(
        np.float32
    )
    g2 = (1000.0 + 0.16 * xx.astype(np.float32) + 0.10 * yy.astype(np.float32) + noise).astype(
        np.float32
    )
    s1 = _pixel_noise_sigma_pp_adu(g1)
    s2 = _pixel_noise_sigma_pp_adu(g2)
    assert 35.0 < s1 < 60.0
    assert abs(s1 - s2) < 8.0
    _, _, std1 = sigma_clipped_stats(g1, sigma=3.0, maxiters=3)
    _, _, std2 = sigma_clipped_stats(g2, sigma=3.0, maxiters=3)
    assert float(std2) >= float(std1)


def test_dao_noise_sigma_uses_sigma_pp() -> None:
    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[0:256, 0:256]
    arr = (
        1200.0
        + 0.12 * xx.astype(np.float32)
        + rng.normal(0, 40.0, (256, 256)).astype(np.float32)
    ).astype(np.float32)
    _, _, std = sigma_clipped_stats(arr, sigma=3.0, maxiters=3)
    sigma_pp = _pixel_noise_sigma_pp_adu(arr)
    got = _dao_noise_sigma_adu(arr, bfac=1, fallback_std=float(std))
    assert abs(got - sigma_pp) < 1.0
