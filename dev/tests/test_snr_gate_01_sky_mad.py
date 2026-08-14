"""SNR-GATE-01: sky MAD noise scale and prematch estimator regression."""
from __future__ import annotations

import numpy as np

from plain_stats import plain_mean_med_std, sky_mad_sigma_adu


def test_sky_mad_sigma_scales_with_sky_level():
    rng = np.random.default_rng(42)
    # Two sky levels, same read noise; add a few bright stars that must not dominate MAD.
    def frame(sky: float) -> np.ndarray:
        img = sky + rng.normal(0.0, np.sqrt(sky / 3.0), size=(256, 256))
        img[40:45, 40:45] += 5000.0
        img[120:126, 180:186] += 8000.0
        return img

    _, s1 = sky_mad_sigma_adu(frame(1400.0))
    _, s2 = sky_mad_sigma_adu(frame(2400.0))
    assert np.isfinite(s1) and np.isfinite(s2)
    assert s2 > s1
    assert (s2 / s1) > 1.05


def test_plain_full_std_does_not_scale_like_sky_noise():
    rng = np.random.default_rng(7)

    def frame(sky: float) -> np.ndarray:
        img = sky + rng.normal(0.0, np.sqrt(max(sky, 1.0) / 3.0), size=(256, 256))
        # Many bright sources: sample std becomes scene-dominated and nearly sky-flat.
        for _ in range(40):
            cy = int(rng.integers(8, 248))
            cx = int(rng.integers(8, 248))
            img[cy : cy + 5, cx : cx + 5] += float(rng.uniform(2000.0, 12000.0))
        return img

    f1, f2 = frame(1400.0), frame(2400.0)
    _, _, p1 = plain_mean_med_std(f1)
    _, _, p2 = plain_mean_med_std(f2)
    _, m1 = sky_mad_sigma_adu(f1)
    _, m2 = sky_mad_sigma_adu(f2)
    assert abs(p2 / p1 - 1.0) < 0.15  # scene-dominated: weak sky response
    assert (m2 / m1) > 1.15


def test_iron_gates_do_not_flag_detection_threshold_form():
    """INV-NOCLIP targets clip APIs / one-sided annulus sky value, not median+k*sigma detection floors."""
    from pathlib import Path
    import sys

    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / "dev" / "tools"))
    import iron_gates_scan as ig

    text = (
        "noise_floor = float(sky_med + k * sky_sig)\n"
        "snr_keep = pmax_arr > noise_floor\n"
    )
    hits = ig._scan_patterns("INV-NOCLIP-01", "pipeline.py", text, ig.NOCLIP_PATTERNS)
    assert hits == []
