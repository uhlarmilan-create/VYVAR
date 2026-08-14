"""TARGET-DEPTH-01: derived measurable depth for target admission."""
from __future__ import annotations

import numpy as np
import pandas as pd

from comp_pool_noise import derive_target_depth_limit


def test_derive_target_depth_from_detect_frac_drop_without_scatter():
    """Ceiling-degenerate detect_frac without scatter_mad -> half-completeness T-R0."""
    rng = np.random.default_rng(0)
    rows = []
    for mag in np.arange(8.0, 16.0, 0.25):
        if mag <= 13.5:
            frac = 1.0
        elif mag < 14.5:
            frac = 0.95
        else:
            frac = 0.40
        for _ in range(12):
            rows.append(
                {
                    "mag_g": float(mag + rng.normal(0, 0.02)),
                    "detect_frac": float(frac),
                    "vsx_known_variable": False,
                }
            )
    lim = derive_target_depth_limit(pd.DataFrame(rows))
    assert lim.mode == "detect_frac"
    assert lim.detect_frac_thr is not None
    assert abs(float(lim.detect_frac_thr) - 0.5) < 1e-6
    assert lim.target_depth_g is not None
    assert 14.0 <= float(lim.target_depth_g) <= 15.0


def test_derive_target_depth_np_half_snr_when_forced_photometry_ceiling():
    """Forced-photometry detect_frac~1 with rising scatter -> NP half-SNR depth."""
    rng = np.random.default_rng(1)
    rows = []
    for mag in np.arange(8.0, 16.25, 0.25):
        # Fully complete through G14.0-14.5; tiny forced-phot drops after that.
        frac = 1.0 if mag < 14.5 else 0.995
        sc = 0.012 * (10 ** (0.25 * (mag - 8.0)))
        for _ in range(12):
            rows.append(
                {
                    "mag_g": float(mag + rng.normal(0, 0.02)),
                    "detect_frac": float(frac),
                    "scatter_mad": float(max(1e-4, sc * (1.0 + 0.03 * rng.normal()))),
                    "vsx_known_variable": False,
                }
            )
    lim = derive_target_depth_limit(pd.DataFrame(rows))
    assert lim.mode == "np_half_snr"
    assert lim.target_depth_g is not None
    assert 13.5 <= float(lim.target_depth_g) <= 15.5


def test_derive_target_depth_insufficient_bright_returns_none():
    rows = [{"mag_g": 10.0, "detect_frac": 1.0, "vsx_known_variable": False} for _ in range(3)]
    lim = derive_target_depth_limit(pd.DataFrame(rows))
    assert lim.target_depth_g is None
    assert lim.n_bright < 8
