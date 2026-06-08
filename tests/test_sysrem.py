"""Unit tests for SysRem (TODO-35) — Tamuz, Mazeh & Zucker (2005), MNRAS 356, 1466."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def test_sysrem_pure_systematic():
    """SysRem must remove a pure common-mode systematic perfectly."""
    from photometry_core import run_sysrem_field

    rng = np.random.default_rng(42)
    n_stars, n_frames = 20, 80
    c_true = np.sin(np.linspace(0, 2 * np.pi, n_frames)) * 0.05
    a_true = rng.uniform(0.5, 1.5, n_stars)
    systematic = np.outer(a_true, c_true)
    noise = rng.normal(0, 0.002, (n_stars, n_frames))
    delta = systematic + noise

    with tempfile.TemporaryDirectory() as tmpdir:
        lc_dir = Path(tmpdir)
        for i in range(n_stars):
            df = pd.DataFrame(
                {
                    "delta_mag": delta[i],
                    "err": np.full(n_frames, 0.002),
                    "flag": ["normal"] * n_frames,
                    "bjd": np.arange(n_frames, dtype=float),
                }
            )
            df.to_csv(lc_dir / f"lightcurve_{1000 + i}.csv", index=False)

        result = run_sysrem_field(lc_dir, n_iter=5)

    assert result["n_stars"] == n_stars
    assert result["rms_improvement_pct"] > 50.0, (
        f"Expected >50% RMS improvement, got {result['rms_improvement_pct']:.1f}%"
    )
