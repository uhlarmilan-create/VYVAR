"""COMP-RMS-NORM-2026-10 (D2): constant pool stars of different brightness with frames where some star
is missing must get a leave-one-out scatter of ~0, not the jump of the raw-magnitude median."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from comp_rms_loo import compute_loo_mag_rms_map


def test_dropouts_of_other_stars_give_no_spurious_scatter():
    ids = [f"10000000000000000{i:02d}" for i in range(5)]
    mags = np.array([9.0, 10.0, 11.0, 12.0, 13.0])
    cache, paths = {}, []
    rng = np.random.default_rng(7)
    for f in range(60):
        keep = rng.random(5) > 0.4  # every star missing on ~40 % of the frames (edge, dropout)
        if keep.sum() < 3:
            keep[:3] = True
        flux = 10 ** (-0.4 * (mags - 25.0)) * 1.05 ** (f % 3)  # common transparency change
        df = pd.DataFrame({"catalog_id": [i for i, k in zip(ids, keep) if k], "dao_flux": flux[keep]})
        p = Path(f"proc_frame_{f:03d}.csv")
        cache[str(p)] = df
        paths.append(p)
    rms, _ = compute_loo_mag_rms_map(set(ids), paths, cache, flux_col="dao_flux")
    assert set(rms) == set(ids)
    assert max(rms.values()) < 1e-9
