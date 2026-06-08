"""Tests for ProcFrameStore (PERF-5 Option B)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from proc_frame_store import ProcFrameStore


def test_proc_frame_store_basic() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir)
        for i in range(3):
            pd.DataFrame(
                {
                    "catalog_id": ["111", "222"],
                    "name": ["111", "222"],
                    "dao_flux": [1000.0 + i, 500.0 + i],
                    "mag": [10.0, 11.0],
                    "airmass": [1.2, 1.2],
                    "x": [100.0, 200.0],
                    "y": [150.0, 250.0],
                }
            ).to_csv(p / f"proc_frame{i:03d}.csv", index=False)

        store = ProcFrameStore.build(p)
        assert len(store) == 3

        key = str(sorted(p.glob("proc_*.csv"))[0])
        df = store.get(key)
        assert df is not None
        assert "dao_flux" in df.columns

        df_proj = store.get_frame(key, cols=["catalog_id", "dao_flux"])
        assert df_proj is not None
        assert list(df_proj.columns) == ["catalog_id", "dao_flux"]

        assert store.get("nonexistent.csv") is None


def test_proc_frame_store_ram_estimate() -> None:
    """Store RAM should be under 500MB for typical draft (scaled-down synthetic)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir)
        rng = np.random.default_rng(0)
        for i in range(10):
            pd.DataFrame(
                {
                    "catalog_id": [str(x) for x in range(1794)],
                    "dao_flux": rng.uniform(100, 50000, 1794),
                    "mag": rng.uniform(9, 15, 1794),
                    "x": rng.uniform(0, 2048, 1794),
                    "y": rng.uniform(0, 1366, 1794),
                    "airmass": np.full(1794, 1.3),
                }
            ).to_csv(p / f"proc_frame{i:03d}.csv", index=False)

        store = ProcFrameStore.build(p)
        assert store.n_frames == 10
        assert store.n_stars_median == 1794
