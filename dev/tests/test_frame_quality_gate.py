"""Frame-quality gate is a passthrough (zero-clipping policy 2026-08-12)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import AppConfig
from photometry_core import _frame_quality_gate_select


def _write_frame(path: Path, *, ratio: float, fwhm: float, n: int = 40, flux: float = 5000.0) -> None:
    rng = np.random.default_rng(abs(hash(path.name)) % (2**32))
    fs = np.full(n, flux) * rng.uniform(0.95, 1.05, n)
    fl = fs * ratio * rng.uniform(0.93, 1.07, n)
    df = pd.DataFrame(
        {
            "flux": fs,
            "flux_large": fl,
            "fwhm_estimate_px": np.full(n, fwhm),
            "mag": rng.uniform(10.5, 13.0, n),
            "likely_saturated": np.zeros(n, dtype=int),
        }
    )
    df.to_csv(path, index=False)


def test_frame_quality_gate_always_passthrough(tmp_path: Path) -> None:
    files = []
    for i in range(30):
        p = tmp_path / f"proc_good_{i:03d}.csv"
        _write_frame(p, ratio=2.7, fwhm=6.0)
        files.append(p)
    for i in range(4):
        p = tmp_path / f"proc_bad_{i:03d}.csv"
        _write_frame(p, ratio=12.0, fwhm=8.6)
        files.append(p)
    files = sorted(files)
    cfg = AppConfig()
    kept, rejected = _frame_quality_gate_select(files, cfg, None)
    assert kept == files
    assert rejected == []
