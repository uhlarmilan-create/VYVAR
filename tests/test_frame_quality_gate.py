"""Round-2 B.2 frame-quality gate (_frame_quality_gate_select).

Default OFF -> byte-identical no-op. When enabled, rejects whole frames whose PSF concentration
(flux_large/flux) is a robust outlier and FWHM >= median; spares clear-but-faint and sharp frames;
honors the min-keep safety floor.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import AppConfig
from photometry_core import _frame_quality_gate_select


def _write_frame(path: Path, *, ratio: float, fwhm: float, n: int = 40, flux: float = 5000.0) -> None:
    """Write a minimal proc_*.csv whose median(flux_large/flux) == ratio."""
    rng = np.random.default_rng(abs(hash(path.name)) % (2**32))
    fs = np.full(n, flux) * rng.uniform(0.95, 1.05, n)
    # independent per-source noise on the large aperture -> realistic per-frame ratio scatter (MAD>0)
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


def _make_night(tmp_path: Path, n_good: int = 30, n_bad: int = 4) -> list[Path]:
    files = []
    for i in range(n_good):
        p = tmp_path / f"proc_good_{i:03d}.csv"
        _write_frame(p, ratio=2.7, fwhm=6.0)
        files.append(p)
    for i in range(n_bad):
        p = tmp_path / f"proc_bad_{i:03d}.csv"
        _write_frame(p, ratio=12.0, fwhm=8.6)  # collapsed: high concentration ratio + blurred
        files.append(p)
    return sorted(files)


def test_gate_default_off_is_noop(tmp_path: Path) -> None:
    files = _make_night(tmp_path)
    cfg = AppConfig()  # frame_quality_gate_enabled defaults False
    kept, rejected = _frame_quality_gate_select(files, cfg, None)
    assert kept == files
    assert rejected == []


def test_gate_rejects_collapsed_frames(tmp_path: Path) -> None:
    files = _make_night(tmp_path, n_good=30, n_bad=4)
    cfg = AppConfig()
    cfg.frame_quality_gate_enabled = True
    kept, rejected = _frame_quality_gate_select(files, cfg, None)
    assert len(rejected) == 4
    assert all("bad" in r for r in rejected)
    assert all("good" in Path(k).name for k in kept)


def test_gate_spares_clear_but_faint_frames(tmp_path: Path) -> None:
    # Faint-but-clear frames keep normal concentration (ratio ~2.7) even at low flux -> not rejected.
    files = []
    for i in range(25):
        p = tmp_path / f"proc_bright_{i:03d}.csv"
        _write_frame(p, ratio=2.7, fwhm=6.0, flux=5000.0)
        files.append(p)
    for i in range(8):
        p = tmp_path / f"proc_faint_{i:03d}.csv"
        _write_frame(p, ratio=2.7, fwhm=6.5, flux=300.0)  # dim but normal PSF concentration
        files.append(p)
    files = sorted(files)
    cfg = AppConfig()
    cfg.frame_quality_gate_enabled = True
    kept, rejected = _frame_quality_gate_select(files, cfg, None)
    assert rejected == []
    assert len(kept) == len(files)


def test_gate_spares_sharp_ratio_outlier(tmp_path: Path) -> None:
    # A ratio outlier on a SHARP (better-than-median) frame is spared by the FWHM guard.
    files = []
    for i in range(30):
        p = tmp_path / f"proc_good_{i:03d}.csv"
        _write_frame(p, ratio=2.7, fwhm=7.0)
        files.append(p)
    sharp = tmp_path / "proc_sharp_outlier.csv"
    _write_frame(sharp, ratio=12.0, fwhm=3.0)  # high ratio but very sharp -> spared
    files.append(sharp)
    files = sorted(files)
    cfg = AppConfig()
    cfg.frame_quality_gate_enabled = True
    kept, rejected = _frame_quality_gate_select(files, cfg, None)
    assert sharp.name not in rejected


def test_gate_safety_floor_skips_when_too_few_remain(tmp_path: Path) -> None:
    # If rejecting would leave < min_keep frames, the gate is skipped (returns input unchanged).
    files = _make_night(tmp_path, n_good=6, n_bad=4)
    cfg = AppConfig()
    cfg.frame_quality_gate_enabled = True
    cfg.frame_quality_min_keep_frames = 8  # 10 - 4 = 6 < 8 -> skip
    kept, rejected = _frame_quality_gate_select(files, cfg, None)
    assert kept == files
    assert rejected == []
