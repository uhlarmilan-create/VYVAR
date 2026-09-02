# -*- coding: ascii -*-
"""Smoke: per-target field map PNG has no catalog_only (no DAO) category."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits


def test_target_field_map_png_no_catalog_only_legend(tmp_path: Path) -> None:
    from photometry_core import save_target_field_map_png

    ms = tmp_path / "MASTERSTAR.fits"
    data = np.random.default_rng(0).normal(1000.0, 10.0, size=(64, 64))
    fits.writeto(ms, data.astype(np.float32), overwrite=True)

    target = pd.Series(
        {"x": 32.0, "y": 32.0, "vsx_name": "T1", "catalog_id": "1", "zone_flag": "good"}
    )
    comps = pd.DataFrame(
        [{"x": 40.0, "y": 40.0}, {"x": 20.0, "y": 20.0}]
    )
    out = tmp_path / "t1_field.png"
    save_target_field_map_png(out, ms, target, comps, ms_data=data)
    assert out.is_file() and out.stat().st_size > 0

    # Source-level: legend string must not advertise catalog_only
    src = Path(__file__).resolve().parents[2] / "src_py" / "photometry_lightcurve.py"
    text = src.read_text(encoding="ascii")
    assert "cyan=catalog_only" not in text
    assert "catalog_only (no DAO)" not in text
    # Function still documents red/green legend
    assert "(red=VSX target, green=comp star)" in text


def test_target_field_map_skips_catalog_only_marker(tmp_path: Path) -> None:
    from photometry_core import save_target_field_map_png

    ms = tmp_path / "MASTERSTAR.fits"
    data = np.full((32, 32), 500.0, dtype=np.float32)
    fits.writeto(ms, data, overwrite=True)
    target = pd.Series(
        {
            "x": 16.0,
            "y": 16.0,
            "vsx_name": "CO",
            "catalog_id": "9",
            "zone_flag": "catalog_only",
        }
    )
    out = tmp_path / "co_field.png"
    # Must not raise; catalog_only target draws no cyan marker
    save_target_field_map_png(out, ms, target, pd.DataFrame(), ms_data=data)
    assert out.is_file()
