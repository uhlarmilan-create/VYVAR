# -*- coding: ascii -*-
"""Smoke: per-target field map PNG has no catalog_only (no DAO) category."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits


def test_target_field_map_png_no_catalog_only_legend(tmp_path: Path) -> None:
    from field_map import TargetFieldMap

    ms = tmp_path / "MASTERSTAR.fits"
    data = np.random.default_rng(0).normal(1000.0, 10.0, size=(64, 64))
    fits.writeto(ms, data.astype(np.float32), overwrite=True)

    target = pd.Series(
        {"x": 32.0, "y": 32.0, "vsx_name": "T1", "catalog_id": "1", "zone_flag": "good"}
    )
    comps = pd.DataFrame(
        [{"x": 40.0, "y": 40.0}, {"x": 20.0, "y": 20.0}]
    )
    out = tmp_path / "t1_field.jpg"
    fm = TargetFieldMap(data)  # OUTPUT-REVIEW step 7: background drawn once, markers per target
    fm.save(out, target, comps)
    out2 = tmp_path / "t2_field.jpg"
    fm.save(out2, target, comps.iloc[:1])
    fm.close()
    assert out.is_file() and out.stat().st_size > 0 and out2.is_file()

    # Source-level: legend string must not advertise catalog_only
    src = Path(__file__).resolve().parents[2] / "src_py" / "field_map.py"
    text = src.read_text(encoding="ascii")
    assert "cyan=catalog_only" not in text
    assert "catalog_only (no DAO)" not in text
    # The renderer still documents the red/green legend
    assert "(red = VSX target, green = comparison stars)" in text
