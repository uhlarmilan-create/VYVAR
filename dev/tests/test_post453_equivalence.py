"""POST-453 entry-point equivalence tests."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[2] / "src_py"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from night_run import NightRunParams  # noqa: E402
from photometry_core import save_lightcurve_csv  # noqa: E402


def test_night_run_location_id_defaults_none() -> None:
    p = NightRunParams(source_dir=Path("."), equipment_id=1, telescope_id=1)
    assert p.location_id is None


def test_lightcurve_csv_always_has_delta_mag_sysrem(tmp_path: Path) -> None:
    n = 3
    save_lightcurve_csv(
        tmp_path / "lightcurve_test.csv",
        bjd=np.arange(n, dtype=float),
        hjd=np.arange(n, dtype=float),
        jd=np.arange(n, dtype=float),
        airmass=np.ones(n),
        is_flipped=None,
        mag_inst=np.full(n, np.nan),
        mag_calib_raw=np.full(n, np.nan),
        mag_calib=np.full(n, np.nan),
        mag_calib_ct=None,
        mag_calib_ac=None,
        delta_mag=np.zeros(n),
        err=np.full(n, 0.01),
        aperture_r_px=np.full(n, 2.0),
        flags=["normal"] * n,
        source_files=["a.csv"] * n,
    )
    df = pd.read_csv(tmp_path / "lightcurve_test.csv")
    assert "delta_mag_sysrem" in df.columns
    assert df["delta_mag_sysrem"].isna().all()
