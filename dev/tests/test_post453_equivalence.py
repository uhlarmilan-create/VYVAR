"""POST-453 entry-point equivalence tests."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[2] / "src_py"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from night_run import NightRunParams  # noqa: E402
from photometry_core import save_lightcurve_csv  # noqa: E402


def test_night_run_location_id_defaults_none() -> None:
    p = NightRunParams(source_dir=Path("."), equipment_id=1, telescope_id=1)
    assert p.location_id is None


def test_lightcurve_csv_has_no_removed_detrend_columns(tmp_path: Path) -> None:
    """SysRem / ALG-4 / AC columns were removed (2026-10); none may be written."""
    n = 3
    save_lightcurve_csv(
        tmp_path / "lightcurve_test.csv",
        bjd=np.arange(n, dtype=float),
        hjd=np.arange(n, dtype=float),
        jd=np.arange(n, dtype=float),
        airmass=np.ones(n),
        mag_inst=np.full(n, np.nan),
        mag_calib=np.full(n, np.nan),
        delta_mag=np.zeros(n),
        err=np.full(n, 0.01),
        aperture_r_px=np.full(n, 2.0),
        flags=["normal"] * n,
        source_files=["a.csv"] * n,
    )
    df = pd.read_csv(tmp_path / "lightcurve_test.csv")
    for col in (
        "delta_mag_sysrem",
        "delta_mag_democratic",
        "err_inflation",
        "mag_calib_ac",
        "ac_correction",
    ):
        assert col not in df.columns
