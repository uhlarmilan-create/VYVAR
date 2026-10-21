"""Tests for PSF-only sidecar merge and INV-PSF-ADDITIVE-01 (EPSF-VALID-02 F6)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src_py"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from epsf_psf_merge import (  # noqa: E402
    INV_PSF_ADDITIVE_01,
    assert_inv_psf_additive_01,
    merge_psf_frame_result,
)
from invariants_runtime import InvariantViolation  # noqa: E402


def test_assert_inv_psf_additive_01_passes_identical() -> None:
    df = pd.DataFrame({"catalog_id": ["1"], "mag": [10.0], "psf_flux": [np.nan]})
    assert_inv_psf_additive_01(df, df.copy(), frame_name="t.fits")


def test_assert_inv_psf_additive_01_trips_on_aperture_drift() -> None:
    before = pd.DataFrame({"catalog_id": ["1"], "dao_flux": [100.0]})
    after = before.copy()
    after["dao_flux"] = 200.0
    with pytest.raises(InvariantViolation, match=INV_PSF_ADDITIVE_01):
        assert_inv_psf_additive_01(before, after, frame_name="t.fits")


CID = "1498486880958321024"


def _fit_df(**extra) -> pd.DataFrame:
    row = {"catalog_id": CID, "psf_flux": 999.0, "psf_flux_err": 9.0, "psf_chi2": 1.2, "psf_flags": 0,
           "psf_converged": True, "psf_snr": 111.0, "psf_sky": 5.0, "x_fit": 16.2, "y_fit": 15.9}
    row.update(extra)
    return pd.DataFrame([row])


def _sidecar(tmp_path: Path, **extra) -> Path:
    p = tmp_path / "proc_BO_CVn_Light_001.csv"
    base = {"catalog_id": [CID, "1"], "x": [16.0, 3.0], "y": [16.0, 3.0], "dao_flux": [12345.6, 5.0], "mag": [11.0, 15.0]}
    base.update(extra)
    pd.DataFrame(base).to_csv(p, index=False)
    return p


def test_merge_psf_missing_sidecar_fails_loud(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Missing proc sidecar"):
        merge_psf_frame_result(tmp_path / "proc_x.csv", _fit_df(), frame_name="x.fits")


def test_merge_psf_preserves_aperture_columns_and_maps_fit(tmp_path: Path) -> None:
    sidecar = _sidecar(tmp_path)
    before = pd.read_csv(sidecar)
    n = merge_psf_frame_result(sidecar, _fit_df(), frame_name="BO_CVn_Light_001.fits")
    after = pd.read_csv(sidecar, dtype={"catalog_id": str})
    assert n == 1
    assert_inv_psf_additive_01(before, after, frame_name="t")
    assert float(after.loc[0, "dao_flux"]) == pytest.approx(12345.6)
    assert float(after.loc[0, "psf_flux"]) == pytest.approx(999.0)
    # OUTPUT-REVIEW-2026-10: only PSF_PROC_COLUMNS of the fit are written
    assert not {"x_fit", "psf_flags", "psf_snr", "psf_sky"} & set(after.columns)
    assert float(after.loc[0, "psf_flux_err"]) == pytest.approx(9.0)
    assert np.isnan(after.loc[1, "psf_flux"])  # not in the fit set
    assert not after["psf_fit_ok"].astype(bool).any()  # set later by the night chi2 locus


def test_merge_psf_drops_columns_of_the_old_fitter(tmp_path: Path) -> None:
    sidecar = _sidecar(tmp_path, psf_quality=["good", ""], psf_ac_factor=[1.0, 1.0], psf_group_n=[0, 0])
    merge_psf_frame_result(sidecar, _fit_df(), frame_name="t.fits")
    cols = set(pd.read_csv(sidecar).columns)
    assert not {"psf_quality", "psf_ac_factor", "psf_group_n"} & cols


def test_x_fit_y_fit_are_psf_columns() -> None:
    from epsf_psf_merge import is_psf_column

    assert is_psf_column("x_fit")
    assert is_psf_column("y_fit")
    assert is_psf_column("psf_flags")
    assert not is_psf_column("dao_flux")
    assert not is_psf_column("x")
