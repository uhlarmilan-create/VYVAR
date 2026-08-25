"""Tests for PSF-only sidecar merge and INV-PSF-ADDITIVE-01 (EPSF-VALID-02 F6)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src_py"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from epsf_psf_merge import (  # noqa: E402
    INV_PSF_ADDITIVE_01,
    assert_inv_psf_additive_01,
    merge_psf_into_sidecar,
)
from invariants_runtime import InvariantViolation  # noqa: E402


def _write_fits(path: Path, *, n: int = 32) -> None:
    data = np.ones((n, n), dtype=np.float32) * 100.0
    hdr = fits.Header()
    hdr["NAXIS1"] = n
    hdr["NAXIS2"] = n
    fits.PrimaryHDU(data=data, header=hdr).writeto(path, overwrite=True)


def test_assert_inv_psf_additive_01_passes_identical() -> None:
    df = pd.DataFrame({"catalog_id": ["1"], "mag": [10.0], "psf_flux": [np.nan]})
    assert_inv_psf_additive_01(df, df.copy(), frame_name="t.fits")


def test_assert_inv_psf_additive_01_trips_on_aperture_drift() -> None:
    before = pd.DataFrame({"catalog_id": ["1"], "dao_flux": [100.0]})
    after = before.copy()
    after["dao_flux"] = 200.0
    with pytest.raises(InvariantViolation, match=INV_PSF_ADDITIVE_01):
        assert_inv_psf_additive_01(before, after, frame_name="t.fits")


def test_merge_psf_missing_sidecar_fails_loud(tmp_path: Path) -> None:
    fits_p = tmp_path / "BO_CVn_Light_001.fits"
    _write_fits(fits_p)
    with pytest.raises(FileNotFoundError, match="Missing proc sidecar"):
        merge_psf_into_sidecar(
            fits_path=fits_p,
            sidecar_path=tmp_path / "proc_BO_CVn_Light_001.csv",
            st={"_run_epsf": True, "epsf_model_path": str(tmp_path / "missing.fits")},
            target_ids=None,
        )


def test_merge_psf_preserves_aperture_columns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fits_p = tmp_path / "BO_CVn_Light_001.fits"
    _write_fits(fits_p)
    sidecar = tmp_path / "proc_BO_CVn_Light_001.csv"
    pd.DataFrame(
        {
            "catalog_id": ["1498486880958321024"],
            "x": [16.0],
            "y": [16.0],
            "dao_flux": [12345.6],
            "mag": [11.0],
        }
    ).to_csv(sidecar, index=False)

    def _fake_fill(df, data, hdr, st, *, target_ids=None):
        out = df.copy()
        out["psf_flux"] = 999.0
        out["psf_fit_ok"] = True
        st["_psf_frame_record"] = {
            "frame_name": fits_p.name,
            "frame_index": 0,
            "n_fit": 1,
            "n_ok": 1,
            "exception_class": None,
            "exception_message": None,
            "traceback_tail": None,
        }
        return out

    import pipeline

    monkeypatch.setattr(pipeline, "_fill_psf_catalog_columns", _fake_fill)

    row = merge_psf_into_sidecar(
        fits_path=fits_p,
        sidecar_path=sidecar,
        st={"_run_epsf": True, "epsf_model_path": str(tmp_path / "model.fits")},
        target_ids={"1498486880958321024"},
    )
    assert row["status"] == "ok"
    out = pd.read_csv(sidecar, dtype={"catalog_id": str})
    assert float(out.loc[0, "dao_flux"]) == pytest.approx(12345.6)
    assert float(out.loc[0, "psf_flux"]) == pytest.approx(999.0)


def test_x_fit_y_fit_are_psf_columns() -> None:
    from epsf_psf_merge import is_psf_column

    assert is_psf_column("x_fit")
    assert is_psf_column("y_fit")
    assert is_psf_column("psf_group_n")
    assert not is_psf_column("dao_flux")
    assert not is_psf_column("x")


def test_merge_psf_x_fit_does_not_break_additivity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fits_p = tmp_path / "BO_CVn_Light_001.fits"
    _write_fits(fits_p)
    sidecar = tmp_path / "proc_BO_CVn_Light_001.csv"
    pd.DataFrame(
        {
            "catalog_id": ["1498486880958321024"],
            "x": [16.0],
            "y": [16.0],
            "dao_flux": [12345.6],
            "mag": [11.0],
        }
    ).to_csv(sidecar, index=False)

    def _fake_fill(df, data, hdr, st, *, target_ids=None):
        out = df.copy()
        out["psf_flux"] = 999.0
        out["psf_fit_ok"] = True
        out["x_fit"] = 16.2
        out["y_fit"] = 15.9
        out["psf_group_n"] = 1
        st["_psf_frame_record"] = {
            "frame_name": fits_p.name,
            "frame_index": 0,
            "n_fit": 1,
            "n_ok": 1,
            "exception_class": None,
            "exception_message": None,
            "traceback_tail": None,
        }
        return out

    import pipeline

    monkeypatch.setattr(pipeline, "_fill_psf_catalog_columns", _fake_fill)
    before = pd.read_csv(sidecar)
    merge_psf_into_sidecar(
        fits_path=fits_p,
        sidecar_path=sidecar,
        st={"_run_epsf": True, "epsf_model_path": str(tmp_path / "model.fits")},
        target_ids={"1498486880958321024"},
    )
    after = pd.read_csv(sidecar, dtype={"catalog_id": str})
    assert_inv_psf_additive_01(before, after, frame_name=fits_p.name)
    assert float(after.loc[0, "x_fit"]) == pytest.approx(16.2)
    assert int(after.loc[0, "psf_group_n"]) == 1


def test_fill_psf_skips_moffat_when_psf_merge_only(monkeypatch: pytest.MonkeyPatch) -> None:
    import pipeline

    called = {"moffat": False}

    def _boom(*args, **kwargs):
        called["moffat"] = True
        raise RuntimeError("moffat should not run")

    monkeypatch.setattr("psf_photometry.fit_moffat_psf_stars", _boom)

    df = pd.DataFrame(
        {
            "catalog_id": ["1498486880958321024"],
            "x": [10.0],
            "y": [10.0],
            "dao_flux": [100.0],
        }
    )
    data = np.ones((32, 32), dtype=np.float32)
    hdr = fits.Header()
    st = {
        "_run_epsf": True,
        "_psf_merge_only": True,
        "epsf_model_path": str(Path("/nonexistent/model.fits")),
    }
    pipeline._fill_psf_catalog_columns(df, data, hdr, st)
    assert called["moffat"] is False


def test_stamp_p4_none_inverts_ac_and_keeps_aperture(tmp_path: Path) -> None:
    from epsf_psf_merge import stamp_p4_none_on_dataframe, stamp_p4_none_sidecar

    sidecar = tmp_path / "proc_Light_001.csv"
    pd.DataFrame(
        {
            "catalog_id": ["1"],
            "dao_flux": [12345.6],
            "psf_flux": [50.0],
            "psf_flux_err": [1.0],
            "psf_ac_factor": [0.5],
            "psf_ac_applied": [True],
            "psf_ac_n_used": [12],
        }
    ).to_csv(sidecar, index=False)
    rec = stamp_p4_none_sidecar(sidecar)
    assert rec["status"] == "ok"
    out = pd.read_csv(sidecar, dtype={"catalog_id": str})
    assert float(out.loc[0, "dao_flux"]) == pytest.approx(12345.6)
    assert float(out.loc[0, "psf_flux"]) == pytest.approx(100.0)
    assert float(out.loc[0, "psf_flux_err"]) == pytest.approx(2.0)
    assert float(out.loc[0, "psf_ac_factor"]) == pytest.approx(1.0)
    assert str(out.loc[0, "psf_ac_policy"]) == "p4_none"
    before = pd.DataFrame({"catalog_id": ["1"], "dao_flux": [10.0], "psf_flux": [1.0]})
    after = stamp_p4_none_on_dataframe(before)
    assert_inv_psf_additive_01(before, after, frame_name="t.csv")


def test_resolve_psf_ac_policy_and_legacy_ac_formula() -> None:
    from psf_photometry import _compute_aperture_correction, resolve_psf_ac_policy

    assert resolve_psf_ac_policy(None) == "p4_none"
    assert resolve_psf_ac_policy("p4_none", apply_aperture_correction=True) == "p4_none"
    assert resolve_psf_ac_policy(None, apply_aperture_correction=True) == "chi2_lt5_legacy"
    psf = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    ref = np.array([20.0, 40.0, 60.0, 80.0, 100.0, 120.0])
    chi2 = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    fac, n_used = _compute_aperture_correction(psf, ref, chi2, chi2_limit=5.0, min_ref_stars=5)
    assert n_used == 6
    assert fac == pytest.approx(2.0)
