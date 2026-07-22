"""Fail-safety hygiene tests (task #4: MASTERSTAR writeto + edge-ok flag)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from astropy.wcs import WCS


def _minimal_masterstar_fits(path: Path) -> None:
    data = np.zeros((64, 64), dtype=np.float32)
    hdr = fits.Header()
    hdr["NAXIS"] = 2
    hdr["NAXIS1"] = 64
    hdr["NAXIS2"] = 64
    fits.PrimaryHDU(data=data, header=hdr).writeto(path, overwrite=True)


def _minimal_wcs() -> WCS:
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [32, 32]
    wcs.wcs.crval = [180.0, 45.0]
    wcs.wcs.cdelt = [-0.0001, 0.0001]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return wcs


def test_masterstar_writeto_failure_fail_closed(tmp_path, monkeypatch):
    from vyvar_platesolver import _SolveWcsWriteError, _solve_wcs_write_results

    fp = tmp_path / "MASTERSTAR.fits"
    _minimal_masterstar_fits(fp)
    hdr0 = fits.getheader(fp)

    def _boom(*_args, **_kwargs):
        raise OSError("simulated disk full")

    monkeypatch.setattr("vyvar_platesolver.fits.writeto", _boom)

    with pytest.raises(_SolveWcsWriteError) as exc_info:
        _solve_wcs_write_results(
            fp=fp,
            hdr0=hdr0,
            wcs_final=_minimal_wcs(),
            sip_meta={},
            pairs_final=[1.0, 2.0],
            match_rate=0.85,
            rms_px=1.2,
            dao_fw=3.0,
            platescale_arcsec_per_px=1.3,
            is_masterstar=True,
            logger=None,
            cone_r=1.0,
            ep_um=5.0,
            n_img=2,
        )
    assert exc_info.value.result.get("solved") is False
    assert "writeto" in str(exc_info.value.result.get("reason", "")).lower()


def test_edge_ok_pipeline_flags_missing_masterstar(tmp_path):
    from photometry_core import _edge_ok_from_masterstar_pipeline

    stars = pd.DataFrame({"x": [32.0, 40.0], "y": [32.0, 40.0]})
    edge_ok, failed = _edge_ok_from_masterstar_pipeline(
        tmp_path / "missing_MASTERSTAR.fits",
        stars,
        {},
    )
    assert failed is True
    assert edge_ok.all()


def test_variability_export_edge_unfiltered_flag(tmp_path, monkeypatch):
    from photometry_core import auto_export_variability_candidates_csv

    rms_df = pd.DataFrame(
        {
            "catalog_id": ["123456789012345678"],
            "x": [100.0],
            "y": [100.0],
            "ra_deg": [180.0],
            "dec_deg": [45.0],
            "mag": [12.0],
            "rms_pct": [5.0],
            "is_variable_candidate": [True],
            "vsx_known_variable": [False],
            "vsx_match": [False],
            "vsx_id": [None],
            "vsx_name": [""],
        }
    )
    vdi_df = pd.DataFrame(
        {
            "catalog_id": ["123456789012345678"],
            "vdi_score": [0.1],
            "vdi_z_score": [1.0],
            "is_variable_candidate": [False],
        }
    )

    import variability_detector

    monkeypatch.setattr(
        variability_detector,
        "load_field_flux_matrix",
        lambda *a, **k: (pd.DataFrame(), pd.DataFrame(), np.array([])),
    )
    monkeypatch.setattr(variability_detector, "compute_rms_variability", lambda *a, **k: rms_df)
    monkeypatch.setattr(variability_detector, "compute_vdi", lambda *a, **k: vdi_df)
    monkeypatch.setattr(
        "photometry_core._edge_ok_from_masterstar_pipeline",
        lambda *a, **k: (pd.Series([True], index=rms_df.index), True),
    )

    class _Cfg:
        variability_sigma_threshold = 2.3
        variability_mag_limit = 14.5
        gaia_db_path = ""

        def to_dict(self):
            return {}

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (tmp_path / "frames").mkdir()

    out = auto_export_variability_candidates_csv(
        masterstar_fits_path=tmp_path / "MASTERSTAR.fits",
        comparison_stars_csv=None,
        per_frame_csv_dir=tmp_path / "frames",
        output_dir=out_dir,
        cfg=_Cfg(),
    )
    assert out is not None
    exported = pd.read_csv(out)
    assert bool(exported["edge_filter_failed"].iloc[0]) is True
    assert "EDGE-UNFILTERED" in str(exported["edge_filter_note"].iloc[0])


def test_app_imports_resolve():
    import app  # noqa: F401
