"""EPSF-LC-LOG-01: internal PSF LC writer + INV-PSF-SUBMIT-01."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from config import AppConfig
from export_reports import export_all_method_lightcurve_reports, export_lightcurve_reports
from invariants_runtime import InvariantViolation
from photometry_core import TIME_BASE_BJD_TDB
from psf_internal_lc import (
    REQUIRED_HEADER_MARKERS,
    write_internal_psf_lightcurves,
)
from report_methods import lc_csv_path

ROOT = Path(__file__).resolve().parents[2]
DRAFT_516 = ROOT / "Archive" / "Drafts" / "draft_000516"
PS_516 = DRAFT_516 / "platesolve" / "NoFilter_60_2"
FRAMES_516 = DRAFT_516 / "detrended_aligned" / "lights" / "NoFilter_60_2"
BO_CVN_ID = "1498613634033133184"
N_SCIENCE_LIGHTS = 134

INV_PSF_SUBMIT_01 = "INV-PSF-SUBMIT-01"


def _minimal_lc() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "bjd": [2460000.5],
            "mag_calib": [12.5],
            "mag_calib_final": [12.5],
            "err": [0.01],
            "flag": ["normal"],
            "time_base": [TIME_BASE_BJD_TDB],
        }
    )


def _minimal_target() -> pd.Series:
    return pd.Series({"vsx_name": "TESTSTAR", "vsx_type": "EA", "catalog_id": "1"})


def _minimal_summary() -> pd.Series:
    return pd.Series({"obs_group": "NoFilter_60_2", "n_good_comp": 4})


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_inv_psf_submit_01_rejects_psf_method(tmp_path: Path) -> None:
    with pytest.raises(InvariantViolation, match=INV_PSF_SUBMIT_01):
        export_lightcurve_reports(
            tmp_path / "reports",
            _minimal_target(),
            _minimal_lc(),
            pd.DataFrame({"catalog_id": ["9"], "mag": [12.0]}),
            _minimal_summary(),
            cfg=AppConfig(),
            export_method="psf",
        )


def test_inv_psf_submit_01_rejects_adaptive_method(tmp_path: Path) -> None:
    with pytest.raises(InvariantViolation, match=INV_PSF_SUBMIT_01):
        export_lightcurve_reports(
            tmp_path / "reports",
            _minimal_target(),
            _minimal_lc(),
            pd.DataFrame({"catalog_id": ["9"], "mag": [12.0]}),
            _minimal_summary(),
            cfg=AppConfig(),
            export_method="adaptive",
        )


def test_inv_psf_submit_01_aperture_still_exports(tmp_path: Path) -> None:
    paths = export_lightcurve_reports(
        tmp_path / "reports",
        _minimal_target(),
        _minimal_lc(),
        pd.DataFrame({"catalog_id": ["9"], "mag": [12.0]}),
        _minimal_summary(),
        observer_code="TEST",
        cfg=AppConfig(),
        export_method="aperture",
    )
    assert "aavso" in paths
    assert "varastro" in paths


def test_export_all_skips_psf_files_for_submission(tmp_path: Path) -> None:
    """Internal PSF LC files must not be routed into AAVSO/VarAstro writers."""
    phot = tmp_path / "photometry"
    lc_dir = phot / "lightcurves"
    reports = phot / "lightcurves_reports"
    lc_dir.mkdir(parents=True)
    reports.mkdir(parents=True)
    cid = "1498613634033133184"
    _minimal_lc().to_csv(lc_dir / f"lightcurve_{cid}.csv", index=False)
    (lc_dir / f"lightcurve_{cid}_psf.csv").write_text(
        "# INTERNAL DIAGNOSTIC PRODUCT - NOT FOR AAVSO/VARASTRO SUBMISSION\n"
        "bjd,psf_delta_mag\n2460000.5,0.01\n",
        encoding="ascii",
    )
    cfg = AppConfig()
    cfg.psf_photometry_enabled = True
    out = export_all_method_lightcurve_reports(
        reports,
        pd.Series({"vsx_name": "BO CVn", "vsx_type": "LB", "catalog_id": cid}),
        lc_dir=lc_dir,
        target_cid=cid,
        comp_df=pd.DataFrame({"catalog_id": ["9"], "mag": [12.0]}),
        summary_row=_minimal_summary(),
        observer_code="TEST",
        cfg=cfg,
    )
    assert "aperture" in out
    assert "psf" not in out
    aavso_dir = reports / "aavso"
    names = [p.name for p in aavso_dir.glob("*.txt")] if aavso_dir.is_dir() else []
    assert not any("_psf" in n for n in names)


def _synthetic_draft(tmp_path: Path) -> tuple[Path, Path, str]:
    ps = tmp_path / "platesolve"
    frames = tmp_path / "frames"
    phot = ps / "photometry"
    lc_dir = phot / "lightcurves"
    ps.mkdir()
    frames.mkdir()
    lc_dir.mkdir(parents=True)
    fits.PrimaryHDU(np.ones((9, 9), dtype=np.float32)).writeto(ps / "masterstar_epsf.fits")
    (ps / "masterstar_epsf_meta.json").write_text(
        json.dumps(
            {
                "n_stars_used": 67,
                "created_utc": "2026-08-22T19:11:31Z",
                "oversampling": 2,
                "smoothing_kernel": "quadratic",
                "cutout_size": 17,
            }
        ),
        encoding="ascii",
    )
    phot.joinpath("gain_photon_transfer.json").write_text(
        json.dumps(
            {
                "authority": {
                    "g_pt": 0.63707,
                    "source": "g_pt",
                    "value_e_per_adu_container": 0.63707,
                }
            }
        ),
        encoding="ascii",
    )
    target = "1001"
    comps = ["2001", "2002", "2003"]
    epochs = [
        ("proc_Light_001.csv", "Light_001.fits", 2460000.1),
        ("proc_Light_002.csv", "Light_002.fits", 2460000.2),
        ("proc_Light_003.csv", "Light_003.fits", 2460000.3),
    ]
    ids = [target] + comps
    for proc_name, fits_name, bjd in epochs:
        rows = []
        for cid in ids:
            psf_ok = not (proc_name.endswith("002.csv") and cid == target)
            flux = 1000.0 if cid == target else 800.0
            rows.append(
                {
                    "catalog_id": cid,
                    "source_file": fits_name,
                    "bjd_tdb_mid": bjd,
                    "hjd_mid": bjd,
                    "jd_mid": bjd,
                    "psf_flux": flux if psf_ok else np.nan,
                    "psf_flux_err": 10.0 if psf_ok else np.nan,
                    "psf_chi2": 1.5 if psf_ok else np.nan,
                    "psf_fit_ok": psf_ok,
                    "flux": flux * 1.1,
                    "dao_flux": flux * 1.1,
                }
            )
        pd.DataFrame(rows).to_csv(frames / proc_name, index=False)
    ap = pd.DataFrame(
        {
            "bjd": [e[2] for e in epochs],
            "hjd": [e[2] for e in epochs],
            "jd": [e[2] for e in epochs],
            "source_file": [e[0] for e in epochs],
            "delta_mag": [0.01, 0.02, 0.015],
            "err": [0.005, 0.006, 0.005],
        }
    )
    ap.to_csv(lc_dir / f"lightcurve_{target}.csv", index=False)
    pd.DataFrame(
        {
            "catalog_id": comps,
            "target_catalog_id": [target] * 3,
            "comp_weight": [1.0, 1.0, 1.0],
        }
    ).to_csv(phot / "comparison_stars_per_target.csv", index=False)
    return ps, frames, target


def test_writer_header_and_nan_epochs_synthetic(tmp_path: Path) -> None:
    ps, frames, target = _synthetic_draft(tmp_path)
    out = write_internal_psf_lightcurves(
        platesolve_dir=ps,
        frames_root=frames,
        target_ids=[target],
    )
    assert out["n_written"] == 1
    path = Path(out["written"][0])
    text = path.read_text(encoding="utf-8")
    for marker in REQUIRED_HEADER_MARKERS:
        assert marker in text, f"missing provenance marker: {marker}"
    df = pd.read_csv(path, comment="#")
    assert len(df) == 3
    assert bool(df.loc[1, "psf_fit_ok"]) is False
    assert pd.isna(df.loc[1, "psf_delta_mag"])
    assert bool(df.loc[0, "psf_fit_ok"]) is True
    assert np.isfinite(float(df.loc[0, "psf_delta_mag"]))


@pytest.mark.skipif(
    not (PS_516 / "photometry" / "lightcurves" / f"lightcurve_{BO_CVN_ID}.csv").is_file(),
    reason="draft 516 BO CVn aperture LC missing",
)
def test_epsf_lc_log_01_draft516_bo_cvn() -> None:
    """T1 + T2 on live draft 516: PSF LC written; aperture/export bytes unchanged."""
    lc_dir = PS_516 / "photometry" / "lightcurves"
    ap_path = lc_csv_path(lc_dir, BO_CVN_ID, "aperture")
    reports = PS_516 / "photometry" / "lightcurves_reports"
    aavso = sorted((reports / "aavso").glob("BO_CVn*.txt")) if (reports / "aavso").is_dir() else []
    varastro = (
        sorted((reports / "varastro").glob("BO_CVn*.txt")) if (reports / "varastro").is_dir() else []
    )
    watch = [ap_path, *aavso, *varastro]
    before = {str(p): _sha(p) for p in watch if p.is_file()}
    assert before, "no aperture/export files to hash"

    result = write_internal_psf_lightcurves(
        platesolve_dir=PS_516,
        frames_root=FRAMES_516,
        target_ids=[BO_CVN_ID],
    )
    assert result["n_written"] == 1
    psf_path = Path(result["written"][0])
    assert psf_path.is_file()
    text = psf_path.read_text(encoding="utf-8")
    for marker in REQUIRED_HEADER_MARKERS:
        assert marker in text, f"missing provenance marker: {marker}"
    df = pd.read_csv(psf_path, comment="#")
    assert len(df) == N_SCIENCE_LIGHTS
    n_fail = int((~df["psf_fit_ok"].astype(bool)).sum())
    n_nan = int(pd.to_numeric(df["psf_delta_mag"], errors="coerce").isna().sum())
    assert n_fail >= 1
    assert n_nan >= n_fail

    after = {str(p): _sha(p) for p in watch if p.is_file()}
    assert after == before
    assert _sha(psf_path) not in before.values()
