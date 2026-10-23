"""EPSF-LC-LOG-01: internal PSF LC writer + INV-PSF-SUBMIT-01."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config import AppConfig
from export_reports import export_all_method_lightcurve_reports, export_lightcurve_reports
from invariants_runtime import InvariantViolation
from photometry_core import TIME_BASE_BJD_TDB
from psf_internal_lc import (
    INV_PSF_LC_PIN_01,
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
            "jd": [2460000.5],
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
    return pd.Series({"band_system": "V", "n_good_comp": 4})


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
            obs_group="NoFilter_60_2",  # production passes the setup (OUTPUT-REVIEW step 6)
            export_method="psf",
        )


def test_inv_psf_submit_01_rejects_any_non_aperture_method(tmp_path: Path) -> None:
    with pytest.raises(InvariantViolation, match=INV_PSF_SUBMIT_01):
        export_lightcurve_reports(
            tmp_path / "reports",
            _minimal_target(),
            _minimal_lc(),
            pd.DataFrame({"catalog_id": ["9"], "mag": [12.0]}),
            _minimal_summary(),
            cfg=AppConfig(),
            obs_group="NoFilter_60_2",  # production passes the setup (OUTPUT-REVIEW step 6)
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
        obs_group="NoFilter_60_2",  # production passes the setup (OUTPUT-REVIEW step 6)
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
    out = export_all_method_lightcurve_reports(
        reports,
        pd.Series({"vsx_name": "BO CVn", "vsx_type": "LB", "catalog_id": cid}),
        lc_dir=lc_dir,
        target_cid=cid,
        comp_df=pd.DataFrame({"catalog_id": ["9"], "mag": [12.0]}),
        summary_row=_minimal_summary(),
        observer_code="TEST",
        cfg=cfg,
        obs_group="NoFilter_60_2",  # production passes the setup (OUTPUT-REVIEW step 6)
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
    (ps / "epsf_frames").mkdir()
    (ps / "epsf_frames" / "epsf_frames_manifest.json").write_text(
        json.dumps(
            {
                "n_build_stars": 67,
                "created_utc": "2026-08-22T19:11:31Z",
                "oversampling": 2,
                "cutout_size": 17,
                "frames": [{"frame": "f001.fits", "sha256": "0" * 64, "smoothing_kernel": "quadratic"}],
            }
        ),
        encoding="ascii",
    )
    phot.joinpath("aperture_per_target.json").write_text(
        json.dumps({"gain": 0.63707, "read_noise": 3.0, "targets": {}}), encoding="ascii"
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
            target_fail = proc_name.endswith("002.csv") and cid == target
            comp_fail = proc_name.endswith("003.csv") and cid == comps[0]
            psf_ok = not (target_fail or comp_fail)
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
    assert "n_group" not in df.columns
    assert "# psf_err_mode=fisher_model_variance" in text
    assert "# psf_fit=frame_psfphotometry_epsf_spatial_local_bkg" in text
    assert "psf_epoch_drop_reason" in df.columns
    assert pd.isna(df.loc[2, "psf_delta_mag"])
    assert str(df.loc[2, "psf_epoch_drop_reason"]) == f"comp_psf_fail:{2001}"
    # OUTPUT-REVIEW-2026-10 step 6: the aperture values stay in the aperture LC only
    assert "delta_mag" not in df.columns and "err" not in df.columns and "product" not in df.columns
    assert {"bjd", "hjd", "jd"} <= set(df.columns)
    assert "# product=internal_psf_diagnostic" in text
    assert INV_PSF_LC_PIN_01 == "INV-PSF-LC-PIN-01"


def test_inv_psf_lc_pin_01_failed_comp_not_renormalized(tmp_path: Path) -> None:
    """Partial ensemble must not leak a ZP; pin-drop is NaN + named reason."""
    ps, frames, target = _synthetic_draft(tmp_path)
    out = write_internal_psf_lightcurves(
        platesolve_dir=ps,
        frames_root=frames,
        target_ids=[target],
    )
    df = pd.read_csv(Path(out["written"][0]), comment="#")
    assert pd.isna(df.loc[2, "psf_delta_mag"])
    reason = str(df.loc[2, "psf_epoch_drop_reason"])
    assert reason.startswith("comp_psf_fail:")
    assert "2001" in reason
    assert np.isfinite(float(df.loc[0, "psf_delta_mag"]))
    assert str(df.loc[0, "psf_epoch_drop_reason"]) in ("", "nan")


def test_p4_invariance_scalar_ac_cancels_in_writer(tmp_path: Path) -> None:
    """Scalar AC on/off must not change psf_delta_mag (ZP cancel)."""
    ps, frames, target = _synthetic_draft(tmp_path)
    out1 = write_internal_psf_lightcurves(
        platesolve_dir=ps,
        frames_root=frames,
        target_ids=[target],
    )
    d1 = pd.read_csv(Path(out1["written"][0]), comment="#")
    for p in frames.glob("proc_*.csv"):
        df = pd.read_csv(p)
        flux = pd.to_numeric(df["psf_flux"], errors="coerce")
        df["psf_flux"] = flux * 0.528
        df.to_csv(p, index=False)
    out2 = write_internal_psf_lightcurves(
        platesolve_dir=ps,
        frames_root=frames,
        target_ids=[target],
    )
    d2 = pd.read_csv(Path(out2["written"][0]), comment="#")
    a = pd.to_numeric(d1["psf_delta_mag"], errors="coerce").to_numpy()
    b = pd.to_numeric(d2["psf_delta_mag"], errors="coerce").to_numpy()
    both = np.isfinite(a) & np.isfinite(b)
    assert int(both.sum()) >= 1
    assert float(np.nanmax(np.abs(a[both] - b[both]))) < 1e-12


@pytest.mark.skipif(
    not (PS_516 / "photometry" / "lightcurves" / f"lightcurve_{BO_CVN_ID}.csv").is_file(),
    reason="draft 516 BO CVn aperture LC missing",
)
def test_epsf_lc_log_01_draft516_bo_cvn(tmp_path: Path) -> None:
    """T1 + T2 on live draft 516: PSF LC written to tmp; live BO PSF LC SHA unchanged."""
    lc_dir = PS_516 / "photometry" / "lightcurves"
    ap_path = lc_csv_path(lc_dir, BO_CVN_ID, "aperture")
    live_psf = lc_csv_path(lc_dir, BO_CVN_ID, "psf")
    live_psf_sha = _sha(live_psf) if live_psf.is_file() else None
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
        output_directory=tmp_path,
    )
    assert result["n_written"] == 1
    psf_path = Path(result["written"][0])
    assert psf_path.is_file()
    assert psf_path.parent.resolve() == tmp_path.resolve()
    text = psf_path.read_text(encoding="utf-8")
    for marker in REQUIRED_HEADER_MARKERS:
        assert marker in text, f"missing provenance marker: {marker}"
    df = pd.read_csv(psf_path, comment="#")
    assert len(df) == N_SCIENCE_LIGHTS
    n_fail = int((~df["psf_fit_ok"].astype(bool)).sum())
    n_nan = int(pd.to_numeric(df["psf_delta_mag"], errors="coerce").isna().sum())
    assert n_fail >= 1
    assert n_nan >= 0

    after = {str(p): _sha(p) for p in watch if p.is_file()}
    assert after == before
    if live_psf_sha is not None:
        assert _sha(live_psf) == live_psf_sha


def test_fit_ok_false_epoch_is_dropped_even_with_finite_flux(tmp_path: Path) -> None:
    """FIT-OK-ADMISSION-01: an epoch with psf_fit_ok=False (finite flux and chi2) is not in the curve."""
    ps, frames, target = _synthetic_draft(tmp_path)
    proc = frames / "proc_Light_001.csv"
    df = pd.read_csv(proc)
    mask = df["catalog_id"].astype(str) == str(target)
    df.loc[mask, "psf_fit_ok"] = False
    df.loc[mask, "psf_chi2"] = 80.0
    df.loc[mask, "psf_flux"] = 1000.0
    df.to_csv(proc, index=False)
    out = write_internal_psf_lightcurves(
        platesolve_dir=ps, frames_root=frames, target_ids=[target], output_directory=tmp_path / "out"
    )
    d = pd.read_csv(Path(out["written"][0]), comment="#")
    assert pd.isna(d.loc[0, "psf_delta_mag"]) and bool(d.loc[0, "psf_fit_ok"]) is False


def test_psf_usable_mask_unit():
    from psf_internal_lc import psf_usable_mask

    assert psf_usable_mask(np.array([True, False, True, True]),
                           np.array([100.0, 100.0, np.nan, -1.0])).tolist() == [True, False, False, False]


def test_output_directory_does_not_rewrite_source_lc_dir(tmp_path: Path) -> None:
    ps, frames, target = _synthetic_draft(tmp_path)
    src_psf = ps / "photometry" / "lightcurves" / f"lightcurve_{target}_psf.csv"
    assert not src_psf.is_file()
    out_dir = tmp_path / "isolated"
    out = write_internal_psf_lightcurves(
        platesolve_dir=ps,
        frames_root=frames,
        target_ids=[target],
        output_directory=out_dir,
    )
    assert Path(out["written"][0]).parent.resolve() == out_dir.resolve()
    assert not src_psf.is_file()


def test_psf_lc_error_adds_the_aperture_scintillation_term(tmp_path: Path) -> None:
    """EPSF-ERR-2026-10: scintillation is per star and epoch, not per method."""
    ps, frames, target = _synthetic_draft(tmp_path)
    lc = ps / "photometry" / "lightcurves" / f"lightcurve_{target}.csv"
    base = write_internal_psf_lightcurves(platesolve_dir=ps, frames_root=frames, target_ids=[target])
    e0 = float(pd.read_csv(base["written"][0], comment="#").loc[0, "psf_delta_mag_err"])
    ap = pd.read_csv(lc)
    ap["err_scint_rel"] = 0.02
    ap.to_csv(lc, index=False)
    out = write_internal_psf_lightcurves(platesolve_dir=ps, frames_root=frames, target_ids=[target])
    e1 = float(pd.read_csv(out["written"][0], comment="#").loc[0, "psf_delta_mag_err"])
    assert e1 > e0
    assert e1 == pytest.approx(np.hypot(e0, 1.0857 * 0.02), rel=0.02)


def test_psf_lc_error_adds_the_night_systematic_floor(tmp_path: Path) -> None:
    """EPSF-SYS-2026-10: the night floor (SIGMA-SYS-NIGHT, flat field / drift) is per star and epoch."""
    ps, frames, target = _synthetic_draft(tmp_path)
    lc = ps / "photometry" / "lightcurves" / f"lightcurve_{target}.csv"
    base = write_internal_psf_lightcurves(platesolve_dir=ps, frames_root=frames, target_ids=[target])
    e0 = float(pd.read_csv(base["written"][0], comment="#").loc[0, "psf_delta_mag_err"])
    ap = pd.read_csv(lc)
    ap["err_sigma_sys_rel"] = 0.006
    ap.to_csv(lc, index=False)
    out = write_internal_psf_lightcurves(platesolve_dir=ps, frames_root=frames, target_ids=[target])
    e1 = float(pd.read_csv(out["written"][0], comment="#").loc[0, "psf_delta_mag_err"])
    assert e1 == pytest.approx(np.hypot(e0, 1.0857 * 0.006), rel=0.02)


def test_saturated_target_epoch_is_dropped_with_reason(tmp_path: Path) -> None:
    """EPSF-BLEND-2026-10: a fit whose box lost pixels to the saturation mask is not used."""
    ps, frames, target = _synthetic_draft(tmp_path)
    p1 = frames / "proc_Light_001.csv"
    df = pd.read_csv(p1, dtype={"catalog_id": str})
    df["psf_sat_masked"] = df["catalog_id"] == target
    df.to_csv(p1, index=False)
    out = write_internal_psf_lightcurves(platesolve_dir=ps, frames_root=frames, target_ids=[target])
    path = Path(out["written"][0])
    lc = pd.read_csv(path, comment="#")
    assert pd.isna(lc.loc[0, "psf_delta_mag"])
    assert str(lc.loc[0, "psf_epoch_drop_reason"]) == "target_saturated"
    assert "# psf_lc_n_epochs_target_saturated=1" in path.read_text(encoding="utf-8")


def test_target_epoch_off_the_aligned_footprint_is_dropped_with_reason(tmp_path: Path) -> None:
    """EPSF-FOOTPRINT-2026-10: a fit whose box or annulus reaches the aligner's constant fill is not used."""
    ps, frames, target = _synthetic_draft(tmp_path)
    p1 = frames / "proc_Light_001.csv"
    df = pd.read_csv(p1, dtype={"catalog_id": str})
    df["psf_edge_masked"] = df["catalog_id"] == target
    df.to_csv(p1, index=False)
    out = write_internal_psf_lightcurves(platesolve_dir=ps, frames_root=frames, target_ids=[target])
    path = Path(out["written"][0])
    lc = pd.read_csv(path, comment="#")
    assert pd.isna(lc.loc[0, "psf_delta_mag"])
    assert str(lc.loc[0, "psf_epoch_drop_reason"]) == "target_outside_footprint"
    assert "# psf_lc_n_epochs_target_outside_footprint=1" in path.read_text(encoding="utf-8")


def test_psf_lc_applies_the_aperture_k2_term(tmp_path: Path) -> None:
    """EPSF-K2-2026-10: the second-order extinction term of the aperture light curve (k'' x colour
    difference x airmass, ``k2_delta_mag``) is subtracted from the PSF light curve epoch by epoch."""
    ps, frames, target = _synthetic_draft(tmp_path)
    lc = ps / "photometry" / "lightcurves" / f"lightcurve_{target}.csv"
    base = write_internal_psf_lightcurves(platesolve_dir=ps, frames_root=frames, target_ids=[target])
    d0 = pd.read_csv(base["written"][0], comment="#")["psf_delta_mag"].to_numpy()
    assert "# psf_lc_k2_applied=false" in Path(base["written"][0]).read_text(encoding="utf-8")
    ap = pd.read_csv(lc)
    ap["k2_delta_mag"] = [-0.010, -0.020, -0.030]
    ap.to_csv(lc, index=False)
    out = write_internal_psf_lightcurves(platesolve_dir=ps, frames_root=frames, target_ids=[target])
    d1 = pd.read_csv(out["written"][0], comment="#")["psf_delta_mag"].to_numpy()
    ok = np.isfinite(d0)
    assert ok.any()
    assert np.allclose(d1[ok] - d0[ok], np.array([0.010, 0.020, 0.030])[ok], atol=2e-6)
    assert "# psf_lc_k2_applied=true" in Path(out["written"][0]).read_text(encoding="utf-8")
