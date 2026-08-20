"""DAO-GAIA-ERA-01: self-calibration derivation + certificate gate tests."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
MS_DIR = ROOT / "Archive" / "Drafts" / "draft_000516" / "platesolve" / "NoFilter_60_2"


def test_derive_match_radius_from_detection_identity_p95() -> None:
    from dao_gaia_calibration import derive_tolerances_from_residuals

    identity = np.linspace(0.2, 2.5, 200)
    faint = np.linspace(0.8, 1.6, 80)
    tol = derive_tolerances_from_residuals(
        identity,
        faint,
        fwhm_px=5.3,
        plate_scale_arcsec_per_px=1.3,
        pass1_sigma=4.5,
        pass2_sigma=4.0,
        match_k=1.7,
        centroid_floor_px=1.0,
        centroid_cap_px=3.0,
    )
    id_p95 = float(np.percentile(identity, 95))
    faint_p95 = float(np.percentile(faint, 95))
    assert abs(tol.residual_p95_px - id_p95) < 1e-6
    assert abs(tol.match_radius_px - round(1.7 * id_p95 * 2) / 2) < 1e-6
    assert abs(tol.pass2_center_tol_px - round(max(1.0, min(3.0, faint_p95)) * 2) / 2) < 1e-6
    assert tol.detection_identity is not None
    assert tol.seed_centroid is not None
    assert tol.detection_identity.n == 200
    assert tol.seed_centroid.n == 80


def test_seed_centroid_offsets_from_census_join() -> None:
    from dao_gaia_calibration import compute_seed_acceptance_centroid_offsets_px

    ms = pd.DataFrame(
        {
            "catalog_id": ["p2a", "p2b"],
            "x": [10.0, 20.0],
            "y": [10.0, 20.5],
            "vy_dao_pass": [2, 2],
        }
    )
    census = pd.DataFrame(
        {
            "catalog_id": ["p2a", "p2b", "seed1"],
            "source_state": ["DETECTED_P2", "DETECTED_P2", "FORCED_SEED"],
            "g_mag": [14.0, 14.0, 14.5],
            "x_gaia": [10.0, 20.0, 30.0],
            "y_gaia": [10.0, 21.0, 30.0],
            "seed_centroid_px": [float("nan"), float("nan"), 1.25],
        }
    )
    dr, label = compute_seed_acceptance_centroid_offsets_px(ms, census)
    assert "FORCED_SEED" in label
    assert dr.size == 3
    assert abs(float(dr[1]) - 0.5) < 1e-6
    assert abs(float(dr[2]) - 1.25) < 1e-6


def test_derive_reproduces_hand_validated_rig_when_identity_p95_high() -> None:
    """When identity p95 ~1.78 px, match ~3 px; seed p95 ~1.9 -> centroid ~2.0 px."""
    from dao_gaia_calibration import derive_tolerances_from_diagnostic, DiagnosticPopulationStats

    id_diag = DiagnosticPopulationStats(
        name="detection_identity",
        n=500,
        p50_px=0.9,
        p95_px=1.78,
        n_raw=500,
        p50_raw_px=0.9,
        p95_raw_px=1.78,
        tail_estimate_px=15.0,
        tail_method="test",
        measurement_mode="diagnostic",
        diagnostic_radius_px=10.0,
    )
    seed_diag = DiagnosticPopulationStats(
        name="seed_centroid",
        n=120,
        p50_px=1.0,
        p95_px=1.9,
        n_raw=120,
        p50_raw_px=1.0,
        p95_raw_px=1.9,
        tail_estimate_px=15.0,
        tail_method="test",
        measurement_mode="diagnostic",
    )
    tol = derive_tolerances_from_diagnostic(
        id_diag,
        seed_diag,
        fwhm_px=5.3,
        plate_scale_arcsec_per_px=1.3,
        pass1_sigma=4.5,
        pass2_sigma=4.0,
        match_k=1.7,
    )
    assert abs(tol.match_radius_px - 3.0) <= 0.5
    assert abs(tol.pass2_center_tol_px - 2.0) <= 0.5


def test_certificate_present_and_pass_on_draft_516_backfill() -> None:
    """Backfill gate: run calibration on live 516 MS build inputs (validation off in unit test)."""
    from astropy.io import fits
    from astropy.wcs import WCS
    from warnings import catch_warnings, simplefilter
    from astropy.wcs import FITSFixedWarning

    from config import AppConfig
    from dao_gaia_calibration import build_calibration_certificate
    from plain_stats import plain_mean_med_std
    from gaia_catalog_id import read_vyvar_csv

    if not (MS_DIR / "MASTERSTAR.fits").is_file():
        pytest.skip("draft 516 MS not present")
    with fits.open(MS_DIR / "MASTERSTAR.fits", memmap=False) as hdul:
        raw = np.asarray(hdul[0].data, dtype=np.float32)
        hdr = hdul[0].header
        fwhm = float(hdr.get("VY_FWHM", 5.3))
    _, med, _ = plain_mean_med_std(raw, sigma=3.0, maxiters=3)
    data0 = np.nan_to_num((raw - med).astype(np.float32), nan=0.0)
    with catch_warnings():
        simplefilter("ignore", FITSFixedWarning)
        wcs = WCS(hdr)
    ms = read_vyvar_csv(MS_DIR / "masterstars_full_match.csv", low_memory=False, dtype={"catalog_id": str})
    census = read_vyvar_csv(MS_DIR / "gaia_source_state_census.csv", low_memory=False, dtype={"catalog_id": str})
    cfg = AppConfig()
    cert = build_calibration_certificate(
        setup="NoFilter_60_2",
        wcs_obj=wcs,
        data0=data0,
        dao_x=pd.to_numeric(ms["x"], errors="coerce").to_numpy(dtype=np.float64),
        dao_y=pd.to_numeric(ms["y"], errors="coerce").to_numpy(dtype=np.float64),
        gaia_x=pd.to_numeric(census["x_gaia"], errors="coerce").to_numpy(dtype=np.float64),
        gaia_y=pd.to_numeric(census["y_gaia"], errors="coerce").to_numpy(dtype=np.float64),
        gaia_g=pd.to_numeric(census["g_mag"], errors="coerce").to_numpy(dtype=np.float64),
        fwhm_px=fwhm,
        pass1_sigma=float(cfg.masterstar_dao_threshold_sigma),
        pass2_sigma=float(cfg.masterstar_dao_pass2_sigma),
        seed_snr_min=float(cfg.masterstar_forced_seed_snr_min),
        target_depth_g=float(cfg.masterstar_gaia_census_target_depth_g),
        edge_margin_px=float(cfg.masterstar_gaia_census_edge_margin_px),
        cfg=cfg,
        ms_df=ms,
        census_df=census,
        run_validation=False,
    )
    assert cert.status == "PASS", cert.fail_reason
    assert cert.empty_sky.inv_det == "PASS"
    assert cert.empty_sky.inv_seed == "PASS"
    assert cert.empty_sky.n_positions >= 2000
    assert cert.derived.detection_identity is not None
    assert cert.derived.seed_centroid is not None
    assert cert.derived.diagnostic is not None
    assert cert.derived.detection_identity.n > 0
    assert cert.derived.seed_centroid.n > 0
    assert cert.derived.diagnostic.detection_identity.n_raw > 0
    assert abs(cert.derived.pass1_sigma - 4.5) < 0.01
    assert abs(cert.derived.pass2_sigma - 4.0) < 0.01


@pytest.mark.slow
def test_validation_gate_516_runs_and_reports() -> None:
    """A-fix 2 integration: gate must run and return structured result (PASS required for Part C)."""
    from astropy.io import fits
    from astropy.wcs import WCS
    from warnings import catch_warnings, simplefilter
    from astropy.wcs import FITSFixedWarning

    from config import AppConfig
    from dao_gaia_calibration import build_calibration_certificate
    from plain_stats import plain_mean_med_std
    from gaia_catalog_id import read_vyvar_csv

    if not (MS_DIR / "MASTERSTAR.fits").is_file():
        pytest.skip("draft 516 MS not present")
    with fits.open(MS_DIR / "MASTERSTAR.fits", memmap=False) as hdul:
        raw = np.asarray(hdul[0].data, dtype=np.float32)
        hdr = hdul[0].header
        fwhm = float(hdr.get("VY_FWHM", 5.3))
    _, med, _ = plain_mean_med_std(raw, sigma=3.0, maxiters=3)
    data0 = np.nan_to_num((raw - med).astype(np.float32), nan=0.0)
    with catch_warnings():
        simplefilter("ignore", FITSFixedWarning)
        wcs = WCS(hdr)
    ms = read_vyvar_csv(MS_DIR / "masterstars_full_match.csv", low_memory=False, dtype={"catalog_id": str})
    census = read_vyvar_csv(MS_DIR / "gaia_source_state_census.csv", low_memory=False, dtype={"catalog_id": str})
    cfg = AppConfig()
    cert = build_calibration_certificate(
        setup="NoFilter_60_2",
        wcs_obj=wcs,
        data0=data0,
        dao_x=pd.to_numeric(ms["x"], errors="coerce").to_numpy(dtype=np.float64),
        dao_y=pd.to_numeric(ms["y"], errors="coerce").to_numpy(dtype=np.float64),
        gaia_x=pd.to_numeric(census["x_gaia"], errors="coerce").to_numpy(dtype=np.float64),
        gaia_y=pd.to_numeric(census["y_gaia"], errors="coerce").to_numpy(dtype=np.float64),
        gaia_g=pd.to_numeric(census["g_mag"], errors="coerce").to_numpy(dtype=np.float64),
        fwhm_px=fwhm,
        pass1_sigma=float(cfg.masterstar_dao_threshold_sigma),
        pass2_sigma=float(cfg.masterstar_dao_pass2_sigma),
        seed_snr_min=float(cfg.masterstar_forced_seed_snr_min),
        target_depth_g=float(cfg.masterstar_gaia_census_target_depth_g),
        edge_margin_px=float(cfg.masterstar_gaia_census_edge_margin_px),
        cfg=cfg,
        ms_df=ms,
        census_df=census,
        run_validation=True,
        repo_root=ROOT,
    )
    derived = cert.derived
    gate = cert.validation
    assert gate is not None
    assert gate.hand_scores
    assert gate.derived_scores
    assert gate.g2_pass
    assert isinstance(gate.regressions, dict)
    if gate.status != "PASS":
        pytest.skip(f"validation gate FAIL: {gate.fail_reason}")


def test_certificate_file_gate(tmp_path: Path) -> None:
    from dao_gaia_calibration import (
        CERT_FILENAME,
        DaoGaiaCalibrationCertificate,
        DerivedTolerances,
        EmptySkyAudit,
        PopulationStats,
        write_calibration_certificate,
    )
    from invariants_runtime import InvariantViolation

    pop_id = PopulationStats(name="detection_identity", n=100, p50_px=1.0, p95_px=1.78)
    pop_faint = PopulationStats(
        name="faint_star_centroid", n=50, p50_px=0.9, p95_px=1.2, g_band="13.0-15.0"
    )
    ok = DaoGaiaCalibrationCertificate(
        setup="test",
        built_utc="2026-08-19T00:00:00+00:00",
        status="PASS",
        fail_reason=None,
        derived=DerivedTolerances(
            residual_p95_px=1.78,
            match_radius_px=3.0,
            pass2_center_tol_px=2.0,
            lock_pair_tol_px=3.0,
            lock_leftover_radius_px=3.0,
            forced_seed_centroid_max_px=2.0,
            plate_scale_arcsec_per_px=1.3,
            fwhm_px=5.3,
            pass1_sigma=4.5,
            pass2_sigma=4.0,
            detection_identity=pop_id,
            faint_star_centroid=pop_faint,
        ),
        empty_sky=EmptySkyAudit(
            n_positions=100,
            pass2_accept=0,
            pass2_rate=0.0,
            seed_accept=0,
            seed_rate=0.0,
            inv_det="PASS",
            inv_seed="PASS",
        ),
    )
    path = write_calibration_certificate(ok, tmp_path, fail_closed=True)
    assert path.name == CERT_FILENAME
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "PASS"

    bad = DaoGaiaCalibrationCertificate(
        setup="test",
        built_utc="2026-08-19T00:00:00+00:00",
        status="FAIL",
        fail_reason="test fail",
        derived=ok.derived,
        empty_sky=ok.empty_sky,
    )
    with pytest.raises(InvariantViolation):
        write_calibration_certificate(bad, tmp_path / "fail", fail_closed=True)
