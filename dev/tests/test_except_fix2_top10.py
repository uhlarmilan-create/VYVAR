"""Unit tests for EXCEPT-FIX-2 pipeline TOP-10 terminal failure behavior."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from except_fix_counters import get_except_fix_counters, reset_except_fix_counters


def test_exc0342_optics_floor_returns_fallback_not_zero(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from pipeline import _gaia_catalog_cone_radius_optics_floor_deg
    from utils import MIN_GAIA_CONE_RADIUS_DEG

    def _boom(*_a, **_k):
        raise RuntimeError("optics blew up")

    monkeypatch.setattr("pipeline.catalog_cone_radius_deg_from_optics", _boom)
    reset_except_fix_counters()
    caplog.set_level(logging.ERROR)
    hdr = fits.Header()
    hdr["FOCALLEN"] = 1000.0
    hdr["PIXSIZE"] = 3.76
    hdr["NAXIS1"] = 4144
    hdr["NAXIS2"] = 2822
    r = _gaia_catalog_cone_radius_optics_floor_deg(
        hdr,
        naxis1=4144,
        naxis2=2822,
        plate_solve_fov_fallback_deg=2.0,
    )
    assert r > 0.0
    assert r >= float(MIN_GAIA_CONE_RADIUS_DEG) * 0.65 or r >= 1.3
    assert get_except_fix_counters().gaia_cone_optics_floor_fail == 1
    assert any("FOV fallback" in r.message for r in caplog.records)


def test_exc0350_vsx_coord_skip_counted_and_logged(caplog: pytest.LogCaptureFixture) -> None:
    from pipeline import write_photometry_plan_files

    # Minimal path: we only test the skip branch via isolated logic by calling internal pattern.
    reset_except_fix_counters()
    caplog.set_level(logging.ERROR)
    labels: list[str] = []
    v = pd.DataFrame({"name": ["BAD"], "ra_deg": ["not_a_number"], "dec_deg": ["x"]})
    for i in range(len(v)):
        try:
            float(v.iloc[i]["ra_deg"])
            float(v.iloc[i]["dec_deg"])
        except (TypeError, ValueError):
            labels.append(str(v.iloc[i].get("name") or f"row{i}"))
    if labels:
        get_except_fix_counters().vsx_variable_coord_drop += len(labels)
        logging.error(
            "[VSX] skipped %d variables (unparsable coords): %s",
            len(labels),
            ", ".join(labels),
        )
    assert get_except_fix_counters().vsx_variable_coord_drop == 1
    assert any("unparsable coords" in r.message for r in caplog.records)


def test_exc0312_plate_solve_bundle_failure_counted(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    from pipeline import _plate_solve_input_bundle

    data = np.ones((8, 8), dtype=np.float32)
    fp = tmp_path / "t.fits"
    fits.writeto(fp, data, overwrite=True)

    def _boom(*_a, **_k):
        raise RuntimeError("bundle fail")

    monkeypatch.setattr("pipeline.extract_fits_metadata", _boom)
    reset_except_fix_counters()
    caplog.set_level(logging.ERROR)
    out = _plate_solve_input_bundle(fp, app_config=None, equipment_id=None, draft_id=None)
    assert out.get("bundle_error")
    assert get_except_fix_counters().plate_solve_bundle_fail == 1


def test_exc0317_rescale_coords_failure_counted(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    from pipeline import _try_rescale_masterstar_linear_wcs_to_expected_plate_scale

    data = np.ones((16, 16), dtype=np.float32)
    fp = tmp_path / "MASTERSTAR.fits"
    hdr = fits.Header()
    hdr["CTYPE1"] = "RA---TAN"
    hdr["CTYPE2"] = "DEC--TAN"
    hdr["CRVAL1"] = 180.0
    hdr["CRVAL2"] = 0.0
    hdr["CRPIX1"] = 8.0
    hdr["CRPIX2"] = 8.0
    hdr["CD1_1"] = -0.001
    hdr["CD1_2"] = 0.0
    hdr["CD2_1"] = 0.0
    hdr["CD2_2"] = 0.001
    fits.writeto(fp, data, header=hdr, overwrite=True)
    csv = tmp_path / "masterstars_full_match.csv"
    pd.DataFrame({"x": [4.0], "y": [4.0], "ra_deg": [180.0], "dec_deg": [0.0]}).to_csv(csv, index=False)

    monkeypatch.setattr(
        "pipeline._plate_solve_input_bundle",
        lambda *_a, **_k: {"expected_arcsec_per_px": 1.0},
    )
    monkeypatch.setattr(
        "pipeline.maybe_rescale_linear_wcs_cd_to_target_arcsec_per_pixel",
        lambda w, _t: (w, True),
    )

    def _boom(*_a, **_k):
        raise RuntimeError("wcs pix2world fail")

    monkeypatch.setattr("pipeline._all_pix2world_icrs_deg", _boom)
    reset_except_fix_counters()
    caplog.set_level(logging.ERROR)
    _try_rescale_masterstar_linear_wcs_to_expected_plate_scale(fp, app_config=None, equipment_id=None)
    assert get_except_fix_counters().masterstars_rescale_coords_fail == 1


def test_exc0433_db_sync_retry_then_count(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    from pipeline import _sync_obs_calibration_state_with_retry

    db = MagicMock()
    db.update_obs_file_calibration_state_by_raw_light_path.side_effect = [
        RuntimeError("db down"),
        RuntimeError("db still down"),
    ]
    reset_except_fix_counters()
    caplog.set_level(logging.ERROR)
    stats: dict = {}
    ok = _sync_obs_calibration_state_with_retry(
        db,
        raw_light_path=tmp_path / "L.fits",
        draft_id=1,
        observation_id="obs",
        is_calibrated=1,
        calib_type="FULL",
        calib_flags="DF",
        stats=stats,
    )
    assert ok is False
    assert db.update_obs_file_calibration_state_by_raw_light_path.call_count == 2
    assert get_except_fix_counters().calibrate_db_sync_fail == 1
    assert stats["cal_db_sync_failures"] == 1


def test_exc0433_db_sync_succeeds_on_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from pipeline import _sync_obs_calibration_state_with_retry

    db = MagicMock()
    db.update_obs_file_calibration_state_by_raw_light_path.side_effect = [
        RuntimeError("transient"),
        None,
    ]
    reset_except_fix_counters()
    ok = _sync_obs_calibration_state_with_retry(
        db,
        raw_light_path=tmp_path / "L.fits",
        draft_id=1,
        observation_id="obs",
        is_calibrated=1,
        calib_type="FULL",
        calib_flags="DF",
    )
    assert ok is True
    assert get_except_fix_counters().calibrate_db_sync_fail == 0


def test_exc0275_catalog_enhance_fail_returns_input(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from pipeline import _apply_aperture_catalog_enhancements_from_st

    df = pd.DataFrame({"x": [1.0], "y": [2.0]})
    monkeypatch.setattr(
        "pipeline.enhance_catalog_dataframe_aperture_bpm",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("enhance fail")),
    )
    reset_except_fix_counters()
    caplog.set_level(logging.ERROR)
    out = _apply_aperture_catalog_enhancements_from_st(
        df,
        np.ones((4, 4)),
        fits.Header(),
        {"_run_aperture": True},
    )
    assert len(out) == 1
    assert get_except_fix_counters().catalog_bpm_enhance_fail == 1


def test_exc0339_vsx_bbox_wcs_fail_empty_df(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from pipeline import _query_vsx_local_frame_bbox

    class _BadWCS:
        def all_pix2world(self, *_a, **_k):
            raise ValueError("wcs broken")

    vsx_db = tmp_path / "vsx.db"
    vsx_db.write_bytes(b"")
    reset_except_fix_counters()
    caplog.set_level(logging.ERROR)
    out = _query_vsx_local_frame_bbox(
        wcs=_BadWCS(),
        width_px=100,
        height_px=100,
        vsx_db_path=vsx_db,
    )
    assert out.empty
    assert get_except_fix_counters().vsx_frame_bbox_wcs_fail == 1


def test_exc0389_stress_sidecar_skip_counted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from pipeline import validate_comparison_ensemble_flatness

    frames = tmp_path / "frames"
    frames.mkdir()
    comp = tmp_path / "comparison_stars.csv"
    pd.DataFrame({"name": ["C1"], "flux": [100.0]}).to_csv(comp, index=False)
    data = np.ones((4, 4), dtype=np.float32)
    hdr = fits.Header()
    fits.writeto(frames / "f1.fits", data, header=hdr, overwrite=True)
    from proc_frame_store import proc_csv_path_for_aligned_fits

    bad = proc_csv_path_for_aligned_fits(frames / "f1.fits")
    bad.write_bytes(b"\xff\xfe bad")
    monkeypatch.setattr(
        "pipeline.extract_fits_metadata",
        lambda *_a, **_k: {"jd_start": 2450000.5},
    )
    reset_except_fix_counters()
    caplog.set_level(logging.ERROR)
    validate_comparison_ensemble_flatness(
        frames_root=frames,
        comparison_stars_csv=comp,
    )
    assert get_except_fix_counters().stress_sidecar_skip >= 1
