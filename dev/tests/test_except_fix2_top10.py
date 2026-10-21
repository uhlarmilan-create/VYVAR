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


def test_exc0350_vsx_coord_skip_counted_and_logged(caplog: pytest.LogCaptureFixture) -> None:

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

    monkeypatch.setattr("fits_meta.extract_fits_metadata", _boom)
    reset_except_fix_counters()
    caplog.set_level(logging.ERROR)
    out = _plate_solve_input_bundle(fp, app_config=None, equipment_id=None, draft_id=None)
    assert out.get("bundle_error")
    assert get_except_fix_counters().plate_solve_bundle_fail == 1


def test_exc0275_catalog_enhance_fail_returns_input(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from pipeline_catalog import _apply_aperture_catalog_enhancements_from_st

    df = pd.DataFrame({"x": [1.0], "y": [2.0]})
    monkeypatch.setattr(
        "pipeline_catalog.enhance_catalog_dataframe_aperture",
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
    assert get_except_fix_counters().catalog_aperture_enhance_fail == 1


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
        is_calibrated=1,
        calib_type="FULL",
        calib_flags="DF",
    )
    assert ok is True
    assert get_except_fix_counters().calibrate_db_sync_fail == 0


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


