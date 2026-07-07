"""CAL-DIAG radiometry gate (VYVAR_CAL_DIAG_SPEC v1.1)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from cal_diag import (
    CalDiagSession,
    _cal_diag_gate_for_obs_group,
    apply_cal_diag_headers,
    cal_diag_config_from_app,
    cal_diag_gate_key,
    dark_np_for_cal_diag,
    ensure_cal_diag_gate,
    passthrough_cal_diag_headers,
    run_cal_diag_pregate,
    write_cal_diag_json,
)
from calibration import get_processed_master, resample_master_to_light_binning
from config import AppConfig
from pipeline import (
    _cal_diag_session_from_export,
    _match_and_crop_pair,
    calibrate_lights_to_calibrated,
)


def _write_fits(path: Path, data: np.ndarray, **hdr_kw: object) -> None:
    hdr = fits.Header()
    for k, v in hdr_kw.items():
        hdr[k] = v
    fits.writeto(path, np.asarray(data, dtype=np.float32), header=hdr, overwrite=True)


def _match(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return _match_and_crop_pair(a, b)


def test_resample_dark_mean_vs_sum_bin2():
    master = np.full((4, 4), 10.0, dtype=np.float32)
    s, _ = resample_master_to_light_binning(
        master, master_binning=1, light_binning=2, kind="dark", dark_resample_mode="sum"
    )
    m, _ = resample_master_to_light_binning(
        master, master_binning=1, light_binning=2, kind="dark", dark_resample_mode="mean"
    )
    assert float(s[0, 0]) == pytest.approx(40.0)
    assert float(m[0, 0]) == pytest.approx(10.0)


def test_gate_matched_sum_pass(tmp_path: Path):
    light = np.full((2, 2), 120.0, dtype=np.float32)
    dark_master = np.full((4, 4), 10.0, dtype=np.float32)
    lp = tmp_path / "light.fits"
    dp = tmp_path / "dark.fits"
    _write_fits(lp, light, XBINNING=2, YBINNING=2, EXPTIME=30.0, FILTER="V")
    _write_fits(dp, dark_master, XBINNING=1, YBINNING=1, EXPTIME=30.0)

    cfg = cal_diag_config_from_app(AppConfig())
    res = _cal_diag_gate_for_obs_group(
        repr_light_path=lp,
        dark_path=dp,
        obs_group_key="V|30|2",
        light_binning=2,
        master_binning=1,
        gate_cfg=cfg,
        match_and_crop_pair=_match,
        saturation_adu=None,
    )
    assert res.aborted is False
    assert res.convention == "SUM"
    assert res.status == "PASS"


def test_gate_autocorrect_mean_driver(tmp_path: Path):
    # Driver-averaged bin2 light: superpixel level ~25; SUM dark median 40 overshoots.
    light = np.full((2, 2), 25.0, dtype=np.float32)
    dark_master = np.full((4, 4), 10.0, dtype=np.float32)
    lp = tmp_path / "light.fits"
    dp = tmp_path / "dark.fits"
    _write_fits(lp, light, XBINNING=2, YBINNING=2)
    _write_fits(dp, dark_master, XBINNING=1, YBINNING=1)

    cfg = cal_diag_config_from_app(AppConfig())
    res = _cal_diag_gate_for_obs_group(
        repr_light_path=lp,
        dark_path=dp,
        obs_group_key="NoFilter|0|2",
        light_binning=2,
        master_binning=1,
        gate_cfg=cfg,
        match_and_crop_pair=_match,
        saturation_adu=None,
    )
    assert res.aborted is False
    assert res.convention == "MEAN_AUTOCORRECTED"
    assert res.status == "WARN"


def test_gate_fail_closed_garbage_dark(tmp_path: Path):
    light = np.full((2, 2), 50.0, dtype=np.float32)
    dark_master = np.full((4, 4), 200.0, dtype=np.float32)
    lp = tmp_path / "light.fits"
    dp = tmp_path / "dark.fits"
    _write_fits(lp, light, XBINNING=2, YBINNING=2)
    _write_fits(dp, dark_master, XBINNING=1, YBINNING=1)

    cfg = cal_diag_config_from_app(AppConfig())
    res = _cal_diag_gate_for_obs_group(
        repr_light_path=lp,
        dark_path=dp,
        obs_group_key="NoFilter|0|2",
        light_binning=2,
        master_binning=1,
        gate_cfg=cfg,
        match_and_crop_pair=_match,
        saturation_adu=None,
    )
    assert res.aborted is True
    assert res.status == "ABORT"


def test_gate_bf1_pairing_fail(tmp_path: Path):
    light = np.full((4, 4), 50.0, dtype=np.float32)
    dark_master = np.full((4, 4), 200.0, dtype=np.float32)
    lp = tmp_path / "light.fits"
    dp = tmp_path / "dark.fits"
    _write_fits(lp, light, XBINNING=1, YBINNING=1)
    _write_fits(dp, dark_master, XBINNING=1, YBINNING=1)

    cfg = cal_diag_config_from_app(AppConfig())
    res = _cal_diag_gate_for_obs_group(
        repr_light_path=lp,
        dark_path=dp,
        obs_group_key="NoFilter|0|1",
        light_binning=1,
        master_binning=1,
        gate_cfg=cfg,
        match_and_crop_pair=_match,
        saturation_adu=None,
    )
    assert res.aborted is True
    assert res.block_factor == 1


def test_calibrate_byte_identical_gate_off_on(tmp_path: Path):
    """Matched SUM: science arrays identical; CAL-DIAG headers additive only."""
    root = tmp_path / "session"
    lights = root / "Raw" / "lights"
    lights.mkdir(parents=True)
    cal = root / "calibrated" / "lights"
    dark_m = np.full((4, 4), 8.0, dtype=np.float32)
    flat_m = np.full((4, 4), 1000.0, dtype=np.float32)
    light = np.full((2, 2), 108.0, dtype=np.float32)
    dp = tmp_path / "md.fits"
    fp = tmp_path / "mf.fits"
    lp = lights / "L1.fits"
    _write_fits(dp, dark_m, XBINNING=1, YBINNING=1, EXPTIME=10.0)
    _write_fits(fp, flat_m, XBINNING=1, YBINNING=1, FILTER="V", VYFLNRD=1)
    _write_fits(lp, light, XBINNING=2, YBINNING=2, EXPTIME=10.0, FILTER="V")

    cfg_off = AppConfig()
    cfg_off.cal_diag_gate_enabled = False
    cal_off = tmp_path / "cal_off"
    calibrate_lights_to_calibrated(
        lights_root=lights,
        calibrated_root=cal_off,
        master_dark_path=dp,
        masterflat_by_filter={"V|10|2": fp, "V": fp},
        pipeline_config=cfg_off,
    )

    cfg_on = AppConfig()
    cfg_on.cal_diag_gate_enabled = True
    cal_on = tmp_path / "cal_on"
    calibrate_lights_to_calibrated(
        lights_root=lights,
        calibrated_root=cal_on,
        master_dark_path=dp,
        masterflat_by_filter={"V|10|2": fp, "V": fp},
        pipeline_config=cfg_on,
    )

    with fits.open(cal_off / "L1.fits") as h0, fits.open(cal_on / "L1.fits") as h1:
        d0 = np.array(h0[0].data, dtype=np.float32)
        d1 = np.array(h1[0].data, dtype=np.float32)
        assert np.allclose(d0, d1)
        assert h1[0].header.get("VY_DKRSMP") == "SUM"
        assert "VY_DKRSMP" not in h0[0].header


def test_pregate_session_export_roundtrip(tmp_path: Path):
    light = np.full((2, 2), 120.0, dtype=np.float32)
    dark_master = np.full((4, 4), 10.0, dtype=np.float32)
    lp = tmp_path / "light.fits"
    dp = tmp_path / "dark.fits"
    _write_fits(lp, light, XBINNING=2, YBINNING=2)
    _write_fits(dp, dark_master, XBINNING=1, YBINNING=1)

    cfg = cal_diag_config_from_app(AppConfig())
    sess = run_cal_diag_pregate(
        [lp],
        obs_group_key_from_path=lambda p: "NoFilter|0|2",
        resolve_dark_path=lambda fp, og, lb: dp,
        light_binning_from_path=lambda p: 2,
        master_binning=1,
        gate_cfg=cfg,
        match_and_crop_pair=_match,
        saturation_for_light=lambda p: None,
    )
    blob = sess.json_export()
    sess2 = _cal_diag_session_from_export(blob)
    assert len(sess2.gate_results) == len(sess.gate_results)


def test_dark_cache_same_convention(tmp_path: Path):
    dark_master = np.full((4, 4), 10.0, dtype=np.float32)
    dp = tmp_path / "dark.fits"
    lp = tmp_path / "L.fits"
    _write_fits(dp, dark_master, XBINNING=1, YBINNING=1)
    _write_fits(lp, np.full((2, 2), 120.0, dtype=np.float32), XBINNING=2, YBINNING=2)
    session = CalDiagSession()
    gr = ensure_cal_diag_gate(
        session,
        obs_group_key="g",
        repr_light_path=lp,
        dark_path=dp,
        light_binning=2,
        master_binning=1,
        gate_cfg=cal_diag_config_from_app(AppConfig()),
        match_and_crop_pair=_match,
        saturation_adu=None,
    )
    assert gr is not None
    a = dark_np_for_cal_diag(
        session,
        master_binning=1,
        dark_path=dp,
        light_binning=2,
        light_shape=(2, 2),
        light_filename="L.fits",
        gate_result=gr,
        gate_enabled=True,
    )
    b = dark_np_for_cal_diag(
        session,
        master_binning=1,
        dark_path=dp,
        light_binning=2,
        light_shape=(2, 2),
        light_filename="L2.fits",
        gate_result=gr,
        gate_enabled=True,
    )
    assert a is not None and b is not None
    assert np.array_equal(a, b)


def test_write_cal_diag_json(tmp_path: Path):
    _write_fits(tmp_path / "L.fits", np.ones((2, 2), dtype=np.float32), XBINNING=1, YBINNING=1)
    _write_fits(tmp_path / "D.fits", np.zeros((2, 2), dtype=np.float32), XBINNING=1, YBINNING=1)
    session = CalDiagSession()
    session.gate_results["k"] = _cal_diag_gate_for_obs_group(
        repr_light_path=tmp_path / "L.fits",
        dark_path=tmp_path / "D.fits",
        obs_group_key="g",
        light_binning=1,
        master_binning=1,
        gate_cfg=cal_diag_config_from_app(AppConfig()),
        match_and_crop_pair=_match,
        saturation_adu=None,
    )
    out = write_cal_diag_json(tmp_path, session, gate_enabled=True)
    assert out is not None and out.is_file()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "keys" in data


def test_gate_near_zero_sky_warn_not_fail(tmp_path: Path):
    """Post-dark sky slightly negative within hard_sigma band -> WARN, not ABORT."""
    rng = np.random.default_rng(42)
    dark_master = np.full((4, 4), 10.0, dtype=np.float32)
    light = np.full((2, 2), 120.0, dtype=np.float32) + rng.normal(0, 0.01, (2, 2)).astype(np.float32)
    lp = tmp_path / "light.fits"
    dp = tmp_path / "dark.fits"
    _write_fits(lp, light, XBINNING=2, YBINNING=2)
    _write_fits(dp, dark_master, XBINNING=1, YBINNING=1)

    cfg = cal_diag_config_from_app(AppConfig())
    res = _cal_diag_gate_for_obs_group(
        repr_light_path=lp,
        dark_path=dp,
        obs_group_key="NoFilter|0|2",
        light_binning=2,
        master_binning=1,
        gate_cfg=cfg,
        match_and_crop_pair=_match,
        saturation_adu=None,
    )
    assert res.aborted is False
    assert res.status in ("PASS", "WARN")


def test_gate_bf1_fail_log_wording(tmp_path: Path):
    light = np.full((4, 4), 50.0, dtype=np.float32)
    dark_master = np.full((4, 4), 200.0, dtype=np.float32)
    lp = tmp_path / "light.fits"
    dp = tmp_path / "dark.fits"
    _write_fits(lp, light, XBINNING=1, YBINNING=1)
    _write_fits(dp, dark_master, XBINNING=1, YBINNING=1)

    cfg = cal_diag_config_from_app(AppConfig())
    res = _cal_diag_gate_for_obs_group(
        repr_light_path=lp,
        dark_path=dp,
        obs_group_key="NoFilter|0|1",
        light_binning=1,
        master_binning=1,
        gate_cfg=cfg,
        match_and_crop_pair=_match,
        saturation_adu=None,
    )
    assert res.aborted is True
    msg = " ".join(res.messages)
    assert "bf=1" in msg or "wrong master pairing" in msg


def test_passthrough_provenance_headers():
    hdr = fits.Header()
    passthrough_cal_diag_headers(hdr, gate_enabled=True)
    assert hdr.get("VY_DKRSMP") == "PASSTHROUGH"
    assert hdr.get("VY_CDSTAT") == "PASS"
    hdr2 = fits.Header()
    passthrough_cal_diag_headers(hdr2, gate_enabled=False)
    assert "VY_DKRSMP" not in hdr2


def test_path_coverage_pregate_same_key(tmp_path: Path):
    """RAM-QC / MP parent pregate: same key yields one stored gate result."""
    light = np.full((2, 2), 120.0, dtype=np.float32)
    dark_master = np.full((4, 4), 10.0, dtype=np.float32)
    lp1 = tmp_path / "a_light.fits"
    lp2 = tmp_path / "b_light.fits"
    dp = tmp_path / "dark.fits"
    _write_fits(lp1, light, XBINNING=2, YBINNING=2)
    _write_fits(lp2, light, XBINNING=2, YBINNING=2)
    _write_fits(dp, dark_master, XBINNING=1, YBINNING=1)

    cfg = cal_diag_config_from_app(AppConfig())
    sess = run_cal_diag_pregate(
        [lp2, lp1],
        obs_group_key_from_path=lambda p: "NoFilter|0|2",
        resolve_dark_path=lambda fp, og, lb: dp,
        light_binning_from_path=lambda p: 2,
        master_binning=1,
        gate_cfg=cfg,
        match_and_crop_pair=_match,
        saturation_for_light=lambda p: None,
    )
    gkey = cal_diag_gate_key("NoFilter|0|2", dp, 2)
    assert gkey in sess.gate_results
    assert len(sess.gate_results) == 1
    blob = sess.json_export()
    sess2 = _cal_diag_session_from_export(blob)
    assert sess2.gate_results[gkey].convention == sess.gate_results[gkey].convention


def test_fail_closed_aborted_group_no_output(tmp_path: Path):
    """Garbage dark aborts one obs_group; sibling group still calibrates."""
    root = tmp_path / "session"
    lights = root / "Raw" / "lights"
    lights.mkdir(parents=True)
    dark_ok = np.full((4, 4), 8.0, dtype=np.float32)
    dark_bad = np.full((4, 4), 500.0, dtype=np.float32)
    flat_m = np.full((4, 4), 1000.0, dtype=np.float32)
    light_ok = np.full((2, 2), 108.0, dtype=np.float32)
    light_bad = np.full((2, 2), 50.0, dtype=np.float32)
    dp_ok = tmp_path / "md_ok.fits"
    dp_bad = tmp_path / "md_bad.fits"
    fp = tmp_path / "mf.fits"
    _write_fits(dp_ok, dark_ok, XBINNING=1, YBINNING=1, EXPTIME=10.0)
    _write_fits(dp_bad, dark_bad, XBINNING=1, YBINNING=1, EXPTIME=20.0)
    _write_fits(fp, flat_m, XBINNING=1, YBINNING=1, FILTER="V", VYFLNRD=1)
    _write_fits(lights / "good.fits", light_ok, XBINNING=2, YBINNING=2, EXPTIME=10.0, FILTER="V")
    _write_fits(lights / "bad.fits", light_bad, XBINNING=2, YBINNING=2, EXPTIME=20.0, FILTER="V")

    cfg = AppConfig()
    cfg.cal_diag_gate_enabled = True
    cal_out = tmp_path / "cal_out"
    stats = calibrate_lights_to_calibrated(
        lights_root=lights,
        calibrated_root=cal_out,
        master_dark_path=dp_ok,
        masterflat_by_filter={"V|10|2": fp, "V|20|2": fp, "V": fp},
        pipeline_config=cfg,
        master_dark_by_obs_key={
            "V|10|2": str(dp_ok),
            "V|20|2": str(dp_bad),
        },
    )
    assert stats.get("cal_diag_aborted_groups", 0) >= 1
    assert (cal_out / "good.fits").is_file()
    assert not (cal_out / "bad.fits").exists()
