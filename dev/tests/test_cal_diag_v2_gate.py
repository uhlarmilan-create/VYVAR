"""CAL-DIAG v2 / INV-CAL-01 gate tests."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from astropy.io import fits

from cal_diag import (
    CalDiagGateResult,
    CalDiagSession,
    PedestalResult,
    apply_cal_diag_headers,
    cal_diag_gate_for_obs_group,
    dark_np_for_cal_diag,
    passthrough_cal_diag_headers,
    resolv_limit_adu,
    run_cal_diag_pregate,
    write_cal_diag_json,
)
from calibration import resample_master_to_light_binning
from pipeline import _cal_diag_session_from_export, _match_and_crop_pair


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


def test_gate_matched_sum_derived(tmp_path: Path):
    light = np.full((2, 2), 120.0, dtype=np.float32)
    dark_master = np.full((4, 4), 10.0, dtype=np.float32)
    lp = tmp_path / "light.fits"
    dp = tmp_path / "dark.fits"
    _write_fits(lp, light, XBINNING=2, YBINNING=2, EXPTIME=30.0, FILTER="V")
    _write_fits(dp, dark_master, XBINNING=1, YBINNING=1, EXPTIME=30.0)

    res = cal_diag_gate_for_obs_group(
        repr_light_path=lp,
        dark_path=dp,
        obs_group_key="V|30|2",
        light_binning=2,
        master_binning=1,
        pedestal_dark_paths=[dp],
        match_and_crop_pair=_match,
        saturation_adu=None,
    )
    assert res.aborted is False
    assert res.convention == "SUM"
    assert res.convention_src == "DERIVED"
    assert res.status == "PASS"


def test_gate_mean_driver_derived(tmp_path: Path):
    light = np.full((2, 2), 25.0, dtype=np.float32)
    dark_master = np.full((4, 4), 10.0, dtype=np.float32)
    lp = tmp_path / "light.fits"
    dp = tmp_path / "dark.fits"
    _write_fits(lp, light, XBINNING=2, YBINNING=2)
    _write_fits(dp, dark_master, XBINNING=1, YBINNING=1)

    res = cal_diag_gate_for_obs_group(
        repr_light_path=lp,
        dark_path=dp,
        obs_group_key="NoFilter|0|2",
        light_binning=2,
        master_binning=1,
        pedestal_dark_paths=[dp],
        match_and_crop_pair=_match,
        saturation_adu=None,
    )
    assert res.aborted is False
    assert res.convention == "MEAN"
    assert res.convention_src == "DERIVED"
    assert res.ratio_r == pytest.approx(2.5, rel=0.01)


def test_gate_ccd_linear_inconsistent_aborts(tmp_path: Path):
    # SUM-derived convention but SUM sky fails while MEAN counterfactual passes.
    light = np.full((2, 2), 120.0, dtype=np.float32)
    dark_master = np.full((4, 4), 10.0, dtype=np.float32)
    lp = tmp_path / "light.fits"
    dp = tmp_path / "dark.fits"
    _write_fits(lp, light, XBINNING=2, YBINNING=2)
    _write_fits(dp, dark_master, XBINNING=1, YBINNING=1)

    calls = {"n": 0}

    def _fake_check_b(*, diff, hard_sigma, sat_frac, saturation_adu, obs_group_key, messages, status):
        _ = (diff, hard_sigma, sat_frac, saturation_adu, obs_group_key, messages, status)
        calls["n"] += 1
        med = float(np.nanmedian(diff))
        if calls["n"] == 1:
            return "ABORT", med, 1.0
        return "PASS", med, 1.0

    with patch("cal_diag._check_b_sky", side_effect=_fake_check_b):
        with patch(
            "cal_diag.derive_pedestal_from_masters",
            return_value=PedestalResult(
                p_adu=10.0,
                sigma_p=0.01,
                k_adu_per_s=0.0,
                k_status="NEGLIGIBLE",
                method="INTERCEPT",
                n_exptimes=2,
                pedestal_measurable=True,
                check_p_consistent=True,
            ),
        ):
            res = cal_diag_gate_for_obs_group(
                repr_light_path=lp,
                dark_path=dp,
                obs_group_key="NoFilter|0|2",
                light_binning=2,
                master_binning=1,
                pedestal_dark_paths=[dp],
                match_and_crop_pair=_match,
                saturation_adu=None,
            )
    assert res.aborted is True
    assert res.abort_reason == "CCD_LINEAR_INCONSISTENT"


def test_gate_indeterminate_negligible(tmp_path: Path):
    light = np.full((2, 2), 120.0, dtype=np.float32)
    dark_master = np.full((4, 4), 10.0, dtype=np.float32)
    lp = tmp_path / "light.fits"
    dp = tmp_path / "dark.fits"
    _write_fits(lp, light, XBINNING=2, YBINNING=2)
    _write_fits(dp, dark_master, XBINNING=1, YBINNING=1)

    huge_sigma = resolv_limit_adu(sigma_p=500.0, block_factor=2) + 1.0
    assert 30.0 < huge_sigma

    with patch(
        "cal_diag.derive_pedestal_from_masters",
        return_value=PedestalResult(
            p_adu=0.1,
            sigma_p=500.0,
            k_adu_per_s=0.0,
            k_status="NEGLIGIBLE",
            method="INTERCEPT",
            n_exptimes=2,
            pedestal_measurable=True,
            check_p_consistent=True,
        ),
    ):
        res = cal_diag_gate_for_obs_group(
            repr_light_path=lp,
            dark_path=dp,
            obs_group_key="NoFilter|0|2",
            light_binning=2,
            master_binning=1,
            pedestal_dark_paths=[dp],
            match_and_crop_pair=_match,
            saturation_adu=None,
        )
    assert res.convention_src == "INDETERMINATE_NEGLIGIBLE"
    assert res.convention == "SUM"
    assert res.status == "WARN"


def test_gate_indeterminate_unmeasured(tmp_path: Path):
    light = np.full((2, 2), 120.0, dtype=np.float32)
    dark_master = np.full((4, 4), 10.0, dtype=np.float32)
    lp = tmp_path / "light.fits"
    dp = tmp_path / "dark.fits"
    _write_fits(lp, light, XBINNING=2, YBINNING=2)
    _write_fits(dp, dark_master, XBINNING=1, YBINNING=1)

    with patch(
        "cal_diag.derive_pedestal_from_masters",
        return_value=PedestalResult(
            p_adu=10.0,
            sigma_p=0.01,
            k_adu_per_s=0.0,
            k_status="UNKNOWN",
            method="SINGLE_MASTER_MEDIAN",
            n_exptimes=1,
            pedestal_measurable=False,
            check_p_consistent=False,
        ),
    ):
        res = cal_diag_gate_for_obs_group(
            repr_light_path=lp,
            dark_path=dp,
            obs_group_key="NoFilter|0|2",
            light_binning=2,
            master_binning=1,
            pedestal_dark_paths=[dp],
            match_and_crop_pair=_match,
            saturation_adu=None,
        )
    assert res.convention_src == "INDETERMINATE_UNMEASURED"
    assert res.convention == "SUM"
    assert res.status == "WARN"


def test_passthrough_headers():
    hdr = fits.Header()
    passthrough_cal_diag_headers(hdr)
    assert hdr["VY_DKRSMP"] == "PASSTHROUGH"
    assert hdr["VY_DKRSMP_SRC"] == "PASSTHROUGH"


def test_apply_headers_includes_src():
    hdr = fits.Header()
    gr = CalDiagGateResult(
        obs_group_key="V|30|2",
        dark_path="/x/dark.fits",
        light_binning=2,
        status="PASS",
        convention="SUM",
        convention_src="DERIVED",
        sky_median=100.0,
        pedestal_p=24.5,
    )
    apply_cal_diag_headers(hdr, gr)
    assert hdr["VY_DKRSMP"] == "SUM"
    assert hdr["VY_DKRSMP_SRC"] == "DERIVED"
    assert float(hdr["VY_CPED"]) == pytest.approx(24.5)


def test_session_export_roundtrip():
    session = CalDiagSession()
    gr = CalDiagGateResult(
        obs_group_key="k",
        dark_path="/d.fits",
        light_binning=2,
        status="PASS",
        convention="SUM",
        convention_src="DERIVED",
    )
    session.gate_results["k|/d.fits|b2"] = gr
    blob = session.json_export()
    back = _cal_diag_session_from_export(blob)
    assert "k|/d.fits|b2" in back.gate_results
    assert back.gate_results["k|/d.fits|b2"].convention == "SUM"


def test_write_cal_diag_json(tmp_path: Path):
    session = CalDiagSession()
    session.gate_results["a"] = CalDiagGateResult(
        obs_group_key="a",
        dark_path=str(tmp_path / "d.fits"),
        light_binning=2,
        status="PASS",
        convention="SUM",
        convention_src="DERIVED",
    )
    out = write_cal_diag_json(tmp_path, session)
    assert out is not None and out.is_file()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data.get("spec_version") == "CAL-DIAG-v2"


def test_dark_cache_same_convention(tmp_path: Path):
    dark_master = np.full((4, 4), 10.0, dtype=np.float32)
    dp = tmp_path / "dark.fits"
    _write_fits(dp, dark_master, XBINNING=1, YBINNING=1)
    session = CalDiagSession()
    gr = CalDiagGateResult(
        obs_group_key="k",
        dark_path=str(dp),
        light_binning=2,
        status="PASS",
        convention="SUM",
        convention_src="DERIVED",
    )
    a = dark_np_for_cal_diag(
        session,
        master_binning=1,
        dark_path=dp,
        light_binning=2,
        light_shape=(2, 2),
        light_filename="l.fits",
        gate_result=gr,
    )
    b = dark_np_for_cal_diag(
        session,
        master_binning=1,
        dark_path=dp,
        light_binning=2,
        light_shape=(2, 2),
        light_filename="l.fits",
        gate_result=gr,
    )
    assert a is b
    assert float(np.nanmedian(a)) == pytest.approx(40.0)


def test_apply_calibrated_stage_skysf_matches_archive(tmp_path: Path):
    """P2 stage helper: pure cal + sky order matches SKYSF archive."""
    from cal_diag import apply_calibrated_stage_for_compare, calibrated_stage_from_header
    from pipeline import _fit_subtract_preprocess_sky_surface

    pure = np.full((8, 8), 100.0, dtype=np.float32)
    pure[2:5, 2:5] += 50.0
    staged, _ = _fit_subtract_preprocess_sky_surface(pure.copy(), order=2)
    hdr = fits.Header()
    hdr["VY_SKYSF"] = True
    hdr["VYSKYORD"] = 2
    replay = apply_calibrated_stage_for_compare(pure, hdr)
    assert calibrated_stage_from_header(hdr)[0] == "SKYSF_2"
    assert np.array_equal(replay, staged)
