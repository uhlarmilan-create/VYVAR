"""INV-CAL-02 calibrated stage stamp, resolve, verify."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from cal_diag import CalStageCompareRefusedError, apply_calibrated_stage_for_compare, calibrated_compare_refused
from cal_stage import (
    CalStageConfidence,
    compute_fits_datasum,
    compute_skysf_apply_stage,
    resolve_calibrated_stage,
    skysf_stage_token,
    stamp_cal_stage_headers,
    verify_fits_datasum,
)
from config import AppConfig
from pipeline import _calibrate_one_light_disk, _qc_enrich_one_frame


def _write_light(path: Path, data: np.ndarray, **hdr_kw: object) -> None:
    hdr = fits.Header()
    for k, v in hdr_kw.items():
        hdr[k] = v
    fits.writeto(path, np.asarray(data, dtype=np.float32), header=hdr, overwrite=True)


def test_stamp_datasum_verifies(tmp_path: Path) -> None:
    data = np.full((4, 4), 100.0, dtype=np.float32)
    hdr = fits.Header()
    ds = stamp_cal_stage_headers(hdr, data, stage="PURE")
    assert hdr["VY_CALSTAGE"] == "PURE"
    assert verify_fits_datasum(data, ds)


def test_resolve_legacy_pure_evidence() -> None:
    hdr = fits.Header()
    hdr["VY_QCBG"] = (2414.0, "")
    hdr["VY_DKRSMP"] = ("SUM", "")
    res = resolve_calibrated_stage(hdr)
    assert res.stage == "PURE"
    assert res.confidence == CalStageConfidence.LEGACY_INFERRED


def test_resolve_legacy_skysf() -> None:
    hdr = fits.Header()
    hdr["VY_SKYSF"] = True
    hdr["VYSKYORD"] = 2
    res = resolve_calibrated_stage(hdr)
    assert res.stage == "SKYSF_2"
    assert res.confidence == CalStageConfidence.LEGACY_INFERRED


def test_resolve_indeterminate_vyskyp2p_without_skysf() -> None:
    hdr = fits.Header()
    hdr["VYSKYP2P"] = 140.0
    res = resolve_calibrated_stage(hdr)
    assert res.confidence == CalStageConfidence.INDETERMINATE_LEGACY
    assert calibrated_compare_refused(hdr) is not None


def test_force_reapply_stage_token_distinguishable() -> None:
    hdr = fits.Header()
    hdr["VY_SKYSF"] = True
    hdr["VYSKYORD"] = 2
    hdr["VY_CALSTAGE"] = ("SKYSF_2", "")
    hdr["VY_SKYPASS"] = (1, "")
    token, pass_n = compute_skysf_apply_stage(hdr, sky_order=2, force_reapply=True)
    assert token == "SKYSF_2_R2"
    assert pass_n == 2
    assert token != skysf_stage_token(order=2, pass_n=1)


def test_qc_enrich_stamps_stage_on_apply(tmp_path: Path) -> None:
    fp = tmp_path / "light.fits"
    yy, xx = np.mgrid[0:32, 0:32]
    data = (1200.0 + 0.1 * xx + 0.05 * yy).astype(np.float32)
    _write_light(fp, data, VY_QCBG=1200.0, VY_DKRSMP="SUM")
    row = _qc_enrich_one_frame(
        str(fp),
        sky_order=2,
        force_reapply=False,
        prefilter_status=None,
        target_ra=None,
        target_dec=None,
        inject_pointing_only_if_missing=True,
    )
    assert row.get("cal_stage") == "SKYSF_2"
    with fits.open(fp) as hdul:
        hdr = hdul[0].header
        assert hdr["VY_CALSTAGE"] == "SKYSF_2"
        assert verify_fits_datasum(hdul[0].data, hdr["VY_CALDATASUM"])


def test_force_reapply_writes_r2_stage(tmp_path: Path) -> None:
    fp = tmp_path / "light.fits"
    yy, xx = np.mgrid[0:32, 0:32]
    data = (1200.0 + 0.1 * xx + 0.05 * yy).astype(np.float32)
    _write_light(fp, data, VY_QCBG=1200.0, VY_DKRSMP="SUM")
    _qc_enrich_one_frame(
        str(fp),
        sky_order=2,
        force_reapply=False,
        prefilter_status=None,
        target_ra=None,
        target_dec=None,
        inject_pointing_only_if_missing=True,
    )
    row2 = _qc_enrich_one_frame(
        str(fp),
        sky_order=2,
        force_reapply=True,
        prefilter_status=None,
        target_ra=None,
        target_dec=None,
        inject_pointing_only_if_missing=True,
    )
    assert row2.get("cal_stage") == "SKYSF_2_R2"
    with fits.open(fp) as hdul:
        assert hdul[0].header["VY_CALSTAGE"] == "SKYSF_2_R2"
        assert int(hdul[0].header["VY_SKYPASS"]) == 2


def test_calibrate_stamps_pure(tmp_path: Path) -> None:
    """P8: stamping must not alter calibration arithmetic (only headers after compute)."""
    lib = Path(__file__).resolve().parents[2] / "CalibrationLibrary"
    md = sorted(lib.glob("Dark_60s*Bin1*.fits"))[0]
    mf = sorted(lib.glob("Flat*.fits"))[0]
    raw = Path(__file__).resolve().parents[2] / "Archive/Drafts/draft_000435/Raw/lights/NoFilter_60_2/BO_CVn_Light_001.fits"
    if not raw.is_file():
        pytest.skip("anchor raw frame missing")
    dst = tmp_path / "cal.fits"
    _calibrate_one_light_disk(
        src=raw,
        dst=dst,
        master_dark_path=md,
        masterflat_by_filter={"NoFilter": mf},
        qc_pack={"enabled": False, "draft_id": 435},
    )
    with fits.open(dst) as hdul:
        hdr = hdul[0].header
        assert hdr["VY_CALSTAGE"] == "PURE"
        assert verify_fits_datasum(hdul[0].data, hdr["VY_CALDATASUM"])


def test_compare_refuses_indeterminate() -> None:
    hdr = fits.Header()
    hdr["VYSKYP2P"] = 150.0
    with pytest.raises(CalStageCompareRefusedError):
        apply_calibrated_stage_for_compare(np.zeros((4, 4), dtype=np.float32), hdr)
