"""INV-CAL-02 calibrated stage stamp, resolve, verify."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits

from cal_stage import stamp_cal_stage_headers, verify_fits_datasum
from pipeline import _qc_enrich_one_frame


def _write_light(path: Path, data: np.ndarray, **hdr_kw: object) -> None:
    hdr = fits.Header()
    for k, v in hdr_kw.items():
        hdr[k] = v
    fits.writeto(path, np.asarray(data, dtype=np.float32), header=hdr, overwrite=True)


def test_stamp_datasum_verifies(tmp_path: Path) -> None:
    data = np.full((4, 4), 100.0, dtype=np.float32)
    hdr = fits.Header()
    ds = stamp_cal_stage_headers(hdr, data, stage="PURE")
    assert hdr["VYCALSTG"] == "PURE"
    assert verify_fits_datasum(data, ds)


def test_qc_enrich_stamps_stage_on_apply(tmp_path: Path) -> None:
    fp = tmp_path / "light.fits"
    yy, xx = np.mgrid[0:32, 0:32]
    data = (1200.0 + 0.1 * xx + 0.05 * yy).astype(np.float32)
    _write_light(fp, data, VY_QCBG=1200.0, VYDKRSMP="SUM")
    row = _qc_enrich_one_frame(
        str(fp),
        sky_order=2,
        prefilter_status=None,
        target_ra=None,
        target_dec=None,
        inject_pointing_only_if_missing=True,
        detection_sigma=4.5,
    )
    assert row.get("cal_stage") == "SKYSF_2"
    with fits.open(fp) as hdul:
        hdr = hdul[0].header
        assert hdr["VYCALSTG"] == "SKYSF_2"
        assert verify_fits_datasum(hdul[0].data, hdr["VYCALSUM"])

