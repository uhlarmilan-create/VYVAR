"""INV-CAL-02: calibrated product stage stamp, resolve, and verify."""

from __future__ import annotations

import logging
import math

import numpy as np
from stats_core import parse_bool
from astropy.io import fits

LOGGER = logging.getLogger(__name__)


def compute_fits_datasum(data: np.ndarray) -> str:
    """Registered FITS DATASUM string for a float32 image array."""
    from astropy.io.fits import PrimaryHDU

    arr = np.asarray(data, dtype=np.float32)
    hdu = PrimaryHDU(data=arr)
    hdu.add_datasum()
    return str(hdu.header["DATASUM"]).strip()


def verify_fits_datasum(data: np.ndarray, expected: str | None) -> bool:
    if expected is None or str(expected).strip() == "":
        return False
    try:
        return compute_fits_datasum(data) == str(expected).strip()
    except Exception:  # noqa: BLE001
        return False


def _header_bool(hdr: fits.Header, key: str) -> bool | None:
    if key not in hdr:
        return None
    v = hdr.get(key)
    return parse_bool(v)


def _header_has_vy_skysf(hdr: fits.Header) -> bool:
    v = _header_bool(hdr, "VY_SKYSF")
    return bool(v)


def compute_skysf_apply_stage(*, sky_order: int) -> tuple[str, int]:
    """Stage token and pass count when applying the sky surface (always the first pass; G01-31:
    the ``_R{n}`` re-pass tokens had no producer)."""
    return f"SKYSF_{int(sky_order)}", 1


def stamp_cal_stage_headers(
    hdr: fits.Header,
    data: np.ndarray,
    *,
    stage: str,
    pstbg: float | None = None,
    skypass: int | None = None,
) -> str:
    """Write ``VYCALSTG`` + ``VYCALSUM`` (+ optional post-sky QC). Returns datasum."""
    datasum = compute_fits_datasum(data)
    hdr["VYCALSTG"] = (str(stage), "INV-CAL-02 calibrated product stage")
    hdr["VYCALSUM"] = (datasum, "FITS DATASUM of primary data array")
    if pstbg is not None and math.isfinite(float(pstbg)):
        hdr["VY_PSTBG"] = (float(pstbg), "Post-stage plain sky median [ADU]")
    if skypass is not None and int(skypass) > 0:
        hdr["VYSKYPAS"] = (int(skypass), "Sky-surface subtract pass count on these pixels")
    return datasum


