"""INV-CAL-02: calibrated product stage stamp, resolve, and verify."""

from __future__ import annotations

import logging
import math
import re

import numpy as np
from astropy.io import fits

LOGGER = logging.getLogger(__name__)


_STAGE_SKYSF_RE = re.compile(r"^SKYSF_(\d+)(?:_R(\d+))?$")


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


def parse_cal_stage_token(stage: str) -> tuple[int | None, int]:
    """Return ``(sky_order, sky_pass)`` for stage tokens."""
    token = str(stage or "").strip().upper()
    if token in ("PURE", "PASSTHROUGH", ""):
        return None, 1
    m = _STAGE_SKYSF_RE.match(token)
    if not m:
        return None, 1
    order = int(m.group(1))
    pass_n = int(m.group(2)) if m.group(2) else 1
    return order, max(1, pass_n)


def skysf_stage_token(*, order: int, pass_n: int = 1) -> str:
    order_i = int(order)
    pass_i = max(1, int(pass_n))
    if pass_i <= 1:
        return f"SKYSF_{order_i}"
    return f"SKYSF_{order_i}_R{pass_i}"


def _header_bool(hdr: fits.Header, key: str) -> bool | None:
    if key not in hdr:
        return None
    v = hdr.get(key)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in ("true", "1", "t", "yes"):
        return True
    if s in ("false", "0", "f", "no"):
        return False
    return None


def _header_has_vy_skysf(hdr: fits.Header) -> bool:
    v = _header_bool(hdr, "VY_SKYSF")
    return bool(v)


def _prior_skypass(hdr: fits.Header) -> int:
    if "VY_SKYPASS" in hdr:
        try:
            return max(1, int(hdr["VY_SKYPASS"]))
        except (TypeError, ValueError):
            pass
    stage = str(hdr.get("VY_CALSTAGE") or "").strip().upper()
    if stage:
        _, pass_n = parse_cal_stage_token(stage)
        return pass_n
    if _header_has_vy_skysf(hdr):
        return 1
    return 0


def compute_skysf_apply_stage(
    hdr: fits.Header,
    *,
    sky_order: int,
    force_reapply: bool,
) -> tuple[str, int]:
    """Stage token and pass count when applying (or re-applying) sky surface."""
    prior_pass = _prior_skypass(hdr)
    if force_reapply and prior_pass >= 1:
        pass_n = prior_pass + 1
    else:
        pass_n = 1
    return skysf_stage_token(order=int(sky_order), pass_n=pass_n), pass_n


def stamp_cal_stage_headers(
    hdr: fits.Header,
    data: np.ndarray,
    *,
    stage: str,
    pstbg: float | None = None,
    skypass: int | None = None,
) -> str:
    """Write ``VY_CALSTAGE`` + ``VY_CALDATASUM`` (+ optional post-sky QC). Returns datasum."""
    datasum = compute_fits_datasum(data)
    hdr["VY_CALSTAGE"] = (str(stage), "INV-CAL-02 calibrated product stage")
    hdr["VY_CALDATASUM"] = (datasum, "FITS DATASUM of primary data array")
    if pstbg is not None and math.isfinite(float(pstbg)):
        hdr["VY_PSTBG"] = (float(pstbg), "Post-stage sigma-clipped sky median [ADU]")
    if skypass is not None and int(skypass) > 0:
        hdr["VY_SKYPASS"] = (int(skypass), "Sky-surface subtract pass count on these pixels")
    return datasum


