"""Shared WCS header-key copy with a core-key integrity guard (EXCEPT-FIX-3 #8).

Copying a WCS header card-by-card into a destination FITS header and swallowing per-key
failures (``try: dst[k] = src[k] except Exception: pass``) can leave a *half-written* WCS
-- e.g. CRVAL/CRPIX present but a CD/PC term missing -- on a frame that is then treated as
"recovered". That silent partial write is worse than no write at all.

``copy_wcs_header_keys`` collects failures instead of swallowing them:

* If any **core celestial** key (CRVAL/CRPIX/CD/PC/CDELT/CTYPE/CUNIT/PV/LONPOLE/LATPOLE
  and the SIP coefficient families) cannot be copied, the destination is left **unmodified**
  (atomic) and the failed core keys are returned so the caller can ABORT -- increments the
  ``wcs_header_key_copy_fail`` counter and logs at ERROR.
* Non-core (cosmetic) key failures never block the copy: the good keys are written and the
  skipped cosmetic keys are logged at WARNING.

Census sites: EXC-0625 (platesolver sibling-WCS recovery) and EXC-0010 (astrometry_optimizer
SIP refit write).
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Structural / non-WCS keys that must never be copied (mirror the pre-existing skip-lists).
_STRUCTURAL_SKIP = {"", "COMMENT", "HISTORY", "SIMPLE", "BITPIX", "NAXIS", "EXTEND"}

# Core celestial keys whose loss corrupts the WCS.
_CORE_EXACT = {"LONPOLE", "LATPOLE", "A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER"}
_CORE_PREFIX = ("CRVAL", "CRPIX", "CDELT", "CTYPE", "CUNIT")
_CORE_RE = re.compile(
    r"^(CD\d+_\d+|PC\d+_\d+|PV\d+_\d+|A_\d+_\d+|B_\d+_\d+|AP_\d+_\d+|BP_\d+_\d+)$"
)


def _should_skip(key: str) -> bool:
    if key in _STRUCTURAL_SKIP:
        return True
    return key.startswith("NAXIS") and key != "NAXIS"


def is_core_wcs_key(key: str) -> bool:
    """True if ``key`` is a core celestial WCS key whose loss breaks the solution."""
    k = str(key).upper()
    if k in _CORE_EXACT:
        return True
    if any(k.startswith(p) for p in _CORE_PREFIX):
        return True
    return bool(_CORE_RE.match(k))


def copy_wcs_header_keys(dst_header: Any, src_header: Any, *, context: str) -> list[str]:
    """Copy WCS keys from ``src_header`` into ``dst_header``, guarding core-key integrity.

    Returns the list of **core** keys that could not be copied. An empty list means the
    copy succeeded (cosmetic-only failures are applied-around and logged at WARNING).

    On a non-empty return the destination is left UNCHANGED, so the caller can abort the
    recovery/refit for that frame without persisting a half-written WCS.
    """
    staged: list[tuple[str, Any]] = []
    failed_core: list[str] = []
    failed_other: list[str] = []

    for k in src_header:
        if _should_skip(k):
            continue
        try:
            # Stage into a throwaway header: a value that fails here would also fail on
            # the real destination, so we can detect failures without mutating dst.
            from astropy.io.fits import Header

            probe = Header()
            probe[k] = src_header[k]
            staged.append((k, src_header[k]))
        except Exception:  # noqa: BLE001 -- classify the failure, do not swallow it
            if is_core_wcs_key(k):
                failed_core.append(k)
            else:
                failed_other.append(k)

    if failed_core:
        from except_fix_counters import get_except_fix_counters

        get_except_fix_counters().wcs_header_key_copy_fail += 1
        logger.error(
            "[WCS-HDR-COPY] %s: core WCS keys uncopyable, aborting copy: %s",
            context,
            failed_core,
        )
        return failed_core

    for k, v in staged:
        dst_header[k] = v

    if failed_other:
        logger.warning(
            "[WCS-HDR-COPY] %s: non-core header keys skipped (cosmetic): %s",
            context,
            failed_other,
        )
    return []
