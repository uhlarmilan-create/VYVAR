"""Run-level counters for EXCEPT-FIX-1 terminal failure surfacing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ExceptFixCounters:
    sky_annulus_mask_fail: int = 0
    sky_annulus_invalid: int = 0
    phase2a_csv_cache_skip: int = 0
    comp_pool_csv_skip: int = 0
    phase1_csv_cache_skip: int = 0
    comp_frame_accumulation_skip: int = 0
    comp_detrend_skip: int = 0
    vt_wcs_refresh_fail: int = 0
    psf_epsf_sky_inject_fail: int = 0
    psf_local_sky_fail: int = 0
    psf_grouped_fit_fail: int = 0

    def snapshot(self) -> dict[str, int]:
        return {
            "sky_annulus_mask_fail": self.sky_annulus_mask_fail,
            "sky_annulus_invalid": self.sky_annulus_invalid,
            "phase2a_csv_cache_skip": self.phase2a_csv_cache_skip,
            "comp_pool_csv_skip": self.comp_pool_csv_skip,
            "phase1_csv_cache_skip": self.phase1_csv_cache_skip,
            "comp_frame_accumulation_skip": self.comp_frame_accumulation_skip,
            "comp_detrend_skip": self.comp_detrend_skip,
            "vt_wcs_refresh_fail": self.vt_wcs_refresh_fail,
            "psf_epsf_sky_inject_fail": self.psf_epsf_sky_inject_fail,
            "psf_local_sky_fail": self.psf_local_sky_fail,
            "psf_grouped_fit_fail": self.psf_grouped_fit_fail,
        }


_COUNTERS = ExceptFixCounters()


def reset_except_fix_counters() -> None:
    global _COUNTERS
    _COUNTERS = ExceptFixCounters()


def get_except_fix_counters() -> ExceptFixCounters:
    return _COUNTERS


def merge_except_fix_summary(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"except_fix_counters": _COUNTERS.snapshot()}
    if extra:
        out.update(extra)
    return out
