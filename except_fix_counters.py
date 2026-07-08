"""Run-level counters for EXCEPT-FIX terminal failure surfacing."""

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
    phase2a_empty_comp_drop: int = 0
    catalog_bpm_enhance_fail: int = 0
    plate_solve_bundle_fail: int = 0
    masterstars_rescale_coords_fail: int = 0
    vytarg_header_write_fail: int = 0
    vsx_frame_bbox_wcs_fail: int = 0
    gaia_cone_optics_floor_fail: int = 0
    vsx_variable_coord_drop: int = 0
    stress_sidecar_skip: int = 0
    masterstar_ref_swap_fail: int = 0
    calibrate_db_sync_fail: int = 0
    # EXCEPT-FIX-3 (tranche 3: importer / platesolver / alignment / astrometry_optimizer)
    importer_filter_read_fail: int = 0
    dark_bpm_sidecar_write_fail: int = 0
    calib_scope_conflict_check_fail: int = 0
    calib_library_register_fail: int = 0
    importer_capture_date_fallback: int = 0
    importer_imagetyp_read_fail: int = 0
    importer_obs_group_meta_skip: int = 0
    wcs_header_key_copy_fail: int = 0
    align_unique_sample_fail: int = 0
    platesolve_match_rate_meta_fail: int = 0

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
            "phase2a_empty_comp_drop": self.phase2a_empty_comp_drop,
            "catalog_bpm_enhance_fail": self.catalog_bpm_enhance_fail,
            "plate_solve_bundle_fail": self.plate_solve_bundle_fail,
            "masterstars_rescale_coords_fail": self.masterstars_rescale_coords_fail,
            "vytarg_header_write_fail": self.vytarg_header_write_fail,
            "vsx_frame_bbox_wcs_fail": self.vsx_frame_bbox_wcs_fail,
            "gaia_cone_optics_floor_fail": self.gaia_cone_optics_floor_fail,
            "vsx_variable_coord_drop": self.vsx_variable_coord_drop,
            "stress_sidecar_skip": self.stress_sidecar_skip,
            "masterstar_ref_swap_fail": self.masterstar_ref_swap_fail,
            "calibrate_db_sync_fail": self.calibrate_db_sync_fail,
            "importer_filter_read_fail": self.importer_filter_read_fail,
            "dark_bpm_sidecar_write_fail": self.dark_bpm_sidecar_write_fail,
            "calib_scope_conflict_check_fail": self.calib_scope_conflict_check_fail,
            "calib_library_register_fail": self.calib_library_register_fail,
            "importer_capture_date_fallback": self.importer_capture_date_fallback,
            "importer_imagetyp_read_fail": self.importer_imagetyp_read_fail,
            "importer_obs_group_meta_skip": self.importer_obs_group_meta_skip,
            "wcs_header_key_copy_fail": self.wcs_header_key_copy_fail,
            "align_unique_sample_fail": self.align_unique_sample_fail,
            "platesolve_match_rate_meta_fail": self.platesolve_match_rate_meta_fail,
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
