"""Run-level counters for EXCEPT-FIX terminal failure surfacing.

G12-30 (Milan 2026-10-31): the counters are reset once per run (``run_night_pipeline``), the
counts of spawn workers are added to the run process (``parallel_frames.map_frames``), and only
counters that some code increments exist (N3: 15 counters without an increment site removed)."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class ExceptFixCounters:
    sky_annulus_mask_fail: int = 0
    sky_annulus_invalid: int = 0
    aperture_csv_cache_skip: int = 0
    vt_wcs_refresh_fail: int = 0
    aperture_empty_comp_drop: int = 0
    catalog_aperture_enhance_fail: int = 0
    plate_solve_bundle_fail: int = 0
    vsx_frame_bbox_wcs_fail: int = 0
    calibrate_db_sync_fail: int = 0
    # EXCEPT-FIX-3 (tranche 3: importer / platesolver / alignment / astrometry_optimizer)
    importer_filter_read_fail: int = 0
    importer_obs_group_meta_skip: int = 0
    align_unique_sample_fail: int = 0
    # EXCEPT-FIX-4 (tranche 4: time/trust/export/optics/check-star)
    timeobs_parse_fallback: int = 0
    jd_mid_compute_fail: int = 0
    date_obs_parse_fail: int = 0
    hjd_bjd_compute_fail: int = 0
    trust_kmag_sidecar_read_fail: int = 0
    k2_airmass_read_fail: int = 0
    export_observer_location_read_fail: int = 0
    optics_draft_override_read_fail: int = 0
    check_star_ensemble_filter_skip: int = 0

    def snapshot(self) -> dict[str, int]:
        """Every counter by name (G12-48: ``dataclasses.asdict``, no hand-written list)."""
        return asdict(self)


_COUNTERS = ExceptFixCounters()


def reset_except_fix_counters() -> None:
    global _COUNTERS
    _COUNTERS = ExceptFixCounters()


def get_except_fix_counters() -> ExceptFixCounters:
    return _COUNTERS


def counter_deltas(before: dict[str, int]) -> dict[str, int]:
    """Counts added since ``before`` (a ``snapshot``); only the non-zero ones."""
    now = _COUNTERS.snapshot()
    return {k: int(v) - int(before.get(k, 0)) for k, v in now.items() if int(v) != int(before.get(k, 0))}


def add_except_fix_counts(deltas: dict[str, int] | None) -> None:
    """Add the counts a worker process collected (``counter_deltas``) to this process's counters."""
    for k, v in (deltas or {}).items():
        setattr(_COUNTERS, k, int(getattr(_COUNTERS, k)) + int(v))
