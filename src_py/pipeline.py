"""Core processing pipeline for FITS observations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any


from config import AppConfig
from database import (
    VyvarDatabase,
)


from fits_meta import (
    log_lights_binning_from_headers_preflight,
    generate_observation_hash,
    fits_metadata_from_primary_header,
    extract_fits_metadata,
)


LOGGER = logging.getLogger(__name__)

from pipeline_constants import N_STARS_DIAG_K, SAT_LIMIT_CONTAINER_CLIP_ADU, SAT_LIMIT_NO_KNEE_FRAC




from cal_stage import _header_has_vy_skysf  # noqa: E402,F401  # re-export; survivor in cal_stage.py


class AstroPipeline:
    """Skeleton for the modular variable-star processing workflow."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.db = VyvarDatabase.for_config(self.config)


    def quick_calibrate_last_import(
        self,
        *,
        archive_path: Path | str,
        master_dark_path: Path | None,
        masterflat_by_filter: dict[str, Path | None],
        progress_cb: "callable | None" = None,
        equipment_id: int | None = None,
        draft_id: int | None = None,
        observation_id: str | None = None,
        masterflat_by_obs_key: dict[str, str | Path | None] | None = None,
        master_dark_by_obs_key: dict[str, str | Path | None] | None = None,
        flat_substitute_by_obs_key: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Calibrate imported draft/raw lights into `calibrated/` under archive_path.

        ``flat_substitute_by_obs_key``: groups whose FLAT is a flagged substitute from another night
        (plan, decision A1). The calibration really done per group is recorded in the draft manifest
        (``calibration_groups``; ``calibration_mode`` PASSTHROUGH when no frame got a dark).
        """
        ap = Path(archive_path)
        # Accept both draft root (.../draft_xxx) and direct non_calibrated path (.../draft_xxx/non_calibrated).
        ap_root = ap.parent if ap.name.casefold() == "non_calibrated" else ap
        LOGGER.info("Archive calibration: %s", ap)
        # Support Draft structure: <archive>/Raw/lights and optional <archive>/non_calibrated/lights.
        # Both sources are calibrated/passthrough-written into the single target: <archive>/calibrated/lights.
        outputs: dict[str, Any] = {"archive_path": str(ap_root), "results": {}, "perf10_qsum": {}}
        merged_perf10: dict[str, dict[str, Any]] = {}

        # G01-01 (decision A1): one lights tree per draft - Raw/lights (VYVAR-calibrated import) or
        # non_calibrated/lights (quick look). A draft imported before the fix may have both (the groups
        # without a flat were split off); every existing tree is calibrated, so no imported light is lost.
        trees = [("non_calibrated", ap_root / "non_calibrated" / "lights"), ("Raw", ap_root / "Raw" / "lights")]
        out_root = ap_root / "calibrated" / "lights"
        flags_by_group: dict[str, dict[str, int]] = {}
        for section, lights_tree in trees:
            if not lights_tree.exists():
                continue
            sec = calibrate_lights_to_calibrated(
                lights_root=lights_tree,
                calibrated_root=out_root,
                master_dark_path=master_dark_path,
                masterflat_by_filter=masterflat_by_filter,
                progress_cb=progress_cb,
                pipeline_config=self.config,
                equipment_id=equipment_id,
                draft_id=draft_id,
                observation_id=observation_id,
                masterflat_by_obs_key=masterflat_by_obs_key,
                master_dark_by_obs_key=master_dark_by_obs_key,
                flat_substitute_by_obs_key=flat_substitute_by_obs_key,
            )
            outputs["results"][section] = sec
            for grp, counts in (sec.get("flags_by_group") or {}).items():
                tgt = flags_by_group.setdefault(str(grp), {})
                for flag, n in counts.items():
                    tgt[str(flag)] = int(tgt.get(str(flag), 0)) + int(n)
        if draft_id is not None and flags_by_group:
            from draft_provenance import record_calibration_outcome  # noqa: PLC0415
            from fits_meta import observation_group_folder_name  # noqa: PLC0415
            from infolog import log_event  # noqa: PLC0415

            outputs["calibration_groups"] = record_calibration_outcome(
                ap_root,
                draft_id=int(draft_id),
                flags_by_group=flags_by_group,
                flat_substitute_by_group={
                    observation_group_folder_name(str(k)): str(v)
                    for k, v in (flat_substitute_by_obs_key or {}).items() if v
                },
            )
            for grp, info in outputs["calibration_groups"].items():
                if info.get("relative_only"):
                    LOGGER.warning("Calibration WARNING [%s]: %s", grp, info.get("reason"))
                    log_event(f"CALIBRATION WARNING [{grp}]: {info.get('reason')} - "
                              "light curves not for AAVSO/VarAstro submission")
                if info.get("flat_substitute"):
                    log_event(f"CALIBRATION WARNING [{grp}]: FLAT from another night "
                              f"({info.get('flat_substitute')}) - flagged substitute")

        for _sec_stats in (outputs.get("results") or {}).values():
            if isinstance(_sec_stats, dict):
                _p10 = _sec_stats.get("perf10_qc_results")
                if isinstance(_p10, dict):
                    merged_perf10.update(_p10)

        if merged_perf10 and draft_id is not None:
            outputs["perf10_qsum"] = apply_perf10_dao_qc_to_obs_files(
                db=self.db,
                draft_id=int(draft_id),
                archive_path=ap_root,
                perf10_qc_results=merged_perf10,
            )

        LOGGER.info("Calibration finished (output sections: %s)", list((outputs.get("results") or {}).keys()))
        if equipment_id is not None:
            cal_root = ap_root / "calibrated" / "lights"
            if cal_root.is_dir():
                osc_out = run_osc_channel_extraction_for_archive(
                    calibrated_lights_root=cal_root,
                    db=self.db,
                    equipment_id=int(equipment_id),
                    app_config=self.config,
                    progress_cb=progress_cb,
                )
                outputs["osc_extraction"] = osc_out
        return outputs


from pipeline_calibrate import (
    SkySurfaceOrderConflictError,
    _cal_diag_session_from_export,
    _calibrate_one_light_disk,
    _decide_preprocess_sky_action,
    _effective_saturation_limit,
    _fit_subtract_preprocess_sky_surface,
    _infer_sat_limit_from_bitpix,
    _iter_light_fits,
    _match_and_crop_pair,
    _qc_enrich_calibrated_in_place,
    _qc_enrich_one_frame,
    _saturate_limit_adu_from_header,
    _sync_obs_calibration_state_with_retry,
    apply_perf10_dao_qc_to_obs_files,
    calibrate_lights_to_calibrated,
    run_draft_ram_calibration_qc_to_obs_files,
    run_osc_channel_extraction_for_archive,
    scan_calibrated_lights_pointing,
)
from utils import norm_fits_path_key
from pipeline_preprocess import (
    _archive_raw_to_calibrated_light,
    build_prefilter_rejected_map,
    calibrated_paths_for_draft_apply_filters,
    filter_files_by_qc_metrics_allowlist,
    load_qc_metrics_status_by_path,
    qc_enrich_calibrated_lights_in_place,
    resolve_obs_file_to_processed_fits,
    resolve_preprocess_target_coordinates,
)


from pipeline_astrometry import (
    _merge_vsx_exoplanet_variable_targets,
    _plate_solve_input_bundle,
    _query_vsx_local_frame_bbox,
    draft_is_multi_group_obs,
    draft_obs_group_count,
    get_masterstar_candidates,
    select_comparison_stars_spatial_grid,
    write_photometry_plan_files,
)
from pipeline_catalog import _dao_targeted_pass2_unmatched_gaia
from astrometry_orchestration import astrometry_align_and_build_masterstar
from pipeline_preprocess import (
    _path_segments_forbidden_for_masterstar_physical_source,
    _resolve_best_effort_path_under,
    resolve_masterstar_input_root,
)
from proc_frame_store import _vyvar_df_to_csv
from pipeline_catalog import (
    _EXPORT_PER_FRAME_WORKER_STATE,
    _MASTERSTAR_ZONE_LOG_ONCE,
    _annotate_masterstars_flux_zones,
    _apply_dao_centroid_wcs_guard,
    _build_exoplanet_promotion_rows_from_masterstars,
    _compute_airmass_from_altaz,
    _dao_convolved_background_rms_adu,
    _dao_detection_threshold_adu,
    _effective_field_catalog_cone_radius_deg,
    _exo_host_annotation_arrays,
    _extract_airmass_from_header,
    _finalize_hybrid_bkg_fallback_sidecar,
    _init_export_per_frame_worker,
    _lock_matched_centroids_to_master_grid,
    _masterstar_zone_linear_threshold,
    _proc_deduplicate_matched_catalog_rows,
    _query_exoplanet_local,
    _query_vsx_local,
    find_qc_metrics_csv,
)
from sat_diag import _box_peak_max_adu, inv_sat_limit_peak_test_adu

from catalog_match import detect_stars_and_match_catalog  # noqa: E402,F401
from frame_export import export_per_frame_catalogs  # noqa: E402,F401
from masterstar_build import generate_masterstar_and_catalog  # noqa: E402,F401

from pipeline_ui_helpers import (  # noqa: E402,F401
    _resolve_light_fits_for_quality_inspection,
)


