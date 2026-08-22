"""Streamlit entrypoint for live and archive views."""

from __future__ import annotations

import contextlib
import html
import logging
import math
import re
import sqlite3
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from config import AppConfig, save_config_json, ui_config_persist
from database import (
    DraftTechnicalMetadataError,
    get_observer_location_by_id,
    get_observer_locations,
)
from infolog import (
    clear_log,
    ensure_infolog_logging,
    get_lines,
    last_job_snapshot,
    log_event,
    log_exception,
    write_run_infolog,
    start_infolog_session,
)
from run_preflight_log import write_run_preflight_error_log
from importer import smart_import_session, smart_scan_source
from optics_selection import (
    VyvarOpticsSelection,
    log_active_optics,
    optics_from_session,
    parse_ui_optics_from_labels,
    resolve_working_optics,
    sync_optics_session,
)
from pipeline import AstroPipeline, scan_usb_folder
import ui_calibration as ui_calibration
import ui_database_explorer as ui_database_explorer
import ui_masterstar_qa as ui_masterstar_qa
import ui_quality_dashboard as ui_quality_dashboard
import ui_components as ui_components
from platesolve_ui_paths import resolve_draft_directory
from ui_aperture_photometry import render_aperture_photometry
from ui_variability import render_variability_dashboard
from utils import generate_session_id, resolve_draft_dir

LOGGER = logging.getLogger(__name__)

# MASTERSTAR: DB fallback (top % FWHM) when explicit UI paths fail to map; no session/UI control.
_DEFAULT_MASTERSTAR_SELECTION_PCT = 10.0


def _vyvar_effective_draft_dir_override() -> Path | None:
    """Explicit user-loaded draft path (``vyvar_draft_dir_override``); not the full session chain.

    Tab modules also call ``utils.resolve_draft_dir`` for job/post-cal/Drafts fallbacks.
    """
    raw = st.session_state.get("vyvar_draft_dir_override")
    if not raw:
        return None
    p = Path(str(raw)).expanduser()
    return p.resolve() if p.is_dir() else None


def _vyvar_format_run_failure_message(*, default: str) -> str:
    """Prefer on-disk preflight log path over generic Infolog hint when early failure left no draft log."""
    log_path = st.session_state.get("vyvar_last_preflight_error_log")
    detail = st.session_state.get("vyvar_last_preflight_error_message")
    if log_path:
        msg = str(detail or "Pipeline preflight failed before draft Infolog was saved.")
        return f"{msg} Preflight log: {log_path}"
    return default


def _vyvar_try_save_infolog_to_disk(cfg: AppConfig) -> str | None:
    """Best-effort Infolog snapshot under the active draft directory."""
    draft_dir = resolve_draft_dir(
        draft_dir_override=st.session_state.get("vyvar_draft_dir_override"),
        draft_id=st.session_state.get("vyvar_last_draft_id"),
        archive_root=cfg.archive_root,
    )
    if not draft_dir:
        return None
    saved_path = write_run_infolog(draft_dir)
    if saved_path:
        log_event(f"Infolog saved -> {saved_path}")
    return saved_path


def _vyvar_reset_variability_session_state() -> None:
    """Pri nacitani noveho draftu zrus variabilitu / TESS / PDF stav v session."""
    st.session_state.pop("_ap_session_id", None)
    st.session_state.pop("var_analysis_done", None)
    st.session_state.pop("tess_auto_done", None)
    st.session_state.pop("crossmatch_auto_done", None)
    st.session_state.pop("var_candidates", None)
    st.session_state.pop("var_catalog_bullets", None)
    st.session_state.pop("var_crossmatch_results", None)
    st.session_state.pop("tess_results", None)
    st.session_state.pop("accepted_period", None)
    st.session_state.pop("accepted_period_msg", None)
    st.session_state["var_analysis_timestamp"] = None
    st.session_state["pdf_ready"] = False
    st.session_state.pop("crossmatch_result", None)
    st.session_state["selected_for_export"] = []
    st.session_state.pop("var_results", None)
    st.session_state.pop("_var_run_sig", None)
    st.session_state.pop("var_candidate_count_autorun", None)


def _run_vyvar_full_pipeline(
    *,
    pipeline: AstroPipeline,
    cfg: AppConfig,
    source_root: str,
    import_equipment_id: int,
    import_telescope_id: int,
    dark_validity_days: int,
    flat_validity_days: int,
    plate_fov_ui: float,
    dao_fwhm_default: float,
    dao_sigma_default: float,
    cat_match_arc: float,
    max_cat_rows: int,
    max_extra_ps: int,
    min_stars: int,
    max_stars: int,
    max_ctrl: int,
    sat_level: float,
    footer_placeholder: Any | None = None,
    pre_calibrated_mode: bool = False,
) -> bool:
    """RUN VYVAR - automaticky retazec scan -> import -> kalibracia -> Analyze -> auto FWHM/TOP1 -> MAKE MASTERSTAR -> 0+1+2A."""
    import numpy as np

    from draft_provenance import (
        CALIBRATION_MODE_PRE,
        CALIBRATION_MODE_VYVAR,
        apply_pre_calibrated_import_plan,
        calibration_mode_report_line,
        record_draft_calibration_provenance,
        record_observer_location_provenance,
        resolve_draft_lights_root,
    )
    from photometry_core import compute_auto_fwhm_limit, run_full_photometry_pipeline
    from pipeline import (
        estimate_archive_memory_profile,
        generate_observation_hash,
        run_draft_ram_calibration_qc_to_obs_files,
        scan_calibrated_lights_pointing,
    )
    from ui_aperture_photometry import _find_phase2a_paths

    _run_label = "RUN VYVAR (non-cal)" if pre_calibrated_mode else "RUN VYVAR"

    _RUNVYVAR_FW_KEY = "_runvyvar_fwhm_threshold"

    def _update(proces: str, stav: str = "Running...") -> None:
        from infolog import log_phase_boundary  # noqa: PLC0415

        log_phase_boundary(proces, status="start")
        log_event(f"[{_run_label}] > {proces}")
        if footer_placeholder is not None:
            _vyvar_footer_set(
                footer_placeholder,
                running=True,
                process=f"{_run_label} - {proces}",
                status_detail=str(stav)[:800],
                pct=None,
            )

    def _fail(name: str, exc: BaseException) -> bool:
        log_event(f"[{_run_label}] x Zlyhanie v kroku '{name}': {exc}")
        try:
            log_path = write_run_preflight_error_log(
                cfg.data_root,
                step=name,
                exc=exc,
                db=pipeline.db,
                cfg=cfg,
            )
            st.session_state["vyvar_last_preflight_error_log"] = str(log_path)
            st.session_state["vyvar_last_preflight_error_message"] = f"{name}: {exc}"
        except Exception as log_exc:  # noqa: BLE001
            LOGGER.debug("run preflight error log write failed: %s", log_exc)
        if footer_placeholder is not None:
            _vyvar_footer_set(
                footer_placeholder,
                running=False,
                process=f"{_run_label} - {name}",
                status_detail=f"Error: {str(exc)[:700]}",
                pct=None,
            )
        return False

    def _prog_cb(i: int, total: int, msg: str) -> None:
        log_event(f"[{_run_label}] [{i}/{max(total, 1)}] {msg}")
        if footer_placeholder is not None:
            pct = int(round(100 * (i / max(total, 1))))
            _vyvar_footer_set(
                footer_placeholder,
                running=True,
                process=f"{_run_label} - sub-step",
                status_detail=str(msg)[:800],
                pct=pct,
                current_file="",
                step=f"{i} / {total}",
            )

    def _prog_phot(msg: str) -> None:
        log_event(f"[{_run_label}] {msg}")
        if footer_placeholder is not None:
            _vyvar_footer_set(
                footer_placeholder,
                running=True,
                process=f"{_run_label} - Phase 0+1 + 2A",
                status_detail=str(msg)[:800],
                pct=None,
            )

    _calibration_mode = CALIBRATION_MODE_PRE if pre_calibrated_mode else CALIBRATION_MODE_VYVAR

    try:
        st.session_state.pop("vyvar_last_preflight_error_log", None)
        st.session_state.pop("vyvar_last_preflight_error_message", None)
        from infolog import log_phase_boundary  # noqa: PLC0415

        log_phase_boundary("run_vyvar", status="start")
        _root = Path(str(source_root).strip())
        if not _root.is_dir():
            return _fail("Scan Source + Import", RuntimeError(f"Invalid Source Directory: {_root}"))

        if pre_calibrated_mode:
            log_event(calibration_mode_report_line(CALIBRATION_MODE_PRE))

        _run_optics = resolve_working_optics(
            pipeline.db,
            ui=VyvarOpticsSelection(
                int(import_equipment_id),
                int(import_telescope_id),
            ),
            context=_run_label,
        )
        sync_optics_session(_run_optics)
        log_active_optics(pipeline.db, _run_optics, context=f"{_run_label} (UI vyber)")
        import_equipment_id = _run_optics.equipment_id
        import_telescope_id = _run_optics.telescope_id

        _import_step = (
            "Scan Source + Import (pre-cal passthrough)"
            if pre_calibrated_mode
            else "Scan Source + Import + calibration"
        )
        _update(_import_step)
        plan = smart_scan_source(
            source_root=_root,
            calibration_library_root=cfg.calibration_library_root,
            masterdark_validity_days=dark_validity_days,
            masterflat_validity_days=flat_validity_days,
            db=pipeline.db,
            id_equipments=int(import_equipment_id),
            id_telescope=int(import_telescope_id),
            calibration_master_ccd_temp_tolerance_c=cfg.calibration_master_ccd_temp_tolerance_c,
        )
        st.session_state["vyvar_smart_plan"] = plan
        st.session_state.pop("vyvar_post_cal_archive_path", None)
        st.session_state.pop("vyvar_post_cal_plan_source", None)

        lights_bad = any(
            r.type == "Lights" and r.status in ("missing", "empty") for r in plan.scan_rows
        )
        if lights_bad:
            return _fail(
                _import_step,
                RuntimeError("Scan plan is missing or has empty light frames."),
            )

        manual_flat_map = st.session_state.get("vyvar_manual_flat_map") or {}
        if manual_flat_map and not pre_calibrated_mode:
            for flt, pth in manual_flat_map.items():
                if pth and Path(pth).exists():
                    plan.masterflat_by_filter[flt] = pth
        if not pre_calibrated_mode:
            _vyvar_apply_smart_plan_flat_fallbacks(plan)
        else:
            apply_pre_calibrated_import_plan(plan)

        from observer_location import (  # noqa: PLC0415
            apply_resolved_observer_location_to_config,
            resolve_observer_location_for_run,
        )

        _resolved_site = resolve_observer_location_for_run(
            cfg.database_path,
            explicit_location_id=int(cfg.observer_location_id),
            cfg=cfg,
            source_hint="ui_selection",
        )
        apply_resolved_observer_location_to_config(cfg, _resolved_site)
        result = smart_import_session(
            plan=plan,
            pipeline=pipeline,
            id_equipment=int(import_equipment_id),
            id_telescope=int(import_telescope_id),
            id_location=_resolved_site.location_id,
            location_source=_resolved_site.source,
            cfg=cfg,
        )
        st.session_state["vyvar_last_import_equipment_id"] = int(import_equipment_id)
        st.session_state["vyvar_last_import_result"] = result
        st.session_state["vyvar_last_import_plan"] = plan
        if getattr(result, "draft_id", None) is None:
            return _fail(_import_step, RuntimeError("Import did not return draft_id."))
        _did = int(result.draft_id)
        st.session_state["vyvar_last_draft_id"] = _did

        ap = Path(str(result.archive_path))
        ap_root = ap.parent if ap.name.casefold() == "non_calibrated" else ap
        from infolog import log_milestone, log_phase_boundary, start_infolog_session  # noqa: PLC0415

        start_infolog_session(ap_root)
        log_milestone(_resolved_site.milestone_line())
        record_draft_calibration_provenance(
            db=pipeline.db,
            archive_path=ap_root,
            draft_id=int(_did),
            calibration_mode=_calibration_mode,
        )
        record_observer_location_provenance(
            archive_path=ap_root,
            draft_id=int(_did),
            resolved=_resolved_site,
        )

        md = Path(plan.dark_master) if getattr(plan, "dark_master", None) else None
        mf_map: dict[str, Path | None] = {}
        if getattr(plan, "masterflat_by_filter", None):
            for k, v in (plan.masterflat_by_filter or {}).items():
                mf_map[str(k)] = Path(v) if v else None
        mf_obs: dict[str, Path | None] = {}
        for k, v in (getattr(plan, "masterflat_by_obs_key", None) or {}).items():
            mf_obs[str(k)] = Path(str(v)) if v else None
        dm_obs: dict[str, Path | None] = {}
        for k, v in (getattr(plan, "dark_master_by_obs_key", None) or {}).items():
            dm_obs[str(k)] = Path(str(v)) if v else None

        if pre_calibrated_mode:
            md = None
            mf_map = {}
            mf_obs = {}
            dm_obs = {}
            log_event(
                "Pre-calibrated mode: skipping calibration - downstream reads "
                f"{resolve_draft_lights_root(ap_root, draft_id=int(_did), db=pipeline.db)}"
            )
        else:
            _update("Calibration")
            cal_out = pipeline.quick_calibrate_last_import(
                archive_path=ap,
                master_dark_path=md if (md and md.exists()) else None,
                masterflat_by_filter=mf_map,
                progress_cb=_prog_cb,
                equipment_id=int(import_equipment_id),
                draft_id=int(_did),
                observation_id=getattr(result, "observation_id", None),
                masterflat_by_obs_key=mf_obs or None,
                master_dark_by_obs_key=dm_obs or None,
            )
            st.session_state["vyvar_post_cal_archive_path"] = str(result.archive_path)
            st.session_state["vyvar_post_cal_plan_source"] = str(plan.source_root)
            st.session_state["vyvar_status_calibrated"] = True
            last_job_snapshot(cal_out)
            log_event(f"[{_run_label}] Import hotovy - draft {_did}, archiv {result.archive_path}")

        _update("Analyze QC (RAM -> manifest files[])")
        lights_root = resolve_draft_lights_root(
            ap_root,
            draft_id=int(_did),
            db=pipeline.db,
        )
        if not lights_root.exists():
            _missing = (
                "Missing non_calibrated/lights after pre-cal import."
                if pre_calibrated_mode
                else "Missing /calibrated/lights after calibration."
            )
            return _fail("Analyze QC (RAM -> manifest files[])", FileNotFoundError(_missing))

        st.session_state["vyvar_memory_profile"] = estimate_archive_memory_profile(ap)
        _eq_a = int(import_equipment_id)
        qsum = run_draft_ram_calibration_qc_to_obs_files(
            db=pipeline.db,
            draft_id=int(_did),
            archive_path=ap_root,
            master_dark_path=md if (md and md.exists()) else None,
            masterflat_by_filter=mf_map,
            masterflat_by_obs_key=mf_obs or None,
            master_dark_by_obs_key=dm_obs or None,
            equipment_id=_eq_a,
            pipeline_config=pipeline.config,
            progress_cb=_prog_cb,
            roundness_reject_above=float(st.session_state.get("max_roundness_error", 1.25)),
        )
        pointing = scan_calibrated_lights_pointing(lights_root, max_files=None)
        r_pref = next(
            (
                r
                for r in pointing["rows"]
                if r.get("display_ra_deg") is not None and r.get("display_dec_deg") is not None
            ),
            None,
        )
        _ra_ui: float | None = None
        _de_ui: float | None = None
        if r_pref:
            _ra_ui = float(r_pref["display_ra_deg"])
            _de_ui = float(r_pref["display_dec_deg"])
        try:
            _mra_q = qsum.get("median_ra_deg")
            _mde_q = qsum.get("median_de_deg")
            if _mra_q is not None and math.isfinite(float(_mra_q)):
                _ra_ui = float(_mra_q)
            if _mde_q is not None and math.isfinite(float(_mde_q)):
                _de_ui = float(_mde_q)
        except (TypeError, ValueError):
            pass
        if _ra_ui is not None and _de_ui is not None and math.isfinite(_ra_ui) and math.isfinite(_de_ui):
            st.session_state["vyvar_pending_center_ra"] = float(_ra_ui)
            st.session_state["vyvar_pending_center_de"] = float(_de_ui)
        try:
            st.session_state["vyvar_observation_processing_hash"] = generate_observation_hash(pipeline.db, int(_did))
        except Exception:  # noqa: BLE001
            st.session_state.pop("vyvar_observation_processing_hash", None)

        _mem_prof = st.session_state.get("vyvar_memory_profile") or {}
        analyze_token = f"ram_qc:{int(_did)}:{qsum.get('n_lights')}:{qsum.get('median_fwhm')}"
        st.session_state["vyvar_last_job_output"] = {
            "job_kind": "analyze",
            "analyze_token": analyze_token,
            "archive_path": str(ap_root),
            "draft_id": int(_did),
            "ram_qc_summary": qsum,
            "qc_suggestions": {},
            "pointing_scan": pointing,
            "prefill_ra_text": "",
            "prefill_dec_text": "",
            "suggested_reject_fwhm_px": None,
            "suggest_max_detected_stars": None,
            "memory_profile": _mem_prof,
        }
        st.session_state["vyvar_last_job_summary"] = {"kind": "analyze", **qsum}
        st.session_state["vyvar_status_analyzed"] = True

        _update("Auto FWHM limit")
        _fwhm_lim = 0.0
        if bool(cfg.auto_fwhm_enabled):
            rows_f = pipeline.db.fetch_draft_light_rows_for_quality(int(_did))
            if rows_f:
                df_f = pd.DataFrame(rows_f)
                _col = next(
                    (c for c in ("fwhm_mean", "FWHM", "fwhm") if c in df_f.columns),
                    None,
                )
                if _col:
                    _vals = df_f[_col].dropna().values
                    _ar = compute_auto_fwhm_limit(_vals, k=float(cfg.auto_fwhm_k_factor))
                    if _ar.get("auto_limit") is not None:
                        _fwhm_lim = float(_ar["auto_limit"])
                        st.session_state[_RUNVYVAR_FW_KEY] = float(_fwhm_lim)
                        log_event(
                            f"[RUN VYVAR] Auto FWHM limit={_fwhm_lim:.3f} px (k={float(cfg.auto_fwhm_k_factor):.2f})"
                        )

        _update("Auto-select MASTERSTAR (TOP1)")
        rows_ms = pipeline.db.fetch_draft_light_rows_for_quality(int(_did))
        if not rows_ms:
            return _fail("Auto-select MASTERSTAR (TOP1)", RuntimeError("Empty manifest files[] for draft."))
        df_ms = pd.DataFrame(rows_ms)
        for col in (
            "CALIB_FLAGS",
            "FWHM",
            "SKY_LEVEL",
            "STAR_COUNT",
            "REJECTED_AUTO",
            "IS_REJECTED",
            "ELONGATION_MEAN",
        ):
            if col not in df_ms.columns:
                df_ms[col] = np.nan if col != "IS_REJECTED" else 0
        df_ms["FWHM"] = pd.to_numeric(df_ms["FWHM"], errors="coerce")
        df_ms["IS_REJECTED"] = pd.to_numeric(df_ms["IS_REJECTED"], errors="coerce").fillna(0).astype(int)
        _ms_eligible = df_ms[df_ms["IS_REJECTED"] == 0].copy()
        if _fwhm_lim > 0.0:
            _ms_eligible = _ms_eligible[_ms_eligible["FWHM"].notna() & (_ms_eligible["FWHM"] <= _fwhm_lim)]
        if _ms_eligible.empty:
            return _fail(
                "Auto-select MASTERSTAR (TOP1)",
                RuntimeError("No light row after IS_REJECTED and FWHM filter."),
            )
        _ms_eligible = _ms_eligible.copy()
        _ms_eligible["_ms_score"] = ui_quality_dashboard._compute_masterstar_score(_ms_eligible)
        _ms_eligible = _ms_eligible.sort_values("_ms_score", ascending=False).reset_index(drop=True)
        _top_path = str(_ms_eligible["FILE_PATH"].iloc[0]).strip()
        _p_job = ui_quality_dashboard._masterstar_candidate_path_for_job(
            ap_root, _top_path, draft_id=int(_did), db=pipeline.db
        )
        _use_path = (_p_job or _top_path).strip()
        if not _use_path:
            return _fail("Auto-select MASTERSTAR (TOP1)", RuntimeError("TOP1 has an empty path."))
        st.session_state["vyvar_masterstar_candidate_paths"] = [_use_path]
        st.session_state["vyvar_ms_candidate_top1_path"] = _use_path
        try:
            pipeline.db.set_obs_draft_masterstar_source_path(int(_did), _use_path)
        except Exception as exc:  # noqa: BLE001
            # EXC-0001: T3 -- UI diagnostic/plot only (try: / pipeline.db.set_obs_draft_masterstar_source_path(int(_d... (EXCEPT-BULK 2026-07-08)
            log_event(f"[RUN VYVAR] Zapis MASTERSTAR source do DB: {exc}")

        _update("MAKE MASTERSTAR (detrend + plate-solve + zarovnanie)")
        from pipeline import resolve_preprocess_target_coordinates

        _ira, _ide = resolve_preprocess_target_coordinates(
            db=pipeline.db,
            draft_id=int(_did),
            ui_ra_deg=_ra_ui,
            ui_dec_deg=_de_ui,
        )
        try:
            _coords_ok = (
                _ira is not None
                and _ide is not None
                and math.isfinite(float(_ira))
                and math.isfinite(float(_ide))
                and not (abs(float(_ira)) < 1e-9 and abs(float(_ide)) < 1e-9)
            )
        except (TypeError, ValueError):
            _coords_ok = False
        _ph = None
        try:
            _ph = generate_observation_hash(pipeline.db, int(_did))
            st.session_state["vyvar_observation_processing_hash"] = _ph
        except Exception:  # noqa: BLE001
            _ph = None

        _j_ms: dict[str, Any] = {
            "kind": "make_masterstar",
            "label": "MAKE MASTERSTAR (RUN VYVAR)...",
            "archive_path": str(ap),
            "fwhm_limit_px": float(_fwhm_lim),
            "inject_pointing_ra_deg": (float(_ira) if _coords_ok else None),
            "inject_pointing_dec_deg": (float(_ide) if _coords_ok else None),
            "quality_filter_draft_id": int(_did),
            "max_control_points": int(max_ctrl),
            "min_detected_stars": int(min_stars),
            "max_detected_stars": int(max_stars),
            "astrometry_api_key": "",
            "platesolve_backend": "vyvar",
            "plate_solve_fov_deg": float(plate_fov_ui),
            "max_extra_platesolve": int(max_extra_ps),
            "catalog_match_max_sep_arcsec": float(cat_match_arc),
            "saturate_level_fraction": float(sat_level),
            "max_catalog_rows": int(max_cat_rows),
            "n_comparison_stars": 0,
            "faintest_mag_limit": None,
            "dao_threshold_sigma": float(dao_sigma_default),
            "dao_fwhm_px": float(dao_fwhm_default),
            "id_equipment": int(import_equipment_id),
            "draft_id": int(_did),
            "catalog_local_gaia_only": True,
            "build_masterstar_and_catalogs": True,
            "masterstar_candidate_paths": [_use_path],
            "masterstar_selection_pct": float(_DEFAULT_MASTERSTAR_SELECTION_PCT),
        }
        if _ph:
            _j_ms["processing_hash"] = _ph
            _j_ms["overwrite_qc_processing"] = True

        pipeline.config.sips_dao_fwhm_px = float(dao_fwhm_default)
        pipeline.config.sips_dao_threshold_sigma = float(dao_sigma_default)
        _vyvar_execute_preprocess_pending(
            pending=_j_ms,
            ap=ap,
            pipeline=pipeline,
            progress_cb=_prog_cb,
        )
        _vyvar_execute_platesolve_pending(
            pending=_j_ms,
            ap=ap,
            pipeline=pipeline,
            progress_cb=_prog_cb,
        )
        _ps_out = st.session_state.get("vyvar_last_job_output")
        if isinstance(_ps_out, dict) and _ps_out.get("error"):
            return _fail("MAKE MASTERSTAR", RuntimeError(str(_ps_out.get("error"))))

        _update("Phase 0+1 + Phase 2A (photometry)")
        all_setups = _find_phase2a_paths(cfg, int(_did), draft_dir_override=None)
        if not all_setups:
            return _fail(
                "Phase 0+1 + Phase 2A (photometry)",
                RuntimeError("No platesolve setups found (per_frame_catalog_index.csv)."),
            )

        draft_dir = (Path(cfg.archive_root) / "Drafts" / f"draft_{int(_did):06d}").resolve()
        aligned_root = draft_dir / "detrended_aligned" / "lights"
        run_groups: list[str] = []
        for nm in sorted(all_setups.keys()):
            og_dir = aligned_root / str(nm)
            if og_dir.is_dir() and any(og_dir.glob("proc_*.csv")):
                run_groups.append(str(nm))
        if not run_groups:
            run_groups = list(sorted(all_setups.keys()))

        errors: list[str] = []
        error_exc: BaseException | None = None
        completed: list[str] = []
        zero_target_setups: list[str] = []
        for nm in run_groups:
            p = all_setups.get(str(nm)) or {}
            ms_fits = Path(p.get("masterstar_fits")) if p.get("masterstar_fits") else None
            og_dir = Path(p.get("obs_group_dir")) if p.get("obs_group_dir") else None
            ms_csv = (og_dir / "masterstars_full_match.csv") if og_dir is not None else None
            vt_csv = (og_dir / "variable_targets.csv") if og_dir is not None else None
            pf_dir = Path(p.get("per_frame_csv_dir")) if p.get("per_frame_csv_dir") else None
            dt_dir = Path(p.get("detrended_aligned_dir")) if p.get("detrended_aligned_dir") else None
            out_d = Path(p.get("output_dir")) if p.get("output_dir") else None
            missing: list[str] = []
            if ms_fits is None or not ms_fits.exists():
                missing.append("MASTERSTAR.fits")
            if ms_csv is None or not ms_csv.exists():
                missing.append("masterstars_full_match.csv")
            if vt_csv is None or not vt_csv.exists():
                missing.append("variable_targets.csv")
            if pf_dir is None or not pf_dir.exists():
                missing.append("per-frame CSV directory")
            if dt_dir is None or not dt_dir.exists():
                missing.append("detrended_aligned directory")
            if out_d is None:
                missing.append("output_dir")
            if missing:
                errors.append(f"{nm}: missing {', '.join(missing)}")
                continue
            try:
                phot_result = run_full_photometry_pipeline(
                    masterstar_fits_path=ms_fits,
                    variable_targets_csv=vt_csv,
                    masterstars_csv=ms_csv,
                    per_frame_csv_dir=pf_dir,
                    detrended_aligned_dir=dt_dir,
                    output_dir=out_d,
                    cfg=cfg,
                    db=pipeline.db,
                    draft_id=int(_did),
                    progress_cb=_prog_phot,
                )
                if phot_result.get("zero_targets"):
                    zero_target_setups.append(str(nm))
                    completed.append(str(nm))
                    log_event(
                        f"[RUN VYVAR] ! {nm}: 0 aktivnych cielov - fotometria nespustena "
                        "(skontroluj VSX katalog / target selection)"
                    )
                    continue
                if phot_result.get("error"):
                    errors.append(f"{nm}: {phot_result.get('error')}")
                    continue
                completed.append(str(nm))
                try:
                    from photometry_report import generate_all_method_photometry_reports  # noqa: PLC0415

                    _pdf_paths = generate_all_method_photometry_reports(
                        draft_dir=draft_dir,
                        obs_group=str(nm),
                        tess_results={},
                        base_report_title="VYVAR - Summary Measure Report",
                    )
                    for _pdf_path in _pdf_paths:
                        log_event(f"[RUN VYVAR] SUMMARY MEASURE REPORT: {Path(_pdf_path).name}")
                except Exception as _pdf_err:  # noqa: BLE001
                    # EXC-0002: T3 -- UI diagnostic/plot only (for _pdf_path in _pdf_paths: / log_event(f'[RUN VYVAR] SUMMARY... (EXCEPT-BULK 2026-07-08)
                    log_event(f"[RUN VYVAR] SUMMARY MEASURE REPORT zlyhal: {_pdf_err}")
            except Exception as exc_nm:  # noqa: BLE001
                errors.append(f"{nm}: {exc_nm}")
                if error_exc is None:
                    error_exc = exc_nm
        _astro_skips = (
            [str(s.get("setup") or "?") for s in (_ps_out.get("skipped_subgroups") or [])]
            if isinstance(_ps_out, dict)
            else []
        )
        _all_problems = list(errors) + [f"{nm}: plate-solve skipped" for nm in _astro_skips]
        if _all_problems and not completed:
            # Prefer the original exception (keeps traceback) over a rebuilt RuntimeError.
            _fail_exc: BaseException
            if error_exc is not None and len(errors) == 1 and not _astro_skips:
                _fail_exc = error_exc
            else:
                _fail_exc = RuntimeError(" ; ".join(_all_problems))
                if error_exc is not None:
                    try:
                        _fail_exc.__cause__ = error_exc
                    except Exception:  # noqa: BLE001
                        pass
            return _fail(
                "Phase 0+1 + Phase 2A (photometry)",
                _fail_exc,
            )
        if _all_problems:
            log_event(
                "! RUN VYVAR dokonceny CIASTOCNE - OK: ["
                + ", ".join(completed)
                + "]; preskocene/zlyhane: ["
                + " ; ".join(_all_problems)
                + "]"
            )

        # Fresh pipeline outputs: force Variability tab to re-run detection on next render.
        _vyvar_reset_variability_session_state()

        _photometry_ran = [s for s in completed if s not in zero_target_setups]
        if zero_target_setups and not _photometry_ran and not _all_problems:
            _zero_msg = (
                "Pipeline dokonceny - 0 aktivnych cielov, fotometria nespustena "
                "(skontroluj VSX katalog / target selection)"
            )
            log_event(f"[RUN VYVAR] ! {_zero_msg}")
            _vyvar_try_save_infolog_to_disk(cfg)
            if footer_placeholder is not None:
                _vyvar_footer_set(
                    footer_placeholder,
                    running=False,
                    process="RUN VYVAR",
                    status_detail=f"Done. ! {_zero_msg}",
                    pct=100,
                    step="",
                )
            return True
        if zero_target_setups and _photometry_ran:
            log_event(
                "! RUN VYVAR dokonceny CIASTOCNE - fotometria OK: ["
                + ", ".join(_photometry_ran)
                + "]; 0 cielov (bez fotometrie): ["
                + ", ".join(zero_target_setups)
                + "]"
            )
        elif not zero_target_setups:
            log_event("[RUN VYVAR] [OK] Pipeline dokonceny uspesne")
        _vyvar_try_save_infolog_to_disk(cfg)
        if footer_placeholder is not None:
            _vyvar_footer_set(
                footer_placeholder,
                running=False,
                process="RUN VYVAR",
                status_detail="Done. [OK]",
                pct=100,
                step="",
            )
        return True
    except Exception as exc:  # noqa: BLE001
        log_exception("[RUN VYVAR] unknown step (detail)", exc)
        return _fail("unknown step", exc)


def _vyvar_execute_preprocess_pending(
    *,
    pending: dict[str, Any],
    ap: Path,
    pipeline: AstroPipeline,
    progress_cb: Any,
) -> None:
    from draft_provenance import resolve_draft_lights_root
    from pipeline import (
        build_prefilter_rejected_map,
        calibrated_paths_for_draft_apply_filters,
        estimate_archive_memory_profile,
        qc_enrich_calibrated_lights_in_place,
        preprocess_sky_summary_from_df,
        _iter_light_fits,
    )

    _app_cfg = pipeline.config
    ap_root = ap.parent if ap.name.casefold() == "non_calibrated" else ap
    noncal = ap_root / "non_calibrated" / "lights"
    proc_root = ap_root / "processed" / "lights"
    _dqf = pending.get("quality_filter_draft_id")
    _fwhm_lim = float(pending.get("fwhm_limit_px") or 0.0)
    p1: list[Path] = []
    source_dir = (
        resolve_draft_lights_root(
            ap_root,
            draft_id=int(_dqf) if _dqf is not None else None,
            db=pipeline.db,
        )
        if _dqf is not None
        else ap_root / "calibrated" / "lights"
    )
    if _dqf is None and (not source_dir.exists()) and noncal.exists():
        source_dir = noncal

    _pp_kw = dict(
        reject_fwhm_px=(float(_fwhm_lim) if float(_fwhm_lim) > 0.0 else None),
        reject_elongation=None,
        inject_pointing_ra_deg=pending.get("inject_pointing_ra_deg"),
        inject_pointing_dec_deg=pending.get("inject_pointing_dec_deg"),
        inject_pointing_only_if_missing=False,
        app_config=_app_cfg,
    )

    if _dqf is not None:
        p1, _p_unused = calibrated_paths_for_draft_apply_filters(
            ap_root,
            pipeline.db,
            int(_dqf),
            fwhm_max_px=_fwhm_lim,
            source_dir=source_dir,
        )
        if not p1:
            _why = ["IS_REJECTED=0"]
            if _fwhm_lim > 0:
                _why.append("FWHM <= limit or FWHM is NULL")
            raise FileNotFoundError(
                "QC filter: no frames matching " + ", ".join(_why) + "."
            )
        dfs_pp: list[pd.DataFrame] = []
        _sky_skip_total = 0
        _sky_force_reapply = False
        _all_lights = _iter_light_fits(source_dir)
        _prefilter_map = build_prefilter_rejected_map(_all_lights, p1)
        tot_pp = len(_all_lights)
        off_pp = 0

        def _pcb_pp(off0: int):
            def _inner(i: int, _t: int, msg: str) -> None:
                if progress_cb is not None:
                    progress_cb(off0 + i, max(tot_pp, 1), msg)

            return _inner

        if p1:
            if not source_dir.exists():
                raise FileNotFoundError("Missing source lights directory for preprocess filter.")
            dfs_pp.append(
                qc_enrich_calibrated_lights_in_place(
                    calibrated_root=source_dir,
                    only_paths=None,
                    prefilter_rejected=_prefilter_map,
                    progress_cb=_pcb_pp(off_pp),
                    db=pipeline.db,
                    draft_id=(int(_dqf) if _dqf is not None else None),
                    **_pp_kw,
                )
            )
            _pp_sum = preprocess_sky_summary_from_df(dfs_pp[-1])
            _sky_skip_total += int(_pp_sum.get("sky_surface_skip_count") or 0)
            _sky_force_reapply = _sky_force_reapply or bool(_pp_sum.get("sky_surface_force_reapply"))
            off_pp += len(p1)
        df = pd.concat(dfs_pp, ignore_index=True) if dfs_pp else pd.DataFrame()
        if dfs_pp:
            df.attrs["preprocess_sky_summary"] = {
                "sky_surface_skip_count": _sky_skip_total,
                "sky_surface_force_reapply": _sky_force_reapply,
            }
    else:
        if not source_dir.exists():
            raise FileNotFoundError("Missing source lights directory. Run calibration/import first.")
        if _fwhm_lim > 0:
            log_event(
                f"Detrend: draft_id chyba - FWHM limit sa neaplikuje z DB; spracuvam vsetky FITS v {source_dir}."
            )
        df = qc_enrich_calibrated_lights_in_place(
            calibrated_root=source_dir,
            progress_cb=progress_cb,
            db=pipeline.db,
            draft_id=None,
            **_pp_kw,
        )
    out2 = pipeline.quick_preprocess_last_import(archive_path=ap_root, run=False)
    st.session_state["vyvar_memory_profile"] = estimate_archive_memory_profile(ap_root)
    st.session_state["vyvar_last_qc_suggestions"] = out2.get("qc_suggestions", {})
    st.session_state["vyvar_last_job_output"] = out2
    st.session_state["vyvar_status_calibrated"] = bool(source_dir.exists())
    try:
        rej_pp = (
            int((df["status"].astype(str).str.startswith("rejected")).sum())
            if not df.empty and "status" in df.columns
            else 0
        )
        _root_pp = str(source_dir)
        _sky_sum = preprocess_sky_summary_from_df(df)
        st.session_state["vyvar_last_job_summary"] = {
            "kind": "preprocess",
            "rows": int(len(df)),
            "rejected": rej_pp,
            "root": _root_pp,
            "sky_surface_skip_count": int(_sky_sum.get("sky_surface_skip_count") or 0),
            "sky_surface_force_reapply": bool(_sky_sum.get("sky_surface_force_reapply")),
        }
    except Exception:  # noqa: BLE001
        st.session_state["vyvar_last_job_summary"] = None
    else:
        st.session_state.pop("vyvar_staged_preprocess_job", None)
        st.session_state.pop("vyvar_staged_processing_hash", None)
        _ph_done = pending.get("processing_hash")
        _dqf_done = pending.get("quality_filter_draft_id")
        if _dqf_done is not None and _ph_done:
            try:
                pipeline.db.record_qc_processing_apply(
                    int(_dqf_done),
                    str(_ph_done),
                    overwrite=bool(pending.get("overwrite_qc_processing")),
                )
                pipeline.db.update_obs_draft_status(int(_dqf_done), "PROCESSED")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Writing QC snapshot / draft status failed: {exc}")


def _vyvar_execute_platesolve_pending(
    *,
    pending: dict[str, Any],
    ap: Path,
    pipeline: AstroPipeline,
    progress_cb: Any,
) -> None:
    from pipeline import astrometry_align_and_build_masterstar, estimate_archive_memory_profile

    st.session_state["vyvar_memory_profile"] = estimate_archive_memory_profile(ap)
    _fwhm_ui = float(
        st.session_state.get(
            "dao_fwhm_px",
            pending.get("dao_fwhm_px", getattr(pipeline.config, "sips_dao_fwhm_px", 2.5)),
        )
    )
    _sigma_ui = float(
        st.session_state.get(
            "dao_threshold_sigma",
            pending.get(
                "dao_threshold_sigma",
                getattr(pipeline.config, "sips_dao_threshold_sigma", 3.5),
            ),
        )
    )
    pipeline.config.sips_dao_fwhm_px = _fwhm_ui
    pipeline.config.sips_dao_threshold_sigma = _sigma_ui
    _cfg_run = pipeline.config
    try:
        _cfg_run.sips_dao_fwhm_px = float(
            pending.get("dao_fwhm_px", st.session_state.get("dao_fwhm_px", _cfg_run.sips_dao_fwhm_px))
        )
    except (TypeError, ValueError):
        pass
    try:
        _cfg_run.sips_dao_threshold_sigma = float(
            pending.get(
                "dao_threshold_sigma",
                st.session_state.get("dao_threshold_sigma", _cfg_run.sips_dao_threshold_sigma),
            )
        )
    except (TypeError, ValueError):
        pass
    _peq = pending.get("id_equipment")
    _ptel = pending.get("id_telescope")
    _draft_ps = pending.get("draft_id")
    try:
        _ui_ps = None
        if _peq is not None and _ptel is not None:
            _ui_ps = VyvarOpticsSelection(int(_peq), int(_ptel))
        _ps_optics = resolve_working_optics(
            pipeline.db,
            draft_id=int(_draft_ps) if _draft_ps is not None else None,
            ui=_ui_ps,
            context="platesolve job",
        )
        _peq = _ps_optics.equipment_id
        log_active_optics(
            pipeline.db,
            _ps_optics,
            draft_id=int(_draft_ps) if _draft_ps is not None else None,
            context="platesolve job",
        )
    except ValueError as _opt_exc:
        log_event(f"platesolve job: optics resolve failed: {_opt_exc}")
    _ms_pct_job = pending.get("masterstar_selection_pct")
    try:
        _ms_pct_job_f = (
            float(_ms_pct_job)
            if _ms_pct_job is not None
            else float(_DEFAULT_MASTERSTAR_SELECTION_PCT)
        )
    except (TypeError, ValueError):
        _ms_pct_job_f = float(_DEFAULT_MASTERSTAR_SELECTION_PCT)
    _plan_ps = st.session_state.get("vyvar_last_import_plan")
    _md_ps = (
        Path(_plan_ps.dark_master)
        if _plan_ps and getattr(_plan_ps, "dark_master", None)
        else None
    )
    if _md_ps is not None and not _md_ps.exists():
        _md_ps = None
    outp = astrometry_align_and_build_masterstar(
        archive_path=ap,
        app_config=_cfg_run,
        astrometry_api_key=(str(pending.get("astrometry_api_key", "")).strip() or None),
        max_control_points=int(pending.get("max_control_points", _cfg_run.alignment_max_control_points)),
        min_detected_stars=int(pending.get("min_detected_stars", 100)),
        max_detected_stars=int(pending.get("max_detected_stars", 500)),
        platesolve_backend=str(pending.get("platesolve_backend", "vyvar")),
        plate_solve_fov_deg=float(pending.get("plate_solve_fov_deg", 1.0)),
        max_extra_platesolve=int(pending.get("max_extra_platesolve", 0)),
        catalog_match_max_sep_arcsec=float(
            pending.get("catalog_match_max_sep_arcsec", 25.0)
        ),
        saturate_level_fraction=float(pending.get("saturate_level_fraction", 0.999)),
        max_catalog_rows=int(pending.get("max_catalog_rows", 12000)),
        n_comparison_stars=int(pending.get("n_comparison_stars", 0)),
        faintest_mag_limit=(
            None
            if pending.get("faintest_mag_limit") is None
            else float(pending["faintest_mag_limit"])
        ),
        dao_threshold_sigma=float(
            pending.get(
                "dao_threshold_sigma",
                st.session_state.get("dao_threshold_sigma", 3.5),
            )
        ),
        id_equipment=int(_peq) if _peq is not None else None,
        draft_id=int(_draft_ps) if _draft_ps is not None else None,
        catalog_local_gaia_only=True,
        build_masterstar_and_catalogs=bool(
            pending.get("build_masterstar_and_catalogs", False)
        ),
        ram_align_and_catalog=True,
        progress_cb=progress_cb,
        masterstar_candidate_paths=list(pending.get("masterstar_candidate_paths") or []),
        masterstar_selection_pct=_ms_pct_job_f,
        master_dark_path=_md_ps,
    )
    st.session_state["vyvar_last_job_output"] = outp
    st.session_state["vyvar_last_job_summary"] = {
        "kind": "platesolve",
        "aligned": int(outp.get("aligned_frames", 0)),
        "input": int(outp.get("input_frames", 0)),
        "masterstar_built": bool(outp.get("build_masterstar_and_catalogs")),
        "per_frame_csv": int(outp.get("per_frame_catalogs_written") or 0),
        "ram_align_handoff": bool(outp.get("ram_align_handoff_used")),
    }


_VYVAR_FOOTER_CSS = """
<style>
.vyvar-footer-bar {
  position: fixed;
  left: 0;
  right: 0;
  bottom: 0;
  z-index: 999991;
  background: linear-gradient(180deg, #1c1c24 0%, #12121a 100%);
  color: #e8e8ef;
  border-top: 1px solid #3d3d52;
  padding: 0.45rem 1rem 0.5rem 1rem;
  font-size: 0.8125rem;
  line-height: 1.4;
  box-shadow: 0 -8px 28px rgba(0,0,0,0.5);
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.6rem 1.5rem;
}
.vyvar-footer-bar .vyvar-ft-seg { display: inline-flex; align-items: baseline; gap: 0.35rem; max-width: 100%; }
.vyvar-footer-bar .vyvar-ft-k {
  color: #9494b0;
  font-weight: 600;
  font-size: 0.65rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  flex-shrink: 0;
}
.vyvar-footer-bar .vyvar-ft-v { color: #f4f4fa; word-break: break-word; }
.vyvar-footer-bar .vyvar-ft-pill-run {
  background: #2563eb;
  color: #fff;
  padding: 0.1rem 0.5rem;
  border-radius: 4px;
  font-size: 0.7rem;
  font-weight: 600;
  flex-shrink: 0;
}
.vyvar-footer-bar .vyvar-ft-pill-idle {
  background: #3f3f4f;
  color: #c8c8d4;
  padding: 0.1rem 0.5rem;
  border-radius: 4px;
  font-size: 0.7rem;
  font-weight: 600;
  flex-shrink: 0;
}
section.main > div.block-container { padding-bottom: 3.5rem !important; }
</style>
"""


def _vyvar_apply_smart_plan_flat_fallbacks(plan: Any) -> None:
    """Apply per-observation flat fallback choices from session_state to the live SmartImportPlan."""
    ogs = getattr(plan, "observation_groups", None) or {}
    if not ogs:
        return
    mf = dict(getattr(plan, "masterflat_by_obs_key", None) or {})
    for p in getattr(plan, "flat_fallback_prompts", None) or []:
        gk = str(p.get("group_key") or "")
        if not gk:
            continue
        choice = st.session_state.get(f"vyvar_flatfb_{gk}", "__skip__")
        if choice and choice != "__skip__":
            src = mf.get(str(choice))
            if src and Path(str(src)).is_file():
                mf[gk] = src
    missing = sorted(
        gk
        for gk in ogs
        if not (mf.get(gk) and str(mf[gk]).strip() and Path(str(mf[gk])).is_file())
    )
    plan.masterflat_by_obs_key = mf
    plan.missing_obs_keys = missing
    plan.missing_flat_filters = sorted({ogs[gk]["filter"] for gk in missing})
    mfb: dict[str, Any] = {}
    for gk, g in ogs.items():
        fln = g["filter"]
        pth = mf.get(gk)
        if fln not in mfb or pth is not None and mfb[fln] is None:
            mfb[fln] = pth
    plan.masterflat_by_filter = mfb


def _vyvar_guess_filename_from_progress(msg: str) -> str:
    if not msg:
        return ""
    m = re.search(r"([\w\-+.]+(?:\\[\w\-+.]+)*\.(?:fits|fit|fts))\b", msg, re.I)
    if m:
        return Path(m.group(1).replace("\\", "/")).name
    return ""


def vyvar_init_footer_state_if_missing() -> None:
    if "vyvar_footer_state" not in st.session_state:
        st.session_state["vyvar_footer_state"] = {
            "running": False,
            "process": "VYVAR",
            "status_detail": "Ready - start a job on the VARSTREM tab.",
            "pct": None,
            "current_file": "",
            "step": "",
        }


def _vyvar_footer_set(
    footer_placeholder: Any | None,
    *,
    running: bool,
    process: str,
    status_detail: str,
    pct: int | None = None,
    current_file: str = "",
    step: str = "",
) -> None:
    vyvar_init_footer_state_if_missing()
    st.session_state["vyvar_footer_state"] = {
        "running": bool(running),
        "process": str(process),
        "status_detail": str(status_detail)[:800],
        "pct": pct,
        "current_file": str(current_file)[:500],
        "step": str(step)[:200],
    }
    if footer_placeholder is not None:
        _vyvar_render_fixed_footer_into(footer_placeholder)


def _vyvar_render_fixed_footer_into(placeholder: Any) -> None:
    vyvar_init_footer_state_if_missing()
    fs = st.session_state["vyvar_footer_state"]
    running = bool(fs.get("running"))
    proc = html.escape(str(fs.get("process") or "-"))
    detail = html.escape(str(fs.get("status_detail") or ""))
    cfile_raw = str(fs.get("current_file") or "").strip()
    cfile = html.escape(cfile_raw) if cfile_raw else ""
    step = html.escape(str(fs.get("step") or ""))
    pct = fs.get("pct")
    pct_html = ""
    if running and pct is not None:
        try:
            p = int(pct)
            pct_html = f'<span class="vyvar-ft-pill-run">{html.escape(str(p))} %</span>'
        except (TypeError, ValueError):
            pct_html = '<span class="vyvar-ft-pill-run">...</span>'
    elif not running:
        pct_html = '<span class="vyvar-ft-pill-idle">Idle</span>'

    file_seg = ""
    if cfile:
        file_seg = (
            f'<span class="vyvar-ft-seg"><span class="vyvar-ft-k">File</span>'
            f'<span class="vyvar-ft-v">{cfile}</span></span>'
        )
    step_seg = ""
    if step:
        step_seg = (
            f'<span class="vyvar-ft-seg"><span class="vyvar-ft-k">Step</span>'
            f'<span class="vyvar-ft-v">{step}</span></span>'
        )

    inner = (
        f"{pct_html}"
        f'<span class="vyvar-ft-seg"><span class="vyvar-ft-k">Process</span>'
        f'<span class="vyvar-ft-v">{proc}</span></span>'
        f'<span class="vyvar-ft-seg"><span class="vyvar-ft-k">Status</span>'
        f'<span class="vyvar-ft-v">{detail or "-"}</span></span>'
        f"{file_seg}{step_seg}"
    )
    placeholder.markdown(
        _VYVAR_FOOTER_CSS + f'<div class="vyvar-footer-bar" data-testid="vyvar-footer-bar">{inner}</div>',
        unsafe_allow_html=True,
    )



def _gaia_db_ok_for_masterstar(gaia_db_path: str) -> bool:
    try:
        from database import validate_gaia_db_schema

        _gdb = str(gaia_db_path or "").strip()
        ok, msg = validate_gaia_db_schema(_gdb)
        if not ok:
            st.error("[X] Gaia DR3 database not found. Please set the path in Settings.")
            st.caption(f"Detail: {msg}")
            return False
    except Exception:  # noqa: BLE001
        st.error("[X] Gaia DR3 database not found. Please set the path in Settings.")
        return False
    return True


def _sync_varstrem_session_state(pipeline: AstroPipeline) -> None:
    """Non-UI session defaults for RUN VYVAR, pending jobs, and Quality Dashboard."""
    _rv_fwhm_m = st.session_state.pop("_runvyvar_fwhm_threshold", None)
    if _rv_fwhm_m is not None:
        try:
            st.session_state["fwhm_threshold"] = float(_rv_fwhm_m)
        except (TypeError, ValueError):
            st.session_state["fwhm_threshold"] = 0.0

    if "fwhm_threshold" not in st.session_state:
        _fb = st.session_state.get("fwhm_limit")
        if _fb is None:
            _fb = st.session_state.get("vyvar_ui_reject_fwhm", 0.0)
        try:
            st.session_state["fwhm_threshold"] = float(_fb)
        except (TypeError, ValueError):
            st.session_state["fwhm_threshold"] = 0.0

    _ra_key = ui_components.DRAFT_CENTER_RA_STATE_KEY
    _de_key = ui_components.DRAFT_CENTER_DE_STATE_KEY
    _pending_ra = st.session_state.get("vyvar_pending_center_ra")
    _pending_de = st.session_state.get("vyvar_pending_center_de")
    try:
        if _pending_ra is not None and math.isfinite(float(_pending_ra)):
            st.session_state[_ra_key] = float(_pending_ra)
            st.session_state["center_ra"] = float(_pending_ra)
    except (TypeError, ValueError):
        pass
    try:
        if _pending_de is not None and math.isfinite(float(_pending_de)):
            st.session_state[_de_key] = float(_pending_de)
            st.session_state["center_de"] = float(_pending_de)
    except (TypeError, ValueError):
        pass
    st.session_state.pop("vyvar_pending_center_ra", None)
    st.session_state.pop("vyvar_pending_center_de", None)
    if _ra_key not in st.session_state or _de_key not in st.session_state:
        _did_center = st.session_state.get("vyvar_last_draft_id")
        _db_ra: float | None = None
        _db_de: float | None = None
        if _did_center is not None:
            try:
                _drow_center = pipeline.db.fetch_obs_draft_by_id(int(_did_center)) or {}
                _ra_raw = _drow_center.get("CENTEROFFIELDRA")
                _de_raw = _drow_center.get("CENTEROFFIELDDE")
                _db_ra = float(_ra_raw) if _ra_raw is not None and math.isfinite(float(_ra_raw)) else None
                _db_de = float(_de_raw) if _de_raw is not None and math.isfinite(float(_de_raw)) else None
            except Exception:  # noqa: BLE001
                _db_ra, _db_de = None, None
        if _ra_key not in st.session_state:
            try:
                _legacy_ra = float(st.session_state.get("center_ra", float("nan")))
            except (TypeError, ValueError):
                _legacy_ra = float("nan")
            st.session_state[_ra_key] = (
                float(_db_ra)
                if _db_ra is not None
                else (float(_legacy_ra) if math.isfinite(_legacy_ra) else 0.0)
            )
        if _de_key not in st.session_state:
            try:
                _legacy_de = float(st.session_state.get("center_de", float("nan")))
            except (TypeError, ValueError):
                _legacy_de = float("nan")
            st.session_state[_de_key] = (
                float(_db_de)
                if _db_de is not None
                else (float(_legacy_de) if math.isfinite(_legacy_de) else 0.0)
            )
    st.session_state["center_ra"] = float(st.session_state.get(_ra_key, 0.0))
    st.session_state["center_de"] = float(st.session_state.get(_de_key, 0.0))
    if "drift_limit_arcmin" not in st.session_state:
        st.session_state["drift_limit_arcmin"] = 5.0
    if "max_roundness_error" not in st.session_state:
        st.session_state["max_roundness_error"] = 1.25
    if "vyvar_ui_max_align_stars" not in st.session_state:
        st.session_state["vyvar_ui_max_align_stars"] = 500

    _ljo_apply = st.session_state.get("vyvar_last_job_output")
    if isinstance(_ljo_apply, dict) and _ljo_apply.get("job_kind") == "analyze":
        _tok_a = str(_ljo_apply.get("analyze_token", ""))
        if st.session_state.get("vyvar_applied_analyze_token") != _tok_a:
            _sm = _ljo_apply.get("suggest_max_detected_stars")
            if _sm is not None:
                try:
                    sm_i = int(_sm)
                    if math.isfinite(float(sm_i)):
                        st.session_state["vyvar_ui_max_align_stars"] = int(max(100, min(5000, sm_i)))
                except (TypeError, ValueError):
                    pass
            st.session_state["vyvar_applied_analyze_token"] = _tok_a
            st.session_state["vyvar_pointing_scan_cache"] = _ljo_apply.get("pointing_scan")
            _rq = _ljo_apply.get("ram_qc_summary") or {}
            _mra = _rq.get("median_ra_deg")
            _mde = _rq.get("median_de_deg")
            try:
                if _mra is not None and math.isfinite(float(_mra)):
                    st.session_state[_ra_key] = float(_mra)
                    st.session_state["center_ra"] = float(_mra)
                if _mde is not None and math.isfinite(float(_mde)):
                    st.session_state[_de_key] = float(_mde)
                    st.session_state["center_de"] = float(_mde)
            except (TypeError, ValueError):
                pass

    _did_persist = st.session_state.get("vyvar_last_draft_id")
    if _did_persist is not None:
        try:
            _cur_ra = float(st.session_state.get(_ra_key, 0.0))
            _cur_de = float(st.session_state.get(_de_key, 0.0))
            _sig_now = f"{int(_did_persist)}|{_cur_ra:.9f}|{_cur_de:.9f}"
            _sig_prev = str(st.session_state.get("vyvar_last_saved_draft_center_sig", ""))
            if _sig_now != _sig_prev:
                pipeline.db.update_obs_draft_status_panel_values(
                    int(_did_persist),
                    center_ra_deg=_cur_ra,
                    center_de_deg=_cur_de,
                )
                st.session_state["vyvar_last_saved_draft_center_sig"] = _sig_now
        except Exception as _exc_center:  # noqa: BLE001
            # EXC-0003: T3 -- UI diagnostic/plot only () / st.session_state['vyvar_last_saved_draft_center_sig'] = _s... (EXCEPT-BULK 2026-07-08)
            log_event(f"Draft center save skipped: {_exc_center!s}")


def _render_pending_job_dispatcher(
    pipeline: AstroPipeline,
    cfg: AppConfig,
    footer_placeholder: Any | None,
) -> bool:
    """Run the pending VYVAR job if one is queued in session state.

    Returns True if a job was dispatched (caller should expect st.rerun()).
    Returns False if no pending job.
    """
    pending = st.session_state.get("vyvar_pending_job")
    if not pending:
        return False

    vyvar_init_footer_state_if_missing()
    st.session_state["vyvar_footer_state"] = {
        "running": True,
        "process": str(pending.get("label") or pending.get("kind") or "job"),
        "status_detail": "Starting...",
        "pct": 0,
        "current_file": "",
        "step": "",
    }
    if footer_placeholder is not None:
        _vyvar_render_fixed_footer_into(footer_placeholder)

    ap = Path(pending.get("archive_path", "")) if pending.get("archive_path") else None
    _hide_inline_job_status = str(pending.get("kind") or "").strip().lower() in {
        "platesolve",
        "make_masterstar",
        "masterstar_catalog_only",
    }
    progress_bar = None if _hide_inline_job_status else st.progress(0, text="Starting...")
    _status_ctx: Any = (
        contextlib.nullcontext(None)
        if _hide_inline_job_status
        else st.status(pending.get("label", "Running..."), expanded=False)
    )
    with _status_ctx as stt:
        try:
            if ap is None or not ap.exists():
                raise FileNotFoundError("Missing/invalid archive path for job.")
            log_event(f"- Spustam: {pending.get('kind')} - {ap}")

            def _cb(i: int, total: int, msg: str) -> None:
                pct = int(round(100 * (i / max(total, 1))))
                if i > 0 and pct < 1:
                    pct = 1
                if progress_bar is not None:
                    progress_bar.progress(pct, text=msg)
                log_event(f"[{i}/{total}] {msg}")
                fn = _vyvar_guess_filename_from_progress(msg)
                st.session_state["vyvar_footer_state"] = {
                    "running": True,
                    "process": str(pending.get("label") or pending.get("kind") or "job"),
                    "status_detail": msg[:800],
                    "pct": pct,
                    "current_file": fn,
                    "step": f"{i} / {total}",
                }
                if footer_placeholder is not None:
                    _vyvar_render_fixed_footer_into(footer_placeholder)

            if pending.get("kind") == "analyze":
                from pipeline import (
                    estimate_archive_memory_profile,
                    run_draft_ram_calibration_qc_to_obs_files,
                    scan_calibrated_lights_pointing,
                )

                ap_root = ap.parent if ap.name.casefold() == "non_calibrated" else ap
                cal = ap_root / "calibrated" / "lights"
                if not cal.exists():
                    raise FileNotFoundError("Missing /calibrated/lights. Run calibration first.")
                plan = st.session_state.get("vyvar_last_import_plan")
                if plan is None:
                    raise FileNotFoundError(
                        "Missing import calibration plan - run **Create Archive & Do Calibration** again "
                        "(session needs `vyvar_last_import_plan` with masters)."
                    )
                _draft_a = pending.get("draft_id")
                if _draft_a is None:
                    _draft_a = st.session_state.get("vyvar_last_draft_id")
                if _draft_a is None:
                    raise FileNotFoundError("Missing Draft ID - import a draft to the archive first.")
                _eq_a = pending.get("equipment_id")
                if _eq_a is None:
                    _eq_a = st.session_state.get("vyvar_last_import_equipment_id")
                _mem_prof = estimate_archive_memory_profile(ap)
                st.session_state["vyvar_memory_profile"] = _mem_prof
                _qcm = _mem_prof.get("qc_analyze") or {}
                if _qcm:
                    log_event(
                        f"Odhad RAM (pred analyze): QC peak ~{_qcm.get('estimated_peak_human', '?')}, "
                        f"snimky: {_qcm.get('n_files', 0)}, volne: {_mem_prof.get('available_ram_human', '?')}"
                    )
                md = Path(plan.dark_master) if getattr(plan, "dark_master", None) else None
                mf_map: dict[str, Path | None] = {}
                if getattr(plan, "masterflat_by_filter", None):
                    for k, v in (plan.masterflat_by_filter or {}).items():
                        mf_map[str(k)] = Path(v) if v else None
                mf_obs: dict[str, Path | None] = {}
                for k, v in (getattr(plan, "masterflat_by_obs_key", None) or {}).items():
                    mf_obs[str(k)] = Path(str(v)) if v else None
                dm_obs: dict[str, Path | None] = {}
                for k, v in (getattr(plan, "dark_master_by_obs_key", None) or {}).items():
                    dm_obs[str(k)] = Path(str(v)) if v else None
                with st.spinner(
                    "RAM calibration analysis (no calibrated FITS written) - may take a while..."
                ):
                    qsum = run_draft_ram_calibration_qc_to_obs_files(
                        db=pipeline.db,
                        draft_id=int(_draft_a),
                        archive_path=ap_root,
                        master_dark_path=md if (md and md.exists()) else None,
                        masterflat_by_filter=mf_map,
                        masterflat_by_obs_key=mf_obs or None,
                        master_dark_by_obs_key=dm_obs or None,
                        equipment_id=int(_eq_a) if _eq_a is not None else None,
                        pipeline_config=pipeline.config,
                        progress_cb=_cb,
                        roundness_reject_above=float(
                            pending.get("roundness_reject_above")
                            if pending.get("roundness_reject_above") is not None
                            else st.session_state.get("max_roundness_error", 1.25)
                        ),
                    )
                # Pointing scan must read the same folder that was analyzed.
                pointing = scan_calibrated_lights_pointing(cal, max_files=None)
                r_pref = next(
                    (
                        r
                        for r in pointing["rows"]
                        if r.get("display_ra_deg") is not None
                        and r.get("display_dec_deg") is not None
                    ),
                    None,
                )
                if r_pref:
                    _pra = float(r_pref["display_ra_deg"])
                    _pde = float(r_pref["display_dec_deg"])
                    prefill_ra = f"{_pra:.10f}".rstrip("0").rstrip(".")
                    prefill_dec = f"{_pde:.10f}".rstrip("0").rstrip(".")
                else:
                    prefill_ra, prefill_dec = "", ""
                analyze_token = f"ram_qc:{int(_draft_a)}:{qsum.get('n_lights')}:{qsum.get('median_fwhm')}"
                outa = {
                    "job_kind": "analyze",
                    "analyze_token": analyze_token,
                    "archive_path": str(ap_root),
                    "draft_id": int(_draft_a),
                    "ram_qc_summary": qsum,
                    "qc_suggestions": {},
                    "pointing_scan": pointing,
                    "prefill_ra_text": prefill_ra,
                    "prefill_dec_text": prefill_dec,
                    "suggested_reject_fwhm_px": None,
                    "suggest_max_detected_stars": None,
                    "memory_profile": _mem_prof,
                }
                st.session_state["vyvar_last_qc_suggestions"] = outa.get("qc_suggestions", {})
                st.session_state.pop("vyvar_last_qc_csv", None)
                st.session_state["vyvar_last_job_output"] = outa
                st.session_state["vyvar_last_job_summary"] = {"kind": "analyze", **qsum}
                st.session_state["vyvar_status_analyzed"] = True
                try:
                    _mra_q = qsum.get("median_ra_deg")
                    _mde_q = qsum.get("median_de_deg")
                    if _mra_q is not None and math.isfinite(float(_mra_q)):
                        st.session_state[ui_components.DRAFT_CENTER_RA_STATE_KEY] = float(_mra_q)
                        st.session_state["center_ra"] = float(_mra_q)
                    if _mde_q is not None and math.isfinite(float(_mde_q)):
                        st.session_state[ui_components.DRAFT_CENTER_DE_STATE_KEY] = float(_mde_q)
                        st.session_state["center_de"] = float(_mde_q)
                except (TypeError, ValueError):
                    pass
                try:
                    from pipeline import generate_observation_hash

                    st.session_state["vyvar_observation_processing_hash"] = generate_observation_hash(
                        pipeline.db, int(_draft_a)
                    )
                except Exception:  # noqa: BLE001
                    st.session_state.pop("vyvar_observation_processing_hash", None)

            elif pending.get("kind") == "preprocess":
                _vyvar_execute_preprocess_pending(
                    pending=pending, ap=ap, pipeline=pipeline, progress_cb=_cb
                )

            elif pending.get("kind") == "make_masterstar":
                _vyvar_execute_preprocess_pending(
                    pending=pending, ap=ap, pipeline=pipeline, progress_cb=_cb
                )
                _pp_job_summary = st.session_state.get("vyvar_last_job_summary")
                _vyvar_execute_platesolve_pending(
                    pending=pending, ap=ap, pipeline=pipeline, progress_cb=_cb
                )
                _ps_job_summary = st.session_state.get("vyvar_last_job_summary")
                if isinstance(_ps_job_summary, dict):
                    _merged = dict(_ps_job_summary)
                    _merged["kind"] = "make_masterstar"
                    if isinstance(_pp_job_summary, dict):
                        _merged["preprocess_summary"] = {
                            k: _pp_job_summary.get(k)
                            for k in ("rows", "rejected", "root")
                            if k in _pp_job_summary
                        }
                    st.session_state["vyvar_last_job_summary"] = _merged

            elif pending.get("kind") == "quality_analysis":
                from pipeline import run_quality_analysis

                _qd = int(pending["draft_id"])
                _rna_q = pending.get("roundness_reject_above")
                summary = run_quality_analysis(
                    db=pipeline.db,
                    draft_id=_qd,
                    archive_path=ap,
                    progress_cb=_cb,
                    roundness_reject_above=float(
                        _rna_q if _rna_q is not None else st.session_state.get("max_roundness_error", 1.25)
                    ),
                )
                st.session_state["vyvar_last_job_output"] = {
                    "job_kind": "quality_analysis",
                    "archive_path": str(ap),
                    "quality_summary": summary,
                }
                st.session_state["vyvar_last_job_summary"] = {"kind": "quality_analysis", **summary}
                st.session_state["vyvar_status_analyzed"] = True

            elif pending.get("kind") == "platesolve":
                _vyvar_execute_platesolve_pending(
                    pending=pending, ap=ap, pipeline=pipeline, progress_cb=_cb
                )

            elif pending.get("kind") == "masterstar_catalog_only":
                from pipeline import (
                    _equipment_saturate_adu_from_db,
                    estimate_archive_memory_profile,
                    generate_masterstar_and_catalog,
                )

                _ms_fits_only = bool(pending.get("masterstar_fits_only"))
                _ms_skip_build = bool(pending.get("masterstar_skip_build"))
                if _ms_fits_only:
                    log_event(
                        f"- MASTERSTAR (len FITS, bez solve): ui_action={pending.get('masterstar_ui_action')!r}, "
                        "zapis do platesolve/..."
                    )
                elif _ms_skip_build:
                    log_event(
                        "- MASTERSTAR platesolve + katalog: vstup = existujuci platesolve/MASTERSTAR.fits "
                        f"(skip build z processed), draft_id={pending.get('draft_id')!r}."
                    )

                st.session_state["vyvar_memory_profile"] = estimate_archive_memory_profile(ap)
                _fwhm_ui = float(
                    st.session_state.get(
                        "dao_fwhm_px",
                        pending.get("dao_fwhm_px", getattr(pipeline.config, "sips_dao_fwhm_px", 2.5)),
                    )
                )
                _sigma_ui = float(
                    st.session_state.get(
                        "dao_threshold_sigma",
                        pending.get(
                            "dao_threshold_sigma",
                            getattr(pipeline.config, "sips_dao_threshold_sigma", 3.5),
                        ),
                    )
                )
                pipeline.config.sips_dao_fwhm_px = _fwhm_ui
                pipeline.config.sips_dao_threshold_sigma = _sigma_ui
                _cfg_run = pipeline.config
                try:
                    _cfg_run.sips_dao_fwhm_px = float(
                        pending.get("dao_fwhm_px", st.session_state.get("dao_fwhm_px", _cfg_run.sips_dao_fwhm_px))
                    )
                except (TypeError, ValueError):
                    pass
                try:
                    _cfg_run.sips_dao_threshold_sigma = float(
                        pending.get(
                            "dao_threshold_sigma",
                            st.session_state.get("dao_threshold_sigma", _cfg_run.sips_dao_threshold_sigma),
                        )
                    )
                except (TypeError, ValueError):
                    pass
                _peq_ms = pending.get("id_equipment")
                _draft_ms = pending.get("draft_id")
                _ms_pct_mc = pending.get("masterstar_selection_pct")
                try:
                    _ms_pct_mc_f = (
                        float(_ms_pct_mc)
                        if _ms_pct_mc is not None
                        else float(_DEFAULT_MASTERSTAR_SELECTION_PCT)
                    )
                except (TypeError, ValueError):
                    _ms_pct_mc_f = float(_DEFAULT_MASTERSTAR_SELECTION_PCT)
                _equip_sat = _equipment_saturate_adu_from_db(
                    int(_peq_ms) if _peq_ms is not None else None
                )
                if _draft_ms is not None:
                    try:
                        from pipeline import resolve_masterstar_input_root

                        _ms_root = resolve_masterstar_input_root(ap)
                        _al = _iter_vyvar_fits_under(_ms_root) if _ms_root is not None else []
                        if _al:
                            _cm = pipeline.db.get_combined_metadata(_al[0], int(_draft_ms))
                            if _cm.get("saturate_adu") is not None:
                                _equip_sat = _cm["saturate_adu"]
                    except (sqlite3.Error, TypeError, ValueError, KeyError) as exc:
                        log_event(
                            f"WARNING: saturate_adu from combined metadata failed "
                            f"(draft {_draft_ms}): {exc}"
                        )
                _plan_mc = st.session_state.get("vyvar_last_import_plan")
                _md_mc = (
                    Path(_plan_mc.dark_master)
                    if _plan_mc and getattr(_plan_mc, "dark_master", None)
                    else None
                )
                if _md_mc is not None and not _md_mc.exists():
                    _md_mc = None
                _md_job = pending.get("master_dark_path")
                if _md_job:
                    try:
                        _mj = Path(str(_md_job).strip())
                        if _mj.is_file():
                            _md_mc = _mj
                    except Exception:  # noqa: BLE001
                        # EXC-0004: T3 -- UI diagnostic/plot only (if _mj.is_file(): / _md_mc = _mj / except Exception:  # noqa: ... (EXCEPT-BULK 2026-07-08)
                        pass
                _hint_ra_j = pending.get("inject_pointing_ra_deg")
                _hint_de_j = pending.get("inject_pointing_dec_deg")
                _hint_ra: float | None = None
                _hint_de: float | None = None
                try:
                    if _hint_ra_j is not None and math.isfinite(float(_hint_ra_j)):
                        _hint_ra = float(_hint_ra_j)
                    if _hint_de_j is not None and math.isfinite(float(_hint_de_j)):
                        _hint_de = float(_hint_de_j)
                except (TypeError, ValueError):
                    pass
                outp = generate_masterstar_and_catalog(
                    archive_path=ap,
                    max_catalog_rows=int(pending.get("max_catalog_rows", 12000)),
                    astrometry_api_key=(str(pending.get("astrometry_api_key", "")).strip() or None),
                    platesolve_backend=str(pending.get("platesolve_backend", "vyvar")),
                    plate_solve_fov_deg=float(
                        pending.get("plate_solve_fov_deg", 1.0)
                    ),
                    catalog_match_max_sep_arcsec=float(
                        pending.get("catalog_match_max_sep_arcsec", 25.0)
                    ),
                    saturate_level_fraction=float(pending.get("saturate_level_fraction", 0.999)),
                    n_comparison_stars=int(pending.get("n_comparison_stars", 0)),
                    faintest_mag_limit=(
                        None
                        if pending.get("faintest_mag_limit") is None
                        else float(pending["faintest_mag_limit"])
                    ),
                    dao_threshold_sigma=float(
                        pending.get(
                            "dao_threshold_sigma",
                            st.session_state.get("dao_threshold_sigma", 3.5),
                        )
                    ),
                    equipment_saturate_adu=_equip_sat,
                    catalog_local_gaia_only=True,
                    app_config=_cfg_run,
                    equipment_id=int(_peq_ms) if _peq_ms is not None else None,
                    draft_id=int(_draft_ms) if _draft_ms is not None else None,
                    masterstar_candidate_paths=pending.get("masterstar_candidate_paths"),
                    masterstar_selection_pct=_ms_pct_mc_f,
                    setup_name=(
                        str(pending.get("setup_name")).strip()
                        if pending.get("setup_name") is not None
                        else None
                    ),
                    master_dark_path=_md_mc,
                    masterstar_fits_only=bool(_ms_fits_only),
                    masterstar_skip_build=bool(pending.get("masterstar_skip_build")),
                    masterstar_platesolve_only=False,
                    hint_ra_deg=_hint_ra,
                    hint_dec_deg=_hint_de,
                )
                if isinstance(outp, dict):
                    outp = dict(outp)
                    outp["job_kind"] = str(pending.get("kind") or "masterstar_catalog_only")
                st.session_state["vyvar_last_job_output"] = outp
                st.session_state["vyvar_last_job_summary"] = {
                    "kind": str(pending.get("kind") or "masterstar_catalog_only"),
                    "masterstar_fits": str(outp.get("masterstar_fits", "")),
                    "catalog_matched": int(outp.get("catalog_matched", 0)),
                }
                _msf = str(outp.get("masterstar_fits", "") or "").strip()
                if _msf:
                    st.session_state["vyvar_db_masterstar_path"] = _msf
                st.session_state["vyvar_masterstar_qa_force_refresh"] = True
                if _ms_fits_only:
                    log_event(
                        f"[OK] MASTERSTAR FITS dokoncene [{pending.get('masterstar_ui_action', '?')}]: {_msf or outp.get('masterstar_fits')} "
                        "(plate-solve a katalog preskocene)."
                    )
                elif _ms_skip_build and not _ms_fits_only:
                    log_event(
                        f"[OK] MASTERSTAR plate-solve + katalog hotove: {_msf or outp.get('masterstar_fits')} "
                        f"(matched={int(outp.get('catalog_matched', 0) or 0)})."
                    )

            elif pending.get("kind") == "run_epsf":
                from psf_photometry import build_epsf_model

                from epsf_psf_merge import run_epsf_psf_merge_job

                _ms_fits = Path(str(pending.get("masterstar_fits_path", "")).strip())
                _ms_csv = Path(str(pending.get("masterstars_csv_path", "")).strip())
                _frames_root = Path(str(pending.get("per_frame_csv_dir", "")).strip())
                _ps_dir = Path(str(pending.get("platesolve_dir", "")).strip())
                _draft_epsf = pending.get("draft_id")
                if _draft_epsf is None:
                    _draft_epsf = st.session_state.get("vyvar_last_draft_id")
                if _draft_epsf is None:
                    raise FileNotFoundError("Missing draft_id for RUN ePSF job.")
                _draft_epsf = int(_draft_epsf)

                if not _ms_fits.is_file():
                    raise FileNotFoundError(f"MASTERSTAR.fits not found: {_ms_fits}")
                if not _ms_csv.is_file():
                    raise FileNotFoundError(f"masterstars_full_match.csv not found: {_ms_csv}")
                if not _frames_root.is_dir():
                    raise FileNotFoundError(f"Aligned frames dir not found: {_frames_root}")
                if not _ps_dir.is_dir():
                    raise FileNotFoundError(f"Platesolve dir not found: {_ps_dir}")

                _cfg_epsf = cfg
                if not bool(getattr(_cfg_epsf, "psf_photometry_enabled", False)):
                    log_event(
                        "[ePSF job] warning: psf_photometry_enabled=False - PSF columns will be empty"
                    )

                log_event("[ePSF job] Building ePSF model...")
                _epsf_path = build_epsf_model(
                    masterstar_fits_path=_ms_fits,
                    masterstars_csv_path=_ms_csv,
                    db=pipeline.db,
                    draft_id=_draft_epsf,
                )
                log_event(f"[ePSF job] Model built: {_epsf_path.name}; re-exporting per-frame catalogs...")

                _dao_fwhm = float(
                    pending.get(
                        "dao_fwhm_px",
                        getattr(pipeline.config, "sips_dao_fwhm_px", 3.7),
                    )
                )
                _dao_sigma = float(
                    pending.get(
                        "dao_threshold_sigma",
                        getattr(pipeline.config, "sips_dao_threshold_sigma", 3.5),
                    )
                )
                _peq = pending.get("equipment_id")
                per_cat = run_epsf_psf_merge_job(
                    frames_root=_frames_root,
                    platesolve_dir=_ps_dir,
                    app_config=_cfg_epsf,
                    draft_id=_draft_epsf,
                    equipment_id=int(_peq) if _peq is not None else None,
                    progress_cb=_cb,
                )
                if isinstance(per_cat.get("epsf_job_summary"), dict):
                    _ejs = per_cat["epsf_job_summary"]
                    log_event(
                        f"[ePSF job] frame accounting: "
                        f"{_ejs.get('frames_with_zero_ok', '?')}/{_ejs.get('frames_total', '?')} "
                        f"zero-ok frames; policy={_ejs.get('inv_psf_frame_01_policy', '?')}"
                    )
                _n_proc = len(list(_frames_root.glob("proc_*.csv")))
                _written = int(per_cat.get("written", 0) or 0)
                _out_epsf = {
                    "job_kind": "run_epsf",
                    "status": "ok",
                    "epsf_path": str(_epsf_path),
                    "setup_name": str(pending.get("setup_name") or ""),
                    "frames_written": _written,
                    "proc_csv_count": _n_proc,
                    "message": (
                        f"ePSF model built; {_written} frame catalog(s) written "
                        f"({_n_proc} proc_*.csv in {_frames_root.name})"
                    ),
                }
                st.session_state["vyvar_last_job_output"] = _out_epsf
                st.session_state["vyvar_last_job_summary"] = {
                    "kind": "run_epsf",
                    "epsf_path": str(_epsf_path),
                    "frames_written": _written,
                }
                log_event(f"[ePSF job] Done - {_out_epsf['message']}")

            else:
                raise ValueError("Unknown job kind.")

            if stt is not None:
                stt.update(label="Done.", state="complete")
            out = st.session_state.get("vyvar_last_job_output")
            if isinstance(out, dict) and not out.get("error"):
                last_job_snapshot(out)
                if pending.get("kind") == "make_masterstar":
                    _vyvar_try_save_infolog_to_disk(cfg)
            st.session_state["vyvar_footer_state"] = {
                "running": False,
                "process": str(pending.get("label") or pending.get("kind") or "job"),
                "status_detail": "Done.",
                "pct": 100,
                "current_file": "",
                "step": "",
            }
            if footer_placeholder is not None:
                _vyvar_render_fixed_footer_into(footer_placeholder)
        except DraftTechnicalMetadataError as exc:
            _err_msg = str(exc).strip()
            if stt is not None:
                stt.update(label=f"Failed: {_err_msg}", state="error")
            st.error(_err_msg)
            st.session_state["vyvar_last_job_output"] = {
                "error": _err_msg,
                "error_type": type(exc).__name__,
            }
            log_event(f"Job zlyhal: {_err_msg} [{type(exc).__name__}]")
            st.session_state["vyvar_footer_state"] = {
                "running": False,
                "process": str(pending.get("label") or pending.get("kind") or "job"),
                "status_detail": f"Failed: {_err_msg}",
                "pct": None,
                "current_file": "",
                "step": "",
            }
            if footer_placeholder is not None:
                _vyvar_render_fixed_footer_into(footer_placeholder)
        except Exception as exc:  # noqa: BLE001
            _em = str(exc).strip()
            _err_msg = _em if _em else f"{type(exc).__name__} (no message)"
            if stt is not None:
                stt.update(label=f"Failed: {_err_msg}", state="error")
            st.session_state["vyvar_last_job_output"] = {
                "error": _err_msg,
                "error_type": type(exc).__name__,
            }
            log_event(f"Job zlyhal: {_err_msg} [{type(exc).__name__}]")
            st.session_state["vyvar_footer_state"] = {
                "running": False,
                "process": str(pending.get("label") or pending.get("kind") or "job"),
                "status_detail": f"Failed: {_err_msg}",
                "pct": None,
                "current_file": "",
                "step": "",
            }
            if footer_placeholder is not None:
                _vyvar_render_fixed_footer_into(footer_placeholder)

    # Clear pending job and rerun into the normal UI view.
    st.session_state.pop("vyvar_pending_job", None)
    st.rerun()
    return True


def render_live_view(
    pipeline: AstroPipeline,
    cfg: AppConfig,
    *,
    footer_placeholder: Any | None = None,
) -> None:
    _sess_key = "vyvar_varstrem_session_id"
    if _sess_key not in st.session_state:
        st.session_state[_sess_key] = generate_session_id()

    st.subheader("VARSTREM")
    st.caption(f"Photometry mode: **{getattr(cfg, 'photometry_mode', 'both')}**")
    # TODO: extract _render_fits_qa_tab - heavy st.session_state / pending-job wiring; defer Phase 2.
    st.write(f"Active session: `{st.session_state[_sess_key]}`")

    st.markdown("---")
    st.subheader("Session Upload Automation")

    dark_validity_days = int(cfg.masterdark_validity_days)
    flat_validity_days = int(cfg.masterflat_validity_days)

    equipments = pipeline.db.get_equipments(active_only=True)
    telescopes = pipeline.db.get_telescopes(active_only=True)
    equipment_options = {
        f"{item['ID']}: {item['CAMERANAME']} ({item['ALIAS']})": int(item["ID"]) for item in equipments
    }
    telescope_options = {
        f"{item['ID']}: {item['TELESCOPENAME']} ({item['ALIAS']})": int(item["ID"]) for item in telescopes
    }
    eq_labels = list(equipment_options.keys())
    tel_labels = list(telescope_options.keys())

    # Phase 2: pre-select the IS_DEFAULT equipment/telescope on first open (explicit
    # user marker, not a silent id=1 fallback). Phase 3 auto-detect may override these.
    def _label_for_id(_options: dict[str, int], _target: int | None) -> str | None:
        if _target is None:
            return None
        for _lbl, _id in _options.items():
            if int(_id) == int(_target):
                return _lbl
        return None

    _def_eq_lbl = _label_for_id(equipment_options, pipeline.db.get_default_id("EQUIPMENTS"))
    _def_tel_lbl = _label_for_id(telescope_options, pipeline.db.get_default_id("TELESCOPE"))
    if _def_eq_lbl and "vyvar_varstrem_equipment" not in st.session_state:
        st.session_state["vyvar_varstrem_equipment"] = _def_eq_lbl
    if _def_tel_lbl and "vyvar_varstrem_telescope" not in st.session_state:
        st.session_state["vyvar_varstrem_telescope"] = _def_tel_lbl

    with st.expander("[folder] Define source data", expanded=True):
        source_root = st.text_input(
            "Source Directory",
            value=str(cfg.archive_root),
            help="Example: USB session root (any structure; will be scanned recursively).",
        )
        st.caption(
            "To find masters in the Calibration Library, choose a **set** (camera + telescope). "
            "Matching or generic (no set) library entries are used per DB rules."
        )

        # Phase 3: fingerprint optics + site from a sample FITS header and auto-fill the
        # selectors (overriding the IS_DEFAULT baseline). The user can still override.
        if st.button(
            "[test] Auto-detect optics from FITS",
            help="Reads a sample light frame and matches camera/telescope/location by "
            "INSTRUME, sensor dimensions, GAIN, FOCALLEN/APTDIA and SITELAT/LONG.",
        ):
            from optics_autodetect import autodetect_from_source

            _act = pipeline.db.sql_expr_active_is_true("ACTIVE")
            _eq_rows = [
                dict(r)
                for r in pipeline.db.conn.execute(
                    f"SELECT ID,CAMERANAME,ALIAS,SENSORTYPE,SENSORSIZE,PIXELSIZE,GAIN_ADU "
                    f"FROM EQUIPMENTS WHERE {_act};"
                )
            ]
            _tel_rows = [
                dict(r)
                for r in pipeline.db.conn.execute(
                    f"SELECT ID,TELESCOPENAME,ALIAS,DIAMETER,FOCAL FROM TELESCOPE WHERE {_act};"
                )
            ]
            _loc_rows = get_observer_locations(str(cfg.database_path), active_only=True)
            _rep = autodetect_from_source(
                source_root,
                equipments=_eq_rows,
                telescopes=_tel_rows,
                locations=_loc_rows,
            )
            if _rep.header_path is None:
                st.warning("Auto-detect: no readable light FITS found under the source directory.")
            else:
                if _rep.equipment.ok and _rep.equipment.label in equipment_options:
                    st.session_state["vyvar_varstrem_equipment"] = _rep.equipment.label
                if _rep.telescope.ok and _rep.telescope.label in telescope_options:
                    st.session_state["vyvar_varstrem_telescope"] = _rep.telescope.label
                if _rep.location.ok:
                    _loc_lbls = {f"{int(r['id'])}: {r['name']}" for r in _loc_rows}
                    if _rep.location.label in _loc_lbls:
                        st.session_state["vyvar_varstrem_location"] = _rep.location.label
                st.session_state["vyvar_autodetect_report"] = _rep
            st.rerun()

        col_scan_eq, col_scan_tel = st.columns(2)
        with col_scan_eq:
            eq_placeholder = "(no camera in DB)" if not eq_labels else eq_labels[0]
            import_equipment_label = st.selectbox(
                "Equipment (library)",
                options=eq_labels if eq_labels else [eq_placeholder],
                key="vyvar_varstrem_equipment",
            )
        with col_scan_tel:
            tel_placeholder = "(no telescope in DB)" if not tel_labels else tel_labels[0]
            import_telescope_label = st.selectbox(
                "Telescope (library)",
                options=tel_labels if tel_labels else [tel_placeholder],
                key="vyvar_varstrem_telescope",
            )
        locations = get_observer_locations(str(cfg.database_path), active_only=True)
        location_options = {
            f"{item['id']}: {item['name']}": int(item["id"]) for item in locations
        }
        loc_labels = list(location_options.keys())
        loc_placeholder = "No locations defined"
        if loc_labels:
            # Phase 2: pre-select the IS_DEFAULT location; fall back to the config
            # location only if no default is marked. Phase 3 may override from headers.
            _loc_default_id = next(
                (int(item["id"]) for item in locations if int(item.get("is_default", 0) or 0) == 1),
                int(cfg.observer_location_id),
            )
            _loc_default_idx = 0
            for _loc_i, _loc_lbl in enumerate(loc_labels):
                if location_options[_loc_lbl] == _loc_default_id:
                    _loc_default_idx = _loc_i
                    break
            if "vyvar_varstrem_location" not in st.session_state:
                st.session_state["vyvar_varstrem_location"] = loc_labels[_loc_default_idx]
        import_location_label = st.selectbox(
            "Location (observatory)",
            options=loc_labels if loc_labels else [loc_placeholder],
            key="vyvar_varstrem_location",
            disabled=not loc_labels,
            help="Observer site for BJD, airmass, and lunar context (saved to config.json).",
        )

        # Phase 3: show the last auto-detect result (match + confidence + evidence).
        _ad_rep = st.session_state.get("vyvar_autodetect_report")
        if _ad_rep is not None:
            _band_icon = {"high": "[OK]", "medium": "[yellow]", "low": "[orange]", "none": "o"}

            def _ad_line(_title: str, _det: Any) -> str:
                _ic = _band_icon.get(_det.band(), "o")
                if _det.matched_id is None:
                    return f"{_ic} **{_title}:** no confident match - set manually (poor-FITS prompt)."
                _ev = "; ".join(_det.reasons) if _det.reasons else "-"
                return (
                    f"{_ic} **{_title}:** {_det.label} "
                    f"(confidence {_det.confidence:.0%}, {_det.band()}) - {_ev}"
                )

            with st.container(border=True):
                st.caption(f"[search] Auto-detect from `{Path(_ad_rep.header_path).name}`")
                st.markdown(_ad_line("Equipment", _ad_rep.equipment))
                st.markdown(_ad_line("Telescope", _ad_rep.telescope))
                st.markdown(_ad_line("Location", _ad_rep.location))
                st.caption(
                    "**high** confidence -> auto-filled (overrides default); "
                    "**medium** -> pre-filled but **unconfirmed - verify**; "
                    "**low/none** -> default kept and surfaced below for confirmation. "
                    "You can override any selector above."
                )

            # Phase 4: poor-FITS prompt - surface ONLY the unresolved gaps, pre-filled
            # from the current (default) selection. The resolver then runs downstream.
            if _ad_rep.unresolved:
                _prefill = {
                    "Equipment (camera)": import_equipment_label,
                    "Telescope": import_telescope_label,
                    "Observer site": import_location_label,
                    "Pointing (RA/Dec)": "blind plate-solve / set draft center after import",
                }
                with st.container(border=True):
                    st.warning(
                        f"! Poor FITS: {len(_ad_rep.unresolved)} field(s) are not in the header "
                        "and were not auto-detected. Confirm or override the pre-filled defaults "
                        "in the selectors above before running."
                    )
                    for _gap in _ad_rep.unresolved:
                        _fld = _gap.get("field", "")
                        st.markdown(
                            f"- **{_fld}** -> pre-filled: `{_prefill.get(_fld, _gap.get('fallback', '-'))}`  \n"
                            f"  <small>{_gap.get('detail', '')}</small>",
                            unsafe_allow_html=True,
                        )
                    st.caption(
                        "These pre-fills feed the import; the unified resolver "
                        "(header/solve -> DB -> config, site = draft -> header -> flagged config) "
                        "then applies them per draft."
                    )

        if loc_labels:
            _sel_loc_id = int(location_options[import_location_label])
            # CONFIG-WRITE-GUARD: persist ONLY on a genuine user change of the selectbox, never as a
            # render side-effect. On first render the selectbox defaults to the DB IS_DEFAULT location,
            # which can differ from config.json (that mismatch used to auto-rewrite config.json on load).
            # Baseline the tracker to the current selection on first render so a plain render never saves.
            _loc_tracker = "vyvar_varstrem_location_persisted_id"
            if _loc_tracker not in st.session_state:
                st.session_state[_loc_tracker] = _sel_loc_id
            elif _sel_loc_id != int(st.session_state[_loc_tracker]):
                _loc_row = get_observer_location_by_id(str(cfg.database_path), _sel_loc_id)
                if _loc_row is not None:
                    cfg.observer_location_id = int(_loc_row["id"])
                    cfg.observer_lat = float(_loc_row["lat"])
                    cfg.observer_lon = float(_loc_row["lon"])
                    cfg.observer_alt_m = float(_loc_row["alt_m"])
                    cfg.observer_location_name = str(_loc_row.get("name") or "")
                    with ui_config_persist():
                        save_config_json(cfg.data_root, cfg.to_json())
                    LOGGER.info(
                        f"Observer location set: {cfg.observer_location_name} "
                        f"(lat={cfg.observer_lat}, lon={cfg.observer_lon}, alt={cfg.observer_alt_m}m)"
                    )
                st.session_state[_loc_tracker] = _sel_loc_id
        try:
            _ui_optics = parse_ui_optics_from_labels(
                equipment_label=import_equipment_label,
                telescope_label=import_telescope_label,
                equipment_options=equipment_options,
                telescope_options=telescope_options,
                eq_labels=eq_labels,
                tel_labels=tel_labels,
                db=pipeline.db,
            )
            sync_optics_session(_ui_optics)
            import_equipment_id = _ui_optics.equipment_id
            import_telescope_id = _ui_optics.telescope_id
        except ValueError as _opt_exc:
            st.error(str(_opt_exc))
            _ui_optics = optics_from_session()
            if _ui_optics is None:
                import_equipment_id = 0
                import_telescope_id = 0
            else:
                import_equipment_id = _ui_optics.equipment_id
                import_telescope_id = _ui_optics.telescope_id

        col_scan, col_run = st.columns([1, 2])
        with col_scan:
            if st.button("[search] Scan Source", type="primary"):
                if import_equipment_id <= 0 or import_telescope_id <= 0:
                    st.error("Vyberte platnu kameru a dalekohlad pred skenom.")
                else:
                    try:
                        plan = smart_scan_source(
                            source_root=source_root,
                            calibration_library_root=cfg.calibration_library_root,
                            masterdark_validity_days=dark_validity_days,
                            masterflat_validity_days=flat_validity_days,
                            db=pipeline.db,
                            id_equipments=import_equipment_id,
                            id_telescope=import_telescope_id,
                            calibration_master_ccd_temp_tolerance_c=cfg.calibration_master_ccd_temp_tolerance_c,
                        )
                        st.session_state["vyvar_smart_plan"] = plan
                        sync_optics_session(
                            VyvarOpticsSelection(import_equipment_id, import_telescope_id)
                        )
                        log_active_optics(
                            pipeline.db,
                            VyvarOpticsSelection(import_equipment_id, import_telescope_id),
                            context="Scan Source",
                        )
                        st.session_state.pop("vyvar_post_cal_archive_path", None)
                        st.session_state.pop("vyvar_post_cal_plan_source", None)
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Scan failed: {exc}")
                        st.session_state.pop("vyvar_smart_plan", None)
        with col_run:
            _sr_rv = str(source_root).strip()
            _gaia_ok = _gaia_db_ok_for_masterstar(str(getattr(cfg, "gaia_db_path", "") or ""))
            run_vyvar_disabled = (
                not _sr_rv
                or not Path(_sr_rv).is_dir()
                or not _gaia_ok
                or import_equipment_id <= 0
                or import_telescope_id <= 0
            )
            btn_cal, btn_nc = st.columns(2)
            with btn_cal:
                run_vyvar_clicked = st.button(
                    "! RUN VYVAR",
                    type="primary",
                    disabled=run_vyvar_disabled,
                    help="One click: scan -> import -> calibrate -> analyze -> MASTERSTAR -> Phase 0+1+2A",
                )
            with btn_nc:
                run_vyvar_nc_clicked = st.button(
                    "[folder] RUN VYVAR (non-cal)",
                    type="secondary",
                    disabled=run_vyvar_disabled,
                    help=(
                        "Skips bias/dark/flat. Treats source frames as already calibrated (e.g. "
                        "Telescope Live exports). Proceeds with alignment -> masterstar -> photometry."
                    ),
                )
            if run_vyvar_clicked or run_vyvar_nc_clicked:
                from run_lifecycle import run_callable_with_exit_log  # noqa: PLC0415

                _pre_cal = bool(run_vyvar_nc_clicked)
                _dao_fwhm_default = float(
                    max(1.0, min(8.0, float(getattr(cfg, "sips_dao_fwhm_px", 2.5))))
                )
                _dao_sigma_default = float(
                    max(1.0, min(10.0, float(getattr(cfg, "sips_dao_threshold_sigma", 3.5))))
                )
                st.session_state["dao_fwhm_px"] = _dao_fwhm_default
                st.session_state["dao_threshold_sigma"] = _dao_sigma_default
                _status_label = (
                    "[folder] RUN VYVAR (non-cal) running..." if _pre_cal else "! RUN VYVAR running..."
                )

                def _run_vyvar_body() -> bool:
                    with st.status(_status_label, expanded=True) as _rv_status:
                        _ok = _run_vyvar_full_pipeline(
                            pipeline=pipeline,
                            cfg=cfg,
                            source_root=_sr_rv,
                            import_equipment_id=int(import_equipment_id),
                            import_telescope_id=int(import_telescope_id),
                            dark_validity_days=int(dark_validity_days),
                            flat_validity_days=int(flat_validity_days),
                            plate_fov_ui=float(cfg.plate_solve_fov_deg),
                            dao_fwhm_default=_dao_fwhm_default,
                            dao_sigma_default=_dao_sigma_default,
                            cat_match_arc=2.0,
                            max_cat_rows=12000,
                            max_extra_ps=0,
                            min_stars=100,
                            max_stars=500,
                            max_ctrl=int(cfg.alignment_max_control_points),
                            sat_level=0.95,
                            footer_placeholder=footer_placeholder,
                            pre_calibrated_mode=_pre_cal,
                        )
                        if _ok:
                            _done_label = (
                                "[OK] RUN VYVAR (non-cal) complete"
                                if _pre_cal
                                else "[OK] RUN VYVAR complete"
                            )
                            _rv_status.update(
                                label=_done_label,
                                state="complete",
                                expanded=False,
                            )
                        else:
                            _err_label = (
                                "[X] RUN VYVAR (non-cal) - chyba"
                                if _pre_cal
                                else "[X] RUN VYVAR - chyba"
                            )
                            _rv_status.update(
                                label=_err_label,
                                state="error",
                                expanded=True,
                            )
                        return bool(_ok)

                ok = run_callable_with_exit_log(_run_vyvar_body, log_event)
                if ok:
                    st.success("Pipeline finished successfully. Review the results.")
                else:
                    st.error(
                        _vyvar_format_run_failure_message(
                            default="Pipeline stopped. See Infolog for error details."
                        )
                    )

    plan = st.session_state.get("vyvar_smart_plan")
    if plan:
        # Dashboard summary
        summary_df = pd.DataFrame(
            [
                {
                    "Type": row.type,
                    "Status": row.status,
                    "Count": row.count,
                    "Parameters": row.parameters,
                }
                for row in plan.scan_rows
            ]
        )
        if getattr(plan, "detected_filters", None) is not None:
            summary_df["Detected Filters"] = ", ".join(plan.detected_filters)
        st.table(summary_df)

        # UI report: show real folder paths and detected types
        try:
            scan_df = scan_usb_folder(plan.source_root)
            if not scan_df.empty:
                st.markdown("---")
                st.subheader("Scan Source (folders)")
                st.table(
                    scan_df[
                        [
                            "Folder Path",
                            "Type",
                            "File Count",
                            "Lights Count",
                            "Darks Count",
                            "Flats Count",
                            "Unknown Count",
                            "Detected Filters",
                            "Params",
                        ]
                    ]
                )
        except Exception:  # noqa: BLE001
            # EXC-0005: T3 -- UI diagnostic/plot only (] / ) / except Exception:  # noqa: BLE001 / pass) (EXCEPT-BULK 2026-07-08)
            pass

        # Mandatory validation: at least one Light found in the whole tree
        lights_bad = any(r.type == "Lights" and r.status in ("missing", "empty") for r in plan.scan_rows)
        if lights_bad:
            st.error(
                "No light frames found (no lights/Lights folder and no FITS with IMAGETYP "
                "light/object/science in the root directory). Import cancelled."
            )

        st.markdown("---")
        _banner_draft_id = (
            int(st.session_state["vyvar_last_draft_id"])
            if st.session_state.get("vyvar_last_draft_id") is not None
            else None
        )
        from ui_finalization import render_known_field_banner

        render_known_field_banner(pipeline=pipeline, draft_id=_banner_draft_id)
        ui_calibration.render_calibration_equipment_header(
            pipeline.db,
            draft_id=int(st.session_state["vyvar_last_draft_id"])
            if st.session_state.get("vyvar_last_draft_id") is not None
            else None,
            equipment_id=int(import_equipment_id),
            telescope_id=int(import_telescope_id),
        )

        raw_present = any(r.type in ("Dark", "Flat", "Darks", "Flats") and r.status == "raw" for r in plan.scan_rows)
        if raw_present:
            st.info(
                "Source contains **raw** dark or flat frames. Prepare and verify master dark/flat "
                "in **Calibration Library** (and generate there if needed), then **Scan Source** again."
            )

        if plan.warnings:
            st.warning("Warnings:\n" + "\n".join(plan.warnings))

        ui_calibration.render_calibration_library_flat_warnings(pipeline.db, plan)

        ogroups = getattr(plan, "observation_groups", None) or {}
        _post_ap = st.session_state.get("vyvar_post_cal_archive_path")
        _post_src = st.session_state.get("vyvar_post_cal_plan_source")
        _use_done = bool(
            _post_ap
            and _post_src
            and str(getattr(plan, "source_root", "") or "").strip() == str(_post_src).strip()
            and Path(str(_post_ap)).is_dir()
        )
        st.markdown("---")
        st.subheader("Multi-observation (Filter + Exp)")
        st.caption("Calibration and plate-solve status by filter + exposure time (all binnings in the group).")
        _obs_df = ui_calibration.build_multi_observation_status_dataframe(
            plan,
            archive_path=Path(_post_ap) if _use_done else None,
            cal_phase="done" if _use_done else "preview",
        )
        st.dataframe(_obs_df, width="stretch", hide_index=True)

        if ogroups:
            with st.expander("Calibration masters and binning mode", expanded=False):
                st.caption(
                    "**Binning Mode** column: e.g. Resampling 1x1 -> 2x2 when light is 2x2 and master is 1x1 in the library."
                )
                st.dataframe(
                    ui_calibration.build_master_calibration_files_dataframe(plan),
                    width="stretch",
                    hide_index=True,
                )
            with st.expander("Technical group overview (binning . scale)", expanded=False):
                _ogr = []
                for gk, g in sorted(ogroups.items(), key=lambda x: x[0]):
                    _mf = (getattr(plan, "masterflat_by_obs_key", None) or {}).get(gk)
                    _md = (getattr(plan, "dark_master_by_obs_key", None) or {}).get(gk)
                    _ogr.append(
                        {
                            "Group": gk,
                            "Filter": g.get("filter"),
                            "Exp (s)": g.get("exposure_s"),
                            "Binning": g.get("binning"),
                            "Frames": len(g.get("light_paths") or []),
                            "Master flat": "yes" if _mf else "no",
                            "Master dark": "yes" if _md else "no",
                            " arcsec/px": g.get("plate_scale_arcsec_per_px"),
                        }
                    )
                st.dataframe(pd.DataFrame(_ogr), width="stretch", hide_index=True)

        _fb_prompts = getattr(plan, "flat_fallback_prompts", None) or []
        if _fb_prompts:
            with st.expander("Missing Master Flat - choose substitute", expanded=True):
                st.caption(
                    "For groups without a flat you can use a master flat from another filter group "
                    "(same exposure time and binning), or leave skipped (frames go to non_calibrated)."
                )
                for p in _fb_prompts:
                    gk = str(p.get("group_key") or "")
                    alts = list(p.get("alternatives") or [])
                    opts: list[str] = ["__skip__"] + [str(a) for a in alts]
                    og = ogroups

                    def _fmt(v: str, og=og) -> str:
                        if v == "__skip__":
                            return "Skip (non_calibrated for this group)"
                        gg = og.get(v, {})
                        return f"Use flat from filter {gg.get('filter', '?')} ({v})"

                    st.selectbox(
                        str(p.get("message_sk") or "Missing Master Flat."),
                        options=opts,
                        format_func=_fmt,
                        key=f"vyvar_flatfb_{gk}",
                    )

        missing = list(getattr(plan, "missing_flat_filters", []) or [])
        if missing:
            st.warning("Some filters are missing MasterFlat: " + ", ".join(missing))
            mode = st.radio(
                "When MasterFlat is missing",
                options=[
                    "Import missing filters as Draft (non_calibrated)",
                    "Select MasterFlat manually",
                ],
                index=0,
            )
            manual_map: dict[str, str] = {}
            if mode == "Select MasterFlat manually":
                for flt in missing:
                    manual_map[flt] = st.text_input(
                        f"MasterFlat path for filter '{flt}'",
                        value="",
                        help="Provide a full path to a MasterFlat FITS file.",
                    )
                st.session_state["vyvar_manual_flat_map"] = manual_map
                st.caption("If any path is empty/non-existent, Import will be disabled.")
            else:
                st.session_state.pop("vyvar_manual_flat_map", None)

        # Import button enabled only if lights ok
        import_disabled = lights_bad
        manual_flat_map = st.session_state.get("vyvar_manual_flat_map") or {}
        if getattr(plan, "missing_flat_filters", None):
            if manual_flat_map:
                # validate manual paths
                for flt in plan.missing_flat_filters:
                    p = manual_flat_map.get(flt, "")
                    if not p or not Path(p).exists():
                        import_disabled = True
            # If missing flats and no manual provided -> allow import (will draft those filters)
        label = "[rocket] Create Archive & Do Calibration (Quick Look Draft)" if plan.quick_look else "[rocket] Create Archive & Do Calibration"
        if st.button(label, type="primary", disabled=import_disabled):
            try:
                # Apply manual overrides if provided
                if manual_flat_map:
                    for flt, pth in manual_flat_map.items():
                        if pth and Path(pth).exists():
                            plan.masterflat_by_filter[flt] = pth
                _vyvar_apply_smart_plan_flat_fallbacks(plan)
                _vyvar_footer_set(
                    footer_placeholder,
                    running=True,
                    process="Archive import",
                    status_detail="Writing draft and copying files...",
                    pct=0,
                )
                result = smart_import_session(
                    plan=plan,
                    pipeline=pipeline,
                    id_equipment=import_equipment_id,
                    id_telescope=import_telescope_id,
                    cfg=cfg,
                )
                st.session_state["vyvar_last_import_equipment_id"] = int(import_equipment_id)
                st.session_state["vyvar_last_import_result"] = result
                st.session_state["vyvar_last_import_plan"] = plan
                if getattr(result, "draft_id", None) is not None:
                    st.session_state["vyvar_last_draft_id"] = int(result.draft_id)
                log_event(
                    f"Import hotovy - draft {result.draft_id}, archiv {result.archive_path}"
                )
                try:
                    _sat_eq = pipeline.db.get_equipment_saturation_adu(int(import_equipment_id))
                    log_event(
                        f"Equipment ID {import_equipment_id}: SATURATE_ADU v DB = "
                        f"{_sat_eq if _sat_eq is not None else '(NULL - pri MASTERSTAR katalogu: hlavicka -> BITPIX -> Settings fallback)'}"
                    )
                except (sqlite3.Error, TypeError, ValueError, KeyError) as exc:
                    log_event(
                        f"WARNING: equipment SATURATE_ADU DB lookup failed "
                        f"(equipment_id={import_equipment_id}): {exc}"
                    )
                if result.warnings:
                    for w in result.warnings:
                        log_event(f"Import varovani: {w}")

                # Immediately run calibration to create /calibrated (progress v paticke, nie samostatny bar)
                def _cal_progress(i: int, total: int, msg: str) -> None:
                    pct = int(round(100 * (i / max(total, 1))))
                    _vyvar_footer_set(
                        footer_placeholder,
                        running=True,
                        process="Calibrating lights -> /calibrated",
                        status_detail=msg,
                        pct=pct,
                        current_file=_vyvar_guess_filename_from_progress(msg),
                        step=f"{i} / {total}",
                    )

                _vyvar_footer_set(
                    footer_placeholder,
                    running=True,
                    process="Calibrating lights -> /calibrated",
                    status_detail="Import complete - applying dark/flat...",
                    pct=0,
                    step="",
                )
                with st.spinner("Calibration in progress - see footer; Multi-observation table updates when finished."):
                    md = Path(plan.dark_master) if getattr(plan, "dark_master", None) else None
                    mf_map: dict[str, Path | None] = {}
                    if getattr(plan, "masterflat_by_filter", None):
                        for k, v in (plan.masterflat_by_filter or {}).items():
                            mf_map[str(k)] = Path(v) if v else None
                    _did = getattr(result, "draft_id", None)
                    mf_obs: dict[str, Path | None] = {}
                    for k, v in (getattr(plan, "masterflat_by_obs_key", None) or {}).items():
                        mf_obs[str(k)] = Path(str(v)) if v else None
                    dm_obs: dict[str, Path | None] = {}
                    for k, v in (getattr(plan, "dark_master_by_obs_key", None) or {}).items():
                        dm_obs[str(k)] = Path(str(v)) if v else None
                    cal_out = pipeline.quick_calibrate_last_import(
                        archive_path=Path(result.archive_path),
                        master_dark_path=md if (md and md.exists()) else None,
                        masterflat_by_filter=mf_map,
                        progress_cb=_cal_progress,
                        equipment_id=int(import_equipment_id),
                        draft_id=int(_did) if _did is not None else None,
                        observation_id=getattr(result, "observation_id", None),
                        masterflat_by_obs_key=mf_obs or None,
                        master_dark_by_obs_key=dm_obs or None,
                    )
                st.session_state["vyvar_post_cal_archive_path"] = str(result.archive_path)
                st.session_state["vyvar_post_cal_plan_source"] = str(plan.source_root)
                _nproc = 0
                try:
                    for _sec in (cal_out.get("results") or {}).values():
                        if isinstance(_sec, dict):
                            _nproc += int(_sec.get("processed", 0) or 0)
                except Exception:  # noqa: BLE001
                    _nproc = 0
                _vyvar_footer_set(
                    footer_placeholder,
                    running=False,
                    process="Import + calibration",
                    status_detail=f"Done - frames processed (calibration): {_nproc}",
                    pct=100,
                    step="",
                )
                log_event("Kalibracia hotova - `/calibrated`")
                last_job_snapshot(cal_out)
                st.session_state["vyvar_status_calibrated"] = True
                _did_imp = getattr(result, "draft_id", None)
                if _did_imp is not None:
                    st.session_state["vyvar_pending_job"] = {
                        "kind": "analyze",
                        "label": "QC analysis (RAM calibration)...",
                        "archive_path": str(result.archive_path),
                        "draft_id": int(_did_imp),
                        "equipment_id": int(import_equipment_id),
                        "roundness_reject_above": float(st.session_state.get("max_roundness_error", 1.25)),
                    }
                    st.success("Import and calibration complete - starting QC analysis in RAM (Analyze)...")
                else:
                    st.success("Import and calibration complete.")
                    st.session_state["vyvar_status_analyzed"] = bool(
                        st.session_state.get("vyvar_status_analyzed", False)
                    )
                if result.warnings:
                    st.warning("Import warnings:\n" + "\n".join(result.warnings))
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                _vyvar_footer_set(
                    footer_placeholder,
                    running=False,
                    process="Import / calibration",
                    status_detail=f"Failed: {exc}",
                    pct=None,
                )
                st.error(f"Import failed: {exc}")
                st.exception(exc)

    # Calibration runs after import; QC RAM analyze is queued at end of the import handler.

    _sync_varstrem_session_state(pipeline)
    if _render_pending_job_dispatcher(
        pipeline=pipeline,
        cfg=cfg,
        footer_placeholder=footer_placeholder,
    ):
        return


def _iter_vyvar_fits_under(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    out: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.casefold() in {".fits", ".fit", ".fts"}:
            out.append(p)
    return sorted(out)


def render_infolog(cfg: AppConfig | None = None) -> None:
    st.subheader("Infolog")
    st.caption(
        "Detailed trace: `pipeline`, `importer` (INFO), and long job steps "
        "(QC / preprocessing / plate solve). After RUN VYVAR the **durable session log** "
        "(complete record incl. guard lines) is finalized under "
        "`draft_dir/infolog_YYYYMMDD_HHMMSS.txt`. The tab below shows the live ring buffer "
        "(last 8000 lines only). Buffer is lost after Streamlit restart."
    )
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("Clear Infolog", type="secondary", key="vyvar_infolog_clear"):
            clear_log()
            st.rerun()
    with col_b:
        if st.button("[save] Save Infolog to disk", key="vyvar_infolog_save_disk"):
            _cfg = cfg if cfg is not None else AppConfig()
            _draft_dir = resolve_draft_dir(
                draft_dir_override=st.session_state.get("vyvar_draft_dir_override"),
                draft_id=st.session_state.get("vyvar_last_draft_id"),
                archive_root=_cfg.archive_root,
            )
            if _draft_dir:
                saved = write_run_infolog(_draft_dir)
                if saved:
                    st.success(f"Saved: {saved}")
                    log_event(f"Infolog saved -> {saved}")
                else:
                    st.error("Save failed - check draft directory")
            else:
                st.warning("No draft directory selected")
    with col_c:
        tail = st.number_input(
            "Show last N lines",
            min_value=100,
            max_value=8000,
            value=2500,
            step=100,
            key="vyvar_infolog_tail",
        )
    lines = get_lines()
    if tail < len(lines):
        lines = lines[-int(tail) :]
    text = "\n".join(lines) if lines else "(no entries yet - start a job on VARSTREM)"
    st.code(text, language=None)


def _config_cache_fingerprint(cfg: AppConfig) -> str:
    """Bump Streamlit cache when config.json changes (Settings save)."""
    path = cfg.data_root / "config.json"
    try:
        st = path.stat()
        return f"{int(st.st_mtime_ns)}:{st.st_size}"
    except OSError:
        return "missing"


@st.cache_resource
def _cached_astro_pipeline(database_path: str, config_fingerprint: str) -> AstroPipeline:
    """One DB connection per Streamlit server process (avoids startup migration races).

    ``config_fingerprint`` is ``config.json`` mtime+size so a Settings save rebuilds
    ``AppConfig`` instead of reusing a stale cached pipeline.
    """
    _ = config_fingerprint
    cfg = AppConfig()
    return AstroPipeline(config=cfg)


def main() -> None:
    from vyvar_runtime import ensure_release_data_dir

    ensure_release_data_dir(Path(__file__).resolve().parent.parent)
    cfg = AppConfig()
    cfg.ensure_base_dirs()
    ensure_infolog_logging()
    pipeline = _cached_astro_pipeline(
        str(cfg.database_path.resolve()),
        _config_cache_fingerprint(cfg),
    )

    # Session guard for long-running pipelines (prevents auto blocks after reload).
    if "_current_session_id" not in st.session_state:
        st.session_state["_current_session_id"] = str(uuid.uuid4())[:8]

    st.set_page_config(
        page_title="VYVAR - Variable Star Processing",
        page_icon="*",
        layout="wide",
    )

    st.title("VYVAR Dashboard")

    vyvar_init_footer_state_if_missing()
    vyvar_footer_ph = st.empty()
    st.session_state["vyvar_ui_rerender_footer"] = lambda: _vyvar_render_fixed_footer_into(vyvar_footer_ph)
    _vyvar_render_fixed_footer_into(vyvar_footer_ph)

    page = st.sidebar.radio(
        "Navigation",
        options=[
            "Pipeline",
            "Calibration Library",
            "Database Explorer",
            "Settings",
        ],
        index=0,
    )

    if page == "Pipeline":
        st.subheader("Draft")
        st.caption(
            "Enter the absolute path to folder ``draft_XXXXXX`` (must contain ``platesolve/``) "
            "or draft number from the archive. Used in FITS QA, MASTERSTAR QA "
            "and Aperture Photometry tabs. Empty field + Load draft clears override. "
            "**DAO-STARS**, **Photometry**, and **Photometry - diagnostics** are under **Settings -> Tools**."
        )
        dcol1, dcol2, dcol3 = st.columns([4, 1, 1])
        with dcol1:
            draft_path_inp = st.text_input(
                "Path or draft number",
                key="vyvar_draft_path_field",
                placeholder=r"e.g. C:\...\Archive\Drafts\draft_000229 or 229",
            )
        with dcol2:
            apply_draft = st.button("Load draft", key="vyvar_draft_path_apply", type="primary")
        with dcol3:
            _cur = st.session_state.get("vyvar_last_draft_id")
            st.caption(f"ID: **{_cur}**" if _cur is not None else "ID: -")
        if apply_draft:
            s = (draft_path_inp or "").strip()
            if not s:
                st.session_state.pop("vyvar_draft_dir_override", None)
                _vyvar_reset_variability_session_state()
                st.info("Draft override cleared - using archive path from configuration.")
                st.rerun()
            else:
                ddir, parsed_id, err = resolve_draft_directory(
                    s, archive_root=Path(cfg.archive_root)
                )
                if err:
                    st.error(err)
                elif ddir is not None:
                    st.session_state["vyvar_draft_dir_override"] = str(ddir)
                    if parsed_id is not None:
                        st.session_state["vyvar_last_draft_id"] = int(parsed_id)
                    _vyvar_reset_variability_session_state()
                    st.success(f"Draft loaded: {ddir}")
                    st.rerun()
        ov = _vyvar_effective_draft_dir_override()
        if ov is not None:
            st.caption(f"Active override: `{ov}`")

        _draft_ov = ov
        tabs = st.tabs(
            [
                "VAR-STREM",
                "FITS QA",
                "MASTERSTAR QA",
                "Aperture Photometry",
                "[microscope] ePSF",
                "[search] Variability",
                "Infolog",
            ]
        )

        with tabs[0]:
            render_live_view(pipeline=pipeline, cfg=cfg, footer_placeholder=vyvar_footer_ph)
        with tabs[1]:
            ui_quality_dashboard.render_quality_dashboard(
                db=pipeline.db,
                draft_id=st.session_state.get("vyvar_last_draft_id"),
                archive_text="",
                cfg=cfg,
            )
        with tabs[2]:
            ui_masterstar_qa.render_masterstar_qa(
                cfg=cfg,
                draft_id=st.session_state.get("vyvar_last_draft_id"),
                pipeline=pipeline,
                draft_dir_override=_draft_ov,
            )
        with tabs[3]:
            render_aperture_photometry(
                cfg=cfg,
                draft_id=st.session_state.get("vyvar_last_draft_id"),
                pipeline=pipeline,
                draft_dir_override=_draft_ov,
            )
        with tabs[4]:
            from ui_epsf_dashboard import render_epsf_dashboard

            render_epsf_dashboard(
                draft_dir=_draft_ov,
                cfg=cfg,
                draft_id=st.session_state.get("vyvar_last_draft_id"),
            )
        with tabs[5]:
            render_variability_dashboard(
                pipeline,
                cfg,
                draft_id=st.session_state.get("vyvar_last_draft_id"),
                draft_dir_override=_draft_ov,
            )
        with tabs[6]:
            render_infolog(cfg)
    elif page == "Calibration Library":
        import ui_calibration_library as ui_calibration_library

        ui_calibration_library.render_calibration_library_dashboard(
            calibration_library_root=Path(cfg.calibration_library_root),
            dark_validity_days=int(cfg.masterdark_validity_days),
            flat_validity_days=int(cfg.masterflat_validity_days),
            db=pipeline.db,
        )
    elif page == "Database Explorer":
        ui_database_explorer.render_database_explorer(pipeline=pipeline)
    elif page == "Settings":
        import ui_settings

        ui_settings.render_settings_dashboard(
            cfg,
            pipeline,
            draft_dir_override=_vyvar_effective_draft_dir_override(),
        )

    _vyvar_render_fixed_footer_into(vyvar_footer_ph)


if __name__ == "__main__":
    main()

