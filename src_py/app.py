"""Streamlit entrypoint for live and archive views."""

from __future__ import annotations

import contextlib
import html
import logging
import math
import re
from pathlib import Path
from typing import Any

import streamlit as st

from config import AppConfig, resolve_detection_threshold_sigma
from database import (
    DraftTechnicalMetadataError,
    get_observer_location_by_id,
    get_observer_locations,
)
from run_preflight_log import write_run_preflight_error_log
from importer import smart_scan_source
from optics_selection import (
    VyvarOpticsSelection,
    log_active_optics,
    optics_from_session,
    parse_ui_optics_from_labels,
    resolve_working_optics,
    sync_optics_session,
)
from pipeline import AstroPipeline
import ui_calibration as ui_calibration
import ui_database_explorer as ui_database_explorer
import ui_masterstar_qa as ui_masterstar_qa
import ui_quality_dashboard as ui_quality_dashboard
import ui_components as ui_components
from platesolve_ui_paths import resolve_draft_directory
from ui_aperture_photometry import render_aperture_photometry
from ui_variability import render_variability_dashboard

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
    """Prefer the on-disk preflight log path when an early failure left no draft infolog."""
    log_path = st.session_state.get("vyvar_last_preflight_error_log")
    detail = st.session_state.get("vyvar_last_preflight_error_message")
    if log_path:
        msg = str(detail or "Pipeline preflight failed before the infolog of the processed field was opened.")
        return f"{msg} Preflight log: {log_path}"
    return default


def _vyvar_reset_variability_session_state() -> None:
    """On loading a new draft, clear the variability / TESS / PDF state in the session."""
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
    import_location_id: int,
    dark_validity_days: int,
    flat_validity_days: int,
    dao_sigma_default: float,
    cat_match_arc: float,
    max_cat_rows: int,
    min_stars: int,
    max_stars: int,
    footer_placeholder: Any | None = None,
    pre_calibrated_mode: bool = False,
    masters_from: str = "lights",
) -> bool:
    """RUN VYVAR - UI resolution then night_run.run_night_pipeline (INV-ONE-ENTRY-01)."""
    from night_run import NightRunParams
    from observer_location import (  # noqa: PLC0415
        apply_resolved_observer_location_to_config,
        resolve_observer_location_for_run,
    )

    _run_label = "RUN VYVAR (non-cal)" if pre_calibrated_mode else "RUN VYVAR"

    def _update(proces: str, stav: str = "Running...") -> None:
        LOGGER.info(f"[{_run_label}] > {proces}")
        if footer_placeholder is not None:
            _vyvar_footer_set(
                footer_placeholder,
                running=True,
                process=f"{_run_label} - {proces}",
                status_detail=str(stav)[:800],
                pct=None,
            )

    def _fail(name: str, exc: BaseException) -> bool:
        LOGGER.info(f"[{_run_label}] x Failure in step '{name}': {exc}")
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

    try:
        st.session_state.pop("vyvar_last_preflight_error_log", None)
        st.session_state.pop("vyvar_last_preflight_error_message", None)
        _root = Path(str(source_root).strip())
        if not _root.is_dir():
            return _fail("Scan Source + Import", RuntimeError(f"Invalid Source Directory: {_root}"))

        _run_optics = resolve_working_optics(
            pipeline.db,
            ui=VyvarOpticsSelection(
                int(import_equipment_id),
                int(import_telescope_id),
            ),
            context=_run_label,
        )
        sync_optics_session(_run_optics)
        log_active_optics(pipeline.db, _run_optics, context=f"{_run_label} (UI selection)")

        manual_flat_map = st.session_state.get("vyvar_manual_flat_map") or {}
        choices: dict[str, str] = {}
        for k, v in list(st.session_state.items()):
            ks = str(k)
            if ks.startswith("vyvar_flatfb_"):
                choices[ks[len("vyvar_flatfb_") :]] = v

        loc_id = int(import_location_id)
        if loc_id <= 0:
            raise ValueError("No observing site selected; select an observatory site before RUN.")
        # The site of this run (SETTINGS-2026-10: not written to config.json; the default site
        # is LOCATION.IS_DEFAULT in Database Explorer).
        _resolved_ui = resolve_observer_location_for_run(
            cfg.database_path,
            explicit_location_id=loc_id,
            cfg=cfg,
            source_hint="ui_selection",
        )
        apply_resolved_observer_location_to_config(cfg, _resolved_ui)

        params = NightRunParams(
            source_dir=_root,
            equipment_id=int(_run_optics.equipment_id),
            telescope_id=int(_run_optics.telescope_id),
            existing_pipeline=pipeline,
            optics=_run_optics,
            location_id=int(loc_id),
            location_source_hint="ui_selection",
            masterdark_validity_days=int(dark_validity_days),
            masterflat_validity_days=int(flat_validity_days),
            manual_flat_map=dict(manual_flat_map) if manual_flat_map else None,
            flat_fallback_choices=choices or None,
            apply_smart_plan_flat_fallbacks=not bool(pre_calibrated_mode),
            dao_threshold_sigma=float(dao_sigma_default),
            catalog_match_max_sep_arcsec=float(cat_match_arc),
            max_catalog_rows=int(max_cat_rows),
            min_detected_stars=int(min_stars),
            max_detected_stars=int(max_stars),
            pre_calibrated_mode=bool(pre_calibrated_mode),
            masters_from=str(masters_from or "lights"),
        )
        _update("Night pipeline")
        from run_job import launch_job  # noqa: PLC0415

        _job_dir = launch_job(params, cfg.data_root, label=_run_label)
        st.session_state["vyvar_active_job"] = {
            "job_dir": str(_job_dir),
            "label": _run_label,
            "equipment_id": int(_run_optics.equipment_id),
            "applied": False,
        }
        LOGGER.info(f"[{_run_label}] started as a separate process (job {_job_dir.name})")
        return True
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("[RUN VYVAR] unknown step (detail)")
        return _fail("unknown step", exc)


def _vyvar_apply_run_result(nr: Any, cfg: AppConfig, footer_placeholder: Any | None, _run_label: str) -> bool:
    """Write the session state from a finished RUN VYVAR job (same as the old in-process tail)."""

    def _fail(name: str, exc: BaseException) -> bool:
        _vyvar_record_run_failure(cfg, _run_label, name, str(exc))
        if footer_placeholder is not None:
            _vyvar_footer_set(
                footer_placeholder,
                running=False,
                process=f"{_run_label} - {name}",
                status_detail=f"Error: {str(exc)[:700]}",
                pct=None,
            )
        return False

    if nr.plan is not None:
        st.session_state["vyvar_smart_plan"] = nr.plan
        st.session_state["vyvar_last_import_plan"] = nr.plan
    st.session_state["vyvar_last_import_equipment_id"] = int(
        (st.session_state.get("vyvar_active_job") or {}).get("equipment_id") or 0
    )
    if nr.import_result is not None:
        st.session_state["vyvar_last_import_result"] = nr.import_result
    if nr.draft_id is not None:
        st.session_state["vyvar_last_draft_id"] = int(nr.draft_id)
    if nr.archive_path is not None:
        st.session_state["vyvar_post_cal_archive_path"] = str(nr.archive_path)
        if nr.plan is not None:
            st.session_state["vyvar_post_cal_plan_source"] = str(
                getattr(nr.plan, "source_root", "") or ""
            )
        st.session_state["vyvar_status_calibrated"] = True
    if nr.memory_profile is not None:
        st.session_state["vyvar_memory_profile"] = nr.memory_profile
    if nr.ra_ui is not None:
        st.session_state["vyvar_pending_center_ra"] = float(nr.ra_ui)
    if nr.de_ui is not None:
        st.session_state["vyvar_pending_center_de"] = float(nr.de_ui)
    if nr.processing_hash:
        st.session_state["vyvar_observation_processing_hash"] = nr.processing_hash
    qsum = dict(nr.ram_qc_summary or {})
    analyze_token = f"ram_qc:{nr.draft_id}:{qsum.get('n_lights')}:{qsum.get('median_fwhm')}"
    st.session_state["vyvar_last_job_output"] = {
        "job_kind": "analyze",
        "analyze_token": analyze_token,
        "archive_path": str(nr.draft_dir or ""),
        "draft_id": nr.draft_id,
        "ram_qc_summary": qsum,
        "qc_suggestions": {},
        "pointing_scan": nr.pointing,
        "prefill_ra_text": "",
        "prefill_dec_text": "",
        "suggested_reject_fwhm_px": None,
        "suggest_max_detected_stars": None,
        "memory_profile": nr.memory_profile,
    }
    if nr.platesolve_job_output is not None:
        st.session_state["vyvar_last_job_output"] = nr.platesolve_job_output
    st.session_state["vyvar_last_job_summary"] = {"kind": "analyze", **qsum}
    st.session_state["vyvar_status_analyzed"] = True

    if not nr.success:
        err = " ; ".join(nr.errors) if nr.errors else "night run failed"
        return _fail("Night pipeline", RuntimeError(err))

    _vyvar_reset_variability_session_state()
    zero_target_setups = list(nr.zero_target_setups or [])
    completed = list(nr.completed_setups or [])
    _photometry_ran = [s for s in completed if s not in zero_target_setups]
    if zero_target_setups and not _photometry_ran:
        _zero_msg = (
            "Pipeline finished - 0 active targets, photometry not run "
            "(check the VSX catalog / target selection)"
        )
        LOGGER.info(f"[RUN VYVAR] ! {_zero_msg}")
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
        LOGGER.info(
            "! RUN VYVAR finished PARTIALLY - photometry OK: ["
            + ", ".join(_photometry_ran)
            + "]; 0 targets (no photometry): ["
            + ", ".join(zero_target_setups)
            + "]"
        )
    else:
        LOGGER.info("[RUN VYVAR] [OK] Pipeline finished successfully")
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


def _vyvar_record_run_failure(cfg: AppConfig, label: str, step: str, detail: str) -> None:
    """Log line + preflight error log file + session keys used by the error banner."""
    LOGGER.info(f"[{label}] x Failure in step '{step}': {detail}")
    st.session_state["vyvar_last_preflight_error_message"] = f"{step}: {detail}"
    try:
        from run_preflight_log import write_run_preflight_error_log  # noqa: PLC0415

        _p = write_run_preflight_error_log(cfg.data_root, step=step, exc=RuntimeError(detail), cfg=cfg)
        st.session_state["vyvar_last_preflight_error_log"] = str(_p)
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("run failure log write failed: %s", exc)


def _vyvar_job_running() -> bool:
    job = st.session_state.get("vyvar_active_job")
    if not job or job.get("applied"):
        return False
    from run_job import job_state  # noqa: PLC0415

    return job_state(job["job_dir"]) == "running"


def _vyvar_render_run_job_monitor(cfg: AppConfig, footer_placeholder: Any | None) -> None:
    """Follow a RUN VYVAR job running in its own process (JOB-2026-10).

    Polls ``run_status.json`` every 2 s inside a fragment, so clicks elsewhere in
    the UI never interrupt the run. When the job ends the result is applied to
    the session state once and the whole app reruns.
    """
    if not st.session_state.get("vyvar_active_job"):
        return
    from run_job import job_state, load_result, read_status, stop_job  # noqa: PLC0415

    @st.fragment(run_every=2.0)
    def _monitor() -> None:
        job = st.session_state.get("vyvar_active_job")
        if not job:
            return
        jd = job["job_dir"]
        label = str(job.get("label") or "RUN VYVAR")
        state = job_state(jd)
        info = read_status(jd) or {}
        msg = str(info.get("message") or "")
        if state == "running":
            vyvar_init_footer_state_if_missing()
            st.session_state["vyvar_footer_state"].update(
                {"running": True, "process": label, "status_detail": msg[:800]}
            )
            _vyvar_render_fixed_footer_into(st.empty())
            c1, c2 = st.columns([5, 1])
            c1.info(f"{label} is running in its own process - {msg}")
            if c2.button("Stop run", key="vyvar_stop_active_job"):
                stop_job(jd)
                LOGGER.info(f"[{label}] stopped by user")
            return
        if job.get("applied"):
            return
        ok = False
        nr = load_result(jd)
        if nr is not None:
            ok = _vyvar_apply_run_result(nr, cfg, footer_placeholder, label)
        else:
            errs = info.get("errors") or [msg or f"job ended with state {state}"]
            if state == "lost":
                errs = [f"process ended without a result - see {jd}/run_stdout.log"]
            _vyvar_record_run_failure(cfg, label, "Night pipeline", "; ".join(map(str, errs))[:800])
            _vyvar_footer_set(
                None, running=False, process=label, status_detail="Error - see the infolog of the processed field", pct=None
            )
        job.update({"applied": True, "ok": bool(ok)})
        st.session_state["vyvar_active_job"] = job
        st.rerun(scope="app")

    _monitor()
    job = st.session_state.get("vyvar_active_job") or {}
    if job.get("applied"):
        if job.get("ok"):
            st.success(f"{job.get('label', 'RUN VYVAR')} finished successfully. Review the results.")
        else:
            st.error(
                _vyvar_format_run_failure_message(
                    default="Pipeline stopped. See the infolog_*.txt file in the processed field folder."
                )
            )


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
        from catalog_db import validate_gaia_db_schema

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
            LOGGER.info(f"Field center save skipped: {_exc_center!s}")


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
            LOGGER.info(f"- Spustam: {pending.get('kind')} - {ap}")

            def _cb(i: int, total: int, msg: str) -> None:
                pct = int(round(100 * (i / max(total, 1))))
                if i > 0 and pct < 1:
                    pct = 1
                if progress_bar is not None:
                    progress_bar.progress(pct, text=msg)
                LOGGER.info(f"[{i}/{total}] {msg}")
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

            if pending.get("kind") == "run_epsf":
                from epsf_stage import EpsfStagePaths, run_epsf_stage

                _draft_epsf = pending.get("draft_id")
                if _draft_epsf is None:
                    _draft_epsf = st.session_state.get("vyvar_last_draft_id")
                if _draft_epsf is None:
                    raise FileNotFoundError("Missing processed field id for the RUN ePSF job.")
                _draft_epsf = int(_draft_epsf)
                _targets_epsf = str(pending.get("epsf_targets") or "ALL")
                _setups_epsf = list(pending.get("setups") or [])
                if not _setups_epsf:
                    raise ValueError("RUN ePSF job without setups.")
                # EPSF-MULTISETUP-2026-10: every setup (filter / obs group) of the draft, same selection.
                _lines_epsf: list[str] = []
                _n_lc_epsf = 0
                for _k_epsf, _su in enumerate(_setups_epsf, start=1):
                    _name_epsf = str(_su.get("setup_name") or "")
                    _ms_fits = Path(str(_su["masterstar_fits_path"]))
                    _ms_csv = Path(str(_su["masterstars_csv_path"]))
                    _frames_root = Path(str(_su["per_frame_csv_dir"]))
                    _ps_dir = Path(str(_su["platesolve_dir"]))
                    for _f, _what in ((_ms_fits, "MASTERSTAR.fits"), (_ms_csv, "masterstars_full_match.csv")):
                        if not _f.is_file():
                            raise FileNotFoundError(f"{_name_epsf}: {_what} not found: {_f}")
                    if not _frames_root.is_dir():
                        raise FileNotFoundError(f"{_name_epsf}: aligned frames dir not found: {_frames_root}")
                    _tag_epsf = f"setup {_k_epsf}/{len(_setups_epsf)} {_name_epsf}"

                    def _stage_cb(msg: str, _tag: str = _tag_epsf) -> None:
                        LOGGER.info(f"[ePSF job] {_tag}: {msg}")
                        parts = str(msg).split()
                        if len(parts) >= 3 and parts[0] == "ePSF" and parts[1] == "fit" and "/" in parts[2]:
                            cur, tot = parts[2].split("/", 1)
                            try:
                                _cb(int(cur), int(tot), f"{_tag}: {msg}")
                            except (TypeError, ValueError):
                                pass

                    LOGGER.info(f"[ePSF job] {_tag_epsf}: run_epsf_stage (build + fit + merge + LCs)")
                    stage_out = run_epsf_stage(
                        params=None,
                        paths=EpsfStagePaths(
                            platesolve_dir=_ps_dir,
                            frames_root=_frames_root,
                            masterstar_fits=_ms_fits,
                            masterstars_csv=_ms_csv,
                        ),
                        cfg=cfg,
                        progress_cb=_stage_cb,
                        db=pipeline.db,
                        draft_id=_draft_epsf,
                        targets=_targets_epsf,
                    )
                    _n_tg = len((stage_out.get("fit_set") or {}).get("target_types") or {})
                    _n_lc = int((stage_out.get("lc") or {}).get("n_written") or 0)
                    _written = int((stage_out.get("merge") or {}).get("written") or 0)
                    _n_lc_epsf += _n_lc
                    _exp = stage_out.get("export")
                    _lines_epsf.append(f"{_name_epsf}: {_n_tg} target(s), {_n_lc} PSF light curve(s), "
                                       f"{_written} frame(s) fitted"
                                       + (f", {_exp['n_written']} light-curve file(s)" if _exp else ""))
                _out_epsf = {
                    "job_kind": "run_epsf",
                    "status": "ok",
                    "archive_path": str(pending.get("archive_path") or ""),
                    "n_lightcurves": _n_lc_epsf,
                    "message": f"ePSF photometry done ({len(_setups_epsf)} setup(s), targets '{_targets_epsf}'): "
                    + "; ".join(_lines_epsf),
                }
                st.session_state["vyvar_last_job_output"] = _out_epsf
                st.session_state["vyvar_last_job_summary"] = {"kind": "run_epsf", "n_lightcurves": _n_lc_epsf}
                LOGGER.info(f"[ePSF job] Done - {_out_epsf['message']}")

            else:
                raise ValueError("Unknown job kind.")

            if stt is not None:
                stt.update(label="Done.", state="complete")
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
            LOGGER.info(f"Job failed: {_err_msg} [{type(exc).__name__}]")
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
            LOGGER.info(f"Job failed: {_err_msg} [{type(exc).__name__}]")
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
    st.subheader("VARSTREM")
    # TODO: extract _render_fits_qa_tab - heavy st.session_state / pending-job wiring; defer Phase 2.
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
                    f"SELECT ID,CAMERANAME,ALIAS,SENSORTYPE,SENSORSIZE,PIXELSIZE,GAIN_ADU,MODEL_KEY "
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
            # SITE-UI-01: site of this session (default: IS_DEFAULT) else none.
            # Persist happens on RUN, never as a render side-effect.
            from observer_location import location_preselect_id  # noqa: PLC0415

            _loc_default_id = location_preselect_id(
                locations, int(getattr(cfg, "observer_location_id", 0) or 0)
            )
            _loc_default_idx = 0
            if _loc_default_id is not None:
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
            help=(
                "Observer site for BJD, airmass, and lunar context. "
                "What you see here is what RUN uses. The preselected site is the default site "
                "(the default site in Database Explorer > Sites)."
            ),
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
                    "Pointing (RA/Dec)": "blind plate-solve / set the field center after import",
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
                        "(header/solve -> DB -> config, site = processed field -> header -> flagged config) "
                        "then applies them per draft."
                    )

        # SITE-UI-01: selection is the run site; do not rewrite config.json on render.
        import_location_id = 0
        if loc_labels and import_location_label in location_options:
            import_location_id = int(location_options[import_location_label])
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
            _mf_label = st.radio(
                "Masters (DARK / FLAT)",
                ["from the LIGHT folder", "from the Calibration Library"],
                index=0 if st.session_state.get("vyvar_masters_from", "lights") == "lights" else 1,
                key="vyvar_masters_from_radio",
                help=(
                    "Masters lying next to the LIGHTs are recognised from their headers. The chosen "
                    "source is tried first, the other one is the fallback. Masters taken from the LIGHT "
                    "folder are imported into the Calibration Library when RUN VYVAR starts. "
                    "New masters are made only in the Calibration Library tab."
                ),
            )
            _masters_from = "lights" if _mf_label.startswith("from the LIGHT") else "library"
            st.session_state["vyvar_masters_from"] = _masters_from
            if st.button("[search] Scan Source", type="primary", help="Information only - nothing is written."):
                if import_equipment_id <= 0 or import_telescope_id <= 0:
                    st.error("Select a valid camera and telescope before scanning.")
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
                            masters_from=_masters_from,
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
            # SITE-UI-01: caption and gate use the same selectbox value that RUN passes.
            _run_site_row = None
            if import_location_id > 0:
                _run_site_row = get_observer_location_by_id(
                    str(cfg.database_path), int(import_location_id)
                )
            if _run_site_row is not None:
                st.caption(
                    f"Site for this run: {_run_site_row.get('name') or import_location_id} "
                    f"(lat={_run_site_row.get('lat')}, lon={_run_site_row.get('lon')})"
                )
            else:
                st.caption("Site for this run: (none selected - RUN blocked)")
            run_vyvar_disabled = (
                not _sr_rv
                or not Path(_sr_rv).is_dir()
                or not _gaia_ok
                or import_equipment_id <= 0
                or import_telescope_id <= 0
                or import_location_id <= 0
                or _vyvar_job_running()
            )
            btn_cal, btn_nc = st.columns(2)
            with btn_cal:
                run_vyvar_clicked = st.button(
                    "! RUN VYVAR",
                    type="primary",
                    disabled=run_vyvar_disabled,
                    help="One click: scan -> import -> calibrate -> analyze -> MASTERSTAR -> target and comparison-star selection -> aperture photometry",
                )
            with btn_nc:
                run_vyvar_nc_clicked = st.button(
                    "[folder] RUN VYVAR (non-cal)",
                    type="secondary",
                    disabled=run_vyvar_disabled,
                    help=(
                        "Skips dark/flat. LIGHTs are used as they are: calibrated by another program, or deliberately uncalibrated (e.g. "
                        "Telescope Live exports). Proceeds with alignment -> masterstar -> photometry."
                    ),
                )
            if run_vyvar_clicked or run_vyvar_nc_clicked:
                _pre_cal = bool(run_vyvar_nc_clicked)
                _dao_sigma_default = resolve_detection_threshold_sigma(cfg.detection_threshold_sigma)
                st.session_state["dao_threshold_sigma"] = _dao_sigma_default
                st.session_state.pop("vyvar_active_job", None)
                _started = _run_vyvar_full_pipeline(
                    pipeline=pipeline,
                    cfg=cfg,
                    source_root=_sr_rv,
                    import_equipment_id=int(import_equipment_id),
                    import_telescope_id=int(import_telescope_id),
                    import_location_id=int(import_location_id),
                    dark_validity_days=int(dark_validity_days),
                    flat_validity_days=int(flat_validity_days),
                    dao_sigma_default=_dao_sigma_default,
                    cat_match_arc=2.0,
                    max_cat_rows=12000,
                    min_stars=100,
                    max_stars=500,
                    footer_placeholder=footer_placeholder,
                    pre_calibrated_mode=_pre_cal,
                    masters_from=_masters_from,
                )
                if not _started:
                    st.error(
                        _vyvar_format_run_failure_message(
                            default="RUN VYVAR could not start. See the infolog_*.txt file in the processed field folder."
                        )
                    )
            _vyvar_render_run_job_monitor(cfg, footer_placeholder)

    plan = st.session_state.get("vyvar_smart_plan")
    if plan:
        # Scan Source is information only (CAL-2026-10): what RUN VYVAR will calibrate with.
        st.markdown("---")
        st.subheader("Scan Source")
        lights_bad = any(r.type == "Lights" and r.status in ("missing", "empty") for r in plan.scan_rows)
        if lights_bad:
            st.error(
                "No light frames found (no FITS with IMAGETYP light/object/science in the source folder)."
            )
        else:
            _cpd = getattr(plan, "calibration_plan", None) or {}
            _ng = len(_cpd.get("groups") or [])
            _filters = ", ".join(getattr(plan, "detected_filters", None) or []) or "-"
            _src = "LIGHT folder" if _cpd.get("masters_from", "lights") == "lights" else "Calibration Library"
            _nxt = _cpd.get("masters_next_to_lights") or []
            st.markdown(
                f"**{len(plan.lights_files)} LIGHTs** in {_ng} group(s), filters {_filters}. "
                f"Masters next to the LIGHTs: {len(_nxt)}"
                + (f" ({', '.join(_nxt)})" if _nxt else "")
                + f". Preferred source: **{_src}**."
            )
            st.dataframe(ui_calibration.build_calibration_plan_dataframe(plan), width="stretch", hide_index=True)
            if plan.warnings:
                with st.expander(f"{len(plan.warnings)} note(s)", expanded=False):
                    for _w in plan.warnings:
                        st.write(f"- {_w}")


    _sync_varstrem_session_state(pipeline)
    if _render_pending_job_dispatcher(
        pipeline=pipeline,
        cfg=cfg,
        footer_placeholder=footer_placeholder,
    ):
        return


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
    pipeline = _cached_astro_pipeline(
        str(cfg.database_path.resolve()),
        _config_cache_fingerprint(cfg),
    )

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
        st.subheader("Load processed field")
        st.caption(
            "A field VYVAR has processed: its folder name under ``Archive/Drafts`` "
            "(``MN_Boo_20260422_001``), the full path of the folder (must contain ``platesolve/``) or its "
            "number. Used in FITS QA, MASTERSTAR QA, Aperture Photometry, ePSF and Variability. "
            "Empty field + Load clears it."
        )
        dcol1, dcol2, dcol3 = st.columns([4, 1, 1])
        with dcol1:
            draft_path_inp = st.text_input(
                "Processed field (name, path or number)",
                key="vyvar_draft_path_field",
                placeholder=r"e.g. MN_Boo_20260422_001 or C:\...\Archive\Drafts\MN_Boo_20260422_001",
            )
        with dcol2:
            apply_draft = st.button("Load", key="vyvar_draft_path_apply", type="primary")
        with dcol3:
            _cur = st.session_state.get("vyvar_last_draft_id")
            st.caption(f"ID: **{_cur}**" if _cur is not None else "ID: -")
        if apply_draft:
            s = (draft_path_inp or "").strip()
            if not s:
                st.session_state.pop("vyvar_draft_dir_override", None)
                _vyvar_reset_variability_session_state()
                st.info("Processed field cleared - using the archive path from the configuration.")
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
                    st.success(f"Processed field loaded: {ddir}")
                    st.rerun()
        ov = _vyvar_effective_draft_dir_override()
        if ov is not None:
            st.caption(f"Active override: `{ov}`")

        _draft_ov = ov
        # A view switch, not st.tabs: Streamlit runs every tab body on each rerun (MASTERSTAR overlay,
        # FITS previews, ePSF plans, variability tables), so every click paid for all six views.
        views = [
            "VAR-STREM",
            "FITS QA",
            "MASTERSTAR QA",
            "Aperture Photometry",
            "[microscope] ePSF",
            "[search] Variability",
        ]
        view = st.segmented_control(
            "View", views, default=views[0], key="vyvar_main_view", label_visibility="collapsed"
        ) or views[0]
        if view == views[0]:
            render_live_view(pipeline=pipeline, cfg=cfg, footer_placeholder=vyvar_footer_ph)
        elif view == views[1]:
            ui_quality_dashboard.render_quality_dashboard(
                db=pipeline.db,
                draft_id=st.session_state.get("vyvar_last_draft_id"),
                archive_text="",
                cfg=cfg,
            )
        elif view == views[2]:
            ui_masterstar_qa.render_masterstar_qa(
                cfg=cfg,
                draft_id=st.session_state.get("vyvar_last_draft_id"),
                pipeline=pipeline,
                draft_dir_override=_draft_ov,
            )
        elif view == views[3]:
            render_aperture_photometry(
                cfg=cfg,
                draft_id=st.session_state.get("vyvar_last_draft_id"),
                draft_dir_override=_draft_ov,
            )
        elif view == views[4]:
            from ui_epsf_dashboard import render_epsf_dashboard

            render_epsf_dashboard(
                draft_dir=_draft_ov,
                cfg=cfg,
                draft_id=st.session_state.get("vyvar_last_draft_id"),
            )
        else:
            render_variability_dashboard(
                cfg,
                draft_id=st.session_state.get("vyvar_last_draft_id"),
                draft_dir_override=_draft_ov,
            )
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

