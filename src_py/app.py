"""Streamlit entrypoint for live and archive views."""

from __future__ import annotations

import html
import logging
from pathlib import Path
from typing import Any

import streamlit as st

from config import AppConfig
from database import (
    get_observer_location_by_id,
    get_observer_locations,
)
from run_preflight_log import write_run_preflight_error_log
from importer import smart_scan_source
from optics_selection import (
    VyvarOpticsSelection,
    log_active_optics,
    parse_ui_optics_from_labels,
    resolve_working_optics,
)
from ui_session import active_draft, clear_active_draft, optics_from_session, set_active_draft, sync_optics_session
from pipeline import AstroPipeline
import ui_calibration as ui_calibration
import ui_database_explorer as ui_database_explorer
import ui_masterstar_qa as ui_masterstar_qa
import ui_quality_dashboard as ui_quality_dashboard
from platesolve_ui_paths import resolve_draft_directory
from ui_aperture_photometry import render_aperture_photometry
from ui_variability import render_variability_dashboard

LOGGER = logging.getLogger(__name__)



def _vyvar_format_run_failure_message(*, default: str) -> str:
    """Prefer the on-disk preflight log path when an early failure left no draft infolog."""
    log_path = st.session_state.get("vyvar_last_preflight_error_log")
    detail = st.session_state.get("vyvar_last_preflight_error_message")
    if log_path:
        msg = str(detail or "Pipeline preflight failed before the infolog of the processed field was opened.")
        return f"{msg} Preflight log: {log_path}"
    return default


def _run_vyvar_full_pipeline(
    *,
    pipeline: AstroPipeline,
    cfg: AppConfig,
    source_root: str,
    import_equipment_id: int,
    import_telescope_id: int,
    import_location_id: int,
    footer_placeholder: Any | None = None,
    pre_calibrated_mode: bool = False,
    masters_from: str = "lights",
) -> bool:
    """RUN VYVAR - the CLI's params from the UI selection, the run's preconditions, then the job
    (night_run.run_night_pipeline in a child process, INV-ONE-ENTRY-01)."""
    from night_run import check_run_preconditions

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

        _run_optics = resolve_working_optics(
            pipeline.db,
            ui=VyvarOpticsSelection(
                int(import_equipment_id),
                int(import_telescope_id),
            ),
            context=_run_label,
        )
        sync_optics_session(_run_optics)

        loc_id = int(import_location_id)
        if loc_id <= 0:
            raise ValueError("No observing site selected; select an observatory site before RUN.")
        # G12-10 / G12-11: the same params as the CLI for the same inputs (validity windows, DAO
        # threshold and optics come from the run's config and DB in the job).
        params = _ui_run_params(
            source_root=str(_root),
            equipment_id=int(_run_optics.equipment_id),
            telescope_id=int(_run_optics.telescope_id),
            location_id=int(loc_id),
            pre_calibrated_mode=bool(pre_calibrated_mode),
            masters_from=str(masters_from or "lights"),
        )
        # G12-15 / G12-16: the run's own precondition check; the site is validated, cfg is not changed
        _pre = check_run_preconditions(cfg, params, pipeline.db)
        if _pre.problems:
            return _fail("Preconditions", RuntimeError("; ".join(_pre.problems)))
        _update("Night pipeline")
        from run_job import launch_job  # noqa: PLC0415

        _job_dir = launch_job(params, cfg.data_root, label=_run_label)
        st.session_state["vyvar_active_job"] = {
            "job_dir": str(_job_dir),
            "label": _run_label,
            "kind": "night",
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
    st.session_state["vyvar_last_import_equipment_id"] = int(
        (st.session_state.get("vyvar_active_job") or {}).get("equipment_id") or 0
    )
    if nr.draft_id is not None:
        # G14-12: the finished run's field becomes the loaded processed field of every view
        _dd = Path(str(nr.draft_dir)) if nr.draft_dir else None
        set_active_draft(_dd if _dd is not None and _dd.is_dir() else None, int(nr.draft_id))
    if not nr.success:
        err = " ; ".join(nr.errors) if nr.errors else "night run failed"
        return _fail("Night pipeline", RuntimeError(err))

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


def _vyvar_job_running(cfg: AppConfig) -> bool:
    """G14-37: any job of this data directory, from any browser session (``run_job.running_job``)."""
    from ui_session import any_job_running  # noqa: PLC0415

    return any_job_running(cfg.data_root)


def _vyvar_finish_task_job(job: dict[str, Any], state: str, info: dict[str, Any]) -> bool:
    """A finished task job (``task_jobs``): the result message replaces the in-thread output."""
    from task_jobs import load_task_result  # noqa: PLC0415

    for k in job.get("on_done_pop") or []:
        st.session_state.pop(str(k), None)
    res = load_task_result(job["job_dir"]) if state == "done" else None
    if res is not None:
        job["message"] = str(res.get("message") or "Done")
        for line in res.get("messages") or []:
            LOGGER.info(f"[{job.get('label')}] {line}")
        return True
    msg = str(info.get("message") or f"job ended with state {state}")
    if state == "lost":
        msg = f"process ended without a result - see {job['job_dir']}/run_stdout.log"
    job["message"] = msg
    return False


def _vyvar_render_run_job_monitor(cfg: AppConfig, footer_placeholder: Any | None) -> None:
    """Follow the job of this session in its own process (JOB-2026-10; G14-04: any view, any page).

    RUN VYVAR (``kind == "night"``) and the task jobs (``task_jobs``: RUN ePSF, variability search,
    TESS, catalogue check, HRD lookup, Build masters). Polls ``run_status.json`` every 2 s inside a
    fragment, so clicks elsewhere never interrupt the job. When it ends the result is applied to the
    session once and the whole app reruns. A job started by another browser session is shown too
    (G14-37: it blocks every start button of this data directory).
    """
    from run_job import job_state, load_result, read_status, running_job, stop_job  # noqa: PLC0415

    if not st.session_state.get("vyvar_active_job"):
        other = running_job(cfg.data_root)
        if other is not None:
            _lbl = (read_status(other) or {}).get("label") or other.name
            st.info(f"{_lbl} is running (job `{other.name}`, started in another browser session or tab, or from the command line). "
                    "New runs and jobs start when it has finished.")
        return

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
            if c2.button("Stop", key="vyvar_stop_active_job"):
                stop_job(jd)
                LOGGER.info(f"[{label}] stopped by user")
            return
        if job.get("applied"):
            return
        ok = False
        if str(job.get("kind") or "night") != "night":
            ok = _vyvar_finish_task_job(job, state, info)
            _vyvar_footer_set(None, running=False, process=label, status_detail=str(job.get("message"))[:800],
                              pct=100 if ok else None)
        else:
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
        if str(job.get("kind") or "night") != "night":
            (st.success if job.get("ok") else st.error)(f"{job.get('label')}: {job.get('message')}")
        elif job.get("ok"):
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


def _ui_run_params(
    *, source_root: str, equipment_id: int, telescope_id: int, location_id: int, pre_calibrated_mode: bool = False,
    masters_from: str = "lights",
) -> Any:
    """The NightRunParams of the UI selection (G12-10: the CLI's params for the same inputs)."""
    from night_run import NightRunParams  # noqa: PLC0415

    return NightRunParams(
        source_dir=Path(str(source_root).strip()),
        equipment_id=int(equipment_id),
        telescope_id=int(telescope_id),
        location_id=int(location_id),
        location_source_hint="ui_selection",
        pre_calibrated_mode=bool(pre_calibrated_mode),
        masters_from=str(masters_from or "lights"),
    )


def _run_button_blockers(cfg: AppConfig, db: Any, *, source_root: str, equipment_id: int, telescope_id: int,
                         location_id: int) -> list[str]:
    """S6-FIX RUN-GATE-2026-10: why RUN VYVAR cannot start now - the full run precondition check
    (``night_run.check_run_preconditions``: source folder, camera and telescope rows, site, Gaia DR3), one call
    per render, plus the UI-only reasons (nothing typed / no site picked / a job of the data directory runs)."""
    from night_run import check_run_preconditions  # noqa: PLC0415

    reasons: list[str] = []
    if not str(source_root).strip():
        reasons.append("No source folder selected")
    if int(location_id) <= 0:
        reasons.append("No observing site selected")
    if not reasons:
        params = _ui_run_params(source_root=source_root, equipment_id=equipment_id, telescope_id=telescope_id,
                                location_id=location_id)
        reasons += check_run_preconditions(cfg, params, db).problems
    if _vyvar_job_running(cfg):
        reasons.append("A job of this data directory is running")
    return reasons


_RIG_EQUIPMENT = "vyvar_rig_equipment"
_RIG_TELESCOPE = "vyvar_rig_telescope"
_RIG_LOCATION = "vyvar_rig_location"
_RIG_SOURCE = "vyvar_rig_source_root"


def _rig_index(options: list[str], store_key: str) -> int:
    """Index of the stored selection in ``options`` (0 when it is not there)."""
    v = st.session_state.get(store_key)
    return options.index(v) if v in options else 0


def _rig_set(store_key: str, widget_key: str, value: str) -> None:
    """Auto-detect: set the stored selection; the widget is rebuilt from it on the rerun."""
    st.session_state[store_key] = value
    st.session_state.pop(widget_key, None)


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
    # G14-06: the selection lives in non-widget keys (_RIG_*), so a view switch (which drops the state
    # of widgets that are not rendered) does not reset it; the widgets take index/value from them.
    if _def_eq_lbl and _RIG_EQUIPMENT not in st.session_state:
        st.session_state[_RIG_EQUIPMENT] = _def_eq_lbl
    if _def_tel_lbl and _RIG_TELESCOPE not in st.session_state:
        st.session_state[_RIG_TELESCOPE] = _def_tel_lbl
    st.session_state.setdefault(_RIG_SOURCE, str(cfg.archive_root))

    with st.expander("[folder] Define source data", expanded=True):
        source_root = st.text_input(
            "Source Directory",
            value=str(st.session_state[_RIG_SOURCE]),
            key="vyvar_varstrem_source",
            help="Example: USB session root (any structure; will be scanned recursively).",
        )
        st.session_state[_RIG_SOURCE] = source_root
        st.caption(
            "To find masters in the Calibration Library, choose a **set** (camera + telescope). "
            "Matching or generic (no set) library entries are used per DB rules."
        )

        # Phase 3: fingerprint camera + telescope from a sample FITS header and auto-fill the
        # selectors (overriding the IS_DEFAULT baseline). The user can still override. The site is
        # never auto-selected (S6-FIX SITE-MANUAL-2026-10): the user picks it below.
        if st.button(
            "[test] Auto-detect optics from FITS",
            help="Reads a sample light frame and matches camera / telescope by "
            "INSTRUME, sensor dimensions, GAIN and FOCALLEN/APTDIA. The site is always your choice below.",
        ):
            from optics_autodetect import autodetect_from_source

            _eq_rows, _tel_rows = pipeline.db.autodetect_candidates()  # G14-23
            _rep = autodetect_from_source(source_root, equipments=_eq_rows, telescopes=_tel_rows)
            if _rep.header_path is None:
                st.warning("Auto-detect: no readable light FITS found under the source directory.")
            else:
                if _rep.equipment.ok and _rep.equipment.label in equipment_options:
                    _rig_set(_RIG_EQUIPMENT, "vyvar_varstrem_equipment", _rep.equipment.label)
                if _rep.telescope.ok and _rep.telescope.label in telescope_options:
                    _rig_set(_RIG_TELESCOPE, "vyvar_varstrem_telescope", _rep.telescope.label)
                st.session_state["vyvar_autodetect_report"] = _rep
            st.rerun()

        col_scan_eq, col_scan_tel = st.columns(2)
        with col_scan_eq:
            eq_placeholder = "(no camera in DB)" if not eq_labels else eq_labels[0]
            _eq_opts = eq_labels if eq_labels else [eq_placeholder]
            import_equipment_label = st.selectbox(
                "Equipment (library)",
                options=_eq_opts,
                index=_rig_index(_eq_opts, _RIG_EQUIPMENT),
                key="vyvar_varstrem_equipment",
            )
            st.session_state[_RIG_EQUIPMENT] = import_equipment_label
        with col_scan_tel:
            tel_placeholder = "(no telescope in DB)" if not tel_labels else tel_labels[0]
            _tel_opts = tel_labels if tel_labels else [tel_placeholder]
            import_telescope_label = st.selectbox(
                "Telescope (library)",
                options=_tel_opts,
                index=_rig_index(_tel_opts, _RIG_TELESCOPE),
                key="vyvar_varstrem_telescope",
            )
            st.session_state[_RIG_TELESCOPE] = import_telescope_label
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
            if _RIG_LOCATION not in st.session_state:
                st.session_state[_RIG_LOCATION] = loc_labels[_loc_default_idx]
        _loc_opts = loc_labels if loc_labels else [loc_placeholder]
        import_location_label = st.selectbox(
            "Location (observatory)",
            options=_loc_opts,
            index=_rig_index(_loc_opts, _RIG_LOCATION),
            key="vyvar_varstrem_location",
            disabled=not loc_labels,
            help=(
                "Observer site for BJD, airmass, and lunar context. "
                "What you see here is what RUN uses. The preselected site is the default site "
                "(the default site in Database Explorer > Sites)."
            ),
        )
        if loc_labels:
            st.session_state[_RIG_LOCATION] = import_location_label

        # Phase 3: show the last auto-detect result (match + confidence + evidence).
        _ad_rep = st.session_state.get("vyvar_autodetect_report")
        if _ad_rep is not None:
            _band_icon = {"high": "[OK]", "medium": "[yellow]", "low": "[orange]", "none": "o"}

            def _ad_line(_title: str, _det: Any) -> str:
                _ic = _band_icon.get(_det.band(), "o")
                if _det.matched_id is None:
                    _why = f" ({'; '.join(_det.reasons)})" if _det.reasons else ""
                    return f"{_ic} **{_title}:** no confident match{_why} - set manually (poor-FITS prompt)."
                _ev = "; ".join(_det.reasons) if _det.reasons else "-"
                return (
                    f"{_ic} **{_title}:** {_det.label} "
                    f"(confidence {_det.confidence:.0%}, {_det.band()}) - {_ev}"
                )

            with st.container(border=True):
                st.caption(f"[search] Auto-detect from `{Path(_ad_rep.header_path).name}`")
                st.markdown(_ad_line("Equipment", _ad_rep.equipment))
                st.markdown(_ad_line("Telescope", _ad_rep.telescope))
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
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Scan failed: {exc}")
                        st.session_state.pop("vyvar_smart_plan", None)
        with col_run:
            _sr_rv = str(source_root).strip()
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
            _run_blockers = _run_button_blockers(
                cfg, pipeline.db, source_root=_sr_rv, equipment_id=int(import_equipment_id),
                telescope_id=int(import_telescope_id), location_id=int(import_location_id),
            )
            run_vyvar_disabled = bool(_run_blockers)
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
            if _run_blockers:
                st.warning("RUN VYVAR is off:\n" + "\n".join(f"- {r}" for r in _run_blockers))
            if run_vyvar_clicked or run_vyvar_nc_clicked:
                _pre_cal = bool(run_vyvar_nc_clicked)
                st.session_state.pop("vyvar_active_job", None)
                _started = _run_vyvar_full_pipeline(
                    pipeline=pipeline,
                    cfg=cfg,
                    source_root=_sr_rv,
                    import_equipment_id=int(import_equipment_id),
                    import_telescope_id=int(import_telescope_id),
                    import_location_id=int(import_location_id),
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




def _config_cache_fingerprint(cfg: AppConfig) -> str:
    """Bump Streamlit cache when config.json changes (Settings save)."""
    path = cfg.data_root / "config.json"
    try:
        stat = path.stat()
        return f"{int(stat.st_mtime_ns)}:{stat.st_size}"
    except OSError:
        return "missing"


def _release_astro_pipeline(pipeline: AstroPipeline) -> None:
    """G14-09: the DB connection of a pipeline evicted from the cache is closed (no leak per Settings save)."""
    pipeline.db.conn.close()


@st.cache_resource(max_entries=1, on_release=_release_astro_pipeline)
def _cached_astro_pipeline(database_path: str, config_fingerprint: str, _cfg: AppConfig) -> AstroPipeline:
    """One DB connection per Streamlit server process (avoids startup migration races).

    The key is the database path and ``config.json`` mtime+size, so a Settings save rebuilds the pipeline
    from the ``AppConfig`` of this run (``_cfg``, G14-09: not a second ``AppConfig()``); the previous
    pipeline is released and its connection closed (``max_entries=1``)."""
    _ = (database_path, config_fingerprint)
    return AstroPipeline(config=_cfg)


def main() -> None:
    from vyvar_runtime import ensure_data_dir_for_entry

    ensure_data_dir_for_entry()  # G12-29: the same bootstrap as the CLI
    cfg = AppConfig()
    cfg.ensure_base_dirs()
    pipeline = _cached_astro_pipeline(
        str(cfg.database_path.resolve()),
        _config_cache_fingerprint(cfg),
        cfg,
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
    _vyvar_render_run_job_monitor(cfg, vyvar_footer_ph)

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
        if apply_draft:
            s = (draft_path_inp or "").strip()
            if not s:
                clear_active_draft()
                st.info("Processed field cleared.")
                st.rerun()
            else:
                ddir, parsed_id, err = resolve_draft_directory(
                    s, archive_root=Path(cfg.archive_root)
                )
                if err:
                    st.error(err)
                elif ddir is not None:
                    set_active_draft(ddir, parsed_id)
                    st.success(f"Processed field loaded: {ddir}")
                    st.rerun()
        # G14-12: one processed field for every view (folder + id resolved once here).
        active = active_draft(cfg.archive_root)
        with dcol3:
            st.caption(f"ID: **{active.draft_id}**" if active.draft_id is not None else "ID: -")
        if active.draft_dir is not None:
            st.caption(f"Processed field: `{active.draft_dir}`")
        elif active.draft_id is not None:
            st.warning(f"Processed field {active.draft_id}: its folder is not under `{Path(cfg.archive_root) / 'Drafts'}`.")

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
            ui_quality_dashboard.render_quality_dashboard(draft_id=active.draft_id, draft_dir=active.draft_dir)
        elif view == views[2]:
            ui_masterstar_qa.render_masterstar_qa(draft_dir=active.draft_dir)
        elif view == views[3]:
            render_aperture_photometry(cfg=cfg, draft_id=active.draft_id, draft_dir=active.draft_dir)
        elif view == views[4]:
            from ui_epsf_dashboard import render_epsf_dashboard

            render_epsf_dashboard(draft_dir=active.draft_dir, cfg=cfg, draft_id=active.draft_id)
        else:
            render_variability_dashboard(cfg, draft_dir=active.draft_dir)
    elif page == "Calibration Library":
        import ui_calibration_library as ui_calibration_library

        ui_calibration_library.render_calibration_library_dashboard(
            calibration_library_root=Path(cfg.calibration_library_root),
            dark_validity_days=int(cfg.masterdark_validity_days),
            flat_validity_days=int(cfg.masterflat_validity_days),
            db=pipeline.db,
            data_root=Path(cfg.data_root),
        )
    elif page == "Database Explorer":
        ui_database_explorer.render_database_explorer(pipeline=pipeline)
    elif page == "Settings":
        import ui_settings

        ui_settings.render_settings_dashboard(cfg, draft_dir=active_draft(cfg.archive_root).draft_dir)

    _vyvar_render_fixed_footer_into(vyvar_footer_ph)


if __name__ == "__main__":
    main()

