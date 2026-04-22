"""Streamlit entrypoint for live and archive views."""

from __future__ import annotations

import contextlib
import html
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from config import AppConfig, save_config_json
from database import DraftTechnicalMetadataError
from infolog import clear_log, ensure_infolog_logging, get_lines, log_event, last_job_snapshot
from importer import smart_import_session, smart_scan_source
from pipeline import AstroPipeline, scan_usb_folder
import ui_calibration as ui_calibration
import ui_database_explorer as ui_database_explorer
import ui_masterstar_qa as ui_masterstar_qa
import ui_select_stars as ui_select_stars
import ui_suspected_lightcurves as ui_suspected_lightcurves
import ui_quality_dashboard as ui_quality_dashboard
import ui_components as ui_components
from platesolve_ui_paths import resolve_draft_directory
from ui_aperture_photometry import render_aperture_photometry
from utils import generate_session_id

# MASTERSTAR: DB fallback (top % FWHM) when explicit UI paths fail to map; no session/UI control.
_DEFAULT_MASTERSTAR_SELECTION_PCT = 10.0


def _vyvar_effective_draft_dir_override() -> Path | None:
    raw = st.session_state.get("vyvar_draft_dir_override")
    if not raw:
        return None
    p = Path(str(raw)).expanduser()
    return p.resolve() if p.is_dir() else None


def _vyvar_execute_preprocess_pending(
    *,
    pending: dict[str, Any],
    ap: Path,
    pipeline: AstroPipeline,
    progress_cb: Any,
) -> None:
    from pipeline import (
        calibrated_paths_for_draft_apply_filters,
        estimate_archive_memory_profile,
        preprocess_calibrated_to_processed,
    )

    ap_root = ap.parent if ap.name.casefold() == "non_calibrated" else ap
    cal = ap_root / "calibrated" / "lights"
    noncal = ap_root / "non_calibrated" / "lights"
    proc_root = ap_root / "processed" / "lights"
    _dqf = pending.get("quality_filter_draft_id")
    _fwhm_lim = float(pending.get("fwhm_limit_px") or 0.0)
    p1: list[Path] = []
    source_dir = cal

    _pp_kw = dict(
        background_method=str(pending.get("background_method", "background2d")),
        poly_order=int(pending.get("poly_order", 2)),
        lacosmic_sigclip=float(pending.get("sigclip", 4.5)),
        lacosmic_objlim=float(pending.get("objlim", 5.0)),
        enable_lacosmic=bool(pending.get("enable_lacosmic", False)),
        enable_background_flattening=bool(
            pending.get("enable_background_flattening", False)
        ),
        reject_fwhm_px=(float(_fwhm_lim) if float(_fwhm_lim) > 0.0 else None),
        reject_elongation=None,
        temporal_sigma_clip=False,
        temporal_sigma=6.0,
        temporal_min_frames=5,
        use_gpu_if_available=False,
        inject_pointing_ra_deg=pending.get("inject_pointing_ra_deg"),
        inject_pointing_dec_deg=pending.get("inject_pointing_dec_deg"),
        inject_pointing_only_if_missing=False,
    )

    if _dqf is not None:
        try:
            drow = pipeline.db.fetch_obs_draft_by_id(int(_dqf)) or {}
            d_is_cal = int(drow.get("IS_CALIBRATED") or 0)
            if d_is_cal == 0 and noncal.exists():
                source_dir = noncal
        except Exception:  # noqa: BLE001
            pass
    if source_dir == cal and (not cal.exists()) and noncal.exists():
        source_dir = noncal

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
                _why.append("FWHM ≤ limit alebo FWHM je NULL")
            raise FileNotFoundError(
                "QC filter: žiadne snímky spĺňajúce " + ", ".join(_why) + "."
            )
        dfs_pp: list[pd.DataFrame] = []
        tot_pp = len(p1)
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
                preprocess_calibrated_to_processed(
                    calibrated_root=source_dir,
                    processed_root=proc_root,
                    only_paths=p1,
                    progress_cb=_pcb_pp(off_pp),
                    db=pipeline.db,
                    draft_id=(int(_dqf) if _dqf is not None else None),
                    **_pp_kw,
                )
            )
            off_pp += len(p1)
        df = pd.concat(dfs_pp, ignore_index=True) if dfs_pp else pd.DataFrame()
    else:
        if not source_dir.exists():
            raise FileNotFoundError("Missing source lights directory. Run calibration/import first.")
        if _fwhm_lim > 0:
            log_event(
                f"Detrend: draft_id chýba — FWHM limit sa neaplikuje z DB; spracúvam všetky FITS v {source_dir}."
            )
        df = preprocess_calibrated_to_processed(
            calibrated_root=source_dir,
            processed_root=proc_root,
            progress_cb=progress_cb,
            db=pipeline.db,
            draft_id=None,
            **_pp_kw,
        )
    try:
        if not df.empty:
            proc_root.mkdir(parents=True, exist_ok=True)
            df.to_csv(proc_root / "qc_metrics.csv", index=False)
    except Exception:  # noqa: BLE001
        pass
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
        _root_pp = str(proc_root)
        st.session_state["vyvar_last_job_summary"] = {
            "kind": "preprocess",
            "rows": int(len(df)),
            "rejected": rej_pp,
            "root": _root_pp,
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
                st.error(f"Zápis QC snapshotu / stavu draftu zlyhal: {exc}")


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
    _draft_ps = pending.get("draft_id")
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
        max_control_points=int(pending.get("max_control_points", 180)),
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
        n_comparison_stars=int(pending.get("n_comparison_stars", 150)),
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
        if fln not in mfb:
            mfb[fln] = pth
        elif pth is not None and mfb[fln] is None:
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
            "status_detail": "Pripravený — spusti úlohu na záložke VARSTREM.",
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
    proc = html.escape(str(fs.get("process") or "—"))
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
            pct_html = '<span class="vyvar-ft-pill-run">…</span>'
    elif not running:
        pct_html = '<span class="vyvar-ft-pill-idle">Idle</span>'

    file_seg = ""
    if cfile:
        file_seg = (
            f'<span class="vyvar-ft-seg"><span class="vyvar-ft-k">Súbor</span>'
            f'<span class="vyvar-ft-v">{cfile}</span></span>'
        )
    step_seg = ""
    if step:
        step_seg = (
            f'<span class="vyvar-ft-seg"><span class="vyvar-ft-k">Krok</span>'
            f'<span class="vyvar-ft-v">{step}</span></span>'
        )

    inner = (
        f"{pct_html}"
        f'<span class="vyvar-ft-seg"><span class="vyvar-ft-k">Proces</span>'
        f'<span class="vyvar-ft-v">{proc}</span></span>'
        f'<span class="vyvar-ft-seg"><span class="vyvar-ft-k">Stav</span>'
        f'<span class="vyvar-ft-v">{detail or "—"}</span></span>'
        f"{file_seg}{step_seg}"
    )
    placeholder.markdown(
        _VYVAR_FOOTER_CSS + f'<div class="vyvar-footer-bar" data-testid="vyvar-footer-bar">{inner}</div>',
        unsafe_allow_html=True,
    )


def render_live_view(
    pipeline: AstroPipeline,
    cfg: AppConfig,
    *,
    footer_placeholder: Any | None = None,
) -> None:
    _ = cfg  # reserved for future VARSTREM options
    _sess_key = "vyvar_varstrem_session_id"
    if _sess_key not in st.session_state:
        st.session_state[_sess_key] = generate_session_id()

    st.subheader("VARSTREM")
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

    with st.expander("📂 Definovat zdrojová data", expanded=True):
        source_root = st.text_input(
            "Source Directory",
            value=str(cfg.archive_root),
            help="Example: USB session root (any structure; will be scanned recursively).",
        )
        st.caption(
            "Pre vyhľadanie masterov v Calibration Library zvoľ **set** (kamera + ďalekohľad). "
            "Zodpovedajúce alebo všeobecné (bez setu) záznamy z knižnice sa použijú podľa pravidiel v DB."
        )
        col_scan_eq, col_scan_tel = st.columns(2)
        with col_scan_eq:
            eq_placeholder = "(žiadna kamera v DB)" if not eq_labels else eq_labels[0]
            import_equipment_label = st.selectbox(
                "Equipment (knižnica)",
                options=eq_labels if eq_labels else [eq_placeholder],
                key="vyvar_varstrem_equipment",
            )
        with col_scan_tel:
            tel_placeholder = "(žiadny ďalekohľad v DB)" if not tel_labels else tel_labels[0]
            import_telescope_label = st.selectbox(
                "Telescope (knižnica)",
                options=tel_labels if tel_labels else [tel_placeholder],
                key="vyvar_varstrem_telescope",
            )
        import_equipment_id = (
            int(equipment_options[import_equipment_label])
            if eq_labels and import_equipment_label in equipment_options
            else 1
        )
        import_telescope_id = (
            int(telescope_options[import_telescope_label])
            if tel_labels and import_telescope_label in telescope_options
            else 1
        )
        if st.button("🔍 Scan Source", type="primary"):
            try:
                plan = smart_scan_source(
                    source_root=source_root,
                    calibration_library_root=cfg.calibration_library_root,
                    masterdark_validity_days=dark_validity_days,
                    masterflat_validity_days=flat_validity_days,
                    db=pipeline.db,
                    id_equipments=import_equipment_id,
                    id_telescope=import_telescope_id,
                )
                st.session_state["vyvar_smart_plan"] = plan
                st.session_state.pop("vyvar_post_cal_archive_path", None)
                st.session_state.pop("vyvar_post_cal_plan_source", None)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Scan failed: {exc}")
                st.session_state.pop("vyvar_smart_plan", None)

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
            pass

        # Mandatory validation: at least one Light found in the whole tree
        lights_bad = any(r.type == "Lights" and r.status in ("missing", "empty") for r in plan.scan_rows)
        if lights_bad:
            st.error(
                "Nenašli sa žiadne light snímky (žiadny priečinok lights/Lights ani FITS s IMAGETYP "
                "light/object/science v koreňovom adresári). Import zrušený."
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
                "Na zdroji sú **surové** dark alebo flat. Master dark/flat si priprav a skontroluj "
                "v **Calibration Library** (knižnica + prípadne generovanie tam), potom znova **Scan Source**."
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
        st.subheader("Multi-pozorovanie (Filter + Exp)")
        st.caption("Stav kalibrácie a plate solvu podľa kombinácie filter + čas expozície (všetky binningy v skupine).")
        _obs_df = ui_calibration.build_multi_observation_status_dataframe(
            plan,
            archive_path=Path(_post_ap) if _use_done else None,
            cal_phase="done" if _use_done else "preview",
        )
        st.dataframe(_obs_df, use_container_width=True, hide_index=True)

        if ogroups:
            with st.expander("Kalibračné mastre a režim binningu", expanded=False):
                st.caption(
                    "Stĺpec **Binning Mode**: napr. „Resampling 1x1 -> 2x2“, ak je light 2×2 a master 1×1 v knižnici."
                )
                st.dataframe(
                    ui_calibration.build_master_calibration_files_dataframe(plan),
                    use_container_width=True,
                    hide_index=True,
                )
            with st.expander("Technický prehľad skupín (binning · mierka)", expanded=False):
                _ogr = []
                for gk, g in sorted(ogroups.items(), key=lambda x: x[0]):
                    _mf = (getattr(plan, "masterflat_by_obs_key", None) or {}).get(gk)
                    _md = (getattr(plan, "dark_master_by_obs_key", None) or {}).get(gk)
                    _ogr.append(
                        {
                            "Skupina": gk,
                            "Filter": g.get("filter"),
                            "Exp (s)": g.get("exposure_s"),
                            "Binning": g.get("binning"),
                            "Snímok": len(g.get("light_paths") or []),
                            "Master flat": "áno" if _mf else "nie",
                            "Master dark": "áno" if _md else "nie",
                            "″/px": g.get("plate_scale_arcsec_per_px"),
                        }
                    )
                st.dataframe(pd.DataFrame(_ogr), use_container_width=True, hide_index=True)

        _fb_prompts = getattr(plan, "flat_fallback_prompts", None) or []
        if _fb_prompts:
            with st.expander("Chýbajúce Master Flat — výber náhrady", expanded=True):
                st.caption(
                    "Pre skupiny bez flatu môžete použiť master flat z inej filtračnej skupiny "
                    "(rovnaký čas a binning), alebo nechať preskočené (záber ide do non_calibrated)."
                )
                for p in _fb_prompts:
                    gk = str(p.get("group_key") or "")
                    alts = list(p.get("alternatives") or [])
                    opts: list[str] = ["__skip__"] + [str(a) for a in alts]
                    og = ogroups

                    def _fmt(v: str) -> str:
                        if v == "__skip__":
                            return "Preskočiť (non_calibrated pre túto skupinu)"
                        gg = og.get(v, {})
                        return f"Použiť flat z filtra {gg.get('filter', '?')} ({v})"

                    st.selectbox(
                        str(p.get("message_sk") or "Chýba Master Flat."),
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
        label = "🚀 Create Archive & Do Calibration (Quick Look Draft)" if plan.quick_look else "🚀 Create Archive & Do Calibration"
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
                    process="Import archívu",
                    status_detail="Zapisujem draft a kopírujem súbory…",
                    pct=0,
                )
                result = smart_import_session(
                    plan=plan,
                    pipeline=pipeline,
                    id_equipment=import_equipment_id,
                    id_telescope=import_telescope_id,
                    id_location=1,
                )
                st.session_state["vyvar_last_import_equipment_id"] = int(import_equipment_id)
                st.session_state["vyvar_last_import_result"] = result
                st.session_state["vyvar_last_import_plan"] = plan
                if getattr(result, "draft_id", None) is not None:
                    st.session_state["vyvar_last_draft_id"] = int(result.draft_id)
                log_event(
                    f"Import hotový — draft {result.draft_id}, archív {result.archive_path}"
                )
                try:
                    _sat_eq = pipeline.db.get_equipment_saturation_adu(int(import_equipment_id))
                    log_event(
                        f"Equipment ID {import_equipment_id}: SATURATE_ADU v DB = "
                        f"{_sat_eq if _sat_eq is not None else '(NULL — pri MASTERSTAR katalógu: hlavička → BITPIX → Settings fallback)'}"
                    )
                except Exception:  # noqa: BLE001
                    pass
                if result.warnings:
                    for w in result.warnings:
                        log_event(f"Import varování: {w}")

                # Immediately run calibration to create /calibrated (progress v pätičke, nie samostatný bar)
                def _cal_progress(i: int, total: int, msg: str) -> None:
                    pct = int(round(100 * (i / max(total, 1))))
                    _vyvar_footer_set(
                        footer_placeholder,
                        running=True,
                        process="Kalibrácia svetiel → /calibrated",
                        status_detail=msg,
                        pct=pct,
                        current_file=_vyvar_guess_filename_from_progress(msg),
                        step=f"{i} / {total}",
                    )

                _vyvar_footer_set(
                    footer_placeholder,
                    running=True,
                    process="Kalibrácia svetiel → /calibrated",
                    status_detail="Import hotový — aplikujem dark/flat…",
                    pct=0,
                    step="",
                )
                with st.spinner("Kalibrácia prebieha — podrobnosti v pätičke; tabuľka Multi-pozorovanie sa aktualizuje po dokončení."):
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
                    process="Import + kalibrácia",
                    status_detail=f"Hotovo — spracovaných snímok (kalibrácia): {_nproc}",
                    pct=100,
                    step="",
                )
                log_event("Kalibrácia hotová — `/calibrated`")
                last_job_snapshot(cal_out)
                st.session_state["vyvar_status_calibrated"] = True
                _did_imp = getattr(result, "draft_id", None)
                if _did_imp is not None:
                    st.session_state["vyvar_pending_job"] = {
                        "kind": "analyze",
                        "label": "QC analysis (RAM calibration)…",
                        "archive_path": str(result.archive_path),
                        "draft_id": int(_did_imp),
                        "equipment_id": int(import_equipment_id),
                        "roundness_reject_above": float(st.session_state.get("max_roundness_error", 1.25)),
                    }
                    st.success("Import a kalibrácia hotové — spúšťam QC analýzu v RAM (Analyze)…")
                else:
                    st.success("Import a kalibrácia hotové.")
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
                    process="Import / kalibrácia",
                    status_detail=f"Zlyhalo: {exc}",
                    pct=None,
                )
                st.error(f"Import failed: {exc}")
                st.exception(exc)

    # Calibration runs after import; QC RAM analyze is queued at end of the import handler.

    st.markdown("---")
    st.subheader("Pre-processing & Detrending")
    st.caption(
        "Runs on `/calibrated` → writes passing frames to `/processed/lights` with `proc_` prefix. "
        "**Default is fully automatic:** star detection for QC adapts per frame (threshold sweep, polarity, data-driven PSF guess); "
        "frame **rejection is off** until you set thresholds under Advanced."
    )

    # Run long jobs in a "minimal view" to avoid Streamlit rerun duplication artefacts.
    pending = st.session_state.get("vyvar_pending_job")
    if pending:
        vyvar_init_footer_state_if_missing()
        st.session_state["vyvar_footer_state"] = {
            "running": True,
            "process": str(pending.get("label") or pending.get("kind") or "úloha"),
            "status_detail": "Štartujem…",
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
            "only_masterstar_platesolve",
        }
        progress_bar = None if _hide_inline_job_status else st.progress(0, text="Starting…")
        _status_ctx: Any = (
            contextlib.nullcontext(None)
            if _hide_inline_job_status
            else st.status(pending.get("label", "Running…"), expanded=False)
        )
        with _status_ctx as stt:
            try:
                if ap is None or not ap.exists():
                    raise FileNotFoundError("Missing/invalid archive path for job.")
                log_event(f"— Spúšťam: {pending.get('kind')} — {ap}")

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
                        "process": str(pending.get("label") or pending.get("kind") or "úloha"),
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
                            "Chýba kalibračný plán importu — znova spusti **Create Archive & Do Calibration** "
                            "(session potrebuje `vyvar_last_import_plan` s mastrami)."
                        )
                    _draft_a = pending.get("draft_id")
                    if _draft_a is None:
                        _draft_a = st.session_state.get("vyvar_last_draft_id")
                    if _draft_a is None:
                        raise FileNotFoundError("Chýba Draft ID — najprv importuj draft do archívu.")
                    _eq_a = pending.get("equipment_id")
                    if _eq_a is None:
                        _eq_a = st.session_state.get("vyvar_last_import_equipment_id")
                    _mem_prof = estimate_archive_memory_profile(ap)
                    st.session_state["vyvar_memory_profile"] = _mem_prof
                    _qcm = _mem_prof.get("qc_analyze") or {}
                    if _qcm:
                        log_event(
                            f"Odhad RAM (pred analyze): QC špička ~{_qcm.get('estimated_peak_human', '?')}, "
                            f"snímky: {_qcm.get('n_files', 0)}, voľné: {_mem_prof.get('available_ram_human', '?')}"
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
                        "Analýza kalibrácie v RAM (žiadny zápis kalibrovaných FITS) — môže chvíľu trvať…"
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

                elif pending.get("kind") in ("masterstar_catalog_only", "only_masterstar_platesolve"):
                    from pipeline import (
                        _equipment_saturate_adu_from_db,
                        estimate_archive_memory_profile,
                        generate_masterstar_and_catalog,
                    )

                    _pls_only_job = pending.get("kind") == "only_masterstar_platesolve"
                    _ms_fits_only = bool(pending.get("masterstar_fits_only")) and not _pls_only_job
                    _ms_skip_build = bool(pending.get("masterstar_skip_build"))
                    if _ms_fits_only:
                        log_event(
                            f"— MASTERSTAR (len FITS, bez solve): ui_action={pending.get('masterstar_ui_action')!r}, "
                            "zápis do platesolve/…"
                        )
                    elif _ms_skip_build:
                        log_event(
                            "— MASTERSTAR platesolve + katalóg: vstup = existujúci platesolve/MASTERSTAR.fits "
                            f"(skip build z processed), draft_id={pending.get('draft_id')!r}."
                        )
                    elif _pls_only_job:
                        log_event(
                            "— ONLY MASTER: build (ak treba) + VYVAR plate-solve — bez katalógov a zarovnania "
                            f"(draft_id={pending.get('draft_id')!r})."
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

                            _al = _iter_vyvar_fits_under(resolve_masterstar_input_root(ap))
                            if _al:
                                _cm = pipeline.db.get_combined_metadata(_al[0], int(_draft_ms))
                                if _cm.get("saturate_adu") is not None:
                                    _equip_sat = _cm["saturate_adu"]
                        except Exception:  # noqa: BLE001
                            pass
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
                        n_comparison_stars=int(pending.get("n_comparison_stars", 150)),
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
                        masterstar_platesolve_only=bool(_pls_only_job),
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
                    if _pls_only_job:
                        log_event(
                            f"✅ ONLY MASTER hotové (len plate-solve): {_msf or outp.get('masterstar_fits')}"
                        )
                    elif _ms_fits_only:
                        log_event(
                            f"✅ MASTERSTAR FITS dokončené [{pending.get('masterstar_ui_action', '?')}]: {_msf or outp.get('masterstar_fits')} "
                            "(plate-solve a katalóg preskočené)."
                        )
                    elif _ms_skip_build and not _ms_fits_only:
                        log_event(
                            f"✅ MASTERSTAR plate-solve + katalóg hotové: {_msf or outp.get('masterstar_fits')} "
                            f"(matched={int(outp.get('catalog_matched', 0) or 0)})."
                        )
                else:
                    raise ValueError("Unknown job kind.")

                if stt is not None:
                    stt.update(label="Done.", state="complete")
                out = st.session_state.get("vyvar_last_job_output")
                if isinstance(out, dict) and not out.get("error"):
                    last_job_snapshot(out)
                st.session_state["vyvar_footer_state"] = {
                    "running": False,
                    "process": str(pending.get("label") or pending.get("kind") or "úloha"),
                    "status_detail": "Hotovo.",
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
                    "process": str(pending.get("label") or pending.get("kind") or "úloha"),
                    "status_detail": f"Zlyhalo: {_err_msg}",
                    "pct": None,
                    "current_file": "",
                    "step": "",
                }
                if footer_placeholder is not None:
                    _vyvar_render_fixed_footer_into(footer_placeholder)
            except Exception as exc:  # noqa: BLE001
                _em = str(exc).strip()
                _err_msg = _em if _em else f"{type(exc).__name__} (bez správy)"
                if stt is not None:
                    stt.update(label=f"Failed: {_err_msg}", state="error")
                st.session_state["vyvar_last_job_output"] = {
                    "error": _err_msg,
                    "error_type": type(exc).__name__,
                }
                log_event(f"Job zlyhal: {_err_msg} [{type(exc).__name__}]")
                st.session_state["vyvar_footer_state"] = {
                    "running": False,
                    "process": str(pending.get("label") or pending.get("kind") or "úloha"),
                    "status_detail": f"Zlyhalo: {_err_msg}",
                    "pct": None,
                    "current_file": "",
                    "step": "",
                }
                if footer_placeholder is not None:
                    _vyvar_render_fixed_footer_into(footer_placeholder)

        # Clear pending job and rerun into the normal UI view.
        st.session_state.pop("vyvar_pending_job", None)
        st.rerun()

    last_res = st.session_state.get("vyvar_last_import_result")
    default_ap = ""
    if last_res and getattr(last_res, "archive_path", None):
        default_ap = str(last_res.archive_path)
    archive_path_override = st.text_input(
        "Archive path for preprocessing (optional override)",
        value=default_ap,
        help="If empty, uses the last imported archive path from this UI session.",
    )

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

    st.markdown("**QC vstupy (Krok 2)**")
    st.caption(
        "Po **Analyze** (automaticky po importe) sa doplní FWHM a stred poľa z `OBS_FILES`. "
        "**FWHM limit** nastav posuvníkom v záložke **Quality Dashboard**; **MAKE MASTERSTAR** používa ten istý prah spolu s RA/DE."
    )
    _q1, _q2, _q3 = st.columns(3)
    with _q1:
        _fwhm_cur = float(st.session_state.get("fwhm_threshold", 0.0))
        st.caption("FWHM limit (px)")
        st.metric(
            label="Aktuálny prah (session)",
            value=f"{_fwhm_cur:.3f} px",
        )
        st.caption("Posuvník: záložka **Quality Dashboard**.")
    with _q2:
        st.number_input(
            "Center RA (deg, ICRS)",
            min_value=0.0,
            max_value=360.0,
            step=0.0001,
            format="%.6f",
            key=_ra_key,
            on_change=ui_components.persist_draft_center_on_change,
            args=(pipeline.db, st.session_state.get("vyvar_last_draft_id")),
            help="Stred poľa v stupňoch (ICRS). Po Analyze medián z DB; upraviteľné pred MAKE MASTERSTAR.",
        )
    with _q3:
        st.number_input(
            "Center DE (deg, ICRS)",
            min_value=-90.0,
            max_value=90.0,
            step=0.0001,
            format="%.6f",
            key=_de_key,
            on_change=ui_components.persist_draft_center_on_change,
            args=(pipeline.db, st.session_state.get("vyvar_last_draft_id")),
            help="Deklinácia v stupňoch (ICRS). Po Analyze medián z DB.",
        )
    st.session_state["center_ra"] = float(st.session_state.get(_ra_key, 0.0))
    st.session_state["center_de"] = float(st.session_state.get(_de_key, 0.0))
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
            log_event(f"Draft center save skipped: {_exc_center!s}")

    with st.expander("Advanced preprocessing (optional tuning)", expanded=False):
        st.caption(
            "L.A.Cosmic a model pozadia pri **MAKE MASTERSTAR**. Výber snímok: `OBS_FILES` (IS_REJECTED + FWHM limit)."
        )
        enable_lacosmic = st.checkbox(
            "Enable L.A.Cosmic (AstroScrappy)",
            value=bool(st.session_state.get("vyvar_enable_lacosmic", False)),
            key="vyvar_enable_lacosmic",
        )
        enable_background_flattening = st.checkbox(
            "Enable Background Flattening (Model pozadia)",
            value=bool(st.session_state.get("vyvar_enable_background_flattening", False)),
            key="vyvar_enable_background_flattening",
        )
        colp1, colp2 = st.columns(2)
        with colp1:
            background_method = st.selectbox("Background model", options=["background2d", "poly2d"], index=0)
        with colp2:
            poly_order = st.selectbox("Poly order", options=[2, 3], index=0, disabled=(background_method != "poly2d"))

        colc1, colc2 = st.columns(2)
        with colc1:
            sigclip = st.number_input("L.A.Cosmic sigclip", min_value=1.0, max_value=20.0, value=4.5, step=0.1)
        with colc2:
            objlim = st.number_input("L.A.Cosmic objlim", min_value=1.0, max_value=20.0, value=5.0, step=0.1)

    st.markdown("**Pre-processing**")
    st.caption(
        "**Analyze** sa spúšťa automaticky po **Create Archive & Do Calibration** (QC v RAM → metriky do `OBS_FILES`). "
        "**MAKE MASTERSTAR** — najprv detrend (`/calibrated` → `/processed/lights`, **IS_REJECTED=0**, **FWHM ≤ limit** ak limit > 0), "
        "potom **VYVAR** plate solve (Gaia DR3), zarovnanie a sidecar CSV v `detrended_aligned/lights/`. "
        "**Quality Dashboard**: FWHM posuvník, grafy, manuálne zamietnutie. "
        "Paralelizmus: automaticky (CPU/RAM) alebo env `VYVAR_PARALLEL_WORKERS` (legacy QC/per-frame env tiež)."
    )
    _rt = ui_calibration.draft_runtime_status(
        pipeline.db,
        draft_id=st.session_state.get("vyvar_last_draft_id"),
        archive_path=archive_path_override if archive_path_override else None,
    )
    _rt_an = bool(st.session_state.get("vyvar_status_analyzed", False) or _rt.get("analyzed"))
    _rt_cal = bool(st.session_state.get("vyvar_status_calibrated", False) or _rt.get("calibrated"))
    st.session_state["vyvar_status_analyzed"] = _rt_an
    st.session_state["vyvar_status_calibrated"] = _rt_cal
    _s1, _s2 = st.columns(2)
    with _s1:
        st.markdown(f"**Analyzed:** {'Yes' if _rt_an else 'No'}")
    with _s2:
        st.markdown(f"**Calibrated:** {'Yes' if _rt_cal else 'No'}")

    # MASTERSTAR kandidát sa nastavuje v FITS QA (tlačidlo "Použiť ako MASTERSTAR")
    _draft_step3 = st.session_state.get("vyvar_last_draft_id")
    _db_masterstar_path: str | None = None
    if _draft_step3 is not None:
        try:
            _drow = pipeline.db.fetch_obs_draft_by_id(int(_draft_step3))
            if _drow is not None:
                _ms_db_raw = str(
                    _drow.get("MASTERSTAR_FITS_PATH")
                    or _drow.get("MASTERSTAR_PATH")
                    or ""
                ).strip()
                if _ms_db_raw:
                    _db_masterstar_path = _ms_db_raw
                    st.session_state["vyvar_db_masterstar_path"] = _db_masterstar_path
        except Exception:  # noqa: BLE001
            pass

    st.markdown("---")
    st.subheader("MAKE MASTERSTAR")
    st.caption(
        "Jeden krok: detrend na `/processed/lights`, potom **VYVAR** plate solve (lokálna Gaia DR3), zarovnanie "
        "a sidecar CSV v `detrended_aligned/lights/`. **Referenčný MASTERSTAR** vyber v **FITS QA** "
        "(najlepší snímok → uloží sa do draftu / session). "
        "**ONLY MASTER** — len zostavenie `MASTERSTAR.fits` + plate-solve (bez preprocessu, bez zarovnania a CSV); "
        "vyžaduje už existujúce `processed/lights` a rovnaký výber kandidáta ako pri plnom kroku."
    )
    try:
        from pipeline import get_auto_fov

        _did_fov = st.session_state.get("vyvar_last_draft_id")
        _eq_fov = st.session_state.get("vyvar_last_import_equipment_id")
        _auto_fov = get_auto_fov(
            archive_path=Path(str(archive_path_override).strip()) if archive_path_override else None,
            masterstar_path=Path(_db_masterstar_path) if _db_masterstar_path else None,
            database_path=cfg.database_path,
            equipment_id=int(_eq_fov) if _eq_fov is not None else None,
            draft_id=int(_did_fov) if _did_fov is not None else None,
        )
    except Exception:  # noqa: BLE001
        _auto_fov = None
    if _auto_fov is None:
        st.caption("FOV: automaticky (z FITS hlavičky / WCS po plate-solve).")
    else:
        st.caption(f"FOV (auto): približne **{float(_auto_fov):.2f}°** (uhlopriečka).")
    plate_fov_ui = float(_auto_fov) if _auto_fov is not None else float(cfg.plate_solve_fov_deg)

    # Geometry details moved to automatic FOV; keep UI clean.

    # MASTERSTAR workflow is mandatory in Step 3 (no UI toggle).
    _dao_fwhm_default = float(max(1.0, min(8.0, float(getattr(cfg, "sips_dao_fwhm_px", 2.5)))))
    _dao_sigma_default = float(max(1.0, min(10.0, float(getattr(cfg, "sips_dao_threshold_sigma", 3.5)))))
    faintest_mag_limit_auto: float | None = None
    sat_level = 0.95
    st.session_state["dao_fwhm_px"] = float(_dao_fwhm_default)
    st.session_state["dao_threshold_sigma"] = float(_dao_sigma_default)
    # Fixed / automated internal params (must match sensible pipeline defaults: 0 would skip DAO on non-ref frames)
    cat_match_arc = 2.0
    max_cat_rows = 12000
    max_extra_ps = 0
    min_stars = 100
    max_stars = 500
    max_ctrl = 180

    _mp = st.session_state.get("vyvar_memory_profile")
    if isinstance(_mp, dict) and _mp:
        _parts: list[str] = []
        _qca = _mp.get("qc_analyze")
        if isinstance(_qca, dict) and _qca.get("n_files", 0):
            _parts.append(
                f"Analyze (QC): špička ~{_qca.get('estimated_peak_human', '?')} · {_qca.get('n_files', 0)} calibrated"
            )
        _prh = _mp.get("platesolve_ram_handoff")
        if isinstance(_prh, dict) and _prh.get("n_files", 0):
            _flag = ""
            if _prh.get("estimate_below_available_ram") is True:
                _flag = " — odhad ≤ voľná RAM"
            elif _prh.get("estimate_below_available_ram") is False:
                _flag = " — pozor: odhad môže presiahnuť voľnú RAM"
            _parts.append(
                f"RAM handoff: ~{_prh.get('estimated_total_conservative_human', '?')} · "
                f"{_prh.get('n_files', 0)} processed/detrended{_flag}"
            )
        if _parts:
            st.caption(
                "Odhad pamäte (z hlavičiek FITS + hrubé faktory): "
                + " · ".join(_parts)
                + f" · voľné v OS: {_mp.get('available_ram_human', 'neznáme')}"
            )

    platesolve_equipment_id = int(st.session_state.get("vyvar_last_import_equipment_id") or 1)

    _build_ms = True
    _ms_paths = list(st.session_state.get("vyvar_masterstar_candidate_paths") or [])
    if not _ms_paths:
        _db_top1 = str(st.session_state.get("vyvar_ms_candidate_top1_path") or "").strip()
        if _db_top1 and Path(_db_top1).is_file():
            _ms_paths = [_db_top1]
    if not _ms_paths:
        _did_ms = st.session_state.get("vyvar_last_draft_id")
        _ap_ms = Path(str(archive_path_override).strip()) if archive_path_override else None
        if _did_ms is not None and _ap_ms is not None and _ap_ms.is_dir():
            try:
                from pipeline import resolve_obs_file_to_processed_fits

                _p_draft = pipeline.db.get_obs_draft_masterstar_path(int(_did_ms))
                if _p_draft:
                    _hit = resolve_obs_file_to_processed_fits(_ap_ms, str(_p_draft).strip())
                    if _hit is not None and _hit.is_file():
                        _ms_paths = [str(_hit.resolve())]
                    elif Path(_p_draft).is_file():
                        _ms_paths = [str(Path(_p_draft).resolve())]
            except Exception:  # noqa: BLE001
                pass

    col_ps_go, col_ps_only = st.columns(2)

    def _gaia_db_ok_for_masterstar() -> bool:
        try:
            from database import validate_gaia_db_schema

            _gdb = str(getattr(cfg, "gaia_db_path", "") or "").strip()
            ok, msg = validate_gaia_db_schema(_gdb)
            if not ok:
                st.error("❌ Gaia DR3 databáza nenájdená. Prosím, nastavte cestu v Settings.")
                st.caption(f"Detail: {msg}")
                return False
        except Exception:
            st.error("❌ Gaia DR3 databáza nenájdená. Prosím, nastavte cestu v Settings.")
            return False
        return True

    def _masterstar_pointing_and_hash() -> tuple[Any, Any, bool, int | None, str | None]:
        def _sess_float_ms(key: str) -> float:
            try:
                return float(st.session_state.get(key, float("nan")))
            except (TypeError, ValueError):
                return float("nan")

        _did = st.session_state.get("vyvar_last_draft_id")
        _ira_ui = _sess_float_ms(_ra_key)
        _ide_ui = _sess_float_ms(_de_key)
        from pipeline import resolve_preprocess_target_coordinates

        _ira, _ide = resolve_preprocess_target_coordinates(
            db=pipeline.db,
            draft_id=(int(_did) if _did is not None else None),
            ui_ra_deg=_ira_ui,
            ui_dec_deg=_ide_ui,
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
        if not _coords_ok:
            st.info(
                "Bez platného stredu RA/Dec (draft, UI alebo 0/0) sa do processed FITS nevpíše VYTARG; "
                "plate-solve použije **súradnice z hlavičky** referenčného snímku alebo **blind solver**."
            )
        _did_int = int(_did) if _did is not None else None
        _ph = None
        if _did_int is not None:
            try:
                from pipeline import generate_observation_hash

                _ph = generate_observation_hash(pipeline.db, _did_int)
                st.session_state["vyvar_observation_processing_hash"] = _ph
            except Exception:  # noqa: BLE001
                _ph = None
        return _ira, _ide, _coords_ok, _did_int, _ph

    with col_ps_go:
        if st.button(
            "MAKE MASTERSTAR",
            type="primary",
            help="Detrend (/processed/lights) + plate solve + zarovnanie + per-frame CSV. Referenčný snímok z FITS QA.",
        ):
            ap = Path(archive_path_override) if archive_path_override else None
            if ap is None:
                st.warning("Please provide an archive path (or Import to Archive first).")
            elif not _gaia_db_ok_for_masterstar():
                pass
            else:
                _ira, _ide, _coords_ok, _did_int, _ph = _masterstar_pointing_and_hash()
                _rfw_ui = float(st.session_state.get("fwhm_threshold", 0.0))
                _rfw = float(_rfw_ui) if _rfw_ui > 0.0 else 0.0
                if _rfw_ui > 0.0:
                    log_event(f"DEBUG: MAKE MASTERSTAR FWHM limit (strict): UI={_rfw_ui:.3f} px")

                _j_ms: dict[str, Any] = {
                    "kind": "make_masterstar",
                    "label": "MAKE MASTERSTAR running…",
                    "archive_path": str(ap),
                    "background_method": str(background_method),
                    "poly_order": int(poly_order),
                    "sigclip": float(sigclip),
                    "objlim": float(objlim),
                    "enable_lacosmic": bool(enable_lacosmic),
                    "enable_background_flattening": bool(enable_background_flattening),
                    "fwhm_limit_px": _rfw,
                    "inject_pointing_ra_deg": (float(_ira) if _coords_ok else None),
                    "inject_pointing_dec_deg": (float(_ide) if _coords_ok else None),
                    "quality_filter_draft_id": (_did_int if _did_int is not None else None),
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
                    "n_comparison_stars": 150,
                    "faintest_mag_limit": faintest_mag_limit_auto,
                    "dao_threshold_sigma": float(_dao_sigma_default),
                    "dao_fwhm_px": float(_dao_fwhm_default),
                    "id_equipment": int(platesolve_equipment_id),
                    "draft_id": st.session_state.get("vyvar_last_draft_id"),
                    "catalog_local_gaia_only": True,
                    "build_masterstar_and_catalogs": True,
                    "masterstar_candidate_paths": (_ms_paths if _build_ms else []),
                    "masterstar_selection_pct": float(_DEFAULT_MASTERSTAR_SELECTION_PCT),
                }
                if _did_int is not None and _ph:
                    _j_ms["processing_hash"] = _ph
                    _j_ms["overwrite_qc_processing"] = True
                st.session_state["vyvar_pending_job"] = _j_ms
                st.rerun()

    with col_ps_only:
        if st.button(
            "ONLY MASTER",
            type="secondary",
            help="Bez preprocessu a zarovnania: z ``/processed/lights`` zostaví ``platesolve/MASTERSTAR.fits``, "
            "spustí výhradne VYVAR plate-solve (Gaia DR3) a skončí — bez CSV katalógu a per-frame krokov. Na testy.",
        ):
            ap = Path(archive_path_override) if archive_path_override else None
            if ap is None:
                st.warning("Please provide an archive path (or Import to Archive first).")
            elif not _gaia_db_ok_for_masterstar():
                pass
            else:
                _ira, _ide, _coords_ok, _did_int, _ph = _masterstar_pointing_and_hash()
                _plan_om = st.session_state.get("vyvar_last_import_plan")
                _md_om = (
                    Path(_plan_om.dark_master)
                    if _plan_om and getattr(_plan_om, "dark_master", None)
                    else None
                )
                if _md_om is not None and not _md_om.exists():
                    _md_om = None
                _j_om: dict[str, Any] = {
                    "kind": "only_masterstar_platesolve",
                    "label": "ONLY MASTER (plate-solve)…",
                    "archive_path": str(ap),
                    "astrometry_api_key": "",
                    "platesolve_backend": "vyvar",
                    "plate_solve_fov_deg": float(plate_fov_ui),
                    "catalog_match_max_sep_arcsec": float(cat_match_arc),
                    "saturate_level_fraction": float(sat_level),
                    "max_catalog_rows": int(max_cat_rows),
                    "n_comparison_stars": 150,
                    "faintest_mag_limit": faintest_mag_limit_auto,
                    "dao_threshold_sigma": float(_dao_sigma_default),
                    "dao_fwhm_px": float(_dao_fwhm_default),
                    "id_equipment": int(platesolve_equipment_id),
                    "draft_id": st.session_state.get("vyvar_last_draft_id"),
                    "catalog_local_gaia_only": True,
                    "masterstar_candidate_paths": (_ms_paths if _build_ms else []),
                    "masterstar_selection_pct": float(_DEFAULT_MASTERSTAR_SELECTION_PCT),
                    "masterstar_fits_only": False,
                    "masterstar_skip_build": False,
                    "inject_pointing_ra_deg": (float(_ira) if _coords_ok else None),
                    "inject_pointing_dec_deg": (float(_ide) if _coords_ok else None),
                }
                if _md_om is not None:
                    _j_om["master_dark_path"] = str(_md_om)
                if _did_int is not None and _ph:
                    _j_om["processing_hash"] = _ph
                st.session_state["vyvar_pending_job"] = _j_om
                st.rerun()


def _iter_vyvar_fits_under(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    out: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.casefold() in {".fits", ".fit", ".fts"}:
            out.append(p)
    return sorted(out)


def render_infolog() -> None:
    st.subheader("Infolog")
    st.caption(
        "Podrobný priebeh: moduly `pipeline`, `importer` (úroveň INFO) a kroky dlhých úloh "
        "(QC / preprocessing / plate solve). Buffer sa stratí po reštarte Streamlit."
    )
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Vymazať Infolog", type="secondary", key="vyvar_infolog_clear"):
            clear_log()
            st.rerun()
    with col_b:
        tail = st.number_input(
            "Zobraziť posledných N riadkov",
            min_value=100,
            max_value=8000,
            value=2500,
            step=100,
            key="vyvar_infolog_tail",
        )
    lines = get_lines()
    if tail < len(lines):
        lines = lines[-int(tail) :]
    text = "\n".join(lines) if lines else "(zatiaľ žiadne záznamy — spusti úlohu na VARSTREM)"
    st.code(text, language=None)


def main() -> None:
    cfg = AppConfig()
    cfg.ensure_base_dirs()
    ensure_infolog_logging()
    pipeline = AstroPipeline(config=cfg)

    st.set_page_config(
        page_title="VYVAR - Variable Star Processing",
        page_icon="✨",
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
            "Zadaj absolútnu cestu k priečinku ``draft_XXXXXX`` (musí obsahovať ``platesolve/``) "
            "alebo len číslo draftu z archívu. Použije sa v záložkách Select Stars, Aperture Photometry "
            "a LightCurves — Suspected. Prázdne pole + „Načítať draft“ zruší override. "
            "**DAO-STARS**, **Fotometria** a **Fotometria — diagnostika** sú v **Nastavenia → Nástroje**."
        )
        dcol1, dcol2, dcol3 = st.columns([4, 1, 1])
        with dcol1:
            draft_path_inp = st.text_input(
                "Cesta alebo číslo draftu",
                key="vyvar_draft_path_field",
                placeholder=r"napr. C:\...\Archive\Drafts\draft_000229 alebo 229",
            )
        with dcol2:
            apply_draft = st.button("Načítať draft", key="vyvar_draft_path_apply", type="primary")
        with dcol3:
            _cur = st.session_state.get("vyvar_last_draft_id")
            st.caption(f"ID: **{_cur}**" if _cur is not None else "ID: —")
        if apply_draft:
            s = (draft_path_inp or "").strip()
            if not s:
                st.session_state.pop("vyvar_draft_dir_override", None)
                st.info("Override draftu vymazaný — používa sa umiestnenie z konfigurácie archívu.")
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
                    st.success(f"Draft načítaný: {ddir}")
                    st.rerun()
        ov = _vyvar_effective_draft_dir_override()
        if ov is not None:
            st.caption(f"Aktívny override: `{ov}`")

        _draft_ov = ov
        tabs = st.tabs(
            [
                "VAR-STREM",
                "FITS QA",
                "MASTERSTAR QA",
                "Select Stars",
                "Aperture Photometry",
                "LightCurves — Suspected",
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
            )
        with tabs[2]:
            ui_masterstar_qa.render_masterstar_qa(
                cfg=cfg,
                draft_id=st.session_state.get("vyvar_last_draft_id"),
                pipeline=pipeline,
                draft_dir_override=_draft_ov,
            )
        with tabs[3]:
            ui_select_stars.render_select_stars(
                cfg=cfg,
                draft_id=st.session_state.get("vyvar_last_draft_id"),
                pipeline=pipeline,
                draft_dir_override=_draft_ov,
            )
        with tabs[4]:
            render_aperture_photometry(
                cfg=cfg,
                draft_id=st.session_state.get("vyvar_last_draft_id"),
                pipeline=pipeline,
                draft_dir_override=_draft_ov,
            )
        with tabs[5]:
            ui_suspected_lightcurves.render_suspected_lightcurves(
                cfg=cfg,
                draft_id=st.session_state.get("vyvar_last_draft_id"),
                pipeline=pipeline,
                draft_dir_override=_draft_ov,
            )
        with tabs[6]:
            render_infolog()
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

