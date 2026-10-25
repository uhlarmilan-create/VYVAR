"""MASTERSTAR QA tab (MSQA-2026-10): informative only - how plate solving went and how the
DAO detections match the Gaia catalogue, with a map of the match."""
from __future__ import annotations

import re
from pathlib import Path

import streamlit as st

from config import AppConfig
from pipeline import AstroPipeline
from platesolve_ui_paths import default_bundle_dir, platesolve_bundle_dirs, setup_label
from utils import resolve_draft_dir


def _infer_setup_name_from_masterstar_candidate_path(p: Path) -> str | None:
    """Infer setup name (e.g. R_60_1) from a selected MASTERSTAR candidate light frame path."""
    parts = list(p.parts)
    for i in range(len(parts) - 1):
        if parts[i].lower() == "lights" and re.fullmatch(r"[A-Za-z]+_\d+_\d+", parts[i + 1]):
            return parts[i + 1]
    for seg in parts:
        if re.fullmatch(r"[A-Za-z]+_\d+_\d+", seg):
            return seg
    return None


def _resolve_draft_path(cfg: AppConfig, draft_id: int | None, draft_dir_override: Path | None) -> Path | None:
    """Processed-field root: override -> its folder under Drafts -> last import result; else a manual path."""
    _session_draft_id = st.session_state.get("vyvar_last_draft_id")
    _did = _session_draft_id if _session_draft_id is not None else draft_id
    if draft_dir_override is not None and Path(draft_dir_override).is_dir():
        return Path(draft_dir_override).resolve()
    _resolved = resolve_draft_dir(None, _did, cfg.archive_root, drafts_before_session=True)
    if _resolved and Path(_resolved).is_dir():
        return Path(_resolved)
    last_res = st.session_state.get("vyvar_last_import_result")
    if last_res and getattr(last_res, "archive_path", None) and Path(str(last_res.archive_path)).is_dir():
        return Path(str(last_res.archive_path))
    default_ap = ""
    if _did is not None:
        try:
            from draft_provenance import draft_dir_for_id  # noqa: PLC0415
            default_ap = str(draft_dir_for_id(cfg.archive_root, int(_did)).resolve())
        except (TypeError, ValueError):
            default_ap = ""
    ap_manual = st.text_input("Archive path (processed field)", value=default_ap, key="vyvar_masterstar_ap")
    if not ap_manual.strip():
        st.info(
            "No processed field loaded. Load one above (**Load processed field**), run **RUN VYVAR**, "
            "or enter the path of its folder (`.../Archive/Drafts/<star>_<YYYYMMDD>_<NNN>`)."
        )
        return None
    return Path(ap_manual.strip())


def _fmt(v: float | None, spec: str, unit: str = "") -> str:
    return "-" if v is None else f"{format(v, spec)}{unit}"


def render_masterstar_qa(
    *,
    cfg: AppConfig,
    draft_id: int | None,
    pipeline: AstroPipeline,
    draft_dir_override: Path | None = None,
) -> None:
    if st.session_state.pop("vyvar_masterstar_qa_force_refresh", False):
        st.rerun()
    ap_path = _resolve_draft_path(cfg, draft_id, draft_dir_override)
    if ap_path is None:
        return
    ps_root = ap_path / "platesolve"
    bundles = platesolve_bundle_dirs(ps_root)
    if not bundles:
        st.warning(
            f"No MASTERSTAR in `{ps_root}` (``MASTERSTAR.fits`` + ``masterstars_full_match.csv``). "
            "Run **RUN VYVAR** or **MASTERSTAR only**."
        )
        return

    # The MASTERSTAR candidate picked in the Quality Dashboard selects its setup here.
    preferred_setup: str | None = None
    if draft_id is not None:
        try:
            mp = pipeline.db.get_obs_draft_masterstar_source_path(int(draft_id))
        except Exception:  # noqa: BLE001 - UI only: no DB row -> no preference
            mp = None
        if mp:
            preferred_setup = _infer_setup_name_from_masterstar_candidate_path(Path(str(mp)))

    from masterstar_qa_plot import load_masterstar_qa, render_masterstar_overlay

    names = [p.name for p in bundles]
    pref = default_bundle_dir(ps_root, preferred_name=preferred_setup)
    pref_nm = pref.name if pref is not None else names[0]
    c_title, c_pick = st.columns([3, 1])
    with c_pick:
        pick_nm = st.selectbox(
            "Observation setup (MASTERSTAR)",
            options=names,
            index=names.index(pref_nm) if pref_nm in names else 0,
            format_func=setup_label,
            key="vyvar_msqa_platesolve_setup",
            help="One MASTERSTAR per filter / exposure / binning of the night.",
        )
    setup_dir = ps_root / pick_nm
    try:
        qa = load_masterstar_qa(setup_dir)
    except ValueError as exc:
        st.warning(str(exc))
        return
    fr, so, ma = qa["frame"], qa["solve"], qa["match"]
    with c_title:
        st.subheader("MASTERSTAR QA")
        st.caption(" \u00b7 ".join(x for x in (ap_path.name, fr["object"], fr["date_obs"], setup_label(pick_nm)) if x))

    _how = ", ".join(x for x in (so["solver"], so["mode"]) if x) or "WCS present"
    _odds = f", verified by log-odds {so['log_odds']:.0f}" if so["log_odds"] is not None else ""
    _sip = f"; WCS with SIP order {so['sip_order']}" if so["sip_order"] else ""
    st.success(f"**Plate solved** - {_how}{_odds}{_sip}.")

    st.markdown("#### Plate solving")
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.metric("Frame size", f"{fr['width_px']} \u00d7 {fr['height_px']} px")
        st.caption(f"{fr['mpix']:.1f} Mpx" + (f" \u00b7 binning {fr['binning']}" if fr["binning"] else ""))
    with p2:
        st.metric("Field centre (ICRS)", fr["ra_str"])
        st.caption(f"Dec {fr['dec_str']}")
    with p3:
        st.metric("Resolution", f"{fr['scale_arcsec_px']:.2f}\u2033 / px")
        st.caption(
            f"Field {fr['fov_w_deg']:.2f}\u00b0 \u00d7 {fr['fov_h_deg']:.2f}\u00b0 \u00b7 "
            f"rotation {fr['rotation_deg']:.1f}\u00b0"
        )
    with p4:
        st.metric("Solution fit", _fmt(so["rms_px"], ".2f", " px RMS"))
        if so["n_fit"] is not None:
            st.caption(f"{so['n_fit']} stars in the fit")

    st.markdown("#### DAO \u2194 Gaia matching")
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.metric("Stars detected (DAO)", ma["n_detected"])
    with d2:
        st.metric("Matched to Gaia", ma["n_matched"])
        st.caption(_fmt(ma["matched_pct"], ".1f", " % of detections"))
    with d3:
        st.metric("Detected, no Gaia star", ma["n_unmatched"])
        st.caption("blends, artefacts, faint stars")
    with d4:
        st.metric("Position agreement", _fmt(ma["resid_p50_px"], ".2f", " px"))
        if ma["resid_p95_px"] is not None:
            st.caption(f"median \u00b7 95 % of stars within {ma['resid_p95_px']:.2f} px")

    st.markdown("#### Map")
    m1, m2, m3 = st.columns([1, 1, 2])
    with m1:
        show_gaia = st.checkbox("Show Gaia", value=True, key="vyvar_msqa_show_gaia")
    with m2:
        show_match = st.checkbox("Show matches", value=True, key="vyvar_msqa_show_match")
    with m3:
        view = st.radio(
            "View", ["Full frame", "Detail"], horizontal=True, key="vyvar_msqa_view", label_visibility="collapsed"
        )
    _depth = f" to G {qa['gaia_depth_g']:.1f}" if qa["gaia_depth_g"] is not None else ""
    st.caption(
        ":green[\u25ef] DAO star matched to Gaia \u00a0\u00a0 :orange[\u25ef] DAO star without Gaia match "
        f"\u00a0\u00a0 :blue[\u25a1] Gaia star (catalogue{_depth})"
    )
    img, cap = render_masterstar_overlay(
        qa["image"],
        matched_xy=qa["matched_xy"],
        unmatched_xy=qa["unmatched_xy"],
        gaia_xy=qa["gaia_xy"],
        show_gaia=bool(show_gaia),
        show_matches=bool(show_match),
        detail=(view == "Detail"),
    )
    st.image(img, caption=f"{qa['fits_name']}, negative stretch 1-99.6 %. {cap}", width="stretch")
