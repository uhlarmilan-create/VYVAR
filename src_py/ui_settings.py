"""Settings page (SETTINGS-2026-10): every config.json key once, in the config.json groups, one Save.

Model, parsing and validation: ``settings_model.py`` (Streamlit-free). Labels, help, units, ranges
and tiers: ``dev/validation/params_registry.json`` (the help line is also the comment written above
the key in config.json). Values VYVAR measures per run (FITS headers, MASTERSTAR) and the
observatory tables (telescopes, cameras, sites: Database Explorer) are not settings.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

import params_registry as pr
import settings_model as sm
from config import AppConfig, save_config_json, ui_config_persist

_P = "vyvar_set_"  # session-state prefix
_STATE_COLOURS = {"ready": ("#1E6B3A", "#E6F4EA"), "old": ("#7A4700", "#FFF4E0"), "unfinished": ("#7A4700", "#FFF4E0"),
                  "missing": ("#8A1C1C", "#FDECEC"), "invalid": ("#8A1C1C", "#FDECEC")}

_CSS = """
<style>
.vy-row {border-top: 1px solid rgba(128,128,128,0.25); margin: 4px 0 2px 0;}
.vy-label {font-weight: 600; font-size: 0.95rem;}
.vy-key {font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.76rem; opacity: 0.65;}
.vy-help {font-size: 0.85rem; opacity: 0.85; margin-top: 2px;}
.vy-meta {font-size: 0.78rem; opacity: 0.7; padding-top: 6px;}
.vy-badge {font-size: 0.72rem; border-radius: 999px; padding: 1px 8px; margin-left: 6px; white-space: nowrap;}
.vy-changed {background: #FFF4E0; color: #7A4700;}
.vy-edited {background: #E8F0FB; color: #174A94;}
.vy-card {border: 1px solid rgba(128,128,128,0.35); border-radius: 8px; padding: 12px 14px; height: 100%;}
.vy-card h4 {margin: 0 0 4px 0; font-size: 1.0rem;}
.vy-card dl {display: grid; grid-template-columns: auto 1fr; gap: 2px 12px; margin: 6px 0; font-size: 0.85rem;}
.vy-card dt {color: #5A6370;} .vy-card dd {margin: 0;}
.vy-path {font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 0.75rem; word-break: break-all; color: #5A6370;}
</style>
"""


# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------
def _wkey(key: str) -> str:
    return f"{_P}w_{key}"


def _init_state(current: dict[str, Any]) -> dict[str, Any]:
    """Edited values (draft) of this session; reset when config.json changed underneath."""
    base_sig = json.dumps(current, sort_keys=True, default=str)
    if st.session_state.get(f"{_P}base_sig") != base_sig:
        st.session_state[f"{_P}base_sig"] = base_sig
        st.session_state[f"{_P}draft"] = json.loads(json.dumps(current, default=str))
        st.session_state[f"{_P}table_gen"] = st.session_state.get(f"{_P}table_gen", 0) + 1
        for k in [k for k in st.session_state if str(k).startswith(f"{_P}w_")]:
            del st.session_state[k]
    return st.session_state[f"{_P}draft"]


def _revert(current: dict[str, Any]) -> None:
    st.session_state[f"{_P}draft"] = json.loads(json.dumps(current, default=str))
    st.session_state[f"{_P}table_gen"] = st.session_state.get(f"{_P}table_gen", 0) + 1
    st.session_state[f"{_P}errors"] = {}
    for k in [k for k in st.session_state if str(k).startswith(f"{_P}w_")]:
        del st.session_state[k]


def _reset_key(key: str, default: Any) -> None:
    st.session_state[f"{_P}draft"][key] = json.loads(json.dumps(default, default=str))
    st.session_state.pop(_wkey(key), None)
    st.session_state[f"{_P}table_gen"] = st.session_state.get(f"{_P}table_gen", 0) + 1


# ---------------------------------------------------------------------------
# one setting
# ---------------------------------------------------------------------------
def _float_step(value: float, rng: list[float] | None) -> float:
    span = abs(float(rng[1]) - float(rng[0])) if rng else abs(value) or 1.0
    if span <= 0.02:
        return 1e-5
    if span <= 1.5:
        return 0.01
    if span <= 30:
        return 0.1
    return 1.0


def _render_widget(key: str, entry: dict[str, Any], draft: dict[str, Any], current: dict[str, Any]) -> str | None:
    """Draw the widget of ``key``; store the parsed value in ``draft``; return an error or None."""
    value = draft[key]
    kind = sm.widget_kind(key, current.get(key, value), entry)
    wk = _wkey(key)
    rng = entry.get("range")
    label = entry.get("label") or key
    if kind == "checkbox":
        if wk not in st.session_state:
            st.session_state[wk] = bool(value)
        draft[key] = bool(st.toggle(label, key=wk, label_visibility="collapsed"))
        return None
    if kind == "select":
        opts = sm.select_options(key, value)
        if wk not in st.session_state:
            st.session_state[wk] = str(value)
        draft[key] = st.selectbox(label, opts, key=wk, label_visibility="collapsed")
        return None
    if kind == "int":
        if wk not in st.session_state:
            st.session_state[wk] = int(value)
        kw: dict[str, Any] = {"step": 1}
        if rng:
            kw.update(min_value=int(rng[0]), max_value=int(rng[1]))
        draft[key] = int(st.number_input(label, key=wk, label_visibility="collapsed", **kw))
        return sm.check_value(key, draft[key], entry)
    if kind == "float":
        if wk not in st.session_state:
            st.session_state[wk] = float(value)
        step = _float_step(float(value), rng)
        kw = {"step": step, "format": "%.6g"}
        if rng:
            kw.update(min_value=float(rng[0]), max_value=float(rng[1]))
        v = float(st.number_input(label, key=wk, label_visibility="collapsed", **kw))
        draft[key] = v if math.isfinite(v) else draft[key]
        return sm.check_value(key, draft[key], entry)
    if kind == "tags":
        if wk not in st.session_state:
            st.session_state[wk] = sm.tags_text(value)
        draft[key] = sm.tags_value(st.text_input(label, key=wk, label_visibility="collapsed",
                                                 placeholder="comma-separated, e.g. ROT, ELL"))
        return None
    if kind == "table":
        gen = st.session_state.get(f"{_P}table_gen", 0)
        rows = sm.table_rows(key, value)
        cols = list(sm.table_rows(key, current.get(key) or ([{"bprp": 0.1}] if key == "comp_color_tiers" else {"X": 0}))[0].keys()) \
            if not rows else list(rows[0].keys())
        df = pd.DataFrame(rows, columns=cols)
        edited = st.data_editor(
            df, key=f"{_P}t_{key}_{gen}", hide_index=True, width="stretch",
            num_rows="fixed" if key == "comp_color_tiers" else "dynamic",
            disabled=["class"] if key == "comp_color_tiers" else [],
        )
        val, err = sm.table_value(key, edited.to_dict("records"))
        if err is None:
            draft[key] = val
        return err
    # text
    if wk not in st.session_state:
        st.session_state[wk] = "" if value is None else str(value)
    draft[key] = st.text_input(label, key=wk, label_visibility="collapsed").strip()
    return sm.check_value(key, draft[key], entry)


def _render_setting(key: str, entry: dict[str, Any], draft: dict[str, Any], current: dict[str, Any],
                    defaults: dict[str, Any], errors: dict[str, str]) -> None:
    """One row: label, key and help on the left; widget, unit / range / default and Reset on the right."""
    unit = entry.get("unit")
    st.markdown('<div class="vy-row"></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([5, 3, 2], vertical_alignment="top")
    with c2:
        err = _render_widget(key, entry, draft, current)
    changed = sm.differs_from_default(key, draft[key], defaults)
    edited = key in current and pr.values_differ(draft[key], current[key])
    with c1:
        badges = ""
        if edited:
            badges += '<span class="vy-badge vy-edited">not saved</span>'
        if changed:
            badges += '<span class="vy-badge vy-changed">changed</span>'
        st.markdown(
            f'<div class="vy-label">{entry.get("label") or key}{badges}</div>'
            f'<div class="vy-key">{key}</div><div class="vy-help">{entry.get("help", "")}</div>',
            unsafe_allow_html=True,
        )
    with c3:
        meta = []
        if unit:
            meta.append(unit)
        if entry.get("range"):
            lo, hi = entry["range"]
            meta.append(f"{lo:g} .. {hi:g}")
        if key in defaults:
            meta.append(f"default {sm.format_default(key, defaults[key], unit)}")
        st.markdown(f'<div class="vy-meta">{"<br>".join(meta)}</div>', unsafe_allow_html=True)
        if changed:
            st.button("Reset", key=f"{_P}reset_{key}", on_click=_reset_key, args=(key, defaults[key]),
                      help="Back to the default value (saved with Save)")
    if err:
        errors[key] = err
        c2.error(err)
    else:
        errors.pop(key, None)


# ---------------------------------------------------------------------------
# groups
# ---------------------------------------------------------------------------
def _render_group(group: str, keys: list[str], registry: dict[str, Any], draft: dict[str, Any],
                  current: dict[str, Any], defaults: dict[str, Any], errors: dict[str, str]) -> None:
    tiers: dict[str, list[str]] = {"basic": [], "advanced": [], "expert": []}
    for k in keys:
        tiers.setdefault(registry[k].get("tier", "advanced"), []).append(k)
    for tier, title in (("basic", "Basic"), ("advanced", "Advanced")):
        if tiers[tier]:
            st.markdown(f"##### {title}")
            for k in tiers[tier]:
                _render_setting(k, registry[k], draft, current, defaults, errors)
    if tiers["expert"]:
        n_ch = sum(sm.differs_from_default(k, draft[k], defaults) for k in tiers["expert"])
        with st.expander(f"Expert settings ({len(tiers['expert'])}"
                         + (f", {n_ch} changed" if n_ch else "") + ")", expanded=bool(n_ch)):
            st.warning("These settings change science results. Read the configuration guide before changing them.")
            for k in tiers["expert"]:
                _render_setting(k, registry[k], draft, current, defaults, errors)


def _card(status: sm.CatalogStatus, *, title: str | None = None, desc: str = "") -> None:
    fg, bg = _STATE_COLOURS.get(status.state, ("#3B434D", "#EEF0F3"))
    facts = "".join(f"<dt>{a}</dt><dd>{b}</dd>" for a, b in status.facts)
    st.markdown(
        f'<div class="vy-card"><h4>{title or status.name}'
        f'<span class="vy-badge" style="background:{bg};color:{fg}">{status.state}</span></h4>'
        f'<div class="vy-help">{desc}</div>'
        + (f"<dl>{facts}</dl>" if facts else "")
        + (f'<div class="vy-meta">{status.note}</div>' if status.note else "")
        + f'<div class="vy-path">{status.path}</div></div>',
        unsafe_allow_html=True,
    )


def _render_paths(cfg: AppConfig, keys: list[str], registry: dict[str, Any], draft: dict[str, Any],
                  current: dict[str, Any], defaults: dict[str, Any], errors: dict[str, str]) -> None:
    st.markdown("##### Folders")
    st.text_input("Data directory (from the install, read-only)", value=str(cfg.data_root), disabled=True)
    for k, is_file in (("archive_root", False), ("calibration_library_root", False), ("database_path", True)):
        if k in keys:
            _render_setting(k, registry[k], draft, current, defaults, errors)
            st.caption(f"Status: {sm.folder_status(draft[k], is_file=is_file)}")

    st.markdown("##### Star catalogs")
    gaia = sm.gaia_status(draft.get("gaia_db_path", ""))
    vsx, exo = sm.objects_status(draft.get("objects_db_path", ""))
    c1, c2, c3 = st.columns(3)
    with c1:
        _card(gaia, desc="Positions, magnitudes and colours: star identification, comparison stars, plate solving.")
    with c2:
        _card(vsx, desc="AAVSO VSX: known variables (targets, never comparison stars).")
    with c3:
        _card(exo, desc="NASA Exoplanet Archive: confirmed planets and TESS TOI (label only).")
    for k in ("gaia_db_path", "objects_db_path"):
        if k in keys:
            _render_setting(k, registry[k], draft, current, defaults, errors)

    st.markdown("##### Plate solving")
    p1, p2 = st.columns(2)
    with p1:
        _card(sm.CatalogStatus("Solving with a position", gaia.path, "ready" if gaia.state == "ready" else gaia.state,
                               [], "Uses the Gaia DR3 catalog; no extra files."),
              desc="FITS header has RA/Dec (or a solved sibling filter gives the hint).")
    with p2:
        _card(sm.blind_index_status(draft.get("astrometry_index_dir"), draft.get("gaia_db_path")),
              desc="No position in the header: 4-star pattern index from Gaia (all 9 scales cover every telescope).")
    if "astrometry_index_dir" in keys:
        _render_setting("astrometry_index_dir", registry["astrometry_index_dir"], draft, current, defaults, errors)

    from ui_catalog_builder import render_catalog_builder  # noqa: PLC0415

    render_catalog_builder(cfg, current, draft)


def _render_observer_site(cfg: AppConfig) -> None:
    from database import get_observer_location_by_id  # noqa: PLC0415

    lid = int(getattr(cfg, "observer_location_id", 0) or 0)
    row = get_observer_location_by_id(str(cfg.database_path), lid) if lid > 0 else None
    with st.container(border=True):
        st.markdown("**Default observing site**")
        if row:
            st.markdown(f"{row['name']} (LOCATION.ID {row['id']}): lat {row['lat']:.5f}, lon {row['lon']:.5f}, "
                        f"alt {row.get('alt_m', 0):.0f} m")
        else:
            st.markdown("No default site.")
        st.caption("The default site is the one marked default in Database Explorer > Sites; "
                   "a run may choose another site on the import page. Telescopes and cameras are in Database Explorer too.")


def _render_dao_helper(cfg: AppConfig, pipeline: Any, draft_dir_override: Path | None, draft: dict[str, Any]) -> None:
    """Suggestion from the active MASTERSTAR for the detection / aperture values (fills the draft;
    Save writes it). Computing these per draft automatically is a later, separate step."""
    from masterstar_context import load_masterstar_context, resolve_masterstar_fits_path  # noqa: PLC0415

    with st.expander("Suggest detection values from the active MASTERSTAR"):
        ms = resolve_masterstar_fits_path(cfg=cfg, db=getattr(pipeline, "db", None),
                                          draft_id=st.session_state.get("vyvar_last_draft_id"),
                                          draft_dir_override=draft_dir_override)
        if ms is None or not Path(ms).is_file():
            st.caption("No MASTERSTAR for the loaded processed field (Pipeline -> Load processed field).")
            return
        ctx = load_masterstar_context(ms)
        fwhm = ctx.vy_fwhm_px
        n = sm.masterstar_detection_count(Path(ms))
        if fwhm is None or not math.isfinite(float(fwhm)):
            st.caption("The MASTERSTAR header has no FWHM.")
            return
        sug = sm.masterstar_suggestions(float(fwhm), n)
        st.markdown(f"MASTERSTAR `{Path(ms).parent.name}`: FWHM {float(fwhm):.2f} px, {n or '?'} stars.")
        st.dataframe(pd.DataFrame([{"setting": k, "now": draft.get(k), "suggested": v} for k, v in sug.items()]),
                     hide_index=True, width="stretch")
        if st.button("Put suggestions into the form", key=f"{_P}dao_apply"):
            for k, v in sug.items():
                draft[k] = v
                st.session_state.pop(_wkey(k), None)
            st.rerun()


# ---------------------------------------------------------------------------
# page
# ---------------------------------------------------------------------------
def _save(cfg: AppConfig, draft: dict[str, Any], current: dict[str, Any]) -> None:
    data = dict(current)
    for k in sm.changed_keys(current, draft):
        data[k] = draft[k]
    with ui_config_persist():
        save_config_json(cfg.data_root, data)
    st.session_state[f"{_P}saved_msg"] = f"Saved {len(sm.changed_keys(current, draft))} change(s) to config.json."
    for k in [k for k in st.session_state if str(k).startswith(f"{_P}w_")]:
        del st.session_state[k]


def _sync_from_widgets(draft: dict[str, Any], registry: dict[str, Any], current: dict[str, Any]) -> None:
    """Scalar widget values of the last interaction into the draft before anything is drawn
    (the navigation counts and the header then match the form)."""
    for key in list(draft):
        wk = _wkey(key)
        if wk not in st.session_state or key not in registry:
            continue
        raw = st.session_state[wk]
        kind = sm.widget_kind(key, current.get(key, draft[key]), registry[key])
        if kind == "checkbox":
            draft[key] = bool(raw)
        elif kind == "int":
            draft[key] = int(raw)
        elif kind == "float":
            v = float(raw)
            if math.isfinite(v):
                draft[key] = v
        elif kind == "tags":
            draft[key] = sm.tags_value(raw)
        elif kind in ("select", "text"):
            draft[key] = str(raw).strip() if kind == "text" else raw


def _render_header(cfg: AppConfig, groups: dict[str, list[str]], registry: dict[str, Any], draft: dict[str, Any],
                   current: dict[str, Any], defaults: dict[str, Any], errors: dict[str, str], n_keys: int) -> None:
    edited = sm.changed_keys(current, draft)
    n_default = sum(sm.differs_from_default(k, current[k], defaults) for ks in groups.values() for k in ks)
    h1, h2, h3 = st.columns([5, 1.2, 1.6], vertical_alignment="center")
    with h1:
        st.subheader("Settings")
        st.caption(f"config.json | {n_keys} settings in {len([g for g in groups if groups[g]])} groups | "
                   f"{n_default} differ from the default" + (f" | **{len(edited)} not saved**" if edited else ""))
    cross = sm.cross_check(draft)
    blocking = bool(cross) or bool(errors)
    with h2:
        st.button("Revert", key=f"{_P}revert", disabled=not edited, on_click=_revert, args=(current,),
                  help="Discard the changes not saved yet")
    with h3:
        if st.button("Save to config.json", key=f"{_P}save", type="primary", disabled=not edited or blocking):
            _save(cfg, draft, current)
            st.rerun()
    msg = st.session_state.pop(f"{_P}saved_msg", None)
    if msg:
        st.success(msg)
    for e in cross:
        st.error(e)
    if errors:
        st.error("Fix: " + ", ".join(f"{registry[k].get('label', k)} ({v})" for k, v in errors.items() if k in registry))


def render_settings_dashboard(
    cfg: AppConfig,
    pipeline: Any,
    *,
    draft_dir_override: Path | None = None,
) -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
    try:
        registry = pr.load_registry()
    except FileNotFoundError as exc:
        st.error(f"Settings need the parameter registry: {exc}")
        return
    phase_help = pr.load_phase_help()
    defaults = pr.appconfig_defaults()
    current = cfg.to_json()
    draft = _init_state(current)
    errors: dict[str, str] = st.session_state.setdefault(f"{_P}errors", {})
    groups = sm.settings_keys(registry, list(current.keys()))
    n_keys = sum(len(v) for v in groups.values())

    _sync_from_widgets(draft, registry, current)
    head = st.container()  # filled at the end, after every widget has updated the draft
    edited = sm.changed_keys(current, draft)

    order = [g for g in sm.group_order() if groups.get(g)]
    def _nav_label(g: str) -> str:
        n_ed = sum(1 for k in groups[g] if k in edited)
        return f"{sm.GROUP_TITLES.get(g, g)}  ({len(groups[g])})" + (f"  | {n_ed} not saved" if n_ed else "")

    nav, main = st.columns([1.15, 4], gap="large")
    with nav:
        group = st.radio("Groups (as in config.json)", order, key=f"{_P}group", format_func=_nav_label)
    with main:
        st.markdown(f"#### {sm.GROUP_TITLES.get(group, group)}")
        if phase_help.get(group):
            st.caption(phase_help[group])
        if group == "observer":
            _render_observer_site(cfg)
        if group == "paths":
            _render_paths(cfg, groups[group], registry, draft, current, defaults, errors)
        else:
            if group == "detection":
                _render_dao_helper(cfg, pipeline, draft_dir_override, draft)
            _render_group(group, groups[group], registry, draft, current, defaults, errors)

    with head:
        _render_header(cfg, groups, registry, draft, current, defaults, errors, n_keys)
