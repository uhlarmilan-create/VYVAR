"""Tiered Parameters dashboard, generated from the parameter registry (STEP 3).

Renders BASIC / ADVANCED / EXPERT tiers of ``widget=auto`` parameters straight from
``validation/params_registry.json`` + ``dataclasses.fields(AppConfig)`` introspection.
``widget=custom`` keys keep their existing hand-built widgets (not rendered here);
``widget=hidden`` keys never render. ``kind=resolved``/``derived`` keys render read-only
as "computed" (configured value plus, when a run provenance is available, the resolved
runtime value).

This is UI + metadata only; it touches no science path. The pure planning helpers
(:func:`plan_auto_widgets`, :func:`count_modified`) are Streamlit-free and unit-tested.
"""
from __future__ import annotations

import json
from typing import Any

import streamlit as st

import params_registry as pr

_KEY_PREFIX = "vyvar_pd_"

# Best-effort mapping resolved-config keys -> pipeline_meta.json runtime value lookups.
# Used only for the read-only "computed" display; missing entries fall back to a note.
_RESOLVED_META_KEYS: dict[str, tuple[str, ...]] = {
    "plate_scale_arcsec_per_px": ("plate_scale_arcsec_per_px", "scale_arcsec_per_px"),
    "phase01_plate_scale_arcsec_per_px": ("plate_scale_arcsec_per_px", "scale_arcsec_per_px"),
    "plate_solve_fov_deg": ("plate_solve_fov_deg", "fov_deg"),
    "gain": ("gain_e_per_adu", "gain"),
    "read_noise": ("read_noise_e", "read_noise"),
    "frame_width_px": ("frame_width_px", "naxis1"),
    "frame_height_px": ("frame_height_px", "naxis2"),
    "qc_preprocess_workers": ("qc_preprocess_workers", "parallel_workers"),
}


# --------------------------------------------------------------------------- #
# Pure, Streamlit-free helpers (unit tested)                                   #
# --------------------------------------------------------------------------- #
def resolve_auto_widget_display(
    cfg: Any,
    key: str,
    entry: dict[str, Any],
    types: dict[str, str],
    defaults: dict[str, Any],
) -> tuple[str, Any, bool]:
    """Resolve widget kind and display value for one ``widget=auto`` key (Streamlit-free).

    Returns ``(kind, display_value, none_fallback)`` where ``none_fallback`` is True when a
    scalar number/select would have coerced ``None`` and the renderer uses the text path instead.
    """
    kind = pr.infer_widget_kind(key, types.get(key, ""), entry)
    cur = getattr(cfg, key, None)
    if cur is None:
        cur = defaults.get(key)
    if entry.get("kind") in ("resolved", "derived"):
        return "resolved", cur, False
    if kind == "checkbox":
        return kind, bool(cur) if cur is not None else False, False
    if kind == "select":
        if cur is None:
            return "text", "", True
        options = list(pr.LITERAL_OPTIONS.get(key, ()))
        if str(cur) not in options:
            options = [str(cur), *options]
        return kind, str(cur), False
    if kind == "number":
        if isinstance(cur, (list, tuple, dict)):
            return "text", json.dumps(cur, sort_keys=True), False
        if cur is None:
            return "text", "", True
        return kind, cur, False
    if isinstance(cur, (dict, list)):
        return kind, json.dumps(cur, sort_keys=True), False
    return kind, "" if cur is None else str(cur), cur is None


def plan_auto_widgets(registry: dict[str, Any] | None = None) -> dict[str, str]:
    """Map every ``widget=auto`` registry key to a concrete widget kind.

    Pure function (no Streamlit). Returns ``{field_name: kind}`` where kind is one of
    ``checkbox`` | ``number`` | ``select`` | ``text``.
    """
    reg = registry if registry is not None else pr.load_registry()
    types = pr.appconfig_field_types()
    plan: dict[str, str] = {}
    for key, entry in reg.items():
        if entry.get("widget") != "auto":
            continue
        plan[key] = pr.infer_widget_kind(key, types.get(key, ""), entry)
    return plan


def count_modified(cfg: Any, owners: tuple[str, ...] | None = None) -> tuple[int, dict[str, Any]]:
    """Return (N modified, deviation report) for config vs dataclass defaults.

    ``owners`` restricts the count (and the returned ``modified`` list) to registry keys
    with one of the given owners; ``None`` counts every key. The dashboard counter passes
    ``("config_runtime",)`` so dead db_static fallbacks (e.g. observer_* on a fresh config)
    no longer inflate the number.
    """
    cfg_dict = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)
    dev = pr.compute_deviations(cfg_dict)
    if owners is not None:
        reg = pr.load_registry()
        want = set(owners)
        dev = {
            "modified": [m for m in dev["modified"] if reg.get(m["key"], {}).get("owner") in want],
            "unknown": dev["unknown"],
        }
    return len(dev["modified"]), dev


def group_keys_by_owner(registry: dict[str, Any] | None = None) -> dict[str, list[str]]:
    """Map each owner to its sorted list of registry keys (pure; unit tested)."""
    reg = registry if registry is not None else pr.load_registry()
    groups: dict[str, list[str]] = {o: [] for o in pr.OWNERS}
    for key, entry in reg.items():
        groups.setdefault(entry.get("owner", "config_runtime"), []).append(key)
    return {o: sorted(v) for o, v in groups.items()}


def editable_config_keys(registry: dict[str, Any] | None = None) -> list[str]:
    """Sorted widget=auto keys owned by config_runtime -- the only keys this dashboard saves."""
    reg = registry if registry is not None else pr.load_registry()
    return sorted(
        k for k, e in reg.items() if e.get("widget") == "auto" and e.get("owner") == "config_runtime"
    )


# --------------------------------------------------------------------------- #
# Streamlit rendering                                                          #
# --------------------------------------------------------------------------- #
def render_modified_counter(cfg: Any) -> int:
    """Render the 'N editable parameters modified' badge; returns the count.

    Counts owner=config_runtime keys ONLY: observatory facts (db_static) and FITS-resolved
    values (fits_dynamic) are not user-editable here, so counting their fallbacks was
    misleading.
    """
    n, _dev = count_modified(cfg, owners=("config_runtime",))
    if n == 0:
        st.caption(
            "**0 editable parameters modified** - config-owned parameters match code defaults."
        )
    else:
        st.caption(
            f"**{n} editable parameter(s) modified** vs code defaults "
            "(config-owned keys only; observatory facts and FITS-resolved values excluded)."
        )
    return n



def _coerce_for_save(key: str, kind: str, raw: Any, entry: dict[str, Any], cur: Any) -> Any:
    rng = entry.get("range")
    if kind == "checkbox":
        return bool(raw)
    if kind == "number":
        try:
            val = float(raw)
        except (TypeError, ValueError):
            return cur
        if rng is not None:
            val = max(float(rng[0]), min(float(rng[1]), val))
        # keep integers integral
        if isinstance(cur, int) and not isinstance(cur, bool):
            return int(round(val))
        return val
    if kind == "select":
        return str(raw)
    # text / dict / list (JSON)
    if isinstance(cur, (dict, list)):
        try:
            parsed = json.loads(raw) if isinstance(raw, str) and raw.strip() else ({} if isinstance(cur, dict) else [])
            if type(parsed) is type(cur):
                return parsed
            return cur
        except (json.JSONDecodeError, TypeError):
            return cur
    return str(raw)


def _render_auto_widget(
    cfg: Any,
    key: str,
    entry: dict[str, Any],
    types: dict[str, str],
    defaults: dict[str, Any],
    pipeline_meta: dict[str, Any] | None,
) -> None:
    kind, display, none_fallback = resolve_auto_widget_display(cfg, key, entry, types, defaults)
    cur = getattr(cfg, key, None)
    if cur is None:
        cur = defaults.get(key)
    default = defaults.get(key)
    is_resolved = entry.get("kind") in ("resolved", "derived")
    skey = f"{_KEY_PREFIX}{key}"

    modified = (key in defaults) and pr.values_differ(cur, default)
    marker = " *(modified)*" if modified else ""
    help_txt = f"{entry.get('help', '')} Default: {pr.default_repr(default)}."
    if none_fallback:
        help_txt += " Empty: set a value or use Reset to default."

    cols = st.columns([6, 1])
    with cols[0]:
        if is_resolved:
            runtime = _resolved_runtime_value(key, pipeline_meta)
            runtime_txt = f" | runtime: {runtime}" if runtime is not None else " | runtime: derived at run"
            st.markdown(
                f"**{entry['label']}** (`{key}`) - computed{marker}  \n"
                f"configured: `{pr.default_repr(cur)}`{runtime_txt}"
            )
            st.caption(entry.get("help", ""))
            return
        if kind == "checkbox":
            st.checkbox(entry["label"] + marker, value=bool(display), key=skey, help=help_txt)
        elif kind == "select":
            options = list(pr.LITERAL_OPTIONS.get(key, ()))
            if str(display) not in options:
                options = [str(display), *options]
            idx = options.index(str(display)) if str(display) in options else 0
            st.selectbox(entry["label"] + marker, options, index=idx, key=skey, help=help_txt)
        elif kind == "number":
            rng = entry.get("range")
            is_int = isinstance(display, int) and not isinstance(display, bool)
            kwargs: dict[str, Any] = {"key": skey, "help": help_txt}
            if rng is not None:
                kwargs["min_value"] = int(rng[0]) if is_int else float(rng[0])
                kwargs["max_value"] = int(rng[1]) if is_int else float(rng[1])
            value = int(display) if is_int else float(display)
            st.number_input(entry["label"] + marker, value=value, **kwargs)
        else:  # text / dict / list / None fallback
            st.text_input(entry["label"] + marker, value=str(display), key=skey, help=help_txt)
    with cols[1]:
        if not is_resolved and st.button("Reset", key=f"{skey}__reset", disabled=not modified):
            st.session_state[skey] = default
            st.rerun()


def _resolved_runtime_value(key: str, pipeline_meta: dict[str, Any] | None) -> Any:
    if not isinstance(pipeline_meta, dict):
        return None
    prov = pipeline_meta.get("provenance") if isinstance(pipeline_meta.get("provenance"), dict) else {}
    for mk in _RESOLVED_META_KEYS.get(key, ()):  # search meta top-level then provenance
        if mk in pipeline_meta:
            return pipeline_meta[mk]
        if isinstance(prov, dict) and mk in prov:
            return prov[mk]
    return None


def _apply_and_save(cfg: Any, auto_keys: list[str], entry_by_key: dict[str, Any], types: dict[str, str]) -> None:
    for key in auto_keys:
        entry = entry_by_key[key]
        if entry.get("kind") in ("resolved", "derived"):
            continue
        skey = f"{_KEY_PREFIX}{key}"
        if skey not in st.session_state:
            continue
        kind = pr.infer_widget_kind(key, types.get(key, ""), entry)
        cur = getattr(cfg, key, None)
        setattr(cfg, key, _coerce_for_save(key, kind, st.session_state[skey], entry, cur))
    from config import save_config_json, ui_config_persist

    with ui_config_persist():
        save_config_json(cfg.data_root, cfg.to_json())
    cfg.ensure_base_dirs()


def _render_db_static_card(cfg: Any, key: str, entry: dict[str, Any], defaults: dict[str, Any]) -> None:
    """Read-only observatory-fact card (owner=db_static). Never editable here."""
    cur = getattr(cfg, key, defaults.get(key))
    st.markdown(
        f"**{entry['label']}** (`{key}`) - observatory fact  \n"
        f"DB-resolved value: `{pr.default_repr(cur)}`"
    )
    st.caption(
        f"{entry.get('help', '')} Managed in **Settings -> Observatory / Database Explorer**, "
        "not on this dashboard."
    )


def _render_fits_dynamic_card(
    cfg: Any, key: str, entry: dict[str, Any], defaults: dict[str, Any], pipeline_meta: dict[str, Any] | None
) -> None:
    """Read-only runtime-resolved card (owner=fits_dynamic). Config value is a fallback."""
    cur = getattr(cfg, key, defaults.get(key))
    runtime = _resolved_runtime_value(key, pipeline_meta)
    if runtime is not None:
        val_txt = f"last-run value: `{runtime}` (resolved from FITS/WCS, from provenance)"
    else:
        val_txt = f"config fallback: `{pr.default_repr(cur)}` (fallback only - resolved from FITS/WCS at run time)"
    st.markdown(f"**{entry['label']}** (`{key}`) - resolved at runtime  \n{val_txt}")
    st.caption(entry.get("help", ""))


def render_params_dashboard(cfg: Any, *, pipeline_meta: dict[str, Any] | None = None) -> None:
    """Render the ownership-grouped parameter dashboard (mounted as the first Settings tab).

    Groups by the registry ``owner`` axis: config_runtime keys are editable (tiered exactly
    as before), db_static keys are read-only observatory facts, fits_dynamic keys are
    read-only runtime-resolved values, and internal keys are not rendered.
    """
    try:
        registry = pr.load_registry()
    except FileNotFoundError as exc:
        st.warning("Parameters dashboard is not available in this install (registry file missing).")
        st.caption(str(exc))
        return
    types = pr.appconfig_field_types()
    defaults = pr.appconfig_defaults()

    st.markdown("### Parameters (generated from registry)")
    render_modified_counter(cfg)
    st.caption(
        "Grouped by ownership. **Config** keys are editable here; **Observatory facts** "
        "(DB-owned) and **Resolved at runtime** (FITS/WCS) are read-only; internal plumbing "
        "keys are not shown. Hand-built composite editors (`widget=custom`) live in the other "
        "Settings tabs and remain authoritative."
    )

    def _cfg_auto(tier: str, phase: str | None = None) -> list[str]:
        return sorted(
            k for k, e in registry.items()
            if e.get("tier") == tier and e.get("widget") == "auto" and e.get("owner") == "config_runtime"
            and (phase is None or e.get("phase") == phase)
        )

    config_auto = editable_config_keys(registry)

    # ---- CONFIG (owner=config_runtime): editable, tiered ------------------------------------
    st.markdown("## Config (editable)")

    st.markdown("#### Basic")
    basic_keys = _cfg_auto("basic")
    if not basic_keys:
        st.caption("(none)")
    for key in basic_keys:
        _render_auto_widget(cfg, key, registry[key], types, defaults, pipeline_meta)

    st.markdown("#### Advanced")
    adv_phases = [p for p in pr.PHASES if _cfg_auto("advanced", p)]
    for phase in adv_phases:
        phase_keys = _cfg_auto("advanced", phase)
        with st.expander(f"{phase} ({len(phase_keys)})", expanded=False):
            _section_reset_button(defaults, phase_keys, f"adv_{phase}")
            for key in phase_keys:
                _render_auto_widget(cfg, key, registry[key], types, defaults, pipeline_meta)

    expert_keys = _cfg_auto("expert")
    with st.expander(f"Expert ({len(expert_keys)}) - changes affect science output", expanded=False):
        st.warning(
            "Expert parameters can change scientific results. Edit only if you understand "
            "the impact; every change is recorded in the run provenance snapshot."
        )
        exp_phases = [p for p in pr.PHASES if _cfg_auto("expert", p)]
        for phase in exp_phases:
            phase_keys = _cfg_auto("expert", phase)
            st.markdown(f"**{phase}**")
            _section_reset_button(defaults, phase_keys, f"exp_{phase}")
            for key in phase_keys:
                _render_auto_widget(cfg, key, registry[key], types, defaults, pipeline_meta)

    st.divider()
    if st.button("Save parameter dashboard to config.json", type="primary", key=f"{_KEY_PREFIX}save"):
        _apply_and_save(cfg, config_auto, registry, types)
        st.success("Saved to `config.json`. Refreshing UI...")
        st.rerun()

    # ---- OBSERVATORY FACTS (owner=db_static): read-only -------------------------------------
    groups = group_keys_by_owner(registry)
    st.markdown("## Observatory facts (DB-owned, read-only)")
    st.caption(
        "Site and identity facts resolved from the database reference tables. Edit them in the "
        "relevant Settings tab / Database Explorer; this dashboard never writes them."
    )
    db_keys = [k for k in groups.get("db_static", []) if registry[k].get("widget") != "hidden"]
    if not db_keys:
        st.caption("(none)")
    for key in db_keys:
        _render_db_static_card(cfg, key, registry[key], defaults)

    # ---- RESOLVED AT RUNTIME (owner=fits_dynamic): read-only --------------------------------
    st.markdown("## Resolved at runtime (FITS/WCS, read-only)")
    st.caption(
        "Values the pipeline resolves from FITS headers / WCS per run; the config value is only "
        "a fallback. Where a run provenance is available, its resolved value is shown."
    )
    fits_keys = [k for k in groups.get("fits_dynamic", []) if registry[k].get("widget") != "hidden"]
    if not fits_keys:
        st.caption("(none)")
    for key in fits_keys:
        _render_fits_dynamic_card(cfg, key, registry[key], defaults, pipeline_meta)


def _section_reset_button(defaults: dict[str, Any], keys: list[str], tag: str) -> None:
    if st.button("Reset section to defaults", key=f"{_KEY_PREFIX}sec_reset_{tag}"):
        for key in keys:
            if key in defaults:
                st.session_state[f"{_KEY_PREFIX}{key}"] = defaults[key]
        st.rerun()
