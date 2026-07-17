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


def count_modified(cfg: Any) -> tuple[int, dict[str, Any]]:
    """Return (N modified, deviation report) for the FULL config vs dataclass defaults."""
    cfg_dict = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)
    dev = pr.compute_deviations(cfg_dict)
    return len(dev["modified"]), dev


# --------------------------------------------------------------------------- #
# Streamlit rendering                                                          #
# --------------------------------------------------------------------------- #
def render_modified_counter(cfg: Any) -> int:
    """Render the global 'N parameters modified' badge; returns the count."""
    n, dev = count_modified(cfg)
    if n == 0:
        st.caption("**0 parameters modified** - configuration matches code defaults.")
    else:
        st.caption(
            f"**{n} parameter(s) modified** vs code defaults "
            "(counts all keys, including custom/hidden ones)."
        )
    return n


def _default_of(defaults: dict[str, Any], key: str) -> Any:
    return defaults.get(key)


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
    # text / dict
    if isinstance(cur, dict):
        try:
            parsed = json.loads(raw) if isinstance(raw, str) and raw.strip() else {}
            return parsed if isinstance(parsed, dict) else cur
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
    kind = pr.infer_widget_kind(key, types.get(key, ""), entry)
    cur = getattr(cfg, key, defaults.get(key))
    default = defaults.get(key)
    is_resolved = entry.get("kind") in ("resolved", "derived")
    skey = f"{_KEY_PREFIX}{key}"

    modified = (key in defaults) and pr.values_differ(cur, default)
    marker = " *(modified)*" if modified else ""
    help_txt = f"{entry.get('help', '')} Default: {pr.default_repr(default)}."

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
            st.checkbox(entry["label"] + marker, value=bool(cur), key=skey, help=help_txt)
        elif kind == "select":
            options = list(pr.LITERAL_OPTIONS.get(key, ()))
            if str(cur) not in options:
                options = [str(cur), *options]
            idx = options.index(str(cur)) if str(cur) in options else 0
            st.selectbox(entry["label"] + marker, options, index=idx, key=skey, help=help_txt)
        elif kind == "number":
            rng = entry.get("range")
            is_int = isinstance(cur, int) and not isinstance(cur, bool)
            kwargs: dict[str, Any] = {"key": skey, "help": help_txt}
            if rng is not None:
                kwargs["min_value"] = int(rng[0]) if is_int else float(rng[0])
                kwargs["max_value"] = int(rng[1]) if is_int else float(rng[1])
            value = int(cur) if is_int else float(cur)
            st.number_input(entry["label"] + marker, value=value, **kwargs)
        else:  # text / dict
            if isinstance(cur, dict):
                st.text_input(entry["label"] + marker, value=json.dumps(cur, sort_keys=True), key=skey, help=help_txt)
            else:
                st.text_input(entry["label"] + marker, value="" if cur is None else str(cur), key=skey, help=help_txt)
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
    from config import save_config_json

    save_config_json(cfg.project_root, cfg.to_json())
    cfg.ensure_base_dirs()


def render_params_dashboard(cfg: Any, *, pipeline_meta: dict[str, Any] | None = None) -> None:
    """Render the tiered parameter dashboard (mounted as the first Settings tab)."""
    registry = pr.load_registry()
    types = pr.appconfig_field_types()
    defaults = pr.appconfig_defaults()

    st.markdown("### Parameters (generated from registry)")
    render_modified_counter(cfg)
    st.caption(
        "Generated from `validation/params_registry.json`. Hand-built composite editors "
        "(`widget=custom`) live in the other Settings tabs and remain authoritative; "
        "`kind=resolved` parameters are shown read-only (runtime value auto-derived)."
    )

    def _auto(tier: str, phase: str | None = None) -> list[str]:
        return sorted(
            k for k, e in registry.items()
            if e.get("tier") == tier and e.get("widget") == "auto"
            and (phase is None or e.get("phase") == phase)
        )

    all_auto = sorted(k for k, e in registry.items() if e.get("widget") == "auto")

    # BASIC -- always visible, flat
    st.markdown("#### Basic")
    basic_keys = _auto("basic")
    if not basic_keys:
        st.caption("(none)")
    for key in basic_keys:
        _render_auto_widget(cfg, key, registry[key], types, defaults, pipeline_meta)

    # ADVANCED -- one expander per phase, collapsed
    st.markdown("#### Advanced")
    adv_phases = [p for p in pr.PHASES if _auto("advanced", p)]
    for phase in adv_phases:
        phase_keys = _auto("advanced", phase)
        with st.expander(f"{phase} ({len(phase_keys)})", expanded=False):
            _section_reset_button(defaults, phase_keys, f"adv_{phase}")
            for key in phase_keys:
                _render_auto_widget(cfg, key, registry[key], types, defaults, pipeline_meta)

    # EXPERT -- single collapsed section, grouped by phase, with a warning
    expert_keys = _auto("expert")
    with st.expander(f"Expert ({len(expert_keys)}) - changes affect science output", expanded=False):
        st.warning(
            "Expert parameters can change scientific results. Edit only if you understand "
            "the impact; every change is recorded in the run provenance snapshot."
        )
        exp_phases = [p for p in pr.PHASES if _auto("expert", p)]
        for phase in exp_phases:
            phase_keys = _auto("expert", phase)
            st.markdown(f"**{phase}**")
            _section_reset_button(defaults, phase_keys, f"exp_{phase}")
            for key in phase_keys:
                _render_auto_widget(cfg, key, registry[key], types, defaults, pipeline_meta)

    st.divider()
    if st.button("Save parameter dashboard to config.json", type="primary", key=f"{_KEY_PREFIX}save"):
        _apply_and_save(cfg, all_auto, registry, types)
        st.success("Saved to `config.json`. Refreshing UI...")
        st.rerun()


def _section_reset_button(defaults: dict[str, Any], keys: list[str], tag: str) -> None:
    if st.button("Reset section to defaults", key=f"{_KEY_PREFIX}sec_reset_{tag}"):
        for key in keys:
            if key in defaults:
                st.session_state[f"{_KEY_PREFIX}{key}"] = defaults[key]
        st.rerun()
