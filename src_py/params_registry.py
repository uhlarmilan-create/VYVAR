"""Machine-readable AppConfig parameter registry (shared core).

The registry (``validation/params_registry.json``) carries only *editorial* metadata
for each public ``AppConfig`` field: tier, phase, label, help, unit, kind, range and
widget hint. Defaults and types are NEVER stored here -- they come from
``dataclasses.fields(AppConfig)`` introspection so the registry cannot drift from code.

Consumers:
  * ``tests/test_params_registry.py`` -- parity + freshness guard.
  * ``tools/gen_params_md.py``        -- regenerates ``docs/VYVAR_PARAMS.md``.
  * ``ui_params_dashboard.py``        -- tiered Settings dashboard.
  * ``photometry_report.py``          -- PDF Configuration page (deviation table).

This is metadata/plumbing only; it touches no science path.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import config

REGISTRY_PATH = Path(__file__).resolve().parent.parent / "dev" / "validation" / "params_registry.json"

TIERS = ("basic", "advanced", "expert")
PHASES = (
    "observer", "paths", "calibration", "qc", "alignment", "detection",
    "photometry", "comp_selection", "trust", "extinction", "reports",
    "export", "system",
)
KINDS = ("static", "derived", "resolved")
WIDGETS = ("auto", "custom", "hidden")
# Storage-and-ownership axis (PARAM-OWNERSHIP-WAVE-A): who owns the value at run time.
#   db_static     -- observatory/site/detector facts owned by DB reference tables (read-only here).
#   config_runtime-- algorithm-behavior knobs the user tunes in config.json / the dashboard.
#   fits_dynamic  -- resolved from FITS/WCS at run time (config value is a fallback only).
#   internal      -- plumbing (paths, worker counts, project_root); not user-facing.
OWNERS = ("db_static", "config_runtime", "fits_dynamic", "internal")
# Physical-scope axis (PARAM-SCOPE-AUDIT): does the correct value depend on equipment/site/session?
#   universal -- one correct value for every setup (algorithm constants, FWHM multiples, statistics).
#   rig       -- depends on telescope + camera + binning (pixels, ADU, FOV, detector geometry).
#   site      -- depends on observing location / atmosphere (extinction, observer coords).
#   session   -- resolved per run from FITS/WCS/plan; config is fallback only.
# Consistent with src_py/param_resolver.py categories (equipment-intrinsic / observation-specific / site).
SCOPES = ("universal", "rig", "site", "session")
SCOPE_CONFIDENCES = ("high", "low")
# Resolution key: what identity the correct value must be looked up by (PARAM-SCOPE-REMEDIATION C').
#   none         -- scope=universal; no equipment/site/frame identity needed.
#   rig          -- (ID_EQUIPMENTS, ID_TELESCOPE) draft-level rig identity.
#   rig_band     -- rig + band token (filter, or TR/TG/TB for OSC Bayer planes).
#   rig_sampling -- rig + effective linear sampling (mono: binning; OSC: 2 x osc_channel_binning).
#   site         -- ID_LOCATION observing site.
#   frame        -- FITS / solved WCS per frame (scope=session).
# OSC note: effective linear sampling is 2 (Bayer superpixel) x osc_channel_binning (default 2 -> 4x).
SCOPE_KEYS = ("none", "rig", "rig_band", "rig_sampling", "site", "frame")
# Triage group for scope=rig only: a=genuine per-rig physics, b=unit artefact (normalise to arcsec/FWHM),
# c=operational tuning; n/a for non-rig entries.
SCOPE_GROUPS = ("n/a", "a", "b", "c")

# Parameters that define effective linear sampling (rig_sampling). They must use scope_key=rig,
# never rig_sampling -- resolving them by a key they define is circular. Add new sampling
# definers here when introduced.
SAMPLING_DEFINING_KEYS = ("osc_channel_binning", "calibration_library_native_binning")

ENTRY_KEYS = (
    "tier", "phase", "label", "help", "unit", "kind", "range", "widget", "owner",
    "scope", "scope_key", "scope_group", "scope_confidence",
)

# str-typed fields whose values are drawn from a fixed vocabulary (selectbox).
LITERAL_OPTIONS: dict[str, tuple[str, ...]] = {
    "photometry_mode": ("aperture", "epsf", "both"),
    "apply_color_term": ("off", "auto", "on"),
    "k2_mode": ("off", "literature", "fit_else_literature"),
    "hrd_color_highlight_mode": ("soft", "scale"),
    "hrd_color_white_point": ("field_median", "d65"),
    "blind_index_select_mode": ("auto", "series_all", "single"),
    "blind_img_select_mode": ("per_cell", "central"),
    "phase01_flux_col": ("dao_flux", "psf_flux"),
}


def load_registry(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Load the registry JSON, keyed by AppConfig field name.

    Reserved ``__``-prefixed top-level keys (e.g. ``__meta__``) carry editorial metadata
    that is NOT a parameter and are stripped here, so parity with AppConfig fields holds.
    """
    p = path or REGISTRY_PATH
    if not p.is_file():
        raise FileNotFoundError(
            f"parameter registry missing: {p} (release bundle must ship dev/validation/params_registry.json)"
        )
    raw = json.loads(p.read_text(encoding="utf-8"))
    return {k: v for k, v in raw.items() if not k.startswith("__")}


def load_registry_meta(path: Path | None = None) -> dict[str, Any]:
    """Return the reserved ``__meta__`` block of the registry (editorial metadata)."""
    p = path or REGISTRY_PATH
    if not p.is_file():
        raise FileNotFoundError(
            f"parameter registry missing: {p} (release bundle must ship dev/validation/params_registry.json)"
        )
    raw = json.loads(p.read_text(encoding="utf-8"))
    meta = raw.get("__meta__")
    return meta if isinstance(meta, dict) else {}


def load_phase_help(path: Path | None = None) -> dict[str, str]:
    """One-line group descriptions per pipeline phase (config.json section comments)."""
    ph = load_registry_meta(path).get("phase_help")
    return ph if isinstance(ph, dict) else {}


def appconfig_fields() -> list[dataclasses.Field]:
    return list(dataclasses.fields(config.AppConfig))


_INTERNAL_APP_CONFIG_FIELDS = frozenset({"data_root"})


def appconfig_field_names() -> set[str]:
    """Public (non underscore-prefixed) AppConfig field names."""
    return {
        f.name
        for f in appconfig_fields()
        if not f.name.startswith("_") and f.name not in _INTERNAL_APP_CONFIG_FIELDS
    }


def appconfig_field_types() -> dict[str, str]:
    return {f.name: _type_str(f.type) for f in appconfig_fields()}


def appconfig_defaults() -> dict[str, Any]:
    """Resolvable dataclass defaults (skips init=False fields with no default)."""
    out: dict[str, Any] = {}
    for f in appconfig_fields():
        if f.default is not dataclasses.MISSING:
            out[f.name] = f.default
        elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            try:
                out[f.name] = f.default_factory()  # type: ignore[misc]
            except Exception:  # noqa: BLE001
                pass
    return out


def _type_str(t: Any) -> str:
    return t if isinstance(t, str) else getattr(t, "__name__", str(t))


def default_repr(value: Any) -> str:
    """Compact, stable string for a default value (doc/table rendering)."""
    from pathlib import Path as _P

    if isinstance(value, _P):
        # Name-independent: checkout folder basename must not appear in docs
        # (SEL-GHOST-01 S7; worktree --clean). Git toplevel -> stable token.
        try:
            import subprocess

            cwd = value if value.is_dir() else value.parent
            top = subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=str(cwd),
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            if top and _P(top).resolve() == value.resolve():
                return "(git toplevel)"
        except Exception:  # noqa: BLE001
            pass
        return "(path)"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, dict):
        return "{}" if not value else json.dumps(value, sort_keys=True)
    if value is None:
        return "None"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _numeric(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def values_differ(run_value: Any, default_value: Any) -> bool:
    """True when a run/config value deviates from the dataclass default."""
    if _numeric(run_value) and _numeric(default_value):
        try:
            return float(run_value) != float(default_value)
        except (TypeError, ValueError, OverflowError):
            return run_value != default_value
    return run_value != default_value


def compute_deviations(
    config_dict: dict[str, Any],
    defaults: dict[str, Any] | None = None,
    field_names: set[str] | None = None,
) -> dict[str, Any]:
    """Deviation report of a config snapshot vs dataclass defaults.

    Returns ``{"modified": [...], "unknown": [...]}`` where ``modified`` is a list of
    ``{"key", "value", "default"}`` (value != default), and ``unknown`` lists snapshot
    keys that are not AppConfig fields (legacy / renamed keys) -- never raises on them.
    """
    defaults = appconfig_defaults() if defaults is None else defaults
    field_names = appconfig_field_names() if field_names is None else field_names
    modified: list[dict[str, Any]] = []
    unknown: list[str] = []
    for key in sorted(config_dict.keys()):
        value = config_dict[key]
        if key not in field_names:
            unknown.append(key)
            continue
        if key in defaults and values_differ(value, defaults[key]):
            modified.append({"key": key, "value": value, "default": defaults[key]})
    return {"modified": modified, "unknown": unknown}


def infer_widget_kind(field_name: str, field_type: str, entry: dict[str, Any]) -> str:
    """Map a registry entry + field type to a concrete widget kind.

    Returns one of: ``checkbox`` | ``number`` | ``select`` | ``text``. Pure function
    (no Streamlit); used by the dashboard renderer and its smoke test.
    """
    t = (field_type or "").lower()
    if "bool" in t:
        return "checkbox"
    if field_name in LITERAL_OPTIONS:
        return "select"
    # Compound containers before scalar float/int (``list[float]`` contains "float").
    if "dict" in t or "list" in t:
        return "text"
    if entry.get("range") is not None:
        return "number"
    if "int" in t or "float" in t:
        return "number"
    return "text"
