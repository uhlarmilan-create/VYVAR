"""Project configuration for the variable-star processing system."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, fields as _dc_fields
import difflib
import json
import logging
import math
import os
import sqlite3
import sys
import textwrap
from pathlib import Path
from typing import Any

from pipeline_constants import GAIA_DB_DEFAULT, OBJECTS_DB_DEFAULT
from stats_core import parse_bool


def is_git_dev_checkout(install_root: Path) -> bool:
    """True when running from a git clone (dev/CI path uses install tree as data root)."""
    return (install_root / ".git").is_dir()


def default_release_data_dir() -> Path:
    """Default user data directory for bundled installs (RELEASE-2 / R1)."""
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA", "").strip()
        if local:
            return (Path(local) / "VYVAR").resolve()
    return (Path.home() / ".local" / "share" / "vyvar").resolve()


def resolve_data_root(install_root: Path) -> Path:
    """Resolve the user data root (config, DB, Archive, catalogs).

    Precedence: ``VYVAR_DATA_DIR`` env -> git dev checkout (install root) ->
    bundled install (platform default) when bundle markers/env present ->
    install root (tests and explicit ``project_root`` overrides).
    """
    override = os.environ.get("VYVAR_DATA_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    root = install_root.resolve()
    if is_git_dev_checkout(root):
        return root
    bundle_flag = os.environ.get("VYVAR_RELEASE_BUNDLE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    bundle_layout = (root / "RUNTIME_PIN.json").is_file() or (
        root / "python" / "python.exe"
    ).is_file() or (root / "python" / "bin" / "python3").is_file()
    if bundle_flag or bundle_layout:
        return default_release_data_dir()
    return root


def resolve_config_path(raw: str | Path, data_root: Path) -> str:
    """Resolve a config path: absolute as-is, relative against ``data_root`` (never CWD)."""
    s = str(raw or "").strip()
    if not s:
        return ""
    p = Path(s).expanduser()
    if not p.is_absolute():
        p = data_root / p
    return str(p.resolve())


def config_json_path(project_root: Path) -> Path:
    return project_root / "config.json"


# CONFIG-HUMAN-EDIT STEP 2: legacy config.json keys the loader migrates silently. These are
# NOT AppConfig fields, so the unknown-key typo warning must skip them (they have explicit
# migration paths in __post_init__). Two families: uppercase env-style aliases superseded by
# lowercase fields, and the WAVE-B scalar tier/aperture keys merged into structured keys.
# CFG-REVIEW-2026-10: keys renamed (old -> new); the old name still loads, with an INFO.
RENAMED_KEYS: dict[str, str] = {
    "snr_cog_isolation_fwhm": "comp_isolation_fwhm",
}


from config_removed_keys import KNOWN_REMOVED_KEYS  # noqa: E402  (G13-40: the table of removed keys)
from jsonc import parse_config_text, strip_jsonc_comments  # noqa: F401  (config API; the parser lives in the leaf jsonc)


DETECTION_THRESHOLD_SIGMA_DEFAULT: float = 4.5
DETECTION_THRESHOLD_SIGMA_MIN: float = 2.0
DETECTION_THRESHOLD_SIGMA_MAX: float = 20.0


def resolve_detection_threshold_sigma(raw: Any) -> float:
    """Coerce a detection threshold (N x sky sigma) to a finite float clamped to 2.0-20.0.

    Accepts a raw number or an object with ``detection_threshold_sigma`` (e.g. AppConfig), so
    the GUI, night runner and every pipeline stage resolve the single threshold identically.
    Non-numeric / non-finite / non-positive input falls back to the 4.5 default.
    """
    if raw is not None and not isinstance(raw, (int, float, str)):
        raw = getattr(raw, "detection_threshold_sigma", None)
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return DETECTION_THRESHOLD_SIGMA_DEFAULT
    if not math.isfinite(v) or v <= 0:
        return DETECTION_THRESHOLD_SIGMA_DEFAULT
    return max(DETECTION_THRESHOLD_SIGMA_MIN, min(DETECTION_THRESHOLD_SIGMA_MAX, v))


_APPCONFIG_FIELD_NAMES_CACHE: tuple[str, ...] | None = None


def _appconfig_field_names() -> tuple[str, ...] | None:
    """Public AppConfig field names, cached. Returns None if AppConfig is unavailable.

    Some report tests monkeypatch ``config.AppConfig`` with a plain factory function; in
    that case ``dataclasses.fields`` raises and we fall back to the last good cache (or
    None), so the best-effort typo net never breaks an unrelated test.
    """
    global _APPCONFIG_FIELD_NAMES_CACHE
    try:
        names = tuple(
            sorted(f.name for f in _dc_fields(AppConfig) if not f.name.startswith("_"))
        )
    except TypeError:
        return _APPCONFIG_FIELD_NAMES_CACHE
    _APPCONFIG_FIELD_NAMES_CACHE = names
    return names


def _warn_unknown_config_keys(data: dict[str, Any]) -> None:
    """Log a WARN for each config.json key that is neither a field nor a known legacy alias.

    Typo safety net for hand editors: names the unknown key and the closest registered
    key (difflib). Migrated/legacy keys stay silent (they are in the allowlist).
    """
    field_names = _appconfig_field_names()
    if field_names is None:
        return  # cannot resolve the known-key set right now; skip the best-effort typo net
    known = set(field_names)
    for key in data:
        if key in known:
            continue
        if key in KNOWN_REMOVED_KEYS:
            logging.info("config.json: %s", KNOWN_REMOVED_KEYS[key])
            continue
        near = difflib.get_close_matches(key, list(field_names), n=1)
        hint = f" (did you mean '{near[0]}'?)" if near else ""
        logging.warning(
            "config.json: unknown key '%s' is ignored%s. See docs/VYVAR_CONFIG_GUIDE_EN.md "
            "or run 'python dev/scripts/validate_config.py'.",
            key,
            hint,
        )


def load_config_json(project_root: Path) -> dict[str, Any]:
    return load_config_file(config_json_path(project_root))


def load_config_file(path: Path) -> dict[str, Any]:
    """Parse one config file (JSONC); a missing file is an empty dict (defaults)."""
    path = Path(path)
    if not path.exists():
        return {}
    try:
        data = parse_config_text(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logging.warning(
            "config.json could not be parsed (%s); using defaults. Run "
            "'python dev/scripts/validate_config.py' to locate the error.",
            exc,
        )
        return {}
    for old, new in RENAMED_KEYS.items():
        if old in data:
            val = data.pop(old)
            if new not in data:
                data[new] = val
            logging.info("config.json: %s renamed to %s (CFG-REVIEW-2026-10)", old, new)
    _warn_unknown_config_keys(data)
    return data


class ConfigPersistError(RuntimeError):
    """Raised when config.json persistence is attempted outside an explicit UI save action."""


# CONFIG-WRITE-GUARD (PARAM-OWNERSHIP-WAVE-A STEP 1): config.json must be persisted ONLY from
# explicit user save actions in the Streamlit UI layer. The headless / pipeline path may resolve
# and USE any values it needs, but must never write them back to config.json (run-effective values
# belong in provenance). UI save handlers wrap their write in ``ui_config_persist()``; every other
# caller (pipeline, night-run, baseline check) trips ``ConfigPersistError``.
_CONFIG_PERSIST_ALLOWED = False


@contextmanager
def ui_config_persist() -> Iterator[None]:
    """Context in which ``save_config_json`` is permitted (explicit UI save action only)."""
    global _CONFIG_PERSIST_ALLOWED
    prev = _CONFIG_PERSIST_ALLOWED
    _CONFIG_PERSIST_ALLOWED = True
    try:
        yield
    finally:
        _CONFIG_PERSIST_ALLOWED = prev


# CONFIG-HUMAN-EDIT STEP 3: config.json is written as a grouped, commented, JSONC-lite
# document so a user can edit it in a text editor without the UI. Sections follow pipeline
# order; within a section keys are ordered basic -> advanced -> expert then alphabetically.
_CONFIG_SECTION_ORDER = (
    "observer", "calibration", "qc", "alignment", "detection", "photometry",
    "comp_selection", "trust", "extinction", "reports", "export", "paths",
)
_CONFIG_SECTION_TITLES = {
    "observer": "Observer & export identity",
    "calibration": "Calibration",
    "qc": "Frame quality control (QC)",
    "alignment": "Alignment",
    "detection": "Detection, plate solving & masterstar",
    "photometry": "Photometry",
    "comp_selection": "Comparison-star selection",
    "trust": "Trust & quality flags",
    "extinction": "Atmospheric extinction & color",
    "reports": "Reports & HRD",
    "export": "Export",
    "paths": "File & catalog paths",
}
_CONFIG_TIER_ORDER = {"basic": 0, "advanced": 1, "expert": 2}
_CONFIG_COMMENT_WIDTH = 78


def _config_header_lines() -> list[str]:
    """The file-header comment block explaining the static/dynamic model + how to edit."""
    para = [
        "VYVAR config.json -- pipeline settings, safe to edit in a text editor.",
        "",
        "This file holds ONLY user-tunable pipeline settings. Two other kinds of values "
        "live elsewhere and are NOT in this file:",
        "  - Static observatory facts (site coordinates, telescope, camera, catalogs) live "
        "in the DATABASE and are managed in the app (Settings -> Observatory).",
        "  - Dynamic per-run values (gain, read noise, frame size, plate scale, filter, "
        "exposure) are read from the FITS headers at run time and appear in the report's "
        "Resolved Facts section.",
        "",
        "Editing without the UI: '//' line comments are allowed (they are ignored on load). "
        "Trailing commas and block comments are NOT allowed. Unknown keys are ignored with "
        "a warning that suggests the closest real key. After editing, validate with:",
        "    python dev/scripts/validate_config.py",
        "Full explanations of every key: docs/VYVAR_CONFIG_GUIDE_EN.md (English) and "
        "docs/VYVAR_CONFIG_GUIDE_CZ.md (Czech).",
        "",
        "NOTE: saving from the UI regenerates this file, its grouping and its comments from "
        "the parameter registry -- any custom comments you add here are not preserved.",
    ]
    bar = "// " + "=" * (_CONFIG_COMMENT_WIDTH - 3)
    lines = [bar]
    for p in para:
        if p == "":
            lines.append("//")
            continue
        for w in textwrap.wrap(p, width=_CONFIG_COMMENT_WIDTH - 3):
            lines.append("// " + w)
    lines.append(bar)
    return lines


def _comment_block(text: str, indent: str) -> list[str]:
    prefix = indent + "// "
    avail = max(20, _CONFIG_COMMENT_WIDTH - len(prefix))
    return [prefix + w for w in (textwrap.wrap(text, width=avail) or [""])]


def render_config_jsonc(data: dict[str, Any]) -> str:
    """Render ``data`` as the canonical grouped + commented config.json (JSONC-lite).

    Deterministic: file header, then sections in pipeline order each opened by a group
    comment, keys ordered basic->advanced->expert then alphabetical, every key preceded by
    its one-line registry help. Values are emitted verbatim (json) so a save->load round
    trip is value-preserving. Keys not in the registry go to a trailing 'Other' section so
    the writer never drops data.
    """
    try:
        import params_registry as _pr  # lazy: params_registry imports config at module load

        registry = _pr.load_registry()
        phase_help = _pr.load_phase_help()
    except Exception:  # noqa: BLE001 -- writer must never fail on registry issues
        registry = {}
        phase_help = {}

    by_phase: dict[str, list[str]] = {}
    other: list[str] = []
    for key in data:
        entry = registry.get(key)
        phase = entry.get("phase") if entry else None
        if phase in _CONFIG_SECTION_ORDER:
            by_phase.setdefault(phase, []).append(key)
        else:
            other.append(key)

    sections: list[tuple[str, str, list[str]]] = []
    for phase in _CONFIG_SECTION_ORDER:
        keys = by_phase.get(phase)
        if not keys:
            continue
        keys_sorted = sorted(
            keys,
            key=lambda k: (_CONFIG_TIER_ORDER.get((registry.get(k) or {}).get("tier"), 1), k),
        )
        sections.append((_CONFIG_SECTION_TITLES[phase], phase_help.get(phase, ""), keys_sorted))
    if other:
        sections.append(("Other", "", sorted(other)))

    total_keys = sum(len(keys) for _, _, keys in sections)
    lines: list[str] = list(_config_header_lines())
    lines.append("{")
    indent = "  "
    idx = 0
    for si, (title, ph, keys) in enumerate(sections):
        if si > 0:
            lines.append("")
        lines.append(f"{indent}// === {title} ===")
        if ph:
            lines.extend(_comment_block(ph, indent))
        for key in keys:
            entry = registry.get(key)
            help_txt = (entry or {}).get("help") if entry else ""
            if help_txt:
                lines.extend(_comment_block(help_txt, indent))
            idx += 1
            comma = "," if idx < total_keys else ""
            key_str = json.dumps(key, ensure_ascii=False)
            val_str = json.dumps(data[key], ensure_ascii=False)
            lines.append(f"{indent}{key_str}: {val_str}{comma}")
    lines.append("}")
    return "\n".join(lines) + "\n"


def write_bootstrap_config_json(data_root: Path, data: dict[str, Any]) -> None:
    """Write first-run ``config.json`` during release bootstrap (not UI-guarded)."""
    path = config_json_path(data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_config_jsonc(data), encoding="utf-8")


def materialize_fresh_config_json(install_root: Path, data_root: Path) -> None:
    """Materialize canonical ``config.json`` with every persisted key at code defaults.

    Used on first bundled launch instead of copying a trimmed template. The canonical
    grouped writer (``render_config_jsonc`` + ``AppConfig.to_json()``) is the same path
    the Settings UI uses on save.
    """
    prev_data = os.environ.get("VYVAR_DATA_DIR")
    os.environ["VYVAR_DATA_DIR"] = str(data_root.resolve())
    try:
        cfg = AppConfig(project_root=install_root.resolve())
        write_bootstrap_config_json(data_root, cfg.to_json())
    finally:
        if prev_data is None:
            os.environ.pop("VYVAR_DATA_DIR", None)
        else:
            os.environ["VYVAR_DATA_DIR"] = prev_data


def save_config_json(project_root: Path, data: dict[str, Any]) -> None:
    if not _CONFIG_PERSIST_ALLOWED:
        raise ConfigPersistError(
            "config.json may be persisted only from an explicit UI save action "
            "(wrap the write in config.ui_config_persist()). The pipeline/headless path "
            "must not write config.json; run-effective values belong in provenance."
        )
    write_bootstrap_config_json(project_root, data)


#: CODE-REVIEW-2026-10 step 7: the numeric config.json keys with their allowed range [lo, hi] -
#: the ONE place a range lives. The loader clamps to it (with a warning), the Settings page and
#: dev/scripts/validate_config.py read it through params_registry. Type = type of the default.
PARAM_RANGES: dict[str, tuple[float, float]] = {
    "masterdark_validity_days": (1, 3650),
    "masterflat_validity_days": (1, 3650),
    "calibration_master_ccd_temp_tolerance_c": (0.01, 20.0),
    "lc_quality_min_frames": (3, 500),
    "comp_trust_min_comps": (3, 20),
    "preprocess_sky_surface_order": (0, 2),
    "osc_channel_binning": (1, 4),
    "aperture_fwhm_factor": (0.25, 6.0),
    "annulus_inner_fwhm": (1.0, 10.0),
    "annulus_outer_fwhm": (1.5, 12.0),
    "detection_threshold_sigma": (DETECTION_THRESHOLD_SIGMA_MIN, DETECTION_THRESHOLD_SIGMA_MAX),
    "alignment_max_stars": (10, 200),  # G02-47: alignment uses at most the 200 brightest (dense fields)
    "phase01_comparison_n_comp_min": (2, 12),
    "phase01_comparison_n_comp_max": (3, 20),
    "snr_min": (3.0, 1000.0),
    "comp_peak_sat_frac": (0.5, 0.9),
    "phase01_comparison_max_comp_rms": (0.01, 0.5),
    "phase01_comparison_min_frames_frac": (0.05, 0.95),
    "comp_max_delta_bprp": (0.0, 5.0),
    "comp_isolation_fwhm": (1.0, 12.0),
    "comp_max_slope_mmag_hr": (0.0, 500.0),
    "comp_catalog_offset_max_mag": (0.0, 2.0),
    "per_frame_sat_min_clean_frac": (0.1, 1.0),
    "masterstar_best_of_n": (1, 25),
    "auto_fwhm_k_factor": (1.0, 4.0),
}

#: On / off config.json keys.
BOOL_KEYS: tuple[str, ...] = (
    "hrd_online_enrich_enabled",
    "hrd_nss_category_enabled",
    "save_lightcurve_png",
    "epsf_auto_run",
    "epsf_export_auto",
    "gs11_dilution_enabled",
    "tess_enabled",
    "transient_search_enabled",
    "qc_after_calibrate_enabled",
    "auto_fwhm_enabled",
)



def _coerce_number(key: str, raw: Any, kind: type, lo: float, hi: float) -> float | int | None:
    """``raw`` as ``kind`` clamped to [lo, hi]; None (with a warning) when it is not a number."""
    try:
        if isinstance(raw, bool):
            raise ValueError
        v = float(raw)
        if not math.isfinite(v):
            raise ValueError
    except (TypeError, ValueError):
        logging.warning("config.json: %s = %r is not a number; the default is used", key, raw)
        return None
    if kind is int:
        if v != int(v):
            logging.warning("config.json: %s = %r is not a whole number; %d is used", key, raw, int(v))
        v = float(int(v))
    c = max(float(lo), min(float(hi), v))
    if c != v:
        logging.warning("config.json: %s = %r is outside [%s, %s]; %s is used", key, raw, lo, hi, kind(c))
    return kind(c)


def _coerce_bool(key: str, raw: Any) -> bool | None:
    """The one flag vocabulary (stats_core.parse_bool, G02-39); unknown -> None with a warning."""
    b = parse_bool(raw)
    if b is not None:
        return b
    logging.warning("config.json: %s = %r is not true / false; the default is used", key, raw)
    return None


def config_rule_problems(values: dict[str, Any]) -> list[str]:
    """Rules between keys (CONFIG-ONE-2026-10, G13-08: one list for the loader, Settings and the validator).

    The loader warns for each and corrects the two count rules; Settings refuses to save."""
    errs: list[str] = []

    def g(k: str) -> Any:
        return values.get(k)

    if g("annulus_outer_fwhm") is not None and g("annulus_inner_fwhm") is not None:
        if float(g("annulus_outer_fwhm")) <= float(g("annulus_inner_fwhm")):
            errs.append("Sky annulus: the outer radius must be larger than the inner radius.")
        if g("aperture_fwhm_factor") is not None and float(g("annulus_inner_fwhm")) <= float(g("aperture_fwhm_factor")):
            errs.append("Sky annulus: the inner radius must be larger than the aperture radius.")
    if g("phase01_comparison_n_comp_max") is not None and g("phase01_comparison_n_comp_min") is not None:
        if int(g("phase01_comparison_n_comp_max")) < int(g("phase01_comparison_n_comp_min")):
            errs.append("Comparison stars: 'Most comparison stars' must be at least 'Fewest comparison stars'.")
        if g("comp_trust_min_comps") is not None and int(g("comp_trust_min_comps")) > int(g("phase01_comparison_n_comp_max")):
            errs.append("Trust: 'Comparison stars for green' cannot exceed 'Most comparison stars'.")
    tiers = g("comp_color_tiers")
    if tiers and g("comp_max_delta_bprp") is not None:
        if float(tiers[-1].get("bprp", 0.0)) > float(g("comp_max_delta_bprp")):
            errs.append("Comparison stars: colour class 3 cannot exceed 'Largest colour difference'.")
    return errs


@dataclass(slots=True)
class AppConfig:
    """Central application config: dataclass defaults, overridden by config.json on load."""

    # src_py/config.py -> repo root is parent.parent; project_root anchors config.json,
    # Archive/, CalibrationLibrary/, vyvar.sqlite3, catalogs/ discovery (all at repo root).
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    # CONFIG-ONE-2026-10 (G12-03): an explicit config file (``night_run --config``, AppConfig.from_file);
    # None = ``<data_root>/config.json``. Relative paths inside it still resolve against data_root.
    _config_file: Path | None = field(default=None, repr=False)

    # Calibration validity defaults (days)
    masterdark_validity_days: int = 90
    masterflat_validity_days: int = 200

    #: Max |DeltaT| ( degC) when matching library dark masters to light ``CCD_TEMP`` (``find_best_calibration_library_path``).
    calibration_master_ccd_temp_tolerance_c: float = 0.5

    #: Local Gaia DR3 SQLite (table ``gaia_dr3``), built by ``catalogs/build_gaia_catalog.py``
    #: Empty / missing key = the default file the builder writes (CONFIG-ONE-2026-10); no file there = no
    #: catalogue (the run gates check the file).
    gaia_db_path: str = GAIA_DB_DEFAULT
    #: PLATESOLVE-2026-10 blind plate-solving index (astro_index build); empty = next to the Gaia DB.
    astrometry_index_dir: str = ""

    #: HRD extreme objects: online lookups (Gaia TAP teff/logg and SIMBAD otype) for the candidates.
    hrd_online_enrich_enabled: bool = True
    #: Include Gaia NSS binary candidates in HRD Stage-1/2 (default off; reversible flag).
    hrd_nss_category_enabled: bool = False

    #: Objects catalog DB (tables ``vsx_data`` and ``exoplanet_data``; built by catalogs/objects_make.py).
    objects_db_path: str = OBJECTS_DB_DEFAULT
    #: VSX types to leave unmeasured (token-match; empty = inactive). See vsx_type_scope.py.
    vsx_out_of_scope_types: list[str] = field(default_factory=list)

    #: Aperture photometry: write PNG files (light curve, cutout, field map). ``False`` = CSV + summary only; the UI plots from the CSV.
    save_lightcurve_png: bool = False
    #: LC-quality frame floor for ``classify_lc_quality`` (aperture-photometry summary).
    lc_quality_min_frames: int = 20
    #: Trust-only comp floor (RED below); comparison-star selection uses ``phase01_comparison_n_comp_min``.
    comp_trust_min_comps: int = 5
    # Export reports (AAVSO + VAR.ASTRO.CZ)
    observer_name: str = "Unknown Observer"
    observer_code: str = ""
    #: User overrides: filter/setup name (uppercase key) -> AAVSO FILT code (e.g. ``"MYLUM": "CV"``).
    aavso_filter_map: dict[str, str] = field(default_factory=dict)
    # Observer location - used for BJD, airmass, lunar context
    #: Run-time only (not in config.json): LOCATION.ID of the default site (IS_DEFAULT), or the site
    #: chosen for the current run; 0 = no default site in the DB.
    observer_location_id: int = 0
    #: Mirror of the site's LOCATION row (NaN = no site; CODE-REVIEW step 7B: no built-in default site).
    observer_lat: float = float("nan")  # degrees N
    observer_lon: float = float("nan")  # degrees E
    observer_alt_m: float = float("nan")  # metres above sea level
    observer_location_name: str = ""  # display name; filled from DB on load when id > 0
    #: Night-run automatic ePSF stage after aperture photometry (EPSF-CHAIN-01B).
    #: Default OFF.
    #: ``NightRunParams.epsf`` True/False overrides; None reads this key.
    epsf_auto_run: bool = False
    #: EPSF-TARGETS-2026-10: targets RUN ePSF measures - ``ALL`` (every VSX target with aperture photometry,
    #: after ``vsx_out_of_scope_types``) or comma-separated VSX types / star names / catalog ids, ``*`` wildcard
    #: (e.g. ``EA, EB, EW`` or ``E*``). The ePSF tab can override it per run.
    epsf_targets: str = "ALL"
    #: EPSF-EXPORT-2026-10: write the ePSF light-curve files for the user (lightcurves_reports/epsf: bjd_tdb,
    #: delta_mag, err + header) automatically when RUN ePSF finishes. Off: the ePSF tab's button writes them.
    epsf_export_auto: bool = True
    #: Per calibrated light: subtract a source-masked polynomial sky surface (no clipping) in preprocess (0=off, default 2).
    preprocess_sky_surface_order: int = 2
    #: OSC: post-extraction NxN average binning on each channel plane (1-4; default 2). Total scale vs raw = 2 x N.
    osc_channel_binning: int = 2
    # NOTE: These are in units of **Gaussian FWHM** (not moment-FWHM).
    # Aperture/annulus radii are computed as factor x fwhm_gaussian_px.
    #: Legacy single aperture factor - used where multi-aperture (B+C) is not active.
    aperture_fwhm_factor: float = 1.35
    annulus_inner_fwhm: float = 2.7
    annulus_outer_fwhm: float = 5.2

    #: CFG-CLEAN 2026-10: the ONE star-detection threshold, N x sky sigma, used by every DAO
    #: detection (QC after calibration, alignment, VYVAR plate solve, MASTERSTAR pass 1, per-frame
    #: catalogs, zone-classifier T1). Replaces sips_dao_threshold_sigma, masterstar_dao_threshold_sigma,
    #: dao_detection_n_equiv, alignment_detection_sigma and qc_dao_detection_sigma. Clamp 2.0-20.0.
    detection_threshold_sigma: float = DETECTION_THRESHOLD_SIGMA_DEFAULT

    #: Frame alignment (DAO detection ladder): max brightest sources detected per frame (astroalign picks a subset).
    alignment_max_stars: int = 160

    #: Comparison-star selection (``photometry_core.select_comparison_stars_per_target``).
    #: No magnitude window and no maximum distance: candidates are ranked RMS -> colour -> distance.
    phase01_comparison_n_comp_min: int = 3
    phase01_comparison_n_comp_max: int = 8
    #: SNR-MIN-2026-10: faint limit for every star (targets, comparison and check stars):
    #: aperture SNR on MASTERSTAR below this -> not measured; a frame where the target SNR
    #: falls below it is not exported. Default 10 (sigma ~ 0.11 mag).
    snr_min: float = 10.0
    #: COMP-SAT-80-2026-10 / COMP-SAT-ONE-2026-10 (Milan 2026-10-29): a comparison star is not admitted
    #: when its raw peak reaches this fraction of the measured saturation level on the MASTERSTAR or on
    #: any frame (AAVSO CCD/CMOS guide 3.3.5 / 3.3.7: comps well inside the linear range). Targets: 0.90.
    comp_peak_sat_frac: float = 0.80
    phase01_comparison_max_comp_rms: float = 0.1
    phase01_comparison_min_frames_frac: float = 0.2
    #: Max |DeltaBP-RP| (Gaia) between target and comparison star (hard filter of the selection).
    comp_max_delta_bprp: float = 0.79
    #: Comparison-star colour classes 1-3: rows {"bprp": |dBP-RP| limit (Gaia)}; the next rung is
    #: ``comp_max_delta_bprp`` and above it the red-target T4 fallback (CFG-REVIEW-2026-10: the 4th row
    #: only mirrored comp_max_delta_bprp).
    comp_color_tiers: list[dict[str, float]] = field(
        default_factory=lambda: [{"bprp": 0.15}, {"bprp": 0.30}, {"bprp": 0.55}]
    )
    #: Aperture photometry: apply BP-RP colour-term correction (``auto`` = on for B/V/Rc broadband, off for L/Clear).
    apply_color_term: str = "auto"
    #: A comparison candidate whose nearest catalogue neighbour is closer than this x FWHM is a blend and
    #: is not a comparison star (``photometry_comp._select_comps_by_rms_then_color``). Was snr_cog_isolation_fwhm.
    comp_isolation_fwhm: float = 3.0
    #: Second-order extinction: ``off`` | ``literature``.
    k2_mode: str = "literature"
    #: Optional per-band k'' overrides (mag/airmass/BP-RP); empty dict uses ``k2_extinction`` converter.
    k2_defaults_bprp: dict[str, float] = field(default_factory=dict)

    #: Aperture photometry: exclude comparison stars with |linear slope| above this (mmag/hr) in stability check.
    comp_max_slope_mmag_hr: float = 5.0
    #: COMP-CATOFF-2026-10: exclude (and replace from the reserve) a comparison candidate whose
    #: catalogue - measured magnitude differs from the median of the candidates by more than this
    #: (mag); 0 = off.
    comp_catalog_offset_max_mag: float = 0.20

    # GS11 - Flux dilution correction
    gs11_dilution_enabled: bool = False

    #: Minimum fraction of clean (unsaturated) frames required to measure a target
    #: (per-frame saturation is always on, SAT-LIN-2026-10). Clamped to [0.1, 1.0].
    per_frame_sat_min_clean_frac: float = 0.5

    #: MASTERSTAR: stack N frames and pick best N for ePSF/catalog build.
    masterstar_best_of_n: int = 10

    #: ``True`` = download and analyse TESS FFI through lightkurve (TessCut), UI + ``tess_runner`` + pipeline hook.
    #: ``False`` = off - nothing is downloaded; log ``[TESS] skipped``.
    tess_enabled: bool = False

    #: TRANSIENT-2026-10 (Milan 2026-10-25): with the field variability search, also search the night's
    #: frames for new sources (novae, supernovae, outbursts; transient_field.py), compare every star with
    #: Gaia (brightened) and name the minor planets / comets in the field (MPC files in catalogs/).
    transient_search_enabled: bool = False

    #: Post-calibration QC metrics (HFR, stars, sky) on each calibrated light; measurements only.
    qc_after_calibrate_enabled: bool = True

    #: FITS QA dashboard: derive the default FWHM limit from the MAD (median + k x sigma_MAD).
    auto_fwhm_enabled: bool = True
    #: Night FWHM limit = median + k x sigma_MAD; k clamped to 1..4 on load.
    auto_fwhm_k_factor: float = 3.0

    # Paths derived from config.json (must stay after all init=True fields for dataclass(slots=True)).
    data_root: Path = field(init=False)
    archive_root: Path = field(init=False)
    calibration_library_root: Path = field(init=False)
    database_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.data_root = resolve_data_root(self.project_root)
        data = load_config_file(self._config_file) if self._config_file is not None else load_config_json(self.data_root)

        # A blank ("" / whitespace) path value means "use the project-root default" -- the
        # same as omitting the key. This keeps a relocated/fresh install working: the
        # installer and hand-editors may blank these to drop the author's absolute paths,
        # and Path("") would otherwise resolve to "." (e.g. database_path="." would make
        # sqlite3.connect fail on first run).
        def _path_or_default(key: str, default: Path) -> Path:
            raw = str(data.get(key) or "").strip()
            if not raw:
                return default
            resolved = resolve_config_path(raw, self.data_root)
            return Path(resolved) if resolved else default

        self.archive_root = _path_or_default("archive_root", self.data_root / "Archive")
        self.calibration_library_root = _path_or_default(
            "calibration_library_root", self.data_root / "CalibrationLibrary"
        )
        self.database_path = _path_or_default(
            "database_path", self.data_root / "vyvar.sqlite3"
        )
        # An empty / missing key means the default file (the one the catalogue builder writes).
        self.gaia_db_path = resolve_config_path(
            str(data.get("gaia_db_path") or "").strip() or GAIA_DB_DEFAULT,
            self.data_root,
        )
        self.astrometry_index_dir = resolve_config_path(
            str(data.get("astrometry_index_dir", "") or "").strip(),
            self.data_root,
        )
        self.objects_db_path = resolve_config_path(
            str(data.get("objects_db_path") or "").strip() or OBJECTS_DB_DEFAULT,
            self.data_root,
        )

        # Table-driven keys: a value that is not a number / not true-false keeps the default,
        # a number outside its range is clamped; both with a warning naming the key.
        for key in BOOL_KEYS:
            if key in data:
                b = _coerce_bool(key, data[key])
                if b is not None:
                    setattr(self, key, b)
        for key, (lo, hi) in PARAM_RANGES.items():
            if key in data:
                kind = _PARAM_KINDS[key]
                v = _coerce_number(key, data[key], kind, lo, hi)
                if v is not None:
                    setattr(self, key, v)

        # Rules between keys (one list, config_rule_problems): every broken rule is a WARNING; the annulus
        # and the two count rules are corrected (G13-35: no silent correction).
        for _msg in config_rule_problems(
            {
                "annulus_inner_fwhm": self.annulus_inner_fwhm,
                "annulus_outer_fwhm": self.annulus_outer_fwhm,
                "aperture_fwhm_factor": self.aperture_fwhm_factor,
                "phase01_comparison_n_comp_min": self.phase01_comparison_n_comp_min,
                "phase01_comparison_n_comp_max": self.phase01_comparison_n_comp_max,
                "comp_trust_min_comps": self.comp_trust_min_comps,
                "comp_color_tiers": data.get("comp_color_tiers") if isinstance(data.get("comp_color_tiers"), list) else None,
                "comp_max_delta_bprp": self.comp_max_delta_bprp,
            }
        ):
            logging.warning("config.json: %s", _msg)
        if self.annulus_outer_fwhm <= self.annulus_inner_fwhm:
            logging.warning("config.json: annulus_outer_fwhm = annulus_inner_fwhm + 1 is used")
            self.annulus_outer_fwhm = self.annulus_inner_fwhm + 1.0
        if int(self.phase01_comparison_n_comp_max) < int(self.phase01_comparison_n_comp_min):
            logging.warning("config.json: phase01_comparison_n_comp_max = phase01_comparison_n_comp_min is used")
            self.phase01_comparison_n_comp_max = int(self.phase01_comparison_n_comp_min)
        if int(self.comp_trust_min_comps) > int(self.phase01_comparison_n_comp_max):
            logging.warning("config.json: comp_trust_min_comps = phase01_comparison_n_comp_max is used")
            self.comp_trust_min_comps = int(self.phase01_comparison_n_comp_max)

        # Structured and text keys.
        _voos = data.get("vsx_out_of_scope_types", self.vsx_out_of_scope_types)
        if _voos is None:
            self.vsx_out_of_scope_types = []
        elif isinstance(_voos, str):
            self.vsx_out_of_scope_types = [p.strip() for p in _voos.split(",") if p.strip()]
        elif isinstance(_voos, (list, tuple)):
            self.vsx_out_of_scope_types = [str(p).strip() for p in _voos if str(p).strip()]
        else:
            self.vsx_out_of_scope_types = []
        _obn = data.get("observer_name")
        if _obn is None or str(_obn).strip() == "":
            _obn = self.observer_name
        self.observer_name = str(_obn or "").strip() or "Unknown Observer"
        self.observer_code = str(data.get("observer_code", self.observer_code) or "").strip()
        _ffm = data.get("aavso_filter_map", {})
        if isinstance(_ffm, dict):
            self.aavso_filter_map = {
                str(k).strip().upper(): str(v).strip()
                for k, v in _ffm.items()
                if str(k).strip() and str(v).strip()
            }
        else:
            self.aavso_filter_map = {}
        self.epsf_targets = str(data.get("epsf_targets", self.epsf_targets) or "ALL").strip() or "ALL"
        _act = str(data.get("apply_color_term", self.apply_color_term) or "auto").strip().lower()
        if parse_bool(_act) is True:
            self.apply_color_term = "on"
        elif parse_bool(_act) is False:
            self.apply_color_term = "off"
        else:
            if _act != "auto":  # G13-35: a typo ("atuo") is not silently read as auto
                logging.warning("config.json: apply_color_term = %r is not on / off / auto; auto is used", _act)
            self.apply_color_term = "auto"
        _k2m = str(data.get("k2_mode", self.k2_mode) or "literature").strip().lower()
        if parse_bool(_k2m) is False or _k2m == "none":
            self.k2_mode = "off"
        else:
            if _k2m != "literature":
                logging.warning("config.json: k2_mode = %r is not literature / off; literature is used", _k2m)
            self.k2_mode = "literature"
        _k2o = data.get("k2_defaults_bprp", self.k2_defaults_bprp)
        if isinstance(_k2o, dict):
            parsed: dict[str, float] = {}
            for k, v in _k2o.items():
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(fv):
                    parsed[str(k)] = fv
            self.k2_defaults_bprp = parsed
        # comp_color_tiers: colour classes 1-3. Older files carry a 4th row (it only mirrored
        # comp_max_delta_bprp) and a per-row weight "w" (never used: Broeg weights) - both are dropped.
        _cct_raw = data.get("comp_color_tiers")
        if isinstance(_cct_raw, list) and _cct_raw:
            _rows: list[dict[str, float]] = []
            for _row in _cct_raw[:3]:
                try:
                    _rows.append({"bprp": max(0.02, min(5.0, float(_row["bprp"])))})
                except (KeyError, TypeError, ValueError):
                    _rows = []
                    break
            if len(_rows) == 3:
                self.comp_color_tiers = _rows
            else:
                logging.warning("config.json: comp_color_tiers needs 3 rows {\"bprp\": limit}; defaults used")

        # Run-time values (not in config.json).
        # SETTINGS-2026-10: the site is not stored in config.json; the default site is the
        # LOCATION row with IS_DEFAULT = 1 (Database Explorer), a run may choose another one.
        from database import get_default_observer_location_id  # noqa: PLC0415

        self.observer_location_id = get_default_observer_location_id(str(self.database_path))
        # WAVE-B STEP 5 (DELETE-DB-DUP): observer_lat/lon/alt_m/name are DB-authoritative
        # (draft LOCATION via observer_location_id). They are no longer loaded from or saved to
        # config.json; the fields remain as run-time hydrated mirrors (block below) with the
        # NaN (no site) when the DB has none. Science reads the draft LOCATION row, not cfg.
        if self.observer_location_id > 0:
            try:
                from database import get_observer_location_by_id

                loc = get_observer_location_by_id(
                    str(self.database_path), int(self.observer_location_id)
                )
                if loc is not None:
                    # DB LOCATION is authoritative (WAVE-B STEP 5): overwrite the mirror
                    # unconditionally when a row is found.
                    self.observer_location_name = str(loc.get("name") or "")
                    self.observer_lat = float(loc.get("lat", 0.0) or 0.0)
                    self.observer_lon = float(loc.get("lon", 0.0) or 0.0)
                    self.observer_alt_m = float(loc.get("alt_m", 0.0) or 0.0)
            except (sqlite3.Error, TypeError, ValueError) as exc:
                logging.getLogger(__name__).warning(
                    "Observer location DB hydrate failed (location_id=%s); "
                    "no observing site until one is chosen: %s",
                    self.observer_location_id,
                    exc,
                )

    # --- structured-key accessors (WAVE-B STEP 4) ---------------------------------- #
    def comp_tier_bprp_limits(self) -> list[float]:
        """|dBP-RP| colour limit per comp tier, row order, from ``comp_color_tiers``."""
        return [float(t.get("bprp", 0.0)) for t in self.comp_color_tiers]

    @classmethod
    def from_file(cls, path: str | Path) -> "AppConfig":
        """Config loaded from exactly this file (``--config``); a missing file is an error."""
        p = Path(path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Config file not found: {p}")
        return cls(_config_file=p)

    def to_json(self) -> dict[str, Any]:
        """The config.json keys (paths, BOOL_KEYS, PARAM_RANGES, structured and text keys)."""
        out: dict[str, Any] = {
            "archive_root": str(self.archive_root),
            "calibration_library_root": str(self.calibration_library_root),
            "database_path": str(self.database_path),
            "gaia_db_path": str(self.gaia_db_path or ""),
            "astrometry_index_dir": str(self.astrometry_index_dir or ""),
            "objects_db_path": str(self.objects_db_path or ""),
        }
        for key in BOOL_KEYS:
            out[key] = bool(getattr(self, key))
        for key in PARAM_RANGES:
            v = getattr(self, key)
            out[key] = int(v) if isinstance(v, int) else float(v)
        out.update(
            {
                "vsx_out_of_scope_types": list(self.vsx_out_of_scope_types),
                "observer_name": str(self.observer_name),
                "observer_code": str(self.observer_code),
                "aavso_filter_map": dict(self.aavso_filter_map),
                "epsf_targets": str(self.epsf_targets),
                "apply_color_term": str(self.apply_color_term),
                "k2_mode": str(self.k2_mode),
                "k2_defaults_bprp": dict(self.k2_defaults_bprp),
                "comp_color_tiers": [{"bprp": float(t.get("bprp", 0.0))} for t in self.comp_color_tiers],
            }
        )
        return out

    # Backward-compatible alias (some callers expect to_dict()).
    def to_dict(self) -> dict[str, Any]:
        return self.to_json()

    def ensure_base_dirs(self) -> None:
        """Create base directories required by file-first workflow."""
        self.archive_root.mkdir(parents=True, exist_ok=True)
        self.calibration_library_root.mkdir(parents=True, exist_ok=True)


#: int / float per PARAM_RANGES key, from the dataclass default (a value is loaded as that type).
_PARAM_KINDS: dict[str, type] = {
    f.name: type(f.default) for f in _dc_fields(AppConfig) if f.name in PARAM_RANGES
}
assert set(_PARAM_KINDS) == set(PARAM_RANGES), sorted(set(PARAM_RANGES) - set(_PARAM_KINDS))


