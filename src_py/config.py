"""Project configuration for the variable-star processing system."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, fields as _dc_fields
import copy
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
_LEGACY_CONFIG_KEYS: frozenset[str] = frozenset({
    "GAIA_DB_PATH",
    "BLIND_INDEX_PATH",
    "BLIND_INDEX_FINE_PATH",
    "BLIND_INDEX_WIDE_PATH",
    "VSX_LOCAL_DB_PATH",
    "EXOPLANET_LOCAL_DB_PATH",
    "aperture_fwhm_factor_small",
    "aperture_fwhm_factor_large",
    "aperture_fwhm_factor_medium",
    "comp_tier1_bprp_limit",
    "comp_tier2_bprp_limit",
    "comp_tier3_bprp_limit",
    "comp_tier4_bprp_limit",
    "comp_tier1_weight",
    "comp_tier2_weight",
    "comp_tier3_weight",
    "comp_tier4_weight",
    "phase01_tier1_mag",
    "phase01_tier2_mag",
    "phase01_tier3_mag",
    "phase01_tier4_mag",
})


# Keys removed from AppConfig; load_config_json logs INFO once per key (no difflib WARN).
KNOWN_REMOVED_KEYS: dict[str, str] = {
    "skip_processed_directory": (
        "skip_processed_directory removed 2026-07; skip behavior is now always on"
    ),
    "vsx_variable_targets_mag_limit": (
        "vsx_variable_targets_mag_limit removed 2026-07; VSX scope is detection-limited (DAO+Gaia match)"
    ),
    "phase01_match_radius_arcsec": (
        "phase01_match_radius_arcsec removed 2026-07; Phase 0 uses catalog_id identity join (PHASE0-IDENTITY-GATE)"
    ),
    "aperture_selection_criterion": (
        "aperture_selection_criterion removed 2026-08-31 CONSOLIDATE-01B; scatter/SNR diagnostic table deleted"
    ),
    "aperture_scatter_r_min_px": (
        "aperture_scatter_r_min_px removed 2026-08-31 CONSOLIDATE-01B; scatter ladder module deleted"
    ),
    "aperture_scatter_r_max_px": (
        "aperture_scatter_r_max_px removed 2026-08-31 CONSOLIDATE-01B; scatter ladder module deleted"
    ),
    "aperture_scatter_r_step_px": (
        "aperture_scatter_r_step_px removed 2026-08-31 CONSOLIDATE-01B; scatter ladder module deleted"
    ),
    "global_comp_pool_enabled": (
        "global_comp_pool_enabled removed 2026-09-01 CONSOLIDATE-01D; global pool is always on (COMP-POOL-01)"
    ),
    "export_err_mode": (
        "export_err_mode removed 2026-09-01 CONSOLIDATE-01D; exported err is always ERR-CALIB calibrated"
    ),
    "err_background_mode": (
        "err_background_mode removed 2026-09-01 CONSOLIDATE-01D; empirical empty-aperture term is always on (F-BINGAIN-1); Howell math remains as missing-sigma fallback"
    ),
}


def strip_jsonc_comments(text: str) -> str:
    """Remove ``//`` line comments that sit OUTSIDE string literals (JSONC-lite).

    A small character state machine so ``//`` inside a string value (e.g. a URL like
    ``"http://x"``) is preserved. Only ``//`` to end-of-line is stripped; block comments
    ``/* */`` and trailing commas are intentionally NOT supported so the format stays
    strict elsewhere and diffs/tools stay sane. Newlines are preserved so line numbers in
    validator error messages stay accurate.
    """
    out: list[str] = []
    i = 0
    n = len(text)
    in_str = False
    escape = False
    while i < n:
        c = text[i]
        if in_str:
            out.append(c)
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = False
            i += 1
            continue
        if c == '"':
            in_str = True
            out.append(c)
            i += 1
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] not in "\r\n":
                i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def parse_config_text(text: str) -> dict[str, Any]:
    """Parse config.json text tolerating ``//`` line comments; raises json.JSONDecodeError."""
    parsed = json.loads(strip_jsonc_comments(text))
    if not isinstance(parsed, dict):
        raise json.JSONDecodeError("config.json top level must be a JSON object", text, 0)
    return parsed


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
    known = set(field_names) | _LEGACY_CONFIG_KEYS
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
    path = config_json_path(project_root)
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
    _warn_unknown_config_keys(data)
    return data


def resolve_comp_sparse_fallback_enabled(cfg: AppConfig | None) -> bool:
    """True when per-target sparse comp fallback is enabled (includes legacy alias)."""
    if cfg is None:
        return False
    if bool(getattr(cfg, "comp_sparse_fallback_enabled", False)):
        return True
    return bool(getattr(cfg, "comp_iterative_clip_enabled", False))


def resolve_comp_sparse_fallback_min(
    cfg: AppConfig | None,
    *,
    n_comp_min: int,
    n_comp_max: int,
) -> int:
    """Minimum default-path comp count before sparse fallback may run."""
    raw = 0
    if cfg is not None:
        try:
            raw = int(getattr(cfg, "comp_sparse_fallback_min", 0) or 0)
        except (TypeError, ValueError):
            raw = 0
    if raw <= 0:
        raw = int(n_comp_min)
    return max(2, min(int(n_comp_max), int(raw)))


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
    "comp_selection", "trust", "extinction", "reports", "export", "system", "paths",
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
    "system": "System & performance",
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


def recommended_vyvar_parallel_workers(*, reserve_ram_gb: float = 1.5) -> int:
    """Jednotny pocet workerov pre QC, preprocess, combined, per-frame CSV (zaklad pred RAM stropom), alignment, calibrate MP.

    Berie minimum z CPU-heuristiky a odhadu podla volnej RAM (rovnaky odhad pamate ako pri per-frame exporte),
    aby jedna hodnota bola bezpecna v celom workflow.
    """
    n = os.cpu_count()
    if n is None or n < 1:
        n = 4
    if n <= 1:
        cpu_cap = 1
    else:
        cpu_cap = max(1, min(32, min(n - 1, 16, max(1, n // 2))))
    h, wpx = 2048, 2048
    per_worker = max(int(h * wpx * 4 * 3), 1)
    try:
        import psutil

        reserve = int(max(0.0, float(reserve_ram_gb)) * (1024**3))
        avail = int(psutil.virtual_memory().available) - reserve
        if avail <= 0:
            ram_cap = 1
        else:
            ram_cap = max(1, min(32, avail // per_worker))
    except Exception:  # noqa: BLE001
        ram_cap = 32
    return max(1, min(32, min(cpu_cap, ram_cap)))


@dataclass(slots=True)
class AppConfig:
    """Central application config.

    SQLite schema is intentionally not defined yet.
    """

    # src_py/config.py -> repo root is parent.parent; project_root anchors config.json,
    # Archive/, CalibrationLibrary/, vyvar.sqlite3, GAIA_DR3/ discovery (all at repo root).
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)

    # Calibration validity defaults (days)
    masterdark_validity_days: int = 90
    masterflat_validity_days: int = 200

    #: Estimated field diameter in degrees for VYVAR Gaia plate solve (used as **minimum** hint only).
    #: The actual Gaia kuzel pre cely cip sa odvodzuje z FOCALLEN+PIXSIZE+NAXIS (pozri ``catalog_cone_radius_deg_from_optics``
    #: v ``utils.py`` a ``solve_wcs_with_local_gaia`` v ``vyvar_platesolver.py``); tato hodnota len zaruci spodnu hranicu.
    plate_solve_fov_deg: float = 1.0

    #: When applying masters from CalibrationLibrary, assume this **sensor** binning in the stored FITS
    #: (typically ``1`` = full resolution). Lights with ``XBINNING`` 2x2 are matched in RAM (temporary resample).
    #: JSON ``null``: read ``XBINNING`` from each master FITS (e.g. Bin2 library files with 2x2 lights).
    calibration_library_native_binning: int | None = 1
    #: Max |DeltaT| ( degC) when matching library dark masters to light ``CCD_TEMP`` (``find_best_calibration_library_path``).
    calibration_master_ccd_temp_tolerance_c: float = 0.5

    #: Path to local Gaia DR3 SQLite database (must contain table ``gaia_dr3`` with indexes on ra/dec).
    gaia_db_path: str = ""

    #: HRD extreme-object online enrichment (Gaia TAP teff/logg for candidates only).
    hrd_online_enrich_enabled: bool = True
    #: HRD SIMBAD otype refinement for extreme candidates (additive label only).
    hrd_simbad_enrich_enabled: bool = True
    #: Max Stage-1 extreme candidates sent to online enrichment (1..100).
    hrd_enrich_max_candidates: int = 20
    #: Per-attempt Gaia TAP timeout (seconds) for HRD online enrichment.
    hrd_enrich_tap_timeout_s: float = 20.0
    #: HRD parallax floor (mas) for M_G reliability; SNR is the primary quality filter.
    hrd_parallax_min_mas: float = 0.15
    #: HRD parallax SNR floor for M_G reliability.
    hrd_parallax_snr_min: float = 5.0
    #: Max table rows per Stage-2 category label (1..20).
    hrd_max_per_category: int = 3
    #: Min reserved Stage-1 enrich slots per candidate net (0..20); NSS net filled last.
    hrd_min_per_net: int = 4
    #: Include Gaia NSS binary candidates in HRD Stage-1/2 (default off; reversible flag).
    hrd_nss_category_enabled: bool = False
    #: Gaia DSC probability floor for HRD "likely" white-dwarf tier (clamp 0.5..1.0).
    hrd_dsc_confirm_prob: float = 0.90
    #: HRD catalog-color field tinting (Gaia BP-RP chrominance x mono luminance).
    hrd_color_field_enabled: bool = True
    #: Desaturation toward white for catalog-color field (0=gray, 1=full Planckian chroma).
    hrd_color_saturation: float = 0.85
    #: Highlight handling: ``soft`` (Reinhard L roll + hue-preserving scale) or ``scale`` only.
    hrd_color_highlight_mode: str = "soft"
    #: Chroma SNR gate softness (0=disabled, 12g behavior); higher = color only on bright signal.
    hrd_color_chroma_snr: float = 3.0
    #: White point: ``d65`` absolute Planckian or ``field_median`` relative von Kries scaling.
    hrd_color_white_point: str = "field_median"
    #: Chroma distance-from-white boost (display enhancement; caption-disclosed).
    hrd_color_chroma_boost: float = 2.2
    #: Local-background box size (px) for chroma SNR gate grid; clamp 32..512.
    hrd_color_bg_box_px: int = 96
    #: Local-background box size (arcsec). None -> fall back to ``hrd_color_bg_box_px`` (legacy px).
    hrd_color_bg_box_arcsec: float | None = None

    #: Fine blind index (Newton / ~1.3 arcsec/px rigs); built by ``build_blind_index.py``.
    blind_index_fine_path: str = ""
    #: Wide blind index (Carl-Zeiss / ~9.77 arcsec/px rigs); built by ``build_blind_index.py``.
    blind_index_wide_path: str = ""
    #: Deprecated alias of ``blind_index_fine_path`` (``single`` mode and legacy readers).
    blind_index_path: str = ""
    #: ``auto`` = scale-aware tier order + verify; ``series_all`` = all tiers; ``single`` = fine path only.
    blind_index_select_mode: str = "auto"
    #: Blind solver: geometric verification of top-N vote candidates (astrometry.net-style).
    blind_verify_enabled: bool = True
    blind_verify_top_n: int = 15
    blind_verify_match_tol_px: float = 2.5
    #: Blind verify match tolerance (arcsec). None -> fall back to ``blind_verify_match_tol_px``.
    blind_verify_match_tol_arcsec: float | None = None
    blind_verify_min_matches: int = 12
    blind_verify_min_fraction: float = 0.15
    #: In-memory bright Gaia + KDTree for blind verify (one load per solve).
    blind_verify_inmemory_catalog: bool = True
    #: G-band limit for verify in-memory catalog (default matches wide index depth).
    verify_mag_limit: float = 14.0
    #: Stop verify sweep once an accepted candidate reaches this many matches (see also early floor).
    blind_verify_early_accept: int = 30
    #: Minimum matches before early-exit (``max(early_accept, min_matches * 6)`` if unset here).
    blind_verify_early_floor: int = 0
    #: Early-exit when accepted candidate fraction >= this (0 = use absolute ``early_floor`` only).
    blind_verify_early_fraction: float = 0.20
    #: Brightest central stars used for blind image kNN triangles (matches index local kNN).
    blind_img_star_budget: int = 80
    #: Image star pick: ``per_cell`` (mirror index SPC/cell) or ``central`` (legacy rig-prior cone).
    blind_img_select_mode: str = "per_cell"
    #: When True, use known plate scale / FOV as hard gates (pre-vote ratio + verify WCS scale).
    blind_use_rig_prior: bool = True
    # WAVE-B STEP 6: blind_prefilter_min, blind_scale_tol_frac, blind_cluster_{min_votes,eps_deg,
    # min_samples,vote_span,coherence_cap} hardcoded as solver internals (module constants in
    # vyvar_blind_solver.py / vyvar_platesolver.py).

    #: Path to local VSX subset SQLite (table ``vsx_data``: oid, ra_deg, dec_deg, ...) for variable-star flags.
    vsx_local_db_path: str = ""
    #: VSX types to leave unmeasured (token-match; empty = inactive). See vsx_type_scope.py.
    vsx_out_of_scope_types: list[str] = field(default_factory=list)

    #: Path to local exoplanet host SQLite (``exoplanet_data``: NASA Exoplanet Archive snapshot).
    exoplanet_local_db_path: str = "exoplanets/vyvar_exoplanet_local.db"
    #: Per-detection exoplanet host cross-match tolerance (arcsec); informational label only.
    exoplanet_match_max_sep_arcsec: float = 3.0

    #: After a cone query, keep at most this many catalog rows (brightest by ``mag``) to avoid RAM/CPU freeze.
    catalog_query_max_rows: int = 15_000

    #: Use ``photutils`` circular aperture + annulus sky (replaces DAO ``flux`` in sidecar CSV when enabled).
    aperture_photometry_enabled: bool = True
    #: Faza 2A: ukladat PNG (lightcurve, cutout, field map). ``False`` = len CSV + summary; UI pouziva Plotly z CSV.
    save_lightcurve_png: bool = False
    #: Diagnostic only: ``True`` = pre-TODO-29 order (airmass fit -> outlier detect). Default ``False`` keeps outlier -> airmass.
    phase2a_airmass_before_outlier: bool = False
    #: TODO-35: SysRem (Tamuz et al. 2005) on exported ``lightcurve_*.csv`` after Phase 2A.
    sysrem_enabled: bool = False
    sysrem_n_iter: int = 3
    #: Post-Phase 2A comp-star LOO QA (Sokolovsky locus); metadata only, no photometry changes.
    comp_qa_enabled: bool = True
    #: Post-Phase 2A trust flag (GREEN/YELLOW/RED); metadata + report/export notes only.
    trust_flag_enabled: bool = True
    #: LC-quality frame floors for ``classify_lc_quality`` (Phase 2A summary).
    lc_quality_min_frames: int = 20
    lc_quality_short_min_frames: int = 3
    lc_quality_min_normal_frac: float = 0.5
    #: Trust-only comp floor (RED below); Phase-1 selection uses ``phase01_comparison_n_comp_min``.
    comp_trust_min_comps: int = 5
    #: Minimum check-star epochs before trust scatter thresholds apply.
    check_star_min_epochs: int = 5
    #: Sparse-trust CI band thresholds (dev/results/specs/VYVAR_SPARSE_TRUST_SPEC.md).
    sparse_trust_T_green: float = 1.5
    sparse_trust_T_red: float = 4.0
    sparse_trust_X2_RED: float = 0.0004
    #: Artefact floor for check-star selection metric (comp_rms / p2p_rms).
    check_select_rms_floor: float = 1e-4
    #: Phase-1 comp selection: drop candidates with comp_rms below this (isolated_bin artefact).
    comp_select_rms_floor: float = 1e-6
    # Export reports (AAVSO + VAR.ASTRO.CZ)
    observer_name: str = "Unknown Observer"
    observer_code: str = ""
    #: Legacy mirror - synced from ``observer_code`` in ``__post_init__`` (kept for older callers).
    aavso_observer_code: str = "UMIA"
    #: User overrides: filter/setup name (uppercase key) -> AAVSO FILT code (e.g. ``"MYLUM": "CV"``).
    aavso_filter_map: dict[str, str] = field(default_factory=dict)
    # Observer location - used for BJD, airmass, lunar context
    observer_location_id: int = 0  # FK to LOCATION table; 0 = unset (no default site)
    observer_lat: float = 50.1121658  # degrees N
    observer_lon: float = 14.6982547  # degrees E
    observer_alt_m: float = 275.0  # metres above sea level
    observer_location_name: str = ""  # display name; filled from DB on load when id > 0
    #: Fallback pixel scale (arcsec/px) for export headers if FITS/WCS is unavailable.
    export_arcsec_per_px: float = 1.3
    #: Opt-in ePSF fitting on per-frame catalogs (adds ``psf_*`` columns; requires ``masterstar_epsf.fits``).
    psf_photometry_enabled: bool = False
    #: Night-run automatic ePSF stage after aperture photometry (EPSF-CHAIN-01B).
    #: Default OFF. Distinct from ``psf_photometry_enabled`` (Phase 2A psf_* columns).
    #: ``NightRunParams.epsf`` True/False overrides; None reads this key.
    epsf_auto_run: bool = False
    #: ePSF spatial variation order when photutils EPSFBuilder supports it:
    #: 0 = single global ePSF (default; sufficient for well-corrected optics).
    #: 1 = linear spatial variation (better for Newton/fast optics with field coma).
    #: 2 = quadratic (rarely needed for ground-based amateur setups).
    #: Note: per-set spatial_order planned for TODO-MULTISET.
    psf_spatial_order: int = 0
    #: Reduced chi^2 cutoff for PSF fit acceptance (``psf_fit_ok``).
    psf_chi2_threshold: float = 50.0
    #: F6 ePSF aperture-correction policy (EPSF-AC-02). ``p4_none`` stamps
    #: uncorrected fit flux (``psf_ac_factor=1``). ``chi2_lt5_legacy`` is the
    #: named fallback: median DAO/PSF among chi2<5 stars (EPSF-AC-01 A2 defect).
    psf_ac_policy: str = "p4_none"
    #: Internal PSF LC ZP membership (INV-PSF-LC-PIN-01). Production default
    #: ``fit_ok_for_zp`` on validated rigs only (EPSF-ZP-OK-01-WIRE v2).
    psf_zp_membership: str = "fit_ok_for_zp"
    #: Rig identity keys ``equipment_id:telescope_id`` allowed to use fit_ok_for_zp.
    #: Draft 516/517 wide pair is ``1:1``. Other rigs stay fit_ok_strict.
    psf_zp_for_zp_validated_rigs: list[str] = field(default_factory=lambda: ["1:1"])
    #: PSF crowded-field joint-fit (SourceGrouper). Default OFF -> production unchanged.
    #: When enabled, each PSF target is fit jointly with its close neighbours so a
    #: bright neighbour does not corrupt a faint blended target.
    psf_grouper_enabled: bool = False
    #: SourceGrouper min_separation, in units of FWHM_px (sources closer than this group).
    psf_group_sep_fwhm: float = 1.5
    #: Neighbour inclusion radius for joint-fit init params, in units of FWHM_px.
    psf_neighbor_include_fwhm: float = 3.0
    #: NEIGHBOR-SUB: joint-fit neighbour, subtract model, aperture residual. Default OFF.
    psf_neighbor_sub_enabled: bool = False
    #: Reduced chi2 proxy ceiling for joint neighbour-sub fit (fit-quality refuse).
    neighbor_sub_chi2_max: float = 120.0
    #: Residual RMS ceiling (ADU) in target region after joint fit.
    neighbor_sub_residual_rms_max: float = 150.0
    #: Cheap pre-gate: sep below this (FWHM) triggers refuse unless fit is excellent.
    neighbor_sub_refuse_sep_fwhm: float = 0.8
    #: Max fitted centroid shift from catalog position, in units of FWHM.
    neighbor_sub_centroid_max_fwhm: float = 1.0
    #: Neighbour is a contaminant when delta_mag_nn <= this (mag_neighbour - mag_target).
    neighbor_sub_nn_contam_dmag: float = 2.5
    #: Catalog anchor: refuse if fitted neighbour mag is brighter than nn_mag by more than this.
    neighbor_sub_max_neighbor_overmag: float = 0.3
    #: Catalog anchor: refuse if recovered target mag is fainter than catalog by more than this.
    neighbor_sub_max_target_undermag: float = 0.2
    #: Minimum recovered aperture SNR (flux / RMS/sqrt(area)) after subtraction.
    neighbor_sub_min_recovered_snr: float = 5.0
    # Preemptive refuse: very bright neighbour within ~1 FWHM (fine-scale edge guard).
    neighbor_sub_regime_dmag_min: float = 2.5
    neighbor_sub_regime_sep_max: float = 1.1
    #: Spatially-varying ePSF (GriddedPSFModel). Default OFF -> single global ePSF.
    #: Master gate for spatial ePSF: spatial active iff (psf_spatial_enabled AND psf_spatial_order > 0).
    #: When enabled, ePSFs are built per detector-region cell and interpolated by (x,y),
    #: which matters on wide fields where the PSF varies (coma / field curvature at edges).
    psf_spatial_enabled: bool = False
    #: Grid layout "NxM" (columns x rows) of detector regions for the spatial ePSF.
    psf_spatial_grid: str = "3x3"
    #: Minimum isolated stars per cell; cells below this fall back to the global ePSF (flagged).
    psf_spatial_min_stars_per_cell: int = 25
    #: When a per-star PSF fit is graded ``bad``, do not emit its PSF flux as usable -
    #: fall back to aperture for that star (sets ``psf_quality_fallback``). Default ON: a bad
    #: PSF fit must never silently become the reported value (the RMS-20.4 lesson).
    psf_quality_fallback_enabled: bool = True
    #: Per-star/per-frame adaptive flux selector (aperture vs PSF). Default OFF -> production
    #: stays pure-aperture. When ON, defaults to aperture and switches to PSF only with
    #: positive evidence AND good PSF quality (see _select_flux_method_row).
    psf_adaptive_enabled: bool = False
    #: A blend is "resolvable" (-> prefer PSF) only if the nearest neighbour is at least this
    #: many FWHM away (fit well-conditioned). At 9.77 arcsec/px this rarely fires (blends merge).
    psf_adaptive_resolve_fwhm: float = 2.0
    #: Faint-star threshold: below this SNR a good PSF (local background) can beat a
    #: contaminated aperture annulus -> prefer PSF.
    psf_adaptive_snr_lo: float = 15.0
    # WAVE-B STEP 6: moffat_chi2_limit hardcoded (module constant _MOFFAT_CHI2_LIMIT in pipeline.py).
    #: Per calibrated light: subtract source-masked sigma-clipped polynomial sky surface in preprocess (0=off, default 2).
    preprocess_sky_surface_order: int = 2
    #: When True, bypass the VY_SKYSF idempotency guard and re-subtract sky surface in-place.
    preprocess_sky_surface_force_reapply: bool = False
    #: OSC: post-extraction NxN average binning on each channel plane (1-4; default 2). Total scale vs raw = 2 x N.
    osc_channel_binning: int = 2
    #: In-place QC diagnostic threshold for estimated FWHM [pix] (legacy; selection uses DB prefilter).
    qc_fwhm_limit: float = 8.0
    #: In-place QC diagnostic threshold for elongation a/b.
    qc_elong_limit: float = 1.8
    #: Minimum clean stars required to build the ePSF model.
    epsf_min_stars: int = 30
    #: Per-frame photometry routing: ``aperture``, ``epsf``, or ``both``.
    photometry_mode: str = "both"
    # NOTE: These are in units of **Gaussian FWHM** (not moment-FWHM).
    # Aperture/annulus radii are computed as factor x fwhm_gaussian_px.
    #: Legacy single aperture factor - used where multi-aperture (B+C) is not active.
    aperture_fwhm_factor: float = 1.35
    #: APERTURE-01: ``f_fixed_night`` (r = f x median FWHM of the night) or ``f_per_frame``.
    aperture_policy_mode: str = "f_fixed_night"
    #: SNR aperture sizing sweep bounds (WAVE-B STEP 4 merge of aperture_fwhm_factor_small/_large):
    #: min ("small") and max ("large") radii as FWHM multiples.
    aperture_snr_sizing: dict[str, float] = field(
        default_factory=lambda: {"small": 1.5, "large": 4.0}
    )
    #: TODO-44: Role-aware scale on SNR-optimal radius (SIPS-style); 1.0 = no change.
    aperture_variable_factor: float = 1.0
    aperture_comp_factor: float = 1.1
    annulus_inner_fwhm: float = 2.7
    annulus_outer_fwhm: float = 5.2
    #: Top ``p`` %% brightest by ``peak_max_adu`` checked for FWHM non-linearity vs field median.
    nonlinearity_peak_percentile: float = 20.0
    nonlinearity_fwhm_ratio: float = 1.25
    #: Master-dark column BPM: MAD multiplier for ``*_dark_bpm.json`` (see ``importer``).
    bpm_dark_mad_sigma: float = 5.0

    # WAVE-B STEP 6: masterstar_solver_use_draft_median_if_hint_sep_deg hardcoded
    # (module constant in pipeline.py).
    #: Saturation safety fraction applied to equipment_saturate_adu before classifying MASTERSTAR zones.
    saturate_limit_fraction: float = 0.80
    # WAVE-B STEP 6: masterstar_optimizer_mirror_extra_log hardcoded (module constant in pipeline.py).
    #: Enable verbose debug logs for plate solving / blind solver / hint plumbing.
    debug_platesolver: bool = False
    #: VYVAR plate-solve na MASTERSTAR: max. SIP stupen (2-5). Solver skusa **nadol** po ``masterstar_platesolve_sip_min_order`` (napr. 5->4->3).
    masterstar_platesolve_sip_max_order: int = 4
    #: Najnizsi SIP stupen pri pade vyssich (typicky 3; nie menej ako 2).
    masterstar_platesolve_sip_min_order: int = 3
    #: DAOStarFinder pass1 threshold = this x star-masked sky sigma (NOT rms_conv; DAO-GAIA iter4).
    masterstar_dao_threshold_sigma: float = 4.5
    #: Deprecated alias kept for zone-classifier T1 only; detection uses masterstar_dao_threshold_sigma.
    dao_detection_n_equiv: float = 4.5
    #: E.2: max DAO-vs-WCS centroid shift (x FWHM) before WCS pixel fallback on matched stars.
    dao_centroid_max_shift_fwhm: float = 1.0
    #: D5-2 / C-1 admission gate: reject comps with peak above this fraction of full well (limit col is 85pct).
    admission_sat_peak_frac: float = 0.70
    #: Pred matchom s Gaia: ponechat detekcie s peakom aspon ``median + kxsigma`` (nizsie = viac slabych hviezd).
    masterstar_prematch_peak_sigma_floor: float = 1.8
    #: MASTERSTAR katalog: DAO FWHM z najlepsieho zdrojoveho snimku (``best_frame_fwhm_px``), nie median ``VY_FWHM`` v hlavicke.
    masterstar_use_best_frame_fwhm: bool = True
    #: MASTERSTAR DAO pass 2: local annulus sigma at Gaia seed positions (born-owned; INV-DET-FALSEFILL-01).
    masterstar_dao_pass2_sigma: float = 4.0
    #: MASTERSTAR DAO pass 2: max centroid offset [px] for targeted cutout acceptance (INV-DET-FALSEFILL-01 audit).
    masterstar_dao_pass2_center_tol_px: float = 2.0
    #: Pass1 spatial dedup radius [px] before pass2 (iter4 I6).
    masterstar_dao_pass1_dedup_px: float = 0.75
    #: Gaia-complete census + pass2 seed window: maximum G magnitude (default 15.0).
    masterstar_gaia_census_target_depth_g: float = 15.0
    #: FORCED_SEED admission (MASTERSTAR-GAIA-01): max COM centroid offset from propagated Gaia [px].
    masterstar_forced_seed_centroid_max_px: float = 2.0
    #: FORCED_SEED admission: minimum aperture SNR on local annulus background.
    masterstar_forced_seed_snr_min: float = 4.0
    #: FORCED_SEED / leftover promotion: greedy match radius for leftover detections to Gaia [px].
    masterstar_lock_leftover_radius_px: float = 3.0
    #: Lock-existing assignment: max shift from locked catalog_id xy to detection [px].
    masterstar_lock_pair_tol_px: float = 3.0
    #: Gaia-complete census: edge margin [px] for EDGE state (matches target_depth edge band).
    masterstar_gaia_census_edge_margin_px: float = 10.0
    #: Comp/CT pool: admit FORCED_SEED stars (default off; separate future decision).
    masterstar_forced_seed_comp_pool_enabled: bool = False
    #: DAO-GAIA-ERA-01 (D-A): match radius = k x astrometric residual p95 [px] (validated ~1.7 -> 3 px @ p95~1.78).
    masterstar_dao_match_radius_k: float = 1.7
    #: DAO-GAIA-ERA-01: centroid QA floor/cap when deriving pass2/seed tolerances from residual p95 [px].
    masterstar_dao_centroid_qa_floor_px: float = 1.0
    masterstar_dao_centroid_qa_cap_px: float = 3.0
    #: Empty-sky audit (INV-DET/SEED-FALSEFILL-01): max false-accept rate at MS build.
    masterstar_dao_empty_sky_false_accept_max: float = 0.01
    masterstar_dao_empty_sky_target_n: int = 2200
    # WAVE-B STEP 6: masterstar_platesolve_prewrite_rms_max_px / _prewrite_relaxed_rms_max_px /
    # _nn_refine_max_rms_px and masterstar_sip_force_rms_guard_ratio hardcoded as solver internals
    # (module constants in pipeline.py / vyvar_platesolver.py).
    #: MASTERSTAR verified-solve: min. catalog recovery (Gaia-in-frame with DAO match at 2.5 px).
    masterstar_catalog_recovery_min: float = 0.65
    #: MASTERSTAR verified-solve: absolute floor on tight-matched catalog stars (not fraction-only pass).
    masterstar_min_matched_floor: int = 40
    #: MASTERSTAR verified-solve: centre RMS cap [px] when distortion is not globally benign.
    masterstar_centre_rms_max_px: float = 1.20
    #: Centre RMS gate (arcsec). None -> fall back to ``masterstar_centre_rms_max_px``.
    masterstar_centre_rms_max_arcsec: float | None = None
    #: MASTERSTAR distortion-limited benign: max edge/centre residual ratio (was 2.50; Brno r ~3.0).
    masterstar_distortion_benign_ratio_max: float = 3.20

    #: MASTERSTAR accept gate: ``odds`` (Bayesian false-alarm) or legacy ``fraction``.
    masterstar_accept_mode: str = "odds"
    # WAVE-B STEP 6: masterstar_odds_match_floor / _odds_k / _odds_min_quadrants /
    # _false_alarm_p_max hardcoded as odds-verification internals (module constants in
    # vyvar_platesolver.py).
    #: Quality flag: n_cat_in_frame at or above this -> ``crowded``.
    masterstar_quality_crowded_n_cat_min: int = 800
    #: Adaptive DAO detection cap for dense fields (scales with catalog richness).
    masterstar_detection_cap_adaptive: bool = True
    masterstar_detection_cap_min: int = 250
    masterstar_detection_cap_max: int = 800
    masterstar_detection_cap_k: float = 0.08

    #: Pass 2 sibling-WCS recovery for filters that failed independent plate-solve (same draft field).
    masterstar_sibling_recovery_enabled: bool = True
    #: Sibling odds gate: minimum tight-matched catalog stars at 2.5 px.
    masterstar_sibling_min_matched: int = 40
    #: Sibling odds gate: maximum RMS of tight matches [px].
    masterstar_sibling_rms_max_px: float = 2.0
    #: Sibling-recovery RMS gate (arcsec). None -> fall back to ``masterstar_sibling_rms_max_px``.
    masterstar_sibling_rms_max_arcsec: float | None = None
    #: Sibling odds gate: minimum quadrants with at least one tight match.
    masterstar_sibling_min_quadrants: int = 3
    #: Median-stack frame count for sibling stacking rescue when single-frame odds fail.
    masterstar_sibling_stack_n: int = 10

    # WAVE-B STEP 6: platesolve_anisotropy_threshold hardcoded (module constant in pipeline.py).

    #: Paralelizmus (QC, preprocess, combined, per-frame CSV, alignment, calibrate MP): jedna hodnota
    #: pocitana v ``__post_init__``; nie v ``config.json``. Runtime override: ``VYVAR_PARALLEL_WORKERS`` alebo legacy env v pipeline.
    qc_preprocess_workers: int = 1
    #: Reserve this much RAM (GB) when capping paralelneho exportu katalogov cez ``psutil`` (nad ramec jednotneho ``_pw``).
    per_frame_mp_reserve_ram_gb: float = 1.5

    #: Frame alignment (DAO detection ladder): max brightest sources detected per frame (astroalign picks a subset).
    alignment_max_stars: int = 160
    #: Frame alignment (``astroalign`` triangle matching): brightest N stars used as control points (independent of detection count).
    alignment_max_control_points: int = 80
    #: DAOStarFinder threshold multiplier vs sigma-clipped background RMS (higher = fewer, more significant peaks).
    alignment_detection_sigma: float = 5.0
    #: Same recipe as QC HFR star detection (``_mean_hfr_bright_stars_dao`` first pass: ``threshold = qc_dao_detection_sigma x std``).
    #: Used for frame alignment DAO so it tracks QC-style sensitivity.
    qc_dao_detection_sigma: float = 5.0

    #: DAOStarFinder FWHM (pixels) tuned for SIPS-like centroid search (aperture ~13 -> ~4-5 px FWHM).
    sips_dao_fwhm_px: float = 2.5
    #: Initial DAO FWHM as multiple of measured FWHM. None -> fall back to ``sips_dao_fwhm_px``.
    sips_dao_fwhm_fwhm_factor: float | None = None
    #: DAOStarFinder threshold = this x background RMS (SIPS 'standard deviation count' ~ 2.5).
    #: Pre hlboky MASTERSTAR / siroke pole niekedy **0.25-1.0** (viac spiciek); pouziva sa aj pri VYVAR plate solve, ak volanie neprebije ``dao_threshold_sigma``.
    sips_dao_threshold_sigma: float = 3.5

    #: Faza 0+1 - vyber porovnavacich hviezd (``photometry_core.select_comparison_stars_per_target``).
    #: Pri **riedkom poli** zvacsi ``phase01_comparison_max_mag_diff`` / ``phase01_comparison_max_dist_deg``,
    #: pripadne zniz ``phase01_comparison_min_frames_frac`` alebo zvys ``phase01_comparison_max_comp_rms`` (slabsi filter stability).
    #: Pri **jasnych cieloch** (``mag`` < ``phase01_comparison_mag_bright_threshold``) sa pouzije aspon
    #: ``phase01_comparison_max_mag_diff_bright_floor`` ako minimalny |Deltamag| pas (``0`` = vypnute).
    # FOV-based comp distance: if plate_scale is known, search within a fraction of half-diagonal.
    phase01_comparison_max_dist_deg: float = 1.5
    #: Max comp distance as fraction of half-diagonal FOV (deg). None -> fall back to ``phase01_comparison_max_dist_deg``.
    phase01_comparison_max_dist_fov_frac: float | None = None
    phase01_comparison_fov_fraction: float = 0.75
    phase01_comparison_max_mag_diff: float = 1.5
    phase01_comparison_mag_bright_threshold: float = 12.75
    phase01_comparison_max_mag_diff_bright_floor: float = 1.5
    #: Absolutny strop pre adaptivne uvolnovanie |Deltamag| pri vybere porovnavaciek.
    #: Nikdy nejdeme vyssie (ochrana pred miesanim uplne inych jasnosti).
    phase01_comparison_max_mag_diff_absolute: float = 3.0
    phase01_comparison_n_comp_min: int = 3
    phase01_comparison_n_comp_max: int = 8
    phase01_comparison_max_comp_rms: float = 0.1
    #: COMP-RMS-DEF-01-B: LOO mag MAD ceiling is k x photon_sigma(snr_ap_pixscaled).
    #: C3-0 (516 R2, 70 comps): p90(r)=3.67 -> k=5 (rule: 3<=p90<=5 => 5).
    comp_rms_loo_photon_k: float = 5.0
    phase01_comparison_min_dist_arcsec: float = 60.0
    phase01_comparison_min_frames_frac: float = 0.2
    phase01_comparison_exclude_gaia_nss: bool = True
    phase01_comparison_exclude_gaia_extobj: bool = True
    #: ``True`` = tier + colour hard filter via BP-RP (Riello linear B-V fallback when needed).
    #: ``False`` = legacy |DeltaB-V| tiers via ``comp_tier*_bv_limit``.
    phase01_use_bprp_primary: bool = True
    #: Max |DeltaBP-RP| v efektivnom farebnom priestore (hard filter pri vybere comp).
    comp_max_delta_bprp: float = 0.79
    #: Comparison-star colour tiers (WAVE-B STEP 4 merge of comp_tier{1..4}_{bprp_limit,weight}).
    #: 4-row table; each row {"bprp": |dBP-RP| limit (Gaia), "w": ensemble/AC weight}.
    comp_color_tiers: list[dict[str, float]] = field(
        default_factory=lambda: [
            {"bprp": 0.15, "w": 1.00},
            {"bprp": 0.30, "w": 0.85},
            {"bprp": 0.55, "w": 0.50},
            {"bprp": 1.10, "w": 0.25},
        ]
    )
    #: Exponential contamination penalty in comp score: score *= exp(-k * contamination_idx).
    comp_contamination_penalty_k: float = 3.0
    #: Phase-0/1 magnitude tiers (WAVE-B STEP 4 merge of phase01_tier{1..4}_mag): |dmag| bounds.
    phase01_tiers: list[float] = field(
        default_factory=lambda: [0.50, 1.00, 1.50, 2.00]
    )
    phase01_plate_scale_arcsec_per_px: float = 1.3
    #: Plate scale (arcsec/px) for Phase 2A metadata, GS11, dilution; Set 1 default 1.3.
    plate_scale_arcsec_per_px: float = 1.3
    #: Faza 2A: minimum number of comps used in color-term fit before applying CT (``should_apply_color_term``).
    phase01_ct_min_comp: int = 7
    #: Faza 2A: apply BP-RP colour-term correction (``auto`` = on for B/V/Rc broadband, off for L/Clear).
    apply_color_term: str = "off"
    #: Per-rig Clear/unfiltered level colour coefficient (mag per BP-RP). None = disabled.
    #: PRE-IMPL-01 G-controlled measurement on this telephoto rig: -0.373 +/- 0.090.
    #: Not universal - remasure per equipment. Applied as export-only constant (no shape term).
    color_level_k_mag_per_bprp: float | None = None
    #: Uncertainty on ``color_level_k_mag_per_bprp`` (mag per BP-RP); propagates into exported err.
    color_level_k_stderr_mag_per_bprp: float | None = None
    #: Isolation radius (FWHM units) for SNR-table growth-curve stars (IMPL-01: 3 FWHM).
    snr_cog_isolation_fwhm: float = 3.0
    #: Second-order extinction: ``off`` | ``literature`` | ``fit_else_literature`` (fit path v2).
    k2_mode: str = "literature"
    #: Optional per-band k'' overrides (mag/airmass/BP-RP); empty dict uses ``k2_extinction`` converter.
    k2_defaults_bprp: dict[str, float] = field(default_factory=dict)
    #: Per-equipment systematic white floor (mag) for production LC err; key = equipment_id str.
    sigma_sys_mag: dict[str, float] = field(default_factory=dict)
    #: Native->container ADU scale (14-bit in 16-bit = 4). Used when g_pt unavailable.
    gain_container_scale: float = 4.0
    #: Max CI width factor (hi/lo) to accept photon-transfer g_pt as authority.
    photon_transfer_ci_max_width_factor: float = 3.0
    #: Hard plausibility ceiling for fitted k'' (v2 pre-gate).
    k2_ceiling: float = 0.1
    #: Enable per-night k'' fit (v2; off in v1 activation bundle).
    k2_fit_enabled: bool = False
    k2_fit_min_detectability: float = 3.0
    k2_fit_consistency_sigma: float = 2.0
    k2_fit_lit_factor: float = 4.0
    #: Faza 2A: BP-RP tolerance (mag) when testing target vs comp range before applying CT (0 = strict).
    phase01_ct_extrapolation_tol: float = 0.0
    #: Column name used for flux in Phase 1 comp selection (dao_flux = aperture DAO; psf_flux = ePSF).
    phase01_flux_col: str = "dao_flux"

    #: ALG-3: Temporal binning of comp ensemble before stability/PyTICS (Hartley & Wilson 2023 MNRAS).
    #: Default OFF: per-frame ensemble common-mode cancellation (validated V0612 differential path).
    temporal_binning_enabled: bool = False
    temporal_bin_window: int = 0  # 0 = auto-optimize among [3,5,7,9,11]

    #: ALG-2: Savitzky-Golay detrend after airmass (opt-in; Aigrain & Irwin 2004 MNRAS).
    savgol_detrend_enabled: bool = False
    savgol_window_frac: float = 0.5  # was 0.3 - more conservative
    savgol_polyorder: int = 2

    #: ALG-4: Democratic Detrender ensemble detrend (Caballero-Nieves et al. 2026 arXiv:2411.09753v2).
    democratic_detrend_enabled: bool = False
    democratic_sg_window_frac: float = 0.5

    #: ALG-5: PyTICS iterative comp intercalibration after stability check (Marconi et al. 2026 RASTI).
    pytics_enabled: bool = True
    pytics_n_iter: int = 5

    #: Faza 2A: exclude comparison stars with |linear slope| above this (mmag/hr) in stability check.
    comp_max_slope_mmag_hr: float = 5.0
    #: Minimum |slope|/stderr (sigma) to treat a post-common-mode residual slope as real (stability check).
    comp_slope_significance_k: float = 3.0
    #: Per-target sparse fallback: generous pool + iterative CM-residual clip when default starved.
    comp_sparse_fallback_enabled: bool = True
    #: Trigger fallback when default yields fewer than this many comps (0 -> use ``n_comp_min``).
    comp_sparse_fallback_min: int = 0
    #: Deprecated alias for ``comp_sparse_fallback_enabled`` (config/UI backward compat).
    comp_iterative_clip_enabled: bool = False

    # GS11 - Flux dilution correction
    gs11_dilution_enabled: bool = False
    gs11_dilution_aperture_arcsec: float = 0.0
    gs11_dilution_mag_limit_delta: float = 5.0
    gs11_comp_max_dilution: float = 0.90
    gs11_comp_suspect_dilution: float = 0.98
    gs11_target_min_dilution: float = 0.50

    #: Aperture correction (Method B): reserved for future pipeline; off by default.
    aperture_correction_enabled: bool = True
    aperture_correction_min_ref_stars: int = 3
    aperture_correction_max_contamination: float = 0.15

    aperture_correction_max_scatter_mag: float = 0.03

    #: Per-frame curve-of-growth aperture correction (puts every star on a common
    #: ref-radius enclosed-flux scale). Distinct from aperture_correction_* (Method B,
    #: flux_large/flux_small). Default OFF - validate before enabling in production.
    cog_aperture_correction_enabled: bool = False
    #: COG reference radius in FWHM units (where the curve of growth flattens).
    cog_ref_fwhm: float = 4.5
    #: Minimum number of COG stars required per frame; else cog_ok=False (no correction).
    cog_min_stars: int = 8
    #: COG-star isolation radius in FWHM units (no neighbour within this distance).
    cog_isolation_fwhm: float = 6.0
    #: COG-star minimum Howell SNR.
    cog_snr_min: float = 50.0
    #: COG-star saturation guard: reject if peak > sat_frac * saturate_limit_adu.
    cog_sat_frac: float = 0.85
    #: COG radius-ladder step (px).
    cog_ladder_step_px: float = 0.5
    #: COG ladder step as multiple of measured FWHM. None -> fall back to ``cog_ladder_step_px``.
    cog_ladder_step_fwhm: float | None = None
    #: Maximum allowed per-star ac_factor (safety clamp).
    cog_ac_factor_max: float = 5.0

    #: Per-frame target saturation gate (PER-FRAME-SAT-GATED). Default OFF keeps
    #: whole-star skip_photometry from master zone_flag. When ON, targets use the
    #: fraction of unsaturated frames vs per_frame_sat_min_clean_frac.
    per_frame_saturation_enabled: bool = False
    #: Minimum fraction of clean (unsaturated) frames required to measure a target
    #: when per_frame_saturation_enabled is True. Clamped to [0.1, 1.0].
    per_frame_sat_min_clean_frac: float = 0.5

    #: CCD gain (e-/ADU) - used in noise model / SNR estimates.
    gain: float = 1.0
    #: CCD read noise (e-) - used in noise model.
    read_noise: float = 10.0

    #: Number of random empty apertures per frame/radius for ``sigma_bkg_ap`` (clamped 16..256).
    err_empty_apertures_n: int = 64
    #: Minimum valid empty apertures before Howell fallback for that frame.
    err_empty_apertures_min: int = 16

    #: MASTERSTAR: stack N frames and pick best N for ePSF/catalog build.
    masterstar_best_of_n: int = 10

    # WAVE-B STEP 6: sky_adu_fallback hardcoded (module constant _SKY_ADU_FALLBACK in pipeline.py).

    #: Aperture correction: reject comp stars with scatter above this (mag).
    #: Phase 1 comp gate: max reduced chi2 for PSF fit acceptance in comp selection.
    phase01_comparison_max_psf_chi2: float = 50.0
    #: Phase 1 comp gate: reject comps with FWHM > this factor x field median FWHM.
    phase01_comparison_max_fwhm_factor: float = 1.5
    #: Phase 1 comp gate: minimum isolation radius (px) - reject comps with neighbour closer than this.
    phase01_comparison_isolation_radius_px: float = 25.0
    #: Isolation radius (arcsec). None -> fall back to ``phase01_comparison_isolation_radius_px``.
    phase01_comparison_isolation_radius_arcsec: float | None = None
    #: Sensor frame dimensions in pixels (used when FITS NAXIS1/2 unavailable).
    frame_width_px: int = 2082
    frame_height_px: int = 1397

    #: Jednotny vnutorny okraj cipu (px) pre **celu Fazu 0+1**: aktivne premenne, porovnavacie hviezdy aj suspected.
    #: Hviezdy s ``x,y`` blizsie ako tento pocet pixelov od okraja referencneho pola sa neberu (zmiernuje artefakty
    #: pri zarovnani / posune pola / okrajoch). ``0`` = vypnute (cely cip). Predvolene 50 px.
    phase01_chip_interior_margin_px: int = 50
    #: Chip interior margin (arcsec). None -> fall back to ``phase01_chip_interior_margin_px``.
    phase01_chip_interior_margin_arcsec: float | None = None

    # Variability Detection
    variability_min_frames: int = 30
    variability_min_frames_frac: float = 0.50
    variability_p85_filter: int = 85
    variability_slope_floor: float = 0.02
    variability_sigma_threshold: float = 2.3
    #: Upper envelope floor = comp P90 rms_pct per mag bin x this factor (TODO-26).
    variability_comp_floor_factor: float = 1.5
    variability_smoothness_max: float = 0.80
    variability_mag_limit: float = 14.5
    variability_min_rms_pct: float = 1.5
    variability_min_amplitude_mag: float = 0.01
    variability_clip_ratio_min: float = 0.80
    variability_vdi_z_threshold: float = 3.0
    variability_min_points_rms: int = 20

    #: ``True`` = stahovanie/analyza TESS FFI cez lightkurve (TessCut), UI + ``tess_runner`` + pipeline hook.
    #: ``False`` = vypnute - ziadne stahovanie; log ``[TESS] preskocene``. Zapnut: ``"tess_enabled": true`` v ``config.json``.
    tess_enabled: bool = False

    #: Hustota pola (hviezd/Mpx z DAO na MASTERSTAR): prahy a adaptivne upravy Fazy 0+1 / apertury (baseline = JSON).
    field_density_sparse_threshold: float = 300.0
    field_density_dense_threshold: float = 1000.0
    field_density_adaptive_enabled: bool = True
    #: [CROWDING-CLASSIFIER] Replace the detection/scale-locked stars/Mpx classifier with
    #: detection-independent ``crowding_index`` signals. Default OFF (stars/Mpx fallback).
    #: Decouples the two concerns the legacy single class conflated:
    #:   LOOSEN keys on comp AVAILABILITY (few usable catalog comps in FOV), not density;
    #:   TIGHTEN keys on real BLEND_FRAC (contamination @ measured depth), not stars/Mpx.
    crowding_classifier_enabled: bool = False
    #: blend_frac (neighbours within 1xFWHM @ measured depth) at/above which to TIGHTEN comps.
    crowding_blend_tighten_threshold: float = 0.04
    #: usable catalog comps in FOV (Gaia stars <= effective limit) below which to LOOSEN comps.
    crowding_comp_availability_loosen_count: float = 500.0
    #: SAMPLING GATE for the blend-TIGHTEN branch. Tighten ONLY when the PSF is resolved
    #: (FWHM_px >= this). On under-sampled fields (wide rig, FWHM~2.6 px) the comp-RMS
    #: 0.08-0.10 tail is the field floor (scintillation/undersampling), NOT resolvable
    #: contamination, so max_comp_rms tightening cuts good comps and thins the ensemble
    #: (verified on 360: -19 comps, +3.5 mmag LC scatter). FWHM>=3 px ~ "well sampled"
    #: (Nyquist is 2 px; PSF-fit/deblend guidance wants >=3 px) - only there does a high
    #: comp-RMS mean real contamination, so tighten pays off (e.g. the Newton cluster).
    crowding_tighten_min_fwhm_px: float = 3.0
    #: COMP-POOL-01 Stage 2 / COMP-ADMIT-03: derived path now only drops known variables;
    #: scatter/colour/distance enter continuous weights (``comp_weights``).
    comp_pool_derived_admission: bool = True
    #: Plan/spatial comparison pool size. ``0`` = uncapped (COMP-POOL-01); legacy default was 150.
    comparison_stars_pool_n: int = 0
    #: COMP-ADMIT-03 weight colour coefficient [mag per BP-RP]. None = derive from |k''|*DeltaX.
    comp_weight_c_col_mag_per_bprp: float | None = None
    #: COMP-ADMIT-03 weight distance coefficient [mag per degree]. None = measure or named zero.
    comp_weight_c_dist_mag_per_deg: float | None = None
    #: Airmass span of the series for c_col = |k2|*DeltaX when not overridden.
    comp_weight_airmass_span: float = 0.0
    #: Optics class for c_col_psf: refractive|mirror|unknown (COMP-WEIGHT-COEFF-01).
    comp_weight_optics_kind: str = "unknown"
    #: FORCED-PHOT-01: inject MASTERSTAR pool-eligible stars missing from DAO each frame.
    forced_photometry_enabled: bool = True
    #: Bounded peak refine search in units of FWHM (recorded; never unbounded).
    forced_photometry_centroid_bound_fwhm: float = 2.5
    #: Extra pixel margin for in-footprint geometry gate on forced rows.
    forced_photometry_margin_px: float = 0.0

    #: Post-calibration QC on each calibrated light (metrics + pass/fail vs limits).
    qc_after_calibrate_enabled: bool = True
    #: PERF-10: DAO QC (FWHM/sky/star_count) during calibration; skips RAM QC pass when True.
    dao_qc_in_calibrate: bool = True
    qc_max_hfr: float = 5.0
    #: QC HFR limit as multiple of measured FWHM. None -> fall back to ``qc_max_hfr`` (legacy px).
    qc_max_hfr_fwhm_ratio: float | None = None
    qc_min_stars: int = 10
    #: If set, fail when plain sky RMS exceeds this (same units as calibrated image).
    qc_max_background_rms: float | None = None

    #: FITS QA dashboard: odvodzovat predvoleny FWHM limit z MAD (median + kxsigma_MAD).
    auto_fwhm_enabled: bool = True
    auto_fwhm_k_factor: float = 1.5
    auto_fwhm_k_min: float = 1.0
    auto_fwhm_k_max: float = 4.0

    #: Fix B: Phase-2A reject-on-alignment-residual gate (default OFF -> byte-identical).
    #: Rejects frames whose per-frame alignment residual (median deviation of bright matched
    #: sources from their across-night median position, recorded in alignment_report.csv) exceeds
    #: ``frame_align_residual_max_frac * science-aperture-radius-px``. Rig-agnostic (relative to the
    #: aperture radius, not a fixed pixel value). Cause-correct alignment-quality signal; complements
    #: the B.2 aperture-integrity gate. Self-deactivating once alignment (Fix C) succeeds.
    frame_align_residual_gate_enabled: bool = False
    frame_align_residual_max_frac: float = 0.25   # reject if residual > frac * science aperture radius (clamped 0.05..1.0)
    frame_align_residual_min_keep_frames: int = 10  # safety floor: skip gate if it would drop below this

    # Paths derived from config.json (must stay after all init=True fields for dataclass(slots=True)).
    data_root: Path = field(init=False)
    archive_root: Path = field(init=False)
    calibration_library_root: Path = field(init=False)
    database_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.data_root = resolve_data_root(self.project_root)
        data = load_config_json(self.data_root)

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

        self.masterdark_validity_days = int(data.get("masterdark_validity_days", 90))
        self.masterflat_validity_days = int(data.get("masterflat_validity_days", 200))

        # ``plate_solve_fov_deg`` is no longer read from JSON - resolved from FITS + DB (see ``resolve_plate_solve_fov_deg_hint``).
        self.plate_solve_fov_deg = 1.0
        # Migration: GAIA_DB_PATH supersedes legacy catalog settings.
        _cln_raw = data.get("calibration_library_native_binning", self.calibration_library_native_binning)
        if _cln_raw is None:
            self.calibration_library_native_binning = None
        else:
            try:
                _cln = int(_cln_raw)
                self.calibration_library_native_binning = max(1, min(16, _cln))
            except (TypeError, ValueError):
                self.calibration_library_native_binning = 1
        try:
            _ctol = float(
                data.get(
                    "calibration_master_ccd_temp_tolerance_c",
                    self.calibration_master_ccd_temp_tolerance_c,
                )
            )
            if math.isfinite(_ctol) and 0.01 <= _ctol <= 20.0:
                self.calibration_master_ccd_temp_tolerance_c = float(_ctol)
        except (TypeError, ValueError):
            pass

        self.gaia_db_path = resolve_config_path(
            str(data.get("gaia_db_path", data.get("GAIA_DB_PATH", "")) or "").strip(),
            self.data_root,
        )

        self.hrd_online_enrich_enabled = bool(
            data.get("hrd_online_enrich_enabled", self.hrd_online_enrich_enabled)
        )
        self.hrd_simbad_enrich_enabled = bool(
            data.get("hrd_simbad_enrich_enabled", self.hrd_simbad_enrich_enabled)
        )
        try:
            self.hrd_enrich_max_candidates = int(
                data.get("hrd_enrich_max_candidates", self.hrd_enrich_max_candidates)
            )
        except (TypeError, ValueError):
            self.hrd_enrich_max_candidates = 20
        self.hrd_enrich_max_candidates = max(1, min(100, int(self.hrd_enrich_max_candidates)))
        try:
            self.hrd_enrich_tap_timeout_s = float(
                data.get("hrd_enrich_tap_timeout_s", self.hrd_enrich_tap_timeout_s)
            )
        except (TypeError, ValueError):
            self.hrd_enrich_tap_timeout_s = 20.0
        self.hrd_enrich_tap_timeout_s = max(5.0, min(120.0, float(self.hrd_enrich_tap_timeout_s)))
        try:
            self.hrd_parallax_min_mas = float(
                data.get("hrd_parallax_min_mas", self.hrd_parallax_min_mas)
            )
        except (TypeError, ValueError):
            self.hrd_parallax_min_mas = 0.15
        self.hrd_parallax_min_mas = max(0.0, min(10.0, float(self.hrd_parallax_min_mas)))
        try:
            self.hrd_parallax_snr_min = float(
                data.get("hrd_parallax_snr_min", self.hrd_parallax_snr_min)
            )
        except (TypeError, ValueError):
            self.hrd_parallax_snr_min = 5.0
        self.hrd_parallax_snr_min = max(1.0, min(20.0, float(self.hrd_parallax_snr_min)))
        try:
            self.hrd_max_per_category = int(
                data.get("hrd_max_per_category", self.hrd_max_per_category)
            )
        except (TypeError, ValueError):
            self.hrd_max_per_category = 3
        self.hrd_max_per_category = max(1, min(20, int(self.hrd_max_per_category)))
        try:
            self.hrd_min_per_net = int(data.get("hrd_min_per_net", self.hrd_min_per_net))
        except (TypeError, ValueError):
            self.hrd_min_per_net = 4
        self.hrd_min_per_net = max(0, min(20, int(self.hrd_min_per_net)))
        self.hrd_nss_category_enabled = bool(
            data.get("hrd_nss_category_enabled", self.hrd_nss_category_enabled)
        )
        try:
            self.hrd_dsc_confirm_prob = float(
                data.get("hrd_dsc_confirm_prob", self.hrd_dsc_confirm_prob)
            )
        except (TypeError, ValueError):
            self.hrd_dsc_confirm_prob = 0.90
        self.hrd_dsc_confirm_prob = max(0.5, min(1.0, float(self.hrd_dsc_confirm_prob)))
        self.hrd_color_field_enabled = bool(
            data.get("hrd_color_field_enabled", self.hrd_color_field_enabled)
        )
        try:
            self.hrd_color_saturation = float(
                data.get("hrd_color_saturation", self.hrd_color_saturation)
            )
        except (TypeError, ValueError):
            self.hrd_color_saturation = 0.85
        self.hrd_color_saturation = max(0.0, min(1.0, float(self.hrd_color_saturation)))
        _hl = str(data.get("hrd_color_highlight_mode", self.hrd_color_highlight_mode) or "soft").strip().lower()
        self.hrd_color_highlight_mode = "scale" if _hl == "scale" else "soft"
        try:
            self.hrd_color_chroma_snr = float(
                data.get("hrd_color_chroma_snr", self.hrd_color_chroma_snr)
            )
        except (TypeError, ValueError):
            self.hrd_color_chroma_snr = 3.0
        self.hrd_color_chroma_snr = max(0.0, min(20.0, float(self.hrd_color_chroma_snr)))
        _wp = str(data.get("hrd_color_white_point", self.hrd_color_white_point) or "field_median").strip().lower()
        self.hrd_color_white_point = "d65" if _wp == "d65" else "field_median"
        try:
            self.hrd_color_chroma_boost = float(
                data.get("hrd_color_chroma_boost", self.hrd_color_chroma_boost)
            )
        except (TypeError, ValueError):
            self.hrd_color_chroma_boost = 2.2
        self.hrd_color_chroma_boost = max(1.0, min(3.0, float(self.hrd_color_chroma_boost)))
        try:
            self.hrd_color_bg_box_px = int(
                data.get("hrd_color_bg_box_px", self.hrd_color_bg_box_px)
            )
        except (TypeError, ValueError):
            self.hrd_color_bg_box_px = 96
        self.hrd_color_bg_box_px = max(32, min(512, int(self.hrd_color_bg_box_px)))

        gaia_dir = self.data_root / "GAIA_DR3"
        _fine_default = str(gaia_dir / "gaia_triangles_fine.pkl")
        _wide_default = str(gaia_dir / "gaia_triangles_wide.pkl")
        legacy_blind = str(
            data.get("blind_index_path", data.get("BLIND_INDEX_PATH", "")) or ""
        ).strip()
        fine = str(
            data.get("blind_index_fine_path", data.get("BLIND_INDEX_FINE_PATH", "")) or ""
        ).strip()
        wide = str(
            data.get("blind_index_wide_path", data.get("BLIND_INDEX_WIDE_PATH", "")) or ""
        ).strip()
        if not fine and not wide and legacy_blind:
            base = Path(resolve_config_path(legacy_blind, self.data_root)).parent
            fine = str((base / "gaia_triangles_fine.pkl").resolve())
            wide = str((base / "gaia_triangles_wide.pkl").resolve())
        if not fine:
            fine = legacy_blind or _fine_default
        if not wide:
            wide = _wide_default
        self.blind_index_fine_path = resolve_config_path(fine, self.data_root) if fine else ""
        self.blind_index_wide_path = resolve_config_path(wide, self.data_root) if wide else ""
        self.blind_index_path = fine
        _mode = str(data.get("blind_index_select_mode", self.blind_index_select_mode) or "auto")
        _mode = _mode.strip().lower()
        self.blind_index_select_mode = (
            _mode if _mode in ("auto", "series_all", "single") else "auto"
        )
        self.blind_verify_enabled = bool(data.get("blind_verify_enabled", self.blind_verify_enabled))
        try:
            self.blind_verify_top_n = int(data.get("blind_verify_top_n", self.blind_verify_top_n))
        except (TypeError, ValueError):
            self.blind_verify_top_n = 15
        self.blind_verify_top_n = max(1, min(50, int(self.blind_verify_top_n)))
        try:
            self.blind_verify_match_tol_px = float(
                data.get("blind_verify_match_tol_px", self.blind_verify_match_tol_px)
            )
        except (TypeError, ValueError):
            self.blind_verify_match_tol_px = 2.5
        self.blind_verify_match_tol_px = max(0.5, min(20.0, float(self.blind_verify_match_tol_px)))
        try:
            self.blind_verify_min_matches = int(
                data.get("blind_verify_min_matches", self.blind_verify_min_matches)
            )
        except (TypeError, ValueError):
            self.blind_verify_min_matches = 12
        self.blind_verify_min_matches = max(3, min(200, int(self.blind_verify_min_matches)))
        try:
            self.blind_verify_min_fraction = float(
                data.get("blind_verify_min_fraction", self.blind_verify_min_fraction)
            )
        except (TypeError, ValueError):
            self.blind_verify_min_fraction = 0.30
        self.blind_verify_min_fraction = max(0.05, min(0.95, float(self.blind_verify_min_fraction)))
        self.blind_verify_inmemory_catalog = bool(
            data.get("blind_verify_inmemory_catalog", self.blind_verify_inmemory_catalog)
        )
        try:
            self.verify_mag_limit = float(data.get("verify_mag_limit", self.verify_mag_limit))
        except (TypeError, ValueError):
            self.verify_mag_limit = 14.0
        self.verify_mag_limit = max(8.0, min(18.0, float(self.verify_mag_limit)))
        try:
            self.blind_verify_early_accept = int(
                data.get("blind_verify_early_accept", self.blind_verify_early_accept)
            )
        except (TypeError, ValueError):
            self.blind_verify_early_accept = 30
        self.blind_verify_early_accept = max(8, min(200, int(self.blind_verify_early_accept)))
        try:
            self.blind_verify_early_floor = int(
                data.get("blind_verify_early_floor", self.blind_verify_early_floor)
            )
        except (TypeError, ValueError):
            self.blind_verify_early_floor = 0
        self.blind_verify_early_floor = max(0, min(200, int(self.blind_verify_early_floor)))
        try:
            self.blind_verify_early_fraction = float(
                data.get("blind_verify_early_fraction", self.blind_verify_early_fraction)
            )
        except (TypeError, ValueError):
            self.blind_verify_early_fraction = 0.20
        self.blind_verify_early_fraction = max(0.0, min(0.95, float(self.blind_verify_early_fraction)))
        # WAVE-B STEP 6: blind_prefilter_min hardcoded (solver internal).
        try:
            self.blind_img_star_budget = int(
                data.get("blind_img_star_budget", self.blind_img_star_budget)
            )
        except (TypeError, ValueError):
            self.blind_img_star_budget = 80
        self.blind_img_star_budget = max(10, min(500, int(self.blind_img_star_budget)))
        _bism = str(data.get("blind_img_select_mode", self.blind_img_select_mode)).strip().lower()
        self.blind_img_select_mode = (
            "central" if _bism in ("central", "legacy", "rig_prior") else "per_cell"
        )
        self.blind_use_rig_prior = bool(
            data.get("blind_use_rig_prior", self.blind_use_rig_prior)
        )
        # WAVE-B STEP 6: blind_scale_tol_frac and blind_cluster_{min_votes,eps_deg,min_samples,
        # vote_span,coherence_cap} hardcoded as solver internals (module constants in
        # vyvar_blind_solver.py).

        self.vsx_local_db_path = resolve_config_path(
            str(data.get("vsx_local_db_path", data.get("VSX_LOCAL_DB_PATH", "")) or "").strip(),
            self.data_root,
        )
        _voos = data.get("vsx_out_of_scope_types", self.vsx_out_of_scope_types)
        if _voos is None:
            self.vsx_out_of_scope_types = []
        elif isinstance(_voos, str):
            self.vsx_out_of_scope_types = [p.strip() for p in _voos.split(",") if p.strip()]
        elif isinstance(_voos, (list, tuple)):
            self.vsx_out_of_scope_types = [str(p).strip() for p in _voos if str(p).strip()]
        else:
            self.vsx_out_of_scope_types = []
        self.exoplanet_local_db_path = resolve_config_path(
            str(
                data.get(
                    "exoplanet_local_db_path",
                    data.get("EXOPLANET_LOCAL_DB_PATH", self.exoplanet_local_db_path),
                )
                or self.exoplanet_local_db_path
                or ""
            ).strip(),
            self.data_root,
        )
        try:
            _exo_sep = float(
                data.get("exoplanet_match_max_sep_arcsec", self.exoplanet_match_max_sep_arcsec)
            )
            if math.isfinite(_exo_sep):
                self.exoplanet_match_max_sep_arcsec = max(0.5, min(30.0, _exo_sep))
        except (TypeError, ValueError):
            self.exoplanet_match_max_sep_arcsec = 3.0
        try:
            self.catalog_query_max_rows = max(
                1000, min(500_000, int(data.get("catalog_query_max_rows", self.catalog_query_max_rows)))
            )
        except (TypeError, ValueError):
            self.catalog_query_max_rows = 15_000

        try:
            self.per_frame_mp_reserve_ram_gb = float(
                data.get("per_frame_mp_reserve_ram_gb", self.per_frame_mp_reserve_ram_gb)
            )
            if not math.isfinite(self.per_frame_mp_reserve_ram_gb) or self.per_frame_mp_reserve_ram_gb < 0:
                self.per_frame_mp_reserve_ram_gb = 1.5
        except (TypeError, ValueError):
            self.per_frame_mp_reserve_ram_gb = 1.5

        _pw = int(
            recommended_vyvar_parallel_workers(reserve_ram_gb=float(self.per_frame_mp_reserve_ram_gb))
        )
        self.qc_preprocess_workers = _pw

        try:
            self.alignment_max_stars = max(
                10, min(5000, int(data.get("alignment_max_stars", self.alignment_max_stars)))
            )
        except (TypeError, ValueError):
            self.alignment_max_stars = 200
        try:
            self.alignment_max_control_points = max(
                12,
                min(
                    500,
                    int(data.get("alignment_max_control_points", self.alignment_max_control_points)),
                ),
            )
        except (TypeError, ValueError):
            self.alignment_max_control_points = 80
        try:
            self.alignment_detection_sigma = float(
                data.get("alignment_detection_sigma", self.alignment_detection_sigma)
            )
            if not math.isfinite(self.alignment_detection_sigma) or self.alignment_detection_sigma <= 0:
                self.alignment_detection_sigma = 5.0
        except (TypeError, ValueError):
            self.alignment_detection_sigma = 5.0
        try:
            self.qc_dao_detection_sigma = float(data.get("qc_dao_detection_sigma", self.qc_dao_detection_sigma))
            if not math.isfinite(self.qc_dao_detection_sigma) or self.qc_dao_detection_sigma <= 0:
                self.qc_dao_detection_sigma = 5.0
        except (TypeError, ValueError):
            self.qc_dao_detection_sigma = 5.0
        try:
            self.sips_dao_fwhm_px = float(data.get("sips_dao_fwhm_px", self.sips_dao_fwhm_px))
            if not math.isfinite(self.sips_dao_fwhm_px) or self.sips_dao_fwhm_px <= 0:
                self.sips_dao_fwhm_px = 2.5
        except (TypeError, ValueError):
            self.sips_dao_fwhm_px = 2.5
        self.sips_dao_fwhm_px = max(1.0, min(8.0, float(self.sips_dao_fwhm_px)))
        try:
            self.sips_dao_threshold_sigma = float(
                data.get("sips_dao_threshold_sigma", self.sips_dao_threshold_sigma)
            )
            if not math.isfinite(self.sips_dao_threshold_sigma) or self.sips_dao_threshold_sigma <= 0:
                self.sips_dao_threshold_sigma = 3.5
        except (TypeError, ValueError):
            self.sips_dao_threshold_sigma = 3.5

        self.qc_after_calibrate_enabled = bool(
            data.get("qc_after_calibrate_enabled", self.qc_after_calibrate_enabled)
        )
        self.dao_qc_in_calibrate = bool(
            data.get("dao_qc_in_calibrate", self.dao_qc_in_calibrate)
        )
        try:
            self.qc_max_hfr = float(data.get("qc_max_hfr", self.qc_max_hfr))
        except (TypeError, ValueError):
            self.qc_max_hfr = 5.0
        try:
            self.qc_min_stars = max(0, int(data.get("qc_min_stars", self.qc_min_stars)))
        except (TypeError, ValueError):
            self.qc_min_stars = 10
        _qmr = data.get("qc_max_background_rms", self.qc_max_background_rms)
        if _qmr is None or _qmr == "":
            self.qc_max_background_rms = None
        else:
            try:
                v = float(_qmr)
                self.qc_max_background_rms = v if v > 0 and math.isfinite(v) else None
            except (TypeError, ValueError):
                self.qc_max_background_rms = None

        self.auto_fwhm_enabled = bool(data.get("auto_fwhm_enabled", self.auto_fwhm_enabled))
        try:
            self.auto_fwhm_k_factor = float(data.get("auto_fwhm_k_factor", self.auto_fwhm_k_factor))
        except (TypeError, ValueError):
            self.auto_fwhm_k_factor = 1.5
        try:
            self.auto_fwhm_k_min = float(data.get("auto_fwhm_k_min", self.auto_fwhm_k_min))
        except (TypeError, ValueError):
            self.auto_fwhm_k_min = 1.0
        try:
            self.auto_fwhm_k_max = float(data.get("auto_fwhm_k_max", self.auto_fwhm_k_max))
        except (TypeError, ValueError):
            self.auto_fwhm_k_max = 4.0
        if self.auto_fwhm_k_min > self.auto_fwhm_k_max:
            self.auto_fwhm_k_min, self.auto_fwhm_k_max = 1.0, 4.0
        self.auto_fwhm_k_factor = max(
            float(self.auto_fwhm_k_min), min(float(self.auto_fwhm_k_max), float(self.auto_fwhm_k_factor))
        )

        self.frame_align_residual_gate_enabled = bool(
            data.get("frame_align_residual_gate_enabled", self.frame_align_residual_gate_enabled)
        )
        try:
            self.frame_align_residual_max_frac = max(
                0.05, min(1.0, float(data.get("frame_align_residual_max_frac", self.frame_align_residual_max_frac)))
            )
        except (TypeError, ValueError):
            self.frame_align_residual_max_frac = 0.25
        try:
            self.frame_align_residual_min_keep_frames = max(
                3,
                min(
                    100000,
                    int(data.get("frame_align_residual_min_keep_frames", self.frame_align_residual_min_keep_frames)),
                ),
            )
        except (TypeError, ValueError):
            self.frame_align_residual_min_keep_frames = 10

        self.aperture_photometry_enabled = bool(data.get("aperture_photometry_enabled", self.aperture_photometry_enabled))
        self.save_lightcurve_png = bool(data.get("save_lightcurve_png", self.save_lightcurve_png))
        self.phase2a_airmass_before_outlier = bool(
            data.get("phase2a_airmass_before_outlier", self.phase2a_airmass_before_outlier)
        )
        self.sysrem_enabled = bool(data.get("sysrem_enabled", self.sysrem_enabled))
        try:
            self.sysrem_n_iter = max(1, int(data.get("sysrem_n_iter", self.sysrem_n_iter)))
        except (TypeError, ValueError):
            self.sysrem_n_iter = 3
        self.comp_qa_enabled = bool(data.get("comp_qa_enabled", self.comp_qa_enabled))
        self.trust_flag_enabled = bool(data.get("trust_flag_enabled", self.trust_flag_enabled))
        # Export reports - prefer ``observer_name`` / ``observer_code``; fall back to legacy JSON key for the code.
        _obn = data.get("observer_name")
        if _obn is None or str(_obn).strip() == "":
            _obn = self.observer_name
        self.observer_name = str(_obn or "").strip() or "Unknown Observer"
        _obc = data.get("observer_code")
        if _obc is None:
            _obc = data.get("aavso_observer_code", self.observer_code)
        self.observer_code = str(_obc or "").strip()
        self.aavso_observer_code = str(self.observer_code)
        _ffm = data.get("aavso_filter_map", {})
        if isinstance(_ffm, dict):
            self.aavso_filter_map = {
                str(k).strip().upper(): str(v).strip()
                for k, v in _ffm.items()
                if str(k).strip() and str(v).strip()
            }
        else:
            self.aavso_filter_map = {}
        try:
            self.observer_location_id = int(
                data.get("observer_location_id", self.observer_location_id)
            )
        except (TypeError, ValueError):
            self.observer_location_id = 0
        self.observer_location_id = max(0, self.observer_location_id)
        # WAVE-B STEP 5 (DELETE-DB-DUP): observer_lat/lon/alt_m/name are DB-authoritative
        # (draft LOCATION via observer_location_id). They are no longer loaded from or saved to
        # config.json; the fields remain as run-time hydrated mirrors (block below) with the
        # dataclass site defaults as fallback. Science reads the draft LOCATION row, not cfg.
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
                    "using dataclass site-default observer coordinates: %s",
                    self.observer_location_id,
                    exc,
                )
        # WAVE-B STEP 5 (DELETE-DB-DUP): export_arcsec_per_px derivable from WCS/optics; no longer
        # loaded from or saved to config.json. Field keeps its dataclass default as a label fallback.
        self.psf_photometry_enabled = bool(data.get("psf_photometry_enabled", self.psf_photometry_enabled))
        self.epsf_auto_run = bool(data.get("epsf_auto_run", self.epsf_auto_run))
        try:
            _pso = int(data.get("psf_spatial_order", self.psf_spatial_order))
            self.psf_spatial_order = max(0, min(2, _pso))
        except (TypeError, ValueError):
            self.psf_spatial_order = 0
        try:
            _pct = float(data.get("psf_chi2_threshold", self.psf_chi2_threshold))
            self.psf_chi2_threshold = _pct if math.isfinite(_pct) and _pct > 0 else 50.0
        except (TypeError, ValueError):
            self.psf_chi2_threshold = 50.0
        _acp = str(data.get("psf_ac_policy", self.psf_ac_policy) or "p4_none").strip().lower()
        self.psf_ac_policy = _acp if _acp in ("p4_none", "chi2_lt5_legacy") else "p4_none"
        _zpm = str(data.get("psf_zp_membership", self.psf_zp_membership) or "fit_ok_for_zp").strip()
        self.psf_zp_membership = (
            _zpm if _zpm in ("fit_ok_strict", "fit_ok_for_zp") else "fit_ok_for_zp"
        )
        _zpr = data.get("psf_zp_for_zp_validated_rigs", self.psf_zp_for_zp_validated_rigs)
        if isinstance(_zpr, str):
            self.psf_zp_for_zp_validated_rigs = [x.strip() for x in _zpr.split(",") if x.strip()]
        elif isinstance(_zpr, (list, tuple)):
            self.psf_zp_for_zp_validated_rigs = [str(x).strip() for x in _zpr if str(x).strip()]
        else:
            self.psf_zp_for_zp_validated_rigs = ["1:1"]
        if not self.psf_zp_for_zp_validated_rigs:
            self.psf_zp_for_zp_validated_rigs = ["1:1"]
        self.psf_grouper_enabled = bool(data.get("psf_grouper_enabled", self.psf_grouper_enabled))
        try:
            _gsf = float(data.get("psf_group_sep_fwhm", self.psf_group_sep_fwhm))
            self.psf_group_sep_fwhm = _gsf if math.isfinite(_gsf) and _gsf > 0 else 1.5
        except (TypeError, ValueError):
            self.psf_group_sep_fwhm = 1.5
        try:
            _nif = float(data.get("psf_neighbor_include_fwhm", self.psf_neighbor_include_fwhm))
            self.psf_neighbor_include_fwhm = _nif if math.isfinite(_nif) and _nif > 0 else 3.0
        except (TypeError, ValueError):
            self.psf_neighbor_include_fwhm = 3.0
        self.psf_neighbor_sub_enabled = bool(
            data.get("psf_neighbor_sub_enabled", self.psf_neighbor_sub_enabled)
        )
        for _attr, _default in (
            ("neighbor_sub_chi2_max", 120.0),
            ("neighbor_sub_residual_rms_max", 150.0),
            ("neighbor_sub_refuse_sep_fwhm", 0.8),
            ("neighbor_sub_centroid_max_fwhm", 1.0),
            ("neighbor_sub_nn_contam_dmag", 2.5),
            ("neighbor_sub_max_neighbor_overmag", 0.3),
            ("neighbor_sub_max_target_undermag", 0.2),
            ("neighbor_sub_min_recovered_snr", 5.0),
            ("neighbor_sub_regime_dmag_min", 2.5),
            ("neighbor_sub_regime_sep_max", 1.1),
        ):
            try:
                _v = float(data.get(_attr, getattr(self, _attr)))
                setattr(self, _attr, _v if math.isfinite(_v) and _v > 0 else _default)
            except (TypeError, ValueError):
                setattr(self, _attr, _default)
        self.psf_spatial_enabled = bool(data.get("psf_spatial_enabled", self.psf_spatial_enabled))
        _grid = str(data.get("psf_spatial_grid", self.psf_spatial_grid) or "3x3").lower().strip()
        self.psf_spatial_grid = _grid if "x" in _grid else "3x3"
        try:
            _mspc = int(data.get("psf_spatial_min_stars_per_cell", self.psf_spatial_min_stars_per_cell))
            self.psf_spatial_min_stars_per_cell = _mspc if _mspc >= 1 else 25
        except (TypeError, ValueError):
            self.psf_spatial_min_stars_per_cell = 25
        self.psf_quality_fallback_enabled = bool(
            data.get("psf_quality_fallback_enabled", self.psf_quality_fallback_enabled)
        )
        self.psf_adaptive_enabled = bool(data.get("psf_adaptive_enabled", self.psf_adaptive_enabled))
        try:
            self.psf_adaptive_resolve_fwhm = float(
                data.get("psf_adaptive_resolve_fwhm", self.psf_adaptive_resolve_fwhm)
            )
        except (TypeError, ValueError):
            self.psf_adaptive_resolve_fwhm = 2.0
        try:
            self.psf_adaptive_snr_lo = float(data.get("psf_adaptive_snr_lo", self.psf_adaptive_snr_lo))
        except (TypeError, ValueError):
            self.psf_adaptive_snr_lo = 15.0
        # WAVE-B STEP 6: moffat_chi2_limit hardcoded (solver/QC internal).
        try:
            _sso = int(data.get("preprocess_sky_surface_order", self.preprocess_sky_surface_order))
            self.preprocess_sky_surface_order = max(0, min(2, _sso))
        except (TypeError, ValueError):
            self.preprocess_sky_surface_order = 2
        self.preprocess_sky_surface_force_reapply = bool(
            data.get(
                "preprocess_sky_surface_force_reapply",
                self.preprocess_sky_surface_force_reapply,
            )
        )
        try:
            _ocb = int(data.get("osc_channel_binning", self.osc_channel_binning))
            self.osc_channel_binning = _ocb if _ocb in (1, 2, 3, 4) else 2
        except (TypeError, ValueError):
            self.osc_channel_binning = 2
        try:
            _ems = int(data.get("epsf_min_stars", self.epsf_min_stars))
            self.epsf_min_stars = max(10, _ems)
        except (TypeError, ValueError):
            self.epsf_min_stars = 30
        _pm = str(data.get("photometry_mode", self.photometry_mode) or "both").strip().lower()
        self.photometry_mode = _pm if _pm in ("aperture", "epsf", "both") else "both"
        try:
            self.aperture_fwhm_factor = float(data.get("aperture_fwhm_factor", self.aperture_fwhm_factor))
            if not math.isfinite(self.aperture_fwhm_factor) or self.aperture_fwhm_factor <= 0:
                self.aperture_fwhm_factor = 2.75
        except (TypeError, ValueError):
            self.aperture_fwhm_factor = 2.75
        self.aperture_fwhm_factor = max(0.25, min(6.0, float(self.aperture_fwhm_factor)))
        _apm = str(data.get("aperture_policy_mode", self.aperture_policy_mode) or "f_fixed_night").strip().lower()
        self.aperture_policy_mode = (
            _apm if _apm in ("f_fixed_night", "f_per_frame") else "f_fixed_night"
        )
        # aperture_snr_sizing (WAVE-B STEP 4 merge of aperture_fwhm_factor_small/_large).
        # New structured form wins; legacy scalar keys are accepted for one transition release.
        _asz = data.get("aperture_snr_sizing")

        def _set_snr_sizing(slot: str, raw: Any) -> None:
            try:
                v = float(raw)
            except (TypeError, ValueError):
                return
            if math.isfinite(v) and v > 0:
                self.aperture_snr_sizing[slot] = max(0.5, min(6.0, v))

        if isinstance(_asz, dict):
            for _slot in ("small", "large"):
                if _slot in _asz:
                    _set_snr_sizing(_slot, _asz[_slot])
        elif "aperture_fwhm_factor_small" in data or "aperture_fwhm_factor_large" in data:
            logging.warning(
                "[DEPRECATED] aperture_fwhm_factor_small/large in config.json are merged into "
                "aperture_snr_sizing (WAVE-B STEP 4)."
            )
            if "aperture_fwhm_factor_small" in data:
                _set_snr_sizing("small", data["aperture_fwhm_factor_small"])
            if "aperture_fwhm_factor_large" in data:
                _set_snr_sizing("large", data["aperture_fwhm_factor_large"])
        try:
            self.aperture_variable_factor = max(
                0.25,
                min(3.0, float(data.get("aperture_variable_factor", self.aperture_variable_factor))),
            )
        except (TypeError, ValueError):
            self.aperture_variable_factor = 1.0
        try:
            self.aperture_comp_factor = max(
                0.25,
                min(3.0, float(data.get("aperture_comp_factor", self.aperture_comp_factor))),
            )
        except (TypeError, ValueError):
            self.aperture_comp_factor = 1.1
        self.aperture_correction_enabled = bool(
            data.get("aperture_correction_enabled", self.aperture_correction_enabled)
        )
        try:
            self.aperture_correction_min_ref_stars = max(
                1,
                min(50, int(data.get("aperture_correction_min_ref_stars", self.aperture_correction_min_ref_stars))),
            )
        except (TypeError, ValueError):
            self.aperture_correction_min_ref_stars = 3
        try:
            _acmc = float(
                data.get("aperture_correction_max_contamination", self.aperture_correction_max_contamination)
            )
            self.aperture_correction_max_contamination = (
                float(_acmc) if math.isfinite(_acmc) and _acmc >= 0 else 0.15
            )
        except (TypeError, ValueError):
            self.aperture_correction_max_contamination = 0.15
        self.aperture_correction_max_contamination = max(0.0, min(2.0, float(self.aperture_correction_max_contamination)))
        try:
            _acms = float(
                data.get("aperture_correction_max_scatter_mag", self.aperture_correction_max_scatter_mag)
            )
            self.aperture_correction_max_scatter_mag = (
                float(_acms) if math.isfinite(_acms) and _acms >= 0 else 0.03
            )
        except (TypeError, ValueError):
            self.aperture_correction_max_scatter_mag = 0.03

        # Per-frame curve-of-growth aperture correction (gated, default OFF).
        self.cog_aperture_correction_enabled = bool(
            data.get("cog_aperture_correction_enabled", self.cog_aperture_correction_enabled)
        )
        try:
            _v = float(data.get("cog_ref_fwhm", self.cog_ref_fwhm))
            self.cog_ref_fwhm = _v if math.isfinite(_v) and _v > 0 else 4.5
        except (TypeError, ValueError):
            self.cog_ref_fwhm = 4.5
        self.cog_ref_fwhm = max(1.5, min(10.0, float(self.cog_ref_fwhm)))
        try:
            self.cog_min_stars = max(1, min(500, int(data.get("cog_min_stars", self.cog_min_stars))))
        except (TypeError, ValueError):
            self.cog_min_stars = 8
        try:
            _v = float(data.get("cog_isolation_fwhm", self.cog_isolation_fwhm))
            self.cog_isolation_fwhm = _v if math.isfinite(_v) and _v > 0 else 6.0
        except (TypeError, ValueError):
            self.cog_isolation_fwhm = 6.0
        try:
            _v = float(data.get("cog_snr_min", self.cog_snr_min))
            self.cog_snr_min = _v if math.isfinite(_v) and _v >= 0 else 50.0
        except (TypeError, ValueError):
            self.cog_snr_min = 50.0
        try:
            _v = float(data.get("cog_sat_frac", self.cog_sat_frac))
            self.cog_sat_frac = _v if math.isfinite(_v) and 0 < _v <= 1.0 else 0.85
        except (TypeError, ValueError):
            self.cog_sat_frac = 0.85
        try:
            _v = float(data.get("cog_ladder_step_px", self.cog_ladder_step_px))
            self.cog_ladder_step_px = _v if math.isfinite(_v) and _v > 0 else 0.5
        except (TypeError, ValueError):
            self.cog_ladder_step_px = 0.5
        try:
            _v = float(data.get("cog_ac_factor_max", self.cog_ac_factor_max))
            self.cog_ac_factor_max = _v if math.isfinite(_v) and _v >= 1.0 else 5.0
        except (TypeError, ValueError):
            self.cog_ac_factor_max = 5.0
        self.per_frame_saturation_enabled = bool(
            data.get("per_frame_saturation_enabled", self.per_frame_saturation_enabled)
        )
        try:
            _v = float(data.get("per_frame_sat_min_clean_frac", self.per_frame_sat_min_clean_frac))
            self.per_frame_sat_min_clean_frac = (
                _v if math.isfinite(_v) else 0.5
            )
        except (TypeError, ValueError):
            self.per_frame_sat_min_clean_frac = 0.5
        self.per_frame_sat_min_clean_frac = max(
            0.1, min(1.0, float(self.per_frame_sat_min_clean_frac))
        )
        self.aperture_correction_max_scatter_mag = max(
            0.0, min(2.0, float(self.aperture_correction_max_scatter_mag))
        )
        # WAVE-B STEP 5 (DELETE-DB-DUP): gain (FITS EGAIN then DB EQUIPMENTS.GAIN_ADU) and
        # read_noise (DB EQUIPMENTS.READNOISE_E then FITS) are resolver-authoritative via
        # param_resolver at run time; no longer loaded from or saved to config.json. The fields
        # keep their dataclass defaults as last-resort fallbacks.
        try:
            self.err_empty_apertures_n = max(
                16,
                min(256, int(data.get("err_empty_apertures_n", self.err_empty_apertures_n))),
            )
        except (TypeError, ValueError):
            self.err_empty_apertures_n = 64
        try:
            self.err_empty_apertures_min = max(
                1,
                min(256, int(data.get("err_empty_apertures_min", self.err_empty_apertures_min))),
            )
        except (TypeError, ValueError):
            self.err_empty_apertures_min = 16
        try:
            self.masterstar_best_of_n = max(
                1,
                min(25, int(data.get("masterstar_best_of_n", self.masterstar_best_of_n))),
            )
        except (TypeError, ValueError):
            self.masterstar_best_of_n = 10
        # WAVE-B STEP 6: sky_adu_fallback hardcoded (module constant in pipeline.py).
        try:
            self.phase01_ct_min_comp = max(
                2,
                min(30, int(data.get("phase01_ct_min_comp", self.phase01_ct_min_comp))),
            )
        except (TypeError, ValueError):
            self.phase01_ct_min_comp = 7
        _act = str(data.get("apply_color_term", self.apply_color_term) or "auto").strip().lower()
        if _act in ("1", "true", "yes", "on"):
            self.apply_color_term = "on"
        elif _act in ("0", "false", "no", "off"):
            self.apply_color_term = "off"
        else:
            self.apply_color_term = "auto"
        for _clk in ("color_level_k_mag_per_bprp", "color_level_k_stderr_mag_per_bprp"):
            _raw = data.get(_clk, getattr(self, _clk))
            if _raw is None or _raw == "":
                setattr(self, _clk, None)
            else:
                try:
                    _fv = float(_raw)
                    setattr(self, _clk, _fv if math.isfinite(_fv) else None)
                except (TypeError, ValueError):
                    setattr(self, _clk, None)
        try:
            _iso = float(data.get("snr_cog_isolation_fwhm", self.snr_cog_isolation_fwhm))
            self.snr_cog_isolation_fwhm = _iso if math.isfinite(_iso) and _iso > 0 else 3.0
        except (TypeError, ValueError):
            self.snr_cog_isolation_fwhm = 3.0
        _k2m = str(data.get("k2_mode", self.k2_mode) or "literature").strip().lower()
        if _k2m in ("0", "false", "no", "off", "none"):
            self.k2_mode = "off"
        elif _k2m in ("fit", "fit_else_literature", "night_fit"):
            self.k2_mode = _k2m
        else:
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
        _ssm = data.get("sigma_sys_mag", self.sigma_sys_mag)
        if isinstance(_ssm, dict):
            parsed_ssm: dict[str, float] = {}
            for k, v in _ssm.items():
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(fv) and fv >= 0:
                    parsed_ssm[str(k)] = fv
            self.sigma_sys_mag = parsed_ssm
        try:
            self.gain_container_scale = max(
                1.0, float(data.get("gain_container_scale", self.gain_container_scale))
            )
        except (TypeError, ValueError):
            self.gain_container_scale = 4.0
        try:
            self.photon_transfer_ci_max_width_factor = max(
                1.0,
                float(
                    data.get(
                        "photon_transfer_ci_max_width_factor",
                        self.photon_transfer_ci_max_width_factor,
                    )
                ),
            )
        except (TypeError, ValueError):
            self.photon_transfer_ci_max_width_factor = 3.0
        self.k2_fit_enabled = bool(data.get("k2_fit_enabled", self.k2_fit_enabled))
        try:
            self.k2_ceiling = max(0.0, float(data.get("k2_ceiling", self.k2_ceiling)))
        except (TypeError, ValueError):
            self.k2_ceiling = 0.1
        try:
            self.k2_fit_min_detectability = max(
                0.1, float(data.get("k2_fit_min_detectability", self.k2_fit_min_detectability))
            )
        except (TypeError, ValueError):
            self.k2_fit_min_detectability = 3.0
        try:
            self.k2_fit_consistency_sigma = max(
                0.5, float(data.get("k2_fit_consistency_sigma", self.k2_fit_consistency_sigma))
            )
        except (TypeError, ValueError):
            self.k2_fit_consistency_sigma = 2.0
        try:
            self.k2_fit_lit_factor = max(1.0, float(data.get("k2_fit_lit_factor", self.k2_fit_lit_factor)))
        except (TypeError, ValueError):
            self.k2_fit_lit_factor = 4.0
        try:
            self.phase01_ct_extrapolation_tol = max(
                0.0,
                float(data.get("phase01_ct_extrapolation_tol", self.phase01_ct_extrapolation_tol)),
            )
        except (TypeError, ValueError):
            self.phase01_ct_extrapolation_tol = 0.0
        self.phase01_flux_col = str(data.get("phase01_flux_col", self.phase01_flux_col)).strip() or "dao_flux"
        self.temporal_binning_enabled = bool(
            data.get("temporal_binning_enabled", self.temporal_binning_enabled)
        )
        try:
            self.temporal_bin_window = max(
                0,
                min(51, int(data.get("temporal_bin_window", self.temporal_bin_window))),
            )
        except (TypeError, ValueError):
            self.temporal_bin_window = 0
        self.democratic_detrend_enabled = bool(
            data.get("democratic_detrend_enabled", self.democratic_detrend_enabled)
        )
        try:
            self.democratic_sg_window_frac = max(
                0.05,
                min(
                    0.95,
                    float(data.get("democratic_sg_window_frac", self.democratic_sg_window_frac)),
                ),
            )
        except (TypeError, ValueError):
            self.democratic_sg_window_frac = 0.5
        self.pytics_enabled = bool(data.get("pytics_enabled", self.pytics_enabled))
        try:
            self.pytics_n_iter = max(1, min(20, int(data.get("pytics_n_iter", self.pytics_n_iter))))
        except (TypeError, ValueError):
            self.pytics_n_iter = 5
        try:
            self.comp_max_slope_mmag_hr = max(
                0.0,
                min(500.0, float(data.get("comp_max_slope_mmag_hr", self.comp_max_slope_mmag_hr))),
            )
        except (TypeError, ValueError):
            self.comp_max_slope_mmag_hr = 5.0
        try:
            self.comp_slope_significance_k = max(
                0.0,
                min(10.0, float(data.get("comp_slope_significance_k", self.comp_slope_significance_k))),
            )
        except (TypeError, ValueError):
            self.comp_slope_significance_k = 3.0
        _legacy_iter = bool(data.get("comp_iterative_clip_enabled", self.comp_iterative_clip_enabled))
        self.comp_sparse_fallback_enabled = bool(
            data.get(
                "comp_sparse_fallback_enabled",
                data.get("comp_iterative_clip_enabled", self.comp_sparse_fallback_enabled),
            )
        )
        if _legacy_iter and not self.comp_sparse_fallback_enabled:
            self.comp_sparse_fallback_enabled = True
        self.comp_iterative_clip_enabled = bool(self.comp_sparse_fallback_enabled)
        try:
            self.comp_sparse_fallback_min = int(
                data.get("comp_sparse_fallback_min", self.comp_sparse_fallback_min)
            )
        except (TypeError, ValueError):
            self.comp_sparse_fallback_min = 0
        try:
            self.annulus_inner_fwhm = float(data.get("annulus_inner_fwhm", self.annulus_inner_fwhm))
            self.annulus_outer_fwhm = float(data.get("annulus_outer_fwhm", self.annulus_outer_fwhm))
        except (TypeError, ValueError):
            self.annulus_inner_fwhm = 5.5
            self.annulus_outer_fwhm = 10.5
        if self.annulus_outer_fwhm <= self.annulus_inner_fwhm:
            self.annulus_outer_fwhm = self.annulus_inner_fwhm + 1.0
        try:
            self.nonlinearity_peak_percentile = float(
                data.get("nonlinearity_peak_percentile", self.nonlinearity_peak_percentile)
            )
        except (TypeError, ValueError):
            self.nonlinearity_peak_percentile = 20.0
        self.nonlinearity_peak_percentile = max(0.0, min(50.0, float(self.nonlinearity_peak_percentile)))
        try:
            self.nonlinearity_fwhm_ratio = float(data.get("nonlinearity_fwhm_ratio", self.nonlinearity_fwhm_ratio))
        except (TypeError, ValueError):
            self.nonlinearity_fwhm_ratio = 1.25
        self.nonlinearity_fwhm_ratio = max(1.01, min(3.0, float(self.nonlinearity_fwhm_ratio)))
        try:
            self.bpm_dark_mad_sigma = float(data.get("bpm_dark_mad_sigma", self.bpm_dark_mad_sigma))
        except (TypeError, ValueError):
            self.bpm_dark_mad_sigma = 5.0
        self.bpm_dark_mad_sigma = max(2.0, min(12.0, float(self.bpm_dark_mad_sigma)))

        # WAVE-B STEP 6: masterstar_solver_use_draft_median_if_hint_sep_deg and
        # masterstar_optimizer_mirror_extra_log hardcoded (module constants in pipeline.py).
        self.debug_platesolver = bool(data.get("debug_platesolver", self.debug_platesolver))
        try:
            self.masterstar_platesolve_sip_max_order = int(
                data.get("masterstar_platesolve_sip_max_order", self.masterstar_platesolve_sip_max_order)
            )
        except (TypeError, ValueError):
            self.masterstar_platesolve_sip_max_order = 5
        self.masterstar_platesolve_sip_max_order = max(2, min(5, int(self.masterstar_platesolve_sip_max_order)))
        try:
            self.masterstar_platesolve_sip_min_order = int(
                data.get("masterstar_platesolve_sip_min_order", self.masterstar_platesolve_sip_min_order)
            )
        except (TypeError, ValueError):
            self.masterstar_platesolve_sip_min_order = 3
        self.masterstar_platesolve_sip_min_order = max(2, min(5, int(self.masterstar_platesolve_sip_min_order)))
        if self.masterstar_platesolve_sip_min_order > self.masterstar_platesolve_sip_max_order:
            self.masterstar_platesolve_sip_min_order = int(self.masterstar_platesolve_sip_max_order)
        try:
            self.masterstar_dao_threshold_sigma = float(
                data.get("masterstar_dao_threshold_sigma", self.masterstar_dao_threshold_sigma)
            )
        except (TypeError, ValueError):
            self.masterstar_dao_threshold_sigma = 1.8
        self.masterstar_dao_threshold_sigma = max(0.1, min(12.0, float(self.masterstar_dao_threshold_sigma)))
        try:
            self.dao_detection_n_equiv = float(data.get("dao_detection_n_equiv", self.dao_detection_n_equiv))
        except (TypeError, ValueError):
            self.dao_detection_n_equiv = 3.78
        self.dao_detection_n_equiv = max(0.5, min(12.0, float(self.dao_detection_n_equiv)))
        try:
            self.dao_centroid_max_shift_fwhm = float(
                data.get("dao_centroid_max_shift_fwhm", self.dao_centroid_max_shift_fwhm)
            )
        except (TypeError, ValueError):
            self.dao_centroid_max_shift_fwhm = 1.0
        self.dao_centroid_max_shift_fwhm = max(0.1, min(5.0, float(self.dao_centroid_max_shift_fwhm)))
        try:
            self.admission_sat_peak_frac = float(
                data.get("admission_sat_peak_frac", self.admission_sat_peak_frac)
            )
        except (TypeError, ValueError):
            self.admission_sat_peak_frac = 0.70
        self.admission_sat_peak_frac = max(0.5, min(0.95, float(self.admission_sat_peak_frac)))
        try:
            self.masterstar_prematch_peak_sigma_floor = float(
                data.get("masterstar_prematch_peak_sigma_floor", self.masterstar_prematch_peak_sigma_floor)
            )
        except (TypeError, ValueError):
            self.masterstar_prematch_peak_sigma_floor = 3.2
        self.masterstar_prematch_peak_sigma_floor = max(0.5, min(6.0, float(self.masterstar_prematch_peak_sigma_floor)))

        # WAVE-B STEP 6: masterstar_platesolve_prewrite_rms_max_px / _prewrite_relaxed_rms_max_px /
        # _nn_refine_max_rms_px and masterstar_sip_force_rms_guard_ratio hardcoded as solver internals
        # (module constants in pipeline.py / vyvar_platesolver.py).

        try:
            self.masterstar_catalog_recovery_min = float(
                data.get("masterstar_catalog_recovery_min", self.masterstar_catalog_recovery_min)
            )
            if not math.isfinite(self.masterstar_catalog_recovery_min):
                self.masterstar_catalog_recovery_min = 0.65
        except (TypeError, ValueError):
            self.masterstar_catalog_recovery_min = 0.65
        self.masterstar_catalog_recovery_min = max(
            0.40, min(0.95, float(self.masterstar_catalog_recovery_min))
        )
        try:
            self.masterstar_min_matched_floor = int(
                data.get("masterstar_min_matched_floor", self.masterstar_min_matched_floor)
            )
        except (TypeError, ValueError):
            self.masterstar_min_matched_floor = 40
        self.masterstar_min_matched_floor = max(
            1, min(500, int(self.masterstar_min_matched_floor))
        )
        try:
            self.masterstar_centre_rms_max_px = float(
                data.get("masterstar_centre_rms_max_px", self.masterstar_centre_rms_max_px)
            )
            if not math.isfinite(self.masterstar_centre_rms_max_px):
                self.masterstar_centre_rms_max_px = 1.20
        except (TypeError, ValueError):
            self.masterstar_centre_rms_max_px = 1.20
        self.masterstar_centre_rms_max_px = max(
            0.5, min(5.0, float(self.masterstar_centre_rms_max_px))
        )
        try:
            self.masterstar_distortion_benign_ratio_max = float(
                data.get(
                    "masterstar_distortion_benign_ratio_max",
                    self.masterstar_distortion_benign_ratio_max,
                )
            )
            if not math.isfinite(self.masterstar_distortion_benign_ratio_max):
                self.masterstar_distortion_benign_ratio_max = 3.20
        except (TypeError, ValueError):
            self.masterstar_distortion_benign_ratio_max = 3.20
        self.masterstar_distortion_benign_ratio_max = max(
            2.0, min(5.0, float(self.masterstar_distortion_benign_ratio_max))
        )
        _mode = str(data.get("masterstar_accept_mode", self.masterstar_accept_mode) or "odds").strip().lower()
        self.masterstar_accept_mode = _mode if _mode in ("odds", "fraction") else "odds"
        # WAVE-B STEP 6: masterstar_odds_match_floor / _odds_k / _odds_min_quadrants /
        # _false_alarm_p_max hardcoded as odds-verification internals (module constants in
        # vyvar_platesolver.py).
        try:
            self.masterstar_quality_crowded_n_cat_min = int(
                data.get("masterstar_quality_crowded_n_cat_min", self.masterstar_quality_crowded_n_cat_min)
            )
        except (TypeError, ValueError):
            self.masterstar_quality_crowded_n_cat_min = 800
        self.masterstar_quality_crowded_n_cat_min = max(
            100, min(20000, int(self.masterstar_quality_crowded_n_cat_min))
        )
        self.masterstar_detection_cap_adaptive = bool(
            data.get("masterstar_detection_cap_adaptive", self.masterstar_detection_cap_adaptive)
        )
        try:
            self.masterstar_detection_cap_min = int(
                data.get("masterstar_detection_cap_min", self.masterstar_detection_cap_min)
            )
        except (TypeError, ValueError):
            self.masterstar_detection_cap_min = 250
        self.masterstar_detection_cap_min = max(50, min(2000, int(self.masterstar_detection_cap_min)))
        try:
            self.masterstar_detection_cap_max = int(
                data.get("masterstar_detection_cap_max", self.masterstar_detection_cap_max)
            )
        except (TypeError, ValueError):
            self.masterstar_detection_cap_max = 800
        self.masterstar_detection_cap_max = max(
            int(self.masterstar_detection_cap_min),
            min(5000, int(self.masterstar_detection_cap_max)),
        )
        try:
            self.masterstar_detection_cap_k = float(
                data.get("masterstar_detection_cap_k", self.masterstar_detection_cap_k)
            )
            if not math.isfinite(self.masterstar_detection_cap_k):
                self.masterstar_detection_cap_k = 0.08
        except (TypeError, ValueError):
            self.masterstar_detection_cap_k = 0.08
        self.masterstar_detection_cap_k = max(0.01, min(1.0, float(self.masterstar_detection_cap_k)))
        self.masterstar_sibling_recovery_enabled = bool(
            data.get("masterstar_sibling_recovery_enabled", self.masterstar_sibling_recovery_enabled)
        )
        try:
            self.masterstar_sibling_min_matched = int(
                data.get("masterstar_sibling_min_matched", self.masterstar_sibling_min_matched)
            )
        except (TypeError, ValueError):
            self.masterstar_sibling_min_matched = 40
        self.masterstar_sibling_min_matched = max(
            1, min(500, int(self.masterstar_sibling_min_matched))
        )
        try:
            self.masterstar_sibling_rms_max_px = float(
                data.get("masterstar_sibling_rms_max_px", self.masterstar_sibling_rms_max_px)
            )
            if not math.isfinite(self.masterstar_sibling_rms_max_px):
                self.masterstar_sibling_rms_max_px = 2.0
        except (TypeError, ValueError):
            self.masterstar_sibling_rms_max_px = 2.0
        self.masterstar_sibling_rms_max_px = max(
            0.5, min(10.0, float(self.masterstar_sibling_rms_max_px))
        )
        try:
            self.masterstar_sibling_min_quadrants = int(
                data.get("masterstar_sibling_min_quadrants", self.masterstar_sibling_min_quadrants)
            )
        except (TypeError, ValueError):
            self.masterstar_sibling_min_quadrants = 3
        self.masterstar_sibling_min_quadrants = max(
            1, min(4, int(self.masterstar_sibling_min_quadrants))
        )
        try:
            self.masterstar_sibling_stack_n = int(
                data.get("masterstar_sibling_stack_n", self.masterstar_sibling_stack_n)
            )
        except (TypeError, ValueError):
            self.masterstar_sibling_stack_n = 10
        self.masterstar_sibling_stack_n = max(2, min(50, int(self.masterstar_sibling_stack_n)))

        # WAVE-B STEP 6: platesolve_anisotropy_threshold hardcoded (module constant in pipeline.py).

        def _f01(key: str, default: float, lo: float, hi: float) -> None:
            try:
                v = float(data.get(key, getattr(self, key)))
                if not math.isfinite(v):
                    raise ValueError
                setattr(self, key, max(lo, min(hi, v)))
            except (TypeError, ValueError, AttributeError):
                setattr(self, key, float(default))

        def _i01(key: str, default: int, lo: int, hi: int) -> None:
            try:
                v = int(data.get(key, getattr(self, key)))
                setattr(self, key, max(lo, min(hi, v)))
            except (TypeError, ValueError, AttributeError):
                setattr(self, key, int(default))

        _f01("phase01_comparison_max_dist_deg", 1.0, 0.05, 10.0)
        _f01("phase01_comparison_max_mag_diff", 0.25, 0.05, 5.0)
        _f01("phase01_comparison_mag_bright_threshold", 12.0, 6.0, 18.0)
        _f01("phase01_comparison_max_mag_diff_bright_floor", 1.25, 0.0, 4.0)
        _f01("phase01_comparison_max_mag_diff_absolute", 3.0, 1.0, 10.0)
        _f01("comp_max_delta_bprp", 0.79, 0.0, 5.0)
        # comp_color_tiers (WAVE-B STEP 4 merge of comp_tier{1..4}_{bprp_limit,weight}).
        # New structured form wins; legacy scalar keys are accepted for one transition release.
        _cct_defaults = [
            {"bprp": 0.15, "w": 1.00},
            {"bprp": 0.30, "w": 0.85},
            {"bprp": 0.55, "w": 0.50},
            {"bprp": 1.10, "w": 0.25},
        ]

        def _clamp_tier_row(bprp: float, w: float) -> dict[str, float]:
            return {
                "bprp": max(0.02, min(5.0, float(bprp))),
                "w": max(0.01, min(1.0, float(w))),
            }

        _cct_raw = data.get("comp_color_tiers")
        _legacy_tier = any(
            f"comp_tier{i}_bprp_limit" in data or f"comp_tier{i}_weight" in data
            for i in (1, 2, 3, 4)
        )
        if isinstance(_cct_raw, list) and _cct_raw:
            _rows: list[dict[str, float]] = []
            for _row in _cct_raw:
                if not isinstance(_row, dict):
                    continue
                try:
                    _rows.append(_clamp_tier_row(_row["bprp"], _row["w"]))
                except (KeyError, TypeError, ValueError):
                    continue
            if _rows:
                self.comp_color_tiers = _rows
        elif _legacy_tier:
            logging.warning(
                "[DEPRECATED] comp_tier{1..4}_{bprp_limit,weight} in config.json are merged into "
                "comp_color_tiers (WAVE-B STEP 4)."
            )
            _rows = []
            for i in (1, 2, 3, 4):
                _d = _cct_defaults[i - 1]
                try:
                    _b = float(data.get(f"comp_tier{i}_bprp_limit", _d["bprp"]))
                    _w = float(data.get(f"comp_tier{i}_weight", _d["w"]))
                except (TypeError, ValueError):
                    _b, _w = _d["bprp"], _d["w"]
                _rows.append(_clamp_tier_row(_b, _w))
            self.comp_color_tiers = _rows
        # phase01_tiers (WAVE-B STEP 4 merge of phase01_tier{1..4}_mag; previously code-default-only).
        _pt_raw = data.get("phase01_tiers")
        if isinstance(_pt_raw, list) and _pt_raw:
            _mags: list[float] = []
            for _m in _pt_raw:
                try:
                    _mags.append(max(0.0, min(10.0, float(_m))))
                except (TypeError, ValueError):
                    continue
            if _mags:
                self.phase01_tiers = _mags
        _f01("comp_contamination_penalty_k", 3.0, 0.0, 20.0)
        _f01("calibration_master_ccd_temp_tolerance_c", 0.5, 0.01, 20.0)
        self.gs11_dilution_enabled = bool(
            data.get("gs11_dilution_enabled", self.gs11_dilution_enabled)
        )
        _f01("gs11_dilution_aperture_arcsec", 0.0, 0.0, 120.0)
        _f01("gs11_dilution_mag_limit_delta", 5.0, 0.5, 15.0)
        _f01("gs11_comp_max_dilution", 0.90, 0.01, 1.0)
        _f01("gs11_comp_suspect_dilution", 0.98, 0.01, 1.0)
        _f01("gs11_target_min_dilution", 0.50, 0.01, 1.0)
        if float(self.gs11_comp_suspect_dilution) < float(self.gs11_comp_max_dilution):
            self.gs11_comp_suspect_dilution = float(self.gs11_comp_max_dilution)
        _i01("phase01_comparison_n_comp_min", 3, 2, 12)
        _i01("phase01_comparison_n_comp_max", 12, 3, 20)
        if int(self.phase01_comparison_n_comp_max) < int(self.phase01_comparison_n_comp_min):
            self.phase01_comparison_n_comp_max = int(self.phase01_comparison_n_comp_min)
        _i01("comp_trust_min_comps", 5, 3, 20)
        if int(self.comp_trust_min_comps) > int(self.phase01_comparison_n_comp_max):
            self.comp_trust_min_comps = int(self.phase01_comparison_n_comp_max)
        _i01("lc_quality_min_frames", 20, 3, 500)
        _i01("lc_quality_short_min_frames", 3, 2, 100)
        _f01("lc_quality_min_normal_frac", 0.5, 0.1, 1.0)
        if int(self.lc_quality_short_min_frames) > int(self.lc_quality_min_frames):
            self.lc_quality_short_min_frames = int(self.lc_quality_min_frames)
        _i01("check_star_min_epochs", 5, 3, 50)
        _f01("sparse_trust_T_green", 1.5, 0.5, 10.0)
        _f01("sparse_trust_T_red", 4.0, 1.0, 20.0)
        _f01("sparse_trust_X2_RED", 0.0004, 0.0, 0.01)
        _f01("check_select_rms_floor", 1e-4, 0.0, 0.01)
        _f01("comp_select_rms_floor", 1e-6, 0.0, 0.01)
        _f01("phase01_comparison_max_comp_rms", 0.05, 0.01, 0.5)
        _f01("comp_rms_loo_photon_k", 5.0, 1.0, 50.0)
        _f01("phase01_comparison_min_dist_arcsec", 60.0, 0.0, 600.0)
        _f01("phase01_comparison_min_frames_frac", 0.3, 0.05, 0.95)
        self.phase01_comparison_exclude_gaia_nss = bool(
            data.get("phase01_comparison_exclude_gaia_nss", self.phase01_comparison_exclude_gaia_nss)
        )
        self.phase01_comparison_exclude_gaia_extobj = bool(
            data.get("phase01_comparison_exclude_gaia_extobj", self.phase01_comparison_exclude_gaia_extobj)
        )
        self.phase01_use_bprp_primary = bool(
            data.get("phase01_use_bprp_primary", self.phase01_use_bprp_primary)
        )
        # Plate-scale ceiling matches the runtime resolver clamp [0.1, 30.0] so that a
        # wide-field config value (e.g. 9.77"/px) survives load instead of being capped
        # at 5.0. phase01_* keeps lo=0.0 because 0.0 is the "auto / unset" sentinel
        # (consumed as ``phase01_plate_scale_arcsec_per_px or 1.3``); clamping it to 0.1
        # would turn "auto" into a real 0.1"/px scale.
        # WAVE-B STEP 5 (DELETE-DB-DUP): plate_scale_arcsec_per_px (WCS/CD then DB optics) and
        # phase01_plate_scale_arcsec_per_px (WCS-superseded) resolve at run time; no longer loaded
        # from or saved to config.json. Fields keep their dataclass defaults as fallbacks.
        _f01("exoplanet_match_max_sep_arcsec", 3.0, 0.5, 30.0)
        _f01("phase01_comparison_max_psf_chi2", 50.0, 1.0, 500.0)
        _f01("phase01_comparison_max_fwhm_factor", 1.5, 0.5, 5.0)
        _f01("phase01_comparison_isolation_radius_px", 25.0, 1.0, 200.0)
        # WAVE-B STEP 3: frame_width_px / frame_height_px are INTERNAL. Frame geometry
        # resolves from FITS NAXIS1/2 at run time; the dataclass default is the only
        # fallback. They are no longer loaded from or saved to config.json. Warn once if a
        # legacy config.json still carries them.
        if "frame_width_px" in data or "frame_height_px" in data:
            logging.warning(
                "[DEPRECATED] frame_width_px/frame_height_px in config.json are ignored; "
                "frame dimensions resolve from FITS NAXIS at run time (WAVE-B STEP 3)."
            )

        _chip_m = data.get("phase01_chip_interior_margin_px")
        if _chip_m is None and "phase01_suspected_interior_margin_px" in data:
            _chip_m = data.get("phase01_suspected_interior_margin_px")
        if _chip_m is not None and _chip_m != "":
            try:
                self.phase01_chip_interior_margin_px = max(0, min(2000, int(_chip_m)))
            except (TypeError, ValueError):
                self.phase01_chip_interior_margin_px = 100

        # Variability Detection
        try:
            self.variability_min_frames = max(
                1, int(data.get("variability_min_frames", self.variability_min_frames))
            )
        except (TypeError, ValueError):
            self.variability_min_frames = 30
        try:
            self.variability_min_frames_frac = float(
                data.get("variability_min_frames_frac", self.variability_min_frames_frac)
            )
        except (TypeError, ValueError):
            self.variability_min_frames_frac = 0.50
        self.variability_min_frames_frac = max(0.05, min(0.99, float(self.variability_min_frames_frac)))

        def _vfloat(key: str, default: float, lo: float, hi: float) -> None:
            try:
                v = float(data.get(key, getattr(self, key)))
                if not math.isfinite(v):
                    raise ValueError
                setattr(self, key, max(lo, min(hi, v)))
            except (TypeError, ValueError, AttributeError):
                setattr(self, key, float(default))

        def _vint(key: str, default: int, lo: int, hi: int) -> None:
            try:
                v = int(data.get(key, getattr(self, key)))
                setattr(self, key, max(lo, min(hi, v)))
            except (TypeError, ValueError, AttributeError):
                setattr(self, key, int(default))

        _vint("variability_p85_filter", 85, 50, 99)
        _vfloat("variability_slope_floor", 0.02, 0.0, 1.0)
        _vfloat("variability_sigma_threshold", 2.3, 0.5, 20.0)
        _vfloat("variability_comp_floor_factor", 1.5, 0.5, 10.0)
        _vfloat("variability_smoothness_max", 0.80, 0.05, 1.0)
        _vfloat("variability_mag_limit", 14.5, 0.0, 30.0)
        _vfloat("variability_min_rms_pct", 1.5, 0.0, 100.0)
        _vfloat("variability_min_amplitude_mag", 0.01, 0.0, 10.0)
        _vfloat("variability_clip_ratio_min", 0.80, 0.0, 1.0)
        _vfloat("variability_vdi_z_threshold", 3.0, 0.0, 50.0)
        _vint("variability_min_points_rms", 20, 5, 10_000)

        try:
            self.field_density_sparse_threshold = float(
                data.get("field_density_sparse_threshold", self.field_density_sparse_threshold)
            )
        except (TypeError, ValueError):
            self.field_density_sparse_threshold = 300.0
        self.field_density_sparse_threshold = max(1.0, min(50_000.0, float(self.field_density_sparse_threshold)))
        try:
            self.field_density_dense_threshold = float(
                data.get("field_density_dense_threshold", self.field_density_dense_threshold)
            )
        except (TypeError, ValueError):
            self.field_density_dense_threshold = 1000.0
        self.field_density_dense_threshold = max(
            float(self.field_density_sparse_threshold) + 1.0,
            min(100_000.0, float(self.field_density_dense_threshold)),
        )
        self.field_density_adaptive_enabled = bool(
            data.get("field_density_adaptive_enabled", self.field_density_adaptive_enabled)
        )
        self.crowding_classifier_enabled = bool(
            data.get("crowding_classifier_enabled", self.crowding_classifier_enabled)
        )
        try:
            self.crowding_blend_tighten_threshold = float(
                data.get("crowding_blend_tighten_threshold", self.crowding_blend_tighten_threshold)
            )
        except (TypeError, ValueError):
            self.crowding_blend_tighten_threshold = 0.04
        self.crowding_blend_tighten_threshold = max(0.0, min(1.0, float(self.crowding_blend_tighten_threshold)))
        try:
            self.crowding_comp_availability_loosen_count = float(
                data.get("crowding_comp_availability_loosen_count", self.crowding_comp_availability_loosen_count)
            )
        except (TypeError, ValueError):
            self.crowding_comp_availability_loosen_count = 500.0
        self.crowding_comp_availability_loosen_count = max(
            0.0, min(1_000_000.0, float(self.crowding_comp_availability_loosen_count))
        )
        try:
            self.crowding_tighten_min_fwhm_px = float(
                data.get("crowding_tighten_min_fwhm_px", self.crowding_tighten_min_fwhm_px)
            )
        except (TypeError, ValueError):
            self.crowding_tighten_min_fwhm_px = 3.0
        self.crowding_tighten_min_fwhm_px = max(0.0, min(30.0, float(self.crowding_tighten_min_fwhm_px)))
        self.comp_pool_derived_admission = bool(
            data.get("comp_pool_derived_admission", self.comp_pool_derived_admission)
        )
        try:
            self.comparison_stars_pool_n = int(
                data.get("comparison_stars_pool_n", self.comparison_stars_pool_n)
            )
        except (TypeError, ValueError):
            self.comparison_stars_pool_n = 0
        self.comparison_stars_pool_n = max(0, min(50000, int(self.comparison_stars_pool_n)))
        for _ck, _default in (
            ("comp_weight_c_col_mag_per_bprp", None),
            ("comp_weight_c_dist_mag_per_deg", None),
        ):
            raw = data.get(_ck, _default)
            if raw is None or raw == "":
                setattr(self, _ck, None)
            else:
                try:
                    fv = float(raw)
                    setattr(self, _ck, fv if math.isfinite(fv) else None)
                except (TypeError, ValueError):
                    setattr(self, _ck, None)
        try:
            self.comp_weight_airmass_span = float(
                data.get("comp_weight_airmass_span", self.comp_weight_airmass_span)
            )
        except (TypeError, ValueError):
            self.comp_weight_airmass_span = 0.0
        if not math.isfinite(float(self.comp_weight_airmass_span)) or float(self.comp_weight_airmass_span) < 0:
            self.comp_weight_airmass_span = 0.0
        self.comp_weight_optics_kind = str(
            data.get("comp_weight_optics_kind", self.comp_weight_optics_kind) or "unknown"
        ).strip().lower() or "unknown"
        self.forced_photometry_enabled = bool(
            data.get("forced_photometry_enabled", self.forced_photometry_enabled)
        )
        try:
            self.forced_photometry_centroid_bound_fwhm = float(
                data.get(
                    "forced_photometry_centroid_bound_fwhm",
                    self.forced_photometry_centroid_bound_fwhm,
                )
            )
        except (TypeError, ValueError):
            self.forced_photometry_centroid_bound_fwhm = 2.5
        if (
            not math.isfinite(float(self.forced_photometry_centroid_bound_fwhm))
            or float(self.forced_photometry_centroid_bound_fwhm) < 0.5
        ):
            self.forced_photometry_centroid_bound_fwhm = 2.5
        try:
            self.forced_photometry_margin_px = float(
                data.get("forced_photometry_margin_px", self.forced_photometry_margin_px)
            )
        except (TypeError, ValueError):
            self.forced_photometry_margin_px = 0.0
        if not math.isfinite(float(self.forced_photometry_margin_px)) or float(self.forced_photometry_margin_px) < 0:
            self.forced_photometry_margin_px = 0.0

        self.tess_enabled = bool(data.get("tess_enabled", self.tess_enabled))

        def _opt_float_none(key: str) -> None:
            v = data.get(key)
            if v is None or v == "":
                setattr(self, key, None)
                return
            try:
                fv = float(v)
                setattr(self, key, fv if math.isfinite(fv) else None)
            except (TypeError, ValueError):
                setattr(self, key, None)

        for _unit_norm_key in (
            "blind_verify_match_tol_arcsec",
            "cog_ladder_step_fwhm",
            "hrd_color_bg_box_arcsec",
            "masterstar_centre_rms_max_arcsec",
            "masterstar_sibling_rms_max_arcsec",
            "phase01_chip_interior_margin_arcsec",
            "phase01_comparison_isolation_radius_arcsec",
            "phase01_comparison_max_dist_fov_frac",
            "qc_max_hfr_fwhm_ratio",
            "sips_dao_fwhm_fwhm_factor",
        ):
            _opt_float_none(_unit_norm_key)

    # --- structured-key accessors (WAVE-B STEP 4) ---------------------------------- #
    def comp_tier_bprp_limits(self) -> list[float]:
        """|dBP-RP| colour limit per comp tier, row order, from ``comp_color_tiers``."""
        return [float(t.get("bprp", 0.0)) for t in self.comp_color_tiers]

    def comp_tier_weights(self) -> list[float]:
        """Ensemble/AC weight per comp tier, row order, from ``comp_color_tiers``."""
        return [float(t.get("w", 0.0)) for t in self.comp_color_tiers]

    def phase01_tier_mags(self) -> list[float]:
        """Phase-0/1 |dmag| tier bounds, row order, from ``phase01_tiers``."""
        return [float(x) for x in self.phase01_tiers]

    def to_json(self) -> dict[str, Any]:
        return {
            "archive_root": str(self.archive_root),
            "calibration_library_root": str(self.calibration_library_root),
            "database_path": str(self.database_path),
            "masterdark_validity_days": int(self.masterdark_validity_days),
            "masterflat_validity_days": int(self.masterflat_validity_days),
            "calibration_library_native_binning": (
                None
                if self.calibration_library_native_binning is None
                else int(self.calibration_library_native_binning)
            ),
            "calibration_master_ccd_temp_tolerance_c": float(
                self.calibration_master_ccd_temp_tolerance_c
            ),
            "gaia_db_path": str(self.gaia_db_path or ""),
            "hrd_online_enrich_enabled": bool(self.hrd_online_enrich_enabled),
            "hrd_simbad_enrich_enabled": bool(self.hrd_simbad_enrich_enabled),
            "hrd_enrich_max_candidates": int(self.hrd_enrich_max_candidates),
            "hrd_enrich_tap_timeout_s": float(self.hrd_enrich_tap_timeout_s),
            "hrd_parallax_min_mas": float(self.hrd_parallax_min_mas),
            "hrd_parallax_snr_min": float(self.hrd_parallax_snr_min),
            "hrd_max_per_category": int(self.hrd_max_per_category),
            "hrd_min_per_net": int(self.hrd_min_per_net),
            "hrd_nss_category_enabled": bool(self.hrd_nss_category_enabled),
            "hrd_dsc_confirm_prob": float(self.hrd_dsc_confirm_prob),
            "hrd_color_field_enabled": bool(self.hrd_color_field_enabled),
            "hrd_color_saturation": float(self.hrd_color_saturation),
            "hrd_color_highlight_mode": str(self.hrd_color_highlight_mode),
            "hrd_color_chroma_snr": float(self.hrd_color_chroma_snr),
            "hrd_color_white_point": str(self.hrd_color_white_point),
            "hrd_color_chroma_boost": float(self.hrd_color_chroma_boost),
            "hrd_color_bg_box_px": int(self.hrd_color_bg_box_px),
            "hrd_color_bg_box_arcsec": self.hrd_color_bg_box_arcsec,
            "blind_index_fine_path": str(self.blind_index_fine_path or ""),
            "blind_index_wide_path": str(self.blind_index_wide_path or ""),
            "blind_index_select_mode": str(self.blind_index_select_mode or "auto"),
            "blind_verify_enabled": bool(self.blind_verify_enabled),
            "blind_verify_top_n": int(self.blind_verify_top_n),
            "blind_verify_match_tol_px": float(self.blind_verify_match_tol_px),
            "blind_verify_match_tol_arcsec": self.blind_verify_match_tol_arcsec,
            "blind_verify_min_matches": int(self.blind_verify_min_matches),
            "blind_verify_min_fraction": float(self.blind_verify_min_fraction),
            "blind_verify_inmemory_catalog": bool(self.blind_verify_inmemory_catalog),
            "verify_mag_limit": float(self.verify_mag_limit),
            "blind_verify_early_accept": int(self.blind_verify_early_accept),
            "blind_verify_early_floor": int(self.blind_verify_early_floor),
            "blind_verify_early_fraction": float(self.blind_verify_early_fraction),
            "blind_img_star_budget": int(self.blind_img_star_budget),
            "blind_img_select_mode": str(self.blind_img_select_mode),
            "blind_use_rig_prior": bool(self.blind_use_rig_prior),
            "debug_platesolver": bool(self.debug_platesolver),
            "vsx_local_db_path": str(self.vsx_local_db_path or ""),
            "vsx_out_of_scope_types": list(self.vsx_out_of_scope_types),
            "exoplanet_local_db_path": str(self.exoplanet_local_db_path or ""),
            "exoplanet_match_max_sep_arcsec": float(self.exoplanet_match_max_sep_arcsec),
            "catalog_query_max_rows": int(self.catalog_query_max_rows),
            "per_frame_mp_reserve_ram_gb": float(self.per_frame_mp_reserve_ram_gb),
            "alignment_max_stars": int(self.alignment_max_stars),
            "alignment_max_control_points": int(self.alignment_max_control_points),
            "alignment_detection_sigma": float(self.alignment_detection_sigma),
            "qc_dao_detection_sigma": float(self.qc_dao_detection_sigma),
            "sips_dao_fwhm_px": float(self.sips_dao_fwhm_px),
            "sips_dao_fwhm_fwhm_factor": self.sips_dao_fwhm_fwhm_factor,
            "sips_dao_threshold_sigma": float(self.sips_dao_threshold_sigma),
            "qc_after_calibrate_enabled": bool(self.qc_after_calibrate_enabled),
            "dao_qc_in_calibrate": bool(self.dao_qc_in_calibrate),
            "qc_max_hfr": float(self.qc_max_hfr),
            "qc_max_hfr_fwhm_ratio": self.qc_max_hfr_fwhm_ratio,
            "qc_min_stars": int(self.qc_min_stars),
            "qc_max_background_rms": (
                float(self.qc_max_background_rms)
                if self.qc_max_background_rms is not None
                else None
            ),
            "auto_fwhm_enabled": bool(self.auto_fwhm_enabled),
            "auto_fwhm_k_factor": float(self.auto_fwhm_k_factor),
            "auto_fwhm_k_min": float(self.auto_fwhm_k_min),
            "auto_fwhm_k_max": float(self.auto_fwhm_k_max),
            "frame_align_residual_gate_enabled": bool(self.frame_align_residual_gate_enabled),
            "frame_align_residual_max_frac": float(self.frame_align_residual_max_frac),
            "frame_align_residual_min_keep_frames": int(self.frame_align_residual_min_keep_frames),
            "aperture_photometry_enabled": bool(self.aperture_photometry_enabled),
            "save_lightcurve_png": bool(self.save_lightcurve_png),
            "phase2a_airmass_before_outlier": bool(self.phase2a_airmass_before_outlier),
            "sysrem_enabled": bool(self.sysrem_enabled),
            "comp_qa_enabled": bool(self.comp_qa_enabled),
            "trust_flag_enabled": bool(self.trust_flag_enabled),
            "lc_quality_min_frames": int(self.lc_quality_min_frames),
            "lc_quality_short_min_frames": int(self.lc_quality_short_min_frames),
            "lc_quality_min_normal_frac": float(self.lc_quality_min_normal_frac),
            "comp_trust_min_comps": int(self.comp_trust_min_comps),
            "check_star_min_epochs": int(self.check_star_min_epochs),
            "sparse_trust_T_green": float(self.sparse_trust_T_green),
            "sparse_trust_T_red": float(self.sparse_trust_T_red),
            "sparse_trust_X2_RED": float(self.sparse_trust_X2_RED),
            "check_select_rms_floor": float(self.check_select_rms_floor),
            "comp_select_rms_floor": float(self.comp_select_rms_floor),
            "sysrem_n_iter": int(self.sysrem_n_iter),
            "observer_name": str(self.observer_name),
            "observer_code": str(self.observer_code),
            "aavso_observer_code": str(self.aavso_observer_code),
            "aavso_filter_map": dict(self.aavso_filter_map),
            "observer_location_id": int(self.observer_location_id),
            "psf_photometry_enabled": bool(self.psf_photometry_enabled),
            "epsf_auto_run": bool(self.epsf_auto_run),
            "psf_spatial_order": int(self.psf_spatial_order),
            "psf_chi2_threshold": float(self.psf_chi2_threshold),
            "psf_ac_policy": str(self.psf_ac_policy),
            "psf_zp_membership": str(self.psf_zp_membership),
            "psf_zp_for_zp_validated_rigs": list(self.psf_zp_for_zp_validated_rigs),
            "psf_grouper_enabled": bool(self.psf_grouper_enabled),
            "psf_group_sep_fwhm": float(self.psf_group_sep_fwhm),
            "psf_neighbor_include_fwhm": float(self.psf_neighbor_include_fwhm),
            "psf_neighbor_sub_enabled": bool(self.psf_neighbor_sub_enabled),
            "neighbor_sub_chi2_max": float(self.neighbor_sub_chi2_max),
            "neighbor_sub_residual_rms_max": float(self.neighbor_sub_residual_rms_max),
            "neighbor_sub_refuse_sep_fwhm": float(self.neighbor_sub_refuse_sep_fwhm),
            "neighbor_sub_centroid_max_fwhm": float(self.neighbor_sub_centroid_max_fwhm),
            "neighbor_sub_nn_contam_dmag": float(self.neighbor_sub_nn_contam_dmag),
            "neighbor_sub_max_neighbor_overmag": float(self.neighbor_sub_max_neighbor_overmag),
            "neighbor_sub_max_target_undermag": float(self.neighbor_sub_max_target_undermag),
            "neighbor_sub_min_recovered_snr": float(self.neighbor_sub_min_recovered_snr),
            "neighbor_sub_regime_dmag_min": float(self.neighbor_sub_regime_dmag_min),
            "neighbor_sub_regime_sep_max": float(self.neighbor_sub_regime_sep_max),
            "psf_spatial_enabled": bool(self.psf_spatial_enabled),
            "psf_spatial_grid": str(self.psf_spatial_grid),
            "psf_spatial_min_stars_per_cell": int(self.psf_spatial_min_stars_per_cell),
            "psf_quality_fallback_enabled": bool(self.psf_quality_fallback_enabled),
            "psf_adaptive_enabled": bool(self.psf_adaptive_enabled),
            "psf_adaptive_resolve_fwhm": float(self.psf_adaptive_resolve_fwhm),
            "psf_adaptive_snr_lo": float(self.psf_adaptive_snr_lo),
            "preprocess_sky_surface_order": int(self.preprocess_sky_surface_order),
            "preprocess_sky_surface_force_reapply": bool(self.preprocess_sky_surface_force_reapply),
            "osc_channel_binning": int(self.osc_channel_binning),
            "qc_fwhm_limit": float(self.qc_fwhm_limit),
            "qc_elong_limit": float(self.qc_elong_limit),
            "epsf_min_stars": int(self.epsf_min_stars),
            "photometry_mode": str(self.photometry_mode),
            "aperture_fwhm_factor": float(self.aperture_fwhm_factor),
            "aperture_policy_mode": str(self.aperture_policy_mode),
            "aperture_snr_sizing": {
                "small": float(self.aperture_snr_sizing.get("small", 1.5)),
                "large": float(self.aperture_snr_sizing.get("large", 4.0)),
            },
            "aperture_variable_factor": float(self.aperture_variable_factor),
            "aperture_comp_factor": float(self.aperture_comp_factor),
            "aperture_correction_enabled": bool(self.aperture_correction_enabled),
            "aperture_correction_min_ref_stars": int(self.aperture_correction_min_ref_stars),
            "aperture_correction_max_contamination": float(self.aperture_correction_max_contamination),
            "aperture_correction_max_scatter_mag": float(self.aperture_correction_max_scatter_mag),
            "cog_aperture_correction_enabled": bool(self.cog_aperture_correction_enabled),
            "cog_ref_fwhm": float(self.cog_ref_fwhm),
            "cog_min_stars": int(self.cog_min_stars),
            "cog_isolation_fwhm": float(self.cog_isolation_fwhm),
            "cog_snr_min": float(self.cog_snr_min),
            "cog_sat_frac": float(self.cog_sat_frac),
            "cog_ladder_step_px": float(self.cog_ladder_step_px),
            "cog_ladder_step_fwhm": self.cog_ladder_step_fwhm,
            "cog_ac_factor_max": float(self.cog_ac_factor_max),
            "per_frame_saturation_enabled": bool(self.per_frame_saturation_enabled),
            "per_frame_sat_min_clean_frac": float(self.per_frame_sat_min_clean_frac),
            "err_empty_apertures_n": int(self.err_empty_apertures_n),
            "err_empty_apertures_min": int(self.err_empty_apertures_min),
            "masterstar_best_of_n": int(self.masterstar_best_of_n),
            "phase01_comparison_max_psf_chi2": float(self.phase01_comparison_max_psf_chi2),
            "phase01_comparison_max_fwhm_factor": float(self.phase01_comparison_max_fwhm_factor),
            "phase01_comparison_isolation_radius_px": float(self.phase01_comparison_isolation_radius_px),
            "phase01_comparison_isolation_radius_arcsec": self.phase01_comparison_isolation_radius_arcsec,
            "annulus_inner_fwhm": float(self.annulus_inner_fwhm),
            "annulus_outer_fwhm": float(self.annulus_outer_fwhm),
            "nonlinearity_peak_percentile": float(self.nonlinearity_peak_percentile),
            "nonlinearity_fwhm_ratio": float(self.nonlinearity_fwhm_ratio),
            "bpm_dark_mad_sigma": float(self.bpm_dark_mad_sigma),
            "masterstar_platesolve_sip_max_order": int(self.masterstar_platesolve_sip_max_order),
            "masterstar_platesolve_sip_min_order": int(self.masterstar_platesolve_sip_min_order),
            "masterstar_dao_threshold_sigma": float(self.masterstar_dao_threshold_sigma),
            "dao_detection_n_equiv": float(self.dao_detection_n_equiv),
            "dao_centroid_max_shift_fwhm": float(self.dao_centroid_max_shift_fwhm),
            "admission_sat_peak_frac": float(self.admission_sat_peak_frac),
            "masterstar_prematch_peak_sigma_floor": float(self.masterstar_prematch_peak_sigma_floor),
            "masterstar_catalog_recovery_min": float(self.masterstar_catalog_recovery_min),
            "masterstar_min_matched_floor": int(self.masterstar_min_matched_floor),
            "masterstar_centre_rms_max_px": float(self.masterstar_centre_rms_max_px),
            "masterstar_centre_rms_max_arcsec": self.masterstar_centre_rms_max_arcsec,
            "masterstar_distortion_benign_ratio_max": float(
                self.masterstar_distortion_benign_ratio_max
            ),
            "masterstar_accept_mode": str(self.masterstar_accept_mode),
            "masterstar_quality_crowded_n_cat_min": int(self.masterstar_quality_crowded_n_cat_min),
            "masterstar_detection_cap_adaptive": bool(self.masterstar_detection_cap_adaptive),
            "masterstar_detection_cap_min": int(self.masterstar_detection_cap_min),
            "masterstar_detection_cap_max": int(self.masterstar_detection_cap_max),
            "masterstar_detection_cap_k": float(self.masterstar_detection_cap_k),
            "masterstar_sibling_recovery_enabled": bool(self.masterstar_sibling_recovery_enabled),
            "masterstar_sibling_min_matched": int(self.masterstar_sibling_min_matched),
            "masterstar_sibling_rms_max_px": float(self.masterstar_sibling_rms_max_px),
            "masterstar_sibling_rms_max_arcsec": self.masterstar_sibling_rms_max_arcsec,
            "masterstar_sibling_min_quadrants": int(self.masterstar_sibling_min_quadrants),
            "masterstar_sibling_stack_n": int(self.masterstar_sibling_stack_n),
            "phase01_comparison_max_dist_deg": float(self.phase01_comparison_max_dist_deg),
            "phase01_comparison_max_dist_fov_frac": self.phase01_comparison_max_dist_fov_frac,
            "phase01_comparison_max_mag_diff": float(self.phase01_comparison_max_mag_diff),
            "phase01_comparison_mag_bright_threshold": float(self.phase01_comparison_mag_bright_threshold),
            "phase01_comparison_max_mag_diff_bright_floor": float(
                self.phase01_comparison_max_mag_diff_bright_floor
            ),
            "phase01_comparison_max_mag_diff_absolute": float(
                self.phase01_comparison_max_mag_diff_absolute
            ),
            "comp_max_delta_bprp": float(self.comp_max_delta_bprp),
            "comp_color_tiers": [
                {"bprp": float(t.get("bprp", 0.0)), "w": float(t.get("w", 0.0))}
                for t in self.comp_color_tiers
            ],
            "phase01_tiers": [float(x) for x in self.phase01_tiers],
            "comp_contamination_penalty_k": float(self.comp_contamination_penalty_k),
            "phase01_comparison_n_comp_min": int(self.phase01_comparison_n_comp_min),
            "phase01_comparison_n_comp_max": int(self.phase01_comparison_n_comp_max),
            "phase01_comparison_max_comp_rms": float(self.phase01_comparison_max_comp_rms),
            "comp_rms_loo_photon_k": float(self.comp_rms_loo_photon_k),
            "phase01_comparison_min_dist_arcsec": float(self.phase01_comparison_min_dist_arcsec),
            "phase01_comparison_min_frames_frac": float(self.phase01_comparison_min_frames_frac),
            "phase01_comparison_exclude_gaia_nss": bool(self.phase01_comparison_exclude_gaia_nss),
            "phase01_comparison_exclude_gaia_extobj": bool(self.phase01_comparison_exclude_gaia_extobj),
            "phase01_use_bprp_primary": bool(self.phase01_use_bprp_primary),
            "phase01_ct_min_comp": int(self.phase01_ct_min_comp),
            "apply_color_term": str(self.apply_color_term),
            "color_level_k_mag_per_bprp": self.color_level_k_mag_per_bprp,
            "color_level_k_stderr_mag_per_bprp": self.color_level_k_stderr_mag_per_bprp,
            "snr_cog_isolation_fwhm": float(self.snr_cog_isolation_fwhm),
            "k2_mode": str(self.k2_mode),
            "k2_defaults_bprp": dict(self.k2_defaults_bprp),
            "sigma_sys_mag": dict(self.sigma_sys_mag),
            "gain_container_scale": float(self.gain_container_scale),
            "photon_transfer_ci_max_width_factor": float(self.photon_transfer_ci_max_width_factor),
            "k2_ceiling": float(self.k2_ceiling),
            "k2_fit_enabled": bool(self.k2_fit_enabled),
            "k2_fit_min_detectability": float(self.k2_fit_min_detectability),
            "k2_fit_consistency_sigma": float(self.k2_fit_consistency_sigma),
            "k2_fit_lit_factor": float(self.k2_fit_lit_factor),
            "phase01_ct_extrapolation_tol": float(self.phase01_ct_extrapolation_tol),
            "phase01_flux_col": str(self.phase01_flux_col),
            "temporal_binning_enabled": bool(self.temporal_binning_enabled),
            "temporal_bin_window": int(self.temporal_bin_window),
            "savgol_detrend_enabled": bool(self.savgol_detrend_enabled),
            "savgol_window_frac": float(self.savgol_window_frac),
            "savgol_polyorder": int(self.savgol_polyorder),
            "democratic_detrend_enabled": bool(self.democratic_detrend_enabled),
            "democratic_sg_window_frac": float(self.democratic_sg_window_frac),
            "pytics_enabled": bool(self.pytics_enabled),
            "pytics_n_iter": int(self.pytics_n_iter),
            "comp_max_slope_mmag_hr": float(self.comp_max_slope_mmag_hr),
            "comp_slope_significance_k": float(self.comp_slope_significance_k),
            "comp_sparse_fallback_enabled": bool(self.comp_sparse_fallback_enabled),
            "comp_sparse_fallback_min": int(self.comp_sparse_fallback_min),
            "comp_iterative_clip_enabled": bool(self.comp_sparse_fallback_enabled),
            "gs11_dilution_enabled": bool(self.gs11_dilution_enabled),
            "gs11_dilution_aperture_arcsec": float(self.gs11_dilution_aperture_arcsec),
            "gs11_dilution_mag_limit_delta": float(self.gs11_dilution_mag_limit_delta),
            "gs11_comp_max_dilution": float(self.gs11_comp_max_dilution),
            "gs11_comp_suspect_dilution": float(self.gs11_comp_suspect_dilution),
            "gs11_target_min_dilution": float(self.gs11_target_min_dilution),
            "phase01_chip_interior_margin_px": int(self.phase01_chip_interior_margin_px),
            "phase01_chip_interior_margin_arcsec": self.phase01_chip_interior_margin_arcsec,
            "variability_min_frames": int(self.variability_min_frames),
            "variability_min_frames_frac": float(self.variability_min_frames_frac),
            "variability_p85_filter": int(self.variability_p85_filter),
            "variability_slope_floor": float(self.variability_slope_floor),
            "variability_sigma_threshold": float(self.variability_sigma_threshold),
            "variability_comp_floor_factor": float(self.variability_comp_floor_factor),
            "variability_smoothness_max": float(self.variability_smoothness_max),
            "variability_mag_limit": float(self.variability_mag_limit),
            "variability_min_rms_pct": float(self.variability_min_rms_pct),
            "variability_min_amplitude_mag": float(self.variability_min_amplitude_mag),
            "variability_clip_ratio_min": float(self.variability_clip_ratio_min),
            "variability_vdi_z_threshold": float(self.variability_vdi_z_threshold),
            "variability_min_points_rms": int(self.variability_min_points_rms),
            "tess_enabled": bool(self.tess_enabled),
            "field_density_sparse_threshold": float(self.field_density_sparse_threshold),
            "field_density_dense_threshold": float(self.field_density_dense_threshold),
            "field_density_adaptive_enabled": bool(self.field_density_adaptive_enabled),
            "crowding_classifier_enabled": bool(self.crowding_classifier_enabled),
            "crowding_blend_tighten_threshold": float(self.crowding_blend_tighten_threshold),
            "crowding_comp_availability_loosen_count": float(self.crowding_comp_availability_loosen_count),
            "crowding_tighten_min_fwhm_px": float(self.crowding_tighten_min_fwhm_px),
            "comp_pool_derived_admission": bool(self.comp_pool_derived_admission),
            "comparison_stars_pool_n": int(self.comparison_stars_pool_n),
            "comp_weight_c_col_mag_per_bprp": self.comp_weight_c_col_mag_per_bprp,
            "comp_weight_c_dist_mag_per_deg": self.comp_weight_c_dist_mag_per_deg,
            "comp_weight_airmass_span": float(self.comp_weight_airmass_span),
            "comp_weight_optics_kind": str(self.comp_weight_optics_kind),
            "forced_photometry_enabled": bool(self.forced_photometry_enabled),
            "forced_photometry_centroid_bound_fwhm": float(self.forced_photometry_centroid_bound_fwhm),
            "forced_photometry_margin_px": float(self.forced_photometry_margin_px),
        }

    # Backward-compatible alias (some callers expect to_dict()).
    def to_dict(self) -> dict[str, Any]:
        return self.to_json()

    def ensure_base_dirs(self) -> None:
        """Create base directories required by file-first workflow."""
        self.archive_root.mkdir(parents=True, exist_ok=True)
        self.calibration_library_root.mkdir(parents=True, exist_ok=True)


# Delta overrides od baseline (normal / JSON). ``phase01_comparison_max_dist_deg`` sa v runtime pricita k FOV-vysledku - viz ``apply_density_overrides``.
DENSITY_OVERRIDES: dict[str, dict[str, float | int]] = {
    "sparse": {
        "phase01_comparison_max_mag_diff": +0.5,
        "phase01_comparison_n_comp_min": -1,
        "comp_max_delta_bprp": +0.20,
        "phase01_comparison_max_dist_deg": +0.3,
    },
    "normal": {},
    "dense": {
        # NOTE: aperture_fwhm_factor is production f in APERTURE-01 (r = f x FWHM).
        # The SNR mag-bin table is a diagnostic artifact only.
        "annulus_inner_fwhm": +1.0,
        "phase01_comparison_min_dist_arcsec": +30.0,
        "comp_max_delta_bprp": -0.15,
        "phase01_comparison_max_comp_rms": -0.02,
    },
}


# [CROWDING-CLASSIFIER] Decoupled overrides for the signal-based classifier
# (``AppConfig.crowding_classifier_enabled``). Each set keys on the PHYSICALLY correct
# signal instead of the conflated stars/Mpx class:
#   LOOSEN  -> low comp AVAILABILITY (few usable catalog comps in the FOV).
#   TIGHTEN -> real contamination (blend_frac @ measured depth).
# Both may fire independently; shared keys (comp_max_delta_bprp) sum additively.
CROWDING_LOOSEN_OVERRIDES: dict[str, float | int] = {
    "phase01_comparison_max_mag_diff": +0.5,   # suppressed when catalog-bottlenecked
    "phase01_comparison_n_comp_min": -1,
    "comp_max_delta_bprp": +0.20,
    "phase01_comparison_max_dist_deg": +0.3,   # additive to FOV result (handled by caller)
}
CROWDING_TIGHTEN_OVERRIDES: dict[str, float | int] = {
    "phase01_comparison_min_dist_arcsec": +30.0,
    "comp_max_delta_bprp": -0.15,
    "phase01_comparison_max_comp_rms": -0.02,
    "annulus_inner_fwhm": +1.0,
}


def compute_field_density(n_masterstar_stars: int, chip_w_px: int, chip_h_px: int) -> float:
    """Vrati hustotu pola v hviezd/Mpx."""
    mpx = (float(chip_w_px) * float(chip_h_px)) / 1_000_000.0
    if mpx <= 0 or n_masterstar_stars < 0:
        return 0.0
    return float(n_masterstar_stars) / float(mpx)


def classify_field_density(density: float, sparse_th: float, dense_th: float) -> str:
    """Vrati ``sparse`` / ``normal`` / ``dense``."""
    if density < float(sparse_th):
        return "sparse"
    if density <= float(dense_th):
        return "normal"
    return "dense"


def apply_density_overrides(cfg: AppConfig, density_class: str) -> AppConfig:
    """Vrati kopiu ``cfg`` s aplikovanymi density override deltami."""
    cfg_eff = copy.copy(cfg)
    overrides = DENSITY_OVERRIDES.get(density_class, {})
    for param, delta in overrides.items():
        if param == "phase01_comparison_max_dist_deg":
            # Do cfg neukladame - v ``run_phase0_and_phase1`` sa pricita k efektivnemu ``max_dist_deg`` (FOV kluc).
            continue
        if not hasattr(cfg_eff, param):
            continue
        cur = getattr(cfg_eff, param)
        if isinstance(cur, bool):
            continue
        if isinstance(cur, int) and not isinstance(cur, bool):
            try:
                new_val = int(cur) + int(delta)
            except (TypeError, ValueError):
                continue
            if param == "phase01_comparison_n_comp_min":
                new_val = max(2, new_val)
            setattr(cfg_eff, param, new_val)
            logging.debug("[DENSITY OVERRIDE] %s: %s -> %s (delta=%+s)", param, cur, new_val, delta)
            continue
        try:
            new_val = float(cur) + float(delta)
        except (TypeError, ValueError):
            continue
        if param == "aperture_fwhm_factor":
            new_val = max(0.25, min(6.0, float(new_val)))
        elif param in {"annulus_inner_fwhm", "annulus_outer_fwhm"}:
            new_val = max(1.0, min(30.0, float(new_val)))
        elif param == "phase01_comparison_max_comp_rms":
            new_val = max(0.01, min(0.5, float(new_val)))
        elif param == "comp_max_delta_bprp":
            new_val = max(0.0, min(5.0, float(new_val)))
        elif param == "phase01_comparison_min_dist_arcsec":
            new_val = max(0.0, min(600.0, float(new_val)))
        setattr(cfg_eff, param, new_val)
        logging.debug("[DENSITY OVERRIDE] %s: %s -> %s (delta=%+s)", param, cur, new_val, delta)

    # Zachovaj annulus_outer > inner + 1
    try:
        inn = float(getattr(cfg_eff, "annulus_inner_fwhm", 0.0) or 0.0)
        out = float(getattr(cfg_eff, "annulus_outer_fwhm", 0.0) or 0.0)
        if math.isfinite(inn) and math.isfinite(out) and out <= inn:
            cfg_eff.annulus_outer_fwhm = float(inn + 1.0)
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).warning(
            "[CONFIG] Annulus config clamp failed, using default: %s", exc
        )
    return cfg_eff


def apply_crowding_overrides(
    cfg: AppConfig,
    *,
    loosen: bool,
    tighten: bool,
    suppress_mag_loosen: bool = False,
) -> tuple[AppConfig, float]:
    """Additive, decoupled comp overrides for the crowding_index classifier.

    ``loosen`` (low comp availability) and ``tighten`` (high blend_frac) are independent
    and may both apply; shared keys (``comp_max_delta_bprp``) sum rather than clobber.
    ``suppress_mag_loosen`` drops the ``max_mag_diff`` loosening when the field is
    catalog-bottlenecked (fainter comps simply don't exist in the catalog, so loosening
    the magnitude tolerance cannot help and only risks worse comps).

    Returns ``(cfg_eff, max_dist_deg_delta)``; the caller adds ``max_dist_deg_delta`` to
    the FOV-derived ``max_dist_deg`` (mirrors :func:`apply_density_overrides`).
    """
    cfg_eff = copy.copy(cfg)
    deltas: dict[str, float] = {}
    max_dist_delta = 0.0
    if loosen:
        for k, v in CROWDING_LOOSEN_OVERRIDES.items():
            if k == "phase01_comparison_max_dist_deg":
                max_dist_delta += float(v)
                continue
            if k == "phase01_comparison_max_mag_diff" and suppress_mag_loosen:
                continue
            deltas[k] = deltas.get(k, 0.0) + float(v)
    if tighten:
        for k, v in CROWDING_TIGHTEN_OVERRIDES.items():
            deltas[k] = deltas.get(k, 0.0) + float(v)

    for param, delta in deltas.items():
        if not hasattr(cfg_eff, param):
            continue
        cur = getattr(cfg_eff, param)
        if isinstance(cur, bool):
            continue
        if isinstance(cur, int) and not isinstance(cur, bool):
            try:
                new_val_i = int(cur) + int(round(float(delta)))
            except (TypeError, ValueError):
                continue
            if param == "phase01_comparison_n_comp_min":
                new_val_i = max(2, new_val_i)
            setattr(cfg_eff, param, new_val_i)
            logging.debug("[CROWDING OVERRIDE] %s: %s -> %s (delta=%+s)", param, cur, new_val_i, delta)
            continue
        try:
            new_val = float(cur) + float(delta)
        except (TypeError, ValueError):
            continue
        if param in {"annulus_inner_fwhm", "annulus_outer_fwhm"}:
            new_val = max(1.0, min(30.0, new_val))
        elif param == "phase01_comparison_max_comp_rms":
            new_val = max(0.01, min(0.5, new_val))
        elif param == "comp_max_delta_bprp":
            new_val = max(0.0, min(5.0, new_val))
        elif param == "phase01_comparison_min_dist_arcsec":
            new_val = max(0.0, min(600.0, new_val))
        elif param == "phase01_comparison_max_mag_diff":
            new_val = max(0.05, min(5.0, new_val))
        setattr(cfg_eff, param, new_val)
        logging.debug("[CROWDING OVERRIDE] %s: %s -> %s (delta=%+s)", param, cur, new_val, delta)

    try:
        inn = float(getattr(cfg_eff, "annulus_inner_fwhm", 0.0) or 0.0)
        out = float(getattr(cfg_eff, "annulus_outer_fwhm", 0.0) or 0.0)
        if math.isfinite(inn) and math.isfinite(out) and out <= inn:
            cfg_eff.annulus_outer_fwhm = float(inn + 1.0)
    except Exception:  # noqa: BLE001
        # EXC-0048: T4 -- optional enrichment skipped (if math.isfinite(inn) and math.isfinite(out) and out <= in... (EXCEPT-BULK 2026-07-08)
        pass
    return cfg_eff, float(max_dist_delta)

