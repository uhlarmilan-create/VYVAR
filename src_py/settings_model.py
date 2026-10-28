"""Settings page model (SETTINGS-2026-10): groups, widgets, value parsing, validation, catalog status.

Streamlit-free so it is unit-tested; ``ui_settings.py`` renders it. One page shows every key of
config.json once, in the config.json groups and order; the registry
(``dev/validation/params_registry.json``) supplies label, help, unit, range and tier.
"""
from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import config
import params_registry as pr

#: config.json group order (same as ``config._CONFIG_SECTION_ORDER``) and titles in the UI
GROUP_TITLES: dict[str, str] = {
    "observer": "Observer",
    "paths": "Paths and catalogs",
    "calibration": "Calibration",
    "qc": "Quality (QC)",
    "alignment": "Alignment",
    "detection": "Detection and astrometry",
    "photometry": "Photometry",
    "comp_selection": "Comparison stars",
    "trust": "Trust",
    "extinction": "Extinction",
    "reports": "Reports",
    "export": "Export",
}
TIER_ORDER = {"basic": 0, "advanced": 1, "expert": 2}

#: keys whose widget is a small table
TABLE_KEYS = frozenset({"comp_color_tiers", "aavso_filter_map", "k2_defaults_bprp"})
#: list[str] keys edited as comma-separated text
TAGS_KEYS = frozenset({"vsx_out_of_scope_types"})


def group_order() -> list[str]:
    """Groups in config.json order, Paths and catalogs second (it is set once after installing)."""
    order = [g for g in config._CONFIG_SECTION_ORDER if g != "paths"]
    return [order[0], "paths", *order[1:]]


def settings_keys(registry: dict[str, Any], config_keys: list[str] | set[str]) -> dict[str, list[str]]:
    """Group -> keys shown on the Settings page (every config.json key once, config.json order:
    tier basic -> advanced -> expert, then alphabetical)."""
    out: dict[str, list[str]] = {g: [] for g in group_order()}
    for key in config_keys:
        entry = registry.get(key)
        if not entry or entry.get("widget") == "hidden":
            continue
        out.setdefault(entry["phase"], []).append(key)
    for g, keys in out.items():
        keys.sort(key=lambda k: (TIER_ORDER.get(registry[k].get("tier"), 1), k))
    return out


def widget_kind(key: str, value: Any, entry: dict[str, Any]) -> str:
    """checkbox | select | int | float | text | tags | table"""
    if key in TABLE_KEYS:
        return "table"
    if key in TAGS_KEYS:
        return "tags"
    if isinstance(value, bool):
        return "checkbox"
    if key in pr.LITERAL_OPTIONS:
        return "select"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    return "text"


def select_options(key: str, value: Any) -> list[str]:
    opts = list(pr.LITERAL_OPTIONS.get(key, ()))
    if str(value) not in opts:
        opts = [str(value), *opts]
    return opts


# ---------------------------------------------------------------------------
# table widgets: rows <-> config value
# ---------------------------------------------------------------------------
def table_rows(key: str, value: Any) -> list[dict[str, Any]]:
    if key == "comp_color_tiers":
        return [{"class": i + 1, "max |dBP-RP| (mag)": float(t.get("bprp", 0.0))} for i, t in enumerate(value or [])]
    if key == "aavso_filter_map":
        return [{"your filter": k, "AAVSO code": v} for k, v in (value or {}).items()]
    if key == "k2_defaults_bprp":
        return [{"band": k, "k2 (mag/airmass per mag BP-RP)": v} for k, v in (value or {}).items()]
    raise KeyError(key)


def table_value(key: str, rows: list[dict[str, Any]]) -> tuple[Any, str | None]:
    """Config value from edited rows, or (None, error)."""
    def cells(r: dict[str, Any]) -> list[Any]:
        return list(r.values())

    if key == "comp_color_tiers":
        lim = []
        for r in rows:
            try:
                v = float(cells(r)[-1])
            except (TypeError, ValueError, IndexError):
                return None, "every colour class needs a number"
            lim.append(v)
        if len(lim) != 3:
            return None, "exactly 3 colour classes are needed"
        if not (lim[0] > 0.0 and all(a < b for a, b in zip(lim, lim[1:]))):
            return None, "colour class limits must be positive and increasing"
        return [{"bprp": v} for v in lim], None
    out: dict[str, Any] = {}
    for r in rows:
        c = cells(r)
        if len(c) < 2 or c[0] is None or str(c[0]).strip() == "":
            continue
        k = str(c[0]).strip()
        if key == "aavso_filter_map":
            v = str(c[1] or "").strip()
            if not v:
                return None, f"filter {k}: AAVSO code is empty"
            out[k.upper()] = v
        else:
            try:
                fv = float(c[1])
            except (TypeError, ValueError):
                return None, f"band {k}: k2 must be a number"
            if not math.isfinite(fv):
                return None, f"band {k}: k2 must be a number"
            out[k] = fv
    return out, None


# ---------------------------------------------------------------------------
# scalar widgets
# ---------------------------------------------------------------------------
def tags_text(value: Any) -> str:
    return ", ".join(str(v) for v in (value or []))


def tags_value(text: str) -> list[str]:
    return [p.strip() for p in str(text or "").split(",") if p.strip()]


def check_value(key: str, value: Any, entry: dict[str, Any]) -> str | None:
    """Range check of one value (None = ok)."""
    rng = entry.get("range")
    if rng and isinstance(value, (int, float)) and not isinstance(value, bool):
        if not (float(rng[0]) <= float(value) <= float(rng[1])):
            return f"must be within {rng[0]} .. {rng[1]}"
    if key == "epsf_targets" and not str(value or "").strip():
        return "empty: use ALL or a list"
    return None


def cross_check(values: dict[str, Any]) -> list[str]:
    """Rules between keys (Save is refused while one fails); the one list in ``config``."""
    return config.config_rule_problems(values)


def changed_keys(current: dict[str, Any], edited: dict[str, Any]) -> list[str]:
    return sorted(k for k, v in edited.items() if k in current and pr.values_differ(v, current[k]))


#: machine paths: their default is "next to the install", never a "changed" setting
PATH_KEYS = frozenset({"archive_root", "calibration_library_root", "database_path", "gaia_db_path",
                       "objects_db_path", "astrometry_index_dir"})


def differs_from_default(key: str, value: Any, defaults: dict[str, Any]) -> bool:
    if key in PATH_KEYS or key not in defaults:
        return False
    d = defaults[key]
    if isinstance(d, Path) or isinstance(value, Path):
        return False
    if key == "comp_color_tiers":
        return [float(t.get("bprp", 0)) for t in value or []] != [float(t.get("bprp", 0)) for t in d or []]
    return pr.values_differ(value, d)


def format_default(key: str, value: Any, unit: str | None) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    if key == "comp_color_tiers":
        return ", ".join(f"{float(t.get('bprp', 0)):g}" for t in value or [])
    if isinstance(value, (dict, list)):
        return json.dumps(value) if value else "empty"
    if isinstance(value, str):
        return value or "empty"
    s = f"{value:g}" if isinstance(value, float) else str(value)
    return f"{s} {unit}" if unit and unit not in ("fraction", "probability") else s


# ---------------------------------------------------------------------------
# catalog status (Paths and catalogs)
# ---------------------------------------------------------------------------
@dataclass
class CatalogStatus:
    name: str
    path: str
    state: str  # ready | missing | invalid | old
    facts: list[tuple[str, str]]
    note: str = ""


def _size_txt(n: int) -> str:
    for unit, f in (("GB", 1 << 30), ("MB", 1 << 20), ("kB", 1 << 10)):
        if n >= f:
            return f"{n / f:.1f} {unit}"
    return f"{n} B"


def _age_days(iso: str | None) -> float | None:
    if not iso:
        return None
    try:
        t = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
    except ValueError:
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - t).total_seconds() / 86400.0


#: objects tables older than this are marked "old" (VSX and the NASA archive change every week)
OBJECTS_OLD_DAYS = 90


def gaia_status(path: str | Path) -> CatalogStatus:
    """Gaia DR3 card from ``catalog_meta`` and ``gaia_coverage`` (no full-table count)."""
    from catalog_db import GAIA_INDEXES, connect_ro, read_catalog_meta, resolve_catalog_path, validate_gaia_db_schema  # noqa: PLC0415

    p = resolve_catalog_path(path) if str(path or "").strip() else Path()
    if not str(path or "").strip() or not p.is_file():
        return CatalogStatus("Gaia DR3", str(path or ""), "missing", [], "Build it with catalogs/build_gaia_catalog.py.")
    ok, code = validate_gaia_db_schema(p)
    if not ok:
        return CatalogStatus("Gaia DR3", str(p), "invalid", [], code)
    rows: list[Any] = []
    indexes: set[str] = set()
    n_strips = 0
    try:
        con = connect_ro(p)
        try:
            if con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='strip_progress'").fetchone():
                n_strips = int(con.execute("SELECT COUNT(*) FROM strip_progress").fetchone()[0])
            has = con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='gaia_coverage'").fetchone()
            rows = con.execute(
                "SELECT ra_min, ra_max, dec_min, dec_max, g_max, source, n_stars, created_utc FROM gaia_coverage"
            ).fetchall() if has else []
            indexes = {str(r[0]) for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='gaia_dr3' AND sql IS NOT NULL")}
        finally:
            con.close()
    except sqlite3.Error:
        pass
    meta = read_catalog_meta(p).get("gaia_dr3") or {}
    base = [r for r in rows if str(r[5]) != "gaia_tap_topup"]
    tops = [r for r in rows if str(r[5]) == "gaia_tap_topup"]
    facts: list[tuple[str, str]] = []
    n_stars = meta.get("row_count") if meta.get("row_count") is not None else (base[0][6] if base else None)
    if n_stars:
        facts.append(("Stars", f"{int(n_stars) / 1e6:.1f} million"))
    if base:
        b = base[0]
        facts.append(("Depth", f"G <= {float(b[4]):.1f}"))
        full = float(b[2]) <= -89.9 and float(b[3]) >= 89.9 and float(b[1]) - float(b[0]) >= 359.9
        facts.append(("Sky", "full sky" if full else f"Dec {float(b[2]):.0f} .. {float(b[3]):.0f}"))
    facts.append(("Online top-ups", f"{len(tops)} fields"))
    facts.append(("Size", _size_txt(p.stat().st_size)))
    built = meta.get("built_utc") or (base[0][7] if base and base[0][7] else
                                      datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat())
    facts.append(("Built", str(built)[:10]))
    if not base and n_strips:
        # the builder records the coverage only after the last strip (gaia_coverage)
        facts.insert(0, ("Strips done", str(n_strips)))
        return CatalogStatus("Gaia DR3", str(p), "unfinished", facts,
                             "The build is not finished: resume it (Catalog builder below).")
    notes = []
    if not meta:
        notes.append("No catalog record: Register it (Catalog builder below).")
    extra = sorted(indexes - set(GAIA_INDEXES))
    missing = sorted(set(GAIA_INDEXES) - indexes)
    if missing:
        notes.append(f"Missing index {', '.join(missing)}: sky queries are slow (rerun catalogs/build_gaia_catalog.py).")
    if extra:
        notes.append(f"Unused index {', '.join(extra)}: Drop unused indexes (Catalog builder below).")
    return CatalogStatus("Gaia DR3", str(p), "ready", facts, " ".join(notes))


def objects_status(path: str | Path) -> list[CatalogStatus]:
    """VSX and exoplanet cards from ``catalog_meta`` of the objects DB."""
    from catalog_db import read_catalog_meta, resolve_catalog_path, validate_exoplanet_schema, validate_vsx_schema  # noqa: PLC0415

    p = resolve_catalog_path(path) if str(path or "").strip() else Path()
    names = (("vsx_data", "VSX variable stars", validate_vsx_schema), ("exoplanet_data", "Exoplanet hosts", validate_exoplanet_schema))
    if not str(path or "").strip() or not p.is_file():
        return [CatalogStatus(n, str(path or ""), "missing", [], "Build it in the Catalog builder below.") for _, n, _ in names]
    meta = read_catalog_meta(p)
    size = _size_txt(p.stat().st_size)
    out = []
    for table, name, check in names:
        ok, code = check(p)
        if not ok:
            out.append(CatalogStatus(name, str(p), "invalid", [], code))
            continue
        m = meta.get(table) or {}
        facts = [("Objects", f"{int(m['row_count']):,}".replace(",", " ") if m.get("row_count") is not None else "?")]
        params = m.get("params") or {}
        if table == "vsx_data" and params.get("mag_limit_max") is not None:
            facts.append(("Cut", f"max mag < {params['mag_limit_max']}"))
        if table == "exoplanet_data" and "confirmed_only" in params:
            facts.append(("Content", "confirmed only" if params["confirmed_only"] else "confirmed + TOI"))
        upd = params.get("update") or {}
        if upd:
            facts.append(("Last update", f"+{upd.get('added', 0)} / -{upd.get('removed', 0)} / "
                                         f"{upd.get('changed', 0)} changed"))
        facts.append(("File size", size))
        facts.append(("Built", str(m.get("built_utc") or "?")[:10]))
        age = _age_days(m.get("built_utc"))
        state = "old" if age is not None and age > OBJECTS_OLD_DAYS else "ready"
        note = str(m.get("source") or "")
        if age is not None:
            note = f"{note} - {age:.0f} days old".strip(" -")
        out.append(CatalogStatus(name, str(p), state, facts, note))
    return out


def blind_index_status(index_dir: str | Path | None, gaia_db: str | Path | None) -> CatalogStatus:
    """Blind plate-solve index card: bands present, their pattern sizes and the folder size."""
    from astro_index import N_BANDS  # noqa: PLC0415
    from astrosolve import default_index_dir  # noqa: PLC0415
    from catalog_db import resolve_catalog_path  # noqa: PLC0415

    gdb = str(resolve_catalog_path(gaia_db)) if str(gaia_db or "").strip() else None
    idx = str(resolve_catalog_path(index_dir)) if str(index_dir or "").strip() else None
    d = default_index_dir(gdb, idx)
    p = Path(d) if d else None
    if p is None or not p.is_dir():
        return CatalogStatus("Blind index", str(p or ""), "missing", [],
                             "Optional: only frames without a position in the header need it (Catalog builder below).")
    bands = sorted(b for b in p.glob("band_*") if (b / "keys.npy").is_file())
    size = sum(f.stat().st_size for b in bands for f in b.iterdir() if f.is_file())
    sizes = []
    for b in bands:
        try:
            k = int(b.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        hi = 0.15 * 2 ** k
        sizes.append(f"{hi / 2:g}-{hi:g} deg")
    facts = [("Scales", f"{len(bands)} of {N_BANDS}"), ("Size", _size_txt(size))]
    if sizes:
        facts.append(("Pattern sizes", f"{sizes[0].split('-')[0]} .. {sizes[-1].split('-')[1]}"))
    state = "ready" if len(bands) >= N_BANDS else ("unfinished" if bands else "missing")
    return CatalogStatus("Blind index", str(p), state, facts,
                         "Used only when a frame has no position in its header; hinted solving needs only Gaia.")


def folder_status(path: str | Path, *, is_file: bool = False) -> str:
    p = Path(str(path or ""))
    if not str(path or "").strip():
        return "not set"
    if is_file:
        return f"ok - {_size_txt(p.stat().st_size)}" if p.is_file() else "missing"
    return "exists" if p.is_dir() else "will be created"


# ---------------------------------------------------------------------------
# MASTERSTAR suggestion (moved from the DAO-STARS tab; fills the Settings form, Save writes it)
# ---------------------------------------------------------------------------
def masterstar_detection_count(fits_path: Path) -> int | None:
    """Stars of the MASTERSTAR: rows of its masterstars CSV, else the VY_NSTAR header keys."""
    from platesolve_ui_paths import masterstars_csv_in_dir  # noqa: PLC0415

    csv_p = masterstars_csv_in_dir(fits_path.parent)
    if csv_p is not None and csv_p.is_file():
        try:
            import pandas as pd  # noqa: PLC0415

            n = len(pd.read_csv(csv_p, low_memory=False, usecols=[0]))
            if n > 0:
                return int(n)
        except (OSError, ValueError):
            pass
    try:
        from astropy.io import fits  # noqa: PLC0415

        hdr = fits.getheader(str(fits_path))
    except OSError:
        return None
    for key in ("VY_NSTAR", "VY_NDET"):
        try:
            v = int(float(hdr[key]))
        except (KeyError, TypeError, ValueError):
            continue
        if v > 0:
            return v
    return None


def masterstar_suggestions(fwhm_px: float, n_stars: int | None) -> dict[str, float]:
    """Rule of thumb of the former DAO-STARS tab: aperture 2.5 px / FWHM within 1.5-2.0 FWHM;
    detection 2.0 / 2.5 / 3.0 sigma for >800 / >400 / fewer stars."""
    fwhm = float(fwhm_px) if math.isfinite(float(fwhm_px)) and float(fwhm_px) > 0 else 2.5
    n = int(n_stars) if n_stars is not None and int(n_stars) > 0 else 500
    dao = 2.0 if n > 800 else (2.5 if n > 400 else 3.0)
    return {
        "aperture_fwhm_factor": round(float(min(2.0, max(1.5, 2.5 / fwhm))), 2),
        "detection_threshold_sigma": dao,
    }
