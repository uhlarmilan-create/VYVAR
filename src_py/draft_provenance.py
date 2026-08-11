"""Draft run provenance - calibration mode and manifest I/O."""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from importer import SmartImportPlan

CALIBRATION_MODE_VYVAR = "vyvar_calibrated"
CALIBRATION_MODE_PRE = "pre_calibrated"
MANIFEST_SCHEMA_VERSION = 3

_MANIFEST_NAME = "draft_manifest.json"

# Phase 2.1 shadow-observe + Phase 2.2 manifest-first rig-id reads.
MANIFEST_SHADOW_MISMATCHES = 0
MANIFEST_SHADOW_ABSENT = 0
MANIFEST_SHADOW_EQUAL = 0
MANIFEST_FALLBACK = 0

_RIG_MANIFEST_TO_DB: dict[str, str] = {
    "equipment_id": "ID_EQUIPMENTS",
    "telescope_id": "ID_TELESCOPE",
    "location_id": "ID_LOCATION",
    "scanning_id": "ID_SCANNING",
}

_MANIFEST_SHADOW_LOAD_CACHE: dict[tuple[int, str, int], dict[str, Any]] = {}


def reset_manifest_shadow_counters() -> None:
    """Reset Phase 2.1/2.2 manifest rig-id counters (tests / batch reports)."""
    global MANIFEST_SHADOW_MISMATCHES, MANIFEST_SHADOW_ABSENT, MANIFEST_SHADOW_EQUAL, MANIFEST_FALLBACK
    MANIFEST_SHADOW_MISMATCHES = 0
    MANIFEST_SHADOW_ABSENT = 0
    MANIFEST_SHADOW_EQUAL = 0
    MANIFEST_FALLBACK = 0


def clear_manifest_shadow_load_cache() -> None:
    """Clear per-draft manifest load cache (tests)."""
    _MANIFEST_SHADOW_LOAD_CACHE.clear()


def manifest_shadow_counter_snapshot() -> dict[str, int]:
    return {
        "mismatch": int(MANIFEST_SHADOW_MISMATCHES),
        "absent": int(MANIFEST_SHADOW_ABSENT),
        "equal": int(MANIFEST_SHADOW_EQUAL),
        "fallback": int(MANIFEST_FALLBACK),
    }


def _normalize_rig_id(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _load_manifest_for_shadow(draft_id: int, archive_root: Path) -> dict[str, Any]:
    manifest_path = archive_root / _MANIFEST_NAME
    try:
        mtime_ns = manifest_path.stat().st_mtime_ns if manifest_path.is_file() else -1
    except OSError:
        mtime_ns = -1
    cache_key = (int(draft_id), str(archive_root.resolve()), int(mtime_ns))
    cached = _MANIFEST_SHADOW_LOAD_CACHE.get(cache_key)
    if cached is not None:
        return cached
    loaded = load_draft_manifest(archive_root)
    _MANIFEST_SHADOW_LOAD_CACHE[cache_key] = loaded
    return loaded


def _resolve_archive_root_for_shadow(
    draft_id: int,
    draft_row: dict[str, Any] | None,
    db: Any | None,
) -> Path | None:
    if draft_row is not None:
        root = resolve_draft_archive_root_from_row(draft_row)
        if root is not None:
            return root
    if db is not None and hasattr(db, "conn"):
        try:
            row = db.conn.execute(
                "SELECT ARCHIVE_PATH, LIGHTS_PATH FROM OBS_DRAFT WHERE ID = ?;",
                (int(draft_id),),
            ).fetchone()
            if row is not None:
                return resolve_draft_archive_root_from_row(dict(row))
        except Exception:  # noqa: BLE001
            return None
    return None


def observe_manifest_rig_ids(
    draft_id: int,
    db_rig: dict[str, int | None],
    *,
    draft_row: dict[str, Any] | None = None,
    db: Any | None = None,
) -> None:
    """Phase 2.1: log manifest-vs-DB rig id mismatches (observe-only)."""
    global MANIFEST_SHADOW_MISMATCHES, MANIFEST_SHADOW_ABSENT, MANIFEST_SHADOW_EQUAL

    if not db_rig:
        return

    did = int(draft_id)
    root = _resolve_archive_root_for_shadow(did, draft_row, db)
    if root is None:
        MANIFEST_SHADOW_ABSENT += 1
        return

    manifest = _load_manifest_for_shadow(did, root)
    if not manifest:
        MANIFEST_SHADOW_ABSENT += 1
        return

    rig_m = manifest.get("rig") if isinstance(manifest.get("rig"), dict) else {}
    mismatches: list[tuple[str, int | None, int | None]] = []
    for key, db_val in db_rig.items():
        man_val = _normalize_rig_id(rig_m.get(key))
        db_norm = _normalize_rig_id(db_val)
        if db_norm != man_val:
            mismatches.append((str(key), db_norm, man_val))

    if mismatches:
        MANIFEST_SHADOW_MISMATCHES += 1
        from infolog import log_event

        for key, db_v, man_v in mismatches:
            msg = (
                f"MANIFEST_SHADOW mismatch draft_id={did} {key} "
                f"db={db_v!r} manifest={man_v!r}"
            )
            logging.warning(msg)
            log_event(msg)
    else:
        MANIFEST_SHADOW_EQUAL += 1


def _log_manifest_fallback(draft_id: int, field: str, reason: str) -> None:
    global MANIFEST_FALLBACK
    MANIFEST_FALLBACK += 1
    from infolog import log_event

    msg = f"MANIFEST_FALLBACK draft_id={int(draft_id)} {field} ({reason})"
    logging.info(msg)
    log_event(msg)


def resolve_rig_id_manifest_first(
    draft_id: int,
    field: str,
    db_value: int | None,
    *,
    draft_row: dict[str, Any] | None = None,
    db: Any | None = None,
) -> int | None:
    """Phase 2.2: return manifest rig id when present, else DB fallback."""
    did = int(draft_id)
    db_norm = _normalize_rig_id(db_value)
    root = _resolve_archive_root_for_shadow(did, draft_row, db)
    if root is None:
        _log_manifest_fallback(did, field, "archive root not resolvable")
        return db_norm

    manifest = _load_manifest_for_shadow(did, root)
    if not manifest:
        _log_manifest_fallback(did, field, "manifest absent")
        return db_norm

    rig_m = manifest.get("rig") if isinstance(manifest.get("rig"), dict) else {}
    if field not in rig_m or rig_m.get(field) is None:
        _log_manifest_fallback(did, field, "rig field missing in manifest")
        return db_norm

    man_val = _normalize_rig_id(rig_m.get(field))
    if man_val is None:
        _log_manifest_fallback(did, field, "rig field null in manifest")
        return db_norm

    if man_val != db_norm:
        observe_manifest_rig_ids(
            did,
            {field: db_norm},
            draft_row=draft_row,
            db=db,
        )
    else:
        global MANIFEST_SHADOW_EQUAL
        MANIFEST_SHADOW_EQUAL += 1
    return man_val


_MANIFEST_FIELD_MISSING = object()

_PATH_MANIFEST_TO_DB: dict[tuple[str, ...], str] = {
    ("paths", "lights"): "LIGHTS_PATH",
    ("paths", "calib"): "CALIB_PATH",
    ("paths", "archive"): "ARCHIVE_PATH",
    ("paths", "masterstar"): "MASTERSTAR_PATH",
    ("paths", "masterstar_fits"): "MASTERSTAR_FITS_PATH",
}

_SCALAR_MANIFEST_TO_DB: dict[tuple[str, ...], str] = {
    ("status",): "STATUS",
    ("observation_start_jd",): "OBSERVATIONSTARTJD",
    ("is_calibrated",): "IS_CALIBRATED",
    ("final_observation_id",): "FINAL_OBSERVATION_ID",
}


def _manifest_nested_value(manifest: dict[str, Any], path: tuple[str, ...]) -> Any:
    cur: Any = manifest
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return _MANIFEST_FIELD_MISSING
        cur = cur[key]
    return cur


def _overlay_manifest_scalar(
    draft_id: int,
    row_dict: dict[str, Any],
    manifest: dict[str, Any],
    path: tuple[str, ...],
    db_col: str,
) -> None:
    man_val = _manifest_nested_value(manifest, path)
    if man_val is _MANIFEST_FIELD_MISSING:
        return
    if db_col == "IS_CALIBRATED":
        try:
            row_dict[db_col] = 1 if int(man_val) != 0 else 0
        except (TypeError, ValueError):
            row_dict[db_col] = 0
        return
    if db_col == "OBSERVATIONSTARTJD":
        jd = _optional_float(man_val)
        row_dict[db_col] = jd
        return
    if db_col == "STATUS":
        row_dict[db_col] = str(man_val).strip() if man_val is not None else None
        return
    if db_col == "FINAL_OBSERVATION_ID":
        s = str(man_val).strip() if man_val is not None else ""
        row_dict[db_col] = s or None
        return
    row_dict[db_col] = man_val


def apply_manifest_core_to_draft_row(
    draft_id: int,
    row_dict: dict[str, Any],
    *,
    db: Any | None = None,
) -> None:
    """Overlay manifest-first rig ids and core draft fields onto an OBS_DRAFT row dict."""
    apply_manifest_rig_to_draft_row(int(draft_id), row_dict, db=db)
    did = int(draft_id)
    root = _resolve_archive_root_for_shadow(did, row_dict, db)
    if root is None:
        for db_col in list(_PATH_MANIFEST_TO_DB.values()) + list(_SCALAR_MANIFEST_TO_DB.values()):
            _log_manifest_fallback(did, db_col, "archive root not resolvable")
        _log_manifest_fallback(did, "center", "archive root not resolvable")
        return

    manifest = _load_manifest_for_shadow(did, root)
    if not manifest:
        for db_col in list(_PATH_MANIFEST_TO_DB.values()) + list(_SCALAR_MANIFEST_TO_DB.values()):
            _log_manifest_fallback(did, db_col, "manifest absent")
        _log_manifest_fallback(did, "center", "manifest absent")
        return

    for path, db_col in _PATH_MANIFEST_TO_DB.items():
        _overlay_manifest_scalar(did, row_dict, manifest, path, db_col)
    for path, db_col in _SCALAR_MANIFEST_TO_DB.items():
        _overlay_manifest_scalar(did, row_dict, manifest, path, db_col)

    center_m = manifest.get("center")
    if isinstance(center_m, dict):
        if "ra_deg" in center_m:
            row_dict["CENTEROFFIELDRA"] = center_m.get("ra_deg")
        if "de_deg" in center_m:
            row_dict["CENTEROFFIELDDE"] = center_m.get("de_deg")


def apply_manifest_rig_to_draft_row(
    draft_id: int,
    row_dict: dict[str, Any],
    *,
    db: Any | None = None,
    fields: frozenset[str] | None = None,
) -> None:
    """Overlay manifest-first rig FK ids onto an OBS_DRAFT row dict (in place)."""
    want = fields or frozenset(_RIG_MANIFEST_TO_DB.keys())
    for man_key, db_col in _RIG_MANIFEST_TO_DB.items():
        if man_key not in want:
            continue
        row_dict[db_col] = resolve_rig_id_manifest_first(
            int(draft_id),
            man_key,
            row_dict.get(db_col),
            draft_row=row_dict,
            db=db,
        )


def draft_archive_root(archive_path: Path | str) -> Path:
    """Normalize draft archive root (parent when path is ``non_calibrated/``)."""
    ap = Path(archive_path).expanduser().resolve()
    if ap.name.casefold() == "non_calibrated":
        return ap.parent
    if ap.name.casefold() == "lights":
        return ap.parent.parent
    if ap.name.casefold() == "calibrated":
        return ap.parent
    return ap


def resolve_draft_archive_root_from_row(row: dict[str, Any]) -> Path | None:
    """Resolve draft archive root from an OBS_DRAFT row (ARCHIVE_PATH, else LIGHTS_PATH)."""
    arch = row.get("ARCHIVE_PATH")
    if arch is not None and str(arch).strip():
        return draft_archive_root(str(arch).strip())
    lights = row.get("LIGHTS_PATH")
    if lights is not None and str(lights).strip():
        return draft_archive_root(str(lights).strip())
    return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _center_from_draft_row(row: dict[str, Any]) -> dict[str, float | None]:
    ra = _optional_float(row.get("CENTEROFFIELDRA"))
    de = _optional_float(row.get("CENTEROFFIELDDE"))
    if ra is None or de is None:
        return {"ra_deg": None, "de_deg": None}
    if ra == 0.0 and de == 0.0:
        return {"ra_deg": None, "de_deg": None}
    return {"ra_deg": ra, "de_deg": de}


def _paths_from_draft_row(row: dict[str, Any]) -> dict[str, str | None]:
    def _s(key: str) -> str | None:
        v = row.get(key)
        if v is None:
            return None
        s = str(v).strip()
        return s or None

    return {
        "lights": _s("LIGHTS_PATH"),
        "calib": _s("CALIB_PATH"),
        "archive": _s("ARCHIVE_PATH"),
        "masterstar": _s("MASTERSTAR_PATH"),
        "masterstar_fits": _s("MASTERSTAR_FITS_PATH"),
    }


def _rig_from_draft_row(row: dict[str, Any]) -> dict[str, int | None]:
    def _i(key: str) -> int | None:
        v = row.get(key)
        if v is None:
            return None
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    return {
        "equipment_id": _i("ID_EQUIPMENTS"),
        "telescope_id": _i("ID_TELESCOPE"),
        "location_id": _i("ID_LOCATION"),
        "scanning_id": _i("ID_SCANNING"),
    }


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def _optional_flag01(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return 1 if int(value) != 0 else 0
    except (TypeError, ValueError):
        return None


def _obs_file_row_to_manifest_entry(row: dict[str, Any]) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "file_path": _optional_str(row.get("FILE_PATH")) or "",
        "imagetyp": _optional_str(row.get("IMAGETYP")),
        "filter": _optional_str(row.get("FILTER")),
        "qc": {
            "hfr": _optional_float(row.get("QC_HFR")),
            "stars": _optional_int(row.get("QC_STARS")),
            "background": _optional_float(row.get("QC_BACKGROUND")),
            "bg_rms": _optional_float(row.get("QC_BG_RMS")),
            "passed": _optional_flag01(row.get("QC_PASSED")),
        },
        "inspection": {
            "fwhm": _optional_float(row.get("FWHM")),
            "sky_level": _optional_float(row.get("SKY_LEVEL")),
            "star_count": _optional_int(row.get("STAR_COUNT")),
            "rejected_auto": _optional_flag01(row.get("REJECTED_AUTO")),
            "is_rejected": _optional_flag01(row.get("IS_REJECTED")),
            "inspection_jd": _optional_float(row.get("INSPECTION_JD")),
            "ra": _optional_float(row.get("RA")),
            "de": _optional_float(row.get("DE")),
            "exptime": _optional_float(row.get("EXPTIME")),
            "drift": _optional_float(row.get("DRIFT")),
            "drift_dra": _optional_float(row.get("DRIFT_DRA")),
            "drift_dde": _optional_float(row.get("DRIFT_DDE")),
            "roundness_mean": _optional_float(row.get("ROUNDNESS_MEAN")),
            "elongation_mean": _optional_float(row.get("ELONGATION_MEAN")),
        },
        "group_key": _optional_str(row.get("OBSERVATION_GROUP_KEY")),
        "id_scanning": _optional_int(row.get("ID_SCANNING")),
        "is_calibrated": _optional_flag01(row.get("IS_CALIBRATED")),
        "calib_type": _optional_str(row.get("CALIB_TYPE")),
        "calib_flags": _optional_str(row.get("CALIB_FLAGS")),
    }
    obs_id = _optional_int(row.get("ID"))
    if obs_id is not None:
        entry["obs_file_id"] = obs_id
    return entry


def _fetch_manifest_files_from_db(db: Any, draft_id: int) -> list[dict[str, Any]]:
    cur = db.conn.execute(
        """
        SELECT FILE_PATH, IMAGETYP, FILTER,
               QC_HFR, QC_STARS, QC_BACKGROUND, QC_BG_RMS, QC_PASSED,
               FWHM, SKY_LEVEL, STAR_COUNT, REJECTED_AUTO, IS_REJECTED, INSPECTION_JD,
               RA, DE, EXPTIME, DRIFT, DRIFT_DRA, DRIFT_DDE, ROUNDNESS_MEAN, ELONGATION_MEAN,
               OBSERVATION_GROUP_KEY, ID_SCANNING, IS_CALIBRATED, CALIB_TYPE, CALIB_FLAGS
        FROM OBS_FILES
        WHERE DRAFT_ID = ?
        ORDER BY FILE_PATH, ID;
        """,
        (int(draft_id),),
    )
    return [_obs_file_row_to_manifest_entry(dict(r)) for r in cur.fetchall()]


def _manifest_entry_to_obs_file_row(
    entry: dict[str, Any],
    *,
    draft_id: int,
    obs_file_id: int | None = None,
) -> dict[str, Any]:
    qc = entry.get("qc") if isinstance(entry.get("qc"), dict) else {}
    insp = entry.get("inspection") if isinstance(entry.get("inspection"), dict) else {}
    row_id = _optional_int(entry.get("obs_file_id"))
    if row_id is None:
        row_id = obs_file_id
    return {
        "ID": row_id,
        "FILE_PATH": _optional_str(entry.get("file_path")) or "",
        "IMAGETYP": _optional_str(entry.get("imagetyp")),
        "FILTER": _optional_str(entry.get("filter")),
        "OBSERVATION_GROUP_KEY": _optional_str(entry.get("group_key")),
        "ID_SCANNING": _optional_int(entry.get("id_scanning")),
        "IS_CALIBRATED": _optional_flag01(entry.get("is_calibrated")),
        "CALIB_TYPE": _optional_str(entry.get("calib_type")),
        "CALIB_FLAGS": _optional_str(entry.get("calib_flags")),
        "FWHM": _optional_float(insp.get("fwhm")),
        "SKY_LEVEL": _optional_float(insp.get("sky_level")),
        "STAR_COUNT": _optional_int(insp.get("star_count")),
        "REJECTED_AUTO": _optional_flag01(insp.get("rejected_auto")),
        "IS_REJECTED": _optional_flag01(insp.get("is_rejected")),
        "INSPECTION_JD": _optional_float(insp.get("inspection_jd")),
        "RA": _optional_float(insp.get("ra")),
        "DE": _optional_float(insp.get("de")),
        "EXPTIME": _optional_float(insp.get("exptime")),
        "DRIFT": _optional_float(insp.get("drift")),
        "DRIFT_DRA": _optional_float(insp.get("drift_dra")),
        "DRIFT_DDE": _optional_float(insp.get("drift_dde")),
        "ROUNDNESS_MEAN": _optional_float(insp.get("roundness_mean")),
        "ELONGATION_MEAN": _optional_float(insp.get("elongation_mean")),
        "DRAFT_ID": int(draft_id),
    }


def _obs_file_id_map_for_draft(db: Any, draft_id: int) -> dict[str, int]:
    """Writer/parity helper: map normalized FILE_PATH -> OBS_FILES.ID."""
    cur = db.conn.execute(
        "SELECT ID, FILE_PATH FROM OBS_FILES WHERE DRAFT_ID = ?;",
        (int(draft_id),),
    )
    out: dict[str, int] = {}
    for r in cur.fetchall():
        fp = _optional_str(r["FILE_PATH"]) if hasattr(r, "keys") else _optional_str(r[1])
        rid = _optional_int(r["ID"]) if hasattr(r, "keys") else _optional_int(r[0])
        if fp and rid is not None:
            out[str(Path(fp).resolve())] = int(rid)
            out[fp] = int(rid)
    return out


def light_rows_from_manifest(
    db: Any,
    draft_id: int,
    *,
    imagetyp: str = "light",
) -> list[dict[str, Any]] | None:
    """Return OBS_FILES-shaped light rows from manifest files[] when present."""
    did = int(draft_id)
    row = _fetch_obs_draft_row_raw(db, did)
    root = resolve_draft_archive_root_from_row(row or {})
    if root is None:
        _log_manifest_fallback(did, "files[]", "archive root not resolvable")
        return None
    manifest = _load_manifest_for_shadow(did, root)
    if not manifest:
        _log_manifest_fallback(did, "files[]", "manifest absent")
        return None
    files = manifest.get("files")
    if not isinstance(files, list):
        _log_manifest_fallback(did, "files[]", "files missing in manifest")
        return None

    id_map = _obs_file_id_map_for_draft(db, did)
    want = str(imagetyp or "light").strip().lower()
    rows: list[dict[str, Any]] = []
    for entry in files:
        if not isinstance(entry, dict):
            continue
        it = str(entry.get("imagetyp") or "").strip().lower()
        if it != want:
            continue
        fp = _optional_str(entry.get("file_path")) or ""
        obs_id = _optional_int(entry.get("obs_file_id"))
        if obs_id is None and fp:
            obs_id = id_map.get(fp) or id_map.get(str(Path(fp).resolve()))
        rows.append(_manifest_entry_to_obs_file_row(entry, draft_id=did, obs_file_id=obs_id))
    rows.sort(key=lambda r: str(r.get("FILE_PATH") or ""))
    return rows


def _manifest_file_entry_mismatch(label: str, db_val: Any, man_val: Any) -> str | None:
    if db_val != man_val:
        return f"{label} DB={db_val!r} manifest={man_val!r}"
    return None


def _compare_manifest_file_entries(
    draft_id: int,
    file_path: str,
    db_entry: dict[str, Any],
    man_entry: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    prefix = f"draft_id={draft_id}: files[{file_path!r}]"

    for key in ("imagetyp", "filter", "group_key", "calib_type", "calib_flags"):
        err = _manifest_file_entry_mismatch(
            f"{prefix}.{key}",
            db_entry.get(key),
            man_entry.get(key),
        )
        if err:
            errors.append(err)

    for key in ("id_scanning", "is_calibrated"):
        err = _manifest_file_entry_mismatch(
            f"{prefix}.{key}",
            db_entry.get(key),
            man_entry.get(key),
        )
        if err:
            errors.append(err)

    db_qc = db_entry.get("qc") if isinstance(db_entry.get("qc"), dict) else {}
    man_qc = man_entry.get("qc") if isinstance(man_entry.get("qc"), dict) else {}
    for key in ("hfr", "stars", "background", "bg_rms", "passed"):
        err = _manifest_file_entry_mismatch(
            f"{prefix}.qc.{key}",
            db_qc.get(key),
            man_qc.get(key),
        )
        if err:
            errors.append(err)
            break

    db_insp = db_entry.get("inspection") if isinstance(db_entry.get("inspection"), dict) else {}
    man_insp = man_entry.get("inspection") if isinstance(man_entry.get("inspection"), dict) else {}
    for key in (
        "fwhm",
        "sky_level",
        "star_count",
        "rejected_auto",
        "is_rejected",
        "inspection_jd",
        "ra",
        "de",
        "exptime",
        "drift",
        "drift_dra",
        "drift_dde",
        "roundness_mean",
        "elongation_mean",
    ):
        err = _manifest_file_entry_mismatch(
            f"{prefix}.inspection.{key}",
            db_insp.get(key),
            man_insp.get(key),
        )
        if err:
            errors.append(err)
            break

    return errors


def is_pre_calibrated_draft(
    archive_path: Path | str,
    *,
    draft_id: int | None = None,
    db: Any = None,
) -> bool:
    ap = draft_archive_root(archive_path)
    return resolve_calibration_mode(draft_id=draft_id, db=db, archive_path=ap) == CALIBRATION_MODE_PRE


def resolve_draft_lights_root(
    archive_path: Path | str,
    *,
    draft_id: int | None = None,
    db: Any = None,
) -> Path:
    """Single lights-source root for a draft: ``non_calibrated/lights`` or ``calibrated/lights``."""
    ap = draft_archive_root(archive_path)
    if is_pre_calibrated_draft(ap, draft_id=draft_id, db=db):
        return ap / "non_calibrated" / "lights"
    return ap / "calibrated" / "lights"


def apply_pre_calibrated_import_plan(plan: SmartImportPlan) -> None:
    """Force quick-look import into ``non_calibrated/lights`` (Telescope Live / pre-cal exports)."""
    plan.quick_look = True
    plan.dark_master = None
    plan.flat_master = None
    plan.masterflat_by_filter = {}
    plan.masterflat_by_obs_key = {}
    plan.dark_master_by_obs_key = {}
    plan.missing_flat_filters = []
    plan.missing_obs_keys = []
    msg = "Pre-calibrated mode: bias/dark/flat skipped; source lights treated as already calibrated."
    if msg not in plan.warnings:
        plan.warnings.append(msg)


def calibration_mode_report_line(mode: str | None) -> str:
    if str(mode or "").strip() == CALIBRATION_MODE_PRE:
        return "Calibration: skipped - source treated as pre-calibrated"
    return "Calibration: VYVAR bias/dark/flat applied"


def write_draft_manifest(
    archive_path: Path | str,
    *,
    draft_id: int,
    calibration_mode: str,
    extra: dict[str, Any] | None = None,
    rig: dict[str, Any] | None = None,
    paths: dict[str, Any] | None = None,
    status: str | None = None,
    final_observation_id: str | None = None,
    is_calibrated: int | None = None,
    center: dict[str, Any] | None = None,
    observation_start_jd: float | None = None,
    schema_version: int = MANIFEST_SCHEMA_VERSION,
    files: list[dict[str, Any]] | None = None,
) -> Path:
    """Persist draft manifest JSON under the draft archive root."""
    root = Path(archive_path).expanduser().resolve()
    payload: dict[str, Any] = {
        "schema_version": int(schema_version),
        "draft_id": int(draft_id),
        "calibration_mode": str(calibration_mode),
        "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if rig is not None:
        payload["rig"] = dict(rig)
        payload["paths"] = dict(paths or {})
        if status is not None:
            payload["status"] = str(status)
        payload["final_observation_id"] = (
            str(final_observation_id) if final_observation_id is not None else None
        )
        if is_calibrated is not None:
            payload["is_calibrated"] = int(is_calibrated)
        payload["center"] = dict(center or {"ra_deg": None, "de_deg": None})
        payload["observation_start_jd"] = observation_start_jd
    if files is not None:
        payload["files"] = list(files)
    if extra:
        payload.update(extra)
    path = root / _MANIFEST_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_draft_manifest(archive_path: Path | str) -> dict[str, Any]:
    path = Path(archive_path).expanduser().resolve() / _MANIFEST_NAME
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception as exc:  # noqa: BLE001
        logging.warning(
            "[EXC-0076] report/export may omit or misstate (raw = json.loads(path.read_text(encoding='utf-8')) ...: %s",
            exc,
        )
        return {}


def _fetch_obs_draft_row_raw(db: Any, draft_id: int) -> dict[str, Any] | None:
    """Load OBS_DRAFT row without Phase 2.1 shadow-observe (manifest writer / internal use)."""
    if not hasattr(db, "conn"):
        return None
    row = db.conn.execute("SELECT * FROM OBS_DRAFT WHERE ID = ?;", (int(draft_id),)).fetchone()
    return dict(row) if row is not None else None


def resolve_calibration_mode(
    *,
    draft_id: int | None = None,
    db: Any = None,
    archive_path: Path | str | None = None,
) -> str:
    """Resolve calibration_mode from DB, then draft manifest, else default."""
    if db is not None and draft_id is not None:
        try:
            row = _fetch_obs_draft_row_raw(db, int(draft_id)) or {}
            mode = row.get("CALIBRATION_MODE") or row.get("calibration_mode")
            if mode:
                return str(mode)
        except Exception:  # noqa: BLE001
            pass
    if archive_path is not None:
        mode = load_draft_manifest(archive_path).get("calibration_mode")
        if mode:
            return str(mode)
    return CALIBRATION_MODE_VYVAR


def record_draft_manifest_core(db: Any, draft_id: int) -> Path | None:
    """Mirror OBS_DRAFT core fields into draft_manifest.json (DB is authority in Phase 1)."""
    from infolog import log_event

    did = int(draft_id)
    row = _fetch_obs_draft_row_raw(db, did)
    if not row:
        log_event(f"draft_manifest: skip draft_id={did} (OBS_DRAFT row missing)")
        return None

    root = resolve_draft_archive_root_from_row(row)
    if root is None:
        log_event(f"draft_manifest: skip draft_id={did} (archive root not resolvable)")
        return None

    existing = load_draft_manifest(root)
    extra: dict[str, Any] = {}
    if isinstance(existing.get("observer_location"), dict):
        extra["observer_location"] = existing["observer_location"]

    mode = resolve_calibration_mode(draft_id=did, db=db, archive_path=root)
    jd_raw = row.get("OBSERVATIONSTARTJD")
    jd_val = _optional_float(jd_raw)
    if jd_val is not None and jd_val == 0.0:
        jd_val = None

    is_cal_raw = row.get("IS_CALIBRATED")
    try:
        is_cal = 0 if is_cal_raw is None else (1 if int(is_cal_raw) != 0 else 0)
    except (TypeError, ValueError):
        is_cal = 0

    final_obs = row.get("FINAL_OBSERVATION_ID")
    final_obs_s = str(final_obs).strip() if final_obs is not None and str(final_obs).strip() else None

    return write_draft_manifest(
        root,
        draft_id=did,
        calibration_mode=mode,
        extra=extra or None,
        rig=_rig_from_draft_row(row),
        paths=_paths_from_draft_row(row),
        status=str(row.get("STATUS") or "").strip() or None,
        final_observation_id=final_obs_s,
        is_calibrated=is_cal,
        center=_center_from_draft_row(row),
        observation_start_jd=jd_val,
        files=_fetch_manifest_files_from_db(db, did),
    )


def record_observer_location_provenance(
    *,
    archive_path: Path | str,
    draft_id: int,
    resolved: Any,
) -> Path:
    """Persist resolved observer site into draft_manifest.json."""
    prov = resolved.as_provenance_dict() if hasattr(resolved, "as_provenance_dict") else dict(resolved)
    root = draft_archive_root(archive_path)
    manifest = load_draft_manifest(root)
    mode = str(manifest.get("calibration_mode") or CALIBRATION_MODE_VYVAR)
    extra = {"observer_location": prov}
    if isinstance(manifest.get("rig"), dict):
        return write_draft_manifest(
            root,
            draft_id=int(draft_id),
            calibration_mode=mode,
            extra=extra,
            rig=manifest.get("rig"),
            paths=manifest.get("paths"),
            status=manifest.get("status"),
            final_observation_id=manifest.get("final_observation_id"),
            is_calibrated=manifest.get("is_calibrated"),
            center=manifest.get("center"),
            observation_start_jd=manifest.get("observation_start_jd"),
            files=manifest.get("files") if isinstance(manifest.get("files"), list) else None,
        )
    return write_draft_manifest(
        root,
        draft_id=int(draft_id),
        calibration_mode=mode,
        extra=extra,
    )


def record_draft_calibration_provenance(
    *,
    db: Any,
    archive_path: Path | str,
    draft_id: int,
    calibration_mode: str,
) -> Path:
    """Persist calibration_mode to OBS_DRAFT and draft_manifest.json."""
    if hasattr(db, "set_obs_draft_calibration_mode"):
        db.set_obs_draft_calibration_mode(int(draft_id), str(calibration_mode))
    path = record_draft_manifest_core(db, int(draft_id))
    if path is not None:
        return path
    return write_draft_manifest(
        archive_path,
        draft_id=int(draft_id),
        calibration_mode=str(calibration_mode),
    )


def manifest_db_parity_errors(db: Any, draft_id: int) -> list[str]:
    """Return parity mismatch messages between manifest and OBS_DRAFT (empty if OK)."""
    did = int(draft_id)
    row = _fetch_obs_draft_row_raw(db, did)
    if not row:
        return [f"draft_id={did}: OBS_DRAFT row missing"]

    root = resolve_draft_archive_root_from_row(row)
    if root is None:
        return [f"draft_id={did}: archive root not resolvable"]

    manifest = load_draft_manifest(root)
    if not manifest:
        return [f"draft_id={did}: draft_manifest.json missing at {root}"]

    errors: list[str] = []

    def _eq(label: str, db_val: Any, man_val: Any) -> None:
        if db_val != man_val:
            errors.append(f"draft_id={did}: {label} DB={db_val!r} manifest={man_val!r}")

    rig_m = manifest.get("rig") if isinstance(manifest.get("rig"), dict) else {}
    rig_db = _rig_from_draft_row(row)
    for key in ("equipment_id", "telescope_id", "location_id", "scanning_id"):
        _eq(f"rig.{key}", rig_db.get(key), rig_m.get(key))

    paths_m = manifest.get("paths") if isinstance(manifest.get("paths"), dict) else {}
    paths_db = _paths_from_draft_row(row)
    for key in ("lights", "calib", "archive", "masterstar", "masterstar_fits"):
        _eq(f"paths.{key}", paths_db.get(key), paths_m.get(key))

    db_status = str(row.get("STATUS") or "").strip() or None
    man_status = manifest.get("status")
    if man_status is not None:
        man_status = str(man_status).strip() or None
    _eq("status", db_status, man_status)

    db_final = row.get("FINAL_OBSERVATION_ID")
    db_final_s = str(db_final).strip() if db_final is not None and str(db_final).strip() else None
    man_final = manifest.get("final_observation_id")
    if man_final is not None:
        man_final = str(man_final).strip() or None
    _eq("final_observation_id", db_final_s, man_final)

    try:
        db_is_cal = 0 if row.get("IS_CALIBRATED") is None else (1 if int(row["IS_CALIBRATED"]) != 0 else 0)
    except (TypeError, ValueError):
        db_is_cal = 0
    man_is_cal = manifest.get("is_calibrated")
    if man_is_cal is not None:
        try:
            man_is_cal = 1 if int(man_is_cal) != 0 else 0
        except (TypeError, ValueError):
            man_is_cal = man_is_cal
    _eq("is_calibrated", db_is_cal, man_is_cal)

    center_m = manifest.get("center") if isinstance(manifest.get("center"), dict) else {}
    center_db = _center_from_draft_row(row)
    _eq("center.ra_deg", center_db.get("ra_deg"), center_m.get("ra_deg"))
    _eq("center.de_deg", center_db.get("de_deg"), center_m.get("de_deg"))

    jd_db = _optional_float(row.get("OBSERVATIONSTARTJD"))
    if jd_db is not None and jd_db == 0.0:
        jd_db = None
    _eq("observation_start_jd", jd_db, manifest.get("observation_start_jd"))

    db_files = _fetch_manifest_files_from_db(db, did)
    man_files = manifest.get("files")
    if not isinstance(man_files, list):
        if db_files:
            errors.append(f"draft_id={did}: files[] missing in manifest (DB has {len(db_files)} rows)")
        return errors

    if len(db_files) != len(man_files):
        errors.append(
            f"draft_id={did}: files count DB={len(db_files)} manifest={len(man_files)}"
        )
        return errors

    man_by_path: dict[str, dict[str, Any]] = {}
    for item in man_files:
        if isinstance(item, dict):
            fp = str(item.get("file_path") or "")
            man_by_path[fp] = item

    for db_entry in db_files:
        fp = str(db_entry.get("file_path") or "")
        man_entry = man_by_path.get(fp)
        if man_entry is None:
            errors.append(f"draft_id={did}: file_path={fp!r} in DB but missing from manifest")
            return errors
        file_errors = _compare_manifest_file_entries(did, fp, db_entry, man_entry)
        if file_errors:
            errors.extend(file_errors[:3])
            return errors

    return errors


def iter_draft_archive_dirs(archive_root: Path | str) -> list[tuple[int, Path]]:
    """List ``(draft_id, archive_dir)`` under ``Archive/Drafts/draft_*`` (sorted by id desc)."""
    drafts_dir = Path(archive_root).expanduser().resolve() / "Drafts"
    if not drafts_dir.is_dir():
        return []
    out: list[tuple[int, Path]] = []
    for child in drafts_dir.iterdir():
        if not child.is_dir():
            continue
        name = child.name.lower()
        if not name.startswith("draft_"):
            continue
        suffix = name.split("_", 1)[-1]
        try:
            did = int(suffix)
        except ValueError:
            continue
        out.append((int(did), child.resolve()))
    out.sort(key=lambda t: t[0], reverse=True)
    return out


def obs_draft_row_from_manifest(manifest: dict[str, Any], draft_id: int) -> dict[str, Any]:
    """Build an OBS_DRAFT-shaped row dict from ``draft_manifest.json`` (UI display source)."""
    rig = manifest.get("rig") if isinstance(manifest.get("rig"), dict) else {}
    paths = manifest.get("paths") if isinstance(manifest.get("paths"), dict) else {}
    center = manifest.get("center") if isinstance(manifest.get("center"), dict) else {}
    row: dict[str, Any] = {
        "ID": int(draft_id),
        "ID_EQUIPMENTS": _optional_int(rig.get("equipment_id")),
        "ID_TELESCOPE": _optional_int(rig.get("telescope_id")),
        "ID_LOCATION": _optional_int(rig.get("location_id")),
        "ID_SCANNING": _optional_int(rig.get("scanning_id")),
        "OBSERVATIONSTARTJD": _optional_float(manifest.get("observation_start_jd")),
        "CENTEROFFIELDRA": center.get("ra_deg"),
        "CENTEROFFIELDDE": center.get("de_deg"),
        "STATUS": _optional_str(manifest.get("status")),
        "FINAL_OBSERVATION_ID": _optional_str(manifest.get("final_observation_id")),
        "LIGHTS_PATH": _optional_str(paths.get("lights")),
        "CALIB_PATH": _optional_str(paths.get("calib")),
        "ARCHIVE_PATH": _optional_str(paths.get("archive")),
        "MASTERSTAR_PATH": _optional_str(paths.get("masterstar")),
        "MASTERSTAR_FITS_PATH": _optional_str(paths.get("masterstar_fits")),
        "IS_CALIBRATED": _optional_flag01(manifest.get("is_calibrated")),
        "CALIBRATION_MODE": _optional_str(manifest.get("calibration_mode")),
    }
    return row


def collect_manifest_draft_rows(archive_root: Path | str) -> list[dict[str, Any]]:
    """All draft rows for Database Explorer (manifest-first, no OBS_DRAFT SQL)."""
    rows: list[dict[str, Any]] = []
    for did, apath in iter_draft_archive_dirs(archive_root):
        manifest = load_draft_manifest(apath)
        if manifest:
            rows.append(obs_draft_row_from_manifest(manifest, did))
        else:
            rows.append(
                {
                    "ID": int(did),
                    "ARCHIVE_PATH": str(apath),
                    "STATUS": None,
                }
            )
    return rows


def collect_manifest_obs_file_rows(
    archive_root: Path | str,
    *,
    draft_id: int | None = None,
    observation_id: str | None = None,
) -> list[dict[str, Any]]:
    """Flatten manifest ``files[]`` into OBS_FILES-shaped rows (UI display)."""
    want_draft = int(draft_id) if draft_id is not None else None
    want_obs = str(observation_id).strip() if observation_id is not None else None
    rows: list[dict[str, Any]] = []
    for did, apath in iter_draft_archive_dirs(archive_root):
        if want_draft is not None and int(did) != want_draft:
            continue
        manifest = load_draft_manifest(apath)
        if not manifest:
            continue
        final_obs = _optional_str(manifest.get("final_observation_id"))
        if want_obs is not None and final_obs != want_obs:
            continue
        files = manifest.get("files")
        if not isinstance(files, list):
            continue
        for entry in files:
            if not isinstance(entry, dict):
                continue
            row = _manifest_entry_to_obs_file_row(
                entry,
                draft_id=int(did),
                obs_file_id=_optional_int(entry.get("obs_file_id")),
            )
            if final_obs:
                row["OBSERVATION_ID"] = final_obs
            rows.append(row)
    rows.sort(
        key=lambda r: (
            str(r.get("OBSERVATION_ID") or ""),
            int(r.get("DRAFT_ID") or 0),
            str(r.get("FILE_PATH") or ""),
        ),
        reverse=True,
    )
    return rows


def collect_manifest_observation_rows(archive_root: Path | str) -> list[dict[str, Any]]:
    """OBSERVATION-shaped rows from finalized manifests (``final_observation_id`` set)."""
    rows: list[dict[str, Any]] = []
    for did, apath in iter_draft_archive_dirs(archive_root):
        manifest = load_draft_manifest(apath)
        if not manifest:
            continue
        obs_id = _optional_str(manifest.get("final_observation_id"))
        if not obs_id:
            continue
        draft_row = obs_draft_row_from_manifest(manifest, did)
        rows.append(
            {
                "ID": obs_id,
                "ID_EQUIPMENTS": draft_row.get("ID_EQUIPMENTS"),
                "ID_TELESCOPE": draft_row.get("ID_TELESCOPE"),
                "ID_LOCATION": draft_row.get("ID_LOCATION"),
                "ID_SCANNING": draft_row.get("ID_SCANNING"),
                "CENTEROFFIELDRA": draft_row.get("CENTEROFFIELDRA"),
                "CENTEROFFIELDDE": draft_row.get("CENTEROFFIELDDE"),
                "OBSERVATIONSTARTJD": draft_row.get("OBSERVATIONSTARTJD"),
                "LIGHTS_PATH": draft_row.get("LIGHTS_PATH"),
                "CALIB_PATH": draft_row.get("CALIB_PATH"),
                "IS_CALIBRATED": draft_row.get("IS_CALIBRATED"),
                "ARCHIVE_PATH": draft_row.get("ARCHIVE_PATH"),
                "DRAFT_ID": int(did),
            }
        )
    rows.sort(key=lambda r: str(r.get("ID") or ""))
    return rows


def draft_scan_summary_from_manifest(manifest: dict[str, Any]) -> dict[str, Any] | None:
    """Exposure/filter/binning from manifest ``files[]`` (no SCANNING SQL)."""
    files = manifest.get("files")
    if not isinstance(files, list):
        return None
    for entry in files:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("imagetyp") or "").strip().lower() != "light":
            continue
        insp = entry.get("inspection") if isinstance(entry.get("inspection"), dict) else {}
        exptime = _optional_float(insp.get("exptime"))
        filt = _optional_str(entry.get("filter"))
        binning: int | None = None
        fp = _optional_str(entry.get("file_path"))
        if fp:
            fpath = Path(fp)
            if fpath.is_file():
                try:
                    from astropy.io import fits

                    with fits.open(fpath, memmap=True) as hdul:
                        xbin = hdul[0].header.get("XBINNING")
                        if xbin is not None:
                            binning = int(xbin)
                except Exception:  # noqa: BLE001
                    binning = None
        return {"exptime": exptime, "filters": filt, "binning": binning}
    return None
