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
    return {
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


def resolve_calibration_mode(
    *,
    draft_id: int | None = None,
    db: Any = None,
    archive_path: Path | str | None = None,
) -> str:
    """Resolve calibration_mode from DB, then draft manifest, else default."""
    if db is not None and draft_id is not None:
        try:
            row = db.fetch_obs_draft_by_id(int(draft_id)) or {}
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
    row = db.fetch_obs_draft_by_id(did) if hasattr(db, "fetch_obs_draft_by_id") else None
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
    row = db.fetch_obs_draft_by_id(did) if hasattr(db, "fetch_obs_draft_by_id") else None
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
