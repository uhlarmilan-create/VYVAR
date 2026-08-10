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
MANIFEST_SCHEMA_VERSION = 2

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

    return errors
