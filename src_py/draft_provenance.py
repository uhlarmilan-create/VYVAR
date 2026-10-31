"""Draft run provenance - calibration mode and manifest I/O."""

from __future__ import annotations

import functools
import inspect
import json
import os
import re
import hashlib
import logging
import math
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # annotation only: the manifest store must not import the importer (LAYER-2026-10)
    from importer import SmartImportPlan

CALIBRATION_MODE_VYVAR = "vyvar_calibrated"
CALIBRATION_MODE_PRE = "pre_calibrated"
#: G12-01: no frame of the draft got a master DARK (passthrough copy of the raw lights)
CALIBRATION_MODE_PASSTHROUGH = "PASSTHROUGH"
MANIFEST_SCHEMA_VERSION = 3

_MANIFEST_NAME = "draft_manifest.json"

def derive_scanning_id(metadata: dict[str, Any]) -> int:
    """Stable scanning id from FITS metadata (no SCANNING SQL table)."""
    exp_time = float(metadata["exposure"])
    filters = str(metadata["filter"])
    binning = int(metadata["binning"])
    _t = metadata.get("temp")
    # G13-18: an unknown sensor temperature is its own setup, never 0 C
    temp_s = f"{float(_t):.3f}" if _t is not None else "unknown"
    gain = float(metadata.get("gain", 0.0) or 0.0)
    payload = f"{exp_time:.6f}|{filters}|{binning}|{temp_s}|{gain:g}"
    digest = hashlib.md5(payload.encode("utf-8"), usedforsecurity=False).hexdigest()
    val = int(digest[:7], 16) & 0x7FFFFFFF
    return val if val > 0 else 1


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
    """Resolve draft archive root from an draft manifest row (ARCHIVE_PATH, else LIGHTS_PATH)."""
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
            "elongation_mean": _optional_float(row.get("ELONGATION_MEAN")),
            "noise_sigma": _optional_float(row.get("NOISE_SIGMA")),
            "qc_reason": _optional_str(row.get("QC_REASON")),
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


def _manifest_entry_to_obs_file_row(
    entry: dict[str, Any],
    *,
    draft_id: int,
    obs_file_id: int | None = None,
) -> dict[str, Any]:
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
        "QC_REASON": _optional_str(insp.get("qc_reason")),
        "INSPECTION_JD": _optional_float(insp.get("inspection_jd")),
        "RA": _optional_float(insp.get("ra")),
        "DE": _optional_float(insp.get("de")),
        "EXPTIME": _optional_float(insp.get("exptime")),
        "DRIFT": _optional_float(insp.get("drift")),
        "DRIFT_DRA": _optional_float(insp.get("drift_dra")),
        "DRIFT_DDE": _optional_float(insp.get("drift_dde")),
        "ELONGATION_MEAN": _optional_float(insp.get("elongation_mean")),
        "NOISE_SIGMA": _optional_float(insp.get("noise_sigma")),
        "DRAFT_ID": int(draft_id),
    }


def light_rows_from_manifest(
    db: Any,
    draft_id: int,
    *,
    imagetyp: str = "light",
) -> list[dict[str, Any]] | None:
    """Return manifest file-row-shaped light rows from manifest files[] when present."""
    root = resolve_draft_dir_for_id(db, int(draft_id))
    if root is None:
        return None
    return light_rows_from_manifest_dir(root, int(draft_id), imagetyp=imagetyp)


def light_rows_from_manifest_dir(
    root: Path | str,
    draft_id: int,
    *,
    imagetyp: str = "light",
) -> list[dict[str, Any]] | None:
    """:func:`light_rows_from_manifest` of the processed field in folder ``root`` (G14-11: the GUI
    reads the loaded folder, not the archive folder with the same id)."""
    manifest = load_draft_manifest(Path(root))
    files = manifest.get("files") if manifest else None
    if not isinstance(files, list):
        return None

    want = str(imagetyp or "light").strip().lower()
    rows: list[dict[str, Any]] = []
    for entry in files:
        if not isinstance(entry, dict):
            continue
        it = str(entry.get("imagetyp") or "").strip().lower()
        if it != want:
            continue
        obs_id = _optional_int(entry.get("obs_file_id"))
        rows.append(_manifest_entry_to_obs_file_row(entry, draft_id=int(draft_id), obs_file_id=obs_id))
    rows.sort(key=lambda r: str(r.get("FILE_PATH") or ""))
    return rows


def _manifest_file_count(manifest: dict[str, Any]) -> int:
    files = manifest.get("files")
    if not isinstance(files, list):
        return 0
    return sum(1 for e in files if isinstance(e, dict))


def count_manifest_files_for_draft(db: Any, draft_id: int) -> int:
    root = resolve_draft_dir_for_id(db, int(draft_id))
    if root is None:
        return 0
    return _manifest_file_count(load_draft_manifest(root))


def count_manifest_files_for_observation(archive_root: Path | str, observation_id: str) -> int:
    return len(collect_manifest_obs_file_rows(archive_root, observation_id=str(observation_id)))


def _manifest_rig_pair(manifest: dict[str, Any]) -> tuple[int | None, int | None]:
    rig = manifest.get("rig") if isinstance(manifest.get("rig"), dict) else {}
    return (_optional_int(rig.get("equipment_id")), _optional_int(rig.get("telescope_id")))


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
    plan.masterflat_by_filter = {}
    plan.masterflat_by_obs_key = {}
    plan.dark_master_by_obs_key = {}
    plan.missing_flat_filters = []
    plan.missing_obs_keys = []
    msg = "Pre-calibrated mode: dark/flat skipped; LIGHTs used as they are (calibrated elsewhere or deliberately uncalibrated)."
    if msg not in plan.warnings:
        plan.warnings.append(msg)


def _has_fits(root: Path) -> bool:
    from fits_suffixes import list_fits_files  # noqa: PLC0415

    return bool(list_fits_files(root, recursive=True))


def purge_calibrated_light_frames(archive_path: Path | str, *, draft_id: int) -> dict[str, Any]:
    """Delete the calibrated light frames of a finished run (OUTPUT-REVIEW-2026-10, Milan 2026-10-21).

    The calibrated frames (``<draft>/calibrated/lights/**/*.fits``) are read only up to the alignment
    and MASTERSTAR step; photometry, ePSF and the GUI read the aligned frames and MASTERSTAR.fits.
    The source frames (``Raw/lights`` or ``non_calibrated/lights``) stay, so the calibrated frames
    can always be made again by a new run. Only FITS files are deleted (``qc_metrics.csv`` and the
    other tables stay); nothing is deleted when no source frames are present. The manifest records
    ``calibrated_frames_purged`` (UTC time, number of files, bytes).
    """
    root = draft_archive_root(archive_path)
    out: dict[str, Any] = {"n_files": 0, "bytes": 0, "skipped": ""}
    cal = root / "calibrated" / "lights"
    if not _has_fits(cal):
        out["skipped"] = "no calibrated frames"
        return out
    if not (_has_fits(root / "Raw" / "lights") or _has_fits(root / "raw" / "lights")
            or _has_fits(root / "non_calibrated" / "lights")):
        out["skipped"] = "no source frames in the draft - calibrated frames kept"
        return out
    from fits_suffixes import list_fits_files  # noqa: PLC0415

    for fp in list_fits_files(cal, recursive=True):
        size = fp.stat().st_size
        fp.unlink()
        out["n_files"] += 1
        out["bytes"] += int(size)
    manifest = load_draft_manifest(root)
    if manifest:
        manifest["calibrated_frames_purged"] = {
            "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_files": int(out["n_files"]),
            "bytes": int(out["bytes"]),
        }
        _persist_manifest(root, int(draft_id), manifest)
    return out


def calibration_mode_report_line(mode: str | None) -> str:
    if str(mode or "").strip() == CALIBRATION_MODE_PRE:
        return "Calibration: skipped - source treated as pre-calibrated"
    if str(mode or "").strip() == CALIBRATION_MODE_PASSTHROUGH:
        return "Calibration: none - no valid master DARK, raw frames used"
    return "Calibration: VYVAR dark/flat per calibration plan"


#: VY_CFLAG of a fully calibrated frame (dark subtracted and flat-fielded)
CFLAG_FULL = "DF"
_CFLAG_TEXT = {
    "D": "dark only, no flat field",
    "FS": "no dark, flat not applied",
    "P": "no calibration frames (raw)",
}


def record_calibration_outcome(
    archive_path: Path | str,
    *,
    draft_id: int,
    flags_by_group: dict[str, dict[str, int]],
    flat_substitute_by_group: dict[str, str] | None = None,
) -> dict[str, Any]:
    """G01-01 / G12-01: record in the manifest the calibration really done, per LIGHT group folder.

    ``calibration_groups[<group folder>]``: ``vy_cflag`` counts, ``flat_substitute`` (capture date of a FLAT
    from another night, decision A1; ``flat_substitute_by_group`` is keyed by group folder), ``relative_only``
    and ``reason``. A group with any frame that is not fully calibrated (VY_CFLAG other than DF: no flat
    field) is a relative product: its light curves are not for AAVSO/VarAstro submission (trust hard
    reason, see :func:`calibration_trust_reasons`). ``calibration_mode`` becomes PASSTHROUGH when no frame
    got a dark (G12-01, N1), else VYVAR (N2). Returns the group block.
    """
    subs = {str(k): str(v) for k, v in (flat_substitute_by_group or {}).items() if v}
    groups: dict[str, Any] = {}
    all_flags: set[str] = set()
    for folder, counts in sorted((flags_by_group or {}).items()):
        cnt = {str(k): int(v) for k, v in counts.items()}
        all_flags |= set(cnt)
        partial = sorted(f for f in cnt if f != CFLAG_FULL)
        reason = ""
        if partial:
            reason = "calibration incomplete: " + ", ".join(
                f"{cnt[f]} frame(s) {_CFLAG_TEXT.get(f, f)}" for f in partial
            ) + " - relative photometry only"
        groups[str(folder)] = {
            "vy_cflag": cnt,
            "flat_substitute": subs.get(str(folder)) or None,
            "relative_only": bool(partial),
            "reason": reason,
        }
    root = draft_archive_root(archive_path)
    # N1: PASSTHROUGH when no frame got a dark (no flag contains D: P and FS frames are raw);
    # N2: the mode is always written, so a field calibrated again with darks is VYVAR again.
    mode: str | None = None
    if all_flags:
        mode = CALIBRATION_MODE_PASSTHROUGH if not any("D" in f for f in all_flags) else CALIBRATION_MODE_VYVAR
    patch_draft_manifest(root, int(draft_id), calibration_mode=mode, extra={"calibration_groups": groups})
    return groups


def calibration_trust_reasons(draft_dir: Path | str, group_folder: str) -> tuple[str, str]:
    """(hard, soft) trust reason of the calibration of one LIGHT group folder (decision A1).

    Hard: the group is a relative product (not fully calibrated). Soft: the FLAT is a flagged substitute
    from another night. ('', '') when the draft manifest has no calibration record for the group.
    """
    groups = load_draft_manifest(draft_dir).get("calibration_groups")
    g = groups.get(str(group_folder)) if isinstance(groups, dict) else None
    if not isinstance(g, dict):
        return "", ""
    hard = str(g.get("reason") or "") if g.get("relative_only") else ""
    soft = f"FLAT from another night ({g['flat_substitute']})" if g.get("flat_substitute") else ""
    return hard, soft


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
    center_by_setup: dict[str, Any] | None = None,
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
        if center_by_setup is not None:
            payload["center_by_setup"] = dict(center_by_setup)  # S6-FIX: one centre per setup, no field centre
        payload["observation_start_jd"] = observation_start_jd
    if files is not None:
        payload["files"] = list(files)
    if extra:
        payload.update(extra)
    path = root / _MANIFEST_NAME
    _write_manifest_file(path, payload)  # G12-21: the one atomic manifest writer
    return path


def load_draft_manifest(archive_path: Path | str) -> dict[str, Any]:
    root = Path(archive_path).expanduser().resolve()
    batch = _MANIFEST_BATCHES.get(str(root))
    if batch is not None:
        # G01-13: inside a manifest batch the stage's manifest lives in memory; a reader gets a
        # fresh copy, exactly what reading the file back would give.
        return batch.copy_manifest()
    path = root / _MANIFEST_NAME
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


def resolve_draft_dir_for_id(db: Any, draft_id: int) -> Path | None:
    """Return draft archive directory containing ``draft_manifest.json`` for ``draft_id``."""
    if db is None:
        return None
    resolver = getattr(db, "resolve_archive_root", None)
    if resolver is None:
        return None
    try:
        archive_root = Path(resolver()).expanduser().resolve()
    except Exception:  # noqa: BLE001
        return None
    # G12-42 (Milan 2026-10-30): processed fields live only under Archive/Drafts
    # (iter_draft_archive_dirs); the archive-wide rglob fallback over all FITS trees is gone.
    found = draft_dir_for_id(archive_root, int(draft_id))
    if found.is_dir():
        return found.resolve()
    return None


def allocate_next_draft_id(archive_root: Path | str, *, floor: int = 0) -> int:
    """Next processed-field id: above every id under ``Archive/Drafts`` and above ``floor`` (the
    largest id the database has seen), so ids are never reused when old folders are deleted."""
    max_id = int(floor or 0)
    for did, _ in iter_draft_archive_dirs(archive_root):
        max_id = max(max_id, int(did))
    return int(max_id) + 1


# PROCESSED-FIELD-NAME-2026-10 (Milan 2026-10-24): a processed field is stored as
# Archive/Drafts/<TARGET>_<YYYYMMDD>_<NNN> - the observed star, the date of the evening the night
# began, a counter for repeated runs of the same star and night.
_GENERIC_TARGET_NAMES = {"", "field", "object", "light", "lights", "target", "unknown", "none", "nan"}


def processed_field_target_name(object_name: Any, source_folder_name: Any) -> str:
    """Folder-safe target: the FITS OBJECT, else the selected source folder's name, else 'field'.
    Spaces become '_', characters outside [A-Za-z0-9+-_] are dropped ('MN Boo' -> 'MN_Boo')."""
    for raw in (object_name, source_folder_name):
        txt = re.sub(r"\s+", "_", str(raw or "").strip())
        txt = re.sub(r"[^A-Za-z0-9+\-_]", "", txt).strip("_")
        txt = re.sub(r"_+", "_", txt)
        if txt and txt.lower() not in _GENERIC_TARGET_NAMES:
            return txt[:60]
    return "field"


def processed_field_night_date(jd_utc: float | None, lon_deg: float | None) -> str:
    """YYYYMMDD of the evening the night began: local mean solar time (site longitude) minus 12 h,
    so frames after midnight keep the previous evening's date. Today (UTC) without a valid JD."""
    try:
        jd = float(jd_utc)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        jd = float("nan")
    if not (jd == jd and jd > 2400000.0):
        return datetime.now(timezone.utc).strftime("%Y%m%d")
    lon = float(lon_deg) if lon_deg is not None and float(lon_deg) == float(lon_deg) else 0.0
    t = datetime(1858, 11, 17, tzinfo=timezone.utc) + timedelta(days=jd - 2400000.5)
    return (t + timedelta(hours=lon / 15.0 - 12.0)).strftime("%Y%m%d")


def make_processed_field_folder(archive_root: Path | str, target: str, night: str) -> Path:
    """Create ``Archive/Drafts/<target>_<night>_<NNN>`` with the first free counter and return it.

    G12-45: each candidate is claimed by ``mkdir(exist_ok=False)``, so two concurrent imports never
    share a folder (the former ``exists()`` probe and a later mkdir could race)."""
    drafts = Path(archive_root).expanduser().resolve() / "Drafts"
    drafts.mkdir(parents=True, exist_ok=True)
    n = 1
    while True:
        cand = drafts / f"{target}_{night}_{n:03d}"
        try:
            cand.mkdir(exist_ok=False)
            return cand
        except FileExistsError:
            n += 1


def _ingest_item_to_manifest_entry(item: dict[str, Any], *, obs_file_id: int) -> dict[str, Any]:
    fake_row = {
        "ID": int(obs_file_id),
        "FILE_PATH": item.get("file_path"),
        "IMAGETYP": item.get("imagetyp"),
        "FILTER": item.get("filter"),
        "OBSERVATION_GROUP_KEY": item.get("observation_group_key"),
        "ID_SCANNING": item.get("id_scanning"),
        "IS_CALIBRATED": item.get("is_calibrated"),
        "CALIB_TYPE": item.get("calib_type"),
        "CALIB_FLAGS": item.get("calib_flags"),
    }
    return _obs_file_row_to_manifest_entry(fake_row)


def _persist_manifest(
    root: Path | str,
    draft_id: int,
    manifest: dict[str, Any],
) -> Path:
    """Write full manifest dict to ``draft_manifest.json`` (atomic: temp file + ``os.replace``).

    Inside a :func:`manifest_batch` for this folder the manifest is kept in memory and written once
    when the batch ends (G01-13)."""
    payload = dict(manifest)
    payload["schema_version"] = int(payload.get("schema_version") or MANIFEST_SCHEMA_VERSION)
    payload["draft_id"] = int(draft_id)
    payload["updated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    root_r = Path(root).expanduser().resolve()
    path = root_r / _MANIFEST_NAME
    batch = _MANIFEST_BATCHES.get(str(root_r))
    if batch is not None:
        batch.replace_manifest(payload)
        return path
    _write_manifest_file(path, payload)
    return path


#: G01-13 atomic manifest write: retries of ``os.replace`` while another process (GUI) holds the
#: file open on Windows (sharing violation); then the error is raised.
_MANIFEST_REPLACE_RETRIES = 20
_MANIFEST_REPLACE_WAIT_S = 0.05


def _write_manifest_file(path: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` to ``path`` atomically (temp file in the same folder + ``os.replace``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, indent=2) + "\n")
            fh.flush()
            os.fsync(fh.fileno())  # G12-21: the renamed file is complete after a crash
        for attempt in range(_MANIFEST_REPLACE_RETRIES):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if attempt == _MANIFEST_REPLACE_RETRIES - 1:
                    raise
                time.sleep(_MANIFEST_REPLACE_WAIT_S)
    finally:
        tmp.unlink(missing_ok=True)  # G12-21: no temp file left behind when the write fails


class _ManifestBatch:
    """One processed field's manifest held in memory for one stage (G01-13)."""

    def __init__(self, root: Path, draft_id: int, manifest: dict[str, Any]) -> None:
        self.root = root
        self.draft_id = int(draft_id)
        self.manifest = manifest
        self.dirty = False
        self._by_id: dict[int, int] | None = None
        self._by_path: dict[str, int] | None = None

    def copy_manifest(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.manifest))

    def replace_manifest(self, payload: dict[str, Any]) -> None:
        # the same values a write + read of the file would give (JSON round trip)
        self.manifest = json.loads(json.dumps(payload))
        self.dirty = True
        self._by_id = self._by_path = None

    def files(self) -> list[Any]:
        files = self.manifest.get("files")
        if not isinstance(files, list):
            files = []
            self.manifest["files"] = files
            self._by_id = self._by_path = None
        return files

    def find_index(self, *, file_path: str | None, obs_file_id: int | None) -> int | None:
        """Same entry as :func:`_find_manifest_file_index` (the first entry matching the id or the
        resolved path), from indexes built once per batch."""
        files = self.files()
        if self._by_id is None or self._by_path is None:
            self._by_id, self._by_path = {}, {}
            for i, entry in enumerate(files):
                if not isinstance(entry, dict):
                    continue
                oid = _optional_int(entry.get("obs_file_id"))
                if oid is not None:
                    self._by_id.setdefault(int(oid), i)
                fp = _optional_str(entry.get("file_path")) or ""
                self._by_path.setdefault(str(Path(fp).resolve()), i)
        hits: list[int] = []
        if obs_file_id is not None and int(obs_file_id) in self._by_id:
            hits.append(self._by_id[int(obs_file_id)])
        want_fp = _optional_str(file_path)
        if want_fp:
            i_fp = self._by_path.get(str(Path(want_fp).resolve()))
            if i_fp is not None:
                hits.append(i_fp)
        return min(hits) if hits else None

    def flush(self) -> None:
        if not self.dirty:
            return
        payload = dict(self.manifest)
        payload["updated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _write_manifest_file(self.root / _MANIFEST_NAME, payload)
        self.dirty = False


#: Active manifest batches of this process, by resolved field folder.
_MANIFEST_BATCHES: dict[str, _ManifestBatch] = {}


@contextmanager
def manifest_batch(db: Any, draft_id: int | None) -> Iterator[None]:
    """Hold the field's manifest in memory for one stage and write it once at the end (G01-13).

    Every manifest read / write of this process for the field goes through the in-memory copy while
    the batch is open (readers get a fresh copy); per-frame ``files[]`` updates use indexes by
    ``obs_file_id`` and resolved path. The file is written once, atomically, when the batch ends -
    also when the stage raises (the updates made so far are kept, as with per-call writes). Nested
    batches of the same field join the outer one. The stage must be the only writer of the manifest
    (other processes do not see the batch). No-op without a database / draft id / field folder."""
    root = resolve_draft_dir_for_id(db, int(draft_id)) if (db is not None and draft_id is not None) else None
    key = str(root) if root is not None else ""
    if root is None or key in _MANIFEST_BATCHES:
        yield
        return
    batch = _ManifestBatch(root, int(draft_id), load_or_init_manifest(root, int(draft_id)))
    _MANIFEST_BATCHES[key] = batch
    try:
        yield
    finally:
        del _MANIFEST_BATCHES[key]
        batch.flush()


def manifest_batched(fn: Any) -> Any:
    """Decorator: run ``fn`` inside :func:`manifest_batch` for its ``db`` and ``draft_id`` arguments."""
    sig = inspect.signature(fn)

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        bound = sig.bind_partial(*args, **kwargs).arguments
        with manifest_batch(bound.get("db"), bound.get("draft_id")):
            return fn(*args, **kwargs)

    return wrapper


def load_or_init_manifest(root: Path | str, draft_id: int) -> dict[str, Any]:
    manifest = load_draft_manifest(root)
    if manifest:
        return manifest
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "draft_id": int(draft_id),
        "calibration_mode": CALIBRATION_MODE_VYVAR,
        "files": [],
    }


def patch_draft_manifest(
    root: Path | str,
    draft_id: int,
    *,
    rig: dict[str, Any] | None = None,
    paths: dict[str, Any] | None = None,
    status: str | None = None,
    center_by_setup: dict[str, Any] | None = None,
    files: list[dict[str, Any]] | None = None,
    final_observation_id: str | None = None,
    is_calibrated: int | None = None,
    observation_start_jd: float | None = None,
    calibration_mode: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Merge fields into manifest and persist (manifest-direct write API)."""
    manifest = load_or_init_manifest(root, int(draft_id))
    if rig is not None:
        manifest["rig"] = dict(rig)
    if paths is not None:
        manifest["paths"] = dict(paths)
    if status is not None:
        manifest["status"] = str(status)
    if center_by_setup is not None:
        manifest["center_by_setup"] = dict(center_by_setup)
    if files is not None:
        manifest["files"] = list(files)
    if final_observation_id is not None:
        manifest["final_observation_id"] = str(final_observation_id) if final_observation_id else None
    if is_calibrated is not None:
        manifest["is_calibrated"] = int(is_calibrated)
    if observation_start_jd is not None:
        manifest["observation_start_jd"] = observation_start_jd
    if calibration_mode is not None:
        manifest["calibration_mode"] = str(calibration_mode)
    if extra:
        manifest.update(extra)
    return _persist_manifest(root, int(draft_id), manifest)


def fetch_obs_draft_row_manifest(db: Any, draft_id: int) -> dict[str, Any] | None:
    """Load draft manifest-shaped row from manifest (sole store)."""
    root = resolve_draft_dir_for_id(db, int(draft_id))
    if root is None:
        return None
    manifest = load_draft_manifest(root)
    if not manifest:
        return None
    row = obs_draft_row_from_manifest(manifest, int(draft_id))
    row["ARCHIVE_PATH"] = row.get("ARCHIVE_PATH") or str(root)
    return row


def create_draft_manifest(
    archive_root: Path | str,
    data: dict[str, Any],
) -> int:
    """Allocate draft id and write initial manifest (no draft manifest SQL).

    The id is above ``min_id_floor`` and every field folder. Concurrent imports must go through
    ``VyvarDatabase.create_draft``, which runs this inside one DB write transaction (NEW-3)."""
    did = allocate_next_draft_id(archive_root, floor=int(data.get("min_id_floor") or 0))
    mode = str(data.get("calibration_mode") or CALIBRATION_MODE_VYVAR)
    jd = _optional_float(data.get("observation_start_jd"))
    target = processed_field_target_name(data.get("object_name"), data.get("source_folder_name"))
    draft_dir = make_processed_field_folder(
        archive_root, target, processed_field_night_date(jd, _optional_float(data.get("site_lon_deg")))
    )
    name = draft_dir.name
    is_cal = 1 if bool(data.get("is_calibrated", False)) else 0
    patch_draft_manifest(
        draft_dir,
        did,
        rig={
            "equipment_id": _optional_int(data.get("id_equipments")),
            "telescope_id": _optional_int(data.get("id_telescope")),
            "location_id": _optional_int(data.get("id_location")),
            "scanning_id": _optional_int(data.get("id_scanning")),
        },
        paths={"archive": str(draft_dir)},
        status="INGESTED",
        observation_start_jd=jd,
        is_calibrated=is_cal,
        calibration_mode=mode,
        files=[],
        extra={"name": name, "target": target},
    )
    return int(did)


def set_draft_manifest_files(
    db: Any,
    draft_id: int,
    files: list[dict[str, Any]],
) -> None:
    """Replace manifest ``files[]`` from ingest evidence dicts."""
    root = resolve_draft_dir_for_id(db, int(draft_id))
    if root is None:
        raise ValueError(f"Draft {draft_id}: archive directory not found for manifest files write")
    entries: list[dict[str, Any]] = []
    for idx, item in enumerate(files):
        obs_id = _optional_int(item.get("obs_file_id")) or (idx + 1)
        entries.append(_ingest_item_to_manifest_entry(item, obs_file_id=int(obs_id)))
    patch_draft_manifest(root, int(draft_id), files=entries)


def _find_manifest_file_index(
    files: list[Any],
    *,
    file_path: str | None = None,
    obs_file_id: int | None = None,
) -> int | None:
    want_id = int(obs_file_id) if obs_file_id is not None else None
    want_fp = _optional_str(file_path)
    want_fp_key = str(Path(want_fp).resolve()) if want_fp else None
    for i, entry in enumerate(files):
        if not isinstance(entry, dict):
            continue
        if want_id is not None and _optional_int(entry.get("obs_file_id")) == want_id:
            return i
        if want_fp_key:
            fp = _optional_str(entry.get("file_path")) or ""
            if fp == want_fp or str(Path(fp).resolve()) == want_fp_key:
                return i
    return None


def update_manifest_file_entry(
    db: Any,
    draft_id: int,
    *,
    obs_file_id: int | None = None,
    file_path: str | None = None,
    qc: dict[str, Any] | None = None,
    inspection: dict[str, Any] | None = None,
    imagetyp: str | None = None,
    filter_name: str | None = None,
    is_calibrated: int | None = None,
    calib_type: str | None = None,
    calib_flags: str | None = None,
    cal_stage: dict[str, Any] | None = None,
) -> bool:
    """Patch one ``files[]`` entry in the manifest."""
    root = resolve_draft_dir_for_id(db, int(draft_id))
    if root is None:
        return False
    batch = _MANIFEST_BATCHES.get(str(root))
    if batch is not None:
        files = batch.files()
        idx = batch.find_index(file_path=file_path, obs_file_id=obs_file_id)
    else:
        manifest = load_or_init_manifest(root, int(draft_id))
        files = manifest.get("files")
        if not isinstance(files, list):
            files = []
        idx = _find_manifest_file_index(files, file_path=file_path, obs_file_id=obs_file_id)
    if idx is None:
        return False
    entry = dict(files[idx]) if isinstance(files[idx], dict) else {}
    if imagetyp is not None:
        entry["imagetyp"] = imagetyp
    if filter_name is not None:
        entry["filter"] = filter_name
    if is_calibrated is not None:
        entry["is_calibrated"] = int(is_calibrated)
    if calib_type is not None:
        entry["calib_type"] = calib_type
    if calib_flags is not None:
        entry["calib_flags"] = calib_flags
    if qc:
        cur_qc = entry.get("qc") if isinstance(entry.get("qc"), dict) else {}
        cur_qc.update(qc)
        entry["qc"] = cur_qc
    if inspection:
        cur_insp = entry.get("inspection") if isinstance(entry.get("inspection"), dict) else {}
        cur_insp.update(inspection)
        entry["inspection"] = cur_insp
    if cal_stage:
        entry["cal_stage"] = str(cal_stage.get("cal_stage") or "")
        if cal_stage.get("cal_datasum") is not None:
            entry["cal_datasum"] = str(cal_stage["cal_datasum"])
        if cal_stage.get("cal_stage_ut") is not None:
            entry["cal_stage_ut"] = str(cal_stage["cal_stage_ut"])
        if cal_stage.get("cal_pstbg") is not None:
            try:
                entry["cal_pstbg"] = float(cal_stage["cal_pstbg"])
            except (TypeError, ValueError):
                pass
    files[idx] = entry
    if batch is not None:
        batch.dirty = True
        return True
    patch_draft_manifest(root, int(draft_id), files=files)
    return True


def set_manifest_light_qc_decisions(db: Any, draft_id: int, decisions: dict[str, str | None]) -> int:
    """Frame QC decision per LIGHT (``frame_qc``): ``is_rejected`` 1 with ``qc_reason``
    when the reason is a string, 0 when None. Lights not in ``decisions`` are kept
    (``is_rejected`` 0). Returns the number of rejected lights."""
    root = resolve_draft_dir_for_id(db, int(draft_id))
    if root is None:
        return 0
    manifest = load_or_init_manifest(root, int(draft_id))
    files = manifest.get("files")
    if not isinstance(files, list):
        return 0
    n_rej = 0
    for i, entry in enumerate(files):
        if not isinstance(entry, dict):
            continue
        if str(entry.get("imagetyp") or "").strip().lower() != "light":
            continue
        reason = decisions.get(str(entry.get("file_path") or ""))
        insp = dict(entry.get("inspection") if isinstance(entry.get("inspection"), dict) else {})
        insp["is_rejected"] = 1 if reason else 0
        insp["qc_reason"] = str(reason) if reason else None
        n_rej += 1 if reason else 0
        entry = dict(entry)
        entry["inspection"] = insp
        files[i] = entry
    patch_draft_manifest(root, int(draft_id), files=files)
    return n_rej


def resolve_calibration_mode(
    *,
    draft_id: int | None = None,
    db: Any = None,
    archive_path: Path | str | None = None,
) -> str:
    """Resolve calibration_mode from draft manifest, else default."""
    if db is not None and draft_id is not None:
        row = fetch_obs_draft_row_manifest(db, int(draft_id))
        if row:
            mode = row.get("CALIBRATION_MODE") or row.get("calibration_mode")
            if mode:
                return str(mode)
        root = resolve_draft_dir_for_id(db, int(draft_id))
        if root is not None:
            mode = load_draft_manifest(root).get("calibration_mode")
            if mode:
                return str(mode)
    if archive_path is not None:
        mode = load_draft_manifest(archive_path).get("calibration_mode")
        if mode:
            return str(mode)
    return CALIBRATION_MODE_VYVAR


def backfill_draft_manifest_from_db(db: Any, draft_id: int) -> Path | None:
    """Ensure manifest exists for a draft archive dir (legacy SQL source retired)."""
    root = resolve_draft_dir_for_id(db, int(draft_id))
    if root is None:
        return None
    manifest_path = root / _MANIFEST_NAME
    if manifest_path.is_file():
        return manifest_path
    return None


def record_observer_location_provenance(
    *,
    archive_path: Path | str,
    draft_id: int,
    resolved: Any,
) -> Path:
    """Persist resolved observer site into draft_manifest.json.

    When the resolved site differs from ``rig.location_id``, update the
    manifest and log ``[SITE] manifest location changed <old> -> <new>``.
    """
    from infolog import log_milestone  # noqa: PLC0415

    prov = resolved.as_provenance_dict() if hasattr(resolved, "as_provenance_dict") else dict(resolved)
    root = draft_archive_root(archive_path)
    manifest = load_draft_manifest(root)
    extra = {"observer_location": prov}
    new_id = _optional_int(prov.get("location_id"))
    rig = manifest.get("rig") if isinstance(manifest.get("rig"), dict) else None
    if isinstance(rig, dict):
        old_id = _optional_int(rig.get("location_id"))
        rig_out = dict(rig)
        if new_id is not None and old_id != new_id:
            rig_out["location_id"] = int(new_id)
            log_milestone(
                f"[SITE] manifest location changed {old_id} -> {new_id}"
            )
        # merge into the manifest: a full rewrite dropped every key it did not know (name, target)
        return patch_draft_manifest(root, int(draft_id), rig=rig_out, extra=extra)
    return patch_draft_manifest(
        root,
        int(draft_id),
        rig={"location_id": int(new_id)} if new_id is not None else None,
        extra=extra,
    )


def record_draft_calibration_provenance(
    *,
    archive_path: Path | str,
    draft_id: int,
    calibration_mode: str,
) -> Path:
    """Persist calibration_mode to draft manifest."""
    root = draft_archive_root(archive_path)
    manifest = load_or_init_manifest(root, int(draft_id))
    return patch_draft_manifest(
        root,
        int(draft_id),
        calibration_mode=str(calibration_mode),
        rig=manifest.get("rig") if isinstance(manifest.get("rig"), dict) else None,
        paths=manifest.get("paths") if isinstance(manifest.get("paths"), dict) else None,
        status=manifest.get("status"),
        files=manifest.get("files") if isinstance(manifest.get("files"), list) else None,
        extra={"observer_location": manifest["observer_location"]}
        if isinstance(manifest.get("observer_location"), dict)
        else None,
    )


def manifest_db_parity_errors(db: Any, draft_id: int) -> list[str]:
    """Retired DB parity check: verify manifest exists for anchor drafts."""
    root = resolve_draft_dir_for_id(db, int(draft_id))
    if root is None:
        return [f"draft_id={int(draft_id)}: draft archive dir missing"]
    if not load_draft_manifest(root):
        return [f"draft_id={int(draft_id)}: draft_manifest.json missing"]
    return []


def iter_draft_archive_dirs(archive_root: Path | str) -> list[tuple[int, Path]]:
    """List ``(id, folder)`` of the processed fields under ``Archive/Drafts`` (sorted by id desc).

    PROCESSED-FIELD-NAME-2026-10: a folder is ``<TARGET>_<YYYYMMDD>_<NNN>`` (or the legacy
    ``draft_NNNNNN``); its id is the ``draft_id`` of its ``draft_manifest.json`` - the folder name
    carries no id. A legacy folder without a manifest still counts by its name."""
    drafts_dir = Path(archive_root).expanduser().resolve() / "Drafts"
    if not drafts_dir.is_dir():
        return []
    out: list[tuple[int, Path]] = []
    for child in drafts_dir.iterdir():
        if not child.is_dir():
            continue
        did: int | None = None
        mf = child / _MANIFEST_NAME
        if mf.is_file():
            try:
                raw = json.loads(mf.read_text(encoding="utf-8"))
                did = int(raw.get("draft_id"))
            except Exception as exc:  # noqa: BLE001
                logging.warning("[MANIFEST] unreadable %s (folder skipped unless named draft_NNNNNN): %s", mf, exc)
                did = None
        if did is None and child.name.lower().startswith("draft_"):
            try:
                did = int(child.name.split("_", 1)[-1])
            except ValueError:
                did = None
        if did is not None:
            out.append((int(did), child.resolve()))
    out.sort(key=lambda t: t[0], reverse=True)
    return out


def draft_dir_for_id(archive_root: Path | str, draft_id: int) -> Path:
    """Folder of processed field ``draft_id`` under ``Archive/Drafts`` (any name, by its manifest);
    the legacy ``draft_NNNNNN`` path when none exists yet."""
    root = Path(archive_root).expanduser().resolve()
    legacy = root / "Drafts" / f"draft_{int(draft_id):06d}"
    if legacy.is_dir():
        return legacy
    key = (str(root), int(draft_id))
    hit = _DRAFT_DIR_CACHE.get(key)
    if hit is not None:
        mtime_now = _draft_dir_still_holds(hit[0], int(draft_id), hit[1])
        if mtime_now is not None:
            _DRAFT_DIR_CACHE[key] = (hit[0], mtime_now)
            return hit[0]
    _DRAFT_DIR_CACHE.pop(key, None)
    for found_id, path in iter_draft_archive_dirs(root):
        if found_id == int(draft_id):
            _DRAFT_DIR_CACHE[key] = (path, _manifest_mtime_ns(path))
            return path
    return legacy


#: G12-20: (archive root, draft id) -> (field folder, manifest mtime_ns) of this process. A hit is
#: checked against the folder (manifest still there; when the file changed, its draft_id is read
#: again), so a deleted or re-used folder name is never returned for another id.
_DRAFT_DIR_CACHE: dict[tuple[str, int], tuple[Path, int | None]] = {}


def _manifest_mtime_ns(folder: Path) -> int | None:
    try:
        return int((folder / _MANIFEST_NAME).stat().st_mtime_ns)
    except OSError:
        return None


def _draft_dir_still_holds(folder: Path, draft_id: int, mtime_ns: int | None) -> int | None:
    """The manifest's current mtime_ns when ``folder`` still holds field ``draft_id``, else None."""
    now = _manifest_mtime_ns(folder)
    if now is None:
        return None
    if mtime_ns is not None and now == mtime_ns:
        return now
    try:
        raw = json.loads((folder / _MANIFEST_NAME).read_text(encoding="utf-8"))
        return now if int(raw.get("draft_id")) == int(draft_id) else None
    except Exception:  # noqa: BLE001
        return None


def obs_draft_row_from_manifest(manifest: dict[str, Any], draft_id: int) -> dict[str, Any]:
    """Build an draft manifest-shaped row dict from ``draft_manifest.json`` (UI display source)."""
    rig = manifest.get("rig") if isinstance(manifest.get("rig"), dict) else {}
    paths = manifest.get("paths") if isinstance(manifest.get("paths"), dict) else {}
    centres = manifest.get("center_by_setup") if isinstance(manifest.get("center_by_setup"), dict) else {}
    row: dict[str, Any] = {
        "ID": int(draft_id),
        "ID_EQUIPMENTS": _optional_int(rig.get("equipment_id")),
        "ID_TELESCOPE": _optional_int(rig.get("telescope_id")),
        "ID_LOCATION": _optional_int(rig.get("location_id")),
        "ID_SCANNING": _optional_int(rig.get("scanning_id")),
        "OBSERVATIONSTARTJD": _optional_float(manifest.get("observation_start_jd")),
        # S6-FIX CENTRE-PER-SETUP: {setup: {ra_deg, de_deg, source}}; a legacy field-level ``center`` is ignored
        "CENTER_BY_SETUP": dict(centres),
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


def collect_manifest_obs_file_rows(
    archive_root: Path | str,
    *,
    draft_id: int | None = None,
    observation_id: str | None = None,
) -> list[dict[str, Any]]:
    """Flatten manifest ``files[]`` into manifest file-row-shaped rows (UI display)."""
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


