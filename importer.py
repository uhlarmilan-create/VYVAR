"""Session importer for VYVAR (file-first workflow)."""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.visualization import ZScaleInterval

import math

from config import AppConfig
from database import DraftTechnicalMetadataError, VyvarDatabase
from fits_suffixes import path_suffix_is_fits
from infolog import log_event
from pipeline import extract_fits_metadata, fits_metadata_from_primary_header
from utils import plate_scale_arcsec_per_pixel


@dataclass(slots=True)
class ImportResult:
    draft_id: int | None
    observation_id: str | None
    lights_path: str
    dark_path: str
    flat_path: str
    archive_path: str
    warnings: list[str]


@dataclass(slots=True)
class CalibrationStatus:
    kind: str
    path: str | None
    status: str  # ok | expired | missing
    last_modified_utc: str | None
    age_days: int | None
    validity_days: int
    message: str


@dataclass(slots=True)
class MasterGenerationResult:
    dark_master_path: str | None
    flat_master_path: str | None
    messages: list[str]


@dataclass(slots=True)
class SmartScanRow:
    type: str  # Lights | Darks | Flats
    status: str  # ok | missing | empty | raw | master | library | expired | draft
    count: int
    parameters: str
    details: str | None = None


def observation_group_key(filter_name: str, exposure_s: float, binning: int) -> str:
    """Unique observation subgroup: (FILTER, EXPTIME, XBINNING) — binning from FITS via metadata."""
    flt = _filter_name_for_calibration_library_flat(filter_name)
    try:
        e = float(exposure_s)
    except (TypeError, ValueError):
        e = 0.0
    b = max(1, int(binning))
    return f"{flt}|{e:g}|{b}"


def observation_group_folder_name(group_key: str) -> str:
    """Filesystem-safe folder under ``lights`` (replaces ``|``)."""
    s = group_key.replace("|", "_").replace("/", "_").replace("\\", "_")
    s = "".join(c if c.isalnum() or c in "._-" else "_" for c in s)
    return s[:120] if len(s) > 120 else s


def _calibration_library_search_roots(calibration_library_root: Path) -> list[Path]:
    """Prefer ``<library>/Masters`` then the library root (matches multi-obs master layout)."""
    roots: list[Path] = []
    seen: set[str] = set()
    for r in (calibration_library_root / "Masters", calibration_library_root):
        try:
            rp = r.resolve()
        except OSError:
            continue
        key = str(rp).casefold()
        if key in seen:
            continue
        seen.add(key)
        if rp.is_dir():
            roots.append(rp)
    return roots


@dataclass(slots=True)
class SmartImportPlan:
    source_root: str
    lights_files: list[str]
    dark_files: list[str]
    flat_files: list[str]
    lights_first_fits: str | None
    metadata: dict[str, Any] | None
    scan_rows: list[SmartScanRow]
    dark_master: str | None
    flat_master: str | None
    masterflat_by_filter: dict[str, str | None]
    masterflat_status: dict[str, str]
    missing_flat_filters: list[str]
    masterdark_status: str
    quick_look: bool
    detected_filters: list[str]
    warnings: list[str]
    #: (FILTER|EXPTIME|BINNING) → group detail dict (paths, scale, …)
    observation_groups: dict[str, dict[str, Any]] = field(default_factory=dict)
    masterflat_by_obs_key: dict[str, str | None] = field(default_factory=dict)
    dark_master_by_obs_key: dict[str, str | None] = field(default_factory=dict)
    missing_obs_keys: list[str] = field(default_factory=list)
    flat_fallback_prompts: list[dict[str, Any]] = field(default_factory=list)


def _list_fits_files(folder: Path) -> list[Path]:
    """List FITS files in folder without duplicates (Windows-safe)."""
    out: list[Path] = []
    seen: set[str] = set()
    for fp in folder.iterdir():
        if not fp.is_file():
            continue
        if not path_suffix_is_fits(fp):
            continue
        key = str(fp.resolve()).casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(fp)
    return sorted(out)


def _is_empty_or_missing(folder: Path) -> bool:
    return (not folder.exists()) or (not folder.is_dir()) or (len(_list_fits_files(folder)) == 0)


def _mtime_utc(path: Path) -> datetime:
    ts = os.path.getmtime(path)
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def _parse_date_obs(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return None
    text = str(value).strip()
    # Common formats: 2026-03-27T21:15:03.123, 2026-03-27T21:15:03Z
    text = text.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _fits_capture_date_yyyymmdd(path: Path) -> str:
    """Get capture date YYYYMMDD from DATE-OBS across frames (earliest)."""
    try:
        with fits.open(path, memmap=False) as hdul:
            header = hdul[0].header
        dt = _parse_date_obs(header.get("DATE-OBS") or header.get("DATEOBS"))
        if dt is None:
            return datetime.now(timezone.utc).strftime("%Y%m%d")
        return dt.strftime("%Y%m%d")
    except Exception:  # noqa: BLE001
        return datetime.now(timezone.utc).strftime("%Y%m%d")


def _earliest_capture_datetime_utc(files: list[Path]) -> datetime | None:
    """Earliest DATE-OBS across source files; fallback to earliest mtime."""
    if not files:
        return None
    best: datetime | None = None
    for fp in files:
        dt: datetime | None = None
        try:
            with fits.open(fp, memmap=True) as hdul:
                hdr = hdul[0].header
            dt = _parse_date_obs(hdr.get("DATE-OBS") or hdr.get("DATEOBS"))
        except Exception:  # noqa: BLE001
            dt = None
        if dt is None:
            try:
                dt = _mtime_utc(fp)
            except Exception:  # noqa: BLE001
                dt = None
        if dt is None:
            continue
        if best is None or dt < best:
            best = dt
    return best


def _filter_name_for_calibration_library_flat(flt: str | None) -> str:
    """Normalize FILTER for CALIBRATION_LIBRARY flat rows (must match import matching)."""
    s = str(flt or "").strip()
    if not s or s.lower() in {"unknown", "none", "nan"}:
        return "NoFilter"
    return s


def _register_master_path_in_calibration_library(
    db: VyvarDatabase | None,
    *,
    kind: str,
    path: Path,
    ncombine: int | None = None,
    id_equipments: int | None = None,
    id_telescope: int | None = None,
) -> None:
    if db is None:
        return
    try:
        meta = extract_fits_metadata(path, db=db)
        k = str(kind).strip().lower()
        if k not in ("dark", "flat"):
            return
        flt = "" if k == "dark" else _filter_name_for_calibration_library_flat(str(meta.get("filter", "")))
        db.register_calibration_library_entry(
            kind=k,
            file_path=path.resolve(),
            xbinning=int(meta.get("binning", 1) or 1),
            exptime=float(meta.get("exposure", 0.0)),
            ccd_temp=float(meta["temp"]) if meta.get("temp") is not None else None,
            filter_name=flt,
            gain=int(meta.get("gain", 0) or 0),
            ncombine=ncombine,
            id_equipments=id_equipments,
            id_telescope=id_telescope,
        )
    except Exception:  # noqa: BLE001
        pass


def _sanitize_token(value: Any) -> str:
    text = str(value).strip()
    text = text.replace(" ", "")
    text = text.replace("/", "-")
    text = text.replace("\\", "-")
    return text or "Unknown"


def _format_exp_seconds(exposure: float) -> str:
    if abs(exposure - round(exposure)) < 1e-6:
        return str(int(round(exposure)))
    return f"{exposure:.2f}".rstrip("0").rstrip(".")


def _format_temp(temp: float) -> str:
    # Keep sign, avoid extra dots in filename
    return f"{temp:.1f}".replace(".", "p")


def _format_temp_deg_for_name(temp: float) -> str:
    """Format temperature token for filenames like -15deg."""
    if abs(temp - round(temp)) < 1e-6:
        return f"{int(round(temp))}deg"
    return f"{temp:.1f}".rstrip("0").rstrip(".") + "deg"


def _looks_like_master(fp: Path) -> bool:
    name = fp.name.upper()
    if name.startswith(("MD_", "MF_", "MASTERDARK", "MASTERFLAT", "DARK_", "FLAT_")):
        return True
    try:
        with fits.open(fp, memmap=False) as hdul:
            hdr = hdul[0].header
        if "NCOMBINE" in hdr:
            return True
        hist = hdr.get("HISTORY")
        if hist and "MASTER" in str(hist).upper():
            return True
    except Exception:  # noqa: BLE001
        return False
    return False


def _first_fits_in_dir(folder: Path) -> Path | None:
    files = _list_fits_files(folder)
    return files[0] if files else None


def _classify_imagetyp(value: Any) -> str:
    t = str(value or "").lower()
    if "light" in t:
        return "light"
    if "dark" in t:
        return "dark"
    if "flat" in t:
        return "flat"
    return "unknown"


def _collect_fits_by_type(
    source_root: Path,
    db: VyvarDatabase | None = None,
) -> dict[str, list[Path]]:
    """Recursively collect FITS files and classify by IMAGETYP/FRAME per file.

    With ``db``, uses ``FITS_HEADER_CACHE`` and refreshes missing/stale rows in one transaction.
    """
    out: dict[str, list[Path]] = {"light": [], "dark": [], "flat": [], "unknown": []}
    seen: set[str] = set()
    pending_cache: list[tuple[Path, int, float, dict[str, Any], str, str | None]] = []
    _force_phys: float | None = None
    try:
        _cfp = AppConfig().force_pixel_size_um
        if _cfp is not None:
            _cfpv = float(_cfp)
            if _cfpv > 0 and math.isfinite(_cfpv):
                _force_phys = _cfpv
    except Exception:  # noqa: BLE001
        _force_phys = None
    for fp in source_root.rglob("*"):
        if not fp.is_file():
            continue
        if not path_suffix_is_fits(fp):
            continue
        key = str(fp.resolve()).casefold()
        if key in seen:
            continue
        seen.add(key)
        cls = "unknown"
        try:
            st = fp.stat()
        except OSError:
            out.setdefault(cls, []).append(fp)
            continue
        row = None
        if db is not None:
            row = db.fits_header_cache_get_if_fresh(
                fp, file_size=int(st.st_size), mtime=float(st.st_mtime)
            )
        if row is not None:
            imagetyp = str(row["IMAGETYP"] or "")
            cls = _classify_imagetyp(imagetyp)
        else:
            try:
                with fits.open(fp, memmap=False) as hdul:
                    hdr = hdul[0].header
                imagetyp_raw = str(
                    hdr.get("IMAGETYP") or hdr.get("FRAME") or hdr.get("IMTYPE") or ""
                )
                cls = _classify_imagetyp(imagetyp_raw)
                if db is not None:
                    meta = fits_metadata_from_primary_header(hdr, force_physical_pixel_um=_force_phys)
                    do = hdr.get("DATE-OBS") or hdr.get("DATEOBS")
                    date_obs = None if do in (None, "") else str(do)
                    pending_cache.append(
                        (fp, int(st.st_size), float(st.st_mtime), meta, imagetyp_raw, date_obs)
                    )
            except Exception:  # noqa: BLE001
                cls = "unknown"
        out.setdefault(cls, []).append(fp)
    if db is not None and pending_cache:
        db.fits_header_cache_upsert_batch(pending_cache)
    for k in out:
        out[k] = sorted(out[k])
    return out


def _read_filter(fp: Path, db: VyvarDatabase | None = None) -> str:
    if db is not None:
        hit = db.fits_header_cache_try_filter(fp)
        if hit is not None:
            return hit
    try:
        with fits.open(fp, memmap=False) as hdul:
            hdr = hdul[0].header
        flt = str(hdr.get("FILTER") or hdr.get("FILT") or "").strip()
        if not flt or flt.strip().lower() in {"unknown", "none", "nan"}:
            return "NoFilter"
        return flt
    except Exception:  # noqa: BLE001
        return "NoFilter"


def _master_kind_matches(fp: Path, kind: str) -> bool:
    """Ensure we don't pick Dark as Flat (and vice versa) from the library."""
    kind = kind.lower()
    name = fp.name.upper()
    # Strong filename hints
    if kind == "dark" and ("DARK" in name or name.startswith("MD_")):
        return True
    if kind == "flat" and ("FLAT" in name or name.startswith("MF_")):
        return True

    # Header hints
    try:
        with fits.open(fp, memmap=False) as hdul:
            hdr = hdul[0].header
        imagetyp = str(hdr.get("IMAGETYP") or hdr.get("FRAME") or hdr.get("IMTYPE") or "").lower()
        hist = str(hdr.get("HISTORY") or "").lower()
    except Exception:  # noqa: BLE001
        imagetyp = ""
        hist = ""

    if kind == "dark" and ("dark" in imagetyp or "dark" in hist):
        return True
    if kind == "flat" and ("flat" in imagetyp or "flat" in hist):
        return True

    return False


def _sanitize_filter_folder(name: str) -> str:
    """Sanitize FILTER value into a Windows-safe folder name."""
    text = (name or "").strip()
    if not text:
        return "NoFilter"
    # Remove spaces and normalize common separators
    text = text.replace(" ", "")
    text = text.replace("/", "-").replace("\\", "-")
    # Windows forbidden characters: <>:"/\|?*
    forbidden = '<>:"\\|?*'
    for ch in forbidden:
        text = text.replace(ch, "_")
    # Avoid trailing dots/spaces (invalid on Windows)
    text = text.rstrip(" .")
    # Avoid Windows reserved device names
    reserved = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "COM1",
        "COM2",
        "COM3",
        "COM4",
        "COM5",
        "COM6",
        "COM7",
        "COM8",
        "COM9",
        "LPT1",
        "LPT2",
        "LPT3",
        "LPT4",
        "LPT5",
        "LPT6",
        "LPT7",
        "LPT8",
        "LPT9",
    }
    if text.upper() in reserved:
        text = f"_{text}"

    # Keep it short to avoid long paths; add stable suffix if truncated
    if len(text) > 24:
        import hashlib

        suffix = hashlib.md5(text.encode("utf-8"), usedforsecurity=False).hexdigest()[:6]
        text = f"{text[:18]}_{suffix}"

    return text or "NoFilter"


def _safe_copy2(src: Path, dst: Path) -> None:
    """Copy with actionable error context (Windows path issues)."""
    try:
        shutil.copy2(src, dst)
    except OSError as exc:
        raise OSError(
            exc.errno,
            f"{exc.strerror} | src='{src}' dst='{dst}'",
        ) from exc


def _sanitize_windows_filename(name: str) -> str:
    """Sanitize a filename for Windows (keep extension if present)."""
    name = (name or "").strip()
    if not name:
        return "file.fits"
    forbidden = '<>:"\\|?*'
    for ch in forbidden:
        name = name.replace(ch, "_")
    # Avoid trailing dots/spaces
    name = name.rstrip(" .")
    # Avoid reserved device names (stem)
    stem, dot, ext = name.partition(".")
    reserved = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "COM1",
        "COM2",
        "COM3",
        "COM4",
        "COM5",
        "COM6",
        "COM7",
        "COM8",
        "COM9",
        "LPT1",
        "LPT2",
        "LPT3",
        "LPT4",
        "LPT5",
        "LPT6",
        "LPT7",
        "LPT8",
        "LPT9",
    }
    if stem.upper() in reserved:
        stem = f"_{stem}"
    if dot:
        return f"{stem}.{ext}"
    return stem


def _dst_path_with_length_limit(dst_dir: Path, original_name: str, *, max_path_len: int = 240) -> Path:
    """Return a destination path that stays within Windows path limits."""
    import hashlib

    safe_name = _sanitize_windows_filename(original_name)
    dst = dst_dir / safe_name
    if len(str(dst)) <= max_path_len:
        return dst

    stem = Path(safe_name).stem
    ext = Path(safe_name).suffix
    digest = hashlib.md5(safe_name.encode("utf-8"), usedforsecurity=False).hexdigest()[:8]
    short_stem = stem[: max(8, 40 - len(digest))].rstrip(" ._")
    candidate = dst_dir / f"{short_stem}_{digest}{ext}"
    if len(str(candidate)) <= max_path_len:
        return candidate

    # last resort: very short
    return dst_dir / f"{digest}{ext or '.fits'}"


def _params_string(meta: dict[str, Any], *, include_filter: bool) -> str:
    exp = float(meta.get("exposure", 0.0))
    gain = int(meta.get("gain", 0))
    temp = float(meta.get("temp", 0.0))
    binning = int(meta.get("binning", 1))
    flt = str(meta.get("filter", "Unknown"))
    base = f"Exp={exp:g}s, Gain={gain}G, Temp={temp:g}C, Bin={binning}"
    return f"{base}, Filter={flt}" if include_filter else base


def _find_matching_master_in_library(
    calibration_library_root: Path,
    *,
    kind: str,  # "dark" | "flat"
    exp: float,
    gain: int,
    binning: int,
    temp: float,
    flt: str | None,
    db: VyvarDatabase | None = None,
    search_roots: list[Path] | None = None,
    id_equipments: int | None = None,
    id_telescope: int | None = None,
) -> Path | None:
    if db is not None:
        flt_key = _filter_name_for_calibration_library_flat(flt) if kind == "flat" else ""
        try:
            hit = db.find_best_calibration_library_path(
                kind=kind,
                xbinning=int(binning),
                exptime=float(exp),
                ccd_temp=float(temp),
                filter_name=flt_key,
                gain=int(gain),
                prefer_unbinned_master=True,
                id_equipments=id_equipments,
                id_telescope=id_telescope,
            )
        except Exception:  # noqa: BLE001
            hit = None
        if hit:
            p_hit = Path(hit)
            if p_hit.is_file() and _master_kind_matches(p_hit, kind) and _looks_like_master(p_hit):
                return p_hit

    roots = search_roots if search_roots else [calibration_library_root]
    candidates: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for ext in ("*.fits", "*.fit", "*.fts", "*.FITS", "*.FIT", "*.FTS"):
            candidates.extend(root.rglob(ext))
    seen_c: set[str] = set()
    cand_u: list[Path] = []
    for fp in candidates:
        try:
            ck = str(fp.resolve()).casefold()
        except OSError:
            continue
        if ck in seen_c:
            continue
        seen_c.add(ck)
        cand_u.append(fp)
    candidates = cand_u
    if not candidates:
        return None

    kind_upper = kind.upper()
    best: Path | None = None
    best_mtime = -1.0

    for fp in candidates:
        name = fp.name.upper()
        if kind_upper not in name and not name.startswith(("MD_", "MF_")):
            # still allow if header indicates master, but prefer name match
            pass

        try:
            meta = extract_fits_metadata(fp, db=db)
        except Exception:  # noqa: BLE001
            continue

        if float(meta.get("exposure", -1.0)) != float(exp):
            continue
        if int(meta.get("gain", 0)) != int(gain):
            continue
        mb = int(meta.get("binning", 0) or 0)
        if mb != int(binning) and not (int(binning) > 1 and mb == 1):
            continue
        if abs(float(meta.get("temp", 0.0)) - float(temp)) > 0.5:
            continue
        if kind == "flat" and flt is not None:
            if str(meta.get("filter", "")).strip() != str(flt).strip():
                continue

        if not _looks_like_master(fp):
            continue
        if not _master_kind_matches(fp, kind):
            continue

        mtime = os.path.getmtime(fp)
        if mtime > best_mtime:
            best = fp
            best_mtime = mtime

    return best


def _age_days(path: Path) -> int | None:
    try:
        mtime = _mtime_utc(path)
    except OSError:
        return None
    now_utc = datetime.now(timezone.utc)
    return int((now_utc - mtime).total_seconds() // 86400)


def _find_best_masterflat_for_filter(
    calibration_library_root: Path,
    *,
    flt: str,
    binning: int,
    validity_days: int,
    db: VyvarDatabase | None = None,
    exp: float | None = None,
    gain: int | None = None,
    temp: float | None = None,
    search_roots: list[Path] | None = None,
    id_equipments: int | None = None,
    id_telescope: int | None = None,
) -> tuple[Path | None, str]:
    """Return best masterflat and a UI-friendly status string."""
    roots = search_roots or _calibration_library_search_roots(calibration_library_root)
    flt_norm = _filter_name_for_calibration_library_flat(flt)
    if (
        db is not None
        and exp is not None
        and gain is not None
        and temp is not None
    ):
        try:
            hit = db.find_best_calibration_library_path(
                kind="flat",
                xbinning=int(binning),
                exptime=float(exp),
                ccd_temp=float(temp),
                filter_name=flt_norm,
                gain=int(gain),
                prefer_unbinned_master=True,
                id_equipments=id_equipments,
                id_telescope=id_telescope,
            )
        except Exception:  # noqa: BLE001
            hit = None
        if hit:
            p_hit = Path(hit)
            if p_hit.is_file() and _master_kind_matches(p_hit, "flat") and _looks_like_master(p_hit):
                age = _age_days(p_hit)
                if age is not None and age <= validity_days:
                    return p_hit, f"MasterFlat (Filter {flt}): ✅ library DB ({age} days old)"
                if age is not None:
                    return (
                        None,
                        f"MasterFlat (Filter {flt}): ⚠️ library DB expirované ({age} dní) — vygeneruj nový",
                    )
                return p_hit, f"MasterFlat (Filter {flt}): ✅ library DB"

    candidates: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for ext in ("*.fits", "*.fit", "*.fts", "*.FITS", "*.FIT", "*.FTS"):
            candidates.extend(root.rglob(ext))
    seen_cf: set[str] = set()
    cand_cf: list[Path] = []
    for fp in candidates:
        try:
            ck = str(fp.resolve()).casefold()
        except OSError:
            continue
        if ck in seen_cf:
            continue
        seen_cf.add(ck)
        cand_cf.append(fp)
    candidates = cand_cf
    best_fresh: tuple[Path, int] | None = None  # (path, age_days)
    best_any: tuple[Path, int] | None = None

    for fp in candidates:
        try:
            meta = extract_fits_metadata(fp, db=db)
        except Exception:  # noqa: BLE001
            continue
        if str(meta.get("filter", "")).strip() != str(flt).strip():
            continue
        mb = int(meta.get("binning", -1))
        if mb != int(binning) and not (int(binning) > 1 and mb == 1):
            continue
        if not _looks_like_master(fp):
            continue
        if not _master_kind_matches(fp, "flat"):
            continue
        age = _age_days(fp)
        if age is None:
            continue
        if best_any is None or age < best_any[1]:
            best_any = (fp, age)
        if age <= validity_days:
            if best_fresh is None or age < best_fresh[1]:
                best_fresh = (fp, age)

    if best_fresh:
        fp, age = best_fresh
        return fp, f"MasterFlat (Filter {flt}): ✅ found ({age} days old)"
    if best_any:
        _fp, age = best_any
        return (
            None,
            f"MasterFlat (Filter {flt}): ⚠️ len expirovaný master ({age} dní) — vygeneruj nový",
        )
    return None, f"MasterFlat (Filter {flt}): ❌ MISSING!"


def _write_master_to_library(
    *,
    kind: str,  # "dark" | "flat"
    files: list[Path],
    calibration_library_root: Path,
    target_binning: int | None = None,
    db: VyvarDatabase | None = None,
    id_equipments: int | None = None,
    id_telescope: int | None = None,
) -> Path:
    if kind not in ("dark", "flat"):
        raise ValueError("kind must be 'dark' or 'flat'")
    if not files:
        raise ValueError("No files to build master from.")

    meta0 = extract_fits_metadata(files[0], db=db)
    exp_s = _format_exp_seconds(float(meta0["exposure"]))
    gain = int(meta0.get("gain", 0))
    temp_token = _format_temp_deg_for_name(float(meta0["temp"]))
    raw_binning = int(meta0["binning"])
    # CalibrationLibrary stores a **native** stack of calibration frames. Software resampling to match
    # light XBINNING happens only in :func:`calibration.get_processed_master` at calibrate time.
    _ = target_binning  # deprecated: do not bin-down masters in the library to match lights
    binning = raw_binning
    flt = _sanitize_token(meta0.get("filter", "Unknown"))
    date_yyyymmdd = min(_fits_capture_date_yyyymmdd(fp) for fp in files)

    type_token = "Dark" if kind == "dark" else "Flat"
    filter_token = "Dark" if kind == "dark" else flt

    filename = f"{type_token}_{exp_s}s_{filter_token}_{gain}G_{temp_token}_Bin{binning}_{date_yyyymmdd}.fits"
    out_root = calibration_library_root
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / filename

    # If exact target already exists, do not recreate.
    if out_path.exists():
        return out_path

    files_for_stack = files
    if kind == "dark":
        master, header = _generate_master_dark(files_for_stack)
    else:
        master, header = _generate_master_flat(files_for_stack)

    # Binning keywords reflect the **calibration frames** used (prefer 1×1 for full-resolution masters).
    header["BINNING"] = binning
    header["XBINNING"] = binning
    header["YBINNING"] = binning
    header["VY_MBLIB"] = (
        1,
        "VYVAR: native stack in CalibrationLibrary; resample to light XBINNING at calibrate",
    )

    if kind == "dark":
        header["HISTORY"] = "VYVAR: Master Dark (mean stack)"
    else:
        header["HISTORY"] = "VYVAR: Master Flat (median stack; norm at calibrate)"
        header["VYFLNRD"] = (
            1,
            "Median normalization deferred to calibrate after resample to light binning",
        )
    dt_src = _earliest_capture_datetime_utc(files)
    if dt_src is not None:
        dt_iso = dt_src.strftime("%Y-%m-%dT%H:%M:%S")
        header["DATE-OBS"] = (dt_iso, "Earliest source raw capture time (UTC)")
        header["VY_CDATE"] = (dt_iso, "VYVAR source capture datetime (UTC)")
    header["NCOMBINE"] = len(files)
    fits.writeto(out_path, master.astype(np.float32), header=header, overwrite=True)
    if kind == "dark":
        try:
            from config import AppConfig
            from photometry import write_dark_bpm_json

            _sig = float(AppConfig().bpm_dark_mad_sigma)
            _nb = int(header.get("XBINNING") or header.get("BINNING") or binning)
            write_dark_bpm_json(out_path, master, mad_sigma=_sig, native_binning=_nb)
        except Exception:  # noqa: BLE001
            pass
    _register_master_path_in_calibration_library(
        db,
        kind=kind,
        path=out_path,
        ncombine=int(header.get("NCOMBINE", len(files))),
        id_equipments=id_equipments,
        id_telescope=id_telescope,
    )
    return out_path


def _find_existing_master_for_raw_set(
    calibration_library_root: Path,
    *,
    kind: str,  # "dark" | "flat"
    sample_file: Path,
    target_binning: int | None = None,
    db: VyvarDatabase | None = None,
    id_equipments: int | None = None,
    id_telescope: int | None = None,
) -> Path | None:
    """Avoid duplicate master creation: find existing master in library by params."""
    try:
        meta0 = extract_fits_metadata(sample_file, db=db)
    except Exception:  # noqa: BLE001
        return None

    exp = float(meta0.get("exposure", 0.0))
    gain = int(meta0.get("gain", 0))
    raw_binning = int(meta0.get("binning", 1))
    _ = target_binning
    binning = raw_binning
    temp = float(meta0.get("temp", 0.0))
    flt = str(meta0.get("filter", "")).strip() if kind == "flat" else None

    cr = Path(calibration_library_root)
    return _find_matching_master_in_library(
        cr,
        kind=kind,
        exp=exp,
        gain=gain,
        binning=binning,
        temp=temp,
        flt=flt,
        db=db,
        search_roots=_calibration_library_search_roots(cr),
        id_equipments=id_equipments,
        id_telescope=id_telescope,
    )


def smart_scan_source(
    *,
    source_root: str | Path,
    calibration_library_root: str | Path,
    masterdark_validity_days: int,
    masterflat_validity_days: int,
    db: VyvarDatabase | None = None,
    id_equipments: int | None = None,
    id_telescope: int | None = None,
) -> SmartImportPlan:
    """Scan source for lights/darks/flats and decide calibration paths.

    With ``db``, refreshes ``FITS_HEADER_CACHE`` (header metadata only) for fast rescans; no observation/draft writes.
    """
    root = Path(source_root)

    scan_rows: list[SmartScanRow] = []
    warnings: list[str] = []

    if not root.exists() or not root.is_dir():
        scan_rows.append(SmartScanRow("Lights", "missing", 0, ""))
        return SmartImportPlan(
            source_root=str(root),
            lights_files=[],
            dark_files=[],
            flat_files=[],
            lights_first_fits=None,
            metadata=None,
            scan_rows=scan_rows,
            dark_master=None,
            flat_master=None,
            masterflat_by_filter={},
            masterflat_status={},
            missing_flat_filters=[],
            masterdark_status="MasterDark: n/a (invalid source root)",
            quick_look=False,
            detected_filters=[],
            warnings=warnings,
        )

    files_by_type = _collect_fits_by_type(root, db=db)
    lights_files = files_by_type.get("light", [])
    dark_files = files_by_type.get("dark", [])
    flat_files = files_by_type.get("flat", [])
    # Filters+binning combos from lights
    detected_filters = sorted(
        {flt for flt in (_read_filter(fp, db=db) for fp in lights_files[:500]) if flt}
    )

    if not lights_files:
        scan_rows.append(SmartScanRow("Lights", "missing", 0, ""))
        # still report what exists
        scan_rows.append(SmartScanRow("Darks", "raw" if dark_files else "missing", len(dark_files), ""))
        scan_rows.append(SmartScanRow("Flats", "raw" if flat_files else "missing", len(flat_files), ""))
        return SmartImportPlan(
            source_root=str(root),
            lights_files=[],
            dark_files=[str(p) for p in dark_files],
            flat_files=[str(p) for p in flat_files],
            lights_first_fits=None,
            metadata=None,
            scan_rows=scan_rows,
            dark_master=None,
            flat_master=None,
            masterflat_by_filter={},
            masterflat_status={},
            missing_flat_filters=[],
            masterdark_status="MasterDark: n/a (no Lights)",
            quick_look=False,
            detected_filters=[],
            warnings=warnings,
        )

    first_light = lights_files[0]
    metadata = extract_fits_metadata(first_light, db=db)
    scan_rows.append(
        SmartScanRow(
            "Lights",
            "ok",
            len(lights_files),
            _params_string(metadata, include_filter=True),
        )
    )

    exp = float(metadata["exposure"])
    gain = int(metadata.get("gain", 0))
    binning = int(metadata["binning"])
    temp = float(metadata["temp"])
    flt = str(metadata.get("filter", "Unknown"))

    calib_root = Path(calibration_library_root)
    cal_roots = _calibration_library_search_roots(calib_root)

    observation_groups: dict[str, dict[str, Any]] = {}
    for fp in lights_files[:8000]:
        try:
            meta_i = extract_fits_metadata(fp, db=db)
        except Exception:  # noqa: BLE001
            continue
        f_i = str(meta_i.get("filter", "")).strip()
        if not f_i or f_i.lower() in {"unknown", "none", "nan"}:
            f_i = "NoFilter"
        exp_i = float(meta_i.get("exposure", 0.0))
        b_i = max(1, int(meta_i.get("binning", 1) or 1))
        gk = observation_group_key(f_i, exp_i, b_i)
        if gk not in observation_groups:
            pu = meta_i.get("effective_pixel_um_plate_scale")
            foc = meta_i.get("focal_length")
            scale = None
            if pu is not None and foc is not None:
                try:
                    scale = plate_scale_arcsec_per_pixel(
                        pixel_pitch_um=float(pu), focal_length_mm=float(foc)
                    )
                except (TypeError, ValueError):
                    scale = None
            observation_groups[gk] = {
                "filter": f_i,
                "exposure_s": exp_i,
                "binning": b_i,
                "gain": int(meta_i.get("gain", 0)),
                "temp": float(meta_i.get("temp", 0.0)),
                "representative_light": str(fp),
                "light_paths": [],
                "plate_scale_arcsec_per_px": scale,
            }
        observation_groups[gk]["light_paths"].append(str(fp))

    masterflat_by_obs_key: dict[str, str | None] = {}
    masterflat_status: dict[str, str] = {}
    dark_master_by_obs_key: dict[str, str | None] = {}
    missing_obs_keys: list[str] = []
    for gk, g in sorted(observation_groups.items(), key=lambda x: x[0]):
        fp_best, status = _find_best_masterflat_for_filter(
            calib_root,
            flt=g["filter"],
            binning=int(g["binning"]),
            validity_days=masterflat_validity_days,
            db=db,
            exp=float(g["exposure_s"]),
            gain=int(g["gain"]),
            temp=float(g["temp"]),
            search_roots=cal_roots,
            id_equipments=id_equipments,
            id_telescope=id_telescope,
        )
        masterflat_by_obs_key[gk] = str(fp_best) if fp_best is not None else None
        masterflat_status[gk] = status
        if fp_best is None:
            missing_obs_keys.append(gk)

        d_found = _find_matching_master_in_library(
            calib_root,
            kind="dark",
            exp=float(g["exposure_s"]),
            gain=int(g["gain"]),
            binning=int(g["binning"]),
            temp=float(g["temp"]),
            flt=None,
            db=db,
            search_roots=cal_roots,
            id_equipments=id_equipments,
            id_telescope=id_telescope,
        )
        if d_found is not None:
            age_d = _age_days(d_found)
            if age_d is not None and age_d > masterdark_validity_days:
                dark_master_by_obs_key[gk] = None
                warnings.append(
                    f"MasterDark pre skupinu {gk} je expirovaný ({age_d} dní) — vygeneruj nový."
                )
            else:
                dark_master_by_obs_key[gk] = str(d_found)
        else:
            dark_master_by_obs_key[gk] = None

    masterflat_by_filter: dict[str, str | None] = {}
    for gk, g in observation_groups.items():
        fln = g["filter"]
        pth = masterflat_by_obs_key.get(gk)
        if fln not in masterflat_by_filter:
            masterflat_by_filter[fln] = pth
        elif pth is not None and masterflat_by_filter[fln] is None:
            masterflat_by_filter[fln] = pth

    missing_flat_filters: list[str] = sorted(
        {observation_groups[k]["filter"] for k in missing_obs_keys}
    )

    flat_fallback_prompts: list[dict[str, Any]] = []
    for gk in missing_obs_keys:
        g = observation_groups[gk]
        label = f"{g['filter']}-{g['exposure_s']:g}s"
        alts: list[str] = []
        for ok, og in observation_groups.items():
            if ok == gk or masterflat_by_obs_key.get(ok) is None:
                continue
            if int(og["binning"]) != int(g["binning"]):
                continue
            if abs(float(og["exposure_s"]) - float(g["exposure_s"])) > 1e-6:
                continue
            if og["filter"] == g["filter"]:
                continue
            alts.append(ok)
        msg = (
            f"Pozorovanie {label} nemá Master Flat. "
            "Chcete použiť Flat z iného filtra alebo preskočiť?"
        )
        flat_fallback_prompts.append(
            {
                "group_key": gk,
                "label": label,
                "message_sk": msg,
                "alternatives": alts,
            }
        )
        warnings.append(msg)

    def _scan_cal(kind: str, files: list[Path]) -> tuple[SmartScanRow, str | None, bool]:
        # Priority: raw on source > master on source > library fallback > draft
        if files:
            any_master = any(_looks_like_master(fp) for fp in files[: min(3, len(files))])
            any_raw = any(not _looks_like_master(fp) for fp in files[: min(3, len(files))])
            if any_raw:
                return (
                    SmartScanRow(kind.title() + "s", "raw", len(files), _params_string(metadata, include_filter=(kind == "flat"))),
                    None,
                    False,
                )
            if any_master:
                master_fp = next((fp for fp in files if _looks_like_master(fp)), files[0])
                return (
                    SmartScanRow(kind.title() + "s", "master", len(files), _params_string(metadata, include_filter=(kind == "flat")), details=str(master_fp)),
                    str(master_fp),
                    False,
                )

        # Library fallback
        found = _find_matching_master_in_library(
            calib_root,
            kind=kind,
            exp=exp,
            gain=gain,
            binning=binning,
            temp=temp,
            flt=flt if kind == "flat" else None,
            db=db,
            search_roots=cal_roots,
            id_equipments=id_equipments,
            id_telescope=id_telescope,
        )
        if found is None:
            return (
                SmartScanRow(kind.title() + "s", "missing", 0, _params_string(metadata, include_filter=(kind == "flat"))),
                None,
                True,
            )
        validity = masterflat_validity_days if kind == "flat" else masterdark_validity_days
        stt = get_calibration_status(found, kind=f"Master {kind.title()}", validity_days=validity)
        if stt.status == "expired":
            warnings.append(stt.message)
            return (
                SmartScanRow(kind.title() + "s", "expired", 1, _params_string(metadata, include_filter=(kind == "flat")), details=str(found)),
                str(found),
                False,
            )
        return (
            SmartScanRow(kind.title() + "s", "library", 1, _params_string(metadata, include_filter=(kind == "flat")), details=str(found)),
            str(found),
            False,
        )

    dark_row, dark_master, dark_missing = _scan_cal("dark", dark_files)
    flat_row, flat_master, flat_missing = _scan_cal("flat", flat_files)
    scan_rows.extend([dark_row, flat_row])

    # MasterDark status: match exp+gain+bin+temp (from first light meta used above)
    dark_found = _find_matching_master_in_library(
        calib_root,
        kind="dark",
        exp=exp,
        gain=gain,
        binning=binning,
        temp=temp,
        flt=None,
        db=db,
        search_roots=cal_roots,
        id_equipments=id_equipments,
        id_telescope=id_telescope,
    )
    # Ensure downstream steps (calibration) have an actual master path if found in library
    if dark_master is None and dark_found is not None:
        dark_master = str(dark_found)
    if dark_found is None and not dark_files:
        masterdark_status = "MasterDark: ❌ MISSING!"
    elif dark_found is None:
        masterdark_status = "MasterDark: raw on source (will build if requested)"
    else:
        stt_dark = get_calibration_status(dark_found, kind="Master Dark", validity_days=masterdark_validity_days)
        if stt_dark.status == "expired":
            masterdark_status = f"MasterDark: ⚠️ found but expired ({stt_dark.age_days} days old)"
            warnings.append(stt_dark.message)
        else:
            masterdark_status = f"MasterDark: ✅ found ({stt_dark.age_days} days old)"

    # Decision:
    # - Missing MasterDark => full Quick Look (Draft)
    # - Missing some MasterFlats => partial draft for those filters (not full Quick Look)
    quick_look = dark_missing
    if dark_missing:
        warnings.append("No suitable MasterDark found -> Quick Look Mode (Draft).")
    return SmartImportPlan(
        source_root=str(root),
        lights_files=[str(p) for p in lights_files],
        dark_files=[str(p) for p in dark_files],
        flat_files=[str(p) for p in flat_files],
        lights_first_fits=str(first_light),
        metadata=metadata,
        scan_rows=scan_rows,
        dark_master=dark_master,
        flat_master=flat_master,
        masterflat_by_filter=masterflat_by_filter,
        masterflat_status=masterflat_status,
        missing_flat_filters=missing_flat_filters,
        masterdark_status=masterdark_status,
        quick_look=quick_look,
        detected_filters=detected_filters,
        warnings=warnings,
        observation_groups=observation_groups,
        masterflat_by_obs_key=masterflat_by_obs_key,
        dark_master_by_obs_key=dark_master_by_obs_key,
        missing_obs_keys=missing_obs_keys,
        flat_fallback_prompts=flat_fallback_prompts,
    )


def generate_master_dark_from_source_dir(
    *,
    source_dir: str | Path,
    calibration_library_root: str | Path,
    db: VyvarDatabase | None = None,
    id_equipments: int | None = None,
    id_telescope: int | None = None,
) -> tuple[Path | None, list[str]]:
    """Master dark from raw frames (recursive, IMAGETYP): per-pixel mean stack. Filename per :func:`_write_master_to_library`."""
    messages: list[str] = []
    root = Path(source_dir)
    out_root = Path(calibration_library_root)
    if not root.is_dir():
        return None, [f"❌ Adresár neexistuje: {root}"]
    files_by_type = _collect_fits_by_type(root, db=db)
    dark_raw = [fp for fp in files_by_type.get("dark", []) if not _looks_like_master(fp)]
    if not dark_raw:
        return None, [
            "❌ V zadanom adresári sa nenašli surové dark snímky (očakáva sa IMAGETYP obsahujúci „dark“, nie hotový master)."
        ]
    existing = _find_existing_master_for_raw_set(
        out_root,
        kind="dark",
        sample_file=dark_raw[0],
        target_binning=None,
        db=db,
        id_equipments=id_equipments,
        id_telescope=id_telescope,
    )
    if existing is not None:
        messages.append(f"ℹ️ Master Dark už v knižnici existuje: {existing.name}")
        _register_master_path_in_calibration_library(
            db,
            kind="dark",
            path=existing,
            ncombine=None,
            id_equipments=id_equipments,
            id_telescope=id_telescope,
        )
        return existing, messages
    out = _write_master_to_library(
        kind="dark",
        files=dark_raw,
        calibration_library_root=out_root,
        target_binning=None,
        db=db,
        id_equipments=id_equipments,
        id_telescope=id_telescope,
    )
    messages.append(f"✅ Master Dark vytvorený: {out.name} ({len(dark_raw)} snímok)")
    return out, messages


def generate_master_flat_from_source_dir(
    *,
    source_dir: str | Path,
    calibration_library_root: str | Path,
    db: VyvarDatabase | None = None,
    id_equipments: int | None = None,
    id_telescope: int | None = None,
) -> tuple[Path | None, list[str]]:
    """Master flat: per-pixel **median** stack, then normalization. Naming ako v knižnici."""
    messages: list[str] = []
    root = Path(source_dir)
    out_root = Path(calibration_library_root)
    if not root.is_dir():
        return None, [f"❌ Adresár neexistuje: {root}"]
    files_by_type = _collect_fits_by_type(root, db=db)
    flat_raw = [fp for fp in files_by_type.get("flat", []) if not _looks_like_master(fp)]
    if not flat_raw:
        return None, [
            "❌ V zadanom adresári sa nenašli surové flat snímky (IMAGETYP obsahujúci „flat“, nie master)."
        ]
    existing = _find_existing_master_for_raw_set(
        out_root,
        kind="flat",
        sample_file=flat_raw[0],
        target_binning=None,
        db=db,
        id_equipments=id_equipments,
        id_telescope=id_telescope,
    )
    if existing is not None:
        messages.append(f"ℹ️ Master Flat už v knižnici existuje: {existing.name}")
        _register_master_path_in_calibration_library(
            db,
            kind="flat",
            path=existing,
            ncombine=None,
            id_equipments=id_equipments,
            id_telescope=id_telescope,
        )
        return existing, messages
    out = _write_master_to_library(
        kind="flat",
        files=flat_raw,
        calibration_library_root=out_root,
        target_binning=None,
        db=db,
        id_equipments=id_equipments,
        id_telescope=id_telescope,
    )
    messages.append(f"✅ Master Flat vytvorený: {out.name} ({len(flat_raw)} snímok)")
    return out, messages


def _copy_fits_folder(src: Path, dst: Path) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    count = 0
    for fp in _list_fits_files(src):
        shutil.copy2(fp, dst / fp.name)
        count += 1
    return count


def smart_import_session(
    *,
    plan: SmartImportPlan,
    pipeline: Any,
    id_equipment: int,
    id_telescope: int,
    id_location: int = 1,
) -> ImportResult:
    """Perform import according to SmartImportPlan decision tree (DB write)."""
    if not plan.lights_files:
        raise FileNotFoundError("Missing 'lights' directory! Import aborted.")
    lights_files = [Path(p) for p in plan.lights_files]
    if not lights_files:
        raise FileNotFoundError("Missing 'lights' directory! Import aborted.")

    metadata = extract_fits_metadata(lights_files[0], db=pipeline.db, app_config=pipeline.config)
    scanning_id = pipeline.db.find_or_create_scanning_id(metadata)

    # Ingestion creates DRAFT only (Session ID created after astrometry).
    missing_obs_keys_set: set[str] = set(getattr(plan, "missing_obs_keys", []) or [])
    draft_filters: set[str] = set(getattr(plan, "missing_flat_filters", []) or [])
    draft_id = pipeline.db.create_draft(
        {
            "id_equipments": int(id_equipment),
            "id_telescope": int(id_telescope),
            "id_location": int(id_location),
            "id_scanning": int(scanning_id),
            "observation_start_jd": float(metadata["jd_start"]),
            "is_calibrated": (len(draft_filters) == 0 and not plan.quick_look),
        }
    )

    try:
        _comb = pipeline.db.get_combined_metadata(lights_files[0], int(draft_id))
        if _comb.get("focal_length_mm") is None or _comb.get("pixel_effective_um") is None:
            plan.warnings.append(str(DraftTechnicalMetadataError(int(draft_id))))
    except Exception:  # noqa: BLE001
        pass

    archive_session = Path(pipeline.config.archive_root) / "Drafts" / f"draft_{draft_id:06d}"
    # Partial draft if some filters missing masterflat
    if plan.quick_look:
        target = archive_session / "non_calibrated"
        lights_root = target / "lights"
        evidence: list[dict[str, Any]] = []
        for fp in lights_files:
            flt = _sanitize_filter_folder(_read_filter(fp, db=pipeline.db))
            try:
                meta_l = extract_fits_metadata(fp, db=pipeline.db, app_config=pipeline.config)
                gk = observation_group_key(
                    flt, float(meta_l.get("exposure", 0.0)), int(meta_l.get("binning", 1) or 1)
                )
                sc_i = int(pipeline.db.find_or_create_scanning_id(meta_l))
            except Exception:  # noqa: BLE001
                gk = observation_group_key(flt, 0.0, 1)
                sc_i = None
            dst_dir = lights_root / observation_group_folder_name(gk)
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst_path = _dst_path_with_length_limit(dst_dir, fp.name)
            _safe_copy2(fp, dst_path)
            evidence.append(
                {
                    "file_path": str(dst_path),
                    "imagetyp": "light",
                    "filter": flt,
                    "observation_group_key": gk,
                    "id_scanning": sc_i,
                    "is_calibrated": 0,
                    "calib_type": "RAW_NON_CALIBRATED",
                }
            )
        now_utc = datetime.now(timezone.utc)
        pipeline.db.update_draft_import_log(
            draft_id,
            lights_path=str(lights_root),
            calib_path=f"draft_non_calibrated={target}",
            imported_at=now_utc.isoformat(timespec="seconds"),
            import_warnings="\n".join(plan.warnings) if plan.warnings else None,
            is_calibrated=False,
            archive_path=str(archive_session),
        )
        pipeline.db.insert_draft_files(draft_id, evidence)
        return ImportResult(
            draft_id=draft_id,
            observation_id=None,
            lights_path=str(lights_root),
            dark_path="",
            flat_path="",
            archive_path=str(archive_session),
            warnings=plan.warnings,
        )

    # calibrated path: copy into Raw structure
    raw_dir = archive_session / "Raw"
    lights_root = raw_dir / "lights"
    darks_dst = raw_dir / "darks"
    flats_dst = raw_dir / "flats"
    lights_root.mkdir(parents=True, exist_ok=True)
    darks_dst.mkdir(parents=True, exist_ok=True)
    flats_dst.mkdir(parents=True, exist_ok=True)

    # If some filters are missing MasterFlat, import those lights into non_calibrated
    draft_root = archive_session / "non_calibrated"
    draft_lights_root = draft_root / "lights"

    # Flexible sorting: even if mixed folders exist, we sort per-file by IMAGETYP
    evidence: list[dict[str, Any]] = []
    for fp in lights_files:
        flt = _sanitize_filter_folder(_read_filter(fp, db=pipeline.db))
        try:
            meta_l = extract_fits_metadata(fp, db=pipeline.db, app_config=pipeline.config)
            gk = observation_group_key(
                flt, float(meta_l.get("exposure", 0.0)), int(meta_l.get("binning", 1) or 1)
            )
            sc_i = int(pipeline.db.find_or_create_scanning_id(meta_l))
        except Exception:  # noqa: BLE001
            gk = observation_group_key(flt, 0.0, 1)
            sc_i = None
        is_draft = gk in missing_obs_keys_set or flt in draft_filters or (
            flt == "NoFilter" and "NoFilter" in draft_filters
        )
        dst_base = draft_lights_root if is_draft else lights_root
        dst_dir = dst_base / observation_group_folder_name(gk)
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_path = _dst_path_with_length_limit(dst_dir, fp.name)
        _safe_copy2(fp, dst_path)
        evidence.append(
            {
                "file_path": str(dst_path),
                "imagetyp": "light",
                "filter": flt,
                "observation_group_key": gk,
                "id_scanning": sc_i,
                "is_calibrated": 0 if is_draft else 1,
                "calib_type": "RAW_PENDING_CALIBRATION" if is_draft else "MASTER_PIPELINE",
            }
        )
    for fp in [Path(p) for p in plan.dark_files]:
        dst_path = _dst_path_with_length_limit(darks_dst, fp.name)
        _safe_copy2(fp, dst_path)
        evidence.append({"file_path": str(dst_path), "imagetyp": "dark", "filter": ""})
    for fp in [Path(p) for p in plan.flat_files]:
        dst_path = _dst_path_with_length_limit(flats_dst, fp.name)
        _safe_copy2(fp, dst_path)
        evidence.append(
            {
                "file_path": str(dst_path),
                "imagetyp": "flat",
                "filter": _sanitize_filter_folder(_read_filter(fp, db=pipeline.db)),
            }
        )

    now_utc = datetime.now(timezone.utc)
    pipeline.db.update_draft_import_log(
        draft_id,
        lights_path=str(lights_root),
        calib_path=(
            f"dark={plan.dark_master};flat_by_filter={plan.masterflat_by_filter};"
            f"flat_by_obs_key={getattr(plan, 'masterflat_by_obs_key', {})};"
            f"dark_by_obs_key={getattr(plan, 'dark_master_by_obs_key', {})}"
        ),
        imported_at=now_utc.isoformat(timespec="seconds"),
        import_warnings="\n".join(plan.warnings) if plan.warnings else None,
        is_calibrated=(len(draft_filters) == 0),
        archive_path=str(archive_session),
    )
    pipeline.db.insert_draft_files(draft_id, evidence)
    return ImportResult(
        draft_id=draft_id,
        observation_id=None,
        lights_path=str(lights_root),
        dark_path=str(plan.dark_master or ""),
        flat_path=";".join(
            [f"{k}={v}" for k, v in (getattr(plan, "masterflat_by_obs_key", None) or plan.masterflat_by_filter or {}).items()]
        ),
        archive_path=str(archive_session),
        warnings=plan.warnings,
    )


def quicklook_preview_png_bytes(fits_path: str | Path) -> bytes:
    """Generate an 8-bit stretched preview PNG bytes (ZScale) from a FITS image."""
    fp = Path(fits_path)
    with fits.open(fp, memmap=False) as hdul:
        data = _to_float32_frame(hdul[0].data)
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(data)
    scaled = np.clip((data - vmin) / (vmax - vmin + 1e-12), 0, 1)
    img8 = (scaled * 255).astype(np.uint8)

    # Encode via PIL if available; fallback to raw bytes not supported by st.image
    try:
        from PIL import Image  # type: ignore
        import io

        im = Image.fromarray(img8)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("PIL is required for PNG preview generation.") from exc

# --- CalibrationLibrary master stacking (no sigma-clipping on the stack axis) ---
#
# **Master dark — plain average (mean):** Low-signal frames; sigma-clipping would wrongly reject
# real hot pixels and harm thermal-noise subtraction SNR. We use per-pixel ``nanmean`` over frames.
#
# **Master flat — plain median:** Per-pixel ``nanmedian`` ignores dust motes / glints while preserving
# vignetting structure. Median normalization is deferred to :func:`calibration.get_processed_master`
# (after resample to the light's binning) and flagged on disk with ``VYFLNRD=1``.

# Large stacks use a temp memmap cube + Y slabs.
_STACK_FULL_RAM_BYTES = 512 * 1024 * 1024
_STACK_LARGE_N = 8
_STACK_SLAB_RAM_BYTES = 256 * 1024 * 1024


def _to_float32_frame(data: np.ndarray) -> np.ndarray:
    """Use a float32 view when already float32; copy only for integer / other dtypes."""
    a = np.asanyarray(data)
    if a.dtype == np.float32:
        return a
    if np.issubdtype(a.dtype, np.floating):
        return a.astype(np.float32, copy=False)
    return np.asarray(a, dtype=np.float32)


def _apply_binning_frame(arr: np.ndarray, bin_factor: int) -> np.ndarray:
    """Average-block bin ``arr`` to match light binning (integer factor ≥ 1)."""
    a = _to_float32_frame(arr)
    if bin_factor <= 1:
        return a
    h, w = a.shape[:2]
    h2 = h - (h % bin_factor)
    w2 = w - (w % bin_factor)
    if h2 <= 0 or w2 <= 0:
        raise ValueError(f"Cannot bin image with shape {a.shape} by factor {bin_factor}")
    if h2 != h or w2 != w:
        a = a[:h2, :w2]
        h, w = a.shape[:2]
    return a.reshape(h // bin_factor, bin_factor, w // bin_factor, bin_factor).mean(axis=(1, 3)).astype(
        np.float32
    )


def _combine_stack_mean(stack: np.ndarray) -> np.ndarray:
    """Average stack for darks: ``stack`` (n, ny, nx) -> per-pixel mean along axis 0 (float32).

    Intentionally **not** sigma-clipped — preserves hot pixels and optimizes SNR for bias/dark subtraction.
    """
    if stack.ndim != 3:
        raise ValueError(f"Expected stack (n, ny, nx), got {stack.shape}")
    out = np.nanmean(stack, axis=0).astype(np.float32)
    bad = ~np.isfinite(out)
    if np.any(bad):
        good = out[np.isfinite(out)]
        fill = float(np.median(good)) if good.size else 0.0
        out[bad] = fill
    return out


def _combine_stack_median(stack: np.ndarray) -> np.ndarray:
    """Median stack for flats: ``stack`` (n, ny, nx) -> per-pixel median along axis 0 (float32).

    Rejects transient dust/glints without sigma-clipping artefacts; vignetting is kept until
    calibrate-time normalization in :func:`calibration.get_processed_master`.
    """
    if stack.ndim != 3:
        raise ValueError(f"Expected stack (n, ny, nx), got {stack.shape}")
    out = np.nanmedian(stack, axis=0).astype(np.float32)
    bad = ~np.isfinite(out)
    if np.any(bad):
        good = out[np.isfinite(out)]
        fill = float(np.median(good)) if good.size else 0.0
        out[bad] = fill
    return out


def _stack_calibration_frames(
    files: list[Path],
    *,
    combine: str,
    bin_factor: int = 1,
    kind: str = "dark",
) -> tuple[np.ndarray, fits.Header]:
    """Stack calibration frames: **mean** for dark, **median** for flat; optional memmap cube for RAM.

    No sigma-clipping — see module comment above.
    """
    if combine not in ("mean", "median"):
        raise ValueError(f"combine must be 'mean' or 'median', got {combine!r}")
    if combine == "mean" and kind != "dark":
        raise ValueError("Mean combine is only used for master dark.")
    if combine == "median" and kind != "flat":
        raise ValueError("Median combine is only used for master flat.")
    if not files:
        raise ValueError("No FITS files provided for stacking.")

    with fits.open(files[0], memmap=False) as hdul0:
        header0 = hdul0[0].header.copy()
        a0b = _apply_binning_frame(_to_float32_frame(hdul0[0].data), bin_factor)

    n = len(files)
    h0, w0 = int(a0b.shape[0]), int(a0b.shape[1])
    cube_bytes = n * h0 * w0 * 4
    use_memmap_cube = n > _STACK_LARGE_N or cube_bytes > _STACK_FULL_RAM_BYTES

    if combine == "mean":
        log_event(
            f"Master {kind}: mean stack (per-pixel) N={n} frames, "
            f"{'memmap cube' if use_memmap_cube else 'RAM cube'} "
            f"({h0}x{w0}, bin={bin_factor})."
        )
    else:
        log_event(
            f"Master {kind}: median stack (per-pixel) N={n} frames, "
            f"{'memmap cube' if use_memmap_cube else 'RAM cube'} "
            f"({h0}x{w0}, bin={bin_factor})."
        )

    def _combine_block(block: np.ndarray) -> np.ndarray:
        if combine == "mean":
            return _combine_stack_mean(block)
        return _combine_stack_median(block)

    if not use_memmap_cube:
        stack = np.empty((n, h0, w0), dtype=np.float32)
        stack[0] = a0b
        for i in range(1, n):
            with fits.open(files[i], memmap=False) as hdul:
                stack[i] = _apply_binning_frame(_to_float32_frame(hdul[0].data), bin_factor)
        master = _combine_block(stack)
    else:
        fd, tmp_path = tempfile.mkstemp(prefix="vyvar_cal_stack_", suffix=".dat")
        os.close(fd)
        try:
            cube = np.memmap(tmp_path, dtype=np.float32, mode="w+", shape=(n, h0, w0))
            cube[0] = a0b
            for i in range(1, n):
                with fits.open(files[i], memmap=False) as hdul:
                    cube[i] = _apply_binning_frame(_to_float32_frame(hdul[0].data), bin_factor)
            cube.flush()
            del cube
            cube_r = np.memmap(tmp_path, dtype=np.float32, mode="r", shape=(n, h0, w0))
            master = np.empty((h0, w0), dtype=np.float32)
            tile_h = max(1, int(_STACK_SLAB_RAM_BYTES // max(1, n * w0 * 4)))
            for y0 in range(0, h0, tile_h):
                y1 = min(h0, y0 + tile_h)
                block = np.asarray(cube_r[:, y0:y1, :], dtype=np.float32)
                master[y0:y1, :] = _combine_block(block)
            del cube_r
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    if combine == "mean":
        header0["VYSTKMOD"] = ("MEAN", "Per-pixel mean along stack axis")
    else:
        header0["VYSTKMOD"] = ("MEDIAN", "Per-pixel median along stack axis")
    return master, header0


def _generate_master_dark(
    darks_files: list[Path],
    *,
    bin_factor: int = 1,
) -> tuple[np.ndarray, fits.Header]:
    """Build master dark: per-pixel mean over input frames (see :func:`_combine_stack_mean`)."""
    return _stack_calibration_frames(
        darks_files,
        combine="mean",
        bin_factor=bin_factor,
        kind="dark",
    )


def _generate_master_flat(
    flats_files: list[Path],
    *,
    bin_factor: int = 1,
) -> tuple[np.ndarray, fits.Header]:
    """Build master flat: per-pixel median over frames (normalization at calibrate; see ``VYFLNRD``)."""
    return _stack_calibration_frames(
        flats_files,
        combine="median",
        bin_factor=bin_factor,
        kind="flat",
    )


def generate_and_archive_masters(
    *,
    session_root: str | Path,
    calibration_library_root: str | Path,
    raw_threshold: int = 5,
    db: VyvarDatabase | None = None,
    id_equipments: int | None = None,
    id_telescope: int | None = None,
) -> MasterGenerationResult:
    """Generate and store master calibration frames from raw USB folders.

    Triggers when darks/ or flats/ contains > raw_threshold FITS files.
    Masters are stored in CalibrationLibrary with naming:
      {MD/MF}_{Exp}s_{Filter}_{Temp}deg_Bin{Binning}_{YYYYMMDD}.fits

    YYYYMMDD is derived from DATE-OBS of calibration frames (earliest file date used).
    """
    root = Path(session_root)
    darks_dir = root / "darks"
    flats_dir = root / "flats"
    out_root = Path(calibration_library_root)
    out_root.mkdir(parents=True, exist_ok=True)

    messages: list[str] = []
    dark_master_path: str | None = None
    flat_master_path: str | None = None

    if darks_dir.exists() and darks_dir.is_dir():
        dark_files = _list_fits_files(darks_dir)
        if len(dark_files) > raw_threshold:
            existing = _find_existing_master_for_raw_set(
                out_root,
                kind="dark",
                sample_file=dark_files[0],
                db=db,
                id_equipments=id_equipments,
                id_telescope=id_telescope,
            )
            if existing is not None:
                dark_master_path = str(existing)
                messages.append(f"ℹ️ Master Dark already exists: {existing.name}")
                _register_master_path_in_calibration_library(
                    db,
                    kind="dark",
                    path=existing,
                    ncombine=None,
                    id_equipments=id_equipments,
                    id_telescope=id_telescope,
                )
            else:
                meta = extract_fits_metadata(dark_files[0], db=db)
                exp_s = _format_exp_seconds(float(meta["exposure"]))
                temp = _format_temp(float(meta["temp"]))
                gain = int(meta.get("gain", 0))
                binning = int(meta["binning"])
                date_yyyymmdd = min(_fits_capture_date_yyyymmdd(fp) for fp in dark_files)
                filename = f"MD_{exp_s}s_Gain{gain}_{temp}deg_Bin{binning}_{date_yyyymmdd}.fits"
                out_path = out_root / filename
                master, header = _generate_master_dark(dark_files)
                header["BINNING"] = binning
                header["XBINNING"] = binning
                header["YBINNING"] = binning
                header["VY_MBLIB"] = (
                    1,
                    "VYVAR: native stack in CalibrationLibrary; resample to light XBINNING at calibrate",
                )
                header["HISTORY"] = "VYVAR: Master Dark (mean stack)"
                dt_src = _earliest_capture_datetime_utc(dark_files)
                if dt_src is not None:
                    dt_iso = dt_src.strftime("%Y-%m-%dT%H:%M:%S")
                    header["DATE-OBS"] = (dt_iso, "Earliest source raw capture time (UTC)")
                    header["VY_CDATE"] = (dt_iso, "VYVAR source capture datetime (UTC)")
                header["NCOMBINE"] = len(dark_files)
                fits.writeto(out_path, master.astype(np.float32), header=header, overwrite=True)
                try:
                    from config import AppConfig
                    from photometry import write_dark_bpm_json

                    _sig = float(AppConfig().bpm_dark_mad_sigma)
                    _nb = int(header.get("XBINNING") or header.get("BINNING") or binning)
                    write_dark_bpm_json(out_path, master, mad_sigma=_sig, native_binning=_nb)
                except Exception:  # noqa: BLE001
                    pass
                dark_master_path = str(out_path)
                messages.append(f"✅ Fresh Master Dark created: {out_path.name}")
                _register_master_path_in_calibration_library(
                    db,
                    kind="dark",
                    path=out_path,
                    ncombine=int(header.get("NCOMBINE", len(dark_files))),
                    id_equipments=id_equipments,
                    id_telescope=id_telescope,
                )
        elif len(dark_files) > 0:
            messages.append("Darks detected, but not enough frames to generate master.")
    else:
        messages.append("No darks/ folder found for master generation.")

    if flats_dir.exists() and flats_dir.is_dir():
        flat_files = _list_fits_files(flats_dir)
        if len(flat_files) > raw_threshold:
            existing = _find_existing_master_for_raw_set(
                out_root,
                kind="flat",
                sample_file=flat_files[0],
                db=db,
                id_equipments=id_equipments,
                id_telescope=id_telescope,
            )
            if existing is not None:
                flat_master_path = str(existing)
                messages.append(f"ℹ️ Master Flat already exists: {existing.name}")
                _register_master_path_in_calibration_library(
                    db,
                    kind="flat",
                    path=existing,
                    ncombine=None,
                    id_equipments=id_equipments,
                    id_telescope=id_telescope,
                )
            else:
                meta = extract_fits_metadata(flat_files[0], db=db)
                exp_s = _format_exp_seconds(float(meta["exposure"]))
                flt = _sanitize_token(meta.get("filter", "Unknown"))
                temp = _format_temp(float(meta["temp"]))
                gain = int(meta.get("gain", 0))
                binning = int(meta["binning"])
                date_yyyymmdd = min(_fits_capture_date_yyyymmdd(fp) for fp in flat_files)
                filename = f"MF_{exp_s}s_{flt}_Gain{gain}_{temp}deg_Bin{binning}_{date_yyyymmdd}.fits"
                out_path = out_root / filename
                master, header = _generate_master_flat(flat_files)
                header["BINNING"] = binning
                header["XBINNING"] = binning
                header["YBINNING"] = binning
                header["VY_MBLIB"] = (
                    1,
                    "VYVAR: native stack in CalibrationLibrary; resample to light XBINNING at calibrate",
                )
                header["HISTORY"] = "VYVAR: Master Flat (median stack; norm at calibrate)"
                header["VYFLNRD"] = (
                    1,
                    "Median normalization deferred to calibrate after resample to light binning",
                )
                dt_src = _earliest_capture_datetime_utc(flat_files)
                if dt_src is not None:
                    dt_iso = dt_src.strftime("%Y-%m-%dT%H:%M:%S")
                    header["DATE-OBS"] = (dt_iso, "Earliest source raw capture time (UTC)")
                    header["VY_CDATE"] = (dt_iso, "VYVAR source capture datetime (UTC)")
                header["NCOMBINE"] = len(flat_files)
                fits.writeto(out_path, master.astype(np.float32), header=header, overwrite=True)
                flat_master_path = str(out_path)
                messages.append(f"✅ Fresh Master Flat created: {out_path.name}")
                _register_master_path_in_calibration_library(
                    db,
                    kind="flat",
                    path=out_path,
                    ncombine=int(header.get("NCOMBINE", len(flat_files))),
                    id_equipments=id_equipments,
                    id_telescope=id_telescope,
                )
        elif len(flat_files) > 0:
            messages.append("Flats detected, but not enough frames to generate master.")
    else:
        messages.append("No flats/ folder found for master generation.")

    return MasterGenerationResult(
        dark_master_path=dark_master_path,
        flat_master_path=flat_master_path,
        messages=messages,
    )


def get_calibration_status(
    path: str | Path | None,
    *,
    kind: str,
    validity_days: int,
) -> CalibrationStatus:
    """Return calibration status for a given file/folder.

    - status=missing: path is None / doesn't exist / empty dir
    - status=expired: newest mtime older than stale_days
    - status=ok: otherwise
    """
    if path is None:
        return CalibrationStatus(
            kind=kind,
            path=None,
            status="missing",
            last_modified_utc=None,
            age_days=None,
            validity_days=validity_days,
            message=f"{kind}: missing",
        )

    p = Path(path)
    if not p.exists():
        return CalibrationStatus(
            kind=kind,
            path=str(p),
            status="missing",
            last_modified_utc=None,
            age_days=None,
            validity_days=validity_days,
            message=f"{kind}: path not found",
        )

    check_path = p
    if check_path.is_dir():
        files = _list_fits_files(check_path)
        if not files:
            return CalibrationStatus(
                kind=kind,
                path=str(check_path),
                status="missing",
                last_modified_utc=None,
                age_days=None,
                validity_days=validity_days,
                message=f"{kind}: directory is empty",
            )
        check_path = max(files, key=lambda fp: os.path.getmtime(fp))

    try:
        mtime = _mtime_utc(check_path)
    except OSError:
        return CalibrationStatus(
            kind=kind,
            path=str(p),
            status="missing",
            last_modified_utc=None,
            age_days=None,
            validity_days=validity_days,
            message=f"{kind}: cannot read modification time",
        )

    now_utc = datetime.now(timezone.utc)
    age = now_utc - mtime
    age_days = int(age.total_seconds() // 86400)
    expired = age_days > validity_days

    if expired:
        return CalibrationStatus(
            kind=kind,
            path=str(p),
            status="expired",
            last_modified_utc=mtime.strftime("%Y-%m-%d %H:%M UTC"),
            age_days=age_days,
            validity_days=validity_days,
            message=f"{kind}: expired (last {mtime.strftime('%Y-%m-%d')}, {age_days} days old)",
        )

    return CalibrationStatus(
        kind=kind,
        path=str(p),
        status="ok",
        last_modified_utc=mtime.strftime("%Y-%m-%d %H:%M UTC"),
        age_days=age_days,
        validity_days=validity_days,
        message=f"{kind}: ok ({age_days} days old)",
    )


def plan_session_import(
    *,
    session_root: str | Path,
    calibration_library_root: str | Path,
    db: VyvarDatabase | None = None,
    id_equipments: int | None = None,
    id_telescope: int | None = None,
) -> dict[str, Any]:
    """Plan import without writing to DB (used by Streamlit UI)."""
    root = Path(session_root)
    lights_dir = root / "lights"
    darks_dir = root / "darks"
    flats_dir = root / "flats"

    if not lights_dir.exists() or not lights_dir.is_dir():
        raise FileNotFoundError(f"Missing 'lights' subdirectory under: {root}")

    lights_files = _list_fits_files(lights_dir)
    if not lights_files:
        raise FileNotFoundError(f"'lights' contains no FITS files: {lights_dir}")

    first_light = lights_files[0]
    metadata = extract_fits_metadata(first_light, db=db)

    exposure = float(metadata["exposure"])
    temp = float(metadata["temp"])
    binning = int(metadata["binning"])
    gain = int(metadata.get("gain", 0) or 0)
    light_filter = str(metadata.get("filter", "") or "")

    dark_path: Path | None
    if _is_empty_or_missing(darks_dir):
        dark_path = _find_best_calibration_file(
            Path(calibration_library_root),
            kind="dark",
            exposure=exposure,
            temp=temp,
            binning=binning,
            db=db,
            gain=gain,
            id_equipments=id_equipments,
            id_telescope=id_telescope,
        )
    else:
        dark_path = darks_dir

    flat_path: Path | None
    if _is_empty_or_missing(flats_dir):
        flat_path = _find_best_calibration_file(
            Path(calibration_library_root),
            kind="flat",
            exposure=exposure,
            temp=temp,
            binning=binning,
            db=db,
            gain=gain,
            filter_name=light_filter,
            id_equipments=id_equipments,
            id_telescope=id_telescope,
        )
    else:
        flat_path = flats_dir

    dark_raw_count = len(_list_fits_files(darks_dir)) if (darks_dir.exists() and darks_dir.is_dir()) else 0
    flat_raw_count = len(_list_fits_files(flats_dir)) if (flats_dir.exists() and flats_dir.is_dir()) else 0

    return {
        "session_root": str(root),
        "lights_dir": str(lights_dir),
        "first_light": str(first_light),
        "metadata": metadata,
        "dark_path": str(dark_path) if dark_path is not None else None,
        "flat_path": str(flat_path) if flat_path is not None else None,
        "dark_raw_count": dark_raw_count,
        "flat_raw_count": flat_raw_count,
    }


def _find_best_calibration_file(
    calibration_root: Path,
    *,
    kind: str,
    exposure: float,
    temp: float,
    binning: int,
    db: VyvarDatabase | None = None,
    gain: int = 0,
    filter_name: str | None = None,
    id_equipments: int | None = None,
    id_telescope: int | None = None,
) -> Path | None:
    """Find newest calibration FITS matching (EXPTIME, TEMP ±0.5, BINNING); prefers CALIBRATION_LIBRARY when ``db`` is set."""
    kind_lower = kind.lower()
    flt_db = _filter_name_for_calibration_library_flat(filter_name) if kind_lower == "flat" else ""
    if db is not None:
        try:
            hit = db.find_best_calibration_library_path(
                kind=kind_lower,
                xbinning=int(binning),
                exptime=float(exposure),
                ccd_temp=float(temp),
                filter_name=flt_db,
                gain=int(gain),
                id_equipments=id_equipments,
                id_telescope=id_telescope,
            )
        except Exception:  # noqa: BLE001
            hit = None
        if hit and Path(hit).is_file():
            return Path(hit)

    if not calibration_root.exists():
        return None

    candidates: list[Path] = []

    for ext in ("*.fits", "*.fit", "*.fts", "*.FITS", "*.FIT", "*.FTS"):
        candidates.extend(calibration_root.rglob(ext))

    if not candidates:
        return None

    # Prefer files with kind in name, then fall back to all.
    preferred = [p for p in candidates if kind_lower in p.name.lower()]
    search_space = preferred if preferred else candidates

    best: Path | None = None
    best_mtime: float = -1.0

    for path in search_space:
        try:
            meta = extract_fits_metadata(path, db=db)
        except Exception:  # noqa: BLE001
            continue

        if abs(float(meta.get("temp", 0.0)) - float(temp)) > 0.5:
            continue
        if int(meta.get("binning", 0)) != int(binning):
            continue
        if float(meta.get("exposure", -1.0)) != float(exposure):
            continue
        if int(meta.get("gain", 0) or 0) != int(gain):
            continue
        if kind_lower == "flat":
            if _filter_name_for_calibration_library_flat(str(meta.get("filter", ""))) != flt_db:
                continue
        if not _looks_like_master(path):
            continue
        if not _master_kind_matches(path, kind_lower):
            continue

        mtime = os.path.getmtime(path)
        if mtime > best_mtime:
            best = path
            best_mtime = mtime

    return best


def import_session(
    *,
    session_root: str | Path,
    pipeline: Any,
    id_equipment: int,
    id_telescope: int,
    id_location: int = 1,
    masterdark_validity_days: int,
    masterflat_validity_days: int,
    force_use_expired: bool = False,
    quick_look_mode: bool = False,
) -> ImportResult:
    """Import a session directory with expected subfolders lights/darks/flats.

    - If darks/flats missing or empty -> search CalibrationLibrary for matching masters.
    - Warn if calibration master is older than `stale_days`.
    - Create OBSERVATION record and write import log columns (paths + imported timestamp).
    """
    root = Path(session_root)
    lights_dir = root / "lights"
    darks_dir = root / "darks"
    flats_dir = root / "flats"

    if not lights_dir.exists() or not lights_dir.is_dir():
        raise FileNotFoundError(f"Missing 'lights' subdirectory under: {root}")

    lights_files = _list_fits_files(lights_dir)
    if not lights_files:
        raise FileNotFoundError(f"'lights' contains no FITS files: {lights_dir}")

    first_light = lights_files[0]
    metadata = extract_fits_metadata(first_light, db=pipeline.db, app_config=pipeline.config)
    warnings: list[str] = []

    exposure = float(metadata["exposure"])
    temp = float(metadata["temp"])
    binning = int(metadata["binning"])
    gain = int(metadata.get("gain", 0) or 0)
    light_filter = str(metadata.get("filter", "") or "")

    dark_path: Path | None
    flat_path: Path | None
    if quick_look_mode:
        dark_path = None
        flat_path = None
        warnings.append("Quick Look Mode: skipping CalibrationLibrary lookup (non-calibrated draft).")
    else:
        dark_path = darks_dir if not _is_empty_or_missing(darks_dir) else _find_best_calibration_file(
            pipeline.config.calibration_library_root,
            kind="dark",
            exposure=exposure,
            temp=temp,
            binning=binning,
            db=pipeline.db,
            gain=gain,
            id_equipments=int(id_equipment),
            id_telescope=int(id_telescope),
        )
        if dark_path is None:
            warnings.append(
                "Missing/empty 'darks' and no matching dark master found in CalibrationLibrary. "
                "Calibration passthrough will be used."
            )

        flat_path = flats_dir if not _is_empty_or_missing(flats_dir) else _find_best_calibration_file(
            pipeline.config.calibration_library_root,
            kind="flat",
            exposure=exposure,
            temp=temp,
            binning=binning,
            db=pipeline.db,
            gain=gain,
            filter_name=light_filter,
            id_equipments=int(id_equipment),
            id_telescope=int(id_telescope),
        )
        if flat_path is None:
            warnings.append(
                "Missing/empty 'flats' and no matching flat master found in CalibrationLibrary. "
                "Calibration passthrough will be used."
            )

    now_utc = datetime.now(timezone.utc)
    if not quick_look_mode:
        dark_status = get_calibration_status(
            dark_path, kind="Master Dark", validity_days=masterdark_validity_days
        )
        flat_status = get_calibration_status(
            flat_path, kind="Master Flat", validity_days=masterflat_validity_days
        )

        for stt in (dark_status, flat_status):
            if stt.status == "expired":
                msg = f"{stt.kind} calibration expired: {stt.path} ({stt.last_modified_utc})"
                warnings.append(msg)

        any_expired = (dark_status.status == "expired") or (flat_status.status == "expired")
        if any_expired and not force_use_expired:
            raise ValueError(
                "Calibration is expired. Use Force Use in UI to continue with expired masters."
            )
        if any_expired and force_use_expired:
            warnings.append("Force Use enabled: proceeding with expired calibration.")

    scanning_id = pipeline.db.find_or_create_scanning_id(metadata)
    payload: dict[str, Any] = {
        "id_equipments": int(id_equipment),
        "id_telescope": int(id_telescope),
        "id_location": int(id_location),
        "id_scanning": int(scanning_id),
        "center_of_field_ra": float(metadata["ra"]),
        "center_of_field_de": float(metadata["dec"]),
        "observation_start_jd": float(metadata["jd_start"]),
    }

    observation_id = pipeline.create_observation_from_payload(payload)

    archive_session = Path(pipeline.config.archive_root) / observation_id
    archive_target = archive_session / ("non_calibrated" if quick_look_mode else "imported")
    archive_target.mkdir(parents=True, exist_ok=True)

    calib_path = (
        f"dark={dark_path};flat={flat_path}"
        if not quick_look_mode
        else f"draft_non_calibrated={archive_target}"
    )
    pipeline.db.update_observation_import_log(
        observation_id,
        lights_path=str(lights_dir),
        calib_path=calib_path,
        imported_at=now_utc.isoformat(timespec="seconds"),
        import_warnings="\n".join(warnings) if warnings else None,
        is_calibrated=not quick_look_mode,
        archive_path=str(archive_session),
    )

    return ImportResult(
        observation_id=observation_id,
        lights_path=str(lights_dir),
        dark_path=str(dark_path) if dark_path is not None else "",
        flat_path=str(flat_path) if flat_path is not None else "",
        archive_path=str(archive_session),
        warnings=warnings,
    )


def check_known_field(
    ra_deg: float,
    dec_deg: float,
    db: VyvarDatabase,
    *,
    match_radius_deg: float = 0.5,
) -> dict[str, Any] | None:
    """Return FIELD_REGISTRY match + comp-star library rows, or ``None`` (never raises)."""
    try:
        field = db.find_matching_field(ra_deg, dec_deg, match_radius_deg=match_radius_deg)
        if field is None:
            return None
        comp_stars = db.get_comp_stars_for_field(
            ra_deg,
            dec_deg,
            match_radius_deg=match_radius_deg,
            only_approved=True,
        )
        return {
            "field": field,
            "comp_stars": comp_stars,
            "n_comp_stars": len(comp_stars),
            "n_observations": int(field.get("N_OBSERVATIONS") or 1),
            "last_observation_id": field.get("LAST_OBSERVATION_ID"),
            "masterstar_path": field.get("MASTERSTAR_PATH"),
            "comparison_csv_path": field.get("COMPARISON_CSV_PATH"),
        }
    except Exception:  # noqa: BLE001
        return None

