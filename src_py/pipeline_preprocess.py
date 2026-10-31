"""Moved from pipeline.py (CONSOLIDATE-01E6a). Facade re-exports these names.

Calibrated->processed orchestration / QC filters.
The four giants stay in pipeline.py this wave (E6b).
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from astropy.io import fits
import pandas as pd
from config import AppConfig
from database import VyvarDatabase
from infolog import log_event, log_milestone
from utils import iter_fits_paths_recursive as _iter_fits_recursive
from proc_frame_store import _safe_proc_name
from pipeline_calibrate import (
    _qc_enrich_calibrated_in_place,
    _resolve_draft_light_raw_path,
    draft_median_pointing_icrs_deg,
)
from utils import load_qc_metrics_status_by_path, norm_fits_path_key  # noqa: F401  (pipeline facade re-export)

# Same named logger as pipeline.LOGGER (logging.getLogger singleton).
# Avoids pipeline -> pipeline_preprocess -> pipeline at module load.
LOGGER = logging.getLogger("pipeline")

def _archive_raw_to_calibrated_light(
    archive: Path,
    raw_fp: Path | str,
) -> tuple[Path, Path] | None:
    """Map archived raw light path to ``(calibrated_fits, lights_root_under_archive)``."""
    ap = Path(archive).expanduser().resolve()
    raw_path = Path(raw_fp)
    try:
        r = raw_path.resolve() if raw_path.is_file() else (ap / raw_path).resolve()
    except OSError:
        r = raw_path if raw_path.is_file() else (ap / raw_path)
    pairs: tuple[tuple[Path, Path], ...] = (
        (ap / "non_calibrated" / "lights", ap / "calibrated" / "lights"),
        (ap / "Raw" / "lights", ap / "calibrated" / "lights"),
    )
    for raw_root, cal_root in pairs:
        if not raw_root.is_dir():
            continue
        try:
            rel = r.relative_to(raw_root.resolve())
        except ValueError:
            continue
        cand = cal_root / rel
        if cand.is_file():
            return cand, cal_root
    return None


def _load_raw_for_frame(st: dict[str, Any], fname: str) -> Any | None:
    arch = str(st.get("sat_diag_archive") or "").strip()
    if not arch:
        return None
    raw_p = _resolve_draft_light_raw_path(Path(arch), fname)
    if raw_p is None or not raw_p.is_file():
        return None
    try:
        from sat_diag import image_adu_array  # noqa: PLC0415

        with fits.open(raw_p, memmap=False) as hdul:
            if int(hdul[0].header.get("BITPIX", 0)) < 0:
                return None
            return image_adu_array(hdul[0])
    except Exception:  # noqa: BLE001
        return None


def _load_raw_hdr_for_frame(st: dict[str, Any], fname: str) -> fits.Header | None:
    arch = str(st.get("sat_diag_archive") or "").strip()
    if not arch:
        return None
    raw_p = _resolve_draft_light_raw_path(Path(arch), fname)
    if raw_p is None or not raw_p.is_file():
        return None
    try:
        with fits.open(raw_p, memmap=False) as hdul:
            return hdul[0].header.copy()
    except Exception:  # noqa: BLE001
        return None


def filter_files_by_qc_metrics_allowlist(
    files: Sequence[Path | str],
    qc_csv: Path | str,
) -> tuple[list[Path], dict[str, str]]:
    """Keep FITS whose ``qc_metrics.csv`` row has ``status == 'ok'`` (exact match).

    Frames on disk but absent from the CSV, or with any non-ok status, are excluded.
    Matching uses normalized resolved absolute paths (casefold).
    """
    status_map = load_qc_metrics_status_by_path(qc_csv)
    ok_keys = {k for k, v in status_map.items() if v == "ok"}
    selected: list[Path] = []
    for raw in files:
        fp = Path(raw)
        if norm_fits_path_key(fp) in ok_keys:
            selected.append(fp)
    return selected, status_map


def build_prefilter_rejected_map(
    all_paths: Sequence[Path | str],
    passing_paths: Sequence[Path | str],
    *,
    reason: str = "rejected_prefilter_fwhm",
) -> dict[Path, str]:
    """Paths in ``all_paths`` not in ``passing_paths`` -> prefilter reject reason."""
    pass_keys = {norm_fits_path_key(p) for p in passing_paths}
    out: dict[Path, str] = {}
    for raw in all_paths:
        fp = Path(raw)
        if norm_fits_path_key(fp) not in pass_keys:
            out[fp.resolve()] = str(reason)
    return out


def resolve_preprocess_target_coordinates(
    *,
    db: VyvarDatabase,
    draft_id: int | None,
    ui_ra_deg: float | None,
    ui_dec_deg: float | None,
) -> tuple[float | None, float | None]:
    """Resolve preprocess target coordinates (the plate-solve hint).

    Priority (G14-03, Milan 2026-10-31: solved WCS > FITS header pointing > manifest centre):
    1) ``ui_ra_deg`` / ``ui_dec_deg``: the run's own pointing (RAM-QC medians, or the header / WCS
       pointing scan of the calibrated lights)
    2) median RA/DE from draft light rows (header pointing)
    None, None when neither resolves; the manifest centre is per setup (S6-FIX CENTRE-PER-SETUP) and is the
    per-light-folder fallback of :func:`manifest_setup_centres` in the QC pass, never a field-level hint.
    """
    try:
        if ui_ra_deg is not None and ui_dec_deg is not None:
            ra_ui = float(ui_ra_deg)
            de_ui = float(ui_dec_deg)
            if math.isfinite(ra_ui) and math.isfinite(de_ui):
                if not (abs(ra_ui) < 1e-9 and abs(de_ui) < 1e-9):
                    log_event(f"DEBUG: Preprocess using UI fallback coordinates: RA={ra_ui}, Dec={de_ui}")
                    return float(ra_ui), float(de_ui)
    except (TypeError, ValueError):
        pass
    if draft_id is not None:
        try:
            med_ra, med_de = draft_median_pointing_icrs_deg(db, int(draft_id))
            if med_ra is not None and med_de is not None:
                if math.isfinite(float(med_ra)) and math.isfinite(float(med_de)):
                    log_event(
                        f"DEBUG: Preprocess using draft median coordinates: RA={float(med_ra)}, Dec={float(med_de)}"
                    )
                    return float(med_ra), float(med_de)
        except Exception:  # noqa: BLE001
            pass
    return None, None


def manifest_setup_centres(db: VyvarDatabase, draft_id: int) -> dict[str, Any]:
    """``center_by_setup`` of the draft manifest (the MASTERSTAR WCS centre of each setup from an earlier run);
    {} when the manifest has none. A manifest that cannot be read is a WARNING, not a silent {}."""
    try:
        return dict((db.fetch_obs_draft_by_id(int(draft_id)) or {}).get("CENTER_BY_SETUP") or {})
    except Exception as exc:  # noqa: BLE001
        log_event(f"WARNING: Preprocess: manifest field centres not read ({exc})")
        return {}


def calibrated_paths_for_draft_apply_filters(
    archive_path: Path | str,
    db: VyvarDatabase,
    draft_id: int,
    *,
    source_dir: Path | str | None = None,
) -> list[Path]:
    """Calibrated FITS paths of the frames that pass frame QC (``IS_REJECTED`` 0 / NULL).

    ``IS_REJECTED`` is the frame QC decision (``frame_qc.run_frame_qc``, per channel); it is honoured as
    written. G12-12 / G01-31: the pooled FWHM cap (always 0 since frame QC) and its rescue pass are gone.
    """
    ap = Path(archive_path).expanduser()
    log_event(f"Preprocess: draft {draft_id} frames with IS_REJECTED=0 (frame QC)")
    cal_paths: list[Path] = []
    if source_dir is None:
        from draft_provenance import resolve_draft_lights_root

        src_root = resolve_draft_lights_root(ap, draft_id=int(draft_id), db=db).resolve()
    else:
        src_root = Path(source_dir).expanduser().resolve()
    rows = [_r for _r in db.fetch_draft_light_rows_for_quality(int(draft_id)) if _r.get("IS_REJECTED") in (None, 0)]
    log_event(f"DEBUG: Preprocess DB filter selected {len(rows)} rows.")

    for row in rows:
        raw = Path(str(row.get("FILE_PATH") or ""))
        m = _archive_raw_to_calibrated_light(ap, raw)
        if m is not None:
            cand_m, _cand_root = m
            try:
                if cand_m.is_file() and cand_m.resolve().is_relative_to(src_root):
                    cal_paths.append(cand_m.resolve())
                    continue
            except Exception:  # noqa: BLE001
                pass
        cand = src_root / raw.name
        if cand.is_file():
            cal_paths.append(cand)
            continue
        resolved_raw = _resolve_draft_light_raw_path(ap, raw)
        if resolved_raw is None or not resolved_raw.is_file():
            continue
        try:
            resolved_raw.relative_to(src_root)
        except ValueError:
            continue
        cal_paths.append(resolved_raw)
    if not cal_paths and src_root.is_dir():
        # Final safety net: keep DB-filter intent by matching the kept DB file names to on-disk FITS.
        # G01-45: exact stem match (legacy ``proc_`` prefix allowed) - a substring match took Light_10..19
        # for a kept Light_1.
        disk_fits = [p for p in _iter_fits_recursive(src_root) if p.is_file()]
        db_names = [Path(str(r.get("FILE_PATH") or "")).name for r in rows]
        db_stems = {Path(n).stem.lower() for n in db_names if str(n).strip()}
        matched: list[Path] = []
        for fp in disk_fits:
            s = fp.stem.lower()
            if s in db_stems or (s.startswith("proc_") and s[len("proc_"):] in db_stems):
                matched.append(fp)
        if matched:
            cal_paths = sorted(matched)
            log_event(
                f"INFO: Disk fallback selected {len(cal_paths)} FITS files from {src_root} (DB-name matched)."
            )
        elif disk_fits:
            # Never take every FITS regardless of QC: rejected frames would re-enter.
            log_event(
                f"WARNING: {len(disk_fits)} FITS in {src_root} but none matches a kept QC row - nothing selected."
            )
    return cal_paths


def resolve_obs_file_to_processed_fits(
    archive_path: Path | str | None,
    obs_file_path: str,
    *,
    setup_name: str | None = None,
    draft_id: int | None = None,
    db: VyvarDatabase | None = None,
) -> Path | None:
    """Map ``manifest files[].FILE_PATH`` onto the draft lights FITS for MASTERSTAR / preprocess.

    VYVAR-calibrated: ``processed/lights/.../proc_*.fits`` (legacy alias kept for callers).
    Pre-calibrated: ``non_calibrated/lights/<setup>/`` - the import frame is the frame (no proc_ remap).
    """
    if archive_path is None:
        return None
    from draft_provenance import draft_archive_root, is_pre_calibrated_draft

    ap = draft_archive_root(Path(archive_path).expanduser())
    if not ap.is_dir():
        return None
    pre_cal = is_pre_calibrated_draft(ap, draft_id=draft_id, db=db)
    root = resolve_masterstar_input_root(
        ap,
        setup_name=setup_name,
        draft_id=draft_id,
        db=db,
    )
    if root is None or not root.exists():
        return None
    hit = _resolve_best_effort_path_under(
        root,
        str(obs_file_path),
        pre_calibrated=pre_cal,
    )
    if hit is None or _path_segments_forbidden_for_masterstar_physical_source(
        hit, pre_calibrated=pre_cal
    ):
        return None
    return hit


def _partition_detrended_by_subfolder(files: list[Path], detrended_root: Path) -> dict[str, list[Path]]:
    """Group detrended FITS by full parent subpath under ``detrended_root``.

    This preserves on-disk structure (e.g. ``NoFilter_120_2`` or deeper nested layouts)
    so alignment, zero-level correction and DAO detection run on the exact same file tree.
    """
    root = detrended_root.resolve()
    out: dict[str, list[Path]] = {}
    for fp in files:
        p = Path(fp)
        try:
            rel = p.relative_to(root)
        except ValueError:
            continue
        parent_rel = rel.parent
        key = "" if str(parent_rel) == "." else parent_rel.as_posix()
        out.setdefault(key, []).append(p)
    for k in out:
        out[k].sort()
    return out


def qc_enrich_calibrated_lights_in_place(
    *,
    calibrated_root: Path,
    progress_cb: Callable[..., None] | None = None,
    inject_pointing_ra_deg: float | None = None,
    inject_pointing_dec_deg: float | None = None,
    inject_pointing_only_if_missing: bool = True,
    only_paths: Sequence[Path | str] | None = None,
    prefilter_rejected: Mapping[Path | str, str] | None = None,
    db: VyvarDatabase | None = None,
    draft_id: int | None = None,
    app_config: AppConfig,
) -> pd.DataFrame:
    """In-place QC enrichment on ``calibrated/lights`` (no ``processed/`` copy tree).

    Mutates calibrated FITS in place:
    - optional order-N polynomial sky-surface subtract (``preprocess_sky_surface_order``)
    - FWHM / elongation QC headers (``VY_FWHM``, ``VY_ELONG``, ``VYVARPR``, ...)
    - INV-CAL-02 stage stamp when pixels change (``VYCALSTG``, ``VYCALSUM``)

    Optional ``inject_pointing_*``: write ``VYTARGRA`` / ``VYTARGDE`` (deg ICRS) for plate-solve hints.

    Parallelism: :func:`parallel_frames.stage_workers` (the run's RAM and CPU budget).
    """
    cfg = app_config
    calibrated_root = Path(calibrated_root)

    centre_by_setup: dict[str, Any] = {}
    if db is not None and draft_id is not None:
        _ra_eff, _de_eff = resolve_preprocess_target_coordinates(
            db=db,
            draft_id=int(draft_id),
            ui_ra_deg=inject_pointing_ra_deg,
            ui_dec_deg=inject_pointing_dec_deg,
        )
        if _ra_eff is not None and _de_eff is not None:
            inject_pointing_ra_deg = float(_ra_eff)
            inject_pointing_dec_deg = float(_de_eff)
        else:
            centre_by_setup = manifest_setup_centres(db, int(draft_id))

    log_event(
        f"Preprocess: running QC in-place on draft lights ({calibrated_root})"
    )
    log_milestone(f"[PREPROCESS] start in-place QC sky_order={int(cfg.preprocess_sky_surface_order)}")
    _out = _qc_enrich_calibrated_in_place(
        calibrated_root,
        app_config=cfg,
        target_ra=inject_pointing_ra_deg,
        target_dec=inject_pointing_dec_deg,
        target_by_setup=centre_by_setup,
        inject_pointing_only_if_missing=inject_pointing_only_if_missing,
        only_paths=only_paths,
        prefilter_rejected=prefilter_rejected,
        progress_cb=progress_cb,
        draft_id=draft_id,
        db=db,
    )
    _skip_n = int(_out.get("sky_surface_skip_count") or 0)
    if _skip_n:
        log_milestone(f"[PREPROCESS] sky_surface skipped {_skip_n} frame(s) (VY_SKYSF guard)")
    df = pd.DataFrame(_out.get("results") or [])
    df.attrs["preprocess_sky_summary"] = {
        "sky_surface_skip_count": _skip_n,
    }
    return df


# --- moved from pipeline_astrometry (LAYER-2026-10) ---


def resolve_masterstar_input_root(
    archive_path: Path | str,
    setup_name: str | None = None,
    *,
    draft_id: int | None = None,
    db: VyvarDatabase | None = None,
) -> Path | None:
    """Pick MASTERSTAR input root under draft lights[/setup_name] with robust fallback.

    Uses ``calibrated/lights`` or ``non_calibrated/lights`` (via :func:`resolve_draft_lights_root`).

    When ``setup_name`` is explicitly provided, only that subdir is considered - no cross-group scan.
    Returns ``None`` when the requested setup is missing/empty (multi-group safe).
    """
    from draft_provenance import draft_archive_root, resolve_draft_lights_root

    ap = draft_archive_root(Path(archive_path).expanduser())
    lights_root = resolve_draft_lights_root(ap, draft_id=draft_id, db=db)
    setup_key = str(setup_name or "").strip()

    def _pick_under(base: Path) -> Path | None:
        if setup_key:
            cand = base / setup_key
            if cand.is_dir() and any(_iter_fits_recursive(cand)):
                return cand
            log_event(
                f"WARN: MASTERSTAR input root for setup {setup_key!r} not found under {base}; "
                "refusing cross-group fallback"
            )
            return None
        if base.is_dir():
            subdirs = sorted([d for d in base.iterdir() if d.is_dir()], key=lambda p: p.name.casefold())
            for sd in subdirs:
                if any(_iter_fits_recursive(sd)):
                    return sd
            if any(_iter_fits_recursive(base)):
                return base
        return None

    hit = _pick_under(lights_root)
    if hit is not None:
        return hit
    return lights_root if not setup_key else None


def _path_segments_forbidden_for_masterstar_physical_source(
    p: Path,
    *,
    pre_calibrated: bool = False,
) -> bool:
    """``True`` if resolved path must not be used as MASTERSTAR physical source.

    In VYVAR-calibrated mode, ``non_calibrated/`` and ``raw/`` are forbidden (use processed/calibrated).
    In pre-calibrated mode, ``non_calibrated/lights/`` **is** the valid source tree; only ``raw/`` is forbidden.
    """
    try:
        parts = Path(p).resolve().parts
    except OSError:
        parts = Path(p).parts
    bad = {"raw"} if pre_calibrated else {"raw", "non_calibrated"}
    return any(seg.casefold() in bad for seg in parts)


def _path_is_under_tree(root: Path, p: Path) -> bool:
    try:
        Path(p).resolve().relative_to(Path(root).resolve())
        return True
    except (OSError, ValueError):
        return False


def _pick_preferred_masterstar_basename_hit(
    hits: list[Path],
    *,
    pre_calibrated: bool = False,
) -> Path | None:
    """With several basename matches prefer ``proc_*.fits`` (processed frame)."""
    if not hits:
        return None
    clean = [
        h
        for h in hits
        if not _path_segments_forbidden_for_masterstar_physical_source(h, pre_calibrated=pre_calibrated)
    ]
    if not clean:
        return None
    proc_first = [h for h in clean if h.name.casefold().startswith("proc_")]
    use = proc_first if proc_first else clean
    return sorted(use, key=lambda x: str(x).casefold())[0]


def _basename_key(name: str) -> str:
    """Case-folded file name without the ``proc_`` prefix of a processed frame."""
    n = str(name).casefold()
    return n[5:] if n.startswith("proc_") else n


def fits_basename_index(root: Path, *, pre_calibrated: bool = False) -> dict[str, list[Path]]:
    """G03-41: one scan of ``root``: basename key (:func:`_basename_key`) -> FITS paths allowed as a
    MASTERSTAR physical source. Build it once and pass it to every
    :func:`_resolve_best_effort_path_under` of a loop (each call scanned the whole tree)."""
    idx: dict[str, list[Path]] = {}
    for x in _iter_fits_recursive(root):
        if not _path_segments_forbidden_for_masterstar_physical_source(x, pre_calibrated=bool(pre_calibrated)):
            idx.setdefault(_basename_key(x.name), []).append(x)
    return idx


def _resolve_best_effort_path_under(
    root: Path,
    raw_path: str,
    *,
    pre_calibrated: bool = False,
    index: dict[str, list[Path]] | None = None,
) -> Path | None:
    """Map an manifest files[].FILE_PATH (often archived raw path) to an existing file under ``root``.

    Strategy: exact relative join, else the same basename (optionally with the ``proc_`` prefix)
    anywhere under ``root`` (imports may store different path bases: absolute vs relative,
    calibrated vs processed). G03-41: no suffix match ("flight_1.fits" is not "light_1.fits");
    ``index`` = :func:`fits_basename_index` of ``root`` (built here when not given).
    """
    _forbid_kw = {"pre_calibrated": bool(pre_calibrated)}
    rp = str(raw_path or "").strip()
    if not rp:
        return None
    p = Path(rp)
    # If already absolute and exists under root tree, accept.
    if p.is_absolute():
        try:
            rel = p.resolve().relative_to(root.resolve())
            cand = (root / rel).resolve()
            if (
                cand.is_file()
                and _path_is_under_tree(root, cand)
                and not _path_segments_forbidden_for_masterstar_physical_source(cand, **_forbid_kw)
            ):
                return cand
        except Exception:  # noqa: BLE001
            pass
    # If relative, try directly.
    cand2 = (root / p).resolve()
    if (
        cand2.is_file()
        and _path_is_under_tree(root, cand2)
        and not _path_segments_forbidden_for_masterstar_physical_source(cand2, **_forbid_kw)
    ):
        return cand2
    # Explicit processed-name fallback in the same folder (e.g. Light_066.fits -> proc_Light_066.fits).
    if not pre_calibrated:
        try:
            cand2_proc = cand2.with_name(_safe_proc_name(cand2.name)).resolve()
            if (
                cand2_proc.is_file()
                and _path_is_under_tree(root, cand2_proc)
                and not _path_segments_forbidden_for_masterstar_physical_source(cand2_proc, **_forbid_kw)
            ):
                return cand2_proc
        except Exception:  # noqa: BLE001
            pass
    # Same basename (with or without the ``proc_`` prefix) anywhere under root.
    name = p.name
    if not name:
        return None
    if index is None:
        index = fits_basename_index(root, pre_calibrated=pre_calibrated)
    hits = list(index.get(_basename_key(name), ()))
    return _pick_preferred_masterstar_basename_hit(hits, pre_calibrated=pre_calibrated)
