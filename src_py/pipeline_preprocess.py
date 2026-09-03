"""Moved from pipeline.py (CONSOLIDATE-01E6a). Facade re-exports these names.

Calibrated->processed orchestration / QC filters.
The four giants stay in pipeline.py this wave (E6b).
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import numpy as np
from astropy.io import fits
import pandas as pd
from config import AppConfig
from database import VyvarDatabase
from infolog import log_event, log_milestone
from utils import iter_fits_paths_recursive as _iter_fits_recursive
from pipeline_calibrate import (
    _qc_enrich_calibrated_in_place,
    _resolve_draft_light_raw_path,
    draft_median_pointing_icrs_deg,
    norm_fits_path_key,
)

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


def load_qc_metrics_status_by_path(qc_csv: Path | str) -> dict[str, str]:
    """Map normalized absolute FITS path -> ``status`` from ``qc_metrics.csv``."""
    p = Path(qc_csv)
    if not p.is_file():
        raise FileNotFoundError(f"qc_metrics.csv not found: {p}")
    df = pd.read_csv(p)
    if "status" not in df.columns:
        raise ValueError(f"qc_metrics.csv missing status column: {p}")
    src_col = "src" if "src" in df.columns else "dst"
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        raw = row.get(src_col)
        if raw is None or (isinstance(raw, float) and not math.isfinite(float(raw))):
            continue
        out[norm_fits_path_key(str(raw))] = str(row["status"]).strip()
    return out


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
    """Resolve preprocess target coordinates with DB-first priority.

    Priority:
    1) ``draft manifest.CENTEROFFIELDRA/DE`` (finite pair; **0/0** sa povazuje za nevyplnene - pokracuje sa dalej)
    2) UI values
    3) median RA/DE from draft light rows
    """
    if draft_id is not None:
        try:
            drow = db.fetch_obs_draft_by_id(int(draft_id)) or {}
            ra_db = drow.get("CENTEROFFIELDRA")
            de_db = drow.get("CENTEROFFIELDDE")
            if ra_db is not None and de_db is not None:
                ra_f = float(ra_db)
                de_f = float(de_db)
                if math.isfinite(ra_f) and math.isfinite(de_f):
                    if not (abs(ra_f) < 1e-9 and abs(de_f) < 1e-9):
                        log_event(
                            f"INFO: Preprocessing forced to DB coordinates RA:{ra_f}, Dec:{de_f} for stability."
                        )
                        return float(ra_f), float(de_f)
                    log_event(
                        "DEBUG: draft manifest center is 0/0 - beriem ako nevyplnene; skusam UI, potom median z manifest files[]."
                    )
        except Exception:  # noqa: BLE001
            pass
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


def calibrated_paths_for_draft_apply_filters(
    archive_path: Path | str,
    db: VyvarDatabase,
    draft_id: int,
    *,
    fwhm_max_px: float,
    drift_max_arcmin: float = 0.0,
    source_dir: Path | str | None = None,
) -> tuple[list[Path], list[Path]]:
    """Calibrated FITS paths that pass QC: ``IS_REJECTED`` 0/NULL + optional ``FWHM`` cap.

    When ``fwhm_max_px`` <= 0, no FWHM filter (still excludes manual reject).
    When ``fwhm_max_px`` > 0, rows with finite ``FWHM`` > max are excluded.
    Rows with missing ``FWHM`` (NULL/NaN) are kept.

    ``drift_max_arcmin`` is kept only for backward-compatible callers and is ignored.
    """
    ap = Path(archive_path).expanduser()
    fwhm_limit = fwhm_max_px
    log_event(
        f"DEBUG: Hladam subory pre Draft {draft_id} s FWHM <= {fwhm_limit}"
    )
    from draft_provenance import reset_manifest_light_is_rejected

    reset_manifest_light_is_rejected(db, int(draft_id))
    log_event(f"DEBUG: Draft {draft_id} reset: all files set to IS_REJECTED=0 before filtering.")
    cal_paths: list[Path] = []
    if source_dir is None:
        from draft_provenance import resolve_draft_lights_root

        src_root = resolve_draft_lights_root(ap, draft_id=int(draft_id), db=db).resolve()
    else:
        src_root = Path(source_dir).expanduser().resolve()
    lim_active = bool(fwhm_max_px is not None and float(fwhm_max_px) > 0)
    lim_v = float(fwhm_max_px) if lim_active else 0.0
    _ = drift_max_arcmin  # backward compatibility: cleaning is FWHM-only
    all_rows = db.fetch_draft_light_rows_for_quality(int(draft_id))
    if lim_active:
        _nonnull_cnt = sum(1 for _r in all_rows if _r.get("FWHM") is not None)
        if _nonnull_cnt > 0:
            rows = [
                _r
                for _r in all_rows
                if _r.get("IS_REJECTED") in (None, 0)
                and _r.get("FWHM") is not None
                and float(_r["FWHM"]) <= float(lim_v)
            ]
        else:
            rows = [
                _r
                for _r in all_rows
                if _r.get("IS_REJECTED") in (None, 0)
                and (_r.get("FWHM") is None or float(_r["FWHM"]) <= float(lim_v))
            ]
        log_event(
            f"DEBUG: Preprocess DB filter selected {len(rows)} rows (limit={float(lim_v):.3f} px, strict, nonnull_fwhm={_nonnull_cnt})."
        )
    else:
        log_event("DEBUG: Preprocess filter - FWHM limit disabled (0/None); keeping all non-rejected frames.")
        rows = [_r for _r in all_rows if _r.get("IS_REJECTED") in (None, 0)]
        log_event(f"DEBUG: Preprocess DB filter selected {len(rows)} rows (FWHM disabled).")

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
    if not cal_paths:
        rescue_rows = [
            _r
            for _r in db.fetch_draft_light_rows_for_quality(int(draft_id))
            if _r.get("IS_REJECTED") in (None, 0)
        ]
        if rescue_rows:
            log_event(f"INFO: Rescue pass found {len(rescue_rows)} files by ignoring QC filters.")
        for row in rescue_rows:
            raw = Path(str(row["FILE_PATH"] or ""))
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
        # Final safety net: keep DB-filter intent by fuzzy matching selected DB file names to on-disk FITS.
        disk_fits = [p for p in _iter_fits_recursive(src_root) if p.is_file()]
        db_names = [Path(str(r.get("FILE_PATH") or "")).name for r in rows]
        db_stems = {Path(n).stem.lower() for n in db_names if str(n).strip()}
        matched: list[Path] = []
        for fp in disk_fits:
            s = fp.stem.lower()
            if any((st in s) or (s in st) for st in db_stems):
                matched.append(fp)
        if matched:
            cal_paths = sorted(matched)
            log_event(
                f"INFO: Disk fallback selected {len(cal_paths)} FITS files from {src_root} (DB-name matched)."
            )
        elif disk_fits:
            cal_paths = sorted(disk_fits)
            log_event(
                f"WARNING: Disk fallback used all {len(cal_paths)} FITS files from {src_root} (no DB-name match)."
            )
    return cal_paths, []


def resolve_obs_file_to_processed_fits(
    archive_path: Path | str | None,
    obs_file_path: str,
    *,
    setup_name: str | None = None,
    app_config: AppConfig | None = None,
    draft_id: int | None = None,
    db: VyvarDatabase | None = None,
) -> Path | None:
    """Map ``manifest files[].FILE_PATH`` onto the draft lights FITS for MASTERSTAR / preprocess.

    VYVAR-calibrated: ``processed/lights/.../proc_*.fits`` (legacy alias kept for callers).
    Pre-calibrated: ``non_calibrated/lights/<setup>/`` - the import frame is the frame (no proc_ remap).
    """
    from pipeline import (  # noqa: PLC0415
        _path_segments_forbidden_for_masterstar_physical_source,
        _resolve_best_effort_path_under,
        resolve_masterstar_input_root,
    )

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
        app_config=app_config,
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
    reject_fwhm_px: float | None = None,
    reject_elongation: float | None = None,
    progress_cb: Callable[..., None] | None = None,
    inject_pointing_ra_deg: float | None = None,
    inject_pointing_dec_deg: float | None = None,
    inject_pointing_only_if_missing: bool = True,
    only_paths: Sequence[Path | str] | None = None,
    prefilter_rejected: Mapping[Path | str, str] | None = None,
    db: VyvarDatabase | None = None,
    draft_id: int | None = None,
    app_config: AppConfig | None = None,
) -> pd.DataFrame:
    """In-place QC enrichment on ``calibrated/lights`` (no ``processed/`` copy tree).

    Mutates calibrated FITS in place:
    - optional order-N polynomial sky-surface subtract (``preprocess_sky_surface_order``)
    - FWHM / elongation QC headers (``VY_FWHM``, ``VY_ELONG``, ``VYVARPR``, ...)
    - INV-CAL-02 stage stamp when pixels change (``VY_CALSTAGE``, ``VY_CALDATASUM``)

    Optional ``inject_pointing_*``: write ``VYTARGRA`` / ``VYTARGDE`` (deg ICRS) for plate-solve hints.

    Parallelism: auto worker count from host CPU (capped) or env
    ``VYVAR_PARALLEL_WORKERS`` / legacy env (see :func:`_vyvar_parallel_worker_count`).
    """
    cfg = app_config or AppConfig()
    calibrated_root = Path(calibrated_root)

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

    log_event(
        f"Preprocess: running QC in-place on draft lights ({calibrated_root})"
    )
    log_milestone(f"[PREPROCESS] start in-place QC sky_order={int(cfg.preprocess_sky_surface_order)}")
    _out = _qc_enrich_calibrated_in_place(
        calibrated_root,
        app_config=cfg,
        fwhm_reject_limit=reject_fwhm_px,
        elong_reject_limit=reject_elongation,
        target_ra=inject_pointing_ra_deg,
        target_dec=inject_pointing_dec_deg,
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
    if _out.get("sky_surface_force_reapply"):
        LOGGER.warning(
            "preprocess_sky_surface_force_reapply=True: sky-surface guard bypassed on this run"
        )
    df = pd.DataFrame(_out.get("results") or [])
    df.attrs["preprocess_sky_summary"] = {
        "sky_surface_skip_count": _skip_n,
        "sky_surface_force_reapply": bool(_out.get("sky_surface_force_reapply")),
    }
    return df


def preprocess_calibrated_to_processed(
    *,
    calibrated_root: Path,
    processed_root: Path,
    reject_fwhm_px: float | None = None,
    reject_elongation: float | None = None,
    use_gpu_if_available: bool = False,
    progress_cb: Callable[..., None] | None = None,
    inject_pointing_ra_deg: float | None = None,
    inject_pointing_dec_deg: float | None = None,
    inject_pointing_only_if_missing: bool = True,
    only_paths: Sequence[Path | str] | None = None,
    prefilter_rejected: Mapping[Path | str, str] | None = None,
    db: VyvarDatabase | None = None,
    draft_id: int | None = None,
    app_config: AppConfig | None = None,
) -> pd.DataFrame:
    """Deprecated alias for :func:`qc_enrich_calibrated_lights_in_place`.

    ``processed_root`` and ``use_gpu_if_available`` are ignored (legacy API).
    """
    _ = processed_root, use_gpu_if_available
    return qc_enrich_calibrated_lights_in_place(
        calibrated_root=calibrated_root,
        reject_fwhm_px=reject_fwhm_px,
        reject_elongation=reject_elongation,
        progress_cb=progress_cb,
        inject_pointing_ra_deg=inject_pointing_ra_deg,
        inject_pointing_dec_deg=inject_pointing_dec_deg,
        inject_pointing_only_if_missing=inject_pointing_only_if_missing,
        only_paths=only_paths,
        prefilter_rejected=prefilter_rejected,
        db=db,
        draft_id=draft_id,
        app_config=app_config,
    )


def _qc_suggest_thresholds(df: "pd.DataFrame") -> dict[str, float | int | None]:
    """Compute best/worst and robust suggested reject thresholds."""
    import numpy as np

    out: dict[str, float | int | None] = {
        "fwhm_min": None,
        "fwhm_median": None,
        "fwhm_max": None,
        "elong_min": None,
        "elong_median": None,
        "elong_max": None,
        "suggest_reject_fwhm_px": None,
        "suggest_reject_elongation": None,
        "n_qc": 0,
    }
    if df is None or df.empty:
        return out

    def _robust_suggest(x: "np.ndarray", k: float) -> float | None:
        x = x[np.isfinite(x)]
        if x.size < 5:
            return None
        med = float(np.nanmedian(x))
        mad = float(np.nanmedian(np.abs(x - med))) + 1e-6
        sigma = 1.4826 * mad
        return float(med + k * sigma)

    f = np.asarray(df.get("fwhm_px"), dtype=np.float64)
    e = np.asarray(df.get("elongation"), dtype=np.float64)
    f_ok = f[np.isfinite(f)]
    e_ok = e[np.isfinite(e)]
    out["n_qc"] = int(min(f_ok.size, e_ok.size) if (f_ok.size and e_ok.size) else max(f_ok.size, e_ok.size))

    if f_ok.size:
        out["fwhm_min"] = float(np.nanmin(f_ok))
        out["fwhm_median"] = float(np.nanmedian(f_ok))
        out["fwhm_max"] = float(np.nanmax(f_ok))
        # Suggest: median + 3*MAD_sigma (conservative)
        out["suggest_reject_fwhm_px"] = _robust_suggest(f_ok, k=3.0)
    if e_ok.size:
        out["elong_min"] = float(np.nanmin(e_ok))
        out["elong_median"] = float(np.nanmedian(e_ok))
        out["elong_max"] = float(np.nanmax(e_ok))
        # Suggest: median + 4*MAD_sigma (elongation is often tighter)
        out["suggest_reject_elongation"] = _robust_suggest(e_ok, k=4.0)

    return out


