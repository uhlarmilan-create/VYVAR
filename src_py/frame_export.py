"""Moved from pipeline.py (CONSOLIDATE-01E6b). Facade re-exports this name.

export_per_frame_catalogs: per-frame catalog CSV export with MP pool.
Workers, initializer, and _EXPORT_PER_FRAME_WORKER_STATE live in
pipeline_catalog (E6a); imported E3 here.
detect_stars_and_match_catalog moves to catalog_match.py (C1);
imported via call-time facade import inside the body.
"""
from __future__ import annotations

import json
import logging
import math
import pickle
from pathlib import Path
from typing import Any, Sequence

from astropy.io import fits
import pandas as pd

from config import AppConfig
from pipeline_constants import DAO_FWHM_FALLBACK_PX
from infolog import log_event
from utils import iter_fits_paths_recursive as _iter_fits_recursive
from pipeline_astrometry import (
    _equipment_saturate_adu_from_db,
    _vyvar_df_to_csv,
)
from pipeline_catalog import (
    _estimate_catalog_frame_hw,
    _export_per_frame_disk_worker_task,
    _export_per_frame_ram_worker_task,
    _finalize_hybrid_bkg_fallback_sidecar,
    _init_export_per_frame_worker,
    find_qc_metrics_csv,
)

LOGGER = logging.getLogger("pipeline")

#: per_frame_catalog_index.csv: one row per aligned frame (OUTPUT-REVIEW-2026-10 step 6).
PER_FRAME_INDEX_COLUMNS: tuple[str, ...] = ("file", "status", "csv", "n_detected", "n_matched")


def write_per_frame_index(rows: Sequence[dict[str, Any]], path: Path | str) -> Path:
    """Write ``per_frame_catalog_index.csv`` (the index columns of the per-frame result rows)."""
    df = pd.DataFrame([{c: r.get(c) for c in PER_FRAME_INDEX_COLUMNS} for r in rows],
                      columns=list(PER_FRAME_INDEX_COLUMNS))
    _vyvar_df_to_csv(df, Path(path))
    return Path(path)


def export_per_frame_catalogs(
    *,
    frames_root: Path,
    platesolve_dir: Path,
    max_catalog_rows: int = 12000,
    catalog_match_max_sep_arcsec: float = 25.0,
    dao_threshold_sigma: float = 3.5,
    dao_fwhm_px: float | None = None,
    write_sidecar_csv_next_to_fits: bool = True,
    mirror_flat_platesolve_folder: bool = False,
    progress_cb: "callable | None" = None,
    masterstars_csv: Path | str | None = None,
    masterstar_fits: Path | str | None = None,
    equipment_saturate_adu: float | None = None,
    aligned_ram: "Sequence[tuple[str, fits.Header, Any]] | None" = None,
    aligned_target_dir: Path | str | None = None,
    defer_disk_writes: bool = False,
    app_config: AppConfig | None = None,
    draft_id: int | None = None,
    equipment_id: int | None = None,
    full_catalog_export: bool = False,
) -> dict[str, Any]:
    """For each FITS under ``frames_root`` with WCS: DAO + catalog table, write one CSV per frame.

    **MASTERSTAR lock (MASTER-LOCK-2026-10):** every frame is matched only to the MASTERSTAR catalogue
    (``masterstars_csv`` + ``masterstar_fits``, required): catalogue identities come from MASTERSTAR,
    positions and fluxes from the frame (DAO + forced photometry at the MASTERSTAR positions).

    ``<platesolve_dir>/per_frame_catalog_index.csv`` lists every file and CSV path.

    **Performance:** each frame still runs DAO + catalog match + disk CSV write (dominant cost for many lights).
    Parallelism: :func:`parallel_frames.stage_workers` / :func:`parallel_frames.map_frames` (the run's
    RAM and CPU budget, frames in order). RAM handoff ``aligned_ram`` sends serialized headers + float32
    pixels to the workers.

    **RAM handoff:** pass ``aligned_ram`` as ``(filename, header, ndarray)`` tuples plus ``aligned_target_dir`` to
    run catalog matching **without** re-reading aligned FITS from disk. With ``defer_disk_writes=True``, sidecar
    CSV (and optional flat mirror) are returned in ``deferred_csv_writes`` for the caller to flush after FITS.

    """
    import numpy as np

    _cfg_ap = app_config or AppConfig()

    ps = Path(platesolve_dir) if platesolve_dir is not None else None
    _qc_fwhm_by_name: dict[str, float] = {}
    _fwhm_night_median_px: float | None = None
    try:
        from aperture_policy import load_qc_fwhm_map  # noqa: PLC0415

        _qc_root = None
        if ps is not None:
            try:
                _qc_root = Path(ps).resolve().parents[1]
            except IndexError:
                _qc_root = None
        _qc_csv = find_qc_metrics_csv(_qc_root) if _qc_root is not None else None
        _qc_fwhm_by_name, _fwhm_night_median_px = load_qc_fwhm_map(_qc_csv)
    except Exception as _qc_exc:  # noqa: BLE001
        LOGGER.warning("[APERTURE-01] QC FWHM map not loaded: %s", _qc_exc)

    # GAIN-ONE-2026-10: the run's gain / read noise (MASTERSTAR header, EQUIPMENTS), not the raw
    # config defaults; used by the background-error ratio of the per-frame catalogs
    from rig_photometry import photometric_gain_read_noise  # noqa: PLC0415

    _ms_hdr_fe = None
    if masterstar_fits is not None and Path(masterstar_fits).is_file():
        try:
            _ms_hdr_fe = fits.getheader(str(masterstar_fits), 0)
        except OSError:
            _ms_hdr_fe = None
    _pg_fe = photometric_gain_read_noise(_ms_hdr_fe, _cfg_ap, equipment_id)
    _ap_st: dict[str, Any] = {
        "aperture_fwhm_factor": float(_cfg_ap.aperture_fwhm_factor),
        "annulus_inner_fwhm": float(_cfg_ap.annulus_inner_fwhm),
        "annulus_outer_fwhm": float(_cfg_ap.annulus_outer_fwhm),
        "database_path": str(Path(_cfg_ap.database_path).resolve()),
        "draft_id": int(draft_id) if draft_id is not None else None,
        "equipment_id": int(equipment_id) if equipment_id is not None else None,
        "fwhm_night_median_px": _fwhm_night_median_px,
        "qc_fwhm_by_name": dict(_qc_fwhm_by_name),
        "platesolve_dir": str(ps.resolve()) if ps is not None else "",
        "gain": float(_pg_fe.gain),
        "read_noise": float(_pg_fe.read_noise),
    }

    use_ram_inputs = aligned_ram is not None
    if use_ram_inputs and aligned_target_dir is None:
        raise ValueError("export_per_frame_catalogs: aligned_target_dir is required when aligned_ram is set")

    if (
        not write_sidecar_csv_next_to_fits
        and not mirror_flat_platesolve_folder
        and not defer_disk_writes
    ):
        write_sidecar_csv_next_to_fits = True

    root = Path(aligned_target_dir) if use_ram_inputs else Path(frames_root)
    if ps is None:
        ps = Path(platesolve_dir)
    out_flat = ps / "per_frame_catalogs"
    if mirror_flat_platesolve_folder:
        out_flat.mkdir(parents=True, exist_ok=True)

    work_ram: list[tuple[str, fits.Header, Any]] | None = None
    if use_ram_inputs:
        root.mkdir(parents=True, exist_ok=True)
        work_ram = sorted(list(aligned_ram), key=lambda t: t[0])
        files = [root / name for name, _, _ in work_ram]
    else:
        if full_catalog_export:
            files = sorted(_iter_fits_recursive(root))
        else:
            from epsf_frame_accounting import list_epsf_science_light_fits

            files = list_epsf_science_light_fits(root)

    if not files:
        return {
            "written": 0,
            "per_frame_dir": str(root),
            "per_frame_csv_mode": "sidecar" if write_sidecar_csv_next_to_fits else "none",
            "index_csv": "",
            "frames": [],
            "deferred_csv_writes": [],
        }

    total = len(files)
    _debug_logged_once = False
    master_tab: pd.DataFrame | None = None
    _msp = Path(masterstars_csv) if masterstars_csv is not None else None
    _msf = Path(masterstar_fits) if masterstar_fits is not None else None
    if _msp is None or not _msp.is_file() or _msf is None or not _msf.is_file():
        raise RuntimeError(
            "Per-frame catalog: MASTERSTAR lock needs masterstars_full_match.csv and MASTERSTAR.fits "
            f"(got {masterstars_csv!s}, {masterstar_fits!s})."
        )
    _mt = pd.read_csv(_msp, low_memory=False, dtype={"catalog_id": str, "name": str})
    if _mt.empty or "ra_deg" not in _mt.columns or "dec_deg" not in _mt.columns:
        raise RuntimeError(f"Per-frame catalog: {_msp.name} has no ra_deg/dec_deg rows.")
    master_tab = _mt
    LOGGER.info(
        "Per-frame catalog: MASTERSTAR lock (single catalog: %s, %s rows)",
        _msp.name,
        len(master_tab),
    )

    _sat_diag_ctx: Any | None = None
    _sat_diag_archive: str = ""
    _ref_ra_deg: float | None = None
    _ref_dec_deg: float | None = None
    _drift_ref_catalog_id: str | None = None
    try:
        from sat_diag import (  # noqa: PLC0415
            draft_archive_from_platesolve,
            run_sat_diag,
        )

        _arch = draft_archive_from_platesolve(ps)
        if _arch is not None:
            _sat_diag_archive = str(_arch)
            _eq_sat = equipment_saturate_adu
            if _eq_sat is None and equipment_id is not None:
                _eq_sat = _equipment_saturate_adu_from_db(int(equipment_id))
            _ref_hdr = fits.Header()
            if files:
                with fits.open(files[0], memmap=False) as _rh:
                    _ref_hdr = _rh[0].header
            _sat_diag_ctx = run_sat_diag(_arch, equipment_adu=_eq_sat, hdr=_ref_hdr)
            if _sat_diag_ctx.sat_adu is not None:
                equipment_saturate_adu = float(_sat_diag_ctx.sat_adu)
            from sat_diag import resolve_drift_ref_sky_deg  # noqa: PLC0415

            _frame_hint = Path(files[0]).name if files else None
            _ref_ra_deg, _ref_dec_deg, _drift_ref_catalog_id = resolve_drift_ref_sky_deg(
                ps, frame_name_hint=_frame_hint
            )
            if _ref_ra_deg is None and master_tab is not None and not getattr(master_tab, "empty", True):
                if "ra_deg" in master_tab.columns and "dec_deg" in master_tab.columns:
                    _mra = pd.to_numeric(master_tab["ra_deg"], errors="coerce")
                    _mde = pd.to_numeric(master_tab["dec_deg"], errors="coerce")
                    _flux_col = "flux" if "flux" in master_tab.columns else None
                    if _flux_col:
                        _ord = pd.to_numeric(master_tab[_flux_col], errors="coerce").fillna(0)
                        _j = int(_ord.idxmax())
                    else:
                        _j = 0
                    if math.isfinite(float(_mra.iloc[_j])) and math.isfinite(float(_mde.iloc[_j])):
                        _ref_ra_deg = float(_mra.iloc[_j])
                        _ref_dec_deg = float(_mde.iloc[_j])
            if _ref_ra_deg is not None:
                LOGGER.info(
                    "[SAT-DIAG] drift reference sky (%.5f, %.5f) from %s",
                    _ref_ra_deg,
                    _ref_dec_deg,
                    _frame_hint or "platesolve",
                )
            LOGGER.info(
                "[SAT-DIAG] sat_adu=%s source=%s lin_adu=%s (archive %s)",
                _sat_diag_ctx.sat_adu,
                _sat_diag_ctx.sat_source,
                _sat_diag_ctx.lin_adu,
                _sat_diag_archive,
            )
    except Exception as _sd_exc:  # noqa: BLE001
        LOGGER.warning("[SAT-DIAG] init skipped: %s", _sd_exc)

    _gauss_override: float | None = None
    try:
        if masterstar_fits is not None:
            _ms_gauss = Path(masterstar_fits)
            if _ms_gauss.is_file():
                with fits.open(_ms_gauss, memmap=False) as _gfh:
                    _ghdr = _gfh[0].header
                    # FWHM-ONE-2026-10: VY_FWHM is the elliptical Gaussian FWHM of frame QC
                    # (MASTERSTAR also carried a copy under an over-long name until 2026-10).
                    try:
                        _vyf = float(_ghdr.get("VY_FWHM"))
                        if 1.0 <= _vyf <= 15.0:
                            _gauss_override = _vyf
                    except (TypeError, ValueError):
                        pass
    except Exception:  # noqa: BLE001
        _gauss_override = None
    _ap_st["gaussian_fwhm_px_override"] = _gauss_override
    if _gauss_override is not None:
        log_event(
            f"[PHOT] gaussian_fwhm_px_override = {float(_gauss_override):.4f}px "
            "(MASTERSTAR VY_FWHM)"
        )
    else:
        log_event("[PHOT] gaussian_fwhm_px_override = None -> fallback to momentx0.619 per frame")

    _dao_fw_export = (
        float(dao_fwhm_px)
        if dao_fwhm_px is not None
        else DAO_FWHM_FALLBACK_PX
    )
    from parallel_frames import frame_bytes, map_frames, stage_workers  # noqa: PLC0415

    _ny, _nx = _estimate_catalog_frame_hw(work_ram if use_ram_inputs else None, files)
    n_workers = stage_workers(total, frame_bytes(_ny, _nx), stage="per-frame catalogs")

    deferred_csv_writes: list[tuple[Path, pd.DataFrame]] = []

    def _catalog_worker_state() -> dict[str, Any]:
        return {
            "master_tab": master_tab,
            "max_catalog_rows": int(max_catalog_rows),
            "catalog_match_max_sep_arcsec": float(catalog_match_max_sep_arcsec),
            "dao_threshold_sigma": float(dao_threshold_sigma),
            "dao_fwhm_px": float(_dao_fw_export),
            "equipment_saturate_adu": equipment_saturate_adu,
            "sat_diag_ctx_dict": (
                _sat_diag_ctx.to_json_dict() if _sat_diag_ctx is not None else None
            ),
            "sat_diag_archive": _sat_diag_archive,
            "sat_diag_ref_ra": _ref_ra_deg,
            "sat_diag_ref_dec": _ref_dec_deg,
            "sat_diag_ref_catalog_id": _drift_ref_catalog_id,
            "write_sidecar_csv_next_to_fits": write_sidecar_csv_next_to_fits,
            "mirror_flat_platesolve_folder": mirror_flat_platesolve_folder,
            "defer_disk_writes": defer_disk_writes,
            "out_flat": str(out_flat.resolve()),
            "aperture_fwhm_factor": float(_cfg_ap.aperture_fwhm_factor),
            "annulus_inner_fwhm": float(_cfg_ap.annulus_inner_fwhm),
            "annulus_outer_fwhm": float(_cfg_ap.annulus_outer_fwhm),
            "database_path": str(Path(_cfg_ap.database_path).resolve()),
            "draft_id": int(draft_id) if draft_id is not None else None,
            "equipment_id": int(equipment_id) if equipment_id is not None else None,
            "gaussian_fwhm_px_override": _gauss_override,
            "fwhm_night_median_px": _fwhm_night_median_px,
            "qc_fwhm_by_name": dict(_qc_fwhm_by_name),
            "platesolve_dir": str(ps.resolve()),
            "observer_lat": float(_cfg_ap.observer_lat),
            "observer_lon": float(_cfg_ap.observer_lon),
            "observer_alt_m": float(_cfg_ap.observer_alt_m),
            "objects_db_path": str(_cfg_ap.objects_db_path or ""),
        }

    # PER-FRAME-ONE-PATH-2026-10 (Milan 2026-10-13): one per-frame catalog function for any
    # number of workers (``_export_per_frame_run_catalog_core``). The former one-worker copy
    # (_run_one_catalog) lacked the MASTERSTAR join (zone, bp_rp, phot_g_mean_mag, ...), the
    # forced photometry and the ePSF records -> on a 1-2 core machine comparison-star selection failed
    # ('numpy.float64' object has no attribute 'isna' in the comp pool admission).
    ws = _catalog_worker_state()
    # EXPORT-RAM-PEAK-2026-10: a RAM-handoff frame is serialised only when its task is submitted.
    # Packing every frame up front held a second copy of all aligned frames next to the RAM
    # buffer (2 x the night in memory): Brno 534, 97 frames 3552x2664, was killed out of memory
    # on a 7 GB machine even with one worker.
    if use_ram_inputs and work_ram is not None:
        keys = [str(it[0]) for it in work_ram]
        _ram_items = work_ram

        def _pack(i: int) -> Any:
            _name, _hdr, _arr = _ram_items[i]
            d = np.ascontiguousarray(_arr, dtype=np.float32)
            return (str((root / _name).resolve()), pickle.dumps(_hdr.copy()), d.tobytes(), int(d.shape[0]), int(d.shape[1]))

        task_fn: Any = _export_per_frame_ram_worker_task
    else:
        keys = [fp.name for fp in files]
        _disk_packs = [str(fp.resolve()) for fp in files]

        def _pack(i: int) -> Any:
            return _disk_packs[i]

        task_fn = _export_per_frame_disk_worker_task

    by_key: dict[str, dict[str, Any]] = {}
    done = 0

    def _collect(key: str, res: dict[str, Any]) -> None:
        nonlocal _debug_logged_once, done
        if (not _debug_logged_once) and isinstance(res, dict) and res.get("debug_pixel_match") is not None:
            _debug_logged_once = True
            try:
                log_event(
                    "DEBUG: per-frame debug_pixel_match (first frame): "
                    + json.dumps(res.get("debug_pixel_match"), ensure_ascii=False, default=str)
                )
            except Exception:  # noqa: BLE001
                log_event(f"DEBUG: per-frame debug_pixel_match (first frame): {res.get('debug_pixel_match')}")
        for msg in res.get("infolog_messages") or []:
            if msg:
                log_event(str(msg))
        for p_str, dff in res.get("deferred_writes") or []:
            deferred_csv_writes.append((Path(p_str), dff))
        by_key[key] = res
        done += 1
        if progress_cb is not None:
            progress_cb(done, total, f"Catalog: {key}")

    # The pool keeps at most two frames per worker in flight, so the packed RAM frames are not
    # all alive at once; results come back in frame order.
    for key, res in zip(
        keys,
        map_frames(task_fn, range(len(keys)), workers=n_workers, initializer=_init_export_per_frame_worker,
                   ctx=ws, pack=_pack, stage="per-frame catalogs"),
        strict=True,
    ):
        _collect(key, res)
    rows_out = []
    frame_rows: dict[str, dict[str, Any]] = {}
    for key in keys:
        rr = dict(by_key[key])
        rr.pop("deferred_writes", None)
        rr.pop("infolog_messages", None)
        _fv = rr.pop("frame_values", None)
        if _fv and rr.get("csv"):
            frame_rows[str(rr["csv"])] = _fv
        rows_out.append(rr)
    if not defer_disk_writes and write_sidecar_csv_next_to_fits and frame_rows:
        from proc_frame_store import write_frame_tables  # noqa: PLC0415

        write_frame_tables(frame_rows)

    index_path = ps / "per_frame_catalog_index.csv"
    if not defer_disk_writes:
        write_per_frame_index(rows_out, index_path)
    n_ok = sum(1 for r in rows_out if r.get("status") == "ok")
    _hybrid_stats: dict[str, Any] = {}
    if not defer_disk_writes:
        _hybrid_stats = _finalize_hybrid_bkg_fallback_sidecar(
            root,
            write_sidecar=bool(write_sidecar_csv_next_to_fits),
            gain=float(_ap_st["gain"]),
            read_noise=float(_ap_st["read_noise"]),
            setup_label=str(root.name),
        )
    if _sat_diag_ctx is not None and _sat_diag_archive:
        try:
            from sat_diag import commit_sat_diag_provenance  # noqa: PLC0415

            # FIX-RAWPEAKS-2026-10: the per-frame rows now carry raw_peaks_used (sat_diag.json said
            # false although every frame got its raw peaks: the flag was never in the rows)
            _placed_raw = any(bool(r.get("raw_peaks_used")) for r in rows_out)
            commit_sat_diag_provenance(
                _sat_diag_ctx,
                _sat_diag_archive,
                placed_aperture_used=_placed_raw,
            )
        except Exception as _sd_write_exc:  # noqa: BLE001
            LOGGER.warning("[SAT-DIAG] provenance commit skipped: %s", _sd_write_exc)
    return {
        "written": int(n_ok),
        "per_frame_dir": str(root),
        "per_frame_csv_mode": "sidecar" if write_sidecar_csv_next_to_fits else ("flat_mirror" if mirror_flat_platesolve_folder else "none"),
        "index_csv": str(index_path),
        "frames": rows_out,
        "mirror_flat_platesolve_folder": bool(mirror_flat_platesolve_folder),
        "deferred_csv_writes": list(deferred_csv_writes) if defer_disk_writes else [],
        "frame_rows": frame_rows,
        "hybrid_bkg_fallback": _hybrid_stats,
        "photometric_gain": float(_ap_st["gain"]),
        "photometric_read_noise": float(_ap_st["read_noise"]),
    }
