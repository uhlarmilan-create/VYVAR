"""Moved from pipeline.py (CONSOLIDATE-01E6b). Facade re-exports this name.

export_per_frame_catalogs: per-frame catalog CSV export with MP pool.
Workers, initializer, and _EXPORT_PER_FRAME_WORKER_STATE live in
pipeline_catalog (E6a); imported E3 here.
detect_stars_and_match_catalog moves to catalog_match.py (C1);
imported via call-time facade import inside the body.
"""
from __future__ import annotations

import contextlib
import json
import logging
import math
import multiprocessing
import os
import pickle
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.wcs import FITSFixedWarning
import pandas as pd

from config import AppConfig
from database import VyvarDatabase
from infolog import log_event
from proc_frame_store import proc_csv_path_for_aligned_fits
from utils import iter_fits_paths_recursive as _iter_fits_recursive
from pipeline_calibrate import (
    _has_valid_wcs,
    _resolve_draft_light_raw_path,
)
from pipeline_astrometry import (
    _equipment_saturate_adu_from_db,
    _export_catalog_psf_st_fields,
    _vyvar_df_to_csv,
    _vyvar_per_frame_csv_workers,
    resolve_plate_solve_fov_deg_hint,
)
from pipeline_catalog import (
    _EXPORT_PER_FRAME_WORKER_STATE,
    _apply_aperture_catalog_enhancements_from_st,
    _apply_exo_host_columns_to_proc_df,
    _effective_field_catalog_cone_radius_deg,
    _epsf_fit_catalog_ids,
    _estimate_catalog_frame_hw,
    _export_per_frame_disk_worker_task,
    _export_per_frame_ram_worker_task,
    _extract_airmass_from_header,
    _field_catalog_cone_meta_path,
    _fill_psf_catalog_columns,
    _finalize_hybrid_bkg_fallback_sidecar,
    _init_export_per_frame_worker,
    _invalidate_field_catalog_cone_cache_if_needed,
    _prefetch_export_shared_catalog_for_process_pool,
    _proc_catalog_keep_matched_rows_only,
    _proc_deduplicate_matched_catalog_rows,
    _proc_drop_unmatched_dao_rows,
    _query_gaia_local,
    _query_vsx_local,
    _vyvar_cap_mp_workers_for_catalog,
    _write_field_catalog_cone_meta,
    build_ucac_catalog_kdtree,
    detect_stars_match_master_reference,
    find_qc_metrics_csv,
)

LOGGER = logging.getLogger("pipeline")
from catalog_match import detect_stars_and_match_catalog


def export_per_frame_catalogs(
    *,
    frames_root: Path,
    platesolve_dir: Path,
    max_catalog_rows: int = 12000,
    catalog_match_max_sep_arcsec: float = 25.0,
    saturate_level_fraction: float = 0.999,
    faintest_mag_limit: float | None = None,
    dao_threshold_sigma: float = 3.5,
    dao_fwhm_px: float | None = None,
    write_sidecar_csv_next_to_fits: bool = True,
    mirror_flat_platesolve_folder: bool = False,
    progress_cb: "callable | None" = None,
    masterstars_csv: Path | str | None = None,
    masterstar_fits: Path | str | None = None,
    use_master_fast_path: bool = True,
    equipment_saturate_adu: float | None = None,
    catalog_local_gaia_only: bool | None = None,
    aligned_ram: "Sequence[tuple[str, fits.Header, Any]] | None" = None,
    aligned_target_dir: Path | str | None = None,
    defer_disk_writes: bool = False,
    app_config: AppConfig | None = None,
    plate_solve_fov_deg: float | None = None,
    master_dark_path: Path | str | None = None,
    draft_id: int | None = None,
    equipment_id: int | None = None,
    full_catalog_export: bool = False,
) -> dict[str, Any]:
    """For each FITS under ``frames_root`` with WCS: DAO + catalog table, write one CSV per frame.

    **Optional fast path:** if ``use_master_fast_path=True`` and ``masterstars_csv`` + ``masterstar_fits`` match
    the frame (same WCS and **same array shape** as MASTERSTAR), exposures are matched only to
    ``masterstars.csv`` sky positions (faster; on chip edges NN distances can exceed the match threshold).
    ``astrometry_align_and_build_masterstar`` defaults this to **off** unless ``VYVAR_PER_FRAME_MASTER_FAST=1``.

    **Fallback:** if paths are missing or WCS differs, uses one shared cone (see ``field_catalog_cone.csv``)
    and full ``detect_stars_and_match_catalog`` per frame (local Gaia).
    A sidecar ``field_catalog_cone_meta.json`` records the angular cone radius used; if a **cropped** MASTERSTAR
    built a too-small cone, full-chip frames automatically trigger a refetch instead of reusing the stale CSV.

    ``<platesolve_dir>/per_frame_catalog_index.csv`` lists every file and CSV path.

    **Performance:** each frame still runs DAO + catalog match + disk CSV write (dominant cost for many lights).
    Parallelism: jednotny pocet z ``app_config`` / env (``VYVAR_PARALLEL_WORKERS`` alebo legacy env, pozri
    :func:`_vyvar_parallel_worker_count`). When ``>1``, uses ``ProcessPoolExecutor`` (``spawn``); the
    parent prefetches Gaia cone. Worker count is capped using ``psutil`` and
    ``per_frame_mp_reserve_ram_gb``. RAM handoff ``aligned_ram`` uses the same process pool with serialized
    headers + float32 pixels. Lower ``max_catalog_rows`` in the UI to reduce DAO work per file.

    **RAM handoff:** pass ``aligned_ram`` as ``(filename, header, ndarray)`` tuples plus ``aligned_target_dir`` to
    run catalog matching **without** re-reading aligned FITS from disk. With ``defer_disk_writes=True``, sidecar
    CSV (and optional flat mirror) are returned in ``deferred_csv_writes`` for the caller to flush after FITS.

    ``master_dark_path``: optional CalibrationLibrary master dark; enables ``*_dark_bpm.json`` column flags when present.
    """
    import numpy as np

    _cfg_ap = app_config or AppConfig()
    _md_bpm_str = ""
    if master_dark_path is not None and str(master_dark_path).strip():
        _mp = Path(str(master_dark_path))
        if _mp.is_file():
            _md_bpm_str = str(_mp.resolve())

    ps = Path(platesolve_dir) if platesolve_dir is not None else None
    _qc_fwhm_by_name: dict[str, float] = {}
    _fwhm_night_median_px: float | None = None
    try:
        from aperture_policy import load_qc_fwhm_map, normalize_aperture_policy_mode  # noqa: PLC0415

        _qc_root = None
        if ps is not None:
            try:
                _qc_root = Path(ps).resolve().parents[1]
            except IndexError:
                _qc_root = None
        _qc_csv = find_qc_metrics_csv(_qc_root) if _qc_root is not None else None
        _qc_fwhm_by_name, _fwhm_night_median_px = load_qc_fwhm_map(_qc_csv)
        _ap_policy_mode = normalize_aperture_policy_mode(
            getattr(_cfg_ap, "aperture_policy_mode", "f_fixed_night")
        )
    except Exception as _qc_exc:  # noqa: BLE001
        LOGGER.warning("[APERTURE-01] QC FWHM map not loaded: %s", _qc_exc)
        _ap_policy_mode = "f_fixed_night"

    _ap_st: dict[str, Any] = {
        "aperture_photometry_enabled": bool(_cfg_ap.aperture_photometry_enabled),
        "aperture_fwhm_factor": float(_cfg_ap.aperture_fwhm_factor),
        "annulus_inner_fwhm": float(_cfg_ap.annulus_inner_fwhm),
        "annulus_outer_fwhm": float(_cfg_ap.annulus_outer_fwhm),
        "nonlinearity_peak_percentile": float(_cfg_ap.nonlinearity_peak_percentile),
        "nonlinearity_fwhm_ratio": float(_cfg_ap.nonlinearity_fwhm_ratio),
        "bpm_dark_mad_sigma": float(_cfg_ap.bpm_dark_mad_sigma),
        "master_dark_path": _md_bpm_str,
        "database_path": str(Path(_cfg_ap.database_path).resolve()),
        "draft_id": int(draft_id) if draft_id is not None else None,
        "equipment_id": int(equipment_id) if equipment_id is not None else None,
        "aperture_correction_enabled": bool(_cfg_ap.aperture_correction_enabled),
        "aperture_fwhm_factor_small": float(_cfg_ap.aperture_snr_sizing.get("small", 1.5)),
        "aperture_fwhm_factor_large": float(_cfg_ap.aperture_snr_sizing.get("large", 4.0)),
        "aperture_policy_mode": str(_ap_policy_mode),
        "fwhm_night_median_px": _fwhm_night_median_px,
        "qc_fwhm_by_name": dict(_qc_fwhm_by_name),
        "platesolve_dir": str(ps.resolve()) if ps is not None else "",
        "cog_aperture_correction_enabled": bool(_cfg_ap.cog_aperture_correction_enabled),
        "cog_ref_fwhm": float(_cfg_ap.cog_ref_fwhm),
        "cog_min_stars": int(_cfg_ap.cog_min_stars),
        "cog_isolation_fwhm": float(_cfg_ap.cog_isolation_fwhm),
        "cog_snr_min": float(_cfg_ap.cog_snr_min),
        "cog_sat_frac": float(_cfg_ap.cog_sat_frac),
        "cog_ladder_step_px": float(_cfg_ap.cog_ladder_step_px),
        "cog_ladder_step_fwhm": getattr(_cfg_ap, "cog_ladder_step_fwhm", None),
        "cog_ac_factor_max": float(_cfg_ap.cog_ac_factor_max),
        "gain": float(_cfg_ap.gain),
        "read_noise": float(_cfg_ap.read_noise),
        "err_background_mode": "empirical",
        "err_empty_apertures_n": int(_cfg_ap.err_empty_apertures_n),
        "err_empty_apertures_min": int(_cfg_ap.err_empty_apertures_min),
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
    _ap_st.update(_export_catalog_psf_st_fields(_cfg_ap, ps))
    _epsf_science_meta: dict[str, Any] | None = None
    if bool(_cfg_ap.psf_photometry_enabled) and bool(_ap_st.get("_run_epsf")):
        from epsf_science_set import build_epsf_science_set

        _sci = build_epsf_science_set(ps)
        if not _sci.catalog_ids:
            raise ValueError(
                "ePSF science set is empty"
                + (f": {_sci.empty_reason}" if _sci.empty_reason else "")
                + "; refusing silent fallback to full LC pool."
            )
        _epsf_science_meta = _sci.to_meta_dict()
        _ap_st["epsf_science_set_meta"] = _epsf_science_meta
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

    _frame_index_by_name = {Path(f).name: i for i, f in enumerate(files)}
    _ap_st["epsf_frame_index_by_name"] = _frame_index_by_name

    if not files:
        return {
            "written": 0,
            "per_frame_dir": str(root),
            "per_frame_csv_mode": "sidecar" if write_sidecar_csv_next_to_fits else "none",
            "index_csv": "",
            "frames": [],
            "deferred_csv_writes": [],
        }

    master_only_mode = bool(use_master_fast_path)
    field_cat_path = ps / "field_catalog_cone.csv"
    meta_path = _field_catalog_cone_meta_path(field_cat_path)
    cat_df: pd.DataFrame | None = None

    _pfov_res: float | None = None
    try:
        _pf0 = float(plate_solve_fov_deg) if plate_solve_fov_deg is not None else float("nan")
        if math.isfinite(_pf0) and _pf0 > 0:
            _pfov_res = _pf0
    except (TypeError, ValueError):
        _pfov_res = None
    if _pfov_res is None and files:
        try:
            _rf0 = files[0]
            with fits.open(_rf0, memmap=False) as _h0:
                _hd0 = _h0[0].header.copy()
                _ar0 = np.asarray(_h0[0].data)
            if _ar0.ndim == 2:
                _pfov_res = resolve_plate_solve_fov_deg_hint(
                    _hd0,
                    int(_ar0.shape[0]),
                    int(_ar0.shape[1]),
                    database_path=_cfg_ap.database_path,
                    equipment_id=int(equipment_id) if equipment_id is not None else None,
                    draft_id=int(draft_id) if draft_id is not None else None,
                )
        except Exception:  # noqa: BLE001
            _pfov_res = None
    if _pfov_res is None:
        try:
            _pfov_res = float(_cfg_ap.plate_solve_fov_deg)
        except Exception:  # noqa: BLE001
            _pfov_res = None

    r_need_deg: float | None = None
    try:
        _ref_fp = files[0]
        with fits.open(_ref_fp, memmap=False) as _hdu0:
            _hdr0 = _hdu0[0].header
            _dat0 = np.asarray(_hdu0[0].data)
            if _dat0.ndim == 2:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FITSFixedWarning)
                    _w0 = WCS(_hdr0)
                if _w0.has_celestial:
                    _h0, _w0px = _dat0.shape
                    _, r_need_deg = _effective_field_catalog_cone_radius_deg(
                        _w0, _h0, _w0px, _pfov_res, fits_header=_hdr0
                    )
    except Exception:  # noqa: BLE001
        r_need_deg = None

    if (not master_only_mode) and r_need_deg is not None:
        _invalidate_field_catalog_cone_cache_if_needed(
            field_cat_path,
            plate_solve_fov_deg=_pfov_res,
            effective_radius_deg=float(r_need_deg),
        )

    if (not master_only_mode) and field_cat_path.is_file():
        try:
            _hdr_fc = pd.read_csv(field_cat_path, nrows=0)
            _dtype_fc: dict[str, type] = {}
            if "catalog_id" in _hdr_fc.columns:
                _dtype_fc["catalog_id"] = str
            _kw_fc: dict[str, Any] = {}
            if _dtype_fc:
                _kw_fc["dtype"] = _dtype_fc
            _cdf = pd.read_csv(field_cat_path, **_kw_fc)
            if len(_cdf) > 0:
                _reuse = True
                if r_need_deg is not None and meta_path.is_file():
                    try:
                        _meta = json.loads(meta_path.read_text(encoding="utf-8"))
                        r_stored = float(_meta.get("cone_radius_deg", 0.0))
                        slack_deg = 45.0 / 3600.0
                        if r_stored <= 0.0 or r_need_deg > r_stored * 1.02 + slack_deg:
                            _reuse = False
                            LOGGER.info(
                                "Per-frame catalog: ignoring cached %s (full chip needs cone_radius_deg~%.6f, "
                                "cached %.6f from %s) - fetching larger Gaia cone",
                                field_cat_path,
                                r_need_deg,
                                r_stored,
                                meta_path.name,
                            )
                    except Exception:  # noqa: BLE001
                        pass
                if _reuse:
                    cat_df = _cdf
                    LOGGER.info(
                        "Per-frame catalog: reusing %s (%s rows) - skipping duplicate cone query",
                        field_cat_path,
                        len(cat_df),
                    )
        except Exception:  # noqa: BLE001
            cat_df = None

    kd_cell: list[tuple[Any, Any]] = []

    def _refresh_shared_cat_kdtree() -> None:
        if kd_cell or cat_df is None or getattr(cat_df, "empty", True) or len(cat_df) < 120:
            return
        bk = build_ucac_catalog_kdtree(cat_df)
        if bk is not None:
            kd_cell.append(bk)
            LOGGER.info(
                "Per-frame catalog: shared cKDTree for %s cone rows (fast sky matching)",
                len(cat_df),
            )

    vsx_df: pd.DataFrame | None = None
    gaia_variable_df: pd.DataFrame | None = None
    total = len(files)
    _debug_logged_once = False
    _uc_um = float(faintest_mag_limit) if faintest_mag_limit is not None else None
    try:
        _cfg_e = AppConfig()
        _gp = (_cfg_e.gaia_db_path or "").strip()
        _uc_root_e = Path(_gp) if _gp else None
    except Exception:  # noqa: BLE001
        _uc_root_e = None
    _export_cat_local = (not master_only_mode)

    master_tab: pd.DataFrame | None = None
    ref_wcs: WCS | None = None
    masterstar_data_shape: tuple[int, int] | None = None
    if use_master_fast_path and masterstars_csv is not None:
        _msp = Path(masterstars_csv)
        if _msp.is_file():
            try:
                _hdr_ms = pd.read_csv(_msp, nrows=0)
                _dtype_ms: dict[str, type] = {}
                if "catalog_id" in _hdr_ms.columns:
                    _dtype_ms["catalog_id"] = str
                if "name" in _hdr_ms.columns:
                    _dtype_ms["name"] = str
                _kw_ms: dict[str, Any] = {}
                if _dtype_ms:
                    _kw_ms["dtype"] = _dtype_ms
                _mt = pd.read_csv(_msp, **_kw_ms)
                if not _mt.empty and "ra_deg" in _mt.columns and "dec_deg" in _mt.columns:
                    master_tab = _mt
            except Exception:  # noqa: BLE001
                master_tab = None
    if use_master_fast_path and masterstar_fits is not None and master_tab is not None:
        _msf = Path(masterstar_fits)
        if _msf.is_file():
            try:
                with fits.open(_msf, memmap=False) as _mh:
                    _marr = np.asarray(_mh[0].data)
                    if _marr.ndim == 2:
                        masterstar_data_shape = (int(_marr.shape[0]), int(_marr.shape[1]))
                    ref_wcs = WCS(_mh[0].header)
                if not ref_wcs.has_celestial:
                    ref_wcs = None
            except Exception:  # noqa: BLE001
                ref_wcs = None
                masterstar_data_shape = None
        else:
            ref_wcs = None
    else:
        ref_wcs = None

    if master_tab is not None:
        LOGGER.info(
            "Per-frame catalog: MASTERSTAR lock enabled (single catalog: masterstars_full_match.csv, %s rows)",
            len(master_tab),
        )
    elif master_only_mode:
        raise RuntimeError(
            "Per-frame catalog lock requested, but masterstars_full_match.csv is missing or invalid."
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
                    # PRIORITA 1: VY_FWHM_GAUSS - 2D Gaussian fit, closest to SExtractor
                    for _gk in ("VY_FWHM_GAUSS", "VY_FWHM_GAUSSIAN"):
                        _vg = _ghdr.get(_gk)
                        if _vg is None:
                            continue
                        try:
                            _vgf = float(_vg)
                            if 1.0 <= _vgf <= 15.0:
                                _gauss_override = _vgf
                                LOGGER.debug(
                                    "[FWHM] gaussian_override from %s: %.3f px",
                                    _gk,
                                    _gauss_override,
                                )
                                break
                        except (TypeError, ValueError):
                            pass

                    # PRIORITA 2: VY_FWHM x 0.667 fallback
                    if _gauss_override is None:
                        _vy = _ghdr.get("VY_FWHM")
                        if _vy is not None:
                            try:
                                _vyf = float(_vy)
                                if 1.0 <= _vyf <= 15.0:
                                    _gauss_override = _vyf * (1.0 / 1.5)
                                    LOGGER.debug(
                                        "[FWHM] gaussian_override from VY_FWHMx0.667: %.3f px",
                                        _gauss_override,
                                    )
                            except (TypeError, ValueError):
                                pass
    except Exception:  # noqa: BLE001
        _gauss_override = None
    _ap_st["gaussian_fwhm_px_override"] = _gauss_override
    if _gauss_override is not None:
        log_event(
            f"[PHOT] gaussian_fwhm_px_override = {float(_gauss_override):.4f}px "
            "(z VY_FWHM alebo VY_FWHM_GAUSS)"
        )
    else:
        log_event("[PHOT] gaussian_fwhm_px_override = None -> fallback na momentx0.619 per frame")

    cfg_for_workers = app_config if app_config is not None else AppConfig()
    _dao_fw_export = (
        float(dao_fwhm_px)
        if dao_fwhm_px is not None
        else float(cfg_for_workers.sips_dao_fwhm_px)
    )
    n_workers = _vyvar_per_frame_csv_workers(cfg_for_workers)
    _ny, _nx = _estimate_catalog_frame_hw(work_ram if use_ram_inputs else None, files)
    n_workers = _vyvar_cap_mp_workers_for_catalog(
        n_workers,
        (_ny, _nx),
        reserve_gb=float(cfg_for_workers.per_frame_mp_reserve_ram_gb),
    )
    if n_workers > 1 and total > 1:
        LOGGER.info(
            "Per-frame catalog: up to %s process worker(s); jednotny parallel count + RAM cap (psutil); "
            "env VYVAR_PARALLEL_WORKERS / legacy",
            n_workers,
        )

    use_parallel_mp = n_workers > 1 and total > 1
    kd_pack_mp: Any = None
    if use_parallel_mp and not master_only_mode:
        if use_ram_inputs and work_ram is not None:
            _h_ref, _d_ref = work_ram[0][1], np.asarray(work_ram[0][2], dtype=np.float32)
            cat_df, vsx_df, gaia_variable_df, kd_pack_mp = _prefetch_export_shared_catalog_for_process_pool(
                files=None,
                reference_hdr_data=(_h_ref.copy(), _d_ref),
                field_cat_path=field_cat_path,
                cat_df=cat_df,
                vsx_df=vsx_df,
                gaia_variable_df=gaia_variable_df,
                gaia_db_path=_uc_root_e,
                gaia_local_max_mag=_uc_um,
                export_cat_local=_export_cat_local,
                plate_solve_fov_deg=_pfov_res,
            )
        else:
            cat_df, vsx_df, gaia_variable_df, kd_pack_mp = _prefetch_export_shared_catalog_for_process_pool(
                files=list(files),
                reference_hdr_data=None,
                field_cat_path=field_cat_path,
                cat_df=cat_df,
                vsx_df=vsx_df,
                gaia_variable_df=gaia_variable_df,
                gaia_db_path=_uc_root_e,
                gaia_local_max_mag=_uc_um,
                export_cat_local=_export_cat_local,
                plate_solve_fov_deg=_pfov_res,
            )
        LOGGER.info(
        "Per-frame catalog: parallel backend=process (%s workers); Gaia cone prefetched in parent",
            n_workers,
        )
    else:
        _refresh_shared_cat_kdtree()

    _prog_seq = 0
    deferred_csv_writes: list[tuple[Path, pd.DataFrame]] = []

    def _append_deferred_csv(p: Path, df: pd.DataFrame) -> None:
        deferred_csv_writes.append((p, df))

    def _ensure_cone_and_variables(
        c_i: SkyCoord, r_i: float, *, naxis1: int = 0, naxis2: int = 0
    ) -> None:
        def _fill() -> None:
            nonlocal cat_df, vsx_df, gaia_variable_df
            if master_only_mode:
                return
            if cat_df is None or cat_df.empty:
                _gaia_db_path: Path | None = None
                try:
                    _gp = (_cfg_ap.gaia_db_path or "").strip()
                    if _gp:
                        _gaia_db_path = Path(_gp)
                except Exception:  # noqa: BLE001
                    _gaia_db_path = None
                cat_df = _query_gaia_local(
                    center=c_i,
                    radius_deg=r_i,
                    gaia_db_path=_gaia_db_path,
                    max_mag=float(_uc_um) if _uc_um is not None else None,
                )
                if cat_df is not None and len(cat_df) > 0:
                    try:
                        field_cat_path.parent.mkdir(parents=True, exist_ok=True)
                        _vyvar_df_to_csv(cat_df, field_cat_path)
                        _write_field_catalog_cone_meta(
                            field_cat_path,
                            center=c_i,
                            radius_deg=float(r_i),
                            naxis1=int(naxis1),
                            naxis2=int(naxis2),
                            plate_solve_fov_deg=_pfov_res,
                        )
                    except Exception as exc:  # noqa: BLE001
                        LOGGER.debug("[PIPE] cone/variables CSV write failed: %s", exc)
                        pass
            if vsx_df is None or getattr(vsx_df, "empty", True):
                _vsx_p2: Path | None = None
                try:
                    _vsp2 = str(_cfg_ap.vsx_local_db_path or "").strip()
                    if _vsp2:
                        _vsx_p2 = Path(_vsp2).expanduser().resolve()
                except Exception:  # noqa: BLE001
                    _vsx_p2 = None
                if _vsx_p2 is not None and _vsx_p2.is_file():
                    vsx_df = _query_vsx_local(
                        center=c_i,
                        radius_deg=float(r_i),
                        vsx_db_path=_vsx_p2,
                    )
                else:
                    vsx_df = pd.DataFrame()
            if gaia_variable_df is None:
                gaia_variable_df = pd.DataFrame()
            _refresh_shared_cat_kdtree()

        _fill()

    def _run_one_catalog(base_path: Path, hdr: fits.Header, data: np.ndarray) -> dict[str, Any]:
        fname = base_path.name
        nonlocal _prog_seq
        if progress_cb is not None and n_workers <= 1:
            _prog_seq += 1
            progress_cb(_prog_seq, total, f"Catalog: {fname}")
        if not _has_valid_wcs(hdr):
            return {"file": fname, "status": "no_wcs", "csv": ""}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            w_i = WCS(hdr)
        h_i, wpx_i = data.shape

        use_fast = (
            bool(use_master_fast_path)
            and master_tab is not None
            and (not getattr(master_tab, "empty", True))
        )
        if master_only_mode and not use_fast:
            return {"file": fname, "status": "error: master_only_mode requires masterstars_full_match.csv", "csv": ""}

        df: pd.DataFrame
        meta: dict[str, Any]
        if use_fast:
            try:
                _raw_arr = None
                _raw_hdr = None
                if _sat_diag_ctx is not None and _sat_diag_archive:
                    _raw_p = _resolve_draft_light_raw_path(Path(_sat_diag_archive), fname)
                    if _raw_p is not None and _raw_p.is_file():
                        try:
                            from sat_diag import image_adu_array  # noqa: PLC0415

                            with fits.open(_raw_p, memmap=False) as _rhd:
                                if int(_rhd[0].header.get("BITPIX", 0)) >= 0:
                                    _raw_arr = image_adu_array(_rhd[0])
                                    _raw_hdr = _rhd[0].header.copy()
                        except Exception:  # noqa: BLE001
                            _raw_arr = None
                df, meta = detect_stars_match_master_reference(
                    data,
                    hdr,
                    master_tab,
                    max_catalog_rows=int(max_catalog_rows),
                    match_sep_arcsec=float(catalog_match_max_sep_arcsec),
                    saturate_level_fraction=float(saturate_level_fraction),
                    faintest_mag_limit=faintest_mag_limit,
                    dao_threshold_sigma=float(dao_threshold_sigma),
                    dao_fwhm_px=float(_dao_fw_export),
                    equipment_saturate_adu=equipment_saturate_adu,
                    frame_name=fname,
                    sat_diag_ctx=_sat_diag_ctx,
                    raw_data=_raw_arr,
                    raw_hdr=_raw_hdr,
                    ref_ra_deg=_ref_ra_deg,
                    ref_dec_deg=_ref_dec_deg,
                    drift_ref_catalog_id=_drift_ref_catalog_id,
                )
            except Exception as exc:  # noqa: BLE001
                return {"file": fname, "status": f"error: {exc}", "csv": ""}
        else:
            if master_only_mode:
                return {"file": fname, "status": "error: master_only_mode fallback to Gaia cone is disabled", "csv": ""}
            c_i, r_i = _effective_field_catalog_cone_radius_deg(
                w_i, h_i, wpx_i, _pfov_res, fits_header=hdr
            )
            _ensure_cone_and_variables(c_i, r_i, naxis1=int(wpx_i), naxis2=int(h_i))

            try:
                df, meta = detect_stars_and_match_catalog(
                    data,
                    hdr,
                    max_catalog_rows=int(max_catalog_rows),
                    cat_df=cat_df,
                    vsx_df=vsx_df,
                    gaia_variable_df=gaia_variable_df,
                    match_sep_arcsec=float(catalog_match_max_sep_arcsec),
                    saturate_level_fraction=float(saturate_level_fraction),
                    faintest_mag_limit=faintest_mag_limit,
                    field_catalog_export_path=None,
                    dao_threshold_sigma=float(dao_threshold_sigma),
                    dao_fwhm_px=float(_dao_fw_export),
                    equipment_saturate_adu=equipment_saturate_adu,
                    catalog_local_gaia_only=_export_cat_local,
                    catalog_kd_pack=kd_cell[0] if kd_cell else None,
                    plate_solve_fov_deg=_pfov_res,
                    fov_database_path=_cfg_ap.database_path,
                    fov_equipment_id=int(equipment_id) if equipment_id is not None else None,
                    fov_draft_id=int(draft_id) if draft_id is not None else None,
                )
            except Exception as exc:  # noqa: BLE001
                return {"file": fname, "status": f"error: {exc}", "csv": ""}

        _exo_st = {
            "exoplanet_local_db_path": str(_cfg_ap.exoplanet_local_db_path or ""),
            "exoplanet_match_max_sep_arcsec": float(_cfg_ap.exoplanet_match_max_sep_arcsec),
            "plate_solve_fov_deg": _pfov_res,
            "database_path": str(Path(_cfg_ap.database_path).resolve()),
            "equipment_id": equipment_id,
            "draft_id": draft_id,
        }
        df = _apply_exo_host_columns_to_proc_df(df, hdr, (h_i, wpx_i), _exo_st, frame_name=fname)

        _before_dao = len(df)
        df = _proc_drop_unmatched_dao_rows(df)
        LOGGER.debug("[TODO-13] catalog-only pre-filter (detect): %d -> %d rows", _before_dao, len(df))

        _run_aperture = bool(_ap_st.get("_run_aperture", True))
        _run_epsf = bool(_ap_st.get("_run_epsf", False))
        if _run_aperture:
            _ap_st["current_frame_name"] = fname
            df = _apply_aperture_catalog_enhancements_from_st(df, data, hdr, _ap_st)
        _psf_on = bool(_ap_st.get("psf_photometry_enabled", False))
        _epsf_ids = (
            _epsf_fit_catalog_ids(ps, psf_photometry_enabled=_psf_on)
            if ps is not None
            else None
        )
        LOGGER.info(
            "[ePSF] _run_one_catalog %s: platesolve_dir=%r n_psf_ids=%s lc_set=%s",
            fname,
            str(ps.resolve()) if ps is not None else None,
            len(_epsf_ids) if _epsf_ids is not None else "ALL",
            _psf_on,
        )
        if not _run_aperture and _run_epsf:
            # PSF-only mode: psf_flux promoted to primary.
            pass
        df = _fill_psf_catalog_columns(df, data, hdr, _ap_st, target_ids=_epsf_ids)

        # --- Time columns (JD / HJD / BJD) ---
        _db_tc = None
        try:
            from time_utils import compute_time_columns

            _dbp = str(_ap_st.get("database_path") or "").strip()
            _did_tc = _ap_st.get("draft_id")
            if _dbp and _did_tc is not None:
                try:
                    _db_tc = VyvarDatabase(Path(_dbp))
                except Exception:  # noqa: BLE001
                    _db_tc = None
            _time_cols = compute_time_columns(
                hdr,
                db=_db_tc,
                draft_id=int(_did_tc) if _did_tc is not None else None,
                cfg=_cfg_ap,
            )
            _tk = ("jd_mid", "hjd_mid", "bjd_tdb_mid")
            _cols_base = list(df.columns)
            _anchors = [c for c in ("jd", "inspection_jd") if c in _cols_base]
            if _anchors:
                _pos = max(_cols_base.index(c) for c in _anchors) + 1
            else:
                _flux_first = next((c for c in _cols_base if c in ("dao_flux", "flux")), None)
                _pos = _cols_base.index(_flux_first) if _flux_first is not None else len(_cols_base)
            for _i, _nm in enumerate(_tk):
                df.insert(_pos + _i, _nm, _time_cols[_nm])

            # Airmass - frame-level hodnota z FITS hlavicky
            _am_val = _extract_airmass_from_header(
                hdr,
                cfg=_cfg_ap,
                db=_db_tc,
                draft_id=int(_did_tc) if _did_tc is not None else None,
            )
            if "airmass" not in df.columns:
                _am_insert_pos = _pos + len(_tk)
                df.insert(_am_insert_pos, "airmass", _am_val)
        except Exception as _tc_exc:  # noqa: BLE001
            log_event(f"Time columns skipped: {_tc_exc}")
            for _tc in ("jd_mid", "hjd_mid", "bjd_tdb_mid"):
                if _tc not in df.columns:
                    df[_tc] = None
            if "airmass" not in df.columns:
                df["airmass"] = float("nan")
        finally:
            if _db_tc is not None:
                try:
                    _db_tc.conn.close()
                except Exception as exc:  # noqa: BLE001
                    LOGGER.debug("[PIPELINE] Cleanup step failed (non-critical): %s", exc)

        df2 = df.copy()
        df2.insert(0, "source_file", fname)

        _before_dedupe = len(df2)
        df2 = _proc_deduplicate_matched_catalog_rows(df2)
        if len(df2) != _before_dedupe:
            LOGGER.debug(
                "[PROC] per-frame catalog dedupe: %d -> %d rows (%s)",
                _before_dedupe,
                len(df2),
                fname,
            )
        _before_cat = len(df2)
        df2 = _proc_catalog_keep_matched_rows_only(df2)
        LOGGER.debug("[TODO-13] catalog-only filter: %d -> %d rows", _before_cat, len(df2))

        csv_paths: list[str] = []
        if write_sidecar_csv_next_to_fits:
            sidecar = proc_csv_path_for_aligned_fits(base_path)
            if defer_disk_writes:
                _append_deferred_csv(sidecar, df2.copy())
            else:
                _vyvar_df_to_csv(df2, sidecar)
            csv_paths.append(str(sidecar))

        if mirror_flat_platesolve_folder:
            stem = Path(fname).stem
            safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in stem)[:120]
            flat_path = out_flat / f"{safe}_catalog.csv"
            if defer_disk_writes:
                _append_deferred_csv(flat_path, df2.copy())
            else:
                _vyvar_df_to_csv(df2, flat_path)
            csv_paths.append(str(flat_path))

        primary_csv = csv_paths[0] if csv_paths else ""
        return {
            "file": fname,
            "status": "ok",
            "csv": primary_csv,
            "csv_paths": ";".join(csv_paths),
            "n_detected": meta.get("n_detected"),
            "n_matched": meta.get("n_matched"),
            "catalog_match_mode": ("master_reference_locked" if master_only_mode else meta.get("catalog_match_mode", "full_cone")),
            "raw_peaks_used": bool(meta.get("raw_peaks_used")),
        }

    def _process_frame(fp: Path) -> dict[str, Any]:
        try:
            try:
                with fits.open(fp, memmap=False) as hdul:
                    hdr = hdul[0].header.copy()
                    data = np.array(hdul[0].data, dtype=np.float32, copy=True)
            except Exception as exc:  # noqa: BLE001
                return {"file": fp.name, "status": f"read_error: {exc}", "csv": ""}
            return _run_one_catalog(fp, hdr, data)
        except Exception as exc:  # noqa: BLE001
            return {"file": fp.name, "status": f"error: {exc}", "csv": ""}

    def _process_ram_item(item: tuple[str, fits.Header, Any]) -> dict[str, Any]:
        name, hdr0, arr0 = item
        try:
            base = Path(aligned_target_dir) / name
            return _run_one_catalog(base, hdr0.copy(), np.asarray(arr0, dtype=np.float32))
        except Exception as exc:  # noqa: BLE001
            return {"file": name, "status": f"error: {exc}", "csv": ""}

    def _catalog_worker_state() -> dict[str, Any]:
        return {
            "cat_df": cat_df,
            "vsx_df": vsx_df,
            "gaia_variable_df": gaia_variable_df,
            "kd_pack": kd_pack_mp,
            "master_tab": master_tab,
            "masterstar_fits_path": (
                str(Path(masterstar_fits).resolve())
                if masterstar_fits is not None and Path(masterstar_fits).is_file()
                else ""
            ),
            "use_master_fast_path": bool(use_master_fast_path),
            "masterstar_data_shape": masterstar_data_shape,
            "max_catalog_rows": int(max_catalog_rows),
            "catalog_match_max_sep_arcsec": float(catalog_match_max_sep_arcsec),
            "saturate_level_fraction": float(saturate_level_fraction),
            "faintest_mag_limit": faintest_mag_limit,
            "dao_threshold_sigma": float(dao_threshold_sigma),
            "dao_fwhm_px": float(_dao_fw_export),
            "forced_photometry_enabled": bool(
                getattr(cfg_for_workers, "forced_photometry_enabled", True)
            ),
            "forced_photometry_centroid_bound_fwhm": float(
                getattr(cfg_for_workers, "forced_photometry_centroid_bound_fwhm", 2.5) or 2.5
            ),
            "forced_photometry_margin_px": float(
                getattr(cfg_for_workers, "forced_photometry_margin_px", 0.0) or 0.0
            ),
            "equipment_saturate_adu": equipment_saturate_adu,
            "sat_diag_ctx_dict": (
                _sat_diag_ctx.to_json_dict() if _sat_diag_ctx is not None else None
            ),
            "sat_diag_archive": _sat_diag_archive,
            "sat_diag_ref_ra": _ref_ra_deg,
            "sat_diag_ref_dec": _ref_dec_deg,
            "sat_diag_ref_catalog_id": _drift_ref_catalog_id,
            "export_cat_local": _export_cat_local,
            "master_only_mode": bool(master_only_mode),
            "plate_solve_fov_deg": _pfov_res,
            "write_sidecar_csv_next_to_fits": write_sidecar_csv_next_to_fits,
            "mirror_flat_platesolve_folder": mirror_flat_platesolve_folder,
            "defer_disk_writes": defer_disk_writes,
            "out_flat": str(out_flat.resolve()),
            "aperture_photometry_enabled": bool(_cfg_ap.aperture_photometry_enabled),
            "aperture_fwhm_factor": float(_cfg_ap.aperture_fwhm_factor),
            "annulus_inner_fwhm": float(_cfg_ap.annulus_inner_fwhm),
            "annulus_outer_fwhm": float(_cfg_ap.annulus_outer_fwhm),
            "nonlinearity_peak_percentile": float(_cfg_ap.nonlinearity_peak_percentile),
            "nonlinearity_fwhm_ratio": float(_cfg_ap.nonlinearity_fwhm_ratio),
            "bpm_dark_mad_sigma": float(_cfg_ap.bpm_dark_mad_sigma),
            "master_dark_path": _md_bpm_str,
            "database_path": str(Path(_cfg_ap.database_path).resolve()),
            "draft_id": int(draft_id) if draft_id is not None else None,
            "equipment_id": int(equipment_id) if equipment_id is not None else None,
            "gaussian_fwhm_px_override": _gauss_override,
            "aperture_correction_enabled": bool(_cfg_ap.aperture_correction_enabled),
            "aperture_fwhm_factor_small": float(_cfg_ap.aperture_snr_sizing.get("small", 1.5)),
            "aperture_fwhm_factor_large": float(_cfg_ap.aperture_snr_sizing.get("large", 4.0)),
            "aperture_policy_mode": str(_ap_policy_mode),
            "fwhm_night_median_px": _fwhm_night_median_px,
            "qc_fwhm_by_name": dict(_qc_fwhm_by_name),
            "platesolve_dir": str(ps.resolve()),
            "observer_lat": float(_cfg_ap.observer_lat),
            "observer_lon": float(_cfg_ap.observer_lon),
            "observer_alt_m": float(_cfg_ap.observer_alt_m),
            "exoplanet_local_db_path": str(_cfg_ap.exoplanet_local_db_path or ""),
            "exoplanet_match_max_sep_arcsec": float(_cfg_ap.exoplanet_match_max_sep_arcsec),
            **_export_catalog_psf_st_fields(_cfg_ap, ps),
            "epsf_frame_index_by_name": dict(_frame_index_by_name),
            "epsf_science_set_meta": _epsf_science_meta,
        }

    if use_parallel_mp and use_ram_inputs and work_ram is not None:
        ctx = multiprocessing.get_context("spawn")
        ws = _catalog_worker_state()
        packs: list[tuple[str, bytes, bytes, int, int]] = []
        for _name, _hdr, _arr in work_ram:
            base_p = (root / _name).resolve()
            d = np.asarray(_arr, dtype=np.float32)
            if not d.flags.c_contiguous:
                d = np.ascontiguousarray(d, dtype=np.float32)
            packs.append(
                (
                    str(base_p),
                    pickle.dumps(_hdr.copy()),
                    d.tobytes(),
                    int(d.shape[0]),
                    int(d.shape[1]),
                )
            )
        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=_init_export_per_frame_worker,
            initargs=(ws,),
        ) as ex:
            futs = {ex.submit(_export_per_frame_ram_worker_task, pack): pack[0] for pack in packs}
            by_name: dict[str, dict[str, Any]] = {}
            done = 0
            for fut in as_completed(futs):
                base_str = futs[fut]
                res = fut.result()
                nm = Path(base_str).name
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
                by_name[nm] = res
                done += 1
                if progress_cb is not None:
                    progress_cb(done, total, f"Catalog: {nm}")
        rows_out = []
        for it in work_ram:
            rr = dict(by_name[str(it[0])])
            rr.pop("deferred_writes", None)
            rr.pop("infolog_messages", None)
            rows_out.append(rr)
    elif use_parallel_mp:
        ctx = multiprocessing.get_context("spawn")
        ws = _catalog_worker_state()
        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=_init_export_per_frame_worker,
            initargs=(ws,),
        ) as ex:
            futs = {ex.submit(_export_per_frame_disk_worker_task, str(fp.resolve())): fp for fp in files}
            by_fp: dict[Path, dict[str, Any]] = {}
            done = 0
            for fut in as_completed(futs):
                fp = futs[fut]
                res = fut.result()
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
                by_fp[fp] = res
                done += 1
                if progress_cb is not None:
                    progress_cb(done, total, f"Catalog: {fp.name}")
                for p_str, dff in res.get("deferred_writes") or []:
                    deferred_csv_writes.append((Path(p_str), dff))
        rows_out = []
        for fp in files:
            rr = dict(by_fp[fp])
            rr.pop("deferred_writes", None)
            rr.pop("infolog_messages", None)
            rows_out.append(rr)
    elif use_ram_inputs and work_ram is not None:
        rows_out = []
        for i, it in enumerate(work_ram, start=1):
            if progress_cb is not None:
                progress_cb(i, total, f"Catalog: {it[0]}")
            r = _process_ram_item(it)
            if (not _debug_logged_once) and isinstance(r, dict) and r.get("debug_pixel_match") is not None:
                _debug_logged_once = True
                try:
                    log_event(
                        "DEBUG: per-frame debug_pixel_match (first frame): "
                        + json.dumps(r.get("debug_pixel_match"), ensure_ascii=False, default=str)
                    )
                except Exception:  # noqa: BLE001
                    log_event(f"DEBUG: per-frame debug_pixel_match (first frame): {r.get('debug_pixel_match')}")
            for msg in r.get("infolog_messages") or []:
                if msg:
                    log_event(str(msg))
            r.pop("infolog_messages", None)
            rows_out.append(r)
    else:
        rows_out = []
        for i, fp in enumerate(files, start=1):
            if progress_cb is not None:
                progress_cb(i, total, f"Catalog: {fp.name}")
            r = _process_frame(fp)
            if (not _debug_logged_once) and isinstance(r, dict) and r.get("debug_pixel_match") is not None:
                _debug_logged_once = True
                try:
                    log_event(
                        "DEBUG: per-frame debug_pixel_match (first frame): "
                        + json.dumps(r.get("debug_pixel_match"), ensure_ascii=False, default=str)
                    )
                except Exception:  # noqa: BLE001
                    log_event(f"DEBUG: per-frame debug_pixel_match (first frame): {r.get('debug_pixel_match')}")
            for msg in r.get("infolog_messages") or []:
                if msg:
                    log_event(str(msg))
            r.pop("infolog_messages", None)
            rows_out.append(r)

    index_path = ps / "per_frame_catalog_index.csv"
    if not defer_disk_writes:
        index_rows = [
            {k: v for k, v in row.items() if k not in ("deferred_writes", "infolog_messages")}
            for row in rows_out
        ]
        _vyvar_df_to_csv(pd.DataFrame(index_rows), index_path)
    n_ok = sum(1 for r in rows_out if r.get("status") == "ok")
    n_master_ref = sum(1 for r in rows_out if r.get("catalog_match_mode") == "master_reference")
    _hybrid_stats: dict[str, Any] = {}
    if not defer_disk_writes:
        _hybrid_stats = _finalize_hybrid_bkg_fallback_sidecar(
            root,
            err_background_mode="empirical",
            write_sidecar=bool(write_sidecar_csv_next_to_fits),
            gain=float(_ap_st.get("gain", _cfg_ap.gain)),
            read_noise=float(_ap_st.get("read_noise", _cfg_ap.read_noise)),
            setup_label=str(root.name),
        )
    if _sat_diag_ctx is not None and _sat_diag_archive:
        try:
            from sat_diag import commit_sat_diag_provenance  # noqa: PLC0415

            _placed_raw = any(bool(r.get("raw_peaks_used")) for r in rows_out)
            commit_sat_diag_provenance(
                _sat_diag_ctx,
                _sat_diag_archive,
                placed_aperture_used=_placed_raw,
            )
        except Exception as _sd_write_exc:  # noqa: BLE001
            LOGGER.warning("[SAT-DIAG] provenance commit skipped: %s", _sd_write_exc)
    _epsf_job_summary: dict[str, Any] | None = None
    if bool(_cfg_ap.psf_photometry_enabled) and bool(_ap_st.get("_run_epsf")):
        from epsf_frame_accounting import finalize_epsf_frame_job

        _psf_recs = [
            r["psf_frame_record"]
            for r in rows_out
            if isinstance(r.get("psf_frame_record"), dict)
        ]
        if _psf_recs:
            _epsf_job_summary = finalize_epsf_frame_job(
                _psf_recs,
                platesolve_dir=ps,
                science_set_meta=_epsf_science_meta,
            )
    return {
        "written": int(n_ok),
        "per_frame_dir": str(root),
        "per_frame_csv_mode": "sidecar" if write_sidecar_csv_next_to_fits else ("flat_mirror" if mirror_flat_platesolve_folder else "none"),
        "index_csv": str(index_path),
        "frames": rows_out,
        "mirror_flat_platesolve_folder": bool(mirror_flat_platesolve_folder),
        "frames_master_reference_match": int(n_master_ref),
        "deferred_csv_writes": list(deferred_csv_writes) if defer_disk_writes else [],
        "hybrid_bkg_fallback": _hybrid_stats,
        "epsf_job_summary": _epsf_job_summary,
    }
