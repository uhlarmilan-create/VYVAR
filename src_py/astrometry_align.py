"""Moved from pipeline.py (CONSOLIDATE-01E6b). Facade re-exports this name.

_astrometry_align_impl_body: frame-alignment MP loop, plate-solve, WCS.
Alignment MP init/task resolved from vyvar_alignment_frame by
fresh module attribute (A-durable; location-independent).
test_astrometry_fault_isolation patches on pipeline facade still bite
because pipeline_astrometry callers use call-time facade imports.
export_per_frame_catalogs / generate_masterstar_and_catalog imported
at call time from the pipeline facade.
"""
from __future__ import annotations

import contextlib
import json
import logging
import math
import multiprocessing
import os
import shutil
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

import time
import pickle
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs import FITSFixedWarning
import pandas as pd

import vyvar_alignment_frame  # A-durable: fresh-attr MP func lookup at dispatch
from vyvar_alignment_frame import (
    _alignment_compute_one_frame,
    _alignment_detect_xy,
    _as_fits_float32_image,
    _astrometry_align_mp_init,
    _astrometry_align_mp_task,
)
from config import AppConfig
from database import VyvarDatabase
from infolog import log_event
from optics_selection import resolve_optics_ids_for_platesolve
from utils import (
    fits_binning_xy_from_header,
    per_frame_catalog_match_sep_arcsec_for_scale,
    wcs_rotation_angle_deg,
)
from utils import dao_detection_fwhm_pixels
from vyvar_platesolver import pointing_hint_from_header as _pointing_hint_from_header
from pipeline_calibrate import (
    _has_valid_wcs,
    _pipeline_ui_error,
    _vyvar_parallel_worker_count,
    estimate_archive_memory_profile,
)
from pipeline_astrometry import (
    _apply_wcs_header_to_fits,
    _assert_alignment_produced_fits,
    _ensure_parent_dirs_for_aligned_fits,
    _equipment_saturate_adu_from_db,
    _photometry_mode_run_flags,
    _pipeline_ui_info,
    _solve_wcs_external,
    _vyvar_open_database,
    _wcs_field_center_radec_deg,
    compute_plate_scale_from_db,
    resolve_plate_solve_fov_deg_hint,
    write_photometry_plan_files,
)
from pipeline_catalog import (
    _finalize_hybrid_bkg_fallback_sidecar,
    _pick_reference_frame_by_star_count,
)

LOGGER = logging.getLogger("pipeline")


def _astrometry_align_impl_body(

    *,
    job: dict[str, Any],
    archive_path: Path,
    astrometry_api_key: str | None = None,
    max_control_points: int = 80,
    min_detected_stars: int = 100,
    max_detected_stars: int = 500,
    platesolve_backend: str = "vyvar",
    plate_solve_fov_deg: float = 1.0,
    max_extra_platesolve: int = 0,
    catalog_match_max_sep_arcsec: float = 25.0,
    saturate_level_fraction: float = 0.999,
    max_catalog_rows: int = 12000,
    n_comparison_stars: int = 0,
    require_non_variable_comparisons: bool = True,
    faintest_mag_limit: float | None = None,
    dao_threshold_sigma: float = 3.5,
    id_equipment: int | None = None,
    draft_id: int | None = None,
    catalog_local_gaia_only: bool | None = None,
    build_masterstar_and_catalogs: bool = False,
    progress_cb: "callable | None" = None,
    ram_align_and_catalog: bool = False,
    app_config: AppConfig | None = None,
    sibling_recovery_use_masterstar: bool = False,
    osc_registration_handoff: dict[str, Any] | None = None,
    osc_write_registration_handoff: bool = False,
) -> dict[str, Any]:
    """Internal: astrometry + alignment + per-frame CSV for one observation subtree (``job``)."""
    from pipeline import (  # noqa: PLC0415  # call-time: giants re-exported by facade
        export_per_frame_catalogs,
        generate_masterstar_and_catalog,
    )

    import numpy as np

    ap = Path(archive_path)
    obs_group_key = str(job.get("gkey", "") or "")
    scanning_id = int(job.get("scanning_id", 0) or 0)
    _job_md = job.get("master_dark_path")
    _master_dark_bpm_path: Path | None = Path(str(_job_md)) if _job_md and str(_job_md).strip() else None
    if _master_dark_bpm_path is not None and not _master_dark_bpm_path.is_file():
        _master_dark_bpm_path = None
    detrended_root = Path(job["detrended_root"])
    aligned_root = Path(job["aligned_root"])
    platesolve_dir = Path(job["platesolve_dir"])
    files = list(job["files"])
    # Zarovnane FITS: {archive}/detrended_aligned/lights/{filter_exp_binning}/... (vnorene cesty podla vstupu)
    os.makedirs(str(aligned_root), exist_ok=True)
    os.makedirs(str(platesolve_dir), exist_ok=True)
    _cfg_align = app_config or AppConfig()
    _align_star_cap = max(10, min(5000, int(_cfg_align.alignment_max_stars)))
    # Keep alignment input stable on dense fields: use at most TOP 200 brightest stars.
    _align_star_cap = min(_align_star_cap, 200)
    _sips_sig = float(_cfg_align.sips_dao_threshold_sigma)
    if not math.isfinite(_sips_sig) or _sips_sig <= 0:
        _sips_sig = 3.5
    _cfg_align_sig = float(_cfg_align.alignment_detection_sigma)
    if not math.isfinite(_cfg_align_sig) or _cfg_align_sig <= 0:
        _cfg_align_sig = _sips_sig
    try:
        _ui_sig = float(dao_threshold_sigma)
    except (TypeError, ValueError):
        _ui_sig = 0.0
    if not math.isfinite(_ui_sig) or _ui_sig <= 0:
        _ui_sig = 0.0
    # Session override > Settings alignment_detection_sigma > sips_dao_threshold_sigma.
    _align_det_sigma = max(
        0.8,
        min(20.0, _ui_sig if _ui_sig > 0 else _cfg_align_sig),
    )
    _fb_align = float(_cfg_align.sips_dao_fwhm_px)
    if not math.isfinite(_fb_align) or _fb_align <= 0:
        _fb_align = 2.5
    _pfov_align: float | None = None
    if build_masterstar_and_catalogs:
        LOGGER.info("Astrometria + MASTERSTAR + per-frame CSV: archiv %s", ap)
    else:
        LOGGER.info("Astrometria + zarovnanie + per-frame CSV (bez MASTERSTAR): archiv %s", ap)
    # MASTERSTAR initial match: allow a looser sep (min 10") for robust first-pass Gaia join.
    _catalog_match_sep_eff = max(10.0, float(catalog_match_max_sep_arcsec))
    if _catalog_match_sep_eff > float(catalog_match_max_sep_arcsec) + 1e-9:
        _pipeline_ui_info(
            f"Katalogovy match prah zvyseny na {_catalog_match_sep_eff:.2f}\" "
            "(minimum pre robustny pociatocny cross-match)."
        )

    _cat_loc_only = bool(catalog_local_gaia_only) if catalog_local_gaia_only is not None else True
    if _cat_loc_only:
        LOGGER.info("Katalog: rezim lokalny Gaia (SQLite)")
    equip_sat_adu = _equipment_saturate_adu_from_db(id_equipment)
    if draft_id is not None and files:
        try:
            _db_sat = VyvarDatabase(Path(_cfg_align.database_path))
            try:
                _cmb_sat = _db_sat.get_combined_metadata(files[0], int(draft_id))
                if _cmb_sat.get("saturate_adu") is not None:
                    equip_sat_adu = _cmb_sat["saturate_adu"]
            finally:
                _db_sat.conn.close()
        except Exception:  # noqa: BLE001
            # EXC-0414: T4 -- `_db_pf.conn.close()` after plate-scale lookup for per-frame match sep; cleanup only. (EXCEPT-BULK-2 2026-07-08)
            pass
    if not files:
        raise FileNotFoundError(
            f"Chybaju FITS v {detrended_root}. Plate solve cita len **spracovane** snimky. "
            "Najprv spusti **MAKE MASTERSTAR** po kroku **Analyze** (zapis do "
            f"`{ap / 'processed' / 'lights'}` alebo starsie `{ap / 'detrended' / 'lights'}`)."
        )

    _t_step3_start = time.time()
    n_files = len(files)
    ref_fp, ref_star_scores = _pick_reference_frame_by_star_count(files)
    if osc_registration_handoff is not None:
        ref_name = str(osc_registration_handoff.get("reference_file") or "")
        for _rf in files:
            if _rf.name == ref_name:
                ref_fp = _rf
                break
    # Read reference once (no lock during solve step).
    with fits.open(ref_fp, memmap=False) as hdul:
        ref_hdr = hdul[0].header.copy()
        ref_data = _as_fits_float32_image(hdul[0].data).astype(np.float32, copy=False)
    _rh, _rw = int(ref_data.shape[0]), int(ref_data.shape[1])
    try:
        _pf_try = float(plate_solve_fov_deg)
        if math.isfinite(_pf_try) and _pf_try > 0:
            _pfov_align = _pf_try
    except (TypeError, ValueError):
        _pfov_align = None
    if _pfov_align is None:
        _pfov_align = resolve_plate_solve_fov_deg_hint(
            ref_hdr,
            _rh,
            _rw,
            database_path=_cfg_align.database_path,
            equipment_id=int(id_equipment) if id_equipment is not None else None,
            draft_id=int(draft_id) if draft_id is not None else None,
        )
    if _pfov_align is None:
        _pfov_align = float(_cfg_align.plate_solve_fov_deg)

    _scale_pf: float | None = None
    _db_pf = _vyvar_open_database(_cfg_align)
    if _db_pf is not None:
        try:
            _eq_pf, _tel_pf = resolve_optics_ids_for_platesolve(
                _db_pf, draft_id, equipment_id=id_equipment
            )
            _xb_pf, _yb_pf = fits_binning_xy_from_header(ref_hdr)
            _bin_pf = max(1, int(_xb_pf), int(_yb_pf))
            _scale_pf = compute_plate_scale_from_db(_eq_pf, _tel_pf, _db_pf.conn, binning=_bin_pf)
        except Exception:  # noqa: BLE001
            _scale_pf = None
        finally:
            try:
                _db_pf.conn.close()
            except Exception:  # noqa: BLE001
                pass
    _j_psep = job.get("per_frame_catalog_match_sep_arcsec")
    if _j_psep is not None:
        try:
            per_frame_match_sep = float(_j_psep)
        except (TypeError, ValueError):
            per_frame_match_sep = per_frame_catalog_match_sep_arcsec_for_scale(_scale_pf)
    else:
        per_frame_match_sep = per_frame_catalog_match_sep_arcsec_for_scale(_scale_pf)
    if not math.isfinite(per_frame_match_sep) or per_frame_match_sep <= 0:
        per_frame_match_sep = per_frame_catalog_match_sep_arcsec_for_scale(_scale_pf)

    has_wcs = _has_valid_wcs(ref_hdr)
    solve_steps = 0 if has_wcs else 1
    master_steps = 1 if build_masterstar_and_catalogs else 0
    global_total = max(1, 1 + solve_steps + n_files + master_steps + n_files)
    prog_i = [0]

    def _prog(msg: str) -> None:
        if progress_cb is None:
            return
        prog_i[0] += 1
        progress_cb(prog_i[0], global_total, msg)

    # --- MASTERSTAR build + plate-solve (per-setup platesolve/) before alignment ---
    # IMPORTANT (multi-filter): each setup must have its own MASTERSTAR + catalogs, otherwise
    # R/V/B runs overwrite each other (MASTERSTAR.fits, masterstars_full_match.csv, VY_MIRR, ...)
    # and reference/per-frame astrometry becomes unstable.
    _masterstar_built = False
    _cat_info_root: dict[str, Any] = {}
    _ps_root = platesolve_dir
    _t_platesolve = time.time()
    if build_masterstar_and_catalogs:
        _prog("platesolve/MASTERSTAR: referencny snimok + plate-solve + katalogy...")
        _cat_info_root = generate_masterstar_and_catalog(
            archive_path=ap,
            max_catalog_rows=int(max_catalog_rows),
            astrometry_api_key=astrometry_api_key,
            source_root=detrended_root,
            platesolve_dir=_ps_root,
            platesolve_backend=platesolve_backend,
            plate_solve_fov_deg=float(_pfov_align),
            catalog_match_max_sep_arcsec=float(_catalog_match_sep_eff),
            saturate_level_fraction=float(saturate_level_fraction),
            n_comparison_stars=int(n_comparison_stars),
            require_non_variable_comparisons=bool(require_non_variable_comparisons),
            faintest_mag_limit=faintest_mag_limit,
            dao_threshold_sigma=float(dao_threshold_sigma),
            equipment_saturate_adu=equip_sat_adu,
            catalog_local_gaia_only=_cat_loc_only,
            app_config=_cfg_align,
            equipment_id=id_equipment,
            draft_id=draft_id,
            master_dark_path=_master_dark_bpm_path,
            masterstar_candidate_paths=job.get("masterstar_candidate_paths"),
            masterstar_selection_pct=job.get("masterstar_selection_pct"),
            setup_name=obs_group_key or None,
            masterstar_basename="MASTERSTAR.fits",
            masterstars_csv_basename="masterstars_full_match.csv",
            masterstar_fits_only=False,
            masterstar_skip_build=False,
        )
        _masterstar_built = True

        # Prefer MASTERSTAR as the canonical alignment reference when available.
        # This guarantees that:
        # - the output pixel grid matches MASTERSTAR (no WCS/data grid mismatch),
        # - per-frame matching against masterstars_full_match.csv works reliably.
        try:
            _ms_fp = _cat_info_root.get("masterstar_fits") if isinstance(_cat_info_root, dict) else None
            if _ms_fp:
                _ms_path = Path(str(_ms_fp)).resolve()
                if _ms_path.is_file():
                    with fits.open(_ms_path, memmap=False) as hdul:
                        ref_hdr = hdul[0].header.copy()
                        ref_data = _as_fits_float32_image(hdul[0].data).astype(np.float32, copy=False)
                    ref_fp = _ms_path
                    has_wcs = _has_valid_wcs(ref_hdr)
                    log_event(f"INFO: Alignment reference set to MASTERSTAR: {ref_fp.name}")
        except Exception as _ms_ref_exc:  # noqa: BLE001
            # EXC-0416: T1 -- Copying MASTERSTAR WCS onto reference FITS fails silently; aligned products can carry a... (EXCEPT-BULK-2 2026-07-08)
            from except_fix_counters import get_except_fix_counters

            get_except_fix_counters().masterstar_ref_swap_fail += 1
            LOGGER.error("Using MASTERSTAR as alignment reference failed: %s", _ms_ref_exc)

    _prog(
        f"detrended_aligned/lights: pripravujem zarovnanie ({n_files} snimok z {detrended_root.name}/...)..."
    )

    # If MASTERSTAR was built for this setup and has a valid WCS, prefer it as the canonical
    # WCS for detrended_aligned products. Some frames already carry a WCS that can be offset by
    # arcminutes from MASTERSTAR; using it would break per-frame Gaia matching (master_reference_sky).
    if build_masterstar_and_catalogs and isinstance(_cat_info_root, dict):
        try:
            _ms_fp = _cat_info_root.get("masterstar_fits")
            if _ms_fp:
                _ms_path = Path(str(_ms_fp)).resolve()
                if _ms_path.is_file():
                    with fits.open(_ms_path, memmap=False) as _ms_hdul:
                        _ms_hdr = _ms_hdul[0].header.copy()
                    if _has_valid_wcs(_ms_hdr):
                        _apply_wcs_header_to_fits(ref_fp, _ms_hdr)
                        with fits.open(ref_fp, memmap=False) as hdul:
                            ref_hdr = hdul[0].header.copy()
                            ref_data = _as_fits_float32_image(hdul[0].data).astype(np.float32, copy=False)
                        has_wcs = True
                        log_event(
                            f"INFO: Reference WCS prevzate z MASTERSTAR ({_ms_path.name}) - pouzijem MASTERSTAR WCS pre alignment aj per-frame match."
                        )
        except Exception as _wcs_copy_exc:  # noqa: BLE001
            try:
                log_event(f"DEBUG: Reference WCS copy from MASTERSTAR failed: {_wcs_copy_exc}")
            except Exception:  # noqa: BLE001
                pass

    if (
        not build_masterstar_and_catalogs
        and sibling_recovery_use_masterstar
        and not _masterstar_built
    ):
        try:
            _ms_path = (platesolve_dir / "MASTERSTAR.fits").resolve()
            if _ms_path.is_file():
                with fits.open(_ms_path, memmap=False) as hdul:
                    _ms_hdr = hdul[0].header.copy()
                    ref_data = _as_fits_float32_image(hdul[0].data).astype(np.float32, copy=False)
                if _has_valid_wcs(_ms_hdr):
                    ref_fp = _ms_path
                    ref_hdr = _ms_hdr
                    has_wcs = True
                    _cat_info_root = {
                        "masterstar_fits": str(_ms_path),
                        "masterstars_csv": str(platesolve_dir / "masterstars_full_match.csv"),
                    }
                    log_event(
                        f"INFO: Sibling-recovery alignment using existing MASTERSTAR: {_ms_path.name}"
                    )
        except Exception as _sib_ms_exc:  # noqa: BLE001
            # EXC-0418: T3 -- DEBUG min/max/mean/NaN stats logging `pass`; alignment detection proceeds regardless. (EXCEPT-BULK-2 2026-07-08)
            log_event(f"DEBUG: Sibling-recovery MASTERSTAR load failed: {_sib_ms_exc}")

    if not has_wcs:
        _prog("Plate solve referencie (moze chvilu trvat)...")

    if not has_wcs:
        # Solve reference file in-place (no open handle on Windows).
        _hra, _hdec, _ = _pointing_hint_from_header(ref_hdr)
        solve = _solve_wcs_external(
            ref_fp,
            backend=platesolve_backend,
            astrometry_api_key=astrometry_api_key,
            plate_solve_fov_deg=float(_pfov_align),
            hint_ra_deg=_hra,
            hint_dec_deg=_hdec,
            app_config=_cfg_align,
            equipment_id=id_equipment,
            draft_id=draft_id,
        )
        if not solve.get("solved", False):
            raise RuntimeError(f"Reference astrometry failed: {solve.get('reason')}")
        # Reload solved header/data
        with fits.open(ref_fp, memmap=False) as hdul:
            ref_hdr = hdul[0].header.copy()
            ref_data = _as_fits_float32_image(hdul[0].data).astype(np.float32, copy=False)

    print(f"  Plate solve: {time.time() - _t_platesolve:.1f}s")
    _t_align = time.time()

    # Use the same FWHM rule as per-frame alignment (VY_FWHM / header), not only ``sips_dao_fwhm_px``.
    # A fixed ~2.5 px kernel on the reference while sources use ~5 px yields different brightest-N
    # orderings -> bogus point pairs -> astroalign "triangles exhausted" and identity/no_wcs cascades.
    _raw_ref_fw = dao_detection_fwhm_pixels(ref_hdr, configured_fallback=_fb_align)
    try:
        _fwv = float(_raw_ref_fw) if _raw_ref_fw is not None else float("nan")
    except (TypeError, ValueError):
        _fwv = float("nan")
    _align_fwhm_ref = float(_fwv) if math.isfinite(_fwv) and _fwv > 0 else float(_fb_align)

    hint_center = _wcs_field_center_radec_deg(ref_fp)
    hint_ra: float | None = hint_center[0] if hint_center else None
    hint_dec: float | None = hint_center[1] if hint_center else None

    extra_platesolve_results: list[dict[str, Any]] = []

    try:
        pass  # type: ignore
    except Exception as exc:  # noqa: BLE001
        logging.error('[EXC-0419] `estimate_archive_memory_profile` failure leaves `use_ram_handoff=True`; may exhaust RA...: %s', exc)
        raise RuntimeError(f"astroalign required for frame registration: {exc}") from exc

    log_event(
        f"Detekcia hviezd: Pouzite FWHM={_align_fwhm_ref:.2f}, Sigma={_align_det_sigma:.2f}"
    )

    def _maybe_refine_aligned(
        hdr_mut: fits.Header, data_mut: np.ndarray, label: str, *, dao_fwhm_px_frame: float
    ) -> None:
        _ = (hdr_mut, data_mut, label, dao_fwhm_px_frame)
        return

    # Adaptive alignment star budget:
    # - if STAR_COUNT > 1000 -> use top 300 brightest
    # - if STAR_COUNT < 100 -> use all
    # - else -> cap at 300
    data_to_detect = np.asarray(ref_data, dtype=np.float32)
    try:
        log_event(
            "DEBUG: Data stats for alignment - "
            f"Min: {np.min(data_to_detect):.2f}, "
            f"Max: {np.max(data_to_detect):.2f}, "
            f"Mean: {np.mean(data_to_detect):.2f}, "
            f"NaN count: {np.isnan(data_to_detect).sum()}"
        )
    except Exception:  # noqa: BLE001
        pass
    ref_xy_all = _alignment_detect_xy(
        data_to_detect,
        int(max(100, max_detected_stars)),
        det_sigma=_align_det_sigma,
        fwhm_px=_align_fwhm_ref,
        label=ref_fp.name,
        log_sink=None,
    )
    n_ref = int(len(ref_xy_all))
    if n_ref > 1000:
        n_keep = 300
    elif n_ref < 100:
        n_keep = n_ref
    else:
        n_keep = min(300, n_ref)
    ref_xy = ref_xy_all[:n_keep]
    if len(ref_xy) < int(min_detected_stars):
        raise RuntimeError(
            f"Reference frame has too few detected stars ({len(ref_xy)} < {min_detected_stars})."
        )
    ref_xy_fit = ref_xy[: int(min(_align_star_cap, len(ref_xy)))]
    log_event(
        f"Zarovnanie referencia {ref_fp.name}: DAO hviezd={len(ref_xy)}, "
        f"cap pre transform={_align_star_cap}, DAO sigma={_align_det_sigma:.2f}, FWHM={_align_fwhm_ref:.2f}px "
        f"(QC VY_FWHM alebo sips_dao_fwhm_px)"
    )

    # Auto RAM management: default in-memory, but switch to disk when estimated working set exceeds 70% of available RAM.
    use_ram_handoff = bool(ram_align_and_catalog)
    try:
        mp = estimate_archive_memory_profile(ap)
        avail = mp.get("available_ram_bytes")
        prh = mp.get("platesolve_ram_handoff") or {}
        tot = prh.get("estimated_total_conservative_bytes")
        if isinstance(avail, int) and isinstance(tot, int) and avail > 0 and tot > 0:
            if tot > int(0.70 * avail):
                use_ram_handoff = False
    except Exception:  # noqa: BLE001
        # EXC-0420: T1 -- `_vy_fwhm_header_value` returns `None` on read error; MASTERSTAR match heuristic falls ... (EXCEPT-BULK-2 2026-07-08)
        pass

    aligned_ram_buffer: list[tuple[str, fits.Header, np.ndarray]] = []
    aligned_files: list[Path] = []
    star_counts: list[dict[str, Any]] = []
    rotation_ref_angle_deg: float | None = None
    rotation_flip_frame_indices_1based: list[int] = []
    rotation_flip_first_index_1based: int | None = None
    _flip_logged = False

    try:
        rotation_ref_angle_deg = wcs_rotation_angle_deg(ref_hdr)
    except Exception:  # noqa: BLE001
        # EXC-0421: T2 -- `_aligned_masterstar_matches_platesolve` stat/read failure returns `False`, allowing al... (EXCEPT-BULK-2 2026-07-08)
        rotation_ref_angle_deg = None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            ref_wcs_obj = WCS(ref_hdr) if _has_valid_wcs(ref_hdr) else None
    except Exception:  # noqa: BLE001
        ref_wcs_obj = None

    # save reference as aligned baseline, keep WCS
    _prog(
        f"detrended_aligned/lights: {'RAM - referencia' if use_ram_handoff else 'zapisujem FITS'} "
        f"{ref_fp.name} (1/{n_files})"
    )
    try:
        ref_rel = ref_fp.relative_to(detrended_root)
    except Exception:  # noqa: BLE001
        # Reference can live outside detrended_root (e.g. MASTERSTAR in platesolve/...).
        # In that case, store it at the aligned root top-level.
        ref_rel = Path(ref_fp.name)
    ref_out = aligned_root / ref_rel
    _ensure_parent_dirs_for_aligned_fits(ref_out)

    def _vy_fwhm_header_value(path: Path) -> float | None:
        try:
            with fits.open(path, memmap=False) as _h:
                _v = _h[0].header.get("VY_FWHM")
            if _v is None:
                return None
            vv = float(_v)
            return float(vv) if math.isfinite(vv) and vv > 0 else None
        except Exception:  # noqa: BLE001
            return None

    def _aligned_masterstar_matches_platesolve(*, platesolve_masterstar: Path, aligned_masterstar: Path) -> bool:
        """Heuristic match: size+mtime OR size+VY_FWHM.

        This is intentionally lightweight to prevent unintended overwrites of the aligned MASTERSTAR baseline.
        """
        try:
            if not platesolve_masterstar.is_file() or not aligned_masterstar.is_file():
                return False
            s_src = platesolve_masterstar.stat()
            s_dst = aligned_masterstar.stat()
            if int(s_src.st_size) != int(s_dst.st_size):
                return False
            # Preferred fast path: preserved mtime via shutil.copy2
            if int(s_src.st_mtime) == int(s_dst.st_mtime):
                return True
        except Exception:  # noqa: BLE001
            return False
        # Header fallback: VY_FWHM should be stable for a given MASTERSTAR (copied+solved).
        v_src = _vy_fwhm_header_value(platesolve_masterstar)
        v_dst = _vy_fwhm_header_value(aligned_masterstar)
        if v_src is None or v_dst is None:
            return False
        return abs(float(v_src) - float(v_dst)) <= 1e-9

    def _ensure_aligned_masterstar_copy(*, platesolve_masterstar: Path, aligned_masterstar: Path) -> None:
        import shutil  # local import: narrow scope

        if aligned_masterstar.is_file() and _aligned_masterstar_matches_platesolve(
            platesolve_masterstar=platesolve_masterstar, aligned_masterstar=aligned_masterstar
        ):
            log_event(
                "[MASTERSTAR] aligned ref already exists and matches platesolve source - skipping write"
            )
            return
        log_event("[MASTERSTAR] copying platesolve MASTERSTAR -> detrended_aligned ref")
        shutil.copy2(platesolve_masterstar, aligned_masterstar)

    _ps_masterstar = (platesolve_dir / "MASTERSTAR.fits")
    _is_platesolve_masterstar_ref = False
    try:
        _is_platesolve_masterstar_ref = (
            ref_fp.name.strip().casefold() == "masterstar.fits"
            and _ps_masterstar.is_file()
            and ref_fp.resolve() == _ps_masterstar.resolve()
            and ref_out.name.strip().casefold() == "masterstar.fits"
        )
    except Exception:  # noqa: BLE001
        _is_platesolve_masterstar_ref = False

    if _is_platesolve_masterstar_ref:
        _ensure_aligned_masterstar_copy(
            platesolve_masterstar=_ps_masterstar.resolve(),
            aligned_masterstar=ref_out,
        )
    else:
        with fits.open(ref_fp, memmap=False) as hdul:
            hdr = hdul[0].header.copy()
            data = _as_fits_float32_image(hdul[0].data)
        hdr["VY_ALGN"] = (True, "Aligned to reference")
        hdr["VYALGOK"] = (True, "Alignment OK")
        hdr["VY_REF"] = (ref_fp.name[:60], "Reference frame for alignment")
        _maybe_refine_aligned(hdr, data, ref_fp.name, dao_fwhm_px_frame=_align_fwhm_ref)
        if use_ram_handoff:
            aligned_ram_buffer.append((ref_rel.as_posix(), hdr.copy(), np.copy(data)))
        else:
            fits.writeto(ref_out, data, header=hdr, overwrite=True)
    aligned_files.append(ref_out)
    star_counts.append(
        {
            "file": ref_fp.name,
            "frame_index": int(files.index(ref_fp) + 1) if ref_fp in files else 1,
            "detected_stars": int(len(ref_xy)),
            "aligned": True,
            "alignment_method": "reference",
            "is_flipped": False,
            "rotation_angle_deg": rotation_ref_angle_deg,
        }
    )

    # Align every other frame to reference (skip duplicate if ref is not files[0]).
    # Astroalign control-point cap from cfg (decoupled from detection ladder max_stars).
    align_cp = int(max(12, min(500, int(_cfg_align.alignment_max_control_points))))
    ref_pts = np.asarray(ref_xy_fit, dtype=np.float32)
    if ref_pts is None or len(ref_pts) == 0:
        raise ValueError("Referencne hviezdy su prazdne pred startom alignmentu!")
    # Keep immutable backup of reference points; never overwrite with per-frame source detections.
    fixed_target_pts = np.copy(ref_pts).astype("float32")
    log_event(f"DEBUG: Start alignment, reference stars N = {len(fixed_target_pts)}")
    # Brute-force isolation from any shared numpy buffers.
    REFERENCE_LIST = fixed_target_pts.tolist()
    LOGGER.info(
        "Astrometry alignment: astroalign uses up to %s control points on up to %s DAO sources per frame",
        align_cp,
        _align_star_cap,
    )

    n_written_align = 1
    n_align_workers = _vyvar_parallel_worker_count(_cfg_align)
    align_tasks: list[tuple[str, int]] = []
    for frame_index_1based, fp in enumerate(files, start=1):
        if fp == ref_fp:
            continue
        align_tasks.append((str(fp.resolve()), int(frame_index_1based)))

    _align_ctx: dict[str, Any] = {
        "ref_data": ref_data,
        "ref_hdr": ref_hdr.copy(),
        "ref_fp_name": ref_fp.name,
        "fixed_target_pts": np.copy(fixed_target_pts).astype(np.float32, copy=False),
        "reference_list": list(REFERENCE_LIST),
        "has_ref_wcs": ref_wcs_obj is not None,
        "platesolve_dir": str(platesolve_dir),
        "align_star_cap": int(_align_star_cap),
        "max_control_points": int(align_cp),
        "min_detected_stars": int(min_detected_stars),
        "max_detected_stars": int(max_detected_stars),
        "fb_align": float(_fb_align),
        "rotation_ref_angle_deg": rotation_ref_angle_deg,
    }
    _osc_registration_capture: dict[str, dict[str, Any]] = {}

    def _flush_one_alignment(res: dict[str, Any]) -> None:
        nonlocal n_written_align, _flip_logged, rotation_flip_first_index_1based
        idx = int(res["frame_index_1based"])
        fp = Path(res["fp"])
        if bool(res.get("is_flipped", False)):
            rotation_flip_frame_indices_1based.append(idx)
            if rotation_flip_first_index_1based is None:
                rotation_flip_first_index_1based = idx
            if not _flip_logged:
                log_event(
                    f"Physical rotation change detected at frame index {idx}. "
                    "Adjusting alignment strategy."
                )
                _flip_logged = True
        if res.get("kind") == "failed_skip":
            star_counts.append(res["star_count"])
            return
        hdr_out = res["hdr"]
        aligned_data = res["aligned_data"]
        fw_i = float(res["fw_i"])
        _maybe_refine_aligned(hdr_out, aligned_data, fp.name, dao_fwhm_px_frame=fw_i)
        try:
            fp_rel = fp.relative_to(detrended_root)
        except Exception:  # noqa: BLE001
            fp_rel = Path(fp.name)
        out_fp = aligned_root / fp_rel
        _ensure_parent_dirs_for_aligned_fits(out_fp)
        n_written_align += 1
        _prog(
            f"detrended_aligned/lights: "
            f"{'RAM - zarovnanie' if use_ram_handoff else 'zapisujem FITS'} "
            f"{fp.name} ({n_written_align}/{n_files})..."
        )
        if use_ram_handoff:
            aligned_ram_buffer.append((fp_rel.as_posix(), hdr_out.copy(), np.copy(aligned_data)))
        else:
            fits.writeto(out_fp, aligned_data, header=hdr_out, overwrite=True)
        aligned_files.append(out_fp)
        star_counts.append(res["star_count"])
        rh = res.get("registration_handoff")
        if isinstance(rh, dict) and res.get("star_count", {}).get("file"):
            _osc_registration_capture[str(res["star_count"]["file"])] = dict(rh)

    if osc_registration_handoff is not None:
        from osc_align import apply_registration_handoff_to_frame

        handoff_frames = dict(osc_registration_handoff.get("frames") or {})
        ref_name = str(osc_registration_handoff.get("reference_file") or ref_fp.name)
        for fp in files:
            if fp == ref_fp and fp.name != ref_name:
                continue
            if fp.name == ref_name:
                continue
            entry = handoff_frames.get(fp.name) or {}
            if entry and not bool(entry.get("aligned", True)):
                star_counts.append(
                    {
                        "file": fp.name,
                        "frame_index": int(files.index(fp) + 1) if fp in files else 0,
                        "detected_stars": 0,
                        "aligned": False,
                        "reason": "donor_not_aligned",
                        "alignment_method": "osc_handoff_skip",
                        "is_flipped": False,
                    }
                )
                continue
            with fits.open(fp, memmap=False) as hdul:
                raw_hdr = hdul[0].header.copy()
                raw_data = _as_fits_float32_image(hdul[0].data).astype(np.float32, copy=False)
            aligned_data, hdr_out, method = apply_registration_handoff_to_frame(
                frame_path=fp,
                frame_data=raw_data,
                frame_hdr=raw_hdr,
                ref_data=ref_data,
                ref_hdr=ref_hdr,
                handoff_entry=entry,
            )
            _flush_one_alignment(
                {
                    "kind": "aligned",
                    "fp": str(fp.resolve()),
                    "frame_index_1based": int(files.index(fp) + 1) if fp in files else 0,
                    "is_flipped": False,
                    "hdr": hdr_out,
                    "aligned_data": aligned_data,
                    "aligned_method": method,
                    "fw_i": float(_fb_align),
                    "star_count": {
                        "file": fp.name,
                        "frame_index": int(files.index(fp) + 1) if fp in files else 0,
                        "detected_stars": 0,
                        "aligned": True,
                        "alignment_method": method,
                        "is_flipped": False,
                    },
                }
            )
    elif n_align_workers > 1 and len(align_tasks) > 1:
        _mp_ctx: dict[str, Any] = {
            "ref_data": np.ascontiguousarray(np.copy(_align_ctx["ref_data"])),
            "ref_hdr": _align_ctx["ref_hdr"].copy(),
            "ref_fp_name": _align_ctx["ref_fp_name"],
            "fixed_target_pts": np.copy(_align_ctx["fixed_target_pts"]).astype(np.float32, copy=False),
            "reference_list": list(_align_ctx["reference_list"]),
            "has_ref_wcs": bool(_align_ctx["has_ref_wcs"]),
            "platesolve_dir": str(_align_ctx["platesolve_dir"]),
            "align_star_cap": int(_align_ctx["align_star_cap"]),
            "max_control_points": int(_align_ctx["max_control_points"]),
            "min_detected_stars": int(_align_ctx["min_detected_stars"]),
            "max_detected_stars": int(_align_ctx["max_detected_stars"]),
            "fb_align": float(_align_ctx["fb_align"]),
            "rotation_ref_angle_deg": _align_ctx["rotation_ref_angle_deg"],
        }
        def _run_alignment_single_process() -> None:
            for fp_s, idx in align_tasks:
                res = _alignment_compute_one_frame(Path(fp_s), int(idx), _align_ctx, None)
                _flush_one_alignment(res)

        # A-durable: resolve the MP init/task by FRESH module attribute at call time, so the
        # objects handed to the spawn pool are exactly what sys.modules resolves - even if the
        # Streamlit file-watcher reloaded vyvar_alignment_frame after pipeline.py was imported
        # (the import-time `from ... import` binding would otherwise go stale -> PicklingError).
        _mp_init = vyvar_alignment_frame._astrometry_align_mp_init
        _mp_task = vyvar_alignment_frame._astrometry_align_mp_task
        try:
            with ProcessPoolExecutor(
                max_workers=n_align_workers,
                initializer=_mp_init,
                initargs=(_mp_ctx,),
            ) as pool:
                raw_list = list(pool.map(_mp_task, align_tasks, chunksize=1))
            for res in raw_list:
                for ln in res.get("log_events", ()):
                    log_event(ln)
                res_flush = {k: v for k, v in res.items() if k != "log_events"}
                _flush_one_alignment(res_flush)
        except pickle.PicklingError as _pkl_err:
            # Robust fallback: if the spawn pool cannot pickle the MP funcs (e.g. mid-run module
            # reload defeats the fresh-attr lookup), run alignment single-process instead of
            # aborting. Photometry is byte-identical to the MP path (same per-frame compute).
            # PicklingError is raised at task-submission (pickling the worker funcs) before any
            # result is flushed, so no partial per-frame state exists here - run single-process.
            _pipeline_ui_info(
                f"Alignment: multiprocessing dispatch failed to pickle worker functions "
                f"({_pkl_err}); falling back to single-process alignment."
            )
            _run_alignment_single_process()
    else:
        for fp_s, idx in align_tasks:
            res = _alignment_compute_one_frame(Path(fp_s), int(idx), _align_ctx, None)
            _flush_one_alignment(res)

    n_aligned = int(sum(1 for r in star_counts if r.get("aligned")))
    n_failed_align = int(sum(1 for r in star_counts if not bool(r.get("aligned"))))
    if n_failed_align > 0:
        reasons: dict[str, int] = {}
        for r in star_counts:
            if bool(r.get("aligned")):
                continue
            rr = str(r.get("reason") or "unknown")
            reasons[rr] = int(reasons.get(rr, 0)) + 1
        reason_txt = ", ".join(f"{k}={v}" for k, v in sorted(reasons.items(), key=lambda kv: (-kv[1], kv[0]))[:5])
        _pipeline_ui_info(
            f"Alignment warning: zlyhalo {n_failed_align}/{len(files)} snimok v {aligned_root.resolve()} "
            f"(dovody: {reason_txt})."
        )
    if n_aligned <= 1:
        msg = (
            f"Alignment zlyhal: uspesne zarovnana len referencia (1/{len(files)}). "
            f"Skontroluj DAO prah/FWHM a WCS vstupy. Vystupny priecinok: {aligned_root.resolve()}."
        )
        _pipeline_ui_error(msg)
        raise RuntimeError(msg)
    rep_path = platesolve_dir / "alignment_report.csv"
    pd.DataFrame(star_counts).to_csv(rep_path, index=False)
    if osc_write_registration_handoff or bool(job.get("osc_write_registration_handoff")):
        from osc_align import write_registration_handoff

        write_registration_handoff(
            platesolve_dir,
            reference_file=str(ref_fp.name),
            frames=_osc_registration_capture,
        )

    print(f"  Zarovnanie: {time.time() - _t_align:.1f}s")
    _t_csv = time.time()

    # If we aligned in RAM, flush aligned FITS to disk before MASTERSTAR (needs files on disk).
    _ram_flushed_before_masterstar = False
    if use_ram_handoff and aligned_ram_buffer and build_masterstar_and_catalogs:
        _prog("detrended_aligned/lights: zapisujem FITS na disk (RAM -> disk, pred MASTERSTAR)...")
        for name, hdr, arr in aligned_ram_buffer:
            _target = aligned_root / Path(name)
            _ensure_parent_dirs_for_aligned_fits(_target)
            fits.writeto(
                _target,
                _as_fits_float32_image(arr),
                header=hdr,
                overwrite=True,
            )
        _ram_flushed_before_masterstar = True
        _aligned_file_list = sorted(aligned_root.glob("proc_*.fits"))
        LOGGER.info(f"[BORDER] RAM flush done - {len(_aligned_file_list)} aligned frames on disk")

    cat_info: dict[str, Any] = {}
    ms_csv: Path | None = None
    ms_fits: Path | None = None
    use_master_fast = False

    if build_masterstar_and_catalogs:
        # Use per-setup platesolve/ artifacts built before alignment.
        cat_info = dict(_cat_info_root or {})
        try:
            ms_csv = Path(str((cat_info.get("masterstars_csv") or (_ps_root / "masterstars_full_match.csv")))).resolve()
        except Exception:  # noqa: BLE001
            # EXC-0422: T3 -- Optional `variable_targets_csv` path probe `pass`; UI metadata field may be absent. (EXCEPT-BULK-2 2026-07-08)
            ms_csv = Path(str(cat_info.get("masterstars_csv") or (_ps_root / "masterstars_full_match.csv")))
        try:
            ms_fits = Path(str((cat_info.get("masterstar_fits") or (_ps_root / "MASTERSTAR.fits")))).resolve()
        except Exception as exc:  # noqa: BLE001
            logging.error('[EXC-0423] Post-RAM-flush DB-aware photometry plan rewrite `pass`es; border-safe bbox may use stal...: %s', exc)
            ms_fits = Path(str(cat_info.get("masterstar_fits") or (_ps_root / "MASTERSTAR.fits")))

        # comparison_stars.csv / variable_targets.csv are produced in this setup directory already.
        try:
            _comp = platesolve_dir / "comparison_stars.csv"
            _var = platesolve_dir / "variable_targets.csv"
            if _comp.is_file():
                cat_info["comparison_stars_csv"] = str(_comp)
            if _var.is_file():
                cat_info["variable_targets_csv"] = str(_var)
        except Exception:  # noqa: BLE001
            pass

        # Masterstar lock for Step 3: per-frame catalogs must use one fixed reference list.
        use_master_fast = True

        # Recompute photometry plan after alignment so border-safe bbox uses aligned frames.
        try:
            _aligned_disk: list[Path] = []
            if use_ram_handoff and aligned_ram_buffer and _ram_flushed_before_masterstar:
                _aligned_disk = list(_aligned_file_list)
            else:
                try:
                    _aligned_disk = sorted(aligned_root.glob("proc_*.fits"))
                except Exception:  # noqa: BLE001
                    _aligned_disk = []
            _has_aligned = bool(_aligned_disk) or bool(use_ram_handoff and aligned_ram_buffer)
            if build_masterstar_and_catalogs and _has_aligned:
                _wp_aligned = write_photometry_plan_files(
                    platesolve_dir=platesolve_dir,
                    masterstar_fits=ms_fits or (platesolve_dir / "MASTERSTAR.fits"),
                    masterstars_csv=ms_csv or (platesolve_dir / "masterstars_full_match.csv"),
                    n_comparison_stars=int(n_comparison_stars),
                    require_non_variable=bool(require_non_variable_comparisons),
                    draft_id=int(draft_id) if draft_id is not None else None,
                    database_path=(
                        Path(str(_cfg_align.database_path))
                        if str(_cfg_align.database_path or "").strip()
                        else None
                    ),
                    aligned_files=_aligned_disk if _aligned_disk else None,
                    aligned_ram_frames=aligned_ram_buffer if use_ram_handoff and aligned_ram_buffer else None,
                    require_safe_bbox=True,
                )
                cat_info.update(_wp_aligned or {})
        except RuntimeError as _wp_exc:
            log_event(f"[BORDER] Post-alignment photometry plan rewrite failed: {_wp_exc!s}")
            if "[BORDER]" in str(_wp_exc):
                raise
        except Exception as _wp_exc:  # noqa: BLE001
            log_event(f"[BORDER] Post-alignment photometry plan rewrite failed: {_wp_exc!s}")

    export_base = prog_i[0]
    _catalog_app_cfg = _cfg_align
    _, _run_epsf = _photometry_mode_run_flags(
        _catalog_app_cfg,
        platesolve_dir=platesolve_dir,
    )

    # TODO-8: Build ePSF model after MASTERSTAR (Phase 2B prep)
    if _run_epsf:
        try:
            if draft_id is not None and str(_catalog_app_cfg.database_path or "").strip():
                _ms_for_epsf = ms_fits
                if _ms_for_epsf is None or not Path(_ms_for_epsf).is_file():
                    _ms_try = platesolve_dir / "MASTERSTAR.fits"
                    if _ms_try.is_file():
                        _ms_for_epsf = _ms_try
                _ms_csv_epsf = ms_csv
                if _ms_csv_epsf is None or not Path(_ms_csv_epsf).is_file():
                    _ms_csv_try = platesolve_dir / "masterstars_full_match.csv"
                    if _ms_csv_try.is_file():
                        _ms_csv_epsf = _ms_csv_try
                if (
                    _ms_for_epsf is not None
                    and _ms_csv_epsf is not None
                    and Path(_ms_for_epsf).is_file()
                    and Path(_ms_csv_epsf).is_file()
                ):
                    from psf_photometry import build_epsf_model

                    _db_epsf = VyvarDatabase(Path(str(_catalog_app_cfg.database_path)))
                    try:
                        _epsf_path = build_epsf_model(
                            masterstar_fits_path=Path(_ms_for_epsf),
                            masterstars_csv_path=Path(_ms_csv_epsf),
                            db=_db_epsf,
                            draft_id=int(draft_id),
                            # TODO-PSF-PHASE2: moffat_centroids not yet available at MASTERSTAR
                            # build time - requires per-frame Moffat run first then aggregate
                            # centroids. Implement in next session.
                        )
                        LOGGER.info("[ePSF] Model built: %s", _epsf_path)
                    finally:
                        _db_epsf.conn.close()
        except Exception as _e:  # noqa: BLE001
            # EXC-0424: T1 -- `normalize_gaia_source_id_series` failure `pass`es before deferred CSV write; per-frame... (EXCEPT-BULK-2 2026-07-08)
            LOGGER.warning("[ePSF] build_epsf_model failed (non-fatal): %s", _e)

    def _cat_prog(i: int, tot: int, msg: str) -> None:
        if progress_cb is None:
            return
        progress_cb(
            min(export_base + i, global_total),
            global_total,
            f"detrended_aligned/lights: CSV ({i}/{tot}) - {msg}",
        )

    if use_ram_handoff:
        per_cat = export_per_frame_catalogs(
            frames_root=aligned_root,
            platesolve_dir=platesolve_dir,
            max_catalog_rows=int(max_catalog_rows),
            catalog_match_max_sep_arcsec=float(per_frame_match_sep),
            saturate_level_fraction=float(saturate_level_fraction),
            faintest_mag_limit=faintest_mag_limit,
            dao_threshold_sigma=float(dao_threshold_sigma),
            masterstars_csv=ms_csv,
            masterstar_fits=ms_fits,
            use_master_fast_path=use_master_fast,
            equipment_saturate_adu=equip_sat_adu,
            catalog_local_gaia_only=_cat_loc_only,
            progress_cb=_cat_prog if progress_cb is not None else None,
            aligned_ram=aligned_ram_buffer,
            aligned_target_dir=aligned_root,
            defer_disk_writes=True,
            app_config=_catalog_app_cfg,
            plate_solve_fov_deg=float(_pfov_align),
            master_dark_path=_master_dark_bpm_path,
            draft_id=draft_id,
            equipment_id=id_equipment,
            full_catalog_export=True,
        )
        _prog("detrended_aligned/lights: zapisujem FITS + CSV na disk (davka po praci v RAM)...")
        if not _ram_flushed_before_masterstar:
            for name, hdr, arr in aligned_ram_buffer:
                _target = aligned_root / Path(name)
                _ensure_parent_dirs_for_aligned_fits(_target)
                fits.writeto(
                    _target,
                    _as_fits_float32_image(arr),
                    header=hdr,
                    overwrite=True,
                )
        for pcsv, df in per_cat.get("deferred_csv_writes", []):
            try:
                from gaia_catalog_id import normalize_gaia_source_id_series  # noqa: PLC0415

                if isinstance(df, pd.DataFrame) and "catalog_id" in df.columns:
                    df = df.copy()
                    df["catalog_id"] = normalize_gaia_source_id_series(df["catalog_id"])
            except Exception:  # noqa: BLE001
                pass
            df.to_csv(pcsv, index=False)
        pd.DataFrame(per_cat.get("frames", [])).to_csv(Path(per_cat["index_csv"]), index=False)
        if per_cat.get("deferred_csv_writes"):
            _hybrid_ram = _finalize_hybrid_bkg_fallback_sidecar(
                aligned_root,
                err_background_mode="empirical",
                write_sidecar=True,
                gain=float(_catalog_app_cfg.gain),
                read_noise=float(_catalog_app_cfg.read_noise),
                setup_label=str(aligned_root.name),
            )
            if _hybrid_ram:
                per_cat["hybrid_bkg_fallback"] = _hybrid_ram
    else:
        per_cat = export_per_frame_catalogs(
            frames_root=aligned_root,
            platesolve_dir=platesolve_dir,
            max_catalog_rows=int(max_catalog_rows),
            catalog_match_max_sep_arcsec=float(per_frame_match_sep),
            saturate_level_fraction=float(saturate_level_fraction),
            faintest_mag_limit=faintest_mag_limit,
            dao_threshold_sigma=float(dao_threshold_sigma),
            masterstars_csv=ms_csv,
            masterstar_fits=ms_fits,
            use_master_fast_path=use_master_fast,
            equipment_saturate_adu=equip_sat_adu,
            catalog_local_gaia_only=_cat_loc_only,
            progress_cb=_cat_prog if progress_cb is not None else None,
            app_config=_catalog_app_cfg,
            plate_solve_fov_deg=float(_pfov_align),
            master_dark_path=_master_dark_bpm_path,
            draft_id=draft_id,
            equipment_id=id_equipment,
            full_catalog_export=True,
        )

    _assert_alignment_produced_fits(aligned_root)

    print(f"  Per-frame CSV: {time.time() - _t_csv:.1f}s")
    print(f"CELKOM krok 3 ({obs_group_key or detrended_root.name}): {time.time() - _t_step3_start:.1f}s")

    LOGGER.info(
        "Astrometria dokoncena: zarovnane %s / %s snimok; per-frame CSV: %s; MASTERSTAR: %s; RAM handoff: %s",
        n_aligned,
        len(files),
        int(per_cat.get("written", 0)),
        "ano" if build_masterstar_and_catalogs else "nie",
        "ano" if use_ram_handoff else "nie",
    )

    return {
        "ram_align_handoff_used": bool(use_ram_handoff),
        "detrended_input_root": str(detrended_root),
        "detrended_files_used": len(files),
        "reference_frame": str(ref_fp),
        "reference_star_counts": dict(ref_star_scores),
        "reference_hint_ra_dec_deg": {"ra": hint_ra, "dec": hint_dec} if hint_ra is not None else None,
        "extra_platesolve": extra_platesolve_results,
        "alignment_max_control_points_used": align_cp,
        "alignment_max_stars_cap": int(_align_star_cap),
        "alignment_detection_sigma": float(_align_det_sigma),
        "aligned_root": str(aligned_root),
        "aligned_frames": n_aligned,
        "input_frames": int(len(files)),
        "alignment_report_csv": str(rep_path),
        "rotation_ref_angle_deg": rotation_ref_angle_deg,
        "rotation_flip_frame_indices_1based": rotation_flip_frame_indices_1based,
        "rotation_flip_first_index_1based": rotation_flip_first_index_1based,
        "build_masterstar_and_catalogs": bool(build_masterstar_and_catalogs),
        "masterstar_built": bool(_masterstar_built) if build_masterstar_and_catalogs else False,
        "masterstar_fits": str(ms_fits) if build_masterstar_and_catalogs and ms_fits is not None else "",
        "masterstars_csv": str(ms_csv) if build_masterstar_and_catalogs and ms_csv is not None else "",
        "catalog_match_max_sep_arcsec": float(_catalog_match_sep_eff),
        "saturate_level_fraction": float(saturate_level_fraction),
        "saturate_limit_adu": (cat_info.get("saturate_limit_adu") if build_masterstar_and_catalogs else None),
        "saturate_limit_source": (cat_info.get("saturate_limit_source") if build_masterstar_and_catalogs else None),
        "max_catalog_rows": int(max_catalog_rows),
        "faintest_mag_limit": (
            cat_info.get("faintest_mag_limit") if build_masterstar_and_catalogs else faintest_mag_limit
        ),
        "per_frame_catalog_dir": per_cat.get("per_frame_dir"),
        "per_frame_catalog_index_csv": per_cat.get("index_csv"),
        "per_frame_catalogs_written": per_cat.get("written"),
        "comparison_stars_csv": cat_info.get("comparison_stars_csv", "") if build_masterstar_and_catalogs else "",
        "variable_targets_csv": cat_info.get("variable_targets_csv", "") if build_masterstar_and_catalogs else "",
        "photometry_plan_json": cat_info.get("photometry_plan_json", "") if build_masterstar_and_catalogs else "",
        "n_comparison_stars_requested": (
            cat_info.get("n_comparison_stars_requested") if build_masterstar_and_catalogs else None
        ),
        "comparison_selection": cat_info.get("comparison_selection") if build_masterstar_and_catalogs else None,
        "id_equipment": int(id_equipment) if id_equipment is not None else None,
        "equipment_saturate_adu_resolved": equip_sat_adu,
        "catalog_local_gaia_only": _cat_loc_only,
        "observation_group_key": obs_group_key,
        "scanning_id": (scanning_id if scanning_id > 0 else None),
    }
