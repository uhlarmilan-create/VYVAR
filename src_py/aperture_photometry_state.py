"""Moved from photometry_core.py (CONSOLIDATE-01E4). Facade re-exports these names."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import logging
import math
import time
from astropy.io import fits as astrofits
import numpy as np
import pandas as pd
from catalog_match_trust import normalize_catalog_match_mode
from config import AppConfig
from infolog import log_event
from photometry_shared import _normalize_gaia_id, _resolve_plate_scale_arcsec_per_px
from proc_frame_store import ProcFrameStore, is_masterstar_proc_name

from aperture_photometry import (
    _ColorTermGroupFit,
    _AperturePhotometryState,
    _build_csv_lookup,
    _compute_group_color_term_fit,
    _draft_dir_from_aperture_paths,
    _ensure_group_comp_pool_csv,
    _aperture_cache_columns,
    _aperture_coerce_skip_photometry,
    _require_comparison_stars_per_target_schema,
    _resolve_aperture_equipment_id,
    _sat_limit_peak_adu,
    compute_optimal_apertures,
    measure_fwhm_from_masterstar,
    read_flux_from_csv,
    resolve_apply_color_term,
)
from photometry_lightcurve import (
    _build_aperture_resolved_facts,
    apply_per_frame_saturation_to_active_targets,
)

from photometry_core import (
    LOGGER,
    _GAIA_ID_DTYPE,
)


def _aperture_prepare_shared_state(
    output_dir: Path,
    lc_dir: Path,
    masterstar_fits_path: Path,
    comparison_stars_csv: Path,
    per_frame_csv_dir: Path,
    progress_cb: Any,
    *,
    active_targets_csv: Path,
    detrended_aligned_dir: Path,
    fwhm_px: float,
    annulus_inner_fwhm: float = 4.0,
    annulus_outer_fwhm: float = 6.0,
    aperture_fwhm_factor: float | None = None,
    sat_limit_adu: float | None = None,
    force_aperture_px: float | None = None,
    cfg: AppConfig | None = None,
    db: Any | None = None,
    draft_id: int | None = None,
    proc_frame_store: ProcFrameStore | None = None,
) -> _AperturePhotometryState:
    def _p2(msg: str) -> None:
        if progress_cb is None:
            return
        try:
            progress_cb(str(msg))
        except UnicodeEncodeError:
            progress_cb(str(msg).encode("ascii", "backslashreplace").decode("ascii"))

    _aligned_dir_2a = Path(detrended_aligned_dir)
    output_dir = Path(output_dir)
    lc_dir = output_dir / "lightcurves"
    lc_dir.mkdir(parents=True, exist_ok=True)

    _cfg = cfg or AppConfig()
    # Obs_group / filter name for decisions (e.g. color term). Prefer FITS header, fallback to directory name.
    obs_group = str(Path(per_frame_csv_dir).name)
    if aperture_fwhm_factor is not None:
        try:
            _apt_fw = float(aperture_fwhm_factor)
            if not math.isfinite(_apt_fw) or _apt_fw <= 0:
                _apt_fw = float(_cfg.aperture_fwhm_factor)
            else:
                _apt_fw = max(0.25, min(6.0, _apt_fw))
        except (TypeError, ValueError):
            _apt_fw = float(_cfg.aperture_fwhm_factor)
    else:
        _apt_fw = float(_cfg.aperture_fwhm_factor)

    # Load inputs (Gaia ID as string - float64 loses digits)
    if not Path(active_targets_csv).is_file():
        raise FileNotFoundError(f"active_targets_csv not found: {active_targets_csv}")
    if not Path(comparison_stars_csv).is_file():
        raise FileNotFoundError(f"comparison_stars_csv not found: {comparison_stars_csv}")
    at_df = pd.read_csv(active_targets_csv, low_memory=False, dtype=_GAIA_ID_DTYPE)
    # OUTPUT-REVIEW-2026-10 step 5: aperture photometry starts from the comparison-star selection rows of
    # the comparison table (admitted rows; the reserve below), never from the used sets of an earlier run.
    from comp_table import admitted as _comp_admitted, load_comp_table, reserve as _comp_reserve  # noqa: PLC0415

    _comp_table = load_comp_table(Path(comparison_stars_csv))
    comp_df = _comp_admitted(_comp_table)

    # Normalize catalog_id
    for df in (at_df, comp_df):
        for col in ("catalog_id", "name"):
            if col in df.columns:
                df[col] = df[col].apply(_normalize_gaia_id)

    at_df["skip_photometry"] = _aperture_coerce_skip_photometry(at_df)
    _require_comparison_stars_per_target_schema(comp_df, Path(comparison_stars_csv))

    # _aperture_load_star_list (inline): chip dims, comp index, BP-RP map, CSV cache - through target loop below.

    # Open MASTERSTAR.fits once - reuse header + data throughout aperture photometry.
    _ms_header: Any = None
    _ms_data: np.ndarray | None = None
    _ms_path = Path(masterstar_fits_path)
    if _ms_path.is_file():
        try:
            with astrofits.open(_ms_path, memmap=False) as _hdul_ms:
                _ms_header = _hdul_ms[0].header.copy()
                if _hdul_ms[0].data is not None:
                    _ms_data = np.asarray(_hdul_ms[0].data, dtype=np.float64)
            LOGGER.debug("[APERTURE] MASTERSTAR.fits opened once (header + data cached)")
            logging.info(
                "[PERF-2] MASTERSTAR.fits loaded once: data shape=%s, header keys=%d",
                _ms_data.shape if _ms_data is not None else None,
                len(_ms_header) if _ms_header is not None else 0,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("[APERTURE] MASTERSTAR.fits open failed: %s", exc)

    # Chip dims for edge checks: prefer MASTERSTAR.fits NAXIS (full detector); CSV max(x,y) underestimates.
    chip_fw: int | None = None
    chip_fh: int | None = None
    try:
        if _ms_header is not None:
            _naxis1 = int(_ms_header.get("NAXIS1", 0) or 0)
            _naxis2 = int(_ms_header.get("NAXIS2", 0) or 0)
        else:
            _naxis1 = _naxis2 = 0
        if _naxis1 > 0 and _naxis2 > 0:
            chip_fw = _naxis1
            chip_fh = _naxis2
            logging.info("[PHOT] chip dims from MASTERSTAR: %d x %d px", int(chip_fw), int(chip_fh))
    except Exception:  # noqa: BLE001
        chip_fw, chip_fh = None, None
        logging.warning("[PHOT] chip dims: MASTERSTAR read failed, fallback to CSV max(x/y)")

    # Fallback: estimate from star positions in CSV (any axis still unknown after MASTERSTAR).
    if chip_fw is None or chip_fh is None:
        try:
            xm = float(
                pd.to_numeric(
                    pd.concat([at_df.get("x", pd.Series(dtype=float)), comp_df.get("x", pd.Series(dtype=float))]),
                    errors="coerce",
                ).max()
            )
            ym = float(
                pd.to_numeric(
                    pd.concat([at_df.get("y", pd.Series(dtype=float)), comp_df.get("y", pd.Series(dtype=float))]),
                    errors="coerce",
                ).max()
            )
            if chip_fw is None and math.isfinite(xm) and xm > 0:
                chip_fw = int(math.ceil(xm)) + 2
            if chip_fh is None and math.isfinite(ym) and ym > 0:
                chip_fh = int(math.ceil(ym)) + 2
        except Exception as exc:  # noqa: BLE001
            # EXC-0165: T4 -- Non-numeric bp_rp on one masterstars row skipped when building target_bp_rp_by_cid (EXCEPT-BULK-2 2026-07-08)
            logging.error('[EXC-0164] Chip height/width inference from target/comp xy max fails - chip margin filter may use ...: %s', exc)
            pass

    if "x" not in comp_df.columns or "y" not in comp_df.columns:
        raise ValueError("comparison_stars_per_target.csv must contain columns x, y for aperture photometry")

    # Pre-index comp_df by target catalog_id - once for the whole loop (O(1) lookup per target).
    _id_col_comp = "target_catalog_id" if "target_catalog_id" in comp_df.columns else "catalog_id"
    _comp_index: dict[str, pd.DataFrame] = {}
    if comp_df is not None and not comp_df.empty and _id_col_comp in comp_df.columns:
        for _tid, _grp in comp_df.groupby(comp_df[_id_col_comp].apply(_normalize_gaia_id), sort=False, dropna=False):
            _tid_s = str(_tid).strip()
            if _tid_s:
                _comp_index[_tid_s] = _grp.reset_index(drop=True)
    else:
        _comp_index = {}

    # COMP-SMART-2026-10: ranked reserve comparison candidates per target (comparison-star selection)
    _reserve_index: dict[str, pd.DataFrame] = {}
    try:
        _res_df = _comp_reserve(_comp_table)
        if not _res_df.empty:
            if "target_catalog_id" in _res_df.columns:
                for _tid, _grp in _res_df.groupby(_res_df["target_catalog_id"].apply(_normalize_gaia_id), sort=False):
                    if str(_tid).strip():
                        _reserve_index[str(_tid).strip()] = _grp.reset_index(drop=True)
    except Exception as exc:  # noqa: BLE001
        logging.warning("[COMP-SMART] reserve not read: %s", exc)
        _reserve_index = {}

    # masterstars_full_match.csv - BP-RP map + comp pool for catalog_only aperture photometry.
    target_bp_rp_by_cid: dict[str, float] = {}
    masterstars_df = pd.DataFrame()
    try:
        ms_full = Path(masterstar_fits_path).resolve().parent / "masterstars_full_match.csv"
        if ms_full.is_file():
            masterstars_df = pd.read_csv(ms_full, low_memory=False, dtype=_GAIA_ID_DTYPE)
            masterstars_df = masterstars_df.copy()
            for _id_col in ("catalog_id", "name"):
                if _id_col in masterstars_df.columns:
                    masterstars_df[_id_col] = masterstars_df[_id_col].apply(_normalize_gaia_id)
            if "bp_rp" in masterstars_df.columns:
                masterstars_df["bp_rp"] = pd.to_numeric(masterstars_df["bp_rp"], errors="coerce")
                for _, r in masterstars_df.iterrows():
                    cid0 = str(r.get("catalog_id") or "").strip()
                    if not cid0:
                        continue
                    v0 = r.get("bp_rp")
                    try:
                        vv = float(v0)
                    except Exception:  # noqa: BLE001
                        continue
                    if math.isfinite(vv):
                        target_bp_rp_by_cid[cid0] = float(vv)
    except Exception:  # noqa: BLE001
        target_bp_rp_by_cid = {}
        masterstars_df = pd.DataFrame()

    # Find per-frame CSVs (FITS is not used)
    if proc_frame_store is not None and len(proc_frame_store) > 0:
        csv_files = sorted(Path(k) for k in proc_frame_store.keys())
    else:
        csv_files = sorted(Path(per_frame_csv_dir).glob("proc_*.csv"))
    _ms_epoch_drop = [p for p in csv_files if is_masterstar_proc_name(p)]
    if _ms_epoch_drop:
        logging.warning(
            "[MASTERSTAR-EPOCH] aperture_photometry filtered %d masterstar proc from epoch set "
            "(canonical list_proc_csvs / ProcFrameStore filter was bypassed): %s",
            len(_ms_epoch_drop),
            ", ".join(p.name for p in _ms_epoch_drop[:3]),
        )
        csv_files = [p for p in csv_files if not is_masterstar_proc_name(p)]
    # ALIGN-VERIFY-2026-10: alignment residuals are measured and gated at alignment time
    # (vyvar_alignment_frame); an unverified frame has no proc CSV and is not measured here.
    # CSV only - no FITS (flux is read from dao_flux in the CSV)
    n_frames = len(csv_files)
    _n_total = int(len(at_df))
    logging.info("[APERTURE] %d targets (DAO+Gaia matched)", _n_total)
    _p2(f"Aperture photometry: {_n_total} targets, {n_frames} frames - loading per-frame catalogs...")

    # Load the CSV cache once for all of aperture photometry (otherwise read_flux_from_csv runs 82x per target).
    if proc_frame_store is not None and len(proc_frame_store) > 0:
        _aperture_csv_cache = proc_frame_store
        logging.info(
            "[PERF-5] run_aperture_photometry: using ProcFrameStore (%d frames, 0 disk reads)",
            len(proc_frame_store),
        )
        _p2(f"Aperture photometry: {len(proc_frame_store)} per-frame catalogs - FWHM and apertures...")
    else:
        logging.info("[APERTURE] Loading CSV cache...")
        _t_cache = time.time()
        _aperture_csv_cache: dict[str, pd.DataFrame] = {}
        _needed_cols_2a = _aperture_cache_columns()
        for _csv_path in csv_files:
            try:
                from proc_frame_store import read_proc_csv  # noqa: PLC0415

                _hdr = read_proc_csv(_csv_path, nrows=0)
                _cols = [c for c in _needed_cols_2a if c in _hdr.columns]
                if not _cols:
                    continue
                # Gaia ID must be str - float64 loses digits
                _dtype_2a: dict[str, type] = {}
                if "catalog_id" in _cols:
                    _dtype_2a["catalog_id"] = str
                if "name" in _cols:
                    _dtype_2a["name"] = str
                _kw: dict[str, Any] = {"usecols": _cols, "low_memory": False}
                if _dtype_2a:
                    _kw["dtype"] = _dtype_2a
                _aperture_csv_cache[str(_csv_path)] = read_proc_csv(_csv_path, **_kw)
            except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
                from except_fix_counters import get_except_fix_counters

                get_except_fix_counters().aperture_csv_cache_skip += 1
                logging.error("[APERTURE] proc CSV cache skip %s: %s", _csv_path, exc)
                continue
        logging.info(
            f"[APERTURE] CSV cache: {len(_aperture_csv_cache)} files "
            f"({time.time() - _t_cache:.1f}s)"
        )
        _p2(f"Aperture photometry: {len(_aperture_csv_cache)} cached catalogs - FWHM and apertures...")

    # Lookup (id_map + xy_df) once per frame - otherwise _build_csv_lookup runs 82x per target.
    _aperture_lookup_cache: dict[str, tuple[dict[str, pd.Series], pd.DataFrame]] = {}
    for _cp in csv_files:
        _key = str(_cp)
        _df_lu = _aperture_csv_cache.get(_key)
        if _df_lu is None or _df_lu.empty:
            continue
        _id_col_lu = "catalog_id" if "catalog_id" in _df_lu.columns else "name"
        _aperture_lookup_cache[_key] = _build_csv_lookup(_df_lu, _id_col_lu)

    # Time + airmass (+ flip flag from alignment_report) from the first valid row of each per-frame CSV
    # (by FITS stem).
    frame_time_lookup: dict[str, dict[str, float]] = {}
    for csv_path in csv_files:
        stem = csv_path.stem
        _csv_tmp = _aperture_csv_cache.get(str(csv_path))
        if _csv_tmp is None or _csv_tmp.empty:
            continue
        _cmm_frame = ""
        if "catalog_match_mode" in _csv_tmp.columns:
            _cmm_s = _csv_tmp["catalog_match_mode"].dropna()
            if len(_cmm_s) > 0:
                _cmm_frame = normalize_catalog_match_mode(str(_cmm_s.iloc[0]))
        try:
            for col_bjd, col_hjd, col_jd in (("bjd_tdb_mid", "hjd_mid", "jd_mid"),):
                if not all(c in _csv_tmp.columns for c in (col_bjd, col_hjd, col_jd)):
                    continue
                vals = _csv_tmp[[col_bjd, col_hjd, col_jd]].dropna()
                if len(vals) == 0:
                    continue
                am_val = float("nan")
                for am_col in ("airmass", "AIRMASS", "air_mass"):
                    if am_col not in _csv_tmp.columns:
                        continue
                    am_series = pd.to_numeric(_csv_tmp[am_col], errors="coerce").dropna()
                    if len(am_series) > 0:
                        am_val = float(am_series.iloc[0])
                    break
                frame_time_lookup[stem] = {
                    "bjd": float(vals[col_bjd].iloc[0]),
                    "hjd": float(vals[col_hjd].iloc[0]),
                    "jd": float(vals[col_jd].iloc[0]),
                    "airmass": am_val,
                }
                if _cmm_frame:
                    frame_time_lookup[stem]["catalog_match_mode"] = _cmm_frame
                break
            if stem not in frame_time_lookup and _cmm_frame:
                frame_time_lookup[stem] = {
                    "bjd": float("nan"),
                    "hjd": float("nan"),
                    "jd": float("nan"),
                    "airmass": float("nan"),
                    "catalog_match_mode": _cmm_frame,
                }
        except Exception as exc:  # noqa: BLE001
            logging.error('[EXC-0167] catalog_match_mode not stored in frame_time_lookup for one frame stem: %s', exc)
            pass

    # Propagate meridian-flip / rotation-change flag from alignment_report.csv when present.
    try:
        align_rep = Path(masterstar_fits_path).resolve().parent / "alignment_report.csv"
        if align_rep.is_file():
            rep = pd.read_csv(align_rep, low_memory=False)
            if "file" in rep.columns:
                for _, r in rep.iterrows():
                    fn = str(r.get("file", "")).strip()
                    if not fn:
                        continue
                    stem = Path(fn).stem
                    if stem not in frame_time_lookup:
                        continue
                    if "aligned" in rep.columns:
                        try:
                            _al = r.get("aligned", True)
                            if isinstance(_al, bool):
                                _aligned = bool(_al)
                            else:
                                s = str(_al).strip().lower()
                                _aligned = s not in ("false", "0", "no", "n")
                            frame_time_lookup[stem]["alignment_failed"] = not _aligned
                        except Exception:  # noqa: BLE001
                            frame_time_lookup[stem]["alignment_failed"] = False
    except Exception as exc:  # noqa: BLE001
        logging.error('[EXC-0168] alignment_report.csv alignment_failed/reason not propagated into frame_time_lookup: %s', exc)
        pass
    # ALIGN-VERIFY-2026-10: frames whose registration was not verified have no proc CSV; count
    # them so the light-curve trust can still report the lost epochs.
    _n_align_skipped = 0
    try:
        _rep_p = Path(masterstar_fits_path).resolve().parent / "alignment_report.csv"
        if _rep_p.is_file():
            _rep = pd.read_csv(_rep_p, low_memory=False)
            if "aligned" in _rep.columns:
                _al = _rep["aligned"].astype(str).str.strip().str.lower()
                _n_align_skipped = int(_al.isin(("false", "0", "no", "n")).sum())
        if _n_align_skipped:
            logging.warning("[ALIGN-VERIFY] %d frame(s) not aligned (registration not verified) - not measured",
                            _n_align_skipped)
    except Exception as exc:  # noqa: BLE001
        logging.warning("[ALIGN-VERIFY] skipped-frame count not read: %s", exc)

    # SAT-TARGET-2026-10 / TARGET-RULES-2026-10: a target linear on MASTERSTAR that is saturated
    # in most of its frames is not measured. Target-selection skips are final (never revived here).
    # Peak test = 0.90 x SAT-DIAG level of the draft (INV-SAT-LIMIT authority).
    _pfs_min = float(getattr(_cfg, "per_frame_sat_min_clean_frac", 0.5))
    _peak_test_adu: float | None = None
    _peak_test_src = ""
    try:
        from sat_diag import draft_peak_test_adu  # noqa: PLC0415

        _peak_test_adu, _peak_test_src = draft_peak_test_adu(Path(masterstar_fits_path).resolve().parent)
    except Exception as exc:  # noqa: BLE001
        logging.error("[EXC-PFS] INV-SAT-LIMIT peak-test resolve failed: %s", exc)
        _peak_test_adu = None
        _peak_test_src = ""
    _per_frame_sat_meta = apply_per_frame_saturation_to_active_targets(
        at_df,
        csv_files=csv_files,
        csv_cache=_aperture_csv_cache,
        min_clean_frac=_pfs_min,
        peak_test_adu=_peak_test_adu,
        peak_test_source=_peak_test_src,
    )
    try:
        at_df.to_csv(active_targets_csv, index=False)
    except Exception as exc:  # noqa: BLE001
        logging.error("[EXC-PFS] active_targets.csv rewrite after per-frame sat failed: %s", exc)
    logging.info(
        "[PER-FRAME-SAT] skipped=%s (min_clean_frac=%.3f peak_test=%.1f ADU src=%s)",
        _per_frame_sat_meta.get("per_frame_sat_n_skipped"),
        _pfs_min,
        float(_per_frame_sat_meta.get("per_frame_sat_peak_test_adu") or float("nan")),
        str(_per_frame_sat_meta.get("per_frame_sat_peak_test_source") or ""),
    )

    # Step 1: global fixed aperture - all stars (target + comp), factor x FWHM
    # Targets with skip_photometry (saturated) are excluded from the aperture / FWHM computation.
    _at_cols = [c for c in ("catalog_id", "x", "y", "mag") if c in at_df.columns]
    _comp_cols = [c for c in ("catalog_id", "x", "y", "mag") if c in comp_df.columns]
    _at_use = at_df.loc[~at_df["skip_photometry"], _at_cols].copy() if _at_cols else at_df.iloc[0:0].copy()
    _at_part = _at_use.copy()
    _comp_part = comp_df[_comp_cols].drop_duplicates("catalog_id").copy()
    if "mag" not in _at_part.columns:
        _at_part["mag"] = float("nan")
    if "mag" not in _comp_part.columns:
        _comp_part["mag"] = float("nan")
    all_stars = pd.concat(
        [
            _at_part[["catalog_id", "x", "y", "mag"]],
            _comp_part[["catalog_id", "x", "y", "mag"]],
        ],
        ignore_index=True,
    ).drop_duplicates("catalog_id")

    # Priority: 1. VY_FWHM_GAUSS (2D fit in header), 2. VY_FWHM (DAO), 3. fit fallback
    _fwhm_from_header: float | None = None
    try:
        hdr = _ms_header
        if hdr is None and _ms_path.is_file():
            with astrofits.open(_ms_path, memmap=False) as _hdul:
                hdr = _hdul[0].header
        if hdr is not None:
            try:
                _obsg = hdr.get("VY_OBSG", None) or hdr.get("OBSG", None) or hdr.get("FILTER", None)
                if _obsg is not None:
                    _s = str(_obsg).strip()
                    if _s:
                        obs_group = _s
            except Exception as exc:  # noqa: BLE001
                logging.error('[EXC-0169] VY_OBSG/OBSG/FILTER header tag not parsed - obs_group label missing from frame metadata: %s', exc)
                pass
            vy_fwhm_gauss = hdr.get("VY_FWHM_GAUSS", None)
            vy_fwhm_dao = hdr.get("VY_FWHM", None)
            if vy_fwhm_gauss is not None:
                _fvg = float(vy_fwhm_gauss)
                if 0.5 < _fvg < 30.0:
                    _fwhm_from_header = _fvg
                    logging.info(
                        f"[APERTURE] FWHM from VY_FWHM_GAUSS (2D fit): {_fwhm_from_header:.3f} px"
                    )
            if _fwhm_from_header is None and vy_fwhm_dao is not None:
                _fvd = float(vy_fwhm_dao)
                if 0.5 < _fvd < 30.0:
                    _fwhm_from_header = _fvd
                    logging.info(
                        f"[APERTURE] FWHM from VY_FWHM (DAO): {_fwhm_from_header:.3f} px"
                    )
    except Exception as _e:  # noqa: BLE001
        logging.error('[EXC-0170] MASTERSTAR header FWHM read throws - measured FWHM from stars used instead (logged warn...: %s', _e)
        logging.warning(f"[APERTURE] Cannot read FWHM from header: {_e}")

    if _fwhm_from_header is not None:
        fwhm_px = _fwhm_from_header
    else:
        _fallback_hint = float(fwhm_px) if math.isfinite(fwhm_px) and fwhm_px > 0 else 3.5
        fwhm_px = measure_fwhm_from_masterstar(
            Path(masterstar_fits_path),
            all_stars,
            dao_fwhm_hint=_fallback_hint,
            ms_data=_ms_data,
        )
        logging.info(f"[APERTURE] FWHM from Gaussian fit: {fwhm_px:.3f} px")

    _p2(f"Aperture photometry: FWHM={float(fwhm_px):.3f} px - field map and light curves...")

    # GAIN-ONE-2026-10: one gain / read-noise authority for every stage (rig_photometry)
    from param_resolver import resolve_gain, resolve_read_noise  # noqa: PLC0415
    from rig_photometry import photometric_gain_read_noise  # noqa: PLC0415

    _equipment_id = _resolve_aperture_equipment_id(
        db,
        draft_id=draft_id,
        output_dir=output_dir,
        masterstar_fits_path=Path(masterstar_fits_path),
    )
    _pg = photometric_gain_read_noise(_ms_header, _cfg, _equipment_id)
    _gain_res = resolve_gain(_ms_header, db=db, equipment_id=_equipment_id, cfg=_cfg)  # provenance record
    _rn_res = resolve_read_noise(_ms_header, db=db, equipment_id=_equipment_id, cfg=_cfg)
    _gain_phot, _rn_phot = float(_pg.gain), float(_pg.read_noise)
    logging.info(
        "[APERTURE] Photometric errors: gain=%.3f e-/ADU_container (native %.3f src=%s / container scale %.0f), "
        "RN=%.1f e- (source: %s)",
        _gain_phot, _pg.gain_native, _pg.gain_source, _pg.container_scale, _rn_phot, _pg.read_noise_source,
    )

    _draft_dir_ee = _draft_dir_from_aperture_paths(output_dir, Path(masterstar_fits_path))
    _snr_ap_table = None

    _variable_target_cids = frozenset(
        c
        for _, row in at_df.iterrows()
        for c in [_normalize_gaia_id(row.get("catalog_id", ""))]
        if c
    )

    _ap_pol: dict[str, Any] | None = None
    if force_aperture_px is not None and force_aperture_px > 0:
        # Fixed aperture for all stars - debug/calibration
        apertures_px = {
            _normalize_gaia_id(row.get("catalog_id", "")): float(force_aperture_px)
            for _, row in all_stars.iterrows()
            if _normalize_gaia_id(row.get("catalog_id", ""))
        }
        logging.info(
            f"[APERTURE] FORCE aperture: {force_aperture_px:.2f}px for all stars"
        )
    else:
        # APERTURE-01: one r for every star. SNR mag-bin table is diagnostic only.
        from aperture_policy import (  # noqa: PLC0415
            fwhm_for_radius,
            load_qc_fwhm_map,
            policy_record,
            resolve_aperture_geometry,
        )
        _qc_csv_p2a = None
        _search_roots = [Path(_draft_dir_ee), Path(output_dir), Path(masterstar_fits_path)]
        for _root in _search_roots:
            if _qc_csv_p2a is not None:
                break
            for _parent in [_root, *_root.parents]:
                for _cand in (
                    _parent / "calibrated" / "lights" / "qc_metrics.csv",
                    _parent / "processed" / "lights" / "qc_metrics.csv",
                ):
                    if _cand.is_file():
                        _qc_csv_p2a = _cand
                        break
                if _qc_csv_p2a is not None:
                    break
        _, _night_p2a = load_qc_fwhm_map(_qc_csv_p2a)
        # NIGHT-FWHM-2026-10: qc_metrics.csv is draft-level and mixes the filters
        # (524: g 5.31, r 4.91, mixed 5.14 px). The night median is taken over THIS
        # group's per-frame CSVs (same helper as comparison-star selection); QC value only as fallback.
        from photometry_comp import night_median_fwhm_px  # noqa: PLC0415

        _qc_night_p2a = _night_p2a
        _night_grp, _night_src = night_median_fwhm_px(
            csv_files, float(_night_p2a) if _night_p2a is not None else float("nan")
        )
        if math.isfinite(float(_night_grp)) and float(_night_grp) > 0:
            _night_p2a = float(_night_grp)
        logging.info(
            "[APERTURE] NIGHT-FWHM group median %s px (%s); draft qc_metrics %s px",
            f"{_night_p2a:.3f}" if _night_p2a is not None else "n/a",
            _night_src,
            f"{_qc_night_p2a:.3f}" if _qc_night_p2a is not None else "n/a",
        )
        _fw_p2a = fwhm_for_radius(
            fwhm_frame_px=_night_p2a,
            fwhm_night_median_px=_night_p2a,
        )
        if _fw_p2a is None:
            from aperture_policy import fwhm_from_header_vy_fwhm  # noqa: PLC0415

            _fw_p2a = fwhm_from_header_vy_fwhm(_ms_header)
        if _fw_p2a is None:
            logging.warning(
                "[APERTURE-01] QC fwhm_px missing (csv=%s); refusing VY_FWHM_GAUSS fallback",
                _qc_csv_p2a,
            )
            _fw_p2a = 5.0
        apertures_px = compute_optimal_apertures(
            all_stars,
            float(_fw_p2a),
            aperture_fwhm_factor=_apt_fw,
        )
        _r_ap_p2a, _r_in_p2a, _r_out_p2a = resolve_aperture_geometry(
            f=float(_apt_fw),
            fwhm_px=float(_fw_p2a),
            annulus_inner_fwhm=float(annulus_inner_fwhm),
            annulus_outer_fwhm=float(annulus_outer_fwhm),
        )
        _ap_pol = policy_record(
            f=float(_apt_fw),
            fwhm_frame_px=None,
            fwhm_night_median_px=_night_p2a,
            r_ap=_r_ap_p2a,
            r_in=_r_in_p2a,
            r_out=_r_out_p2a,
            fwhm_used_px=_fw_p2a,
        )
        logging.info(
            "[APERTURE] APERTURE-01 per_target f=%.3f FWHM_used=%.3f r_ap=%.3f (n_stars=%d)",
            float(_apt_fw),
            float(_fw_p2a),
            float(_r_ap_p2a),
            len(apertures_px),
        )

    star_xy: dict[str, tuple[float, float]] = {}
    for _, row in all_stars.iterrows():
        cid = _normalize_gaia_id(row.get("catalog_id", ""))
        if not cid:
            continue
        try:
            star_xy[cid] = (float(row["x"]), float(row["y"]))
        except (KeyError, TypeError, ValueError):
            pass

    sat_limit_resolved = sat_limit_adu if sat_limit_adu is not None else _sat_limit_peak_adu()

    # Field map of the whole setup (FIELD-MAP-2026-10: JPEG with RA/Dec grid, scale, orientation)
    from field_map import FIELD_MAP_NAME, save_field_map  # noqa: PLC0415

    field_map_path = output_dir / FIELD_MAP_NAME
    try:
        _fm_data = _ms_data if _ms_data is not None else astrofits.getdata(Path(masterstar_fits_path))
        _fm_hdr = _ms_header if _ms_header is not None else astrofits.getheader(Path(masterstar_fits_path))
        save_field_map(field_map_path, _fm_data, _fm_hdr, at_df, comp_df)
    except (OSError, ValueError, TypeError) as _fm_exc:
        logging.warning("[APERTURE] field map not drawn: %s", _fm_exc)

    # PERF-8: collect all star IDs needed across all targets (union targets + comps).
    _all_lc_ids: set[str] = set()
    for _, _trow in at_df.iterrows():
        _tcid = _normalize_gaia_id(_trow.get("catalog_id", ""))
        if _tcid:
            _all_lc_ids.add(_tcid)
        _tcomps = _comp_index.get(_tcid, pd.DataFrame())
        if not _tcomps.empty and "catalog_id" in _tcomps.columns:
            for _ccid in _tcomps["catalog_id"].astype(str):
                _n = _normalize_gaia_id(_ccid)
                if _n:
                    _all_lc_ids.add(_n)
        # COMP-SMART-2026-10: reserve candidates may replace a suspect comp -> same flux matrix
        _tres = _reserve_index.get(_tcid, pd.DataFrame())
        if not _tres.empty and "catalog_id" in _tres.columns:
            for _ccid in _tres["catalog_id"].astype(str):
                _n = _normalize_gaia_id(_ccid)
                if _n:
                    _all_lc_ids.add(_n)
    _all_lc_ids_list = sorted(_all_lc_ids)
    logging.info(
        "[PERF-8] Flux matrix: %d unique star IDs across %d targets",
        len(_all_lc_ids_list),
        len(at_df),
    )

    # PERF-8: one read_flux_from_csv pass per frame for all LC stars (not per target).
    _flux_matrix_rows: list[pd.DataFrame] = []
    _t_flux_matrix = time.perf_counter()
    for _csv_path in csv_files:
        _cached_df = _aperture_csv_cache.get(str(_csv_path))
        _lookup_row = _aperture_lookup_cache.get(str(_csv_path))
        if _cached_df is None:
            continue
        _ft = frame_time_lookup.get(_csv_path.stem)
        _df_all = read_flux_from_csv(
            _csv_path,
            _all_lc_ids_list,
            apertures_px,
            sat_limit_adu=sat_limit_resolved,
            star_xy=star_xy,
            xy_tol_px=18.0,
            frame_times=_ft,
            csv_df=_cached_df,
            lookup=_lookup_row,
            gain=float(_gain_phot),
            read_noise=float(_rn_phot),
            variable_target_catalog_ids=_variable_target_cids,
        )
        if not _df_all.empty:
            _flux_matrix_rows.append(_df_all)
    _flux_matrix: pd.DataFrame = pd.DataFrame()
    if _flux_matrix_rows:
        _flux_matrix = pd.concat(_flux_matrix_rows, ignore_index=True)
        logging.info(
            "[PERF-8] Flux matrix built: %d rows (%d stars x %d frames) in %.2fs",
            len(_flux_matrix),
            len(_all_lc_ids_list),
            len(csv_files),
            time.perf_counter() - _t_flux_matrix,
        )
    else:
        logging.warning("[PERF-8] Flux matrix empty - per-target per-frame fallback")
    logging.info(
        "[PERF-8] Flux matrix build time: included in photometry timing"
    )

    if not _flux_matrix.empty and (chip_fw is None or chip_fh is None):
        if "x" in _flux_matrix.columns and "y" in _flux_matrix.columns:
            try:
                _xm = float(pd.to_numeric(_flux_matrix["x"], errors="coerce").max())
                _ym = float(pd.to_numeric(_flux_matrix["y"], errors="coerce").max())
            except Exception:  # noqa: BLE001
                _xm, _ym = float("nan"), float("nan")
            if chip_fw is None and math.isfinite(_xm) and _xm > 0:
                chip_fw = int(math.ceil(_xm)) + 2
            if chip_fh is None and math.isfinite(_ym) and _ym > 0:
                chip_fh = int(math.ceil(_ym)) + 2

    _nt = int(len(at_df))
    _plate_scale_arcsec = _resolve_plate_scale_arcsec_per_px(
        _cfg, _ms_path, ms_header=_ms_header
    )
    _gaia_db_path = str(_cfg.gaia_db_path or "").strip() or None
    if _plate_scale_arcsec is not None:
        logging.info(
            "[APERTURE] GS11 context: plate_scale=%.4f arcsec/px gaia_db=%s",
            float(_plate_scale_arcsec),
            "set" if _gaia_db_path else "none",
        )
    else:
        logging.warning(
            "[APERTURE] GS11 context: plate_scale unknown (derive-or-None) gaia_db=%s",
            "set" if _gaia_db_path else "none",
        )

    # Observer site resolved ONCE per draft (param_resolver): draft ID_LOCATION ->
    # header SITELAT -> flagged config. Threaded into BJD/HJD recompute + meta so
    # aperture photometry no longer silently reads cfg.observer_* (config-drift trap).
    from param_resolver import resolve_site as _resolve_site  # noqa: PLC0415

    _site = _resolve_site(_ms_header, db=db, draft_id=draft_id, cfg=_cfg)
    if not _site.ok or _site.lat is None or _site.lon is None:
        # CODE-REVIEW step 7B: no silent default site (BJD, airmass and extinction need it).
        raise ValueError(
            "No observing site for this draft: set the draft's site (import page / --location-id) "
            "or a default site (IS_DEFAULT) in Database Explorer -> LOCATION."
        )
    _site_loc_id: int | None = None
    if db is not None and draft_id is not None:
        try:
            if hasattr(db, "get_draft_location_id"):
                _site_loc_id = db.get_draft_location_id(int(draft_id))
        except Exception:  # noqa: BLE001
            _site_loc_id = None
    logging.info(
        "[APERTURE] Observer site: source=%s lat=%s lon=%s alt=%s ok=%s",
        _site.source,
        f"{_site.lat:.4f}" if _site.lat is not None else "None",
        f"{_site.lon:.4f}" if _site.lon is not None else "None",
        f"{_site.elev:.0f}" if _site.elev is not None else "None",
        _site.ok,
    )

    from k2_extinction import resolve_k2_bprp_value  # noqa: PLC0415

    _k2_bprp, _k2_src_enum = resolve_k2_bprp_value(_cfg, obs_group)
    if _k2_src_enum.value != "none" and math.isfinite(float(_k2_bprp)):
        log_event(f"[K2] obs_group {obs_group}: k2={float(_k2_bprp):.6f} source={_k2_src_enum.value}")

    _apply_ct = resolve_apply_color_term(_cfg, obs_group)
    _group_ct: _ColorTermGroupFit | None = None
    if _apply_ct:
        _pool_csv = _ensure_group_comp_pool_csv(
            platesolve_dir=Path(masterstar_fits_path).resolve().parent,
            masterstar_fits=Path(masterstar_fits_path),
            masterstars_csv=Path(comparison_stars_csv).resolve().parent / "masterstars_full_match.csv",
            cfg=_cfg,
            draft_id=draft_id,
        )
        _group_ct = _compute_group_color_term_fit(
            comparison_stars_csv=_pool_csv,
            flux_matrix=_flux_matrix,
            csv_files=csv_files,
            obs_group=obs_group,
            cfg=_cfg,
            k2_value=float(_k2_bprp),
            k2_source=_k2_src_enum,
        )
        if _group_ct is not None:
            log_event(f"[COLOR TERM] group pool fit ({obs_group}): {_group_ct.gate_reason}")
        else:
            logging.warning("[COLOR TERM] group pool fit unavailable for %s", obs_group)
    else:
        log_event(f"[COLOR TERM] disabled for {obs_group} (apply_color_term toggle / filter type)")

    # Run-effective resolved facts for the honest full-config report (metadata only).
    _resolved_facts = _build_aperture_resolved_facts(
        cfg=_cfg,
        gain_res=_gain_res,
        rn_res=_rn_res,
        gain_value=_gain_phot,
        rn_value=_rn_phot,
        site=_site,
        sat_limit=sat_limit_resolved,
        plate_scale_arcsec=_plate_scale_arcsec,
        frame_width_px=chip_fw,
        frame_height_px=chip_fh,
        ms_header=_ms_header,
        obs_group=obs_group,
    )

    # GSPC-ZP-2026-10: second zero point from Gaia DR3 synthetic photometry (GSPC; Gaia
    # Collaboration, Montegriffo et al. 2023) for the comps of this group. One online query to
    # the Gaia archive per group, cached in the photometry folder (no local database). The
    # standard calibration stays the Gaia G / BP-RP relation; GSPC only adds a second scale.
    _gspc_rows: dict[str, Any] = {}
    _gspc_status = "not used"
    try:
        from gaia_bands import filter_band_for_obs_group as _fbo_g  # noqa: PLC0415
        from gspc_zp import GSPC_BAND, fetch_gspc  # noqa: PLC0415

        _zb_g = _fbo_g(obs_group).zp_band
        if _zb_g in GSPC_BAND and comp_df is not None and "catalog_id" in comp_df.columns:
            from gaia_catalog_id import normalize_gaia_source_id  # noqa: PLC0415

            # never via float: 19-digit Gaia ids are not exact in float64
            # COMP-SMART-2026-10: reserve comps may replace suspects -> fetch them too
            _res_ids_g = [x for _rdf in _reserve_index.values() if "catalog_id" in _rdf.columns
                          for x in _rdf["catalog_id"].tolist()]
            _ids_g = sorted({
                str(_n) for _n in (normalize_gaia_source_id(x) for x in list(comp_df["catalog_id"].tolist()) + _res_ids_g)
                if _n and str(_n).isdigit()
            })
            if _ids_g:
                _gspc_rows = fetch_gspc(_ids_g, Path(output_dir))
                _gspc_status = f"ok ({len(_gspc_rows)} of {len(_ids_g)} comps in the archive result)"
        else:
            _gspc_status = f"band {_zb_g} not in GSPC"
    except Exception as _gspc_exc:  # noqa: BLE001
        _gspc_status = f"unavailable ({type(_gspc_exc).__name__}: {_gspc_exc})"[:200]
        logging.warning("[GSPC] second zero point not available: %s", _gspc_exc)
    logging.info("[GSPC] %s", _gspc_status)

    return _AperturePhotometryState(
        at_df=at_df,
        comp_df=comp_df,
        _comp_index=_comp_index,
        reserve_index=_reserve_index,
        target_bp_rp_by_cid=target_bp_rp_by_cid,
        csv_files=csv_files,
        n_frames=n_frames,
        _aperture_csv_cache=_aperture_csv_cache,
        _aperture_lookup_cache=_aperture_lookup_cache,
        frame_time_lookup=frame_time_lookup,
        n_align_skipped=_n_align_skipped,
        fwhm_px=fwhm_px,
        apertures_px=apertures_px,
        star_xy=star_xy,
        chip_fw=chip_fw,
        chip_fh=chip_fh,
        _ms_header=_ms_header,
        _ms_data=_ms_data,
        _flux_matrix=_flux_matrix,
        _all_lc_ids_list=_all_lc_ids_list,
        field_map_path=field_map_path,
        obs_group=obs_group,
        _gain_phot=_gain_phot,
        _rn_phot=_rn_phot,
        equipment_id=_equipment_id,
        sat_limit_resolved=sat_limit_resolved,
        _aligned_dir_2a=_aligned_dir_2a,
        _cfg=_cfg,
        _nt=_nt,
        plate_scale_arcsec=(
            float(_plate_scale_arcsec)
            if _plate_scale_arcsec is not None
            and math.isfinite(float(_plate_scale_arcsec))
            and float(_plate_scale_arcsec) > 0
            else float("nan")
        ),
        gaia_db_path=_gaia_db_path,
        masterstars_df=masterstars_df,
        site_lat=_site.lat,
        site_lon=_site.lon,
        site_alt=_site.elev,
        site_source=_site.source,
        site_ok=bool(_site.ok),
        site_location_id=_site_loc_id,
        group_color_term=_group_ct,
        apply_color_term=bool(_apply_ct),
        k2_bprp=float(_k2_bprp),
        k2_source=str(_k2_src_enum.value),
        variable_target_catalog_ids=_variable_target_cids,
        snr_ap_table=_snr_ap_table,
        resolved_facts=_resolved_facts,
        per_frame_sat_meta=dict(_per_frame_sat_meta or {}),
        aperture_policy=_ap_pol,
        gspc_rows=_gspc_rows,
        gspc_status=_gspc_status,
    )
