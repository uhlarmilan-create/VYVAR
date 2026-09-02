"""Moved from photometry_core.py (CONSOLIDATE-01E4). Facade re-exports these names."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import json
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

from photometry_phase2a import (
    _ColorTermGroupFit,
    _Phase2AState,
    _build_csv_lookup,
    _compute_frame_align_residuals,
    _compute_group_color_term_fit,
    _draft_dir_from_phase2a_paths,
    _ensure_group_comp_pool_csv,
    _frame_align_residual_gate_select,
    _phase2a_attempt_k2_night_fit,
    _phase2a_cache_columns,
    _phase2a_coerce_skip_photometry,
    _record_align_residuals_to_report,
    _require_comparison_stars_per_target_schema,
    _resolve_phase2a_equipment_id,
    _sat_limit_peak_adu,
    compute_optimal_apertures,
    measure_fwhm_from_masterstar,
    read_flux_from_csv,
    resolve_apply_color_term,
    save_field_map_png,
)
from photometry_lightcurve import (
    _build_phase2a_resolved_facts,
    apply_per_frame_saturation_to_active_targets,
    evaluate_cog_night_apcorr_gate,
)

from photometry_core import (
    ERR_BKG_MODE_EMPIRICAL,
    LOGGER,
    _ADAPTIVE_BLEND_CACHE,
    _GAIA_ID_DTYPE,
)


def _phase2a_prepare_shared_state(
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
) -> _Phase2AState:
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
    _save_png = bool(_cfg.save_lightcurve_png)
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

    # Nacitaj vstupy (Gaia ID ako string - float64 straca cifry)
    if not Path(active_targets_csv).is_file():
        raise FileNotFoundError(f"active_targets_csv not found: {active_targets_csv}")
    if not Path(comparison_stars_csv).is_file():
        raise FileNotFoundError(f"comparison_stars_csv not found: {comparison_stars_csv}")
    at_df = pd.read_csv(active_targets_csv, low_memory=False, dtype=_GAIA_ID_DTYPE)
    comp_df = pd.read_csv(
        comparison_stars_csv,
        low_memory=False,
        dtype={**_GAIA_ID_DTYPE, "target_catalog_id": str},
    )

    # Normalizuj catalog_id
    for df in (at_df, comp_df):
        for col in ("catalog_id", "name"):
            if col in df.columns:
                df[col] = df[col].apply(_normalize_gaia_id)

    at_df["skip_photometry"] = _phase2a_coerce_skip_photometry(at_df)
    _require_comparison_stars_per_target_schema(comp_df, Path(comparison_stars_csv))

    # _phase2a_load_star_list (inline): chip dims, comp index, BP-RP map, CSV cache - through target loop below.

    # Open MASTERSTAR.fits once - reuse header + data throughout Phase 2A.
    _ms_header: Any = None
    _ms_data: np.ndarray | None = None
    _ms_path = Path(masterstar_fits_path)
    if _ms_path.is_file():
        try:
            with astrofits.open(_ms_path, memmap=False) as _hdul_ms:
                _ms_header = _hdul_ms[0].header.copy()
                if _hdul_ms[0].data is not None:
                    _ms_data = np.asarray(_hdul_ms[0].data, dtype=np.float64)
            LOGGER.debug("[PHASE 2A] MASTERSTAR.fits opened once (header + data cached)")
            logging.info(
                "[PERF-2] MASTERSTAR.fits loaded once: data shape=%s, header keys=%d",
                _ms_data.shape if _ms_data is not None else None,
                len(_ms_header) if _ms_header is not None else 0,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("[PHASE 2A] MASTERSTAR.fits open failed: %s", exc)

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
        raise ValueError("comparison_stars_per_target.csv musi obsahovat stlpce x, y pre Fazu 2A")

    # Pre-index comp_df podla target catalog_id - raz pre cely cyklus (O(1) lookup per target).
    _id_col_comp = "target_catalog_id" if "target_catalog_id" in comp_df.columns else "catalog_id"
    _comp_index: dict[str, pd.DataFrame] = {}
    if comp_df is not None and not comp_df.empty and _id_col_comp in comp_df.columns:
        for _tid, _grp in comp_df.groupby(comp_df[_id_col_comp].apply(_normalize_gaia_id), sort=False, dropna=False):
            _tid_s = str(_tid).strip()
            if _tid_s:
                _comp_index[_tid_s] = _grp.reset_index(drop=True)
    else:
        _comp_index = {}

    # masterstars_full_match.csv - BP-RP map + comp pool pre catalog_only Phase 2A.
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

    # Gated: generate crowding_targets.csv for full LC star set (adaptive blend map input).
    if bool(getattr(_cfg, "psf_adaptive_enabled", False)) and draft_id is not None and db is not None:
        try:
            from crowding_index import ensure_crowding_targets_for_lc
            from database import get_gaia_db_max_g_mag

            _setup_name = str(Path(per_frame_csv_dir).name)
            _draft_dir = Path(masterstar_fits_path).resolve().parent.parent.parent
            _gaia_max = float(get_gaia_db_max_g_mag(str(_cfg.gaia_db_path)))
            _crowd_csv = ensure_crowding_targets_for_lc(
                _draft_dir,
                _setup_name,
                db,
                int(draft_id),
                gaia_db_max_g=_gaia_max,
            )
            if _crowd_csv is not None:
                _ADAPTIVE_BLEND_CACHE.pop(str(Path(masterstar_fits_path)), None)
                _p2(f"Faza 2A: crowding_targets.csv ({_crowd_csv.name}) - adaptive blend map")
        except Exception as _cr_exc:  # noqa: BLE001
            LOGGER.warning(
                "[ePSF] crowding_targets generation failed (adaptive rule 2 disabled): %s",
                _cr_exc,
            )

    # Najdi per-frame CSV (FITS sa nepouziva)
    if proc_frame_store is not None and len(proc_frame_store) > 0:
        csv_files = sorted(Path(k) for k in proc_frame_store.keys())
    else:
        csv_files = sorted(Path(per_frame_csv_dir).glob("proc_*.csv"))
    _ms_epoch_drop = [p for p in csv_files if is_masterstar_proc_name(p)]
    if _ms_epoch_drop:
        logging.warning(
            "[MASTERSTAR-EPOCH] phase2a filtered %d masterstar proc from epoch set "
            "(canonical list_proc_csvs / ProcFrameStore filter was bypassed): %s",
            len(_ms_epoch_drop),
            ", ".join(p.name for p in _ms_epoch_drop[:3]),
        )
        csv_files = [p for p in csv_files if not is_masterstar_proc_name(p)]
    # Fix B: per-frame alignment residual (px). Always-on QC: compute on the full frame set and
    # record into alignment_report.csv (additive metadata -> photometry stays byte-identical).
    # Then, only if the gate is enabled, drop frames whose residual exceeds the rig-agnostic
    # threshold (fraction of the science aperture radius). Cause-correct counterpart to B.2.
    try:
        _align_resid, _align_apr = _compute_frame_align_residuals(csv_files, proc_frame_store)
        _align_report_path = Path(masterstar_fits_path).resolve().parent / "alignment_report.csv"
        _record_align_residuals_to_report(_align_report_path, _align_resid)
    except Exception as _ar_exc:  # noqa: BLE001
        LOGGER.warning("[ALIGN-QC] residual compute/record failed (gate disabled this run): %s", _ar_exc)
        _align_resid, _align_apr = {}, float("nan")
    if getattr(_cfg, "frame_align_residual_gate_enabled", False):
        _kept_ar, _rejected_ar, _ar_thr = _frame_align_residual_gate_select(
            csv_files, _cfg, _align_resid, _align_apr
        )
        if _rejected_ar:
            csv_files = _kept_ar
            logging.info(
                "[ALIGN-QC] alignment-residual gate: rejected %d/%d frames "
                "(thr=%.2f px = %.2f*%.2f apr; %s%s)",
                len(_rejected_ar),
                len(_rejected_ar) + len(_kept_ar),
                _ar_thr,
                float(_cfg.frame_align_residual_max_frac),
                float(_align_apr),
                ", ".join(_rejected_ar[:5]),
                " ..." if len(_rejected_ar) > 5 else "",
            )
            _p2(
                f"Align-residual gate: {len(_rejected_ar)} mis-aligned frames rejected, "
                f"{len(_kept_ar)} kept"
            )
    # Len CSV - bez FITS (flux sa cita z dao_flux v CSV)
    n_frames = len(csv_files)
    _n_total = int(len(at_df))
    logging.info("[PHASE 2A] %d targets (DAO+Gaia matched)", _n_total)
    _p2(f"Phase 2A: {_n_total} targets, {n_frames} frames - loading CSV cache...")

    # Nacitaj CSV cache raz pre celu Fazu 2A (read_flux_from_csv per target inak 82x na target).
    if proc_frame_store is not None and len(proc_frame_store) > 0:
        _phase2a_csv_cache = proc_frame_store
        logging.info(
            "[PERF-5] run_phase2a: using ProcFrameStore (%d frames, 0 disk reads)",
            len(proc_frame_store),
        )
        _p2(f"Faza 2A: ProcFrameStore {len(proc_frame_store)} CSV - vypocet FWHM / apertur...")
    else:
        logging.info("[FAZA 2A] Nacitavam CSV cache...")
        _t_cache = time.time()
        _phase2a_csv_cache: dict[str, pd.DataFrame] = {}
        _needed_cols_2a = _phase2a_cache_columns()
        for _csv_path in csv_files:
            try:
                _hdr = pd.read_csv(_csv_path, nrows=0)
                _cols = [c for c in _needed_cols_2a if c in _hdr.columns]
                if not _cols:
                    continue
                # Gaia ID musi byt str - float64 straca cifry
                _dtype_2a: dict[str, type] = {}
                if "catalog_id" in _cols:
                    _dtype_2a["catalog_id"] = str
                if "name" in _cols:
                    _dtype_2a["name"] = str
                _kw: dict[str, Any] = {"usecols": _cols, "low_memory": False}
                if _dtype_2a:
                    _kw["dtype"] = _dtype_2a
                _phase2a_csv_cache[str(_csv_path)] = pd.read_csv(_csv_path, **_kw)
            except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
                from except_fix_counters import get_except_fix_counters

                get_except_fix_counters().phase2a_csv_cache_skip += 1
                logging.error("[PHASE 2A] proc CSV cache skip %s: %s", _csv_path, exc)
                continue
        logging.info(
            f"[FAZA 2A] CSV cache: {len(_phase2a_csv_cache)} suborov "
            f"({time.time() - _t_cache:.1f}s)"
        )
        _p2(f"Faza 2A: cache {len(_phase2a_csv_cache)} CSV - vypocet FWHM / apertur...")

    # Lookup (id_map + xy_df) raz na snimku - inak _build_csv_lookup 82x na target.
    _phase2a_lookup_cache: dict[str, tuple[dict[str, pd.Series], pd.DataFrame]] = {}
    for _cp in csv_files:
        _key = str(_cp)
        _df_lu = _phase2a_csv_cache.get(_key)
        if _df_lu is None or _df_lu.empty:
            continue
        _id_col_lu = "catalog_id" if "catalog_id" in _df_lu.columns else "name"
        _phase2a_lookup_cache[_key] = _build_csv_lookup(_df_lu, _id_col_lu)

    # Cas + airmass (+ flip flag z alignment_report) z prveho platneho riadku kazdeho per-frame CSV
    # (podla stem FITS).
    frame_time_lookup: dict[str, dict[str, float]] = {}
    for csv_path in csv_files:
        stem = csv_path.stem
        _csv_tmp = _phase2a_csv_cache.get(str(csv_path))
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
                    if "is_flipped" in rep.columns:
                        try:
                            frame_time_lookup[stem]["is_flipped"] = bool(r.get("is_flipped", False))
                        except Exception:  # noqa: BLE001
                            frame_time_lookup[stem]["is_flipped"] = False
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
                    if "reason" in rep.columns:
                        _rs = str(r.get("reason", "") or "").strip()
                        if _rs:
                            frame_time_lookup[stem]["alignment_reason"] = _rs
    except Exception as exc:  # noqa: BLE001
        logging.error('[EXC-0168] alignment_report.csv alignment_failed/reason not propagated into frame_time_lookup: %s', exc)
        pass

    # PER-FRAME-SAT-GATED: when ON, revise target skip_photometry from per-frame
    # clean fraction before aperture/FWHM star-set selection. OFF = no-op.
    # Peak-test uses INV-SAT-LIMIT catalog authority (not raw container clip).
    _pfs_enabled = bool(getattr(_cfg, "per_frame_saturation_enabled", False))
    _pfs_min = float(getattr(_cfg, "per_frame_sat_min_clean_frac", 0.5))
    _peak_test_adu: float | None = None
    _peak_test_src = ""
    try:
        from pipeline import inv_sat_limit_peak_test_adu  # noqa: PLC0415

        _peak_test_adu, _peak_test_src = inv_sat_limit_peak_test_adu()
    except Exception as exc:  # noqa: BLE001
        logging.error("[EXC-PFS] INV-SAT-LIMIT peak-test resolve failed: %s", exc)
        _peak_test_adu = None
        _peak_test_src = ""
    _per_frame_sat_meta = apply_per_frame_saturation_to_active_targets(
        at_df,
        csv_files=csv_files,
        csv_cache=_phase2a_csv_cache,
        sat_limit_adu=None,
        enabled=_pfs_enabled,
        min_clean_frac=_pfs_min,
        peak_test_adu=_peak_test_adu,
        peak_test_source=_peak_test_src,
    )
    if _pfs_enabled:
        try:
            at_df.to_csv(active_targets_csv, index=False)
        except Exception as exc:  # noqa: BLE001
            logging.error(
                "[EXC-PFS] active_targets.csv rewrite after per-frame sat failed: %s",
                exc,
            )
        logging.info(
            "[PER-FRAME-SAT] enabled: rescued=%s skipped=%s fallback=%s "
            "(min_clean_frac=%.3f peak_test=%.1f ADU src=%s container_clip=%.0f ADU)",
            _per_frame_sat_meta.get("per_frame_sat_n_rescued"),
            _per_frame_sat_meta.get("per_frame_sat_n_skipped"),
            _per_frame_sat_meta.get("per_frame_sat_n_fallback"),
            _pfs_min,
            float(_per_frame_sat_meta.get("per_frame_sat_peak_test_adu") or float("nan")),
            str(_per_frame_sat_meta.get("per_frame_sat_peak_test_source") or ""),
            float(_per_frame_sat_meta.get("per_frame_sat_container_clip_adu") or float("nan")),
        )

    # Krok 1: Globalna fixna apertura - vsetky hviezdy (target + comp), faktor x FWHM
    # Ciele so skip_photometry (saturovane) nepatria do vypoctu apertur / FWHM z targetov.
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

    # Priorita: 1. VY_FWHM_GAUSS (2D fit v hlavicke), 2. VY_FWHM (DAO), 3. fit fallback
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
                        f"[FAZA 2A] FWHM z VY_FWHM_GAUSS (2D fit): {_fwhm_from_header:.3f} px"
                    )
            if _fwhm_from_header is None and vy_fwhm_dao is not None:
                _fvd = float(vy_fwhm_dao)
                if 0.5 < _fvd < 30.0:
                    _fwhm_from_header = _fvd
                    logging.info(
                        f"[FAZA 2A] FWHM z VY_FWHM (DAO): {_fwhm_from_header:.3f} px"
                    )
    except Exception as _e:  # noqa: BLE001
        logging.error('[EXC-0170] MASTERSTAR header FWHM read throws - measured FWHM from stars used instead (logged warn...: %s', _e)
        logging.warning(f"[FAZA 2A] Nemozem citat FWHM z hlavicky: {_e}")

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
        logging.info(f"[FAZA 2A] FWHM z Gaussian fit: {fwhm_px:.3f} px")

    _p2(f"Faza 2A: FWHM={float(fwhm_px):.3f} px - mapa pola a svetelne krivky...")

    # Gain / read noise for photometric errors and SNR aperture table.
    # Gain: header-first (e-/ADU or index-mapped) -> DB -> config. RN: DB-first.
    from param_resolver import resolve_gain, resolve_read_noise  # noqa: PLC0415

    _equipment_id = _resolve_phase2a_equipment_id(
        db,
        draft_id=draft_id,
        output_dir=output_dir,
        masterstar_fits_path=Path(masterstar_fits_path),
    )
    _gain_res = resolve_gain(_ms_header, db=db, equipment_id=_equipment_id, cfg=_cfg)
    _rn_res = resolve_read_noise(_ms_header, db=db, equipment_id=_equipment_id, cfg=_cfg)
    _gain_native = float(_gain_res.value) if _gain_res.ok else 1.0
    _rn_phot = float(_rn_res.value) if _rn_res.ok else 10.0
    # WIDE-ERR-03: science gain is container-domain (g_pt or DB/scale), never bare native.
    from gain_photon_transfer import (  # noqa: PLC0415
        DEFAULT_CONTAINER_SCALE,
        apply_photometric_gain_authority,
        resolve_photon_transfer_aperture_r_px,
    )

    _proc_dir_gain = Path(_aligned_dir_2a) if _aligned_dir_2a is not None else None
    # GAIN-PT-RADIUS-01: pin sky-dominated r=4.0. Do not read leftover
    # dynamic_params.aperture_r_px and do not let force_aperture_px override PT
    # (that flag sizes star photometry, not empty-aperture PT).
    _ap_r_gain, _ap_r_src = resolve_photon_transfer_aperture_r_px()
    _scale = float(getattr(_cfg, "gain_container_scale", DEFAULT_CONTAINER_SCALE) or DEFAULT_CONTAINER_SCALE)
    _ci_w = float(getattr(_cfg, "photon_transfer_ci_max_width_factor", 3.0) or 3.0)
    _sidecar = Path(output_dir) / "gain_photon_transfer.json" if output_dir is not None else None
    _gain_phot, _gain_auth, _ = apply_photometric_gain_authority(
        g_db_native=_gain_native,
        native_source=_gain_res.source if _gain_res.ok else "default",
        proc_dir=_proc_dir_gain,
        aperture_r_px=_ap_r_gain,
        container_scale=_scale,
        ci_max_width_factor=_ci_w,
        persist_sidecar=_sidecar,
        draft_meta={"draft_id": draft_id, "stage": "phase2a"},
        aperture_r_px_source=_ap_r_src,
    )
    if not math.isfinite(_gain_phot) or _gain_phot <= 0:
        _gain_phot = _gain_native / _scale if _scale > 0 else 1.0
    logging.info(
        "[PHASE 2A] Photometric errors: gain=%.3f e-/ADU_container (authority=%s; native=%.3f src=%s), "
        "RN=%.1f e- (source: %s)",
        float(_gain_phot),
        _gain_auth.source,
        float(_gain_native),
        _gain_res.source if _gain_res.ok else "default",
        float(_rn_phot),
        _rn_res.source if _rn_res.ok else "default",
    )

    _draft_dir_ee = _draft_dir_from_phase2a_paths(output_dir, Path(masterstar_fits_path))
    _snr_ap_table = None

    _variable_target_cids = frozenset(
        c
        for _, row in at_df.iterrows()
        for c in [_normalize_gaia_id(row.get("catalog_id", ""))]
        if c
    )

    _ap_pol: dict[str, Any] | None = None
    if force_aperture_px is not None and force_aperture_px > 0:
        # Fixna apertura pre vsetky hviezdy - debug/kalibracia
        apertures_px = {
            _normalize_gaia_id(row.get("catalog_id", "")): float(force_aperture_px)
            for _, row in all_stars.iterrows()
            if _normalize_gaia_id(row.get("catalog_id", ""))
        }
        logging.info(
            f"[FAZA 2A] FORCE apertura: {force_aperture_px:.2f}px pre vsetky hviezdy"
        )
    else:
        # APERTURE-01: one r for every star. SNR mag-bin table is diagnostic only.
        from aperture_policy import (  # noqa: PLC0415
            fwhm_for_radius,
            load_qc_fwhm_map,
            normalize_aperture_policy_mode,
            policy_record,
            resolve_aperture_geometry,
        )
        _ap_mode = normalize_aperture_policy_mode(
            getattr(_cfg, "aperture_policy_mode", "f_fixed_night")
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
        _qc_map_p2a, _night_p2a = load_qc_fwhm_map(_qc_csv_p2a)
        _ = _qc_map_p2a
        _fw_p2a = fwhm_for_radius(
            _ap_mode,
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
            Path(masterstar_fits_path),
            all_stars,
            float(_fw_p2a),
            aperture_fwhm_factor=_apt_fw,
            annulus_inner_fwhm=annulus_inner_fwhm,
            annulus_outer_fwhm=annulus_outer_fwhm,
        )
        _r_ap_p2a, _r_in_p2a, _r_out_p2a = resolve_aperture_geometry(
            f=float(_apt_fw),
            fwhm_px=float(_fw_p2a),
            annulus_inner_fwhm=float(annulus_inner_fwhm),
            annulus_outer_fwhm=float(annulus_outer_fwhm),
        )
        _ap_pol = policy_record(
            mode=_ap_mode,
            f=float(_apt_fw),
            fwhm_frame_px=None,
            fwhm_night_median_px=_night_p2a,
            r_ap=_r_ap_p2a,
            r_in=_r_in_p2a,
            r_out=_r_out_p2a,
            fwhm_used_px=_fw_p2a,
        )
        try:
            _pol_path = Path(output_dir) / "aperture_policy.json"
            with _pol_path.open("w", encoding="utf-8") as _pf:
                json.dump(_ap_pol, _pf, indent=2)
            logging.info(
                "[FAZA 2A] APERTURE-01 %s f=%.3f FWHM_used=%.3f r_ap=%.3f (n_stars=%d)",
                _ap_mode,
                float(_apt_fw),
                float(_fw_p2a),
                float(_r_ap_p2a),
                len(apertures_px),
            )
        except Exception as _pol_exc:  # noqa: BLE001
            logging.warning("[FAZA 2A] aperture_policy.json write failed: %s", _pol_exc)

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

    # Field map PNG (raz pre cele pole) - vzdy; UI potrebuje mapu aj bez PNG kriviek
    field_map_path = output_dir / "field_map.png"
    save_field_map_png(
        field_map_path,
        Path(masterstar_fits_path),
        at_df,
        comp_df,
        ms_data=_ms_data,
    )

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
    _all_lc_ids_list = sorted(_all_lc_ids)
    logging.info(
        "[PERF-8] Flux matrix: %d unique star IDs across %d targets",
        len(_all_lc_ids_list),
        len(at_df),
    )

    # APCORR-MIXEDFRAME-ALLORNOTHING: night-level COG gate before any LC flux routing.
    _cog_enabled = bool(getattr(_cfg, "cog_aperture_correction_enabled", False))
    _cog_frame_dfs = [_phase2a_csv_cache.get(str(_p)) for _p in csv_files]
    _cog_gate = evaluate_cog_night_apcorr_gate(_cog_frame_dfs, enabled=_cog_enabled)
    _use_apcorr_flux = bool(_cog_gate["use_apcorr_flux"])
    _cog_night_fallback = bool(_cog_gate["cog_night_fallback"])
    _cog_n_bad = int(_cog_gate["n_without_cog_ok"])
    _cog_n_frames = int(_cog_gate["n_frames"])
    if _cog_night_fallback and _cog_enabled:
        logging.info(
            "[APCORR] COG night fallback: %d/%d frames without cog_ok "
            "-> whole night uses standard AC",
            _cog_n_bad,
            _cog_n_frames,
        )

    # PERF-8: one read_flux_from_csv pass per frame for all LC stars (not per target).
    _flux_matrix_rows: list[pd.DataFrame] = []
    _t_flux_matrix = time.perf_counter()
    for _csv_path in csv_files:
        _cached_df = _phase2a_csv_cache.get(str(_csv_path))
        _lookup_row = _phase2a_lookup_cache.get(str(_csv_path))
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
            use_apcorr_flux=_use_apcorr_flux,
            variable_target_catalog_ids=_variable_target_cids,
            err_background_mode=ERR_BKG_MODE_EMPIRICAL,
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
            "[PHASE 2A] GS11 context: plate_scale=%.4f arcsec/px gaia_db=%s",
            float(_plate_scale_arcsec),
            "set" if _gaia_db_path else "none",
        )
    else:
        logging.warning(
            "[PHASE 2A] GS11 context: plate_scale unknown (derive-or-None) gaia_db=%s",
            "set" if _gaia_db_path else "none",
        )

    # Observer site resolved ONCE per draft (param_resolver): draft ID_LOCATION ->
    # header SITELAT -> flagged config. Threaded into BJD/HJD recompute + meta so
    # Phase 2A no longer silently reads cfg.observer_* (config-drift trap).
    from param_resolver import resolve_site as _resolve_site  # noqa: PLC0415

    _site = _resolve_site(_ms_header, db=db, draft_id=draft_id, cfg=_cfg)
    _site_loc_id: int | None = None
    if db is not None and draft_id is not None:
        try:
            if hasattr(db, "get_draft_location_id"):
                _site_loc_id = db.get_draft_location_id(int(draft_id))
        except Exception:  # noqa: BLE001
            _site_loc_id = None
    logging.info(
        "[PHASE 2A] Observer site: source=%s lat=%s lon=%s alt=%s ok=%s",
        _site.source,
        f"{_site.lat:.4f}" if _site.lat is not None else "None",
        f"{_site.lon:.4f}" if _site.lon is not None else "None",
        f"{_site.elev:.0f}" if _site.elev is not None else "None",
        _site.ok,
    )

    from k2_extinction import resolve_k2_bprp_value  # noqa: PLC0415

    # NIGHT_FIT v2 (gated): attempt only when k2_fit_enabled; default OFF skips entirely.
    _k2_night_fit = None
    _k2_fit_meta: dict[str, Any] = {}
    if bool(getattr(_cfg, "k2_fit_enabled", False)):
        _k2_night_fit = _phase2a_attempt_k2_night_fit(
            cfg=_cfg,
            obs_group=obs_group,
            flux_matrix=_flux_matrix,
            csv_files=csv_files,
            comparison_stars_csv=Path(comparison_stars_csv),
            masterstar_fits_path=Path(masterstar_fits_path),
        )
        if _k2_night_fit is not None:
            _k2_fit_meta = dict(_k2_night_fit.to_meta())

    _k2_bprp, _k2_src_enum = resolve_k2_bprp_value(
        _cfg, obs_group, night_fit_result=_k2_night_fit
    )
    if _k2_src_enum.value != "none" and math.isfinite(float(_k2_bprp)):
        log_event(f"[K2] obs_group {obs_group}: k2={float(_k2_bprp):.6f} source={_k2_src_enum.value}")
    if _k2_fit_meta.get("k2_fit_refuse_reason"):
        log_event(
            f"[K2-FIT] refused ({_k2_fit_meta.get('k2_fit_refuse_reason')}); "
            f"using source={_k2_src_enum.value}"
        )

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
    _resolved_facts = _build_phase2a_resolved_facts(
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

    return _Phase2AState(
        at_df=at_df,
        comp_df=comp_df,
        _comp_index=_comp_index,
        target_bp_rp_by_cid=target_bp_rp_by_cid,
        csv_files=csv_files,
        n_frames=n_frames,
        _phase2a_csv_cache=_phase2a_csv_cache,
        _phase2a_lookup_cache=_phase2a_lookup_cache,
        frame_time_lookup=frame_time_lookup,
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
        use_apcorr_flux=_use_apcorr_flux,
        cog_night_fallback=_cog_night_fallback,
        cog_night_fallback_n_without_ok=_cog_n_bad,
        cog_night_fallback_n_frames=_cog_n_frames,
        per_frame_sat_meta=dict(_per_frame_sat_meta or {}),
        k2_fit_meta=dict(_k2_fit_meta or {}),
        aperture_policy=_ap_pol,
    )
