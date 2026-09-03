"""Moved from pipeline.py (CONSOLIDATE-01E6a). Facade re-exports these names.

Catalog queries, field-catalog cone cache, per-frame export workers,
DAO-to-catalog match, saturation zones, airmass fills.
Spawn-worker global `_EXPORT_PER_FRAME_WORKER_STATE` lives here so
`global` binds one namespace.
The four giants stay in pipeline.py this wave (E6b).
"""
from __future__ import annotations

import json
import logging
import pickle
import math
import traceback
import warnings
from pathlib import Path
from typing import Any, Sequence
import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.wcs import FITSFixedWarning
import pandas as pd
from config import AppConfig
from catalog_match_trust import export_catalog_match_mode_from_internal
from database import VyvarDatabase, query_local_gaia, query_local_exoplanet, query_local_vsx
from time_utils import _header_float as _header_float_tu
from photometry import compute_fwhm_gaussian_for_aperture_catalog, enhance_catalog_dataframe_aperture_bpm
from gaia_catalog_id import catalog_id_series_for_masterstars_export, normalize_gaia_source_id, read_vyvar_csv
from infolog import log_event
from proc_frame_store import proc_csv_path_for_aligned_fits
from plain_stats import plain_mean_med_std
from utils import DAO_STAR_FINDER_NO_ROUNDNESS_FILTER, MIN_GAIA_CONE_RADIUS_DEG, catalog_cone_radius_deg_from_optics, catalog_cone_radius_from_fov_diameter_deg, dao_detection_fwhm_pixels, effective_binned_pixel_pitch_um, fits_binning_xy_from_header, normalize_telescope_focal_mm_for_plate_scale, strip_celestial_wcs_keys
from masterstar_gaia_accounting import _dao_xy_binned_to_full
from pipeline_calibrate import (
    _effective_saturation_limit,
    _has_valid_wcs,
)
from pipeline_preprocess import (
    _load_raw_for_frame,
    _load_raw_hdr_for_frame,
)
from pipeline_astrometry import (
    _finite_positive_adu,
    _vyvar_df_to_csv,
    resolve_plate_solve_fov_deg_hint,
)
from pipeline_constants import (
    SAT_LIMIT_CONTAINER_CLIP_ADU,
    SAT_LIMIT_NO_KNEE_FRAC,
    SAT_LIMIT_PEAK_TEST_SOURCE,
    _EXO_HOST_ANNOTATION_COLUMNS,
)

# Same named logger as pipeline.LOGGER (logging.getLogger singleton).
# Avoids pipeline -> pipeline_catalog -> pipeline at module load.
LOGGER = logging.getLogger("pipeline")

def _apply_aperture_catalog_enhancements_from_st(
    df: pd.DataFrame,
    data: Any,
    hdr: fits.Header,
    st: dict[str, Any],
) -> pd.DataFrame:
    """Aperture photometry + linearity/BPM flags for per-frame catalog DataFrames."""
    from pipeline import enhance_catalog_dataframe_aperture_bpm  # noqa: PLC0415

    if not bool(st.get("_run_aperture", st.get("aperture_photometry_enabled", True))):
        return df
    mdp = st.get("master_dark_path") or ""
    mdp = str(mdp).strip() or None

    r_small_px: float | None = None
    r_large_px: float | None = None
    if bool(st.get("aperture_correction_enabled", False)):
        try:
            data_arr = np.asarray(data, dtype=np.float32)
            _apt_fw = float(st.get("aperture_fwhm_factor", 1.7))
            _fper, _fmed, fw_g = compute_fwhm_gaussian_for_aperture_catalog(
                df,
                data_arr,
                hdr,
                gaussian_fwhm_px_override=st.get("gaussian_fwhm_px_override"),
                aperture_fwhm_factor=_apt_fw,
            )
            if math.isfinite(fw_g) and fw_g > 0:
                r_small_px = float(st.get("aperture_fwhm_factor_small", 1.5)) * float(fw_g)
                r_large_px = float(st.get("aperture_fwhm_factor_large", 4.0)) * float(fw_g)
        except Exception:  # noqa: BLE001
            r_small_px, r_large_px = None, None

    cog_params: dict[str, Any] | None = None
    if bool(st.get("cog_aperture_correction_enabled", False)):
        cog_params = {
            "ref_fwhm": float(st.get("cog_ref_fwhm", 4.5)),
            "ladder_step_px": float(st.get("cog_ladder_step_px", 0.5)),
            "ladder_step_fwhm": st.get("cog_ladder_step_fwhm"),
            "min_stars": int(st.get("cog_min_stars", 8)),
            "isolation_fwhm": float(st.get("cog_isolation_fwhm", 6.0)),
            "snr_min": float(st.get("cog_snr_min", 50.0)),
            "sat_frac": float(st.get("cog_sat_frac", 0.85)),
            "gain": float(st.get("gain", 1.0)),
            "read_noise": float(st.get("read_noise", 10.0)),
            "ac_factor_max": float(st.get("cog_ac_factor_max", 5.0)),
        }

    try:
        _go = st.get("gaussian_fwhm_px_override")
        try:
            _go_f = float(_go) if _go is not None else None
        except (TypeError, ValueError):
            _go_f = None
        return enhance_catalog_dataframe_aperture_bpm(
            df,
            data,
            hdr,
            aperture_enabled=True,
            aperture_fwhm_factor=float(st.get("aperture_fwhm_factor", 1.7)),
            annulus_inner_fwhm=float(st.get("annulus_inner_fwhm", 4.0)),
            annulus_outer_fwhm=float(st.get("annulus_outer_fwhm", 6.0)),
            nonlinearity_peak_percentile=float(st.get("nonlinearity_peak_percentile", 20.0)),
            nonlinearity_fwhm_ratio=float(st.get("nonlinearity_fwhm_ratio", 1.25)),
            master_dark_path=mdp,
            gaussian_fwhm_px_override=_go_f,
            r_small_px=r_small_px,
            r_large_px=r_large_px,
            cog_params=cog_params,
            err_background_mode="empirical",
            err_empty_apertures_n=int(st.get("err_empty_apertures_n", 64)),
            err_empty_apertures_min=int(st.get("err_empty_apertures_min", 16)),
            aperture_variable_factor=float(st.get("aperture_variable_factor", 1.0)),
            aperture_comp_factor=float(st.get("aperture_comp_factor", 1.0)),
            variable_target_catalog_ids=frozenset(st.get("variable_target_catalog_ids") or []),
            aperture_policy_mode=st.get("aperture_policy_mode"),
            fwhm_night_median_px=st.get("fwhm_night_median_px"),
            qc_fwhm_by_name=st.get("qc_fwhm_by_name"),
            frame_name=st.get("current_frame_name"),
        )
    except Exception as exc:  # noqa: BLE001
        from except_fix_counters import get_except_fix_counters

        get_except_fix_counters().catalog_bpm_enhance_fail += 1
        LOGGER.error("[CATALOG] aperture/BPM/COG enhancement failed: %s", exc)
        return df


def _fits_header_vy_algn_aligned(hdr: fits.Header) -> bool:
    """True when frame pixels are on the MASTERSTAR reference alignment grid (VY_ALGN)."""
    try:
        val = hdr.get("VY_ALGN")
        if val is None:
            return True  # legacy frames without tag - keep pixel-fallback behaviour
        if isinstance(val, tuple):
            val = val[0]
        if isinstance(val, bool):
            return bool(val)
        s = str(val).strip().lower()
        if s in ("true", "1", "t", "yes"):
            return True
        if s in ("false", "0", "f", "no"):
            return False
    except Exception:  # noqa: BLE001
        pass
    return True


def _epsf_target_catalog_ids(platesolve_dir: Path, *, top_comps: int = 40) -> set[str] | None:
    """Catalog IDs for targeted ePSF: active targets + best comparison stars by ``comp_rms``."""
    phot = Path(platesolve_dir) / "photometry"
    ids: set[str] = set()
    at_p = phot / "active_targets.csv"
    if at_p.is_file():
        try:
            at = read_vyvar_csv(at_p, low_memory=False)
            if "catalog_id" in at.columns:
                for _, row in at.iterrows():
                    z = str(row.get("zone_flag", row.get("zone", "")) or "").strip().lower()
                    if z == "catalog_only":
                        continue
                    raw = row.get("catalog_id")
                    if raw is None:
                        continue
                    s = str(raw).strip()
                    if not s or s.lower() in ("nan", "none"):
                        continue
                    try:
                        ids.add(str(normalize_gaia_source_id(s)).strip())
                    except Exception:  # noqa: BLE001
                        ids.add(s)
        except Exception:  # noqa: BLE001
            pass
    for comp_p in (phot / "comparison_stars.csv", Path(platesolve_dir) / "comparison_stars.csv"):
        if not comp_p.is_file():
            continue
        try:
            cdf = read_vyvar_csv(comp_p, low_memory=False)
            if "catalog_id" in cdf.columns and not cdf.empty:
                for raw in cdf["catalog_id"].fillna("").astype(str).str.strip():
                    if not raw or raw.lower() in ("nan", "none"):
                        continue
                    try:
                        ids.add(str(normalize_gaia_source_id(raw)).strip())
                    except Exception:  # noqa: BLE001
                        ids.add(raw)
            break
        except Exception as _comp_exc:  # noqa: BLE001
            LOGGER.warning("[ePSF] comparison_stars load failed: %s", _comp_exc)
    comp_p = phot / "comparison_stars_per_target.csv"
    if comp_p.is_file() and len(ids) < 5:
        try:
            cdf = read_vyvar_csv(comp_p, low_memory=False)
            if "catalog_id" in cdf.columns and not cdf.empty:
                work = cdf.copy()
                if "comp_rms" in work.columns:
                    work["_cr"] = pd.to_numeric(work["comp_rms"], errors="coerce")
                    work = work.sort_values("_cr", ascending=True, na_position="last")
                seen: list[str] = []
                for raw in work["catalog_id"].fillna("").astype(str).str.strip():
                    if not raw or raw.lower() in ("nan", "none"):
                        continue
                    try:
                        nk = str(normalize_gaia_source_id(raw)).strip()
                    except Exception:  # noqa: BLE001
                        nk = raw
                    if nk in seen:
                        continue
                    seen.append(nk)
                    ids.add(nk)
                    if len(seen) >= int(top_comps):
                        break
        except Exception as _comp_exc:  # noqa: BLE001
            LOGGER.warning("[ePSF] comparison_stars_per_target load failed: %s", _comp_exc)
    if ids:
        LOGGER.debug("[ePSF] target catalog_ids loaded: %d (active+top comps)", len(ids))
    return ids if ids else None


def _epsf_fit_catalog_ids(
    platesolve_dir: Path,
    *,
    psf_photometry_enabled: bool = False,
) -> set[str] | None:
    """Target-only subset (default) vs science set when PSF photometry is enabled."""
    if psf_photometry_enabled:
        from epsf_science_set import build_epsf_science_set

        result = build_epsf_science_set(platesolve_dir)
        if not result.catalog_ids:
            raise ValueError(
                "ePSF science set is empty"
                + (f": {result.empty_reason}" if result.empty_reason else "")
                + "; refusing silent fallback to full LC pool."
            )
        LOGGER.debug(
            "[ePSF] science-set catalog_ids loaded: %d (targets=%d comps=%d checks=%d blended=%d)",
            result.n_total,
            result.n_targets,
            result.n_per_target_comps,
            result.n_check_stars,
            result.n_blended,
        )
        return set(result.catalog_ids)
    return _epsf_target_catalog_ids(platesolve_dir)


def _fill_psf_catalog_columns(
    df: pd.DataFrame,
    data: Any,
    hdr: fits.Header,
    st: dict[str, Any],
    *,
    target_ids: set[str] | None = None,
) -> pd.DataFrame:
    """TODO-8 Phase 2B: PSF photometry columns after aperture (additive; never replaces dao_flux)."""
    _run_epsf = bool(st.get("_run_epsf", False))
    _run_aperture = bool(st.get("_run_aperture", False))
    _epsf_raw = str(st.get("epsf_model_path", "") or "").strip()
    _epsf_path = Path(_epsf_raw) if _epsf_raw else None
    _frame_name = str(st.get("epsf_frame_name") or st.get("current_frame_name") or "")
    _frame_index = st.get("epsf_frame_index")
    _psf_record: dict[str, Any] = {
        "frame_name": _frame_name,
        "frame_index": _frame_index,
        "n_fit": 0,
        "n_ok": 0,
        "exception_class": None,
        "exception_message": None,
        "traceback_tail": None,
    }
    if _run_epsf and _epsf_path is not None and _epsf_path.is_file():
        _n_fit = 0
        _n_ok = 0
        try:
            from psf_photometry import psf_photometry_stars as _psf_phot

            _gain = float(st.get("gain", 1.0) or 1.0)
            _rn = float(st.get("read_noise", 10.0) or 10.0)
            if not math.isfinite(_gain) or _gain <= 0:
                _gain = 1.0
            if not math.isfinite(_rn) or _rn <= 0:
                _rn = 10.0
            d_arr = np.asarray(data, dtype=np.float32)
            _err_map = np.sqrt(np.abs(d_arr) / _gain + (_rn / _gain) ** 2).astype(np.float32)

            if not {"catalog_id", "x", "y"}.issubset(df.columns):
                raise ValueError("catalog missing catalog_id/x/y for PSF photometry")

            _tid = target_ids
            if _tid is None:
                _raw_tid = st.get("epsf_target_ids")
                if isinstance(_raw_tid, (set, frozenset, list, tuple)) and _raw_tid:
                    _tid = {str(x).strip() for x in _raw_tid if str(x).strip()}

            for _c in ("psf_flux", "psf_flux_err", "psf_chi2"):
                if _c not in df.columns:
                    df[_c] = float("nan")
            if "psf_fit_ok" not in df.columns:
                df["psf_fit_ok"] = False
            if "psf_quality" not in df.columns:
                df["psf_quality"] = ""
            if "psf_snr" not in df.columns:
                df["psf_snr"] = float("nan")

            _fit_df = df
            if _tid is not None:
                _cid_key = df["catalog_id"].fillna("").astype(str).str.strip().map(
                    lambda x: (
                        str(normalize_gaia_source_id(x)).strip()
                        if x and str(x).lower() not in ("nan", "none")
                        else ""
                    )
                )
                _fit_df = df.loc[_cid_key.isin(_tid)].copy()

            _n_all = int(len(df))
            _n_fit = int(len(_fit_df))
            if _n_fit > 0:
                _pos = _fit_df[["catalog_id", "x", "y"]].copy()
                if "name" in _fit_df.columns:
                    _pos["name"] = _fit_df["name"]
                else:
                    _pos["name"] = _fit_df["catalog_id"].astype(str)

                _ref_fluxes: np.ndarray | None = None
                if "dao_flux" in _fit_df.columns:
                    _ref_fluxes = pd.to_numeric(_fit_df["dao_flux"], errors="coerce").to_numpy(
                        dtype=float
                    )
                    if len(_ref_fluxes) != len(_pos):
                        _ref_fluxes = None

                _psf_df = _psf_phot(
                    frame_data=d_arr,
                    frame_hdr=hdr,
                    star_positions=_pos,
                    epsf_model_path=_epsf_path,
                    error=_err_map,
                    ref_fluxes=_ref_fluxes,
                    apply_aperture_correction=False,
                    psf_ac_policy="p4_none",
                )
                _psf_idx = _psf_df.drop_duplicates(subset=["catalog_id"], keep="last").set_index(
                    "catalog_id"
                )
                for _col in (
                    "psf_flux",
                    "psf_flux_err",
                    "psf_chi2",
                    "psf_fit_ok",
                    "psf_ac_factor",
                    "psf_ac_n_used",
                    "psf_ac_applied",
                    "psf_ac_policy",
                    "psf_snr",
                    "psf_quality",
                    "psf_quality_fallback",
                    "psf_group_n",
                    "x_fit",
                    "y_fit",
                ):
                    if _col in _psf_idx.columns:
                        df[_col] = df["catalog_id"].map(_psf_idx[_col])
                _n_ok = int(pd.to_numeric(df.get("psf_fit_ok"), errors="coerce").fillna(0).astype(bool).sum())
            else:
                _n_ok = 0

            _psf_record.update({"n_fit": _n_fit, "n_ok": _n_ok})
            if _tid is not None:
                LOGGER.info(
                    "[ePSF] fitting %d/%d stars (targeted: variables+comps only), %d psf_fit_ok",
                    _n_fit,
                    _n_all,
                    _n_ok,
                )
            else:
                LOGGER.debug(
                    "[ePSF] frame PSF fit: %d/%d stars ok",
                    _n_ok,
                    _n_all,
                )
        except Exception as _psf_e:  # noqa: BLE001
            import traceback

            _exc_cls = type(_psf_e).__name__
            _exc_msg = str(_psf_e)
            _tb_tail = "\n".join(traceback.format_exc().splitlines()[-8:])
            _psf_record.update(
                {
                    "n_fit": _n_fit,
                    "n_ok": 0,
                    "exception_class": _exc_cls,
                    "exception_message": _exc_msg,
                    "traceback_tail": _tb_tail,
                }
            )
            LOGGER.warning(
                "[ePSF] per-frame PSF failed on frame %s: %s: %s",
                _frame_name or "?",
                _exc_cls,
                _exc_msg,
            )
            for _c in ("psf_flux", "psf_flux_err", "psf_chi2"):
                if _c not in df.columns:
                    df[_c] = float("nan")
            if "psf_fit_ok" not in df.columns:
                df["psf_fit_ok"] = False
            if "psf_quality" not in df.columns:
                df["psf_quality"] = ""
            if "psf_snr" not in df.columns:
                df["psf_snr"] = float("nan")
    else:
        for _c in ("psf_flux", "psf_flux_err", "psf_chi2"):
            if _c not in df.columns:
                df[_c] = float("nan")
        if "psf_fit_ok" not in df.columns:
            df["psf_fit_ok"] = False

    if _run_epsf:
        st["_psf_frame_record"] = _psf_record

    # -- Moffat PSF fit (Step 1 of two-step ePSF pipeline only) -----------------
    if _run_epsf and not bool(st.get("_psf_merge_only", False)):
        try:
            from config import AppConfig
            from psf_photometry import fit_moffat_psf_stars

            _cfg = AppConfig()

            _gain = float(st.get("gain", 1.0) or 1.0)
            _rn = float(st.get("read_noise", 10.0) or 10.0)
            if not math.isfinite(_gain) or _gain <= 0:
                _gain = 1.0
            if not math.isfinite(_rn) or _rn <= 0:
                _rn = 10.0
            d_arr = np.asarray(data, dtype=np.float32)
            _error_map = np.sqrt(np.abs(d_arr) / _gain + (_rn / _gain) ** 2).astype(np.float32)

            if not {"catalog_id", "x", "y"}.issubset(df.columns):
                raise ValueError("catalog missing catalog_id/x/y for Moffat PSF fit")

            try:
                _fwhm_px = float(hdr.get("VY_FWHM", 3.5) or 3.5)
            except Exception:  # noqa: BLE001
                _fwhm_px = 3.5
            if not math.isfinite(_fwhm_px) or _fwhm_px <= 0:
                _fwhm_px = 3.5

            _fit_df = df
            _tid = target_ids
            if _tid is None:
                _raw_tid = st.get("epsf_target_ids")
                if isinstance(_raw_tid, (set, frozenset, list, tuple)) and _raw_tid:
                    _tid = {str(x).strip() for x in _raw_tid if str(x).strip()}
            if _tid is not None:
                _cid_key = df["catalog_id"].fillna("").astype(str).str.strip().map(
                    lambda x: (
                        str(normalize_gaia_source_id(x)).strip()
                        if x and str(x).lower() not in ("nan", "none")
                        else ""
                    )
                )
                _fit_df = df.loc[_cid_key.isin(_tid)].copy()

            if len(_fit_df) > 0:
                _cols_want = ["catalog_id", "x", "y", "peak_dao", "dao_flux"]
                _pos = _fit_df.reindex(columns=_cols_want).copy()

                _moffat_df = fit_moffat_psf_stars(
                    frame_data=d_arr,
                    frame_hdr=hdr,
                    star_positions=_pos,
                    fwhm_guess_px=_fwhm_px,
                    error=_error_map,
                    saturate_limit_adu=float(getattr(_cfg, "saturate_limit_adu", 65000)),
                    peak_col="peak_dao",
                    chi2_limit=float(_MOFFAT_CHI2_LIMIT),  # WAVE-B STEP 6: hardcoded QC internal
                )

                # Moffat aperture correction
                if _moffat_df is not None and len(_moffat_df) > 0:
                    from psf_photometry import _compute_moffat_aperture_correction

                    _dao_arr = pd.to_numeric(
                        _fit_df.get("dao_flux", pd.Series(dtype=float)),
                        errors="coerce",
                    ).values
                    _mac_factor, _mac_n = _compute_moffat_aperture_correction(
                        _moffat_df,
                        _dao_arr,
                        chi2_limit=5.0,
                        min_flux_snr=50000.0,
                    )
                    if _mac_factor != 1.0:
                        log_event(
                            f"Moffat AC: factor={_mac_factor:.4f}, "
                            f"n_ref={_mac_n}"
                        )
                        for _col in ("moffat_flux", "moffat_flux_err"):
                            if _col in _moffat_df.columns:
                                _moffat_df[_col] = (
                                    pd.to_numeric(_moffat_df[_col], errors="coerce")
                                    * _mac_factor
                                )
                    _moffat_df["moffat_ac_factor"] = _mac_factor
                    _moffat_df["moffat_ac_n_used"] = _mac_n

                _moffat_cols = [c for c in _moffat_df.columns if str(c).startswith("moffat_")]
                if _moffat_cols and len(_moffat_df) > 0:
                    _moffat_df["_cid_key"] = _moffat_df["catalog_id"].astype(str).str.strip()
                    df["_cid_key"] = df["catalog_id"].astype(str).str.strip()
                    df = (
                        df.merge(_moffat_df[["_cid_key"] + _moffat_cols], on="_cid_key", how="left")
                        .drop(columns=["_cid_key"])
                    )
            else:
                LOGGER.debug("[PSF] Moffat skipped - no stars selected for fit")
        except Exception as _mex:  # noqa: BLE001
            log_event(f"Moffat PSF fit error (non-fatal): {_mex}")

    return df


# WAVE-B STEP 6 (HARDCODE): solver / QC internals, formerly AppConfig knobs. Fixed to their
# long-standing defaults (never tuned in config history); solver mechanics, not user tuning.
_MOFFAT_CHI2_LIMIT = 50.0                              # was cfg.moffat_chi2_limit (50.0)


def _sat_ctx_from_worker(st: dict[str, Any]) -> Any | None:
    raw = st.get("sat_diag_ctx_dict")
    if not raw:
        return None
    try:
        from sat_diag import PileupResult, SatDiagContext  # noqa: PLC0415

        pileup_raw = raw.get("pileup")
        pileup = PileupResult(**pileup_raw) if isinstance(pileup_raw, dict) else None
        d = dict(raw)
        d.pop("pileup", None)
        d["pileup"] = pileup
        return SatDiagContext(**{k: v for k, v in d.items() if k in SatDiagContext.__dataclass_fields__})
    except Exception:  # noqa: BLE001
        return None


def find_qc_metrics_csv(
    archive_path: Path | str,
    app_config: AppConfig | None = None,
    *,
    draft_id: int | None = None,
    db: VyvarDatabase | None = None,
) -> Path | None:
    """Find ``qc_metrics.csv`` under draft lights root (legacy fallback: processed/lights)."""
    from draft_provenance import draft_archive_root, resolve_draft_lights_root

    ap = draft_archive_root(Path(archive_path).expanduser())
    candidates: list[Path] = [
        resolve_draft_lights_root(ap, draft_id=draft_id, db=db) / "qc_metrics.csv",
        ap / "processed" / "lights" / "qc_metrics.csv",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def _bin2d_mean(arr: "np.ndarray", factor: int) -> "np.ndarray":
    """Mean-bin a 2D array by integer ``factor`` (>=1). Used to speed up reference scoring."""
    import numpy as np

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2 or int(factor) < 2:
        return arr
    f = int(factor)
    h, w = arr.shape
    h2, w2 = (h // f) * f, (w // f) * f
    if h2 < f or w2 < f:
        return arr
    a = arr[:h2, :w2].reshape(h2 // f, f, w2 // f, f)
    return np.mean(a, axis=(1, 3)).astype(np.float32)


def _dao_star_count_from_array(arr: "np.ndarray", *, fwhm_px: float = 3.0) -> int:
    """Count DAOStarFinder sources (same recipe as alignment star detection)."""
    import numpy as np
    from photutils.detection import DAOStarFinder

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        return 0
    finite = np.isfinite(arr)
    if not np.any(finite):
        return 0
    _, med, std = plain_mean_med_std(arr[finite], sigma=3.0, maxiters=5)
    std = float(std) if np.isfinite(std) else 0.0
    if std <= 0:
        return 0
    img2 = np.nan_to_num((arr - float(med)).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    try:
        finder = DAOStarFinder(
            fwhm=float(fwhm_px),
            threshold=max(3.0 * std, 1e-6),
            **DAO_STAR_FINDER_NO_ROUNDNESS_FILTER,
        )
        tbl = finder(img2)
    except Exception:  # noqa: BLE001
        return 0
    if tbl is None or len(tbl) == 0:
        return 0
    return int(len(tbl))


def _pick_reference_frame_by_star_count(files: list[Path]) -> tuple[Path, dict[str, int]]:
    """Choose the FITS with the most detected stars as alignment / plate-solve reference.

    Uses 2x2 binning + slightly smaller DAO FWHM for speed; preserves relative rankings.
    """
    import numpy as np

    scores: dict[str, int] = {}
    for fp in files:
        try:
            with fits.open(fp, memmap=False) as hdul:
                data = np.array(hdul[0].data, dtype=np.float32, copy=True)
            data_b = _bin2d_mean(data, 2)
            # 2x2 binning: FWHM v px ~ sips_dao_fwhm_px/2 (sirsie okno pre komu v rohoch).
            scores[str(fp)] = _dao_star_count_from_array(data_b, fwhm_px=2.5)
        except Exception:  # noqa: BLE001
            scores[str(fp)] = 0
    if not files:
        raise ValueError("no FITS files")
    if not scores or max(scores.values(), default=0) <= 0:
        return files[0], scores
    best_n = max(scores.values())
    for fp in files:
        if scores.get(str(fp), 0) == best_n:
            return fp, scores
    return files[0], scores


def _catalog_df_cap_brightest_by_mag(df: pd.DataFrame, max_rows: int | None = None) -> pd.DataFrame:
    """Keep at most ``max_rows`` catalog rows, brightest first (lowest ``mag``)."""
    if df is None or getattr(df, "empty", True):
        return df
    try:
        cap = int(max_rows) if max_rows is not None else int(AppConfig().catalog_query_max_rows)
    except Exception:  # noqa: BLE001
        cap = 50_000
    cap = max(1000, min(500_000, cap))
    if len(df) <= cap:
        return df
    out = df.copy()
    if "mag" not in out.columns:
        return out.iloc[:cap].copy().reset_index(drop=True)
    m = pd.to_numeric(out["mag"], errors="coerce")
    out = out.assign(_vyvar_mag_sort=m)
    out = out.sort_values("_vyvar_mag_sort", na_position="last").head(int(cap))
    return out.drop(columns=["_vyvar_mag_sort"], errors="ignore").reset_index(drop=True)


def _query_gaia_local(
    *,
    center: SkyCoord,
    radius_deg: float,
    gaia_db_path: Path | None,
    max_mag: float | None = None,
    focal_mm_for_log: float | None = None,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Query **local Gaia DR3 SQLite** for the field; return unified dataframe (ICRS deg)."""
    try:
        _ra_l = float(center.icrs.ra.deg)
        _de_l = float(center.icrs.dec.deg)
    except Exception:  # noqa: BLE001
        _ra_l, _de_l = float("nan"), float("nan")
    _f_l = f"{float(focal_mm_for_log):g}" if focal_mm_for_log is not None else "?"
    log_event(
        f"CATALOG SEARCH (GAIA local): Ra={_ra_l}, Dec={_de_l}, Radius={float(radius_deg):.2f} deg (pre {_f_l}mm)"
    )
    if gaia_db_path is None:
        return pd.DataFrame()
    gp = Path(gaia_db_path).expanduser().resolve()
    if not gp.is_file():
        return pd.DataFrame()

    ra_min = float(_ra_l) - float(radius_deg)
    ra_max = float(_ra_l) + float(radius_deg)
    de_min = float(_de_l) - float(radius_deg)
    de_max = float(_de_l) + float(radius_deg)
    # query_local_gaia: mag_limit=None => no g_mag SQL cap; pass max_mag only when set
    # so MASTERSTAR / cone export honor faintest_mag_limit in SQL (not only a redundant pandas filter).
    _ql_kw: dict[str, Any] = {
        "ra_min": ra_min,
        "ra_max": ra_max,
        "dec_min": de_min,
        "dec_max": de_max,
    }
    if max_mag is not None:
        try:
            _mm = float(max_mag)
            if math.isfinite(_mm) and _mm > 0:
                _ql_kw["mag_limit"] = _mm
        except (TypeError, ValueError):
            pass
    # SQLite uses a square ra/dec box + ORDER BY g_mag LIMIT. For wide boxes (optics floor vs WCS) the 100k
    # brightest stars are often far from the field center, so cone stars never enter the result set.
    _cap_out: int | None = None
    if max_rows is not None:
        try:
            _mr0 = int(max_rows)
            if _mr0 > 0:
                _cap_out = _mr0
        except (TypeError, ValueError):
            _cap_out = None
    _sql_fetch = _cap_out
    if _cap_out is not None and float(radius_deg) > 6.0:
        _af = max(1.0, (float(radius_deg) / 5.5) ** 2)
        # Oversample for cone cut; cap keeps SQLite practical (ORDER BY g_mag on huge boxes is costly).
        _sql_fetch = min(800_000, max(_cap_out, int(_cap_out * _af * 2.5)))
    if _sql_fetch is not None:
        _ql_kw["max_rows"] = int(_sql_fetch)
    rows = query_local_gaia(gp, **_ql_kw)
    if not rows:
        return pd.DataFrame()
    df0 = pd.DataFrame(rows)
    if "bp_rp" not in df0.columns and "bp_mag" in df0.columns and "rp_mag" in df0.columns:
        df0["bp_rp"] = pd.to_numeric(df0["bp_mag"], errors="coerce") - pd.to_numeric(
            df0["rp_mag"], errors="coerce"
        )
    df = df0.rename(
        columns={"source_id": "catalog_id", "ra": "ra_deg", "dec": "dec_deg", "g_mag": "mag", "bp_rp": "bp_rp"}
    )
    df["catalog"] = "GAIA_DR3"
    # Great-circle cone cut (query box is square; LIMIT is isotropic in mag, not in radius).
    _raq = pd.to_numeric(df["ra_deg"], errors="coerce")
    _deq = pd.to_numeric(df["dec_deg"], errors="coerce")
    _okq = _raq.notna() & _deq.notna()
    if bool(_okq.any()):
        sub = df.loc[_okq].copy()
        _coo_q = SkyCoord(
            ra=pd.to_numeric(sub["ra_deg"], errors="coerce").astype(float).to_numpy() * u.deg,
            dec=pd.to_numeric(sub["dec_deg"], errors="coerce").astype(float).to_numpy() * u.deg,
            frame="icrs",
        )
        _inner = center.separation(_coo_q).deg <= float(radius_deg) + 1e-9
        df = sub.loc[_inner].reset_index(drop=True)
    if max_mag is not None and "mag" in df.columns:
        m = pd.to_numeric(df["mag"], errors="coerce")
        df = df[(m.notna()) & (m <= float(max_mag))].copy()
    df = df.reset_index(drop=True)
    # Gaia provides BP-RP; do not map it into B-V (different color index).
    if "b_v" not in df.columns:
        df = df.copy()
        df["b_v"] = np.nan
    if "bp_rp" in df.columns:
        df["bp_rp"] = pd.to_numeric(df["bp_rp"], errors="coerce")
    return _catalog_df_cap_brightest_by_mag(df, max_rows=_cap_out)


def _query_vsx_local(
    *,
    center: SkyCoord,
    radius_deg: float,
    vsx_db_path: Path | None,
    max_rows: int | None = None,
    require_db: bool = False,
) -> pd.DataFrame:
    """Query **local VSX SQLite** for the field; same kuzel ako Gaia (najprv obdlznik, potom great-circle orez)."""
    from database import count_vsx_local_rows, require_vsx_local_db_path

    if require_db:
        vp = require_vsx_local_db_path(vsx_db_path)
        n_total = count_vsx_local_rows(vp)
    elif vsx_db_path is None:
        return pd.DataFrame()
    else:
        vp = Path(vsx_db_path).expanduser().resolve()
        if not vp.is_file():
            return pd.DataFrame()
        n_total = count_vsx_local_rows(vp)
    try:
        _ra_l = float(center.icrs.ra.deg)
        _de_l = float(center.icrs.dec.deg)
    except Exception:  # noqa: BLE001
        if require_db:
            raise
        return pd.DataFrame()
    ra_min = float(_ra_l) - float(radius_deg)
    ra_max = float(_ra_l) + float(radius_deg)
    de_min = float(_de_l) - float(radius_deg)
    de_max = float(_de_l) + float(radius_deg)
    _cap = max_rows
    if _cap is None:
        try:
            _cap = int(AppConfig().catalog_query_max_rows)
        except Exception:  # noqa: BLE001
            _cap = 500_000
        _cap = max(10_000, min(500_000, int(_cap)))
    if float(radius_deg) > 6.0:
        _af = max(1.0, (float(radius_deg) / 5.5) ** 2)
        _cap = min(800_000, max(int(_cap), int(_cap * _af * 2.5)))

    rows = query_local_vsx(
        vp,
        ra_min=ra_min,
        ra_max=ra_max,
        dec_min=de_min,
        dec_max=de_max,
        max_rows=int(_cap),
    )
    if not rows:
        if require_db:
            log_event(
                f"VSX cone=0 on {vp} ({n_total} total rows) - field genuinely empty "
                f"(r~{float(radius_deg):.3f} deg, Ra={_ra_l:.4f}, Dec={_de_l:.4f})"
            )
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "ra_deg" not in df.columns or "dec_deg" not in df.columns:
        if require_db:
            from database import VSXCatalogError

            raise VSXCatalogError(f"VSX query returned unexpected columns from {vp}.")
        return pd.DataFrame()
    _raq = pd.to_numeric(df["ra_deg"], errors="coerce")
    _deq = pd.to_numeric(df["dec_deg"], errors="coerce")
    _okq = _raq.notna() & _deq.notna()
    if not bool(_okq.any()):
        if require_db:
            log_event(
                f"VSX cone=0 on {vp} ({n_total} total rows) - field genuinely empty "
                f"(r~{float(radius_deg):.3f} deg, Ra={_ra_l:.4f}, Dec={_de_l:.4f})"
            )
        return pd.DataFrame()
    sub = df.loc[_okq].copy()
    _coo_q = SkyCoord(
        ra=pd.to_numeric(sub["ra_deg"], errors="coerce").astype(float).to_numpy() * u.deg,
        dec=pd.to_numeric(sub["dec_deg"], errors="coerce").astype(float).to_numpy() * u.deg,
        frame="icrs",
    )
    _inner = center.separation(_coo_q).deg <= float(radius_deg) + 1e-9
    out = sub.loc[_inner].reset_index(drop=True)
    if require_db and out.empty:
        log_event(
            f"VSX cone=0 on {vp} ({n_total} total rows) - field genuinely empty "
            f"(r~{float(radius_deg):.3f} deg, Ra={_ra_l:.4f}, Dec={_de_l:.4f})"
        )
    elif not out.empty:
        try:
            log_event(
                f"CATALOG SEARCH (VSX local): {len(out)} zdrojov v kuzeli r~{float(radius_deg):.3f} deg "
                f"(Ra={_ra_l:.4f}, Dec={_de_l:.4f})"
            )
        except Exception:  # noqa: BLE001
            pass
    return out


def _query_exoplanet_local(
    *,
    center: SkyCoord,
    radius_deg: float,
    exoplanet_db_path: Path | None,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Query local exoplanet host SQLite for the field; box query + great-circle cone filter."""
    if exoplanet_db_path is None:
        return pd.DataFrame()
    from database import require_exoplanet_local_db_path  # noqa: PLC0415

    ep = require_exoplanet_local_db_path(exoplanet_db_path)
    try:
        _ra_l = float(center.icrs.ra.deg)
        _de_l = float(center.icrs.dec.deg)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    ra_min = float(_ra_l) - float(radius_deg)
    ra_max = float(_ra_l) + float(radius_deg)
    de_min = float(_de_l) - float(radius_deg)
    de_max = float(_de_l) + float(radius_deg)
    _cap = max_rows
    if _cap is None:
        try:
            _cap = int(AppConfig().catalog_query_max_rows)
        except Exception:  # noqa: BLE001
            _cap = 500_000
        _cap = max(10_000, min(500_000, int(_cap)))

    rows = query_local_exoplanet(
        ep,
        ra_min=ra_min,
        ra_max=ra_max,
        dec_min=de_min,
        dec_max=de_max,
        max_rows=int(_cap),
    )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "ra_deg" not in df.columns or "dec_deg" not in df.columns:
        return pd.DataFrame()
    _raq = pd.to_numeric(df["ra_deg"], errors="coerce")
    _deq = pd.to_numeric(df["dec_deg"], errors="coerce")
    _okq = _raq.notna() & _deq.notna()
    if not bool(_okq.any()):
        return pd.DataFrame()
    sub = df.loc[_okq].copy()
    _coo_q = SkyCoord(
        ra=pd.to_numeric(sub["ra_deg"], errors="coerce").astype(float).to_numpy() * u.deg,
        dec=pd.to_numeric(sub["dec_deg"], errors="coerce").astype(float).to_numpy() * u.deg,
        frame="icrs",
    )
    _inner = center.separation(_coo_q).deg <= float(radius_deg) + 1e-9
    out = sub.loc[_inner].reset_index(drop=True)
    try:
        log_event(
            f"CATALOG SEARCH (exoplanet local): {len(out)} hostov v kuzeli r~{float(radius_deg):.3f} deg "
            f"(Ra={_ra_l:.4f}, Dec={_de_l:.4f})"
        )
    except Exception:  # noqa: BLE001
        pass
    return out


def _exo_host_annotation_arrays(
    det_coords: SkyCoord,
    exo_df: pd.DataFrame,
    max_sep_arcsec: float,
    *,
    frame_name: str = "",
) -> tuple[dict[str, Any], list[str]]:
    """Nearest exoplanet host per detection within ``max_sep_arcsec`` (informational only)."""
    import numpy as np
    from astropy.coordinates import search_around_sky

    n = len(det_coords)
    out: dict[str, Any] = {
        "exo_host_obj_id": np.array([""] * n, dtype=object),
        "exo_host_name": np.array([""] * n, dtype=object),
        "exo_cat_source": np.array([""] * n, dtype=object),
        "exo_disposition": np.array([""] * n, dtype=object),
        "exo_match_sep_arcsec": np.full(n, np.nan, dtype=np.float64),
    }
    warnings_out: list[str] = []
    if exo_df is None or exo_df.empty or n == 0:
        return out, warnings_out

    exc = SkyCoord(
        ra=np.asarray(exo_df["ra_deg"], dtype=float) * u.deg,
        dec=np.asarray(exo_df["dec_deg"], dtype=float) * u.deg,
        frame="icrs",
    )
    max_sep = float(max_sep_arcsec)
    idx_nearest, sep2d, _ = det_coords.match_to_catalog_sky(exc)
    sep_arc = np.asarray(sep2d.to(u.arcsec).value, dtype=np.float64)
    hit = np.isfinite(sep_arc) & (sep_arc <= max_sep)

    try:
        idx_d, idx_e, sep_a, _ = search_around_sky(det_coords, exc, max_sep * u.arcsec)
        counts: dict[int, int] = {}
        for i_d in np.asarray(idx_d, dtype=np.int64):
            counts[int(i_d)] = counts.get(int(i_d), 0) + 1
        for i_d, cnt in counts.items():
            if cnt > 1:
                msg = (
                    f"[EXO MATCH] {cnt} exoplanet hosts within {max_sep:g} arcsec of detection "
                    f"index {i_d}"
                    + (f" ({frame_name})" if frame_name else "")
                )
                warnings_out.append(msg)
                LOGGER.warning(msg)
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("[EXO MATCH] ambiguity search skipped: %s", exc)

    for i in np.flatnonzero(hit):
        j = int(idx_nearest[i])
        if j < 0 or j >= len(exo_df):
            continue
        row = exo_df.iloc[j]
        out["exo_host_obj_id"][i] = str(row.get("obj_id", "") or "").strip()
        out["exo_host_name"][i] = str(row.get("host_name", "") or "").strip()
        out["exo_cat_source"][i] = str(row.get("cat_source", "") or "").strip()
        out["exo_disposition"][i] = str(row.get("disposition", "") or "").strip()
        out["exo_match_sep_arcsec"][i] = float(sep_arc[i])

    return out, warnings_out


def _slice_exo_annotation(exo_ann: dict[str, Any], keep: Any) -> dict[str, Any]:
    import numpy as np

    k = np.asarray(keep, dtype=bool)
    return {col: np.asarray(exo_ann[col])[k] for col in _EXO_HOST_ANNOTATION_COLUMNS}


def _apply_exo_host_columns_to_proc_df(
    df: pd.DataFrame,
    hdr: fits.Header,
    data_shape: tuple[int, int],
    st: dict[str, Any],
    *,
    frame_name: str = "",
) -> pd.DataFrame:
    """Add informational ``exo_*`` columns when local DB path is configured.

    ``detect_stars_and_match_catalog`` already annotates; this covers the MASTERSTAR fast path.
    """
    if df is None or df.empty:
        return df
    if any(c in df.columns for c in _EXO_HOST_ANNOTATION_COLUMNS):
        return df

    _exo_path: Path | None = None
    try:
        _exs = str(st.get("exoplanet_local_db_path") or "").strip()
        if _exs:
            _exo_path = Path(_exs).expanduser().resolve()
    except Exception:  # noqa: BLE001
        _exo_path = None
    if _exo_path is None or not _exo_path.is_file():
        return df

    exo_max = 3.0
    try:
        exo_max = float(st.get("exoplanet_match_max_sep_arcsec", 3.0))
        if not math.isfinite(exo_max):
            exo_max = 3.0
    except (TypeError, ValueError):
        exo_max = 3.0
    exo_max = max(0.5, min(30.0, float(exo_max)))

    if "ra_deg" not in df.columns or "dec_deg" not in df.columns:
        return df

    import numpy as np

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        wcs_obj = WCS(hdr)
    if not wcs_obj.has_celestial:
        return df

    h, wpx = int(data_shape[0]), int(data_shape[1])
    _fov_hint = st.get("plate_solve_fov_deg")
    try:
        if _fov_hint is not None and not math.isfinite(float(_fov_hint)):
            _fov_hint = None
    except (TypeError, ValueError):
        _fov_hint = None
    if _fov_hint is None:
        try:
            _fov_hint = resolve_plate_solve_fov_deg_hint(
                hdr,
                h,
                wpx,
                database_path=st.get("database_path"),
                equipment_id=st.get("equipment_id"),
                draft_id=st.get("draft_id"),
            )
        except Exception:  # noqa: BLE001
            _fov_hint = None
    if _fov_hint is None:
        try:
            _fov_hint = float(AppConfig().plate_solve_fov_deg)
        except Exception:  # noqa: BLE001
            _fov_hint = None

    center, radius_deg = _effective_field_catalog_cone_radius_deg(
        wcs_obj, h, wpx, _fov_hint, fits_header=hdr
    )
    exo_df = _query_exoplanet_local(
        center=center,
        radius_deg=radius_deg,
        exoplanet_db_path=_exo_path,
    )

    ra = pd.to_numeric(df["ra_deg"], errors="coerce").to_numpy(dtype=np.float64)
    de = pd.to_numeric(df["dec_deg"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.any(np.isfinite(ra) & np.isfinite(de)):
        return df

    det_coords = SkyCoord(ra=ra * u.deg, dec=de * u.deg, frame="icrs")
    exo_ann, _ = _exo_host_annotation_arrays(
        det_coords,
        exo_df if exo_df is not None else pd.DataFrame(),
        exo_max,
        frame_name=frame_name,
    )
    out = df.copy()
    for col in _EXO_HOST_ANNOTATION_COLUMNS:
        out[col] = exo_ann[col]
    return out


def _build_exoplanet_promotion_rows_from_masterstars(
    master_df: pd.DataFrame,
    hdr: fits.Header,
    cfg: AppConfig,
    *,
    frame_w_px: int,
    frame_h_px: int,
    margin_px: float = 50.0,
) -> pd.DataFrame:
    """Promote masterstars rows with exoplanet host match within configured separation."""
    _exs = str(cfg.exoplanet_local_db_path or "").strip()
    if not _exs:
        return pd.DataFrame()
    from database import require_exoplanet_local_db_path  # noqa: PLC0415

    try:
        _exo_path = require_exoplanet_local_db_path(_exs)
    except Exception:
        raise
    if master_df is None or master_df.empty:
        return pd.DataFrame()
    if "catalog_id" not in master_df.columns or "ra_deg" not in master_df.columns or "dec_deg" not in master_df.columns:
        return pd.DataFrame()

    import numpy as np

    from gaia_catalog_id import catalog_id_series_for_masterstars_export, masterstar_row_gaia_key

    m = master_df.copy()
    if "catalog_id" in m.columns:
        m["catalog_id"] = catalog_id_series_for_masterstars_export(m)
    cid_s = m["catalog_id"].fillna("").astype(str).str.strip()
    ok_cid = cid_s.ne("") & ~cid_s.str.lower().isin({"nan", "none"})
    ra = pd.to_numeric(m["ra_deg"], errors="coerce")
    de = pd.to_numeric(m["dec_deg"], errors="coerce")
    ok_sky = ra.notna() & de.notna()
    m = m.loc[ok_cid & ok_sky].copy()
    if m.empty:
        return pd.DataFrame()

    if "x" in m.columns and "y" in m.columns:
        xn = pd.to_numeric(m["x"], errors="coerce")
        yn = pd.to_numeric(m["y"], errors="coerce")
        in_frame = xn.between(-float(margin_px), float(frame_w_px) + float(margin_px)) & yn.between(
            -float(margin_px), float(frame_h_px) + float(margin_px)
        )
        m = m.loc[in_frame].copy()
    if m.empty:
        return pd.DataFrame()

    exo_max = 3.0
    try:
        exo_max = float(cfg.exoplanet_match_max_sep_arcsec)
        if not math.isfinite(exo_max):
            exo_max = 3.0
    except (TypeError, ValueError):
        exo_max = 3.0
    exo_max = max(0.5, min(30.0, float(exo_max)))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        wcs_obj = WCS(hdr)
    if not wcs_obj.has_celestial:
        return pd.DataFrame()
    try:
        _fov_hint = float(cfg.plate_solve_fov_deg)
        if not math.isfinite(_fov_hint):
            _fov_hint = None
    except (TypeError, ValueError):
        _fov_hint = None
    center, radius_deg = _effective_field_catalog_cone_radius_deg(
        wcs_obj, int(frame_h_px), int(frame_w_px), _fov_hint, fits_header=hdr
    )
    exo_df = _query_exoplanet_local(
        center=center,
        radius_deg=radius_deg,
        exoplanet_db_path=_exo_path,
    )
    n_hosts_in_field = int(len(exo_df)) if exo_df is not None and not exo_df.empty else 0

    ra_arr = pd.to_numeric(m["ra_deg"], errors="coerce").to_numpy(dtype=np.float64)
    de_arr = pd.to_numeric(m["dec_deg"], errors="coerce").to_numpy(dtype=np.float64)
    det_coords = SkyCoord(ra=ra_arr * u.deg, dec=de_arr * u.deg, frame="icrs")
    exo_ann, _ = _exo_host_annotation_arrays(
        det_coords,
        exo_df if exo_df is not None else pd.DataFrame(),
        exo_max,
    )

    exo_ids = np.asarray(exo_ann["exo_host_obj_id"], dtype=object)
    exo_sep = np.asarray(exo_ann["exo_match_sep_arcsec"], dtype=np.float64)
    promote_mask = np.zeros(len(m), dtype=bool)
    for i in range(len(m)):
        oid = str(exo_ids[i] or "").strip()
        sep = float(exo_sep[i]) if math.isfinite(float(exo_sep[i])) else float("nan")
        promote_mask[i] = bool(oid) and math.isfinite(sep) and sep <= exo_max
    if not bool(np.any(promote_mask)):
        log_event(
            f"[EXO TARGET] funnel: hosts_in_field={n_hosts_in_field} masterstars_in_frame={len(m)} "
            f"promoted=0 sep_max={exo_max:g} arcsec"
        )
        return pd.DataFrame()

    sub = m.loc[promote_mask].copy()
    prom_idx = np.where(promote_mask)[0]

    from gaia_catalog_id import masterstar_row_gaia_key

    rows: list[dict[str, Any]] = []
    for j, (_, row) in enumerate(sub.iterrows()):
        ii = int(prom_idx[j])
        cid_norm = masterstar_row_gaia_key(row)
        if not cid_norm:
            continue
        host_name = str(exo_ann["exo_host_name"][ii] or "").strip()
        obj_id = str(exo_ann["exo_host_obj_id"][ii] or "").strip()
        disp = str(exo_ann["exo_disposition"][ii] or "").strip()
        src = str(exo_ann["exo_cat_source"][ii] or "").strip()
        try:
            mag_v = float(pd.to_numeric(row.get("mag", row.get("phot_g_mean_mag")), errors="coerce"))
        except (TypeError, ValueError):
            mag_v = float("nan")
        if not math.isfinite(mag_v):
            mag_v = float("nan")
        rows.append(
            {
                "name": host_name or obj_id,
                "catalog_id": cid_norm,
                "catalog": "EXOPLANET",
                "ra_deg": float(row["ra_deg"]),
                "dec_deg": float(row["dec_deg"]),
                "priority": 2,
                "notes": f"{src} {disp}".strip(),
                "vsx_name": "",
                "vsx_type": "",
                "vsx_period": np.nan,
                "x": pd.to_numeric(row.get("x"), errors="coerce"),
                "y": pd.to_numeric(row.get("y"), errors="coerce"),
                "mag": mag_v,
                "zone": str(row.get("zone", "") or "").strip().lower(),
                "gaia_match_arcsec": float(exo_sep[ii]),
                "gaia_match_quality": "good",
                "gaia_match_source": "masterstars_exo",
                "vsx_mag_max": np.nan,
                "exo_host_obj_id": obj_id,
                "exo_host_name": host_name,
                "exo_cat_source": src,
                "exo_disposition": disp,
                "exo_match_sep_arcsec": float(exo_sep[ii]),
                "target_origin": "EXOPLANET",
            }
        )
    if not rows:
        log_event(
            f"[EXO TARGET] funnel: hosts_in_field={n_hosts_in_field} masterstars_in_frame={len(m)} "
            f"promoted=0 sep_max={exo_max:g} arcsec (no Gaia catalog_id on matched rows)"
        )
        return pd.DataFrame()
    log_event(
        f"[EXO TARGET] funnel: hosts_in_field={n_hosts_in_field} masterstars_in_frame={len(m)} "
        f"promoted={len(rows)} sep_max={exo_max:g} arcsec"
    )
    log_event(f"[EXO TARGET] {len(rows)} exoplanet host(s) promoted from masterstars (<={exo_max:g} arcsec)")
    return pd.DataFrame(rows)


def inv_sat_limit_peak_test_adu() -> tuple[float, str]:
    """Catalog-zone peak-test (INV-SAT-LIMIT). One authority for zone and per-frame.

    Hard container clip is ``SAT_LIMIT_CONTAINER_CLIP_ADU`` (named separately;
    not used as the clean-frame test). This returns the peak-test used for
    ``zone==saturated``: currently 0.80 x 65535 = 52428 ADU when the D1-2 knee
    is unmeasured. n_stack is not applied here: per-frame CSVs are one exposure.
    """
    peak = float(SAT_LIMIT_CONTAINER_CLIP_ADU) * float(SAT_LIMIT_NO_KNEE_FRAC)
    return peak, SAT_LIMIT_PEAK_TEST_SOURCE


def _box_peak_max_adu(data: "np.ndarray", x: float, y: float, half: int = 3) -> float:
    """Maximum pixel value in ``(2*half+1)^2`` box around ``(x,y)`` on the **original** image (linear units)."""
    import numpy as np

    arr = np.asarray(data)
    if arr.ndim != 2:
        return float("nan")
    h, w = arr.shape
    xi = int(round(float(x)))
    yi = int(round(float(y)))
    y0, y1 = max(0, yi - half), min(h, yi + half + 1)
    x0, x1 = max(0, xi - half), min(w, xi + half + 1)
    if y0 >= y1 or x0 >= x1:
        return float("nan")
    return float(np.nanmax(arr[y0:y1, x0:x1]))


def _box_peaks_at_centroids(
    arr: "np.ndarray",
    x: "np.ndarray",
    y: "np.ndarray",
    *,
    half: int = 3,
) -> "np.ndarray":
    """Maximum ADU in each ``(2*half+1)^2`` box centred on ``round(x), round(y)`` (vectorized).

    Used for per-star saturation on thousands of DAO detections; matches ``_box_peak_max_adu`` on
    interior pixels. Falls back to a Python loop if SciPy is unavailable.
    """
    import numpy as np

    xa = np.asarray(x, dtype=np.float64).reshape(-1)
    ya = np.asarray(y, dtype=np.float64).reshape(-1)
    n = int(xa.size)
    if n == 0:
        return np.zeros([], dtype=np.float64)
    if n != int(ya.size):
        raise ValueError("x and y must have the same length")

    try:
        from scipy.ndimage import maximum_filter  # type: ignore
    except ImportError:
        return np.asarray(
            [_box_peak_max_adu(arr, float(xa[i]), float(ya[i]), half=half) for i in range(n)],
            dtype=np.float64,
        )

    a = np.asarray(np.nan_to_num(np.asarray(arr, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0))
    size = int(2 * int(half) + 1)
    mf = maximum_filter(a, size=size, mode="nearest")
    xi = np.rint(xa).astype(np.intp, copy=False)
    yi = np.rint(ya).astype(np.intp, copy=False)
    h, w = mf.shape
    xi = np.clip(xi, 0, w - 1)
    yi = np.clip(yi, 0, h - 1)
    return np.asarray(mf[yi, xi], dtype=np.float64)


def _icrs_deg_to_unitxyz(ra_deg: "np.ndarray", dec_deg: "np.ndarray") -> "np.ndarray":
    """ICRS degrees -> unit direction vectors on the celestial sphere (N,3)."""
    import numpy as np

    ra = np.radians(np.asarray(ra_deg, dtype=np.float64).ravel())
    de = np.radians(np.asarray(dec_deg, dtype=np.float64).ravel())
    cd = np.cos(de)
    return np.column_stack([cd * np.cos(ra), cd * np.sin(ra), np.sin(de)])


def _chord_to_arcsec(dist_chord: "np.ndarray") -> "np.ndarray":
    """Chord length between unit sphere points (0...2) -> great-circle separation in arcseconds."""
    import numpy as np

    d = np.asarray(dist_chord, dtype=np.float64)
    half = np.clip(d * 0.5, 0.0, 1.0)
    return np.degrees(2.0 * np.arcsin(half)) * 3600.0


def build_ucac_catalog_kdtree(cat_df: pd.DataFrame) -> tuple[Any, "np.ndarray"] | None:
    """Build a SciPy ``cKDTree`` on finite ``ra_deg``/``dec_deg`` rows and row indices into ``cat_df``.

    Read-only ``query`` calls are thread-safe across workers sharing the same tree.
    """
    import numpy as np

    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return None
    if cat_df is None or cat_df.empty or "ra_deg" not in cat_df.columns or "dec_deg" not in cat_df.columns:
        return None
    ra = np.asarray(pd.to_numeric(cat_df["ra_deg"], errors="coerce"), dtype=np.float64)
    de = np.asarray(pd.to_numeric(cat_df["dec_deg"], errors="coerce"), dtype=np.float64)
    m = np.isfinite(ra) & np.isfinite(de)
    if not np.any(m):
        return None
    orig_idx = np.nonzero(m)[0].astype(np.int64)
    xyz = _icrs_deg_to_unitxyz(ra[m], de[m])
    return cKDTree(xyz), orig_idx


def nearest_sky_nn_kdtree(
    tree: Any,
    det_ra_deg: "np.ndarray",
    det_dec_deg: "np.ndarray",
) -> tuple["np.ndarray", "np.ndarray"]:
    """Nearest catalog point on the sphere for each detection (k=1, same idea as ``match_to_catalog_sky``).

    Returns ``(idx_compact, sep_arcsec)`` where ``idx_compact`` indexes the finite subset used to build
    ``tree``, or ``-1`` if invalid.
    """
    import numpy as np

    det_xyz = _icrs_deg_to_unitxyz(det_ra_deg, det_dec_deg)
    dist, idx = tree.query(det_xyz, k=1)
    sep = _chord_to_arcsec(dist)
    idx_a = np.asarray(idx, dtype=np.int64).ravel()
    dist_a = np.asarray(dist, dtype=np.float64).ravel()
    ntree = int(getattr(tree, "n", 0))
    bad = ~np.isfinite(dist_a) | (idx_a < 0) | (idx_a >= max(ntree, 1))
    sep = np.asarray(sep, dtype=np.float64).ravel()
    sep[bad] = np.inf
    idx_a = idx_a.copy()
    idx_a[bad] = -1
    return idx_a, sep


def _all_pix2world_icrs_deg(wcs_obj: WCS, x: "np.ndarray", y: "np.ndarray") -> tuple["np.ndarray", "np.ndarray"]:
    """Vectorized pixel -> world (degrees) for celestial axes; single WCS call (no per-star Python loops)."""
    import numpy as np

    xa = np.asarray(x, dtype=np.float64).ravel()
    ya = np.asarray(y, dtype=np.float64).ravel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        world0, world1 = wcs_obj.all_pix2world(xa, ya, 0)
    return (
        np.asarray(world0, dtype=np.float64).ravel(),
        np.asarray(world1, dtype=np.float64).ravel(),
    )


def _saturated_core_plateau_vectorized(
    data: "np.ndarray",
    x: "np.ndarray",
    y: "np.ndarray",
    *,
    half_inner: int = 1,
    plateau_rel: float = 0.996,
    min_plateau_pixels: int = 5,
) -> "np.ndarray":
    """Vectorized saturated-core plateau over ``(x,y)`` centroids (3x3 patch per star)."""
    import numpy as np

    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2:
        return np.zeros(len(np.asarray(x).ravel()), dtype=bool)
    h, w = arr.shape
    n = int(np.asarray(x, dtype=np.float64).size)
    yi = np.rint(np.asarray(y, dtype=np.float64).ravel()).astype(np.int32, copy=False)
    xi = np.rint(np.asarray(x, dtype=np.float64).ravel()).astype(np.int32, copy=False)
    hi = max(0, int(half_inner))
    offs = np.arange(-hi, hi + 1, dtype=np.int32)
    yy = yi.reshape(-1, 1, 1) + offs.reshape(1, -1, 1)
    xx = xi.reshape(-1, 1, 1) + offs.reshape(1, 1, -1)
    yy = np.clip(yy, 0, h - 1)
    xx = np.clip(xx, 0, w - 1)
    patches = arr[yy, xx]
    pmax = np.nanmax(patches.reshape(n, -1), axis=1)
    thr = pmax * float(plateau_rel)
    thr3 = thr.reshape(-1, 1, 1)
    n_high = np.sum(np.isfinite(patches) & (patches >= thr3), axis=(1, 2))
    return (n_high >= int(min_plateau_pixels)) & np.isfinite(pmax) & (pmax > 0)


def _vectorized_star_saturation_columns(
    arr: "np.ndarray",
    x: "np.ndarray",
    y: "np.ndarray",
    *,
    sat_limit: float | None,
    sat_frac: float,
    peak_dao: "np.ndarray",
    peak_max_adu: "np.ndarray",
) -> dict[str, "np.ndarray"]:
    """Per-star saturation flags as column arrays (vectorized over all centroids)."""
    import numpy as np

    n = int(np.asarray(x, dtype=np.float64).size)
    pmax = np.asarray(peak_max_adu, dtype=np.float64).reshape(-1)
    lim = sat_limit
    sf = float(sat_frac)
    pdv = np.asarray(peak_dao, dtype=np.float64).reshape(-1)

    if lim is not None:
        sat_peak = np.isfinite(pmax) & (pmax >= float(lim) * sf)
        plateau_skip_thr = float(lim) * sf * 0.55
        need_plateau = np.isfinite(pmax) & (pmax >= plateau_skip_thr)
    else:
        sat_peak = np.zeros(n, dtype=bool)
        need_plateau = np.ones(n, dtype=bool)

    pl_full = _saturated_core_plateau_vectorized(arr, x, y, half_inner=1, plateau_rel=0.996, min_plateau_pixels=5)
    if lim is None:
        sat_plateau = pl_full
    else:
        sat_plateau = np.where(need_plateau, pl_full, False)

    likely = sat_peak | sat_plateau
    peak_dao_col = pdv.copy()
    peak_dao_col[~np.isfinite(peak_dao_col)] = np.nan
    peak_max_col = pmax.copy()
    peak_max_col[~np.isfinite(peak_max_col)] = np.nan
    sl = np.full(n, np.nan, dtype=np.float64)
    if lim is not None:
        sl[:] = float(lim)
    return {
        "peak_dao": peak_dao_col,
        "peak_max_adu": peak_max_col,
        "saturate_limit_adu": sl,
        "saturated_from_peak": sat_peak,
        "saturated_plateau": sat_plateau,
        "likely_saturated": likely,
        "photometry_ok": ~likely,
    }


def _proc_sat_block_for_csv(sat_block: dict[str, Any]) -> tuple[dict[str, Any], int, int]:
    """Omit obsolete proc columns; return (csv_block, n_saturated_from_peak, n_saturated_plateau) for meta."""
    import numpy as np

    pk = sat_block.get("saturated_from_peak")
    pl = sat_block.get("saturated_plateau")
    n_pk = int(np.asarray(pk, dtype=bool).sum()) if pk is not None else 0
    n_pl = int(np.asarray(pl, dtype=bool).sum()) if pl is not None else 0
    csv_block = {
        k: v for k, v in sat_block.items() if k not in ("saturated_from_peak", "saturated_plateau")
    }
    return csv_block, n_pk, n_pl


def _fits_header_first_positive_float(hdr: fits.Header, keys: tuple[str, ...]) -> float | None:
    for k in keys:
        if k not in hdr:
            continue
        try:
            v = float(hdr[k])
            if math.isfinite(v) and v > 0:
                return v
        except (TypeError, ValueError):
            continue
    return None


def _gaia_catalog_cone_radius_optics_floor_deg(
    hdr: fits.Header | None,
    *,
    naxis1: int,
    naxis2: int,
    plate_solve_fov_fallback_deg: float,
) -> float:
    """Minimum Gaia cone radius from FOCALLEN + PIXSIZE + binning (ignores flawed linear WCS at chip edges).

    A solved TAN-without-SIP header often under-predicts corner separation vs true sky; the rectangular
    SQL prefilter then clips the catalog so QA shows 'missing Gaia' stripes at left/right edges.
    """
    from pipeline import catalog_cone_radius_deg_from_optics  # noqa: PLC0415

    if hdr is None:
        return 0.0
    foc = _fits_header_first_positive_float(
        hdr, ("FOCALLEN", "FOCALLENGTH", "FOCAL", "FOC_LEN")
    )
    pix = _fits_header_first_positive_float(
        hdr, ("PIXSIZE", "XPIXSZ", "PIXSZ", "PIXELSIZE", "PIX_SIZE")
    )
    if foc is None or pix is None:
        return 0.0
    try:
        foc_n, _ = normalize_telescope_focal_mm_for_plate_scale(float(foc))
        xb, _yb = fits_binning_xy_from_header(hdr)
        eff_um = effective_binned_pixel_pitch_um(base_pixel_um_1x1=float(pix), binning=int(xb))
        # margin ~2.0: VYVAR-solver-class corner coverage + CRVAL slop / high-Dec geometry
        return float(
            catalog_cone_radius_deg_from_optics(
                naxis1=int(naxis1),
                naxis2=int(naxis2),
                pixel_pitch_um=float(eff_um),
                focal_length_mm=float(foc_n),
                margin=2.05,
                fov_diameter_fallback_deg=float(plate_solve_fov_fallback_deg),
            )
        )
    except Exception as exc:  # noqa: BLE001
        from except_fix_counters import get_except_fix_counters

        get_except_fix_counters().gaia_cone_optics_floor_fail += 1
        LOGGER.error("[CATALOG] Gaia cone optics floor failed; using FOV fallback: %s", exc)
        fb = max(0.05, float(plate_solve_fov_fallback_deg))
        return float(max(MIN_GAIA_CONE_RADIUS_DEG, fb * 0.65))


def _field_center_and_radius_from_wcs(w: WCS, h: int, wpx: int) -> tuple[SkyCoord, float]:
    """Footprint for **one** Vizier/Gaia cone query tied to the detector - not 'cela obloha'.

    Uses the geometric pixel centre, then the **maximum** great-circle separation from that centre to a
    dense sample of the **full rectangle border** (not only corners). With sip / distortion, the farthest
    sky point from the centre is not always a corner - undersized cones produced a visible circular
    'matched only in the middle' QA overlay. Adds multiplicative + absolute margin for edge stars and
    small plate-solve errors.
    """
    center = SkyCoord.from_pixel((wpx - 1) / 2.0, (h - 1) / 2.0, wcs=w, origin=0)
    max_sep_deg = 0.0
    step = max(1, int(round(float(max(wpx, h)) / 120.0)))
    pixels: set[tuple[int, int]] = set()
    for cx, cy in ((0, 0), (wpx - 1, 0), (0, h - 1), (wpx - 1, h - 1)):
        pixels.add((int(cx), int(cy)))
    for x in range(0, wpx, step):
        pixels.add((x, 0))
        pixels.add((x, h - 1))
    for y in range(0, h, step):
        pixels.add((0, y))
        pixels.add((wpx - 1, y))
    for cx, cy in pixels:
        cc = SkyCoord.from_pixel(float(cx), float(cy), wcs=w, origin=0)
        max_sep_deg = max(max_sep_deg, float(center.separation(cc).deg))
    # Margin: WCS at edges, rectangle vs cone on sphere, plate-solve error. Larger margin because
    # ``field_catalog_cone.csv`` built from a cropped MASTERSTAR must not underserve full-chip frames.
    radius_deg = max(max_sep_deg * 1.38, max_sep_deg + 45.0 / 3600.0)
    # If CD scale is wrong (too 'zoomed' on sky), spherical sampling collapses - blend in tangent-plane
    # half-diagonal from pixel scales so the cone stays physically plausible.
    try:
        scales = w.proj_plane_pixel_scales()
        sx_deg = abs(float(scales[0].to(u.deg).value))
        sy_deg = abs(float(scales[1].to(u.deg).value))
        r_cd = float(math.hypot(0.5 * (wpx - 1) * sx_deg, 0.5 * (h - 1) * sy_deg))
        radius_deg = max(radius_deg, r_cd * 1.22)
    except Exception:  # noqa: BLE001
        pass
    radius_deg = max(float(radius_deg), float(MIN_GAIA_CONE_RADIUS_DEG))
    return center, radius_deg


def _effective_field_catalog_cone_radius_deg(
    w: WCS,
    h: int,
    wpx: int,
    plate_solve_fov_deg: float | None,
    fits_header: fits.Header | None = None,
) -> tuple[SkyCoord, float]:
    """WCS-derived cone radius, optics floor from FITS (FOCALLEN+PIXSIZE), optional UI FOV minimum."""
    center, r = _field_center_and_radius_from_wcs(w, h, wpx)
    try:
        _pf_fb = float(plate_solve_fov_deg) if plate_solve_fov_deg is not None else 1.5
        if not math.isfinite(_pf_fb) or _pf_fb <= 0:
            _pf_fb = 1.5
    except (TypeError, ValueError):
        _pf_fb = 1.5
    r_opt = _gaia_catalog_cone_radius_optics_floor_deg(
        fits_header,
        naxis1=int(wpx),
        naxis2=int(h),
        plate_solve_fov_fallback_deg=float(_pf_fb),
    )
    r = max(float(r), float(r_opt))
    r_physical = float(r)
    if plate_solve_fov_deg is not None:
        try:
            pf = float(plate_solve_fov_deg)
            if math.isfinite(pf) and pf > 0:
                r_fov = catalog_cone_radius_from_fov_diameter_deg(pf)
                # UI FOV je bezpecnostne minimum, ale nesmie 'rozsirit' kuzel ovela nad realny cip:
                # pri zle nastavenom velkom FOV (napr. 20 deg+) by inak vznikol polomer ~13 deg a SQL by
                # tahalo 500k+ hviezd (minuty behu). Ak uz WCS + optika davaju rozumny polomer,
                # obmedzime prispevok z FOV na ~30 % nad fyzikalnu stopu.
                if r_physical >= 2.5:
                    r_fov_eff = min(float(r_fov), r_physical * 1.30 + 0.35)
                else:
                    r_fov_eff = float(r_fov)
                r = min(22.0, max(r_physical, r_fov_eff))
        except (TypeError, ValueError):
            pass
    return center, float(r)


def _invalidate_field_catalog_cone_cache_if_needed(
    field_catalog_csv: Path,
    *,
    plate_solve_fov_deg: float | None,
    effective_radius_deg: float,
) -> None:
    """Remove ``field_catalog_cone.csv`` + meta when UI FOV or required cone size no longer matches cache."""
    p_csv = Path(field_catalog_csv)
    meta_p = _field_catalog_cone_meta_path(p_csv)
    if not p_csv.is_file() and not meta_p.is_file():
        return
    try:
        meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.is_file() else {}
    except Exception:  # noqa: BLE001
        meta = {}
    r_stored = float(meta.get("cone_radius_deg") or 0.0)
    fov_stored = meta.get("plate_solve_fov_deg")
    slack_deg = 45.0 / 3600.0
    reasons: list[str] = []

    if plate_solve_fov_deg is not None:
        try:
            pf = float(plate_solve_fov_deg)
            if math.isfinite(pf):
                if fov_stored is None:
                    reasons.append("meta chyba plate_solve_fov_deg (stary cache)")
                else:
                    try:
                        if abs(float(fov_stored) - pf) > 1e-4:
                            reasons.append(f"plate_solve_fov_deg {float(fov_stored):.6f} -> {pf:.6f}")
                    except (TypeError, ValueError):
                        reasons.append("neplatny ulozeny plate_solve_fov_deg")
        except (TypeError, ValueError):
            pass

    r_eff = float(effective_radius_deg)
    if r_stored > 0 and r_eff > r_stored * 1.02 + slack_deg:
        reasons.append(f"kuzel r~{r_eff:.4f} deg > ulozene {r_stored:.4f} deg")

    if not reasons:
        return
    p_csv.unlink(missing_ok=True)
    meta_p.unlink(missing_ok=True)
    log_event("Katalog: field_catalog_cone cache vymazana - " + "; ".join(reasons) + " (nacitam nanovo).")


def _field_catalog_cone_meta_path(field_catalog_csv: Path) -> Path:
    return field_catalog_csv.parent / "field_catalog_cone_meta.json"


def _write_field_catalog_cone_meta(
    field_catalog_csv: Path,
    *,
    center: SkyCoord,
    radius_deg: float,
    naxis1: int,
    naxis2: int,
    plate_solve_fov_deg: float | None = None,
) -> None:
    """Persist cone parameters used to build ``field_catalog_cone.csv`` (cache invalidation for larger chips)."""
    p = _field_catalog_cone_meta_path(field_catalog_csv)
    rec = {
        "cone_radius_deg": float(radius_deg),
        "center_ra_icrs_deg": float(center.ra.deg),
        "center_dec_icrs_deg": float(center.dec.deg),
        "naxis1": int(naxis1),
        "naxis2": int(naxis2),
        "plate_solve_fov_deg": float(plate_solve_fov_deg)
        if plate_solve_fov_deg is not None and math.isfinite(float(plate_solve_fov_deg))
        else None,
    }
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, indent=1), encoding="utf-8")


_MASTERSTAR_ZONE_LOG_ONCE: set[str] = set()


def _masterstar_zone_linear_threshold(
    dao_detection_n_equiv: Any,
) -> float | None:
    """Linear-zone lower bound from DAO detection significance (peak_dao/bg_sigma >= N_equiv)."""
    try:
        t1 = float(dao_detection_n_equiv)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(t1) or t1 <= 0:
        return None
    return t1


def _detect_empirical_clip_level_adu(data: "np.ndarray") -> float | None:
    """Return the ADU level of frame truncation pile-up, or None if the frame is not clipped."""
    import numpy as np

    arr = np.asarray(data, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    vmax = float(np.max(finite))
    if not math.isfinite(vmax) or vmax <= 0:
        return None
    at_max = np.abs(finite - vmax) <= 0.5
    if int(np.count_nonzero(at_max)) >= 8:
        return vmax
    for clip_candidate in (65535.0, 32767.0):
        near = np.abs(finite - clip_candidate) <= 0.5
        if int(np.count_nonzero(near)) >= 8:
            return clip_candidate
    return None


def _resolve_peak_saturation_limit_adu(
    *,
    camera_sat_limit_adu: float | None,
    saturate_fraction: float,
    n_stack: int = 1,
    sky_median_adu: float | None = None,
    frame_max_adu: float | None = None,
    empirical_clip_adu: float | None = None,
) -> float | None:
    """Peak-test saturation ceiling: empirical clip first; camera constant only on raw-scale data."""
    ns = max(1, int(n_stack))
    frac = float(saturate_fraction)
    if empirical_clip_adu is not None:
        try:
            ec = float(empirical_clip_adu)
            if math.isfinite(ec) and ec > 0:
                return ec * frac / float(ns)
        except (TypeError, ValueError):
            pass
    if camera_sat_limit_adu is None:
        return None
    try:
        raw_cam = float(camera_sat_limit_adu)
        cam_lim = raw_cam * frac / float(ns)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(cam_lim) or cam_lim <= 0:
        return None
    sky = float("nan")
    fmax = float("nan")
    try:
        if sky_median_adu is not None:
            sky = float(sky_median_adu)
        if frame_max_adu is not None:
            fmax = float(frame_max_adu)
    except (TypeError, ValueError):
        pass
    # Float MASTERSTAR stacks can overshoot the 16-bit clip by a few percent
    # (interpolation). Veto only when the image is clearly a different unit scale
    # (SAT-LIMIT-01: 515 stack max 68429 vs clip 65535 is not a unit change).
    if math.isfinite(fmax) and math.isfinite(raw_cam) and fmax > raw_cam * 1.20:
        return None
    if math.isfinite(sky) and sky > 0.20 * cam_lim:
        return None
    return cam_lim


def _masterstar_zone_log_once(key: str, msg: str) -> None:
    if key in _MASTERSTAR_ZONE_LOG_ONCE:
        return
    _MASTERSTAR_ZONE_LOG_ONCE.add(key)
    logging.warning(msg)


def _resolve_masterstar_bg_sigma_adu(
    *,
    sigma_px: Any,
    noise_floor_adu: Any,
    sky_median_adu: Any,
    prematch_peak_sigma_floor: Any,
) -> float | None:
    """Return per-pixel background sigma used in the DAO noise-floor formula."""
    try:
        if sigma_px is not None and str(sigma_px).strip() != "":
            sp = float(sigma_px)
            if math.isfinite(sp) and sp > 0:
                return sp
    except (TypeError, ValueError):
        pass
    try:
        nf = float(noise_floor_adu)
        sky = float(sky_median_adu)
        k = float(prematch_peak_sigma_floor)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(nf) and math.isfinite(sky) and math.isfinite(k) and k > 0):
        return None
    inv = (nf - sky) / k
    if math.isfinite(inv) and inv > 0:
        _masterstar_zone_log_once(
            "sigma_inverted",
            "[MASTERSTAR zone] bg_sigma_adu inferred from noise_floor_adu, sky_median_adu and "
            "prematch_peak_sigma_floor (prefer det_meta['bg_sigma_adu'] when available).",
        )
        return float(inv)
    return None


def _annotate_masterstars_flux_zones(
    df: pd.DataFrame,
    *,
    noise_floor_adu: Any,
    equipment_saturate_adu: float | None,
    saturate_limit_adu_fallback: Any = None,
    n_stack: int | None = None,
    saturate_limit_fraction: float = SAT_LIMIT_NO_KNEE_FRAC,
    sigma_px: Any = None,
    sky_median_adu: Any = None,
    prematch_peak_sigma_floor: Any = None,
    frame_max_adu: Any = None,
    empirical_clip_adu: Any = None,
    dao_detection_n_equiv: Any = None,
) -> pd.DataFrame:
    """Tag MASTERSTAR catalog rows by peak significance (peak_dao/bg_sigma) and saturation.

    Linear/usability boundary T1 = ``dao_detection_n_equiv`` (same N-sigma as DAOFIND threshold).
    Noisy sub-bands step below in whole sigma (``_MASTERSTAR_ZONE_SIGMA_STEP``).

    ``noise_floor_adu`` must match the DAO pre-match SNR filter (``median + kxsigma``) from
    :func:`detect_stars_and_match_catalog` (see ``det_meta["noise_floor_adu"]``).
    """
    import numpy as np

    out = df.copy()
    if "flux" not in out.columns:
        return out
    flux_s = pd.to_numeric(out["flux"], errors="coerce")

    nf: float | None = None
    try:
        if noise_floor_adu is not None and str(noise_floor_adu).strip() != "":
            nf = float(noise_floor_adu)
            if not math.isfinite(nf):
                nf = None
    except (TypeError, ValueError):
        nf = None

    ns = int(n_stack) if n_stack is not None else 1
    ns = max(1, ns)

    # RAW camera/container clip (do not pre-scale; _resolve_peak_saturation_limit_adu applies frac).
    sat_lim_raw = _finite_positive_adu(equipment_saturate_adu)
    peak_frac = float(saturate_limit_fraction)
    unresolved_clip = sat_lim_raw is None
    if unresolved_clip:
        sat_lim_raw = _finite_positive_adu(saturate_limit_adu_fallback)
        unresolved_clip = sat_lim_raw is None
    if unresolved_clip:
        sat_lim_raw = float(SAT_LIMIT_CONTAINER_CLIP_ADU)
        peak_frac = float(SAT_LIMIT_NO_KNEE_FRAC)
        _masterstar_zone_log_once(
            "sat_limit_unresolved",
            "[INV-SAT-LIMIT] header/equipment/sat_diag clip unresolved; "
            f"using clip={SAT_LIMIT_CONTAINER_CLIP_ADU:.0f} ADU (GAIN-DOMAIN-01 container), "
            f"peak-test frac={SAT_LIMIT_NO_KNEE_FRAC:.2f} "
            f"(peak-test={SAT_LIMIT_CONTAINER_CLIP_ADU * SAT_LIMIT_NO_KNEE_FRAC:.1f} ADU). "
            "Never silently admit.",
        )

    out["noise_floor_adu"] = nf if nf is not None else np.nan
    out["saturate_limit_adu"] = float(sat_lim_raw) if sat_lim_raw is not None else np.nan

    if "peak_max_adu" in out.columns:
        peak_col_used = "peak_max_adu"
        peak_s = pd.to_numeric(out["peak_max_adu"], errors="coerce")
    elif "peak_dao" in out.columns:
        peak_col_used = "peak_dao"
        peak_s = pd.to_numeric(out["peak_dao"], errors="coerce")
    else:
        peak_col_used = "flux"
        peak_s = flux_s

    peak_sat_lim = _resolve_peak_saturation_limit_adu(
        camera_sat_limit_adu=sat_lim_raw,
        saturate_fraction=peak_frac,
        n_stack=ns,
        sky_median_adu=sky_median_adu,
        frame_max_adu=frame_max_adu,
        empirical_clip_adu=empirical_clip_adu,
    )
    if peak_sat_lim is None and unresolved_clip:
        # Scale veto rejected even the conservative default. INV-SAT-LIMIT: still apply peak test.
        peak_sat_lim = float(sat_lim_raw) * float(peak_frac) / float(ns)
        _masterstar_zone_log_once(
            "sat_limit_unresolved_scale_veto",
            "[INV-SAT-LIMIT] peak-test scale veto ignored for unresolved clip; "
            f"applying peak-test={float(peak_sat_lim):.1f} ADU.",
        )
    if peak_sat_lim is None:
        peak_sat_lim = float(sat_lim_raw) if sat_lim_raw is not None else None
        _masterstar_zone_log_once(
            "sat_limit_85pct_nan",
            "[ZONE-SAT-01] saturate_limit_adu_85pct unresolved; peak test uses saturate_limit_adu "
            f"({peak_sat_lim}).",
        )
    out["saturate_limit_adu_85pct"] = peak_sat_lim if peak_sat_lim is not None else np.nan
    out["zone_peak_column"] = peak_col_used
    out["zone_sat_limit_used"] = float(peak_sat_lim) if peak_sat_lim is not None else np.nan

    out["zone"] = ""
    if peak_sat_lim is not None:
        out.loc[peak_s > float(peak_sat_lim), "zone"] = "saturated"

    bg_sigma = _resolve_masterstar_bg_sigma_adu(
        sigma_px=sigma_px,
        noise_floor_adu=noise_floor_adu,
        sky_median_adu=sky_median_adu,
        prematch_peak_sigma_floor=prematch_peak_sigma_floor,
    )
    if bg_sigma is None:
        _masterstar_zone_log_once(
            "sigma_unresolvable",
            "[MASTERSTAR zone] bg_sigma_adu unresolvable - "
            "leaving zone empty (maps to neznama_zona downstream).",
        )
    else:
        t1 = _masterstar_zone_linear_threshold(dao_detection_n_equiv)
        if t1 is None:
            _masterstar_zone_log_once(
                "n_equiv_missing",
                "[MASTERSTAR zone] dao_detection_n_equiv unresolvable - "
                "leaving zone empty (maps to neznama_zona downstream).",
            )
        else:
            if "peak_dao" in out.columns:
                peak_dao_s = pd.to_numeric(out["peak_dao"], errors="coerce")
            else:
                peak_dao_s = pd.Series(np.nan, index=out.index, dtype=float)
            peak_sig = peak_dao_s / float(bg_sigma)
            unsat = out["zone"].eq("")
            sig_ok = unsat & peak_sig.notna()
            sig_miss = unsat & peak_sig.isna()
            out.loc[sig_miss, "zone"] = "unknown"
            out.loc[sig_ok & (peak_sig >= t1), "zone"] = "linear"
            out.loc[sig_ok & (peak_sig < t1), "zone"] = "noise"
            if sig_miss.any():
                _masterstar_zone_log_once(
                    "peak_dao_missing",
                    "[MASTERSTAR zone] peak_dao missing/NaN rows "
                    "marked zone=unknown (not flux fallback).",
                )

    if peak_sat_lim is not None:
        out["is_saturated"] = (peak_s > float(peak_sat_lim)).fillna(False)
    else:
        # Camera clip was supplied but rejected as wrong-scale (precalibrated stack).
        out["is_saturated"] = False

    out["is_noisy"] = out["zone"].eq("noise")
    out["is_usable"] = out["zone"].eq("linear") & flux_s.notna()
    return out


def _dao_auto_binning_factor(h: int, w: int) -> int:
    """2x2 mean binning for DAO on large chips (~4x fewer pixels); skipped below ~5 MP."""
    mp = float(int(h) * int(w)) / 1_000_000.0
    if mp < 5.0:
        return 1
    return 2


def _pixel_noise_sigma_pp_adu(arr: Any) -> float:
    """Gradient-immune pixel noise from adjacent differences (MAD estimator).

    For smooth large-scale structure, ``I[i+1]-I[i]`` is nearly constant so MAD tracks
    pixel-to-pixel noise, not sky gradient. Validated on BO CVn MASTERSTAR (~46 ADU stable
    across draft 435/450 while ``plain_mean_med_std`` moved 26%; see SIGMA-ESTIMATOR-VERIFY).
    """
    import numpy as np

    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 2 or a.size < 64:
        return float("nan")
    finite = np.isfinite(a)
    chunks: list[np.ndarray] = []
    dh = a[:, 1:] - a[:, :-1]
    mh = finite[:, 1:] & finite[:, :-1]
    if int(np.count_nonzero(mh)) >= 32:
        chunks.append(dh[mh].ravel())
    dv = a[1:, :] - a[:-1, :]
    mv = finite[1:, :] & finite[:-1, :]
    if int(np.count_nonzero(mv)) >= 32:
        chunks.append(dv[mv].ravel())
    if not chunks:
        return float("nan")
    diffs = np.concatenate(chunks)
    if diffs.size < 64:
        return float("nan")
    med = float(np.median(diffs))
    mad = float(np.median(np.abs(diffs - med)))
    if not math.isfinite(mad) or mad <= 0:
        return float("nan")
    return float(mad / 0.674489750196082 / math.sqrt(2.0))


def _dao_noise_sigma_adu(
    arr: Any,
    *,
    bfac: int,
    fallback_std: float,
    data_dao: Any | None = None,
) -> float:
    """Legacy per-pixel noise scale (diagnostic only; not used for DAO threshold under option B)."""

    sigma_pp = _pixel_noise_sigma_pp_adu(arr)
    if math.isfinite(float(sigma_pp)) and float(sigma_pp) > 0:
        return float(sigma_pp) / math.sqrt(float(max(1, bfac)))
    if int(bfac) > 1 and data_dao is not None:
        _, _, std_dao = plain_mean_med_std(data_dao, sigma=3.0, maxiters=3)
        logging.warning("[DAO] sigma_pp unavailable; falling back to plain_mean_med_std on binned frame")
        return float(std_dao)
    logging.warning("[DAO] sigma_pp unavailable; falling back to global plain_mean_med_std")
    return float(fallback_std)


def _dao_convolved_background_rms_adu(
    data_dao: Any,
    *,
    fwhm_px: float,
    sigma_radius: float = 1.5,
) -> tuple[float, float]:
    """Robust RMS of the DAO convolved detection image (option B threshold basis).

    Builds the same zero-sum kernel DAOStarFinder uses, convolves ``data_dao``, and
    returns (rms_convolved, kernel.rel_err). Caller sets ``scale_threshold=False`` and
    ``threshold = N * rms_convolved`` so nominal N-sigma holds on correlated/resampled data.
    """
    import numpy as np
    from photutils.detection.core import _StarFinderKernel
    from scipy.ndimage import convolve

    arr = np.asarray(data_dao, dtype=np.float32)
    if arr.ndim != 2 or arr.size < 64:
        return float("nan"), float("nan")
    fwhm = max(1.2, float(fwhm_px))
    kernel = _StarFinderKernel(fwhm=fwhm, sigma_radius=float(sigma_radius))
    conv = convolve(arr, kernel.data, mode="nearest")
    _, _, rms = plain_mean_med_std(conv, sigma=3.0, maxiters=3)
    return float(rms), float(kernel.rel_err)


_BATCH_E_N_EQUIV_LOGGED = False


def _dao_detection_threshold_adu(
    rms_conv: float,
    *,
    cfg: Any | None,
    dao_threshold_sigma: float,
) -> tuple[float, float]:
    """T4-1 Option B: threshold = N_equiv * rms_conv (measured N from Part 2b)."""
    global _BATCH_E_N_EQUIV_LOGGED  # noqa: PLW0603
    rms = float(rms_conv)
    if not math.isfinite(rms) or rms <= 0:
        return 1e-6, float("nan")
    n_eff = float(getattr(cfg, "dao_detection_n_equiv", dao_threshold_sigma)) if cfg is not None else float(
        dao_threshold_sigma
    )
    if not math.isfinite(n_eff) or n_eff <= 0:
        n_eff = float(dao_threshold_sigma)
    if cfg is not None and not _BATCH_E_N_EQUIV_LOGGED:
        LOGGER.info("[BATCH-E E.4] N_equiv=%.2f applied for DAO detection threshold", n_eff)
        _BATCH_E_N_EQUIV_LOGGED = True
    return max(n_eff * rms, 1e-6), n_eff


def _apply_dao_centroid_wcs_guard(
    x: "np.ndarray",
    y: "np.ndarray",
    *,
    matched: "np.ndarray",
    safe: "np.ndarray",
    master_df: pd.DataFrame,
    fwhm_px: float,
    max_shift_fwhm: float,
) -> tuple["np.ndarray", "np.ndarray", int]:
    """Replace DAO centroids with MASTERSTAR reference pixels when shift exceeds guard."""
    import numpy as np

    xo = np.asarray(x, dtype=np.float64).copy()
    yo = np.asarray(y, dtype=np.float64).copy()
    if master_df is None or master_df.empty or "x" not in master_df.columns or "y" not in master_df.columns:
        return xo, yo, 0
    mx = pd.to_numeric(master_df["x"], errors="coerce").to_numpy(dtype=np.float64)
    my = pd.to_numeric(master_df["y"], errors="coerce").to_numpy(dtype=np.float64)
    m = np.asarray(matched, dtype=bool)
    if not m.any():
        return xo, yo, 0
    s = np.clip(np.asarray(safe, dtype=np.int64), 0, max(len(mx) - 1, 0))
    x_ref = mx[s]
    y_ref = my[s]
    ok = m & np.isfinite(x_ref) & np.isfinite(y_ref) & np.isfinite(xo) & np.isfinite(yo)
    if not ok.any():
        return xo, yo, 0
    max_px = float(max(0.1, max_shift_fwhm)) * float(max(1.2, fwhm_px))
    shift = np.hypot(xo - x_ref, yo - y_ref)
    use_wcs = ok & (shift > max_px)
    xo[use_wcs] = x_ref[use_wcs]
    yo[use_wcs] = y_ref[use_wcs]
    return xo, yo, int(np.count_nonzero(use_wcs))


def _lock_matched_centroids_to_master_grid(
    arr: "np.ndarray",
    x: "np.ndarray",
    y: "np.ndarray",
    *,
    matched: "np.ndarray",
    safe: "np.ndarray",
    master_df: pd.DataFrame,
    fwhm_px: float,
    search_fwhm: float = 2.5,
) -> tuple["np.ndarray", "np.ndarray", int]:
    """Lock matched catalog stars to MASTERSTAR grid with local peak refinement.

    Master-reference per-frame catalogs must measure on the shared alignment grid,
    not on arbitrary DAO detections that can land on faint neighbours after transform
    smearing.  Each matched row is snapped to the master (x, y) then refined within
    a small search window for the brightest pixel (sub-pixel centre not required).
    """
    import numpy as np

    xo = np.asarray(x, dtype=np.float64).copy()
    yo = np.asarray(y, dtype=np.float64).copy()
    if master_df is None or master_df.empty or "x" not in master_df.columns or "y" not in master_df.columns:
        return xo, yo, 0
    img = np.asarray(arr, dtype=np.float64)
    if img.ndim != 2 or img.size == 0:
        return xo, yo, 0
    h_img, w_img = int(img.shape[0]), int(img.shape[1])
    mx = pd.to_numeric(master_df["x"], errors="coerce").to_numpy(dtype=np.float64)
    my = pd.to_numeric(master_df["y"], errors="coerce").to_numpy(dtype=np.float64)
    m = np.asarray(matched, dtype=bool)
    if not m.any():
        return xo, yo, 0
    s = np.clip(np.asarray(safe, dtype=np.int64), 0, max(len(mx) - 1, 0))
    radius = int(max(3, math.ceil(float(max(1.2, fwhm_px)) * float(max(1.0, search_fwhm)))))
    n_locked = 0
    for i in np.nonzero(m)[0]:
        si = int(s[i])
        if si < 0 or si >= len(mx):
            continue
        x_ref = float(mx[si])
        y_ref = float(my[si])
        if not (math.isfinite(x_ref) and math.isfinite(y_ref)):
            continue
        xi = int(round(x_ref))
        yi = int(round(y_ref))
        x_lo = max(0, xi - radius)
        x_hi = min(w_img, xi + radius + 1)
        y_lo = max(0, yi - radius)
        y_hi = min(h_img, yi + radius + 1)
        if x_lo >= x_hi or y_lo >= y_hi:
            xo[i] = x_ref
            yo[i] = y_ref
            n_locked += 1
            continue
        patch = img[y_lo:y_hi, x_lo:x_hi]
        if patch.size == 0 or not np.any(np.isfinite(patch)):
            xo[i] = x_ref
            yo[i] = y_ref
            n_locked += 1
            continue
        flat_idx = int(np.nanargmax(patch))
        py, px = np.unravel_index(flat_idx, patch.shape)
        xo[i] = float(x_lo + int(px))
        yo[i] = float(y_lo + int(py))
        n_locked += 1
    return xo, yo, n_locked


def _proc_deduplicate_matched_catalog_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per non-empty ``catalog_id`` (highest peak / flux wins).

    Multiple DAO detections can match the same Gaia source in one frame; duplicates
    inflate Phase-1 comp RMS and can select faint spurious matches.
    """
    if df is None or len(df) == 0 or "catalog_id" not in df.columns:
        return df
    out = df.copy()
    cid = out["catalog_id"].fillna("").astype(str).str.strip()
    matched = cid.ne("") & ~cid.str.lower().isin({"nan", "none"})
    if not bool(matched.any()):
        return out
    # Always a Series: out.get("flux") is None when the column is absent (empty-DAO
    # frames after forced-phot inject - draft 515 crash), and pd.to_numeric(None)
    # collapses to numpy.float64 without .fillna.
    if "peak_max_adu" in out.columns:
        score = pd.to_numeric(out["peak_max_adu"], errors="coerce")
    elif "dao_flux" in out.columns:
        score = pd.to_numeric(out["dao_flux"], errors="coerce")
    elif "flux" in out.columns:
        score = pd.to_numeric(out["flux"], errors="coerce")
    else:
        score = pd.Series(float("nan"), index=out.index, dtype="float64")
    score = pd.Series(pd.to_numeric(score, errors="coerce"), index=out.index, dtype="float64")
    out["_dedupe_score"] = score.fillna(-1.0)
    keep_idx = (
        out.loc[matched]
        .sort_values("_dedupe_score", ascending=False, kind="mergesort")
        .drop_duplicates(subset=["catalog_id"], keep="first")
        .index
    )
    keep_mask = out.index.isin(keep_idx) | ~matched
    out = out.loc[keep_mask].drop(columns=["_dedupe_score"]).reset_index(drop=True)
    n_dropped = int(matched.sum()) - int(len(keep_idx))
    if n_dropped > 0:
        LOGGER.debug("[PROC] deduplicated %d duplicate catalog_id rows in per-frame CSV", n_dropped)
    return out


def _mean_bin2d_for_dao(data0: "np.ndarray", factor: int) -> tuple["np.ndarray", int]:
    import numpy as np

    f = int(factor)
    a = np.asarray(data0, dtype=np.float32)
    if f <= 1:
        return a, 1
    h, w = a.shape
    h2, w2 = (h // f) * f, (w // f) * f
    if h2 < f or w2 < f:
        return a, 1
    b = a[:h2, :w2].reshape(h2 // f, f, w2 // f, f).mean(axis=(1, 3)).astype(np.float32)
    return b, f


def _gaia_chip_xy_from_catalog(
    cat_df: pd.DataFrame | None,
    wcs_obj: Any,
    *,
    wpx: int,
    h: int,
) -> pd.DataFrame:
    """On-chip Gaia rows with x_gaia/y_gaia for star-mask and born-owned pass2."""
    import numpy as np

    if cat_df is None or cat_df.empty or "ra_deg" not in cat_df.columns or "dec_deg" not in cat_df.columns:
        return pd.DataFrame()
    ra = pd.to_numeric(cat_df["ra_deg"], errors="coerce").to_numpy(dtype=np.float64)
    de = pd.to_numeric(cat_df["dec_deg"], errors="coerce").to_numpy(dtype=np.float64)
    ok = np.isfinite(ra) & np.isfinite(de)
    if not bool(ok.any()):
        return pd.DataFrame()
    gx, gy = wcs_obj.world_to_pixel_values(ra[ok], de[ok])
    sub = cat_df.loc[ok].copy().reset_index(drop=True)
    sub["x_gaia"] = gx
    sub["y_gaia"] = gy
    sub["g_mag"] = pd.to_numeric(sub.get("mag"), errors="coerce")
    inb = (
        (sub["x_gaia"] >= 0)
        & (sub["x_gaia"] < float(wpx))
        & (sub["y_gaia"] >= 0)
        & (sub["y_gaia"] < float(h))
    )
    return sub.loc[inb].reset_index(drop=True)


def _proc_rename_det_names_to_catalog_id(df: pd.DataFrame) -> pd.DataFrame:
    """Matched rows still named DET_* get ``name`` = ``catalog_id`` (stale master names)."""
    if df is None or df.empty or "catalog_id" not in df.columns or "name" not in df.columns:
        return df
    _matched_mask = (
        df["catalog_id"].notna()
        & (df["catalog_id"].astype(str).str.strip() != "")
        & (df["name"].astype(str).str.startswith("DET_"))
    )
    if not _matched_mask.any():
        return df
    out = df.copy()
    out.loc[_matched_mask, "name"] = out.loc[_matched_mask, "catalog_id"].astype(str)
    return out


def _proc_drop_unmatched_dao_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop unmatched DAO_ONLY detections (no catalog_id) BEFORE expensive per-star work.

    GAIA_MATCHED rows carry a catalog_id and are kept; DAO_ONLY rows
    (empty catalog_id) are discarded by the final keep-matched filter anyway. Keying on
    catalog_id (not source_type) is correct in both export paths, including the in-process path
    where source_type=GAIA_MATCHED is assigned only later.
    """
    if df is None or len(df) == 0 or "catalog_id" not in df.columns:
        return df
    cid = df["catalog_id"].fillna("").astype(str).str.strip()
    keep = cid.ne("") & ~cid.str.lower().isin({"nan", "none"})
    return df.loc[keep].reset_index(drop=True)


def _proc_catalog_keep_matched_rows_only(df: pd.DataFrame) -> pd.DataFrame:
    """TODO-13: keep catalog rows with valid ``catalog_id`` and ``source_type``."""
    if df is None or len(df) == 0 or "catalog_id" not in df.columns:
        return df
    cid = df["catalog_id"].fillna("").astype(str).str.strip()
    _valid_cid = cid.ne("") & ~cid.str.lower().isin({"nan", "none"})
    if "source_type" in df.columns:
        _valid_source = (
            df["source_type"].fillna("").astype(str).str.strip().str.upper().eq("GAIA_MATCHED")
        )
        keep = _valid_cid & _valid_source
    else:
        keep = _valid_cid
    return df.loc[keep].reset_index(drop=True)


def _prefilter_dao_table_brightest(tbl: Any, keep_top: int) -> Any:
    """Cap DAO rows before sorting when the finder returns an enormous table (slow + useless)."""
    import numpy as np

    if tbl is None or len(tbl) <= int(keep_top):
        return tbl
    flux_np = np.asarray(tbl["flux"], dtype=np.float64)
    k = int(keep_top)
    take = np.argpartition(flux_np, -k)[-k:]
    return tbl[take]


def _dao_spatial_flux_cap_row_indices(
    tbl: Any,
    *,
    max_n: int,
    width_px: float,
    height_px: float,
) -> "np.ndarray":
    """Row indices into ``tbl`` for up to ``max_n`` sources, spread across the detector.

    Sorting globally by flux and truncating biases toward the frame centre when vignetting or
    gradients make edge stars fainter in ADU - QA then falsely looks like a ``catalog disc``. This
    fills a coarse grid brightest-first per cell, then tops up by global flux (same cap).
    """
    import numpy as np

    n = len(tbl)
    m = int(max_n)
    if n <= m:
        return np.arange(n, dtype=np.int64)
    x = np.asarray(tbl["x_centroid"], dtype=np.float64)
    y = np.asarray(tbl["y_centroid"], dtype=np.float64)
    flux = np.asarray(tbl["flux"], dtype=np.float64)
    w = max(float(width_px), 1.0)
    h = max(float(height_px), 1.0)
    # ~25 sources per cell on average, bounded grid
    ncell_target = max(48, min(512, max(1, m // 25)))
    aspect = w / h
    ny = max(4, int(round((ncell_target / aspect) ** 0.5)))
    nx = max(4, int(round(ncell_target / float(ny))))
    ncells = nx * ny
    per_cell = max(1, m // ncells)
    ix = np.clip((x / w * nx).astype(np.int64), 0, nx - 1)
    iy = np.clip((y / h * ny).astype(np.int64), 0, ny - 1)
    cell_id = ix + iy * nx
    order = np.argsort(-flux)
    taken = np.zeros(n, dtype=bool)
    picked: list[int] = []
    counts = np.zeros(ncells, dtype=np.int32)
    for idx in order:
        c = int(cell_id[idx])
        if counts[c] < per_cell and len(picked) < m:
            picked.append(int(idx))
            counts[c] += 1
            taken[idx] = True
    for idx in order:
        if len(picked) >= m:
            break
        if not taken[idx]:
            picked.append(int(idx))
            taken[idx] = True
    return np.asarray(sorted(picked), dtype=np.int64)


def detect_stars_match_master_reference(
    data: "np.ndarray",
    hdr: fits.Header,
    master_df: pd.DataFrame,
    *,
    max_catalog_rows: int = 12000,
    match_sep_arcsec: float = 8.0,
    saturate_level_fraction: float = 0.999,
    faintest_mag_limit: float | None = None,
    dao_threshold_sigma: float = 3.5,
    dao_fwhm_px: float | None = None,
    fallback_saturate_adu: float | None = None,
    equipment_saturate_adu: float | None = None,
    frame_name: str = "",
    sat_diag_ctx: Any | None = None,
    raw_data: "np.ndarray | None" = None,
    raw_hdr: fits.Header | None = None,
    ref_ra_deg: float | None = None,
    ref_dec_deg: float | None = None,
    drift_ref_catalog_id: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """DAO on this frame + nearest-neighbor match to ``masterstars.csv`` (no Vizier / cone).

    Catalog IDs and static columns come from the master row; ``x``, ``y``, ``flux``, ``peak_*`` and
    saturation flags are per-frame. Intended for ``detrended_aligned`` data whose WCS matches
    ``MASTERSTAR.fits`` astrometry.
    """
    from pipeline import _dao_targeted_pass2_unmatched_gaia  # noqa: PLC0415

    import numpy as np

    m = master_df
    # Primary mode: sky match using per-frame WCS (DAO x/y -> RA/Dec), then NN match against MASTERSTAR ra/dec.
    # Fallback: if frame has no usable celestial WCS, do a pixel NN match (15 px) against MASTERSTAR x/y.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        wcs_obj = WCS(hdr)

    # MASTERSTAR sky catalog
    if "ra_deg" not in m.columns or "dec_deg" not in m.columns:
        raise ValueError("masterstars table must contain ra_deg, dec_deg")
    _ra = pd.to_numeric(m["ra_deg"], errors="coerce")
    _de = pd.to_numeric(m["dec_deg"], errors="coerce")
    okm_sky = _ra.notna() & _de.notna()
    if not bool(okm_sky.any()):
        raise ValueError("No valid ra_deg/dec_deg rows in masterstars")
    m_valid = m.loc[okm_sky].reset_index(drop=True)
    ra_m = pd.to_numeric(m_valid["ra_deg"], errors="coerce").to_numpy(dtype=np.float64)
    de_m = pd.to_numeric(m_valid["dec_deg"], errors="coerce").to_numpy(dtype=np.float64)
    master_coords = SkyCoord(ra=ra_m * u.deg, dec=de_m * u.deg, frame="icrs")

    match_mode = "sky"
    plate_scale_arcsec_per_px = None

    arr = np.asarray(data, dtype=np.float32)
    mean, med, std = plain_mean_med_std(arr, sigma=3.0, maxiters=3)
    data0 = np.nan_to_num((arr - med).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    bfac = 1
    _cfg_dao = AppConfig()
    _fb_m = float(dao_fwhm_px) if dao_fwhm_px is not None else float(_cfg_dao.sips_dao_fwhm_px)
    _base_fw_m = dao_detection_fwhm_pixels(hdr, configured_fallback=_fb_m)
    try:
        from photutils.detection import DAOStarFinder  # type: ignore

        thr_s = float(dao_threshold_sigma)
        thr_s = max(0.5, min(20.0, thr_s))
        dao_scale = _dao_auto_binning_factor(*data0.shape)
        data_dao, bfac = _mean_bin2d_for_dao(data0, dao_scale)
        fwhm_eff = max(1.2, _base_fw_m / float(bfac))
        rms_conv, _dao_rel_err = _dao_convolved_background_rms_adu(data_dao, fwhm_px=fwhm_eff)
        sigma_pp_diag = _dao_noise_sigma_adu(arr, bfac=bfac, fallback_std=float(std), data_dao=data_dao)
        std_dao = rms_conv
        if std_dao is None or (not np.isfinite(float(std_dao))) or float(std_dao) <= 0:
            try:
                std_dao = float(np.nanstd(arr))
            except Exception:  # noqa: BLE001
                std_dao = float("nan")
        if std_dao is None or (not np.isfinite(float(std_dao))) or float(std_dao) <= 0:
            _nm = str(frame_name or hdr.get("FILENAME") or "").strip() or "frame"
            try:
                finite = np.isfinite(arr)
                n_finite = int(np.count_nonzero(finite))
                if n_finite > 0:
                    vals = arr[finite]
                    n_unique = int(len(np.unique(vals)))
                    mn = float(np.nanmin(vals))
                    mx = float(np.nanmax(vals))
                else:
                    n_unique, mn, mx = 0, float("nan"), float("nan")
                print(
                    f"DEBUG std=0: {_nm} n_unique={n_unique} n_finite={n_finite} "
                    f"min={mn:.1f} max={mx:.1f}"
                )
            except Exception:  # noqa: BLE001
                pass
            # If the frame isn't constant (min != max), try std from non-zero finite pixels.
            try:
                nonzero_mask = (arr != 0) & np.isfinite(arr)
                if int(np.count_nonzero(nonzero_mask)) > 100:
                    std_dao = float(np.std(arr[nonzero_mask]))
            except Exception:  # noqa: BLE001
                pass
        if std_dao is None or (not np.isfinite(float(std_dao))) or float(std_dao) <= 0:
            _nm = str(frame_name or hdr.get("FILENAME") or "").strip() or "frame"
            print(f"WARNING: {_nm} std=0 aj po fallback, preskakujem")
            empty_meta: dict[str, Any] = {"status": "std_zero"}
            return pd.DataFrame(), empty_meta
        _thr, _n_equiv_used = _dao_detection_threshold_adu(
            float(std_dao), cfg=_cfg_dao, dao_threshold_sigma=float(thr_s),
        )
        try:
            _nm = str(frame_name or hdr.get("FILENAME") or "").strip() or "frame"
            print(
                f"DEBUG DAO INPUT: {_nm} mean={float(np.nanmean(arr)):.1f} std={float(np.nanstd(arr)):.1f} "
                f"threshold={float(_thr):.1f} rms_conv={float(rms_conv):.2f} n_equiv={float(_n_equiv_used):.2f} "
                f"sigma_pp_diag={float(sigma_pp_diag):.2f} "
                f"fwhm={float(fwhm_eff):.2f}"
            )
        except Exception:  # noqa: BLE001
            pass
        finder = DAOStarFinder(
            fwhm=float(fwhm_eff),
            threshold=float(_thr),
            scale_threshold=False,
            n_brightest=None,
            **DAO_STAR_FINDER_NO_ROUNDNESS_FILTER,
        )
        tbl = finder(data_dao)
        if tbl is not None and len(tbl) > 0:
            tbl = _prefilter_dao_table_brightest(tbl, max(int(max_catalog_rows) * 12, 36_000))
            from astropy.table import Table as _AstropyTable

            tbl = _AstropyTable(tbl, copy=True)
            tbl["vy_dao_pass"] = np.ones(len(tbl), dtype=np.int16)
        n_pass1_dao_m = int(len(tbl)) if tbl is not None else 0
        _h_m, _wpx_m = int(data0.shape[0]), int(data0.shape[1])
        try:
            _sigma_p2_m = float(_cfg_dao.masterstar_dao_pass2_sigma)
        except (TypeError, ValueError):
            _sigma_p2_m = 1.9
        try:
            _ctol_p2_m = float(_cfg_dao.masterstar_dao_pass2_center_tol_px)
        except (TypeError, ValueError):
            _ctol_p2_m = 5.0
        tbl, _n_unmatched_master, _n_pass2_raw_m = _dao_targeted_pass2_unmatched_gaia(
            data0,
            tbl,
            cat_df=m_valid,
            wcs_obj=wcs_obj,
            bfac=int(bfac),
            fwhm_px=float(max(1.2, _base_fw_m)),
            pass2_sigma=float(_sigma_p2_m),
            pass2_center_tol_px=float(_ctol_p2_m),
            match_sep_arcsec=float(match_sep_arcsec),
            wpx=int(_wpx_m),
            h=int(_h_m),
        )
        n_merged_dao_m = int(len(tbl)) if tbl is not None else 0
        LOGGER.info(
            "[DAO pass 1 master] %d detections, %d MASTERSTAR unmatched",
            int(n_pass1_dao_m),
            int(_n_unmatched_master),
        )
        LOGGER.info(
            "[DAO pass 2 master] %d additional detections from %d targeted positions",
            int(_n_pass2_raw_m),
            int(_n_unmatched_master),
        )
        LOGGER.info("[DAO total master] %d detections after merge", int(n_merged_dao_m))
    except Exception:  # noqa: BLE001
        tbl = None
        bfac = 1

    _fb_sat = fallback_saturate_adu

    if sat_diag_ctx is not None and getattr(sat_diag_ctx, "sat_adu", None) is not None:
        sat_limit = float(sat_diag_ctx.sat_adu)
        sat_limit_src = str(getattr(sat_diag_ctx, "sat_source", "sat_diag"))
    else:
        sat_limit, sat_limit_src = _effective_saturation_limit(
            hdr, fallback_adu=_fb_sat, equipment_saturate_adu=equipment_saturate_adu
        )
    sat_frac = float(saturate_level_fraction)
    sat_frac = min(max(sat_frac, 0.5), 1.0)

    foot_meta = {
        "catalog_footprint": {"method": "master_reference_only"},
        "saturation": {
            "effective_limit_adu": float(sat_limit) if sat_limit is not None else None,
            "limit_source": sat_limit_src,
            "plateau_half_inner_px": 1,
            "plateau_rel": 0.996,
            "plateau_min_pixels": 5,
        },
    }

    empty_meta = {
        "n_detected": 0,
        "n_detected_dao": 0,
        "n_matched": 0,
        "n_matched_before_mag_limit": 0,
        "catalog_rows": int(len(master_df)),
        "catalog_match_mode": export_catalog_match_mode_from_internal(match_mode),
        "n_likely_saturated": 0,
        "n_saturated_from_peak": 0,
        "n_saturated_plateau": 0,
        "saturate_limit_adu": float(sat_limit) if sat_limit is not None else None,
        "saturate_limit_source": sat_limit_src,
        "n_vsx_in_field": 0,
        "n_gaia_variable_in_field": 0,
        "field_catalog_cone_csv": None,
        "dao_threshold_sigma": float(dao_threshold_sigma),
        "dao_fwhm_px": float(max(1.2, _base_fw_m)),
        "dao_detect_binning": 1,
        "match_sep_arcsec_requested": float(match_sep_arcsec),
        "match_sep_arcsec_effective": float(match_sep_arcsec),
        "plate_scale_arcsec_per_px": (
            float(plate_scale_arcsec_per_px) if plate_scale_arcsec_per_px is not None else None
        ),
        **foot_meta,
    }

    _chip_h, _chip_wpx = int(arr.shape[0]), int(arr.shape[1])

    if tbl is None or len(tbl) == 0:
        df_out = pd.DataFrame()
        df_out = _proc_rename_det_names_to_catalog_id(df_out)
        empty_meta["faintest_mag_limit"] = float(faintest_mag_limit) if faintest_mag_limit is not None else None
        empty_meta["n_dropped_fainter_than_limit"] = 0
        empty_meta["n_detected"] = int(len(df_out))
        empty_meta["n_matched"] = int(
            df_out["catalog_id"].fillna("").astype(str).str.strip().ne("").sum()
        ) if len(df_out) and "catalog_id" in df_out.columns else 0
        return df_out, empty_meta

    _fwhm_used_m = float(max(1.2, _base_fw_m / float(bfac)))
    _d_h, _d_w = int(data_dao.shape[0]), int(data_dao.shape[1])
    _keep = _dao_spatial_flux_cap_row_indices(
        tbl, max_n=int(max_catalog_rows), width_px=float(_d_w), height_px=float(_d_h)
    )
    tbl = tbl[_keep]
    tbl.sort("flux")
    tbl = tbl[::-1]
    n = len(tbl)
    xb = np.asarray(tbl["x_centroid"], dtype=np.float64)
    yb = np.asarray(tbl["y_centroid"], dtype=np.float64)
    x, y = _dao_xy_binned_to_full(xb, yb, bfac)
    flux = np.asarray(tbl["flux"], dtype=np.float64)
    peak_dao = np.asarray(tbl["peak"], dtype=np.float64) if "peak" in tbl.colnames else np.full(n, np.nan)

    match_thr = float(match_sep_arcsec)

    def _pixel_nn_match(*, dist_thr_px: float) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Pixel-space NN match against MASTERSTAR x/y.

        Returns: (icomp, sep_arr, oix)
        - icomp: indices into the *filtered* master_xy array (as from KDTree query)
        - sep_arr: distance in px (NaN when unmatched)
        - oix: mapping from filtered master_xy indices back to m_valid row indices
        """
        ic: np.ndarray
        sep: np.ndarray
        ox: np.ndarray | None
        if "x" not in m_valid.columns or "y" not in m_valid.columns:
            ic = np.zeros(n, dtype=np.int64)
            sep = np.full(n, np.nan, dtype=np.float64)
            ox = None
            return ic, sep, ox
        try:
            from scipy.spatial import cKDTree  # type: ignore

            mx = pd.to_numeric(m_valid["x"], errors="coerce").to_numpy(dtype=np.float64)
            my = pd.to_numeric(m_valid["y"], errors="coerce").to_numpy(dtype=np.float64)
            okxy = np.isfinite(mx) & np.isfinite(my)
            mx2 = mx[okxy]
            my2 = my[okxy]
            ox = np.nonzero(okxy)[0].astype(np.int64)
            if mx2.size == 0:
                ic = np.zeros(n, dtype=np.int64)
                sep = np.full(n, np.nan, dtype=np.float64)
                return ic, sep, ox
            master_xy = np.column_stack([mx2, my2])
            det_xy = np.column_stack([np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)])
            tree = cKDTree(master_xy)
            dist_px, idx = tree.query(det_xy, distance_upper_bound=float(dist_thr_px))
            ic = np.asarray(idx, dtype=np.int64)
            sep = np.asarray(dist_px, dtype=np.float64)  # px
            sep[~np.isfinite(sep)] = np.nan
            return ic, sep, ox
        except Exception:  # noqa: BLE001
            ic = np.zeros(n, dtype=np.int64)
            sep = np.full(n, np.nan, dtype=np.float64)
            ox = None
            return ic, sep, ox

    icomp: np.ndarray | None = None
    sep_arcsec_arr: np.ndarray
    oix: np.ndarray | None = None
    ra_deg: np.ndarray
    dec_deg: np.ndarray
    _frame_on_ref_grid = _fits_header_vy_algn_aligned(hdr)

    # Robust strategy:
    # - If celestial WCS exists, try sky-match first (arcsec threshold).
    # - If sky-match looks suspiciously bad (e.g. WCS offset), fall back to pixel match
    #   only when the frame is on the reference alignment grid (VY_ALGN=True).
    # Matching unaligned DAO xy to the reference pixel grid is invalid by construction.
    if getattr(wcs_obj, "has_celestial", False):
        ra_deg, dec_deg = _all_pix2world_icrs_deg(wcs_obj, x, y)
        det_coords = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
        idx_sky, sep2d, _ = det_coords.match_to_catalog_sky(master_coords)
        sep_sky = np.asarray(sep2d.to_value(u.arcsec), dtype=np.float64)
        ic_sky = np.asarray(idx_sky, dtype=np.int64)

        # Heuristic: if typical nearest-neighbor sep is far above the threshold,
        # the WCS is likely offset (e.g. wrong reference grid / flip / stale solve).
        med_sep = float(np.nanmedian(sep_sky)) if np.isfinite(np.nanmedian(sep_sky)) else float("inf")
        if (not np.isfinite(med_sep)) or (med_sep > max(30.0, match_thr * 3.0)):
            if not _frame_on_ref_grid:
                icomp = ic_sky
                sep_arcsec_arr = sep_sky
                oix = None
                match_mode = "sky_unaligned_no_pixel_fallback"
            else:
                ic_px, sep_px, oix_px = _pixel_nn_match(dist_thr_px=float(6.0))
                # Prefer pixel match if it yields any finite distances.
                if int(np.count_nonzero(np.isfinite(sep_px))) > 0:
                    icomp = ic_px
                    sep_arcsec_arr = sep_px  # px (kept numeric for mask)
                    oix = oix_px
                    match_mode = "pixel_fallback_bad_wcs"
                    ra_deg = np.full(n, np.nan, dtype=np.float64)
                    dec_deg = np.full(n, np.nan, dtype=np.float64)
                else:
                    icomp = ic_sky
                    sep_arcsec_arr = sep_sky
                    oix = None
                    match_mode = "sky"
        else:
            icomp = ic_sky
            sep_arcsec_arr = sep_sky
            oix = None
            match_mode = "sky"
    else:
        ra_deg = np.full(n, np.nan, dtype=np.float64)
        dec_deg = np.full(n, np.nan, dtype=np.float64)
        if not _frame_on_ref_grid:
            match_mode = "nondet_unaligned_no_wcs"
            icomp = np.zeros(n, dtype=np.int64)
            sep_arcsec_arr = np.full(n, np.nan, dtype=np.float64)
            oix = None
        else:
            match_mode = "pixel_fallback_no_wcs"
            ic_px, sep_px, oix_px = _pixel_nn_match(dist_thr_px=float(15.0))
            icomp = ic_px
            sep_arcsec_arr = sep_px  # px (kept numeric for mask)
            oix = oix_px

    nm_m = len(m_valid)
    idx_det = np.arange(1, n + 1, dtype=np.int32)
    det_str = np.array([f"DET_{i:04d}" for i in idx_det], dtype=object)
    finite_sep = np.isfinite(sep_arcsec_arr)
    icomp_a = icomp if icomp is not None else np.zeros(n, dtype=np.int64)
    if match_mode in ("pixel_fallback_no_wcs", "pixel_fallback_bad_wcs") and oix is not None:
        ok_ix = (icomp_a >= 0) & (icomp_a < int(len(oix)))
        cat_row = np.full(n, -1, dtype=np.int64)
        cat_row[ok_ix] = oix[icomp_a[ok_ix]]
        thr_px = float(15.0) if match_mode == "pixel_fallback_no_wcs" else float(6.0)
        matched = finite_sep & ok_ix & (sep_arcsec_arr <= thr_px) & (cat_row >= 0)
        # For output semantics, keep match_sep_arcsec as NaN when unmatched (already handled).
    else:
        cat_row = np.clip(icomp_a.astype(np.int64, copy=False), 0, max(nm_m - 1, 0))
        matched = finite_sep & (sep_arcsec_arr <= match_thr) & (nm_m > 0)

    safe = np.clip(np.where(matched, cat_row, 0), 0, max(nm_m - 1, 0))
    cat_s_m = (
        m_valid["catalog"].fillna("").astype(str).str.strip().to_numpy(dtype=object)
        if "catalog" in m_valid.columns
        else np.array([""] * nm_m, dtype=object)
    )
    cid_m = (
        m_valid["catalog_id"].fillna("").astype(str).str.strip().to_numpy(dtype=object)
        if "catalog_id" in m_valid.columns
        else np.array([""] * nm_m, dtype=object)
    )
    nm_c = (
        m_valid["name"].fillna("").astype(str).str.strip().to_numpy(dtype=object)
        if "name" in m_valid.columns
        else np.array([""] * nm_m, dtype=object)
    )
    mag_m = (
        pd.to_numeric(m_valid["mag"], errors="coerce").to_numpy(dtype=np.float64)
        if "mag" in m_valid.columns
        else np.full(nm_m, np.nan, dtype=np.float64)
    )
    bv_m = (
        pd.to_numeric(m_valid["b_v"], errors="coerce").to_numpy(dtype=np.float64)
        if "b_v" in m_valid.columns
        else np.full(nm_m, np.nan, dtype=np.float64)
    )
    gn_m = (
        pd.to_numeric(m_valid["gaia_nss"], errors="coerce").to_numpy(dtype=np.float64)
        if "gaia_nss" in m_valid.columns
        else np.full(nm_m, np.nan, dtype=np.float64)
    )
    gq_m = (
        pd.to_numeric(m_valid["gaia_qso"], errors="coerce").to_numpy(dtype=np.float64)
        if "gaia_qso" in m_valid.columns
        else np.full(nm_m, np.nan, dtype=np.float64)
    )
    gg_m = (
        pd.to_numeric(m_valid["gaia_gal"], errors="coerce").to_numpy(dtype=np.float64)
        if "gaia_gal" in m_valid.columns
        else np.full(nm_m, np.nan, dtype=np.float64)
    )
    vx_m = (
        m_valid["vsx_known_variable"].fillna(False).astype(bool).to_numpy()
        if "vsx_known_variable" in m_valid.columns
        else np.zeros(nm_m, dtype=bool)
    )
    gv_m = (
        m_valid["gaia_dr3_variable_catalog"].fillna(False).astype(bool).to_numpy()
        if "gaia_dr3_variable_catalog" in m_valid.columns
        else np.zeros(nm_m, dtype=bool)
    )
    if "catalog_known_variable" in m_valid.columns:
        ck_m = m_valid["catalog_known_variable"].fillna(False).astype(bool).to_numpy()
    else:
        ck_m = vx_m | gv_m

    _fwhm_cent = float(max(1.2, _base_fw_m / float(bfac)))
    if _frame_on_ref_grid:
        x, y, _n_dao_wcs_fallback = _lock_matched_centroids_to_master_grid(
            arr,
            x,
            y,
            matched=matched,
            safe=safe,
            master_df=m_valid,
            fwhm_px=_fwhm_cent,
        )
    else:
        x, y, _n_dao_wcs_fallback = _apply_dao_centroid_wcs_guard(
            x,
            y,
            matched=matched,
            safe=safe,
            master_df=m_valid,
            fwhm_px=_fwhm_cent,
            max_shift_fwhm=float(getattr(_cfg_dao, "dao_centroid_max_shift_fwhm", 1.0)),
        )

    pmax_arr = _box_peaks_at_centroids(arr, x, y)
    _frame_max_adu = float(np.nanmax(arr))
    _empirical_clip_adu = _detect_empirical_clip_level_adu(arr)
    _peak_sat_lim = _resolve_peak_saturation_limit_adu(
        camera_sat_limit_adu=sat_limit,
        saturate_fraction=sat_frac,
        sky_median_adu=float(med),
        frame_max_adu=_frame_max_adu,
        empirical_clip_adu=_empirical_clip_adu,
    )
    _sat_block = _vectorized_star_saturation_columns(
        arr,
        x,
        y,
        sat_limit=_peak_sat_lim,
        sat_frac=sat_frac,
        peak_dao=peak_dao,
        peak_max_adu=pmax_arr,
    )
    _sat_csv, n_sat_pk, n_sat_pl = _proc_sat_block_for_csv(_sat_block)

    cat_sel = cat_s_m[safe]
    cid_sel = cid_m[safe]
    nm_sel = nm_c[safe]
    name_cand = np.where(
        nm_sel != "",
        nm_sel,
        np.where(cid_sel != "", cid_sel, np.where(cat_sel != "", cat_sel, det_str)),
    )
    name_out = np.where(matched, name_cand, det_str)

    mag_out = np.full(n, np.nan, dtype=np.float64)
    bv_out = np.full(n, np.nan, dtype=np.float64)
    gn_out = np.full(n, np.nan, dtype=np.float64)
    gq_out = np.full(n, np.nan, dtype=np.float64)
    gg_out = np.full(n, np.nan, dtype=np.float64)
    mag_out[matched] = mag_m[safe[matched]]
    bv_out[matched] = bv_m[safe[matched]]
    gn_out[matched] = gn_m[safe[matched]]
    gq_out[matched] = gq_m[safe[matched]]
    gg_out[matched] = gg_m[safe[matched]]

    vx_out = np.zeros(n, dtype=bool)
    gv_out = np.zeros(n, dtype=bool)
    ck_out = np.zeros(n, dtype=bool)
    vx_out[matched] = vx_m[safe[matched]]
    gv_out[matched] = gv_m[safe[matched]]
    ck_out[matched] = ck_m[safe[matched]]

    cat_out = np.array([""] * n, dtype=object)
    cid_out = np.array([""] * n, dtype=object)
    cat_out[matched] = cat_s_m[safe[matched]]
    cid_out[matched] = cid_m[safe[matched]]

    n_matched = int(np.count_nonzero(matched & (cat_s_m[safe].astype(str) != "")))

    df_out = pd.DataFrame(
        {
            "name": name_out,
            "ra_deg": ra_deg,
            "dec_deg": dec_deg,
            "mag": mag_out,
            "b_v": bv_out,
            "catalog": cat_out,
            "catalog_id": cid_out,
            "x": x,
            "y": y,
            "flux": flux,
            "vsx_known_variable": vx_out,
            "gaia_dr3_variable_catalog": gv_out,
            **_sat_csv,
        }
    )
    n_detected_dao = int(n)
    n_matched_before_mag = int(n_matched)
    n_before_mag = len(df_out)
    if faintest_mag_limit is not None and np.isfinite(float(faintest_mag_limit)):
        lim_m = float(faintest_mag_limit)
        mcol = pd.to_numeric(df_out["mag"], errors="coerce")
        # Drop only matched stars fainter than limit; keep unmatched detections (no catalog mag) for QA.
        df_out = df_out.loc[mcol.isna() | (mcol <= lim_m)].reset_index(drop=True)
        meta_mag = {
            "faintest_mag_limit": lim_m,
            "n_dropped_fainter_than_limit": int(n_before_mag - len(df_out)),
        }
    else:
        meta_mag = {"faintest_mag_limit": None, "n_dropped_fainter_than_limit": 0}

    df_out = _proc_rename_det_names_to_catalog_id(df_out)

    n_sat = int(df_out["likely_saturated"].sum()) if len(df_out) and "likely_saturated" in df_out.columns else 0
    cat_nonempty = (
        df_out["catalog_id"].fillna("").astype(str).str.strip().ne("")
        if len(df_out) and "catalog_id" in df_out.columns
        else pd.Series([], dtype=bool)
    )
    n_matched_final = int(cat_nonempty.sum()) if len(df_out) else 0
    meta = {
        "n_detected_dao": n_detected_dao,
        "n_detected": int(len(df_out)),
        "n_matched_before_mag_limit": n_matched_before_mag,
        "n_matched": n_matched_final,
        "catalog_rows": int(len(master_df)),
        "catalog_match_mode": export_catalog_match_mode_from_internal(match_mode),
        "n_likely_saturated": n_sat,
        "n_saturated_from_peak": n_sat_pk,
        "n_saturated_plateau": n_sat_pl,
        "n_dao_wcs_centroid_fallback": int(_n_dao_wcs_fallback),
        "saturate_limit_adu": float(sat_limit) if sat_limit is not None else None,
        "saturate_limit_source": sat_limit_src,
        "n_vsx_in_field": 0,
        "n_gaia_variable_in_field": 0,
        **foot_meta,
        "field_catalog_cone_csv": None,
        "dao_threshold_sigma": float(dao_threshold_sigma),
        "dao_fwhm_px": _fwhm_used_m,
        "dao_detect_binning": int(bfac),
        "match_sep_arcsec_requested": float(match_sep_arcsec),
        "match_sep_arcsec_effective": float(match_thr),
        "plate_scale_arcsec_per_px": (
            float(plate_scale_arcsec_per_px) if plate_scale_arcsec_per_px is not None else None
        ),
        **meta_mag,
    }
    if int(_n_dao_wcs_fallback) > 0:
        LOGGER.info(
            "[BATCH-E E.2] centroid WCS fallback triggered on %d star-frames",
            int(_n_dao_wcs_fallback),
        )
    if (
        sat_diag_ctx is not None
        and raw_data is not None
        and raw_hdr is not None
        and len(df_out) > 0
    ):
        try:
            from sat_diag import apply_raw_peaks_to_proc_df  # noqa: PLC0415

            apply_raw_peaks_to_proc_df(
                df_out,
                np.asarray(raw_data),
                raw_hdr,
                sat_diag_ctx,
                ref_ra=ref_ra_deg,
                ref_dec=ref_dec_deg,
                drift_ref_catalog_id=drift_ref_catalog_id,
                aligned_hdr=hdr,
            )
            n_sat = (
                int(df_out["likely_saturated_raw"].sum())
                if "likely_saturated_raw" in df_out.columns
                else n_sat
            )
            meta["sat_limit_source"] = str(getattr(sat_diag_ctx, "sat_source", sat_limit_src))
            meta["sat_peak_source"] = str(
                getattr(sat_diag_ctx, "sat_peak_source", "PLACED_APERTURE")
            )
            meta["raw_peaks_used"] = True
        except Exception as _sat_exc:  # noqa: BLE001
            LOGGER.warning("[SAT-DIAG] raw peak merge failed for %s: %s", frame_name, _sat_exc)
    return df_out, meta


def _icrs_center_radius_from_hdr_data(
    hdr: fits.Header,
    data: Any,
    *,
    plate_solve_fov_deg: float | None = None,
) -> tuple[SkyCoord, float, int, int] | None:
    """ICRS center + cone radius from an in-memory frame (RAM handoff), same semantics as disk scan."""
    import numpy as np

    d = np.asarray(data)
    if d.ndim != 2:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            w = WCS(hdr)
        if not w.has_celestial:
            return None
        h_i, wpx_i = int(d.shape[0]), int(d.shape[1])
        c_i, r_i = _effective_field_catalog_cone_radius_deg(
            w, h_i, wpx_i, plate_solve_fov_deg, fits_header=hdr
        )
        return c_i, r_i, wpx_i, h_i
    except Exception:  # noqa: BLE001
        return None


def _export_first_icrs_center_radius(
    files: list[Path],
    *,
    plate_solve_fov_deg: float | None = None,
) -> tuple[SkyCoord, float, int, int] | None:
    import numpy as np

    for fp in files:
        try:
            with fits.open(fp, memmap=False) as h:
                hdr = h[0].header
                d = np.asarray(h[0].data)
                if d.ndim != 2:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FITSFixedWarning)
                    w = WCS(hdr)
                if not w.has_celestial:
                    continue
                h_i, wpx_i = int(d.shape[0]), int(d.shape[1])
                c_i, r_i = _effective_field_catalog_cone_radius_deg(
                    w, h_i, wpx_i, plate_solve_fov_deg, fits_header=hdr
                )
                return c_i, r_i, wpx_i, h_i
        except Exception:  # noqa: BLE001
            continue
    return None


def _prefetch_export_shared_catalog_for_process_pool(
    *,
    files: list[Path] | None = None,
    reference_hdr_data: tuple[Any, Any] | None = None,
    field_cat_path: Path,
    cat_df: pd.DataFrame | None,
    vsx_df: pd.DataFrame | None,
    gaia_variable_df: pd.DataFrame | None,
    gaia_db_path: Path | None,
    gaia_local_max_mag: float | None,
    export_cat_local: bool,
    plate_solve_fov_deg: float | None = None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None, Any]:
    """Load cone + VSX/Gaia + KD-tree in the parent before ``ProcessPoolExecutor`` (workers share no memory)."""
    cr: tuple[SkyCoord, float, int, int] | None = None
    if reference_hdr_data is not None:
        _h0, _d0 = reference_hdr_data
        cr = _icrs_center_radius_from_hdr_data(_h0, _d0, plate_solve_fov_deg=plate_solve_fov_deg)
    if cr is None and files:
        cr = _export_first_icrs_center_radius(files, plate_solve_fov_deg=plate_solve_fov_deg)
    c_df = cat_df
    v_df = vsx_df
    g_df = gaia_variable_df

    if cr is not None:
        c_i, r_i, wpx_i, h_i = cr
        _cfg_pf = AppConfig()
        if c_df is None or getattr(c_df, "empty", True):
            _gaia_db_path = Path(gaia_db_path) if gaia_db_path is not None else None
            if _gaia_db_path is None:
                try:
                    _gp = (_cfg_pf.gaia_db_path or "").strip()
                    if _gp:
                        _gaia_db_path = Path(_gp)
                except Exception:  # noqa: BLE001
                    _gaia_db_path = None
            c_df = _query_gaia_local(
                center=c_i,
                radius_deg=r_i,
                gaia_db_path=_gaia_db_path,
                max_mag=float(gaia_local_max_mag) if gaia_local_max_mag is not None else None,
            )
            if c_df is not None and len(c_df) > 0:
                try:
                    field_cat_path.parent.mkdir(parents=True, exist_ok=True)
                    _vyvar_df_to_csv(c_df, field_cat_path)
                    _write_field_catalog_cone_meta(
                        field_cat_path,
                        center=c_i,
                        radius_deg=float(r_i),
                        naxis1=int(wpx_i),
                        naxis2=int(h_i),
                        plate_solve_fov_deg=plate_solve_fov_deg,
                    )
                except Exception as exc:  # noqa: BLE001
                    LOGGER.debug("[PIPE] shared-catalog prefetch write skipped: %s", exc)
                    pass

        if v_df is None or getattr(v_df, "empty", True):
            _vsx_p: Path | None = None
            try:
                _vsp = str(_cfg_pf.vsx_local_db_path or "").strip()
                if _vsp:
                    _vsx_p = Path(_vsp).expanduser().resolve()
            except Exception:  # noqa: BLE001
                _vsx_p = None
            if _vsx_p is not None and _vsx_p.is_file():
                v_df = _query_vsx_local(
                    center=c_i,
                    radius_deg=float(r_i),
                    vsx_db_path=_vsx_p,
                )
            else:
                v_df = pd.DataFrame()
        if g_df is None:
            g_df = pd.DataFrame()

    kd_pack = None
    if c_df is not None and not getattr(c_df, "empty", True) and len(c_df) >= 120:
        kd_pack = build_ucac_catalog_kdtree(c_df)
        if kd_pack is not None:
            LOGGER.info(
                "Per-frame catalog: shared cKDTree for %s cone rows (process pool)",
                len(c_df),
            )
    return c_df, v_df, g_df, kd_pack


_EXPORT_PER_FRAME_WORKER_STATE: dict[str, Any] = {}


def _init_export_per_frame_worker(state: dict[str, Any]) -> None:
    global _EXPORT_PER_FRAME_WORKER_STATE
    _EXPORT_PER_FRAME_WORKER_STATE = state
    if not bool(state.get("use_master_fast_path")):
        return
    mpath = str(state.get("masterstar_fits_path") or "").strip()
    if not mpath or not Path(mpath).is_file():
        return
    try:
        with fits.open(mpath, memmap=False) as h:
            w = WCS(h[0].header)
            if w.has_celestial:
                _EXPORT_PER_FRAME_WORKER_STATE["ref_wcs"] = w
            d = h[0].data
            if d is not None and getattr(d, "ndim", 0) == 2:
                _EXPORT_PER_FRAME_WORKER_STATE["masterstar_data_shape"] = (
                    int(d.shape[0]),
                    int(d.shape[1]),
                )
    except Exception:  # noqa: BLE001
        pass


def _airmass_from_altitude_deg(alt_deg: float) -> float:
    """Kasten & Young (1989) airmass from altitude in degrees."""
    alt_rad = math.radians(float(alt_deg))
    return 1.0 / (
        math.sin(alt_rad) + 0.50572 * (float(alt_deg) + 6.07995) ** (-1.6364)
    )


def _compute_airmass_from_altaz(
    hdr: fits.Header,
    cfg: AppConfig | None = None,
    *,
    db: VyvarDatabase | None = None,
    draft_id: int | None = None,
) -> float:
    """Compute airmass from field center RA/Dec + observer site + mid-exposure JD (AltAz).

    Site is resolved via the unified resolver (per-draft ID_LOCATION -> header
    SITELAT -> flagged config) so airmass uses the same location as BJD/HJD and is
    no longer tied to ``cfg.observer_*`` alone (config-drift trap).
    """
    from time_utils import resolve_observer_location  # noqa: PLC0415

    lat_r, lon_r, alt_r = resolve_observer_location(hdr, db, draft_id, cfg=cfg)
    if lat_r is None or lon_r is None:
        return float("nan")
    lat = float(lat_r)
    lon = float(lon_r)
    alt_m = float(alt_r) if alt_r is not None else 0.0
    if lat == 0.0 and lon == 0.0:
        return float("nan")

    from time_utils import mid_exposure_jd  # noqa: PLC0415

    jd = mid_exposure_jd(hdr)
    if jd is None:
        return float("nan")

    # Safe: airmass from CRVAL1/2 or OBJCTRA/DEC; returns nan if missing - handled downstream.
    ra = _header_float_tu(hdr, "CRVAL1")
    dec = _header_float_tu(hdr, "CRVAL2")
    if ra is None or dec is None:
        if "OBJCTRA" in hdr and "OBJCTDEC" in hdr:
            try:
                from time_utils import _parse_objctradec  # noqa: PLC0415

                ra, dec = _parse_objctradec(str(hdr["OBJCTRA"]), str(hdr["OBJCTDEC"]))
            except Exception:  # noqa: BLE001
                ra, dec = None, None
    if ra is None or dec is None:
        return float("nan")

    try:
        from astropy.coordinates import AltAz, EarthLocation, SkyCoord  # noqa: PLC0415
        from astropy.time import Time  # noqa: PLC0415

        location = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=alt_m * u.m)
        t = Time(float(jd), format="jd", scale="utc", location=location)
        target = SkyCoord(ra=float(ra) * u.deg, dec=float(dec) * u.deg, frame="icrs")
        altaz = target.transform_to(AltAz(obstime=t, location=location))
        alt_deg = float(altaz.alt.deg)
        if not math.isfinite(alt_deg) or alt_deg < 5.0 or alt_deg > 90.0:
            return float("nan")
        am = _airmass_from_altitude_deg(alt_deg)
        if 0.9 <= am <= 10.0:
            LOGGER.info(
                "Airmass from AltAz fallback: %.4f (alt=%.2f deg)",
                am,
                alt_deg,
            )
            return round(am, 5)
    except Exception:  # noqa: BLE001
        return float("nan")
    return float("nan")


def _extract_airmass_from_header(
    hdr: fits.Header,
    cfg: AppConfig | None = None,
    *,
    db: VyvarDatabase | None = None,
    draft_id: int | None = None,
) -> float:
    """Extract airmass from a FITS header for per-frame catalog metadata.

    Priority order:
    1. ``AIRMASS`` / ``AIRMAS`` / ``SECZ`` header keywords when in the physical range.
    2. ``ALT_OBJ`` / ``OBJCTALT`` / ``ALTITUDE`` / ``TELALT`` converted via
       Kasten & Young (1989) (:func:`_airmass_from_altitude_deg`).
    3. Field-center AltAz computed from WCS/object coordinates, observer site, and
       mid-exposure JD (:func:`_compute_airmass_from_altaz`), also using Kasten & Young (1989).

    Returns ``float('nan')`` when no path yields a finite value.
    """
    # Priamy keyword
    for kw in ("AIRMASS", "AIRMAS", "SECZ"):
        val = hdr.get(kw)
        if val is not None:
            try:
                v = float(val)
                if 0.9 <= v <= 10.0:  # fyzikalny rozsah
                    return v
            except (TypeError, ValueError):
                pass

    # Fallback: altitude -> airmass cez aproximaciu Kasten & Young (1989)
    for kw in ("ALT_OBJ", "OBJCTALT", "ALTITUDE", "TELALT"):
        val = hdr.get(kw)
        if val is not None:
            try:
                alt_deg = float(val)
                if 5.0 <= alt_deg <= 90.0:
                    am = _airmass_from_altitude_deg(alt_deg)
                    if 0.9 <= am <= 10.0:
                        return round(am, 5)
            except (TypeError, ValueError):
                pass

    am_altaz = _compute_airmass_from_altaz(hdr, cfg, db=db, draft_id=draft_id)
    if math.isfinite(am_altaz):
        return am_altaz

    return float("nan")


def _cfg_from_export_worker_state(st: dict[str, Any]) -> AppConfig:
    """Rebuild minimal AppConfig observer fields from per-frame worker state dict."""
    c = AppConfig()
    c.observer_lat = float(st.get("observer_lat", 0.0) or 0.0)
    c.observer_lon = float(st.get("observer_lon", 0.0) or 0.0)
    c.observer_alt_m = float(st.get("observer_alt_m", 0.0) or 0.0)
    return c


def _export_per_frame_run_catalog_core(
    base_path: Path,
    hdr: fits.Header,
    data: Any,
    st: dict[str, Any],
) -> dict[str, Any]:
    from pipeline import detect_stars_and_match_catalog  # noqa: PLC0415

    log_event("DEBUG: per-frame worker entry point called")
    fname = base_path.name
    st["epsf_frame_name"] = fname
    _idx_map = st.get("epsf_frame_index_by_name")
    if isinstance(_idx_map, dict):
        st["epsf_frame_index"] = _idx_map.get(fname)
    debug_pixel_match: dict[str, Any] = {
        "file": fname,
        "use_fast": bool(st.get("use_master_fast_path")),
        "master_cols": None,
        "have_x": None,
        "have_y": None,
        "match_mode": None,
        "plate_scale_arcsec_per_px": None,
        "n_matched": None,
    }
    deferred_writes: list[tuple[str, pd.DataFrame]] = []
    if not _has_valid_wcs(hdr):
        return {
            "file": fname,
            "status": "no_wcs",
            "csv": "",
            "deferred_writes": deferred_writes,
            "infolog_messages": [],
        }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        _ = WCS(hdr)
    h_i, wpx_i = data.shape

    master_tab = st.get("master_tab")
    use_master_fast_path = bool(st.get("use_master_fast_path"))
    master_only_mode = bool(st.get("master_only_mode"))

    use_fast = (
        use_master_fast_path
        and master_tab is not None
        and not getattr(master_tab, "empty", True)
    )
    if master_only_mode and not use_fast:
        return {
            "file": fname,
            "status": "error: master_only_mode requires masterstars_full_match.csv",
            "csv": "",
            "deferred_writes": deferred_writes,
            "infolog_messages": [
                f"Per-frame catalog {fname}: master_only_mode requires masterstars_full_match.csv",
            ],
        }

    cat_df = st.get("cat_df")
    vsx_df = st.get("vsx_df")
    gaia_variable_df = st.get("gaia_variable_df")
    kd_pack = st.get("kd_pack")
    export_cat_local = bool(st.get("export_cat_local"))

    df: pd.DataFrame
    meta: dict[str, Any]
    if use_fast:
        try:
            # One-time debug: verify MASTERSTAR pixel columns used for KD-tree matching.
            try:
                if not bool(st.get("_debug_logged_master_xy")) and master_tab is not None:
                    st["_debug_logged_master_xy"] = True
                    cols = [c for c in ("x", "y", "ra_deg", "dec_deg") if c in master_tab.columns]
                    if cols:
                        log_event(
                            "DEBUG: masterstars_full_match.csv sample columns "
                            + ",".join(cols)
                            + ":\n"
                            + master_tab.loc[:, cols].head(5).to_string(index=False)
                        )
            except Exception:  # noqa: BLE001
                pass
            try:
                if master_tab is not None and "catalog_id" in master_tab.columns and "ra_deg" in master_tab.columns:
                    print(
                        "DEBUG SKY-MATCH: "
                        f"master_tab rows={len(master_tab)}, "
                        f"catalog_id notna={int(master_tab['catalog_id'].notna().sum())}, "
                        f"ra_deg notna={int(master_tab['ra_deg'].notna().sum())}"
                    )
            except Exception:  # noqa: BLE001
                pass
            try:
                if master_tab is not None:
                    debug_pixel_match["master_cols"] = list(master_tab.columns[:10])
                    debug_pixel_match["have_x"] = bool("x" in master_tab.columns)
                    debug_pixel_match["have_y"] = bool("y" in master_tab.columns)
            except Exception:  # noqa: BLE001
                pass
            df, meta = detect_stars_match_master_reference(
                data,
                hdr,
                master_tab,
                max_catalog_rows=int(st["max_catalog_rows"]),
                match_sep_arcsec=float(st["catalog_match_max_sep_arcsec"]),
                saturate_level_fraction=float(st["saturate_level_fraction"]),
                faintest_mag_limit=st.get("faintest_mag_limit"),
                dao_threshold_sigma=float(st["dao_threshold_sigma"]),
                dao_fwhm_px=float(st.get("dao_fwhm_px", 2.5)),
                equipment_saturate_adu=st.get("equipment_saturate_adu"),
                frame_name=fname,
                sat_diag_ctx=_sat_ctx_from_worker(st),
                raw_data=_load_raw_for_frame(st, fname),
                raw_hdr=_load_raw_hdr_for_frame(st, fname),
                ref_ra_deg=st.get("sat_diag_ref_ra"),
                ref_dec_deg=st.get("sat_diag_ref_dec"),
                drift_ref_catalog_id=st.get("sat_diag_ref_catalog_id"),
            )
        except Exception as exc:  # noqa: BLE001
            logging.error('[EXC-0373] detect_stars_and_match_catalog exception returns status error with empty csv for that f...: %s', exc)
            return {
                "file": fname,
                "status": f"error: {exc}",
                "csv": "",
                "deferred_writes": deferred_writes,
                "infolog_messages": [f"Per-frame catalog {fname}: {exc}"],
                "debug_pixel_match": debug_pixel_match,
            }
    else:
        if master_only_mode:
            return {
                "file": fname,
                "status": "error: master_only_mode fallback to Gaia cone is disabled",
                "csv": "",
                "deferred_writes": deferred_writes,
                "infolog_messages": [
                    f"Per-frame catalog {fname}: master_only_mode fallback to Gaia cone is disabled",
                ],
            }
        try:
            df, meta = detect_stars_and_match_catalog(
                data,
                hdr,
                max_catalog_rows=int(st["max_catalog_rows"]),
                cat_df=cat_df,
                vsx_df=vsx_df,
                gaia_variable_df=gaia_variable_df,
                match_sep_arcsec=float(st["catalog_match_max_sep_arcsec"]),
                saturate_level_fraction=float(st["saturate_level_fraction"]),
                faintest_mag_limit=st.get("faintest_mag_limit"),
                field_catalog_export_path=None,
                dao_threshold_sigma=float(st["dao_threshold_sigma"]),
                dao_fwhm_px=float(st.get("dao_fwhm_px", 2.5)),
                equipment_saturate_adu=st.get("equipment_saturate_adu"),
                catalog_local_gaia_only=export_cat_local,
                catalog_kd_pack=kd_pack,
                plate_solve_fov_deg=st.get("plate_solve_fov_deg"),
                fov_database_path=st.get("database_path"),
                fov_equipment_id=int(st["equipment_id"]) if st.get("equipment_id") is not None else None,
                fov_draft_id=int(st["draft_id"]) if st.get("draft_id") is not None else None,
                frame_name=fname,
            )
        except Exception as exc:  # noqa: BLE001
            # EXC-0374: T1 -- [SILENT-DROP] MASTERSTAR zone/is_usable/bp_rp merge failure leaves per-frame proc rows ... (EXCEPT-BULK 2026-07-08)
            return {
                "file": fname,
                "status": f"error: {exc}",
                "csv": "",
                "deferred_writes": deferred_writes,
                "infolog_messages": [f"Per-frame catalog {fname}: {exc}"],
                "debug_pixel_match": debug_pixel_match,
            }

    df = _apply_exo_host_columns_to_proc_df(df, hdr, (h_i, wpx_i), st, frame_name=fname)

    _before_dao = len(df)
    df = _proc_drop_unmatched_dao_rows(df)
    LOGGER.debug("[TODO-13] catalog-only pre-filter (detect): %d -> %d rows", _before_dao, len(df))

    # --- Join MASTERSTAR annotations into per-frame catalog (via catalog_id) ---
    # For matched rows (catalog_id non-empty), bring stable MASTERSTAR columns like zone/is_usable/bp_rp/etc.
    try:
        if master_tab is not None and "catalog_id" in df.columns and "catalog_id" in master_tab.columns:
            _JOIN_COLS = [
                "zone",
                "is_saturated",
                "is_usable",
                "bp_rp",
                "phot_g_mean_mag",
                "catalog_mag",
                "vy_identity_gate",
                "gaia_dao_resid_px",
                "edge_safe_10px",
                "snr50_ok",
                "noise_floor_adu",
                "saturate_limit_adu_85pct",
                "source_type",
            ]
            join_cols = [c for c in _JOIN_COLS if c in master_tab.columns]
            if join_cols:
                master_lookup = master_tab[["catalog_id"] + join_cols].copy()
                master_lookup["catalog_id"] = (
                    master_lookup["catalog_id"].astype(str).str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
                )
                master_lookup = master_lookup.dropna(subset=["catalog_id"])
                master_lookup = master_lookup.drop_duplicates(subset=["catalog_id"], keep="first")

                df = df.copy()
                df["catalog_id"] = df["catalog_id"].astype(str).str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

                # Avoid column collisions; MASTERSTAR values should win for these columns.
                df = df.drop(columns=[c for c in join_cols if c in df.columns], errors="ignore")
                df = df.merge(master_lookup, on="catalog_id", how="left")
    except Exception:  # noqa: BLE001
        pass

    # FORCED-PHOT-01: inject force-eligible MASTERSTAR members missing from DAO match.
    # DAO path above is unchanged (n_raw_dao / match counts preserved). Forced rows are
    # measured at locked MASTERSTAR (x,y) with bounded peak refine.
    try:
        if master_tab is not None and bool(st.get("forced_photometry_enabled", True)):
            from forced_photometry import inject_forced_masterstar_rows  # noqa: PLC0415

            _force_ids = st.get("forced_photometry_catalog_ids")
            _force_set = (
                {str(x).strip() for x in _force_ids if str(x).strip()}
                if _force_ids
                else None
            )
            _fwhm_fp = float(st.get("dao_fwhm_px", 2.5) or 2.5)
            _bound = float(st.get("forced_photometry_centroid_bound_fwhm", 2.5) or 2.5)
            _margin = float(st.get("forced_photometry_margin_px", 0.0) or 0.0)
            df, _fp_meta = inject_forced_masterstar_rows(
                df,
                master_tab,
                image=np.asarray(data) if data is not None else None,
                fwhm_px=_fwhm_fp,
                centroid_bound_fwhm=_bound,
                margin_px=_margin,
                force_ids=_force_set,
            )
            try:
                meta["forced_photometry"] = dict(_fp_meta)
                debug_pixel_match["forced_photometry"] = dict(_fp_meta)
            except Exception:  # noqa: BLE001
                pass
            LOGGER.debug(
                "[FORCED-PHOT] %s injected=%s geometry_miss=%s",
                fname,
                _fp_meta.get("n_injected"),
                _fp_meta.get("n_geometry_miss"),
            )
    except Exception as _fp_exc:  # noqa: BLE001
        LOGGER.error("[FORCED-PHOT] inject failed (DAO path kept): %s", _fp_exc)

    try:
        debug_pixel_match["match_mode"] = meta.get("catalog_match_mode")
        debug_pixel_match["plate_scale_arcsec_per_px"] = meta.get("plate_scale_arcsec_per_px")
        debug_pixel_match["n_matched"] = meta.get("n_matched")
    except Exception:  # noqa: BLE001
        pass

    _run_aperture = bool(st.get("_run_aperture", True))
    _run_epsf = bool(st.get("_run_epsf", False))
    if _run_aperture:
        st["current_frame_name"] = fname
        df = _apply_aperture_catalog_enhancements_from_st(df, data, hdr, st)
    _ps_dir = str(st.get("platesolve_dir") or "").strip()
    _psf_on = bool(st.get("psf_photometry_enabled", False))
    _epsf_ids = (
        _epsf_fit_catalog_ids(Path(_ps_dir), psf_photometry_enabled=_psf_on)
        if _ps_dir
        else None
    )
    LOGGER.info(
        "[ePSF] _export_per_frame_run_catalog_core %s: platesolve_dir=%r n_psf_ids=%s lc_set=%s",
        fname,
        _ps_dir or None,
        len(_epsf_ids) if _epsf_ids is not None else "ALL",
        _psf_on,
    )
    if not _run_aperture and _run_epsf:
        # PSF-only mode: psf_flux promoted to primary.
        pass
    df = _fill_psf_catalog_columns(df, data, hdr, st, target_ids=_epsf_ids)

    # --- Time columns (JD / HJD / BJD) ---
    _db_tc = None
    try:
        from time_utils import compute_time_columns

        _dbp = str(st.get("database_path") or "").strip()
        _did_tc = st.get("draft_id")
        if _dbp and _did_tc is not None:
            try:
                _db_tc = VyvarDatabase(Path(_dbp))
            except Exception:  # noqa: BLE001
                _db_tc = None
        _geo_cfg = _cfg_from_export_worker_state(st)
        _time_cols = compute_time_columns(
            hdr,
            db=_db_tc,
            draft_id=int(_did_tc) if _did_tc is not None else None,
            cfg=_geo_cfg,
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

        # Airmass - rovnaka hodnota pre vsetky hviezdy v snimke (frame-level)
        _am_val = _extract_airmass_from_header(
            hdr,
            cfg=_geo_cfg,
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
    _cmm_export = str(meta.get("catalog_match_mode") or "").strip()
    if _cmm_export:
        df2["catalog_match_mode"] = _cmm_export

    _before_cat = len(df2)
    df2 = _proc_catalog_keep_matched_rows_only(df2)
    LOGGER.debug("[TODO-13] catalog-only filter: %d -> %d rows", _before_cat, len(df2))

    # FORCED-PHOT / DRAFT-514-TRIAGE A2: multiple DAO detections can match the same
    # Gaia id; dedupe before write. Alternate export path already called this;
    # production ``_export_per_frame_run_catalog_core`` did not (root cause of
    # 127 duplicate rows in proc_BO_CVn_Light_001 on draft 514).
    _before_dedupe = len(df2)
    df2 = _proc_deduplicate_matched_catalog_rows(df2)
    if len(df2) != _before_dedupe:
        LOGGER.debug(
            "[PROC] dedupe catalog_id: %d -> %d rows",
            _before_dedupe,
            len(df2),
        )

    csv_paths: list[str] = []
    write_sidecar = bool(st.get("write_sidecar_csv_next_to_fits"))
    mirror_flat = bool(st.get("mirror_flat_platesolve_folder"))
    defer = bool(st.get("defer_disk_writes"))
    out_flat = Path(str(st.get("out_flat") or "."))

    if write_sidecar:
        sidecar = proc_csv_path_for_aligned_fits(base_path)
        if defer:
            deferred_writes.append((str(sidecar), df2.copy()))
        else:
            _vyvar_df_to_csv(df2, sidecar)
        csv_paths.append(str(sidecar))

    if mirror_flat:
        stem = Path(fname).stem
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in stem)[:120]
        flat_path = out_flat / f"{safe}_catalog.csv"
        if defer:
            deferred_writes.append((str(flat_path), df2.copy()))
        else:
            _vyvar_df_to_csv(df2, flat_path)
        csv_paths.append(str(flat_path))

    primary_csv = csv_paths[0] if csv_paths else ""
    _psf_rec = st.get("_psf_frame_record")
    return {
        "file": fname,
        "status": "ok",
        "csv": primary_csv,
        "csv_paths": ";".join(csv_paths),
        "n_detected": meta.get("n_detected"),
        "n_matched": meta.get("n_matched"),
        "catalog_match_mode": ("master_reference_locked" if master_only_mode else meta.get("catalog_match_mode", "full_cone")),
        "deferred_writes": deferred_writes,
        "infolog_messages": [],
        "debug_pixel_match": debug_pixel_match,
        "psf_frame_record": dict(_psf_rec) if isinstance(_psf_rec, dict) else None,
    }


def _export_per_frame_disk_worker_task(fp_str: str) -> dict[str, Any]:
    print("DEBUG: REAL WORKER CALLED")
    import numpy as np

    st = _EXPORT_PER_FRAME_WORKER_STATE
    fp = Path(fp_str)
    try:
        with fits.open(fp, memmap=False) as hdul:
            hdr = hdul[0].header.copy()
            data = np.array(hdul[0].data, dtype=np.float32, copy=True)
    except Exception as exc:  # noqa: BLE001
        return {
            "file": fp.name,
            "status": f"read_error: {exc}",
            "csv": "",
            "deferred_writes": [],
            "infolog_messages": [f"Per-frame catalog {fp.name}: read_error: {exc}"],
        }
    return _export_per_frame_run_catalog_core(fp, hdr, data, st)


def _export_per_frame_ram_worker_task(
    packed: tuple[str, bytes, bytes, int, int],
) -> dict[str, Any]:
    """Picklable worker: (fits_path_str, pickled_fits_Header, float32_bytes, ny, nx)."""
    print("DEBUG: REAL WORKER CALLED")
    import numpy as np

    st = _EXPORT_PER_FRAME_WORKER_STATE
    base_str, hdr_pkl, raw, ny, nx = packed
    fname = Path(base_str).name
    empty_def: list[tuple[str, pd.DataFrame]] = []
    try:
        hdr = pickle.loads(hdr_pkl)
    except Exception as exc:  # noqa: BLE001
        return {
            "file": fname,
            "status": f"header_error: {exc}",
            "csv": "",
            "deferred_writes": empty_def,
            "infolog_messages": [f"Per-frame catalog {fname}: header_error: {exc}"],
        }
    try:
        data = np.frombuffer(raw, dtype=np.float32, count=ny * nx).reshape((ny, nx)).copy()
    except Exception as exc:  # noqa: BLE001
        return {
            "file": fname,
            "status": f"buffer_error: {exc}",
            "csv": "",
            "deferred_writes": empty_def,
            "infolog_messages": [f"Per-frame catalog {fname}: buffer_error: {exc}"],
        }
    return _export_per_frame_run_catalog_core(Path(base_str), hdr, data, st)


def _finalize_hybrid_bkg_fallback_sidecar(
    proc_dir: Path,
    *,
    err_background_mode: str,
    write_sidecar: bool,
    gain: float,
    read_noise: float,
    setup_label: str,
) -> dict[str, Any]:
    """Post-export hybrid Howell fallback scaling when sidecar proc CSVs exist."""
    _ = err_background_mode  # CONSOLIDATE-01D: empirical is the only policy
    if not write_sidecar:
        return {}
    try:
        from photometry_core import finalize_hybrid_bkg_fallback_proc_dir

        return finalize_hybrid_bkg_fallback_proc_dir(
            proc_dir,
            gain=float(gain),
            read_noise=float(read_noise),
            setup_label=str(setup_label),
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[PHOT] hybrid bkg fallback finalize skipped: %s", exc)
        return {}


def _apply_wcs_tan_fragment_to_header(h: fits.Header, wh: fits.Header, history_note: str) -> None:
    strip_celestial_wcs_keys(h)
    for k in wh:
        if k in ("", "COMMENT", "HISTORY", "SIMPLE", "BITPIX", "NAXIS", "EXTEND"):
            continue
        if k.startswith("NAXIS") and k != "NAXIS":
            continue
        try:
            h[k] = wh[k]
        except Exception:  # noqa: BLE001
            pass
    h.add_history(history_note)


def _estimate_catalog_frame_hw(
    work_ram: Sequence[tuple[str, Any, Any]] | None,
    files: list[Path],
) -> tuple[int, int]:
    """Rough (ny, nx) for RAM cap heuristics (per-frame catalog export)."""
    import numpy as np

    if work_ram:
        d = np.asarray(work_ram[0][2])
        if d.ndim == 2:
            return int(d.shape[0]), int(d.shape[1])
    for fp in files[:1]:
        try:
            with fits.open(fp, memmap=False) as h:
                nh = h[0].header
                ny = int(nh.get("NAXIS2", 0) or 0)
                nx = int(nh.get("NAXIS1", 0) or 0)
                if ny > 0 and nx > 0:
                    return ny, nx
        except Exception:  # noqa: BLE001
            continue
    return 2048, 2048


def _vyvar_cap_mp_workers_for_catalog(
    n_workers: int,
    frame_hw: tuple[int, int],
    *,
    reserve_gb: float,
) -> int:
    """Cap process pool size using available RAM (rough float32 frame footprint per worker)."""
    h, w = frame_hw
    if h <= 0 or w <= 0:
        h, w = 2048, 2048
    per_worker = max(int(h * w * 4 * 3), 1)
    try:
        import psutil
    except ImportError:
        return max(1, n_workers)
    reserve = int(max(0.0, float(reserve_gb)) * (1024**3))
    avail = int(psutil.virtual_memory().available) - reserve
    if avail <= 0:
        return 1
    mx = max(1, avail // per_worker)
    return max(1, min(int(n_workers), mx))


