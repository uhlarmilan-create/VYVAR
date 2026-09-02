"""Photometry core - zluceny modul (photometry + photometry_phase2a)."""
from __future__ import annotations

import copy
import json
import logging
import math
import os
import random
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import AbstractSet, Any, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from astropy.io import fits as astrofits

from stats_core import _flux_to_mag
from comp_pool_rms import attach_comp_rms_to_pool_rows, compute_global_pool_rms_map
from comp_rms_loo import (
    COMP_RMS_FRAMES_BASIS,
    COMP_RMS_LOO_PHOTON_K_DEFAULT,
    LN10_OVER_2P5,
    compute_loo_mag_rms_map,
)
from proc_frame_store import (
    PROC_CSV_GLOB,
    PROC_STORE_COLS,
    ProcFrameStore,
    is_masterstar_proc_name,
    proc_csv_path_for_aligned_fits,
)
from config import (
    AppConfig,
    DENSITY_OVERRIDES,
    apply_crowding_overrides,
    apply_density_overrides,
    classify_field_density,
    compute_field_density,
    resolve_comp_sparse_fallback_enabled,
)
from database import query_local_gaia, query_local_gaia_by_source_ids
from gaia_catalog_id import (
    GAIA_PROC_CSV_READ_DTYPE,
    masterstar_row_gaia_key,
    normalize_gaia_source_id,
    read_vyvar_csv,
)
from infolog import log_event
from plain_stats import plain_mean_med_std

from catalog_match_trust import is_wcs_untrusted_catalog_match_mode, normalize_catalog_match_mode
from jd_axis_format import jd_axis_title, jd_series_relative
from utils import iter_fits_paths_recursive as _iter_fits_recursive
from unit_resolver import (
    phase01_chip_interior_margin_px as _resolve_chip_interior_margin_px,
    phase01_comparison_isolation_radius_px as _resolve_isolation_radius_px,
    resolve_max_dist_fallback_deg,
    resolve_px_from_arcsec,
    resolve_px_from_fwhm_factor,
    plate_scale_arcsec_per_px_from_header,
    sips_dao_fwhm_px as _resolve_sips_dao_fwhm_px,
)

LOGGER = logging.getLogger(__name__)

_MAD_CONSISTENCY = 0.6745  # normalizacny faktor MAD -> sigma ekvivalent

# Explicit annulus sky (ADU/px) for Howell err; ``noise_floor_adu`` remains detection-floor legacy.
SKY_ADU_PER_PX_ANNULUS_COL = "sky_adu_per_px_annulus"
SKY_SURFACE_BG_MEDIAN_ADU_COL = "sky_surface_bg_median_adu"

# F-BINGAIN-1: empirical background noise (empty-aperture scatter) + provenance.
SIGMA_BKG_AP_COL = "sigma_bkg_ap"
ERR_BKG_SOURCE_COL = "err_bkg_source"
ERR_BKG_MODE_EMPIRICAL = "empirical"
ERR_BKG_MODE_HOWELL = "howell"
ERR_BKG_SOURCE_EMPIRICAL = "empirical"
ERR_BKG_SOURCE_HOWELL_FALLBACK = "howell_fallback"
ERR_BKG_SOURCE_HOWELL_SCALED = "howell_scaled"
BKG_SCALE_R_CLAMP_LO = 0.05
BKG_SCALE_R_CLAMP_HI = 2.0





# Per-target LC time provenance (F-BJD-1): labels BJD recompute path, does not alter time values.
TIME_BASE_COL = "time_base"
TIME_BASE_BJD_TDB = "BJD_TDB"
TIME_BASE_JD_FALLBACK = "JD_FALLBACK"










# Comp tier: Gaia BP-RP outside this band -> unreliable vs field comps (use B-V fallback).
_BPRP_VALID_MIN = 0.1
_BPRP_VALID_MAX = 3.5

# Gaia ID (`catalog_id`, VSX / masterstars `name`) musi byt str - float64 straca cifry
_GAIA_ID_DTYPE: dict[str, type] = dict(GAIA_PROC_CSV_READ_DTYPE)




_COMP_QUALITY_JSON_META_KEYS = frozenset(
    {
        "selected_tier",
        "tier4_warning",
        "n_tier1",
        "n_tier2",
        "n_tier3",
        "n_tier4",
        "aperture_correction",
        "qa_degraded",
        "qa_degraded_reason",
    }
)









# ---------------------------------------------------------------------------
# Pomocne funkcie
# ---------------------------------------------------------------------------














# ---------------------------------------------------------------------------
# KROK 1: Globalna fixna apertura z PSF FWHM (MASTERSTAR VY_FWHM alebo fit)
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# KROK 2: Aperturna fotometria per snimka - medianovy sky
# ---------------------------------------------------------------------------




def _clamp_err_empty_apertures_min(n: int) -> int:
    try:
        v = int(n)
    except (TypeError, ValueError):
        v = 16
    return max(1, min(256, v))




def _robust_scatter_mad(values: np.ndarray) -> float:
    """Sigma-clipped MAD scatter (Labbe et al. 2003 empty-aperture convention)."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 3:
        return float("nan")
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    if mad <= 0:
        return 0.0
    return mad / _MAD_CONSISTENCY


def _build_star_exclusion_mask(
    shape: tuple[int, ...],
    star_x: np.ndarray,
    star_y: np.ndarray,
    exclusion_radius_px: float,
    edge_margin_px: float,
) -> np.ndarray:
    """Boolean mask: True where empty apertures must not be placed."""
    ny, nx = int(shape[0]), int(shape[1])
    blocked = np.zeros((ny, nx), dtype=bool)
    em = int(math.ceil(max(0.0, float(edge_margin_px))))
    if em > 0:
        blocked[:em, :] = True
        blocked[-em:, :] = True
        blocked[:, :em] = True
        blocked[:, -em:] = True
    ex_r = float(exclusion_radius_px)
    if ex_r <= 0:
        return blocked
    ex_r2 = ex_r * ex_r
    xs = np.asarray(star_x, dtype=np.float64)
    ys = np.asarray(star_y, dtype=np.float64)
    ok = np.isfinite(xs) & np.isfinite(ys)
    # Canonical order: mask is OR-commutative, but keep draw/debug paths order-stable.
    order = np.lexsort((xs[ok], ys[ok]))
    xs_s = xs[ok][order]
    ys_s = ys[ok][order]
    for xi, yi in zip(xs_s, ys_s):
        x0 = max(0, int(math.floor(float(xi) - ex_r)) - 1)
        x1 = min(nx, int(math.ceil(float(xi) + ex_r)) + 2)
        y0 = max(0, int(math.floor(float(yi) - ex_r)) - 1)
        y1 = min(ny, int(math.ceil(float(yi) + ex_r)) + 2)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        dist2 = (xx - float(xi)) ** 2 + (yy - float(yi)) ** 2
        blocked[y0:y1, x0:x1] |= dist2 <= ex_r2
    return blocked


def _canonicalize_star_xy(
    star_x: np.ndarray,
    star_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Sort finite (x,y) pairs and return (xs, ys, sha256 hex of canonical list)."""
    import hashlib

    xs = np.asarray(star_x, dtype=np.float64).ravel()
    ys = np.asarray(star_y, dtype=np.float64).ravel()
    n = min(xs.size, ys.size)
    xs, ys = xs[:n], ys[:n]
    ok = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[ok], ys[ok]
    if xs.size == 0:
        return xs, ys, hashlib.sha256(b"").hexdigest()
    order = np.lexsort((xs, ys))
    xs, ys = xs[order], ys[order]
    # Deduplicate exact duplicates after sort (stable membership).
    if xs.size > 1:
        keep = np.empty(xs.size, dtype=bool)
        keep[0] = True
        keep[1:] = (xs[1:] != xs[:-1]) | (ys[1:] != ys[:-1])
        xs, ys = xs[keep], ys[keep]
    blob = np.column_stack([xs, ys]).astype("<f8", copy=False).tobytes()
    return xs, ys, hashlib.sha256(blob).hexdigest()


def _labbe_debug_dump_enabled() -> bool:
    import os

    return str(os.environ.get("VYVAR_LABBE_DEBUG_DUMP", "")).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _labbe_debug_dump_path() -> Path:
    import os

    raw = str(os.environ.get("VYVAR_LABBE_DEBUG_DUMP_PATH", "")).strip()
    if raw:
        return Path(raw)
    return Path("tmp") / "labbe_debug_dump.jsonl"


def _labbe_append_debug_record(record: dict[str, Any]) -> None:
    if not _labbe_debug_dump_enabled():
        return
    path = _labbe_debug_dump_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    except OSError:
        pass












































LAST_EXCLUDED_TARGETS: pd.DataFrame = pd.DataFrame(
    columns=["name", "vsx_name", "vsx_type", "ra_deg", "dec_deg", "mag", "reason"]
)


















def _median_bkg_var_adu2_per_px_from_proc_cache(
    csv_cache: dict[str, pd.DataFrame],
) -> float | None:
    """Median per-pixel background variance [ADU^2/px] from empirical ``sigma_bkg_ap`` in proc CSVs."""
    vals: list[float] = []
    for _df in csv_cache.values():
        if _df is None or _df.empty:
            continue
        if SIGMA_BKG_AP_COL not in _df.columns or "aperture_r_px" not in _df.columns:
            continue
        sig = pd.to_numeric(_df[SIGMA_BKG_AP_COL], errors="coerce")
        rap = pd.to_numeric(_df["aperture_r_px"], errors="coerce")
        ok = sig.notna() & rap.notna() & (rap > 0) & (sig >= 0)
        if not ok.any():
            continue
        area = math.pi * np.asarray(rap[ok], dtype=np.float64) ** 2
        var_ap = np.asarray(sig[ok], dtype=np.float64) ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            var_px = var_ap / np.maximum(area, 1e-12)
        vals.extend([float(v) for v in var_px if math.isfinite(float(v)) and float(v) >= 0])
    if not vals:
        return None
    med = float(np.nanmedian(np.asarray(vals, dtype=np.float64)))
    return med if math.isfinite(med) and med >= 0 else None
















# ---------------------------------------------------------------------------
# ALG-3: Temporal binning of comparison ensemble (MNRAS 2023)
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# ALG-5: PyTICS iterative comp intercalibration (RASTI 2026)
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# KROK 3: Stability check porovnavaciek (Abbeho p2p scatter + MAD)
# ---------------------------------------------------------------------------

# Observed-band / catalog mag before broad Gaia G for SNR-optimal aperture sizing.
_APERTURE_SIZING_MAG_COLS: tuple[str, ...] = (
    "mag",
    "catalog_mag",
    "lc_median_mag",
    "phot_g_mean_mag",
)


def _star_mag_for_aperture_sizing(row: Any) -> float | None:
    """Brightness for SNR aperture table: prefer observed-band ``mag`` over Gaia G."""
    for mag_col in _APERTURE_SIZING_MAG_COLS:
        try:
            if mag_col not in row.index if hasattr(row, "index") else mag_col not in row:
                continue
        except Exception:  # noqa: BLE001
            # EXC-0133: T4 -- Bad mag value on one masterstar row skipped - loop tries next row for aperture sizing (EXCEPT-BULK-2 2026-07-08)
            if isinstance(row, dict) and mag_col not in row:
                continue
        try:
            mv = float(pd.to_numeric(row.get(mag_col) if hasattr(row, "get") else row[mag_col], errors="coerce"))
        except Exception:  # noqa: BLE001
            continue
        if math.isfinite(mv):
            return mv
    return None












# ---------------------------------------------------------------------------
# KROK 4: Ensemble normalizacia
# ---------------------------------------------------------------------------














# ---------------------------------------------------------------------------
# Color term (BP-RP) - globalny shift na noc
# ---------------------------------------------------------------------------














def _is_broadband_photometric_filter(obs_group: str) -> bool:
    """True for Johnson/Cousins/Sloan broadband filters (B/V/Rc/...); false for L/Clear/unknown."""
    from band_classify import classify_photometric_band, color_term_auto_from_band

    band = classify_photometric_band(obs_group)
    return bool(color_term_auto_from_band(band))


























_CT_PROTOTYPE_CSV_FIELDS: tuple[str, ...] = (
    "catalog_id",
    "vsx_name",
    "obs_group",
    "n_comp_used",
    "c1",
    "c1_stderr",
    "stderr_ratio",
    "target_bp_rp",
    "comp_med_bp_rp",
    "ct_corr",
    "cat_inst_scatter",
    "cat_inst_scatter_resid",
    "gate_would_pass",
)







# ---------------------------------------------------------------------------
# KROK 5: Outlier detekcia
# ---------------------------------------------------------------------------
















# ---------------------------------------------------------------------------
# KROK 6: Vystup - lightcurve CSV
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# KROK 6: Vystup - PNG grafy
# ---------------------------------------------------------------------------










# ---------------------------------------------------------------------------
# Hlavny wrapper - run_phase2a
# ---------------------------------------------------------------------------


_EDGE_FILTER_NOTE_OK = ""
_EDGE_FILTER_NOTE_FAILED = "EDGE-UNFILTERED: edge safety check failed"












# PFS-SEMANTICS-01: never rescue these skip_reason values (TARGET-DEPTH-02 outranks PFS).
PFS_NEVER_RESCUE_REASONS = frozenset({"zone_noise", "below_target_depth"})
PFS_SATURATION_SKIP_REASONS = frozenset(
    {
        "zone_flag",
        "saturovany ciel",
        "per_frame_saturation",
        "likely_saturated",
        "saturated",
    }
)




















_LC_QUALITY_FLAGS: tuple[str, ...] = (
    "good",
    "noisy",
    "noisy_moon",
    "short_baseline",
    "no_data",
    "saturated",
)














_GIT_PROVENANCE_WARNED = False
# src_py/photometry_core.py -> repo root is parent.parent (git cwd + porcelain path base).
_REPO_ROOT_FOR_PROVENANCE = Path(__file__).resolve().parent.parent
































_ADAPTIVE_BLEND_CACHE: dict[str, dict[str, BlendMapEntry]] = {}






from mag_constants import MAG_ERR_SCALE

_PSF_ERR_MAG_SCALE = MAG_ERR_SCALE












def _get_lc_star_method(cid: str, all_frames: pd.DataFrame, star_method: str) -> np.ndarray:
    """Inst mag for one star using a fixed method for all frames (NaN if PSF missing)."""
    sub = all_frames[all_frames["catalog_id"] == cid]
    if sub.empty:
        return np.array([], dtype=float)
    if str(star_method).strip().lower() != "psf":
        return _get_lc(cid, all_frames)
    return _get_lc_psf_strict(cid, all_frames)




























def _phase2a_process_one_target(
    target_row: Any,
    *,
    ti: int,
    state: _Phase2AState,
    summary_rows: list,
    n_lc: int,
    lc_dir: Path,
    output_dir: Path,
    progress_cb: Any,
    masterstar_fits_path: Path,
    annulus_inner_fwhm: float,
    annulus_outer_fwhm: float,
    outlier_sigma: float,
    stability_sigma: float,
    _apt_fw: float,
    _save_png: bool,
    ac_sign_logged: list[bool],
) -> tuple[list, int]:
    """Process one target through the full Phase 2A photometry pipeline.

    Returns updated (summary_rows, n_lc).
    """
    def _p2(msg: str) -> None:
        if progress_cb is None:
            return
        try:
            progress_cb(str(msg))
        except UnicodeEncodeError:
            progress_cb(str(msg).encode("ascii", "backslashreplace").decode("ascii"))

    _comp_index = state._comp_index
    target_bp_rp_by_cid = state.target_bp_rp_by_cid
    csv_files = state.csv_files
    _phase2a_csv_cache = state._phase2a_csv_cache
    _phase2a_lookup_cache = state._phase2a_lookup_cache
    frame_time_lookup = state.frame_time_lookup
    fwhm_px = state.fwhm_px
    apertures_px = state.apertures_px
    star_xy = state.star_xy
    chip_fw = state.chip_fw
    chip_fh = state.chip_fh
    _ms_data = state._ms_data
    _flux_matrix = state._flux_matrix
    obs_group = state.obs_group
    _gain_phot = state._gain_phot
    _rn_phot = state._rn_phot
    sat_limit_resolved = state.sat_limit_resolved
    _aligned_dir_2a = state._aligned_dir_2a
    _cfg = state._cfg
    _nt = state._nt
    comp_df = state.comp_df
    _lunar = state.lunar_context

    target_cid = _normalize_gaia_id(target_row.get("catalog_id", ""))
    target_name = _target_display_name(target_row, fallback_cid=target_cid)
    target_vsx_type = str(target_row.get("vsx_type", "") or "").strip()
    _sp = target_row.get("skip_photometry", False)
    if isinstance(_sp, (bool, np.bool_)):
        skip_photo = bool(_sp)
    else:
        skip_photo = str(_sp).strip().lower() in ("1", "true", "yes", "t")
    _zf_row = str(target_row.get("zone_flag", "")).strip()
    _zf_low = _zf_row.lower()
    # When per-frame sat is ON, skip_photometry already encodes the decision;
    # do not re-force whole-star skip from master zone_flag.
    _pfs_on = bool(getattr(_cfg, "per_frame_saturation_enabled", False))
    # TARGET-DEPTH-02 outranks PFS: noise never enters photometry.
    # Saturation-zone skip is re-forced only when PFS is OFF (PFS already
    # encoded the saturation decision). Do not exempt the whole {saturated, noise} set.
    if _zf_low == "noise":
        skip_photo = True
    elif (not _pfs_on) and _zf_low == "saturated":
        skip_photo = True
    if progress_cb is not None and (
        ti == 1 or ti == _nt or (_nt > 1 and ti % max(1, _nt // 12) == 0)
    ):
        _p2(f"Faza 2A: ciel {ti}/{_nt}: {target_name[:50]}")
    if skip_photo:
        _sr_col = str(target_row.get("skip_reason", "") or "").strip()
        if _sr_col:
            _skip_reason = _sr_col
        elif _zf_low == "noise":
            _skip_reason = "zone_noise"
        else:
            _skip_reason = "saturovany ciel"
        logging.info(f"[FAZA 2A] Preskakujem fotometriu ({_skip_reason}): {target_name}")
        _skip_sum: dict[str, Any] = {
            "catalog_id": target_cid,
            "vsx_name": target_name,
            "zone_flag": _zf_row,
            "n_frames": 0,
            "n_good_comp": 0,
            "n_saturated": 0,
            "lc_rms": float("nan"),
            "lc_median_mag": float("nan"),
            "aperture_px": float("nan"),
            "am_slope": float("nan"),
            "am_detrended": False,
            "lc_csv": "",
            "lc_png": "",
        }
        if _pfs_on:
            _skip_sum["skip_reason"] = _skip_reason
            _scf = float(pd.to_numeric(target_row.get("sat_clean_frac"), errors="coerce"))
            _skip_sum["sat_clean_frac"] = _scf
            _skip_sum["per_frame_sat_fallback"] = bool(
                target_row.get("per_frame_sat_fallback", False)
            )
        summary_rows.append(_skip_sum)
        return summary_rows, n_lc
    logging.info(
        f"[FAZA 2A] Spustam: target={target_name}, "
        f"frames={len(csv_files)}, "
        f"apertura={_apt_fw * float(fwhm_px):.2f}px "
        f"(FWHM={float(fwhm_px):.3f}px x {_apt_fw:.2f})"
    )

    # Comp hviezdy pre tento target
    target_comps = _comp_index.get(target_cid, pd.DataFrame()).copy()
    _star_xy = dict(star_xy)

    if target_comps.empty:
        summary_rows = _phase2a_skip_empty_comps_target(
            target_cid=target_cid,
            target_name=target_name,
            zone_flag=_zf_row,
            summary_rows=summary_rows,
        )
        return summary_rows, n_lc

    comp_ids: list[str] = []
    _seen_comp: set[str] = set()
    for c in target_comps["catalog_id"].tolist():
        nc = _normalize_gaia_id(c)
        if nc and nc not in _seen_comp:
            _seen_comp.add(nc)
            comp_ids.append(nc)
    all_ids = [target_cid] + comp_ids

    # Katalogove magnitudy comp hviezd
    comp_catalog_mag = {
        _normalize_gaia_id(r["catalog_id"]): float(r.get("mag", float("nan")))
        for _, r in target_comps.iterrows()
    }
    _cfg_tw = _cfg.comp_tier_weights()
    tier_weights = {
        1: float(_cfg_tw[0]),
        2: float(_cfg_tw[1]),
        3: float(_cfg_tw[2]),
        4: float(_cfg_tw[3]),
    }
    for _k in list(tier_weights.keys()):
        try:
            _v = float(tier_weights[_k])
        except Exception:  # noqa: BLE001
            _v = float("nan")
        if not math.isfinite(_v) or _v <= 0:
            tier_weights[_k] = 0.01
        else:
            tier_weights[_k] = max(0.01, float(_v))

    comp_tier_map: dict[str, int] = {}
    for _, r in target_comps.iterrows():
        cid0 = _normalize_gaia_id(r["catalog_id"])
        try:
            t0 = int(pd.to_numeric(r.get("comp_tier", 4), errors="coerce") or 4)
        except Exception:  # noqa: BLE001
            t0 = 4
        comp_tier_map[cid0] = int(max(1, min(4, t0)))

    comp_rms_map: dict[str, float] = {}
    for _, r in target_comps.iterrows():
        cid0 = _normalize_gaia_id(r["catalog_id"])
        try:
            rms_raw = float(r.get("comp_rms", float("nan")))
        except Exception:  # noqa: BLE001
            rms_raw = float("nan")
        # COMP-ADMIT-03: do not bake tier into rms; colour/distance enter sigma_eff.
        comp_rms_map[cid0] = float(rms_raw)

    # Continuous weights: sigma_eff^2 = rms^2 + (c_col*|dBP-RP|)^2 + (c_dist*r)^2
    from comp_weights import resolve_comp_weight_coeffs, sigma_eff_mag, weight_from_sigma_eff  # noqa: PLC0415

    _tx = float(pd.to_numeric(target_row.get("x"), errors="coerce"))
    _ty = float(pd.to_numeric(target_row.get("y"), errors="coerce"))
    _tra = float(pd.to_numeric(target_row.get("ra_deg", target_row.get("ra")), errors="coerce"))
    _tde = float(pd.to_numeric(target_row.get("dec_deg", target_row.get("dec")), errors="coerce"))
    _tbpr = float(pd.to_numeric(target_row.get("bp_rp"), errors="coerce"))
    _plate = float(getattr(_cfg, "plate_scale_arcsec_per_px", 0.0) or 0.0)
    _c_col_ov = getattr(_cfg, "comp_weight_c_col_mag_per_bprp", None)
    _c_dist_ov = getattr(_cfg, "comp_weight_c_dist_mag_per_deg", None)
    try:
        _c_col_ov_f = float(_c_col_ov) if _c_col_ov is not None else None
    except (TypeError, ValueError):
        _c_col_ov_f = None
    try:
        _c_dist_ov_f = float(_c_dist_ov) if _c_dist_ov is not None else None
    except (TypeError, ValueError):
        _c_dist_ov_f = None
    _k2 = None
    try:
        from k2_extinction import resolve_k2_bprp_value  # noqa: PLC0415

        _k2, _ = resolve_k2_bprp_value(_cfg, str(getattr(_cfg, "active_obs_group", "") or ""))
    except Exception:  # noqa: BLE001
        _k2 = None
    _am_span = float(getattr(_cfg, "comp_weight_airmass_span", float("nan")) or float("nan"))
    if not math.isfinite(_am_span):
        _am_span = 0.0
    _r_list: list[float] = []
    _sc_list: list[float] = []
    for _, r in target_comps.iterrows():
        try:
            _rr = float(pd.to_numeric(r.get("ra_deg", r.get("ra")), errors="coerce"))
            _dd = float(pd.to_numeric(r.get("dec_deg", r.get("dec")), errors="coerce"))
            _rms = float(pd.to_numeric(r.get("comp_rms"), errors="coerce"))
        except Exception:  # noqa: BLE001
            continue
        if math.isfinite(_rr) and math.isfinite(_dd) and math.isfinite(_tra) and math.isfinite(_tde):
            dra = math.radians(_rr - _tra) * math.cos(math.radians(0.5 * (_dd + _tde)))
            dde = math.radians(_dd - _tde)
            _r_list.append(float(math.degrees(math.hypot(dra, dde))))
            if math.isfinite(_rms):
                _sc_list.append(_rms)
        elif math.isfinite(_tx) and math.isfinite(_ty) and _plate > 0:
            try:
                _cx = float(pd.to_numeric(r.get("x"), errors="coerce"))
                _cy = float(pd.to_numeric(r.get("y"), errors="coerce"))
            except Exception:  # noqa: BLE001
                continue
            if math.isfinite(_cx) and math.isfinite(_cy):
                _r_list.append(float(math.hypot(_cx - _tx, _cy - _ty) * _plate / 3600.0))
                if math.isfinite(_rms):
                    _sc_list.append(_rms)
    _optics = str(getattr(_cfg, "comp_weight_optics_kind", "") or "").strip()
    if not _optics:
        try:
            from comp_weights import infer_optics_kind_from_header_or_name  # noqa: PLC0415

            _optics = infer_optics_kind_from_header_or_name(
                telescop=str(getattr(_cfg, "telescope_name", "") or ""),
                telescope_name=str(getattr(_cfg, "telescope_name", "") or ""),
            )
        except Exception:  # noqa: BLE001
            _optics = "unknown"
    if not math.isfinite(_am_span) or _am_span <= 0:
        # Best-effort airmass span from frame table if present on comps flux cache later; keep 0.
        _am_span = 0.0
    _coeffs = resolve_comp_weight_coeffs(
        k2_bprp=_k2,
        airmass_span=_am_span,
        optics_kind=_optics,
        r_deg=_r_list,
        residual_scatter_mag=_sc_list,
        c_col_override=_c_col_ov_f,
        c_dist_override=_c_dist_ov_f,
    )
    comp_weight_map: dict[str, float] = {}
    for _, r in target_comps.iterrows():
        cid0 = _normalize_gaia_id(r["catalog_id"])
        rms0 = float(comp_rms_map.get(cid0, float("nan")))
        try:
            bpr0 = float(pd.to_numeric(r.get("bp_rp"), errors="coerce"))
        except Exception:  # noqa: BLE001
            bpr0 = float("nan")
        db = abs(bpr0 - _tbpr) if math.isfinite(bpr0) and math.isfinite(_tbpr) else 0.0
        rdeg = 0.0
        try:
            _rr = float(pd.to_numeric(r.get("ra_deg", r.get("ra")), errors="coerce"))
            _dd = float(pd.to_numeric(r.get("dec_deg", r.get("dec")), errors="coerce"))
            if math.isfinite(_rr) and math.isfinite(_dd) and math.isfinite(_tra) and math.isfinite(_tde):
                dra = math.radians(_rr - _tra) * math.cos(math.radians(0.5 * (_dd + _tde)))
                dde = math.radians(_dd - _tde)
                rdeg = float(math.degrees(math.hypot(dra, dde)))
            elif math.isfinite(_tx) and math.isfinite(_ty) and _plate > 0:
                _cx = float(pd.to_numeric(r.get("x"), errors="coerce"))
                _cy = float(pd.to_numeric(r.get("y"), errors="coerce"))
                if math.isfinite(_cx) and math.isfinite(_cy):
                    rdeg = float(math.hypot(_cx - _tx, _cy - _ty) * _plate / 3600.0)
        except Exception:  # noqa: BLE001
            rdeg = 0.0
        se = sigma_eff_mag(
            sigma_rms_mag=rms0,
            delta_bprp=db,
            r_deg=rdeg,
            c_col_mag_per_bprp=_coeffs.c_col_mag_per_bprp,
            c_dist_mag_per_deg=_coeffs.c_dist_mag_per_deg,
        )
        comp_weight_map[cid0] = weight_from_sigma_eff(se)

    _chk_cid_pref: str | None = None
    try:
        from pinned_ensembles import (  # noqa: PLC0415
            get_pinned_check_for_target,
            is_pinned_target,
            validate_pinned_check_member,
        )

        if is_pinned_target(str(target_cid)):
            _pin_chk = get_pinned_check_for_target(str(target_cid))
            if _pin_chk is not None:
                _chk_ms = state.masterstars_df.loc[
                    state.masterstars_df["catalog_id"].astype(str).str.strip().eq(_pin_chk.check_catalog_id)
                ]
                if not _chk_ms.empty:
                    _chk_row_ms = _chk_ms.iloc[0]
                    _chk_dist = float("nan")
                    if "_dist_deg" in target_comps.columns:
                        _sub = target_comps.loc[
                            target_comps["catalog_id"].astype(str).str.strip().eq(_pin_chk.check_catalog_id)
                        ]
                        if not _sub.empty and "_dist_deg" in _sub.columns:
                            _chk_dist = float(pd.to_numeric(_sub["_dist_deg"].iloc[0], errors="coerce")) * 3600.0
                    if not math.isfinite(_chk_dist):
                        try:
                            _cra = float(
                                pd.to_numeric(
                                    _chk_row_ms.get("ra_deg", _chk_row_ms.get("ra")),
                                    errors="coerce",
                                )
                            )
                            _cde = float(
                                pd.to_numeric(
                                    _chk_row_ms.get("dec_deg", _chk_row_ms.get("dec")),
                                    errors="coerce",
                                )
                            )
                            if (
                                math.isfinite(_cra)
                                and math.isfinite(_cde)
                                and math.isfinite(_tra)
                                and math.isfinite(_tde)
                            ):
                                _chk_dist = _angular_distance_deg(_tra, _tde, _cra, _cde) * 3600.0
                            elif math.isfinite(_tx) and math.isfinite(_ty) and _plate > 0:
                                _cx = float(pd.to_numeric(_chk_row_ms.get("x"), errors="coerce"))
                                _cy = float(pd.to_numeric(_chk_row_ms.get("y"), errors="coerce"))
                                if math.isfinite(_cx) and math.isfinite(_cy):
                                    _chk_dist = float(
                                        math.hypot(_cx - _tx, _cy - _ty) * _plate / 3600.0
                                    )
                        except Exception:  # noqa: BLE001
                            _chk_dist = float("nan")
                    _chk_rms = float(
                        pd.to_numeric(
                            comp_rms_map.get(_pin_chk.check_catalog_id, float("nan")),
                            errors="coerce",
                        )
                    )
                    _ok_chk, _reason_chk = validate_pinned_check_member(
                        _chk_row_ms,
                        target_cid=str(target_cid),
                        dist_arcsec=_chk_dist,
                        comp_rms=_chk_rms,
                        min_dist_arcsec=float(_cfg.phase01_comparison_min_dist_arcsec),
                        max_comp_rms=float(_cfg.phase01_comparison_max_comp_rms),
                    )
                    if _ok_chk:
                        _chk_cid_pref = _pin_chk.check_catalog_id
                        log_event(
                            f"[PIN] check star {_pin_chk.check_catalog_id} "
                            f"kname={_pin_chk.check_kname!r} target={target_cid}"
                        )
                    else:
                        logging.warning(
                            "[PIN-DROP] check star %s for target %s: %s",
                            _pin_chk.check_catalog_id,
                            target_cid,
                            _reason_chk,
                        )
    except Exception as _pin_chk_exc:  # noqa: BLE001
        logging.debug("[PIN] check star pin skipped for %s: %s", target_cid, _pin_chk_exc)

    if _chk_cid_pref is None:
        try:
            from check_star_kmag import (  # noqa: PLC0415
                field_check_star_candidate_pool,
                select_check_star,
            )

            _chk_pool_pref = field_check_star_candidate_pool(
                state.comp_df,
                target_comps=target_comps,
            )
            if not _chk_pool_pref.empty:
                _chk_row_pref = select_check_star(
                    _chk_pool_pref,
                    ensemble_ids=set(comp_ids),
                    n_comp_min=max(1, min(3, len(_chk_pool_pref))),
                    cfg=_cfg,
                )
                if _chk_row_pref is not None:
                    _chk_cid_pref = _normalize_gaia_id(_chk_row_pref.get("catalog_id", ""))
        except (ImportError, KeyError, TypeError, ValueError, AttributeError) as _ck_pref_exc:
            logging.debug("[CHECK-KMAG] preselect skipped for %s: %s", target_cid, _ck_pref_exc)

    if _chk_cid_pref:
        if (
            _chk_cid_pref not in comp_ids
            and _chk_cid_pref != target_cid
        ):
            all_ids.append(_chk_cid_pref)
            _chk_row_pref = None
            try:
                from check_star_kmag import field_check_star_candidate_pool  # noqa: PLC0415

                _chk_pool_pref = field_check_star_candidate_pool(
                    state.comp_df,
                    target_comps=target_comps,
                )
                if not _chk_pool_pref.empty:
                    _m = _chk_pool_pref["catalog_id"].astype(str).str.strip().eq(_chk_cid_pref)
                    if bool(_m.any()):
                        _chk_row_pref = _chk_pool_pref.loc[_m].iloc[0]
            except Exception:  # noqa: BLE001
                _chk_row_pref = None
            if _chk_row_pref is None:
                _chk_ms = state.masterstars_df.loc[
                    state.masterstars_df["catalog_id"].astype(str).str.strip().eq(_chk_cid_pref)
                ]
                _chk_row_pref = _chk_ms.iloc[0] if not _chk_ms.empty else None
            if _chk_row_pref is not None:
                for _mk in ("mag", "phot_g_mean_mag"):
                    try:
                        _cm = float(pd.to_numeric(_chk_row_pref.get(_mk), errors="coerce"))
                    except Exception:  # noqa: BLE001
                        _cm = float("nan")
                    if math.isfinite(_cm):
                        comp_catalog_mag[_chk_cid_pref] = _cm
                        break
                try:
                    _cx = float(pd.to_numeric(_chk_row_pref.get("x"), errors="coerce"))
                    _cy = float(pd.to_numeric(_chk_row_pref.get("y"), errors="coerce"))
                except Exception:  # noqa: BLE001
                    _cx, _cy = float("nan"), float("nan")
                if math.isfinite(_cx) and math.isfinite(_cy):
                    _star_xy[_chk_cid_pref] = (_cx, _cy)

    # Krok 2: Fotometria per snimka (PERF-8: slice shared flux matrix when built)
    frame_results: list[pd.DataFrame] = []
    if not _flux_matrix.empty:
        _id_set = set(all_ids)
        _target_slice = _flux_matrix[_flux_matrix["catalog_id"].isin(_id_set)]
        for csv_path in csv_files:
            _sf = csv_path.name
            _df_sub = _target_slice[_target_slice["source_file"] == _sf]
            if _df_sub.empty:
                continue
            df_frame = _df_sub.copy()
            _ft = frame_time_lookup.get(csv_path.stem)
            _cached_df = _phase2a_csv_cache.get(str(csv_path))
            if (chip_fw is None or chip_fh is None) and ("x" in df_frame.columns and "y" in df_frame.columns):
                try:
                    _xm = float(pd.to_numeric(df_frame["x"], errors="coerce").max())
                    _ym = float(pd.to_numeric(df_frame["y"], errors="coerce").max())
                except Exception:  # noqa: BLE001
                    _xm, _ym = float("nan"), float("nan")
                if chip_fw is None and math.isfinite(_xm) and _xm > 0:
                    chip_fw = int(math.ceil(_xm)) + 2
                if chip_fh is None and math.isfinite(_ym) and _ym > 0:
                    chip_fh = int(math.ceil(_ym)) + 2
            if chip_fw is not None and chip_fh is not None and int(chip_fw) > 0 and int(chip_fh) > 0:
                tmask = df_frame["catalog_id"].astype(str).str.strip().eq(target_cid)
                if bool(tmask.any()):
                    tr = df_frame.loc[tmask].iloc[0]
                    try:
                        x_t = float(pd.to_numeric(tr.get("x"), errors="coerce"))
                        y_t = float(pd.to_numeric(tr.get("y"), errors="coerce"))
                    except Exception:  # noqa: BLE001
                        x_t, y_t = float("nan"), float("nan")
                    try:
                        r_out_t = float(pd.to_numeric(tr.get("sky_annulus_r_out_px", 30.0), errors="coerce"))
                    except Exception:  # noqa: BLE001
                        r_out_t = 30.0
                    if not (math.isfinite(r_out_t) and r_out_t > 0):
                        r_out_t = 30.0
                    if math.isfinite(x_t) and math.isfinite(y_t):
                        edge_ok = (
                            (x_t - r_out_t >= 0)
                            and (x_t + r_out_t <= float(chip_fw))
                            and (y_t - r_out_t >= 0)
                            and (y_t + r_out_t <= float(chip_fh))
                        )
                        if not edge_ok:
                            df_frame = df_frame.copy()
                            df_frame.loc[tmask, "mag_inst"] = float("nan")
                            df_frame.loc[tmask, "flag"] = "edge_fail"
                            if "edge_fail" in df_frame.columns:
                                df_frame.loc[tmask, "edge_fail"] = True
                            logging.info(
                                "[TARGET EDGE] %s: frame %s vyradeny - annulus mimo cip (x=%.0f, y=%.0f, r_out=%.1fpx)",
                                str(target_name),
                                str(csv_path.name),
                                float(x_t),
                                float(y_t),
                                float(r_out_t),
                            )
            frame_results.append(df_frame)
    else:
        for csv_path in csv_files:
            _ft = frame_time_lookup.get(csv_path.stem)
            _key_csv = str(csv_path)
            _cached_df = _phase2a_csv_cache.get(_key_csv)
            _lookup_row = _phase2a_lookup_cache.get(_key_csv)

            df_frame = read_flux_from_csv(
                csv_path,
                all_ids,
                apertures_px,
                sat_limit_adu=sat_limit_resolved,
                star_xy=_star_xy,
                xy_tol_px=18.0,
                frame_times=_ft,
                csv_df=_cached_df,
                lookup=_lookup_row,
                gain=float(_gain_phot),
                read_noise=float(_rn_phot),
                use_apcorr_flux=bool(state.use_apcorr_flux),
                variable_target_catalog_ids=state.variable_target_catalog_ids,
                err_background_mode=ERR_BKG_MODE_EMPIRICAL,
            )
            if not df_frame.empty:
                if (chip_fw is None or chip_fh is None) and ("x" in df_frame.columns and "y" in df_frame.columns):
                    try:
                        _xm = float(pd.to_numeric(df_frame["x"], errors="coerce").max())
                        _ym = float(pd.to_numeric(df_frame["y"], errors="coerce").max())
                    except Exception:  # noqa: BLE001
                        _xm, _ym = float("nan"), float("nan")
                    if chip_fw is None and math.isfinite(_xm) and _xm > 0:
                        chip_fw = int(math.ceil(_xm)) + 2
                    if chip_fh is None and math.isfinite(_ym) and _ym > 0:
                        chip_fh = int(math.ceil(_ym)) + 2

                if chip_fw is not None and chip_fh is not None and int(chip_fw) > 0 and int(chip_fh) > 0:
                    tmask = df_frame["catalog_id"].astype(str).str.strip().eq(target_cid)
                    if bool(tmask.any()):
                        tr = df_frame.loc[tmask].iloc[0]
                        try:
                            x_t = float(pd.to_numeric(tr.get("x"), errors="coerce"))
                            y_t = float(pd.to_numeric(tr.get("y"), errors="coerce"))
                        except Exception:  # noqa: BLE001
                            x_t, y_t = float("nan"), float("nan")
                        try:
                            r_out_t = float(pd.to_numeric(tr.get("sky_annulus_r_out_px", 30.0), errors="coerce"))
                        except Exception:  # noqa: BLE001
                            r_out_t = 30.0
                        if not (math.isfinite(r_out_t) and r_out_t > 0):
                            r_out_t = 30.0
                        if math.isfinite(x_t) and math.isfinite(y_t):
                            edge_ok = (
                                (x_t - r_out_t >= 0)
                                and (x_t + r_out_t <= float(chip_fw))
                                and (y_t - r_out_t >= 0)
                                and (y_t + r_out_t <= float(chip_fh))
                            )
                            if not edge_ok:
                                df_frame = df_frame.copy()
                                df_frame.loc[tmask, "mag_inst"] = float("nan")
                                df_frame.loc[tmask, "flag"] = "edge_fail"
                                if "edge_fail" in df_frame.columns:
                                    df_frame.loc[tmask, "edge_fail"] = True
                                logging.info(
                                    "[TARGET EDGE] %s: frame %s vyradeny - annulus mimo cip (x=%.0f, y=%.0f, r_out=%.1fpx)",
                                    str(target_name),
                                    str(csv_path.name),
                                    float(x_t),
                                    float(y_t),
                                    float(r_out_t),
                                )
                frame_results.append(df_frame)

    if not frame_results:
        return summary_rows, n_lc

    ac_result: dict[str, Any] = {
        "ok": False,
        "delta_m_corr": None,
        "scatter_mag": None,
        "n_ref_stars": 0,
        "ref_star_ids": [],
        "reason": "disabled",
    }
    if bool(_cfg.aperture_correction_enabled):
        try:
            ac_result = compute_aperture_correction(
                comp_df=target_comps,
                frame_results=frame_results,
                min_ref_stars=int(_cfg.aperture_correction_min_ref_stars),
                max_contamination=float(_cfg.aperture_correction_max_contamination),
                max_scatter_mag=float(_cfg.aperture_correction_max_scatter_mag),
            )
            if bool(ac_result.get("ok")):
                log_event(
                    f"[AC] DeltaM_corr={float(ac_result['delta_m_corr']):.4f} "
                    f"scatter={float(ac_result['scatter_mag']):.4f} "
                    f"n_ref={int(ac_result['n_ref_stars'])}"
                )
            else:
                log_event(f"[AC] skipped: {ac_result.get('reason', '')}")
        except Exception as _ac_exc:  # noqa: BLE001
            log_event(f"[AC] skipped: exception {_ac_exc!s}")
            ac_result = {
                "ok": False,
                "delta_m_corr": None,
                "scatter_mag": None,
                "n_ref_stars": 0,
                "ref_star_ids": [],
                "reason": "exception",
            }
    _ = ac_result  # Krokom 3: aplikacia na mag_calib / CSV

    all_frames = pd.concat(frame_results, ignore_index=True)

    # Zostav casove rady per hviezda
    target_lc = _get_lc(target_cid, all_frames)
    comp_lc = {cid: _get_lc(cid, all_frames) for cid in comp_ids}

    # Flux sources for method-keyed LC outputs (aperture always primary/default).
    _psf_enabled = bool(_cfg.psf_photometry_enabled)
    _adaptive = bool(getattr(_cfg, "psf_adaptive_enabled", False))
    _have_psf_cols = "psf_flux" in all_frames.columns and "psf_fit_ok" in all_frames.columns
    if _have_psf_cols and (_adaptive or _psf_enabled):
        _blend_map = _load_adaptive_blend_map(masterstar_fits_path)
        all_frames["lc_flux_method"] = compute_lc_flux_method(
            all_frames,
            _blend_map,
            resolve_fwhm=float(getattr(_cfg, "psf_adaptive_resolve_fwhm", 2.0)),
            snr_lo=float(getattr(_cfg, "psf_adaptive_snr_lo", 15.0)),
        )
    # Primary published LC is always aperture (target_lc / comp_lc from _get_lc above).
    _lc_export_method = "aperture"

    # ALG-3: Temporal binning of comp ensemble (MNRAS 2023)
    comp_lc = temporal_bin_comp_lc(
        comp_lc=comp_lc,
        comp_quality={},
        all_frames=all_frames,
        window=int(_cfg.temporal_bin_window),
        enabled=bool(_cfg.temporal_binning_enabled),
    )

    # COMP-ASSIGN-01 D4/D5: membership is fixed from Phase 1 (3-8). Stability is a
    # post-photometry verdict only - do not let it re-select before ensemble.
    comp_quality = {cid: {"quality": "good"} for cid in comp_ids}

    # ALG-5: PyTICS iterative comp star intercalibration (RASTI 2026)
    comp_rms_map = pytics_iterative_weights(
        comp_lc=comp_lc,
        comp_quality=comp_quality,
        comp_rms_map=comp_rms_map,
        n_iter=int(_cfg.pytics_n_iter),
        enabled=bool(_cfg.pytics_enabled),
    )

    # Krok 4: Ensemble normalizacia (consumes the delivered set as given)
    mag_calib, delta_mag, ensemble_scatter = ensemble_normalize(
        target_lc,
        comp_lc,
        comp_catalog_mag,
        comp_quality,
        comp_rms_map=comp_rms_map,
        comp_tier_map=comp_tier_map,
        tier_weights=tier_weights,
        comp_weight_map=comp_weight_map,
        n_comp_min=max(1, int(getattr(_cfg, "phase01_comparison_n_comp_min", 3))),
        n_comp_max=int(_cfg.phase01_comparison_n_comp_max),
    )
    _ensemble_scatter_by_file = _ensemble_scatter_by_source_file(
        all_frames, target_cid, ensemble_scatter
    )

    _dilution_result: dict[str, Any] = {
        "dilution_factor": 1.0,
        "dilution_delta_mag": 0.0,
        "n_neighbors": 0,
        "neighbor_flux_sum": 0.0,
        "aperture_arcsec": float("nan"),
        "search_radius_arcsec": float("nan"),
    }
    if bool(_cfg.gs11_dilution_enabled) and state.gaia_db_path:
        from dilution import apply_target_dilution_to_mag_calib, compute_dilution_factor  # noqa: PLC0415

        try:
            _target_ra = float(
                pd.to_numeric(
                    target_row.get("ra_deg", target_row.get("ra", float("nan"))),
                    errors="coerce",
                )
            )
            _target_dec = float(
                pd.to_numeric(
                    target_row.get("dec_deg", target_row.get("dec", float("nan"))),
                    errors="coerce",
                )
            )
        except (TypeError, ValueError):
            _target_ra = _target_dec = float("nan")
        _target_g_mag = float("nan")
        for _gk in ("mag", "phot_g_mean_mag", "catalog_mag"):
            try:
                _gv = float(pd.to_numeric(target_row.get(_gk, float("nan")), errors="coerce"))
            except (TypeError, ValueError):
                _gv = float("nan")
            if math.isfinite(_gv):
                _target_g_mag = _gv
                break
        _ap_cfg = float(_cfg.gs11_dilution_aperture_arcsec)
        _dilution_skipped_ap = False
        if math.isfinite(_ap_cfg) and _ap_cfg > 0:
            _ap_arcsec = _ap_cfg
        else:
            _ap_px, _ap_src = _resolve_photometric_aperture_px_for_gs11(
                target_cid,
                apertures_px,
                _target_g_mag,
                state.snr_ap_table,
                aperture_fwhm_factor=float(_apt_fw),
                fwhm_px=float(fwhm_px),
            )
            if _ap_px is None:
                logging.warning(
                    "[GS11] target %s: photometric aperture unavailable - dilution skipped",
                    target_cid or "?",
                )
                log_event(
                    f"[GS11] target {target_cid or '?'}: photometric aperture unavailable - dilution skipped"
                )
                _dilution_skipped_ap = True
                _ap_arcsec = float("nan")
            else:
                _ap_arcsec = float(_ap_px) * float(state.plate_scale_arcsec)
        _cid_int = None
        try:
            from dilution import _normalize_exclude_source_id  # noqa: PLC0415

            _cid_int = _normalize_exclude_source_id(target_cid)
        except Exception:  # noqa: BLE001
            _cid_int = None
        if _dilution_skipped_ap:
            _dilution_result = {
                "dilution_factor": 1.0,
                "dilution_delta_mag": 0.0,
                "n_neighbors": 0,
                "neighbor_flux_sum": 0.0,
                "aperture_arcsec": float("nan"),
                "search_radius_arcsec": float("nan"),
                "dilution_skipped": True,
                "dilution_skip_reason": "photometric_aperture_unavailable",
            }
        else:
            _dilution_result = compute_dilution_factor(
                _target_ra,
                _target_dec,
                _target_g_mag,
                _ap_arcsec,
                str(state.gaia_db_path),
                catalog_id=_cid_int,
                mag_limit_delta=float(_cfg.gs11_dilution_mag_limit_delta),
            )
        _mag_pre_gs11 = float("nan")
        _finite_pre = mag_calib[np.isfinite(mag_calib)]
        if len(_finite_pre) > 0:
            _mag_pre_gs11 = float(np.median(_finite_pre))
        mag_calib, _dilution_result = apply_target_dilution_to_mag_calib(
            mag_calib,
            _dilution_result,
            _cfg,
            target_cid=str(target_cid),
        )
        _mag_post_gs11 = float("nan")
        _finite_post = mag_calib[np.isfinite(mag_calib)]
        if len(_finite_post) > 0:
            _mag_post_gs11 = float(np.median(_finite_post))
    else:
        _mag_pre_gs11 = float("nan")
        _mag_post_gs11 = float("nan")

    # -- Aperture correction (AC) --
    ac_ok = bool(ac_result.get("ok", False)) if isinstance(ac_result, dict) else False
    delta_m_corr = ac_result.get("delta_m_corr") if isinstance(ac_result, dict) else None
    if ac_ok and delta_m_corr is not None and np.isfinite(float(delta_m_corr)):
        mag_calib_ac = mag_calib + float(delta_m_corr)
    else:
        mag_calib_ac = np.full_like(mag_calib, float("nan"))

    # Sanity log znamienka: pri delta_m_corr < 0 ma byt mag_calib_ac < mag_calib.
    if (not ac_sign_logged[0]) and ac_ok and delta_m_corr is not None and np.isfinite(float(delta_m_corr)):
        if len(mag_calib) > 0 and math.isfinite(float(mag_calib[0])) and math.isfinite(float(mag_calib_ac[0])):
            log_event(
                f"[AC SIGN] mag_calib0={float(mag_calib[0]):.4f} "
                f"delta_m_corr={float(delta_m_corr):.4f} "
                f"mag_calib_ac0={float(mag_calib_ac[0]):.4f}"
            )
            ac_sign_logged[0] = True

    # -- Color term (BP-RP) - global comp-pool fit; toggle controls correction only --
    target_bp_rp = float(target_bp_rp_by_cid.get(target_cid, float("nan")))
    comp_bp_rp: dict[str, float] = {}
    if "bp_rp" in target_comps.columns:
        for _, rr in target_comps.iterrows():
            cidc = _normalize_gaia_id(rr.get("catalog_id", ""))
            if not cidc:
                continue
            v = pd.to_numeric(rr.get("bp_rp"), errors="coerce")
            try:
                fv = float(v)
            except Exception:  # noqa: BLE001
                fv = float("nan")
            if math.isfinite(fv):
                comp_bp_rp[cidc] = float(fv)

    from k2_extinction import K2Source, apply_k2_per_frame, bp_rp_comp_median  # noqa: PLC0415

    k2_value_lc = float("nan")
    k2_colour_ref = float("nan")
    k2_source_rows = [K2Source.NONE.value] * len(mag_calib)
    _k2_val = float(getattr(state, "k2_bprp", float("nan")))
    _k2_src = str(getattr(state, "k2_source", K2Source.NONE.value))
    if _k2_src in (
        K2Source.LITERATURE_DEFAULT.value,
        K2Source.NIGHT_FIT.value,
    ) and math.isfinite(_k2_val):
        _tf_k2 = all_frames[all_frames["catalog_id"] == target_cid]
        if "airmass" in _tf_k2.columns:
            _airmass_k2 = _tf_k2["airmass"].to_numpy(dtype=float)
        else:
            _airmass_k2 = np.full(len(mag_calib), float("nan"), dtype=float)
        _bp_med_k2 = bp_rp_comp_median(comp_bp_rp, comp_quality)
        _k2_src_enum = (
            K2Source.NIGHT_FIT
            if _k2_src == K2Source.NIGHT_FIT.value
            else K2Source.LITERATURE_DEFAULT
        )
        mag_calib, _k2_delta, k2_source_rows = apply_k2_per_frame(
            mag_calib,
            _airmass_k2,
            object_bp_rp=float(target_bp_rp),
            bp_rp_comp_med=_bp_med_k2,
            k2_value=_k2_val,
            k2_source=_k2_src_enum,
        )
        k2_value_lc = _k2_val
        k2_colour_ref = _bp_med_k2

    c1 = 0.0
    c1_stderr = float("nan")
    ct_mode = ""
    ct_n_comp = 0
    mag_calib_ct = mag_calib.copy()
    ct_corr = 0.0
    bp_rp_comp_med = float("nan")
    ct_ok = False
    _group_ct = state.group_color_term
    if state.apply_color_term and _group_ct is not None and _group_ct.apply_gate:
        c1 = float(_group_ct.c1)
        c1_stderr = float(_group_ct.c1_stderr)
        ct_mode = str(getattr(_group_ct, "mode", "fit") or "fit")
        ct_n_comp = int(_group_ct.n_comp)
        _ref_bp, _ref_q = ct_ensemble_reference_maps(comp_bp_rp, comp_quality)
        _ct_in_range = _check_color_term_extrapolation(
            target_bp_rp=float(target_bp_rp),
            comp_bp_rp_values=[float(v) for v in _ref_bp.values()],
            target_name=str(target_name),
            extrapolation_tol=float(_cfg.phase01_ct_extrapolation_tol),
        )
        try:
            from pinned_ensembles import baseline_lc_ct_ok_for_target, is_pinned_target  # noqa: PLC0415

            if is_pinned_target(str(target_cid)):
                _pin_ct_ok = baseline_lc_ct_ok_for_target(str(target_cid))
                if _pin_ct_ok is False:
                    _ct_in_range = False
                elif _pin_ct_ok is True and str(ct_mode) == "clear_level":
                    _ct_in_range = True
        except Exception as _pin_ct_rng_exc:  # noqa: BLE001
            LOGGER.debug("[PIN] CT extrapolation pin gate skip: %s", _pin_ct_rng_exc)
        if _ct_in_range:
            mag_calib_ct, ct_corr, bp_rp_comp_med = apply_color_term(
                mag_calib,
                target_bp_rp,
                _ref_bp,
                _ref_q,
                c1,
                comp_weights=comp_weight_map if ct_mode == "clear_level" else None,
            )
            ct_ok = (
                bool(math.isfinite(float(target_bp_rp)))
                and float(c1) != 0.0
                and math.isfinite(float(bp_rp_comp_med))
            )
        else:
            logging.info(
                "[COLOR TERM] extrapolation -> CT skipped (target kept, uncorrected)"
            )
            mag_calib_ct = mag_calib.copy()
            ct_corr = 0.0
            bp_rp_comp_med = float("nan")
            ct_ok = False

    if _ct_prototype_enabled():
        _proto_c1 = 0.0
        _proto_c1_stderr = float("nan")
        _proto_n_comp = 0
        if comp_bp_rp:
            _proto_c1, _proto_c1_stderr, _proto_n_comp = fit_color_term_c1(
                comp_lc,
                comp_catalog_mag,
                comp_bp_rp,
                comp_quality,
                min_comp=5,
                sigma_clip_sigma=3.0,
            )
        _proto_corr = 0.0
        _proto_comp_med = float("nan")
        if comp_bp_rp and float(_proto_c1) != 0.0:
            _, _proto_corr, _proto_comp_med = apply_color_term(
                mag_calib,
                float(target_bp_rp),
                comp_bp_rp,
                comp_quality,
                float(_proto_c1),
            )
        _proto_scatter, _proto_scatter_resid = (
            _color_term_cat_inst_scatter_pair(
                comp_lc,
                comp_catalog_mag,
                comp_bp_rp,
                comp_quality,
                float(_proto_c1),
                min_comp=5,
                sigma_clip_sigma=3.0,
            )
            if comp_bp_rp
            else (float("nan"), float("nan"))
        )
        _proto_stderr_ratio = float("nan")
        if float(_proto_c1) != 0.0 and math.isfinite(float(_proto_c1_stderr)):
            _proto_stderr_ratio = abs(float(_proto_c1_stderr) / float(_proto_c1))
        _proto_gate = (
            int(_proto_n_comp) >= int(_cfg.phase01_ct_min_comp)
            and float(_proto_c1) != 0.0
            and math.isfinite(_proto_stderr_ratio)
            and float(_proto_stderr_ratio) <= 0.5
        )
        _append_ct_prototype_row(
            _draft_dir_from_phase2a_paths(output_dir, Path(masterstar_fits_path)),
            {
                "catalog_id": target_cid,
                "vsx_name": target_name,
                "obs_group": str(obs_group),
                "n_comp_used": int(_proto_n_comp),
                "c1": float(_proto_c1),
                "c1_stderr": float(_proto_c1_stderr),
                "stderr_ratio": _proto_stderr_ratio,
                "target_bp_rp": float(target_bp_rp),
                "comp_med_bp_rp": float(_proto_comp_med),
                "ct_corr": float(_proto_corr),
                "cat_inst_scatter": _proto_scatter,
                "cat_inst_scatter_resid": _proto_scatter_resid,
                "gate_would_pass": bool(_proto_gate),
            },
        )

    # Casove hodnoty targetu - sort by source_file so ensemble_scatter index aligns
    # with ``_get_lc`` / ``_ensemble_scatter_by_source_file`` (LABBE-DET / SEM determinism).
    target_frames = all_frames[all_frames["catalog_id"] == target_cid]
    if not target_frames.empty and "source_file" in target_frames.columns:
        target_frames = target_frames.sort_values(["source_file"], kind="mergesort")
    _measured_ap_target = _measured_aperture_from_proc_cache(target_cid, state._phase2a_csv_cache)
    if math.isfinite(_measured_ap_target) and _measured_ap_target > 0 and not target_frames.empty:
        target_frames = target_frames.copy()
        target_frames["aperture_r_px"] = float(_measured_ap_target)
    bjd = target_frames["bjd"].to_numpy(dtype=float)
    hjd = target_frames["hjd"].to_numpy(dtype=float)
    jd = target_frames["jd"].to_numpy(dtype=float)

    # BJD-PERTARGET: recompute with target's own RA/Dec (not field-center LTT)
    _target_ra = float(pd.to_numeric(target_row.get("ra_deg", target_row.get("ra", float("nan"))), errors="coerce"))
    _target_dec = float(
        pd.to_numeric(target_row.get("dec_deg", target_row.get("dec", float("nan"))), errors="coerce")
    )
    bjd, hjd, time_base = _recompute_bjd_hjd_with_status(
        jd,
        _target_ra,
        _target_dec,
        _cfg,
        site=(state.site_lat, state.site_lon, state.site_alt) if state.site_ok else None,
    )

    err = target_frames["err"].to_numpy(dtype=float)
    err, err_method_rows = _route_lc_per_frame_err(target_frames, err)
    err_photon_arr = np.asarray(err, dtype=np.float64).copy()
    if "airmass" in target_frames.columns:
        airmass_arr = target_frames["airmass"].to_numpy(dtype=float)
    else:
        airmass_arr = np.full(len(target_frames), float("nan"), dtype=float)
    # Per-point uncertainty = photon/SNR base error (term-1) (+) ensemble zeropoint uncertainty
    # (term-3, ``ensemble_scatter``). Joined by EXACT ``source_file`` (G2-F004), not positional index.
    _src_for_err = target_frames["source_file"].astype(str).tolist()
    from sigma_budget import resolve_rig_scintillation_params  # noqa: PLC0415
    from sigma_floor_core import resolve_sigma_sys_mag, scintillation_mag_per_epoch  # noqa: PLC0415

    _sigma_sys_mag = resolve_sigma_sys_mag(
        state.equipment_id,
        _cfg,
        rig_label=str(state.obs_group or ""),
    )
    _draft_id_lc: int | None = None
    try:
        from platesolve_ui_paths import parse_draft_id_from_text  # noqa: PLC0415

        _draft_id_lc = parse_draft_id_from_text(str(output_dir))
    except Exception:  # noqa: BLE001
        _draft_id_lc = None
    _rig_scint = resolve_rig_scintillation_params(
        draft_id=_draft_id_lc,
        setup=str(state.obs_group or ""),
        cfg=_cfg,
        pipeline_meta=(
            {"observer_location": {"alt_m": float(state.site_alt)}}
            if state.site_ok and state.site_alt is not None
            else None
        ),
    )
    _scint_mag_arr = np.array(
        [
            scintillation_mag_per_epoch(
                telescope_diameter_m=_rig_scint.telescope_diameter_m,
                airmass=float(am),
                exposure_s=_rig_scint.exposure_s,
                altitude_m=_rig_scint.altitude_m,
                c_y=_rig_scint.c_y,
            )
            if math.isfinite(float(am)) and float(am) >= 1.0
            else 0.0
            for am in airmass_arr
        ],
        dtype=np.float64,
    )
    err, err_scatter_unmatched_arr = _combine_err_with_ensemble_scatter_keyed(
        err,
        _src_for_err,
        _ensemble_scatter_by_file,
        sigma_sys_mag=_sigma_sys_mag,
        sigma_scint_mag=_scint_mag_arr,
        target_name=str(target_name),
    )
    # WIDE-ERR-03: Pont/Gillon calibration layer on combined model err.
    # CONSOLIDATE-01D: always ERR-CALIB calibrated (export_err_mode=model branch deleted).
    try:
        from err_calibration import (  # noqa: PLC0415
            ERR_CALIB_SIDECAR,
            apply_calibration_rel,
            bins_from_sidecar,
            load_sidecar,
            smooth_from_sidecar,
        )

        _cal_path = Path(output_dir) / ERR_CALIB_SIDECAR if output_dir is not None else None
        _cal = load_sidecar(_cal_path) if _cal_path is not None else None
        if _cal:
            _smooth = smooth_from_sidecar(_cal)
            _bins = bins_from_sidecar(_cal) if not _smooth else []
            _calib_obj = _smooth if _smooth is not None else _bins
            _g_tgt = float("nan")
            try:
                _g_tgt = float(
                    pd.to_numeric(
                        target_row.get("phot_g_mean_mag", target_row.get("mag", float("nan"))),
                        errors="coerce",
                    )
                )
            except Exception:  # noqa: BLE001
                _g_tgt = float("nan")
            if math.isfinite(_g_tgt) and _calib_obj:
                err = np.asarray(
                    [
                        apply_calibration_rel(float(e), _g_tgt, _calib_obj)
                        for e in np.asarray(err, dtype=np.float64)
                    ],
                    dtype=np.float64,
                )
                logging.info(
                    "[ERR-CALIB] applied export_err_mode=calibrated for G=%.3f (%s)",
                    _g_tgt,
                    "smooth" if _smooth is not None else f"{len(_bins)} bins",
                )
    except Exception as _cal_exc:  # noqa: BLE001
        logging.warning("[ERR-CALIB] skip apply: %s", _cal_exc)
    # Propagate colour-level coefficient uncertainty into exported err (constant per LC).
    if bool(ct_ok) and math.isfinite(float(c1_stderr)) and math.isfinite(float(ct_corr)):
        # corr = c1 * (target - ref) => sigma_corr = |target-ref| * sigma_c1
        _dcol = float(target_bp_rp) - float(bp_rp_comp_med) if math.isfinite(float(bp_rp_comp_med)) else float("nan")
        if math.isfinite(_dcol):
            _err_ct = abs(_dcol) * float(c1_stderr)
            if math.isfinite(_err_ct) and _err_ct > 0:
                err = np.sqrt(np.square(np.asarray(err, dtype=np.float64)) + _err_ct**2)
                logging.info(
                    "[COLOR TERM] err += %.4f mag from k uncertainty (delta_colour=%+.3f)",
                    float(_err_ct),
                    float(_dcol),
                )
    err_photon_export, err_sem_rel_export, err_scint_rel_export, err_sigma_sys_rel_export = (
        _err_budget_components_keyed(
            err_photon_arr,
            _src_for_err,
            _ensemble_scatter_by_file,
            sigma_sys_mag=_sigma_sys_mag,
            sigma_scint_mag=_scint_mag_arr,
        )
    )
    ap_arr = target_frames["aperture_r_px"].to_numpy(dtype=float)
    src_files = target_frames["source_file"].tolist()
    sat_flags = (target_frames["flag"] == "saturated").to_numpy(dtype=bool)

    # Airmass / flip arrays for export + the democratic detrender (no per-target airmass detrend here:
    # airmass is handled by the differential comp ensemble).
    flip_arr = (
        target_frames["is_flipped"].fillna(False).astype(bool).to_numpy()
        if "is_flipped" in target_frames.columns
        else np.zeros_like(bjd, dtype=bool)
    )
    align_fail_arr = (
        target_frames["alignment_failed"].fillna(False).astype(bool).to_numpy()
        if "alignment_failed" in target_frames.columns
        else np.zeros_like(bjd, dtype=bool)
    )
    n_alignment_failed = int(np.count_nonzero(align_fail_arr))
    alignment_failed_frac = float(n_alignment_failed) / max(int(len(bjd)), 1)
    if "catalog_match_mode" in target_frames.columns:
        catalog_match_mode_list = [
            normalize_catalog_match_mode(v) for v in target_frames["catalog_match_mode"].tolist()
        ]
    else:
        catalog_match_mode_list = [""] * len(bjd)
    if "wcs_untrusted" in target_frames.columns:
        wcs_untrusted_arr = target_frames["wcs_untrusted"].fillna(False).astype(bool).to_numpy()
    else:
        wcs_untrusted_arr = np.array(
            [is_wcs_untrusted_catalog_match_mode(m) for m in catalog_match_mode_list],
            dtype=bool,
        )
    n_wcs_untrusted = int(np.count_nonzero(wcs_untrusted_arr))
    wcs_untrusted_frac = float(n_wcs_untrusted) / max(int(len(bjd)), 1)

    if "flag" in target_frames.columns:
        _raw_tf = target_frames["flag"].astype(str).str.strip().str.lower().reset_index(drop=True)
    else:
        _raw_tf = pd.Series(["__none__"] * len(mag_calib))
    base_flags: list[str] = []
    for i in range(len(mag_calib)):
        if bool(sat_flags[i]):
            base_flags.append("saturated")
        elif i < len(_raw_tf) and str(_raw_tf.iloc[i]) == "nondetection":
            base_flags.append("nondetection")
        elif math.isfinite(mag_calib[i]):
            base_flags.append("normal")
        else:
            base_flags.append("no_data")

    # Reporting path (Workstream B): see ``apply_reporting_postprocess``.
    mag_calib_raw, mag_calib, mag_calib_ct, mag_calib_ac, out_flags = apply_reporting_postprocess(
        mag_calib,
        mag_calib_ct,
        target_row=target_row,
        target_name=target_name,
        sat_flags=sat_flags,
        target_frames=target_frames,
        outlier_sigma=outlier_sigma,
        ct_ok=bool(ct_ok),
        ac_ok=bool(ac_ok),
        delta_m_corr=(float(delta_m_corr) if delta_m_corr is not None else None),
        cfg=_cfg,
    )

    # ALG-2: Savitzky-Golay non-linear detrending (Savitzky & Golay 1964)
    # Removes slow systematic trends (airmass is handled by the differential comp ensemble).
    _sg_enabled = bool(_cfg.savgol_detrend_enabled)
    if _sg_enabled:
        mag_calib = savgol_detrend_lc(
            mag_calib=mag_calib,
            bjd=bjd,
            flags=list(out_flags) if out_flags is not None else ["normal"] * len(mag_calib),
            window_frac=float(_cfg.savgol_window_frac),
            polyorder=int(_cfg.savgol_polyorder),
            enabled=True,
        )
        if ac_ok and delta_m_corr is not None and np.isfinite(float(delta_m_corr)):
            mag_calib_ac = mag_calib + float(delta_m_corr)

    # ALG-4: Democratic Detrender (arXiv:2411.09753v2, 2026)
    _dem_enabled = bool(_cfg.democratic_detrend_enabled)
    _mag_democratic: np.ndarray | None = None
    _err_inflation: np.ndarray | None = None
    if _dem_enabled:
        _mag_democratic, _err_inflation = democratic_detrend_lc(
            mag_calib=mag_calib,
            bjd=bjd,
            airmass=airmass_arr,
            flags=list(out_flags) if out_flags is not None else ["normal"] * len(mag_calib),
            window_frac=float(_cfg.democratic_sg_window_frac),
            enabled=True,
        )

    try:
        from check_star_kmag import (  # noqa: PLC0415
            build_comp_photon_mag_from_frames,
            check_kmag_sidecar_path,
            compute_check_ensemble_mag_calib,
            save_check_kmag_sidecar,
        )

        _chk_cid = _chk_cid_pref
        if _chk_cid:
            _ext_lc = dict(comp_lc)
            if _chk_cid not in _ext_lc:
                _chk_series = _get_lc(_chk_cid, all_frames)
                if _chk_series is not None and np.isfinite(_chk_series).any():
                    _ext_lc[_chk_cid] = _chk_series
            if _chk_cid in _ext_lc:
                _phot_ids = list(dict.fromkeys(list(comp_ids) + [_chk_cid]))
                _comp_photon = build_comp_photon_mag_from_frames(all_frames, _phot_ids, src_files)
                _chk_result = compute_check_ensemble_mag_calib(
                    _chk_cid,
                    list(comp_ids),
                    _ext_lc,
                    comp_catalog_mag,
                    comp_quality,
                    comp_rms_map=comp_rms_map,
                    comp_tier_map=comp_tier_map,
                    tier_weights=tier_weights,
                    cfg=_cfg,
                    n_comp_min=2,
                    n_comp_max=int(_cfg.phase01_comparison_n_comp_max),
                    comp_photon_mag=_comp_photon,
                    sigma_sys_mag=_sigma_sys_mag,
                )
                if _chk_result is not None and np.isfinite(_chk_result.kmag).any():
                    save_check_kmag_sidecar(
                        check_kmag_sidecar_path(lc_dir, target_cid),
                        check_cid=_chk_cid,
                        bjd=bjd,
                        source_files=src_files,
                        kmag=_chk_result.kmag,
                        ensemble=_chk_result,
                    )
                else:
                    logging.warning(
                        "[CHECK-KMAG] ensemble returned empty for target=%s check=%s",
                        target_cid,
                        _chk_cid,
                    )
            else:
                logging.warning(
                    "[CHECK-KMAG] check star %s has no LC series for target %s",
                    _chk_cid,
                    target_cid,
                )
        else:
            logging.warning("[CHECK-KMAG] no check star selected for target %s", target_cid)
    except (ImportError, KeyError, TypeError, ValueError, AttributeError, OSError) as _ck_exc:
        logging.warning("[CHECK-KMAG] sidecar skipped for %s: %s", target_cid, _ck_exc)

    # Krok 6: Ulozenie vystupov
    lc_csv = lc_dir / f"lightcurve_{target_cid}.csv"
    if isinstance(_lunar, dict):
        _lc_lunar_phase = float(_lunar.get("lunar_phase_pct", float("nan")))
        _lc_lunar_sep = float(_lunar.get("lunar_separation_deg", float("nan")))
        _lc_lunar_risk = str(_lunar.get("lunar_risk", "UNKNOWN") or "UNKNOWN")
    else:
        _lc_lunar_phase = float("nan")
        _lc_lunar_sep = float("nan")
        _lc_lunar_risk = "UNKNOWN"
    # I-04: exclude epochs with unmatched ensemble scatter from LC export.
    if err_scatter_unmatched_arr is not None and np.any(err_scatter_unmatched_arr):
        _keep_lc = ~np.asarray(err_scatter_unmatched_arr, dtype=bool)
        if err_method_rows is not None and len(err_method_rows) == len(_keep_lc):
            err_method_rows = [err_method_rows[i] for i in range(len(_keep_lc)) if _keep_lc[i]]
        if _mag_democratic is not None and len(_mag_democratic) == len(_keep_lc):
            _mag_democratic = np.asarray(_mag_democratic, dtype=float)[_keep_lc]
        if _err_inflation is not None and len(_err_inflation) == len(_keep_lc):
            _err_inflation = np.asarray(_err_inflation, dtype=float)[_keep_lc]
        (
            bjd,
            hjd,
            jd,
            airmass_arr,
            flip_arr,
            target_lc,
            mag_calib_raw,
            mag_calib,
            mag_calib_ct,
            mag_calib_ac,
            delta_mag,
            err,
            ap_arr,
            out_flags,
            src_files,
            align_fail_arr,
            err_scatter_unmatched_arr,
            catalog_match_mode_list,
            wcs_untrusted_arr,
            err_photon_export,
            err_sem_rel_export,
            err_scint_rel_export,
            err_sigma_sys_rel_export,
        ) = _exclude_err_scatter_unmatched_epochs(
            ~_keep_lc,
            bjd,
            hjd,
            jd,
            airmass_arr,
            flip_arr,
            target_lc,
            mag_calib_raw,
            mag_calib,
            mag_calib_ct,
            mag_calib_ac,
            delta_mag,
            err,
            ap_arr,
            out_flags,
            src_files,
            align_fail_arr,
            err_scatter_unmatched_arr,
            catalog_match_mode_list,
            wcs_untrusted_arr,
            err_photon_export,
            err_sem_rel_export,
            err_scint_rel_export,
            err_sigma_sys_rel_export,
        )
    # Pinned-era LC metadata: preserve anchor ct_n_comp for byte continuity (477dc8cf).
    if bool(ct_ok):
        try:
            from pinned_ensembles import baseline_lc_ct_n_comp_for_target, is_pinned_target  # noqa: PLC0415

            if is_pinned_target(target_cid):
                _pin_ct_n = baseline_lc_ct_n_comp_for_target(target_cid)
                if _pin_ct_n is not None:
                    ct_n_comp = int(_pin_ct_n)
        except Exception as _pin_ct_exc:  # noqa: BLE001
            LOGGER.debug("[PIN] ct_n_comp overlay skip: %s", _pin_ct_exc)
    save_lightcurve_csv(
        lc_csv,
        bjd,
        hjd,
        jd,
        airmass_arr,
        flip_arr,
        target_lc,
        mag_calib_raw,
        mag_calib,
        np.asarray(mag_calib_ct, dtype=np.float64),
        mag_calib_ac,
        delta_mag,
        err,
        ap_arr,
        out_flags,
        src_files,
        ct_correction=(float(ct_corr) if bool(ct_ok) else float("nan")),
        ct_c1=(float(c1) if bool(ct_ok) else float("nan")),
        ct_c1_stderr=(float(c1_stderr) if bool(ct_ok) else float("nan")),
        ct_mode=(str(ct_mode) if bool(ct_ok) else ""),
        ct_bp_rp_target=(float(target_bp_rp) if bool(ct_ok) else float("nan")),
        ct_bp_rp_comp_med=(float(bp_rp_comp_med) if bool(ct_ok) else float("nan")),
        ct_n_comp=(int(ct_n_comp) if bool(ct_ok) else None),
        ct_ok=bool(ct_ok),
        k2_source=k2_source_rows,
        k2_value=(float(k2_value_lc) if math.isfinite(float(k2_value_lc)) else float("nan")),
        k2_colour_ref=(float(k2_colour_ref) if math.isfinite(float(k2_colour_ref)) else float("nan")),
        ac_result=(ac_result if isinstance(ac_result, dict) else None),
        mag_democratic=_mag_democratic,
        err_inflation=_err_inflation,
        lunar_phase_pct=_lc_lunar_phase,
        lunar_separation_deg=_lc_lunar_sep,
        lunar_risk=_lc_lunar_risk,
        dilution_factor=float(_dilution_result.get("dilution_factor", 1.0)),
        method=_lc_export_method,
        alignment_failed=align_fail_arr,
        err_scatter_unmatched=err_scatter_unmatched_arr,
        catalog_match_mode=catalog_match_mode_list,
        wcs_untrusted=wcs_untrusted_arr,
        time_base=time_base,
        err_method=err_method_rows,
        sigma_sys_mag=_sigma_sys_mag,
        err_photon=err_photon_export,
        err_sem_rel=err_sem_rel_export,
        err_scint_rel=err_scint_rel_export,
        err_sigma_sys_rel=err_sigma_sys_rel_export,
        aperture_policy=getattr(state, "aperture_policy", None),
    )
    # EPSF-LC-LOG-01 / INV-PSF-SUBMIT-01: PSF LC files are an internal diagnostic
    # product written by psf_internal_lc (RUN ePSF path), not Phase 2A.

    # COMP-ASSIGN-01 D4: stability AFTER photometry - verdict only (membership unchanged).
    comp_bjd = {cid: _get_comp_bjd_series(cid, all_frames) for cid in comp_ids}
    comp_quality = check_comparison_stability(
        comp_lc,
        comp_rms_map=comp_rms_map,
        comp_bjd=comp_bjd,
        n_comp_min=3,
        outlier_sigma=stability_sigma,
        max_comp_p2p=float(_cfg.phase01_comparison_max_comp_rms),
        max_comp_slope_mmag_hr=float(_cfg.comp_max_slope_mmag_hr),
        comp_slope_significance_k=float(getattr(_cfg, "comp_slope_significance_k", 3.0)),
        common_mode_detrend=True,
        stability_run_flags=state.stability_run_flags,
    )

    # Kvalita comp pre UI (tabulka 'Porovnavacie hviezdy')
    _cq_path = lc_dir / f"comp_quality_{target_cid}.json"
    try:
        selected_tier = ""
        tier4_warning = False
        n_t1 = n_t2 = n_t3 = n_t4 = 0
        try:
            if "selected_tier" in comp_df.columns:
                _sub = _comp_index.get(target_cid, pd.DataFrame())
                if not _sub.empty:
                    stv = str(_sub.iloc[0].get("selected_tier", "") or "").strip()
                    selected_tier = stv
                    tier4_warning = bool(_sub.iloc[0].get("tier4_warning", False))
                    try:
                        n_t1 = int(pd.to_numeric(_sub.iloc[0].get("n_tier1", 0), errors="coerce") or 0)
                        n_t2 = int(pd.to_numeric(_sub.iloc[0].get("n_tier2", 0), errors="coerce") or 0)
                        n_t3 = int(pd.to_numeric(_sub.iloc[0].get("n_tier3", 0), errors="coerce") or 0)
                        n_t4 = int(pd.to_numeric(_sub.iloc[0].get("n_tier4", 0), errors="coerce") or 0)
                    except Exception:  # noqa: BLE001
                        n_t1 = n_t2 = n_t3 = n_t4 = 0
        except Exception:  # noqa: BLE001
            selected_tier = ""

        _cq_payload: dict[str, Any] = {}
        for cid, info in comp_quality.items():
            nk = _normalize_gaia_id(cid)
            q = str(info.get("quality", "") or "").strip()
            note = str(info.get("note", "") or "").strip()
            if q == "good" and not note:
                _cq_payload[nk] = "good"
            else:
                _cq_payload[nk] = {"quality": q, "note": note}
        _cq_payload["selected_tier"] = str(selected_tier)
        _cq_payload["tier4_warning"] = bool(tier4_warning)
        _cq_payload["n_tier1"] = int(n_t1)
        _cq_payload["n_tier2"] = int(n_t2)
        _cq_payload["n_tier3"] = int(n_t3)
        _cq_payload["n_tier4"] = int(n_t4)
        _cq_payload["aperture_correction"] = {
            "ok": (bool(ac_result.get("ok", False)) if isinstance(ac_result, dict) else False),
            "delta_m_corr": (ac_result.get("delta_m_corr") if isinstance(ac_result, dict) else None),
            "scatter_mag": (ac_result.get("scatter_mag") if isinstance(ac_result, dict) else None),
            "n_ref_stars": (int(ac_result.get("n_ref_stars", 0)) if isinstance(ac_result, dict) else 0),
            "ref_star_ids": (ac_result.get("ref_star_ids", []) if isinstance(ac_result, dict) else []),
            "reason": (str(ac_result.get("reason", "disabled")) if isinstance(ac_result, dict) else "disabled"),
        }
        try:
            from pinned_ensembles import get_pinned_provenance_for_target  # noqa: PLC0415

            _pin_prov = get_pinned_provenance_for_target(target_cid)
            if _pin_prov:
                _cq_payload["comp_provenance"] = _pin_prov
        except Exception as _pin_cq_exc:  # noqa: BLE001
            LOGGER.debug("[PIN] comp_provenance sidecar skip: %s", _pin_cq_exc)
        _cq_path.write_text(json.dumps(_cq_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[PHASE 2A] Optional artifact write failed (comp_quality.json): %s", exc)

    lc_png = lc_dir / f"lightcurve_{target_cid}.png"
    if _save_png:
        try:
            save_lightcurve_png(
                lc_png,
                bjd,
                mag_calib,
                err,
                out_flags,
                target_name,
                comp_quality,
                delta_mag_mode=False,
                delta_mag=delta_mag,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("[PHASE 2A] Optional artifact write failed (lightcurve PNG): %s", exc)

    cutout_png = lc_dir / f"cutout_{target_cid}.png"
    if _save_png:
        try:
            save_cutout_png(
                cutout_png,
                Path(masterstar_fits_path),
                float(target_row["x"]),
                float(target_row["y"]),
                target_name,
                ms_data=_ms_data,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("[PHASE 2A] Optional artifact write failed (cutout PNG): %s", exc)

    # Per-target field map s cislovanymi comp hviezdami - vzdy (UI)
    try:
        _target_comp = _comp_index.get(target_cid, pd.DataFrame()).copy()
        _fm_target_path = lc_dir / f"field_map_{target_cid}.png"
        save_target_field_map_png(
            _fm_target_path,
            Path(masterstar_fits_path),
            target_row,
            _target_comp,
            ms_data=_ms_data,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[PHASE 2A] Optional artifact write failed (field map PNG): %s", exc)

    # Summary riadok
    finite_calib = mag_calib[np.isfinite(mag_calib)]
    n_good_comp = sum(
        1 for q in comp_quality.values() if q.get("quality") in ("good", "suspect")
    )
    n_stability_good = sum(1 for q in comp_quality.values() if q.get("quality") == "good")
    n_stability_suspect = sum(1 for q in comp_quality.values() if q.get("quality") == "suspect")
    n_sat = sum(1 for f in out_flags if f == "saturated")

    _measured_ap = (
        float(_measured_ap_target)
        if math.isfinite(_measured_ap_target) and _measured_ap_target > 0
        else float("nan")
    )
    if not math.isfinite(_measured_ap) and not target_frames.empty and "aperture_r_px" in target_frames.columns:
        _ap_meas = pd.to_numeric(target_frames["aperture_r_px"], errors="coerce").dropna()
        if not _ap_meas.empty:
            _measured_ap = float(np.median(_ap_meas.to_numpy(dtype=float)))
    _lc_rms_full = float(np.std(finite_calib)) if len(finite_calib) > 1 else float("nan")
    _lc_rms_ooe = compute_lc_rms_ooe(mag_calib, out_flags)

    _comp_path = "default"
    _n_tier12 = 0
    if not target_comps.empty:
        if "comp_path" in target_comps.columns:
            _cpaths = target_comps["comp_path"].astype(str).str.strip().str.lower()
            if (_cpaths == "sparse_fallback").any():
                _comp_path = "sparse_fallback"
        if "comp_tier" in target_comps.columns:
            _tiers = pd.to_numeric(target_comps["comp_tier"], errors="coerce")
            _n_tier12 = int(_tiers.isin([1, 2]).sum())

    _sum_row: dict[str, Any] = {
        "catalog_id": target_cid,
        "vsx_name": target_name,
        "vsx_type": target_vsx_type,
        "zone_flag": str(target_row.get("zone_flag", "")).strip(),
        "n_frames": len(bjd),
        "n_good_comp": n_good_comp,
        "n_tier12": _n_tier12,
        "comp_path": _comp_path,
        "n_stability_good": n_stability_good,
        "n_stability_suspect": n_stability_suspect,
        "n_saturated": n_sat,
        "n_alignment_failed": n_alignment_failed,
        "alignment_failed_frac": alignment_failed_frac,
        "n_wcs_untrusted": n_wcs_untrusted,
        "wcs_untrusted_frac": wcs_untrusted_frac,
        "lc_rms": _lc_rms_full,
        "lc_rms_ooe": _lc_rms_ooe,
        "lc_median_mag": float(np.median(finite_calib)) if len(finite_calib) > 0 else float("nan"),
        "aperture_px": _measured_ap if math.isfinite(_measured_ap) else float(apertures_px.get(target_cid, float("nan"))),
        "aperture_px_planned": float(apertures_px.get(target_cid, float("nan"))),
        "am_slope": float("nan"),
        "am_detrended": False,
        "dilution_factor": float(_dilution_result.get("dilution_factor", 1.0)),
        "dilution_delta_mag": float(_dilution_result.get("dilution_delta_mag", 0.0)),
        "n_neighbors_aperture": int(_dilution_result.get("n_neighbors", 0)),
        "gs11_aperture_arcsec": float(_dilution_result.get("aperture_arcsec", float("nan"))),
        "gs11_dilution_skipped": bool(_dilution_result.get("dilution_skipped", False)),
        "gs11_dilution_skip_reason": str(_dilution_result.get("dilution_skip_reason", "") or ""),
        "mag_median_pre_gs11": _mag_pre_gs11,
        "mag_median_post_gs11": _mag_post_gs11,
        "lc_csv": str(lc_csv),
        "lc_png": str(lc_png),
        "ct_ok": bool(ct_ok),
        "ct_corr": float(ct_corr) if bool(ct_ok) and math.isfinite(float(ct_corr)) else float("nan"),
        "ct_c1": float(c1) if bool(ct_ok) and math.isfinite(float(c1)) else float("nan"),
        "ct_c1_stderr": float(c1_stderr) if bool(ct_ok) and math.isfinite(float(c1_stderr)) else float("nan"),
        "ct_mode": str(ct_mode) if bool(ct_ok) else "",
        "ct_n_comp": int(ct_n_comp) if bool(ct_ok) else 0,
        **_ac_summary_fields(ac_result if bool(_cfg.aperture_correction_enabled) else {"ok": False, "reason": "disabled"}),
    }
    if _pfs_on:
        _sum_row["skip_reason"] = str(target_row.get("skip_reason", "") or "")
        _sum_row["sat_clean_frac"] = float(
            pd.to_numeric(target_row.get("sat_clean_frac"), errors="coerce")
        )
        _sum_row["per_frame_sat_fallback"] = bool(
            target_row.get("per_frame_sat_fallback", False)
        )
    summary_rows.append(_sum_row)
    n_lc += 1
    lc_rms = float(summary_rows[-1]["lc_rms"])
    lc_rms_ooe = float(summary_rows[-1].get("lc_rms_ooe", float("nan")))
    r_ap = float(summary_rows[-1]["aperture_px"])
    logging.info(
        f"[FAZA 2A] {target_name}: "
        f"lc_rms={lc_rms:.4f}, lc_rms_ooe={lc_rms_ooe:.4f}, "
        f"n_comp={n_good_comp} (stability_good={n_stability_good}), "
        f"apertura={r_ap:.2f}px (measured)"
    )


    state.chip_fw = chip_fw
    state.chip_fh = chip_fh
    return summary_rows, n_lc









# ======================================================================
# photometry.py (zlucene do photometry_core)
# ======================================================================

from utils import (
    fits_binning_xy_from_header,
    plate_scale_arcsec_per_pixel,
    plate_solve_fov_deg_diagonal_from_scale,
)











# Stlpce nacitavane z per-frame CSV pre bootstrap (78 % uspora pamate)
_PHASE_USECOLS_PERFRAME: list[str] = [
    "name",
    "catalog_id",
    "bjd_tdb_mid",
    "flux",
    "dao_flux",
    "noise_floor_adu",
    "sky_adu_per_px_annulus",
    "aperture_r_px",
    "is_usable",
    "is_saturated",
    "is_noisy",
    "snr50_ok",
    "vsx_known_variable",
    "likely_saturated",
]
















def _finite_pixel_bbox_from_array(
    data: "np.ndarray",
    *,
    finite_stride: int,
) -> tuple[float, float, float, float] | None:
    """Per-frame finite-pixel bbox from a 2-D array (strided sampling)."""
    import numpy as np

    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2:
        return None
    stride = max(1, int(finite_stride))
    samp = arr[::stride, ::stride]
    fin = np.isfinite(samp)
    if not bool(fin.any()):
        return None
    ys, xs = np.where(fin)
    x0 = float(xs.min() * stride)
    y0 = float(ys.min() * stride)
    x1 = float(min(arr.shape[1] - 1, xs.max() * stride + (stride - 1)))
    y1 = float(min(arr.shape[0] - 1, ys.max() * stride + (stride - 1)))
    return (x0, y0, x1, y1)


def _intersection_bbox_from_frame_bboxes(
    bboxes: list[tuple[float, float, float, float]],
) -> tuple[float, float, float, float] | None:
    if len(bboxes) < 2:
        return None
    x0_i, y0_i = 0.0, 0.0
    x1_i, y1_i = float("inf"), float("inf")
    for x0, y0, x1, y1 in bboxes:
        x0_i = max(x0_i, x0)
        y0_i = max(y0_i, y0)
        x1_i = min(x1_i, x1)
        y1_i = min(y1_i, y1)
    if not (math.isfinite(x0_i) and math.isfinite(y0_i) and math.isfinite(x1_i) and math.isfinite(y1_i)):
        return None
    if x1_i <= x0_i or y1_i <= y0_i:
        return None
    return (x0_i, y0_i, x1_i, y1_i)












def compute_auto_fwhm_limit(
    fwhm_values: np.ndarray | Sequence[float],
    k: float = 1.5,
) -> dict[str, Any]:
    """
    Vypocita automaticky FWHM limit pomocou MAD statistiky.

    Vracia dict:
        median_fwhm, mad, sigma_mad, auto_limit, k, n_total, n_kept, n_cut
    (``auto_limit`` moze byt ``None`` pri prilis malo bodoch.)
    """
    arr = np.asarray(fwhm_values, dtype=np.float64)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if len(arr) < 3:
        return {
            "median_fwhm": None,
            "mad": None,
            "sigma_mad": None,
            "auto_limit": None,
            "k": float(k),
            "n_total": int(len(arr)),
            "n_kept": int(len(arr)),
            "n_cut": 0,
        }
    median_f = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median_f)))
    sigma_mad = mad * 1.4826
    auto_limit = median_f + float(k) * sigma_mad
    n_kept = int(np.sum(arr <= auto_limit))
    return {
        "median_fwhm": round(median_f, 3),
        "mad": round(mad, 4),
        "sigma_mad": round(sigma_mad, 4),
        "auto_limit": round(float(auto_limit), 3),
        "k": float(k),
        "n_total": int(len(arr)),
        "n_kept": n_kept,
        "n_cut": int(len(arr) - n_kept),
    }






def _aperture_flux_sky_batch(
    d: np.ndarray,
    pos: np.ndarray,
    r_ap: float | np.ndarray,
    r_in: float | np.ndarray,
    r_out: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sky-subtracted circular aperture sums (photutils ``method='exact'``).

    Shared by production per-star photometry and the scatter/CoG radius ladder
    (IMPL-04). When all stars share one radius, uses one batch aperture call.
    """
    from photutils.aperture import CircularAnnulus, CircularAperture
    from photutils.aperture import aperture_photometry as _aphot

    pos = np.asarray(pos, dtype=np.float64)
    n = int(pos.shape[0])
    flux_arr = np.full(n, np.nan, dtype=np.float64)
    sky_pp_arr = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return flux_arr, sky_pp_arr

    r_ap_arr = np.full(n, float(r_ap), dtype=np.float64) if np.isscalar(r_ap) else np.asarray(r_ap, dtype=np.float64)
    r_in_arr = np.full(n, float(r_in), dtype=np.float64) if np.isscalar(r_in) else np.asarray(r_in, dtype=np.float64)
    r_out_arr = np.full(n, float(r_out), dtype=np.float64) if np.isscalar(r_out) else np.asarray(r_out, dtype=np.float64)

    # Uniform radius: one batch exact sum.
    if (
        np.all(np.isfinite(r_ap_arr))
        and np.all(r_ap_arr > 0)
        and float(np.nanmax(r_ap_arr) - np.nanmin(r_ap_arr)) < 1e-9
        and np.all(np.isfinite(r_in_arr))
        and np.all(np.isfinite(r_out_arr))
        and float(np.nanmax(r_in_arr) - np.nanmin(r_in_arr)) < 1e-9
        and float(np.nanmax(r_out_arr) - np.nanmin(r_out_arr)) < 1e-9
        and float(r_out_arr[0]) > float(r_in_arr[0]) > 0
    ):
        r0 = float(r_ap_arr[0])
        rin = float(r_in_arr[0])
        rout = float(r_out_arr[0])
        try:
            ap = CircularAperture(pos, r=r0)
            phot = _aphot(d, ap, method="exact")
            sums = np.asarray(phot["aperture_sum"], dtype=np.float64)
            area = float(ap.area)
            an = CircularAnnulus(pos, r_in=rin, r_out=rout)
            masks = an.to_mask(method="center")
            if not isinstance(masks, (list, tuple)):
                masks = [masks]
            for i, m in enumerate(masks):
                try:
                    ann_img = m.to_image(d.shape)
                    sky_pp_arr[i] = _sky_pp_from_annulus_image(d, ann_img)
                except Exception:  # noqa: BLE001
                    sky_pp_arr[i] = float("nan")
            flux_arr = sums - sky_pp_arr * area
            return flux_arr, sky_pp_arr
        except Exception as exc:  # noqa: BLE001
            logging.debug("[PHOT] batch exact aperture failed, falling back per-star: %s", exc)

    # Heterogeneous radii: per-star exact (photutils 2.3 scalar-r constraint).
    return _aperture_flux_sky_per_star(d, pos, r_ap_arr, r_in_arr, r_out_arr)




def compute_per_frame_cog_correction(
    data: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    dao_flux: np.ndarray,
    aperture_r_px: np.ndarray,
    sky_pp: np.ndarray,
    *,
    fwhm_px: float,
    peak_max_adu: np.ndarray | None = None,
    sat_limit_adu: np.ndarray | None = None,
    ref_fwhm: float = 4.5,
    ladder_step_px: float = 0.5,
    min_stars: int = 8,
    isolation_fwhm: float = 6.0,
    snr_min: float = 50.0,
    sat_frac: float = 0.85,
    gain: float = 1.0,
    read_noise: float = 10.0,
    ac_factor_max: float = 5.0,
    max_stars: int = 60,
    fallback_ee: tuple[np.ndarray, np.ndarray] | None = None,
    ladder_outer_factor: float = 1.0,
) -> dict[str, Any]:
    """Per-frame curve-of-growth (encircled-energy) aperture correction.

    Builds an EE(r) curve (normalised to ``ref_fwhm x FWHM``) from bright, isolated,
    unsaturated, high-SNR stars and returns a per-star multiplicative correction
    ``ac_factor = 1 / EE(r_star)`` that puts every star on the common ref-radius
    enclosed-flux scale (removing the per-star SNR-radius differential bias).

    ``ladder_outer_factor`` > 1 extends the ladder past the normalisation radius so a
    real flatness check can compare EE(outer) to EE(ref) (IMPL-02). Normalisation
    remains at ``ref_r``, not at the last ladder point.

    Returns dict: ``ac_factor`` (len n, >=1.0), ``cog_ok``, ``n_cog``, ``ref_r_px``,
    ``ee_radii``, ``ee_curve``, ``flatness_outer_over_norm``. When fewer than
    ``min_stars`` COG stars are found and no ``fallback_ee`` is given, ``cog_ok=False``
    and every ``ac_factor=1.0``.
    """
    from photutils.aperture import CircularAperture
    from photutils.aperture import aperture_photometry as _aphot

    n = int(len(x))
    out: dict[str, Any] = {
        "ac_factor": np.ones(n, dtype=np.float64),
        "cog_ok": False,
        "n_cog": 0,
        "ref_r_px": float(ref_fwhm) * float(fwhm_px) if math.isfinite(fwhm_px) else float("nan"),
        "ee_radii": None,
        "ee_curve": None,
        "flatness_outer_over_norm": float("nan"),
        "ladder_outer_r_px": float("nan"),
    }
    if n == 0 or not (math.isfinite(fwhm_px) and fwhm_px > 0):
        return out

    d = np.asarray(data, dtype=np.float64)
    if np.any(~np.isfinite(d)):
        fill = float(np.nanmedian(d)) if np.any(np.isfinite(d)) else 0.0
        d = np.where(np.isfinite(d), d, fill)
    height, width = d.shape

    ref_r = float(ref_fwhm) * float(fwhm_px)
    try:
        outer_factor = float(ladder_outer_factor)
    except (TypeError, ValueError):
        outer_factor = 1.0
    if not math.isfinite(outer_factor) or outer_factor < 1.0:
        outer_factor = 1.0
    outer_r = float(outer_factor) * ref_r
    # Isolation uses the configured multiple of FWHM (do not force >= ref_r).
    iso_r = float(isolation_fwhm) * float(fwhm_px)
    step = float(ladder_step_px) if math.isfinite(ladder_step_px) and ladder_step_px > 0 else 0.5
    radii = np.arange(step, outer_r + 1e-6, step, dtype=np.float64)
    if radii.size == 0 or radii[-1] < outer_r - 1e-6:
        radii = np.append(radii, outer_r)
    # Ensure both normalisation and outer radii are exact ladder points.
    if not np.any(np.isclose(radii, ref_r, rtol=0.0, atol=1e-6)):
        radii = np.sort(np.append(radii, ref_r))
    if not np.any(np.isclose(radii, outer_r, rtol=0.0, atol=1e-6)):
        radii = np.sort(np.append(radii, outer_r))
    i_norm = int(np.argmin(np.abs(radii - ref_r)))
    radii[i_norm] = ref_r
    out["ladder_outer_r_px"] = float(radii[-1])

    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    flux = np.asarray(dao_flux, dtype=np.float64)
    rap = np.asarray(aperture_r_px, dtype=np.float64)
    skp = np.asarray(sky_pp, dtype=np.float64)

    finite_xy = np.isfinite(xx) & np.isfinite(yy)
    nn = np.full(n, np.inf, dtype=np.float64)
    try:
        from scipy.spatial import cKDTree

        pts = np.column_stack([xx[finite_xy], yy[finite_xy]])
        if pts.shape[0] >= 2:
            tree = cKDTree(pts)
            dist, _ = tree.query(pts, k=2)
            nn[finite_xy] = dist[:, 1]
    except (ImportError, AttributeError, ValueError, TypeError):
        logging.debug("[COG] cKDTree unavailable - isolation check skipped")

    g = float(gain) if math.isfinite(gain) and gain > 0 else 1.0
    rn = float(read_noise) if math.isfinite(read_noise) and read_noise >= 0 else 10.0
    area_rap = math.pi * np.square(np.where(np.isfinite(rap) & (rap > 0), rap, np.nan))
    var = flux / g + np.maximum(0.0, skp) / g * area_rap + (rn / g) ** 2 * area_rap
    snr = np.where((flux > 0) & np.isfinite(var) & (var > 0), flux / np.sqrt(var), 0.0)

    if peak_max_adu is not None and sat_limit_adu is not None:
        pk = np.asarray(peak_max_adu, dtype=np.float64)
        sl = np.asarray(sat_limit_adu, dtype=np.float64)
        unsat = ~(np.isfinite(pk) & np.isfinite(sl) & (pk > float(sat_frac) * sl))
    else:
        unsat = np.ones(n, dtype=bool)

    margin = float(radii[-1]) + 1.0
    in_bounds = (xx > margin) & (xx < (width - margin)) & (yy > margin) & (yy < (height - margin))

    sel = (
        finite_xy
        & (flux > 0)
        & np.isfinite(skp)
        & unsat
        & in_bounds
        & (nn > iso_r)
        & (snr >= float(snr_min))
    )
    cog_idx = np.where(sel)[0]
    # Cap to the highest-SNR subset - a robust median EE needs only a few dozen stars.
    if int(max_stars) > 0 and cog_idx.size > int(max_stars):
        order = np.argsort(snr[cog_idx])[::-1][: int(max_stars)]
        cog_idx = cog_idx[order]

    fracs: list[np.ndarray] = []
    for i in cog_idx:
        try:
            xy = [(float(xx[i]), float(yy[i]))]
            sums = np.array(
                [
                    float(_aphot(d, CircularAperture(xy, r=float(rr)), method="exact")["aperture_sum"][0])
                    for rr in radii
                ],
                dtype=np.float64,
            )
            ee = sums - float(skp[i]) * math.pi * np.square(radii)
            ref_val = float(ee[i_norm])
            if math.isfinite(ref_val) and ref_val > 0:
                fr = ee / ref_val
                if np.all(np.isfinite(fr)):
                    fracs.append(fr)
        except Exception as exc:  # noqa: BLE001
            logging.error('[EXC-0190] One bright star skipped in COG encircled-energy curve - aperture COG correction biased: %s', exc)
            continue

    n_cog = len(fracs)
    out["n_cog"] = n_cog
    ee_radii = radii
    ee_curve: np.ndarray | None = None
    if n_cog >= int(min_stars):
        ee_curve = np.median(np.vstack(fracs), axis=0)
        ee_curve = np.clip(ee_curve, 1e-3, None)
        # Renormalise so EE(ref)=1 exactly after the median (outer may exceed 1 slightly).
        norm_v = float(ee_curve[i_norm])
        if math.isfinite(norm_v) and norm_v > 0:
            ee_curve = ee_curve / norm_v
        ee_curve[i_norm] = 1.0
        out["cog_ok"] = True
        out["flatness_outer_over_norm"] = float(ee_curve[-1])
    elif fallback_ee is not None:
        ee_radii, ee_curve = fallback_ee
        ee_radii = np.asarray(ee_radii, dtype=np.float64)
        ee_curve = np.asarray(ee_curve, dtype=np.float64)
        out["cog_ok"] = False  # fallback used; flag as not-fresh
    else:
        return out  # too few COG stars and no fallback -> no correction (ac_factor=1)

    out["ee_radii"] = ee_radii
    out["ee_curve"] = ee_curve

    ee_at = np.interp(np.clip(rap, ee_radii[0], ee_radii[-1]), ee_radii, ee_curve)
    acf = np.where((ee_at > 0) & np.isfinite(ee_at), 1.0 / ee_at, 1.0)
    acf = np.clip(acf, 1.0, float(ac_factor_max))
    acf = np.where(np.isfinite(rap) & (rap > 0), acf, 1.0)
    out["ac_factor"] = acf
    return out
















































def run_full_photometry_pipeline(
    *,
    masterstar_fits_path: Path,
    variable_targets_csv: Path,
    masterstars_csv: Path,
    per_frame_csv_dir: Path,
    detrended_aligned_dir: Path,
    output_dir: Path,
    cfg: AppConfig | None = None,
    db: Any = None,
    draft_id: int | None = None,
    progress_cb: Any = None,
) -> dict[str, Any]:
    """Jedno-krokovy wrapper: Faza 0+1 + Faza 2A ako jeden celok.

    UI to pouziva ako jednu akciu 'RUN Aperture Photometry' pre dany obs_group.
    """
    _cfg = cfg or AppConfig()

    ensure_full_variable_targets_if_presel_stub(
        variable_targets_csv=Path(variable_targets_csv),
        masterstars_csv=Path(masterstars_csv),
        masterstar_fits=Path(masterstar_fits_path),
        cfg=_cfg,
        draft_id=draft_id,
    )

    def _p(msg: str) -> None:
        if progress_cb is None:
            return
        try:
            progress_cb(str(msg))
        except UnicodeEncodeError:
            progress_cb(str(msg).encode("ascii", "backslashreplace").decode("ascii"))

    # FWHM: prefer header (VY_FWHM_GAUSS/VY_FWHM), inak default z configu.
    fwhm_px = _resolve_sips_dao_fwhm_px(_cfg, fwhm_px=None)
    _ms_header_shared: Any | None = None
    _ms_path_shared = Path(masterstar_fits_path)
    if _ms_path_shared.is_file():
        try:
            from astropy.io import fits as astrofits  # noqa: PLC0415

            with astrofits.open(_ms_path_shared, memmap=False) as _hdul:
                _ms_header_shared = _hdul[0].header.copy()
        except Exception as exc:  # noqa: BLE001
            logging.error('[EXC-0220] Shared MASTERSTAR header cache load fails - repeated FITS opens (perf), not science num...: %s', exc)
            logging.warning("[PERF-2] Cannot open MASTERSTAR.fits for header: %s", exc)
    if _ms_header_shared is not None:
        try:
            # Prefer night seeing (VY_FWHM) over Gaussian-core (VY_FWHM_GAUSS) for
            # Phase-1 isolation / blend geometry. Core FWHM under-states the CoG
            # 3-FWHM single-source radius (COMP-ASSIGN-03 / A-1).
            for key in ("VY_FWHM", "VY_FWHM_GAUSS", "VY_FWHM_GAUSSIAN"):
                v = _ms_header_shared.get(key)
                if v is None:
                    continue
                fv = float(v)
                if 0.5 < fv < 30.0:
                    fwhm_px = fv
                    break
        except Exception as exc:  # noqa: BLE001
            logging.error('[EXC-0221] VY_FWHM/VY_FWHM_GAUSS header parse fails - pipeline uses default/config FWHM for phase0+1: %s', exc)
            pass

    # -- FAZA 0+1 --
    _p("Faza 0+1: select targets + comparison stars...")
    _plate_scale = _get_plate_scale_from_cfg(
        _cfg,
        db=db,
        draft_id=draft_id,
        fits_path=Path(masterstar_fits_path),
        ms_header=_ms_header_shared,
    )
    if _plate_scale is None:
        _plate_scale = _read_plate_scale_from_fits_path(
            Path(masterstar_fits_path),
            ms_header=_ms_header_shared,
        )
        if _plate_scale is not None and math.isfinite(float(_plate_scale)) and float(_plate_scale) > 0:
            logging.info(
                "[FOV] plate_scale from MASTERSTAR.fits header -> %.4f arcsec/px",
                float(_plate_scale),
            )
    _fw_pipe, _fh_pipe, _frame_hw_src = _resolve_frame_hw_px_from_masterstar(
        Path(masterstar_fits_path),
        frame_w_px=int(_cfg.frame_width_px),
        frame_h_px=int(_cfg.frame_height_px),
        db=db,
        draft_id=draft_id,
    )
    if _frame_hw_src != "caller_default":
        logging.info(
            "[PHASE 0+1] Pipeline frame dimensions %dx%d px from %s",
            int(_fw_pipe),
            int(_fh_pipe),
            _frame_hw_src,
        )
    _pt_mags = _cfg.phase01_tier_mags()
    p01 = run_phase0_and_phase1(
        variable_targets_csv=Path(variable_targets_csv),
        masterstars_csv=Path(masterstars_csv),
        per_frame_csv_dir=Path(per_frame_csv_dir),
        output_dir=Path(output_dir),
        fwhm_px=float(fwhm_px),
        frame_w_px=int(_fw_pipe),
        frame_h_px=int(_fh_pipe),
        chip_interior_margin_px=_resolve_chip_interior_margin_px(_cfg, arcsec_per_px=_plate_scale),
        plate_scale_arcsec_px=_plate_scale,
        max_dist_deg=_compute_fov_max_dist(
            frame_w_px=int(_fw_pipe),
            frame_h_px=int(_fh_pipe),
            plate_scale=_plate_scale,
            fov_fraction=float(_cfg.phase01_comparison_fov_fraction),
            fallback_deg=resolve_max_dist_fallback_deg(
                _cfg,
                frame_w_px=int(_fw_pipe),
                frame_h_px=int(_fh_pipe),
                plate_scale_arcsec_px=_plate_scale,
            ),
        ),
        max_mag_diff=float(_cfg.phase01_comparison_max_mag_diff),
        comp_max_delta_bprp=float(_cfg.comp_max_delta_bprp),
        max_mag_diff_t1=float(_pt_mags[0]),
        max_mag_diff_t2=float(_pt_mags[1]),
        max_mag_diff_t3=float(_pt_mags[2]),
        max_mag_diff_t4=float(_pt_mags[3]),
        n_comp_min=int(_cfg.phase01_comparison_n_comp_min),
        n_comp_max=int(_cfg.phase01_comparison_n_comp_max),
        max_comp_rms=float(_cfg.phase01_comparison_max_comp_rms),
        min_dist_arcsec=float(_cfg.phase01_comparison_min_dist_arcsec),
        min_frames_frac=float(_cfg.phase01_comparison_min_frames_frac),
        rms_outlier_sigma=3.0,
        exclude_gaia_nss=bool(_cfg.phase01_comparison_exclude_gaia_nss),
        exclude_gaia_extobj=bool(_cfg.phase01_comparison_exclude_gaia_extobj),
        mag_bright_threshold=float(_cfg.phase01_comparison_mag_bright_threshold),
        max_mag_diff_bright_floor=float(
            _cfg.phase01_comparison_max_mag_diff_bright_floor or 0.0
        ),
        max_psf_chi2=float(_cfg.phase01_comparison_max_psf_chi2),
        max_fwhm_factor=float(_cfg.phase01_comparison_max_fwhm_factor),
        isolation_radius_px=_resolve_isolation_radius_px(_cfg, arcsec_per_px=_plate_scale),
        flux_col=_cfg.phase01_flux_col,
        cfg=_cfg,
        progress_cb=progress_cb,
        draft_id=draft_id,
        db=db,
    )

    active_targets_csv = Path(str(p01.get("active_targets_csv") or ""))
    comparison_stars_csv = Path(str(p01.get("comparison_stars_csv") or ""))
    n_active = int(p01.get("n_active_targets") or 0)
    if n_active <= 0:
        return {
            "phase01": p01,
            "phase2a": None,
            "output_dir": str(Path(output_dir)),
            "zero_targets": True,
            "n_active_targets": 0,
        }
    if not active_targets_csv.is_file() or not comparison_stars_csv.is_file():
        return {
            "phase01": p01,
            "phase2a": None,
            "output_dir": str(Path(output_dir)),
            "error": "Faza 0+1 nevygenerovala active_targets/comparison_stars CSV.",
        }

    # INV-DAG-01: phase01 stamp after successful Phase 0+1.
    try:
        from invariants_runtime import stamp_stage_on_disk  # noqa: PLC0415

        stamp_stage_on_disk(Path(output_dir), "phase01", enforce_upstream=True)
    except Exception as _dag_p01_exc:  # noqa: BLE001
        logging.debug("[INV-DAG-01] phase01 stamp skipped: %s", _dag_p01_exc)

    # -- FAZA 2A --
    _p("Faza 2A: aperture photometry + lightcurves...")
    _cfg2a = p01.get("cfg_effective_for_photometry") or _cfg
    p2a = run_phase2a(
        masterstar_fits_path=Path(masterstar_fits_path),
        active_targets_csv=active_targets_csv,
        comparison_stars_csv=comparison_stars_csv,
        per_frame_csv_dir=Path(per_frame_csv_dir),
        detrended_aligned_dir=Path(detrended_aligned_dir),
        output_dir=Path(output_dir),
        fwhm_px=float(fwhm_px),
        annulus_inner_fwhm=float(_cfg.annulus_inner_fwhm),
        annulus_outer_fwhm=float(_cfg.annulus_outer_fwhm),
        cfg=_cfg2a,
        progress_cb=progress_cb,
        db=db,
        draft_id=draft_id,
        proc_frame_store=p01.get("proc_store"),
    )

    sysrem_result: dict[str, Any] | None = None
    if bool(_cfg.sysrem_enabled):
        _p("SysRem: removing systematic trends...")
        _sysrem_lc_dir = Path(output_dir) / "lightcurves"
        sysrem_result = run_sysrem_field(
            _sysrem_lc_dir,
            n_iter=int(_cfg.sysrem_n_iter),
        )
        logging.info(
            "[SysRem] %d stars | RMS improvement %.1f%% (%d iter)",
            int(sysrem_result.get("n_stars", 0)),
            float(sysrem_result.get("rms_improvement_pct", float("nan"))),
            int(sysrem_result.get("n_iter", 0)),
        )

    return {
        "phase01": p01,
        "phase2a": p2a,
        "sysrem": sysrem_result,
        "output_dir": str(Path(output_dir)),
        "proc_frame_store": p01.get("proc_store"),
    }




__all__ = [
    # photometry (legacy)
    "StressTestResult",
    "_get_lc_adaptive",
    "apply_reporting_postprocess",
    "check_comparison_stability",
    "common_field_intersection_bbox_px",
    "compute_aperture_correction",
    "compute_fwhm_gaussian_for_aperture_catalog",
    "compute_lc_rms_ooe",
    "compute_mag_calib_final",
    "compute_optimal_apertures",
    "detect_outliers",
    "empirical_feature_mask_mag",
    "enhance_catalog_dataframe_aperture_bpm",
    "ensemble_normalize",
    "ensure_full_variable_targets_if_presel_stub",
    "load_epsf_metrics_for_draft",
    # photometry_phase2a (legacy)
    "measure_fwhm_from_masterstar",
    "pytics_iterative_weights",
    "read_flux_from_csv",
    "recommended_aperture_by_color",
    "resolve_apply_color_term",
    "run_full_photometry_pipeline",
    "run_phase0_and_phase1",
    "run_phase2a",
    "run_sysrem_field",
    "save_cutout_png",
    "save_field_map_png",
    "save_lightcurve_csv",
    "save_lightcurve_png",
    "save_target_field_map_png",
    "select_active_targets",
    "select_comparison_stars_per_target",
    "stress_test_relative_rms_from_sidecars",
    "vsx_is_known_variable_top3_per_bin",
]

from photometry_ui_helpers import (  # noqa: E402,F401
    resolve_lc_time_base,
    lc_time_axis_short_label,
)

from photometry_shared import (  # noqa: E402,F401
    _safe_polyfit,
    _normalize_gaia_id,
    finalize_hybrid_bkg_fallback_proc_dir,
    stamp_masterstar_snr_columns,
    _target_display_name,
    stamp_vsx_known_variable_on_masterstars,
    build_gs11_summary,
    _get_lc_adaptive,
    _get_plate_scale_from_cfg,
    _resolve_plate_scale_arcsec_per_px,
    _cd_matrix_scale_arcsec_per_px,
    _read_plate_scale_from_fits_path,
    _angular_distance_deg,
    StressTestResult,
    stress_test_relative_rms_from_sidecars,
    vsx_is_known_variable_top3_per_bin,
    common_field_intersection_bbox_px_from_arrays,
    common_field_intersection_bbox_px,
    recommended_aperture_by_color,
    bad_columns_for_light_frame,
    _fwhm_moment_at,
    compute_fwhm_gaussian_for_aperture_catalog,
    enhance_catalog_dataframe_aperture_bpm,
)

import sys as _sys_e4
_p2a_mod = _sys_e4.modules.get("photometry_phase2a")
if _p2a_mod is None or getattr(_p2a_mod, "run_phase2a", None) is not None:
    from photometry_phase2a import (  # noqa: E402,F401
    parse_comp_quality_json_map,
    _build_csv_lookup,
    _lookup_star_in_csv,
    _sat_limit_peak_adu,
    _mad_sigma_or_std_floor,
    measure_fwhm_from_masterstar,
    compute_optimal_apertures,
    _howell_variance_adu2,
    _photometric_error,
    _photometric_error_with_bkg_mode,
    _phase2a_proc_column_requirements,
    _phase2a_cache_columns,
    _phase2a_empirical_sigma_bkg_ap,
    _sky_pp_for_photometric_error,
    _resolve_phase2a_equipment_id,
    _draft_dir_from_phase2a_paths,
    _require_comparison_stars_per_target_schema,
    _median_sky_from_phase2a_csv_cache,
    _measured_aperture_from_proc_cache,
    _resolve_photometric_aperture_px_for_gs11,
    read_flux_from_csv,
    compute_aperture_correction,
    fit_color_term_c1,
    should_apply_color_term,
    _obs_group_filter_key,
    resolve_apply_color_term,
    _ColorTermGroupFit,
    _group_comp_mag_inst_from_flux_matrix,
    _group_comp_mag_inst_from_proc_csvs,
    _comp_maps_from_comparison_stars_csv,
    _phase2a_attempt_k2_night_fit,
    _compute_group_color_term_fit,
    _ensure_group_comp_pool_csv,
    _target_row_is_vsx_known_variable,
    empirical_feature_mask_mag,
    detect_outliers,
    apply_reporting_postprocess,
    democratic_detrend_lc,
    save_field_map_png,
    _edge_ok_from_masterstar_pipeline,
    resolve_variable_targets_csv,
    auto_export_variability_candidates_csv,
    _phase2a_coerce_skip_photometry,
    build_rms_mag_model,
    expected_rms_from_model,
    classify_lc_quality,
    build_lc_quality_summary,
    _phase2a_write_summary,
    _phase2a_observer_location_dict,
    _sky_surface_meta_from_qc,
    _phase2a_resolve_field_center_ra_dec,
    _phase2a_collect_session_jd_values,
    _Phase2AState,
    _build_phase2a_dynamic_params,
    _phase2a_compute_lunar_context,
    _preserve_nondetection_flags_helper,
    _proc_stem,
    _compute_frame_align_residuals,
    _record_align_residuals_to_report,
    _frame_align_residual_gate_select,
    _propagate_phase2a_skip_reason_to_active,
    _phase2a_finalize_exports,
        run_phase2a,
    )

from photometry_lightcurve import (  # noqa: E402,F401
    _ac_summary_fields,
    _phase2a_empty_comp_summary_row,
    _phase2a_skip_empty_comps_target,
    _coerce_bool_cell,
    _frame_has_usable_cog,
    evaluate_cog_night_apcorr_gate,
    temporal_bin_comp_lc,
    pytics_iterative_weights,
    _common_mode_detrend_comp_lc,
    _comp_lc_frame_ensemble_residual,
    compute_lc_rms_ooe,
    check_comparison_stability,
    ensemble_normalize,
    _ensemble_scatter_by_source_file,
    _combine_err_with_ensemble_scatter_keyed,
    _err_budget_components_keyed,
    _exclude_err_scatter_unmatched_epochs,
    ct_ensemble_reference_maps,
    apply_color_term,
    _check_color_term_extrapolation,
    _ct_prototype_enabled,
    _color_term_cat_inst_scatter_pair,
    _append_ct_prototype_row,
    savgol_detrend_lc,
    compute_mag_calib_final,
    save_lightcurve_csv,
    save_lightcurve_png,
    save_cutout_png,
    save_target_field_map_png,
    pfs_rescue_eligible,
    _keep_recorded_skip_reason,
    decide_target_saturation_policy,
    _per_frame_sat_flags_for_catalog_id,
    _resolve_pfs_peak_test,
    apply_per_frame_saturation_to_active_targets,
    _fits_header_facts,
    _build_phase2a_resolved_facts,
    BlendMapEntry,
    _load_blend_worklist,
    _load_adaptive_blend_map,
    _route_lc_per_frame_err,
    _get_lc,
    _get_comp_bjd_series,
    compute_lc_flux_method,
    _recompute_bjd_hjd_with_status,
    run_sysrem_field,
)

import photometry_shared as _photometry_shared_e4c  # noqa: E402
_photometry_shared_e4c._coerce_bool_cell = _coerce_bool_cell

from photometry_gate_helpers import (  # noqa: E402,F401
    _sigma_bkg_r_key,
    _assert_inv_err_sigma_acct_01,
    comp_quality_quality_strings,
    _clamp_err_empty_apertures_n,
    _normalize_err_background_mode,
    _labbe_content_seed_from_header,
    measure_empty_aperture_sigma_bkg,
    estimate_star_free_per_pixel_variance_adu2,
    _howell_bkg_variance_adu2,
    _clamp_bkg_scale_r,
    bkg_scale_ratio_empirical_over_howell,
    compute_setup_bkg_scale_r,
    scaled_sigma_bkg_ap_from_howell,
    measure_growth_curve_ee,
    _phase2a_star_mag_lookup,
    discover_aligned_science_fits,
    _median_bkg_var_from_aligned_frames,
    _estimate_annulus_sky_pp,
    _annulus_sky_subtracted_flux,
    _resolve_star_flux_method,
    _frame_quality_gate_select,
    _recompute_bjd_hjd_per_target,
    photometer_check_star_production_path,
    _compute_fov_max_dist,
    _sky_pp_from_annulus_image,
    _aperture_flux_sky_per_star,
)

import photometry_shared as _photometry_shared  # noqa: E402
_photometry_shared._assert_inv_err_sigma_acct_01 = _assert_inv_err_sigma_acct_01
_photometry_shared._clamp_err_empty_apertures_n = _clamp_err_empty_apertures_n
_photometry_shared._labbe_content_seed_from_header = _labbe_content_seed_from_header
_photometry_shared._sigma_bkg_r_key = _sigma_bkg_r_key
_photometry_shared._sky_pp_from_annulus_image = _sky_pp_from_annulus_image
_photometry_shared.bkg_scale_ratio_empirical_over_howell = (
    bkg_scale_ratio_empirical_over_howell
)
_photometry_shared.compute_setup_bkg_scale_r = compute_setup_bkg_scale_r
_photometry_shared.measure_empty_aperture_sigma_bkg = measure_empty_aperture_sigma_bkg
_photometry_shared.scaled_sigma_bkg_ap_from_howell = scaled_sigma_bkg_ap_from_howell

from photometry_exports import (  # noqa: E402,F401
    lc_has_finite_airmass,
    apply_comp_w_rel_for_display,
    ensemble_member_ids,
    _get_lc_psf_strict,
    _get_lc_adaptive_per_star,
)

from epsf_hooks import load_epsf_metrics_for_draft  # noqa: E402,F401

from photometry_provenance import (  # noqa: E402,F401
    _is_import_relevant_py_path,
    _porcelain_status_by_path,
    classify_git_dirty_paths,
    _resolve_git_provenance,
    _json_safe_snapshot_value,
    _complete_config_snapshot,
    _build_pipeline_provenance_block,
    merge_photometry_pipeline_meta,
)

import photometry_provenance as _photometry_provenance  # noqa: E402
_photometry_provenance._resolve_git_provenance = (
    lambda *a, **k: _resolve_git_provenance(*a, **k)
)

from photometry_comp import (  # noqa: E402,F401
    _sid_int,
    _enrich_comp_bp_rp,
    _ensure_active_target_display_names,
    _variable_targets_looks_like_ct_presel_stub,
    ensure_full_variable_targets_if_presel_stub,
    _normalize_id_value,
    _normalize_id_series,
    _bool_col,
    _phase0_effective_frame_hw_px,
    _active_target_zone_flag,
    _auto_repair_catalog_ids,
    _enrich_active_targets_bp_rp,
    _resolve_frame_hw_px_from_masterstar,
    _read_field_density_inputs,
    _refresh_variable_targets_xy,
    _attach_predicted_dilution_report,
    select_active_targets,
    _batch_enrich_targets_bp_rp_from_gaia_db,
    _enrich_target_bp_rp_from_gaia_db,
    _bprp_tier_ladder_for_selection,
    _select_comps_by_rms_then_color,
    _select_comps_by_color_then_rms,
    _select_comps_tiered,
    build_global_comp_pool,
    _dedupe_comp_pool_by_gaia_key,
    _warn_zero_compstars_edge,
    _count_gate_passing_comps,
    select_comparison_stars_per_target,
    _write_suspected_variables,
)

import functools
import sys
import photometry_comp as _photometry_comp  # noqa: E402
_photometry_comp._enrich_active_targets_bp_rp = (
    lambda *a, **k: _enrich_active_targets_bp_rp(*a, **k)
)
_photometry_comp._ensure_active_target_display_names = (
    lambda *a, **k: _ensure_active_target_display_names(*a, **k)
)
_e3_select_active_targets = _photometry_comp.select_active_targets

@functools.wraps(_e3_select_active_targets)
def select_active_targets(*a, **k):
    out = _e3_select_active_targets(*a, **k)
    global LAST_EXCLUDED_TARGETS
    LAST_EXCLUDED_TARGETS = _photometry_comp.LAST_EXCLUDED_TARGETS
    _p01 = sys.modules.get("phase01_run")
    if _p01 is not None:
        _p01.LAST_EXCLUDED_TARGETS = LAST_EXCLUDED_TARGETS
    return out

_photometry_comp.select_active_targets = select_active_targets

from phase01_run import run_phase0_and_phase1  # noqa: E402,F401
