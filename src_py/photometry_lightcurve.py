"""Moved from photometry_core.py (CONSOLIDATE-01E4). Facade re-exports these names."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import logging
import math
import os
from astropy.io import fits as astrofits
import numpy as np
import pandas as pd
from catalog_match_trust import normalize_catalog_match_mode
from config import AppConfig
from infolog import log_event
from jd_axis_format import jd_axis_title, jd_series_relative
from photometry_shared import _normalize_gaia_id
from pipeline_constants import SAT_LIMIT_CONTAINER_CLIP_ADU

from photometry_core import (
    LOGGER,
    PFS_NEVER_RESCUE_REASONS,
    PFS_SATURATION_SKIP_REASONS,
    TIME_BASE_BJD_TDB,
    TIME_BASE_JD_FALLBACK,
    _ADAPTIVE_BLEND_CACHE,
    _CT_PROTOTYPE_CSV_FIELDS,
    _PSF_ERR_MAG_SCALE,
)


def _ac_summary_fields(ac_result: dict[str, Any] | None) -> dict[str, Any]:
    ac = ac_result if isinstance(ac_result, dict) else {}
    applied = bool(ac.get("ok"))
    fields: dict[str, Any] = {
        "ac_applied": applied,
        "ac_skip_reason": "" if applied else str(ac.get("reason", "") or ""),
    }
    if applied:
        for key, src in (
            ("ac_delta_m_corr", "delta_m_corr"),
            ("ac_scatter", "scatter_mag"),
        ):
            val = ac.get(src)
            try:
                fields[key] = float(val) if val is not None else float("nan")
            except (TypeError, ValueError):
                fields[key] = float("nan")
        fields["ac_n_ref"] = int(ac.get("n_ref_stars") or 0)
    return fields

def _phase2a_empty_comp_summary_row(
    *,
    target_cid: str,
    target_name: str,
    zone_flag: str,
) -> dict[str, Any]:
    """Summary stub when an active target has no per-target comparison stars."""
    return {
        "catalog_id": target_cid,
        "vsx_name": target_name,
        "zone_flag": zone_flag,
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
        "ac_applied": False,
        "ac_skip_reason": "no_comps",
    }

def _phase2a_skip_empty_comps_target(
    *,
    target_cid: str,
    target_name: str,
    zone_flag: str,
    summary_rows: list,
) -> list:
    """Record counted drop when an active target has no per-target comparison stars."""
    from except_fix_counters import get_except_fix_counters

    _ctr = get_except_fix_counters()
    _ctr.phase2a_empty_comp_drop += 1
    logging.error(
        "[FAZA 2A] Target %s (%s): ziadne comp hviezdy - preskocene "
        "(phase2a_empty_comp_drop=%d)",
        target_name,
        target_cid,
        _ctr.phase2a_empty_comp_drop,
    )
    summary_rows.append(
        _phase2a_empty_comp_summary_row(
            target_cid=target_cid,
            target_name=target_name,
            zone_flag=zone_flag,
        )
    )
    return summary_rows

def _coerce_bool_cell(v: Any) -> bool:
    """Robustly coerce a CSV cell (bool / 'True' / 1 / NaN / '') to bool; NaN/empty -> False."""
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    if isinstance(v, (int, float)):
        try:
            if math.isnan(float(v)):
                return False
        except (TypeError, ValueError):
            return False
        return float(v) != 0.0
    s = str(v).strip().lower()
    return s in ("true", "1", "yes", "y", "t")

def _frame_has_usable_cog(df: pd.DataFrame | None) -> bool:
    """True when a proc-frame CSV carries a usable COG correction (``cog_ok``).

    Per-frame ``fallback_ee`` wiring is a FUTURE refinement (APCORR-MIXEDFRAME);
    tonight a frame is usable only when ``cog_ok`` is True on at least one row.
    """
    if df is None or getattr(df, "empty", True):
        return False
    if "cog_ok" not in df.columns:
        return False
    try:
        series = df["cog_ok"]
    except Exception:  # noqa: BLE001
        return False
    for v in series.tolist():
        if _coerce_bool_cell(v):
            return True
    return False

def evaluate_cog_night_apcorr_gate(
    frame_dfs: list[pd.DataFrame | None] | tuple[pd.DataFrame | None, ...],
    *,
    enabled: bool,
) -> dict[str, Any]:
    """Night-level all-or-nothing COG aperture-correction gate.

    Design: ``APCORR-MIXEDFRAME-ALLORNOTHING`` (DECISIONS). When COG is enabled
    and any science frame of the night lacks a usable correction, COG application
    is disabled for the entire night so light curves cannot mix corrected and
    uncorrected frames.

    Returns a dict with:
      ``use_apcorr_flux`` - pass to :func:`read_flux_from_csv`
      ``cog_night_fallback`` - provenance flag for ``pipeline_meta``
      ``n_without_cog_ok``, ``n_frames`` - counts for the log line
    """
    if not enabled:
        return {
            "use_apcorr_flux": False,
            "cog_night_fallback": False,
            "n_without_cog_ok": 0,
            "n_frames": 0,
        }
    frames = list(frame_dfs)
    n_frames = len(frames)
    n_bad = sum(1 for df in frames if not _frame_has_usable_cog(df))
    if n_frames == 0 or n_bad > 0:
        return {
            "use_apcorr_flux": False,
            "cog_night_fallback": True,
            "n_without_cog_ok": int(n_bad),
            "n_frames": int(n_frames),
        }
    return {
        "use_apcorr_flux": True,
        "cog_night_fallback": False,
        "n_without_cog_ok": 0,
        "n_frames": int(n_frames),
    }

def temporal_bin_comp_lc(
    comp_lc: dict[str, np.ndarray],
    comp_quality: dict[str, dict],
    all_frames: pd.DataFrame,
    *,
    window: int = 0,
    enabled: bool = True,
) -> dict[str, np.ndarray]:
    """Optimized temporal binning of comparison star measurements (MNRAS 2023, 526, 3482).

    Reference: Broeg-Bischoff & Dreizler (2023) MNRAS 526, 3482-3489 -
    'Optimised temporal binning of comparison star measurements
    for differential photometry'

    Applies rolling-window median smoothing to comp star mag_inst series,
    reducing high-frequency shot noise before ensemble normalization.
    Target star is never touched - real variability is preserved.

    Args:
        comp_lc:       dict[catalog_id -> mag_inst array, length=n_frames]
        comp_quality:  from check_comparison_stability() - excluded comps skipped;
                       empty dict -> all comps in comp_lc are active
        all_frames:    concat DataFrame with 'catalog_id' + 'bjd' columns (reserved)
        window:        smoothing window in frames (0 = auto-optimize [3,5,7,9,11])
        enabled:       False -> return original comp_lc unchanged

    Returns:
        dict[catalog_id -> smoothed mag_inst array] (same keys, same length)
    """
    _ = all_frames  # frame-ordered LC; BJD-aware binning reserved for later
    if not enabled or len(comp_lc) < 3:
        return dict(comp_lc)

    if not comp_quality:
        active_cids = sorted(comp_lc.keys(), key=str)
    else:
        active_cids = sorted(
            (
                cid
                for cid, q in comp_quality.items()
                if q.get("quality") != "excluded" and cid in comp_lc
            ),
            key=str,
        )
    if len(active_cids) < 3:
        return dict(comp_lc)

    def _rolling_median(arr: np.ndarray, w: int) -> np.ndarray:
        """Rolling median with edge fill from original values."""
        if w < 3 or w > len(arr):
            return arr.copy()
        out = arr.copy()
        half = w // 2
        # Safe: guarded by window-size check above; no fix needed.
        for i in range(half, len(arr) - half):
            out[i] = np.nanmedian(arr[i - half : i + half + 1])
        return out

    def _comp_scatter(smoothed: dict[str, np.ndarray]) -> float:
        """Mean std of comp residuals vs ensemble median."""
        matrix = np.column_stack([smoothed[cid] for cid in active_cids])
        ensemble = np.nanmedian(matrix, axis=1)
        residuals = matrix - ensemble[:, np.newaxis]
        return float(np.nanmean(np.nanstd(residuals, axis=0, ddof=1)))

    # Adaptive window cap: more comps = smaller max window
    # (large ensemble already stable; over-smoothing degrades quality)
    n_active = len(active_cids)
    if n_active >= 8:
        max_window = 5
    elif n_active >= 5:
        max_window = 7
    else:
        max_window = 11

    if window == 0:
        candidate_windows = [w for w in (3, 5, 7, 9, 11) if w <= max_window]
        best_w = 1
        best_scatter = _comp_scatter({cid: comp_lc[cid] for cid in active_cids})
        for w in candidate_windows:
            candidate = {cid: _rolling_median(comp_lc[cid], w) for cid in active_cids}
            s = _comp_scatter(candidate)
            if s < best_scatter - 1e-7:
                best_scatter = s
                best_w = w
        opt_window = best_w
    else:
        opt_window = max(1, int(window))

    if opt_window < 3:
        return dict(comp_lc)

    result = dict(comp_lc)
    for cid in active_cids:
        result[cid] = _rolling_median(comp_lc[cid], opt_window)

    LOGGER.debug(
        "[ALG-3 TempBin] window=%d (max=%d), %d comps, %d frames",
        opt_window,
        max_window if window == 0 else window,
        len(active_cids),
        len(next(iter(comp_lc.values()))),
    )
    return result

def pytics_iterative_weights(
    comp_lc: dict[str, np.ndarray],
    comp_quality: dict[str, dict],
    comp_rms_map: dict[str, float],
    *,
    n_iter: int = 5,
    enabled: bool = True,
) -> dict[str, float]:
    """Iterative comp star intercalibration - PyTICS (RASTI 2026).

    Reference: Marconi et al. (2026) RASTI -
    'PyTICS: an iterative method for photometric light-curve
    intercalibration using comparison stars'

    Algorithm:
        1. Compute per-frame ZP = weighted median of comp_lc
           (weights = 1/rms^2, Broeg 2005 prior)
        2. Per-comp residuals = comp_lc[cid] - ZP_frame
        3. Per-comp scatter = std(residuals) -> updated rms_map
        4. Update weights -> repeat n_iter times
        5. Return refined comp_rms_map

    Only 'good' and 'suspect' comps participate - 'excluded' stay excluded.
    Returns original comp_rms_map unchanged if enabled=False or < 3 good comps.
    """
    if not enabled:
        return dict(comp_rms_map)

    # Only use non-excluded comps (canonical cid order - LABBE-DET / SEM determinism).
    active_cids = sorted(
        cid
        for cid, q in comp_quality.items()
        if q.get("quality") != "excluded" and cid in comp_lc
    )
    if len(active_cids) < 3:
        return dict(comp_rms_map)

    # Build matrix: rows=frames, cols=comps
    lc_matrix = np.column_stack([comp_lc[cid].astype(float) for cid in active_cids])
    n_frames, _n_comps = lc_matrix.shape

    # Initial weights from comp_rms_map (Broeg prior)
    rms_arr = np.array(
        [float(comp_rms_map.get(cid, 1.0)) for cid in active_cids],
        dtype=float,
    )
    rms_arr = np.where(rms_arr > 0, rms_arr, 1.0)
    initial_rms = rms_arr.copy()

    iteration = 0
    for iteration in range(n_iter):
        weights = 1.0 / (rms_arr**2)
        s = weights.sum()
        if not np.isfinite(s) or s <= 0:
            break
        weights /= s

        # Per-frame ZP = weighted mean across comps
        zp_per_frame = lc_matrix @ weights  # shape: (n_frames,)

        # Per-comp residuals vs ZP
        residuals = lc_matrix - zp_per_frame[:, np.newaxis]  # (n_frames, n_comps)

        # New per-comp RMS from residuals
        new_rms = np.std(residuals, axis=0, ddof=1)
        new_rms = np.where(new_rms > 1e-6, new_rms, rms_arr)

        # Convergence check
        delta = np.abs(new_rms - rms_arr)
        rms_arr = new_rms
        if np.max(delta) < 1e-6:
            LOGGER.debug("[ALG-5 PyTICS] converged at iteration %d", iteration + 1)
            break

    # Build updated map - only update active comps, keep excluded untouched
    updated_map = dict(comp_rms_map)
    for cid, new_rms_val in zip(active_cids, rms_arr, strict=True):
        updated_map[cid] = float(new_rms_val)

    LOGGER.debug(
        "[ALG-5 PyTICS] %d comps, %d frames, %d iter -> max_Deltarms=%.6f",
        len(active_cids),
        n_frames,
        iteration + 1,
        float(np.max(np.abs(rms_arr - initial_rms))),
    )
    return updated_map

def _common_mode_detrend_comp_lc(
    comp_lc: dict[str, np.ndarray],
    comp_bjd: dict[str, np.ndarray] | None,
    *,
    min_frames: int = 20,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Remove shared linear night trend from comp LCs (differential-photometry residual).

    Returns (detrended_lc, detrended_bjd). When detrend is impossible, returns shallow copies.
    """
    if comp_bjd is None or not comp_lc:
        return (
            {
                cid: np.asarray(comp_lc[cid], dtype=np.float64).copy()
                for cid in sorted(comp_lc.keys(), key=str)
            },
            dict(comp_bjd or {}),
        )

    from scipy.stats import linregress as _lr

    _all_bjd: list[np.ndarray] = []
    _all_mag_matrix: list[np.ndarray] = []
    _active_cids: list[str] = []
    for cid in sorted(comp_lc.keys(), key=str):
        lc = comp_lc[cid]
        bjd_arr = comp_bjd.get(cid)
        if bjd_arr is None:
            continue
        m = np.asarray(lc, dtype=np.float64)
        b = np.asarray(bjd_arr, dtype=np.float64)
        ok = np.isfinite(b) & np.isfinite(m)
        if int(ok.sum()) < int(min_frames):
            continue
        bo, mo = b[ok], m[ok]
        order = np.argsort(bo, kind="mergesort")
        _all_bjd.append(bo[order])
        _all_mag_matrix.append(mo[order])
        _active_cids.append(cid)

    if len(_all_mag_matrix) < 2:
        return (
            {
                cid: np.asarray(comp_lc[cid], dtype=np.float64).copy()
                for cid in sorted(comp_lc.keys(), key=str)
            },
            dict(comp_bjd),
        )

    # Prefer longest series; break ties by cid order (already sorted) for determinism.
    _ref_idx = int(np.argmax([len(x) for x in _all_bjd]))
    _ref_bjd = _all_bjd[_ref_idx]
    _stack = []
    for b_arr, m_arr in zip(_all_bjd, _all_mag_matrix, strict=True):
        _stack.append(np.interp(_ref_bjd, b_arr, m_arr))
    _common = np.median(np.vstack(_stack), axis=0)
    _lr_common = _lr(_ref_bjd, _common)

    _detrended_lc: dict[str, np.ndarray] = {}
    _detrended_bjd: dict[str, np.ndarray] = {}
    for cid in sorted(comp_lc.keys(), key=str):
        b = comp_bjd.get(cid)
        m = comp_lc.get(cid)
        if b is None or m is None:
            continue
        b = np.asarray(b, dtype=np.float64)
        m = np.asarray(m, dtype=np.float64)
        ok = np.isfinite(b) & np.isfinite(m)
        m_detrended = m.copy()
        m_detrended[ok] = m[ok] - (_lr_common.slope * b[ok] + _lr_common.intercept) + float(_common.mean())
        _detrended_lc[cid] = m_detrended
        _detrended_bjd[cid] = b

    return _detrended_lc, _detrended_bjd

def _comp_lc_frame_ensemble_residual(comp_lc: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per-frame ensemble-median subtraction (differential residual per comp).

    Same principle as Phase-1 ``comp_rms`` (flux / bin median): comp intrinsic scatter
    is only visible after the shared per-frame signal is removed.

    Uses sorted cid order and truncates to a common length only after sorting keys so
    PYTHONHASHSEED / dict insertion order cannot change the residual field.
    """
    if not comp_lc:
        return {}
    cids = sorted(comp_lc.keys(), key=str)
    arrays = [np.asarray(comp_lc[c], dtype=np.float64) for c in cids]
    lengths = [int(a.size) for a in arrays if a.size > 0]
    if not lengths:
        return {c: np.array([], dtype=np.float64) for c in cids}
    n = int(min(lengths))
    if n < 3:
        return {c: a[:n].copy() for c, a in zip(cids, arrays, strict=True)}
    stack = np.column_stack([a[:n] for a in arrays])
    med = np.nanmedian(stack, axis=1)
    out: dict[str, np.ndarray] = {}
    for j, cid in enumerate(cids):
        out[cid] = stack[:, j] - med
    return out

def compute_lc_rms_ooe(
    mag_calib: np.ndarray,
    flags: Sequence[str],
    *,
    brightest_frac: float = 0.33,
) -> float:
    """Out-of-eclipse / brightest-tertile scatter (precision metric for variables).

    For eclipsing variables the faintest frames are in eclipse; the brightest ``brightest_frac``
  fraction approximates OOE scatter without requiring a period.
    """
    m = np.asarray(mag_calib, dtype=np.float64)
    if m.size < 3:
        return float("nan")
    fl = [str(f).strip().lower() for f in flags]
    if len(fl) == m.size:
        ok = np.array([f == "normal" for f in fl], dtype=bool)
    else:
        ok = np.ones(m.size, dtype=bool)
    finite = m[ok & np.isfinite(m)]
    if finite.size < 3:
        finite = m[np.isfinite(m)]
    if finite.size < 3:
        return float("nan")
    frac = float(brightest_frac)
    if not (0.0 < frac < 1.0):
        frac = 0.33
    thr = float(np.quantile(finite, frac))
    ooe = finite[finite <= thr]
    if ooe.size < 3:
        ooe = finite
    return float(np.std(ooe))

def check_comparison_stability(
    comp_lc: dict[str, np.ndarray],
    *,
    comp_rms_map: dict[str, float] | None = None,
    comp_bjd: dict[str, np.ndarray] | None = None,
    n_comp_min: int = 3,
    outlier_sigma: float = 3.0,
    max_comp_p2p: float = 0.1,
    max_comp_slope_mmag_hr: float = 5.0,
    comp_slope_significance_k: float = 3.0,
    common_mode_detrend: bool = True,
    stability_run_flags: dict[str, Any] | None = None,
) -> dict[str, dict]:
    """Krok 3: Stability check porovnavaciek.

    Abbeho point-to-point scatter on **common-mode-detrended** comp residuals:
        rms_p2p = std(diff(mag_resid)) / sqrt(2)

    ``max_comp_p2p`` (config ``phase01_comparison_max_comp_rms``) gates **p2p** scatter.
    Phase-1 ``comp_rms`` uses the same config key but measures full LC RMS - different
    metric, shared ceiling knob (see ``select_comparison_stars_per_target``).

    Shared atmospheric drift is removed before p2p/MAD (same differential logic as ensemble).

    Returns:
        dict {catalog_id: {"rms_p2p": float, "lc_rms": float, "quality": str, "p2p_threshold": float}}
        quality: "good" / "suspect" / "excluded"; zaznamy su zoradene (good -> suspect -> excluded, v ramci good podla rms_p2p).
    """
    result: dict[str, dict[str, Any]] = {}

    _detrended_lc: dict[str, np.ndarray] = {}
    _detrended_bjd: dict[str, np.ndarray] = {}
    _residual_lc = _comp_lc_frame_ensemble_residual(comp_lc)
    if common_mode_detrend and comp_bjd is not None:
        _detrended_lc, _detrended_bjd = _common_mode_detrend_comp_lc(_residual_lc, comp_bjd)
        if stability_run_flags is not None and len(_detrended_lc) >= 2:
            stability_run_flags["common_mode_detrend_applied"] = True
            stability_run_flags["frame_ensemble_residual"] = True
            logging.info(
                "[STABILITY] Frame-ensemble residual + CM detrend before p2p: %d comps",
                len(_detrended_lc),
            )
    else:
        _detrended_lc = {cid: np.asarray(lc, dtype=np.float64).copy() for cid, lc in _residual_lc.items()}
        _detrended_bjd = dict(comp_bjd or {})
        if stability_run_flags is not None:
            stability_run_flags["frame_ensemble_residual"] = True

    # Vypocitaj metriky na per-frame differential rezidualnych radach
    for cid, lc in comp_lc.items():
        resid = _detrended_lc.get(cid, lc)
        finite = np.asarray(resid, dtype=np.float64)[np.isfinite(resid)]
        if len(finite) < 3:
            result[cid] = {
                "rms_p2p": float("nan"),
                "lc_rms": float("nan"),
                "quality": "excluded",
                "note": "few_frames",
            }
            continue
        lc_rms = float(np.std(finite))
        diff = np.diff(finite)
        rms_p2p = float(np.std(diff) / math.sqrt(2)) if len(diff) > 1 else float("nan")
        result[cid] = {"rms_p2p": rms_p2p, "lc_rms": lc_rms, "quality": "good"}

    # Ak hviezda ma comp_rms~0 z Phase 1 -> oznac ako suspect (pravdepodobny isolated-bin normalizacny artefakt).
    if comp_rms_map:
        for cid in result:
            try:
                phase1_rms = float(comp_rms_map.get(cid, float("nan")))
            except Exception:  # noqa: BLE001
                phase1_rms = float("nan")
            if math.isfinite(phase1_rms) and phase1_rms < 1e-6:
                result[cid]["quality"] = "suspect"
                result[cid]["note"] = "isolated_bin"

    # COMP-ADMIT-03: p2p is diagnostic only - never eject members (weight via sigma_rms).
    _ = outlier_sigma
    threshold = float(max_comp_p2p) if math.isfinite(float(max_comp_p2p)) and float(max_comp_p2p) > 0 else 0.1
    for cid, info in result.items():
        if not math.isfinite(info["rms_p2p"]):
            continue
        if info["rms_p2p"] > threshold:
            result[cid]["quality"] = "suspect"
            result[cid]["note"] = (
                f"p2p_high (p2p={info['rms_p2p']:.4f} > thr={threshold:.4f}; kept COMP-ADMIT-03)"
            )

    # Slope filter: exclude comps with a night-long linear trend (slow drifts pass p2p RMS).
    if comp_bjd is not None and max_comp_slope_mmag_hr > 0:
        from scipy.stats import linregress  # lazy import - scipy already in deps

        n_good_slope = sum(1 for v in result.values() if v["quality"] == "good")
        for cid, info in result.items():
            if info["quality"] == "excluded":
                continue
            bjd_arr = _detrended_bjd.get(cid) if _detrended_bjd else comp_bjd.get(cid)
            mag_arr = _detrended_lc.get(cid) if _detrended_lc else comp_lc.get(cid)
            if bjd_arr is None or mag_arr is None:
                continue
            ok = np.isfinite(bjd_arr) & np.isfinite(mag_arr)
            if int(ok.sum()) < 20:
                continue
            lr = linregress(bjd_arr[ok], mag_arr[ok])
            slope_mmag_hr = abs(float(lr.slope)) * 1000.0 / 24.0
            _se = float(getattr(lr, "stderr", float("nan")))
            slope_sig = (
                abs(float(lr.slope)) / _se
                if math.isfinite(_se) and _se > 0
                else float("inf")
            )
            if slope_mmag_hr > float(max_comp_slope_mmag_hr) and slope_sig >= float(
                comp_slope_significance_k
            ):
                logging.info(
                    "Comp %s slope-high (kept COMP-ADMIT-03): %.1f mmag/hr (%.1fsigma) > %s mmag/hr @ %ssigma",
                    cid,
                    slope_mmag_hr,
                    slope_sig,
                    max_comp_slope_mmag_hr,
                    comp_slope_significance_k,
                )
                info["slope_mmag_hr"] = slope_mmag_hr
                info["slope_sigma"] = slope_sig
                info["quality"] = "suspect"
                note = f"slope={slope_mmag_hr:.1f} mmag/hr ({slope_sig:.1f}sigma; kept COMP-ADMIT-03)"
                if info.get("note"):
                    info["note"] = f"{info['note']}; {note}"
                else:
                    info["note"] = note

    for info in result.values():
        info["p2p_threshold"] = threshold

    n_good_final = sum(1 for v in result.values() if v["quality"] == "good")
    thr_log = f"{threshold:.5f}" if math.isfinite(threshold) else "N/A"
    logging.info(
        f"[FAZA 2A] Stability check: {n_good_final}/{len(result)} good comp "
        f"(p2p threshold={thr_log})"
    )

    # Zoradenie: good (podla rms_p2p), suspect, excluded - poradie v ensemble / PNG tabulke
    sorted_result = dict(
        sorted(
            result.items(),
            key=lambda x: (
                0 if x[1]["quality"] == "good" else 1 if x[1]["quality"] == "suspect" else 2,
                x[1]["rms_p2p"] if math.isfinite(x[1].get("rms_p2p", float("nan"))) else 999.0,
            ),
        )
    )
    return sorted_result

def ensemble_normalize(
    target_mag_inst: np.ndarray,
    comp_mag_inst: dict[str, np.ndarray],
    comp_catalog_mag: dict[str, float],
    comp_quality: dict[str, dict],
    *,
    comp_rms_map: dict[str, float] | None = None,
    comp_tier_map: dict[str, int] | None = None,
    tier_weights: dict[int, float] | None = None,
    comp_weight_map: dict[str, float] | None = None,
    comp_likely_saturated: dict[str, np.ndarray] | None = None,
    n_comp_min: int = 3,
    n_comp_max: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Krok 4: Ensemble normalizacia per snimka.

    ``mag_ensemble`` = ``-2.5*log10(sum 10**(-0.4*m_comp))`` (sucet fluxov ako AIJ ``tot_C_cnts``).

    ``delta_mag = mag_inst(target) - mag_ensemble`` (tvar voci suctu fluxov ako AIJ).

    Zeropoint uses continuous weights ``w=1/sigma_eff^2`` when
    ``comp_weight_map`` is supplied; otherwise ``1/rms^2``. No tier multiplier, no
    per-frame rejection, no membership re-selection (COMP-ASSIGN-01 /
    INV-COMP-MEMBERSHIP). ``n_comp_min`` / ``n_comp_max`` /
    ``comp_tier_map`` / ``tier_weights`` are call-site compatibility stubs -
    membership is decided in Phase 1 step 2 and consumed as given.

    FORCED-PHOT-01 / per-frame saturation: optional ``comp_likely_saturated`` maps
    catalog_id -> bool array (one per frame). Saturated epochs are **kept** in the
    series but excluded from that frame's zeropoint flux sum (measurability: the
    measurement is invalid that frame, not imprecise). This is explicit, not silent.
    Membership ``good_ids`` does not change.

    Returns:
        (mag_calib, delta_mag, ensemble_scatter) - arrays dlzky n_frames
    """
    _ = (comp_tier_map, tier_weights, n_comp_min, n_comp_max)
    n_frames = len(target_mag_inst)
    mag_calib = np.full(n_frames, float("nan"))
    delta_mag = np.full(n_frames, float("nan"))
    ensemble_scatter = np.full(n_frames, float("nan"))

    comp_rms_map = comp_rms_map or {}
    comp_weight_map = comp_weight_map or {}
    comp_likely_saturated = comp_likely_saturated or {}

    # Full admitted membership (COMP-ADMIT-03). Quality annotations must not eject members.
    # Keep ``good_ids = selected`` token for INV-COMP-MEMBERSHIP static scan.
    selected = sorted(comp_quality.keys(), key=str) if comp_quality else sorted(comp_mag_inst.keys(), key=str)
    good_ids = selected
    if not good_ids:
        log_event("ensemble_normalize: no valid comp stars - returning all-NaN LC")
        return mag_calib, delta_mag, ensemble_scatter

    cat_mags = np.asarray([comp_catalog_mag.get(cid, float("nan")) for cid in good_ids])
    cat_offset = float(np.nanmedian(cat_mags))
    logging.debug(
        f"[FAZA 2A] Ensemble: {len(good_ids)} comps (COMP-ADMIT-03 full membership), "
        f"catalog_mag median={cat_offset:.3f}"
    )

    # Per-comp across-night reference = median of its own instrumental magnitudes. Referencing each
    # comp to its OWN reference (rather than comparing comps' absolute instrumental mags) cancels the
    # comps' brightness AND constant colour offsets, so the per-frame ``ensemble_scatter`` below is the
    # genuine zeropoint scatter (Honeycutt 1992), not the comps' brightness spread. Used ONLY for the
    # error model; does not touch mag_calib/delta_mag/ens_med.
    comp_ref_map: dict[str, float] = {}
    for cid in good_ids:
        arr_cid = comp_mag_inst.get(cid)
        if arr_cid is None:
            continue
        _vals = np.asarray(arr_cid, dtype=np.float64)
        _fin = _vals[np.isfinite(_vals)]
        if _fin.size:
            comp_ref_map[cid] = float(np.median(_fin))

    n_sat_excluded = 0
    for i in range(n_frames):
        comp_pairs: list[tuple[str, float]] = []
        for cid in good_ids:
            if cid not in comp_mag_inst:
                continue
            try:
                mv = float(comp_mag_inst[cid][i])
            except Exception as exc:  # noqa: BLE001
                logging.error("[EXC-0135] One comp's inst mag on one frame skipped - ensemble normalization ignores that comp for...: %s", exc)
                continue
            # FORCED-PHOT-01: per-frame sat -> keep row elsewhere; exclude from ZP this frame.
            sat_arr = comp_likely_saturated.get(cid)
            if sat_arr is not None:
                try:
                    if bool(sat_arr[i]):
                        n_sat_excluded += 1
                        continue
                except Exception:  # noqa: BLE001
                    pass
            if math.isfinite(mv):
                comp_pairs.append((cid, mv))

        if (not comp_pairs) or not math.isfinite(target_mag_inst[i]):
            continue

        comp_vals = np.asarray([m for _, m in comp_pairs], dtype=np.float64)

        # Combination = AIJ/Honeycutt flux sum (tot_C_cnts); Broeg 1/rms^2 applies to selection
        # ordering + catalog zeropoint offset below, not to ens_med.
        # Priamy sucet fluxov - rovnaka metoda ako AIJ (tot_C_cnts = C2+C3+C4).
        # Vahovany priemer 1/rms^2 deformuje extinkcny slope ensemble -> zaporny slope.
        comp_fluxes_list: list[float] = [10 ** (-0.4 * m) for _, m in comp_pairs]
        f_arr = np.asarray(comp_fluxes_list, dtype=np.float64)

        ens_flux_sum = float(np.sum(f_arr))
        if math.isfinite(ens_flux_sum) and ens_flux_sum > 0:
            ens_med = float(-2.5 * math.log10(ens_flux_sum))
        else:
            ens_med = float(np.median(comp_vals))

        # Per-point ensemble zeropoint uncertainty (Honeycutt 1992 PASP 104:435): the standard error
        # of the comps' per-frame residuals about the ensemble mean, where each residual is the comp's
        # deviation from its OWN across-night reference (``comp_ref_map``). This cancels the comps'
        # brightness/colour spread - the previous ``np.std(comp_vals)`` on raw instrumental mags
        # injected a fixed ~comp-brightness-difference floor (the inflated-err bug). Small n: a near-
        # zero residual SEM leaves err = photon base (the floor); we do not use comp_rms here (it is
        # dropped at the err-assembly site to avoid double-counting this same ensemble term).
        comp_resid = [
            (m - comp_ref_map[cid_j])
            for cid_j, m in comp_pairs
            if cid_j in comp_ref_map and math.isfinite(comp_ref_map[cid_j])
        ]
        if len(comp_resid) >= 2:
            from sigma_floor_core import (  # noqa: PLC0415
                ensemble_sem_mag_from_residuals,
                ensemble_sem_mag_from_residuals_weighted,
            )

            if comp_weight_map:
                w_resid = []
                x_resid = []
                for cid_j, m in comp_pairs:
                    if cid_j not in comp_ref_map or not math.isfinite(comp_ref_map[cid_j]):
                        continue
                    wj = float(comp_weight_map.get(cid_j, float("nan")))
                    if not math.isfinite(wj) or wj <= 0:
                        # fall back to 1/rms^2 if weight missing
                        rms = float(comp_rms_map.get(cid_j, float("nan")))
                        wj = (1.0 / (rms * rms)) if math.isfinite(rms) and rms > 0 else 1.0
                    x_resid.append(m - comp_ref_map[cid_j])
                    w_resid.append(wj)
                ensemble_scatter[i] = float(
                    ensemble_sem_mag_from_residuals_weighted(x_resid, w_resid)
                )
            else:
                ensemble_scatter[i] = float(ensemble_sem_mag_from_residuals(comp_resid))
        else:
            ensemble_scatter[i] = 0.0
        delta_mag[i] = target_mag_inst[i] - ens_med

        # Honeycutt (1992) PASP 104:435 - per-frame ensemble zeropoint from constant comps.
        # mag_calib[i] = target_inst + ZP_frame; delta_mag[i] = target_inst - ens_med (AIJ flux sum).
        # Hence mag_calib - delta_mag = ZP_frame + ens_med (frame-dependent; not identical zeropoints).
        # ``delta_mag + median(cat)`` by bolo nesuladne s ``ens_med`` zo suctu fluxov (-2.5 log SigmaF).
        zp_offs: list[float] = []
        zp_vals: list[float] = []
        weights: list[float] = []
        for cid_j, m_j in comp_pairs:
            cm_j = float(comp_catalog_mag.get(cid_j, float("nan")))
            if math.isfinite(cm_j) and math.isfinite(m_j):
                d = float(cm_j - m_j)
                zp_offs.append(d)
                w_j = float(comp_weight_map.get(cid_j, float("nan")))
                if not (math.isfinite(w_j) and w_j > 0):
                    rms_j = float(comp_rms_map.get(cid_j, float("nan")))
                    if math.isfinite(rms_j) and rms_j > 1e-6:
                        # Broeg 2005 1/sigma^2; COMP-ADMIT-03 drops the tier multiplier.
                        w_j = 1.0 / (rms_j**2)
                    else:
                        w_j = float("nan")
                if math.isfinite(w_j) and w_j > 0:
                    zp_vals.append(d)
                    weights.append(float(w_j))
        if weights:
            # Continuous weights over ALL admitted comps - no per-frame rejection
            # (INV-COMP-MEMBERSHIP; COMP-ADMIT-03; ZP MAD clip removed 2026-08-12).
            w = np.asarray(weights, dtype=np.float64)
            z = np.asarray(zp_vals, dtype=np.float64)
            if len(z) >= 2 and float(np.sum(w)) > 0:
                mag_calib[i] = target_mag_inst[i] + float(np.sum(w * z) / np.sum(w))
            elif zp_offs:
                mag_calib[i] = target_mag_inst[i] + float(
                    np.nanmedian(np.asarray(zp_offs, dtype=np.float64))
                )
            else:
                mag_calib[i] = delta_mag[i] + cat_offset
        elif zp_offs:
            mag_calib[i] = target_mag_inst[i] + float(np.nanmedian(np.asarray(zp_offs, dtype=np.float64)))
        else:
            mag_calib[i] = delta_mag[i] + cat_offset

    if n_sat_excluded > 0:
        logging.info(
            "[FORCED-PHOT] ensemble_normalize: excluded %d per-frame saturated "
            "comp epochs from ZP (rows kept; membership unchanged)",
            int(n_sat_excluded),
        )

    return mag_calib, delta_mag, ensemble_scatter

def _ensemble_scatter_by_source_file(
    all_frames: pd.DataFrame,
    target_cid: str,
    ensemble_scatter: np.ndarray | None,
) -> dict[str, float]:
    """Map proc CSV filename -> ensemble_scatter (G2-F004 epoch-level join key).

    Must use the same ``source_file`` sort as ``_get_lc`` so indices of
    ``ensemble_scatter`` (built from sorted target LC) map to the correct files.
    """
    if ensemble_scatter is None:
        return {}
    sc = np.asarray(ensemble_scatter, dtype=np.float64)
    if sc.size == 0:
        return {}
    sub = all_frames[all_frames["catalog_id"] == target_cid]
    if sub.empty:
        return {}
    if "source_file" in sub.columns:
        sub = sub.sort_values(["source_file"], kind="mergesort")
    out: dict[str, float] = {}
    for i, sf in enumerate(sub["source_file"].astype(str).str.strip().tolist()):
        if i >= len(sc):
            break
        key = str(sf)
        if key:
            out[key] = float(sc[i])
    return out

def _combine_err_with_ensemble_scatter_keyed(
    err_photon: np.ndarray,
    source_files: list[str] | np.ndarray,
    scatter_by_file: dict[str, float],
    *,
    sigma_sys_mag: float = 0.0,
    sigma_scint_mag: np.ndarray | list[float] | None = None,
    target_name: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """Join photon ``err`` with ensemble scatter by EXACT ``source_file`` match (G2-F004).

    Domain contract: ``err_photon`` is relative flux (err/flux); ``scatter_by_file`` values are
    ensemble SEM in magnitudes (Honeycutt residual std/sqrt(n) with c4 correction from
    ``ensemble_normalize``). Per-rig ``sigma_sys_mag`` and per-epoch ``sigma_scint_mag`` (mag)
    are added in quadrature after SEM.

    err_total^2 = err_photon^2 + sem_rel^2 + scint_rel^2 + sigma_sys_rel^2 (relative-flux domain).

    Matched epoch, finite scatter -> quadrature with SEM (+ scint + floor when configured).
    Matched epoch, NaN scatter -> scatter treated as 0.0 (photon-only + scint + floor).
    Unmatched ``source_file`` -> **NaN err**, ``err_scatter_unmatched`` True (I-04: exclude epoch).
    """
    from sigma_floor_core import combine_production_err_rel  # noqa: PLC0415

    err_out = np.asarray(err_photon, dtype=np.float64).copy()
    unmatched = np.zeros(len(err_out), dtype=bool)
    _floor = float(sigma_sys_mag) if math.isfinite(float(sigma_sys_mag)) and float(sigma_sys_mag) > 0 else 0.0
    _scint_arr: np.ndarray | None = None
    if sigma_scint_mag is not None:
        _scint_arr = np.asarray(sigma_scint_mag, dtype=np.float64)
        if len(_scint_arr) != len(err_out):
            _scint_arr = None

    n_unmatched = 0
    for i, sf in enumerate(np.asarray(source_files, dtype=object)):
        key = str(sf).strip()
        sc_mag = 0.0
        if key and scatter_by_file and key in scatter_by_file:
            sc = float(scatter_by_file[key])
            sc_mag = float(sc) if math.isfinite(sc) else 0.0
        elif scatter_by_file:
            unmatched[i] = True
            n_unmatched += 1
            err_out[i] = float("nan")
            continue
        ep = float(err_out[i]) if math.isfinite(err_out[i]) else float("nan")
        scint_m = 0.0
        if _scint_arr is not None:
            v = float(_scint_arr[i])
            scint_m = v if math.isfinite(v) and v > 0 else 0.0
        err_out[i] = combine_production_err_rel(
            ep, sc_mag, sigma_sys_mag=_floor, sigma_scint_mag=scint_m,
        )

    if n_unmatched > 0:
        logging.warning(
            "[G2-F004] %s: %d/%d epochs missing ensemble_scatter for source_file "
            "- err set NaN, epoch excluded (I-04)",
            target_name or "?",
            n_unmatched,
            len(err_out),
        )
    return err_out, unmatched

def _err_budget_components_keyed(
    err_photon: np.ndarray,
    source_files: list[str] | np.ndarray,
    scatter_by_file: dict[str, float],
    *,
    sigma_sys_mag: float = 0.0,
    sigma_scint_mag: np.ndarray | list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-epoch error budget terms in relative-flux domain (export / diagnostics)."""
    from sigma_floor_core import mag_sigma_to_rel  # noqa: PLC0415

    ep = np.asarray(err_photon, dtype=np.float64)
    n = len(ep)
    sem_rel = np.zeros(n, dtype=np.float64)
    scint_rel = np.zeros(n, dtype=np.float64)
    sys_rel = np.full(n, mag_sigma_to_rel(float(sigma_sys_mag)), dtype=np.float64)
    _scint_arr: np.ndarray | None = None
    if sigma_scint_mag is not None:
        _scint_arr = np.asarray(sigma_scint_mag, dtype=np.float64)
        if len(_scint_arr) != n:
            _scint_arr = None
    for i, sf in enumerate(np.asarray(source_files, dtype=object)):
        key = str(sf).strip()
        if scatter_by_file and key in scatter_by_file:
            sc = float(scatter_by_file[key])
            if math.isfinite(sc):
                sem_rel[i] = mag_sigma_to_rel(sc)
        if _scint_arr is not None:
            v = float(_scint_arr[i])
            if math.isfinite(v) and v > 0:
                scint_rel[i] = mag_sigma_to_rel(v)
    return ep, sem_rel, scint_rel, sys_rel

def _exclude_err_scatter_unmatched_epochs(
    err_scatter_unmatched: np.ndarray,
    *arrays: np.ndarray | list,
) -> tuple[np.ndarray, ...]:
    """Drop epochs flagged ``err_scatter_unmatched`` (I-04 export exclusion)."""
    mask = ~np.asarray(err_scatter_unmatched, dtype=bool)
    out: list[Any] = [mask]
    for arr in arrays:
        a = np.asarray(arr)
        if len(a) == len(mask):
            out.append(a[mask])
        else:
            out.append(arr)
    return tuple(out)

def ct_ensemble_reference_maps(
    ensemble_bp_rp: dict[str, float],
    ensemble_quality: dict[str, dict],
) -> tuple[dict[str, float], dict[str, dict]]:
    """Per-target CT colour-ref membership is the ZP ensemble, never ``comparison_stars.csv``.

    ``c1`` may still come from a field-level fit; ``ct_bp_rp_comp_med`` and the
    weighted colour reference must use the same stars (and weights) as the
    ensemble zero-point. An export-pool subset of the ensemble is not a valid
    reference.
    """
    bp = dict(ensemble_bp_rp or {})
    q = dict(ensemble_quality or {})
    return bp, q

def apply_color_term(
    mag_calib: np.ndarray,
    target_bp_rp: float,
    comp_bp_rp: dict[str, float],
    comp_quality: dict[str, dict],
    c1: float,
    *,
    comp_weights: dict[str, float] | None = None,
) -> tuple[np.ndarray, float, float]:
    """Apply a constant colour correction to the calibrated light curve.

    Caller must pass the ZP ensemble maps (``ct_ensemble_reference_maps``), not
    the ``comparison_stars.csv`` export pool.

    Formula (existing path, removes PRE-IMPL-01 level bias):
      bp_rp_comp_ref = weighted_mean(bp_rp) when weights given, else median
      ct_correction  = c1 * (target_bp_rp - bp_rp_comp_ref)
      mag_calib_ct   = mag_calib + ct_correction

    The correction is a **per-target / per-draft constant** (no airmass / epoch
    dependence). PRE-IMPL-01 measured level ~ k * (ens - target); adding
    ``c1 * (target - ens)`` with ``c1 = k_level`` removes that bias.

    Returns: (mag_calib_ct, ct_correction, bp_rp_comp_ref)
    """
    if mag_calib is None:
        return np.asarray([], dtype=np.float64), 0.0, float("nan")
    base = np.asarray(mag_calib, dtype=np.float64)
    if (not math.isfinite(float(c1))) or float(c1) == 0.0 or (not math.isfinite(float(target_bp_rp))):
        return base.copy(), 0.0, float("nan")

    usable = [cid for cid, q in comp_quality.items() if q.get("quality") in ("good", "suspect")]
    bps: list[float] = []
    wts: list[float] = []
    for cid in usable:
        bp = float(comp_bp_rp.get(cid, float("nan")))
        if not math.isfinite(bp):
            continue
        if comp_weights is not None:
            w = float(comp_weights.get(cid, float("nan")))
            if not (math.isfinite(w) and w > 0):
                continue
            bps.append(bp)
            wts.append(w)
        else:
            bps.append(bp)
            wts.append(1.0)
    if not bps:
        return base.copy(), 0.0, float("nan")
    bp_arr = np.asarray(bps, dtype=np.float64)
    wt_arr = np.asarray(wts, dtype=np.float64)
    if comp_weights is not None and float(np.sum(wt_arr)) > 0:
        bp_ref = float(np.sum(wt_arr * bp_arr) / np.sum(wt_arr))
        ref_label = "comp_wmean"
    else:
        bp_ref = float(np.median(bp_arr))
        ref_label = "comp_med"
    corr = float(c1) * (float(target_bp_rp) - float(bp_ref))
    out = base + float(corr)
    logging.info(
        "[COLOR TERM] target bp_rp=%.3f, %s bp_rp=%.3f, correction=%+.4f mag (constant)",
        float(target_bp_rp),
        ref_label,
        float(bp_ref),
        float(corr),
    )
    return out, float(corr), float(bp_ref)

def _check_color_term_extrapolation(
    target_bp_rp: float,
    comp_bp_rp_values: list[float],
    target_name: str = "",
    *,
    extrapolation_tol: float = 0.0,
) -> bool:
    """Return True when target BP-RP is within the comp BP-RP range (+- ``extrapolation_tol``).

    Return False when outside range -> caller must skip CT (target kept, uncorrected).
    """
    finite_vals: list[float] = []
    for v in comp_bp_rp_values:
        try:
            vf = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(vf):
            finite_vals.append(vf)
    try:
        tgt = float(target_bp_rp)
    except (TypeError, ValueError):
        return True
    if len(finite_vals) < 2 or not math.isfinite(tgt):
        return True
    try:
        tol = max(0.0, float(extrapolation_tol))
    except (TypeError, ValueError):
        tol = 0.0
    min_bprp = float(min(finite_vals))
    max_bprp = float(max(finite_vals))
    in_range = (min_bprp - tol) <= tgt <= (max_bprp + tol)
    if not in_range:
        logging.warning(
            "[COLOR TERM] Target %s BP-RP=%.3f je mimo rozsahu comp [%.3f, %.3f] - color term extrapolacia moze viest k systematike.",
            str(target_name or ""),
            float(tgt),
            float(min_bprp),
            float(max_bprp),
        )
    return in_range

def _ct_prototype_enabled() -> bool:
    return os.environ.get("VYVAR_CT_PROTOTYPE", "").strip() == "1"

def _color_term_cat_inst_scatter_pair(
    comp_mag_inst: dict[str, np.ndarray],
    comp_catalog_mag: dict[str, float],
    comp_bp_rp: dict[str, float],
    comp_quality: dict[str, dict],
    c1: float,
    *,
    min_comp: int = 5,
    sigma_clip_sigma: float = 3.0,
) -> tuple[float, float]:
    """Per-comp cat-inst scatter before/after removing the fitted c1.Delta(bp_rp) trend."""
    _ = sigma_clip_sigma  # no residual clip (zero-clipping 2026-08-12)
    try:
        min_comp_i = int(min_comp)
    except Exception:  # noqa: BLE001
        min_comp_i = 5
    min_comp_i = max(2, min_comp_i)

    usable = [cid for cid, q in comp_quality.items() if q.get("quality") in ("good", "suspect")]
    ys: list[float] = []
    bp_vals: list[float] = []

    for cid in usable:
        bp = float(comp_bp_rp.get(cid, float("nan")))
        if not math.isfinite(bp):
            continue
        if cid not in comp_mag_inst:
            continue
        inst = np.asarray(comp_mag_inst[cid], dtype=np.float64)
        finite = inst[np.isfinite(inst)]
        if finite.size < min_comp_i:
            continue
        cat = float(comp_catalog_mag.get(cid, float("nan")))
        if not math.isfinite(cat):
            continue
        y = float(np.nanmedian(cat - finite))
        if not math.isfinite(y):
            continue
        bp_vals.append(bp)
        ys.append(y)

    if len(ys) < min_comp_i:
        return float("nan"), float("nan")

    bp_med = float(np.median(np.asarray(bp_vals, dtype=np.float64)))
    xs = [float(bp) - bp_med for bp in bp_vals]
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)

    if x.size < 2:
        return float("nan"), float("nan")

    scatter = float(np.std(y))
    if not (math.isfinite(float(c1)) and float(c1) != 0.0):
        return scatter, float("nan")
    resid_ct = y - float(c1) * x
    scatter_resid = float(np.std(resid_ct)) if resid_ct.size >= 2 else float("nan")
    return scatter, scatter_resid

def _append_ct_prototype_row(draft_dir: Path, row: dict[str, Any]) -> None:
    import csv  # noqa: PLC0415

    draft_dir = Path(draft_dir)
    path = draft_dir / "ct_prototype.csv"
    write_header = not path.is_file() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(_CT_PROTOTYPE_CSV_FIELDS))
        if write_header:
            writer.writeheader()
        out_row: dict[str, Any] = {}
        for key in _CT_PROTOTYPE_CSV_FIELDS:
            val = row.get(key, "")
            if key == "gate_would_pass" or isinstance(val, (bool, np.bool_)):
                out_row[key] = "True" if bool(val) else "False"
            else:
                out_row[key] = val
        writer.writerow(out_row)

def savgol_detrend_lc(
    mag_calib: np.ndarray,
    bjd: np.ndarray,
    flags: list[str],
    *,
    window_frac: float = 0.5,
    polyorder: int = 2,
    min_points: int = 10,
    enabled: bool = True,
) -> np.ndarray:
    """Savitzky-Golay non-linear detrending of calibrated LC (ALG-2).

    Reference: Savitzky & Golay (1964) Anal. Chem. 36, 1627;
    applied to differential photometry detrending as in
    Aigrain & Irwin (2004) MNRAS 350, 331.

    Runs AFTER airmass detrend - removes slow systematic trends
    (transparency drift, seeing variations) not captured by linear
    airmass model. Applied to residuals, not raw mag_calib.

    Args:
        mag_calib:    detrended mag array (length = n_frames)
        bjd:          BJD timestamps (same length)
        flags:        per-frame flags; only "normal" frames used for fit
        window_frac:  SG window as fraction of n_normal frames (default 0.5)
        polyorder:    SG polynomial order (default 2)
        min_points:   minimum normal frames required (default 10)
        enabled:      False -> return mag_calib unchanged

    Returns:
        mag_sg: detrended mag array (same length as input)
    """
    _ = bjd  # frame order implicit in index; time-aware SG reserved for later
    if not enabled:
        return mag_calib.copy()

    from scipy.signal import savgol_filter  # lazy import - scipy already in deps

    mag = np.asarray(mag_calib, dtype=float)
    normal_mask = np.array([f == "normal" for f in flags])
    n_normal = int(normal_mask.sum())

    if n_normal < min_points:
        return mag.copy()

    raw_w = max(5, int(n_normal * window_frac))
    window = raw_w if raw_w % 2 == 1 else raw_w + 1
    window = min(window, n_normal if n_normal % 2 == 1 else n_normal - 1)

    if window <= polyorder:
        return mag.copy()

    # Warn if window covers more than 40% of the LC - risk of suppressing real variability
    if window > 0.4 * n_normal:
        LOGGER.warning(
            "[ALG-2 SG] window=%d is %.0f%% of n_normal=%d - "
            "may suppress real variability; consider decreasing savgol_window_frac "
            "(current=%.2f) or disabling savgol_detrend_enabled",
            window,
            100.0 * window / n_normal,
            n_normal,
            window_frac,
        )

    idx_normal = np.where(normal_mask)[0]
    mag_normal = mag[idx_normal]

    try:
        mag_smooth = savgol_filter(mag_normal, window_length=window, polyorder=polyorder)
    except Exception:  # noqa: BLE001
        # EXC-0144: T4 -- SavGol detrend failure returns un-detrended mag array unchanged (EXCEPT-BULK-2 2026-07-08)
        return mag.copy()

    trend_all = np.interp(
        np.arange(len(mag)),
        idx_normal,
        mag_smooth,
    )

    mag_sg = mag - trend_all + np.nanmedian(mag_normal)

    LOGGER.debug(
        "[ALG-2 SG] window=%d/%d frames, poly=%d, n_normal=%d",
        window,
        len(mag),
        polyorder,
        n_normal,
    )
    return mag_sg

def compute_mag_calib_final(
    mag_calib: np.ndarray,
    *,
    ct_ok: bool = False,
    ct_correction: float | None = None,
    ac_ok: bool = False,
    delta_m_corr: float | None = None,
    mag_calib_ac: np.ndarray | None = None,
) -> np.ndarray:
    """Canonical published magnitude: final ``mag_calib`` + CT + AC (per-target/night constants).

    CT and AC are additive on the ensemble-calibrated base. When CT is off, result matches
    ``mag_calib_ac`` (same ``mag_calib`` + ``delta_m_corr``) for byte-identical export under CT-off config.
    """
    base = np.asarray(mag_calib, dtype=np.float64)
    ct_add = 0.0
    if bool(ct_ok) and ct_correction is not None:
        try:
            v = float(ct_correction)
            if math.isfinite(v):
                ct_add = v
        except (TypeError, ValueError):
            pass
    ac_add = 0.0
    if bool(ac_ok) and delta_m_corr is not None:
        try:
            v = float(delta_m_corr)
            if math.isfinite(v):
                ac_add = v
        except (TypeError, ValueError):
            pass
    if ct_add == 0.0 and ac_add == 0.0:
        return base.copy()
    if ct_add == 0.0 and ac_add != 0.0 and mag_calib_ac is not None:
        ac_arr = np.asarray(mag_calib_ac, dtype=np.float64)
        if ac_arr.shape == base.shape:
            return ac_arr.copy()
    return base + ct_add + ac_add

def save_lightcurve_csv(
    output_path: Path,
    bjd: np.ndarray,
    hjd: np.ndarray,
    jd: np.ndarray,
    airmass: np.ndarray,
    is_flipped: np.ndarray | None,
    mag_inst: np.ndarray,
    mag_calib_raw: np.ndarray,
    mag_calib: np.ndarray,
    mag_calib_ct: np.ndarray | None,
    mag_calib_ac: np.ndarray | None,
    delta_mag: np.ndarray,
    err: np.ndarray,
    aperture_r_px: np.ndarray,
    flags: list[str],
    source_files: list[str],
    *,
    method: str = "aperture",
    ct_correction: float | None = None,
    ct_c1: float | None = None,
    ct_c1_stderr: float | None = None,
    ct_mode: str | None = None,
    ct_bp_rp_target: float | None = None,
    ct_bp_rp_comp_med: float | None = None,
    ct_n_comp: int | None = None,
    ct_ok: bool | None = None,
    k2_source: list[str] | np.ndarray | None = None,
    k2_value: float | None = None,
    k2_colour_ref: float | None = None,
    ac_result: dict[str, Any] | None = None,
    mag_democratic: np.ndarray | None = None,
    err_inflation: np.ndarray | None = None,
    lunar_phase_pct: float = float("nan"),
    lunar_separation_deg: float = float("nan"),
    lunar_risk: str = "UNKNOWN",
    dilution_factor: float = 1.0,
    alignment_failed: np.ndarray | None = None,
    err_scatter_unmatched: np.ndarray | None = None,
    catalog_match_mode: list[str] | np.ndarray | None = None,
    wcs_untrusted: np.ndarray | None = None,
    time_base: str = TIME_BASE_BJD_TDB,
    err_method: list[str] | None = None,
    sigma_sys_mag: float | None = None,
    err_photon: np.ndarray | None = None,
    err_sem_rel: np.ndarray | None = None,
    err_scint_rel: np.ndarray | None = None,
    err_sigma_sys_rel: np.ndarray | None = None,
    aperture_policy: dict[str, Any] | None = None,
) -> None:
    """Ulozi lightcurve CSV.

    LC schema note: ``time_base`` labels the BJD/HJD recompute path (``BJD_TDB`` vs
    ``JD_FALLBACK``); it does not alter ``bjd``/``hjd``/``jd`` values.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = int(len(bjd))
    if mag_calib_ct is None:
        mag_calib_ct_arr = np.asarray(mag_calib, dtype=np.float64).copy()
    else:
        mag_calib_ct_arr = np.asarray(mag_calib_ct, dtype=np.float64)
        if len(mag_calib_ct_arr) != n:
            mag_calib_ct_arr = np.asarray(mag_calib, dtype=np.float64).copy()

    ac_ok = bool(ac_result.get("ok", False)) if isinstance(ac_result, dict) else False
    delta_m_corr = (
        float(ac_result.get("delta_m_corr"))
        if (ac_ok and isinstance(ac_result, dict) and ac_result.get("delta_m_corr", None) is not None)
        else float("nan")
    )
    ac_scatter = (
        float(ac_result.get("scatter_mag"))
        if (isinstance(ac_result, dict) and ac_result.get("scatter_mag", None) is not None)
        else float("nan")
    )
    ac_n_ref = int(ac_result.get("n_ref_stars", 0)) if isinstance(ac_result, dict) else 0

    if mag_calib_ac is None:
        if ac_ok and math.isfinite(delta_m_corr):
            mag_calib_ac_arr = np.asarray(mag_calib, dtype=np.float64) + float(delta_m_corr)
        else:
            mag_calib_ac_arr = np.full_like(np.asarray(mag_calib, dtype=np.float64), float("nan"), dtype=np.float64)
    else:
        mag_calib_ac_arr = np.asarray(mag_calib_ac, dtype=np.float64)
        if len(mag_calib_ac_arr) != n:
            mag_calib_ac_arr = np.full_like(np.asarray(mag_calib, dtype=np.float64), float("nan"), dtype=np.float64)

    mag_calib_final_arr = compute_mag_calib_final(
        mag_calib,
        ct_ok=bool(ct_ok),
        ct_correction=ct_correction,
        ac_ok=ac_ok,
        delta_m_corr=delta_m_corr if ac_ok else None,
        mag_calib_ac=mag_calib_ac_arr,
    )

    def _fill_scalar(v: float | None, default: float) -> np.ndarray:
        vv = float(v) if v is not None else float(default)
        return np.full(n, vv, dtype=np.float64)

    def _fill_bool(v: bool | None) -> np.ndarray:
        vb = bool(v) if v is not None else False
        return np.full(n, vb, dtype=bool)

    def _flag_cell(f: Any) -> str:
        if f is None:
            return ""
        s = str(f).strip()
        return s if s.lower() not in ("none", "nan") else ""

    df = pd.DataFrame(
        {
            "bjd": bjd,
            "hjd": hjd,
            "jd": jd,
            "time_base": [str(time_base or TIME_BASE_BJD_TDB)] * n,
            "airmass": airmass,
            "is_flipped": (is_flipped if is_flipped is not None else np.full_like(bjd, False, dtype=bool)),
            "alignment_failed": (
                alignment_failed if alignment_failed is not None else np.full_like(bjd, False, dtype=bool)
            ),
            "err_scatter_unmatched": (
                err_scatter_unmatched
                if err_scatter_unmatched is not None
                else np.full_like(bjd, False, dtype=bool)
            ),
            "catalog_match_mode": (
                [normalize_catalog_match_mode(m) for m in catalog_match_mode]
                if catalog_match_mode is not None
                else [""] * n
            ),
            "wcs_untrusted": (
                wcs_untrusted if wcs_untrusted is not None else np.full_like(bjd, False, dtype=bool)
            ),
            "mag_inst": np.round(mag_inst, 6),
            "mag_calib_raw": np.round(mag_calib_raw, 6),
            "mag_calib": np.round(mag_calib, 6),
            "mag_calib_ct": np.round(mag_calib_ct_arr, 6),
            "ct_correction": np.round(_fill_scalar(ct_correction, float("nan")), 6),
            "ct_c1": np.round(_fill_scalar(ct_c1, float("nan")), 6),
            "ct_c1_stderr": np.round(_fill_scalar(ct_c1_stderr, float("nan")), 6),
            "ct_mode": str(ct_mode or ""),
            "ct_bp_rp_target": np.round(_fill_scalar(ct_bp_rp_target, float("nan")), 6),
            "ct_bp_rp_comp_med": np.round(_fill_scalar(ct_bp_rp_comp_med, float("nan")), 6),
            "ct_n_comp": np.full(n, int(ct_n_comp) if ct_n_comp is not None else -1, dtype=int),
            "ct_ok": _fill_bool(ct_ok),
            "k2_source": (
                [str(x) for x in k2_source]
                if k2_source is not None
                else [""] * n
            ),
            "k2_value": np.round(_fill_scalar(k2_value, float("nan")), 6),
            "k2_colour_ref": np.round(_fill_scalar(k2_colour_ref, float("nan")), 6),
            "ac_correction": np.round(_fill_scalar(delta_m_corr if (ac_ok and math.isfinite(delta_m_corr)) else None, float("nan")), 6),
            "ac_scatter": np.round(_fill_scalar(ac_scatter if math.isfinite(ac_scatter) else None, float("nan")), 6),
            "ac_n_ref": np.full(n, int(ac_n_ref), dtype=int),
            "ac_ok": np.full(n, bool(ac_ok), dtype=bool),
            "mag_calib_ac": np.round(mag_calib_ac_arr, 6),
            "mag_calib_final": np.round(mag_calib_final_arr, 6),
            "delta_mag": np.round(delta_mag, 6),
            "err": np.round(err, 6),
            "aperture_r_px": np.round(aperture_r_px, 3),
            "flag": [_flag_cell(f) for f in flags],
            "method": method,
            "source_file": source_files,
        }
    )
    if mag_democratic is not None:
        _md = np.asarray(mag_democratic, dtype=float)
        df["delta_mag_democratic"] = np.round(
            _md - float(np.nanmedian(_md)),
            6,
        )
    if err_inflation is not None:
        df["err_inflation"] = np.round(np.asarray(err_inflation, dtype=float), 6)
    _lp = float(lunar_phase_pct) if math.isfinite(float(lunar_phase_pct)) else float("nan")
    _ls = float(lunar_separation_deg) if math.isfinite(float(lunar_separation_deg)) else float("nan")
    _dfac = float(dilution_factor) if math.isfinite(float(dilution_factor)) else 1.0
    df["dilution_factor"] = np.round(np.full(n, _dfac, dtype=np.float64), 6)
    df["lunar_phase_pct"] = np.round(np.full(n, _lp, dtype=np.float64), 6)
    df["lunar_separation_deg"] = np.round(np.full(n, _ls, dtype=np.float64), 6)
    df["lunar_risk"] = [str(lunar_risk or "UNKNOWN")] * n
    if err_method is not None:
        df["err_method"] = [str(m) for m in err_method]
    _ssm = float(sigma_sys_mag) if sigma_sys_mag is not None and math.isfinite(float(sigma_sys_mag)) else float("nan")
    df["sigma_sys_mag"] = np.round(np.full(n, _ssm, dtype=np.float64), 6)
    if err_photon is not None:
        df["err_photon"] = np.round(np.asarray(err_photon, dtype=np.float64), 6)
    if err_sem_rel is not None:
        df["err_sem_rel"] = np.round(np.asarray(err_sem_rel, dtype=np.float64), 6)
    if err_scint_rel is not None:
        df["err_scint_rel"] = np.round(np.asarray(err_scint_rel, dtype=np.float64), 6)
    if err_sigma_sys_rel is not None:
        df["err_sigma_sys_rel"] = np.round(np.asarray(err_sigma_sys_rel, dtype=np.float64), 6)
    df["delta_mag_sysrem"] = np.round(np.full(n, float("nan"), dtype=np.float64), 6)
    if isinstance(aperture_policy, dict) and aperture_policy:
        df["aperture_policy"] = str(aperture_policy.get("mode") or "")
        try:
            df["aperture_f"] = np.round(
                np.full(n, float(aperture_policy.get("f", float("nan"))), dtype=np.float64), 6
            )
        except (TypeError, ValueError):
            df["aperture_f"] = np.full(n, float("nan"), dtype=np.float64)
        try:
            _fn = aperture_policy.get("fwhm_night_median_px")
            df["fwhm_night_median_px"] = np.round(
                np.full(n, float(_fn) if _fn is not None else float("nan"), dtype=np.float64),
                4,
            )
        except (TypeError, ValueError):
            df["fwhm_night_median_px"] = np.full(n, float("nan"), dtype=np.float64)
    df.to_csv(output_path, index=False)

def save_lightcurve_png(
    output_path: Path,
    bjd: np.ndarray,
    mag_calib: np.ndarray,
    err: np.ndarray,
    flags: list[str],
    target_name: str,
    comp_quality: dict[str, dict],
    *,
    delta_mag_mode: bool = False,
    delta_mag: np.ndarray | None = None,
) -> None:
    """Ulozi PNG graf svetelnej krivky s farebnymi flagmi a comp status tabulkou."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        logging.warning("[FAZA 2A] matplotlib nie je dostupny, PNG sa nevygeneruje")
        return

    y_data = delta_mag if (delta_mag_mode and delta_mag is not None) else mag_calib
    y_label = "Deltamag (ensemble)" if delta_mag_mode else "mag_calib"

    flag_colors = {
        "normal": "#1a1a2e",
        "saturated": "#aaaaaa",
        "outlier_hi": "#ff6b35",
        "outlier_lo": "#7b2d8b",
        "no_data": "#cccccc",
    }

    fig, (ax_lc, ax_comp) = plt.subplots(
        1,
        2,
        figsize=(14, 5),
        gridspec_kw={"width_ratios": [3, 1]},
    )
    fig.suptitle(f"VYVAR - {target_name}", fontsize=11, fontweight="bold")

    bjd_plot_all, bjd_axis_int = jd_series_relative(bjd)

    # Svetelna krivka
    for flag, color in flag_colors.items():
        mask = np.array([f == flag for f in flags])
        if not mask.any():
            continue
        bjd_f = bjd_plot_all[mask]
        y_f = y_data[mask]
        err_f = err[mask]
        valid = np.isfinite(y_f)
        if not valid.any():
            continue
        ax_lc.errorbar(
            bjd_f[valid],
            y_f[valid],
            yerr=err_f[valid],
            fmt="o",
            color=color,
            markersize=4,
            elinewidth=0.8,
            capsize=2,
            label=flag,
            alpha=0.85,
        )

    ax_lc.set_xlabel(jd_axis_title("BJD (TDB)", bjd_axis_int), fontsize=9)
    ax_lc.set_ylabel(y_label, fontsize=9)
    ax_lc.invert_yaxis()
    ax_lc.grid(True, alpha=0.3, linewidth=0.5)
    legend_patches = [
        mpatches.Patch(color=c, label=f) for f, c in flag_colors.items() if f != "no_data"
    ]
    ax_lc.legend(handles=legend_patches, fontsize=7, loc="upper right")

    # Comp quality tabulka
    ax_comp.axis("off")
    comp_lines = []
    for i, (_, info) in enumerate(comp_quality.items(), 1):
        q = str(info["quality"])
        p2p = info.get("rms_p2p", float("nan"))
        icon = "[OK]" if q == "good" else ("[??]" if q == "suspect" else "[X]")
        p2p_str = f"{p2p:.4f}" if math.isfinite(float(p2p)) else "N/A"
        comp_lines.append(f"{icon} C{i:02d}  {p2p_str}  {q}")

    comp_text = "\n".join(comp_lines[:15])  # max 15 riadkov
    ax_comp.text(
        0.05,
        0.95,
        "Comparison Stars\n(rms_p2p | quality)\n\n" + comp_text,
        transform=ax_comp.transAxes,
        fontsize=7,
        verticalalignment="top",
        fontfamily="monospace",
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

def save_cutout_png(
    output_path: Path,
    masterstar_fits_path: Path,
    xc: float,
    yc: float,
    target_name: str,
    *,
    size_px: int = 200,
    ms_data: np.ndarray | None = None,
) -> None:
    """Ulozi vyrez 200x200px z MASTERSTAR okolo targetu."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        return

    try:
        if ms_data is not None:
            data = np.asarray(ms_data, dtype=np.float64)
        else:
            with astrofits.open(masterstar_fits_path, memmap=False) as hdul:
                data = np.asarray(hdul[0].data, dtype=np.float64)
    except Exception:  # noqa: BLE001
        # EXC-0145: T3 -- Per-star cutout PNG export aborted when MASTERSTAR data cannot be loaded (EXCEPT-BULK-2 2026-07-08)
        return

    h, w = data.shape
    half = size_px // 2
    x0 = max(0, int(xc) - half)
    y0 = max(0, int(yc) - half)
    x1 = min(w, x0 + size_px)
    y1 = min(h, y0 + size_px)
    cutout = data[y0:y1, x0:x1]
    if cutout.size == 0:
        return

    # Percentilova skala
    vmin = float(np.percentile(cutout, 5))
    vmax = float(np.percentile(cutout, 99))

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cutout, origin="lower", cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")

    # Cerveny stvorecok pre target
    cx = xc - x0
    cy = yc - y0
    rect = mpatches.Rectangle(
        (cx - 10, cy - 10),
        20,
        20,
        linewidth=1.5,
        edgecolor="red",
        facecolor="none",
    )
    ax.add_patch(rect)
    ax.set_title(f"{target_name}", fontsize=8)
    ax.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)

def save_target_field_map_png(
    output_path: Path,
    masterstar_fits_path: Path,
    target_row: pd.Series,
    comp_rows: pd.DataFrame,
    *,
    percentile_lo: float = 5.0,
    percentile_hi: float = 99.5,
    ms_data: np.ndarray | None = None,
) -> None:
    """Per-target field map: cele pole, cerveny stvorec=target, zelene kruzky=comp (cislovane)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        return

    try:
        if ms_data is not None:
            data = np.asarray(ms_data, dtype=np.float64)
        else:
            with astrofits.open(masterstar_fits_path, memmap=False) as hdul:
                data = np.asarray(hdul[0].data, dtype=np.float64)
    except Exception:  # noqa: BLE001
        # EXC-0147: T3 -- Target field map PNG export aborted when MASTERSTAR data cannot be loaded (EXCEPT-BULK-2 2026-07-08)
        return

    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return
    vmin = float(np.percentile(finite, percentile_lo))
    vmax = float(np.percentile(finite, percentile_hi))

    fig, ax = plt.subplots(figsize=(18, 12))
    ax.imshow(data, origin="lower", cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")

    # Target - red square (DAO-matched photometry targets only; no catalog_only marker)
    try:
        tx, ty = float(target_row["x"]), float(target_row["y"])
        _z_t = str(
            target_row.get("zone_flag", target_row.get("zone", "")) or ""
        ).strip().lower()
        if _z_t == "catalog_only":
            # No photometry on catalog_only; do not draw a cyan "no DAO" marker.
            pass
        else:
            rect_t = mpatches.Rectangle(
                (tx - 15, ty - 15),
                30,
                30,
                linewidth=2.0,
                edgecolor="red",
                facecolor="none",
            )
            ax.add_patch(rect_t)
            tname = str(target_row.get("vsx_name", target_row.get("catalog_id", "T")))[:20]
            ax.text(
                tx + 18,
                ty,
                f"T: {tname}",
                color="red",
                fontsize=7,
                va="center",
                fontweight="bold",
            )
    except (KeyError, TypeError, ValueError):
        pass

    # Comp hviezdy - zelene kruzky s cislom (vsetky, bez orezania)
    for i, (_, crow) in enumerate(comp_rows.iterrows(), 1):
        try:
            cx, cy = float(crow["x"]), float(crow["y"])
        except (KeyError, TypeError, ValueError):
            continue
        circ = mpatches.Circle(
            (cx, cy),
            radius=14,
            linewidth=1.5,
            edgecolor="#00cc44",
            facecolor="none",
        )
        ax.add_patch(circ)
        ax.text(
            cx + 16,
            cy,
            f"C{i:02d}",
            color="#00cc44",
            fontsize=7,
            va="center",
            fontweight="bold",
        )

    target_name = str(target_row.get("vsx_name", target_row.get("catalog_id", "")))
    ax.set_title(
        f"VYVAR - {target_name}\n"
        f"(red=VSX target, green=comp star)",
        fontsize=10,
    )
    ax.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)

def pfs_rescue_eligible(
    *,
    zone_flag: str,
    skip_reason: str = "",
    likely_saturated: bool = False,
) -> bool:
    """True only for saturation-caused skips. Keyed on recorded skip_reason.

    Rescue-eligible: zone_flag==saturated, likely_saturated, or an explicit
    saturation skip_reason. NEVER rescue zone_noise or below_target_depth.
    Bare skip_photometry without a saturation reason is not eligible.
    """
    zf = str(zone_flag or "").strip().lower()
    sr = str(skip_reason or "").strip().lower()
    if sr in PFS_NEVER_RESCUE_REASONS:
        return False
    if zf == "noise":
        return False
    if zf in ("saturated", "likely_saturated") or bool(likely_saturated):
        return True
    if sr in PFS_SATURATION_SKIP_REASONS and zf != "linear":
        return True
    if sr in {"saturovany ciel", "per_frame_saturation", "likely_saturated", "saturated"}:
        return True
    return False

def _keep_recorded_skip_reason(*, skip_reason: str, zone_flag: str, legacy_skip: bool) -> str:
    recorded = str(skip_reason or "").strip()
    if recorded:
        return recorded
    zf = str(zone_flag or "").strip().lower()
    if zf == "noise":
        return "zone_noise"
    if legacy_skip or zf == "saturated":
        return "zone_flag"
    return ""

def decide_target_saturation_policy(
    *,
    zone_flag: str,
    legacy_skip: bool,
    frame_saturated: list[bool] | tuple[bool, ...] | np.ndarray | None,
    enabled: bool,
    min_clean_frac: float = 0.5,
    likely_saturated: bool = False,
    skip_reason: str = "",
) -> dict[str, Any]:
    """Per-target saturation gate (PER-FRAME-SAT-GATED / PFS-SEMANTICS-01).

    When ``enabled`` is False: legacy whole-star skip (zone / skip_photometry).
    When True: rescue only saturation-caused skips (recorded skip_reason /
    zone_flag==saturated). TARGET-DEPTH-02 (zone_noise, below_target_depth)
    outranks PFS: those skips are never cleared. Missing per-frame peak/sat
    data on a rescue-eligible target falls back to legacy zone behavior.
    """
    zf = str(zone_flag or "").strip().lower()
    recorded = str(skip_reason or "").strip()
    legacy = bool(legacy_skip) or zf == "saturated"
    eligible = pfs_rescue_eligible(
        zone_flag=zf,
        skip_reason=recorded,
        likely_saturated=bool(likely_saturated),
    )
    advisory = bool(legacy) or zf in ("saturated", "likely_saturated") or bool(likely_saturated)
    thr = float(min_clean_frac)
    if not math.isfinite(thr):
        thr = 0.5
    thr = max(0.1, min(1.0, thr))
    kept_reason = _keep_recorded_skip_reason(
        skip_reason=recorded, zone_flag=zf, legacy_skip=bool(legacy_skip)
    )

    if not enabled:
        return {
            "skip_photometry": bool(legacy),
            "skip_reason": "zone_flag" if legacy else "",
            "sat_clean_frac": float("nan"),
            "per_frame_sat_fallback": False,
            "n_frames": 0,
            "n_clean": 0,
        }

    if not eligible:
        n = 0
        n_clean = 0
        clean_frac = float("nan")
        if frame_saturated is not None:
            flags = [bool(x) for x in list(frame_saturated)]
            n = len(flags)
            if n > 0:
                n_clean = int(sum(1 for s in flags if not s))
                clean_frac = float(n_clean) / float(n)
        return {
            "skip_photometry": bool(legacy_skip),
            "skip_reason": kept_reason,
            "sat_clean_frac": clean_frac,
            "per_frame_sat_fallback": False,
            "n_frames": n,
            "n_clean": n_clean,
        }

    if frame_saturated is None:
        return {
            "skip_photometry": bool(legacy),
            "skip_reason": "zone_flag" if legacy else kept_reason,
            "sat_clean_frac": float("nan"),
            "per_frame_sat_fallback": True,
            "n_frames": 0,
            "n_clean": 0,
        }

    flags = [bool(x) for x in list(frame_saturated)]
    n = len(flags)
    if n == 0:
        return {
            "skip_photometry": bool(legacy),
            "skip_reason": "zone_flag" if legacy else kept_reason,
            "sat_clean_frac": float("nan"),
            "per_frame_sat_fallback": True,
            "n_frames": 0,
            "n_clean": 0,
        }

    n_clean = int(sum(1 for s in flags if not s))
    clean_frac = float(n_clean) / float(n)
    if not advisory:
        return {
            "skip_photometry": False,
            "skip_reason": "",
            "sat_clean_frac": clean_frac,
            "per_frame_sat_fallback": False,
            "n_frames": n,
            "n_clean": n_clean,
        }
    if clean_frac >= thr:
        return {
            "skip_photometry": False,
            "skip_reason": "",
            "sat_clean_frac": clean_frac,
            "per_frame_sat_fallback": False,
            "n_frames": n,
            "n_clean": n_clean,
        }
    return {
        "skip_photometry": True,
        "skip_reason": "per_frame_saturation",
        "sat_clean_frac": clean_frac,
        "per_frame_sat_fallback": False,
        "n_frames": n,
        "n_clean": n_clean,
    }

def _per_frame_sat_flags_for_catalog_id(
    catalog_id: str,
    csv_files: list[Path],
    csv_cache: dict[str, Any],
    *,
    sat_limit_adu: float | None = None,
    peak_test_adu: float | None = None,
) -> list[bool] | None:
    """Return per-frame saturation bools for a target, or None if data unavailable.

    Clean test uses ``peak_test_adu`` (INV-SAT-LIMIT catalog peak-test), not the
    raw container clip and not a stale ``is_saturated`` column. ``sat_limit_adu``
    is accepted as an explicit caller override when ``peak_test_adu`` is omitted
    (unit tests). Container clip is a different physical question (hard clip) and
    is not used here.
    """
    cid = _normalize_gaia_id(catalog_id)
    if not cid:
        return None
    lim_raw = peak_test_adu if peak_test_adu is not None else sat_limit_adu
    try:
        lim = float(lim_raw) if lim_raw is not None else float("nan")
    except (TypeError, ValueError):
        lim = float("nan")
    flags: list[bool] = []
    n_matched = 0
    for path in csv_files:
        df = csv_cache.get(str(path))
        if df is None or getattr(df, "empty", True):
            continue
        if "catalog_id" not in df.columns:
            return None
        ids = df["catalog_id"].astype(str).map(_normalize_gaia_id)
        m = ids.eq(cid)
        if not bool(m.any()):
            continue
        row = df.loc[m].iloc[0]
        n_matched += 1
        peak = float(pd.to_numeric(row.get("peak_max_adu"), errors="coerce"))
        if math.isfinite(peak) and math.isfinite(lim) and lim > 0:
            flags.append(bool(peak > lim))
            continue
        if "is_saturated" in df.columns:
            flags.append(_coerce_bool_cell(row.get("is_saturated")))
            continue
        if "likely_saturated" in df.columns and _coerce_bool_cell(row.get("likely_saturated")):
            flags.append(True)
            continue
        # Matched row but no usable sat diagnostic -> cannot evaluate.
        return None
    if n_matched == 0 or len(flags) == 0:
        return None
    return flags

def _resolve_pfs_peak_test(
    *,
    peak_test_adu: float | None,
    peak_test_source: str,
    sat_limit_adu: float | None,
) -> tuple[float | None, str, float, str]:
    """Return (peak_test, peak_test_source, container_clip, container_source)."""
    from pipeline import inv_sat_limit_peak_test_adu  # noqa: PLC0415

    container = float(SAT_LIMIT_CONTAINER_CLIP_ADU)
    container_src = "SAT_LIMIT_CONTAINER_CLIP_ADU"
    if peak_test_adu is not None:
        try:
            pt = float(peak_test_adu)
        except (TypeError, ValueError):
            pt = float("nan")
        if math.isfinite(pt) and pt > 0:
            src = str(peak_test_source or "").strip() or "caller_peak_test_adu"
            return pt, src, container, container_src
    if sat_limit_adu is not None:
        try:
            sl = float(sat_limit_adu)
        except (TypeError, ValueError):
            sl = float("nan")
        if math.isfinite(sl) and sl > 0:
            return sl, "caller_sat_limit_adu", container, container_src
    pt, src = inv_sat_limit_peak_test_adu()
    return float(pt), str(src), container, container_src

def apply_per_frame_saturation_to_active_targets(
    at_df: pd.DataFrame,
    *,
    csv_files: list[Path],
    csv_cache: dict[str, Any],
    sat_limit_adu: float | None,
    enabled: bool,
    min_clean_frac: float,
    peak_test_adu: float | None = None,
    peak_test_source: str = "",
) -> dict[str, Any]:
    """Mutate ``at_df`` skip columns per PER-FRAME-SAT-GATED; return night meta.

    When ``enabled`` is False: no-op (INV-CFG-01 - no new markers).
    Rescue is keyed on recorded skip_reason (PFS-SEMANTICS-01), not bare
    skip_photometry. Peak-test and container clip are named separately.
    """
    if not enabled:
        return {}

    pt, pt_src, container, container_src = _resolve_pfs_peak_test(
        peak_test_adu=peak_test_adu,
        peak_test_source=peak_test_source,
        sat_limit_adu=sat_limit_adu,
    )
    meta: dict[str, Any] = {
        "per_frame_sat_enabled": True,
        "per_frame_sat_min_clean_frac": float(min_clean_frac),
        "per_frame_sat_n_targets": int(len(at_df)),
        "per_frame_sat_n_fallback": 0,
        "per_frame_sat_n_rescued": 0,
        "per_frame_sat_n_skipped": 0,
        "per_frame_sat_peak_test_adu": float(pt) if pt is not None else float("nan"),
        "per_frame_sat_peak_test_source": str(pt_src),
        "per_frame_sat_container_clip_adu": float(container),
        "per_frame_sat_container_clip_source": str(container_src),
    }
    at_df["sat_clean_frac"] = float("nan")
    if "skip_reason" not in at_df.columns:
        at_df["skip_reason"] = ""
    else:
        at_df["skip_reason"] = at_df["skip_reason"].astype(object)
    at_df["per_frame_sat_fallback"] = False

    for idx, row in at_df.iterrows():
        cid = _normalize_gaia_id(row.get("catalog_id", ""))
        zf = str(row.get("zone_flag", "") or "").strip()
        legacy = bool(row.get("skip_photometry", False))
        recorded = str(row.get("skip_reason", "") or "").strip()
        likely = (
            _coerce_bool_cell(row.get("likely_saturated"))
            if "likely_saturated" in at_df.columns
            else False
        )
        flags = _per_frame_sat_flags_for_catalog_id(
            cid,
            csv_files,
            csv_cache,
            sat_limit_adu=sat_limit_adu,
            peak_test_adu=pt,
        )
        dec = decide_target_saturation_policy(
            zone_flag=zf,
            legacy_skip=legacy,
            frame_saturated=flags,
            enabled=True,
            min_clean_frac=float(min_clean_frac),
            likely_saturated=likely,
            skip_reason=recorded,
        )
        at_df.at[idx, "skip_photometry"] = bool(dec["skip_photometry"])
        at_df.at[idx, "skip_reason"] = str(dec["skip_reason"] or "")
        at_df.at[idx, "sat_clean_frac"] = float(dec["sat_clean_frac"])
        at_df.at[idx, "per_frame_sat_fallback"] = bool(dec["per_frame_sat_fallback"])
        if dec["per_frame_sat_fallback"]:
            meta["per_frame_sat_n_fallback"] = int(meta["per_frame_sat_n_fallback"]) + 1
        if (not dec["skip_photometry"]) and legacy:
            meta["per_frame_sat_n_rescued"] = int(meta["per_frame_sat_n_rescued"]) + 1
        if dec["skip_photometry"] and str(dec["skip_reason"]) == "per_frame_saturation":
            meta["per_frame_sat_n_skipped"] = int(meta["per_frame_sat_n_skipped"]) + 1
    return meta

def _fits_header_facts(header: Any) -> dict[str, Any]:
    """Best-effort filter / exptime / binning from a FITS header (metadata only)."""
    out: dict[str, Any] = {"filter": None, "exptime_s": None, "binning": None}
    if header is None:
        return out
    try:
        _flt = header.get("FILTER")
        if _flt is not None and str(_flt).strip():
            out["filter"] = str(_flt).strip()
    except Exception:  # noqa: BLE001
        pass
    try:
        _exp = header.get("EXPTIME", header.get("EXPOSURE"))
        if _exp is not None and str(_exp).strip() != "":
            out["exptime_s"] = float(_exp)
    except Exception:  # noqa: BLE001
        pass
    try:
        _xb = header.get("XBINNING")
        _yb = header.get("YBINNING")
        if _xb is not None and _yb is not None:
            out["binning"] = f"{int(float(_xb))}x{int(float(_yb))}"
        elif _xb is not None:
            out["binning"] = f"{int(float(_xb))}x{int(float(_xb))}"
    except Exception:  # noqa: BLE001
        pass
    return out

def _build_phase2a_resolved_facts(
    *,
    cfg: Any,
    gain_res: Any,
    rn_res: Any,
    gain_value: float | None,
    rn_value: float | None,
    site: Any,
    sat_limit: float | None,
    plate_scale_arcsec: float | None,
    frame_width_px: int | None,
    frame_height_px: int | None,
    ms_header: Any,
    obs_group: str,
) -> dict[str, Any]:
    """Run-effective resolved facts for ``pipeline_meta.resolved_facts`` (metadata only).

    Records the values the run ACTUALLY used (with sources for gain/read-noise/site) so the
    report can show an honest "resolved facts" block. Numeric behaviour is unchanged: this is
    never read by the science path and the anchor comparator ignores ``pipeline_meta.json``.
    """
    def _num_or_none(v: Any) -> float | None:
        try:
            f = float(v)
            return f if math.isfinite(f) else None
        except (TypeError, ValueError):
            return None

    _hdr = _fits_header_facts(ms_header)
    if not _hdr.get("filter"):
        _hdr["filter"] = str(obs_group or "") or None

    # Site: coordinates from the per-draft resolver when it succeeded, else config; id/name
    # mirror pipeline_meta.observer_location (drawn from cfg).
    site_ok = bool(getattr(site, "ok", False))
    if site_ok:
        site_lat = _num_or_none(getattr(site, "lat", None))
        site_lon = _num_or_none(getattr(site, "lon", None))
        site_alt = _num_or_none(getattr(site, "elev", None))
    else:
        site_lat = _num_or_none(getattr(cfg, "observer_lat", None))
        site_lon = _num_or_none(getattr(cfg, "observer_lon", None))
        site_alt = _num_or_none(getattr(cfg, "observer_alt_m", None))
    try:
        loc_id = int(getattr(cfg, "observer_location_id", 0) or 0)
    except (TypeError, ValueError):
        loc_id = 0

    return {
        "site": {
            "location_id": loc_id,
            "name": str(getattr(cfg, "observer_location_name", "") or "").strip(),
            "lat": site_lat,
            "lon": site_lon,
            "alt_m": site_alt,
            "source": str(getattr(site, "source", "unresolved")),
            "ok": site_ok,
        },
        "gain": {
            "value": _num_or_none(gain_value),
            "source": (getattr(gain_res, "source", None) if getattr(gain_res, "ok", False) else "default"),
            "key": getattr(gain_res, "key", None),
        },
        "read_noise": {
            "value": _num_or_none(rn_value),
            "source": (getattr(rn_res, "source", None) if getattr(rn_res, "ok", False) else "default"),
            "key": getattr(rn_res, "key", None),
        },
        "saturation_adu": _num_or_none(sat_limit),
        "plate_scale_arcsec_per_px": _num_or_none(plate_scale_arcsec),
        "frame_width_px": int(frame_width_px) if frame_width_px else None,
        "frame_height_px": int(frame_height_px) if frame_height_px else None,
        "binning": _hdr["binning"],
        "filter": _hdr["filter"],
        "exptime_s": _hdr["exptime_s"],
    }

@dataclass(frozen=True)
class BlendMapEntry:
    """One row from ``crowding_targets.csv`` for NEIGHBOR-SUB / adaptive routing."""

    is_blended: bool
    nn_dist_fwhm: float
    nn_catalog_id: str | None = None
    delta_mag_nn: float | None = None
    nn_ra_deg: float | None = None
    nn_dec_deg: float | None = None
    mag: float | None = None

def _load_blend_worklist(masterstar_fits_path: Path) -> dict[str, BlendMapEntry]:
    """``catalog_id`` -> blend worklist row from ``crowding_targets.csv`` (cached)."""
    key = str(masterstar_fits_path)
    if key in _ADAPTIVE_BLEND_CACHE:
        return _ADAPTIVE_BLEND_CACHE[key]
    m: dict[str, BlendMapEntry] = {}
    try:
        p = Path(masterstar_fits_path).parent / "crowding_targets.csv"
        if p.is_file():
            df = pd.read_csv(p, low_memory=False)
            for _, r in df.iterrows():
                cid = str(r.get("catalog_id", "")).strip()
                if not cid or cid.lower() == "nan":
                    continue
                isb = _coerce_bool_cell(r.get("is_blended"))
                nn = float(pd.to_numeric(r.get("nn_dist_fwhm"), errors="coerce"))
                nncid = str(r.get("nn_catalog_id", "") or "").strip() or None
                dmag = float(pd.to_numeric(r.get("delta_mag_nn"), errors="coerce"))
                dmag_v = dmag if math.isfinite(dmag) else None
                ra = float(pd.to_numeric(r.get("ra_deg"), errors="coerce"))
                de = float(pd.to_numeric(r.get("dec_deg"), errors="coerce"))
                mag_v = float(pd.to_numeric(r.get("mag"), errors="coerce"))
                entry = BlendMapEntry(
                    is_blended=isb,
                    nn_dist_fwhm=nn,
                    nn_catalog_id=nncid,
                    delta_mag_nn=dmag_v,
                    nn_ra_deg=ra if math.isfinite(ra) else None,
                    nn_dec_deg=de if math.isfinite(de) else None,
                    mag=mag_v if math.isfinite(mag_v) else None,
                )
                m[cid] = entry
                ncid = _normalize_gaia_id(cid)
                if ncid:
                    m[ncid] = entry
    except Exception as e:  # noqa: BLE001
        logging.error('[EXC-0162] ePSF blend worklist JSON load fails - adaptive blend deblend map empty, crowded stars u...: %s', e)
        logging.warning("[ePSF] blend worklist load failed (%s)", e)
    _ADAPTIVE_BLEND_CACHE[key] = m
    return m

def _load_adaptive_blend_map(masterstar_fits_path: Path) -> dict[str, tuple[bool, float]]:
    """``catalog_id`` -> ``(is_blended, nn_dist_fwhm)`` (legacy tuple API; see ``BlendMapEntry``)."""
    return {
        k: (v.is_blended, v.nn_dist_fwhm) for k, v in _load_blend_worklist(masterstar_fits_path).items()
    }

def _route_lc_per_frame_err(
    target_frames: pd.DataFrame,
    err: np.ndarray,
) -> tuple[np.ndarray, list[str] | None]:
    """Use PSF sandwich err for PSF-routed frames when finite; else keep aperture err."""
    if "lc_flux_method" not in target_frames.columns:
        return err, None
    methods = target_frames["lc_flux_method"].astype(str).to_numpy()
    if not np.any(methods == "psf"):
        return err, None
    out = np.asarray(err, dtype=float).copy()
    err_methods = ["aperture"] * len(out)
    pf = pd.to_numeric(target_frames.get("psf_flux"), errors="coerce").to_numpy(dtype=float)
    pfe = pd.to_numeric(target_frames.get("psf_flux_err"), errors="coerce").to_numpy(dtype=float)
    psf_mask = (methods == "psf") & np.isfinite(pf) & (pf > 0) & np.isfinite(pfe) & (pfe > 0)
    if np.any(psf_mask):
        out[psf_mask] = _PSF_ERR_MAG_SCALE * pfe[psf_mask] / pf[psf_mask]
        for i in np.where(psf_mask)[0]:
            err_methods[int(i)] = "psf"
    return out, err_methods

def _get_lc(cid: str, all_frames: pd.DataFrame) -> np.ndarray:
    sub = all_frames[all_frames["catalog_id"] == cid]
    if sub.empty:
        return np.array([], dtype=float)
    if "source_file" in sub.columns:
        sub = sub.sort_values(["source_file"], kind="mergesort")
    return sub["mag_inst"].to_numpy(dtype=float)

def _get_comp_bjd_series(cid: str, all_frames: pd.DataFrame) -> np.ndarray:
    """BJD (or JD) time series for a comp, same row order as ``_get_lc``."""
    sub = all_frames[all_frames["catalog_id"] == cid]
    if sub.empty:
        return np.array([], dtype=float)
    if "source_file" in sub.columns:
        sub = sub.sort_values(["source_file"], kind="mergesort")
    if "bjd" in sub.columns:
        return sub["bjd"].to_numpy(dtype=float)
    if "jd" in sub.columns:
        return sub["jd"].to_numpy(dtype=float)
    return np.array([], dtype=float)

def compute_lc_flux_method(
    all_frames: pd.DataFrame,
    blend_map: dict[str, tuple[bool, float]] | None = None,
    *,
    resolve_fwhm: float = 2.0,
    snr_lo: float = 15.0,
) -> pd.Series:
    """Per-star/per-frame adaptive flux-source choice in {``aperture``, ``psf``} (b.4).

    CONSERVATIVE: default to aperture, switch to PSF only with positive evidence AND good
    PSF quality. ``blend_map`` is accepted for API compatibility (crowding_targets.csv)
    but is not used for routing - resolvable-blend -> PSF (former rule 2) was removed
    because ``is_blended`` (nn <= 1.5 FWHM) and ``resolve_fwhm`` (>= 2.0) are mutually
    exclusive and grouped deblending showed no precision gain at 0.39"/px on draft 364.

    Rules (first match wins):
      1. psf_quality == bad / not fit_ok / no finite psf_flux  -> aperture (the b.5 fallback)
      2. faint (aperture SNR <= snr_lo) AND psf_quality == good   -> psf
      3. else                                                    -> aperture
    """
    n = len(all_frames)
    idx = all_frames.index
    if n == 0:
        return pd.Series([], dtype=object)
    _ = blend_map  # unused; kept so callers need not change when psf_adaptive_enabled

    psf_flux = pd.to_numeric(all_frames.get("psf_flux", pd.Series(np.nan, index=idx)), errors="coerce")
    psf_q = all_frames.get("psf_quality", pd.Series("", index=idx)).astype(str).str.strip().str.lower()
    _ok_raw = all_frames.get("psf_fit_ok", pd.Series(False, index=idx))
    psf_ok = _ok_raw.map(_coerce_bool_cell).to_numpy(dtype=bool)
    # Aperture SNR from the (mag) error: err_mag ~ 1.0857 / SNR.
    err = pd.to_numeric(all_frames.get("err", pd.Series(np.nan, index=idx)), errors="coerce").to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr_aper = np.where(np.isfinite(err) & (err > 0), 1.0857362 / err, np.inf)

    pf = psf_flux.to_numpy(dtype=float)
    _ac_raw = all_frames.get("psf_ac_applied", pd.Series(False, index=idx))
    psf_ac_ok = _ac_raw.map(_coerce_bool_cell).to_numpy(dtype=bool)
    psf_usable = psf_ok & np.isfinite(pf) & (pf > 0) & (psf_q.to_numpy() != "bad") & psf_ac_ok

    rule_faint_psf = psf_usable & (snr_aper <= float(snr_lo)) & (psf_q.to_numpy() == "good")

    method = np.where(rule_faint_psf, "psf", "aperture").astype(object)
    return pd.Series(method, index=idx)

def _recompute_bjd_hjd_with_status(
    jd_array: np.ndarray,
    ra_deg: float,
    dec_deg: float,
    cfg: AppConfig,
    site: tuple[float | None, float | None, float | None] | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Recompute per-target BJD(TDB) and HJD from frame JD values.

    Returns ``(bjd, hjd, time_base)`` where ``time_base`` is ``BJD_TDB`` on the astropy
    success path or ``JD_FALLBACK`` when raw JD is copied for both BJD and HJD.

    See ``_recompute_bjd_hjd_per_target`` for full docstring / references.
    """
    jd_arr = np.asarray(jd_array, dtype=float)
    bjd_out = np.full_like(jd_arr, float("nan"), dtype=float)
    hjd_out = np.full_like(jd_arr, float("nan"), dtype=float)

    if site is not None and site[0] is not None and site[1] is not None:
        lat = float(site[0])
        lon = float(site[1])
        alt = float(site[2]) if site[2] is not None else 0.0
    else:
        lat = float(cfg.observer_lat)
        lon = float(cfg.observer_lon)
        alt = float(cfg.observer_alt_m)

    from param_resolver import is_null_island_coords  # noqa: PLC0415

    if not math.isfinite(ra_deg) or not math.isfinite(dec_deg):
        LOGGER.warning(
            "BJD-PERTARGET: invalid coords ra=%s dec=%s - using frame JD fallback",
            ra_deg,
            dec_deg,
        )
        return jd_arr.copy(), jd_arr.copy(), TIME_BASE_JD_FALLBACK

    if is_null_island_coords(lat, lon):
        LOGGER.warning("BJD-PERTARGET: observer location not set - using frame JD fallback")
        return jd_arr.copy(), jd_arr.copy(), TIME_BASE_JD_FALLBACK

    try:
        import astropy.units as u
        from astropy.coordinates import EarthLocation, SkyCoord
        from astropy.time import Time

        location = EarthLocation(
            lat=lat * u.deg,
            lon=lon * u.deg,
            height=alt * u.m,
        )
        target = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
        finite = np.isfinite(jd_arr)
        if not np.any(finite):
            return bjd_out, hjd_out, TIME_BASE_BJD_TDB

        t = Time(jd_arr[finite], format="jd", scale="utc", location=location)
        ltt_bary = t.light_travel_time(target, "barycentric")
        bjd_finite = (t.tdb + ltt_bary).jd
        ltt_helio = t.light_travel_time(target, "heliocentric")
        hjd_finite = (t + ltt_helio).jd

        bjd_out[finite] = np.asarray(bjd_finite, dtype=float)
        hjd_out[finite] = np.asarray(hjd_finite, dtype=float)
        n_ok = int(np.sum(np.isfinite(bjd_out[finite])))
        LOGGER.debug(
            "BJD-PERTARGET: %d/%d frames recomputed (batch) for ra=%.4f dec=%.4f",
            n_ok,
            int(finite.sum()),
            ra_deg,
            dec_deg,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("BJD-PERTARGET: batch recompute failed (%s) - frame JD fallback", exc)
        return jd_arr.copy(), jd_arr.copy(), TIME_BASE_JD_FALLBACK

    return bjd_out, hjd_out, TIME_BASE_BJD_TDB

def run_sysrem_field(
    lc_dir: Path,
    *,
    n_iter: int = 3,
    flag_col: str = "flag",
    delta_col: str = "delta_mag",
    err_col: str = "err",
    out_col: str = "delta_mag_sysrem",
) -> dict[str, Any]:
    """Run SysRem systematic noise removal on all exported light curves.

    Implements the iterative algorithm of Tamuz, Mazeh & Zucker (2005),
    MNRAS 356, 1466. Removes common systematic trends from a set of
    differential light curves using inverse-variance weighting.

    Algorithm (one iteration):
        r_ij  = residuals (star i, frame j), NaN where flag != 'normal'
        w_ij  = 1 / err_ij^2  (weight; 0 where NaN or err <= 0)
        c_j   = sum_i(a_i * r_ij * w_ij) / sum_i(a_i^2 * w_ij)
        a_i   = sum_j(c_j * r_ij * w_ij) / sum_j(c_j^2 * w_ij)
        r_ij -= a_i * c_j

    Writes column ``delta_mag_sysrem`` into each lightcurve_*.csv.
    Stars with fewer than 10 valid frames are excluded from the
    system-vector fit but still corrected using the derived c_j vector.

    Args:
        lc_dir: Directory containing lightcurve_*.csv files.
        n_iter: Number of SysRem iterations (default 3).
        flag_col: Column name for quality flag.
        delta_col: Input column (default 'delta_mag').
        err_col: Uncertainty column (default 'err').
        out_col: Output column written to CSV (default 'delta_mag_sysrem').

    Returns:
        dict with keys:
            'n_stars': int - number of LC files processed
            'n_frames': int - number of frames (columns in matrix)
            'n_iter': int - iterations applied
            'rms_before': float - median RMS across stars before SysRem
            'rms_after': float - median RMS across stars after SysRem
            'rms_improvement_pct': float - (rms_before - rms_after) / rms_before * 100
            'skipped': list[str] - catalog_ids skipped (missing columns etc.)
    """
    lc_dir = Path(lc_dir)
    lc_files = sorted(lc_dir.glob("lightcurve_*.csv"))
    _empty: dict[str, Any] = {
        "n_stars": 0,
        "n_frames": 0,
        "n_iter": 0,
        "rms_before": float("nan"),
        "rms_after": float("nan"),
        "rms_improvement_pct": float("nan"),
        "skipped": [],
    }
    if not lc_files:
        logging.warning("[SysRem] No lightcurve_*.csv found in %s", lc_dir)
        return _empty

    dfs: dict[str, pd.DataFrame] = {}
    skipped: list[str] = []

    for f in lc_files:
        try:
            df = pd.read_csv(f, low_memory=False)
            required = {delta_col, err_col, flag_col}
            if not required.issubset(df.columns):
                skipped.append(f.stem)
                logging.warning("[SysRem] Missing columns in %s, skipping", f.name)
                continue
            cid = f.stem.replace("lightcurve_", "")
            dfs[cid] = df
        except Exception as exc:  # noqa: BLE001
            skipped.append(f.stem)
            logging.warning("[SysRem] Cannot read %s: %s", f.name, exc)

    if not dfs:
        logging.warning("[SysRem] No valid LC files loaded")
        return {**_empty, "skipped": skipped}

    n_frames = max(len(df) for df in dfs.values())
    star_ids = list(dfs.keys())
    n_stars = len(star_ids)

    R = np.full((n_stars, n_frames), np.nan)
    W = np.zeros((n_stars, n_frames))

    for i, cid in enumerate(star_ids):
        df = dfs[cid]
        nf = len(df)
        delta = pd.to_numeric(df[delta_col], errors="coerce").to_numpy()
        err = pd.to_numeric(df[err_col], errors="coerce").to_numpy()
        flags = df[flag_col].astype(str).to_numpy()

        valid = (flags == "normal") & np.isfinite(delta) & np.isfinite(err) & (err > 0)
        if valid.any():
            R[i, :nf] = np.where(valid, delta - float(np.nanmedian(delta[valid])), np.nan)
        W[i, :nf] = np.where(valid, 1.0 / (err**2), 0.0)

    min_valid_frames = 10
    fit_mask = np.sum(np.isfinite(R), axis=1) >= min_valid_frames

    rms_before_arr = np.array(
        [
            np.nanstd(R[i][np.isfinite(R[i])]) if np.isfinite(R[i]).sum() > 2 else np.nan
            for i in range(n_stars)
        ]
    )
    rms_before = float(np.nanmedian(rms_before_arr))

    a = np.ones(n_stars)
    fit_mask_f = fit_mask.astype(np.float64)

    for iteration in range(n_iter):
        R_filled = np.where(np.isfinite(R), R, 0.0)
        numer_c = np.einsum("i,ij,ij->j", a * fit_mask_f, R_filled, W)
        denom_c = np.einsum("i,ij->j", (a**2) * fit_mask_f, W)
        c = np.where(denom_c > 0, numer_c / denom_c, 0.0)

        numer_a = np.einsum("j,ij,ij->i", c, R_filled, W)
        denom_a = np.einsum("j,ij->i", c**2, W)
        a = np.where(denom_a > 0, numer_a / denom_a, 0.0)

        correction = np.outer(a, c)
        R = R - np.where(np.isfinite(R), correction, 0.0)

        logging.info(
            "[SysRem] Iteration %d/%d - median |c_j|=%.5f, median |a_i|=%.4f",
            iteration + 1,
            n_iter,
            float(np.median(np.abs(c))),
            float(np.median(np.abs(a))),
        )

    rms_after_arr = np.array(
        [
            np.nanstd(R[i][np.isfinite(R[i])]) if np.isfinite(R[i]).sum() > 2 else np.nan
            for i in range(n_stars)
        ]
    )
    rms_after = float(np.nanmedian(rms_after_arr))
    rms_improvement = (
        (rms_before - rms_after) / rms_before * 100.0 if rms_before > 0 else float("nan")
    )

    logging.info(
        "[SysRem] Done: %d stars x %d frames x %d iter | "
        "RMS before=%.4f after=%.4f improvement=%.1f%%",
        n_stars,
        n_frames,
        n_iter,
        rms_before,
        rms_after,
        rms_improvement,
    )

    for i, cid in enumerate(star_ids):
        df = dfs[cid]
        nf = len(df)
        delta_orig = pd.to_numeric(df[delta_col], errors="coerce").to_numpy()
        flags = df[flag_col].astype(str).to_numpy()
        valid = (flags == "normal") & np.isfinite(delta_orig)
        median_orig = float(np.nanmedian(delta_orig[valid])) if valid.any() else 0.0
        sysrem_col = np.full(nf, np.nan)
        sysrem_col[:nf] = R[i, :nf] + median_orig
        non_normal = flags != "normal"
        sysrem_col[non_normal] = delta_orig[non_normal]
        df = df.copy()
        df[out_col] = np.round(sysrem_col, 6)
        lc_path = lc_dir / f"lightcurve_{cid}.csv"
        df.to_csv(lc_path, index=False)

    return {
        "n_stars": n_stars,
        "n_frames": n_frames,
        "n_iter": n_iter,
        "rms_before": rms_before,
        "rms_after": rms_after,
        "rms_improvement_pct": rms_improvement,
        "skipped": skipped,
    }
