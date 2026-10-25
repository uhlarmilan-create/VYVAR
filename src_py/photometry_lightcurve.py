"""Moved from photometry_core.py (CONSOLIDATE-01E4). Facade re-exports these names."""
from __future__ import annotations

import astro_env  # noqa: F401  (IERS-OFF-2026-10: no IERS downloads)

from pathlib import Path
from typing import Any, Sequence
import logging

from mag_constants import MAG_ERR_SCALE
import math
import numpy as np
import pandas as pd
from config import AppConfig
from infolog import log_event
from jd_axis_format import jd_axis_title, jd_series_relative
from photometry_shared import _normalize_gaia_id
from pipeline_constants import SAT_LIMIT_CONTAINER_CLIP_ADU

from photometry_core import (
    LOGGER,
    TIME_BASE_BJD_TDB,
    TIME_BASE_JD_FALLBACK,
)


def _aperture_empty_comp_summary_row(
    *,
    target_cid: str,
    target_name: str,
    zone_flag: str,
    vsx_type: str = "",
) -> dict[str, Any]:
    """Summary stub when an active target has no per-target comparison stars."""
    return {
        "catalog_id": target_cid,
        "vsx_name": target_name,
        "vsx_type": vsx_type,
        "zone_flag": zone_flag,
        "n_frames": 0,
        "n_good_comp": 0,
        "n_saturated": 0,
        "lc_rms": float("nan"),
        "lc_median_mag": float("nan"),
        "aperture_px": float("nan"),
        "lc_csv": "",
        "lc_png": "",
        "skip_reason": "no_comps",
    }

def _aperture_skip_empty_comps_target(
    *,
    target_cid: str,
    target_name: str,
    zone_flag: str,
    summary_rows: list,
    vsx_type: str = "",
) -> list:
    """Record counted drop when an active target has no per-target comparison stars."""
    from except_fix_counters import get_except_fix_counters

    _ctr = get_except_fix_counters()
    _ctr.aperture_empty_comp_drop += 1
    logging.error(
        "[APERTURE] Target %s (%s): no comp stars - skipped "
        "(aperture_empty_comp_drop=%d)",
        target_name,
        target_cid,
        _ctr.aperture_empty_comp_drop,
    )
    summary_rows.append(
        _aperture_empty_comp_summary_row(
            target_cid=target_cid,
            target_name=target_name,
            zone_flag=zone_flag,
            vsx_type=vsx_type,
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

    Same principle as the comparison-star selection ``comp_rms`` (flux / bin median): comp intrinsic scatter
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
    max_comp_p2p: float = 0.1,
    max_comp_slope_mmag_hr: float = 5.0,
    comp_slope_significance_k: float = 3.0,
    common_mode_detrend: bool = True,
    stability_run_flags: dict[str, Any] | None = None,
) -> dict[str, dict]:
    """Step 3: stability check of the comparison stars.

    Abbe point-to-point scatter on **common-mode-detrended** comp residuals:
        rms_p2p = std(diff(mag_resid)) / sqrt(2)

    ``max_comp_p2p`` (config ``phase01_comparison_max_comp_rms``) gates **p2p** scatter.
    Comparison-star selection ``comp_rms`` uses the same config key but measures full LC RMS - different
    metric, shared ceiling knob (see ``select_comparison_stars_per_target``).

    Shared atmospheric drift is removed before p2p/MAD (same differential logic as ensemble).

    Returns:
        dict {catalog_id: {"rms_p2p": float, "lc_rms": float, "quality": str, "p2p_threshold": float}}
        quality: "good" / "suspect" / "excluded"; entries are sorted (good -> suspect -> excluded, within good by rms_p2p).
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

    # Compute metrics on the per-frame differential residual series
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

    # A star with comp_rms~0 from comparison-star selection -> mark as suspect
    # (likely an isolated-bin normalisation artefact).
    if comp_rms_map:
        for cid in result:
            try:
                comp_selection_rms = float(comp_rms_map.get(cid, float("nan")))
            except Exception:  # noqa: BLE001
                comp_selection_rms = float("nan")
            if math.isfinite(comp_selection_rms) and comp_selection_rms < 1e-6:
                result[cid]["quality"] = "suspect"
                result[cid]["note"] = "isolated_bin"

    # COMP-ADMIT-03: p2p is diagnostic only - never eject members (weight via sigma_rms).
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
        f"[APERTURE] Stability check: {n_good_final}/{len(result)} good comp "
        f"(p2p threshold={thr_log})"
    )

    # Sorting: good (by rms_p2p), suspect, excluded - the order used in the ensemble / PNG table
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

BROEG_TOL = 1e-5
BROEG_MAX_ITER = 200


def broeg_weights(
    y: np.ndarray,
    *,
    tol: float = BROEG_TOL,
    max_iter: int = BROEG_MAX_ITER,
) -> tuple[np.ndarray, int]:
    """Broeg, Fernandez & Neuhaeuser (2005) weights for comparison stars.

    ``y`` is (n_comp, n_frames) of normalised instrumental magnitudes (each comp minus its
    own median; NaN = missing). Each weight is ``1/sigma_j^2`` where ``sigma_j`` is the
    standard deviation of comp j against the weighted ensemble of the OTHER comps
    (leave-one-out); iterated until the largest relative weight change is < ``tol``.
    No clipping: a variable comparison star simply ends with a small weight.
    Returns (weights normalised to sum 1, n_iterations).
    """
    y = np.asarray(y, dtype=np.float64)
    n = int(y.shape[0])
    if n == 0:
        return np.zeros(0), 0
    if n == 1:
        return np.ones(1), 0
    fin = np.isfinite(y)
    y0 = np.where(fin, y, 0.0)

    def _sig(r: np.ndarray, m: np.ndarray) -> float:
        v = r[m]
        return float(np.std(v, ddof=1)) if v.size >= 3 else float("nan")

    w = np.array([_sig(y[j], fin[j]) for j in range(n)], dtype=np.float64)
    w = np.where(np.isfinite(w) & (w > 0), 1.0 / (w * w), np.nan)
    if not np.isfinite(w).any():
        return np.full(n, 1.0 / n), 0
    w = np.where(np.isfinite(w), w, np.nanmin(w))
    it = 0
    for it in range(1, int(max_iter) + 1):
        wf = w[:, None] * fin
        tot_w = wf.sum(axis=0)
        tot_wy = (wf * y0).sum(axis=0)
        new = np.empty(n, dtype=np.float64)
        for j in range(n):
            ow = tot_w - wf[j]
            ens = np.where(ow > 0, (tot_wy - wf[j] * y0[j]) / np.where(ow > 0, ow, 1.0), np.nan)
            r = y[j] - ens
            sj = _sig(r, np.isfinite(r))
            new[j] = 1.0 / (sj * sj) if math.isfinite(sj) and sj > 0 else np.nan
        if not np.isfinite(new).any():
            break
        new = np.where(np.isfinite(new), new, np.nanmin(new))
        rel = np.max(np.abs(new - w) / np.maximum(w, 1e-300))
        w = new
        if rel < float(tol):
            break
    return w / float(np.sum(w)), int(it)


def ensemble_normalize(
    target_mag_inst: np.ndarray,
    comp_mag_inst: dict[str, np.ndarray],
    comp_catalog_mag: dict[str, float],
    comp_quality: dict[str, dict],
    *,
    comp_likely_saturated: dict[str, np.ndarray] | None = None,
    n_comp_min: int = 3,
    weights_out: dict[str, float] | None = None,
    levels_out: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Step 4: ensemble (artificial comparison star) per frame - Broeg et al. (2005).

    ENSEMBLE-BROEG-2026-10:

    * Membership is exactly the comparison-star selection set (``comp_quality`` keys), identical for every
      frame. Each comp is normalised to its own night median, y_jf = m_jf - median_f(m_j),
      so a comp missing on a frame does not shift the level (the weighted mean of the
      present normalised comps is unbiased).
    * Weights w_j = 1/sigma_j^2 from :func:`broeg_weights` (LOO, iterated, no clipping).
    * Frame zero point Z_f = sum_present w_j y_jf / sum_present w_j. A frame with fewer
      than ``n_comp_min`` measured comps gives NaN (not exported).
    * Absolute level C = sum_all w_j (cat_j - median(m_j)) / sum_all w_j;
      mag_calib = m_target - Z_f + C;
      delta_mag = m_target - (Z_f + sum_all w_j median(m_j) / sum_all w_j).
    * ensemble_scatter = error of the weighted ensemble on that frame,
      sqrt(1 / sum_present (1/sigma_j^2)) - combined in quadrature with the target error.
    * ``comp_likely_saturated`` (catalog_id -> bool per frame): that comp epoch is not used.

    Returns (mag_calib, delta_mag, ensemble_scatter), arrays of length n_frames.
    """
    t = np.asarray(target_mag_inst, dtype=np.float64)
    n_frames = int(len(t))
    mag_calib = np.full(n_frames, float("nan"))
    delta_mag = np.full(n_frames, float("nan"))
    ensemble_scatter = np.full(n_frames, float("nan"))
    comp_likely_saturated = comp_likely_saturated or {}

    selected = sorted(comp_quality.keys(), key=str) if comp_quality else sorted(comp_mag_inst.keys(), key=str)
    good_ids = [c for c in selected if c in comp_mag_inst]  # INV-COMP-MEMBERSHIP: good_ids = selected
    if not good_ids:
        log_event("ensemble_normalize: no valid comp stars - returning all-NaN LC")
        return mag_calib, delta_mag, ensemble_scatter

    m = np.full((len(good_ids), n_frames), np.nan, dtype=np.float64)
    n_sat_excluded = 0
    for j, cid in enumerate(good_ids):
        arr = np.asarray(comp_mag_inst[cid], dtype=np.float64).reshape(-1)
        k = min(n_frames, int(arr.size))
        m[j, :k] = arr[:k]
        sat = comp_likely_saturated.get(cid)
        if sat is not None:
            sa = np.asarray(sat, dtype=bool).reshape(-1)[:k]
            n_sat_excluded += int(np.count_nonzero(sa & np.isfinite(m[j, :k])))
            m[j, :k] = np.where(sa, np.nan, m[j, :k])

    med = np.array([float(np.median(r[np.isfinite(r)])) if np.isfinite(r).any() else np.nan for r in m])
    keep = np.isfinite(med)
    if not keep.any():
        log_event("ensemble_normalize: comps have no finite measurements - all-NaN LC")
        return mag_calib, delta_mag, ensemble_scatter
    ids = [c for c, k in zip(good_ids, keep, strict=False) if k]
    m = m[keep]
    med = med[keep]
    # Per-comp reference levels M_j consistent with a common frame term E_f
    # (m_jf = M_j + E_f, Honeycutt 1992): a comp missing on some frames must not get a
    # reference biased by WHICH frames it has. Alternate Broeg weights <-> levels.
    fin_m = np.isfinite(m)
    m0 = np.where(fin_m, m, 0.0)
    for _outer in range(50):
        y = m - med[:, None]
        w, n_it = broeg_weights(y)
        wf_m = w[:, None] * fin_m
        sw_m = wf_m.sum(axis=0)
        E = np.where(sw_m > 0, (wf_m * (m0 - med[:, None])).sum(axis=0) / np.where(sw_m > 0, sw_m, 1.0), np.nan)
        if np.isfinite(E).any():
            E = E - float(np.nanmedian(E))  # fix the gauge (common offset of all levels)
        new_med = np.array(
            [
                float(np.median((m[j] - E)[fin_m[j] & np.isfinite(E)]))
                if np.any(fin_m[j] & np.isfinite(E))
                else med[j]
                for j in range(len(ids))
            ]
        )
        shift = float(np.max(np.abs(new_med - med))) if new_med.size else 0.0
        med = new_med
        if shift < 1e-7:
            break
    y = m - med[:, None]
    w, n_it = broeg_weights(y)
    # inverse-variance scale for the ensemble error: w_abs = 1/sigma^2 (before normalisation)
    sig = np.full(len(ids), np.nan)
    for j in range(len(ids)):
        if len(ids) >= 2:
            oth = np.delete(np.arange(len(ids)), j)
            fin_o = np.isfinite(y[oth])
            wo = w[oth][:, None] * fin_o
            den = wo.sum(axis=0)
            ens = np.where(den > 0, (wo * np.where(fin_o, y[oth], 0.0)).sum(axis=0) / np.where(den > 0, den, 1.0), np.nan)
            r = y[j] - ens
        else:
            r = y[j]
        r = r[np.isfinite(r)]
        sig[j] = float(np.std(r, ddof=1)) if r.size >= 3 else np.nan
    inv_var = np.where(np.isfinite(sig) & (sig > 0), 1.0 / (sig * sig), np.nan)

    cat = np.array([float(comp_catalog_mag.get(c, float("nan"))) for c in ids], dtype=np.float64)
    ok_cat = np.isfinite(cat)
    if ok_cat.any():
        C = float(np.sum(w[ok_cat] * (cat[ok_cat] - med[ok_cat])) / np.sum(w[ok_cat]))
    else:
        C = float("nan")
    inst_level = float(np.sum(w * med) / np.sum(w))

    fin = np.isfinite(y)
    n_present = fin.sum(axis=0)
    wf = w[:, None] * fin
    sw = wf.sum(axis=0)
    Z = np.where(sw > 0, (wf * np.where(fin, y, 0.0)).sum(axis=0) / np.where(sw > 0, sw, 1.0), np.nan)
    iv = np.where(fin & np.isfinite(inv_var)[:, None], inv_var[:, None], 0.0).sum(axis=0)
    n_min = max(1, min(int(n_comp_min), len(ids)))
    frame_ok = (n_present >= n_min) & np.isfinite(t) & np.isfinite(Z)
    delta_mag = np.where(frame_ok, t - (Z + inst_level), np.nan)
    if math.isfinite(C):
        mag_calib = np.where(frame_ok, t - Z + C, np.nan)
    else:
        mag_calib = np.where(frame_ok, delta_mag + float(np.nanmedian(cat)) if ok_cat.any() else np.nan, np.nan)
    ensemble_scatter = np.where(frame_ok & (iv > 0), np.sqrt(1.0 / np.where(iv > 0, iv, 1.0)), np.where(frame_ok, 0.0, np.nan))

    if weights_out is not None:
        weights_out.clear()
        weights_out.update({c: float(v) for c, v in zip(ids, w, strict=False)})
    if levels_out is not None:
        # Honeycutt reference level M_j of every comp (the one the zero point C uses)
        levels_out.clear()
        levels_out.update({c: float(v) for c, v in zip(ids, med, strict=False)})
    n_short = int(np.count_nonzero(np.isfinite(t) & (n_present < n_min)))
    logging.info(
        "[ENSEMBLE-BROEG] n_comp=%d iter=%d w=[%s] frames_without_%d_comps=%d sat_epochs_excluded=%d",
        len(ids),
        n_it,
        ", ".join(f"{v:.3f}" for v in w),
        n_min,
        n_short,
        n_sat_excluded,
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
) -> tuple[np.ndarray, float, float]:
    """Apply a constant colour correction to the calibrated light curve.

    Caller must pass the ZP ensemble maps (``ct_ensemble_reference_maps``), not
    the ``comparison_stars.csv`` export pool.

    Formula:
      bp_rp_comp_ref = median(bp_rp) of the usable ensemble comps
      ct_correction  = c1 * (target_bp_rp - bp_rp_comp_ref)
      mag_calib_ct   = mag_calib + ct_correction

    The correction is a **per-target / per-draft constant** (no airmass / epoch dependence).

    Returns: (mag_calib_ct, ct_correction, bp_rp_comp_ref)
    """
    if mag_calib is None:
        return np.asarray([], dtype=np.float64), 0.0, float("nan")
    base = np.asarray(mag_calib, dtype=np.float64)
    if (not math.isfinite(float(c1))) or float(c1) == 0.0 or (not math.isfinite(float(target_bp_rp))):
        return base.copy(), 0.0, float("nan")

    usable = [cid for cid, q in comp_quality.items() if q.get("quality") in ("good", "suspect")]
    bps = [float(comp_bp_rp.get(cid, float("nan"))) for cid in usable]
    bps = [b for b in bps if math.isfinite(b)]
    if not bps:
        return base.copy(), 0.0, float("nan")
    bp_ref = float(np.median(np.asarray(bps, dtype=np.float64)))
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
            "[COLOR TERM] Target %s BP-RP=%.3f is outside the comp range [%.3f, %.3f] - color-term extrapolation may cause systematics.",
            str(target_name or ""),
            float(tgt),
            float(min_bprp),
            float(max_bprp),
        )
    return in_range

# OUTPUT-REVIEW-2026-10 step 6 (Milan 2026-10-21): the light curve holds the per-epoch values only.
# Per-target constants (colour term, k'') are in photometry_summary.csv, per-night ones in
# pipeline_meta.json, the aperture choice in aperture_per_target.json.
LC_PUBLISHED_MAG_COLUMNS: tuple[str, ...] = ("mag_calib_ct", "mag_calib_final", "mag_calib")


def published_mag_column(df: pd.DataFrame) -> str | None:
    """The calibrated magnitude a light curve publishes: ``mag_calib_ct`` (colour term applied,
    written when the setup uses a colour term), else ``mag_calib``; ``mag_calib_final`` of drafts
    made before OUTPUT-REVIEW step 6 (equal to their ``mag_calib_ct``)."""
    for col in LC_PUBLISHED_MAG_COLUMNS:
        if col in df.columns:
            return col
    return None


def save_lightcurve_csv(
    output_path: Path,
    *,
    bjd: np.ndarray,
    hjd: np.ndarray,
    jd: np.ndarray,
    airmass: np.ndarray,
    mag_inst: np.ndarray,
    mag_calib: np.ndarray,
    delta_mag: np.ndarray,
    err: np.ndarray,
    aperture_r_px: np.ndarray,
    flags: list[str],
    source_files: list[str],
    mag_calib_ct: np.ndarray | None = None,
    mag_gspc: np.ndarray | None = None,
    flag_reasons: list[str] | None = None,
    time_base: str = TIME_BASE_BJD_TDB,
    err_photon: np.ndarray | None = None,
    err_sem_rel: np.ndarray | None = None,
    err_scint_rel: np.ndarray | None = None,
    err_sigma_sys_rel: np.ndarray | None = None,
    k2_coeff: float = float("nan"),
) -> None:
    """Write ``lightcurve_<id>.csv`` (one row per epoch).

    ``k2_delta_mag`` = ``k2_coeff`` x airmass is the second-order extinction term subtracted from the
    magnitudes (``k2_coeff`` = k'' x (target - comparison colour)); written only when k'' was applied.

    ``mag_calib_ct`` (``mag_calib`` + colour-term correction) is written only when given: the
    setup applies a colour term (photometric filter). ``mag_gspc`` only when the GSPC zero point
    exists. ``time_base`` labels the BJD/HJD path (``BJD_TDB`` vs ``JD_FALLBACK``).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = int(len(bjd))

    def _flag_cell(f: Any) -> str:
        if f is None:
            return ""
        s = str(f).strip()
        return s if s.lower() not in ("none", "nan") else ""

    def _col(a: np.ndarray, nd: int) -> np.ndarray:
        return np.round(np.asarray(a, dtype=np.float64), nd)

    cols: dict[str, Any] = {
        "bjd": bjd,
        "hjd": hjd,
        "jd": jd,
        "time_base": [str(time_base or TIME_BASE_BJD_TDB)] * n,
        "airmass": airmass,
        "mag_inst": _col(mag_inst, 6),
        "mag_calib": _col(mag_calib, 6),
    }
    if mag_calib_ct is not None:
        if len(mag_calib_ct) != n:
            raise ValueError(f"mag_calib_ct length {len(mag_calib_ct)} != LC length {n}")
        cols["mag_calib_ct"] = _col(mag_calib_ct, 6)
    if mag_gspc is not None and len(mag_gspc) == n:
        # GSPC-ZP-2026-10: second scale (Gaia DR3 synthetic photometry zero point).
        cols["mag_gspc"] = _col(mag_gspc, 6)
    cols["delta_mag"] = _col(delta_mag, 6)
    cols["err"] = _col(err, 6)
    for name, arr in (
        ("err_photon", err_photon),
        ("err_sem_rel", err_sem_rel),
        ("err_scint_rel", err_scint_rel),
        ("err_sigma_sys_rel", err_sigma_sys_rel),
    ):
        if arr is not None:
            cols[name] = _col(arr, 6)
    if math.isfinite(float(k2_coeff)):
        cols["k2_delta_mag"] = _col(float(k2_coeff) * np.asarray(airmass, dtype=np.float64), 6)
    cols["aperture_r_px"] = _col(aperture_r_px, 3)
    cols["flag"] = [_flag_cell(f) for f in flags]
    cols["flag_reason"] = [
        _flag_cell(r)
        for r in (flag_reasons if flag_reasons is not None and len(flag_reasons) == n else [""] * n)
    ]
    cols["source_file"] = source_files
    pd.DataFrame(cols).to_csv(output_path, index=False)

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
    """Save a PNG light-curve plot with colour-coded flags and a comp status table."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        logging.warning("[APERTURE] matplotlib is not available, no PNG is generated")
        return

    y_data = delta_mag if (delta_mag_mode and delta_mag is not None) else mag_calib
    y_label = "Deltamag (ensemble)" if delta_mag_mode else "mag_calib"

    flag_colors = {
        "normal": "#1a1a2e",
        "saturated": "#aaaaaa",
        "outlier_hi": "#ff6b35",
        "outlier_lo": "#7b2d8b",
        "no_data": "#cccccc",
        # FIX-PNGFLAGS-2026-10: these flags were missing, so their points vanished from the PNG
        "low_snr": "#c0392b",
        "high_err": "#c0392b",
        "frame_qc": "#c0392b",
        "artifact": "#c0392b",
        "spike_unconfirmed": "#c0392b",
    }

    fig, (ax_lc, ax_comp) = plt.subplots(
        1,
        2,
        figsize=(14, 5),
        gridspec_kw={"width_ratios": [3, 1]},
    )
    fig.suptitle(f"VYVAR - {target_name}", fontsize=11, fontweight="bold")

    bjd_plot_all, bjd_axis_int = jd_series_relative(bjd)

    # Light curve
    for flag, color in flag_colors.items():
        mask = np.array([f == flag for f in flags])
        if not mask.any():
            continue
        bjd_f = bjd_plot_all[mask]
        y_f = y_data[mask]
        err_f = MAG_ERR_SCALE * err[mask]  # FIX-ERRUNIT-2026-10: relative flux err -> mag
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

    # Comp quality table
    ax_comp.axis("off")
    comp_lines = []
    for i, (_, info) in enumerate(comp_quality.items(), 1):
        q = str(info["quality"])
        p2p = info.get("rms_p2p", float("nan"))
        icon = "[OK]" if q == "good" else ("[??]" if q == "suspect" else "[X]")
        p2p_str = f"{p2p:.4f}" if math.isfinite(float(p2p)) else "N/A"
        comp_lines.append(f"{icon} C{i:02d}  {p2p_str}  {q}")

    comp_text = "\n".join(comp_lines[:15])  # max 15 lines
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

#: active_targets.csv written before TARGET-RULES-2026-10 (an aperture-photometry-only re-run of an old draft)
_LEGACY_SELECTION_SKIP = {"zone_flag": "saturated", "saturovany ciel": "saturated", "likely_saturated": "saturated",
                          "below_target_depth": "below_snr_min", "zone_noise": "below_snr_min"}


def _target_selection_skip_reason(row: Any) -> str:
    """The target-selection skip of an active_targets row ("" = measurable).

    Target-selection cuts and per_frame_saturation are kept; a MASTERSTAR-saturated zone is the upper
    cut even when an old run cleared it; reasons written by aperture photometry for other purposes
    (e.g. ``no_comps``) are not selection skips."""
    from photometry_core import FINAL_SKIP_REASONS  # noqa: PLC0415

    sr = str(row.get("skip_reason", "") or "").strip()
    sr = _LEGACY_SELECTION_SKIP.get(sr, sr)
    if sr in FINAL_SKIP_REASONS or sr == "per_frame_saturation":
        return sr
    if str(row.get("zone_flag", "") or "").strip().lower() == "saturated":
        return "saturated"
    return ""


def decide_per_frame_saturation(
    *,
    skip_reason: str,
    frame_saturated: list[bool] | tuple[bool, ...] | np.ndarray | None,
    min_clean_frac: float = 0.5,
) -> dict[str, Any]:
    """SAT-TARGET-2026-10 / TARGET-RULES-2026-10: per-frame saturation of a measurable target.

    A target already skipped in target selection (saturated / below_snr_min / out of scope) or by an
    earlier run keeps its skip - nothing revives it. A target linear on MASTERSTAR is not
    measured when it is saturated in more than (1 - min_clean_frac) of its frames
    (``per_frame_saturation``). Without per-frame peak data there is no per-frame verdict.
    """
    recorded = str(skip_reason or "").strip()
    flags = [bool(x) for x in list(frame_saturated)] if frame_saturated is not None else []
    n = len(flags)
    n_clean = int(sum(1 for f in flags if not f))
    clean_frac = float(n_clean) / float(n) if n > 0 else float("nan")
    thr = float(min_clean_frac)
    thr = max(0.1, min(1.0, thr)) if math.isfinite(thr) else 0.5
    if recorded:
        skip, reason = True, recorded
    elif n > 0 and clean_frac < thr:
        skip, reason = True, "per_frame_saturation"
    else:
        skip, reason = False, ""
    return {"skip_photometry": skip, "skip_reason": reason, "sat_clean_frac": clean_frac,
            "n_frames": n, "n_clean": n_clean}


def _per_frame_sat_flags_for_catalog_id(
    catalog_id: str,
    csv_files: list[Path],
    csv_cache: dict[str, Any],
    *,
    sat_limit_adu: float | None = None,
    peak_test_adu: float | None = None,
    id_pos_cache: dict[str, dict[str, int]] | None = None,
) -> list[bool] | None:
    """Return per-frame saturation bools for a target, or None if data unavailable.

    ``id_pos_cache`` (SPEED-2026-10 S1): path -> {normalised id -> first row position}, shared by the
    targets of one run so each frame's ids are normalised once.

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
        _key = str(path)
        _pos = id_pos_cache.get(_key) if id_pos_cache is not None else None
        if _pos is None:
            _pos = {}
            for _i, _c in enumerate(df["catalog_id"].astype(str).map(_normalize_gaia_id).tolist()):
                _pos.setdefault(_c, _i)
            if id_pos_cache is not None:
                id_pos_cache[_key] = _pos
        _ip = _pos.get(cid)
        if _ip is None:
            continue
        row = df.iloc[_ip]
        n_matched += 1
        peak = float(pd.to_numeric(row.get("peak_max_adu"), errors="coerce"))
        # SAT-LIN-2026-10: the frame's own verdict (raw peak >= LINEARITY_FRAC x SAT from
        # sat_diag) wins; then the frame's limit column; then the catalog peak-test.
        if sat_limit_adu is None:
            if "likely_saturated" in df.columns and pd.notna(row.get("likely_saturated")):
                flags.append(_coerce_bool_cell(row.get("likely_saturated")))
                continue
            _l85 = float(pd.to_numeric(row.get("saturate_limit_adu_85pct"), errors="coerce"))
            if math.isfinite(peak) and math.isfinite(_l85) and _l85 > 0:
                flags.append(bool(peak >= _l85))
                continue
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
    from pipeline_catalog import inv_sat_limit_peak_test_adu  # noqa: PLC0415

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
    min_clean_frac: float,
    peak_test_adu: float | None = None,
    peak_test_source: str = "",
) -> dict[str, Any]:
    """Mutate ``at_df`` skip columns with the per-frame saturation rule; return night meta.

    Only targets without a recorded skip are tested (decide_per_frame_saturation)."""
    pt, pt_src, container, container_src = _resolve_pfs_peak_test(
        peak_test_adu=peak_test_adu,
        peak_test_source=peak_test_source,
        sat_limit_adu=None,
    )
    meta: dict[str, Any] = {
        "per_frame_sat_min_clean_frac": float(min_clean_frac),
        "per_frame_sat_n_targets": int(len(at_df)),
        "per_frame_sat_n_skipped": 0,
        "per_frame_sat_peak_test_adu": float(pt) if pt is not None else float("nan"),
        "per_frame_sat_peak_test_source": str(pt_src),
        "per_frame_sat_container_clip_adu": float(container),
        "per_frame_sat_container_clip_source": str(container_src),
    }
    if "skip_reason" not in at_df.columns:
        at_df["skip_reason"] = ""
    # NaN read from CSV is not a reason (bool(NaN) is True -> "nan" was written as a reason)
    at_df["skip_reason"] = at_df["skip_reason"].fillna("").astype(str).str.strip().astype(object)
    at_df["skip_reason"] = at_df["skip_reason"].where(at_df["skip_reason"].str.lower() != "nan", "")

    _id_pos: dict[str, dict[str, int]] = {}
    for idx, row in at_df.iterrows():
        recorded = _target_selection_skip_reason(row)
        flags = None
        if not recorded:
            flags = _per_frame_sat_flags_for_catalog_id(
                _normalize_gaia_id(row.get("catalog_id", "")),
                csv_files,
                csv_cache,
                peak_test_adu=pt,
                id_pos_cache=_id_pos,
            )
        dec = decide_per_frame_saturation(
            skip_reason=recorded, frame_saturated=flags, min_clean_frac=float(min_clean_frac)
        )
        at_df.at[idx, "skip_photometry"] = bool(dec["skip_photometry"])
        at_df.at[idx, "skip_reason"] = str(dec["skip_reason"] or "")
        if dec["skip_photometry"] and str(dec["skip_reason"]) == "per_frame_saturation" and not recorded:
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

def _build_aperture_resolved_facts(
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


def target_airmass(
    jd_array: np.ndarray,
    ra_deg: float,
    dec_deg: float,
    cfg: AppConfig,
    site: tuple[float | None, float | None, float | None] | None = None,
    fallback: np.ndarray | None = None,
) -> np.ndarray:
    """EXPORT-AUDIT-2026-10 (Milan 2026-10-21): airmass of the target itself per epoch
    (Kasten & Young 1989 at the target's altitude for the frame's mid-exposure JD and the site of
    BJD/HJD). The frame value (field centre) differs by up to +-0.03 on a 5.7 deg field. Epochs
    without a result keep ``fallback`` (the frame value)."""
    jd_arr = np.asarray(jd_array, dtype=float)
    out = (np.asarray(fallback, dtype=float).copy() if fallback is not None and len(fallback) == len(jd_arr)
           else np.full(len(jd_arr), np.nan))
    if site is not None and site[0] is not None and site[1] is not None:
        lat, lon = float(site[0]), float(site[1])
        alt = float(site[2]) if site[2] is not None else 0.0
    else:
        lat, lon, alt = float(cfg.observer_lat), float(cfg.observer_lon), float(cfg.observer_alt_m)
    from param_resolver import is_null_island_coords  # noqa: PLC0415

    finite = np.isfinite(jd_arr)
    if not (math.isfinite(ra_deg) and math.isfinite(dec_deg)) or is_null_island_coords(lat, lon) or not finite.any():
        return out
    try:
        import astropy.units as u
        from astropy.coordinates import AltAz, EarthLocation, SkyCoord
        from astropy.time import Time

        from pipeline_catalog import _airmass_from_altitude_deg  # noqa: PLC0415

        location = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=alt * u.m)
        t = Time(jd_arr[finite], format="jd", scale="utc", location=location)
        alt_deg = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs").transform_to(
            AltAz(obstime=t, location=location)).alt.deg
        am = np.array([_airmass_from_altitude_deg(a) if 5.0 <= a <= 90.0 else np.nan
                       for a in np.atleast_1d(alt_deg)], dtype=float)
        idx = np.flatnonzero(finite)
        ok = np.isfinite(am)
        out[idx[ok]] = np.round(am[ok], 6)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("AIRMASS-PERTARGET: failed (%s) - frame airmass kept", exc)
    return out


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
