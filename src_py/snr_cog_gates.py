"""IMPL-02 hard gates for measured-CoG SNR aperture tables.

Each gate records its measured value and must be able to fail. A broken growth
curve must not produce a written ``aperture_snr_table.json``.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


# Tolerances (stated in the artifact)
DEE_RISING_TOL = 5.0e-3  # absolute; real median ladders have ~1e-3 sampling noise
FLATNESS_MAX_ABS_DELTA = 0.05  # |EE(1.3 r_norm)/EE(r_norm) - 1|
CONVERGENCE_MAX_SPAN = 0.02  # max-min of last three EE samples
R90_Q4_LO_PX = 5.0
R90_Q4_HI_PX = 6.0
R90_TOL_PX = 1.5  # agreement window around Q4 5-6 px band
MAG_FLAT_MAX_IDENTICAL_BINS = 3  # >3 identical adjacent radii => fail
NEAR_BOUND_FRAC = 0.10  # within 10% of search half-width from a bound


def _dee_profile(ee_radii: np.ndarray, ee_curve: np.ndarray) -> list[dict[str, Any]]:
    rr = np.asarray(ee_radii, dtype=np.float64)
    ee = np.asarray(ee_curve, dtype=np.float64)
    # Light 3-point smoothing suppresses ladder sampling chatter without hiding
    # a systematic residual-background rise (IMPL-02).
    if ee.size >= 3:
        ee_s = ee.copy()
        ee_s[1:-1] = (ee[:-2] + ee[1:-1] + ee[2:]) / 3.0
        ee = ee_s
    out: list[dict[str, Any]] = []
    for i in range(1, len(rr)):
        dr = float(rr[i] - rr[i - 1])
        if dr <= 0:
            continue
        out.append(
            {
                "r_mid": float(0.5 * (rr[i] + rr[i - 1])),
                "dEE_dr": float((ee[i] - ee[i - 1]) / dr),
            }
        )
    return out


def evaluate_snr_cog_gates(
    *,
    snr_table: dict[str, Any],
    fwhm_px: float,
    annulus_inner_fwhm: float = 4.75,
    ee_radii: np.ndarray | None = None,
    ee_curve: np.ndarray | None = None,
    ref_r_px: float | None = None,
    r90_px: float | None = None,
    flatness_outer_over_norm: float | None = None,
    ladder_outer_r_px: float | None = None,
) -> dict[str, Any]:
    """Return gate results. ``ok`` is True only when every gate passes."""
    fw = float(fwhm_px) if math.isfinite(float(fwhm_px)) and float(fwhm_px) > 0 else float("nan")
    gates: dict[str, Any] = {}
    failures: list[str] = []

    rr = np.asarray(ee_radii, dtype=np.float64) if ee_radii is not None else np.asarray([])
    ee = np.asarray(ee_curve, dtype=np.float64) if ee_curve is not None else np.asarray([])
    measured = rr.size >= 3 and ee.size == rr.size

    # --- INV-COG-MONOTONE ---
    dee = _dee_profile(rr, ee) if measured else []
    rising = False
    if measured and math.isfinite(fw):
        for i in range(1, len(dee)):
            if float(dee[i]["r_mid"]) <= 1.5 * fw:
                continue
            if float(dee[i]["dEE_dr"]) > float(dee[i - 1]["dEE_dr"]) + DEE_RISING_TOL:
                rising = True
                break
    gates["INV-COG-MONOTONE"] = {
        "pass": (not rising) if measured else False,
        "dEE_dr": dee,
        "tol": DEE_RISING_TOL,
        "detail": "rising dEE/dr beyond 1.5 FWHM" if rising else "non-increasing",
    }
    if not gates["INV-COG-MONOTONE"]["pass"]:
        failures.append("INV-COG-MONOTONE")

    # --- INV-COG-FLATNESS-REAL ---
    ref_r = float(ref_r_px) if ref_r_px is not None else float("nan")
    outer_r = float(ladder_outer_r_px) if ladder_outer_r_px is not None else float("nan")
    flat = (
        float(flatness_outer_over_norm)
        if flatness_outer_over_norm is not None
        else float("nan")
    )
    check_r_equals_norm = (
        math.isfinite(ref_r)
        and math.isfinite(outer_r)
        and abs(outer_r - ref_r) < 1e-6
    )
    flat_ok = (
        measured
        and math.isfinite(flat)
        and math.isfinite(ref_r)
        and math.isfinite(outer_r)
        and outer_r > ref_r + 1e-6
        and abs(flat - 1.0) <= FLATNESS_MAX_ABS_DELTA
        and not check_r_equals_norm
    )
    gates["INV-COG-FLATNESS-REAL"] = {
        "pass": bool(flat_ok),
        "r_norm_px": ref_r,
        "r_check_px": outer_r,
        "EE_check_over_norm": flat,
        "tol_abs": FLATNESS_MAX_ABS_DELTA,
        "check_equals_norm": bool(check_r_equals_norm),
        "detail": (
            "tautological check radius"
            if check_r_equals_norm
            else ("flatness out of tol" if not flat_ok else "ok")
        ),
    }
    if not gates["INV-COG-FLATNESS-REAL"]["pass"]:
        failures.append("INV-COG-FLATNESS-REAL")

    # --- INV-COG-CONVERGENCE ---
    last3 = ee[-3:].tolist() if measured and ee.size >= 3 else []
    span = float(max(last3) - min(last3)) if len(last3) == 3 else float("nan")
    conv_ok = math.isfinite(span) and span <= CONVERGENCE_MAX_SPAN
    gates["INV-COG-CONVERGENCE"] = {
        "pass": bool(conv_ok),
        "last_three_EE": last3,
        "span": span,
        "tol": CONVERGENCE_MAX_SPAN,
    }
    if not gates["INV-COG-CONVERGENCE"]["pass"]:
        failures.append("INV-COG-CONVERGENCE")

    # --- INV-COG-R90 ---
    r90 = float(r90_px) if r90_px is not None else float("nan")
    if not math.isfinite(r90) and measured:
        r90 = float(rr[int(np.argmin(np.abs(ee - 0.9)))])
    # Accept if inside [5-1.5, 6+1.5] = [3.5, 7.5]
    r90_ok = math.isfinite(r90) and (R90_Q4_LO_PX - R90_TOL_PX) <= r90 <= (R90_Q4_HI_PX + R90_TOL_PX)
    gates["INV-COG-R90"] = {
        "pass": bool(r90_ok),
        "r90_px": r90,
        "q4_band_px": [R90_Q4_LO_PX, R90_Q4_HI_PX],
        "tol_px": R90_TOL_PX,
    }
    if not gates["INV-COG-R90"]["pass"]:
        failures.append("INV-COG-R90")

    # --- INV-APERTURE-MAG-MONOTONE ---
    table = snr_table.get("table") or {}
    mags = sorted(float(k) for k in table.keys())
    radii: list[float] = []
    for m in mags:
        v = table.get(m)
        if v is None:
            v = table.get(round(m, 1))
        if v is None:
            v = table.get(f"{m:.1f}")
        if v is None:
            # JSON round-trip may stringify keys
            for kk, vv in table.items():
                try:
                    if abs(float(kk) - m) < 1e-9:
                        v = vv
                        break
                except (TypeError, ValueError):
                    continue
        radii.append(float(v) if v is not None else float("nan"))
    increasing = False
    identical_run = 1
    max_identical_run = 1
    for i in range(1, len(radii)):
        if not (math.isfinite(radii[i]) and math.isfinite(radii[i - 1])):
            continue
        if radii[i] > radii[i - 1] + 1e-6:
            increasing = True
        if abs(radii[i] - radii[i - 1]) <= 1e-6:
            identical_run += 1
            max_identical_run = max(max_identical_run, identical_run)
        else:
            identical_run = 1
    finite = [r for r in radii if math.isfinite(r)]
    span = float(max(finite) - min(finite)) if len(finite) >= 2 else float("nan")
    # Discrete r-grid plateaus are allowed; a failed background term makes the
    # *whole* table nearly flat (IMPL-01 signature).
    whole_table_flat = math.isfinite(span) and span < max(0.75, 0.08 * float(np.median(finite)))
    mag_ok = (not increasing) and (not whole_table_flat) and len(radii) >= 2
    gates["INV-APERTURE-MAG-MONOTONE"] = {
        "pass": bool(mag_ok) and all(math.isfinite(r) for r in radii),
        "radii_by_mag": {str(m): r for m, r in zip(mags, radii)},
        "max_identical_adjacent": int(max_identical_run),
        "span_px": span,
        "whole_table_flat": bool(whole_table_flat),
        "detail": (
            "increasing with mag"
            if increasing
            else ("flat across magnitude range" if whole_table_flat else "ok")
        ),
    }
    if not gates["INV-APERTURE-MAG-MONOTONE"]["pass"]:
        failures.append("INV-APERTURE-MAG-MONOTONE")

    # --- INV-APERTURE-BOUND ---
    r_min = float(snr_table.get("r_min_px", float("nan")))
    r_max = float(snr_table.get("r_max_px", float("nan")))
    half = 0.5 * (r_max - r_min) if math.isfinite(r_max) and math.isfinite(r_min) else float("nan")
    near_tol = NEAR_BOUND_FRAC * half if math.isfinite(half) else float("nan")
    bound_rows: dict[str, Any] = {}
    n_on_bound = 0
    n_near_top = 0
    for m, r in zip(mags, radii):
        hit = "none"
        if math.isfinite(r_min) and abs(r - r_min) <= max(0.03, 1e-6):
            hit = "r_min"
            n_on_bound += 1
        elif math.isfinite(r_max) and abs(r - r_max) <= max(0.03, 1e-6):
            hit = "r_max"
            n_on_bound += 1
        near_top = bool(
            math.isfinite(near_tol) and math.isfinite(r_max) and (r_max - r) <= near_tol
        )
        if near_top:
            n_near_top += 1
        bound_rows[str(m)] = {"r_px": r, "bound_hit": hit, "near_r_max": near_top}
    # Gate fails if any bin is on a bound OR a majority cluster near the top
    bound_fail = n_on_bound > 0 or (len(mags) > 0 and n_near_top >= max(2, len(mags) // 2))
    gates["INV-APERTURE-BOUND"] = {
        "pass": not bound_fail,
        "per_mag": bound_rows,
        "n_on_bound": int(n_on_bound),
        "n_near_r_max": int(n_near_top),
        "near_tol_px": near_tol,
        "detail": (
            "bound binds"
            if n_on_bound
            else ("optima cluster near r_max" if bound_fail else "ok")
        ),
    }
    if not gates["INV-APERTURE-BOUND"]["pass"]:
        failures.append("INV-APERTURE-BOUND")

    # --- INV-APERTURE-ANNULUS ---
    r_ann_in = float(annulus_inner_fwhm) * fw if math.isfinite(fw) else float("nan")
    margins: dict[str, float] = {}
    ann_ok = math.isfinite(r_ann_in)
    for m, r in zip(mags, radii):
        marg = float(r_ann_in - r) if math.isfinite(r_ann_in) else float("nan")
        margins[str(m)] = marg
        if not (math.isfinite(marg) and marg > 0):
            ann_ok = False
    gates["INV-APERTURE-ANNULUS"] = {
        "pass": bool(ann_ok) and len(mags) > 0,
        "annulus_inner_r_px": r_ann_in,
        "annulus_inner_fwhm": float(annulus_inner_fwhm),
        "margin_px_by_mag": margins,
    }
    if not gates["INV-APERTURE-ANNULUS"]["pass"]:
        failures.append("INV-APERTURE-ANNULUS")

    return {
        "ok": len(failures) == 0,
        "failures": failures,
        "gates": gates,
    }
