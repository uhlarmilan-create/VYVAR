#!/usr/bin/env python3
"""T3 empirical acceptance: preprocess sky surface vs draft_429 archive Light_008."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline import (
    DAO_STAR_FINDER_NO_ROUNDNESS_FILTER,
    _fit_subtract_preprocess_sky_surface,
)

DRAFT = _ROOT / "Archive" / "Drafts" / "draft_000429"
MS = DRAFT / "detrended_aligned" / "lights" / "NoFilter_60_2" / "MASTERSTAR.fits"
CAL = DRAFT / "calibrated" / "lights" / "NoFilter_60_2" / "BO_CVn_Light_008.fits"
OUT = _ROOT / "tmp" / "t3_sky_surface_429"


def dao_pass1_count(img: np.ndarray, *, sigma: float = 2.1, fwhm: float = 2.5) -> int:
    arr = np.asarray(img, dtype=np.float32)
    finite = np.isfinite(arr)
    arr = np.where(finite, arr, np.nanmedian(arr[finite]) if finite.any() else 0.0)
    _, med, std = sigma_clipped_stats(arr, sigma=3.0, maxiters=3)
    data0 = np.nan_to_num((arr - med).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    thr = max(float(sigma) * float(std), 1e-6)
    finder = DAOStarFinder(
        fwhm=max(1.2, float(fwhm)),
        threshold=float(thr),
        brightest=None,
        **DAO_STAR_FINDER_NO_ROUNDNESS_FILTER,
    )
    tbl = finder(data0)
    return int(len(tbl)) if tbl is not None else 0


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if not CAL.is_file() or not MS.is_file():
        print(f"ERROR: missing archive frames under {DRAFT}", file=sys.stderr)
        return 1

    cal = np.asarray(fits.getdata(CAL), dtype=np.float64)
    ms = np.asarray(fits.getdata(MS), dtype=np.float64)
    proc, stats = _fit_subtract_preprocess_sky_surface(cal.astype(np.float32), order=2)

    residual = ms - proc.astype(np.float64)
    R_amp = float(np.max(np.abs(ms - cal)))
    resid_maxabs = float(np.max(np.abs(residual)))
    resid_pcts = {str(p): float(np.percentile(np.abs(residual), p)) for p in (50, 90, 95, 99)}

    sky_proc = float(np.median(proc))
    sky_ms = float(np.median(ms))
    delta_med = float(np.median(ms) - np.median(cal))

    dao_n = dao_pass1_count(proc, sigma=2.1, fwhm=2.5)
    dao_ms = dao_pass1_count(ms, sigma=2.1, fwhm=2.5)
    dao_cal = dao_pass1_count(cal, sigma=2.1, fwhm=2.5)

    report = {
        "cal_path": str(CAL),
        "ms_path": str(MS),
        "sky_surface_stats": stats,
        "residual_to_ms429": {
            "maxabs": resid_maxabs,
            "maxabs_fraction_of_transform": resid_maxabs / max(R_amp, 1.0),
            "abs_percentiles": resid_pcts,
            "median": float(np.median(residual)),
            "abs_p50": float(np.percentile(np.abs(residual), 50)),
            "abs_p99": float(np.percentile(np.abs(residual), 99)),
        },
        "transform_amplitude_maxabs_ms_minus_cal": R_amp,
        "sky_median_adu": {
            "cal": float(np.median(cal)),
            "ms429": sky_ms,
            "proc_simulated": sky_proc,
            "delta_median_ms_cal": delta_med,
            "delta_median_proc_cal": float(np.median(proc) - np.median(cal)),
        },
        "dao_pass1_sim": {
            "cal": dao_cal,
            "proc_simulated": dao_n,
            "ms429": dao_ms,
            "band_2500_3000": 2500 <= dao_n <= 3000,
        },
        "acceptance": {
            "residual_smooth_p99_lt_200": float(np.percentile(np.abs(residual), 99)) < 200.0,
            "dao_in_band": 2500 <= dao_n <= 3000,
            "note_sky_1478": "pipeline_meta sky_adu (~1478) is Labbe annulus metric, not frame median (~1860)",
            "note_dao_gap": "cal-only fit lands ~2571 vs 429 logged 2816; mask/clip residual documented",
        },
    }
    OUT.joinpath("acceptance.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    ok = bool(report["acceptance"]["residual_smooth_p99_lt_200"] and report["acceptance"]["dao_in_band"])
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
