#!/usr/bin/env python3
"""Step 1i: locate EE_target(1.916) numerator failure mechanism.

Diagnostic + E5 control recomputation only. No config change.

Usage:
  python dev/tools/closure_step1i_diagnose_numerator_mechanism.py \\
    --draft Archive/Drafts/draft_000435_snapshot_skysurface_20260716 \\
    --step1b-json tmp/closure_step1b_results.json \\
    --step1g-json tmp/closure_step1g_results.json \\
    --cache tmp/closure_step1f_ee_cache.npz \\
    --out tmp/closure_step1i_diagnostics.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from astropy import modeling
from astropy.io import fits
from astropy.wcs import WCS
from photutils.aperture import CircularAnnulus, CircularAperture, aperture_photometry
from scipy import stats

from closure_step1b_differential_aperture import TABLE_FWHM, _ee_at_radius, _load_all_proc
from closure_step1c_differential_aperture import PROXY_R_AP, _comp_subsets
from closure_step1e_differential_aperture import (
    ANNULUS_IN,
    ANNULUS_OUT,
    COG_RADII,
    FIT_BOX,
    NORM_RADIUS,
    _curve_of_growth_photutils,
    _gaussian_centroid_or_none,
)

warnings.filterwarnings("ignore", category=fits.verify.VerifyWarning)

R_AP = PROXY_R_AP
BETA = 3.0


def _ee_expected(r50: float, r: float = R_AP) -> float:
    alpha = r50 / math.sqrt(2.0 ** (1.0 / (BETA - 1.0)) - 1.0)
    ee_r = 1.0 - (1.0 + (r / alpha) ** 2) ** (1.0 - BETA)
    ee_12 = 1.0 - (1.0 + (12.0 / alpha) ** 2) ** (1.0 - BETA)
    return ee_r / ee_12 if ee_12 > 0 else float("nan")


def _gaussian_fit_diagnostics(
    data: np.ndarray,
    x_proc: float,
    y_proc: float,
    *,
    fwhm_hint: float,
    box: int = FIT_BOX,
) -> dict[str, Any] | None:
    h, w = data.shape
    xc_i, yc_i = int(round(x_proc)), int(round(y_proc))
    if not (box <= xc_i < w - box and box <= yc_i < h - box):
        return None
    cut = data[yc_i - box : yc_i + box + 1, xc_i - box : xc_i + box + 1].astype(np.float64)
    yy, xx = np.mgrid[0 : cut.shape[0], 0 : cut.shape[1]]
    bg = float(np.median(cut))
    amp0 = max(float(np.max(cut) - bg), 1.0)
    peak = float(np.max(cut))
    fitter = modeling.fitting.TRFLSQFitter()
    c0 = modeling.models.Const2D(amplitude=bg)
    g0 = modeling.models.Gaussian2D(
        amplitude=amp0,
        x_mean=float(box),
        y_mean=float(box),
        x_stddev=fwhm_hint / 2.355,
        y_stddev=fwhm_hint / 2.355,
        theta=0.0,
    )
    g0.x_mean.bounds = (box - 2, box + 2)
    g0.y_mean.bounds = (box - 2, box + 2)
    g0.x_stddev.bounds = (0.4, 15.0)
    g0.y_stddev.bounds = (0.4, 15.0)
    g0.theta.fixed = True
    try:
        mg = fitter(g0 + c0, xx, yy, cut, maxiter=200)
        model = mg(xx, yy)
        resid = cut - model
        dof = cut.size - 6
        s2 = float(np.sum(resid**2) / max(dof, 1))
        sigma = max(float(np.std(cut - bg)), 1.0)
        chi2_red = float(np.sum((resid / sigma) ** 2) / max(dof, 1))
        xi = xc_i - box + float(mg.x_mean_0.value)
        yi = yc_i - box + float(mg.y_mean_0.value)
        shift = math.hypot(xi - x_proc, yi - y_proc)
        fit_amp = float(mg.amplitude_0.value)
        return {
            "centroid_x": xi,
            "centroid_y": yi,
            "chi2_reduced": chi2_red,
            "centroid_shift_px": shift,
            "fit_amplitude": fit_amp,
            "peak_adu": peak,
            "fit_amp_over_peak": fit_amp / peak if peak > 0 else float("nan"),
            "fitted_fwhm_px": 0.5 * (2.355 * float(mg.x_stddev_0.value) + 2.355 * float(mg.y_stddev_0.value)),
            "resid_rms": float(np.sqrt(s2)),
        }
    except Exception:  # noqa: BLE001
        return None


def _cog_detail(
    data: np.ndarray,
    xc: float,
    yc: float,
) -> dict[str, Any] | None:
    h, w = data.shape
    max_r = float(np.max(COG_RADII)) + ANNULUS_OUT + 2.0
    if xc - max_r < 0 or yc - max_r < 0 or xc + max_r >= w or yc + max_r >= h:
        return None
    ann = CircularAnnulus([(xc, yc)], r_in=ANNULUS_IN, r_out=ANNULUS_OUT)
    ap_sum = float(aperture_photometry(data, ann)["aperture_sum"][0])
    ann_area = float(ann.area)
    sky_pp = ap_sum / ann_area
    m = ann.to_mask(method="center")[0]
    cut = m.cutout(data)
    if cut is not None:
        ann_vals = cut[m.data.astype(bool)]
    else:
        ann_vals = np.array([])
    ann_std = float(np.std(ann_vals)) if ann_vals.size else float("nan")

    flux: list[float] = []
    for r in COG_RADII:
        ap = CircularAperture([(xc, yc)], r=float(r))
        s = float(aperture_photometry(data, ap)["aperture_sum"][0])
        flux.append(s - sky_pp * ap.area)
    arr = np.asarray(flux, dtype=np.float64)
    if arr[-1] <= 0:
        return None
    ee_arr = arr / arr[-1]
    f_ap = float(np.interp(R_AP, COG_RADII, arr))
    ee_ap = float(_ee_at_radius(COG_RADII, ee_arr, R_AP))
    return {
        "sky_annulus_adu_per_px": sky_pp,
        "annulus_std_adu": ann_std,
        "annulus_area_px": ann_area,
        "F_1.916_adu": f_ap,
        "F_12_adu": float(arr[-1]),
        "EE_1.916": ee_ap,
    }


def _sources_in_annulus(
    xc: float,
    yc: float,
    frame_df: pd.DataFrame,
) -> int:
    if frame_df.empty:
        return 0
    dx = frame_df["x"].astype(float) - xc
    dy = frame_df["y"].astype(float) - yc
    r = np.hypot(dx, dy)
    return int(((r >= ANNULUS_IN) & (r <= ANNULUS_OUT)).sum())


def _cutout_stats(data: np.ndarray, xc: float, yc: float, half: int = 20) -> dict[str, Any]:
    xi, yi = int(round(xc)), int(round(yc))
    h, w = data.shape
    x0, x1 = max(0, xi - half), min(w, xi + half)
    y0, y1 = max(0, yi - half), min(h, yi + half)
    cut = data[y0:y1, x0:x1].astype(np.float64)
    return {
        "shape": list(cut.shape),
        "min_adu": float(np.min(cut)),
        "max_adu": float(np.max(cut)),
        "median_adu": float(np.median(cut)),
        "center_adu": float(data[yi, xi]) if 0 <= yi < h and 0 <= xi < w else float("nan"),
        "origin_xy": [x0, y0],
        "centre_offset_from_cut": [float(xc - xi), float(yc - yi)],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", type=Path, required=True)
    ap.add_argument("--step1b-json", type=Path, default=Path("tmp/closure_step1b_results.json"))
    ap.add_argument("--step1g-json", type=Path, default=Path("tmp/closure_step1g_results.json"))
    ap.add_argument("--cache", type=Path, default=Path("tmp/closure_step1f_ee_cache.npz"))
    ap.add_argument("--out", type=Path, default=Path("tmp/closure_step1i_diagnostics.json"))
    args = ap.parse_args()

    draft = args.draft.resolve()
    lights = draft / "detrended_aligned/lights/NoFilter_60_2"
    csvs, catalog_full, n_frames = _load_all_proc(lights)

    with args.step1b_json.open(encoding="utf-8") as f:
        s1b = json.load(f)
    with args.step1g_json.open(encoding="utf-8") as f:
        s1g = json.load(f)

    star_ids: list[str] = s1b["part_a"]["star_ids"]
    proxies: list[str] = s1g["proxies"]
    cat = catalog_full[catalog_full["catalog_id"].astype(str).isin(star_ids)].copy()
    cat_ra_dec: dict[str, tuple[float, float]] = {
        str(r["catalog_id"]): (float(r["ra_deg"]), float(r["dec_deg"]))
        for _, r in cat.iterrows()
        if math.isfinite(float(r.get("ra_deg", float("nan"))))
    }
    phot_g: dict[str, float] = {
        str(r["catalog_id"]): float(r["phot_g"]) for _, r in cat.iterrows()
    }

    npz = np.load(args.cache, allow_pickle=True)
    ee_cache = npz["ee_cache"].item()
    frame_names = list(npz["frame_names"])
    r50_series = list(npz["r50_series"])
    sky_med = list(npz["sky_med"])

    out: dict[str, Any] = {
        "proxies": proxies,
        "n_frames": n_frames,
        "E1": {},
        "E2": {},
        "E3": [],
        "E4": {},
        "E5": {},
    }

    # accumulators
    e1_rows: list[dict[str, Any]] = []
    e2_rows: list[dict[str, Any]] = []
    e4_rows: list[dict[str, Any]] = []
    e5_fit: dict[str, list[float]] = {p: [] for p in proxies}
    e5_wcs: dict[str, list[float]] = {p: [] for p in proxies}
    extreme: list[tuple[float, str, int, str]] = []  # |ee-expected|, pid, fi, frame

    for fi, proc in enumerate(csvs):
        fn = proc.name
        fits_path = lights / fn.replace(".csv", ".fits")
        df = pd.read_csv(proc, dtype={"catalog_id": str})
        df_idx = df.set_index("catalog_id")
        frame_df = df[["x", "y"]].copy()

        with fits.open(fits_path, memmap=False) as hdul:
            data = hdul[0].data.astype(np.float64)
            wcs = WCS(hdul[0].header)
            global_sky = float(np.median(data))

        r50_f = float(r50_series[fi])

        for pid in proxies:
            if pid not in ee_cache.get(fi, {}):
                continue
            if pid not in df_idx.index:
                continue
            row = df_idx.loc[pid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]

            dao_x, dao_y = float(row["x"]), float(row["y"])
            ra_c, dec_c = cat_ra_dec[pid]
            wcs_x, wcs_y = wcs.world_to_pixel_values(ra_c, dec_c)
            wcs_x, wcs_y = float(wcs_x), float(wcs_y)

            cached = ee_cache[fi][pid]
            fit_x, fit_y = float(cached["centroid_x"]), float(cached["centroid_y"])
            ee_meas = float(_ee_at_radius(cached["radii"], cached["ee"], R_AP))
            ee_exp = float(_ee_expected(r50_f))
            ee_dev = abs(ee_meas - ee_exp)

            sep_fit_wcs = math.hypot(fit_x - wcs_x, fit_y - wcs_y)
            sep_fit_dao = math.hypot(fit_x - dao_x, fit_y - dao_y)
            sep_wcs_dao = math.hypot(wcs_x - dao_x, wcs_y - dao_y)

            e1_rows.append({
                "proxy_id": pid,
                "frame": fn,
                "frame_idx": fi,
                "fit_x": fit_x,
                "fit_y": fit_y,
                "dao_x": dao_x,
                "dao_y": dao_y,
                "wcs_x": wcs_x,
                "wcs_y": wcs_y,
                "sep_fit_wcs_px": sep_fit_wcs,
                "sep_fit_dao_px": sep_fit_dao,
                "sep_wcs_dao_px": sep_wcs_dao,
                "ee_measured": ee_meas,
                "ee_expected": ee_exp,
                "ee_dev": ee_dev,
            })

            extreme.append((ee_dev, pid, fi, fn))

            fwhm_hint = float(row.get("fwhm_estimate_px", TABLE_FWHM))
            if not math.isfinite(fwhm_hint):
                fwhm_hint = TABLE_FWHM
            fit_diag = _gaussian_fit_diagnostics(data, dao_x, dao_y, fwhm_hint=fwhm_hint)
            if fit_diag:
                e2_rows.append({
                    "proxy_id": pid,
                    "frame": fn,
                    "phot_g": phot_g[pid],
                    **fit_diag,
                    "ee_measured": ee_meas,
                    "ee_expected": ee_exp,
                    "ee_dev": ee_dev,
                })

            # E5 WCS-position COG
            cog_wcs = _curve_of_growth_photutils(data, wcs_x, wcs_y)
            if cog_wcs is not None:
                ee_w = float(_ee_at_radius(cog_wcs["radii"], cog_wcs["ee"], R_AP))
                e5_wcs[pid].append(ee_w)
            e5_fit[pid].append(ee_meas)

            # E4 at fitted centroid
            det = _cog_detail(data, fit_x, fit_y)
            if det:
                proc_sky = float(row.get("sky_adu_per_px_annulus", float("nan")))
                e4_rows.append({
                    "proxy_id": pid,
                    "frame": fn,
                    "phot_g": phot_g[pid],
                    "sky_annulus_cog": det["sky_annulus_adu_per_px"],
                    "sky_proc_csv": proc_sky,
                    "sky_frame_median": global_sky,
                    "sky_frame_r50_median": float(sky_med[fi]),
                    "annulus_std": det["annulus_std_adu"],
                    "sources_in_annulus": _sources_in_annulus(fit_x, fit_y, frame_df),
                    "F_12_adu": det["F_12_adu"],
                    "F_1.916_adu": det["F_1.916_adu"],
                    "ee_measured": ee_meas,
                    "ee_expected": ee_exp,
                    "ee_dev": ee_dev,
                    "sky_annulus_minus_global": det["sky_annulus_adu_per_px"] - global_sky,
                    "sky_annulus_minus_proc": det["sky_annulus_adu_per_px"] - proc_sky if math.isfinite(proc_sky) else float("nan"),
                })

    # E1 summary per proxy
    for pid in proxies:
        rows = [r for r in e1_rows if r["proxy_id"] == pid]
        seps = np.array([r["sep_fit_wcs_px"] for r in rows])
        devs = np.array([r["ee_dev"] for r in rows])
        seps_f = seps[np.isfinite(seps)]
        devs_f = devs[np.isfinite(devs)]
        pearson = float(np.corrcoef(seps_f, devs_f)[0, 1]) if len(seps_f) >= 5 else float("nan")
        out["E1"][pid] = {
            "phot_g": phot_g[pid],
            "n": len(rows),
            "sep_fit_wcs_median_px": float(np.median(seps_f)) if seps_f.size else float("nan"),
            "sep_fit_wcs_p95_px": float(np.percentile(seps_f, 95)) if seps_f.size else float("nan"),
            "sep_fit_wcs_max_px": float(np.max(seps_f)) if seps_f.size else float("nan"),
            "n_exceed_1px": int(np.sum(seps_f > 1.0)),
            "n_exceed_2px": int(np.sum(seps_f > 2.0)),
            "n_exceed_3px": int(np.sum(seps_f > 3.0)),
            "pearson_ee_dev_vs_sep_fit_wcs": pearson,
        }

    # E2 summary + cross-tabs
    chi2_vals = np.array([r["chi2_reduced"] for r in e2_rows if math.isfinite(r["chi2_reduced"])])
    out["E2"]["all_proxies"] = {
        "n_star_frames": len(e2_rows),
        "chi2_reduced_median": float(np.median(chi2_vals)) if chi2_vals.size else float("nan"),
        "chi2_reduced_p95": float(np.percentile(chi2_vals, 95)) if chi2_vals.size else float("nan"),
        "pearson_ee_dev_vs_chi2": float(np.corrcoef(
            [r["ee_dev"] for r in e2_rows],
            [r["chi2_reduced"] for r in e2_rows],
        )[0, 1]) if len(e2_rows) >= 5 else float("nan"),
        "pearson_ee_dev_vs_centroid_shift": float(np.corrcoef(
            [r["ee_dev"] for r in e2_rows],
            [r["centroid_shift_px"] for r in e2_rows],
        )[0, 1]) if len(e2_rows) >= 5 else float("nan"),
    }
    for pid in proxies:
        rows = [r for r in e2_rows if r["proxy_id"] == pid]
        if not rows:
            continue
        out["E2"][pid] = {
            "phot_g": phot_g[pid],
            "chi2_median": float(np.median([r["chi2_reduced"] for r in rows])),
            "shift_median_px": float(np.median([r["centroid_shift_px"] for r in rows])),
            "shift_p95_px": float(np.percentile([r["centroid_shift_px"] for r in rows], 95)),
            "fit_amp_over_peak_median": float(np.median([r["fit_amp_over_peak"] for r in rows])),
            "high_ee_dev_frames": [
                {"frame": r["frame"], "ee_dev": r["ee_dev"], "chi2": r["chi2_reduced"],
                 "shift_px": r["centroid_shift_px"], "ee_meas": r["ee_measured"]}
                for r in sorted(rows, key=lambda t: -t["ee_dev"])[:5]
            ],
        }

    # E4 summary
    sky_diff = np.array([r["sky_annulus_minus_global"] for r in e4_rows if math.isfinite(r["sky_annulus_minus_global"])])
    ee_dev_arr = np.array([r["ee_dev"] for r in e4_rows])
    out["E4"]["all"] = {
        "sky_annulus_minus_global_median": float(np.median(sky_diff)) if sky_diff.size else float("nan"),
        "sky_annulus_minus_global_p95_abs": float(np.percentile(np.abs(sky_diff), 95)) if sky_diff.size else float("nan"),
        "pearson_ee_dev_vs_sky_minus_global": float(np.corrcoef(ee_dev_arr, sky_diff)[0, 1]) if len(ee_dev_arr) >= 5 else float("nan"),
        "pearson_ee_dev_vs_F12": float(np.corrcoef(
            ee_dev_arr, [r["F_12_adu"] for r in e4_rows]
        )[0, 1]) if len(ee_dev_arr) >= 5 else float("nan"),
        "sources_in_annulus_median": float(np.median([r["sources_in_annulus"] for r in e4_rows])),
    }
    for pid in proxies:
        rows = [r for r in e4_rows if r["proxy_id"] == pid]
        sd = np.array([r["sky_annulus_minus_global"] for r in rows])
        ed = np.array([r["ee_dev"] for r in rows])
        out["E4"][pid] = {
            "phot_g": phot_g[pid],
            "sky_minus_global_median": float(np.median(sd)),
            "sky_minus_global_p95_abs": float(np.percentile(np.abs(sd), 95)),
            "pearson_ee_dev_vs_sky_diff": float(np.corrcoef(ed, sd)[0, 1]) if len(ed) >= 5 else float("nan"),
            "F12_median_adu": float(np.median([r["F_12_adu"] for r in rows])),
            "F12_min_adu": float(np.min([r["F_12_adu"] for r in rows])),
            "worst_frames": sorted(
                [{"frame": r["frame"], "ee_meas": r["ee_measured"], "sky_ann": r["sky_annulus_cog"],
                  "sky_minus_global": r["sky_annulus_minus_global"], "F12": r["F_12_adu"],
                  "sources_in_annulus": r["sources_in_annulus"]}
                 for r in rows],
                key=lambda t: -abs(t["ee_meas"] - _ee_expected(r50_series[frame_names.index(t["frame"])])),
            )[:3],
        }

    # E5
    e5_table = []
    for pid in proxies:
        ef = np.array(e5_fit[pid])
        ew = np.array(e5_wcs[pid])
        m = np.isfinite(ef) & np.isfinite(ew)
        row = {
            "proxy_id": pid,
            "phot_g": phot_g[pid],
            "EE_std_fitted_centroid": float(np.std(ef[m])) if m.sum() else float("nan"),
            "EE_std_wcs_position": float(np.std(ew[m])) if m.sum() else float("nan"),
            "ratio_wcs_over_fit": float(np.std(ew[m]) / np.std(ef[m])) if m.sum() and np.std(ef[m]) > 0 else float("nan"),
        }
        e5_table.append(row)
        out["E5"][pid] = row

    out["E5"]["table"] = e5_table

    # E3: top 10 extreme |EE - expected| across all proxies
    extreme.sort(key=lambda t: -t[0])
    e3_cases = extreme[:10]

    for ee_dev, pid, fi, fn in e3_cases:
        proc = csvs[fi]
        fits_path = lights / fn.replace(".csv", ".fits")
        df = pd.read_csv(proc, dtype={"catalog_id": str}).set_index("catalog_id")
        row = df.loc[pid]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        with fits.open(fits_path, memmap=False) as hdul:
            data = hdul[0].data.astype(np.float64)
            wcs_e3 = WCS(hdul[0].header)
        cached = ee_cache[fi][pid]
        fx, fy = float(cached["centroid_x"]), float(cached["centroid_y"])
        fwhm_hint = float(row.get("fwhm_estimate_px", TABLE_FWHM))
        fit_diag = _gaussian_fit_diagnostics(data, float(row["x"]), float(row["y"]), fwhm_hint=fwhm_hint)
        det = _cog_detail(data, fx, fy)
        ee_meas = float(_ee_at_radius(cached["radii"], cached["ee"], R_AP))
        out["E3"].append({
            "proxy_id": pid,
            "phot_g": phot_g[pid],
            "frame": fn,
            "frame_idx": fi,
            "r50_frame": float(r50_series[fi]),
            "sky_frame_median": float(sky_med[fi]),
            "ee_measured": ee_meas,
            "ee_expected": float(_ee_expected(r50_series[fi])),
            "ee_dev": ee_dev,
            "aperture_centroid": [fx, fy],
            "dao_position": [float(row["x"]), float(row["y"])],
            "wcs_catalog_position": [float(x) for x in wcs_e3.world_to_pixel_values(*cat_ra_dec[pid])],
            "fit_diagnostics": fit_diag,
            "cog_detail": det,
            "cutout": _cutout_stats(data, fx, fy, half=20),
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out}")
    print("E5 table:")
    for r in e5_table:
        print(f"  G {r['phot_g']:.2f}  fit_std={r['EE_std_fitted_centroid']:.4f}  wcs_std={r['EE_std_wcs_position']:.4f}")


if __name__ == "__main__":
    main()
