#!/usr/bin/env python3
"""Step 1j: test F(12) normalisation and sky annulus (J1-J4).

Diagnostic only. No production change.

Usage:
  python dev/tools/closure_step1j_test_f12_normalisation.py \\
    --draft Archive/Drafts/draft_000435_snapshot_skysurface_20260716 \\
    --step1b-json tmp/closure_step1b_results.json \\
    --step1g-json tmp/closure_step1g_results.json \\
    --cache tmp/closure_step1f_ee_cache.npz \\
    --out tmp/closure_step1j_diagnostics.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from itertools import combinations
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from astropy import modeling
from astropy.io import fits
from astropy.stats import sigma_clip
from photutils.aperture import CircularAnnulus, CircularAperture, aperture_photometry
from scipy import stats

from closure_step1b_differential_aperture import TABLE_FWHM, _ee_at_radius, _load_all_proc
from closure_step1c_differential_aperture import PROXY_R_AP
from closure_step1e_differential_aperture import COG_DR, COG_RMAX, FIT_BOX, NORM_RADIUS

warnings.filterwarnings("ignore", category=fits.verify.VerifyWarning)

R_AP = PROXY_R_AP
FW_PROD = 2.395
ANN_INNER_FWHM = 4.75
ANN_OUTER_FWHM = 9.0
COG_RADII = np.arange(COG_DR, COG_RMAX + COG_DR / 2, COG_DR)

GEOM_HARNESS = (25.0, 45.0)
GEOM_NARROW = (12.0, 20.0)


def _prod_annulus(r_ap: float, fw: float = FW_PROD) -> tuple[float, float]:
    r_in = max(r_ap + 0.5, ANN_INNER_FWHM * fw)
    r_out = max(r_in + 0.5, ANN_OUTER_FWHM * fw)
    return r_in, r_out


def _annulus_pixels(data: np.ndarray, xc: float, yc: float, r_in: float, r_out: float) -> np.ndarray:
    ann = CircularAnnulus([(xc, yc)], r_in=r_in, r_out=r_out)
    m = ann.to_mask(method="center")[0]
    cut = m.cutout(data)
    if cut is None:
        return np.array([])
    return cut[m.data.astype(bool)]


def _measure_cog(
    data: np.ndarray,
    xc: float,
    yc: float,
    r_in: float,
    r_out: float,
) -> dict[str, float] | None:
    h, w = data.shape
    max_r = max(float(np.max(COG_RADII)), r_out) + 2.0
    if xc - max_r < 0 or yc - max_r < 0 or xc + max_r >= w or yc + max_r >= h:
        return None
    ann = CircularAnnulus([(xc, yc)], r_in=r_in, r_out=r_out)
    sky_pp = float(aperture_photometry(data, ann)["aperture_sum"][0] / ann.area)
    flux: list[float] = []
    for r in COG_RADII:
        ap = CircularAperture([(xc, yc)], r=float(r))
        s = float(aperture_photometry(data, ap)["aperture_sum"][0])
        flux.append(s - sky_pp * ap.area)
    arr = np.asarray(flux, dtype=np.float64)
    if arr[-1] <= 0:
        return None
    ee = arr / arr[-1]
    f12 = float(arr[-1])
    f_ap = float(np.interp(R_AP, COG_RADII, arr))
    ee_ap = float(_ee_at_radius(COG_RADII, ee, R_AP))
    pix = _annulus_pixels(data, xc, yc, r_in, r_out)
    if pix.size == 0:
        ann_med = ann_mean = ann_std = clip_mean = float("nan")
    else:
        ann_med = float(np.median(pix))
        ann_mean = float(np.mean(pix))
        ann_std = float(np.std(pix))
        clipped = sigma_clip(pix, sigma=3.0, maxiters=5)
        clip_mean = float(np.mean(clipped)) if clipped.size else float("nan")
    return {
        "sky_ann": sky_pp,
        "F_12": f12,
        "F_1.916": f_ap,
        "EE_1.916": ee_ap,
        "annulus_median": ann_med,
        "annulus_mean": ann_mean,
        "annulus_std": ann_std,
        "annulus_clip_mean": clip_mean,
        "annulus_med_minus_clip": ann_med - clip_mean if math.isfinite(clip_mean) else float("nan"),
    }


def _sources_in_annulus(
    xc: float,
    yc: float,
    frame_df: pd.DataFrame,
    r_in: float,
    r_out: float,
    *,
    catalog_g: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for _, row in frame_df.iterrows():
        dx = float(row["x"]) - xc
        dy = float(row["y"]) - yc
        sep = math.hypot(dx, dy)
        if r_in <= sep <= r_out:
            cid = str(row.get("catalog_id", ""))
            g = float(row.get("phot_g_mean_mag", row.get("catalog_mag", float("nan"))))
            if catalog_g and cid in catalog_g:
                g = catalog_g[cid]
            out.append({"sep_px": sep, "phot_g": g, "catalog_id": cid})
    return sorted(out, key=lambda t: t["sep_px"])


def _gaussian_amp_over_peak(
    data: np.ndarray,
    x_proc: float,
    y_proc: float,
    *,
    fwhm_hint: float,
) -> float | None:
    h, w = data.shape
    box = FIT_BOX
    xc_i, yc_i = int(round(x_proc)), int(round(y_proc))
    if not (box <= xc_i < w - box and box <= yc_i < h - box):
        return None
    cut = data[yc_i - box : yc_i + box + 1, xc_i - box : xc_i + box + 1].astype(np.float64)
    yy, xx = np.mgrid[0 : cut.shape[0], 0 : cut.shape[1]]
    bg = float(np.median(cut))
    amp0 = max(float(np.max(cut) - bg), 1.0)
    peak = float(np.max(cut))
    fitter = modeling.fitting.TRFLSQFitter()
    g0 = modeling.models.Gaussian2D(
        amplitude=amp0, x_mean=float(box), y_mean=float(box),
        x_stddev=fwhm_hint / 2.355, y_stddev=fwhm_hint / 2.355, theta=0.0,
    )
    g0.x_mean.bounds = (box - 2, box + 2)
    g0.y_mean.bounds = (box - 2, box + 2)
    g0.theta.fixed = True
    c0 = modeling.models.Const2D(amplitude=bg)
    try:
        mg = fitter(g0 + c0, xx, yy, cut, maxiter=200)
        return float(mg.amplitude_0.value) / peak if peak > 0 else float("nan")
    except Exception:  # noqa: BLE001
        return None


def _j1_metrics(rows: list[dict[str, Any]], phot_g: dict[str, float]) -> dict[str, Any]:
    """J1 summary from star-frame rows with F_12 and sky_ann."""
    valid = [r for r in rows if r.get("F_12", 0) > 0 and math.isfinite(r["F_12"])]
    if len(valid) < 5:
        return {"n": len(valid), "insufficient": True}

    gs = np.array([phot_g[r["star_id"]] for r in valid])
    logf = np.log10(np.array([r["F_12"] for r in valid]))
    slope, intercept = np.polyfit(gs, logf, 1)
    pred = slope * gs + intercept
    resid = logf - pred
    rms = float(np.sqrt(np.mean(resid**2)))

    # per-star median residual
    star_res: dict[str, list[float]] = {}
    star_sky: dict[str, list[float]] = {}
    for r, rv in zip(valid, resid):
        star_res.setdefault(r["star_id"], []).append(float(rv))
        star_sky.setdefault(r["star_id"], []).append(r["sky_ann"] - r["global_sky"])

    star_med_res = {sid: float(np.median(v)) for sid, v in star_res.items()}
    star_med_sky = {sid: float(np.median(v)) for sid, v in star_sky.items()}
    sids = list(star_med_res.keys())
    if len(sids) >= 3:
        pearson_res_sky = float(np.corrcoef(
            [star_med_res[s] for s in sids],
            [star_med_sky[s] for s in sids],
        )[0, 1])
    else:
        pearson_res_sky = float("nan")

    # same-magnitude pairs (per-star medians)
    star_med_f12: dict[str, float] = {}
    star_med_g: dict[str, float] = {}
    for sid in sids:
        fvals = [r["F_12"] for r in valid if r["star_id"] == sid]
        star_med_f12[sid] = float(np.median(fvals))
        star_med_g[sid] = phot_g[sid]

    pair_ratios: list[float] = []
    pair_dg: list[float] = []
    for a, b in combinations(sids, 2):
        dg = abs(star_med_g[a] - star_med_g[b])
        if dg <= 0.05:
            pair_ratios.append(star_med_f12[a] / star_med_f12[b])
            pair_dg.append(dg)

    return {
        "n_star_frames": len(valid),
        "n_stars": len(sids),
        "fitted_slope_logF12_vs_G": float(slope),
        "expected_slope": -0.4,
        "intercept": float(intercept),
        "scatter_rms_dex": rms,
        "pearson_star_median_F12_residual_vs_sky_offset": pearson_res_sky,
        "same_mag_pairs_n": len(pair_ratios),
        "same_mag_pair_F12_ratio_median": float(np.median(pair_ratios)) if pair_ratios else float("nan"),
        "same_mag_pair_F12_ratio_p95": float(np.percentile(pair_ratios, 95)) if pair_ratios else float("nan"),
        "same_mag_pair_F12_ratio_max": float(np.max(pair_ratios)) if pair_ratios else float("nan"),
        "same_mag_pair_ratios": pair_ratios[:20],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", type=Path, required=True)
    ap.add_argument("--step1b-json", type=Path, default=Path("tmp/closure_step1b_results.json"))
    ap.add_argument("--step1g-json", type=Path, default=Path("tmp/closure_step1g_results.json"))
    ap.add_argument("--cache", type=Path, default=Path("tmp/closure_step1f_ee_cache.npz"))
    ap.add_argument("--out", type=Path, default=Path("tmp/closure_step1j_diagnostics.json"))
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
    phot_g: dict[str, float] = {}
    for sid in star_ids:
        row = catalog_full[catalog_full["catalog_id"].astype(str) == sid]
        if not row.empty:
            phot_g[sid] = float(row.iloc[0]["phot_g"])

    npz = np.load(args.cache, allow_pickle=True)
    ee_cache = npz["ee_cache"].item()
    aperture_by_frame = npz["aperture_by_frame"].item()

    # representative frames for J2 detail (20 evenly spaced)
    rep_frames = set(int(round(i)) for i in np.linspace(0, n_frames - 1, 20))

    geoms = {
        "harness_25_45": GEOM_HARNESS,
        "narrow_12_20": GEOM_NARROW,
    }

    rows_by_geom: dict[str, list[dict[str, Any]]] = {k: [] for k in geoms}
    rows_by_geom["production_scaled"] = []
    j2_detail: list[dict[str, Any]] = []
    amp_peak_rows: list[dict[str, Any]] = []

    for fi, proc in enumerate(csvs):
        fn = proc.name
        fits_path = lights / fn.replace(".csv", ".fits")
        df = pd.read_csv(proc, dtype={"catalog_id": str})
        frame_df = df.copy()

        with fits.open(fits_path, memmap=False) as hdul:
            data = hdul[0].data.astype(np.float64)
            hdr = hdul[0].header
            global_sky = float(np.median(data))
            cx_field = float(hdr.get("CRPIX1", data.shape[1] / 2))
            cy_field = float(hdr.get("CRPIX2", data.shape[0] / 2))

        for sid in star_ids:
            if sid not in ee_cache.get(fi, {}):
                continue
            cached = ee_cache[fi][sid]
            xc = float(cached["centroid_x"])
            yc = float(cached["centroid_y"])
            proc_row = df.loc[df["catalog_id"] == sid]
            if proc_row.empty:
                continue
            proc_row = proc_row.iloc[0]
            dao_x, dao_y = float(proc_row["x"]), float(proc_row["y"])
            fwhm_hint = float(proc_row.get("fwhm_estimate_px", TABLE_FWHM))
            r_ap_star = float(aperture_by_frame.get(fi, {}).get(sid, R_AP))

            base = {
                "star_id": sid,
                "frame": fn,
                "frame_idx": fi,
                "phot_g": phot_g.get(sid, float("nan")),
                "global_sky": global_sky,
                "field_center_sep_px": math.hypot(xc - cx_field, yc - cy_field),
            }

            for gname, (rin, rout) in geoms.items():
                m = _measure_cog(data, xc, yc, rin, rout)
                if m is None:
                    continue
                rows_by_geom[gname].append({**base, **m, "r_in": rin, "r_out": rout})

            rin_p, rout_p = _prod_annulus(r_ap_star)
            mp = _measure_cog(data, xc, yc, rin_p, rout_p)
            if mp is not None:
                rows_by_geom["production_scaled"].append({
                    **base, **mp, "r_in": rin_p, "r_out": rout_p, "r_ap": r_ap_star,
                })

            if fi in rep_frames:
                m_h = _measure_cog(data, xc, yc, *GEOM_HARNESS)
                if m_h is not None:
                    nbs = _sources_in_annulus(xc, yc, frame_df, *GEOM_HARNESS, catalog_g=phot_g)
                    j2_detail.append({
                        **base,
                        **m_h,
                        "n_sources_in_annulus": len(nbs),
                        "sources_in_annulus": nbs[:10],
                        "sky_ann_minus_global": m_h["sky_ann"] - global_sky,
                        "local_sky_from_pipeline": None,
                        "local_sky_note": (
                            "Pipeline sky-surface fit at star position not stored in draft "
                            "FITS/CSV; VY_SKYSF coefficients not accessible."
                        ),
                    })

            apop = _gaussian_amp_over_peak(data, dao_x, dao_y, fwhm_hint=fwhm_hint)
            if apop is not None and math.isfinite(apop):
                amp_peak_rows.append({
                    "star_id": sid,
                    "frame": fn,
                    "phot_g": phot_g.get(sid, float("nan")),
                    "is_proxy": sid in proxies,
                    "amp_over_peak": apop,
                })

    # J1 per geometry
    j1: dict[str, Any] = {}
    for gname, rows in rows_by_geom.items():
        j1[gname] = _j1_metrics(rows, phot_g)

    # highlight G 11.52 vs 11.53 pair
    proxy_f12: dict[str, float] = {}
    for sid in proxies:
        vals = [r["F_12"] for r in rows_by_geom["harness_25_45"] if r["star_id"] == sid]
        if vals:
            proxy_f12[sid] = float(np.median(vals))
    j1["proxy_pair_check"] = {
        k: {"phot_g": phot_g.get(k), "median_F12_harness": proxy_f12.get(k)}
        for k in proxies if k in proxy_f12
    }
    g1152 = "1497368849430107904"
    g1153 = "1497091703781835776"
    if g1152 in proxy_f12 and g1153 in proxy_f12:
        j1["proxy_pair_check"]["ratio_11.52_over_11.53"] = proxy_f12[g1152] / proxy_f12[g1153]
        j1["proxy_pair_check"]["delta_G"] = phot_g[g1152] - phot_g[g1153]
        sky52 = np.median([r["sky_ann"] - r["global_sky"] for r in rows_by_geom["harness_25_45"] if r["star_id"] == g1152])
        sky53 = np.median([r["sky_ann"] - r["global_sky"] for r in rows_by_geom["harness_25_45"] if r["star_id"] == g1153])
        j1["proxy_pair_check"]["sky_ann_minus_global_diff"] = float(sky52 - sky53)

    # J2 correlations on harness rep frames
    if j2_detail:
        nsrc = np.array([d["n_sources_in_annulus"] for d in j2_detail])
        sky_diff = np.array([d["sky_ann_minus_global"] for d in j2_detail])
        fc_sep = np.array([d["field_center_sep_px"] for d in j2_detail])
        j2 = {
            "n_star_frames": len(j2_detail),
            "local_sky_accessible": False,
            "local_sky_note": j2_detail[0]["local_sky_note"],
            "n_sources_in_annulus_median": float(np.median(nsrc)),
            "pearson_sky_diff_vs_n_sources": float(np.corrcoef(sky_diff, nsrc)[0, 1]) if len(j2_detail) >= 5 else float("nan"),
            "pearson_sky_diff_vs_field_center_sep": float(np.corrcoef(sky_diff, fc_sep)[0, 1]) if len(j2_detail) >= 5 else float("nan"),
            "annulus_med_minus_clip_mean": float(np.mean([d["annulus_med_minus_clip"] for d in j2_detail])),
            "sample": j2_detail[:5],
        }
    else:
        j2 = {"insufficient": True}

    # J3 EE std per proxy per geometry
    j3: dict[str, Any] = {"J1_by_geometry": {k: j1[k] for k in rows_by_geom}, "EE_std_per_proxy": {}}
    for gname, rows in rows_by_geom.items():
        j3["EE_std_per_proxy"][gname] = {}
        for pid in proxies:
            ees = [r["EE_1.916"] for r in rows if r["star_id"] == pid and math.isfinite(r.get("EE_1.916", float("nan")))]
            j3["EE_std_per_proxy"][gname][pid] = {
                "phot_g": phot_g.get(pid),
                "EE_std": float(np.std(ees)) if len(ees) >= 2 else float("nan"),
                "F12_median": float(np.median([r["F_12"] for r in rows if r["star_id"] == pid])) if ees else float("nan"),
            }

    # J4 amp/peak distribution
    ap_arr = np.array([r["amp_over_peak"] for r in amp_peak_rows])
    j4: dict[str, Any] = {
        "n_star_frames": len(amp_peak_rows),
        "distribution": {
            "median": float(np.median(ap_arr)),
            "p5": float(np.percentile(ap_arr, 5)),
            "p16": float(np.percentile(ap_arr, 16)),
            "p84": float(np.percentile(ap_arr, 84)),
            "p95": float(np.percentile(ap_arr, 95)),
            "min": float(np.min(ap_arr)),
        },
        "threshold_candidates": [],
    }
    for thr in [0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        rej = [r for r in amp_peak_rows if r["amp_over_peak"] < thr]
        by_proxy = {pid: sum(1 for r in rej if r["star_id"] == pid) for pid in proxies}
        j4["threshold_candidates"].append({
            "threshold": thr,
            "n_rejected_total": len(rej),
            "frac_rejected": len(rej) / max(len(amp_peak_rows), 1),
            "per_proxy_rejected": by_proxy,
        })
    # justify threshold at p16 (~1 sigma low tail for roughly normal)
    p16_thr = float(np.percentile(ap_arr, 16))
    j4["proposed_threshold"] = {
        "value": p16_thr,
        "justification": "16th percentile of amp/peak across all star-frames (lower envelope of converged fits)",
        "n_rejected": int(np.sum(ap_arr < p16_thr)),
    }

    out = {"J1": j1, "J2": j2, "J3": j3, "J4": j4, "star_ids": star_ids, "proxies": proxies}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out}")
    print("J1 harness slope:", j1["harness_25_45"].get("fitted_slope_logF12_vs_G"))
    print("J1 prod slope:", j1["production_scaled"].get("fitted_slope_logF12_vs_G"))
    print("J1 narrow slope:", j1["narrow_12_20"].get("fitted_slope_logF12_vs_G"))


if __name__ == "__main__":
    main()
