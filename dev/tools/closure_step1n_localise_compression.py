#!/usr/bin/env python3
"""Step 1n: localise bright-end flux compression (N1-N4).

Diagnostic only. No production change.

Usage:
  python dev/tools/closure_step1n_localise_compression.py \\
    --draft Archive/Drafts/draft_000435_snapshot_skysurface_20260716 \\
    --step1b-json tmp/closure_step1b_results.json \\
    --out tmp/closure_step1n_diagnostics.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from photutils.aperture import CircularAnnulus, CircularAperture, aperture_photometry
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src_py"))

from closure_step1b_differential_aperture import TABLE_FWHM, _load_all_proc

warnings.filterwarnings("ignore")

R_LARGE_FWHM = 4.0
FW_TABLE = TABLE_FWHM
R_LARGE_PX = R_LARGE_FWHM * FW_TABLE  # 9.58
STAMP_R = int(max(4, round(3.5 * FW_TABLE)))  # 8 px at fwhm_eff=2.395


def _fit_slope(y: np.ndarray, g: np.ndarray) -> dict[str, Any]:
    m = np.isfinite(y) & np.isfinite(g)
    if m.sum() < 5:
        return {"n": int(m.sum()), "insufficient": True}
    if float(np.std(g[m])) < 1e-9:
        return {"n": int(m.sum()), "insufficient": True, "note": "G constant in bin"}
    try:
        slope, intercept, r, _, se = stats.linregress(g[m], y[m])
    except ValueError:
        return {"n": int(m.sum()), "insufficient": True, "note": "linregress failed"}
    return {"n": int(m.sum()), "slope": float(slope), "slope_se": float(se), "intercept": float(intercept), "r_squared": float(r**2)}


def _ap_flux(data: np.ndarray, xc: float, yc: float, r: float, sky_in: float, sky_out: float) -> float | None:
    h, w = data.shape
    if xc - sky_out < 0 or yc - sky_out < 0 or xc + sky_out >= w or yc + sky_out >= h:
        return None
    ann = CircularAnnulus([(xc, yc)], r_in=sky_in, r_out=sky_out)
    sky_pp = float(aperture_photometry(data, ann)["aperture_sum"][0] / ann.area)
    ap = CircularAperture([(xc, yc)], r=r)
    raw = float(aperture_photometry(data, ap)["aperture_sum"][0])
    return raw - sky_pp * ap.area


def _surface_at_stars(
    data: np.ndarray,
    order: int,
    fwhm_px: float,
    star_xy: list[tuple[float, float]],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return surface ADU at each star position via refit (same logic as pipeline)."""
    import numpy as np
    from astropy.stats import sigma_clip, sigma_clipped_stats
    from photutils.detection import DAOStarFinder
    from pipeline import DAO_STAR_FINDER_NO_ROUNDNESS_FILTER

    arr = np.asarray(data, dtype=np.float32)
    h, w = arr.shape
    finite = np.isfinite(arr)
    fill = float(np.nanmedian(arr[finite])) if finite.any() else 0.0
    work = np.where(finite, arr, fill)
    mask = np.ones((h, w), dtype=bool)
    margin = 40
    if h > 2 * margin and w > 2 * margin:
        mask[:margin, :] = mask[-margin:, :] = mask[:, :margin] = mask[:, -margin:] = False
    fwhm_eff = max(1.2, float(fwhm_px))
    _, med, std = sigma_clipped_stats(work, sigma=3.0, maxiters=3)
    data0 = np.nan_to_num((work - med).astype(np.float32), nan=0.0)
    thr = max(3.0 * float(std), 1e-6)
    finder = DAOStarFinder(fwhm=fwhm_eff, threshold=thr, n_brightest=5000, **DAO_STAR_FINDER_NO_ROUNDNESS_FILTER)
    tbl = finder(data0)
    stamp_r = int(max(4, round(3.5 * fwhm_eff)))
    if tbl is not None and len(tbl) > 0:
        r2 = stamp_r * stamp_r
        for row in tbl:
            cy, cx = int(round(float(row["y_centroid"]))), int(round(float(row["x_centroid"])))
            if not (0 <= cy < h and 0 <= cx < w):
                continue
            y0, y1 = max(0, cy - stamp_r), min(h, cy + stamp_r + 1)
            x0, x1 = max(0, cx - stamp_r), min(w, cx + stamp_r + 1)
            yy_l, xx_l = np.ogrid[y0:y1, x0:x1]
            mask[y0:y1, x0:x1] &= ~((yy_l - cy) ** 2 + (xx_l - cx) ** 2 <= r2)
    bg_median, _, _ = sigma_clipped_stats(work, mask=mask, sigma=3.0, maxiters=5)
    calm_thr = 100.0
    fit_mask = mask & (np.abs(work - float(bg_median)) < calm_thr)
    step = 4
    yy_s, xx_s = np.mgrid[0:h:step, 0:w:step]
    z_s = (work[::step, ::step] - float(bg_median)).astype(np.float64)
    m_s = fit_mask[::step, ::step]
    use_mask = m_s & np.isfinite(z_s)
    order_i = min(2, max(1, int(order)))
    min_coef = (order_i + 1) * (order_i + 2) // 2
    if int(np.count_nonzero(use_mask)) < min_coef + 10:
        return np.full(len(star_xy), np.nan), {"applied": False}
    z_samples = z_s[use_mask]
    clipped = sigma_clip(z_samples, sigma=3.0, maxiters=5, masked=True)
    good = ~clipped.mask
    x_fit = xx_s[use_mask][good].astype(np.float64)
    y_fit = yy_s[use_mask][good].astype(np.float64)
    z_fit = z_samples[good]
    cols: list[np.ndarray] = []
    for i in range(order_i + 1):
        for j in range(order_i + 1 - i):
            cols.append((x_fit**i) * (y_fit**j))
    coef, *_ = np.linalg.lstsq(np.column_stack(cols), z_fit, rcond=None)
    yy_f, xx_f = np.mgrid[0:h, 0:w]
    cols_f: list[np.ndarray] = []
    x_flat = xx_f.ravel().astype(np.float64)
    y_flat = yy_f.ravel().astype(np.float64)
    for i in range(order_i + 1):
        for j in range(order_i + 1 - i):
            cols_f.append((x_flat**i) * (y_flat**j))
    surf = (np.column_stack(cols_f) @ coef).reshape(h, w)
    vals = []
    for xc, yc in star_xy:
        ix, iy = int(round(xc)), int(round(yc))
        if 0 <= ix < w and 0 <= iy < h:
            vals.append(float(surf[iy, ix]))
        else:
            vals.append(float("nan"))
    stats = {
        "applied": True,
        "p2p_adu": float(np.nanmax(surf) - np.nanmin(surf)),
        "bg_median_adu": float(bg_median),
    }
    return np.array(vals, dtype=np.float64), stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", type=Path, required=True)
    ap.add_argument("--step1b-json", type=Path, default=Path("tmp/closure_step1b_results.json"))
    ap.add_argument("--out", type=Path, default=Path("tmp/closure_step1n_diagnostics.json"))
    args = ap.parse_args()

    draft = args.draft.resolve()
    draft_root = draft.parent / "draft_000435"
    lights = draft / "detrended_aligned/lights/NoFilter_60_2"
    csvs, catalog_full, _ = _load_all_proc(lights)

    with args.step1b_json.open(encoding="utf-8") as f:
        star_ids: list[str] = json.load(f)["part_a"]["star_ids"]

    meta: dict[str, float] = {}
    for sid in star_ids:
        row = catalog_full[catalog_full["catalog_id"].astype(str) == sid]
        if row.empty:
            continue
        meta[sid] = float(row.iloc[0]["phot_g"])

    qc = pd.read_csv(draft_root / "processed/lights/qc_metrics.csv")
    qc["frame"] = qc["dst"].str.extract(r"proc_BO_CVn_Light_(\d+)\.fits")[0].astype(int)

    sky_in = max(R_LARGE_PX + 0.5, 4.75 * FW_TABLE)
    sky_out = max(sky_in + 0.5, 9.0 * FW_TABLE)

    rows: list[dict[str, Any]] = []
    n3_before: list[float] = []
    n3_after: list[float] = []
    n3_g: list[float] = []
    n3_surf: list[float] = []
    n3_p2p: list[float] = []
    n3_resid: list[float] = []

    for fi, proc in enumerate(csvs):
        frame_num = int(proc.name.split("_")[-1].replace(".csv", ""))
        cal_path = draft_root / f"calibrated/lights/NoFilter_60_2/BO_CVn_Light_{frame_num:03d}.fits"
        proc_path = draft_root / f"processed/lights/NoFilter_60_2/proc_BO_CVn_Light_{frame_num:03d}.fits"
        if not cal_path.exists() or not proc_path.exists():
            continue

        qc_row = qc.loc[qc["frame"] == frame_num]
        fwhm_qc = float(qc_row.iloc[0]["fwhm_px"]) if len(qc_row) else FW_TABLE
        p2p = float(qc_row.iloc[0]["sky_surface_p2p_adu"]) if len(qc_row) else float("nan")
        sky_applied = bool(qc_row.iloc[0]["sky_surface_applied"]) if len(qc_row) else False

        df = pd.read_csv(proc, dtype={"catalog_id": str})
        with fits.open(cal_path, memmap=False) as hdul:
            data_cal = hdul[0].data.astype(np.float64)
            wcs = WCS(hdul[0].header)
        with fits.open(proc_path, memmap=False) as hdul:
            data_proc = hdul[0].data.astype(np.float64)

        star_xy: list[tuple[float, float]] = []
        star_batch: list[dict[str, Any]] = []

        for sid in star_ids:
            if sid not in meta:
                continue
            pr = df.loc[df["catalog_id"] == sid]
            if pr.empty:
                continue
            pr = pr.iloc[0]
            ok = str(pr.get("photometry_ok", "true")).lower() in ("true", "1", "yes")
            usable = str(pr.get("is_usable", "true")).lower() in ("true", "1", "yes")
            if not (ok and usable):
                continue
            ra, dec = float(pr["ra_deg"]), float(pr["dec_deg"])
            xc, yc = wcs.all_world2pix(ra, dec, 0)
            xc, yc = float(np.asarray(xc).ravel()[0]), float(np.asarray(yc).ravel()[0])
            if not (math.isfinite(xc) and math.isfinite(yc)):
                continue

            flux_b = _ap_flux(data_cal, xc, yc, R_LARGE_PX, sky_in, sky_out)
            flux_a = _ap_flux(data_proc, xc, yc, R_LARGE_PX, sky_in, sky_out)
            if flux_b is None or flux_a is None or flux_b <= 0 or flux_a <= 0:
                continue

            g = meta[sid]
            peak = float(pr.get("peak_max_adu", float("nan")))
            sat_lim = float(pr.get("saturate_limit_adu_85pct", 55705.0))
            fl_proc = float(pr.get("flux_large", float("nan")))
            if math.isfinite(fl_proc) and fl_proc > 0:
                fl_raw = float(pr.get("flux", float("nan")))
            else:
                fl_raw = float("nan")
                fl_proc = float("nan")

            rec = {
                "star_id": sid,
                "frame_idx": fi,
                "frame_num": frame_num,
                "phot_g": g,
                "flux": fl_raw,
                "flux_large_proc": fl_proc,
                "flux_large_before_sky": flux_b,
                "flux_large_after_sky": flux_a,
                "peak_max_adu": peak,
                "saturate_limit_85": sat_lim,
                "is_saturated": bool(pr.get("is_saturated", False)),
                "likely_saturated": bool(pr.get("likely_saturated", False)),
                "sky_surface_p2p_adu": p2p,
                "sky_surface_applied": sky_applied,
                "xc": xc, "yc": yc,
            }
            rows.append(rec)
            star_xy.append((xc, yc))
            star_batch.append(rec)

        if star_xy:
            surf_vals, _ = _surface_at_stars(data_cal, order=2, fwhm_px=fwhm_qc, star_xy=star_xy)
            for rec, sv in zip(star_batch, surf_vals):
                rec["sky_surface_at_star_adu"] = float(sv) if math.isfinite(sv) else float("nan")
                n3_before.append(rec["flux_large_before_sky"])
                n3_after.append(rec["flux_large_after_sky"])
                n3_g.append(rec["phot_g"])
                n3_surf.append(rec["sky_surface_at_star_adu"])
                n3_p2p.append(rec["sky_surface_p2p_adu"])
                if math.isfinite(rec.get("flux_large_proc", float("nan"))):
                    pass

    # N1 bin slopes
    g_all = np.array([r["phot_g"] for r in rows])
    def _col(name: str) -> np.ndarray:
        return np.array([r.get(name, float("nan")) for r in rows], dtype=np.float64)

    g_arr = _col("phot_g")
    log_f = np.log10(np.where(_col("flux") > 0, _col("flux"), np.nan))
    log_fl = np.log10(np.where(_col("flux_large_proc") > 0, _col("flux_large_proc"), np.nan))

    g_min, g_max = float(np.nanmin(g_arr)), float(np.nanmax(g_arr))
    bins = []
    lo = math.floor(g_min * 2) / 2  # 0.5 mag steps -> use 1.0 mag
    lo = math.floor(g_min)
    while lo < g_max:
        hi = lo + 1.0
        m = (g_arr >= lo) & (g_arr < hi) & np.isfinite(log_f)
        m_large = (g_arr >= lo) & (g_arr < hi) & np.isfinite(log_fl)
        entry = {"G_lo": lo, "G_hi": hi}
        if m.sum() >= 5:
            entry["flux"] = _fit_slope(log_f[m], g_arr[m])
        else:
            entry["flux"] = {"n": int(m.sum()), "insufficient": True}
        if m_large.sum() >= 5:
            entry["flux_large"] = _fit_slope(log_fl[m_large], g_arr[m_large])
        else:
            entry["flux_large"] = {"n": int(m_large.sum()), "insufficient": True}
        bins.append(entry)
        lo = hi

    med_g = float(np.nanmedian(g_arr))
    n1_halves = {
        "median_G_split": med_g,
        "faint_flux": _fit_slope(log_f[g_arr >= med_g], g_arr[g_arr >= med_g]),
        "bright_flux": _fit_slope(log_f[g_arr < med_g], g_arr[g_arr < med_g]),
        "faint_flux_large": _fit_slope(log_fl[g_arr >= med_g], g_arr[g_arr >= med_g]),
        "bright_flux_large": _fit_slope(log_fl[g_arr < med_g], g_arr[g_arr < med_g]),
    }

    # N2 peak stats per star
    n2_stars = []
    sat_lim_default = 55705.0
    for sid in star_ids:
        if sid not in meta:
            continue
        peaks = [r["peak_max_adu"] for r in rows if r["star_id"] == sid and math.isfinite(r["peak_max_adu"])]
        if not peaks:
            continue
        pa = np.array(peaks)
        lim = float([r["saturate_limit_85"] for r in rows if r["star_id"] == sid][0])
        n2_stars.append({
            "star_id": sid,
            "phot_g": meta[sid],
            "peak_min": float(np.min(pa)),
            "peak_median": float(np.median(pa)),
            "peak_p95": float(np.percentile(pa, 95)),
            "peak_max": float(np.max(pa)),
            "n_frames": len(pa),
            "n_above_70pct_limit": int(np.sum(pa > 0.7 * lim)),
            "n_above_85pct_limit": int(np.sum(pa > 0.85 * lim)),
            "is_saturated_any": any(r["is_saturated"] for r in rows if r["star_id"] == sid),
            "likely_saturated_any": any(r["likely_saturated"] for r in rows if r["star_id"] == sid),
        })

    bright_m = g_arr < med_g
    lim_arr = np.where(np.isfinite(_col("saturate_limit_85")), _col("saturate_limit_85"), sat_lim_default)
    peak_arr = _col("peak_max_adu")
    exclude_m = bright_m & (peak_arr > 0.7 * lim_arr)
    keep_m = bright_m & ~exclude_m & np.isfinite(log_f)
    n2 = {
        "per_star": sorted(n2_stars, key=lambda t: t["phot_g"]),
        "bright_half_slope_flux_all": _fit_slope(log_f[bright_m & np.isfinite(log_f)], g_arr[bright_m & np.isfinite(log_f)]),
        "bright_half_slope_flux_exclude_peak_gt_70pct": _fit_slope(log_f[keep_m], g_arr[keep_m]),
        "n_excluded_star_frames": int(exclude_m.sum()),
        "n_kept_star_frames": int(keep_m.sum()),
    }

    # N3
    n3_b = np.array(n3_before)
    n3_a = np.array(n3_after)
    n3_gv = np.array(n3_g)
    log_b = np.log10(n3_b[n3_b > 0])
    log_a = np.log10(n3_a[n3_a > 0])
    gb = n3_gv[n3_b > 0]
    ga = n3_gv[n3_a > 0]
    surf_arr = np.array(n3_surf)
    p2p_arr = np.array(n3_p2p)
    resid_proc = log_f - (_fit_slope(log_f, g_arr)["slope"] * g_arr + _fit_slope(log_f, g_arr)["intercept"])
    m_p2p = np.isfinite(resid_proc) & np.isfinite(p2p_arr)
    n3 = {
        "n_star_frames": len(n3_before),
        "method": "flux_large 9.58 px on calibrated (before) vs processed (after) native frames; WCS from ra/dec",
        "slope_before_sky": _fit_slope(np.log10(n3_b), n3_gv),
        "slope_after_sky": _fit_slope(np.log10(n3_a), n3_gv),
        "slope_proc_csv_flux_large_aligned": _fit_slope(log_fl[np.isfinite(log_fl)], g_arr[np.isfinite(log_fl)]),
        "surface_at_star_vs_G": _fit_slope(surf_arr[np.isfinite(surf_arr)], n3_gv[np.isfinite(surf_arr)]),
        "pearson_residual_vs_p2p": float(np.corrcoef(resid_proc[m_p2p], p2p_arr[m_p2p])[0, 1]) if m_p2p.sum() >= 5 else float("nan"),
        "median_delta_log10_after_minus_before": float(np.median(np.log10(n3_a) - np.log10(n3_b))),
    }

    # N4 wing flux brightest stars
    bright_sids = sorted(star_ids, key=lambda s: meta.get(s, 99))[:3]
    n4 = []
    cal_path = draft_root / "calibrated/lights/NoFilter_60_2/BO_CVn_Light_001.fits"
    with fits.open(cal_path, memmap=False) as hdul:
        data = hdul[0].data.astype(np.float64)
        wcs = WCS(hdul[0].header)
    df0 = pd.read_csv(csvs[0], dtype={"catalog_id": str})
    for sid in bright_sids:
        pr = df0.loc[df0["catalog_id"] == sid].iloc[0]
        xcp, ycp = wcs.all_world2pix(float(pr["ra_deg"]), float(pr["dec_deg"]), 0)
        xc, yc = float(np.asarray(xcp).ravel()[0]), float(np.asarray(ycp).ravel()[0])
        # COG total to 12px from cache if available else measure
        from closure_step1b_differential_aperture import _ee_at_radius
        npz = np.load("tmp/closure_step1f_ee_cache.npz", allow_pickle=True)
        ee = npz["ee_cache"].item()[0][sid]
        ee8 = float(_ee_at_radius(ee["radii"], ee["ee"], STAMP_R))
        ee12 = float(_ee_at_radius(ee["radii"], ee["ee"], 12.0))
        f12 = _ap_flux(data, xc, yc, 12.0, sky_in, sky_out)
        if f12 and f12 > 0:
            f_total_est = f12 / ee12
            wing8 = f_total_est * (ee12 - ee8)  # flux between 8 and 12 px approx
            wing_frac = (ee12 - ee8) / ee12 if ee12 > 0 else float("nan")
        else:
            wing8, wing_frac = float("nan"), float("nan")
        p2p = float(qc.loc[qc["frame"] == 1, "sky_surface_p2p_adu"].iloc[0])
        n4.append({
            "star_id": sid,
            "phot_g": meta[sid],
            "stamp_r_px": STAMP_R,
            "ee_at_8px": ee8,
            "ee_at_12px": ee12,
            "estimated_wing_flux_outside_8px_adu": wing8,
            "wing_fraction_of_12px_flux": wing_frac,
            "sky_surface_p2p_frame001": p2p,
            "wing_to_p2p_ratio": float(wing8 / p2p) if math.isfinite(wing8) and p2p > 0 else float("nan"),
        })

    out = {
        "N1": {"bins_1mag": bins, "halves": n1_halves},
        "N2": n2,
        "N3": n3,
        "N4": {"brightest_stars_frame001": n4, "stamp_r_px": STAMP_R},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out}")
    print("N3 before", n3["slope_before_sky"].get("slope"), "after", n3["slope_after_sky"].get("slope"))
    print("N1 bright half flux", n1_halves["bright_flux"].get("slope"))


if __name__ == "__main__":
    main()
