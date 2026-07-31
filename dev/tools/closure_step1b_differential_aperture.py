#!/usr/bin/env python3
"""Closure Step 1b (A-1 repair): differential aperture systematic without FWHM ground truth.

Standalone harness -- does NOT import VYVAR aperture sizing code.

Usage:
  python dev/tools/closure_step1b_differential_aperture.py \\
    --draft Archive/Drafts/draft_000435_snapshot_skysurface_20260716 \\
    --out tmp/closure_step1b_results.json
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.modeling import fitting, models
from scipy import stats


FOCUS_ID = "1498135552633294976"
TABLE_FWHM = 2.395
MIN_FRAME_FRAC = 0.95
MAG_BINS = np.arange(8.0, 15.5, 0.5)  # G 8 .. 15 for SNR table coverage
MIN_STARS_PER_BIN = 3
MIN_SEP_PX = 15.0  # fixed image px -> arcsec via plate scale (not disputed FWHM)
DELTA_G_NEIGH = 5.0
COG_DR = 0.25
COG_RMAX = 12.0
SAT_FRAC = 0.85


def _mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.median(np.abs(x - np.median(x))))


def _arcsec_per_px(hdr: fits.Header) -> float:
    pc = hdr.get("PC1_1")
    if pc is not None:
        return abs(float(pc)) * 3600.0
    cd = hdr.get("CD1_1")
    if cd is not None:
        return abs(float(cd)) * 3600.0
    return 9.77  # BO CVn wide rig fallback


def _angular_sep_arcsec(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    ra1, dec1, ra2, dec2 = map(math.radians, (ra1, dec1, ra2, dec2))
    dra = ra2 - ra1
    ddec = dec2 - dec1
    a = math.sin(ddec / 2) ** 2 + math.cos(dec1) * math.cos(dec2) * math.sin(dra / 2) ** 2
    return math.degrees(2 * math.asin(min(1.0, math.sqrt(max(0.0, a))))) * 3600.0


def _load_all_proc(lights: Path) -> tuple[list[Path], pd.DataFrame, int]:
    csvs = sorted(lights.glob("proc_BO_CVn_Light_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No proc CSV in {lights}")
    n_frames = len(csvs)
    need = int(math.ceil(MIN_FRAME_FRAC * n_frames))
    presence: dict[str, int] = defaultdict(int)
    sat_fail: dict[str, bool] = defaultdict(bool)
    meta: dict[str, dict[str, float]] = {}
    for proc in csvs:
        df = pd.read_csv(proc, dtype={"catalog_id": str})
        sat = float(df["saturate_limit_adu"].iloc[0])
        ok = df["photometry_ok"].astype(str).str.lower().isin(("true", "1", "yes"))
        for _, row in df.loc[ok].iterrows():
            cid = str(row["catalog_id"])
            presence[cid] += 1
            peak = float(row["peak_max_adu"])
            if peak >= SAT_FRAC * sat:
                sat_fail[cid] = True
            if cid not in meta:
                meta[cid] = {
                    "phot_g_mean_mag": float(row.get("phot_g_mean_mag", float("nan"))),
                    "mag": float(row.get("mag", float("nan"))),
                    "catalog_mag": float(row.get("catalog_mag", float("nan"))),
                    "ra_deg": float(row.get("ra_deg", float("nan"))),
                    "dec_deg": float(row.get("dec_deg", float("nan"))),
                    "aperture_r_px": float(row.get("aperture_r_px", float("nan"))),
                }
    eligible = [
        cid
        for cid, n in presence.items()
        if n >= need and not sat_fail.get(cid, False)
    ]
    rows = []
    for cid in eligible:
        m = meta[cid]
        gmag = m["phot_g_mean_mag"]
        if not math.isfinite(gmag):
            gmag = m["mag"] if math.isfinite(m["mag"]) else m["catalog_mag"]
        rows.append({"catalog_id": cid, "phot_g": gmag, **m})
    catalog = pd.DataFrame(rows)
    return csvs, catalog, n_frames


def _growth_curve_flat_ok(ee: np.ndarray, radii: np.ndarray) -> bool:
    """Monotonic to 12 px, flattening, no shoulder in outer half."""
    if ee.size < 4:
        return False
    if np.any(np.diff(ee) < -0.02):  # large downward step
        return False
    mid = len(ee) // 2
    tail = ee[mid:]
    tr = radii[mid:]
    if tail[-1] <= 0:
        return False
    # flattening: last third slope small
    i0 = max(0, len(tail) * 2 // 3)
    if i0 >= len(tail) - 2:
        return True
    slope = (tail[-1] - tail[i0]) / max(tr[-1] - tr[i0], 1e-6)
    if slope > 0.08:  # still rising fast at end -> shoulder / neighbour
        return False
    return True


def _curve_of_growth(
    data: np.ndarray,
    x0: float,
    y0: float,
    *,
    radii: np.ndarray,
    annulus_inner_px: float = 25.0,
    annulus_outer_px: float = 45.0,
) -> dict[str, Any] | None:
    h, w = data.shape
    xc, yc = int(round(x0)), int(round(y0))
    max_r = int(math.ceil(float(np.max(radii)) + annulus_outer_px + 2))
    if xc - max_r < 0 or yc - max_r < 0 or xc + max_r >= w or yc + max_r >= h:
        return None
    yy, xx = np.mgrid[yc - max_r : yc + max_r + 1, xc - max_r : xc + max_r + 1]
    patch = data[yc - max_r : yc + max_r + 1, xc - max_r : xc + max_r + 1].astype(np.float64)
    dist = np.hypot(xx - xc, yy - yc)
    fluxes: list[float] = []
    for r in radii:
        ap = dist <= r
        sky_mask = (dist >= annulus_inner_px) & (dist <= annulus_outer_px)
        sky = float(np.median(patch[sky_mask])) if np.any(sky_mask) else 0.0
        flux = float(np.sum((patch - sky)[ap]))
        fluxes.append(max(flux, 0.0))
    arr = np.asarray(fluxes, dtype=np.float64)
    norm = arr[-1] if arr[-1] > 0 else 1.0
    ee = arr / norm
    return {"radii": radii, "ee": ee}


def _ee_at_radius(radii: np.ndarray, ee: np.ndarray, r_ap: float) -> float:
    if not math.isfinite(r_ap) or r_ap <= 0:
        return float("nan")
    return float(np.interp(r_ap, radii, ee, left=ee[0], right=ee[-1]))


def _r_at_ee(radii: np.ndarray, ee: np.ndarray, target: float) -> float:
    hit = np.where(ee >= target)[0]
    if hit.size == 0:
        return float("nan")
    i = int(hit[0])
    if i == 0:
        return float(radii[0])
    r0, r1 = radii[i - 1], radii[i]
    e0, e1 = ee[i - 1], ee[i]
    if e1 <= e0:
        return float(r1)
    t = (target - e0) / (e1 - e0)
    return float(r0 + t * (r1 - r0))


def _fit_profiles(
    data: np.ndarray,
    x0: float,
    y0: float,
    *,
    box: int,
    fwhm_hint: float,
) -> dict[str, Any]:
    """Gaussian2D+Const2D and Moffat2D+Const2D; report failure cause."""
    h, w = data.shape
    xc, yc = int(round(x0)), int(round(y0))
    out: dict[str, Any] = {"ok_gauss": False, "ok_moffat": False}
    if not (box <= xc < w - box and box <= yc < h - box):
        out["fail"] = "edge"
        return out
    cut = data[yc - box : yc + box + 1, xc - box : xc + box + 1].astype(np.float64)
    yy, xx = np.mgrid[0 : cut.shape[0], 0 : cut.shape[1]]
    bg = float(np.median(cut))
    amp = max(float(np.max(cut) - bg), 1.0)
    fitter = fitting.TRFLSQFitter()
    c0 = models.Const2D(amplitude=bg)
    # Step 1 failure: compound model uses x_stddev_0 not x_stddev; unbounded centroid drift
    g0 = models.Gaussian2D(
        amplitude=amp,
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
        sx = float(mg.x_stddev_0.value)
        sy = float(mg.y_stddev_0.value)
        fwhm_g = 0.5 * (2.355 * sx + 2.355 * sy)
        if math.isfinite(fwhm_g) and 0.5 < fwhm_g < 20:
            out["ok_gauss"] = True
            out["fwhm_gauss"] = fwhm_g
    except Exception as exc:  # noqa: BLE001
        out["gauss_error"] = f"{type(exc).__name__}: {exc}"

    m0 = models.Moffat2D(
        amplitude=amp,
        x_0=float(box),
        y_0=float(box),
        gamma=fwhm_hint / 2.0,
        alpha=3.0,
    )
    m0.x_0.bounds = (box - 2, box + 2)
    m0.y_0.bounds = (box - 2, box + 2)
    m0.gamma.bounds = (0.4, 15.0)
    m0.alpha.bounds = (1.5, 8.0)
    try:
        mm = fitter(m0 + c0, xx, yy, cut, maxiter=200)
        gamma = float(mm.gamma_0.value)
        alpha = float(mm.alpha_0.value)
        fwhm_m = 2.0 * gamma * math.sqrt(2.0 ** (1.0 / alpha) - 1.0)
        if math.isfinite(fwhm_m) and 0.5 < fwhm_m < 20:
            out["ok_moffat"] = True
            out["fwhm_moffat"] = fwhm_m
            out["moffat_alpha"] = alpha
            out["moffat_gamma"] = gamma
            out["moffat_beta"] = alpha  # astropy alpha == beta in (1+(r/g)^2)^(-beta)
    except Exception as exc:  # noqa: BLE001
        out["moffat_error"] = f"{type(exc).__name__}: {exc}"
    return out


def _pick_fixed_stars(
    catalog: pd.DataFrame,
    *,
    arcsec_per_px: float,
    sample_fits: Path,
    sample_proc: Path,
    old_fwhm_table: float,
) -> dict[str, Any]:
    """Build fixed star set with isolation + growth-curve QC on sample frame."""
    with fits.open(sample_fits, memmap=False) as hdul:
        data = hdul[0].data.astype(np.float64)
    sample_df = pd.read_csv(sample_proc, dtype={"catalog_id": str})
    pos = sample_df.set_index("catalog_id")
    min_sep_arcsec = MIN_SEP_PX * arcsec_per_px
    radii = np.arange(COG_DR, COG_RMAX + COG_DR / 2, COG_DR)

    # nearest neighbour angular isolation
    cat = catalog.dropna(subset=["ra_deg", "dec_deg", "phot_g"]).copy()
    cat = cat.reset_index(drop=True)
    isolated: set[str] = set()
    old_rule_ok: set[str] = set()
    growth_ok: set[str] = set()
    min_sep_old_px = 6.0 * old_fwhm_table

    for i, row in cat.iterrows():
        cid = row["catalog_id"]
        g_i = row["phot_g"]
        ra_i, dec_i = row["ra_deg"], row["dec_deg"]
        near = cat.loc[
            (cat.index != i)
            & ((cat["phot_g"] - g_i).abs() < DELTA_G_NEIGH)
        ]
        if near.empty:
            isolated.add(cid)
        else:
            seps = [
                _angular_sep_arcsec(ra_i, dec_i, r["ra_deg"], r["dec_deg"])
                for _, r in near.iterrows()
            ]
            if min(seps) > min_sep_arcsec:
                isolated.add(cid)
        if cid not in pos.index:
            continue
        pr = pos.loc[cid]
        if isinstance(pr, pd.DataFrame):
            pr = pr.iloc[0]
        x, y = float(pr["x"]), float(pr["y"])
        # old circular px rule among proc stars on sample frame
        others = sample_df[
            sample_df["catalog_id"].astype(str) != cid
        ]
        if all(
            math.hypot(x - float(r["x"]), y - float(r["y"])) >= min_sep_old_px
            for _, r in others.iterrows()
        ):
            old_rule_ok.add(cid)
        cog = _curve_of_growth(data, x, y, radii=radii)
        if cog and _growth_curve_flat_ok(cog["ee"], cog["radii"]):
            growth_ok.add(cid)

    passed = isolated & growth_ok & set(cat["catalog_id"])
    rejected_by_new = old_rule_ok - passed

    # bin selection
    picked: list[str] = []
    bin_log: dict[str, list[str]] = {}
    pool = cat[cat["catalog_id"].isin(passed)].copy()
    if FOCUS_ID in passed:
        picked.append(FOCUS_ID)
    elif FOCUS_ID in set(catalog["catalog_id"]):
        picked.append(FOCUS_ID)  # force include even if failed QC (report in C.1)

    for mb in MAG_BINS:
        lo, hi = mb, mb + 0.5
        sub = pool[(pool["phot_g"] >= lo) & (pool["phot_g"] < hi)]
        sub = sub.sort_values("phot_g")
        ids = sub["catalog_id"].head(MIN_STARS_PER_BIN).tolist()
        bin_log[f"{lo:.1f}"] = ids
        for cid in ids:
            if cid not in picked:
                picked.append(cid)

    return {
        "n_eligible": len(catalog),
        "n_isolated_angular": len(isolated),
        "n_growth_ok": len(growth_ok),
        "n_pass_all": len(passed),
        "n_old_rule": len(old_rule_ok),
        "n_old_admitted_new_rejects": len(rejected_by_new),
        "min_sep_arcsec": min_sep_arcsec,
        "min_sep_px": MIN_SEP_PX,
        "arcsec_per_px": arcsec_per_px,
        "star_ids": picked,
        "bins": bin_log,
        "focus_in_qc": FOCUS_ID in passed,
    }


def _regress_through_origin(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    m = np.isfinite(x) & np.isfinite(y) & (x != 0)
    x, y = x[m], y[m]
    if x.size < 5:
        return {"slope": float("nan"), "frac_scatter": float("nan"), "spearman": float("nan"), "n": int(x.size)}
    slope = float(np.sum(x * y) / np.sum(x * x))
    resid = y - slope * x
    frac = _mad(resid) / abs(slope) if slope != 0 else float("nan")
    sp, _ = stats.spearmanr(x, y)
    return {"slope": slope, "frac_scatter": frac, "spearman": float(sp), "n": int(x.size)}


def _snr_table_for_frame(
    fwhm_px: float,
    sky_adu: float,
    *,
    gain: float,
    rn: float,
    zero_point: float,
    bkg_var: float | None = None,
) -> dict[float, float]:
    sigma = fwhm_px / 2.355
    r_min, r_max = 0.8 * fwhm_px, 2.5 * fwhm_px
    r_values = np.arange(r_min, r_max, 0.05)
    table: dict[float, float] = {}
    for mag in np.arange(7.0, 18.5, 0.5):
        flux_total = 10.0 ** ((zero_point - mag) / 2.5)
        best_snr, best_r = -1.0, float(r_values[0])
        for r in r_values:
            enc = flux_total * (1.0 - math.exp(-(r**2) / (2 * sigma**2)))
            area = math.pi * r**2
            n_photon = enc / gain
            if bkg_var is not None and math.isfinite(bkg_var):
                n_bkg = area * bkg_var / gain
            else:
                n_bkg = area * sky_adu / gain + area * (rn / gain) ** 2
            snr = n_photon / math.sqrt(max(n_photon + n_bkg, 1e-12))
            if snr > best_snr:
                best_snr, best_r = snr, float(r)
        table[round(float(mag), 1)] = round(best_r, 3)
    return table


def _mag_for_aperture(row: pd.Series) -> float:
    for col in ("mag", "catalog_mag", "lc_median_mag", "phot_g_mean_mag"):
        if col in row.index:
            v = float(pd.to_numeric(row[col], errors="coerce"))
            if math.isfinite(v):
                return v
    return float("nan")


def _lookup_table_r(mag: float, table: dict[float, float]) -> float:
    if not math.isfinite(mag):
        return float("nan")
    keys = [float(k) for k in table.keys()]
    if not keys:
        return float("nan")
    nearest = min(keys, key=lambda m: abs(m - mag))
    return float(table[nearest])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("tmp/closure_step1b_results.json"))
    args = ap.parse_args()
    draft = args.draft.resolve()
    lights = draft / "detrended_aligned/lights/NoFilter_60_2"
    snr_path = draft / "aperture_snr_table.json"
    with snr_path.open(encoding="utf-8") as f:
        snr_disk = json.load(f)

    csvs, catalog, n_frames = _load_all_proc(lights)
    sample_fits = lights / csvs[0].name.replace(".csv", ".fits")
    with fits.open(sample_fits, memmap=False) as hdul:
        arcsec_px = _arcsec_per_px(hdul[0].header)

    star_pick = _pick_fixed_stars(
        catalog,
        arcsec_per_px=arcsec_px,
        sample_fits=sample_fits,
        sample_proc=csvs[0],
        old_fwhm_table=TABLE_FWHM,
    )
    star_ids = star_pick["star_ids"]
    radii = np.arange(COG_DR, COG_RMAX + COG_DR / 2, COG_DR)

    # per-frame measurements
    r50_series: list[float] = []
    vy_series: list[float] = []
    moment_med: list[float] = []
    gauss_med: list[float] = []
    moffat_med: list[float] = []
    sky_med: list[float] = []
    frame_names: list[str] = []
    fit_gauss_n = fit_moffat_n = fit_attempts = 0
    step1_fail_cause = (
        "Step 1 harness used getattr(m.x_stddev) on Gaussian2D+Const2D compound model "
        "(parameters are x_stddev_0/y_stddev_0); unbounded centroid allowed divergent fits."
    )

    # store growth curves: frame_idx -> catalog_id -> ee curve
    ee_cache: dict[int, dict[str, dict[str, Any]]] = {}
    aperture_by_frame: dict[int, dict[str, float]] = {}

    box = max(12, int(math.ceil(8.0 * 4.0)))  # 8 x ~4 px scale proxy

    for fi, proc in enumerate(csvs):
        fits_path = lights / proc.name.replace(".csv", ".fits")
        df = pd.read_csv(proc, dtype={"catalog_id": str})
        df = df.set_index("catalog_id")
        with fits.open(fits_path, memmap=False) as hdul:
            data = hdul[0].data.astype(np.float64)
            vy = float(hdul[0].header.get("VY_FWHM", float("nan")))
        frame_names.append(proc.name)
        vy_series.append(vy)
        r50_stars: list[float] = []
        moments: list[float] = []
        fwhm_g_list: list[float] = []
        fwhm_m_list: list[float] = []
        skies: list[float] = []
        ee_cache[fi] = {}
        aperture_by_frame[fi] = {}
        for cid in star_ids:
            if cid not in df.index:
                continue
            row = df.loc[cid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            x, y = float(row["x"]), float(row["y"])
            aperture_by_frame[fi][cid] = float(row.get("aperture_r_px", float("nan")))
            if "fwhm_estimate_px" in row.index:
                fv = float(row["fwhm_estimate_px"])
                if math.isfinite(fv):
                    moments.append(fv)
            if "sky_adu_per_px_annulus" in row.index:
                sv = float(row["sky_adu_per_px_annulus"])
                if math.isfinite(sv):
                    skies.append(sv)
            cog = _curve_of_growth(data, x, y, radii=radii)
            if cog is None:
                continue
            ee_cache[fi][cid] = cog
            r50 = _r_at_ee(cog["radii"], cog["ee"], 0.5)
            if math.isfinite(r50):
                r50_stars.append(r50)
            fit_attempts += 1
            pf = _fit_profiles(data, x, y, box=box, fwhm_hint=4.0)
            if pf.get("ok_gauss"):
                fit_gauss_n += 1
                fwhm_g_list.append(pf["fwhm_gauss"])
            if pf.get("ok_moffat"):
                fit_moffat_n += 1
                fwhm_m_list.append(pf["fwhm_moffat"])
        r50_series.append(float(np.median(r50_stars)) if r50_stars else float("nan"))
        moment_med.append(float(np.median(moments)) if moments else float("nan"))
        gauss_med.append(float(np.median(fwhm_g_list)) if fwhm_g_list else float("nan"))
        moffat_med.append(float(np.median(fwhm_m_list)) if fwhm_m_list else float("nan"))
        sky_med.append(float(np.median(skies)) if skies else float("nan"))

    r50_arr = np.array(r50_series, dtype=np.float64)
    vy_arr = np.array(vy_series, dtype=np.float64)
    moment_arr = np.array(moment_med, dtype=np.float64)
    gauss_arr = np.array(gauss_med, dtype=np.float64)
    moffat_arr = np.array(moffat_med, dtype=np.float64)

    tracking = {
        "VY_FWHM": _regress_through_origin(r50_arr, vy_arr),
        "moment_median": _regress_through_origin(r50_arr, moment_arr),
        "gauss_fit_median": _regress_through_origin(r50_arr, gauss_arr),
        "moffat_fit_median": _regress_through_origin(r50_arr, moffat_arr),
    }
    best_est = min(
        (k for k in ("VY_FWHM", "moment_median", "gauss_fit_median", "moffat_fit_median")),
        key=lambda k: tracking[k]["frac_scatter"] if math.isfinite(tracking[k]["frac_scatter"]) else 1e9,
    )
    est_map = {
        "VY_FWHM": vy_arr,
        "moment_median": moment_arr,
        "gauss_fit_median": gauss_arr,
        "moffat_fit_median": moffat_arr,
    }
    scale_series = est_map[best_est].copy()
    med_scale = float(np.nanmedian(scale_series))
    scale_adj = scale_series * (TABLE_FWHM / med_scale) if med_scale > 0 else scale_series

    # B.3 delta_ap
    focus_mag = float(
        catalog.loc[catalog["catalog_id"] == FOCUS_ID, "phot_g"].iloc[0]
        if (catalog["catalog_id"] == FOCUS_ID).any()
        else float("nan")
    )

    def _comp_subsets(catalog_df: pd.DataFrame, ids: list[str]) -> dict[str, list[str]]:
        sub: dict[str, list[str]] = {"G8_9": [], "G9_11": [], "G_gt_11": []}
        for cid in ids:
            if cid == FOCUS_ID:
                continue
            row = catalog_df.loc[catalog_df["catalog_id"] == cid]
            if row.empty:
                continue
            g = float(row.iloc[0]["phot_g"])
            if not math.isfinite(g):
                continue
            if 8.0 <= g < 9.0:
                sub["G8_9"].append(cid)
            elif 9.0 <= g < 11.0:
                sub["G9_11"].append(cid)
            elif g >= 11.0:
                sub["G_gt_11"].append(cid)
        return sub

    comp_subs = _comp_subsets(catalog, star_ids)

    def _delta_ap_series(comp_list: list[str]) -> np.ndarray:
        deltas = []
        for fi in range(n_frames):
            if FOCUS_ID not in ee_cache[fi]:
                deltas.append(float("nan"))
                continue
            rap_t = aperture_by_frame[fi].get(FOCUS_ID, float("nan"))
            ee_t = _ee_at_radius(
                ee_cache[fi][FOCUS_ID]["radii"],
                ee_cache[fi][FOCUS_ID]["ee"],
                rap_t,
            )
            ee_c: list[float] = []
            for cid in comp_list:
                if cid not in ee_cache[fi]:
                    continue
                rap = aperture_by_frame[fi].get(cid, float("nan"))
                ee_c.append(
                    _ee_at_radius(
                        ee_cache[fi][cid]["radii"],
                        ee_cache[fi][cid]["ee"],
                        rap,
                    )
                )
            if not ee_c or not math.isfinite(ee_t):
                deltas.append(float("nan"))
                continue
            med_c = float(np.median(ee_c))
            if med_c <= 0 or ee_t <= 0:
                deltas.append(float("nan"))
                continue
            deltas.append(-2.5 * math.log10(ee_t / med_c))
        return np.array(deltas, dtype=np.float64)

    delta_ap: dict[str, Any] = {}
    for label, clist in comp_subs.items():
        d = _delta_ap_series(clist)
        valid = d[np.isfinite(d)]
        idx_best = int(np.nanargmin(r50_arr))
        idx_worst = int(np.nanargmax(r50_arr))
        slope, _ = (0.0, 0.0)
        if valid.size >= 5:
            m = np.isfinite(d) & np.isfinite(r50_arr)
            if m.sum() >= 5:
                slope = float(np.polyfit(r50_arr[m], d[m], 1)[0])
        delta_ap[label] = {
            "n_comps": len(clist),
            "delta_ap_all_frames": d.tolist(),
            "range_best_worst_mmag": float(d[idx_worst] - d[idx_best]) if np.isfinite(d[idx_worst]) and np.isfinite(d[idx_best]) else float("nan"),
            "min_r50_frame": frame_names[idx_best],
            "max_r50_frame": frame_names[idx_worst],
            "slope_mmag_per_r50": slope,
            "corr_with_r50": float(np.corrcoef(r50_arr[np.isfinite(d)], d[np.isfinite(d)])[0, 1])
            if np.isfinite(d).sum() >= 5
            else float("nan"),
            "median_mmag": float(np.median(valid)) if valid.size else float("nan"),
        }

    # B.5 frozen k_i counterfactual
    k_i: dict[str, float] = {}
    for cid in star_ids:
        row = catalog.loc[catalog["catalog_id"] == cid]
        if row.empty:
            continue
        rap = float(row.iloc[0]["aperture_r_px"])
        k_i[cid] = rap / TABLE_FWHM if math.isfinite(rap) else float("nan")

    def _delta_ap_counterfactual(use_frozen_k: bool, per_frame_reopt: bool) -> dict[str, float]:
        out_sub: dict[str, float] = {}
        for label, clist in comp_subs.items():
            deltas = []
            for fi in range(n_frames):
                sc = scale_adj[fi] if math.isfinite(scale_adj[fi]) else float("nan")
                if FOCUS_ID not in ee_cache[fi]:
                    continue
                if per_frame_reopt:
                    fw = vy_arr[fi] if math.isfinite(vy_arr[fi]) else TABLE_FWHM
                    sky = sky_med[fi] if math.isfinite(sky_med[fi]) else float(snr_disk["sky_adu_per_px"])
                    tbl = _snr_table_for_frame(
                        fw, sky, gain=float(snr_disk["gain"]), rn=float(snr_disk["read_noise"]), zero_point=25.0
                    )
                    df = pd.read_csv(csvs[fi], dtype={"catalog_id": str})
                    row_t = df.loc[df["catalog_id"] == FOCUS_ID].iloc[0]
                    mag_t = _mag_for_aperture(row_t)
                    rap_t = _lookup_table_r(mag_t, tbl)
                elif use_frozen_k:
                    rap_t = k_i.get(FOCUS_ID, float("nan")) * sc
                else:
                    rap_t = aperture_by_frame[fi].get(FOCUS_ID, float("nan"))
                ee_t = _ee_at_radius(
                    ee_cache[fi][FOCUS_ID]["radii"],
                    ee_cache[fi][FOCUS_ID]["ee"],
                    rap_t,
                )
                ee_c = []
                for cid in clist:
                    if cid not in ee_cache[fi]:
                        continue
                    if per_frame_reopt:
                        df = pd.read_csv(csvs[fi], dtype={"catalog_id": str})
                        row_c = df.loc[df["catalog_id"] == cid].iloc[0]
                        mag_c = _mag_for_aperture(row_c)
                        tbl = _snr_table_for_frame(
                            vy_arr[fi] if math.isfinite(vy_arr[fi]) else TABLE_FWHM,
                            sky_med[fi] if math.isfinite(sky_med[fi]) else float(snr_disk["sky_adu_per_px"]),
                            gain=float(snr_disk["gain"]),
                            rn=float(snr_disk["read_noise"]),
                            zero_point=25.0,
                        )
                        rap = _lookup_table_r(mag_c, tbl)
                    elif use_frozen_k:
                        rap = k_i.get(cid, float("nan")) * sc
                    else:
                        rap = aperture_by_frame[fi].get(cid, float("nan"))
                    ee_c.append(
                        _ee_at_radius(
                            ee_cache[fi][cid]["radii"],
                            ee_cache[fi][cid]["ee"],
                            rap,
                        )
                    )
                if ee_c and math.isfinite(ee_t):
                    med_c = float(np.median(ee_c))
                    if med_c > 0 and ee_t > 0:
                        deltas.append(-2.5 * math.log10(ee_t / med_c))
            d = np.array(deltas, dtype=np.float64)
            v = d[np.isfinite(d)]
            idx_b, idx_w = int(np.nanargmin(r50_arr)), int(np.nanargmax(r50_arr))
            out_sub[label] = {
                "range_best_worst_mmag": float(d[idx_w] - d[idx_b]) if len(v) else float("nan"),
                "median_mmag": float(np.median(v)) if v.size else float("nan"),
            }
        return out_sub

    b5 = _delta_ap_counterfactual(use_frozen_k=True, per_frame_reopt=False)
    b6_full = {}
    for label, clist in comp_subs.items():
        dre = []
        for fi in range(n_frames):
            if FOCUS_ID not in ee_cache[fi]:
                dre.append(float("nan"))
                continue
            fw = vy_arr[fi] if math.isfinite(vy_arr[fi]) else TABLE_FWHM
            sky = sky_med[fi] if math.isfinite(sky_med[fi]) else float(snr_disk["sky_adu_per_px"])
            tbl = _snr_table_for_frame(
                fw, sky, gain=float(snr_disk["gain"]), rn=float(snr_disk["read_noise"]), zero_point=25.0
            )
            df = pd.read_csv(csvs[fi], dtype={"catalog_id": str})
            row_t = df.loc[df["catalog_id"] == FOCUS_ID].iloc[0]
            rap_t = _lookup_table_r(_mag_for_aperture(row_t), tbl)
            ee_t = _ee_at_radius(ee_cache[fi][FOCUS_ID]["radii"], ee_cache[fi][FOCUS_ID]["ee"], rap_t)
            ee_c = []
            for cid in clist:
                if cid not in ee_cache[fi]:
                    continue
                row_c = df.loc[df["catalog_id"] == cid].iloc[0]
                rap = _lookup_table_r(_mag_for_aperture(row_c), tbl)
                ee_c.append(_ee_at_radius(ee_cache[fi][cid]["radii"], ee_cache[fi][cid]["ee"], rap))
            if not ee_c or not math.isfinite(ee_t) or ee_t <= 0:
                dre.append(float("nan"))
                continue
            med_c = float(np.median(ee_c))
            if med_c <= 0:
                dre.append(float("nan"))
                continue
            dre.append(-2.5 * math.log10(ee_t / med_c))
        dre = np.array(dre, dtype=np.float64)
        m = np.isfinite(dre) & np.isfinite(np.array(sky_med))
        b6_full[label] = {
            "range_best_worst_mmag": float(np.nanmax(dre) - np.nanmin(dre)) if np.isfinite(dre).any() else float("nan"),
            "corr_with_sky": float(np.corrcoef(np.array(sky_med)[m], dre[m])[0, 1]) if m.sum() >= 5 else float("nan"),
        }

    # C.1 focus target forensics on 4 frames
    forensic_frames = ["proc_BO_CVn_Light_002.csv", "proc_BO_CVn_Light_063.csv", "proc_BO_CVn_Light_087.csv", "proc_BO_CVn_Light_080.csv"]
    c1: dict[str, Any] = {}
    comp_curve_med = None
    for fn in forensic_frames:
        fi = frame_names.index(fn) if fn in frame_names else None
        if fi is None:
            continue
        proc = csvs[fi]
        df = pd.read_csv(proc, dtype={"catalog_id": str})
        row = df.loc[df["catalog_id"] == FOCUS_ID]
        if row.empty:
            continue
        row = row.iloc[0]
        c1[fn] = {
            "fwhm_estimate_px": float(row.get("fwhm_estimate_px", float("nan"))),
            "x": float(row["x"]),
            "y": float(row["y"]),
            "peak_max_adu": float(row["peak_max_adu"]),
            "aperture_r_px": float(row.get("aperture_r_px", float("nan"))),
        }
        if FOCUS_ID in ee_cache[fi]:
            c1[fn]["ee_curve"] = ee_cache[fi][FOCUS_ID]["ee"].tolist()
    # nearest neighbour for focus
    frow = catalog.loc[catalog["catalog_id"] == FOCUS_ID]
    if not frow.empty:
        ra_f, dec_f = float(frow.iloc[0]["ra_deg"]), float(frow.iloc[0]["dec_deg"])
        g_f = float(frow.iloc[0]["phot_g"])
        others = catalog[catalog["catalog_id"] != FOCUS_ID].copy()
        others["sep"] = others.apply(
            lambda r: _angular_sep_arcsec(ra_f, dec_f, r["ra_deg"], r["dec_deg"]), axis=1
        )
        others["dg"] = (others["phot_g"] - g_f).abs()
        near = others.sort_values("sep").head(5)
        c1["nearest_neighbours"] = near[["catalog_id", "phot_g", "sep", "dg"]].to_dict("records")

    # C.2 sizing mag for focus
    proc063 = draft / "detrended_aligned/lights/NoFilter_60_2/proc_BO_CVn_Light_063.csv"
    df063 = pd.read_csv(proc063, dtype={"catalog_id": str})
    f063 = df063.loc[df063["catalog_id"] == FOCUS_ID].iloc[0]
    sizing_mag = _mag_for_aperture(f063)

    # C.4 ZP recomputation table
    zp_table = _snr_table_for_frame(
        TABLE_FWHM,
        float(snr_disk["sky_adu_per_px"]),
        gain=float(snr_disk["gain"]),
        rn=float(snr_disk["read_noise"]),
        zero_point=21.68,
    )

    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "draft": str(draft),
        "n_frames": n_frames,
        "step1_fit_failure_cause": step1_fail_cause,
        "part_a": {
            **star_pick,
            "fit_convergence": {
                "attempts": fit_attempts,
                "gauss_ok": fit_gauss_n,
                "moffat_ok": fit_moffat_n,
                "gauss_rate": fit_gauss_n / fit_attempts if fit_attempts else 0.0,
                "moffat_rate": fit_moffat_n / fit_attempts if fit_attempts else 0.0,
            },
        },
        "part_b": {
            "r50_frame": {
                "series": r50_series,
                "frame_names": frame_names,
                "min": float(np.nanmin(r50_arr)),
                "median": float(np.nanmedian(r50_arr)),
                "max": float(np.nanmax(r50_arr)),
            },
            "tracking": tracking,
            "best_scale_estimator": best_est,
            "delta_ap": delta_ap,
            "delta_ap_frozen_k_option_i": b5,
            "delta_ap_reopt_per_frame": b6_full,
        },
        "part_c": {
            "focus_forensics": c1,
            "focus_sizing_mag": sizing_mag,
            "focus_phot_g": focus_mag,
            "focus_aperture_r_on_disk": float(f063.get("aperture_r_px", float("nan"))),
            "snr_table_at_zp_21_68": zp_table,
            "snr_table_on_disk": snr_disk["table"],
            "masterstar_vy_fwhm_mechanism": "pipeline.py:2988-2990 overwrites VY_FWHM with median processed-set FWHM",
        },
        "focus_id": FOCUS_ID,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out}")
    print(f"Stars: {len(star_ids)}  gauss rate: {fit_gauss_n}/{fit_attempts}")


if __name__ == "__main__":
    main()
