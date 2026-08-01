#!/usr/bin/env python3
"""Step 1m: COG-normalise flux; settle D5-2 mechanism (M1-M4).

Diagnostic only. No production change.

Usage:
  python dev/tools/closure_step1m_cog_normalise_flux.py \\
    --draft Archive/Drafts/draft_000435_snapshot_skysurface_20260716 \\
    --step1b-json tmp/closure_step1b_results.json \\
    --cache tmp/closure_step1f_ee_cache.npz \\
    --out tmp/closure_step1m_diagnostics.json
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
from scipy import stats

from closure_step1b_differential_aperture import TABLE_FWHM, _ee_at_radius, _load_all_proc

warnings.filterwarnings("ignore")

# Multi-aperture radii: pipeline.py aperture_fwhm_factor_small/large x frame FWHM
R_SMALL_FWHM = 1.5
R_LARGE_FWHM = 4.0


def _fit_slope(y: np.ndarray, g: np.ndarray) -> dict[str, float]:
    m = np.isfinite(y) & np.isfinite(g)
    if m.sum() < 5:
        return {"n": int(m.sum()), "insufficient": True}
    slope, intercept, r, _, se = stats.linregress(g[m], y[m])
    return {
        "n": int(m.sum()),
        "slope": float(slope),
        "slope_se": float(se),
        "intercept": float(intercept),
        "r_squared": float(r**2),
    }


def _vif_g_aperture(g: np.ndarray, ap: np.ndarray) -> dict[str, float]:
    m = np.isfinite(g) & np.isfinite(ap)
    if m.sum() < 5:
        return {"insufficient": True}
    slope, intercept, r, _, _ = stats.linregress(g[m], ap[m])
    r2 = float(r**2)
    return {
        "n": int(m.sum()),
        "pearson_G_aperture_r": float(r if slope < 0 else -r if r < 0 else r),
        "r_squared_G_on_aperture": r2,
        "VIF_aperture_r": float(1.0 / max(1.0 - r2, 1e-12)),
        "corr_G_aperture": float(np.corrcoef(g[m], ap[m])[0, 1]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", type=Path, required=True)
    ap.add_argument("--step1b-json", type=Path, default=Path("tmp/closure_step1b_results.json"))
    ap.add_argument("--cache", type=Path, default=Path("tmp/closure_step1f_ee_cache.npz"))
    ap.add_argument("--out", type=Path, default=Path("tmp/closure_step1m_diagnostics.json"))
    args = ap.parse_args()

    draft = args.draft.resolve()
    lights = draft / "detrended_aligned/lights/NoFilter_60_2"
    csvs, catalog_full, _ = _load_all_proc(lights)

    with args.step1b_json.open(encoding="utf-8") as f:
        s1b = json.load(f)
    star_ids: list[str] = s1b["part_a"]["star_ids"]

    meta: dict[str, float] = {}
    for sid in star_ids:
        row = catalog_full[catalog_full["catalog_id"].astype(str) == sid]
        if row.empty:
            continue
        meta[sid] = float(row.iloc[0]["phot_g"])

    npz = np.load(args.cache, allow_pickle=True)
    ee_cache = npz["ee_cache"].item()
    r50_cache = np.array(npz["r50_series"], dtype=np.float64)

    rows: list[dict[str, Any]] = []
    for fi, proc in enumerate(csvs):
        df = pd.read_csv(proc, dtype={"catalog_id": str})
        # frame-level FWHM for multi-aperture (draft uses constant TABLE_FWHM)
        fw_frame = TABLE_FWHM
        r_small = R_SMALL_FWHM * fw_frame
        r_large = R_LARGE_FWHM * fw_frame

        for sid in star_ids:
            if sid not in ee_cache.get(fi, {}):
                continue
            pr = df.loc[df["catalog_id"] == sid]
            if pr.empty:
                continue
            pr = pr.iloc[0]
            ok = str(pr.get("photometry_ok", "true")).lower() in ("true", "1", "yes")
            usable = str(pr.get("is_usable", "true")).lower() in ("true", "1", "yes")
            if not (ok and usable):
                continue

            cached = ee_cache[fi][sid]
            radii = cached["radii"]
            ee = cached["ee"]
            rap = float(pr.get("aperture_r_px", float("nan")))
            if not math.isfinite(rap):
                continue

            ee_ap = float(_ee_at_radius(radii, ee, rap))
            ee_sm = float(_ee_at_radius(radii, ee, r_small))
            ee_lg = float(_ee_at_radius(radii, ee, r_large))
            if ee_ap <= 0 or not math.isfinite(ee_ap):
                continue

            g = meta[sid]
            row = {
                "star_id": sid,
                "frame_idx": fi,
                "phot_g": g,
                "mag": float(pr.get("mag", g)),
                "aperture_r_px": rap,
                "ee_at_aperture": ee_ap,
                "ee_at_r_small": ee_sm,
                "ee_at_r_large": ee_lg,
                "r50_frame": float(r50_cache[fi]),
                "fwhm_estimate_px": float(pr.get("fwhm_estimate_px", float("nan"))),
            }
            for col, ee_r in [
                ("flux", ee_ap),
                ("dao_flux", ee_ap),
                ("flux_small", ee_sm),
                ("flux_large", ee_lg),
            ]:
                fv = float(pr.get(col, float("nan")))
                if math.isfinite(fv) and fv > 0 and math.isfinite(ee_r) and ee_r > 0:
                    row[f"log10_{col}_raw"] = math.log10(fv)
                    row[f"log10_{col}_corr"] = math.log10(fv) - math.log10(ee_r)
            rows.append(row)

    # M1 fits
    g_arr = np.array([r["phot_g"] for r in rows])
    ee_ap_arr = np.array([r["ee_at_aperture"] for r in rows])
    log_ee_ap = np.log10(ee_ap_arr)

    m1: dict[str, Any] = {
        "N_star_frames": len(rows),
        "slope_log10_EE_at_aperture_vs_G": _fit_slope(log_ee_ap, g_arr),
        "reference_radius_px": 12.0,
        "normalisation": "log10(F_corr) = log10(F) - log10(EE_measured(r))",
        "columns": {},
    }

    for col in ("flux", "dao_flux", "flux_small", "flux_large"):
        raw = np.array([r.get(f"log10_{col}_raw", float("nan")) for r in rows])
        corr = np.array([r.get(f"log10_{col}_corr", float("nan")) for r in rows])
        m1["columns"][col] = {
            "slope_raw_vs_G": _fit_slope(raw, g_arr),
            "slope_corr_vs_G": _fit_slope(corr, g_arr),
        }
        if col == "flux":
            m1["columns"][col]["multi_aperture_r_px"] = {
                "flux": "aperture_r_px per star",
                "dao_flux": "aperture_r_px per star",
                "flux_small": R_SMALL_FWHM * TABLE_FWHM,
                "flux_large": R_LARGE_FWHM * TABLE_FWHM,
            }

    # M2 VIF
    ap_arr = np.array([r["aperture_r_px"] for r in rows])
    m2 = {
        "note": "Joint G + aperture_r_px OLS retired as evidence (collinear instability)",
        "vif": _vif_g_aperture(g_arr, ap_arr),
        "joint_fit_G_slope_reported_step1l": 0.048,
    }

    # M3 fwhm_estimate_px per star across frames
    fwhm_by_star: dict[str, list[float]] = {}
    for r in rows:
        sid = r["star_id"]
        fv = r["fwhm_estimate_px"]
        if math.isfinite(fv):
            fwhm_by_star.setdefault(sid, []).append(fv)

    m3_samples = []
    for sid in star_ids[:8]:
        vals = fwhm_by_star.get(sid, [])
        if vals:
            u = sorted(set(round(v, 6) for v in vals))
            m3_samples.append({
                "star_id": sid,
                "phot_g": meta.get(sid),
                "n_frames": len(vals),
                "n_unique": len(u),
                "min": float(min(vals)),
                "max": float(max(vals)),
                "constant_within_star": len(u) == 1,
            })

    m3 = {
        "summary": {
            "stars_with_data": len(fwhm_by_star),
            "stars_constant": sum(1 for v in fwhm_by_star.values() if len(set(v)) == 1),
            "stars_varying": sum(1 for v in fwhm_by_star.values() if len(set(v)) > 1),
        },
        "sample_stars": m3_samples,
        "residual_vs_fwhm_estimate_pearson": float(
            np.corrcoef(
                np.array([r.get("log10_flux_raw", float("nan")) for r in rows])
                - (_fit_slope(np.array([r.get("log10_flux_raw", float("nan")) for r in rows]), g_arr).get("slope", 0) * g_arr
                    + _fit_slope(np.array([r.get("log10_flux_raw", float("nan")) for r in rows]), g_arr).get("intercept", 0)),
                np.array([r["fwhm_estimate_px"] for r in rows]),
            )[0, 1]
        ) if len(rows) >= 5 else float("nan"),
    }
    # fix residual correlation properly
    raw_flux = np.array([r.get("log10_flux_raw", float("nan")) for r in rows])
    fit_raw = _fit_slope(raw_flux, g_arr)
    resid = raw_flux - (fit_raw["slope"] * g_arr + fit_raw["intercept"])
    m = np.isfinite(resid) & np.isfinite(np.array([r["fwhm_estimate_px"] for r in rows]))
    m3["residual_vs_fwhm_estimate_pearson"] = float(np.corrcoef(
        resid[m], np.array([r["fwhm_estimate_px"] for r in rows])[m]
    )[0, 1]) if m.sum() >= 5 else float("nan")

    # M4 r50 reconciliation
    r50_step1b = s1b.get("part_b", {}).get("r50_frame", {}).get("series", [])
    r50_b = np.array(r50_step1b, dtype=np.float64) if r50_step1b else np.array([])
    m4 = {
        "step1b_r50_frame": {
            "method": "integer-centre COG on proc x,y; all 35 stars per frame",
            "min": float(np.nanmin(r50_b)) if r50_b.size else float("nan"),
            "max": float(np.nanmax(r50_b)) if r50_b.size else float("nan"),
            "median": float(np.nanmedian(r50_b)) if r50_b.size else float("nan"),
            "n_frames": int(r50_b.size),
        },
        "step1f_cache_r50": {
            "method": "photutils COG + Gaussian centroid; C1-admissible stars only",
            "min": float(np.nanmin(r50_cache)),
            "max": float(np.nanmax(r50_cache)),
            "median": float(np.nanmedian(r50_cache)),
            "n_frames": int(r50_cache.size),
        },
        "explanation": (
            "Narrower span in step1f cache: (1) photutils vs integer-centre COG; "
            "(2) C1 admissibility drops stars before r50 median; "
            "(3) Gaussian centroid vs proc CSV x,y. Not fixture annulus (r50 from COG curve only)."
        ),
    }
    if r50_b.size == r50_cache.size:
        m4["per_frame_delta_median"] = {
            "mean_cache_minus_step1b": float(np.nanmean(r50_cache - r50_b)),
            "max_abs_delta": float(np.nanmax(np.abs(r50_cache - r50_b))),
        }

    out = {"M1": m1, "M2": m2, "M3": m3, "M4": m4}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    corr_slope = m1["columns"]["flux"]["slope_corr_vs_G"].get("slope")
    ee_slope = m1["slope_log10_EE_at_aperture_vs_G"].get("slope")
    print(f"Wrote {args.out}")
    print("M1 slope log10(EE(r_ap)) vs G:", ee_slope)
    print("M1 normalised flux slope:", corr_slope)
    for c in ("flux", "dao_flux", "flux_small", "flux_large"):
        print(f"  {c} corr:", m1["columns"][c]["slope_corr_vs_G"].get("slope"))


if __name__ == "__main__":
    main()
