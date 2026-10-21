#!/usr/bin/env python3
"""Measure draft 510 flux change: clipped sky (old) vs plain median (new)."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src_py"))

from astropy.io import fits as pyfits
from photometry_core import _aperture_flux_sky_per_star

DRAFT = REPO / "Archive/Drafts/draft_000510"
PROC = DRAFT / "detrended_aligned/lights/NoFilter_60_2"
PHOT = DRAFT / "platesolve/NoFilter_60_2/photometry"
TARGET = "1498613634033133184"
ANNULUS_INNER_FWHM = 4.75


def annulus_r_in(r_ap: float, fwhm_ap: float) -> float:
    return max(r_ap + 0.5, ANNULUS_INNER_FWHM * fwhm_ap)


def old_sky_pp(d: np.ndarray, ann_img: np.ndarray) -> float:
    sky_pixels = d[ann_img > 0]
    if sky_pixels.size >= 5:
        sky_med = float(np.median(sky_pixels))
        sky_std = float(np.std(sky_pixels))
        clipped = sky_pixels[sky_pixels < sky_med + 2.0 * sky_std]
        if clipped.size >= 5:
            return float(np.median(clipped))
        return sky_med
    return float(np.median(d))


def old_flux(d, x, y, rap, rin, rout):
    from photutils.aperture import CircularAnnulus, CircularAperture, aperture_photometry

    ap = CircularAperture(np.array([[x, y]]), r=rap)
    an = CircularAnnulus(np.array([[x, y]]), r_in=rin, r_out=rout)
    s = float(aperture_photometry(d, ap)["aperture_sum"][0])
    m = an.to_mask(method="center")
    m = m[0] if isinstance(m, (list, tuple)) else m
    sky = old_sky_pp(d, m.to_image(d.shape))
    return s - sky * float(ap.area)


def proc_to_fits(name: str) -> Path:
    stem = Path(name).name
    if stem.startswith("proc_"):
        stem = stem[5:]
    if stem.endswith(".csv"):
        stem = stem[:-4]
    return PROC / f"{stem}.fits"


def main() -> None:
    comps = pd.read_csv(PHOT / "comparison_stars_per_target.csv", dtype=str)
    cs = comps[comps["target_catalog_id"] == TARGET]["catalog_id"].astype(str).tolist()
    ids = set(cs) | {TARGET}
    import glob

    rows = []
    for proc_path in sorted(glob.glob(str(PROC / "proc_BO_CVn_Light_*.csv"))):
        df = pd.read_csv(proc_path, dtype=str)
        sub = df[df["catalog_id"].astype(str).isin(ids)]
        fits_path = proc_to_fits(proc_path)
        if not fits_path.is_file():
            continue
        with pyfits.open(fits_path, memmap=False) as h:
            d = np.ascontiguousarray(h[0].data, dtype=np.float64)
        if np.any(~np.isfinite(d)):
            fill = float(np.nanmedian(d)) if np.any(np.isfinite(d)) else 0.0
            d = np.where(np.isfinite(d), d, fill)
        for _, r in sub.iterrows():
            x, y = float(r["x"]), float(r["y"])
            rap = float(r["aperture_r_px"])
            rout = float(r["sky_annulus_r_out_px"])
            rin = annulus_r_in(rap, float(r["fwhm_px_for_aperture"]))
            stored = float(r["dao_flux"])
            f_old = old_flux(d, x, y, rap, rin, rout)
            f_new, _ = _aperture_flux_sky_per_star(
                d,
                np.array([[x, y]]),
                np.array([rap]),
                np.array([rin]),
                np.array([rout]),
            )
            f_new = float(f_new[0])
            rows.append(
                {
                    "frame": Path(proc_path).name,
                    "catalog_id": r["catalog_id"],
                    "stored_dao": stored,
                    "old_clip_recompute": f_old,
                    "new_median": f_new,
                    "delta_new_minus_old": f_new - f_old,
                    "frac_new_vs_old": (f_new - f_old) / f_old if f_old else float("nan"),
                }
            )

    out = pd.DataFrame(rows)
    out_path = REPO / "tmp" / "sky_clip_510_flux_delta.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    bo = out[out["catalog_id"] == TARGET]
    summary = {
        "n_rows": int(len(out)),
        "median_frac_all": float(out["frac_new_vs_old"].median()),
        "median_frac_target": float(bo["frac_new_vs_old"].median()),
        "median_frac_comps": float(out[out["catalog_id"] != TARGET]["frac_new_vs_old"].median()),
        "stored_vs_old_max_abs_frac": float((out["stored_dao"] - out["old_clip_recompute"]).abs().div(out["old_clip_recompute"]).max()),
        "by_star_median_frac": out.groupby("catalog_id")["frac_new_vs_old"].median().to_dict(),
        "output_csv": str(out_path),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
