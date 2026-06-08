"""
VYVAR Cross-Validation Script
Compares VYVAR aperture photometry vs independent photutils measurement.

Usage:
  python validate_photometry_crossval.py \
    --fits /path/to/proc_BO_CVn_Light_001.fits \
    --csv  /path/to/proc_BO_CVn_Light_001.csv \
    --masterstar /path/to/MASTERSTAR.fits \
    --output /path/to/crossval_result.csv

What it does:
  1. Load VYVAR proc_*.csv (x, y, catalog_id, dao_flux, aperture_r_px)
  2. Run photutils aperture_photometry on same FITS at same positions
  3. Use same aperture radius (from aperture_r_px column or default 4.0 px)
  4. Compare flux_vyvar vs flux_photutils per star
  5. Report: ratio, scatter, outliers, per-mag-bin statistics
"""

import argparse
import sys

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.aperture import CircularAnnulus, CircularAperture, aperture_photometry

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="VYVAR photometry cross-validator")
parser.add_argument("--fits", required=True, help="proc_*.fits file")
parser.add_argument("--csv", required=True, help="proc_*.csv file")
parser.add_argument("--output", default="crossval_result.csv")
parser.add_argument(
    "--aperture",
    type=float,
    default=None,
    help="Override aperture radius [px]. Default: from CSV.",
)
parser.add_argument(
    "--annulus_in",
    type=float,
    default=6.0,
    help="Sky annulus inner radius [px]",
)
parser.add_argument(
    "--annulus_out",
    type=float,
    default=10.0,
    help="Sky annulus outer radius [px]",
)
parser.add_argument("--gain", type=float, default=3.17, help="Gain [e-/ADU]")
args = parser.parse_args()

# ── Load FITS ──────────────────────────────────────────────────────────────────
print(f"Loading FITS: {args.fits}")
with fits.open(args.fits) as hdul:
    data = hdul[0].data.astype(np.float64)
    header = hdul[0].header

print(f"  Image: {data.shape[1]}×{data.shape[0]} px")

# ── Load VYVAR CSV ─────────────────────────────────────────────────────────────
print(f"Loading CSV:  {args.csv}")
df = pd.read_csv(args.csv, dtype={"catalog_id": str, "name": str})

# Filter: valid position + flux
df = df[df["x"].notna() & df["y"].notna() & df["dao_flux"].notna()].copy()
df = df[df["dao_flux"] > 0].copy()
print(f"  Stars with valid flux: {len(df)}")

# ── Sky background (global sigma-clipped) ─────────────────────────────────────
_, sky_med, sky_std = sigma_clipped_stats(data, sigma=3.0, maxiters=5)
print(f"  Sky: median={sky_med:.1f} ADU/px  std={sky_std:.1f}")

# ── Photutils aperture photometry ─────────────────────────────────────────────
positions = list(zip(df["x"].values, df["y"].values))

# Per-star aperture radius
if args.aperture:
    r_arr = np.full(len(df), args.aperture)
else:
    if "aperture_r_px" in df.columns and df["aperture_r_px"].notna().any():
        r_arr = df["aperture_r_px"].fillna(4.0).values
    else:
        r_arr = np.full(len(df), 4.0)
        print("  aperture_r_px missing — using 4.0 px default")

# Run photometry per unique aperture size (group for efficiency)
flux_photutils = np.full(len(df), np.nan)
flux_err_photutils = np.full(len(df), np.nan)

for r_val in np.unique(np.round(r_arr, 2)):
    idx = np.where(np.round(r_arr, 2) == r_val)[0]
    pos_sub = [positions[i] for i in idx]

    ap = CircularAperture(pos_sub, r=r_val)
    ann = CircularAnnulus(
        pos_sub,
        r_in=args.annulus_in,
        r_out=args.annulus_out,
    )

    phot_table = aperture_photometry(data - sky_med, ap)
    flux_photutils[idx] = phot_table["aperture_sum"].value

    # Photometric error (CCD equation — Howell 1989)
    g = args.gain
    area = np.pi * r_val**2
    for j, i in enumerate(idx):
        F = max(flux_photutils[i], 0)
        sky_ann = sky_med
        variance = F / g + sky_ann / g * area + (sky_std / g) ** 2 * area
        flux_err_photutils[i] = np.sqrt(variance)

# ── Comparison ────────────────────────────────────────────────────────────────
df["flux_vyvar"] = df["dao_flux"].values
df["flux_photutils"] = flux_photutils
df["flux_err_putils"] = flux_err_photutils
df["ratio"] = df["flux_photutils"] / df["flux_vyvar"]
df["ratio_pct_diff"] = (df["ratio"] - 1.0) * 100.0

# Remove outliers for stats
valid = df["ratio"].notna() & np.isfinite(df["ratio"])
ratio_arr = df.loc[valid, "ratio"].values
_, ratio_med, ratio_std = sigma_clipped_stats(ratio_arr, sigma=3.0)

print("\n=== Cross-validation results ===")
print(f"  Stars compared:    {valid.sum()}")
print(f"  Flux ratio median: {ratio_med:.4f}  (1.0000 = perfect)")
print(f"  Flux ratio std:    {ratio_std:.4f}  ({ratio_std * 100:.2f}%)")
print(f"  Outliers (>10%):   {(np.abs(ratio_arr - 1.0) > 0.10).sum()}")

# Per mag-bin
if "mag" in df.columns or "phot_g_mean_mag" in df.columns:
    mag_col = "mag" if "mag" in df.columns else "phot_g_mean_mag"
    df["mag_bin"] = (df[mag_col].fillna(99) // 1).astype(int)
    print("\n  Per magnitude bin:")
    print(f"  {'Mag bin':>8}  {'N':>5}  {'Median ratio':>13}  {'Std':>8}")
    for b, grp in df[valid].groupby("mag_bin"):
        if len(grp) < 3:
            continue
        _, m, s = sigma_clipped_stats(grp["ratio"].values, sigma=3.0)
        print(f"  {b:>8}–{b + 1:<3}  {len(grp):>5}  {m:>13.4f}  {s:>8.4f}")

# ── Save results ───────────────────────────────────────────────────────────────
out_cols = [
    "catalog_id",
    "x",
    "y",
    "flux_vyvar",
    "flux_photutils",
    "flux_err_putils",
    "ratio",
    "ratio_pct_diff",
]
if "mag" in df.columns:
    out_cols.insert(3, "mag")

df[out_cols].to_csv(args.output, index=False)
print(f"\nSaved: {args.output}")
print("Done.")
