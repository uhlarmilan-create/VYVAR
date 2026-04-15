"""Smoke test Fázy 2A — draft_000195 (nový dataset pre verifikáciu opráv)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Konfigurácia
# ---------------------------------------------------------------------------

DRAFT_DIR = Path(r"C:\ASTRO\python\VYVAR\Archive\Drafts\draft_000195")
PS_DIR = DRAFT_DIR / "platesolve"
OBS_GROUP = "NoFilter_120_2"
MASTERSTAR_PATH = PS_DIR / "MASTERSTAR.fits"
ALIGNED_DIR = DRAFT_DIR / "detrended_aligned" / "lights" / OBS_GROUP
OBS_DIR = PS_DIR / OBS_GROUP
PHOTOMETRY_DIR = OBS_DIR / "photometry"
ACTIVE_TARGETS_CSV = PHOTOMETRY_DIR / "active_targets.csv"
COMP_STARS_CSV = PHOTOMETRY_DIR / "comparison_stars_per_target.csv"
OUTPUT_DIR = PHOTOMETRY_DIR

sys.path.insert(0, str(Path(r"C:\ASTRO\python\VYVAR")))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
    force=True,
)
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")


def _p(s: str = "") -> None:
    print(s, flush=True)


# ---------------------------------------------------------------------------
# Vstupné súbory
# ---------------------------------------------------------------------------

_p("=== draft_000195 — vstupné súbory ===")
for label, path in [
    ("MASTERSTAR.fits", MASTERSTAR_PATH),
    ("active_targets.csv", ACTIVE_TARGETS_CSV),
    ("comparison_stars_per_target.csv", COMP_STARS_CSV),
    ("ALIGNED_DIR", ALIGNED_DIR),
]:
    status = "OK" if Path(path).exists() else "CHYBA"
    _p(f"  [{status}] {label}: {path}")

# ---------------------------------------------------------------------------
# FWHM diagnostika
# ---------------------------------------------------------------------------

_p()
_p("=== FWHM diagnostika ===")
from astropy.io import fits as _fits
from photometry_phase2a import measure_fwhm_from_masterstar

with _fits.open(MASTERSTAR_PATH, memmap=False) as _hdul:
    _vy_fwhm = _hdul[0].header.get("VY_FWHM", None)

if _vy_fwhm is not None and float(_vy_fwhm) > 0:
    _vy_fwhm = float(_vy_fwhm)
    _p(f"  VY_FWHM (DAO z FITS hlavičky): {_vy_fwhm:.3f} px")
else:
    _vy_fwhm = 3.5
    _p(f"  VY_FWHM: nenájdené, fallback hint={_vy_fwhm:.3f} px")

_at = pd.read_csv(ACTIVE_TARGETS_CSV)
_comp = pd.read_csv(COMP_STARS_CSV)
_stars_for_fwhm = pd.concat(
    [
        _at[["x", "y"] + (["mag"] if "mag" in _at.columns else [])],
        _comp[["x", "y"] + (["mag"] if "mag" in _comp.columns else [])],
    ],
    ignore_index=True,
).drop_duplicates(subset=["x", "y"])

_gauss_fwhm = measure_fwhm_from_masterstar(
    MASTERSTAR_PATH,
    _stars_for_fwhm,
    dao_fwhm_hint=_vy_fwhm,
)
_p(f"  Gaussian FWHM (fit na MASTERSTAR): {_gauss_fwhm:.3f} px")
_p(f"  Pomer VY_FWHM / Gaussian FWHM: {_vy_fwhm/_gauss_fwhm:.3f}×")

# ---------------------------------------------------------------------------
# Spustenie Fázy 2A
# ---------------------------------------------------------------------------

_p()
_p("=== Spúšťam Fázu 2A ===")
from photometry_phase2a import run_phase2a

result = run_phase2a(
    masterstar_fits_path=MASTERSTAR_PATH,
    active_targets_csv=ACTIVE_TARGETS_CSV,
    comparison_stars_csv=COMP_STARS_CSV,
    per_frame_csv_dir=ALIGNED_DIR,
    detrended_aligned_dir=ALIGNED_DIR,
    output_dir=OUTPUT_DIR,
    fwhm_px=_vy_fwhm,
)

_p(f"  n_targets:     {result['n_targets']}")
_p(f"  n_frames:      {result['n_frames']}")
_p(f"  n_lightcurves: {result['n_lightcurves']}")

# ---------------------------------------------------------------------------
# Výsledky — summary
# ---------------------------------------------------------------------------

_p()
_p("=== Photometry summary ===")
summary = pd.read_csv(OUTPUT_DIR / "photometry_summary.csv")
cols = ["vsx_name", "n_frames", "n_good_comp", "lc_rms", "aperture_px", "am_slope", "am_detrended"]
cols = [c for c in cols if c in summary.columns]
_p(summary[cols].to_string(index=False))

# ---------------------------------------------------------------------------
# Detaily — prvých 5 targetov s najlepším RMS
# ---------------------------------------------------------------------------

_p()
_p("=== Top 5 targetov (najlepší lc_rms) ===")
top5 = summary.nsmallest(5, "lc_rms") if "lc_rms" in summary.columns else summary.head(5)
for _, trow in top5.iterrows():
    cid = str(trow.get("catalog_id", ""))
    name = str(trow.get("vsx_name", cid))
    lc_path = OUTPUT_DIR / "lightcurves" / f"lightcurve_{cid}.csv"
    if not lc_path.exists():
        _p(f"  {name}: CSV nenájdené")
        continue
    lc = pd.read_csv(lc_path)
    normal = lc[lc["flag"] == "normal"]["mag_calib"].dropna()
    if len(normal) < 3:
        _p(f"  {name}: nedostatok normal bodov")
        continue
    p2p = np.sqrt(np.mean(np.diff(normal.values) ** 2)) / np.sqrt(2)
    ap = lc["aperture_r_px"].iloc[0] if "aperture_r_px" in lc.columns else float("nan")
    flags = lc["flag"].value_counts().to_dict()
    _p(f"  {name}: p2p={p2p:.5f} mag | ap={ap:.2f}px | flags={flags}")

# ---------------------------------------------------------------------------
# Štatistiky
# ---------------------------------------------------------------------------

_p()
_p("=== Štatistiky ===")
rms = summary["lc_rms"].dropna()
_p(f"  lc_rms median:  {rms.median():.4f} mag")
_p(f"  lc_rms < 0.05:  {(rms < 0.05).sum()} targetov")
_p(f"  lc_rms < 0.10:  {(rms < 0.10).sum()} targetov")
_p(f"  gaussian_fwhm:  {_gauss_fwhm:.3f} px")
_p(f"  vy_fwhm:        {_vy_fwhm:.3f} px")

if "aperture_px" in summary.columns:
    _p()
    _p("  Apertura distribúcia:")
    for ap, grp in summary.groupby("aperture_px"):
        _p(f"    {ap:.3f}px: {len(grp)} targetov, RMS median={grp['lc_rms'].median():.4f}")
