#!/usr/bin/env python3
"""Part 1: re-measure PSF elongation with cutout scaling (read-only).

Three methods on the same bright isolated stars per draft:
  (i)   fixed 9x9 Gaussian fit (baseline)
  (ii)  Gaussian fit, half-width = round(3 * FWHM_px)
  (iii) model-free weighted second moments in the FWHM-scaled window
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import AppConfig  # noqa: E402
from database import VyvarDatabase  # noqa: E402
from psf_photometry import _read_plate_scale_arcsec_px_from_fits, get_epsf_fwhm_from_context  # noqa: E402

diag_path = _ROOT / "scripts" / "diagnose_psf_elongation_362.py"
_spec = importlib.util.spec_from_file_location("diag", diag_path)
diag = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(diag)

CASES = [
    (362, "NoFilter_60_2"),
    (364, "Luminance_180_2"),
]


def _odd_size(half: int) -> int:
    h = int(max(1, half))
    size = 2 * h + 1
    return size if size % 2 == 1 else size + 1


def _extract_cutout(data: np.ndarray, x0: int, y0: int, half: int) -> np.ndarray | None:
    h, w = data.shape
    y1, y2 = max(0, y0 - half), min(h, y0 + half + 1)
    x1, x2 = max(0, x0 - half), min(w, x0 + half + 1)
    cut = data[y1:y2, x1:x2]
    if cut.size < 9:
        return None
    return cut


def _second_moments_elongation(cutout: np.ndarray) -> dict[str, Any]:
    z = diag._background_subtract_cutout(cutout)
    z = np.maximum(z, 0.0)
    total = float(z.sum())
    if not math.isfinite(total) or total <= 0:
        return {"ok": False, "reason": "nonpositive_flux"}
    h, w = z.shape
    yy, xx = np.mgrid[:h, :w]
    cx = float((z * xx).sum() / total)
    cy = float((z * yy).sum() / total)
    dx = xx - cx
    dy = yy - cy
    q11 = float((z * dx * dx).sum() / total)
    q22 = float((z * dy * dy).sum() / total)
    q12 = float((z * dx * dy).sum() / total)
    trace = q11 + q22
    det = q11 * q22 - q12 * q12
    disc = max(0.0, 0.25 * trace * trace - det)
    sqrt_disc = math.sqrt(disc)
    lam_major = 0.5 * trace + sqrt_disc
    lam_minor = 0.5 * trace - sqrt_disc
    if lam_minor <= 0 or not math.isfinite(lam_major) or not math.isfinite(lam_minor):
        return {"ok": False, "reason": "bad_eigenvalues"}
    elong = math.sqrt(lam_major / lam_minor)
    return {"ok": True, "elongation": float(elong)}


def run_draft(draft_id: int, setup: str) -> dict[str, Any]:
    cfg = AppConfig()
    db = VyvarDatabase(cfg.database_path)
    draft = Path(cfg.archive_root) / "Drafts" / f"draft_{draft_id:06d}"
    aligned = draft / "detrended_aligned" / "lights" / setup
    ps = draft / "platesolve" / setup
    ms = ps / "MASTERSTAR.fits"

    fwhm_px = float(get_epsf_fwhm_from_context(ms, db, draft_id))
    plate_scale = float(_read_plate_scale_arcsec_px_from_fits(ms))
    half_scaled = int(round(3.0 * fwhm_px))
    size_scaled = _odd_size(half_scaled)
    half_9 = 4
    size_9 = 9
    select_half = max(half_9, half_scaled)
    fit_shape = (_odd_size(select_half), _odd_size(select_half))

    from param_resolver import resolve_gain, resolve_read_noise  # noqa: E402

    with fits.open(ms, memmap=True) as hd:
        mhdr = hd[0].header
    gain = float(resolve_gain(mhdr, db=db, equipment_id=None, cfg=cfg).value or 1.0)
    rn = float(resolve_read_noise(mhdr, db=db, equipment_id=None, cfg=cfg).value or 10.0)

    pairs = [
        (f, aligned / f.name.replace(".fits", ".csv"))
        for f in sorted(aligned.glob("proc_*.fits"))
        if (aligned / f.name.replace(".fits", ".csv")).is_file()
    ]

    elong_i: list[float] = []
    elong_ii: list[float] = []
    elong_iii: list[float] = []

    for fits_path, csv_path in pairs:
        frame_df = pd.read_csv(csv_path, low_memory=False, dtype={"catalog_id": str})
        with fits.open(fits_path, memmap=True) as hd:
            data = np.asarray(hd[0].data, dtype=np.float32)
        img_shape = data.shape
        picked = diag._select_frame_stars_from_proc(
            frame_df,
            ps,
            fwhm_px=fwhm_px,
            plate_scale_arcsec_px=plate_scale,
            fit_shape=fit_shape,
            gain=gain,
            rn=rn,
            img_shape=img_shape,
        )
        for _, star in picked.iterrows():
            x0 = int(round(float(star["x"])))
            y0 = int(round(float(star["y"])))
            cut9 = _extract_cutout(data, x0, y0, half_9)
            cut_sc = _extract_cutout(data, x0, y0, half_scaled)
            if cut9 is None or cut_sc is None:
                continue
            fit9 = diag._fit_elliptical_gaussian(cut9)
            if fit9.get("ok"):
                elong_i.append(float(fit9["elongation"]))
            fit_sc = diag._fit_elliptical_gaussian(cut_sc)
            if fit_sc.get("ok"):
                elong_ii.append(float(fit_sc["elongation"]))
            mom = _second_moments_elongation(cut_sc)
            if mom.get("ok"):
                elong_iii.append(float(mom["elongation"]))

    med_i = float(np.median(elong_i)) if elong_i else float("nan")
    med_ii = float(np.median(elong_ii)) if elong_ii else float("nan")
    med_iii = float(np.median(elong_iii)) if elong_iii else float("nan")

    return {
        "draft_id": draft_id,
        "setup": setup,
        "fwhm_px": fwhm_px,
        "half_scaled_px": half_scaled,
        "cutout_scaled_px": size_scaled,
        "n_stars_i": len(elong_i),
        "n_stars_ii": len(elong_ii),
        "n_stars_iii": len(elong_iii),
        "median_i_9x9": med_i,
        "median_ii_scaled_gauss": med_ii,
        "median_iii_scaled_moments": med_iii,
    }


def _verdict(med_i: float, med_ii: float, med_iii: float) -> str:
    if not all(math.isfinite(x) for x in (med_i, med_ii, med_iii)):
        return "inconclusive (insufficient fits)"
    if med_i >= 1.12 and med_ii <= 1.06 and med_iii <= 1.06:
        return "cutout/fitter artifact — stars essentially round under scaled windows"
    if med_i >= 1.12 and med_ii >= 1.12 and med_iii >= 1.12:
        return "real on-sky ellipticity (persists under scaled Gaussian and moments)"
    if abs(med_ii - med_i) < 0.03 and abs(med_iii - med_i) < 0.03:
        return "elongation stable across methods — likely real"
    return "mixed — review per-method medians"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rows = []
    print("Part 1 — elongation re-measure (same star pool, three methods)\n")
    print(f"{'draft':>5} | {'FWHM':>5} | {'cutout(ii)':>10} | {'N':>5} | {'(i)9x9':>7} | {'(ii)Gauss':>9} | {'(iii)Mom':>8} | verdict")
    print("-" * 95)
    for draft_id, setup in CASES:
        r = run_draft(draft_id, setup)
        v = _verdict(r["median_i_9x9"], r["median_ii_scaled_gauss"], r["median_iii_scaled_moments"])
        r["verdict"] = v
        rows.append(r)
        print(
            f"{draft_id:5d} | {r['fwhm_px']:5.2f} | {r['cutout_scaled_px']:4d}px half={r['half_scaled_px']:2d} | "
            f"{r['n_stars_i']:5d} | {r['median_i_9x9']:7.4f} | {r['median_ii_scaled_gauss']:9.4f} | "
            f"{r['median_iii_scaled_moments']:8.4f} | {v}"
        )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
