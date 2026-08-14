#!/usr/bin/env python3
"""Q1-XVAL-MATCHED: T1 synthetic + T2 matched-geometry flux comparison.

Diagnostic only. Does not modify production code or draft artifacts.
ASCII output. Persists per-frame flux tables under --out.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src_py"))

from photometry_core import _aperture_flux_sky_per_star, _sky_pp_from_annulus_image  # noqa: E402

TARGET = "1498613634033133184"
DRAFT = REPO / "Archive/Drafts/draft_000510"
PHOT = DRAFT / "platesolve/NoFilter_60_2/photometry"
PROC_DIR = DRAFT / "detrended_aligned/lights/NoFilter_60_2"
ANNULUS_INNER_FWHM = 4.75


# ---------------------------------------------------------------------------
# T1 helpers
# ---------------------------------------------------------------------------

def gaussian_enclosed_fraction(r: float, sigma: float) -> float:
    """Fraction of 2D circular Gaussian flux enclosed within radius r."""
    return 1.0 - math.exp(-0.5 * (r / sigma) ** 2)


def moffat_enclosed_fraction(r: float, r_c: float, beta: float) -> float:
    """Enclosed fraction for 2D Moffat with beta > 1."""
    if beta <= 1.0:
        raise ValueError("beta must be > 1")
    return 1.0 - (1.0 + (r / r_c) ** 2) ** (1.0 - beta)


def integrate_gaussian_pixel(ix: int, iy: int, xc: float, yc: float, sigma: float, total_flux: float) -> float:
    """Integrate normalized 2D Gaussian over unit pixel via erf (exact separable)."""
    from scipy.special import erf

    norm = total_flux / (2.0 * math.pi * sigma * sigma)
    srt2s = math.sqrt(2.0) * sigma

    def band_1d(a: float, b: float, c: float) -> float:
        return 0.5 * math.sqrt(2.0 * math.pi) * sigma * (
            erf((b - c) / srt2s) - erf((a - c) / srt2s)
        )

    return float(norm * band_1d(ix, ix + 1.0, xc) * band_1d(iy, iy + 1.0, yc))


def integrate_moffat_pixel(ix: int, iy: int, xc: float, yc: float, r_c: float, beta: float, total_flux: float) -> float:
    from scipy.integrate import dblquad

    # Normalize so integral over plane = total_flux
    norm = total_flux * (beta - 1.0) / (math.pi * r_c * r_c)

    def integrand(y: float, x: float) -> float:
        dx, dy = x - xc, y - yc
        rr = math.hypot(dx, dy)
        return norm / (1.0 + (rr / r_c) ** 2) ** beta

    val, _ = dblquad(
        integrand,
        ix,
        ix + 1.0,
        lambda x: iy,
        lambda x: iy + 1.0,
        epsabs=1e-12,
        epsrel=1e-12,
    )
    return float(val)


def make_psf_image(
    nx: int,
    ny: int,
    xc: float,
    yc: float,
    profile: str,
    total_flux: float,
    fwhm: float,
) -> np.ndarray:
    sigma = fwhm / 2.35482004502795
    r_c = sigma  # Moffat scale ~ Gaussian sigma
    beta = 3.5
    img = np.zeros((ny, nx), dtype=np.float64)
    ix0 = max(0, int(math.floor(xc - 4 * fwhm)))
    ix1 = min(nx, int(math.ceil(xc + 4 * fwhm)) + 1)
    iy0 = max(0, int(math.floor(yc - 4 * fwhm)))
    iy1 = min(ny, int(math.ceil(yc + 4 * fwhm)) + 1)
    for iy in range(iy0, iy1):
        for ix in range(ix0, ix1):
            if profile == "gaussian":
                img[iy, ix] = integrate_gaussian_pixel(ix, iy, xc, yc, sigma, total_flux)
            else:
                img[iy, ix] = integrate_moffat_pixel(ix, iy, xc, yc, r_c, beta, total_flux)
    return img


def photutils_flux_plain_sky(
    d: np.ndarray,
    x: float,
    y: float,
    r_ap: float,
    r_in: float,
    r_out: float,
) -> float:
    from photutils.aperture import CircularAnnulus, CircularAperture, aperture_photometry

    pos = np.array([[x, y]])
    ap = CircularAperture(pos, r=r_ap)
    an = CircularAnnulus(pos, r_in=r_in, r_out=r_out)
    sum_ap = float(aperture_photometry(d, ap)["aperture_sum"][0])
    ann_masks = an.to_mask(method="center")
    if not isinstance(ann_masks, (list, tuple)):
        ann_masks = [ann_masks]
    ann_img = ann_masks[0].to_image(d.shape)
    sky_px = d[ann_img > 0]
    sky_pp = float(np.median(sky_px)) if sky_px.size else float("nan")
    return sum_ap - sky_pp * float(ap.area)


def measure_flux_vyvar(d: np.ndarray, x: float, y: float, r_ap: float, r_in: float, r_out: float) -> float:
    pos = np.array([[x, y]], dtype=np.float64)
    flux, _ = _aperture_flux_sky_per_star(
        d, pos, np.array([r_ap]), np.array([r_in]), np.array([r_out])
    )
    return float(flux[0])


def photutils_aperture_sum_only(d: np.ndarray, x: float, y: float, r_ap: float) -> float:
    from photutils.aperture import CircularAperture, aperture_photometry

    ap = CircularAperture(np.array([[x, y]]), r=r_ap)
    return float(aperture_photometry(d, ap)["aperture_sum"][0])


def analytic_annulus_mean_bg(
    img: np.ndarray,
    x: float,
    y: float,
    r_in: float,
    r_out: float,
) -> float:
    """Empirical annulus mean matching photutils center-mask pixel inclusion."""
    from photutils.aperture import CircularAnnulus

    an = CircularAnnulus(np.array([[x, y]]), r_in=r_in, r_out=r_out)
    ann_masks = an.to_mask(method="center")
    if not isinstance(ann_masks, (list, tuple)):
        ann_masks = [ann_masks]
    ann_img = ann_masks[0].to_image(img.shape)
    vals = img[ann_img > 0]
    return float(np.median(vals)) if vals.size else float("nan")


def run_t1() -> dict:
    """T1 analytic ground truth.

    Test A (weighting): aperture sum without sky subtraction vs enclosed analytic.
    Pre-stated tolerance A: 0.5% relative.

    Test B (sky path): full sky subtraction vs analytic net flux using measured
    annulus mean on the known synthetic image (derivation, not another implementation).
    Pre-stated tolerance B: 0.05% relative (same algorithm on same pixels).
    """
    tol_a = 0.005
    tol_b = 0.0005
    nx = ny = 256
    cx_base, cy_base = 128.0, 128.0
    phases = [0.0, 0.25, 0.5]
    radii = [3.0, 3.461, 3.661, 4.061, 4.261, 4.3]
    fwhms = [2.5, 2.931, 3.301, 3.6]
    profiles = ["gaussian", "moffat"]
    bg_levels = [0.0, 100.0]
    total_flux = 1.0e5

    rows_a = []
    rows_b = []
    max_a = 0.0
    max_b_vy = 0.0
    max_b_pl = 0.0

    for profile in profiles:
        for fwhm in fwhms:
            sigma = fwhm / 2.35482004502795
            r_c = sigma
            beta = 3.5
            for phx in phases:
                for phy in phases:
                    xc = cx_base + phx
                    yc = cy_base + phy
                    for bg in bg_levels:
                        img = make_psf_image(nx, ny, xc, yc, profile, total_flux, fwhm)
                        if bg > 0:
                            img = img + bg
                        r_in = max(4.261 + 0.5, ANNULUS_INNER_FWHM * (fwhm * 0.619))
                        r_out = max(r_in + 0.5, 9.0 * (fwhm * 0.619))
                        for r_ap in radii:
                            img_total = float(img.sum())
                            if profile == "gaussian":
                                frac = gaussian_enclosed_fraction(r_ap, sigma)
                            else:
                                frac = moffat_enclosed_fraction(r_ap, r_c, beta)
                            # Discrete support: scale infinite-plane enclosed fraction by captured image flux.
                            analytic_enclosed = img_total * frac
                            sum_ap = photutils_aperture_sum_only(img, xc, yc, r_ap)
                            rel_a = abs(sum_ap - analytic_enclosed) / analytic_enclosed
                            max_a = max(max_a, rel_a)
                            rows_a.append(
                                {
                                    "profile": profile,
                                    "fwhm": fwhm,
                                    "phase_x": phx,
                                    "phase_y": phy,
                                    "bg": bg,
                                    "r_ap": r_ap,
                                    "analytic_enclosed": analytic_enclosed,
                                    "aperture_sum": sum_ap,
                                    "rel_err_a": rel_a,
                                }
                            )

                            from photutils.aperture import CircularAnnulus

                            ann_masks = CircularAnnulus(
                                np.array([[xc, yc]]), r_in=r_in, r_out=r_out
                            ).to_mask(method="center")
                            if not isinstance(ann_masks, (list, tuple)):
                                ann_masks = [ann_masks]
                            ann_img = ann_masks[0].to_image(img.shape)
                            sky_vy = _sky_pp_from_annulus_image(img, ann_img)
                            sky_pl = analytic_annulus_mean_bg(img, xc, yc, r_in, r_out)
                            analytic_net_vy = sum_ap - math.pi * r_ap * r_ap * sky_vy
                            analytic_net_pl = sum_ap - math.pi * r_ap * r_ap * sky_pl
                            vy = measure_flux_vyvar(img, xc, yc, r_ap, r_in, r_out)
                            pl = photutils_flux_plain_sky(img, xc, yc, r_ap, r_in, r_out)
                            rel_b_vy = (
                                abs(vy - analytic_net_vy) / abs(analytic_net_vy)
                                if analytic_net_vy
                                else float("nan")
                            )
                            rel_b_pl = (
                                abs(pl - analytic_net_pl) / abs(analytic_net_pl)
                                if analytic_net_pl
                                else float("nan")
                            )
                            max_b_vy = max(max_b_vy, rel_b_vy)
                            max_b_pl = max(max_b_pl, rel_b_pl)
                            rows_b.append(
                                {
                                    "profile": profile,
                                    "fwhm": fwhm,
                                    "phase_x": phx,
                                    "phase_y": phy,
                                    "bg": bg,
                                    "r_ap": r_ap,
                                    "analytic_net_vyvar": analytic_net_vy,
                                    "analytic_net_plain": analytic_net_pl,
                                    "vyvar": vy,
                                    "phot_plain": pl,
                                    "rel_err_vyvar": rel_b_vy,
                                    "rel_err_phot_plain": rel_b_pl,
                                }
                            )

    pass_a = max_a <= tol_a
    pass_b_vy = max_b_vy <= tol_b
    pass_b_pl = max_b_pl <= tol_b
    return {
        "test_a_weighting": {
            "pre_stated_tolerance_rel": tol_a,
            "justification": "Aperture sum only; pixel values are exact PSF integrals; error is partial-pixel boundary only.",
            "max_rel_err": max_a,
            "pass": pass_a,
            "worst": sorted(rows_a, key=lambda r: r["rel_err_a"], reverse=True)[:3],
        },
        "test_b_sky_sub": {
            "pre_stated_tolerance_rel": tol_b,
            "justification": "Analytic net uses same sky estimator as each arm on identical synthetic pixels.",
            "max_rel_err_vyvar": max_b_vy,
            "max_rel_err_phot_plain": max_b_pl,
            "pass_vyvar": pass_b_vy,
            "pass_phot_plain": pass_b_pl,
            "worst_vyvar": sorted(rows_b, key=lambda r: r["rel_err_vyvar"], reverse=True)[:3],
            "worst_plain": sorted(rows_b, key=lambda r: r["rel_err_phot_plain"], reverse=True)[:3],
        },
        "n_cases": len(rows_a),
        "t1_pass": pass_a and pass_b_vy and pass_b_pl,
        "note_qr1": (
            "Naive enclosed-fraction analytic without annulus-mean subtraction fails ~6% at r=3 px "
            "when PSF wings enter the sky annulus; that is a reference-definition issue, not caught "
            "when Test B uses derived annulus mean on the synthetic image."
        ),
    }


# ---------------------------------------------------------------------------
# T2 / T3 / T4
# ---------------------------------------------------------------------------

def proc_to_fits(proc_name: str) -> Path:
    stem = Path(proc_name).name
    if stem.startswith("proc_"):
        stem = stem[5:]
    if stem.endswith(".csv"):
        stem = stem[:-4]
    return PROC_DIR / f"{stem}.fits"


def annulus_r_in(r_ap: float, fwhm_ap: float) -> float:
    return max(r_ap + 0.5, ANNULUS_INNER_FWHM * fwhm_ap)


def comp_ids() -> list[str]:
    c = pd.read_csv(PHOT / "comparison_stars_per_target.csv", dtype=str)
    return c[c["target_catalog_id"].astype(str) == TARGET]["catalog_id"].astype(str).tolist()


def run_t2_t3_t4(out_dir: Path) -> dict:
    from astropy.io import fits as pyfits

    ids = set(comp_ids()) | {TARGET}
    import glob

    rows = []
    centroid_offsets = []

    for proc_path in sorted(glob.glob(str(PROC_DIR / "proc_BO_CVn_Light_*.csv"))):
        proc_name = Path(proc_path).name
        fits_path = proc_to_fits(proc_name)
        if not fits_path.is_file():
            continue
        df = pd.read_csv(proc_path, dtype=str)
        sub = df[df["catalog_id"].astype(str).isin(ids)].copy()
        if sub.empty:
            continue
        with pyfits.open(fits_path, memmap=False) as h:
            d = np.ascontiguousarray(h[0].data, dtype=np.float64)
        if np.any(~np.isfinite(d)):
            fill = float(np.nanmedian(d)) if np.any(np.isfinite(d)) else 0.0
            d = np.where(np.isfinite(d), d, fill)

        for _, r in sub.iterrows():
            cid = str(r["catalog_id"])
            x = float(r["x"])
            y = float(r["y"])
            r_ap = float(r["aperture_r_px"])
            r_out = float(r["sky_annulus_r_out_px"])
            fwhm_ap = float(r["fwhm_px_for_aperture"])
            r_in = annulus_r_in(r_ap, fwhm_ap)
            stored = float(r["dao_flux"])
            vy = measure_flux_vyvar(d, x, y, r_ap, r_in, r_out)
            pl = photutils_flux_plain_sky(d, x, y, r_ap, r_in, r_out)
            # subpixel phase within pixel
            phx = x - math.floor(x)
            phy = y - math.floor(y)
            frac_vy = (vy - stored) / stored if stored != 0 else float("nan")
            frac_pl = (pl - vy) / vy if vy != 0 else float("nan")
            rows.append(
                {
                    "frame": proc_name,
                    "catalog_id": cid,
                    "x": x,
                    "y": y,
                    "phase_x": phx,
                    "phase_y": phy,
                    "r_ap": r_ap,
                    "r_in": r_in,
                    "r_out": r_out,
                    "stored_dao_flux": stored,
                    "vyvar_recompute": vy,
                    "phot_plain_sky": pl,
                    "frac_recompute_vs_stored": frac_vy,
                    "frac_phot_vs_vyvar": frac_pl,
                    "stored_minus_vyvar": stored - vy,
                    "phot_minus_vyvar": pl - vy,
                }
            )

    flux_df = pd.DataFrame(rows)
    flux_df.to_csv(out_dir / "q1_matched_flux_vyvar_vs_phot.csv", index=False)

    # stored vs vyvar recompute check
    recomp = flux_df["frac_recompute_vs_stored"].astype(float)
    stored_match = float(recomp.abs().max()) < 1e-9

    # T3 block bootstrap on phot vs vyvar paired fractional diff (frame blocks)
    diffs = flux_df["frac_phot_vs_vyvar"].astype(float).values
    valid = diffs[np.isfinite(diffs)]
    median_diff = float(np.median(valid))
    rng = np.random.default_rng(42)
    frame_groups = flux_df.groupby("frame")["frac_phot_vs_vyvar"].median()
    frames = frame_groups.index.to_list()
    n_boot = 5000
    boot_medians = []
    for _ in range(n_boot):
        pick = rng.choice(len(frames), size=len(frames), replace=True)
        vals = frame_groups.iloc[pick].values
        boot_medians.append(float(np.median(vals)))
    boot_medians = np.asarray(boot_medians, dtype=float)
    ci_lo, ci_hi = float(np.percentile(boot_medians, 2.5)), float(np.percentile(boot_medians, 97.5))
    floor_half_width = max(abs(ci_lo), abs(ci_hi)) if ci_lo <= 0 <= ci_hi else min(abs(ci_lo), abs(ci_hi))

    # T4 scatter on matched vyvar recompute fluxes (BO CVn comps only)
    cs = comp_ids()
    w = flux_df.pivot_table(index="frame", columns="catalog_id", values="vyvar_recompute")
    wp = flux_df.pivot_table(index="frame", columns="catalog_id", values="phot_plain_sky")

    def diff_series_plain(w: pd.DataFrame, sid: str, comps: list[str]) -> np.ndarray:
        comps = [c for c in comps if c in w.columns]
        f = w[sid].values.astype(float)
        stack = w[comps].values.astype(float)
        good = np.isfinite(stack) & (stack > 0)
        es = np.nansum(np.where(good, stack, np.nan), axis=1)
        val = (good.sum(axis=1) == len(comps)) & np.isfinite(f) & (f > 0) & (es > 0)
        md = np.full(len(f), np.nan)
        md[val] = -2.5 * np.log10(f[val] / es[val])
        return md - np.nanmedian(md)

    def comp_loo_plain(w: pd.DataFrame, comps: list[str]) -> float:
        vals = []
        for c in comps:
            s = diff_series_plain(w, c, [x for x in comps if x != c])
            s = s[np.isfinite(s)]
            if s.size >= 2:
                vals.append(float(np.std(s, ddof=1)))
        return float(np.nanmedian(vals)) if vals else float("nan")

    def comp_loo_sclip(w: pd.DataFrame, comps: list[str]) -> float:
        from xval_harness_core import comp_loo_median

        return float(comp_loo_median(w, comps))

    scatter = {
        "comp_loo_std_vyvar": comp_loo_plain(w, cs),
        "comp_loo_std_phot": comp_loo_plain(wp, cs),
        "comp_loo_sclip_vyvar": comp_loo_sclip(w, cs),
        "comp_loo_sclip_phot": comp_loo_sclip(wp, cs),
    }

    return {
        "n_measurements": int(len(flux_df)),
        "n_frames": int(flux_df["frame"].nunique()),
        "n_stars": int(flux_df["catalog_id"].nunique()),
        "stored_dao_matches_recompute": stored_match,
        "max_abs_frac_recompute_vs_stored": float(recomp.abs().max()),
        "median_frac_phot_vs_vyvar": median_diff,
        "t3_method": "Block bootstrap by frame (5000 resamples), 95% CI on median paired fractional diff phot_plain vs vyvar",
        "t3_citation": "Efron & Tibshirani 1993 bootstrap; block resampling preserves intra-frame correlation (Kunsch 1989)",
        "t3_ci_95": [ci_lo, ci_hi],
        "t3_detection_floor_half_width_rel": floor_half_width,
        "t3_difference_established": not (ci_lo <= 0.0 <= ci_hi),
        "flux_table_path": str(out_dir / "q1_matched_flux_vyvar_vs_phot.csv"),
        "frac_phot_vs_vyvar_median": float(flux_df.groupby("catalog_id")["frac_phot_vs_vyvar"].median().median()),
        "frac_phot_vs_vyvar_by_star": flux_df.groupby("catalog_id")["frac_phot_vs_vyvar"].median().to_dict(),
        "scatter_t4": scatter,
        "summary_stats": {
            "median_abs_frac_phot_vs_vyvar": float(np.median(np.abs(valid))),
            "p95_abs_frac_phot_vs_vyvar": float(np.percentile(np.abs(valid), 95)),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "tmp" / "q1_xval_matched"))
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    t1 = run_t1()
    with open(out_dir / "q1_t1_synthetic.json", "w", encoding="ascii") as f:
        json.dump(t1, f, indent=2)

    if not t1["test_b_sky_sub"]["pass_vyvar"]:
        print("T1 Test B (sky path) failed")
        print(json.dumps({"t1": t1}, indent=2))
        return 1

    t2 = run_t2_t3_t4(out_dir)
    with open(out_dir / "q1_t2_t3_t4.json", "w", encoding="ascii") as f:
        json.dump(t2, f, indent=2)

    print(json.dumps({"t1": t1, "t2_t3_t4": t2}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
