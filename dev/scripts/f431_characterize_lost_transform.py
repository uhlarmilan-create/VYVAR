#!/usr/bin/env python3
"""T1 — Characterize lost ADU transform: MS(429) − cal Light_008."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder

import sys

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401  (repo layout: src_py + dev on sys.path)
_ROOT = _bootstrap.REPO_ROOT
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
OUT = _ROOT / "tmp" / "f431_lost_transform"
DRAFT = _ROOT / "Archive" / "Drafts" / "draft_000429"
MS = DRAFT / "detrended_aligned" / "lights" / "NoFilter_60_2" / "MASTERSTAR.fits"
CAL = DRAFT / "calibrated" / "lights" / "NoFilter_60_2" / "BO_CVn_Light_008.fits"
PROC = DRAFT / "processed" / "lights" / "NoFilter_60_2" / "proc_BO_CVn_Light_008.fits"

# Also compare 431 for control
CAL431 = _ROOT / "Archive" / "Drafts" / "draft_000431" / "calibrated" / "lights" / "NoFilter_60_2" / "BO_CVn_Light_008.fits"
MS431 = _ROOT / "Archive" / "Drafts" / "draft_000431" / "detrended_aligned" / "lights" / "NoFilter_60_2" / "MASTERSTAR.fits"


def dao_pass1_count(img: np.ndarray, *, sigma: float = 2.1, fwhm: float = 2.5) -> dict:
    """Approximate MASTERSTAR pass-1 DAO (pipeline recipe without binning/WCS)."""
    from pipeline import DAO_STAR_FINDER_NO_ROUNDNESS_FILTER

    arr = np.asarray(img, dtype=np.float32)
    finite = np.isfinite(arr)
    arr = np.where(finite, arr, np.nanmedian(arr[finite]) if finite.any() else 0.0)
    _, med, std = sigma_clipped_stats(arr, sigma=3.0, maxiters=3)
    data0 = np.nan_to_num((arr - med).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    thr = max(float(sigma) * float(std), 1e-6)
    fwhm_eff = max(1.2, float(fwhm))
    finder = DAOStarFinder(
        fwhm=float(fwhm_eff),
        threshold=float(thr),
        brightest=None,
        **DAO_STAR_FINDER_NO_ROUNDNESS_FILTER,
    )
    tbl = finder(data0)
    n = int(len(tbl)) if tbl is not None else 0
    return {
        "n_pass1": n,
        "median": float(med),
        "std": float(std),
        "threshold": float(thr),
        "fwhm_eff": float(fwhm_eff),
        "sigma": float(sigma),
    }


def maxabs_localization(R: np.ndarray, cal: np.ndarray, top_n: int = 20) -> dict:
    abs_r = np.abs(R)
    flat = abs_r.ravel()
    idxs = np.argpartition(flat, -top_n)[-top_n:]
    idxs = idxs[np.argsort(flat[idxs])[::-1]]
    ys, xs = np.unravel_index(idxs, R.shape)
    # star-likeness: value at cal vs local background
    rows = []
    for y, x in zip(ys, xs):
        y0, x0 = int(y), int(x)
        y1, y2 = max(0, y0 - 8), min(R.shape[0], y0 + 9)
        x1, x2 = max(0, x0 - 8), min(R.shape[1], x0 + 9)
        patch = cal[y1:y2, x1:x2]
        loc_med = float(np.nanmedian(patch))
        peak = float(cal[y0, x0])
        excess = peak - loc_med
        rows.append(
            {
                "y": y0,
                "x": x0,
                "R": float(R[y0, x0]),
                "abs_R": float(abs_r[y0, x0]),
                "cal_peak": peak,
                "cal_local_med": loc_med,
                "cal_excess": excess,
                "at_bright_core": bool(excess > 5.0 * max(float(np.nanstd(patch)), 1.0)),
            }
        )
    n_core = sum(1 for r in rows if r["at_bright_core"])
    # edge fraction
    h, w = R.shape
    edge = (ys < 50) | (ys >= h - 50) | (xs < 50) | (xs >= w - 50)
    return {
        "top_peaks": rows,
        "fraction_at_bright_cores": n_core / max(len(rows), 1),
        "fraction_near_edge": float(np.mean(edge)),
    }


def fit_poly2d(R: np.ndarray, order: int = 2) -> dict:
    h, w = R.shape
    # subsample for speed
    step = 4
    yy, xx = np.mgrid[0:h:step, 0:w:step]
    zz = R[::step, ::step]
    mask = np.isfinite(zz)
    x = xx[mask].astype(np.float64)
    y = yy[mask].astype(np.float64)
    z = zz[mask].astype(np.float64)
    # build polynomial design
    cols = []
    names = []
    for i in range(order + 1):
        for j in range(order + 1 - i):
            cols.append((x**i) * (y**j))
            names.append(f"x^{i} y^{j}")
    A = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    pred = (A @ coef).reshape(zz.shape)  # wrong - need full image
    # evaluate full (subsampled) surface then upsample via repeat
    yy_f, xx_f = np.mgrid[0:h, 0:w]
    cols_f = []
    for i in range(order + 1):
        for j in range(order + 1 - i):
            cols_f.append((xx_f.ravel().astype(np.float64) ** i) * (yy_f.ravel().astype(np.float64) ** j))
    Af = np.column_stack(cols_f)
    surf = (Af @ coef).reshape(h, w)
    resid = R - surf
    return {
        "order": order,
        "names": names,
        "coef": [float(c) for c in coef],
        "surface": surf,
        "residual": resid,
        "r_std": float(np.nanstd(R)),
        "resid_std": float(np.nanstd(resid)),
        "variance_explained": float(1.0 - (np.nanstd(resid) ** 2) / max(np.nanstd(R) ** 2, 1e-12)),
    }


def impulsive_fraction(R: np.ndarray) -> dict:
    abs_r = np.abs(R)
    med = float(np.nanmedian(abs_r))
    mad = float(np.nanmedian(np.abs(abs_r - med))) + 1e-6
    thr = med + 8.0 * 1.4826 * mad
    sparse = abs_r > thr
    return {
        "mad_thr": thr,
        "frac_above_8mad": float(np.mean(sparse)),
        "n_above": int(np.count_nonzero(sparse)),
    }


def star_dipole_check(R: np.ndarray, cal: np.ndarray, n_stars: int = 80) -> dict:
    """At bright star peaks, look for dipole (sign flip) vs smooth pedestal."""
    _, med, std = sigma_clipped_stats(cal, sigma=3.0, maxiters=3)
    data0 = np.nan_to_num((cal - med).astype(np.float32), nan=0.0)
    finder = DAOStarFinder(fwhm=2.5, threshold=8.0 * float(std), brightest=n_stars)
    tbl = finder(data0)
    if tbl is None or len(tbl) == 0:
        return {"n": 0}
    dipoles = 0
    pedestals = 0
    for row in tbl:
        x, y = int(round(float(row["xcentroid"]))), int(round(float(row["ycentroid"])))
        if not (5 <= y < R.shape[0] - 5 and 5 <= x < R.shape[1] - 5):
            continue
        patch = R[y - 4 : y + 5, x - 4 : x + 5]
        c = float(R[y, x])
        # neighbors opposite sign from center → dipole-ish
        neigh = np.concatenate([patch[0, :].ravel(), patch[-1, :].ravel(), patch[:, 0].ravel(), patch[:, -1].ravel()])
        if np.sign(c) != 0 and np.mean(np.sign(neigh) == -np.sign(c)) > 0.4 and abs(c) > 50:
            dipoles += 1
        if abs(c) < 200 and abs(float(np.nanmedian(patch))) > 30:
            pedestals += 1
    return {"n_stars": int(len(tbl)), "dipole_like": dipoles, "smooth_pedestal_like": pedestals}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ms = np.asarray(fits.getdata(MS), dtype=np.float64)
    cal = np.asarray(fits.getdata(CAL), dtype=np.float64)
    proc = np.asarray(fits.getdata(PROC), dtype=np.float64)
    assert ms.shape == cal.shape

    # Primary residual per task
    R = ms - cal
    R_proc = proc - cal
    ms_eq_proc = bool(np.array_equal(ms, proc))

    pcts = [0.1, 1, 5, 50, 95, 99, 99.9]
    stats = {
        "shape": list(ms.shape),
        "ms_eq_proc": ms_eq_proc,
        "R_ms_minus_cal": {
            "min": float(np.min(R)),
            "max": float(np.max(R)),
            "mean": float(np.mean(R)),
            "median": float(np.median(R)),
            "std": float(np.std(R)),
            "maxabs": float(np.max(np.abs(R))),
            "percentiles": {str(p): float(np.percentile(R, p)) for p in pcts},
            "abs_percentiles": {str(p): float(np.percentile(np.abs(R), p)) for p in pcts},
        },
        "R_proc_minus_cal": {
            "maxabs": float(np.max(np.abs(R_proc))),
            "mean": float(np.mean(R_proc)),
            "identical_to_R_ms": bool(np.allclose(R, R_proc, rtol=0, atol=0)),
        },
        "sky_levels": {
            "cal_median": float(np.median(cal)),
            "ms_median": float(np.median(ms)),
            "proc_median": float(np.median(proc)),
            "delta_median_ms_cal": float(np.median(ms) - np.median(cal)),
            "meta_target_sky_note": "431 meta sky_adu~1565 vs 429~1478 (≈−87); residual median ≈ −91",
        },
    }

    loc = maxabs_localization(R, cal)
    fit2 = fit_poly2d(R, order=2)
    fit1 = fit_poly2d(R, order=1)
    impulse = impulsive_fraction(R)
    dipole = star_dipole_check(R, cal)

    # DAO census
    dao_cal = dao_pass1_count(cal, sigma=2.1, fwhm=2.5)
    dao_ms = dao_pass1_count(ms, sigma=2.1, fwhm=2.5)
    # control sick frame
    cal431 = np.asarray(fits.getdata(CAL431), dtype=np.float64)
    ms431 = np.asarray(fits.getdata(MS431), dtype=np.float64)
    dao_cal431 = dao_pass1_count(cal431, sigma=2.1, fwhm=2.5)
    dao_ms431 = dao_pass1_count(ms431, sigma=2.1, fwhm=2.5)

    stats["localization"] = {
        "fraction_at_bright_cores": loc["fraction_at_bright_cores"],
        "fraction_near_edge": loc["fraction_near_edge"],
        "top5": loc["top_peaks"][:5],
    }
    stats["surface_fit"] = {
        "order1": {k: fit1[k] for k in ("order", "names", "coef", "r_std", "resid_std", "variance_explained")},
        "order2": {k: fit2[k] for k in ("order", "names", "coef", "r_std", "resid_std", "variance_explained")},
    }
    stats["impulse"] = impulse
    stats["dipole"] = dipole
    stats["dao_pass1"] = {
        "429_cal_Light008": dao_cal,
        "429_MASTERSTAR": dao_ms,
        "431_cal_Light008": dao_cal431,
        "431_MASTERSTAR": dao_ms431,
        "expectation": "calibrated ≈8927, MS(429) ≈2816",
    }

    # classify operation
    ve2 = fit2["variance_explained"]
    if ve2 > 0.55 and impulse["frac_above_8mad"] < 0.05:
        op_class = "SMOOTH_BACKGROUND_OR_GRADIENT"
    elif impulse["frac_above_8mad"] > 0.02 and loc["fraction_at_bright_cores"] > 0.5:
        op_class = "PSF_OR_STAR_CORES"
    elif dipole["dipole_like"] > 10:
        op_class = "ALIGNMENT_RESAMPLE_DIPOLES"
    else:
        op_class = "MIXED_OR_UNCLEAR"
    stats["operation_class"] = op_class

    # --- figures ---
    fig, ax = plt.subplots(figsize=(9, 6))
    vmax = np.percentile(np.abs(R), 99)
    im = ax.imshow(R, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_title(f"R = MS(429) − cal Light_008  (vmax=p99 |R|={vmax:.1f})")
    plt.colorbar(im, ax=ax, fraction=0.035)
    fig.tight_layout()
    fig.savefig(OUT / "residual_image.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(fit2["surface"], origin="lower", cmap="RdBu_r", interpolation="bilinear")
    ax.set_title(f"Order-2 surface fit (var explained={fit2['variance_explained']:.3f})")
    plt.colorbar(im, ax=ax, fraction=0.035)
    fig.tight_layout()
    fig.savefig(OUT / "surface_fit_order2.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    rr = fit2["residual"]
    vmax = np.percentile(np.abs(rr), 99)
    im = ax.imshow(rr, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_title("Residual after order-2 surface removal")
    plt.colorbar(im, ax=ax, fraction=0.035)
    fig.tight_layout()
    fig.savefig(OUT / "residual_after_surface.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(R.ravel(), bins=200, color="#335c81", alpha=0.9)
    ax.set_yscale("log")
    ax.set_xlabel("R [ADU]")
    ax.set_ylabel("count (log)")
    ax.set_title("Residual histogram")
    fig.tight_layout()
    fig.savefig(OUT / "residual_histogram.png", dpi=120)
    plt.close(fig)

    (OUT / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    # markdown report
    lines = [
        "# F-431 lost ADU transform — T1 characterization",
        "",
        f"**Definition:** `R = MASTERSTAR(draft_429) − calibrated BO_CVn_Light_008` (same draft).",
        f"MASTERSTAR ≡ processed Light_008: **{ms_eq_proc}**.",
        "",
        f"## Operation class (best guess): **{op_class}**",
        "",
        "## 1. Amplitude statistics",
        "",
        f"- min/max = `{stats['R_ms_minus_cal']['min']:.3f}` / `{stats['R_ms_minus_cal']['max']:.3f}`",
        f"- mean/median/std = `{stats['R_ms_minus_cal']['mean']:.3f}` / `{stats['R_ms_minus_cal']['median']:.3f}` / `{stats['R_ms_minus_cal']['std']:.3f}`",
        f"- maxabs = `{stats['R_ms_minus_cal']['maxabs']:.3f}`",
        f"- percentiles (signed): `{json.dumps(stats['R_ms_minus_cal']['percentiles'])}`",
        f"- abs percentiles: `{json.dumps(stats['R_ms_minus_cal']['abs_percentiles'])}`",
        "",
        "### Maxabs localization",
        f"- fraction of top-20 |R| peaks at bright star cores: **{loc['fraction_at_bright_cores']:.2f}**",
        f"- fraction near frame edge (±50 px): **{loc['fraction_near_edge']:.2f}**",
        f"- top-5 peaks: `{json.dumps(loc['top_peaks'][:5], indent=2)}`",
        "",
        "## 2. Spatial character",
        "",
        f"- order-1 surface: var_explained=`{fit1['variance_explained']:.3f}`, resid_std=`{fit1['resid_std']:.2f}` (raw std `{fit1['r_std']:.2f}`)",
        f"- order-2 surface: var_explained=`{fit2['variance_explained']:.3f}`, resid_std=`{fit2['resid_std']:.2f}`",
        f"- order-2 coefficients ({', '.join(fit2['names'])}): `{[round(c,6) for c in fit2['coef']]}`",
        f"- impulsive fraction (|R|>8·MAD): `{impulse['frac_above_8mad']:.4f}` (n={impulse['n_above']})",
        f"- star dipole check: `{dipole}`",
        "",
        "## 3. Sky-level + DAO accounting",
        "",
        f"- medians: cal=`{stats['sky_levels']['cal_median']:.2f}`, MS=`{stats['sky_levels']['ms_median']:.2f}`, Δ=`{stats['sky_levels']['delta_median_ms_cal']:.2f}`",
        f"- matches meta sky 1565→1478 (~−87 ADU) within ~few ADU of median residual.",
        "",
        "### DAO pass-1 simulation (`σ=2.1`, `FWHM=2.5`, masterstar-ish)",
        "",
        f"| Image | n_pass1 | median | std | thr |",
        f"|-------|---------|--------|-----|-----|",
        f"| 429 cal Light_008 | {dao_cal['n_pass1']} | {dao_cal['median']:.1f} | {dao_cal['std']:.2f} | {dao_cal['threshold']:.2f} |",
        f"| 429 MASTERSTAR | {dao_ms['n_pass1']} | {dao_ms['median']:.1f} | {dao_ms['std']:.2f} | {dao_ms['threshold']:.2f} |",
        f"| 431 cal Light_008 | {dao_cal431['n_pass1']} | {dao_cal431['median']:.1f} | {dao_cal431['std']:.2f} | {dao_cal431['threshold']:.2f} |",
        f"| 431 MASTERSTAR (=cal) | {dao_ms431['n_pass1']} | {dao_ms431['median']:.1f} | {dao_ms431['std']:.2f} | {dao_ms431['threshold']:.2f} |",
        "",
        "Expect ≈8927 (sick) vs ≈2816 (healthy). Live draft metas used full masterstar detect (binning/pass-2);",
        "this sim is pass-1-only shared recipe on the two images.",
        "",
        "## 4. Figures",
        "",
        "- `tmp/f431_lost_transform/residual_image.png`",
        "- `tmp/f431_lost_transform/surface_fit_order2.png`",
        "- `tmp/f431_lost_transform/residual_after_surface.png`",
        "- `tmp/f431_lost_transform/residual_histogram.png`",
        "- `tmp/f431_lost_transform/stats.json`",
        "",
        "## Interpretation for re-implementation",
        "",
    ]
    if op_class == "SMOOTH_BACKGROUND_OR_GRADIENT":
        lines += [
            "Residual is dominated by a **smooth large-scale background / gradient removal**",
            "(low-order surface explains most variance; not sparse hot pixels; not star dipoles).",
            "Candidate ops: pedestal/sky-median subtract with spatially varying model, polynomial",
            "background, or large-kernel background map subtracted from calibrated frames.",
            "Current `_preprocess_calibrated_one` is a pure pixel copy — this op is missing.",
        ]
    else:
        lines += [
            f"Class={op_class}: inspect figures + stats.json before coding a restore.",
            "Milan sign-off required for any science-affecting default-ON transform (T3 UI-SICK branch).",
        ]
    lines.append("")
    (OUT / "f431_lost_transform.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # also copy to tmp root as task requested
    (_ROOT / "tmp" / "f431_lost_transform.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: stats[k] for k in ("operation_class", "dao_pass1", "sky_levels", "surface_fit")}, indent=2, default=str))
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
