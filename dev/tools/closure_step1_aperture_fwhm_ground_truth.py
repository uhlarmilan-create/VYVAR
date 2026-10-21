#!/usr/bin/env python3
"""Closure Step 1 (A-1): independent FWHM ground truth vs SNR-table apertures.

Standalone measurement script -- does NOT import VYVAR aperture sizing code.
Output: JSON path passed as --out (default tmp/closure_step1_results.json).

Usage:
  python dev/tools/closure_step1_aperture_fwhm_ground_truth.py \\
    --draft Archive/Drafts/draft_000435_snapshot_skysurface_20260716 \\
    --out tmp/closure_step1_results.json
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.modeling import fitting, models


def _mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)))


def _rank_frames_by_vy_fwhm(lights_dir: Path) -> list[tuple[float, str, Path]]:
    rows: list[tuple[float, str, Path]] = []
    for fp in sorted(lights_dir.glob("proc_BO_CVn_Light_*.fits")):
        with fits.open(fp, memmap=False) as hdul:
            vy = hdul[0].header.get("VY_FWHM", float("nan"))
        try:
            vf = float(vy)
        except (TypeError, ValueError):
            vf = float("nan")
        rows.append((vf, fp.name, fp))
    rows.sort(key=lambda t: (t[0] if math.isfinite(t[0]) else 1e9, t[1]))
    return rows


def _select_stars_from_proc_csv(
    proc_csv: Path,
    *,
    sat_limit: float,
    nominal_fwhm: float,
    min_peak_frac: float = 0.10,
    max_peak_frac: float = 0.60,
    min_stars: int = 15,
) -> pd.DataFrame:
    df = pd.read_csv(proc_csv, dtype={"catalog_id": str})
    need = {"x", "y", "peak_max_adu", "photometry_ok"}
    if not need.issubset(df.columns):
        raise RuntimeError(f"proc CSV missing columns: {need - set(df.columns)}")
    sat = float(sat_limit) if math.isfinite(sat_limit) and sat_limit > 0 else 65535.0
    lo, hi = min_peak_frac * sat, max_peak_frac * sat
    ok = df["photometry_ok"].astype(str).str.lower().isin(("true", "1", "yes"))
    cand = df.loc[ok & df["peak_max_adu"].between(lo, hi)].copy()
    cand = cand.sort_values("peak_max_adu", ascending=False)
    min_sep = 6.0 * nominal_fwhm
    edge = 40.0
    picked: list[int] = []
    coords: list[tuple[float, float]] = []
    for idx, row in cand.iterrows():
        x, y = float(row["x"]), float(row["y"])
        if any(math.hypot(x - px, y - py) < min_sep for px, py in coords):
            continue
        picked.append(int(idx))
        coords.append((x, y))
        if len(picked) >= min_stars:
            break
    return cand.loc[picked] if picked else cand.head(min_stars)


def _fit_gaussian_with_background(
    data: np.ndarray,
    x0: float,
    y0: float,
    *,
    box: int,
    fwhm_hint: float,
) -> dict[str, float] | None:
    h, w = data.shape
    xc, yc = int(round(x0)), int(round(y0))
    if not (box <= xc < w - box and box <= yc < h - box):
        return None
    cut = data[yc - box : yc + box + 1, xc - box : xc + box + 1].astype(np.float64)
    if cut.shape != (2 * box + 1, 2 * box + 1):
        return None
    yy, xx = np.mgrid[0 : cut.shape[0], 0 : cut.shape[1]]
    fitter = fitting.LevMarLSQFitter()
    g0 = models.Gaussian2D(
        amplitude=max(float(np.max(cut) - np.median(cut)), 1.0),
        x_mean=float(box),
        y_mean=float(box),
        x_stddev=fwhm_hint / 2.355,
        y_stddev=fwhm_hint / 2.355,
    )
    c0 = models.Const2D(amplitude=float(np.median(cut)))
    try:
        m = fitter(g0 + c0, xx, yy, cut)
        sx = abs(float(getattr(m.x_stddev, "value", m.x_stddev)))
        sy = abs(float(getattr(m.y_stddev, "value", m.y_stddev)))
        fwhm_x = 2.355 * sx
        fwhm_y = 2.355 * sy
        if not all(math.isfinite(v) and v > 0 for v in (fwhm_x, fwhm_y)):
            return None
        return {"fwhm_x": fwhm_x, "fwhm_y": fwhm_y, "fwhm_mean": 0.5 * (fwhm_x + fwhm_y)}
    except (ValueError, TypeError, IndexError, np.linalg.LinAlgError, fitting.NonFiniteValueError):
        return None


def _curve_of_growth(
    data: np.ndarray,
    x0: float,
    y0: float,
    *,
    radii: np.ndarray,
    r_in_factor: float,
    r_out_factor: float,
    fwhm_hint: float,
) -> dict[str, Any]:
    h, w = data.shape
    xc, yc = int(round(x0)), int(round(y0))
    max_r = int(math.ceil(float(np.max(radii)) + max(r_in_factor, r_out_factor) * fwhm_hint + 2))
    if xc - max_r < 0 or yc - max_r < 0 or xc + max_r >= w or yc + max_r >= h:
        return {"ok": False}
    yy, xx = np.mgrid[yc - max_r : yc + max_r + 1, xc - max_r : xc + max_r + 1]
    patch = data[yc - max_r : yc + max_r + 1, xc - max_r : xc + max_r + 1].astype(np.float64)
    rel_y = yy - yc
    rel_x = xx - xc
    dist = np.hypot(rel_x, rel_y)
    fluxes: list[float] = []
    for r in radii:
        ap = dist <= r
        rin = max(r + 0.5, r_in_factor * fwhm_hint)
        rout = max(rin + 0.5, r_out_factor * fwhm_hint)
        sky_mask = (dist >= rin) & (dist <= rout)
        sky = float(np.median(patch[sky_mask])) if np.any(sky_mask) else 0.0
        flux = float(np.sum((patch - sky)[ap]))
        fluxes.append(flux)
    arr = np.asarray(fluxes, dtype=np.float64)
    norm = arr[-1] if arr[-1] > 0 else 1.0
    ee = arr / norm
    def _radius_at(target: float) -> float:
        hit = np.where(ee >= target)[0]
        return float(radii[hit[0]]) if hit.size else float("nan")
    r50 = _radius_at(0.5)
    r90 = _radius_at(0.9)
    fwhm_from_r50 = 2.0 * r50 if math.isfinite(r50) else float("nan")
    fwhm_from_r90 = r90 / 1.34 if math.isfinite(r90) else float("nan")
    return {
        "ok": True,
        "radii": radii.tolist(),
        "ee": ee.tolist(),
        "r_at_ee_0.5": r50,
        "r_at_ee_0.9": r90,
        "fwhm_from_r50": fwhm_from_r50,
        "fwhm_from_r90": fwhm_from_r90,
    }


def _measure_frame(
    fits_path: Path,
    proc_csv: Path,
    *,
    sat_limit: float,
    nominal_fwhm: float,
) -> dict[str, Any]:
    with fits.open(fits_path, memmap=False) as hdul:
        data = hdul[0].data.astype(np.float64)
        hdr = hdul[0].header
    stars = _select_stars_from_proc_csv(
        proc_csv, sat_limit=sat_limit, nominal_fwhm=nominal_fwhm
    )
    n_stars_selected = int(len(stars))
    box = max(8, int(math.ceil(4.0 * nominal_fwhm)))
    fwhm_hint = nominal_fwhm
    fits_list: list[float] = []
    for _, row in stars.iterrows():
        res = _fit_gaussian_with_background(
            data, float(row["x"]), float(row["y"]), box=box, fwhm_hint=fwhm_hint
        )
        if res:
            fits_list.append(res["fwhm_mean"])
    radii = np.arange(0.5, 12.05, 0.5)
    cogs: list[dict[str, Any]] = []
    for _, row in stars.iterrows():
        cogs.append(
            _curve_of_growth(
                data,
                float(row["x"]),
                float(row["y"]),
                radii=radii,
                r_in_factor=4.75,
                r_out_factor=9.0,
                fwhm_hint=fwhm_hint,
            )
        )
    cogs_ok = [c for c in cogs if c.get("ok")]
    fwhm_cog = [c["fwhm_from_r50"] for c in cogs_ok if math.isfinite(c.get("fwhm_from_r50", float("nan")))]
    # median COG curve
    ee_stack = np.median(np.array([c["ee"] for c in cogs_ok], dtype=np.float64), axis=0) if cogs_ok else None
    return {
        "fits_path": str(fits_path),
        "vy_fwhm_header": float(hdr.get("VY_FWHM", float("nan"))),
        "n_stars_selected": n_stars_selected,
        "n_stars_fit": len(fits_list),
        "fwhm_gauss_median": float(np.median(fits_list)) if fits_list else float("nan"),
        "fwhm_gauss_mad": _mad(np.array(fits_list)),
        "fwhm_cog_from_r50_median": float(np.median(fwhm_cog)) if fwhm_cog else float("nan"),
        "ee_median_curve": ee_stack.tolist() if ee_stack is not None else None,
        "radii": radii.tolist(),
    }


def _proc_stats_for_stars(
    draft: Path,
    star_ids: list[str],
    *,
    frame_name: str = "proc_BO_CVn_Light_063.csv",
) -> dict[str, Any]:
    proc_dir = draft / "detrended_aligned/lights/NoFilter_60_2"
    cols = [
        "aperture_r_px",
        "fwhm_estimate_px",
        "sky_annulus_r_out_px",
        "fwhm_px_for_aperture",
        "fwhm_px_scope",
        "snr_aperture_mode",
        "aperture_factor_applied",
    ]
    per_star: dict[str, Any] = {}
    all_frames: list[pd.DataFrame] = []
    for proc in sorted(proc_dir.glob("proc_BO_CVn_Light_*.csv")):
        df = pd.read_csv(proc, dtype={"catalog_id": str})
        all_frames.append(df)
    for sid in star_ids:
        vals: dict[str, list[float]] = {c: [] for c in cols if c in all_frames[0].columns}
        for df in all_frames:
            sub = df.loc[df["catalog_id"] == sid]
            if sub.empty:
                continue
            row = sub.iloc[0]
            for c in vals:
                try:
                    v = float(row[c])
                    if math.isfinite(v):
                        vals[c].append(v)
                except (TypeError, ValueError):
                    pass
        per_star[sid] = {
            c: {"min": float(np.min(v)), "median": float(np.median(v)), "max": float(np.max(v)), "n": len(v)}
            for c, v in vals.items()
            if v
        }
    one = proc_dir / frame_name
    df1 = pd.read_csv(one, dtype={"catalog_id": str})
    dist = df1["aperture_r_px"].value_counts().sort_index()
    r_min = 1.9160000000000001
    r_max = 5.9875
    return {
        "per_star_all_frames": per_star,
        "one_frame_distribution": {
            "frame": frame_name,
            "distinct_aperture_r_px": {str(k): int(v) for k, v in dist.items()},
            "n_on_r_min": int((df1["aperture_r_px"] - r_min).abs().lt(1e-6).sum()),
            "n_on_r_max": int((df1["aperture_r_px"] - r_max).abs().lt(1e-6).sum()),
            "n_stars": len(df1),
        },
        "missing_provenance_cols": [c for c in cols if c not in df1.columns],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("tmp/closure_step1_results.json"))
    args = ap.parse_args()
    draft = args.draft.resolve()
    lights = draft / "detrended_aligned/lights/NoFilter_60_2"
    snr_path = draft / "aperture_snr_table.json"
    with snr_path.open(encoding="utf-8") as f:
        snr_table = json.load(f)

    ranked = _rank_frames_by_vy_fwhm(lights)
    picks = {
        "best": ranked[0],
        "median": ranked[len(ranked) // 2],
        "worst": ranked[-1],
    }

    # saturation from first proc csv
    sample_csv = lights / "proc_BO_CVn_Light_001.csv"
    sat = float(pd.read_csv(sample_csv, nrows=1)["saturate_limit_adu"].iloc[0])
    nominal = float(snr_table["fwhm_px"])

    frame_results: dict[str, Any] = {}
    for label, (_vy, name, fpath) in picks.items():
        proc_csv = lights / name.replace(".fits", ".csv")
        frame_results[label] = _measure_frame(
            fpath, proc_csv, sat_limit=sat, nominal_fwhm=nominal
        )

    # MASTERSTAR at platesolve path
    ms_path = draft / "platesolve/NoFilter_60_2/MASTERSTAR.fits"
    ms_proc = lights / "proc_BO_CVn_Light_008.csv"  # nearest catalog positions
    frame_results["masterstar"] = _measure_frame(
        ms_path, ms_proc, sat_limit=sat, nominal_fwhm=nominal
    )

    focus = "1498135552633294976"
    comps = [
        "1496300948763054976",  # G~7.99
        "1498927097925811072",  # G~8.29
        "1498735778606786816",  # G~9.11
        "1498072743031629824",  # G~10.93
        "1497558618266311808",  # G~9.13 mid-faint
    ]

    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "draft": str(draft),
        "snr_table_path": str(snr_path),
        "snr_table_mtime": datetime.fromtimestamp(snr_path.stat().st_mtime).isoformat(),
        "snr_table": snr_table,
        "vy_fwhm_ranking": [{"vy_fwhm": vy, "file": name} for vy, name, _ in ranked],
        "masterstar_source_processed": "processed/lights/NoFilter_60_2/proc_BO_CVn_Light_008.fits",
        "frame_picks": {k: {"vy_fwhm": v[0], "file": v[1]} for k, v in picks.items()},
        "part_b": frame_results,
        "part_a4_a5": _proc_stats_for_stars(draft, [focus] + comps),
        "a6_scale_check": _scale_check(draft),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out}")


def _scale_check(draft: Path) -> dict[str, Any]:
    ms = draft / "platesolve/NoFilter_60_2/MASTERSTAR.fits"
    sci = draft / "detrended_aligned/lights/NoFilter_60_2/proc_BO_CVn_Light_063.fits"
    out: dict[str, Any] = {}
    for label, p in [("masterstar", ms), ("science_frame_063", sci)]:
        with fits.open(p, memmap=False) as hdul:
            h = hdul[0].header
            out[label] = {
                "path": str(p),
                "naxis1": int(h.get("NAXIS1", 0)),
                "naxis2": int(h.get("NAXIS2", 0)),
                "cdelt1": float(h.get("CDELT1")) if h.get("CDELT1") is not None else None,
                "pc1_1": float(h.get("PC1_1")) if h.get("PC1_1") is not None else None,
                "cd1_1": float(h.get("CD1_1")) if h.get("CD1_1") is not None else None,
            }
    return out


if __name__ == "__main__":
    main()
