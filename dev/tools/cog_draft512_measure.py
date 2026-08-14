#!/usr/bin/env python3
"""COG-A1-01: curves of growth on draft 512 (measurement only, no science pixel modification)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from photutils.aperture import CircularAnnulus, CircularAperture

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src_py"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from photometry_core import _sky_pp_from_annulus_image  # noqa: E402

DRAFT = REPO / "Archive" / "Drafts" / "draft_000512"
ALIGNED = DRAFT / "detrended_aligned" / "lights" / "NoFilter_60_2"
STAR_CSV = REPO / "dev" / "results" / "draft512_star_by_frame.csv"
FWHM_CSV = REPO / "dev" / "results" / "draft512_per_frame_fwhm.csv"
OUT_DIR = REPO / "dev" / "results"

TARGET_ID = "1498613634033133184"
R_IN = 15.68165
R_OUT = 29.7126
R_MAX = R_IN - 0.25
R_STEP = 0.25
R_MIN = 0.5
ASYM_TOL = 0.01


def _radii_ladder() -> np.ndarray:
    rs = np.arange(R_MIN, R_MAX + 1e-6, R_STEP, dtype=np.float64)
    return rs


def _sky_sub_flux(d: np.ndarray, x: float, y: float, r: float, r_in: float, r_out: float) -> float:
    pos = (float(x), float(y))
    ap = CircularAperture(pos, r=r)
    ann = CircularAnnulus(pos, r_in=r_in, r_out=r_out)
    raw = float(ap.do_photometry(d, method="exact")[0][0])
    ann_mask = ann.to_mask(method="center")
    sky = _sky_pp_from_annulus_image(d, ann_mask.to_image(d.shape))
    area = float(ap.area)
    return raw - sky * area


def _interp_radius(radii: np.ndarray, fluxes: np.ndarray, frac: float) -> float:
    fmax = float(fluxes[-1])
    if not math.isfinite(fmax) or fmax <= 0:
        return float("nan")
    target = frac * fmax
    for i in range(1, len(radii)):
        if fluxes[i] >= target:
            r0, r1 = float(radii[i - 1]), float(radii[i])
            f0, f1 = float(fluxes[i - 1]), float(fluxes[i])
            if f1 <= f0:
                return r1
            t = (target - f0) / (f1 - f0)
            return r0 + t * (r1 - r0)
    return float("nan")


def _asymptote(radii: np.ndarray, fluxes: np.ndarray) -> tuple[float, str, bool]:
    """Return (f_asym, method, reached_flat_inside_r_in)."""
    if len(fluxes) < 4:
        return float(fluxes[-1]), "last_point", False
    tail = fluxes[-4:]
    rel = np.abs(np.diff(tail)) / np.maximum(np.abs(tail[:-1]), 1e-9)
    if np.all(rel < ASYM_TOL):
        return float(np.median(tail)), "tail_median_flat", True
    # linear extrapolation in 1/r from last two points (diagnostic only)
    r1, r2 = float(radii[-2]), float(radii[-1])
    f1, f2 = float(fluxes[-2]), float(fluxes[-1])
    if r2 > r1:
        slope = (f2 - f1) / (r2 - r1)
        f_ext = f2 + slope * max(0.0, 0.5)
        if f_ext > f2 * 1.02:
            return float(f_ext), "linear_extrap_+0.5px", False
    return float(f2), "last_point_not_flat", False


def load_star_positions() -> pd.DataFrame:
    sb = pd.read_csv(STAR_CSV)
    rows = []
    proc_dir = ALIGNED
    for proc_name, grp in sb.groupby("frame_proc_csv"):
        proc_path = proc_dir / proc_name
        if not proc_path.is_file():
            continue
        pdf = pd.read_csv(
            proc_path,
            usecols=["catalog_id", "x", "y", "dao_flux", "airmass", "aperture_r_px"],
        )
        pdf["catalog_id"] = pdf["catalog_id"].astype(str)
        keep = set(grp["catalog_id"].astype(str))
        pdf = pdf[pdf["catalog_id"].isin(keep)]
        fits_name = proc_name.replace("proc_", "").replace(".csv", ".fits")
        for _, r in pdf.iterrows():
            rows.append(
                {
                    "frame": fits_name,
                    "frame_proc_csv": proc_name,
                    "catalog_id": str(r["catalog_id"]),
                    "x": float(r["x"]),
                    "y": float(r["y"]),
                    "dao_flux": float(r["dao_flux"]),
                    "airmass": float(r["airmass"]),
                    "aperture_r_px": float(r["aperture_r_px"]),
                    "is_target": str(r["catalog_id"]) == TARGET_ID,
                }
            )
    return pd.DataFrame(rows)


def measure_cog(pos_df: pd.DataFrame) -> pd.DataFrame:
    radii = _radii_ladder()
    out_rows: list[dict] = []
    summary_rows: list[dict] = []

    frames = sorted(pos_df["frame"].unique())
    for fi, frame in enumerate(frames):
        fpath = ALIGNED / frame
        if not fpath.is_file():
            continue
        with fits.open(fpath, memmap=True) as hdul:
            d = np.asarray(hdul[0].data, dtype=np.float64)
        fpos = pos_df[pos_df["frame"] == frame]
        for _, star in fpos.iterrows():
            x, y = float(star["x"]), float(star["y"])
            fluxes = np.array([_sky_sub_flux(d, x, y, r, R_IN, R_OUT) for r in radii])
            f_asym, asym_method, flat_ok = _asymptote(radii, fluxes)
            rap = float(star["aperture_r_px"])
            f_at_ap = float(np.interp(rap, radii, fluxes))
            ee_ap = f_at_ap / f_asym if f_asym > 0 else float("nan")
            summary_rows.append(
                {
                    "frame": frame,
                    "catalog_id": star["catalog_id"],
                    "is_target": bool(star["is_target"]),
                    "x": x,
                    "y": y,
                    "dao_flux": float(star["dao_flux"]),
                    "airmass": float(star["airmass"]),
                    "aperture_r_px": rap,
                    "f_asymptote": f_asym,
                    "asymptote_method": asym_method,
                    "curve_flat_inside_r_in": flat_ok,
                    "f_at_pipeline_r": f_at_ap,
                    "ee_at_pipeline_r": ee_ap,
                    "r50": _interp_radius(radii, fluxes, 0.50),
                    "r80": _interp_radius(radii, fluxes, 0.80),
                    "r90": _interp_radius(radii, fluxes, 0.90),
                    "r95": _interp_radius(radii, fluxes, 0.95),
                    "r_in_limit_px": R_IN,
                    "max_radius_measured_px": float(radii[-1]),
                }
            )
            for r, fl in zip(radii, fluxes, strict=True):
                out_rows.append(
                    {
                        "frame": frame,
                        "catalog_id": star["catalog_id"],
                        "radius_px": float(r),
                        "flux_sky_sub": float(fl),
                        "f_asymptote": f_asym,
                        "ee": float(fl / f_asym) if f_asym > 0 else float("nan"),
                    }
                )
        if (fi + 1) % 20 == 0:
            print(f"COG progress: {fi + 1}/{len(frames)} frames", flush=True)
    return pd.DataFrame(out_rows), pd.DataFrame(summary_rows)


def ensemble_ee(summary: pd.DataFrame) -> pd.DataFrame:
    """Target-to-ensemble enclosed-fraction ratio time series (flux-weighted comp EE)."""
    comp_ids = sorted(summary.loc[~summary["is_target"], "catalog_id"].unique())
    frames = sorted(summary["frame"].unique())
    fwhm = pd.read_csv(FWHM_CSV)
    fwhm["frame"] = fwhm["frame"].astype(str)
    rows = []
    for frame in frames:
        sub = summary[summary["frame"] == frame]
        tgt = sub[sub["is_target"]]
        if tgt.empty:
            continue
        ee_t = float(tgt.iloc[0]["ee_at_pipeline_r"])
        comps = sub[sub["catalog_id"].isin(comp_ids)]
        w = comps["dao_flux"].to_numpy(dtype=np.float64)
        ee_c = comps["ee_at_pipeline_r"].to_numpy(dtype=np.float64)
        if np.sum(w > 0) < 1:
            ee_ens = float("nan")
        else:
            ee_ens = float(np.sum(w * ee_c) / np.sum(w))
        ratio = ee_t / ee_ens if ee_ens > 0 else float("nan")
        dm_mmag = 2500.0 * math.log10(ratio) if ratio > 0 else float("nan")
        fw_row = fwhm[fwhm["frame"] == frame]
        fwhm_px = float(fw_row.iloc[0]["fwhm_px"]) if not fw_row.empty else float("nan")
        sky_med = float(comps["dao_flux"].median())  # placeholder; overwritten below
        rows.append(
            {
                "frame": frame,
                "ee_target": ee_t,
                "ee_ensemble_flux_weighted": ee_ens,
                "ee_ratio_target_over_ensemble": ratio,
                "delta_mmag_from_ee_ratio": dm_mmag,
                "fwhm_px": fwhm_px,
                "airmass": float(tgt.iloc[0]["airmass"]),
                "n_comps": int(len(comps)),
            }
        )
    ts = pd.DataFrame(rows)
    # sky from star csv
    sb = pd.read_csv(STAR_CSV)
    sb["frame"] = sb["frame_proc_csv"].str.replace("proc_", "", regex=False).str.replace(".csv", ".fits", regex=False)
    sky_by_frame = (
        sb.groupby("frame")["sky_adu_per_px_annulus"].median().reset_index().rename(columns={"sky_adu_per_px_annulus": "sky_adu_per_px"})
    )
    ts = ts.merge(sky_by_frame, on="frame", how="left")
    return ts


def block_bootstrap_floor(ts: pd.DataFrame, *, n_boot: int = 5000) -> dict:
    """Detection floor on peak-to-peak of delta_mmag series (block bootstrap by frame)."""
    y = ts["delta_mmag_from_ee_ratio"].to_numpy(dtype=np.float64)
    y = y[np.isfinite(y)]
    if y.size < 4:
        return {"floor_mmag": float("nan"), "observed_p2p_mmag": float("nan")}
    obs_p2p = float(np.max(y) - np.min(y))
    n = y.size
    rng = np.random.default_rng(42)
    p2ps = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        samp = y[idx]
        p2ps.append(float(np.max(samp) - np.min(samp)))
    p2ps = np.asarray(p2ps)
    floor = float(np.percentile(p2ps, 95))
    return {
        "observed_p2p_mmag": obs_p2p,
        "floor_mmag_p95_bootstrap": floor,
        "n_frames": int(n),
        "method": "Block bootstrap (5000 resamples, iid frame draws) 95th pct of p2p",
        "citation": "Efron and Tibshirani 1993; block resampling Kunsch 1989",
        "above_floor": bool(obs_p2p > floor),
    }


def correlations(ts: pd.DataFrame) -> dict:
    out = {}
    y = ts["delta_mmag_from_ee_ratio"]
    for col in ("fwhm_px", "airmass", "sky_adu_per_px"):
        x = ts[col]
        m = np.isfinite(x.to_numpy()) & np.isfinite(y.to_numpy())
        if m.sum() >= 5:
            out[f"pearson_r_{col}"] = float(np.corrcoef(x[m], y[m])[0, 1])
        else:
            out[f"pearson_r_{col}"] = float("nan")
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pos = load_star_positions()
    print(f"Loaded {len(pos)} star-frame positions", flush=True)
    cog_long, cog_sum = measure_cog(pos)
    cog_long.to_csv(OUT_DIR / "draft512_cog_curves.csv", index=False)
    cog_sum.to_csv(OUT_DIR / "draft512_cog_summary.csv", index=False)

    ts = ensemble_ee(cog_sum)
    ts.to_csv(OUT_DIR / "draft512_cog_ee_timeseries.csv", index=False)

    floor = block_bootstrap_floor(ts)
    stats = {
        "n_frames": int(len(ts)),
        "ee_target_median": float(ts["ee_target"].median()),
        "ee_ensemble_median": float(ts["ee_ensemble_flux_weighted"].median()),
        "delta_mmag_std": float(ts["delta_mmag_from_ee_ratio"].std(ddof=1)),
        "delta_mmag_p2p": floor["observed_p2p_mmag"],
        "detection_floor": floor,
        "correlations": correlations(ts),
        "curve_flat_fraction": float(cog_sum["curve_flat_inside_r_in"].mean()),
    }
    with open(OUT_DIR / "draft512_cog_part_c_stats.json", "w", encoding="ascii") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
