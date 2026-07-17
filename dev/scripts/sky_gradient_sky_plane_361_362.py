#!/usr/bin/env python3
"""Sky-gradient / TODO-SKY-PLANE decision metric — drafts 361 & 362 (read-only)."""
from __future__ import annotations

import json
import math
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.aperture import CircularAnnulus, CircularAperture

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from comp_selection_per_target import _angular_distance_deg_vectorized  # noqa: E402
from config import AppConfig  # noqa: E402
from database import VyvarDatabase  # noqa: E402
from photometry_core import (  # noqa: E402
    _aperture_radius_from_snr_table,
    _photometric_error,
    _sky_pp_from_annulus_image,
    load_snr_aperture_table_from_draft_dir,
)

warnings.filterwarnings("ignore", category=UserWarning, module="photutils")

DRAFTS = (361, 362)
SETUP = "NoFilter_60_2"
N_FRAME_SAMPLES = 25
N_LOCAL_FRAMES = 5
MAX_STARS = 450
GRID_N = 10
MIN_CELL_PX = 40
STAR_EXCLUSION_R = 25.0  # px around catalog stars when picking sky cells


def _fit_plane(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[float, float, float]:
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[ok], y[ok], z[ok]
    if z.size < 3:
        return float("nan"), float("nan"), float("nan")
    a = np.column_stack([np.ones_like(x), x, y])
    coef, _, _, _ = np.linalg.lstsq(a, z, rcond=None)
    return float(coef[0]), float(coef[1]), float(coef[2])


def _plane_eval(a: float, b: float, c: float, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return a + b * x + c * y


def _frame_time_key(path: Path) -> tuple:
    try:
        with fits.open(path, memmap=True) as hd:
            h = hd[0].header
        for key in ("JD", "MJD-OBS", "DATE-OBS"):
            if key in h:
                return (0, str(h[key]))
    except Exception:  # noqa: BLE001
        pass
    return (1, path.name)


def _grid_sky_samples(
    data: np.ndarray,
    *,
    star_xy: np.ndarray,
    n_grid: int = GRID_N,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = data.shape
    xs, ys, zs = [], [], []
    sx = max(MIN_CELL_PX, w // n_grid)
    sy = max(MIN_CELL_PX, h // n_grid)
    for j in range(n_grid):
        for i in range(n_grid):
            x1, x2 = i * sx, min(w, (i + 1) * sx)
            y1, y2 = j * sy, min(h, (j + 1) * sy)
            if x2 - x1 < MIN_CELL_PX or y2 - y1 < MIN_CELL_PX:
                continue
            cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
            if star_xy.size:
                dxy = np.hypot(star_xy[:, 0] - cx, star_xy[:, 1] - cy)
                if np.any(dxy < STAR_EXCLUSION_R):
                    continue
            cut = data[y1:y2, x1:x2]
            finite = cut[np.isfinite(cut)]
            if finite.size < 20:
                continue
            _, med, _ = sigma_clipped_stats(finite, sigma=3.0, maxiters=2)
            if not math.isfinite(float(med)):
                continue
            xs.append(cx)
            ys.append(cy)
            zs.append(float(med))
    return np.asarray(xs), np.asarray(ys), np.asarray(zs)


def _frame_gradient_stats(
    data: np.ndarray,
    star_xy: np.ndarray,
) -> dict[str, float]:
    xs, ys, zs = _grid_sky_samples(data, star_xy=star_xy)
    if zs.size < 6:
        return {"n_cells": float(zs.size), "grad_pp_adu": float("nan"), "grad_pct": float("nan"), "sky_med": float("nan")}
    a, b, c = _fit_plane(xs, ys, zs)
    h, w = data.shape
    corners_x = np.array([0.0, w - 1.0, 0.0, w - 1.0])
    corners_y = np.array([0.0, 0.0, h - 1.0, h - 1.0])
    corner_z = _plane_eval(a, b, c, corners_x, corners_y)
    grad_pp = float(np.max(corner_z) - np.min(corner_z))
    sky_med = float(np.median(zs))
    grad_pct = 100.0 * grad_pp / sky_med if sky_med > 0 else float("nan")
    return {
        "n_cells": float(zs.size),
        "grad_pp_adu": grad_pp,
        "grad_pct": grad_pct,
        "sky_med": sky_med,
        "plane_b": b,
        "plane_c": c,
    }


def _local_sky_pair(
    data: np.ndarray,
    x: float,
    y: float,
    *,
    r_ap: float,
    ann_in: float,
    ann_out: float,
) -> dict[str, float]:
    pos = np.array([[x, y]])
    ann = CircularAnnulus(pos, r_in=ann_in, r_out=ann_out)
    ap = CircularAperture(pos, r=r_ap)
    masks = ann.to_mask(method="center")
    mask_obj = masks[0] if isinstance(masks, list) else masks
    ann_img = mask_obj.to_image(data.shape)
    sel = ann_img > 0
    if sel.sum() < 5:
        return {}
    ys, xs = np.where(sel)
    vals = data[ys, xs].astype(np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size < 5:
        return {}
    sky_med = _sky_pp_from_annulus_image(data, ann_img)
    a, b, c = _fit_plane(xs.astype(float), ys.astype(float), vals)
    plane_center = float(a + b * x + c * y)
    plane_pred = _plane_eval(a, b, c, xs.astype(float), ys.astype(float))
    sky_mean = float(np.mean(vals))
    linear_part = plane_center - sky_mean
    residual_part = sky_mean - sky_med
    model_resid = float(np.std(vals - plane_pred))
    area = float(ap.area)
    return {
        "sky_median": sky_med,
        "sky_plane": plane_center,
        "delta_pp": plane_center - sky_med,
        "delta_flux": (plane_center - sky_med) * area,
        "linear_part_pp": linear_part,
        "residual_part_pp": residual_part,
        "model_resid_rms": model_resid,
        "area": area,
        "grad_mag_pp_per_px": float(math.hypot(b, c)),
    }


def _sample_stars(ms_csv: Path, *, max_n: int = MAX_STARS) -> pd.DataFrame:
    df = pd.read_csv(ms_csv, low_memory=False)
    df["mag"] = pd.to_numeric(df.get("mag"), errors="coerce")
    df["x"] = pd.to_numeric(df.get("x"), errors="coerce")
    df["y"] = pd.to_numeric(df.get("y"), errors="coerce")
    df["flux"] = pd.to_numeric(df.get("flux"), errors="coerce")
    df["noise_floor_adu"] = pd.to_numeric(df.get("noise_floor_adu"), errors="coerce")
    usable = df.get("is_usable")
    if usable is not None:
        df = df.loc[pd.Series(usable).fillna(False).astype(bool)]
    df = df.loc[
        np.isfinite(df["mag"])
        & np.isfinite(df["x"])
        & np.isfinite(df["y"])
        & (df["mag"] >= 9.0)
        & (df["mag"] <= 16.0)
        & df.get("likely_saturated", False).fillna(False).eq(False)
    ].copy()
    if len(df) > max_n:
        df = df.sample(n=max_n, random_state=42)
    return df.reset_index(drop=True)


def _comp_target_separation(ps_dir: Path) -> dict[str, float]:
    vt = ps_dir / "variable_targets.csv"
    cs = ps_dir / "comparison_stars.csv"
    if not vt.is_file() or not cs.is_file():
        return {}
    t = pd.read_csv(vt, low_memory=False)
    c = pd.read_csv(cs, low_memory=False)
    t_ra = pd.to_numeric(t.get("ra_deg"), errors="coerce")
    t_de = pd.to_numeric(t.get("dec_deg"), errors="coerce")
    c_ra = pd.to_numeric(c.get("ra_deg"), errors="coerce")
    c_de = pd.to_numeric(c.get("dec_deg"), errors="coerce")
    t = t.loc[np.isfinite(t_ra) & np.isfinite(t_de)].copy()
    c = c.loc[np.isfinite(c_ra) & np.isfinite(c_de)].copy()
    if t.empty or c.empty:
        return {}
    cra = c_ra.to_numpy(dtype=float)
    cde = c_de.to_numpy(dtype=float)
    nearest: list[float] = []
    all_pairs: list[float] = []
    for ra, de in zip(t_ra, t_de):
        d = _angular_distance_deg_vectorized(float(ra), float(de), cra, cde)
        nearest.append(float(np.min(d)))
        all_pairs.extend(d.tolist())
    return {
        "n_targets": int(len(t)),
        "n_comps": int(len(c)),
        "nearest_comp_deg_median": float(np.median(nearest)),
        "nearest_comp_deg_p90": float(np.percentile(nearest, 90)),
        "all_target_comp_deg_median": float(np.median(all_pairs)),
        "all_target_comp_deg_p90": float(np.percentile(all_pairs, 90)),
    }


def run_draft(draft_id: int, *, cfg: AppConfig, db: VyvarDatabase) -> dict[str, Any]:
    draft_dir = Path(cfg.archive_root) / "Drafts" / f"draft_{draft_id:06d}"
    ps_dir = draft_dir / "platesolve" / SETUP
    aligned = draft_dir / "detrended_aligned" / "lights" / SETUP
    ms_csv = ps_dir / "masterstars_full_match.csv"
    meta_path = ps_dir / "masterstar_epsf_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    fwhm_px = float(meta.get("fwhm_px", 3.37))
    plate_scale = float(meta.get("plate_scale_arcsec_px", 9.77))

    snr_table = load_snr_aperture_table_from_draft_dir(draft_dir) or {}
    fwhm_snr = float(snr_table.get("fwhm_px", fwhm_px))
    gain = float(snr_table.get("gain", 3.17))
    read_noise = float(snr_table.get("read_noise", 7.6))
    ann_in = float(cfg.annulus_inner_fwhm) * fwhm_snr
    ann_out = float(cfg.annulus_outer_fwhm) * fwhm_snr

    equip = db.fetch_obs_draft_telescope_equipment(draft_id) or {}
    frame_files = sorted(aligned.glob("proc_*.fits"), key=_frame_time_key)
    if not frame_files:
        raise FileNotFoundError(f"No frames in {aligned}")

    ms = pd.read_csv(ms_csv, low_memory=False, usecols=["x", "y"])
    star_xy = ms[["x", "y"]].apply(pd.to_numeric, errors="coerce").dropna().to_numpy(dtype=float)

    # --- 1. Frame-scale gradient ---
    idxs = np.linspace(0, len(frame_files) - 1, min(N_FRAME_SAMPLES, len(frame_files)), dtype=int)
    frame_rows: list[dict[str, Any]] = []
    for k, fi in enumerate(idxs):
        fpath = frame_files[int(fi)]
        with fits.open(fpath, memmap=True) as hd:
            data = np.asarray(hd[0].data, dtype=np.float64)
        g = _frame_gradient_stats(data, star_xy)
        g["frame"] = fpath.name
        g["frame_idx"] = int(fi)
        frame_rows.append(g)

    fdf = pd.DataFrame(frame_rows)
    nightly_med = float(fdf["sky_med"].median())
    fdf["twilight_like"] = fdf["sky_med"] > 1.25 * nightly_med
    steepest = fdf.loc[fdf["grad_pp_adu"].idxmax()]
    flattest = fdf.loc[fdf["grad_pp_adu"].idxmin()]
    mid = fdf.iloc[len(fdf) // 2]

    # --- 2. Local plane vs median ---
    stars = _sample_stars(ms_csv)
    local_frame_idxs = np.linspace(0, len(frame_files) - 1, min(N_LOCAL_FRAMES, len(frame_files)), dtype=int)
    local_rows: list[dict[str, Any]] = []
    h_img = w_img = 0
    cx = cy = 0.0
    for fi in local_frame_idxs:
        fpath = frame_files[int(fi)]
        with fits.open(fpath, memmap=True) as hd:
            data = np.asarray(hd[0].data, dtype=np.float64)
        h_img, w_img = data.shape
        cx, cy = w_img / 2.0, h_img / 2.0
        for _, row in stars.iterrows():
            mag = float(row["mag"])
            r_ap = _aperture_radius_from_snr_table(
                mag, snr_table, aperture_fwhm_factor=float(cfg.aperture_fwhm_factor), fwhm_px=fwhm_snr
            )
            loc = _local_sky_pair(
                data, float(row["x"]), float(row["y"]), r_ap=r_ap, ann_in=ann_in, ann_out=ann_out
            )
            if not loc:
                continue
            flux = float(row["flux"]) if math.isfinite(float(row["flux"])) else float("nan")
            area = loc["area"]
            sky_pp = loc["sky_median"]
            if not math.isfinite(flux) or flux <= 0:
                nf = float(row.get("noise_floor_adu", float("nan")))
                sigma_flux = nf * math.sqrt(area) if math.isfinite(nf) else float("nan")
            else:
                rel = _photometric_error(flux, sky_pp, area, gain=gain, read_noise=read_noise)
                sigma_flux = rel * flux if math.isfinite(rel) else float("nan")
            delta_flux = loc["delta_flux"]
            frac = abs(delta_flux) / sigma_flux if math.isfinite(sigma_flux) and sigma_flux > 0 else float("nan")
            r_field = math.hypot(float(row["x"]) - cx, float(row["y"]) - cy)
            r_norm = r_field / max(math.hypot(cx, cy), 1.0)
            local_rows.append(
                {
                    "frame_idx": int(fi),
                    "catalog_id": str(row.get("catalog_id", "")),
                    "mag": mag,
                    "delta_flux": delta_flux,
                    "sigma_flux": sigma_flux,
                    "frac_of_phot_err": frac,
                    "linear_part_pp": loc["linear_part_pp"],
                    "residual_part_pp": loc["residual_part_pp"],
                    "model_resid_rms": loc["model_resid_rms"],
                    "r_norm": r_norm,
                }
            )

    ldf = pd.DataFrame(local_rows)
    valid = ldf[np.isfinite(ldf["frac_of_phot_err"])]
    frac_med = float(valid["frac_of_phot_err"].median()) if len(valid) else float("nan")
    frac_p90 = float(valid["frac_of_phot_err"].quantile(0.9)) if len(valid) else float("nan")
    gt05 = float((valid["frac_of_phot_err"] > 0.5).mean()) if len(valid) else float("nan")
    gt10 = float((valid["frac_of_phot_err"] > 1.0).mean()) if len(valid) else float("nan")
    corr_r = float(valid["frac_of_phot_err"].corr(valid["r_norm"])) if len(valid) > 2 else float("nan")

    abs_lin = valid["linear_part_pp"].abs() * valid.get("area", pd.Series(np.nan, index=valid.index))
    # linear/residual split on pp scale (median abs)
    med_abs_lin_pp = float(valid["linear_part_pp"].abs().median()) if len(valid) else float("nan")
    med_abs_res_pp = float(valid["residual_part_pp"].abs().median()) if len(valid) else float("nan")
    med_model_resid = float(valid["model_resid_rms"].median()) if len(valid) else float("nan")
    total_pp = med_abs_lin_pp + med_abs_res_pp
    lin_frac = med_abs_lin_pp / total_pp if total_pp > 0 else float("nan")

    sep = _comp_target_separation(ps_dir)

    out_dir = draft_dir / "diagnostics" / "sky_gradient_sky_plane"
    out_dir.mkdir(parents=True, exist_ok=True)
    fdf.to_csv(out_dir / f"d{draft_id}_frame_gradient.csv", index=False)
    ldf.to_csv(out_dir / f"d{draft_id}_local_sky_diff.csv", index=False)

    return {
        "draft_id": draft_id,
        "setup": SETUP,
        "rig": equip.get("equipment_name"),
        "plate_scale_arcsec_px": plate_scale,
        "fwhm_px": fwhm_px,
        "fwhm_snr_px": fwhm_snr,
        "ann_in_px": ann_in,
        "ann_out_px": ann_out,
        "n_frames_total": len(frame_files),
        "frame_gradient": {
            "n_sampled": int(len(fdf)),
            "grad_pp_adu_median": float(fdf["grad_pp_adu"].median()),
            "grad_pp_adu_p90": float(fdf["grad_pp_adu"].quantile(0.9)),
            "grad_pct_median": float(fdf["grad_pct"].median()),
            "grad_pct_p90": float(fdf["grad_pct"].quantile(0.9)),
            "steepest_frame": str(steepest["frame"]),
            "steepest_grad_pp_adu": float(steepest["grad_pp_adu"]),
            "steepest_grad_pct": float(steepest["grad_pct"]),
            "flattest_grad_pp_adu": float(flattest["grad_pp_adu"]),
            "n_twilight_like": int(fdf["twilight_like"].sum()),
            "mid_frame": str(mid["frame"]),
            "mid_grad_pp_adu": float(mid["grad_pp_adu"]),
            "mid_grad_pct": float(mid["grad_pct"]),
        },
        "local_metric": {
            "n_stars_sampled": int(len(stars)),
            "n_measurements": int(len(valid)),
            "median_frac_of_phot_err": frac_med,
            "p90_frac_of_phot_err": frac_p90,
            "frac_stars_gt_0p5x_err": gt05,
            "frac_stars_gt_1p0x_err": gt10,
            "corr_frac_vs_field_radius": corr_r,
            "median_abs_linear_part_pp": med_abs_lin_pp,
            "median_abs_residual_part_pp": med_abs_res_pp,
            "median_model_resid_rms_pp": med_model_resid,
            "linear_fraction_of_pp_split": lin_frac,
        },
        "comp_target_separation_deg": sep,
    }


def _format_report(results: list[dict[str, Any]], *, psf_flags: dict[str, bool]) -> str:
    lines = [
        "SKY GRADIENT / TODO-SKY-PLANE — drafts 361 & 362 (NoFilter_60_2)",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "Decision metric: plane-vs-median sky difference vs photometric error (not raw gradient alone).",
        f"PSF flags unchanged: {psf_flags}",
        "",
    ]
    for r in results:
        did = r["draft_id"]
        fg = r["frame_gradient"]
        lm = r["local_metric"]
        sep = r.get("comp_target_separation_deg") or {}
        lines.append(f"=== Draft {did} ({r['setup']}) — {r.get('rig')} ===")
        lines.append(
            f"  Plate scale {r['plate_scale_arcsec_px']:.3f} \"/px | FWHM {r['fwhm_px']:.2f} px"
            f" | annulus {r['ann_in_px']:.1f}-{r['ann_out_px']:.1f} px | {r['n_frames_total']} frames"
        )
        lines.append("  Frame-scale gradient (grid sky cells, 2D plane):")
        lines.append(
            f"    sampled {fg['n_sampled']} frames | median peak-to-peak {fg['grad_pp_adu_median']:.1f} ADU"
            f" ({fg['grad_pct_median']:.2f}% of sky) | p90 {fg['grad_pp_adu_p90']:.1f} ADU ({fg['grad_pct_p90']:.2f}%)"
        )
        lines.append(
            f"    steepest: {fg['steepest_frame']} -> {fg['steepest_grad_pp_adu']:.1f} ADU ({fg['steepest_grad_pct']:.2f}%)"
        )
        lines.append(
            f"    flattest/mid: {fg['flattest_grad_pp_adu']:.1f} / {fg['mid_grad_pp_adu']:.1f} ADU"
            f" | twilight-like frames: {fg['n_twilight_like']}"
        )
        lines.append("  Local plane-vs-median (symmetric annulus; production median + clipped plane fit):")
        lines.append(
            f"    stars {lm['n_stars_sampled']} x {N_LOCAL_FRAMES} frames -> {lm['n_measurements']} measurements"
        )
        lines.append(
            f"    |delta_flux|/sigma_flux: median {lm['median_frac_of_phot_err']:.4f}, p90 {lm['p90_frac_of_phot_err']:.4f}"
        )
        lines.append(
            f"    fraction >0.5x phot err: {100*lm['frac_stars_gt_0p5x_err']:.1f}%"
            f" | >1.0x: {100*lm['frac_stars_gt_1p0x_err']:.1f}%"
        )
        lines.append(
            f"    corr(|delta|/sigma, field radius): {lm['corr_frac_vs_field_radius']:.3f}"
        )
        lines.append(
            f"    pp split (median |linear| vs |residual|): {lm['median_abs_linear_part_pp']:.4f}"
            f" vs {lm['median_abs_residual_part_pp']:.4f} ADU/px"
            f" (linear fraction {100*lm['linear_fraction_of_pp_split']:.1f}%)"
            f" | model residual RMS {lm['median_model_resid_rms_pp']:.2f} ADU/px"
        )
        if sep:
            lines.append(
                f"  Comp-target separation: nearest comp median {sep.get('nearest_comp_deg_median', float('nan')):.3f}°"
                f" (p90 {sep.get('nearest_comp_deg_p90', float('nan')):.3f}°)"
                f" | all pairs median {sep.get('all_target_comp_deg_median', float('nan')):.3f}°"
                f" — wide field -> local sky tilts differ between comp and target."
            )
        lines.append("")
    lines.append("=== VERDICT INPUT ===")
    return "\n".join(lines)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cfg = AppConfig()
    psf_flags = {
        "psf_photometry_enabled": bool(cfg.psf_photometry_enabled),
        "psf_adaptive_enabled": bool(cfg.psf_adaptive_enabled),
    }
    db = VyvarDatabase(cfg.database_path)
    results = [run_draft(d, cfg=cfg, db=db) for d in DRAFTS]
    report = _format_report(results, psf_flags=psf_flags)

    # Append verdict from metrics
    verdict_lines = []
    for r in results:
        lm = r["local_metric"]
        fg = r["frame_gradient"]
        materially = lm["frac_stars_gt_0p5x_err"] > 0.05 or lm["median_frac_of_phot_err"] > 0.1
        verdict_lines.append(
            f"Draft {r['draft_id']}: frame gradients reach {fg['grad_pct_p90']:.1f}% of sky (context);"
            f" plane-median flux correction is median {lm['median_frac_of_phot_err']:.3f}x photometric error,"
            f" {100*lm['frac_stars_gt_0p5x_err']:.1f}% stars >0.5x err -> "
            f"{'MATERIAL on some frames/stars' if materially else 'NEGLIGIBLE for most stars'}."
        )
    report = report + "\n".join(verdict_lines) + "\n\nStandalone; no production/config changes.\n"

    out = _ROOT / "tmp" / "sky_gradient_sky_plane_361_362_report.txt"
    out.write_text(report, encoding="utf-8")
    (_ROOT / "tmp" / "sky_gradient_sky_plane_361_362_result.json").write_text(
        json.dumps({"drafts": results, "psf_flags": psf_flags}, indent=2), encoding="utf-8"
    )
    print(report)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
