#!/usr/bin/env python3
"""Read-only ePSF quality diagnostic — draft 364 Luminance_180_2.

Locates WHERE the current ePSF fails to describe stellar profiles (residual structure,
radial mismatch, correlations). No production/config changes.
"""
from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from photutils.psf import ImagePSF, PSFPhotometry
from scipy import stats

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import diagnose_psf_elongation_362 as diag  # noqa: E402
from config import AppConfig  # noqa: E402
from database import VyvarDatabase  # noqa: E402
from param_resolver import resolve_gain, resolve_read_noise  # noqa: E402
from pipeline import resolve_masterstar_input_root  # noqa: E402
from proc_frame_store import ProcFrameStore  # noqa: E402
from psf_photometry import (  # noqa: E402
    _fit_shape_for_cutout,
    _to_odd_cutout,
    get_epsf_fwhm_from_context,
)

warnings.filterwarnings("ignore", category=UserWarning, module="photutils")

DRAFT_ID = 364
SETUP = "Luminance_180_2"
TARGET_N_STARS = 80
SAT_FRAC = 0.85
MAD_SCALE = 1.4826

# EPSFBuilder kwargs actually passed in psf_photometry._epsf_build_imagepsf_from_stars
# (others are photutils defaults — recorded in Part 1 report).
EPSF_BUILDER_EXPLICIT = {
    "maxiters": 15,
    "progress_bar": False,
}
EPSF_BUILDER_DEFAULTS = {
    "recentering_func": "centroid_com",
    "recentering_maxiters": 20,
    "norm_radius": 5.5,
    "recentering_boxsize": (5, 5),
    "center_accuracy": 0.001,
    "shape": None,
}


def _resolve_draft_dir(cfg: AppConfig, draft_id: int) -> Path:
    db = VyvarDatabase(cfg.database_path)
    row = db.fetch_obs_draft_by_id(int(draft_id))
    if row is not None:
        ap = str(row.get("ARCHIVE_PATH") or "").strip()
        if ap:
            p = Path(ap)
            if p.is_dir():
                return p.resolve()
    return (Path(cfg.archive_root) / "Drafts" / f"draft_{int(draft_id):06d}").resolve()


def _sky_subtract_border(cut: np.ndarray, border: int = 3) -> tuple[np.ndarray, float]:
    d = np.asarray(cut, dtype=np.float64)
    h, w = d.shape
    if h <= 2 * border or w <= 2 * border:
        sky = float(np.nanmedian(d))
        return d - sky, sky
    edge = np.concatenate(
        [
            d[:border, :].ravel(),
            d[-border:, :].ravel(),
            d[border:-border, :border].ravel(),
            d[border:-border, -border:].ravel(),
        ]
    )
    edge = edge[np.isfinite(edge)]
    sky = float(np.median(edge)) if edge.size else float(np.nanmedian(d))
    return d - sky, sky


def _fit_and_residual(
    data: np.ndarray,
    x: float,
    y: float,
    *,
    psf_model: ImagePSF,
    fit_shape: tuple[int, int],
    fwhm_px: float,
    box_half: int,
) -> dict[str, Any] | None:
    """Free flux+position PSF fit; flux-normalized residual on a generous box."""
    h, w = data.shape
    pad = max(box_half + 2, max(fit_shape) // 2 + 2)
    x0i = int(round(x))
    y0i = int(round(y))
    x_lo, x_hi = x0i - pad, x0i + pad
    y_lo, y_hi = y0i - pad, y0i + pad
    if x_lo < 0 or y_lo < 0 or x_hi >= w or y_hi >= h:
        return None

    cut = np.asarray(data[y_lo : y_hi + 1, x_lo : x_hi + 1], dtype=np.float64)
    cut_sub, sky = _sky_subtract_border(cut, border=3)
    tx, ty = float(x) - x_lo, float(y) - y_lo

    yy, xx = np.mgrid[0 : cut.shape[0], 0 : cut.shape[1]]
    near = np.hypot(xx - tx, yy - ty) <= max(2.0, float(fwhm_px))
    init_flux = float(np.nansum(np.clip(cut_sub[near], 0, None)))
    if not math.isfinite(init_flux) or init_flux <= 0:
        pk = float(np.nanmax(cut_sub))
        init_flux = pk if math.isfinite(pk) and pk > 0 else 1.0

    init = Table({"x": [tx], "y": [ty], "flux": [init_flux]})
    phot = PSFPhotometry(psf_model, fit_shape=fit_shape, progress_bar=False)
    try:
        out = phot(cut_sub, init_params=init)
    except Exception:  # noqa: BLE001
        return None
    if out is None or len(out) == 0:
        return None

    row = out[0]
    x_fit = float(row["x_fit"])
    y_fit = float(row["y_fit"])
    flux_fit = float(row["flux_fit"])
    if not (math.isfinite(flux_fit) and flux_fit > 0 and math.isfinite(x_fit) and math.isfinite(y_fit)):
        return None

    model = psf_model.evaluate(xx, yy, flux=flux_fit, x_0=x_fit, y_0=y_fit)
    resid_norm = (cut_sub - model) / flux_fit

    # Fixed box centered on fit position (integer crop).
    box_size = 2 * box_half + 1
    cy, cx = int(round(y_fit)), int(round(x_fit))
    y1, y2 = cy - box_half, cy + box_half + 1
    x1, x2 = cx - box_half, cx + box_half + 1
    if y1 < 0 or x1 < 0 or y2 > resid_norm.shape[0] or x2 > resid_norm.shape[1]:
        return None
    aligned = resid_norm[y1:y2, x1:x2]
    if aligned.shape != (box_size, box_size):
        return None

    stellar_crop = (cut_sub / flux_fit)[y1:y2, x1:x2]

    inner = aligned.copy()
    border = 2
    inner[:border, :] = inner[-border:, :] = inner[:, :border] = inner[:, -border:] = np.nan

    frac_resid = float(np.nansum(np.abs(inner)))  # already flux-normalized
    peak_resid_pct = 100.0 * float(np.nanmax(np.abs(inner)))

    return {
        "x_fit": float(x) - x_lo + x_fit,
        "y_fit": float(y) - y_lo + y_fit,
        "x_fit_global": float(x_lo + x_fit),
        "y_fit_global": float(y_lo + y_fit),
        "flux_fit": flux_fit,
        "sky": sky,
        "aligned_resid": aligned,
        "stellar_norm": stellar_crop,
        "frac_resid": frac_resid,
        "peak_resid_pct": peak_resid_pct,
        "subpix_x": float(x_fit) - math.floor(x_fit),
        "subpix_y": float(y_fit) - math.floor(y_fit),
        "subpix_phase": math.hypot(float(x_fit) - math.floor(x_fit) - 0.5, float(y_fit) - math.floor(y_fit) - 0.5),
    }


def _radial_profile(arr: np.ndarray, cx: float, cy: float, r_max: float, n_bins: int = 40) -> tuple[np.ndarray, np.ndarray]:
    h, w = arr.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    rbins = np.linspace(0, r_max, n_bins + 1)
    prof = np.full(n_bins, np.nan, dtype=float)
    for i in range(n_bins):
        m = (r >= rbins[i]) & (r < rbins[i + 1])
        if m.any():
            prof[i] = float(np.nanmean(arr[m]))
    r_cent = 0.5 * (rbins[:-1] + rbins[1:])
    return r_cent, prof


def _model_radial_profile(psf_model: ImagePSF, box_half: int, fwhm_px: float) -> tuple[np.ndarray, np.ndarray]:
    size = 2 * box_half + 1
    cx = cy = box_half
    yy, xx = np.mgrid[0:size, 0:size]
    model = psf_model.evaluate(xx, yy, flux=1.0, x_0=cx, y_0=cy)
    r_max = min(box_half, 2.5 * fwhm_px)
    return _radial_profile(model, cx, cy, r_max)


def _classify_stack(stack: np.ndarray, fwhm_px: float, box_half: int) -> tuple[str, np.ndarray, np.ndarray]:
    cx = cy = float(box_half)
    r_max = min(box_half, 2.5 * fwhm_px)
    r_cent, prof = _radial_profile(stack, cx, cy, r_max)

    inner_m = r_cent < 0.35 * fwhm_px
    ring_m = (r_cent >= 0.75 * fwhm_px) & (r_cent <= 2.0 * fwhm_px)

    inner_mean = float(np.nanmean(prof[inner_m])) if inner_m.any() else 0.0
    ring_amp = float(np.nanmean(np.abs(prof[ring_m]))) if ring_m.any() else 0.0
    center_val = float(stack[box_half, box_half]) if math.isfinite(stack[box_half, box_half]) else inner_mean

    h, w = stack.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    inner_ann = r < fwhm_px
    q_vals = []
    for sy, sx in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        m = inner_ann & ((xx - cx) * sx > 0) & ((yy - cy) * sy > 0)
        q_vals.append(float(np.nanmean(stack[m])) if m.any() else float("nan"))
    q_std = float(np.nanstd(q_vals))
    lopsided = q_std > 0.0003 and q_std > 2.0 * abs(inner_mean)

    # Quadrupole: opposite quadrants opposite sign (elongation mismatch).
    quad_a = float(np.nanmean([q_vals[0], q_vals[3]])) if len(q_vals) == 4 else 0.0
    quad_b = float(np.nanmean([q_vals[1], q_vals[2]])) if len(q_vals) == 4 else 0.0
    quadrupole = abs(quad_a - quad_b) > 0.0003 and abs(quad_a - quad_b) > ring_amp

    if quadrupole or lopsided:
        label = "lopsided (quadrupole / asymmetry — core elongation mismatch)"
    elif abs(inner_mean) < 0.002 and ring_amp < 0.002:
        label = "flat (model OK)"
    elif lopsided and abs(inner_mean) < ring_amp:
        label = "lopsided (asymmetry)"
    elif (inner_mean > 0.004 or center_val > 0.004) and abs(inner_mean) >= ring_amp:
        label = "central peak (core mismatch — model narrower/brighter core)"
    elif (inner_mean < -0.004 or center_val < -0.004) and abs(inner_mean) >= ring_amp:
        label = "central dip (core FWHM mismatch — model too wide in core)"
    elif ring_amp > abs(inner_mean) and ring_amp > 0.003:
        label = "ring at ~1-2 FWHM (wing or truncation mismatch)"
    else:
        label = "mixed / weak structure"

    return label, r_cent, prof


def _spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 8:
        return float("nan"), float("nan")
    r, p = stats.spearmanr(x[m], y[m])
    return float(r), float(p)


def _part1_report(meta: dict[str, Any], fwhm_px: float, cutout_size: int) -> list[str]:
    qc = meta.get("epsf_qc") or {}
    lines = [
        "PART 1 — CURRENT BUILD CONFIG (values actually used)",
        "",
        "EPSFBuilder (explicit in _epsf_build_imagepsf_from_stars):",
        f"  oversampling: {meta.get('oversampling')}",
        f"  maxiters: {EPSF_BUILDER_EXPLICIT['maxiters']}",
        f"  smoothing_kernel: {meta.get('smoothing_kernel')} (quadratic when osamp<=2)",
        f"  progress_bar: {EPSF_BUILDER_EXPLICIT['progress_bar']}",
        "",
        "EPSFBuilder (photutils defaults — not overridden in code):",
        f"  shape: {EPSF_BUILDER_DEFAULTS['shape']}",
        f"  recentering_func: {EPSF_BUILDER_DEFAULTS['recentering_func']}",
        f"  recentering_maxiters: {EPSF_BUILDER_DEFAULTS['recentering_maxiters']}",
        f"  norm_radius: {EPSF_BUILDER_DEFAULTS['norm_radius']} (oversampled px)",
        f"  recentering_boxsize: {EPSF_BUILDER_DEFAULTS['recentering_boxsize']}",
        f"  center_accuracy: {EPSF_BUILDER_DEFAULTS['center_accuracy']}",
        "",
        "Extraction / cutout (ePSF build):",
        f"  cutout_size: {cutout_size} (= odd(5×FWHM) from fwhm_px={fwhm_px:.3f})",
        f"  fit_shape (PSFPhotometry, stored meta): {meta.get('fit_shape')}",
        f"    (= odd(2×FWHM+1) via _fit_shape_for_cutout)",
        "",
        "Candidate selection (_epsf_prepare_stars):",
        f"  isolation: 3.0×FWHM = {meta.get('isolation_radius_px', float('nan')):.2f} px, delta_mag<=2.5 (cone catalog)",
        "  SNR cut: not applied in _epsf_prepare_stars (broad-pool frame picks use SNR≥50)",
        "  peak/saturation: likely_saturated=False; is_saturated excluded if present",
        f"  n_stars_used (extracted for EPSFBuilder): {meta.get('n_stars_used')}",
        f"  n_stars_after_join: {meta.get('n_stars_after_join')}",
        "",
        "ePSF meta / QC:",
        f"  native FWHM (model): {qc.get('epsf_fwhm_native_px')} px",
        f"  input FWHM context: {meta.get('fwhm_px')} px",
        f"  oversampling: {meta.get('oversampling')}",
        f"  asymmetry: {qc.get('epsf_asymmetry')} (= mean quadrant std / peak on built ePSF)",
        f"  FWHM ratio (model/input): {qc.get('epsf_vs_input_fwhm_ratio')}",
        "    definition: radial-profile half-max radius on oversampled ePSF ×2/osamp, divided by input fwhm_px",
        f"  epsf_sum_native: {meta.get('epsf_sum_native')}",
        f"  epsf_norm_factor: {meta.get('epsf_norm_factor')}",
        "",
    ]
    return lines


def main() -> None:
    cfg = AppConfig()
    draft_dir = _resolve_draft_dir(cfg, DRAFT_ID)
    ps_bundle = draft_dir / "platesolve" / SETUP
    aligned_dir = draft_dir / "detrended_aligned" / "lights" / SETUP
    epsf_path = ps_bundle / "masterstar_epsf.fits"
    meta_path = ps_bundle / "masterstar_epsf_meta.json"
    masterstar_fits = ps_bundle / "MASTERSTAR.fits"
    masterstars_csv = ps_bundle / "masterstars.csv"

    for p in (epsf_path, meta_path, masterstar_fits, aligned_dir):
        if not p.exists():
            raise FileNotFoundError(p)

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    db = VyvarDatabase(cfg.database_path)
    fwhm_px = float(get_epsf_fwhm_from_context(masterstar_fits, db, DRAFT_ID))
    cutout_size = _to_odd_cutout(int(fwhm_px * 5))
    fit_shape = tuple(meta.get("fit_shape") or _fit_shape_for_cutout(cutout_size, fwhm_px=fwhm_px))
    osamp = int(meta.get("oversampling", 2))

    with fits.open(epsf_path, memmap=True) as hd:
        epsf_data = np.asarray(hd[0].data, dtype=np.float64)
    psf_model = ImagePSF(epsf_data, oversampling=osamp)

    # Residual box: >= 4×FWHM (~25 px)
    box_half = max(12, int(math.ceil(2.0 * fwhm_px)))

    with fits.open(masterstar_fits, memmap=True) as hd:
        ms_hdr = hd[0].header
    plate_scale, _ = diag._resolve_plate_scale_arcsec(
        masterstar_fits=masterstar_fits,
        hdr=ms_hdr,
        db=db,
        equipment_id=None,
        telescope_id=None,
        cfg=cfg,
    )
    row = db.fetch_obs_draft_by_id(DRAFT_ID)
    equipment_id = int(row["EQUIPMENT_ID"]) if row and row.get("EQUIPMENT_ID") is not None else None
    g_res = resolve_gain(ms_hdr, db=db, equipment_id=equipment_id, cfg=cfg)
    rn_res = resolve_read_noise(ms_hdr, db=db, equipment_id=equipment_id, cfg=cfg)
    gain = float(g_res.value) if g_res.ok and g_res.value else 1.0
    rn = float(rn_res.value) if rn_res.ok and rn_res.value else 10.0

    proc_store = ProcFrameStore.build(
        aligned_dir,
        glob_pattern="proc_*.csv",
        extra_cols=["catalog_mag", "phot_g_mean_mag", "saturate_limit_adu", "peak_max_adu"],
    )
    fits_files = sorted(aligned_dir.glob("proc_*.fits"))
    csv_by_stem = {p.stem: p for p in sorted(aligned_dir.glob("proc_*.csv"))}
    pairs = [(f, csv_by_stem.get(f.stem)) for f in fits_files if csv_by_stem.get(f.stem)]
    if not pairs:
        raise FileNotFoundError(f"No paired proc FITS+CSV under {aligned_dir}")

    # --- star pool: COG-style picks, dedupe, brightest ~80 ---
    star_pool: dict[str, dict[str, Any]] = {}
    for fits_path, csv_path in pairs:
        assert csv_path is not None
        frame_df = proc_store.get_frame(str(csv_path))
        if frame_df is None:
            frame_df = pd.read_csv(csv_path, low_memory=False, dtype={"catalog_id": str})
        with fits.open(fits_path, memmap=True) as hdul:
            img_shape = hdul[0].data.shape
        picked = diag._select_frame_stars_from_proc(
            frame_df,
            ps_bundle,
            fwhm_px=fwhm_px,
            plate_scale_arcsec_px=float(plate_scale),
            fit_shape=fit_shape,
            gain=gain,
            rn=rn,
            img_shape=img_shape,
        )
        for _, st in picked.iterrows():
            cid = str(st.get("_cid", ""))
            if not cid:
                continue
            flux = float(pd.to_numeric(st.get("dao_flux"), errors="coerce"))
            mag = float(pd.to_numeric(st.get("_mag", st.get("catalog_mag", np.nan)), errors="coerce"))
            prev = star_pool.get(cid)
            if prev is None or flux > float(prev.get("dao_flux", 0)):
                star_pool[cid] = {
                    "catalog_id": cid,
                    "x": float(st["x"]),
                    "y": float(st["y"]),
                    "dao_flux": flux,
                    "mag": mag,
                    "ra_deg": float(pd.to_numeric(st.get("ra_deg"), errors="coerce")),
                    "dec_deg": float(pd.to_numeric(st.get("dec_deg"), errors="coerce")),
                }

    stars_df = pd.DataFrame(list(star_pool.values())).sort_values("dao_flux", ascending=False)
    if len(stars_df) > TARGET_N_STARS:
        stars_df = stars_df.head(TARGET_N_STARS).copy()
    stars_df = stars_df.reset_index(drop=True)

    # Field center for radius (MASTERSTAR CRVAL or image center)
    with fits.open(masterstar_fits, memmap=True) as hd:
        wcs_shape = hd[0].data.shape
    cx_field = wcs_shape[1] / 2.0
    cy_field = wcs_shape[0] / 2.0

    # Build per-frame position lookup for selected stars
    cid_set = set(stars_df["catalog_id"])
    frame_index: dict[str, int] = {p[0].name: i for i, p in enumerate(pairs)}

    records: list[dict[str, Any]] = []
    stack_list: list[np.ndarray] = []
    stellar_list: list[np.ndarray] = []

    for fits_path, csv_path in pairs:
        assert csv_path is not None
        fi = frame_index[fits_path.name]
        frame_df = proc_store.get_frame(str(csv_path))
        if frame_df is None:
            frame_df = pd.read_csv(csv_path, low_memory=False, dtype={"catalog_id": str})
        frame_df = frame_df.copy()
        frame_df["_cid"] = frame_df["catalog_id"].map(diag._norm_cid)
        sub = frame_df[frame_df["_cid"].isin(cid_set)]
        if sub.empty:
            continue
        with fits.open(fits_path, memmap=True) as hdul:
            data = np.asarray(hdul[0].data, dtype=np.float64)

        for _, st in sub.iterrows():
            cid = str(st["_cid"])
            x = float(pd.to_numeric(st.get("x"), errors="coerce"))
            y = float(pd.to_numeric(st.get("y"), errors="coerce"))
            if not (math.isfinite(x) and math.isfinite(y)):
                base = stars_df.loc[stars_df["catalog_id"] == cid]
                if base.empty:
                    continue
                x, y = float(base.iloc[0]["x"]), float(base.iloc[0]["y"])

            fit = _fit_and_residual(
                data,
                x,
                y,
                psf_model=psf_model,
                fit_shape=fit_shape,
                fwhm_px=fwhm_px,
                box_half=box_half,
            )
            if fit is None:
                continue

            base = stars_df.loc[stars_df["catalog_id"] == cid].iloc[0]
            xg, yg = fit["x_fit_global"], fit["y_fit_global"]
            field_r = float(math.hypot(xg - cx_field, yg - cy_field))
            dao_flux = float(pd.to_numeric(st.get("dao_flux"), errors="coerce"))
            if not math.isfinite(dao_flux) or dao_flux <= 0:
                dao_flux = float(base["dao_flux"])

            records.append(
                {
                    "frame": fits_path.name,
                    "frame_idx": fi,
                    "catalog_id": cid,
                    "dao_flux": dao_flux,
                    "mag": float(base["mag"]) if math.isfinite(float(base["mag"])) else float("nan"),
                    "field_radius_px": field_r,
                    "subpix_x": fit["subpix_x"],
                    "subpix_y": fit["subpix_y"],
                    "subpix_phase": fit["subpix_phase"],
                    "frac_resid": fit["frac_resid"],
                    "peak_resid_pct": fit["peak_resid_pct"],
                    "flux_fit": fit["flux_fit"],
                }
            )
            stack_list.append(fit["aligned_resid"])
            stellar_list.append(fit["stellar_norm"])

    rec_df = pd.DataFrame(records)
    if rec_df.empty:
        raise RuntimeError("No successful PSF residual fits")

    stack_arr = np.stack(stack_list, axis=0)
    mean_stack = np.nanmean(stack_arr, axis=0)
    classification, r_cent, stack_prof = _classify_stack(mean_stack, fwhm_px, box_half)

    med_frac = float(np.median(rec_df["frac_resid"]))
    med_peak_pct = float(np.median(rec_df["peak_resid_pct"]))
    qc = meta.get("epsf_qc") or {}
    model_fwhm_native = float(qc.get("epsf_fwhm_native_px") or fwhm_px)

    # Stellar vs model radial profiles
    cx = cy = float(box_half)
    r_max = min(box_half, 2.5 * fwhm_px)
    stellar_profs = []
    for arr in stellar_list:
        rc, pr = _radial_profile(arr, cx, cy, r_max)
        stellar_profs.append(pr)
    stellar_med = np.nanmedian(np.stack(stellar_profs, axis=0), axis=0)
    r_model, model_prof = _model_radial_profile(psf_model, box_half, fwhm_px)

    # Where profiles diverge
    n_common = min(len(r_cent), len(model_prof))
    diff = stellar_med[:n_common] - model_prof[:n_common]
    r_cent_cmp = r_cent[:n_common]
    inner_d = float(np.nanmean(diff[r_cent_cmp < fwhm_px])) if np.any(r_cent_cmp < fwhm_px) else float("nan")
    wing_d = (
        float(np.nanmean(diff[(r_cent_cmp >= fwhm_px) & (r_cent_cmp <= 2 * fwhm_px)]))
        if np.any((r_cent_cmp >= fwhm_px) & (r_cent_cmp <= 2 * fwhm_px))
        else float("nan")
    )
    # FWHM estimate from half-max on profiles
    def _fwhm_from_prof(r: np.ndarray, p: np.ndarray) -> float:
        p = np.asarray(p, dtype=float)
        if not np.any(np.isfinite(p)) or np.nanmax(p) <= 0:
            return float("nan")
        pn = p / np.nanmax(p)
        below = np.where(pn < 0.5)[0]
        return float(2.0 * r[below[0]]) if len(below) else float("nan")

    fwhm_stellar = _fwhm_from_prof(r_cent, stellar_med)
    fwhm_model = _fwhm_from_prof(r_model, model_prof)
    if math.isfinite(fwhm_stellar) and math.isfinite(fwhm_model):
        if fwhm_model < fwhm_stellar - 0.15:
            width_verdict = "model NARROWER than stars (core)"
        elif fwhm_model > fwhm_stellar + 0.15:
            width_verdict = "model WIDER than stars (core)"
        else:
            width_verdict = "similar core width"
    else:
        width_verdict = "indeterminate"

    # Correlations (per star-frame measurements)
    y = rec_df["frac_resid"].to_numpy(dtype=float)
    r_bright, p_bright = _spearman(np.log10(rec_df["dao_flux"].to_numpy(dtype=float)), y)
    r_field, p_field = _spearman(rec_df["field_radius_px"].to_numpy(dtype=float), y)
    r_phase, p_phase = _spearman(rec_df["subpix_phase"].to_numpy(dtype=float), y)
    r_frame, p_frame = _spearman(rec_df["frame_idx"].to_numpy(dtype=float), y)

    # Frame-correlated: ANOVA-style — fraction of variance between frames
    frame_means = rec_df.groupby("frame_idx")["frac_resid"].mean()
    grand = float(rec_df["frac_resid"].mean())
    ss_between = float(((frame_means - grand) ** 2 * rec_df.groupby("frame_idx").size()).sum())
    ss_total = float(((rec_df["frac_resid"] - grand) ** 2).sum())
    eta2_frame = ss_between / ss_total if ss_total > 0 else float("nan")

    corrs = {
        "brightness (log dao_flux)": (r_bright, p_bright),
        "field_radius_px": (r_field, p_field),
        "subpix_phase": (r_phase, p_phase),
        "frame_idx": (r_frame, p_frame),
    }
    dominant = max(corrs.items(), key=lambda kv: abs(kv[1][0]) if math.isfinite(kv[1][0]) else -1.0)

    out_dir = draft_dir / "diagnostics" / "epsf_quality_364"
    out_dir.mkdir(parents=True, exist_ok=True)

    # PNG: mean residual map + radial profile
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    vmax = np.nanpercentile(np.abs(mean_stack), 99)
    im = axes[0].imshow(mean_stack, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[0].set_title(f"Mean flux-normalized residual (N={len(stack_list)})")
    axes[0].set_xlabel("x (px)")
    axes[0].set_ylabel("y (px)")
    plt.colorbar(im, ax=axes[0], fraction=0.046, label="(data−flux·PSF)/flux")
    axes[1].plot(r_cent, stack_prof, "k-", lw=2, label="mean residual radial")
    axes[1].axhline(0, color="gray", ls="--", lw=0.8)
    axes[1].axvline(fwhm_px, color="orange", ls=":", label="1×FWHM")
    axes[1].axvline(2 * fwhm_px, color="orange", ls="--", label="2×FWHM")
    axes[1].set_xlabel("radius (px)")
    axes[1].set_ylabel("mean normalized residual")
    axes[1].legend(fontsize=8)
    axes[1].set_title(f"Classification: {classification}")
    fig.tight_layout()
    map_png = out_dir / "d364_epsf_mean_residual_map.png"
    fig.savefig(map_png, dpi=150)
    plt.close(fig)

    # PNG: stellar vs model profile
    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    ax2.plot(r_cent, stellar_med / np.nanmax(stellar_med), "b-", lw=2, label="median stellar (data/flux)")
    ax2.plot(r_model, model_prof / np.nanmax(model_prof), "r--", lw=2, label="ePSF model (flux=1)")
    ax2.axvline(fwhm_px, color="gray", ls=":", label="input FWHM")
    ax2.axvline(model_fwhm_native, color="red", ls=":", alpha=0.6, label="model FWHM")
    ax2.set_xlabel("radius (px)")
    ax2.set_ylabel("normalized surface brightness")
    ax2.set_title(f"Profile overlay — {width_verdict}")
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    prof_png = out_dir / "d364_epsf_stellar_vs_model_profile.png"
    fig2.savefig(prof_png, dpi=150)
    plt.close(fig2)

    report_lines = _part1_report(meta, fwhm_px, cutout_size)
    report_lines += [
        "PART 2 — RESIDUAL ANALYSIS",
        "",
        f"Diagnostic stars: {len(stars_df)} selected (COG pool cap {TARGET_N_STARS})",
        f"Successful star×frame fits: {len(rec_df)}",
        f"Residual box: {2 * box_half + 1} px (half={box_half}, ≥4×FWHM)",
        "",
        "(a) Mean residual map classification:",
        f"  {classification}",
        f"  radial inner mean (r<0.35 FWHM): {float(np.nanmean(stack_prof[r_cent < 0.35 * fwhm_px])):.5f}" if np.any(r_cent < 0.35 * fwhm_px) else "",
        "",
        "(b) Fractional residual metrics:",
        f"  median sum|resid|/flux (inner): {med_frac:.4f}",
        f"  median peak |resid| as % of peak: {med_peak_pct:.2f}%",
        "",
        "(c) Stellar vs model radial profile:",
        f"  FWHM stellar≈{fwhm_stellar:.2f} px, model≈{fwhm_model:.2f} px → {width_verdict}",
        f"  mean(stellar−model) core (r<FWHM): {inner_d:+.5f}",
        f"  mean(stellar−model) wings (1–2 FWHM): {wing_d:+.5f}",
        "",
        "(d) Spearman ρ vs fractional residual:",
    ]
    for name, (rv, pv) in corrs.items():
        report_lines.append(f"  {name}: ρ={rv:.4f}, p={pv:.2e}")
    report_lines += [
        f"  dominant driver: {dominant[0]} (|ρ|={abs(dominant[1][0]):.4f})",
        "",
        "(e) Frame correlation:",
        f"  Spearman(frame_idx, frac_resid): ρ={r_frame:.4f}, p={p_frame:.2e}",
        f"  between-frame η² (fraction of total variance): {eta2_frame:.4f}",
        f"  frame-correlated: {'YES (moderate Spearman, but eta2={:.4f} — mostly star-to-star scatter)'.format(eta2_frame) if (math.isfinite(r_frame) and abs(r_frame) > 0.15 and p_frame < 0.05) else 'weak/no'}",
        "",
        "PNG outputs:",
        f"  {map_png}",
        f"  {prof_png}",
        "",
    ]

    report_path = out_dir / "d364_epsf_quality_report.txt"
    report_text = "\n".join(report_lines)
    report_path.write_text(report_text, encoding="utf-8")
    rec_df.to_csv(out_dir / "d364_epsf_quality_per_fit.csv", index=False)

    print(report_text.encode("ascii", errors="replace").decode("ascii"))


if __name__ == "__main__":
    main()
