#!/usr/bin/env python3
"""Fixed vs free-position forced photometry — bright isolated G12-16 (read-only).

Separates catalog fixed-position confound from real PSF disadvantage on draft 364.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import math
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from photutils.aperture import CircularAnnulus, CircularAperture, aperture_photometry
from photutils.psf import ImagePSF, PSFPhotometry

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import AppConfig  # noqa: E402
from photometry_core import (  # noqa: E402
    _aperture_radius_from_snr_table,
    _catalog_only_fixed_aperture_flux,
    load_snr_aperture_table_from_draft_dir,
)

warnings.filterwarnings("ignore", category=UserWarning, module="photutils")
logging.getLogger("astropy").setLevel(logging.ERROR)

DRAFT_ID = 364
SETUP = "Luminance_180_2"
MAG_LO = 12.0
MAG_HI = 16.0
MAD_SCALE = 1.4826
MIN_FRAMES = 5
FREE_MAX_SHIFT_PX = 3.0
SAT_FRAC = 0.85
ZP_CAL_MAG_MAX = 13.5

_fp_path = _ROOT / "scripts" / "forced_photometry_pal7.py"
_spec = importlib.util.spec_from_file_location("fp_pal7", _fp_path)
_fp = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_fp)


def _robust_rms_mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    med = float(np.median(x))
    return float(MAD_SCALE * np.median(np.abs(x - med)))


def _psf_cutout(data: np.ndarray, x: float, y: float, fit_shape: tuple[int, int]) -> tuple[np.ndarray, float, float] | None:
    h, w = data.shape
    fh, fw = int(fit_shape[0]), int(fit_shape[1])
    half_y, half_x = fh // 2, fw // 2
    x0, y0 = int(round(x)), int(round(y))
    y1, y2 = max(0, y0 - half_y), min(h, y0 + half_y + 1)
    x1, x2 = max(0, x0 - half_x), min(w, x0 + half_x + 1)
    cut = np.asarray(data[y1:y2, x1:x2], dtype=np.float64)
    if cut.size < 9:
        return None
    border = np.ones(cut.shape, dtype=bool)
    if cut.shape[0] > 4 and cut.shape[1] > 4:
        border[2:-2, 2:-2] = False
    sky = float(np.median(cut[border])) if border.any() else float(np.median(cut))
    return cut - sky, float(x) - x1, float(y) - y1


def _init_flux(cut_sub: np.ndarray, tx: float, ty: float, fwhm_px: float) -> float:
    yy, xx = np.mgrid[0 : cut_sub.shape[0], 0 : cut_sub.shape[1]]
    near = np.hypot(xx - tx, yy - ty) <= max(2.0, float(fwhm_px))
    tflux = float(np.nansum(cut_sub[near].clip(min=0)))
    if not math.isfinite(tflux) or tflux <= 0:
        tflux = max(1.0, float(np.nanmax(cut_sub)) if math.isfinite(float(np.nanmax(cut_sub))) else 1.0)
    return tflux


def _psf_fixed(cut_sub: np.ndarray, tx: float, ty: float, tflux: float, psf_model: ImagePSF, fit_shape: tuple[int, int]) -> float:
    try:
        phot = PSFPhotometry(psf_model, fit_shape, progress_bar=False)
        init = Table([[tx], [ty], [tflux]], names=("x_0", "y_0", "flux_0"))
        try:
            phot.set_fixed_params(["x_0", "y_0"])
        except Exception:  # noqa: BLE001
            pass
        res = phot(cut_sub, init_params=init)
        flux = float(res["flux_fit"][0])
        return flux if math.isfinite(flux) and flux > 0 else float("nan")
    except Exception:  # noqa: BLE001
        return float("nan")


def _psf_free(
    cut_sub: np.ndarray,
    tx: float,
    ty: float,
    tflux: float,
    psf_model: ImagePSF,
    fit_shape: tuple[int, int],
) -> tuple[float, float, float]:
    """Return (flux, dx_image_px, dy_image_px) relative to catalog init in cutout-local then global."""
    try:
        phot = PSFPhotometry(psf_model, fit_shape, progress_bar=False)
        init = Table([[tx], [ty], [tflux]], names=("x_0", "y_0", "flux_0"))
        res = phot(cut_sub, init_params=init)
        xf = float(res["x_fit"][0])
        yf = float(res["y_fit"][0])
        dx = xf - tx
        dy = yf - ty
        if math.hypot(dx, dy) > FREE_MAX_SHIFT_PX:
            return float("nan"), dx, dy
        flux = float(res["flux_fit"][0])
        if not math.isfinite(flux) or flux <= 0:
            return float("nan"), dx, dy
        return flux, dx, dy
    except Exception:  # noqa: BLE001
        return float("nan"), float("nan"), float("nan")


def _star_rms_table(
    df: pd.DataFrame,
    *,
    psf_col: str,
    resid_col: str | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cid, grp in df.groupby("catalog_id", sort=False):
        mag_val = float(pd.to_numeric(grp["mag"], errors="coerce").median())
        apt = grp["mag_aper"].to_numpy(dtype=float)
        psf = grp[psf_col].to_numpy(dtype=float)
        ok = np.isfinite(apt) & np.isfinite(psf)
        if ok.sum() < MIN_FRAMES:
            continue
        apt = apt[ok]
        psf = psf[ok]
        offset = float(np.median(apt - psf))
        psf_norm = psf + offset
        apt_res = apt - float(np.median(apt))
        psf_res = psf_norm - float(np.median(psf_norm))
        row: dict[str, Any] = {
            "catalog_id": cid,
            "catalog_mag": mag_val,
            "n_frames": int(ok.sum()),
            "rms_aperture": _robust_rms_mad(apt_res),
            "rms_psf": _robust_rms_mad(psf_res),
        }
        row["ratio_psf_aper"] = row["rms_psf"] / row["rms_aperture"] if row["rms_aperture"] > 0 else float("nan")
        if resid_col and resid_col in grp.columns:
            r = pd.to_numeric(grp.iloc[np.where(ok)[0]][resid_col], errors="coerce").to_numpy(dtype=float)
            r = r[np.isfinite(r)]
            row["pos_residual_rms_px"] = float(np.sqrt(np.mean(r**2))) if r.size else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def _bin_summary(star_df: pd.DataFrame, *, psf_mode: str) -> list[dict[str, Any]]:
    bins = [
        ("12-13", 12.0, 13.0),
        ("13-14", 13.0, 14.0),
        ("14-15", 14.0, 15.0),
        ("15-16", 15.0, 16.0),
        ("12-16", 12.0, 16.0),
    ]
    out: list[dict[str, Any]] = []
    for label, lo, hi in bins:
        sub = star_df[(star_df["catalog_mag"] > lo) & (star_df["catalog_mag"] <= hi)]
        out.append(
            {
                "mag_bin": label,
                "psf_mode": psf_mode,
                "N": int(len(sub)),
                "median_rms_aperture": float(sub["rms_aperture"].median()) if len(sub) else float("nan"),
                "median_rms_psf": float(sub["rms_psf"].median()) if len(sub) else float("nan"),
                "median_ratio_psf_aper": float(sub["ratio_psf_aper"].median()) if len(sub) else float("nan"),
                "median_pos_residual_rms_px": float(sub["pos_residual_rms_px"].median())
                if len(sub) and "pos_residual_rms_px" in sub.columns
                else float("nan"),
            }
        )
    return out


def run() -> dict[str, Any]:
    cfg = AppConfig()
    if cfg.psf_photometry_enabled:
        raise RuntimeError("psf_photometry_enabled must remain false")

    draft_dir = Path(cfg.archive_root) / "Drafts" / f"draft_{DRAFT_ID:06d}"
    ps_dir = draft_dir / "platesolve" / SETUP
    aligned = draft_dir / "detrended_aligned" / "lights" / SETUP
    ms_fits = ps_dir / "MASTERSTAR.fits"
    epsf_path = ps_dir / "masterstar_epsf.fits"
    meta_path = ps_dir / "masterstar_epsf_meta.json"

    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    fwhm_px = float(meta.get("fwhm_px", 6.22))
    osamp = int(meta.get("oversampling", 2))
    fit_shape = tuple(meta.get("fit_shape", [15, 15]))
    plate_scale = float(meta.get("plate_scale_arcsec_px", 0.389))

    snr_table = load_snr_aperture_table_from_draft_dir(draft_dir) or {}
    fwhm_snr = float(snr_table.get("fwhm_px", fwhm_px))
    ann_in = float(cfg.annulus_inner_fwhm) * fwhm_snr
    ann_out = float(cfg.annulus_outer_fwhm) * fwhm_snr
    sat_limit = 60000.0

    with fits.open(ms_fits, memmap=True) as hd:
        ms_wcs = WCS(hd[0].header)
        naxis1 = int(hd[0].header.get("NAXIS1", hd[0].data.shape[1]))
        naxis2 = int(hd[0].header.get("NAXIS2", hd[0].data.shape[0]))

    cone = _fp._load_deep_cone(ps_dir, ms_fits)
    ra = cone["ra_deg"].to_numpy(dtype=float)
    de = cone["dec_deg"].to_numpy(dtype=float)
    xp, yp = ms_wcs.all_world2pix(np.column_stack([ra, de]), 0).T
    cone = cone.assign(x=xp, y=yp)
    margin = 2.0 * fwhm_px
    mags = pd.to_numeric(cone["mag"], errors="coerce").to_numpy(dtype=float)
    cone = cone.loc[
        (cone["x"] >= margin)
        & (cone["x"] < naxis1 - margin)
        & (cone["y"] >= margin)
        & (cone["y"] < naxis2 - margin)
        & (mags > MAG_LO)
        & (mags <= MAG_HI)
    ].copy().reset_index(drop=True)
    mags = pd.to_numeric(cone["mag"], errors="coerce").to_numpy(dtype=float)
    cone["crowded"] = _fp._cone_crowding_kdtree(cone, fwhm_px=fwhm_px, plate_scale=plate_scale)
    cone = cone.loc[~cone["crowded"]].reset_index(drop=True)
    mags = pd.to_numeric(cone["mag"], errors="coerce").to_numpy(dtype=float)

    r_ap = np.array(
        [
            _aperture_radius_from_snr_table(
                m if math.isfinite(m) else 99.0,
                snr_table,
                aperture_fwhm_factor=float(cfg.aperture_fwhm_factor),
                fwhm_px=fwhm_snr,
            )
            for m in mags
        ],
        dtype=float,
    )

    psf_data = np.asarray(fits.getdata(epsf_path), dtype=np.float64)
    psf_model = ImagePSF(psf_data, oversampling=osamp)
    fit_shape_t = (int(fit_shape[0]), int(fit_shape[1]))

    x_all = cone["x"].to_numpy(dtype=float)
    y_all = cone["y"].to_numpy(dtype=float)
    cid_all = cone["catalog_id"].astype(str).to_numpy()
    cal_mask = mags <= ZP_CAL_MAG_MAX

    frame_files = sorted(aligned.glob("proc_*.fits"))
    records: list[dict[str, Any]] = []

    for fpath in frame_files:
        with fits.open(fpath, memmap=True) as hd:
            data = np.asarray(hd[0].data, dtype=np.float64)

        a_flux = np.full(len(cone), np.nan)
        p_fix = np.full(len(cone), np.nan)
        p_free = np.full(len(cone), np.nan)
        resid_r = np.full(len(cone), np.nan)

        for j in range(len(cone)):
            x, y = float(x_all[j]), float(y_all[j])
            rap = float(r_ap[j])
            a_flux[j], sky, peak = _catalog_only_fixed_aperture_flux(data, x, y, rap, ann_in, ann_out)
            if peak > SAT_FRAC * sat_limit:
                a_flux[j] = float("nan")
                continue
            cut_info = _psf_cutout(data, x, y, fit_shape_t)
            if cut_info is None:
                continue
            cut_sub, tx, ty = cut_info
            tflux = _init_flux(cut_sub, tx, ty, fwhm_px)
            p_fix[j] = _psf_fixed(cut_sub, tx, ty, tflux, psf_model, fit_shape_t)
            pf, dx, dy = _psf_free(cut_sub, tx, ty, tflux, psf_model, fit_shape_t)
            p_free[j] = pf
            if math.isfinite(dx) and math.isfinite(dy):
                resid_r[j] = math.hypot(dx, dy)

        zp_aper = float("nan")
        ok_a = cal_mask & np.isfinite(a_flux) & (a_flux > 0)
        if ok_a.any():
            zp_aper = float(np.median(mags[ok_a] + 2.5 * np.log10(a_flux[ok_a])))

        for j in range(len(cone)):
            mag_aper = mag_fix = mag_free = float("nan")
            if math.isfinite(zp_aper) and math.isfinite(a_flux[j]) and a_flux[j] > 0:
                mag_aper = zp_aper - 2.5 * math.log10(a_flux[j])
            if math.isfinite(zp_aper) and math.isfinite(p_fix[j]) and p_fix[j] > 0:
                mag_fix = zp_aper - 2.5 * math.log10(p_fix[j])
            if math.isfinite(zp_aper) and math.isfinite(p_free[j]) and p_free[j] > 0:
                mag_free = zp_aper - 2.5 * math.log10(p_free[j])
            records.append(
                {
                    "frame": fpath.name,
                    "catalog_id": cid_all[j],
                    "mag": float(mags[j]),
                    "mag_aper": mag_aper,
                    "mag_psf_fixed": mag_fix,
                    "mag_psf_free": mag_free,
                    "pos_residual_px": float(resid_r[j]) if math.isfinite(resid_r[j]) else float("nan"),
                }
            )

    all_df = pd.DataFrame(records)
    df_fix = all_df.rename(columns={"mag_psf_fixed": "mag_psf"})
    df_free = all_df.rename(columns={"mag_psf_free": "mag_psf"})

    star_fix = _star_rms_table(df_fix, psf_col="mag_psf")
    star_free = _star_rms_table(df_free, psf_col="mag_psf", resid_col="pos_residual_px")

    resid_by_cid = star_free.set_index("catalog_id")["pos_residual_rms_px"].to_dict()
    star_fix["pos_residual_rms_px"] = star_fix["catalog_id"].map(resid_by_cid)

    bin_fix = _bin_summary(star_fix, psf_mode="fixed")
    bin_free = _bin_summary(star_free, psf_mode="free")
    summary_df = pd.DataFrame(bin_fix + bin_free)

    out_dir = draft_dir / "diagnostics" / "forced_photometry_pal7"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "d364_fixed_vs_free_position_g12_16.csv"
    stars_csv = out_dir / "d364_fixed_vs_free_position_per_star.csv"
    summary_df.to_csv(summary_csv, index=False)
    star_fix_out = star_fix.copy()
    star_fix_out["psf_mode"] = "fixed"
    star_free_out = star_free.copy()
    star_free_out["psf_mode"] = "free"
    pd.concat([star_fix_out, star_free_out], ignore_index=True).to_csv(stars_csv, index=False)

    row_1216_fix = next((r for r in bin_fix if r["mag_bin"] == "12-16"), {})
    row_1216_free = next((r for r in bin_free if r["mag_bin"] == "12-16"), {})
    ratio_fix = float(row_1216_fix.get("median_ratio_psf_aper", float("nan")))
    ratio_free = float(row_1216_free.get("median_ratio_psf_aper", float("nan")))
    med_resid = float(row_1216_free.get("median_pos_residual_rms_px", float("nan")))

    if math.isfinite(ratio_free) and ratio_free <= 1.0:
        verdict = (
            "fixed-position confound CONFIRMED — FREE psf/aper drops to <=1 "
            f"({ratio_free:.2f} vs FIXED {ratio_fix:.2f}); bright-end forced penalty is a method artifact"
        )
    elif math.isfinite(ratio_free) and ratio_free > 1.0:
        verdict = (
            "PSF does not beat aperture on full isolated G12-16 even with FREE positions "
            f"(FREE psf/aper={ratio_free:.2f}, FIXED={ratio_fix:.2f}); DAO win likely selection on detected subset"
        )
    else:
        verdict = "inconclusive — insufficient valid stars"

    return {
        "draft_id": DRAFT_ID,
        "setup": SETUP,
        "n_isolated_g12_16": int(len(cone)),
        "n_stars_valid_ge5_fixed": int(len(star_fix)),
        "n_stars_valid_ge5_free": int(len(star_free)),
        "summary_csv": str(summary_csv),
        "per_star_csv": str(stars_csv),
        "bin_table": summary_df.to_dict(orient="records"),
        "g12_16_fixed_ratio": ratio_fix,
        "g12_16_free_ratio": ratio_free,
        "g12_16_median_pos_residual_rms_px": med_resid,
        "verdict": verdict,
        "psf_flag_in_config": bool(cfg.psf_photometry_enabled),
    }


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    result = run()
    (_ROOT / "pilot_palomar7_fixed_vs_free_result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
