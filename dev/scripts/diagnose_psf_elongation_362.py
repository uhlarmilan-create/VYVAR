#!/usr/bin/env python3
"""Per-frame PSF elongation diagnostic for draft 362 (read-only).

Tests whether frame-variable tracking smear (OAT mount) explains PSF vs aperture
scatter despite a nearly round stacked ePSF.

Usage:
    python scripts/diagnose_psf_elongation_362.py
    python scripts/diagnose_psf_elongation_362.py --draft 362 --out-dir PATH
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.modeling import fitting, models
from scipy import stats

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401  (repo layout: src_py + dev on sys.path)
_ROOT = _bootstrap.REPO_ROOT
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from comp_selection_per_target import _angular_distance_deg_vectorized  # noqa: E402
from config import AppConfig  # noqa: E402
from database import VyvarDatabase  # noqa: E402
from param_resolver import resolve_binning, resolve_focal_mm, resolve_pixel_um  # noqa: E402
from pipeline import resolve_masterstar_input_root  # noqa: E402
from proc_frame_store import ProcFrameStore  # noqa: E402
from psf_photometry import (  # noqa: E402
    _load_cone_catalog,
    _read_plate_scale_arcsec_px_from_fits,
    get_epsf_fwhm_from_context,
)
from utils import effective_binned_pixel_pitch_um  # noqa: E402

DRAFT_ID_DEFAULT = 362
ISOLATION_FWHM = 6.0
ISOLATION_DELTA_MAG = 2.5
SNR_MIN = 50.0
SAT_FRAC = 0.85
MAX_STARS_PER_FRAME = 60
MAD_SCALE = 1.4826
FWHM_FROM_STD = 2.0 * math.sqrt(2.0 * math.log(2.0))
CHI2_OUTLIER_SIGMA = 3.0


def _repo_root() -> Path:
    return _ROOT


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


def _find_setup(draft_dir: Path) -> Path:
    base = draft_dir / "detrended_aligned" / "lights"
    if not base.is_dir():
        raise FileNotFoundError(f"Missing detrended_aligned/lights under {draft_dir}")
    for cand in sorted(base.iterdir()):
        if cand.is_dir() and any(cand.glob("proc_*.csv")) and any(cand.glob("proc_*.fits")):
            return cand
    raise FileNotFoundError(f"No setup with paired proc_*.csv and proc_*.fits under {base}")


def _resolve_platesolve_bundle(draft_dir: Path, setup_name: str) -> Path:
    ps = draft_dir / "platesolve" / setup_name
    if not (ps / "MASTERSTAR.fits").is_file():
        raise FileNotFoundError(f"MASTERSTAR.fits not found under {ps}")
    return ps


def _resolve_plate_scale_arcsec(
    *,
    masterstar_fits: Path,
    hdr: fits.Header,
    db: VyvarDatabase,
    equipment_id: int | None,
    telescope_id: int | None,
    cfg: AppConfig,
) -> tuple[float, str]:
    """Plate scale (arcsec/px) — WCS-first, then param_resolver optics chain."""
    wcs_ps = _read_plate_scale_arcsec_px_from_fits(masterstar_fits)
    if wcs_ps is not None and math.isfinite(wcs_ps) and 0.1 <= wcs_ps <= 30.0:
        return float(wcs_ps), "WCS/CD (MASTERSTAR)"

    focal = resolve_focal_mm(hdr, db=db, equipment_id=equipment_id, telescope_id=telescope_id, cfg=cfg)
    pixel = resolve_pixel_um(hdr, db=db, equipment_id=equipment_id, cfg=cfg)
    bin_res = resolve_binning(hdr, cfg=cfg)
    if focal.ok and pixel.ok and focal.value and pixel.value:
        b = max(1, int(round(float(bin_res.value)))) if bin_res.ok and bin_res.value else 1
        eff_um = effective_binned_pixel_pitch_um(base_pixel_um_1x1=float(pixel.value), binning=b)
        if eff_um > 0 and float(focal.value) > 0:
            ps = 206.264806 * (eff_um * 1e-3) / float(focal.value)
            if math.isfinite(ps) and 0.1 <= ps <= 30.0:
                return float(ps), f"param_resolver(focal={focal.source}, pixel={pixel.source}, bin={bin_res.source})"

    raise ValueError(
        "Could not resolve plate scale: no usable WCS/CD on MASTERSTAR and param_resolver optics chain failed."
    )


def _norm_cid(raw: Any) -> str:
    if raw is None:
        return ""
    s = str(raw).strip()
    if not s or s.lower() in ("nan", "none"):
        return ""
    try:
        from gaia_catalog_id import normalize_gaia_source_id  # noqa: PLC0415

        return normalize_gaia_source_id(raw)
    except Exception:  # noqa: BLE001
        return s


def _scalar_false(v: Any) -> bool:
    if v is False:
        return True
    if v is True or v is None:
        return False
    s = str(v).strip().lower()
    return s in ("false", "0", "no")


def _scalar_true(v: Any) -> bool:
    if v is True:
        return True
    if v is False or v is None:
        return False
    s = str(v).strip().lower()
    return s in ("true", "1", "yes", "y")


def _apply_cone_isolation_mask(
    ra: np.ndarray,
    dec: np.ndarray,
    mag: np.ndarray,
    cone_dir: Path,
    *,
    fwhm_px: float,
    plate_scale_arcsec_px: float,
) -> np.ndarray:
    iso_r_px = float(ISOLATION_FWHM) * float(fwhm_px)
    radius_deg = iso_r_px * float(plate_scale_arcsec_px) / 3600.0
    self_deg = 0.5 * float(plate_scale_arcsec_px) / 3600.0
    cone = _load_cone_catalog(cone_dir)
    if cone is None:
        raise FileNotFoundError(f"Missing field_catalog_cone.csv under {cone_dir}")
    cone_ra, cone_dec, cone_mag = cone

    keep = np.ones(len(ra), dtype=bool)
    for j in range(len(ra)):
        ra_i = float(ra[j])
        de_i = float(dec[j])
        if not (math.isfinite(ra_i) and math.isfinite(de_i)):
            keep[j] = False
            continue
        cosd = max(math.cos(math.radians(de_i)), 0.2)
        box = (np.abs(cone_dec - de_i) <= radius_deg * 1.5) & (
            np.abs(cone_ra - ra_i) <= radius_deg * 1.5 / cosd
        )
        if not box.any():
            continue
        d_deg = _angular_distance_deg_vectorized(ra_i, de_i, cone_ra[box], cone_dec[box])
        m_box = cone_mag[box]
        cm = float(mag[j]) if math.isfinite(mag[j]) else float("nan")
        near = (d_deg > self_deg) & (d_deg <= radius_deg)
        if math.isfinite(cm):
            contaminating = near & ((m_box - cm) <= ISOLATION_DELTA_MAG)
        else:
            contaminating = near
        if bool(np.any(contaminating)):
            keep[j] = False
    return keep


def _select_frame_stars_from_proc(
    frame_df: pd.DataFrame,
    cone_dir: Path,
    *,
    fwhm_px: float,
    plate_scale_arcsec_px: float,
    fit_shape: tuple[int, int],
    gain: float,
    rn: float,
    img_shape: tuple[int, int],
) -> pd.DataFrame:
    """COG-style star pick on the per-frame proc catalog (Phase-2A positions)."""
    if frame_df.empty:
        return pd.DataFrame()

    df = frame_df.copy()
    df["_cid"] = df["catalog_id"].map(_norm_cid)
    df = df[df["_cid"] != ""]
    for col in ("x", "y", "ra_deg", "dec_deg", "dao_flux", "noise_floor_adu", "aperture_r_px"):
        if col not in df.columns:
            return pd.DataFrame()

    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["ra_deg"] = pd.to_numeric(df["ra_deg"], errors="coerce")
    df["dec_deg"] = pd.to_numeric(df["dec_deg"], errors="coerce")
    df["dao_flux"] = pd.to_numeric(df["dao_flux"], errors="coerce")
    df["noise_floor_adu"] = pd.to_numeric(df["noise_floor_adu"], errors="coerce")
    df["aperture_r_px"] = pd.to_numeric(df["aperture_r_px"], errors="coerce")

    mag_col = "catalog_mag" if "catalog_mag" in df.columns else ("mag" if "mag" in df.columns else None)
    if mag_col:
        df["_mag"] = pd.to_numeric(df[mag_col], errors="coerce")
    else:
        df["_mag"] = np.nan

    if "photometry_ok" in df.columns:
        df = df[df["photometry_ok"].map(_scalar_true)]
    if "vsx_known_variable" in df.columns:
        df = df[df["vsx_known_variable"].map(_scalar_false)]
    if "likely_saturated" in df.columns:
        df = df[~df["likely_saturated"].fillna(False).astype(bool)]
    if "is_saturated" in df.columns:
        df = df[~df["is_saturated"].fillna(False).astype(bool)]

    iso = _apply_cone_isolation_mask(
        df["ra_deg"].to_numpy(dtype=float),
        df["dec_deg"].to_numpy(dtype=float),
        df["_mag"].to_numpy(dtype=float),
        cone_dir,
        fwhm_px=fwhm_px,
        plate_scale_arcsec_px=plate_scale_arcsec_px,
    )
    df = df.loc[iso]

    peak = pd.to_numeric(df["peak_max_adu"], errors="coerce") if "peak_max_adu" in df.columns else pd.Series(np.nan, index=df.index)
    if "saturate_limit_adu" in df.columns:
        sat = pd.to_numeric(df["saturate_limit_adu"], errors="coerce")
    else:
        sat = pd.Series(np.nan, index=df.index)
    if sat.isna().all() and "saturate_limit_adu_85pct" in df.columns:
        sat = pd.to_numeric(df["saturate_limit_adu_85pct"], errors="coerce")
        unsat = ~(peak.notna() & sat.notna() & (peak > sat))
    else:
        unsat = ~(peak.notna() & sat.notna() & (peak > SAT_FRAC * sat))

    flux = df["dao_flux"]
    sky = df["noise_floor_adu"]
    rap = df["aperture_r_px"]
    snr = np.array(
        [_compute_snr(float(f), float(s), float(r), gain, rn) for f, s, r in zip(flux, sky, rap, strict=True)],
        dtype=float,
    )
    snr_ok = df.get("snr50_ok")
    if snr_ok is not None:
        snr_mask = snr_ok.fillna(False).astype(bool) | (snr >= SNR_MIN)
    else:
        snr_mask = snr >= SNR_MIN

    margin = max(fit_shape) + 2.0
    h, w = img_shape
    x, y = df["x"], df["y"]
    in_bounds = (x > margin) & (x < (w - margin)) & (y > margin) & (y < (h - margin))

    sel = unsat & snr_mask & in_bounds & flux.notna() & (flux > 0)
    picked = df.loc[sel].copy()
    picked = picked.sort_values("dao_flux", ascending=False)
    if MAX_STARS_PER_FRAME > 0 and len(picked) > MAX_STARS_PER_FRAME:
        picked = picked.head(MAX_STARS_PER_FRAME)
    return picked.reset_index(drop=True)


def _compute_snr(flux: float, sky: float, rap: float, gain: float, rn: float) -> float:
    if not (math.isfinite(flux) and flux > 0 and math.isfinite(sky) and math.isfinite(rap) and rap > 0):
        return 0.0
    g = gain if gain > 0 else 1.0
    area = math.pi * rap * rap
    var = flux / g + max(0.0, sky) / g * area + (rn / g) ** 2 * area
    if var <= 0:
        return 0.0
    return float(flux / math.sqrt(var))


def _background_subtract_cutout(data: np.ndarray, border: int = 3) -> np.ndarray:
    d = np.asarray(data, dtype=np.float64)
    h, w = d.shape
    if h <= 2 * border or w <= 2 * border:
        return d - float(np.nanmedian(d))
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
    return d - sky


def _fit_elliptical_gaussian(cutout: np.ndarray) -> dict[str, Any]:
    z = _background_subtract_cutout(cutout)
    h, w = z.shape
    yy, xx = np.mgrid[:h, :w]
    peak = float(np.nanmax(z))
    if not math.isfinite(peak) or peak <= 0:
        return {"ok": False, "reason": "nonpositive_peak"}

    cy, cx = np.unravel_index(int(np.nanargmax(z)), z.shape)
    amp0 = max(peak, 1e-3)
    sig0 = max(h, w) / 6.0
    model = models.Gaussian2D(
        amplitude=amp0,
        x_mean=float(cx),
        y_mean=float(cy),
        x_stddev=sig0,
        y_stddev=sig0,
        theta=0.0,
    )
    model.x_stddev.bounds = (0.05, None)
    model.y_stddev.bounds = (0.05, None)
    fitter = fitting.LevMarLSQFitter()
    try:
        fitted = fitter(model, xx, yy, z)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"fit_fail:{exc}"}

    sx = float(fitted.x_stddev.value)
    sy = float(fitted.y_stddev.value)
    if not (math.isfinite(sx) and math.isfinite(sy) and sx > 0 and sy > 0):
        return {"ok": False, "reason": "bad_stddev"}

    major, minor = (sx, sy) if sx >= sy else (sy, sx)
    elong = major / minor
    theta = float(fitted.theta.value)
    pa_deg = math.degrees(theta) % 180.0
    resid = z - fitted(xx, yy)
    dof = max(1, z.size - 6)
    chi2 = float(np.nansum(resid**2) / dof)
    return {
        "ok": True,
        "elongation": float(elong),
        "pa_deg": float(pa_deg),
        "fwhm_major_px": float(major * FWHM_FROM_STD),
        "fwhm_minor_px": float(minor * FWHM_FROM_STD),
        "chi2": chi2,
        "amplitude": float(fitted.amplitude.value),
    }


def _mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)))


def _robust_rms_mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    med = float(np.median(x))
    return float(MAD_SCALE * np.median(np.abs(x - med)))


def _circular_mean_deg(p_deg: np.ndarray) -> float:
    p = np.deg2rad(np.asarray(p_deg, dtype=float) * 2.0)
    p = p[np.isfinite(p)]
    if p.size == 0:
        return float("nan")
    c = float(np.mean(np.cos(p)))
    s = float(np.mean(np.sin(p)))
    if c == 0 and s == 0:
        return float("nan")
    ang = math.degrees(math.atan2(s, c)) / 2.0
    return float(ang % 180.0)


def _circular_std_deg(p_deg: np.ndarray) -> float:
    p = np.deg2rad(np.asarray(p_deg, dtype=float) * 2.0)
    p = p[np.isfinite(p)]
    if p.size == 0:
        return float("nan")
    r = math.hypot(float(np.mean(np.cos(p))), float(np.mean(np.sin(p))))
    r = min(max(r, 1e-12), 1.0)
    return float(math.degrees(math.sqrt(-2.0 * math.log(r))) / 2.0)


def _corr_pair(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 3:
        return {"pearson_r": float("nan"), "spearman_r": float("nan"), "n": int(m.sum())}
    pr = stats.pearsonr(x[m], y[m])
    sr = stats.spearmanr(x[m], y[m])
    return {"pearson_r": float(pr.statistic), "spearman_r": float(sr.statistic), "n": int(m.sum())}


def _verdict(med_elong: float, p75_elong: float, pa_circ_std: float) -> str:
    low = (math.isfinite(med_elong) and med_elong <= 1.05) and (
        not math.isfinite(p75_elong) or p75_elong <= 1.08
    )
    high = (math.isfinite(med_elong) and med_elong >= 1.08) or (
        math.isfinite(p75_elong) and p75_elong >= 1.12
    )
    pa_stable = math.isfinite(pa_circ_std) and pa_circ_std <= 20.0
    pa_varies = math.isfinite(pa_circ_std) and pa_circ_std >= 35.0
    if low:
        return "(i) elongation low everywhere (~1.0-1.05) -> smear negligible; scatter likely undersampling/noise"
    if high and pa_stable:
        return "(ii) elongation elevated but PA stable -> coherent tracking smear"
    if high and pa_varies:
        return "(iii) elongation elevated and PA varies frame-to-frame -> frame-variable smear (static ePSF cannot capture)"
    if high:
        return "elongation moderately elevated; PA coherence inconclusive -- inspect PA vs time plot"
    return "mixed/borderline elongation -- no strong smear signature"


def _part_b_mag_bins(proc_dir: Path, masterstars_csv: Path) -> tuple[pd.DataFrame, str, pd.DataFrame]:
    proc_files = sorted(proc_dir.glob("proc_*.csv"))
    if not proc_files:
        return pd.DataFrame(), "No proc_*.csv files — Part B skipped.", pd.DataFrame()

    sample = pd.read_csv(proc_files[0], nrows=0)
    need = {"catalog_id", "dao_flux", "psf_flux", "psf_fit_ok"}
    if not need.issubset(sample.columns):
        missing = sorted(need - set(sample.columns))
        return pd.DataFrame(), f"proc CSV missing columns {missing} — Part B skipped.", pd.DataFrame()

    ms = pd.read_csv(masterstars_csv, low_memory=False, usecols=lambda c: c in ("catalog_id", "mag", "catalog_mag", "phot_g_mean_mag"))
    ms["_cid"] = ms["catalog_id"].map(_norm_cid)
    mag_col = next((c for c in ("mag", "catalog_mag", "phot_g_mean_mag") if c in ms.columns), None)
    if mag_col:
        ms["catalog_mag_bin"] = pd.to_numeric(ms[mag_col], errors="coerce")
    else:
        ms["catalog_mag_bin"] = np.nan

    usecols = ["catalog_id", "dao_flux", "psf_flux", "psf_fit_ok", "catalog_mag", "phot_g_mean_mag"]
    chunks: list[pd.DataFrame] = []
    for p in proc_files:
        df = pd.read_csv(p, usecols=lambda c: c in usecols, low_memory=False, dtype={"catalog_id": str})
        df["_cid"] = df["catalog_id"].map(_norm_cid)
        chunks.append(df)
    all_df = pd.concat(chunks, ignore_index=True)
    all_df["dao_flux"] = pd.to_numeric(all_df["dao_flux"], errors="coerce")
    all_df["psf_flux"] = pd.to_numeric(all_df["psf_flux"], errors="coerce")
    all_df["psf_fit_ok"] = all_df["psf_fit_ok"].fillna(False).astype(bool)
    ok = (
        all_df["psf_fit_ok"]
        & all_df["dao_flux"].notna()
        & (all_df["dao_flux"] > 0)
        & all_df["psf_flux"].notna()
        & (all_df["psf_flux"] > 0)
    )
    all_df = all_df.loc[ok]
    if all_df.empty:
        return pd.DataFrame(), "No paired psf_fit_ok + dao_flux frames — Part B skipped.", pd.DataFrame()

    mag_map = ms.set_index("_cid")["catalog_mag_bin"].to_dict()
    cm_proc = pd.to_numeric(all_df.get("phot_g_mean_mag"), errors="coerce")
    if cm_proc.isna().all():
        cm_proc = pd.to_numeric(all_df.get("catalog_mag"), errors="coerce")
    all_df["mag_use"] = cm_proc
    all_df.loc[all_df["mag_use"].isna(), "mag_use"] = all_df.loc[all_df["mag_use"].isna(), "_cid"].map(mag_map)

    rows: list[dict[str, Any]] = []
    for cid, grp in all_df.groupby("_cid", sort=False):
        dao = grp["dao_flux"].to_numpy(dtype=float)
        psf = grp["psf_flux"].to_numpy(dtype=float)
        if dao.size < 5:
            continue
        apt_mag = -2.5 * np.log10(dao)
        psf_mag = -2.5 * np.log10(psf)
        offset = float(np.nanmedian(apt_mag) - np.nanmedian(psf_mag))
        psf_norm = psf_mag + offset
        apt_res = apt_mag - float(np.nanmedian(apt_mag))
        psf_res = psf_norm - float(np.nanmedian(psf_norm))
        mag_val = float(pd.to_numeric(grp["mag_use"], errors="coerce").median())
        rows.append(
            {
                "catalog_id": cid,
                "catalog_mag": mag_val,
                "n_frames": int(len(grp)),
                "rms_aperture": _robust_rms_mad(apt_res),
                "rms_psf": _robust_rms_mad(psf_res),
            }
        )

    if not rows:
        return pd.DataFrame(), "No stars with >=5 paired frames after filtering — Part B skipped.", pd.DataFrame()

    star_df = pd.DataFrame(rows)
    star_df["ratio_psf_aper"] = star_df["rms_psf"] / star_df["rms_aperture"].replace(0, np.nan)

    bins = [
        ("<11", -np.inf, 11.0),
        ("11-12", 11.0, 12.0),
        ("12-13", 12.0, 13.0),
        ("13-14", 13.0, 14.0),
        (">14", 14.0, np.inf),
    ]
    out_rows: list[dict[str, Any]] = []
    for label, lo, hi in bins:
        sub = star_df[(star_df["catalog_mag"] > lo) & (star_df["catalog_mag"] <= hi)]
        out_rows.append(
            {
                "mag_bin": label,
                "N": int(len(sub)),
                "median_rms_aperture": float(sub["rms_aperture"].median()) if len(sub) else float("nan"),
                "median_rms_psf": float(sub["rms_psf"].median()) if len(sub) else float("nan"),
                "median_ratio_psf_aper": float(sub["ratio_psf_aper"].median()) if len(sub) else float("nan"),
            }
        )
    return pd.DataFrame(out_rows), "", star_df


def run_diagnostic(*, draft_id: int, out_dir: Path | None = None) -> str:
    cfg = AppConfig()
    draft_dir = _resolve_draft_dir(cfg, draft_id)
    if not draft_dir.is_dir():
        raise FileNotFoundError(f"Draft directory not found: {draft_dir}")

    aligned_dir = _find_setup(draft_dir)
    setup_name = aligned_dir.name
    ps_bundle = _resolve_platesolve_bundle(draft_dir, setup_name)
    masterstar_fits = ps_bundle / "MASTERSTAR.fits"
    masterstars_csv = ps_bundle / "masterstars_full_match.csv"
    if not masterstars_csv.is_file():
        raise FileNotFoundError(f"Missing {masterstars_csv}")

    db = VyvarDatabase(cfg.database_path)
    drow = db.fetch_obs_draft_by_id(int(draft_id)) or {}
    equipment_id = drow.get("ID_EQUIPMENTS")
    telescope_id = drow.get("ID_TELESCOPE")

    with fits.open(masterstar_fits, memmap=True) as hdul:
        ms_hdr = hdul[0].header

    plate_scale, plate_src = _resolve_plate_scale_arcsec(
        masterstar_fits=masterstar_fits,
        hdr=ms_hdr,
        db=db,
        equipment_id=int(equipment_id) if equipment_id is not None else None,
        telescope_id=int(telescope_id) if telescope_id is not None else None,
        cfg=cfg,
    )
    fwhm_px = float(get_epsf_fwhm_from_context(masterstar_fits, db, int(draft_id)))

    meta_json = ps_bundle / "masterstar_epsf_meta.json"
    epsf_asym = float("nan")
    fit_shape = (9, 9)
    if meta_json.is_file():
        meta = json.loads(meta_json.read_text(encoding="utf-8"))
        qc = meta.get("epsf_qc") or {}
        epsf_asym = float(qc.get("epsf_asymmetry", float("nan")))
        fs = meta.get("fit_shape")
        if isinstance(fs, (list, tuple)) and len(fs) == 2:
            fit_shape = (int(fs[0]), int(fs[1]))

    masterstar_input_root = resolve_masterstar_input_root(draft_dir, setup_name=setup_name, app_config=cfg)
    proc_store = ProcFrameStore.build(
        aligned_dir,
        glob_pattern="proc_*.csv",
        extra_cols=[
            "psf_flux",
            "psf_fit_ok",
            "catalog_mag",
            "phot_g_mean_mag",
            "saturate_limit_adu",
            "peak_dao",
        ],
    )

    fits_files = sorted(aligned_dir.glob("proc_*.fits"))
    csv_by_stem = {p.stem: p for p in sorted(aligned_dir.glob("proc_*.csv"))}
    pairs = [(f, csv_by_stem.get(f.stem)) for f in fits_files if csv_by_stem.get(f.stem)]
    if not pairs:
        raise FileNotFoundError(f"No paired proc FITS+CSV under {aligned_dir}")

    from param_resolver import resolve_gain, resolve_read_noise  # noqa: PLC0415

    g_res = resolve_gain(ms_hdr, db=db, equipment_id=equipment_id, cfg=cfg)
    rn_res = resolve_read_noise(ms_hdr, db=db, equipment_id=equipment_id, cfg=cfg)
    gain = float(g_res.value) if g_res.ok and g_res.value else 1.0
    rn = float(rn_res.value) if rn_res.ok and rn_res.value else 10.0

    cutout_h = int(max(fit_shape) * 2)
    if cutout_h % 2 == 0:
        cutout_h += 1

    per_star_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    n_attempt = 0
    n_ok = 0
    n_drop = 0

    for fits_path, csv_path in pairs:
        assert csv_path is not None
        frame_df = proc_store.get_frame(str(csv_path))
        if frame_df is None:
            frame_df = pd.read_csv(csv_path, low_memory=False, dtype={"catalog_id": str})
        with fits.open(fits_path, memmap=True) as hdul:
            data = np.asarray(hdul[0].data, dtype=np.float32)
            fhdr = hdul[0].header
        img_shape = data.shape
        picked = _select_frame_stars_from_proc(
            frame_df,
            ps_bundle,
            fwhm_px=fwhm_px,
            plate_scale_arcsec_px=plate_scale,
            fit_shape=fit_shape,
            gain=gain,
            rn=rn,
            img_shape=img_shape,
        )

        bjd = float("nan")
        if "bjd_tdb_mid" in frame_df.columns:
            bjd = float(pd.to_numeric(frame_df["bjd_tdb_mid"], errors="coerce").dropna().median())
        elif "jd_mid" in frame_df.columns:
            bjd = float(pd.to_numeric(frame_df["jd_mid"], errors="coerce").dropna().median())
        airmass = float("nan")
        if "airmass" in frame_df.columns:
            airmass = float(pd.to_numeric(frame_df["airmass"], errors="coerce").dropna().median())
        if not math.isfinite(airmass):
            try:
                airmass = float(fhdr.get("AIRMASS", float("nan")))
            except (TypeError, ValueError):
                pass

        frame_fits: list[dict[str, Any]] = []

        for _, star in picked.iterrows():
            x0 = int(round(float(star["x"])))
            y0 = int(round(float(star["y"])))
            half = cutout_h // 2
            y1, y2 = max(0, y0 - half), min(img_shape[0], y0 + half + 1)
            x1, x2 = max(0, x0 - half), min(img_shape[1], x0 + half + 1)
            cut = data[y1:y2, x1:x2]
            if cut.size < 9:
                continue
            n_attempt += 1
            fit = _fit_elliptical_gaussian(cut)
            if not fit.get("ok"):
                n_drop += 1
                continue
            frame_fits.append(
                {
                    "frame": fits_path.name,
                    "catalog_id": star["_cid"],
                    "bjd": bjd,
                    "airmass": airmass,
                    **{k: fit[k] for k in ("elongation", "pa_deg", "fwhm_major_px", "fwhm_minor_px", "chi2")},
                }
            )

        chi2_vals = [float(r["chi2"]) for r in frame_fits]
        if chi2_vals:
            chi_med = float(np.median(chi2_vals))
            chi_mad = _mad(np.array(chi2_vals))
            chi_thr = chi_med + CHI2_OUTLIER_SIGMA * MAD_SCALE * chi_mad if chi_mad > 0 else float("inf")
        else:
            chi_thr = float("inf")

        elong_vals: list[float] = []
        pa_vals: list[float] = []
        for row in frame_fits:
            if row["chi2"] > chi_thr:
                n_drop += 1
                continue
            n_ok += 1
            elong_vals.append(float(row["elongation"]))
            pa_vals.append(float(row["pa_deg"]))
            per_star_rows.append(row)

        if elong_vals:
            frame_rows.append(
                {
                    "frame": fits_path.name,
                    "bjd": bjd,
                    "airmass": airmass,
                    "n_candidates": int(len(picked)),
                    "n_used": int(len(elong_vals)),
                    "median_elongation": float(np.median(elong_vals)),
                    "mad_elongation": _mad(np.array(elong_vals)),
                    "circular_median_pa_deg": _circular_mean_deg(np.array(pa_vals)),
                    "circular_spread_pa_deg": _circular_std_deg(np.array(pa_vals)),
                }
            )
        else:
            frame_rows.append(
                {
                    "frame": fits_path.name,
                    "bjd": bjd,
                    "airmass": airmass,
                    "n_candidates": int(len(picked)),
                    "n_used": 0,
                    "median_elongation": float("nan"),
                    "mad_elongation": float("nan"),
                    "circular_median_pa_deg": float("nan"),
                    "circular_spread_pa_deg": float("nan"),
                }
            )

    frame_df_out = pd.DataFrame(frame_rows)
    drop_rate = (100.0 * n_drop / n_attempt) if n_attempt else float("nan")
    n_stars_used = frame_df_out["n_used"].to_numpy(dtype=float)
    n_stars_used = n_stars_used[np.isfinite(n_stars_used)]

    elong_stats = frame_df_out["median_elongation"].dropna()
    elong_q = elong_stats.quantile([0.0, 0.25, 0.5, 0.75, 1.0]) if not elong_stats.empty else pd.Series(dtype=float)

    pa_per_frame = frame_df_out["circular_median_pa_deg"].dropna().to_numpy(dtype=float)
    pa_circ_std = _circular_std_deg(pa_per_frame)

    bjd_arr = frame_df_out["bjd"].to_numpy(dtype=float)
    am_arr = frame_df_out["airmass"].to_numpy(dtype=float)
    elong_arr = frame_df_out["median_elongation"].to_numpy(dtype=float)
    pa_arr = frame_df_out["circular_median_pa_deg"].to_numpy(dtype=float)

    corr_elong_bjd = _corr_pair(elong_arr, bjd_arr)
    corr_elong_am = _corr_pair(elong_arr, am_arr)
    corr_pa_bjd = _corr_pair(pa_arr, bjd_arr)

    med_elong = float(elong_stats.median()) if not elong_stats.empty else float("nan")
    p75_elong = float(elong_q.get(0.75, float("nan")))
    verdict = _verdict(med_elong, p75_elong, pa_circ_std)

    part_b_df, part_b_note, part_b_stars = _part_b_mag_bins(aligned_dir, masterstars_csv)

    if out_dir is None:
        out_dir = draft_dir / "diagnostics" / f"psf_elongation_{draft_id}"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_star_csv = out_dir / f"d{draft_id}_per_star_gaussian_fits.csv"
    frame_csv = out_dir / f"d{draft_id}_per_frame_elongation.csv"
    pd.DataFrame(per_star_rows).to_csv(per_star_csv, index=False)
    frame_df_out.to_csv(frame_csv, index=False)
    part_b_csv = out_dir / f"d{draft_id}_aperture_vs_psf_rms_by_mag.csv"
    part_b_stars_csv = out_dir / f"d{draft_id}_aperture_vs_psf_rms_per_star.csv"
    if not part_b_df.empty:
        part_b_df.to_csv(part_b_csv, index=False)
    if not part_b_stars.empty:
        part_b_stars.to_csv(part_b_stars_csv, index=False)

    png_paths: list[Path] = []
    try:
        import matplotlib.pyplot as plt

        def _scatter(x, y, xlabel, ylabel, title, fname):
            fig, ax = plt.subplots(figsize=(7, 4.5))
            m = np.isfinite(x) & np.isfinite(y)
            ax.scatter(x[m], y[m], s=18, alpha=0.75)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            fig.tight_layout()
            p = out_dir / fname
            fig.savefig(p, dpi=120)
            plt.close(fig)
            png_paths.append(p)

        _scatter(
            bjd_arr,
            elong_arr,
            "BJD",
            "Median elongation (major/minor σ)",
            f"Draft {draft_id}: per-frame PSF elongation vs BJD",
            f"d{draft_id}_elongation_vs_bjd.png",
        )
        _scatter(
            am_arr,
            elong_arr,
            "Airmass",
            "Median elongation (major/minor σ)",
            f"Draft {draft_id}: per-frame PSF elongation vs airmass",
            f"d{draft_id}_elongation_vs_airmass.png",
        )
        _scatter(
            bjd_arr,
            pa_arr,
            "BJD",
            "Circular-median PA (deg, 0–180)",
            f"Draft {draft_id}: per-frame PSF PA vs BJD",
            f"d{draft_id}_pa_vs_bjd.png",
        )
    except Exception as exc:  # noqa: BLE001
        png_note = f"PNG generation failed: {exc}"
    else:
        png_note = ""

    report_path = out_dir / f"d{draft_id}_psf_elongation_report.md"
    lines: list[str] = []
    lines.append(f"# PSF elongation diagnostic — draft {draft_id}")
    lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("")
    lines.append("## Resolved header")
    lines.append(f"- drafts root: `{draft_dir}`")
    lines.append(f"- setup: `{setup_name}`")
    lines.append(f"- plate scale: **{plate_scale:.4f} arcsec/px** ({plate_src})")
    lines.append(f"- reference FWHM: **{fwhm_px:.3f} px**")
    lines.append(f"- n_frames (aligned FITS with proc CSV): **{len(pairs)}**")
    lines.append(f"- aligned FITS root: `{aligned_dir}`")
    lines.append(f"- resolve_masterstar_input_root (Phase-2 input): `{masterstar_input_root}`")
    lines.append(f"- ProcFrameStore frames: **{len(proc_store)}**")
    lines.append(
        f"- star selection: COG-style isolation {ISOLATION_FWHM}xFWHM cone test, SNR>={SNR_MIN}, "
        f"peak<{SAT_FRAC}x sat, cap {MAX_STARS_PER_FRAME}/frame"
    )
    if len(n_stars_used):
        lines.append(
            f"- median n_stars/frame used: **{float(np.median(n_stars_used)):.1f}** "
            f"(min {int(np.min(n_stars_used))}, max {int(np.max(n_stars_used))})"
        )
    else:
        lines.append("- median n_stars/frame used: **N/A (no successful fits)**")
    lines.append(f"- stacked ePSF epsf_asymmetry (meta): **{epsf_asym:.4f}**" if math.isfinite(epsf_asym) else "- stacked ePSF epsf_asymmetry: **missing in meta**")
    lines.append("")
    lines.append("## Part A — per-frame elongation")
    lines.append(f"- Gaussian fit attempts: {n_attempt}; dropped (fail/chi2): {n_drop} ({drop_rate:.1f}%)")
    if not elong_stats.empty:
        lines.append(
            "- Per-frame median elongation distribution: "
            f"min={elong_q.get(0.0, float('nan')):.4f}, "
            f"25th={elong_q.get(0.25, float('nan')):.4f}, "
            f"median={elong_q.get(0.5, float('nan')):.4f}, "
            f"75th={elong_q.get(0.75, float('nan')):.4f}, "
            f"max={elong_q.get(1.0, float('nan')):.4f}"
        )
    else:
        lines.append("- Per-frame median elongation distribution: **no data**")
    lines.append(f"- PA across frames: circular std of per-frame median PA = **{pa_circ_std:.1f} deg**")
    lines.append(
        f"- Corr(elongation, BJD): Pearson r={corr_elong_bjd['pearson_r']:.3f}, "
        f"Spearman r={corr_elong_bjd['spearman_r']:.3f} (n={corr_elong_bjd['n']})"
    )
    lines.append(
        f"- Corr(elongation, airmass): Pearson r={corr_elong_am['pearson_r']:.3f}, "
        f"Spearman r={corr_elong_am['spearman_r']:.3f} (n={corr_elong_am['n']})"
    )
    lines.append(
        f"- Corr(PA, BJD): Pearson r={corr_pa_bjd['pearson_r']:.3f}, "
        f"Spearman r={corr_pa_bjd['spearman_r']:.3f} (n={corr_pa_bjd['n']})"
    )
    lines.append(f"- **Verdict:** {verdict}")
    lines.append("")
    lines.append("## Part B — aperture vs PSF RMS by magnitude")
    if part_b_note:
        lines.append(f"- **Skipped:** {part_b_note}")
    elif not part_b_df.empty:
        lines.append("| mag_bin | N | median RMS_aperture | median RMS_psf | median psf/aper |")
        lines.append("|---------|---|--------------------:|---------------:|----------------:|")
        for _, r in part_b_df.iterrows():
            lines.append(
                f"| {r['mag_bin']} | {int(r['N'])} | {r['median_rms_aperture']:.5f} | "
                f"{r['median_rms_psf']:.5f} | {r['median_ratio_psf_aper']:.3f} |"
            )
        lines.append("")
        lines.append("RMS = 1.48*MAD of per-frame mag residuals; PSF ZP-normalized to aperture median (dashboard style).")
    lines.append("")
    lines.append("## Output files")
    lines.append(f"- report: `{report_path}`")
    lines.append(f"- per-star CSV: `{per_star_csv}`")
    lines.append(f"- per-frame CSV: `{frame_csv}`")
    if not part_b_df.empty:
        lines.append(f"- Part B bin CSV: `{part_b_csv}`")
    if not part_b_stars.empty:
        lines.append(f"- Part B per-star CSV: `{part_b_stars_csv}`")

    for p in png_paths:
        lines.append(f"- PNG: `{p}`")
    if png_note:
        lines.append(f"- PNG note: {png_note}")

    report_text = "\n".join(lines)
    report_path.write_text(report_text, encoding="utf-8")

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(report_text)
    return report_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Draft PSF elongation diagnostic (read-only).")
    parser.add_argument("--draft", type=int, default=DRAFT_ID_DEFAULT)
    parser.add_argument("--out-dir", type=str, default="", help="Output directory (default: draft/diagnostics/...)")
    args = parser.parse_args()
    out = Path(args.out_dir.strip()) if args.out_dir.strip() else None
    run_diagnostic(draft_id=int(args.draft), out_dir=out)


if __name__ == "__main__":
    main()
