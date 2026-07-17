#!/usr/bin/env python3
"""Forced photometry diagnostic — Palomar 7 draft 364 Luminance_180_2 (read-only)."""
from __future__ import annotations

import json
import logging
import math
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from scipy.spatial import cKDTree

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401  (repo layout: src_py + dev on sys.path)
_ROOT = _bootstrap.REPO_ROOT
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from comp_selection_per_target import _angular_distance_deg_vectorized  # noqa: E402
from config import AppConfig  # noqa: E402
from database import VyvarDatabase, get_gaia_db_max_g_mag, query_local_gaia  # noqa: E402
from gaia_catalog_id import normalize_gaia_source_id  # noqa: E402
from photometry_core import (  # noqa: E402
    _aperture_radius_from_snr_table,
    load_snr_aperture_table_from_draft_dir,
)
from psf_photometry import _grouped_psf_fit  # noqa: E402
from photutils.aperture import CircularAnnulus, CircularAperture, aperture_photometry  # noqa: E402
from photutils.psf import ImagePSF, PSFPhotometry  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning, module="photutils")
logging.getLogger("astropy").setLevel(logging.ERROR)

DRAFT_ID = 364
SETUP = "Luminance_180_2"
FIELD_DB = _ROOT / "GAIA_DR3" / "vyvar_gaia_dr3_pal7_field.db"
MAD_SCALE = 1.4826
MIN_FRAMES = 5
SAT_FRAC = 0.85
ZP_CAL_MAG_MAX = 13.5
ISOLATION_DELTA_MAG = 2.5


def _norm_cid(raw: Any) -> str:
    if raw is None:
        return ""
    s = str(raw).strip()
    if not s or s.lower() in ("nan", "none"):
        return ""
    try:
        return normalize_gaia_source_id(raw)
    except Exception:  # noqa: BLE001
        return s


def _robust_rms_mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    med = float(np.median(x))
    return float(MAD_SCALE * np.median(np.abs(x - med)))


def _load_deep_cone(ps_dir: Path, ms_fits: Path) -> pd.DataFrame:
    if FIELD_DB.is_file():
        with fits.open(ms_fits, memmap=True) as hd:
            hdr = hd[0].header
        cr1 = float(hdr.get("CRVAL1", 272.684))
        cr2 = float(hdr.get("CRVAL2", -7.208))
        radius = 0.275
        max_g = float(get_gaia_db_max_g_mag(FIELD_DB))
        rows = query_local_gaia(
            FIELD_DB,
            ra_min=cr1 - radius,
            ra_max=cr1 + radius,
            dec_min=cr2 - radius,
            dec_max=cr2 + radius,
            mag_limit=max_g,
            max_rows=50_000,
        )
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.rename(columns={"source_id": "catalog_id", "g_mag": "mag"})
            df["catalog_id"] = df["catalog_id"].map(_norm_cid)
            df["ra_deg"] = pd.to_numeric(df["ra"], errors="coerce")
            df["dec_deg"] = pd.to_numeric(df["dec"], errors="coerce")
            df["mag"] = pd.to_numeric(df["mag"], errors="coerce")
            center = SkyCoord(ra=cr1 * u.deg, dec=cr2 * u.deg)
            sep = center.separation(SkyCoord(df["ra_deg"], df["dec_deg"], unit="deg")).deg
            return df.loc[sep <= radius].copy()

    cone_csv = ps_dir / "field_catalog_cone.csv"
    if cone_csv.is_file():
        df = pd.read_csv(cone_csv, low_memory=False)
        df["catalog_id"] = df.get("catalog_id", df.get("SOURCE_ID_GAIA", "")).map(_norm_cid)
        if "mag" not in df.columns and "phot_g_mean_mag" in df.columns:
            df["mag"] = pd.to_numeric(df["phot_g_mean_mag"], errors="coerce")
        return df
    raise FileNotFoundError("No field DB or field_catalog_cone.csv available")


def _cone_crowding_kdtree(df: pd.DataFrame, *, fwhm_px: float, plate_scale: float) -> np.ndarray:
    ra = df["ra_deg"].to_numpy(dtype=float)
    de = df["dec_deg"].to_numpy(dtype=float)
    mag = df["mag"].to_numpy(dtype=float)
    iso_r_deg = 2.0 * float(fwhm_px) * float(plate_scale) / 3600.0
    self_deg = 0.5 * float(plate_scale) / 3600.0
    iso_r_px = 2.0 * float(fwhm_px)
    xy = df[["x", "y"]].to_numpy(dtype=float)
    tree = cKDTree(xy)
    crowded = np.zeros(len(df), dtype=bool)
    for j in range(len(df)):
        if not (math.isfinite(ra[j]) and math.isfinite(de[j])):
            continue
        idx = tree.query_ball_point(xy[j], r=iso_r_px)
        idx = [k for k in idx if k != j]
        if not idx:
            continue
        cosd = max(math.cos(math.radians(de[j])), 0.2)
        box = (np.abs(de[idx] - de[j]) <= iso_r_deg * 1.5) & (
            np.abs(ra[idx] - ra[j]) <= iso_r_deg * 1.5 / cosd
        )
        if not box.any():
            continue
        d_deg = _angular_distance_deg_vectorized(ra[j], de[j], ra[idx], de[idx])
        m_box = mag[idx]
        cm = float(mag[j]) if math.isfinite(mag[j]) else float("nan")
        near = (d_deg > self_deg) & (d_deg <= iso_r_deg)
        if math.isfinite(cm):
            contaminating = near & ((m_box - cm) <= ISOLATION_DELTA_MAG)
        else:
            contaminating = near
        crowded[j] = bool(np.any(contaminating))
    return crowded


def _forced_single_psf(
    data: np.ndarray,
    x: float,
    y: float,
    *,
    psf_model: ImagePSF,
    fit_shape: tuple[int, int],
    fwhm_px: float,
) -> float:
    h, w = data.shape
    fh, fw = int(fit_shape[0]), int(fit_shape[1])
    half_y, half_x = fh // 2, fw // 2
    x0, y0 = int(round(x)), int(round(y))
    y1, y2 = max(0, y0 - half_y), min(h, y0 + half_y + 1)
    x1, x2 = max(0, x0 - half_x), min(w, x0 + half_x + 1)
    cut = np.asarray(data[y1:y2, x1:x2], dtype=np.float64)
    if cut.size < 9:
        return float("nan")
    border = np.ones(cut.shape, dtype=bool)
    if cut.shape[0] > 4 and cut.shape[1] > 4:
        border[2:-2, 2:-2] = False
    sky = float(np.median(cut[border])) if border.any() else float(np.median(cut))
    cut_sub = cut - sky
    tx, ty = float(x) - x1, float(y) - y1
    yy, xx = np.mgrid[0 : cut.shape[0], 0 : cut.shape[1]]
    near = np.hypot(xx - tx, yy - ty) <= max(2.0, float(fwhm_px))
    tflux = float(np.nansum(cut_sub[near].clip(min=0)))
    if not math.isfinite(tflux) or tflux <= 0:
        tflux = max(1.0, float(np.nanmax(cut_sub)) if math.isfinite(float(np.nanmax(cut_sub))) else 1.0)
    try:
        phot = PSFPhotometry(psf_model, fit_shape, progress_bar=False)
        init = Table([[tx], [ty], [tflux]], names=("x_0", "y_0", "flux_0"))
        try:
            phot.set_fixed_params(["x_0", "y_0"])
        except Exception:  # noqa: BLE001
            pass
        res = phot(cut_sub, init_params=init)
        flux = float(res["flux_fit"][0])
        xf = float(res["x_fit"][0])
        yf = float(res["y_fit"][0])
        if math.hypot(xf - tx, yf - ty) > 1.5:
            return float("nan")
        return flux if math.isfinite(flux) and flux > 0 else float("nan")
    except Exception:  # noqa: BLE001
        return float("nan")


def _batch_aperture_flux(
    data: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    r_ap: np.ndarray,
    *,
    ann_in: float,
    ann_out: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(x)
    flux = np.full(n, np.nan, dtype=float)
    peak = np.full(n, np.nan, dtype=float)
    pos = np.column_stack([x, y])
    for r in np.unique(r_ap):
        mask = np.isclose(r_ap, r)
        if not mask.any():
            continue
        ap = CircularAperture(pos[mask], r=float(r))
        an = CircularAnnulus(pos[mask], r_in=ann_in, r_out=ann_out)
        phot_ap = aperture_photometry(data, ap)
        phot_an = aperture_photometry(data, an)
        sums = np.asarray(phot_ap["aperture_sum"], dtype=float)
        areas = float(ap.area)
        sky = np.asarray(phot_an["aperture_sum"], dtype=float) / float(an.area)
        net = sums - sky * areas
        flux[mask] = net
        try:
            masks = ap.to_mask(method="center")
            if not isinstance(masks, list):
                masks = [masks]
            for k, m in enumerate(masks):
                vals = m.get_values(data)
                if vals is not None:
                    peak[np.where(mask)[0][k]] = float(np.nanmax(vals))
        except Exception:  # noqa: BLE001
            pass
    return flux, peak


def _mag_bins() -> list[tuple[str, float, float]]:
    return [
        ("<12", -np.inf, 12.0),
        ("12-13", 12.0, 13.0),
        ("13-14", 13.0, 14.0),
        ("14-15", 14.0, 15.0),
        ("15-16", 15.0, 16.0),
        ("16-17", 16.0, 17.0),
        ("17-18", 17.0, 18.0),
        ("18-19", 18.0, 19.0),
        ("19-20", 19.0, 20.0),
        (">20", 20.0, np.inf),
    ]


def _psf_worker(args: tuple) -> tuple[int, float]:
    j, crowded, x, y, data, nbr_xy, psf_path, osamp, fit_shape, fwhm_px = args
    psf_data = np.asarray(fits.getdata(psf_path), dtype=np.float64)
    psf_model = ImagePSF(psf_data, oversampling=osamp)
    fit_shape_t = (int(fit_shape[0]), int(fit_shape[1]))
    if crowded:
        grp = _grouped_psf_fit(
            data,
            None,
            x,
            y,
            fwhm_px=fwhm_px,
            fit_shape=fit_shape_t,
            psf_model=psf_model,
            neighbor_xy=nbr_xy,
            neighbor_flux=np.full(len(nbr_xy), np.nan),
            group_sep_fwhm=1.5,
            neighbor_include_fwhm=3.0,
            chi2_limit=50.0,
        )
        return j, float(grp["psf_flux"]) if grp else float("nan")
    return j, _forced_single_psf(data, x, y, psf_model=psf_model, fit_shape=fit_shape_t, fwhm_px=fwhm_px)


def run() -> dict[str, Any]:
    cfg = AppConfig()
    if cfg.psf_photometry_enabled:
        raise RuntimeError("psf_photometry_enabled must be false — this script loads ePSF directly")
    db = VyvarDatabase(cfg.database_path)
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
    try:
        row = db.fetch_obs_draft_by_id(DRAFT_ID) or {}
        eq = db.conn.execute(
            "SELECT SATURATE_ADU FROM EQUIPMENTS WHERE ID=?", (int(row.get("EQUIPMENT_ID") or 3),)
        ).fetchone()
        if eq and eq[0]:
            sat_limit = float(eq[0])
    except Exception:  # noqa: BLE001
        pass

    with fits.open(ms_fits, memmap=True) as hd:
        ms_wcs = WCS(hd[0].header)
        naxis1 = int(hd[0].header.get("NAXIS1", hd[0].data.shape[1]))
        naxis2 = int(hd[0].header.get("NAXIS2", hd[0].data.shape[0]))

    cone = _load_deep_cone(ps_dir, ms_fits)
    ra = cone["ra_deg"].to_numpy(dtype=float)
    de = cone["dec_deg"].to_numpy(dtype=float)
    xp, yp = ms_wcs.all_world2pix(np.column_stack([ra, de]), 0).T
    cone = cone.assign(x=xp, y=yp)
    margin = 2.0 * fwhm_px
    cone = cone.loc[
        (cone["x"] >= margin)
        & (cone["x"] < naxis1 - margin)
        & (cone["y"] >= margin)
        & (cone["y"] < naxis2 - margin)
    ].copy().reset_index(drop=True)

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
    cone["r_ap"] = r_ap
    cone["crowded"] = _cone_crowding_kdtree(cone, fwhm_px=fwhm_px, plate_scale=plate_scale)
    cone["isolated"] = ~cone["crowded"]

    psf_data = np.asarray(fits.getdata(epsf_path), dtype=np.float64)
    _ = psf_data  # loaded once to verify ePSF readable
    fit_shape_t = (int(fit_shape[0]), int(fit_shape[1]))
    r_ap_report = _aperture_radius_from_snr_table(
        15.0, snr_table, aperture_fwhm_factor=float(cfg.aperture_fwhm_factor), fwhm_px=fwhm_snr
    )

    cal_mask = cone["isolated"] & (mags <= ZP_CAL_MAG_MAX)
    nbr_xy = cone[["x", "y"]].to_numpy(dtype=float)
    x_all = cone["x"].to_numpy(dtype=float)
    y_all = cone["y"].to_numpy(dtype=float)
    cid_all = cone["catalog_id"].astype(str).to_numpy()
    crowded_all = cone["crowded"].to_numpy(dtype=bool)

    frame_files = sorted(aligned.glob("proc_*.fits"))
    if not frame_files:
        raise FileNotFoundError(f"No aligned frames under {aligned}")

    records: list[dict[str, Any]] = []
    for fi, fpath in enumerate(frame_files):
        print(f"[forced] frame {fi + 1}/{len(frame_files)}: {fpath.name}", flush=True)
        with fits.open(fpath, memmap=True) as hd:
            data = np.asarray(hd[0].data, dtype=np.float64)
        a_flux, peaks = _batch_aperture_flux(data, x_all, y_all, r_ap, ann_in=ann_in, ann_out=ann_out)
        sat_mask = peaks > SAT_FRAC * sat_limit
        a_flux[sat_mask] = np.nan

        p_flux = np.full(len(cone), np.nan, dtype=float)
        valid_j = np.where(np.isfinite(a_flux) & (a_flux > 0))[0]
        worker_args = [
            (
                int(j),
                bool(crowded_all[j]),
                float(x_all[j]),
                float(y_all[j]),
                data,
                nbr_xy,
                str(epsf_path),
                osamp,
                fit_shape,
                fwhm_px,
            )
            for j in valid_j
        ]
        if worker_args:
            with ProcessPoolExecutor(max_workers=4) as pool:
                futs = [pool.submit(_psf_worker, a) for a in worker_args]
                for fut in as_completed(futs):
                    j, pf = fut.result()
                    p_flux[j] = pf

        cal = cal_mask.to_numpy()
        zp_aper = zp_psf = float("nan")
        ok_a = cal & np.isfinite(a_flux) & (a_flux > 0)
        ok_p = cal & np.isfinite(p_flux) & (p_flux > 0)
        if ok_a.any():
            zp_aper = float(np.median(mags[ok_a] + 2.5 * np.log10(a_flux[ok_a])))
        if ok_p.any():
            zp_psf = float(np.median(mags[ok_p] + 2.5 * np.log10(p_flux[ok_p])))

        for j in range(len(cone)):
            mag_aper = mag_psf = float("nan")
            if math.isfinite(zp_aper) and math.isfinite(a_flux[j]) and a_flux[j] > 0:
                mag_aper = zp_aper - 2.5 * math.log10(a_flux[j])
            if math.isfinite(zp_psf) and math.isfinite(p_flux[j]) and p_flux[j] > 0:
                mag_psf = zp_psf - 2.5 * math.log10(p_flux[j])
            records.append(
                {
                    "frame": fpath.name,
                    "catalog_id": cid_all[j],
                    "mag": float(mags[j]) if math.isfinite(mags[j]) else float("nan"),
                    "crowded": bool(crowded_all[j]),
                    "aper_flux": float(a_flux[j]),
                    "psf_flux": float(p_flux[j]),
                    "mag_aper": mag_aper,
                    "mag_psf": mag_psf,
                }
            )

    all_df = pd.DataFrame(records)
    star_rows: list[dict[str, Any]] = []
    for cid, grp in all_df.groupby("catalog_id", sort=False):
        mag_val = float(pd.to_numeric(grp["mag"], errors="coerce").median())
        apt = grp["mag_aper"].to_numpy(dtype=float)
        psf = grp["mag_psf"].to_numpy(dtype=float)
        ok = np.isfinite(apt) & np.isfinite(psf)
        if ok.sum() < MIN_FRAMES:
            continue
        apt = apt[ok]
        psf = psf[ok]
        offset = float(np.median(apt - psf))
        psf_norm = psf + offset
        apt_res = apt - float(np.median(apt))
        psf_res = psf_norm - float(np.median(psf_norm))
        star_rows.append(
            {
                "catalog_id": cid,
                "catalog_mag": mag_val,
                "n_frames": int(ok.sum()),
                "crowded": bool(grp["crowded"].iloc[0]),
                "rms_aperture": _robust_rms_mad(apt_res),
                "rms_psf": _robust_rms_mad(psf_res),
            }
        )
    star_df = pd.DataFrame(star_rows)
    if not star_df.empty:
        star_df["ratio_psf_aper"] = star_df["rms_psf"] / star_df["rms_aperture"].replace(0, np.nan)
        star_df["crowding_class"] = np.where(star_df["crowded"], "crowded", "isolated")

    mag_rows, crowd_rows = [], []
    for label, lo, hi in _mag_bins():
        sub = star_df[(star_df["catalog_mag"] > lo) & (star_df["catalog_mag"] <= hi)]
        mag_rows.append(
            {
                "mag_bin": label,
                "N": int(len(sub)),
                "median_rms_aperture": float(sub["rms_aperture"].median()) if len(sub) else float("nan"),
                "median_rms_psf": float(sub["rms_psf"].median()) if len(sub) else float("nan"),
                "median_ratio_psf_aper": float(sub["ratio_psf_aper"].median()) if len(sub) else float("nan"),
            }
        )
        for cls in ("isolated", "crowded"):
            sc = sub[sub["crowding_class"] == cls]
            crowd_rows.append(
                {
                    "mag_bin": label,
                    "crowding_class": cls,
                    "N": int(len(sc)),
                    "median_rms_aperture": float(sc["rms_aperture"].median()) if len(sc) else float("nan"),
                    "median_rms_psf": float(sc["rms_psf"].median()) if len(sc) else float("nan"),
                    "median_ratio_psf_aper": float(sc["ratio_psf_aper"].median()) if len(sc) else float("nan"),
                }
            )
    mag_df = pd.DataFrame(mag_rows)
    crowd_df = pd.DataFrame(crowd_rows)

    out_dir = draft_dir / "diagnostics" / "forced_photometry_pal7"
    out_dir.mkdir(parents=True, exist_ok=True)
    mag_csv = out_dir / "d364_forced_aperture_vs_psf_by_mag.csv"
    crowd_csv = out_dir / "d364_forced_aperture_vs_psf_crowding.csv"
    stars_csv = out_dir / "d364_forced_aperture_vs_psf_per_star.csv"
    mag_df.to_csv(mag_csv, index=False)
    crowd_df.to_csv(crowd_csv, index=False)
    star_df.to_csv(stars_csv, index=False)

    faint_valid = star_df[star_df["catalog_mag"] >= 18.0]
    return {
        "draft_id": DRAFT_ID,
        "setup": SETUP,
        "n_forced_positions": int(len(cone)),
        "n_frames": int(len(frame_files)),
        "aperture_r_px_at_G15": float(r_ap_report),
        "annulus_inner_px": float(ann_in),
        "annulus_outer_px": float(ann_out),
        "fwhm_px": fwhm_px,
        "epsf_path": str(epsf_path),
        "cone_faintest_g": float(np.nanmax(mags)),
        "n_stars_valid_ge5_frames": int(len(star_df)),
        "n_faint_ge18_valid": int(len(faint_valid)),
        "mag_bin_csv": str(mag_csv),
        "crowding_csv": str(crowd_csv),
        "per_star_csv": str(stars_csv),
        "mag_bin_table": mag_df.to_dict(orient="records"),
        "crowding_table": crowd_df.to_dict(orient="records"),
        "psf_flag_in_config": bool(cfg.psf_photometry_enabled),
    }


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    result = run()
    (_ROOT / "tmp" / "pilot_palomar7_forced_photometry_result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
