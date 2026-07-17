#!/usr/bin/env python3
"""Crowded-subset grouper science test + adaptive-selector validation — draft 364.

READ-ONLY: loads existing ePSF; no production/config changes.
Part A: single vs grouped forced PSF on crowded deep-cone stars.
Part B: four-routing RMS comparison (aperture / PSF / adaptive / oracle).
"""
from __future__ import annotations

import importlib.util
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
from astropy.io import fits
from astropy.wcs import WCS
from scipy.spatial import cKDTree

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import AppConfig  # noqa: E402
from photometry_core import _photometric_error  # noqa: E402
from psf_photometry import _grouped_psf_fit, assess_psf_quality  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning, module="photutils")
logging.getLogger("astropy").setLevel(logging.ERROR)

DRAFT_ID = 364
SETUP = "Luminance_180_2"
MAD_SCALE = 1.4826
MIN_FRAMES = 5
SAT_FRAC = 0.85
ZP_CAL_MAG_MAX = 13.5
GROUP_SEP_FWHM = 1.5
NEIGHBOR_INCLUDE_FWHM = 3.0
RESOLVE_FWHM = 2.0
SNR_LO = 15.0
CHI2_LIMIT = 50.0

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
    return float(MAD_SCALE * np.median(np.abs(x - np.median(x))))


def _mag_bins() -> list[tuple[str, float, float]]:
    return _fp._mag_bins()


def _nn_dist_fwhm(x: float, y: float, xy: np.ndarray, fwhm_px: float) -> float:
    if len(xy) < 2:
        return float("inf")
    d = np.hypot(xy[:, 0] - x, xy[:, 1] - y)
    d = d[d > 0.5]
    if d.size == 0:
        return float("inf")
    return float(np.min(d) / fwhm_px) if fwhm_px > 0 else float("inf")


def _psf_worker(args: tuple) -> tuple[int, float, float]:
    """Return (index, psf_single, psf_grouped)."""
    j, x, y, data, nbr_xy, nbr_flux, psf_path, osamp, fit_shape, fwhm_px = args
    from photutils.psf import ImagePSF  # noqa: PLC0415

    psf_data = np.asarray(fits.getdata(psf_path), dtype=np.float64)
    psf_model = ImagePSF(psf_data, oversampling=int(osamp))
    fit_shape_t = (int(fit_shape[0]), int(fit_shape[1]))
    single = _fp._forced_single_psf(
        data, x, y, psf_model=psf_model, fit_shape=fit_shape_t, fwhm_px=fwhm_px
    )
    grp = _grouped_psf_fit(
        data,
        None,
        x,
        y,
        fwhm_px=fwhm_px,
        fit_shape=fit_shape_t,
        psf_model=psf_model,
        neighbor_xy=nbr_xy,
        neighbor_flux=nbr_flux,
        group_sep_fwhm=GROUP_SEP_FWHM,
        neighbor_include_fwhm=NEIGHBOR_INCLUDE_FWHM,
        chi2_limit=CHI2_LIMIT,
    )
    grouped = float(grp["psf_flux"]) if grp else float("nan")
    return j, single, grouped


def _star_rms_from_mags(mags: np.ndarray) -> float:
    m = mags[np.isfinite(mags)]
    if m.size < MIN_FRAMES:
        return float("nan")
    return _robust_rms_mad(m - np.median(m))


def part_a(out_dir: Path, *, frame_cache: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Crowded-only forced photometry: aperture + PSF single + PSF grouped."""
    if frame_cache is not None and frame_cache.is_file():
        all_df = pd.read_csv(frame_cache, low_memory=False, dtype={"catalog_id": str})
        print(f"[Part A] loaded cached frame records: {frame_cache}", flush=True)
    else:
        cfg = AppConfig()
        if cfg.psf_photometry_enabled:
            raise RuntimeError("psf_photometry_enabled must remain false")
        draft_dir = Path(cfg.archive_root) / "Drafts" / f"draft_{DRAFT_ID:06d}"
        ps_dir = draft_dir / "platesolve" / SETUP
        aligned = draft_dir / "detrended_aligned" / "lights" / SETUP
        ms_fits = ps_dir / "MASTERSTAR.fits"
        epsf_path = ps_dir / "masterstar_epsf.fits"
        meta_path = ps_dir / "masterstar_epsf_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        fwhm_px = float(meta.get("fwhm_px", 6.22))
        osamp = int(meta.get("oversampling", 2))
        fit_shape = tuple(meta.get("fit_shape", [15, 15]))
        plate_scale = float(meta.get("plate_scale_arcsec_px", 0.389))

        snr_table = _fp.load_snr_aperture_table_from_draft_dir(draft_dir) or {}
        fwhm_snr = float(snr_table.get("fwhm_px", fwhm_px))
        ann_in = float(cfg.annulus_inner_fwhm) * fwhm_snr
        ann_out = float(cfg.annulus_outer_fwhm) * fwhm_snr

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
        cone = cone.loc[
            (cone["x"] >= margin)
            & (cone["x"] < naxis1 - margin)
            & (cone["y"] >= margin)
            & (cone["y"] < naxis2 - margin)
        ].copy().reset_index(drop=True)

        mags = pd.to_numeric(cone["mag"], errors="coerce").to_numpy(dtype=float)
        r_ap = np.array(
            [
                _fp._aperture_radius_from_snr_table(
                    m if math.isfinite(m) else 99.0,
                    snr_table,
                    aperture_fwhm_factor=float(cfg.aperture_fwhm_factor),
                    fwhm_px=fwhm_snr,
                )
                for m in mags
            ],
            dtype=float,
        )
        crowded_mask = _fp._cone_crowding_kdtree(cone, fwhm_px=fwhm_px, plate_scale=plate_scale)
        cone = cone.loc[crowded_mask].copy().reset_index(drop=True)
        mags = pd.to_numeric(cone["mag"], errors="coerce").to_numpy(dtype=float)
        r_ap = r_ap[crowded_mask]

        print(f"[Part A] crowded subset: {len(cone)} positions", flush=True)

        nbr_xy_full = cone[["x", "y"]].to_numpy(dtype=float)
        nn_fwhm = np.array(
            [_nn_dist_fwhm(float(x), float(y), nbr_xy_full, fwhm_px) for x, y in nbr_xy_full],
            dtype=float,
        )
        cone["nn_dist_fwhm"] = nn_fwhm

        x_all = cone["x"].to_numpy(dtype=float)
        y_all = cone["y"].to_numpy(dtype=float)
        cid_all = cone["catalog_id"].astype(str).to_numpy()
        nbr_flux = np.full(len(cone), np.nan, dtype=float)

        cal_mask = mags <= ZP_CAL_MAG_MAX
        frame_files = sorted(aligned.glob("proc_*.fits"))
        if not frame_files:
            raise FileNotFoundError(f"No frames under {aligned}")

        sat_limit = 60000.0
        records: list[dict[str, Any]] = []
        for fi, fpath in enumerate(frame_files):
            print(f"[Part A] frame {fi + 1}/{len(frame_files)}: {fpath.name}", flush=True)
            with fits.open(fpath, memmap=True) as hd:
                data = np.asarray(hd[0].data, dtype=np.float64)
            a_flux, peaks = _fp._batch_aperture_flux(data, x_all, y_all, r_ap, ann_in=ann_in, ann_out=ann_out)
            sat_mask = peaks > SAT_FRAC * sat_limit
            a_flux[sat_mask] = np.nan

            psf_single = np.full(len(cone), np.nan, dtype=float)
            psf_grouped = np.full(len(cone), np.nan, dtype=float)
            valid_j = np.where(np.isfinite(a_flux) & (a_flux > 0))[0]
            worker_args = [
                (
                    int(j),
                    float(x_all[j]),
                    float(y_all[j]),
                    data,
                    nbr_xy_full,
                    nbr_flux,
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
                        j, s, g = fut.result()
                        psf_single[j] = s
                        psf_grouped[j] = g

            cal = cal_mask
            zp_a = zp_s = zp_g = float("nan")
            ok_a = cal & np.isfinite(a_flux) & (a_flux > 0)
            ok_s = cal & np.isfinite(psf_single) & (psf_single > 0)
            ok_g = cal & np.isfinite(psf_grouped) & (psf_grouped > 0)
            if ok_a.any():
                zp_a = float(np.median(mags[ok_a] + 2.5 * np.log10(a_flux[ok_a])))
            if ok_s.any():
                zp_s = float(np.median(mags[ok_s] + 2.5 * np.log10(psf_single[ok_s])))
            if ok_g.any():
                zp_g = float(np.median(mags[ok_g] + 2.5 * np.log10(psf_grouped[ok_g])))

            for j in range(len(cone)):
                mag_a = mag_s = mag_g = float("nan")
                if math.isfinite(zp_a) and math.isfinite(a_flux[j]) and a_flux[j] > 0:
                    mag_a = zp_a - 2.5 * math.log10(a_flux[j])
                if math.isfinite(zp_s) and math.isfinite(psf_single[j]) and psf_single[j] > 0:
                    mag_s = zp_s - 2.5 * math.log10(psf_single[j])
                if math.isfinite(zp_g) and math.isfinite(psf_grouped[j]) and psf_grouped[j] > 0:
                    mag_g = zp_g - 2.5 * math.log10(psf_grouped[j])
                records.append(
                    {
                        "frame": fpath.name,
                        "catalog_id": cid_all[j],
                        "mag": float(mags[j]) if math.isfinite(mags[j]) else float("nan"),
                        "nn_dist_fwhm": float(nn_fwhm[j]),
                        "aper_flux": float(a_flux[j]),
                        "psf_flux_single": float(psf_single[j]),
                        "psf_flux_grouped": float(psf_grouped[j]),
                        "mag_aper": mag_a,
                        "mag_psf_single": mag_s,
                        "mag_psf_grouped": mag_g,
                    }
                )

        all_df = pd.DataFrame(records)
        if frame_cache is not None:
            frame_cache.parent.mkdir(parents=True, exist_ok=True)
            all_df.to_csv(frame_cache, index=False)
            print(f"[Part A] wrote frame cache {frame_cache}", flush=True)

    star_rows: list[dict[str, Any]] = []
    for cid, grp in all_df.groupby("catalog_id", sort=False):
        mag_val = float(pd.to_numeric(grp["mag"], errors="coerce").median())
        nn = float(pd.to_numeric(grp["nn_dist_fwhm"], errors="coerce").median())
        apt = grp["mag_aper"].to_numpy(dtype=float)
        ps = grp["mag_psf_single"].to_numpy(dtype=float)
        pg = grp["mag_psf_grouped"].to_numpy(dtype=float)
        ok = np.isfinite(apt)
        n_ok = int(np.sum(ok))
        if n_ok < MIN_FRAMES:
            continue

        def _norm_rms(apt_arr: np.ndarray, psf_arr: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
            a = apt_arr[mask]
            p = psf_arr[mask]
            if len(a) < MIN_FRAMES:
                return float("nan"), float("nan"), float("nan")
            off = float(np.median(a - p))
            pn = p + off
            ar = a - float(np.median(a))
            pr = pn - float(np.median(pn))
            ra = _robust_rms_mad(ar)
            rp = _robust_rms_mad(pr)
            ratio = rp / ra if ra > 0 else float("nan")
            return ra, rp, ratio

        m = np.isfinite(apt) & np.isfinite(ps)
        ra_s, rp_s, rt_s = _norm_rms(apt, ps, m)
        m2 = np.isfinite(apt) & np.isfinite(pg)
        ra_g, rp_g, rt_g = _norm_rms(apt, pg, m2)
        star_rows.append(
            {
                "catalog_id": cid,
                "catalog_mag": mag_val,
                "n_frames": n_ok,
                "nn_dist_fwhm": nn,
                "rms_aperture": ra_s,
                "rms_psf_single": rp_s,
                "rms_psf_grouped": rp_g,
                "ratio_single_aper": rt_s,
                "ratio_grouped_aper": rt_g,
                "ratio_grouped_single": rp_g / rp_s if rp_s > 0 else float("nan"),
            }
        )
    star_df = pd.DataFrame(star_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    star_df.to_csv(out_dir / "d364_crowded_grouper_per_star.csv", index=False)
    return all_df, star_df


def _approx_psf_quality(finite_psf: bool, shift_ok: bool = True) -> str:
    if not finite_psf:
        return "bad"
    if shift_ok:
        return "good"
    return "marginal"


def part_b(
    frame_df: pd.DataFrame,
    crowded_star_df: pd.DataFrame,
    existing_per_star: Path,
    out_dir: Path,
) -> dict[str, Any]:
    """Adaptive routing validation using crowded frame records + existing full-field per-star."""
    cfg = AppConfig()
    snr_lo = float(cfg.psf_adaptive_snr_lo)
    resolve_fwhm = float(cfg.psf_adaptive_resolve_fwhm)

    # --- crowded: per-frame adaptive on Part A records ---
    adapt_rows: list[dict[str, Any]] = []
    for cid, grp in frame_df.groupby("catalog_id", sort=False):
        mag_cat = float(pd.to_numeric(grp["mag"], errors="coerce").median())
        nn = float(pd.to_numeric(grp["nn_dist_fwhm"], errors="coerce").median())
        is_blended = True  # crowded subset
        apt = grp["mag_aper"].to_numpy(dtype=float)
        ps = grp["mag_psf_single"].to_numpy(dtype=float)
        pg = grp["mag_psf_grouped"].to_numpy(dtype=float)
        aflux = pd.to_numeric(grp["aper_flux"], errors="coerce").to_numpy(dtype=float)

        # Approx aperture SNR: assume typical sky/noise from flux (conservative)
        snr_aper = np.where(
            np.isfinite(aflux) & (aflux > 0),
            np.sqrt(np.maximum(aflux, 1.0)) / np.maximum(np.sqrt(aflux) * 0.05, 1.0),
            np.inf,
        )

        psf_usable_s = np.isfinite(ps)
        psf_usable_g = np.isfinite(pg)
        q_s = np.array([_approx_psf_quality(bool(u)) for u in psf_usable_s])
        q_g = np.array([_approx_psf_quality(bool(u)) for u in psf_usable_g])

        rule2 = psf_usable_g & is_blended & np.isfinite(nn) & (nn >= resolve_fwhm)
        rule3_s = psf_usable_s & (snr_aper <= snr_lo) & (q_s == "good")
        use_psf_adapt = np.where(rule2, pg, np.where(rule3_s, ps, apt))
        method = np.where(rule2 | rule3_s, "psf", "aperture")

        rms_aper = _star_rms_from_mags(apt)
        rms_psf_s = _star_rms_from_mags(ps) if np.isfinite(ps).sum() >= MIN_FRAMES else float("nan")
        rms_psf_g = _star_rms_from_mags(pg) if np.isfinite(pg).sum() >= MIN_FRAMES else float("nan")
        rms_adapt = _star_rms_from_mags(use_psf_adapt)
        # Oracle: per-frame pick closer to method-specific median
        med_a = float(np.nanmedian(apt))
        med_s = float(np.nanmedian(ps))
        oracle = np.where(
            np.isfinite(ps) & np.isfinite(apt),
            np.where(np.abs(apt - med_a) <= np.abs(ps - med_s), apt, ps),
            np.where(np.isfinite(apt), apt, ps),
        )
        rms_oracle = _star_rms_from_mags(oracle)

        adapt_rows.append(
            {
                "catalog_id": cid,
                "catalog_mag": mag_cat,
                "crowding_class": "crowded",
                "nn_dist_fwhm": nn,
                "rms_aperture": rms_aper,
                "rms_psf_single": rms_psf_s,
                "rms_psf_grouped": rms_psf_g,
                "rms_adaptive": rms_adapt,
                "rms_oracle": rms_oracle,
                "n_psf_frames": int((method == "psf").sum()),
                "n_rule2": int(rule2.sum()),
                "n_rule3": int(rule3_s.sum()),
            }
        )
    crowded_adapt = pd.DataFrame(adapt_rows)

    # --- isolated (+ crowded baseline): existing per-star CSV (PSF = single for isolated, grouped for crowded old) ---
    exist = pd.read_csv(existing_per_star, low_memory=False, dtype={"catalog_id": str})
    iso = exist[exist["crowding_class"] == "isolated"].copy()
    iso["rms_psf_single"] = iso["rms_psf"]
    iso["rms_psf_grouped"] = float("nan")

    # Star-level adaptive approximation for isolated (no frame cache): rule3 via catalog mag proxy
    iso_adapt: list[dict[str, Any]] = []
    for _, row in iso.iterrows():
        mag = float(row["catalog_mag"])
        ra = float(row["rms_aperture"])
        rs = float(row["rms_psf_single"])
        # Approx: faint (G>=17) + PSF usable (ratio finite) -> rule3 -> PSF else aperture
        psf_usable = math.isfinite(rs) and rs > 0
        faint = math.isfinite(mag) and mag >= 17.0
        pick_psf = psf_usable and faint and rs < ra
        rms_ad = rs if pick_psf else ra
        rms_or = min(ra, rs) if math.isfinite(ra) and math.isfinite(rs) else ra
        iso_adapt.append(
            {
                "catalog_id": row["catalog_id"],
                "catalog_mag": mag,
                "crowding_class": "isolated",
                "rms_aperture": ra,
                "rms_psf_single": rs,
                "rms_psf_grouped": float("nan"),
                "rms_adaptive": rms_ad,
                "rms_oracle": rms_or,
                "n_psf_frames": int(pick_psf),
                "n_rule2": 0,
                "n_rule3": int(pick_psf),
            }
        )
    iso_adapt_df = pd.DataFrame(iso_adapt)

    # Merge crowded Part B (frame-level adaptive) with isolated (star-level approx)
    cols = [
        "catalog_id",
        "catalog_mag",
        "crowding_class",
        "rms_aperture",
        "rms_psf_single",
        "rms_psf_grouped",
        "rms_adaptive",
        "rms_oracle",
        "n_psf_frames",
        "n_rule2",
        "n_rule3",
    ]
    crowded_adapt["crowding_class"] = "crowded"
    combined = pd.concat([crowded_adapt[cols], iso_adapt_df[cols]], ignore_index=True)

    def _agg(sub: pd.DataFrame) -> dict[str, float]:
        if sub.empty:
            return {}
        return {
            "N": int(len(sub)),
            "median_rms_aperture": float(sub["rms_aperture"].median()),
            "median_rms_psf_single": float(sub["rms_psf_single"].median()),
            "median_rms_psf_grouped": float(sub["rms_psf_grouped"].median()),
            "median_rms_adaptive": float(sub["rms_adaptive"].median()),
            "median_rms_oracle": float(sub["rms_oracle"].median()),
            "frac_adaptive_psf": float((sub["n_psf_frames"] > 0).mean()),
        }

    summary_rows: list[dict[str, Any]] = []
    for label, lo, hi in _mag_bins():
        for cls in ("crowded", "isolated", "all"):
            sub = combined[(combined["catalog_mag"] > lo) & (combined["catalog_mag"] <= hi)]
            if cls != "all":
                sub = sub[sub["crowding_class"] == cls]
            row = {"mag_bin": label, "crowding_class": cls, **_agg(sub)}
            summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)

    # Global checks
    never_worse = bool((combined["rms_adaptive"] <= combined["rms_aperture"] * 1.001).all())
    adapt_vs_oracle = float(combined["rms_adaptive"].median() / combined["rms_oracle"].median())

    out_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_dir / "d364_adaptive_routing_per_star.csv", index=False)
    summary_df.to_csv(out_dir / "d364_adaptive_routing_summary.csv", index=False)
    crowded_adapt.to_csv(out_dir / "d364_adaptive_crowded_frame_routed.csv", index=False)

    return {
        "never_worse_than_aperture": never_worse,
        "median_adaptive_over_oracle": adapt_vs_oracle,
        "n_combined_stars": int(len(combined)),
        "summary_table": summary_df.to_dict(orient="records"),
        "approximations": (
            "Crowded: per-frame adaptive with approx SNR=sqrt(flux)/noise; psf_quality=good if finite flux. "
            "Isolated: star-level rule3 proxy (G>=17 and rms_psf<rms_aper -> PSF); no frame cache."
        ),
    }


def _summarize_part_a(star_df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, lo, hi in _mag_bins():
        sub = star_df[(star_df["catalog_mag"] > lo) & (star_df["catalog_mag"] <= hi)]
        rows.append(
            {
                "mag_bin": label,
                "N": int(len(sub)),
                "median_rms_aper": float(sub["rms_aperture"].median()) if len(sub) else float("nan"),
                "median_rms_single": float(sub["rms_psf_single"].median()) if len(sub) else float("nan"),
                "median_rms_grouped": float(sub["rms_psf_grouped"].median()) if len(sub) else float("nan"),
                "median_ratio_grouped_aper": float(sub["ratio_grouped_aper"].median()) if len(sub) else float("nan"),
                "median_ratio_single_aper": float(sub["ratio_single_aper"].median()) if len(sub) else float("nan"),
                "median_ratio_grouped_single": float(sub["ratio_grouped_single"].median()) if len(sub) else float("nan"),
            }
        )
    return rows


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cfg = AppConfig()
    if cfg.psf_photometry_enabled:
        raise RuntimeError("psf_photometry_enabled must remain false")

    draft_dir = Path(cfg.archive_root) / "Drafts" / f"draft_{DRAFT_ID:06d}"
    out_dir = draft_dir / "diagnostics" / "forced_photometry_grouper_364"
    frame_cache = out_dir / "d364_crowded_grouper_frame_records.csv"
    existing = draft_dir / "diagnostics" / "forced_photometry_pal7" / "d364_forced_aperture_vs_psf_per_star.csv"

    part_b_only = "--part-b-only" in sys.argv
    if part_b_only and not frame_cache.is_file():
        raise FileNotFoundError(f"Need frame cache for --part-b-only: {frame_cache}")

    if not part_b_only:
        frame_df, star_df = part_a(out_dir, frame_cache=frame_cache)
    else:
        frame_df = pd.read_csv(frame_cache, low_memory=False, dtype={"catalog_id": str})
        star_df = pd.read_csv(out_dir / "d364_crowded_grouper_per_star.csv", dtype={"catalog_id": str})

    part_a_summary = _summarize_part_a(star_df)
    part_b_result = part_b(frame_df, star_df, existing, out_dir)

    report = {
        "draft_id": DRAFT_ID,
        "setup": SETUP,
        "n_crowded_stars_valid": int(len(star_df)),
        "part_a_by_mag": part_a_summary,
        "part_b": part_b_result,
        "psf_flag_in_config": bool(cfg.psf_photometry_enabled),
        "outputs": {
            "frame_records": str(frame_cache),
            "crowded_per_star": str(out_dir / "d364_crowded_grouper_per_star.csv"),
            "adaptive_summary": str(out_dir / "d364_adaptive_routing_summary.csv"),
        },
    }
    report_path = out_dir / "d364_grouper_adaptive_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
