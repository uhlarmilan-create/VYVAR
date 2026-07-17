#!/usr/bin/env python3
"""Weighted LC ceiling test — drafts 361-364 (read-only, standalone).

Measures upper bound on w*PSF + (1-w)*aperture light-curve RMS vs aperture-alone
using relaxed PSF quality (finite positive flux). Does not modify production config
or pipeline.
"""
from __future__ import annotations

import json
import logging
import math
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from photutils.aperture import CircularAnnulus, CircularAperture, aperture_photometry
from photutils.psf import ImagePSF, PSFPhotometry

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401  (repo layout: src_py + dev on sys.path)
_ROOT = _bootstrap.REPO_ROOT
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import AppConfig  # noqa: E402
from database import VyvarDatabase  # noqa: E402
from gaia_catalog_id import normalize_gaia_source_id  # noqa: E402
from photometry_core import (  # noqa: E402
    _aperture_radius_from_snr_table,
    load_snr_aperture_table_from_draft_dir,
)
from psf_photometry import (  # noqa: E402
    _read_plate_scale_arcsec_px_from_fits,
    build_epsf_model,
    get_epsf_fwhm_from_context,
)
import importlib.util

_fp_path = _ROOT / "scripts" / "forced_photometry_pal7.py"
_spec = importlib.util.spec_from_file_location("forced_photometry_pal7", _fp_path)
_fp = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_fp)
_batch_aperture_flux = _fp._batch_aperture_flux
_forced_single_psf = _fp._forced_single_psf
_mag_bins = _fp._mag_bins
_robust_rms_mad = _fp._robust_rms_mad

warnings.filterwarnings("ignore", category=UserWarning, module="photutils")
logging.getLogger("astropy").setLevel(logging.ERROR)

DRAFT_SETUPS: dict[int, str] = {
    361: "NoFilter_60_2",
    362: "NoFilter_60_2",
    363: "L_60_1",
    364: "Luminance_180_2",
}
MIN_FRAMES = 5
MAX_STARS_TOTAL = 800
MAX_STARS_PER_BIN = 120
SAT_FRAC = 0.85
WEIGHTS = np.round(np.arange(0.0, 1.01, 0.1), 1)
RESULT_JSON = _ROOT / "tmp" / "weighted_lc_ceiling_361_364_result.json"


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


def _mag_bins_sample() -> list[tuple[str, float, float]]:
    return [
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


def _sample_stars_ms(ms_csv: Path, *, max_total: int = MAX_STARS_TOTAL) -> pd.DataFrame:
    df = pd.read_csv(ms_csv, low_memory=False)
    df["catalog_id"] = df.get("catalog_id", df.get("name", "")).map(_norm_cid)
    mag_col = "mag" if "mag" in df.columns else "phot_g_mean_mag"
    df["mag"] = pd.to_numeric(df.get(mag_col), errors="coerce")
    df["x"] = pd.to_numeric(df.get("x"), errors="coerce")
    df["y"] = pd.to_numeric(df.get("y"), errors="coerce")
    df = df.loc[
        df["catalog_id"].astype(str).str.len().gt(0)
        & np.isfinite(df["mag"])
        & np.isfinite(df["x"])
        & np.isfinite(df["y"])
    ].copy()
    picked: list[pd.DataFrame] = []
    for _label, lo, hi in _mag_bins_sample():
        sub = df[(df["mag"] > lo) & (df["mag"] <= hi)]
        if sub.empty:
            continue
        n = min(MAX_STARS_PER_BIN, len(sub))
        picked.append(sub.sample(n=n, random_state=42) if len(sub) > n else sub)
    if not picked:
        return df.head(0)
    out = pd.concat(picked, ignore_index=True)
    if len(out) > max_total:
        out = out.sample(n=max_total, random_state=42).reset_index(drop=True)
    return out


def _sample_stars_364_reuse(ps_dir: Path, draft_dir: Path) -> pd.DataFrame | None:
    """Reuse star positions from prior pal7 per-star CSV when available."""
    pal7 = draft_dir / "diagnostics" / "forced_photometry_pal7" / "d364_forced_aperture_vs_psf_per_star.csv"
    if not pal7.is_file():
        return None
    prior = pd.read_csv(pal7, low_memory=False)
    prior["catalog_id"] = prior["catalog_id"].map(_norm_cid)
    prior["catalog_mag"] = pd.to_numeric(prior["catalog_mag"], errors="coerce")
    ms_csv = ps_dir / "masterstars_full_match.csv"
    if ms_csv.is_file():
        ms = pd.read_csv(ms_csv, low_memory=False)
        ms["catalog_id"] = ms.get("catalog_id", ms.get("name", "")).map(_norm_cid)
        ms["x"] = pd.to_numeric(ms.get("x"), errors="coerce")
        ms["y"] = pd.to_numeric(ms.get("y"), errors="coerce")
        xy = ms[["catalog_id", "x", "y"]].drop_duplicates("catalog_id")
        prior = prior.merge(xy, on="catalog_id", how="left")
    else:
        cone = ps_dir / "field_catalog_cone.csv"
        if not cone.is_file():
            return None
        c = pd.read_csv(cone, low_memory=False)
        c["catalog_id"] = c.get("catalog_id", c.get("SOURCE_ID_GAIA", "")).map(_norm_cid)
        with fits.open(ps_dir / "MASTERSTAR.fits", memmap=True) as hd:
            wcs = WCS(hd[0].header)
        ra = pd.to_numeric(c.get("ra_deg", c.get("ra")), errors="coerce")
        de = pd.to_numeric(c.get("dec_deg", c.get("dec")), errors="coerce")
        xp, yp = wcs.all_world2pix(np.column_stack([ra, de]), 0).T
        c = c.assign(x=xp, y=yp)
        prior = prior.merge(c[["catalog_id", "x", "y"]], on="catalog_id", how="left")

    prior = prior.rename(columns={"catalog_mag": "mag"})
    prior = prior.loc[np.isfinite(prior["mag"]) & np.isfinite(prior["x"]) & np.isfinite(prior["y"])].copy()
    picked: list[pd.DataFrame] = []
    for _label, lo, hi in _mag_bins_sample():
        sub = prior[(prior["mag"] > lo) & (prior["mag"] <= hi)]
        if sub.empty:
            continue
        n = min(MAX_STARS_PER_BIN, len(sub))
        picked.append(sub.sample(n=n, random_state=42) if len(sub) > n else sub)
    if not picked:
        return None
    out = pd.concat(picked, ignore_index=True)
    if len(out) > MAX_STARS_TOTAL:
        out = out.sample(n=MAX_STARS_TOTAL, random_state=42).reset_index(drop=True)
    return out[["catalog_id", "mag", "x", "y"]]


def _psf_worker(args: tuple) -> tuple[int, float]:
    j, x, y, data, psf_path, osamp, fit_shape, fwhm_px = args
    psf_data = np.asarray(fits.getdata(psf_path), dtype=np.float64)
    psf_model = ImagePSF(psf_data, oversampling=osamp)
    fit_shape_t = (int(fit_shape[0]), int(fit_shape[1]))
    return j, _forced_single_psf(
        data, x, y, psf_model=psf_model, fit_shape=fit_shape_t, fwhm_px=fwhm_px
    )


def _ensure_epsf(
    *,
    draft_id: int,
    ps_dir: Path,
    ms_fits: Path,
    ms_csv: Path,
    db: VyvarDatabase,
) -> tuple[Path | None, str, str | None]:
    epsf_path = ps_dir / "masterstar_epsf.fits"
    if epsf_path.is_file():
        return epsf_path, "reused", None
    if not ms_fits.is_file() or not ms_csv.is_file():
        return None, "skipped", "MASTERSTAR or masterstars CSV missing"
    try:
        out = build_epsf_model(
            masterstar_fits_path=ms_fits,
            masterstars_csv_path=ms_csv,
            db=db,
            draft_id=draft_id,
        )
        return Path(out), "built", None
    except Exception as exc:  # noqa: BLE001
        return None, "skipped", str(exc)


def _load_frame_cache(cache_path: Path) -> pd.DataFrame | None:
    if not cache_path.is_file():
        return None
    df = pd.read_csv(cache_path, low_memory=False)
    need = {"frame", "catalog_id", "aper_flux", "psf_flux", "mag"}
    if not need.issubset(df.columns):
        return None
    return df


def _run_forced_flux(
    *,
    draft_id: int,
    draft_dir: Path,
    ps_dir: Path,
    setup: str,
    stars: pd.DataFrame,
    epsf_path: Path,
    meta: dict[str, Any],
    cfg: AppConfig,
    db: VyvarDatabase,
    cache_path: Path,
) -> pd.DataFrame:
    cached = _load_frame_cache(cache_path)
    if cached is not None:
        cids = set(stars["catalog_id"].astype(str))
        sub = cached.loc[cached["catalog_id"].astype(str).isin(cids)].copy()
        if not sub.empty:
            print(f"[d{draft_id}] loaded frame cache {cache_path.name} ({len(sub)} rows)", flush=True)
            return sub

    aligned = draft_dir / "detrended_aligned" / "lights" / setup
    frame_files = sorted(aligned.glob("proc_*.fits"))
    fwhm_px = float(meta.get("fwhm_px", 6.0))
    osamp = int(meta.get("oversampling", 2))
    fit_shape = tuple(meta.get("fit_shape", [15, 15]))

    snr_table = load_snr_aperture_table_from_draft_dir(draft_dir) or {}
    fwhm_snr = float(snr_table.get("fwhm_px", fwhm_px))
    ann_in = float(cfg.annulus_inner_fwhm) * fwhm_snr
    ann_out = float(cfg.annulus_outer_fwhm) * fwhm_snr

    sat_limit = 60000.0
    try:
        row = db.fetch_obs_draft_by_id(draft_id) or {}
        eq = db.conn.execute(
            "SELECT SATURATE_ADU FROM EQUIPMENTS WHERE ID=?", (int(row.get("EQUIPMENT_ID") or 3),)
        ).fetchone()
        if eq and eq[0]:
            sat_limit = float(eq[0])
    except Exception:  # noqa: BLE001
        pass

    mags = stars["mag"].to_numpy(dtype=float)
    x_all = stars["x"].to_numpy(dtype=float)
    y_all = stars["y"].to_numpy(dtype=float)
    cid_all = stars["catalog_id"].astype(str).to_numpy()
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

    records: list[dict[str, Any]] = []
    for fi, fpath in enumerate(frame_files):
        print(f"[d{draft_id}] frame {fi + 1}/{len(frame_files)}: {fpath.name}", flush=True)
        with fits.open(fpath, memmap=True) as hd:
            data = np.asarray(hd[0].data, dtype=np.float64)
        a_flux, peaks = _batch_aperture_flux(data, x_all, y_all, r_ap, ann_in=ann_in, ann_out=ann_out)
        sat_mask = peaks > SAT_FRAC * sat_limit
        a_flux[sat_mask] = np.nan

        p_flux = np.full(len(stars), np.nan, dtype=float)
        valid_j = np.where(np.isfinite(a_flux) & (a_flux > 0))[0]
        worker_args = [
            (
                int(j),
                float(x_all[j]),
                float(y_all[j]),
                data,
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
                    if math.isfinite(pf) and pf > 0:
                        p_flux[j] = pf

        for j in range(len(stars)):
            records.append(
                {
                    "frame": fpath.name,
                    "catalog_id": cid_all[j],
                    "mag": float(mags[j]) if math.isfinite(mags[j]) else float("nan"),
                    "aper_flux": float(a_flux[j]),
                    "psf_flux": float(p_flux[j]),
                }
            )

    out = pd.DataFrame(records)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache_path, index=False)
    return out


def _weighted_scan(all_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Per-star w-scan on relative flux; return star table + summary."""
    star_rows: list[dict[str, Any]] = []
    w_cols = [f"rms_w{w:.1f}" for w in WEIGHTS]

    for cid, grp in all_df.groupby("catalog_id", sort=False):
        mag_val = float(pd.to_numeric(grp["mag"], errors="coerce").median())
        a = grp["aper_flux"].to_numpy(dtype=float)
        p = grp["psf_flux"].to_numpy(dtype=float)
        ok = np.isfinite(a) & (a > 0) & np.isfinite(p) & (p > 0)
        if ok.sum() < MIN_FRAMES:
            continue
        a = a[ok]
        p = p[ok]
        med_a = float(np.median(a))
        med_p = float(np.median(p))
        if med_a <= 0 or med_p <= 0:
            continue
        a_rel = a / med_a
        p_rel = p / med_p
        row: dict[str, Any] = {
            "catalog_id": cid,
            "catalog_mag": mag_val,
            "n_frames": int(ok.sum()),
        }
        best_w = 0.0
        best_rms = float("inf")
        for w in WEIGHTS:
            comb = w * p_rel + (1.0 - w) * a_rel
            rms = _robust_rms_mad(comb - float(np.median(comb)))
            row[f"rms_w{w:.1f}"] = rms
            if rms < best_rms:
                best_rms = rms
                best_w = float(w)
        row["w_opt"] = best_w
        row["rms_w_opt"] = best_rms
        row["rms_w0"] = row["rms_w0.0"]
        row["rms_w1"] = row["rms_w1.0"]
        star_rows.append(row)

    star_df = pd.DataFrame(star_rows)
    if star_df.empty:
        return star_df, {"n_stars": 0}

    def _opt_w_overall(df: pd.DataFrame) -> tuple[float, float]:
        med_by_w = []
        for w in WEIGHTS:
            col = f"rms_w{w:.1f}"
            med_by_w.append((w, float(df[col].median())))
        w_opt, rms_opt = min(med_by_w, key=lambda t: t[1])
        return float(w_opt), float(rms_opt)

    w_opt, rms_opt = _opt_w_overall(star_df)
    rms_aper = float(star_df["rms_w0"].median())
    rms_psf = float(star_df["rms_w1"].median())

    mag_rows: list[dict[str, Any]] = []
    for label, lo, hi in _mag_bins():
        sub = star_df[(star_df["catalog_mag"] > lo) & (star_df["catalog_mag"] <= hi)]
        if sub.empty:
            mag_rows.append(
                {
                    "mag_bin": label,
                    "N": 0,
                    "w_opt": float("nan"),
                    "median_rms_w_opt": float("nan"),
                    "median_rms_aperture": float("nan"),
                    "median_rms_psf": float("nan"),
                    "gain_vs_aperture": float("nan"),
                }
            )
            continue
        w_b, r_b = _opt_w_overall(sub)
        r_a = float(sub["rms_w0"].median())
        r_p = float(sub["rms_w1"].median())
        mag_rows.append(
            {
                "mag_bin": label,
                "N": int(len(sub)),
                "w_opt": w_b,
                "median_rms_w_opt": r_b,
                "median_rms_aperture": r_a,
                "median_rms_psf": r_p,
                "gain_vs_aperture": r_b / r_a if r_a > 0 else float("nan"),
                "gain_vs_psf": r_b / r_p if r_p > 0 else float("nan"),
            }
        )

    summary = {
        "n_stars": int(len(star_df)),
        "w_opt_overall": w_opt,
        "median_rms_w_opt": rms_opt,
        "median_rms_aperture": rms_aper,
        "median_rms_psf": rms_psf,
        "gain_vs_aperture": rms_opt / rms_aper if rms_aper > 0 else float("nan"),
        "gain_vs_psf": rms_opt / rms_psf if rms_psf > 0 else float("nan"),
        "mag_bins": mag_rows,
    }
    return star_df, summary


def _discover_setup(
    draft_id: int,
    draft_dir: Path,
    ps_dir: Path,
    setup: str,
    db: VyvarDatabase,
) -> dict[str, Any]:
    ms_fits = ps_dir / "MASTERSTAR.fits"
    meta_path = ps_dir / "masterstar_epsf_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}

    aligned = draft_dir / "detrended_aligned" / "lights" / setup
    n_frames = len(list(aligned.glob("proc_*.fits")))

    plate_scale = meta.get("plate_scale_arcsec_px")
    if plate_scale is None and ms_fits.is_file():
        try:
            plate_scale = _read_plate_scale_arcsec_px_from_fits(ms_fits)
        except Exception:  # noqa: BLE001
            plate_scale = float("nan")
    fwhm_px = float(meta.get("fwhm_px", float("nan")))
    if not math.isfinite(fwhm_px) and ms_fits.is_file():
        try:
            fwhm_px = float(get_epsf_fwhm_from_context(ms_fits, db, draft_id))
        except Exception:  # noqa: BLE001
            fwhm_px = float("nan")

    equip = db.fetch_obs_draft_telescope_equipment(draft_id) or {}
    rig = equip.get("equipment_name") or equip.get("telescope_name") or setup

    return {
        "draft_id": draft_id,
        "setup": setup,
        "rig": str(rig),
        "telescope": equip.get("telescope_name"),
        "equipment": equip.get("equipment_name"),
        "plate_scale_arcsec_px": float(plate_scale) if plate_scale is not None else float("nan"),
        "fwhm_px": fwhm_px,
        "n_aligned_frames": n_frames,
        "masterstar_fits": str(ms_fits),
        "epsf_meta": meta,
    }


def run_draft(draft_id: int, *, cfg: AppConfig, db: VyvarDatabase) -> dict[str, Any]:
    setup = DRAFT_SETUPS[draft_id]
    draft_dir = Path(cfg.archive_root) / "Drafts" / f"draft_{draft_id:06d}"
    ps_dir = draft_dir / "platesolve" / setup
    out_dir = draft_dir / "diagnostics" / "weighted_lc_ceiling"
    out_dir.mkdir(parents=True, exist_ok=True)

    info = _discover_setup(draft_id, draft_dir, ps_dir, setup, db)
    result: dict[str, Any] = {"status": "ok", **info}

    if info["n_aligned_frames"] < MIN_FRAMES:
        result["status"] = "skipped"
        result["skip_reason"] = f"fewer than {MIN_FRAMES} aligned frames"
        return result

    ms_fits = Path(info["masterstar_fits"])
    ms_csv = ps_dir / "masterstars_full_match.csv"
    epsf_path, epsf_source, epsf_err = _ensure_epsf(
        draft_id=draft_id,
        ps_dir=ps_dir,
        ms_fits=ms_fits,
        ms_csv=ms_csv,
        db=db,
    )
    result["epsf_source"] = epsf_source
    result["epsf_path"] = str(epsf_path) if epsf_path else None
    if epsf_path is None:
        result["status"] = "skipped"
        result["skip_reason"] = epsf_err or "ePSF unavailable"
        return result

    meta_path = ps_dir / "masterstar_epsf_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else info["epsf_meta"]

    if draft_id == 364:
        stars = _sample_stars_364_reuse(ps_dir, draft_dir)
        star_source = "364_pal7_per_star_reuse"
        if stars is None or stars.empty:
            stars = _sample_stars_ms(ms_csv)
            star_source = "masterstars_full_match"
    else:
        stars = _sample_stars_ms(ms_csv)
        star_source = "masterstars_full_match"
    result["star_sample_source"] = star_source
    result["n_sample_stars"] = int(len(stars))
    if stars.empty:
        result["status"] = "skipped"
        result["skip_reason"] = "no sample stars"
        return result

    cache_path = out_dir / f"d{draft_id}_frame_flux_records.csv"
    all_df = _run_forced_flux(
        draft_id=draft_id,
        draft_dir=draft_dir,
        ps_dir=ps_dir,
        setup=setup,
        stars=stars,
        epsf_path=epsf_path,
        meta=meta,
        cfg=cfg,
        db=db,
        cache_path=cache_path,
    )
    result["frame_cache"] = str(cache_path)

    star_df, w_summary = _weighted_scan(all_df)
    result.update(w_summary)
    if star_df.empty:
        result["status"] = "skipped"
        result["skip_reason"] = f"fewer than {MIN_FRAMES} valid frames per star after forced photometry"
        return result

    star_csv = out_dir / f"d{draft_id}_weighted_lc_per_star.csv"
    mag_csv = out_dir / f"d{draft_id}_weighted_lc_by_mag.csv"
    star_df.to_csv(star_csv, index=False)
    pd.DataFrame(w_summary["mag_bins"]).to_csv(mag_csv, index=False)
    result["per_star_csv"] = str(star_csv)
    result["mag_bin_csv"] = str(mag_csv)
    return result


def _format_report(results: list[dict[str, Any]], *, psf_flags: dict[str, bool]) -> str:
    lines = [
        "WEIGHTED LC CEILING — drafts 361-364",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "Standalone read-only; relaxed PSF quality (finite flux) = upper bound.",
        f"Config PSF flags: {psf_flags}",
        "",
    ]
    cross: list[dict[str, Any]] = []

    for r in results:
        did = r["draft_id"]
        lines.append(f"=== Draft {did} ({r.get('setup', '?')}) ===")
        lines.append(
            f"  Rig: {r.get('rig')} | plate scale: {r.get('plate_scale_arcsec_px', float('nan')):.4f} \"/px"
            f" | FWHM: {r.get('fwhm_px', float('nan')):.3f} px | frames: {r.get('n_aligned_frames')}"
        )
        lines.append(f"  MASTERSTAR: {r.get('masterstar_fits')}")
        lines.append(f"  ePSF: {r.get('epsf_source')} ({r.get('epsf_path')})")
        if r.get("status") == "skipped":
            lines.append(f"  SKIPPED: {r.get('skip_reason')}")
            lines.append("")
            cross.append(
                {
                    "draft": did,
                    "setup": r.get("setup"),
                    "plate_scale": r.get("plate_scale_arcsec_px"),
                    "fwhm_px": r.get("fwhm_px"),
                    "status": "skipped",
                }
            )
            continue
        lines.append(f"  Stars sampled: {r.get('n_sample_stars')} ({r.get('star_sample_source')})")
        lines.append(f"  Valid stars (>=5 frames): {r.get('n_stars')}")
        lines.append(
            f"  w_opt={r.get('w_opt_overall'):.1f} | RMS w_opt={r.get('median_rms_w_opt'):.5f}"
            f" | aperture={r.get('median_rms_aperture'):.5f} | PSF={r.get('median_rms_psf'):.5f}"
        )
        g = r.get("gain_vs_aperture")
        lines.append(
            f"  Gain vs aperture (ratio w_opt/aper): {g:.4f} ({(1 - g) * 100:.1f}% reduction)"
            if math.isfinite(g)
            else "  Gain vs aperture: n/a"
        )
        lines.append("  By mag bin:")
        for mb in r.get("mag_bins", []):
            if mb.get("N", 0) == 0:
                continue
            lines.append(
                f"    {mb['mag_bin']:>6} N={mb['N']:3d} w_opt={mb['w_opt']:.1f}"
                f" RMS_opt={mb['median_rms_w_opt']:.5f} aper={mb['median_rms_aperture']:.5f}"
                f" gain={mb.get('gain_vs_aperture', float('nan')):.3f}"
            )
        lines.append("")
        cross.append(
            {
                "draft": did,
                "setup": r.get("setup"),
                "plate_scale": r.get("plate_scale_arcsec_px"),
                "fwhm_px": r.get("fwhm_px"),
                "w_opt": r.get("w_opt_overall"),
                "gain_vs_aperture": r.get("gain_vs_aperture"),
                "n_stars": r.get("n_stars"),
                "status": "ok",
            }
        )

    lines.append("=== CROSS-DRAFT ===")
    lines.append("draft | setup | \"/px | FWHM px | w_opt | gain vs aper | N stars")
    for c in cross:
        if c.get("status") == "skipped":
            lines.append(
                f"  {c['draft']:3d} | {c.get('setup','?'):16s} | "
                f"{c.get('plate_scale', float('nan')):6.3f} | {c.get('fwhm_px', float('nan')):6.2f} | SKIP"
            )
        else:
            lines.append(
                f"  {c['draft']:3d} | {c.get('setup','?'):16s} | "
                f"{c.get('plate_scale', float('nan')):6.3f} | {c.get('fwhm_px', float('nan')):6.2f} | "
                f"{c.get('w_opt', float('nan')):4.1f} | {c.get('gain_vs_aperture', float('nan')):6.4f} | "
                f"{c.get('n_stars', 0)}"
            )
    lines.append("")
    lines.append("No production/config changes; PSF flags unchanged.")
    return "\n".join(lines)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cfg = AppConfig()
    psf_flags = {
        "psf_photometry_enabled": bool(cfg.psf_photometry_enabled),
        "psf_adaptive_enabled": bool(cfg.psf_adaptive_enabled),
    }
    if psf_flags["psf_photometry_enabled"]:
        raise RuntimeError("psf_photometry_enabled must be false for this standalone test")

    db = VyvarDatabase(cfg.database_path)
    results: list[dict[str, Any]] = []
    for draft_id in (361, 362, 363, 364):
        print(f"\n========== draft {draft_id} ==========", flush=True)
        results.append(run_draft(draft_id, cfg=cfg, db=db))

    report = _format_report(results, psf_flags=psf_flags)
    report_path = _ROOT / "tmp" / "weighted_lc_ceiling_361_364_report.txt"
    report_path.write_text(report, encoding="utf-8")

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "psf_flags": psf_flags,
        "drafts": results,
        "report_path": str(report_path),
    }
    RESULT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\n" + report)
    print(f"\nWrote {report_path}")
    print(f"Wrote {RESULT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
