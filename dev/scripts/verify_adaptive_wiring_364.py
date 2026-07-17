#!/usr/bin/env python3
"""End-to-end verification: adaptive PSF wiring on draft 364 Luminance_180_2.

Backfills LC-star PSF on proc CSVs, generates crowding_targets.csv, measures
coverage / routing / aggregate comp RMS vs aperture-only. Restores config flags
and proc CSV backups when done.
"""
from __future__ import annotations

import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401  (repo layout: src_py + dev on sys.path)
_ROOT = _bootstrap.REPO_ROOT
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import AppConfig  # noqa: E402
from crowding_index import ensure_crowding_targets_for_lc  # noqa: E402
from database import VyvarDatabase, get_gaia_db_max_g_mag  # noqa: E402
from gaia_catalog_id import normalize_gaia_source_id  # noqa: E402
from photometry_core import (  # noqa: E402
    _load_adaptive_blend_map,
    compute_lc_flux_method,
)
from pipeline import (  # noqa: E402
    _epsf_fit_catalog_ids,
    _epsf_target_catalog_ids,
    _export_catalog_psf_st_fields,
    _fill_psf_catalog_columns,
)

DRAFT_ID = 364
SETUP = "Luminance_180_2"
MAD_SCALE = 1.4826
CONFIG_PATH = _ROOT / "config.json"


def _robust_rms_mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    return float(MAD_SCALE * np.median(np.abs(x - np.median(x))))


def _mag_bins() -> list[tuple[str, float, float]]:
    return [
        ("G13-14", 13.0, 14.0),
        ("G14-15", 14.0, 15.0),
        ("G15-16", 15.0, 16.0),
        ("G16-17", 16.0, 17.0),
        ("G17-18", 17.0, 18.0),
        ("G18-19", 18.0, 19.0),
        ("G19-20", 19.0, 20.0),
    ]


def _norm_id(raw: str) -> str:
    try:
        return str(normalize_gaia_source_id(str(raw).strip())).strip()
    except Exception:  # noqa: BLE001
        return str(raw).strip()


def _backfill_psf(
    proc_dir: Path,
    ps_dir: Path,
    cfg: AppConfig,
    *,
    target_ids: set[str],
) -> float:
    st = _export_catalog_psf_st_fields(cfg, ps_dir)
    st["platesolve_dir"] = str(ps_dir)
    t0 = time.perf_counter()
    for csv_path in sorted(proc_dir.glob("proc_*.csv")):
        fits_path = csv_path.with_suffix(".fits")
        if not fits_path.is_file():
            continue
        df = pd.read_csv(csv_path, low_memory=False, dtype={"catalog_id": str})
        with fits.open(fits_path, memmap=True) as hd:
            data = hd[0].data
            hdr = hd[0].header
        df = _fill_psf_catalog_columns(df, data, hdr, st, target_ids=target_ids)
        df.to_csv(csv_path, index=False)
    return time.perf_counter() - t0


def _lc_ids_from_csvs(ps_dir: Path) -> set[str]:
    ids = _epsf_fit_catalog_ids(ps_dir, psf_photometry_enabled=True)
    return ids or set()


def _measure_coverage(proc_dir: Path, lc_ids: set[str]) -> dict[str, float]:
    lc_norm = {_norm_id(x) for x in lc_ids}
    n_lc_rows = 0
    n_psf_ok = 0
    n_psf_flux = 0
    for csv_path in sorted(proc_dir.glob("proc_*.csv")):
        df = pd.read_csv(
            csv_path,
            usecols=lambda c: c in ("catalog_id", "psf_flux", "psf_fit_ok"),
            low_memory=False,
            dtype={"catalog_id": str},
        )
        df["_cid"] = df["catalog_id"].map(_norm_id)
        sub = df[df["_cid"].isin(lc_norm)]
        n_lc_rows += len(sub)
        ok = sub["psf_fit_ok"].fillna(False).astype(bool)
        pf = pd.to_numeric(sub["psf_flux"], errors="coerce")
        n_psf_ok += int(ok.sum())
        n_psf_flux += int((ok & np.isfinite(pf) & (pf > 0)).sum())
    return {
        "n_lc_rows": n_lc_rows,
        "frac_psf_fit_ok": n_psf_ok / n_lc_rows if n_lc_rows else float("nan"),
        "frac_psf_flux_ok": n_psf_flux / n_lc_rows if n_lc_rows else float("nan"),
    }


def _build_lc_frame_table(proc_dir: Path, lc_ids: set[str]) -> pd.DataFrame:
    lc_norm = {_norm_id(x) for x in lc_ids}
    rows: list[dict] = []
    usecols = [
        "catalog_id",
        "dao_flux",
        "psf_flux",
        "psf_fit_ok",
        "psf_quality",
        "mag",
        "err",
    ]
    for csv_path in sorted(proc_dir.glob("proc_*.csv")):
        hdr = pd.read_csv(csv_path, nrows=0)
        cols = [c for c in usecols if c in hdr.columns]
        df = pd.read_csv(csv_path, usecols=cols, low_memory=False, dtype={"catalog_id": str})
        df["_cid"] = df["catalog_id"].map(_norm_id)
        sub = df[df["_cid"].isin(lc_norm)].copy()
        if sub.empty:
            continue
        flux = pd.to_numeric(sub["dao_flux"], errors="coerce")
        sub["mag_inst"] = np.where(flux > 0, -2.5 * np.log10(flux), np.nan)
        sub["frame"] = csv_path.name
        rows.append(sub)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _routing_and_rms(
    all_frames: pd.DataFrame,
    blend_map: dict,
    cfg: AppConfig,
    crowd_csv: Path,
) -> dict:
    crowd = pd.read_csv(crowd_csv, low_memory=False, dtype={"catalog_id": str})
    crowd["catalog_id"] = crowd["catalog_id"].map(_norm_id)
    nn_map = dict(zip(crowd["catalog_id"], pd.to_numeric(crowd["nn_dist_fwhm"], errors="coerce")))
    blended_map = dict(zip(crowd["catalog_id"], crowd["is_blended"].fillna(False).astype(bool)))

    all_frames = all_frames.copy()
    all_frames["catalog_id"] = all_frames["_cid"]
    methods = compute_lc_flux_method(
        all_frames,
        blend_map,
        resolve_fwhm=float(cfg.psf_adaptive_resolve_fwhm),
        snr_lo=float(cfg.psf_adaptive_snr_lo),
    )
    all_frames["lc_flux_method"] = methods

    n_psf = int((methods == "psf").sum())
    n_total = len(methods)

    # Per-star RMS: aperture (mag_inst) vs adaptive
    star_rms_aper: list[float] = []
    star_rms_adapt: list[float] = []
    per_star: list[dict] = []
    for cid, grp in all_frames.groupby("_cid", sort=False):
        mag_aper = grp["mag_inst"].to_numpy(dtype=float)
        pf = pd.to_numeric(grp["psf_flux"], errors="coerce").to_numpy(dtype=float)
        use_psf = (grp["lc_flux_method"].astype(str).to_numpy() == "psf") & np.isfinite(pf) & (pf > 0)
        mag_psf = np.where(use_psf, -2.5 * np.log10(pf), np.nan)
        mag_ad = np.where(np.isfinite(mag_psf), mag_psf, mag_aper)
        ra = _robust_rms_mad(mag_aper - np.nanmedian(mag_aper))
        rd = _robust_rms_mad(mag_ad - np.nanmedian(mag_ad))
        if math.isfinite(ra):
            star_rms_aper.append(ra)
        if math.isfinite(rd):
            star_rms_adapt.append(rd)
        mag_cat = float(pd.to_numeric(grp.get("mag", pd.Series([float("nan")])), errors="coerce").median())
        nn = float(nn_map.get(cid, float("nan")))
        is_iso = not bool(blended_map.get(cid, False))
        per_star.append(
            {
                "catalog_id": cid,
                "mag": mag_cat,
                "nn_dist_fwhm": nn,
                "crowding": "isolated" if is_iso else "crowded",
                "rms_aperture": ra,
                "rms_adaptive": rd,
                "frac_psf_frames": float(use_psf.mean()) if len(use_psf) else 0.0,
            }
        )

    psf_df = pd.DataFrame(per_star)
    routing_rows: list[dict] = []
    for label, lo, hi in _mag_bins():
        for cls in ("isolated", "crowded", "all"):
            sub = psf_df[(psf_df["mag"] > lo) & (psf_df["mag"] <= hi)]
            if cls != "all":
                sub = sub[sub["crowding"] == cls]
            if sub.empty:
                continue
            routing_rows.append(
                {
                    "mag_bin": label,
                    "crowding": cls,
                    "N": len(sub),
                    "frac_psf_routed": float(sub["frac_psf_frames"].mean()),
                    "median_rms_aper": float(sub["rms_aperture"].median()),
                    "median_rms_adaptive": float(sub["rms_adaptive"].median()),
                }
            )

    return {
        "n_frame_rows": n_total,
        "n_psf_routed": n_psf,
        "frac_psf_routed": n_psf / n_total if n_total else 0.0,
        "median_comp_rms_aperture": float(np.median(star_rms_aper)) if star_rms_aper else float("nan"),
        "median_comp_rms_adaptive": float(np.median(star_rms_adapt)) if star_rms_adapt else float("nan"),
        "routing_by_mag_crowding": routing_rows,
        "per_star": psf_df.to_dict(orient="records"),
    }


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cfg = AppConfig()
    draft_dir = Path(cfg.archive_root) / "Drafts" / f"draft_{DRAFT_ID:06d}"
    ps_dir = draft_dir / "platesolve" / SETUP
    proc_dir = draft_dir / "detrended_aligned" / "lights" / SETUP
    out_dir = draft_dir / "diagnostics" / "adaptive_wiring_verify_364"
    out_dir.mkdir(parents=True, exist_ok=True)

    orig_cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    orig_psf = bool(orig_cfg.get("psf_photometry_enabled", False))
    orig_adapt = bool(orig_cfg.get("psf_adaptive_enabled", False))

    report: dict = {"draft_id": DRAFT_ID, "setup": SETUP}

    # Coverage before backfill
    lc_ids = _lc_ids_from_csvs(ps_dir)
    target_ids = _epsf_target_catalog_ids(ps_dir) or set()
    report["n_lc_ids"] = len(lc_ids)
    report["n_target_only_ids"] = len(target_ids)
    report["coverage_before"] = _measure_coverage(proc_dir, lc_ids)

    # Enable flags in memory + config for this draft run only
    data = dict(orig_cfg)
    data["psf_photometry_enabled"] = True
    data["psf_adaptive_enabled"] = True
    CONFIG_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    cfg.psf_photometry_enabled = True
    cfg.psf_adaptive_enabled = True

    backup_dir = out_dir / "proc_csv_backup"
    if backup_dir.is_dir():
        shutil.rmtree(backup_dir)
    backup_dir.mkdir(parents=True)
    for p in sorted(proc_dir.glob("proc_*.csv")):
        shutil.copy2(p, backup_dir / p.name)

    db = VyvarDatabase(cfg.database_path)
    gaia_max = float(get_gaia_db_max_g_mag(cfg.gaia_db_path))

    try:
        # Runtime: target-only vs full LC PSF backfill
        t_target = _backfill_psf(proc_dir, ps_dir, cfg, target_ids=target_ids)
        cov_target = _measure_coverage(proc_dir, lc_ids)
        # Restore and backfill LC set
        for p in sorted(backup_dir.glob("proc_*.csv")):
            shutil.copy2(p, proc_dir / p.name)
        t_lc = _backfill_psf(proc_dir, ps_dir, cfg, target_ids=lc_ids)
        cov_lc = _measure_coverage(proc_dir, lc_ids)

        crowd_path = ensure_crowding_targets_for_lc(
            draft_dir,
            SETUP,
            db,
            DRAFT_ID,
            gaia_db_max_g=gaia_max,
            force=True,
        )
        assert crowd_path is not None
        crowd_df = pd.read_csv(crowd_path, low_memory=False, dtype={"catalog_id": str})
        n_blended = int(crowd_df["is_blended"].fillna(False).astype(bool).sum()) if "is_blended" in crowd_df.columns else 0

        blend_map = _load_adaptive_blend_map(ps_dir / "MASTERSTAR.fits")
        all_frames = _build_lc_frame_table(proc_dir, lc_ids)
        metrics = _routing_and_rms(all_frames, blend_map, cfg, crowd_path)

        report.update(
            {
                "coverage_after_lc_backfill": cov_lc,
                "coverage_after_target_only_timing": cov_target,
                "runtime_sec_target_only_psf": round(t_target, 2),
                "runtime_sec_lc_psf": round(t_lc, 2),
                "crowding_targets_csv": str(crowd_path),
                "crowding_n_rows": len(crowd_df),
                "crowding_n_blended": n_blended,
                **metrics,
            }
        )
    finally:
        # Restore proc CSVs from backup
        if backup_dir.is_dir():
            for p in sorted(backup_dir.glob("proc_*.csv")):
                shutil.copy2(p, proc_dir / p.name)
        data["psf_photometry_enabled"] = orig_psf
        data["psf_adaptive_enabled"] = orig_adapt
        CONFIG_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    report["flags_restored"] = {
        "psf_photometry_enabled": orig_psf,
        "psf_adaptive_enabled": orig_adapt,
    }
    out_json = out_dir / "verify_adaptive_wiring_364.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nWrote {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
