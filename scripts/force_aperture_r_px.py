#!/usr/bin/env python3
"""Force fixed circular aperture radius in VYVAR (proc CSV + Phase 2A).

Re-measures flux in every ``proc_*.csv`` from aligned FITS with one radius for all
stars (e.g. AIJ Multi-Aperture r=7 px), then optionally re-runs Phase 2A.

Usage:
  python scripts/force_aperture_r_px.py --draft draft_000312 --radius 7
  python scripts/force_aperture_r_px.py --draft draft_000312 --radius 7 --export-only
  python scripts/force_aperture_r_px.py --draft draft_000312 --radius 7 --phase2a-only

Temporary diagnostic — promote to pipeline flag when stable.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import AppConfig
from gaia_catalog_id import GAIA_PROC_CSV_READ_DTYPE, normalize_gaia_source_id
from photometry_core import (
    compute_fwhm_gaussian_for_aperture_catalog,
    enhance_catalog_dataframe_aperture_bpm,
    run_phase2a,
)
from ui_aperture_photometry import _load_fwhm

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)

TARGETS = {
    "BO CVn": "1498613634033133184",
    "FW CVn": "1497343732462852864",
}


def _fixed_snr_table(r_px: float, fwhm_px: float) -> dict:
    """SNR table with every mag bin clamped to the same radius."""
    r = float(r_px)
    table = {round(float(m), 1): r for m in np.arange(7.0, 18.5, 0.5)}
    return {
        "table": table,
        "fwhm_px": float(fwhm_px),
        "sky_adu_per_px": 0.0,
        "gain": 3.17,
        "read_noise": 7.6,
        "r_min_px": r,
        "r_max_px": r,
        "fixed_radius_px": r,
        "note": "force_aperture_r_px.py — uniform radius for all magnitudes",
    }


def _read_summary_row(summary: Path, cid: str) -> dict | None:
    if not summary.is_file():
        return None
    df = pd.read_csv(summary, dtype={"catalog_id": str})
    row = df[df["catalog_id"].astype(str) == cid]
    if row.empty:
        return None
    r = row.iloc[0]
    return {
        "lc_rms": float(r["lc_rms"]),
        "aperture_px": float(r.get("aperture_px", float("nan"))),
        "am_detrended": bool(r.get("am_detrended", False)),
    }


def _print_targets(label: str, summary: Path) -> None:
    print(f"\n[{label}]")
    for name, cid in TARGETS.items():
        d = _read_summary_row(summary, cid)
        if d is None:
            print(f"  {name}: (not in summary)")
            continue
        print(
            f"  {name}: lc_rms={d['lc_rms']:.4f}  ap={d['aperture_px']:.3f}  "
            f"am_detrended={d['am_detrended']}"
        )


def _photometry_catalog_ids(phot_dir: Path) -> set[str]:
    """Stars that Phase 2A reads from proc CSV (targets + comparison pool)."""
    cids: set[str] = set()
    at = phot_dir / "active_targets.csv"
    if at.is_file():
        df = pd.read_csv(at, dtype={"catalog_id": str})
        for v in df.get("catalog_id", pd.Series(dtype=str)):
            k = normalize_gaia_source_id(v)
            if k:
                cids.add(k)
    cs = phot_dir / "comparison_stars_per_target.csv"
    if cs.is_file():
        df = pd.read_csv(cs, dtype={"catalog_id": str, "target_catalog_id": str})
        for col in ("catalog_id", "target_catalog_id"):
            if col in df.columns:
                for v in df[col]:
                    k = normalize_gaia_source_id(v)
                    if k:
                        cids.add(k)
    return cids


def remeasure_proc_csvs(
    aligned_dir: Path,
    *,
    r_px: float,
    cfg: AppConfig,
    snr_table: dict,
    fwhm_px: float,
    needed_cids: set[str],
    backup: bool,
) -> int:
    """Re-run aperture photometry on existing proc CSV + FITS pairs."""
    csv_files = sorted(aligned_dir.glob("proc_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No proc_*.csv in {aligned_dir}")

    if backup:
        bak = aligned_dir / f"_backup_proc_csv_before_r{int(r_px) if r_px == int(r_px) else r_px}"
        if not bak.is_dir():
            LOGGER.info("Backing up proc CSV to %s", bak)
            bak.mkdir(parents=True, exist_ok=True)
            for p in csv_files:
                shutil.copy2(p, bak / p.name)

    n_ok = 0
    t0 = time.time()
    for i, csv_path in enumerate(csv_files, 1):
        fits_path = aligned_dir / f"{csv_path.stem}.fits"
        if not fits_path.is_file():
            LOGGER.warning("Skip %s — missing %s", csv_path.name, fits_path.name)
            continue
        df = pd.read_csv(csv_path, low_memory=False, dtype=GAIA_PROC_CSV_READ_DTYPE)
        with fits.open(fits_path, memmap=True) as hdul:
            data = hdul[0].data
            hdr = hdul[0].header
        if "catalog_id" not in df.columns:
            LOGGER.warning("Skip %s — no catalog_id column", csv_path.name)
            continue
        cid_norm = df["catalog_id"].map(normalize_gaia_source_id)
        mask = cid_norm.isin(needed_cids)
        if not bool(mask.any()):
            n_ok += 1
            continue
        sub = df.loc[mask].copy()
        arr = np.asarray(data, dtype=np.float32)
        _, _, fw_frame = compute_fwhm_gaussian_for_aperture_catalog(
            sub,
            arr,
            hdr,
            gaussian_fwhm_px_override=None,
            aperture_fwhm_factor=float(cfg.aperture_fwhm_factor),
        )
        fw_use = float(fw_frame) if math.isfinite(float(fw_frame)) and float(fw_frame) > 0 else float(
            snr_table.get("fwhm_px", fwhm_px)
        )
        # photutils here needs scalar r (not per-star array); fixed r_px via factor × FWHM.
        apt_factor = float(r_px) / fw_use
        sub_out = enhance_catalog_dataframe_aperture_bpm(
            sub,
            data,
            hdr,
            aperture_enabled=True,
            aperture_fwhm_factor=apt_factor,
            annulus_inner_fwhm=float(cfg.annulus_inner_fwhm),
            annulus_outer_fwhm=float(cfg.annulus_outer_fwhm),
            nonlinearity_peak_percentile=float(cfg.nonlinearity_peak_percentile),
            nonlinearity_fwhm_ratio=float(cfg.nonlinearity_fwhm_ratio),
            master_dark_path=None,
            snr_aperture_table=None,
        )
        out = df.copy()
        for col in ("flux", "dao_flux", "noise_floor_adu"):
            if col in sub_out.columns:
                out.loc[mask, col] = sub_out[col].to_numpy()
        if "aperture_r_px" in sub_out.columns:
            out.loc[mask, "aperture_r_px"] = float(r_px)
        out.to_csv(csv_path, index=False)
        n_ok += 1
        if i % 25 == 0 or i == len(csv_files):
            LOGGER.info("Re-measured %d/%d (%.1fs)", i, len(csv_files), time.time() - t0)
    return n_ok


def run_phase2a_for_draft(
    draft: Path,
    setup: str,
    *,
    r_px: float,
    cfg: AppConfig,
    fwhm_px: float,
) -> None:
    ps = draft / "platesolve" / setup
    aligned = draft / "detrended_aligned" / "lights" / setup
    phot = ps / "photometry"
    summary = phot / "photometry_summary.csv"
    backup = phot / f"photometry_summary_before_force_r{int(r_px) if r_px == int(r_px) else r_px}px.csv"
    if summary.is_file():
        shutil.copy2(summary, backup)
        LOGGER.info("Summary backup: %s", backup)

    run_phase2a(
        masterstar_fits_path=ps / "MASTERSTAR.fits",
        active_targets_csv=phot / "active_targets.csv",
        comparison_stars_csv=phot / "comparison_stars_per_target.csv",
        per_frame_csv_dir=aligned,
        detrended_aligned_dir=aligned,
        output_dir=phot,
        fwhm_px=float(fwhm_px),
        cfg=cfg,
        force_aperture_px=float(r_px),
        draft_id=int(draft.name.split("_")[-1]) if "_" in draft.name else None,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Force fixed aperture radius in VYVAR.")
    ap.add_argument("--draft", type=Path, default=Path("Archive/Drafts/draft_000312"))
    ap.add_argument("--setup", default="NoFilter_60_2")
    ap.add_argument("--radius", type=float, default=7.0, help="Circular aperture radius [px]")
    ap.add_argument("--export-only", action="store_true", help="Only re-measure proc_*.csv")
    ap.add_argument("--phase2a-only", action="store_true", help="Only Phase 2A (CSV unchanged)")
    ap.add_argument("--no-backup", action="store_true", help="Do not backup proc CSV before rewrite")
    args = ap.parse_args()

    draft = (_ROOT / args.draft).resolve() if not args.draft.is_absolute() else args.draft.resolve()
    setup = str(args.setup)
    r_px = float(args.radius)
    if r_px <= 0:
        raise SystemExit("--radius must be > 0")

    ps = draft / "platesolve" / setup
    aligned = draft / "detrended_aligned" / "lights" / setup
    phot = ps / "photometry"
    summary = phot / "photometry_summary.csv"
    ms = ps / "MASTERSTAR.fits"

    if not aligned.is_dir():
        raise SystemExit(f"Missing aligned dir: {aligned}")
    if not ms.is_file():
        raise SystemExit(f"Missing MASTERSTAR: {ms}")

    cfg = AppConfig(project_root=_ROOT)
    fwhm_px = float(_load_fwhm(ms))
    snr_table = _fixed_snr_table(r_px, fwhm_px)

    snr_path = draft / "aperture_snr_table.json"
    with snr_path.open("w", encoding="utf-8") as f:
        json.dump(snr_table, f, indent=2)
    LOGGER.info("Wrote fixed SNR table: %s (r=%.3f px)", snr_path, r_px)

    print("=" * 60)
    print(f"Force aperture r={r_px:.3f} px  draft={draft.name}  setup={setup}")
    print(f"FWHM={fwhm_px:.3f} px  annulus={cfg.annulus_inner_fwhm:.2f}–{cfg.annulus_outer_fwhm:.2f}×FWHM")
    print("=" * 60)

    _print_targets("BEFORE", summary)

    do_export = not args.phase2a_only
    do_p2a = not args.export_only

    needed_cids = _photometry_catalog_ids(phot)
    LOGGER.info("Re-measure flux for %d catalog_ids (targets + comps)", len(needed_cids))

    if do_export:
        n = remeasure_proc_csvs(
            aligned,
            r_px=r_px,
            cfg=cfg,
            snr_table=snr_table,
            fwhm_px=fwhm_px,
            needed_cids=needed_cids,
            backup=not args.no_backup,
        )
        LOGGER.info("Re-measured %d proc CSV files in %s", n, aligned)

    if do_p2a:
        t0 = time.time()
        run_phase2a_for_draft(draft, setup, r_px=r_px, cfg=cfg, fwhm_px=fwhm_px)
        LOGGER.info("Phase 2A done in %.1fs", time.time() - t0)

    _print_targets("AFTER", summary)
    print(f"\nLC files: {phot / 'lightcurves'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
