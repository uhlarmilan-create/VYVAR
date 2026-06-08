"""Test: phase2a_airmass_before_outlier=True vs False for BO CVn / FW CVn.

Usage: python scripts/test_airmass_order.py

Temporary diagnostic — do not commit.
"""
from __future__ import annotations

import logging
import os
import shutil
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

logging.basicConfig(level=logging.WARNING)

from config import AppConfig
from photometry_core import run_phase2a
from ui_aperture_photometry import _load_fwhm

DRAFT_DIR = Path(r"C:\ASTRO\python\VYVAR\Archive\Drafts\draft_000311")
SETUP = "NoFilter_60_2"
PHOT_DIR = DRAFT_DIR / "platesolve" / SETUP / "photometry"
PROC_DIR = DRAFT_DIR / "detrended_aligned" / "lights" / SETUP
PS_DIR = DRAFT_DIR / "platesolve" / SETUP

TARGET_CIDS = {
    "BO CVn": "1498613634033133184",
    "FW CVn": "1497343732462852864",
}


def get_rms(summary_path: Path, cid: str) -> tuple[float | None, bool | None, float | None]:
    try:
        df = pd.read_csv(summary_path, dtype={"catalog_id": str})
        row = df[df["catalog_id"] == cid]
        if row.empty:
            return None, None, None
        slope = None
        if "am_slope" in row.columns:
            v = row["am_slope"].iloc[0]
            slope = float(v) if pd.notna(v) else None
        return (
            float(row["lc_rms"].iloc[0]),
            bool(row["am_detrended"].iloc[0]),
            slope,
        )
    except Exception:
        return None, None, None


def main() -> int:
    summary_path = PHOT_DIR / "photometry_summary.csv"
    backup_path = PHOT_DIR / "photometry_summary_before_airmass_order_test.csv"

    cfg = AppConfig()
    cfg.phase2a_airmass_before_outlier = False

    print("=" * 60)
    print("Test: phase2a_airmass_before_outlier")
    print("Draft:", DRAFT_DIR)
    print("=" * 60)

    if summary_path.is_file():
        shutil.copy2(summary_path, backup_path)
        print(f"\nBackup summary -> {backup_path.name}")

    print("\n[CURRENT — phase2a_airmass_before_outlier=False (TODO-29)]")
    before: dict[str, tuple] = {}
    for name, cid in TARGET_CIDS.items():
        rms, detrended, slope = get_rms(summary_path, cid)
        before[name] = (rms, detrended, slope)
        print(f"  {name}: lc_rms={rms:.4f} am_detrended={detrended} am_slope={slope}")

    n_proc = len(list(PROC_DIR.glob("proc_*.csv")))
    print(f"\n[RE-RUN — phase2a_airmass_before_outlier=True (pre-TODO-29)]")
    print(f"  proc_*.csv files: {n_proc}")
    cfg.phase2a_airmass_before_outlier = True

    t0 = time.time()
    try:
        run_phase2a(
            masterstar_fits_path=PS_DIR / "MASTERSTAR.fits",
            active_targets_csv=PHOT_DIR / "active_targets.csv",
            comparison_stars_csv=PHOT_DIR / "comparison_stars_per_target.csv",
            per_frame_csv_dir=PROC_DIR,
            detrended_aligned_dir=PROC_DIR,
            output_dir=PHOT_DIR,
            fwhm_px=float(_load_fwhm(PS_DIR / "MASTERSTAR.fits")),
            cfg=cfg,
            draft_id=311,
        )
    except Exception as exc:
        print(f"  ERROR: {exc}")
        import traceback

        traceback.print_exc()
        return 1

    print(f"\n  Phase 2A done in {time.time() - t0:.1f}s. Results:")
    after: dict[str, tuple] = {}
    for name, cid in TARGET_CIDS.items():
        rms, detrended, slope = get_rms(summary_path, cid)
        after[name] = (rms, detrended, slope)
        print(f"  {name}: lc_rms={rms:.4f} am_detrended={detrended} am_slope={slope}")

    print("\n[DELTA — True minus False (from backup vs new run)]")
    for name in TARGET_CIDS:
        b_rms, b_det, b_slope = before[name]
        a_rms, a_det, a_slope = after[name]
        if b_rms is not None and a_rms is not None:
            print(
                f"  {name}: d_rms={a_rms - b_rms:+.4f} "
                f"detrended {b_det} -> {a_det} slope {b_slope} -> {a_slope}"
            )

    cfg.phase2a_airmass_before_outlier = False
    print("\n[RESTORE] cfg.phase2a_airmass_before_outlier = False (default)")
    print("Summary on disk reflects True-flag run; restore backup if needed:")
    print(f"  {backup_path}")

    print("\n" + "=" * 60)
    print("DONE — check lightcurves in:", PHOT_DIR / "lightcurves")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
