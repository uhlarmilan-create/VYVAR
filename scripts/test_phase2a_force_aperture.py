#!/usr/bin/env python3
"""Re-run Phase 2A with fixed aperture radius (e.g. AIJ Source_Radius=7 px).

Compare lc_rms for BO/FW CVn vs default SNR-table aperture.

Usage:
  python scripts/test_phase2a_force_aperture.py --aperture 7
  python scripts/test_phase2a_force_aperture.py --aperture 3.318

Temporary diagnostic — do not commit unless promoted.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from config import AppConfig
from photometry_core import run_phase2a
from ui_aperture_photometry import _load_fwhm

DRAFT = Path(r"C:\ASTRO\python\VYVAR\Archive\Drafts\draft_000311")
SETUP = "NoFilter_60_2"

TARGETS = {
    "BO CVn": "1498613634033133184",
    "FW CVn": "1497343732462852864",
}


def read_stars(summary: Path) -> dict[str, dict]:
    df = pd.read_csv(summary, dtype={"catalog_id": str})
    out: dict[str, dict] = {}
    for name, cid in TARGETS.items():
        row = df[df["catalog_id"] == cid]
        if row.empty:
            continue
        r = row.iloc[0]
        out[name] = {
            "lc_rms": float(r["lc_rms"]),
            "aperture_px": float(r.get("aperture_px", float("nan"))),
            "am_detrended": bool(r.get("am_detrended", False)),
            "am_slope": float(r["am_slope"]) if pd.notna(r.get("am_slope")) else None,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aperture", type=float, default=7.0, help="Forced circular radius [px]")
    ap.add_argument("--draft", type=Path, default=DRAFT)
    args = ap.parse_args()

    draft = args.draft.resolve()
    ps = draft / "platesolve" / SETUP
    aligned = draft / "detrended_aligned" / "lights" / SETUP
    phot = ps / "photometry"
    summary = phot / "photometry_summary.csv"
    backup = phot / f"photometry_summary_before_aperture_{args.aperture:.1f}px.csv"

    if summary.is_file():
        shutil.copy2(summary, backup)

    before = read_stars(summary)
    print("=" * 60)
    print(f"Phase 2A force_aperture_px={args.aperture:.3f}  draft={draft.name}")
    print("=" * 60)
    print("\n[BEFORE]")
    for name, d in before.items():
        print(
            f"  {name}: lc_rms={d['lc_rms']:.4f}  ap={d['aperture_px']:.3f}  "
            f"am_detrended={d['am_detrended']}  slope={d['am_slope']}"
        )

    cfg = AppConfig()
    fwhm = float(_load_fwhm(ps / "MASTERSTAR.fits"))
    print(f"\nMASTERSTAR FWHM={fwhm:.3f} px  (VYVAR default ap~{cfg.aperture_fwhm_factor * fwhm:.2f} px)")

    t0 = time.time()
    run_phase2a(
        masterstar_fits_path=ps / "MASTERSTAR.fits",
        active_targets_csv=phot / "active_targets.csv",
        comparison_stars_csv=phot / "comparison_stars_per_target.csv",
        per_frame_csv_dir=aligned,
        detrended_aligned_dir=aligned,
        output_dir=phot,
        fwhm_px=fwhm,
        cfg=cfg,
        force_aperture_px=float(args.aperture),
        draft_id=int(draft.name.split("_")[-1]) if "_" in draft.name else None,
    )
    print(f"\nPhase 2A done in {time.time() - t0:.1f}s")

    after = read_stars(summary)
    print("\n[AFTER]")
    for name, d in after.items():
        b = before.get(name, {})
        dr = d["lc_rms"] - b.get("lc_rms", d["lc_rms"]) if b else 0.0
        print(
            f"  {name}: lc_rms={d['lc_rms']:.4f} (d={dr:+.4f})  ap={d['aperture_px']:.3f}  "
            f"am_detrended={d['am_detrended']}  slope={d['am_slope']}"
        )
    print(f"\nBackup: {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
