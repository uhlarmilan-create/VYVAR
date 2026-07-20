"""
Quick test: compute safe_bbox for draft_000283 using existing aligned frames.
Does NOT re-run alignment - just re-runs write_photometry_plan_files logic.
Run: python scripts/test_border_bbox.py
"""

from pathlib import Path
import sys, os
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401  (repo layout: src_py + dev on sys.path)

from pipeline import write_photometry_plan_files

DRAFT = Path(r"C:\ASTRO\python\VYVAR\Archive\Drafts\draft_000283")
PLATESOLVE = DRAFT / "platesolve" / "NoFilter_60_2"
ALIGNED_DIR = DRAFT / "detrended_aligned" / "lights" / "NoFilter_60_2"

aligned_files = sorted(ALIGNED_DIR.glob("proc_*.fits"))
print(f"Found {len(aligned_files)} aligned frames")

result = write_photometry_plan_files(
    platesolve_dir=PLATESOLVE,
    masterstar_fits=PLATESOLVE / "MASTERSTAR.fits",
    masterstars_csv=PLATESOLVE / "masterstars_full_match.csv",
    n_comparison_stars=150,
    require_non_variable=True,
    aligned_files=aligned_files,
)

import json
plan = json.loads((PLATESOLVE / "photometry_plan.json").read_text())
bbox = plan.get("safe_bbox_px")
print(f"safe_bbox_px = {bbox}")
print("[OK] Border filter works!" if bbox else "[X] safe_bbox_px still None")

