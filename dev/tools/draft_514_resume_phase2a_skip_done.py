"""Resume Phase 2A for draft 514, skipping targets that already have lightcurve_*.csv."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src_py"))
os.environ.setdefault("VYVAR_PARALLEL_WORKERS", "1")

import pandas as pd  # noqa: E402
from config import AppConfig  # noqa: E402
from database import VyvarDatabase  # noqa: E402
from photometry_core import run_phase2a  # noqa: E402


def main() -> None:
    draft = ROOT / "Archive" / "Drafts" / "draft_000514"
    setup = "NoFilter_60_2"
    og = draft / "platesolve" / setup
    phot = og / "photometry"
    lc_dir = phot / "lightcurves"
    at_path = phot / "active_targets.csv"
    at = pd.read_csv(at_path)
    done = {p.stem.replace("lightcurve_", "") for p in lc_dir.glob("lightcurve_*.csv")}
    # Mark completed photometry targets as skip so run_phase2a does not redo them.
    # Saturated/linear zone already has skip_photometry=True.
    n_mark = 0
    for i, row in at.iterrows():
        cid = str(row.get("catalog_id", "")).strip()
        if cid and cid in done:
            at.at[i, "skip_photometry"] = True
            at.at[i, "skip_reason"] = "triage_resume_already_has_lc"
            n_mark += 1
    resume_csv = phot / "active_targets_resume_514.csv"
    at.to_csv(resume_csv, index=False)
    print(f"marked_skip_done={n_mark} remaining={len(at) - int(at['skip_photometry'].astype(str).str.lower().isin(['1','true','yes','t']).sum())}", flush=True)

    cfg = AppConfig()
    db = VyvarDatabase(Path(cfg.database_path))
    t0 = time.perf_counter()

    def _prog(msg: str) -> None:
        print(f"[{time.perf_counter()-t0:7.1f}s] {msg}", flush=True)

    result = run_phase2a(
        masterstar_fits_path=og / "MASTERSTAR.fits",
        active_targets_csv=resume_csv,
        comparison_stars_csv=phot / "comparison_stars_per_target.csv",
        per_frame_csv_dir=draft / "detrended_aligned" / "lights" / setup,
        detrended_aligned_dir=draft / "detrended_aligned" / "lights" / setup,
        output_dir=phot,
        fwhm_px=None,
        cfg=cfg,
        progress_cb=_prog,
        db=db,
        draft_id=514,
    )
    elapsed = time.perf_counter() - t0
    print("RESULT keys", list(result.keys()) if isinstance(result, dict) else type(result))
    if isinstance(result, dict):
        print("error", result.get("error"))
    print(f"ELAPSED_S {elapsed:.1f}")
    print("n_lc", len(list(lc_dir.glob("lightcurve_*.csv"))))


if __name__ == "__main__":
    main()
