"""Resume Phase 2A only for draft 514 after A1/A2 fixes (DRAFT-514-TRIAGE)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src_py"))

from config import AppConfig  # noqa: E402
from database import VyvarDatabase  # noqa: E402
from photometry_core import run_phase2a  # noqa: E402


def main() -> None:
    draft = ROOT / "Archive" / "Drafts" / "draft_000514"
    setup = "NoFilter_60_2"
    og = draft / "platesolve" / setup
    phot = og / "photometry"
    cfg = AppConfig()
    db = VyvarDatabase(Path(cfg.database_path))
    t0 = time.perf_counter()

    def _prog(msg: str) -> None:
        print(f"[{time.perf_counter()-t0:7.1f}s] {msg}", flush=True)

    result = run_phase2a(
        masterstar_fits_path=og / "MASTERSTAR.fits",
        active_targets_csv=phot / "active_targets.csv",
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
        print("n_lc", result.get("n_lc") or result.get("n_lightcurves"))
    print(f"ELAPSED_S {elapsed:.1f}")


if __name__ == "__main__":
    main()
