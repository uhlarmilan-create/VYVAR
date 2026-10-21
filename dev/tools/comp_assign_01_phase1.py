"""COMP-ASSIGN-01: rebuild Phase 1 comparison membership for draft 514."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src_py"))
os.environ.setdefault("VYVAR_PARALLEL_WORKERS", "1")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

from config import AppConfig  # noqa: E402
from database import VyvarDatabase  # noqa: E402
from photometry_core import run_phase0_and_phase1  # noqa: E402


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

    ms_csv = og / "masterstars_full_match.csv"
    if not ms_csv.is_file():
        alt = phot / "masterstars_full_match.csv"
        ms_csv = alt if alt.is_file() else ms_csv

    result = run_phase0_and_phase1(
        variable_targets_csv=og / "variable_targets.csv",
        masterstars_csv=ms_csv,
        per_frame_csv_dir=draft / "detrended_aligned" / "lights" / setup,
        output_dir=phot,
        n_comp_min=int(cfg.phase01_comparison_n_comp_min),
        n_comp_max=int(cfg.phase01_comparison_n_comp_max),
        max_comp_rms=float(cfg.phase01_comparison_max_comp_rms),
        cfg=cfg,
        progress_cb=_prog,
        db=db,
        draft_id=514,
    )
    print(
        "RESULT",
        {k: result.get(k) for k in list(result or {})[:12]} if isinstance(result, dict) else result,
        flush=True,
    )
    print(f"ELAPSED_S {time.perf_counter()-t0:.1f}", flush=True)


if __name__ == "__main__":
    main()
