#!/usr/bin/env python3
"""Headless BO CVn run for draft 455 entry-point equivalence (location_id=2 Jirny)."""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src_py"
sys.path.insert(0, str(SRC))

OUT = REPO / "dev/results/context/session_20260728_draft454/draft455_headless.json"
LOG = REPO / "tmp/draft455_headless.log"


def main() -> int:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG, encoding="utf-8"),
        ],
    )
    from night_run import NightRunParams, run_night_pipeline

    t0 = time.time()
    params = NightRunParams(
        source_dir=Path(r"D:\BO_CVn"),
        equipment_id=1,
        telescope_id=1,
        location_id=2,
        progress_cb=lambda msg: logging.info("[Progress] %s", msg),
    )
    result = run_night_pipeline(params)
    elapsed = time.time() - t0
    payload = {
        "success": result.success,
        "draft_id": result.draft_id,
        "draft_dir": str(result.draft_dir) if result.draft_dir else None,
        "n_lightcurves": result.n_lightcurves,
        "n_frames": result.n_frames,
        "phase_timings": result.phase_timings,
        "errors": result.errors,
        "elapsed_s": elapsed,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logging.info("Wrote %s", OUT)
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
