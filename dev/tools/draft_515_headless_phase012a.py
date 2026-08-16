"""Headless Phase 0+1+2A for draft 515 on existing calibrated/aligned products.

RUN-HARDEN-01 Part A: no mid-Phase-1 comps were persisted (abort at T16), so this
re-runs Phase 0+1+2A from MASTERSTAR + proc CSVs. Does not re-calibrate or
re-platesolve.
"""
from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src_py"))

from config import AppConfig  # noqa: E402
from database import VyvarDatabase  # noqa: E402
from photometry_core import run_full_photometry_pipeline  # noqa: E402


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(ROOT),
                text=True,
                stderr=subprocess.DEVNULL,
            )
            .strip()
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def main() -> int:
    draft_id = 515
    setup = "NoFilter_60_2"
    draft = ROOT / "Archive" / "Drafts" / f"draft_{draft_id:06d}"
    og = draft / "platesolve" / setup
    phot = og / "photometry"
    pf_dir = draft / "detrended_aligned" / "lights" / setup
    sha = _git_sha()
    t0 = time.perf_counter()
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"HARNESS draft_515_headless_phase012a", flush=True)
    print(f"GIT_SHA {sha}", flush=True)
    print(f"START_UTC {started}", flush=True)
    print(f"DRAFT {draft}", flush=True)

    cfg = AppConfig()
    db = VyvarDatabase(Path(cfg.database_path))

    def _prog(msg: str) -> None:
        print(f"[{time.perf_counter() - t0:8.1f}s] {msg}", flush=True)

    result = run_full_photometry_pipeline(
        masterstar_fits_path=og / "MASTERSTAR.fits",
        variable_targets_csv=og / "variable_targets.csv",
        masterstars_csv=og / "masterstars_full_match.csv",
        per_frame_csv_dir=pf_dir,
        detrended_aligned_dir=pf_dir,
        output_dir=phot,
        cfg=cfg,
        db=db,
        draft_id=draft_id,
        progress_cb=_prog,
    )
    elapsed = time.perf_counter() - t0
    print(f"ELAPSED_S {elapsed:.1f}", flush=True)
    if isinstance(result, dict):
        print(f"ERROR {result.get('error')!r}", flush=True)
        print(f"ZERO_TARGETS {result.get('zero_targets')!r}", flush=True)
        for k in ("n_lc", "n_lightcurves", "n_targets", "n_active"):
            if k in result:
                print(f"{k.upper()} {result.get(k)!r}", flush=True)
        print(f"RESULT_KEYS {sorted(result.keys())}", flush=True)
        return 1 if result.get("error") else 0
    print(f"RESULT_TYPE {type(result)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
