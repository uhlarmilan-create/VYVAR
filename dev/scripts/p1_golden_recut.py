#!/usr/bin/env python3
"""P1 golden re-cut: two independent wipe+rebuild headless runs, byte-identity gate."""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src_py"))
sys.path.insert(0, str(REPO / "dev"))
sys.path.insert(0, str(REPO / "dev" / "tests"))

from config import AppConfig  # noqa: E402
from tests.photometry_sha import compute_photometry_sha  # noqa: E402
from test_invariants_p1_golden import (  # noqa: E402
    MINI_NAME,
    SETUP,
    _mini_root,
    _p1_headless_chain,
    _wipe_photometry,
)

OUT = REPO / "dev" / "results" / "context" / "session_20260728_golden_recut"


def _census(mini: Path) -> dict:
    from astropy.io import fits
    import pandas as pd

    with fits.open(mini / "platesolve" / SETUP / "MASTERSTAR.fits") as hdul:
        dao = int(hdul[0].header["VY_NDAO"])
    idx = pd.read_csv(mini / "platesolve" / SETUP / "per_frame_catalog_index.csv")
    active = mini / "platesolve" / SETUP / "photometry" / "active_targets.csv"
    n_active = len(pd.read_csv(active)) if active.is_file() else -1
    lc_dir = mini / "platesolve" / SETUP / "photometry" / "lightcurves"
    n_lc = len(list(lc_dir.glob("lightcurve_*.csv"))) if lc_dir.is_dir() else 0
    return {
        "dao_pass1_vy_ndao": dao,
        "n_detected_mean": float(idx["n_detected"].mean()),
        "n_matched_mean": float(idx["n_matched"].mean()),
        "n_summary_targets": n_active,
        "n_lightcurves": n_lc,
    }


def _run_once(label: str, mini: Path) -> tuple[Path, float, dict]:
    work = REPO / "tmp" / "p1_golden_recut" / label
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(mini, work, ignore=shutil.ignore_patterns("photometry"))
    out = work / "platesolve" / SETUP / "photometry"
    if out.exists():
        shutil.rmtree(out)
    t0 = time.perf_counter()
    _p1_headless_chain(work, output_dir=out)
    elapsed = time.perf_counter() - t0
    core, nc = compute_photometry_sha(work, include_comp_qa=False)
    ext, ne = compute_photometry_sha(work, include_comp_qa=True)
    stats = {
        "core_sha": core,
        "core_n": nc,
        "extended_sha": ext,
        "extended_n": ne,
        "elapsed_s": round(elapsed, 1),
        "census": _census(work),
    }
    return work, elapsed, stats


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    mini = _mini_root()
    if not mini.is_dir():
        print("missing mini:", mini)
        return 1

    print("=== P1 golden re-cut run A ===")
    work_a, t_a, stats_a = _run_once("run_a", mini)
    print(json.dumps(stats_a, indent=2))

    print("\n=== P1 golden re-cut run B ===")
    work_b, t_b, stats_b = _run_once("run_b", mini)
    print(json.dumps(stats_b, indent=2))

    if stats_a["core_sha"] != stats_b["core_sha"] or stats_a["extended_sha"] != stats_b["extended_sha"]:
        print("STOP: run A and B disagree")
        return 1

    print("\n=== Locking mini from run A ===")
    dst = mini / "platesolve" / SETUP / "photometry"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(work_a / "platesolve" / SETUP / "photometry", dst)

    summary = {
        "run_a": stats_a,
        "run_b": stats_b,
        "agree": True,
        "mini": str(mini),
    }
    (OUT / "p1_golden_recut.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")
    print("Wrote", OUT / "p1_golden_recut.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
