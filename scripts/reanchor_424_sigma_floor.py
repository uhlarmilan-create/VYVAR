#!/usr/bin/env python3
"""Re-anchor draft_424 after PROD-SIGMA-FLOOR (two fresh runs + snapshot lock)."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

DRAFT_ID = 424
SETUP = "NoFilter_60_2"
SNAPSHOT_SUFFIX = "sigma_floor_20260713"


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_ROOT, text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _run_once(work_root: Path, cfg: Any, db: Any) -> None:
    from photometry_core import run_full_photometry_pipeline  # noqa: PLC0415

    ps = Path(cfg.archive_root) / "Drafts" / f"draft_{DRAFT_ID:06d}" / "platesolve" / SETUP
    lights = Path(cfg.archive_root) / "Drafts" / f"draft_{DRAFT_ID:06d}" / "detrended_aligned" / "lights" / SETUP
    out_phot = work_root / "platesolve" / SETUP / "photometry"
    out_phot.mkdir(parents=True, exist_ok=True)
    run_full_photometry_pipeline(
        masterstar_fits_path=ps / "MASTERSTAR.fits",
        variable_targets_csv=ps / "variable_targets.csv",
        masterstars_csv=ps / "masterstars_full_match.csv",
        per_frame_csv_dir=lights,
        detrended_aligned_dir=lights,
        output_dir=out_phot,
        cfg=cfg,
        db=db,
        draft_id=DRAFT_ID,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=_ROOT / "tmp" / "reanchor_424")
    parser.add_argument("--apply-snapshot", action="store_true")
    args = parser.parse_args()

    from config import AppConfig  # noqa: PLC0415
    from database import VyvarDatabase  # noqa: PLC0415
    from except_fix_counters import reset_except_fix_counters  # noqa: PLC0415
    from tests.photometry_sha import compare_photometry_science_meaningful, compute_photometry_sha  # noqa: PLC0415

    cfg = AppConfig()
    cfg.k2_mode = "literature"
    cfg.save_lightcurve_png = False
    out_dir = Path(args.out_dir)
    run_a = out_dir / "run_a"
    run_b = out_dir / "run_b"

    db = VyvarDatabase(cfg.database_path)
    try:
        reset_except_fix_counters()
        _run_once(run_a, cfg, db)
        reset_except_fix_counters()
        _run_once(run_b, cfg, db)
    finally:
        db.conn.close()

    core_a, n_a = compute_photometry_sha(run_a, include_comp_qa=False)
    core_b, n_b = compute_photometry_sha(run_b, include_comp_qa=False)
    ext_a, ne_a = compute_photometry_sha(run_a, include_comp_qa=True)
    ext_b, ne_b = compute_photometry_sha(run_b, include_comp_qa=True)
    repro_ok = core_a == core_b and ext_a == ext_b

    old_snap = Path(cfg.archive_root) / "Drafts" / "draft_000424_snapshot_20260708_full"
    cmp = compare_photometry_science_meaningful(run_a, old_snap, setups=(SETUP,))

    report: dict[str, Any] = {
        "git_head": _git_head(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "reproducibility": {
            "core_sha": core_a,
            "extended_sha": ext_a,
            "n_core": n_a,
            "n_extended": ne_a,
            "byte_identical": repro_ok,
        },
        "science_compare_vs_old_anchor": cmp,
        "run_a": str(run_a),
        "run_b": str(run_b),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "reanchor_424_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.apply_snapshot and repro_ok:
        new_name = f"draft_{DRAFT_ID:06d}_snapshot_{SNAPSHOT_SUFFIX}"
        dest = Path(cfg.archive_root) / "Drafts" / new_name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(run_a, dest)
        report["snapshot_path"] = str(dest)
        (out_dir / "reanchor_424_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Snapshot written: {dest}")
        print(f"UPDATE session_baseline_check.py SHAs: core={core_a} extended={ext_a}")

    print(json.dumps(report.get("reproducibility"), indent=2))
    return 0 if repro_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
