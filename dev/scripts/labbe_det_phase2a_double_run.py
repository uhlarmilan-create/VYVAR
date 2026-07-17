#!/usr/bin/env python3
"""LABBE-DET L3: phase2a-only double-run on draft_435 with fixed comps; SHA + err census."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

DRAFT_ID = 435
SETUP = "NoFilter_60_2"
OUT = _ROOT / "tmp" / "labbe_det_phase2a"


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True).strip()


def _run_phase2a(cfg, ps: Path, lights: Path, phot: Path, fwhm_px: float) -> None:
    from photometry_core import run_phase2a

    run_phase2a(
        masterstar_fits_path=ps / "MASTERSTAR.fits",
        active_targets_csv=phot / "active_targets.csv",
        comparison_stars_csv=phot / "comparison_stars_per_target.csv",
        per_frame_csv_dir=lights,
        detrended_aligned_dir=lights,
        output_dir=phot,
        fwhm_px=float(fwhm_px),
        cfg=cfg,
        draft_id=DRAFT_ID,
        progress_cb=lambda m: print(m, flush=True),
    )


def _clear_lc_only(phot: Path) -> None:
    for name in ("lightcurves", "lightcurves_reports", "photometry_summary.csv", "pipeline_meta.json"):
        p = phot / name
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.is_file():
            p.unlink(missing_ok=True)


def _restore_fixed_comps(phot: Path, fixed: Path) -> None:
    for name in ("active_targets.csv", "comparison_stars_per_target.csv"):
        src = fixed / name
        if src.is_file():
            shutil.copy2(src, phot / name)


def _err_census(lc_a: Path, lc_b: Path) -> dict[str, Any]:
    files_a = {p.name: p for p in lc_a.glob("lightcurve_*.csv")}
    files_b = {p.name: p for p in lc_b.glob("lightcurve_*.csv")}
    common = sorted(set(files_a) & set(files_b))
    n_diff = 0
    n_err_only = 0
    col_counts: dict[str, int] = {}
    for name in common:
        da = pd.read_csv(files_a[name], low_memory=False)
        db = pd.read_csv(files_b[name], low_memory=False)
        if da.shape != db.shape or list(da.columns) != list(db.columns):
            n_diff += 1
            col_counts["shape_or_cols"] = col_counts.get("shape_or_cols", 0) + 1
            continue
        differing = []
        for c in da.columns:
            if da[c].dtype == object or db[c].dtype == object:
                if not da[c].astype(str).equals(db[c].astype(str)):
                    differing.append(c)
            else:
                a = pd.to_numeric(da[c], errors="coerce")
                b = pd.to_numeric(db[c], errors="coerce")
                if not ((a.isna() & b.isna()) | (a == b)).all():
                    # float tolerance: exact byte path uses file SHA; here catch scientific diffs
                    if not np_allclose_nan(a.to_numpy(), b.to_numpy()):
                        differing.append(c)
        if differing:
            n_diff += 1
            for c in differing:
                col_counts[c] = col_counts.get(c, 0) + 1
            if differing == ["err"]:
                n_err_only += 1
    return {
        "n_common_lc": len(common),
        "n_diff_files": n_diff,
        "n_lc_with_err_only_diff": n_err_only,
        "column_diff_counts": col_counts,
    }


def np_allclose_nan(a, b) -> bool:
    import numpy as np

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    both_nan = np.isnan(a) & np.isnan(b)
    return bool(np.all(both_nan | (a == b)))


def main() -> int:
    from config import AppConfig
    from tests.photometry_sha import compute_photometry_sha

    OUT.mkdir(parents=True, exist_ok=True)
    cfg = AppConfig()
    draft = Path(cfg.archive_root) / "Drafts" / f"draft_{DRAFT_ID:06d}"
    ps = draft / "platesolve" / SETUP
    lights = draft / "detrended_aligned" / "lights" / SETUP
    phot = ps / "photometry"
    backup = _ROOT / "tmp" / "anchor435_protocol_v2" / "pass1_photometry_backup"
    fixed = OUT / "fixed_comps"
    if backup.is_dir():
        fixed.mkdir(parents=True, exist_ok=True)
        for name in ("active_targets.csv", "comparison_stars_per_target.csv"):
            src = backup / name
            if src.is_file():
                shutil.copy2(src, fixed / name)
    else:
        fixed.mkdir(parents=True, exist_ok=True)
        for name in ("active_targets.csv", "comparison_stars_per_target.csv"):
            src = phot / name
            if src.is_file():
                shutil.copy2(src, fixed / name)

    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": _git_head(),
        "draft_id": DRAFT_ID,
        "setup": SETUP,
        "mode": "phase2a_only_fixed_comps",
    }

    from ui_aperture_photometry import _load_fwhm

    fwhm_px = float(_load_fwhm(ps / "MASTERSTAR.fits") or 3.0)
    report["fwhm_px"] = fwhm_px

    # Run A
    _clear_lc_only(phot)
    _restore_fixed_comps(phot, fixed)
    print("=== phase2a RUN A ===", flush=True)
    _run_phase2a(cfg, ps, lights, phot, fwhm_px)
    core_a, n_a = compute_photometry_sha(draft, include_comp_qa=False)
    ext_a, ne_a = compute_photometry_sha(draft, include_comp_qa=True)
    lc_a = OUT / "lc_run_a"
    if lc_a.exists():
        shutil.rmtree(lc_a)
    shutil.copytree(phot / "lightcurves", lc_a)
    report["run_a"] = {"core_sha": core_a, "core_n": n_a, "extended_sha": ext_a, "extended_n": ne_a}
    print(f"RUN A core={core_a} n={n_a}", flush=True)

    # Run B
    _clear_lc_only(phot)
    _restore_fixed_comps(phot, fixed)
    print("=== phase2a RUN B ===", flush=True)
    _run_phase2a(cfg, ps, lights, phot, fwhm_px)
    core_b, n_b = compute_photometry_sha(draft, include_comp_qa=False)
    ext_b, ne_b = compute_photometry_sha(draft, include_comp_qa=True)
    lc_b = OUT / "lc_run_b"
    if lc_b.exists():
        shutil.rmtree(lc_b)
    shutil.copytree(phot / "lightcurves", lc_b)
    report["run_b"] = {"core_sha": core_b, "core_n": n_b, "extended_sha": ext_b, "extended_n": ne_b}
    print(f"RUN B core={core_b} n={n_b}", flush=True)

    census = _err_census(lc_a, lc_b)
    report["census"] = census
    report["sha_gate"] = {
        "byte_identical_core": core_a == core_b and n_a == n_b,
        "byte_identical_extended": ext_a == ext_b and ne_a == ne_b,
        "pass": core_a == core_b and n_a == n_b and ext_a == ext_b and ne_a == ne_b,
    }
    (OUT / "phase2a_double_run_report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["sha_gate"], indent=2), flush=True)
    print(json.dumps(census, indent=2), flush=True)
    return 0 if report["sha_gate"]["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
