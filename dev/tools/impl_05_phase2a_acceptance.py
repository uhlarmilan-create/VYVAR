#!/usr/bin/env python3
"""IMPL-05 acceptance Phase 2A: subset targets, per-mag aperture table (no force)."""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src_py"))
os.environ.setdefault("VYVAR_PARALLEL_WORKERS", "1")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import pandas as pd  # noqa: E402
from config import AppConfig  # noqa: E402
from database import VyvarDatabase  # noqa: E402
from photometry_core import run_phase2a  # noqa: E402


def main() -> None:
    draft = ROOT / "Archive" / "Drafts" / "draft_000514"
    setup = "NoFilter_60_2"
    og = draft / "platesolve" / setup
    phot = og / "photometry"
    at_path = phot / "active_targets.csv"
    at = pd.read_csv(at_path, dtype={"catalog_id": str})

    impl03 = ROOT / "dev" / "results" / "IMPL_03_production_scatter.json"
    cids = {
        "1498613634033133184",
        "1497343732462852864",
    }
    if impl03.is_file():
        data = json.loads(impl03.read_text(encoding="utf-8"))
        for k in (data.get("after") or {}).get("quiet_targets") or {}:
            cids.add(str(k).strip())
        for k in (data.get("after") or {}).get("targets") or {}:
            cid = (data["after"]["targets"][k] or {}).get("catalog_id")
            if cid:
                cids.add(str(cid).strip())
    cids = {c for c in cids if c and c.isdigit()}
    print(f"ACCEPT n={len(cids)}", sorted(cids), flush=True)

    bak = phot / "active_targets_full_before_impl05.csv"
    full_bak = phot / "active_targets_full_before_impl04.csv"
    if not bak.is_file():
        src = full_bak if full_bak.is_file() else at_path
        shutil.copy2(src, bak)
    # Prefer restoring full list before subsetting
    if full_bak.is_file():
        at = pd.read_csv(full_bak, dtype={"catalog_id": str})
    sub = at[at["catalog_id"].astype(str).str.strip().isin(cids)].copy()
    if sub.empty:
        raise SystemExit("no acceptance targets in active_targets.csv")
    sub.to_csv(at_path, index=False)
    print(f"wrote subset active_targets n={len(sub)}", flush=True)

    cfg = AppConfig()
    db = VyvarDatabase(Path(cfg.database_path))
    t0 = time.perf_counter()

    def _prog(msg: str) -> None:
        print(f"[{time.perf_counter()-t0:7.1f}s] {msg}", flush=True)

    result = run_phase2a(
        masterstar_fits_path=og / "MASTERSTAR.fits",
        active_targets_csv=at_path,
        comparison_stars_csv=phot / "comparison_stars_per_target.csv",
        per_frame_csv_dir=draft / "detrended_aligned" / "lights" / setup,
        detrended_aligned_dir=draft / "detrended_aligned" / "lights" / setup,
        output_dir=phot,
        fwhm_px=None,
        cfg=cfg,
        force_aperture_px=None,  # use gated per-mag aperture_scatter_table.json
        progress_cb=_prog,
        db=db,
        draft_id=514,
    )
    if bak.is_file():
        shutil.copy2(bak, at_path)
        print("restored full active_targets.csv", flush=True)
    print(
        "RESULT",
        result if not isinstance(result, dict) else {k: result.get(k) for k in list(result)[:12]},
        flush=True,
    )
    print(f"ELAPSED_S {time.perf_counter()-t0:.1f}", flush=True)


if __name__ == "__main__":
    main()
