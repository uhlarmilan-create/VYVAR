#!/usr/bin/env python3
"""WIDE-ERR-03B B3: Phase 2A re-export draft 515; assert mag byte-identity; BO/FW err table."""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src_py"))

from config import AppConfig  # noqa: E402
from database import VyvarDatabase  # noqa: E402
from invariants_runtime import STAGE_ORDER, load_pipeline_meta, save_pipeline_meta  # noqa: E402
from photometry_core import run_phase2a  # noqa: E402
from ui_aperture_photometry import _load_fwhm  # noqa: E402

DRAFT_ID = 515
SETUP = "NoFilter_60_2"
DRAFT = ROOT / "Archive" / "Drafts" / f"draft_{DRAFT_ID:06d}"
PS = DRAFT / "platesolve" / SETUP
LIGHTS = DRAFT / "detrended_aligned" / "lights" / SETUP
PHOT = PS / "photometry"
LC = PHOT / "lightcurves"
BACKUP = ROOT / "tmp" / "wide_err_03b_lc_backup"
OUT = ROOT / "dev" / "results" / "WIDE_ERR_03B_B3.json"

BO = "1498613634033133184"
FW = "1497343732462852864"
MAG_COLS = (
    "mag_inst",
    "mag_calib_raw",
    "mag_calib",
    "mag_calib_ct",
    "mag_calib_ac",
    "mag_calib_final",
    "delta_mag",
)


def _snapshot_lcs(src: Path, dst: Path) -> int:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(src.glob("lightcurve_*.csv")):
        shutil.copy2(p, dst / p.name)
        n += 1
    return n


def _median_err(path: Path) -> float:
    df = pd.read_csv(path, comment="#", low_memory=False)
    if "err" not in df.columns:
        return float("nan")
    return float(pd.to_numeric(df["err"], errors="coerce").median())


def _mag_identity(before: Path, after: Path) -> dict:
    files_a = {p.name: p for p in before.glob("lightcurve_*.csv")}
    files_b = {p.name: p for p in after.glob("lightcurve_*.csv")}
    common = sorted(set(files_a) & set(files_b))
    n_ok = 0
    n_fail = 0
    fail_examples = []
    for name in common:
        da = pd.read_csv(files_a[name], comment="#", low_memory=False)
        db = pd.read_csv(files_b[name], comment="#", low_memory=False)
        cols = [c for c in MAG_COLS if c in da.columns and c in db.columns]
        ok = True
        for c in cols:
            a = pd.to_numeric(da[c], errors="coerce").to_numpy(dtype=float)
            b = pd.to_numeric(db[c], errors="coerce").to_numpy(dtype=float)
            if a.shape != b.shape or not np.array_equal(a, b, equal_nan=True):
                ok = False
                break
        if ok:
            n_ok += 1
        else:
            n_fail += 1
            if len(fail_examples) < 5:
                fail_examples.append(name)
    return {
        "n_common": len(common),
        "n_mag_byte_identical": n_ok,
        "n_mag_mismatch": n_fail,
        "pass": n_fail == 0 and len(common) > 0,
        "fail_examples": fail_examples,
    }


def main() -> int:
    if not (PHOT / "err_calibration.json").is_file():
        print("MISSING err_calibration.json - run heldout first", flush=True)
        return 2
    if not (PHOT / "gain_photon_transfer.json").is_file():
        print("MISSING gain_photon_transfer.json", flush=True)
        return 2

    backup_existed = BACKUP.is_dir() and any(BACKUP.glob("lightcurve_*.csv"))
    if backup_existed:
        before_bo = _median_err(BACKUP / f"lightcurve_{BO}.csv")
        before_fw = _median_err(BACKUP / f"lightcurve_{FW}.csv")
        n_snap = len(list(BACKUP.glob("lightcurve_*.csv")))
        print(f"reusing existing backup {BACKUP} ({n_snap} LCs)", flush=True)
    else:
        before_bo = _median_err(LC / f"lightcurve_{BO}.csv")
        before_fw = _median_err(LC / f"lightcurve_{FW}.csv")
        n_snap = _snapshot_lcs(LC, BACKUP)
        print(f"backed up {n_snap} LCs to {BACKUP}", flush=True)
    print(f"before median err BO={before_bo:.6f} FW={before_fw:.6f}", flush=True)

    cfg = AppConfig()
    # Ensure calibrated export path is active
    if hasattr(cfg, "export_err_mode"):
        cfg.export_err_mode = "calibrated"
    db = VyvarDatabase(Path(cfg.database_path))
    # Allow phase2a re-stamp after a prior postprocess stamp (INV-DAG-01).
    meta = load_pipeline_meta(PHOT)
    stages = meta.get("stages") if isinstance(meta.get("stages"), list) else []
    p2_seq = STAGE_ORDER.index("phase2a")
    meta["stages"] = [
        s
        for s in stages
        if isinstance(s, dict) and str(s.get("name") or "") in STAGE_ORDER
        and STAGE_ORDER.index(str(s.get("name"))) < p2_seq
    ]
    save_pipeline_meta(PHOT, meta)
    print(f"truncated DAG stages after phase01; remaining={len(meta['stages'])}", flush=True)

    fw = float(_load_fwhm(PS / "MASTERSTAR.fits"))
    t0 = time.time()
    run_phase2a(
        masterstar_fits_path=PS / "MASTERSTAR.fits",
        active_targets_csv=PHOT / "active_targets.csv",
        comparison_stars_csv=PHOT / "comparison_stars_per_target.csv",
        per_frame_csv_dir=LIGHTS,
        detrended_aligned_dir=LIGHTS,
        output_dir=PHOT,
        fwhm_px=fw,
        cfg=cfg,
        db=db,
        draft_id=DRAFT_ID,
        progress_cb=lambda m: print(m, flush=True),
    )
    elapsed = time.time() - t0
    print(f"phase2a done in {elapsed:.1f}s", flush=True)

    after_bo = _median_err(LC / f"lightcurve_{BO}.csv")
    after_fw = _median_err(LC / f"lightcurve_{FW}.csv")
    ident = _mag_identity(BACKUP, LC)

    payload = {
        "task": "WIDE-ERR-03B B3",
        "draft_id": DRAFT_ID,
        "setup": SETUP,
        "elapsed_s": elapsed,
        "n_lc_snapshot": n_snap,
        "mag_byte_identity": ident,
        "err_before_after": {
            "BO": {
                "catalog_id": BO,
                "median_err_before_mag": before_bo,
                "median_err_after_mag": after_bo,
                "median_err_before_mmag": before_bo * 1000.0,
                "median_err_after_mmag": after_bo * 1000.0,
            },
            "FW": {
                "catalog_id": FW,
                "median_err_before_mag": before_fw,
                "median_err_after_mag": after_fw,
                "median_err_before_mmag": before_fw * 1000.0,
                "median_err_after_mmag": after_fw * 1000.0,
            },
        },
        "backup_dir": str(BACKUP),
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("WROTE", OUT)
    print("mag identity", ident)
    print(
        f"BO err mmag {before_bo*1000:.3f} -> {after_bo*1000:.3f}; "
        f"FW {before_fw*1000:.3f} -> {after_fw*1000:.3f}"
    )
    return 0 if ident["pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
