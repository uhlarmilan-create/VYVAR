#!/usr/bin/env python3
"""Validate EXO-AS-TARGET on draft 422 via production pipeline path."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from config import AppConfig
from database import VyvarDatabase
from pipeline import write_photometry_plan_files
from photometry_core import run_full_photometry_pipeline

DRAFT_ID = 422
SETUP = "V_60_2"
# True Gaia DR3 source_id (verified in vyvar_gaia_dr3.db; NOT the float64-rounded ...0400).
TRUE_GAIA_CID = "1625373404725030528"
FLOAT_CORRUPTED_CID = "1625373404725030400"
TOI_OBJ = "TOI-1131.01"
VAL = _ROOT / "tmp" / "exo_as_target_422"
BASELINE = VAL / "baseline" / "photometry"
REPORT = VAL / "validation_report.json"
_GAIA_DTYPE = {"catalog_id": str, "name": str, "target_catalog_id": str}


def _comp_sets(comp_csv: Path) -> dict[str, frozenset[str]]:
    df = pd.read_csv(comp_csv, dtype=_GAIA_DTYPE)
    out: dict[str, frozenset[str]] = {}
    for tid, grp in df.groupby("target_catalog_id", sort=False):
        out[str(tid)] = frozenset(grp["catalog_id"].astype(str).str.strip())
    return out


def _lc_compare(off_dir: Path, on_dir: Path, cid: str) -> list[str]:
    errs: list[str] = []
    off_lc = off_dir / "lightcurves" / f"lightcurve_{cid}.csv"
    on_lc = on_dir / "lightcurves" / f"lightcurve_{cid}.csv"
    if not off_lc.is_file() or not on_lc.is_file():
        return [f"missing LC for {cid}"]
    d_off = pd.read_csv(off_lc)
    d_on = pd.read_csv(on_lc)
    m = d_off.merge(d_on, on=["bjd", "source_file"], suffixes=("_off", "_on"), how="outer", indicator=True)
    if (m["_merge"] != "both").any():
        errs.append(f"{cid}: LC key mismatch")
    for col in ("mag_inst", "mag_calib", "err"):
        a = pd.to_numeric(m[f"{col}_off"], errors="coerce").to_numpy(dtype=np.float64)
        b = pd.to_numeric(m[f"{col}_on"], errors="coerce").to_numpy(dtype=np.float64)
        if not np.array_equal(a, b, equal_nan=True):
            errs.append(f"{cid}: {col} differs")
    return errs


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    VAL.mkdir(parents=True, exist_ok=True)

    cfg = AppConfig()
    cfg.exoplanet_local_db_path = str((_ROOT / "exoplanets" / "vyvar_exoplanet_local.db").resolve())
    db = VyvarDatabase(cfg.database_path)
    draft = Path(cfg.archive_root) / "Drafts" / f"draft_{DRAFT_ID:06d}"
    ps = draft / "platesolve" / SETUP
    lights = draft / "detrended_aligned" / "lights" / SETUP
    out_dir = VAL / "photometry"
    if out_dir.is_dir():
        shutil.rmtree(out_dir)

    print("=== write_photometry_plan_files ===", flush=True)
    write_photometry_plan_files(
        platesolve_dir=ps,
        masterstar_fits=ps / "MASTERSTAR.fits",
        masterstars_csv=ps / "masterstars_full_match.csv",
        draft_id=DRAFT_ID,
        database_path=cfg.database_path,
    )
    vt = pd.read_csv(ps / "variable_targets.csv", dtype=_GAIA_DTYPE)
    print(f"variable_targets rows: {len(vt)}", flush=True)

    print("=== run_full_photometry_pipeline ===", flush=True)
    res = run_full_photometry_pipeline(
        masterstar_fits_path=ps / "MASTERSTAR.fits",
        variable_targets_csv=ps / "variable_targets.csv",
        masterstars_csv=ps / "masterstars_full_match.csv",
        per_frame_csv_dir=lights,
        detrended_aligned_dir=lights,
        output_dir=out_dir,
        cfg=cfg,
        db=db,
        draft_id=DRAFT_ID,
    )
    p2a = res.get("phase2a") or {}
    print(json.dumps(p2a, indent=2, default=str), flush=True)

    report: dict = {
        "passed": False,
        "errors": [],
        "true_gaia_catalog_id": TRUE_GAIA_CID,
        "float_corrupted_catalog_id": FLOAT_CORRUPTED_CID,
    }

    at = pd.read_csv(out_dir / "active_targets.csv", dtype=_GAIA_DTYPE)
    report["active_count"] = int(len(at))
    report["active_catalog_ids"] = at["catalog_id"].astype(str).tolist()

    toi_vt = vt[vt["catalog_id"].astype(str).str.strip() == TRUE_GAIA_CID]
    report["toi_in_variable_targets_by_true_cid"] = not toi_vt.empty
    if not toi_vt.empty:
        report["toi_variable_targets_row"] = toi_vt.iloc[0].to_dict()

    toi_at = at[at["catalog_id"].astype(str).str.strip() == TRUE_GAIA_CID]
    report["toi_in_active_targets_by_true_cid"] = not toi_at.empty
    if not toi_at.empty:
        row = toi_at.iloc[0]
        report["toi_active_row"] = row.to_dict()
        report["toi_skip_photometry"] = bool(row.get("skip_photometry", False))
        report["toi_zone"] = str(row.get("zone_flag", "") or "")

    report["catalog_id_is_true_gaia"] = (
        report["toi_in_variable_targets_by_true_cid"] and report["toi_in_active_targets_by_true_cid"]
    )
    report["catalog_id_is_float_corrupted"] = bool(
        (vt["catalog_id"].astype(str).str.strip() == FLOAT_CORRUPTED_CID).any()
        or (at["catalog_id"].astype(str).str.strip() == FLOAT_CORRUPTED_CID).any()
    )

    lc_path = out_dir / "lightcurves" / f"lightcurve_{TRUE_GAIA_CID}.csv"
    report["toi_lc_exists"] = lc_path.is_file()
    if lc_path.is_file():
        lc = pd.read_csv(lc_path)
        report["toi_lc_n_epochs"] = int(len(lc))

    comp_pool = pd.read_csv(ps / "comparison_stars.csv", dtype=_GAIA_DTYPE)
    comp_pt = pd.read_csv(out_dir / "comparison_stars_per_target.csv", dtype=_GAIA_DTYPE)
    report["toi_in_comparison_stars_pool"] = bool(
        (comp_pool["catalog_id"].astype(str).str.strip() == TRUE_GAIA_CID).any()
    )
    report["toi_used_as_comp"] = bool((comp_pt["catalog_id"].astype(str).str.strip() == TRUE_GAIA_CID).any())
    report["toi_wrong_cid_in_comps"] = bool(
        (comp_pool["catalog_id"].astype(str).str.strip() == FLOAT_CORRUPTED_CID).any()
        or (comp_pt["catalog_id"].astype(str).str.strip() == FLOAT_CORRUPTED_CID).any()
    )

    base_at = pd.read_csv(BASELINE / "active_targets.csv", dtype=_GAIA_DTYPE)
    base_cids = [str(x).strip() for x in base_at["catalog_id"].astype(str)]
    harm_errs: list[str] = []
    for cid in base_cids:
        harm_errs.extend(_lc_compare(BASELINE, out_dir, cid))
    base_comp = _comp_sets(BASELINE / "comparison_stars_per_target.csv")
    new_comp = _comp_sets(out_dir / "comparison_stars_per_target.csv")
    for tid in base_cids:
        if tid not in new_comp:
            harm_errs.append(f"comp target missing {tid}")
            continue
        if base_comp.get(tid) != new_comp.get(tid):
            harm_errs.append(f"comp set changed for {tid}")
    report["do_no_harm_errors"] = harm_errs

    report["passed"] = (
        report["catalog_id_is_true_gaia"]
        and not report["catalog_id_is_float_corrupted"]
        and not report["toi_in_comparison_stars_pool"]
        and not report["toi_used_as_comp"]
        and report["active_count"] == 9
        and not harm_errs
    )
    if not report["passed"]:
        report["errors"] = harm_errs

    REPORT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
