"""Numeric photometry SHA helpers (Chi_and_H zaloha anchor; draft-independent).

Recorded values (2026-06-11 re-baseline, draft_000387 re-cut x2): core 3f7c9e7a... (2806),
full d5b72d08... (4285). **Note (2026-07-08):** original draft_387 zaloha no longer exists;
current production anchor is draft_424 (`e1a7a311…` provenance hash, QUICKWINS-0708) verified
via `scripts/session_baseline_check.py --full`. Re-verify via regeneration recipe in
VYVAR_STATE.md / VYVAR_CHIANDH_BASELINE_RUNBOOK.md (where fixture present).
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd

# Core photometry only (LC + Phase-2A comp_quality + comparison pool).
PHOTOMETRY_SHA_CORE = (
    "3f7c9e7a5d8078317cb27678fde028cacf1986d3778547a0c50b087db5f19487"
)
PHOTOMETRY_SHA_CORE_PREFIX = "3f7c9e7a"

# Full reference: core fileset + comp_qa sidecars (4285 files).
PHOTOMETRY_SHA_BASELINE = (
    "d5b72d0874a38b6bec69e7a3e56abb63b759b6906495c18aa6bbf4379525b2b6"
)
PHOTOMETRY_SHA_BASELINE_PREFIX = "d5b72d08"

# Historical (2026-06-11 zaloha cut; current code no longer byte-reproduces).
PHOTOMETRY_SHA_CORE_PRE_SPARSE_FB = (
    "203254fd75ea5874f5986eac3f478260c2e7e5a9c2636bfecf2b31244cfb09ba"
)
PHOTOMETRY_SHA_BASELINE_PRE_SPARSE_FB = (
    "95a5515a6c15a473b6fcd29d3afe0c3b78d88a2da434f8a1c03f28dbe2783c24"
)

# Science-meaningful acceptance (regression vs prior anchor / additive re-baseline).
PHOTOMETRY_PROVENANCE_COLS = frozenset(
    {
        "comp_path",
        "comp_pool_n_candidates",
        "comp_pool_n_clipped",
        "comp_pool_n_final",
        "comp_clip_iterations",
        "k2_source",
        "k2_value",
        "k2_colour_ref",
    }
)
PHOTOMETRY_QC_COLS_LC = frozenset(
    {
        "err",
        "err_inflation",
        "flag",
        "method",
        "source_file",
        "lunar_phase_pct",
        "lunar_separation_deg",
        "lunar_risk",
        "time_base",
    }
)
PHOTOMETRY_TIME_COLS = frozenset({"bjd", "hjd", "jd", "mjd", "time", "bjd_tdb", "bjd_bary"})
PHOTOMETRY_SCIENCE_COLS_LC = frozenset(
    {
        "mag",
        "mag_calib",
        "mag_calib_raw",
        "mag_calib_ct",
        "mag_calib_ac",
        "mag_calib_final",
        "mag_inst",
        "delta_mag",
        "flux",
        "flux_err",
        "dao_flux",
        "dao_flux_err",
    }
)
TOL_TIME_D = 1e-6
TOL_SCIENCE = 1e-6

_SHA_PATTERNS_CORE = (
    "**/photometry/**/lightcurve_*.csv",
    "**/photometry/**/comp_quality_*.json",
    "**/platesolve/**/comparison_stars_per_target.csv",
)
_SHA_PATTERN_COMP_QA = "**/photometry/**/lightcurves/comp_qa_*.json"


def photometry_sha_files(
    draft_root: Path,
    *,
    include_comp_qa: bool = False,
) -> list[Path]:
    draft_root = Path(draft_root)
    patterns = list(_SHA_PATTERNS_CORE)
    if include_comp_qa:
        patterns.append(_SHA_PATTERN_COMP_QA)
    files: set[Path] = set()
    for pat in patterns:
        files.update(draft_root.glob(pat))
    return sorted(files)


def compute_photometry_sha(
    draft_root: Path,
    *,
    include_comp_qa: bool = False,
) -> tuple[str, int]:
    files = photometry_sha_files(draft_root, include_comp_qa=include_comp_qa)
    h = hashlib.sha256()
    for p in files:
        h.update(p.relative_to(draft_root).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest(), len(files)


def _lc_map(root: Path, setup: str) -> dict[str, Path]:
    lc_dir = root / "platesolve" / setup / "photometry" / "lightcurves"
    out: dict[str, Path] = {}
    if not lc_dir.is_dir():
        return out
    for p in lc_dir.glob("lightcurve_*.csv"):
        tid = p.stem.replace("lightcurve_", "").split("_")[0]
        out[tid] = p
    return out


def _compare_lc_science(a: Path, b: Path) -> dict:
    if not a.is_file() or not b.is_file():
        return {"status": "missing", "science_ok": False, "time_ok": False}
    da = pd.read_csv(a, low_memory=False)
    db = pd.read_csv(b, low_memory=False)
    if len(da) != len(db):
        return {"status": "row_count", "science_ok": False, "time_ok": False}
    max_delta: dict[str, float] = {}
    science_ok = True
    time_ok = True
    for col in sorted(set(da.columns) & set(db.columns)):
        if col in PHOTOMETRY_PROVENANCE_COLS or col in PHOTOMETRY_QC_COLS_LC:
            continue
        if da[col].dtype == bool or db[col].dtype == bool:
            if not da[col].equals(db[col]):
                return {"status": "bool_diff", "col": col, "science_ok": False, "time_ok": False}
            continue
        na = pd.to_numeric(da[col], errors="coerce")
        nb = pd.to_numeric(db[col], errors="coerce")
        if not (na.notna().any() and nb.notna().any()):
            if not da[col].astype(str).equals(db[col].astype(str)):
                return {"status": "string_diff", "col": col, "science_ok": False, "time_ok": False}
            continue
        delta = float(np.nanmax(np.abs(na - nb))) if len(na) else 0.0
        if not math.isfinite(delta):
            delta = 0.0
        if delta > 0.0:
            max_delta[col] = delta
        tol = TOL_TIME_D if col.lower() in PHOTOMETRY_TIME_COLS else TOL_SCIENCE
        is_time = col.lower() in PHOTOMETRY_TIME_COLS
        is_science = (
            col.lower() in PHOTOMETRY_SCIENCE_COLS_LC
            or col.lower().startswith("mag")
            or col.lower().startswith("flux")
        )
        if is_time and delta > tol:
            time_ok = False
        elif is_science and delta > tol:
            science_ok = False
    return {
        "status": "ok",
        "max_delta": max_delta,
        "science_ok": science_ok,
        "time_ok": time_ok,
    }


def compare_photometry_science_meaningful(
    root_a: Path,
    root_b: Path,
    *,
    setups: tuple[str, ...] = ("B_20_2", "L_20_2", "R_20_2", "V_20_2"),
) -> dict:
    """Tolerance-based photometry compare for re-baseline / additive gates."""
    root_a = Path(root_a)
    root_b = Path(root_b)
    max_bjd = 0.0
    max_hjd = 0.0
    science_failures: list[dict] = []
    time_failures: list[dict] = []
    n_compared = 0
    per_setup: dict[str, dict] = {}

    for setup in setups:
        ca = root_a / "platesolve" / setup / "photometry" / "comparison_stars_per_target.csv"
        cb = root_b / "platesolve" / setup / "photometry" / "comparison_stars_per_target.csv"
        comp_rep: dict = {"comp_csv_ok": False}
        if ca.is_file() and cb.is_file():
            da = pd.read_csv(ca, dtype=str, low_memory=False)
            db = pd.read_csv(cb, dtype=str, low_memory=False)
            shared_cols = [c for c in da.columns if c in db.columns and c not in PHOTOMETRY_PROVENANCE_COLS]
            if "target_catalog_id" in da.columns and "catalog_id" in da.columns:
                da["_k"] = da["target_catalog_id"].astype(str) + "|" + da["catalog_id"].astype(str)
                db["_k"] = db["target_catalog_id"].astype(str) + "|" + db["catalog_id"].astype(str)
                ma = da.set_index("_k")
                mb = db.set_index("_k")
                common = ma.index.intersection(mb.index)
                row_diffs = 0
                for col in shared_cols:
                    if col in ("target_catalog_id", "catalog_id", "name"):
                        continue
                    row_diffs += int((ma.loc[common, col].astype(str) != mb.loc[common, col].astype(str)).sum())
                comp_rep = {
                    "cols_only_a": sorted(set(da.columns) - set(db.columns)),
                    "cols_only_b": sorted(set(db.columns) - set(da.columns)),
                    "shared_row_value_diffs": row_diffs,
                    "comp_csv_ok": row_diffs == 0,
                }
        lca, lcb = _lc_map(root_a, setup), _lc_map(root_b, setup)
        shared_tids = sorted(set(lca) & set(lcb))
        for tid in shared_tids:
            cmp = _compare_lc_science(lca[tid], lcb[tid])
            n_compared += 1
            md = cmp.get("max_delta") or {}
            max_bjd = max(max_bjd, float(md.get("bjd", 0.0)))
            max_hjd = max(max_hjd, float(md.get("hjd", 0.0)))
            if not cmp.get("science_ok", True):
                science_failures.append({"setup": setup, "tid": tid, **cmp})
            if not cmp.get("time_ok", True):
                time_failures.append({"setup": setup, "tid": tid, **cmp})
        per_setup[setup] = {
            **comp_rep,
            "n_shared_lcs": len(shared_tids),
            "only_b_targets": sorted(set(lcb) - set(lca)),
            "only_a_targets": sorted(set(lca) - set(lcb)),
        }

    return {
        "setups": per_setup,
        "summary": {
            "n_lc_compared": n_compared,
            "max_abs_delta_bjd": max_bjd,
            "max_abs_delta_hjd": max_hjd,
            "science_failures": len(science_failures),
            "science_failure_sample": science_failures[:5],
            "time_failures": len(time_failures),
            "benign": len(science_failures) == 0 and len(time_failures) == 0,
        },
    }
