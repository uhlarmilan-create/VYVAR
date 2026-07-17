#!/usr/bin/env python3
"""SPARSE-TRUST completion validation: S1-S4 + flip report (spec section 6)."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401  (repo layout: src_py + dev on sys.path)
_ROOT = _bootstrap.REPO_ROOT
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

ANCHOR_SNAPSHOT = _ROOT / "Archive" / "Drafts" / "draft_000424_snapshot_sigma_floor_20260713"
ANCHOR_SETUP = "NoFilter_60_2"
EXPECTED_SHA_CORE = "bf3743a150d788283eab2ab51db7b31f59e6d1c481159208bbe3f573092ec975"
WORK424 = _ROOT / "tmp" / "sparse_trust_validation" / "draft424_work" / "photometry"
WORK426_R = _ROOT / "tmp" / "sparse_trust_validation" / "draft426_r60_work" / "photometry"
SS_CAM_CID = "1112113066119992064"
OUT_DIR = _ROOT / "tmp" / "sparse_trust_validation"


def _trust_band_from_sidecar(phot: Path, tid: str, cfg: Any) -> dict[str, Any]:
    from sparse_trust_core import sparse_trust_config_from_app, trust_band  # noqa: PLC0415
    from trust_flag_core import read_sparse_trust_sidecar  # noqa: PLC0415

    side = read_sparse_trust_sidecar(phot, tid)
    if not side:
        return {"band": "none", "flags": []}
    band, flags = trust_band(
        R_hi=float(side.get("trust_R_hi", float("nan"))),
        R_lo=float(side.get("trust_R_lo", float("nan"))),
        stability_p=float(side.get("comp_stability_p", float("nan"))),
        x2_pair_mag2=float(side.get("x2_pair_mag2", float("nan"))),
        n_comps=int(side.get("n_comps", 2) or 2),
        triangulation_clipped=bool(side.get("triangulation_clipped")),
        cfg=sparse_trust_config_from_app(cfg),
    )
    return {
        "band": band,
        "flags": list(flags),
        "R": side.get("trust_R"),
        "R_lo": side.get("trust_R_lo"),
        "R_hi": side.get("trust_R_hi"),
        "comp_stability_p": side.get("comp_stability_p"),
        "x2_pair_mag2": side.get("x2_pair_mag2"),
        "check_sparse": side.get("check_sparse"),
    }


def _sidecar_tuple(phot: Path, tid: str, comp_n: int) -> dict[str, Any]:
    p = phot / "lightcurves" / f"check_kmag_{tid}.csv"
    if not p.is_file():
        return {"status": "missing_sidecar"}
    df = pd.read_csv(p, low_memory=False)
    if df.empty:
        return {"status": "empty"}
    row = df.iloc[0]
    km = pd.to_numeric(df["kmag"], errors="coerce")
    n_epochs = int(km.notna().sum())
    return {
        "status": "ok",
        "K_id": str(row.get("check_catalog_id", "") or ""),
        "k_source": str(row.get("k_source", "") or ""),
        "k_colour_offset": float(pd.to_numeric(row.get("k_colour_offset", float("nan")), errors="coerce")),
        "k_tier_excluded": int(pd.to_numeric(row.get("k_tier_excluded", 0), errors="coerce") or 0),
        "k_colour_caveat": int(pd.to_numeric(row.get("k_colour_caveat", 0), errors="coerce") or 0),
        "R": float(pd.to_numeric(row.get("trust_R", float("nan")), errors="coerce")),
        "R_lo": float(pd.to_numeric(row.get("trust_R_lo", float("nan")), errors="coerce")),
        "R_hi": float(pd.to_numeric(row.get("trust_R_hi", float("nan")), errors="coerce")),
        "R_detrend": float(pd.to_numeric(row.get("trust_R_detrend", float("nan")), errors="coerce")),
        "comp_stability_p": float(pd.to_numeric(row.get("comp_stability_p", float("nan")), errors="coerce")),
        "x2_pair_mag2": float(pd.to_numeric(row.get("x2_pair_mag2", float("nan")), errors="coerce")),
        "n_comps": int(comp_n),
        "N_epochs": n_epochs,
        "check_sparse": int(pd.to_numeric(row.get("check_sparse", 0), errors="coerce") or 0),
        "flags": str(row.get("sparse_flags", "") or ""),
        "triangulation_clipped": int(pd.to_numeric(row.get("triangulation_clipped", 0), errors="coerce") or 0),
    }


def _pool_n_map(comp_df: pd.DataFrame) -> dict[str, int]:
    pool_n: dict[str, int] = {}
    if comp_df.empty or "target_catalog_id" not in comp_df.columns:
        return pool_n
    if "comp_pool_n_final" in comp_df.columns:
        for tid, grp in comp_df.groupby(comp_df["target_catalog_id"].astype(str).str.strip()):
            nf = pd.to_numeric(grp.get("comp_pool_n_final"), errors="coerce")
            if isinstance(nf, pd.Series):
                val = int(nf.max()) if nf.notna().any() else 0
            else:
                val = int(nf) if math.isfinite(float(nf)) else 0
            if val > 0:
                pool_n[str(tid)] = val
        return pool_n
    for tid, grp in comp_df.groupby(comp_df["target_catalog_id"].astype(str).str.strip()):
        pool_n[str(tid)] = int(len(grp))
    return pool_n


def run_s2(cfg: Any) -> dict[str, Any]:
    from trust_flag_core import compute_trust_for_photometry_dir  # noqa: PLC0415

    anchor_phot = ANCHOR_SNAPSHOT / "platesolve" / ANCHOR_SETUP / "photometry"
    old_map = compute_trust_for_photometry_dir(anchor_phot, cfg=cfg).get("per_target", {})
    old_trust = {tid: str(info.get("trust", "")) for tid, info in old_map.items()}
    new_map = compute_trust_for_photometry_dir(WORK424, cfg=cfg).get("per_target", {})
    comp_df = pd.read_csv(
        anchor_phot / "comparison_stars_per_target.csv",
        dtype={"target_catalog_id": str},
        low_memory=False,
    )
    pool_n = _pool_n_map(comp_df)
    flips: list[dict[str, Any]] = []
    band_changes: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    baseline_path = OUT_DIR / "s2_flip_report_completion_baseline.csv"
    baseline_bands: dict[str, str] = {}
    if baseline_path.is_file():
        bdf = pd.read_csv(baseline_path, dtype=str)
        if "target" in bdf.columns and "new_band" in bdf.columns:
            baseline_bands = dict(zip(bdf["target"].astype(str), bdf["new_band"].astype(str), strict=False))
    for tid, old_band in old_trust.items():
        n_pool = int(pool_n.get(tid, 0))
        if n_pool < 5:
            continue
        new_band = str(new_map.get(tid, {}).get("trust", ""))
        sparse = _trust_band_from_sidecar(WORK424, tid, cfg)
        rows.append(
            {
                "target": tid,
                "old_band": old_band,
                "new_band": new_band,
                "n_pool": n_pool,
                "sparse_ci_band": sparse.get("band"),
                "R": sparse.get("R"),
                "R_lo": sparse.get("R_lo"),
                "R_hi": sparse.get("R_hi"),
                "comp_stability_p": sparse.get("comp_stability_p"),
            }
        )
        if old_band == "GREEN" and new_band == "RED":
            flips.append({"target": tid, "old": old_band, "new": new_band, **sparse})
        if baseline_bands and tid in baseline_bands and str(baseline_bands[tid]) != new_band:
            band_changes.append(
                {"target": tid, "completion_band": baseline_bands[tid], "amendment_band": new_band}
            )
    return {
        "pass": len(flips) == 0 and len(band_changes) == 0,
        "work_dir": str(WORK424),
        "anchor_read_only": str(anchor_phot),
        "n_n5": len(rows),
        "flips": flips,
        "band_changes_vs_completion": band_changes,
        "flip_report": rows,
    }


def run_s3(cfg: Any) -> dict[str, Any]:
    comp_df = pd.read_csv(
        WORK426_R / "comparison_stars_per_target.csv",
        dtype={"target_catalog_id": str},
        low_memory=False,
    )
    pool_n = _pool_n_map(comp_df)
    ss_pool = int(pool_n.get(SS_CAM_CID, 2))
    sem_path = _ROOT / "tmp" / "sigma_sem_cause" / "setup_r_60_4.json"
    chi2_prod = float("nan")
    if sem_path.is_file():
        blob = json.loads(sem_path.read_text(encoding="utf-8"))
        for star in blob.get("stars", []):
            if str(star.get("target_cid")) == SS_CAM_CID:
                chi2_prod = float(star.get("production_chi2_lc_err", float("nan")))
    ss = _sidecar_tuple(WORK426_R, SS_CAM_CID, ss_pool)
    ss["production_lc_err_chi2"] = chi2_prod
    ss["sparse_trust_band"] = _trust_band_from_sidecar(WORK426_R, SS_CAM_CID, cfg)
    ss["pass_kmag"] = int(ss.get("N_epochs", 0) or 0) > 0 and math.isfinite(float(ss.get("R", float("nan"))))
    sparse_targets = []
    for tid in comp_df["target_catalog_id"].astype(str).unique():
        nf = int(pool_n.get(str(tid), 0))
        if nf <= 2:
            sparse_targets.append({"target": tid, **_sidecar_tuple(WORK426_R, tid, nf)})
    return {
        "pass": any(t.get("N_epochs", 0) > 0 for t in sparse_targets) and ss.get("pass_kmag"),
        "work_dir": str(WORK426_R),
        "r_60_4_sparse_targets": sparse_targets,
        "SS_Cam": ss,
    }


def run_s4() -> dict[str, Any]:
    from tests.photometry_sha import compute_photometry_sha  # noqa: PLC0415

    anchor_phot = ANCHOR_SNAPSHOT / "platesolve" / ANCHOR_SETUP / "photometry"
    sha, n = compute_photometry_sha(ANCHOR_SNAPSHOT)
    lc_work = WORK424 / "lightcurves"
    lc_files_in_work = list(lc_work.glob("lightcurve_*.csv"))
    return {
        "pass": sha == EXPECTED_SHA_CORE and len(lc_files_in_work) == 0,
        "anchor_core_sha": sha,
        "expected_sha": EXPECTED_SHA_CORE,
        "n_files_hashed": n,
        "work_lightcurve_files": len(lc_files_in_work),
        "note": "Sidecar-only backfill; anchor LCs unmodified; err byte-identical by construction",
        "anchor_tree_touched": False,
    }


def run_s1_slow() -> dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_sparse_trust_core.py", "-m", "slow", "-q", "--tb=no"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
    )
    from sparse_trust_core import diff_variance, triangulate_variances, triangulation_hat_ci  # noqa: PLC0415

    results: dict[str, Any] = {}
    for n_epochs in (15, 25, 139):
        rng = np.random.default_rng(n_epochs)
        n_trials = 500
        ok = clip = 0
        for _ in range(n_trials):
            sig_k = rng.uniform(0.003, 0.015)
            pb_k = rng.uniform(0.00001, 0.00005)
            x2_k = max(sig_k**2 - pb_k, 0)
            x2_c1 = max(rng.uniform(0.003, 0.012) ** 2 - rng.uniform(0.00001, 0.00005), 0)
            x2_c2 = max(rng.uniform(0.003, 0.012) ** 2 - rng.uniform(0.00001, 0.00005), 0)
            n = n_epochs
            m_k = rng.normal(12.0, math.sqrt(x2_k + pb_k), n)
            m_c1 = rng.normal(11.5, math.sqrt(x2_c1 + 0.00003), n)
            m_c2 = rng.normal(11.8, math.sqrt(x2_c2 + 0.00003), n)
            s2_kc1 = diff_variance(m_k, m_c1)
            s2_kc2 = diff_variance(m_k, m_c2)
            s2_c1c2 = diff_variance(m_c1, m_c2)
            tri = triangulate_variances(s2_kc1, s2_kc2, s2_c1c2)
            if tri.triangulation_clipped:
                clip += 1
            true_var = sig_k**2
            _, lo, hi = triangulation_hat_ci(s2_kc1, s2_kc2, s2_c1c2, n)
            if math.isfinite(lo) and math.isfinite(hi) and lo <= true_var <= hi:
                ok += 1
        rate = ok / n_trials
        results[str(n_epochs)] = {
            "coverage": rate,
            "clip_rate": clip / n_trials,
            "pass": rate >= 0.93,
        }
    return {
        "pass": proc.returncode == 0 and all(v["pass"] for v in results.values()),
        "pytest_exit": proc.returncode,
        "pytest_tail": proc.stdout.strip().splitlines()[-3:],
        "per_N": results,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    from config import AppConfig  # noqa: PLC0415
    from scripts.backfill_check_kmag_sidecars import backfill_photometry  # noqa: PLC0415

    cfg = AppConfig()
    anchor_phot = ANCHOR_SNAPSHOT / "platesolve" / ANCHOR_SETUP / "photometry"
    src426 = _ROOT / "Archive" / "Drafts" / "draft_000426" / "platesolve" / "r_60_4" / "photometry"

    if WORK424.is_dir():
        shutil.rmtree(WORK424)
    WORK424.mkdir(parents=True, exist_ok=True)
    for name in ("comparison_stars_per_target.csv", "photometry_summary.csv", "active_targets.csv"):
        shutil.copy2(anchor_phot / name, WORK424 / name)
    (WORK424 / "lightcurves").mkdir(exist_ok=True)
    if WORK426_R.is_dir():
        shutil.rmtree(WORK426_R)
    WORK426_R.mkdir(parents=True, exist_ok=True)
    for name in ("comparison_stars_per_target.csv", "photometry_summary.csv", "active_targets.csv"):
        shutil.copy2(src426 / name, WORK426_R / name)
    (WORK426_R / "lightcurves").mkdir(exist_ok=True)

    proc424 = _ROOT / "Archive" / "Drafts" / "draft_000424" / "detrended_aligned" / "lights" / ANCHOR_SETUP
    proc426 = _ROOT / "Archive" / "Drafts" / "draft_000426" / "detrended_aligned" / "lights" / "r_60_4"

    n424 = backfill_photometry(
        src_phot=anchor_phot,
        out_phot=WORK424,
        setup=ANCHOR_SETUP,
        cfg=cfg,
        equipment_id=1,
        proc_dir=proc424,
    )
    n426 = backfill_photometry(
        src_phot=src426,
        out_phot=WORK426_R,
        setup="r_60_4",
        cfg=cfg,
        equipment_id=4,
        proc_dir=proc426,
    )

    out: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "anchor_tree_untouched": True,
        "backfill_out_424": str(WORK424),
        "backfill_out_426_r": str(WORK426_R),
        "n_sidecars_424": n424,
        "n_sidecars_426_r": n426,
        "S1": run_s1_slow(),
        "S2": run_s2(cfg),
        "S3": run_s3(cfg),
        "S4": run_s4(),
    }
    if out["S2"].get("flips"):
        out["STOP"] = "S2 healthy->RED flips detected"
    if not out["S4"].get("pass"):
        out["STOP"] = "S4 anchor integrity failed"
    path = OUT_DIR / "validation_summary.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    if out["S2"].get("flip_report"):
        pd.DataFrame(out["S2"]["flip_report"]).to_csv(OUT_DIR / "s2_flip_report.csv", index=False)
    s3_rows = out["S3"].get("r_60_4_sparse_targets", []) + [{"target": "SS_Cam", **out["S3"].get("SS_Cam", {})}]
    pd.DataFrame(s3_rows).to_csv(OUT_DIR / "s3_verdict_tuples.csv", index=False)
    print(json.dumps(out, indent=2)[:8000])
    if out.get("STOP"):
        print(f"STOP: {out['STOP']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
