#!/usr/bin/env python3
"""A-6b: measure nearest-neighbour distance for unmatched_in_range DAO_ONLY rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src_py"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import AppConfig  # noqa: E402
from dao_reconcile import annotate_dao_only_magnitude_classes, resolve_effective_match_depth  # noqa: E402
from gaia_catalog_id import read_vyvar_csv  # noqa: E402

DRAFTS = (
    ("draft_000501", "V_60_2"),
    ("draft_000435_snapshot_skysurface_20260716", "NoFilter_60_2"),
    ("draft_000500", "NoFilter_60_2"),
)

THRESHOLDS_PX = (1.0, 2.0, 3.0, 5.0)
BRIGHT_G = 16.3


def _nearest_matched(df: pd.DataFrame, query_idx: int, matched: pd.DataFrame) -> tuple[float, float]:
    x0 = float(df.at[query_idx, "x"])
    y0 = float(df.at[query_idx, "y"])
    xm = pd.to_numeric(matched["x"], errors="coerce").to_numpy(dtype=float)
    ym = pd.to_numeric(matched["y"], errors="coerce").to_numpy(dtype=float)
    d = np.hypot(xm - x0, ym - y0)
    j = int(np.nanargmin(d))
    g = pd.to_numeric(matched.iloc[j].get("phot_g_mean_mag", matched.iloc[j].get("mag")), errors="coerce")
    return float(d[j]), float(g) if pd.notna(g) else float("nan")


def _fraction_within(dist: np.ndarray, thr: float) -> float:
    ok = np.isfinite(dist)
    if not ok.any():
        return float("nan")
    return float(np.mean(dist[ok] <= thr))


def _snr_floor_stats(df: pd.DataFrame) -> dict:
    dao = df["source_type"].astype(str).str.upper().eq("DAO_ONLY") if "source_type" in df.columns else pd.Series(False, index=df.index)
    n_dao = int(dao.sum())
    if "snr50_ok" not in df.columns:
        return {"n_dao_only": n_dao, "n_snr50_fail": None, "frac_snr50_fail": None}
    snr_ok = df.loc[dao, "snr50_ok"]
    fail = ~snr_ok.fillna(False).astype(bool)
    n_fail = int(fail.sum())
    return {
        "n_dao_only": n_dao,
        "n_snr50_fail": n_fail,
        "frac_snr50_fail": float(n_fail / n_dao) if n_dao else float("nan"),
    }


def measure_draft(csv_path: Path, gaia_db: str) -> dict:
    df = read_vyvar_csv(csv_path, low_memory=False, dtype={"catalog_id": str})
    if "source_type" not in df.columns:
        cid = df.get("catalog_id", pd.Series([""] * len(df))).fillna("").astype(str).str.strip()
        df["source_type"] = pd.Series(["DAO_ONLY"] * len(df)).where(cid.eq(""), "GAIA_MATCHED")
    pm_path = csv_path.parent / "photometry" / "pipeline_meta.json"
    match_depth = None
    cone_lim = None
    noise = None
    if pm_path.is_file():
        pm = json.loads(pm_path.read_text(encoding="utf-8"))
        det_meta = {"faintest_mag_limit": pm.get("faintest_mag_limit"), "provenance": pm.get("provenance")}
        match_depth = resolve_effective_match_depth(det_meta, is_masterstar=True).get("match_depth")
        if pm.get("faintest_mag_limit") is not None:
            cone_lim = float(pm["faintest_mag_limit"])
        noise = pm.get("noise_floor_adu")
    out, meta = annotate_dao_only_magnitude_classes(
        df,
        gaia_db_path=gaia_db,
        effective_match_depth=match_depth,
        cone_query_mag_limit=cone_lim,
        frame_noise_adu=noise,
    )
    matched = out[out["source_type"].astype(str).str.upper().ne("DAO_ONLY")].copy()
    in_range = out[out["dao_only_class"] == "unmatched_in_range"]
    dists: list[float] = []
    ng_mag: list[float] = []
    for idx in in_range.index:
        d, g = _nearest_matched(out, idx, matched)
        dists.append(d)
        ng_mag.append(g)
    dist_arr = np.asarray(dists, dtype=float)
    ng_arr = np.asarray(ng_mag, dtype=float)
    bright_neigh = np.isfinite(ng_arr) & (ng_arr < BRIGHT_G)
    report: dict = {
        "confirmable_depth_g": meta.get("confirmable_depth_g"),
        "n_unmatched_in_range": int(len(in_range)),
        "snr_floor": _snr_floor_stats(out),
        "nearest_px": {
            "median": float(np.nanmedian(dist_arr)) if dist_arr.size else None,
            "p90": float(np.nanpercentile(dist_arr, 90)) if dist_arr.size else None,
        },
        "fraction_within_px": {str(t): _fraction_within(dist_arr, t) for t in THRESHOLDS_PX},
        "fraction_within_px_bright_neighbour": {
            str(t): _fraction_within(dist_arr[bright_neigh], t) if bright_neigh.any() else None for t in THRESHOLDS_PX
        },
        "n_bright_neighbour_lt_16p3": int(bright_neigh.sum()),
    }
    # control: random matched sample same size
    n_ctrl = min(len(in_range), len(matched))
    if n_ctrl > 0:
        ctrl = matched.sample(n=n_ctrl, random_state=42)
        ctrl_d: list[float] = []
        for idx in ctrl.index:
            sub = matched.drop(index=idx)
            if sub.empty:
                continue
            d, _ = _nearest_matched(matched, idx, sub)
            ctrl_d.append(d)
        c_arr = np.asarray(ctrl_d, dtype=float)
        report["control_matched_nearest_px"] = {
            "median": float(np.nanmedian(c_arr)) if c_arr.size else None,
            "fraction_within_px": {str(t): _fraction_within(c_arr, t) for t in THRESHOLDS_PX},
        }
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "tmp" / "a6b_split_detection.json")
    args = ap.parse_args()
    cfg = AppConfig()
    gaia_db = str(cfg.gaia_db_path or "")
    archive = Path(cfg.archive_root) / "Drafts"
    results = {}
    for draft, setup in DRAFTS:
        key = f"{draft}/{setup}"
        csv_path = archive / draft / "platesolve" / setup / "masterstars_full_match.csv"
        if not csv_path.is_file():
            results[key] = {"error": f"missing {csv_path}"}
            continue
        results[key] = measure_draft(csv_path, gaia_db)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
