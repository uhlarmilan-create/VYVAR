#!/usr/bin/env python3
"""WIDE-ERR AUDIT 4: per-frame mean comp_resid across check-star fields (read-only)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src_py"))
sys.path.insert(0, str(REPO / "dev" / "scripts"))
import _bootstrap  # noqa: E402,F401

from config import AppConfig  # noqa: E402
from photometry_core import (  # noqa: E402
    _group_comp_mag_inst_from_proc_csvs,
    check_comparison_stability,
    temporal_bin_comp_lc,
)

SETUP = "NoFilter_60_2"
DRAFT = REPO / "Archive" / "Drafts" / "draft_000435_snapshot_skysurface_20260716"
CHECK_CID = "1499906247391001088"
OUT = REPO / "tmp" / "wide_err_audit"


def _iqr(x: np.ndarray) -> float:
    v = np.asarray(x, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    q25, q75 = np.quantile(v, [0.25, 0.75])
    return float(q75 - q25)


def _select_good_ids(
    comp_quality: dict[str, dict],
    comp_rms_map: dict[str, float],
    *,
    n_comp_min: int,
    n_comp_max: int,
) -> list[str]:
    p2p_thr = float("nan")
    for q in comp_quality.values():
        t = q.get("p2p_threshold")
        if t is not None and math.isfinite(float(t)):
            p2p_thr = float(t)
            break
    usable_all = [cid for cid, q in comp_quality.items() if q.get("quality") in ("good", "suspect")]
    usable_sorted = sorted(
        usable_all,
        key=lambda c: (
            0 if comp_quality[c].get("quality") == "good" else 1,
            float(comp_rms_map.get(c, float("inf"))),
            str(c),
        ),
    )
    selected: list[str] = []
    for cid in usable_sorted:
        if len(selected) >= n_comp_max:
            break
        p2p = float(comp_quality[cid].get("rms_p2p", float("nan")))
        if (
            len(selected) < n_comp_min
            or (math.isfinite(p2p_thr) and math.isfinite(p2p) and p2p < p2p_thr)
            or not math.isfinite(p2p_thr)
        ):
            selected.append(cid)
    return selected[:n_comp_max]


def _field_frame_means(
    target_cid: str,
    comp_all: pd.DataFrame,
    csv_files: list[Path],
    cfg: AppConfig,
    all_frames_stub: pd.DataFrame,
) -> dict[str, Any] | None:
    """Check-star production ensemble: parent comps minus check star."""
    comp_ids = [
        str(c).strip()
        for c in comp_all.loc[comp_all["target_catalog_id"].astype(str).str.strip() == target_cid, "catalog_id"]
        if str(c).strip() and str(c).strip() != CHECK_CID
    ]
    if len(comp_ids) < 2:
        return None
    comp_mag_raw = _group_comp_mag_inst_from_proc_csvs(comp_ids, csv_files)
    comp_rms_map = {
        str(r["catalog_id"]).strip(): float(pd.to_numeric(r.get("comp_rms"), errors="coerce"))
        for _, r in comp_all.loc[comp_all["target_catalog_id"].astype(str).str.strip() == target_cid].iterrows()
        if str(r["catalog_id"]).strip() in comp_ids
    }
    n_comp_min = max(1, int(getattr(cfg, "phase01_comparison_n_comp_min", 3)))
    n_comp_max = int(cfg.phase01_comparison_n_comp_max)
    comp_bjd = {cid: np.arange(len(comp_mag_raw[cid]), dtype=np.float64) for cid in comp_ids}
    comp_lc_bin = temporal_bin_comp_lc(
        comp_lc=comp_mag_raw,
        comp_quality={},
        all_frames=all_frames_stub,
        window=int(cfg.temporal_bin_window),
        enabled=bool(cfg.temporal_binning_enabled),
    )
    comp_quality = check_comparison_stability(
        comp_lc_bin,
        comp_rms_map=comp_rms_map,
        comp_bjd=comp_bjd,
        n_comp_min=n_comp_min,
        outlier_sigma=float(getattr(cfg, "stability_sigma", 3.0)),
        max_comp_slope_mmag_hr=float(cfg.comp_max_slope_mmag_hr),
        comp_slope_significance_k=float(getattr(cfg, "comp_slope_significance_k", 3.0)),
        common_mode_detrend=True,
    )
    good_ids = _select_good_ids(comp_quality, comp_rms_map, n_comp_min=n_comp_min, n_comp_max=n_comp_max)
    if len(good_ids) < 2:
        return None

    comp_ref: dict[str, float] = {}
    for cid in good_ids:
        arr = comp_lc_bin.get(cid)
        if arr is None:
            continue
        fin = arr[np.isfinite(arr)]
        if fin.size:
            comp_ref[cid] = float(np.median(fin))

    frame_means: list[float] = []
    n_frames = len(csv_files)
    for i in range(n_frames):
        resid: list[float] = []
        for cid in good_ids:
            if cid not in comp_lc_bin or cid not in comp_ref:
                continue
            mv = float(comp_lc_bin[cid][i])
            if math.isfinite(mv) and math.isfinite(comp_ref[cid]):
                resid.append(mv - comp_ref[cid])
        if len(resid) >= 2:
            frame_means.append(float(np.mean(resid)))

    if not frame_means:
        return None
    fm = np.asarray(frame_means, dtype=np.float64)
    return {
        "target_cid": target_cid,
        "n_frames": int(fm.size),
        "median_frame_mean": float(np.median(fm)),
        "std_frame_mean": float(np.std(fm, ddof=1)) if fm.size >= 2 else float("nan"),
        "global_frame_means": frame_means,
    }


def main() -> int:
    cfg = AppConfig()
    OUT.mkdir(parents=True, exist_ok=True)
    ps = DRAFT / "platesolve" / SETUP
    phot = ps / "photometry"
    lc_dir = phot / "lightcurves"
    lights = DRAFT / "detrended_aligned" / "lights" / SETUP
    comp_all = pd.read_csv(
        phot / "comparison_stars_per_target.csv",
        low_memory=False,
        dtype={"catalog_id": str, "target_catalog_id": str},
    )
    csv_files = sorted(lights.glob("proc_*.csv"))
    all_frames_stub = pd.DataFrame({"catalog_id": [], "bjd": []})

    target_ids = sorted(
        t
        for t in {str(x).strip() for x in comp_all["target_catalog_id"].astype(str)}
        if t and (lc_dir / f"check_kmag_{t}.csv").is_file()
    )

    per_field: list[dict[str, Any]] = []
    all_frame_means: list[float] = []
    for t in target_ids:
        ck = pd.read_csv(lc_dir / f"check_kmag_{t}.csv", nrows=1, low_memory=False)
        id_col = "check_catalog_id" if "check_catalog_id" in ck.columns else "check_cid"
        if str(ck[id_col].iloc[0]).strip() != CHECK_CID:
            continue
        field = _field_frame_means(t, comp_all, csv_files, cfg, all_frames_stub)
        if field is None:
            continue
        per_field.append(
            {
                "target_cid": t,
                "median_frame_mean": field["median_frame_mean"],
                "std_frame_mean": field["std_frame_mean"],
                "n_frames": field["n_frames"],
            }
        )
        all_frame_means.extend(field["global_frame_means"])

    stds = np.asarray([r["std_frame_mean"] for r in per_field], dtype=np.float64)
    medians = np.asarray([r["median_frame_mean"] for r in per_field], dtype=np.float64)
    global_all = np.asarray(all_frame_means, dtype=np.float64)

    out = {
        "check_cid": CHECK_CID,
        "n_fields": len(per_field),
        "per_field_median_of_frame_mean_median": float(np.median(medians)) if medians.size else float("nan"),
        "per_field_median_of_frame_mean_std_across_fields": float(np.std(medians, ddof=1))
        if medians.size >= 2
        else float("nan"),
        "iqr_of_std_frame_mean_across_fields": _iqr(stds),
        "global_median_frame_mean_all_frames_fields": float(np.median(global_all))
        if global_all.size
        else float("nan"),
        "per_field": per_field,
    }
    (OUT / "audit4_comp_resid_frame_mean.json").write_text(
        json.dumps(out, indent=2, sort_keys=True),
        encoding="ascii",
    )
    print(json.dumps({k: v for k, v in out.items() if k != "per_field"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
