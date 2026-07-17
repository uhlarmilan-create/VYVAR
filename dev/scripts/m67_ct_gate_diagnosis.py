#!/usr/bin/env python3
"""Diagnose Green/Red CT gate failures for draft_000368 (Step 1)."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401  (repo layout: src_py + dev on sys.path)
_ROOT = _bootstrap.REPO_ROOT
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import AppConfig  # noqa: E402
from photometry_core import (  # noqa: E402
    _check_color_term_extrapolation,
    should_apply_color_term,
)

MIN_COMP_CT = 7
MAX_STDERR_RATIO = 0.5
GR_FILTERS = ("Green", "Red")


def _normalize_cid(val) -> str:
    s = str(val or "").strip()
    if not s or s.lower() in ("nan", "none"):
        return ""
    if "e" in s.lower():
        try:
            return str(int(float(s)))
        except (TypeError, ValueError):
            return s
    if s.endswith(".0"):
        s = s[:-2]
    return s


def _classify_gate(
    *,
    n_comp: int,
    c1: float,
    c1_stderr: float,
    stderr_ratio: float,
    in_range: bool,
    apply_ct: bool,
    ct_ok: bool,
) -> str:
    if int(n_comp) == 0:
        return "a_fit_starved"
    if not in_range:
        return "d_extrapolation"
    if int(n_comp) < MIN_COMP_CT:
        return "b_count"
    if not (math.isfinite(stderr_ratio) and float(stderr_ratio) <= MAX_STDERR_RATIO):
        return "c_stderr"
    if apply_ct and not ct_ok:
        return "d_extrapolation"
    return "pass"


def _gate_reason_from_row(n_comp: int, c1: float, c1_stderr: float, obs_group: str) -> str:
    apply, reason = should_apply_color_term(
        obs_group=obs_group.split("_")[0],
        c1=float(c1),
        c1_stderr=float(c1_stderr),
        n_comp=int(n_comp),
        min_comp_for_ct=MIN_COMP_CT,
        max_stderr_ratio=MAX_STDERR_RATIO,
    )
    return reason


def _comp_pool_stats(comp_csv: Path) -> tuple[float, float, int, int]:
    comp = pd.read_csv(comp_csv, low_memory=False)
    bps = pd.to_numeric(comp.get("bp_rp"), errors="coerce").dropna()
    sat_n = 0
    if "is_saturated" in comp.columns:
        sat_n = int(pd.to_numeric(comp["is_saturated"], errors="coerce").fillna(0).astype(bool).sum())
    elif "likely_saturated" in comp.columns:
        sat_n = int(pd.to_numeric(comp["likely_saturated"], errors="coerce").fillna(0).astype(bool).sum())
    return (
        float(bps.min()) if len(bps) else float("nan"),
        float(bps.max()) if len(bps) else float("nan"),
        int(len(bps)),
        sat_n,
    )


def _target_comp_bp_rp(comp_pt_csv: Path, target_cid: str) -> tuple[float, float, list[float]]:
    if not comp_pt_csv.is_file():
        return float("nan"), float("nan"), []
    df = pd.read_csv(comp_pt_csv, low_memory=False, dtype={"catalog_id": str, "target_catalog_id": str})
    sub = df[df["target_catalog_id"].astype(str) == str(target_cid)]
    if sub.empty or "bp_rp" not in sub.columns:
        return float("nan"), float("nan"), []
    bps = pd.to_numeric(sub["bp_rp"], errors="coerce").dropna()
    vals = [float(v) for v in bps if np.isfinite(v)]
    if not vals:
        return float("nan"), float("nan"), []
    return float(min(vals)), float(max(vals)), vals


def _lc_first_row(lc_path: Path) -> dict:
    if not lc_path.is_file():
        return {}
    try:
        return pd.read_csv(lc_path, nrows=1, low_memory=False).iloc[0].to_dict()
    except Exception:  # noqa: BLE001
        return {}


def diagnose_draft(draft_id: int) -> dict:
    cfg = AppConfig()
    draft = Path(cfg.archive_root) / "Drafts" / f"draft_{draft_id:06d}"
    ps_root = draft / "platesolve"
    rows: list[dict] = []
    group_meta: list[dict] = []

    for og_dir in sorted(ps_root.iterdir()):
        if not og_dir.is_dir():
            continue
        flt = og_dir.name.split("_")[0]
        if flt not in GR_FILTERS:
            continue
        vt_csv = og_dir / "variable_targets.csv"
        if not vt_csv.is_file():
            continue
        vt = pd.read_csv(vt_csv, low_memory=False, dtype={"catalog_id": str})
        comp_csv = og_dir / "comparison_stars.csv"
        comp_pt = og_dir / "photometry" / "comparison_stars_per_target.csv"
        pool_min, pool_max, pool_n, pool_sat = _comp_pool_stats(comp_csv)

        dt_dir = draft / "detrended_aligned" / "lights" / og_dir.name
        n_frames = len(list(dt_dir.glob("*_cal.csv"))) if dt_dir.is_dir() else 0

        group_meta.append(
            {
                "obs_group": og_dir.name,
                "filter": flt,
                "n_frames": n_frames,
                "comp_pool_n": pool_n,
                "comp_bp_rp_min": pool_min,
                "comp_bp_rp_max": pool_max,
                "comp_bp_rp_width": float(pool_max - pool_min) if np.isfinite(pool_min) and np.isfinite(pool_max) else float("nan"),
                "comp_saturated_count": pool_sat,
            }
        )

        for _, trow in vt.iterrows():
            cid = _normalize_cid(trow.get("catalog_id"))
            if not cid:
                continue
            lc_path = og_dir / "photometry" / "lightcurves" / f"lightcurve_{cid}.csv"
            lc0 = _lc_first_row(lc_path)
            tgt_bp = pd.to_numeric(trow.get("bp_rp"), errors="coerce")
            if not math.isfinite(float(tgt_bp)):
                tgt_bp = pd.to_numeric(lc0.get("target_bp_rp"), errors="coerce")
            comp_min, comp_max, comp_bps = _target_comp_bp_rp(comp_pt, cid)
            if not comp_bps and comp_pt.is_file():
                comp_min, comp_max = pool_min, pool_max
                comp_bps = []

            n_comp = int(pd.to_numeric(lc0.get("ct_n_comp", lc0.get("n_comp_used", 0)), errors="coerce") or 0)
            c1 = float(pd.to_numeric(lc0.get("ct_c1", lc0.get("c1", 0)), errors="coerce") or 0.0)
            c1_stderr = float(pd.to_numeric(lc0.get("ct_c1_stderr", lc0.get("c1_stderr")), errors="coerce"))
            stderr_ratio = float(pd.to_numeric(lc0.get("ct_stderr_ratio", lc0.get("stderr_ratio")), errors="coerce"))
            if not math.isfinite(stderr_ratio) and c1 != 0.0 and math.isfinite(c1_stderr):
                stderr_ratio = abs(c1_stderr / c1)

            if not lc0 and (og_dir / "photometry" / "photometry_summary.csv").is_file():
                summ = pd.read_csv(og_dir / "photometry" / "photometry_summary.csv", low_memory=False, dtype={"catalog_id": str})
                ssub = summ[summ["catalog_id"].astype(str) == cid]
                if not ssub.empty:
                    sr = ssub.iloc[0]
                    n_comp = int(pd.to_numeric(sr.get("n_good_comp", 0), errors="coerce") or 0)

            proto = pd.read_csv(draft / "ct_prototype.csv", low_memory=False, dtype={"catalog_id": str}) if (draft / "ct_prototype.csv").is_file() else pd.DataFrame()
            psub = proto[(proto["catalog_id"].astype(str) == cid) & (proto["obs_group"].astype(str) == flt)]
            if not psub.empty:
                pr = psub.iloc[-1]
                n_comp = int(pr.get("n_comp_used", n_comp) or n_comp)
                c1 = float(pr.get("c1", c1) or c1)
                c1_stderr = float(pd.to_numeric(pr.get("c1_stderr"), errors="coerce"))
                stderr_ratio = float(pd.to_numeric(pr.get("stderr_ratio"), errors="coerce"))
                if not math.isfinite(float(tgt_bp)):
                    tgt_bp = float(pr.get("target_bp_rp"))

            tgt_bp_f = float(tgt_bp) if math.isfinite(float(tgt_bp)) else float("nan")
            in_range = True
            if len(comp_bps) >= 2:
                in_range = _check_color_term_extrapolation(tgt_bp_f, comp_bps, extrapolation_tol=0.0)
            elif math.isfinite(pool_min) and math.isfinite(pool_max) and math.isfinite(tgt_bp_f):
                in_range = pool_min <= tgt_bp_f <= pool_max

            ct_ok = str(lc0.get("ct_ok", "")).strip().lower() in ("true", "1", "yes")
            apply_ct, _ = should_apply_color_term(
                obs_group=flt,
                c1=c1,
                c1_stderr=c1_stderr,
                n_comp=n_comp,
                min_comp_for_ct=MIN_COMP_CT,
                max_stderr_ratio=MAX_STDERR_RATIO,
            )
            bucket = _classify_gate(
                n_comp=n_comp,
                c1=c1,
                c1_stderr=c1_stderr,
                stderr_ratio=stderr_ratio,
                in_range=in_range,
                apply_ct=apply_ct,
                ct_ok=ct_ok,
            )
            presel_in_range = math.isfinite(tgt_bp_f) and math.isfinite(pool_min) and pool_min <= tgt_bp_f <= pool_max

            rows.append(
                {
                    "catalog_id": cid,
                    "obs_group": og_dir.name,
                    "filter": flt,
                    "target_bp_rp": tgt_bp_f,
                    "comp_bp_rp_min": comp_min if math.isfinite(comp_min) else pool_min,
                    "comp_bp_rp_max": comp_max if math.isfinite(comp_max) else pool_max,
                    "presel_in_range": presel_in_range,
                    "n_comp_used": n_comp,
                    "c1": c1,
                    "c1_stderr": c1_stderr,
                    "stderr_ratio": stderr_ratio,
                    "gate_would_pass_min7": apply_ct,
                    "ct_ok": ct_ok,
                    "gate_bucket": bucket,
                    "gate_reason": _gate_reason_from_row(n_comp, c1, c1_stderr, og_dir.name),
                }
            )

    df = pd.DataFrame(rows)
    out_csv = _ROOT / f"m67_ct_gate_diagnosis_draft{draft_id:06d}.csv"
    df.to_csv(out_csv, index=False)

    aggregates: dict = {"group_meta": group_meta, "by_filter": {}, "in_range_presel_only": {}}
    for flt in GR_FILTERS:
        sub = df[df["filter"] == flt]
        presel = sub[sub["presel_in_range"] == True]  # noqa: E712
        counts = sub["gate_bucket"].value_counts().to_dict()
        presel_counts = presel["gate_bucket"].value_counts().to_dict()
        aggregates["by_filter"][flt] = {
            "n_targets": int(len(sub)),
            "gate_buckets_all": counts,
            "gate_buckets_in_range_presel": presel_counts,
            "ct_ok_true": int(sub["ct_ok"].sum()),
            "gate_would_pass_min7": int(sub["gate_would_pass_min7"].sum()),
        }
        aggregates["in_range_presel_only"][flt] = presel_counts

    result = {
        "draft_id": draft_id,
        "min_comp_ct": MIN_COMP_CT,
        "max_stderr_ratio": MAX_STDERR_RATIO,
        "per_target_csv": str(out_csv),
        "n_rows": int(len(df)),
        **aggregates,
    }
    out_json = _ROOT / f"m67_ct_gate_diagnosis_draft{draft_id:06d}.json"
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draft", type=int, default=368)
    args = ap.parse_args()
    rep = diagnose_draft(int(args.draft))
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
