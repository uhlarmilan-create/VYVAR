#!/usr/bin/env python3
"""Forensic comp-pool comparison for draft_426 r_60_4 (June stale vs fresh regen)."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from check_star_kmag import build_aligned_comp_inst, comp_ensemble_maps, resolve_proc_csv_dir  # noqa: E402
from comp_pool_rms import compute_global_pool_rms_map, sort_per_frame_csv_paths  # noqa: E402
from config import AppConfig  # noqa: E402
from gaia_catalog_id import normalize_gaia_source_id  # noqa: E402
from photometry_core import (  # noqa: E402
    _bprp_tier_ladder_for_selection,
    _select_comps_by_color_then_rms,
    check_comparison_stability,
    parse_comp_quality_json_map,
)
from proc_frame_store import ProcFrameStore  # noqa: E402

DRAFT_ID = 426
SETUP = "r_60_4"
TARGET_V0611 = "1112127291051695744"
EVIDENCE = "draft_000426_stale_20260626"
OUT_DIR = _ROOT / "tmp" / "comp_pool_r"


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _norm(s: Any) -> str:
    return str(normalize_gaia_source_id(s) or "").strip()


def _good_from_cq(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {_norm(k) for k, v in parse_comp_quality_json_map(raw).items() if v.get("quality") == "good"}


def _load_side(label: str, archive: Path) -> dict[str, Any]:
    if label == "stale":
        base = archive / "evidence" / EVIDENCE
    else:
        base = archive / "Drafts" / f"draft_{DRAFT_ID:06d}"
    ps = base / "platesolve" / SETUP
    phot = ps / "photometry"
    proc_dir = base / "detrended_aligned" / "lights" / SETUP
    store = ProcFrameStore.build(proc_dir)
    csv_paths = [Path(k) for k in store.keys()]
    ms = pd.read_csv(ps / "masterstars_full_match.csv", low_memory=False, dtype={"catalog_id": str})
    pt = pd.read_csv(
        phot / "comparison_stars_per_target.csv",
        low_memory=False,
        dtype={"catalog_id": str, "target_catalog_id": str},
    )
    pt_v = pt[pt["target_catalog_id"].map(_norm) == TARGET_V0611].copy()
    cq_path = phot / "lightcurves" / f"comp_quality_{TARGET_V0611}.json"
    lc_path = phot / "lightcurves" / f"lightcurve_{TARGET_V0611}.csv"
    lc_df = pd.read_csv(lc_path, low_memory=False) if lc_path.is_file() else pd.DataFrame()
    source_files = lc_df["source_file"].astype(str).tolist() if "source_file" in lc_df.columns else []
    return {
        "label": label,
        "base": base,
        "phot": phot,
        "proc_dir": proc_dir,
        "store": store,
        "csv_paths": csv_paths,
        "ms": ms,
        "per_target": pt_v,
        "good_cq": _good_from_cq(cq_path),
        "source_files": source_files,
        "lc_df": lc_df,
    }


def _recompute_pool_rms(side: dict[str, Any], cand_ids: set[str], cfg: AppConfig) -> dict[str, float]:
    cache = side["store"]
    paths = side["csv_paths"]
    ms = side["ms"]
    chip_fw = int(pd.to_numeric(ms["x"], errors="coerce").max()) if "x" in ms.columns else None
    chip_fh = int(pd.to_numeric(ms["y"], errors="coerce").max()) if "y" in ms.columns else None
    return compute_global_pool_rms_map(
        cand_ids,
        ms,
        paths,
        cache,
        flux_col="dao_flux",
        min_frames_frac=float(cfg.phase01_comparison_min_frames_frac),
        max_comp_rms=float(cfg.phase01_comparison_max_comp_rms),
        chip_fw=chip_fw,
        chip_fh=chip_fh,
    )


def _gate_stability(
    comp_ids: list[str],
    source_files: list[str],
    proc_dir: Path,
    cfg: AppConfig,
    comp_rms_map: dict[str, float],
) -> dict[str, dict]:
    if not comp_ids or not source_files:
        return {}
    comp_lc = build_aligned_comp_inst(proc_dir, comp_ids, source_files, cfg, "aperture")
    comp_bjd: dict[str, np.ndarray] = {}
    for cid in comp_ids:
        if cid not in comp_lc:
            continue
        comp_bjd[cid] = np.arange(len(comp_lc[cid]), dtype=float)
    return check_comparison_stability(
        comp_lc,
        comp_rms_map=comp_rms_map,
        comp_bjd=comp_bjd,
        n_comp_min=3,
        outlier_sigma=3.0,
        max_comp_slope_mmag_hr=float(cfg.comp_max_slope_mmag_hr),
        comp_slope_significance_k=float(cfg.comp_slope_significance_k),
        common_mode_detrend=True,
    )


def _color_gate_verdict(
    bp_rp: float,
    target_bprp: float,
    cfg: AppConfig,
    *,
    max_delta_bprp: float = 0.79,
) -> tuple[bool, float, str]:
    if not math.isfinite(bp_rp) or not math.isfinite(target_bprp):
        return True, float("nan"), "no_bprp"
    delta = abs(float(bp_rp) - float(target_bprp))
    ladder = _bprp_tier_ladder_for_selection(cfg, max_delta_bprp)
    first = float(ladder[0]) if ladder else float(max_delta_bprp)
    ok = delta <= first
    return ok, delta, f"delta={delta:.3f} thr={first:.3f}"


def build_table(archive: Path, cfg: AppConfig) -> dict[str, Any]:
    stale = _load_side("stale", archive)
    fresh = _load_side("fresh", archive)
    union = sorted(stale["good_cq"] | fresh["good_cq"])

    # target bp_rp from per-target (stale/fresh)
    def _target_bprp(side: dict[str, Any]) -> float:
        pt = side["per_target"]
        if pt.empty or "target_bp_rp" not in pt.columns:
            return float("nan")
        return float(pd.to_numeric(pt["target_bp_rp"].iloc[0], errors="coerce"))

    stale_tbprp = _target_bprp(stale)
    fresh_tbprp = _target_bprp(fresh)

    # field rows for bp_rp / mag
    def _ms_row(cid: str, ms: pd.DataFrame) -> dict[str, float]:
        m = ms[ms["catalog_id"].map(_norm) == cid]
        if m.empty:
            return {"bp_rp": float("nan"), "mag": float("nan")}
        r = m.iloc[0]
        return {
            "bp_rp": float(pd.to_numeric(r.get("bp_rp"), errors="coerce")),
            "mag": float(pd.to_numeric(r.get("mag", r.get("phot_g_mean_mag")), errors="coerce")),
        }

    stale_rms_re = _recompute_pool_rms(stale, set(union), cfg)
    fresh_rms_re = _recompute_pool_rms(fresh, set(union), cfg)

    stale_pt = { _norm(r["catalog_id"]): r for _, r in stale["per_target"].iterrows() }
    fresh_pt = { _norm(r["catalog_id"]): r for _, r in fresh["per_target"].iterrows() }

    stale_stab = _gate_stability(
        union, stale["source_files"], stale["proc_dir"], cfg,
        {c: stale_rms_re.get(c, float("nan")) for c in union},
    )
    fresh_stab = _gate_stability(
        union, fresh["source_files"], fresh["proc_dir"], cfg,
        {c: fresh_rms_re.get(c, float("nan")) for c in union},
    )

    rows: list[dict[str, Any]] = []
    max_rms = float(cfg.phase01_comparison_max_comp_rms)
    floor = float(cfg.comp_select_rms_floor)

    for cid in union:
        ms_s = _ms_row(cid, stale["ms"])
        ms_f = _ms_row(cid, fresh["ms"])
        rms_s_pt = float(stale_pt[cid]["comp_rms"]) if cid in stale_pt else float("nan")
        rms_f_pt = float(fresh_pt[cid]["comp_rms"]) if cid in fresh_pt else float("nan")
        rms_s_re = float(stale_rms_re.get(cid, float("nan")))
        rms_f_re = float(fresh_rms_re.get(cid, float("nan")))

        cg_s, d_s, cg_note_s = _color_gate_verdict(ms_s["bp_rp"], stale_tbprp, cfg)
        cg_f, d_f, cg_note_f = _color_gate_verdict(ms_f["bp_rp"], fresh_tbprp, cfg)

        stab_s = stale_stab.get(cid, {})
        stab_f = fresh_stab.get(cid, {})

        june_good = cid in stale["good_cq"]
        regen_good = cid in fresh["good_cq"]

        def _flip_reason() -> str:
            if june_good == regen_good:
                return "unchanged"
            if june_good and not regen_good:
                if cid not in fresh_pt:
                    if not cg_f:
                        return "not_selected_phase1_color_or_rms"
                    if math.isfinite(rms_f_re) and rms_f_re > max_rms:
                        return "not_selected_phase1_rms_gate"
                    return "not_selected_phase1_pool"
                if stab_f.get("quality") == "excluded":
                    return f"phase2a_stability:{stab_f.get('note','')}"
                return "phase2a_or_phase1"
            if regen_good and not june_good:
                return "newly_selected"
            return "unknown"

        rows.append(
            {
                "catalog_id": cid,
                "june_good": june_good,
                "regen_good": regen_good,
                "flip_reason": _flip_reason(),
                "stale": {
                    "selected_phase1": cid in stale_pt,
                    "comp_rms_per_target": rms_s_pt,
                    "comp_rms_recomputed": rms_s_re,
                    "comp_tier": int(stale_pt[cid]["comp_tier"]) if cid in stale_pt else None,
                    "bp_rp": ms_s["bp_rp"],
                    "delta_bprp": d_s,
                    "color_gate_pass": cg_s,
                    "rms_gate_pass": math.isfinite(rms_s_re) and rms_s_re <= max_rms,
                    "rms_floor_pass": math.isfinite(rms_s_re) and rms_s_re >= floor,
                    "p2p_rms": stab_s.get("rms_p2p"),
                    "p2p_quality": stab_s.get("quality"),
                    "p2p_note": stab_s.get("note"),
                },
                "fresh": {
                    "selected_phase1": cid in fresh_pt,
                    "comp_rms_per_target": rms_f_pt,
                    "comp_rms_recomputed": rms_f_re,
                    "comp_tier": int(fresh_pt[cid]["comp_tier"]) if cid in fresh_pt else None,
                    "bp_rp": ms_f["bp_rp"],
                    "delta_bprp": d_f,
                    "color_gate_pass": cg_f,
                    "rms_gate_pass": math.isfinite(rms_f_re) and rms_f_re <= max_rms,
                    "rms_floor_pass": math.isfinite(rms_f_re) and rms_f_re >= floor,
                    "p2p_rms": stab_f.get("rms_p2p"),
                    "p2p_quality": stab_f.get("quality"),
                    "p2p_note": stab_f.get("note"),
                },
            }
        )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": _git_head(),
        "setup": SETUP,
        "target_catalog_id": TARGET_V0611,
        "config_gates": {
            "phase01_comparison_max_comp_rms": max_rms,
            "comp_select_rms_floor": floor,
            "comp_max_slope_mmag_hr": float(cfg.comp_max_slope_mmag_hr),
            "comp_slope_significance_k": float(cfg.comp_slope_significance_k),
            "pytics_enabled": bool(cfg.pytics_enabled),
        },
        "target_bp_rp": {"stale": stale_tbprp, "fresh": fresh_tbprp},
        "n_june_good": len(stale["good_cq"]),
        "n_regen_good": len(fresh["good_cq"]),
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive-root", type=Path, default=_ROOT / "Archive")
    args = ap.parse_args()
    cfg = AppConfig()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = build_table(args.archive_root.resolve(), cfg)
    out = OUT_DIR / "per_comp_table.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
