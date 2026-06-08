#!/usr/bin/env python3
"""Backfill check_kmag_{target}.csv sidecars for export (additive; no LC recompute)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import AppConfig
from check_star_kmag import (
    check_kmag_sidecar_path,
    compute_check_ensemble_mag_calib,
    comp_ensemble_maps,
    resolve_proc_csv_dir,
    save_check_kmag_sidecar,
    select_check_star,
    build_aligned_comp_inst,
)
from gaia_catalog_id import normalize_gaia_source_id
from photometry_core import check_comparison_stability, parse_comp_quality_json_map


def _norm_id(x: object) -> str:
    return str(normalize_gaia_source_id(x) or "").strip()


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--draft", type=int, required=True)
    parser.add_argument("--setup", type=str, default="NoFilter_60_2")
    args = parser.parse_args()

    cfg = AppConfig()
    phot = _ROOT / "Archive" / "Drafts" / f"draft_{args.draft:06d}" / "platesolve" / args.setup / "photometry"
    lc_dir = phot / "lightcurves"
    proc_dir = resolve_proc_csv_dir(phot, args.setup)
    if proc_dir is None:
        print(f"FATAL: proc dir not found for {args.setup}", file=sys.stderr)
        return 1

    comp_all = pd.read_csv(phot / "comparison_stars_per_target.csv", dtype={"catalog_id": str, "target_catalog_id": str})
    comp_index = {_norm_id(tid): sub.copy() for tid, sub in comp_all.groupby("target_catalog_id")}
    proc_cache: dict[str, pd.DataFrame] = {}
    n_ok = 0

    for lc_path in sorted(lc_dir.glob("lightcurve_*.csv")):
        if "_psf" in lc_path.stem or "_adaptive" in lc_path.stem:
            continue
        target_cid = lc_path.stem.replace("lightcurve_", "", 1)
        lc_df = pd.read_csv(lc_path, low_memory=False)
        if lc_df.empty or "source_file" not in lc_df.columns:
            continue
        comp_df = comp_index.get(_norm_id(target_cid), pd.DataFrame())
        chk = select_check_star(comp_df)
        if chk is None:
            continue
        check_cid = _norm_id(chk.get("catalog_id", ""))
        comp_ids = [_norm_id(c) for c in comp_df["catalog_id"].tolist() if _norm_id(c)]
        if check_cid not in comp_ids:
            comp_ids.append(check_cid)
        source_files = lc_df["source_file"].astype(str).tolist()
        comp_lc = build_aligned_comp_inst(proc_dir, comp_ids, source_files, cfg, "aperture", csv_cache=proc_cache)
        cat, tier, rms, tw = comp_ensemble_maps(comp_df, cfg)
        cq_path = lc_dir / f"comp_quality_{target_cid}.json"
        cq_map: dict[str, str] = {}
        if cq_path.is_file():
            raw = json.loads(cq_path.read_text(encoding="utf-8"))
            for qk, qv in parse_comp_quality_json_map(raw).items():
                cq_map[_norm_id(qk)] = str(qv.get("quality", "")).strip().lower()
        other_ids = [c for c in comp_ids if c != check_cid]
        other_lc = {c: comp_lc[c] for c in other_ids if c in comp_lc}
        comp_quality = check_comparison_stability(other_lc, comp_rms_map=rms, n_comp_min=3, outlier_sigma=3.0, common_mode_detrend=True)
        for cid, q in cq_map.items():
            if cid in comp_quality and q == "excluded":
                comp_quality[cid]["quality"] = "excluded"
        kmag = compute_check_ensemble_mag_calib(check_cid, comp_ids, comp_lc, cat, comp_quality, comp_rms_map=rms, comp_tier_map=tier, tier_weights=tw, cfg=cfg)
        if kmag is None:
            continue
        bjd = pd.to_numeric(lc_df["bjd"], errors="coerce").to_numpy(dtype=float)
        save_check_kmag_sidecar(
            check_kmag_sidecar_path(lc_dir, target_cid),
            check_cid=check_cid,
            bjd=bjd,
            source_files=source_files,
            kmag=kmag,
        )
        n_ok += 1

    print(f"Wrote {n_ok} check_kmag sidecars -> {lc_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
