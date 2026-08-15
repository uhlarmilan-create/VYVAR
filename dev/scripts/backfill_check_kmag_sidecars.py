#!/usr/bin/env python3
"""Backfill check_kmag sidecars (additive; writes only to --out-phot work dir)."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401  (repo layout: src_py + dev on sys.path)
_ROOT = _bootstrap.REPO_ROOT
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import AppConfig
from check_star_kmag import (
    build_comp_photon_mag_from_frames,
    check_kmag_sidecar_path,
    compute_check_ensemble_mag_calib,
    comp_ensemble_maps,
    field_check_star_candidate_pool,
    resolve_proc_csv_dir,
    save_check_kmag_sidecar,
    select_check_star,
    select_external_check_star,
    build_aligned_comp_inst,
    _ensemble_median_bprp,
    _target_mag_from_row,
)
from gaia_catalog_id import normalize_gaia_source_id
from photometry_core import (
    check_comparison_stability,
    ensemble_member_ids,
    parse_comp_quality_json_map,
)
from sigma_floor_core import resolve_sigma_sys_mag


def _norm_id(x: object) -> str:
    return str(normalize_gaia_source_id(x) or "").strip()


def _resolve_phot(src: Path, draft: int | None, setup: str) -> Path:
    if src is not None:
        return Path(src)
    if draft is not None:
        return _ROOT / "Archive" / "Drafts" / f"draft_{draft:06d}" / "platesolve" / setup / "photometry"
    raise ValueError("need --src-phot or --draft")


def _copy_photometry_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=False)


def _load_proc_frames(proc_dir: Path, comp_ids: list[str], source_files: list[str]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for sf in source_files:
        p = proc_dir / str(sf)
        if not p.is_file():
            continue
        try:
            df = pd.read_csv(p, low_memory=False, dtype={"catalog_id": str})
        except Exception:  # noqa: BLE001
            continue
        df = df.copy()
        df["source_file"] = str(sf)
        if "catalog_id" in df.columns:
            df["_nid"] = df["catalog_id"].map(lambda x: _norm_id(x))
            sub = df[df["_nid"].isin(comp_ids)]
            if not sub.empty:
                rows.append(sub)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def backfill_photometry(
    *,
    src_phot: Path,
    out_phot: Path,
    setup: str,
    cfg: AppConfig,
    equipment_id: int | None = None,
    proc_dir: Path | None = None,
) -> int:
    """Write check_kmag sidecars under out_phot; src_phot is read-only."""
    src_phot = Path(src_phot)
    out_phot = Path(out_phot)
    lc_src = src_phot / "lightcurves"
    lc_out = out_phot / "lightcurves"
    lc_out.mkdir(parents=True, exist_ok=True)
    if proc_dir is None:
        proc_dir = resolve_proc_csv_dir(src_phot, setup)
    else:
        proc_dir = Path(proc_dir)
    if proc_dir is None or not proc_dir.is_dir():
        raise FileNotFoundError(f"proc dir not found for {setup}")
    sigma_sys = resolve_sigma_sys_mag(equipment_id, cfg, rig_label=setup)
    comp_all = pd.read_csv(
        src_phot / "comparison_stars_per_target.csv",
        dtype={"catalog_id": str, "target_catalog_id": str},
    )
    comp_index = {_norm_id(tid): sub.copy() for tid, sub in comp_all.groupby("target_catalog_id")}
    field_pool = field_check_star_candidate_pool(comp_all, target_comps=None)
    proc_cache: dict[str, pd.DataFrame] = {}
    n_ok = 0
    for lc_path in sorted(lc_src.glob("lightcurve_*.csv")):
        if "_psf" in lc_path.stem or "_adaptive" in lc_path.stem:
            continue
        target_cid = lc_path.stem.replace("lightcurve_", "", 1)
        lc_df = pd.read_csv(lc_path, low_memory=False)
        if lc_df.empty or "source_file" not in lc_df.columns:
            continue
        comp_df = comp_index.get(_norm_id(target_cid), pd.DataFrame())
        cat, tier, rms, tw = comp_ensemble_maps(comp_df, cfg)
        cq_path = lc_src / f"comp_quality_{target_cid}.json"
        cq_map: dict[str, str] = {}
        comp_quality_full: dict[str, dict] = {}
        if cq_path.is_file():
            raw = json.loads(cq_path.read_text(encoding="utf-8"))
            comp_quality_full = parse_comp_quality_json_map(raw)
            for qk, qv in comp_quality_full.items():
                cq_map[_norm_id(qk)] = str(qv.get("quality", "")).strip().lower()
        ens_ids = (
            ensemble_member_ids(
                comp_quality_full,
                rms,
                n_comp_min=2,
                n_comp_max=int(cfg.phase01_comparison_n_comp_max),
            )
            if comp_quality_full
            else set()
        )
        n_ens = len(ens_ids)
        sparse_branch = n_ens <= 2
        target_mag = _target_mag_from_row(comp_df, target_cid)
        target_bprp = float("nan")
        if not comp_df.empty and "bp_rp" in comp_df.columns:
            br = pd.to_numeric(comp_df.get("bp_rp"), errors="coerce")
            if br.notna().any():
                target_bprp = float(br.iloc[0])
        k_source = ""
        k_colour_offset = float("nan")
        k_tier_excluded = False
        chk = None
        if sparse_branch:
            ext = select_external_check_star(
                field_check_star_candidate_pool(field_pool, target_comps=comp_df),
                ensemble_ids=ens_ids,
                target_mag=target_mag,
                target_bprp=target_bprp if math.isfinite(target_bprp) else None,
                ensemble_bprp_median=_ensemble_median_bprp(comp_df, ens_ids),
                cfg=cfg,
            )
            if ext is not None:
                chk = ext.row
                k_source = ext.k_source
                k_colour_offset = ext.k_colour_offset
                k_tier_excluded = ext.k_tier_excluded
        else:
            chk = select_check_star(
                field_check_star_candidate_pool(field_pool, target_comps=comp_df),
                ensemble_ids=ens_ids,
                cfg=cfg,
                n_comp_min=3,
            )
        sidecar_path = check_kmag_sidecar_path(lc_out, target_cid)
        existing_src_side = check_kmag_sidecar_path(lc_src, target_cid)
        check_cid = ""
        if existing_src_side.is_file():
            try:
                sdf = pd.read_csv(existing_src_side, nrows=1, low_memory=False)
                check_cid = _norm_id(sdf.get("check_catalog_id", [""]).iloc[0])
            except Exception:  # noqa: BLE001
                check_cid = ""
        if chk is None and not check_cid and len(comp_df) >= 2 and sparse_branch:
            ext = select_external_check_star(
                field_check_star_candidate_pool(field_pool, target_comps=comp_df),
                ensemble_ids=ens_ids,
                target_mag=target_mag,
                target_bprp=target_bprp if math.isfinite(target_bprp) else None,
                ensemble_bprp_median=_ensemble_median_bprp(comp_df, ens_ids),
                cfg=cfg,
            )
            if ext is not None:
                chk = ext.row
                k_source = ext.k_source
                k_colour_offset = ext.k_colour_offset
                k_tier_excluded = ext.k_tier_excluded
        if chk is None and not check_cid and len(comp_df) >= 2 and not sparse_branch:
            df_pick = comp_df.copy()
            if "comp_rms" in df_pick.columns:
                df_pick["comp_rms"] = pd.to_numeric(df_pick["comp_rms"], errors="coerce")
                df_pick = df_pick.sort_values("comp_rms", ascending=True, kind="mergesort")
            check_cid = _norm_id(df_pick.iloc[0].get("catalog_id", ""))
            ens_ids = set()
        if chk is None and not check_cid:
            if sidecar_path.is_file():
                sidecar_path.unlink()
            continue
        if not check_cid:
            check_cid = _norm_id(chk.get("catalog_id", ""))
        if not check_cid:
            check_cid = _norm_id(chk.get("catalog_id", ""))
        ensemble_ids_list = sorted(_norm_id(c) for c in ens_ids if _norm_id(c))
        comp_ids = list(dict.fromkeys(ensemble_ids_list + ([check_cid] if check_cid else [])))
        source_files = lc_df["source_file"].astype(str).tolist()
        comp_lc = build_aligned_comp_inst(proc_dir, comp_ids, source_files, cfg, "aperture", csv_cache=proc_cache)
        other_ids = [c for c in comp_ids if c != check_cid]
        other_lc = {c: comp_lc[c] for c in other_ids if c in comp_lc}
        comp_quality = check_comparison_stability(
            other_lc, comp_rms_map=rms, n_comp_min=2, outlier_sigma=3.0, common_mode_detrend=True,
        )
        for cid, q in cq_map.items():
            if cid in comp_quality and q == "excluded":
                comp_quality[cid]["quality"] = "excluded"
        photon_ids = list(dict.fromkeys(comp_ids + [check_cid]))
        frames = _load_proc_frames(proc_dir, photon_ids, source_files)
        comp_photon = build_comp_photon_mag_from_frames(frames, photon_ids, source_files, cfg=cfg)
        airmass_arr = pd.to_numeric(lc_df.get("airmass"), errors="coerce").to_numpy(dtype=float)
        kmag_result = compute_check_ensemble_mag_calib(
            check_cid, comp_ids, comp_lc, cat, comp_quality,
            comp_rms_map=rms, comp_tier_map=tier, tier_weights=tw, cfg=cfg,
            n_comp_min=3 if not sparse_branch else 2,
            comp_photon_mag=comp_photon, sigma_sys_mag=sigma_sys,
            airmass=airmass_arr,
            k_source=k_source,
            k_colour_offset=k_colour_offset,
            k_tier_excluded=k_tier_excluded,
            sparse_external_k=sparse_branch,
        )
        if kmag_result is None:
            continue
        bjd = pd.to_numeric(lc_df["bjd"], errors="coerce").to_numpy(dtype=float)
        save_check_kmag_sidecar(
            sidecar_path,
            check_cid=check_cid,
            bjd=bjd,
            source_files=source_files,
            kmag=kmag_result.kmag,
            ensemble=kmag_result,
        )
        n_ok += 1
    return n_ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft", type=int, help="draft id (legacy; uses live draft tree)")
    parser.add_argument("--setup", type=str, default="NoFilter_60_2")
    parser.add_argument("--src-phot", type=Path, help="read-only source photometry dir")
    parser.add_argument("--out-phot", type=Path, help="writable work photometry dir (sidecars only)")
    parser.add_argument("--copy-tree", action="store_true", help="copy full photometry tree to out-phot first")
    parser.add_argument("--proc-dir", type=Path, help="override proc CSV directory")
    parser.add_argument("--equipment-id", type=int, default=None, help="rig equipment_id for sigma_sys")
    args = parser.parse_args()

    cfg = AppConfig()
    src = _resolve_phot(args.src_phot, args.draft, args.setup)
    if args.out_phot is None:
        if args.draft is not None:
            args.out_phot = src  # legacy in-place (avoid for anchor)
        else:
            print("FATAL: --out-phot required when not using --draft legacy mode", file=sys.stderr)
            return 1
    out = Path(args.out_phot)
    if args.copy_tree:
        _copy_photometry_tree(src, out)
    elif not out.is_dir():
        out.mkdir(parents=True, exist_ok=True)
        (out / "lightcurves").mkdir(parents=True, exist_ok=True)
    n = backfill_photometry(
        src_phot=src,
        out_phot=out,
        setup=args.setup,
        cfg=cfg,
        equipment_id=args.equipment_id,
        proc_dir=args.proc_dir,
    )
    print(f"Wrote {n} check_kmag sidecars -> {out / 'lightcurves'} (src read-only: {src})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
