#!/usr/bin/env python3
"""Sandbox: merge same-filter exposures (240s+60s) and re-run G/R CT at min_comp=7."""
from __future__ import annotations

import json
import math
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ["VYVAR_CT_PROTOTYPE"] = "1"

DRAFT_ID = 368
FIELD_DB = _ROOT / "GAIA_DR3" / "vyvar_gaia_dr3_m67_field.db"
CONFIG_PATH = _ROOT / "config.json"
RESULT_PATH = _ROOT / "m67_gr_merge_min7_result.json"
MIN_COMP_CT = 7
MAX_STDERR_RATIO = 0.5
FIT_MIN_FRAMES = 5

MERGE_SPECS: dict[str, tuple[str, str]] = {
    "Green": ("Green_240_1", "Green_60_1"),
    "Red": ("Red_240_1", "Red_60_1"),
}

PLATESOLVE_FILES = (
    "MASTERSTAR.fits",
    "masterstars_full_match.csv",
    "variable_targets.csv",
    "per_frame_catalog_index.csv",
)


def _patch_gaia_only() -> dict:
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    orig = {"gaia_db_path": data.get("gaia_db_path")}
    data["gaia_db_path"] = str(FIELD_DB.resolve())
    CONFIG_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return orig


def _restore_gaia(orig: dict) -> None:
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if "gaia_db_path" in orig:
        data["gaia_db_path"] = orig["gaia_db_path"]
    CONFIG_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _normalize_cid(val: Any) -> str:
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


def _draft_dir(cfg) -> Path:
    return Path(cfg.archive_root) / "Drafts" / f"draft_{DRAFT_ID:06d}"


def _copy_tree_files(src_dir: Path, dst_dir: Path, pattern: str) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in sorted(src_dir.glob(pattern)):
        dst = dst_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
            n += 1
    return n


def _setup_merged_group(draft_dir: Path, merged_name: str, src_240: str, src_60: str) -> dict[str, Any]:
    aligned_root = draft_dir / "detrended_aligned" / "lights"
    ps_root = draft_dir / "platesolve"
    src_ps = ps_root / src_240
    merged_lights = aligned_root / merged_name
    merged_ps = ps_root / merged_name
    merged_phot = merged_ps / "photometry"

    if merged_lights.exists():
        shutil.rmtree(merged_lights)
    if merged_ps.exists():
        shutil.rmtree(merged_ps)
    merged_lights.mkdir(parents=True)
    merged_ps.mkdir(parents=True)

    n240 = _copy_tree_files(aligned_root / src_240, merged_lights, "*_cal.csv")
    n60 = _copy_tree_files(aligned_root / src_60, merged_lights, "*_cal.csv")
    _copy_tree_files(aligned_root / src_240, merged_lights, "*_cal.fits")
    _copy_tree_files(aligned_root / src_60, merged_lights, "*_cal.fits")

    for fname in PLATESOLVE_FILES:
        src = src_ps / fname
        if src.is_file():
            shutil.copy2(src, merged_ps / fname)

    merged_phot.mkdir(parents=True, exist_ok=True)
    return {
        "merged_name": merged_name,
        "src_240": src_240,
        "src_60": src_60,
        "n_frames_240": n240,
        "n_frames_60": n60,
        "n_frames_merged": n240 + n60,
        "masterstar_fits": merged_ps / "MASTERSTAR.fits",
        "variable_targets_csv": merged_ps / "variable_targets.csv",
        "masterstars_csv": merged_ps / "masterstars_full_match.csv",
        "per_frame_csv_dir": merged_lights,
        "detrended_aligned_dir": merged_lights,
        "output_dir": merged_phot,
        "obs_group_dir": merged_ps,
    }


def _sat_limit_for_csv(csv_path: Path) -> float:
    try:
        df = pd.read_csv(csv_path, nrows=5, low_memory=False)
    except Exception:  # noqa: BLE001
        return float("inf")
    for col in ("saturate_limit_adu", "saturate_limit_adu_85pct"):
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce").dropna()
            if not v.empty and math.isfinite(float(v.iloc[0])):
                return float(v.iloc[0])
    from photometry_core import _sat_limit_peak_adu  # noqa: E402

    lim = _sat_limit_peak_adu()
    return float(lim) if lim is not None and math.isfinite(float(lim)) else float("inf")


def _frame_usable_by_comp(csv_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for csv_path in sorted(csv_dir.glob("*_cal.csv")):
        sat_lim = _sat_limit_for_csv(csv_path)
        try:
            df = pd.read_csv(csv_path, low_memory=False, dtype={"catalog_id": str})
        except Exception:  # noqa: BLE001
            continue
        if df.empty or "catalog_id" not in df.columns:
            continue
        flux_col = "dao_flux" if "dao_flux" in df.columns else None
        for _, row in df.iterrows():
            cid = _normalize_cid(row.get("catalog_id"))
            if not cid:
                continue
            flux = pd.to_numeric(row.get(flux_col or "dao_flux"), errors="coerce") if flux_col else 1.0
            if flux_col and (not math.isfinite(float(flux)) or float(flux) <= 0):
                continue
            peak = pd.to_numeric(row.get("peak_max_adu"), errors="coerce")
            is_sat = math.isfinite(float(peak)) and math.isfinite(sat_lim) and float(peak) > sat_lim
            if is_sat:
                continue
            counts[cid] = counts.get(cid, 0) + 1
    return counts


def _patch_active_targets_per_frame_sat(
    draft_dir: Path,
    merged_name: str,
    src_240: str,
    src_60: str,
    active_csv: Path,
) -> dict[str, Any]:
    """Undo whole-star skip when 60s (or any merged frame) has unsaturated detections."""
    df = pd.read_csv(active_csv, low_memory=False, dtype={"catalog_id": str})
    a60_path = draft_dir / "platesolve" / src_60 / "photometry" / "active_targets.csv"
    if a60_path.is_file():
        a60 = pd.read_csv(a60_path, low_memory=False, dtype={"catalog_id": str})
        a60 = a60.set_index("catalog_id", drop=False)
    else:
        a60 = pd.DataFrame()

    merged_lights = draft_dir / "detrended_aligned" / "lights" / merged_name
    target_unsat_frames: dict[str, int] = {}
    for csv_path in sorted(merged_lights.glob("*_cal.csv")):
        sat_lim = _sat_limit_for_csv(csv_path)
        try:
            fr = pd.read_csv(csv_path, low_memory=False, dtype={"catalog_id": str})
        except Exception:  # noqa: BLE001
            continue
        if fr.empty or "catalog_id" not in fr.columns:
            continue
        flux_col = "dao_flux" if "dao_flux" in fr.columns else None
        for _, row in fr.iterrows():
            cid = _normalize_cid(row.get("catalog_id"))
            if not cid:
                continue
            if flux_col:
                flux = pd.to_numeric(row.get(flux_col), errors="coerce")
                if not math.isfinite(float(flux)) or float(flux) <= 0:
                    continue
            peak = pd.to_numeric(row.get("peak_max_adu"), errors="coerce")
            is_sat = math.isfinite(float(peak)) and math.isfinite(sat_lim) and float(peak) > sat_lim
            if not is_sat:
                target_unsat_frames[cid] = target_unsat_frames.get(cid, 0) + 1

    n_patched = 0
    n_from_60 = 0
    n_from_frames = 0
    for i, row in df.iterrows():
        cid = _normalize_cid(row.get("catalog_id"))
        if not cid:
            continue
        skip = str(row.get("skip_photometry", "")).strip().lower() in ("1", "true", "yes", "t")
        zf = str(row.get("zone_flag", "")).strip().lower()
        if not skip and zf != "saturated":
            continue
        rescued = False
        if cid in a60.index:
            r60 = a60.loc[cid]
            if isinstance(r60, pd.DataFrame):
                r60 = r60.iloc[0]
            if str(r60.get("skip_photometry", "")).strip().lower() not in ("1", "true", "yes", "t"):
                df.at[i, "skip_photometry"] = False
                df.at[i, "zone_flag"] = r60.get("zone_flag", "linear")
                rescued = True
                n_from_60 += 1
        if not rescued and target_unsat_frames.get(cid, 0) >= 3:
            df.at[i, "skip_photometry"] = False
            df.at[i, "zone_flag"] = "linear"
            rescued = True
            n_from_frames += 1
        if rescued:
            n_patched += 1

    df.to_csv(active_csv, index=False)
    return {
        "n_patched_targets": n_patched,
        "n_rescued_from_60s_active": n_from_60,
        "n_rescued_from_per_frame_unsat": n_from_frames,
    }


def _run_merged_photometry(
    *,
    spec: dict[str, Any],
    cfg,
    db,
    draft_dir: Path,
) -> dict[str, Any]:
    from photometry_core import run_phase0_and_phase1, run_phase2a  # noqa: E402

    merged_name = str(spec["merged_name"])
    src_240 = str(spec["src_240"])
    src_60 = str(spec["src_60"])
    output_dir = Path(spec["output_dir"])
    masterstar_fits_path = Path(spec["masterstar_fits"])

    fwhm_px = float(cfg.sips_dao_fwhm_px)
    try:
        from astropy.io import fits as astrofits  # noqa: PLC0415

        with astrofits.open(masterstar_fits_path, memmap=False) as hdul:
            hdr = hdul[0].header
            for key in ("VY_FWHM_GAUSS", "VY_FWHM_GAUSSIAN", "VY_FWHM"):
                v = hdr.get(key)
                if v is not None:
                    fv = float(v)
                    if 0.5 < fv < 30.0:
                        fwhm_px = fv
                        break
    except Exception:  # noqa: BLE001
        pass

    from photometry_core import _compute_fov_max_dist, _get_plate_scale_from_cfg, _read_plate_scale_from_fits_path  # noqa: E402

    _plate_scale = _get_plate_scale_from_cfg(
        cfg, db=db, draft_id=DRAFT_ID, fits_path=masterstar_fits_path, ms_header=None
    )
    if _plate_scale is None:
        _plate_scale = _read_plate_scale_from_fits_path(masterstar_fits_path, ms_header=None)

    p01 = run_phase0_and_phase1(
        variable_targets_csv=Path(spec["variable_targets_csv"]),
        masterstars_csv=Path(spec["masterstars_csv"]),
        per_frame_csv_dir=Path(spec["per_frame_csv_dir"]),
        output_dir=output_dir,
        fwhm_px=float(fwhm_px),
        frame_w_px=int(cfg.frame_width_px),
        frame_h_px=int(cfg.frame_height_px),
        chip_interior_margin_px=int(cfg.phase01_chip_interior_margin_px),
        match_radius_arcsec=float(cfg.phase01_match_radius_arcsec),
        plate_scale_arcsec_px=_plate_scale,
        max_dist_deg=_compute_fov_max_dist(
            frame_w_px=int(cfg.frame_width_px),
            frame_h_px=int(cfg.frame_height_px),
            plate_scale=_plate_scale,
            fov_fraction=float(cfg.phase01_comparison_fov_fraction),
            fallback_deg=float(cfg.phase01_comparison_max_dist_deg),
        ),
        max_mag_diff=float(cfg.phase01_comparison_max_mag_diff),
        comp_max_delta_bprp=float(cfg.comp_max_delta_bprp),
        max_mag_diff_t1=float(cfg.phase01_tier1_mag),
        max_mag_diff_t2=float(cfg.phase01_tier2_mag),
        max_mag_diff_t3=float(cfg.phase01_tier3_mag),
        max_mag_diff_t4=float(cfg.phase01_tier4_mag),
        n_comp_min=int(cfg.phase01_comparison_n_comp_min),
        n_comp_max=int(cfg.phase01_comparison_n_comp_max),
        max_comp_rms=float(cfg.phase01_comparison_max_comp_rms),
        min_dist_arcsec=float(cfg.phase01_comparison_min_dist_arcsec),
        min_frames_frac=float(cfg.phase01_comparison_min_frames_frac),
        rms_outlier_sigma=float(cfg.phase01_comparison_rms_outlier_sigma),
        exclude_gaia_nss=bool(cfg.phase01_comparison_exclude_gaia_nss),
        exclude_gaia_extobj=bool(cfg.phase01_comparison_exclude_gaia_extobj),
        mag_bright_threshold=float(cfg.phase01_comparison_mag_bright_threshold),
        max_mag_diff_bright_floor=float(cfg.phase01_comparison_max_mag_diff_bright_floor or 0.0),
        max_psf_chi2=float(cfg.phase01_comparison_max_psf_chi2),
        max_fwhm_factor=float(cfg.phase01_comparison_max_fwhm_factor),
        isolation_radius_px=float(cfg.phase01_comparison_isolation_radius_px),
        flux_col=cfg.phase01_flux_col,
        cfg=cfg,
        draft_id=DRAFT_ID,
        db=db,
    )

    active_targets_csv = Path(str(p01.get("active_targets_csv") or ""))
    comparison_stars_csv = Path(str(p01.get("comparison_stars_csv") or ""))
    patch_stats = _patch_active_targets_per_frame_sat(
        draft_dir, merged_name, src_240, src_60, active_targets_csv
    )

    _cfg2a = p01.get("cfg_effective_for_photometry") or cfg
    p2a = run_phase2a(
        masterstar_fits_path=masterstar_fits_path,
        active_targets_csv=active_targets_csv,
        comparison_stars_csv=comparison_stars_csv,
        per_frame_csv_dir=Path(spec["per_frame_csv_dir"]),
        detrended_aligned_dir=Path(spec["detrended_aligned_dir"]),
        output_dir=output_dir,
        fwhm_px=float(fwhm_px),
        cfg=_cfg2a,
        db=db,
        draft_id=DRAFT_ID,
        proc_frame_store=p01.get("proc_store"),
    )
    return {
        "phase01": p01,
        "phase2a": p2a,
        "target_patch": patch_stats,
        "n_lightcurves": int((p2a or {}).get("n_lightcurves") or 0),
        "n_targets": int((p2a or {}).get("n_targets") or 0),
    }


def _saturation_audit(draft_dir: Path, merged_name: str, src_240: str, src_60: str) -> dict[str, Any]:
    aligned_root = draft_dir / "detrended_aligned" / "lights"
    master = draft_dir / "platesolve" / src_240 / "masterstars_full_match.csv"
    ms = pd.read_csv(master, low_memory=False, dtype={"catalog_id": str})
    ms["catalog_id"] = ms["catalog_id"].map(_normalize_cid)
    whole_sat: set[str] = set()
    if "is_saturated" in ms.columns:
        whole_sat = {
            _normalize_cid(v)
            for v in ms.loc[ms["is_saturated"].astype(str).str.lower().isin(("true", "1", "yes")), "catalog_id"]
            if _normalize_cid(v)
        }

    u240 = _frame_usable_by_comp(aligned_root / src_240)
    u60 = _frame_usable_by_comp(aligned_root / src_60)
    umerged = _frame_usable_by_comp(aligned_root / merged_name)
    comp_ids = set(u240) | set(u60) | set(umerged)

    rescued_merge = [
        cid
        for cid in comp_ids
        if u240.get(cid, 0) < FIT_MIN_FRAMES and umerged.get(cid, 0) >= FIT_MIN_FRAMES
    ]
    rescued_sat_per_frame = [
        cid
        for cid in comp_ids
        if cid in whole_sat and umerged.get(cid, 0) >= FIT_MIN_FRAMES
    ]
    lost_whole_star_if_global = [
        cid
        for cid in whole_sat
        if u240.get(cid, 0) < FIT_MIN_FRAMES and u60.get(cid, 0) >= FIT_MIN_FRAMES
    ]

    return {
        "n_comps_any_frame": len(comp_ids),
        "n_whole_star_saturated_masterstar": len(whole_sat),
        "n_rescued_by_merge_lt5_to_ge5": len(rescued_merge),
        "n_rescued_whole_star_flag_but_ge5_per_frame": len(rescued_sat_per_frame),
        "n_would_lose_if_whole_star_drop": len(lost_whole_star_if_global),
        "sample_rescued_merge": rescued_merge[:8],
        "sample_rescued_per_frame_sat": rescued_sat_per_frame[:8],
    }


def _comp_pool_stats(comp_pt: Path, target_cids: list[str]) -> dict[str, Any]:
    if not comp_pt.is_file():
        return {"comp_bp_rp_min": float("nan"), "comp_bp_rp_max": float("nan"), "comp_bp_rp_width": float("nan")}
    df = pd.read_csv(comp_pt, low_memory=False, dtype={"catalog_id": str, "target_catalog_id": str})
    sub = df[df["target_catalog_id"].astype(str).isin(target_cids)]
    bps = pd.to_numeric(sub.get("bp_rp"), errors="coerce").dropna()
    if bps.empty:
        return {"comp_bp_rp_min": float("nan"), "comp_bp_rp_max": float("nan"), "comp_bp_rp_width": float("nan")}
    lo, hi = float(bps.min()), float(bps.max())
    return {"comp_bp_rp_min": lo, "comp_bp_rp_max": hi, "comp_bp_rp_width": hi - lo, "n_comp_rows": int(len(bps))}


def _load_presel_in_range(diag_csv: Path) -> dict[str, set[str]]:
    df = pd.read_csv(diag_csv, low_memory=False, dtype={"catalog_id": str})
    out: dict[str, set[str]] = {"Green": set(), "Red": set()}
    for flt in out:
        sub = df[(df["filter"].astype(str) == flt) & (df["presel_in_range"].astype(str).str.lower().isin(("true", "1", "yes")))]
        out[flt] = {_normalize_cid(v) for v in sub["catalog_id"] if _normalize_cid(v)}
    return out


def _split_min7_by_filter(min7_csv: Path) -> dict[str, pd.DataFrame]:
    df = pd.read_csv(min7_csv, low_memory=False, dtype={"catalog_id": str})
    if "filter" not in df.columns and "obs_group" in df.columns:
        df = df.copy()
        df["filter"] = df["obs_group"].astype(str).str.split("_").str[0]
    out: dict[str, pd.DataFrame] = {}
    for flt in ("Green", "Red"):
        sub = df[df["filter"].astype(str) == flt].copy()
        by_cid: dict[str, dict] = {}
        for _, row in sub.iterrows():
            cid = _normalize_cid(row["catalog_id"])
            if not cid:
                continue
            n = pd.to_numeric(row.get("n_comp_used"), errors="coerce")
            n_i = int(n) if math.isfinite(float(n)) else -999
            prev = by_cid.get(cid)
            if prev is None or n_i > int(prev.get("n_comp_used", -999)):
                by_cid[cid] = row.to_dict()
        out[flt] = pd.DataFrame(by_cid.values()) if by_cid else pd.DataFrame()
    return out


def _analyze_filter(
    *,
    flt: str,
    merged_name: str,
    draft_dir: Path,
    presel_ids: set[str],
    split_min7: pd.DataFrame,
    min5_ids: set[str],
) -> dict[str, Any]:
    from photometry_core import should_apply_color_term  # noqa: E402

    proto_path = draft_dir / "ct_prototype.csv"
    proto = pd.read_csv(proto_path, low_memory=False, dtype={"catalog_id": str})
    merged = proto[proto["obs_group"].astype(str) == merged_name].copy()
    lc_dir = draft_dir / "platesolve" / merged_name / "photometry" / "lightcurves"

    ct_ok_ids: set[str] = set()
    lc_rows: dict[str, dict] = {}
    if lc_dir.is_dir():
        for lc in lc_dir.glob("lightcurve_*.csv"):
            cid = lc.stem.replace("lightcurve_", "", 1)
            r0 = pd.read_csv(lc, nrows=1, low_memory=False).iloc[0]
            lc_rows[cid] = r0.to_dict()
            if str(r0.get("ct_ok", "")).strip().lower() in ("true", "1", "yes"):
                ct_ok_ids.add(cid)

    in_presel = merged[merged["catalog_id"].astype(str).isin(presel_ids)].copy()
    n_comp = pd.to_numeric(in_presel.get("n_comp_used"), errors="coerce")
    in_presel["_n_comp"] = n_comp

    comp_pt = draft_dir / "platesolve" / merged_name / "photometry" / "comparison_stars_per_target.csv"
    pool = _comp_pool_stats(comp_pt, sorted(presel_ids))

    newly_pass = sorted(ct_ok_ids & presel_ids)
    detail: list[dict] = []
    for cid in newly_pass:
        pr = merged[merged["catalog_id"].astype(str) == cid]
        row = pr.iloc[0] if not pr.empty else {}
        split_row = split_min7[split_min7["catalog_id"].astype(str) == cid]
        split_scatter = float("nan")
        split_n = float("nan")
        if not split_row.empty:
            split_scatter = pd.to_numeric(split_row.iloc[0].get("cat_inst_scatter"), errors="coerce")
            split_n = pd.to_numeric(split_row.iloc[0].get("n_comp_used"), errors="coerce")
        post_scatter = pd.to_numeric(row.get("cat_inst_scatter_resid"), errors="coerce")
        pre_scatter = pd.to_numeric(row.get("cat_inst_scatter"), errors="coerce")
        if not math.isfinite(float(pre_scatter)) and math.isfinite(float(split_scatter)):
            pre_scatter = float(split_scatter)
        pct = float("nan")
        if math.isfinite(float(pre_scatter)) and float(pre_scatter) > 0 and math.isfinite(float(post_scatter)):
            pct = 100.0 * (float(pre_scatter) - float(post_scatter)) / float(pre_scatter)
        detail.append(
            {
                "catalog_id": cid,
                "n_comp_merged": int(pd.to_numeric(row.get("n_comp_used"), errors="coerce")),
                "n_comp_split_best": int(split_n) if math.isfinite(float(split_n)) else None,
                "stderr_ratio": float(pd.to_numeric(row.get("stderr_ratio"), errors="coerce")),
                "ct_corr": float(pd.to_numeric(row.get("ct_corr"), errors="coerce")),
                "cat_inst_scatter_pre": float(pre_scatter),
                "cat_inst_scatter_post": float(post_scatter),
                "scatter_pct_improved": float(pct),
            }
        )

    n_ge7 = int((in_presel["_n_comp"] >= MIN_COMP_CT).sum())
    dist = in_presel["_n_comp"].value_counts(dropna=False).sort_index().to_dict()
    dist = {str(k): int(v) for k, v in dist.items()}

    return {
        "merged_obs_group": merged_name,
        "merged_frame_count": int(len(list((draft_dir / "detrended_aligned" / "lights" / merged_name).glob("*_cal.csv")))),
        "n_usable_comps_ge5_frames": int(
            sum(1 for v in _frame_usable_by_comp(draft_dir / "detrended_aligned" / "lights" / merged_name).values() if v >= FIT_MIN_FRAMES)
        ),
        "comp_bp_rp_baseline": pool,
        "in_range_presel_n": len(presel_ids),
        "n_comp_ge7_in_range": n_ge7,
        "n_comp_distribution_in_range": dist,
        "ct_ok_merged_min7": len(ct_ok_ids & presel_ids),
        "ct_ok_catalog_ids": sorted(ct_ok_ids & presel_ids),
        "reference_ct_ok_min7_split": 0,
        "reference_ct_ok_min5_split": len(min5_ids & presel_ids),
        "newly_passing_detail": detail,
        "proto_gate_pass_in_range": int(in_presel["gate_would_pass"].sum()) if "gate_would_pass" in in_presel.columns else None,
    }


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from config import AppConfig  # noqa: E402
    from database import VyvarDatabase  # noqa: E402

    cfg = AppConfig()
    draft_dir = _draft_dir(cfg)
    diag_csv = _ROOT / "m67_ct_gate_diagnosis_draft000368.csv"
    min7_backup = draft_dir / "ct_prototype_min7_backup.csv"
    min5_result = _ROOT / "m67_gr_sandbox_min5_result.json"

    presel = _load_presel_in_range(diag_csv) if diag_csv.is_file() else {"Green": set(), "Red": set()}
    split_min7 = _split_min7_by_filter(min7_backup) if min7_backup.is_file() else {"Green": pd.DataFrame(), "Red": pd.DataFrame()}
    min5_ids = {"Green": set(), "Red": set()}
    if min5_result.is_file():
        m5 = json.loads(min5_result.read_text(encoding="utf-8"))
        for flt in ("Green", "Red"):
            min5_ids[flt] = set(m5.get("summary", {}).get("by_filter", {}).get(flt, {}).get("ct_ok_catalog_ids", []))

    proto = draft_dir / "ct_prototype.csv"
    proto_min5_backup = draft_dir / "ct_prototype_min5_sandbox_backup.csv"
    if proto.is_file() and not proto_min5_backup.is_file():
        shutil.copy2(proto, proto_min5_backup)

    setups: dict[str, dict] = {}
    analyze_only = os.environ.get("VYVAR_MERGE_ANALYZE_ONLY", "").strip() == "1"
    for merged_name, (s240, s60) in MERGE_SPECS.items():
        if analyze_only:
            aligned = draft_dir / "detrended_aligned" / "lights" / merged_name
            ps = draft_dir / "platesolve" / merged_name
            setups[merged_name] = {
                "merged_name": merged_name,
                "src_240": s240,
                "src_60": s60,
                "masterstar_fits": ps / "MASTERSTAR.fits",
                "variable_targets_csv": ps / "variable_targets.csv",
                "masterstars_csv": ps / "masterstars_full_match.csv",
                "per_frame_csv_dir": aligned,
                "detrended_aligned_dir": aligned,
                "output_dir": ps / "photometry",
                "obs_group_dir": ps,
            }
        else:
            setups[merged_name] = _setup_merged_group(draft_dir, merged_name, s240, s60)

    orig_gaia = _patch_gaia_only()
    report: dict[str, Any] = {
        "draft_id": DRAFT_ID,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "min_comp_for_ct": MIN_COMP_CT,
        "max_stderr_ratio": MAX_STDERR_RATIO,
        "merge_specs": MERGE_SPECS,
        "setups": {
            k: {kk: str(vv) if isinstance(vv, Path) else vv for kk, vv in v.items() if kk != "obs_group_dir"}
            for k, v in setups.items()
        },
    }
    try:
        if proto.is_file() and not analyze_only:
            proto.unlink()
        db = VyvarDatabase(cfg.database_path)
        phot: dict[str, Any] = {}
        if not analyze_only:
            for merged_name, spec in setups.items():
                phot[merged_name] = _run_merged_photometry(
                    spec=spec,
                    cfg=cfg,
                    db=db,
                    draft_dir=draft_dir,
                )
                phot[merged_name] = {
                    "n_lightcurves": int(phot[merged_name].get("n_lightcurves") or 0),
                    "n_targets": int(phot[merged_name].get("n_targets") or 0),
                    "target_patch": phot[merged_name].get("target_patch"),
                }
        else:
            for merged_name, spec in setups.items():
                lc_dir = Path(spec["output_dir"]) / "lightcurves"
                phot[merged_name] = {"n_lightcurves": len(list(lc_dir.glob("lightcurve_*.csv")))}
        report["photometry"] = phot

        sat_audit: dict[str, Any] = {}
        analysis: dict[str, Any] = {}
        for merged_name, (s240, s60) in MERGE_SPECS.items():
            sat_audit[merged_name] = _saturation_audit(draft_dir, merged_name, s240, s60)
            analysis[merged_name] = _analyze_filter(
                flt=merged_name,
                merged_name=merged_name,
                draft_dir=draft_dir,
                presel_ids=presel[merged_name],
                split_min7=split_min7[merged_name],
                min5_ids=min5_ids[merged_name],
            )
            analysis[merged_name]["saturation_audit"] = sat_audit[merged_name]
        report["saturation_audit"] = sat_audit
        report["analysis"] = analysis
    finally:
        _restore_gaia(orig_gaia)

    report["finished_utc"] = datetime.now(timezone.utc).isoformat()

    def _json_default(obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        raise TypeError(type(obj).__name__)

    RESULT_PATH.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
