#!/usr/bin/env python3
"""Audit Stage 3 Part 2b: MASTERSTAR-path DAO threshold sweep (rebuild catalogue per N)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "dev" / "scripts"))
import _bootstrap  # noqa: E402,F401

sys.path.insert(0, str(_bootstrap.REPO_ROOT / "src_py"))

from config import AppConfig  # noqa: E402
from invariants_runtime import dao_only_fraction_from_masterstars  # noqa: E402
from pipeline import (  # noqa: E402
    _dao_auto_binning_factor,
    _dao_convolved_background_rms_adu,
    _mean_bin2d_for_dao,
    build_ucac_catalog_kdtree,
    detect_stars_and_match_catalog,
    dao_detection_fwhm_pixels,
)
from photometry_core import _resolve_git_provenance  # noqa: E402


def _estimate_dao_only_g_bins(df: pd.DataFrame) -> dict[str, int]:
    """Zeropoint fit on GAIA_MATCHED; estimate G for DAO_ONLY (dao_only_verify method)."""
    if df.empty or "flux" not in df.columns:
        return {"dao_only_G_lt_16": 0, "dao_only_G_16_175": 0, "dao_only_G_gt_175": 0}
    cid = df.get("catalog_id", pd.Series([""] * len(df))).fillna("").astype(str).str.strip()
    matched = cid.ne("")
    flux = pd.to_numeric(df["flux"], errors="coerce")
    mag_col = "mag" if "mag" in df.columns else "phot_g_mean_mag"
    gmag = pd.to_numeric(df.get(mag_col), errors="coerce")
    inst = -2.5 * np.log10(flux.to_numpy(dtype=np.float64))
    fit_ok = matched.to_numpy() & np.isfinite(inst) & np.isfinite(gmag.to_numpy()) & (flux.to_numpy() > 0)
    if int(np.count_nonzero(fit_ok)) < 30:
        return {"dao_only_G_lt_16": 0, "dao_only_G_16_175": 0, "dao_only_G_gt_175": 0}
    slope, intercept = np.polyfit(inst[fit_ok], gmag.to_numpy()[fit_ok], 1)
    dao = ~matched.to_numpy() & np.isfinite(inst) & (flux.to_numpy() > 0)
    est_g = slope * inst + intercept
    eg = est_g[dao & np.isfinite(est_g)]
    return {
        "dao_only_G_lt_16": int(np.sum(eg < 16.0)),
        "dao_only_G_16_175": int(np.sum((eg >= 16.0) & (eg <= 17.5))),
        "dao_only_G_gt_175": int(np.sum(eg > 17.5)),
    }


def _sweep_row(
    *,
    data: np.ndarray,
    hdr: fits.Header,
    cat_df: pd.DataFrame,
    kd_pack: Any,
    n_sigma: float,
    ms_max_catalog_rows: int,
    prematch_k: float,
    faintest_mag: float,
    cone_path: Path,
    equipment_id: int,
    draft_id: int,
    cfg: AppConfig,
    rms_conv: float,
) -> dict[str, Any]:
    df_out, meta = detect_stars_and_match_catalog(
        data,
        hdr,
        max_catalog_rows=int(ms_max_catalog_rows),
        cat_df=cat_df,
        catalog_kd_pack=kd_pack,
        match_sep_arcsec=float(getattr(cfg, "max_catalog_match_distance_arcsec", 8.0)),
        saturate_level_fraction=float(cfg.saturate_limit_fraction),
        faintest_mag_limit=float(faintest_mag),
        field_catalog_export_path=cone_path,
        dao_threshold_sigma=float(n_sigma),
        dao_fwhm_px=None,
        equipment_saturate_adu=None,
        catalog_local_gaia_only=True,
        plate_solve_fov_deg=float(cfg.plate_solve_fov_deg),
        fov_database_path=cfg.database_path,
        fov_equipment_id=int(equipment_id),
        fov_draft_id=int(draft_id),
        prematch_peak_sigma_floor=float(prematch_k),
        dao_fwhm_bypass_header=False,
        frame_name="MASTERSTAR.fits",
    )
    n_raw = int(meta.get("n_detected_dao_raw", 0) or 0)
    n_spatial = int(meta.get("n_dao_after_spatial_cap", 0) or 0)
    n_snr = int(meta.get("n_detected_dao", 0) or 0)
    n_rows = int(len(df_out))
    dao_n = 0
    if n_rows and "catalog_id" in df_out.columns:
        cid = df_out["catalog_id"].fillna("").astype(str).str.strip()
        dao_n = int((cid == "").sum())
    dao_frac = float(dao_only_fraction_from_masterstars(df_out)) if n_rows else float("nan")
    cap_notes: list[str] = []
    prefilter_cap = max(int(ms_max_catalog_rows) * 12, 36_000)
    if n_raw >= prefilter_cap:
        cap_notes.append(f"brightest_prefilter_cap={prefilter_cap}")
    if n_spatial >= int(ms_max_catalog_rows):
        cap_notes.append(f"spatial_cap_max_catalog_rows={ms_max_catalog_rows}")
    g_bins = _estimate_dao_only_g_bins(df_out)
    return {
        "N": float(n_sigma),
        "threshold_adu": float(n_sigma) * float(rms_conv),
        "n_pass1_raw": n_raw,
        "n_after_snr_filter": n_snr,
        "n_dao_after_spatial_cap": n_spatial,
        "n_catalog_rows": n_rows,
        "dao_only_n": dao_n,
        "dao_only_frac": dao_frac,
        **g_bins,
        "cap_binding": cap_notes,
        "usable_for_calibration": len(cap_notes) == 0,
    }


def _log_log_slope(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [r for r in rows if r.get("usable_for_calibration") and r.get("n_pass1_raw", 0) > 0]
    if len(usable) < 3:
        return {"slope": None, "n_points": len(usable), "passes_sanity": False}
    x = np.log10(np.asarray([r["threshold_adu"] for r in usable], dtype=np.float64))
    y = np.log10(np.asarray([r["n_pass1_raw"] for r in usable], dtype=np.float64))
    slope, _ = np.polyfit(x, y, 1)
    passes = bool(np.isfinite(slope) and slope < -0.5)
    return {
        "slope": float(slope),
        "n_points": len(usable),
        "passes_sanity": passes,
        "criterion": "slope in (-inf, -0.5); stellar field expect ~-1 to -2",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft-id", type=int, default=499)
    parser.add_argument("--setup", default="NoFilter_60_2")
    parser.add_argument("--equipment-id", type=int, default=1)
    parser.add_argument("--n-min", type=float, default=2.5)
    parser.add_argument("--n-max", type=float, default=8.0)
    parser.add_argument("--n-step", type=float, default=0.25)
    parser.add_argument("--out", type=Path, default=REPO / "tmp" / "audit_stage3_part2b_sweep.json")
    args = parser.parse_args()

    cfg = AppConfig()
    draft = Path(cfg.archive_root) / "Drafts" / f"draft_{args.draft_id:06d}"
    ps = draft / "platesolve" / args.setup
    ms_fits = ps / "MASTERSTAR.fits"
    cone_path = ps / "field_catalog_cone.csv"
    if not ms_fits.is_file():
        raise FileNotFoundError(ms_fits)

    with fits.open(ms_fits, memmap=False) as hd:
        data = np.asarray(hd[0].data, dtype=np.float32)
        hdr = hd[0].header

    if cone_path.is_file():
        cat_df = pd.read_csv(cone_path, low_memory=False)
    else:
        cat_df = None
    kd_pack = build_ucac_catalog_kdtree(cat_df) if cat_df is not None and len(cat_df) >= 120 else None

    from astropy.stats import sigma_clipped_stats

    _, med, _ = sigma_clipped_stats(data, sigma=3.0, maxiters=3)
    data0 = np.nan_to_num((data - med).astype(np.float32))
    bfac = _dao_auto_binning_factor(*data0.shape)
    data_dao, _ = _mean_bin2d_for_dao(data0, bfac)
    fwhm_eff = max(1.2, dao_detection_fwhm_pixels(hdr, configured_fallback=float(cfg.sips_dao_fwhm_px)) / float(bfac))
    rms_conv, rel_err = _dao_convolved_background_rms_adu(data_dao, fwhm_px=fwhm_eff)

    ms_max_rows = max(int(getattr(cfg, "masterstar_max_catalog_rows", 12000) or 12000), 100_000)
    prematch_k = float(cfg.masterstar_prematch_peak_sigma_floor)
    faintest = 18.0

    gh, dirty, _ = _resolve_git_provenance()
    n_values = np.round(np.arange(args.n_min, args.n_max + 1e-9, args.n_step), 2)
    rows: list[dict[str, Any]] = []
    for n in n_values:
        print(f"Sweep N={n:.2f}...", flush=True)
        rows.append(
            _sweep_row(
                data=data,
                hdr=hdr,
                cat_df=cat_df,
                kd_pack=kd_pack,
                n_sigma=float(n),
                ms_max_catalog_rows=ms_max_rows,
                prematch_k=prematch_k,
                faintest_mag=faintest,
                cone_path=cone_path,
                equipment_id=int(args.equipment_id),
                draft_id=int(args.draft_id),
                cfg=cfg,
                rms_conv=float(rms_conv),
            )
        )

    legacy_equiv_n = (192.8 * float(rel_err)) / float(rms_conv) if rms_conv > 0 else None
    sanity = _log_log_slope(rows)

    payload: dict[str, Any] = {
        "provenance": {"git_hash": gh, "git_dirty": dirty},
        "draft_id": args.draft_id,
        "path": "detect_stars_and_match_catalog (MASTERSTAR construction path)",
        "rms_conv_adu": float(rms_conv),
        "kernel_rel_err": float(rel_err),
        "legacy_scale_threshold_equiv_N": float(legacy_equiv_n) if legacy_equiv_n else None,
        "legacy_arithmetic": "N_equiv = (192.8 ADU nominal * rel_err) / rms_conv = (192.8 * {:.3f}) / {:.2f}".format(
            rel_err, rms_conv
        ),
        "prematch_peak_sigma_floor": prematch_k,
        "max_catalog_rows": ms_max_rows,
        "sweep": rows,
        "log_log_sanity": sanity,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out} slope={sanity.get('slope')} passes={sanity.get('passes_sanity')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
