#!/usr/bin/env python3
"""DAO-CLOSE: confusion-blend test for unmatched_in_range DAO_ONLY rows."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src_py"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import AppConfig  # noqa: E402
from dao_reconcile import (  # noqa: E402
    annotate_dao_only_magnitude_classes,
    implied_g_from_flux,
    resolve_effective_match_depth,
)
from dilution import flux_from_gmag, query_gaia_neighbors  # noqa: E402
from gaia_catalog_id import read_vyvar_csv  # noqa: E402

DRAFTS = (
    ("draft_000501", "V_60_2"),
    ("draft_000435_snapshot_skysurface_20260716", "NoFilter_60_2"),
    ("draft_000500", "NoFilter_60_2"),
)

FWHM_MULTS = (1.0, 2.0, 3.0)
BRIGHTNESS_MATCH_MAG = 0.75


def _masterstar_fwhm_arcsec(ms_fits: Path) -> float:
    hdr = fits.getheader(str(ms_fits))
    fwhm_px = float(hdr.get("VY_FWHM") or 0.0)
    if fwhm_px <= 0:
        return float("nan")
    try:
        w = WCS(hdr)
        arcsec_per_px = abs(float(w.pixel_scale_matrix[1, 1]) * 3600.0)
    except Exception:  # noqa: BLE001
        arcsec_per_px = float("nan")
    if not math.isfinite(arcsec_per_px) or arcsec_per_px <= 0:
        return float("nan")
    return fwhm_px * arcsec_per_px


def _neighbor_stats(
    ra: float,
    dec: float,
    *,
    gaia_db: str,
    fwhm_arcsec: float,
    zp: float,
) -> dict[str, float | int]:
    out: dict[str, float | int] = {}
    for mult in FWHM_MULTS:
        r = float(fwhm_arcsec * mult)
        neighbors = query_gaia_neighbors(ra, dec, r, gaia_db)
        n = len(neighbors)
        flux_sum = float(sum(flux_from_gmag(float(nbr["g_mag"])) for nbr in neighbors))
        ig_sum = (
            implied_g_from_flux(flux_sum, zp=zp)
            if flux_sum > 0 and math.isfinite(zp)
            else float("nan")
        )
        key = f"{mult:g}x_fwhm"
        out[f"n_gaia_{key}"] = n
        out[f"implied_g_sum_{key}"] = ig_sum
    return out


def measure_draft(csv_path: Path, ms_fits: Path, gaia_db: str) -> dict:
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
        det_meta = {
            "faintest_mag_limit": pm.get("faintest_mag_limit"),
            "provenance": pm.get("provenance"),
        }
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
    ff = meta.get("flux_fit") or {}
    zp = float(ff.get("zp")) if ff.get("zp") is not None else float("nan")
    zp_rms = float(ff.get("residual_rms")) if ff.get("residual_rms") is not None else float("nan")
    fwhm_arcsec = _masterstar_fwhm_arcsec(ms_fits)

    in_range = out[out["dao_only_class"] == "unmatched_in_range"].copy()
    matched = out[out["source_type"].astype(str).str.upper().ne("DAO_ONLY")].copy()
    n_in = int(len(in_range))
    sample_too_small = n_in < 30

    row_stats: list[dict] = []
    for idx in in_range.index:
        ra = float(out.at[idx, "ra_deg"])
        dec = float(out.at[idx, "dec_deg"])
        ig_det = float(out.at[idx, "implied_g_mag"]) if pd.notna(out.at[idx, "implied_g_mag"]) else float("nan")
        st = _neighbor_stats(ra, dec, gaia_db=gaia_db, fwhm_arcsec=fwhm_arcsec, zp=zp)
        st["implied_g_detection"] = ig_det
        for mult in FWHM_MULTS:
            key = f"{mult:g}x_fwhm"
            ig_sum = st.get(f"implied_g_sum_{key}")
            if ig_sum is not None and math.isfinite(float(ig_sum)) and math.isfinite(ig_det):
                st[f"delta_g_sum_minus_det_{key}"] = float(ig_sum) - ig_det
        row_stats.append(st)

    # Control: Gaia-matched rows with similar catalogue G
    ctrl_stats: list[dict] = []
    med_g = float(pd.to_numeric(in_range.get("implied_g_mag"), errors="coerce").median()) if n_in else float("nan")
    if n_in and matched.shape[0]:
        gcol = pd.to_numeric(matched.get("phot_g_mean_mag", matched.get("mag")), errors="coerce")
        band = gcol.notna() & (np.abs(gcol - med_g) <= BRIGHTNESS_MATCH_MAG)
        ctrl_pool = matched.loc[band] if band.any() else matched
        n_ctrl = min(n_in, int(ctrl_pool.shape[0]))
        ctrl = ctrl_pool.sample(n=n_ctrl, random_state=42) if n_ctrl > 0 else ctrl_pool.iloc[0:0]
        for idx in ctrl.index:
            ra = float(ctrl.at[idx, "ra_deg"])
            dec = float(ctrl.at[idx, "dec_deg"])
            ctrl_stats.append(
                _neighbor_stats(ra, dec, gaia_db=gaia_db, fwhm_arcsec=fwhm_arcsec, zp=zp)
            )

    def _summarize(rows: list[dict], prefix: str) -> dict:
        if not rows:
            return {}
        agg: dict[str, float] = {}
        for mult in FWHM_MULTS:
            key = f"{mult:g}x_fwhm"
            nk = f"n_gaia_{key}"
            vals = [float(r[nk]) for r in rows if nk in r]
            if vals:
                agg[f"{prefix}_median_n_gaia_{key}"] = float(np.median(vals))
            dk = f"delta_g_sum_minus_det_{key}"
            dvals = [float(r[dk]) for r in rows if dk in r and math.isfinite(float(r[dk]))]
            if dvals:
                agg[f"{prefix}_median_delta_g_{key}"] = float(np.median(dvals))
        return agg

    test_sum = _summarize(row_stats, "test")
    ctrl_sum = _summarize(ctrl_stats, "control")

    verdict = "inconclusive"
    reason = ""
    if sample_too_small:
        verdict = "inconclusive"
        reason = f"n_unmatched_in_range={n_in} too small for population inference"
    elif not row_stats or not math.isfinite(fwhm_arcsec):
        verdict = "inconclusive"
        reason = "missing FWHM or empty test sample"
    else:
        dens_higher = False
        flux_consistent = False
        for mult in FWHM_MULTS:
            key = f"{mult:g}x_fwhm"
            t_n = test_sum.get(f"test_median_n_gaia_{key}")
            c_n = ctrl_sum.get(f"control_median_n_gaia_{key}")
            if t_n is not None and c_n is not None and t_n > c_n + 0.5:
                dens_higher = True
            t_d = test_sum.get(f"test_median_delta_g_{key}")
            if t_d is not None and abs(t_d) <= max(1.0, 2.0 * zp_rms if math.isfinite(zp_rms) else 1.0):
                flux_consistent = True
        if dens_higher and flux_consistent:
            verdict = "confirmed_blend_hypothesis"
            reason = "higher local Gaia density and summed-neighbour G consistent with detection flux"
        elif dens_higher and not flux_consistent:
            verdict = "inconclusive"
            reason = "higher local density but summed flux does not match detection implied G within scatter"
        else:
            verdict = "refuted_or_inconclusive"
            reason = "no clear excess local Gaia density vs matched controls at 1-3 x FWHM"

    return {
        "n_unmatched_in_range": n_in,
        "sample_too_small": sample_too_small,
        "fwhm_arcsec": fwhm_arcsec,
        "flux_fit_zp": zp,
        "flux_fit_residual_rms": zp_rms,
        "confirmable_depth_g": meta.get("confirmable_depth_g"),
        "sigma_g_unmeasurable_fraction": meta.get("sigma_g_unmeasurable_fraction"),
        "test_summary": test_sum,
        "control_summary": ctrl_sum,
        "verdict": verdict,
        "verdict_reason": reason,
        "future_evidence": (
            "deeper all-sky catalogue, finer plate scale, or per-frame persistence with adequate drift baseline"
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "tmp" / "dao_close_confusion_blend.json")
    args = ap.parse_args()
    cfg = AppConfig()
    gaia_db = str(cfg.gaia_db_path or "")
    archive = Path(cfg.archive_root) / "Drafts"
    results: dict[str, dict] = {}
    for draft, setup in DRAFTS:
        key = f"{draft}/{setup}"
        base = archive / draft / "platesolve" / setup
        csv_path = base / "masterstars_full_match.csv"
        ms_fits = base / "MASTERSTAR.fits"
        if not csv_path.is_file():
            results[key] = {"error": f"missing {csv_path}"}
            continue
        results[key] = measure_draft(csv_path, ms_fits, gaia_db)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
