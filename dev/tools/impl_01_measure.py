#!/usr/bin/env python
"""IMPL-01 acceptance measurements on draft 514 (ASCII)."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src_py"))

from photometry_core import (  # noqa: E402
    compute_snr_optimal_aperture_table,
    measure_growth_curve_ee,
    precompute_and_save_snr_aperture_table_for_draft,
)
from config import AppConfig  # noqa: E402

DRAFT = ROOT / "Archive" / "Drafts" / "draft_000514"
PROC = DRAFT / "detrended_aligned" / "lights" / "NoFilter_60_2"
OUT = ROOT / "dev" / "results"
OLD_R = 2.711
EE_OLD = 0.663
# PRE-IMPL-01 production table (gaussian + clamp) before this rebuild overwrote the file.
OLD_TBL_PREIMPL = {
    7.0: 4.861,
    8.0: 4.561,
    9.0: 4.261,
    10.0: 4.061,
    11.0: 3.661,
    12.0: 3.261,
    13.0: 2.911,
    14.0: 2.711,
    15.0: 2.711,
    16.0: 2.711,
    17.0: 2.711,
    18.0: 2.711,
}


def _sha() -> str:
    import subprocess

    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT)
            .decode()
            .strip()
        )
    except Exception:  # noqa: BLE001
        return "unknown"


def _annulus_flux(data, x, y, r, r_in=10.0, r_out=15.0):
    from photutils.aperture import CircularAnnulus, CircularAperture, aperture_photometry

    ap = CircularAperture([(float(x), float(y))], r=float(r))
    ann = CircularAnnulus([(float(x), float(y))], r_in=float(r_in), r_out=float(r_out))
    phot = aperture_photometry(data, ap, method="exact")
    bkg = aperture_photometry(data, ann, method="exact")
    area = math.pi * float(r) ** 2
    bkg_area = math.pi * (float(r_out) ** 2 - float(r_in) ** 2)
    sky = float(bkg["aperture_sum"][0]) / bkg_area if bkg_area > 0 else 0.0
    return float(phot["aperture_sum"][0]) - sky * area


def main() -> int:
    cfg = AppConfig(project_root=ROOT)

    ms = None
    for cand in (
        DRAFT / "platesolve" / "NoFilter_60_2" / "MASTERSTAR.fits",
        DRAFT / "platesolve" / "MASTERSTAR.fits",
    ):
        if cand.is_file():
            ms = cand
            break
    aligned = sorted(PROC.glob("proc_*.fits"))[:12] if PROC.is_dir() else []
    if not aligned:
        aligned = sorted((DRAFT / "calibrated" / "lights" / "NoFilter_60_2").glob("*.fits"))[:12]

    # Snapshot old table if present (may already be post-IMPL); prefer PRE-IMPL constants.
    old_tbl = dict(OLD_TBL_PREIMPL)
    old_path = DRAFT / "aperture_snr_table.json"
    if old_path.is_file():
        try:
            old_doc = json.loads(old_path.read_text(encoding="utf-8"))
            if str(old_doc.get("ee_path") or "") == "gaussian_fallback" or "ee_path" not in old_doc:
                old_tbl = {float(k): float(v) for k, v in (old_doc.get("table") or {}).items()}
        except Exception:  # noqa: BLE001
            pass

    snr = precompute_and_save_snr_aperture_table_for_draft(
        DRAFT,
        masterstar_fits_path=ms,
        aligned_fits_paths=aligned,
        cfg=cfg,
        sky_fallback=1581.6,
    )
    if snr is None:
        print("FAIL: snr table None")
        return 1

    new_tbl = {float(k): float(v) for k, v in (snr.get("table") or {}).items()}
    ee_at = {float(k): float(v) for k, v in (snr.get("ee_at_opt_by_mag") or {}).items()}
    bounds = {float(k): str(v) for k, v in (snr.get("bound_hit_by_mag") or {}).items()}

    # Mag bins of interest
    bins = [8.0, 10.0, 12.0, 14.0, 16.0]
    per_bin = []
    for m in bins:
        # nearest key
        if new_tbl:
            mk = min(new_tbl.keys(), key=lambda x: abs(x - m))
        else:
            mk = m
        per_bin.append(
            {
                "mag": mk,
                "r_old_px": old_tbl.get(mk, OLD_R),
                "r_new_px": new_tbl.get(mk),
                "ee_at_new": ee_at.get(mk),
                "ee_old_at_2p711": EE_OLD,
                "bound_hit": bounds.get(mk, "unknown"),
            }
        )

    # Check-star scatter before/after on same frames (instrumental differential)
    # Use BO CVn check from active_targets / summary if available.
    check_scatter = {"before_mmag": None, "after_mmag": None, "method": None, "n_frames": 0}
    summary = DRAFT / "platesolve" / "NoFilter_60_2" / "photometry" / "summary.csv"
    if not summary.is_file():
        summary = next(DRAFT.rglob("summary.csv"), None)
    check_cid = None
    if summary is not None and Path(summary).is_file():
        sdf = pd.read_csv(summary)
        # Prefer named check / lowest scatter non-target
        if "check_scatter" in sdf.columns or "scatter" in sdf.columns:
            pass
        if "catalog_id" in sdf.columns and "vsx_name" in sdf.columns:
            non_var = sdf[~sdf["vsx_name"].astype(str).str.contains("BO CVn", case=False, na=False)]
            if not non_var.empty and "check_scatter" in non_var.columns:
                row = non_var.nsmallest(1, "check_scatter").iloc[0]
                check_cid = str(row["catalog_id"])
                check_scatter["before_mmag"] = float(row["check_scatter"]) * 1000.0
                check_scatter["note"] = "before from existing summary check_scatter (pre-IMPL radii)"

    # Direct A/B: LOO differential mag MAD at old vs new radius for bright isolated comps
    proc_csvs = sorted(PROC.glob("proc_*.csv"))[:40] if PROC.is_dir() else []
    fits_map = {}
    if PROC.is_dir():
        for fp in PROC.glob("*.fits"):
            fits_map[fp.stem.replace("proc_", "")] = fp
            fits_map[fp.name] = fp

    r_new_med = float(np.nanmedian(list(new_tbl.values()))) if new_tbl else float("nan")
    if proc_csvs and math.isfinite(r_new_med):
        # Pick up to 8 bright stars from frame 001
        df0 = pd.read_csv(proc_csvs[0])
        mag = pd.to_numeric(df0.get("mag"), errors="coerce")
        flux = pd.to_numeric(df0.get("dao_flux", df0.get("flux")), errors="coerce")
        x = pd.to_numeric(df0["x"], errors="coerce")
        y = pd.to_numeric(df0["y"], errors="coerce")
        cid = df0["catalog_id"].astype(str)
        order = np.argsort(-flux.to_numpy(dtype=float))
        stars = []
        for i in order:
            if len(stars) >= 8:
                break
            if not (math.isfinite(float(x.iloc[i])) and math.isfinite(float(y.iloc[i]))):
                continue
            if math.isfinite(float(mag.iloc[i])) and not (8.0 <= float(mag.iloc[i]) <= 12.0):
                continue
            stars.append(str(cid.iloc[i]))
        # Build per-frame mags at old/new r
        series_old: dict[str, list[float]] = {s: [] for s in stars}
        series_new: dict[str, list[float]] = {s: [] for s in stars}
        n_ok = 0
        for csvp in proc_csvs:
            dfc = pd.read_csv(csvp)
            # locate fits
            stem = csvp.stem.replace("proc_", "")
            fp = None
            for key in (stem, csvp.stem, csvp.name.replace(".csv", ".fits")):
                if key in fits_map:
                    fp = fits_map[key]
                    break
            if fp is None:
                # try matching by frame number
                for k, v in fits_map.items():
                    if stem in str(k) or stem in str(v):
                        fp = v
                        break
            if fp is None or not Path(fp).is_file():
                continue
            with fits.open(fp, memmap=True) as hd:
                data = np.asarray(hd[0].data, dtype=float)
            xy = {
                str(r["catalog_id"]): (
                    float(pd.to_numeric(r["x"], errors="coerce")),
                    float(pd.to_numeric(r["y"], errors="coerce")),
                )
                for _, r in dfc.iterrows()
            }
            for s in stars:
                if s not in xy:
                    series_old[s].append(float("nan"))
                    series_new[s].append(float("nan"))
                    continue
                xx, yy = xy[s]
                fo = _annulus_flux(data, xx, yy, OLD_R)
                fn = _annulus_flux(data, xx, yy, r_new_med)
                series_old[s].append(-2.5 * math.log10(fo) if fo > 0 else float("nan"))
                series_new[s].append(-2.5 * math.log10(fn) if fn > 0 else float("nan"))
            n_ok += 1

        def _loo_mad_mmag(series: dict[str, list[float]]) -> float:
            ids = list(series.keys())
            if len(ids) < 3:
                return float("nan")
            mads = []
            for t in ids:
                peers = [p for p in ids if p != t]
                arr_t = np.asarray(series[t], float)
                stack = np.vstack([np.asarray(series[p], float) for p in peers])
                ens = np.nanmedian(stack, axis=0)
                diff = arr_t - ens
                diff = diff - np.nanmedian(diff)
                fin = diff[np.isfinite(diff)]
                if fin.size >= 10:
                    mads.append(1.4826 * float(np.median(np.abs(fin))))
            return float(np.nanmedian(mads)) * 1000.0 if mads else float("nan")

        check_scatter["after_direct_mmag"] = _loo_mad_mmag(series_new)
        check_scatter["before_direct_mmag"] = _loo_mad_mmag(series_old)
        check_scatter["method"] = (
            "LOO median-ensemble MAD of instrumental mags at fixed r "
            f"(old={OLD_R:.3f}px vs new_med={r_new_med:.3f}px) on same frames; "
            "not full Phase-2A re-export"
        )
        check_scatter["n_frames"] = n_ok
        check_scatter["n_stars"] = len(stars)
        check_scatter["r_new_median_px"] = r_new_med

    artifact = {
        "commit_sha": _sha(),
        "ee_path": snr.get("ee_path"),
        "ee_n_cog": snr.get("ee_n_cog"),
        "ee_ref_r_px": snr.get("ee_ref_r_px"),
        "ee_flatness_tail_over_norm": snr.get("ee_flatness_tail_over_norm"),
        "ee_isolation_fwhm": snr.get("ee_isolation_fwhm"),
        "ee_min_stars_required": snr.get("ee_min_stars_required"),
        "ee_fallback_reason": snr.get("ee_fallback_reason"),
        "n_bound_hits": snr.get("n_bound_hits"),
        "fwhm_px": snr.get("fwhm_px"),
        "per_mag_bin": per_bin,
        "check_star_scatter": check_scatter,
        "q4_night_ee_quantity": (
            "common_mode_absolute_EE_single_star_frame_to_frame_demeaned_MAD; "
            "NOT residual after ensemble common-mode cancellation"
        ),
        "q4_night_ee_mad_mmag": 0.008,
        "q4_wide_err_note": (
            "Named as common-mode absolute EE variation. Tiny 0.008 mmag on draft 514 "
            "does not establish differential residual; aperture-loss-vs-seeing as WIDE-ERR "
            "driver is not supported by this quantity as measured."
        ),
        "color_level_literature": {
            "k_level_mag_per_bprp": -0.373,
            "k_level_stderr": 0.090,
            "plausible": True,
            "cite": (
                "AAVSO CCD Photometry Guide: untransformed Clear/standard residuals "
                "'as high as several tenths of a magnitude'; |k|=0.373 mag/BP-RP sits "
                "in that range for Delta(BP-RP)~1. Gaia G-V linear terms are smaller "
                "(~0.01-0.05); Clear red CMOS to Gaia G is expected larger. Per-rig only."
            ),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "IMPL_01_aperture_cog.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps({k: artifact[k] for k in ("commit_sha", "ee_path", "n_bound_hits", "per_mag_bin", "check_star_scatter")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
