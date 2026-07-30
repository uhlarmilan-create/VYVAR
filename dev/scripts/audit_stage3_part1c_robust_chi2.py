#!/usr/bin/env python3
"""Audit Stage 3 Part 1c: robust check-star chi2, outlier forensics, sigma_sys audit."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "dev" / "scripts"))
import _bootstrap  # noqa: E402,F401

for p in (_bootstrap.REPO_ROOT / "src_py", _bootstrap.REPO_ROOT / "dev"):
    sys.path.insert(0, str(p))

from config import AppConfig  # noqa: E402
from photometry_core import (  # noqa: E402
    _phase2a_prepare_shared_state,
    photometer_check_star_production_path,
    _resolve_git_provenance,
)
from sigma_budget import resolve_rig_scintillation_params, scintillation_sigma  # noqa: E402
from sigma_floor_core import resolve_sigma_sys_mag  # noqa: E402

sys.path.insert(0, str(REPO / "dev" / "scripts"))
from chi2_sigma_gate import reduced_chi2_constant  # noqa: E402

SETUP = "NoFilter_60_2"
CLIP_SIGMA = 3.0
CLIP_MAXITERS = 5


def _dist_stats(vals: list[float]) -> dict[str, float | None]:
    if not vals:
        return {"median": None, "p05": None, "p95": None, "min": None, "max": None, "n": 0}
    a = np.asarray(vals, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"median": None, "p05": None, "p95": None, "min": None, "max": None, "n": 0}
    return {
        "median": float(np.median(a)),
        "p05": float(np.percentile(a, 5)),
        "p95": float(np.percentile(a, 95)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "n": int(a.size),
    }


def _iterative_sigma_clip_chi2(
    mags: np.ndarray,
    errs: np.ndarray,
    *,
    sigma: float = CLIP_SIGMA,
    maxiters: int = CLIP_MAXITERS,
) -> tuple[float, int, float]:
    """Clip residuals vs weighted mean; return chi2_red, n_removed, outlier_fraction."""
    m = np.asarray(mags, dtype=np.float64)
    e = np.asarray(errs, dtype=np.float64)
    ok = np.isfinite(m) & np.isfinite(e) & (e > 0)
    m = m[ok]
    e = e[ok]
    n0 = int(m.size)
    if n0 < 3:
        return float("nan"), 0, float("nan")
    keep = np.ones(n0, dtype=bool)
    mag_ref = float(np.median(m))
    for _ in range(maxiters):
        mm = m[keep]
        ee = e[keep]
        if mm.size < 3:
            break
        w = 1.0 / (ee * ee)
        wsum = float(np.sum(w))
        mag_ref = float(np.sum(w * mm) / wsum) if wsum > 0 else float(np.median(mm))
        resid = m - mag_ref
        sig = float(np.nanmedian(np.abs(resid[keep] - np.median(resid[keep])))) * 1.4826
        if not math.isfinite(sig) or sig <= 0:
            sig = float(np.nanstd(resid[keep]))
        if not math.isfinite(sig) or sig <= 0:
            break
        new_keep = keep.copy()
        new_keep[keep] = np.abs(resid[keep]) <= sigma * sig
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    n_removed = n0 - int(np.count_nonzero(keep))
    _, _, chi2_red, _ = reduced_chi2_constant(m[keep], e[keep], mag_ref=mag_ref)
    frac = float(n_removed / n0) if n0 else float("nan")
    return chi2_red, n_removed, frac


def _mad_robust_chi2(mags: np.ndarray, errs: np.ndarray) -> float:
    """median(((m_i - median(m)) / err_i)^2) * 1.4826^2; robust to outliers."""
    m = np.asarray(mags, dtype=np.float64)
    e = np.asarray(errs, dtype=np.float64)
    ok = np.isfinite(m) & np.isfinite(e) & (e > 0)
    m = m[ok]
    e = e[ok]
    if m.size < 3:
        return float("nan")
    med = float(np.median(m))
    z2 = ((m - med) / e) ** 2
    return float(np.median(z2) * (1.4826**2))


def _epoch_contributors(
    lc_df: pd.DataFrame,
    *,
    check_cid: str,
    target_cid: str,
) -> list[dict[str, Any]]:
    m = pd.to_numeric(lc_df["mag_calib_final"], errors="coerce").to_numpy(dtype=np.float64)
    e = pd.to_numeric(lc_df["err"], errors="coerce").to_numpy(dtype=np.float64)
    ok = np.isfinite(m) & np.isfinite(e) & (e > 0)
    if int(np.count_nonzero(ok)) < 3:
        return []
    m = m[ok]
    e = e[ok]
    w = 1.0 / (e * e)
    ref = float(np.sum(w * m) / np.sum(w))
    chi2_pt = ((m - ref) / e) ** 2
    idx_ok = np.where(ok)[0]
    rows: list[dict[str, Any]] = []
    bjd = pd.to_numeric(lc_df.get("bjd"), errors="coerce").to_numpy(dtype=np.float64)
    sf = lc_df.get("source_file", pd.Series([""] * len(lc_df))).astype(str).tolist()
    for local_i, gchi in enumerate(chi2_pt):
        gi = int(idx_ok[local_i])
        rows.append(
            {
                "target_cid": target_cid,
                "check_cid": check_cid,
                "epoch_index": gi,
                "bjd": float(bjd[gi]) if gi < len(bjd) and np.isfinite(bjd[gi]) else None,
                "source_file": sf[gi] if gi < len(sf) else "",
                "mag_calib_final": float(m[local_i]),
                "mag_ref": ref,
                "residual_mag": float(m[local_i] - ref),
                "err": float(e[local_i]),
                "chi2_point": float(gchi),
            }
        )
    return rows


def _enrich_outlier_from_proc(row: dict[str, Any], proc_dir: Path) -> dict[str, Any]:
    sf = str(row.get("source_file") or "").strip()
    cid = str(row.get("check_cid") or "")
    proc_path = proc_dir / sf if sf else None
    out = dict(row)
    out["peak_max_adu"] = None
    out["fwhm_estimate_px"] = None
    out["psf_chi2"] = None
    out["sharp_vs_psf"] = None
    if proc_path is None or not proc_path.is_file():
        return out
    try:
        pdf = pd.read_csv(proc_path, low_memory=False, dtype={"catalog_id": str})
    except Exception:  # noqa: BLE001
        return out
    id_col = "catalog_id" if "catalog_id" in pdf.columns else "name"
    sub = pdf.loc[pdf[id_col].astype(str).str.strip() == cid]
    if sub.empty:
        return out
    pr = sub.iloc[0]
    peak = float(pd.to_numeric(pr.get("peak_max_adu"), errors="coerce"))
    fwhm = float(pd.to_numeric(pr.get("fwhm_estimate_px"), errors="coerce"))
    psf_chi2 = float(pd.to_numeric(pr.get("psf_chi2"), errors="coerce"))
    out["peak_max_adu"] = peak if math.isfinite(peak) else None
    out["fwhm_estimate_px"] = fwhm if math.isfinite(fwhm) else None
    out["psf_chi2"] = psf_chi2 if math.isfinite(psf_chi2) else None
    if math.isfinite(fwhm):
        out["sharp_vs_psf"] = "sharp_cr_candidate" if fwhm < 2.0 else "psf_like"
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft-id", type=int, default=435)
    parser.add_argument(
        "--draft-root",
        type=Path,
        default=None,
        help="Anchor snapshot root (default: draft_000435_snapshot_skysurface_20260716)",
    )
    parser.add_argument("--out", type=Path, default=REPO / "tmp" / "audit_stage3_part1c_results.json")
    args = parser.parse_args()

    cfg = AppConfig()
    if args.draft_root is not None:
        draft = Path(args.draft_root)
    else:
        draft = Path(cfg.archive_root) / "Drafts" / "draft_000435_snapshot_skysurface_20260716"
    ps = draft / "platesolve" / SETUP
    phot = ps / "photometry"
    lc_dir = phot / "lightcurves"
    lights = draft / "detrended_aligned" / "lights" / SETUP

    gh, dirty, _ = _resolve_git_provenance()
    equipment_id = 1
    sigma_sys = resolve_sigma_sys_mag(equipment_id, cfg, rig_label=SETUP)
    scint_params = resolve_rig_scintillation_params(draft_id=args.draft_id, setup=SETUP, cfg=cfg)

    state = _phase2a_prepare_shared_state(
        output_dir=phot,
        lc_dir=lc_dir,
        masterstar_fits_path=ps / "MASTERSTAR.fits",
        comparison_stars_csv=phot / "comparison_stars_per_target.csv",
        per_frame_csv_dir=lights,
        progress_cb=None,
        active_targets_csv=ps / "variable_targets.csv",
        detrended_aligned_dir=lights,
        fwhm_px=3.2,
        cfg=cfg,
        db=None,
        draft_id=args.draft_id,
    )

    check_rows: list[dict[str, Any]] = []
    all_contributors: list[dict[str, Any]] = []

    for ck_path in sorted(lc_dir.glob("check_kmag_*.csv")):
        target_cid = ck_path.stem.replace("check_kmag_", "")
        ckdf = pd.read_csv(ck_path, low_memory=False)
        if ckdf.empty:
            continue
        id_col = "check_catalog_id" if "check_catalog_id" in ckdf.columns else "check_cid"
        check_cid = str(ckdf[id_col].iloc[0]).strip()
        lc_df = photometer_check_star_production_path(
            state=state,
            parent_target_cid=target_cid,
            check_cid=check_cid,
            masterstar_fits_path=ps / "MASTERSTAR.fits",
            lc_dir=phot / "diag_check_lc" / target_cid,
            output_dir=phot,
        )
        row: dict[str, Any] = {"target_cid": target_cid, "check_cid": check_cid}
        if lc_df is None or "mag_calib_final" not in lc_df.columns:
            check_rows.append(row)
            continue
        m = pd.to_numeric(lc_df["mag_calib_final"], errors="coerce").to_numpy(dtype=np.float64)
        e = pd.to_numeric(lc_df["err"], errors="coerce").to_numpy(dtype=np.float64)
        ok = np.isfinite(m) & np.isfinite(e) & (e > 0)
        row["N_epochs"] = int(np.count_nonzero(ok))
        if int(np.count_nonzero(ok)) >= 3:
            _, _, chi2_raw, _ = reduced_chi2_constant(m[ok], e[ok])
            chi2_clip, n_out, out_frac = _iterative_sigma_clip_chi2(m[ok], e[ok])
            row["chi2_red_raw"] = float(chi2_raw)
            row["chi2_red_clipped"] = float(chi2_clip)
            row["clip_sigma"] = CLIP_SIGMA
            row["clip_maxiters"] = CLIP_MAXITERS
            row["n_outliers"] = int(n_out)
            row["outlier_fraction"] = float(out_frac)
            row["chi2_mad_robust"] = float(_mad_robust_chi2(m[ok], e[ok]))
            row["err_min"] = float(np.min(e[ok]))
            row["err_median"] = float(np.median(e[ok]))
            row["scatter_mag"] = float(np.std(m[ok] - np.median(m[ok]), ddof=1))
            all_contributors.extend(
                _epoch_contributors(lc_df, check_cid=check_cid, target_cid=target_cid)
            )
        check_rows.append(row)

    top20 = sorted(all_contributors, key=lambda r: r["chi2_point"], reverse=True)[:20]
    proc_dir = lights
    top20_enriched = [_enrich_outlier_from_proc(r, proc_dir) for r in top20]

    # Frame-wide vs single-star: same source_file in multiple top contributors
    sf_counts: dict[str, int] = {}
    for r in top20:
        sf = str(r.get("source_file") or "")
        if sf:
            sf_counts[sf] = sf_counts.get(sf, 0) + 1

    out: dict[str, Any] = {
        "provenance": {"git_hash": gh, "git_dirty": dirty},
        "draft_root": str(draft),
        "equipment_id": equipment_id,
        "sigma_sys_mag_resolved": float(sigma_sys),
        "sigma_sys_mag_config": dict(cfg.sigma_sys_mag),
        "clip_params": {"sigma": CLIP_SIGMA, "maxiters": CLIP_MAXITERS},
        "mad_estimator": "median(((m-median(m))/err)^2)*1.4826^2",
        "check_stars": check_rows,
        "distribution": {
            "chi2_red_raw": _dist_stats([r["chi2_red_raw"] for r in check_rows if "chi2_red_raw" in r]),
            "chi2_red_clipped": _dist_stats([r["chi2_red_clipped"] for r in check_rows if "chi2_red_clipped" in r]),
            "chi2_mad_robust": _dist_stats([r["chi2_mad_robust"] for r in check_rows if "chi2_mad_robust" in r]),
            "outlier_fraction": _dist_stats([r["outlier_fraction"] for r in check_rows if "outlier_fraction" in r]),
        },
        "top20_outlier_epochs": top20_enriched,
        "frame_repeat_in_top20": {k: v for k, v in sf_counts.items() if v > 1},
    }
    if scint_params is not None:
        out["scintillation_sigma_median_mmag"] = float(
            1000.0
            * scintillation_sigma(
                telescope_diameter_m=float(scint_params.telescope_diameter_m),
                airmass=1.0,
                exposure_s=float(scint_params.exposure_s),
                altitude_m=float(scint_params.altitude_m),
                c_y=float(scint_params.c_y),
            )
        )
        out["rig_telescope_diameter_m"] = float(scint_params.telescope_diameter_m)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {args.out} median chi2_raw={out['distribution']['chi2_red_raw'].get('median')} "
        f"chi2_clip={out['distribution']['chi2_red_clipped'].get('median')}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
