#!/usr/bin/env python3
"""Audit Stage 3 Part 1b: check-star chi2 via production photometry path."""

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

sys.path.insert(0, str(REPO / "dev" / "scripts"))
from chi2_sigma_gate import reduced_chi2_constant  # noqa: E402

SETUP = "NoFilter_60_2"
DRAFT_ID = 435


def _err_budget_fractions(lc_df: pd.DataFrame) -> dict[str, float]:
    cols = ["err_photon", "err_sem_rel", "err_sigma_sys_rel", "err"]
    if not all(c in lc_df.columns for c in cols):
        return {}
    ep = pd.to_numeric(lc_df["err_photon"], errors="coerce").to_numpy(dtype=np.float64)
    sem = pd.to_numeric(lc_df["err_sem_rel"], errors="coerce").to_numpy(dtype=np.float64)
    sys = pd.to_numeric(lc_df["err_sigma_sys_rel"], errors="coerce").to_numpy(dtype=np.float64)
    err = pd.to_numeric(lc_df["err"], errors="coerce").to_numpy(dtype=np.float64)
    var = err ** 2
    ok = np.isfinite(var) & (var > 0)
    if not np.any(ok):
        return {}
    return {
        "frac_photon": float(np.nanmean((ep[ok] ** 2) / var[ok])),
        "frac_sem": float(np.nanmean((sem[ok] ** 2) / var[ok])),
        "frac_sys": float(np.nanmean((sys[ok] ** 2) / var[ok])),
    }


def _target_err_proxy_chi2(lc_dir: Path, target_cid: str) -> float | None:
    lc = lc_dir / f"lightcurve_{target_cid}.csv"
    if not lc.is_file():
        return None
    df = pd.read_csv(lc, usecols=["mag_calib_final", "err"])
    m = pd.to_numeric(df["mag_calib_final"], errors="coerce").to_numpy(dtype=np.float64)
    e = pd.to_numeric(df["err"], errors="coerce").to_numpy(dtype=np.float64)
    ok = np.isfinite(m) & np.isfinite(e) & (e > 0)
    if int(np.count_nonzero(ok)) < 3:
        return None
    return float(reduced_chi2_constant(m[ok], e[ok])[0])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft-id", type=int, default=DRAFT_ID)
    parser.add_argument(
        "--draft-root",
        type=Path,
        default=None,
        help="Override draft folder (e.g. anchor snapshot)",
    )
    parser.add_argument("--out", type=Path, default=REPO / "tmp" / "audit_stage3_part1b_results.json")
    args = parser.parse_args()

    cfg = AppConfig()
    if args.draft_root is not None:
        draft = Path(args.draft_root)
    else:
        draft = Path(cfg.archive_root) / "Drafts" / f"draft_{args.draft_id:06d}"
    ps = draft / "platesolve" / SETUP
    phot = ps / "photometry"
    lc_dir = phot / "lightcurves"
    lights = draft / "detrended_aligned" / "lights" / SETUP

    gh, dirty, _ = _resolve_git_provenance()
    out: dict[str, Any] = {
        "provenance": {"git_hash": gh, "git_dirty": dirty},
        "draft_id": args.draft_id,
        "check_stars": [],
    }

    diag_lc = phot / "diag_check_lc"
    diag_lc.mkdir(parents=True, exist_ok=True)

    state = _phase2a_prepare_shared_state(
        output_dir=phot,
        lc_dir=lc_dir,
        masterstar_fits_path=ps / "MASTERSTAR.fits",
        comparison_stars_csv=ps / "photometry" / "comparison_stars_per_target.csv",
        per_frame_csv_dir=lights,
        progress_cb=None,
        active_targets_csv=ps / "variable_targets.csv",
        detrended_aligned_dir=lights,
        fwhm_px=3.2,
        cfg=cfg,
        db=None,
        draft_id=args.draft_id,
    )

    scint_params = resolve_rig_scintillation_params(draft_id=args.draft_id, setup=SETUP, cfg=cfg)

    for ck_path in sorted(lc_dir.glob("check_kmag_*.csv")):
        target_cid = ck_path.stem.replace("check_kmag_", "")
        ckdf = pd.read_csv(ck_path, low_memory=False)
        if ckdf.empty:
            continue
        id_col = "check_catalog_id" if "check_catalog_id" in ckdf.columns else "check_cid"
        if id_col not in ckdf.columns:
            continue
        check_cid = str(ckdf[id_col].iloc[0]).strip()
        subdir = diag_lc / target_cid
        subdir.mkdir(parents=True, exist_ok=True)
        lc_df = photometer_check_star_production_path(
            state=state,
            parent_target_cid=target_cid,
            check_cid=check_cid,
            masterstar_fits_path=ps / "MASTERSTAR.fits",
            lc_dir=subdir,
            output_dir=phot,
        )
        row: dict[str, Any] = {
            "target_cid": target_cid,
            "check_cid": check_cid,
            "N_sidecar": int(len(ckdf)),
        }
        if lc_df is not None and "mag_calib_final" in lc_df.columns and "err" in lc_df.columns:
            m = pd.to_numeric(lc_df["mag_calib_final"], errors="coerce").to_numpy(dtype=np.float64)
            e = pd.to_numeric(lc_df["err"], errors="coerce").to_numpy(dtype=np.float64)
            ok = np.isfinite(m) & np.isfinite(e) & (e > 0)
            row["N_production"] = int(np.count_nonzero(ok))
            row["chi2_red_production"] = (
                float(reduced_chi2_constant(m[ok], e[ok])[0]) if int(np.count_nonzero(ok)) >= 3 else None
            )
            row["err_budget"] = _err_budget_fractions(lc_df)
            # Part 1 flawed method: sidecar kmag + reconstructed err (for reconciliation)
            if "kmag" in ckdf.columns:
                km = pd.to_numeric(ckdf["kmag"], errors="coerce").to_numpy(dtype=np.float64)
                okk = np.isfinite(km)
                if int(np.count_nonzero(okk)) >= 3:
                    # old script used inflated err; approximate from sidecar if err col exists
                    if "err" in ckdf.columns:
                        ek = pd.to_numeric(ckdf["err"], errors="coerce").to_numpy(dtype=np.float64)
                        ok2 = okk & np.isfinite(ek) & (ek > 0)
                        if int(np.count_nonzero(ok2)) >= 3:
                            row["chi2_red_sidecar"] = float(reduced_chi2_constant(km[ok2], ek[ok2]))
        row["chi2_red_target_proxy"] = _target_err_proxy_chi2(lc_dir, target_cid)
        if row.get("chi2_red_target_proxy") is not None and isinstance(row["chi2_red_target_proxy"], tuple):
            row["chi2_red_target_proxy"] = float(row["chi2_red_target_proxy"][0])
        if scint_params is not None and lc_df is not None and "airmass" in lc_df.columns:
            am = pd.to_numeric(lc_df["airmass"], errors="coerce").to_numpy(dtype=np.float64)
            sig_scint = np.array(
                [
                    scintillation_sigma(
                        telescope_diameter_m=float(scint_params.telescope_diameter_m),
                        airmass=float(a),
                        exposure_s=float(scint_params.exposure_s),
                        altitude_m=float(scint_params.altitude_m),
                        c_y=float(scint_params.c_y),
                    )
                    if math.isfinite(a) and float(a) >= 1.0
                    else float("nan")
                    for a in am
                ],
                dtype=np.float64,
            )
            row["sigma_scint_median"] = float(np.nanmedian(sig_scint))
        out["check_stars"].append(row)

    prod = [r["chi2_red_production"] for r in out["check_stars"] if r.get("chi2_red_production") is not None]
    out["median_chi2_red_production"] = float(np.median(prod)) if prod else None
    out["n_check_stars"] = len(out["check_stars"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out} median_chi2_red_production={out['median_chi2_red_production']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
