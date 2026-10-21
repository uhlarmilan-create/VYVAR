"""DRAFT-514-TRIAGE D1/D2 structural measurements on proc CSV (no ensemble cut)."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "dev" / "results"


def main() -> None:
    proc = (
        ROOT
        / "Archive"
        / "Drafts"
        / "draft_000514"
        / "detrended_aligned"
        / "lights"
        / "NoFilter_60_2"
        / "proc_BO_CVn_Light_001.csv"
    )
    sha = "pending"
    try:
        import subprocess

        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        pass

    df = pd.read_csv(proc)
    n_raw = len(df)
    # Dedup for analysis (A2 cause is separate)
    df_u = df.drop_duplicates(subset=["catalog_id"], keep="first").copy()
    n_unique = len(df_u)
    n_dup_extra = n_raw - n_unique

    plate = 9.55169  # arcsec/px bin2 (COMP-POOL-02)
    fwhm = pd.to_numeric(df_u.get("fwhm_estimate_px"), errors="coerce")
    ap = pd.to_numeric(df_u.get("aperture_r_px"), errors="coerce")
    med_fwhm = float(np.nanmedian(fwhm))
    med_ap = float(np.nanmedian(ap))
    r_over_fwhm = med_ap / med_fwhm if med_fwhm > 0 else float("nan")
    # Gaussian EE at r = k * FWHM: EE = 1 - exp(-0.5*(r/sigma)^2), sigma=FWHM/2.355
    sigma = med_fwhm / 2.354820045
    ee = float(1.0 - math.exp(-0.5 * (med_ap / sigma) ** 2)) if sigma > 0 else float("nan")

    x = pd.to_numeric(df_u["x"], errors="coerce").to_numpy()
    y = pd.to_numeric(df_u["y"], errors="coerce").to_numpy()
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    # Neighbour within aperture radius (median)
    r_ap = med_ap
    # O(n^2) on 3500 is OK
    n = len(x)
    has_neighbour = np.zeros(n, dtype=bool)
    sep_fwhm: list[float] = []
    for i in range(n):
        dx = x - x[i]
        dy = y - y[i]
        dist = np.hypot(dx, dy)
        dist[i] = np.inf
        m = dist < r_ap
        if m.any():
            has_neighbour[i] = True
            for d in dist[m]:
                if med_fwhm > 0:
                    sep_fwhm.append(float(d / med_fwhm))

    # Pool members: from admission or comparison_stars_per_target
    pool_path = (
        ROOT
        / "Archive"
        / "Drafts"
        / "draft_000514"
        / "platesolve"
        / "NoFilter_60_2"
        / "photometry"
        / "comparison_stars_per_target.csv"
    )
    pool_ids: set[str] = set()
    if pool_path.is_file():
        pool = pd.read_csv(pool_path, usecols=["catalog_id"])
        pool_ids = set(pool["catalog_id"].astype(str))
    ids_u = df_u.loc[ok, "catalog_id"].astype(str).to_numpy()
    n_pool_affected = int(sum(1 for i, flag in enumerate(has_neighbour) if flag and ids_u[i] in pool_ids))

    # FWHM sources from infolog / pipeline_meta
    meta_path = pool_path.parent / "pipeline_meta.json"
    meta = {}
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    out = {
        "commit_sha": sha,
        "proc_file": str(proc),
        "A2_duplicates": {
            "rows": n_raw,
            "unique_catalog_id": n_unique,
            "extra_duplicate_rows": n_dup_extra,
            "cause": (
                "_export_per_frame_run_catalog_core did not call "
                "_proc_deduplicate_matched_catalog_rows before write; "
                "alternate export path did. Multiple DAO detections match same Gaia id."
            ),
            "forced_photometry_role": "not the cause (duplicate rows have forced_photometry=False)",
        },
        "D1_blends": {
            "plate_scale_arcsec_per_px": plate,
            "median_fwhm_estimate_px": med_fwhm,
            "median_fwhm_arcsec": med_fwhm * plate,
            "median_aperture_r_px": med_ap,
            "n_stars_unique": n_unique,
            "n_with_neighbour_inside_aperture": int(has_neighbour.sum()),
            "frac_with_neighbour_inside_aperture": float(has_neighbour.mean()),
            "neighbour_sep_fwhm_p16": float(np.percentile(sep_fwhm, 16)) if sep_fwhm else None,
            "neighbour_sep_fwhm_p50": float(np.percentile(sep_fwhm, 50)) if sep_fwhm else None,
            "neighbour_sep_fwhm_p84": float(np.percentile(sep_fwhm, 84)) if sep_fwhm else None,
            "n_pool_members_with_neighbour": n_pool_affected,
            "n_pool_ids_in_csv": len(pool_ids),
            "unit_sep": "FWHM",
        },
        "D2_aperture_vs_psf": {
            "r_over_fwhm": r_over_fwhm,
            "gaussian_enclosed_flux_fraction": ee,
            "median_fwhm_estimate_px": med_fwhm,
            "phase2a_fwhm_px_from_infolog": 3.301,
            "auto_fwhm_median_from_task": 5.311,
            "pipeline_meta_fwhm": meta.get("fwhm_px") or meta.get("fwhm") or meta.get("resolved_facts", {}).get("fwhm_px"),
            "note": (
                "Three FWHM values disagree. Production aperture uses Phase 2A "
                "resolved fwhm_px (infolog 3.301) times aperture_fwhm_factor; "
                "per-star fwhm_estimate_px is a separate moment estimate."
            ),
        },
    }
    path = OUT / "DRAFT_514_TRIAGE_D.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()
