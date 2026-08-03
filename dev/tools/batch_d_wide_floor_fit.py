#!/usr/bin/env python3
"""Fit sigma_sys_mag floor for equipment_id 1 (wide rig) with scintillation wired (batch D GATE 1)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "dev" / "scripts"))
import _bootstrap  # noqa: E402,F401

from config import AppConfig  # noqa: E402
from scripts.chi2_sigma_gate import reduced_chi2_constant  # noqa: E402
from scripts.fit_sigma_floor import (  # noqa: E402
    _build_star_arrays,
    _passes_saturation_gate,
    _pooled_chi2_dof,
    _root_find_sigma_sys,
    collect_comp_candidates,
    pick_anchor_target,
    pick_g_coverage,
)
from sigma_budget import resolve_rig_scintillation_params  # noqa: E402
from sigma_floor_core import combine_production_err_mag, scintillation_mag_per_epoch  # noqa: E402

DRAFT_ID = 435
SETUP = "NoFilter_60_2"
EQUIPMENT_ID = 1


def _sigma_mag_with_scint(
    photon_rel: np.ndarray,
    sem_mag: np.ndarray,
    airmass: np.ndarray,
    rig: Any,
    *,
    sigma_sys_mag: float,
) -> np.ndarray:
    n = len(photon_rel)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        am = float(airmass[i])
        scint_m = (
            scintillation_mag_per_epoch(
                telescope_diameter_m=float(rig.telescope_diameter_m),
                airmass=am,
                exposure_s=float(rig.exposure_s),
                altitude_m=float(rig.altitude_m),
                c_y=float(rig.c_y),
            )
            if math.isfinite(am) and am >= 1.0
            else 0.0
        )
        out[i] = combine_production_err_mag(
            float(photon_rel[i]),
            float(sem_mag[i]),
            sigma_sys_mag=float(sigma_sys_mag),
            sigma_scint_mag=float(scint_m),
        )
    return out


def _median_chi2_clipped(stars: list[dict[str, Any]], sigma_sys_mag: float, rig: Any) -> float:
    chi2_vals: list[float] = []
    for st in stars:
        sig = _sigma_mag_with_scint(
            st["photon_rel"],
            st["sem_mag"],
            st["airmass"],
            rig,
            sigma_sys_mag=float(sigma_sys_mag),
        )
        c2, _, c2d, _ = reduced_chi2_constant(st["mags"], sig)
        if math.isfinite(c2d):
            chi2_vals.append(float(c2d))
    return float(np.median(chi2_vals)) if chi2_vals else float("nan")


def _pooled_chi2_scint(stars: list[dict[str, Any]], sigma_sys_mag: float, rig: Any) -> float:
    chi2_sum = 0.0
    dof_sum = 0
    for st in stars:
        sig = _sigma_mag_with_scint(
            st["photon_rel"],
            st["sem_mag"],
            st["airmass"],
            rig,
            sigma_sys_mag=float(sigma_sys_mag),
        )
        c2, dof, _, _ = reduced_chi2_constant(st["mags"], sig)
        if math.isfinite(c2) and dof > 0:
            chi2_sum += c2
            dof_sum += dof
    return chi2_sum / dof_sum if dof_sum > 0 else float("nan")


def _root_find_with_scint(stars: list[dict[str, Any]], rig: Any, *, hi: float = 0.03) -> tuple[float, float]:
    lo, hi_v = 0.0, float(hi)
    f_lo = _pooled_chi2_scint(stars, lo, rig) - 1.0
    f_hi = _pooled_chi2_scint(stars, hi_v, rig) - 1.0
    if not math.isfinite(f_lo):
        return float("nan"), float("nan")
    if f_lo <= 0:
        return 0.0, _pooled_chi2_scint(stars, 0.0, rig)
    if not math.isfinite(f_hi) or f_hi > 0:
        for mult in (2.0, 4.0, 8.0, 16.0):
            hi2 = hi_v * mult
            if _pooled_chi2_scint(stars, hi2, rig) - 1.0 <= 0:
                hi_v = hi2
                f_hi = _pooled_chi2_scint(stars, hi_v, rig) - 1.0
                break
        else:
            return float("nan"), float("nan")
    for _ in range(60):
        mid = 0.5 * (lo + hi_v)
        f_mid = _pooled_chi2_scint(stars, mid, rig) - 1.0
        if abs(f_mid) < 1e-4:
            return mid, _pooled_chi2_scint(stars, mid, rig)
        if f_mid > 0:
            lo = mid
        else:
            hi_v = mid
    mid = 0.5 * (lo + hi_v)
    return mid, _pooled_chi2_scint(stars, mid, rig)


def _build_stars_with_airmass(cfg: AppConfig) -> tuple[list[dict[str, Any]], Any, dict[str, Any]]:
    draft = Path(cfg.archive_root) / "Drafts" / "draft_000435_snapshot_skysurface_20260716"
    phot_dir = draft / "platesolve" / SETUP / "photometry"
    meta = (
        json.loads((phot_dir / "pipeline_meta.json").read_text(encoding="utf-8"))
        if (phot_dir / "pipeline_meta.json").is_file()
        else {}
    )
    rig = resolve_rig_scintillation_params(
        draft_id=DRAFT_ID, setup=SETUP, cfg=cfg, pipeline_meta=meta,
    )
    from check_star_kmag import resolve_proc_csv_dir  # noqa: PLC0415

    proc_dir = resolve_proc_csv_dir(phot_dir, SETUP)
    if proc_dir is None:
        raise RuntimeError("proc_dir missing")
    comp_all = (
        __import__("pandas").read_csv(
            phot_dir / "comparison_stars_per_target.csv",
            low_memory=False,
            dtype={"catalog_id": str, "target_catalog_id": str},
        )
        if (phot_dir / "comparison_stars_per_target.csv").is_file()
        else __import__("pandas").DataFrame()
    )
    candidates = collect_comp_candidates(phot_dir, min_frames=15)
    anchor = pick_anchor_target(phot_dir, comp_all)
    calibrators = pick_g_coverage(candidates, aim=12)
    gain = float(meta.get("gain") or 1.0)
    rn = float(meta.get("read_noise") or 10.0)
    stars: list[dict[str, Any]] = []
    for _, row in calibrators.iterrows():
        cid = str(row["catalog_id"]).strip()
        built = _build_star_arrays(
            cid,
            phot_dir=phot_dir,
            setup=SETUP,
            anchor=anchor,
            proc_dir=proc_dir,
            cfg=cfg,
            gain=gain,
            read_noise=rn,
        )
        if not built:
            continue
        from scripts.select_constant_calibrators import build_loo_differential_lc  # noqa: PLC0415

        loo = build_loo_differential_lc(
            cid, phot_dir=phot_dir, setup=SETUP, anchor_target=anchor, cfg=cfg,
        )
        if loo is None or loo.empty or "airmass" not in loo.columns:
            continue
        am = __import__("pandas").to_numeric(loo["airmass"], errors="coerce").to_numpy(dtype=np.float64)
        n = min(len(built["mags"]), len(am))
        built["airmass"] = am[:n]
        built["mags"] = built["mags"][:n]
        built["photon_rel"] = built["photon_rel"][:n]
        built["sem_mag"] = built["sem_mag"][:n]
        if _passes_saturation_gate(
            built["catalog_id"],
            phot_dir=phot_dir,
            proc_dir=proc_dir,
            setup=SETUP,
            anchor=anchor,
            cfg=cfg,
        ):
            stars.append(built)
    info = {
        "draft_id": DRAFT_ID,
        "equipment_id": EQUIPMENT_ID,
        "anchor_target": anchor,
        "n_stars": len(stars),
        "rig": rig.to_dict(),
    }
    return stars, rig, info


def main() -> int:
    cfg = AppConfig()
    stars, rig, info = _build_stars_with_airmass(cfg)
    if len(stars) < 3:
        print(json.dumps({"error": "insufficient stars", **info}, indent=2))
        return 1

    # Baseline: no scintillation, no floor (legacy production err budget)
    chi2_before = _median_chi2_clipped(
        [
            {
                **st,
                "sem_mag": st["sem_mag"],
            }
            for st in stars
        ],
        0.0,
        rig,
    )
    # Override: before = photon+sem only (no scint)
    chi2_before_vals: list[float] = []
    for st in stars:
        sig = np.array(
            [
                combine_production_err_mag(float(p), float(s), sigma_sys_mag=0.0, sigma_scint_mag=0.0)
                for p, s in zip(st["photon_rel"], st["sem_mag"], strict=True)
            ],
            dtype=np.float64,
        )
        _, _, c2d, _ = reduced_chi2_constant(st["mags"], sig)
        if math.isfinite(c2d):
            chi2_before_vals.append(float(c2d))
    chi2_before_med = float(np.median(chi2_before_vals)) if chi2_before_vals else float("nan")

    chi2_scint_only = _median_chi2_clipped(stars, 0.0, rig)
    floor_mag, chi2_at_fit = _root_find_with_scint(stars, rig)
    chi2_scint_floor = _median_chi2_clipped(stars, float(floor_mag), rig) if math.isfinite(floor_mag) else float("nan")

    quoted_err_mmag = float(
        np.median(
            [
                combine_production_err_mag(float(p), float(s), sigma_sys_mag=0.0, sigma_scint_mag=0.0) * 1000.0
                for st in stars
                for p, s in zip(st["photon_rel"], st["sem_mag"], strict=True)
            ]
        )
    )
    post_scint_mmag = float(
        np.median(
            [
                combine_production_err_mag(
                    float(p),
                    float(s),
                    sigma_sys_mag=0.0,
                    sigma_scint_mag=scintillation_mag_per_epoch(
                        telescope_diameter_m=float(rig.telescope_diameter_m),
                        airmass=float(am),
                        exposure_s=float(rig.exposure_s),
                        altitude_m=float(rig.altitude_m),
                        c_y=float(rig.c_y),
                    ),
                )
                * 1000.0
                for st in stars
                for p, s, am in zip(st["photon_rel"], st["sem_mag"], st["airmass"], strict=True)
                if math.isfinite(float(am)) and float(am) >= 1.0
            ]
        )
    )

    out = {
        **info,
        "median_quoted_err_photon_sem_mmag": quoted_err_mmag,
        "median_quoted_err_post_scint_mmag": post_scint_mmag,
        "chi2_red_clipped_before": chi2_before_med,
        "chi2_red_clipped_scint_only": chi2_scint_only,
        "chi2_red_clipped_scint_plus_floor": chi2_scint_floor,
        "sigma_sys_mag_fitted": floor_mag,
        "sigma_sys_mmag_fitted": floor_mag * 1000.0 if math.isfinite(floor_mag) else None,
        "pooled_chi2_at_fit": chi2_at_fit,
        "sanity_2_5_mmag": (
            0.002 <= floor_mag <= 0.005 if math.isfinite(floor_mag) else False
        ),
    }
    out_path = REPO / "tmp" / "batch_d_wide_floor_fit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
