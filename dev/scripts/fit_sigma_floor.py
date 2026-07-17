#!/usr/bin/env python3
"""Fit per-rig sigma_sys floor from constant-check-star cohorts (PROD-SIGMA-FLOOR Part A/B)."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import AppConfig  # noqa: E402
from gaia_catalog_id import normalize_gaia_source_id  # noqa: E402
from scripts.chi2_sigma_gate import (  # noqa: E402
    reduced_chi2_constant,
    saturation_margin_distribution,
    write_summary_json,
)
from scripts.select_constant_calibrators import (  # noqa: E402
    build_loo_differential_lc,
    collect_comp_candidates,
    compute_loo_production_ensemble_scatter,
    pick_anchor_target,
    pick_g_coverage,
)
from scripts.sparse_comp_diag import SS_CAM_CID  # noqa: E402
from sigma_budget import resolve_rig_scintillation_params  # noqa: E402
from sigma_floor_core import (  # noqa: E402
    combine_production_err_mag,
    pzq_fit_sigma_r,
)

# Wide-rig SIGMA-A3 reference (mmag); consistency gate Part A.4
WIDE_RIG_PRIOR_MMAG = 6.5
WIDE_RIG_PRIOR_CI_MMAG = (5.5, 7.5)
MIN_EPOCHS = 15
BOOTSTRAP_DRAWS = 500
SPLIT_SEED = 424426


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_ROOT, text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _norm_id(x: object) -> str:
    return str(normalize_gaia_source_id(x) or "").strip()


def _stamp(payload: dict[str, Any]) -> dict[str, Any]:
    payload["git_head"] = _git_head()
    payload["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    return payload


def _photon_err_rel_from_proc(
    loo: pd.DataFrame,
    proc_dir: Path,
    catalog_id: str,
    *,
    gain: float,
    read_noise: float,
) -> np.ndarray:
    from scripts.chi2_sigma_gate import load_proc_row_for_source, _proc_aperture_area_px  # noqa: PLC0415
    from scripts.chi2_sigma_gate import _relative_flux_sigma_with_bkg, _sky_pp_for_photometric_error  # noqa: PLC0415
    from photometry_core import ERR_BKG_SOURCE_COL, SIGMA_BKG_AP_COL  # noqa: PLC0415

    n = len(loo)
    out = np.full(n, np.nan, dtype=np.float64)
    for i, sf in enumerate(loo.get("source_file", pd.Series([""] * n)).astype(str).tolist()):
        row = load_proc_row_for_source(proc_dir, sf, catalog_id)
        if row is None:
            continue
        flux = float(pd.to_numeric(row.get("dao_flux"), errors="coerce"))
        if not math.isfinite(flux) or flux <= 0:
            continue
        sky = float(_sky_pp_for_photometric_error(row))
        area = _proc_aperture_area_px(row)
        sig_bkg_raw = float(pd.to_numeric(row.get(SIGMA_BKG_AP_COL), errors="coerce"))
        sig_bkg_ap = sig_bkg_raw if math.isfinite(sig_bkg_raw) else None
        err_bkg_source = str(row.get(ERR_BKG_SOURCE_COL, "")).strip() or None
        err_rel, _ = _relative_flux_sigma_with_bkg(
            flux, sky, area,
            sigma_bkg_ap=sig_bkg_ap,
            err_bkg_source=err_bkg_source,
            gain=gain,
            read_noise=read_noise,
        )
        if math.isfinite(err_rel) and err_rel > 0:
            out[i] = err_rel
    return out


def _build_star_arrays(
    cid: str,
    *,
    phot_dir: Path,
    setup: str,
    anchor: str,
    proc_dir: Path,
    cfg: AppConfig,
    gain: float,
    read_noise: float,
) -> dict[str, Any] | None:
    loo = build_loo_differential_lc(cid, phot_dir=phot_dir, setup=setup, anchor_target=anchor, cfg=cfg)
    if loo is None or loo.empty:
        return None
    prod_scatter = compute_loo_production_ensemble_scatter(
        cid, phot_dir=phot_dir, setup=setup, anchor_target=anchor, cfg=cfg,
    )
    if prod_scatter is None:
        return None
    mags = pd.to_numeric(loo.get("delta_mag"), errors="coerce").to_numpy(dtype=np.float64)
    photon = _photon_err_rel_from_proc(loo, proc_dir, cid, gain=gain, read_noise=read_noise)
    sem = np.asarray(prod_scatter, dtype=np.float64)
    n = min(len(mags), len(photon), len(sem))
    if n < MIN_EPOCHS:
        return None
    mags, photon, sem = mags[:n], photon[:n], sem[:n]
    ok = np.isfinite(mags) & np.isfinite(photon) & (photon > 0)
    if int(ok.sum()) < MIN_EPOCHS:
        return None
    return {
        "catalog_id": cid,
        "mags": mags[ok],
        "photon_rel": photon[ok],
        "sem_mag": sem[ok],
        "n_epochs": int(ok.sum()),
        "source_files": loo["source_file"].astype(str).tolist()[:n],
    }


def _sigma_mag_array(photon_rel: np.ndarray, sem_mag: np.ndarray, sigma_sys_mag: float) -> np.ndarray:
    n = len(photon_rel)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        out[i] = combine_production_err_mag(
            float(photon_rel[i]), float(sem_mag[i]), sigma_sys_mag=float(sigma_sys_mag),
        )
    return out


def _pooled_chi2_dof(stars: list[dict[str, Any]], sigma_sys_mag: float) -> float:
    chi2_sum = 0.0
    dof_sum = 0
    for st in stars:
        sig = _sigma_mag_array(st["photon_rel"], st["sem_mag"], sigma_sys_mag)
        c2, dof, c2d, _ = reduced_chi2_constant(st["mags"], sig)
        if math.isfinite(c2) and dof > 0:
            chi2_sum += c2
            dof_sum += dof
    return chi2_sum / dof_sum if dof_sum > 0 else float("nan")


def _root_find_sigma_sys(stars: list[dict[str, Any]], *, hi: float = 0.03) -> tuple[float, float]:
    lo, hi = 0.0, float(hi)
    f_lo = _pooled_chi2_dof(stars, lo) - 1.0
    f_hi = _pooled_chi2_dof(stars, hi) - 1.0
    if not math.isfinite(f_lo) or not math.isfinite(f_hi):
        return float("nan"), float("nan")
    if f_lo <= 0:
        return 0.0, float(f_lo + 1.0)
    if f_hi > 0:
        # extend search
        for mult in (2.0, 4.0, 8.0):
            hi2 = hi * mult
            if _pooled_chi2_dof(stars, hi2) - 1.0 <= 0:
                hi = hi2
                f_hi = _pooled_chi2_dof(stars, hi) - 1.0
                break
        else:
            return float("nan"), float("nan")
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        f_mid = _pooled_chi2_dof(stars, mid) - 1.0
        if abs(f_mid) < 1e-4:
            return mid, _pooled_chi2_dof(stars, mid)
        if f_mid > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), _pooled_chi2_dof(stars, 0.5 * (lo + hi))


def _bootstrap_floor(
    stars: list[dict[str, Any]],
    *,
    n_boot: int = BOOTSTRAP_DRAWS,
    seed: int = 0,
    alpha: float = 0.16,
) -> tuple[float | None, float | None]:
    if len(stars) < 3 or n_boot <= 0:
        return None, None
    rng = np.random.default_rng(seed)
    floors: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(stars), size=len(stars))
        sample = [stars[int(i)] for i in idx]
        fl, _ = _root_find_sigma_sys(sample)
        if math.isfinite(fl):
            floors.append(fl)
    if len(floors) < 10:
        return None, None
    arr = np.sort(np.asarray(floors, dtype=float))
    return float(np.quantile(arr, alpha)), float(np.quantile(arr, 1.0 - alpha))


def _split_half_validate(
    stars: list[dict[str, Any]],
    *,
    seed: int = SPLIT_SEED,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(stars))
    rng.shuffle(idx)
    half = len(stars) // 2
    a_idx, b_idx = idx[:half], idx[half:]
    a = [stars[int(i)] for i in a_idx]
    b = [stars[int(i)] for i in b_idx]
    out: dict[str, Any] = {}
    for label, fit_set, val_set in (
        ("A_fit_B_val", a, b),
        ("B_fit_A_val", b, a),
    ):
        fl, _ = _root_find_sigma_sys(fit_set)
        val_c2d = _pooled_chi2_dof(val_set, fl) if math.isfinite(fl) else float("nan")
        out[label] = {
            "sigma_sys_mag": fl,
            "sigma_sys_mmag": fl * 1000.0 if math.isfinite(fl) else None,
            "validate_chi2_dof": val_c2d,
        }
    return out


def _passes_saturation_gate(
    cid: str,
    *,
    phot_dir: Path,
    proc_dir: Path,
    setup: str,
    anchor: str,
    cfg: AppConfig,
) -> bool:
    loo = build_loo_differential_lc(cid, phot_dir=phot_dir, setup=setup, anchor_target=anchor, cfg=cfg)
    if loo is None or loo.empty:
        return False
    margin = saturation_margin_distribution(loo, proc_dir, cid)
    fill_p95 = margin.get("fill_p95")
    if fill_p95 is not None and math.isfinite(float(fill_p95)) and float(fill_p95) >= 0.85:
        return False
    return True


def _g60_epoch_audit(phot_dir: Path, proc_dir: Path, anchor: str, cfg: AppConfig) -> dict[str, Any]:
    """Identify g_60_4 epochs dropped as NaN (Part A.5)."""
    from check_star_kmag import resolve_proc_csv_dir  # noqa: PLC0415

    lc_dir = phot_dir / "lightcurves"
    lc_path = lc_dir / f"lightcurve_{anchor}.csv"
    if not lc_path.is_file():
        return {"status": "missing_anchor_lc"}
    lc = pd.read_csv(lc_path, low_memory=False)
    n_lc = int(len(lc))
    proc_files = sorted(proc_dir.glob("proc_*.csv")) if proc_dir.is_dir() else []
    dropped: list[dict[str, str]] = []
    for sf in lc.get("source_file", pd.Series(dtype=str)).astype(str).tolist():
        row = None
        try:
            from scripts.chi2_sigma_gate import load_proc_row_for_source  # noqa: PLC0415

            row = load_proc_row_for_source(proc_dir, sf, anchor)
        except Exception:  # noqa: BLE001
            row = None
        reason = "ok"
        if row is None:
            reason = "proc_row_missing"
        else:
            flux = float(pd.to_numeric(row.get("dao_flux"), errors="coerce"))
            if not math.isfinite(flux) or flux <= 0:
                reason = "nondetection_or_bad_flux"
            flag = str(row.get("flag", "")).strip().lower()
            if flag in ("saturated", "nondetection"):
                reason = f"flag_{flag}"
        mag = float(pd.to_numeric(lc.loc[lc["source_file"].astype(str) == sf, "delta_mag"].iloc[0], errors="coerce")) if (lc["source_file"].astype(str) == sf).any() else float("nan")
        if not math.isfinite(mag):
            dropped.append({"source_file": sf, "reason": reason if reason != "ok" else "nan_delta_mag"})
    return {
        "n_lc_epochs": n_lc,
        "n_proc_files": len(proc_files),
        "n_dropped_nan": len(dropped),
        "dropped_frames": dropped,
        "resolve_proc": str(proc_dir),
    }


def _plot_pzq(rig_label: str, pzq_rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(pzq_rows), 1)))
    for i, row in enumerate(pzq_rows):
        bins = row.get("pzq", {}).get("bins") or []
        if not bins:
            continue
        ns = [b["N"] for b in bins]
        sig = [b["sigma_N"] for b in bins]
        white = [b["sigma_white_expect"] for b in bins]
        ax.plot(ns, sig, "o-", color=colors[i % len(colors)], label=row.get("catalog_id", "")[-8:])
        ax.plot(ns, white, "--", color=colors[i % len(colors)], alpha=0.4)
    ax.set_xlabel("bin size N")
    ax.set_ylabel("sigma_N (mag)")
    ax.set_title("PZQ binned RMS - " + str(rig_label))
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def fit_rig_cohort(
    *,
    draft_id: int,
    setup: str,
    cfg: AppConfig,
    equipment_id: int,
    rig_label: str,
    wide_consistency: bool = False,
    pending: bool = False,
) -> dict[str, Any]:
    phot_dir = Path(cfg.archive_root) / "Drafts" / f"draft_{draft_id:06d}" / "platesolve" / setup / "photometry"
    meta = json.loads((phot_dir / "pipeline_meta.json").read_text(encoding="utf-8")) if (phot_dir / "pipeline_meta.json").is_file() else {}
    rig = resolve_rig_scintillation_params(draft_id=draft_id, setup=setup, cfg=cfg, pipeline_meta=meta)
    from check_star_kmag import resolve_proc_csv_dir  # noqa: PLC0415

    proc_dir = resolve_proc_csv_dir(phot_dir, setup)
    out: dict[str, Any] = {
        "draft_id": draft_id,
        "setup": setup,
        "equipment_id": equipment_id,
        "rig_label": rig_label,
        "pending": pending,
        "status": "pending" if pending else "fit",
    }
    if pending:
        out["note"] = "r_60_4 pending COMP-POOL-R / SPARSE-TRUST"
        return out
    if proc_dir is None or not phot_dir.is_dir():
        out["status"] = "missing_data"
        return out

    comp_all = pd.read_csv(
        phot_dir / "comparison_stars_per_target.csv",
        low_memory=False,
        dtype={"catalog_id": str, "target_catalog_id": str},
    ) if (phot_dir / "comparison_stars_per_target.csv").is_file() else pd.DataFrame()
    candidates = collect_comp_candidates(phot_dir, min_frames=MIN_EPOCHS)
    # exclude SS Cam sparse-path star
    candidates = candidates.loc[~candidates["catalog_id"].map(_norm_id).eq(_norm_id(SS_CAM_CID))]
    anchor = pick_anchor_target(phot_dir, comp_all)
    calibrators = pick_g_coverage(candidates, aim=12)
    out["anchor_target"] = anchor
    out["n_candidates"] = int(len(candidates))

    if calibrators.empty or anchor is None:
        out["status"] = "no_cohort"
        return out

    gain = float(meta.get("gain") or 1.0)
    rn = float(meta.get("read_noise") or 10.0)

    stars: list[dict[str, Any]] = []
    for _, row in calibrators.iterrows():
        cid = _norm_id(row["catalog_id"])
        built = _build_star_arrays(
            cid, phot_dir=phot_dir, setup=setup, anchor=anchor, proc_dir=proc_dir,
            cfg=cfg, gain=gain, read_noise=rn,
        )
        if built:
            stars.append(built)
    stars = [
        st for st in stars
        if _passes_saturation_gate(
            st["catalog_id"],
            phot_dir=phot_dir,
            proc_dir=proc_dir,
            setup=setup,
            anchor=anchor,
            cfg=cfg,
        )
    ]
    out["n_stars"] = len(stars)
    if len(stars) < 3:
        out["status"] = "insufficient_stars"
        return out

    if setup == "g_60_4" and draft_id == 426:
        out["g60_epoch_audit"] = _g60_epoch_audit(phot_dir, proc_dir, anchor, cfg)

    sigma_sys, chi2_at_fit = _root_find_sigma_sys(stars)
    ci_lo, ci_hi = _bootstrap_floor(stars, n_boot=BOOTSTRAP_DRAWS, seed=int(draft_id))
    unstable = False
    if ci_lo is not None and ci_hi is not None and sigma_sys > 0:
        if ci_hi > 2.0 * sigma_sys or ci_lo < 0.5 * sigma_sys:
            unstable = True

    split = _split_half_validate(stars)

    pzq_rows: list[dict[str, Any]] = []
    sigma_r_vals: list[float] = []
    for st in stars:
        pzq = pzq_fit_sigma_r(st["mags"])
        pzq_rows.append({"catalog_id": st["catalog_id"], "pzq": pzq})
        sr = pzq.get("sigma_r")
        if sr is not None and math.isfinite(float(sr)):
            sigma_r_vals.append(float(sr))

    out.update({
        "sigma_sys_mag": None if unstable else sigma_sys,
        "sigma_sys_mmag": None if unstable or not math.isfinite(sigma_sys) else sigma_sys * 1000.0,
        "chi2_dof_at_fit": chi2_at_fit,
        "ci_lo_mag": ci_lo,
        "ci_hi_mag": ci_hi,
        "ci_lo_mmag": ci_lo * 1000.0 if ci_lo is not None else None,
        "ci_hi_mmag": ci_hi * 1000.0 if ci_hi is not None else None,
        "unstable": unstable,
        "split_half": split,
        "pzq_median_sigma_r": float(np.median(sigma_r_vals)) if sigma_r_vals else None,
        "pzq_per_star": pzq_rows,
        "stars": [{"catalog_id": s["catalog_id"], "n_epochs": s["n_epochs"]} for s in stars],
    })

    if wide_consistency and not unstable and math.isfinite(sigma_sys):
        mmag = sigma_sys * 1000.0
        lo, hi = WIDE_RIG_PRIOR_CI_MMAG
        out["wide_consistency"] = {
            "fitted_mmag": mmag,
            "prior_mmag": WIDE_RIG_PRIOR_MMAG,
            "prior_ci_mmag": list(WIDE_RIG_PRIOR_CI_MMAG),
            "consistent": bool(lo <= mmag <= hi),
        }
        if not out["wide_consistency"]["consistent"]:
            out["status"] = "STOP_wide_inconsistent"
            out["sigma_sys_mag"] = None
    else:
        out["status"] = "unstable" if unstable else "ok"

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit per-rig sigma_sys floor")
    parser.add_argument("--out-dir", type=Path, default=_ROOT / "tmp" / "sigma_floor")
    args = parser.parse_args()
    cfg = AppConfig()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cohorts = [
        (424, "NoFilter_60_2", 1, "wide_Carl-Zeiss", True, False),
        (426, "g_60_4", 4, "Newton_g", False, False),
        (426, "i_70_4", 4, "Newton_i", False, False),
        (426, "r_60_4", 4, "Newton_r", False, True),
    ]
    results: list[dict[str, Any]] = []
    stop = False
    for draft_id, setup, eq_id, label, wide, pending in cohorts:
        res = fit_rig_cohort(
            draft_id=draft_id, setup=setup, cfg=cfg, equipment_id=eq_id,
            rig_label=label, wide_consistency=wide, pending=pending,
        )
        results.append(res)
        if res.get("status") == "STOP_wide_inconsistent":
            stop = True
        if not pending and res.get("pzq_per_star"):
            _plot_pzq(label, res["pzq_per_star"], out_dir / f"pzq_{label}.png")

    # Pool setups sharing equipment_id (Newton g+i -> one floor per camera).
    by_eq: dict[int, list[dict[str, Any]]] = {}
    for r in results:
        if r.get("pending") or r.get("status") not in ("ok", "unstable"):
            continue
        stars_all: list[dict[str, Any]] = []
        for s in r.get("stars") or []:
            cid = s["catalog_id"]
            draft_id = int(r["draft_id"])
            setup = str(r["setup"])
            phot_dir = Path(cfg.archive_root) / "Drafts" / f"draft_{draft_id:06d}" / "platesolve" / setup / "photometry"
            from check_star_kmag import resolve_proc_csv_dir  # noqa: PLC0415

            proc_dir = resolve_proc_csv_dir(phot_dir, setup)
            anchor = r.get("anchor_target")
            meta = json.loads((phot_dir / "pipeline_meta.json").read_text(encoding="utf-8")) if (phot_dir / "pipeline_meta.json").is_file() else {}
            gain = float(meta.get("gain") or 1.0)
            rn = float(meta.get("read_noise") or 10.0)
            if anchor and proc_dir:
                built = _build_star_arrays(
                    cid, phot_dir=phot_dir, setup=setup, anchor=str(anchor), proc_dir=proc_dir,
                    cfg=cfg, gain=gain, read_noise=rn,
                )
                if built:
                    stars_all.append(built)
        eq = int(r["equipment_id"])
        by_eq.setdefault(eq, []).extend(stars_all)

    equipment_fits: dict[str, Any] = {}
    config_map: dict[str, float] = {}
    for eq_id, stars in by_eq.items():
        if len(stars) < 3:
            continue
        sigma_sys, chi2_at = _root_find_sigma_sys(stars)
        ci_lo, ci_hi = _bootstrap_floor(stars, n_boot=BOOTSTRAP_DRAWS, seed=eq_id)
        unstable = False
        if ci_lo is not None and ci_hi is not None and math.isfinite(sigma_sys) and sigma_sys > 0:
            if ci_hi > 2.0 * sigma_sys or ci_lo < 0.5 * sigma_sys:
                unstable = True
        split = _split_half_validate(stars, seed=SPLIT_SEED + eq_id)
        entry = {
            "equipment_id": eq_id,
            "n_stars_pooled": len(stars),
            "sigma_sys_mag": None if unstable else sigma_sys,
            "sigma_sys_mmag": None if unstable or not math.isfinite(sigma_sys) else sigma_sys * 1000.0,
            "chi2_dof_at_fit": chi2_at,
            "ci_lo_mmag": ci_lo * 1000.0 if ci_lo is not None else None,
            "ci_hi_mmag": ci_hi * 1000.0 if ci_hi is not None else None,
            "unstable": unstable,
            "split_half": split,
        }
        if eq_id == 1:
            mmag = sigma_sys * 1000.0 if math.isfinite(sigma_sys) else float("nan")
            entry["wide_consistency"] = {
                "fitted_mmag": mmag,
                "prior_mmag": WIDE_RIG_PRIOR_MMAG,
                "prior_ci_mmag": list(WIDE_RIG_PRIOR_CI_MMAG),
                "consistent": bool(WIDE_RIG_PRIOR_CI_MMAG[0] <= mmag <= WIDE_RIG_PRIOR_CI_MMAG[1]),
            }
            if not unstable and not entry["wide_consistency"]["consistent"]:
                stop = True
                entry["status"] = "STOP_wide_inconsistent"
        equipment_fits[str(eq_id)] = entry
        if not unstable and math.isfinite(sigma_sys) and sigma_sys > 0:
            config_map[str(eq_id)] = float(sigma_sys)

    payload = _stamp({
        "stop": stop,
        "cohorts": results,
        "equipment_fits": equipment_fits,
        "config_sigma_sys_mag": config_map,
    })
    write_summary_json(payload, out_dir / "sigma_floor_fit.json")
    print(json.dumps(payload, indent=2))
    return 2 if stop else 0


if __name__ == "__main__":
    raise SystemExit(main())
