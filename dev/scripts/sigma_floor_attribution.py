#!/usr/bin/env python3
"""Sigma-floor attribution diagnostics: k2, PRNU phase, airmass, time (sandbox)."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from check_star_kmag import (  # noqa: E402
    build_aligned_comp_inst,
    comp_ensemble_maps,
    resolve_proc_csv_dir,
)
from config import AppConfig  # noqa: E402
from gaia_catalog_id import normalize_gaia_source_id  # noqa: E402
from photometry_core import check_comparison_stability, parse_comp_quality_json_map  # noqa: E402
from scripts.chi2_sigma_gate import (  # noqa: E402
    CalibratorEnsembleInput,
    fit_f_resid_sigma_floor_ensemble,
    load_proc_row_for_source,
    write_summary_json,
)
from scripts.select_constant_calibrators import (  # noqa: E402
    build_loo_differential_lc,
    collect_comp_candidates,
    compute_loo_production_ensemble_scatter,
    pick_anchor_target,
    pick_g_coverage,
)
from sigma_budget import resolve_rig_scintillation_params  # noqa: E402


def _norm_id(x: object) -> str:
    return str(normalize_gaia_source_id(x) or "").strip()


def residuals_from_delta_mag(mags: np.ndarray) -> np.ndarray:
    m = np.asarray(mags, dtype=np.float64)
    ok = np.isfinite(m)
    if int(ok.sum()) < 2:
        return np.full_like(m, np.nan)
    ref = float(np.mean(m[ok]))
    return m - ref


def residual_rms_mag(residuals: np.ndarray) -> float:
    r = np.asarray(residuals, dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return float("nan")
    return float(np.std(r, ddof=1))


def variance_explained(y: np.ndarray, y_hat: np.ndarray) -> float:
    yv = np.asarray(y, dtype=np.float64)
    yh = np.asarray(y_hat, dtype=np.float64)
    ok = np.isfinite(yv) & np.isfinite(yh)
    if int(ok.sum()) < 3:
        return float("nan")
    y0 = yv[ok]
    h0 = yh[ok]
    ss_tot = float(np.sum((y0 - np.mean(y0)) ** 2))
    if ss_tot <= 0:
        return 0.0
    ss_res = float(np.sum((y0 - h0) ** 2))
    return float(max(0.0, 1.0 - ss_res / ss_tot))


def ols_through_origin(y: np.ndarray, x: np.ndarray) -> tuple[float, np.ndarray]:
    yv = np.asarray(y, dtype=np.float64)
    xv = np.asarray(x, dtype=np.float64)
    ok = np.isfinite(yv) & np.isfinite(xv)
    if int(ok.sum()) < 2:
        return float("nan"), np.full_like(yv, np.nan)
    x0 = xv[ok]
    y0 = yv[ok]
    denom = float(np.dot(x0, x0))
    slope = float(np.dot(x0, y0) / denom) if denom > 0 else float("nan")
    yhat = np.full_like(yv, np.nan)
    yhat[ok] = slope * x0
    return slope, yhat


def ols_with_intercept(y: np.ndarray, x: np.ndarray) -> tuple[float, float, np.ndarray]:
    yv = np.asarray(y, dtype=np.float64)
    xv = np.asarray(x, dtype=np.float64)
    ok = np.isfinite(yv) & np.isfinite(xv)
    if int(ok.sum()) < 3:
        return float("nan"), float("nan"), np.full_like(yv, np.nan)
    x0 = xv[ok]
    y0 = yv[ok]
    X = np.column_stack([np.ones(len(x0)), x0])
    beta, *_ = np.linalg.lstsq(X, y0, rcond=None)
    intercept = float(beta[0])
    slope = float(beta[1])
    yhat = np.full_like(yv, np.nan)
    yhat[ok] = intercept + slope * x0
    return intercept, slope, yhat


def ols_multivariate(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    yv = np.asarray(y, dtype=np.float64)
    Xv = np.asarray(X, dtype=np.float64)
    if Xv.ndim == 1:
        Xv = Xv[:, np.newaxis]
    ok = np.isfinite(yv) & np.all(np.isfinite(Xv), axis=1)
    if int(ok.sum()) < Xv.shape[1] + 1:
        return np.full(Xv.shape[1], np.nan), np.full_like(yv, np.nan)
    X0 = Xv[ok]
    y0 = yv[ok]
    Xd = np.column_stack([np.ones(len(y0)), X0])
    beta, *_ = np.linalg.lstsq(Xd, y0, rcond=None)
    yhat = np.full_like(yv, np.nan)
    yhat[ok] = Xd @ beta
    return beta[1:], yhat


def bootstrap_slope_ci_origin(
    y: np.ndarray,
    x: np.ndarray,
    *,
    n_boot: int = 400,
    seed: int = 0,
    alpha: float = 0.16,
) -> tuple[float | None, float | None]:
    yv = np.asarray(y, dtype=np.float64)
    xv = np.asarray(x, dtype=np.float64)
    ok = np.isfinite(yv) & np.isfinite(xv)
    y0 = yv[ok]
    x0 = xv[ok]
    n = int(y0.size)
    if n < 5 or n_boot <= 0:
        return None, None
    rng = np.random.default_rng(seed)
    slopes: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        s, _ = ols_through_origin(y0[idx], x0[idx])
        if math.isfinite(s):
            slopes.append(s)
    if len(slopes) < 10:
        return None, None
    arr = np.sort(np.asarray(slopes, dtype=float))
    return float(np.quantile(arr, alpha)), float(np.quantile(arr, 1.0 - alpha))


def phase_fraction(coord: np.ndarray) -> np.ndarray:
    c = np.asarray(coord, dtype=np.float64)
    out = np.full_like(c, np.nan)
    ok = np.isfinite(c)
    out[ok] = np.mod(c[ok], 1.0)
    return out


def phase_harmonic_design(frac_x: np.ndarray, frac_y: np.ndarray) -> np.ndarray:
    fx = np.asarray(frac_x, dtype=np.float64)
    fy = np.asarray(frac_y, dtype=np.float64)
    ang_x = 2.0 * math.pi * fx
    ang_y = 2.0 * math.pi * fy
    return np.column_stack([np.sin(ang_x), np.cos(ang_x), np.sin(ang_y), np.cos(ang_y)])


def quadrant_phase_rms(
    residuals: np.ndarray,
    frac_x: np.ndarray,
    frac_y: np.ndarray,
) -> dict[str, float]:
    r = np.asarray(residuals, dtype=np.float64)
    fx = np.asarray(frac_x, dtype=np.float64)
    fy = np.asarray(frac_y, dtype=np.float64)
    out: dict[str, float] = {}
    for qx, qy, label in (
        (0, 0, "Q00"),
        (0, 1, "Q01"),
        (1, 0, "Q10"),
        (1, 1, "Q11"),
    ):
        x_ok = (fx < 0.5) if qx == 0 else (fx >= 0.5)
        y_ok = (fy < 0.5) if qy == 0 else (fy >= 0.5)
        mask = np.isfinite(r) & np.isfinite(fx) & np.isfinite(fy) & x_ok & y_ok
        sub = r[mask]
        out[f"rms_{label}"] = residual_rms_mag(sub) if sub.size >= 2 else float("nan")
    return out


def flux_weighted_mean(values: list[float], weights: list[float]) -> float:
    if not values or not weights:
        return float("nan")
    w = np.asarray(weights, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(w) & np.isfinite(v) & (w > 0)
    if int(ok.sum()) == 0:
        return float("nan")
    return float(np.sum(w[ok] * v[ok]) / np.sum(w[ok]))


def refit_sigma_floor_mm(
    ensemble_inputs: list[CalibratorEnsembleInput],
    *,
    include_ensemble: bool = True,
    f_resid: float = 0.0,
    n_boot: int = 120,
) -> float:
    joint = fit_f_resid_sigma_floor_ensemble(
        ensemble_inputs,
        include_ensemble=include_ensemble,
        n_boot=n_boot,
        seed=1,
    )
    return float(joint.sigma_floor_mag * 1000.0)


@dataclass
class CalibratorFramePack:
    catalog_id: str
    mag_g: float | None
    residuals: np.ndarray
    airmass: np.ndarray
    bjd: np.ndarray
    delta_color: np.ndarray
    frac_x: np.ndarray
    frac_y: np.ndarray
    sh: np.ndarray
    ss: np.ndarray
    sem: np.ndarray


def _bp_rp_map(comp_df: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for _, row in comp_df.iterrows():
        cid = _norm_id(row.get("catalog_id"))
        if not cid:
            continue
        out[cid] = float(pd.to_numeric(row.get("bp_rp"), errors="coerce"))
    return out


def _frame_comp_fluxes(
    proc_dir: Path,
    source_file: str,
    comp_ids: list[str],
) -> dict[str, float]:
    path = proc_dir / str(source_file).strip()
    if not path.is_file():
        return {}
    try:
        df = pd.read_csv(path, low_memory=False, dtype={"catalog_id": str})
    except Exception:  # noqa: BLE001
        return {}
    id_col = "catalog_id" if "catalog_id" in df.columns else "name"
    df["_nid"] = df[id_col].map(lambda x: _norm_id(x))
    out: dict[str, float] = {}
    for cid in comp_ids:
        sub = df.loc[df["_nid"] == cid]
        if sub.empty:
            continue
        flux = float(pd.to_numeric(sub.iloc[0].get("dao_flux"), errors="coerce"))
        if math.isfinite(flux) and flux > 0:
            out[cid] = flux
    return out


def build_calibrator_frame_packs(
    draft_id: int,
    setup: str,
    *,
    cfg: AppConfig,
    min_frames: int = 120,
) -> tuple[list[CalibratorFramePack], dict[str, Any]]:
    phot_dir = Path(cfg.archive_root) / "Drafts" / f"draft_{draft_id:06d}" / "platesolve" / setup / "photometry"
    meta = (
        json.loads((phot_dir / "pipeline_meta.json").read_text(encoding="utf-8"))
        if (phot_dir / "pipeline_meta.json").is_file()
        else {}
    )
    rig = resolve_rig_scintillation_params(draft_id=draft_id, setup=setup, cfg=cfg, pipeline_meta=meta)
    proc_dir = resolve_proc_csv_dir(phot_dir, setup)
    comp_all = pd.read_csv(
        phot_dir / "comparison_stars_per_target.csv",
        low_memory=False,
        dtype={"catalog_id": str, "target_catalog_id": str},
    )
    anchor = pick_anchor_target(phot_dir, comp_all)
    calibrators = pick_g_coverage(collect_comp_candidates(phot_dir, min_frames=min_frames), aim=8)
    if calibrators.empty or anchor is None or proc_dir is None:
        return [], {"draft_id": draft_id, "setup": setup, "anchor_target": anchor, "rig": rig.to_dict()}

    comp_df = comp_all.loc[comp_all["target_catalog_id"].map(_norm_id) == _norm_id(anchor)]
    bp_rp_anchor = _bp_rp_map(comp_df)
    bp_rp_field = _bp_rp_map(comp_all)
    comp_ids = [_norm_id(c) for c in comp_df["catalog_id"].tolist() if _norm_id(c)]

    packs: list[CalibratorFramePack] = []
    for _, row in calibrators.iterrows():
        cid = _norm_id(row["catalog_id"])
        loo = build_loo_differential_lc(cid, phot_dir=phot_dir, setup=setup, anchor_target=anchor, cfg=cfg)
        if loo is None:
            continue
        prod_scatter = compute_loo_production_ensemble_scatter(
            cid, phot_dir=phot_dir, setup=setup, anchor_target=anchor, cfg=cfg,
        )
        from scripts.chi2_sigma_gate import sigma_arrays_from_lc_and_proc  # noqa: PLC0415

        _, _, sh, ss, sem_meta = sigma_arrays_from_lc_and_proc(
            loo, proc_dir, cid, rig_params=rig, production_ensemble_scatter=prod_scatter,
        )
        sem = np.asarray(sem_meta["ensemble_sem_primary"], dtype=np.float64)
        mags = pd.to_numeric(loo["delta_mag"], errors="coerce").to_numpy(dtype=np.float64)
        resid = residuals_from_delta_mag(mags)
        airmass = pd.to_numeric(loo["airmass"], errors="coerce").to_numpy(dtype=np.float64)
        bjd = pd.to_numeric(loo["bjd"], errors="coerce").to_numpy(dtype=np.float64)
        cal_bp = bp_rp_field.get(cid, bp_rp_anchor.get(cid, float("nan")))

        comp_lc = build_aligned_comp_inst(
            proc_dir, comp_ids, loo["source_file"].astype(str).tolist(), cfg, "aperture",
        )
        other_ids = [c for c in comp_ids if c != cid]

        delta_color = np.full(len(loo), np.nan, dtype=np.float64)
        frac_x = np.full(len(loo), np.nan, dtype=np.float64)
        frac_y = np.full(len(loo), np.nan, dtype=np.float64)
        for i, sf in enumerate(loo["source_file"].astype(str).tolist()):
            flux_map = _frame_comp_fluxes(proc_dir, sf, other_ids)
            vals = [bp_rp_anchor.get(c, float("nan")) for c in other_ids if c in flux_map]
            wts = [flux_map[c] for c in other_ids if c in flux_map]
            ens_bp = flux_weighted_mean(vals, wts)
            if math.isfinite(cal_bp) and math.isfinite(ens_bp):
                delta_color[i] = cal_bp - ens_bp
            prow = load_proc_row_for_source(proc_dir, sf, cid)
            if prow is not None:
                x = float(pd.to_numeric(prow.get("x"), errors="coerce"))
                y = float(pd.to_numeric(prow.get("y"), errors="coerce"))
                if math.isfinite(x):
                    frac_x[i] = float(np.mod(x, 1.0))
                if math.isfinite(y):
                    frac_y[i] = float(np.mod(y, 1.0))

        packs.append(
            CalibratorFramePack(
                catalog_id=cid,
                mag_g=float(row["mag_g"]) if math.isfinite(float(row["mag_g"])) else None,
                residuals=resid,
                airmass=airmass,
                bjd=bjd,
                delta_color=delta_color,
                frac_x=frac_x,
                frac_y=frac_y,
                sh=sh,
                ss=ss,
                sem=sem,
            )
        )

    meta_out = {
        "draft_id": draft_id,
        "setup": setup,
        "anchor_target": anchor,
        "rig": rig.to_dict(),
        "n_calibrators": len(packs),
    }
    return packs, meta_out


def _ensemble_inputs_from_residuals(
    packs: list[CalibratorFramePack],
    residuals_list: list[np.ndarray],
) -> list[CalibratorEnsembleInput]:
    out: list[CalibratorEnsembleInput] = []
    for pack, resid in zip(packs, residuals_list):
        out.append((resid, pack.sh, pack.ss, pack.sem))
    return out


def run_attribution(
    draft_id: int,
    setup: str,
    *,
    cfg: AppConfig,
    out_dir: Path,
) -> dict[str, Any]:
    packs, meta = build_calibrator_frame_packs(draft_id, setup, cfg=cfg)
    if not packs:
        return {"error": "no calibrators", **meta}

    y_all: list[float] = []
    x_k2_all: list[float] = []
    x_air_all: list[float] = []
    t_all: list[float] = []
    phase_rows: list[np.ndarray] = []
    per_cal: dict[str, dict[str, Any]] = {}
    floor_before = refit_sigma_floor_mm(_ensemble_inputs_from_residuals(packs, [p.residuals for p in packs]))

    # k2 pooled
    for p in packs:
        k2_pred = p.airmass * p.delta_color
        for i in range(len(p.residuals)):
            if np.isfinite(p.residuals[i]) and np.isfinite(k2_pred[i]):
                y_all.append(float(p.residuals[i]))
                x_k2_all.append(float(k2_pred[i]))
    yv = np.asarray(y_all, dtype=np.float64)
    xk = np.asarray(x_k2_all, dtype=np.float64)
    k2_slope, k2_hat_all = ols_through_origin(yv, xk)
    k2_ci_lo, k2_ci_hi = bootstrap_slope_ci_origin(yv, xk)

    # phase pooled
    y_phase: list[float] = []
    X_phase: list[np.ndarray] = []
    for p in packs:
        Xp = phase_harmonic_design(p.frac_x, p.frac_y)
        for i in range(len(p.residuals)):
            if np.isfinite(p.residuals[i]) and np.all(np.isfinite(Xp[i])):
                y_phase.append(float(p.residuals[i]))
                X_phase.append(Xp[i])
    yp = np.asarray(y_phase, dtype=np.float64)
    Xpm = np.asarray(X_phase, dtype=np.float64) if X_phase else np.empty((0, 4))
    phase_hat_pool = np.array([], dtype=np.float64)
    if yp.size and Xpm.size:
        _, phase_hat_pool = ols_multivariate(yp, Xpm)

    # controls pooled
    ya_list: list[float] = []
    xa_list: list[float] = []
    for p in packs:
        ok = np.isfinite(p.residuals) & np.isfinite(p.airmass)
        ya_list.extend(p.residuals[ok].tolist())
        xa_list.extend(p.airmass[ok].tolist())
    ya = np.asarray(ya_list, dtype=np.float64)
    xa = np.asarray(xa_list, dtype=np.float64)
    _, x_slope, x_hat_pool = ols_with_intercept(ya, xa) if ya.size >= 3 else (np.nan, np.nan, np.array([]))

    tb_list = []
    yb_list = []
    for p in packs:
        ok = np.isfinite(p.residuals) & np.isfinite(p.bjd)
        if int(ok.sum()) < 2:
            continue
        t0 = float(np.median(p.bjd[ok]))
        tb_list.extend((p.bjd[ok] - t0).tolist())
        yb_list.extend(p.residuals[ok].tolist())
    yb = np.asarray(yb_list, dtype=np.float64)
    xb = np.asarray(tb_list, dtype=np.float64)
    _, t_slope, t_hat_pool = ols_with_intercept(yb, xb) if yb.size >= 3 else (np.nan, np.nan, np.array([]))

    rows: list[dict[str, Any]] = []
    corrected: dict[str, list[np.ndarray]] = {
        "k2_signature": [],
        "phase_signature": [],
        "x_linear": [],
        "time_linear": [],
    }

    for p in packs:
        k2_pred = p.airmass * p.delta_color
        _, k2_hat = ols_through_origin(p.residuals, k2_pred)
        Xp = phase_harmonic_design(p.frac_x, p.frac_y)
        _, phase_hat = ols_multivariate(p.residuals, Xp)
        _, _, x_hat = ols_with_intercept(p.residuals, p.airmass)
        t0 = float(np.nanmedian(p.bjd[np.isfinite(p.bjd)])) if np.isfinite(p.bjd).any() else float("nan")
        _, _, t_hat = ols_with_intercept(p.residuals, p.bjd - t0)

        resid_k2 = p.residuals - k2_hat
        resid_phase = p.residuals - phase_hat
        resid_x = p.residuals - x_hat
        resid_t = p.residuals - t_hat

        corrected["k2_signature"].append(resid_k2)
        corrected["phase_signature"].append(resid_phase)
        corrected["x_linear"].append(resid_x)
        corrected["time_linear"].append(resid_t)

        per_cal[p.catalog_id] = {
            "mag_g": p.mag_g,
            "variance_explained_k2": variance_explained(p.residuals, k2_hat),
            "variance_explained_phase": variance_explained(p.residuals, phase_hat),
            "variance_explained_x": variance_explained(p.residuals, x_hat),
            "variance_explained_time": variance_explained(p.residuals, t_hat),
            "quadrant_rms": quadrant_phase_rms(p.residuals, p.frac_x, p.frac_y),
            "delta_color_median": float(np.nanmedian(p.delta_color)),
        }

    pooled_r2 = {
        "k2_signature": variance_explained(yv, k2_slope * xk) if yv.size else float("nan"),
        "phase_signature": variance_explained(yp, phase_hat_pool) if yp.size else float("nan"),
        "x_linear": variance_explained(ya, x_hat_pool) if ya.size and x_hat_pool.size else float("nan"),
        "time_linear": variance_explained(yb, t_hat_pool) if yb.size and t_hat_pool.size else float("nan"),
    }

    for label, key in (
        ("k2_signature", "k2_signature"),
        ("phase_signature", "phase_signature"),
        ("X_linear", "x_linear"),
        ("time_linear", "time_linear"),
    ):
        floor_after = refit_sigma_floor_mm(
            _ensemble_inputs_from_residuals(packs, corrected[key]),
            include_ensemble=True,
            f_resid=0.0,
        )
        rows.append(
            {
                "candidate": label,
                "variance_explained_pooled": pooled_r2[key],
                "floor_before_mmag": floor_before,
                "floor_after_mmag": floor_after,
                "floor_delta_mmag": floor_before - floor_after,
            }
        )

    # plot
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(packs), 1)))
    for i, p in enumerate(packs):
        xplot = p.airmass * p.delta_color
        ok = np.isfinite(p.residuals) & np.isfinite(xplot)
        ax.scatter(
            xplot[ok],
            p.residuals[ok],
            s=12,
            alpha=0.55,
            color=colors[i % len(colors)],
            label=f"G{p.mag_g:.1f}" if p.mag_g is not None else p.catalog_id[:8],
        )
    if yv.size and math.isfinite(k2_slope):
        xs = np.linspace(float(np.min(xk)), float(np.max(xk)), 50)
        ax.plot(xs, k2_slope * xs, "k--", linewidth=1, label=f"k2={k2_slope:.4f}")
    ax.set_xlabel("X * Delta_color")
    ax.set_ylabel("residual (mag)")
    ax.set_title(f"Floor attribution draft_{draft_id:06d}/{setup}")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    plot_path = out_dir / f"floor_attribution_draft{draft_id:06d}_{setup}.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)

    payload = {
        **meta,
        "floor_before_mmag": floor_before,
        "k2_effective": k2_slope,
        "k2_effective_ci_lo": k2_ci_lo,
        "k2_effective_ci_hi": k2_ci_hi,
        "attribution_rows": rows,
        "per_calibrator": per_cal,
        "plot": str(plot_path),
    }
    json_path = write_summary_json(payload, out_dir / f"floor_attribution_draft{draft_id:06d}_{setup}.json")
    payload["json"] = json_path
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Sigma floor attribution (sandbox)")
    ap.add_argument("--draft", type=int, default=424)
    ap.add_argument("--setup", default="NoFilter_60_2")
    ap.add_argument("--out-dir", type=Path, default=Path("tmp/sigma_budget"))
    args = ap.parse_args()
    cfg = AppConfig()
    out = run_attribution(args.draft, args.setup, cfg=cfg, out_dir=Path(args.out_dir))
    print(out.get("json", out))


if __name__ == "__main__":
    main()
