#!/usr/bin/env python3
"""Compute B/V/R colour terms from comp pool on draft_000375 and dump analysis CSVs."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import AppConfig  # noqa: E402
from gaia_catalog_id import normalize_gaia_source_id  # noqa: E402
from photometry_core import (  # noqa: E402
    _check_color_term_extrapolation,
    _color_term_cat_inst_scatter_pair,
    apply_color_term,
    fit_color_term_c1,
    should_apply_color_term,
)

DRAFT_ID = 375
SETUPS = ("B_20_2", "V_20_2", "R_20_2")
FILTER_LABEL = {"B_20_2": "B", "V_20_2": "V", "R_20_2": "Rc"}
MIN_COMP_CT = 7
MAX_STDERR_RATIO = 0.5
SUMMARY_CSV = _ROOT / "ct_summary_chiandh_BVR.csv"
TARGETS_CSV = _ROOT / "ct_targets_chiandh_BVR.csv"


def _norm_cid(val: Any) -> str:
    return str(normalize_gaia_source_id(val) or "").strip()


def _flux_to_mag(flux: float) -> float:
    if not math.isfinite(flux) or flux <= 0:
        return float("nan")
    return float(-2.5 * math.log10(flux))


def _build_comp_mag_inst(proc_dir: Path, comp_ids: list[str]) -> dict[str, np.ndarray]:
    proc_files = sorted(proc_dir.glob("proc_*.csv"))
    if not proc_files:
        proc_files = sorted(proc_dir.glob("*.csv"))
    n = len(proc_files)
    out: dict[str, np.ndarray] = {cid: np.full(n, float("nan"), dtype=np.float64) for cid in comp_ids}
    id_set = set(comp_ids)
    for i, path in enumerate(proc_files):
        try:
            df = pd.read_csv(path, low_memory=False, dtype={"catalog_id": str})
        except Exception:  # noqa: BLE001
            continue
        if "catalog_id" not in df.columns:
            continue
        df = df.copy()
        df["_nid"] = df["catalog_id"].map(_norm_cid)
        sub = df[df["_nid"].isin(id_set)]
        flux_col = "dao_flux" if "dao_flux" in sub.columns else "flux"
        if flux_col not in sub.columns:
            continue
        for _, row in sub.iterrows():
            cid = str(row["_nid"])
            if cid not in out:
                continue
            flux = float(pd.to_numeric(row.get(flux_col), errors="coerce"))
            out[cid][i] = _flux_to_mag(flux)
    return out


def _comp_quality_from_df(comp_df: pd.DataFrame) -> dict[str, dict]:
    q: dict[str, dict] = {}
    for _, row in comp_df.iterrows():
        cid = _norm_cid(row.get("catalog_id", row.get("name")))
        if not cid:
            continue
        usable = True
        for col in ("is_usable", "photometry_ok"):
            if col in row.index:
                v = row.get(col)
                if str(v).strip().lower() in ("false", "0", "no"):
                    usable = False
        q[cid] = {"quality": "good" if usable else "excluded"}
    return q


def _comp_catalog_mag(comp_df: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for _, row in comp_df.iterrows():
        cid = _norm_cid(row.get("catalog_id", row.get("name")))
        if not cid:
            continue
        for col in ("phot_g_mean_mag", "catalog_mag", "mag"):
            v = pd.to_numeric(row.get(col), errors="coerce")
            if math.isfinite(float(v)):
                out[cid] = float(v)
                break
    return out


def _comp_bp_rp(comp_df: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for _, row in comp_df.iterrows():
        cid = _norm_cid(row.get("catalog_id", row.get("name")))
        if not cid:
            continue
        v = pd.to_numeric(row.get("bp_rp"), errors="coerce")
        if math.isfinite(float(v)):
            out[cid] = float(v)
    return out


def _fit_resid_rms(
    comp_mag_inst: dict[str, np.ndarray],
    comp_catalog_mag: dict[str, float],
    comp_bp_rp: dict[str, float],
    comp_quality: dict[str, dict],
    c1: float,
    *,
    min_comp: int = 5,
    sigma_clip_sigma: float = 3.0,
) -> float:
    """RMS of sigma-clipped comp residuals after c1 fit (matches fit_color_term_c1 logic)."""
    from photometry_core import _mad_sigma, _safe_polyfit  # noqa: PLC0415

    usable = [cid for cid, q in comp_quality.items() if q.get("quality") in ("good", "suspect")]
    ys: list[float] = []
    bp_vals: list[float] = []
    for cid in usable:
        bp = float(comp_bp_rp.get(cid, float("nan")))
        if not math.isfinite(bp) or cid not in comp_mag_inst:
            continue
        inst = np.asarray(comp_mag_inst[cid], dtype=np.float64)
        finite = inst[np.isfinite(inst)]
        if finite.size < min_comp:
            continue
        cat = float(comp_catalog_mag.get(cid, float("nan")))
        if not math.isfinite(cat):
            continue
        y = float(np.nanmedian(cat - finite))
        if not math.isfinite(y):
            continue
        bp_vals.append(bp)
        ys.append(y)
    if len(ys) < min_comp:
        return float("nan")
    bp_med = float(np.median(np.asarray(bp_vals, dtype=np.float64)))
    xs = np.asarray([b - bp_med for b in bp_vals], dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    p0 = _safe_polyfit(xs, y, 1)
    if p0 is None:
        return float("nan")
    c1_init, zp_init = float(p0[0]), float(p0[1])
    resid = y - (c1_init * xs + zp_init)
    sig = _mad_sigma(resid)
    if not math.isfinite(sig) or sig <= 0:
        mask = np.ones_like(resid, dtype=bool)
    else:
        mask = np.abs(resid) <= float(sigma_clip_sigma) * float(sig)
    x_cl, y_cl = xs[mask], y[mask]
    if x_cl.size < 2:
        return float("nan")
    resid_final = y_cl - (float(c1) * x_cl + float(np.median(y_cl - float(c1) * x_cl)))
    return float(np.sqrt(np.mean(np.square(resid_final))))


def _target_scatter_pair(
    lc_path: Path,
    *,
    cat_mag: float,
) -> tuple[float, float]:
    if not lc_path.is_file() or not math.isfinite(cat_mag):
        return float("nan"), float("nan")
    try:
        lc = pd.read_csv(lc_path, low_memory=False)
    except Exception:  # noqa: BLE001
        return float("nan"), float("nan")
    pre_col = "mag_calib" if "mag_calib" in lc.columns else None
    post_col = "mag_calib_ct" if "mag_calib_ct" in lc.columns else pre_col
    if not pre_col:
        return float("nan"), float("nan")
    pre = pd.to_numeric(lc[pre_col], errors="coerce")
    post = pd.to_numeric(lc[post_col], errors="coerce") if post_col else pre
    dpre = (float(cat_mag) - pre).dropna()
    dpost = (float(cat_mag) - post).dropna()
    spre = float(dpre.std()) if len(dpre) >= 2 else float("nan")
    spost = float(dpost.std()) if len(dpost) >= 2 else float("nan")
    return spre, spost


def _process_setup(*, draft_dir: Path, setup: str, cfg: AppConfig) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    flt = FILTER_LABEL.get(setup, setup.split("_")[0])
    ps_dir = draft_dir / "platesolve" / setup
    proc_dir = draft_dir / "detrended_aligned" / "lights" / setup
    comp_df = pd.read_csv(ps_dir / "comparison_stars.csv", low_memory=False)
    comp_df["catalog_id"] = comp_df["catalog_id"].map(_norm_cid)

    comp_ids = [c for c in comp_df["catalog_id"].astype(str).tolist() if c]
    comp_mag_inst = _build_comp_mag_inst(proc_dir, comp_ids)
    comp_catalog_mag = _comp_catalog_mag(comp_df)
    comp_bp_rp = _comp_bp_rp(comp_df)
    comp_quality = _comp_quality_from_df(comp_df)

    c1, c1_stderr, n_comp_used = fit_color_term_c1(
        comp_mag_inst,
        comp_catalog_mag,
        comp_bp_rp,
        comp_quality,
        min_comp=5,
        sigma_clip_sigma=3.0,
    )
    stderr_ratio = (
        abs(float(c1_stderr) / float(c1))
        if float(c1) != 0.0 and math.isfinite(float(c1_stderr))
        else float("nan")
    )
    scatter_pre, scatter_post = _color_term_cat_inst_scatter_pair(
        comp_mag_inst,
        comp_catalog_mag,
        comp_bp_rp,
        comp_quality,
        float(c1),
        min_comp=5,
        sigma_clip_sigma=3.0,
    )
    resid_rms = _fit_resid_rms(
        comp_mag_inst,
        comp_catalog_mag,
        comp_bp_rp,
        comp_quality,
        float(c1),
    )

    bp_vals = [float(v) for v in comp_bp_rp.values() if math.isfinite(float(v))]
    comp_min = float(min(bp_vals)) if bp_vals else float("nan")
    comp_max = float(max(bp_vals)) if bp_vals else float("nan")

    apply_ct, gate_reason = should_apply_color_term(
        obs_group=flt,
        c1=float(c1),
        c1_stderr=float(c1_stderr),
        n_comp=int(n_comp_used),
        min_comp_for_ct=MIN_COMP_CT,
        max_stderr_ratio=MAX_STDERR_RATIO,
    )

    targets_path = ps_dir / "photometry" / "active_targets.csv"
    if not targets_path.is_file():
        targets_path = ps_dir / "variable_targets.csv"
    targets = pd.read_csv(targets_path, low_memory=False, dtype={"catalog_id": str})
    targets["catalog_id"] = targets["catalog_id"].map(_norm_cid)

    target_rows: list[dict[str, Any]] = []
    n_in_range = 0
    n_ct_ok = 0
    n_red_blocked = 0

    for _, trow in targets.iterrows():
        cid = str(trow.get("catalog_id", "")).strip()
        if not cid:
            continue
        tgt_bp = float(pd.to_numeric(trow.get("bp_rp"), errors="coerce"))
        cat_mag = float(pd.to_numeric(trow.get("mag"), errors="coerce"))
        in_range = (
            math.isfinite(tgt_bp)
            and math.isfinite(comp_min)
            and math.isfinite(comp_max)
            and comp_min <= tgt_bp <= comp_max
        )
        if in_range:
            n_in_range += 1
        elif math.isfinite(tgt_bp) and math.isfinite(comp_max) and tgt_bp > comp_max:
            n_red_blocked += 1

        ct_corr = 0.0
        ct_ok = False
        if apply_ct and in_range and math.isfinite(tgt_bp):
            in_range_chk = _check_color_term_extrapolation(
                target_bp_rp=tgt_bp,
                comp_bp_rp_values=bp_vals,
                target_name=cid,
                extrapolation_tol=float(cfg.phase01_ct_extrapolation_tol),
            )
            if in_range_chk:
                _, ct_corr, _ = apply_color_term(
                    np.asarray([0.0]),
                    tgt_bp,
                    comp_bp_rp,
                    comp_quality,
                    float(c1),
                )
                ct_ok = bool(math.isfinite(ct_corr) and float(c1) != 0.0)

        if ct_ok:
            n_ct_ok += 1

        lc_path = ps_dir / "photometry" / "lightcurves" / f"lightcurve_{cid}.csv"
        t_scatter_pre, t_scatter_post = _target_scatter_pair(lc_path, cat_mag=cat_mag)

        target_rows.append(
            {
                "filter": flt,
                "catalog_id": cid,
                "bp_rp": tgt_bp if math.isfinite(tgt_bp) else "",
                "in_range": bool(in_range),
                "ct_ok": bool(ct_ok),
                "ct_corr": float(ct_corr) if math.isfinite(ct_corr) else "",
                "scatter_pre": t_scatter_pre if math.isfinite(t_scatter_pre) else "",
                "scatter_post": t_scatter_post if math.isfinite(t_scatter_post) else "",
            }
        )

    summary = {
        "filter": flt,
        "c1": float(c1),
        "c1_stderr": float(c1_stderr) if math.isfinite(float(c1_stderr)) else "",
        "stderr_ratio": stderr_ratio if math.isfinite(stderr_ratio) else "",
        "n_comp": int(n_comp_used),
        "comp_bp_rp_min": comp_min if math.isfinite(comp_min) else "",
        "comp_bp_rp_max": comp_max if math.isfinite(comp_max) else "",
        "resid_rms": resid_rms if math.isfinite(resid_rms) else "",
        "comp_scatter_pre": scatter_pre if math.isfinite(scatter_pre) else "",
        "comp_scatter_post": scatter_post if math.isfinite(scatter_post) else "",
        "n_in_range": int(n_in_range),
        "n_ct_ok": int(n_ct_ok),
        "n_red_giant_blocked": int(n_red_blocked),
        "gate_apply": bool(apply_ct),
        "gate_reason": gate_reason,
    }
    return summary, target_rows


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    cfg = AppConfig()
    draft_dir = Path(cfg.archive_root) / "Drafts" / f"draft_{DRAFT_ID:06d}"
    summaries: list[dict[str, Any]] = []
    all_targets: list[dict[str, Any]] = []

    for setup in SETUPS:
        summary, targets = _process_setup(draft_dir=draft_dir, setup=setup, cfg=cfg)
        summaries.append(summary)
        all_targets.extend(targets)
        print(
            f"[{summary['filter']}] c1={summary['c1']:+.4f} ± {summary['c1_stderr']} "
            f"stderr_ratio={summary['stderr_ratio']} n_comp={summary['n_comp']} "
            f"bp_rp=[{summary['comp_bp_rp_min']}, {summary['comp_bp_rp_max']}] "
            f"resid_rms={summary['resid_rms']} ct_ok={summary['n_ct_ok']}/{summary['n_in_range']} "
            f"red_giant_blocked={summary['n_red_giant_blocked']}",
            flush=True,
        )
        print(f"  gate: {summary['gate_reason']}", flush=True)

    pd.DataFrame(summaries).to_csv(SUMMARY_CSV, index=False)
    pd.DataFrame(all_targets).to_csv(TARGETS_CSV, index=False)

    report = {
        "draft_id": DRAFT_ID,
        "summary_csv": str(SUMMARY_CSV),
        "targets_csv": str(TARGETS_CSV),
        "summaries": summaries,
        "n_target_rows": len(all_targets),
    }
    (_ROOT / "chiandh_ct_dump_bvr375_result.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
