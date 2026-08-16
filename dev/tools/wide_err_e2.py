#!/usr/bin/env python3
"""WIDE-ERR E2: comp residual correlation and spatial common mode (read-only)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src_py"))
sys.path.insert(0, str(REPO / "dev" / "scripts"))
import _bootstrap  # noqa: E402,F401

from config import AppConfig  # noqa: E402
from photometry_core import (  # noqa: E402
    _group_comp_mag_inst_from_proc_csvs,
    check_comparison_stability,
    ensemble_normalize,
    temporal_bin_comp_lc,
)
from sigma_floor_core import c4_small_sample, ensemble_sem_mag_from_residuals  # noqa: E402

SETUP = "NoFilter_60_2"
DRAFT = REPO / "Archive" / "Drafts" / "draft_000435_snapshot_skysurface_20260716"
CHECK_CID = "1499906247391001088"
OUT = REPO / "tmp" / "wide_err_e2"
W1W2_JSON = REPO / "tmp" / "wide_err_w1w2" / "wide_err_w1w2.json"
E1_JSON = REPO / "tmp" / "wide_err_e1" / "wide_err_e1.json"


def _iqr(x: np.ndarray) -> list[float]:
    v = np.asarray(x, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return [float("nan")] * 3
    q25, q50, q75 = np.quantile(v, [0.25, 0.5, 0.75])
    return [float(q25), float(q50), float(q75)]


def _select_good_ids(
    comp_quality: dict[str, dict],
    comp_rms_map: dict[str, float],
    *,
    n_comp_min: int,
    n_comp_max: int,
) -> list[str]:
    p2p_thr = float("nan")
    for q in comp_quality.values():
        t = q.get("p2p_threshold")
        if t is not None and math.isfinite(float(t)):
            p2p_thr = float(t)
            break
    usable_all = [cid for cid, q in comp_quality.items() if q.get("quality") in ("good", "suspect")]
    usable_sorted = sorted(
        usable_all,
        key=lambda c: (
            0 if comp_quality[c].get("quality") == "good" else 1,
            float(comp_rms_map.get(c, float("inf"))),
            str(c),
        ),
    )
    selected: list[str] = []
    for cid in usable_sorted:
        if len(selected) >= n_comp_max:
            break
        p2p = float(comp_quality[cid].get("rms_p2p", float("nan")))
        if (
            len(selected) < n_comp_min
            or (math.isfinite(p2p_thr) and math.isfinite(p2p) and p2p < p2p_thr)
            or not math.isfinite(p2p_thr)
        ):
            selected.append(cid)
    return selected[:n_comp_max]


def _field_comp_lc(
    target_cid: str,
    comp_all: pd.DataFrame,
    csv_files: list[Path],
    cfg: AppConfig,
    all_frames_stub: pd.DataFrame,
) -> dict[str, Any] | None:
    comp_ids = [
        str(c).strip()
        for c in comp_all.loc[comp_all["target_catalog_id"].astype(str).str.strip() == target_cid, "catalog_id"]
        if str(c).strip() and str(c).strip() != CHECK_CID
    ]
    if len(comp_ids) < 2:
        return None
    comp_mag_raw = _group_comp_mag_inst_from_proc_csvs(comp_ids, csv_files)
    comp_rms_map = {
        str(r["catalog_id"]).strip(): float(pd.to_numeric(r.get("comp_rms"), errors="coerce"))
        for _, r in comp_all.loc[comp_all["target_catalog_id"].astype(str).str.strip() == target_cid].iterrows()
        if str(r["catalog_id"]).strip() in comp_ids
    }
    n_comp_min = max(1, int(getattr(cfg, "phase01_comparison_n_comp_min", 3)))
    n_comp_max = int(cfg.phase01_comparison_n_comp_max)
    comp_bjd = {cid: np.arange(len(comp_mag_raw[cid]), dtype=np.float64) for cid in comp_ids}
    comp_lc_bin = temporal_bin_comp_lc(
        comp_lc=comp_mag_raw,
        comp_quality={},
        all_frames=all_frames_stub,
        window=int(cfg.temporal_bin_window),
        enabled=bool(cfg.temporal_binning_enabled),
    )
    comp_quality = check_comparison_stability(
        comp_lc_bin,
        comp_rms_map=comp_rms_map,
        comp_bjd=comp_bjd,
        n_comp_min=n_comp_min,
        outlier_sigma=float(getattr(cfg, "stability_sigma", 3.0)),
        max_comp_slope_mmag_hr=float(cfg.comp_max_slope_mmag_hr),
        comp_slope_significance_k=float(getattr(cfg, "comp_slope_significance_k", 3.0)),
        common_mode_detrend=True,
    )
    good_ids = _select_good_ids(comp_quality, comp_rms_map, n_comp_min=n_comp_min, n_comp_max=n_comp_max)
    if len(good_ids) < 2:
        return None

    comp_ref: dict[str, float] = {}
    for cid in good_ids:
        arr = comp_lc_bin.get(cid)
        if arr is None:
            continue
        fin = arr[np.isfinite(arr)]
        if fin.size:
            comp_ref[cid] = float(np.median(fin))

    n_frames = len(csv_files)
    resid_mat = np.full((len(good_ids), n_frames), np.nan, dtype=np.float64)
    spread_per_frame: list[float] = []
    e14_ratios: list[float] = []
    quoted_sem: list[float] = []

    for i in range(n_frames):
        pairs: list[tuple[str, float]] = []
        for j, cid in enumerate(good_ids):
            if cid not in comp_lc_bin:
                continue
            mv = float(comp_lc_bin[cid][i])
            if math.isfinite(mv) and cid in comp_ref:
                pairs.append((cid, mv))
                resid_mat[j, i] = mv - comp_ref[cid]
        if len(pairs) >= 2:
            mags = [m for _, m in pairs]
            spread_per_frame.append(float(np.max(mags) - np.min(mags)))
            comp_resid = [m - comp_ref[cid] for cid, m in pairs]
            q = float(ensemble_sem_mag_from_residuals(comp_resid))
            quoted_sem.append(q)
            fluxes = [10 ** (-0.4 * m) for _, m in pairs]
            s = float(np.sum(fluxes))
            if s > 0:
                ens_med = float(-2.5 * math.log10(s))
                actual_r = [m - ens_med for _, m in pairs]
                n = len(actual_r)
                std = float(np.std(actual_r, ddof=1))
                c4 = c4_small_sample(n)
                actual_sem = std / c4 / math.sqrt(n) if c4 > 0 else float("nan")
                if math.isfinite(q) and q > 0 and math.isfinite(actual_sem):
                    e14_ratios.append(actual_sem / q)

    return {
        "target_cid": target_cid,
        "good_ids": good_ids,
        "comp_ref": comp_ref,
        "comp_lc": comp_lc_bin,
        "resid_mat": resid_mat,
        "comp_spread_median": float(np.median(spread_per_frame)) if spread_per_frame else float("nan"),
        "n_comp": len(good_ids),
        "e14_ratio_median": float(np.median(e14_ratios)) if e14_ratios else float("nan"),
        "quoted_sem_median_mmag": float(np.median(quoted_sem)) * 1000.0 if quoted_sem else float("nan"),
    }


def _rho_bar(resid_mat: np.ndarray) -> float:
    """Mean pairwise Pearson correlation across comp rows."""
    n_comp, _ = resid_mat.shape
    if n_comp < 2:
        return float("nan")
    rhos: list[float] = []
    for a in range(n_comp):
        for b in range(a + 1, n_comp):
            ra = resid_mat[a, :]
            rb = resid_mat[b, :]
            ok = np.isfinite(ra) & np.isfinite(rb)
            if int(np.count_nonzero(ok)) < 5:
                continue
            r = float(np.corrcoef(ra[ok], rb[ok])[0, 1])
            if math.isfinite(r):
                rhos.append(r)
    return float(np.mean(rhos)) if rhos else float("nan")


def _implied_factor(n_comp: int, rho_bar: float) -> float:
    if n_comp < 2 or not math.isfinite(rho_bar):
        return float("nan")
    return float(math.sqrt(max(0.0, 1.0 + (n_comp - 1) * rho_bar)))


def _pca_pc1(resid_mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (loadings per comp, scores per frame) for PC1."""
    mat = np.asarray(resid_mat, dtype=np.float64)
    # impute column means for missing
    col_mean = np.nanmean(mat, axis=0)
    filled = mat.copy()
    for j in range(filled.shape[1]):
        bad = ~np.isfinite(filled[:, j])
        if bad.any():
            filled[bad, j] = col_mean[j] if math.isfinite(col_mean[j]) else 0.0
    filled = filled - np.nanmean(filled, axis=0, keepdims=True)
    if filled.shape[0] < 2 or filled.shape[1] < 3:
        return np.full(filled.shape[0], np.nan), np.full(filled.shape[1], np.nan)
    u, s, vt = np.linalg.svd(filled, full_matrices=False)
    loadings = u[:, 0] * s[0]
    scores = vt[0, :]
    return loadings, scores


def _spatial_analysis(
    field: dict[str, Any],
    ms: pd.DataFrame,
    check_mag: np.ndarray | None,
) -> dict[str, Any]:
    good_ids = field["good_ids"]
    resid_mat = field["resid_mat"]
    loadings, scores = _pca_pc1(resid_mat)
    pos: dict[str, tuple[float, float]] = {}
    for cid in good_ids:
        row = ms.loc[ms["catalog_id"].astype(str).str.strip() == cid]
        if row.empty:
            continue
        x = float(pd.to_numeric(row["x"].iloc[0], errors="coerce"))
        y = float(pd.to_numeric(row["y"].iloc[0], errors="coerce"))
        if math.isfinite(x) and math.isfinite(y):
            pos[cid] = (x, y)
    if len(pos) < 3:
        return {"skip": True}

    # flux-weighted centroid from comp_ref (proxy for brightness)
    weights = []
    xs = []
    ys = []
    for cid in good_ids:
        if cid not in pos:
            continue
        ref = field["comp_ref"].get(cid, float("nan"))
        w = 10 ** (-0.4 * ref) if math.isfinite(ref) else float("nan")
        if math.isfinite(w) and w > 0:
            weights.append(w)
            xs.append(pos[cid][0])
            ys.append(pos[cid][1])
    if not weights:
        return {"skip": True}
    wsum = float(np.sum(weights))
    cx = float(np.sum(np.asarray(weights) * np.asarray(xs)) / wsum)
    cy = float(np.sum(np.asarray(weights) * np.asarray(ys)) / wsum)

    ld: list[float] = []
    xvals: list[float] = []
    yvals: list[float] = []
    offsets: list[float] = []
    for j, cid in enumerate(good_ids):
        if cid not in pos or j >= len(loadings):
            continue
        lv = float(loadings[j])
        if not math.isfinite(lv):
            continue
        x, y = pos[cid]
        ld.append(lv)
        xvals.append(x)
        yvals.append(y)
        offsets.append(float(math.hypot(x - cx, y - cy)))

    out: dict[str, Any] = {"skip": False}
    if len(ld) >= 4:
        for name, pred in (("x", xvals), ("y", yvals), ("radial_offset_px", offsets)):
            pv = np.asarray(pred, dtype=np.float64)
            lv = np.asarray(ld, dtype=np.float64)
            r, p = stats.pearsonr(pv, lv)
            sl, ic, _, _, se = stats.linregress(pv, lv)
            out[f"loading_vs_{name}"] = {
                "pearson_r": float(r),
                "p_value": float(p),
                "slope": float(sl),
                "slope_se": float(se),
                "significant_p05": bool(p < 0.05),
            }

    ck = ms.loc[ms["catalog_id"].astype(str).str.strip() == CHECK_CID]
    if not ck.empty:
        ckx = float(pd.to_numeric(ck["x"].iloc[0], errors="coerce"))
        cky = float(pd.to_numeric(ck["y"].iloc[0], errors="coerce"))
        out["check_offset_px"] = float(math.hypot(ckx - cx, cky - cy))
        out["check_xy"] = [ckx, cky]
        out["centroid_xy"] = [cx, cy]
        if check_mag is not None and len(check_mag) == resid_mat.shape[1]:
            ck_ref = float(np.nanmedian(check_mag[np.isfinite(check_mag)]))
            ck_resid = check_mag - ck_ref
            ok = np.isfinite(ck_resid) & np.isfinite(scores)
            if int(np.count_nonzero(ok)) >= 5:
                out["check_pc1_corr"] = float(np.corrcoef(ck_resid[ok], scores[ok])[0, 1])
            if "loading_vs_x" in out and "loading_vs_y" in out:
                sx = out["loading_vs_x"]["slope"]
                sy = out["loading_vs_y"]["slope"]
                ic = out["loading_vs_x"].get("intercept", 0.0)
                # linregress intercept stored separately
                slx, icx, _, _, _ = stats.linregress(np.asarray(xvals), np.asarray(ld))
                sly, icy, _, _, _ = stats.linregress(np.asarray(yvals), np.asarray(ld))
                pred_loading = icx + slx * (ckx - np.mean(xvals)) + icy + sly * (cky - np.mean(yvals))
                out["check_loading_predicted"] = float(pred_loading)
                out["check_loading_actual_proxy"] = float(np.nanmean(ck_resid[ok])) if ok.any() else float("nan")

    return out


def main() -> int:
    cfg = AppConfig()
    OUT.mkdir(parents=True, exist_ok=True)
    ps = DRAFT / "platesolve" / SETUP
    phot = ps / "photometry"
    lights = DRAFT / "detrended_aligned" / "lights" / SETUP
    comp_all = pd.read_csv(
        phot / "comparison_stars_per_target.csv",
        low_memory=False,
        dtype={"catalog_id": str, "target_catalog_id": str},
    )
    ms = pd.read_csv(ps / "masterstars_full_match.csv", dtype={"catalog_id": str})
    csv_files = sorted(lights.glob("proc_*.csv"))
    all_frames_stub = pd.DataFrame({"catalog_id": [], "bjd": []})

    w1 = json.loads(W1W2_JSON.read_text(encoding="ascii"))
    w1_ratio = {str(r["target_cid"]): float(r["ratio_total_robust"]) for r in w1.get("per_field", [])}

    target_ids = sorted(
        t
        for t in {str(x).strip() for x in comp_all["target_catalog_id"].astype(str)}
        if t and (phot / "lightcurves" / f"check_kmag_{t}.csv").is_file()
    )

    e20_rows: list[dict[str, float]] = []
    e21_rows: list[dict[str, float]] = []
    e22_rows: list[dict[str, Any]] = []

    for t in target_ids:
        field = _field_comp_lc(t, comp_all, csv_files, cfg, all_frames_stub)
        if field is None:
            continue
        e20_rows.append(
            {
                "target_cid": t,
                "e14_ratio": field["e14_ratio_median"],
                "comp_spread": field["comp_spread_median"],
                "n_comp": field["n_comp"],
            }
        )
        rho = _rho_bar(field["resid_mat"])
        n = field["n_comp"]
        pred = _implied_factor(n, rho)
        meas = w1_ratio.get(t, float("nan"))
        e21_rows.append(
            {
                "target_cid": t,
                "rho_bar": rho,
                "n_comp": n,
                "predicted_factor": pred,
                "measured_ratio": meas,
            }
        )
        if rho > 0.1 or True:  # always collect; E2.2 conditional read in report
            ck_mag = None
            ck_path = REPO / "tmp" / "wide_err_w1w2" / "diag_check_lc" / t / f"lightcurve_{CHECK_CID}.csv"
            if ck_path.is_file():
                lc = pd.read_csv(ck_path, low_memory=False)
                ck_mag = pd.to_numeric(lc.get("mag_inst"), errors="coerce").to_numpy(dtype=np.float64)
                if not np.isfinite(ck_mag).any():
                    ck_mag = pd.to_numeric(lc.get("mag_calib_final"), errors="coerce").to_numpy(dtype=np.float64)
            spa = _spatial_analysis(field, ms, ck_mag)
            if not spa.get("skip"):
                spa["target_cid"] = t
                spa["rho_bar"] = rho
                e22_rows.append(spa)

    df20 = pd.DataFrame(e20_rows)
    df21 = pd.DataFrame(e21_rows)

    def _spearman(a: pd.Series, b: pd.Series) -> dict[str, float]:
        ok = a.notna() & b.notna()
        if int(ok.sum()) < 5:
            return {"rho": float("nan"), "p": float("nan"), "n": int(ok.sum())}
        r, p = stats.spearmanr(a[ok], b[ok])
        return {"rho": float(r), "p": float(p), "n": int(ok.sum())}

    e20_spread = _spearman(df20["e14_ratio"], df20["comp_spread"])
    e20_ncomp = _spearman(df20["e14_ratio"], df20["n_comp"])

    pred_arr = df21["predicted_factor"].to_numpy(dtype=np.float64)
    meas_arr = df21["measured_ratio"].to_numpy(dtype=np.float64)
    e21_spear = _spearman(df21["predicted_factor"], df21["measured_ratio"])

    # E2.2 aggregates
    sig_x = [r for r in e22_rows if r.get("loading_vs_x", {}).get("significant_p05")]
    sig_y = [r for r in e22_rows if r.get("loading_vs_y", {}).get("significant_p05")]
    sig_rad = [r for r in e22_rows if r.get("loading_vs_radial_offset_px", {}).get("significant_p05")]

    out = {
        "E2_0": {
            "n_fields": len(df20),
            "e14_vs_comp_spread_spearman": e20_spread,
            "e14_vs_n_comp_spearman": e20_ncomp,
            "e14_ratio_median": float(df20["e14_ratio"].median()),
            "comp_spread_median": float(df20["comp_spread"].median()),
            "artifact_threshold": "strong positive rho with spread -> retract E1.4",
            "retract_e14": bool(
                math.isfinite(e20_spread["rho"])
                and e20_spread["rho"] > 0.5
                and e20_spread["p"] < 0.05
            ),
        },
        "E2_1": {
            "n_fields": len(df21),
            "rho_bar_median": float(df21["rho_bar"].median()),
            "rho_bar_iqr": _iqr(df21["rho_bar"].to_numpy(dtype=np.float64)),
            "n_comp_median": float(df21["n_comp"].median()),
            "predicted_factor_median": float(df21["predicted_factor"].median()),
            "predicted_factor_iqr": _iqr(pred_arr),
            "measured_ratio_median": float(df21["measured_ratio"].median()),
            "predicted_vs_measured_spearman": e21_spear,
        },
        "E2_2": {
            "n_fields_analyzed": len(e22_rows),
            "fraction_sig_x": float(len(sig_x) / len(e22_rows)) if e22_rows else float("nan"),
            "fraction_sig_y": float(len(sig_y) / len(e22_rows)) if e22_rows else float("nan"),
            "fraction_sig_radial": float(len(sig_rad) / len(e22_rows)) if e22_rows else float("nan"),
            "median_check_offset_px": float(np.median([r.get("check_offset_px", float("nan")) for r in e22_rows])),
            "sample_fields": e22_rows[:5],
        },
        "E2_3": {"status": "NOT AVAILABLE", "reason": "no Newton/Dablice draft with check_kmag LCs on disk"},
    }
    (OUT / "wide_err_e2.json").write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="ascii")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
