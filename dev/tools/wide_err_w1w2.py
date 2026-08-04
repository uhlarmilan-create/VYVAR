#!/usr/bin/env python3
"""WIDE-ERR W1+W2: discriminator and N_eff test on restored draft_435 snapshot."""

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
    _phase2a_prepare_shared_state,
    photometer_check_star_production_path,
)

SETUP = "NoFilter_60_2"
DRAFT_NAME = "draft_000435_snapshot_skysurface_20260716"
DRAFT_ID = 435
CHECK_CID = "1499906247391001088"
MAG_ERR_SCALE = 1000.0
MAD_SCALE = 1.4826
OUT_ROOT = REPO / "tmp" / "wide_err_w1w2"
DIAG_LC_ROOT = OUT_ROOT / "diag_check_lc"


def _iqr(x: np.ndarray) -> tuple[float, float, float]:
    v = np.asarray(x, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    q25, q50, q75 = np.quantile(v, [0.25, 0.5, 0.75])
    return float(q25), float(q50), float(q75)


def _weighted_scatter(mags: np.ndarray, errs: np.ndarray) -> float:
    m = np.asarray(mags, dtype=np.float64)
    e = np.asarray(errs, dtype=np.float64)
    ok = np.isfinite(m) & np.isfinite(e) & (e > 0)
    m = m[ok]
    e = e[ok]
    if m.size < 3:
        return float("nan")
    w = 1.0 / (e * e)
    ref = float(np.sum(w * m) / np.sum(w))
    resid = m - ref
    return float(np.std(resid, ddof=1))


def _mad_sigma(x: np.ndarray) -> float:
    v = np.asarray(x, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < 3:
        return float("nan")
    med = float(np.median(v))
    return float(MAD_SCALE * np.median(np.abs(v - med)))


def _p2p_scatter(mags: np.ndarray) -> float:
    m = np.asarray(mags, dtype=np.float64)
    m = m[np.isfinite(m)]
    if m.size < 3:
        return float("nan")
    diffs = np.diff(m)
    return float(np.std(diffs, ddof=1) / math.sqrt(2.0))


def _p2p_robust(mags: np.ndarray) -> float:
    m = np.asarray(mags, dtype=np.float64)
    m = m[np.isfinite(m)]
    if m.size < 3:
        return float("nan")
    diffs = np.diff(m)
    return float(_mad_sigma(diffs) / math.sqrt(2.0))


def _mad_clip_rejected(m: np.ndarray, e: np.ndarray, *, sigma: float = 5.0) -> int:
    ok = np.isfinite(m) & np.isfinite(e) & (e > 0)
    mo = m[ok]
    if mo.size < 3:
        return 0
    med = float(np.median(mo))
    mad = float(np.median(np.abs(mo - med)))
    if mad <= 0:
        return 0
    thresh = sigma * MAD_SCALE * mad
    return int(np.sum(np.abs(mo - med) > thresh))


def _detrend_stats(t: np.ndarray, y: np.ndarray) -> dict[str, float]:
    ok = np.isfinite(t) & np.isfinite(y)
    t = t[ok]
    y = y[ok]
    if t.size < 4:
        return {
            "slope": float("nan"),
            "slope_se": float("nan"),
            "f_p_linear": float("nan"),
            "f_p_quad": float("nan"),
            "sigma_after_linear": float("nan"),
            "sigma_after_quad": float("nan"),
        }
    t0 = t - float(np.mean(t))
    # constant
    ss_const = float(np.sum((y - np.mean(y)) ** 2))
    # linear
    A1 = np.column_stack([t0, np.ones(t0.size)])
    c1, _, _, _ = np.linalg.lstsq(A1, y, rcond=None)
    pred1 = A1 @ c1
    resid1 = y - pred1
    ss_lin = float(np.sum(resid1 ** 2))
    df1 = max(1, t0.size - 2)
    mse1 = ss_lin / df1
    cov1 = mse1 * np.linalg.inv(A1.T @ A1)
    slope_se = float(math.sqrt(cov1[0, 0])) if cov1.size else float("nan")
    f_lin = ((ss_const - ss_lin) / 1.0) / (ss_lin / df1) if ss_lin > 0 else float("nan")
    p_lin = float(1.0 - stats.f.cdf(f_lin, 1, df1)) if math.isfinite(f_lin) else float("nan")
    # quadratic
    A2 = np.column_stack([t0 ** 2, t0, np.ones(t0.size)])
    c2, _, _, _ = np.linalg.lstsq(A2, y, rcond=None)
    pred2 = A2 @ c2
    resid2 = y - pred2
    ss_quad = float(np.sum(resid2 ** 2))
    df2 = max(1, t0.size - 3)
    f_quad = ((ss_const - ss_quad) / 2.0) / (ss_quad / df2) if ss_quad > 0 else float("nan")
    p_quad = float(1.0 - stats.f.cdf(f_quad, 2, df2)) if math.isfinite(f_quad) else float("nan")
    return {
        "slope": float(c1[0]),
        "slope_se": slope_se,
        "f_p_linear": p_lin,
        "f_p_quad": p_quad,
        "sigma_after_linear": float(np.std(resid1, ddof=1)),
        "sigma_after_quad": float(np.std(resid2, ddof=1)),
    }


def _neff_frame(comp_mags: list[float]) -> tuple[int, float, float]:
    mags = [float(x) for x in comp_mags if math.isfinite(x)]
    n = len(mags)
    if n == 0:
        return 0, float("nan"), float("nan")
    fluxes = [10 ** (-0.4 * m) for m in mags]
    s = sum(fluxes)
    s2 = sum(f * f for f in fluxes)
    if s <= 0 or s2 <= 0:
        return n, float("nan"), float("nan")
    n_eff = (s * s) / s2
    factor = math.sqrt(n / n_eff) if n_eff > 0 else float("nan")
    return n, float(n_eff), float(factor)


def _field_neff(
    *,
    target_cid: str,
    check_cid: str,
    comp_df: pd.DataFrame,
    csv_files: list[Path],
    lc_df: pd.DataFrame,
) -> dict[str, Any]:
    sub = comp_df.loc[comp_df["target_catalog_id"].astype(str).str.strip() == target_cid]
    comp_ids = [
        str(c).strip()
        for c in sub["catalog_id"].tolist()
        if str(c).strip() and str(c).strip() != check_cid
    ]
    comp_mag_inst = _group_comp_mag_inst_from_proc_csvs(comp_ids, csv_files)
    n_frames = len(csv_files)
    per_frame: list[dict[str, float]] = []
    spreads: list[float] = []
    for i in range(n_frames):
        pairs = []
        for cid in comp_ids:
            if cid not in comp_mag_inst:
                continue
            mv = float(comp_mag_inst[cid][i])
            if math.isfinite(mv):
                pairs.append(mv)
        if len(pairs) < 2:
            continue
        n, n_eff, fac = _neff_frame(pairs)
        per_frame.append({"N": n, "N_eff": n_eff, "factor": fac})
        spreads.append(max(pairs) - min(pairs))
    if not per_frame:
        return {"n_frames": 0}
    df = pd.DataFrame(per_frame)
    return {
        "n_frames": int(len(df)),
        "N_median": float(df["N"].median()),
        "N_eff_median": float(df["N_eff"].median()),
        "factor_median": float(df["factor"].median()),
        "spread_median": float(np.median(spreads)) if spreads else float("nan"),
        "spread_p90": float(np.quantile(spreads, 0.9)) if spreads else float("nan"),
    }


def main() -> int:
    cfg = AppConfig()
    draft = Path(cfg.archive_root) / "Drafts" / DRAFT_NAME
    ps = draft / "platesolve" / SETUP
    phot = ps / "photometry"
    lc_dir = phot / "lightcurves"
    lights = draft / "detrended_aligned" / "lights" / SETUP
    scratch = OUT_ROOT / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    DIAG_LC_ROOT.mkdir(parents=True, exist_ok=True)

    comp_all = pd.read_csv(
        phot / "comparison_stars_per_target.csv",
        low_memory=False,
        dtype={"catalog_id": str, "target_catalog_id": str},
    )

    state = _phase2a_prepare_shared_state(
        output_dir=scratch,
        lc_dir=scratch / "lightcurves",
        masterstar_fits_path=ps / "MASTERSTAR.fits",
        comparison_stars_csv=phot / "comparison_stars_per_target.csv",
        per_frame_csv_dir=lights,
        progress_cb=None,
        active_targets_csv=ps / "variable_targets.csv",
        detrended_aligned_dir=lights,
        fwhm_px=3.2,
        cfg=cfg,
        db=None,
        draft_id=DRAFT_ID,
    )
    csv_files = state.csv_files

    rows: list[dict[str, Any]] = []
    skipped = 0
    for ck_path in sorted(lc_dir.glob("check_kmag_*.csv")):
        target_cid = ck_path.stem.replace("check_kmag_", "")
        ckdf = pd.read_csv(ck_path, nrows=1, low_memory=False)
        id_col = "check_catalog_id" if "check_catalog_id" in ckdf.columns else "check_cid"
        check_cid = str(ckdf[id_col].iloc[0]).strip()
        if check_cid != CHECK_CID:
            continue
        lc_path = DIAG_LC_ROOT / target_cid / f"lightcurve_{check_cid}.csv"
        if lc_path.is_file():
            lc_df = pd.read_csv(lc_path, low_memory=False)
        else:
            try:
                lc_df = photometer_check_star_production_path(
                    state=state,
                    parent_target_cid=target_cid,
                    check_cid=check_cid,
                    masterstar_fits_path=ps / "MASTERSTAR.fits",
                    lc_dir=DIAG_LC_ROOT / target_cid,
                    output_dir=scratch,
                )
            except Exception:  # noqa: BLE001
                skipped += 1
                continue
        if lc_df is None or "mag_calib_final" not in lc_df.columns:
            skipped += 1
            continue
        m = pd.to_numeric(lc_df["mag_calib_final"], errors="coerce").to_numpy(dtype=np.float64)
        e = pd.to_numeric(lc_df["err"], errors="coerce").to_numpy(dtype=np.float64)
        t = pd.to_numeric(lc_df.get("bjd"), errors="coerce").to_numpy(dtype=np.float64)
        ok = np.isfinite(m) & np.isfinite(e) & (e > 0)
        n_ok = int(np.count_nonzero(ok))
        if n_ok < 3:
            skipped += 1
            continue
        mo = m[ok]
        eo = e[ok]
        to = t[ok] if np.isfinite(t[ok]).sum() >= 4 else t[np.isfinite(m)]
        err_med = float(np.median(eo))
        st = _weighted_scatter(mo, eo)
        str_ = _mad_sigma(mo)
        sp = _p2p_scatter(mo)
        spr = _p2p_robust(mo)
        n_out = _mad_clip_rejected(mo, eo)
        det = _detrend_stats(to[: mo.size], mo)
        neff = _field_neff(
            target_cid=target_cid,
            check_cid=check_cid,
            comp_df=comp_all,
            csv_files=csv_files,
            lc_df=lc_df,
        )
        row = {
            "target_cid": target_cid,
            "n_epochs": n_ok,
            "n_outliers_5mad": n_out,
            "sigma_total": st,
            "sigma_total_robust": str_,
            "sigma_p2p": sp,
            "sigma_p2p_robust": spr,
            "err_median": err_med,
            "ratio_total": st / err_med if err_med > 0 else float("nan"),
            "ratio_total_robust": str_ / err_med if err_med > 0 else float("nan"),
            "ratio_p2p": sp / err_med if err_med > 0 else float("nan"),
            "ratio_p2p_robust": spr / err_med if err_med > 0 else float("nan"),
            "ratio_after_linear_detrend": det["sigma_after_linear"] / err_med
            if err_med > 0
            else float("nan"),
            "ratio_after_quad_detrend": det["sigma_after_quad"] / err_med
            if err_med > 0
            else float("nan"),
            **det,
            **neff,
        }
        rows.append(row)

    out = {
        "check_cid": CHECK_CID,
        "n_fields_attempted": int(len(list(lc_dir.glob("check_kmag_*.csv")))),
        "n_fields_with_check_id": int(
            sum(
                1
                for p in lc_dir.glob("check_kmag_*.csv")
                if str(
                    pd.read_csv(p, nrows=1, low_memory=False)[
                        "check_catalog_id"
                        if "check_catalog_id" in pd.read_csv(p, nrows=0).columns
                        else "check_cid"
                    ].iloc[0]
                ).strip()
                == CHECK_CID
            )
        ),
        "n_fields_valid": len(rows),
        "n_skipped": skipped,
        "total_outliers_5mad": int(sum(r["n_outliers_5mad"] for r in rows)),
        "per_field": rows,
    }

    if rows:
        df = pd.DataFrame(rows)
        for col, key in (
            ("ratio_total", "ratio_total"),
            ("ratio_total_robust", "ratio_total_robust"),
            ("ratio_p2p", "ratio_p2p"),
            ("ratio_p2p_robust", "ratio_p2p_robust"),
            ("ratio_after_linear_detrend", "ratio_after_linear_detrend"),
            ("ratio_after_quad_detrend", "ratio_after_quad_detrend"),
            ("N_median", "N_median"),
            ("N_eff_median", "N_eff_median"),
            ("factor_median", "factor_median"),
            ("spread_median", "spread_median"),
            ("spread_p90", "spread_p90"),
        ):
            if col in df.columns:
                q25, q50, q75 = _iqr(df[col].to_numpy(dtype=np.float64))
                out[f"{key}_iqr"] = [q25, q50, q75]
        mask = df["factor_median"].notna() & df["ratio_total_robust"].notna()
        if int(mask.sum()) >= 5:
            rho, p = stats.spearmanr(
                df.loc[mask, "factor_median"],
                df.loc[mask, "ratio_total_robust"],
            )
            out["spearman_factor_vs_ratio_robust"] = {
                "rho": float(rho),
                "p": float(p),
                "n": int(mask.sum()),
            }

    out_path = OUT_ROOT / "wide_err_w1w2.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="ascii")
    print(json.dumps({k: out[k] for k in out if k != "per_field"}, indent=2))
    print(f"n_valid={len(rows)}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
