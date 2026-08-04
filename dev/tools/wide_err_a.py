#!/usr/bin/env python3
"""WIDE-ERR A: level-dependent vs constant rig floor (read-only Archive)."""

from __future__ import annotations

import json
import math
import sqlite3
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src_py"))
sys.path.insert(0, str(REPO / "dev" / "scripts"))
import _bootstrap  # noqa: E402,F401

from config import AppConfig  # noqa: E402
from gaia_catalog_id import normalize_gaia_source_id  # noqa: E402
from scripts.fit_sigma_floor import (  # noqa: E402
    _build_star_arrays,
    _passes_saturation_gate,
    collect_comp_candidates,
    pick_anchor_target,
    pick_g_coverage,
)
from scripts.select_constant_calibrators import build_loo_differential_lc  # noqa: E402
from sigma_budget import resolve_rig_scintillation_params, scintillation_sigma  # noqa: E402

SETUP = "NoFilter_60_2"
DRAFT_NAME = "draft_000435_snapshot_skysurface_20260716"
DRAFT_ID = 435
CHECK_CID = "1499906247391001088"
MAD_SCALE = 1.4826
MAG_ERR_SCALE = 1000.0
OUT_ROOT = REPO / "tmp" / "wide_err_a"
W1W2_LC_ROOT = REPO / "tmp" / "wide_err_w1w2" / "diag_check_lc"
PEAK_COL = "peak_max_adu"
FWHM_COL = "fwhm_estimate_px"


def _mad_sigma(x: np.ndarray) -> float:
    v = np.asarray(x, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < 3:
        return float("nan")
    med = float(np.median(v))
    return float(MAD_SCALE * np.median(np.abs(v - med)))


def _iqr(x: np.ndarray) -> list[float]:
    v = np.asarray(x, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return [float("nan"), float("nan"), float("nan")]
    q25, q50, q75 = np.quantile(v, [0.25, 0.5, 0.75])
    return [float(q25), float(q50), float(q75)]


def _load_proc_index(proc_dir: Path) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for p in sorted(proc_dir.glob("proc_*.csv")):
        df = pd.read_csv(p, low_memory=False, dtype={"catalog_id": str})
        df["_nid"] = df["catalog_id"].apply(lambda x: str(normalize_gaia_source_id(x) or "").strip())
        out[p.name] = df
    return out


def _star_proc_row(proc_index: dict[str, pd.DataFrame], source_file: str, cid: str) -> pd.Series | None:
    key = Path(str(source_file)).name
    df = proc_index.get(key)
    if df is None:
        return None
    sub = df.loc[df["_nid"] == str(cid)]
    if sub.empty:
        return None
    return sub.iloc[0]


def _weighted_mean(m: np.ndarray, e: np.ndarray) -> float:
    ok = np.isfinite(m) & np.isfinite(e) & (e > 0)
    mo = m[ok]
    eo = e[ok]
    if mo.size == 0:
        return float("nan")
    w = 1.0 / (eo * eo)
    return float(np.sum(w * mo) / np.sum(w))


def _lc_excess(lc_df: pd.DataFrame) -> dict[str, float]:
    m = pd.to_numeric(lc_df.get("mag_calib_final"), errors="coerce").to_numpy(dtype=np.float64)
    e = pd.to_numeric(lc_df.get("err"), errors="coerce").to_numpy(dtype=np.float64)
    ok = np.isfinite(m) & np.isfinite(e) & (e > 0)
    if int(np.count_nonzero(ok)) < 3:
        return {"sigma_robust_mmag": float("nan"), "err_mmag": float("nan"), "excess_mmag": float("nan"), "n": 0}
    mo = m[ok]
    eo = e[ok]
    sig = _mad_sigma(mo)
    err = float(np.median(eo))
    exc = math.sqrt(max(0.0, sig * sig - err * err)) * MAG_ERR_SCALE if sig > err else 0.0
    return {
        "sigma_robust_mmag": sig * MAG_ERR_SCALE,
        "err_mmag": err * MAG_ERR_SCALE,
        "excess_mmag": exc,
        "n": int(mo.size),
    }


def run_a1(proc_index: dict[str, pd.DataFrame]) -> dict[str, Any]:
    field_rows: list[dict[str, Any]] = []
    pooled: list[dict[str, float]] = []
    peak_vals: list[float] = []

    for lc_path in sorted(W1W2_LC_ROOT.glob(f"*/lightcurve_{CHECK_CID}.csv")):
        target = lc_path.parent.name
        lc = pd.read_csv(lc_path, low_memory=False)
        m = pd.to_numeric(lc.get("mag_calib_final"), errors="coerce").to_numpy(dtype=np.float64)
        e = pd.to_numeric(lc.get("err"), errors="coerce").to_numpy(dtype=np.float64)
        ok = np.isfinite(m) & np.isfinite(e) & (e > 0)
        if int(np.count_nonzero(ok)) < 5:
            continue
        ref = _weighted_mean(m, e)
        r_mmag = (m[ok] - ref) * MAG_ERR_SCALE
        peaks = []
        fwhms = []
        for i, src in enumerate(lc.loc[ok, "source_file"].astype(str)):
            row = _star_proc_row(proc_index, src, CHECK_CID)
            if row is None:
                peaks.append(float("nan"))
                fwhms.append(float("nan"))
                continue
            peaks.append(float(pd.to_numeric(row.get(PEAK_COL), errors="coerce")))
            fwhms.append(float(pd.to_numeric(row.get(FWHM_COL), errors="coerce")))
            peak_vals.append(peaks[-1])
            pooled.append(
                {
                    "peak": peaks[-1],
                    "fwhm": fwhms[-1],
                    "abs_r_mmag": abs(float(r_mmag[i])),
                    "r_mmag": float(r_mmag[i]),
                }
            )
        p = np.asarray(peaks, dtype=np.float64)
        f = np.asarray(fwhms, dtype=np.float64)
        r = np.asarray(r_mmag, dtype=np.float64)
        mask = np.isfinite(p) & np.isfinite(r)
        slope = slope_se = pval = float("nan")
        if int(mask.sum()) >= 5:
            lr = stats.linregress(p[mask], r[mask])
            slope, slope_se, pval = float(lr.slope), float(lr.stderr), float(lr.pvalue)
        partial_note = "NOT SEPARABLE"
        partial_p = float("nan")
        partial_slope = float("nan")
        m2 = mask & np.isfinite(f)
        if int(m2.sum()) >= 8:
            # partial: r ~ peak + fwhm; peak significance with FWHM held
            X = np.column_stack([p[m2], f[m2], np.ones(int(m2.sum()))])
            beta, _, _, _ = np.linalg.lstsq(X, r[m2], rcond=None)
            partial_slope = float(beta[0])
            resid = r[m2] - X @ beta
            mse = float(np.sum(resid ** 2) / max(1, int(m2.sum()) - 3))
            try:
                cov = mse * np.linalg.inv(X.T @ X)
                partial_se = math.sqrt(max(0.0, cov[0, 0]))
                partial_t = partial_slope / partial_se if partial_se > 0 else float("nan")
                partial_p = float(2 * (1 - stats.t.cdf(abs(partial_t), int(m2.sum()) - 3)))
                partial_note = "partial_peak_with_fwhm"
            except np.linalg.LinAlgError:
                partial_note = "NOT SEPARABLE"
        field_rows.append(
            {
                "target_cid": target,
                "n_epochs": int(mask.sum()),
                "slope_mmag_per_adu": slope,
                "slope_se": slope_se,
                "p_value": pval,
                "partial_peak_slope": partial_slope,
                "partial_peak_p": partial_p,
                "partial_note": partial_note,
            }
        )

    slopes = np.asarray([r["slope_mmag_per_adu"] for r in field_rows], dtype=np.float64)
    pvals = np.asarray([r["p_value"] for r in field_rows], dtype=np.float64)
    pos = slopes > 0
    neg = slopes < 0
    sig_pos = float(np.mean((pvals < 0.05) & pos)) if slopes.size else float("nan")
    sig_neg = float(np.mean((pvals < 0.05) & neg)) if slopes.size else float("nan")
    dominant_sign = "positive" if float(np.nanmedian(slopes)) >= 0 else "negative"
    frac_sig_same = sig_pos if dominant_sign == "positive" else sig_neg

    deciles: list[dict[str, float]] = []
    if pooled:
        pdf = pd.DataFrame(pooled)
        pdf = pdf.loc[pdf["peak"].notna()]
        if not pdf.empty:
            pdf["decile"] = pd.qcut(pdf["peak"], 10, duplicates="drop")
            for dec, grp in pdf.groupby("decile", observed=True):
                deciles.append(
                    {
                        "decile": str(dec),
                        "peak_median": float(grp["peak"].median()),
                        "median_abs_r_mmag": float(grp["abs_r_mmag"].median()),
                        "robust_scatter_mmag": float(_mad_sigma(grp["r_mmag"].to_numpy()) * 1.0),
                        "n": int(len(grp)),
                    }
                )

    peak_arr = np.asarray([x for x in peak_vals if math.isfinite(x)], dtype=np.float64)
    return {
        "peak_column": PEAK_COL,
        "peak_p05_p50_p95": [
            float(np.quantile(peak_arr, 0.05)),
            float(np.quantile(peak_arr, 0.5)),
            float(np.quantile(peak_arr, 0.95)),
        ]
        if peak_arr.size
        else [float("nan")] * 3,
        "n_fields": len(field_rows),
        "slope_iqr": _iqr(slopes),
        "fraction_p_lt_0_05": float(np.mean(pvals < 0.05)) if pvals.size else float("nan"),
        "fraction_p_lt_0_05_consistent_sign": frac_sig_same,
        "dominant_slope_sign": dominant_sign,
        "deciles": deciles,
        "per_field": field_rows,
    }


def run_a2(
    *,
    draft: Path,
    phot: Path,
    proc_index: dict[str, pd.DataFrame],
    ms: pd.DataFrame,
    cfg: AppConfig,
) -> dict[str, Any]:
    lc_dir = phot / "lightcurves"
    stars: list[dict[str, Any]] = []
    for lc_path in sorted(lc_dir.glob("lightcurve_*.csv")):
        cid = lc_path.stem.replace("lightcurve_", "").split("_")[0]
        lc = pd.read_csv(lc_path, low_memory=False)
        exc = _lc_excess(lc)
        if exc["n"] < 3:
            continue
        g = float("nan")
        ms_row = ms.loc[ms["catalog_id"].astype(str).str.strip() == cid]
        if not ms_row.empty:
            g = float(pd.to_numeric(ms_row["phot_g_mean_mag"].iloc[0], errors="coerce"))
        peaks = []
        fwhms = []
        for src in lc["source_file"].astype(str):
            row = _star_proc_row(proc_index, src, cid)
            if row is None:
                continue
            pv = float(pd.to_numeric(row.get(PEAK_COL), errors="coerce"))
            fv = float(pd.to_numeric(row.get(FWHM_COL), errors="coerce"))
            if math.isfinite(pv):
                peaks.append(pv)
            if math.isfinite(fv):
                fwhms.append(fv)
        stars.append(
            {
                "catalog_id": cid,
                "mag_g": g,
                "median_peak_adu": float(np.median(peaks)) if peaks else float("nan"),
                "median_fwhm_px": float(np.median(fwhms)) if fwhms else float("nan"),
                **exc,
            }
        )

    sdf = pd.DataFrame(stars)
    if sdf.empty:
        return {"n_stars": 0}

    def _bin_median(xcol: str, ycol: str, *, nbins: int = 10) -> list[dict[str, float]]:
        sub = sdf.loc[sdf[xcol].notna() & sdf[ycol].notna()].copy()
        if sub.empty:
            return []
        sub["bin"] = pd.qcut(sub[xcol], nbins, duplicates="drop")
        out = []
        for b, grp in sub.groupby("bin", observed=True):
            out.append(
                {
                    "bin": str(b),
                    f"{xcol}_median": float(grp[xcol].median()),
                    f"{ycol}_median": float(grp[ycol].median()),
                    "n": int(len(grp)),
                }
            )
        return out

    # partial Spearman excess vs peak controlling G (rank residuals)
    sub = sdf.loc[sdf["excess_mmag"].notna() & sdf["median_peak_adu"].notna() & sdf["mag_g"].notna()]
    partial = {"rho": float("nan"), "p": float("nan"), "n": 0}
    if len(sub) >= 8:
        rg = stats.rankdata(sub["mag_g"].to_numpy())
        rp = stats.rankdata(sub["median_peak_adu"].to_numpy())
        re = stats.rankdata(sub["excess_mmag"].to_numpy())
        # residualize ranks
        A = np.column_stack([rg, np.ones(len(rg))])
        bp, _, _, _ = np.linalg.lstsq(A, rp, rcond=None)
        be, _, _, _ = np.linalg.lstsq(A, re, rcond=None)
        rp_res = rp - A @ bp
        re_res = re - A @ be
        rho, p = stats.spearmanr(rp_res, re_res)
        partial = {"rho": float(rho), "p": float(p), "n": int(len(sub))}

    # calibrator cohort (batch D n=12)
    from check_star_kmag import resolve_proc_csv_dir  # noqa: PLC0415

    proc_dir = resolve_proc_csv_dir(phot, SETUP)
    comp_all = pd.read_csv(
        phot / "comparison_stars_per_target.csv",
        low_memory=False,
        dtype={"catalog_id": str, "target_catalog_id": str},
    )
    meta = json.loads((phot / "pipeline_meta.json").read_text(encoding="utf-8"))
    gain = float(meta.get("gain") or 1.0)
    rn = float(meta.get("read_noise") or 10.0)
    anchor = pick_anchor_target(phot, comp_all)
    calibrators = pick_g_coverage(collect_comp_candidates(phot, min_frames=15), aim=12)
    cal_rows: list[dict[str, Any]] = []
    for _, row in calibrators.iterrows():
        cid = str(row["catalog_id"]).strip()
        loo = build_loo_differential_lc(
            cid, phot_dir=phot, setup=SETUP, anchor_target=anchor, cfg=cfg,
        )
        if loo is None or loo.empty:
            continue
        exc = _lc_excess(loo)
        peaks = []
        for src in loo.get("source_file", pd.Series(dtype=str)).astype(str):
            pr = _star_proc_row(proc_index, src, cid)
            if pr is not None:
                pv = float(pd.to_numeric(pr.get(PEAK_COL), errors="coerce"))
                if math.isfinite(pv):
                    peaks.append(pv)
        cal_rows.append(
            {
                "catalog_id": cid,
                "mag_g": float(pd.to_numeric(row.get("mag_g"), errors="coerce")),
                "median_peak_adu": float(np.median(peaks)) if peaks else float("nan"),
                **exc,
            }
        )

    check_exc = _lc_excess(
        pd.read_csv(next(W1W2_LC_ROOT.glob(f"*/lightcurve_{CHECK_CID}.csv")), low_memory=False)
    )
    check_peaks = []
    for lc_path in W1W2_LC_ROOT.glob(f"*/lightcurve_{CHECK_CID}.csv"):
        lc = pd.read_csv(lc_path, low_memory=False)
        for src in lc["source_file"].astype(str):
            pr = _star_proc_row(proc_index, src, CHECK_CID)
            if pr is not None:
                pv = float(pd.to_numeric(pr.get(PEAK_COL), errors="coerce"))
                if math.isfinite(pv):
                    check_peaks.append(pv)

    return {
        "n_stars": int(len(sdf)),
        "excess_vs_g_bins": _bin_median("mag_g", "excess_mmag"),
        "excess_vs_peak_bins": _bin_median("median_peak_adu", "excess_mmag"),
        "partial_spearman_excess_vs_peak_controlling_g": partial,
        "calibrator_cohort": cal_rows,
        "calibrator_cohort_median_excess_mmag": float(np.median([r["excess_mmag"] for r in cal_rows]))
        if cal_rows
        else float("nan"),
        "check_star_excess_mmag": float(
            math.sqrt(max(0.0, (17.8 / MAG_ERR_SCALE) ** 2 - (9.4 / MAG_ERR_SCALE) ** 2)) * MAG_ERR_SCALE
        ),
        "check_star_median_peak_adu": float(np.median(check_peaks)) if check_peaks else float("nan"),
        "population_median_excess_mmag": float(sdf["excess_mmag"].median()),
    }


def run_a3(*, draft: Path, phot: Path, proc_dir: Path, cfg: AppConfig) -> dict[str, Any]:
    db = sqlite3.connect(str(cfg.database_path))
    tel_rows = db.execute(
        "SELECT ID, TELESCOPENAME, DIAMETER, FOCAL FROM TELESCOPE WHERE TELESCOPENAME LIKE '%Carl-Zeiss%'"
    ).fetchall()
    tel_cols = [d[0] for d in db.execute("PRAGMA table_info(TELESCOPE)").fetchall()]
    db.close()

    fits_files = sorted((draft / "detrended_aligned" / "lights" / SETUP).glob("*.fits"))
    hdr = fits.getheader(fits_files[0])
    xpix = float(hdr.get("XPIXSZ", float("nan")))
    ypix = float(hdr.get("YPIXSZ", float("nan")))
    xbin = int(hdr.get("XBINNING", 1))
    ybin = int(hdr.get("YBINNING", 1))
    scale = float(hdr.get("SCALE", float("nan")))
    focal_hdr = float(hdr.get("FOCALLEN", float("nan")))
    apt_hdr = float(hdr.get("APTDIA", float("nan")))
    pitch_mm = (xpix * xbin) / 1000.0
    f_implied = 206265.0 * pitch_mm / scale if scale > 0 else float("nan")
    f_ratio_implied = f_implied / float(tel_rows[0][2]) if tel_rows and tel_rows[0][2] else float("nan")

    meta = json.loads((phot / "pipeline_meta.json").read_text(encoding="utf-8"))
    rig = resolve_rig_scintillation_params(
        draft_id=DRAFT_ID, setup=SETUP, cfg=cfg, pipeline_meta=meta,
    )
    airmasses: list[float] = []
    for p in sorted(proc_dir.glob("proc_*.csv")):
        df = pd.read_csv(p, usecols=["airmass"], low_memory=False)
        airmasses.extend(pd.to_numeric(df["airmass"], errors="coerce").dropna().tolist())
    am = np.asarray(airmasses, dtype=np.float64)
    am = am[np.isfinite(am) & (am >= 1.0)]

    def _scint_mm(d_m: float, x: float) -> float:
        s = scintillation_sigma(
            telescope_diameter_m=d_m,
            airmass=x,
            exposure_s=float(rig.exposure_s),
            altitude_m=float(rig.altitude_m),
            c_y=float(rig.c_y),
        )
        return float(s * MAG_ERR_SCALE) if math.isfinite(s) else float("nan")

    scint = {}
    for d in (0.200, 0.072):
        for label, q in [("p05", 0.05), ("p50", 0.5), ("p95", 0.95)]:
            scint[f"D{d:.3f}_{label}"] = _scint_mm(d, float(np.quantile(am, q)))

    # err correction on pooled check-star epochs from one representative LC aggregate
    ratios_unc = []
    ratios_corr = []
    for lc_path in W1W2_LC_ROOT.glob(f"*/lightcurve_{CHECK_CID}.csv"):
        lc = pd.read_csv(lc_path, low_memory=False)
        m = pd.to_numeric(lc.get("mag_calib_final"), errors="coerce").to_numpy(dtype=np.float64)
        e = pd.to_numeric(lc.get("err"), errors="coerce").to_numpy(dtype=np.float64)
        am_lc = pd.to_numeric(lc.get("airmass"), errors="coerce").to_numpy(dtype=np.float64)
        ok = np.isfinite(m) & np.isfinite(e) & (e > 0) & np.isfinite(am_lc)
        if int(np.count_nonzero(ok)) < 3:
            continue
        sig_r = _mad_sigma(m[ok])
        e_corr = []
        for amv, ev in zip(am_lc[ok], e[ok], strict=False):
            s02 = _scint_mm(0.200, float(amv)) / MAG_ERR_SCALE
            s007 = _scint_mm(0.072, float(amv)) / MAG_ERR_SCALE
            if not (math.isfinite(s02) and math.isfinite(s007)):
                continue
            e_old = float(ev)
            e_new = math.sqrt(max(0.0, e_old * e_old - s02 * s02 + s007 * s007))
            if e_new > 0:
                e_corr.append(e_new)
        if not e_corr:
            continue
        ratios_unc.append(sig_r / float(np.median(e[ok])))
        ratios_corr.append(sig_r / float(np.median(e_corr)))
    ratio_uncorr_median = float(np.median(ratios_unc)) if ratios_unc else float("nan")
    ratio_corr_median = float(np.median(ratios_corr)) if ratios_corr else float("nan")

    return {
        "telescope_query_columns": tel_cols,
        "telescope_rows": [dict(zip(["ID", "TELESCOPENAME", "DIAMETER", "FOCAL"], r, strict=False)) for r in tel_rows],
        "fits_header": {
            "XPIXSZ_um": xpix,
            "YPIXSZ_um": ypix,
            "XBINNING": xbin,
            "YBINNING": ybin,
            "SCALE_arcsec_per_px": scale,
            "FOCALLEN_mm_header": focal_hdr,
            "APTDIA_mm_header": apt_hdr,
        },
        "pixel_pitch_mm_binned": pitch_mm,
        "implied_focal_length_mm": f_implied,
        "implied_f_ratio_using_db_diameter": f_ratio_implied,
        "airmass_p05_p50_p95": [float(np.quantile(am, q)) for q in (0.05, 0.5, 0.95)],
        "scintillation_mmag": scint,
        "median_sigma_total_robust_over_err": ratio_uncorr_median,
        "median_sigma_total_robust_over_err_corrected_D072": ratio_corr_median,
        "rig": rig.to_dict(),
    }


def main() -> int:
    cfg = AppConfig()
    draft = Path(cfg.archive_root) / "Drafts" / DRAFT_NAME
    phot = draft / "platesolve" / SETUP / "photometry"
    from check_star_kmag import resolve_proc_csv_dir  # noqa: PLC0415

    proc_dir = resolve_proc_csv_dir(phot, SETUP)
    if proc_dir is None:
        raise RuntimeError("proc_dir missing")
    proc_index = _load_proc_index(Path(proc_dir))
    ms = pd.read_csv(
        draft / "platesolve" / SETUP / "masterstars_full_match.csv",
        low_memory=False,
        dtype={"catalog_id": str},
    )

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out = {
        "A1": run_a1(proc_index),
        "A2": run_a2(draft=draft, phot=phot, proc_index=proc_index, ms=ms, cfg=cfg),
        "A3": run_a3(draft=draft, phot=phot, proc_dir=Path(proc_dir), cfg=cfg),
    }
    out_path = OUT_ROOT / "wide_err_a.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="ascii")
    print(json.dumps({"A1_summary": {k: out["A1"][k] for k in out["A1"] if k != "per_field"},
                      "A2_summary": {k: out["A2"][k] for k in out["A2"] if k not in ("calibrator_cohort",)},
                      "A3": out["A3"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
