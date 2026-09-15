# -*- coding: ascii -*-
"""EPSF-VAL-01: PSF-path validation under D-EPSF-XVAL-DOD-02.

Dev-only. Zero new photometry. src_py must not import this module.
Criteria 1 (precision) and 2 (accuracy) on live 516 products.
Criterion 3 (fix list) is post-520 and out of scope here.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "dev" / "xval_psfex") not in sys.path:
    sys.path.insert(0, str(REPO / "dev" / "xval_psfex"))

from epsf_core_03 import (  # noqa: E402
    CHECK_CID,
    ENS_IDS,
    LIVE_ALN,
    LIVE_PS,
    MS_PATH,
    TARGET_CID,
    g4_live_516,
    load_stems,
    rebuild_delta,
)

OUT = REPO / "dev" / "results" / "context" / "session_20260915_epsf_val_01"
EPSF_META = LIVE_PS / "masterstar_epsf_meta.json"

# D-EPSF-XVAL-DOD-02 thresholds at HEAD 6693537 (docs/VYVAR_DECISIONS.md:27-34).
T1A_MEDIAN_RATIO = 1.25
T1B_MAX_RATIO = 1.50
T2B_SLOPE_G_MMAG_PER_MAG = 5.0
T2R_RESIDUAL_RMS_MMAG = 10.0
T2C_SLOPE_BPRP_MMAG_PER_MAG = 10.0
VSX_MATCH_MAX_SEP_ARCSEC = 5.0  # catalog_match.py:89 detect_stars_and_match_catalog
G_CUT_PRIMARY = 11.5
G_CUT_RELAXED = 12.0
N_EPOCH_MIN = 130
N_OK_ACCURACY = 100
G_ACC_LO = 8.5
G_ACC_HI = 12.5


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def rms_med(d: np.ndarray) -> float:
    x = np.asarray(d, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 8:
        return float("nan")
    return float(np.sqrt(np.mean((x - float(np.median(x))) ** 2)))


def _bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).astype(bool)
    return s.astype(str).str.strip().str.lower().isin(("true", "1", "yes", "t"))


def load_all_proc(stems: list[str]) -> pd.DataFrame:
    cols = (
        "catalog_id",
        "dao_flux",
        "psf_flux",
        "psf_fit_ok",
        "psf_chi2",
        "is_saturated",
        "vsx_known_variable",
        "aperture_r_px",
        "aperture_factor_applied",
    )
    rows = []
    for stem in stems:
        path = LIVE_ALN / f"proc_{stem}.csv"
        df = pd.read_csv(path, dtype={"catalog_id": str}, usecols=lambda c: c in cols)
        df["catalog_id"] = df["catalog_id"].astype(str).str.strip()
        df["stem"] = stem
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out["dao"] = pd.to_numeric(out["dao_flux"], errors="coerce")
    out["psf"] = pd.to_numeric(out["psf_flux"], errors="coerce")
    out["chi2"] = pd.to_numeric(out["psf_chi2"], errors="coerce")
    out["fit_ok"] = _bool_series(out["psf_fit_ok"]) if "psf_fit_ok" in out.columns else False
    out["sat"] = _bool_series(out["is_saturated"]) if "is_saturated" in out.columns else False
    out["vsx_row"] = (
        _bool_series(out["vsx_known_variable"]) if "vsx_known_variable" in out.columns else False
    )
    out["ok_ap"] = np.isfinite(out["dao"]) & (out["dao"] > 0)
    out["ok_prod"] = np.isfinite(out["psf"]) & (out["psf"] > 0) & np.isfinite(out["chi2"])
    out["ok_strict"] = out["ok_prod"] & out["fit_ok"]
    return out


def load_masterstars() -> pd.DataFrame:
    ms = pd.read_csv(MS_PATH, dtype={"catalog_id": str}, low_memory=False)
    ms["catalog_id"] = ms["catalog_id"].astype(str).str.strip()
    ms["G"] = pd.to_numeric(ms["phot_g_mean_mag"], errors="coerce")
    ms["bp_rp"] = pd.to_numeric(ms["bp_rp"], errors="coerce")
    ms["vsx"] = _bool_series(ms["vsx_known_variable"]) if "vsx_known_variable" in ms.columns else False
    return ms


def select_constant_stars(
    allp: pd.DataFrame, ms: pd.DataFrame, g_cut: float
) -> tuple[pd.DataFrame, dict]:
    gstat = (
        allp.groupby("catalog_id", as_index=False)
        .agg(
            n_ap=("ok_ap", "sum"),
            n_prod=("ok_prod", "sum"),
            n_strict=("ok_strict", "sum"),
            n_sat=("sat", "sum"),
            n_fit_ok=("fit_ok", "sum"),
        )
    )
    work = gstat.merge(ms[["catalog_id", "G", "bp_rp", "vsx"]], on="catalog_id", how="left")
    exclude = set(ENS_IDS) | {TARGET_CID}
    work["is_historical_check"] = work["catalog_id"] == CHECK_CID
    work["pass_selection"] = (
        (work["G"] <= float(g_cut))
        & (work["n_ap"] >= N_EPOCH_MIN)
        & (work["n_prod"] >= N_EPOCH_MIN)
        & (~work["catalog_id"].isin(exclude))
        & (~_bool_series(work["vsx"]))
        & (work["n_sat"] == 0)
    )
    selected = work[work["pass_selection"]].copy().sort_values(["G", "catalog_id"])
    # Always include historical check in the candidate table.
    chk = work[work["catalog_id"] == CHECK_CID].copy()
    if not chk.empty and CHECK_CID not in set(selected["catalog_id"]):
        table = pd.concat([selected, chk], ignore_index=True)
    else:
        table = selected.copy()
    table = table.sort_values(["pass_selection", "G"], ascending=[False, True])
    meta = {
        "g_cut": float(g_cut),
        "n_selected": int(selected["catalog_id"].nunique()),
        "n_table_rows": int(len(table)),
        "historical_check_passes": bool(chk["pass_selection"].iloc[0]) if not chk.empty else False,
        "vsx_product": (
            f"masterstars_full_match.csv / proc vsx_known_variable "
            f"(catalog_match.detect_stars_and_match_catalog "
            f"vsx_match_max_sep_arcsec={VSX_MATCH_MAX_SEP_ARCSEC}; "
            f"catalog_match.py:89)"
        ),
        "exclude": sorted(exclude),
        "n_epoch_min": N_EPOCH_MIN,
    }
    return table, meta


def _flux_series(allp: pd.DataFrame, cid: str, stems: list[str], col: str, mask_col: str) -> np.ndarray:
    sub = allp[allp["catalog_id"] == cid].set_index("stem")
    out = np.full(len(stems), np.nan, dtype=np.float64)
    for i, stem in enumerate(stems):
        if stem not in sub.index:
            continue
        row = sub.loc[stem]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        if bool(row[mask_col]):
            out[i] = float(row[col])
    return out


def _ens_flux(
    allp: pd.DataFrame, stems: list[str], col: str, mask_col: str
) -> dict[str, np.ndarray]:
    return {eid: _flux_series(allp, eid, stems, col, mask_col) for eid in ENS_IDS}


def precision_table(
    allp: pd.DataFrame,
    stems: list[str],
    selected_ids: list[str],
    *,
    variant: str,
) -> pd.DataFrame:
    mask = "ok_prod" if variant == "V-prod" else "ok_strict"
    ens_psf = _ens_flux(allp, stems, "psf", mask)
    ens_ap = _ens_flux(allp, stems, "dao", "ok_ap")
    rows = []
    for cid in selected_ids:
        psf = _flux_series(allp, cid, stems, "psf", mask)
        ap = _flux_series(allp, cid, stems, "dao", "ok_ap")
        n_psf = int(np.isfinite(psf).sum())
        n_ap = int(np.isfinite(ap).sum())
        d_psf = rebuild_delta(psf, ens_psf)
        d_ap = rebuild_delta(ap, ens_ap)
        # identical epochs: both finite
        both = np.isfinite(d_psf) & np.isfinite(d_ap)
        rms_p = rms_med(d_psf[both]) * 1000.0
        rms_a = rms_med(d_ap[both]) * 1000.0
        ratio = rms_p / rms_a if (math.isfinite(rms_p) and math.isfinite(rms_a) and rms_a > 0) else float("nan")
        rows.append(
            {
                "catalog_id": cid,
                "variant": variant,
                "n_psf_epochs": n_psf,
                "n_ap_epochs": n_ap,
                "n_identical": int(both.sum()),
                "below_100_psf": bool(n_psf < 100),
                "rms_med_psf_mmag": rms_p,
                "rms_med_ap_mmag": rms_a,
                "ratio": ratio,
            }
        )
    return pd.DataFrame(rows)


def accuracy_table(allp: pd.DataFrame, stems: list[str], ms: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    gstat = (
        allp.groupby("catalog_id", as_index=False)
        .agg(n_prod=("ok_prod", "sum"), n_ap=("ok_ap", "sum"))
    )
    work = gstat.merge(ms[["catalog_id", "G", "bp_rp"]], on="catalog_id", how="left")
    pop = work[
        (work["n_prod"] >= N_OK_ACCURACY)
        & (work["n_ap"] >= N_OK_ACCURACY)
        & (work["G"] >= G_ACC_LO)
        & (work["G"] <= G_ACC_HI)
    ].copy()
    rows = []
    for cid in pop["catalog_id"]:
        sub = allp[allp["catalog_id"] == cid]
        ok = sub["ok_prod"] & sub["ok_ap"]
        psf = sub.loc[ok, "psf"].to_numpy(dtype=np.float64)
        ap = sub.loc[ok, "dao"].to_numpy(dtype=np.float64)
        if psf.size < N_OK_ACCURACY:
            continue
        m_psf = -2.5 * np.log10(psf)
        m_ap = -2.5 * np.log10(ap)
        d = m_psf - m_ap
        g = float(pop.loc[pop["catalog_id"] == cid, "G"].iloc[0])
        bprp = float(pop.loc[pop["catalog_id"] == cid, "bp_rp"].iloc[0])
        rows.append(
            {
                "catalog_id": cid,
                "G": g,
                "bp_rp": bprp,
                "n_ok": int(psf.size),
                "d_median_mag": float(np.median(d)),
                "d_median_mmag": float(np.median(d)) * 1000.0,
                "is_target": cid == TARGET_CID,
                "is_check": cid == CHECK_CID,
                "is_ensemble": cid in ENS_IDS,
            }
        )
    acc = pd.DataFrame(rows)
    meta = {
        "population": (
            f"stars with n_ok>={N_OK_ACCURACY} on both paths and "
            f"{G_ACC_LO}<=G<={G_ACC_HI}; includes target, check, ensemble"
        ),
        "n_stars": int(len(acc)),
    }
    return acc, meta


def _theilsen_fit(x: np.ndarray, y: np.ndarray) -> dict:
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(xx) & np.isfinite(yy)
    xx, yy = xx[ok], yy[ok]
    n = int(xx.size)
    if n < 8:
        return {
            "n": n,
            "slope": float("nan"),
            "intercept": float("nan"),
            "residual_rms_mmag": float("nan"),
        }
    slope, intercept, _, _ = stats.theilslopes(yy, xx)
    resid = yy - (intercept + slope * xx)
    return {
        "n": n,
        "slope": float(slope),
        "intercept": float(intercept),
        "residual_rms_mmag": float(np.sqrt(np.mean(resid * resid))),
    }


def accuracy_fits(acc: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # d in mmag for slope units mmag/mag
    y = acc["d_median_mmag"].to_numpy(dtype=np.float64)
    xg = acc["G"].to_numpy(dtype=np.float64) - 10.0
    xb = acc["bp_rp"].to_numpy(dtype=np.float64) - 1.0
    fit_g = _theilsen_fit(xg, y)
    fit_c = _theilsen_fit(xb, y)
    fit_g["axis"] = "G-10"
    fit_c["axis"] = "BP-RP-1.0"
    fit_g["criterion_limit_abs"] = T2B_SLOPE_G_MMAG_PER_MAG
    fit_c["criterion_limit_abs"] = T2C_SLOPE_BPRP_MMAG_PER_MAG
    fit_g["residual_limit_mmag"] = T2R_RESIDUAL_RMS_MMAG
    fits = pd.DataFrame([fit_g, fit_c])

    # 0.5 mag bins in G
    bins = np.arange(G_ACC_LO, G_ACC_HI + 0.5, 0.5)
    acc = acc.copy()
    acc["G_bin"] = pd.cut(acc["G"], bins=bins, right=False)
    binned = (
        acc.groupby("G_bin", observed=False)
        .agg(
            n=("catalog_id", "count"),
            G_mid=("G", "mean"),
            d_median_mmag=("d_median_mmag", "median"),
            d_mean_mmag=("d_median_mmag", "mean"),
        )
        .reset_index()
    )
    binned["G_bin"] = binned["G_bin"].astype(str)
    return fits, binned


def readings(prec_prod: pd.DataFrame, prec_strict: pd.DataFrame, fits: pd.DataFrame) -> list[str]:
    fired = []
    r = pd.to_numeric(prec_prod["ratio"], errors="coerce")
    r = r[np.isfinite(r)]
    med = float(np.median(r)) if len(r) else float("nan")
    mx = float(np.max(r)) if len(r) else float("nan")
    if math.isfinite(med) and math.isfinite(mx) and med <= T1A_MEDIAN_RATIO and mx <= T1B_MAX_RATIO:
        fired.append(
            f"R-V1 PASS (V-prod): median r={med:.3f} <= {T1A_MEDIAN_RATIO} AND max r={mx:.3f} <= {T1B_MAX_RATIO}."
        )
    else:
        offenders = prec_prod[pd.to_numeric(prec_prod["ratio"], errors="coerce") > T1B_MAX_RATIO]
        names = ",".join(offenders["catalog_id"].astype(str).tolist()[:12])
        fired.append(
            f"R-V1 FAIL (V-prod): median r={med:.3f} max r={mx:.3f} "
            f"(limits {T1A_MEDIAN_RATIO}/{T1B_MAX_RATIO}); offenders>{T1B_MAX_RATIO}: {names or 'none'}."
        )
    rs = pd.to_numeric(prec_strict["ratio"], errors="coerce")
    rs = rs[np.isfinite(rs)]
    if len(rs):
        fired.append(
            f"V-strict informational: median r={float(np.median(rs)):.3f} max r={float(np.max(rs)):.3f} "
            f"n_stars_with_ratio={len(rs)} n_below_100={int(prec_strict['below_100_psf'].sum())}."
        )
    else:
        fired.append("V-strict informational: no star with finite ratio (all n_psf < usable).")

    fg = fits[fits["axis"] == "G-10"].iloc[0]
    fc = fits[fits["axis"] == "BP-RP-1.0"].iloc[0]
    b = abs(float(fg["slope"]))
    res = float(fg["residual_rms_mmag"])
    c = abs(float(fc["slope"]))
    if (
        math.isfinite(b)
        and math.isfinite(res)
        and math.isfinite(c)
        and b <= T2B_SLOPE_G_MMAG_PER_MAG
        and res <= T2R_RESIDUAL_RMS_MMAG
        and c <= T2C_SLOPE_BPRP_MMAG_PER_MAG
    ):
        fired.append(
            f"R-V2 PASS: |b|={b:.3f} <= {T2B_SLOPE_G_MMAG_PER_MAG}, "
            f"resid RMS={res:.3f} <= {T2R_RESIDUAL_RMS_MMAG}, "
            f"|c|={c:.3f} <= {T2C_SLOPE_BPRP_MMAG_PER_MAG}."
        )
    else:
        fired.append(
            f"R-V2 FAIL: |b|={b:.3f} (lim {T2B_SLOPE_G_MMAG_PER_MAG}), "
            f"resid RMS={res:.3f} (lim {T2R_RESIDUAL_RMS_MMAG}), "
            f"|c|={c:.3f} (lim {T2C_SLOPE_BPRP_MMAG_PER_MAG})."
        )

    v1_pass = any(x.startswith("R-V1 PASS") for x in fired)
    v2_pass = any(x.startswith("R-V2 PASS") for x in fired)
    if v1_pass and v2_pass:
        fired.append(
            "R-V3: both PASS; criteria 1 and 2 of DOD-02 are met on 516; "
            "EPSF-XVAL-01 closure waits only on criterion 3 at the 520 re-cut."
        )
    else:
        fails = []
        if not v1_pass:
            fails.append("criterion 1 (precision)")
        if not v2_pass:
            fails.append("criterion 2 (accuracy)")
        fired.append(f"R-V3: FAIL on {', '.join(fails)}; sequencing is Milan's.")
    return fired


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    stems = load_stems()
    print(f"[val01] n_stems={len(stems)}")
    allp = load_all_proc(stems)
    ms = load_masterstars()

    table, sel_meta = select_constant_stars(allp, ms, G_CUT_PRIMARY)
    g_cut_used = G_CUT_PRIMARY
    if sel_meta["n_selected"] < 4:
        print(f"[val01] only {sel_meta['n_selected']} at G<={G_CUT_PRIMARY}; relax to {G_CUT_RELAXED}")
        table, sel_meta = select_constant_stars(allp, ms, G_CUT_RELAXED)
        g_cut_used = G_CUT_RELAXED
        sel_meta["relaxed"] = True
    else:
        sel_meta["relaxed"] = False
    if sel_meta["n_selected"] < 4:
        summary = {
            "stop": True,
            "reason": f"fewer than 4 constant stars after G<={G_CUT_RELAXED}",
            "selection": sel_meta,
            "thresholds_quoted": {
                "T1a_median_ratio": T1A_MEDIAN_RATIO,
                "T1b_max_ratio": T1B_MAX_RATIO,
                "T2b_slope_G_mmag_per_mag": T2B_SLOPE_G_MMAG_PER_MAG,
                "T2r_residual_rms_mmag": T2R_RESIDUAL_RMS_MMAG,
                "T2c_slope_bprp_mmag_per_mag": T2C_SLOPE_BPRP_MMAG_PER_MAG,
                "source": "D-EPSF-XVAL-DOD-02 at HEAD (docs/VYVAR_DECISIONS.md:27-34)",
            },
            "g4": g4_live_516(),
        }
        table.to_csv(OUT / "constant_star_candidates.csv", index=False)
        (OUT / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
        print("[val01] STOP", summary["reason"])
        return 2

    cand_cols = [
        "catalog_id",
        "G",
        "bp_rp",
        "n_ap",
        "n_prod",
        "n_strict",
        "n_fit_ok",
        "n_sat",
        "vsx",
        "pass_selection",
        "is_historical_check",
    ]
    table[cand_cols].to_csv(OUT / "constant_star_candidates.csv", index=False)
    print(f"[val01] selected={sel_meta['n_selected']} g_cut={g_cut_used}")

    selected_ids = table.loc[table["pass_selection"], "catalog_id"].astype(str).tolist()
    # Ensure historical check is in precision table even if it somehow failed (it should pass).
    if CHECK_CID not in selected_ids and sel_meta["historical_check_passes"]:
        selected_ids.append(CHECK_CID)

    prec_prod = precision_table(allp, stems, selected_ids, variant="V-prod")
    prec_strict = precision_table(allp, stems, selected_ids, variant="V-strict")
    prec = pd.concat([prec_prod, prec_strict], ignore_index=True)
    prec.to_csv(OUT / "precision_per_star.csv", index=False)
    print(
        "[val01] precision V-prod",
        float(np.nanmedian(prec_prod["ratio"])),
        float(np.nanmax(prec_prod["ratio"])),
    )

    acc, acc_meta = accuracy_table(allp, stems, ms)
    acc.to_csv(OUT / "accuracy_per_star.csv", index=False)
    fits, binned = accuracy_fits(acc)
    fits.to_csv(OUT / "accuracy_fits.csv", index=False)
    binned.to_csv(OUT / "accuracy_g_bins.csv", index=False)
    print("[val01] accuracy", fits.to_dict("records"))

    meta_epsf = json.loads(EPSF_META.read_text(encoding="utf-8")) if EPSF_META.is_file() else {}
    ap_r = float(pd.to_numeric(allp.get("aperture_r_px"), errors="coerce").median())
    ap_pol = (
        str(allp["aperture_factor_applied"].dropna().astype(str).mode().iloc[0])
        if "aperture_factor_applied" in allp.columns and allp["aperture_factor_applied"].notna().any()
        else "unknown"
    )

    fired = readings(prec_prod, prec_strict, fits)
    print("[val01] readings", fired)
    g4 = g4_live_516()
    summary = {
        "thresholds_quoted_from_HEAD": {
            "decision": "D-EPSF-XVAL-DOD-02",
            "file": "docs/VYVAR_DECISIONS.md",
            "commit": "6693537",
            "precision": (
                "RMS_med(PSF diff LC)/RMS_med(aperture diff LC) median <= 1.25 "
                "and no star > 1.50 (lines 27-30)"
            ),
            "accuracy": (
                "d=a+b*(G-10) with |b|<=5.0 mmag/mag and residual RMS<=10 mmag; "
                "vs BP-RP with |c|<=10 mmag/mag (lines 31-34)"
            ),
            "T1a_median_ratio": T1A_MEDIAN_RATIO,
            "T1b_max_ratio": T1B_MAX_RATIO,
            "T2b_slope_G_mmag_per_mag": T2B_SLOPE_G_MMAG_PER_MAG,
            "T2r_residual_rms_mmag": T2R_RESIDUAL_RMS_MMAG,
            "T2c_slope_bprp_mmag_per_mag": T2C_SLOPE_BPRP_MMAG_PER_MAG,
            "note": "Criterion 3 (CODE / fix list) is out of scope for this task.",
        },
        "selection": sel_meta,
        "g_cut_used": g_cut_used,
        "precision": {
            "population": (
                f"{len(selected_ids)} selected constant stars; pinned 4-star AIJ flux-sum; "
                f"{len(stems)} identical-ensemble epochs; RMS_med mmag"
            ),
            "V-prod": {
                "median_ratio": float(np.nanmedian(prec_prod["ratio"])),
                "max_ratio": float(np.nanmax(prec_prod["ratio"])),
                "n_stars": int(len(prec_prod)),
            },
            "V-strict": {
                "median_ratio": float(np.nanmedian(prec_strict["ratio"])) if prec_strict["ratio"].notna().any() else None,
                "max_ratio": float(np.nanmax(prec_strict["ratio"])) if prec_strict["ratio"].notna().any() else None,
                "n_stars_finite_ratio": int(pd.to_numeric(prec_strict["ratio"], errors="coerce").notna().sum()),
                "n_below_100": int(prec_strict["below_100_psf"].sum()),
            },
        },
        "accuracy": {
            **acc_meta,
            "fits": fits.to_dict(orient="records"),
            "g_bins": binned.to_dict(orient="records"),
            "aperture_convention": (
                f"dao_flux; aperture_factor_applied={ap_pol}; "
                f"median aperture_r_px={ap_r:.4g}; annulus APERTURE-01d 2.7/5.2 FWHM"
            ),
            "psf_convention": (
                f"psf_flux from live ePSF ImagePSF; epsf_sum_native="
                f"{meta_epsf.get('epsf_sum_native', 1.0)} "
                f"(production normalize sum=osamp^2 so native sum=1; "
                f"psf_photometry.py:649-661). Constant a absorbs aperture vs PSF "
                f"scale and is not a criterion."
            ),
        },
        "readings": fired,
        "g4": g4,
    }
    (OUT / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    print("[val01] g4", g4)
    return 0


if __name__ == "__main__":
    sys.exit(main())
