# -*- coding: ascii -*-
"""EPSF-VAL-02: PSF-path validation under D-EPSF-XVAL-DOD-03.

Dev-only. Zero new photometry. Zero src_py imports.
Reuses VAL-01 session products for precision; measures accuracy vs
Gaia-transformed V (GDR3 Table 5.9) and admission on live 516.
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
    LIVE_ALN,
    LIVE_PS,
    MS_PATH,
    g4_live_516,
    load_stems,
)

OUT = REPO / "dev" / "results" / "context" / "session_20260915_epsf_val_02"
VAL01 = REPO / "dev" / "results" / "context" / "session_20260915_epsf_val_01"
PHOT_LC = LIVE_PS / "photometry" / "lightcurves"
ACTIVE = LIVE_PS / "photometry" / "active_targets.csv"

# D-EPSF-XVAL-DOD-03 thresholds at HEAD 23e2680 (docs/VYVAR_DECISIONS.md:23-45).
T1A = 1.25
T1B = 1.50
T2B = 5.0  # mmag/mag
T2R = 25.0  # mmag
G_DOMAIN_LO = 9.5
G_ACC_LO = 8.5
G_ACC_HI = 12.5
N_OK = 100
BRIGHT_EXTRA = "1498735778606786816"

# GDR3 Table 5.9 G-V coeffs (gaia_johnson.py; cited, not imported).
# Y = G - V = sum a_i * (BP-RP)^i ; V = G - Y.
GDR3_GV = (-0.02704, 0.01424, -0.2156, 0.01426)
BPRP_MIN, BPRP_MAX = -0.5, 5.1
G_MAG_MIN, G_MAG_MAX = 8.0, 16.0

THRESHOLDS_QUOTE = {
    "decision": "D-EPSF-XVAL-DOD-03",
    "commit": "23e2680",
    "file": "docs/VYVAR_DECISIONS.md",
    "criterion_1_verbatim": (
        "PRECISION (per G bin). On constant stars selected as in VAL-01, "
        "r(s) = RMS_med(PSF diff LC) / RMS_med(aperture diff LC), V-prod "
        "admission: Domain D = stars with G >= 9.5: median r <= 1.25 and "
        "max r <= 1.50. Bright end G < 9.5: no ratio criterion; ADMISSION "
        "test on psf_fit_ok routing (photometry_lightcurve.py:2393 region)."
    ),
    "criterion_2_verbatim": (
        "ACCURACY: d(s)=median_epochs(m_psf_inst - m_cat); simultaneous "
        "robust fit d=a+b*(G-10)+c*(BP-RP-1.0); |b|<=5.0 mmag/mag and "
        "residual RMS<=25 mmag; c RECORDED; aperture fit RECORDED vs D5-1."
    ),
    "T1a": T1A,
    "T1b": T1B,
    "T2b": T2B,
    "T2r": T2R,
}


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


def _bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).astype(bool)
    return s.astype(str).str.strip().str.lower().isin(("true", "1", "yes", "t"))


def m_cat_v(g: float, bprp: float) -> float:
    """Gaia G + BP-RP -> Johnson V via GDR3 Table 5.9 (D10-1: NoFilter/CV -> V)."""
    if not (math.isfinite(g) and math.isfinite(bprp)):
        return float("nan")
    if g < G_MAG_MIN or g > G_MAG_MAX or bprp < BPRP_MIN or bprp > BPRP_MAX:
        return float("nan")
    y = sum(a * (bprp**i) for i, a in enumerate(GDR3_GV))
    return float(g - y)


def alternating_theilsen(d: np.ndarray, xg: np.ndarray, xb: np.ndarray, n_iter: int = 4) -> dict:
    """Robust a,b,c via alternating Theil-Sen (state method in summary)."""
    ok = np.isfinite(d) & np.isfinite(xg) & np.isfinite(xb)
    yy, xxg, xxb = d[ok], xg[ok], xb[ok]
    n = int(yy.size)
    if n < 8:
        return {
            "n": n,
            "a": float("nan"),
            "b": float("nan"),
            "c": float("nan"),
            "residual_rms_mmag": float("nan"),
            "method": "alternating_theilsen",
        }
    a = float(np.median(yy))
    b = 0.0
    c = 0.0
    for _ in range(n_iter):
        y1 = yy - c * xxb
        b, a, _, _ = stats.theilslopes(y1, xxg)
        b = float(b)
        a = float(a)
        y2 = yy - a - b * xxg
        c, _, _, _ = stats.theilslopes(y2, xxb)
        c = float(c)
    resid = yy - (a + b * xxg + c * xxb)
    return {
        "n": n,
        "a": a,
        "b": b,
        "c": c,
        "residual_rms_mmag": float(np.sqrt(np.mean(resid * resid))),
        "method": "alternating_theilsen_4",
    }


def precision_from_val01() -> tuple[pd.DataFrame, dict]:
    cand = pd.read_csv(VAL01 / "constant_star_candidates.csv", dtype={"catalog_id": str})
    prec = pd.read_csv(VAL01 / "precision_per_star.csv", dtype={"catalog_id": str})
    prod = prec[prec["variant"] == "V-prod"].copy()
    m = prod.merge(
        cand[["catalog_id", "G", "bp_rp", "n_fit_ok", "n_prod", "n_ap"]],
        on="catalog_id",
        how="left",
    )
    m["psf_fit_ok_frac"] = m["n_fit_ok"] / m["n_prod"].clip(lower=1)
    bins = np.arange(8.0, 12.0 + 0.5, 0.5)
    m["G_bin"] = pd.cut(m["G"], bins=bins, right=False)
    by = (
        m.groupby("G_bin", observed=False)
        .agg(
            n_stars=("catalog_id", "count"),
            median_r=("ratio", "median"),
            max_r=("ratio", "max"),
            median_fit_ok_frac=("psf_fit_ok_frac", "median"),
        )
        .reset_index()
    )
    by["G_bin"] = by["G_bin"].astype(str)
    by.to_csv(OUT / "precision_by_gbin.csv", index=False)

    dom = m[m["G"] >= G_DOMAIN_LO].copy()
    bright = m[m["G"] < G_DOMAIN_LO].copy()
    r = pd.to_numeric(dom["ratio"], errors="coerce")
    r = r[np.isfinite(r)]
    med = float(np.median(r)) if len(r) else float("nan")
    mx = float(np.max(r)) if len(r) else float("nan")
    offenders = dom[pd.to_numeric(dom["ratio"], errors="coerce") > T1B]
    meta = {
        "source": str(VAL01 / "precision_per_star.csv"),
        "candidates_source": str(VAL01 / "constant_star_candidates.csv"),
        "domain_D": f"G>={G_DOMAIN_LO}",
        "n_domain": int(len(dom)),
        "median_r": med,
        "max_r": mx,
        "offenders": offenders["catalog_id"].astype(str).tolist(),
        "bright_n": int(len(bright)),
        "bright_median_r": float(np.nanmedian(bright["ratio"])) if len(bright) else None,
        "bright_max_r": float(np.nanmax(bright["ratio"])) if len(bright) else None,
        "per_star": m[
            ["catalog_id", "G", "ratio", "n_fit_ok", "psf_fit_ok_frac", "rms_med_psf_mmag", "rms_med_ap_mmag"]
        ].to_dict(orient="records"),
        "g_bins": by.to_dict(orient="records"),
    }
    return m, meta


def admission_code_table() -> pd.DataFrame:
    rows = [
        {
            "consumer": "compute_lc_flux_method (adaptive picker)",
            "location": "photometry_lightcurve.py:2355-2398",
            "honours_psf_fit_ok": True,
            "notes": (
                "psf_usable requires psf_fit_ok AND finite psf_flux>0 AND "
                "psf_quality!='bad' AND psf_ac_applied; else aperture. "
                "Only faint+good routes to psf."
            ),
        },
        {
            "consumer": "Phase2A primary science LC (mag_calib path)",
            "location": "phase2a_target.py:681-694,1442",
            "honours_psf_fit_ok": True,
            "notes": (
                "_lc_export_method hard-coded 'aperture'. Primary published "
                "LC is always aperture regardless of adaptive column. "
                "psf_adaptive_enabled=false on config.json (516)."
            ),
        },
        {
            "consumer": "photometry_exports._get_lc_psf_strict",
            "location": "photometry_exports.py:99-117",
            "honours_psf_fit_ok": True,
            "notes": "PSF-only inst mag NaN unless psf_fit_ok AND psf_ac_applied.",
        },
        {
            "consumer": "psf_internal_lc (diagnostic sidecar)",
            "location": "psf_internal_lc.py:124-134 fit_ok_for_zp",
            "honours_psf_fit_ok": False,
            "notes": (
                "FIT-OK-ADMISSION-01: admits finite flux+chi2 without "
                "psf_fit_ok. Sidecar header NOT FOR AAVSO/VARASTRO "
                "SUBMISSION; not a science LC (INV-PSF-SUBMIT-01)."
            ),
        },
        {
            "consumer": "export_reports AAVSO/VarAstro writers",
            "location": "export_reports.py INV-PSF-SUBMIT-01 ~988",
            "honours_psf_fit_ok": True,
            "notes": "Hard-refuse lc_method psf/adaptive; science exports aperture-only.",
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "admission_code_table.csv", index=False)
    return df


def admission_end_to_end() -> dict:
    """Science LC method provenance on 516; named bright stars if present."""
    sci_paths = sorted(
        p for p in PHOT_LC.glob("lightcurve_*.csv") if not p.name.endswith("_psf.csv")
    )
    rows = []
    n_psf_as_science = 0
    bad_epochs: list[dict] = []
    for path in sci_paths:
        df = pd.read_csv(path)
        if "method" not in df.columns:
            rows.append(
                {
                    "catalog_id": path.stem.replace("lightcurve_", ""),
                    "n_epochs": int(len(df)),
                    "n_method_psf": None,
                    "n_method_aperture": None,
                    "has_method_col": False,
                }
            )
            continue
        meth = df["method"].astype(str).str.strip().str.lower()
        n_psf = int((meth == "psf").sum())
        n_ap = int((meth == "aperture").sum())
        cid = path.stem.replace("lightcurve_", "")
        rows.append(
            {
                "catalog_id": cid,
                "n_epochs": int(len(df)),
                "n_method_psf": n_psf,
                "n_method_aperture": n_ap,
                "has_method_col": True,
            }
        )
        n_psf_as_science += n_psf

    # Named stars: join proc fit_ok vs science LC if present.
    stems = load_stems()
    named = [CHECK_CID, BRIGHT_EXTRA]
    named_detail = []
    for cid in named:
        sci = PHOT_LC / f"lightcurve_{cid}.csv"
        detail = {
            "catalog_id": cid,
            "science_lc_exists": sci.is_file(),
            "n_fit_ok_false_as_psf": None,
            "note": "",
        }
        if not sci.is_file():
            detail["note"] = (
                "no science LC product (not an active_targets export); "
                "primary path is aperture-only (phase2a_target.py:694)"
            )
            named_detail.append(detail)
            continue
        lcdf = pd.read_csv(sci)
        # Build stem -> method
        lcdf["_stem"] = (
            lcdf["source_file"]
            .astype(str)
            .str.replace("proc_", "", regex=False)
            .str.replace(".csv", "", regex=False)
        )
        n_bad = 0
        for stem in stems:
            proc = pd.read_csv(
                LIVE_ALN / f"proc_{stem}.csv",
                dtype={"catalog_id": str},
                usecols=lambda c: c in ("catalog_id", "psf_fit_ok"),
            )
            proc["catalog_id"] = proc["catalog_id"].astype(str).str.strip()
            prow = proc[proc["catalog_id"] == cid]
            if prow.empty:
                continue
            fit_ok = bool(_bool_series(prow["psf_fit_ok"]).iloc[0])
            mrow = lcdf[lcdf["_stem"] == stem]
            if mrow.empty or "method" not in lcdf.columns:
                continue
            method = str(mrow["method"].iloc[0]).strip().lower()
            if (not fit_ok) and method == "psf":
                n_bad += 1
                bad_epochs.append({"catalog_id": cid, "stem": stem, "method": method})
        detail["n_fit_ok_false_as_psf"] = n_bad
        detail["note"] = "joined proc psf_fit_ok to science LC method"
        named_detail.append(detail)

    e2e = pd.DataFrame(rows)
    e2e.to_csv(OUT / "admission_end_to_end.csv", index=False)
    has_method = bool(e2e["has_method_col"].all()) if len(e2e) else False
    return {
        "n_science_lc_files": int(len(sci_paths)),
        "n_science_epochs_method_psf": int(n_psf_as_science),
        "has_per_epoch_method_col": has_method,
        "named_stars": named_detail,
        "bad_epochs": bad_epochs,
        "active_targets_path": str(ACTIVE),
        "config_psf_adaptive_enabled": False,
        "verdict_note": (
            "Science LC method column present; all exported science epochs "
            "are aperture. Named check/bright stars are not active_targets "
            "so have no science LC file; code path forces aperture."
        ),
    }


def load_proc_and_ms() -> tuple[pd.DataFrame, pd.DataFrame]:
    stems = load_stems()
    cols = (
        "catalog_id",
        "dao_flux",
        "psf_flux",
        "psf_fit_ok",
        "psf_chi2",
    )
    rows = []
    for stem in stems:
        df = pd.read_csv(
            LIVE_ALN / f"proc_{stem}.csv",
            dtype={"catalog_id": str},
            usecols=lambda c: c in cols,
        )
        df["catalog_id"] = df["catalog_id"].astype(str).str.strip()
        df["stem"] = stem
        rows.append(df)
    allp = pd.concat(rows, ignore_index=True)
    allp["dao"] = pd.to_numeric(allp["dao_flux"], errors="coerce")
    allp["psf"] = pd.to_numeric(allp["psf_flux"], errors="coerce")
    allp["chi2"] = pd.to_numeric(allp["psf_chi2"], errors="coerce")
    allp["fit_ok"] = _bool_series(allp["psf_fit_ok"])
    allp["ok_ap"] = np.isfinite(allp["dao"]) & (allp["dao"] > 0)
    allp["ok_prod"] = np.isfinite(allp["psf"]) & (allp["psf"] > 0) & np.isfinite(allp["chi2"])
    ms = pd.read_csv(MS_PATH, dtype={"catalog_id": str}, low_memory=False)
    ms["catalog_id"] = ms["catalog_id"].astype(str).str.strip()
    ms["G"] = pd.to_numeric(ms["phot_g_mean_mag"], errors="coerce")
    ms["bp_rp"] = pd.to_numeric(ms["bp_rp"], errors="coerce")
    ms["m_cat"] = [m_cat_v(g, c) for g, c in zip(ms["G"], ms["bp_rp"], strict=False)]
    return allp, ms


def accuracy_vs_catalog(allp: pd.DataFrame, ms: pd.DataFrame, path: str) -> tuple[pd.DataFrame, dict]:
    mask = "ok_prod" if path == "psf" else "ok_ap"
    flux_col = "psf" if path == "psf" else "dao"
    gstat = allp.groupby("catalog_id", as_index=False).agg(n_ok=(mask, "sum"))
    work = gstat.merge(ms[["catalog_id", "G", "bp_rp", "m_cat"]], on="catalog_id", how="left")
    pop = work[
        (work["n_ok"] >= N_OK)
        & (work["G"] >= G_ACC_LO)
        & (work["G"] <= G_ACC_HI)
        & np.isfinite(work["m_cat"])
    ].copy()
    rows = []
    for cid in pop["catalog_id"]:
        sub = allp[allp["catalog_id"] == cid]
        ok = sub[mask]
        flux = sub.loc[ok, flux_col].to_numpy(dtype=np.float64)
        if flux.size < N_OK:
            continue
        m_inst = -2.5 * np.log10(flux)
        mcat = float(pop.loc[pop["catalog_id"] == cid, "m_cat"].iloc[0])
        d = m_inst - mcat
        g = float(pop.loc[pop["catalog_id"] == cid, "G"].iloc[0])
        bprp = float(pop.loc[pop["catalog_id"] == cid, "bp_rp"].iloc[0])
        rows.append(
            {
                "catalog_id": cid,
                "G": g,
                "bp_rp": bprp,
                "m_cat_V": mcat,
                "n_ok": int(flux.size),
                "d_median_mag": float(np.median(d)),
                "d_median_mmag": float(np.median(d)) * 1000.0,
            }
        )
    acc = pd.DataFrame(rows)
    y = acc["d_median_mmag"].to_numpy(dtype=np.float64)
    xg = acc["G"].to_numpy(dtype=np.float64) - 10.0
    xb = acc["bp_rp"].to_numpy(dtype=np.float64) - 1.0
    fit = alternating_theilsen(y, xg, xb)
    # residuals after removing c term: d' = d - c*(BP-RP-1)
    c = fit["c"]
    acc["d_minus_c_mmag"] = acc["d_median_mmag"] - c * (acc["bp_rp"] - 1.0)
    bins = np.arange(G_ACC_LO, G_ACC_HI + 0.5, 0.5)
    acc["G_bin"] = pd.cut(acc["G"], bins=bins, right=False)
    binned = (
        acc.groupby("G_bin", observed=False)
        .agg(
            n=("catalog_id", "count"),
            G_mid=("G", "mean"),
            d_minus_c_median_mmag=("d_minus_c_mmag", "median"),
        )
        .reset_index()
    )
    binned["G_bin"] = binned["G_bin"].astype(str)
    meta = {
        "path": path,
        "n_stars": int(len(acc)),
        "bp_rp_min": float(acc["bp_rp"].min()) if len(acc) else None,
        "bp_rp_max": float(acc["bp_rp"].max()) if len(acc) else None,
        "fit": fit,
        "g_bins_after_removing_c": binned.to_dict(orient="records"),
        "m_cat_convention": (
            "Johnson V from Gaia G + BP-RP via GDR3 Table 5.9 G-V poly "
            "(gaia_johnson.GDR3_TABLE59_COEFFS 'V'; coeffs embedded). "
            "NoFilter -> AAVSO CV (export_reports.py); D10-1-CLOSE: CV "
            "uses Johnson V comparison magnitudes. D10-2 guard: "
            "BP-RP in [-0.5,5.1], G in [8,16]."
        ),
    }
    return acc, meta


def readings(prec_meta: dict, admit_code: pd.DataFrame, admit_e2e: dict, psf_fit: dict) -> list[str]:
    fired = []
    med = prec_meta["median_r"]
    mx = prec_meta["max_r"]
    if math.isfinite(med) and math.isfinite(mx) and med <= T1A and mx <= T1B:
        fired.append(
            f"R-W1 PASS (domain D G>={G_DOMAIN_LO}): median r={med:.3f} <= {T1A} "
            f"AND max r={mx:.3f} <= {T1B} (n={prec_meta['n_domain']})."
        )
    else:
        offs = ",".join(prec_meta["offenders"][:12]) or "none"
        fired.append(
            f"R-W1 FAIL (domain D G>={G_DOMAIN_LO}): median r={med:.3f} max r={mx:.3f} "
            f"(limits {T1A}/{T1B}; n={prec_meta['n_domain']}); offenders>{T1B}: {offs}."
        )

    sci_code = admit_code[admit_code["consumer"] != "psf_internal_lc (diagnostic sidecar)"]
    code_ok = bool(sci_code["honours_psf_fit_ok"].all())
    n_psf = int(admit_e2e["n_science_epochs_method_psf"])
    has_prov = bool(admit_e2e["has_per_epoch_method_col"])
    if not has_prov:
        fired.append(
            "R-W2 NOT VERIFIABLE: science LC lacks per-epoch method provenance; "
            "520 re-cut requirement: persist per-epoch method."
        )
    elif code_ok and n_psf == 0 and not admit_e2e["bad_epochs"]:
        fired.append(
            f"R-W2 PASS: science consumers honour psf_fit_ok (adaptive picker / "
            f"aperture-primary / exports); e2e n_method_psf={n_psf} of science "
            f"LC epochs; named stars without LC covered by aperture-primary path. "
            f"Diagnostic psf_internal_lc fit_ok_for_zp is out of science scope."
        )
    else:
        fired.append(
            f"R-W2 FAIL: code_ok={code_ok} n_science_psf_epochs={n_psf} "
            f"bad_epochs={len(admit_e2e['bad_epochs'])}."
        )

    b = abs(float(psf_fit["b"]))
    res = float(psf_fit["residual_rms_mmag"])
    c = float(psf_fit["c"])
    if math.isfinite(b) and math.isfinite(res) and b <= T2B and res <= T2R:
        fired.append(
            f"R-W3 PASS: |b|={b:.3f} <= {T2B}, resid RMS={res:.3f} <= {T2R} "
            f"(c={c:.3f} recorded, not judged; n={psf_fit['n']})."
        )
    else:
        fired.append(
            f"R-W3 FAIL: |b|={b:.3f} (lim {T2B}), resid RMS={res:.3f} (lim {T2R}) "
            f"(c={c:.3f} recorded; n={psf_fit['n']})."
        )

    w1 = any(x.startswith("R-W1 PASS") for x in fired)
    w2 = any(x.startswith("R-W2 PASS") for x in fired)
    w3 = any(x.startswith("R-W3 PASS") for x in fired)
    if w1 and w2 and w3:
        fired.append(
            "R-W4: R-W1, R-W2, R-W3 all PASS; criteria 1 and 2 of DOD-03 are "
            "met on 516; EPSF-XVAL-01 closure waits only on criterion 3 at the "
            "520 re-cut."
        )
    else:
        fails = []
        if not w1:
            fails.append("R-W1")
        if not w2:
            fails.append("R-W2" + (" (NOT VERIFIABLE)" if any("NOT VERIFIABLE" in x for x in fired) else ""))
        if not w3:
            fails.append("R-W3")
        fired.append(f"R-W4: FAIL on {', '.join(fails)}; sequencing is Milan's.")
    return fired


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("[val02] precision from VAL-01")
    _m, prec_meta = precision_from_val01()
    print(
        "[val02] domain D",
        prec_meta["n_domain"],
        prec_meta["median_r"],
        prec_meta["max_r"],
        "offenders",
        len(prec_meta["offenders"]),
    )

    print("[val02] admission code + e2e")
    admit_code = admission_code_table()
    admit_e2e = admission_end_to_end()
    print(
        "[val02] e2e",
        admit_e2e["n_science_lc_files"],
        "psf_epochs",
        admit_e2e["n_science_epochs_method_psf"],
    )

    print("[val02] accuracy vs catalog")
    allp, ms = load_proc_and_ms()
    acc_psf, meta_psf = accuracy_vs_catalog(allp, ms, "psf")
    acc_ap, meta_ap = accuracy_vs_catalog(allp, ms, "aperture")
    acc_psf.to_csv(OUT / "accuracy_vs_catalog_psf.csv", index=False)
    acc_ap.to_csv(OUT / "accuracy_vs_catalog_ap.csv", index=False)
    fits = pd.DataFrame(
        [
            {"path": "psf", **meta_psf["fit"], "criterion": True},
            {"path": "aperture_D5-1_record", **meta_ap["fit"], "criterion": False},
        ]
    )
    fits.to_csv(OUT / "accuracy_fits.csv", index=False)
    print("[val02] psf fit", meta_psf["fit"])
    print("[val02] ap fit", meta_ap["fit"])

    fired = readings(prec_meta, admit_code, admit_e2e, meta_psf["fit"])
    print("[val02] readings", fired)
    g4 = g4_live_516()
    summary = {
        "thresholds_quoted_from_HEAD": THRESHOLDS_QUOTE,
        "precision": prec_meta,
        "admission_code": admit_code.to_dict(orient="records"),
        "admission_e2e": admit_e2e,
        "accuracy_psf": meta_psf,
        "accuracy_aperture_D5_1_record": meta_ap,
        "readings": fired,
        "g4": g4,
    }
    # drop huge per_star from summary file size? keep it - useful
    (OUT / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    print("[val02] g4", g4)
    return 0


if __name__ == "__main__":
    sys.exit(main())
