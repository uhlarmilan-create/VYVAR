"""Measure c_col / c_dist for COMP-WEIGHT-COEFF-01 on drafts 512/513/435.

Ordering note: run after FORCED-PHOT-01 code lands. This harness uses existing
proc products (pre-forced rebuild). Re-verify c_dist after a full forced rebuild.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src_py"))

from comp_weights import (  # noqa: E402
    C_COL_PSF_BPRP_SPAN,
    C_COL_PSF_EE_MMAG,
    C_COL_PSF_REFRACTIVE_MAG_PER_BPRP,
    resolve_comp_weight_coeffs,
    sigma_eff_mag,
    weight_from_sigma_eff,
)


def _find_draft_phot(draft_id: int) -> Path | None:
    # Common Archive layouts
    candidates = list(ROOT.glob(f"**/draft_{draft_id}/**/photometry"))
    candidates += list(ROOT.glob(f"**/Draft_{draft_id}/**/photometry"))
    for p in candidates:
        if p.is_dir():
            return p
    return None


def _load_comp_table(phot: Path) -> pd.DataFrame | None:
    for name in (
        "comparison_stars.csv",
        "comp_quality.csv",
        "global_comp_pool.csv",
        "comp_pool.csv",
    ):
        f = phot / name
        if f.is_file():
            return pd.read_csv(f)
    # per-target comps
    hits = list(phot.glob("**/comparison_stars_*.csv"))
    if hits:
        return pd.read_csv(hits[0])
    return None


def _airmass_span_from_proc(proc_dir: Path) -> float:
    files = sorted(proc_dir.glob("proc_*.csv"))[:50]
    ams: list[float] = []
    for f in files:
        try:
            df = pd.read_csv(f, usecols=lambda c: c.lower() == "airmass")
        except Exception:  # noqa: BLE001
            continue
        if df.empty:
            continue
        v = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        if v.notna().any():
            ams.append(float(v.dropna().iloc[0]))
    if len(ams) < 2:
        return 0.0
    return float(max(ams) - min(ams))


def measure_draft(draft_id: int) -> dict:
    out: dict = {"draft_id": draft_id, "optics_kind": "refractive"}
    phot = _find_draft_phot(draft_id)
    out["phot_dir"] = str(phot) if phot else None
    if phot is None:
        out["error"] = "photometry_dir_not_found"
        # Still report refractive c_col from measurement
        coeffs = resolve_comp_weight_coeffs(optics_kind="refractive", k2_bprp=None, airmass_span=0.0)
        out["c_col"] = {
            "value": coeffs.c_col_mag_per_bprp,
            "unit": "mag_per_bprp",
            "uncertainty": float("nan"),
            "derivation": coeffs.c_col_source,
            "k2_term": coeffs.c_col_k2_mag_per_bprp,
            "psf_term": coeffs.c_col_psf_mag_per_bprp,
            "psf_numerator_mmag": C_COL_PSF_EE_MMAG,
            "psf_bprp_span": C_COL_PSF_BPRP_SPAN,
            "notes": list(coeffs.notes),
        }
        out["c_dist"] = {
            "value": 0.0,
            "unit": "mag_per_deg",
            "uncertainty": float("nan"),
            "derivation": "named_gap:no_proc_products_for_regression",
        }
        return out

    proc = phot.parent / "proc"
    if not proc.is_dir():
        proc = phot / "proc"
    am_span = _airmass_span_from_proc(proc) if proc.is_dir() else 0.0
    out["airmass_span"] = am_span

    comps = _load_comp_table(phot)
    r_list: list[float] = []
    sc_list: list[float] = []
    if comps is not None and not comps.empty:
        # Prefer residual-like scatter columns
        rms_col = next(
            (c for c in ("comp_rms", "scatter_mad", "p2p_rms", "rms") if c in comps.columns),
            None,
        )
        ra_col = next((c for c in ("ra_deg", "ra") if c in comps.columns), None)
        dec_col = next((c for c in ("dec_deg", "dec") if c in comps.columns), None)
        if rms_col and ra_col and dec_col:
            # Use median of comps as pseudo-target for field-centre separation if no target.
            tra = float(pd.to_numeric(comps[ra_col], errors="coerce").median())
            tde = float(pd.to_numeric(comps[dec_col], errors="coerce").median())
            for _, row in comps.iterrows():
                rms = float(pd.to_numeric(row.get(rms_col), errors="coerce"))
                rr = float(pd.to_numeric(row.get(ra_col), errors="coerce"))
                dd = float(pd.to_numeric(row.get(dec_col), errors="coerce"))
                if not (math.isfinite(rms) and math.isfinite(rr) and math.isfinite(dd)):
                    continue
                dra = math.radians(rr - tra) * math.cos(math.radians(0.5 * (dd + tde)))
                dde = math.radians(dd - tde)
                r_list.append(float(math.degrees(math.hypot(dra, dde))))
                sc_list.append(rms)

    # CLEAR/unfiltered: k2 is NONE (literature). No CHOSEN CMOS k'' invented.
    coeffs = resolve_comp_weight_coeffs(
        optics_kind="refractive",
        k2_bprp=None,
        airmass_span=am_span,
        r_deg=r_list or None,
        residual_scatter_mag=sc_list or None,
    )
    out["c_col"] = {
        "value": coeffs.c_col_mag_per_bprp,
        "unit": "mag_per_bprp",
        "uncertainty": float("nan"),  # PSF term from single published EE estimate
        "derivation": coeffs.c_col_source,
        "k2_term": coeffs.c_col_k2_mag_per_bprp,
        "psf_term": coeffs.c_col_psf_mag_per_bprp,
        "psf_numerator_mmag": C_COL_PSF_EE_MMAG,
        "psf_bprp_span": C_COL_PSF_BPRP_SPAN,
        "k2_policy": "CLEAR/unfiltered -> literature NONE (DECISIONS k''); no CHOSEN CMOS default",
        "notes": list(coeffs.notes),
    }
    out["c_dist"] = {
        "value": coeffs.c_dist_mag_per_deg,
        "unit": "mag_per_deg",
        "uncertainty": coeffs.c_dist_slope_unc_mag_per_deg,
        "derivation": coeffs.c_dist_source,
        "n": coeffs.c_dist_n,
        "r_value": coeffs.c_dist_r_value,
        "estimator": "OLS polyfit degree 1 of residual_scatter vs r_deg",
        "n_regression_points": len(r_list),
        "ordering": "measured_on_pre_forced_proc; re-verify after forced rebuild",
    }
    out["n_comps_table"] = int(len(comps)) if comps is not None else 0
    return out


def fw_cvn_weights(draft_id: int = 513) -> dict:
    """Report weights for archived FW CVn comps if table found."""
    # Typical archived set from prior results; fill if found in photometry.
    archived = []
    phot = _find_draft_phot(draft_id)
    result: dict = {"draft_id": draft_id, "comps": []}
    if phot is None:
        result["error"] = "no_phot"
        return result
    # Look for target comps file mentioning FW
    hits = list(phot.glob("**/*FW*")) + list(phot.glob("**/comparison_stars*.csv"))
    comps = None
    for h in hits:
        if h.suffix.lower() == ".csv":
            try:
                df = pd.read_csv(h)
            except Exception:  # noqa: BLE001
                continue
            if "bp_rp" in df.columns or "comp_rms" in df.columns:
                comps = df
                result["source"] = str(h)
                break
    if comps is None:
        result["error"] = "no_comp_csv"
        return result
    # Target BP-RP: use bluest among top-4 by weight proxy or median
    bpr = pd.to_numeric(comps.get("bp_rp"), errors="coerce")
    rms = pd.to_numeric(comps.get("comp_rms", comps.get("rms")), errors="coerce")
    if bpr.notna().sum() < 2:
        result["error"] = "no_bprp"
        return result
    # Heuristic target colour near archived note: three comps ~0.4 redder than target
    # Use C03 as the least-red among four brightest finite-rms if n>=4
    sub = comps.loc[rms.notna() & bpr.notna()].copy()
    if len(sub) < 4:
        result["error"] = f"n_comps={len(sub)}"
        return result
    sub = sub.nsmallest(4, "comp_rms") if "comp_rms" in sub.columns else sub.head(4)
    target_bprp = float(bpr.loc[sub.index].min())  # bluest as proxy for C03-like
    c_col = C_COL_PSF_REFRACTIVE_MAG_PER_BPRP
    rows = []
    for _, r in sub.iterrows():
        cid = str(r.get("catalog_id", r.get("name", "")))
        db = abs(float(r["bp_rp"]) - target_bprp)
        se = sigma_eff_mag(
            sigma_rms_mag=float(r.get("comp_rms", 0.01) or 0.01),
            delta_bprp=db,
            r_deg=0.0,
            c_col_mag_per_bprp=c_col,
            c_dist_mag_per_deg=0.0,
        )
        w = weight_from_sigma_eff(se)
        rows.append(
            {
                "catalog_id": cid,
                "bp_rp": float(r["bp_rp"]),
                "delta_bprp": db,
                "comp_rms": float(r.get("comp_rms", float("nan"))),
                "sigma_eff": se,
                "weight": w,
            }
        )
    # Reference: C03 at dBP-RP 0.044
    se_ref = sigma_eff_mag(
        sigma_rms_mag=0.01,
        delta_bprp=0.044,
        r_deg=0.0,
        c_col_mag_per_bprp=c_col,
        c_dist_mag_per_deg=0.0,
    )
    w_ref = weight_from_sigma_eff(se_ref)
    for row in rows:
        row["weight_vs_c03_d0044"] = row["weight"] / w_ref if w_ref > 0 else float("nan")
    result["target_bprp_proxy"] = target_bprp
    result["c_col"] = c_col
    result["w_ref_c03_d0044"] = w_ref
    result["comps"] = rows
    return result


def main() -> None:
    reports = {str(d): measure_draft(d) for d in (512, 513, 435)}
    reports["fw_cvn_513"] = fw_cvn_weights(513)
    # Sensitivity: ZP offset proxy = weighted mean of colour term contribution
    c0 = C_COL_PSF_REFRACTIVE_MAG_PER_BPRP
    sens = []
    for fac in (0.5, 1.0, 1.5):
        cc = c0 * fac
        # Synthetic: 3 red comps at dBP-RP=0.4, rms=0.01
        ws = []
        for db in (0.044, 0.4, 0.4, 0.4):
            se = sigma_eff_mag(
                sigma_rms_mag=0.01,
                delta_bprp=db,
                r_deg=0.0,
                c_col_mag_per_bprp=cc,
                c_dist_mag_per_deg=0.0,
            )
            ws.append(weight_from_sigma_eff(se))
        w = np.asarray(ws, dtype=float)
        # Colour bias proxy: weighted mean of c_col*|dBP-RP|
        bias = float(np.sum(w * np.array([0.044, 0.4, 0.4, 0.4]) * cc) / np.sum(w))
        sens.append({"c_col_factor": fac, "c_col": cc, "weighted_colour_bias_mag": bias})
    reports["c_col_sensitivity"] = sens
    out = ROOT / "dev" / "results" / "COMP_WEIGHT_COEFF_01_measurements.json"
    out.write_text(json.dumps(reports, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(reports, indent=2, allow_nan=True)[:4000])
    print("wrote", out)


if __name__ == "__main__":
    main()
