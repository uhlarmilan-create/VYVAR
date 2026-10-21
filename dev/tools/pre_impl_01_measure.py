"""PRE-IMPL-01 measurements Q1-Q5 on draft 514. Report only for science; weight CSV rewrite is a fix."""
from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src_py"))

from comp_qa_core import loo_diff_series  # noqa: E402
from comp_weights import (  # noqa: E402
    C_COL_PSF_REFRACTIVE_MAG_PER_BPRP,
    sigma_eff_mag,
    weight_from_sigma_eff,
)
from photometry_core import _annulus_sky_subtracted_flux, ensemble_normalize  # noqa: E402

PHOT = ROOT / "Archive/Drafts/draft_000514/platesolve/NoFilter_60_2/photometry"
PROC = ROOT / "Archive/Drafts/draft_000514/detrended_aligned/lights/NoFilter_60_2"
OUT = ROOT / "dev/results"
C_COL = float(C_COL_PSF_REFRACTIVE_MAG_PER_BPRP)
PROD_R = 2.711323792774397
ISO_SEP_FWHM = 3.0  # isolation: no neighbour within 3*FWHM


def _sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, cwd=str(ROOT)
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def mad_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 5:
        return float("nan")
    med = float(np.median(x))
    return float(1.4826 * np.median(np.abs(x - med)))


def load_mags() -> tuple[list[str], dict[str, dict[str, float]], dict[str, dict[str, bool]], dict[str, tuple[float, float]], dict[str, float]]:
    files = sorted(p for p in PROC.glob("proc_*.csv") if "MASTER" not in p.name.upper())
    mag: dict[str, dict[str, float]] = {}
    sat: dict[str, dict[str, bool]] = {}
    xy: dict[str, tuple[float, float]] = {}
    gmag: dict[str, float] = {}
    fids: list[str] = []
    for fp in files:
        fid = fp.stem.replace("proc_", "")
        df = pd.read_csv(fp)
        cid = df["catalog_id"].astype(str)
        # PRE-IMPL-01: proc ``mag`` is catalog/Gaia mag (constant across frames).
        # Instrumental photometry is ``flux`` -> -2.5 log10(flux).
        flux = pd.to_numeric(df.get("flux"), errors="coerce")
        m = pd.Series(np.full(len(df), np.nan), index=df.index, dtype=float)
        okf = np.isfinite(flux.to_numpy(dtype=float)) & (flux.to_numpy(dtype=float) > 0)
        m.loc[okf] = -2.5 * np.log10(flux.loc[okf].to_numpy(dtype=float))
        satc = pd.to_numeric(df.get("likely_saturated"), errors="coerce").fillna(0)
        xs = pd.to_numeric(df.get("x"), errors="coerce")
        ys = pd.to_numeric(df.get("y"), errors="coerce")
        gm = pd.to_numeric(df.get("phot_g_mean_mag", df.get("mag")), errors="coerce")
        mm: dict[str, float] = {}
        ss: dict[str, bool] = {}
        for i in range(len(df)):
            c = str(cid.iloc[i])
            mv = float(m.iloc[i])
            if math.isfinite(mv):
                mm[c] = mv
            ss[c] = bool(float(satc.iloc[i]) >= 0.5)
            if c not in xy and math.isfinite(float(xs.iloc[i])) and math.isfinite(float(ys.iloc[i])):
                xy[c] = (float(xs.iloc[i]), float(ys.iloc[i]))
            if c not in gmag and math.isfinite(float(gm.iloc[i])):
                gmag[c] = float(gm.iloc[i])
        mag[fid] = mm
        sat[fid] = ss
        fids.append(fid)
    return fids, mag, sat, xy, gmag


def series_for_ids(
    fids: list[str],
    mag: dict[str, dict[str, float]],
    sat: dict[str, dict[str, bool]],
    ids: list[str],
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    n = len(fids)
    for cid in ids:
        arr = np.full(n, np.nan)
        for i, fid in enumerate(fids):
            if sat[fid].get(cid, False):
                continue
            mv = mag[fid].get(cid)
            if mv is not None and math.isfinite(mv):
                arr[i] = mv
        out[cid] = arr
    return out


def weights_map(comps: pd.DataFrame, tb: float) -> dict[str, float]:
    tra = float(pd.to_numeric(comps.get("ra_deg"), errors="coerce").median())
    tde = float(pd.to_numeric(comps.get("dec_deg"), errors="coerce").median())
    out: dict[str, float] = {}
    for _, r in comps.iterrows():
        cid = str(r["catalog_id"])
        rms = float(pd.to_numeric(r.get("comp_rms"), errors="coerce"))
        bpr = float(pd.to_numeric(r.get("bp_rp"), errors="coerce"))
        ra = float(pd.to_numeric(r.get("ra_deg"), errors="coerce"))
        dec = float(pd.to_numeric(r.get("dec_deg"), errors="coerce"))
        db = abs(bpr - tb) if math.isfinite(bpr) and math.isfinite(tb) else 0.0
        if math.isfinite(ra) and math.isfinite(dec) and math.isfinite(tra) and math.isfinite(tde):
            dra = math.radians(ra - tra) * math.cos(math.radians(0.5 * (dec + tde)))
            dde = math.radians(dec - tde)
            rdeg = float(math.degrees(math.hypot(dra, dde)))
        else:
            rdeg = 0.0
        se = sigma_eff_mag(
            sigma_rms_mag=rms if math.isfinite(rms) else float("nan"),
            delta_bprp=db,
            r_deg=rdeg,
            c_col_mag_per_bprp=C_COL,
            c_dist_mag_per_deg=0.0,
        )
        out[cid] = weight_from_sigma_eff(se)
    return out


def sigma_eff_map(comps: pd.DataFrame, tb: float) -> dict[str, dict[str, float]]:
    tra = float(pd.to_numeric(comps.get("ra_deg"), errors="coerce").median())
    tde = float(pd.to_numeric(comps.get("dec_deg"), errors="coerce").median())
    out: dict[str, dict[str, float]] = {}
    for _, r in comps.iterrows():
        cid = str(r["catalog_id"])
        rms = float(pd.to_numeric(r.get("comp_rms"), errors="coerce"))
        bpr = float(pd.to_numeric(r.get("bp_rp"), errors="coerce"))
        ra = float(pd.to_numeric(r.get("ra_deg"), errors="coerce"))
        dec = float(pd.to_numeric(r.get("dec_deg"), errors="coerce"))
        g = float(pd.to_numeric(r.get("phot_g_mean_mag", r.get("mag")), errors="coerce"))
        db = abs(bpr - tb) if math.isfinite(bpr) and math.isfinite(tb) else 0.0
        if math.isfinite(ra) and math.isfinite(dec) and math.isfinite(tra) and math.isfinite(tde):
            dra = math.radians(ra - tra) * math.cos(math.radians(0.5 * (dec + tde)))
            dde = math.radians(dec - tde)
            rdeg = float(math.degrees(math.hypot(dra, dde)))
        else:
            rdeg = 0.0
        se = sigma_eff_mag(
            sigma_rms_mag=rms if math.isfinite(rms) else float("nan"),
            delta_bprp=db,
            r_deg=rdeg,
            c_col_mag_per_bprp=C_COL,
            c_dist_mag_per_deg=0.0,
        )
        out[cid] = {
            "sigma_eff": se,
            "comp_rms": rms,
            "delta_bprp": db,
            "r_deg": rdeg,
            "bp_rp": bpr,
            "g_mag": g,
            "colour_term_mag": C_COL * db,
        }
    return out


def measure_q1(fids, mag, sat, comps_all, at) -> dict:
    """sigma_obs (MAD of LOO differential) vs sigma_eff. Isolate colour term."""
    # Use targets with comps; sample up to 8 targets for coverage, all their comps
    rows = []
    targets = at[at["skip_photometry"].astype(str).str.lower().isin(["0", "false", "nan", ""]) | at["skip_photometry"].isna()]
    # Prefer named CVn + a few more with comps
    names_pref = ["BO CVn", "FW CVn", "R CVn", "FU CVn", "SS CVn", "FY CVn", "FZ CVn", "GH CVn"]
    used = 0
    for name in names_pref:
        hit = at[at["name"].astype(str) == name]
        if hit.empty:
            continue
        tid = str(hit.iloc[0]["catalog_id"])
        tb = float(pd.to_numeric(hit.iloc[0].get("bp_rp"), errors="coerce"))
        sub = comps_all[comps_all["target_catalog_id"].astype(str) == tid]
        if sub.empty:
            continue
        ids = sub["catalog_id"].astype(str).tolist()
        series = series_for_ids(fids, mag, sat, ids)
        semap = sigma_eff_map(sub, tb)
        for cid in ids:
            diff = loo_diff_series(series, cid, ids)
            sobs = mad_sigma(diff)
            meta = semap.get(cid, {})
            se = float(meta.get("sigma_eff", float("nan")))
            rms = float(meta.get("comp_rms", float("nan")))
            db = float(meta.get("delta_bprp", float("nan")))
            ct = float(meta.get("colour_term_mag", float("nan")))
            ratio = sobs / se if (math.isfinite(sobs) and math.isfinite(se) and se > 0) else float("nan")
            # Non-tautological: excess beyond rms
            excess2 = sobs * sobs - rms * rms if (math.isfinite(sobs) and math.isfinite(rms)) else float("nan")
            colour2 = ct * ct if math.isfinite(ct) else float("nan")
            rows.append(
                {
                    "target": name,
                    "catalog_id": cid,
                    "g_mag": meta.get("g_mag"),
                    "delta_bprp": db,
                    "r_deg": meta.get("r_deg"),
                    "sigma_obs_mad_loo_mag": sobs,
                    "sigma_eff_mag": se,
                    "comp_rms_mag": rms,
                    "ratio_obs_over_eff": ratio,
                    "excess2_obs_minus_rms": excess2,
                    "colour_term2": colour2,
                }
            )
        used += 1
    df = pd.DataFrame(rows)
    ok = df["ratio_obs_over_eff"].apply(lambda v: math.isfinite(float(v)))
    ratios = df.loc[ok, "ratio_obs_over_eff"].to_numpy(float)
    # Colour isolation: among stars with |dBP-RP| bins, median excess2 / colour2
    colour_ok = df["colour_term2"].apply(lambda v: math.isfinite(float(v)) and float(v) > 1e-8)
    excess_ok = df["excess2_obs_minus_rms"].apply(lambda v: math.isfinite(float(v)))
    both = colour_ok & excess_ok
    if int(both.sum()) >= 20:
        x = df.loc[both, "delta_bprp"].to_numpy(float)
        y = df.loc[both, "excess2_obs_minus_rms"].to_numpy(float)
        # predicted excess2 = colour2; correlation of excess2 with colour2
        c2 = df.loc[both, "colour_term2"].to_numpy(float)
        # clip negative excess (obs quieter than rms) to 0 for ratio
        y_pos = np.maximum(y, 0.0)
        corr = float(np.corrcoef(c2, y_pos)[0, 1]) if c2.size > 5 else float("nan")
        # slope: excess2 ~ a * colour2
        if np.sum(c2) > 0:
            a = float(np.sum(c2 * y_pos) / np.sum(c2 * c2))
        else:
            a = float("nan")
    else:
        corr = a = float("nan")

    # ratio vs G, colour, sep
    def _bin_med(col: str, edges: list[float]) -> list[dict]:
        out = []
        v = pd.to_numeric(df[col], errors="coerce")
        for i in range(len(edges) - 1):
            m = ok & (v >= edges[i]) & (v < edges[i + 1])
            if int(m.sum()) < 8:
                continue
            out.append(
                {
                    "lo": edges[i],
                    "hi": edges[i + 1],
                    "n": int(m.sum()),
                    "ratio_median": float(np.median(df.loc[m, "ratio_obs_over_eff"])),
                }
            )
        return out

    return {
        "commit_sha": _sha(),
        "sigma_obs_estimator": "1.4826*MAD of loo_diff_series (focus - ensemble flux-sum, median-subtracted)",
        "n_rows": int(len(df)),
        "n_finite_ratio": int(ok.sum()),
        "ratio_median": float(np.median(ratios)) if ratios.size else None,
        "ratio_p16": float(np.percentile(ratios, 16)) if ratios.size else None,
        "ratio_p84": float(np.percentile(ratios, 84)) if ratios.size else None,
        "ratio_vs_G": _bin_med("g_mag", [8, 10, 11, 12, 13, 14, 15]),
        "ratio_vs_abs_delta_bprp": _bin_med("delta_bprp", [0, 0.2, 0.5, 1.0, 2.0, 5.0]),
        "ratio_vs_r_deg": _bin_med("r_deg", [0, 0.5, 1.0, 2.0, 5.0, 15.0]),
        "tautological_part": "sigma_eff includes comp_rms; sigma_obs and rms both measure night scatter of the same star (related domains)",
        "non_tautological": {
            "test": "excess2 = max(0, sigma_obs^2 - rms^2) vs colour_term^2 = (c_col*|dBP-RP|)^2",
            "corr_excess2_vs_colour2": corr,
            "scale_factor_excess2_over_colour2": a,
            "note": "a~1 means colour term predicts excess scatter; a~0 means colour term does not",
        },
        "rows_sample": df.head(20).to_dict(orient="records"),
        "falsification": "ratio flat near 1 with colour excess scaling a~1, OR ratio strongly mag-dependent",
    }


def _cum_keep(w: np.ndarray, frac: float) -> np.ndarray:
    order = np.argsort(-w)
    c = np.cumsum(w[order])
    thr = frac * c[-1]
    n = int(np.searchsorted(c, thr, side="left") + 1)
    keep = np.zeros(w.size, dtype=bool)
    keep[order[: max(n, 2)]] = True
    return keep


def measure_q2(fids, mag, sat, comps_all, at, suspected: set[str]) -> dict:
    """Check-star scatter vs cumulative-weight truncation (real mags)."""
    hit = at[at["name"] == "BO CVn"]
    if hit.empty:
        return {"error": "BO CVn missing", "commit_sha": _sha()}
    tid = str(hit.iloc[0]["catalog_id"])
    tb = float(pd.to_numeric(hit.iloc[0].get("bp_rp"), errors="coerce"))
    sub = comps_all[comps_all["target_catalog_id"].astype(str) == tid]
    ids = sub["catalog_id"].astype(str).tolist()
    wmap = weights_map(sub, tb)
    w = np.asarray([wmap.get(c, 0.0) for c in ids], dtype=float)
    series = series_for_ids(fids, mag, sat, ids)
    am = None
    lc = PHOT / "lightcurves" / f"lightcurve_{tid}.csv"
    if lc.is_file():
        lcd = pd.read_csv(lc)
        if "airmass" in lcd.columns and len(lcd) == len(fids):
            am = pd.to_numeric(lcd["airmass"], errors="coerce").to_numpy(float)

    tgt_ids = set(at["catalog_id"].astype(str))
    g = pd.to_numeric(sub.get("phot_g_mean_mag", sub.get("mag")), errors="coerce")
    checks = []
    for i, cid in enumerate(ids):
        if cid in suspected or cid in tgt_ids:
            continue
        if not (math.isfinite(w[i]) and w[i] > 0):
            continue
        gi = float(g.iloc[i]) if i < len(g) else float("nan")
        if not (9.0 <= gi <= 12.5):
            continue
        checks.append(cid)
    checks = checks[:25]
    if len(checks) < 5:
        return {"error": "too few check stars", "n": len(checks), "commit_sha": _sha()}

    fracs = [1.0, 0.99, 0.95, 0.90, 0.50]
    per_check = []
    summary: dict[float, list[float]] = {f: [] for f in fracs}
    slopes: dict[float, list[float]] = {f: [] for f in fracs}
    for cid in checks:
        row = {"catalog_id": cid, "truncations": []}
        for frac in fracs:
            if frac >= 1.0:
                keep_ids = list(ids)
            else:
                mask = _cum_keep(w, frac)
                keep_ids = [c for c, m in zip(ids, mask, strict=False) if m]
            # Focus = check star; ensemble = kept peers (exclude self). Real mags only.
            ens_ids = [c for c in keep_ids if c != cid]
            if len(ens_ids) < 5:
                continue
            diff = loo_diff_series(series, cid, ens_ids)
            sc = mad_sigma(diff)
            slope = float("nan")
            if am is not None and np.isfinite(diff).sum() >= 10:
                fin = np.isfinite(diff) & np.isfinite(am)
                if int(fin.sum()) >= 10:
                    slope = float(np.polyfit(am[fin], diff[fin], 1)[0])
            sc_mmag = sc * 1000.0 if math.isfinite(sc) else None
            row["truncations"].append(
                {
                    "cum_weight_frac": frac,
                    "n_ens": len(ens_ids),
                    "scatter_mad_mmag": sc_mmag,
                    "airmass_slope_mmag_per_airmass": slope * 1000.0 if math.isfinite(slope) else None,
                }
            )
            if sc_mmag is not None:
                summary[frac].append(sc_mmag)
            if math.isfinite(slope):
                slopes[frac].append(abs(slope) * 1000.0)
        per_check.append(row)

    med = {str(f): (float(np.median(v)) if v else None) for f, v in summary.items()}
    med_slope = {str(f): (float(np.median(v)) if v else None) for f, v in slopes.items()}
    full = med.get("1.0")
    t99 = med.get("0.99")
    t50 = med.get("0.5")
    if full is not None and t99 is not None and t50 is not None:
        if abs(t99 - full) < 0.5 and abs(t50 - full) < 1.0:
            decision = "tail_inert_performance_cut_ok"
        elif t50 > full + 1.0:
            decision = "tail_carries_information_keep"
        elif t50 < full - 1.0:
            decision = "tail_carries_unmodelled_systematics"
        else:
            decision = "mixed_or_weak"
    else:
        decision = "insufficient"

    return {
        "commit_sha": _sha(),
        "ensemble_target_for_weights": "BO CVn",
        "method": "loo_diff_series MAD on real proc mags; cumulative-weight truncation of peer set only",
        "n_check_stars": len(checks),
        "check_star_ids": checks,
        "median_scatter_mad_mmag_by_truncation": med,
        "median_abs_airmass_slope_mmag_by_truncation": med_slope,
        "decision": decision,
        "per_check": per_check,
        "falsification": "median check-star MAD rises or falls by >1 mmag from full to 50% truncation",
    }


def measure_q3(fids, mag, sat, comps_all, at, suspected: set[str]) -> dict:
    """Colour term via LOO on non-variable comps as pseudo-targets."""
    hit = at[at["name"] == "BO CVn"]
    tid = str(hit.iloc[0]["catalog_id"])
    tb = float(pd.to_numeric(hit.iloc[0].get("bp_rp"), errors="coerce"))
    sub = comps_all[comps_all["target_catalog_id"].astype(str) == tid]
    ids = sub["catalog_id"].astype(str).tolist()
    wmap = weights_map(sub, tb)
    series = series_for_ids(fids, mag, sat, ids)
    bpr = {
        str(r["catalog_id"]): float(pd.to_numeric(r.get("bp_rp"), errors="coerce"))
        for _, r in sub.iterrows()
    }
    am = None
    lc = PHOT / "lightcurves" / f"lightcurve_{tid}.csv"
    if lc.is_file():
        lcd = pd.read_csv(lc)
        if "airmass" in lcd.columns and len(lcd) == len(fids):
            am = pd.to_numeric(lcd["airmass"], errors="coerce").to_numpy(float)

    tgt_ids = set(at["catalog_id"].astype(str))
    rows = []
    for cid in ids:
        if cid in suspected or cid in tgt_ids:
            continue
        if not (math.isfinite(wmap.get(cid, float("nan"))) and wmap[cid] > 0):
            continue
        peers = [c for c in ids if c != cid and math.isfinite(wmap.get(c, float("nan"))) and wmap[c] > 0]
        if len(peers) < 20:
            continue
        diff = loo_diff_series(series, cid, peers)
        fin = np.isfinite(diff)
        if int(fin.sum()) < 20:
            continue
        # weighted ensemble colour of peers
        ww = np.asarray([wmap[c] for c in peers], dtype=float)
        bb = np.asarray([bpr.get(c, float("nan")) for c in peers], dtype=float)
        okb = np.isfinite(bb) & np.isfinite(ww) & (ww > 0)
        if not okb.any() or not math.isfinite(bpr.get(cid, float("nan"))):
            continue
        ens_col = float(np.sum(ww[okb] * bb[okb]) / np.sum(ww[okb]))
        dcol = ens_col - float(bpr[cid])
        mean_off = float(np.nanmedian(diff[fin]))  # already median-subtracted in loo -> ~0
        # use raw focus-ens before median sub for level? loo_diff_series median-subtracts.
        # Rebuild level from focus - ens without median sub:
        m_f = series[cid]
        stack = np.vstack([series[c] for c in peers])
        flux = np.nansum(10.0 ** (-0.4 * stack), axis=0)
        ens = np.full(len(fids), np.nan)
        okf = np.isfinite(flux) & (flux > 0)
        ens[okf] = -2.5 * np.log10(flux[okf])
        raw = m_f - ens
        use = np.isfinite(raw)
        level = float(np.nanmedian(raw[use])) if use.any() else float("nan")
        slope = float("nan")
        if am is not None:
            aa = am[fin]
            rr = diff[fin]
            ok = np.isfinite(aa) & np.isfinite(rr)
            if int(ok.sum()) >= 15:
                slope = float(np.polyfit(aa[ok], rr[ok], 1)[0])
        rows.append(
            {
                "catalog_id": cid,
                "delta_colour_bprp": dcol,
                "airmass_slope_mag_per_airmass": slope,
                "airmass_slope_mmag_per_airmass": slope * 1000.0 if math.isfinite(slope) else None,
                "mean_level_offset_mag": level,
                "mean_level_offset_mmag": level * 1000.0 if math.isfinite(level) else None,
                "bp_rp": bpr[cid],
            }
        )

    if len(rows) < 15:
        return {"commit_sha": _sha(), "error": "too few stars", "n": len(rows)}

    dc = np.asarray([r["delta_colour_bprp"] for r in rows], float)
    sl = np.asarray([r["airmass_slope_mag_per_airmass"] for r in rows], float)
    lv = np.asarray([r["mean_level_offset_mag"] for r in rows], float)
    ok_s = np.isfinite(dc) & np.isfinite(sl)
    ok_l = np.isfinite(dc) & np.isfinite(lv)

    def _fit(x, y):
        if x.size < 10:
            return float("nan"), float("nan"), float("nan")
        # slope, intercept; stderr via residual
        A = np.vstack([x, np.ones(len(x))]).T
        coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ coef
        resid = y - pred
        dof = max(1, len(x) - 2)
        s2 = float(np.sum(resid**2) / dof)
        xtx_inv = np.linalg.inv(A.T @ A)
        se0 = math.sqrt(s2 * float(xtx_inv[0, 0]))
        return float(coef[0]), se0, float(np.corrcoef(x, y)[0, 1])

    kpp, kpp_se, kpp_corr = _fit(dc[ok_s], sl[ok_s])
    lev, lev_se, lev_corr = _fit(dc[ok_l], lv[ok_l])

    # significance: |coef| > 2*se
    shape_sig = bool(math.isfinite(kpp) and math.isfinite(kpp_se) and abs(kpp) > 2 * kpp_se)
    level_sig = bool(math.isfinite(lev) and math.isfinite(lev_se) and abs(lev) > 2 * lev_se)

    if not shape_sig and not level_sig:
        design = "a_weighting_alone"
    elif level_sig and not shape_sig:
        design = "c_level_term_exports_only"
    else:
        design = "b_or_c_shape_term_required"

    return {
        "commit_sha": _sha(),
        "n_stars": len(rows),
        "shape_term_kpp_mag_per_bprp_per_airmass": kpp,
        "shape_term_kpp_mmag_per_bprp_per_airmass": kpp * 1000.0 if math.isfinite(kpp) else None,
        "shape_term_se_mmag": kpp_se * 1000.0 if math.isfinite(kpp_se) else None,
        "shape_term_corr": kpp_corr,
        "shape_significant_2se": shape_sig,
        "level_term_mag_per_bprp": lev,
        "level_term_mmag_per_bprp": lev * 1000.0 if math.isfinite(lev) else None,
        "level_term_se_mmag": lev_se * 1000.0 if math.isfinite(lev_se) else None,
        "level_term_corr": lev_corr,
        "level_significant_2se": level_sig,
        "compare_c_col_mmag_per_bprp": C_COL * 1000.0,
        "compare_k2_clear": "NONE",
        "design_choice": design,
        "falsification": "shape and/or level term |coef| > 2*se, or both consistent with zero",
        "stars": rows,
    }


def measure_q4(xy, gmag, fwhm_med: float) -> dict:
    """Empirical CoG on isolated bright unsaturated stars."""
    # Isolation from MASTER positions
    ids = list(xy.keys())
    arr = np.asarray([xy[c] for c in ids], float)
    # neighbour distance
    from scipy.spatial import cKDTree

    tree = cKDTree(arr)
    # FWHM for isolation: use ~5.3 AUTO / dao scale for physical isolation
    fwhm_iso = max(fwhm_med, 5.0)
    min_sep = ISO_SEP_FWHM * fwhm_iso
    dists, _ = tree.query(arr, k=2)
    nn = dists[:, 1]
    isolated = [ids[i] for i in range(len(ids)) if math.isfinite(nn[i]) and nn[i] >= min_sep]
    # bright unsaturated: G 8-11, and not saturated on frame 001
    proc1 = pd.read_csv(PROC / "proc_BO_CVn_Light_001.csv")
    sat1 = {
        str(r["catalog_id"]): bool(float(pd.to_numeric(r.get("likely_saturated"), errors="coerce") or 0) >= 0.5)
        for _, r in proc1.iterrows()
    }
    stars = []
    for cid in isolated:
        g = gmag.get(cid, float("nan"))
        if not (8.0 <= g <= 11.0):
            continue
        if sat1.get(cid, False):
            continue
        stars.append(cid)
    stars = stars[:12]
    if len(stars) < 3:
        return {"commit_sha": _sha(), "error": "too few isolated", "n_iso": len(isolated), "n_bright": len(stars)}

    frames = sorted(PROC.glob("BO_CVn_Light_*.fits"))[:40]
    radii = np.arange(0.5, 15.25, 0.25)
    r_norm = 12.0
    # Per-star median CoG across frames
    curves = []
    ee_at_prod = []
    for cid in stars:
        x, y = xy[cid]
        flux_r = []
        for fp in frames:
            with fits.open(fp, memmap=True) as hd:
                d = np.asarray(hd[0].data, dtype=float)
            vals = []
            for r in radii:
                f, _, _ = _annulus_sky_subtracted_flux(d, x, y, float(r), 10.0, 15.0)
                vals.append(f if math.isfinite(f) else float("nan"))
            flux_r.append(vals)
        med = np.nanmedian(np.asarray(flux_r, float), axis=0)
        # normalize at r_norm
        i_norm = int(np.argmin(np.abs(radii - r_norm)))
        # flatness: median of last 5 bins / norm
        tail = med[-5:]
        flat = float(np.nanmedian(tail) / med[i_norm]) if med[i_norm] > 0 else float("nan")
        ee = med / med[i_norm] if med[i_norm] > 0 else med * np.nan
        i_prod = int(np.argmin(np.abs(radii - PROD_R)))
        ee_prod = float(ee[i_prod]) if np.isfinite(ee[i_prod]) else float("nan")
        ee_at_prod.append(ee_prod)
        curves.append(
            {
                "catalog_id": cid,
                "g_mag": gmag.get(cid),
                "ee_at_prod_r": ee_prod,
                "flatness_tail_over_norm": flat,
                "r50": float(radii[np.argmin(np.abs(ee - 0.5))]) if np.isfinite(ee).any() else None,
                "r90": float(radii[np.argmin(np.abs(ee - 0.9))]) if np.isfinite(ee).any() else None,
            }
        )

    # Per-frame EE variation at fixed r=PROD_R using frame FWHM from proc if present
    # Approximate: EE(r/FWHM) for Gaussian: 1-exp(-k (r/sigma)^2); use empirical per-frame
    # from first isolated star across all frames
    cid0 = stars[0]
    x0, y0 = xy[cid0]
    ee_frames = []
    for fp in frames:
        with fits.open(fp, memmap=True) as hd:
            d = np.asarray(hd[0].data, dtype=float)
        # normalize at 12
        f_n, _, _ = _annulus_sky_subtracted_flux(d, x0, y0, r_norm, 10.0, 15.0)
        f_p, _, _ = _annulus_sky_subtracted_flux(d, x0, y0, PROD_R, 10.0, 15.0)
        if math.isfinite(f_n) and f_n > 0 and math.isfinite(f_p):
            ee_frames.append(f_p / f_n)
    ee_f = np.asarray(ee_frames, float)
    # variation in mmag: -2.5log10(ee) relative to median
    if ee_f.size >= 5:
        mmag = -2.5 * np.log10(np.clip(ee_f, 1e-6, None))
        mmag = mmag - float(np.median(mmag))
        ee_var_mmag = float(1.4826 * np.median(np.abs(mmag)))
        ee_p2p_mmag = float(np.nanpercentile(mmag, 84) - np.nanpercentile(mmag, 16))
    else:
        ee_var_mmag = ee_p2p_mmag = float("nan")

    # r_min clamp note
    snr = json.loads((ROOT / "Archive/Drafts/draft_000514/aperture_snr_table.json").read_text())
    return {
        "commit_sha": _sha(),
        "production_r_px": PROD_R,
        "norm_radius_px": r_norm,
        "isolation_min_sep_px": min_sep,
        "isolation_fwhm_px_used": fwhm_iso,
        "n_isolated_bright": len(stars),
        "n_frames_cog": len(frames),
        "ee_at_prod_median": float(np.nanmedian(ee_at_prod)),
        "ee_at_prod_p16": float(np.nanpercentile(ee_at_prod, 16)),
        "ee_at_prod_p84": float(np.nanpercentile(ee_at_prod, 84)),
        "per_star": curves,
        "ee_night_variation_mad_mmag": ee_var_mmag,
        "ee_night_variation_p16_p84_mmag": ee_p2p_mmag,
        "register_84p6_resolution": (
            "84.6% EE was measured on draft 510 at production radii ~4.1 px "
            "(CURSOR_RESULT_a1_growth_curves). Draft 514 median aperture equals "
            "SNR r_min=2.711 px (faint-star clamp), a different radius. "
            "r90~5.0-5.8 px and EE~84.6% at r~4.1 are consistent with each other; "
            "they are inconsistent only if conflated with r=2.711."
        ),
        "r_min_clamp": {
            "snr_r_min_px": snr["r_min_px"],
            "median_aperture_equals_r_min": abs(float(snr["r_min_px"]) - PROD_R) < 1e-6,
            "intended": (
                "Yes as a numerical floor in the SNR table for faint mag bins; "
                "not an optimized radius for the median star. Most catalogue stars "
                "are faint so the median sits on the clamp boundary."
            ),
        },
        "falsification": "EE at 2.711 px near 0.85, or night EE variation << 1 mmag MAD",
    }


def measure_q5(fids, mag, sat, xy, comps_all, at, fwhm_med: float) -> dict:
    hit = at[at["name"] == "BO CVn"]
    tid = str(hit.iloc[0]["catalog_id"])
    tb = float(pd.to_numeric(hit.iloc[0].get("bp_rp"), errors="coerce"))
    sub = comps_all[comps_all["target_catalog_id"].astype(str) == tid]
    ids = sub["catalog_id"].astype(str).tolist()
    series = series_for_ids(fids, mag, sat, ids)
    bpr = {str(r["catalog_id"]): float(pd.to_numeric(r.get("bp_rp"), errors="coerce")) for _, r in sub.iterrows()}
    g = {str(r["catalog_id"]): float(pd.to_numeric(r.get("phot_g_mean_mag", r.get("mag")), errors="coerce")) for _, r in sub.iterrows()}

    # blend: neighbour within 1 aperture radius among catalogue xy
    from scipy.spatial import cKDTree

    pool_xy = []
    pool_ids = []
    for cid in ids:
        if cid in xy:
            pool_xy.append(xy[cid])
            pool_ids.append(cid)
    tree = cKDTree(np.asarray(pool_xy, float))
    # also search against all catalogue for fair blend flag
    all_ids = list(xy.keys())
    all_xy = np.asarray([xy[c] for c in all_ids], float)
    tree_all = cKDTree(all_xy)
    blended = set()
    blend_dcol = {}
    for cid in pool_ids:
        d, ix = tree_all.query(xy[cid], k=5)
        for dist, j in zip(np.atleast_1d(d), np.atleast_1d(ix), strict=False):
            oid = all_ids[int(j)]
            if oid == cid:
                continue
            if dist <= PROD_R:
                blended.add(cid)
                db = abs(bpr.get(cid, float("nan")) - bpr.get(oid, float("nan")))
                if math.isfinite(db):
                    blend_dcol[cid] = max(blend_dcol.get(cid, 0.0), db)
                break

    am = None
    lc = PHOT / "lightcurves" / f"lightcurve_{tid}.csv"
    if lc.is_file():
        lcd = pd.read_csv(lc)
        if "airmass" in lcd.columns and len(lcd) == len(fids):
            am = pd.to_numeric(lcd["airmass"], errors="coerce").to_numpy(float)

    def stats_for(cids: list[str]) -> list[dict]:
        out = []
        for cid in cids:
            diff = loo_diff_series(series, cid, ids)
            sc = mad_sigma(diff)
            slope = float("nan")
            if am is not None:
                fin = np.isfinite(diff) & np.isfinite(am)
                if int(fin.sum()) >= 15:
                    slope = float(np.polyfit(am[fin], diff[fin], 1)[0])
            out.append(
                {
                    "catalog_id": cid,
                    "g_mag": g.get(cid),
                    "scatter_mad_mag": sc,
                    "airmass_slope_mag": slope,
                    "blend_delta_bprp": blend_dcol.get(cid),
                }
            )
        return out

    # matched magnitude: bin and pair
    b_stats = stats_for([c for c in ids if c in blended])
    i_stats = stats_for([c for c in ids if c not in blended])

    def _matched_compare(bs, iso, mag_tol=0.25):
        pairs = []
        for b in bs:
            if not math.isfinite(b.get("g_mag") or float("nan")):
                continue
            cands = [
                i
                for i in iso
                if math.isfinite(i.get("g_mag") or float("nan"))
                and abs(i["g_mag"] - b["g_mag"]) <= mag_tol
            ]
            if not cands:
                continue
            # nearest mag
            cands.sort(key=lambda z: abs(z["g_mag"] - b["g_mag"]))
            i0 = cands[0]
            if math.isfinite(b["scatter_mad_mag"]) and math.isfinite(i0["scatter_mad_mag"]):
                pairs.append(
                    {
                        "blend_scatter": b["scatter_mad_mag"],
                        "iso_scatter": i0["scatter_mad_mag"],
                        "blend_slope": b["airmass_slope_mag"],
                        "iso_slope": i0["airmass_slope_mag"],
                        "blend_dcol": b.get("blend_delta_bprp"),
                        "g": b["g_mag"],
                    }
                )
        return pairs

    pairs = _matched_compare(b_stats, i_stats)
    if len(pairs) >= 10:
        ds = np.asarray([p["blend_scatter"] - p["iso_scatter"] for p in pairs], float)
        dsl = np.asarray(
            [
                abs(p["blend_slope"]) - abs(p["iso_slope"])
                for p in pairs
                if math.isfinite(p["blend_slope"] or float("nan")) and math.isfinite(p["iso_slope"] or float("nan"))
            ],
            float,
        )
        med_ds = float(np.median(ds))
        # track colour of blend pair
        dcol = np.asarray([p["blend_dcol"] for p in pairs if p.get("blend_dcol") is not None], float)
        ds_c = np.asarray(
            [p["blend_scatter"] - p["iso_scatter"] for p in pairs if p.get("blend_dcol") is not None],
            float,
        )
        corr_c = float(np.corrcoef(dcol, ds_c)[0, 1]) if dcol.size >= 8 else float("nan")
    else:
        med_ds = float("nan")
        dsl = np.array([])
        corr_c = float("nan")

    # decision: blended worse if scatter excess > 1 mmag median
    if math.isfinite(med_ds):
        if med_ds * 1000 > 1.0:
            decision = "blended_worse_merging_required"
        elif abs(med_ds) * 1000 < 0.5:
            decision = "same_defer_merging"
        else:
            decision = "marginal"
    else:
        decision = "insufficient"

    return {
        "commit_sha": _sha(),
        "n_pool": len(ids),
        "n_blended": len(blended),
        "n_isolated": len(ids) - len(blended),
        "n_matched_pairs": len(pairs),
        "median_scatter_excess_blend_minus_iso_mmag": med_ds * 1000.0 if math.isfinite(med_ds) else None,
        "median_abs_slope_excess_mmag_per_airmass": float(np.median(dsl) * 1000) if dsl.size else None,
        "corr_scatter_excess_vs_blend_delta_bprp": corr_c,
        "decision": decision,
        "falsification": "matched-mag scatter excess |median| > 1 mmag, or tracks blend colour difference",
    }


def main() -> None:
    t0 = time.perf_counter()
    sha = _sha()
    print("loading frames...", flush=True)
    fids, mag, sat, xy, gmag = load_mags()
    comps = pd.read_csv(PHOT / "comparison_stars_per_target.csv")
    at = pd.read_csv(PHOT / "active_targets.csv")
    sus = set()
    sp = PHOT / "suspected_variables.csv"
    if sp.is_file():
        sus = set(pd.read_csv(sp)["catalog_id"].astype(str))
    # FWHM for isolation from SNR table dao
    snr = json.loads((ROOT / "Archive/Drafts/draft_000514/aperture_snr_table.json").read_text())
    fwhm_med = float(snr.get("vy_fwhm_dao_px") or snr.get("fwhm_px") or 5.2)

    print("Q1...", flush=True)
    q1 = measure_q1(fids, mag, sat, comps, at)
    (OUT / "PRE_IMPL_01_Q1.json").write_text(json.dumps(q1, indent=2), encoding="utf-8")
    print("Q1 ratio_median", q1.get("ratio_median"), "colour_scale", q1.get("non_tautological"), flush=True)

    print("Q2...", flush=True)
    q2 = measure_q2(fids, mag, sat, comps, at, sus)
    (OUT / "PRE_IMPL_01_Q2.json").write_text(json.dumps(q2, indent=2), encoding="utf-8")
    print("Q2", q2.get("decision"), q2.get("median_scatter_mad_mmag_by_truncation"), flush=True)

    print("Q3...", flush=True)
    q3 = measure_q3(fids, mag, sat, comps, at, sus)
    (OUT / "PRE_IMPL_01_Q3.json").write_text(json.dumps(q3, indent=2), encoding="utf-8")
    print("Q3 design", q3.get("design_choice"), "kpp_mmag", q3.get("shape_term_kpp_mmag_per_bprp_per_airmass"), "level_mmag", q3.get("level_term_mmag_per_bprp"), flush=True)

    print("Q4...", flush=True)
    q4 = measure_q4(xy, gmag, fwhm_med)
    (OUT / "PRE_IMPL_01_Q4.json").write_text(json.dumps(q4, indent=2), encoding="utf-8")
    print("Q4 EE", q4.get("ee_at_prod_median"), "var_mmag", q4.get("ee_night_variation_mad_mmag"), flush=True)

    print("Q5...", flush=True)
    q5 = measure_q5(fids, mag, sat, xy, comps, at, fwhm_med)
    (OUT / "PRE_IMPL_01_Q5.json").write_text(json.dumps(q5, indent=2), encoding="utf-8")
    print("Q5", q5.get("decision"), q5.get("median_scatter_excess_blend_minus_iso_mmag"), flush=True)

    print(f"DONE wall_s={time.perf_counter()-t0:.1f} sha={sha}", flush=True)


if __name__ == "__main__":
    main()
