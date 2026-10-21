"""Corrected Q2 only for PRE-IMPL-01."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src_py"))
sys.path.insert(0, str(ROOT / "dev/tools"))

import pre_impl_01_measure as m  # noqa: E402


def measure_q2_weighted(fids, mag, sat, comps_all, at, suspected, xy):
    hit = at[at["name"] == "BO CVn"]
    tid = str(hit.iloc[0]["catalog_id"])
    tb = float(pd.to_numeric(hit.iloc[0].get("bp_rp"), errors="coerce"))
    sub = comps_all[comps_all["target_catalog_id"].astype(str) == tid]
    ids = sub["catalog_id"].astype(str).tolist()
    wmap = m.weights_map(sub, tb)
    w = np.asarray([wmap.get(c, 0.0) for c in ids], dtype=float)
    series = m.series_for_ids(fids, mag, sat, ids)
    am = None
    lc = m.PHOT / "lightcurves" / f"lightcurve_{tid}.csv"
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
        if 9.0 <= gi <= 12.5:
            checks.append(cid)
    checks = checks[:25]

    def peers_iso(focus, cands):
        if focus not in xy:
            return [c for c in cands if c != focus]
        fx, fy = xy[focus]
        out = []
        for c in cands:
            if c == focus or c not in xy:
                continue
            cx, cy = xy[c]
            if math.hypot(cx - fx, cy - fy) < 3.0 * 5.2:
                continue
            out.append(c)
        return out

    def weighted_diff(focus, peers):
        n = len(fids)
        out = np.full(n, np.nan)
        mf = series[focus]
        for i in range(n):
            if not math.isfinite(mf[i]):
                continue
            num = den = 0.0
            for c in peers:
                mv = series[c][i]
                ww = wmap.get(c, float("nan"))
                if math.isfinite(mv) and math.isfinite(ww) and ww > 0:
                    num += ww * mv
                    den += ww
            if den > 0:
                out[i] = mf[i] - num / den
        fin = np.isfinite(out)
        if fin.any():
            out[fin] = out[fin] - float(np.median(out[fin]))
        return out

    fracs = [1.0, 0.99, 0.95, 0.90, 0.50]
    summary = {f: [] for f in fracs}
    slopes = {f: [] for f in fracs}
    per_check = []
    for cid in checks:
        row = {"catalog_id": cid, "truncations": []}
        for frac in fracs:
            keep = ids if frac >= 1.0 else [c for c, mk in zip(ids, m._cum_keep(w, frac), strict=False) if mk]
            peers = peers_iso(cid, keep)
            if len(peers) < 5:
                continue
            diff = weighted_diff(cid, peers)
            sc = m.mad_sigma(diff)
            slope = float("nan")
            if am is not None:
                fin = np.isfinite(diff) & np.isfinite(am)
                if int(fin.sum()) >= 10:
                    slope = float(np.polyfit(am[fin], diff[fin], 1)[0])
            sc_mmag = sc * 1000 if math.isfinite(sc) else None
            row["truncations"].append(
                {
                    "cum_weight_frac": frac,
                    "n_ens": len(peers),
                    "scatter_mad_mmag": sc_mmag,
                    "airmass_slope_mmag_per_airmass": slope * 1000 if math.isfinite(slope) else None,
                }
            )
            if sc_mmag is not None:
                summary[frac].append(sc_mmag)
            if math.isfinite(slope):
                slopes[frac].append(abs(slope) * 1000)
        per_check.append(row)

    med = {str(f): float(np.median(v)) if v else None for f, v in summary.items()}
    med_s = {str(f): float(np.median(v)) if v else None for f, v in slopes.items()}
    full, t99, t50 = med.get("1.0"), med.get("0.99"), med.get("0.5")
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
        "commit_sha": m._sha(),
        "ensemble_target_for_weights": "BO CVn",
        "spec_defect_original": (
            "Named loo_diff_series (flux-sum) under weight truncation is the wrong "
            "discriminator here: flux-sum ignores weights, and truncated peer sets "
            "produced MAD=0 discrete residuals. Corrected to weighted-mean peers."
        ),
        "method": "weighted mean peer inst mags; exclude <3 FWHM neighbours; MAD of residual",
        "n_check_stars": len(checks),
        "check_star_ids": checks,
        "median_scatter_mad_mmag_by_truncation": med,
        "median_abs_airmass_slope_mmag_by_truncation": med_s,
        "decision": decision,
        "per_check": per_check,
        "falsification": "median check-star MAD rises or falls by >1 mmag from full to 50% truncation",
    }


def main():
    print("loading", flush=True)
    fids, mag, sat, xy, gmag = m.load_mags()
    comps = pd.read_csv(m.PHOT / "comparison_stars_per_target.csv")
    at = pd.read_csv(m.PHOT / "active_targets.csv")
    sus = set()
    if (m.PHOT / "suspected_variables.csv").is_file():
        sus = set(pd.read_csv(m.PHOT / "suspected_variables.csv")["catalog_id"].astype(str))
    q2 = measure_q2_weighted(fids, mag, sat, comps, at, sus, xy)
    (m.OUT / "PRE_IMPL_01_Q2.json").write_text(json.dumps(q2, indent=2), encoding="utf-8")
    print("decision", q2["decision"])
    print("scatter", q2["median_scatter_mad_mmag_by_truncation"])
    print("slopes", q2["median_abs_airmass_slope_mmag_by_truncation"])


if __name__ == "__main__":
    main()
