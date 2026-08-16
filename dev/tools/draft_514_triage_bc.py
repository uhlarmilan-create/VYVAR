"""DRAFT-514-TRIAGE B+C measurements after Phase 2A (no ensemble-size cut)."""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src_py"))

from comp_weights import sigma_eff_mag, weight_from_sigma_eff  # noqa: E402
from photometry_core import ensemble_normalize  # noqa: E402

PHOT = (
    ROOT
    / "Archive"
    / "Drafts"
    / "draft_000514"
    / "platesolve"
    / "NoFilter_60_2"
    / "photometry"
)
PROC = (
    ROOT
    / "Archive"
    / "Drafts"
    / "draft_000514"
    / "detrended_aligned"
    / "lights"
    / "NoFilter_60_2"
)
OUT = ROOT / "dev" / "results"


def _sha() -> str:
    try:
        import subprocess

        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _neff(w: np.ndarray) -> float:
    w = w[np.isfinite(w) & (w > 0)]
    if w.size == 0:
        return 0.0
    s = float(np.sum(w))
    s2 = float(np.sum(w * w))
    if s2 <= 0:
        return 0.0
    return (s * s) / s2


def _cum_count(w: np.ndarray, frac: float) -> int:
    w = np.sort(w[np.isfinite(w) & (w > 0)])[::-1]
    if w.size == 0:
        return 0
    c = np.cumsum(w)
    thr = frac * c[-1]
    return int(np.searchsorted(c, thr, side="left") + 1)


def weights_for_target(sub: pd.DataFrame, target_bprp: float, c_col: float, c_dist: float) -> pd.DataFrame:
    rows = []
    tra = float(pd.to_numeric(sub.get("ra_deg"), errors="coerce").median())
    tde = float(pd.to_numeric(sub.get("dec_deg"), errors="coerce").median())
    for _, r in sub.iterrows():
        cid = str(r.get("catalog_id", ""))
        rms = float(pd.to_numeric(r.get("comp_rms"), errors="coerce"))
        bpr = float(pd.to_numeric(r.get("bp_rp"), errors="coerce"))
        ra = float(pd.to_numeric(r.get("ra_deg"), errors="coerce"))
        dec = float(pd.to_numeric(r.get("dec_deg"), errors="coerce"))
        db = abs(bpr - target_bprp) if math.isfinite(bpr) and math.isfinite(target_bprp) else 0.0
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
            c_col_mag_per_bprp=c_col,
            c_dist_mag_per_deg=c_dist,
        )
        w = weight_from_sigma_eff(se)
        rows.append(
            {
                "catalog_id": cid,
                "bp_rp": bpr,
                "delta_bprp": db,
                "r_deg": rdeg,
                "comp_rms": rms,
                "sigma_eff": se,
                "weight": w,
            }
        )
    return pd.DataFrame(rows)


def measure_b1_c1(comps: pd.DataFrame, at: pd.DataFrame, c_col: float, c_dist: float) -> dict:
    rows = []
    for _, trow in at.iterrows():
        tid = str(trow.get("catalog_id", ""))
        name = str(trow.get("name", tid))
        sub = comps[comps["target_catalog_id"].astype(str) == tid]
        if sub.empty:
            continue
        tb = float(pd.to_numeric(trow.get("bp_rp"), errors="coerce"))
        if not math.isfinite(tb):
            tb = float(pd.to_numeric(sub.get("target_bp_rp"), errors="coerce").iloc[0])
        wt = weights_for_target(sub, tb, c_col, c_dist)
        w = wt["weight"].to_numpy(dtype=float)
        wsum = float(np.nansum(w))
        heaviest = float(np.nanmax(w) / wsum) if wsum > 0 else float("nan")
        # colour balance
        bpr = wt["bp_rp"].to_numpy(dtype=float)
        ok = np.isfinite(bpr) & np.isfinite(w) & (w > 0)
        if ok.any() and math.isfinite(tb):
            dcol = float(np.sum(w[ok] * bpr[ok]) / np.sum(w[ok]) - tb)
            n_blue = int(np.sum(ok & (bpr < tb)))
            n_red = int(np.sum(ok & (bpr > tb)))
        else:
            dcol = float("nan")
            n_blue = n_red = 0
        rows.append(
            {
                "target_catalog_id": tid,
                "name": name,
                "n_comps": int(len(wt)),
                "N_eff": _neff(w),
                "n_for_50pct_weight": _cum_count(w, 0.50),
                "n_for_90pct_weight": _cum_count(w, 0.90),
                "n_for_99pct_weight": _cum_count(w, 0.99),
                "n_for_999pct_weight": _cum_count(w, 0.999),
                "heaviest_frac": heaviest,
                "delta_colour_ensemble": dcol,
                "n_comps_bluer_than_target": n_blue,
                "n_comps_redder_than_target": n_red,
                "target_bp_rp": tb,
            }
        )
    return {"commit_sha": _sha(), "unit_weight": "1/sigma_eff^2", "targets": rows}


def measure_b2(
    comps: pd.DataFrame,
    at: pd.DataFrame,
    names: list[str],
    c_col: float,
    c_dist: float,
    n_frames: int = 40,
) -> dict:
    """Truncation sensitivity using synthetic inst mags + real weights (measuring instrument)."""
    out: dict = {"commit_sha": _sha(), "targets": {}}
    for name in names:
        hit = at[at["name"].astype(str) == name]
        if hit.empty:
            out["targets"][name] = {"error": "not_in_active"}
            continue
        tid = str(hit.iloc[0]["catalog_id"])
        sub = comps[comps["target_catalog_id"].astype(str) == tid]
        if sub.empty:
            out["targets"][name] = {"error": "no_comps"}
            continue
        tb = float(pd.to_numeric(hit.iloc[0].get("bp_rp"), errors="coerce"))
        wt = weights_for_target(sub, tb, c_col, c_dist).sort_values("weight", ascending=False)
        ids = wt["catalog_id"].astype(str).tolist()
        wmap = dict(zip(ids, wt["weight"].tolist(), strict=False))
        # Synthetic: common-mode airmass + individual noise scaled by rms
        rng = np.random.default_rng(abs(hash(name)) % (2**32))
        common = 0.01 * np.linspace(0, 1, n_frames)
        comps_m = {}
        cat = {}
        qual = {}
        for _, r in wt.iterrows():
            cid = str(r["catalog_id"])
            rms = float(r["comp_rms"]) if math.isfinite(float(r["comp_rms"])) else 0.02
            comps_m[cid] = 11.0 + common + rng.normal(0, rms, size=n_frames)
            cat[cid] = 10.0
            qual[cid] = {}
        target = 12.0 + common + rng.normal(0, 0.005, size=n_frames)

        def _run(keep_ids: list[str]) -> tuple[np.ndarray, float]:
            t0 = time.perf_counter()
            cm = {k: comps_m[k] for k in keep_ids}
            cq = {k: qual[k] for k in keep_ids}
            cc = {k: cat[k] for k in keep_ids}
            wm = {k: wmap[k] for k in keep_ids}
            mag_c, _, _ = ensemble_normalize(
                target, cm, cc, cq, comp_weight_map=wm, n_comp_min=2, n_comp_max=10
            )
            return mag_c, time.perf_counter() - t0

        full_ids = ids
        mag_full, t_full = _run(full_ids)
        trunc_rows = []
        for frac in (0.999, 0.99, 0.95, 0.90, 0.50):
            n_keep = _cum_count(wt["weight"].to_numpy(dtype=float), frac)
            keep = ids[: max(n_keep, 2)]
            mag_t, t_t = _run(keep)
            d = mag_t - mag_full
            fin = np.isfinite(d)
            trunc_rows.append(
                {
                    "cum_weight_frac": frac,
                    "n_comps_kept": len(keep),
                    "zp_diff_rms_mmag": float(np.nanstd(d[fin]) * 1000.0) if fin.any() else None,
                    "zp_diff_mean_mmag": float(np.nanmean(d[fin]) * 1000.0) if fin.any() else None,
                    "wall_clock_s": t_t,
                    "wall_clock_full_s": t_full,
                }
            )
        out["targets"][name] = {
            "n_comps_full": len(ids),
            "N_eff": _neff(wt["weight"].to_numpy(dtype=float)),
            "truncation": trunc_rows,
            "note": "Synthetic inst mags; cumulative-weight truncation is a measuring instrument only",
        }
    return out


def measure_c2(lc_dir: Path, b1_rows: list[dict]) -> dict:
    """Regress LC residual vs airmass; correlate slope with delta_colour_ensemble."""
    rows = []
    for r in b1_rows:
        tid = r["target_catalog_id"]
        # find lightcurve
        hits = list(lc_dir.glob(f"*{tid[-8:]}*.csv")) + list(lc_dir.glob(f"lightcurve_*{tid}*.csv"))
        if not hits:
            # try by name
            name = str(r.get("name", "")).replace(" ", "_")
            hits = list(lc_dir.glob(f"*{name}*.csv"))
        if not hits:
            continue
        lc = pd.read_csv(hits[0])
        # mag / airmass columns
        mag_col = next((c for c in ("mag_calib", "mag", "delta_mag") if c in lc.columns), None)
        am_col = next((c for c in ("airmass", "AIRMASS") if c in lc.columns), None)
        if mag_col is None or am_col is None:
            continue
        m = pd.to_numeric(lc[mag_col], errors="coerce")
        am = pd.to_numeric(lc[am_col], errors="coerce")
        ok = m.notna() & am.notna()
        if int(ok.sum()) < 8:
            continue
        # residual about median
        resid = m[ok] - float(m[ok].median())
        x = am[ok].to_numpy(dtype=float)
        y = resid.to_numpy(dtype=float)
        slope = float(np.polyfit(x, y, 1)[0])
        rows.append(
            {
                "name": r["name"],
                "target_catalog_id": tid,
                "delta_colour_ensemble": r["delta_colour_ensemble"],
                "airmass_slope_mag_per_airmass": slope,
                "airmass_slope_mmag_per_airmass": slope * 1000.0,
                "n_epochs": int(ok.sum()),
                "lc_file": str(hits[0].name),
            }
        )
    if len(rows) >= 5:
        dc = np.array([r["delta_colour_ensemble"] for r in rows], dtype=float)
        sl = np.array([r["airmass_slope_mag_per_airmass"] for r in rows], dtype=float)
        ok = np.isfinite(dc) & np.isfinite(sl)
        if int(ok.sum()) >= 5:
            corr = float(np.corrcoef(dc[ok], sl[ok])[0, 1])
            # slope of slope vs delta_colour: mmag/airmass per BP-RP
            fit = np.polyfit(dc[ok], sl[ok], 1)
            coeff = float(fit[0]) * 1000.0
        else:
            corr = coeff = float("nan")
    else:
        corr = coeff = float("nan")
    return {
        "commit_sha": _sha(),
        "n_targets_with_lc": len(rows),
        "corr_slope_vs_delta_colour": corr,
        "coeff_mmag_per_bprp_per_airmass": coeff,
        "compare_k2_literature_clear": "NONE (CLEAR)",
        "compare_c_col_psf_mag_per_bprp": 0.029485,
        "targets": rows,
    }


def main() -> None:
    from comp_weights import C_COL_PSF_REFRACTIVE_MAG_PER_BPRP

    c_col = float(C_COL_PSF_REFRACTIVE_MAG_PER_BPRP)
    c_dist = 0.0
    comps = pd.read_csv(PHOT / "comparison_stars_per_target.csv")
    at = pd.read_csv(PHOT / "active_targets.csv")
    b1 = measure_b1_c1(comps, at, c_col, c_dist)
    (OUT / "DRAFT_514_TRIAGE_B1.json").write_text(json.dumps(b1, indent=2), encoding="utf-8")
    (OUT / "DRAFT_514_TRIAGE_C1.json").write_text(
        json.dumps(
            {
                "commit_sha": b1["commit_sha"],
                "delta_colour_ensemble_unit": "BP-RP",
                "targets": [
                    {
                        "name": t["name"],
                        "delta_colour_ensemble": t["delta_colour_ensemble"],
                        "n_blue": t["n_comps_bluer_than_target"],
                        "n_red": t["n_comps_redder_than_target"],
                        "target_bp_rp": t["target_bp_rp"],
                    }
                    for t in b1["targets"]
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    # summary stats
    ne = [t["N_eff"] for t in b1["targets"]]
    dc = [t["delta_colour_ensemble"] for t in b1["targets"] if math.isfinite(t["delta_colour_ensemble"])]
    print("B1 n_targets", len(b1["targets"]))
    print("N_eff median", float(np.median(ne)) if ne else None, "min", float(np.min(ne)) if ne else None)
    print("delta_colour median", float(np.median(dc)) if dc else None, "worst", max(dc, key=abs) if dc else None)

    b2 = measure_b2(comps, at, ["BO CVn", "FW CVn", "R CVn"], c_col, c_dist)
    (OUT / "DRAFT_514_TRIAGE_B2.json").write_text(json.dumps(b2, indent=2), encoding="utf-8")
    print("B2", json.dumps(b2, indent=2)[:1500])

    lc_dir = PHOT / "lightcurves"
    c2 = measure_c2(lc_dir, b1["targets"])
    (OUT / "DRAFT_514_TRIAGE_C2.json").write_text(json.dumps(c2, indent=2), encoding="utf-8")
    print("C2 n", c2["n_targets_with_lc"], "corr", c2["corr_slope_vs_delta_colour"], "coeff", c2["coeff_mmag_per_bprp_per_airmass"])


if __name__ == "__main__":
    main()
