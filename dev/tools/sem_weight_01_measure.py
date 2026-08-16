"""SEM-WEIGHT-01: measure weighted vs unweighted ensemble SEM on draft 514.

Does NOT change exported error bars. Report-only harness.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src_py"))

from sigma_floor_core import c4_small_sample, ensemble_sem_mag_from_residuals  # noqa: E402
from comp_weights import (  # noqa: E402
    C_COL_PSF_REFRACTIVE_MAG_PER_BPRP,
    sigma_eff_mag,
    weight_from_sigma_eff,
)

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
# WIDE_ERR_LOC_01_results.json
WIDE_ERR_R_G10 = 2.053633278465339  # mag_center 10.25
WIDE_ERR_R_BRIGHT = 2.308488592760189  # mag_center 8.25


def _sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, cwd=str(ROOT)
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def neff(w: np.ndarray) -> float:
    w = w[np.isfinite(w) & (w > 0)]
    if w.size == 0:
        return float("nan")
    s = float(np.sum(w))
    s2 = float(np.sum(w * w))
    if s2 <= 0:
        return float("nan")
    return (s * s) / s2


def sem_reliability_weighted(x: np.ndarray, w: np.ndarray) -> tuple[float, float, float]:
    """Empirical SEM of a weighted mean for reliability / inverse-variance weights.

    Estimator (reduces to unweighted s/sqrt(n) when all w equal):

        mu = sum(w x) / sum(w)
        V1 = sum(w), V2 = sum(w^2)
        N_eff = V1^2 / V2
        s^2 = sum(w (x-mu)^2) / (V1 - V2/V1)   # unbiased reliability variance
        SEM = s / sqrt(N_eff)

    Citation: same algebra as the reliability-weight sample variance in
    Wikipedia "Weighted arithmetic mean" / Cochran (1977) survey sampling;
    SE of the mean uses N_eff in place of n. Chosen because production weights
    are inverse-variance style (w=1/sigma_eff^2) used as reliability weights on
    an empirical residual population, not as frequency counts, and this form
    matches the current unweighted estimator when weights are equal.

    Returns (sem, N_eff, s_weighted).
    """
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x = x[m]
    w = w[m]
    n = int(x.size)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    v1 = float(np.sum(w))
    v2 = float(np.sum(w * w))
    if v1 <= 0 or v2 <= 0:
        return float("nan"), float("nan"), float("nan")
    n_eff = (v1 * v1) / v2
    mu = float(np.sum(w * x) / v1)
    denom = v1 - (v2 / v1)
    if denom <= 0:
        return float("nan"), n_eff, float("nan")
    s2 = float(np.sum(w * (x - mu) ** 2) / denom)
    if s2 < 0:
        return float("nan"), n_eff, float("nan")
    s = math.sqrt(s2)
    # Optional c4 on N_eff (rounded) for parity with production; apply after.
    return s / math.sqrt(n_eff), n_eff, s


def sem_ivw_model(w: np.ndarray) -> float:
    """Model SEM when w_i = 1/sigma_i^2 are the true inverse variances: 1/sqrt(sum w)."""
    w = w[np.isfinite(w) & (w > 0)]
    if w.size < 2:
        return float("nan")
    s = float(np.sum(w))
    if s <= 0:
        return float("nan")
    return 1.0 / math.sqrt(s)


def load_frame_tables() -> tuple[list[str], dict[str, dict[str, float]], dict[str, dict[str, bool]], dict[str, dict[str, float]]]:
    """Return (frame_ids, mag[frame][cid], sat[frame][cid], catmag[cid])."""
    files = sorted(PROC.glob("proc_*.csv"))
    # Prefer light frames only
    files = [f for f in files if "MASTER" not in f.name.upper()]
    mag: dict[str, dict[str, float]] = {}
    sat: dict[str, dict[str, bool]] = {}
    catmag: dict[str, float] = {}
    frame_ids: list[str] = []
    for fp in files:
        fid = fp.stem.replace("proc_", "")
        df = pd.read_csv(
            fp,
            usecols=lambda c: c
            in ("catalog_id", "mag", "flux", "likely_saturated", "catalog_mag", "phot_g_mean_mag"),
        )
        cid = df["catalog_id"].astype(str)
        m = pd.to_numeric(df["mag"], errors="coerce")
        # fallback from flux if mag missing
        flux = pd.to_numeric(df.get("flux"), errors="coerce")
        miss = ~np.isfinite(m.to_numpy(dtype=float)) & np.isfinite(flux.to_numpy(dtype=float)) & (flux.to_numpy(dtype=float) > 0)
        if miss.any():
            m = m.copy()
            m.loc[miss] = -2.5 * np.log10(flux.loc[miss].to_numpy(dtype=float))
        sat_col = pd.to_numeric(df.get("likely_saturated"), errors="coerce").fillna(0)
        cm = pd.to_numeric(df.get("catalog_mag"), errors="coerce")
        if cm.isna().all() and "phot_g_mean_mag" in df.columns:
            cm = pd.to_numeric(df["phot_g_mean_mag"], errors="coerce")
        mm: dict[str, float] = {}
        ss: dict[str, bool] = {}
        for i in range(len(df)):
            c = str(cid.iloc[i])
            mv = float(m.iloc[i])
            if math.isfinite(mv):
                mm[c] = mv
            ss[c] = bool(sat_col.iloc[i] >= 0.5)
            cv = float(cm.iloc[i]) if i < len(cm) else float("nan")
            if math.isfinite(cv) and c not in catmag:
                catmag[c] = cv
        mag[fid] = mm
        sat[fid] = ss
        frame_ids.append(fid)
    return frame_ids, mag, sat, catmag


def weights_for_comps(comps: pd.DataFrame, target_bprp: float) -> dict[str, float]:
    """Recompute production Phase-2A weights (CSV comp_weight is not per-target)."""
    c_col = float(C_COL_PSF_REFRACTIVE_MAG_PER_BPRP)
    c_dist = 0.0
    tra = float(pd.to_numeric(comps.get("ra_deg"), errors="coerce").median())
    tde = float(pd.to_numeric(comps.get("dec_deg"), errors="coerce").median())
    out: dict[str, float] = {}
    for _, r in comps.iterrows():
        cid = str(r["catalog_id"])
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
        out[cid] = weight_from_sigma_eff(se)
    return out


def measure_target(
    name: str,
    tid: str,
    comps: pd.DataFrame,
    target_bprp: float,
    frame_ids: list[str],
    mag: dict[str, dict[str, float]],
    sat: dict[str, dict[str, bool]],
    catmag: dict[str, float],
) -> dict:
    ids = comps["catalog_id"].astype(str).tolist()
    wmap = weights_for_comps(comps, target_bprp)

    # Night reference = median inst mag per comp
    ref: dict[str, float] = {}
    for cid in ids:
        vals = []
        for fid in frame_ids:
            if sat[fid].get(cid, False):
                continue
            mv = mag[fid].get(cid)
            if mv is not None and math.isfinite(mv):
                vals.append(mv)
        if vals:
            ref[cid] = float(np.median(vals))

    rows = []
    membership_mismatch = 0
    for fid in frame_ids:
        # Same exclusion as ensemble_normalize: finite mag, not saturated
        pairs: list[tuple[str, float]] = []
        for cid in ids:
            if sat[fid].get(cid, False):
                continue
            mv = mag[fid].get(cid)
            if mv is not None and math.isfinite(mv):
                pairs.append((cid, mv))
        if len(pairs) < 2:
            continue

        # SEM set (production): residuals m - ref for comps with ref
        resid_ids = []
        resid_vals = []
        for cid, mv in pairs:
            if cid in ref and math.isfinite(ref[cid]):
                resid_ids.append(cid)
                resid_vals.append(mv - ref[cid])

        # ZP set (production): finite catalog mag + positive weight
        zp_ids = []
        zp_x = []
        zp_w = []
        for cid, mv in pairs:
            cm = catmag.get(cid, float("nan"))
            # comps CSV catalog_mag as backup
            w = wmap.get(cid, float("nan"))
            if not (math.isfinite(cm) and math.isfinite(mv)):
                continue
            if not (math.isfinite(w) and w > 0):
                continue
            zp_ids.append(cid)
            zp_x.append(cm - mv)
            zp_w.append(w)

        if len(resid_vals) < 2 or len(zp_ids) < 2:
            continue

        # Membership: SEM residual set vs ZP-weighted set
        set_sem = set(resid_ids)
        set_zp = set(zp_ids)
        if set_sem != set_zp:
            membership_mismatch += 1

        sem_cur = float(ensemble_sem_mag_from_residuals(resid_vals))

        # A: same residuals as production SEM, but reliability-weighted + N_eff
        # Align weights onto resid_ids (0 => drop)
        rw = np.asarray([wmap.get(c, float("nan")) for c in resid_ids], dtype=float)
        rx = np.asarray(resid_vals, dtype=float)
        m = np.isfinite(rw) & (rw > 0)
        sem_w_same, ne_same, s_w = sem_reliability_weighted(rx[m], rw[m])
        # Apply c4(N_eff) like production applies c4(n)
        if math.isfinite(ne_same) and ne_same >= 2:
            c4e = c4_small_sample(max(2, int(round(ne_same))))
            if math.isfinite(c4e) and c4e > 0:
                sem_w_same_c4 = sem_w_same / c4e
            else:
                sem_w_same_c4 = sem_w_same
        else:
            sem_w_same_c4 = sem_w_same

        # B: consistent with weighted ZP mean of (cat - m)
        zx = np.asarray(zp_x, dtype=float)
        zw = np.asarray(zp_w, dtype=float)
        sem_w_zp, ne_zp, _ = sem_reliability_weighted(zx, zw)
        if math.isfinite(ne_zp) and ne_zp >= 2:
            c4z = c4_small_sample(max(2, int(round(ne_zp))))
            sem_w_zp_c4 = sem_w_zp / c4z if (math.isfinite(c4z) and c4z > 0) else sem_w_zp
        else:
            sem_w_zp_c4 = sem_w_zp

        sem_model = sem_ivw_model(zw)

        # Decomposition diagnostics on same residual set
        n = int(m.sum())
        n_eff = float(ne_same) if math.isfinite(ne_same) else float("nan")
        # unweighted std of residuals
        if n >= 2:
            mu_u = float(np.mean(rx[m]))
            std_u = float(np.sqrt(np.sum((rx[m] - mu_u) ** 2) / (n - 1)))
        else:
            std_u = float("nan")
        ratio_denom_only = math.sqrt(n / n_eff) if (math.isfinite(n_eff) and n_eff > 0 and n > 0) else float("nan")
        ratio_num_only = (s_w / std_u) if (math.isfinite(s_w) and math.isfinite(std_u) and std_u > 0) else float("nan")

        ratio_primary = (
            sem_w_same_c4 / sem_cur
            if (math.isfinite(sem_w_same_c4) and math.isfinite(sem_cur) and sem_cur > 0)
            else float("nan")
        )
        ratio_zp = (
            sem_w_zp_c4 / sem_cur
            if (math.isfinite(sem_w_zp_c4) and math.isfinite(sem_cur) and sem_cur > 0)
            else float("nan")
        )

        rows.append(
            {
                "frame": fid,
                "n": n,
                "N_eff": n_eff,
                "sem_current_mag": sem_cur,
                "sem_weighted_same_resid_mag": sem_w_same_c4,
                "sem_weighted_zp_mag": sem_w_zp_c4,
                "sem_ivw_model_mag": sem_model,
                "ratio_same_resid": ratio_primary,
                "ratio_zp_consistent": ratio_zp,
                "ratio_denom_only_sqrt_n_over_neff": ratio_denom_only,
                "ratio_num_only_sw_over_std": ratio_num_only,
                "n_zp": int(len(zp_ids)),
                "N_eff_zp": float(ne_zp) if math.isfinite(ne_zp) else None,
                "membership_sem_vs_zp_equal": set_sem == set_zp,
            }
        )

    if not rows:
        return {"name": name, "target_catalog_id": tid, "error": "no_epochs"}

    ratios = np.asarray([r["ratio_same_resid"] for r in rows], dtype=float)
    ratios = ratios[np.isfinite(ratios)]
    neffs = np.asarray([r["N_eff"] for r in rows], dtype=float)
    neffs = neffs[np.isfinite(neffs)]

    return {
        "name": name,
        "target_catalog_id": tid,
        "n_epochs": len(rows),
        "n_comps_pool": len(ids),
        "membership_mismatch_epochs": membership_mismatch,
        "ratio_same_resid_median": float(np.median(ratios)) if ratios.size else None,
        "ratio_same_resid_p16": float(np.percentile(ratios, 16)) if ratios.size else None,
        "ratio_same_resid_p84": float(np.percentile(ratios, 84)) if ratios.size else None,
        "ratio_same_resid_min": float(np.min(ratios)) if ratios.size else None,
        "ratio_same_resid_max": float(np.max(ratios)) if ratios.size else None,
        "N_eff_median": float(np.median(neffs)) if neffs.size else None,
        "n_median": float(np.median([r["n"] for r in rows])),
        "sem_current_median_mag": float(np.median([r["sem_current_mag"] for r in rows])),
        "sem_weighted_median_mag": float(
            np.median([r["sem_weighted_same_resid_mag"] for r in rows if math.isfinite(r["sem_weighted_same_resid_mag"])])
        ),
        "ratio_zp_consistent_median": float(
            np.nanmedian([r["ratio_zp_consistent"] for r in rows])
        ),
        "ratio_denom_only_median": float(
            np.nanmedian([r["ratio_denom_only_sqrt_n_over_neff"] for r in rows])
        ),
        "ratio_num_only_median": float(
            np.nanmedian([r["ratio_num_only_sw_over_std"] for r in rows])
        ),
        "epochs": rows,
    }


def main() -> None:
    t0 = time.perf_counter()
    sha = _sha()
    lc_dir = PHOT / "lightcurves"
    lc_files = sorted(lc_dir.glob("lightcurve_*.csv"))
    # exclude psf variants
    lc_files = [p for p in lc_files if "_psf" not in p.name and "_adaptive" not in p.name]
    at = pd.read_csv(PHOT / "active_targets.csv")
    comps_all = pd.read_csv(PHOT / "comparison_stars_per_target.csv")

    print(f"loading {len(list(PROC.glob('proc_*.csv')))} proc frames...", flush=True)
    frame_ids, mag, sat, catmag = load_frame_tables()
    # Enrich catmag from comps table
    for _, r in comps_all.drop_duplicates("catalog_id").iterrows():
        cid = str(r["catalog_id"])
        cm = float(pd.to_numeric(r.get("catalog_mag"), errors="coerce"))
        if not math.isfinite(cm):
            cm = float(pd.to_numeric(r.get("phot_g_mean_mag"), errors="coerce"))
        if math.isfinite(cm):
            catmag[cid] = cm

    print(f"frames={len(frame_ids)} lcs={len(lc_files)}", flush=True)
    targets = []
    for fp in lc_files:
        tid = fp.stem.replace("lightcurve_", "")
        hit = at[at["catalog_id"].astype(str) == tid]
        name = str(hit.iloc[0]["name"]) if len(hit) else tid
        tb = float(pd.to_numeric(hit.iloc[0].get("bp_rp"), errors="coerce")) if len(hit) else float("nan")
        sub = comps_all[comps_all["target_catalog_id"].astype(str) == tid]
        if sub.empty:
            continue
        if not math.isfinite(tb):
            tb = float(pd.to_numeric(sub.get("target_bp_rp"), errors="coerce").iloc[0])
        print(f"  measure {name} n_comps={len(sub)} bp_rp={tb:.3f}", flush=True)
        targets.append(
            measure_target(name, tid, sub, tb, frame_ids, mag, sat, catmag)
        )

    # Draft-wide epoch pool
    all_ratios = []
    all_neff = []
    all_pairs = []  # (ratio, N_eff, name)
    for t in targets:
        if "error" in t:
            continue
        for e in t["epochs"]:
            r = e["ratio_same_resid"]
            ne = e["N_eff"]
            if math.isfinite(r) and math.isfinite(ne):
                all_ratios.append(r)
                all_neff.append(ne)
                all_pairs.append((r, ne, t["name"]))

    ar = np.asarray(all_ratios, dtype=float)
    an = np.asarray(all_neff, dtype=float)

    # Dependence: bin by N_eff
    bins = []
    for lo, hi in [(0, 100), (100, 150), (150, 200), (200, 300), (300, 1000)]:
        m = (an >= lo) & (an < hi)
        if m.sum() < 5:
            continue
        bins.append(
            {
                "N_eff_lo": lo,
                "N_eff_hi": hi,
                "n_epochs": int(m.sum()),
                "ratio_median": float(np.median(ar[m])),
                "ratio_p16": float(np.percentile(ar[m], 16)),
                "ratio_p84": float(np.percentile(ar[m], 84)),
            }
        )

    # Correlation ratio vs N_eff
    corr = float(np.corrcoef(an, ar)[0, 1]) if ar.size >= 5 else float("nan")

    # Extremes by target median
    by_tgt = [
        (t["name"], t.get("ratio_same_resid_median"), t.get("N_eff_median"))
        for t in targets
        if t.get("ratio_same_resid_median") is not None
    ]
    by_tgt_sorted = sorted(by_tgt, key=lambda x: x[1] if x[1] is not None else 0)

    med_ratio = float(np.median(ar)) if ar.size else None
    # Compare to WIDE-ERR
    wide = WIDE_ERR_R_G10
    if med_ratio is not None:
        if abs(med_ratio - 1.0) < 0.15:
            named = "Explains none of it"
        elif abs(med_ratio - wide) / wide < 0.25 and med_ratio > 1.3:
            named = "Explains it"
        elif med_ratio > 1.15:
            named = "Explains part of it"
        else:
            named = "Explains none of it"
    else:
        named = "Explains none of it"

    residual_deficit = None
    if med_ratio is not None and med_ratio > 0:
        # If SEM was under-reported by med_ratio, correcting it reduces R by that factor
        # Remaining R ? wide / med_ratio if SEM fully explained the gap and SEM dominated.
        # More carefully: report wide and ratio separately.
        residual_deficit = float(wide / med_ratio)

    out = {
        "commit_sha": sha,
        "draft": 514,
        "unit_sem": "mag",
        "estimators": {
            "sem_current": "ensemble_sem_mag_from_residuals: unweighted s_ddof1/c4(n)/sqrt(n) on (m-med_night)",
            "sem_weighted_primary": (
                "Reliability-weight SEM on SAME residuals as production, with production "
                "w=1/sigma_eff^2 (comp_weight): s_w/c4(round(N_eff))/sqrt(N_eff), "
                "s_w^2 = sum(w(x-mu)^2)/(V1-V2/V1). Reduces to current formula when w equal."
            ),
            "sem_weighted_zp": "Same reliability SEM but on ZP offsets (catalog_mag - m), the quantity actually averaged with weights",
            "sem_ivw_model": "1/sqrt(sum w); assumes sigma_eff are exact residual sigmas",
            "citations": [
                "Cochran, W.G. (1977) Sampling Techniques - reliability weights",
                "Wikipedia Weighted arithmetic mean - reliability-weight variance",
                "Bevington & Robinson - IVW model SE = 1/sqrt(sum w) when w=1/sigma^2 known",
            ],
        },
        "weight_source": "recomputed sigma_eff per target (CSV comp_weight identical across targets; not used)",
        "wide_err_R_G10": WIDE_ERR_R_G10,
        "wide_err_R_bright_G8p25": WIDE_ERR_R_BRIGHT,
        "wide_err_R_note": "WIDE_ERR_LOC_01 G10.25 R=2.054; task ~2.3x matches bright bin G8.25 R=2.31",
        "named_outcome": named,
        "residual_R_if_sem_scaled_by_median_ratio": residual_deficit,
        "n_targets": len([t for t in targets if "error" not in t]),
        "n_epochs_total": int(ar.size),
        "ratio_same_resid_distribution": {
            "median": med_ratio,
            "p16": float(np.percentile(ar, 16)) if ar.size else None,
            "p84": float(np.percentile(ar, 84)) if ar.size else None,
            "min": float(np.min(ar)) if ar.size else None,
            "max": float(np.max(ar)) if ar.size else None,
            "mean": float(np.mean(ar)) if ar.size else None,
        },
        "ratio_vs_N_eff_corr": corr,
        "ratio_by_N_eff_bin": bins,
        "decomposition_median_across_targets": {
            "ratio_denom_only_sqrt_n_over_neff": float(
                np.nanmedian([t.get("ratio_denom_only_median") for t in targets])
            ),
            "ratio_num_only_sw_over_std": float(
                np.nanmedian([t.get("ratio_num_only_median") for t in targets])
            ),
            "product_should_approx_ratio": None,
        },
        "extremes_by_target_median_ratio": {
            "lowest": by_tgt_sorted[:5],
            "highest": by_tgt_sorted[-5:][::-1],
        },
        "c4_note": (
            "Production uses c4(n) with n=len(residuals)~1292 -> c4~0.9998. "
            "Weighted form should use c4(N_eff); N_eff~150-350 makes c4 matter at the ~0.1% level still, "
            "relevant only when N_eff is small (tens)."
        ),
        "targets": [{k: v for k, v in t.items() if k != "epochs"} for t in targets],
        "targets_with_epochs": targets,
        "wall_s": time.perf_counter() - t0,
    }
    d = out["decomposition_median_across_targets"]
    if d["ratio_denom_only_sqrt_n_over_neff"] and d["ratio_num_only_sw_over_std"]:
        d["product_should_approx_ratio"] = float(
            d["ratio_denom_only_sqrt_n_over_neff"] * d["ratio_num_only_sw_over_std"]
        )

    (OUT / "SEM_WEIGHT_01_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    # Slim copy without per-epoch for quick read
    slim = {k: v for k, v in out.items() if k != "targets_with_epochs"}
    (OUT / "SEM_WEIGHT_01_summary.json").write_text(json.dumps(slim, indent=2), encoding="utf-8")
    print("named_outcome", named)
    print("ratio median", med_ratio, "p16", out["ratio_same_resid_distribution"]["p16"], "p84", out["ratio_same_resid_distribution"]["p84"])
    print("decomp", out["decomposition_median_across_targets"])
    print("wrote", OUT / "SEM_WEIGHT_01_results.json")


if __name__ == "__main__":
    main()
