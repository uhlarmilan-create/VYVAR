#!/usr/bin/env python3
"""D10-1: CV vs CR band-letter measurement on frozen era-516 NoFilter snapshot."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src_py"
sys.path.insert(0, str(SRC))

from gaia_johnson import (  # noqa: E402
    BPRP_MAX,
    BPRP_MIN,
    G_MAG_MAX,
    G_MAG_MIN,
    GDR3_TABLE59_COEFFS,
    transform_gaia_to_johnson,
)
from photometry_core import _flux_to_mag  # noqa: E402

SNAP = REPO / "Archive" / "Drafts" / "draft_000516_snapshot_era03_20260820"
SETUP = "NoFilter_60_2"
PS = SNAP / "platesolve" / SETUP
PROC_DIR = SNAP / "detrended_aligned" / "lights" / SETUP
OUT = REPO / "dev" / "results" / "context" / "session_20260821_d10_1"
OUT.mkdir(parents=True, exist_ok=True)

CLEAN_STATES = frozenset({"DETECTED_P1", "DETECTED_P2"})
WIDEN_STATES = frozenset({"DETECTED_P1", "DETECTED_P2", "DAO_ONLY", "EDGE"})


def _read_csv(path: Path) -> pd.DataFrame:
    """Load CSV with string catalog_id to avoid int64 float rounding."""
    try:
        return pd.read_csv(path, dtype={"catalog_id": str}, low_memory=False)
    except (ValueError, TypeError):
        df = pd.read_csv(path, low_memory=False)
        if "catalog_id" in df.columns:
            df["catalog_id"] = df["catalog_id"].map(_cid)
        return df


def _ols_slope_intercept(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """Return a, b, b_stderr, combined residual std."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 3:
        return float("nan"), float("nan"), float("nan"), float("nan")
    xm = float(np.mean(x))
    ym = float(np.mean(y))
    ssx = float(np.sum((x - xm) ** 2))
    if ssx <= 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    b = float(np.sum((x - xm) * (y - ym)) / ssx)
    a = ym - b * xm
    yhat = a + b * x
    resid = y - yhat
    dof = max(int(x.size - 2), 1)
    s2 = float(np.sum(resid**2) / dof)
    b_se = math.sqrt(s2 / ssx) if ssx > 0 else float("nan")
    return a, b, b_se, math.sqrt(s2)


def _cid(v: object) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    if not s or s.lower() == "nan":
        return ""
    if s.isdigit():
        return s
    try:
        if isinstance(v, (int, np.integer)):
            return str(int(v))
    except (TypeError, ValueError):
        pass
    try:
        f = float(s)
        if math.isfinite(f):
            return str(int(round(f)))
    except (TypeError, ValueError):
        pass
    return s


def _filter_ms(ms: pd.DataFrame, *, widen_isolation: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    ms = ms.copy()
    ms["_cid"] = ms["catalog_id"].map(_cid)
    states = WIDEN_STATES if widen_isolation else CLEAN_STATES
    excl_rows: list[dict] = []
    keep_mask = np.ones(len(ms), dtype=bool)
    cid = ms["_cid"]

    def _excl(mask: pd.Series, reason: str) -> None:
        nonlocal keep_mask
        drop = mask & keep_mask
        for c in cid[drop]:
            if c:
                excl_rows.append({"catalog_id": c, "criterion": reason})
        keep_mask &= ~mask

    _excl(ms["zone"].astype(str) != "linear", "zone_not_linear")
    _excl(ms["vsx_known_variable"].astype(bool), "vsx_known_variable")
    _excl(~ms["source_state"].astype(str).isin(states), "source_state_not_clean")
    g = pd.to_numeric(ms["phot_g_mean_mag"], errors="coerce")
    bprp = pd.to_numeric(ms["bp_rp"], errors="coerce")
    _excl(~np.isfinite(g) | (g < G_MAG_MIN) | (g > G_MAG_MAX), "g_mag_outside_table59")
    _excl(~np.isfinite(bprp) | (bprp < BPRP_MIN) | (bprp > BPRP_MAX), "bprp_outside_table59")

    return ms.loc[keep_mask].copy(), pd.DataFrame(excl_rows)


def _load_proc_series(comp_ids: set[str]) -> tuple[list[str], dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Scan proc CSVs; return frame keys, mag_inst per cid, airmass, jd."""
    proc_files = sorted(PROC_DIR.glob("proc_*.csv"))
    chunks: list[pd.DataFrame] = []
    for i, pf in enumerate(proc_files):
        df = pd.read_csv(
            pf,
            dtype={"catalog_id": str},
            usecols=lambda c: c in {"catalog_id", "flux", "airmass", "jd_mid", "photometry_ok"},
            low_memory=False,
        )
        if df.empty:
            continue
        df["frame"] = i
        df["_cid"] = df["catalog_id"].map(_cid)
        df = df[df["_cid"] != ""]
        flux = pd.to_numeric(df["flux"], errors="coerce")
        ok = pd.to_numeric(df["photometry_ok"], errors="coerce").fillna(1).astype(bool)
        mag = flux.map(lambda f: _flux_to_mag(float(f)) if math.isfinite(f) and f > 0 else float("nan"))
        mag[~ok] = float("nan")
        chunks.append(
            pd.DataFrame(
                {
                    "frame": df["frame"].to_numpy(int),
                    "_cid": df["_cid"].to_numpy(str),
                    "mag_inst": mag.to_numpy(float),
                    "airmass": pd.to_numeric(df["airmass"], errors="coerce").to_numpy(float),
                    "jd_mid": pd.to_numeric(df["jd_mid"], errors="coerce").to_numpy(float),
                }
            )
        )
    if not chunks:
        return [], {}, np.array([]), np.array([])
    long = pd.concat(chunks, ignore_index=True)
    frame_keys = [pf.name for pf in proc_files]
    n_frames = len(frame_keys)
    airmass = (
        long.groupby("frame", sort=True)["airmass"].first().reindex(range(n_frames)).to_numpy(float)
    )
    jd_mid = long.groupby("frame", sort=True)["jd_mid"].first().reindex(range(n_frames)).to_numpy(float)
    wide = long.pivot_table(index="frame", columns="_cid", values="mag_inst", aggfunc="first")
    wide = wide.reindex(range(n_frames))
    mag_arr = {str(c): wide[c].to_numpy(float) for c in wide.columns}
    return frame_keys, mag_arr, airmass, jd_mid


def _frame_zp_median(
    comp_id_list: list[str],
    comp_cat: dict[str, float],
    comp_mat: np.ndarray,
) -> np.ndarray:
    """Per-frame zp offset = median(G_comp - mag_inst_comp), matching ensemble_normalize fallback."""
    n_comp, n_frames = comp_mat.shape
    cat = np.asarray([comp_cat[c] for c in comp_id_list], dtype=np.float64)
    zp = np.full(n_frames, np.nan, dtype=np.float64)
    for f in range(n_frames):
        inst = comp_mat[:, f]
        m = np.isfinite(inst) & np.isfinite(cat)
        if int(m.sum()) < 3:
            continue
        zp[f] = float(np.nanmedian(cat[m] - inst[m]))
    return zp


def main() -> None:
    ms = _read_csv(PS / "masterstars_full_match.csv")
    comps = _read_csv(PS / "comparison_stars.csv")
    comp_ids = {_cid(x) for x in comps["catalog_id"] if _cid(x)}

    stars, excl = _filter_ms(ms, widen_isolation=False)
    widen_note = ""
    bprp_span = float("nan")
    if len(stars) >= 3:
        bprp_span = float(stars["bp_rp"].max() - stars["bp_rp"].min())
    if not math.isfinite(bprp_span) or bprp_span < 1.0:
        stars_w, excl_w = _filter_ms(ms, widen_isolation=True)
        if len(stars_w) > len(stars):
            stars = stars_w
            excl = pd.concat([excl, excl_w], ignore_index=True)
            widen_note = "isolation widened: allowed DAO_ONLY+EDGE (dropped BLENDED/VSX/linear guards unchanged)"
            bprp_span = float(stars["bp_rp"].max() - stars["bp_rp"].min())

    frame_keys, mag_inst_map, airmass, _jd = _load_proc_series(comp_ids)
    n_frames = len(frame_keys)

    comp_catalog_mag = {
        _cid(r.catalog_id): float(r.phot_g_mean_mag if pd.notna(r.phot_g_mean_mag) else r.mag)
        for r in comps.itertuples()
        if _cid(r.catalog_id)
    }
    comp_id_list = sorted(cid for cid in comp_ids if cid in mag_inst_map)
    comp_mat = np.vstack([mag_inst_map[c] for c in comp_id_list])
    zp = _frame_zp_median(comp_id_list, comp_catalog_mag, comp_mat)

    rows: list[dict] = []
    min_frames = max(10, n_frames // 4)
    for r in stars.itertuples():
        cid = _cid(r.catalog_id)
        if cid not in mag_inst_map:
            excl = pd.concat(
                [excl, pd.DataFrame([{"catalog_id": cid, "criterion": "no_proc_photometry"}])],
                ignore_index=True,
            )
            continue
        arr = mag_inst_map[cid]
        mag_calib = arr + zp
        if int(np.isfinite(mag_calib).sum()) < min_frames:
            excl = pd.concat(
                [
                    excl,
                    pd.DataFrame(
                        [{"catalog_id": cid, "criterion": f"fewer_than_{min_frames}_finite_frames"}]
                    ),
                ],
                ignore_index=True,
            )
            continue
        night_mag = float(np.nanmedian(mag_calib))
        if not math.isfinite(night_mag):
            continue
        g = float(r.phot_g_mean_mag)
        bprp = float(r.bp_rp)
        tv = transform_gaia_to_johnson(g, bprp, "V")
        tr = transform_gaia_to_johnson(g, bprp, "RC")
        if not (tv.ok and tr.ok):
            excl = pd.concat(
                [excl, pd.DataFrame([{"catalog_id": cid, "criterion": "johnson_transform_fail"}])],
                ignore_index=True,
            )
            continue
        rows.append(
            {
                "catalog_id": cid,
                "name": str(r.name),
                "bp_rp": bprp,
                "g_mag": g,
                "mag_catalog_V": tv.johnson_mag,
                "mag_catalog_R": tr.johnson_mag,
                "mag_calib_raw_median": night_mag,
                "n_finite_frames": int(np.isfinite(mag_calib).sum()),
                "resid_V": night_mag - tv.johnson_mag,
                "resid_R": night_mag - tr.johnson_mag,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "star_residuals.csv", index=False)
    excl.to_csv(OUT / "exclusions.csv", index=False)

    x = df["bp_rp"].to_numpy(float)
    rv = df["resid_V"].to_numpy(float)
    rr = df["resid_R"].to_numpy(float)
    a_v, b_v, b_v_se, _ = _ols_slope_intercept(x, rv)
    a_r, b_r, b_r_se, _ = _ols_slope_intercept(x, rr)

    bv, br = abs(b_v), abs(b_r)
    ratio = max(bv, br) / min(bv, br) if min(bv, br) > 1e-12 else float("inf")
    abs_diff = abs(bv - br)
    comb_se = math.hypot(b_v_se, b_r_se)
    sign_b_diff = b_v - b_r

    if ratio >= 2.0 and abs_diff > 3.0 * comb_se:
        verdict = "CR" if br < bv else "CV"
    else:
        verdict = "INCONCLUSIVE"

    # Pinned BO/FW comp ensemble median BP-RP vs target BP-RP
    cst = _read_csv(PS / "photometry" / "comparison_stars_per_target.csv")
    if "target_catalog_id" in cst.columns:
        cst["target_catalog_id"] = cst["target_catalog_id"].map(_cid)
    pin_targets = {
        "BO CVn": "1498613634033133184",
        "FW CVn": "1497343732462852864",
        "GH CVn": "1498804639818507904",
    }
    submission: dict[str, dict] = {}
    for tname, tcid in pin_targets.items():
        sub = cst[cst["target_catalog_id"].map(_cid) == tcid]
        mrow = ms[ms["catalog_id"].map(_cid) == tcid]
        target_bprp = float(mrow.iloc[0]["bp_rp"]) if len(mrow) else float("nan")
        comp_med = float(pd.to_numeric(sub["bp_rp"], errors="coerce").median()) if len(sub) else float("nan")
        d_bprp = target_bprp - comp_med if math.isfinite(target_bprp) and math.isfinite(comp_med) else float("nan")
        submission[tname] = {
            "target_catalog_id": tcid,
            "target_bp_rp": target_bprp,
            "comp_ensemble_median_bp_rp": comp_med,
            "delta_bp_rp_target_minus_comp": d_bprp,
            "systematic_V_mmag_if_bV": 1000.0 * b_v * d_bprp if math.isfinite(d_bprp) else float("nan"),
            "systematic_R_mmag_if_bR": 1000.0 * b_r * d_bprp if math.isfinite(d_bprp) else float("nan"),
        }

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    for ax, resid, band, b, a in [
        (axes[0], rv, "V", b_v, a_v),
        (axes[1], rr, "R", b_r, a_r),
    ]:
        ax.scatter(x, resid * 1000.0, s=8, alpha=0.5, c="tab:blue")
        xs = np.linspace(float(np.min(x)), float(np.max(x)), 50)
        ax.plot(xs, (a + b * xs) * 1000.0, "r-", lw=1.5)
        ax.axhline(0, color="k", lw=0.5, alpha=0.4)
        ax.set_xlabel("Gaia BP-RP")
        ax.set_ylabel("Residual (mmag)")
        ax.set_title(f"resid_{band} vs BP-RP  b={1000*b:.1f} mmag/mag")
    fig.suptitle(f"D10-1 era-516 NoFilter n={len(df)} BP-RP span={bprp_span:.2f}")
    fig.tight_layout()
    fig.savefig(OUT / "residual_vs_bprp.png", dpi=150)
    plt.close(fig)

    am_fin = airmass[np.isfinite(airmass)]
    summary = {
        "snapshot": str(SNAP.name),
        "n_frames": n_frames,
        "airmass_min": float(np.min(am_fin)) if am_fin.size else float("nan"),
        "airmass_max": float(np.max(am_fin)) if am_fin.size else float("nan"),
        "n_stars": len(df),
        "bprp_span": bprp_span,
        "widen_note": widen_note,
        "mag_column": "median(mag_inst + per_frame_zp) where zp_f = median(G_comp - mag_inst_comp); pre-CT ensemble path (photometry_core.py:4546-4579)",
        "zp_method": "unweighted median of (Gaia G catalog - comp mag_inst) per frame; production uses 1/rms^2 weights when comp_weight_map set",
        "coeff_set": "GDR3_TABLE59_COEFFS",
        "bprp_validity": {"min": BPRP_MIN, "max": BPRP_MAX, "enforced_at": "gaia_johnson.transform_gaia_to_johnson:148-155"},
        "g_validity": {"min": G_MAG_MIN, "max": G_MAG_MAX},
        "a_V": a_v,
        "b_V": b_v,
        "b_V_stderr": b_v_se,
        "b_V_mmag_per_mag": 1000.0 * b_v,
        "a_R": a_r,
        "b_R": b_r,
        "b_R_stderr": b_r_se,
        "b_R_mmag_per_mag": 1000.0 * b_r,
        "abs_b_ratio": ratio,
        "abs_b_diff": abs_diff,
        "combined_b_stderr": comb_se,
        "sign_b_V_minus_b_R": sign_b_diff,
        "expected_sign_b_V_minus_b_R": "negative (red-leaning sensor + V catalog: b_V more negative than b_R)",
        "verdict": verdict,
        "submission_systematics": submission,
        "t1_abs_gh_baseline_mmag": 141.0,
        "t1_abs_gh_candidate_mmag": 332.0,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="ascii")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
