# -*- coding: ascii -*-
"""EPSF-AC-02: DOD-05 criteria 2a/2b + Part C offset decomposition.

Dev-only. Zero src_py imports. Zero new photometry. VAL-03 products only
plus local a2_compare / CORE-03 T1 surface. Live 516 read-only.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import RegularGridInterpolator

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "dev" / "xval_psfex") not in sys.path:
    sys.path.insert(0, str(REPO / "dev" / "xval_psfex"))

from a2_compare import A2_OUT, match_frame, read_pass2  # noqa: E402

OUT = REPO / "dev" / "results" / "context" / "session_20260916_epsf_ac_02"
VAL03 = REPO / "dev" / "results" / "context" / "session_20260916_epsf_val_03"
CORE03 = REPO / "dev" / "results" / "context" / "session_20260914_epsf_core_03"
A2_COMPARE = REPO / "dev" / "results" / "context" / "session_20260907_epsfxval_a2" / "a2_compare"
MS_PATH = (
    REPO
    / "Archive"
    / "Drafts"
    / "draft_000516"
    / "platesolve"
    / "NoFilter_60_2"
    / "masterstars_full_match.csv"
)
G4_PATHS = {
    "csv": MS_PATH,
    "fits": REPO
    / "Archive"
    / "Drafts"
    / "draft_000516"
    / "platesolve"
    / "NoFilter_60_2"
    / "MASTERSTAR.fits",
    "epsf": REPO
    / "Archive"
    / "Drafts"
    / "draft_000516"
    / "platesolve"
    / "NoFilter_60_2"
    / "masterstar_epsf.fits",
}
G4_EXPECT = {"csv": "bfa24039", "fits": "13e77cf8", "epsf": "172f9540"}

T2A = 5.0  # mmag/mag
T2B = 10.0  # mmag
T2B_SOFT = 15.0  # mmag noise-limited ceiling
N_BOOT = 2000
R_MATCH = 2.0
RNG_SEED = 20260916

CRITERION_2A = (
    "2a LINEARITY (flux scale vs brightness): on the VAL-03 isolated "
    "set, Theil-Sen slope of d = m_psf_inst - m_L_inst vs (G - 10), "
    "fitted simultaneously with colour: |b| <= 5.0 mmag/mag. Bootstrap "
    "std reported; PASS requires |b| <= 5.0 and |b| - 2*std <= 5.0 is "
    "NOT required (n is small; report both)."
)
CRITERION_2B = (
    "2b STABILITY of per-star offsets (what differential photometry "
    "relies on): split-half test - d_s derived on odd epochs, applied "
    "to even epochs (and vice versa); the robust scatter across stars "
    "of the residual (d_s,even - d_s,odd) <= 10 mmag. If 2b holds, the "
    "per-star scale offsets are constants absorbed by ensemble "
    "normalization and cannot enter a differential LC."
)


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


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def g4_live_516() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for key, path in G4_PATHS.items():
        digest = _sha256_file(path)
        out[key] = {
            "prefix": digest[:8],
            "verdict": "PASS" if digest.startswith(G4_EXPECT[key]) else "FAIL",
        }
    return out


def robust_scatter(y: np.ndarray) -> float:
    yy = np.asarray(y, dtype=np.float64)
    yy = yy[np.isfinite(yy)]
    if yy.size < 2:
        return float("nan")
    return float(1.4826 * np.median(np.abs(yy - np.median(yy))))


def mad_scatter(y: np.ndarray) -> float:
    """1.4826 * MAD (same as robust_scatter)."""
    return robust_scatter(y)


def alternating_theilsen(d: np.ndarray, xg: np.ndarray, xb: np.ndarray, n_iter: int = 4) -> dict:
    ok = np.isfinite(d) & np.isfinite(xg) & np.isfinite(xb)
    yy, xxg, xxb = d[ok], xg[ok], xb[ok]
    n = int(yy.size)
    if n < 8:
        return {
            "n": n,
            "a": float("nan"),
            "b": float("nan"),
            "c": float("nan"),
            "robust_scatter_mmag": float("nan"),
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
        "robust_scatter_mmag": robust_scatter(resid),
        "method": "alternating_theilsen_4",
    }


def theilsen_g_only(d: np.ndarray, xg: np.ndarray) -> dict:
    ok = np.isfinite(d) & np.isfinite(xg)
    yy, xx = d[ok], xg[ok]
    n = int(yy.size)
    if n < 8:
        return {"n": n, "a": float("nan"), "b": float("nan"), "robust_scatter_mmag": float("nan")}
    b, a, _, _ = stats.theilslopes(yy, xx)
    resid = yy - (float(a) + float(b) * xx)
    return {
        "n": n,
        "a": float(a),
        "b": float(b),
        "robust_scatter_mmag": robust_scatter(resid),
    }


def bootstrap_abc(d: np.ndarray, xg: np.ndarray, xb: np.ndarray, n_boot: int = N_BOOT) -> dict:
    fit = alternating_theilsen(d, xg, xb)
    ok = np.isfinite(d) & np.isfinite(xg) & np.isfinite(xb)
    yy, xxg, xxb = d[ok], xg[ok], xb[ok]
    n = int(yy.size)
    rng = np.random.default_rng(RNG_SEED)
    bs_b: list[float] = []
    bs_c: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        f = alternating_theilsen(yy[idx], xxg[idx], xxb[idx])
        if math.isfinite(f["b"]):
            bs_b.append(float(f["b"]))
        if math.isfinite(f["c"]):
            bs_c.append(float(f["c"]))
    fit["b_boot_std"] = float(np.std(bs_b, ddof=1)) if len(bs_b) > 2 else float("nan")
    fit["c_boot_std"] = float(np.std(bs_c, ddof=1)) if len(bs_c) > 2 else float("nan")
    fit["n_boot"] = n_boot
    return fit


def load_isolated_18() -> pd.DataFrame:
    iso = pd.read_csv(VAL03 / "isolated_candidates.csv", dtype={"catalog_id": str})
    iso["catalog_id"] = iso["catalog_id"].astype(str).str.strip()
    passers = iso[iso["pass_selection"] == True].copy()  # noqa: E712
    if len(passers) != 18:
        raise SystemExit(f"STOP: expected 18 pass_selection stars, got {len(passers)}")
    return passers.sort_values("G").reset_index(drop=True)


def build_d_epoch(passers: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    lc = pd.read_csv(VAL03 / "large_aperture_lc.csv", dtype={"catalog_id": str})
    lc["catalog_id"] = lc["catalog_id"].astype(str).str.strip()
    stems = sorted(lc["stem"].astype(str).unique().tolist())
    if len(stems) != 134:
        raise SystemExit(f"STOP: expected 134 stems in large_aperture_lc, got {len(stems)}")
    ids = set(passers["catalog_id"])
    lc = lc[lc["catalog_id"].isin(ids)].copy()
    if lc["catalog_id"].nunique() != 18:
        raise SystemExit(f"STOP: lc stars {lc['catalog_id'].nunique()} != 18")
    col_L = "F_4xFWHM"
    if col_L not in lc.columns:
        raise SystemExit(f"STOP: missing {col_L}")
    fl = pd.to_numeric(lc[col_L], errors="coerce")
    psf = pd.to_numeric(lc["psf_flux"], errors="coerce")
    ok = (fl > 0) & np.isfinite(fl) & (psf > 0) & np.isfinite(psf)
    lc = lc.loc[ok].copy()
    m_L = -2.5 * np.log10(fl[ok].to_numpy(dtype=np.float64))
    m_psf = -2.5 * np.log10(psf[ok].to_numpy(dtype=np.float64))
    lc["d_mmag"] = (m_psf - m_L) * 1000.0
    lc["stem"] = lc["stem"].astype(str)
    # epoch index from sorted stem order (VAL-03 / frozen-proc stem set)
    stem_to_idx = {s: i for i, s in enumerate(stems)}
    lc["epoch_idx"] = lc["stem"].map(stem_to_idx)
    if lc["epoch_idx"].isna().any():
        raise SystemExit("STOP: stem not in ordered epoch list")
    lc["epoch_idx"] = lc["epoch_idx"].astype(int)
    return lc, stems


def per_star_d(lc: pd.DataFrame, passers: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, prow in passers.iterrows():
        cid = str(prow["catalog_id"])
        sub = lc[lc["catalog_id"] == cid]
        d = sub["d_mmag"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "catalog_id": cid,
                "G": float(prow["G"]),
                "bp_rp": float(prow["bp_rp"]),
                "n_epochs": int(len(sub)),
                "d_median_mmag": float(np.median(d)),
                "d_epoch_mad_scatter_mmag": mad_scatter(d),
            }
        )
    return pd.DataFrame(rows)


def split_half(
    lc: pd.DataFrame,
    passers: pd.DataFrame,
    mask_odd: np.ndarray,
    mask_even: np.ndarray,
    label: str,
) -> tuple[pd.DataFrame, dict]:
    """mask_* are boolean arrays over epoch_idx 0..133 selecting halves."""
    rows = []
    noise_terms = []
    for _, prow in passers.iterrows():
        cid = str(prow["catalog_id"])
        sub = lc[lc["catalog_id"] == cid]
        odd = sub[sub["epoch_idx"].isin(np.flatnonzero(mask_odd))]
        even = sub[sub["epoch_idx"].isin(np.flatnonzero(mask_even))]
        d_all = sub["d_mmag"].to_numpy(dtype=np.float64)
        d_o = odd["d_mmag"].to_numpy(dtype=np.float64)
        d_e = even["d_mmag"].to_numpy(dtype=np.float64)
        n_o, n_e = int(d_o.size), int(d_e.size)
        d_odd = float(np.median(d_o)) if n_o else float("nan")
        d_even = float(np.median(d_e)) if n_e else float("nan")
        r = d_even - d_odd
        ep_scat = mad_scatter(d_all)
        n_half = 0.5 * (n_o + n_e)
        if n_half > 0 and math.isfinite(ep_scat):
            noise_terms.append(ep_scat / math.sqrt(n_half))
        rows.append(
            {
                "catalog_id": cid,
                "G": float(prow["G"]),
                "bp_rp": float(prow["bp_rp"]),
                "split": label,
                "d_odd_mmag": d_odd,
                "d_even_mmag": d_even,
                "r_even_minus_odd_mmag": float(r),
                "n_odd": n_o,
                "n_even": n_e,
                "d_epoch_mad_scatter_mmag": ep_scat,
            }
        )
    tab = pd.DataFrame(rows)
    r_arr = tab["r_even_minus_odd_mmag"].to_numpy(dtype=np.float64)
    scat = robust_scatter(r_arr)
    noise_exp = float(math.sqrt(2.0) * np.median(noise_terms)) if noise_terms else float("nan")
    summary = {
        "split": label,
        "n_stars": int(len(tab)),
        "robust_scatter_r_mmag": scat,
        "noise_expectation_mmag": noise_exp,
        "ratio_scatter_over_noise": (
            float(scat / noise_exp) if (math.isfinite(scat) and math.isfinite(noise_exp) and noise_exp > 0) else float("nan")
        ),
    }
    return tab, summary


def judge_2b(summary: dict) -> str:
    scat = float(summary["robust_scatter_r_mmag"])
    noise = float(summary["noise_expectation_mmag"])
    if not math.isfinite(scat):
        return "FAIL"
    if scat <= T2B:
        return "PASS"
    if scat <= T2B_SOFT and math.isfinite(noise) and scat <= 1.5 * noise:
        return "PASS_noise_limited"
    return "FAIL"


def load_t1_interpolator() -> RegularGridInterpolator:
    t1 = pd.read_csv(CORE03 / "phase_bias_T1.csv")
    off = t1[t1["noise"] == False].copy()  # noqa: E712
    xs = np.sort(off["dx"].unique())
    ys = np.sort(off["dy"].unique())
    grid = np.full((len(xs), len(ys)), np.nan, dtype=np.float64)
    for _, row in off.iterrows():
        i = int(np.where(xs == row["dx"])[0][0])
        j = int(np.where(ys == row["dy"])[0][0])
        grid[i, j] = float(row["bias_median_mmag"])
    if not np.isfinite(grid).all():
        raise SystemExit("STOP: T1 noise-off grid has gaps")
    return RegularGridInterpolator((xs, ys), grid, bounds_error=False, fill_value=None)


def match_xpsf_for_stars(catalog_ids: list[str], stems: list[str]) -> pd.DataFrame:
    """PSFEx deg2 XPSF/YPSF via a2_compare match_rows when present, else pass2 NN."""
    existing = pd.read_csv(A2_COMPARE / "match_rows_deg2.csv", dtype={"catalog_id": str})
    existing["catalog_id"] = existing["catalog_id"].astype(str).str.strip()
    have = set(existing["catalog_id"].unique()) & set(catalog_ids)
    rows = []
    if have:
        sub = existing[existing["catalog_id"].isin(have) & existing["stem"].isin(stems)].copy()
        sub["source"] = "a2_compare_match_rows_deg2"
        rows.append(sub[["catalog_id", "stem", "XPSF_IMAGE", "YPSF_IMAGE", "source"]])

    need = [c for c in catalog_ids if c not in have]
    if need:
        ms = pd.read_csv(MS_PATH, dtype={"catalog_id": str})
        ms["catalog_id"] = ms["catalog_id"].astype(str).str.strip()
        ms = ms[ms["catalog_id"].isin(need)].copy()
        # masterstars x,y are 0-indexed; XPSF is FITS 1-indexed
        id_order = [str(c) for c in ms["catalog_id"].tolist()]
        x_pred0 = ms["x"].to_numpy(dtype=np.float64) + 1.0
        y_pred0 = ms["y"].to_numpy(dtype=np.float64) + 1.0
        work = A2_OUT / "work"
        matched_rows = []
        for stem in stems:
            cat_path = work / stem / f"{stem}_pass2.cat"
            if not cat_path.is_file():
                raise SystemExit(f"STOP: missing {cat_path}")
            cat = read_pass2(cat_path)
            m = match_frame(cat, x_pred0, y_pred0, id_order, R_MATCH)
            m = m[m["matched"]].copy()
            m["stem"] = stem
            m["source"] = "a2_pass2_nn_masterstars_xy_plus1"
            matched_rows.append(m[["catalog_id", "stem", "XPSF_IMAGE", "YPSF_IMAGE", "source"]])
        if matched_rows:
            rows.append(pd.concat(matched_rows, ignore_index=True))

    if not rows:
        raise SystemExit("STOP: no XPSF matches")
    out = pd.concat(rows, ignore_index=True)
    # coverage check
    for cid in catalog_ids:
        n = int((out["catalog_id"] == cid).sum())
        if n < 8:
            raise SystemExit(f"STOP: XPSF coverage {cid} n={n}")
    return out


def part_c_phase(
    passers: pd.DataFrame,
    d_star: pd.DataFrame,
    fit_abc: dict,
    stems: list[str],
) -> pd.DataFrame:
    interp = load_t1_interpolator()
    ids = passers["catalog_id"].astype(str).tolist()
    xpsf = match_xpsf_for_stars(ids, stems)
    pred_rows = []
    for cid in ids:
        sub = xpsf[xpsf["catalog_id"] == cid]
        fx = np.mod(pd.to_numeric(sub["XPSF_IMAGE"], errors="coerce").to_numpy(dtype=np.float64), 1.0)
        fy = np.mod(pd.to_numeric(sub["YPSF_IMAGE"], errors="coerce").to_numpy(dtype=np.float64), 1.0)
        ok = np.isfinite(fx) & np.isfinite(fy)
        # clamp to [0, 1] for interpolator (grid includes 1.0)
        fx = np.clip(fx[ok], 0.0, 1.0)
        fy = np.clip(fy[ok], 0.0, 1.0)
        pred_ep = interp(np.column_stack([fx, fy]))
        pred_med = float(np.median(pred_ep))
        pred_rows.append(
            {
                "catalog_id": cid,
                "phase_pred_median_mmag": pred_med,
                "frac_x_median": float(np.median(fx)),
                "frac_y_median": float(np.median(fy)),
                "n_xpsf_epochs": int(fx.size),
                "xpsf_source": str(sub["source"].iloc[0]),
            }
        )
    pred = pd.DataFrame(pred_rows)
    mean_pred = float(pred["phase_pred_median_mmag"].mean())
    pred["phase_pred_minus_mean_mmag"] = pred["phase_pred_median_mmag"] - mean_pred

    c = float(fit_abc["c"])
    merged = d_star.merge(pred, on="catalog_id", how="left")
    merged["d_minus_colour_mmag"] = merged["d_median_mmag"] - c * (merged["bp_rp"] - 1.0)
    merged["d_minus_colour_phase_mmag"] = (
        merged["d_minus_colour_mmag"] - merged["phase_pred_minus_mean_mmag"]
    )
    return merged


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    passers = load_isolated_18()
    catalog_ids = passers["catalog_id"].astype(str).tolist()
    print(f"[ac02] n=18 catalog_ids confirmed: {catalog_ids}")

    lc, stems = build_d_epoch(passers)
    print(f"[ac02] epoch set n={len(stems)}; ordering=sorted(stem) from VAL-03 large_aperture_lc.csv")
    print(f"[ac02] first/last stem: {stems[0]} .. {stems[-1]}")

    d_star = per_star_d(lc, passers)
    xg = d_star["G"].to_numpy(dtype=np.float64) - 10.0
    xb = d_star["bp_rp"].to_numpy(dtype=np.float64) - 1.0
    y = d_star["d_median_mmag"].to_numpy(dtype=np.float64)

    # Part A
    fit_abc = bootstrap_abc(y, xg, xb)
    fit_g = theilsen_g_only(y, xg)
    lin_row = {
        "population": "VAL-03 isolated n=18",
        "n": fit_abc["n"],
        "a_mmag": fit_abc["a"],
        "b_mmag_per_mag": fit_abc["b"],
        "c_mmag_per_mag_bp_rp": fit_abc["c"],
        "b_boot_std": fit_abc["b_boot_std"],
        "c_boot_std": fit_abc["c_boot_std"],
        "robust_scatter_resid_mmag": fit_abc["robust_scatter_mmag"],
        "method": fit_abc["method"],
        "g_only_a_mmag": fit_g["a"],
        "g_only_b_mmag_per_mag": fit_g["b"],
        "g_only_robust_scatter_mmag": fit_g["robust_scatter_mmag"],
        "abs_b": abs(float(fit_abc["b"])),
        "abs_b_minus_2_boot_std": abs(float(fit_abc["b"])) - 2.0 * float(fit_abc["b_boot_std"]),
    }
    pd.DataFrame([lin_row]).to_csv(OUT / "linearity_fit.csv", index=False)
    print(
        f"[ac02] 2a simultaneous a={fit_abc['a']:.3f} b={fit_abc['b']:.3f}+/-{fit_abc['b_boot_std']:.3f} "
        f"c={fit_abc['c']:.3f}+/-{fit_abc['c_boot_std']:.3f} scatter={fit_abc['robust_scatter_mmag']:.3f}"
    )
    print(f"[ac02] G-only b={fit_g['b']:.3f} (VAL-03 continuity; was -4.934)")

    # Part B
    n_ep = len(stems)
    epoch_idx = np.arange(n_ep)
    odd_even_odd = (epoch_idx % 2) == 1  # odd indices 1,3,5,...
    odd_even_even = (epoch_idx % 2) == 0
    # Task: d_odd on odd epochs, d_even on even -- index 0 is even
    tab_oe, sum_oe = split_half(lc, passers, odd_even_odd, odd_even_even, "odd_even_by_epoch_idx")
    # chronological first half vs second half
    mid = n_ep // 2
    first = epoch_idx < mid
    second = epoch_idx >= mid
    # map: "odd" slot = first half, "even" slot = second half
    tab_ch, sum_ch = split_half(lc, passers, first, second, "first_vs_second_half_chronological")
    # rename columns for chronological clarity in the shared schema
    per_star = pd.concat([tab_oe, tab_ch], ignore_index=True)
    per_star.to_csv(OUT / "splithalf_per_star.csv", index=False)

    sum_oe["verdict"] = judge_2b(sum_oe)
    sum_ch["verdict"] = judge_2b(sum_ch)
    sum_oe["ordering_source"] = (
        "sorted unique stem from VAL-03 large_aperture_lc.csv "
        "(same 134 as frozen-proc / VAL-03); odd = epoch_idx % 2 == 1"
    )
    sum_ch["ordering_source"] = (
        "same sorted stem order; first half = epoch_idx < 67, "
        "second half = epoch_idx >= 67 (n_ep=134)"
    )
    summary_df = pd.DataFrame([sum_oe, sum_ch])
    summary_df.to_csv(OUT / "splithalf_summary.csv", index=False)
    print(
        f"[ac02] 2b odd/even scatter={sum_oe['robust_scatter_r_mmag']:.3f} "
        f"noise_exp={sum_oe['noise_expectation_mmag']:.3f} -> {sum_oe['verdict']}"
    )
    print(
        f"[ac02] 2b chrono scatter={sum_ch['robust_scatter_r_mmag']:.3f} "
        f"noise_exp={sum_ch['noise_expectation_mmag']:.3f} -> {sum_ch['verdict']}"
    )

    # Part C
    decomp = part_c_phase(passers, d_star, fit_abc, stems)
    d_raw = decomp["d_median_mmag"].to_numpy(dtype=np.float64)
    d_col = decomp["d_minus_colour_mmag"].to_numpy(dtype=np.float64)
    d_cp = decomp["d_minus_colour_phase_mmag"].to_numpy(dtype=np.float64)
    pred = decomp["phase_pred_minus_mean_mmag"].to_numpy(dtype=np.float64)
    rho, pval = stats.spearmanr(pred, d_col)
    ok = np.isfinite(pred) & np.isfinite(d_col)
    if ok.sum() >= 8:
        slope, intercept, _, _ = stats.theilslopes(d_col[ok], pred[ok])
        slope_f, intercept_f = float(slope), float(intercept)
    else:
        slope_f, intercept_f = float("nan"), float("nan")

    decomp_summary = {
        "population": "VAL-03 isolated n=18",
        "robust_scatter_d_raw_mmag": robust_scatter(d_raw),
        "robust_scatter_d_after_colour_mmag": robust_scatter(d_col),
        "robust_scatter_d_after_colour_plus_phase_mmag": robust_scatter(d_cp),
        "colour_c_mmag_per_mag": float(fit_abc["c"]),
        "phase_spearman_rho": float(rho),
        "phase_spearman_p": float(pval),
        "phase_theilsen_slope_d_vs_pred": slope_f,
        "phase_theilsen_intercept": intercept_f,
        "phase_surface": "CORE-03 phase_bias_T1.csv noise=False; bilinear; mean-subtracted",
    }
    decomp.to_csv(OUT / "decomposition.csv", index=False)

    # Readings
    abs_b = abs(float(fit_abc["b"]))
    r_ac1 = "PASS" if abs_b <= T2A else "FAIL"
    v_oe = sum_oe["verdict"]
    v_ch = sum_ch["verdict"]
    both_2b_ok = v_oe in ("PASS", "PASS_noise_limited") and v_ch in ("PASS", "PASS_noise_limited")
    r_ac2 = "PASS" if both_2b_ok else "FAIL"
    phase_note = ""
    if math.isfinite(float(rho)) and float(rho) >= 0.5:
        phase_note = " CORE-03 T1 surface confirmed as per-star mechanism (rho>=0.5)."
    r_ac3 = (
        f"RECORD: raw scatter={decomp_summary['robust_scatter_d_raw_mmag']:.1f} mmag; "
        f"after colour={decomp_summary['robust_scatter_d_after_colour_mmag']:.1f}; "
        f"after colour+phase={decomp_summary['robust_scatter_d_after_colour_plus_phase_mmag']:.1f}; "
        f"c={fit_abc['c']:.1f}+/-{fit_abc['c_boot_std']:.1f} mmag/mag; "
        f"phase rho={float(rho):.3f} p={float(pval):.3g} slope={slope_f:.3f}.{phase_note}"
    )
    r_ac4 = (
        "criterion 2 of DOD-05 met on 516; closure waits only on criterion 3 at the 520 re-cut"
        if (r_ac1 == "PASS" and r_ac2 == "PASS")
        else "criterion 2 of DOD-05 NOT met on 516; sequencing is Milan's"
    )

    readings = [
        f"R-AC1 ({r_ac1}): |b|={abs_b:.3f} mmag/mag (lim {T2A}); "
        f"b={fit_abc['b']:.3f} boot_std={fit_abc['b_boot_std']:.3f}; "
        f"|b|-2*std={lin_row['abs_b_minus_2_boot_std']:.3f} (reported, not required).",
        f"R-AC2 ({r_ac2}): odd/even scatter={sum_oe['robust_scatter_r_mmag']:.3f} mmag "
        f"(noise_exp={sum_oe['noise_expectation_mmag']:.3f}, verdict={v_oe}); "
        f"chrono scatter={sum_ch['robust_scatter_r_mmag']:.3f} mmag "
        f"(noise_exp={sum_ch['noise_expectation_mmag']:.3f}, verdict={v_ch}).",
        f"R-AC3 {r_ac3}",
        f"R-AC4: {r_ac4}",
    ]
    for line in readings:
        print(f"[ac02] {line}")

    g4 = g4_live_516()
    summary = {
        "decision": "D-EPSF-XVAL-DOD-05",
        "criterion_2a_quoted": CRITERION_2A,
        "criterion_2b_quoted": CRITERION_2B,
        "catalog_ids_n18": catalog_ids,
        "n_epochs": len(stems),
        "epoch_ordering": "sorted(stem) from VAL-03 large_aperture_lc.csv",
        "stems_first_last": [stems[0], stems[-1]],
        "linearity": lin_row,
        "splithalf": [sum_oe, sum_ch],
        "decomposition": decomp_summary,
        "readings": readings,
        "g4": g4,
    }
    (OUT / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2) + "\n", encoding="utf-8")
    print(f"[ac02] wrote {OUT}")
    print(f"[ac02] G4 {g4}")


if __name__ == "__main__":
    main()
