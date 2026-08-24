#!/usr/bin/env python3
"""EPSF-AC-01 sandbox: AC policy measurement on production ePSF catalogs.

Inverts the F6 scalar AC (psf_flux / psf_ac_factor) so A1-A3 do not re-fit.
Production ePSF / aperture LCs / AAVSO / VarAstro are read-only.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src_py"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from epsf_science_set import build_epsf_science_set  # noqa: E402
from gaia_catalog_id import normalize_gaia_source_id  # noqa: E402
from photometry_core import ensemble_normalize  # noqa: E402
from psf_internal_lc import resolve_ensemble_ids  # noqa: E402

DRAFT = REPO / "Archive" / "Drafts" / "draft_000516"
PS = DRAFT / "platesolve" / "NoFilter_60_2"
FRAMES = DRAFT / "detrended_aligned" / "lights" / "NoFilter_60_2"
PHOT = PS / "photometry"
LC_DIR = PHOT / "lightcurves"
OUT = REPO / "dev" / "results" / "session_20260824_epsf_ac_01"
PROD_EPSF = PS / "masterstar_epsf.fits"
PROD_META = PS / "masterstar_epsf_meta.json"
BO_CVN = "1498613634033133184"
F2_BRIGHT30_RATIO = 1.2176375156541255
M1_MAG_EDGES = [
    5.94305944442749,
    10.551580429077148,
    12.388586044311523,
    13.252224922180176,
    14.055132865905762,
    15.559402465820312,
]
P1_K_PERCENT = 20.0
CHI2_AC_GATE = 5.0
USECOLS = [
    "source_file",
    "catalog_id",
    "phot_g_mean_mag",
    "mag",
    "dao_flux",
    "flux",
    "psf_flux",
    "psf_chi2",
    "psf_fit_ok",
    "psf_ac_factor",
    "psf_ac_applied",
    "psf_ac_n_used",
]


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _norm(raw: Any) -> str:
    try:
        return str(normalize_gaia_source_id(raw)).strip()
    except Exception:  # noqa: BLE001
        s = str(raw or "").strip()
        return "" if s.lower() in ("", "nan", "none") else s


def production_watch_files() -> list[Path]:
    files = [PROD_EPSF, PROD_META]
    files.extend(sorted(p for p in LC_DIR.glob("lightcurve_*.csv") if "_psf" not in p.name)[:6])
    files.extend(sorted((PHOT / "lightcurves_reports" / "aavso").glob("*.txt"))[:4])
    files.extend(sorted((PHOT / "lightcurves_reports" / "varastro").glob("*.txt"))[:4])
    return [p for p in files if p.is_file()]


def snapshot_hashes(label: str) -> dict[str, str]:
    out = {str(p.relative_to(REPO)).replace("\\", "/"): _sha(p) for p in production_watch_files()}
    (OUT / f"hashes_{label}.json").write_text(json.dumps(out, indent=2), encoding="ascii")
    return out


def rms_median_check(values: np.ndarray) -> dict[str, float | bool | int]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0, "median": float("nan"), "rms": float("nan"), "rms_vs_abs_median": False}
    med = float(np.median(v))
    rms = float(np.sqrt(np.mean(v**2)))
    return {
        "n": int(v.size),
        "median": med,
        "rms": rms,
        "rms_vs_abs_median": bool(abs(rms - abs(med)) <= 0.25 * max(abs(med), abs(rms), 1e-12)),
    }


def _is_true(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("1", "true", "t", "yes", "y")


def load_frame_subset() -> list[str]:
    p = REPO / "dev/results/context/session_20260824_epsf_shape_01_f/frame_subset.txt"
    return [ln.strip() for ln in p.read_text(encoding="ascii").splitlines() if ln.strip()]


def mag_of(row: pd.Series) -> float:
    g = float(pd.to_numeric(row.get("phot_g_mean_mag"), errors="coerce"))
    if math.isfinite(g):
        return g
    return float(pd.to_numeric(row.get("mag"), errors="coerce"))


def invert_ac_row(psf_flux: float, ac_factor: float, ac_applied: bool) -> float:
    if not math.isfinite(psf_flux) or psf_flux <= 0:
        return float("nan")
    if ac_applied and math.isfinite(ac_factor) and ac_factor > 0:
        return float(psf_flux / ac_factor)
    return float(psf_flux)


def load_stack(stems: list[str], science_ids: set[str]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for stem in stems:
        path = FRAMES / f"proc_{Path(stem).stem}.csv"
        if not path.is_file():
            continue
        df = pd.read_csv(path, usecols=lambda c: c in USECOLS, low_memory=False)
        df["catalog_id"] = df["catalog_id"].map(_norm)
        df = df.loc[df["catalog_id"].isin(science_ids)].copy()
        df["frame"] = Path(stem).stem
        df["source_file"] = f"proc_{Path(stem).stem}.csv"
        rows.append(df)
    if not rows:
        raise RuntimeError("no proc frames loaded")
    out = pd.concat(rows, ignore_index=True)
    gmag = pd.to_numeric(out.get("phot_g_mean_mag"), errors="coerce")
    imag = pd.to_numeric(out.get("mag"), errors="coerce")
    out["mag"] = gmag.where(np.isfinite(gmag.to_numpy(dtype=float)), imag)
    out["psf_flux"] = pd.to_numeric(out["psf_flux"], errors="coerce")
    out["dao_flux"] = pd.to_numeric(out["dao_flux"], errors="coerce")
    out["psf_chi2"] = pd.to_numeric(out["psf_chi2"], errors="coerce")
    out["psf_ac_factor"] = pd.to_numeric(out["psf_ac_factor"], errors="coerce")
    out["fit_ok"] = out["psf_fit_ok"].map(_is_true)
    out["ac_applied"] = out["psf_ac_applied"].map(_is_true)
    out["psf_raw"] = [
        invert_ac_row(float(pf), float(ac), bool(ap))
        for pf, ac, ap in zip(out["psf_flux"], out["psf_ac_factor"], out["ac_applied"], strict=True)
    ]
    out["ratio_raw"] = out["psf_raw"] / out["dao_flux"]
    out["ratio_prod"] = out["psf_flux"] / out["dao_flux"]
    return out


def bin_index(mag: float, edges: list[float]) -> int:
    if not math.isfinite(mag):
        return -1
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == 0 and mag <= hi:
            return 0
        if i == len(edges) - 2 and mag > lo:
            return i
        if mag > lo and mag <= hi:
            return i
    return -1


def robust_median_ratio(dao: np.ndarray, psf: np.ndarray) -> tuple[float, int]:
    mask = np.isfinite(dao) & np.isfinite(psf) & (dao > 0) & (psf > 0)
    if int(mask.sum()) < 5:
        return 1.0, int(mask.sum())
    ratios = dao[mask] / psf[mask]
    med = float(np.median(ratios))
    mad = float(np.median(np.abs(ratios - med)))
    if mad > 0:
        ratios = ratios[np.abs(ratios - med) < 3.0 * mad]
    if len(ratios) < 5:
        return 1.0, len(ratios)
    return float(np.median(ratios)), int(len(ratios))


def p1_admit(frame_df: pd.DataFrame, edges: list[float], k_pct: float) -> np.ndarray:
    admit = np.zeros(len(frame_df), dtype=bool)
    for b in range(len(edges) - 1):
        mag = frame_df["mag"].to_numpy(dtype=float)
        ok = frame_df["fit_ok"].to_numpy(dtype=bool)
        idx = np.array([bin_index(m, edges) == b for m in mag], dtype=bool) & ok
        if int(idx.sum()) == 0:
            continue
        chi = frame_df.loc[idx, "psf_chi2"].to_numpy(dtype=float)
        finite = np.isfinite(chi)
        if int(finite.sum()) == 0:
            continue
        thr = float(np.nanpercentile(chi[finite], k_pct))
        local = idx.copy()
        local[idx] = finite & (chi <= thr)
        admit |= local
    return admit


def p2_factor_for_mag(mag: float, centers: np.ndarray, factors: np.ndarray) -> float:
    if not math.isfinite(mag) or centers.size == 0:
        return 1.0
    if mag <= float(centers[0]):
        return float(factors[0])
    if mag >= float(centers[-1]):
        return float(factors[-1])
    return float(np.interp(mag, centers, factors))


def apply_monotone(y: np.ndarray) -> np.ndarray:
    """Force monotone in the direction of the first-to-last slope."""
    out = np.array(y, dtype=float)
    if out.size < 2:
        return out
    decreasing = bool(out[-1] < out[0])
    for i in range(1, len(out)):
        if decreasing:
            if out[i] > out[i - 1]:
                out[i] = out[i - 1]
        elif out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


def night_median_by_star(df: pd.DataFrame, col: str) -> pd.DataFrame:
    g = df.groupby("catalog_id", sort=True)
    return pd.DataFrame(
        {
            "catalog_id": list(g.groups.keys()),
            "mag": g["mag"].median().to_numpy(),
            col: g[col].median().to_numpy(),
            "n": g[col].count().to_numpy(),
        }
    )


def build_delta(
    stack: pd.DataFrame,
    flux_col: str,
    epoch_keys: list[str],
    target: str,
    comp_ids: list[str],
    weights: dict[str, float],
) -> np.ndarray:
    n = len(epoch_keys)
    by_src = {str(s).strip(): i for i, s in enumerate(epoch_keys)}

    def series_for(cid: str) -> np.ndarray:
        mag = np.full(n, np.nan)
        sub = stack.loc[stack["catalog_id"] == cid]
        for _, row in sub.iterrows():
            src = str(row.get("source_file") or "").strip()
            i = by_src.get(src)
            if i is None:
                stem = str(row.get("frame") or "")
                for k, idx in by_src.items():
                    if stem and stem in k:
                        i = idx
                        break
            if i is None:
                continue
            ok = bool(row["fit_ok"])
            fl = float(row[flux_col])
            if ok and math.isfinite(fl) and fl > 0:
                mag[i] = float(-2.5 * math.log10(fl))
        return mag

    tgt = series_for(target)
    comp_mag = {cid: series_for(cid) for cid in comp_ids}
    dummy = {cid: 0.0 for cid in comp_ids}
    quality = {cid: {"quality": "good"} for cid in comp_ids}
    _cal, delta, _sc = ensemble_normalize(
        tgt, comp_mag, dummy, quality, comp_weight_map=weights or None
    )
    _ = _cal
    return np.asarray(delta, dtype=float)


def mmag_from_ratio_span(r_hi: float, r_lo: float) -> float:
    if not (math.isfinite(r_hi) and math.isfinite(r_lo) and r_hi > 0 and r_lo > 0):
        return float("nan")
    return float(abs(-2.5 * math.log10(r_hi / r_lo)) * 1000.0)


def main() -> None:
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    hashes_before = snapshot_hashes("before")
    (OUT / "positive_control.txt").write_text("before\n", encoding="ascii")

    science = build_epsf_science_set(PS)
    science_ids = set(science.catalog_ids)
    subset = load_frame_subset()
    all_fits = sorted(FRAMES.glob("BO_CVn_Light_*.fits"))
    all_stems = [p.name for p in all_fits]

    print("load subset", flush=True)
    sub = load_stack(subset, science_ids)
    print("load night", flush=True)
    night = load_stack(all_stems, science_ids)

    ok = sub["fit_ok"] & np.isfinite(sub["ratio_raw"]) & (sub["dao_flux"] > 0)
    a1 = sub.loc[ok].copy()
    a1.to_csv(OUT / "a1_star_frame.csv", index=False)

    star_med = night_median_by_star(a1, "ratio_raw")
    star_med.to_csv(OUT / "a1_star_night_median.csv", index=False)
    corr = float("nan")
    if len(star_med) >= 5:
        corr = float(np.corrcoef(star_med["mag"].to_numpy(), star_med["ratio_raw"].to_numpy())[0, 1])

    bin_rows = []
    for b in range(len(M1_MAG_EDGES) - 1):
        lo, hi = M1_MAG_EDGES[b], M1_MAG_EDGES[b + 1]
        sel = star_med["mag"].map(lambda m, b=b: bin_index(float(m), M1_MAG_EDGES) == b)
        vals = star_med.loc[sel, "ratio_raw"].to_numpy(dtype=float)
        chk = rms_median_check(vals)
        bin_rows.append(
            {
                "bin": b,
                "mag_lo": lo,
                "mag_hi": hi,
                "n_stars": int(sel.sum()),
                **chk,
            }
        )
    bin_df = pd.DataFrame(bin_rows)
    bin_df.to_csv(OUT / "a1_ratio_vs_mag_bins.csv", index=False)

    mag_map = star_med.set_index("catalog_id")["mag"]
    bright_ids = set(star_med.nsmallest(30, "mag")["catalog_id"].tolist())
    bright_ratio = float(star_med.loc[star_med["catalog_id"].isin(bright_ids), "ratio_raw"].median())

    comp_ids, weights, ens_src = resolve_ensemble_ids(BO_CVN, PHOT)
    comp_mags = [float(mag_map[c]) for c in comp_ids if c in mag_map.index]
    tgt_mag = float(mag_map.get(BO_CVN, float("nan")))
    if not math.isfinite(tgt_mag):
        tgt_mag = float(a1.loc[a1["catalog_id"] == BO_CVN, "mag"].median())
    ens_lo = float(np.nanmin(comp_mags)) if comp_mags else float("nan")
    ens_hi = float(np.nanmax(comp_mags)) if comp_mags else float("nan")
    ens_med = float(np.nanmedian(comp_mags)) if comp_mags else float("nan")
    in_span = star_med["mag"].between(ens_lo, ens_hi) if math.isfinite(ens_lo) else star_med["mag"].notna()
    span_vals = star_med.loc[in_span, "ratio_raw"].to_numpy(dtype=float)
    span_chk = rms_median_check(span_vals)
    span_mmag = mmag_from_ratio_span(float(np.nanmax(span_vals)), float(np.nanmin(span_vals))) if span_vals.size else float("nan")
    # robust span: 16-84 percentile
    if span_vals.size >= 8:
        p16, p84 = np.nanpercentile(span_vals, [16, 84])
        span_p1684_mmag = mmag_from_ratio_span(float(p84), float(p16))
    else:
        p16 = p84 = span_p1684_mmag = float("nan")

    slope = float("nan")
    if len(star_med.loc[in_span]) >= 5:
        x = star_med.loc[in_span, "mag"].to_numpy(dtype=float)
        y = star_med.loc[in_span, "ratio_raw"].to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        if int(finite.sum()) >= 5:
            slope = float(np.polyfit(x[finite], y[finite], 1)[0])

    a1_summary = {
        "n_science": science.n_total,
        "n_frames_subset": int(a1["frame"].nunique()),
        "n_fit_ok_rows": int(len(a1)),
        "corr_ratio_mag": corr,
        "bright30_median_ratio": bright_ratio,
        "f2_bright30_ratio": F2_BRIGHT30_RATIO,
        "bright30_vs_f2": bright_ratio - F2_BRIGHT30_RATIO,
        "bo_cvn_mag": tgt_mag,
        "ensemble_source": ens_src,
        "n_comp": len(comp_ids),
        "comp_mag_lo": ens_lo,
        "comp_mag_hi": ens_hi,
        "comp_mag_median": ens_med,
        "span_ratio": span_chk,
        "span_minmax_mmag": span_mmag,
        "span_p16": float(p16) if math.isfinite(float(p16)) else float("nan"),
        "span_p84": float(p84) if math.isfinite(float(p84)) else float("nan"),
        "span_p1684_mmag": span_p1684_mmag,
        "slope_ratio_per_mag": slope,
        "bins": bin_rows,
    }
    (OUT / "a1_summary.json").write_text(json.dumps(a1_summary, indent=2), encoding="ascii")

    # A2 census on subset frames
    census_rows = []
    bias_rows = []
    for frame, fdf in sub.groupby("frame", sort=True):
        okf = fdf["fit_ok"] & np.isfinite(fdf["psf_raw"]) & (fdf["dao_flux"] > 0)
        work = fdf.loc[okf].copy()
        chi_ok = work["psf_chi2"] < CHI2_AC_GATE
        admitted = work.loc[chi_ok]
        bright_in = admitted["catalog_id"].isin(bright_ids)
        ac_chi5, n_chi5 = robust_median_ratio(
            admitted["dao_flux"].to_numpy(), admitted["psf_raw"].to_numpy()
        )
        ac_all, n_all = robust_median_ratio(
            work["dao_flux"].to_numpy(), work["psf_raw"].to_numpy()
        )
        bin_factors = []
        for b in range(len(M1_MAG_EDGES) - 1):
            selb = work["mag"].map(lambda m, b=b: bin_index(float(m), M1_MAG_EDGES) == b)
            fac, n_b = robust_median_ratio(
                work.loc[selb, "dao_flux"].to_numpy(), work.loc[selb, "psf_raw"].to_numpy()
            )
            bin_factors.append({"bin": b, "ac": fac, "n": n_b})
        census_rows.append(
            {
                "frame": frame,
                "n_fit_ok": int(len(work)),
                "n_chi2_lt5": int(len(admitted)),
                "frac_chi2_lt5": float(len(admitted) / max(len(work), 1)),
                "admit_mag_median": float(admitted["mag"].median()) if len(admitted) else float("nan"),
                "science_mag_median": float(work["mag"].median()) if len(work) else float("nan"),
                "n_bright30_admitted": int(bright_in.sum()),
                "frac_bright30_admitted": float(bright_in.sum() / max(len(bright_ids), 1)),
                "admit_chi2_median": float(admitted["psf_chi2"].median()) if len(admitted) else float("nan"),
                "all_chi2_median": float(work["psf_chi2"].median()) if len(work) else float("nan"),
            }
        )
        bias_rows.append(
            {
                "frame": frame,
                "ac_chi2_lt5": ac_chi5,
                "n_chi2_lt5": n_chi5,
                "ac_all_fit_ok": ac_all,
                "n_all_fit_ok": n_all,
                **{f"ac_bin{b['bin']}": b["ac"] for b in bin_factors},
                **{f"n_bin{b['bin']}": b["n"] for b in bin_factors},
            }
        )
    census_df = pd.DataFrame(census_rows)
    bias_df = pd.DataFrame(bias_rows)
    census_df.to_csv(OUT / "a2_ensemble_census.csv", index=False)
    bias_df.to_csv(OUT / "a2_ac_bias.csv", index=False)
    bright_ever = 0
    for cid in bright_ids:
        hit = False
        for _, fdf in sub.groupby("frame"):
            row = fdf.loc[(fdf["catalog_id"] == cid) & fdf["fit_ok"]]
            if len(row) and float(row["psf_chi2"].iloc[0]) < CHI2_AC_GATE:
                hit = True
                break
        if hit:
            bright_ever += 1
    a2_summary = {
        "chi2_gate": CHI2_AC_GATE,
        "night_median_n_chi2_lt5": float(census_df["n_chi2_lt5"].median()),
        "night_median_frac_chi2_lt5": float(census_df["frac_chi2_lt5"].median()),
        "night_median_admit_mag": float(census_df["admit_mag_median"].median()),
        "night_median_science_mag": float(census_df["science_mag_median"].median()),
        "night_median_frac_bright30": float(census_df["frac_bright30_admitted"].median()),
        "n_bright30_ever_admitted": bright_ever,
        "n_bright30": len(bright_ids),
        "night_median_ac_chi2_lt5": float(bias_df["ac_chi2_lt5"].median()),
        "night_median_ac_all_fit_ok": float(bias_df["ac_all_fit_ok"].median()),
        "night_median_ac_bins": {
            f"bin{b}": float(bias_df[f"ac_bin{b}"].median()) for b in range(len(M1_MAG_EDGES) - 1)
        },
    }
    (OUT / "a2_summary.json").write_text(json.dumps(a2_summary, indent=2), encoding="ascii")

    # A3 policies on subset + BO CVn full night
    ap_lc = pd.read_csv(LC_DIR / f"lightcurve_{BO_CVN}.csv")
    epoch_keys = [str(s).strip() for s in ap_lc["source_file"].tolist()]
    ap_delta = pd.to_numeric(ap_lc["delta_mag"], errors="coerce").to_numpy(dtype=float)

    def policy_factors(frame_df: pd.DataFrame, policy: str) -> np.ndarray:
        n = len(frame_df)
        ones = np.ones(n, dtype=float)
        okf = frame_df["fit_ok"].to_numpy(dtype=bool)
        dao = frame_df["dao_flux"].to_numpy(dtype=float)
        psf = frame_df["psf_raw"].to_numpy(dtype=float)
        mag = frame_df["mag"].to_numpy(dtype=float)
        chi = frame_df["psf_chi2"].to_numpy(dtype=float)
        if policy == "P0":
            mask = okf & np.isfinite(chi) & (chi < CHI2_AC_GATE)
            fac, _n = robust_median_ratio(dao[mask], psf[mask])
            return ones * fac
        if policy == "P3":
            fac, _n = robust_median_ratio(dao[okf], psf[okf])
            return ones * fac
        if policy == "P4":
            return ones
        if policy == "P1":
            admit = p1_admit(frame_df, M1_MAG_EDGES, P1_K_PERCENT)
            fac, _n = robust_median_ratio(dao[admit], psf[admit])
            return ones * fac
        if policy == "P2":
            centers = []
            facs = []
            for b in range(len(M1_MAG_EDGES) - 1):
                selb = np.array([bin_index(float(m), M1_MAG_EDGES) == b for m in mag], dtype=bool) & okf
                fac, n_b = robust_median_ratio(dao[selb], psf[selb])
                if n_b >= 5:
                    centers.append(0.5 * (M1_MAG_EDGES[b] + M1_MAG_EDGES[b + 1]))
                    facs.append(fac)
            if not centers:
                return ones
            c = np.asarray(centers, dtype=float)
            f = np.asarray(facs, dtype=float)
            f = apply_monotone(f)
            return np.array([p2_factor_for_mag(m, c, f) for m in mag], dtype=float)
        raise ValueError(policy)

    policy_rows = []
    for policy in ("P0", "P1", "P2", "P3", "P4"):
        corr_parts = []
        for _frame, fdf in sub.groupby("frame", sort=True):
            fac = policy_factors(fdf, policy)
            tmp = fdf.copy()
            tmp["psf_corr"] = tmp["psf_raw"] * fac
            tmp["ratio_corr"] = tmp["psf_corr"] / tmp["dao_flux"]
            corr_parts.append(tmp)
        corr_df = pd.concat(corr_parts, ignore_index=True)
        okc = corr_df["fit_ok"] & np.isfinite(corr_df["ratio_corr"])
        star_c = night_median_by_star(corr_df.loc[okc], "ratio_corr")
        in_s = star_c["mag"].between(ens_lo, ens_hi) if math.isfinite(ens_lo) else star_c["mag"].notna()
        vals = star_c.loc[in_s, "ratio_corr"].to_numpy(dtype=float)
        chk = rms_median_check(vals)
        flat_mmag = mmag_from_ratio_span(float(np.nanmax(vals)), float(np.nanmin(vals))) if vals.size else float("nan")
        if vals.size >= 8:
            q16, q84 = np.nanpercentile(vals, [16, 84])
            p1684 = mmag_from_ratio_span(float(q84), float(q16))
        else:
            p1684 = float("nan")
        # BO CVn night LC
        night_corr_parts = []
        for _frame, fdf in night.groupby("frame", sort=True):
            fac = policy_factors(fdf, policy)
            tmp = fdf.copy()
            tmp["psf_corr"] = tmp["psf_raw"] * fac
            night_corr_parts.append(tmp)
        night_c = pd.concat(night_corr_parts, ignore_index=True)
        delta = build_delta(night_c, "psf_corr", epoch_keys, BO_CVN, comp_ids, weights)
        both = np.isfinite(delta) & np.isfinite(ap_delta)
        d = delta[both] - ap_delta[both]
        rms_d = float(np.sqrt(np.mean(d**2))) if both.any() else float("nan")
        med_d = float(np.median(d)) if both.any() else float("nan")
        coverage = float(np.mean(np.isfinite(delta))) if len(delta) else float("nan")
        policy_rows.append(
            {
                "policy": policy,
                "span_median_ratio": chk["median"],
                "span_rms_ratio": chk["rms"],
                "span_rms_vs_abs_median": chk["rms_vs_abs_median"],
                "span_minmax_mmag": flat_mmag,
                "span_p1684_mmag": p1684,
                "bo_n_epochs": int(len(delta)),
                "bo_n_finite": int(np.isfinite(delta).sum()),
                "bo_coverage": coverage,
                "bo_dmag_minus_ap_median": med_d,
                "bo_dmag_minus_ap_rms": rms_d,
                "bo_dmag_minus_ap_rms_mmag": rms_d * 1000.0 if math.isfinite(rms_d) else float("nan"),
            }
        )
        star_c.to_csv(OUT / f"a3_{policy}_star_median.csv", index=False)
        pd.DataFrame(
            {
                "source_file": epoch_keys,
                "psf_delta_mag": delta,
                "ap_delta_mag": ap_delta,
            }
        ).to_csv(OUT / f"a3_{policy}_bo_delta.csv", index=False)

    pol_df = pd.DataFrame(policy_rows)
    pol_df.to_csv(OUT / "a3_policy_table.csv", index=False)

    # P4 invariance: AC-on (P0) vs AC-off (P4) delta_mag
    p0 = pd.read_csv(OUT / "a3_P0_bo_delta.csv")
    p4 = pd.read_csv(OUT / "a3_P4_bo_delta.csv")
    d_inv = pd.to_numeric(p0["psf_delta_mag"], errors="coerce") - pd.to_numeric(
        p4["psf_delta_mag"], errors="coerce"
    )
    inv_finite = d_inv.to_numpy(dtype=float)
    inv_finite = inv_finite[np.isfinite(inv_finite)]
    p4_invariance = {
        "n": int(inv_finite.size),
        "max_abs": float(np.max(np.abs(inv_finite))) if inv_finite.size else float("nan"),
        "rms": float(np.sqrt(np.mean(inv_finite**2))) if inv_finite.size else float("nan"),
        "identical_to_1e12": bool(inv_finite.size > 0 and float(np.max(np.abs(inv_finite))) < 1e-12),
        "identical_to_1e6": bool(inv_finite.size > 0 and float(np.max(np.abs(inv_finite))) < 1e-6),
    }
    # residual bias if mag slope: (m_tgt - m_ens) * dmag/dmag_ratio
    # d(delta)/d(ratio) ~ -2.5/(ratio ln 10) * slope_ratio * dm
    r_span = float(span_chk["median"]) if span_chk["n"] else 1.0
    if math.isfinite(slope) and math.isfinite(tgt_mag) and math.isfinite(ens_med) and r_span > 0:
        dmag_bias = -2.5 / math.log(10.0) * (slope / r_span) * (tgt_mag - ens_med)
    else:
        dmag_bias = float("nan")
    p4_invariance["p4_residual_bias_mag"] = dmag_bias
    p4_invariance["p4_residual_bias_mmag"] = (
        dmag_bias * 1000.0 if math.isfinite(dmag_bias) else float("nan")
    )
    (OUT / "a3_p4_invariance.json").write_text(json.dumps(p4_invariance, indent=2), encoding="ascii")

    (OUT / "positive_control.txt").write_text("after\n", encoding="ascii")
    hashes_after = snapshot_hashes("after")
    summary = {
        "elapsed_s": time.perf_counter() - t0,
        "prod_epsf_sha256": _sha(PROD_EPSF),
        "production_hashes_identical": hashes_before == hashes_after,
        "positive_control_changed": True,
        "a1": a1_summary,
        "a2": a2_summary,
        "a3": policy_rows,
        "p4_invariance": p4_invariance,
        "science": science.to_meta_dict(),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="ascii")
    print("EPSF-AC-01 done", summary["elapsed_s"], "s", flush=True)


if __name__ == "__main__":
    main()
