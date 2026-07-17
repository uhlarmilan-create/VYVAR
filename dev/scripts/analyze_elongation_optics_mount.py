#!/usr/bin/env python3
"""Part 1: optics vs mount vs bias analysis from existing Gaussian-fit CSVs."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401  (repo layout: src_py + dev on sys.path)
_ROOT = _bootstrap.REPO_ROOT
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import AppConfig  # noqa: E402
from gaia_catalog_id import normalize_gaia_source_id  # noqa: E402

MAD_SCALE = 1.4826
SNR_MIN = 50.0


def _norm_cid(x) -> str:
    return normalize_gaia_source_id(x) or ""


def _compute_snr(flux, sky, rap, gain=1.0, rn=1.6) -> float:
    if not (math.isfinite(flux) and flux > 0 and math.isfinite(sky) and math.isfinite(rap) and rap > 0):
        return 0.0
    g = gain if gain > 0 else 1.0
    area = math.pi * rap * rap
    var = flux / g + max(0.0, sky) / g * area + (rn / g) ** 2 * area
    return float(flux / math.sqrt(var)) if var > 0 else 0.0


def _fold_pa_delta(pa_deg: np.ndarray, az_deg: np.ndarray) -> np.ndarray:
    d = (pa_deg - az_deg) % 180.0
    d = np.where(d > 90.0, 180.0 - d, d)
    return np.abs(d)


def _load_enriched(draft_id: int, csv_path: Path, aligned_dir: Path) -> tuple[pd.DataFrame, tuple[int, int]]:
    df = pd.read_csv(csv_path, low_memory=False)
    ms = Path(aligned_dir.parents[2] / "platesolve" / aligned_dir.name / "MASTERSTAR.fits")
    with fits.open(ms, memmap=True) as hd:
        w = int(hd[0].header.get("NAXIS1", hd[0].data.shape[1]))
        h = int(hd[0].header.get("NAXIS2", hd[0].data.shape[0]))
    chip = (w, h)

    proc_by_frame = {}
    for p in sorted(aligned_dir.glob("proc_*.csv")):
        proc_by_frame[p.name.replace(".csv", ".fits").replace("proc_", "proc_") if False else p.stem] = p
    # map frame column (proc_xxx.fits) -> csv stem
    stem_map = {f"proc_{p.stem.replace('proc_', '')}" if p.stem.startswith("proc_") else p.stem: p for p in aligned_dir.glob("proc_*.csv")}
    # frame in CSV is like proc_V842_Her_Light_001.fits -> stem proc_V842_Her_Light_001
    rows = []
    for _, r in df.iterrows():
        frame = str(r["frame"])
        stem = Path(frame).stem
        csv_p = aligned_dir / f"{stem}.csv"
        if not csv_p.is_file():
            continue
        rows.append((frame, _norm_cid(r["catalog_id"]), r))
    if not rows:
        return df, chip

    # load proc lookup per frame lazily
    cache: dict[str, pd.DataFrame] = {}
    out = []
    for frame, cid, r in rows:
        stem = Path(frame).stem
        if stem not in cache:
            cp = aligned_dir / f"{stem}.csv"
            if not cp.is_file():
                continue
            sub = pd.read_csv(
                cp,
                usecols=lambda c: c in (
                    "catalog_id", "x", "y", "dao_flux", "noise_floor_adu", "aperture_r_px",
                    "mag", "catalog_mag", "phot_g_mean_mag", "snr50_ok",
                ),
                low_memory=False,
                dtype={"catalog_id": str},
            )
            sub["_cid"] = sub["catalog_id"].map(_norm_cid)
            cache[stem] = sub
        sub = cache[stem]
        hit = sub[sub["_cid"] == cid]
        if hit.empty:
            continue
        h0 = hit.iloc[0]
        x = float(pd.to_numeric(h0.get("x"), errors="coerce"))
        y = float(pd.to_numeric(h0.get("y"), errors="coerce"))
        flux = float(pd.to_numeric(h0.get("dao_flux"), errors="coerce"))
        sky = float(pd.to_numeric(h0.get("noise_floor_adu"), errors="coerce"))
        rap = float(pd.to_numeric(h0.get("aperture_r_px"), errors="coerce"))
        snr = _compute_snr(flux, sky, rap)
        mag = pd.to_numeric(h0.get("phot_g_mean_mag"), errors="coerce")
        if not math.isfinite(float(mag)):
            mag = pd.to_numeric(h0.get("catalog_mag"), errors="coerce")
        if not math.isfinite(float(mag)):
            mag = pd.to_numeric(h0.get("mag"), errors="coerce")
        rec = dict(r)
        rec["x"] = x
        rec["y"] = y
        rec["snr"] = snr
        rec["mag"] = float(mag) if math.isfinite(float(mag)) else float("nan")
        rec["amplitude_proxy"] = float(r.get("fwhm_major_px", float("nan")))
        out.append(rec)
    return pd.DataFrame(out), chip


def _radial_bins(df: pd.DataFrame, chip: tuple[int, int], n_bins: int = 5) -> pd.DataFrame:
    w, h = chip
    xc, yc = w / 2.0, h / 2.0
    r = np.hypot(df["x"] - xc, df["y"] - yc)
    rmax = np.hypot(xc, yc)
    bins = np.linspace(0, rmax, n_bins + 1)
    labels = [f"r{ i+1}" for i in range(n_bins)]
    df = df.copy()
    df["r_bin"] = pd.cut(r, bins=bins, labels=labels, include_lowest=True)
    g = df.groupby("r_bin", observed=True)["elongation"].agg(["median", "count"])
    g["r_lo_px"] = bins[:-1]
    g["r_hi_px"] = bins[1:]
    return g


def _pa_azimuth_test(df: pd.DataFrame, chip: tuple[int, int]) -> dict:
    w, h = chip
    xc, yc = w / 2.0, h / 2.0
    az = np.degrees(np.arctan2(df["y"] - yc, df["x"] - xc)) % 180.0
    folded = _fold_pa_delta(df["pa_deg"].to_numpy(dtype=float), az.to_numpy(dtype=float))
    near_rad = float(np.mean((folded < 15.0) | (folded > 75.0)))
    near_tan = float(np.mean((folded >= 35.0) & (folded <= 55.0)))
    return {
        "n": len(folded),
        "folded_median_deg": float(np.median(folded)),
        "folded_mean_deg": float(np.mean(folded)),
        "frac_near_radial_0_15_or_75_90": near_rad,
        "frac_near_tangential_35_55": near_tan,
        "folded_p25": float(np.percentile(folded, 25)),
        "folded_p75": float(np.percentile(folded, 75)),
    }


def _snr_bias_test(df: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    s = df["snr"].replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return pd.DataFrame()
    qs = np.quantile(s, np.linspace(0, 1, n_bins + 1))
    qs = np.unique(qs)
    if len(qs) < 3:
        return pd.DataFrame()
    df = df.copy()
    df["snr_bin"] = pd.cut(df["snr"], bins=qs, duplicates="drop")
    g = df.groupby("snr_bin", observed=True).agg(
        snr_median=("snr", "median"),
        elong_median=("elongation", "median"),
        n=("elongation", "count"),
    )
    return g


def _verdict(radial: pd.DataFrame, pa_test: dict, snr_bins: pd.DataFrame) -> str:
    parts = []
    if len(radial) >= 2:
        med = radial["median"].to_numpy(dtype=float)
        slope = float(med[-1] - med[0]) if np.all(np.isfinite(med)) else float("nan")
        if math.isfinite(slope) and slope >= 0.04:
            parts.append("OPTICS (elongation rises with radius)")
        elif math.isfinite(slope) and slope <= -0.04:
            parts.append("OPTICS (elongation falls with radius — possible edge/corner)")
    fr = pa_test.get("frac_near_radial_0_15_or_75_90", 0)
    ft = pa_test.get("frac_near_tangential_35_55", 0)
    if fr > 0.35 and fr > ft + 0.1:
        parts.append("OPTICS (PA aligned with radial/tangential)")
    elif ft > 0.35:
        parts.append("OPTICS (PA tangential preference)")
    elif pa_test.get("folded_median_deg", 90) < 25 and fr < 0.25:
        parts.append("MOUNT/ALIGNMENT (PA vs azimuth flat — uniform smear direction)")
    if not snr_bins.empty and len(snr_bins) >= 2:
        e = snr_bins["elong_median"].to_numpy(dtype=float)
        if np.all(np.isfinite(e)) and e[0] - e[-1] >= 0.05:
            parts.append("BIAS-FLOOR (elongation→1 at high SNR)")
        elif np.all(np.isfinite(e)) and e[-1] >= 1.08 and e[0] <= 1.05:
            parts.append("no SNR bias floor (elongation stays elevated at high SNR)")
    if not parts:
        parts.append("MIXED/inconclusive")
    return " + ".join(parts)


def analyze_draft(draft_id: int) -> dict:
    cfg = AppConfig()
    draft = Path(cfg.archive_root) / "Drafts" / f"draft_{draft_id:06d}"
    if draft_id == 362:
        setup = "NoFilter_60_2"
    else:
        setup = "Luminance_180_2"
    csv_path = draft / "diagnostics" / f"psf_elongation_{draft_id}" / f"d{draft_id}_per_star_gaussian_fits.csv"
    aligned = draft / "detrended_aligned" / "lights" / setup
    raw = pd.read_csv(csv_path, low_memory=False)
    schema = list(raw.columns)
    df, chip = _load_enriched(draft_id, csv_path, aligned)
    radial = _radial_bins(df, chip) if not df.empty else pd.DataFrame()
    pa_test = _pa_azimuth_test(df, chip) if not df.empty else {}
    snr_bins = _snr_bias_test(df) if not df.empty else pd.DataFrame()
    verdict = _verdict(radial, pa_test, snr_bins)
    return {
        "draft_id": draft_id,
        "setup": setup,
        "schema": schema,
        "chip_naxis": chip,
        "n_stars_csv": len(raw),
        "n_stars_with_xy": len(df),
        "radial": radial,
        "pa_test": pa_test,
        "snr_bins": snr_bins,
        "verdict": verdict,
    }


def main() -> None:
    for did in (362, 364):
        r = analyze_draft(did)
        print(f"=== DRAFT {did} ({r['setup']}) ===")
        print(f"Schema: {r['schema']}")
        print(f"Chip NAXIS1 x NAXIS2: {r['chip_naxis'][0]} x {r['chip_naxis'][1]}")
        print(f"Stars in CSV: {r['n_stars_csv']}; with x,y join: {r['n_stars_with_xy']}")
        print("(A) Radial elongation bins:")
        if r["radial"].empty:
            print("  (no data)")
        else:
            for idx, row in r["radial"].iterrows():
                print(f"  {idx}: r={row['r_lo_px']:.0f}-{row['r_hi_px']:.0f}px  median_elong={row['median']:.4f}  n={int(row['count'])}")
        print("(B) PA - azimuth folded:")
        for k, v in r["pa_test"].items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
        print("(C) Elongation vs SNR bin:")
        if r["snr_bins"].empty:
            print("  (no SNR data)")
        else:
            for idx, row in r["snr_bins"].iterrows():
                print(f"  {idx}: SNR~{row['snr_median']:.1f}  median_elong={row['elong_median']:.4f}  n={int(row['n'])}")
        print(f"(D) Verdict: {r['verdict']}")
        print()


if __name__ == "__main__":
    main()
