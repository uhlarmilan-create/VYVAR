# -*- coding: ascii -*-
"""C3-0: k = p90 of (iv LOO MAD mag / photon) on R2 516 live comps. No wiring."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\ASTRO\python\VYVAR")
sys.path.insert(0, str(ROOT / "src_py"))
from masterstar_gaia_accounting import _norm_cid  # noqa: E402

OUTDIR = Path(__file__).resolve().parent
B3 = ROOT / "dev" / "results" / "session_20260825_sel_ghost_01_b3"
R2 = B3 / "t3_r2"
SETUP = "NoFilter_60_2"
LIVE520 = ROOT / "Archive" / "Drafts" / "draft_000520"
T4 = B3 / "t4_520"
LN10_2P5 = 1.0857362047581294
SEVEN = [
    "1112113680298377344",
    "1111920204908702336",
    "1112110695298081664",
    "1111749157833870208",
    "1112121862213003648",
    "1112121067641532160",
    "1111737033143440768",
]


def cid(v: object) -> str:
    return _norm_cid(v)


def mag_from_flux(f: float) -> float:
    if not (math.isfinite(f) and f > 0):
        return float("nan")
    return -2.5 * math.log10(f)


def mad_sigma(xs: list[float]) -> float:
    a = np.asarray([x for x in xs if math.isfinite(x)], dtype=np.float64)
    if a.size < 3:
        return float("nan")
    med = float(np.median(a))
    return float(1.4826 * np.median(np.abs(a - med)))


def load_proc(phot: Path) -> dict[str, pd.DataFrame]:
    cache: dict[str, pd.DataFrame] = {}
    lights = phot / "proc"
    if not lights.is_dir():
        lights = phot
    for p in sorted(phot.glob("proc_*.csv")) + sorted((phot / "lights").glob("*.csv") if (phot / "lights").is_dir() else []):
        try:
            df = pd.read_csv(p, low_memory=False)
        except Exception:
            continue
        col = "catalog_id" if "catalog_id" in df.columns else ("name" if "name" in df.columns else None)
        if col is None:
            continue
        df = df.copy()
        df["_cid"] = df[col].map(cid)
        cache[str(p)] = df
    if not cache:
        for p in sorted(phot.rglob("*.csv")):
            name = p.name.lower()
            if not (name.startswith("proc_") or "light" in name):
                continue
            if "lightcurve" in name or "comparison" in name or "masterstar" in name:
                continue
            try:
                df = pd.read_csv(p, low_memory=False)
            except Exception:
                continue
            col = "catalog_id" if "catalog_id" in df.columns else ("name" if "name" in df.columns else None)
            if col is None:
                continue
            df = df.copy()
            df["_cid"] = df[col].map(cid)
            cache[str(p)] = df
    return cache


def loo_dmag(cache: dict[str, pd.DataFrame], star: str, pool: list[str]) -> list[float]:
    others = [s for s in pool if s != star]
    dms: list[float] = []
    for df in cache.values():
        col = "dao_flux" if "dao_flux" in df.columns else "flux"
        hit = df.loc[df["_cid"] == star]
        if hit.empty:
            continue
        ft = float(pd.to_numeric(hit.iloc[0][col], errors="coerce"))
        if not (math.isfinite(ft) and ft > 0):
            continue
        fl = pd.to_numeric(df.loc[df["_cid"].isin(others), col], errors="coerce").to_numpy(dtype=float)
        fl = fl[np.isfinite(fl) & (fl > 0)]
        if fl.size < 1:
            continue
        fc = float(np.median(fl))
        if not (math.isfinite(fc) and fc > 0):
            continue
        dms.append(mag_from_flux(ft) - mag_from_flux(fc))
    return dms


def photon(ms: pd.DataFrame, star: str) -> float:
    hit = ms.loc[ms["_cid"] == star]
    if hit.empty:
        return float("nan")
    for col in ("snr_ap_pixscaled", "snr"):
        if col in hit.columns:
            snr = float(pd.to_numeric(hit.iloc[0][col], errors="coerce"))
            if math.isfinite(snr) and snr > 0:
                return LN10_2P5 / snr
    return float("nan")


def round_up_1sig(x: float) -> float:
    if not (math.isfinite(x) and x > 0):
        return float("nan")
    exp = math.floor(math.log10(x))
    mant = x / (10 ** exp)
    mant_up = math.ceil(mant - 1e-12)
    return float(mant_up * (10 ** exp))


def k_from_p90(p90: float) -> float:
    if 3.0 <= p90 <= 5.0:
        return 5.0
    return round_up_1sig(p90)


def hist_counts(vals: list[float], edges: list[float]) -> list[dict]:
    a = np.asarray([v for v in vals if math.isfinite(v)], dtype=np.float64)
    counts, ed = np.histogram(a, bins=np.asarray(edges, dtype=np.float64))
    out = []
    for i, c in enumerate(counts):
        out.append({"lo": float(ed[i]), "hi": float(ed[i + 1]), "n": int(c)})
    return out


def main() -> int:
    phot = R2 / "platesolve" / SETUP / "photometry"
    ms = pd.read_csv(R2 / "platesolve" / SETUP / "masterstars_full_match.csv", dtype={"catalog_id": str})
    ms["_cid"] = ms["catalog_id"].map(cid)
    comps = pd.read_csv(phot / "comparison_stars_per_target.csv", dtype=str)
    comps["_tid"] = comps["target_id"].map(cid) if "target_id" in comps.columns else comps.iloc[:, 0].map(cid)
    cid_col = "catalog_id" if "catalog_id" in comps.columns else "comp_id"
    comps["_cid"] = comps[cid_col].map(cid)
    live = sorted({c for c in comps["_cid"].tolist() if c})
    ensembles: dict[str, list[str]] = {}
    for tid, g in comps.groupby("_tid"):
        ensembles[str(tid)] = [c for c in g["_cid"].tolist() if c]
    mates: dict[str, set[str]] = {s: set() for s in live}
    for members in ensembles.values():
        sset = set(members)
        for s in sset:
            if s in mates:
                mates[s] |= sset - {s}

    cache = load_proc(phot)
    lights_proc = R2 / "detrended_aligned" / "lights" / SETUP
    if lights_proc.is_dir():
        cache.update(load_proc(lights_proc))
    rows = []
    ratios = []
    for s in live:
        pool = sorted(mates.get(s) or [])
        if len(pool) < 1:
            pool = [x for x in live if x != s]
        loo = loo_dmag(cache, s, pool + [s])
        iv = mad_sigma(loo)
        ph = photon(ms, s)
        r = iv / ph if (math.isfinite(iv) and math.isfinite(ph) and ph > 0) else float("nan")
        ratios.append(r)
        rows.append(
            {
                "catalog_id": s,
                "n_ensemble_mates": len(pool),
                "n_loo": len(loo),
                "iv_mad_loo": iv,
                "snr_ap_pixscaled": float(pd.to_numeric(ms.loc[ms["_cid"] == s, "snr_ap_pixscaled"].iloc[0], errors="coerce"))
                if "snr_ap_pixscaled" in ms.columns and (ms["_cid"] == s).any()
                else float("nan"),
                "photon": ph,
                "r": r,
            }
        )
    finite_r = [x for x in ratios if math.isfinite(x)]
    finite_r.sort()

    def pct(xs: list[float], p: float) -> float:
        if not xs:
            return float("nan")
        i = int(round((p / 100.0) * (len(xs) - 1)))
        return float(xs[max(0, min(i, len(xs) - 1))])

    p50, p90, rmax = pct(finite_r, 50), pct(finite_r, 90), (max(finite_r) if finite_r else float("nan"))
    k = k_from_p90(p90)

    seven_rows = []
    cache520 = {}
    ms520_p = LIVE520 / "platesolve" / "g_60_4" / "masterstars_full_match.csv"
    if not ms520_p.is_file() and T4.is_dir():
        hits = list(T4.rglob("masterstars_full_match.csv"))
        ms520_p = hits[0] if hits else ms520_p
    if ms520_p.is_file():
        ms520 = pd.read_csv(ms520_p, dtype={"catalog_id": str})
        ms520["_cid"] = ms520["catalog_id"].map(cid)
        phot520 = ms520_p.parent / "photometry"
        if not phot520.is_dir():
            phot520 = T4 / "platesolve" / "g_60_4" / "photometry" if (T4 / "platesolve").is_dir() else phot520
        cache520 = load_proc(phot520) if phot520.is_dir() else {}
        if not cache520:
            for p in sorted(T4.rglob("proc_*.csv")):
                df = pd.read_csv(p, low_memory=False)
                col = "catalog_id" if "catalog_id" in df.columns else "name"
                df = df.copy()
                df["_cid"] = df[col].map(cid)
                cache520[str(p)] = df
        for s in SEVEN:
            loo = loo_dmag(cache520, s, SEVEN)
            iv = mad_sigma(loo)
            ph = photon(ms520, s)
            r = iv / ph if (math.isfinite(iv) and math.isfinite(ph) and ph > 0) else float("nan")
            seven_rows.append({"catalog_id": s, "iv_mad_loo": iv, "photon": ph, "r": r, "n_loo": len(loo)})

    rec = {
        "premise": "iv LOO MAD mag vs own ensembles on T3 R2 516; photon=1.0857/snr_ap_pixscaled",
        "n_live_comps": len(live),
        "n_proc_frames": len(cache),
        "n_ratio_finite": len(finite_r),
        "p50": p50,
        "p90": p90,
        "max": rmax,
        "k_rule": "p90 rounded UP to 1 sig fig; if 3<=p90<=5 then k=5",
        "k": k,
        "histogram": hist_counts(finite_r, [0, 1, 2, 3, 4, 5, 8, 10, 20, 50, 100, 1e9]),
        "rows": rows,
        "seven_520": seven_rows,
        "ms_snr_col": "snr_ap_pixscaled" if "snr_ap_pixscaled" in ms.columns else list(ms.columns)[:20],
    }
    outp = OUTDIR / "c30_k.json"
    outp.write_text(json.dumps(rec, indent=2, default=str), encoding="ascii")
    print("C3-0 n", len(live), "frames", len(cache), "p50", p50, "p90", p90, "max", rmax, "k", k)
    print("wrote", outp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
