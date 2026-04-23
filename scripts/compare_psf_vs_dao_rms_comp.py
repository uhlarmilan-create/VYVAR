"""Compare COMP star flux stability: dao_flux (proc CSV) vs PSF flux (*_psf.csv), draft_000246 epsf_data."""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import pandas as pd


def cid_key(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    s = str(v).strip()
    if not s or s.lower() in ("nan", "none"):
        return ""
    try:
        f = float(v)
        if math.isfinite(f) and abs(f) > 1e10:
            return str(int(f))
        if math.isfinite(f) and float(int(f)) == f:
            return str(int(f))
    except (TypeError, ValueError, OverflowError):
        pass
    if s.endswith(".0") and s[:-2].isdigit():
        return s[:-2]
    return s


def main() -> None:
    root = Path(r"C:\ASTRO\python\VYVAR\Archive\Drafts\draft_000246\epsf_data\lights\NoFilter_60_2")
    psf_dir = Path(r"C:\ASTRO\python\VYVAR\Archive\Drafts\draft_000246\epsf_data\psf_results")
    sum_path = psf_dir / "psf_summary.csv"

    proc_files = sorted(root.glob("proc_*.csv"))
    print(f"proc_*.csv frames: {len(proc_files)}")
    if not proc_files:
        return
    d0 = pd.read_csv(proc_files[0], nrows=1)
    print(f"dao_flux in proc CSV: {'dao_flux' in d0.columns}")

    pairs: list[tuple[Path, Path]] = []
    for pcsv in proc_files:
        psfp = psf_dir / f"{pcsv.stem}_psf.csv"
        if psfp.is_file():
            pairs.append((pcsv, psfp))
    print(f"aligned proc+psf pairs: {len(pairs)}")

    dao_v: dict[str, list[float]] = defaultdict(list)
    psf_v: dict[str, list[float]] = defaultdict(list)
    for pcsv, psfp in pairs:
        proc = pd.read_csv(pcsv, usecols=["catalog_id", "dao_flux"], low_memory=False)
        psf = pd.read_csv(psfp, usecols=["catalog_id", "role", "psf_flux", "psf_fit_ok"], low_memory=False)
        proc["_cid"] = proc["catalog_id"].map(cid_key)
        proc["dao"] = pd.to_numeric(proc["dao_flux"], errors="coerce")
        psf["_cid"] = psf["catalog_id"].map(cid_key)
        psf["pf"] = pd.to_numeric(psf["psf_flux"], errors="coerce")
        psf["ok"] = psf["psf_fit_ok"].fillna(False).astype(bool)
        psf_comp = psf[psf["role"].astype(str).str.upper() == "COMP"]
        m = psf_comp.merge(proc[["_cid", "dao"]].dropna(), on="_cid", how="inner")
        m = m[m["ok"] & m["dao"].notna() & (m["dao"] > 0) & m["pf"].notna() & (m["pf"] > 0)]
        for _, r in m.iterrows():
            k = str(r["_cid"])
            dao_v[k].append(float(r["dao"]))
            psf_v[k].append(float(r["pf"]))

    sum_df = pd.read_csv(sum_path, low_memory=False)
    sum_df["_cid"] = sum_df["catalog_id"].astype(str).map(cid_key)
    comp_ids = set(
        sum_df[
            (sum_df["role"].astype(str).str.upper() == "COMP")
            & (sum_df["n_fit_ok"] == 143)
            & (sum_df["med_psf_flux"] > 1000)
        ]["_cid"].astype(str)
    )

    # (A) Summary-based PSF scatter vs all-frame dao (user's original recipe)
    rows_a: list[dict] = []
    for cid in comp_ids:
        if not cid:
            continue
        dao = dao_v.get(cid, [])
        if len(dao) < 2:
            continue
        sd = pd.Series(dao)
        md = float(sd.mean())
        if md <= 0:
            continue
        dao_rms_pct = 100.0 * float(sd.std(ddof=1)) / md
        sub = sum_df[sum_df["_cid"] == cid]
        if sub.empty:
            continue
        med = float(sub["med_psf_flux"].iloc[0])
        rms = float(sub["rms_psf_flux"].iloc[0])
        if not (math.isfinite(med) and med > 0 and math.isfinite(rms)):
            continue
        psf_rms_pct = 100.0 * rms / med
        rows_a.append(
            {
                "catalog_id": cid,
                "n_dao": len(dao),
                "dao_rms_pct": dao_rms_pct,
                "psf_rms_pct_summary": psf_rms_pct,
                "better": psf_rms_pct < dao_rms_pct,
            }
        )
    out_a = pd.DataFrame(rows_a)
    print("\n--- (A) psf_rms from psf_summary vs dao from all proc frames ---")
    if len(out_a):
        nb = int(out_a["better"].sum())
        print(f"COMP matched: {len(out_a)}")
        print(f"psf_rms_pct < dao_rms_pct: {nb}/{len(out_a)} ({100.0 * nb / len(out_a):.1f}%)")
        print(f"median dao_rms_pct: {float(out_a['dao_rms_pct'].median()):.3f}")
        print(f"median psf_rms_pct (summary): {float(out_a['psf_rms_pct_summary'].median()):.3f}")

    # (B) Paired frames only (same frame, psf fit_ok, dao>0)
    rows_b: list[dict] = []
    for cid in comp_ids:
        if not cid:
            continue
        d = dao_v.get(cid, [])
        p = psf_v.get(cid, [])
        if len(d) < 2 or len(p) < 2:
            continue
        n = min(len(d), len(p))
        d, p = d[:n], p[:n]
        sd, sp = pd.Series(d), pd.Series(p)
        md, sdd = float(sd.mean()), float(sd.std(ddof=1))
        mp, spp = float(sp.mean()), float(sp.std(ddof=1))
        if md <= 0 or mp <= 0:
            continue
        dao_rms_pct = 100.0 * sdd / md
        psf_rms_pct = 100.0 * spp / mp
        rows_b.append(
            {
                "catalog_id": cid,
                "n_paired": n,
                "dao_rms_pct": dao_rms_pct,
                "psf_rms_pct_paired": psf_rms_pct,
                "better": psf_rms_pct < dao_rms_pct,
            }
        )
    out_b = pd.DataFrame(rows_b)
    print("\n--- (B) Both metrics from paired frames (psf fit_ok, dao>0) ---")
    if len(out_b):
        nb = int(out_b["better"].sum())
        print(f"COMP matched: {len(out_b)}")
        print(f"psf_rms_pct < dao_rms_pct: {nb}/{len(out_b)} ({100.0 * nb / len(out_b):.1f}%)")
        print(f"median dao_rms_pct: {float(out_b['dao_rms_pct'].median()):.3f}")
        print(f"median psf_rms_pct (paired): {float(out_b['psf_rms_pct_paired'].median()):.3f}")


if __name__ == "__main__":
    main()
