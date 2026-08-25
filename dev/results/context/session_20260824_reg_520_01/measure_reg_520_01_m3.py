"""REG-520-01 addon: Gaia-DAO residuals, completeness-filtered bright LC."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src_py"))

G520 = REPO / "Archive" / "Drafts" / "draft_000520" / "platesolve" / "g_60_4"
MS = pd.read_csv(G520 / "masterstars_full_match.csv", dtype={"catalog_id": str})
CEN = pd.read_csv(G520 / "gaia_source_state_census.csv", dtype={"catalog_id": str})
PROC = REPO / "Archive" / "Drafts" / "draft_000520" / "detrended_aligned" / "lights" / "g_60_4"
TARGET = "1111749368289526912"


def cid(v: object) -> str:
    s = str(v or "").strip()
    if s.endswith(".0") and s[:-2].replace("-", "").isdigit():
        s = s[:-2]
    return s


def lc_from(frames, target, comps, flux_col="dao_flux"):
    dms = []
    nused = []
    want = set(comps)
    for df in frames:
        c = df["catalog_id"].map(cid)
        trow = df.loc[c == target]
        if trow.empty:
            continue
        ft = float(pd.to_numeric(trow.iloc[0][flux_col], errors="coerce"))
        if not (math.isfinite(ft) and ft > 0):
            continue
        fl = pd.to_numeric(df.loc[c.isin(want), flux_col], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(fl) & (fl > 0)
        if int(ok.sum()) < 2:
            continue
        fc = float(np.median(fl[ok]))
        dms.append(-2.5 * math.log10(ft / fc))
        nused.append(int(ok.sum()))
    arr = np.asarray(dms, dtype=float)
    ooe = arr[arr <= np.quantile(arr, 0.33)] if arr.size >= 6 else arr
    return {
        "n": int(arr.size),
        "lc_rms": float(np.std(arr)) if arr.size else None,
        "lc_rms_ooe": float(np.std(ooe)) if ooe.size >= 3 else None,
        "mean_n_comp": float(np.mean(nused)) if nused else None,
        "n_comp_ids": len(comps),
    }


def main() -> None:
    MS["catalog_id"] = MS["catalog_id"].map(cid)
    CEN["catalog_id"] = CEN["catalog_id"].map(cid)
    m = MS.merge(CEN, on="catalog_id", suffixes=("", "_cen"))
    m["dx"] = pd.to_numeric(m["x"], errors="coerce") - pd.to_numeric(m["x_gaia"], errors="coerce")
    m["dy"] = pd.to_numeric(m["y"], errors="coerce") - pd.to_numeric(m["y_gaia"], errors="coerce")
    m["d_gaia_px"] = np.hypot(m["dx"], m["dy"])
    m["g"] = pd.to_numeric(m["phot_g_mean_mag"], errors="coerce")

    abc = json.loads((HERE / "m1_abc.json").read_text(encoding="utf-8"))
    sel = set(abc["selected_ids_today"])
    june = set(abc["june_band_ids"])

    def stats(ids):
        sub = m[m["catalog_id"].isin(ids)]
        d = pd.to_numeric(sub["d_gaia_px"], errors="coerce").dropna()
        return {
            "n": int(len(sub)),
            "d_median": float(d.median()) if len(d) else None,
            "d_p95": float(d.quantile(0.95)) if len(d) else None,
            "n_d_le_1.0": int((d <= 1.0).sum()),
            "n_d_le_2.0": int((d <= 2.0).sum()),
        }

    frames = []
    for p in sorted(PROC.glob("proc_*.csv")):
        df = pd.read_csv(p, usecols=lambda c: c in {"catalog_id", "dao_flux", "flux", "bjd_tdb_mid"}, dtype={"catalog_id": str})
        df["catalog_id"] = df["catalog_id"].map(cid)
        frames.append(df)
    nfr = len(frames)
    counts = {}
    for df in frames:
        for c, fl in zip(df["catalog_id"], pd.to_numeric(df["dao_flux"], errors="coerce")):
            if math.isfinite(float(fl)) and float(fl) > 0:
                counts[c] = counts.get(c, 0) + 1

    june_complete = [c for c in abc["june_detected_ids"] if counts.get(c, 0) >= 20]
    june_complete8 = sorted(june_complete, key=lambda c: float(m.loc[m["catalog_id"] == c, "g"].iloc[0]) if c in set(m["catalog_id"]) else 99)[:8]
    g14_complete = [
        c
        for c in m.loc[(m["g"] < 14) & (m["catalog_id"] != TARGET) & (m["source_state"].isin(["DETECTED_P1", "DETECTED_P2"])), "catalog_id"]
        if counts.get(c, 0) >= 20
    ]
    g14_complete8 = sorted(
        g14_complete,
        key=lambda c: float(m.loc[m["catalog_id"] == c, "g"].iloc[0]),
    )[:8]

    out = {
        "residual_selected": stats(sel),
        "residual_june_band": stats(june),
        "residual_Glt12": stats(set(m.loc[m["g"] < 12, "catalog_id"])),
        "n_frames": nfr,
        "june_n_complete_ge20": len(june_complete),
        "june_complete8": june_complete8,
        "lc_june_complete8": lc_from(frames, TARGET, june_complete8),
        "lc_june_complete_all": lc_from(frames, TARGET, june_complete),
        "g14_complete_ge20": len(g14_complete),
        "lc_g14_complete8": lc_from(frames, TARGET, g14_complete8),
        "lc_g14_complete_all": lc_from(frames, TARGET, g14_complete),
        "lc_today_selected": lc_from(frames, TARGET, list(sel)),
    }
    (HERE / "m3_bright_ensemble.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
