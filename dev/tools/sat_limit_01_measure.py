#!/usr/bin/env python3
"""SAT-LIMIT-01 measurements: catalog NaNs, DB row, knee, optional reclassify."""
from __future__ import annotations

import json
import math
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src_py"))

DRAFT = ROOT / "Archive" / "Drafts" / "draft_000515"
SETUP = "NoFilter_60_2"
MS_CSV = DRAFT / "platesolve" / SETUP / "masterstars_full_match.csv"
MS_FITS = DRAFT / "platesolve" / SETUP / "MASTERSTAR.fits"
SAT_JSON = DRAFT / "sat_diag.json"
COMP_CSV = DRAFT / "platesolve" / SETUP / "photometry" / "comparison_stars_per_target.csv"
BO = "1498613634033133184"
C2 = "1500748301498613248"
CONTAINER_CLIP = 65535.0


def _finite(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)


def catalog_counts() -> dict:
    ms = pd.read_csv(MS_CSV, dtype={"catalog_id": str}, low_memory=False)
    ms["catalog_id"] = ms["catalog_id"].astype(str).str.strip()
    out: dict = {"n_rows": int(len(ms))}
    for c in ("saturate_limit_adu", "saturate_limit_adu_85pct"):
        s = pd.to_numeric(ms[c], errors="coerce") if c in ms.columns else pd.Series(dtype=float)
        out[c] = {
            "n_nan": int(s.isna().sum()) if c in ms.columns else None,
            "n_finite": int(s.notna().sum()) if c in ms.columns else None,
        }
    out["is_saturated_true"] = int(ms["is_saturated"].astype(bool).sum()) if "is_saturated" in ms.columns else None
    out["likely_saturated_true"] = (
        int(ms["likely_saturated"].astype(bool).sum()) if "likely_saturated" in ms.columns else None
    )
    if "zone" in ms.columns:
        out["zone_counts"] = {
            str(k): int(v) for k, v in ms["zone"].astype(str).str.strip().str.lower().value_counts().items()
        }
    pk = pd.to_numeric(ms["peak_max_adu"], errors="coerce")
    out["peak_max_adu"] = {
        "min": float(pk.min()),
        "median": float(pk.median()),
        "max": float(pk.max()),
        "n_ge_0.80clip": int((pk >= 0.80 * CONTAINER_CLIP).sum()),
        "n_ge_0.85clip": int((pk >= 0.85 * CONTAINER_CLIP).sum()),
        "n_ge_65000": int((pk >= 65000).sum()),
    }
    row = ms[ms["catalog_id"] == C2]
    if len(row):
        r = row.iloc[0]
        out["C2"] = {
            "catalog_id": C2,
            "phot_g_mean_mag": float(pd.to_numeric(r.get("phot_g_mean_mag"), errors="coerce")),
            "peak_max_adu": float(pd.to_numeric(r.get("peak_max_adu"), errors="coerce")),
            "zone": str(r.get("zone")),
            "is_saturated": bool(r.get("is_saturated")),
            "likely_saturated": bool(r.get("likely_saturated")),
            "pct_of_clip": float(pd.to_numeric(r.get("peak_max_adu"), errors="coerce")) / CONTAINER_CLIP * 100.0,
        }
    if "likely_saturated" in ms.columns:
        hit = ms[ms["likely_saturated"].astype(bool)]
        out["likely_saturated_ids"] = hit["catalog_id"].astype(str).tolist()
        out["likely_saturated_peaks"] = pd.to_numeric(hit["peak_max_adu"], errors="coerce").tolist()
    return out


def equipment_row() -> dict:
    from config import AppConfig

    dbp = Path(AppConfig().database_path)
    con = sqlite3.connect(str(dbp))
    con.row_factory = sqlite3.Row
    eq_cols = [r[1] for r in con.execute("PRAGMA table_info(EQUIPMENTS)").fetchall()]
    eq_sel = ", ".join(
        c for c in ("ID", "CAMERANAME", "ALIAS", "SATURATE_ADU", "GAIN_ADU", "READNOISE_E") if c in eq_cols
    )
    eq = con.execute(f"SELECT {eq_sel} FROM EQUIPMENTS WHERE ID = 1").fetchone()
    d_cols = [r[1] for r in con.execute("PRAGMA table_info(DRAFTS)").fetchall()]
    d_sel = ", ".join(c for c in ("ID", "ID_EQUIPMENT", "NAME", "TITLE") if c in d_cols)
    draft: dict | sqlite3.Row | None = None
    if d_sel:
        try:
            row = con.execute(f"SELECT {d_sel} FROM DRAFTS WHERE ID = 515").fetchone()
            draft = dict(row) if row else None
        except sqlite3.Error:
            draft = None
    if draft is None:
        names = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        for tname in names:
            cols = [r[1] for r in con.execute(f"PRAGMA table_info({tname})").fetchall()]
            if "ID_EQUIPMENT" in cols and "ID" in cols:
                try:
                    row = con.execute(f"SELECT ID, ID_EQUIPMENT FROM {tname} WHERE ID = 515").fetchone()
                except sqlite3.Error:
                    continue
                if row is not None:
                    draft = {"table": tname, **dict(row)}
                    break
        if draft is None:
            draft = {"draft_table_not_found": True, "id_equipment_tables": [n for n in names if "DRAFT" in n.upper()]}
    con.close()
    return {
        "database_path": str(dbp),
        "equipment_id_1": dict(eq) if eq else None,
        "draft_515": dict(draft) if draft is not None else None,
    }


def sat_diag_summary() -> dict:
    data = json.loads(SAT_JSON.read_text(encoding="utf-8"))
    keys = (
        "sat_adu",
        "sat_source",
        "lin_adu",
        "lin_source",
        "header_value",
        "equipment_value",
        "bitpix_ceiling",
        "derived_ceiling",
    )
    return {k: data.get(k) for k in keys}


def masterstar_fits_summary() -> dict:
    from astropy.io import fits

    if not MS_FITS.is_file():
        return {"missing": True}
    with fits.open(MS_FITS, memmap=True) as hd:
        h = hd[0].header
        d = hd[0].data
        out = {
            "BITPIX": h.get("BITPIX"),
            "SATURATE": h.get("SATURATE"),
            "MAXLIN": h.get("MAXLIN"),
            "LINLIMIT": h.get("LINLIMIT"),
            "MAXADU": h.get("MAXADU"),
            "DATAMAX": h.get("DATAMAX"),
            "MAXPIX": h.get("MAXPIX"),
        }
        if d is not None:
            a = np.asarray(d, dtype=np.float64)
            out["data_min"] = float(np.nanmin(a))
            out["data_median"] = float(np.nanmedian(a))
            out["data_max"] = float(np.nanmax(a))
        return out


def knee_measurement() -> dict:
    """D1-2 cheap check: inst-minus-catalog mag residual vs peak_max_adu."""
    ms = pd.read_csv(MS_CSV, dtype={"catalog_id": str}, low_memory=False)
    g = pd.to_numeric(ms.get("phot_g_mean_mag", ms.get("mag")), errors="coerce")
    flux = pd.to_numeric(ms.get("flux"), errors="coerce")
    pk = pd.to_numeric(ms.get("peak_max_adu"), errors="coerce")
    zone = ms["zone"].astype(str).str.strip().str.lower() if "zone" in ms.columns else pd.Series("", index=ms.index)
    sat = ms["is_saturated"].astype(bool) if "is_saturated" in ms.columns else pd.Series(False, index=ms.index)
    vsx = (
        ms["vsx_known_variable"].astype(bool)
        if "vsx_known_variable" in ms.columns
        else pd.Series(False, index=ms.index)
    )
    inst = -2.5 * np.log10(flux)
    ok = np.isfinite(g) & np.isfinite(inst) & np.isfinite(pk) & (flux > 0) & ~sat & ~vsx
    # Prefer currently-linear stars; fall back to all finite if zone empty/unknown.
    lin = ok & zone.eq("linear")
    use = lin if int(lin.sum()) >= 50 else ok
    resid = inst - g
    # ZP: median residual on a mid-brightness, mid-peak cohort (not the bright end).
    mid = use & (g >= 10.0) & (g <= 13.0) & (pk >= 2000) & (pk <= 20000)
    if int(mid.sum()) < 20:
        mid = use & (g >= 10.0) & (g <= 13.0)
    zp = float(np.median(resid[mid.to_numpy()])) if int(mid.sum()) else float(np.median(resid[use.to_numpy()]))
    adj = resid - zp
    peaks = pk.to_numpy(dtype=float)
    adj_a = adj.to_numpy(dtype=float)
    mask = use.to_numpy()
    # Bin by peak ADU; look for a monotonic bright-end departure of median residual.
    edges = np.array([0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000, 65536], dtype=float)
    bins = []
    for i in range(len(edges) - 1):
        m = mask & (peaks >= edges[i]) & (peaks < edges[i + 1])
        n = int(m.sum())
        if n < 5:
            bins.append({"lo": float(edges[i]), "hi": float(edges[i + 1]), "n": n, "median_resid_mag": None})
            continue
        bins.append(
            {
                "lo": float(edges[i]),
                "hi": float(edges[i + 1]),
                "n": n,
                "median_resid_mag": float(np.median(adj_a[m])),
                "median_peak_adu": float(np.median(peaks[m])),
                "median_G": float(np.median(g.to_numpy()[m])),
            }
        )
    # Detector: first bin above 20k ADU whose |median resid| exceeds 3x the 5-20k scatter.
    # Physics: accept only if the next finite bin continues the same-sign departure.
    ref = [b for b in bins if b["lo"] >= 5000 and b["hi"] <= 20000 and b["median_resid_mag"] is not None]
    ref_s = float(np.std([b["median_resid_mag"] for b in ref])) if len(ref) >= 2 else float("nan")
    auto_knee_adu = None
    for b in bins:
        if b["lo"] < 20000 or b["median_resid_mag"] is None:
            continue
        if math.isfinite(ref_s) and abs(b["median_resid_mag"]) > max(0.03, 3.0 * ref_s):
            auto_knee_adu = float(b["lo"])
            break
    physics_resolved = False
    physics_review = "no detector flag"
    if auto_knee_adu is not None:
        flagged = [b for b in bins if b["lo"] == auto_knee_adu][0]
        later = [b for b in bins if b["lo"] > auto_knee_adu and b["median_resid_mag"] is not None]
        sign0 = math.copysign(1.0, float(flagged["median_resid_mag"]))
        if later:
            r1 = float(later[0]["median_resid_mag"])
            if math.copysign(1.0, r1) == sign0 and abs(r1) >= 0.5 * abs(float(flagged["median_resid_mag"])):
                physics_resolved = True
                physics_review = "next finite bin continues same-sign departure"
            else:
                physics_review = (
                    "NOT a resolved linearity knee (detector bin reverses or shrinks next; "
                    "bright-end clip saturation is not a sub-clip knee)"
                )
        else:
            physics_review = "NOT a resolved linearity knee (no next finite bin to confirm)"
    chosen = float(auto_knee_adu) if physics_resolved else 0.80 * CONTAINER_CLIP
    return {
        "n_used": int(mask.sum()),
        "n_linear_mask": int(lin.sum()),
        "zp_mag": zp,
        "n_zp_cohort": int(mid.sum()),
        "ref_bin_std_mag": ref_s,
        "knee_adu": float(auto_knee_adu) if physics_resolved else None,
        "knee_resolved": bool(physics_resolved),
        "auto_detector_knee_adu": auto_knee_adu,
        "auto_detector_resolved": auto_knee_adu is not None,
        "physics_review": physics_review,
        "bins": bins,
        "chosen_limit_adu": chosen,
        "chosen_source": (
            f"data_knee_lo_edge_{auto_knee_adu:.0f}"
            if physics_resolved
            else "conservative_default_0.80x_container_clip_65535"
        ),
    }


def main() -> None:
    payload = {
        "catalog": catalog_counts(),
        "equipment": equipment_row(),
        "sat_diag": sat_diag_summary(),
        "masterstar_fits": masterstar_fits_summary(),
        "knee": knee_measurement(),
    }
    out = ROOT / "dev" / "results" / "SAT_LIMIT_01_summary.json"
    out.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="ascii")
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
