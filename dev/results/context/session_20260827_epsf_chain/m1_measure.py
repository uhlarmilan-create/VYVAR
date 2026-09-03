# -*- coding: ascii -*-
"""M1: freeze proc CSV psf_* for BO CVn comps at frame 0 and frame 60."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src_py"))
sys.path.insert(0, str(REPO / "dev"))

from gaia_catalog_id import normalize_gaia_source_id  # noqa: E402
from pinned_ensembles import get_pinned_members_for_target  # noqa: E402
from psf_internal_lc import resolve_ensemble_ids  # noqa: E402

BO = "1498613634033133184"
SNAP = REPO / "Archive" / "Drafts" / "draft_000516_snapshot_era04_20260826"
SETUP = "NoFilter_60_2"
OUT = Path(__file__).resolve().parent


def _norm(x) -> str:
    try:
        return str(normalize_gaia_source_id(x)).strip()
    except Exception:
        return str(x or "").strip()


def _cell(row: pd.Series, col: str):
    if col not in row.index:
        return {"present": False, "value": None}
    v = row[col]
    if pd.isna(v):
        return {"present": True, "value": None, "isnan": True}
    if col == "psf_fit_ok":
        return {"present": True, "value": bool(v)}
    try:
        fv = float(v)
        return {"present": True, "value": fv, "isfinite": math.isfinite(fv)}
    except (TypeError, ValueError):
        return {"present": True, "value": str(v)}


def main() -> None:
    lights = SNAP / "detrended_aligned" / "lights" / SETUP
    phot = SNAP / "platesolve" / SETUP / "photometry"
    procs = sorted(lights.glob("proc_*.csv"))
    fits = sorted(
        p for p in lights.glob("*.fits") if p.name.upper() != "MASTERSTAR.FITS"
    )
    rec = {
        "n_proc": len(procs),
        "n_fits": len(fits),
        "frame0": str(procs[0].name) if procs else None,
        "frame60": str(procs[60].name) if len(procs) > 60 else None,
    }
    ids, weights, src = resolve_ensemble_ids(BO, phot)
    rec["ensemble_source"] = src
    rec["n_comps"] = len(ids)
    rec["comp_ids"] = ids
    rec["weights"] = {k: float(v) for k, v in weights.items()}
    pinned = get_pinned_members_for_target(BO)
    rec["pinned_n"] = 0 if pinned is None else len(pinned)

    frames = {}
    for key, idx in (("frame0", 0), ("frame60", 60)):
        if idx >= len(procs):
            frames[key] = {"error": "missing"}
            continue
        p = procs[idx]
        df = pd.read_csv(p, low_memory=False)
        cols = [c for c in df.columns if str(c).startswith("psf_")]
        cid = df["catalog_id"].map(_norm) if "catalog_id" in df.columns else pd.Series([], dtype=str)
        rows = []
        missing_in_csv = []
        for cid_s in ids:
            hit = df.loc[cid == cid_s]
            if hit.empty:
                missing_in_csv.append(cid_s)
                rows.append({"catalog_id": cid_s, "in_csv": False})
                continue
            r = hit.iloc[0]
            rows.append(
                {
                    "catalog_id": cid_s,
                    "in_csv": True,
                    "psf_fit_ok": _cell(r, "psf_fit_ok"),
                    "psf_flux": _cell(r, "psf_flux"),
                    "psf_chi2": _cell(r, "psf_chi2"),
                }
            )
        frames[key] = {
            "file": p.name,
            "n_rows": int(len(df)),
            "psf_columns": cols,
            "comps": rows,
            "comps_not_in_csv": missing_in_csv,
        }
    rec["frames"] = frames
    (OUT / "m1.json").write_text(json.dumps(rec, indent=2) + "\n", encoding="ascii")
    print("wrote m1.json")
    print("ensemble", src, "n", len(ids))
    print("n_proc", rec["n_proc"], "frame0", rec["frame0"], "frame60", rec["frame60"])
    for key in ("frame0", "frame60"):
        fr = frames[key]
        print(key, fr.get("file"), "psf_cols", fr.get("psf_columns"))
        for row in fr.get("comps") or []:
            print(" ", row.get("catalog_id"), "in", row.get("in_csv"),
                  "ok", (row.get("psf_fit_ok") or {}).get("value"),
                  "flux", (row.get("psf_flux") or {}).get("value"),
                  "chi2", (row.get("psf_chi2") or {}).get("value"))


if __name__ == "__main__":
    main()
