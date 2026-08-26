# -*- coding: ascii -*-
"""Ledger v5: era03 vs APERTURE-01c era04 (f=1.35). STOP if any UNNAMED."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\ASTRO\python\VYVAR")
ERA03 = ROOT / "Archive" / "Drafts" / "draft_000516_snapshot_era03_20260820"
ERA04 = ROOT / "Archive" / "Drafts" / "draft_000516_snapshot_era04_20260826"
SETUP = "NoFilter_60_2"
OUT = Path(__file__).resolve().parent
V4 = ROOT / "dev" / "results" / "context" / "session_20260826_c6" / "c63_era03_era04_ledger_v4.csv"
BO = "1498613634033133184"
FW = "1497343732462852864"
GH = "1498804639818507904"
NAMED_KEEP = {
    "1498000793739050368": "FRAME-29",
    "1497284015237511808": "D3-D5",
    "1499209638054824320": "D3-D5",
    "1496315070616056064": "C3-K5",
    "1497169940906156032": "NAME-FIX",
    "1485560025830226432": "EDGE",
    "1496037650087948160": "EDGE",
    "1496733984545821696": "EDGE",
    "1497491273179203456": "EDGE",
}


def phot(root: Path) -> Path:
    return root / "platesolve" / SETUP / "photometry"


def med_delta_mmag(a: pd.DataFrame, b: pd.DataFrame, col: str) -> float:
    ka = a["source_file"].map(lambda s: Path(str(s)).name)
    kb = b["source_file"].map(lambda s: Path(str(s)).name)
    m = pd.DataFrame({"k": ka, "a": pd.to_numeric(a[col], errors="coerce")}).merge(
        pd.DataFrame({"k": kb, "b": pd.to_numeric(b[col], errors="coerce")}), on="k"
    )
    d = m["b"] - m["a"]
    d = d[np.isfinite(d)]
    if d.empty:
        return float("nan")
    return float(d.median() * 1000.0)


def demeaned_rms(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 8:
        return float("nan")
    d = x - float(np.median(x))
    return float(np.sqrt(np.mean(d * d)) * 1000.0)


def main() -> int:
    e3 = phot(ERA03) / "lightcurves"
    e4 = phot(ERA04) / "lightcurves"
    ids3 = {
        p.stem.replace("lightcurve_", "")
        for p in e3.glob("lightcurve_*.csv")
        if not p.stem.endswith(("_psf", "_adaptive"))
    }
    ids4 = {
        p.stem.replace("lightcurve_", "")
        for p in e4.glob("lightcurve_*.csv")
        if not p.stem.endswith(("_psf", "_adaptive"))
    }
    v4_map = {}
    if V4.is_file():
        v4 = pd.read_csv(V4, dtype={"target": str})
        for _, row in v4.iterrows():
            v4_map[str(row["target"])] = str(row.get("cause") or "")

    rows = []
    unnamed = []
    residuals = {}
    for tid in sorted(ids3 | ids4):
        p3 = e3 / f"lightcurve_{tid}.csv"
        p4 = e4 / f"lightcurve_{tid}.csv"
        files = []
        n3 = n4 = 0
        dmag_c = dmag_f = drms = float("nan")
        swapped = ""
        if p3.is_file() and p4.is_file():
            d3 = pd.read_csv(p3)
            d4 = pd.read_csv(p4)
            n3 = int(d3["comp_ids"].iloc[0].count(";") + 1) if "comp_ids" in d3.columns and pd.notna(d3["comp_ids"].iloc[0]) and str(d3["comp_ids"].iloc[0]).strip() else 0
            n4 = int(d4["comp_ids"].iloc[0].count(";") + 1) if "comp_ids" in d4.columns and pd.notna(d4["comp_ids"].iloc[0]) and str(d4["comp_ids"].iloc[0]).strip() else 0
            if "comp_ids" in d3.columns and "comp_ids" in d4.columns:
                s3 = set(str(d3["comp_ids"].iloc[0]).split(";")) - {""}
                s4 = set(str(d4["comp_ids"].iloc[0]).split(";")) - {""}
                swapped = ";".join(sorted((s3 | s4) - (s3 & s4)))
            dmag_c = med_delta_mmag(d3, d4, "mag_calib")
            dmag_f = med_delta_mmag(d3, d4, "mag_calib_final")
            r3 = demeaned_rms(d3["mag_calib"]) if "mag_calib" in d3.columns else float("nan")
            r4 = demeaned_rms(d4["mag_calib"]) if "mag_calib" in d4.columns else float("nan")
            if np.isfinite(r3) and np.isfinite(r4):
                drms = r4 - r3
            files.append("aperture_lc")
            if (e3 / f"lightcurve_{tid}_psf.csv").is_file() or (e4 / f"lightcurve_{tid}_psf.csv").is_file():
                files.append("psf_lc")
            aav3 = phot(ERA03) / "aavso" / f"aavso_{tid}.txt"
            aav4 = phot(ERA04) / "aavso" / f"aavso_{tid}.txt"
            if aav3.is_file() or aav4.is_file():
                files.append("aavso")
        elif p3.is_file() and not p4.is_file():
            files.append("aperture_lc")
            d3 = pd.read_csv(p3)
            n3 = int(d3["comp_ids"].iloc[0].count(";") + 1) if "comp_ids" in d3.columns and pd.notna(d3["comp_ids"].iloc[0]) and str(d3["comp_ids"].iloc[0]).strip() else 0
        elif p4.is_file() and not p3.is_file():
            files.append("aperture_lc")
            d4 = pd.read_csv(p4)
            n4 = int(d4["comp_ids"].iloc[0].count(";") + 1) if "comp_ids" in d4.columns and pd.notna(d4["comp_ids"].iloc[0]) and str(d4["comp_ids"].iloc[0]).strip() else 0

        keep = NAMED_KEEP.get(tid)
        old = v4_map.get(tid, "")
        tags = []
        if keep:
            tags.append(keep)
        elif "EDGE" in old and tid in NAMED_KEEP:
            tags.append("EDGE")
        if "CT-REF" in old and "CT-REF" not in tags:
            tags.append("CT-REF")
        # EDGE is only the C6-3c out_of_frame set (NAMED_KEEP), not every missing era04 LC.
        if p3.is_file() and p4.is_file() and np.isfinite(dmag_c):
            tags.append("APERTURE-01 f=1.35")
        if tid == GH:
            if "CROWDING" not in tags:
                tags.append("CROWDING")
        if not tags:
            tags.append("UNNAMED")
            unnamed.append(tid)
        cause = "[" + ";".join(tags) + "]"
        if np.isfinite(dmag_c):
            residuals[tid] = round(float(dmag_c), 4)
            cause = f"{cause} residual={dmag_c:.3f}mmag"
        if tid == GH:
            cause = f"{cause} D11-1=docs/VYVAR_LIMITATIONS.md#dilution-crowding-g-proxy"
        rows.append(
            {
                "target": tid,
                "files_changed": ";".join(files),
                "n_comps_era03": n3,
                "n_comps_era04": n4,
                "median_dmag_mmag": dmag_c,
                "dmag_final_mmag": dmag_f,
                "dRMS_mmag": drms,
                "cause": cause,
                "ids_swapped": swapped,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "a1c_era03_era04_ledger_v5.csv", index=False)
    out = {
        "n_targets_union": int(len(df)),
        "n_era03": len(ids3),
        "n_era04": len(ids4),
        "n_unnamed": len(unnamed),
        "unnamed_ids": unnamed,
        "tag_counts": df["cause"].str.extract(r"\[([^\]]+)\]")[0].value_counts().to_dict(),
        "P-A1_residual_mmag": {
            "BO": residuals.get(BO),
            "FW": residuals.get(FW),
            "GH": residuals.get(GH),
        },
        "lock_ok": len(unnamed) == 0,
    }
    (OUT / "a1c_ledger_v5.json").write_text(json.dumps(out, indent=2, default=str), encoding="ascii")
    print(json.dumps(out, indent=2, default=str)[:2500])
    return 0 if out["lock_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
