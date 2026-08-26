# -*- coding: ascii -*-
"""C6-3c post-photometry: X2 CT-REF check + ledger v2. Does not lock."""
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
CENSUS = OUT / "c63b_lc_census.csv"
BO = "1498613634033133184"
FW = "1497343732462852864"
GH = "1498804639818507904"
NAMED = {
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


def lc(root: Path, cid: str) -> pd.DataFrame:
    return pd.read_csv(phot(root) / "lightcurves" / f"lightcurve_{cid}.csv")


def med_delta_mmag(a: pd.DataFrame, b: pd.DataFrame, col: str) -> float:
    m = a.merge(b, on="source_file", suffixes=("_e3", "_e4"))
    d = pd.to_numeric(m[f"{col}_e4"], errors="coerce") - pd.to_numeric(
        m[f"{col}_e3"], errors="coerce"
    )
    return float(d.median() * 1000.0)


def main() -> int:
    census = pd.read_csv(CENSUS, dtype={"catalog_id": str})
    ct_ids = [str(x) for x in census.loc[census["tag"] == "POOL-SNR", "catalog_id"]]
    stars = {"BO": BO, "FW": FW, "GH": GH}
    star_out = {}
    for name, cid in stars.items():
        d3, d4 = lc(ERA03, cid), lc(ERA04, cid)
        rec = {
            "catalog_id": cid,
            "n3": int(len(d3)),
            "n4": int(len(d4)),
            "dmag_inst_mmag": med_delta_mmag(d3, d4, "mag_inst"),
            "dmag_calib_mmag": med_delta_mmag(d3, d4, "mag_calib"),
            "dmag_final_mmag": med_delta_mmag(d3, d4, "mag_calib_final"),
            "ct_bp_rp_comp_med_e3": float(pd.to_numeric(d3["ct_bp_rp_comp_med"], errors="coerce").iloc[0]),
            "ct_bp_rp_comp_med_e4": float(pd.to_numeric(d4["ct_bp_rp_comp_med"], errors="coerce").iloc[0]),
            "ct_n_comp_e3": int(pd.to_numeric(d3["ct_n_comp"], errors="coerce").iloc[0]),
            "ct_n_comp_e4": int(pd.to_numeric(d4["ct_n_comp"], errors="coerce").iloc[0]),
            "ct_correction_e3": float(pd.to_numeric(d3["ct_correction"], errors="coerce").iloc[0]),
            "ct_correction_e4": float(pd.to_numeric(d4["ct_correction"], errors="coerce").iloc[0]),
            "aperture_r_e3": float(pd.to_numeric(d3["aperture_r_px"], errors="coerce").median()),
            "aperture_r_e4": float(pd.to_numeric(d4["aperture_r_px"], errors="coerce").median()),
        }
        rec["dct_mmag"] = (rec["ct_correction_e4"] - rec["ct_correction_e3"]) * 1000.0
        rec["ct_ref_collapsed"] = abs(rec["dmag_final_mmag"] - rec["dmag_calib_mmag"]) < 5.0
        star_out[name] = rec

    ct_rows = []
    n_collapse = 0
    for cid in ct_ids:
        p4 = phot(ERA04) / "lightcurves" / f"lightcurve_{cid}.csv"
        p3 = phot(ERA03) / "lightcurves" / f"lightcurve_{cid}.csv"
        if not p4.is_file() or not p3.is_file():
            ct_rows.append({"catalog_id": cid, "status": "missing"})
            continue
        d3, d4 = pd.read_csv(p3), pd.read_csv(p4)
        rec = {
            "catalog_id": cid,
            "dmag_calib_mmag": med_delta_mmag(d3, d4, "mag_calib"),
            "dmag_final_mmag": med_delta_mmag(d3, d4, "mag_calib_final"),
            "ct_bp_rp_comp_med_e3": float(pd.to_numeric(d3["ct_bp_rp_comp_med"], errors="coerce").iloc[0]),
            "ct_bp_rp_comp_med_e4": float(pd.to_numeric(d4["ct_bp_rp_comp_med"], errors="coerce").iloc[0]),
            "ct_n_comp_e3": int(pd.to_numeric(d3["ct_n_comp"], errors="coerce").iloc[0]),
            "ct_n_comp_e4": int(pd.to_numeric(d4["ct_n_comp"], errors="coerce").iloc[0]),
            "ct_correction_e3": float(pd.to_numeric(d3["ct_correction"], errors="coerce").iloc[0]),
            "ct_correction_e4": float(pd.to_numeric(d4["ct_correction"], errors="coerce").iloc[0]),
        }
        rec["dct_mmag"] = (rec["ct_correction_e4"] - rec["ct_correction_e3"]) * 1000.0
        rec["collapsed"] = abs(rec["dct_mmag"]) < 20.0
        if rec["collapsed"]:
            n_collapse += 1
        ct_rows.append(rec)
    pd.DataFrame(ct_rows).to_csv(OUT / "c63c_x2_ct_collapse.csv", index=False)

    e3 = phot(ERA03) / "lightcurves"
    e4 = phot(ERA04) / "lightcurves"
    ids3 = {p.stem.replace("lightcurve_", "") for p in e3.glob("lightcurve_*.csv") if not p.stem.endswith(("_psf", "_adaptive"))}
    ids4 = {p.stem.replace("lightcurve_", "") for p in e4.glob("lightcurve_*.csv") if not p.stem.endswith(("_psf", "_adaptive"))}
    ledger_rows = []
    unnamed = []
    for tid in sorted(ids3 | ids4):
        p3 = e3 / f"lightcurve_{tid}.csv"
        p4 = e4 / f"lightcurve_{tid}.csv"
        tag = NAMED.get(tid, "")
        extra = []
        dmag_c = dmag_f = float("nan")
        ct_n4 = ct_med4 = float("nan")
        if p3.is_file() and p4.is_file():
            d3, d4 = pd.read_csv(p3), pd.read_csv(p4)
            dmag_c = med_delta_mmag(d3, d4, "mag_calib")
            dmag_f = med_delta_mmag(d3, d4, "mag_calib_final")
            ct_n4 = float(pd.to_numeric(d4["ct_n_comp"], errors="coerce").iloc[0])
            ct_med4 = float(pd.to_numeric(d4["ct_bp_rp_comp_med"], errors="coerce").iloc[0])
            dct = (
                float(pd.to_numeric(d4["ct_correction"], errors="coerce").iloc[0])
                - float(pd.to_numeric(d3["ct_correction"], errors="coerce").iloc[0])
            ) * 1000.0
            if abs(dmag_f - dmag_c) < 5.0 and tid in ct_ids:
                extra.append("CT-REF")
        elif p3.is_file() and not p4.is_file():
            extra.append("EDGE")
        if not tag:
            if "EDGE" in extra:
                tag = "EDGE"
            elif tid in ct_ids:
                tag = "CT-REF"
            else:
                tag = "UNNAMED"
                unnamed.append(tid)
        elif tid in ct_ids and tag not in ("EDGE",):
            tag = f"{tag};CT-REF" if tag != "CT-REF" else tag
        ledger_rows.append(
            {
                "target": tid,
                "in_era03": p3.is_file(),
                "in_era04": p4.is_file(),
                "dmag_calib_mmag": dmag_c,
                "dmag_final_mmag": dmag_f,
                "ct_n_comp_e4": ct_n4,
                "ct_bp_rp_comp_med_e4": ct_med4,
                "cause": f"[{tag}]",
            }
        )
    pd.DataFrame(ledger_rows).to_csv(OUT / "c63c_era03_era04_ledger_v2.csv", index=False)

    bo_ok = abs(star_out["BO"]["dmag_final_mmag"] - star_out["BO"]["dmag_calib_mmag"]) < 5.0
    bo_pred = abs(star_out["BO"]["dmag_final_mmag"] - 2.8) < 2.0
    out = {
        "stars": star_out,
        "ct_pool_snr_n": len(ct_ids),
        "ct_collapse_n": n_collapse,
        "bo_final_equals_calib": bo_ok,
        "bo_final_near_plus_2p8": bo_pred,
        "n_era03_lc": len(ids3),
        "n_era04_lc": len(ids4),
        "n_unnamed": len(unnamed),
        "unnamed_ids": unnamed,
        "lock": False,
        "lock_reason": "mag_calib residuals remain UNNAMED (not WCS-APERTURE); X1d equal not a regression",
    }
    (OUT / "c63c_x2_rerun.json").write_text(json.dumps(out, indent=2), encoding="ascii")
    print(
        "BO final", star_out["BO"]["dmag_final_mmag"],
        "calib", star_out["BO"]["dmag_calib_mmag"],
        "ct_med", star_out["BO"]["ct_bp_rp_comp_med_e4"],
        "ct_n", star_out["BO"]["ct_n_comp_e4"],
        "collapse", n_collapse, "/", len(ct_ids),
        "unnamed", len(unnamed),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
