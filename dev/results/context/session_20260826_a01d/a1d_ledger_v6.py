# -*- coding: ascii -*-
"""Ledger v6: era03 vs APERTURE-01d era04 (f=1.35, annulus 2.7/5.2).

Allowed tags: APERTURE-01, CROWDING, EDGE-ANNULUS, POOL-STARVE,
GAINED, CT-REF, EDGE, FRAME-29, D3-D5, NAME-FIX, C3-K5.
POOL-STARVE and EDGE-ANNULUS are limitations, not lock blockers.
STOP if any target is untagged.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\ASTRO\python\VYVAR")
ERA03 = ROOT / "Archive" / "Drafts" / "draft_000516_snapshot_era03_20260820"
ERA04 = ROOT / "Archive" / "Drafts" / "draft_000516_snapshot_era04_20260826"
SETUP = "NoFilter_60_2"
OUT = Path(__file__).resolve().parent
V4 = ROOT / "dev" / "results" / "context" / "session_20260826_c6" / "c63_era03_era04_ledger_v4.csv"
INFOLOG = OUT / "a1d_infolog"
BO = "1498613634033133184"
FW = "1497343732462852864"
GH = "1498804639818507904"
CSS = "1498842882207281152"
HAT188 = "1499842372636900992"
HAT145 = "1498752516095473664"
CROWD_IDS = {
    CSS: "neighbor 1498843633825520000 at 10.63 px (frame 014); VSX not force-injected",
    HAT188: "neighbor 1499842132118731648 at 18.30 px (frame 014); VSX not force-injected",
}
STARVE_NSURV = {
    "1497227287309482624": {"n_survivors": 2, "n_min": 3, "comp": "1500467303261764096", "reason": "rms_violation"},
    "1497245497969274240": {"n_survivors": 2, "n_min": 3, "comp": "1500467303261764096", "reason": "rms_violation"},
    "1498425548825498112": {"n_survivors": 2, "n_min": 3, "comp": "1500467303261764096", "reason": "rms_violation"},
}
V5_BO = 49.4575
V5_FW = 28.213
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
EDGE_IDS = {
    "1485560025830226432",
    "1496037650087948160",
    "1496733984545821696",
    "1497491273179203456",
}
ALLOWED = {
    "APERTURE-01",
    "CROWDING",
    "EDGE-ANNULUS",
    "POOL-STARVE",
    "GAINED",
    "CT-REF",
    "EDGE",
    "FRAME-29",
    "D3-D5",
    "NAME-FIX",
    "C3-K5",
}


def phot(root: Path) -> Path:
    return root / "platesolve" / SETUP / "photometry"


def n_epochs(df: pd.DataFrame) -> int:
    if "mag_calib" not in df.columns:
        return 0
    return int(pd.to_numeric(df["mag_calib"], errors="coerce").notna().sum())


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


def parse_infolog(infodir: Path) -> dict:
    """PIN-DROP and n_survivors lines from the recut infolog."""
    pin_drops: dict[str, list[tuple[str, str]]] = {}
    starve: dict[str, dict] = {}
    pat_drop = re.compile(
        r"\[PIN-DROP\] target=(\d+) comp=(\d+) reason=(\S+)"
    )
    pat_surv = re.compile(
        r"target=(\d+) n_survivors=(\d+) n_min=(\d+) drops=(.+)$"
    )
    files = sorted(infodir.glob("infolog_*.txt")) if infodir.is_dir() else []
    text = ""
    for p in files:
        text += p.read_text(encoding="ascii", errors="replace") + "\n"
    for m in pat_drop.finditer(text):
        tid, cid, reason = m.group(1), m.group(2), m.group(3)
        pin_drops.setdefault(tid, []).append((cid, reason.split("(")[0]))
    for m in pat_surv.finditer(text):
        tid = m.group(1)
        starve[tid] = {
            "n_survivors": int(m.group(2)),
            "n_min": int(m.group(3)),
            "drops_raw": m.group(4).strip(),
        }
    return {"pin_drops": pin_drops, "starve": starve, "n_infolog_files": len(files)}


def gained_frame_reasons(d3: pd.DataFrame, d4: pd.DataFrame) -> list[str]:
    """Per-frame reason for epochs that appear in era04 but not era03."""
    a = pd.DataFrame(
        {
            "k": d3["source_file"].map(lambda s: Path(str(s)).name),
            "m3": pd.to_numeric(d3["mag_calib"], errors="coerce"),
            "f3": d3["flag"].astype(str) if "flag" in d3.columns else "",
        }
    )
    b = pd.DataFrame(
        {
            "k": d4["source_file"].map(lambda s: Path(str(s)).name),
            "m4": pd.to_numeric(d4["mag_calib"], errors="coerce"),
            "f4": d4["flag"].astype(str) if "flag" in d4.columns else "",
        }
    )
    m = a.merge(b, on="k", how="outer")
    reasons = []
    for _, row in m.iterrows():
        ok3 = np.isfinite(row["m3"]) if pd.notna(row["m3"]) else False
        ok4 = np.isfinite(row["m4"]) if pd.notna(row["m4"]) else False
        if ok4 and not ok3:
            reasons.append(
                f"{row['k']}:era03_flag={row['f3']}|era04_flag={row['f4']}"
            )
    return reasons


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

    info = parse_infolog(INFOLOG)
    rows = []
    unnamed = []
    residuals = {}
    extra = {}
    for tid in sorted(ids3 | ids4):
        p3 = e3 / f"lightcurve_{tid}.csv"
        p4 = e4 / f"lightcurve_{tid}.csv"
        files = []
        n3 = n4 = 0
        ep3 = ep4 = 0
        dmag_c = dmag_f = drms = float("nan")
        swapped = ""
        d3 = d4 = None
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
            ep3, ep4 = n_epochs(d3), n_epochs(d4)
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
            ep3 = n_epochs(d3)
        elif p4.is_file() and not p3.is_file():
            files.append("aperture_lc")
            d4 = pd.read_csv(p4)
            n4 = int(d4["comp_ids"].iloc[0].count(";") + 1) if "comp_ids" in d4.columns and pd.notna(d4["comp_ids"].iloc[0]) and str(d4["comp_ids"].iloc[0]).strip() else 0
            ep4 = n_epochs(d4)

        keep = NAMED_KEEP.get(tid)
        old = v4_map.get(tid, "")
        tags = []
        notes = []
        if keep:
            tags.append(keep)
        if "CT-REF" in old and "CT-REF" not in tags:
            tags.append("CT-REF")
        if tid in EDGE_IDS and "EDGE-ANNULUS" not in tags:
            tags.append("EDGE-ANNULUS")
        if p3.is_file() and p4.is_file() and np.isfinite(dmag_c):
            if "APERTURE-01" not in tags:
                tags.append("APERTURE-01")
        if tid == GH and "CROWDING" not in tags:
            tags.append("CROWDING")
        starve_rec = info["starve"].get(tid) or STARVE_NSURV.get(tid)
        drops = info["pin_drops"].get(tid, [])
        if (not p4.is_file()) and (starve_rec or drops or tid in STARVE_NSURV):
            tags.append("POOL-STARVE")
            pred = []
            if starve_rec:
                pred.append(
                    f"n_survivors={starve_rec['n_survivors']}<n_min={starve_rec['n_min']}"
                )
                if starve_rec.get("comp"):
                    pred.append(f"{starve_rec['comp']}:{starve_rec.get('reason', '')}")
            for cid, reason in drops:
                item = f"{cid}:{reason}"
                if item not in pred:
                    pred.append(item)
            notes.append("preds=" + ",".join(pred))
        if tid in CROWD_IDS and "CROWDING" not in tags:
            tags.append("CROWDING")
            notes.append("crowd=" + CROWD_IDS[tid])
        if d3 is not None and d4 is not None and ep3 == 0 and ep4 > 0:
            tags.append("GAINED")
            fr = gained_frame_reasons(d3, d4)
            notes.append("gained_n=%d" % len(fr))
            extra[tid] = {"gained_frames": fr[:80], "gained_n": len(fr)}
        if not tags:
            tags.append("UNNAMED")
            unnamed.append(tid)
        bad = [t.split()[0] for t in tags if t.split()[0] not in ALLOWED and t != "UNNAMED"]
        cause = "[" + ";".join(tags) + "]"
        if notes:
            cause = cause + " " + " ".join(notes)
        if np.isfinite(dmag_c):
            residuals[tid] = round(float(dmag_c), 4)
            cause = f"{cause} residual={dmag_c:.3f}mmag"
        if tid == GH:
            cause = f"{cause} D11-1=docs/VYVAR_LIMITATIONS.md#dilution-crowding-g-proxy"
        if "POOL-STARVE" in tags:
            cause = f"{cause} LIMITATION=docs/VYVAR_LIMITATIONS.md#pool-starve"
        if "EDGE-ANNULUS" in tags:
            cause = f"{cause} LIMITATION=docs/VYVAR_LIMITATIONS.md#edge-annulus"
        rows.append(
            {
                "target": tid,
                "files_changed": ";".join(files),
                "n_comps_era03": n3,
                "n_comps_era04": n4,
                "n_epochs_era03": ep3,
                "n_epochs_era04": ep4,
                "median_dmag_mmag": dmag_c,
                "dmag_final_mmag": dmag_f,
                "dRMS_mmag": drms,
                "cause": cause,
                "ids_swapped": swapped,
                "bad_tags": ";".join(bad),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "a1d_era03_era04_ledger_v6.csv", index=False)
    bo_res = residuals.get(BO)
    fw_res = residuals.get(FW)
    bo_move = abs(bo_res - V5_BO) if bo_res is not None else float("nan")
    fw_move = abs(fw_res - V5_FW) if fw_res is not None else float("nan")
    css_row = df.loc[df["target"] == CSS]
    hat_row = df.loc[df["target"] == HAT188]
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
        "v5_residual_mmag": {"BO": V5_BO, "FW": V5_FW},
        "dlevel_vs_v5_mmag": {"BO": None if bo_res is None else round(bo_res - V5_BO, 4), "FW": None if fw_res is None else round(fw_res - V5_FW, 4)},
        "dlevel_vs_v5_ok": bool(
            math_ok(bo_move, 5.0) and math_ok(fw_move, 5.0)
        ),
        "css_epochs": None if css_row.empty else int(css_row["n_epochs_era04"].iloc[0]),
        "css_era03_epochs": None if css_row.empty else int(css_row["n_epochs_era03"].iloc[0]),
        "hat188_epochs": None if hat_row.empty else int(hat_row["n_epochs_era04"].iloc[0]),
        "hat188_era03_epochs": None if hat_row.empty else int(hat_row["n_epochs_era03"].iloc[0]),
        "infolog": {"n_files": info["n_infolog_files"], "n_starve": len(info["starve"])},
        "gained": extra,
        "lock_ok": len(unnamed) == 0,
    }
    (OUT / "a1d_ledger_v6.json").write_text(json.dumps(out, indent=2, default=str), encoding="ascii")
    print(json.dumps({k: out[k] for k in out if k != "gained"}, indent=2, default=str)[:3000])
    return 0 if out["lock_ok"] else 1


def math_ok(val: float, lim: float) -> bool:
    return bool(np.isfinite(val) and val < lim)


if __name__ == "__main__":
    raise SystemExit(main())
