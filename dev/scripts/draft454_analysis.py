#!/usr/bin/env python3
"""Draft 454 analysis harness (CURSOR_TASK DRAFT-454)."""
from __future__ import annotations

import csv
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src_py"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(REPO / "dev") not in sys.path:
    sys.path.insert(0, str(REPO / "dev"))

OUT_DIR = REPO / "dev/results/context/session_20260728_draft454"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DRAFTS = {
    "452": REPO / "Archive/Drafts/draft_000452",
    "453": REPO / "Archive/Drafts/draft_000453",
    "454": REPO / "Archive/Drafts/draft_000454",
}
CAL_SUB = Path("calibrated/lights/NoFilter_60_2")


def _parse_hms(s: str) -> int:
    h, m, sec = (int(x) for x in s.split(":"))
    return h * 3600 + m * 60 + sec


def _read_infolog(draft: Path) -> tuple[Path, str]:
    logs = sorted(draft.glob("infolog_*.txt"))
    if not logs:
        raise FileNotFoundError(draft)
    # Prefer durable session log (largest / session marker).
    session = [p for p in logs if "session: durable" in p.read_text(encoding="utf-8", errors="replace")[:500]]
    path = session[0] if session else logs[-1]
    return path, path.read_text(encoding="utf-8", errors="replace")


def _milestones(text: str) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    in_ms = False
    for ln in text.splitlines():
        if "milestones (never evicted)" in ln:
            in_ms = True
            continue
        if in_ms:
            if not ln.strip():
                break
            if ln.startswith("#"):
                break
            m = re.match(r"(\d{2}:\d{2}:\d{2})\s+(.*)", ln.strip())
            if m:
                out.append((_parse_hms(m.group(1)), m.group(2)))
    if out:
        return out
    for ln in text.splitlines():
        m = re.match(r"(\d{2}:\d{2}:\d{2})\s+\[PHASE\]", ln.strip())
        if m:
            out.append((_parse_hms(m.group(1)), ln.split(None, 1)[1].strip()))
        if m and len(out) >= 12:
            break
    return out


def _grep_line(text: str, pat: str) -> str:
    rx = re.compile(pat)
    for ln in text.splitlines():
        if rx.search(ln):
            return ln.strip()
    return ""


def _phase_table_454(text: str) -> list[dict]:
    ms = _milestones(text)
    rows: list[dict] = []
    for i, (t0, msg) in enumerate(ms):
        t1 = ms[i + 1][0] if i + 1 < len(ms) else None
        dur = (t1 - t0) if t1 is not None else None
        rows.append({"phase": msg, "start_s": t0, "duration_s": dur})
    # preprocess wall from PREPROCESS start to QC summary
    pre_start = _grep_line(text, r"\[PREPROCESS\] start")
    pre_end = _grep_line(text, r"QC in-place: \d+ ok")
    if pre_start and pre_end:
        t0 = re.match(r"(\d{2}:\d{2}:\d{2})", pre_start).group(1)
        t1 = re.match(r"(\d{2}:\d{2}:\d{2})", pre_end).group(1)
        pre_s = _parse_hms(t1) - _parse_hms(t0)
        n_ok = re.search(r"QC in-place: (\d+) ok", pre_end)
        n = int(n_ok.group(1)) if n_ok else 150
        rows.append(
            {
                "phase": "preprocess_qc_in_place_wall",
                "duration_s": pre_s,
                "per_frame_s": pre_s / 150.0,
                "n_frames": 150,
            }
        )
    # first artifact: first calibration write or [1/150] Calibrating
    run_start = ms[0][0] if ms else None
    first_cal = _grep_line(text, r"Kalibracia - zapis OK|Calibrating BO_CVn_Light_001")
    if run_start is not None and first_cal:
        t_first = re.match(r"(\d{2}:\d{2}:\d{2})", first_cal).group(1)
        gap = _parse_hms(t_first) - run_start
        rows.append({"phase": "run_start_to_first_artifact_s", "duration_s": gap})
    # wall: first/last timestamp in body
    ts_all = [int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
              for m in re.finditer(r"^(\d{2}):(\d{2}):(\d{2})\s+", text, re.M)]
    if ts_all:
        rows.append({"phase": "infolog_wall_clock_s", "duration_s": max(ts_all) - min(ts_all)})
        rows.append({"phase": "infolog_wall_clock_min", "duration_s": (max(ts_all) - min(ts_all)) / 60.0})
    return rows


def _acceptance(draft: Path, log_text: str) -> dict:
    ps = draft / "platesolve/NoFilter_60_2"
    ms = pd.read_csv(ps / "masterstars_full_match.csv")
    dao_only = int((ms["source_type"] == "DAO_ONLY").sum())
    mf = ps / "MASTERSTAR.fits"
    bg_std = dao_threshold = float("nan")
    if mf.is_file():
        with fits.open(mf) as hd:
            d = hd[0].data.astype("float32")
        _, _, std = sigma_clipped_stats(d - float(np.nanmedian(d)), sigma=3, maxiters=3)
        bg_std = float(std)
        dao_threshold = 2.1 * bg_std
    act = pd.read_csv(ps / "photometry/active_targets.csv")
    vt = pd.read_csv(ps / "variable_targets.csv")
    exo_n = int(vt["exo_host_obj_id"].notna().sum()) if "exo_host_obj_id" in vt.columns else 0
    dao_pass1 = None
    m = re.search(r"\[DAO pass 1\]\s*(\d+)\s*detections", log_text)
    if m:
        dao_pass1 = int(m.group(1))
    return {
        "dao_pass1": dao_pass1,
        "masterstars_rows": len(ms),
        "dao_only_frac": dao_only / len(ms) if len(ms) else float("nan"),
        "bg_std": bg_std,
        "dao_threshold": dao_threshold,
        "active_targets": len(act),
        "vt_rows": len(vt),
        "exo_promoted_vt": exo_n,
        "n_lightcurves": len(list((ps / "photometry/lightcurves").glob("lightcurve_*.csv"))),
    }


def _compare_calibrated(a: Path, b: Path) -> dict:
    da, db = a / CAL_SUB, b / CAL_SUB
    fa = sorted(da.glob("BO_CVn_Light_*.fits"))
    fb = {p.name: p for p in db.glob("BO_CVn_Light_*.fits")}
    rows = []
    n_zero = 0
    for pa in fa:
        pb = fb.get(pa.name)
        if pb is None:
            rows.append({"frame": pa.name, "max_abs_diff": None, "note": "missing_in_b"})
            continue
        da_a = fits.getdata(pa).astype(np.float64)
        da_b = fits.getdata(pb).astype(np.float64)
        diff = float(np.max(np.abs(da_a - da_b)))
        if diff == 0.0:
            n_zero += 1
        rows.append({"frame": pa.name, "max_abs_diff": diff})
    csv_path = OUT_DIR / "calibrated_452_vs_454.csv"
    with csv_path.open("w", newline="", encoding="ascii") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "max_abs_diff", "note"])
        w.writeheader()
        w.writerows(rows)
    diffs = [r["max_abs_diff"] for r in rows if r.get("max_abs_diff") is not None]
    return {
        "n_compared": len(diffs),
        "n_exact_zero": n_zero,
        "max_diff": max(diffs) if diffs else float("nan"),
        "nonzero_frames": [r for r in rows if r.get("max_abs_diff") not in (None, 0.0)],
        "csv": str(csv_path.relative_to(REPO)),
    }


def _vy_skyp2p(draft: Path) -> list[dict]:
    cal = draft / CAL_SUB
    rows = []
    for fp in sorted(cal.glob("BO_CVn_Light_*.fits")):
        with fits.open(fp) as hd:
            p2p = hd[0].header.get("VYSKYP2P")
            order = hd[0].header.get("VYSKYORD")
        rows.append({"frame": fp.name, "VYSKYP2P": float(p2p) if p2p is not None else float("nan"), "VYSKYORD": order})
    csv_path = OUT_DIR / "draft454_vy_skyp2p.csv"
    with csv_path.open("w", newline="", encoding="ascii") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "VYSKYP2P", "VYSKYORD"])
        w.writeheader()
        w.writerows(rows)
    return rows


def main() -> int:
    t0 = time.time()
    log_path, log_text = _read_infolog(DRAFTS["454"])
    phases = _phase_table_454(log_text)
    proofs = {
        "SITE": _grep_line(log_text, r"\[SITE\].*"),
        "INV-PREP-01": _grep_line(log_text, r"INV-PREP-01 Preprocess gradient guard"),
        "INV-MS-01": _grep_line(log_text, r"INV-MS-01 MASTERSTAR purity guard"),
        "VSX-GAIA XM": _grep_line(log_text, r"VSX-GAIA XM:"),
        "FAZA 0 funnel": _grep_line(log_text, r"FAZA 0 funnel:"),
        "EXO TARGET": _grep_line(log_text, r"\[EXO TARGET\]"),
        "DAO pass 1": _grep_line(log_text, r"\[DAO pass 1\]"),
        "DAO pass 2": _grep_line(log_text, r"\[DAO pass 2\]"),
    }
    acc454 = _acceptance(DRAFTS["454"], log_text)
    acc452 = _acceptance(DRAFTS["452"], "")
    skyp2p = _vy_skyp2p(DRAFTS["454"])
    cal_cmp = _compare_calibrated(DRAFTS["452"], DRAFTS["454"])

    # export phase csv
    phase_csv = OUT_DIR / "ui_startup_phases_454.csv"
    with phase_csv.open("w", newline="", encoding="ascii") as f:
        if phases:
            w = csv.DictWriter(f, fieldnames=sorted({k for p in phases for k in p}))
            w.writeheader()
            w.writerows(phases)

    summary = {
        "infolog_used": str(log_path.relative_to(REPO)),
        "phases": phases,
        "proofs": proofs,
        "acceptance_454": acc454,
        "acceptance_452_context": acc452,
        "calibrated_452_vs_454": cal_cmp,
        "vy_skyp2p_454_frames_001_010": skyp2p[:10],
        "vy_skyp2p_454_stats": {
            "min": float(np.nanmin([r["VYSKYP2P"] for r in skyp2p])),
            "median": float(np.nanmedian([r["VYSKYP2P"] for r in skyp2p])),
            "max": float(np.nanmax([r["VYSKYP2P"] for r in skyp2p])),
            "frame001": skyp2p[0]["VYSKYP2P"] if skyp2p else float("nan"),
            "frame002": skyp2p[1]["VYSKYP2P"] if len(skyp2p) > 1 else float("nan"),
        },
        "runtime_s": time.time() - t0,
    }
    out_json = OUT_DIR / "draft454_analysis.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
