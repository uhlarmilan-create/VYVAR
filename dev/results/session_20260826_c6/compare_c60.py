# -*- coding: ascii -*-
"""C6-0 60-row R2 (T3 HEAD) vs R1' (c592ecf + A files)."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\ASTRO\python\VYVAR")
sys.path.insert(0, str(ROOT / "src_py"))
from masterstar_gaia_accounting import _norm_cid  # noqa: E402

SESSION = Path(__file__).resolve().parent
T3 = ROOT / "dev" / "results" / "session_20260825_sel_ghost_01_b3"
LIVE = ROOT / "Archive" / "Drafts" / "draft_000516"
SETUP = "NoFilter_60_2"
R1P = SESSION / "t3_r1p"
R2 = T3 / "t3_r2"
PRED_D3_COMP = "1498964240802993408"
PRED_D3_TARGET = "1497284015237511808"
PRED_CSS = "1497169940906156032"
S3_ONLY = [
    "1485911972629595392",
    "1497137402233837952",
    "1497195367112531712",
    "1502034042907960704",
]
C592_ONLY = [
    "1485912110068525824",
    "1504304603139151872",
]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def phot_dir(root: Path) -> Path:
    return root / "platesolve" / SETUP / "photometry"


def lc60_ids() -> list[str]:
    at = pd.read_csv(phot_dir(LIVE) / "active_targets.csv", dtype={"catalog_id": str})
    skip = at["skip_photometry"]
    if skip.dtype == bool:
        keep = ~skip
    else:
        keep = skip.astype(str).str.strip().str.lower().isin(["false", "0", ""])
    ids = sorted({_norm_cid(v) for v, k in zip(at["catalog_id"].tolist(), keep.tolist()) if k and _norm_cid(v)})
    return ids


def ensemble_map(root: Path) -> dict[str, set[str]]:
    p = phot_dir(root) / "comparison_stars_per_target.csv"
    if not p.is_file():
        return {}
    df = pd.read_csv(p, dtype=str)
    out: dict[str, set[str]] = {}
    for _, row in df.iterrows():
        t = _norm_cid(row.get("target_catalog_id"))
        c = _norm_cid(row.get("catalog_id"))
        if t and c:
            out.setdefault(t, set()).add(c)
    return out


def summary_map(root: Path) -> dict[str, dict]:
    p = phot_dir(root) / "photometry_summary.csv"
    if not p.is_file():
        return {}
    df = pd.read_csv(p, dtype={"catalog_id": str})
    out = {}
    for _, row in df.iterrows():
        cid = _norm_cid(row.get("catalog_id"))
        if not cid:
            continue
        rec = {}
        for col in ("lc_rms", "lc_rms_ooe"):
            if col in df.columns:
                rec[col] = float(pd.to_numeric(row[col], errors="coerce"))
        out[cid] = rec
    return out


def mag_series(root: Path, tid: str) -> pd.Series | None:
    p = phot_dir(root) / "lightcurves" / f"lightcurve_{tid}.csv"
    if not p.is_file():
        return None
    df = pd.read_csv(p)
    for col in ("mag_calib", "mag", "dmag"):
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
    return None


def report_files(root: Path, kind: str, tid: str) -> list[Path]:
    d = phot_dir(root) / "lightcurves_reports" / kind
    if not d.is_dir():
        return []
    return [p for p in d.iterdir() if p.is_file() and tid in p.name]


def sha_pair(a: Path | None, b: Path | None) -> str:
    if a is None or b is None or not a.is_file() or not b.is_file():
        return "N"
    return "Y" if sha256_file(a) == sha256_file(b) else "N"


def lc_sha(root: Path, tid: str) -> str | None:
    p = phot_dir(root) / "lightcurves" / f"lightcurve_{tid}.csv"
    if p.is_file():
        return sha256_file(p)
    return None


def ms_cause(tid: str, ens_a: set[str], ens_b: set[str], ms_a: Path, ms_b: Path) -> str:
    swapped = sorted(ens_a.symmetric_difference(ens_b))
    if not swapped:
        return "none (ensemble identical)"
    reasons = []
    delta_ids = set(S3_ONLY) | set(C592_ONLY)
    hit_delta = [s for s in swapped if s in delta_ids]
    if hit_delta:
        reasons.append("catalog_id delta " + ",".join(hit_delta))
    if PRED_D3_COMP in swapped:
        reasons.append("D3/D5 SNR " + PRED_D3_COMP)
    if tid == PRED_CSS or PRED_CSS in swapped:
        reasons.append("NAME-FIX CSS_J134925 " + PRED_CSS)
    if ms_b.is_file() and "snr" in pd.read_csv(ms_b, nrows=0).columns:
        msb = pd.read_csv(ms_b, dtype={"catalog_id": str}, low_memory=False)
        msb["_cid"] = msb["catalog_id"].map(_norm_cid)
        by = msb.drop_duplicates("_cid").set_index("_cid")
        d3_bits = []
        for s in swapped:
            if s not in by.index:
                d3_bits.append(f"{s} absent_in_R2_MS")
                continue
            row = by.loc[s]
            snr = row.get("snr")
            gate = str(row.get("vy_identity_gate") or "")
            st = str(row.get("source_state") or "")
            try:
                snr_f = float(snr)
            except (TypeError, ValueError):
                snr_f = float("nan")
            if st not in ("DETECTED_P1", "DETECTED_P2"):
                d3_bits.append(f"{s} D3 source_state={st}")
            elif gate.lower() == "fail":
                d3_bits.append(f"{s} D3 gate=fail")
            elif math.isfinite(snr_f) and snr_f < 10:
                d3_bits.append(f"{s} D3 snr={snr_f:.3f}<10")
        if d3_bits:
            reasons.append("D3 predicate " + "; ".join(d3_bits))
    if not reasons:
        reasons.append("UNNAMED: swapped " + ",".join(swapped))
    return " | ".join(reasons)


def compare_pair(left: Path, right: Path, ids: list[str]) -> pd.DataFrame:
    e_l = ensemble_map(left)
    e_r = ensemble_map(right)
    s_l = summary_map(left)
    s_r = summary_map(right)
    ms_l = left / "platesolve" / SETUP / "masterstars_full_match.csv"
    ms_r = right / "platesolve" / SETUP / "masterstars_full_match.csv"
    rows = []
    for tid in ids:
        el = e_l.get(tid, set())
        er = e_r.get(tid, set())
        ident = el == er
        swapped = sorted(el.symmetric_difference(er))
        mag_l = mag_series(left, tid)
        mag_r = mag_series(right, tid)
        dmag = float("nan")
        dmag_unique = []
        if mag_l is not None and mag_r is not None:
            n = min(len(mag_l), len(mag_r))
            if n:
                delta = (mag_r.to_numpy()[:n] - mag_l.to_numpy()[:n]) * 1000.0
                dmag = float(np.nanmedian(np.abs(delta)))
                finite = delta[np.isfinite(delta)]
                if len(finite):
                    uniq = np.unique(np.round(finite, 6))
                    dmag_unique = [float(x) for x in uniq[:12]]
        rms_l = (s_l.get(tid) or {}).get("lc_rms")
        rms_r = (s_r.get(tid) or {}).get("lc_rms")
        drms = float("nan")
        if rms_l is not None and rms_r is not None and math.isfinite(rms_l) and math.isfinite(rms_r):
            drms = (rms_r - rms_l) * 1000.0
        sha_l = lc_sha(left, tid)
        sha_r = lc_sha(right, tid)
        lc_eq = "Y" if sha_l and sha_r and sha_l == sha_r else "N"
        aavso_l = report_files(left, "aavso", tid)
        aavso_r = report_files(right, "aavso", tid)
        va_l = report_files(left, "varastro", tid)
        va_r = report_files(right, "varastro", tid)
        aavso_eq = "Y"
        if aavso_l or aavso_r:
            names = {p.name for p in aavso_l} | {p.name for p in aavso_r}
            aavso_eq = "Y" if names and all(
                sha_pair(phot_dir(left) / "lightcurves_reports" / "aavso" / n, phot_dir(right) / "lightcurves_reports" / "aavso" / n) == "Y"
                for n in names
            ) else "N"
        va_eq = "Y"
        if va_l or va_r:
            names = {p.name for p in va_l} | {p.name for p in va_r}
            va_eq = "Y" if names and all(
                sha_pair(phot_dir(left) / "lightcurves_reports" / "varastro" / n, phot_dir(right) / "lightcurves_reports" / "varastro" / n) == "Y"
                for n in names
            ) else "N"
        rows.append(
            {
                "target": tid,
                "ensemble_identical": "Y" if ident else "N",
                "n_comps_r1p": len(el),
                "n_comps_r2": len(er),
                "ids_swapped": ";".join(swapped),
                "median_dmag_mmag": dmag,
                "dmag_unique_mmag": ";".join(f"{x:.6f}" for x in dmag_unique),
                "dRMS_mmag": drms,
                "LC_SHA_equal": lc_eq,
                "AAVSO_VarAstro_SHA_equal": "Y" if aavso_eq == "Y" and va_eq == "Y" else "N",
                "named_cause": ms_cause(tid, el, er, ms_l, ms_r),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    ids = lc60_ids()
    df = compare_pair(R1P, R2, ids)
    df.to_csv(SESSION / "t3_r2_vs_r1p.csv", index=False)
    ens_chg = df[df["ensemble_identical"] != "Y"]
    same = df[df["ensemble_identical"] == "Y"]
    sha_move_same = same[same["LC_SHA_equal"] != "Y"]
    rec: dict = {
        "n_targets": len(ids),
        "n_ensemble_change": int(len(ens_chg)),
        "ensemble_change_targets": ens_chg[["target", "ids_swapped", "named_cause", "n_comps_r1p", "n_comps_r2", "median_dmag_mmag", "dRMS_mmag"]].to_dict("records"),
        "n_same_ensemble": int(len(same)),
        "n_same_ens_lc_identical": int((same["LC_SHA_equal"] == "Y").sum()),
        "same_ens_sha_moved": sha_move_same["target"].tolist(),
        "same_ens_sha_moved_median_dmag_mmag": sha_move_same["median_dmag_mmag"].tolist() if len(sha_move_same) else [],
        "css_present_r1p": (phot_dir(R1P) / "lightcurves" / f"lightcurve_{PRED_CSS}.csv").is_file(),
        "css_present_r2": (phot_dir(R2) / "lightcurves" / f"lightcurve_{PRED_CSS}.csv").is_file(),
        "d3_target_row": df.loc[df["target"] == PRED_D3_TARGET].to_dict("records"),
    }
    predicted_ens = {PRED_D3_TARGET, PRED_CSS}
    extra = [r for r in rec["ensemble_change_targets"] if r["target"] not in predicted_ens]
    rec["c6_0_p1_extra_ensemble_changes"] = extra
    rec["c6_0_p1_hit"] = (not extra) and rec["n_ensemble_change"] <= 2
    (SESSION / "t3_r2_vs_r1p.json").write_text(json.dumps(rec, indent=2, default=str), encoding="ascii")
    print(json.dumps({k: rec[k] for k in rec if k != "ensemble_change_targets"}, indent=2, default=str))
    print("ensemble_changes", rec["n_ensemble_change"])
    for r in rec["ensemble_change_targets"]:
        print(r["target"], r["named_cause"], r["ids_swapped"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
