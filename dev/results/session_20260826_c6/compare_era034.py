# -*- coding: ascii -*-
"""C6-3 era03 vs era04 ledger + same-ensemble mag_calib sanity."""
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

ERA03 = ROOT / "Archive" / "Drafts" / "draft_000516_snapshot_era03_20260820"
ERA04 = ROOT / "Archive" / "Drafts" / "draft_000516_snapshot_era04_20260826"
SETUP = "NoFilter_60_2"
SESSION = Path(__file__).resolve().parent
AC_MMAG_NOM = 2.4
AC_MMAG_TOL = 0.6
FRAME29 = "1498000793739050368"
DEPTH_VSX = "1500387696044768384"
D3_A = "1498964240802993408"
D3_B = "1500579870061241088"
D3_TARGET = "1497284015237511808"
C3K5 = "1496315070616056064"
CSS = "1497169940906156032"
BO = "1498613634033133184"
FW = "1497343732462852864"
GH = "1498804639818507904"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def phot(root: Path) -> Path:
    return root / "platesolve" / SETUP / "photometry"


def ensemble_map(root: Path) -> dict[str, set[str]]:
    p = phot(root) / "comparison_stars_per_target.csv"
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
    p = phot(root) / "photometry_summary.csv"
    if not p.is_file():
        return {}
    df = pd.read_csv(p, dtype={"catalog_id": str})
    out = {}
    for _, row in df.iterrows():
        cid = _norm_cid(row.get("catalog_id"))
        if cid:
            rec = {}
            if "lc_rms" in df.columns:
                rec["lc_rms"] = float(pd.to_numeric(row["lc_rms"], errors="coerce"))
            out[cid] = rec
    return out


def mag_df(root: Path, tid: str) -> pd.DataFrame | None:
    p = phot(root) / "lightcurves" / f"lightcurve_{tid}.csv"
    if not p.is_file():
        return None
    return pd.read_csv(p)


def report_names(root: Path, kind: str, tid: str) -> set[str]:
    d = phot(root) / "lightcurves_reports" / kind
    if not d.is_dir():
        return set()
    return {p.name for p in d.iterdir() if p.is_file() and tid in p.name}


def psf_path(root: Path, tid: str) -> Path | None:
    p = phot(root) / "lightcurves" / f"lightcurve_{tid}_psf.csv"
    return p if p.is_file() else None


def lc_targets(root: Path) -> set[str]:
    d = phot(root) / "lightcurves"
    if not d.is_dir():
        return set()
    out = set()
    for p in d.glob("lightcurve_*.csv"):
        if p.stem.endswith("_psf") or p.stem.endswith("_adaptive"):
            continue
        cid = _norm_cid(p.stem.replace("lightcurve_", "", 1))
        if cid:
            out.add(cid)
    return out


def zone_saturated(root: Path) -> set[str]:
    p = root / "platesolve" / SETUP / "masterstars_full_match.csv"
    if not p.is_file():
        return set()
    df = pd.read_csv(p, dtype=str, low_memory=False)
    if "catalog_id" not in df.columns or "zone" not in df.columns:
        return set()
    out = set()
    z = df["zone"].astype(str).str.strip().str.lower()
    for cid, zz in zip(df["catalog_id"], z):
        if zz == "saturated":
            n = _norm_cid(cid)
            if n:
                out.add(n)
    return out


def cause_tags(tid: str, swapped: list[str], files: list[str], zone_new: set[str] | None = None) -> list[str]:
    tags = []
    if tid == FRAME29 or FRAME29 in swapped:
        tags.append("FRAME-29")
    if tid == DEPTH_VSX or any("G>" in s or s == DEPTH_VSX for s in swapped):
        tags.append("DEPTH")
    if tid == CSS or CSS in swapped:
        tags.append("NAME-FIX")
    if D3_A in swapped or D3_B in swapped or tid == D3_TARGET:
        tags.append("D3-D5")
    if tid == C3K5 or C3K5 in swapped:
        tags.append("C3-K5")
    if zone_new and any(s in zone_new for s in swapped):
        tags.append("ZONE")
    if any(x.endswith("_psf") or x == "psf" for x in files) and not any(
        x.startswith("aperture") or x.startswith("aavso") or x.startswith("varastro") for x in files
    ):
        tags.append("ZP-OK")
    if not tags and swapped:
        tags.append("UNNAMED")
    if not tags and files:
        tags.append("UNNAMED")
    return tags


def main() -> int:
    e3 = ensemble_map(ERA03)
    e4 = ensemble_map(ERA04)
    s3 = summary_map(ERA03)
    s4 = summary_map(ERA04)
    z3 = zone_saturated(ERA03)
    z4 = zone_saturated(ERA04)
    zone_new = z4 - z3
    ids = sorted(lc_targets(ERA03) | lc_targets(ERA04))
    rows = []
    unnamed = []
    sanity_fail = []
    n_unchanged = 0
    tag_counts: dict[str, int] = {}
    for tid in ids:
        el = e3.get(tid, set())
        er = e4.get(tid, set())
        swapped = sorted(el.symmetric_difference(er))
        files = []
        p3 = phot(ERA03) / "lightcurves" / f"lightcurve_{tid}.csv"
        p4 = phot(ERA04) / "lightcurves" / f"lightcurve_{tid}.csv"
        ap_eq = p3.is_file() and p4.is_file() and sha256_file(p3) == sha256_file(p4)
        if p3.is_file() != p4.is_file() or (p3.is_file() and p4.is_file() and not ap_eq):
            files.append("aperture_lc")
        for kind in ("aavso", "varastro"):
            n3, n4 = report_names(ERA03, kind, tid), report_names(ERA04, kind, tid)
            changed = n3 != n4
            if not changed:
                for n in n3:
                    a = phot(ERA03) / "lightcurves_reports" / kind / n
                    b = phot(ERA04) / "lightcurves_reports" / kind / n
                    if a.is_file() and b.is_file() and sha256_file(a) != sha256_file(b):
                        changed = True
                        break
            if changed:
                files.append(kind)
        s3p, s4p = psf_path(ERA03, tid), psf_path(ERA04, tid)
        if bool(s3p) != bool(s4p) or (s3p and s4p and sha256_file(s3p) != sha256_file(s4p)):
            files.append("psf_lc")
        if not files:
            n_unchanged += 1
            continue
        mag3 = mag_df(ERA03, tid)
        mag4 = mag_df(ERA04, tid)
        dmag = float("nan")
        epoch_deltas = []
        if mag3 is not None and mag4 is not None and "mag_calib" in mag3.columns and "mag_calib" in mag4.columns:
            a = pd.to_numeric(mag3["mag_calib"], errors="coerce").to_numpy()
            b = pd.to_numeric(mag4["mag_calib"], errors="coerce").to_numpy()
            n = min(len(a), len(b))
            if n:
                delta = (b[:n] - a[:n]) * 1000.0
                dmag = float(np.nanmedian(np.abs(delta)))
                finite = delta[np.isfinite(delta)]
                epoch_deltas = [float(x) for x in finite]
        rms3 = (s3.get(tid) or {}).get("lc_rms")
        rms4 = (s4.get(tid) or {}).get("lc_rms")
        drms = float("nan")
        if rms3 is not None and rms4 is not None and math.isfinite(rms3) and math.isfinite(rms4):
            drms = (rms4 - rms3) * 1000.0
        tags = cause_tags(tid, swapped, files, zone_new=zone_new)
        if el == er and epoch_deltas:
            uniq = np.unique(np.round(np.array(epoch_deltas), 4))
            bad = [float(x) for x in uniq if abs(x) > 1e-6 and abs(abs(x) - AC_MMAG_NOM) > AC_MMAG_TOL]
            if bad:
                sanity_fail.append({"target": tid, "nonzero_deltas_mmag": bad[:12]})
            elif any(abs(abs(x) - AC_MMAG_NOM) <= AC_MMAG_TOL for x in uniq) and "D1-AC" not in tags:
                tags.append("D1-AC")
        if "UNNAMED" in tags:
            unnamed.append(tid)
        for t in tags:
            tag_counts[t] = tag_counts.get(t, 0) + 1
        rows.append(
            {
                "target": tid,
                "files_changed": ";".join(files),
                "n_comps_era03": len(el),
                "n_comps_era04": len(er),
                "median_dmag_mmag": dmag,
                "dRMS_mmag": drms,
                "cause": " ".join(f"[{t}]" for t in tags),
                "ids_swapped": ";".join(swapped),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(SESSION / "c63_era03_era04_ledger.csv", index=False)
    rec = {
        "n_targets_union": len(ids),
        "n_unchanged_all_files": n_unchanged,
        "n_changed": len(rows),
        "tag_counts": tag_counts,
        "unnamed": unnamed,
        "sanity_fail": sanity_fail,
        "BO_CVn": [r for r in rows if r["target"] == BO],
        "FW_CVn": [r for r in rows if r["target"] == FW],
        "GH_CVn": [r for r in rows if r["target"] == GH],
        "zone_new_saturated": sorted(zone_new),
        "stop_unnamed": bool(unnamed),
        "stop_sanity": bool(sanity_fail),
    }
    (SESSION / "c63_era03_era04.json").write_text(json.dumps(rec, indent=2, default=str), encoding="ascii")
    print(json.dumps(rec, indent=2, default=str))
    if rec["stop_unnamed"] or rec["stop_sanity"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
