# -*- coding: ascii -*-
"""IDENT-JUMP-01 M1 census: per-star position jumps vs master reference."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\ASTRO\python\VYVAR")
OUT = ROOT / "dev" / "results" / "context" / "session_20261001_ident_jump_01"
SETUP = "NoFilter_60_2"

DRAFTS = {
    "521": ROOT / "Archive" / "Drafts" / "draft_000521",
    "516_era06": ROOT
    / "Archive"
    / "Drafts"
    / "draft_000516_snapshot_era06_20260928",
    "516_live": ROOT / "Archive" / "Drafts" / "draft_000516",
}

ANCHORS_516 = {
    "BO": "1498613634033133184",
    "FW": "1497343732462852864",
    "GH": "1497771992240531712",  # may need verify
}


def mag_bin(g: float) -> str:
    if not math.isfinite(g):
        return "unknown"
    if g < 10:
        return "G<10"
    if g < 11:
        return "10-11"
    if g < 12:
        return "11-12"
    if g < 13:
        return "12-13"
    if g < 14:
        return "13-14"
    if g < 15:
        return "14-15"
    return "G>=15"


def load_master_xy(draft: Path) -> pd.DataFrame:
    photo = draft / "platesolve" / SETUP / "photometry"
    candidates = [
        photo / "active_targets.csv",
        draft / "platesolve" / SETUP / "masterstars_full_match.csv",
        draft / "platesolve" / SETUP / "masterstars.csv",
    ]
    # Prefer masterstars for full field; merge G from active if present
    ms = None
    for c in candidates[1:]:
        if c.is_file():
            ms = pd.read_csv(c, dtype={"catalog_id": str}, low_memory=False)
            break
    if ms is None:
        raise FileNotFoundError(f"no masterstars in {draft}")
    ms["catalog_id"] = ms["catalog_id"].astype(str).str.strip()
    ms["x_ref"] = pd.to_numeric(ms["x"], errors="coerce")
    ms["y_ref"] = pd.to_numeric(ms["y"], errors="coerce")
    gcol = "phot_g_mean_mag" if "phot_g_mean_mag" in ms.columns else "mag"
    ms["g_mag"] = pd.to_numeric(ms.get(gcol), errors="coerce")
    return ms[["catalog_id", "x_ref", "y_ref", "g_mag"]].drop_duplicates("catalog_id")


def census_draft(label: str, draft: Path) -> dict:
    frames_dir = draft / "detrended_aligned" / "lights" / SETUP
    if not frames_dir.is_dir():
        return {"label": label, "ok": False, "reason": "no_frames_dir"}
    master = load_master_xy(draft)
    ref = master.set_index("catalog_id")
    procs = sorted(frames_dir.glob("proc_*.csv"))
    if not procs:
        return {"label": label, "ok": False, "reason": "no_proc"}

    # pick reference frame (prefer *035* else first)
    ref_frame = None
    for p in procs:
        if "_035" in p.stem or p.stem.endswith("_035"):
            ref_frame = p
            break
    if ref_frame is None:
        ref_frame = procs[0]

    rows = []
    frame_stats = []
    for proc in procs:
        df = pd.read_csv(
            proc,
            usecols=lambda c: c
            in {
                "catalog_id",
                "x",
                "y",
                "forced_photometry",
                "source_type",
                "flux",
                "gaia_dao_resid_px",
                "vy_identity_gate",
                "phot_g_mean_mag",
                "mag",
            },
            dtype={"catalog_id": str},
            low_memory=False,
        )
        if "catalog_id" not in df.columns:
            continue
        df["catalog_id"] = df["catalog_id"].astype(str).str.strip()
        df = df[df["catalog_id"].ne("") & df["catalog_id"].str.lower().ne("nan")]
        df["x"] = pd.to_numeric(df["x"], errors="coerce")
        df["y"] = pd.to_numeric(df["y"], errors="coerce")
        joined = df.join(ref, on="catalog_id", how="inner", rsuffix="_m")
        # join via map if index join fails shape
        if "x_ref" not in joined.columns:
            joined = df.merge(master, on="catalog_id", how="inner")
        dx = joined["x"] - joined["x_ref"]
        dy = joined["y"] - joined["y_ref"]
        off = np.hypot(dx.to_numpy(dtype=float), dy.to_numpy(dtype=float))
        g = pd.to_numeric(joined.get("g_mag"), errors="coerce").to_numpy(dtype=float)
        if "phot_g_mean_mag" in joined.columns:
            g2 = pd.to_numeric(joined["phot_g_mean_mag"], errors="coerce").to_numpy(dtype=float)
            g = np.where(np.isfinite(g2), g2, g)
        forced = (
            joined["forced_photometry"].astype(str).str.lower().isin({"true", "1", "yes"})
            if "forced_photometry" in joined.columns
            else pd.Series(False, index=joined.index)
        )
        # integer snap rate
        xarr = joined["x"].to_numpy(dtype=float)
        int_frac = float(np.mean(np.isfinite(xarr) & (np.abs(xarr - np.round(xarr)) < 1e-6)))
        stem = proc.stem.replace("proc_", "", 1)
        n = len(joined)
        frame_stats.append(
            {
                "frame": stem,
                "n": n,
                "frac_gt1": float(np.mean(off > 1)) if n else None,
                "frac_gt3": float(np.mean(off > 3)) if n else None,
                "frac_gt8": float(np.mean(off > 8)) if n else None,
                "n_gt3": int(np.sum(off > 3)),
                "n_gt8": int(np.sum(off > 8)),
                "frac_x_integer": int_frac,
            }
        )
        for i in range(n):
            rows.append(
                {
                    "frame": stem,
                    "catalog_id": str(joined["catalog_id"].iloc[i]),
                    "offset_px": float(off[i]) if np.isfinite(off[i]) else None,
                    "g_mag": float(g[i]) if np.isfinite(g[i]) else None,
                    "mag_bin": mag_bin(float(g[i]) if np.isfinite(g[i]) else float("nan")),
                    "forced": bool(forced.iloc[i]),
                    "x": float(joined["x"].iloc[i]) if np.isfinite(joined["x"].iloc[i]) else None,
                    "y": float(joined["y"].iloc[i]) if np.isfinite(joined["y"].iloc[i]) else None,
                    "x_ref": float(joined["x_ref"].iloc[i]),
                    "y_ref": float(joined["y_ref"].iloc[i]),
                }
            )

    rdf = pd.DataFrame(rows)
    fdf = pd.DataFrame(frame_stats)
    rdf.to_csv(OUT / f"offsets_all_{label}.csv", index=False)
    fdf.to_csv(OUT / f"frame_stats_{label}.csv", index=False)

    # by mag bin / forced
    summary_bins = []
    for forced_val in (False, True, None):
        sub = rdf if forced_val is None else rdf[rdf["forced"] == forced_val]
        tag = "all" if forced_val is None else ("forced" if forced_val else "dao")
        for b in ["G<10", "10-11", "11-12", "12-13", "13-14", "14-15", "G>=15", "unknown"]:
            s2 = sub[sub["mag_bin"] == b]
            if s2.empty:
                continue
            off = pd.to_numeric(s2["offset_px"], errors="coerce")
            summary_bins.append(
                {
                    "subset": tag,
                    "mag_bin": b,
                    "n_rows": int(len(s2)),
                    "frac_gt1": float((off > 1).mean()),
                    "frac_gt3": float((off > 3).mean()),
                    "frac_gt8": float((off > 8).mean()),
                }
            )
    pd.DataFrame(summary_bins).to_csv(OUT / f"bin_summary_{label}.csv", index=False)

    # per-target jumped epochs (targets with LCs)
    lc_dir = draft / "platesolve" / SETUP / "photometry" / "lightcurves"
    target_ids = set()
    if lc_dir.is_dir():
        for p in lc_dir.glob("*.csv"):
            # try read catalog_id from filename or first row
            try:
                head = pd.read_csv(p, nrows=2, dtype=str)
                if "catalog_id" in head.columns:
                    target_ids.add(str(head["catalog_id"].iloc[0]).strip())
            except Exception:
                pass
    # also active_targets
    at = draft / "platesolve" / SETUP / "photometry" / "active_targets.csv"
    if at.is_file():
        atd = pd.read_csv(at, dtype=str)
        if "catalog_id" in atd.columns:
            target_ids |= set(atd["catalog_id"].astype(str).str.strip())

    per_tgt = []
    for tid in sorted(target_ids):
        s = rdf[rdf["catalog_id"] == tid]
        if s.empty:
            continue
        off = pd.to_numeric(s["offset_px"], errors="coerce")
        per_tgt.append(
            {
                "catalog_id": tid,
                "n_frames": int(len(s)),
                "n_gt1": int((off > 1).sum()),
                "n_gt3": int((off > 3).sum()),
                "n_gt8": int((off > 8).sum()),
                "max_offset_px": float(off.max()) if len(off) else None,
                "median_offset_px": float(off.median()) if len(off) else None,
            }
        )
    tdf = pd.DataFrame(per_tgt).sort_values("n_gt8", ascending=False)
    tdf.to_csv(OUT / f"per_target_jumps_{label}.csv", index=False)

    # comps used
    comps_path = draft / "platesolve" / SETUP / "photometry" / "comparison_stars_per_target.csv"
    comp_jump = []
    if comps_path.is_file():
        cp = pd.read_csv(comps_path, dtype=str)
        cids = set(cp["catalog_id"].astype(str).str.strip())
        for cid in cids:
            s = rdf[rdf["catalog_id"] == cid]
            if s.empty:
                continue
            off = pd.to_numeric(s["offset_px"], errors="coerce")
            comp_jump.append(
                {
                    "catalog_id": cid,
                    "n_frames": int(len(s)),
                    "n_gt3": int((off > 3).sum()),
                    "n_gt8": int((off > 8).sum()),
                    "max_offset_px": float(off.max()) if len(off) else None,
                }
            )
        pd.DataFrame(comp_jump).sort_values("n_gt8", ascending=False).to_csv(
            OUT / f"per_comp_jumps_{label}.csv", index=False
        )

    # anchors for 516
    anchors = {}
    for name, cid in ANCHORS_516.items():
        s = rdf[rdf["catalog_id"] == cid]
        if s.empty:
            anchors[name] = {"present": False}
            continue
        off = pd.to_numeric(s["offset_px"], errors="coerce")
        jumped = s.loc[off > 3, "frame"].tolist()
        anchors[name] = {
            "present": True,
            "catalog_id": cid,
            "n_gt3": int((off > 3).sum()),
            "n_gt8": int((off > 8).sum()),
            "max_offset_px": float(off.max()),
            "jumped_frames_gt3": jumped[:20],
        }

    # resid staleness check: unique gaia_dao_resid for one star
    resid_unique = None
    try:
        sample = pd.read_csv(procs[0], dtype=str, nrows=5)
        # pick a GAIA_MATCHED id
    except Exception:
        pass

    return {
        "label": label,
        "ok": True,
        "draft": str(draft),
        "n_frames": len(fdf),
        "n_offset_rows": len(rdf),
        "median_frac_gt3": float(fdf["frac_gt3"].median()) if len(fdf) else None,
        "median_frac_gt8": float(fdf["frac_gt8"].median()) if len(fdf) else None,
        "median_frac_x_integer": float(fdf["frac_x_integer"].median()) if len(fdf) else None,
        "worst_frames_gt8": fdf.sort_values("n_gt8", ascending=False).head(5).to_dict(orient="records"),
        "n_targets_with_gt8": int((tdf["n_gt8"] > 0).sum()) if len(tdf) else 0,
        "n_comps_with_gt8": int(sum(1 for r in comp_jump if r["n_gt8"] > 0)),
        "anchors": anchors,
        "bin_summary_path": str(OUT / f"bin_summary_{label}.csv"),
        "frame_stats_path": str(OUT / f"frame_stats_{label}.csv"),
        "per_target_path": str(OUT / f"per_target_jumps_{label}.csv"),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"task": "IDENT-JUMP-01-M1"}
    for label, path in DRAFTS.items():
        if not path.is_dir():
            summary[label] = {"ok": False, "reason": "missing", "path": str(path)}
            print("SKIP", label, path)
            continue
        print("CENSUS", label, flush=True)
        summary[label] = census_draft(label, path)
        print(json.dumps({k: summary[label].get(k) for k in summary[label] if k != "anchors"}, indent=2), flush=True)
        if summary[label].get("anchors"):
            print("anchors", summary[label]["anchors"], flush=True)
    (OUT / "m1_census_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="ascii")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
