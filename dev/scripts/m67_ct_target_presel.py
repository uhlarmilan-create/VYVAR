#!/usr/bin/env python3
"""Pre-select variable_targets for M67 CT validation: in-range BP-RP + red-giant Path-B checks."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import AppConfig  # noqa: E402

VAR_COLS = [
    "name",
    "catalog_id",
    "catalog",
    "ra_deg",
    "dec_deg",
    "priority",
    "notes",
    "vsx_name",
    "vsx_type",
    "vsx_period",
    "x",
    "y",
    "mag",
    "zone",
    "gaia_match_arcsec",
    "gaia_match_quality",
    "gaia_match_source",
    "vsx_mag_max",
]


def _draft_dir(draft_id: int) -> Path:
    cfg = AppConfig()
    return Path(cfg.archive_root) / "Drafts" / f"draft_{draft_id:06d}"


def _comp_bp_rp_range(comp_csv: Path) -> tuple[float, float, set[str]]:
    comp = pd.read_csv(comp_csv, low_memory=False, dtype={"catalog_id": str})
    comp_ids: set[str] = set()
    if "catalog_id" in comp.columns:
        comp_ids = {str(v).strip() for v in comp["catalog_id"].dropna().astype(str) if str(v).strip()}
    if "bp_rp" not in comp.columns:
        return float("nan"), float("nan"), comp_ids
    bps = pd.to_numeric(comp["bp_rp"], errors="coerce").dropna()
    if bps.empty:
        return float("nan"), float("nan"), comp_ids
    return float(bps.min()), float(bps.max()), comp_ids


def _normalize_cid(val: Any) -> str:
    s = str(val or "").strip()
    if not s or s.lower() in ("nan", "none"):
        return ""
    if "e" in s.lower():
        try:
            return str(int(float(s)))
        except (TypeError, ValueError):
            return s
    if s.endswith(".0"):
        s = s[:-2]
    return s


def _row_from_master(ms_row: pd.Series, *, label: str, notes: str) -> dict[str, Any]:
    cid = _normalize_cid(ms_row.get("catalog_id"))
    mag = pd.to_numeric(ms_row.get("mag", ms_row.get("phot_g_mean_mag")), errors="coerce")
    return {
        "name": label,
        "catalog_id": cid,
        "catalog": str(ms_row.get("catalog", "Gaia DR3") or "Gaia DR3"),
        "ra_deg": float(ms_row.get("ra_deg")),
        "dec_deg": float(ms_row.get("dec_deg")),
        "priority": 1,
        "notes": notes,
        "vsx_name": "",
        "vsx_type": "",
        "vsx_period": "",
        "x": float(ms_row.get("x")),
        "y": float(ms_row.get("y")),
        "mag": float(mag) if math.isfinite(float(mag)) else float("nan"),
        "zone": str(ms_row.get("zone", "") or ""),
        "gaia_match_arcsec": 0.0,
        "gaia_match_quality": "good",
        "gaia_match_source": "masterstars",
        "vsx_mag_max": float("nan"),
    }


def presel_obs_group(
    ps_dir: Path,
    *,
    n_in_range: int = 25,
    n_red_giants: int = 5,
    mag_lo: float = 10.0,
    mag_hi: float = 16.5,
) -> dict[str, Any]:
    comp_csv = ps_dir / "comparison_stars.csv"
    ms_csv = ps_dir / "masterstars_full_match.csv"
    if not ms_csv.is_file():
        ms_csv = ps_dir / "masterstars.csv"
    if not comp_csv.is_file() or not ms_csv.is_file():
        return {"obs_group": ps_dir.name, "error": "missing comp/masterstar files"}

    comp_min, comp_max, comp_ids = _comp_bp_rp_range(comp_csv)
    ms = pd.read_csv(ms_csv, low_memory=False)
    ms["catalog_id_norm"] = ms["catalog_id"].map(_normalize_cid)
    ms["bp_rp"] = pd.to_numeric(ms.get("bp_rp"), errors="coerce")
    ms["mag"] = pd.to_numeric(ms.get("mag", ms.get("phot_g_mean_mag")), errors="coerce")
    ms["ra_deg"] = pd.to_numeric(ms.get("ra_deg"), errors="coerce")
    ms["dec_deg"] = pd.to_numeric(ms.get("dec_deg"), errors="coerce")
    ms["x"] = pd.to_numeric(ms.get("x"), errors="coerce")
    ms["y"] = pd.to_numeric(ms.get("y"), errors="coerce")

    usable = ms[
        ms["catalog_id_norm"].ne("")
        & ms["bp_rp"].notna()
        & ms["mag"].between(mag_lo, mag_hi)
        & ms["ra_deg"].notna()
        & ms["dec_deg"].notna()
        & ms["x"].notna()
        & ms["y"].notna()
        & ~ms["catalog_id_norm"].isin(comp_ids)
    ].copy()

    in_range = usable[usable["bp_rp"].between(comp_min, comp_max, inclusive="both")].sort_values("mag")
    out_range = usable[usable["bp_rp"] > comp_max + 0.15].sort_values("bp_rp", ascending=False)

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _, r in in_range.head(int(n_in_range)).iterrows():
        cid = str(r["catalog_id_norm"])
        if cid in seen:
            continue
        seen.add(cid)
        rows.append(
            _row_from_master(
                r,
                label=f"M67 in-range {cid}",
                notes=f"CT presel in-range BP-RP [{comp_min:.2f},{comp_max:.2f}]",
            )
        )
    for _, r in out_range.head(int(n_red_giants)).iterrows():
        cid = str(r["catalog_id_norm"])
        if cid in seen:
            continue
        seen.add(cid)
        rows.append(
            _row_from_master(
                r,
                label=f"M67 red-giant {cid}",
                notes=f"CT presel out-of-range BP-RP > {comp_max:.2f} (Path-B check)",
            )
        )

    out_df = pd.DataFrame(rows)
    for c in VAR_COLS:
        if c not in out_df.columns:
            out_df[c] = ""
    out_df = out_df[VAR_COLS]
    vt_path = ps_dir / "variable_targets.csv"
    presel_path = ps_dir / "variable_targets.presel.csv"
    out_df.to_csv(presel_path, index=False)
    import os

    if os.environ.get("VYVAR_CT_PROTOTYPE", "").strip().lower() in ("1", "true", "yes", "on"):
        out_df.to_csv(vt_path, index=False)
        written_path = vt_path
    else:
        written_path = presel_path

    return {
        "obs_group": ps_dir.name,
        "comp_min": comp_min,
        "comp_max": comp_max,
        "n_comp": len(comp_ids),
        "n_in_range_candidates": int(len(in_range)),
        "n_out_range_candidates": int(len(out_range)),
        "n_variable_targets_written": int(len(out_df)),
        "n_in_range_written": int(min(len(in_range), n_in_range)),
        "n_red_giant_written": int(min(len(out_range), n_red_giants)),
        "variable_targets_csv": str(written_path),
        "variable_targets_presel_csv": str(presel_path),
        "overwrote_production_vt": written_path == vt_path,
    }


def presel_draft(
    draft_id: int,
    *,
    filters: tuple[str, ...] = ("Blue", "Green", "Red"),
    **kwargs: Any,
) -> list[dict[str, Any]]:
    draft = _draft_dir(draft_id)
    ps_root = draft / "platesolve"
    results: list[dict[str, Any]] = []
    for d in sorted(ps_root.iterdir()):
        if not d.is_dir():
            continue
        flt = d.name.split("_")[0]
        if flt not in filters:
            continue
        results.append(presel_obs_group(d, **kwargs))
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draft", type=int, required=True)
    ap.add_argument("--n-in-range", type=int, default=25)
    ap.add_argument("--n-red-giants", type=int, default=5)
    args = ap.parse_args()
    reps = presel_draft(
        args.draft,
        n_in_range=int(args.n_in_range),
        n_red_giants=int(args.n_red_giants),
    )
    for r in reps:
        print(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
