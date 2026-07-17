#!/usr/bin/env python3
"""428 vs 429 pass-2 contamination + BO CVn comp diff (read-only)."""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401  (repo layout: src_py + dev on sys.path)
_ROOT = _bootstrap.REPO_ROOT
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from gaia_catalog_id import normalize_gaia_source_id
from repair_catalog_ids import _pick_gaia_table, _sep_arcsec

D428 = _ROOT / "Archive/Drafts/draft_000428/platesolve/NoFilter_60_2"
D429 = _ROOT / "Archive/Drafts/draft_000429/platesolve/NoFilter_60_2"
BO_CID = "1498613634033133184"


def _grep_log(draft_root: Path, pattern: str) -> list[str]:
    logs = sorted(draft_root.glob("infolog_*.txt"))
    if not logs:
        return []
    rx = re.compile(pattern)
    return [ln for ln in logs[-1].read_text(encoding="utf-8", errors="replace").splitlines() if rx.search(ln)]


def main() -> None:
    out: dict = {}

    for label, dr in [("428", D428.parent.parent), ("429", D429.parent.parent)]:
        out[f"log_{label}"] = {
            "dao_pass2": _grep_log(dr, r"\[DAO pass 2\]"),
            "dao_master": _grep_log(dr, r"\[DAO pass 2 master\]"),
            "dao_raw": _grep_log(dr, r"DAO na sn|n_raw_dao|MASTERSTAR JSON consistency"),
        }

    ms428 = pd.read_csv(D428 / "masterstars_full_match.csv", dtype={"catalog_id": str})
    ms429 = pd.read_csv(D429 / "masterstars_full_match.csv", dtype={"catalog_id": str})
    um428 = ms428[ms428["catalog_id"].fillna("").astype(str).str.strip().eq("")]
    um429 = ms429[ms429["catalog_id"].fillna("").astype(str).str.strip().eq("")]
    m428 = {
        normalize_gaia_source_id(x)
        for x in ms428["catalog_id"]
        if str(x).strip()
    }
    m429 = {
        normalize_gaia_source_id(x)
        for x in ms429["catalog_id"]
        if str(x).strip()
    }
    only428 = m428 - m429
    sub = ms428[ms428["catalog_id"].map(lambda x: normalize_gaia_source_id(x) in only428)]

    con = sqlite3.connect(_ROOT / "GAIA_DR3/vyvar_gaia_dr3.db")
    table = _pick_gaia_table(con)
    seps = []
    for _, r in sub.iterrows():
        cid = normalize_gaia_source_id(r["catalog_id"])
        row = con.execute(f"SELECT ra,dec FROM {table} WHERE source_id=?", (cid,)).fetchone()
        if not row:
            continue
        seps.append(_sep_arcsec(float(r["ra_deg"]), float(r["dec_deg"]), row[0], row[1]))
    con.close()
    seps_a = np.asarray(seps, float)

    out["pass2_contamination"] = {
        "unmatched_428": int(len(um428)),
        "unmatched_429": int(len(um429)),
        "um428_mag_median": float(pd.to_numeric(um428["mag"], errors="coerce").median()),
        "um429_mag_median": float(pd.to_numeric(um429["mag"], errors="coerce").median()),
        "matched_only_428": int(len(only428)),
        "only428_mag_median": float(pd.to_numeric(sub["mag"], errors="coerce").median()),
        "only428_sep_median_arcsec": float(np.median(seps_a)) if seps_a.size else float("nan"),
        "only428_sep_p95_arcsec": float(np.percentile(seps_a, 95)) if seps_a.size else float("nan"),
        "only428_sep_gt2_arcsec": int(np.sum(seps_a > 2.0)) if seps_a.size else 0,
    }

    # config diff detail
    for label, d in [("428", D428), ("429", D429)]:
        meta = json.loads((d / "photometry/pipeline_meta.json").read_text(encoding="utf-8"))
        snap = (meta.get("provenance") or {}).get("config_snapshot") or {}
        out[f"config_{label}"] = {
            k: snap.get(k)
            for k in [
                "annulus_inner_fwhm",
                "comp_max_delta_bprp",
                "hrd_enrich_tap_timeout_s",
                "phase01_comparison_max_comp_rms",
                "phase01_comparison_min_dist_arcsec",
                "config_schema_version",
            ]
        }

    # BO CVn comp diff
    bo: dict = {}
    for label, d in [("428", D428), ("429", D429)]:
        summ = pd.read_csv(d / "photometry/photometry_summary.csv", low_memory=False)
        row = summ[summ.get("vsx_name", pd.Series(dtype=str)).astype(str).eq("BO CVn")]
        if row.empty and "catalog_id" in summ.columns:
            row = summ[summ["catalog_id"].astype(str).eq(BO_CID)]
        comp = pd.read_csv(d / "photometry/comparison_stars_per_target.csv", low_memory=False)
        bo_comp = comp[comp.get("target_catalog_id", comp.get("catalog_id", pd.Series(dtype=str))).astype(str).eq(BO_CID)]
        if bo_comp.empty and "target_name" in comp.columns:
            bo_comp = comp[comp["target_name"].astype(str).str.contains("BO CVn", na=False)]
        bo[label] = {
            "lc_median_mag": float(row.iloc[0]["lc_median_mag"]) if not row.empty and "lc_median_mag" in row.columns else None,
            "n_good_comp": int(row.iloc[0]["n_good_comp"]) if not row.empty and "n_good_comp" in row.columns else None,
            "n_comp": int(row.iloc[0]["n_comp"]) if not row.empty and "n_comp" in row.columns else None,
            "comp_ids": bo_comp.get("comp_catalog_id", bo_comp.get("catalog_id", pd.Series(dtype=str))).astype(str).tolist()[:10],
            "n_comp_rows": int(len(bo_comp)),
        }
    only428_comp = set(bo["428"].get("comp_ids") or []) - set(bo["429"].get("comp_ids") or [])
    only429_comp = set(bo["429"].get("comp_ids") or []) - set(bo["428"].get("comp_ids") or [])
    bo["comp_only_428"] = list(only428_comp)
    bo["comp_only_429"] = list(only429_comp)
    out["bo_cvn"] = bo

    # hrd tap attempts
    for label, d in [("428", D428), ("429", D429)]:
        p = d / "_hrd_cache/summary.json"
        out[f"hrd_{label}"] = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}

    # phase0 census vs excluded
    ex429 = pd.read_csv(D429 / "photometry/excluded_targets.csv")
    out["excluded_429_reasons"] = ex429["reason"].value_counts().to_dict() if "reason" in ex429.columns else {}
    out["phase0_log"] = _grep_log(D429.parent.parent, r"select_active_targets|Phase-0|excluded")[:8]

    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
