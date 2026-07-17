#!/usr/bin/env python3
"""VALIDATE-429 + 428/429 comparison forensics (read-only)."""
from __future__ import annotations

import json
import math
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any

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

ARCHIVE = _ROOT / "Archive" / "Drafts"
SETUP = "NoFilter_60_2"
D428 = ARCHIVE / "draft_000428" / "platesolve" / SETUP
D429 = ARCHIVE / "draft_000429" / "platesolve" / SETUP


def _read_meta(path: Path) -> dict[str, Any]:
    p = path / "photometry" / "pipeline_meta.json"
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _infolog_lines(draft_setup: Path) -> list[str]:
    draft_root = draft_setup.parent.parent  # .../draft_NNNNNN
    logs = sorted(draft_root.glob("infolog_*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not logs:
        return []
    return logs[0].read_text(encoding="utf-8", errors="replace").splitlines()


def _grep_lines(lines: list[str], pattern: str) -> list[str]:
    rx = re.compile(pattern)
    return [ln for ln in lines if rx.search(ln)]


def _config_diff(a: dict, b: dict) -> list[str]:
    diffs: list[str] = []
    keys = sorted(set(a.keys()) | set(b.keys()))
    for k in keys:
        va, vb = a.get(k), b.get(k)
        if va != vb:
            diffs.append(k)
    return diffs


def _load_gaia(con: sqlite3.Connection, table: str, cid: str) -> tuple[float, float] | None:
    row = con.execute(f"SELECT ra, dec FROM {table} WHERE source_id=?", (cid,)).fetchone()
    if not row:
        return None
    return float(row[0]), float(row[1])


def main() -> int:
    out: dict[str, Any] = {"setup": SETUP}
    log429 = _infolog_lines(D429)
    log428 = _infolog_lines(D428)

    # Part A infolog evidence
    out["infolog_429"] = {
        "path": str(sorted(D429.parent.parent.glob("infolog_*.txt"))[-1]) if list(D429.parent.parent.glob("infolog_*.txt")) else None,
        "utc_header": _grep_lines(log429, r"Infolog timestamps unified UTC|UTC")[:3],
        "epsf_once": _grep_lines(log429, r"ePSF|epsf")[:5],
        "repair_summary": _grep_lines(log429, r"REPAIR summary")[:3],
        "identity_gate": _grep_lines(log429, r"post_match_identity_gate|identity gate|post-match identity")[:5],
        "optimizer_identity": _grep_lines(log429, r"optimizer post-match identity|Astrometry optimizer post-match")[:3],
        "wcs_roundtrip": _grep_lines(log429, r"WCS invertibility gate PASS|WCS round-trip PASS")[:8],
        "finalize_coords": _grep_lines(log429, r"finalize_masterstar_sky_coords")[:3],
        "vsx_stamp": _grep_lines(log429, r"vsx_known_variable stamp")[:5],
        "border_prealign": _grep_lines(log429, r"\[BORDER\] Glob found 0 aligned")[:3],
        "ac_lines": _grep_lines(log429, r"\[AC\]")[:8],
        "ac_run_summary": _grep_lines(log429, r"\[AC\] run summary")[:3],
        "tap_timeout": _grep_lines(log429, r"Gaia TAP")[:10],
        "hrd_skip": _grep_lines(log429, r"HRD enrichment skipped")[:3],
    }

    meta428 = _read_meta(D428)
    meta429 = _read_meta(D429)
    prov429 = meta429.get("provenance") or {}
    out["pipeline_meta_429"] = {
        "git_hash": prov429.get("git_hash"),
        "git_dirty": prov429.get("git_dirty"),
        "entry_point": prov429.get("entry_point"),
        "wcs_roundtrip_p99_px": meta429.get("wcs_roundtrip_p99_px"),
        "wcs_roundtrip_pass": meta429.get("wcs_roundtrip_pass"),
    }

    snap428 = (meta428.get("provenance") or {}).get("config_snapshot") or {}
    snap429 = prov429.get("config_snapshot") or {}
    cfg_diff = _config_diff(snap428, snap429)
    out["config_snapshot_diff_428_vs_429"] = {
        "n_diff_keys": len(cfg_diff),
        "diff_keys": cfg_diff[:40],
        "identical": len(cfg_diff) == 0,
    }

    # variability candidates
    vc429 = pd.read_csv(D429 / "variability_candidates.csv", low_memory=False)
    vt429 = pd.read_csv(D429 / "variable_targets.csv", dtype={"catalog_id": str})
    ms429 = pd.read_csv(D429 / "masterstars_full_match.csv", dtype={"catalog_id": str})
    vt_ids = {normalize_gaia_source_id(x) for x in vt429["catalog_id"] if str(x).strip()}
    vc_ids = set()
    if "catalog_id" in vc429.columns:
        vc_ids = {normalize_gaia_source_id(x) for x in vc429["catalog_id"] if str(x).strip()}
    known_in_vc = vc_ids & vt_ids
    ms_vsx = ms429[ms429.get("vsx_known_variable", False).astype(bool)] if "vsx_known_variable" in ms429.columns else pd.DataFrame()
    out["variability_candidates_429"] = {
        "n_rows": int(len(vc429)),
        "n_with_catalog_id": int(vc429.get("catalog_id", pd.Series(dtype=object)).astype(str).str.strip().ne("").sum()) if "catalog_id" in vc429.columns else 0,
        "known_vsx_in_candidates": int(len(known_in_vc)),
        "vt_export_line": _grep_lines(log429, r"Gaia<=10|CSV=245|variable_targets export|variability_candidates")[:5],
        "ms_vsx_true": int(ms_vsx.shape[0]) if not ms_vsx.empty else int(ms429.get("vsx_known_variable", pd.Series([False]*len(ms429))).sum()),
    }

    ex429 = pd.read_csv(D429 / "photometry" / "excluded_targets.csv") if (D429 / "photometry" / "excluded_targets.csv").is_file() else pd.DataFrame()
    out["excluded_targets_429"] = {"n": int(len(ex429)), "cols": list(ex429.columns)}

    # masterstars coord
    gdb = str((_ROOT / "GAIA_DR3" / "vyvar_gaia_dr3.db").resolve())
    con = sqlite3.connect(gdb)
    table = _pick_gaia_table(con)
    matched = ms429[ms429["catalog_id"].fillna("").astype(str).str.strip().ne("")]
    seps: list[float] = []
    for _, row in matched.head(5000).iterrows():
        cid = normalize_gaia_source_id(row["catalog_id"])
        g = _load_gaia(con, table, cid)
        if g is None:
            continue
        seps.append(_sep_arcsec(float(row["ra_deg"]), float(row["dec_deg"]), g[0], g[1]))
    con.close()
    seps_a = np.asarray(seps, float)
    out["masterstars_429_coord"] = {
        "coord_source_counts": ms429.get("coord_source", pd.Series(dtype=object)).value_counts().to_dict() if "coord_source" in ms429.columns else {},
        "matched_n": int(len(matched)),
        "sep_vs_gaia_median_arcsec": float(np.nanmedian(seps_a)) if seps_a.size else float("nan"),
        "sep_vs_gaia_p95_arcsec": float(np.nanpercentile(seps_a, 95)) if seps_a.size else float("nan"),
        "n_sep_gt_2arcsec": int(np.sum(seps_a > 2.0)) if seps_a.size else 0,
    }

    # Part B census
    ms428 = pd.read_csv(D428 / "masterstars_full_match.csv", dtype={"catalog_id": str})
    m428 = ms428[ms428["catalog_id"].fillna("").astype(str).str.strip().ne("")]["catalog_id"].map(normalize_gaia_source_id)
    m429 = matched["catalog_id"].map(normalize_gaia_source_id)
    set428 = set(m428.tolist())
    set429 = set(m429.tolist())
    only428 = set428 - set429
    only429 = set429 - set428
    um428 = ms428[ms428["catalog_id"].fillna("").astype(str).str.strip().eq("")]
    um429 = ms429[ms429["catalog_id"].fillna("").astype(str).str.strip().eq("")]
    out["census_compare"] = {
        "n_ms_428": int(len(ms428)),
        "n_ms_429": int(len(ms429)),
        "matched_428": int(len(set428)),
        "matched_429": int(len(set429)),
        "unmatched_428": int(len(um428)),
        "unmatched_429": int(len(um429)),
        "matched_only_428": int(len(only428)),
        "matched_only_429": int(len(only429)),
        "only428_mag_median": float(pd.to_numeric(ms428[ms428["catalog_id"].map(normalize_gaia_source_id).isin(only428)]["mag"], errors="coerce").median()) if only428 else float("nan"),
    }

    # BO CVn
    at428 = pd.read_csv(D428 / "photometry" / "photometry_summary.csv", low_memory=False)
    at429 = pd.read_csv(D429 / "photometry" / "photometry_summary.csv", low_memory=False)
    def _filter_bo(df: pd.DataFrame) -> pd.DataFrame:
        if "name" not in df.columns:
            return df.iloc[0:0]
        mask = df["name"].astype(str).str.contains("BO CVn", na=False)
        return df.loc[mask]

    bo428 = _filter_bo(at428)
    bo429 = _filter_bo(at429)
    out["bo_cvn"] = {
        "428": bo428[["name", "lc_median_mag", "n_comp", "comp_path"]].to_dict("records") if not bo428.empty else [],
        "429": bo429[["name", "lc_median_mag", "n_comp", "comp_path"]].to_dict("records") if not bo429.empty else [],
    }

    # masterstar JSON pass counts if present
    for label, base in [("428", D428), ("429", D429)]:
        for name in ("masterstar_qa.json", "pipeline_meta.json"):
            p = base / name
            if p.is_file():
                j = json.loads(p.read_text(encoding="utf-8"))
                if "n_raw_dao" in j:
                    out[f"json_{label}"] = {"file": str(p), "n_raw_dao": j.get("n_raw_dao")}

    txt = json.dumps(out, indent=2, default=str)
    dest = _ROOT / "tmp" / "validate_429_wcsinv.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(txt + "\n", encoding="utf-8")
    sys.stdout.write(txt + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
