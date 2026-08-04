#!/usr/bin/env python3
"""Audit Stage 3 Part 0c: delta tail stratification and anchor cohort restriction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "dev" / "scripts"))
import _bootstrap  # noqa: E402,F401

sys.path.insert(0, str(_bootstrap.REPO_ROOT / "src_py"))

from config import AppConfig  # noqa: E402
from photometry_core import _resolve_git_provenance  # noqa: E402

SETUP = "NoFilter_60_2"
REBUILD_DRAFT = 499
ANCHOR_SNAPSHOT = "draft_000435_snapshot_skysurface_20260716"


def _delta_table(rebuilt_lc: Path, anchor_lc: Path, target_cid: str) -> pd.DataFrame | None:
    rb = rebuilt_lc / f"lightcurve_{target_cid}.csv"
    an = anchor_lc / f"lightcurve_{target_cid}.csv"
    if not rb.is_file() or not an.is_file():
        return None
    rdf = pd.read_csv(rb, low_memory=False)
    adf = pd.read_csv(an, low_memory=False)
    if rdf.empty or adf.empty or "source_file" not in rdf.columns or "source_file" not in adf.columns:
        return None
    m = adf.merge(rdf, on="source_file", suffixes=("_anchor", "_rebuild"), how="inner")
    if m.empty:
        return None
    out = pd.DataFrame(
        {
            "target_cid": target_cid,
            "source_file": m["source_file"].astype(str),
            "mag_anchor": pd.to_numeric(m["mag_calib_final_anchor"], errors="coerce"),
            "mag_rebuild": pd.to_numeric(m["mag_calib_final_rebuild"], errors="coerce"),
            "err_anchor": pd.to_numeric(m["err_anchor"], errors="coerce"),
            "err_rebuild": pd.to_numeric(m["err_rebuild"], errors="coerce"),
            "n_good_comp_rebuild": pd.to_numeric(m.get("n_good_comp_rebuild", np.nan), errors="coerce"),
            "n_good_comp_anchor": pd.to_numeric(m.get("n_good_comp_anchor", np.nan), errors="coerce"),
            "trust_rebuild": m.get("trust_flag_rebuild", pd.Series([""] * len(m))).astype(str),
            "trust_anchor": m.get("trust_flag_anchor", pd.Series([""] * len(m))).astype(str),
        }
    )
    out["delta_mag"] = out["mag_rebuild"] - out["mag_anchor"]
    out["delta_err"] = out["err_rebuild"] - out["err_anchor"]
    out["ensemble_changed"] = out["n_good_comp_rebuild"] != out["n_good_comp_anchor"]
    return out


def _stats(vals: np.ndarray) -> dict[str, float | int]:
    v = vals[np.isfinite(vals)]
    if v.size == 0:
        return {"n": 0}
    return {
        "n": int(v.size),
        "median": float(np.median(v)),
        "p95": float(np.percentile(v, 95)),
        "max": float(np.max(v)),
        "p99": float(np.percentile(v, 99)),
    }


def _target_meta(ps_dir: Path, target_cid: str) -> dict[str, Any]:
    vt = ps_dir / "variable_targets.csv"
    if not vt.is_file():
        return {}
    df = pd.read_csv(vt, low_memory=False, dtype={"catalog_id": str})
    id_col = "catalog_id" if "catalog_id" in df.columns else "target_catalog_id"
    if id_col not in df.columns:
        return {}
    sub = df.loc[df[id_col].astype(str).str.strip() == str(target_cid)]
    if sub.empty:
        return {"in_variable_targets": False}
    row = sub.iloc[0]
    mag = pd.to_numeric(row.get("mag") or row.get("phot_g_mean_mag"), errors="coerce")
    trust = str(row.get("trust_flag") or row.get("trust") or "").strip()
    skip = str(row.get("skip_photometry") or "").strip().lower() in {"1", "true", "yes"}
    return {
        "in_variable_targets": True,
        "target_mag": float(mag) if np.isfinite(mag) else None,
        "trust_flag": trust,
        "skip_photometry": skip,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=REPO / "tmp" / "audit_stage3_part0c_results.json")
    args = parser.parse_args()

    cfg = AppConfig()
    rebuild = Path(cfg.archive_root) / "Drafts" / f"draft_{REBUILD_DRAFT:06d}"
    anchor = Path(cfg.archive_root) / "Drafts" / ANCHOR_SNAPSHOT
    rb_ps = rebuild / "platesolve" / SETUP
    an_ps = anchor / "platesolve" / SETUP
    rb_lc = rb_ps / "photometry" / "lightcurves"
    an_lc = an_ps / "photometry" / "lightcurves"

    gh, dirty, _ = _resolve_git_provenance()

    rb_ids = {p.stem.replace("lightcurve_", "") for p in rb_lc.glob("lightcurve_*.csv")}
    an_ids = {p.stem.replace("lightcurve_", "") for p in an_lc.glob("lightcurve_*.csv")}
    common = sorted(rb_ids & an_ids)
    only_rb = sorted(rb_ids - an_ids)
    only_an = sorted(an_ids - rb_ids)

    all_deltas: list[pd.DataFrame] = []
    for cid in common:
        dt = _delta_table(rb_lc, an_lc, cid)
        if dt is not None:
            meta = _target_meta(rb_ps, cid)
            for k, v in meta.items():
                dt[k] = v
            all_deltas.append(dt)

    if all_deltas:
        big = pd.concat(all_deltas, ignore_index=True)
    else:
        big = pd.DataFrame()

    # Full cohort (0b comparison)
    full_delta_stats = _stats(big["delta_mag"].to_numpy(dtype=np.float64)) if not big.empty else {"n": 0}

    # Anchor-only cohort (same as restricting to anchor target set = common IDs)
    anchor_cohort = big  # common IDs only by construction

    # Stratify tail
    strat: dict[str, Any] = {}
    if not big.empty:
        dm = big["delta_mag"].to_numpy(dtype=np.float64)
        abs_dm = np.abs(dm)
        tail_cut = float(np.percentile(abs_dm[np.isfinite(abs_dm)], 95)) if np.isfinite(abs_dm).any() else 0.43
        tail = big.loc[abs_dm >= tail_cut].copy()
        strat["tail_threshold_abs_mag"] = tail_cut
        strat["tail_n_epochs"] = int(len(tail))
        # By target magnitude
        tm = pd.to_numeric(tail.get("target_mag"), errors="coerce")
        strat["tail_target_mag_median"] = float(np.nanmedian(tm)) if tm.notna().any() else None
        strat["tail_target_mag_lt_14"] = int((tm < 14).sum()) if tm.notna().any() else 0
        strat["tail_target_mag_ge_14"] = int((tm >= 14).sum()) if tm.notna().any() else 0
        # Bright well-measured: target_mag < 14 and median |delta| per target
        per_target = (
            big.groupby("target_cid")
            .agg(
                abs_delta_p95=("delta_mag", lambda s: float(np.percentile(np.abs(s), 95))),
                target_mag=("target_mag", "first"),
            )
            .reset_index()
        )
        bright = per_target.loc[pd.to_numeric(per_target["target_mag"], errors="coerce") < 14]
        strat["bright_targets_n"] = int(len(bright))
        strat["bright_targets_abs_delta_p95_median"] = (
            float(np.median(bright["abs_delta_p95"])) if len(bright) else None
        )
        strat["bright_targets_abs_delta_p95_max"] = (
            float(np.max(bright["abs_delta_p95"])) if len(bright) else None
        )
        strat["tail_ensemble_changed_fraction"] = float(tail["ensemble_changed"].mean())
        strat["tail_trust_rebuild_counts"] = tail["trust_rebuild"].value_counts().to_dict()
        # Per-target max delta for worst offenders
        worst = (
            big.groupby("target_cid")["delta_mag"]
            .apply(lambda s: float(np.max(np.abs(s))))
            .sort_values(ascending=False)
            .head(10)
        )
        strat["worst_targets_abs_delta_max"] = {str(k): float(v) for k, v in worst.items()}

    # Rebuild-only targets profile
    rb_only_meta: list[dict[str, Any]] = []
    for cid in only_rb:
        m = _target_meta(rb_ps, cid)
        m["target_cid"] = cid
        rb_only_meta.append(m)

    vt_rb = pd.read_csv(rb_ps / "variable_targets.csv", low_memory=False) if (rb_ps / "variable_targets.csv").is_file() else pd.DataFrame()
    vt_an = pd.read_csv(an_ps / "variable_targets.csv", low_memory=False) if (an_ps / "variable_targets.csv").is_file() else pd.DataFrame()

    out: dict[str, Any] = {
        "provenance": {"git_hash": gh, "git_dirty": dirty},
        "rebuild_draft": REBUILD_DRAFT,
        "anchor_snapshot": ANCHOR_SNAPSHOT,
        "cohort": {
            "n_rebuild_lcs": len(rb_ids),
            "n_anchor_lcs": len(an_ids),
            "n_common": len(common),
            "n_rebuild_only": len(only_rb),
            "n_anchor_only": len(only_an),
        },
        "delta_mag_all_common": full_delta_stats,
        "delta_mag_anchor_cohort": _stats(anchor_cohort["delta_mag"].to_numpy(dtype=np.float64))
        if not anchor_cohort.empty
        else {"n": 0},
        "tail_stratification": strat,
        "rebuild_only_targets": rb_only_meta,
        "variable_targets_rows": {"rebuild": int(len(vt_rb)), "anchor": int(len(vt_an))},
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out} common={len(common)} rebuild_only={len(only_rb)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
