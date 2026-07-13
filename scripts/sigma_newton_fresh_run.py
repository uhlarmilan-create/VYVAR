#!/usr/bin/env python3
"""SIGMA-NEWTON fresh baseline on regenerated draft_426 (production_lc_err)."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import AppConfig  # noqa: E402
from scripts.bingain_fix_validate import resolve_archive_root  # noqa: E402
from scripts.provenance_guard import add_allow_unstamped_arg, assert_stamped, stamp_output_meta  # noqa: E402
from scripts.sigma_newton_run import (  # noqa: E402
    DRAFT_ID,
    _list_check_star_ids,
    _production_chi2_from_payload,
    run_star_gate,
)
from scripts.sparse_comp_diag import V0611_CID  # noqa: E402
from tests.photometry_sha import PHOTOMETRY_QC_COLS_LC, PHOTOMETRY_SCIENCE_COLS_LC, TOL_SCIENCE  # noqa: E402

SETUPS = ("g_60_4", "i_70_4", "r_60_4", "z_90_4")
EVIDENCE_NAME = "draft_000426_stale_20260626"
FORENSIC_REF = {
    "i_70_4": {"V0611_chi2": 2.131, "SS_CAM_chi2": 24.9},
}


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_ROOT, text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _stamp(payload: dict[str, Any]) -> dict[str, Any]:
    payload["git_head"] = _git_head()
    payload["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    return payload


def compare_shared_epochs(stale_lc: Path, fresh_lc: Path) -> dict[str, Any]:
    if not stale_lc.is_file() or not fresh_lc.is_file():
        return {"status": "missing"}
    da = pd.read_csv(stale_lc, low_memory=False)
    db = pd.read_csv(fresh_lc, low_memory=False)
    if "source_file" not in da.columns or "source_file" not in db.columns:
        return {"status": "no_source_file"}
    da["_sf"] = da["source_file"].astype(str)
    db["_sf"] = db["source_file"].astype(str)
    merged = da.merge(db, on="_sf", suffixes=("_a", "_b"), how="inner")
    if merged.empty:
        return {"status": "no_overlap", "n_shared": 0}
    max_delta: dict[str, float] = {}
    science_ok = True
    skip = PHOTOMETRY_QC_COLS_LC | {"source_file", "flag", "method", "err_method", "_sf"}
    cols_a = {c[:-2] for c in merged.columns if c.endswith("_a")}
    for col in sorted(cols_a):
        if col in skip or col.startswith("err"):
            continue
        ca, cb = f"{col}_a", f"{col}_b"
        if ca not in merged.columns or cb not in merged.columns:
            continue
        if merged[ca].dtype == bool or merged[cb].dtype == bool:
            if not merged[ca].equals(merged[cb]):
                science_ok = False
            continue
        na = pd.to_numeric(merged[ca], errors="coerce")
        nb = pd.to_numeric(merged[cb], errors="coerce")
        if not (na.notna().any() and nb.notna().any()):
            continue
        delta = float(np.nanmax(np.abs(na - nb))) if len(na) else 0.0
        if delta > 0:
            max_delta[col] = delta
        is_science = col.lower() in PHOTOMETRY_SCIENCE_COLS_LC or col.lower().startswith("mag")
        if is_science and delta > TOL_SCIENCE:
            science_ok = False
    idx_a = set(da["_sf"].tolist())
    idx_b = set(db["_sf"].tolist())
    return {
        "status": "ok",
        "n_shared": int(len(merged)),
        "n_only_stale": len(idx_a - idx_b),
        "n_only_fresh": len(idx_b - idx_a),
        "only_fresh_files": sorted(idx_b - idx_a)[:10],
        "science_ok": science_ok,
        "max_delta": max_delta,
    }


def _all_lc_star_ids(lc_dir: Path) -> list[str]:
    return [p.stem.replace("lightcurve_", "", 1) for p in sorted(lc_dir.glob("lightcurve_*.csv"))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive-root", default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("tmp/sigma_newton_fresh"))
    ap.add_argument("--evidence-name", default=EVIDENCE_NAME)
    add_allow_unstamped_arg(ap)
    args = ap.parse_args()

    cfg = AppConfig()
    archive_root = resolve_archive_root(args.archive_root, cfg=cfg)
    cfg.archive_root = archive_root
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    evidence = archive_root / "evidence" / args.evidence_name

    summary_setups: dict[str, Any] = {}
    science_all: dict[str, dict[str, Any]] = {}
    guard_flags: list[dict[str, Any]] = []

    for setup in SETUPS:
        phot = archive_root / "Drafts" / f"draft_{DRAFT_ID:06d}" / "platesolve" / setup / "photometry"
        if not (phot / "pipeline_meta.json").is_file():
            raise SystemExit(f"ERROR: missing regenerated photometry for {setup}: {phot}")
        guard = assert_stamped(phot, draft_id=DRAFT_ID, setup=setup, allow_unstamped=args.allow_unstamped)
        guard_flags.append({"setup": setup, **guard})
        lc_dir = phot / "lightcurves"
        setup_dir = out_dir / setup
        targets = list(dict.fromkeys([V0611_CID] + _list_check_star_ids(lc_dir) + _all_lc_star_ids(lc_dir)))
        targets = [t for t in targets if (lc_dir / f"lightcurve_{t}.csv").is_file()]
        per_star_table: list[dict[str, Any]] = []
        science_setup: dict[str, Any] = {}

        for cid in targets:
            pl = run_star_gate(
                phot_dir=phot, setup=setup, catalog_id=cid, out_dir=setup_dir, cfg=cfg,
                is_v0611=(cid == V0611_CID),
            )
            prod = _production_chi2_from_payload(pl)
            med = pl.get("err_decomposition", {}).get("medians", {})
            em2 = med.get("err_lc_mag2")
            err_mag = (
                math.sqrt(float(em2))
                if em2 is not None and math.isfinite(float(em2)) and float(em2) >= 0
                else None
            )
            row = {
                "catalog_id": cid,
                "is_v0611": cid == V0611_CID,
                "err_median_mag": err_mag,
                "photon_share_median": med.get("photon_share"),
                "background_share_median": med.get("background_share"),
                "ensemble_share_median": med.get("ensemble_share"),
                "chi2_dof": prod.get("chi2_dof") if prod else None,
                "ci_lo": prod.get("chi2_dof_ci_lo") if prod else None,
                "ci_hi": prod.get("chi2_dof_ci_hi") if prod else None,
                "n_frames": prod.get("n_frames") if prod else None,
            }
            per_star_table.append(row)
            stale_p = evidence / "platesolve" / setup / "photometry" / "lightcurves" / f"lightcurve_{cid}.csv"
            fresh_p = lc_dir / f"lightcurve_{cid}.csv"
            science_setup[cid] = compare_shared_epochs(stale_p, fresh_p)

        summary_setups[setup] = {
            "per_star_table": per_star_table,
            "v0611": next((r for r in per_star_table if r["is_v0611"]), None),
            "forensic_delta": _forensic_delta(setup, per_star_table),
        }
        science_all[setup] = science_setup

    payload = _stamp(
        {
            "task": "SIGMA-NEWTON-FRESH",
            "draft_id": DRAFT_ID,
            "setups": summary_setups,
            "science_compare_shared_epochs": science_all,
            "baseline_note": "Replaces invalidated SIGMA-NEWTON N1 (stale draft_426 archive).",
        }
    )
    payload = stamp_output_meta(
        payload,
        {
            "stamped": all(g.get("stamped") for g in guard_flags),
            "provenance_unstamped": any(g.get("provenance_unstamped") for g in guard_flags),
            "git_hash": guard_flags[0].get("git_hash") if guard_flags else None,
        },
    )
    payload["provenance_guard_details"] = guard_flags
    summary_path = out_dir / "sigma_newton_fresh_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_baseline_table_figure(summary_setups, out_dir / "baseline_chi2_table.png")
    print(summary_path)
    return 0


def _forensic_delta(setup: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    ref = FORENSIC_REF.get(setup, {})
    v = next((r for r in rows if r.get("is_v0611")), None)
    out: dict[str, Any] = {}
    if v and ref.get("V0611_chi2") is not None:
        out["v0611_chi2_forensic"] = ref["V0611_chi2"]
        out["v0611_chi2_fresh"] = v.get("chi2_dof")
        out["v0611_chi2_delta"] = (
            float(v["chi2_dof"]) - float(ref["V0611_chi2"])
            if v.get("chi2_dof") is not None
            else None
        )
    return out


def _write_baseline_table_figure(setups: dict[str, Any], path: Path) -> None:
    rows: list[list[str]] = []
    for setup, data in setups.items():
        for r in data.get("per_star_table", []):
            chi = r.get("chi2_dof")
            rows.append(
                [
                    setup,
                    str(r.get("catalog_id", ""))[-6:],
                    f"{r.get('err_median_mag', float('nan')):.4f}" if r.get("err_median_mag") else "nan",
                    f"{chi:.2f}" if chi is not None and math.isfinite(float(chi)) else "nan",
                ]
            )
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(10, max(3, 0.35 * len(rows))))
    ax.axis("off")
    tbl = ax.table(
        cellText=rows,
        colLabels=["setup", "star", "err_mag", "chi2"],
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
