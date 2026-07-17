"""Draft 367 crowding characterization (VY_FWHM_GAUSS core; read-only diagnostic)."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import AppConfig
from crowding_index import compute_crowding_index
from database import VyvarDatabase, get_gaia_db_max_g_mag

DRAFT_ID = 367
DEFAULT_DRAFT = _ROOT / "Archive" / "Drafts" / "draft_000367"
SETUPS = (
    "Blue_60_2",
    "Blue_180_2",
    "Green_60_2",
    "Green_180_2",
    "Red_60_2",
    "Red_180_2",
)
RICHEST_SETUP = "Red_180_2"


def _nn_buckets(df: pd.DataFrame) -> dict[str, int]:
    if df.empty or "nn_dist_fwhm" not in df.columns:
        return {}
    nn = pd.to_numeric(df["nn_dist_fwhm"], errors="coerce")
    nn = nn[np.isfinite(nn)]

    def cnt(lo: float, hi: float | None) -> int:
        if hi is None:
            return int((nn > lo).sum())
        return int(((nn >= lo) & (nn < hi)).sum())

    n_blend = int(df["is_blended"].astype(bool).sum()) if "is_blended" in df.columns else 0
    return {
        "n_total": int(len(df)),
        "n_is_blended_true": n_blend,
        "hard_lt_1.0": cnt(-1.0, 1.0),
        "sep_1.0_1.5": cnt(1.0, 1.5),
        "sep_1.5_2.0": cnt(1.5, 2.0),
        "gt_2.0": cnt(2.0, None),
    }


def analyze_setup(
    draft_dir: Path,
    db: VyvarDatabase,
    setup: str,
    *,
    gaia_max: float,
) -> dict:
    res, targets = compute_crowding_index(
        draft_dir, setup, db, DRAFT_ID, gaia_db_max_g=gaia_max, lc_star_set=True
    )
    buckets = _nn_buckets(targets)
    filt = setup.split("_")[0]
    return {
        "setup": setup,
        "filter": filt,
        "fwhm_px": res.get("fwhm_px"),
        "plate_scale_arcsec_px": res.get("plate_scale_arcsec_px"),
        "gaia_density_per_arcmin2": res.get("gaia_density_per_arcmin2"),
        "blend_frac_1fwhm": res.get("blend_frac_1fwhm"),
        "blend_frac_2fwhm": res.get("blend_frac_2fwhm"),
        "nn_buckets": buckets,
    }


def crowding_verdict(richest: dict) -> str:
    b = richest.get("nn_buckets") or {}
    n_blend = int(b.get("n_is_blended_true", 0))
    hard = int(b.get("hard_lt_1.0", 0))
    if n_blend >= 20 and hard >= 10:
        return "PROCEED_2B_CANDIDATE"
    if n_blend >= 10 or hard >= 5:
        return "MODERATE_BLEND_POPULATION"
    return "SPARSE"


def analyze_draft_367(
    draft_dir: Path | None = None,
    db: VyvarDatabase | None = None,
) -> dict:
    draft_dir = Path(draft_dir or DEFAULT_DRAFT)
    cfg = AppConfig()
    db = db or VyvarDatabase(cfg.database_path)
    gaia_max = float(get_gaia_db_max_g_mag(cfg.gaia_db_path))
    rows = []
    for setup in SETUPS:
        ps = draft_dir / "platesolve" / setup / "MASTERSTAR.fits"
        if not ps.is_file():
            continue
        rows.append(analyze_setup(draft_dir, db, setup, gaia_max=gaia_max))
    by_filter: dict[str, list[dict]] = {}
    for r in rows:
        by_filter.setdefault(r["filter"], []).append(r)
    richest_per_filter = {
        f: max(rs, key=lambda x: (x.get("blend_frac_2fwhm") or 0.0)) for f, rs in by_filter.items()
    }
    richest = next((r for r in rows if r["setup"] == RICHEST_SETUP), rows[0] if rows else {})
    verdict = crowding_verdict(richest)
    return {
        "draft_id": DRAFT_ID,
        "draft_dir": str(draft_dir),
        "setups": rows,
        "richest_per_filter": richest_per_filter,
        "richest_setup": RICHEST_SETUP,
        "richest": richest,
        "crowding_verdict": verdict,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Draft 367 crowding audit")
    ap.add_argument("--out", type=Path, default=_ROOT / "tmp" / "crowding_audit_367.json")
    args = ap.parse_args()
    out = analyze_draft_367()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="ascii")
    r = out["richest"]
    b = r.get("nn_buckets") or {}
    print(f"richest: {out['richest_setup']}")
    print(f"is_blended={b.get('n_is_blended_true')} hard_lt_1.0={b.get('hard_lt_1.0')}")
    print(f"blend@1={r.get('blend_frac_1fwhm')} blend@2={r.get('blend_frac_2fwhm')}")
    print(f"verdict: {out['crowding_verdict']}")


if __name__ == "__main__":
    main()
