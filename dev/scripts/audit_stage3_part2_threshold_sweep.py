#!/usr/bin/env python3
"""Audit Stage 3 Part 2: DAO threshold N sweep on scratch rebuild."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "dev" / "scripts"))
import _bootstrap  # noqa: E402,F401

sys.path.insert(0, str(_bootstrap.REPO_ROOT / "src_py"))

from config import AppConfig  # noqa: E402
from invariants_runtime import dao_only_fraction_from_masterstars  # noqa: E402
from pipeline import (  # noqa: E402
    detect_stars_match_master_reference,
    _dao_convolved_background_rms_adu,
    _mean_bin2d_for_dao,
    _dao_auto_binning_factor,
)
from photometry_core import _resolve_git_provenance  # noqa: E402


def _sweep_row(
    *,
    ms_fits: Path,
    ms_csv: Path,
    n_sigma: float,
) -> dict[str, Any]:
    with fits.open(ms_fits, memmap=False) as hd:
        data = np.asarray(hd[0].data, dtype=np.float32)
        hdr = hd[0].header
    mdf = pd.read_csv(ms_csv, low_memory=False)
    tbl, meta = detect_stars_match_master_reference(
        data,
        hdr,
        mdf,
        dao_threshold_sigma=float(n_sigma),
        max_catalog_rows=12000,
    )
    pass1 = int(meta.get("n_detected_dao") or len(tbl) if tbl is not None else 0)
    full = pd.read_csv(ms_csv, low_memory=False)
    dao_frac = float(dao_only_fraction_from_masterstars(full))
    out = {"N": float(n_sigma), "pass1_dao_meta": pass1, "dao_only_fraction_baseline_csv": dao_frac}
    if "g_mag" in full.columns and "match_status" in full.columns:
        g = pd.to_numeric(full["g_mag"], errors="coerce")
        st = full["match_status"].astype(str).str.upper()
        dao_only = ~st.isin({"GAIA_MATCHED", "FORCED_APERTURE"})
        out["dao_only_below_g16"] = int((dao_only & (g < 16.0)).sum())
        out["dao_only_g16_175"] = int((dao_only & (g >= 16.0) & (g <= 17.5)).sum())
        out["dao_only_beyond_g175"] = int((dao_only & (g > 17.5)).sum())
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft-id", type=int, default=499)
    parser.add_argument("--setup", default="NoFilter_60_2")
    parser.add_argument("--out", type=Path, default=REPO / "tmp" / "audit_stage3_part2_sweep.json")
    args = parser.parse_args()

    cfg = AppConfig()
    draft = Path(cfg.archive_root) / "Drafts" / f"draft_{args.draft_id:06d}"
    ps = draft / "platesolve" / args.setup
    ms_fits = ps / "MASTERSTAR.fits"
    ms_csv = ps / "masterstars_full_match.csv"
    if not ms_fits.is_file():
        raise FileNotFoundError(ms_fits)

    gh, dirty, _ = _resolve_git_provenance()
    n_values = np.round(np.arange(2.5, 6.01, 0.25), 2)
    rows = [_sweep_row(ms_fits=ms_fits, ms_csv=ms_csv, n_sigma=float(n)) for n in n_values]

    payload = {
        "provenance": {"git_hash": gh, "git_dirty": dirty},
        "draft_id": args.draft_id,
        "sweep": rows,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
