#!/usr/bin/env python3
"""Build mag14 blind index tiers (fine + wide) and write blind_index_series.json manifest."""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_GAIA = _ROOT / "GAIA_DR3"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_GAIA) not in sys.path:
    sys.path.insert(0, str(_GAIA))

from blind_index_build import build_and_save  # noqa: E402
from vyvar_blind_series import target_density_deg2  # noqa: E402


def _tier_meta(pkl: Path, *, name: str, cell_deg: float, stars_per_cell: int) -> dict:
    with open(pkl, "rb") as f:
        data = pickle.load(f)
    log_min = float(data.get("log_L3_min", 0))
    log_max = float(data.get("log_L3_max", 0))
    return {
        "name": name,
        "path": str(pkl.resolve()),
        "mag_limit": float(data.get("mag_limit", 14)),
        "cell_deg": float(data.get("cell_deg", cell_deg)),
        "stars_per_cell": int(data.get("stars_per_cell", stars_per_cell)),
        "k_neighbors": int(data.get("k_neighbors", 8)),
        "target_density_deg2": float(
            data.get("target_density_deg2", target_density_deg2(cell_deg=cell_deg, stars_per_cell=stars_per_cell))
        ),
        "tolerance": float(data.get("tolerance", 0.002)),
        "hash_dim": int(data.get("hash_dim", 3)),
        "log_L3_min": log_min,
        "log_L3_max": log_max,
        "n_stars": int(data.get("n_stars", 0)),
        "n_triangles": int(data.get("n_triangles", 0)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=_GAIA / "vyvar_gaia_dr3.db")
    ap.add_argument("--skip-wide-build", action="store_true")
    ap.add_argument("--skip-fine-copy", action="store_true")
    args = ap.parse_args()

    fine_src = _GAIA / "gaia_triangles_mag14.pkl"
    if not fine_src.is_file():
        fine_src = _GAIA / "gaia_triangles.pkl"
    fine_out = _GAIA / "gaia_triangles_fine.pkl"
    wide_out = _GAIA / "gaia_triangles_wide.pkl"
    manifest_path = _GAIA / "blind_index_series.json"

    if not args.skip_fine_copy:
        if not fine_src.is_file():
            raise SystemExit(f"fine source missing: {fine_src}")
        shutil.copy2(fine_src, fine_out)
        print(f"fine: {fine_out.name} <- {fine_src.name}")

    if not args.skip_wide_build:
        if not args.db.is_file():
            raise SystemExit(f"Gaia DB missing: {args.db}")
        build_and_save(
            db_path=str(args.db),
            output_pkl=str(wide_out),
            mag_limit=14.0,
            cell_deg=2.0,
            stars_per_cell=95,
        )
    elif not wide_out.is_file():
        raise SystemExit(f"wide index missing: {wide_out}")

    tiers = [
        _tier_meta(fine_out, name="fine", cell_deg=1.0, stars_per_cell=95),
        _tier_meta(wide_out, name="wide", cell_deg=2.0, stars_per_cell=16),
    ]
    manifest = {
        "version": 1,
        "mag_limit": 14,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "tiers": tiers,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
