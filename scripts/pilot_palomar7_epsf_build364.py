#!/usr/bin/env python3
"""Part 2b: attempt ePSF build on draft 364 with funnel reporting."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from config import AppConfig  # noqa: E402
from database import VyvarDatabase  # noqa: E402
from psf_photometry import build_epsf_model  # noqa: E402

DRAFT_ID = 364
SETUPS = ("Luminance_180_2", "Luminance_60_2")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cfg = AppConfig()
    db = VyvarDatabase(cfg.database_path)
    draft = Path(cfg.archive_root) / "Drafts" / f"draft_{DRAFT_ID:06d}"
    results = {}
    for setup in SETUPS:
        ps = draft / "platesolve" / setup
        ms = ps / "MASTERSTAR.fits"
        csv = ps / "masterstars_full_match.csv"
        print(f"\n=== build_epsf_model {setup} ===", flush=True)
        try:
            out = build_epsf_model(
                masterstar_fits_path=ms,
                masterstars_csv_path=csv,
                db=db,
                draft_id=DRAFT_ID,
            )
            results[setup] = {"status": "built", "path": str(out)}
            print(f"BUILT: {out}", flush=True)
        except ValueError as exc:
            results[setup] = {"status": "failed", "error": str(exc)}
            print(f"FAILED cleanly: {exc}", flush=True)
        except Exception as exc:
            results[setup] = {"status": "error", "error": repr(exc)}
            print(f"ERROR: {exc!r}", flush=True)
    Path(_ROOT / "tmp" / "pilot_palomar7_epsf_build364.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
