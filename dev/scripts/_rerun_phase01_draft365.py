#!/usr/bin/env python3
"""Re-run Phase 0+1 comp selection for draft_000365."""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import AppConfig  # noqa: E402
from photometry_core import run_phase0_and_phase1  # noqa: E402
from ui_aperture_photometry import _load_fwhm  # noqa: E402

DRAFT = _ROOT / "Archive/Drafts/draft_000365"
SETUP = "NoFilter_60_2"
PHOT = DRAFT / "platesolve" / SETUP / "photometry"
PS = DRAFT / "platesolve" / SETUP


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    cfg = AppConfig()
    t0 = time.time()
    run_phase0_and_phase1(
        variable_targets_csv=PS / "variable_targets.csv",
        masterstars_csv=PS / "masterstars_full_match.csv",
        per_frame_csv_dir=DRAFT / "detrended_aligned/lights" / SETUP,
        output_dir=PHOT,
        fwhm_px=float(_load_fwhm(PS / "MASTERSTAR.fits")),
        plate_scale_arcsec_px=float(cfg.phase01_plate_scale_arcsec_per_px or 1.3) or 1.3,
        cfg=cfg,
    )
    print(f"elapsed_s={time.time() - t0:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
