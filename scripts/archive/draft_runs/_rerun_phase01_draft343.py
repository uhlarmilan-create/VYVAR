"""Re-run Phase 0+1 for draft_000343 after mag-limit tweak."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from config import AppConfig  # noqa: E402
from photometry_core import run_phase0_and_phase1  # noqa: E402
from ui_aperture_photometry import _load_fwhm  # noqa: E402

DRAFT = _ROOT / "Archive/Drafts/draft_000343"
SETUP = "NoFilter_60_2"
PHOT = DRAFT / "platesolve" / SETUP / "photometry"
PS = DRAFT / "platesolve" / SETUP


def main() -> int:
    cfg = AppConfig()
    print("vsx_variable_targets_mag_limit:", cfg.vsx_variable_targets_mag_limit)
    print("observer_location_name:", cfg.observer_location_name)
    r = run_phase0_and_phase1(
        variable_targets_csv=PHOT / "variable_targets.csv",
        masterstars_csv=PS / "masterstars_full_match.csv",
        per_frame_csv_dir=DRAFT / "detrended_aligned" / "lights" / SETUP,
        output_dir=PHOT,
        fwhm_px=float(_load_fwhm(PS / "MASTERSTAR.fits")),
        plate_scale_arcsec_px=float(cfg.phase01_plate_scale_arcsec_per_px or 1.3) or 1.3,
        cfg=cfg,
    )
    print("phase0+1 result:", r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
