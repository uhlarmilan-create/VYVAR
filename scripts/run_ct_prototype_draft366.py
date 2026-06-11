"""Run Phase 2A on draft_000366 with VYVAR_CT_PROTOTYPE=1 (diagnostic only)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ["VYVAR_CT_PROTOTYPE"] = "1"

from config import AppConfig  # noqa: E402
from photometry_core import run_phase2a  # noqa: E402
from ui_aperture_photometry import _load_fwhm  # noqa: E402


def main() -> int:
    draft = _ROOT / "Archive" / "Drafts" / "draft_000366"
    setup = "NoFilter_60_2"
    ps = draft / "platesolve" / setup
    aligned = draft / "detrended_aligned" / "lights" / setup
    phot = ps / "photometry"
    ct_csv = draft / "ct_prototype.csv"
    if ct_csv.is_file():
        ct_csv.unlink()

    cfg = AppConfig()
    cfg.save_lightcurve_png = False
    fw = float(_load_fwhm(ps / "MASTERSTAR.fits"))

    run_phase2a(
        masterstar_fits_path=ps / "MASTERSTAR.fits",
        active_targets_csv=phot / "active_targets.csv",
        comparison_stars_csv=phot / "comparison_stars_per_target.csv",
        per_frame_csv_dir=aligned,
        detrended_aligned_dir=aligned,
        output_dir=phot,
        fwhm_px=fw,
        cfg=cfg,
        draft_id=366,
    )
    print(f"Wrote {ct_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
