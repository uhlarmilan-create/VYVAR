"""Smoke: Faza 2A na draft_000298 + DEBUG log pre _catalog_only_merge_frame_flux (jeden Gaia ID)."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import AppConfig  # noqa: E402
from photometry_core import run_phase2a  # noqa: E402
from ui_aperture_photometry import _load_fwhm  # noqa: E402


def main() -> int:
    draft = Path(r"C:\ASTRO\python\VYVAR\Archive\Drafts\draft_000298")
    setup = "NoFilter_60_2"
    ps = draft / "platesolve" / setup
    aligned = draft / "detrended_aligned" / "lights" / setup
    phot = ps / "photometry"
    out = phot / "_cursor_smoke_phase2a_298"
    out.mkdir(parents=True, exist_ok=True)

    class _OnlyCatalogOnlyMerge(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return "[CATALOG_ONLY_MERGE]" in record.getMessage()

    fmt = logging.Formatter("%(levelname)s %(message)s")
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.addFilter(_OnlyCatalogOnlyMerge())
    root.addHandler(sh)

    cfg = AppConfig()
    fw = float(_load_fwhm(ps / "MASTERSTAR.fits"))

    run_phase2a(
        masterstar_fits_path=ps / "MASTERSTAR.fits",
        active_targets_csv=phot / "active_targets.csv",
        comparison_stars_csv=phot / "comparison_stars_per_target.csv",
        per_frame_csv_dir=aligned,
        detrended_aligned_dir=aligned,
        output_dir=out,
        fwhm_px=fw,
        cfg=cfg,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
