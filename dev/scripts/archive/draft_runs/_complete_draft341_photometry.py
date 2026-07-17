"""Resume Phase 0+1 + 2A and re-export proc CSVs for draft_000341."""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import AppConfig  # noqa: E402
from photometry_core import run_full_photometry_pipeline  # noqa: E402
from pipeline import AstroPipeline, export_per_frame_catalogs  # noqa: E402
from ui_aperture_photometry import _find_phase2a_paths, _load_fwhm  # noqa: E402

DRAFT_ID = 341
SETUP = "NoFilter_60_2"


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(_ROOT / "complete_draft341.log", encoding="utf-8"),
        ],
    )
    cfg = AppConfig()
    pipeline = AstroPipeline(cfg)
    draft_dir = Path(cfg.archive_root) / "Drafts" / f"draft_{DRAFT_ID:06d}"
    setups = _find_phase2a_paths(cfg, DRAFT_ID)
    if SETUP not in setups:
        logging.error("Setup %s not in %s", SETUP, list(setups.keys()))
        return 1
    p = setups[SETUP]
    ps = Path(p["obs_group_dir"])
    aligned = Path(p["detrended_aligned_dir"])
    phot = Path(p["output_dir"])
    ms_fits = Path(p["masterstar_fits"])
    ms_csv = ps / "masterstars_full_match.csv"
    vt_csv = ps / "variable_targets.csv"

    t0 = time.time()
    logging.info("=== Phase 0+1 + 2A draft %d ===", DRAFT_ID)
    phot_result = run_full_photometry_pipeline(
        masterstar_fits_path=ms_fits,
        variable_targets_csv=vt_csv,
        masterstars_csv=ms_csv,
        per_frame_csv_dir=aligned,
        detrended_aligned_dir=aligned,
        output_dir=phot,
        cfg=cfg,
        db=pipeline.db,
        draft_id=DRAFT_ID,
        progress_cb=lambda m: logging.info("[phot] %s", m),
    )
    p2a = phot_result.get("phase2a") or {}
    logging.info(
        "Phase2A done in %.1fs: n_lc=%s n_frames=%s",
        time.time() - t0,
        p2a.get("n_lightcurves"),
        p2a.get("n_frames"),
    )

    logging.info("=== Re-export per-frame catalogs (targeted ePSF) ===")
    t1 = time.time()
    per = export_per_frame_catalogs(
        frames_root=aligned,
        platesolve_dir=ps,
        max_catalog_rows=12000,
        catalog_match_max_sep_arcsec=10.0,
        saturate_level_fraction=0.95,
        faintest_mag_limit=18.0,
        dao_threshold_sigma=3.5,
        masterstars_csv=ms_csv,
        masterstar_fits=ms_fits,
        use_master_fast_path=True,
        app_config=cfg,
        draft_id=DRAFT_ID,
        equipment_id=1,
    )
    logging.info("Re-export written=%s in %.1fs", per.get("written"), time.time() - t1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
