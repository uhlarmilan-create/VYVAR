#!/usr/bin/env python3
"""Build ePSF + re-export proc CSVs for draft 364, then rerun Phase-2A photometry."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import AppConfig  # noqa: E402
from database import VyvarDatabase  # noqa: E402
from pipeline import export_per_frame_catalogs  # noqa: E402
from photometry_core import run_full_photometry_pipeline  # noqa: E402
from psf_photometry import build_epsf_model  # noqa: E402
from ui_aperture_photometry import _find_phase2a_paths  # noqa: E402

DRAFT_ID = 364
CONFIG_PATH = _ROOT / "config.json"
SETUPS = ("Luminance_180_2", "Luminance_60_2")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    orig = bool(json.loads(CONFIG_PATH.read_text(encoding="utf-8")).get("psf_photometry_enabled", False))
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    data["psf_photometry_enabled"] = True
    CONFIG_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    cfg = AppConfig()
    cfg.psf_photometry_enabled = True
    db = VyvarDatabase(cfg.database_path)
    draft = Path(cfg.archive_root) / "Drafts" / f"draft_{DRAFT_ID:06d}"
    row = db.fetch_obs_draft_by_id(DRAFT_ID) or {}
    eq_id = int(row.get("EQUIPMENT_ID") or row.get("ID_EQUIPMENTS") or 3)

    try:
        for setup in SETUPS:
            ps = draft / "platesolve" / setup
            aligned = draft / "detrended_aligned" / "lights" / setup
            ms_fits = ps / "MASTERSTAR.fits"
            ms_csv = ps / "masterstars_full_match.csv"
            print(f"=== ePSF {setup} ===", flush=True)
            epsf = build_epsf_model(
                masterstar_fits_path=ms_fits,
                masterstars_csv_path=ms_csv,
                db=db,
                draft_id=DRAFT_ID,
            )
            print(f"built {epsf}", flush=True)
            per = export_per_frame_catalogs(
                frames_root=aligned,
                platesolve_dir=ps,
                max_catalog_rows=15000,
                catalog_match_max_sep_arcsec=3.0,
                dao_threshold_sigma=3.5,
                dao_fwhm_px=2.5,
                masterstars_csv=ms_csv,
                masterstar_fits=ms_fits,
                use_master_fast_path=True,
                catalog_local_gaia_only=True,
                app_config=cfg,
                draft_id=DRAFT_ID,
                equipment_id=eq_id,
            )
            print(f"export written={per.get('written')}", flush=True)

        setups = _find_phase2a_paths(cfg, DRAFT_ID, draft_dir_override=None)
        for nm, p in sorted((setups or {}).items()):
            if nm not in SETUPS:
                continue
            print(f"=== photometry {nm} ===", flush=True)
            run_full_photometry_pipeline(
                masterstar_fits_path=Path(p["masterstar_fits"]),
                variable_targets_csv=Path(p["obs_group_dir"]) / "variable_targets.csv",
                masterstars_csv=Path(p["obs_group_dir"]) / "masterstars_full_match.csv",
                per_frame_csv_dir=Path(p["per_frame_csv_dir"]),
                detrended_aligned_dir=Path(p["detrended_aligned_dir"]),
                output_dir=Path(p["output_dir"]),
                cfg=cfg,
                db=db,
                draft_id=DRAFT_ID,
                progress_cb=lambda m: print(m, flush=True),
            )
    finally:
        data["psf_photometry_enabled"] = orig
        CONFIG_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print(f"restored psf_photometry_enabled={orig}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
