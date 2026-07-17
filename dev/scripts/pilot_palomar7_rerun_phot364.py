#!/usr/bin/env python3
"""Re-run Phase-2A photometry on draft 364 with PSF flag active in memory."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401  (repo layout: src_py + dev on sys.path)
_ROOT = _bootstrap.REPO_ROOT
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import AppConfig  # noqa: E402
from database import VyvarDatabase  # noqa: E402
from photometry_core import run_full_photometry_pipeline  # noqa: E402
from ui_aperture_photometry import _find_phase2a_paths  # noqa: E402

DRAFT_ID = 364
CONFIG_PATH = _ROOT / "config.json"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    orig = bool(json.loads(CONFIG_PATH.read_text(encoding="utf-8")).get("psf_photometry_enabled", False))
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    data["psf_photometry_enabled"] = True
    CONFIG_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    cfg = AppConfig()
    cfg.psf_photometry_enabled = True
    db = VyvarDatabase(cfg.database_path)
    setups = _find_phase2a_paths(cfg, DRAFT_ID, draft_dir_override=None)

    try:
        for nm, p in sorted((setups or {}).items()):
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
