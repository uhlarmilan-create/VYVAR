#!/usr/bin/env python3
"""Run photometry on draft_367 *_180_2 B/G/R setups (append to ct_prototype)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401  (repo layout: src_py + dev on sys.path)
_ROOT = _bootstrap.REPO_ROOT
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ["VYVAR_CT_PROTOTYPE"] = "1"
FIELD_DB = _ROOT / "GAIA_DR3" / "vyvar_gaia_dr3_pal7_field.db"
CONFIG_PATH = _ROOT / "config.json"
DRAFT_ID = 367


def main() -> int:
    from config import AppConfig  # noqa: E402
    from database import VyvarDatabase  # noqa: E402
    from photometry_core import run_full_photometry_pipeline  # noqa: E402
    from ui_aperture_photometry import _find_phase2a_paths  # noqa: E402

    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    orig = data.get("gaia_db_path")
    data["gaia_db_path"] = str(FIELD_DB.resolve())
    CONFIG_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    cfg = AppConfig()
    cfg.save_lightcurve_png = False
    db = VyvarDatabase(cfg.database_path)
    setups = _find_phase2a_paths(cfg, DRAFT_ID)
    out = {}
    for nm in ("Blue_180_2", "Green_180_2", "Red_180_2"):
        p = setups.get(nm)
        if not p:
            out[nm] = "missing"
            continue
        pr = run_full_photometry_pipeline(
            masterstar_fits_path=Path(p["masterstar_fits"]),
            variable_targets_csv=Path(p["obs_group_dir"]) / "variable_targets.csv",
            masterstars_csv=Path(p["obs_group_dir"]) / "masterstars_full_match.csv",
            per_frame_csv_dir=Path(p["per_frame_csv_dir"]),
            detrended_aligned_dir=Path(p["detrended_aligned_dir"]),
            output_dir=Path(p["output_dir"]),
            cfg=cfg,
            db=db,
            draft_id=DRAFT_ID,
        )
        out[nm] = pr.get("phase2a", {})
    data["gaia_db_path"] = orig
    CONFIG_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
