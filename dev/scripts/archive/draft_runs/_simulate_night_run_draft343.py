"""_simulate_night_run_draft343.py - Full night run for V0842 Her (draft_343).

Source: D:\\V842_Her | Location: Jirny (id=2) | Setup: NoFilter_60_2
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import AppConfig  # noqa: E402
from night_run import NightRunParams, run_night_pipeline  # noqa: E402

_RUN_CFG_PATH = _ROOT / "_draft343_run_config.json"


def _build_run_config() -> Path:
    """Runtime overrides only - does not modify project config.json."""
    cfg = AppConfig()
    cfg.observer_location_id = 2
    cfg.phase01_plate_scale_arcsec_per_px = 1.3
    cfg.sysrem_enabled = False
    cfg.gs11_dilution_enabled = False
    cfg.psf_photometry_enabled = False
    cfg.comp_max_slope_mmag_hr = 5.0
    base = json.loads((_ROOT / "config.json").read_text(encoding="utf-8"))
    base.update(
        {
            "observer_location_id": 2,
            "phase01_plate_scale_arcsec_per_px": 1.3,
            "sysrem_enabled": False,
            "gs11_dilution_enabled": False,
            "psf_photometry_enabled": False,
            "comp_max_slope_mmag_hr": 5.0,
        }
    )
    _RUN_CFG_PATH.write_text(json.dumps(base, indent=2), encoding="utf-8")
    return _RUN_CFG_PATH


def main() -> int:
    cfg_path = _build_run_config()
    print("Run config:", cfg_path)
    params = NightRunParams(
        source_dir=Path(r"D:\V842_Her"),
        equipment_id=1,
        telescope_id=1,
        location_id=2,
        config_path=cfg_path,
        sysrem_enabled=False,
    )
    t0 = time.time()
    try:
        result = run_night_pipeline(params)
    except Exception:
        traceback.print_exc()
        return 1

    elapsed = time.time() - t0
    print(f"Draft ID: {result.draft_id}")
    print(f"Elapsed: {elapsed:.1f} s")
    print(f"LCs: {result.n_lightcurves}")
    print(f"Success: {result.success}")
    if result.errors:
        print("Errors:")
        for e in result.errors:
            print(f"  - {e}")
    if result.warnings:
        print("Warnings:")
        for w in result.warnings:
            print(f"  - {w}")
    if result.draft_dir:
        print(f"Draft dir: {result.draft_dir}")
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
