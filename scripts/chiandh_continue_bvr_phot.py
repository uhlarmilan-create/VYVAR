#!/usr/bin/env python3
"""Re-run B/V/R photometry on chiandh draft after CT presel."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ["VYVAR_CT_PROTOTYPE"] = "1"

FIELD_DB = _ROOT / "GAIA_DR3" / "vyvar_gaia_dr3_chiandh_field.db"
CONFIG_PATH = _ROOT / "config.json"
CT_FILTERS = ("B", "V", "R")


def _patch_config() -> dict:
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    orig = {"gaia_db_path": data.get("gaia_db_path")}
    data["gaia_db_path"] = str(FIELD_DB.resolve())
    CONFIG_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return orig


def _restore_config(orig: dict) -> None:
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if "gaia_db_path" in orig:
        data["gaia_db_path"] = orig["gaia_db_path"]
    CONFIG_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draft", type=int, required=True)
    ap.add_argument("--result", type=Path, default=_ROOT / "chiandh_bvr_phot_result.json")
    args = ap.parse_args()

    from config import AppConfig  # noqa: E402
    from database import VyvarDatabase  # noqa: E402
    from photometry_core import run_full_photometry_pipeline  # noqa: E402
    from ui_aperture_photometry import _find_phase2a_paths  # noqa: E402

    sys.path.insert(0, str(_ROOT / "scripts"))
    import chiandh_ct_target_presel as presel  # noqa: E402

    draft_id = int(args.draft)
    cfg = AppConfig()
    db = VyvarDatabase(cfg.database_path)
    draft_dir = Path(cfg.archive_root) / "Drafts" / f"draft_{draft_id:06d}"

    orig_cfg = _patch_config()
    try:
        cfg = AppConfig()
        report: dict[str, Any] = {
            "draft_id": draft_id,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "presel": presel.presel_draft(draft_id),
        }
        proto = draft_dir / "ct_prototype.csv"
        if proto.is_file():
            proto.unlink()

        setups = _find_phase2a_paths(cfg, draft_id, draft_dir_override=None) or {}
        phot_results: dict[str, Any] = {}
        for nm in sorted(setups.keys()):
            if str(nm).split("_")[0] not in CT_FILTERS:
                continue
            p = setups[nm]
            phot_results[nm] = run_full_photometry_pipeline(
                masterstar_fits_path=Path(p["masterstar_fits"]),
                variable_targets_csv=Path(p["obs_group_dir"]) / "variable_targets.csv",
                masterstars_csv=Path(p["obs_group_dir"]) / "masterstars_full_match.csv",
                per_frame_csv_dir=Path(p["per_frame_csv_dir"]),
                detrended_aligned_dir=Path(p["detrended_aligned_dir"]),
                output_dir=Path(p["output_dir"]),
                cfg=cfg,
                db=db,
                draft_id=draft_id,
            )
            p2a = phot_results[nm].get("phase2a") or {}
            phot_results[nm] = {
                "n_lightcurves": int(p2a.get("n_lightcurves") or 0),
                "n_targets": int(p2a.get("n_targets") or 0),
            }
        report["photometry"] = phot_results
    finally:
        _restore_config(orig_cfg)

    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    args.result.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
