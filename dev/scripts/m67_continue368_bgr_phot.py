#!/usr/bin/env python3
"""Re-run B/G/R photometry on draft_368 after CT target presel fix."""
from __future__ import annotations

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

DRAFT_ID = 368
FIELD_DB = _ROOT / "GAIA_DR3" / "vyvar_gaia_dr3_m67_field.db"
CONFIG_PATH = _ROOT / "config.json"
RESULT_PATH = _ROOT / "tmp" / "m67_bgr_phot368_result.json"


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
    from config import AppConfig  # noqa: E402
    from database import VyvarDatabase  # noqa: E402
    from photometry_core import run_full_photometry_pipeline  # noqa: E402
    from ui_aperture_photometry import _find_phase2a_paths  # noqa: E402

    sys.path.insert(0, str(_ROOT / "scripts"))
    import m67_ct_target_presel as presel  # noqa: E402

    cfg = AppConfig()
    db = VyvarDatabase(cfg.database_path)
    draft_dir = Path(cfg.archive_root) / "Drafts" / f"draft_{DRAFT_ID:06d}"

    orig_cfg = _patch_config()
    try:
        cfg = AppConfig()
        report: dict = {
            "draft_id": DRAFT_ID,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "presel": presel.presel_draft(DRAFT_ID),
        }

        setups = _find_phase2a_paths(cfg, DRAFT_ID, draft_dir_override=None) or {}
        phot_results: dict[str, Any] = {}
        for nm in sorted(setups.keys()):
            if str(nm).split("_")[0] not in ("Blue", "Green", "Red"):
                continue
            p = setups[nm]
            ms_fits = Path(p["masterstar_fits"])
            vt_csv = Path(p["obs_group_dir"]) / "variable_targets.csv"
            ms_csv = Path(p["obs_group_dir"]) / "masterstars_full_match.csv"
            pf_dir = Path(p["per_frame_csv_dir"])
            dt_dir = Path(p["detrended_aligned_dir"])
            out_d = Path(p["output_dir"])
            print(f"Photometry {nm} targets={sum(1 for _ in open(vt_csv))-1}", flush=True)
            t0 = __import__("time").time()
            phot_results[nm] = run_full_photometry_pipeline(
                masterstar_fits_path=ms_fits,
                variable_targets_csv=vt_csv,
                masterstars_csv=ms_csv,
                per_frame_csv_dir=pf_dir,
                detrended_aligned_dir=dt_dir,
                output_dir=out_d,
                cfg=cfg,
                db=db,
                draft_id=DRAFT_ID,
            )
            phot_results[nm]["elapsed_sec"] = __import__("time").time() - t0
            p2a = phot_results[nm].get("phase2a") or {}
            phot_results[nm]["n_lightcurves"] = int(p2a.get("n_lightcurves") or 0)

        report["photometry"] = phot_results
    finally:
        _restore_config(orig_cfg)

    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    ct = draft_dir / "ct_prototype.csv"
    report["ct_prototype_csv"] = str(ct) if ct.is_file() else None
    RESULT_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
