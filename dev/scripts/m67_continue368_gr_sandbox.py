#!/usr/bin/env python3
"""Sandbox Green/Red photometry re-run with phase01_ct_min_comp override (no config edit)."""
from __future__ import annotations

import json
import os
import shutil
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
RESULT_PATH = _ROOT / "tmp" / "m67_gr_sandbox_min5_result.json"
GR_FILTERS = ("Green", "Red")
DEFAULT_MIN_COMP = 5


def _patch_gaia_only() -> dict:
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    orig = {"gaia_db_path": data.get("gaia_db_path")}
    data["gaia_db_path"] = str(FIELD_DB.resolve())
    CONFIG_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return orig


def _restore_gaia(orig: dict) -> None:
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if "gaia_db_path" in orig:
        data["gaia_db_path"] = orig["gaia_db_path"]
    CONFIG_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _summarize_gr(draft_dir: Path, min_comp: int) -> dict:
    import pandas as pd
    from photometry_core import should_apply_color_term  # noqa: E402

    out: dict[str, Any] = {"min_comp_for_ct": min_comp, "by_filter": {}}
    proto = draft_dir / "ct_prototype.csv"
    if not proto.is_file():
        return out
    df = pd.read_csv(proto, low_memory=False, dtype={"catalog_id": str})
    for flt in GR_FILTERS:
        sub = df[df["obs_group"].astype(str) == flt].copy()
        ct_ok_ids: set[str] = set()
        for og_dir in sorted((draft_dir / "platesolve").iterdir()):
            if not og_dir.is_dir() or og_dir.name.split("_")[0] != flt:
                continue
            lc_dir = og_dir / "photometry" / "lightcurves"
            if not lc_dir.is_dir():
                continue
            for lc in lc_dir.glob("lightcurve_*.csv"):
                r0 = pd.read_csv(lc, nrows=1, low_memory=False).iloc[0]
                if str(r0.get("ct_ok", "")).strip().lower() in ("true", "1", "yes"):
                    ct_ok_ids.add(lc.stem.replace("lightcurve_", "", 1))
        out["by_filter"][flt] = {
            "ct_ok_unique_targets": len(ct_ok_ids),
            "ct_ok_catalog_ids": sorted(ct_ok_ids),
            "proto_gate_pass": int(sub["gate_would_pass"].sum()) if "gate_would_pass" in sub.columns else None,
        }
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from config import AppConfig  # noqa: E402
    from database import VyvarDatabase  # noqa: E402
    from photometry_core import run_full_photometry_pipeline  # noqa: E402
    from ui_aperture_photometry import _find_phase2a_paths  # noqa: E402

    min_comp = int(os.environ.get("VYVAR_CT_MIN_COMP", DEFAULT_MIN_COMP))

    from config import AppConfig as _AC  # noqa: E402

    cfg0 = _AC()
    draft_dir = Path(cfg0.archive_root) / "Drafts" / f"draft_{DRAFT_ID:06d}"
    proto = draft_dir / "ct_prototype.csv"
    backup = draft_dir / "ct_prototype_min7_backup.csv"
    if proto.is_file() and not backup.is_file():
        shutil.copy2(proto, backup)

    orig_gaia = _patch_gaia_only()
    report: dict = {
        "draft_id": DRAFT_ID,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "phase01_ct_min_comp_override": min_comp,
        "max_stderr_ratio": 0.5,
        "filters": list(GR_FILTERS),
    }
    try:
        cfg = AppConfig()
        cfg.phase01_ct_min_comp = int(min_comp)
        db = VyvarDatabase(cfg.database_path)
        if proto.is_file():
            proto.unlink()

        setups = _find_phase2a_paths(cfg, DRAFT_ID, draft_dir_override=None) or {}
        phot_results: dict[str, Any] = {}
        for nm in sorted(setups.keys()):
            if str(nm).split("_")[0] not in GR_FILTERS:
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
                draft_id=DRAFT_ID,
            )
            p2a = phot_results[nm].get("phase2a") or {}
            phot_results[nm] = {
                "n_lightcurves": int(p2a.get("n_lightcurves") or 0),
                "n_targets": int(p2a.get("n_targets") or 0),
            }
        report["photometry"] = phot_results
        report["summary"] = _summarize_gr(draft_dir, min_comp)
    finally:
        _restore_gaia(orig_gaia)

    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    RESULT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
