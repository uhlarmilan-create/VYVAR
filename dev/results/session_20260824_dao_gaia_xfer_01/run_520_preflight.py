#!/usr/bin/env python3
"""DAO-GAIA-XFER-01 W5: draft 520 platesolve/MASTERSTAR preflight only (no photometry)."""
from __future__ import annotations

import json
import logging
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "src_py"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

OUT = Path(__file__).resolve().parent
DRAFT = REPO / "Archive" / "Drafts" / "draft_000520"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from config import AppConfig
    from infolog import end_infolog_session, log_event, start_infolog_session
    from pipeline import astrometry_align_and_build_masterstar

    cfg = AppConfig()
    start_infolog_session(OUT)
    log_event("DAO-GAIA-XFER-01 W5: 520 preflight start (platesolve/MASTERSTAR only)")
    result: dict = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "draft": str(DRAFT),
        "success": False,
        "error": None,
        "output": None,
        "certs": {},
    }
    try:
        outp = astrometry_align_and_build_masterstar(
            archive_path=DRAFT,
            app_config=cfg,
            platesolve_backend="vyvar",
            plate_solve_fov_deg=1.0,
            catalog_match_max_sep_arcsec=25.0,
            saturate_level_fraction=0.999,
            max_catalog_rows=12000,
            dao_threshold_sigma=3.5,
            id_equipment=4,
            draft_id=520,
            catalog_local_gaia_only=True,
            build_masterstar_and_catalogs=True,
            ram_align_and_catalog=True,
            masterstar_selection_pct=10.0,
        )
        result["success"] = True
        result["output"] = {
            "aligned_frames": outp.get("aligned_frames"),
            "input_frames": outp.get("input_frames"),
            "skipped_subgroups": outp.get("skipped_subgroups"),
            "keys": sorted(str(k) for k in outp.keys()),
        }
        log_event("DAO-GAIA-XFER-01 W5: astrometry returned without abort")
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        log_event(f"DAO-GAIA-XFER-01 W5: astrometry raised: {exc}")

    for setup in ("g_60_4", "i_70_4", "r_60_4", "z_90_4"):
        cert_path = DRAFT / "platesolve" / setup / "dao_gaia_calibration.json"
        ms = DRAFT / "platesolve" / setup / "MASTERSTAR.fits"
        rec: dict = {
            "cert_exists": cert_path.is_file(),
            "masterstar_exists": ms.is_file(),
        }
        if cert_path.is_file():
            payload = json.loads(cert_path.read_text(encoding="utf-8"))
            rec["status"] = payload.get("status")
            rec["fail_reason"] = payload.get("fail_reason")
            rec["derived_pass2_center_tol_px"] = payload.get("derived_pass2_center_tol_px")
            rec["derived_forced_seed_centroid_max_px"] = payload.get(
                "derived_forced_seed_centroid_max_px"
            )
            rec["sandbox_params"] = payload.get("sandbox_params")
            rec["gaia_fingerprint"] = payload.get("gaia_fingerprint")
            rec["tol_drift_warn_status"] = (payload.get("tol_drift_warn") or {}).get("status")
            val = payload.get("validation") or {}
            rec["validation_status"] = val.get("status")
            rec["max_regression_pp"] = val.get("max_regression_pp")
            if setup == "g_60_4":
                g_copy = OUT / "g_60_4_dao_gaia_calibration.json"
                g_copy.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                rec["copied_to"] = str(g_copy)
        result["certs"][setup] = rec

    out_json = OUT / "w5_520_preflight.json"
    out_json.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    log_event(f"DAO-GAIA-XFER-01 W5: wrote {out_json}")
    end_infolog_session()
    print(json.dumps({k: result[k] for k in ("success", "error", "certs")}, indent=2, default=str))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
