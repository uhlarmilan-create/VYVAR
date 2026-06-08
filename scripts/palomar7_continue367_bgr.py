#!/usr/bin/env python3
"""Continue draft_000367 B/G/R: per-group preprocess/platesolve/photometry + CT prototype."""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ["VYVAR_CT_PROTOTYPE"] = "1"

FIELD_DB = _ROOT / "GAIA_DR3" / "vyvar_gaia_dr3_pal7_field.db"
CONFIG_PATH = _ROOT / "config.json"
DRAFT_ID = 367
RESULT_PATH = _ROOT / "palomar7_bgr_continue367_result.json"


def _set_gaia_db(path: Path) -> dict:
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    orig = {"gaia_db_path": data.get("gaia_db_path")}
    data["gaia_db_path"] = str(path.resolve())
    CONFIG_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return orig


def _restore_gaia(orig: dict) -> None:
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if "gaia_db_path" in orig:
        data["gaia_db_path"] = orig["gaia_db_path"]
    CONFIG_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _restore_psf_flag(original: bool) -> None:
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    data["psf_photometry_enabled"] = bool(original)
    CONFIG_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _enable_psf_flag() -> bool:
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    original = bool(data.get("psf_photometry_enabled", False))
    data["psf_photometry_enabled"] = True
    CONFIG_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return original


def _progress(i: int, total: int, msg: str) -> None:
    line = f"[{i}/{total}] {msg}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", errors="replace").decode("ascii"), flush=True)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from config import AppConfig  # noqa: E402
    from database import VyvarDatabase  # noqa: E402
    from pipeline import (  # noqa: E402
        astrometry_align_and_build_masterstar,
        preprocess_calibrated_to_processed,
        resolve_preprocess_target_coordinates,
    )
    from photometry_core import run_full_photometry_pipeline  # noqa: E402
    from ui_aperture_photometry import _find_phase2a_paths  # noqa: E402

    orig_cfg = _set_gaia_db(FIELD_DB)
    cfg = AppConfig()
    cfg.save_lightcurve_png = False
    db = VyvarDatabase(cfg.database_path)
    draft_dir = Path(cfg.archive_root) / "Drafts" / f"draft_{DRAFT_ID:06d}"
    ap_root = draft_dir.resolve()
    cal = ap_root / "calibrated" / "lights"
    proc = ap_root / "processed" / "lights"

    report: dict = {
        "draft_id": DRAFT_ID,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "field_db": str(FIELD_DB),
    }

    db.set_obs_draft_masterstar_source_path(DRAFT_ID, None)
    ira, ide = resolve_preprocess_target_coordinates(
        db=db, draft_id=DRAFT_ID, ui_ra_deg=272.684, ui_dec_deg=-7.208
    )
    report["pointing"] = {"ra": ira, "dec": ide}

    t0 = time.time()
    preprocess_calibrated_to_processed(
        calibrated_root=cal,
        processed_root=proc,
        inject_pointing_ra_deg=ira,
        inject_pointing_dec_deg=ide,
        inject_pointing_only_if_missing=False,
        db=db,
        draft_id=DRAFT_ID,
        app_config=cfg,
        progress_cb=_progress,
    )
    report["preprocess_sec"] = time.time() - t0
    report["processed_counts"] = {
        g.name: {
            "proc_fits": len(list(g.glob("proc_*.fits"))),
            "proc_csv": len(list(g.glob("proc_*.csv"))),
        }
        for g in sorted(proc.iterdir())
        if g.is_dir()
    }

    row = db.fetch_obs_draft_by_id(DRAFT_ID) or {}
    eq_id = int(row.get("EQUIPMENT_ID") or row.get("ID_EQUIPMENTS") or 3)
    cfg.sips_dao_fwhm_px = 2.5
    cfg.sips_dao_threshold_sigma = 3.5

    psf_orig = _enable_psf_flag()
    try:
        t0 = time.time()
        ps_out = astrometry_align_and_build_masterstar(
            archive_path=ap_root,
            plate_solve_fov_deg=0.55,
            min_detected_stars=200,
            max_detected_stars=2000,
            max_control_points=200,
            catalog_match_max_sep_arcsec=3.0,
            max_catalog_rows=15000,
            dao_threshold_sigma=3.5,
            id_equipment=eq_id,
            draft_id=DRAFT_ID,
            catalog_local_gaia_only=True,
            build_masterstar_and_catalogs=True,
            masterstar_candidate_paths=None,
            masterstar_selection_pct=10.0,
            ram_align_and_catalog=True,
            app_config=cfg,
            progress_cb=_progress,
        )
        report["platesolve_sec"] = time.time() - t0
        if isinstance(ps_out, dict) and ps_out.get("error"):
            report["platesolve_error"] = ps_out.get("error")
            return 1

        setups = _find_phase2a_paths(cfg, DRAFT_ID, draft_dir_override=None)
        report["setups"] = list(setups.keys()) if setups else []
        phot_results = {}
        for nm, p in sorted((setups or {}).items()):
            if not str(nm).endswith("_60_2"):
                phot_results[nm] = {"skipped": "non-60s group (metadata artefact)"}
                continue
            t1 = time.time()
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
                progress_cb=lambda m, _n=nm: print(f"[phot {_n}] {m}", flush=True),
            )
            p2a = pr.get("phase2a") or {}
            phot_results[nm] = {
                "sec": time.time() - t1,
                "n_lightcurves": int(p2a.get("n_lightcurves") or 0),
            }
        report["photometry"] = phot_results
        ct = draft_dir / "ct_prototype.csv"
        report["ct_prototype_csv"] = str(ct) if ct.is_file() else None
    finally:
        _restore_psf_flag(psf_orig)
        _restore_gaia(orig_cfg)

    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    RESULT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
