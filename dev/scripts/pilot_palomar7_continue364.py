#!/usr/bin/env python3
"""Continue Phase C on draft 364 after MASTERSTAR cross-group path failure."""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import AppConfig  # noqa: E402
from database import VyvarDatabase  # noqa: E402
from pipeline import (  # noqa: E402
    preprocess_calibrated_to_processed,
    resolve_preprocess_target_coordinates,
    astrometry_align_and_build_masterstar,
)
from photometry_core import run_full_photometry_pipeline  # noqa: E402
from ui_aperture_photometry import _find_phase2a_paths  # noqa: E402

DRAFT_ID = 364
CONFIG_PATH = _ROOT / "config.json"
RESULT_PATH = _ROOT / "tmp" / "pilot_palomar7_continue364_result.json"


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
    cfg = AppConfig()
    db = VyvarDatabase(cfg.database_path)
    draft_dir = Path(cfg.archive_root) / "Drafts" / f"draft_{DRAFT_ID:06d}"
    ap_root = draft_dir.resolve()
    cal = ap_root / "calibrated" / "lights"
    proc = ap_root / "processed" / "lights"

    report: dict = {"draft_id": DRAFT_ID, "started_utc": datetime.now(timezone.utc).isoformat()}

    # Clear explicit MASTERSTAR source so each obs-group picks its own top frame.
    db.set_obs_draft_masterstar_source_path(DRAFT_ID, None)

    ira, ide = resolve_preprocess_target_coordinates(db=db, draft_id=DRAFT_ID, ui_ra_deg=272.68, ui_dec_deg=-7.21)
    report["pointing_ra_deg"] = ira
    report["pointing_dec_deg"] = ide

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

    counts = {}
    for g in sorted(proc.iterdir()):
        if g.is_dir():
            counts[g.name] = {
                "proc_fits": len(list(g.glob("proc_*.fits"))),
                "proc_csv": len(list(g.glob("proc_*.csv"))),
            }
    report["processed_counts"] = counts

    row = db.fetch_obs_draft_by_id(DRAFT_ID) or {}
    eq_id = int(row.get("EQUIPMENT_ID") or row.get("ID_EQUIPMENTS") or 3)

    cfg.sips_dao_fwhm_px = 2.5
    cfg.sips_dao_threshold_sigma = 3.5

    psf_orig = _enable_psf_flag()
    report["psf_flag_original"] = psf_orig
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
        report["platesolve_out_keys"] = list(ps_out.keys()) if isinstance(ps_out, dict) else str(type(ps_out))
        if isinstance(ps_out, dict) and ps_out.get("error"):
            report["platesolve_error"] = ps_out.get("error")
            _restore_psf_flag(psf_orig)
            report["psf_flag_restored"] = True
            RESULT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(report, indent=2))
            return 1

        setups = _find_phase2a_paths(cfg, DRAFT_ID, draft_dir_override=None)
        report["setups"] = list(setups.keys()) if setups else []
        phot_results = {}
        for nm, p in sorted((setups or {}).items()):
            ms_fits = Path(p["masterstar_fits"]) if p.get("masterstar_fits") else None
            og_dir = Path(p["obs_group_dir"]) if p.get("obs_group_dir") else None
            ms_csv = (og_dir / "masterstars_full_match.csv") if og_dir else None
            vt_csv = (og_dir / "variable_targets.csv") if og_dir else None
            pf_dir = Path(p["per_frame_csv_dir"]) if p.get("per_frame_csv_dir") else None
            dt_dir = Path(p["detrended_aligned_dir"]) if p.get("detrended_aligned_dir") else None
            out_d = Path(p["output_dir"]) if p.get("output_dir") else None
            t1 = time.time()
            pr = run_full_photometry_pipeline(
                masterstar_fits_path=ms_fits,
                variable_targets_csv=vt_csv,
                masterstars_csv=ms_csv,
                per_frame_csv_dir=pf_dir,
                detrended_aligned_dir=dt_dir,
                output_dir=out_d,
                cfg=cfg,
                db=db,
                draft_id=DRAFT_ID,
                progress_cb=lambda m: print(f"[phot {nm}] {m}", flush=True),
            )
            phot_results[nm] = {"sec": time.time() - t1, "keys": list(pr.keys()) if isinstance(pr, dict) else []}
        report["photometry"] = phot_results
    finally:
        _restore_psf_flag(psf_orig)
        report["psf_flag_restored"] = True
        chk = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        report["psf_flag_after_restore"] = bool(chk.get("psf_photometry_enabled", False))

    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    RESULT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
