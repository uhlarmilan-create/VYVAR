#!/usr/bin/env python3
"""Inject Double Cluster pointing into draft 369 FITS + platesolve + B/V/R photometry."""
from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401  (repo layout: src_py + dev on sys.path)
_ROOT = _bootstrap.REPO_ROOT
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ["VYVAR_CT_PROTOTYPE"] = "1"

FIELD_DB = _ROOT / "GAIA_DR3" / "vyvar_gaia_dr3_chiandh_field.db"
CONFIG_PATH = _ROOT / "config.json"
FIELD_CENTER = (35.15, 57.13)
DRAFT_ID = 369


def _patch_config() -> dict:
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    orig = {
        "gaia_db_path": data.get("gaia_db_path"),
        "skip_processed_directory": data.get("skip_processed_directory"),
        "psf_photometry_enabled": data.get("psf_photometry_enabled"),
    }
    data["gaia_db_path"] = str(FIELD_DB.resolve())
    data["skip_processed_directory"] = True
    data["psf_photometry_enabled"] = False
    CONFIG_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return orig


def _restore_config(orig: dict) -> None:
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    for k, v in orig.items():
        data[k] = v
    CONFIG_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _inject_all_pointing(draft_dir: Path, cfg) -> int:
    from pipeline import preprocess_calibrated_to_processed  # noqa: E402
    from database import VyvarDatabase  # noqa: E402

    db = VyvarDatabase(cfg.database_path)
    db.update_obs_draft_center(DRAFT_ID, FIELD_CENTER[0], FIELD_CENTER[1])
    cal = draft_dir / "calibrated" / "lights"
    proc = draft_dir / "processed" / "lights"
    proc.mkdir(parents=True, exist_ok=True)
    paths = sorted(cal.rglob("*.fits"))
    preprocess_calibrated_to_processed(
        calibrated_root=cal,
        processed_root=proc,
        only_paths=paths,
        inject_pointing_ra_deg=float(FIELD_CENTER[0]),
        inject_pointing_dec_deg=float(FIELD_CENTER[1]),
        inject_pointing_only_if_missing=False,
        app_config=cfg,
        db=db,
        draft_id=DRAFT_ID,
    )
    from astropy.io import fits  # noqa: E402

    n = sum(1 for p in paths if fits.getheader(p).get("VYTARGRA") is not None)
    return n


def main() -> int:
    from config import AppConfig  # noqa: E402
    from database import VyvarDatabase  # noqa: E402
    from night_run import _night_run_platesolve  # noqa: E402
    from pipeline import AstroPipeline  # noqa: E402

    sys.path.insert(0, str(_ROOT / "scripts"))
    import chiandh_night_run_bvr as nr  # noqa: E402

    orig = _patch_config()
    report: dict = {"draft_id": DRAFT_ID, "started_utc": datetime.now(timezone.utc).isoformat()}
    try:
        cfg = AppConfig()
        draft_dir = Path(cfg.archive_root) / "Drafts" / f"draft_{DRAFT_ID:06d}"
        for sub in ("platesolve", "detrended_aligned"):
            p = draft_dir / sub
            if p.is_dir():
                shutil.rmtree(p)

        report["n_pointing_injected"] = _inject_all_pointing(draft_dir, cfg)
        pipeline = AstroPipeline(cfg)
        db = VyvarDatabase(cfg.database_path)

        job_ms = {
            "archive_path": str(draft_dir),
            "inject_pointing_ra_deg": float(FIELD_CENTER[0]),
            "inject_pointing_dec_deg": float(FIELD_CENTER[1]),
            "quality_filter_draft_id": DRAFT_ID,
            "max_control_points": 250,
            "min_detected_stars": 200,
            "max_detected_stars": 4000,
            "platesolve_backend": "vyvar",
            "plate_solve_fov_deg": 1.25,
            "catalog_match_max_sep_arcsec": 3.0,
            "max_catalog_rows": 20000,
            "n_comparison_stars": 150,
            "dao_threshold_sigma": 3.5,
            "dao_fwhm_px": 3.5,
            "id_equipment": 2,
            "draft_id": DRAFT_ID,
            "build_masterstar_and_catalogs": True,
            "masterstar_candidate_paths": [],
            "masterstar_selection_pct": 10.0,
        }

        def _prog(i: int, t: int, msg: str) -> None:
            if i == 1 or i == t or i % max(1, t // 8) == 0:
                print(f"[{i}/{t}] {msg}", flush=True)

        ps_out = _night_run_platesolve(
            pending=job_ms,
            ap=draft_dir,
            pipeline=pipeline,
            plan=None,
            progress_cb=_prog,
        )
        report["platesolve"] = ps_out if isinstance(ps_out, dict) else {"ok": True}

        nr._post_platesolve_hook(DRAFT_ID, draft_dir, cfg, pipeline)

        from photometry_core import run_full_photometry_pipeline  # noqa: E402
        from ui_aperture_photometry import _find_phase2a_paths  # noqa: E402
        import chiandh_ct_target_presel as presel  # noqa: E402

        report["presel"] = presel.presel_draft(DRAFT_ID)
        proto = draft_dir / "ct_prototype.csv"
        if proto.is_file():
            proto.unlink()
        setups = _find_phase2a_paths(cfg, DRAFT_ID) or {}
        report["setups"] = sorted(setups.keys())
        phot_results = {}
        for nm in sorted(setups.keys()):
            if str(nm).split("_")[0] not in ("B", "V", "R"):
                continue
            p = setups[nm]
            print(f"Photometry {nm} ...", flush=True)
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
    finally:
        _restore_config(orig)

    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    out = _ROOT / "tmp" / "chiandh_bvr_resume_result.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
