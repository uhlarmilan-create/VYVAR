# -*- coding: ascii -*-
"""C6-1: full chain 516 from Raw into era04. Never writes era03 or live 516 products."""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(r"C:\ASTRO\python\VYVAR")
sys.path.insert(0, str(ROOT / "src_py"))
sys.path.insert(0, str(ROOT / "dev" / "tests"))
sys.path.insert(0, str(ROOT / "dev" / "scripts"))
SESSION = Path(__file__).resolve().parent
LIVE = ROOT / "Archive" / "Drafts" / "draft_000516"
ERA03 = ROOT / "Archive" / "Drafts" / "draft_000516_snapshot_era03_20260820"
ERA04 = ROOT / "Archive" / "Drafts" / "draft_000516_snapshot_era04_20260826"
SETUP = "NoFilter_60_2"
DARK = ROOT / "CalibrationLibrary" / "Dark_60s_Dark_0G_-10deg_Bin1_20260422.fits"
FLAT = ROOT / "CalibrationLibrary" / "Flat_0.15s_NoFilter_0G_-10.5deg_Bin1_20260422.fits"
HINT_RA = 209.5043
HINT_DEC = 41.19122
LIVE_SHA = {
    "516_csv": "bfa24039778f437b2bf7ed37056b6b507e068d52d2c7b4a222a73002125b250a",
    "516_fits": "13e77cf8a1dcb4e73fae0558437d7234feeb70a5ae4aa85064a8316812b01345",
    "516_epsf": "172f95403beae36dc9c7b35e4758f37996bb661e3d96d180d1444ded71369a20",
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> str:
    import subprocess

    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()


def _live_shas() -> dict:
    ps = LIVE / "platesolve" / SETUP
    return {
        "516_csv": sha256_file(ps / "masterstars_full_match.csv"),
        "516_fits": sha256_file(ps / "MASTERSTAR.fits"),
        "516_epsf": sha256_file(ps / "masterstar_epsf.fits"),
    }


def _collect_inv(ps: Path, phot: Path) -> dict:
    import pandas as pd

    out: dict = {}
    ms = ps / "masterstars_full_match.csv"
    if ms.is_file():
        df = pd.read_csv(ms, low_memory=False)
        if "vy_identity_gate" in df.columns:
            g = df["vy_identity_gate"].astype(str).str.strip().str.lower()
            out["identity_gate"] = {
                "n": int(len(g)),
                "ok": int((g == "ok").sum()),
                "warn": int((g == "warn").sum()),
                "fail": int((g == "fail").sum()),
                "pass": int((g == "pass").sum()),
            }
        if "zone" in df.columns:
            z = df["zone"].astype(str).str.strip().str.lower()
            out["zone_saturated_n"] = int((z == "saturated").sum())
        for col in ("vy_lock_reject", "lock_geometry_reject", "geometry_reject"):
            if col in df.columns:
                out[col] = int(pd.to_numeric(df[col], errors="coerce").fillna(0).astype(bool).sum())
    opt = ps / "astrometry_optimizer_meta.json"
    if not opt.is_file():
        opt = ps / "optimizer_meta.json"
    if opt.is_file():
        out["optimizer_meta"] = json.loads(opt.read_text(encoding="utf-8"))
    drops = phot / "comp_drop_log.csv"
    if not drops.is_file():
        drops = phot / "comparison_drop_counts.json"
    if drops.suffix == ".json" and drops.is_file():
        out["comp_drops"] = json.loads(drops.read_text(encoding="utf-8"))
    elif drops.is_file():
        dfd = pd.read_csv(drops, low_memory=False)
        if "predicate" in dfd.columns:
            out["comp_drops"] = dfd["predicate"].astype(str).value_counts().to_dict()
    sv = phot / "suspected_variables.csv"
    if sv.is_file():
        out["suspected_variables_n"] = int(len(pd.read_csv(sv)))
    meta = phot / "pipeline_meta.json"
    if meta.is_file():
        pm = json.loads(meta.read_text(encoding="utf-8"))
        out["pipeline_meta_inv"] = pm.get("invariants")
        if "identity_gate" in pm:
            out["identity_gate_meta"] = pm["identity_gate"]
        orf = pm.get("optimizer_refit") or {}
        inner = orf.get("optimizer_refit") if isinstance(orf, dict) else None
        if isinstance(inner, dict):
            out["optimizer_refit"] = {
                "rejected": inner.get("rejected"),
                "reason": inner.get("reason"),
                "rms_sip": inner.get("rms_sip"),
                "n": inner.get("n"),
                "p95_entry": inner.get("p95_entry"),
                "p95_candidate": inner.get("p95_candidate"),
            }
        elif isinstance(orf, dict) and "p95_entry" in orf:
            out["optimizer_refit"] = {
                "rejected": orf.get("rejected"),
                "reason": orf.get("reason"),
                "rms_sip": orf.get("rms_sip"),
                "n": orf.get("n"),
                "p95_entry": orf.get("p95_entry"),
                "p95_candidate": orf.get("p95_candidate"),
            }
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not ERA03.is_dir():
        raise SystemExit(f"era03 missing: {ERA03}")
    if not LIVE.is_dir():
        raise SystemExit(f"live 516 missing: {LIVE}")
    if not DARK.is_file() or not FLAT.is_file():
        raise SystemExit("calibration masters missing")
    rec: dict = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "era03": str(ERA03),
        "era04": str(ERA04),
        "err": None,
        "timings_s": {},
        "live_before": _live_shas(),
        "masterstar_top1": "BO_CVn_Light_109.fits",
    }
    t_all = time.perf_counter()
    live_before = rec["live_before"]
    resume_ps = "--from-ps" in sys.argv or "--phot-only" in sys.argv
    cal_ok = (ERA04 / "calibrated" / "lights" / SETUP).is_dir() and any(
        (ERA04 / "calibrated" / "lights" / SETUP).glob("*.fit*")
    )
    if ERA04.exists() and not resume_ps and not cal_ok:
        logging.info("removing incomplete era04 %s", ERA04)
        shutil.rmtree(ERA04)
    if not resume_ps and not cal_ok:
        ERA04.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        if (ERA04 / "Raw").exists():
            shutil.rmtree(ERA04 / "Raw")
        shutil.copytree(LIVE / "Raw", ERA04 / "Raw")
        for name in ("draft_manifest.json", "sat_diag.json"):
            src = LIVE / name
            if src.is_file():
                shutil.copy2(src, ERA04 / name)
        rec["timings_s"]["copy_raw"] = round(time.perf_counter() - t0, 1)
    else:
        rec["timings_s"]["copy_raw"] = 0.0
        logging.info("resume: keep era04 calibrated/preprocess")

    from config import AppConfig
    from database import VyvarDatabase
    from infolog import end_infolog_session, start_infolog_session
    from night_run import _night_run_platesolve, _night_run_preprocess
    from pipeline import AstroPipeline
    from photometry_core import run_full_photometry_pipeline
    from psf_internal_lc import write_internal_psf_lightcurves
    from photometry_sha import compute_photometry_sha

    cfg = AppConfig()
    cfg.save_lightcurve_png = False
    cfg.k2_mode = "literature"
    cfg.per_frame_saturation_enabled = True
    cfg.export_err_mode = "calibrated"
    pipeline = AstroPipeline(cfg)
    infodir = SESSION / "c61_infolog"
    infodir.mkdir(parents=True, exist_ok=True)
    start_infolog_session(infodir)

    phot_only = "--phot-only" in sys.argv
    if phot_only:
        rec["cal_out"] = {"skipped": True, "reason": "phot-only"}
        rec["timings_s"]["calibration"] = 0.0
        rec["timings_s"]["preprocess"] = 0.0
        rec["timings_s"]["platesolve_ms"] = 0.0
        resume_ps = True
        cal_ok = True

    t0 = time.perf_counter()
    if resume_ps or cal_ok:
        rec["cal_out"] = {"skipped": True, "reason": "calibrated lights already present"}
        rec["timings_s"]["calibration"] = 0.0
    else:
        try:
            rec["cal_out"] = pipeline.quick_calibrate_last_import(
                archive_path=ERA04,
                master_dark_path=DARK,
                masterflat_by_filter={"NoFilter": FLAT},
                masterflat_by_obs_key={"NoFilter|60|2": FLAT},
                master_dark_by_obs_key={"NoFilter|60|2": DARK},
                equipment_id=1,
                draft_id=None,
                roundness_reject_above=1.25,
            )
        except Exception as exc:
            rec["err"] = f"CAL {type(exc).__name__}: {exc}"
            logging.exception("C6-1 calibration failed")
            end_infolog_session()
            (SESSION / "c61.json").write_text(json.dumps(rec, indent=2, default=str), encoding="ascii")
            return 1
        rec["timings_s"]["calibration"] = round(time.perf_counter() - t0, 1)

    top1 = ERA04 / "calibrated" / "lights" / SETUP / "BO_CVn_Light_109.fits"
    if not top1.is_file():
        rec["err"] = f"TOP1 missing: {top1}"
        (SESSION / "c61.json").write_text(json.dumps(rec, indent=2, default=str), encoding="ascii")
        return 1
    job = {
        "kind": "make_masterstar",
        "archive_path": str(ERA04),
        "fwhm_limit_px": float(getattr(cfg, "qc_fwhm_limit", 8.0)),
        "inject_pointing_ra_deg": HINT_RA,
        "inject_pointing_dec_deg": HINT_DEC,
        "max_control_points": int(cfg.alignment_max_control_points),
        "min_detected_stars": 200,
        "max_detected_stars": 4000,
        "platesolve_backend": "vyvar",
        "plate_solve_fov_deg": float(cfg.plate_solve_fov_deg),
        "catalog_match_max_sep_arcsec": 2.0,
        "saturate_level_fraction": 0.999,
        "max_catalog_rows": 12000,
        "n_comparison_stars": 150,
        "dao_threshold_sigma": float(cfg.masterstar_dao_threshold_sigma),
        "dao_fwhm_px": float(cfg.sips_dao_fwhm_px),
        "id_equipment": 1,
        "draft_id": None,
        "catalog_local_gaia_only": True,
        "build_masterstar_and_catalogs": True,
        "masterstar_candidate_paths": [str(top1)],
        "masterstar_selection_pct": 10.0,
    }
    if phot_only:
        rec["timings_s"]["preprocess"] = 0.0
        rec["ps_out"] = {"skipped": True}
        rec["timings_s"]["platesolve_ms"] = 0.0
    else:
        t0 = time.perf_counter()
        try:
            _night_run_preprocess(
                pending=job, ap=ERA04, pipeline=pipeline, progress_cb=lambda i, t, m: None
            )
        except Exception as exc:
            rec["err"] = f"PRE {type(exc).__name__}: {exc}"
            logging.exception("C6-1 preprocess failed")
            end_infolog_session()
            (SESSION / "c61.json").write_text(json.dumps(rec, indent=2, default=str), encoding="ascii")
            return 1
        rec["timings_s"]["preprocess"] = round(time.perf_counter() - t0, 1)

        t0 = time.perf_counter()
        try:
            rec["ps_out"] = _night_run_platesolve(
                pending=job, ap=ERA04, pipeline=pipeline, plan=None, progress_cb=lambda i, t, m: None
            )
        except Exception as exc:
            rec["err"] = f"PS {type(exc).__name__}: {exc}"
            logging.exception("C6-1 platesolve/MS failed")
            end_infolog_session()
            (SESSION / "c61.json").write_text(json.dumps(rec, indent=2, default=str), encoding="ascii")
            return 1
        rec["timings_s"]["platesolve_ms"] = round(time.perf_counter() - t0, 1)

    ps = ERA04 / "platesolve" / SETUP
    lights = ERA04 / "detrended_aligned" / "lights" / SETUP
    phot = ps / "photometry"
    phot.mkdir(parents=True, exist_ok=True)
    for dest in (ps, ps.parent):
        for name in ("cal_diag.json", "sat_diag.json", "draft_manifest.json"):
            src = ERA04 / name
            if src.is_file():
                shutil.copy2(src, dest / name)
    if phot_only:
        rec["ps_out"] = {"skipped": True}
        rec["timings_s"]["platesolve_ms"] = 0.0
    db = VyvarDatabase(cfg.database_path)
    t0 = time.perf_counter()
    try:
        rec["phot_out"] = run_full_photometry_pipeline(
            masterstar_fits_path=ps / "MASTERSTAR.fits",
            variable_targets_csv=ps / "variable_targets.csv",
            masterstars_csv=ps / "masterstars_full_match.csv",
            per_frame_csv_dir=lights,
            detrended_aligned_dir=lights,
            output_dir=phot,
            cfg=cfg,
            db=db,
            draft_id=516,
        )
    except Exception as exc:
        rec["err"] = f"PHOT {type(exc).__name__}: {exc}"
        logging.exception("C6-1 photometry failed")
    rec["timings_s"]["photometry"] = round(time.perf_counter() - t0, 1)

    t0 = time.perf_counter()
    try:
        rec["psf_out"] = write_internal_psf_lightcurves(
            platesolve_dir=ps,
            frames_root=lights,
            photometry_dir=phot,
            cfg=cfg,
        )
    except Exception as exc:
        rec["err"] = (rec.get("err") or "") + f" PSF {type(exc).__name__}: {exc}"
        logging.exception("C6-1 PSF LC failed")
    rec["timings_s"]["psf_lc"] = round(time.perf_counter() - t0, 1)
    end_infolog_session()
    try:
        db.conn.close()
    except Exception:
        pass

    try:
        rec["sha_core"], rec["n_core"] = compute_photometry_sha(ERA04, include_comp_qa=False)
        rec["sha_ext"], rec["n_ext"] = compute_photometry_sha(ERA04, include_comp_qa=True)
    except Exception as exc:
        rec["sha_err"] = str(exc)
    rec["inv"] = _collect_inv(ps, phot)
    rec["live_after"] = _live_shas()
    rec["live_unchanged"] = rec["live_after"] == live_before
    rec["live_sha_guard"] = rec["live_after"] == LIVE_SHA
    rec["era03_still_present"] = ERA03.is_dir()
    rec["timings_s"]["total"] = round(time.perf_counter() - t_all, 1)
    (SESSION / "c61.json").write_text(json.dumps(rec, indent=2, default=str), encoding="ascii")
    print("C6-1", "err", rec.get("err"), "sha_core", rec.get("sha_core"), "n_core", rec.get("n_core"), "s", rec["timings_s"])
    return 1 if rec.get("err") else 0


if __name__ == "__main__":
    raise SystemExit(main())
