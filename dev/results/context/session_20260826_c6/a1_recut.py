# -*- coding: ascii -*-
"""APERTURE-01 recut: rename era04 -> era04_candidate1, photometry-only into era04.

MASTERSTAR/alignment/WCS inherited from C6-1 full chain (aperture does not
touch those stages). Live 516 and era03 are not written.
"""
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

from config import AppConfig  # noqa: E402
from photometry_sha import compute_photometry_sha  # noqa: E402

SESSION = ROOT / "dev" / "results" / "context" / "session_20260826_c6"
LIVE = ROOT / "Archive" / "Drafts" / "draft_000516"
ERA03 = ROOT / "Archive" / "Drafts" / "draft_000516_snapshot_era03_20260820"
ERA04 = ROOT / "Archive" / "Drafts" / "draft_000516_snapshot_era04_20260826"
CAND1 = ROOT / "Archive" / "Drafts" / "draft_000516_snapshot_era04_candidate1_20260826"
SETUP = "NoFilter_60_2"
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


def _live_shas() -> dict:
    ps = LIVE / "platesolve" / SETUP
    return {
        "516_csv": sha256_file(ps / "masterstars_full_match.csv"),
        "516_fits": sha256_file(ps / "MASTERSTAR.fits"),
        "516_epsf": sha256_file(ps / "masterstar_epsf.fits"),
    }


def _empty_dir(p: Path) -> None:
    if not p.exists():
        return
    for child in p.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def prepare_tree() -> dict:
    out = {"renamed": False, "copied": False}
    if ERA04.is_dir() and not CAND1.is_dir():
        ERA04.rename(CAND1)
        out["renamed"] = True
    if CAND1.is_dir() and not ERA04.is_dir():
        shutil.copytree(CAND1, ERA04)
        out["copied"] = True
    phot = ERA04 / "platesolve" / SETUP / "photometry"
    phot.mkdir(parents=True, exist_ok=True)
    _empty_dir(phot)
    pfc = ERA04 / "platesolve" / SETUP / "per_frame_catalogs"
    if pfc.is_dir():
        _empty_dir(pfc)
    lights = ERA04 / "detrended_aligned" / "lights" / SETUP
    out["phot_emptied"] = True
    out["proc_csv_removed"] = 0
    # Keep existing proc_*.csv: Phase 0/1 D3 pool reads them; catalog export overwrites.
    for dest in (ERA04 / "platesolve" / SETUP, ERA04 / "platesolve"):
        dest.mkdir(parents=True, exist_ok=True)
        for name in ("cal_diag.json", "sat_diag.json", "draft_manifest.json"):
            src = ERA04 / name
            if src.is_file():
                shutil.copy2(src, dest / name)
    return out


def main() -> int:
    t_all = time.perf_counter()
    SESSION.mkdir(parents=True, exist_ok=True)
    rec: dict = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "task": "APERTURE-01 recut mode f_fixed_night",
        "timings_s": {},
    }
    live_before = _live_shas()
    rec["live_before"] = live_before
    rec["prepare"] = prepare_tree()
    rec["era03_still_present"] = ERA03.is_dir()
    rec["candidate1_present"] = CAND1.is_dir()

    from config import AppConfig
    from database import VyvarDatabase
    from infolog import end_infolog_session, start_infolog_session
    from photometry_core import run_full_photometry_pipeline
    from pipeline import export_per_frame_catalogs
    from psf_internal_lc import write_internal_psf_lightcurves
    from photometry_sha import compute_photometry_sha

    cfg = AppConfig()
    cfg.save_lightcurve_png = False
    rec["aperture_f"] = float(cfg.aperture_fwhm_factor)
    rec["aperture_policy_mode"] = str(cfg.aperture_policy_mode)

    infodir = SESSION / "a1_infolog"
    infodir.mkdir(parents=True, exist_ok=True)
    start_infolog_session(infodir)
    logging.basicConfig(level=logging.INFO)
    ps = ERA04 / "platesolve" / SETUP
    lights = ERA04 / "detrended_aligned" / "lights" / SETUP
    phot = ps / "photometry"
    db = VyvarDatabase(cfg.database_path)

    t0 = time.perf_counter()
    try:
        rec["catalog_export"] = export_per_frame_catalogs(
            frames_root=lights,
            platesolve_dir=ps,
            masterstars_csv=ps / "masterstars_full_match.csv",
            masterstar_fits=ps / "MASTERSTAR.fits",
            use_master_fast_path=True,
            app_config=cfg,
            draft_id=516,
            write_sidecar_csv_next_to_fits=True,
        )
    except Exception as exc:
        rec["err"] = f"CATALOG {type(exc).__name__}: {exc}"
        logging.exception("APERTURE-01 catalog export failed")
    rec["timings_s"]["catalog_export"] = round(time.perf_counter() - t0, 1)
    try:
        import pandas as pd

        sample = next(lights.glob("proc_*.csv"))
        sdf = pd.read_csv(sample, usecols=lambda c: c in (
            "aperture_r_px", "aperture_policy_mode", "aperture_f",
            "fwhm_px_for_aperture", "sky_annulus_r_in_px", "sky_annulus_r_out_px",
        ))
        rec["proc_sample"] = {
            "file": sample.name,
            "n": int(len(sdf)),
            "r_ap_unique": [round(float(x), 4) for x in sorted(sdf["aperture_r_px"].dropna().unique().tolist())][:8]
            if "aperture_r_px" in sdf.columns else [],
            "mode": str(sdf["aperture_policy_mode"].iloc[0]) if "aperture_policy_mode" in sdf.columns else "",
            "f": float(sdf["aperture_f"].iloc[0]) if "aperture_f" in sdf.columns else None,
            "fwhm": float(sdf["fwhm_px_for_aperture"].iloc[0]) if "fwhm_px_for_aperture" in sdf.columns else None,
            "r_in": float(sdf["sky_annulus_r_in_px"].iloc[0]) if "sky_annulus_r_in_px" in sdf.columns else None,
            "r_out": float(sdf["sky_annulus_r_out_px"].iloc[0]) if "sky_annulus_r_out_px" in sdf.columns else None,
        }
        runiq = rec["proc_sample"]["r_ap_unique"]
        if len(runiq) != 1 or abs(runiq[0] - 2.0) > 0.15:
            rec["err"] = (rec.get("err") or "") + f" PROC_R_AP {runiq}"
    except Exception as exc:
        rec["proc_sample_err"] = str(exc)

    t0 = time.perf_counter()
    try:
        if rec.get("err"):
            raise RuntimeError("skip photometry after catalog export failure")
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
        logging.exception("APERTURE-01 photometry failed")
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
        logging.exception("APERTURE-01 PSF LC failed")
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
    rec["live_after"] = _live_shas()
    rec["live_unchanged"] = rec["live_after"] == live_before
    rec["live_sha_guard"] = rec["live_after"] == LIVE_SHA
    rec["era03_still_present"] = ERA03.is_dir()
    rec["timings_s"]["total"] = round(time.perf_counter() - t_all, 1)
    (SESSION / "a1_recut.json").write_text(
        json.dumps(rec, indent=2, default=str), encoding="ascii"
    )
    print(
        "A1 recut",
        "err",
        rec.get("err"),
        "sha_core",
        rec.get("sha_core"),
        "n",
        rec.get("n_core"),
        "s",
        rec["timings_s"],
    )
    return 1 if rec.get("err") else 0


if __name__ == "__main__":
    raise SystemExit(main())
