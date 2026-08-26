# -*- coding: ascii -*-
"""Re-enhance 3 frames that missed aperture_r_px, fill sigma, run photometry."""
from __future__ import annotations

import json
import logging
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

ROOT = Path(r"C:\ASTRO\python\VYVAR")
sys.path.insert(0, str(ROOT / "src_py"))
sys.path.insert(0, str(ROOT / "dev" / "tests"))

from aperture_policy import load_qc_fwhm_map  # noqa: E402
from config import AppConfig  # noqa: E402
from database import VyvarDatabase  # noqa: E402
from infolog import end_infolog_session, start_infolog_session  # noqa: E402
from photometry_core import (  # noqa: E402
    ERR_BKG_SOURCE_COL,
    ERR_BKG_SOURCE_EMPIRICAL,
    SIGMA_BKG_AP_COL,
    _labbe_content_seed_from_header,
    enhance_catalog_dataframe_aperture_bpm,
    measure_empty_aperture_sigma_bkg,
    run_full_photometry_pipeline,
)
from photometry_sha import compute_photometry_sha  # noqa: E402
from psf_internal_lc import write_internal_psf_lightcurves  # noqa: E402

ERA04 = ROOT / "Archive" / "Drafts" / "draft_000516_snapshot_era04_20260826"
CAND2 = ROOT / "Archive" / "Drafts" / "draft_000516_snapshot_era04_candidate2_20260826"
SETUP = "NoFilter_60_2"
SESSION = ROOT / "dev" / "results" / "context" / "session_20260826_a01c"
LIVE = ROOT / "Archive" / "Drafts" / "draft_000516"
LIVE_SHA = {
    "516_csv": "bfa24039778f437b2bf7ed37056b6b507e068d52d2c7b4a222a73002125b250a",
    "516_fits": "13e77cf8a1dcb4e73fae0558437d7234feeb70a5ae4aa85064a8316812b01345",
    "516_epsf": "172f95403beae36dc9c7b35e4758f37996bb661e3d96d180d1444ded71369a20",
}
MISSING = (
    "BO_CVn_Light_001",
    "BO_CVn_Light_004",
    "BO_CVn_Light_012",
)


def sha256_file(p: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def reenhance_one(lights: Path, stem: str, cfg, qc_map, night) -> dict:
    pc = lights / f"proc_{stem}.csv"
    fp = lights / f"{stem}.fits"
    df = pd.read_csv(pc, low_memory=False)
    with fits.open(fp, memmap=False) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float64)
        hdr = hdul[0].header
    rec = {"stem": stem, "has_r": False, "r": None, "f": None, "err": ""}
    try:
        out = enhance_catalog_dataframe_aperture_bpm(
            df,
            data,
            hdr,
            aperture_enabled=True,
            aperture_fwhm_factor=float(cfg.aperture_fwhm_factor),
            annulus_inner_fwhm=float(cfg.annulus_inner_fwhm),
            annulus_outer_fwhm=float(cfg.annulus_outer_fwhm),
            nonlinearity_peak_percentile=float(cfg.nonlinearity_peak_percentile),
            nonlinearity_fwhm_ratio=float(cfg.nonlinearity_fwhm_ratio),
            master_dark_path=None,
            aperture_policy_mode=str(cfg.aperture_policy_mode),
            fwhm_night_median_px=night,
            qc_fwhm_by_name=qc_map,
            frame_name=fp.name,
        )
    except Exception as exc:  # noqa: BLE001
        rec["err"] = f"{type(exc).__name__}: {exc}"
        return rec
    rec["has_r"] = "aperture_r_px" in out.columns
    if rec["has_r"]:
        rec["r"] = round(float(pd.to_numeric(out["aperture_r_px"], errors="coerce").median()), 4)
        rec["f"] = float(out["aperture_f"].iloc[0]) if "aperture_f" in out.columns else None
        out.to_csv(pc, index=False)
    return rec


def fill_sigma(lights: Path) -> dict:
    n_ok = n_fail = n_skip = 0
    last_reason = ""
    for pc in sorted(lights.glob("proc_*.csv")):
        df = pd.read_csv(pc, low_memory=False)
        if "aperture_r_px" not in df.columns:
            n_fail += 1
            last_reason = f"no_r_ap {pc.name}"
            continue
        if SIGMA_BKG_AP_COL in df.columns:
            sigs = pd.to_numeric(df[SIGMA_BKG_AP_COL], errors="coerce")
            if int(sigs.notna().sum()) == len(df) and float(sigs.min()) >= 0:
                n_skip += 1
                continue
        fits_name = pc.name.replace("proc_", "").replace(".csv", ".fits")
        fp = lights / fits_name
        r_ap = float(pd.to_numeric(df["aperture_r_px"], errors="coerce").median())
        r_in = float(pd.to_numeric(df["sky_annulus_r_in_px"], errors="coerce").median())
        r_out = float(pd.to_numeric(df["sky_annulus_r_out_px"], errors="coerce").median())
        xs = pd.to_numeric(df["x"], errors="coerce").to_numpy(dtype=np.float64)
        ys = pd.to_numeric(df["y"], errors="coerce").to_numpy(dtype=np.float64)
        with fits.open(fp, memmap=False) as hdul:
            data = np.asarray(hdul[0].data, dtype=np.float64)
            hdr = hdul[0].header
        seed = _labbe_content_seed_from_header(hdr, r_ap=r_ap)
        sig, _nv, reason = measure_empty_aperture_sigma_bkg(
            data, xs, ys, r_ap, r_in, r_out, seed=int(seed),
            frame_id=fits_name, star_list_source="aperture01c_fill",
        )
        if math.isfinite(float(sig)) and float(sig) >= 0:
            df[SIGMA_BKG_AP_COL] = float(sig)
            df[ERR_BKG_SOURCE_COL] = ERR_BKG_SOURCE_EMPIRICAL
            df.to_csv(pc, index=False)
            n_ok += 1
        else:
            last_reason = str(reason)
            n_fail += 1
    return {"n_ok": n_ok, "n_fail": n_fail, "n_skip": n_skip, "last_reason": last_reason}


def main() -> int:
    t_all = time.perf_counter()
    rec: dict = {"task": "APERTURE-01c repair 3 frames + p2a", "timings_s": {}}
    lights = ERA04 / "detrended_aligned" / "lights" / SETUP
    ps = ERA04 / "platesolve" / SETUP
    phot = ps / "photometry"
    cfg = AppConfig()
    cfg.save_lightcurve_png = False
    qc_map, night = load_qc_fwhm_map(ERA04 / "calibrated" / "lights" / "qc_metrics.csv")
    rec["f"] = float(cfg.aperture_fwhm_factor)
    rec["night_fwhm"] = night
    rec["repair"] = []
    t0 = time.perf_counter()
    src_lights = CAND2 / "detrended_aligned" / "lights" / SETUP
    rec["restored_from_candidate2"] = []
    for stem in MISSING:
        src = src_lights / f"proc_{stem}.csv"
        dst = lights / f"proc_{stem}.csv"
        if src.is_file():
            shutil.copy2(src, dst)
            rec["restored_from_candidate2"].append(stem)
        rec["repair"].append(reenhance_one(lights, stem, cfg, qc_map, night))
    rec["timings_s"]["repair"] = round(time.perf_counter() - t0, 1)
    if not all(r["has_r"] and r["r"] is not None and abs(r["r"] - 7.009) < 0.15 for r in rec["repair"]):
        rec["err"] = f"REPAIR {rec['repair']}"
        (SESSION / "a1c_repair.json").write_text(json.dumps(rec, indent=2, default=str), encoding="ascii")
        print("REPAIR FAIL", rec["repair"])
        return 1

    t0 = time.perf_counter()
    rec["fill"] = fill_sigma(lights)
    rec["timings_s"]["fill"] = round(time.perf_counter() - t0, 1)
    if rec["fill"]["n_fail"]:
        rec["err"] = f"FILL {rec['fill']}"
        (SESSION / "a1c_repair.json").write_text(json.dumps(rec, indent=2, default=str), encoding="ascii")
        print("FILL FAIL", rec["fill"])
        return 1

    infodir = SESSION / "a1c_infolog"
    infodir.mkdir(parents=True, exist_ok=True)
    start_infolog_session(infodir)
    logging.basicConfig(level=logging.INFO)
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
        logging.exception("photometry failed")
    rec["timings_s"]["photometry"] = round(time.perf_counter() - t0, 1)
    t0 = time.perf_counter()
    try:
        rec["psf_out"] = write_internal_psf_lightcurves(
            platesolve_dir=ps, frames_root=lights, photometry_dir=phot, cfg=cfg
        )
    except Exception as exc:
        rec["err"] = (rec.get("err") or "") + f" PSF {type(exc).__name__}: {exc}"
    rec["timings_s"]["psf_lc"] = round(time.perf_counter() - t0, 1)
    end_infolog_session()
    try:
        db.conn.close()
    except Exception:
        pass
    rec["sha_core"], rec["n_core"] = compute_photometry_sha(ERA04, include_comp_qa=False)
    rec["sha_ext"], rec["n_ext"] = compute_photometry_sha(ERA04, include_comp_qa=True)
    rec["live_after"] = {
        "516_csv": sha256_file(LIVE / "platesolve" / SETUP / "masterstars_full_match.csv"),
        "516_fits": sha256_file(LIVE / "platesolve" / SETUP / "MASTERSTAR.fits"),
        "516_epsf": sha256_file(LIVE / "platesolve" / SETUP / "masterstar_epsf.fits"),
    }
    rec["live_unchanged"] = rec["live_after"] == LIVE_SHA
    rec["timings_s"]["total"] = round(time.perf_counter() - t_all, 1)
    (SESSION / "a1c_repair.json").write_text(json.dumps(rec, indent=2, default=str), encoding="ascii")
    print("A1c repair", rec.get("err"), rec.get("sha_core"), rec.get("n_core"), rec["timings_s"])
    return 1 if rec.get("err") else 0


if __name__ == "__main__":
    raise SystemExit(main())
