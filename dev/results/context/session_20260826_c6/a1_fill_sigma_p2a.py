# -*- coding: ascii -*-
"""Fill sigma_bkg_ap on era04 proc CSVs after APERTURE-01 catalog export.

Does not rewrite fluxes. Then run Phase 2A + PSF LCs only.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

ROOT = Path(r"C:\ASTRO\python\VYVAR")
sys.path.insert(0, str(ROOT / "src_py"))
sys.path.insert(0, str(ROOT / "dev" / "tests"))

from config import AppConfig  # noqa: E402
from database import VyvarDatabase  # noqa: E402
from infolog import end_infolog_session, start_infolog_session  # noqa: E402
from photometry_core import (  # noqa: E402
    ERR_BKG_SOURCE_COL,
    ERR_BKG_SOURCE_EMPIRICAL,
    SIGMA_BKG_AP_COL,
    _labbe_content_seed_from_header,
    measure_empty_aperture_sigma_bkg,
    run_phase2a,
)
from photometry_sha import compute_photometry_sha  # noqa: E402
from psf_internal_lc import write_internal_psf_lightcurves  # noqa: E402

ERA04 = ROOT / "Archive" / "Drafts" / "draft_000516_snapshot_era04_20260826"
SETUP = "NoFilter_60_2"
SESSION = ROOT / "dev" / "results" / "context" / "session_20260826_c6"
LIVE = ROOT / "Archive" / "Drafts" / "draft_000516"
LIVE_SHA = {
    "516_csv": "bfa24039778f437b2bf7ed37056b6b507e068d52d2c7b4a222a73002125b250a",
    "516_fits": "13e77cf8a1dcb4e73fae0558437d7234feeb70a5ae4aa85064a8316812b01345",
    "516_epsf": "172f95403beae36dc9c7b35e4758f37996bb661e3d96d180d1444ded71369a20",
}


def sha256_file(p: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fill_sigma(lights: Path) -> dict:
    n_ok = n_fail = 0
    last_reason = ""
    for pc in sorted(lights.glob("proc_*.csv")):
        fits_name = pc.name.replace("proc_", "").replace(".csv", ".fits")
        fp = lights / fits_name
        df = pd.read_csv(pc, low_memory=False)
        r_ap = float(pd.to_numeric(df["aperture_r_px"], errors="coerce").median())
        r_in = float(pd.to_numeric(df["sky_annulus_r_in_px"], errors="coerce").median())
        r_out = float(pd.to_numeric(df["sky_annulus_r_out_px"], errors="coerce").median())
        xs = pd.to_numeric(df["x"], errors="coerce").to_numpy(dtype=np.float64)
        ys = pd.to_numeric(df["y"], errors="coerce").to_numpy(dtype=np.float64)
        with fits.open(fp, memmap=False) as hdul:
            data = np.asarray(hdul[0].data, dtype=np.float64)
            hdr = hdul[0].header
        seed = _labbe_content_seed_from_header(hdr, r_ap=r_ap)
        sig, nv, reason = measure_empty_aperture_sigma_bkg(
            data,
            xs,
            ys,
            r_ap,
            r_in,
            r_out,
            seed=int(seed),
            frame_id=fits_name,
            star_list_source="aperture01_fill",
        )
        if math_isfinite(sig) and sig >= 0:
            df[SIGMA_BKG_AP_COL] = float(sig)
            df[ERR_BKG_SOURCE_COL] = ERR_BKG_SOURCE_EMPIRICAL
            n_ok += 1
        else:
            last_reason = str(reason)
            n_fail += 1
            continue
        df.to_csv(pc, index=False)
    return {"n_ok": n_ok, "n_fail": n_fail, "last_reason": last_reason}


def math_isfinite(v: float) -> bool:
    import math

    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def main() -> int:
    t_all = time.perf_counter()
    rec: dict = {"task": "APERTURE-01 fill sigma + phase2a"}
    lights = ERA04 / "detrended_aligned" / "lights" / SETUP
    ps = ERA04 / "platesolve" / SETUP
    phot = ps / "photometry"
    t0 = time.perf_counter()
    rec["fill"] = fill_sigma(lights)
    rec["timings_s"] = {"fill": round(time.perf_counter() - t0, 1)}
    if rec["fill"]["n_fail"] or rec["fill"]["n_ok"] < 100:
        rec["err"] = f"FILL {rec['fill']}"
        (SESSION / "a1_p2a.json").write_text(json.dumps(rec, indent=2, default=str), encoding="ascii")
        print("FILL FAIL", rec["fill"])
        return 1

    cfg = AppConfig()
    cfg.save_lightcurve_png = False
    infodir = SESSION / "a1_infolog"
    infodir.mkdir(parents=True, exist_ok=True)
    start_infolog_session(infodir)
    logging.basicConfig(level=logging.INFO)
    db = VyvarDatabase(cfg.database_path)
    t0 = time.perf_counter()
    try:
        rec["phot_out"] = run_phase2a(
            masterstar_fits_path=ps / "MASTERSTAR.fits",
            active_targets_csv=phot / "active_targets.csv",
            comparison_stars_csv=phot / "comparison_stars_per_target.csv",
            per_frame_csv_dir=lights,
            detrended_aligned_dir=lights,
            output_dir=phot,
            fwhm_px=5.191733,
            cfg=cfg,
            db=db,
            draft_id=516,
        )
    except Exception as exc:
        rec["err"] = f"P2A {type(exc).__name__}: {exc}"
        logging.exception("phase2a failed")
    rec["timings_s"]["phase2a"] = round(time.perf_counter() - t0, 1)
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
    (SESSION / "a1_p2a.json").write_text(json.dumps(rec, indent=2, default=str), encoding="ascii")
    print("A1 p2a", rec.get("err"), rec.get("sha_core"), rec.get("n_core"), rec["timings_s"])
    return 1 if rec.get("err") else 0


if __name__ == "__main__":
    raise SystemExit(main())
