#!/usr/bin/env python3
"""Audit Stage 3 Part 0b: full-chain rebuild from draft_435 calibrated lights."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "dev" / "scripts"))
import _bootstrap  # noqa: E402,F401

sys.path.insert(0, str(_bootstrap.REPO_ROOT / "src_py"))
sys.path.insert(0, str(_bootstrap.REPO_ROOT / "dev"))

SOURCE_DRAFT_ID = 435
SCRATCH_DRAFT_ID = 499
SETUP = "NoFilter_60_2"
ANCHOR_SNAPSHOT = "draft_000435_snapshot_skysurface_20260716"
P10_FRAME_INDICES = [1, 17, 34, 50, 67, 83, 100, 116, 133, 150]


def _git_provenance() -> dict[str, Any]:
    from photometry_core import _resolve_git_provenance

    gh, dirty, files = _resolve_git_provenance()
    return {"git_hash": gh, "git_dirty": dirty, "git_dirty_files": files}


def _setup_scratch_draft(cfg) -> Path:
    src = Path(cfg.archive_root) / "Drafts" / f"draft_{SOURCE_DRAFT_ID:06d}"
    dst = Path(cfg.archive_root) / "Drafts" / f"draft_{SCRATCH_DRAFT_ID:06d}"
    if dst.is_dir():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    shutil.copytree(src / "calibrated", dst / "calibrated")
    manifest = {"draft_id": SCRATCH_DRAFT_ID, "calibration_mode": "vyvar_calibrated", "audit0b": True}
    (dst / "draft_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return dst


def _ensure_db_draft(db, draft_dir: Path) -> None:
    row = db.conn.execute(
        "SELECT ID FROM OBS_DRAFT WHERE ID=?;", (SCRATCH_DRAFT_ID,)
    ).fetchone()
    src = db.conn.execute(
        "SELECT ID_EQUIPMENTS,ID_TELESCOPE,ID_LOCATION,ID_SCANNING,OBSERVATIONSTARTJD,"
        "CENTEROFFIELDRA,CENTEROFFIELDDE FROM OBS_DRAFT WHERE ID=?;",
        (SOURCE_DRAFT_ID,),
    ).fetchone()
    if row is None and src is not None:
        db.conn.execute(
            """
            INSERT INTO OBS_DRAFT (
                ID, ID_EQUIPMENTS, ID_TELESCOPE, ID_LOCATION, ID_SCANNING,
                OBSERVATIONSTARTJD, CENTEROFFIELDRA, CENTEROFFIELDDE, STATUS, IS_CALIBRATED
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'INGESTED', 1);
            """,
            (
                SCRATCH_DRAFT_ID,
                int(src["ID_EQUIPMENTS"]),
                int(src["ID_TELESCOPE"]),
                int(src["ID_LOCATION"]),
                int(src["ID_SCANNING"]),
                float(src["OBSERVATIONSTARTJD"]),
                float(src["CENTEROFFIELDRA"]),
                float(src["CENTEROFFIELDDE"]),
            ),
        )
        db.conn.commit()
    db.update_draft_import_log(
        SCRATCH_DRAFT_ID,
        lights_path=str(draft_dir / "calibrated" / "lights"),
        calib_path=str(draft_dir / "calibrated"),
        imported_at=datetime.now(timezone.utc).isoformat(),
        is_calibrated=True,
        archive_path=str(draft_dir),
    )


def _run_chain(*, skip_photometry: bool = False) -> dict[str, Any]:
    from config import AppConfig
    from database import VyvarDatabase
    from night_run import _night_run_platesolve, _night_run_preprocess
    from pipeline import AstroPipeline
    from tools.reference_seed import seed_reference_observatory

    cfg = AppConfig()
    cfg.save_lightcurve_png = False
    cfg.k2_mode = "literature"
    draft_dir = _setup_scratch_draft(cfg)
    db = VyvarDatabase(cfg.database_path)
    seed_reference_observatory(db)
    _ensure_db_draft(db, draft_dir)
    pipeline = AstroPipeline(cfg)

    src_row = db.conn.execute(
        "SELECT CENTEROFFIELDRA,CENTEROFFIELDDE FROM OBS_DRAFT WHERE ID=?;",
        (SOURCE_DRAFT_ID,),
    ).fetchone()
    ira = float(src_row["CENTEROFFIELDRA"]) if src_row else None
    ide = float(src_row["CENTEROFFIELDDE"]) if src_row else None

    job_ms = {
        "kind": "make_masterstar",
        "archive_path": str(draft_dir),
        "fwhm_limit_px": 0.0,
        "inject_pointing_ra_deg": ira,
        "inject_pointing_dec_deg": ide,
        "quality_filter_draft_id": SCRATCH_DRAFT_ID,
        "max_control_points": int(cfg.alignment_max_control_points),
        "min_detected_stars": 200,
        "max_detected_stars": 4000,
        "platesolve_backend": "vyvar",
        "plate_solve_fov_deg": float(cfg.plate_solve_fov_deg),
        "catalog_match_max_sep_arcsec": 3.0,
        "max_catalog_rows": 12000,
        "n_comparison_stars": 150,
        "dao_threshold_sigma": float(cfg.masterstar_dao_threshold_sigma),
        "dao_fwhm_px": float(cfg.sips_dao_fwhm_px),
        "id_equipment": 1,
        "draft_id": SCRATCH_DRAFT_ID,
        "catalog_local_gaia_only": True,
        "build_masterstar_and_catalogs": True,
        "masterstar_candidate_paths": [],
        "masterstar_selection_pct": 10.0,
    }

    t0 = time.time()
    _night_run_preprocess(pending=job_ms, ap=draft_dir, pipeline=pipeline, progress_cb=lambda i, t, m: None)
    preprocess_s = time.time() - t0

    t0 = time.time()
    ps_out = _night_run_platesolve(
        pending=job_ms, ap=draft_dir, pipeline=pipeline, plan=None, progress_cb=lambda i, t, m: None
    )
    platesolve_s = time.time() - t0

    out: dict[str, Any] = {
        "draft_dir": str(draft_dir),
        "preprocess_elapsed_s": preprocess_s,
        "platesolve_elapsed_s": platesolve_s,
        "platesolve_out": ps_out if isinstance(ps_out, dict) else {"ok": True},
    }

    ps_dir = draft_dir / "platesolve" / SETUP
    ms_csv = ps_dir / "masterstars_full_match.csv"
    ms_fits = ps_dir / "MASTERSTAR.fits"
    if ms_csv.is_file():
        msdf = pd.read_csv(ms_csv, low_memory=False)
        from invariants_runtime import dao_only_fraction_from_masterstars

        out["masterstar_rows"] = int(len(msdf))
        out["dao_only_fraction"] = float(dao_only_fraction_from_masterstars(msdf))
        if "match_status" in msdf.columns:
            st = msdf["match_status"].astype(str).str.upper()
            out["pass1_dao_proxy"] = int((~st.isin({"GAIA_MATCHED", "FORCED_APERTURE"})).sum() + st.isin({"GAIA_MATCHED"}).sum())
            # pass-1 = all DAO detections before pass2 merge approximated from status counts
            out["n_gaia_matched"] = int(st.eq("GAIA_MATCHED").sum())
            out["n_forced"] = int(st.eq("FORCED_APERTURE").sum())
            out["n_dao_only"] = int((~st.isin({"GAIA_MATCHED", "FORCED_APERTURE"})).sum())
    if ms_fits.is_file():
        with fits.open(ms_fits, memmap=False) as hd:
            hdr = hd[0].header
            data = np.asarray(hd[0].data, dtype=np.float64)
            out["bg_std_masterstar"] = float(hdr.get("BGSTD") or np.nanstd(data))
            out["masterstar_fwhm"] = float(hdr.get("VY_FWHM") or float("nan"))

    vt = ps_dir / "variable_targets.csv"
    if vt.is_file():
        vtdf = pd.read_csv(vt, low_memory=False)
        if "skip_photometry" in vtdf.columns:
            active = vtdf.loc[~vtdf["skip_photometry"].astype(str).str.lower().isin(("1", "true", "yes"))]
        else:
            active = vtdf
        out["active_targets"] = int(len(active))

    phot_s = 0.0
    if not skip_photometry and ms_fits.is_file() and vt.is_file():
        from photometry_core import run_full_photometry_pipeline

        lights = draft_dir / "detrended_aligned" / "lights" / SETUP
        out_phot = ps_dir / "photometry"
        out_phot.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        run_full_photometry_pipeline(
            masterstar_fits_path=ms_fits,
            variable_targets_csv=vt,
            masterstars_csv=ms_csv,
            per_frame_csv_dir=lights,
            detrended_aligned_dir=lights,
            output_dir=out_phot,
            cfg=cfg,
            db=db,
            draft_id=SCRATCH_DRAFT_ID,
        )
        phot_s = time.time() - t0
        meta_path = out_phot / "pipeline_meta.json"
        if meta_path.is_file():
            out["pipeline_meta"] = json.loads(meta_path.read_text(encoding="utf-8"))
    out["photometry_elapsed_s"] = phot_s
    db.conn.close()
    return out


def _p10_residual_p2p(proc_dir: Path, frame_indices: list[int]) -> dict[str, Any]:
    from invariants_runtime import residual_large_scale_p99_adu
    from pipeline import _fit_subtract_preprocess_sky_surface

    rows: list[dict[str, Any]] = []
    for idx in frame_indices:
        matches = sorted(proc_dir.rglob(f"*Light_{idx:03d}.fits"))
        if not matches:
            matches = sorted(proc_dir.rglob(f"*_{idx:03d}.fits"))
        if not matches:
            continue
        fp = matches[0]
        with fits.open(fp, memmap=False) as hd:
            data = np.asarray(hd[0].data, dtype=np.float32)
        p2p = float(residual_large_scale_p99_adu(data, order=1))
        rows.append({"frame_index": idx, "path": str(fp.name), "residual_p99_adu": p2p})
    return {"frames": rows}


def _compare_photometry_vs_anchor(rebuilt: Path, anchor: Path) -> dict[str, Any]:
    from gaia_catalog_id import normalize_gaia_source_id

    rb_lc = rebuilt / "platesolve" / SETUP / "photometry" / "lightcurves"
    an_lc = anchor / "platesolve" / SETUP / "photometry" / "lightcurves"
    if not rb_lc.is_dir() or not an_lc.is_dir():
        return {"error": "missing lightcurve dirs"}

    mag_deltas: list[float] = []
    err_deltas: list[float] = []
    rb_ids: set[str] = set()
    an_ids: set[str] = set()
    for lc in rb_lc.glob("lightcurve_*.csv"):
        cid = lc.stem.replace("lightcurve_", "")
        rb_ids.add(cid)
        anc = an_lc / lc.name
        if not anc.is_file():
            continue
        an_ids.add(cid)
        rdf = pd.read_csv(lc, usecols=["mag_calib_final", "err"])
        adf = pd.read_csv(anc, usecols=["mag_calib_final", "err"])
        n = min(len(rdf), len(adf))
        if n == 0:
            continue
        dm = (rdf["mag_calib_final"].iloc[:n] - adf["mag_calib_final"].iloc[:n]).astype(float)
        de = (rdf["err"].iloc[:n] - adf["err"].iloc[:n]).astype(float)
        mag_deltas.extend(dm[np.isfinite(dm)].tolist())
        err_deltas.extend(de[np.isfinite(de)].tolist())

    def _stats(vals: list[float]) -> dict[str, float]:
        if not vals:
            return {}
        a = np.asarray(vals, dtype=np.float64)
        return {
            "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)),
            "max": float(np.max(a)),
            "n": int(a.size),
        }

    return {
        "mag_calib_final_delta": _stats(mag_deltas),
        "err_delta": _stats(err_deltas),
        "active_only_rebuilt": sorted(rb_ids - an_ids),
        "active_only_anchor": sorted(an_ids - rb_ids),
        "n_common": len(rb_ids & an_ids),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-photometry", action="store_true")
    parser.add_argument("--out", type=Path, default=REPO / "tmp" / "audit_stage3_part0b_results.json")
    args = parser.parse_args()

    prov = _git_provenance()
    print(f"git_hash={prov.get('git_hash')} git_dirty={prov.get('git_dirty')}", flush=True)

    results: dict[str, Any] = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": prov,
        "source_draft": SOURCE_DRAFT_ID,
        "scratch_draft": SCRATCH_DRAFT_ID,
    }
    results["chain"] = _run_chain(skip_photometry=args.skip_photometry)

    from config import AppConfig

    cfg = AppConfig()
    scratch = Path(cfg.archive_root) / "Drafts" / f"draft_{SCRATCH_DRAFT_ID:06d}"
    anchor = Path(cfg.archive_root) / "Drafts" / ANCHOR_SNAPSHOT
    old_proc = Path(cfg.archive_root) / "Drafts" / f"draft_{SOURCE_DRAFT_ID:06d}" / "processed" / "lights"
    new_proc = scratch / "processed" / "lights"

    results["p10_old_proc"] = _p10_residual_p2p(old_proc, P10_FRAME_INDICES)
    results["p10_new_proc"] = _p10_residual_p2p(new_proc, P10_FRAME_INDICES)
    if not args.skip_photometry:
        results["photometry_compare"] = _compare_photometry_vs_anchor(scratch, anchor)

    # threshold ADU diagnostic from first aligned frame if available
    try:
        from pipeline import _dao_convolved_background_rms_adu, _mean_bin2d_for_dao, _dao_auto_binning_factor
        from astropy.stats import sigma_clipped_stats

        al_dir = scratch / "detrended_aligned" / "lights" / SETUP
        al_fits = sorted(al_dir.glob("*.fits"))
        if al_fits:
            with fits.open(al_fits[0], memmap=False) as hd:
                arr = np.asarray(hd[0].data, dtype=np.float32)
            _, med, _ = sigma_clipped_stats(arr, sigma=3.0, maxiters=3)
            data0 = np.nan_to_num((arr - med).astype(np.float32))
            bfac = _dao_auto_binning_factor(*data0.shape)
            data_dao, _ = _mean_bin2d_for_dao(data0, bfac)
            rms_conv, rel = _dao_convolved_background_rms_adu(data_dao, fwhm_px=3.2 / max(1, bfac))
            thr_sigma = float(AppConfig().masterstar_dao_threshold_sigma)
            results["threshold_adu_at_sigma"] = {
                "rms_conv": rms_conv,
                "threshold_sigma": thr_sigma,
                "threshold_adu": thr_sigma * rms_conv,
                "kernel_rel_err": rel,
            }
    except Exception as exc:  # noqa: BLE001
        results["threshold_adu_error"] = str(exc)

    results["finished_utc"] = datetime.now(timezone.utc).isoformat()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
