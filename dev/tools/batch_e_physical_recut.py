#!/usr/bin/env python3
"""Batch E physical re-cut: full chain from calibrated lights (not frozen cache).

Entry point: same chain as audit_stage3_part0b_rebuild.py:
  _night_run_preprocess -> _night_run_platesolve -> run_full_photometry_pipeline
Input: Archive/Drafts/draft_000435/calibrated/lights/NoFilter_60_2 (150 FITS)
Scratch: draft_000500 (fresh detrended_aligned + proc CSVs)

Usage:
  python dev/tools/batch_e_physical_recut.py
  python dev/tools/batch_e_physical_recut.py --analyze-only  # post-run metrics from scratch 500
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "dev" / "scripts"))
import _bootstrap  # noqa: E402,F401

sys.path.insert(0, str(_bootstrap.REPO_ROOT / "src_py"))
sys.path.insert(0, str(_bootstrap.REPO_ROOT / "dev"))

SOURCE_DRAFT_ID = 435
SCRATCH_DRAFT_ID = 500
SETUP = "NoFilter_60_2"
ANCHOR_SNAPSHOT = "draft_000435_snapshot_skysurface_20260716"
BATCH_D_SNAPSHOT = ANCHOR_SNAPSHOT  # batch D fingerprints on this snapshot
LOG_PATH = REPO / "tmp" / "batch_e_physical_recut.log"
OUT_JSON = REPO / "tmp" / "batch_e_physical_recut_results.json"

PART0E_SHIFT_PX = 2.0  # Part 0e: unstable when DAO shift exceeds ~2 px vs anchor


def _centroid_shift_targets(scratch: Path, anchor: Path) -> dict[str, Any]:
    """E.2: star-frames where proc x,y shifted vs batch-D anchor proc CSVs."""
    proc_new = scratch / "detrended_aligned" / "lights" / SETUP
    proc_old = anchor / "detrended_aligned" / "lights" / SETUP
    if not proc_new.is_dir() or not proc_old.is_dir():
        return {"error": "proc dirs missing"}
    shifts: list[float] = []
    target_hits: dict[str, int] = {}
    n_frames_fallback = 0
    for proc_n in sorted(proc_new.glob("proc_*.csv")):
        proc_a = proc_old / proc_n.name
        if not proc_a.is_file():
            continue
        dn = pd.read_csv(proc_n, dtype={"catalog_id": str})
        da = pd.read_csv(proc_a, dtype={"catalog_id": str})
        m = da.merge(dn, on="catalog_id", suffixes=("_anchor", "_new"), how="inner")
        if m.empty:
            continue
        dx = pd.to_numeric(m["x_new"], errors="coerce") - pd.to_numeric(m["x_anchor"], errors="coerce")
        dy = pd.to_numeric(m["y_new"], errors="coerce") - pd.to_numeric(m["y_anchor"], errors="coerce")
        dist = np.hypot(dx, dy)
        ok = dist[np.isfinite(dist)]
        if ok.size:
            shifts.extend(ok.tolist())
            big = dist > PART0E_SHIFT_PX
            n_frames_fallback += int(big.sum())
            for cid in m.loc[big, "catalog_id"].astype(str):
                target_hits[cid] = target_hits.get(cid, 0) + 1
    unstable_targets = sorted(target_hits.keys(), key=lambda c: -target_hits[c])
    return {
        "n_star_frames_shift_gt_2px": n_frames_fallback,
        "n_targets_with_unstable_shift": len(unstable_targets),
        "unstable_targets_top": unstable_targets[:25],
        "shift_px_median": float(np.median(shifts)) if shifts else float("nan"),
        "shift_px_p95": float(np.percentile(shifts, 95)) if shifts else float("nan"),
    }
def _setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    root.addHandler(sh)


def _setup_scratch_draft(cfg) -> Path:
    src = Path(cfg.archive_root) / "Drafts" / f"draft_{SOURCE_DRAFT_ID:06d}"
    dst = Path(cfg.archive_root) / "Drafts" / f"draft_{SCRATCH_DRAFT_ID:06d}"
    if dst.is_dir():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    shutil.copytree(src / "calibrated", dst / "calibrated")
    manifest = {
        "draft_id": SCRATCH_DRAFT_ID,
        "calibration_mode": "vyvar_calibrated",
        "batch_e_physical_recut": True,
    }
    (dst / "draft_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return dst


def _ensure_db_draft(db, draft_dir: Path) -> None:
    row = db.conn.execute("SELECT ID FROM OBS_DRAFT WHERE ID=?;", (SCRATCH_DRAFT_ID,)).fetchone()
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


def _run_physical_chain() -> dict[str, Any]:
    from config import AppConfig
    from database import VyvarDatabase
    from night_run import _night_run_platesolve, _night_run_preprocess
    from pipeline import AstroPipeline
    from photometry_core import run_full_photometry_pipeline
    from tools.reference_seed import seed_reference_observatory

    cfg = AppConfig()
    cfg.save_lightcurve_png = False
    cfg.k2_mode = "literature"
    cfg.preprocess_sky_surface_force_reapply = True
    cfg.enable_lacosmic = True
    logging.info(
        "batch E physical re-cut: preprocess_sky_surface_force_reapply=True enable_lacosmic=True "
        "dao_detection_n_equiv=%s admission_sat_peak_frac=%s",
        getattr(cfg, "dao_detection_n_equiv", "?"),
        getattr(cfg, "admission_sat_peak_frac", "?"),
    )

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
        "overwrite_qc_processing": True,
    }

    t0 = time.time()
    _night_run_preprocess(
        pending=job_ms, ap=draft_dir, pipeline=pipeline, progress_cb=lambda i, t, m: None
    )
    preprocess_s = time.time() - t0

    t0 = time.time()
    ps_out = _night_run_platesolve(
        pending=job_ms, ap=draft_dir, pipeline=pipeline, plan=None, progress_cb=lambda i, t, m: None
    )
    platesolve_s = time.time() - t0

    ps_dir = draft_dir / "platesolve" / SETUP
    ms_csv = ps_dir / "masterstars_full_match.csv"
    ms_fits = ps_dir / "MASTERSTAR.fits"
    vt = ps_dir / "variable_targets.csv"
    lights = draft_dir / "detrended_aligned" / "lights" / SETUP
    out_phot = ps_dir / "photometry"
    out_phot.mkdir(parents=True, exist_ok=True)

    phot_s = 0.0
    if ms_fits.is_file() and vt.is_file():
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

    db.conn.close()
    return {
        "draft_dir": str(draft_dir),
        "preprocess_elapsed_s": preprocess_s,
        "platesolve_elapsed_s": platesolve_s,
        "photometry_elapsed_s": phot_s,
        "platesolve_out": ps_out if isinstance(ps_out, dict) else {"ok": True},
        "input_calibrated_lights": str(
            Path(cfg.archive_root) / "Drafts" / f"draft_{SOURCE_DRAFT_ID:06d}" / "calibrated" / "lights" / SETUP
        ),
        "entry_point": "_night_run_preprocess -> _night_run_platesolve -> run_full_photometry_pipeline",
    }


def _scan_cr_headers(scratch: Path) -> dict[str, Any]:
    cal_dir = scratch / "calibrated" / "lights" / SETUP
    n_frames = 0
    n_frames_with_cr = 0
    total_pixels = 0
    per_frame: list[dict[str, Any]] = []
    for fp in sorted(cal_dir.glob("*.fits")):
        n_frames += 1
        with fits.open(fp, memmap=False) as hd:
            hdr = hd[0].header
        npx = int(hdr.get("VY_COSMNPX") or 0)
        if npx > 0:
            n_frames_with_cr += 1
            total_pixels += npx
            per_frame.append({"frame": fp.name, "n_pixels": npx})
    return {
        "n_frames": n_frames,
        "n_frames_with_cr": n_frames_with_cr,
        "total_cr_pixels": total_pixels,
        "sample_frames": per_frame[:10],
    }


def _parse_batch_e_log_lines(log_path: Path) -> dict[str, Any]:
    if not log_path.is_file():
        return {"error": "log missing"}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    e3 = re.findall(r"\[BATCH-E E\.3\] astroscrappy removed (\d+) pixels on frame (\S+)", text)
    e2 = re.findall(r"\[BATCH-E E\.2\] centroid WCS fallback triggered on (\d+) star-frames", text)
    e4 = re.findall(r"\[BATCH-E E\.4\] N_equiv=([\d.]+) applied", text)
    e5 = re.findall(r"\[BATCH-E E\.5\] saturation gate excluded (\d+) comps", text)
    return {
        "E3_lines": [f"astroscrappy removed {n} pixels on frame {f}" for n, f in e3[:5]],
        "E3_total_lines": len(e3),
        "E3_total_pixels_from_log": sum(int(n) for n, _ in e3),
        "E2_lines": [f"centroid fallback triggered on {n} star-frames" for n in e2[:5]],
        "E2_total_fallback_events": sum(int(n) for n in e2),
        "E2_frame_events": len(e2),
        "E4_lines": [f"N_equiv={v} applied" for v in e4[:3]],
        "E4_n_equiv": float(e4[0]) if e4 else None,
        "E5_lines": [f"saturation gate excluded {n} comps" for n in e5],
        "E5_total_excluded_comps": sum(int(n) for n in e5),
    }


def _g89_slope_from_proc_dir(proc_lights: Path, star_ids: list[str] | None = None) -> dict[str, Any]:
    if not proc_lights.is_dir():
        return {"error": "proc dir missing"}
    csvs = sorted(proc_lights.glob("proc_BO_CVn_Light_*.csv"))
    if not csvs:
        csvs = sorted(proc_lights.glob("proc_*.csv"))
    rows: list[dict[str, Any]] = []
    for proc in csvs:
        df = pd.read_csv(proc, dtype={"catalog_id": str})
        for _, pr in df.iterrows():
            if str(pr.get("photometry_ok", "")).lower() not in ("true", "1", "yes"):
                continue
            if str(pr.get("is_usable", "")).lower() not in ("true", "1", "yes"):
                continue
            sid = str(pr.get("catalog_id", ""))
            if star_ids and sid not in star_ids:
                continue
            flux = float(pr.get("flux", float("nan")))
            mag = float(pr.get("mag", float("nan")))
            if not (math.isfinite(flux) and flux > 0 and math.isfinite(mag)):
                continue
            rows.append({"mag": mag, "logF": math.log10(flux), "sid": sid})
    d = pd.DataFrame(rows)
    if d.empty:
        return {"n_star_frames": 0}
    sub = d[(d["mag"] >= 8.0) & (d["mag"] < 9.0)]
    out: dict[str, Any] = {"n_star_frames": int(len(d)), "G8_9_n": int(len(sub))}
    if len(sub) >= 5 and sub["mag"].std() > 1e-9:
        fit = stats.linregress(sub["mag"], sub["logF"])
        out["G8_9_slope_flux"] = float(fit.slope)
        out["G8_9_slope_se"] = float(fit.stderr)
        out["deficit_vs_0p4"] = float(0.4 + fit.slope)
    return out


def _compare_lc_source_file(rebuilt: Path, anchor: Path, unstable_cids: set[str] | None = None) -> dict[str, Any]:
    rb_lc = rebuilt / "platesolve" / SETUP / "photometry" / "lightcurves"
    an_lc = anchor / "platesolve" / SETUP / "photometry" / "lightcurves"
    if not rb_lc.is_dir() or not an_lc.is_dir():
        return {"error": "missing LC dirs"}
    unstable_cids = unstable_cids or set()
    all_deltas: list[float] = []
    n_lcs_changed = 0
    n_lcs = 0
    unstable_deltas: list[float] = []
    per_target: list[dict[str, Any]] = []

    for lc in sorted(rb_lc.glob("lightcurve_*.csv")):
        cid = lc.stem.replace("lightcurve_", "")
        anc = an_lc / lc.name
        if not anc.is_file():
            continue
        n_lcs += 1
        rdf = pd.read_csv(lc, low_memory=False)
        adf = pd.read_csv(anc, low_memory=False)
        if "source_file" not in rdf.columns or "source_file" not in adf.columns:
            continue
        m = adf.merge(rdf, on="source_file", suffixes=("_anchor", "_rebuild"), how="inner")
        if m.empty:
            continue
        dm = (
            pd.to_numeric(m["mag_calib_final_rebuild"], errors="coerce")
            - pd.to_numeric(m["mag_calib_final_anchor"], errors="coerce")
        ).to_numpy(dtype=np.float64)
        fin = dm[np.isfinite(dm)]
        if fin.size == 0:
            continue
        med = float(np.median(fin))
        if np.any(np.abs(fin) > 1e-6):
            n_lcs_changed += 1
        all_deltas.extend(fin.tolist())
        if cid in unstable_cids:
            unstable_deltas.extend(fin.tolist())
        per_target.append({"cid": cid, "n_epochs": int(fin.size), "median_delta_mag": med, "max_abs": float(np.max(np.abs(fin)))})

    def _stats(vals: list[float]) -> dict[str, float | int]:
        if not vals:
            return {"n": 0}
        a = np.asarray(vals, dtype=np.float64)
        return {
            "n": int(a.size),
            "median": float(np.median(a)),
            "p95_abs": float(np.percentile(np.abs(a), 95)),
            "max_abs": float(np.max(np.abs(a))),
        }

    return {
        "n_lcs_common": n_lcs,
        "n_lcs_with_nonzero_mag_delta": n_lcs_changed,
        "mag_delta_all_epochs": _stats(all_deltas),
        "mag_delta_unstable_centroid_targets": _stats(unstable_deltas),
        "top_targets_by_max_abs": sorted(per_target, key=lambda x: -x["max_abs"])[:15],
    }


def _masterstar_detection_delta(scratch: Path, anchor: Path) -> dict[str, Any]:
    rb = scratch / "platesolve" / SETUP / "masterstars_full_match.csv"
    an = anchor / "platesolve" / SETUP / "masterstars_full_match.csv"
    if not rb.is_file() or not an.is_file():
        return {"error": "masterstars csv missing"}
    rdf = pd.read_csv(rb, low_memory=False, dtype={"catalog_id": str})
    adf = pd.read_csv(an, low_memory=False, dtype={"catalog_id": str})
    rb_ids = set(rdf["catalog_id"].astype(str).str.strip())
    an_ids = set(adf["catalog_id"].astype(str).str.strip())
    return {
        "n_rebuilt": len(rb_ids),
        "n_anchor": len(an_ids),
        "entered": sorted(rb_ids - an_ids)[:20],
        "left": sorted(an_ids - rb_ids)[:20],
        "n_entered": len(rb_ids - an_ids),
        "n_left": len(an_ids - rb_ids),
    }


def _compute_shas(draft_root: Path) -> dict[str, Any]:
    from tests.photometry_sha import compute_photometry_sha

    core, nc = compute_photometry_sha(draft_root, include_comp_qa=False)
    ext, ne = compute_photometry_sha(draft_root, include_comp_qa=True)
    return {"sha_core": core, "sha_core_n": nc, "sha_extended": ext, "sha_extended_n": ne}


def analyze_results(cfg, log_path: Path = LOG_PATH) -> dict[str, Any]:
    scratch = Path(cfg.archive_root) / "Drafts" / f"draft_{SCRATCH_DRAFT_ID:06d}"
    anchor = Path(cfg.archive_root) / "Drafts" / ANCHOR_SNAPSHOT
    proc_new = scratch / "detrended_aligned" / "lights" / SETUP
    proc_anchor = anchor / "detrended_aligned" / "lights" / SETUP

    out: dict[str, Any] = {
        "scratch_draft": SCRATCH_DRAFT_ID,
        "anchor_snapshot": ANCHOR_SNAPSHOT,
        "batch_e_execution": _parse_batch_e_log_lines(log_path),
        "E3_cr_header_scan": _scan_cr_headers(scratch),
    }

    if (scratch / "platesolve" / SETUP / "photometry").is_dir():
        out["fingerprints_physical"] = _compute_shas(scratch)
        out["fingerprints_batch_d"] = _compute_shas(anchor)
        out["fingerprints_batch_d_ledger"] = {
            "sha_core": "b9c9489aa88b1df815bf6157911b35af5bb1c42a3b0eaf58995042fcdd007a39",
            "sha_core_n": 325,
            "sha_extended": "65bc826cac433453f689dbc5ab2883e783b7a7c7563092c02cfa443058f48cc2",
            "sha_extended_n": 487,
        }

    out["E2_centroid_shift"] = _centroid_shift_targets(scratch, anchor)
    unstable_cids = set(out["E2_centroid_shift"].get("unstable_targets_top") or [])
    out["lc_compare_vs_batch_d"] = _compare_lc_source_file(scratch, anchor, unstable_cids)
    out["E4_detection_delta"] = _masterstar_detection_delta(scratch, anchor)

    # E.5 G 8-9 slope: anchor proc CSVs vs physical proc CSVs (after full regen + gate)
    out["E5_G89_slope_before"] = _g89_slope_from_proc_dir(proc_anchor)
    out["E5_G89_slope_after"] = _g89_slope_from_proc_dir(proc_new)
    out["E5_G89_batch_b_revised_baseline"] = {
        "G8_9_slope_flux": -0.258,
        "source": "dev/results/CURSOR_RESULT_batch_B_revised.md",
    }

    # E.3 header evidence (worker logs may not reach main log file)
    cr = out.get("E3_cr_header_scan") or {}
    if cr.get("total_cr_pixels", 0) > 0:
        out["batch_e_execution"]["E3_confirm"] = (
            f"astroscrappy removed {cr['total_cr_pixels']} pixels across "
            f"{cr.get('n_frames_with_cr', 0)} frames (VY_COSMNPX headers)"
        )

    # CR star-core safety: brightest stars in G 8-9 should remain photometry_ok
    if proc_new.is_dir():
        ok_bright = 0
        for proc in sorted(proc_new.glob("*.csv"))[:150]:
            df = pd.read_csv(proc, dtype={"catalog_id": str})
            bright = df[(pd.to_numeric(df["mag"], errors="coerce") >= 8) & (pd.to_numeric(df["mag"], errors="coerce") < 9)]
            if not bright.empty:
                ok_bright += int((bright["photometry_ok"].astype(str).str.lower().isin(("true", "1", "yes"))).sum())
        out["E3_bright_star_frames_ok"] = ok_bright

    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze-only", action="store_true", help="Skip chain; analyze scratch 500")
    args = parser.parse_args()

    _setup_logging(LOG_PATH)
    from config import AppConfig

    cfg = AppConfig()
    results: dict[str, Any] = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "source_draft": SOURCE_DRAFT_ID,
        "scratch_draft": SCRATCH_DRAFT_ID,
        "anchor_snapshot": ANCHOR_SNAPSHOT,
    }

    if not args.analyze_only:
        logging.info("Starting batch E physical re-cut from calibrated lights")
        results["chain"] = _run_physical_chain()
        results["chain"]["finished_utc"] = datetime.now(timezone.utc).isoformat()

    results["analysis"] = analyze_results(cfg)
    results["finished_utc"] = datetime.now(timezone.utc).isoformat()

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    logging.info("Wrote %s", OUT_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
