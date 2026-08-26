# -*- coding: ascii -*-
"""C6-0 R1' production-path 516 chain. Isolated c592ecf + A files from 0684ba9."""
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
SESSION = Path(__file__).resolve().parent
LIVE = ROOT / "Archive" / "Drafts" / "draft_000516"
SETUP = "NoFilter_60_2"
WT = ROOT / ".worktrees" / "c6_r1p_c592ecf"
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


def _copy_cal_sidecars(work_root: Path, ps_dst: Path) -> list[str]:
    copied: list[str] = []
    for name in ("cal_diag.json", "draft_manifest.json", "sat_diag.json"):
        src = LIVE / name
        if src.is_file():
            shutil.copy2(src, work_root / name)
            shutil.copy2(src, ps_dst / name)
            copied.append(name)
    return copied


def _copy_production_inputs(work_root: Path) -> tuple[Path, Path]:
    ps_src = LIVE / "platesolve" / SETUP
    lights_src = LIVE / "detrended_aligned" / "lights" / SETUP
    ps_dst = work_root / "platesolve" / SETUP
    lights_dst = work_root / "detrended_aligned" / "lights" / SETUP
    if ps_dst.exists():
        shutil.rmtree(ps_dst)
    shutil.copytree(
        ps_src,
        ps_dst,
        ignore=shutil.ignore_patterns("photometry", "_hrd_cache", "*.pdf"),
    )
    if lights_dst.exists():
        shutil.rmtree(lights_dst)
    shutil.copytree(lights_src, lights_dst)
    copied = _copy_cal_sidecars(work_root, ps_dst)
    out_phot = ps_dst / "photometry"
    if out_phot.exists():
        shutil.rmtree(out_phot)
    out_phot.mkdir(parents=True, exist_ok=True)
    if "cal_diag.json" not in copied:
        raise RuntimeError("INV-CAL-01 harness: live 516 cal_diag.json missing; refusing to run")
    return ps_dst, lights_dst


def _inv_cal_status(phot_dir: Path) -> dict:
    meta_p = phot_dir / "pipeline_meta.json"
    out: dict = {"meta": meta_p.is_file(), "inv_cal_01": None, "calibration_mode": None}
    if not meta_p.is_file():
        return out
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    out["calibration_mode"] = meta.get("calibration_mode")
    inv = meta.get("invariants") or meta.get("invariant_results") or {}
    if isinstance(inv, dict):
        out["inv_cal_01"] = inv.get("INV-CAL-01") or inv.get("inv-cal-01")
    checks = meta.get("invariant_checks") or []
    if isinstance(checks, list):
        for row in checks:
            if isinstance(row, dict) and str(row.get("id") or row.get("name") or "") == "INV-CAL-01":
                out["inv_cal_01"] = row
    parent = phot_dir.resolve().parent
    out["cal_diag_parent"] = (parent / "cal_diag.json").is_file()
    out["cal_diag_parent2"] = (parent.parent / "cal_diag.json").is_file()
    return out


def main() -> int:
    src_py = WT / "src_py"
    if not src_py.is_dir():
        raise SystemExit(f"missing R1p worktree {WT}")
    _head_src = str((ROOT / "src_py").resolve())
    sys.path[:] = [p for p in sys.path if str(Path(p).resolve()) != _head_src]
    sys.path.insert(0, str(src_py.resolve()))
    work_root = SESSION / "t3_r1pp"
    label = "c592ecf+A+shim3"

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from config import AppConfig
    from database import VyvarDatabase
    from infolog import end_infolog_session, start_infolog_session
    from pipeline import generate_masterstar_and_catalog
    from photometry_core import run_full_photometry_pipeline

    live_before = {
        "516_csv": sha256_file(LIVE / "platesolve" / SETUP / "masterstars_full_match.csv"),
        "516_fits": sha256_file(LIVE / "platesolve" / SETUP / "MASTERSTAR.fits"),
        "516_epsf": sha256_file(LIVE / "platesolve" / SETUP / "masterstar_epsf.fits"),
    }
    rec: dict = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "run": "r1p",
        "label": label,
        "pipeline_file": generate_masterstar_and_catalog.__code__.co_filename,
        "err": None,
        "live_before": live_before,
    }
    work_root.mkdir(parents=True, exist_ok=True)
    t_copy0 = time.perf_counter()
    ps_dst, lights_dst = _copy_production_inputs(work_root)
    rec["copy_s"] = round(time.perf_counter() - t_copy0, 1)
    rec["n_lights"] = len(list(lights_dst.glob("*.fit*")))
    rec["cal_copied"] = True

    infodir = SESSION / "t3_r1pp_infolog"
    infodir.mkdir(parents=True, exist_ok=True)
    start_infolog_session(infodir)
    t_ms0 = time.perf_counter()
    try:
        ms_kw: dict = dict(
            archive_path=work_root,
            platesolve_dir=ps_dst,
            platesolve_backend="vyvar",
            plate_solve_fov_deg=1.0,
            catalog_match_max_sep_arcsec=2.0,
            saturate_level_fraction=0.999,
            max_catalog_rows=12000,
            dao_threshold_sigma=4.5,
            catalog_local_gaia_only=True,
            app_config=AppConfig(),
            equipment_id=1,
            draft_id=None,
            setup_name=SETUP,
            masterstar_skip_build=True,
            masterstar_platesolve_skip_solve=True,
            hint_ra_deg=209.48299383556684,
            hint_dec_deg=41.156331644805974,
        )
        rec["ms_out"] = generate_masterstar_and_catalog(**ms_kw)
    except Exception as exc:
        rec["err"] = f"MS {type(exc).__name__}: {exc}"
        logging.exception("C6-0 R1pp MASTERSTAR failed")
        end_infolog_session()
        (SESSION / "t3_r1pp.json").write_text(json.dumps(rec, indent=2, default=str), encoding="ascii")
        return 1
    rec["ms_s"] = round(time.perf_counter() - t_ms0, 1)

    cfg = AppConfig()
    cfg.save_lightcurve_png = False
    cfg.psf_photometry_enabled = False
    cfg.k2_mode = "literature"
    cfg.per_frame_saturation_enabled = True
    cfg.export_err_mode = "calibrated"
    db = VyvarDatabase(cfg.database_path)
    phot_out = ps_dst / "photometry"
    t_ph0 = time.perf_counter()
    try:
        rec["phot_out"] = run_full_photometry_pipeline(
            masterstar_fits_path=ps_dst / "MASTERSTAR.fits",
            variable_targets_csv=ps_dst / "variable_targets.csv",
            masterstars_csv=ps_dst / "masterstars_full_match.csv",
            per_frame_csv_dir=lights_dst,
            detrended_aligned_dir=lights_dst,
            output_dir=phot_out,
            cfg=cfg,
            db=db,
            draft_id=516,
        )
    except Exception as exc:
        rec["err"] = f"PHOT {type(exc).__name__}: {exc}"
        logging.exception("C6-0 R1pp photometry failed")
    rec["phot_s"] = round(time.perf_counter() - t_ph0, 1)
    end_infolog_session()
    try:
        db.conn.close()
    except Exception:
        pass

    rec["inv_cal"] = _inv_cal_status(phot_out)
    rec["live_after"] = {
        "516_csv": sha256_file(LIVE / "platesolve" / SETUP / "masterstars_full_match.csv"),
        "516_fits": sha256_file(LIVE / "platesolve" / SETUP / "MASTERSTAR.fits"),
        "516_epsf": sha256_file(LIVE / "platesolve" / SETUP / "masterstar_epsf.fits"),
    }
    rec["live_unchanged"] = rec["live_after"] == live_before
    rec["live_sha_guard"] = rec["live_after"] == LIVE_SHA
    rec["elapsed_s"] = round(rec.get("copy_s", 0) + rec.get("ms_s", 0) + rec.get("phot_s", 0), 1)
    (SESSION / "t3_r1pp.json").write_text(json.dumps(rec, indent=2, default=str), encoding="ascii")
    print(
        "C6-0 R1pp",
        "err",
        rec.get("err"),
        "copy_s",
        rec.get("copy_s"),
        "ms_s",
        rec.get("ms_s"),
        "phot_s",
        rec.get("phot_s"),
        "inv",
        rec.get("inv_cal"),
    )
    if rec.get("err"):
        return 1
    inv = rec.get("inv_cal") or {}
    if inv.get("cal_diag_parent") is False and inv.get("cal_diag_parent2") is False:
        print("INV-CAL-01 attachment missing after run - INVALID")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
