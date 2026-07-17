#!/usr/bin/env python3
"""Chi_and_H clean all-filter overnight run + CT/audit verification dumps.

Fresh import from Archive/Chi_and_H (pre-calibrated, equipment set #2), coordinate hint
injection, standard vyvar_gaia_dr3.db, full photometry (decoupled CT path, apply_color_term=auto).
No VYVAR_CT_PROTOTYPE — presel opt-in only.

Outputs (repo root):
  ct_summary_chiandh_allfilters.csv
  audit_chiandh_allfilters.csv
  platesolve_summary_chiandh_allfilters.csv
  chiandh_allfilters_overnight_result.json
"""
from __future__ import annotations

import json
import math
import os
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401  (repo layout: src_py + dev on sys.path)
_ROOT = _bootstrap.REPO_ROOT
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

CONFIG_PATH = _ROOT / "config.json"
STANDARD_GAIA_DB = _ROOT / "GAIA_DR3" / "vyvar_gaia_dr3.db"
SOURCE_ROOT = _ROOT / "Archive" / "Chi_and_H"
EQUIPMENT_ID = 2
TELESCOPE_ID = 2
LOCATION_ID = 1
FIELD_RA_DEG = 35.175
FIELD_DEC_DEG = 57.133
SETUPS = ("B_20_2", "L_20_2", "R_20_2", "V_20_2")
PHOT_FILTERS = ("B", "L", "R", "V")
FILTER_LABEL = {"B_20_2": "B", "V_20_2": "V", "R_20_2": "Rc", "L_20_2": "L"}
MIN_COMP_CT = 7
MAX_STDERR_RATIO = 0.5

CT_SUMMARY_CSV = _ROOT / "tmp" / "ct_summary_chiandh_allfilters.csv"
AUDIT_CSV = _ROOT / "tmp" / "audit_chiandh_allfilters.csv"
PLATESOLVE_CSV = _ROOT / "tmp" / "platesolve_summary_chiandh_allfilters.csv"
RESULT_JSON = _ROOT / "tmp" / "chiandh_allfilters_overnight_result.json"


def _patch_config() -> dict[str, Any]:
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    orig = {
        "gaia_db_path": data.get("gaia_db_path"),
        "skip_processed_directory": data.get("skip_processed_directory"),
        "psf_photometry_enabled": data.get("psf_photometry_enabled"),
        "apply_color_term": data.get("apply_color_term"),
    }
    data["skip_processed_directory"] = True
    data["psf_photometry_enabled"] = False
    data["apply_color_term"] = "auto"
    if str(data.get("gaia_db_path", "")).strip() != str(STANDARD_GAIA_DB.resolve()):
        data["gaia_db_path"] = str(STANDARD_GAIA_DB.resolve())
    CONFIG_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return orig


def _restore_config(orig: dict[str, Any]) -> None:
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    for key in orig:
        data[key] = orig[key]
    CONFIG_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _fresh_app_config():
    """Reload config from disk — avoid stale in-memory clone after JSON patch."""
    from config import AppConfig

    cfg = AppConfig()
    cfg.__post_init__()
    return cfg


def _gaia_coverage_guard(db_path: Path) -> int:
    if not db_path.is_file():
        raise FileNotFoundError(f"Standard Gaia DB missing: {db_path}")
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM gaia_dr3 WHERE ra BETWEEN 34 AND 36 AND dec BETWEEN 56 AND 58"
        ).fetchone()
        count = int(row[0] if row else 0)
    finally:
        conn.close()
    if count <= 0:
        raise RuntimeError(
            "Standard vyvar_gaia_dr3.db has no stars near h & chi Per (RA 34-36, Dec 56-58). "
            "Extend the G<=16 catalog to that region before solving."
        )
    return count


def _sexagesimal_keywords(ra_deg: float, dec_deg: float) -> tuple[str, str]:
    from astropy.coordinates import Angle
    import astropy.units as u

    objctra = Angle(ra_deg, u.deg).to_string(unit=u.hour, sep=":", precision=0, pad=True)
    objctdec = Angle(dec_deg, u.deg).to_string(
        unit=u.deg, sep=":", alwayssign=True, precision=0, pad=True
    )
    return objctra, objctdec


def _inject_pointing_fits(path: Path, *, ra_deg: float, dec_deg: float) -> None:
    from astropy.io import fits

    objctra, objctdec = _sexagesimal_keywords(ra_deg, dec_deg)
    with fits.open(path, mode="update", memmap=False) as hdul:
        hdr = hdul[0].header
        hdr["VYTARGRA"] = (float(ra_deg), "VYVAR plate-solve hint RA [deg] ICRS")
        hdr["VYTARGDE"] = (float(dec_deg), "VYVAR plate-solve hint Dec [deg] ICRS")
        hdr["RA"] = (float(ra_deg), "Field centre RA [deg] ICRS (VYVAR hint)")
        hdr["DEC"] = (float(dec_deg), "Field centre Dec [deg] ICRS (VYVAR hint)")
        hdr["OBJCTRA"] = (objctra, "Field centre RA HMS (VYVAR hint)")
        hdr["OBJCTDEC"] = (objctdec, "Field centre Dec DMS (VYVAR hint)")
        hdr.add_history(f"VYVAR chiandh_allfilters: pointing hint RA={ra_deg:.3f} Dec={dec_deg:.3f}")
        hdul.flush()


def _inject_all_pointing(draft_dir: Path) -> dict[str, Any]:
    injected: list[str] = []
    lights = draft_dir / "non_calibrated" / "lights"
    for setup in SETUPS:
        setup_dir = lights / setup
        if not setup_dir.is_dir():
            continue
        for fp in sorted(setup_dir.glob("*.fits")):
            _inject_pointing_fits(fp, ra_deg=FIELD_RA_DEG, dec_deg=FIELD_DEC_DEG)
            injected.append(str(fp.relative_to(draft_dir)))

    masterstars: list[str] = []
    for setup in SETUPS:
        ms = draft_dir / "platesolve" / setup / "MASTERSTAR.fits"
        if ms.is_file():
            _inject_pointing_fits(ms, ra_deg=FIELD_RA_DEG, dec_deg=FIELD_DEC_DEG)
            masterstars.append(str(ms.relative_to(draft_dir)))

    return {"n_source_frames": len(injected), "n_masterstars": len(masterstars)}


def _ensure_proc_aliases(draft_dir: Path) -> dict[str, int]:
    aligned_root = draft_dir / "detrended_aligned" / "lights"
    counts: dict[str, int] = {}
    for setup in SETUPS:
        d = aligned_root / setup
        if not d.is_dir():
            continue
        n = 0
        for fp in sorted(d.glob("*.fits")):
            if fp.name.casefold().startswith("proc_"):
                continue
            proc_fits = d / f"proc_{fp.name}"
            if not proc_fits.is_file():
                shutil.copy2(fp, proc_fits)
            from proc_frame_store import proc_csv_path_for_aligned_fits

            proc_csv = proc_csv_path_for_aligned_fits(fp)
            legacy_csv = fp.with_suffix(".csv")
            if legacy_csv.is_file() and not proc_csv.is_file():
                shutil.copy2(legacy_csv, proc_csv)
            n += 1
        counts[setup] = n
    return counts


def _clear_platesolve_artifacts(ps_dir: Path, *, keep_masterstar: bool) -> None:
    if not ps_dir.is_dir():
        ps_dir.mkdir(parents=True, exist_ok=True)
        return
    for child in ps_dir.iterdir():
        if keep_masterstar and child.name.upper() == "MASTERSTAR.FITS":
            continue
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            try:
                child.unlink(missing_ok=True)
            except OSError:
                pass


def _solve_setup(*, draft_dir: Path, setup: str, draft_id: int, pipeline, cfg) -> dict[str, Any]:
    from pipeline import generate_masterstar_and_catalog

    ps_dir = draft_dir / "platesolve" / setup
    ms_path = ps_dir / "MASTERSTAR.fits"
    skip_build = ms_path.is_file()
    _clear_platesolve_artifacts(ps_dir, keep_masterstar=skip_build)

    if skip_build:
        _inject_pointing_fits(ms_path, ra_deg=FIELD_RA_DEG, dec_deg=FIELD_DEC_DEG)

    out = generate_masterstar_and_catalog(
        archive_path=draft_dir,
        max_catalog_rows=20000,
        astrometry_api_key=None,
        platesolve_dir=ps_dir,
        platesolve_backend="vyvar",
        plate_solve_fov_deg=1.25,
        catalog_match_max_sep_arcsec=3.0,
        saturate_level_fraction=0.95,
        n_comparison_stars=150,
        faintest_mag_limit=None,
        dao_threshold_sigma=3.5,
        catalog_local_gaia_only=True,
        app_config=cfg,
        equipment_id=EQUIPMENT_ID,
        draft_id=draft_id,
        setup_name=setup,
        masterstar_fits_only=False,
        masterstar_skip_build=skip_build,
        hint_ra_deg=FIELD_RA_DEG,
        hint_dec_deg=FIELD_DEC_DEG,
    )
    solve = out.get("solve") if isinstance(out, dict) else {}
    solve = solve if isinstance(solve, dict) else {}
    crval = None
    try:
        from astropy.io import fits

        h = fits.getheader(ps_dir / "MASTERSTAR.fits")
        crval = (h.get("CRVAL1"), h.get("CRVAL2"))
    except Exception:  # noqa: BLE001
        crval = None
    return {
        "setup": setup,
        "filter": FILTER_LABEL.get(setup, setup.split("_")[0]),
        "masterstar_skip_build": skip_build,
        "match_rate": solve.get("match_rate"),
        "hint_sep_deg": solve.get("hint_sep_deg") or solve.get("hint_vs_solved_deg"),
        "solved": solve.get("solved"),
        "rms_px": solve.get("rms_px"),
        "crval_ra": crval[0] if crval else None,
        "crval_dec": crval[1] if crval else None,
    }


def _import_fresh(*, pipeline, cfg) -> tuple[int, Path]:
    from draft_provenance import (
        CALIBRATION_MODE_PRE,
        apply_pre_calibrated_import_plan,
        record_draft_calibration_provenance,
    )
    from importer import smart_import_session, smart_scan_source

    plan = smart_scan_source(
        source_root=SOURCE_ROOT,
        calibration_library_root=cfg.calibration_library_root,
        masterdark_validity_days=int(cfg.masterdark_validity_days),
        masterflat_validity_days=int(cfg.masterflat_validity_days),
        db=pipeline.db,
        id_equipments=EQUIPMENT_ID,
        id_telescope=TELESCOPE_ID,
    )
    lights_bad = any(
        r.type == "Lights" and r.status in ("missing", "empty") for r in plan.scan_rows
    )
    if lights_bad or not plan.lights_files:
        raise RuntimeError(f"No light frames found under {SOURCE_ROOT}")

    apply_pre_calibrated_import_plan(plan)
    import_result = smart_import_session(
        plan=plan,
        pipeline=pipeline,
        id_equipment=EQUIPMENT_ID,
        id_telescope=TELESCOPE_ID,
        id_location=LOCATION_ID,
    )
    if getattr(import_result, "draft_id", None) is None:
        raise RuntimeError("Import did not return draft_id")

    draft_id = int(import_result.draft_id)
    ap = Path(str(import_result.archive_path))
    ap_root = ap.parent if ap.name.casefold() == "non_calibrated" else ap
    record_draft_calibration_provenance(
        db=pipeline.db,
        archive_path=ap_root,
        draft_id=draft_id,
        calibration_mode=CALIBRATION_MODE_PRE,
    )
    return draft_id, ap_root.resolve()


def _run_ram_qc(*, draft_id: int, draft_dir: Path, pipeline, cfg, progress_cb) -> None:
    from pipeline import run_draft_ram_calibration_qc_to_obs_files

    run_draft_ram_calibration_qc_to_obs_files(
        db=pipeline.db,
        draft_id=draft_id,
        archive_path=draft_dir,
        master_dark_path=None,
        masterflat_by_filter={},
        masterflat_by_obs_key=None,
        master_dark_by_obs_key=None,
        equipment_id=EQUIPMENT_ID,
        pipeline_config=cfg,
        progress_cb=progress_cb,
    )


# --- CT dump helpers (from chiandh_ct_dump_bvr_375) ---

def _norm_cid(val: Any) -> str:
    from gaia_catalog_id import normalize_gaia_source_id

    return str(normalize_gaia_source_id(val) or "").strip()


def _flux_to_mag(flux: float) -> float:
    if not math.isfinite(flux) or flux <= 0:
        return float("nan")
    return float(-2.5 * math.log10(flux))


def _build_comp_mag_inst(proc_dir: Path, comp_ids: list[str]) -> dict[str, np.ndarray]:
    proc_files = sorted(proc_dir.glob("proc_*.csv"))
    if not proc_files:
        proc_files = sorted(proc_dir.glob("*.csv"))
    n = len(proc_files)
    out: dict[str, np.ndarray] = {cid: np.full(n, float("nan"), dtype=np.float64) for cid in comp_ids}
    id_set = set(comp_ids)
    for i, path in enumerate(proc_files):
        try:
            df = pd.read_csv(path, low_memory=False, dtype={"catalog_id": str})
        except Exception:  # noqa: BLE001
            continue
        if "catalog_id" not in df.columns:
            continue
        df = df.copy()
        df["_nid"] = df["catalog_id"].map(_norm_cid)
        sub = df[df["_nid"].isin(id_set)]
        flux_col = "dao_flux" if "dao_flux" in sub.columns else "flux"
        if flux_col not in sub.columns:
            continue
        for _, row in sub.iterrows():
            cid = str(row["_nid"])
            if cid not in out:
                continue
            flux = float(pd.to_numeric(row.get(flux_col), errors="coerce"))
            out[cid][i] = _flux_to_mag(flux)
    return out


def _comp_quality_from_df(comp_df: pd.DataFrame) -> dict[str, dict]:
    q: dict[str, dict] = {}
    for _, row in comp_df.iterrows():
        cid = _norm_cid(row.get("catalog_id", row.get("name")))
        if not cid:
            continue
        usable = True
        for col in ("is_usable", "photometry_ok"):
            if col in row.index:
                v = row.get(col)
                if str(v).strip().lower() in ("false", "0", "no"):
                    usable = False
        q[cid] = {"quality": "good" if usable else "excluded"}
    return q


def _comp_catalog_mag(comp_df: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for _, row in comp_df.iterrows():
        cid = _norm_cid(row.get("catalog_id", row.get("name")))
        if not cid:
            continue
        for col in ("phot_g_mean_mag", "catalog_mag", "mag"):
            v = pd.to_numeric(row.get(col), errors="coerce")
            if math.isfinite(float(v)):
                out[cid] = float(v)
                break
    return out


def _comp_bp_rp(comp_df: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for _, row in comp_df.iterrows():
        cid = _norm_cid(row.get("catalog_id", row.get("name")))
        if not cid:
            continue
        v = pd.to_numeric(row.get("bp_rp"), errors="coerce")
        if math.isfinite(float(v)):
            out[cid] = float(v)
    return out


def _fit_resid_rms(comp_mag_inst, comp_catalog_mag, comp_bp_rp, comp_quality, c1) -> float:
    from photometry_core import _mad_sigma, _safe_polyfit

    usable = [cid for cid, q in comp_quality.items() if q.get("quality") in ("good", "suspect")]
    ys: list[float] = []
    bp_vals: list[float] = []
    for cid in usable:
        bp = float(comp_bp_rp.get(cid, float("nan")))
        if not math.isfinite(bp) or cid not in comp_mag_inst:
            continue
        inst = np.asarray(comp_mag_inst[cid], dtype=np.float64)
        finite = inst[np.isfinite(inst)]
        if finite.size < 5:
            continue
        cat = float(comp_catalog_mag.get(cid, float("nan")))
        if not math.isfinite(cat):
            continue
        y = float(np.nanmedian(cat - finite))
        if not math.isfinite(y):
            continue
        bp_vals.append(bp)
        ys.append(y)
    if len(ys) < 5:
        return float("nan")
    bp_med = float(np.median(np.asarray(bp_vals, dtype=np.float64)))
    xs = np.asarray([b - bp_med for b in bp_vals], dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    p0 = _safe_polyfit(xs, y, 1)
    if p0 is None:
        return float("nan")
    c1_init, zp_init = float(p0[0]), float(p0[1])
    resid = y - (c1_init * xs + zp_init)
    sig = _mad_sigma(resid)
    mask = np.abs(resid) <= 3.0 * float(sig) if math.isfinite(sig) and sig > 0 else np.ones_like(resid, dtype=bool)
    x_cl, y_cl = xs[mask], y[mask]
    if x_cl.size < 2:
        return float("nan")
    resid_final = y_cl - (float(c1) * x_cl + float(np.median(y_cl - float(c1) * x_cl)))
    return float(np.sqrt(np.mean(np.square(resid_final))))


def _ct_summary_for_setup(*, draft_dir: Path, setup: str, cfg) -> dict[str, Any]:
    from photometry_core import (
        _color_term_cat_inst_scatter_pair,
        _check_color_term_extrapolation,
        apply_color_term,
        fit_color_term_c1,
        resolve_apply_color_term,
        should_apply_color_term,
    )

    flt = FILTER_LABEL.get(setup, setup.split("_")[0])
    ps_dir = draft_dir / "platesolve" / setup
    proc_dir = draft_dir / "detrended_aligned" / "lights" / setup

    if not (ps_dir / "comparison_stars.csv").is_file():
        return {"filter": flt, "setup": setup, "error": "missing comparison_stars.csv"}

    apply_ct_toggle = resolve_apply_color_term(cfg, setup)
    if not apply_ct_toggle or flt == "L":
        return {
            "filter": flt,
            "setup": setup,
            "c1": "",
            "c1_stderr": "",
            "stderr_ratio": "",
            "n_comp": "",
            "comp_bp_rp_min": "",
            "comp_bp_rp_max": "",
            "resid_rms": "",
            "comp_scatter_pre": "",
            "comp_scatter_post": "",
            "n_in_range": "",
            "n_ct_ok": "",
            "n_pathB_blocked": "",
            "gate_apply": False,
            "gate_reason": "L/Clear/luminance — CT toggle off",
        }

    comp_df = pd.read_csv(ps_dir / "comparison_stars.csv", low_memory=False)
    comp_df["catalog_id"] = comp_df["catalog_id"].map(_norm_cid)
    comp_ids = [c for c in comp_df["catalog_id"].astype(str).tolist() if c]
    comp_mag_inst = _build_comp_mag_inst(proc_dir, comp_ids)
    comp_catalog_mag = _comp_catalog_mag(comp_df)
    comp_bp_rp = _comp_bp_rp(comp_df)
    comp_quality = _comp_quality_from_df(comp_df)

    c1, c1_stderr, n_comp_used = fit_color_term_c1(
        comp_mag_inst, comp_catalog_mag, comp_bp_rp, comp_quality, min_comp=5, sigma_clip_sigma=3.0
    )
    stderr_ratio = (
        abs(float(c1_stderr) / float(c1))
        if float(c1) != 0.0 and math.isfinite(float(c1_stderr))
        else float("nan")
    )
    scatter_pre, scatter_post = _color_term_cat_inst_scatter_pair(
        comp_mag_inst, comp_catalog_mag, comp_bp_rp, comp_quality, float(c1), min_comp=5, sigma_clip_sigma=3.0
    )
    resid_rms = _fit_resid_rms(comp_mag_inst, comp_catalog_mag, comp_bp_rp, comp_quality, float(c1))
    bp_vals = [float(v) for v in comp_bp_rp.values() if math.isfinite(float(v))]
    comp_min = float(min(bp_vals)) if bp_vals else float("nan")
    comp_max = float(max(bp_vals)) if bp_vals else float("nan")

    apply_ct, gate_reason = should_apply_color_term(
        obs_group=flt,
        c1=float(c1),
        c1_stderr=float(c1_stderr),
        n_comp=int(n_comp_used),
        min_comp_for_ct=MIN_COMP_CT,
        max_stderr_ratio=MAX_STDERR_RATIO,
    )

    targets_path = ps_dir / "photometry" / "active_targets.csv"
    if not targets_path.is_file():
        targets_path = ps_dir / "variable_targets.csv"
    targets = pd.read_csv(targets_path, low_memory=False, dtype={"catalog_id": str})
    targets["catalog_id"] = targets["catalog_id"].map(_norm_cid)

    n_in_range = 0
    n_ct_ok = 0
    n_pathB_blocked = 0
    for _, trow in targets.iterrows():
        tgt_bp = float(pd.to_numeric(trow.get("bp_rp"), errors="coerce"))
        in_range = (
            math.isfinite(tgt_bp)
            and math.isfinite(comp_min)
            and math.isfinite(comp_max)
            and comp_min <= tgt_bp <= comp_max
        )
        if in_range:
            n_in_range += 1
        elif math.isfinite(tgt_bp) and math.isfinite(comp_max) and tgt_bp > comp_max:
            n_pathB_blocked += 1
        if apply_ct and in_range and math.isfinite(tgt_bp):
            in_range_chk = _check_color_term_extrapolation(
                target_bp_rp=tgt_bp,
                comp_bp_rp_values=bp_vals,
                target_name=str(trow.get("catalog_id", "")),
                extrapolation_tol=float(cfg.phase01_ct_extrapolation_tol),
            )
            if in_range_chk:
                _, ct_corr, _ = apply_color_term(
                    np.asarray([0.0]), tgt_bp, comp_bp_rp, comp_quality, float(c1)
                )
                if math.isfinite(ct_corr) and float(c1) != 0.0:
                    n_ct_ok += 1

    return {
        "filter": flt,
        "setup": setup,
        "c1": float(c1),
        "c1_stderr": float(c1_stderr) if math.isfinite(float(c1_stderr)) else "",
        "stderr_ratio": stderr_ratio if math.isfinite(stderr_ratio) else "",
        "n_comp": int(n_comp_used),
        "comp_bp_rp_min": comp_min if math.isfinite(comp_min) else "",
        "comp_bp_rp_max": comp_max if math.isfinite(comp_max) else "",
        "resid_rms": resid_rms if math.isfinite(resid_rms) else "",
        "comp_scatter_pre": scatter_pre if math.isfinite(scatter_pre) else "",
        "comp_scatter_post": scatter_post if math.isfinite(scatter_post) else "",
        "n_in_range": int(n_in_range),
        "n_ct_ok": int(n_ct_ok),
        "n_pathB_blocked": int(n_pathB_blocked),
        "gate_apply": bool(apply_ct),
        "gate_reason": gate_reason,
    }


def _audit_for_setup(*, draft_dir: Path, setup: str) -> dict[str, Any]:
    flt = FILTER_LABEL.get(setup, setup.split("_")[0])
    ps_dir = draft_dir / "platesolve" / setup
    phot_dir = ps_dir / "photometry"
    lc_dir = phot_dir / "lightcurves"

    summary_path = phot_dir / "photometry_summary.csv"
    active_path = phot_dir / "active_targets.csv"
    comp_path = ps_dir / "comparison_stars.csv"

    row: dict[str, Any] = {
        "filter": flt,
        "setup": setup,
        "n_targets": 0,
        "n_with_real_name": 0,
        "n_nan_name": 0,
        "n_comps_with_lc": 0,
        "n_clean_gt0": 0,
        "trust_GREEN": 0,
        "trust_YELLOW": 0,
        "trust_RED": 0,
        "n_lightcurves": 0,
    }

    if active_path.is_file():
        at = pd.read_csv(active_path, low_memory=False, dtype={"catalog_id": str})
        row["n_targets"] = len(at)
        name_col = "vsx_name" if "vsx_name" in at.columns else "name"
        if name_col in at.columns:
            names = at[name_col].astype(str)
            row["n_nan_name"] = int(((names.str.lower() == "nan") | names.isna()).sum())
            row["n_with_real_name"] = int(
                (names.notna() & (names.str.strip() != "") & (names.str.lower() != "nan")).sum()
            )

    if summary_path.is_file():
        sm = pd.read_csv(summary_path, low_memory=False, dtype={"catalog_id": str})
        row["n_lightcurves"] = len(sm)
        if "n_clean" in sm.columns:
            nc = pd.to_numeric(sm["n_clean"], errors="coerce").fillna(0)
            row["n_clean_gt0"] = int((nc > 0).sum())
        if "trust" in sm.columns:
            vc = sm["trust"].astype(str).str.upper().value_counts()
            row["trust_GREEN"] = int(vc.get("GREEN", 0))
            row["trust_YELLOW"] = int(vc.get("YELLOW", 0))
            row["trust_RED"] = int(vc.get("RED", 0))

    if comp_path.is_file() and lc_dir.is_dir():
        n_check = len(list(lc_dir.glob("check_*.csv")))
        row["n_comps_with_lc"] = n_check

    row["trust_counts"] = f"G={row['trust_GREEN']} Y={row['trust_YELLOW']} R={row['trust_RED']}"
    return row


def _dump_verification(*, draft_dir: Path, cfg, solve_results: dict[str, Any]) -> dict[str, Any]:
    ct_rows = [_ct_summary_for_setup(draft_dir=draft_dir, setup=s, cfg=cfg) for s in SETUPS]
    audit_rows = [_audit_for_setup(draft_dir=draft_dir, setup=s) for s in SETUPS]
    ps_rows = [solve_results[s] for s in SETUPS if s in solve_results]

    pd.DataFrame(ct_rows).to_csv(CT_SUMMARY_CSV, index=False)
    pd.DataFrame(audit_rows).to_csv(AUDIT_CSV, index=False)
    pd.DataFrame(ps_rows).to_csv(PLATESOLVE_CSV, index=False)

    return {
        "ct_summary_csv": str(CT_SUMMARY_CSV),
        "audit_csv": str(AUDIT_CSV),
        "platesolve_csv": str(PLATESOLVE_CSV),
        "ct_rows": ct_rows,
        "audit_rows": audit_rows,
    }


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    if "VYVAR_CT_PROTOTYPE" in os.environ:
        del os.environ["VYVAR_CT_PROTOTYPE"]

    report: dict[str, Any] = {
        "source_dir": str(SOURCE_ROOT),
        "field_center_deg": [FIELD_RA_DEG, FIELD_DEC_DEG],
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "calibration_mode": "pre_calibrated",
        "apply_color_term": "auto",
        "VYVAR_CT_PROTOTYPE": "off",
    }

    orig_cfg = _patch_config()
    try:
        cfg = _fresh_app_config()
        from database import VyvarDatabase, get_gaia_db_max_g_mag
        from night_run import _night_run_platesolve
        from pipeline import AstroPipeline

        report["gaia_db_path"] = str(cfg.gaia_db_path)
        report["gaia_max_g"] = get_gaia_db_max_g_mag(cfg.gaia_db_path)
        report["db_coverage_count"] = _gaia_coverage_guard(Path(cfg.gaia_db_path))

        pipeline = AstroPipeline(_fresh_app_config())
        cfg = pipeline.config
        cfg.sips_dao_fwhm_px = 3.5
        cfg.sips_dao_threshold_sigma = 3.5

        print("=== Step 1: Import Chi_and_H (pre-calibrated) ===", flush=True)
        draft_id, draft_dir = _import_fresh(pipeline=pipeline, cfg=cfg)
        report["draft_id"] = draft_id
        report["draft_dir"] = str(draft_dir)

        db = VyvarDatabase(cfg.database_path)
        db.update_obs_draft_center(draft_id, FIELD_RA_DEG, FIELD_DEC_DEG)

        print("=== Step 2: Inject coordinate hint into all source FITS ===", flush=True)
        report["hint_injection"] = _inject_all_pointing(draft_dir)

        def _prog_cb(i: int, t: int, msg: str) -> None:
            if i == 1 or i == t or i % max(1, t // 10) == 0:
                print(f"[{i}/{t}] {msg}", flush=True)

        print("=== Step 3: RAM QC (pre-cal in-place) ===", flush=True)
        _run_ram_qc(draft_id=draft_id, draft_dir=draft_dir, pipeline=pipeline, cfg=cfg, progress_cb=_prog_cb)

        aligned = draft_dir / "detrended_aligned"
        if aligned.is_dir():
            shutil.rmtree(aligned)
            report["cleared_detrended_aligned"] = True

        print("=== Step 4: Per-group plate-solve ===", flush=True)
        solve_results: dict[str, Any] = {}
        for setup in SETUPS:
            print(f"--- plate-solve {setup} ---", flush=True)
            solve_results[setup] = _solve_setup(
                draft_dir=draft_dir, setup=setup, draft_id=draft_id, pipeline=pipeline, cfg=cfg
            )
            print(json.dumps(solve_results[setup], indent=2), flush=True)

        job_ps = {
            "kind": "platesolve",
            "archive_path": str(draft_dir),
            "draft_id": draft_id,
            "id_equipment": EQUIPMENT_ID,
            "astrometry_api_key": "",
            "platesolve_backend": "vyvar",
            "plate_solve_fov_deg": 1.25,
            "max_extra_platesolve": 0,
            "catalog_match_max_sep_arcsec": 3.0,
            "saturate_level_fraction": 0.95,
            "max_catalog_rows": 20000,
            "n_comparison_stars": 150,
            "faintest_mag_limit": None,
            "dao_threshold_sigma": 3.5,
            "dao_fwhm_px": 3.5,
            "max_control_points": 250,
            "min_detected_stars": 200,
            "max_detected_stars": 4000,
            "build_masterstar_and_catalogs": False,
            "masterstar_candidate_paths": [],
            "masterstar_selection_pct": 10.0,
        }
        print("=== Step 5: Alignment + per-frame catalogs ===", flush=True)
        align_out = _night_run_platesolve(
            pending=job_ps, ap=draft_dir, pipeline=pipeline, plan=None, progress_cb=_prog_cb
        )
        report["alignment"] = align_out if isinstance(align_out, dict) else {"ok": True}
        report["proc_aliases"] = _ensure_proc_aliases(draft_dir)

        from photometry_core import run_full_photometry_pipeline
        from ui_aperture_photometry import _find_phase2a_paths

        setups = _find_phase2a_paths(cfg, draft_id) or {}
        report["phase2a_setups"] = sorted(setups.keys())
        phot_results: dict[str, Any] = {}

        print("=== Step 6: Full photometry (all filters) ===", flush=True)
        for nm in sorted(setups.keys()):
            filt = str(nm).split("_")[0]
            if filt not in PHOT_FILTERS:
                continue
            p = setups[nm]
            print(f"--- photometry {nm} ---", flush=True)
            phot_out = run_full_photometry_pipeline(
                masterstar_fits_path=Path(p["masterstar_fits"]),
                variable_targets_csv=Path(p["obs_group_dir"]) / "variable_targets.csv",
                masterstars_csv=Path(p["obs_group_dir"]) / "masterstars_full_match.csv",
                per_frame_csv_dir=Path(p["per_frame_csv_dir"]),
                detrended_aligned_dir=Path(p["detrended_aligned_dir"]),
                output_dir=Path(p["output_dir"]),
                cfg=_fresh_app_config(),
                db=db,
                draft_id=draft_id,
            )
            p2a = phot_out.get("phase2a") or {}
            phot_results[nm] = {
                "n_lightcurves": int(p2a.get("n_lightcurves") or 0),
                "n_targets": int(p2a.get("n_targets") or 0),
            }
            print(json.dumps(phot_results[nm], indent=2), flush=True)

        report["photometry"] = phot_results
        report["platesolve"] = solve_results

        print("=== Step 7: Verification dumps ===", flush=True)
        report["verification"] = _dump_verification(draft_dir=draft_dir, cfg=cfg, solve_results=solve_results)

        report["success"] = all(
            float((solve_results.get(s) or {}).get("match_rate") or 0.0) >= 0.60 for s in SETUPS
        )
    finally:
        _restore_config(orig_cfg)

    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    RESULT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str), flush=True)
    return 0 if report.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
