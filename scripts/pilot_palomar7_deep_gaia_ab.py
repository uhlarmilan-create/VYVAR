#!/usr/bin/env python3
"""Part A (deep Gaia field DB + gate) and Part B (aperture-vs-PSF on draft 364 Luminance_180_2)."""
from __future__ import annotations

import importlib.util
import json
import logging
import math
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

CONFIG_PATH = _ROOT / "config.json"
FIELD_DB = _ROOT / "GAIA_DR3" / "vyvar_gaia_dr3_pal7_field.db"
DRAFT_ID = 364
SETUP = "Luminance_180_2"
PAL_RA = 272.684
PAL_DEC = -7.208
CONE_RADIUS_DEG = 0.35
MAG_LIMIT_INITIAL = 20.0
MAG_LIMIT_CAP = 19.5
RESULT_PATH = _ROOT / "pilot_palomar7_deep_gaia_ab_result.json"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger("pal7_deep_gaia")


def _load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _save_config(data: dict[str, Any]) -> None:
    CONFIG_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _set_config_keys(*, gaia_db_path: str | None = None, psf_enabled: bool | None = None) -> dict[str, Any]:
    data = _load_config()
    orig = {"gaia_db_path": data.get("gaia_db_path"), "psf_photometry_enabled": data.get("psf_photometry_enabled")}
    if gaia_db_path is not None:
        data["gaia_db_path"] = gaia_db_path
    if psf_enabled is not None:
        data["psf_photometry_enabled"] = bool(psf_enabled)
    _save_config(data)
    return orig


def _restore_config(orig: dict[str, Any]) -> None:
    data = _load_config()
    if "gaia_db_path" in orig:
        data["gaia_db_path"] = orig["gaia_db_path"]
    if "psf_photometry_enabled" in orig:
        data["psf_photometry_enabled"] = orig["psf_photometry_enabled"]
    _save_config(data)


def part_a1_introspect() -> dict[str, Any]:
    from database import get_gaia_db_max_g_mag

    db_path = Path(_load_config()["gaia_db_path"])
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    schema = cur.execute("PRAGMA table_info(gaia_dr3)").fetchall()
    n = int(cur.execute("SELECT COUNT(*) FROM gaia_dr3").fetchone()[0])
    gmax = float(cur.execute("SELECT MAX(g_mag) FROM gaia_dr3").fetchone()[0] or float("nan"))
    conn.close()
    return {
        "path": str(db_path),
        "tables": tables,
        "gaia_dr3_columns": [{"name": r[1], "type": r[2], "pk": bool(r[5])} for r in schema],
        "row_count": n,
        "max_g_mag": gmax,
        "get_gaia_db_max_g_mag": float(get_gaia_db_max_g_mag(db_path)),
    }


def part_a2_astroquery_cone() -> tuple[pd.DataFrame, dict[str, Any]]:
    from astroquery.gaia import Gaia

    Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source_lite"
    Gaia.ROW_LIMIT = -1
    meta: dict[str, Any] = {"mag_limit_used": MAG_LIMIT_INITIAL, "capped": False, "method": None}
    center = SkyCoord(ra=PAL_RA * u.deg, dec=PAL_DEC * u.deg, frame="icrs")

    def _minimal_adql(mag_lim: float) -> str:
        return f"""
        SELECT source_id, ra, dec, phot_g_mean_mag AS g_mag, bp_rp
        FROM gaiadr3.gaia_source_lite
        WHERE 1=CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {PAL_RA}, {PAL_DEC}, {CONE_RADIUS_DEG})
        )
        AND phot_g_mean_mag <= {mag_lim}
        """

    def _full_adql(mag_lim: float) -> str:
        return f"""
        SELECT
            source_id,
            ra,
            dec,
            phot_g_mean_mag AS g_mag,
            phot_bp_mean_mag AS bp_mag,
            phot_rp_mean_mag AS rp_mag,
            bp_rp,
            (phot_g_mean_flux_error / NULLIF(phot_g_mean_flux, 0.0)) AS g_flux_error_rel,
            parallax,
            parallax_error,
            parallax_over_error,
            teff_gspphot,
            logg_gspphot,
            mh_gspphot,
            distance_gspphot,
            phot_variable_flag AS var_flag,
            non_single_star
        FROM gaiadr3.gaia_source_lite
        WHERE 1=CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {PAL_RA}, {PAL_DEC}, {CONE_RADIUS_DEG})
        )
        AND phot_g_mean_mag <= {mag_lim}
        """

    last_exc: Exception | None = None
    df: pd.DataFrame | None = None
    for attempt, (method, mag_lim) in enumerate(
        [
            ("cone_search", MAG_LIMIT_INITIAL),
            ("adql_minimal", MAG_LIMIT_INITIAL),
            ("adql_full", MAG_LIMIT_INITIAL),
            ("adql_minimal", MAG_LIMIT_CAP),
            ("cone_search", MAG_LIMIT_CAP),
        ]
    ):
        try:
            LOGGER.info("Gaia query attempt %d method=%s G<=%s", attempt + 1, method, mag_lim)
            if method == "cone_search":
                job = Gaia.cone_search_async(center, radius=CONE_RADIUS_DEG * u.deg)
                raw = job.get_results().to_pandas()
                meta["method"] = "cone_search"
                gcol = "phot_g_mean_mag" if "phot_g_mean_mag" in raw.columns else "g_mag"
                raw = raw.rename(columns={gcol: "g_mag"}) if gcol != "g_mag" and gcol in raw.columns else raw
                if "g_mag" not in raw.columns and "phot_g_mean_mag" in raw.columns:
                    raw["g_mag"] = raw["phot_g_mean_mag"]
                df = raw[pd.to_numeric(raw["g_mag"], errors="coerce") <= float(mag_lim)].copy()
                meta["mag_limit_used"] = float(mag_lim)
            else:
                adql = _minimal_adql(mag_lim) if method == "adql_minimal" else _full_adql(mag_lim)
                job = Gaia.launch_job_async(adql)
                df = job.get_results().to_pandas()
                meta["method"] = method
                meta["mag_limit_used"] = float(mag_lim)
                if mag_lim == MAG_LIMIT_CAP:
                    meta["capped"] = True
            if df is not None and len(df) > 0:
                break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            LOGGER.warning("Gaia query attempt %d failed: %s", attempt + 1, exc)
            time.sleep(min(120, 10 * (2**attempt)))

    if df is None or df.empty:
        raise RuntimeError(f"All Gaia cone query attempts failed: {last_exc}")

    meta["row_count"] = int(len(df))
    if len(df) > 250_000 and meta["mag_limit_used"] > MAG_LIMIT_CAP:
        meta["capped"] = True
        meta["mag_limit_used"] = MAG_LIMIT_CAP
        g = pd.to_numeric(df.get("g_mag"), errors="coerce")
        df = df.loc[g <= MAG_LIMIT_CAP].copy()
        meta["row_count_capped"] = int(len(df))

    if "phot_g_mean_mag" in df.columns and "g_mag" not in df.columns:
        df = df.rename(columns={"phot_g_mean_mag": "g_mag"})
    return df, meta


def part_a3_build_field_db(df: pd.DataFrame) -> dict[str, Any]:
    sys.path.insert(0, str(_ROOT / "GAIA_DR3"))
    from gaia_dr3_make_fast import init_db, insert_dataframe, _normalize_tap_dataframe  # noqa: PLC0415

    FIELD_DB.parent.mkdir(parents=True, exist_ok=True)
    if FIELD_DB.is_file():
        FIELD_DB.unlink()
    conn = sqlite3.connect(str(FIELD_DB))
    conn.execute("PRAGMA journal_mode=WAL;")
    init_db(conn)
    norm = _normalize_tap_dataframe(df)
    n_ins = insert_dataframe(conn, norm)
    for idx_sql in (
        "CREATE INDEX IF NOT EXISTS idx_ra_dec ON gaia_dr3 (ra, dec)",
        "CREATE INDEX IF NOT EXISTS idx_ra ON gaia_dr3 (ra)",
        "CREATE INDEX IF NOT EXISTS idx_dec ON gaia_dr3 (dec)",
        "CREATE INDEX IF NOT EXISTS idx_g_mag ON gaia_dr3 (g_mag)",
    ):
        conn.execute(idx_sql)
    conn.commit()
    n = int(conn.execute("SELECT COUNT(*) FROM gaia_dr3").fetchone()[0])
    gmax = float(conn.execute("SELECT MAX(g_mag) FROM gaia_dr3").fetchone()[0] or float("nan"))
    faintest = float(conn.execute("SELECT MAX(g_mag) FROM gaia_dr3 WHERE g_mag IS NOT NULL").fetchone()[0] or float("nan"))
    conn.close()
    return {"path": str(FIELD_DB), "inserted": int(n_ins), "row_count": n, "faintest_g": faintest, "max_g": gmax}


def part_a4_gate_check(field_db: Path) -> dict[str, Any]:
    from config import AppConfig
    from database import VyvarDatabase, get_gaia_db_max_g_mag
    from pipeline import _query_gaia_local

    orig = _set_config_keys(gaia_db_path=str(field_db.resolve()))
    try:
        cfg = AppConfig()
        db = VyvarDatabase(cfg.database_path)
        draft = Path(cfg.archive_root) / "Drafts" / f"draft_{DRAFT_ID:06d}"
        ps = draft / "platesolve" / SETUP
        ms = ps / "MASTERSTAR.fits"
        center = SkyCoord(ra=PAL_RA * u.deg, dec=PAL_DEC * u.deg, frame="icrs")
        radius = 0.55 / 2.0  # night-run plate_fov_deg
        if ms.is_file():
            with fits.open(ms, memmap=True) as hd:
                crval1 = hd[0].header.get("CRVAL1")
                crval2 = hd[0].header.get("CRVAL2")
                if crval1 is not None and crval2 is not None:
                    try:
                        center = SkyCoord(ra=float(crval1) * u.deg, dec=float(crval2) * u.deg, frame="icrs")
                    except (TypeError, ValueError):
                        pass
        max_mag = float(get_gaia_db_max_g_mag(field_db))
        df = _query_gaia_local(
            center=center,
            radius_deg=float(radius),
            gaia_db_path=field_db,
            max_mag=max_mag,
            max_rows=int(cfg.catalog_query_max_rows or 15000),
        )
        mags = pd.to_numeric(df.get("mag"), errors="coerce") if not df.empty else pd.Series(dtype=float)
        faintest = float(mags.max()) if mags.notna().any() else float("nan")
        return {
            "gate_passed": bool(len(df) > 0 and math.isfinite(faintest) and faintest >= 15.0),
            "n_cone_stars": int(len(df)),
            "faintest_g": faintest,
            "center_ra": float(center.ra.deg),
            "center_dec": float(center.dec.deg),
            "radius_deg": float(radius),
            "max_mag_sql": max_mag,
            "field_db_max_g": max_mag,
        }
    except Exception as exc:  # noqa: BLE001
        _restore_config(orig)
        return {"gate_passed": False, "error": str(exc)}
    finally:
        # keep field db path for Part B if gate passed — restored at end
        pass


def _collect_match_stats(draft_dir: Path, setup: str) -> dict[str, Any]:
    ps = draft_dir / "platesolve" / setup
    stats: dict[str, Any] = {"setup": setup}
    for csv_name in ("masterstars_full_match.csv", "masterstars.csv"):
        p = ps / csv_name
        if not p.is_file():
            continue
        df = pd.read_csv(p, low_memory=False)
        stats["n_detected"] = int(len(df))
        if "catalog_id" in df.columns:
            cid = df["catalog_id"].astype(str).str.strip()
            matched = cid.notna() & (cid != "") & (~cid.str.lower().isin(("nan", "none")))
            stats["n_matched"] = int(matched.sum())
        mag_col = next((c for c in ("mag", "catalog_mag", "phot_g_mean_mag") if c in df.columns), None)
        if mag_col:
            m = pd.to_numeric(df[mag_col], errors="coerce")
            stats["faintest_matched_g"] = float(m.max()) if m.notna().any() else float("nan")
        break
    cone = ps / "field_catalog_cone.csv"
    if cone.is_file():
        cdf = pd.read_csv(cone, low_memory=False)
        if "mag" in cdf.columns:
            cm = pd.to_numeric(cdf["mag"], errors="coerce")
            stats["cone_n"] = int(len(cdf))
            stats["cone_faintest_g"] = float(cm.max()) if cm.notna().any() else float("nan")
    return stats


def part_b_run(field_db: Path, orig_config: dict[str, Any]) -> dict[str, Any]:
    from config import AppConfig
    from database import VyvarDatabase
    from pipeline import astrometry_align_and_build_masterstar, export_per_frame_catalogs
    from photometry_core import run_full_photometry_pipeline
    from psf_photometry import build_epsf_model, get_epsf_fwhm_from_context
    from ui_aperture_photometry import _find_phase2a_paths

    out: dict[str, Any] = {}
    _set_config_keys(gaia_db_path=str(field_db.resolve()), psf_enabled=True)
    cfg = AppConfig()
    cfg.psf_photometry_enabled = True
    db = VyvarDatabase(cfg.database_path)
    draft_dir = Path(cfg.archive_root) / "Drafts" / f"draft_{DRAFT_ID:06d}"

    row = db.fetch_obs_draft_by_id(DRAFT_ID) or {}
    eq_id = int(row.get("EQUIPMENT_ID") or row.get("ID_EQUIPMENTS") or 3)
    cfg.sips_dao_fwhm_px = 2.5
    cfg.sips_dao_threshold_sigma = 3.5

    t0 = time.time()
    ps_out = astrometry_align_and_build_masterstar(
        archive_path=draft_dir.resolve(),
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
    )
    out["platesolve_sec"] = time.time() - t0
    out["platesolve_error"] = ps_out.get("error") if isinstance(ps_out, dict) else None
    out["match_stats"] = _collect_match_stats(draft_dir, SETUP)

    ps = draft_dir / "platesolve" / SETUP
    aligned = draft_dir / "detrended_aligned" / "lights" / SETUP
    ms_fits = ps / "MASTERSTAR.fits"
    ms_csv = ps / "masterstars_full_match.csv"

    epsf_path = build_epsf_model(
        masterstar_fits_path=ms_fits,
        masterstars_csv_path=ms_csv,
        db=db,
        draft_id=DRAFT_ID,
    )
    out["epsf_path"] = str(epsf_path)
    meta_json = ps / "masterstar_epsf_meta.json"
    if meta_json.is_file():
        out["epsf_meta"] = json.loads(meta_json.read_text(encoding="utf-8"))

    fwhm = float(get_epsf_fwhm_from_context(ms_fits, db, DRAFT_ID))
    out["fwhm_px"] = fwhm

    per = export_per_frame_catalogs(
        frames_root=aligned,
        platesolve_dir=ps,
        max_catalog_rows=15000,
        catalog_match_max_sep_arcsec=3.0,
        dao_threshold_sigma=3.5,
        dao_fwhm_px=2.5,
        masterstars_csv=ms_csv,
        masterstar_fits=ms_fits,
        use_master_fast_path=True,
        catalog_local_gaia_only=True,
        app_config=cfg,
        draft_id=DRAFT_ID,
        equipment_id=eq_id,
    )
    out["export_written"] = per.get("written")

    setups = _find_phase2a_paths(cfg, DRAFT_ID, draft_dir_override=None)
    p = setups.get(SETUP) if setups else None
    if not p:
        out["photometry_error"] = f"setup {SETUP} not in phase2a paths"
        return out

    pr = run_full_photometry_pipeline(
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
    out["photometry_keys"] = list(pr.keys()) if isinstance(pr, dict) else []

    diag_path = _ROOT / "scripts" / "diagnose_psf_elongation_362.py"
    spec = importlib.util.spec_from_file_location("diag", diag_path)
    diag = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(diag)

    part_b_df, note, star_df = _extended_part_b(
        aligned, ms_csv, ps, fwhm_px=fwhm, draft_dir=draft_dir, diag=diag
    )
    out["part_b_note"] = note
    diag_dir = draft_dir / "diagnostics" / "psf_aperture_pal7_deep"
    diag_dir.mkdir(parents=True, exist_ok=True)
    mag_csv = diag_dir / "d364_aperture_vs_psf_by_mag.csv"
    crowd_csv = diag_dir / "d364_aperture_vs_psf_crowding.csv"
    stars_csv = diag_dir / "d364_aperture_vs_psf_per_star.csv"
    if not part_b_df.empty:
        part_b_df.to_csv(mag_csv, index=False)
        out["mag_bin_csv"] = str(mag_csv)
    if not star_df.empty:
        star_df.to_csv(stars_csv, index=False)
        out["per_star_csv"] = str(stars_csv)
    crowd_df = star_df.groupby(["mag_bin", "crowding_class"], dropna=False).agg(
        N=("catalog_id", "count"),
        median_rms_aperture=("rms_aperture", "median"),
        median_rms_psf=("rms_psf", "median"),
        median_ratio_psf_aper=("ratio_psf_aper", "median"),
    ).reset_index()
    if not crowd_df.empty:
        crowd_df.to_csv(crowd_csv, index=False)
        out["crowding_csv"] = str(crowd_csv)
    out["mag_bin_table"] = part_b_df.to_dict(orient="records") if not part_b_df.empty else []
    out["crowding_table"] = crowd_df.to_dict(orient="records") if not crowd_df.empty else []
    return out


def _extended_part_b(
    proc_dir: Path,
    masterstars_csv: Path,
    ps_dir: Path,
    *,
    fwhm_px: float,
    draft_dir: Path,
    diag: Any,
) -> tuple[pd.DataFrame, str, pd.DataFrame]:
    """Extended mag bins 16-20 + isolated vs crowded split."""
    part_b_df, note, star_df = diag._part_b_mag_bins(proc_dir, masterstars_csv)
    if star_df.empty:
        return part_b_df, note, star_df

    bins = [
        ("<12", -np.inf, 12.0),
        ("12-13", 12.0, 13.0),
        ("13-14", 13.0, 14.0),
        ("14-15", 14.0, 15.0),
        ("15-16", 15.0, 16.0),
        ("16-17", 16.0, 17.0),
        ("17-18", 17.0, 18.0),
        ("18-19", 18.0, 19.0),
        ("19-20", 19.0, 20.0),
        (">20", 20.0, np.inf),
    ]
    ext_rows: list[dict[str, Any]] = []
    for label, lo, hi in bins:
        sub = star_df[(star_df["catalog_mag"] > lo) & (star_df["catalog_mag"] <= hi)]
        ext_rows.append(
            {
                "mag_bin": label,
                "N": int(len(sub)),
                "median_rms_aperture": float(sub["rms_aperture"].median()) if len(sub) else float("nan"),
                "median_rms_psf": float(sub["rms_psf"].median()) if len(sub) else float("nan"),
                "median_ratio_psf_aper": float(sub["ratio_psf_aper"].median()) if len(sub) else float("nan"),
            }
        )
    ext_df = pd.DataFrame(ext_rows)

    # Crowding classification
    crowd_map: dict[str, str] = {}
    ct_csv = ps_dir / "crowding_targets.csv"
    if ct_csv.is_file():
        ct = pd.read_csv(ct_csv, low_memory=False, dtype={"catalog_id": str})
        cid_col = "catalog_id" if "catalog_id" in ct.columns else None
        blend_col = next((c for c in ("is_blended", "blended", "blend_flag") if c in ct.columns), None)
        if cid_col and blend_col:
            for _, r in ct.iterrows():
                cid = diag._norm_cid(r.get(cid_col))
                if cid:
                    crowd_map[cid] = "crowded" if bool(r.get(blend_col)) else "isolated"

    if not crowd_map:
        ci_json = ps_dir / "crowding_index.json"
        if ci_json.is_file():
            pass  # scalar only; fall through to cone neighbor count

    if not crowd_map:
        cone = ps_dir / "field_catalog_cone.csv"
        if cone.is_file():
            from config import AppConfig
            from psf_photometry import _read_plate_scale_arcsec_px_from_fits

            cdf = pd.read_csv(cone, low_memory=False)
            cra = pd.to_numeric(cdf["ra_deg"], errors="coerce").to_numpy(dtype=float)
            cde = pd.to_numeric(cdf["dec_deg"], errors="coerce").to_numpy(dtype=float)
            cmag = pd.to_numeric(cdf.get("mag"), errors="coerce").to_numpy(dtype=float)
            ms = pd.read_csv(
                masterstars_csv,
                low_memory=False,
                usecols=lambda c: c in ("catalog_id", "ra_deg", "dec_deg", "mag", "catalog_mag", "phot_g_mean_mag"),
                dtype={"catalog_id": str},
            )
            ms["_cid"] = ms["catalog_id"].map(diag._norm_cid)
            mag_col = next((c for c in ("mag", "catalog_mag", "phot_g_mean_mag") if c in ms.columns), None)
            plate_scale = _read_plate_scale_arcsec_px_from_fits(ps_dir / "MASTERSTAR.fits") or 0.39
            iso_r_deg = 2.0 * float(fwhm_px) * float(plate_scale) / 3600.0
            from comp_selection_per_target import _angular_distance_deg_vectorized

            ra_map = pd.to_numeric(ms.get("ra_deg"), errors="coerce")
            de_map = pd.to_numeric(ms.get("dec_deg"), errors="coerce")
            mg_map = pd.to_numeric(ms[mag_col], errors="coerce") if mag_col else pd.Series(np.nan, index=ms.index)
            for i, row in ms.iterrows():
                cid = str(row.get("_cid", ""))
                if not cid:
                    continue
                ra_i = float(ra_map.iloc[i]) if i < len(ra_map) else float("nan")
                de_i = float(de_map.iloc[i]) if i < len(de_map) else float("nan")
                if not (math.isfinite(ra_i) and math.isfinite(de_i)):
                    continue
                cosd = max(math.cos(math.radians(de_i)), 0.2)
                box = (np.abs(cde - de_i) <= iso_r_deg * 1.5) & (np.abs(cra - ra_i) <= iso_r_deg * 1.5 / cosd)
                if not box.any():
                    crowd_map[cid] = "isolated"
                    continue
                d_deg = _angular_distance_deg_vectorized(ra_i, de_i, cra[box], cde[box])
                m_box = cmag[box]
                cm = float(mg_map.iloc[i]) if i < len(mg_map) and math.isfinite(float(mg_map.iloc[i])) else float("nan")
                near = (d_deg > (0.5 * float(plate_scale) / 3600.0)) & (d_deg <= iso_r_deg)
                if math.isfinite(cm):
                    contaminating = near & ((m_box - cm) <= 2.5)
                else:
                    contaminating = near
                crowd_map[cid] = "crowded" if bool(np.any(contaminating)) else "isolated"

    star_df = star_df.copy()
    star_df["_cid"] = star_df["catalog_id"].map(diag._norm_cid)
    star_df["crowding_class"] = star_df["_cid"].map(lambda c: crowd_map.get(str(c), "unknown"))
    star_df["mag_bin"] = pd.cut(
        star_df["catalog_mag"],
        bins=[-np.inf, 12, 13, 14, 15, 16, 17, 18, 19, 20, np.inf],
        labels=["<12", "12-13", "13-14", "14-15", "15-16", "16-17", "17-18", "18-19", "19-20", ">20"],
    )
    return ext_df, note, star_df


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    report: dict[str, Any] = {"started_utc": datetime.now(timezone.utc).isoformat()}
    orig_config = {"gaia_db_path": _load_config().get("gaia_db_path"), "psf_photometry_enabled": _load_config().get("psf_photometry_enabled")}

    # Part A
    report["part_a"] = {}
    report["part_a"]["schema"] = part_a1_introspect()
    print("Part A1 schema:", json.dumps(report["part_a"]["schema"], indent=2))

    try:
        cone_df, cone_meta = part_a2_astroquery_cone()
        report["part_a"]["cone_query"] = cone_meta
        report["part_a"]["cone_query"]["faintest_g"] = float(pd.to_numeric(cone_df.get("g_mag"), errors="coerce").max())
        print(f"Part A2 cone rows={cone_meta.get('row_count')} capped={cone_meta.get('capped')} mag_limit={cone_meta.get('mag_limit_used')}")
    except Exception as exc:  # noqa: BLE001
        report["part_a"]["cone_query_error"] = str(exc)
        _restore_config(orig_config)
        report["restored"] = True
        RESULT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("Part A2 FAILED:", exc)
        return 1

    report["part_a"]["field_db"] = part_a3_build_field_db(cone_df)
    print("Part A3 field DB:", report["part_a"]["field_db"])

    gate = part_a4_gate_check(FIELD_DB)
    report["part_a"]["gate_check"] = gate
    print("Part A4 gate:", gate)

    if not gate.get("gate_passed"):
        _restore_config(orig_config)
        report["part_b"] = {"skipped": True, "reason": "Part A gate failed"}
        report["restored"] = True
        RESULT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return 1

    # Part B — field DB path stays set; PSF enabled temporarily
    try:
        report["part_b"] = part_b_run(FIELD_DB, orig_config)
    except Exception as exc:  # noqa: BLE001
        report["part_b"] = {"error": str(exc)}
        import traceback

        report["part_b"]["traceback"] = traceback.format_exc()
    finally:
        _restore_config(orig_config)
        chk = _load_config()
        report["restored"] = {
            "gaia_db_path": chk.get("gaia_db_path"),
            "psf_photometry_enabled": chk.get("psf_photometry_enabled"),
            "matches_original": chk.get("gaia_db_path") == orig_config["gaia_db_path"]
            and bool(chk.get("psf_photometry_enabled")) == bool(orig_config["psf_photometry_enabled"]),
        }

    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    RESULT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report.get("part_b", {}).get("error") is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
