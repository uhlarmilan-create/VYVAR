#!/usr/bin/env python3
"""Resume draft_000375 plate-solve with Double Cluster coordinate hint (standard Gaia DB)."""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

CONFIG_PATH = _ROOT / "config.json"
STANDARD_GAIA_DB = _ROOT / "GAIA_DR3" / "vyvar_gaia_dr3.db"
RESULT_PATH = _ROOT / "tmp" / "chiandh_continue375_result.json"

DRAFT_ID = 375
EQUIPMENT_ID = 2
FIELD_RA_DEG = 35.175
FIELD_DEC_DEG = 57.133
SETUPS = ("B_20_2", "L_20_2", "R_20_2", "V_20_2")
PHOT_FILTERS = ("B", "L", "R", "V")


def _patch_config() -> dict[str, Any]:
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    orig = {
        "gaia_db_path": data.get("gaia_db_path"),
        "skip_processed_directory": data.get("skip_processed_directory"),
        "psf_photometry_enabled": data.get("psf_photometry_enabled"),
    }
    data["skip_processed_directory"] = True
    data["psf_photometry_enabled"] = False
    if str(data.get("gaia_db_path", "")).strip() != str(STANDARD_GAIA_DB.resolve()):
        data["gaia_db_path"] = str(STANDARD_GAIA_DB.resolve())
    CONFIG_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return orig


def _restore_config(orig: dict[str, Any]) -> None:
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    for key in ("gaia_db_path", "skip_processed_directory", "psf_photometry_enabled"):
        if key in orig:
            data[key] = orig[key]
    CONFIG_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


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
    objctdec = Angle(dec_deg, u.deg).to_string(unit=u.deg, sep=":", alwayssign=True, precision=0, pad=True)
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
        hdr.add_history(
            f"VYVAR chiandh_continue375: pointing hint RA={ra_deg:.3f} Dec={dec_deg:.3f}"
        )
        hdul.flush()


def _inject_all_pointing(draft_dir: Path) -> dict[str, Any]:
    from astropy.io import fits

    lights = draft_dir / "non_calibrated" / "lights"
    injected: list[str] = []
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

    sample = injected[0] if injected else None
    sample_hdr: dict[str, Any] = {}
    if sample:
        h = fits.getheader(draft_dir / sample)
        sample_hdr = {
            "file": sample,
            "VYTARGRA": h.get("VYTARGRA"),
            "VYTARGDE": h.get("VYTARGDE"),
            "OBJCTRA": h.get("OBJCTRA"),
            "OBJCTDEC": h.get("OBJCTDEC"),
        }
    return {
        "n_source_frames": len(injected),
        "n_masterstars": len(masterstars),
        "sample_header": sample_hdr,
        "masterstar_paths": masterstars,
    }


def _ensure_proc_aliases(draft_dir: Path) -> dict[str, int]:
    """Pre-cal aligned frames use native basenames; photometry expects proc_* aliases."""
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


def _solve_setup(
    *,
    draft_dir: Path,
    setup: str,
    pipeline,
    cfg,
) -> dict[str, Any]:
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
        draft_id=DRAFT_ID,
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
        "masterstar_skip_build": skip_build,
        "match_rate": solve.get("match_rate"),
        "hint_sep_deg": solve.get("hint_sep_deg") or solve.get("hint_vs_solved_deg"),
        "solved": solve.get("solved"),
        "rms_px": solve.get("rms_px"),
        "crval": crval,
        "catalog_matched": out.get("catalog_matched") if isinstance(out, dict) else None,
        "detected_stars": out.get("detected_stars") if isinstance(out, dict) else None,
        "masterstar_fits": str(out.get("masterstar_fits") or ms_path),
    }


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    report: dict[str, Any] = {
        "draft_id": DRAFT_ID,
        "field_center_deg": [FIELD_RA_DEG, FIELD_DEC_DEG],
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "calibration_mode_expected": "pre_calibrated",
    }

    orig_cfg = _patch_config()
    try:
        from config import AppConfig
        from database import VyvarDatabase, get_gaia_db_max_g_mag
        from night_run import _night_run_platesolve
        from pipeline import AstroPipeline

        cfg = AppConfig()
        report["gaia_db_path"] = str(cfg.gaia_db_path)
        report["gaia_max_g"] = get_gaia_db_max_g_mag(cfg.gaia_db_path)
        report["db_coverage_count"] = _gaia_coverage_guard(Path(cfg.gaia_db_path))

        db = VyvarDatabase(cfg.database_path)
        db.update_obs_draft_center(DRAFT_ID, FIELD_RA_DEG, FIELD_DEC_DEG)

        draft_dir = Path(cfg.archive_root) / "Drafts" / f"draft_{DRAFT_ID:06d}"
        if not draft_dir.is_dir():
            raise FileNotFoundError(f"Missing draft directory: {draft_dir}")

        manifest_path = draft_dir / "draft_manifest.json"
        if manifest_path.is_file():
            report["calibration_mode"] = json.loads(manifest_path.read_text(encoding="utf-8")).get(
                "calibration_mode"
            )

        report["hint_injection"] = _inject_all_pointing(draft_dir)

        aligned = draft_dir / "detrended_aligned"
        if aligned.is_dir():
            shutil.rmtree(aligned)
            report["cleared_detrended_aligned"] = True

        pipeline = AstroPipeline(cfg)
        cfg.sips_dao_fwhm_px = 3.5
        cfg.sips_dao_threshold_sigma = 3.5

        def _prog_cb(i: int, t: int, msg: str) -> None:
            if i == 1 or i == t or i % max(1, t // 10) == 0:
                print(f"[{i}/{t}] {msg}", flush=True)

        solve_results: dict[str, Any] = {}
        for setup in SETUPS:
            print(f"=== plate-solve {setup} ===", flush=True)
            solve_results[setup] = _solve_setup(
                draft_dir=draft_dir,
                setup=setup,
                pipeline=pipeline,
                cfg=cfg,
            )
            print(json.dumps(solve_results[setup], indent=2), flush=True)
        report["platesolve"] = solve_results

        job_ps = {
            "kind": "platesolve",
            "archive_path": str(draft_dir),
            "draft_id": DRAFT_ID,
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
        print("=== alignment + per-frame catalogs ===", flush=True)
        align_out = _night_run_platesolve(
            pending=job_ps,
            ap=draft_dir,
            pipeline=pipeline,
            plan=None,
            progress_cb=_prog_cb,
        )
        report["alignment"] = align_out if isinstance(align_out, dict) else {"ok": True}
        report["proc_aliases"] = _ensure_proc_aliases(draft_dir)

        if os.environ.get("VYVAR_CT_PROTOTYPE", "").strip().lower() in ("1", "true", "yes", "on"):
            sys.path.insert(0, str(_ROOT / "scripts"))
            import chiandh_ct_target_presel as presel  # noqa: E402

            report["presel"] = presel.presel_draft(DRAFT_ID)
            proto = draft_dir / "ct_prototype.csv"
            if proto.is_file():
                proto.unlink()

        from photometry_core import run_full_photometry_pipeline
        from ui_aperture_photometry import _find_phase2a_paths

        setups = _find_phase2a_paths(cfg, DRAFT_ID) or {}
        report["phase2a_setups"] = sorted(setups.keys())
        phot_results: dict[str, Any] = {}
        for nm in sorted(setups.keys()):
            filt = str(nm).split("_")[0]
            if filt not in PHOT_FILTERS:
                continue
            p = setups[nm]
            print(f"=== photometry {nm} ===", flush=True)
            phot_out = run_full_photometry_pipeline(
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
            p2a = phot_out.get("phase2a") or {}
            phot_results[nm] = {
                "n_lightcurves": int(p2a.get("n_lightcurves") or 0),
                "n_targets": int(p2a.get("n_targets") or 0),
            }
        report["photometry"] = phot_results
        report["success"] = all(
            float((solve_results.get(s) or {}).get("match_rate") or 0.0) >= 0.60 for s in SETUPS
        )
    finally:
        _restore_config(orig_cfg)

    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    RESULT_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    return 0 if report.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
