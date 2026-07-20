#!/usr/bin/env python3
"""TOI-1131.01.b V-band night_run - Newton bin2 pre-calibrated, aperture, common-mode validation."""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401  (repo layout: src_py + dev on sys.path)
_ROOT = _bootstrap.REPO_ROOT
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

CONFIG_PATH = _ROOT / "config.json"
RESULT_PATH = _ROOT / "tmp" / "toi1131_night_run_result.json"
SOURCE_ROOT = _ROOT / "Archive" / "TOI-1131.01.b"
FIELD_CENTER = (248.83158596, 61.60825418)  # WCS center, frame 0
CHIANDH_BIN2_DAO_FWHM_PX = 3.5
PLATE_SCALE_BIN2_ARCSEC_PX = 1.3


def _git_rev_parse_head() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=_ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            )
            .strip()
        )
    except Exception:  # noqa: BLE001
        return ""


def _patch_config(*, skip_processed: bool, psf_enabled: bool) -> dict:
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    orig = {
        "skip_processed_directory": data.get("skip_processed_directory"),
        "psf_photometry_enabled": data.get("psf_photometry_enabled"),
        "sysrem_enabled": data.get("sysrem_enabled"),
        "savgol_detrend_enabled": data.get("savgol_detrend_enabled"),
        "democratic_detrend_enabled": data.get("democratic_detrend_enabled"),
        "phase01_plate_scale_arcsec_per_px": data.get("phase01_plate_scale_arcsec_per_px"),
    }
    data["skip_processed_directory"] = bool(skip_processed)
    data["psf_photometry_enabled"] = bool(psf_enabled)
    data["sysrem_enabled"] = False
    data["savgol_detrend_enabled"] = False
    data["democratic_detrend_enabled"] = False
    data["phase01_plate_scale_arcsec_per_px"] = PLATE_SCALE_BIN2_ARCSEC_PX
    CONFIG_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return orig


def _restore_config(orig: dict) -> None:
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    for key in orig:
        if key in orig:
            data[key] = orig[key]
    CONFIG_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _config_snapshot(cfg: object) -> dict[str, object]:
    return {
        "gaia_db_path": str(getattr(cfg, "gaia_db_path", "") or ""),
        "skip_processed_directory": bool(getattr(cfg, "skip_processed_directory", False)),
        "psf_photometry_enabled": bool(getattr(cfg, "psf_photometry_enabled", False)),
        "sysrem_enabled": bool(getattr(cfg, "sysrem_enabled", False)),
        "savgol_detrend_enabled": bool(getattr(cfg, "savgol_detrend_enabled", False)),
        "comp_slope_significance_k": float(getattr(cfg, "comp_slope_significance_k", 3.0)),
        "phase01_plate_scale_arcsec_per_px": float(
            getattr(cfg, "phase01_plate_scale_arcsec_per_px", float("nan"))
        ),
    }


def _rig_plate_scale_from_db(camera_id: int, telescope_id: int) -> dict:
    import sqlite3  # noqa: PLC0415

    conn = sqlite3.connect(_ROOT / "vyvar.sqlite3")
    cam = conn.execute(
        "SELECT CAMERANAME, PIXELSIZE FROM EQUIPMENTS WHERE ID=?", (camera_id,)
    ).fetchone()
    tel = conn.execute(
        "SELECT TELESCOPENAME, FOCAL FROM TELESCOPE WHERE ID=?", (telescope_id,)
    ).fetchone()
    conn.close()
    pix_um = float(cam[1]) if cam else float("nan")
    focal_mm = float(tel[1]) if tel else float("nan")
    scale_bin1 = 206.265 * pix_um / focal_mm if math.isfinite(pix_um) and focal_mm > 0 else float("nan")
    return {
        "camera": cam[0] if cam else None,
        "telescope": tel[0] if tel else None,
        "pixel_um": pix_um,
        "focal_mm": focal_mm,
        "plate_scale_bin1_arcsec_px": scale_bin1,
        "plate_scale_bin2_expected_arcsec_px": PLATE_SCALE_BIN2_ARCSEC_PX,
    }


def _gaia_coverage_report(ra: float, dec: float, db_path: Path) -> dict:
    import sqlite3  # noqa: PLC0415

    out: dict = {"db_path": str(db_path), "exists": db_path.is_file()}
    if not db_path.is_file():
        return out
    con = sqlite3.connect(str(db_path))
    total = con.execute("SELECT COUNT(*) FROM gaia_dr3").fetchone()[0]
    n = con.execute(
        "SELECT COUNT(*), MIN(g_mag), MAX(g_mag) FROM gaia_dr3 "
        "WHERE ra BETWEEN ? AND ? AND dec BETWEEN ? AND ?",
        (ra - 0.5, ra + 0.5, dec - 0.5, dec + 0.5),
    ).fetchone()
    con.close()
    out["total_stars_db"] = int(total)
    out["cone_0p5deg_count"] = int(n[0])
    out["cone_g_min"] = float(n[1]) if n[1] is not None else None
    out["cone_g_max"] = float(n[2]) if n[2] is not None else None
    return out


def main() -> int:
    from config import AppConfig  # noqa: E402
    from database import get_gaia_db_max_g_mag  # noqa: E402
    from night_run import NightRunParams, run_night_pipeline  # noqa: E402

    sys.path.insert(0, str(_ROOT / "scripts"))
    import chiandh_phases_ac as equip  # noqa: E402
    import pilot_palomar7_phases_ac as pal  # noqa: E402

    fits = sorted(SOURCE_ROOT.glob("*.fits"))
    if not fits:
        print(f"Missing source FITS under {SOURCE_ROOT}")
        return 1

    git_commit = _git_rev_parse_head()
    ids = equip.phase_a_register()
    rig = _rig_plate_scale_from_db(int(ids["camera_id"]), int(ids["telescope_id"]))
    orig_cfg = _patch_config(skip_processed=True, psf_enabled=False)

    cfg_pre = AppConfig()
    gaia_path = Path(str(cfg_pre.gaia_db_path))
    gaia_cov = _gaia_coverage_report(FIELD_CENTER[0], FIELD_CENTER[1], gaia_path)

    report: dict = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "source_dir": str(SOURCE_ROOT),
        "n_source_fits": len(fits),
        "field_center": list(FIELD_CENTER),
        "equipment_ids": ids,
        "rig_plate_scale": rig,
        "gaia_coverage": gaia_cov,
        "catalog_source": "zaloha G<=16 (vyvar_gaia_dr3.db)" if "zaloha" in str(gaia_path) else str(gaia_path),
        "pre_calibrated_mode": True,
        "psf_photometry_enabled": False,
        "sysrem_enabled": False,
        "savgol_detrend_enabled": False,
    }

    try:
        cfg = AppConfig()
        report["config_snapshot"] = _config_snapshot(cfg)
        report["gaia_max_g"] = get_gaia_db_max_g_mag(cfg.gaia_db_path)

        if int(gaia_cov.get("cone_0p5deg_count") or 0) < 50:
            report["error"] = "Insufficient Gaia coverage - build field DB via TAP before run"
            RESULT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(report["error"])
            return 1

        params = NightRunParams(
            source_dir=SOURCE_ROOT,
            equipment_id=int(ids["camera_id"]),
            telescope_id=int(ids["telescope_id"]),
            location_id=int(ids["location_id"]),
            config_path=CONFIG_PATH,
            pre_calibrated_mode=True,
            sysrem_enabled=False,
            plate_fov_deg=1.1,
            dao_fwhm_px=CHIANDH_BIN2_DAO_FWHM_PX,
            dao_threshold_sigma=3.5,
            catalog_match_max_sep_arcsec=3.0,
            max_catalog_rows=20000,
            min_detected_stars=200,
            max_detected_stars=4000,
            max_control_points=250,
        )
        nr = run_night_pipeline(params)
        report["night_run_success"] = nr.success
        report["draft_id"] = nr.draft_id
        report["draft_dir"] = str(nr.draft_dir) if nr.draft_dir else None
        report["errors"] = nr.errors
        report["warnings"] = nr.warnings
        report["phase_timings"] = nr.phase_timings
        report["n_lightcurves"] = nr.n_lightcurves
        report["photometry_completeness"] = nr.photometry_completeness

        if nr.draft_dir:
            report["masterstar_stats"] = pal._collect_masterstar_stats(Path(nr.draft_dir))
            manifest = Path(nr.draft_dir) / "draft_manifest.json"
            if manifest.is_file():
                report["draft_manifest"] = json.loads(manifest.read_text(encoding="utf-8"))
    finally:
        _restore_config(orig_cfg)

    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("night_run_success", "draft_id", "draft_dir", "errors")}, indent=2))
    return 0 if report.get("night_run_success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
