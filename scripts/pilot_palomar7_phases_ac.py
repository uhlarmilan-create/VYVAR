#!/usr/bin/env python3
"""Phase A library registration + Phase C Palomar 7 Luminance pilot (scoped writes)."""
from __future__ import annotations

import json
import math
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import AppConfig  # noqa: E402
from database import VyvarDatabase, get_gaia_db_max_g_mag  # noqa: E402
from night_run import NightRunParams, run_night_pipeline  # noqa: E402
from psf_photometry import _read_plate_scale_arcsec_px_from_fits  # noqa: E402

SOURCE_DIR = Path(r"C:\ASTRO\python\VYVAR\Archive\palomar7\Luminance")
CONFIG_PATH = _ROOT / "config.json"
RESULT_PATH = _ROOT / "pilot_palomar7_result.json"

CAMERA = {
    "name": "QHY 600M",
    "alias": "GAIN/RN APPROX (Telescope Live config unknown)",
    "sensor_type": "CMOS",
    "sensor_size": "9576x6388",
    "pixel_size": 3.76,
    "gain": 1.0,
    "read_noise": 1.6,
    "saturate_adu": 60000.0,
}
TELESCOPE = {
    "name": "Planewave CDK24",
    "alias": "",
    "focal": 3962.0,
    "diameter": 610.0,
}
LOCATION = {
    "name": "El Sauce (Obstech), Rio Hurtado, Chile",
    "lat": -30.4703,
    "lon": -70.7647,
    "elev": 1570.0,
}


def _find_equipment_id(db: VyvarDatabase, name: str) -> int | None:
    row = db.conn.execute(
        "SELECT ID FROM EQUIPMENTS WHERE CAMERANAME = ? ORDER BY ID LIMIT 1;",
        (name,),
    ).fetchone()
    return int(row[0]) if row else None


def _find_telescope_id(db: VyvarDatabase, name: str) -> int | None:
    row = db.conn.execute(
        "SELECT ID FROM TELESCOPE WHERE TELESCOPENAME = ? ORDER BY ID LIMIT 1;",
        (name,),
    ).fetchone()
    return int(row[0]) if row else None


def _find_location_id(db: VyvarDatabase, name: str) -> int | None:
    row = db.conn.execute(
        "SELECT ID FROM LOCATION WHERE PLACENAME = ? ORDER BY ID LIMIT 1;",
        (name,),
    ).fetchone()
    return int(row[0]) if row else None


def phase_a_register() -> dict[str, int]:
    cfg = AppConfig()
    db = VyvarDatabase(cfg.database_path)
    out: dict[str, int] = {}

    eq_id = _find_equipment_id(db, CAMERA["name"])
    if eq_id is None:
        eq_id = db.insert_equipment(
            CAMERA["name"],
            CAMERA["alias"],
            CAMERA["sensor_type"],
            CAMERA["sensor_size"],
            float(CAMERA["pixel_size"]),
        )
        status = "created"
    else:
        status = "reused"
    db.set_equipment_cosmic_params(eq_id, CAMERA["gain"], CAMERA["read_noise"])
    db.conn.execute(
        "UPDATE EQUIPMENTS SET SATURATE_ADU = ? WHERE ID = ?;",
        (float(CAMERA["saturate_adu"]), int(eq_id)),
    )
    db.conn.commit()
    out["camera_id"] = int(eq_id)
    out["camera_status"] = status

    tel_id = _find_telescope_id(db, TELESCOPE["name"])
    if tel_id is None:
        tel_id = db.insert_telescope(
            TELESCOPE["name"],
            TELESCOPE["alias"],
            float(TELESCOPE["diameter"]),
            float(TELESCOPE["focal"]),
        )
        status = "created"
    else:
        status = "reused"
    out["telescope_id"] = int(tel_id)
    out["telescope_status"] = status

    loc_id = _find_location_id(db, LOCATION["name"])
    if loc_id is None:
        loc_id = db.insert_location(
            LOCATION["name"],
            float(LOCATION["lat"]),
            float(LOCATION["lon"]),
            float(LOCATION["elev"]),
        )
        status = "created"
    else:
        status = "reused"
    out["location_id"] = int(loc_id)
    out["location_status"] = status
    return out


def _restore_psf_flag(original: bool) -> None:
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    data["psf_photometry_enabled"] = bool(original)
    CONFIG_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _enable_psf_flag() -> bool:
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    original = bool(data.get("psf_photometry_enabled", False))
    data["psf_photometry_enabled"] = True
    CONFIG_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return original


def _collect_masterstar_stats(draft_dir: Path) -> dict:
    import pandas as pd
    from astropy.io import fits

    stats: dict = {}
    ps_root = draft_dir / "platesolve"
    if not ps_root.is_dir():
        stats["error"] = "no platesolve dir"
        return stats

    setups = [d for d in ps_root.iterdir() if d.is_dir()]
    stats["setups"] = [s.name for s in setups]
    all_fwhm: list[float] = []
    all_scale: list[float] = []
    n_detected = 0
    n_matched = 0
    faintest = float("nan")

    for setup in setups:
        ms_fits = setup / "MASTERSTAR.fits"
        if ms_fits.is_file():
            try:
                sc = _read_plate_scale_arcsec_px_from_fits(ms_fits)
                if sc is not None and math.isfinite(float(sc)):
                    all_scale.append(float(sc))
            except Exception:
                pass
            try:
                with fits.open(ms_fits) as hd:
                    vf = hd[0].header.get("VY_FWHM")
                    if vf is not None and math.isfinite(float(vf)):
                        all_fwhm.append(float(vf))
            except Exception:
                pass

        for csv_name in ("masterstars_full_match.csv", "masterstars.csv"):
            p = setup / csv_name
            if not p.is_file():
                continue
            df = pd.read_csv(p, low_memory=False)
            if "catalog_id" in df.columns:
                matched = df["catalog_id"].notna() & (df["catalog_id"].astype(str).str.strip() != "")
                n_matched += int(matched.sum())
            if "mag" in df.columns:
                m = pd.to_numeric(df["mag"], errors="coerce")
                if m.notna().any():
                    fm = float(m.max())
                    faintest = fm if not math.isfinite(faintest) else max(faintest, fm)
            if "dao_flux" in df.columns or "flux" in df.columns:
                n_detected += len(df)
            break

        cone = setup / "field_catalog_cone.csv"
        if cone.is_file() and math.isnan(faintest):
            cdf = pd.read_csv(cone, usecols=lambda c: c in ("mag",), low_memory=False)
            if "mag" in cdf.columns:
                m = pd.to_numeric(cdf["mag"], errors="coerce")
                if m.notna().any():
                    faintest = float(m.max())

    stats["plate_scale_arcsec_px_median"] = float(sorted(all_scale)[len(all_scale) // 2]) if all_scale else None
    stats["fwhm_px_median"] = float(sorted(all_fwhm)[len(all_fwhm) // 2]) if all_fwhm else None
    stats["n_detected"] = n_detected
    stats["n_matched"] = n_matched
    stats["faintest_matched_mag"] = faintest if math.isfinite(faintest) else None
    return stats


def phase_c_pilot(ids: dict[str, int]) -> dict:
    result: dict = {"started_utc": datetime.now(timezone.utc).isoformat()}
    if not SOURCE_DIR.is_dir():
        result["blocker"] = f"Missing source dir: {SOURCE_DIR}"
        return result

    n_fits = len(list(SOURCE_DIR.glob("*_cal.fits"))) + len(list(SOURCE_DIR.glob("*.fits")))
    result["n_source_fits"] = n_fits

    psf_original = _enable_psf_flag()
    result["psf_flag_original"] = psf_original
    result["psf_flag_set_true"] = True

    try:
        params = NightRunParams(
            source_dir=SOURCE_DIR,
            equipment_id=int(ids["camera_id"]),
            telescope_id=int(ids["telescope_id"]),
            location_id=int(ids["location_id"]),
            config_path=CONFIG_PATH,
            plate_fov_deg=0.55,
            dao_fwhm_px=2.5,
            dao_threshold_sigma=3.5,
            catalog_match_max_sep_arcsec=3.0,
            max_catalog_rows=15000,
            min_detected_stars=200,
            max_detected_stars=2000,
            max_control_points=200,
        )
        nr = run_night_pipeline(params)
        result["night_run_success"] = nr.success
        result["draft_id"] = nr.draft_id
        result["draft_dir"] = str(nr.draft_dir) if nr.draft_dir else None
        result["errors"] = nr.errors
        result["warnings"] = nr.warnings
        result["phase_timings"] = nr.phase_timings

        if nr.errors:
            for e in nr.errors:
                if "index" in e.lower() or "blind" in e.lower() or "platesolv" in e.lower():
                    result["platesolve_blocker"] = e

        if nr.draft_dir:
            result["masterstar_stats"] = _collect_masterstar_stats(Path(nr.draft_dir))
    finally:
        _restore_psf_flag(psf_original)
        result["psf_flag_restored"] = True
        cfg_check = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        result["psf_flag_after_restore"] = bool(cfg_check.get("psf_photometry_enabled", False))

    return result


def main() -> int:
    cfg = AppConfig()
    report: dict = {
        "phase_a": phase_a_register(),
        "phase_b": {
            "gaia_db_path": str(cfg.gaia_db_path),
            "gaia_max_g_mag": get_gaia_db_max_g_mag(cfg.gaia_db_path),
            "vsx_local_db_path": str(cfg.vsx_local_db_path),
            "blind_index_path": str(cfg.blind_index_path),
            "blind_index_exists": Path(cfg.blind_index_path).is_file(),
        },
    }
    report["phase_c"] = phase_c_pilot(report["phase_a"])
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    RESULT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
