#!/usr/bin/env python3
"""Palomar 7 B/G/R night_run for colour-term path validation (CT machinery only)."""
from __future__ import annotations

import json
import os
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

os.environ["VYVAR_CT_PROTOTYPE"] = "1"

FIELD_DB = _ROOT / "GAIA_DR3" / "vyvar_gaia_dr3_pal7_field.db"
CONFIG_PATH = _ROOT / "config.json"
RESULT_PATH = _ROOT / "tmp" / "palomar7_bgr_night_run_result.json"
BGR_SOURCE = _ROOT / "Archive" / "palomar7" / "BGR_all"


def _link_bgr_source() -> int:
    BGR_SOURCE.mkdir(parents=True, exist_ok=True)
    for old in BGR_SOURCE.glob("*.fits"):
        old.unlink()
    n = 0
    for filt in ("Blue", "Green", "Red"):
        src = _ROOT / "Archive" / "palomar7" / filt
        if not src.is_dir():
            continue
        for fp in sorted(src.glob("*.fits")):
            dest = BGR_SOURCE / fp.name
            if dest.exists():
                dest.unlink()
            try:
                os.link(fp, dest)
            except OSError:
                import shutil

                shutil.copy2(fp, dest)
            n += 1
    return n


def _set_gaia_db(path: Path) -> dict:
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    orig = {"gaia_db_path": data.get("gaia_db_path")}
    data["gaia_db_path"] = str(path.resolve())
    CONFIG_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return orig


def _restore_gaia(orig: dict) -> None:
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if "gaia_db_path" in orig:
        data["gaia_db_path"] = orig["gaia_db_path"]
    CONFIG_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    from config import AppConfig  # noqa: E402
    from database import VyvarDatabase, get_gaia_db_max_g_mag  # noqa: E402
    from night_run import NightRunParams, run_night_pipeline  # noqa: E402

    sys.path.insert(0, str(_ROOT / "scripts"))
    import pilot_palomar7_phases_ac as pal  # noqa: E402

    if not FIELD_DB.is_file():
        print(f"Missing field DB: {FIELD_DB} — run palomar7_build_field_db.py first")
        return 1

    n_linked = _link_bgr_source()
    ids = pal.phase_a_register()
    orig_cfg = _set_gaia_db(FIELD_DB)

    report: dict = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "n_source_fits": n_linked,
        "source_dir": str(BGR_SOURCE),
        "field_db": str(FIELD_DB),
        "equipment_ids": ids,
        "VYVAR_CT_PROTOTYPE": "1",
    }

    try:
        cfg = AppConfig()
        report["gaia_max_g"] = get_gaia_db_max_g_mag(cfg.gaia_db_path)
        params = NightRunParams(
            source_dir=BGR_SOURCE,
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
        report["night_run_success"] = nr.success
        report["draft_id"] = nr.draft_id
        report["draft_dir"] = str(nr.draft_dir) if nr.draft_dir else None
        report["errors"] = nr.errors
        report["warnings"] = nr.warnings
        report["phase_timings"] = nr.phase_timings
        report["n_lightcurves"] = nr.n_lightcurves
        if nr.draft_dir:
            report["masterstar_stats"] = pal._collect_masterstar_stats(Path(nr.draft_dir))
            ct = Path(nr.draft_dir) / "ct_prototype.csv"
            report["ct_prototype_csv"] = str(ct) if ct.is_file() else None
    finally:
        _restore_gaia(orig_cfg)
        report["config_restored"] = True

    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    RESULT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report.get("night_run_success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
