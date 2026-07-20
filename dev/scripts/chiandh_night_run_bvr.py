#!/usr/bin/env python3
"""h & chi Persei B/V/Rc native night_run - science-grade CT validation (pre-calibrated)."""
from __future__ import annotations

import json
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

os.environ["VYVAR_CT_PROTOTYPE"] = "1"

CONFIG_PATH = _ROOT / "config.json"
RESULT_PATH = _ROOT / "tmp" / "chiandh_bvr_night_run_result.json"
SOURCE_ROOT = _ROOT / "Archive" / "Chi_and_H"
FIELD_CENTER = (35.15, 57.13)


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


def _patch_field_center_resolve() -> None:
    """FITS have no RA/DEC - force Double Cluster center for preprocess/platesolve."""
    import pipeline as _pipeline  # noqa: E402

    _orig = _pipeline.resolve_preprocess_target_coordinates

    def _resolve(*, db, draft_id, ui_ra_deg, ui_dec_deg):
        ra, de = _orig(
            db=db,
            draft_id=draft_id,
            ui_ra_deg=ui_ra_deg if ui_ra_deg is not None else FIELD_CENTER[0],
            ui_dec_deg=ui_dec_deg if ui_dec_deg is not None else FIELD_CENTER[1],
        )
        if ra is None or de is None:
            return float(FIELD_CENTER[0]), float(FIELD_CENTER[1])
        return ra, de

    _pipeline.resolve_preprocess_target_coordinates = _resolve


def _patch_config(*, skip_processed: bool, psf_enabled: bool) -> dict:
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    orig = {
        "skip_processed_directory": data.get("skip_processed_directory"),
        "psf_photometry_enabled": data.get("psf_photometry_enabled"),
    }
    data["skip_processed_directory"] = bool(skip_processed)
    data["psf_photometry_enabled"] = bool(psf_enabled)
    CONFIG_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return orig


def _restore_config(orig: dict) -> None:
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    for key in ("skip_processed_directory", "psf_photometry_enabled"):
        if key in orig:
            data[key] = orig[key]
    CONFIG_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _config_snapshot(cfg: object) -> dict[str, object]:
    return {
        "gaia_db_path": str(getattr(cfg, "gaia_db_path", "") or ""),
        "blind_index_fine_path": str(getattr(cfg, "blind_index_fine_path", "") or ""),
        "blind_index_wide_path": str(getattr(cfg, "blind_index_wide_path", "") or ""),
        "skip_processed_directory": bool(getattr(cfg, "skip_processed_directory", False)),
        "psf_photometry_enabled": bool(getattr(cfg, "psf_photometry_enabled", False)),
        "VYVAR_CT_PROTOTYPE": os.environ.get("VYVAR_CT_PROTOTYPE", ""),
    }


def _post_platesolve_hook(draft_id: int, draft_dir: Path, _cfg, _pipeline) -> None:
    sys.path.insert(0, str(_ROOT / "scripts"))
    import chiandh_ct_target_presel as presel  # noqa: E402

    reps = presel.presel_draft(draft_id)
    print("CT target presel:", json.dumps(reps, indent=2), flush=True)


def main() -> int:
    from config import AppConfig  # noqa: E402
    from database import get_gaia_db_max_g_mag  # noqa: E402
    from night_run import NightRunParams, run_night_pipeline  # noqa: E402

    sys.path.insert(0, str(_ROOT / "scripts"))
    import chiandh_phases_ac as chi  # noqa: E402
    import pilot_palomar7_phases_ac as pal  # noqa: E402

    if not SOURCE_ROOT.is_dir() or not any(SOURCE_ROOT.rglob("*.fits")):
        print(f"Missing source FITS under {SOURCE_ROOT}")
        return 1

    git_commit = _git_rev_parse_head()

    ids = chi.phase_a_register()
    _patch_field_center_resolve()
    orig_cfg = _patch_config(skip_processed=True, psf_enabled=False)

    report: dict = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "source_dir": str(SOURCE_ROOT),
        "field_center": list(FIELD_CENTER),
        "equipment_ids": ids,
        "VYVAR_CT_PROTOTYPE": "1",
        "skip_processed_directory": True,
        "psf_photometry_enabled": False,
    }

    try:
        cfg = AppConfig()
        report["config_snapshot"] = _config_snapshot(cfg)
        report["gaia_max_g"] = get_gaia_db_max_g_mag(cfg.gaia_db_path)
        params = NightRunParams(
            source_dir=SOURCE_ROOT,
            equipment_id=int(ids["camera_id"]),
            telescope_id=int(ids["telescope_id"]),
            location_id=int(ids["location_id"]),
            config_path=CONFIG_PATH,
            plate_fov_deg=1.25,
            dao_fwhm_px=3.5,
            dao_threshold_sigma=3.5,
            catalog_match_max_sep_arcsec=3.0,
            max_catalog_rows=20000,
            min_detected_stars=200,
            max_detected_stars=4000,
            max_control_points=250,
            post_platesolve_hook=_post_platesolve_hook,
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
            ct = Path(nr.draft_dir) / "ct_prototype.csv"
            report["ct_prototype_csv"] = str(ct) if ct.is_file() else None
            cal = Path(nr.draft_dir) / "calibrated" / "lights"
            if cal.is_dir():
                report["calibrated_setups"] = sorted(d.name for d in cal.iterdir() if d.is_dir())
            ps = Path(nr.draft_dir) / "platesolve"
            if ps.is_dir():
                report["platesolve_setups"] = sorted(
                    d.name for d in ps.iterdir() if d.is_dir() and (d / "MASTERSTAR.fits").is_file()
                )
            try:
                from tests.photometry_sha import (  # noqa: PLC0415
                    PHOTOMETRY_SHA_BASELINE,
                    PHOTOMETRY_SHA_CORE,
                    compute_photometry_sha,
                )

                core_sha, core_n = compute_photometry_sha(Path(nr.draft_dir))
                full_sha, full_n = compute_photometry_sha(Path(nr.draft_dir), include_comp_qa=True)
                report["photometry_sha"] = {
                    "core": core_sha,
                    "core_n": core_n,
                    "core_match": core_sha == PHOTOMETRY_SHA_CORE,
                    "full": full_sha,
                    "full_n": full_n,
                    "full_match": full_sha == PHOTOMETRY_SHA_BASELINE,
                }
            except Exception as exc:  # noqa: BLE001
                report["photometry_sha_error"] = str(exc)
    finally:
        _restore_config(orig_cfg)
        report["config_restored"] = True

    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    RESULT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report.get("night_run_success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
