#!/usr/bin/env python3
"""DY Peg B/V/R native night_run — zaloha-only, pre-calibrated, bin1 aperture baseline."""
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
RESULT_PATH = _ROOT / "tmp" / "dy_peg_draft390_acceptance_result.json"
SOURCE_ROOT = Path(r"C:\ASTRO\python\VYVAR\Archive\DY Peg\lights")
FIELD_CENTER = (347.2133, 17.2156)
EXPECTED_SETUPS = ("B_60_1", "V_60_1", "R_60_1")
PLATE_SCALE_ARCSEC_PX = 206.265 * 3.76 / 1200.0  # ~0.646 @ bin1, FL=1200 mm
CHIANDH_BIN2_DAO_FWHM_PX = 3.5


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
    """FITS have no RA/DEC — always force DY Peg field center."""
    import pipeline as _pipeline  # noqa: E402

    def _resolve(*, db, draft_id, ui_ra_deg, ui_dec_deg):
        del db, draft_id, ui_ra_deg, ui_dec_deg
        return float(FIELD_CENTER[0]), float(FIELD_CENTER[1])

    _pipeline.resolve_preprocess_target_coordinates = _resolve


def _patch_config(*, skip_processed: bool, psf_enabled: bool) -> dict:
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    orig = {
        "skip_processed_directory": data.get("skip_processed_directory"),
        "psf_photometry_enabled": data.get("psf_photometry_enabled"),
        "phase01_plate_scale_arcsec_per_px": data.get("phase01_plate_scale_arcsec_per_px"),
    }
    data["skip_processed_directory"] = bool(skip_processed)
    data["psf_photometry_enabled"] = bool(psf_enabled)
    data["phase01_plate_scale_arcsec_per_px"] = round(PLATE_SCALE_ARCSEC_PX, 4)
    CONFIG_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return orig


def _restore_config(orig: dict) -> None:
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    for key in ("skip_processed_directory", "psf_photometry_enabled", "phase01_plate_scale_arcsec_per_px"):
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
        "auto_fwhm_enabled": bool(getattr(cfg, "auto_fwhm_enabled", True)),
        "phase01_plate_scale_arcsec_per_px": float(
            getattr(cfg, "phase01_plate_scale_arcsec_per_px", float("nan"))
        ),
    }


def _sep_arcsec(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    r = math.radians
    dra = (ra2 - ra1) * math.cos(r((dec1 + dec2) / 2.0))
    ddec = dec2 - dec1
    return math.degrees(math.sqrt(dra * dra + ddec * ddec)) * 3600.0


def _estimate_median_fwhm_px(sample_fits: Path) -> float | None:
    """Quick stellar FWHM estimate on one pre-cal frame (for dao_fwhm seed)."""
    try:
        import numpy as np  # noqa: PLC0415
        from astropy.io import fits  # noqa: PLC0415
        from pipeline import _estimate_dao_fwhm_guess  # noqa: PLC0415

        with fits.open(sample_fits, memmap=True) as hdul:
            data = np.asarray(hdul[0].data, dtype=np.float64)
        med = float(np.nanmedian(data))
        std = float(np.nanstd(data))
        if not math.isfinite(std) or std <= 0:
            return None
        guess = float(_estimate_dao_fwhm_guess(data - med, std))
        fallback = CHIANDH_BIN2_DAO_FWHM_PX * 2.0
        if not math.isfinite(guess) or guess <= 0:
            return fallback
        return max(3.0, min(12.0, guess))
    except Exception:  # noqa: BLE001
        return None


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
    scale = 206.265 * pix_um / focal_mm if math.isfinite(pix_um) and focal_mm > 0 else float("nan")
    return {
        "camera": cam[0] if cam else None,
        "telescope": tel[0] if tel else None,
        "pixel_um": pix_um,
        "focal_mm": focal_mm,
        "plate_scale_arcsec_px": scale,
    }


def _analyze_draft(draft_dir: Path, *, cfg, collect_ms_stats) -> dict:
    import pandas as pd  # noqa: PLC0415
    from night_run import audit_photometry_completeness  # noqa: PLC0415

    out: dict = {
        "lc_per_band": {},
        "completeness_gate": {},
        "trust_totals": {"GREEN": 0, "YELLOW": 0, "RED": 0},
        "dy_peg": {},
        "pdf_overflow_total": 0,
    }
    ps_root = draft_dir / "platesolve"
    if not ps_root.is_dir():
        out["error"] = "no platesolve dir"
        return out

    setups = sorted(d.name for d in ps_root.iterdir() if d.is_dir())
    out["setups_found"] = setups

    for setup in setups:
        band = setup.split("_")[0]
        phot = ps_root / setup / "photometry"
        lc_dir = phot / "lightcurves"
        summary = phot / "photometry_summary.csv"
        n_lc = len(list(lc_dir.glob("lightcurve_*.csv"))) if lc_dir.is_dir() else 0
        out["lc_per_band"][band] = {"setup": setup, "n_lightcurves": n_lc}

        audit = audit_photometry_completeness(phot)
        out["completeness_gate"][setup] = {
            "ok": bool(audit.get("ok")),
            "ratio": audit.get("ratio"),
            "n_summary_rows": audit.get("n_summary_rows"),
            "n_active_targets": audit.get("n_active_targets"),
            "error": audit.get("error"),
        }

        if summary.is_file():
            sm = pd.read_csv(summary, low_memory=False, dtype={"catalog_id": str})
            if "trust" in sm.columns:
                vc = sm["trust"].astype(str).str.upper().value_counts()
                for k in ("GREEN", "YELLOW", "RED"):
                    out["trust_totals"][k] += int(vc.get(k, 0))

    # DY Peg: nearest LC to field center
    ra_c, dec_c = FIELD_CENTER
    best: dict | None = None
    for setup in setups:
        band = setup.split("_")[0]
        phot = ps_root / setup / "photometry"
        summary = phot / "photometry_summary.csv"
        lc_dir = phot / "lightcurves"
        if not summary.is_file() or not lc_dir.is_dir():
            continue
        sm = pd.read_csv(summary, low_memory=False, dtype={"catalog_id": str})
        ra_col = next((c for c in ("ra_deg", "ra", "RA") if c in sm.columns), None)
        dec_col = next((c for c in ("dec_deg", "dec", "DEC") if c in sm.columns), None)
        if ra_col is None or dec_col is None:
            continue
        sm["_ra"] = pd.to_numeric(sm[ra_col], errors="coerce")
        sm["_dec"] = pd.to_numeric(sm[dec_col], errors="coerce")
        sm = sm[sm["_ra"].notna() & sm["_dec"].notna()].copy()
        if sm.empty:
            continue
        sm["_sep"] = sm.apply(
            lambda r: _sep_arcsec(ra_c, dec_c, float(r["_ra"]), float(r["_dec"])),
            axis=1,
        )
        row = sm.loc[sm["_sep"].idxmin()]
        sep = float(row["_sep"])
        cid = str(row.get("catalog_id", row.get("target_id", "")))
        lc_path = None
        for pat in (f"lightcurve_{cid}.csv", f"lightcurve_*{cid}*.csv"):
            hits = list(lc_dir.glob(pat))
            if hits:
                lc_path = hits[0]
                break
        if lc_path is None:
            hits = sorted(lc_dir.glob("lightcurve_*.csv"))
            lc_path = hits[0] if hits else None

        p2p = float("nan")
        n_epochs = 0
        t_span_h = float("nan")
        if lc_path and lc_path.is_file():
            lc = pd.read_csv(lc_path, low_memory=False)
            mag_col = next(
                (c for c in ("mag", "MAG", "ensemble_mag", "cal_mag") if c in lc.columns),
                None,
            )
            t_col = next((c for c in ("bjd", "BJD", "jd", "JD", "time") if c in lc.columns), None)
            if mag_col:
                mags = pd.to_numeric(lc[mag_col], errors="coerce").dropna()
                if len(mags) >= 2:
                    p2p = float(mags.max() - mags.min())
                n_epochs = int(len(mags))
            if t_col:
                times = pd.to_numeric(lc[t_col], errors="coerce").dropna()
                if len(times) >= 2:
                    t_span_h = float((times.max() - times.min()) * 24.0)

        rec = {
            "setup": setup,
            "band": band,
            "sep_arcsec": sep,
            "catalog_id": cid,
            "ra_deg": float(row["_ra"]),
            "dec_deg": float(row["_dec"]),
            "vsx_name": str(row.get("vsx_name", row.get("name", ""))),
            "lc_file": str(lc_path) if lc_path else None,
            "n_epochs": n_epochs,
            "peak_to_peak_mag": p2p,
            "time_span_hours": t_span_h,
        }
        if best is None or sep < float(best.get("sep_arcsec", 9999.0)):
            best = rec
        out["dy_peg"].setdefault("per_band", {})[band] = rec

    if best:
        out["dy_peg"]["best_match"] = best

    out["masterstar_stats"] = collect_ms_stats(draft_dir)

    # PDF overflow — regenerate each setup with layout verify (night_run PDFs omit verify mode)
    out["pdf_overflow_by_setup"] = {}
    pdf_overflow_total = 0
    try:
        from photometry_report import generate_photometry_report  # noqa: PLC0415

        for setup in setups:
            verify_pdf = (
                ps_root / setup / f"VYVAR_report_{setup}_overflow_verify.pdf"
            )
            generate_photometry_report(
                draft_dir,
                setup,
                verify_pdf,
                verify_overflow=True,
            )
            n_v = int(getattr(generate_photometry_report, "last_overflow_violations", 0))
            out["pdf_overflow_by_setup"][setup] = n_v
            pdf_overflow_total += n_v
        out["pdf_overflow_total"] = pdf_overflow_total
    except Exception as exc:  # noqa: BLE001
        out["pdf_overflow_error"] = str(exc)
        out["pdf_overflow_total"] = pdf_overflow_total

    out["zaloha"] = {
        "gaia_db_path": str(getattr(cfg, "gaia_db_path", "")),
        "gaia_max_g": None,
    }
    try:
        from database import get_gaia_db_max_g_mag  # noqa: PLC0415

        out["zaloha"]["gaia_max_g"] = get_gaia_db_max_g_mag(cfg.gaia_db_path)
    except Exception:  # noqa: BLE001
        pass

    return out


def main() -> int:
    from config import AppConfig  # noqa: E402
    from database import get_gaia_db_max_g_mag  # noqa: E402
    from night_run import NightRunParams, run_night_pipeline  # noqa: E402

    sys.path.insert(0, str(_ROOT / "scripts"))
    import chiandh_phases_ac as equip  # noqa: E402
    import pilot_palomar7_phases_ac as pal  # noqa: E402

    if not SOURCE_ROOT.is_dir() or not any(SOURCE_ROOT.glob("*.fits")):
        print(f"Missing source FITS under {SOURCE_ROOT}")
        return 1

    sample_b = next(SOURCE_ROOT.glob("*_B_*.fits"), None)
    measured_fwhm = _estimate_median_fwhm_px(sample_b) if sample_b else None
    dao_fwhm_seed = float(measured_fwhm) if measured_fwhm else CHIANDH_BIN2_DAO_FWHM_PX * 2.0

    git_commit = _git_rev_parse_head()
    ids = equip.phase_a_register()
    rig = _rig_plate_scale_from_db(int(ids["camera_id"]), int(ids["telescope_id"]))

    _patch_field_center_resolve()
    orig_cfg = _patch_config(skip_processed=True, psf_enabled=False)

    report: dict = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "source_dir": str(SOURCE_ROOT),
        "field_center": list(FIELD_CENTER),
        "equipment_ids": ids,
        "rig_plate_scale": rig,
        "bin1_plate_scale_expected_arcsec_px": PLATE_SCALE_ARCSEC_PX,
        "plate_fov_deg": 0.72,
        "dao_fwhm_px_seed": dao_fwhm_seed,
        "dao_fwhm_sample_fits": str(sample_b) if sample_b else None,
        "skip_processed_directory": True,
        "psf_photometry_enabled": False,
        "pre_calibrated_mode": True,
        "filters": ["B", "V", "R"],
        "expected_setups": list(EXPECTED_SETUPS),
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
            pre_calibrated_mode=True,
            plate_fov_deg=0.72,
            dao_fwhm_px=dao_fwhm_seed,
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
            draft_dir = Path(nr.draft_dir)
            report["masterstar_stats"] = pal._collect_masterstar_stats(draft_dir)
            cal = draft_dir / "calibrated" / "lights"
            nc = draft_dir / "non_calibrated" / "lights"
            if cal.is_dir():
                report["calibrated_setups"] = sorted(d.name for d in cal.iterdir() if d.is_dir())
            elif nc.is_dir():
                report["import_setups"] = sorted(d.name for d in nc.iterdir() if d.is_dir())
            ps = draft_dir / "platesolve"
            if ps.is_dir():
                report["platesolve_setups"] = sorted(
                    d.name for d in ps.iterdir() if d.is_dir() and (d / "MASTERSTAR.fits").is_file()
                )
            report["post_analysis"] = _analyze_draft(
                draft_dir, cfg=cfg, collect_ms_stats=pal._collect_masterstar_stats
            )

            try:
                from tests.photometry_sha import compute_photometry_sha  # noqa: PLC0415

                core_sha, core_n = compute_photometry_sha(draft_dir)
                full_sha, full_n = compute_photometry_sha(draft_dir, include_comp_qa=True)
                report["photometry_sha"] = {
                    "core": core_sha,
                    "core_n": core_n,
                    "core_prefix": core_sha[:8],
                    "full": full_sha,
                    "full_n": full_n,
                    "full_prefix": full_sha[:8],
                    "provisional_dy_peg_anchor": True,
                    "note": "Captured only; not locked until byte-identical repeat run.",
                }
            except Exception as exc:  # noqa: BLE001
                report["photometry_sha_error"] = str(exc)

        # Acceptance summary
        pa = report.get("post_analysis") or {}
        cg = pa.get("completeness_gate") or {}
        comp_ok = all(
            cg.get(s, {}).get("ok") for s in EXPECTED_SETUPS if s in cg
        ) and len([s for s in EXPECTED_SETUPS if s in cg]) == 3
        dy = pa.get("dy_peg", {}).get("best_match") or {}
        dy_ok = bool(dy.get("lc_file")) and float(dy.get("sep_arcsec", 999)) < 30.0
        p2p = float(dy.get("peak_to_peak_mag", float("nan")))
        pulsation_ok = math.isfinite(p2p) and p2p >= 0.15
        pdf_ok = int(pa.get("pdf_overflow_total", -1)) == 0
        report["acceptance"] = {
            "completeness_gate_all_setups": comp_ok,
            "dy_peg_lc_present": dy_ok,
            "dy_peg_pulsation_visible": pulsation_ok,
            "pdf_overflow_zero": pdf_ok,
            "first_run_pass": bool(
                report.get("night_run_success")
                and comp_ok
                and dy_ok
                and pulsation_ok
                and pdf_ok
            ),
        }
    finally:
        _restore_config(orig_cfg)
        report["config_restored"] = True

    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    RESULT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report.get("night_run_success") and report.get("acceptance", {}).get("first_run_pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
