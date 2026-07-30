#!/usr/bin/env python3
"""Qatar-8 V-band night_run — Dablice Newton bin1, pre-calibrated, transit validation."""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
RESULT_PATH = _ROOT / "tmp" / "qatar8_night_run_result.json"
OUT_PNG_DIR = _ROOT / "tmp" / "qatar8_host"
SOURCE_ROOT = _ROOT / "Archive" / "QATAR-8"

# Qatar-8 host (NEA / catalog)
HOST_GAIA = "1076515002779544960"
HOST_RA = 157.41232123600938
HOST_DEC = 70.52708217790601
HOST_V_MAG = 11.71
CATALOG_RA = 157.413
CATALOG_DEC = 70.527

PLATE_SCALE_BIN1 = round(206.265 * 3.76 / 1200.0, 4)  # ~0.646″/px C3-26000 @ 1200mm
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


def _sep_arcsec(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    r = math.radians
    dra = (ra2 - ra1) * math.cos(r((dec1 + dec2) / 2.0))
    ddec = dec2 - dec1
    return math.degrees(math.sqrt(dra * dra + ddec * ddec)) * 3600.0


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
    data["phase01_plate_scale_arcsec_per_px"] = PLATE_SCALE_BIN1
    CONFIG_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return orig


def _restore_config(orig: dict) -> None:
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    for key in orig:
        data[key] = orig[key]
    CONFIG_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _config_snapshot(cfg: object) -> dict[str, object]:
    return {
        "gaia_db_path": str(getattr(cfg, "gaia_db_path", "") or ""),
        "skip_processed_directory": bool(getattr(cfg, "skip_processed_directory", False)),
        "psf_photometry_enabled": bool(getattr(cfg, "psf_photometry_enabled", False)),
        "sysrem_enabled": bool(getattr(cfg, "sysrem_enabled", False)),
        "savgol_detrend_enabled": bool(getattr(cfg, "savgol_detrend_enabled", False)),
        "phase01_plate_scale_arcsec_per_px": float(
            getattr(cfg, "phase01_plate_scale_arcsec_per_px", float("nan"))
        ),
    }


def _rig_plate_scale_from_db(camera_id: int, telescope_id: int) -> dict:
    import sqlite3  # noqa: PLC0415

    conn = sqlite3.connect(_ROOT / "vyvar.sqlite3")
    cam = conn.execute("SELECT CAMERANAME, PIXELSIZE FROM EQUIPMENTS WHERE ID=?", (camera_id,)).fetchone()
    tel = conn.execute("SELECT TELESCOPENAME, FOCAL FROM TELESCOPE WHERE ID=?", (telescope_id,)).fetchone()
    conn.close()
    pix_um = float(cam[1]) if cam else float("nan")
    focal_mm = float(tel[1]) if tel else float("nan")
    scale = 206.265 * pix_um / focal_mm if math.isfinite(pix_um) and focal_mm > 0 else float("nan")
    return {
        "camera": cam[0] if cam else None,
        "telescope": tel[0] if tel else None,
        "pixel_um": pix_um,
        "focal_mm": focal_mm,
        "plate_scale_bin1_arcsec_px": scale,
    }


def _fits_provenance(sample: Path) -> dict:
    from astropy.io import fits  # noqa: PLC0415

    h = fits.getheader(sample, ignore_missing_end=True)
    cdel = abs(float(h.get("CDELT1", 0) or 0)) * 3600.0 if h.get("CDELT1") else None
    sec = float(h.get("SECPIX1", float("nan")))
    return {
        "file": sample.name,
        "xbinning": int(h.get("XBINNING", 0) or 0),
        "ybinning": int(h.get("YBINNING", 0) or 0),
        "secpix1_arcsec_px": sec,
        "cdelt1_arcsec_px": cdel,
        "exptime_s": float(h.get("EXPTIME", float("nan"))),
        "filter": str(h.get("FILTER", "")),
        "crval1": float(h.get("CRVAL1", float("nan"))),
        "crval2": float(h.get("CRVAL2", float("nan"))),
        "field_ra_deg": float(h.get("CRVAL1", float("nan"))),
        "field_dec_deg": float(h.get("CRVAL2", float("nan"))),
    }


def _gaia_coverage(ra: float, dec: float, db_path: Path) -> dict:
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
    host = con.execute(
        "SELECT source_id, ra, dec, g_mag, bp_rp FROM gaia_dr3 WHERE source_id=?",
        (HOST_GAIA,),
    ).fetchone()
    con.close()
    out["total_stars_db"] = int(total)
    out["cone_0p5deg_count"] = int(n[0])
    out["host_lookup"] = {
        "source_id": str(host[0]) if host else None,
        "ra": float(host[1]) if host else None,
        "dec": float(host[2]) if host else None,
        "g_mag": float(host[3]) if host else None,
        "bp_rp": float(host[4]) if host else None,
    }
    return out


def _identity_precheck(sample_fits: Path) -> dict:
    prov = _fits_provenance(sample_fits)
    field_ra, field_dec = prov["field_ra_deg"], prov["field_dec_deg"]
    host_sep_field = _sep_arcsec(HOST_RA, HOST_DEC, field_ra, field_dec)
    cat_sep_host = _sep_arcsec(CATALOG_RA, CATALOG_DEC, HOST_RA, HOST_DEC)
    return {
        "fits_embedded_gaia_id": None,
        "coordinate_resolved_host_gaia": HOST_GAIA,
        "catalog_ra_dec": [CATALOG_RA, CATALOG_DEC],
        "gaia_host_ra_dec": [HOST_RA, HOST_DEC],
        "field_center_ra_dec": [field_ra, field_dec],
        "sep_catalog_to_gaia_host_arcsec": round(cat_sep_host, 2),
        "sep_gaia_host_to_field_center_arcsec": round(host_sep_field, 2),
        "verdict": (
            "PASS — no FITS-embedded Gaia ID; host Gaia matches catalog coords (≤1″); "
            f"host is {host_sep_field/60:.1f}′ from field center (in FOV)"
            if cat_sep_host < 2.0 and host_sep_field < 3600
            else "REVIEW"
        ),
        "provenance": prov,
    }


def _patch_blind_platesolve_hint() -> None:
    """Pre-cal NINA WCS vs VYVAR solver center can differ by ~0.3–0.4° — use relaxed hint guard."""
    import pipeline as _pipeline  # noqa: PLC0415
    import vyvar_platesolver as vps  # noqa: PLC0415

    def _hint(header):
        try:
            cr1 = header.get("CRVAL1")
            cr2 = header.get("CRVAL2")
            if cr1 is not None and cr2 is not None:
                return float(cr1), float(cr2), "blind solver"
        except (TypeError, ValueError):
            pass
        return None, None, "blind solver"

    vps.pointing_hint_from_header = _hint
    _pipeline._pointing_hint_from_header = _hint


def _median_field_center(fits_list: list[Path]) -> tuple[float, float]:
    from astropy.io import fits  # noqa: PLC0415

    ras, decs = [], []
    for f in fits_list[:: max(1, len(fits_list) // 20)]:
        h = fits.getheader(f, ignore_missing_end=True)
        if h.get("CRVAL1") is not None and h.get("CRVAL2") is not None:
            ras.append(float(h["CRVAL1"]))
            decs.append(float(h["CRVAL2"]))
    return float(np.median(ras)), float(np.median(decs))


def _patch_preprocess_field_center(field_ra: float, field_dec: float) -> None:
    """Inject catalog-host coordinates for MASTERSTAR hint (solver lands ~0.25° from host)."""
    import pipeline as _pipeline  # noqa: PLC0415

    hint_ra, hint_dec = float(HOST_RA), float(HOST_DEC)

    def _resolve(*, db, draft_id, ui_ra_deg, ui_dec_deg):
        del db, draft_id, ui_ra_deg, ui_dec_deg
        return hint_ra, hint_dec

    _pipeline.resolve_preprocess_target_coordinates = _resolve
    try:
        import night_run as _nr  # noqa: PLC0415

        _nr.resolve_preprocess_target_coordinates = _resolve
    except Exception:  # noqa: BLE001
        pass


def _estimate_dao_fwhm(sample_fits: Path) -> float:
    try:
        from astropy.io import fits  # noqa: PLC0415
        from pipeline import _estimate_dao_fwhm_guess  # noqa: PLC0415

        with fits.open(sample_fits, memmap=True) as hdul:
            data = np.asarray(hdul[0].data, dtype=np.float64)
        med = float(np.nanmedian(data))
        std = float(np.nanstd(data))
        guess = float(_estimate_dao_fwhm_guess(data - med, std))
        fallback = CHIANDH_BIN2_DAO_FWHM_PX * 2.0
        if not math.isfinite(guess) or guess <= 0:
            return fallback
        return max(3.0, min(12.0, guess))
    except Exception:  # noqa: BLE001
        return CHIANDH_BIN2_DAO_FWHM_PX * 2.0


def _ensure_qatar8_host(_draft_id: int, draft_dir: Path, _cfg, _pipeline) -> None:
    """Inject coordinate-resolved Qatar-8 host if VSX export omitted it."""
    ps_root = draft_dir / "platesolve"
    if not ps_root.is_dir():
        return
    for setup_dir in ps_root.iterdir():
        if not setup_dir.is_dir():
            continue
        vt_path = setup_dir / "variable_targets.csv"
        ms_path = setup_dir / "masterstars_full_match.csv"
        if not ms_path.is_file():
            continue
        ms = pd.read_csv(ms_path, dtype={"catalog_id": str, "name": str})
        host_ms = ms[ms["catalog_id"].astype(str) == HOST_GAIA]
        if host_ms.empty:
            ms["_sep"] = ms.apply(
                lambda r: _sep_arcsec(
                    HOST_RA,
                    HOST_DEC,
                    float(r.get("ra_deg", r.get("ra", float("nan")))),
                    float(r.get("dec_deg", r.get("dec", float("nan")))),
                )
                if pd.notna(r.get("ra_deg", r.get("ra")))
                else 9999.0,
                axis=1,
            )
            host_ms = ms.nsmallest(1, "_sep")
        row = host_ms.iloc[0]
        cid = str(row.get("catalog_id", HOST_GAIA))
        vt = pd.read_csv(vt_path, dtype=str) if vt_path.is_file() else pd.DataFrame()
        if not vt.empty and (vt["catalog_id"].astype(str) == cid).any():
            continue
        new_row = {
            "name": f"Gaia DR3 {cid}",
            "catalog_id": cid,
            "catalog": "GAIA_DR3",
            "ra_deg": str(row.get("ra_deg", row.get("ra", HOST_RA))),
            "dec_deg": str(row.get("dec_deg", row.get("dec", HOST_DEC))),
            "priority": "1",
            "notes": "Qatar-8 host (programme inject)",
            "vsx_name": "Qatar-8",
            "vsx_type": "PL",
            "x": str(row.get("x", "")),
            "y": str(row.get("y", "")),
            "mag": str(row.get("mag", row.get("mag_inst", HOST_V_MAG))),
            "zone": str(row.get("zone", "linear")),
        }
        vt = pd.concat([vt, pd.DataFrame([new_row])], ignore_index=True)
        vt.to_csv(vt_path, index=False)
        print(f"[QATAR-8] injected host {cid} into {vt_path}", flush=True)


def _nea_qatar8_ephem() -> dict:
    queries = [
        "select pl_name,hostname,ra,dec,pl_orbper,pl_tranmid,pl_trandep,pl_trandurh "
        "from pscomppars where pl_name like 'Qatar-8%'",
        "select pl_name,hostname,ra,dec,pl_orbper,pl_tranmid,pl_trandep,pl_trandurh "
        "from ps where pl_name like 'Qatar-8%'",
    ]
    last_err = ""
    for q in queries:
        url = (
            "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="
            + urllib.parse.quote(q)
            + "&format=csv"
        )
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            df = pd.read_csv(StringIO(raw))
            out: dict = {"url": url, "rows": df.to_dict(orient="records")}
            if df.empty:
                continue
            sub = df[df["pl_name"].astype(str).str.contains("Qatar-8 b", case=False, na=False)]
            r = sub.iloc[0] if len(sub) else df.iloc[0]
            per = float(r.get("pl_orbper"))
            t0 = float(r.get("pl_tranmid"))
            out["ephemeris"] = {
                "pl_name": str(r.get("pl_name")),
                "period_d": per,
                "t0_bjd": t0,
                "depth_ppm": float(r.get("pl_trandep")) if pd.notna(r.get("pl_trandep")) else None,
                "duration_h": float(r.get("pl_trandurh")) if pd.notna(r.get("pl_trandurh")) else None,
            }
            return out
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)
    return {"error": last_err, "fallback": {"period_d": 3.719, "depth_ppm": 10000, "duration_h": 3.5}}


def _transit_windows(t0: float, per: float, dur_h: float, t_start: float, t_end: float) -> list[tuple[float, float]]:
    dur_d = dur_h / 24.0
    k = round((0.5 * (t_start + t_end) - t0) / per)
    wins: list[tuple[float, float]] = []
    for dk in range(-5, 6):
        tc = t0 + (k + dk) * per
        half = dur_d / 2.0
        if tc + half >= t_start and tc - half <= t_end:
            wins.append((tc - half, tc + half))
    return wins


def _analyze_host(draft_dir: Path, *, nea: dict) -> dict:
    ps_root = draft_dir / "platesolve"
    setups = sorted(d.name for d in ps_root.iterdir() if d.is_dir()) if ps_root.is_dir() else []
    if not setups:
        return {"error": "no platesolve setups"}
    setup = setups[0]
    phot = ps_root / setup / "photometry"
    lc_dir = phot / "lightcurves"
    sm_path = phot / "photometry_summary.csv"
    comp_path = phot / "comparison_stars_per_target.csv"

    sm = pd.read_csv(sm_path, dtype={"catalog_id": str})
    host = sm[sm["catalog_id"].astype(str) == HOST_GAIA]
    if host.empty:
        sm["_sep"] = sm.apply(
            lambda r: _sep_arcsec(
                HOST_RA,
                HOST_DEC,
                float(r.get("ra_deg", float("nan"))),
                float(r.get("dec_deg", float("nan"))),
            )
            if "ra_deg" in sm.columns
            else 9999.0,
            axis=1,
        )
        host = sm.nsmallest(1, "_sep")
    row = host.iloc[0]
    cid = str(row["catalog_id"])
    lc_path = lc_dir / f"lightcurve_{cid}.csv"
    lc = pd.read_csv(lc_path)
    bjd = pd.to_numeric(lc["bjd"], errors="coerce").dropna()
    mag_col = "mag_calib_ct" if "mag_calib_ct" in lc.columns else "mag"
    mag = pd.to_numeric(lc[mag_col], errors="coerce").dropna()
    err_col = "err" if "err" in lc.columns else "merr"
    merr = pd.to_numeric(lc.get(err_col), errors="coerce").dropna()

    t_start, t_end = float(bjd.min()), float(bjd.max())
    baseline_h = (t_end - t_start) * 24.0
    lc_rms_mmag = float(row.get("lc_rms", np.nanstd(mag))) * 1000
    p2p_mmag = (float(mag.max()) - float(mag.min())) * 1000

    windows: list[tuple[float, float]] = []
    eph = nea.get("ephemeris") or {}
    if eph.get("t0_bjd") and eph.get("period_d"):
        windows = _transit_windows(
            float(eph["t0_bjd"]),
            float(eph["period_d"]),
            float(eph.get("duration_h") or 3.5),
            t_start,
            t_end,
        )

    # Simple transit depth: median outside windows vs min inside
    depth_mmag = float("nan")
    if windows:
        out_mask = np.ones(len(bjd), dtype=bool)
        bjd_a = bjd.to_numpy()
        mag_a = mag.to_numpy()
        for t0, t1 in windows:
            out_mask &= (bjd_a < t0) | (bjd_a > t1)
        if out_mask.sum() >= 5:
            med_out = float(np.median(mag_a[out_mask]))
            in_mask = ~out_mask
            if in_mask.any():
                depth_mmag = (med_out - float(np.min(mag_a[in_mask]))) * 1000.0

    OUT_PNG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=150)
    if len(merr) == len(mag) and np.any(merr > 0):
        ax.errorbar(bjd, mag, yerr=merr, fmt="o", ms=2.5, capsize=1, color="#1f4e79", ecolor="#7faad4", alpha=0.85)
    else:
        ax.plot(bjd, mag, "o", ms=2.5, color="#1f4e79", alpha=0.85)
    for i, (t0, t1) in enumerate(windows):
        ax.axvspan(t0, t1, color="#c0392b", alpha=0.12, label="predicted transit" if i == 0 else "")
    ax.set_xlabel("BJD")
    ax.set_ylabel("Calibrated mag")
    ax.set_title(
        f"Qatar-8 host Gaia …{cid[-8:]}\n"
        f"{len(mag)} frames, {baseline_h:.2f} h | lc_rms={lc_rms_mmag:.1f} mmag | "
        f"meas depth≈{depth_mmag:.1f} mmag | trust={row.get('trust','')}"
    )
    ax.invert_yaxis()
    ax.grid(True, alpha=0.25)
    if windows:
        ax.legend(loc="best", fontsize=8)
    lc_png = OUT_PNG_DIR / "qatar8_host_lc.png"
    fig.savefig(lc_png, bbox_inches="tight")
    plt.close(fig)

    # PDF page extract
    summary_png = None
    pdfs = sorted((ps_root / setup).glob("VYVAR_report*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
    if pdfs:
        try:
            import fitz  # noqa: PLC0415

            doc = fitz.open(pdfs[0])
            for i in range(len(doc)):
                if cid in doc[i].get_text() and "LC" in doc[i].get_text():
                    pix = doc[i].get_pixmap(matrix=fitz.Matrix(2, 2))
                    summary_png = OUT_PNG_DIR / "qatar8_host_summary_page.png"
                    pix.save(str(summary_png))
                    break
            doc.close()
        except Exception:  # noqa: BLE001
            pass

    n_comp = 0
    if comp_path.is_file():
        comp = pd.read_csv(comp_path, dtype={"target_catalog_id": str, "catalog_id": str})
        n_comp = int(len(comp[comp["target_catalog_id"] == cid]))

    return {
        "setup": setup,
        "catalog_id_used": cid,
        "lc_rms_mmag": round(lc_rms_mmag, 2),
        "p2p_mmag": round(p2p_mmag, 2),
        "n_frames": int(len(bjd)),
        "baseline_h": round(baseline_h, 3),
        "merr_median_mmag": round(float(merr.median()) * 1000, 2) if len(merr) else None,
        "trust": str(row.get("trust", "")),
        "trust_reason": str(row.get("trust_reason", "")),
        "n_comp_selected": n_comp,
        "n_good_comp": int(row.get("n_good_comp", 0) or 0),
        "aperture_px": float(row.get("aperture_px", float("nan"))),
        "measured_transit_depth_mmag": round(depth_mmag, 2) if math.isfinite(depth_mmag) else None,
        "expected_depth_mmag_range": [10, 14],
        "transit_windows_bjd": windows,
        "png_lc": str(lc_png),
        "png_summary_page": str(summary_png) if summary_png else None,
        "pdf_source": str(pdfs[0]) if pdfs else None,
    }


def _parse_stability_logs(draft_dir: Path) -> list[str]:
    hits: list[str] = []
    for p in draft_dir.rglob("*.log"):
        try:
            txt = p.read_text(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            continue
        for line in txt.splitlines():
            if "STABILITY" in line or "Common-mode" in line or "[STABILITY]" in line:
                hits.append(line.strip())
    # also grep stdout captured in result json phase - search photometry dir text files
    for p in draft_dir.rglob("*"):
        if p.suffix.lower() in (".txt", ".json") and "stability" in p.name.lower():
            try:
                hits.extend(p.read_text(encoding="utf-8", errors="replace").splitlines()[:20])
            except Exception:  # noqa: BLE001
                pass
    return hits[:30]


def _masterstar_stats(draft_dir: Path) -> dict:
    sys.path.insert(0, str(_ROOT / "scripts"))
    import pilot_palomar7_phases_ac as pal  # noqa: PLC0415

    return pal._collect_masterstar_stats(draft_dir)


def _pdf_overflow(draft_dir: Path) -> dict:
    from photometry_report import generate_photometry_report  # noqa: PLC0415

    out: dict = {}
    ps = draft_dir / "platesolve"
    for setup_dir in ps.iterdir() if ps.is_dir() else []:
        pdf = setup_dir / f"VYVAR_report_{setup_dir.name}_overflow_verify.pdf"
        try:
            generate_photometry_report(
                draft_dir,
                setup_dir.name,
                pdf,
                report_draft_label=draft_dir.name,
                verify_overflow=True,
            )
            from photometry_report import generate_photometry_report as gpr  # noqa: PLC0415

            n = int(getattr(gpr, "last_overflow_violations", 0) or 0)
        except Exception as exc:  # noqa: BLE001
            n = -1
            out[setup_dir.name] = {"overflow": n, "error": str(exc)}
            continue
        out[setup_dir.name] = {"overflow": n}
    return out


def main() -> int:
    from config import AppConfig  # noqa: E402
    from database import get_gaia_db_max_g_mag  # noqa: E402

    sys.path.insert(0, str(_ROOT / "scripts"))
    import chiandh_phases_ac as equip  # noqa: E402

    fits = sorted(SOURCE_ROOT.glob("*.fits"))
    if len(fits) != 125:
        print(f"Warning: expected 125 FITS, found {len(fits)}")

    sample = fits[0]
    field_ra, field_dec = _median_field_center(fits)
    _patch_blind_platesolve_hint()
    _patch_preprocess_field_center(field_ra, field_dec)

    from night_run import NightRunParams, run_night_pipeline  # noqa: E402

    git_commit = _git_rev_parse_head()
    ids = equip.phase_a_register()
    rig = _rig_plate_scale_from_db(int(ids["camera_id"]), int(ids["telescope_id"]))
    identity = _identity_precheck(sample)
    identity["field_center_median_crval"] = [field_ra, field_dec]
    dao_fwhm = _estimate_dao_fwhm(sample)
    orig_cfg = _patch_config(skip_processed=True, psf_enabled=False)
    cfg_pre = AppConfig()
    gaia_path = Path(str(cfg_pre.gaia_db_path))
    gaia_cov = _gaia_coverage(HOST_RA, HOST_DEC, gaia_path)
    nea = _nea_qatar8_ephem()

    report: dict = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "source_dir": str(SOURCE_ROOT),
        "n_source_fits": len(fits),
        "identity_precheck": identity,
        "rig_plate_scale": rig,
        "fits_provenance_bin1": identity["provenance"],
        "gaia_coverage": gaia_cov,
        "catalog_source": "zaloha G<=16 (vyvar_gaia_dr3.db)" if "zaloha" in str(gaia_path) else str(gaia_path),
        "nea_ephemeris": nea,
        "dao_fwhm_px_used": dao_fwhm,
        "pre_calibrated_mode": True,
        "psf_sysrem_savgol_off": True,
    }

    try:
        cfg = AppConfig()
        report["config_snapshot"] = _config_snapshot(cfg)
        report["gaia_max_g"] = get_gaia_db_max_g_mag(cfg.gaia_db_path)

        if int(gaia_cov.get("cone_0p5deg_count") or 0) < 50:
            report["error"] = "Insufficient Gaia coverage"
            RESULT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
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
            dao_fwhm_px=dao_fwhm,
            dao_threshold_sigma=3.5,
            catalog_match_max_sep_arcsec=3.0,
            max_catalog_rows=25000,
            min_detected_stars=200,
            max_detected_stars=6000,
            max_control_points=250,
            post_platesolve_hook=_ensure_qatar8_host,
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

        if nr.draft_dir and nr.success:
            draft_dir = Path(nr.draft_dir)
            report["masterstar_stats"] = _masterstar_stats(draft_dir)
            report["host_analysis"] = _analyze_host(draft_dir, nea=nea)
            report["pdf_overflow"] = _pdf_overflow(draft_dir)
            report["stability_log_hits"] = _parse_stability_logs(draft_dir)
            manifest = draft_dir / "draft_manifest.json"
            if manifest.is_file():
                report["draft_manifest"] = json.loads(manifest.read_text(encoding="utf-8"))
            prov_path = draft_dir / "draft_provenance.json"
            if prov_path.is_file():
                report["draft_provenance"] = json.loads(prov_path.read_text(encoding="utf-8"))
        elif nr.draft_dir:
            report["partial_draft_dir"] = str(nr.draft_dir)
    finally:
        _restore_config(orig_cfg)

    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report.get(k) for k in ("night_run_success", "draft_id", "host_analysis")}, indent=2))
    return 0 if report.get("night_run_success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
