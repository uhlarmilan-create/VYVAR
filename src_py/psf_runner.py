from __future__ import annotations

import argparse
import json
import math
import os
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits

from gaia_catalog_id import normalize_gaia_source_id

# Gaia ID musi byt str - float64 straca cifry
_GAIA_ID_DTYPE: dict[str, type] = {"catalog_id": str, "name": str}

# -- KONFIGURACIA -- upravuj podla potreby --------------------------
DRAFT_ID = 248
ARCHIVE_ROOT = Path(r"C:\ASTRO\python\VYVAR\Archive")
OBS_GROUP = "NoFilter_60_2"
USE_SANDBOX = True
OVERSAMPLING = 2
MIN_STARS = 15
DRY_RUN = False  # True = len vypise co by spravil, nezapise CSV
# QHY294PROM, Read Mode 0, Gain setting 0 -> 3.17 e-/ADU
# Zdroj: DB EQUIPMENTS.GAIN_ADU (get_equipment_cosmic_params)
# Fallback ak DB vrati None alebo 0:
GAIN_FALLBACK_E_PER_ADU = 3.17
BORDER_WIDTH = 3  # pixely okraja pre lokalny odhad oblohy v cutoute
SATURATE_PEAK_FRAC = 0.85  # Skip COMP PSF if peak_max_adu > frac * saturate_limit_adu
SUMMARY_MIN_MED_PSF_FLUX = 100.0  # Drop numerical dust in psf_summary.csv
SUMMARY_MIN_N_FIT_OK = 5  # Require enough good frames per star in summary (pouziva sa s n_fit_ok_report)
MAX_COMP_PEAK_ADU = 40_000  # vyrad prilis jasne COMP
# Engine: ``psf_fit_ok`` = konvergencia and (chi^2 < PSF_VAR_CHI2_MAX pre VAR, chi^2 < PSF_COMP_CHI2_MAX pre COMP).
# Reporting v step_4 pouziva rovnake prahy pri ``n_fit_ok_report``.
PSF_VAR_CHI2_MAX = 1000.0  # VAR - svetelna krivka: akceptuj vyssie chi^2 (error mapa casto podhodnotena pri jasnych)
PSF_COMP_CHI2_MAX = 20.0  # COMP - prisnejsi prah
MIN_COMP_SEPARATION_PX = 3 * 3.2  # ~10 px = 3x FWHM (crowding filter)
# Krok 5: styri COMP hviezdy pre frame-wise ZP z psf_summary (med_psf_flux).
COMP_CALIB_CATALOG_IDS: tuple[str, ...] = (
    "1496835173974894848",
    "1499921468754586112",
    "1498311264039176192",
    "1497439905370466816",
)
PSF_CAL_MAG_ZP_OFFSET = 20.0  # arbitrarny offset pre relativnu krivku (mag)
# ------------------------------------------------------------------

# Cielene VAR hviezdy pre tento test (ak neprazdne, nahradi variable_targets.csv v step_2_load_targets)
PSF_TARGET_OVERRIDE: list[dict[str, Any]] = [
    {
        "catalog_id": 1498486880958321200,
        "vsx_name": "CSS_J140918.7+423422",
        "vsx_type": "EW",
        "x": 265.155,
        "y": 177.707,
        "note": "Slaba EW mag=12.75 - hlavny PSF test",
    },
    {
        "catalog_id": 1497418258735289300,
        "vsx_name": "FU CVn",
        "vsx_type": "EW",
        "x": 1444.436,
        "y": 641.743,
        "note": "EW mag=11.0 - porovnanie s draft 247",
    },
]

# Jedine povolene importy z projektu:
from astropy.table import Table  # noqa: E402
from photutils.psf import ImagePSF, PSFPhotometry  # noqa: E402

from psf_photometry import build_epsf_model  # noqa: E402
from database import VyvarDatabase  # noqa: E402
from config import AppConfig  # noqa: E402
import logging
from plain_stats import plain_mean_med_std


def _force_utf8_stdout() -> None:
    try:
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    except Exception:  # noqa: BLE001
        pass
    try:
        import sys

        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass


def subtract_local_sky(cutout: np.ndarray, border_width: int = 3) -> tuple[np.ndarray, float, float]:
    """Sigma-clipped median sky from border pixels; subtract from cutout.

    Returns (sky_subtracted_cutout, sky_median_adu, sky_rms_adu).
    """
    cut = np.asarray(cutout, dtype=np.float64)
    h, w = int(cut.shape[0]), int(cut.shape[1])
    bw = max(1, int(border_width))
    if h <= 2 * bw + 1 or w <= 2 * bw + 1:
        sky = float(np.nanmedian(cut))
        if not math.isfinite(sky):
            sky = 0.0
        rms = float(np.nanstd(cut - sky))
        if not math.isfinite(rms) or rms <= 0:
            rms = 1.0
        return cut - sky, sky, rms

    border = np.zeros((h, w), dtype=bool)
    border[:bw, :] = True
    border[-bw:, :] = True
    border[:, :bw] = True
    border[:, -bw:] = True
    vals = cut[border]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        sky = float(np.nanmedian(cut))
        if not math.isfinite(sky):
            sky = 0.0
        rms = float(np.nanstd(cut - sky))
        if not math.isfinite(rms) or rms <= 0:
            rms = 1.0
        return cut - sky, sky, rms

    try:

        _, med, std = plain_mean_med_std(vals, sigma=3.0, maxiters=5, cenfunc="median", stdfunc="std")
        sky_val = float(med)
        sky_rms = float(std)
    except Exception:  # noqa: BLE001
        sky_val = float(np.median(vals))
        sky_rms = float(np.std(vals))

    if not math.isfinite(sky_val):
        sky_val = 0.0
    if not math.isfinite(sky_rms) or sky_rms <= 0:
        sky_rms = max(1e-6, float(np.std(vals)))
    sub = cut - sky_val
    return sub, sky_val, sky_rms


def _header_egain_only(hdr: Any) -> float | None:
    """Pozitivny e-/ADU len z ``EGAIN`` (``GAIN`` z INDI casto 0 - neberieme)."""
    if "EGAIN" not in hdr:
        return None
    try:
        g = float(hdr["EGAIN"])
        if math.isfinite(g) and g > 0:
            return g
    except (TypeError, ValueError):
        pass
    return None


def get_gain_from_header(hdr: Any, *, db_gain_e_per_adu: float | None) -> tuple[float, str]:
    """Priorita: DB ``GAIN_ADU`` -> FITS ``EGAIN`` -> ``GAIN_FALLBACK_E_PER_ADU``."""
    if db_gain_e_per_adu is not None and math.isfinite(float(db_gain_e_per_adu)) and float(db_gain_e_per_adu) > 0:
        return float(db_gain_e_per_adu), "DB"
    eg = _header_egain_only(hdr)
    if eg is not None:
        return float(eg), "FITS EGAIN"
    return float(GAIN_FALLBACK_E_PER_ADU), "fallback"


def _vy_qcrms_from_header(hdr: Any) -> float | None:
    """Kladne ``VY_QCRMS`` z hlavicky (ADU), alebo None."""
    if "VY_QCRMS" not in hdr:
        return None
    try:
        v = float(hdr["VY_QCRMS"])
    except (TypeError, ValueError):
        return None
    if math.isfinite(v) and v > 0:
        return v
    return None


def _vy_fwhm_from_header(hdr: Any) -> float:
    """``VY_FWHM`` [px]; NaN ak chyba alebo neplatne."""
    if "VY_FWHM" not in hdr:
        return float("nan")
    try:
        v = float(hdr["VY_FWHM"])
        return v if math.isfinite(v) and v > 0 else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _draft_equipment_id(db: VyvarDatabase, draft_id: int) -> int | None:
    """Manifest-first equipment id for this draft, if set and positive."""
    eid = db.get_draft_equipment_id(int(draft_id))
    if eid is None:
        return None
    try:
        val = int(eid)
    except (TypeError, ValueError):
        return None
    return val if val > 0 else None


def _draft_db_gain_e_per_adu(db: VyvarDatabase | None, draft_id: int) -> float | None:
    """``EQUIPMENTS.GAIN_ADU`` pre ``draft manifest.ID_EQUIPMENTS``, ak je kladne."""
    if db is None:
        return None
    eq_id = _draft_equipment_id(db, draft_id)
    if eq_id is None:
        return None
    g_db, _ = db.get_equipment_cosmic_params(int(eq_id))
    if g_db is not None and math.isfinite(float(g_db)) and float(g_db) > 0:
        return float(g_db)
    return None


def _runner_database_path() -> Path:
    cfg = AppConfig()
    db_path = Path(getattr(cfg, "database_path", "") or "").expanduser()
    if not str(db_path):
        db_path = ARCHIVE_ROOT / "vyvar.db"
    if not db_path.is_file():
        alt = ARCHIVE_ROOT / "vyvar.db"
        if alt.is_file():
            db_path = alt
    return db_path


def _fit_shape_for_cutout_legacy_even_down(cutout_size: int) -> tuple[int, int]:
    """Odd fit window: cutout-4, even values decremented. Differs from psf_photometry (FWHM path or even-up)."""
    fs = max(3, int(cutout_size) - 4)
    if fs % 2 == 0:
        fs -= 1
    fs = max(3, fs)
    return (fs, fs)


def _load_psf_photometry_bundle(epsf_path: Path) -> tuple[PSFPhotometry, int]:
    """Load ePSF FITS + meta; return (PSFPhotometry instance, cutout_size)."""
    ep = Path(epsf_path)
    if not ep.is_file():
        raise FileNotFoundError(f"EPSF FITS not found: {ep}")
    meta_fp = ep.parent / "masterstar_epsf_meta.json"
    if not meta_fp.is_file():
        raise FileNotFoundError(f"Missing meta JSON: {meta_fp}")
    meta = json.loads(meta_fp.read_text(encoding="utf-8"))
    cutout_size = int(meta["cutout_size"])
    os_meta = meta.get("oversampling", 2)
    if isinstance(os_meta, list):
        osamp = int(os_meta[0]) if len(os_meta) else 2
    else:
        osamp = int(os_meta)
    if cutout_size % 2 == 0 or cutout_size < 3:
        raise ValueError(f"cutout_size must be odd and >= 3, got {cutout_size}")
    psf_data = np.asarray(fits.getdata(ep), dtype=np.float64)
    # Pozn.: sucet pixelov ePSF gridu (napr. ~4 pri oversampling=2) nie je priamo 'flux v ADU';
    # PSFPhotometry skaluje amplitudu voci datam. Pre ``psf_flux_norm`` by sa pouzil integral modelu z Photutils, nie len ``sum(psf_data)``.
    psf_model = ImagePSF(psf_data, oversampling=osamp)
    fit_shape = _fit_shape_for_cutout_legacy_even_down(cutout_size)
    phot = PSFPhotometry(psf_model, fit_shape=fit_shape, progress_bar=False)
    return phot, cutout_size


def _per_cutout_error_map(cut_sub: np.ndarray, gain: float, sky_rms: float) -> np.ndarray:
    """Per-pixel error (ADU) from Poisson (signal/gain) + local sky variance."""
    g = float(gain) if math.isfinite(float(gain)) and float(gain) > 0 else 1.0
    s = float(sky_rms) if math.isfinite(float(sky_rms)) and float(sky_rms) > 0 else 1.0
    signal = np.maximum(np.asarray(cut_sub, dtype=np.float64), 0.0)
    variance = signal / g + s**2
    return np.sqrt(np.maximum(variance, 1.0))


def _psf_stars_local_cutouts(
    frame_data_psf: np.ndarray,
    star_positions: pd.DataFrame,
    phot: PSFPhotometry,
    cutout_size: int,
    *,
    border_width: int,
    gain: float,
    vy_qcrms_adu: float | None = None,
) -> tuple[pd.DataFrame, float, float]:
    """Per-star cutout: local sky subtract + Poisson/sky error map; same output columns as psf_photometry_stars."""
    h, w = int(frame_data_psf.shape[0]), int(frame_data_psf.shape[1])
    half = int(cutout_size) // 2
    sky_vals: list[float] = []
    sky_rms_vals: list[float] = []

    _cols = [
        "catalog_id",
        "name",
        "x",
        "y",
        "psf_flux",
        "psf_flux_err",
        "psf_chi2",
        "psf_converged",
        "psf_fit_ok",
    ]
    if star_positions.empty:
        return pd.DataFrame(columns=_cols), float("nan"), float("nan")

    out_rows: list[dict[str, Any]] = []
    for _, row in star_positions.iterrows():
        cid = row["catalog_id"]
        name = row["name"]
        role_u = str(row.get("role", "")).strip().upper()
        chi2_cap = float(PSF_VAR_CHI2_MAX) if role_u == "VAR" else float(PSF_COMP_CHI2_MAX)
        try:
            x = float(row["x"])
            y = float(row["y"])
        except (TypeError, ValueError):
            out_rows.append(
                {
                    "catalog_id": cid,
                    "name": name,
                    "x": row["x"],
                    "y": row["y"],
                    "psf_flux": float("nan"),
                    "psf_flux_err": float("nan"),
                    "psf_chi2": float("nan"),
                    "psf_converged": False,
                    "psf_fit_ok": False,
                }
            )
            continue

        base = {
            "catalog_id": cid,
            "name": name,
            "x": x,
            "y": y,
            "psf_flux": float("nan"),
            "psf_flux_err": float("nan"),
            "psf_chi2": float("nan"),
            "psf_converged": False,
            "psf_fit_ok": False,
        }

        xi, yi = int(round(x)), int(round(y))
        if xi < half or yi < half or xi >= w - half or yi >= h - half:
            out_rows.append(base)
            continue

        x1 = xi - half
        y1 = yi - half
        x2 = x1 + cutout_size
        y2 = y1 + cutout_size

        try:
            cut = np.asarray(frame_data_psf[y1:y2, x1:x2], dtype=np.float64)
            if cut.shape != (cutout_size, cutout_size):
                out_rows.append(base)
                continue

            cut_sub, sky_val, sky_rms_border = subtract_local_sky(cut, border_width=int(border_width))
            sky_vals.append(sky_val)
            sky_rms_vals.append(sky_rms_border)

            use_hdr_rms = (
                vy_qcrms_adu is not None and math.isfinite(float(vy_qcrms_adu)) and float(vy_qcrms_adu) > 0
            )
            sky_rms_err = float(vy_qcrms_adu) if use_hdr_rms else float(sky_rms_border)

            xc = x - x1
            yc = y - y1
            flux_guess = float(np.nansum(cut_sub))
            if not math.isfinite(flux_guess) or flux_guess <= 0.0:
                flux_guess = float(np.nanmax(cut_sub)) * 0.5 * cutout_size * cutout_size
                if not math.isfinite(flux_guess) or flux_guess <= 0.0:
                    flux_guess = 1.0

            init = Table([[xc], [yc], [flux_guess]], names=("x_0", "y_0", "flux_0"))
            err_cut = _per_cutout_error_map(cut_sub, gain, sky_rms_err)
            res = phot(cut_sub, init_params=init, error=err_cut)

            flux_fit = float(res["flux_fit"][0])
            flux_err = float(res["flux_err"][0])
            chi2 = float(res["reduced_chi2"][0])
            flags = int(res["flags"][0])
            converged = (flags & 8) == 0
            chi2_ok = math.isfinite(chi2) and chi2 < float(chi2_cap)
            fit_ok = bool(converged and chi2_ok)

            out_rows.append(
                {
                    "catalog_id": cid,
                    "name": name,
                    "x": x,
                    "y": y,
                    "psf_flux": flux_fit,
                    "psf_flux_err": flux_err,
                    "psf_chi2": chi2,
                    "psf_converged": bool(converged),
                    "psf_fit_ok": fit_ok,
                }
            )
        except Exception:  # noqa: BLE001
            out_rows.append(base)

    sky_med = float(np.median(sky_vals)) if sky_vals else float("nan")
    sky_rms_med = float(np.median(sky_rms_vals)) if sky_rms_vals else float("nan")
    return pd.DataFrame(out_rows, columns=_cols), sky_med, sky_rms_med


def _print_chi2_distribution(chi: pd.Series) -> None:
    """Histogram text: vsetky riadky vysledku (vratane NaN po preskoceni fitu)."""
    c = pd.to_numeric(chi, errors="coerce")
    n_nan = int(c.isna().sum())
    finite = c.dropna()
    arr = finite.to_numpy(dtype=np.float64)
    mask0 = np.isfinite(arr) & np.isclose(arr, 0.0, rtol=0.0, atol=1e-12)
    n0 = int(mask0.sum())
    pos = arr[np.isfinite(arr) & ~np.isclose(arr, 0.0, rtol=0.0, atol=1e-12)]
    n_lt1 = int(np.sum((pos > 0) & (pos < 1)))
    n_13 = int(np.sum((pos >= 1) & (pos < 3)))
    n_35 = int(np.sum((pos >= 3) & (pos < 5)))
    n_520 = int(np.sum((pos >= 5) & (pos <= 20)))
    n_gt20 = int(np.sum(pos > 20))
    print("  chi2 distribucia (vsetky riadky vysledku):")
    print(f"    chi2 < 1:    {n_lt1}")
    print(f"    chi2 1-3:    {n_13}   <- idealna zona")
    print(f"    chi2 3-5:    {n_35}   <- akceptovatelne")
    print(f"    chi2 5-20:   {n_520}   <- problematicke")
    print(f"    chi2 > 20:   {n_gt20}   <- divergencia")
    print(f"    chi2 = 0:    {n0}   <- numericky artefakt")
    print(f"    chi2 = NaN:  {n_nan}")


def _comp_fail_reason_psf(
    x: float,
    y: float,
    fw: int,
    fh: int,
    half_cs: int,
    peak_adu: float,
    sat_lim_adu: float,
) -> str:
    """Kratky text pre diagnostiku zlyhaneho COMP fitu."""
    if not (math.isfinite(x) and math.isfinite(y)):
        return "neplatna pozicia"
    xi, yi = int(round(x)), int(round(y))
    parts: list[str] = []
    if xi < half_cs or yi < half_cs or xi >= fw - half_cs or yi >= fh - half_cs:
        parts.append("okraj cipu")
    if (
        math.isfinite(peak_adu)
        and math.isfinite(sat_lim_adu)
        and sat_lim_adu > 0
        and peak_adu > float(SATURATE_PEAK_FRAC) * sat_lim_adu
    ):
        parts.append("blizko saturacie")
    if not parts:
        parts.append("blend / PSF nezhoda / ine")
    return ", ".join(parts)


def _print_dry_run_two_frame_report(
    bundle: list[tuple[pd.DataFrame, pd.DataFrame]],
    comp_df: pd.DataFrame,
) -> None:
    """After --dry-run --frames 2: chi2 / fit_ok + dao vs PSF RMS%% for first 5 COMP."""
    if len(bundle) < 1:
        return
    all_res = pd.concat([b[0] for b in bundle], ignore_index=True)
    chi = pd.to_numeric(all_res["psf_chi2"], errors="coerce")
    ok = all_res["psf_fit_ok"].fillna(False).astype(bool)
    var_m = all_res["role"].astype(str).str.upper() == "VAR"
    comp_m = all_res["role"].astype(str).str.upper() == "COMP"
    n = int(len(all_res))
    n_ok = int(ok.sum())
    pct = 100.0 * float(n_ok) / float(n) if n else 0.0
    chi_var_ok = chi[var_m & ok & chi.notna()]
    chi_comp_ok = chi[comp_m & ok & chi.notna()]
    med_v_ok = float(chi_var_ok.median()) if len(chi_var_ok) else float("nan")
    med_c_ok = float(chi_comp_ok.median()) if len(chi_comp_ok) else float("nan")
    chi_var_all = chi[var_m & chi.notna()]
    chi_comp_all = chi[comp_m & chi.notna()]
    med_v_all = float(chi_var_all.median()) if len(chi_var_all) else float("nan")
    med_c_all = float(chi_comp_all.median()) if len(chi_comp_all) else float("nan")
    print("\n=== DRY-RUN report (2 framy) ===")
    print(f"  chi2 median VAR (fit_ok only): {med_v_ok:.3f}  |  (vsetky konecne): {med_v_all:.3f}")
    print(f"  chi2 median COMP (fit_ok only): {med_c_ok:.3f}  |  (vsetky konecne): {med_c_all:.3f}")
    print(f"  fit_ok: {n_ok}/{n} ({pct:.1f}%)")
    _print_chi2_distribution(chi)

    print(f"  prvych 5 COMP s >=2 platnymi psf+dao bodmi - dao_rms_pct vs psf_rms_pct (n={len(bundle)} framov):")
    shown = 0
    for raw_cid in comp_df["catalog_id"].tolist():
        cid = normalize_gaia_source_id(raw_cid)
        if not cid:
            continue
        psf_fluxes: list[float] = []
        dao_fluxes: list[float] = []
        for res_df, proc_df in bundle:
            res_df = res_df.copy()
            res_df["_cid"] = res_df["catalog_id"].map(normalize_gaia_source_id)
            sub = res_df[(res_df["_cid"] == cid) & (res_df["role"].astype(str).str.upper() == "COMP")]
            if sub.empty:
                continue
            pf = pd.to_numeric(sub["psf_flux"], errors="coerce").iloc[0]
            if pd.notna(pf) and math.isfinite(float(pf)):
                psf_fluxes.append(float(pf))
            if proc_df is not None and not proc_df.empty and "catalog_id" in proc_df.columns:
                pcopy = proc_df.copy()
                pcopy["_cid"] = pcopy["catalog_id"].map(normalize_gaia_source_id)
                psub = pcopy[pcopy["_cid"] == cid]
                if not psub.empty and "dao_flux" in psub.columns:
                    d = pd.to_numeric(psub["dao_flux"], errors="coerce").iloc[0]
                    if pd.notna(d) and float(d) > 0:
                        dao_fluxes.append(float(d))

        def _rms_pct(vals: list[float]) -> float:
            if len(vals) < 2:
                return float("nan")
            s = pd.Series(vals, dtype=np.float64)
            m = float(s.mean())
            if m <= 0:
                return float("nan")
            return 100.0 * float(s.std(ddof=1)) / m

        if len(psf_fluxes) < 2 or len(dao_fluxes) < 2:
            continue
        dr = _rms_pct(dao_fluxes)
        pr = _rms_pct(psf_fluxes)
        print(f"    {cid}: dao_rms_pct={dr:.2f}  psf_rms_pct={pr:.2f}")
        shown += 1
        if shown >= 5:
            break
    if shown == 0:
        print("    (ziadna COMP s dvoma platnymi PSF aj dao bodmi v tychto framoch)")


def _proc_metrics_by_catalog(proc_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """catalog_id -> peak_max_adu, dao_flux, saturate_limit_adu from per-frame proc CSV."""
    out: dict[str, dict[str, float]] = {}
    if proc_df is None or proc_df.empty or "catalog_id" not in proc_df.columns:
        return out
    for _, r in proc_df.iterrows():
        k = normalize_gaia_source_id(r.get("catalog_id"))
        if not k:
            continue
        peak = pd.to_numeric(r.get("peak_max_adu"), errors="coerce")
        dao = pd.to_numeric(r.get("dao_flux"), errors="coerce")
        if pd.isna(dao):
            dao = pd.to_numeric(r.get("flux"), errors="coerce")
        lim = pd.to_numeric(r.get("saturate_limit_adu"), errors="coerce")
        out[k] = {
            "peak_max_adu": float(peak) if pd.notna(peak) else float("nan"),
            "dao_flux": float(dao) if pd.notna(dao) else float("nan"),
            "saturate_limit_adu": float(lim) if pd.notna(lim) else float("nan"),
        }
    return out


def _comp_catalog_sat_peak(comp_df: pd.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
    """From comparison_stars.csv: catalog_id -> saturate_limit_adu and peak_max_adu (catalog)."""
    lims: dict[str, float] = {}
    peaks: dict[str, float] = {}
    if comp_df is None or comp_df.empty or "catalog_id" not in comp_df.columns:
        return lims, peaks
    for _, r in comp_df.iterrows():
        cid = normalize_gaia_source_id(r.get("catalog_id"))
        if not cid:
            continue
        lim = pd.to_numeric(r.get("saturate_limit_adu"), errors="coerce")
        if pd.isna(lim) or float(lim) <= 0:
            lim = pd.to_numeric(r.get("saturate_limit_adu_85pct"), errors="coerce")
        if pd.notna(lim) and float(lim) > 0:
            lims[cid] = float(lim)
        pk = pd.to_numeric(r.get("peak_max_adu"), errors="coerce")
        if pd.notna(pk) and float(pk) > 0:
            peaks[cid] = float(pk)
    return lims, peaks


def _print_table(df: pd.DataFrame, cols: list[str], *, max_rows: int = 30) -> None:
    if df.empty:
        print("  (prazdne)")
        return
    show = df.copy()
    for c in cols:
        if c not in show.columns:
            show[c] = ""
    show = show[cols]
    if len(show) > max_rows:
        show = show.head(max_rows)
    print(show.to_string(index=False))
    if len(df) > max_rows:
        print(f"  ... +{len(df) - max_rows} dalsich riadkov")


def flag_blended_stars(comp_df: pd.DataFrame, *, fwhm_px: float = 3.2) -> pd.Series:
    """True kde ma COMP suseda blizsie nez ``MIN_COMP_SEPARATION_PX`` (predvolene 3x ``fwhm_px``)."""
    _ = fwhm_px  # volitelne API; prah je z konfiguracie
    min_sep = float(MIN_COMP_SEPARATION_PX)
    if comp_df.empty or len(comp_df) < 2:
        return pd.Series(False, index=comp_df.index, dtype=bool)
    xs = pd.to_numeric(comp_df["x"], errors="coerce").to_numpy(dtype=np.float64)
    ys = pd.to_numeric(comp_df["y"], errors="coerce").to_numpy(dtype=np.float64)
    dx = xs[:, np.newaxis] - xs[np.newaxis, :]
    dy = ys[:, np.newaxis] - ys[np.newaxis, :]
    dist = np.hypot(dx, dy)
    np.fill_diagonal(dist, np.inf)
    min_d = np.min(dist, axis=1)
    blended = np.isfinite(min_d) & (min_d < float(min_sep))
    return pd.Series(blended, index=comp_df.index, dtype=bool)


def _paths() -> dict[str, Path]:
    draft_dir = ARCHIVE_ROOT / "Drafts" / f"draft_{DRAFT_ID:06d}"
    ps_dir = draft_dir / "platesolve" / OBS_GROUP
    masterstar_fits = ps_dir / "MASTERSTAR.fits"
    masterstars_csv = ps_dir / "masterstars_full_match.csv"
    epsf_fits = ps_dir / "masterstar_epsf.fits"
    epsf_meta = ps_dir / "masterstar_epsf_meta.json"
    variable_targets = ps_dir / "variable_targets.csv"
    comparison_stars = ps_dir / "comparison_stars.csv"
    # Per-frame CSV from pipeline is stored under detrended_aligned (proc_*.csv).
    epsf_data_dir = draft_dir / "detrended_aligned" / "lights" / OBS_GROUP
    # Sandbox output lives under epsf_data/ to avoid touching production directories.
    output_psf_dir = (draft_dir / "epsf_data" / "psf_results") if bool(USE_SANDBOX) else (draft_dir / "psf_results")
    return {
        "draft_dir": draft_dir,
        "ps_dir": ps_dir,
        "masterstar_fits": masterstar_fits,
        "masterstars_csv": masterstars_csv,
        "epsf_fits": epsf_fits,
        "epsf_meta": epsf_meta,
        "variable_targets": variable_targets,
        "comparison_stars": comparison_stars,
        "epsf_data_dir": epsf_data_dir,
        "output_psf_dir": output_psf_dir,
    }


def _build_frame_xy_lookup(proc_df: pd.DataFrame) -> tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]]]:
    """Return (by_catalog_id, by_name) -> (x,y)."""
    by_cid: dict[str, tuple[float, float]] = {}
    by_name: dict[str, tuple[float, float]] = {}
    if proc_df is None or proc_df.empty:
        return by_cid, by_name
    if "x" not in proc_df.columns or "y" not in proc_df.columns:
        return by_cid, by_name

    # Prefer rows that have a non-empty catalog_id.
    for _, r in proc_df.iterrows():
        try:
            x = float(r.get("x"))
            y = float(r.get("y"))
        except Exception:  # noqa: BLE001
            continue
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        cid = normalize_gaia_source_id(r.get("catalog_id")) if "catalog_id" in proc_df.columns else ""
        if cid and cid not in by_cid:
            by_cid[cid] = (x, y)
        nm = str(r.get("name", "")).strip()
        if nm and nm not in by_name:
            by_name[nm] = (x, y)
    return by_cid, by_name


def main() -> int:
    _force_utf8_stdout()
    parser = argparse.ArgumentParser(description="VYVAR PSF Runner")
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Preskoci budovanie ePSF (pouzije existujuci)",
    )
    parser.add_argument(
        "--only-build",
        action="store_true",
        help="Len postavi ePSF model, bez fotometrie",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Spracuj len prvych N framov (pre rychly test)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Nic nezapisuj, len vypis co by sa stalo",
    )
    args = parser.parse_args()

    global DRY_RUN
    if args.dry_run:
        DRY_RUN = True

    print("VYVAR PSF Runner")
    print(f"Draft ID    : {DRAFT_ID}")
    print(f"Archive     : {ARCHIVE_ROOT}")
    print(f"Obs group   : {OBS_GROUP}")
    print(f"Dry run     : {DRY_RUN}")
    print()

    from epsf_stage import EpsfStagePaths, run_epsf_stage

    p = _paths()
    cfg = AppConfig()
    db_path = _runner_database_path()
    if not db_path.is_file():
        raise FileNotFoundError(f"Database not found: {db_path}")
    db = VyvarDatabase(db_path)
    max_frames = args.frames
    do_build = not args.skip_build
    do_fit = not args.only_build
    do_lc = not args.only_build
    if args.frames is not None and int(args.frames) == 0:
        do_fit = False
        do_lc = False
        max_frames = None
        print("[--frames 0] skip fit+LC (build only if not --skip-build)")
    try:
        run_epsf_stage(
            params=None,
            paths=EpsfStagePaths(
                platesolve_dir=p["ps_dir"],
                frames_root=p["epsf_data_dir"],
                masterstar_fits=p["masterstar_fits"],
                masterstars_csv=p["masterstars_csv"],
            ),
            cfg=cfg,
            progress_cb=lambda msg: print(msg),
            db=db,
            draft_id=int(DRAFT_ID),
            do_build=do_build,
            do_fit_merge=do_fit,
            do_lc=do_lc,
            max_frames=max_frames,
            dry_run=bool(args.dry_run),
        )
    finally:
        try:
            db.conn.close()
        except Exception:  # noqa: BLE001
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

