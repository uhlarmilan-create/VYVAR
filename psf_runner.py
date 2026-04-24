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

# ── KONFIGURÁCIA ── upravuj podľa potreby ──────────────────────────
DRAFT_ID = 248
ARCHIVE_ROOT = Path(r"C:\ASTRO\python\VYVAR\Archive")
OBS_GROUP = "NoFilter_60_2"
USE_SANDBOX = True
OVERSAMPLING = 2
MIN_STARS = 15
DRY_RUN = False  # True = len vypíše čo by spravil, nezapíše CSV
# QHY294PROM, Read Mode 0, Gain setting 0 → 3.17 e⁻/ADU
# Zdroj: DB EQUIPMENTS.GAIN_ADU (get_equipment_cosmic_params)
# Fallback ak DB vráti None alebo 0:
GAIN_FALLBACK_E_PER_ADU = 3.17
BORDER_WIDTH = 3  # pixely okraja pre lokálny odhad oblohy v cutoute
SATURATE_PEAK_FRAC = 0.85  # Skip COMP PSF if peak_max_adu > frac * saturate_limit_adu
SUMMARY_MIN_MED_PSF_FLUX = 100.0  # Drop numerical dust in psf_summary.csv
SUMMARY_MIN_N_FIT_OK = 5  # Require enough good frames per star in summary (používa sa s n_fit_ok_report)
MAX_COMP_PEAK_ADU = 40_000  # vyraď príliš jasné COMP
# Engine: ``psf_fit_ok`` = konvergencia ∧ (χ² < PSF_VAR_CHI2_MAX pre VAR, χ² < PSF_COMP_CHI2_MAX pre COMP).
# Reporting v step_4 používa rovnaké prahy pri ``n_fit_ok_report``.
PSF_VAR_CHI2_MAX = 1000.0  # VAR — svetelná krivka: akceptuj vyššie χ² (error mapa často podhodnotená pri jasných)
PSF_COMP_CHI2_MAX = 20.0  # COMP — prísnejší prah
MIN_COMP_SEPARATION_PX = 3 * 3.2  # ~10 px = 3× FWHM (crowding filter)
# Krok 5: štyri COMP hviezdy pre frame-wise ZP z psf_summary (med_psf_flux).
COMP_CALIB_CATALOG_IDS: tuple[str, ...] = (
    "1496835173974894848",
    "1499921468754586112",
    "1498311264039176192",
    "1497439905370466816",
)
PSF_CAL_MAG_ZP_OFFSET = 20.0  # arbitrárny offset pre relatívnu krivku (mag)
# ──────────────────────────────────────────────────────────────────

# Cielené VAR hviezdy pre tento test (ak neprázdne, nahradí variable_targets.csv v step_2_load_targets)
PSF_TARGET_OVERRIDE: list[dict[str, Any]] = [
    {
        "catalog_id": 1498486880958321200,
        "vsx_name": "CSS_J140918.7+423422",
        "vsx_type": "EW",
        "x": 265.155,
        "y": 177.707,
        "note": "Slabá EW mag=12.75 — hlavný PSF test",
    },
    {
        "catalog_id": 1497418258735289300,
        "vsx_name": "FU CVn",
        "vsx_type": "EW",
        "x": 1444.436,
        "y": 641.743,
        "note": "EW mag=11.0 — porovnanie s draft 247",
    },
]

# Jediné povolené importy z projektu:
from astropy.table import Table  # noqa: E402
from photutils.psf import ImagePSF, PSFPhotometry  # noqa: E402

from psf_photometry import build_epsf_model  # noqa: E402
from database import VyvarDatabase  # noqa: E402
from config import AppConfig  # noqa: E402
from infolog import log_event  # noqa: E402


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
        from astropy.stats import sigma_clipped_stats  # noqa: PLC0415

        _, med, std = sigma_clipped_stats(vals, sigma=3.0, maxiters=5, cenfunc="median", stdfunc="std")
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
    """Pozitívny e-/ADU len z ``EGAIN`` (``GAIN`` z INDI často 0 — neberieme)."""
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
    """Priorita: DB ``GAIN_ADU`` → FITS ``EGAIN`` → ``GAIN_FALLBACK_E_PER_ADU``."""
    if db_gain_e_per_adu is not None and math.isfinite(float(db_gain_e_per_adu)) and float(db_gain_e_per_adu) > 0:
        return float(db_gain_e_per_adu), "DB"
    eg = _header_egain_only(hdr)
    if eg is not None:
        return float(eg), "FITS EGAIN"
    return float(GAIN_FALLBACK_E_PER_ADU), "fallback"


def _vy_qcrms_from_header(hdr: Any) -> float | None:
    """Kladné ``VY_QCRMS`` z hlavičky (ADU), alebo None."""
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
    """``VY_FWHM`` [px]; NaN ak chýba alebo neplatné."""
    if "VY_FWHM" not in hdr:
        return float("nan")
    try:
        v = float(hdr["VY_FWHM"])
        return v if math.isfinite(v) and v > 0 else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _draft_equipment_id(db: VyvarDatabase, draft_id: int) -> int | None:
    """``OBS_DRAFT.ID_EQUIPMENTS`` for this draft, if set and positive."""
    row = db.conn.execute(
        "SELECT ID_EQUIPMENTS FROM OBS_DRAFT WHERE ID = ?;",
        (int(draft_id),),
    ).fetchone()
    if row is None or row["ID_EQUIPMENTS"] is None:
        return None
    try:
        eid = int(row["ID_EQUIPMENTS"])
    except (TypeError, ValueError):
        return None
    return eid if eid > 0 else None


def _draft_db_gain_e_per_adu(db: VyvarDatabase | None, draft_id: int) -> float | None:
    """``EQUIPMENTS.GAIN_ADU`` pre ``OBS_DRAFT.ID_EQUIPMENTS``, ak je kladné."""
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


def _fit_shape_for_cutout(cutout_size: int) -> tuple[int, int]:
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
    # Pozn.: súčet pixelov ePSF gridu (napr. ~4 pri oversampling=2) nie je priamo „flux v ADU“;
    # PSFPhotometry škáluje amplitúdu voči dátam. Pre ``psf_flux_norm`` by sa použil integrál modelu z Photutils, nie len ``sum(psf_data)``.
    psf_model = ImagePSF(psf_data, oversampling=osamp)
    fit_shape = _fit_shape_for_cutout(cutout_size)
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
    """Histogram text: všetky riadky výsledku (vrátane NaN po preskočení fitu)."""
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
    print("  chi2 distribúcia (všetky riadky výsledku):")
    print(f"    chi2 < 1:    {n_lt1}")
    print(f"    chi2 1–3:    {n_13}   ← ideálna zóna")
    print(f"    chi2 3–5:    {n_35}   ← akceptovateľné")
    print(f"    chi2 5–20:   {n_520}   ← problematické")
    print(f"    chi2 > 20:   {n_gt20}   ← divergencia")
    print(f"    chi2 = 0:    {n0}   ← numerický artefakt")
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
    """Krátky text pre diagnostiku zlyhaného COMP fitu."""
    if not (math.isfinite(x) and math.isfinite(y)):
        return "neplatná pozícia"
    xi, yi = int(round(x)), int(round(y))
    parts: list[str] = []
    if xi < half_cs or yi < half_cs or xi >= fw - half_cs or yi >= fh - half_cs:
        parts.append("okraj čipu")
    if (
        math.isfinite(peak_adu)
        and math.isfinite(sat_lim_adu)
        and sat_lim_adu > 0
        and peak_adu > float(SATURATE_PEAK_FRAC) * sat_lim_adu
    ):
        parts.append("blízko saturácie")
    if not parts:
        parts.append("blend / PSF nezhoda / iné")
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
    print("\n═══ DRY-RUN report (2 framy) ═══")
    print(f"  chi2 median VAR (fit_ok only): {med_v_ok:.3f}  |  (všetky konečné): {med_v_all:.3f}")
    print(f"  chi2 median COMP (fit_ok only): {med_c_ok:.3f}  |  (všetky konečné): {med_c_all:.3f}")
    print(f"  fit_ok: {n_ok}/{n} ({pct:.1f}%)")
    _print_chi2_distribution(chi)

    print(f"  prvých 5 COMP s ≥2 platnými psf+dao bodmi — dao_rms_pct vs psf_rms_pct (n={len(bundle)} framov):")
    shown = 0
    for raw_cid in comp_df["catalog_id"].tolist():
        cid = _cid_key(raw_cid)
        if not cid:
            continue
        psf_fluxes: list[float] = []
        dao_fluxes: list[float] = []
        for res_df, proc_df in bundle:
            res_df = res_df.copy()
            res_df["_cid"] = res_df["catalog_id"].map(_cid_key)
            sub = res_df[(res_df["_cid"] == cid) & (res_df["role"].astype(str).str.upper() == "COMP")]
            if sub.empty:
                continue
            pf = pd.to_numeric(sub["psf_flux"], errors="coerce").iloc[0]
            if pd.notna(pf) and math.isfinite(float(pf)):
                psf_fluxes.append(float(pf))
            if proc_df is not None and not proc_df.empty and "catalog_id" in proc_df.columns:
                pcopy = proc_df.copy()
                pcopy["_cid"] = pcopy["catalog_id"].map(_cid_key)
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
        print("    (žiadna COMP s dvoma platnými PSF aj dao bodmi v týchto framoch)")


def _proc_metrics_by_catalog(proc_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """catalog_id → peak_max_adu, dao_flux, saturate_limit_adu from per-frame proc CSV."""
    out: dict[str, dict[str, float]] = {}
    if proc_df is None or proc_df.empty or "catalog_id" not in proc_df.columns:
        return out
    for _, r in proc_df.iterrows():
        k = _cid_key(r.get("catalog_id"))
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
    """From comparison_stars.csv: catalog_id → saturate_limit_adu and peak_max_adu (catalog)."""
    lims: dict[str, float] = {}
    peaks: dict[str, float] = {}
    if comp_df is None or comp_df.empty or "catalog_id" not in comp_df.columns:
        return lims, peaks
    for _, r in comp_df.iterrows():
        cid = _cid_key(r.get("catalog_id"))
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


def _cid_key(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return ""
    # Preserve precision for huge Gaia IDs (float conversion loses digits).
    if isinstance(v, (int, np.integer)):
        try:
            return str(int(v))
        except Exception:  # noqa: BLE001
            return str(v).strip()
    if isinstance(v, float) and math.isnan(v):
        return ""
    s = str(v).strip()
    if not s or s.lower() in ("nan", "none"):
        return ""
    # Common case: float like 4.7e+17
    try:
        f = float(v)
        if math.isfinite(f) and abs(f) > 1e10:
            return str(int(f))
        if math.isfinite(f) and float(int(f)) == f:
            return str(int(f))
    except (TypeError, ValueError, OverflowError):
        pass
    # Strip trailing .0 if present
    if s.endswith(".0") and s[:-2].isdigit():
        return s[:-2]
    return s


def _print_table(df: pd.DataFrame, cols: list[str], *, max_rows: int = 30) -> None:
    if df.empty:
        print("  (prázdne)")
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
        print(f"  ... +{len(df) - max_rows} ďalších riadkov")


def flag_blended_stars(comp_df: pd.DataFrame, *, fwhm_px: float = 3.2) -> pd.Series:
    """True kde má COMP suseda bližšie než ``MIN_COMP_SEPARATION_PX`` (predvolene 3× ``fwhm_px``)."""
    _ = fwhm_px  # voliteľné API; prah je z konfigurácie
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


def step_1_build_epsf() -> Path:
    print("═══ KROK 1: Budovanie ePSF modelu ═══")
    p = _paths()
    masterstar_fits = p["masterstar_fits"]
    masterstars_csv = p["masterstars_csv"]
    if not masterstar_fits.is_file():
        raise FileNotFoundError(f"Missing MASTERSTAR FITS: {masterstar_fits}")
    if not masterstars_csv.is_file():
        raise FileNotFoundError(f"Missing masterstars CSV: {masterstars_csv}")

    print(f"MASTERSTAR: {masterstar_fits}")
    print(f"Masterstars CSV: {masterstars_csv}")

    db_path = _runner_database_path()
    if not db_path.is_file():
        raise FileNotFoundError(f"Database not found: {db_path}")
    db = VyvarDatabase(db_path)
    try:
        epsf_path = build_epsf_model(
            masterstar_fits,
            masterstars_csv,
            db,
            int(DRAFT_ID),
            oversampling=int(OVERSAMPLING),
            min_stars=int(MIN_STARS),
        )
    finally:
        try:
            db.conn.close()
        except Exception:  # noqa: BLE001
            pass

    print(f"✓ ePSF uložený: {epsf_path}")

    meta_fp = p["epsf_meta"]
    try:
        meta = json.loads(meta_fp.read_text(encoding="utf-8"))
        keys = [
            "fwhm_px",
            "cutout_size",
            "oversampling",
            "n_stars_used",
            "created_utc",
            "draft_id",
        ]
        for k in keys:
            if k in meta:
                print(f"  {k:<12}: {meta[k]}")
        extra = sorted([k for k in meta.keys() if k not in keys])
        for k in extra:
            print(f"  {k:<12}: {meta[k]}")
    except Exception as exc:  # noqa: BLE001
        print(f"  (meta JSON sa nepodarilo načítať: {exc})")

    return Path(epsf_path)


def step_2_load_targets() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("═══ KROK 2: Načítanie VAR + COMP hviezd ═══")
    p = _paths()
    var_path = p["variable_targets"]
    comp_path = p["comparison_stars"]

    if not comp_path.is_file():
        raise FileNotFoundError(f"Missing comparison_stars.csv: {comp_path}")

    comp_df = pd.read_csv(comp_path, low_memory=False)

    if PSF_TARGET_OVERRIDE:
        var_df = pd.DataFrame(PSF_TARGET_OVERRIDE)
        var_df["role"] = "VAR"
        print(f"PSF_TARGET_OVERRIDE aktívny: {len(var_df)} cielených VAR hviezd")
        for _, r in var_df.iterrows():
            try:
                nm = str(r.get("vsx_name", "") or "").strip()
                vt = str(r.get("vsx_type", "") or "").strip()
                x = float(r.get("x"))
                y = float(r.get("y"))
                note = str(r.get("note", "") or "").strip()
                print(f"  🎯 {nm} ({vt}) x={x:.1f} y={y:.1f} — {note}")
            except Exception:  # noqa: BLE001
                pass
        print()
    else:
        if not var_path.is_file():
            raise FileNotFoundError(f"Missing variable_targets.csv: {var_path}")
        var_df = pd.read_csv(var_path, low_memory=False)

    if "catalog_id" in var_df.columns:
        var_df = var_df.copy()
        var_df["catalog_id"] = var_df["catalog_id"].map(_cid_key)
        var_df = var_df[var_df["catalog_id"].astype(str).str.strip() != ""]
    var_df = var_df.reset_index(drop=True)

    if "is_usable" in comp_df.columns:
        comp_df = comp_df[comp_df["is_usable"].fillna(False).astype(bool)]
    if "catalog_id" in comp_df.columns:
        comp_df = comp_df.copy()
        comp_df["catalog_id"] = comp_df["catalog_id"].map(_cid_key)
        comp_df = comp_df[comp_df["catalog_id"].astype(str).str.strip() != ""]
    comp_df = comp_df.reset_index(drop=True)

    # Avoid duplicated sources across roles (variables must not appear in COMP list).
    if "catalog_id" in var_df.columns and "catalog_id" in comp_df.columns:
        var_ids = set(var_df["catalog_id"].astype(str).tolist())
        comp_df = comp_df[~comp_df["catalog_id"].astype(str).isin(var_ids)].reset_index(drop=True)

    print("  COMP filtre (jas + crowding):")
    n_before = len(comp_df)
    if "peak_dao" in comp_df.columns:
        peak_adu = pd.to_numeric(comp_df["peak_dao"], errors="coerce")
    elif "peak_max_adu" in comp_df.columns:
        peak_adu = pd.to_numeric(comp_df["peak_max_adu"], errors="coerce")
    else:
        peak_adu = pd.Series(np.nan, index=comp_df.index)
    comp_df = comp_df[peak_adu.isna() | (peak_adu < float(MAX_COMP_PEAK_ADU))].copy()
    print(f"  Vyradené príliš jasné COMP (peak>={MAX_COMP_PEAK_ADU}): {n_before - len(comp_df)}")

    blended = flag_blended_stars(comp_df, fwhm_px=3.2)
    n_bl = int(blended.sum())
    comp_df = comp_df[~blended].reset_index(drop=True)
    print(f"  Vyradené blendované COMP (<{MIN_COMP_SEPARATION_PX:.0f}px od suseda): {n_bl}")

    print(f"  COMP po filtroch: {len(comp_df)}")
    print()

    print(f"VAR hviezdy (N={len(var_df)}):")
    _print_table(var_df, ["catalog_id", "vsx_name", "vsx_type", "x", "y"], max_rows=30)
    print()
    print(f"COMP hviezdy (N={len(comp_df)}):")
    _print_table(comp_df, ["catalog_id", "mag", "bp_rp", "x", "y"], max_rows=30)
    print()

    return var_df, comp_df


def _build_frame_xy_lookup(proc_df: pd.DataFrame) -> tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]]]:
    """Return (by_catalog_id, by_name) → (x,y)."""
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
        cid = _cid_key(r.get("catalog_id")) if "catalog_id" in proc_df.columns else ""
        if cid and cid not in by_cid:
            by_cid[cid] = (x, y)
        nm = str(r.get("name", "")).strip()
        if nm and nm not in by_name:
            by_name[nm] = (x, y)
    return by_cid, by_name


def step_3_run_psf_on_frames(
    var_df: pd.DataFrame,
    comp_df: pd.DataFrame,
    epsf_path: Path,
    *,
    max_frames: int | None = None,
) -> None:
    print("═══ KROK 3: PSF fotometria na framoch ═══")
    p = _paths()
    epsf_data_dir = p["epsf_data_dir"]
    output_psf_dir = p["output_psf_dir"]
    output_psf_dir.mkdir(parents=True, exist_ok=True)

    if not epsf_data_dir.is_dir():
        raise FileNotFoundError(f"epsf_data_dir missing: {epsf_data_dir}")

    frames = sorted(epsf_data_dir.glob("proc_*.csv"))
    if max_frames is not None:
        frames = frames[: int(max_frames)]
    print(f"Nájdených framov: {len(frames)}")
    if not frames:
        return

    db_path = _runner_database_path()
    db: VyvarDatabase | None = None
    if db_path.is_file():
        db = VyvarDatabase(db_path)
        print(f"  DB (záložný gain): {db_path}")
    else:
        print(
            f"  VAROVANIE: databáza neexistuje ({db_path}) — gain z EGAIN alebo GAIN_FALLBACK_E_PER_ADU ({GAIN_FALLBACK_E_PER_ADU})"
        )
    db_gain_e_per_adu: float | None = None
    try:
        db_gain_e_per_adu = _draft_db_gain_e_per_adu(db, int(DRAFT_ID))
        if db_gain_e_per_adu is not None:
            print(f"  Gain z DB (GAIN_ADU): {db_gain_e_per_adu:g} e/ADU")
        else:
            print("  Gain z DB: nie je — použije sa FITS EGAIN alebo GAIN_FALLBACK_E_PER_ADU")
    finally:
        if db is not None:
            try:
                db.conn.close()
            except Exception:  # noqa: BLE001
                pass

    warned_fallback_gain = False

    # Common star_positions template (master positions).
    var_part = pd.DataFrame(
        {
            "catalog_id": var_df.get("catalog_id", pd.Series([], dtype=str)).map(_cid_key),
            "name": var_df.get("vsx_name", var_df.get("name", var_df.get("catalog_id"))).astype(str),
            "x": pd.to_numeric(var_df.get("x"), errors="coerce"),
            "y": pd.to_numeric(var_df.get("y"), errors="coerce"),
            "role": "VAR",
        }
    )
    comp_part = pd.DataFrame(
        {
            "catalog_id": comp_df.get("catalog_id", pd.Series([], dtype=str)).map(_cid_key),
            "name": comp_df.get("catalog_id", pd.Series([], dtype=str)).map(_cid_key),
            "x": pd.to_numeric(comp_df.get("x"), errors="coerce"),
            "y": pd.to_numeric(comp_df.get("y"), errors="coerce"),
            "role": "COMP",
        }
    )
    star_positions = pd.concat([var_part, comp_part], ignore_index=True)
    star_positions = star_positions[star_positions["catalog_id"].astype(str).str.strip() != ""].reset_index(drop=True)

    phot, cutout_size = _load_psf_photometry_bundle(Path(epsf_path))
    dry_bundle: list[tuple[pd.DataFrame, pd.DataFrame]] = []

    n_failed = 0
    n_processed = 0
    total_fits = 0
    total_ok = 0

    for i, csv_path in enumerate(frames, start=1):
        frame_stem = csv_path.stem
        print(f"[{i:3d}/{len(frames)}] {csv_path.name} ...")
        try:
            fits_path = csv_path.with_suffix(".fits")
            if not fits_path.is_file():
                print("  ✗ FITS nenájdený, preskočujem")
                continue

            with fits.open(fits_path, memmap=False) as hdul:
                hdr = hdul[0].header
                frame_data = np.asarray(hdul[0].data, dtype=np.float32)

            gain, gain_src = get_gain_from_header(hdr, db_gain_e_per_adu=db_gain_e_per_adu)
            if gain_src == "fallback" and not warned_fallback_gain:
                print(
                    f"  VAROVANIE: chýba GAIN_ADU v DB a EGAIN vo FITS — používam GAIN_FALLBACK_E_PER_ADU={GAIN_FALLBACK_E_PER_ADU} e/ADU."
                )
                warned_fallback_gain = True

            vy_q_hdr = _vy_qcrms_from_header(hdr)
            vy_fwhm = _vy_fwhm_from_header(hdr)

            # PSFPhotometry model here does not fit a background term; subtract a robust global background
            # so the fitted flux corresponds to source flux (not background×area).
            try:
                bg = float(np.nanmedian(frame_data))
            except Exception:  # noqa: BLE001
                bg = float("nan")
            if math.isfinite(bg):
                frame_data_psf = np.asarray(frame_data - bg, dtype=np.float32)
            else:
                frame_data_psf = np.asarray(frame_data, dtype=np.float32)
            frame_data_psf = np.nan_to_num(frame_data_psf, nan=0.0, posinf=0.0, neginf=0.0)
            if math.isfinite(bg):
                print(f"  background subtracted: median={bg:.2f} ADU")

            proc_df = pd.read_csv(csv_path, low_memory=False)
            by_cid, by_name = _build_frame_xy_lookup(proc_df)
            proc_meta = _proc_metrics_by_catalog(proc_df)
            comp_lim, comp_peak = _comp_catalog_sat_peak(comp_df)

            upd = star_positions.copy()
            used_pf = 0
            used_fb = 0
            xs: list[float] = []
            ys: list[float] = []
            for _, r in upd.iterrows():
                cid = _cid_key(r.get("catalog_id"))
                nm = str(r["name"]).strip()
                if cid and cid in by_cid:
                    x, y = by_cid[cid]
                    used_pf += 1
                elif nm in by_name:
                    x, y = by_name[nm]
                    used_pf += 1
                else:
                    try:
                        x = float(r["x"])
                        y = float(r["y"])
                    except Exception:  # noqa: BLE001
                        x, y = float("nan"), float("nan")
                    used_fb += 1
                xs.append(x)
                ys.append(y)
            upd["x"] = xs
            upd["y"] = ys
            print(f"  pozície: per-frame={used_pf}, fallback(master)={used_fb}")

            upd_ann = upd.copy()
            upd_ann["_ord"] = np.arange(len(upd_ann), dtype=np.int64)
            fh, fw = int(frame_data_psf.shape[0]), int(frame_data_psf.shape[1])
            half_cs = int(cutout_size) // 2
            skip_pre: list[bool] = []
            n_sat = 0
            n_edge = 0
            for _, r in upd_ann.iterrows():
                cid = _cid_key(r.get("catalog_id"))
                meta = proc_meta.get(cid, {})
                peak = float(meta.get("peak_max_adu", float("nan")))
                lim = float(meta.get("saturate_limit_adu", float("nan")))
                if not (math.isfinite(lim) and lim > 0):
                    lim = float(comp_lim.get(cid, float("nan")))
                if not (math.isfinite(peak) and peak > 0):
                    peak = float(comp_peak.get(cid, float("nan")))
                role_u = str(r.get("role", "")).strip().upper()
                is_sat = (
                    role_u == "COMP"
                    and math.isfinite(peak)
                    and math.isfinite(lim)
                    and lim > 0
                    and peak > float(SATURATE_PEAK_FRAC) * lim
                )
                try:
                    xe = float(r["x"])
                    ye = float(r["y"])
                except (TypeError, ValueError):
                    xe, ye = float("nan"), float("nan")
                is_edge = False
                if role_u == "COMP" and math.isfinite(xe) and math.isfinite(ye):
                    xir, yir = int(round(xe)), int(round(ye))
                    is_edge = xir < half_cs or yir < half_cs or xir >= fw - half_cs or yir >= fh - half_cs
                skip_pre.append(bool(is_sat or is_edge))
                if is_sat:
                    n_sat += 1
                if is_edge:
                    n_edge += 1

            upd_ann["_skip_pre"] = skip_pre
            if n_sat:
                print(f"  Preskočené saturované COMP (pred fitom): {n_sat}")
            if n_edge:
                print(f"  Preskočené COMP pri okraji (cutout {cutout_size}×{cutout_size}): {n_edge}")

            upd_fit = upd_ann[~upd_ann["_skip_pre"]].copy()
            upd_skip = upd_ann[upd_ann["_skip_pre"]].copy()

            if upd_fit.empty:
                res = pd.DataFrame(
                    columns=[
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
                )
                sky_med, sky_rms_med = float("nan"), float("nan")
            else:
                part_main, sky_med, sky_rms_med = _psf_stars_local_cutouts(
                    frame_data_psf,
                    upd_fit[["x", "y", "catalog_id", "name", "role"]].reset_index(drop=True),
                    phot,
                    cutout_size,
                    border_width=int(BORDER_WIDTH),
                    gain=gain,
                    vy_qcrms_adu=vy_q_hdr,
                )
                part_main["_ord"] = upd_fit["_ord"].to_numpy()
                res = part_main.sort_values("_ord").drop(columns=["_ord"], errors="ignore")

            rms_show = float(vy_q_hdr) if vy_q_hdr is not None else float(sky_rms_med)
            if not math.isfinite(rms_show):
                rms_show = 0.0
            fwhm_show = vy_fwhm if math.isfinite(vy_fwhm) else float("nan")
            print(
                f"  gain={gain:.2f} e/ADU ({gain_src}) | VY_QCRMS={rms_show:.1f} | VY_FWHM={fwhm_show:.2f}px"
                if math.isfinite(fwhm_show)
                else f"  gain={gain:.2f} e/ADU ({gain_src}) | VY_QCRMS={rms_show:.1f} | VY_FWHM=nan"
            )

            if not upd_skip.empty:
                res_skip = pd.DataFrame(
                    {
                        "catalog_id": upd_skip["catalog_id"].to_numpy(),
                        "name": upd_skip["name"].to_numpy(),
                        "x": upd_skip["x"].to_numpy(),
                        "y": upd_skip["y"].to_numpy(),
                        "psf_flux": np.full(len(upd_skip), np.nan),
                        "psf_flux_err": np.full(len(upd_skip), np.nan),
                        "psf_chi2": np.full(len(upd_skip), np.nan),
                        "psf_converged": np.zeros(len(upd_skip), dtype=bool),
                        "psf_fit_ok": np.zeros(len(upd_skip), dtype=bool),
                    }
                )
                res_skip["_ord"] = upd_skip["_ord"].to_numpy()
                res = pd.concat([res, res_skip], ignore_index=True) if not res.empty else res_skip
                res = res.sort_values("_ord").drop(columns=["_ord"], errors="ignore")

            # Join role + frame metadata (optional bjd).
            role_map = {_cid_key(r["catalog_id"]): r["role"] for _, r in upd_ann.iterrows()}
            res = res.copy()
            res["role"] = res["catalog_id"].map(lambda x: role_map.get(_cid_key(x), ""))
            res["frame_stem"] = frame_stem
            if "bjd_tdb_mid" in proc_df.columns:
                try:
                    bjd_val = float(pd.to_numeric(proc_df["bjd_tdb_mid"], errors="coerce").dropna().iloc[0])
                except Exception:  # noqa: BLE001
                    bjd_val = float("nan")
                res["bjd_tdb_mid"] = bjd_val
            else:
                res["bjd_tdb_mid"] = float("nan")

            # Stats printout.
            var_mask = res["role"] == "VAR"
            comp_mask = res["role"] == "COMP"
            n_var = int(var_mask.sum())
            n_comp = int(comp_mask.sum())
            n_var_ok = int((var_mask & res["psf_fit_ok"].fillna(False).astype(bool)).sum())
            n_comp_ok = int((comp_mask & res["psf_fit_ok"].fillna(False).astype(bool)).sum())
            chi = pd.to_numeric(res["psf_chi2"], errors="coerce")
            okm = res["psf_fit_ok"].fillna(False).astype(bool)
            chi_med = float(chi.dropna().median()) if len(res) else float("nan")
            chi_var_med = float(chi[var_mask & okm].dropna().median()) if (var_mask & okm).any() else float("nan")
            chi_comp_med = float(chi[comp_mask & okm].dropna().median()) if (comp_mask & okm).any() else float("nan")
            print(
                f"  ✓ fit_ok: {n_var_ok}/{n_var} VAR, {n_comp_ok}/{n_comp} COMP | "
                f"chi2 med (VAR/COMP/all): {chi_var_med:.2f} / {chi_comp_med:.2f} / {chi_med:.2f}"
            )

            # Per-VAR diagnostics (requested for quick comparisons)
            try:
                var_rows = res[res["role"] == "VAR"].copy()
                if not var_rows.empty:
                    var_rows["_chi"] = pd.to_numeric(var_rows["psf_chi2"], errors="coerce")
                    var_rows["_ok"] = var_rows["psf_fit_ok"].fillna(False).astype(bool)
                    var_rows["_nm"] = var_rows["name"].astype(str)
                    # stable order by name then catalog_id
                    var_rows = var_rows.sort_values(["_nm", "catalog_id"])
                    for _, vr in var_rows.iterrows():
                        nm = str(vr.get("_nm", "")).strip()
                        cid = _cid_key(vr.get("catalog_id"))
                        ok = bool(vr.get("_ok", False))
                        chi_v = vr.get("_chi", float("nan"))
                        chi_s = f"{float(chi_v):.2f}" if pd.notna(chi_v) and math.isfinite(float(chi_v)) else "nan"
                        print(f"    VAR {nm} ({cid}): fit_ok={ok} chi2={chi_s}")
            except Exception:  # noqa: BLE001
                pass

            if DRY_RUN:
                comp_rows = res[res["role"].astype(str).str.upper() == "COMP"].copy()
                comp_rows["_xc"] = pd.to_numeric(comp_rows["psf_chi2"], errors="coerce")
                bad_comp = comp_rows[comp_rows["_xc"].notna() & (comp_rows["_xc"] > 5.0)].sort_values(
                    "_xc", ascending=False
                )
                if not bad_comp.empty:
                    print("  Failed COMP (chi2>5), prvých 5:")
                    for _, rr in bad_comp.head(5).iterrows():
                        cid = _cid_key(rr.get("catalog_id"))
                        mx = proc_meta.get(cid, {})
                        pk = float(mx.get("peak_max_adu", float("nan")))
                        lm = float(mx.get("saturate_limit_adu", float("nan")))
                        if not (math.isfinite(lm) and lm > 0):
                            lm = float(comp_lim.get(cid, float("nan")))
                        rx, ry = float(rr["x"]), float(rr["y"])
                        chim = float(rr["_xc"])
                        reason = _comp_fail_reason_psf(rx, ry, fw, fh, half_cs, pk, lm)
                        pk_s = f"{pk:.0f}" if math.isfinite(pk) else "nan"
                        print(f"    {cid} | {rx:.1f} | {ry:.1f} | peak_adu={pk_s} | chi2={chim:.2f} | {reason}")

            if DRY_RUN and max_frames is not None and int(max_frames) == 2:
                dry_bundle.append((res.copy(), proc_df.copy()))

            out_fp = output_psf_dir / f"{frame_stem}_psf.csv"
            if DRY_RUN:
                print(f"  [DRY_RUN] neukladám: {out_fp.name}")
            else:
                res.to_csv(out_fp, index=False)

            n_processed += 1
            total_fits += int(len(res))
            total_ok += int(res["psf_fit_ok"].fillna(False).astype(bool).sum())
        except Exception as exc:  # noqa: BLE001
            n_failed += 1
            print(f"  ✗ CHYBA: {exc}")
            print(traceback.format_exc())
            continue

    if DRY_RUN and max_frames is not None and int(max_frames) == 2 and dry_bundle:
        _print_dry_run_two_frame_report(dry_bundle, comp_df)

    print("═══ SÚHRN ═══")
    print(f"Framov spracovaných : {n_processed}/{len(frames)}")
    print(f"Framov zlyhalo      : {n_failed}")
    print(f"Celkových PSF fitov : {total_fits}")
    if total_fits > 0:
        pct = 100.0 * float(total_ok) / float(total_fits)
        print(f"Úspešných fitov     : {total_ok} ({pct:.1f}%)")
    else:
        print("Úspešných fitov     : 0 (—)")


def step_4_build_summary() -> None:
    print("═══ KROK 4: PSF súhrnná štatistika ═══")
    p = _paths()
    out_dir = p["output_psf_dir"]
    files = sorted(out_dir.glob("*_psf.csv"))
    if not files:
        print("Žiadne *_psf.csv súbory.")
        return
    frames: list[pd.DataFrame] = []
    for fp in files:
        try:
            frames.append(pd.read_csv(fp, low_memory=False))
        except Exception:  # noqa: BLE001
            continue
    if not frames:
        print("Žiadne načítateľné *_psf.csv súbory.")
        return
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        print("Súhrn prázdny.")
        return

    df["catalog_id"] = df["catalog_id"].map(_cid_key)
    df["role"] = df.get("role", "").astype(str)
    df["psf_fit_ok"] = df.get("psf_fit_ok", False).fillna(False).astype(bool)
    df["psf_flux"] = pd.to_numeric(df.get("psf_flux"), errors="coerce")
    df["psf_chi2"] = pd.to_numeric(df.get("psf_chi2"), errors="coerce")
    if "psf_converged" in df.columns:
        df["psf_converged"] = df["psf_converged"].fillna(False).astype(bool)
    else:
        # Staré CSV: stĺpec neexistuje — „fit sa pokúsil“ ≈ konečné χ² a flux z fitu (nie preskočený cutout).
        ch0 = pd.to_numeric(df["psf_chi2"], errors="coerce")
        fl0 = pd.to_numeric(df["psf_flux"], errors="coerce")
        df["psf_converged"] = ch0.notna() & fl0.notna()

    g = df.groupby(["catalog_id", "role"], dropna=False)
    rows: list[dict[str, Any]] = []
    for (cid, role), sub in g:
        n = int(len(sub))
        strict = sub["psf_fit_ok"].fillna(False).astype(bool)
        conv = sub["psf_converged"].fillna(False).astype(bool)
        chi = sub["psf_chi2"]
        flx = sub["psf_flux"]
        role_u = str(role).strip().upper()
        chi_cap = float(PSF_VAR_CHI2_MAX) if role_u == "VAR" else float(PSF_COMP_CHI2_MAX)
        report_ok = conv & flx.notna() & (flx > 0) & chi.notna() & (chi < chi_cap)
        n_strict = int(strict.sum())
        n_report = int(report_ok.sum())
        flux_r = sub.loc[report_ok, "psf_flux"]
        chi_r = sub.loc[report_ok, "psf_chi2"]
        med_flux = float(flux_r.median()) if len(flux_r) else float("nan")
        rms_flux = float(flux_r.std(ddof=1)) if len(flux_r) > 1 else float("nan")
        med_chi2 = float(chi_r.median()) if len(chi_r) else float("nan")
        if len(flux_r) > 1:
            m_fl = float(flux_r.mean())
            rms_pct = (100.0 * rms_flux / m_fl) if math.isfinite(m_fl) and m_fl > 0 else float("nan")
        else:
            rms_pct = float("nan")
        rows.append(
            {
                "catalog_id": cid,
                "role": role,
                "n_frames": n,
                "n_fit_ok_strict": n_strict,
                "n_fit_ok_report": n_report,
                "med_psf_flux": med_flux,
                "rms_psf_flux": rms_flux,
                "median_chi2": med_chi2,
                "rms_pct": rms_pct,
            }
        )

    out = pd.DataFrame(rows)
    # VAR first, then COMP
    out["_role_order"] = out["role"].map(lambda r: 0 if str(r).upper() == "VAR" else 1)
    out = out.sort_values(["_role_order", "catalog_id"]).drop(columns=["_role_order"])

    med_flux = pd.to_numeric(out["med_psf_flux"], errors="coerce")
    n_rep = pd.to_numeric(out["n_fit_ok_report"], errors="coerce").fillna(0).astype(int)
    n_before = int(len(out))
    print(f"Hviezd pred filtrami: {n_before}")
    print(f"  med_psf_flux <= {SUMMARY_MIN_MED_PSF_FLUX:g}:  {int((med_flux <= float(SUMMARY_MIN_MED_PSF_FLUX)).sum())}")
    print(f"  n_fit_ok_report < {SUMMARY_MIN_N_FIT_OK}:   {int((n_rep < int(SUMMARY_MIN_N_FIT_OK)).sum())}")
    mn_id = _cid_key("1591057651117374976")
    mn = out[out["catalog_id"].astype(str) == mn_id]
    if mn.empty:
        print(f"  MN Boo ({mn_id}) pred filtrom: (nie je v agregácii — žiadne *_psf riadky)")
    else:
        r0 = mn.iloc[0]
        print(
            f"  MN Boo pred filtrom: n_strict={int(r0['n_fit_ok_strict'])}/"
            f"n_report={int(r0['n_fit_ok_report'])}/{int(r0['n_frames'])}, "
            f"med_flux={float(r0['med_psf_flux']):.1f}, median_chi2={float(r0['median_chi2']):.4g}"
        )

    keep = (med_flux > float(SUMMARY_MIN_MED_PSF_FLUX)) & (n_rep >= int(SUMMARY_MIN_N_FIT_OK))
    out_save = out.loc[keep].copy()
    print(f"Po čistení súhrnu: {len(out_save)} hviezd (pred čistením: {n_before})")
    disp = out_save.copy()
    disp["strict/report/frames"] = (
        disp["n_fit_ok_strict"].astype(str)
        + "/"
        + disp["n_fit_ok_report"].astype(str)
        + "/"
        + disp["n_frames"].astype(str)
    )
    show_cols = [
        "catalog_id",
        "role",
        "strict/report/frames",
        "med_psf_flux",
        "rms_pct",
        "median_chi2",
    ]
    print(disp[show_cols].to_string(index=False))

    if DRY_RUN:
        print("[DRY_RUN] neukladám psf_summary.csv")
    else:
        out_fp = out_dir / "psf_summary.csv"
        out_save.to_csv(out_fp, index=False)
        print(f"✓ uložené: {out_fp}")


def step_5_calibrate_lightcurve() -> None:
    """Kalibrácia PSF fluxov podľa COMP (median flux_norm / ZP_frame) → CSV + MN Boo plot."""
    print("═══ KROK 5: Kalibrácia svetelných kriviek (COMP ZP) ═══")
    p = _paths()
    out_dir = p["output_psf_dir"]
    summary_fp = out_dir / "psf_summary.csv"
    psf_files = sorted(out_dir.glob("*_psf.csv"))
    if not psf_files:
        print("Žiadne *_psf.csv — krok 5 preskočený.")
        return
    if not summary_fp.is_file():
        print(f"Chýba {summary_fp} — spusti krok 4 alebo vygeneruj súhrn. Krok 5 preskočený.")
        return

    summary = pd.read_csv(summary_fp, low_memory=False)
    summary["catalog_id"] = summary["catalog_id"].map(_cid_key)
    med_by_cid: dict[str, float] = {}
    for _, r in summary.iterrows():
        cid = _cid_key(r.get("catalog_id"))
        if not cid:
            continue
        m = pd.to_numeric(r.get("med_psf_flux"), errors="coerce")
        med_by_cid[cid] = float(m) if pd.notna(m) else float("nan")

    comp_ids = [_cid_key(x) for x in COMP_CALIB_CATALOG_IDS]
    comp_ids = [c for c in comp_ids if c]
    for c in comp_ids:
        if c not in med_by_cid or not math.isfinite(med_by_cid[c]) or med_by_cid[c] <= 0:
            print(f"  Upozornenie: COMP {c} nemá platné med_psf_flux v súhrne.")

    var_catalog_ids: set[str] = set()
    name_by_cid: dict[str, str] = {}
    # frame_stem, bjd_tdb_mid, var catalog_id, psf_flux_raw, n_comp_used, zp_frame, psf_chi2, psf_fit_ok
    frame_rows: list[tuple[str, float, str, float, int, float, float, bool]] = []

    n_frames_total = len(psf_files)

    for fp in psf_files:
        try:
            df = pd.read_csv(fp, low_memory=False)
        except Exception:  # noqa: BLE001
            continue
        if df.empty:
            continue
        df = df.copy()
        df["catalog_id"] = df["catalog_id"].map(_cid_key)
        df["role"] = df.get("role", "").astype(str).str.upper()
        df["psf_flux"] = pd.to_numeric(df.get("psf_flux"), errors="coerce")

        flux_norms: list[float] = []
        n_comp_used = 0
        for cid in comp_ids:
            med_c = med_by_cid.get(cid, float("nan"))
            if not math.isfinite(med_c) or med_c <= 0:
                continue
            sub = df[df["catalog_id"] == cid]
            if sub.empty:
                continue
            fl = float(sub.iloc[0]["psf_flux"])
            if not math.isfinite(fl) or fl <= 0:
                continue
            flux_norms.append(fl / med_c)
            n_comp_used += 1

        if len(flux_norms) < 2:
            zp = float("nan")
        else:
            zp = float(np.nanmedian(flux_norms))

        for _, row in df.iterrows():
            cid = _cid_key(row.get("catalog_id"))
            if not cid:
                continue
            role = str(row.get("role", "")).strip().upper()
            if role == "VAR":
                var_catalog_ids.add(cid)
                nm = str(row.get("name", "")).strip()
                if nm and cid not in name_by_cid:
                    name_by_cid[cid] = nm
            if role != "VAR" or not math.isfinite(zp) or zp <= 0:
                continue
            raw_fl = float(row["psf_flux"]) if pd.notna(row.get("psf_flux")) else float("nan")
            chi2 = float(pd.to_numeric(row.get("psf_chi2"), errors="coerce"))
            if not math.isfinite(chi2):
                chi2 = float("nan")
            fit_ok = bool(row.get("psf_fit_ok", False)) if pd.notna(row.get("psf_fit_ok")) else False
            try:
                bjd = float(pd.to_numeric(row.get("bjd_tdb_mid"), errors="coerce"))
            except Exception:  # noqa: BLE001
                bjd = float("nan")
            if not math.isfinite(bjd):
                bjd = float("nan")
            stem = str(row.get("frame_stem", fp.stem.replace("_psf", ""))).strip()
            frame_rows.append((stem, bjd, cid, raw_fl, n_comp_used, zp, chi2, fit_ok))

    if not var_catalog_ids:
        print("V *_psf.csv nie sú žiadne VAR riadky — krok 5 končí.")
        return

    mn_cid = _cid_key("1591057651117374976")

    def _mag_cal(fcal: float) -> float:
        if not math.isfinite(fcal) or fcal <= 0:
            return float("nan")
        return -2.5 * math.log10(fcal) + float(PSF_CAL_MAG_ZP_OFFSET)

    if DRY_RUN:
        print("[DRY_RUN] neukladám lightcurves/*.csv ani *_psf_cal.png")

    if not DRY_RUN:
        lc_dir = out_dir / "lightcurves"
        lc_dir.mkdir(parents=True, exist_ok=True)

    for cid in sorted(var_catalog_ids):
        rows = [t for t in frame_rows if t[2] == cid]
        if not rows:
            continue
        out_lc: list[dict[str, Any]] = []
        for stem, bjd, _cid2, raw_fl, n_comp, zp, chi2, fit_ok in rows:
            if not math.isfinite(zp) or zp <= 0:
                continue
            if math.isfinite(raw_fl) and raw_fl > 0:
                fcal = raw_fl / zp
                mag = _mag_cal(fcal)
            else:
                fcal = float("nan")
                mag = float("nan")
            out_lc.append(
                {
                    "frame_stem": stem,
                    "bjd_tdb_mid": bjd,
                    "psf_flux_raw": raw_fl,
                    "psf_flux_cal": fcal,
                    "psf_mag_cal": mag,
                    "zp_frame": zp,
                    "n_comp_used": int(n_comp),
                    "psf_chi2": chi2,
                    "psf_fit_ok": fit_ok,
                }
            )
        n_cal = len(out_lc)
        name = name_by_cid.get(cid, cid)
        zps = [float(r["zp_frame"]) for r in out_lc if math.isfinite(float(r["zp_frame"]))]
        zp_min = float(np.nanmin(zps)) if zps else float("nan")
        zp_max = float(np.nanmax(zps)) if zps else float("nan")
        zp_lo = f"{zp_min:.4f}" if math.isfinite(zp_min) else "nan"
        zp_hi = f"{zp_max:.4f}" if math.isfinite(zp_max) else "nan"
        mags = np.array([float(r["psf_mag_cal"]) for r in out_lc], dtype=np.float64)
        mags = mags[np.isfinite(mags)]
        rms = float(np.std(mags, ddof=1)) if mags.size > 1 else float("nan")
        rms_s = f"{rms:.4f}" if math.isfinite(rms) else "nan"

        print(f"VAR hviezda {name}: {n_cal}/{n_frames_total} framov kalibrovaných")
        print(f"  ZP rozsah: {zp_lo} - {zp_hi} (ideálne blízko 1.0)")
        print(f"  psf_mag_cal RMS: {rms_s} mag")

        if not DRY_RUN:
            out_df = pd.DataFrame(out_lc)
            out_df = out_df.sort_values(["bjd_tdb_mid", "frame_stem"], na_position="last")
            cal_fp = lc_dir / f"{cid}_psf_cal.csv"
            out_df.to_csv(cal_fp, index=False)
            print(f"  ✓ {cal_fp.name}")

    if DRY_RUN or not frame_rows:
        return

    # Plot pre VAR hviezdy (override-friendly).
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        print(f"matplotlib nedostupné ({exc}) — plot preskočený.")
        return

    # Prefer názvy z PSF_TARGET_OVERRIDE, inak z name_by_cid, inak samotné ID.
    override_name_by_cid: dict[str, str] = {}
    try:
        for d in PSF_TARGET_OVERRIDE:
            cid0 = _cid_key(d.get("catalog_id"))
            if cid0:
                override_name_by_cid[cid0] = str(d.get("vsx_name", "") or "").strip()
    except Exception:  # noqa: BLE001
        override_name_by_cid = {}

    for cid in sorted(var_catalog_ids):
        fp = lc_dir / f"{cid}_psf_cal.csv"
        if not fp.is_file():
            continue
        try:
            df = pd.read_csv(fp, low_memory=False)
        except Exception:  # noqa: BLE001
            continue
        if df.empty:
            continue
        bjd = pd.to_numeric(df.get("bjd_tdb_mid"), errors="coerce").to_numpy(dtype=float)
        mag = pd.to_numeric(df.get("psf_mag_cal"), errors="coerce").to_numpy(dtype=float)
        okm = np.isfinite(bjd) & np.isfinite(mag)
        if not bool(np.any(okm)):
            continue

        nm = override_name_by_cid.get(cid) or name_by_cid.get(cid) or cid
        nm_file = str(nm).strip().replace(" ", "_").replace("/", "_").replace(":", "_")

        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(bjd[okm], mag[okm], "k.", ms=3, alpha=0.7)
        ax.set_xlabel("BJD_TDB (mid)")
        ax.set_ylabel("psf_mag_cal")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{nm} — PSF kalibrovaná krivka")

        plot_fp = lc_dir / f"{nm_file}_psf_cal.png"
        fig.savefig(plot_fp, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"✓ plot: {plot_fp}")


def main() -> int:
    _force_utf8_stdout()
    parser = argparse.ArgumentParser(description="VYVAR PSF Runner")
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Preskočí budovanie ePSF (použije existujúci)",
    )
    parser.add_argument(
        "--only-build",
        action="store_true",
        help="Len postaví ePSF model, bez fotometrie",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Spracuj len prvých N framov (pre rýchly test)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Nič nezapisuj, len vypíš čo by sa stalo",
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

    p = _paths()
    if not args.skip_build:
        epsf_path = step_1_build_epsf()
    else:
        epsf_path = p["epsf_fits"]
        print(f"[--skip-build] Používam existujúci ePSF: {epsf_path}")

    if args.only_build:
        print("[--only-build] Koniec.")
        return 0

    var_df, comp_df = step_2_load_targets()

    # Special mode: --frames 0 means "skip PSF fitting, run only calibration"
    # (useful when *_psf.csv + psf_summary.csv already exist and you only want step 5 outputs).
    if args.frames is not None and int(args.frames) == 0:
        print("[--frames 0] Preskakujem krok 3+4, spúšťam len krok 5 (kalibrácia).")
        step_5_calibrate_lightcurve()
        return 0

    if args.frames:
        print(f"[--frames {args.frames}] Obmedzujem na prvých {args.frames} framov")

    step_3_run_psf_on_frames(var_df, comp_df, epsf_path, max_frames=args.frames)
    step_4_build_summary()
    step_5_calibrate_lightcurve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

