"""Photometry core — zlúčený modul (photometry + photometry_phase2a)."""
from __future__ import annotations

import copy
import json
import logging
import math
import os
import random
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import AbstractSet, Any, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from astropy.io import fits as astrofits

from comp_pool_rms import attach_comp_rms_to_pool_rows, compute_global_pool_rms_map
from proc_frame_store import (
    PROC_CSV_GLOB,
    PROC_STORE_COLS,
    ProcFrameStore,
    is_masterstar_proc_name,
    proc_csv_path_for_aligned_fits,
)
from config import (
    AppConfig,
    DENSITY_OVERRIDES,
    apply_crowding_overrides,
    apply_density_overrides,
    classify_field_density,
    compute_field_density,
    resolve_comp_sparse_fallback_enabled,
)
from database import query_local_gaia, query_local_gaia_by_source_ids
from gaia_catalog_id import (
    GAIA_PROC_CSV_READ_DTYPE,
    masterstar_row_gaia_key,
    normalize_gaia_source_id,
    read_vyvar_csv,
)
from infolog import log_event

from catalog_match_trust import is_wcs_untrusted_catalog_match_mode, normalize_catalog_match_mode
from jd_axis_format import jd_axis_title, jd_series_relative
from utils import iter_fits_paths_recursive as _iter_fits_recursive

LOGGER = logging.getLogger(__name__)

_MAD_CONSISTENCY = 0.6745  # normalizačný faktor MAD → σ ekvivalent

# Explicit annulus sky (ADU/px) for Howell err; ``noise_floor_adu`` remains detection-floor legacy.
SKY_ADU_PER_PX_ANNULUS_COL = "sky_adu_per_px_annulus"

# F-BINGAIN-1: empirical background noise (empty-aperture scatter) + provenance.
SIGMA_BKG_AP_COL = "sigma_bkg_ap"
ERR_BKG_SOURCE_COL = "err_bkg_source"
ERR_BKG_MODE_EMPIRICAL = "empirical"
ERR_BKG_MODE_HOWELL = "howell"
ERR_BKG_SOURCE_EMPIRICAL = "empirical"
ERR_BKG_SOURCE_HOWELL_FALLBACK = "howell_fallback"
ERR_BKG_SOURCE_HOWELL_SCALED = "howell_scaled"
BKG_SCALE_R_CLAMP_LO = 0.05
BKG_SCALE_R_CLAMP_HI = 2.0

# Per-target LC time provenance (F-BJD-1): labels BJD recompute path, does not alter time values.
TIME_BASE_COL = "time_base"
TIME_BASE_BJD_TDB = "BJD_TDB"
TIME_BASE_JD_FALLBACK = "JD_FALLBACK"


def _safe_polyfit(
    x: np.ndarray,
    y: np.ndarray,
    deg: int,
    *,
    cov: bool = False,
) -> np.ndarray | tuple[np.ndarray, Any] | None:
    """``np.polyfit`` that returns ``None`` when the fit is underdetermined or degenerate."""
    if deg < 0:
        return None
    x_a = np.asarray(x, dtype=np.float64)
    y_a = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x_a) & np.isfinite(y_a)
    x_a = x_a[ok]
    y_a = y_a[ok]
    if x_a.size < deg + 1 or y_a.size < deg + 1:
        return None
    if float(np.ptp(x_a)) == 0.0:
        return None
    try:
        if cov:
            return np.polyfit(x_a, y_a, int(deg), cov=True)
        return np.polyfit(x_a, y_a, int(deg))
    except Exception:  # noqa: BLE001
        # EXC-0120: T4 -- Polyfit failure returns None - callers skip that detrend model branch (EXCEPT-BULK 2026-07-08)
        return None


# Comp tier: Gaia BP-RP outside this band → unreliable vs field comps (use B-V fallback).
_BPRP_VALID_MIN = 0.1
_BPRP_VALID_MAX = 3.5

# Gaia ID (`catalog_id`, VSX / masterstars `name`) musí byť str — float64 stráca cifry
_GAIA_ID_DTYPE: dict[str, type] = dict(GAIA_PROC_CSV_READ_DTYPE)




def _sid_int(v: Any) -> int | None:
    sid = normalize_gaia_source_id(v)
    if sid and sid.isdigit():
        try:
            return int(sid)
        except Exception:  # noqa: BLE001
            # EXC-0121: T4 -- Non-integer Gaia source_id returns None - downstream treats star as uncatalogued (EXCEPT-BULK 2026-07-08)
            return None
    return None


_COMP_QUALITY_JSON_META_KEYS = frozenset(
    {
        "selected_tier",
        "tier4_warning",
        "n_tier1",
        "n_tier2",
        "n_tier3",
        "n_tier4",
        "aperture_correction",
    }
)


def parse_comp_quality_json_map(raw: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Return ``catalog_id`` → ``{quality, note}`` from ``comp_quality_*.json`` (strip metadata keys).

    Accepts legacy flat strings (``"good"``) and structured objects
    (``{"quality": "suspect", "note": "..."}``).
    """
    out: dict[str, dict[str, str]] = {}
    if not isinstance(raw, dict):
        return out
    for k, v in raw.items():
        ks = str(k).strip()
        if ks in _COMP_QUALITY_JSON_META_KEYS:
            continue
        if isinstance(v, str) and v.strip().lower() in ("good", "suspect", "excluded"):
            out[ks] = {"quality": v.strip().lower(), "note": ""}
        elif isinstance(v, dict):
            q = str(v.get("quality", "") or "").strip().lower()
            if q in ("good", "suspect", "excluded"):
                out[ks] = {"quality": q, "note": str(v.get("note", "") or "").strip()}
    return out


def comp_quality_quality_strings(
    qmap: dict[str, dict[str, str]] | dict[str, str] | None,
) -> dict[str, str]:
    """Flatten parsed comp-quality map to ``catalog_id`` → quality string (for w_rel / export helpers)."""
    if not qmap:
        return {}
    out: dict[str, str] = {}
    for k, v in qmap.items():
        if isinstance(v, dict):
            out[str(k)] = str(v.get("quality", "") or "").strip().lower()
        else:
            out[str(k)] = str(v).strip().lower()
    return out


def apply_comp_w_rel_for_display(
    comp_df: pd.DataFrame,
    quality_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Add ``w_rel`` = comp_weight / max(comp_weight) over **non-excluded** comps only.

    Phase-2A ``excluded`` stars remain in the table for transparency but get ``w_rel=0``.
    """
    df = comp_df.copy()
    if df.empty or "comp_weight" not in df.columns:
        return df
    w = pd.to_numeric(df["comp_weight"], errors="coerce")
    excluded = pd.Series(False, index=df.index)
    qmap = comp_quality_quality_strings(quality_map) if quality_map else {}
    if qmap and "catalog_id" in df.columns:
        for i, row in df.iterrows():
            cid = normalize_gaia_source_id(row.get("catalog_id"))
            if cid and str(qmap.get(cid, "")).strip().lower() == "excluded":
                excluded.loc[i] = True
    w_use = w.mask(excluded)
    w_max = float(w_use.max()) if w_use.notna().any() else float("nan")
    if math.isfinite(w_max) and w_max > 0:
        df["w_rel"] = (w / w_max).round(3)
    else:
        df["w_rel"] = float("nan")
    df.loc[excluded, "w_rel"] = 0.0
    return df


def _enrich_comp_bp_rp(
    candidates: pd.DataFrame,
    gaia_db_path: str | None,
    *,
    gaia_prefetch: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Doplní ``bp_rp`` pre comp hviezdy kde chýba (Gaia DR3 podľa ``source_id``)."""
    if candidates is None or getattr(candidates, "empty", True):
        return pd.DataFrame()

    df = candidates.copy()
    if "bp_rp" not in df.columns:
        df["bp_rp"] = float("nan")

    df["bp_rp"] = pd.to_numeric(df.get("bp_rp"), errors="coerce")
    if "ra_deg" in df.columns:
        df["ra_deg"] = pd.to_numeric(df.get("ra_deg"), errors="coerce")
    if "dec_deg" in df.columns:
        df["dec_deg"] = pd.to_numeric(df.get("dec_deg"), errors="coerce")

    con = None
    gaia_cols: set[str] = set()
    gaia_path = str(gaia_db_path or "").strip()
    if gaia_path and os.path.exists(gaia_path):
        try:
            import sqlite3  # noqa: PLC0415

            con = sqlite3.connect(gaia_path)
            con.row_factory = sqlite3.Row
            gaia_cols = {
                str(r[1]).strip().lower() for r in con.execute("PRAGMA table_info('gaia_dr3')").fetchall()
            }
        except Exception:  # noqa: BLE001
            con = None
            gaia_cols = set()

    sel_bp = "bp_rp" in gaia_cols
    gaia_bp_cache: dict[int, float] = {}

    def _gaia_bp_rp(sid_i: int) -> float:
        if sid_i in gaia_bp_cache:
            return gaia_bp_cache[sid_i]
        bp_r = float("nan")
        gid_pf = normalize_gaia_source_id(sid_i)
        if gaia_prefetch and gid_pf and gid_pf in gaia_prefetch:
            try:
                vbp = gaia_prefetch[gid_pf].get("bp_rp")
                if vbp is not None and math.isfinite(float(vbp)):
                    bp_r = float(vbp)
            except (TypeError, ValueError):
                pass
            gaia_bp_cache[int(sid_i)] = bp_r
            return bp_r
        if con is not None and sel_bp:
            try:
                rw = con.execute(
                    "SELECT bp_rp FROM gaia_dr3 WHERE source_id=? LIMIT 1;",
                    (int(sid_i),),
                ).fetchone()
                if rw is not None and rw["bp_rp"] is not None:
                    bp_r = float(rw["bp_rp"])
            except Exception as exc:  # noqa: BLE001
                logging.error('[EXC-0122] Gaia DB bp_rp row fetch fails - comp star keeps NaN bp_rp and wrong colour tier: %s', exc)
                pass
        gaia_bp_cache[int(sid_i)] = bp_r if math.isfinite(bp_r) else float("nan")
        return gaia_bp_cache[int(sid_i)]

    try:
        for idx, row in df.iterrows():
            try:
                bp_now = float(pd.to_numeric(row.get("bp_rp"), errors="coerce"))
            except Exception:  # noqa: BLE001
                bp_now = float("nan")
            if math.isfinite(bp_now):
                continue
            sid_i = _sid_int(row.get("source_id") or row.get("catalog_id") or row.get("name"))
            if sid_i is None:
                continue
            gaia_bp = _gaia_bp_rp(sid_i)
            if math.isfinite(gaia_bp):
                df.at[idx, "bp_rp"] = float(gaia_bp)
    finally:
        try:
            if con is not None:
                con.close()
        except Exception:  # noqa: BLE001
            # EXC-0123: T2 -- sqlite con.close() failure during comp bp_rp enrichment ignored (EXCEPT-BULK-2 2026-07-08)
            pass

    return df

# ---------------------------------------------------------------------------
# Pomocné funkcie
# ---------------------------------------------------------------------------


def _normalize_gaia_id(x: Any) -> str:
    """Gaia ``source_id`` key for joins; delegates to ``normalize_gaia_source_id`` (legacy bool/``none`` guards)."""
    if isinstance(x, (bool, np.bool_)):
        return ""
    out = normalize_gaia_source_id(x)
    if out.lower() == "none":
        return ""
    return out


def _build_csv_lookup(
    csv_df: pd.DataFrame,
    id_col: str,
) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    """Vytvorí dva lookup mechanizmy:
    1. Primárny: dict {normalized_id → row}
    2. Záložný: riadky s numerickými x,y pre nearest-neighbor match (plné stĺpce CSV).

    Proc CSV z pipeline má ``catalog_id`` často ako float / vedeckú notáciu (strata presnosti),
    zatiaľ čo ``name`` obsahuje presný Gaia ``source_id`` — indexujeme oboje (``setdefault``),
    aby Fáza 2A netrafila NN na suseda namiesto správnej porovnávačky.
    """
    id_map: dict[str, pd.Series] = {}
    _id_series = csv_df[id_col]
    for i in range(len(csv_df)):
        row = csv_df.iloc[i]
        keys: set[str] = set()
        pk = masterstar_row_gaia_key(row)
        if pk:
            keys.add(pk)
        if "name" in csv_df.columns:
            nk = normalize_gaia_source_id(row.get("name"))
            if nk and re.fullmatch(r"\d{12,22}", nk):
                keys.add(nk)
        cid = _normalize_gaia_id(_id_series.iloc[i])
        if cid:
            keys.add(cid)
        for k in keys:
            id_map.setdefault(k, row)
    # Plná kópia: NN fallback musí vrátiť Series so všetkými stĺpcami (dao_flux, časy, …).
    xy_df = csv_df.copy()
    if "name" in xy_df.columns:
        _nk = xy_df["name"].map(normalize_gaia_source_id)
        _is_gaia_name = _nk.map(lambda s: bool(s and re.fullmatch(r"\d{12,22}", s)))
        _cid_from_col = xy_df[id_col].map(_normalize_gaia_id)
        xy_df["_cid_norm"] = _nk.where(_is_gaia_name, _cid_from_col)
    else:
        xy_df["_cid_norm"] = xy_df[id_col].apply(_normalize_gaia_id)
    xy_df["x"] = pd.to_numeric(xy_df["x"], errors="coerce")
    xy_df["y"] = pd.to_numeric(xy_df["y"], errors="coerce")
    return id_map, xy_df.dropna(subset=["x", "y"])


def _lookup_star_in_csv(
    cid: str,
    id_map: dict[str, pd.Series],
    xy_df: pd.DataFrame,
    ref_x: float | None,
    ref_y: float | None,
    *,
    xy_tol_px: float = 15.0,
) -> pd.Series | None:
    """Hľadaj hviezdu v CSV — primárne cez ID, fallback cez x,y."""
    cid_key = _normalize_gaia_id(cid)
    if cid_key and cid_key in id_map:
        return id_map[cid_key]

    if ref_x is None or ref_y is None or xy_df.empty:
        return None
    if not (math.isfinite(ref_x) and math.isfinite(ref_y)):
        return None

    dx = xy_df["x"].to_numpy(dtype=np.float64) - float(ref_x)
    dy = xy_df["y"].to_numpy(dtype=np.float64) - float(ref_y)
    dists = np.sqrt(dx * dx + dy * dy)
    tol = float(xy_tol_px)
    if "dao_flux" in xy_df.columns:
        flux_arr = pd.to_numeric(xy_df["dao_flux"], errors="coerce").to_numpy(dtype=np.float64)
        candidate = np.isfinite(dists) & (dists <= tol) & np.isfinite(flux_arr) & (flux_arr > 0.0)
    else:
        candidate = np.isfinite(dists) & (dists <= tol)
    if not candidate.any():
        return None
    dist_masked = np.where(candidate, dists, np.inf)
    j = int(np.argmin(dist_masked))
    if not math.isfinite(float(dists[j])) or float(dists[j]) > tol:
        return None
    _hit = xy_df.iloc[j]
    _mid = str(_hit.get("_cid_norm", ""))
    logging.debug(
        "[FÁZA 2A] CSV NN fallback ok: requested_cid=%s matched_csv_id=%s dist_px=%.2f tol=%.1f",
        cid,
        _mid,
        float(dists[j]),
        tol,
    )
    return _hit


def _sat_limit_peak_adu(cfg: AppConfig | None = None) -> float | None:
    """Hranica peak_max_adu z configu (voliteľné). Bez globálneho fallbacku — saturácia z FITS/DB v pipeline."""
    _ = cfg
    return None


def _mad_sigma(arr: np.ndarray) -> float:
    """Robustný σ estimátor cez MAD / 0.6745."""
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    if not math.isfinite(mad) or mad <= 0:
        return float(np.std(arr)) / _MAD_CONSISTENCY or 1e-9
    return mad / _MAD_CONSISTENCY


def _aperture_to_mask_single(ap: Any) -> Any:
    """photutils môže vrátiť jednu masku alebo zoznam (jedna pozícia → prvý prvok)."""
    m = ap.to_mask(method="center")
    if isinstance(m, (list, tuple)):
        return m[0]
    return m


def measure_fwhm_from_masterstar(
    masterstar_fits_path: Path,
    star_positions: pd.DataFrame,
    *,
    n_stars: int = 20,
    fit_box_fwhm: float = 8.0,
    dao_fwhm_hint: float = 3.5,
    ms_data: np.ndarray | None = None,
) -> float:
    """Zmeria skutočné Gaussian FWHM z MASTERSTAR FITS.

    Fituje 2D Gaussian na izolované, nesaturované hviezdy z ``star_positions``
    a vracia mediánové FWHM v pixeloch. Toto je fyzikálne správne FWHM
    (zodpovedá AIJ/IRAF definícii), na rozdiel od DAO odhadu ktorý
    systematicky preceňuje FWHM.

    Args:
        masterstar_fits_path: Cesta k MASTERSTAR.fits
        star_positions: DataFrame so stĺpcami x, y, mag (catalog_id voliteľný)
        n_stars: Počet hviezd na fit (vyberie izolované, stredne jasné)
        fit_box_fwhm: Veľkosť okna pre fit v jednotkách dao_fwhm_hint
        dao_fwhm_hint: Hrubý DAO odhad pre určenie veľkosti okna

    Returns:
        Mediánové Gaussian FWHM v pixeloch.
    """
    from astropy.modeling import fitting, models

    if ms_data is not None:
        data = np.asarray(ms_data, dtype=np.float64)
    else:
        with astrofits.open(masterstar_fits_path, memmap=False) as hdul:
            data = np.asarray(hdul[0].data, dtype=np.float64)
    fill = float(np.nanmedian(data))
    data = np.where(np.isfinite(data), data, fill)
    h, w = data.shape

    df = star_positions.copy()
    if df.empty:
        logging.warning(
            "[FÁZA 2A] Gaussian FWHM fit: prázdne star_positions, fallback dao_fwhm_hint=%.2f px",
            float(dao_fwhm_hint),
        )
        return float(dao_fwhm_hint)
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["x", "y"])
    if len(df) < 3:
        logging.warning(
            "[FÁZA 2A] Gaussian FWHM fit: málo riadkov s x,y, fallback dao_fwhm_hint=%.2f px",
            float(dao_fwhm_hint),
        )
        return float(dao_fwhm_hint)

    box = int(fit_box_fwhm * dao_fwhm_hint)
    margin = box + 5
    if box < 3 or margin * 2 >= min(h, w):
        logging.warning(
            "[FÁZA 2A] Gaussian FWHM fit: príliš malý/obrovský box=%s, fallback dao_fwhm_hint=%.2f px",
            box,
            float(dao_fwhm_hint),
        )
        return float(dao_fwhm_hint)

    df = df[(df["x"] > margin) & (df["x"] < w - margin) & (df["y"] > margin) & (df["y"] < h - margin)].copy()
    if len(df) < 3:
        logging.warning(
            "[FÁZA 2A] Gaussian FWHM fit: málo hviezd po okrajovom filtri, fallback dao_fwhm_hint=%.2f px",
            float(dao_fwhm_hint),
        )
        return float(dao_fwhm_hint)

    if "mag" in df.columns:
        df["_mag"] = pd.to_numeric(df["mag"], errors="coerce")
        df = df.dropna(subset=["_mag"]).sort_values("_mag")
        n_skip = max(1, len(df) // 10)
        df = df.iloc[n_skip : n_skip + n_stars * 3]
    else:
        df = df.iloc[: n_stars * 3]

    if len(df) < 1:
        logging.warning(
            "[FÁZA 2A] Gaussian FWHM fit: prázdny výber po mag, fallback dao_fwhm_hint=%.2f px",
            float(dao_fwhm_hint),
        )
        return float(dao_fwhm_hint)

    xs = df["x"].to_numpy(dtype=float)
    ys = df["y"].to_numpy(dtype=float)
    isolated: list[Any] = []
    for i in range(len(xs)):
        dists = np.sqrt((xs - xs[i]) ** 2 + (ys - ys[i]) ** 2)
        dists[i] = np.inf
        if float(np.min(dists)) > 2.0 * box:
            isolated.append(df.index[i])
        if len(isolated) >= n_stars:
            break

    if not isolated:
        isolated = list(df.index[: min(n_stars, len(df))])

    df = df.loc[isolated]

    fitter = fitting.LevMarLSQFitter()
    fwhm_values: list[float] = []

    y_grid, x_grid = np.mgrid[0 : 2 * box + 1, 0 : 2 * box + 1]

    for _, row in df.iterrows():
        try:
            xc = int(round(float(row["x"])))
            yc = int(round(float(row["y"])))
            if not (box <= xc < w - box and box <= yc < h - box):
                continue

            cutout = data[yc - box : yc + box + 1, xc - box : xc + box + 1].copy()
            if cutout.shape != (2 * box + 1, 2 * box + 1):
                continue

            border = np.concatenate(
                [cutout[0, :], cutout[-1, :], cutout[1:-1, 0], cutout[1:-1, -1]]
            )
            sky = float(np.median(border))
            cutout -= sky
            peak = float(np.max(cutout))
            if peak <= 0:
                continue

            g_init = models.Gaussian2D(
                amplitude=peak,
                x_mean=float(box),
                y_mean=float(box),
                x_stddev=dao_fwhm_hint / 2.355,
                y_stddev=dao_fwhm_hint / 2.355,
            )
            g_fit = fitter(g_init, x_grid, y_grid, cutout)

            sx = abs(float(getattr(g_fit.x_stddev, "value", g_fit.x_stddev)))
            sy = abs(float(getattr(g_fit.y_stddev, "value", g_fit.y_stddev)))
            fwhm_x = 2.355 * sx
            fwhm_y = 2.355 * sy

            if (
                0.5 * dao_fwhm_hint < fwhm_x < 4.0 * dao_fwhm_hint
                and 0.5 * dao_fwhm_hint < fwhm_y < 4.0 * dao_fwhm_hint
            ):
                fwhm_values.append((fwhm_x + fwhm_y) / 2.0)

        except Exception:  # noqa: BLE001
            # EXC-0124: T4 -- One star skipped in Gaussian FWHM loop - median still computed from remaining stars (EXCEPT-BULK-2 2026-07-08)
            continue

    if len(fwhm_values) < 3:
        logging.warning(
            "[FÁZA 2A] Gaussian FWHM fit: len menej ako 3 hviezd (%s), fallback dao_fwhm_hint=%.2f px",
            len(fwhm_values),
            float(dao_fwhm_hint),
        )
        return float(dao_fwhm_hint)

    result = float(np.median(fwhm_values))
    logging.info(
        "[FÁZA 2A] Gaussian FWHM z MASTERSTAR: %.3f px (z %s hviezd, DAO hint %.3f px)",
        result,
        len(fwhm_values),
        float(dao_fwhm_hint),
    )
    return result


# ---------------------------------------------------------------------------
# KROK 1: Globálna fixná apertura z PSF FWHM (MASTERSTAR VY_FWHM alebo fit)
# ---------------------------------------------------------------------------


def compute_optimal_apertures(
    masterstar_fits_path: Path,
    star_positions: pd.DataFrame,
    fwhm_px: float,
    *,
    aperture_fwhm_factor: float = 1.75,
    annulus_inner_fwhm: float = 4.5,
    annulus_outer_fwhm: float = 6.0,
) -> dict[str, float]:
    """Globálna fixná apertura = aperture_fwhm_factor × FWHM.

    Fyzikálne zdôvodnenie:
    - PSF FWHM (typicky ``VY_FWHM`` DAO z MASTERSTAR): r ≈ 1.75× FWHM zachytí väčšinu fluxu
    - Konzistentná fixná apertura je robustnejšia ako per-hviezda
      metódy v hustom poli (kontaminácia susedmi)
    - Zodpovedá AIJ metodike: fixná apertura z FWHM

    Args:
        masterstar_fits_path: Nepoužíva sa — zachované pre kompatibilitu.
        star_positions: DataFrame so stĺpcami catalog_id (voliteľne name).
        fwhm_px: FWHM v pixeloch (Fáza 2A: ``VY_FWHM`` z hlavičky alebo Gaussian fit).
        aperture_fwhm_factor: Násobok FWHM. Default 1.75.
        annulus_inner_fwhm: Zachované pre kompatibilitu signatúry.
        annulus_outer_fwhm: Zachované pre kompatibilitu signatúry.

    Returns:
        dict {catalog_id: apertura_px} — všetky hviezdy majú rovnakú hodnotu.
    """
    _ = masterstar_fits_path
    _ = annulus_inner_fwhm
    _ = annulus_outer_fwhm

    global_ap = float(aperture_fwhm_factor * fwhm_px)

    logging.info(
        f"[FÁZA 2A] Globálna apertura: {global_ap:.3f}px "
        f"({aperture_fwhm_factor:.2f}× FWHM={fwhm_px:.3f}px)"
    )

    result: dict[str, float] = {}
    for _, row in star_positions.iterrows():
        cid = _normalize_gaia_id(row.get("catalog_id", row.get("name", "")))
        if cid:
            result[cid] = global_ap

    return result


# ---------------------------------------------------------------------------
# KROK 2: Aperturná fotometria per snímka — mediánový sky
# ---------------------------------------------------------------------------


def _flux_to_mag(flux: float) -> float:
    """Inštrumentálna magnitúda z flux."""
    if not math.isfinite(flux) or flux <= 0:
        return float("nan")
    return -2.5 * math.log10(flux)


def _clamp_err_empty_apertures_n(n: int) -> int:
    """Clamp ``err_empty_apertures_n`` to registry range 16..256."""
    try:
        v = int(n)
    except (TypeError, ValueError):
        v = 64
    return max(16, min(256, v))


def _clamp_err_empty_apertures_min(n: int) -> int:
    try:
        v = int(n)
    except (TypeError, ValueError):
        v = 16
    return max(1, min(256, v))


def _normalize_err_background_mode(mode: str | None) -> str:
    m = str(mode or ERR_BKG_MODE_EMPIRICAL).strip().lower()
    if m in (ERR_BKG_MODE_HOWELL, "legacy"):
        return ERR_BKG_MODE_HOWELL
    return ERR_BKG_MODE_EMPIRICAL


def _robust_scatter_mad(values: np.ndarray) -> float:
    """Sigma-clipped MAD scatter (Labbe et al. 2003 empty-aperture convention)."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 3:
        return float("nan")
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    if mad <= 0:
        return 0.0
    return mad / _MAD_CONSISTENCY


def _build_star_exclusion_mask(
    shape: tuple[int, ...],
    star_x: np.ndarray,
    star_y: np.ndarray,
    exclusion_radius_px: float,
    edge_margin_px: float,
) -> np.ndarray:
    """Boolean mask: True where empty apertures must not be placed."""
    ny, nx = int(shape[0]), int(shape[1])
    blocked = np.zeros((ny, nx), dtype=bool)
    em = int(math.ceil(max(0.0, float(edge_margin_px))))
    if em > 0:
        blocked[:em, :] = True
        blocked[-em:, :] = True
        blocked[:, :em] = True
        blocked[:, -em:] = True
    ex_r = float(exclusion_radius_px)
    if ex_r <= 0:
        return blocked
    ex_r2 = ex_r * ex_r
    xs = np.asarray(star_x, dtype=np.float64)
    ys = np.asarray(star_y, dtype=np.float64)
    ok = np.isfinite(xs) & np.isfinite(ys)
    # Canonical order: mask is OR-commutative, but keep draw/debug paths order-stable.
    order = np.lexsort((xs[ok], ys[ok]))
    xs_s = xs[ok][order]
    ys_s = ys[ok][order]
    for xi, yi in zip(xs_s, ys_s):
        x0 = max(0, int(math.floor(float(xi) - ex_r)) - 1)
        x1 = min(nx, int(math.ceil(float(xi) + ex_r)) + 2)
        y0 = max(0, int(math.floor(float(yi) - ex_r)) - 1)
        y1 = min(ny, int(math.ceil(float(yi) + ex_r)) + 2)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        dist2 = (xx - float(xi)) ** 2 + (yy - float(yi)) ** 2
        blocked[y0:y1, x0:x1] |= dist2 <= ex_r2
    return blocked


def _canonicalize_star_xy(
    star_x: np.ndarray,
    star_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Sort finite (x,y) pairs and return (xs, ys, sha256 hex of canonical list)."""
    import hashlib

    xs = np.asarray(star_x, dtype=np.float64).ravel()
    ys = np.asarray(star_y, dtype=np.float64).ravel()
    n = min(xs.size, ys.size)
    xs, ys = xs[:n], ys[:n]
    ok = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[ok], ys[ok]
    if xs.size == 0:
        return xs, ys, hashlib.sha256(b"").hexdigest()
    order = np.lexsort((xs, ys))
    xs, ys = xs[order], ys[order]
    # Deduplicate exact duplicates after sort (stable membership).
    if xs.size > 1:
        keep = np.empty(xs.size, dtype=bool)
        keep[0] = True
        keep[1:] = (xs[1:] != xs[:-1]) | (ys[1:] != ys[:-1])
        xs, ys = xs[keep], ys[keep]
    blob = np.column_stack([xs, ys]).astype("<f8", copy=False).tobytes()
    return xs, ys, hashlib.sha256(blob).hexdigest()


def _labbe_debug_dump_enabled() -> bool:
    import os

    return str(os.environ.get("VYVAR_LABBE_DEBUG_DUMP", "")).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _labbe_debug_dump_path() -> Path:
    import os

    raw = str(os.environ.get("VYVAR_LABBE_DEBUG_DUMP_PATH", "")).strip()
    if raw:
        return Path(raw)
    return Path("tmp") / "labbe_debug_dump.jsonl"


def _labbe_append_debug_record(record: dict[str, Any]) -> None:
    if not _labbe_debug_dump_enabled():
        return
    path = _labbe_debug_dump_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    except OSError:
        pass


def _labbe_content_seed_from_header(hdr: Any, *, r_ap: float) -> int:
    """Stable Labbe RNG seed from frame identity + aperture radius (F-431 / LABBE-DET)."""
    import hashlib

    def _hget(key: str) -> str:
        try:
            return str(hdr.get(key) or "")
        except Exception:  # noqa: BLE001
            return ""

    parts = [
        _hget("DATE-OBS"),
        _hget("FILENAME"),
        _hget("FRAME"),
        _hget("NAXIS1"),
        _hget("NAXIS2"),
        f"{float(r_ap):.4f}",
    ]
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False) % (2**63 - 1)


def measure_empty_aperture_sigma_bkg(
    data: np.ndarray,
    star_x: np.ndarray,
    star_y: np.ndarray,
    r_ap: float,
    r_in: float,
    r_out: float,
    *,
    n_apertures: int = 64,
    min_valid: int = 16,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    frame_id: str | None = None,
    star_list_source: str = "in_memory",
) -> tuple[float, int, str]:
    """Empirical aperture background noise via random empty apertures (Labbe et al. 2003).

    Each placement uses the same annulus sky subtraction as production science apertures
    (``_annulus_sky_subtracted_flux``). The robust scatter of net sums is ``sigma_bkg_ap``
    [ADU] and already includes background Poisson, read noise, resampling covariance,
    pedestal offsets, and the Merline & Howell (1995) sky-estimation term — do **not**
    add a separate RN or annulus term (would double-count).

    Determinism (LABBE-DET): star list is canonicalized (sorted by x,y); when ``rng`` is
    None a child Generator is derived via ``SeedSequence`` from ``seed`` (and r_ap) so
    draws are independent of call order / shared parent RNG.

    Args:
        seed: When ``rng`` is None, seed the Generator (F-431). Prefer a content-derived
            seed from the caller so same-draft re-photometry is byte-stable.
        frame_id / star_list_source: optional debug-dump metadata
            (``VYVAR_LABBE_DEBUG_DUMP=1``).

    Returns:
        (sigma_bkg_ap, n_valid, reason) — reason non-empty when measurement failed.
    """
    import hashlib

    xs_c, ys_c, star_list_hash = _canonicalize_star_xy(star_x, star_y)
    seed_value = int(seed) if seed is not None else None
    if rng is None:
        if seed_value is None:
            rng = np.random.default_rng(None)
        else:
            # Independent child RNG keyed on content seed + r_ap (order-independent).
            ss = np.random.SeedSequence(
                [int(seed_value), int(round(float(r_ap) * 10000.0)) & 0xFFFFFFFF]
            )
            rng = np.random.default_rng(ss)
    n_target = _clamp_err_empty_apertures_n(n_apertures)
    n_min = _clamp_err_empty_apertures_min(min_valid)
    if not (
        math.isfinite(r_ap)
        and r_ap > 0
        and math.isfinite(r_in)
        and r_in > 0
        and math.isfinite(r_out)
        and r_out > r_in
    ):
        return float("nan"), 0, "invalid_annulus_geometry"

    d = np.asarray(data, dtype=np.float64)
    if d.ndim != 2 or d.size == 0:
        return float("nan"), 0, "empty_image"

    margin_px = max(2.0, float(r_out) - float(r_in))
    excl_r = float(r_out) + margin_px
    edge_margin = float(r_out) + float(r_ap) + 1.0
    blocked = _build_star_exclusion_mask(d.shape, xs_c, ys_c, excl_r, edge_margin)
    mask_hash = hashlib.sha256(np.ascontiguousarray(blocked).view(np.uint8)).hexdigest()
    free_y, free_x = np.nonzero(~blocked)
    labbe_input_hash = hashlib.sha256(
        f"{star_list_hash}|{mask_hash}|{float(r_ap):.4f}|{seed_value}".encode("utf-8")
    ).hexdigest()

    if free_x.size < n_min:
        _labbe_append_debug_record(
            {
                "frame_id": frame_id,
                "r_ap": float(r_ap),
                "seed_value": seed_value,
                "star_list_source": star_list_source,
                "n_stars": int(xs_c.size),
                "star_list_hash": star_list_hash,
                "mask_hash": mask_hash,
                "labbe_input_hash": labbe_input_hash,
                "n_attempted": 0,
                "n_valid_apertures": 0,
                "first5_aperture_xy": [],
                "sigma_result": None,
                "reason": f"crowding: only {int(free_x.size)} candidate pixels (< {n_min})",
            }
        )
        return float("nan"), 0, f"crowding: only {int(free_x.size)} candidate pixels (< {n_min})"

    n_try = min(int(free_x.size), max(n_target * 8, n_target))
    idx = rng.choice(free_x.size, size=n_try, replace=False)
    net_sums: list[float] = []
    first5: list[list[float]] = []
    for j in idx:
        xc = float(free_x[j]) + 0.5
        yc = float(free_y[j]) + 0.5
        flux_net, _, _ = _annulus_sky_subtracted_flux(d, xc, yc, float(r_ap), float(r_in), float(r_out))
        if math.isfinite(flux_net):
            net_sums.append(float(flux_net))
            if len(first5) < 5:
                first5.append([xc, yc])
        if len(net_sums) >= n_target:
            break

    n_valid = len(net_sums)
    if n_valid < n_min:
        _labbe_append_debug_record(
            {
                "frame_id": frame_id,
                "r_ap": float(r_ap),
                "seed_value": seed_value,
                "star_list_source": star_list_source,
                "n_stars": int(xs_c.size),
                "star_list_hash": star_list_hash,
                "mask_hash": mask_hash,
                "labbe_input_hash": labbe_input_hash,
                "n_attempted": int(n_try),
                "n_valid_apertures": int(n_valid),
                "first5_aperture_xy": first5,
                "sigma_result": None,
                "reason": f"crowding: {n_valid} valid empty apertures (< {n_min})",
            }
        )
        return float("nan"), n_valid, f"crowding: {n_valid} valid empty apertures (< {n_min})"
    sigma = _robust_scatter_mad(np.asarray(net_sums, dtype=np.float64))
    if not math.isfinite(sigma) or sigma < 0:
        return float("nan"), n_valid, "non_finite_scatter"
    _labbe_append_debug_record(
        {
            "frame_id": frame_id,
            "r_ap": float(r_ap),
            "seed_value": seed_value,
            "star_list_source": star_list_source,
            "n_stars": int(xs_c.size),
            "star_list_hash": star_list_hash,
            "mask_hash": mask_hash,
            "labbe_input_hash": labbe_input_hash,
            "n_attempted": int(n_try),
            "n_valid_apertures": int(n_valid),
            "first5_aperture_xy": first5,
            "sigma_result": float(sigma),
            "reason": "",
        }
    )
    return float(sigma), n_valid, ""


def estimate_star_free_per_pixel_variance_adu2(
    data: np.ndarray,
    star_x: np.ndarray | None = None,
    star_y: np.ndarray | None = None,
    exclusion_radius_px: float = 12.0,
) -> float | None:
    """Robust per-pixel background variance [ADU²/px] from star-free pixels in one frame."""
    d = np.asarray(data, dtype=np.float64)
    if d.ndim != 2 or d.size == 0:
        return None
    xs = np.asarray(star_x if star_x is not None else [], dtype=np.float64)
    ys = np.asarray(star_y if star_y is not None else [], dtype=np.float64)
    blocked = _build_star_exclusion_mask(
        d.shape,
        xs,
        ys,
        float(exclusion_radius_px),
        float(exclusion_radius_px),
    )
    vals = d[~blocked]
    vals = vals[np.isfinite(vals)]
    if vals.size < 64:
        return None
    med = float(np.median(vals))
    resid = vals - med
    sig = _robust_scatter_mad(resid)
    if not math.isfinite(sig) or sig < 0:
        return None
    return float(sig * sig)


def _howell_variance_adu2(
    flux: float,
    sky_pp: float,
    area: float,
    gain: float = 1.0,
    read_noise: float = 10.0,
) -> float:
    """Total variance [ADU²] from Howell (1989) eq. 2 (legacy level-based background term)."""
    if not math.isfinite(flux) or flux <= 0:
        return float("nan")
    if not math.isfinite(sky_pp) or sky_pp < 0:
        sky_pp = 0.0
    if not math.isfinite(area) or area <= 0:
        return float("nan")
    g = float(gain) if math.isfinite(gain) and gain > 0 else 1.0
    rn = float(read_noise) if math.isfinite(read_noise) and read_noise >= 0 else 10.0
    return flux / g + max(0.0, sky_pp) / g * area + (rn / g) ** 2 * area


def _howell_bkg_variance_adu2(
    sky_pp: float,
    area: float,
    gain: float = 1.0,
    read_noise: float = 10.0,
) -> float:
    """Background + read-noise variance [ADU²] from Howell (1989) eq. 2 (excludes source Poisson F/g)."""
    if not math.isfinite(sky_pp) or sky_pp < 0:
        sky_pp = 0.0
    if not math.isfinite(area) or area <= 0:
        return float("nan")
    g = float(gain) if math.isfinite(gain) and gain > 0 else 1.0
    rn = float(read_noise) if math.isfinite(read_noise) and read_noise >= 0 else 10.0
    return max(0.0, sky_pp) / g * area + (rn / g) ** 2 * area


def _clamp_bkg_scale_r(r: float) -> float:
    if not math.isfinite(r):
        return float("nan")
    return float(max(BKG_SCALE_R_CLAMP_LO, min(BKG_SCALE_R_CLAMP_HI, float(r))))


def bkg_scale_ratio_empirical_over_howell(
    sigma_bkg_ap: float,
    sky_pp: float,
    area: float,
    *,
    gain: float = 1.0,
    read_noise: float = 10.0,
) -> float:
    """Per-measurement r = sigma_bkg_ap² / howell_bkg_variance for hybrid fallback calibration."""
    sig = float(sigma_bkg_ap)
    if not math.isfinite(sig) or sig < 0:
        return float("nan")
    hb = _howell_bkg_variance_adu2(sky_pp, area, gain=gain, read_noise=read_noise)
    if not math.isfinite(hb) or hb <= 0:
        return float("nan")
    return float(sig * sig / hb)


def compute_setup_bkg_scale_r(ratios: list[float]) -> tuple[float, int]:
    """Median empirical/Howell background variance ratio; clamped to [0.05, 2.0]."""
    ok = [float(r) for r in ratios if math.isfinite(float(r)) and float(r) > 0]
    if not ok:
        return float("nan"), 0
    return _clamp_bkg_scale_r(float(np.median(np.asarray(ok, dtype=np.float64)))), len(ok)


def scaled_sigma_bkg_ap_from_howell(
    sky_pp: float,
    area: float,
    *,
    gain: float = 1.0,
    read_noise: float = 10.0,
    r_setup: float,
) -> float:
    """Calibrated fallback: sqrt(r_setup * howell_bkg_variance) [ADU] at aperture scale."""
    r_c = _clamp_bkg_scale_r(float(r_setup))
    hb = _howell_bkg_variance_adu2(sky_pp, area, gain=gain, read_noise=read_noise)
    if not math.isfinite(r_c) or not math.isfinite(hb) or hb < 0:
        return float("nan")
    return float(math.sqrt(r_c * hb))


def finalize_hybrid_bkg_fallback_proc_dir(
    proc_dir: Path,
    *,
    gain: float = 1.0,
    read_noise: float = 10.0,
    setup_label: str = "",
) -> dict[str, Any]:
    """Post-pass: replace raw Howell fallback rows with setup-calibrated ``howell_scaled``.

    ``r_setup`` = median over rows with empirical ``sigma_bkg_ap`` of
    ``sigma_bkg_ap² / (A·sky/g + A·(RN/g)²)``.  Raw ``howell_fallback`` remains only when
    no empirical frames exist in the setup (Casertano et al. 2000 transferred correction).
    """
    from infolog import log_event

    proc_dir = Path(proc_dir)
    ratios: list[float] = []
    files = sorted(proc_dir.glob("proc_*.csv"))
    for proc_path in files:
        try:
            df = pd.read_csv(proc_path, low_memory=False)
        except Exception:  # noqa: BLE001
            continue
        if df.empty:
            continue
        for _, row in df.iterrows():
            src = str(row.get(ERR_BKG_SOURCE_COL, "")).strip()
            if src != ERR_BKG_SOURCE_EMPIRICAL:
                continue
            sig = float(pd.to_numeric(row.get(SIGMA_BKG_AP_COL), errors="coerce"))
            sky = _sky_pp_for_photometric_error(row)
            area = float(pd.to_numeric(row.get("aperture_area_px"), errors="coerce"))
            if not math.isfinite(area) or area <= 0:
                r_ap = float(pd.to_numeric(row.get("aperture_r_px"), errors="coerce"))
                area = math.pi * r_ap * r_ap if math.isfinite(r_ap) and r_ap > 0 else float("nan")
            rv = bkg_scale_ratio_empirical_over_howell(sig, sky, area, gain=gain, read_noise=read_noise)
            if math.isfinite(rv) and rv > 0:
                ratios.append(rv)

    r_setup, n_ratios = compute_setup_bkg_scale_r(ratios)
    stats: dict[str, Any] = {
        "setup": setup_label or str(proc_dir),
        "n_ratio_samples": n_ratios,
        "r_setup": float(r_setup) if math.isfinite(r_setup) else None,
        "n_files": len(files),
        "n_scaled_rows": 0,
        "n_raw_fallback_rows": 0,
    }
    if not math.isfinite(r_setup):
        return stats

    if not hasattr(finalize_hybrid_bkg_fallback_proc_dir, "_logged_setups"):
        finalize_hybrid_bkg_fallback_proc_dir._logged_setups = set()  # type: ignore[attr-defined]
    _logged: set[str] = finalize_hybrid_bkg_fallback_proc_dir._logged_setups  # type: ignore[attr-defined]
    _key = setup_label or str(proc_dir.resolve())
    if _key not in _logged:
        log_event(
            f"[PHOT] err_bkg howell_scaled setup={_key} r_setup={r_setup:.4f} "
            f"(n_empirical_ratios={n_ratios}, clamp=[{BKG_SCALE_R_CLAMP_LO},{BKG_SCALE_R_CLAMP_HI}])"
        )
        _logged.add(_key)

    for proc_path in files:
        try:
            df = pd.read_csv(proc_path, low_memory=False)
        except Exception:  # noqa: BLE001
            continue
        if df.empty or ERR_BKG_SOURCE_COL not in df.columns:
            continue
        changed = False
        src_col = df[ERR_BKG_SOURCE_COL].astype(str)
        for i in range(len(df)):
            if src_col.iloc[i] != ERR_BKG_SOURCE_HOWELL_FALLBACK:
                continue
            row = df.iloc[i]
            sky = _sky_pp_for_photometric_error(row)
            area = float(pd.to_numeric(row.get("aperture_area_px"), errors="coerce"))
            if not math.isfinite(area) or area <= 0:
                r_ap = float(pd.to_numeric(row.get("aperture_r_px"), errors="coerce"))
                area = math.pi * r_ap * r_ap if math.isfinite(r_ap) and r_ap > 0 else float("nan")
            sig_scaled = scaled_sigma_bkg_ap_from_howell(
                sky, area, gain=gain, read_noise=read_noise, r_setup=r_setup
            )
            if math.isfinite(sig_scaled) and sig_scaled >= 0:
                df.at[df.index[i], SIGMA_BKG_AP_COL] = sig_scaled
                df.at[df.index[i], ERR_BKG_SOURCE_COL] = ERR_BKG_SOURCE_HOWELL_SCALED
                stats["n_scaled_rows"] += 1
                changed = True
            else:
                stats["n_raw_fallback_rows"] += 1
        if changed:
            df.to_csv(proc_path, index=False)
    return stats


def _photometric_error(
    flux: float,
    sky_pp: float,
    area: float,
    gain: float = 1.0,
    read_noise: float = 10.0,
) -> float:
    """Relative photometric error from Poisson + read-noise variance (Howell 1989 eq. 2).

    Units at boundary: ``flux`` and ``sky_pp`` in ADU (per px for sky); ``gain`` in e-/ADU;
    ``read_noise`` in e-; internal variance in ADU². Returns dimensionless err/flux.
    Legacy ``howell`` mode — byte-identical to pre F-BINGAIN-1 behaviour.
    """
    variance = _howell_variance_adu2(flux, sky_pp, area, gain=gain, read_noise=read_noise)
    if not math.isfinite(variance) or variance < 0:
        return float("nan")
    if not math.isfinite(flux) or flux <= 0:
        return float("nan")
    return math.sqrt(variance) / flux


def _photometric_error_with_bkg_mode(
    flux: float,
    *,
    err_background_mode: str,
    sky_pp: float,
    area: float,
    gain: float = 1.0,
    read_noise: float = 10.0,
    sigma_bkg_ap: float | None = None,
) -> tuple[float, str]:
    """Relative err/flux with empirical or legacy Howell background term.

    Empirical (default): ``var = F/g + sigma_bkg_ap^2`` — photutils/SExtractor pattern with
    measured ``sigma_bkg`` at aperture scale (Labbe et al. 2003).
    """
    mode = _normalize_err_background_mode(err_background_mode)
    if not math.isfinite(flux) or flux <= 0:
        return float("nan"), ERR_BKG_SOURCE_HOWELL_FALLBACK
    g = float(gain) if math.isfinite(gain) and gain > 0 else 1.0
    sig_ap = float(sigma_bkg_ap) if sigma_bkg_ap is not None else float("nan")
    if mode == ERR_BKG_MODE_EMPIRICAL and math.isfinite(sig_ap) and sig_ap >= 0:
        variance = flux / g + sig_ap * sig_ap
        if math.isfinite(variance) and variance >= 0:
            return math.sqrt(variance) / flux, ERR_BKG_SOURCE_EMPIRICAL
    err = _photometric_error(flux, sky_pp, area, gain=gain, read_noise=read_noise)
    src = ERR_BKG_MODE_HOWELL if mode == ERR_BKG_MODE_HOWELL else ERR_BKG_SOURCE_HOWELL_FALLBACK
    return err, src


def _sky_pp_for_photometric_error(row: Any) -> float:
    """Sky level (ADU/px) for Howell ``_photometric_error`` from a proc-CSV row.

    Prefer ``sky_adu_per_px_annulus`` (explicit annulus median from aperture export).
    Fall back to ``noise_floor_adu`` for older proc CSVs (happy path: annulus; edge: detection).
    """
    ann = float(pd.to_numeric(row.get(SKY_ADU_PER_PX_ANNULUS_COL), errors="coerce"))
    if math.isfinite(ann) and ann >= 0:
        return ann
    legacy = float(pd.to_numeric(row.get("noise_floor_adu"), errors="coerce"))
    if math.isfinite(legacy) and legacy >= 0:
        return legacy
    return 0.0


def compute_snr_optimal_aperture_table(
    fwhm_px: float,
    sky_adu_per_px: float,
    gain: float = 1.0,
    read_noise: float = 10.0,
    mag_range: tuple[float, float] = (7.0, 18.0),
    mag_step: float = 0.5,
    r_min_fwhm: float = 0.8,
    r_max_fwhm: float = 2.5,
    r_step_px: float = 0.05,
    zero_point: float = 25.0,
    bkg_var_adu2_per_px: float | None = None,
) -> dict[str, Any]:
    """SNR-optimal circular aperture radius per magnitude bin (Gaussian PSF enclosed flux).

    SNR = F(r)/g / sqrt(F(r)/g + N_pix·bkg_var/g)  (dimensionless; flux and noise in e⁻).

    When ``bkg_var_adu2_per_px`` is supplied, per-pixel background variance is taken from
    measured star-free pixels (same frame) instead of reconstructing ``sky/g + (RN/g)²``.
    Residual limitation: a per-pixel metric ignores aperture-scale covariance on resampled
    frames (Fruchter & Hook 2002); acceptable here because ranking, not absolute SNR,
    drives the optimal-radius choice.
    """
    # SNR-optimal aperture selection
    # Howell (1989) PASP 101:616, §3 — SNR(r) = F(r) / sqrt(F(r)/g + pi*r^2*sky/g + pi*r^2*(RN/g)^2)
    fw = float(fwhm_px)
    if not math.isfinite(fw) or fw <= 0:
        fw = 3.5
    sky = float(sky_adu_per_px)
    if not math.isfinite(sky) or sky < 0:
        sky = 0.0
    g = float(gain) if math.isfinite(gain) and gain > 0 else 1.0
    rn = float(read_noise) if math.isfinite(read_noise) and read_noise >= 0 else 10.0
    use_meas_bkg = (
        bkg_var_adu2_per_px is not None
        and math.isfinite(float(bkg_var_adu2_per_px))
        and float(bkg_var_adu2_per_px) >= 0
    )
    bkg_var_px = float(bkg_var_adu2_per_px) if use_meas_bkg else float("nan")

    sigma = fw / 2.355
    r_min_px = float(r_min_fwhm) * fw
    r_max_px = float(r_max_fwhm) * fw
    r_values = np.arange(r_min_px, r_max_px, float(r_step_px))
    if r_values.size == 0:
        r_values = np.array([max(0.5, r_min_px)])

    table: dict[float, float] = {}
    mags = np.arange(float(mag_range[0]), float(mag_range[1]) + float(mag_step), float(mag_step))
    for mag in mags:
        flux_total = 10.0 ** ((float(zero_point) - float(mag)) / 2.5)
        best_snr = -1.0
        best_r = float(r_values[0])
        for r in r_values:
            enclosed = flux_total * (1.0 - np.exp(-(float(r) ** 2) / (2.0 * sigma**2)))
            area = math.pi * float(r) ** 2
            n_photon = enclosed / g
            if use_meas_bkg:
                n_bkg = area * bkg_var_px / g
            else:
                n_sky = area * sky / g
                n_read = area * (rn / g) ** 2
                n_bkg = n_sky + n_read
            noise = math.sqrt(max(n_photon + n_bkg, 1e-12))
            snr = (enclosed / g) / noise if noise > 0 else 0.0
            if snr > best_snr:
                best_snr = snr
                best_r = float(r)
        table[round(float(mag), 1)] = round(best_r, 3)

    return {
        "table": table,
        "fwhm_px": fw,
        "sky_adu_per_px": sky,
        "gain": g,
        "read_noise": rn,
        "r_min_px": r_min_px,
        "r_max_px": r_max_px,
    }


def _resolve_phase2a_equipment_id(
    db: Any | None,
    *,
    draft_id: int | None,
    output_dir: Path,
    masterstar_fits_path: Path,
) -> int | None:
    """``OBS_DRAFT.ID_EQUIPMENTS`` from ``draft_id`` or path segment ``draft_NNN``."""
    if db is None:
        return None
    did = draft_id
    if did is None:
        for base in (Path(output_dir), Path(masterstar_fits_path)):
            for part in base.parts:
                m = re.match(r"draft_(\d+)$", str(part), re.IGNORECASE)
                if m:
                    did = int(m.group(1))
                    break
            if did is not None:
                break
    if did is None:
        return None
    try:
        row = db.conn.execute(
            "SELECT ID_EQUIPMENTS FROM OBS_DRAFT WHERE ID = ?;",
            (int(did),),
        ).fetchone()
        if row is not None and row[0] is not None:
            return int(row[0])
    except Exception as exc:  # noqa: BLE001
        logging.error('[EXC-0125] OBS_DRAFT equipment_id DB read fails - gain/RN resolver falls back without equipment scope: %s', exc)
        return None
    return None


def _draft_dir_from_phase2a_paths(output_dir: Path, masterstar_fits_path: Path) -> Path:
    for base in (Path(output_dir), Path(masterstar_fits_path)):
        for parent in [base, *base.parents]:
            if re.match(r"draft_\d+$", parent.name, re.IGNORECASE):
                return parent
    return Path(output_dir).parent.parent


def _require_comparison_stars_per_target_schema(comp_df: pd.DataFrame, csv_path: Path) -> None:
    """Phase 2A comp routing requires ``target_catalog_id`` (not the pool CSV)."""
    if comp_df is None or comp_df.empty:
        return
    if "target_catalog_id" in comp_df.columns:
        return
    pool_markers = ("comp_id", "role")
    if any(col in comp_df.columns for col in pool_markers):
        raise ValueError(
            "comparison_stars_csv must be comparison_stars_per_target.csv "
            f"(missing target_catalog_id; got comparison pool file): {csv_path}"
        )
    raise ValueError(
        "comparison_stars_csv missing target_catalog_id column "
        f"(required for Phase 2A per-target comp routing): {csv_path}"
    )


LAST_EXCLUDED_TARGETS: pd.DataFrame = pd.DataFrame(
    columns=["name", "vsx_name", "vsx_type", "ra_deg", "dec_deg", "mag", "reason"]
)


def _ac_summary_fields(ac_result: dict[str, Any] | None) -> dict[str, Any]:
    ac = ac_result if isinstance(ac_result, dict) else {}
    applied = bool(ac.get("ok"))
    fields: dict[str, Any] = {
        "ac_applied": applied,
        "ac_skip_reason": "" if applied else str(ac.get("reason", "") or ""),
    }
    if applied:
        for key, src in (
            ("ac_delta_m_corr", "delta_m_corr"),
            ("ac_scatter", "scatter_mag"),
        ):
            val = ac.get(src)
            try:
                fields[key] = float(val) if val is not None else float("nan")
            except (TypeError, ValueError):
                fields[key] = float("nan")
        fields["ac_n_ref"] = int(ac.get("n_ref_stars") or 0)
    return fields


def _phase2a_empty_comp_summary_row(
    *,
    target_cid: str,
    target_name: str,
    zone_flag: str,
) -> dict[str, Any]:
    """Summary stub when an active target has no per-target comparison stars."""
    return {
        "catalog_id": target_cid,
        "vsx_name": target_name,
        "zone_flag": zone_flag,
        "n_frames": 0,
        "n_good_comp": 0,
        "n_saturated": 0,
        "lc_rms": float("nan"),
        "lc_median_mag": float("nan"),
        "aperture_px": float("nan"),
        "am_slope": float("nan"),
        "am_detrended": False,
        "lc_csv": "",
        "lc_png": "",
        "ac_applied": False,
        "ac_skip_reason": "no_comps",
    }


def _phase2a_skip_empty_comps_target(
    *,
    target_cid: str,
    target_name: str,
    zone_flag: str,
    summary_rows: list,
) -> list:
    """Record counted drop when an active target has no per-target comparison stars."""
    from except_fix_counters import get_except_fix_counters

    _ctr = get_except_fix_counters()
    _ctr.phase2a_empty_comp_drop += 1
    logging.error(
        "[FÁZA 2A] Target %s (%s): žiadne comp hviezdy — preskočené "
        "(phase2a_empty_comp_drop=%d)",
        target_name,
        target_cid,
        _ctr.phase2a_empty_comp_drop,
    )
    summary_rows.append(
        _phase2a_empty_comp_summary_row(
            target_cid=target_cid,
            target_name=target_name,
            zone_flag=zone_flag,
        )
    )
    return summary_rows


def _phase2a_star_mag_lookup(
    at_df: pd.DataFrame,
    comp_df: pd.DataFrame,
    masterstar_fits_path: Path,
) -> dict[str, float]:
    """Best-effort observed-band / catalog mag per ``catalog_id`` for SNR aperture lookup.

    Prefers ``mag`` / ``catalog_mag`` (filter-native) over broad Gaia ``phot_g_mean_mag``.
    """
    out: dict[str, float] = {}
    for df in (at_df, comp_df):
        if df is None or df.empty or "catalog_id" not in df.columns:
            continue
        for mag_col in _APERTURE_SIZING_MAG_COLS:
            if mag_col not in df.columns:
                continue
            for _, r in df.iterrows():
                cid = _normalize_gaia_id(r.get("catalog_id", ""))
                if not cid or cid in out:
                    continue
                v = pd.to_numeric(r.get(mag_col), errors="coerce")
                if math.isfinite(float(v)):
                    out[cid] = float(v)
    try:
        ms_full = Path(masterstar_fits_path).resolve().parent / "masterstars_full_match.csv"
        if ms_full.is_file():
            ms_df0 = pd.read_csv(
                ms_full,
                low_memory=False,
                usecols=lambda c: c in ("catalog_id", *_APERTURE_SIZING_MAG_COLS),
                dtype=_GAIA_ID_DTYPE,
            )
            ms_df0["catalog_id"] = ms_df0["catalog_id"].apply(_normalize_gaia_id)
            for mag_col in _APERTURE_SIZING_MAG_COLS:
                if mag_col not in ms_df0.columns:
                    continue
                for _, r in ms_df0.iterrows():
                    cid = str(r.get("catalog_id") or "").strip()
                    if not cid or cid in out:
                        continue
                    v = pd.to_numeric(r.get(mag_col), errors="coerce")
                    if math.isfinite(float(v)):
                        out[cid] = float(v)
    except Exception as exc:  # noqa: BLE001
        logging.error("[EXC-0126] Per-star mag from masterstars CSV cache load fails - aperture sizing lacks that star's ...: %s", exc)
        pass
    return out


def _median_sky_from_phase2a_csv_cache(
    csv_cache: dict[str, pd.DataFrame],
    *,
    fallback: float = 1581.6,
) -> float:
    vals: list[float] = []
    for df in csv_cache.values():
        if df is None or df.empty or "noise_floor_adu" not in df.columns:
            continue
        s = pd.to_numeric(df["noise_floor_adu"], errors="coerce").dropna()
        vals.extend(float(x) for x in s.tolist() if math.isfinite(float(x)))
    if not vals:
        return float(fallback)
    med = float(np.nanmedian(np.asarray(vals, dtype=np.float64)))
    return med if math.isfinite(med) and med > 0 else float(fallback)


def _measured_aperture_from_proc_cache(
    catalog_id: str,
    csv_cache: dict[str, pd.DataFrame],
    *,
    id_col: str = "catalog_id",
) -> float:
    """Median ``aperture_r_px`` from per-frame proc CSV (flux measurement truth)."""
    cid = _normalize_gaia_id(catalog_id) or str(catalog_id).strip()
    vals: list[float] = []
    for df in csv_cache.values():
        if df is None or df.empty or "aperture_r_px" not in df.columns:
            continue
        col = id_col if id_col in df.columns else (
            "catalog_id" if "catalog_id" in df.columns else "name"
        )
        if col not in df.columns:
            continue
        try:
            ids = df[col].apply(_normalize_gaia_id)
        except Exception:  # noqa: BLE001
            ids = df[col].astype(str)
        sub = df[ids.astype(str) == cid]
        if sub.empty:
            continue
        ap = pd.to_numeric(sub["aperture_r_px"], errors="coerce").dropna()
        vals.extend(float(x) for x in ap.tolist() if math.isfinite(float(x)) and float(x) > 0)
    if not vals:
        return float("nan")
    return float(np.median(np.asarray(vals, dtype=np.float64)))


def _snr_table_radius_for_mag_bin(table: dict[Any, Any], nearest: float) -> float | None:
    """Lookup radius in SNR table (float or JSON string mag keys)."""
    for key in (nearest, round(float(nearest), 1), str(nearest), str(round(float(nearest), 1))):
        if key in table:
            try:
                val = float(table[key])
            except (TypeError, ValueError):
                continue
            if math.isfinite(val):
                return val
    return None


def _aperture_radius_from_snr_table(
    star_mag: float,
    snr_table: dict[str, Any],
    *,
    aperture_fwhm_factor: float,
    fwhm_px: float,
) -> float:
    """Nearest mag bin in SNR table, clamped to ``r_min_px``..``r_max_px``."""
    table = snr_table.get("table") or {}
    if not table:
        return float(aperture_fwhm_factor) * float(fwhm_px)
    mag_bins = [float(k) for k in table]
    if not math.isfinite(star_mag):
        star_mag = 99.0
    nearest = min(mag_bins, key=lambda m: abs(m - float(star_mag)))
    r_opt = _snr_table_radius_for_mag_bin(table, nearest)
    if r_opt is None:
        return float(aperture_fwhm_factor) * float(fwhm_px)
    r_min = float(snr_table.get("r_min_px", r_opt))
    r_max = float(snr_table.get("r_max_px", r_opt))
    return max(r_min, min(r_max, r_opt))


def _resolve_photometric_aperture_px_for_gs11(
    target_cid: str,
    apertures_px: dict[str, float],
    target_g_mag: float,
    snr_ap_table: dict[str, Any] | None,
    *,
    aperture_fwhm_factor: float,
    fwhm_px: float,
) -> tuple[float | None, str]:
    """Layered photometric aperture for GS11 dilution (Seager 2003 / Howell 2006).

    1. Per-star map from Phase 2A SNR sizing (same build as ``apertures_px``).
    2. Derive from SNR table at ``target_g_mag`` via ``_aperture_radius_from_snr_table``.
    3. Unavailable — caller must skip dilution (no fixed-pixel fallback).
    """
    cid = _normalize_gaia_id(target_cid) if target_cid else ""
    if cid and cid in apertures_px:
        ap = float(apertures_px[cid])
        if math.isfinite(ap) and ap > 0:
            return ap, "map"
    if snr_ap_table is not None and math.isfinite(float(target_g_mag)):
        ap = float(
            _aperture_radius_from_snr_table(
                float(target_g_mag),
                snr_ap_table,
                aperture_fwhm_factor=float(aperture_fwhm_factor),
                fwhm_px=float(fwhm_px),
            )
        )
        if math.isfinite(ap) and ap > 0:
            return ap, "snr_derived"
    return None, "unavailable"


def _get_star_aperture_px(
    catalog_id: str,
    star_mag: float | None,
    snr_table: dict[str, Any] | None,
    *,
    fallback_r: float,
) -> float:
    """Vráti r_opt z SNR table pre danú hviezdu, alebo fallback."""
    _ = catalog_id  # reserved for per-id overrides; lookup is mag-binned today
    if snr_table is None:
        return float(fallback_r)
    _table = snr_table.get("table") or {}
    _r_min = float(snr_table.get("r_min_px", float(fallback_r) * 0.5))
    _r_max = float(snr_table.get("r_max_px", float(fallback_r) * 2.0))
    if not _table:
        return float(fallback_r)
    if star_mag is None:
        return float(fallback_r)
    try:
        _mag_f = float(star_mag)
    except (TypeError, ValueError):
        return float(fallback_r)
    if not math.isfinite(_mag_f):
        return float(fallback_r)
    _mag_bins = [float(k) for k in _table]
    if not _mag_bins:
        return float(fallback_r)
    _nearest = min(_mag_bins, key=lambda m: abs(m - _mag_f))
    _r_opt = _snr_table_radius_for_mag_bin(_table, _nearest)
    if _r_opt is None:
        return float(fallback_r)
    return max(_r_min, min(_r_max, _r_opt))


def resolve_draft_dir_for_snr_aperture_table(
    *,
    archive_root: str | Path | None = None,
    draft_id: int | None = None,
    platesolve_dir: Path | str | None = None,
    masterstar_fits_path: Path | str | None = None,
) -> Path | None:
    """Best-effort draft folder (``draft_NNNNNN``) for ``aperture_snr_table.json``."""
    if draft_id is not None:
        try:
            did = int(draft_id)
            if did > 0 and archive_root is not None:
                p = Path(archive_root) / "Drafts" / f"draft_{did:06d}"
                if p.is_dir():
                    return p.resolve()
        except (TypeError, ValueError):
            pass
    for raw in (platesolve_dir, masterstar_fits_path):
        if raw is None or not str(raw).strip():
            continue
        try:
            return _draft_dir_from_phase2a_paths(Path(str(raw)), Path(str(raw)))
        except Exception:  # noqa: BLE001
            # EXC-0127: T2 -- Draft directory inference from path fails - SNR table loader tries next candidate path (EXCEPT-BULK-2 2026-07-08)
            continue
    return None


def load_snr_aperture_table_from_draft_dir(
    draft_dir: Path | str | None,
) -> dict[str, Any] | None:
    """Načíta ``aperture_snr_table.json`` z draft priečinka (ak existuje)."""
    if draft_dir is None or not str(draft_dir).strip():
        return None
    dd = Path(draft_dir)
    if not dd.is_dir():
        return None
    path = dd / "aperture_snr_table.json"
    if not path.is_file():
        logging.warning("[FÁZA 2A] aperture_snr_table.json nenájdená — používam globálnu apertúru")
        return None
    try:
        with path.open(encoding="utf-8") as f:
            table = json.load(f)
    except Exception as exc:  # noqa: BLE001
        logging.error('[EXC-0128] aperture_snr_table.json unreadable - Phase 2A uses global default aperture instead of p...: %s', exc)
        logging.warning("[FÁZA 2A] aperture_snr_table.json nečitateľná (%s) — globálna apertúra", exc)
        return None
    if not isinstance(table, dict):
        logging.warning("[FÁZA 2A] aperture_snr_table.json neplatný formát — globálna apertúra")
        return None
    logging.info(
        "[FÁZA 2A] Načítaná SNR aperture table: fwhm=%spx gain=%s sky=%s",
        table.get("fwhm_px"),
        table.get("gain"),
        table.get("sky_adu_per_px"),
    )
    return table


def _noise_floor_adu_from_image_array(
    data: Any,
    *,
    prematch_peak_sigma_floor: float = 10.0,
) -> float | None:
    """DAO-style noise floor (median + k×σ) for SNR table sky estimate."""
    from astropy.stats import sigma_clipped_stats

    arr = np.asarray(data, dtype=np.float64)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return None
    _, med, std = sigma_clipped_stats(arr[finite], sigma=3.0, maxiters=3)
    k = float(prematch_peak_sigma_floor)
    if not math.isfinite(k):
        k = 10.0
    k = min(15.0, max(0.5, k))
    std_f = float(std) if np.isfinite(std) else 0.0
    nf = float(med) + k * max(std_f, 1.0)
    return nf if math.isfinite(nf) and nf > 0 else None


def resolve_fwhm_px_for_snr_aperture_table(
    *,
    masterstar_fits_path: Path | str | None,
    masterstar_selection: dict[str, Any] | None,
    fwhm_fallback_px: float | None = None,
) -> float | None:
    """FWHM for SNR table: VY_FWHM_GAUSS / VY_FWHM, then best_frame_fwhm_px, then fallback."""
    if masterstar_fits_path is not None and str(masterstar_fits_path).strip():
        try:
            with astrofits.open(Path(masterstar_fits_path), memmap=False) as hdul:
                hdr = hdul[0].header
            for key in ("VY_FWHM_GAUSS", "VY_FWHM"):
                v = hdr.get(key)
                if v is None:
                    continue
                try:
                    vf = float(v)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(vf) and 0.5 < vf < 30.0:
                    return vf
        except Exception as exc:  # noqa: BLE001
            logging.error('[EXC-0129] MASTERSTAR header FWHM key parse fails - SNR/aperture table uses later fallback FWHM so...: %s', exc)
            pass
    sel = masterstar_selection if isinstance(masterstar_selection, dict) else {}
    try:
        bf = float(sel.get("best_frame_fwhm_px"))
        if math.isfinite(bf) and 0.5 < bf < 30.0:
            return bf
    except (TypeError, ValueError):
        pass
    if fwhm_fallback_px is not None:
        try:
            fb = float(fwhm_fallback_px)
            if math.isfinite(fb) and fb > 0:
                return fb
        except (TypeError, ValueError):
            pass
    return None


def estimate_median_sky_adu_per_px_for_snr_table(
    *,
    aligned_fits_paths: Sequence[Path | str] | None = None,
    aligned_ram_frames: Sequence[tuple[str, Any, Any]] | None = None,
    max_frames: int = 12,
    prematch_peak_sigma_floor: float = 10.0,
    fallback: float = 1581.6,
) -> float:
    """Median DAO noise-floor estimate across aligned frames (pre–per-frame CSV)."""
    vals: list[float] = []
    n_max = max(1, int(max_frames))

    if aligned_ram_frames:
        for _name, _hdr, arr in list(aligned_ram_frames)[:n_max]:
            nf = _noise_floor_adu_from_image_array(
                arr,
                prematch_peak_sigma_floor=prematch_peak_sigma_floor,
            )
            if nf is not None:
                vals.append(float(nf))

    if aligned_fits_paths:
        for raw in list(aligned_fits_paths)[:n_max]:
            p = Path(raw)
            if not p.is_file():
                continue
            try:
                with astrofits.open(p, memmap=True) as hdul:
                    d = hdul[0].data
                if d is None:
                    continue
                nf = _noise_floor_adu_from_image_array(
                    d,
                    prematch_peak_sigma_floor=prematch_peak_sigma_floor,
                )
                if nf is not None:
                    vals.append(float(nf))
            except Exception as exc:  # noqa: BLE001
                logging.error('[EXC-0130] One frame skipped when estimating median sky for SNR table - sky_adu biased if few frames: %s', exc)
                continue

    if not vals:
        return float(fallback)
    med = float(np.nanmedian(np.asarray(vals, dtype=np.float64)))
    return med if math.isfinite(med) and med > 0 else float(fallback)


def _median_bkg_var_adu2_per_px_from_proc_cache(
    csv_cache: dict[str, pd.DataFrame],
) -> float | None:
    """Median per-pixel background variance [ADU²/px] from empirical ``sigma_bkg_ap`` in proc CSVs."""
    vals: list[float] = []
    for _df in csv_cache.values():
        if _df is None or _df.empty:
            continue
        if SIGMA_BKG_AP_COL not in _df.columns or "aperture_r_px" not in _df.columns:
            continue
        sig = pd.to_numeric(_df[SIGMA_BKG_AP_COL], errors="coerce")
        rap = pd.to_numeric(_df["aperture_r_px"], errors="coerce")
        ok = sig.notna() & rap.notna() & (rap > 0) & (sig >= 0)
        if not ok.any():
            continue
        area = math.pi * rap[ok].to_numpy(dtype=np.float64) ** 2
        var_ap = sig[ok].to_numpy(dtype=np.float64) ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            var_px = var_ap / np.maximum(area, 1e-12)
        vals.extend([float(v) for v in var_px if math.isfinite(float(v)) and float(v) >= 0])
    if not vals:
        return None
    med = float(np.nanmedian(np.asarray(vals, dtype=np.float64)))
    return med if math.isfinite(med) and med >= 0 else None


def _median_bkg_var_from_aligned_frames(
    *,
    aligned_fits_paths: Sequence[Path | str] | None = None,
    aligned_ram_frames: Sequence[tuple[str, Any, Any]] | None = None,
    max_frames: int = 6,
) -> float | None:
    """Median star-free per-pixel variance across aligned frames (no catalog — edge patches)."""
    vals: list[float] = []
    n_max = max(1, int(max_frames))

    if aligned_ram_frames:
        for _name, _hdr, arr in list(aligned_ram_frames)[:n_max]:
            v = estimate_star_free_per_pixel_variance_adu2(arr)
            if v is not None:
                vals.append(float(v))

    if aligned_fits_paths:
        for raw in list(aligned_fits_paths)[:n_max]:
            p = Path(raw)
            if not p.is_file():
                continue
            try:
                with astrofits.open(p, memmap=True) as hdul:
                    d = hdul[0].data
                if d is None:
                    continue
                v = estimate_star_free_per_pixel_variance_adu2(d)
                if v is not None:
                    vals.append(float(v))
            except Exception:  # noqa: BLE001
                continue

    if not vals:
        return None
    med = float(np.nanmedian(np.asarray(vals, dtype=np.float64)))
    return med if math.isfinite(med) and med >= 0 else None


def precompute_and_save_snr_aperture_table_for_draft(
    draft_dir: Path | str,
    *,
    masterstar_fits_path: Path | str | None = None,
    masterstar_selection: dict[str, Any] | None = None,
    fwhm_fallback_px: float | None = None,
    aligned_fits_paths: Sequence[Path | str] | None = None,
    aligned_ram_frames: Sequence[tuple[str, Any, Any]] | None = None,
    database_path: Path | str | None = None,
    draft_id: int | None = None,
    equipment_id: int | None = None,
    sky_fallback: float = 1581.6,
    prematch_peak_sigma_floor: float = 10.0,
) -> dict[str, Any] | None:
    """Build and write ``aperture_snr_table.json`` before per-frame catalog export."""
    dd = Path(draft_dir)
    if not dd.is_dir():
        logging.warning("[PIPELINE] SNR table: draft_dir not found: %s", dd)
        return None

    fwhm_px = resolve_fwhm_px_for_snr_aperture_table(
        masterstar_fits_path=masterstar_fits_path,
        masterstar_selection=masterstar_selection,
        fwhm_fallback_px=fwhm_fallback_px,
    )
    if fwhm_px is None or not math.isfinite(float(fwhm_px)) or float(fwhm_px) <= 0:
        logging.warning("[PIPELINE] SNR table: no valid FWHM — skip precompute")
        return None

    gain_p = 1.0
    rn_p = 10.0
    gain_src = "default"
    db = None
    if database_path is not None and str(database_path).strip():
        try:
            from database import VyvarDatabase

            db = VyvarDatabase(Path(database_path))
        except Exception:  # noqa: BLE001
            db = None
    eq_id = equipment_id
    if db is not None and eq_id is None:
        eq_id = _resolve_phase2a_equipment_id(
            db,
            draft_id=draft_id,
            output_dir=dd,
            masterstar_fits_path=Path(masterstar_fits_path) if masterstar_fits_path else dd,
        )
    _snr_header = None
    if masterstar_fits_path is not None and str(masterstar_fits_path).strip():
        try:
            with astrofits.open(Path(masterstar_fits_path), memmap=False) as hdul:
                _snr_header = hdul[0].header
        except Exception:  # noqa: BLE001
            _snr_header = None
    if db is not None and eq_id is not None:
        # Unified gain resolution (param_resolver): header e-/ADU or index-mapped ->
        # DB -> config. Read noise stays DB-first. Shared with Phase 2A and error map.
        from param_resolver import resolve_gain, resolve_read_noise  # noqa: PLC0415

        _g_res = resolve_gain(_snr_header, db=db, equipment_id=int(eq_id))
        _rn_res = resolve_read_noise(_snr_header, db=db, equipment_id=int(eq_id))
        if _g_res.ok:
            gain_p = float(_g_res.value)
            gain_src = _g_res.source
        if _rn_res.ok:
            rn_p = float(_rn_res.value)
    if db is not None:
        try:
            db.conn.close()
        except Exception:  # noqa: BLE001
            # EXC-0131: T2 -- db.conn.close() after SNR table precompute ignored (EXCEPT-BULK-2 2026-07-08)
            pass

    logging.info(
        "[PIPELINE] SNR table gain=%.3f e-/ADU RN=%.1f e- (source: %s)",
        float(gain_p),
        float(rn_p),
        gain_src,
    )

    sky_adu = estimate_median_sky_adu_per_px_for_snr_table(
        aligned_fits_paths=aligned_fits_paths,
        aligned_ram_frames=aligned_ram_frames,
        prematch_peak_sigma_floor=prematch_peak_sigma_floor,
        fallback=float(sky_fallback),
    )
    n_sky = 0
    if aligned_ram_frames:
        n_sky += min(len(aligned_ram_frames), 12)
    if aligned_fits_paths:
        n_sky = max(n_sky, min(len(list(aligned_fits_paths)), 12))
    logging.info(
        "[PIPELINE] Sky for SNR table: %.1f ADU/px (median of up to %s frames)",
        float(sky_adu),
        int(n_sky) if n_sky > 0 else 0,
    )

    bkg_var_px = _median_bkg_var_from_aligned_frames(
        aligned_fits_paths=aligned_fits_paths,
        aligned_ram_frames=aligned_ram_frames,
    )
    if bkg_var_px is not None:
        logging.info(
            "[PIPELINE] SNR table: measured star-free bkg var = %.4g ADU²/px (per-pixel; ranking-only)",
            float(bkg_var_px),
        )

    snr_table = compute_snr_optimal_aperture_table(
        fwhm_px=float(fwhm_px),
        sky_adu_per_px=float(sky_adu),
        gain=float(gain_p),
        read_noise=float(rn_p),
        bkg_var_adu2_per_px=bkg_var_px,
    )
    out_path = dd / "aperture_snr_table.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(snr_table, f, indent=2)

    tbl = snr_table.get("table") or {}

    def _r_at(mag: float) -> float:
        v = _snr_table_radius_for_mag_bin(tbl, mag)
        return float(v) if v is not None else float("nan")

    logging.info(
        "[PIPELINE] aperture_snr_table.json uložená pred exportom CSV: "
        "mag7→%.2fpx mag11→%.2fpx mag14→%.2fpx (%s)",
        _r_at(7.0),
        _r_at(11.0),
        _r_at(14.0),
        out_path,
    )
    return snr_table


def _coerce_bool_cell(v: Any) -> bool:
    """Robustly coerce a CSV cell (bool / 'True' / 1 / NaN / '') to bool; NaN/empty → False."""
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    if isinstance(v, (int, float)):
        try:
            if math.isnan(float(v)):
                return False
        except (TypeError, ValueError):
            return False
        return float(v) != 0.0
    s = str(v).strip().lower()
    return s in ("true", "1", "yes", "y", "t")


def read_flux_from_csv(
    frame_csv_path: Path,
    star_ids: list[str],
    apertures_px: dict[str, float],
    *,
    sat_limit_adu: float | None = None,
    star_xy: dict[str, tuple[float, float]] | None = None,
    xy_tol_px: float = 15.0,
    frame_times: dict[str, Any] | None = None,
    csv_df: pd.DataFrame | None = None,
    lookup: tuple[dict[str, pd.Series], pd.DataFrame] | None = None,
    gain: float = 1.0,
    read_noise: float = 10.0,
    use_apcorr_flux: bool = False,
    variable_target_catalog_ids: frozenset[str] | None = None,
    err_background_mode: str = ERR_BKG_MODE_EMPIRICAL,
) -> pd.DataFrame:
    """Krok 2: Načítaj flux z per-frame CSV (dao_flux).

    Namiesto čítania FITS a vlastnej aperturnej fotometrie používa
    dao_flux ktorý pipeline vypočítala počas DAO detekcie.
    dao_flux je sky-subtrahovaný flux zmeraný s aperture_r_px z CSV.

    Returns:
        DataFrame: catalog_id, bjd, hjd, jd, airmass, mag_inst, err,
                   aperture_r_px, sky_pp, flag, source_file

    Args:
        lookup: Voliteľný výstup z ``_build_csv_lookup`` pre zdieľaný ``csv_df``
            (Fáza 2A — jedna výstavba lookupu na snímku namiesto 1× na target).
    """
    if csv_df is None:
        if not Path(frame_csv_path).is_file():
            return pd.DataFrame()
        try:
            # Keep Gaia IDs stable (avoid float/scientific precision loss in per-frame CSV).
            csv_df = pd.read_csv(frame_csv_path, low_memory=False, dtype=_GAIA_ID_DTYPE)
        except (OSError, ValueError) as exc:
            logging.warning(f"[FÁZA 2A] Nemôžem čítať CSV {frame_csv_path}: {exc}")
            return pd.DataFrame()

    if csv_df.empty:
        return pd.DataFrame()

    _lim_raw = sat_limit_adu if sat_limit_adu is not None else _sat_limit_peak_adu()
    if _lim_raw is None or (isinstance(_lim_raw, float) and not math.isfinite(_lim_raw)):
        _sat_lim = float("inf")
    else:
        _sat_lim = float(_lim_raw)
    source_file = frame_csv_path.name

    id_col = "catalog_id" if "catalog_id" in csv_df.columns else "name"
    if lookup is not None:
        id_map, xy_df_lookup = lookup
    else:
        id_map, xy_df_lookup = _build_csv_lookup(csv_df, id_col)

    # Airmass z frame_times
    am_frame = float("nan")
    flip_frame: bool | None = None
    align_failed_frame: bool = False
    frame_catalog_match_mode = ""
    if frame_times:
        try:
            _am = float(frame_times.get("airmass", float("nan")))
            if math.isfinite(_am):
                am_frame = _am
        except (TypeError, ValueError):
            pass
        try:
            _fl = frame_times.get("is_flipped", None)
            if isinstance(_fl, bool):
                flip_frame = _fl
            elif _fl is not None:
                s = str(_fl).strip().lower()
                if s in ("true", "1", "yes", "y"):
                    flip_frame = True
                elif s in ("false", "0", "no", "n"):
                    flip_frame = False
        except Exception:  # noqa: BLE001
            flip_frame = None
        try:
            _af = frame_times.get("alignment_failed", None)
            if isinstance(_af, bool):
                align_failed_frame = bool(_af)
            elif _af is not None:
                s = str(_af).strip().lower()
                if s in ("true", "1", "yes", "y"):
                    align_failed_frame = True
                elif s in ("false", "0", "no", "n"):
                    align_failed_frame = False
            elif frame_times.get("aligned", None) is not None:
                _al = frame_times.get("aligned")
                if isinstance(_al, bool):
                    align_failed_frame = not _al
                else:
                    s = str(_al).strip().lower()
                    align_failed_frame = s in ("false", "0", "no", "n")
        except Exception:  # noqa: BLE001
            align_failed_frame = False
        _cmm_ft = frame_times.get("catalog_match_mode", None)
        if _cmm_ft is not None:
            frame_catalog_match_mode = normalize_catalog_match_mode(str(_cmm_ft))

    rows: list[dict] = []

    for cid in star_ids:
        base: dict[str, Any] = {
            "catalog_id": cid,
            "source_file": source_file,
            "bjd": float("nan"),
            "hjd": float("nan"),
            "jd": float("nan"),
            "airmass": am_frame,
            "is_flipped": flip_frame,
            "alignment_failed": align_failed_frame,
            "catalog_match_mode": frame_catalog_match_mode,
            "wcs_untrusted": is_wcs_untrusted_catalog_match_mode(frame_catalog_match_mode),
            "mag_inst": float("nan"),
            "err": float("nan"),
            # PSF photometry (b.5) columns — carried through so Phase 2A star-method / adaptive
            # routing can see them. Default NaN/False so a frame/CSV without PSF stays pure-aperture.
            "psf_flux": float("nan"),
            "psf_flux_err": float("nan"),
            "psf_fit_ok": False,
            "psf_quality": "",
            "psf_quality_fallback": False,
            "psf_snr": float("nan"),
            "psf_ac_factor": float("nan"),
            "psf_ac_n_used": 0,
            "psf_ac_applied": False,
            "aperture_r_px": apertures_px.get(cid, float("nan")),
            "x": float("nan"),
            "y": float("nan"),
            "sky_annulus_r_out_px": float("nan"),
            "edge_fail": False,
            "sky_pp": float("nan"),
            "flux_raw": float("nan"),
            "flux_small": float("nan"),
            "flux_large": float("nan"),
            "flag": "no_data",
        }

        ref_x, ref_y = None, None
        if star_xy and cid in star_xy:
            rx, ry = star_xy[cid]
            ref_x = float(rx) if math.isfinite(float(rx)) else None
            ref_y = float(ry) if math.isfinite(float(ry)) else None

        cid_key = _normalize_gaia_id(cid)
        _is_variable_target = (
            variable_target_catalog_ids is not None
            and cid_key
            and cid_key in variable_target_catalog_ids
        )
        if _is_variable_target:
            if not cid_key or cid_key not in id_map:
                rows.append(base)
                continue
            row_csv = id_map[cid_key]
        else:
            row_csv = _lookup_star_in_csv(
                cid_key or cid, id_map, xy_df_lookup, ref_x, ref_y, xy_tol_px=xy_tol_px
            )
            if row_csv is None:
                rows.append(base)
                continue

            # XY fallback (nie priamy ID hit): comp pool keeps legacy guard.
            if not cid_key or cid_key not in id_map:
                fallback_flux = float(row_csv.get("dao_flux", float("nan")))
                if math.isfinite(fallback_flux) and fallback_flux > 0:
                    fallback_mag = _flux_to_mag(fallback_flux)
                    if math.isfinite(fallback_mag) and fallback_mag > -8.0:
                        logging.warning(
                            "[FÁZA 2A] XY fallback wrong star: cid=%s, fallback_mag=%.2f > -8.0, "
                            "nastavujem NaN",
                            cid,
                            fallback_mag,
                        )
                        rows.append(base)
                        continue

        # PSF photometry (b.5) — read per-star/per-frame PSF flux + quality if present.
        base["psf_flux"] = float(pd.to_numeric(row_csv.get("psf_flux"), errors="coerce"))
        base["psf_flux_err"] = float(pd.to_numeric(row_csv.get("psf_flux_err"), errors="coerce"))
        base["psf_snr"] = float(pd.to_numeric(row_csv.get("psf_snr"), errors="coerce"))
        base["psf_fit_ok"] = _coerce_bool_cell(row_csv.get("psf_fit_ok"))
        base["psf_quality_fallback"] = _coerce_bool_cell(row_csv.get("psf_quality_fallback"))
        base["psf_ac_factor"] = float(pd.to_numeric(row_csv.get("psf_ac_factor"), errors="coerce"))
        _ac_n = pd.to_numeric(row_csv.get("psf_ac_n_used"), errors="coerce")
        base["psf_ac_n_used"] = int(_ac_n) if pd.notna(_ac_n) else 0
        base["psf_ac_applied"] = _coerce_bool_cell(row_csv.get("psf_ac_applied"))
        _pq = row_csv.get("psf_quality")
        base["psf_quality"] = str(_pq).strip().lower() if _pq is not None and not (
            isinstance(_pq, float) and math.isnan(_pq)
        ) else ""

        _row_cmm = normalize_catalog_match_mode(row_csv.get("catalog_match_mode"))
        if _row_cmm:
            base["catalog_match_mode"] = _row_cmm
            base["wcs_untrusted"] = is_wcs_untrusted_catalog_match_mode(_row_cmm)

        # Časové značky
        base["bjd"] = float(row_csv.get("bjd_tdb_mid", float("nan")))
        base["hjd"] = float(row_csv.get("hjd_mid", float("nan")))
        base["jd"] = float(row_csv.get("jd_mid", float("nan")))

        # Airmass fallback: ak frame_times nebolo dostupné, čítaj priamo z CSV riadku
        if not math.isfinite(am_frame):
            am_csv = float(row_csv.get("airmass", float("nan")))
            if math.isfinite(am_csv):
                base["airmass"] = am_csv

        # dao_flux — sky-subtrahovaný flux z DAO fotometrie
        flux = float(row_csv.get("dao_flux", float("nan")))
        if not math.isfinite(flux):
            rows.append(base)
            continue
        base["flux_raw"] = flux

        # Curve-of-growth aperture correction (gated). Carry diagnostics; route corrected
        # flux into mag_inst only when enabled, columns present and cog_ok. Never touches dao_flux.
        _acf = float(pd.to_numeric(row_csv.get("ac_factor"), errors="coerce"))
        _dao_apc = float(pd.to_numeric(row_csv.get("dao_flux_apcorr"), errors="coerce"))
        _cog_ok = _coerce_bool_cell(row_csv.get("cog_ok"))
        base["ac_factor"] = _acf if math.isfinite(_acf) else float("nan")
        base["dao_flux_apcorr"] = _dao_apc if math.isfinite(_dao_apc) else float("nan")
        base["cog_ok"] = bool(_cog_ok)
        if use_apcorr_flux and _cog_ok and math.isfinite(_dao_apc) and _dao_apc > 0:
            flux = _dao_apc

        for _fx in ("flux_small", "flux_large"):
            try:
                _fv = float(pd.to_numeric(row_csv.get(_fx), errors="coerce"))
            except Exception:  # noqa: BLE001
                _fv = float("nan")
            base[_fx] = _fv if math.isfinite(_fv) else float("nan")

        # Apertura z CSV (tá čo pipeline použila pri DAO)
        ap_csv = float(row_csv.get("aperture_r_px", float("nan")))
        if math.isfinite(ap_csv) and ap_csv > 0:
            base["aperture_r_px"] = ap_csv

        # Sky per pixel for Howell err: explicit annulus column, legacy noise_floor fallback.
        sky_pp = _sky_pp_for_photometric_error(row_csv)
        if math.isfinite(sky_pp):
            base["sky_pp"] = sky_pp

        # Saturácia
        peak = float(row_csv.get("peak_max_adu", float("nan")))
        is_sat = math.isfinite(peak) and math.isfinite(_sat_lim) and peak > _sat_lim

        if flux <= 0:
            base["flag"] = "no_data"
            rows.append(base)
            continue

        # Inštrumentálna magnitúda
        base["mag_inst"] = _flux_to_mag(flux)

        # Geometry (for per-frame annulus-aware edge checks)
        try:
            base["x"] = float(pd.to_numeric(row_csv.get("x"), errors="coerce"))
            base["y"] = float(pd.to_numeric(row_csv.get("y"), errors="coerce"))
        except Exception:  # noqa: BLE001
            base["x"] = float("nan")
            base["y"] = float("nan")
        try:
            base["sky_annulus_r_out_px"] = float(pd.to_numeric(row_csv.get("sky_annulus_r_out_px"), errors="coerce"))
        except Exception:  # noqa: BLE001
            base["sky_annulus_r_out_px"] = float("nan")

        # Chyba — fotónový šum + background (empirical empty-aperture or Howell legacy)
        r_ap = base["aperture_r_px"]
        area = math.pi * r_ap * r_ap if math.isfinite(r_ap) and r_ap > 0 else float("nan")
        _sig_bkg = float(pd.to_numeric(row_csv.get(SIGMA_BKG_AP_COL), errors="coerce"))
        if not math.isfinite(_sig_bkg):
            _sig_bkg = None
        _err, _err_src = _photometric_error_with_bkg_mode(
            flux,
            err_background_mode=err_background_mode,
            sky_pp=sky_pp if math.isfinite(sky_pp) else 0.0,
            area=area if math.isfinite(area) else 0.0,
            gain=gain,
            read_noise=read_noise,
            sigma_bkg_ap=_sig_bkg,
        )
        base["err"] = _err
        base[ERR_BKG_SOURCE_COL] = _err_src
        if _sig_bkg is not None:
            base[SIGMA_BKG_AP_COL] = _sig_bkg
        base["flag"] = "saturated" if is_sat else "normal"

        rows.append(base)

    return pd.DataFrame(rows)




def _annulus_sky_subtracted_flux(
    data: np.ndarray,
    x_c: float,
    y_c: float,
    r_ap: float,
    r_in: float,
    r_out: float,
) -> tuple[float, float, float]:
    """Sky-subtracted aperture sum, annulus sky median, peak in aperture (shared DAO/PSF path)."""
    if not (math.isfinite(x_c) and math.isfinite(y_c) and math.isfinite(r_ap) and r_ap > 0):
        return float("nan"), float("nan"), float("nan")
    try:
        from photutils.aperture import CircularAnnulus, CircularAperture
        from photutils.aperture import aperture_photometry as _aphot
    except ImportError:
        return float("nan"), float("nan"), float("nan")

    d = np.asarray(data, dtype=np.float64)
    if np.any(~np.isfinite(d)):
        fill = float(np.nanmedian(d)) if np.any(np.isfinite(d)) else 0.0
        d = np.where(np.isfinite(d), d, fill)

    pos = np.array([[float(x_c), float(y_c)]], dtype=np.float64)
    ap = CircularAperture(pos, r=float(r_ap))
    an = CircularAnnulus(pos, r_in=float(r_in), r_out=float(r_out))

    phot_ap = _aphot(d, ap)
    sum_ap = float(np.asarray(phot_ap["aperture_sum"], dtype=np.float64).ravel()[0])
    area_ap = float(ap.area)

    sky_pp = float("nan")
    sky_ok = False
    ann_masks = an.to_mask(method="center")
    if not isinstance(ann_masks, (list, tuple)):
        ann_masks = [ann_masks]
    for amask in ann_masks:
        try:
            cut = amask.get_values(d)
            cut = np.asarray(cut, dtype=np.float64).ravel()
            cut = cut[np.isfinite(cut)]
            if cut.size > 0:
                sky_pp = float(np.median(cut))
                sky_ok = True
                break
        except (ValueError, TypeError, IndexError) as exc:
            from except_fix_counters import get_except_fix_counters

            get_except_fix_counters().sky_annulus_mask_fail += 1
            logging.error(
                "[PHOT] annulus sky mask failed x=%.2f y=%.2f: %s",
                float(x_c),
                float(y_c),
                exc,
            )
            continue

    if not sky_ok:
        from except_fix_counters import get_except_fix_counters

        get_except_fix_counters().sky_annulus_invalid += 1
        logging.error(
            "[PHOT] annulus sky invalid (no usable pixels) x=%.2f y=%.2f r_ap=%.2f",
            float(x_c),
            float(y_c),
            float(r_ap),
        )
        peak_local = float("nan")
        try:
            m_ap = ap.to_mask(method="center")
            if isinstance(m_ap, (list, tuple)):
                m0 = m_ap[0]
            else:
                m0 = m_ap
            vals = m0.get_values(d)
            peak_local = (
                float(np.nanmax(np.asarray(vals, dtype=np.float64)))
                if vals is not None
                else float("nan")
            )
        except (ValueError, TypeError, IndexError):
            peak_local = float("nan")
        return float("nan"), float("nan"), peak_local

    flux_net = float(sum_ap - sky_pp * area_ap)
    try:
        m_ap = ap.to_mask(method="center")
        if isinstance(m_ap, (list, tuple)):
            m0 = m_ap[0]
        else:
            m0 = m_ap
        vals = m0.get_values(d)
        peak_local = float(np.nanmax(np.asarray(vals, dtype=np.float64))) if vals is not None else float("nan")
    except Exception:  # noqa: BLE001
        peak_local = float("nan")

    return flux_net, sky_pp, peak_local


# ---------------------------------------------------------------------------
# ALG-3: Temporal binning of comparison ensemble (MNRAS 2023)
# ---------------------------------------------------------------------------


def temporal_bin_comp_lc(
    comp_lc: dict[str, np.ndarray],
    comp_quality: dict[str, dict],
    all_frames: pd.DataFrame,
    *,
    window: int = 0,
    enabled: bool = True,
) -> dict[str, np.ndarray]:
    """Optimized temporal binning of comparison star measurements (MNRAS 2023, 526, 3482).

    Reference: Broeg-Bischoff & Dreizler (2023) MNRAS 526, 3482-3489 —
    'Optimised temporal binning of comparison star measurements
    for differential photometry'

    Applies rolling-window median smoothing to comp star mag_inst series,
    reducing high-frequency shot noise before ensemble normalization.
    Target star is never touched — real variability is preserved.

    Args:
        comp_lc:       dict[catalog_id → mag_inst array, length=n_frames]
        comp_quality:  from check_comparison_stability() — excluded comps skipped;
                       empty dict → all comps in comp_lc are active
        all_frames:    concat DataFrame with 'catalog_id' + 'bjd' columns (reserved)
        window:        smoothing window in frames (0 = auto-optimize [3,5,7,9,11])
        enabled:       False → return original comp_lc unchanged

    Returns:
        dict[catalog_id → smoothed mag_inst array] (same keys, same length)
    """
    _ = all_frames  # frame-ordered LC; BJD-aware binning reserved for later
    if not enabled or len(comp_lc) < 3:
        return dict(comp_lc)

    if not comp_quality:
        active_cids = sorted(comp_lc.keys(), key=str)
    else:
        active_cids = sorted(
            (
                cid
                for cid, q in comp_quality.items()
                if q.get("quality") != "excluded" and cid in comp_lc
            ),
            key=str,
        )
    if len(active_cids) < 3:
        return dict(comp_lc)

    def _rolling_median(arr: np.ndarray, w: int) -> np.ndarray:
        """Rolling median with edge fill from original values."""
        if w < 3 or w > len(arr):
            return arr.copy()
        out = arr.copy()
        half = w // 2
        # Safe: guarded by window-size check above; no fix needed.
        for i in range(half, len(arr) - half):
            out[i] = np.nanmedian(arr[i - half : i + half + 1])
        return out

    def _comp_scatter(smoothed: dict[str, np.ndarray]) -> float:
        """Mean std of comp residuals vs ensemble median."""
        matrix = np.column_stack([smoothed[cid] for cid in active_cids])
        ensemble = np.nanmedian(matrix, axis=1)
        residuals = matrix - ensemble[:, np.newaxis]
        return float(np.nanmean(np.nanstd(residuals, axis=0, ddof=1)))

    # Adaptive window cap: more comps = smaller max window
    # (large ensemble already stable; over-smoothing degrades quality)
    n_active = len(active_cids)
    if n_active >= 8:
        max_window = 5
    elif n_active >= 5:
        max_window = 7
    else:
        max_window = 11

    if window == 0:
        candidate_windows = [w for w in (3, 5, 7, 9, 11) if w <= max_window]
        best_w = 1
        best_scatter = _comp_scatter({cid: comp_lc[cid] for cid in active_cids})
        for w in candidate_windows:
            candidate = {cid: _rolling_median(comp_lc[cid], w) for cid in active_cids}
            s = _comp_scatter(candidate)
            if s < best_scatter - 1e-7:
                best_scatter = s
                best_w = w
        opt_window = best_w
    else:
        opt_window = max(1, int(window))

    if opt_window < 3:
        return dict(comp_lc)

    result = dict(comp_lc)
    for cid in active_cids:
        result[cid] = _rolling_median(comp_lc[cid], opt_window)

    LOGGER.debug(
        "[ALG-3 TempBin] window=%d (max=%d), %d comps, %d frames",
        opt_window,
        max_window if window == 0 else window,
        len(active_cids),
        len(next(iter(comp_lc.values()))),
    )
    return result


# ---------------------------------------------------------------------------
# ALG-5: PyTICS iterative comp intercalibration (RASTI 2026)
# ---------------------------------------------------------------------------


def pytics_iterative_weights(
    comp_lc: dict[str, np.ndarray],
    comp_quality: dict[str, dict],
    comp_rms_map: dict[str, float],
    *,
    n_iter: int = 5,
    enabled: bool = True,
) -> dict[str, float]:
    """Iterative comp star intercalibration — PyTICS (RASTI 2026).

    Reference: Marconi et al. (2026) RASTI —
    'PyTICS: an iterative method for photometric light-curve
    intercalibration using comparison stars'

    Algorithm:
        1. Compute per-frame ZP = weighted median of comp_lc
           (weights = 1/rms^2, Broeg 2005 prior)
        2. Per-comp residuals = comp_lc[cid] - ZP_frame
        3. Per-comp scatter = std(residuals) → updated rms_map
        4. Update weights → repeat n_iter times
        5. Return refined comp_rms_map

    Only 'good' and 'suspect' comps participate — 'excluded' stay excluded.
    Returns original comp_rms_map unchanged if enabled=False or < 3 good comps.
    """
    if not enabled:
        return dict(comp_rms_map)

    # Only use non-excluded comps (canonical cid order — LABBE-DET / SEM determinism).
    active_cids = sorted(
        cid
        for cid, q in comp_quality.items()
        if q.get("quality") != "excluded" and cid in comp_lc
    )
    if len(active_cids) < 3:
        return dict(comp_rms_map)

    # Build matrix: rows=frames, cols=comps
    lc_matrix = np.column_stack([comp_lc[cid].astype(float) for cid in active_cids])
    n_frames, _n_comps = lc_matrix.shape

    # Initial weights from comp_rms_map (Broeg prior)
    rms_arr = np.array(
        [float(comp_rms_map.get(cid, 1.0)) for cid in active_cids],
        dtype=float,
    )
    rms_arr = np.where(rms_arr > 0, rms_arr, 1.0)
    initial_rms = rms_arr.copy()

    iteration = 0
    for iteration in range(n_iter):
        weights = 1.0 / (rms_arr**2)
        s = weights.sum()
        if not np.isfinite(s) or s <= 0:
            break
        weights /= s

        # Per-frame ZP = weighted mean across comps
        zp_per_frame = lc_matrix @ weights  # shape: (n_frames,)

        # Per-comp residuals vs ZP
        residuals = lc_matrix - zp_per_frame[:, np.newaxis]  # (n_frames, n_comps)

        # New per-comp RMS from residuals
        new_rms = np.std(residuals, axis=0, ddof=1)
        new_rms = np.where(new_rms > 1e-6, new_rms, rms_arr)

        # Convergence check
        delta = np.abs(new_rms - rms_arr)
        rms_arr = new_rms
        if np.max(delta) < 1e-6:
            LOGGER.debug("[ALG-5 PyTICS] converged at iteration %d", iteration + 1)
            break

    # Build updated map — only update active comps, keep excluded untouched
    updated_map = dict(comp_rms_map)
    for cid, new_rms_val in zip(active_cids, rms_arr, strict=True):
        updated_map[cid] = float(new_rms_val)

    LOGGER.debug(
        "[ALG-5 PyTICS] %d comps, %d frames, %d iter → max_Δrms=%.6f",
        len(active_cids),
        n_frames,
        iteration + 1,
        float(np.max(np.abs(rms_arr - initial_rms))),
    )
    return updated_map


# ---------------------------------------------------------------------------
# KROK 3: Stability check porovnávačiek (Abbeho p2p scatter + MAD)
# ---------------------------------------------------------------------------

# Observed-band / catalog mag before broad Gaia G for SNR-optimal aperture sizing.
_APERTURE_SIZING_MAG_COLS: tuple[str, ...] = (
    "mag",
    "catalog_mag",
    "lc_median_mag",
    "phot_g_mean_mag",
)


def _star_mag_for_aperture_sizing(row: Any) -> float | None:
    """Brightness for SNR aperture table: prefer observed-band ``mag`` over Gaia G."""
    for mag_col in _APERTURE_SIZING_MAG_COLS:
        try:
            if mag_col not in row.index if hasattr(row, "index") else mag_col not in row:
                continue
        except Exception:  # noqa: BLE001
            # EXC-0133: T4 -- Bad mag value on one masterstar row skipped - loop tries next row for aperture sizing (EXCEPT-BULK-2 2026-07-08)
            if isinstance(row, dict) and mag_col not in row:
                continue
        try:
            mv = float(pd.to_numeric(row.get(mag_col) if hasattr(row, "get") else row[mag_col], errors="coerce"))
        except Exception:  # noqa: BLE001
            continue
        if math.isfinite(mv):
            return mv
    return None


def _common_mode_detrend_comp_lc(
    comp_lc: dict[str, np.ndarray],
    comp_bjd: dict[str, np.ndarray] | None,
    *,
    min_frames: int = 20,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Remove shared linear night trend from comp LCs (differential-photometry residual).

    Returns (detrended_lc, detrended_bjd). When detrend is impossible, returns shallow copies.
    """
    if comp_bjd is None or not comp_lc:
        return (
            {
                cid: np.asarray(comp_lc[cid], dtype=np.float64).copy()
                for cid in sorted(comp_lc.keys(), key=str)
            },
            dict(comp_bjd or {}),
        )

    from scipy.stats import linregress as _lr

    _all_bjd: list[np.ndarray] = []
    _all_mag_matrix: list[np.ndarray] = []
    _active_cids: list[str] = []
    for cid in sorted(comp_lc.keys(), key=str):
        lc = comp_lc[cid]
        bjd_arr = comp_bjd.get(cid)
        if bjd_arr is None:
            continue
        m = np.asarray(lc, dtype=np.float64)
        b = np.asarray(bjd_arr, dtype=np.float64)
        ok = np.isfinite(b) & np.isfinite(m)
        if int(ok.sum()) < int(min_frames):
            continue
        bo, mo = b[ok], m[ok]
        order = np.argsort(bo, kind="mergesort")
        _all_bjd.append(bo[order])
        _all_mag_matrix.append(mo[order])
        _active_cids.append(cid)

    if len(_all_mag_matrix) < 2:
        return (
            {
                cid: np.asarray(comp_lc[cid], dtype=np.float64).copy()
                for cid in sorted(comp_lc.keys(), key=str)
            },
            dict(comp_bjd),
        )

    # Prefer longest series; break ties by cid order (already sorted) for determinism.
    _ref_idx = int(np.argmax([len(x) for x in _all_bjd]))
    _ref_bjd = _all_bjd[_ref_idx]
    _stack = []
    for b_arr, m_arr in zip(_all_bjd, _all_mag_matrix, strict=True):
        _stack.append(np.interp(_ref_bjd, b_arr, m_arr))
    _common = np.median(np.vstack(_stack), axis=0)
    _lr_common = _lr(_ref_bjd, _common)

    _detrended_lc: dict[str, np.ndarray] = {}
    _detrended_bjd: dict[str, np.ndarray] = {}
    for cid in sorted(comp_lc.keys(), key=str):
        b = comp_bjd.get(cid)
        m = comp_lc.get(cid)
        if b is None or m is None:
            continue
        b = np.asarray(b, dtype=np.float64)
        m = np.asarray(m, dtype=np.float64)
        ok = np.isfinite(b) & np.isfinite(m)
        m_detrended = m.copy()
        m_detrended[ok] = m[ok] - (_lr_common.slope * b[ok] + _lr_common.intercept) + float(_common.mean())
        _detrended_lc[cid] = m_detrended
        _detrended_bjd[cid] = b

    return _detrended_lc, _detrended_bjd


def _comp_lc_frame_ensemble_residual(comp_lc: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per-frame ensemble-median subtraction (differential residual per comp).

    Same principle as Phase-1 ``comp_rms`` (flux / bin median): comp intrinsic scatter
    is only visible after the shared per-frame signal is removed.

    Uses sorted cid order and truncates to a common length only after sorting keys so
    PYTHONHASHSEED / dict insertion order cannot change the residual field.
    """
    if not comp_lc:
        return {}
    cids = sorted(comp_lc.keys(), key=str)
    arrays = [np.asarray(comp_lc[c], dtype=np.float64) for c in cids]
    lengths = [int(a.size) for a in arrays if a.size > 0]
    if not lengths:
        return {c: np.array([], dtype=np.float64) for c in cids}
    n = int(min(lengths))
    if n < 3:
        return {c: a[:n].copy() for c, a in zip(cids, arrays, strict=True)}
    stack = np.column_stack([a[:n] for a in arrays])
    med = np.nanmedian(stack, axis=1)
    out: dict[str, np.ndarray] = {}
    for j, cid in enumerate(cids):
        out[cid] = stack[:, j] - med
    return out


def compute_lc_rms_ooe(
    mag_calib: np.ndarray,
    flags: Sequence[str],
    *,
    brightest_frac: float = 0.33,
) -> float:
    """Out-of-eclipse / brightest-tertile scatter (precision metric for variables).

    For eclipsing variables the faintest frames are in eclipse; the brightest ``brightest_frac``
  fraction approximates OOE scatter without requiring a period.
    """
    m = np.asarray(mag_calib, dtype=np.float64)
    if m.size < 3:
        return float("nan")
    fl = [str(f).strip().lower() for f in flags]
    if len(fl) == m.size:
        ok = np.array([f == "normal" for f in fl], dtype=bool)
    else:
        ok = np.ones(m.size, dtype=bool)
    finite = m[ok & np.isfinite(m)]
    if finite.size < 3:
        finite = m[np.isfinite(m)]
    if finite.size < 3:
        return float("nan")
    frac = float(brightest_frac)
    if not (0.0 < frac < 1.0):
        frac = 0.33
    thr = float(np.quantile(finite, frac))
    ooe = finite[finite <= thr]
    if ooe.size < 3:
        ooe = finite
    return float(np.std(ooe))


def check_comparison_stability(
    comp_lc: dict[str, np.ndarray],
    *,
    comp_rms_map: dict[str, float] | None = None,
    comp_bjd: dict[str, np.ndarray] | None = None,
    n_comp_min: int = 3,
    outlier_sigma: float = 3.0,
    max_comp_slope_mmag_hr: float = 5.0,
    comp_slope_significance_k: float = 3.0,
    common_mode_detrend: bool = True,
    stability_run_flags: dict[str, Any] | None = None,
) -> dict[str, dict]:
    """Krok 3: Stability check porovnávačiek.

    Abbeho point-to-point scatter on **common-mode-detrended** comp residuals:
        rms_p2p = std(diff(mag_resid)) / sqrt(2)

    Shared atmospheric drift is removed before p2p/MAD (same differential logic as ensemble).

    Returns:
        dict {catalog_id: {"rms_p2p": float, "lc_rms": float, "quality": str, "p2p_threshold": float}}
        quality: "good" / "suspect" / "excluded"; záznamy sú zoradené (good → suspect → excluded, v rámci good podľa rms_p2p).
    """
    result: dict[str, dict[str, Any]] = {}

    _detrended_lc: dict[str, np.ndarray] = {}
    _detrended_bjd: dict[str, np.ndarray] = {}
    _residual_lc = _comp_lc_frame_ensemble_residual(comp_lc)
    if common_mode_detrend and comp_bjd is not None:
        _detrended_lc, _detrended_bjd = _common_mode_detrend_comp_lc(_residual_lc, comp_bjd)
        if stability_run_flags is not None and len(_detrended_lc) >= 2:
            stability_run_flags["common_mode_detrend_applied"] = True
            stability_run_flags["frame_ensemble_residual"] = True
            logging.info(
                "[STABILITY] Frame-ensemble residual + CM detrend before p2p: %d comps",
                len(_detrended_lc),
            )
    else:
        _detrended_lc = {cid: np.asarray(lc, dtype=np.float64).copy() for cid, lc in _residual_lc.items()}
        _detrended_bjd = dict(comp_bjd or {})
        if stability_run_flags is not None:
            stability_run_flags["frame_ensemble_residual"] = True

    # Vypočítaj metriky na per-frame differential reziduálnych radách
    for cid, lc in comp_lc.items():
        resid = _detrended_lc.get(cid, lc)
        finite = np.asarray(resid, dtype=np.float64)[np.isfinite(resid)]
        if len(finite) < 3:
            result[cid] = {
                "rms_p2p": float("nan"),
                "lc_rms": float("nan"),
                "quality": "excluded",
                "note": "few_frames",
            }
            continue
        lc_rms = float(np.std(finite))
        diff = np.diff(finite)
        rms_p2p = float(np.std(diff) / math.sqrt(2)) if len(diff) > 1 else float("nan")
        result[cid] = {"rms_p2p": rms_p2p, "lc_rms": lc_rms, "quality": "good"}

    # Ak hviezda má comp_rms≈0 z Phase 1 → označ ako suspect (pravdepodobný isolated-bin normalizačný artefakt).
    if comp_rms_map:
        for cid in result:
            try:
                phase1_rms = float(comp_rms_map.get(cid, float("nan")))
            except Exception:  # noqa: BLE001
                phase1_rms = float("nan")
            if math.isfinite(phase1_rms) and phase1_rms < 1e-6:
                result[cid]["quality"] = "suspect"
                result[cid]["note"] = "isolated_bin"

    # MAD filter na rms_p2p
    valid_p2p = np.asarray(
        [v["rms_p2p"] for v in result.values() if math.isfinite(v["rms_p2p"])],
        dtype=np.float64,
    )
    threshold = float("nan")
    if valid_p2p.size >= 2:
        med = float(np.median(valid_p2p))
        sigma = _mad_sigma(valid_p2p)
        threshold = med + outlier_sigma * sigma
        # Absolútny strop — comp hviezda s p2p RMS > 0.10 mag je vždy zlá
        _ABS_MAX_P2P = 0.10
        if math.isfinite(threshold):
            threshold = min(float(threshold), _ABS_MAX_P2P)

        n_good = sum(
            1
            for v in result.values()
            if v["quality"] == "good" and math.isfinite(v["rms_p2p"]) and v["rms_p2p"] <= threshold
        )

        for cid, info in result.items():
            if not math.isfinite(info["rms_p2p"]):
                continue
            if info["rms_p2p"] > threshold:
                # Ak by sme mali menej ako n_comp_min good, označ ako suspect nie excluded
                if n_good < n_comp_min:
                    result[cid]["quality"] = "suspect"
                    result[cid]["note"] = "outlier (kept: n_good<min)"
                else:
                    result[cid]["quality"] = "excluded"
                    result[cid]["note"] = (
                        f"outlier (p2p={info['rms_p2p']:.4f} > thr={threshold:.4f})"
                    )

    # Slope filter: exclude comps with a night-long linear trend (slow drifts pass p2p RMS).
    if comp_bjd is not None and max_comp_slope_mmag_hr > 0:
        from scipy.stats import linregress  # lazy import — scipy already in deps

        n_good_slope = sum(1 for v in result.values() if v["quality"] == "good")
        for cid, info in result.items():
            if info["quality"] == "excluded":
                continue
            bjd_arr = _detrended_bjd.get(cid) if _detrended_bjd else comp_bjd.get(cid)
            mag_arr = _detrended_lc.get(cid) if _detrended_lc else comp_lc.get(cid)
            if bjd_arr is None or mag_arr is None:
                continue
            ok = np.isfinite(bjd_arr) & np.isfinite(mag_arr)
            if int(ok.sum()) < 20:
                continue
            lr = linregress(bjd_arr[ok], mag_arr[ok])
            slope_mmag_hr = abs(float(lr.slope)) * 1000.0 / 24.0
            _se = float(getattr(lr, "stderr", float("nan")))
            slope_sig = (
                abs(float(lr.slope)) / _se
                if math.isfinite(_se) and _se > 0
                else float("inf")
            )
            if slope_mmag_hr > float(max_comp_slope_mmag_hr) and slope_sig >= float(
                comp_slope_significance_k
            ):
                logging.info(
                    "Comp %s slope-excluded: %.1f mmag/hr (%.1fσ) > %s mmag/hr @ %sσ",
                    cid,
                    slope_mmag_hr,
                    slope_sig,
                    max_comp_slope_mmag_hr,
                    comp_slope_significance_k,
                )
                info["slope_mmag_hr"] = slope_mmag_hr
                info["slope_sigma"] = slope_sig
                if n_good_slope < n_comp_min:
                    info["quality"] = "suspect"
                    note = f"slope={slope_mmag_hr:.1f} mmag/hr ({slope_sig:.1f}σ, kept: n_good<min)"
                else:
                    info["quality"] = "excluded"
                    note = f"slope={slope_mmag_hr:.1f} mmag/hr ({slope_sig:.1f}σ)"
                if info.get("note"):
                    info["note"] = f"{info['note']}; {note}"
                else:
                    info["note"] = note

    for info in result.values():
        info["p2p_threshold"] = threshold

    n_good_final = sum(1 for v in result.values() if v["quality"] == "good")
    thr_log = f"{threshold:.5f}" if math.isfinite(threshold) else "N/A"
    logging.info(
        f"[FÁZA 2A] Stability check: {n_good_final}/{len(result)} good comp "
        f"(p2p threshold={thr_log})"
    )

    # Zoradenie: good (podľa rms_p2p), suspect, excluded — poradie v ensemble / PNG tabuľke
    sorted_result = dict(
        sorted(
            result.items(),
            key=lambda x: (
                0 if x[1]["quality"] == "good" else 1 if x[1]["quality"] == "suspect" else 2,
                x[1]["rms_p2p"] if math.isfinite(x[1].get("rms_p2p", float("nan"))) else 999.0,
            ),
        )
    )
    return sorted_result


def compute_aperture_correction(
    comp_df: pd.DataFrame,
    frame_results: Sequence[pd.DataFrame],
    min_ref_stars: int = 3,
    max_contamination: float = 0.15,
    max_scatter_mag: float = 0.03,
) -> dict[str, Any]:
    """Metóda B: medián ΔM_corr = mag_large − mag_small medzi referenčnými comp cez framy.

    ``frame_results``: jeden DataFrame na snímku (výstup ``read_flux_from_csv``), riadky = hviezdy,
    kľúč hviezdy je ``catalog_id`` (zhodné s Fázou 1).

    Ref hviezdy pre ΔM_corr: preferuj T1, ak < min_ref_stars doplň T2; potom contamination filter.
    """
    empty_out: dict[str, Any] = {
        "ok": False,
        "delta_m_corr": None,
        "scatter_mag": None,
        "n_ref_stars": 0,
        "ref_star_ids": [],
        "reason": "",
    }

    def _fail(reason: str) -> dict[str, Any]:
        out = dict(empty_out)
        out["reason"] = reason
        return out

    if comp_df is None or getattr(comp_df, "empty", True):
        return _fail("no_comp_df")

    if not frame_results:
        return _fail("no_frames")

    # --- KROK A: referenčné hviezdy ---
    df = comp_df.copy()
    if "comp_tier" not in df.columns:
        return _fail("no_comp_tier")
    df["comp_tier"] = pd.to_numeric(df["comp_tier"], errors="coerce").fillna(4).astype(int)
    ref_t1 = df[df["comp_tier"] == 1].copy()
    ref_t2 = df[df["comp_tier"] == 2].copy()

    ref_stars = ref_t1
    if int(len(ref_stars)) < int(min_ref_stars):
        ref_stars = pd.concat([ref_t1, ref_t2], ignore_index=True)

    # Aplikuj contamination filter
    if "contamination_idx" in ref_stars.columns:
        ref_stars = ref_stars[
            ref_stars["contamination_idx"].apply(
                lambda x: float(x) <= float(max_contamination) if pd.notna(x) else False
            )
        ]
    else:
        return _fail("no_contamination_idx")

    # Aplikuj comp_rms filter
    if "comp_rms" in ref_stars.columns:
        cr = pd.to_numeric(ref_stars["comp_rms"], errors="coerce")
        ref_stars = ref_stars[np.isfinite(cr.to_numpy(dtype=float)) & (cr.to_numpy(dtype=float) > 0)].copy()
    else:
        return _fail("no_comp_rms")

    if int(len(ref_stars)) < int(min_ref_stars):
        return _fail("insufficient_ref_stars")

    ref_ids = [
        _normalize_gaia_id(r.get("catalog_id", r.get("name", "")))
        for _, r in ref_stars.iterrows()
    ]
    ref_ids = [x for x in ref_ids if x]
    ref_ids = list(dict.fromkeys(ref_ids))
    if len(ref_ids) < int(min_ref_stars):
        return _fail("insufficient_ref_stars")

    # --- KROK B: medián Δm per ref hviezda cez framy ---
    delta_per_star: list[float] = []
    ref_used: list[str] = []

    for cid in ref_ids:
        dms_frame: list[float] = []
        for df_fr in frame_results:
            if df_fr is None or getattr(df_fr, "empty", True):
                continue
            if "catalog_id" not in df_fr.columns:
                continue
            sub = df_fr.loc[df_fr["catalog_id"].astype(str).map(_normalize_gaia_id).eq(cid)]
            if sub.empty:
                continue
            row0 = sub.iloc[0]
            fs = pd.to_numeric(row0.get("flux_small"), errors="coerce")
            fl = pd.to_numeric(row0.get("flux_large"), errors="coerce")
            try:
                fsv = float(fs)
                flv = float(fl)
            except Exception:  # noqa: BLE001
                # EXC-0134: T4 -- Non-finite small/large aperture flux pair skipped in per-frame delta-mag collection (EXCEPT-BULK-2 2026-07-08)
                continue
            if not (math.isfinite(fsv) and math.isfinite(flv) and fsv > 0 and flv > 0):
                continue
            mag_s = -2.5 * math.log10(fsv)
            mag_l = -2.5 * math.log10(flv)
            if not (math.isfinite(mag_s) and math.isfinite(mag_l)):
                continue
            dms_frame.append(float(mag_l - mag_s))

        if not dms_frame:
            continue
        dm_med = float(np.nanmedian(np.asarray(dms_frame, dtype=np.float64)))
        if not math.isfinite(dm_med):
            continue
        delta_per_star.append(dm_med)
        ref_used.append(cid)

    # --- KROK C ---
    if len(ref_used) < int(min_ref_stars):
        return _fail("insufficient_usable_frames")

    dm_arr = np.asarray(delta_per_star, dtype=np.float64)
    delta_m_corr = float(np.median(dm_arr))
    scatter_mag = float(np.median(np.abs(dm_arr - delta_m_corr)))

    if scatter_mag > float(max_scatter_mag):
        return _fail("scatter_too_high")

    return {
        "ok": True,
        "delta_m_corr": delta_m_corr,
        "scatter_mag": scatter_mag,
        "n_ref_stars": len(ref_used),
        "ref_star_ids": list(ref_used),
        "reason": "ok",
    }


# ---------------------------------------------------------------------------
# KROK 4: Ensemble normalizácia
# ---------------------------------------------------------------------------


def ensemble_member_ids(
    comp_quality: dict[str, dict],
    comp_rms_map: dict[str, float] | None = None,
    *,
    n_comp_min: int = 3,
    n_comp_max: int = 10,
) -> set[str]:
    """Catalog ids selected for Phase-2A ``ensemble_normalize`` (check-star must be outside)."""
    comp_rms_map = comp_rms_map or {}
    p2p_thr = float("nan")
    for q in comp_quality.values():
        t = q.get("p2p_threshold")
        if t is not None and math.isfinite(float(t)):
            p2p_thr = float(t)
            break
    usable_all = [
        cid for cid, q in comp_quality.items() if q.get("quality") in ("good", "suspect")
    ]
    usable_sorted = sorted(
        usable_all,
        key=lambda c: (
            0 if comp_quality[c].get("quality") == "good" else 1,
            float(comp_rms_map.get(c, float("inf"))),
            str(c),
        ),
    )
    selected: list[str] = []
    for cid in usable_sorted:
        if len(selected) >= int(n_comp_max):
            break
        p2p = float(comp_quality[cid].get("rms_p2p", float("nan")))
        if (
            len(selected) < int(n_comp_min)
            or (math.isfinite(p2p_thr) and math.isfinite(p2p) and p2p < p2p_thr)
            or not math.isfinite(p2p_thr)
        ):
            selected.append(cid)
    return {str(c) for c in selected[: int(n_comp_max)]}


def ensemble_normalize(
    target_mag_inst: np.ndarray,
    comp_mag_inst: dict[str, np.ndarray],
    comp_catalog_mag: dict[str, float],
    comp_quality: dict[str, dict],
    *,
    comp_rms_map: dict[str, float] | None = None,
    comp_tier_map: dict[str, int] | None = None,
    tier_weights: dict[int, float] | None = None,
    n_comp_min: int = 3,
    n_comp_max: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Krok 4: Ensemble normalizácia per snímka.

    ``mag_ensemble`` = ``-2.5*log10(sum 10**(-0.4*m_comp))`` (súčet fluxov ako AIJ ``tot_C_cnts``).

    ``delta_mag = mag_inst(target) - mag_ensemble`` (tvar voči súčtu fluxov ako AIJ).

    ``mag_calib`` musí mať iný zeropoint ako samotný ``median(katalóg)``: súčet fluxov dáva
    ``m_ensemble = -2.5 log10(Σ F_i)``, čo pri n comps zhruba zodpovedá ``m_i - 2.5 log10(n)``
    pri podobných ``m_i`` — pripočítanie len ``median(cat)`` by posunulo krivku o ~``2.5 log10(n)``
    mag. Preto ``mag_calib = mag_inst(target) + median_j(cat_mag_j - mag_inst_j)`` (klasický
    diferenciálny posun); ``delta_mag`` ostáva oproti ``mag_ensemble`` z AIJ súčtu.

    Výber comps: zoradenie podľa ``comp_rms`` (Fáza 1), prvých ``n_comp_min`` vždy;
    ďalšie len ak ``rms_p2p`` < ``p2p_threshold`` z stability; max ``n_comp_max``.

    Returns:
        (mag_calib, delta_mag, ensemble_scatter) — arrays dĺžky n_frames
    """
    n_frames = len(target_mag_inst)
    mag_calib = np.full(n_frames, float("nan"))
    delta_mag = np.full(n_frames, float("nan"))
    ensemble_scatter = np.full(n_frames, float("nan"))

    comp_rms_map = comp_rms_map or {}
    comp_tier_map = comp_tier_map or {}
    tier_weights = tier_weights or {1: 1.0, 2: 0.85, 3: 0.50, 4: 0.25}

    p2p_thr = float("nan")
    # Canonical: prefer a shared threshold; take first finite from sorted cids (not dict.values order).
    for _cid in sorted(comp_quality.keys(), key=str):
        q = comp_quality[_cid]
        t = q.get("p2p_threshold")
        if t is not None and math.isfinite(float(t)):
            p2p_thr = float(t)
            break

    # Ensemble: good aj suspect; excluded nie. (RMS sa používa na výber poradia, nie na váhu fluxu.)
    usable_all = [
        cid for cid, q in comp_quality.items() if q.get("quality") in ("good", "suspect")
    ]
    usable_sorted = sorted(
        usable_all,
        key=lambda c: (
            0 if comp_quality[c].get("quality") == "good" else 1,
            float(comp_rms_map.get(c, float("inf"))),
            str(c),
        ),
    )

    selected: list[str] = []
    for cid in usable_sorted:
        if len(selected) >= n_comp_max:
            break
        p2p = float(comp_quality[cid].get("rms_p2p", float("nan")))
        if (
            len(selected) < n_comp_min
            or (math.isfinite(p2p_thr) and math.isfinite(p2p) and p2p < p2p_thr)
            or not math.isfinite(p2p_thr)
        ):
            selected.append(cid)

    good_ids = selected[:n_comp_max]
    if not good_ids:
        log_event("ensemble_normalize: no valid comp stars — returning all-NaN LC")
        return mag_calib, delta_mag, ensemble_scatter

    cat_mags = np.asarray([comp_catalog_mag.get(cid, float("nan")) for cid in good_ids])
    cat_offset = float(np.nanmedian(cat_mags))
    logging.debug(
        f"[FÁZA 2A] Ensemble: {len(good_ids)} comps (good+suspect), "
        f"catalog_mag median={cat_offset:.3f} (mag_calib zeropoint = median(cat−inst) per frame)"
    )

    # Per-comp across-night reference = median of its own instrumental magnitudes. Referencing each
    # comp to its OWN reference (rather than comparing comps' absolute instrumental mags) cancels the
    # comps' brightness AND constant colour offsets, so the per-frame ``ensemble_scatter`` below is the
    # genuine zeropoint scatter (Honeycutt 1992), not the comps' brightness spread. Used ONLY for the
    # error model; does not touch mag_calib/delta_mag/ens_med.
    comp_ref_map: dict[str, float] = {}
    for cid in good_ids:
        arr_cid = comp_mag_inst.get(cid)
        if arr_cid is None:
            continue
        _vals = np.asarray(arr_cid, dtype=np.float64)
        _fin = _vals[np.isfinite(_vals)]
        if _fin.size:
            comp_ref_map[cid] = float(np.median(_fin))

    for i in range(n_frames):
        comp_pairs: list[tuple[str, float]] = []
        for cid in good_ids:
            if cid not in comp_mag_inst:
                continue
            try:
                mv = float(comp_mag_inst[cid][i])
            except Exception as exc:  # noqa: BLE001
                logging.error("[EXC-0135] One comp's inst mag on one frame skipped - ensemble normalization ignores that comp for...: %s", exc)
                continue
            if math.isfinite(mv):
                comp_pairs.append((cid, mv))

        if (not comp_pairs) or not math.isfinite(target_mag_inst[i]):
            continue

        comp_vals = np.asarray([m for _, m in comp_pairs], dtype=np.float64)

        # Combination = AIJ/Honeycutt flux sum (tot_C_cnts); Broeg 1/rms² applies to selection
        # ordering + catalog zeropoint offset below, not to ens_med.
        # Priamy súčet fluxov — rovnaká metóda ako AIJ (tot_C_cnts = C2+C3+C4).
        # Váhovaný priemer 1/rms² deformuje extinkčný slope ensemble → záporný slope.
        comp_fluxes_list: list[float] = [10 ** (-0.4 * m) for _, m in comp_pairs]
        f_arr = np.asarray(comp_fluxes_list, dtype=np.float64)

        ens_flux_sum = float(np.sum(f_arr))
        if math.isfinite(ens_flux_sum) and ens_flux_sum > 0:
            ens_med = float(-2.5 * math.log10(ens_flux_sum))
        else:
            ens_med = float(np.median(comp_vals))

        # Per-point ensemble zeropoint uncertainty (Honeycutt 1992 PASP 104:435): the standard error
        # of the comps' per-frame residuals about the ensemble mean, where each residual is the comp's
        # deviation from its OWN across-night reference (``comp_ref_map``). This cancels the comps'
        # brightness/colour spread — the previous ``np.std(comp_vals)`` on raw instrumental mags
        # injected a fixed ~comp-brightness-difference floor (the inflated-err bug). Small n: a near-
        # zero residual SEM leaves err = photon base (the floor); we do not use comp_rms here (it is
        # dropped at the err-assembly site to avoid double-counting this same ensemble term).
        comp_resid = [
            (m - comp_ref_map[cid_j])
            for cid_j, m in comp_pairs
            if cid_j in comp_ref_map and math.isfinite(comp_ref_map[cid_j])
        ]
        if len(comp_resid) >= 2:
            from sigma_floor_core import ensemble_sem_mag_from_residuals  # noqa: PLC0415

            ensemble_scatter[i] = float(ensemble_sem_mag_from_residuals(comp_resid))
        else:
            ensemble_scatter[i] = 0.0
        delta_mag[i] = target_mag_inst[i] - ens_med

        # Honeycutt (1992) PASP 104:435 — per-frame ensemble zeropoint from constant comps.
        # mag_calib[i] = target_inst + ZP_frame; delta_mag[i] = target_inst - ens_med (AIJ flux sum).
        # Hence mag_calib - delta_mag = ZP_frame + ens_med (frame-dependent; not identical zeropoints).
        # ``delta_mag + median(cat)`` by bolo nesúladné s ``ens_med`` zo súčtu fluxov (−2.5 log ΣF).
        zp_offs: list[float] = []
        zp_vals: list[float] = []
        weights: list[float] = []
        for cid_j, m_j in comp_pairs:
            cm_j = float(comp_catalog_mag.get(cid_j, float("nan")))
            if math.isfinite(cm_j) and math.isfinite(m_j):
                d = float(cm_j - m_j)
                zp_offs.append(d)
                rms_j = float(comp_rms_map.get(cid_j, float("nan")))
                if math.isfinite(rms_j) and rms_j > 1e-6:
                    zp_vals.append(d)
                    tier_j = int(comp_tier_map.get(cid_j, 4))
                    tw = float(tier_weights.get(tier_j, 0.25))
                    if not (math.isfinite(tw) and tw > 0):
                        tw = 0.25
                    # Broeg, Fernandez & Neuhäuser (2005) AN 326:134
                    # Optimal weights: w_i = 1 / sigma_i^2 (inverse variance weighting)
                    weights.append((1.0 / (rms_j**2)) * tw)
        if weights:
            w = np.asarray(weights, dtype=np.float64)
            z = np.asarray(zp_vals, dtype=np.float64)
            if len(z) >= 4:
                # DAOPHOT/IRAF standard: iterative sigma-clip on ZP residuals
                # Stetson (1987) PASP 99:191
                _med = float(np.nanmedian(z))
                _mad = float(np.nanmedian(np.abs(z - _med)))
                _sigma = max(_mad / _MAD_CONSISTENCY, 1e-6)
                _keep = np.abs(z - _med) <= 3.0 * _sigma
                if _keep.sum() >= 2:
                    if _keep.sum() < len(z):
                        logging.debug(
                            "[ZP] Frame sigma-clip: %d/%d comps kept "
                            "(rejected %d outliers, σ=%.4f)",
                            int(_keep.sum()),
                            len(z),
                            int((~_keep).sum()),
                            _sigma,
                        )
                    z = z[_keep]
                    w = w[_keep]
            if len(z) >= 2 and float(np.sum(w)) > 0:
                mag_calib[i] = target_mag_inst[i] + float(np.sum(w * z) / np.sum(w))
            elif zp_offs:
                mag_calib[i] = target_mag_inst[i] + float(
                    np.nanmedian(np.asarray(zp_offs, dtype=np.float64))
                )
            else:
                mag_calib[i] = delta_mag[i] + cat_offset
        elif zp_offs:
            # fallback to median if we don't have usable RMS weights
            mag_calib[i] = target_mag_inst[i] + float(np.nanmedian(np.asarray(zp_offs, dtype=np.float64)))
        else:
            mag_calib[i] = delta_mag[i] + cat_offset

    return mag_calib, delta_mag, ensemble_scatter


def _ensemble_scatter_by_source_file(
    all_frames: pd.DataFrame,
    target_cid: str,
    ensemble_scatter: np.ndarray | None,
) -> dict[str, float]:
    """Map proc CSV filename -> ensemble_scatter (G2-F004 epoch-level join key).

    Must use the same ``source_file`` sort as ``_get_lc`` so indices of
    ``ensemble_scatter`` (built from sorted target LC) map to the correct files.
    """
    if ensemble_scatter is None:
        return {}
    sc = np.asarray(ensemble_scatter, dtype=np.float64)
    if sc.size == 0:
        return {}
    sub = all_frames[all_frames["catalog_id"] == target_cid]
    if sub.empty:
        return {}
    if "source_file" in sub.columns:
        sub = sub.sort_values(["source_file"], kind="mergesort")
    out: dict[str, float] = {}
    for i, sf in enumerate(sub["source_file"].astype(str).str.strip().tolist()):
        if i >= len(sc):
            break
        key = str(sf)
        if key:
            out[key] = float(sc[i])
    return out


def _combine_err_with_ensemble_scatter_keyed(
    err_photon: np.ndarray,
    source_files: list[str] | np.ndarray,
    scatter_by_file: dict[str, float],
    *,
    sigma_sys_mag: float = 0.0,
    target_name: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """Join photon ``err`` with ensemble scatter by EXACT ``source_file`` match (G2-F004).

    Domain contract: ``err_photon`` is relative flux (err/flux); ``scatter_by_file`` values are
    ensemble SEM in magnitudes (Honeycutt residual std/sqrt(n) with c4 correction from
    ``ensemble_normalize``). Per-rig ``sigma_sys_mag`` (mag) is added in quadrature after SEM.

    err_total^2 = err_photon^2 + sem_rel^2 + sigma_sys_rel^2 (relative-flux domain).

    Matched epoch, finite scatter -> quadrature with SEM (+ floor when configured).
    Matched epoch, NaN scatter -> scatter treated as 0.0 (photon-only + floor), same as legacy
    ``np.where(isfinite, scatter, 0.0)``.
    Unmatched ``source_file`` -> photon-only err (+ floor), ``err_scatter_unmatched`` True,
    WARNING logged.
    """
    from sigma_floor_core import combine_production_err_rel  # noqa: PLC0415

    err_out = np.asarray(err_photon, dtype=np.float64).copy()
    unmatched = np.zeros(len(err_out), dtype=bool)
    _floor = float(sigma_sys_mag) if math.isfinite(float(sigma_sys_mag)) and float(sigma_sys_mag) > 0 else 0.0

    n_unmatched = 0
    for i, sf in enumerate(np.asarray(source_files, dtype=object)):
        key = str(sf).strip()
        sc_mag = 0.0
        if key and scatter_by_file and key in scatter_by_file:
            sc = float(scatter_by_file[key])
            sc_mag = float(sc) if math.isfinite(sc) else 0.0
        elif scatter_by_file:
            unmatched[i] = True
            n_unmatched += 1
        ep = float(err_out[i]) if math.isfinite(err_out[i]) else float("nan")
        err_out[i] = combine_production_err_rel(ep, sc_mag, sigma_sys_mag=_floor)

    if n_unmatched > 0:
        logging.warning(
            "[G2-F004] %s: %d/%d epochs missing ensemble_scatter for source_file "
            "— photon-only err kept",
            target_name or "?",
            n_unmatched,
            len(err_out),
        )
    return err_out, unmatched


# ---------------------------------------------------------------------------
# Color term (BP-RP) — globálny shift na noc
# ---------------------------------------------------------------------------


def fit_color_term_c1(
    comp_mag_inst: dict[str, np.ndarray],
    comp_catalog_mag: dict[str, float],
    comp_bp_rp: dict[str, float],
    comp_quality: dict[str, dict],
    *,
    min_comp: int = 5,
    sigma_clip_sigma: float = 3.0,
) -> tuple[float, float, int]:
    """
    Fituje color term koeficient c1 z COMP hviezd.

    Pre každú good/suspect COMP hviezdu:
      x_i = bp_rp_i - median(bp_rp všetkých použitých COMP)
      y_i = median(cat_mag_i - inst_mag_i)  [cez všetky framy]

    Lineárny fit: y = c1 * x + ZP_offset

    Returns: (c1, c1_stderr, n_comp_used)
    Pri chybe alebo málo COMP: (0.0, nan, 0)
    """
    try:
        min_comp_i = int(min_comp)
    except Exception:  # noqa: BLE001
        min_comp_i = 5
    min_comp_i = max(2, min_comp_i)

    usable = [cid for cid, q in comp_quality.items() if q.get("quality") in ("good", "suspect")]
    xs: list[float] = []
    ys: list[float] = []
    bp_vals: list[float] = []

    for cid in usable:
        bp = float(comp_bp_rp.get(cid, float("nan")))
        if not math.isfinite(bp):
            continue
        if cid not in comp_mag_inst:
            continue
        inst = np.asarray(comp_mag_inst[cid], dtype=np.float64)
        finite = inst[np.isfinite(inst)]
        if finite.size < min_comp_i:
            continue
        cat = float(comp_catalog_mag.get(cid, float("nan")))
        if not math.isfinite(cat):
            continue
        y = float(np.nanmedian(cat - finite))
        if not math.isfinite(y):
            continue
        bp_vals.append(bp)
        ys.append(y)
        # x will be centered later once we know bp median

    n0 = len(ys)
    if n0 < min_comp_i:
        return 0.0, float("nan"), 0

    bp_med = float(np.median(np.asarray(bp_vals, dtype=np.float64)))
    for bp in bp_vals:
        xs.append(float(bp) - bp_med)

    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)

    # Initial fit for sigma clipping
    p0 = _safe_polyfit(x, y, 1)
    if p0 is None:
        return 0.0, float("nan"), 0
    try:
        c1_init = float(p0[0])
        zp_init = float(p0[1])
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("[PHOT] color-term c1 fit failed; no correction: %s", exc)
        return 0.0, float("nan"), 0

    resid = y - (c1_init * x + zp_init)
    sig = _mad_sigma(resid)
    if not math.isfinite(sig) or sig <= 0:
        # No robust scatter estimate → keep all
        mask = np.ones_like(resid, dtype=bool)
    else:
        mask = np.abs(resid) <= float(sigma_clip_sigma) * float(sig)

    n_removed = int((~mask).sum())
    x_cl = x[mask]
    y_cl = y[mask]
    if x_cl.size < min_comp_i:
        return 0.0, float("nan"), 0

    fit_cl = _safe_polyfit(x_cl, y_cl, 1, cov=True)
    if fit_cl is None:
        return 0.0, float("nan"), 0
    try:
        coeffs, cov = fit_cl
        c1 = float(coeffs[0])
        c1_stderr = float(math.sqrt(float(cov[0, 0]))) if cov is not None else float("nan")
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("[PHOT] color-term c1 fit (inner) failed; no correction: %s", exc)
        return 0.0, float("nan"), 0

    bp_min = float(np.min(np.asarray(bp_vals, dtype=np.float64)))
    bp_max = float(np.max(np.asarray(bp_vals, dtype=np.float64)))
    logging.info(
        "[COLOR TERM] c1=%.4f ± %.4f, bp_rp_range=[%.2f, %.2f], n_comp=%s, sigma_clip_removed=%s",
        c1,
        c1_stderr,
        bp_min,
        bp_max,
        int(x_cl.size),
        int(n_removed),
    )
    return c1, c1_stderr, int(x_cl.size)


def apply_color_term(
    mag_calib: np.ndarray,
    target_bp_rp: float,
    comp_bp_rp: dict[str, float],
    comp_quality: dict[str, dict],
    c1: float,
) -> tuple[np.ndarray, float, float]:
    """
    Aplikuje color term korekciu na kalibrovanú krivku.

    Vzorec:
      bp_rp_comp_med = median(bp_rp použitých COMP)
      ct_correction  = c1 * (target_bp_rp - bp_rp_comp_med)
      mag_calib_ct   = mag_calib + ct_correction

    Returns: (mag_calib_ct, ct_correction, bp_rp_comp_med)
    """
    if mag_calib is None:
        return np.asarray([], dtype=np.float64), 0.0, float("nan")
    base = np.asarray(mag_calib, dtype=np.float64)
    if (not math.isfinite(float(c1))) or float(c1) == 0.0 or (not math.isfinite(float(target_bp_rp))):
        return base.copy(), 0.0, float("nan")

    usable = [cid for cid, q in comp_quality.items() if q.get("quality") in ("good", "suspect")]
    bps = [
        float(comp_bp_rp.get(cid, float("nan")))
        for cid in usable
        if math.isfinite(float(comp_bp_rp.get(cid, float("nan"))))
    ]
    if not bps:
        return base.copy(), 0.0, float("nan")
    bp_med = float(np.median(np.asarray(bps, dtype=np.float64)))
    corr = float(c1) * (float(target_bp_rp) - float(bp_med))
    out = base + float(corr)
    logging.info(
        "[COLOR TERM] target bp_rp=%.3f, comp_med bp_rp=%.3f, correction=%+.4f mag",
        float(target_bp_rp),
        float(bp_med),
        float(corr),
    )
    return out, float(corr), float(bp_med)


def _check_color_term_extrapolation(
    target_bp_rp: float,
    comp_bp_rp_values: list[float],
    target_name: str = "",
    *,
    extrapolation_tol: float = 0.0,
) -> bool:
    """Return True when target BP-RP is within the comp BP-RP range (± ``extrapolation_tol``).

    Return False when outside range → caller must skip CT (target kept, uncorrected).
    """
    finite_vals: list[float] = []
    for v in comp_bp_rp_values:
        try:
            vf = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(vf):
            finite_vals.append(vf)
    try:
        tgt = float(target_bp_rp)
    except (TypeError, ValueError):
        return True
    if len(finite_vals) < 2 or not math.isfinite(tgt):
        return True
    try:
        tol = max(0.0, float(extrapolation_tol))
    except (TypeError, ValueError):
        tol = 0.0
    min_bprp = float(min(finite_vals))
    max_bprp = float(max(finite_vals))
    in_range = (min_bprp - tol) <= tgt <= (max_bprp + tol)
    if not in_range:
        logging.warning(
            "[COLOR TERM] Target %s BP-RP=%.3f je mimo rozsahu comp [%.3f, %.3f] - color term extrapolacia moze viest k systematike.",
            str(target_name or ""),
            float(tgt),
            float(min_bprp),
            float(max_bprp),
        )
    return in_range


def should_apply_color_term(
    obs_group: str,
    c1: float,
    c1_stderr: float,
    n_comp: int,
    *,
    min_comp_for_ct: int = 7,
    max_stderr_ratio: float = 0.5,
) -> tuple[bool, str]:
    """
    Auto-rozhodnutie či aplikovať color term korekciu.

    Returns: (apply: bool, reason: str)
    reason = krátky popis prečo sa CT aplikuje alebo nie
    """
    from band_classify import classify_photometric_band, color_term_auto_from_band

    filter_raw = str(obs_group or "").split("|")[0].strip()
    band = classify_photometric_band(obs_group)
    if not color_term_auto_from_band(band):
        return False, f"{band.value} ({filter_raw}) — CT nie je potrebný"

    try:
        n_comp_i = int(n_comp)
    except Exception:  # noqa: BLE001
        n_comp_i = 0
    if n_comp_i < int(min_comp_for_ct):
        return (
            False,
            (
                f"Filter {filter_raw} — CT preskočený: "
                f"málo COMP ({n_comp_i} < {int(min_comp_for_ct)})"
            ),
        )

    if not (float(c1) != 0.0 and abs(float(c1)) > 1e-6):
        return False, f"Filter {filter_raw} — CT preskočený: c1 ≈ 0"

    stderr_ratio = abs(float(c1_stderr) / float(c1)) if float(c1) != 0.0 else float("inf")
    if not math.isfinite(stderr_ratio):
        return False, f"Filter {filter_raw} — CT nespoľahlivý: stderr/c1=NaN"
    if float(stderr_ratio) > float(max_stderr_ratio):
        return (
            False,
            (
                f"Filter {filter_raw} — CT nespoľahlivý: "
                f"stderr/c1={stderr_ratio:.2f} > {float(max_stderr_ratio):.2f}"
            ),
        )

    return True, (
        f"Filter {filter_raw} — CT aplikovaný: "
        f"c1={float(c1):+.4f} ± {float(c1_stderr):.4f} "
        f"(stderr/c1={stderr_ratio:.2f}, n_comp={n_comp_i})"
    )


def _obs_group_filter_key(obs_group: str) -> str:
    raw = str(obs_group or "").split("|")[0].strip()
    part = raw.split("_")[0].strip()
    return part.lower() if part else raw.lower()


def _is_nofilter_obs_group(obs_group: str) -> bool:
    filter_norm = _obs_group_filter_key(obs_group)
    no_filter_names = {
        "nofilter",
        "no_filter",
        "no filter",
        "clear",
        "clr",
        "cl",
        "none",
        "lum",
        "luminance",
        "l",
        "",
    }
    return filter_norm in no_filter_names


def _is_broadband_photometric_filter(obs_group: str) -> bool:
    """True for Johnson/Cousins/Sloan broadband filters (B/V/Rc/…); false for L/Clear/unknown."""
    from band_classify import classify_photometric_band, color_term_auto_from_band

    band = classify_photometric_band(obs_group)
    return bool(color_term_auto_from_band(band))


def resolve_apply_color_term(
    cfg: Any | None,
    obs_group: str,
    *,
    fits_filter: str | None = None,
    aavso_code: str | None = None,
) -> bool:
    """User/config toggle: CT applies correction only — never limits the target set."""
    from band_classify import classify_photometric_band, color_term_auto_from_band

    mode = str(getattr(cfg, "apply_color_term", "auto") or "auto").strip().lower()
    if mode in ("0", "false", "no", "off"):
        return False
    if mode in ("1", "true", "yes", "on"):
        return True
    band = classify_photometric_band(
        obs_group,
        fits_filter=fits_filter,
        aavso_code=aavso_code,
    )
    return bool(color_term_auto_from_band(band))


def _target_display_name(row: Any, *, fallback_cid: str = "") -> str:
    """VSX name when present, else Gaia ``catalog_id`` — never the literal ``nan``."""
    if row is None:
        return str(fallback_cid or "").strip() or "unknown"
    for key in ("vsx_name", "name"):
        try:
            v = row.get(key, "")
        except Exception:  # noqa: BLE001
            v = ""
        if v is None:
            continue
        if isinstance(v, float) and not math.isfinite(v):
            continue
        s = str(v).strip()
        if s and s.lower() not in ("nan", "none"):
            return s
    cid = str(fallback_cid or "").strip() or _normalize_gaia_id(
        row.get("catalog_id", "") if hasattr(row, "get") else ""
    )
    return cid or "unknown"


def _ensure_active_target_display_names(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or getattr(df, "empty", True):
        return df
    out = df.copy()
    if "vsx_name" not in out.columns:
        out["vsx_name"] = ""
    filled: list[str] = []
    for _, row in out.iterrows():
        cid = _normalize_gaia_id(row.get("catalog_id", ""))
        filled.append(_target_display_name(row, fallback_cid=cid))
    out["vsx_name"] = filled
    if "name" in out.columns:
        blank_name = out["name"].astype(str).str.strip().str.lower().isin(("", "nan", "none"))
        out.loc[blank_name, "name"] = out.loc[blank_name, "catalog_id"]
    return out


@dataclass
class _ColorTermGroupFit:
    c1: float
    c1_stderr: float
    n_comp: int
    comp_bp_rp: dict[str, float]
    comp_quality: dict[str, dict]
    comp_catalog_mag: dict[str, float]
    apply_gate: bool
    gate_reason: str


def _group_comp_mag_inst_from_flux_matrix(
    flux_matrix: pd.DataFrame,
    comp_ids: list[str],
    csv_files: list[Path],
) -> dict[str, np.ndarray]:
    n = len(csv_files)
    out: dict[str, np.ndarray] = {cid: np.full(n, float("nan"), dtype=np.float64) for cid in comp_ids}
    if flux_matrix is None or getattr(flux_matrix, "empty", True):
        return out
    if "mag_inst" not in flux_matrix.columns or "source_file" not in flux_matrix.columns:
        return out
    id_set = set(comp_ids)
    fm = flux_matrix[flux_matrix["catalog_id"].astype(str).isin(id_set)].copy()
    for i, csv_path in enumerate(csv_files):
        stem = csv_path.name
        sub = fm[fm["source_file"].astype(str) == stem]
        for cid in comp_ids:
            hit = sub[sub["catalog_id"].astype(str) == cid]
            if hit.empty:
                continue
            v = pd.to_numeric(hit.iloc[0]["mag_inst"], errors="coerce")
            if math.isfinite(float(v)):
                out[cid][i] = float(v)
    return out


def _group_comp_mag_inst_from_proc_csvs(
    comp_ids: list[str],
    csv_files: list[Path],
) -> dict[str, np.ndarray]:
    """Inst magnitudes for global comp pool — one array per comp across all frames."""
    n = len(csv_files)
    out: dict[str, np.ndarray] = {cid: np.full(n, float("nan"), dtype=np.float64) for cid in comp_ids}
    id_set = set(comp_ids)
    for i, csv_path in enumerate(csv_files):
        try:
            df = pd.read_csv(csv_path, low_memory=False, dtype=_GAIA_ID_DTYPE)
        except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
            from except_fix_counters import get_except_fix_counters

            get_except_fix_counters().comp_pool_csv_skip += 1
            logging.error("[PHOT] comp-pool CSV skip %s (frame %d): %s", csv_path, i, exc)
            continue
        if "catalog_id" not in df.columns:
            continue
        df = df.copy()
        df["_nid"] = df["catalog_id"].apply(_normalize_gaia_id)
        sub = df[df["_nid"].isin(id_set)]
        mag_col = "mag_inst" if "mag_inst" in sub.columns else None
        flux_col = None
        if mag_col is None:
            for fc in ("dao_flux", "flux", "aperture_flux"):
                if fc in sub.columns:
                    flux_col = fc
                    break
        for _, row in sub.iterrows():
            cid = str(row["_nid"])
            if cid not in out:
                continue
            if mag_col is not None:
                v = pd.to_numeric(row.get(mag_col), errors="coerce")
                if math.isfinite(float(v)):
                    out[cid][i] = float(v)
            elif flux_col is not None:
                flux = float(pd.to_numeric(row.get(flux_col), errors="coerce"))
                if math.isfinite(flux) and flux > 0:
                    out[cid][i] = float(-2.5 * math.log10(flux))
    return out


def _comp_maps_from_comparison_stars_csv(comp_csv: Path) -> tuple[dict[str, float], dict[str, float], dict[str, dict]]:
    comp_df = pd.read_csv(comp_csv, low_memory=False, dtype=_GAIA_ID_DTYPE)
    comp_bp_rp: dict[str, float] = {}
    comp_catalog_mag: dict[str, float] = {}
    comp_quality: dict[str, dict] = {}
    for _, row in comp_df.iterrows():
        cid = _normalize_gaia_id(row.get("catalog_id", row.get("name")))
        if not cid:
            continue
        bp = pd.to_numeric(row.get("bp_rp"), errors="coerce")
        if math.isfinite(float(bp)):
            comp_bp_rp[cid] = float(bp)
        for col in ("phot_g_mean_mag", "catalog_mag", "mag"):
            v = pd.to_numeric(row.get(col), errors="coerce")
            if math.isfinite(float(v)):
                comp_catalog_mag[cid] = float(v)
                break
        usable = True
        for qcol in ("is_usable", "photometry_ok"):
            if qcol in row.index:
                if str(row.get(qcol)).strip().lower() in ("false", "0", "no"):
                    usable = False
        comp_quality[cid] = {"quality": "good" if usable else "excluded"}
    return comp_bp_rp, comp_catalog_mag, comp_quality


def _compute_group_color_term_fit(
    *,
    comparison_stars_csv: Path,
    flux_matrix: pd.DataFrame,
    csv_files: list[Path],
    obs_group: str,
    cfg: Any,
) -> _ColorTermGroupFit | None:
    comp_csv = Path(comparison_stars_csv)
    if not comp_csv.is_file():
        return None
    comp_bp_rp, comp_catalog_mag, comp_quality = _comp_maps_from_comparison_stars_csv(comp_csv)
    comp_ids = sorted(comp_bp_rp.keys())
    if not comp_ids:
        return None
    comp_mag_inst = _group_comp_mag_inst_from_proc_csvs(comp_ids, csv_files)
    n_from_matrix = sum(
        1
        for cid in comp_ids
        if int(np.isfinite(comp_mag_inst[cid]).sum()) >= 3
    )
    if n_from_matrix < max(10, len(comp_ids) // 4) and flux_matrix is not None:
        comp_mag_inst = _group_comp_mag_inst_from_flux_matrix(flux_matrix, comp_ids, csv_files)
    from k2_extinction import (  # noqa: PLC0415
        K2Source,
        apply_k2_to_comp_mag_inst,
        airmass_from_proc_csvs,
        bp_rp_comp_median,
        resolve_k2_bprp_value,
    )

    _k2_val, _k2_src = resolve_k2_bprp_value(cfg, obs_group)
    if _k2_src is K2Source.LITERATURE_DEFAULT and math.isfinite(float(_k2_val)):
        _bp_med = bp_rp_comp_median(comp_bp_rp, comp_quality)
        if math.isfinite(_bp_med):
            comp_mag_inst = apply_k2_to_comp_mag_inst(
                comp_mag_inst,
                comp_bp_rp,
                comp_quality,
                airmass_from_proc_csvs(csv_files),
                float(_k2_val),
                _bp_med,
            )
    c1, c1_stderr, n_comp = fit_color_term_c1(
        comp_mag_inst,
        comp_catalog_mag,
        comp_bp_rp,
        comp_quality,
        min_comp=5,
        sigma_clip_sigma=3.0,
    )
    apply_gate, gate_reason = should_apply_color_term(
        obs_group=_obs_group_filter_key(obs_group),
        c1=float(c1),
        c1_stderr=float(c1_stderr),
        n_comp=int(n_comp),
        min_comp_for_ct=int(getattr(cfg, "phase01_ct_min_comp", 7)),
    )
    return _ColorTermGroupFit(
        c1=float(c1),
        c1_stderr=float(c1_stderr),
        n_comp=int(n_comp),
        comp_bp_rp=comp_bp_rp,
        comp_quality=comp_quality,
        comp_catalog_mag=comp_catalog_mag,
        apply_gate=bool(apply_gate),
        gate_reason=str(gate_reason),
    )


def _ensure_group_comp_pool_csv(
    *,
    platesolve_dir: Path,
    masterstar_fits: Path,
    masterstars_csv: Path,
    cfg: Any,
    draft_id: int | None,
    min_pool: int = 20,
) -> Path:
    """Ensure ``comparison_stars.csv`` has enough stars for global colour-term fit."""
    ps_dir = Path(platesolve_dir)
    comp_csv = ps_dir / "comparison_stars.csv"
    n_pool = 0
    if comp_csv.is_file():
        try:
            n_pool = int(len(pd.read_csv(comp_csv, usecols=["catalog_id"])))
        except Exception:  # noqa: BLE001
            try:
                n_pool = int(len(pd.read_csv(comp_csv)))
            except Exception:  # noqa: BLE001
                n_pool = 0
    if n_pool >= int(min_pool):
        return comp_csv
    ms_fits = Path(masterstar_fits)
    ms_path = Path(masterstars_csv)
    if not ms_fits.is_file() or not ms_path.is_file():
        return comp_csv
    try:
        from pipeline import write_photometry_plan_files  # noqa: PLC0415

        write_photometry_plan_files(
            platesolve_dir=ps_dir,
            masterstar_fits=ms_fits,
            masterstars_csv=ms_path,
            n_comparison_stars=int(getattr(cfg, "comparison_stars_pool_n", 150) or 150),
            require_non_variable=bool(getattr(cfg, "phase01_comparison_require_non_variable", True)),
            draft_id=int(draft_id) if draft_id is not None else None,
            database_path=getattr(cfg, "database_path", None),
        )
        log_event(f"[PHOT] Refreshed comparison_stars.csv pool ({n_pool}→spatial grid) for CT fit in {ps_dir.name}")
    except Exception as exc:  # noqa: BLE001
        logging.error('[EXC-0137] comparison_stars.csv spatial pool refresh fails - CT fit uses stale/sparse comp pool: %s', exc)
        logging.warning("[PHOT] comparison_stars pool refresh failed: %s", exc)
    return comp_csv


def _variable_targets_looks_like_ct_presel_stub(vt_path: Path, *, masterstars_csv: Path) -> bool:
    if not vt_path.is_file() or not masterstars_csv.is_file():
        return False
    try:
        vt = pd.read_csv(vt_path, low_memory=False, nrows=500)
        ms = pd.read_csv(masterstars_csv, low_memory=False, usecols=["catalog_id"])
    except Exception as exc:  # noqa: BLE001
        logging.error('[EXC-0138] variable_targets stub detection CSV read fails - returns False (stub may go undetected): %s', exc)
        return False
    if len(vt) >= max(80, int(len(ms) * 0.05)):
        return False
    if "notes" in vt.columns:
        notes = vt["notes"].astype(str).str.contains("CT presel", case=False, na=False)
        if bool(notes.any()):
            return True
    if "name" in vt.columns:
        names = vt["name"].astype(str)
        if bool(names.str.contains("M67 in-range|M67 red-giant", case=False, regex=True).any()):
            return True
    return len(vt) < 50 and len(ms) > 200


def ensure_full_variable_targets_if_presel_stub(
    *,
    variable_targets_csv: Path,
    masterstars_csv: Path,
    masterstar_fits: Path,
    cfg: Any | None = None,
    draft_id: int | None = None,
) -> bool:
    """Restore full-field ``variable_targets.csv`` when CT presel stub replaced production list."""
    vt_path = Path(variable_targets_csv)
    ms_path = Path(masterstars_csv)
    if not _variable_targets_looks_like_ct_presel_stub(vt_path, masterstars_csv=ms_path):
        return False
    ps_dir = vt_path.parent
    ms_fits = Path(masterstar_fits)
    if not ms_fits.is_file():
        ms_fits = ps_dir / "MASTERSTAR.fits"
    if not ms_fits.is_file() or not ms_path.is_file():
        logging.warning(
            "[PHOT] CT presel stub detected but cannot restore variable_targets (missing MASTERSTAR/masterstars)"
        )
        return False
    try:
        from pipeline import write_photometry_plan_files  # noqa: PLC0415

        _cfg = cfg or AppConfig()
        write_photometry_plan_files(
            platesolve_dir=ps_dir,
            masterstar_fits=ms_fits,
            masterstars_csv=ms_path,
            n_comparison_stars=int(getattr(_cfg, "comparison_stars_pool_n", 150) or 150),
            require_non_variable=bool(getattr(_cfg, "phase01_comparison_require_non_variable", True)),
            draft_id=int(draft_id) if draft_id is not None else None,
            database_path=getattr(_cfg, "database_path", None),
        )
        log_event(
            f"[PHOT] Restored full variable_targets.csv from field cone (replaced CT presel stub in {ps_dir.name})"
        )
        return True
    except Exception as exc:  # noqa: BLE001
        logging.error('[EXC-0139] Full variable_targets restore from field cone fails - CT presel stub may remain as targ...: %s', exc)
        logging.warning("[PHOT] variable_targets restore failed: %s", exc)
        return False


_CT_PROTOTYPE_CSV_FIELDS: tuple[str, ...] = (
    "catalog_id",
    "vsx_name",
    "obs_group",
    "n_comp_used",
    "c1",
    "c1_stderr",
    "stderr_ratio",
    "target_bp_rp",
    "comp_med_bp_rp",
    "ct_corr",
    "cat_inst_scatter",
    "cat_inst_scatter_resid",
    "gate_would_pass",
)


def _ct_prototype_enabled() -> bool:
    return os.environ.get("VYVAR_CT_PROTOTYPE", "").strip() == "1"


def _color_term_cat_inst_scatter_pair(
    comp_mag_inst: dict[str, np.ndarray],
    comp_catalog_mag: dict[str, float],
    comp_bp_rp: dict[str, float],
    comp_quality: dict[str, dict],
    c1: float,
    *,
    min_comp: int = 5,
    sigma_clip_sigma: float = 3.0,
) -> tuple[float, float]:
    """Per-comp cat−inst scatter before/after removing the fitted c1·Δ(bp_rp) trend (sigma-clipped comps)."""
    try:
        min_comp_i = int(min_comp)
    except Exception:  # noqa: BLE001
        min_comp_i = 5
    min_comp_i = max(2, min_comp_i)

    usable = [cid for cid, q in comp_quality.items() if q.get("quality") in ("good", "suspect")]
    ys: list[float] = []
    bp_vals: list[float] = []

    for cid in usable:
        bp = float(comp_bp_rp.get(cid, float("nan")))
        if not math.isfinite(bp):
            continue
        if cid not in comp_mag_inst:
            continue
        inst = np.asarray(comp_mag_inst[cid], dtype=np.float64)
        finite = inst[np.isfinite(inst)]
        if finite.size < min_comp_i:
            continue
        cat = float(comp_catalog_mag.get(cid, float("nan")))
        if not math.isfinite(cat):
            continue
        y = float(np.nanmedian(cat - finite))
        if not math.isfinite(y):
            continue
        bp_vals.append(bp)
        ys.append(y)

    if len(ys) < min_comp_i:
        return float("nan"), float("nan")

    bp_med = float(np.median(np.asarray(bp_vals, dtype=np.float64)))
    xs = [float(bp) - bp_med for bp in bp_vals]
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)

    p0 = _safe_polyfit(x, y, 1)
    if p0 is None:
        return float("nan"), float("nan")
    try:
        c1_init = float(p0[0])
        zp_init = float(p0[1])
    except Exception as exc:  # noqa: BLE001
        logging.error('[EXC-0140] Color-term fit coefficient unpack fails - returns NaN c1/zp and downstream CT fit abort...: %s', exc)
        return float("nan"), float("nan")

    resid = y - (c1_init * x + zp_init)
    sig = _mad_sigma(resid)
    if not math.isfinite(sig) or sig <= 0:
        mask = np.ones_like(resid, dtype=bool)
    else:
        mask = np.abs(resid) <= float(sigma_clip_sigma) * float(sig)

    x_cl = x[mask]
    y_cl = y[mask]
    if x_cl.size < 2:
        return float("nan"), float("nan")

    scatter = float(np.std(y_cl))
    if not (math.isfinite(float(c1)) and float(c1) != 0.0):
        return scatter, float("nan")
    resid_ct = y_cl - float(c1) * x_cl
    scatter_resid = float(np.std(resid_ct)) if resid_ct.size >= 2 else float("nan")
    return scatter, scatter_resid


def _append_ct_prototype_row(draft_dir: Path, row: dict[str, Any]) -> None:
    import csv  # noqa: PLC0415

    draft_dir = Path(draft_dir)
    path = draft_dir / "ct_prototype.csv"
    write_header = not path.is_file() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(_CT_PROTOTYPE_CSV_FIELDS))
        if write_header:
            writer.writeheader()
        out_row: dict[str, Any] = {}
        for key in _CT_PROTOTYPE_CSV_FIELDS:
            val = row.get(key, "")
            if key == "gate_would_pass" or isinstance(val, (bool, np.bool_)):
                out_row[key] = "True" if bool(val) else "False"
            else:
                out_row[key] = val
        writer.writerow(out_row)

# ---------------------------------------------------------------------------
# KROK 5: Outlier detekcia
# ---------------------------------------------------------------------------


def _target_row_is_vsx_known_variable(target_row: pd.Series) -> bool:
    """True when target is a catalogued variable (VSX name/type), not a Gaia-only label."""
    vn = str(target_row.get("vsx_name", target_row.get("name", "")) or "").strip()
    if vn and vn.lower() not in ("nan", "none", "—", "-"):
        if not vn.lower().startswith("gaia dr3"):
            return True
    vt = str(target_row.get("vsx_type", "") or "").strip()
    return bool(vt) and vt.lower() not in ("nan", "none")


def empirical_feature_mask_mag(
    mag: np.ndarray,
    *,
    k: float = 3.0,
    min_run: int = 3,
) -> np.ndarray:
    """Mask eclipses/transits before sigma-clipping (TESS subdwarf recipe, arXiv:2402.16018).

    In flux space: mark runs of >= ``min_run`` consecutive points below ``-k`` sigma_MAD
    from the median flux; extend the mask on both sides until flux returns above the median.
    """
    n = len(mag)
    protected = np.zeros(n, dtype=bool)
    m = np.asarray(mag, dtype=float)
    finite = np.isfinite(m)
    if int(finite.sum()) < max(int(min_run) + 2, 5):
        return protected

    flux = np.power(10.0, -0.4 * m)
    f_ref = float(np.nanmedian(flux[finite]))
    if not math.isfinite(f_ref) or f_ref <= 0:
        return protected

    resid = flux - f_ref
    r_fin = resid[finite]
    r_med = float(np.median(r_fin))
    mad = float(np.median(np.abs(r_fin - r_med)))
    sigma = max(mad / _MAD_CONSISTENCY, 1e-12)
    deep = (resid < (-float(k) * sigma)) & finite

    i = 0
    while i < n:
        if not bool(deep[i]):
            i += 1
            continue
        j = i
        while j < n and bool(deep[j]):
            j += 1
        if (j - i) >= int(min_run):
            lo = i
            while lo > 0 and bool(finite[lo - 1]) and float(flux[lo - 1]) < f_ref:
                lo -= 1
            hi = j
            while hi < n and bool(finite[hi]) and float(flux[hi]) < f_ref:
                hi += 1
            protected[lo:hi] = True
        i = j
    return protected


def detect_outliers(
    mag_calib: np.ndarray,
    flags_saturated: np.ndarray,
    *,
    outlier_sigma: float = 3.0,
    feature_mask: np.ndarray | None = None,
    skip_sigma_clip: bool = False,
) -> list[str]:
    """Outlier detekcia v svetelnej krivke (reporting path; mask-first for features).

    ``feature_mask`` protects eclipse/transit epochs (arXiv:2402.16018) from sigma-clipping.
    ``skip_sigma_clip`` relaxes clipping for VSX-known variables (catalogued astrophysical signal).

    Returns:
        list flagov: "normal" / "saturated" / "outlier_hi" / "outlier_lo" / "no_data"
    """
    n = len(mag_calib)
    flags = ["no_data"] * n
    finite_mask = np.isfinite(mag_calib)

    _prot = np.zeros(n, dtype=bool)
    if feature_mask is not None:
        _fm = np.asarray(feature_mask, dtype=bool)
        if len(_fm) == n:
            _prot = _fm

    for i in range(n):
        if not math.isfinite(mag_calib[i]):
            flags[i] = "no_data"
        elif bool(flags_saturated[i]):
            flags[i] = "saturated"
        elif bool(_prot[i]) or bool(skip_sigma_clip):
            flags[i] = "normal"
        else:
            flags[i] = "normal"

    if skip_sigma_clip or finite_mask.sum() < 3:
        return flags

    clip_mask = finite_mask & ~_prot
    if int(clip_mask.sum()) < 3:
        clip_mask = finite_mask

    finite_vals = mag_calib[clip_mask]
    med = float(np.median(finite_vals))
    sigma = _mad_sigma(finite_vals)
    thr = outlier_sigma * sigma

    for i in range(n):
        if flags[i] != "normal":
            continue
        if bool(_prot[i]):
            continue
        if mag_calib[i] < med - thr:
            flags[i] = "outlier_hi"
        elif mag_calib[i] > med + thr:
            flags[i] = "outlier_lo"

    return flags


def apply_reporting_postprocess(
    mag_calib: np.ndarray,
    mag_calib_ct: np.ndarray,
    *,
    target_row: pd.Series,
    target_name: str,
    sat_flags: np.ndarray,
    target_frames: pd.DataFrame,
    outlier_sigma: float,
    ct_ok: bool,
    ac_ok: bool,
    delta_m_corr: float | None,
    cfg: AppConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Workstream B: ship ensemble-calibrated mag; mask-first outliers; no target airmass LSQ.

    Basis: Plavchan et al. (2007) arXiv:0704.3584; TESS subdwarf mask arXiv:2402.16018.
    """
    _cfg = cfg or AppConfig()
    mag_calib_raw = np.asarray(mag_calib, dtype=np.float64).copy()
    mag_for_report = np.asarray(mag_calib_ct, dtype=np.float64)
    _feature_mask = empirical_feature_mask_mag(mag_for_report)
    _vsx_known = _target_row_is_vsx_known_variable(target_row)
    if int(_feature_mask.sum()) > 0:
        logging.info(
            "[OUTLIER] Feature mask: %d/%d frames protected before clip (arXiv:2402.16018)",
            int(_feature_mask.sum()),
            len(_feature_mask),
        )
    if _vsx_known:
        logging.debug("[OUTLIER] VSX-known variable %s: sigma clip skipped", target_name)
    out_flags = detect_outliers(
        mag_for_report,
        sat_flags,
        outlier_sigma=outlier_sigma,
        feature_mask=_feature_mask,
        skip_sigma_clip=_vsx_known,
    )
    _preserve_nondetection_flags_helper(out_flags, target_frames)
    mag_out = mag_calib_raw.copy()
    if ac_ok and delta_m_corr is not None and np.isfinite(float(delta_m_corr)):
        mag_calib_ac = mag_out + float(delta_m_corr)
    else:
        mag_calib_ac = np.full_like(mag_out, float("nan"))
    mag_calib_ct_out = np.asarray(mag_calib_ct, dtype=np.float64).copy()
    if not ct_ok:
        mag_calib_ct_out = mag_out.copy()
    if bool(_cfg.phase2a_airmass_before_outlier):
        logging.info(
            "[PHASE 2A] phase2a_airmass_before_outlier=True ignored for shipped columns "
            "(Workstream B: no target airmass LSQ on reporting path)"
        )
    return mag_calib_raw, mag_out, mag_calib_ct_out, mag_calib_ac, out_flags


def democratic_detrend_lc(
    mag_calib: np.ndarray,
    bjd: np.ndarray,
    airmass: np.ndarray,
    flags: list[str],
    *,
    window_frac: float = 0.5,
    polyorder: int = 2,
    min_points: int = 10,
    enabled: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Democratic Detrender — ensemble multi-model detrending (arXiv:2411.09753v2, 2026).

    Reference: Caballero-Nieves et al. (2026) arXiv:2411.09753v2 —
    'The democratic detrender: Ensemble-Based Removal of the
    Nuisance Signal in Stellar Time-Series Photometry'

    Runs three independent detrending models in parallel:
        A) Linear airmass fit (standard VYVAR model)
        B) Degree-2 polynomial fit on BJD (slow time trend)
        C) Savitzky-Golay filter (window_frac of normal frames)

    Returns marginalized mean of all valid models plus MAD-based
    error inflation factor (model-selection uncertainty).

    Args:
        mag_calib:    calibrated mag array (post-ensemble, post-airmass)
        bjd:          BJD timestamps (same length)
        airmass:      per-frame airmass (NaN allowed)
        flags:        per-frame flags; only "normal" used for fitting
        window_frac:  SG window fraction (default 0.5)
        polyorder:    SG polynomial order (default 2)
        min_points:   minimum normal frames required
        enabled:      False → return (mag_calib copy, zeros)

    Returns:
        (mag_democratic, err_inflation)
        mag_democratic: marginalized detrended mag (length = n_frames)
        err_inflation:  per-frame MAD across models (0 if only 1 model)
    """
    if not enabled:
        return mag_calib.copy(), np.zeros(len(mag_calib))

    from scipy.signal import savgol_filter  # lazy import — scipy already in deps

    mag = np.asarray(mag_calib, dtype=float)
    bjd_arr = np.asarray(bjd, dtype=float)
    am_arr = np.asarray(airmass, dtype=float)
    normal_mask = np.array([f == "normal" for f in flags])
    n_normal = int(normal_mask.sum())

    if n_normal < min_points:
        return mag.copy(), np.zeros(len(mag))

    idx_normal = np.where(normal_mask)[0]
    models: list[np.ndarray] = []

    am_normal = am_arr[idx_normal]
    mag_normal = mag[idx_normal]

    # Model A: linear airmass fit
    am_finite = np.isfinite(am_normal)
    if int(am_finite.sum()) >= min_points:
        try:
            coeffs = _safe_polyfit(am_normal[am_finite], mag_normal[am_finite], 1)
            if coeffs is not None:
                trend_a = np.where(
                    np.isfinite(am_arr),
                    np.polyval(coeffs, am_arr),
                    np.nanmedian(mag_normal),
                )
                mag_a = mag - trend_a + np.nanmedian(mag_normal)
                models.append(mag_a)
        except Exception as exc:  # noqa: BLE001
            logging.error('[EXC-0141] Airmass detrend model branch dropped - democratic detrend uses fewer models, LC trend r...: %s', exc)
            pass

    # Model B: degree-2 polynomial on BJD (slow time trend)
    bjd_normal = bjd_arr[idx_normal]
    bjd_finite = np.isfinite(bjd_normal)
    if int(bjd_finite.sum()) >= min_points:
        try:
            bjd_ref = np.nanmedian(bjd_arr)
            bjd_centered = bjd_arr - bjd_ref
            bjd_c_normal = bjd_normal[bjd_finite] - bjd_ref
            coeffs_b = _safe_polyfit(bjd_c_normal, mag_normal[bjd_finite], 2)
            if coeffs_b is not None:
                trend_b = np.polyval(coeffs_b, bjd_centered)
                mag_b = mag - trend_b + np.nanmedian(mag_normal)
                models.append(mag_b)
        except Exception as exc:  # noqa: BLE001
            logging.error('[EXC-0142] BJD polynomial detrend model branch dropped - democratic detrend LC differs: %s', exc)
            pass

    # Model C: Savitzky-Golay on time axis
    raw_w = max(5, int(n_normal * window_frac))
    sg_window = raw_w if raw_w % 2 == 1 else raw_w + 1
    sg_window = min(sg_window, n_normal if n_normal % 2 == 1 else n_normal - 1)
    if sg_window > polyorder and len(idx_normal) >= min_points:
        try:
            mag_smooth = savgol_filter(
                mag_normal, window_length=sg_window, polyorder=polyorder
            )
            trend_c = np.interp(np.arange(len(mag)), idx_normal, mag_smooth)
            mag_c = mag - trend_c + np.nanmedian(mag_normal)
            models.append(mag_c)
        except Exception as exc:  # noqa: BLE001
            logging.error('[EXC-0143] Savitzky-Golay detrend model branch dropped - democratic detrend LC differs: %s', exc)
            pass

    if not models:
        return mag.copy(), np.zeros(len(mag))

    model_stack = np.column_stack(models)
    mag_democratic = np.nanmean(model_stack, axis=1)

    if model_stack.shape[1] > 1:
        mad_per_frame = np.nanmedian(
            np.abs(model_stack - mag_democratic[:, np.newaxis]),
            axis=1,
        )
    else:
        mad_per_frame = np.zeros(len(mag))

    LOGGER.debug(
        "[ALG-4 Democratic] %d models, %d normal frames, "
        "median_err_inflation=%.5f mag",
        len(models),
        n_normal,
        float(np.nanmedian(mad_per_frame)),
    )
    return mag_democratic, mad_per_frame


def savgol_detrend_lc(
    mag_calib: np.ndarray,
    bjd: np.ndarray,
    flags: list[str],
    *,
    window_frac: float = 0.5,
    polyorder: int = 2,
    min_points: int = 10,
    enabled: bool = True,
) -> np.ndarray:
    """Savitzky-Golay non-linear detrending of calibrated LC (ALG-2).

    Reference: Savitzky & Golay (1964) Anal. Chem. 36, 1627;
    applied to differential photometry detrending as in
    Aigrain & Irwin (2004) MNRAS 350, 331.

    Runs AFTER airmass detrend — removes slow systematic trends
    (transparency drift, seeing variations) not captured by linear
    airmass model. Applied to residuals, not raw mag_calib.

    Args:
        mag_calib:    detrended mag array (length = n_frames)
        bjd:          BJD timestamps (same length)
        flags:        per-frame flags; only "normal" frames used for fit
        window_frac:  SG window as fraction of n_normal frames (default 0.5)
        polyorder:    SG polynomial order (default 2)
        min_points:   minimum normal frames required (default 10)
        enabled:      False → return mag_calib unchanged

    Returns:
        mag_sg: detrended mag array (same length as input)
    """
    _ = bjd  # frame order implicit in index; time-aware SG reserved for later
    if not enabled:
        return mag_calib.copy()

    from scipy.signal import savgol_filter  # lazy import — scipy already in deps

    mag = np.asarray(mag_calib, dtype=float)
    normal_mask = np.array([f == "normal" for f in flags])
    n_normal = int(normal_mask.sum())

    if n_normal < min_points:
        return mag.copy()

    raw_w = max(5, int(n_normal * window_frac))
    window = raw_w if raw_w % 2 == 1 else raw_w + 1
    window = min(window, n_normal if n_normal % 2 == 1 else n_normal - 1)

    if window <= polyorder:
        return mag.copy()

    # Warn if window covers more than 40% of the LC — risk of suppressing real variability
    if window > 0.4 * n_normal:
        LOGGER.warning(
            "[ALG-2 SG] window=%d is %.0f%% of n_normal=%d — "
            "may suppress real variability; consider decreasing savgol_window_frac "
            "(current=%.2f) or disabling savgol_detrend_enabled",
            window,
            100.0 * window / n_normal,
            n_normal,
            window_frac,
        )

    idx_normal = np.where(normal_mask)[0]
    mag_normal = mag[idx_normal]

    try:
        mag_smooth = savgol_filter(mag_normal, window_length=window, polyorder=polyorder)
    except Exception:  # noqa: BLE001
        # EXC-0144: T4 -- SavGol detrend failure returns un-detrended mag array unchanged (EXCEPT-BULK-2 2026-07-08)
        return mag.copy()

    trend_all = np.interp(
        np.arange(len(mag)),
        idx_normal,
        mag_smooth,
    )

    mag_sg = mag - trend_all + np.nanmedian(mag_normal)

    LOGGER.debug(
        "[ALG-2 SG] window=%d/%d frames, poly=%d, n_normal=%d",
        window,
        len(mag),
        polyorder,
        n_normal,
    )
    return mag_sg


def compute_mag_calib_final(
    mag_calib: np.ndarray,
    *,
    ct_ok: bool = False,
    ct_correction: float | None = None,
    ac_ok: bool = False,
    delta_m_corr: float | None = None,
    mag_calib_ac: np.ndarray | None = None,
) -> np.ndarray:
    """Canonical published magnitude: final ``mag_calib`` + CT + AC (per-target/night constants).

    CT and AC are additive on the ensemble-calibrated base. When CT is off, result matches
    ``mag_calib_ac`` (same ``mag_calib`` + ``delta_m_corr``) for byte-identical export under CT-off config.
    """
    base = np.asarray(mag_calib, dtype=np.float64)
    ct_add = 0.0
    if bool(ct_ok) and ct_correction is not None:
        try:
            v = float(ct_correction)
            if math.isfinite(v):
                ct_add = v
        except (TypeError, ValueError):
            pass
    ac_add = 0.0
    if bool(ac_ok) and delta_m_corr is not None:
        try:
            v = float(delta_m_corr)
            if math.isfinite(v):
                ac_add = v
        except (TypeError, ValueError):
            pass
    if ct_add == 0.0 and ac_add == 0.0:
        return base.copy()
    if ct_add == 0.0 and ac_add != 0.0 and mag_calib_ac is not None:
        ac_arr = np.asarray(mag_calib_ac, dtype=np.float64)
        if ac_arr.shape == base.shape:
            return ac_arr.copy()
    return base + ct_add + ac_add


# ---------------------------------------------------------------------------
# KROK 6: Výstup — lightcurve CSV
# ---------------------------------------------------------------------------


def save_lightcurve_csv(
    output_path: Path,
    bjd: np.ndarray,
    hjd: np.ndarray,
    jd: np.ndarray,
    airmass: np.ndarray,
    is_flipped: np.ndarray | None,
    mag_inst: np.ndarray,
    mag_calib_raw: np.ndarray,
    mag_calib: np.ndarray,
    mag_calib_ct: np.ndarray | None,
    mag_calib_ac: np.ndarray | None,
    delta_mag: np.ndarray,
    err: np.ndarray,
    aperture_r_px: np.ndarray,
    flags: list[str],
    source_files: list[str],
    *,
    method: str = "aperture",
    ct_correction: float | None = None,
    ct_c1: float | None = None,
    ct_bp_rp_target: float | None = None,
    ct_bp_rp_comp_med: float | None = None,
    ct_n_comp: int | None = None,
    ct_ok: bool | None = None,
    k2_source: list[str] | np.ndarray | None = None,
    k2_value: float | None = None,
    k2_colour_ref: float | None = None,
    ac_result: dict[str, Any] | None = None,
    mag_democratic: np.ndarray | None = None,
    err_inflation: np.ndarray | None = None,
    lunar_phase_pct: float = float("nan"),
    lunar_separation_deg: float = float("nan"),
    lunar_risk: str = "UNKNOWN",
    dilution_factor: float = 1.0,
    alignment_failed: np.ndarray | None = None,
    err_scatter_unmatched: np.ndarray | None = None,
    catalog_match_mode: list[str] | np.ndarray | None = None,
    wcs_untrusted: np.ndarray | None = None,
    time_base: str = TIME_BASE_BJD_TDB,
    err_method: list[str] | None = None,
    sigma_sys_mag: float | None = None,
) -> None:
    """Uloží lightcurve CSV.

    LC schema note: ``time_base`` labels the BJD/HJD recompute path (``BJD_TDB`` vs
    ``JD_FALLBACK``); it does not alter ``bjd``/``hjd``/``jd`` values.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = int(len(bjd))
    if mag_calib_ct is None:
        mag_calib_ct_arr = np.asarray(mag_calib, dtype=np.float64).copy()
    else:
        mag_calib_ct_arr = np.asarray(mag_calib_ct, dtype=np.float64)
        if len(mag_calib_ct_arr) != n:
            mag_calib_ct_arr = np.asarray(mag_calib, dtype=np.float64).copy()

    ac_ok = bool(ac_result.get("ok", False)) if isinstance(ac_result, dict) else False
    delta_m_corr = (
        float(ac_result.get("delta_m_corr"))
        if (ac_ok and isinstance(ac_result, dict) and ac_result.get("delta_m_corr", None) is not None)
        else float("nan")
    )
    ac_scatter = (
        float(ac_result.get("scatter_mag"))
        if (isinstance(ac_result, dict) and ac_result.get("scatter_mag", None) is not None)
        else float("nan")
    )
    ac_n_ref = int(ac_result.get("n_ref_stars", 0)) if isinstance(ac_result, dict) else 0

    if mag_calib_ac is None:
        if ac_ok and math.isfinite(delta_m_corr):
            mag_calib_ac_arr = np.asarray(mag_calib, dtype=np.float64) + float(delta_m_corr)
        else:
            mag_calib_ac_arr = np.full_like(np.asarray(mag_calib, dtype=np.float64), float("nan"), dtype=np.float64)
    else:
        mag_calib_ac_arr = np.asarray(mag_calib_ac, dtype=np.float64)
        if len(mag_calib_ac_arr) != n:
            mag_calib_ac_arr = np.full_like(np.asarray(mag_calib, dtype=np.float64), float("nan"), dtype=np.float64)

    mag_calib_final_arr = compute_mag_calib_final(
        mag_calib,
        ct_ok=bool(ct_ok),
        ct_correction=ct_correction,
        ac_ok=ac_ok,
        delta_m_corr=delta_m_corr if ac_ok else None,
        mag_calib_ac=mag_calib_ac_arr,
    )

    def _fill_scalar(v: float | None, default: float) -> np.ndarray:
        vv = float(v) if v is not None else float(default)
        return np.full(n, vv, dtype=np.float64)

    def _fill_bool(v: bool | None) -> np.ndarray:
        vb = bool(v) if v is not None else False
        return np.full(n, vb, dtype=bool)

    def _flag_cell(f: Any) -> str:
        if f is None:
            return ""
        s = str(f).strip()
        return s if s.lower() not in ("none", "nan") else ""

    df = pd.DataFrame(
        {
            "bjd": bjd,
            "hjd": hjd,
            "jd": jd,
            "time_base": [str(time_base or TIME_BASE_BJD_TDB)] * n,
            "airmass": airmass,
            "is_flipped": (is_flipped if is_flipped is not None else np.full_like(bjd, False, dtype=bool)),
            "alignment_failed": (
                alignment_failed if alignment_failed is not None else np.full_like(bjd, False, dtype=bool)
            ),
            "err_scatter_unmatched": (
                err_scatter_unmatched
                if err_scatter_unmatched is not None
                else np.full_like(bjd, False, dtype=bool)
            ),
            "catalog_match_mode": (
                [normalize_catalog_match_mode(m) for m in catalog_match_mode]
                if catalog_match_mode is not None
                else [""] * n
            ),
            "wcs_untrusted": (
                wcs_untrusted if wcs_untrusted is not None else np.full_like(bjd, False, dtype=bool)
            ),
            "mag_inst": np.round(mag_inst, 6),
            "mag_calib_raw": np.round(mag_calib_raw, 6),
            "mag_calib": np.round(mag_calib, 6),
            "mag_calib_ct": np.round(mag_calib_ct_arr, 6),
            "ct_correction": np.round(_fill_scalar(ct_correction, float("nan")), 6),
            "ct_c1": np.round(_fill_scalar(ct_c1, float("nan")), 6),
            "ct_bp_rp_target": np.round(_fill_scalar(ct_bp_rp_target, float("nan")), 6),
            "ct_bp_rp_comp_med": np.round(_fill_scalar(ct_bp_rp_comp_med, float("nan")), 6),
            "ct_n_comp": np.full(n, int(ct_n_comp) if ct_n_comp is not None else -1, dtype=int),
            "ct_ok": _fill_bool(ct_ok),
            "k2_source": (
                [str(x) for x in k2_source]
                if k2_source is not None
                else [""] * n
            ),
            "k2_value": np.round(_fill_scalar(k2_value, float("nan")), 6),
            "k2_colour_ref": np.round(_fill_scalar(k2_colour_ref, float("nan")), 6),
            "ac_correction": np.round(_fill_scalar(delta_m_corr if (ac_ok and math.isfinite(delta_m_corr)) else None, float("nan")), 6),
            "ac_scatter": np.round(_fill_scalar(ac_scatter if math.isfinite(ac_scatter) else None, float("nan")), 6),
            "ac_n_ref": np.full(n, int(ac_n_ref), dtype=int),
            "ac_ok": np.full(n, bool(ac_ok), dtype=bool),
            "mag_calib_ac": np.round(mag_calib_ac_arr, 6),
            "mag_calib_final": np.round(mag_calib_final_arr, 6),
            "delta_mag": np.round(delta_mag, 6),
            "err": np.round(err, 6),
            "aperture_r_px": np.round(aperture_r_px, 3),
            "flag": [_flag_cell(f) for f in flags],
            "method": method,
            "source_file": source_files,
        }
    )
    if mag_democratic is not None:
        _md = np.asarray(mag_democratic, dtype=float)
        df["delta_mag_democratic"] = np.round(
            _md - float(np.nanmedian(_md)),
            6,
        )
    if err_inflation is not None:
        df["err_inflation"] = np.round(np.asarray(err_inflation, dtype=float), 6)
    _lp = float(lunar_phase_pct) if math.isfinite(float(lunar_phase_pct)) else float("nan")
    _ls = float(lunar_separation_deg) if math.isfinite(float(lunar_separation_deg)) else float("nan")
    _dfac = float(dilution_factor) if math.isfinite(float(dilution_factor)) else 1.0
    df["dilution_factor"] = np.round(np.full(n, _dfac, dtype=np.float64), 6)
    df["lunar_phase_pct"] = np.round(np.full(n, _lp, dtype=np.float64), 6)
    df["lunar_separation_deg"] = np.round(np.full(n, _ls, dtype=np.float64), 6)
    df["lunar_risk"] = [str(lunar_risk or "UNKNOWN")] * n
    if err_method is not None:
        df["err_method"] = [str(m) for m in err_method]
    _ssm = float(sigma_sys_mag) if sigma_sys_mag is not None and math.isfinite(float(sigma_sys_mag)) else float("nan")
    df["sigma_sys_mag"] = np.round(np.full(n, _ssm, dtype=np.float64), 6)
    df.to_csv(output_path, index=False)


# ---------------------------------------------------------------------------
# KROK 6: Výstup — PNG grafy
# ---------------------------------------------------------------------------


def save_lightcurve_png(
    output_path: Path,
    bjd: np.ndarray,
    mag_calib: np.ndarray,
    err: np.ndarray,
    flags: list[str],
    target_name: str,
    comp_quality: dict[str, dict],
    *,
    delta_mag_mode: bool = False,
    delta_mag: np.ndarray | None = None,
) -> None:
    """Uloží PNG graf svetelnej krivky s farebnými flagmi a comp status tabuľkou."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        logging.warning("[FÁZA 2A] matplotlib nie je dostupný, PNG sa nevygeneruje")
        return

    y_data = delta_mag if (delta_mag_mode and delta_mag is not None) else mag_calib
    y_label = "Δmag (ensemble)" if delta_mag_mode else "mag_calib"

    flag_colors = {
        "normal": "#1a1a2e",
        "saturated": "#aaaaaa",
        "outlier_hi": "#ff6b35",
        "outlier_lo": "#7b2d8b",
        "no_data": "#cccccc",
    }

    fig, (ax_lc, ax_comp) = plt.subplots(
        1,
        2,
        figsize=(14, 5),
        gridspec_kw={"width_ratios": [3, 1]},
    )
    fig.suptitle(f"VYVAR — {target_name}", fontsize=11, fontweight="bold")

    bjd_plot_all, bjd_axis_int = jd_series_relative(bjd)

    # Svetelná krivka
    for flag, color in flag_colors.items():
        mask = np.array([f == flag for f in flags])
        if not mask.any():
            continue
        bjd_f = bjd_plot_all[mask]
        y_f = y_data[mask]
        err_f = err[mask]
        valid = np.isfinite(y_f)
        if not valid.any():
            continue
        ax_lc.errorbar(
            bjd_f[valid],
            y_f[valid],
            yerr=err_f[valid],
            fmt="o",
            color=color,
            markersize=4,
            elinewidth=0.8,
            capsize=2,
            label=flag,
            alpha=0.85,
        )

    ax_lc.set_xlabel(jd_axis_title("BJD (TDB)", bjd_axis_int), fontsize=9)
    ax_lc.set_ylabel(y_label, fontsize=9)
    ax_lc.invert_yaxis()
    ax_lc.grid(True, alpha=0.3, linewidth=0.5)
    legend_patches = [
        mpatches.Patch(color=c, label=f) for f, c in flag_colors.items() if f != "no_data"
    ]
    ax_lc.legend(handles=legend_patches, fontsize=7, loc="upper right")

    # Comp quality tabuľka
    ax_comp.axis("off")
    comp_lines = []
    for i, (_, info) in enumerate(comp_quality.items(), 1):
        q = str(info["quality"])
        p2p = info.get("rms_p2p", float("nan"))
        icon = "[OK]" if q == "good" else ("[??]" if q == "suspect" else "[X]")
        p2p_str = f"{p2p:.4f}" if math.isfinite(float(p2p)) else "N/A"
        comp_lines.append(f"{icon} C{i:02d}  {p2p_str}  {q}")

    comp_text = "\n".join(comp_lines[:15])  # max 15 riadkov
    ax_comp.text(
        0.05,
        0.95,
        "Comparison Stars\n(rms_p2p | quality)\n\n" + comp_text,
        transform=ax_comp.transAxes,
        fontsize=7,
        verticalalignment="top",
        fontfamily="monospace",
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_cutout_png(
    output_path: Path,
    masterstar_fits_path: Path,
    xc: float,
    yc: float,
    target_name: str,
    *,
    size_px: int = 200,
    ms_data: np.ndarray | None = None,
) -> None:
    """Uloží výrez 200×200px z MASTERSTAR okolo targetu."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        return

    try:
        if ms_data is not None:
            data = np.asarray(ms_data, dtype=np.float64)
        else:
            with astrofits.open(masterstar_fits_path, memmap=False) as hdul:
                data = np.asarray(hdul[0].data, dtype=np.float64)
    except Exception:  # noqa: BLE001
        # EXC-0145: T3 -- Per-star cutout PNG export aborted when MASTERSTAR data cannot be loaded (EXCEPT-BULK-2 2026-07-08)
        return

    h, w = data.shape
    half = size_px // 2
    x0 = max(0, int(xc) - half)
    y0 = max(0, int(yc) - half)
    x1 = min(w, x0 + size_px)
    y1 = min(h, y0 + size_px)
    cutout = data[y0:y1, x0:x1]
    if cutout.size == 0:
        return

    # Percentilová škála
    vmin = float(np.percentile(cutout, 5))
    vmax = float(np.percentile(cutout, 99))

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cutout, origin="lower", cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")

    # Červený štvorecok pre target
    cx = xc - x0
    cy = yc - y0
    rect = mpatches.Rectangle(
        (cx - 10, cy - 10),
        20,
        20,
        linewidth=1.5,
        edgecolor="red",
        facecolor="none",
    )
    ax.add_patch(rect)
    ax.set_title(f"{target_name}", fontsize=8)
    ax.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def save_field_map_png(
    output_path: Path,
    masterstar_fits_path: Path,
    active_targets: pd.DataFrame,
    comp_df: pd.DataFrame,
    *,
    percentile_lo: float = 5.0,
    percentile_hi: float = 99.5,
    ms_data: np.ndarray | None = None,
) -> None:
    """Uloží prehľadový PNG celého poľa — červené=target, zelené=comp."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        return

    try:
        if ms_data is not None:
            data = np.asarray(ms_data, dtype=np.float64)
        else:
            with astrofits.open(masterstar_fits_path, memmap=False) as hdul:
                data = np.asarray(hdul[0].data, dtype=np.float64)
    except Exception:  # noqa: BLE001
        # EXC-0146: T3 -- Field map PNG export aborted when MASTERSTAR data cannot be loaded (EXCEPT-BULK-2 2026-07-08)
        return

    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return
    vmin = float(np.percentile(finite, percentile_lo))
    vmax = float(np.percentile(finite, percentile_hi))

    fig, ax = plt.subplots(figsize=(18, 12))
    ax.imshow(data, origin="lower", cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")

    # Dedup: never draw a comp marker on a known target (same Gaia catalog_id).
    target_ids: set[str] = set()
    try:
        for _, _trow in active_targets.iterrows():
            tid = _normalize_gaia_id(_trow.get("catalog_id", ""))
            if tid:
                target_ids.add(tid)
    except Exception:  # noqa: BLE001
        target_ids = set()

    # Zelené štvorčeky — comp hviezdy (unikátne pozície)
    comp_plotted: set[str] = set()
    skipped_as_target = 0
    for _, row in comp_df.iterrows():
        cid = _normalize_gaia_id(row.get("catalog_id", ""))
        if not cid or cid in comp_plotted:
            continue
        if cid in target_ids:
            skipped_as_target += 1
            continue
        comp_plotted.add(cid)
        try:
            xc, yc = float(row["x"]), float(row["y"])
        except (KeyError, TypeError, ValueError):
            continue
        rect = mpatches.Rectangle(
            (xc - 8, yc - 8),
            16,
            16,
            linewidth=1.0,
            edgecolor="#00cc44",
            facecolor="none",
        )
        ax.add_patch(rect)

    if skipped_as_target > 0:
        logging.info(f"[FIELD MAP] Skipped {int(skipped_as_target)} comp markers — star is a known target")

    # Target hviezdy — DAO-matched only (exclude catalog_only from field map)
    for _, row in active_targets.iterrows():
        z = str(row.get("zone_flag", row.get("zone", "")) or "").strip().lower()
        if z == "catalog_only":
            continue
        try:
            xc, yc = float(row["x"]), float(row["y"])
        except (KeyError, TypeError, ValueError):
            continue
        rect = mpatches.Rectangle(
            (xc - 12, yc - 12),
            24,
            24,
            linewidth=1.5,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(rect)
        name = _target_display_name(row)[:20]
        ax.text(xc + 14, yc, name, color="red", fontsize=5, va="center")

    ax.set_title(
        "VYVAR — Field Map (red=VSX target, green=comp star)",
        fontsize=10,
    )
    ax.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def save_target_field_map_png(
    output_path: Path,
    masterstar_fits_path: Path,
    target_row: pd.Series,
    comp_rows: pd.DataFrame,
    *,
    percentile_lo: float = 5.0,
    percentile_hi: float = 99.5,
    ms_data: np.ndarray | None = None,
) -> None:
    """Per-target field map: celé pole, červený štvorec=target, zelené krúžky=comp (číslované)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        return

    try:
        if ms_data is not None:
            data = np.asarray(ms_data, dtype=np.float64)
        else:
            with astrofits.open(masterstar_fits_path, memmap=False) as hdul:
                data = np.asarray(hdul[0].data, dtype=np.float64)
    except Exception:  # noqa: BLE001
        # EXC-0147: T3 -- Target field map PNG export aborted when MASTERSTAR data cannot be loaded (EXCEPT-BULK-2 2026-07-08)
        return

    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return
    vmin = float(np.percentile(finite, percentile_lo))
    vmax = float(np.percentile(finite, percentile_hi))

    fig, ax = plt.subplots(figsize=(18, 12))
    ax.imshow(data, origin="lower", cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")

    # Target — štandard červený štvorec; catalog_only cyan prerušovaný krúžok
    try:
        tx, ty = float(target_row["x"]), float(target_row["y"])
        _z_t = str(target_row.get("zone", "") or "").strip().lower()
        if _z_t == "catalog_only":
            circ_t = mpatches.Circle(
                (tx, ty),
                radius=16,
                linewidth=2.0,
                edgecolor="cyan",
                facecolor="none",
                linestyle="--",
            )
            ax.add_patch(circ_t)
            tcol = "cyan"
        else:
            rect_t = mpatches.Rectangle(
                (tx - 15, ty - 15),
                30,
                30,
                linewidth=2.0,
                edgecolor="red",
                facecolor="none",
            )
            ax.add_patch(rect_t)
            tcol = "red"
        tname = str(target_row.get("vsx_name", target_row.get("catalog_id", "T")))[:20]
        ax.text(
            tx + 18,
            ty,
            f"T: {tname}",
            color=tcol,
            fontsize=7,
            va="center",
            fontweight="bold",
        )
    except (KeyError, TypeError, ValueError):
        pass

    # Comp hviezdy — zelené krúžky s číslom (všetky, bez orezania)
    for i, (_, crow) in enumerate(comp_rows.iterrows(), 1):
        try:
            cx, cy = float(crow["x"]), float(crow["y"])
        except (KeyError, TypeError, ValueError):
            continue
        circ = mpatches.Circle(
            (cx, cy),
            radius=14,
            linewidth=1.5,
            edgecolor="#00cc44",
            facecolor="none",
        )
        ax.add_patch(circ)
        ax.text(
            cx + 16,
            cy,
            f"C{i:02d}",
            color="#00cc44",
            fontsize=7,
            va="center",
            fontweight="bold",
        )

    target_name = str(target_row.get("vsx_name", target_row.get("catalog_id", "")))
    ax.set_title(
        f"VYVAR — {target_name}\n"
        f"(red=VSX target, cyan=catalog_only (no DAO), green=comp star)",
        fontsize=10,
    )
    ax.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Hlavný wrapper — run_phase2a
# ---------------------------------------------------------------------------


_EDGE_FILTER_NOTE_OK = ""
_EDGE_FILTER_NOTE_FAILED = "EDGE-UNFILTERED: edge safety check failed"


def _edge_ok_from_masterstar_pipeline(
    masterstar_fits: Path,
    stars_df: pd.DataFrame,
    cfg_dict: dict[str, Any],
    *,
    ms_header: Any | None = None,
    ms_data: np.ndarray | None = None,
) -> tuple[pd.Series, bool]:
    """
    Per-star edge safety (annulus-aware, best-effort).

    Copy of UI logic (ui_variability._edge_ok_from_masterstar) without Streamlit dependency.
    Returns (edge_ok, edge_filter_failed). On failure, fail-open (all edge-ok) with flag set.
    """
    if stars_df is None or stars_df.empty:
        return pd.Series(dtype=bool), False
    masterstar_fits = Path(masterstar_fits)
    if not masterstar_fits.exists():
        LOGGER.warning(
            "[PHOT] edge-ok check failed (MASTERSTAR missing); treating all stars as edge-ok: %s",
            masterstar_fits,
        )
        return pd.Series(True, index=stars_df.index), True

    try:
        from astropy.io import fits as astrofits  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[PHOT] edge-ok check failed; treating all stars as edge-ok: %s", exc)
        return pd.Series(True, index=stars_df.index), True

    nx = ny = None
    fwhm_px = float("nan")
    hdr: Any | None = ms_header
    try:
        if hdr is None and masterstar_fits.is_file():
            with astrofits.open(masterstar_fits, memmap=False) as hdul:
                hdr = hdul[0].header
        if hdr is not None:
            try:
                fwhm_px = float(hdr.get("VY_FWHM", float("nan")))
            except (TypeError, ValueError):
                fwhm_px = float("nan")
            try:
                _n1 = int(hdr.get("NAXIS1", 0) or 0)
                _n2 = int(hdr.get("NAXIS2", 0) or 0)
                if _n1 > 0 and _n2 > 0:
                    nx, ny = _n1, _n2
            except (TypeError, ValueError):
                nx = ny = None
        if ms_data is not None and hasattr(ms_data, "shape") and len(ms_data.shape) >= 2:
            ny, nx = int(ms_data.shape[-2]), int(ms_data.shape[-1])
    except Exception:  # noqa: BLE001
        nx = ny = None

    try:
        base_margin = float(cfg_dict.get("phase01_chip_interior_margin_px", 100))
    except (TypeError, ValueError):
        base_margin = 100.0
    try:
        ann_outer_fwhm = float(cfg_dict.get("annulus_outer_fwhm", 9.0))
    except (TypeError, ValueError):
        ann_outer_fwhm = 9.0
    ann_margin = float(ann_outer_fwhm) * float(fwhm_px) + 5.0 if np.isfinite(fwhm_px) else float("nan")
    margin = float(base_margin)
    if np.isfinite(ann_margin):
        margin = max(float(margin), float(ann_margin))

    x = pd.to_numeric(stars_df.get("x"), errors="coerce")
    y = pd.to_numeric(stars_df.get("y"), errors="coerce")
    ok = np.isfinite(x) & np.isfinite(y)
    if nx is not None and ny is not None and nx > 0 and ny > 0 and np.isfinite(margin) and margin >= 0:
        ok = ok & (x >= margin) & (x <= float(nx) - margin) & (y >= margin) & (y <= float(ny) - margin)
    edge_filter_failed = nx is None or ny is None or nx <= 0 or ny <= 0
    if edge_filter_failed:
        LOGGER.warning(
            "[PHOT] edge-ok check incomplete (chip dims unknown); candidates EDGE-UNFILTERED"
        )
    return ok.fillna(False).astype(bool), bool(edge_filter_failed)


def stamp_vsx_known_variable_on_masterstars(
    ms_df: pd.DataFrame,
    variable_targets_df: pd.DataFrame | None,
    *,
    log_fn: Any | None = None,
    positional_fallback_arcsec: float = 8.0,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Set ``vsx_known_variable`` on masterstars by catalog_id join (primary).

    Positional matching is used only for variable-target rows without a Gaia ``catalog_id``.
    """
    from gaia_catalog_id import normalize_gaia_source_id_series

    out = ms_df.copy()
    if "vsx_known_variable" not in out.columns:
        out["vsx_known_variable"] = False
    else:
        out["vsx_known_variable"] = (
            pd.to_numeric(out["vsx_known_variable"], errors="coerce").fillna(0).astype(bool)
        )

    stats = {"id_join": 0, "positional_fallback": 0}
    if variable_targets_df is None or getattr(variable_targets_df, "empty", True):
        return out, stats

    vt = variable_targets_df.copy()
    vt_ids: set[str] = set()
    if "catalog_id" in vt.columns:
        vt_ids = {
            str(x).strip()
            for x in normalize_gaia_source_id_series(vt["catalog_id"]).tolist()
            if str(x).strip()
        }

    if vt_ids:
        ms_cid = normalize_gaia_source_id_series(out.get("catalog_id", pd.Series([""] * len(out))))
        id_hit = ms_cid.isin(vt_ids)
        stats["id_join"] = int(id_hit.sum())
        out.loc[id_hit, "vsx_known_variable"] = True

    vt_no_id = vt
    if "catalog_id" in vt.columns:
        vt_no_id = vt[normalize_gaia_source_id_series(vt["catalog_id"]).eq("")]
    if (
        not vt_no_id.empty
        and "ra_deg" in vt_no_id.columns
        and "dec_deg" in vt_no_id.columns
        and "ra_deg" in out.columns
        and "dec_deg" in out.columns
    ):
        try:
            from astropy.coordinates import SkyCoord  # noqa: PLC0415
            import astropy.units as u  # noqa: PLC0415

            v_ra = pd.to_numeric(vt_no_id["ra_deg"], errors="coerce")
            v_de = pd.to_numeric(vt_no_id["dec_deg"], errors="coerce")
            ok_v = v_ra.notna() & v_de.notna()
            if bool(ok_v.any()):
                ms_ra = pd.to_numeric(out["ra_deg"], errors="coerce")
                ms_de = pd.to_numeric(out["dec_deg"], errors="coerce")
                ok_m = ms_ra.notna() & ms_de.notna()
                if bool(ok_m.any()):
                    ms_coo = SkyCoord(
                        ra=ms_ra.loc[ok_m].astype(float).to_numpy() * u.deg,
                        dec=ms_de.loc[ok_m].astype(float).to_numpy() * u.deg,
                        frame="icrs",
                    )
                    vt_coo = SkyCoord(
                        ra=v_ra.loc[ok_v].astype(float).to_numpy() * u.deg,
                        dec=v_de.loc[ok_v].astype(float).to_numpy() * u.deg,
                        frame="icrs",
                    )
                    _idx, sep2d, _ = ms_coo.match_to_catalog_sky(vt_coo)
                    near = sep2d <= (float(positional_fallback_arcsec) * u.arcsec)
                    pos_idx = out.index[ok_m][np.asarray(near, dtype=bool)]
                    stats["positional_fallback"] = int(len(pos_idx))
                    if len(pos_idx):
                        out.loc[pos_idx, "vsx_known_variable"] = True
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("[PHOT] VSX positional fallback stamp skipped: %s", exc)

    if log_fn is not None:
        log_fn(
            f"MASTERSTAR vsx_known_variable stamp: catalog_id join={stats['id_join']} "
            f"positional_fallback={stats['positional_fallback']}"
        )
    return out, stats


def resolve_variable_targets_csv(
    *,
    comparison_stars_csv: Path | str | None = None,
    vsx_targets_csv: Path | str | None = None,
    platesolve_dir: Path | str | None = None,
    masterstar_fits_path: Path | str | None = None,
) -> Path | None:
    """Locate ``variable_targets.csv`` for VSX crossmatch (UI + headless layouts)."""
    if vsx_targets_csv is not None:
        explicit = Path(vsx_targets_csv)
        if explicit.is_file():
            LOGGER.info("[PHOT] variable_targets.csv resolved (explicit): %s", explicit)
            return explicit

    candidates: list[Path] = []
    if comparison_stars_csv:
        comp_parent = Path(comparison_stars_csv).resolve().parent
        candidates.append(comp_parent / "variable_targets.csv")
        candidates.append(comp_parent.parent / "variable_targets.csv")
    if platesolve_dir is not None:
        candidates.append(Path(platesolve_dir).resolve() / "variable_targets.csv")
    if masterstar_fits_path is not None:
        candidates.append(Path(masterstar_fits_path).resolve().parent / "variable_targets.csv")

    tried: list[str] = []
    seen: set[str] = set()
    for cand in candidates:
        key = str(cand.resolve()) if cand.is_absolute() else str(cand)
        if key in seen:
            continue
        seen.add(key)
        tried.append(key)
        if cand.is_file():
            LOGGER.info("[PHOT] variable_targets.csv resolved: %s", cand)
            return cand

    LOGGER.warning(
        "[PHOT] variable_targets.csv not found (VSX crossmatch disabled); tried: %s",
        "; ".join(tried) if tried else "(no candidates)",
    )
    return None


def auto_export_variability_candidates_csv(
    *,
    masterstar_fits_path: Path,
    comparison_stars_csv: Path | None,
    per_frame_csv_dir: Path,
    output_dir: Path,
    cfg: Any,
    flux_pivot: pd.DataFrame | None = None,
    csv_cache: dict[str, pd.DataFrame] | None = None,
    ms_header: Any | None = None,
    ms_data: np.ndarray | None = None,
    vsx_targets_csv: Path | None = None,
    platesolve_dir: Path | None = None,
) -> Path | None:
    """
    Pipeline variability detection (no UI): compute RMS + VDI and export candidates CSV.

    Produces `output_dir/variability_candidates.csv` using the same candidate mask as UI:
    `is_candidate_combined & ~vsx_known_variable & edge_ok`.
    """
    try:
        from variability_detector import compute_rms_variability, compute_vdi, load_field_flux_matrix  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[PHOT] variability candidates export failed: %s", exc)
        return None

    output_dir = Path(output_dir)
    cfg_dict = cfg.to_dict() if hasattr(cfg, "to_dict") else {}

    min_frames_frac = 1.0
    try:
        sigma_thr = float(cfg.variability_sigma_threshold)
    except Exception:  # noqa: BLE001
        sigma_thr = 2.3
    try:
        mag_limit = float(cfg.vsx_variable_targets_mag_limit)
    except Exception:  # noqa: BLE001
        mag_limit = 13.0

    flux_col = "dao_flux"
    fm, meta, _bjd = load_field_flux_matrix(
        Path(per_frame_csv_dir),
        flux_col=flux_col,
        min_frames_frac=float(min_frames_frac),
        config=dict(cfg_dict, variability_min_frames_frac=float(min_frames_frac)),
        flux_pivot=flux_pivot,
        csv_cache=csv_cache,
    )

    _at_path = Path(output_dir) / "active_targets.csv"
    if _at_path.is_file() and not meta.empty:
        try:
            _at_zf = pd.read_csv(_at_path, low_memory=False, dtype=_GAIA_ID_DTYPE)
            if "zone_flag" in _at_zf.columns and "catalog_id" in _at_zf.columns:
                _zf_map = (
                    _at_zf.drop_duplicates("catalog_id", keep="first")
                    .set_index("catalog_id")["zone_flag"]
                    .astype(str)
                )
                meta = meta.copy()
                meta["zone_flag"] = meta.index.astype(str).map(_zf_map)
        except Exception as exc:  # noqa: BLE001
            logging.error('[EXC-0148] zone_flag merge into variability-candidate metadata fails - exported rows lack zone_flag: %s', exc)
            pass
    comp_ids: list[str] = []
    if comparison_stars_csv:
        try:
            cdf = pd.read_csv(
                Path(comparison_stars_csv),
                low_memory=False,
                dtype={**_GAIA_ID_DTYPE, "target_catalog_id": str},
            )
            if "catalog_id" in cdf.columns:
                try:
                    from gaia_catalog_id import normalize_gaia_source_id  # noqa: PLC0415
                except Exception:  # noqa: BLE001
                    normalize_gaia_source_id = None  # type: ignore[assignment]
                comp_ids = [
                    (
                        str(normalize_gaia_source_id(x)).strip()  # type: ignore[misc]
                        if normalize_gaia_source_id is not None
                        else str(x).strip()
                    )
                    for x in cdf["catalog_id"].dropna().tolist()
                    if str(x).strip() and str(x).strip().lower() not in ("nan", "none")
                ]
        except Exception:  # noqa: BLE001
            comp_ids = []

    vsx_targets_csv = resolve_variable_targets_csv(
        comparison_stars_csv=comparison_stars_csv,
        vsx_targets_csv=vsx_targets_csv,
        platesolve_dir=platesolve_dir,
        masterstar_fits_path=masterstar_fits_path,
    )

    cfg_run = dict(cfg_dict)
    cfg_run["variability_sigma_threshold"] = float(sigma_thr)
    cfg_run["variability_mag_limit"] = float(mag_limit)
    cfg_run["variability_min_frames_frac"] = float(min_frames_frac)
    rms_df = compute_rms_variability(
        fm,
        meta,
        comp_ids,
        sigma_threshold=float(sigma_thr),
        vsx_targets_csv=vsx_targets_csv,
        config=cfg_run,
        comp_rms_map=None,
    )
    cfg_run2 = dict(cfg_dict)
    cfg_run2["variability_min_frames"] = int(cfg_run2.get("variability_min_frames", 30))
    vdi_df = compute_vdi(fm, meta, min_frames=30, config=cfg_run2)

    if rms_df is None or rms_df.empty:
        return None

    # Normalize Gaia IDs early to prevent float64/scientific-notation precision loss during merges.
    try:
        from gaia_catalog_id import normalize_gaia_source_id_series  # noqa: PLC0415

        if "catalog_id" in rms_df.columns:
            rms_df = rms_df.copy()
            rms_df["catalog_id"] = normalize_gaia_source_id_series(rms_df["catalog_id"])
        if vdi_df is not None and (not vdi_df.empty) and ("catalog_id" in vdi_df.columns):
            vdi_df = vdi_df.copy()
            vdi_df["catalog_id"] = normalize_gaia_source_id_series(vdi_df["catalog_id"])
    except Exception as exc:  # noqa: BLE001
        logging.error('[EXC-0149] catalog_id normalize on rms/vdi frames fails - float-corrupted Gaia IDs may break comp ...: %s', exc)
        pass

    results_df = rms_df.copy()
    if vdi_df is not None and (not vdi_df.empty) and ("catalog_id" in vdi_df.columns):
        results_df = results_df.merge(
            vdi_df[["catalog_id", "vdi_score", "vdi_z_score", "is_variable_candidate"]],
            on="catalog_id",
            how="left",
            suffixes=("_rms", "_vdi"),
        )
        results_df = results_df.rename(columns={"is_variable_candidate": "is_variable_candidate_vdi"})
    else:
        results_df["vdi_score"] = np.nan
        results_df["vdi_z_score"] = np.nan
        results_df["is_variable_candidate_vdi"] = False

    if "is_variable_candidate" in results_df.columns and "is_variable_candidate_rms" not in results_df.columns:
        results_df = results_df.rename(columns={"is_variable_candidate": "is_variable_candidate_rms"})
    results_df["is_variable_candidate_rms"] = results_df["is_variable_candidate_rms"].fillna(False).astype(bool)
    results_df["is_variable_candidate_vdi"] = results_df["is_variable_candidate_vdi"].fillna(False).astype(bool)
    results_df["is_candidate_combined"] = (
        results_df["is_variable_candidate_rms"] | results_df["is_variable_candidate_vdi"]
    )

    results_df["detection_method"] = "—"
    results_df.loc[results_df["is_variable_candidate_rms"], "detection_method"] = "RMS"
    results_df.loc[results_df["is_variable_candidate_vdi"], "detection_method"] = "VDI"
    results_df.loc[
        results_df["is_variable_candidate_rms"] & results_df["is_variable_candidate_vdi"],
        "detection_method",
    ] = "RMS+VDI"

    work = results_df.copy()
    work["is_candidate_combined"] = work["is_candidate_combined"].fillna(False).astype(bool)
    work["vsx_known_variable"] = work.get("vsx_known_variable", False)
    try:
        work["vsx_known_variable"] = pd.to_numeric(work["vsx_known_variable"], errors="coerce").fillna(False).astype(bool)
    except Exception:  # noqa: BLE001
        work["vsx_known_variable"] = work["vsx_known_variable"].fillna(False).astype(bool)

    _at_path = Path(output_dir) / "active_targets.csv"
    if _at_path.is_file() and "catalog_id" in work.columns:
        try:
            _at_zf = pd.read_csv(_at_path, low_memory=False, dtype=_GAIA_ID_DTYPE)
            if "zone_flag" in _at_zf.columns and "catalog_id" in _at_zf.columns:
                _zf_map = (
                    _at_zf.drop_duplicates("catalog_id", keep="first")
                    .set_index("catalog_id")["zone_flag"]
                    .astype(str)
                )
                work["zone_flag"] = work["catalog_id"].astype(str).map(_zf_map)
        except Exception as exc:  # noqa: BLE001
            logging.error('[EXC-0150] zone_flag map onto variability work frame fails - edge filter may use missing zone: %s', exc)
            pass

    edge_ok, edge_filter_failed = _edge_ok_from_masterstar_pipeline(
        Path(masterstar_fits_path),
        work,
        cfg_dict,
        ms_header=ms_header,
        ms_data=ms_data,
    )
    work["edge_ok"] = edge_ok.reindex(work.index).fillna(False).astype(bool)
    if edge_filter_failed:
        LOGGER.error(
            "[VARIABILITY] edge_filter_failed=True: candidate list is EDGE-UNFILTERED "
            "(scrutinize edge candidates manually)"
        )

    # EXACT cand_mask as UI
    vsx_known = rms_df["vsx_known_variable"].fillna(False).astype(bool)
    vsx_matched = rms_df["vsx_match"].fillna(False).astype(bool)
    cand_mask = work["is_candidate_combined"] & ~(vsx_known | vsx_matched) & work["edge_ok"]
    cand_df = work.loc[cand_mask].copy()
    cand_df["edge_filter_failed"] = bool(edge_filter_failed)
    cand_df["edge_filter_note"] = (
        _EDGE_FILTER_NOTE_FAILED if edge_filter_failed else _EDGE_FILTER_NOTE_OK
    )
    # Final guard: make sure exported IDs are stable strings (no trailing .0 etc.).
    try:
        from gaia_catalog_id import normalize_gaia_source_id_series  # noqa: PLC0415

        if "catalog_id" in cand_df.columns:
            cand_df["catalog_id"] = normalize_gaia_source_id_series(cand_df["catalog_id"])
    except Exception as exc:  # noqa: BLE001
        logging.error('[EXC-0151] Final cand_df catalog_id normalization fails - exported candidate IDs may retain .0 cor...: %s', exc)
        pass

    # Best-effort: if Gaia IDs are still corrupted (float64 precision loss), repair using RA/DEC nearest match
    # in the local Gaia DB with a slightly larger box than the generic repair script uses.
    try:
        import sqlite3  # noqa: PLC0415

        from repair_catalog_ids import _pick_gaia_table, _sep_arcsec  # noqa: PLC0415
        from gaia_catalog_id import normalize_gaia_source_id  # noqa: PLC0415

        gdb = Path(str(cfg.gaia_db_path or "").strip())
        if gdb.is_file() and (not cand_df.empty) and ("catalog_id" in cand_df.columns):
            # Need RA/DEC to repair.
            ra_col = "ra_deg" if "ra_deg" in cand_df.columns else ("ra" if "ra" in cand_df.columns else None)
            dec_col = "dec_deg" if "dec_deg" in cand_df.columns else ("dec" if "dec" in cand_df.columns else None)
            if ra_col and dec_col:
                con = sqlite3.connect(str(gdb))
                try:
                    tab = _pick_gaia_table(con)

                    def _exists_source_id(source_id: str) -> bool:
                        try:
                            sid_i = int(str(source_id).strip())
                        except (TypeError, ValueError):
                            return False
                        r = con.execute(
                            f"SELECT source_id FROM {tab} WHERE source_id=? LIMIT 1;",
                            (sid_i,),
                        ).fetchone()
                        return bool(r and r[0] is not None)

                    box = 0.02  # deg (~72 arcsec); filter later by max_sep_arcsec
                    max_sep = 60.0  # arcsec (only applied for missing/clearly-bad IDs)
                    for i in range(len(cand_df)):
                        old = str(cand_df.at[i, "catalog_id"] if i in cand_df.index else "")
                        if old and _exists_source_id(old):
                            continue
                        ra0 = cand_df.at[i, ra_col]
                        dec0 = cand_df.at[i, dec_col]
                        try:
                            ra_f = float(ra0)
                            dec_f = float(dec0)
                        except (TypeError, ValueError) as exc:
                            LOGGER.debug("[CSV] Skipping row due to parse error: %s", exc)
                            continue
                        if not (math.isfinite(ra_f) and math.isfinite(dec_f)):
                            continue
                        row = con.execute(
                            f"""
                            SELECT source_id, ra, dec
                            FROM {tab}
                            WHERE ra  BETWEEN ? AND ?
                              AND dec BETWEEN ? AND ?
                            ORDER BY (ra-?)*(ra-?) + (dec-?)*(dec-?)
                            LIMIT 1
                            """,
                            (ra_f - box, ra_f + box, dec_f - box, dec_f + box, ra_f, ra_f, dec_f, dec_f),
                        ).fetchone()
                        if not row or row[0] is None or row[1] is None or row[2] is None:
                            continue
                        new_id = int(row[0])
                        sep = float(_sep_arcsec(ra_f, dec_f, float(row[1]), float(row[2])))
                        # Apply relaxed threshold only for obviously corrupted values (common float64 symptom: trailing zeros).
                        if old.strip().endswith("000"):
                            ok_rep = math.isfinite(sep) and sep <= max_sep
                        else:
                            ok_rep = math.isfinite(sep) and sep <= 10.0
                        if ok_rep:
                            cand_df.at[i, "catalog_id"] = str(new_id)
                finally:
                    con.close()
    except Exception:  # noqa: BLE001
        # EXC-0152: T2 -- Gaia DB con.close() after catalog_id repair ignored (EXCEPT-BULK-2 2026-07-08)
        pass

    # If Gaia DB doesn't cover the box (or is too sparse), repair from MASTERSTAR catalog (best effort).
    try:
        ms_csv = Path(output_dir).parent / "masterstars_full_match.csv"
        if ms_csv.is_file() and (not cand_df.empty) and ("catalog_id" in cand_df.columns):
            ra_col = "ra_deg" if "ra_deg" in cand_df.columns else ("ra" if "ra" in cand_df.columns else None)
            dec_col = "dec_deg" if "dec_deg" in cand_df.columns else ("dec" if "dec" in cand_df.columns else None)
            if ra_col and dec_col:
                ms = pd.read_csv(ms_csv, low_memory=False, dtype=_GAIA_ID_DTYPE)
                if "ra_deg" in ms.columns and "dec_deg" in ms.columns:
                    ms_ra = pd.to_numeric(ms["ra_deg"], errors="coerce")
                    ms_de = pd.to_numeric(ms["dec_deg"], errors="coerce")
                    ok_ms = np.isfinite(ms_ra) & np.isfinite(ms_de)
                    ms = ms.loc[ok_ms].copy()
                    ms_ra = ms_ra.loc[ok_ms]
                    ms_de = ms_de.loc[ok_ms]
                    # For each candidate, pick nearest MS row in a small box and replace catalog_id by MS `name`/`catalog_id`.
                    box = 0.01
                    max_sep = 3.0  # arcsec
                    for idx in cand_df.index.tolist():
                        old = str(cand_df.at[idx, "catalog_id"] or "").strip()
                        if old and old.isdigit() and len(old) >= 18 and not old.endswith("000"):
                            continue
                        try:
                            ra_f = float(cand_df.at[idx, ra_col])
                            de_f = float(cand_df.at[idx, dec_col])
                        except (TypeError, ValueError) as exc:
                            LOGGER.debug("[CSV] Skipping row due to parse error: %s", exc)
                            continue
                        if not (math.isfinite(ra_f) and math.isfinite(de_f)):
                            continue
                        m = (ms_ra.between(ra_f - box, ra_f + box)) & (ms_de.between(de_f - box, de_f + box))
                        if not bool(m.any()):
                            continue
                        sub = ms.loc[m].copy()
                        if sub.empty:
                            continue
                        sub_ra = ms_ra.loc[m].to_numpy(dtype=float)
                        sub_de = ms_de.loc[m].to_numpy(dtype=float)
                        best_i = None
                        best_sep = float("inf")
                        for j in range(len(sub)):
                            sep = float(_sep_arcsec(ra_f, de_f, float(sub_ra[j]), float(sub_de[j])))
                            if math.isfinite(sep) and sep < best_sep:
                                best_sep = sep
                                best_i = j
                        if best_i is None or not (math.isfinite(best_sep) and best_sep <= max_sep):
                            continue
                        r = sub.iloc[int(best_i)]
                        cand_id = normalize_gaia_source_id(r.get("name")) or normalize_gaia_source_id(r.get("catalog_id"))
                        if cand_id:
                            cand_df.at[idx, "catalog_id"] = str(cand_id)
    except Exception as exc:  # noqa: BLE001
        logging.error('[EXC-0153] MASTERSTAR NN catalog_id repair block fails - corrupted comp IDs may persist in candida...: %s', exc)
        pass

    export_cols = [
        c
        for c in [
            "catalog_id",
            "ra_deg",
            "dec_deg",
            "mag",
            "bp_rp",
            "rms_pct",
            "smoothness_ratio",
            "vdi_score",
            "vdi_z_score",
            "detection_method",
            "variability_score",
            "zone",
            "zone_flag",
            "edge_filter_failed",
            "edge_filter_note",
            "vsx_known_variable",
            "vsx_match",
            "gaia_dr3_variable_catalog",
        ]
        if c in cand_df.columns
    ]

    out_csv = output_dir / "variability_candidates.csv"
    try:
        if export_cols:
            cand_df[export_cols].to_csv(out_csv, index=False)
        else:
            cand_df.to_csv(out_csv, index=False)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[PHOT] variability candidates export step failed: %s", exc)
        return None

    logging.info("[VARIABILITY] Auto-export: %d kandidatov -> %s", int(len(cand_df)), str(out_csv))
    return out_csv


def _phase2a_coerce_skip_photometry(df: pd.DataFrame) -> pd.Series:
    """Normalize skip_photometry / saturated zone_flag into a boolean Series."""
    if "skip_photometry" in df.columns:

        def _to_b(x: Any) -> bool:
            if isinstance(x, (bool, np.bool_)):
                return bool(x)
            s = str(x).strip().lower()
            return s in ("1", "true", "yes", "t")

        return df["skip_photometry"].map(_to_b).fillna(False).astype(bool)
    if "zone_flag" in df.columns:
        zf = df["zone_flag"].astype(str).str.strip().str.lower()
        return zf.eq("saturated").astype(bool)
    return pd.Series(False, index=df.index, dtype=bool)


def build_rms_mag_model(
    summary_rows: list[dict],
    *,
    zone_filter: tuple[str, ...] = ("linear", "noisy1"),
    min_stars: int = 10,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Fit mag-dependent RMS baseline from stable comp/target stars.

    Uses only ``zone_filter`` stars with valid ``lc_rms`` and ``lc_median_mag``.
    Fits: log10(rms) = a * mag + b.

    Returns:
        (coeffs, mags_used) where coeffs = [a, b] from ``np.polyfit(mag, log10(rms), 1)``,
        or None if fewer than ``min_stars`` valid points.
    """
    mags: list[float] = []
    rmss: list[float] = []
    for row in summary_rows:
        zf = str(row.get("zone_flag", "") or "").strip().lower()
        if zf not in zone_filter:
            continue
        try:
            rms = float(row.get("lc_rms", float("nan")))
            mag = float(row.get("lc_median_mag", float("nan")))
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(rms) and math.isfinite(mag)):
            continue
        if rms <= 0 or mag < 6.0 or mag > 20.0:
            continue
        mags.append(mag)
        rmss.append(rms)

    if len(mags) < int(min_stars):
        logging.warning(
            "RMS model fit skipped: only %d stars (need %d)",
            len(mags),
            int(min_stars),
        )
        return None

    mags_arr = np.asarray(mags, dtype=float)
    rmss_arr = np.asarray(rmss, dtype=float)
    coeffs = _safe_polyfit(mags_arr, np.log10(rmss_arr), 1)
    if coeffs is None:
        logging.warning("RMS model fit skipped: degenerate mag axis")
        return None
    a, b = float(coeffs[0]), float(coeffs[1])
    logging.info(
        "RMS model fit: %d stars, slope=%.3f, intercept=%.3f",
        len(mags),
        a,
        b,
    )
    return coeffs, mags_arr


def expected_rms_from_model(mag: float, coeffs: np.ndarray) -> float:
    """Return expected RMS for given mag from polyfit coefficients."""
    try:
        m = float(mag)
        if not math.isfinite(m):
            return float("nan")
        log_rms = float(np.polyval(coeffs, m))
        if not math.isfinite(log_rms):
            return float("nan")
        return float(10.0**log_rms)
    except Exception:  # noqa: BLE001
        # EXC-0154: T4 -- RMS-vs-mag model evaluation failure returns NaN expected_rms - quality flag may show no... (EXCEPT-BULK-2 2026-07-08)
        return float("nan")


def classify_lc_quality(
    zone_flag: str,
    lc_rms: float,
    lc_median_mag: float,
    n_frames: int,
    n_normal_frames: int,
    lunar_risk: str = "UNKNOWN",
    *,
    rms_model_coeffs: np.ndarray | None = None,
    rms_noisy_k: float = 3.0,
    min_frames: int = 20,
    short_min_frames: int = 3,
    min_normal_frac: float = 0.5,
) -> str:
    """Classify light-curve quality: saturated | no_data | short_baseline | noisy | noisy_moon | good."""
    try:
        zf = str(zone_flag or "").strip().lower()
    except Exception:  # noqa: BLE001
        zf = ""

    if zf == "saturated":
        return "saturated"

    try:
        nf = int(n_frames)
    except (TypeError, ValueError):
        nf = 0
    try:
        nn = int(n_normal_frames)
    except (TypeError, ValueError):
        nn = 0
    nn = max(0, nn)
    nf = max(0, nf)

    short_min = int(short_min_frames)
    min_f = int(min_frames)
    min_frac = float(min_normal_frac)

    if nf < short_min:
        return "no_data"
    if nf < min_f:
        if nf > 0 and (nn / float(nf)) >= min_frac:
            return "short_baseline"
        return "no_data"
    if nf > 0 and (nn / float(nf)) < min_frac:
        return "no_data"

    noisy_from_zone = zf in ("noisy2", "noisy3")
    noisy_from_rms = False
    if rms_model_coeffs is not None:
        try:
            rms = float(lc_rms)
            mag = float(lc_median_mag)
            if math.isfinite(rms) and math.isfinite(mag) and rms > 0:
                expected = expected_rms_from_model(mag, rms_model_coeffs)
                if math.isfinite(expected) and expected > 0 and rms > float(rms_noisy_k) * expected:
                    noisy_from_rms = True
        except Exception:  # noqa: BLE001
            noisy_from_rms = False

    if noisy_from_zone:
        return "noisy"
    if noisy_from_rms:
        moon = str(lunar_risk or "UNKNOWN").strip().upper()
        if moon == "HIGH":
            return "noisy_moon"
        return "noisy"
    return "good"


_LC_QUALITY_FLAGS: tuple[str, ...] = (
    "good",
    "noisy",
    "noisy_moon",
    "short_baseline",
    "no_data",
    "saturated",
)


def build_lc_quality_summary(
    summary_rows: list[dict],
    *,
    rms_model_coeffs: np.ndarray | None = None,
    rms_model_n_stars: int = 0,
    rms_noisy_k: float = 3.0,
) -> dict[str, Any]:
    """Aggregate ``lc_quality_flag`` counts for ``pipeline_meta.json``."""
    counts = {f: 0 for f in _LC_QUALITY_FLAGS}
    has_flag = False
    for row in summary_rows:
        if "lc_quality_flag" not in row:
            continue
        has_flag = True
        qf = str(row.get("lc_quality_flag", "") or "").strip().lower()
        if qf in counts:
            counts[qf] += 1
    total = int(sum(counts.values())) if has_flag else 0
    slope = float("nan")
    intercept = float("nan")
    if rms_model_coeffs is not None:
        try:
            slope = float(rms_model_coeffs[0])
            intercept = float(rms_model_coeffs[1])
        except Exception as exc:  # noqa: BLE001
            logging.error('[EXC-0155] RMS model slope/intercept not written to lc_quality summary metadata: %s', exc)
            pass
    out: dict[str, Any] = {
        "good": int(counts["good"]),
        "noisy": int(counts["noisy"]),
        "noisy_moon": int(counts["noisy_moon"]),
        "short_baseline": int(counts["short_baseline"]),
        "no_data": int(counts["no_data"]),
        "saturated": int(counts["saturated"]),
        "total": total,
        "rms_model_slope": slope,
        "rms_model_intercept": intercept,
        "rms_model_n_stars": int(rms_model_n_stars),
        "rms_noisy_k": float(rms_noisy_k),
        "available": bool(has_flag),
    }
    if not has_flag:
        out.update({f: 0 for f in _LC_QUALITY_FLAGS})
        out["total"] = 0
    return out


def build_gs11_summary(
    summary_rows: list[dict[str, Any]],
    cfg: Any,
    *,
    comps_gs11_rejected: int = 0,
    plate_scale_arcsec: float = 1.3,
) -> dict[str, Any]:
    """Aggregate GS11 dilution stats for ``pipeline_meta.json``."""
    enabled = bool(cfg.gs11_dilution_enabled)
    min_d = float(cfg.gs11_target_min_dilution)
    ap_cfg = float(cfg.gs11_dilution_aperture_arcsec)
    aperture_arcsec = ap_cfg if math.isfinite(ap_cfg) and ap_cfg > 0 else float("nan")
    corrections_mmag: list[float] = []
    targets_corrected = 0
    targets_skipped_low_d = 0
    if enabled:
        ap_samples_gs11: list[float] = []
        for row in summary_rows:
            try:
                d = float(row.get("dilution_factor", 1.0))
            except (TypeError, ValueError):
                d = 1.0
            try:
                dm = float(row.get("dilution_delta_mag", 0.0))
            except (TypeError, ValueError):
                dm = 0.0
            if not math.isfinite(d):
                continue
            try:
                gs11_ap = float(row.get("gs11_aperture_arcsec", float("nan")))
            except (TypeError, ValueError):
                gs11_ap = float("nan")
            if math.isfinite(gs11_ap) and gs11_ap > 0:
                ap_samples_gs11.append(gs11_ap)
            if d < 1.0 and d < min_d:
                targets_skipped_low_d += 1
            elif d < 1.0 and d >= min_d and math.isfinite(dm) and dm > 0:
                targets_corrected += 1
                corrections_mmag.append(float(dm) * 1000.0)
        if not (math.isfinite(aperture_arcsec) and aperture_arcsec > 0) and ap_samples_gs11:
            aperture_arcsec = float(np.median(np.asarray(ap_samples_gs11, dtype=np.float64)))
        if not (math.isfinite(aperture_arcsec) and aperture_arcsec > 0):
            aperture_arcsec = float(plate_scale_arcsec)
    med_mmag = float(np.median(corrections_mmag)) if corrections_mmag else 0.0
    max_mmag = float(np.max(corrections_mmag)) if corrections_mmag else 0.0
    return {
        "enabled": enabled,
        "aperture_arcsec": float(aperture_arcsec) if math.isfinite(aperture_arcsec) else float("nan"),
        "comps_gs11_rejected": int(comps_gs11_rejected),
        "targets_corrected": int(targets_corrected),
        "targets_skipped_low_d": int(targets_skipped_low_d),
        "median_correction_mmag": med_mmag,
        "max_correction_mmag": max_mmag,
    }


def _phase2a_write_summary(
    summary_rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    lunar_context: dict[str, Any] | None = None,
    cfg: Any | None = None,
    plate_scale_arcsec: float = 1.3,
) -> tuple[Path, pd.DataFrame]:
    """Write photometry_summary.csv. Returns path and written frame."""
    _cfg_summary = cfg or AppConfig()
    _lunar_risk = "UNKNOWN"
    if isinstance(lunar_context, dict):
        _lunar_risk = str(lunar_context.get("lunar_risk", "UNKNOWN") or "UNKNOWN")

    _rms_fit = build_rms_mag_model(summary_rows)
    _rms_coeffs = _rms_fit[0] if _rms_fit is not None else None

    for row in summary_rows:
        try:
            nf = int(row.get("n_frames") or 0)
        except (TypeError, ValueError):
            nf = 0
        try:
            ns = int(row.get("n_saturated") or 0)
        except (TypeError, ValueError):
            ns = 0
        n_normal = max(0, nf - ns)
        row["lc_quality_flag"] = classify_lc_quality(
            zone_flag=str(row.get("zone_flag", "") or ""),
            lc_rms=float(row.get("lc_rms", float("nan"))),
            lc_median_mag=float(row.get("lc_median_mag", float("nan"))),
            n_frames=nf,
            n_normal_frames=n_normal,
            lunar_risk=_lunar_risk,
            rms_model_coeffs=_rms_coeffs,
            min_frames=int(getattr(_cfg_summary, "lc_quality_min_frames", 20) or 20),
            short_min_frames=int(getattr(_cfg_summary, "lc_quality_short_min_frames", 3) or 3),
            min_normal_frac=float(getattr(_cfg_summary, "lc_quality_min_normal_frac", 0.5) or 0.5),
        )

    summary_csv = Path(output_dir) / "photometry_summary.csv"
    _sum_df = pd.DataFrame(summary_rows)
    try:
        from gaia_catalog_id import normalize_gaia_source_id_series  # noqa: PLC0415

        if "catalog_id" in _sum_df.columns:
            _sum_df["catalog_id"] = normalize_gaia_source_id_series(_sum_df["catalog_id"])
    except Exception as exc:  # noqa: BLE001
        logging.error('[EXC-0156] photometry_summary.csv catalog_id normalization fails - summary rows may carry float-tr...: %s', exc)
        pass

    _rms_n_stars = int(len(_rms_fit[1])) if _rms_fit is not None else 0
    _qs = build_lc_quality_summary(
        _sum_df.to_dict("records"),
        rms_model_coeffs=_rms_coeffs,
        rms_model_n_stars=_rms_n_stars,
    )
    merge_photometry_pipeline_meta(output_dir, {"lc_quality_summary": _qs})

    _cfg_gs11 = _cfg_summary
    _comps_gs11_rej = 0
    try:
        _meta_path = Path(output_dir) / "pipeline_meta.json"
        if _meta_path.is_file():
            _prev = json.loads(_meta_path.read_text(encoding="utf-8"))
            _gs11_prev = _prev.get("gs11_summary")
            if isinstance(_gs11_prev, dict):
                _comps_gs11_rej = int(_gs11_prev.get("comps_gs11_rejected", 0) or 0)
    except Exception as exc:  # noqa: BLE001
        logging.error('[EXC-0157] Prior gs11_summary comps_gs11_rejected count not loaded - GS11 summary undercounts reje...: %s', exc)
        pass
    _gs11 = build_gs11_summary(
        _sum_df.to_dict("records"),
        _cfg_gs11,
        comps_gs11_rejected=_comps_gs11_rej,
        plate_scale_arcsec=float(plate_scale_arcsec),
    )
    merge_photometry_pipeline_meta(output_dir, {"gs11_summary": _gs11})

    _sum_df.to_csv(summary_csv, index=False)
    return summary_csv, _sum_df


def _phase2a_observer_location_dict(
    cfg: object,
    site: tuple[float | None, float | None, float | None] | None = None,
    site_source: str | None = None,
) -> dict[str, Any]:
    """Observer site for ``pipeline_meta.json`` (AAVSO/VAR.ASTRO use same cfg fields).

    When ``site`` (the per-draft resolved location) is provided it overrides the
    config coordinates so the exported metadata matches the location actually used
    for BJD/airmass (no config-drift mismatch).
    """
    try:
        loc_id = int(cfg.observer_location_id)
    except (TypeError, ValueError):
        loc_id = 0
    try:
        lat = float(cfg.observer_lat)
    except (TypeError, ValueError):
        lat = 0.0
    try:
        lon = float(cfg.observer_lon)
    except (TypeError, ValueError):
        lon = 0.0
    try:
        alt_m = float(cfg.observer_alt_m)
    except (TypeError, ValueError):
        alt_m = 0.0
    out_source = "config"
    if site is not None and site[0] is not None and site[1] is not None:
        lat = float(site[0])
        lon = float(site[1])
        alt_m = float(site[2]) if site[2] is not None else 0.0
        out_source = str(site_source or "resolved")
    return {
        "name": str(cfg.observer_location_name or "").strip(),
        "lat": lat,
        "lon": lon,
        "alt_m": alt_m,
        "location_id": loc_id,
        "source": out_source,
    }


def _fits_header_facts(header: Any) -> dict[str, Any]:
    """Best-effort filter / exptime / binning from a FITS header (metadata only)."""
    out: dict[str, Any] = {"filter": None, "exptime_s": None, "binning": None}
    if header is None:
        return out
    try:
        _flt = header.get("FILTER")
        if _flt is not None and str(_flt).strip():
            out["filter"] = str(_flt).strip()
    except Exception:  # noqa: BLE001
        pass
    try:
        _exp = header.get("EXPTIME", header.get("EXPOSURE"))
        if _exp is not None and str(_exp).strip() != "":
            out["exptime_s"] = float(_exp)
    except Exception:  # noqa: BLE001
        pass
    try:
        _xb = header.get("XBINNING")
        _yb = header.get("YBINNING")
        if _xb is not None and _yb is not None:
            out["binning"] = f"{int(float(_xb))}x{int(float(_yb))}"
        elif _xb is not None:
            out["binning"] = f"{int(float(_xb))}x{int(float(_xb))}"
    except Exception:  # noqa: BLE001
        pass
    return out


def _build_phase2a_resolved_facts(
    *,
    cfg: Any,
    gain_res: Any,
    rn_res: Any,
    gain_value: float | None,
    rn_value: float | None,
    site: Any,
    sat_limit: float | None,
    plate_scale_arcsec: float | None,
    frame_width_px: int | None,
    frame_height_px: int | None,
    ms_header: Any,
    obs_group: str,
) -> dict[str, Any]:
    """Run-effective resolved facts for ``pipeline_meta.resolved_facts`` (metadata only).

    Records the values the run ACTUALLY used (with sources for gain/read-noise/site) so the
    report can show an honest "resolved facts" block. Numeric behaviour is unchanged: this is
    never read by the science path and the anchor comparator ignores ``pipeline_meta.json``.
    """
    def _num_or_none(v: Any) -> float | None:
        try:
            f = float(v)
            return f if math.isfinite(f) else None
        except (TypeError, ValueError):
            return None

    _hdr = _fits_header_facts(ms_header)
    if not _hdr.get("filter"):
        _hdr["filter"] = str(obs_group or "") or None

    # Site: coordinates from the per-draft resolver when it succeeded, else config; id/name
    # mirror pipeline_meta.observer_location (drawn from cfg).
    site_ok = bool(getattr(site, "ok", False))
    if site_ok:
        site_lat = _num_or_none(getattr(site, "lat", None))
        site_lon = _num_or_none(getattr(site, "lon", None))
        site_alt = _num_or_none(getattr(site, "elev", None))
    else:
        site_lat = _num_or_none(getattr(cfg, "observer_lat", None))
        site_lon = _num_or_none(getattr(cfg, "observer_lon", None))
        site_alt = _num_or_none(getattr(cfg, "observer_alt_m", None))
    try:
        loc_id = int(getattr(cfg, "observer_location_id", 0) or 0)
    except (TypeError, ValueError):
        loc_id = 0

    return {
        "site": {
            "location_id": loc_id,
            "name": str(getattr(cfg, "observer_location_name", "") or "").strip(),
            "lat": site_lat,
            "lon": site_lon,
            "alt_m": site_alt,
            "source": str(getattr(site, "source", "unresolved")),
            "ok": site_ok,
        },
        "gain": {
            "value": _num_or_none(gain_value),
            "source": (getattr(gain_res, "source", None) if getattr(gain_res, "ok", False) else "default"),
            "key": getattr(gain_res, "key", None),
        },
        "read_noise": {
            "value": _num_or_none(rn_value),
            "source": (getattr(rn_res, "source", None) if getattr(rn_res, "ok", False) else "default"),
            "key": getattr(rn_res, "key", None),
        },
        "saturation_adu": _num_or_none(sat_limit),
        "plate_scale_arcsec_per_px": _num_or_none(plate_scale_arcsec),
        "frame_width_px": int(frame_width_px) if frame_width_px else None,
        "frame_height_px": int(frame_height_px) if frame_height_px else None,
        "binning": _hdr["binning"],
        "filter": _hdr["filter"],
        "exptime_s": _hdr["exptime_s"],
    }


_GIT_PROVENANCE_WARNED = False
# src_py/photometry_core.py -> repo root is parent.parent (git cwd + porcelain path base).
_REPO_ROOT_FOR_PROVENANCE = Path(__file__).resolve().parent.parent


def _is_import_relevant_py_path(path: str) -> bool:
    """True for VYVAR modules imported by the pipeline: ``src_py/*.py`` plus the root ``app.py`` shim.

    Everything under ``dev/`` (tests, scripts, tools, validation, sandbox, orchestrator) and
    ``tmp/`` / ``docs/`` is scratch: it never trips the FAIL-CLOSED dirty-code gate (T3 FIX B).
    """
    p = path.replace("\\", "/").lstrip("./")
    if not p.endswith(".py"):
        return False
    if p == "app.py":  # thin root Streamlit shim (the only import-relevant module at repo root)
        return True
    return p.startswith("src_py/")


def _porcelain_status_by_path(porcelain: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in porcelain.splitlines():
        if not line.strip() or len(line) < 4:
            continue
        status = line[:2]
        path_part = line[3:].strip()
        if " -> " in path_part:
            path_part = path_part.split(" -> ", 1)[-1].strip()
        path_part = path_part.strip('"').replace("\\", "/")
        out[path_part] = status
    return out


def classify_git_dirty_paths(
    porcelain: str,
    dirty_files: Sequence[dict[str, str]],
) -> tuple[bool, list[str], list[str]]:
    """Split dirty paths into import-relevant code vs scratch (F-431 / T3 dirty-gate).

    ``dirty_code`` = tracked modifications to import-relevant ``*.py`` OR untracked
    import-relevant ``*.py`` (repo root only). Everything else is ``dirty_scratch``.
    """
    status_by_path = _porcelain_status_by_path(porcelain)
    code_paths: list[str] = []
    scratch_paths: list[str] = []
    for entry in dirty_files:
        path = str(entry.get("path") or "").replace("\\", "/")
        if not path or path == "…truncated…":
            continue
        status = status_by_path.get(path, "??")
        is_import_py = _is_import_relevant_py_path(path)
        is_untracked = status.startswith("??")
        is_tracked_mod = not is_untracked
        if is_import_py and (is_tracked_mod or is_untracked):
            code_paths.append(path)
        else:
            scratch_paths.append(path)
    return bool(code_paths), code_paths, scratch_paths


def _resolve_git_provenance() -> tuple[str | None, bool | None, list[dict[str, str]]]:
    """Return (HEAD hash, dirty flag, dirty file rows) from repo root; nulls when git unavailable.

    When dirty, each row is ``{path, content_sha256}`` for tracked/untracked paths listed by
    ``git status --porcelain`` (F-431 provenance hardening).
    """
    global _GIT_PROVENANCE_WARNED
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT_FOR_PROVENANCE,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=_REPO_ROOT_FOR_PROVENANCE,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        dirty = bool(status.strip())
        dirty_files: list[dict[str, str]] = []
        if dirty:
            import hashlib

            for line in status.splitlines():
                if not line.strip():
                    continue
                # porcelain: XY PATH or XY ORIG -> PATH
                path_part = line[3:].strip() if len(line) > 3 else line.strip()
                if " -> " in path_part:
                    path_part = path_part.split(" -> ", 1)[-1].strip()
                path_part = path_part.strip('"')
                fp = _REPO_ROOT_FOR_PROVENANCE / path_part
                sha = ""
                try:
                    if fp.is_file():
                        h = hashlib.sha256()
                        with fp.open("rb") as fh:
                            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                                h.update(chunk)
                        sha = h.hexdigest()
                    elif fp.is_dir():
                        sha = "DIR"
                    else:
                        sha = "MISSING"
                except OSError:
                    sha = "UNREADABLE"
                dirty_files.append({"path": path_part.replace("\\", "/"), "content_sha256": sha})
                if len(dirty_files) >= 200:
                    dirty_files.append({"path": "…truncated…", "content_sha256": ""})
                    break
        return (head or None), dirty, dirty_files
    except Exception:  # noqa: BLE001
        if not _GIT_PROVENANCE_WARNED:
            LOGGER.warning(
                "[PHOT] pipeline provenance: git unavailable; git_hash/git_dirty set to null"
            )
            _GIT_PROVENANCE_WARNED = True
        return None, None, []


def _build_pipeline_provenance_block(cfg: Any, *, entry_point: str) -> dict[str, Any]:
    """Run provenance stamped into ``pipeline_meta.json`` (last writer wins)."""
    git_hash, git_dirty, dirty_files = _resolve_git_provenance()
    if hasattr(cfg, "to_dict"):
        config_snapshot = cfg.to_dict()
    elif hasattr(cfg, "to_json"):
        config_snapshot = cfg.to_json()
    else:
        from dataclasses import asdict

        config_snapshot = asdict(cfg)
    block: dict[str, Any] = {
        "git_hash": git_hash,
        "git_dirty": git_dirty,
        "config_snapshot": config_snapshot,
        "stamped_at_utc": datetime.now(timezone.utc).isoformat(),
        "entry_point": entry_point,
        "labbe_rng_seed_policy": "content_frame_hash_v1",
    }
    if git_dirty is True:
        block["git_dirty_files"] = dirty_files
        try:
            porcelain = subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=_REPO_ROOT_FOR_PROVENANCE,
                text=True,
                stderr=subprocess.DEVNULL,
            )
        except Exception:  # noqa: BLE001
            porcelain = ""
        code_dirty, code_paths, scratch_paths = classify_git_dirty_paths(porcelain, dirty_files)
        block["git_dirty_code"] = code_dirty
        block["git_dirty_code_files"] = code_paths
        block["git_dirty_scratch_files"] = scratch_paths
    elif git_dirty is False:
        block["git_dirty_code"] = False
    else:
        block["git_dirty_code"] = None
    return block


def _sky_surface_meta_from_qc(draft_dir: Path | None) -> dict[str, Any]:
    """Read preprocess sky-surface summary from draft ``qc_metrics.csv`` for pipeline_meta."""
    if draft_dir is None:
        return {}
    try:
        from pipeline import find_qc_metrics_csv  # noqa: PLC0415

        qc_path = find_qc_metrics_csv(Path(draft_dir))
        if qc_path is None or not qc_path.is_file():
            return {}
        qdf = pd.read_csv(qc_path, low_memory=False)
        if qdf.empty or "sky_surface_applied" not in qdf.columns:
            return {}
        applied = qdf["sky_surface_applied"].fillna(False).astype(bool)
        n_applied = int(applied.sum())
        order = 0
        if "sky_surface_order" in qdf.columns and n_applied > 0:
            order = int(
                pd.to_numeric(qdf.loc[applied, "sky_surface_order"], errors="coerce").dropna().mode().iloc[0]
            )
        p2p_med = float("nan")
        if "sky_surface_p2p_adu" in qdf.columns and n_applied > 0:
            p2p_med = float(
                pd.to_numeric(qdf.loc[applied, "sky_surface_p2p_adu"], errors="coerce").median()
            )
        out: dict[str, Any] = {
            "sky_surface_order": int(order),
            "sky_surface_n_applied": n_applied,
            "sky_surface_n_frames": int(len(qdf)),
        }
        if math.isfinite(p2p_med):
            out["sky_surface_p2p_median_adu"] = p2p_med
        return out
    except Exception as exc:  # noqa: BLE001
        logging.debug("[SKY-SURFACE] meta from qc_metrics skipped: %s", exc)
        return {}


def merge_photometry_pipeline_meta(
    photometry_dir: Path | str,
    updates: dict[str, Any],
    cfg: Any = None,
    *,
    entry_point: str | None = None,
) -> None:
    """Merge keys into ``photometry/pipeline_meta.json`` (MASTERSTAR + Phase 2A)."""
    _meta_path = Path(photometry_dir) / "pipeline_meta.json"
    try:
        _meta_path.parent.mkdir(parents=True, exist_ok=True)
        _existing: dict[str, Any] = {}
        if _meta_path.is_file():
            try:
                _existing = json.loads(_meta_path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                logging.error('[EXC-0158] Existing pipeline_meta.json parse failure replaced with empty dict - prior meta keys si...: %s', exc)
                pass
        _merged = dict(updates)
        if cfg is not None and entry_point:
            _merged["provenance"] = _build_pipeline_provenance_block(cfg, entry_point=entry_point)
        _existing.update(_merged)
        _meta_path.write_text(json.dumps(_existing, indent=2), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("[PHOT] pipeline_meta write failed: %s", exc)
        pass


def _phase2a_resolve_field_center_ra_dec(
    ms_header: Any,
    at_df: pd.DataFrame,
) -> tuple[float, float, str]:
    """Field center (deg ICRS): MASTERSTAR WCS CRVAL, else median of active_targets ra/dec."""
    if ms_header is not None:
        try:
            cr1 = ms_header.get("CRVAL1")
            cr2 = ms_header.get("CRVAL2")
            if cr1 is not None and cr2 is not None:
                ra = float(cr1)
                dec = float(cr2)
                if math.isfinite(ra) and math.isfinite(dec):
                    return ra, dec, "MASTERSTAR_CRVAL"
        except Exception as exc:  # noqa: BLE001
            logging.error('[EXC-0159] MASTERSTAR CRVAL1/2 parse fails - field center falls back to active_targets median (may...: %s', exc)
            pass
    if at_df is not None and not at_df.empty:
        ra_col = "ra_deg" if "ra_deg" in at_df.columns else ("ra" if "ra" in at_df.columns else None)
        de_col = "dec_deg" if "dec_deg" in at_df.columns else ("dec" if "dec" in at_df.columns else None)
        if ra_col and de_col:
            ra_s = pd.to_numeric(at_df[ra_col], errors="coerce")
            de_s = pd.to_numeric(at_df[de_col], errors="coerce")
            ok = ra_s.notna() & de_s.notna()
            if int(ok.sum()) > 0:
                ra_m = float(ra_s[ok].median())
                de_m = float(de_s[ok].median())
                if math.isfinite(ra_m) and math.isfinite(de_m):
                    return ra_m, de_m, "active_targets_median"
    return float("nan"), float("nan"), "unavailable"


def _phase2a_collect_session_jd_values(frame_time_lookup: dict[str, dict[str, float]]) -> list[float]:
    """Per-frame ``jd_mid`` values from Phase 2A ``frame_time_lookup``."""
    out: list[float] = []
    for entry in frame_time_lookup.values():
        try:
            v = float(entry.get("jd", float("nan")))
        except (TypeError, ValueError):
            continue
        if math.isfinite(v):
            out.append(v)
    return out


@dataclass
class _Phase2AState:
    at_df: pd.DataFrame
    comp_df: pd.DataFrame
    _comp_index: dict
    target_bp_rp_by_cid: dict
    csv_files: list
    n_frames: int
    _phase2a_csv_cache: dict
    _phase2a_lookup_cache: dict
    frame_time_lookup: dict
    fwhm_px: float
    apertures_px: dict
    star_xy: dict
    chip_fw: int | None
    chip_fh: int | None
    _ms_header: object
    _ms_data: object
    _flux_matrix: object
    _all_lc_ids_list: list
    field_map_path: Path
    obs_group: str
    _gain_phot: float
    _rn_phot: float
    sat_limit_resolved: float | None
    _aligned_dir_2a: Path
    _cfg: object
    _nt: int
    lunar_context: dict[str, Any] | None = None
    plate_scale_arcsec: float = 1.3
    gaia_db_path: str | None = None
    masterstars_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    #: Observer site resolved once per draft via param_resolver.resolve_site
    #: (draft ID_LOCATION -> header SITELAT -> flagged config). Threaded into
    #: per-target BJD/HJD recompute and pipeline_meta so Phase 2A no longer
    #: silently uses cfg.observer_* (the config-drift trap).
    site_lat: float | None = None
    site_lon: float | None = None
    site_alt: float | None = None
    site_source: str = "unresolved"
    site_ok: bool = False
    group_color_term: _ColorTermGroupFit | None = None
    apply_color_term: bool = False
    k2_bprp: float = float("nan")
    k2_source: str = "none"
    stability_run_flags: dict[str, Any] = field(default_factory=dict)
    variable_target_catalog_ids: frozenset[str] = field(default_factory=frozenset)
    snr_ap_table: dict[str, Any] | None = None
    equipment_id: int | None = None
    #: Run-effective resolved facts (metadata only; PARAM-OWNERSHIP-WAVE-A STEP 4).
    #: gain/read-noise/site sources, saturation, plate scale, frame dims, binning,
    #: filter, exptime -- captured from the resolvers for the honest full-config report.
    #: Never read by the science path; the anchor comparator ignores pipeline_meta.
    resolved_facts: dict[str, Any] = field(default_factory=dict)


def _build_phase2a_dynamic_params(
    state: _Phase2AState,
    output_dir: Path,
    *,
    aperture_fwhm_factor: float,
) -> dict[str, Any]:
    """Runtime scalars for ``pipeline_meta.json`` (Phase 2A start)."""
    _median_sky = _median_sky_from_phase2a_csv_cache(state._phase2a_csv_cache)
    sky_adu_per_px: float | None = (
        float(_median_sky) if math.isfinite(float(_median_sky)) else None
    )

    density_class: str | None = None
    _fd_path = Path(output_dir) / "field_density.json"
    if _fd_path.is_file():
        try:
            _fd = json.loads(_fd_path.read_text(encoding="utf-8"))
            _dc = _fd.get("density_class")
            if _dc is not None and str(_dc).strip():
                density_class = str(_dc).strip()
        except Exception as exc:  # noqa: BLE001
            logging.error('[EXC-0160] field_density.json density_class not loaded - adaptive crowding params use default density: %s', exc)
            pass

    vy_ndao: int | None = None
    if state._ms_header is not None:
        try:
            _vn = state._ms_header.get("VY_NDAO")
            if _vn is not None and str(_vn).strip() != "":
                vy_ndao = int(float(_vn))
        except Exception:  # noqa: BLE001
            vy_ndao = None
    if vy_ndao is not None and vy_ndao <= 0:
        vy_ndao = None

    safe_bbox: list[float] | None = None
    for _plan_path in (
        Path(output_dir) / "photometry_plan.json",
        Path(output_dir).parent / "photometry_plan.json",
    ):
        if not _plan_path.is_file():
            continue
        try:
            _sb = json.loads(_plan_path.read_text(encoding="utf-8")).get("safe_bbox_px")
            if isinstance(_sb, (list, tuple)) and len(_sb) == 4:
                safe_bbox = [float(_sb[0]), float(_sb[1]), float(_sb[2]), float(_sb[3])]
                break
        except Exception as exc:  # noqa: BLE001
            logging.error('[EXC-0161] alignment safe_bbox from plan JSON not loaded - chip margin/edge filters may use full f...: %s', exc)
            pass

    aperture_r_px: float | None = None
    if state.apertures_px:
        _ap_vals = [
            float(v)
            for v in state.apertures_px.values()
            if v is not None and math.isfinite(float(v)) and float(v) > 0
        ]
        if _ap_vals:
            aperture_r_px = float(np.median(_ap_vals))
    if aperture_r_px is None:
        try:
            _fw = float(state.fwhm_px)
            _af = float(aperture_fwhm_factor)
            if math.isfinite(_fw) and _fw > 0 and math.isfinite(_af) and _af > 0:
                aperture_r_px = _af * _fw
        except (TypeError, ValueError):
            aperture_r_px = None
    if aperture_r_px is not None and (not math.isfinite(aperture_r_px) or aperture_r_px <= 0):
        aperture_r_px = None

    _plate = getattr(state, "plate_scale_arcsec", None)
    if _plate is None or not math.isfinite(float(_plate)):
        _plate = None

    from sigma_floor_core import resolve_sigma_sys_mag  # noqa: PLC0415

    _ssm = resolve_sigma_sys_mag(state.equipment_id, state._cfg, rig_label=str(state.obs_group or ""))

    return {
        "fwhm_px": float(state.fwhm_px) if state.fwhm_px is not None and math.isfinite(float(state.fwhm_px)) else None,
        "plate_scale_arcsec_px": float(_plate) if _plate is not None else None,
        "gain": float(state._gain_phot) if state._gain_phot is not None and math.isfinite(float(state._gain_phot)) else None,
        "read_noise": float(state._rn_phot) if state._rn_phot is not None and math.isfinite(float(state._rn_phot)) else None,
        "sky_adu_per_px": sky_adu_per_px,
        "density_class": density_class,
        "n_stars_dao": vy_ndao,
        "safe_bbox": safe_bbox,
        "aperture_r_px": aperture_r_px,
        "sigma_floor": {
            "equipment_id": state.equipment_id,
            "sigma_sys_mag": float(_ssm),
            "c4_correction": True,
            "err_model": "err_photon^2 + sem_ens^2 + sigma_sys^2 (rel flux)",
            "red_noise_diagnostic": "report-only (PZQ 2006); not in per-point bars",
        },
    }


def _phase2a_compute_lunar_context(state: _Phase2AState) -> dict[str, Any] | None:
    """Session-level lunar context (once per ``run_phase2a``)."""
    from lunar_context import get_jd_midpoint, get_lunar_context  # noqa: PLC0415

    _cfg = state._cfg
    # Use the per-draft resolved observer site (param_resolver: draft ID_LOCATION ->
    # header SITELAT -> flagged config) so lunar context uses the SAME location as
    # BJD/HJD/airmass and is independent of config drift between sessions.
    if state.site_ok and state.site_lat is not None and state.site_lon is not None:
        lat = float(state.site_lat)
        lon = float(state.site_lon)
        alt_m = float(state.site_alt) if state.site_alt is not None else 0.0
    else:
        lat = float(_cfg.observer_lat)
        lon = float(_cfg.observer_lon)
        alt_m = float(_cfg.observer_alt_m)
    if lat == 0.0 and lon == 0.0:
        logging.warning("Observer location not set — lunar context skipped")
        return None
    jd_vals = _phase2a_collect_session_jd_values(state.frame_time_lookup)
    if not jd_vals:
        logging.warning("[PHASE 2A] No frame JD values — lunar context skipped")
        return None
    try:
        jd_mid = get_jd_midpoint(jd_vals)
    except ValueError:
        logging.warning("[PHASE 2A] JD midpoint unavailable — lunar context skipped")
        return None
    ra_field, dec_field, src = _phase2a_resolve_field_center_ra_dec(state._ms_header, state.at_df)
    if not (math.isfinite(ra_field) and math.isfinite(dec_field)):
        logging.warning("[PHASE 2A] Field center RA/Dec unavailable — lunar context skipped")
        return None
    lunar = get_lunar_context(
        jd_mid=float(jd_mid),
        ra_field=float(ra_field),
        dec_field=float(dec_field),
        lat=lat,
        lon=lon,
        alt_m=alt_m,
    )
    lunar["field_center_source"] = str(src)
    lunar["jd_mid"] = float(jd_mid)
    logging.info(
        "Lunar context: %s risk — phase %.1f%%, separation %.1f°, altitude %.1f°",
        lunar["lunar_risk"],
        float(lunar["lunar_phase_pct"]),
        float(lunar["lunar_separation_deg"]),
        float(lunar["lunar_altitude_deg"]),
    )
    return lunar


@dataclass(frozen=True)
class BlendMapEntry:
    """One row from ``crowding_targets.csv`` for NEIGHBOR-SUB / adaptive routing."""

    is_blended: bool
    nn_dist_fwhm: float
    nn_catalog_id: str | None = None
    delta_mag_nn: float | None = None
    nn_ra_deg: float | None = None
    nn_dec_deg: float | None = None
    mag: float | None = None


_ADAPTIVE_BLEND_CACHE: dict[str, dict[str, BlendMapEntry]] = {}


def _load_blend_worklist(masterstar_fits_path: Path) -> dict[str, BlendMapEntry]:
    """``catalog_id`` -> blend worklist row from ``crowding_targets.csv`` (cached)."""
    key = str(masterstar_fits_path)
    if key in _ADAPTIVE_BLEND_CACHE:
        return _ADAPTIVE_BLEND_CACHE[key]
    m: dict[str, BlendMapEntry] = {}
    try:
        p = Path(masterstar_fits_path).parent / "crowding_targets.csv"
        if p.is_file():
            df = pd.read_csv(p, low_memory=False)
            for _, r in df.iterrows():
                cid = str(r.get("catalog_id", "")).strip()
                if not cid or cid.lower() == "nan":
                    continue
                isb = _coerce_bool_cell(r.get("is_blended"))
                nn = float(pd.to_numeric(r.get("nn_dist_fwhm"), errors="coerce"))
                nncid = str(r.get("nn_catalog_id", "") or "").strip() or None
                dmag = float(pd.to_numeric(r.get("delta_mag_nn"), errors="coerce"))
                dmag_v = dmag if math.isfinite(dmag) else None
                ra = float(pd.to_numeric(r.get("ra_deg"), errors="coerce"))
                de = float(pd.to_numeric(r.get("dec_deg"), errors="coerce"))
                mag_v = float(pd.to_numeric(r.get("mag"), errors="coerce"))
                entry = BlendMapEntry(
                    is_blended=isb,
                    nn_dist_fwhm=nn,
                    nn_catalog_id=nncid,
                    delta_mag_nn=dmag_v,
                    nn_ra_deg=ra if math.isfinite(ra) else None,
                    nn_dec_deg=de if math.isfinite(de) else None,
                    mag=mag_v if math.isfinite(mag_v) else None,
                )
                m[cid] = entry
                ncid = _normalize_gaia_id(cid)
                if ncid:
                    m[ncid] = entry
    except Exception as e:  # noqa: BLE001
        logging.error('[EXC-0162] ePSF blend worklist JSON load fails - adaptive blend deblend map empty, crowded stars u...: %s', exc)
        logging.warning("[ePSF] blend worklist load failed (%s)", e)
    _ADAPTIVE_BLEND_CACHE[key] = m
    return m


def _load_adaptive_blend_map(masterstar_fits_path: Path) -> dict[str, tuple[bool, float]]:
    """``catalog_id`` -> ``(is_blended, nn_dist_fwhm)`` (legacy tuple API; see ``BlendMapEntry``)."""
    return {
        k: (v.is_blended, v.nn_dist_fwhm) for k, v in _load_blend_worklist(masterstar_fits_path).items()
    }


from mag_constants import MAG_ERR_SCALE

_PSF_ERR_MAG_SCALE = MAG_ERR_SCALE


def _route_lc_per_frame_err(
    target_frames: pd.DataFrame,
    err: np.ndarray,
) -> tuple[np.ndarray, list[str] | None]:
    """Use PSF sandwich err for PSF-routed frames when finite; else keep aperture err."""
    if "lc_flux_method" not in target_frames.columns:
        return err, None
    methods = target_frames["lc_flux_method"].astype(str).to_numpy()
    if not np.any(methods == "psf"):
        return err, None
    out = np.asarray(err, dtype=float).copy()
    err_methods = ["aperture"] * len(out)
    pf = pd.to_numeric(target_frames.get("psf_flux"), errors="coerce").to_numpy(dtype=float)
    pfe = pd.to_numeric(target_frames.get("psf_flux_err"), errors="coerce").to_numpy(dtype=float)
    psf_mask = (methods == "psf") & np.isfinite(pf) & (pf > 0) & np.isfinite(pfe) & (pfe > 0)
    if np.any(psf_mask):
        out[psf_mask] = _PSF_ERR_MAG_SCALE * pfe[psf_mask] / pf[psf_mask]
        for i in np.where(psf_mask)[0]:
            err_methods[int(i)] = "psf"
    return out, err_methods


def _get_lc(cid: str, all_frames: pd.DataFrame) -> np.ndarray:
    sub = all_frames[all_frames["catalog_id"] == cid]
    if sub.empty:
        return np.array([], dtype=float)
    if "source_file" in sub.columns:
        sub = sub.sort_values(["source_file"], kind="mergesort")
    return sub["mag_inst"].to_numpy(dtype=float)


def _get_comp_bjd_series(cid: str, all_frames: pd.DataFrame) -> np.ndarray:
    """BJD (or JD) time series for a comp, same row order as ``_get_lc``."""
    sub = all_frames[all_frames["catalog_id"] == cid]
    if sub.empty:
        return np.array([], dtype=float)
    if "source_file" in sub.columns:
        sub = sub.sort_values(["source_file"], kind="mergesort")
    if "bjd" in sub.columns:
        return sub["bjd"].to_numpy(dtype=float)
    if "jd" in sub.columns:
        return sub["jd"].to_numpy(dtype=float)
    return np.array([], dtype=float)


def _resolve_star_flux_method(cid: str, all_frames: pd.DataFrame) -> str:
    """PSF-only inst mag: NaN when PSF flux unavailable (no aperture fallback)."""
    sub = all_frames[all_frames["catalog_id"] == cid]
    if sub.empty:
        return np.array([], dtype=float)
    if "psf_flux" not in sub.columns or "psf_fit_ok" not in sub.columns:
        return np.full(len(sub), float("nan"), dtype=float)
    psf_flux = pd.to_numeric(sub["psf_flux"], errors="coerce").to_numpy(dtype=float)
    psf_ok = sub["psf_fit_ok"].map(_coerce_bool_cell).to_numpy(dtype=bool)
    psf_mag = np.where(
        psf_ok & np.isfinite(psf_flux) & (psf_flux > 0),
        -2.5 * np.log10(psf_flux),
        np.nan,
    )
    return np.asarray(psf_mag, dtype=float)


def _resolve_star_flux_method(cid: str, all_frames: pd.DataFrame) -> str:
    """One routing decision per star (majority of per-frame lc_flux_method)."""
    sub = all_frames[all_frames["catalog_id"] == cid]
    if sub.empty or "lc_flux_method" not in sub.columns:
        return "aperture"
    counts = sub["lc_flux_method"].astype(str).str.strip().str.lower().value_counts()
    if counts.empty:
        return "aperture"
    if int(counts.get("psf", 0)) > int(counts.get("aperture", 0)):
        return "psf"
    return "aperture"


def _get_lc_star_method(cid: str, all_frames: pd.DataFrame, star_method: str) -> np.ndarray:
    """Inst mag for one star using a fixed method for all frames (NaN if PSF missing)."""
    sub = all_frames[all_frames["catalog_id"] == cid]
    if sub.empty:
        return np.array([], dtype=float)
    if str(star_method).strip().lower() != "psf":
        return _get_lc(cid, all_frames)
    return _get_lc_psf_strict(cid, all_frames)


def _get_lc_adaptive_per_star(cid: str, all_frames: pd.DataFrame) -> np.ndarray:
    """Adaptive LC: one method per star applied consistently across all frames."""
    sm = _resolve_star_flux_method(cid, all_frames)
    return _get_lc_star_method(cid, all_frames, sm)


def compute_lc_flux_method(
    all_frames: pd.DataFrame,
    blend_map: dict[str, tuple[bool, float]] | None = None,
    *,
    resolve_fwhm: float = 2.0,
    snr_lo: float = 15.0,
) -> pd.Series:
    """Per-star/per-frame adaptive flux-source choice ∈ {``aperture``, ``psf``} (b.4).

    CONSERVATIVE: default to aperture, switch to PSF only with positive evidence AND good
    PSF quality. ``blend_map`` is accepted for API compatibility (crowding_targets.csv)
    but is not used for routing — resolvable-blend → PSF (former rule 2) was removed
    because ``is_blended`` (nn ≤ 1.5 FWHM) and ``resolve_fwhm`` (≥ 2.0) are mutually
    exclusive and grouped deblending showed no precision gain at 0.39"/px on draft 364.

    Rules (first match wins):
      1. psf_quality == bad / not fit_ok / no finite psf_flux  → aperture (the b.5 fallback)
      2. faint (aperture SNR ≤ snr_lo) AND psf_quality == good   → psf
      3. else                                                    → aperture
    """
    n = len(all_frames)
    idx = all_frames.index
    if n == 0:
        return pd.Series([], dtype=object)
    _ = blend_map  # unused; kept so callers need not change when psf_adaptive_enabled

    psf_flux = pd.to_numeric(all_frames.get("psf_flux", pd.Series(np.nan, index=idx)), errors="coerce")
    psf_q = all_frames.get("psf_quality", pd.Series("", index=idx)).astype(str).str.strip().str.lower()
    _ok_raw = all_frames.get("psf_fit_ok", pd.Series(False, index=idx))
    psf_ok = _ok_raw.map(_coerce_bool_cell).to_numpy(dtype=bool)
    # Aperture SNR from the (mag) error: err_mag ≈ 1.0857 / SNR.
    err = pd.to_numeric(all_frames.get("err", pd.Series(np.nan, index=idx)), errors="coerce").to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr_aper = np.where(np.isfinite(err) & (err > 0), 1.0857362 / err, np.inf)

    pf = psf_flux.to_numpy(dtype=float)
    _ac_raw = all_frames.get("psf_ac_applied", pd.Series(False, index=idx))
    psf_ac_ok = _ac_raw.map(_coerce_bool_cell).to_numpy(dtype=bool)
    psf_usable = psf_ok & np.isfinite(pf) & (pf > 0) & (psf_q.to_numpy() != "bad") & psf_ac_ok

    rule_faint_psf = psf_usable & (snr_aper <= float(snr_lo)) & (psf_q.to_numpy() == "good")

    method = np.where(rule_faint_psf, "psf", "aperture").astype(object)
    return pd.Series(method, index=idx)


def _get_lc_adaptive(cid: str, all_frames: pd.DataFrame) -> np.ndarray:
    """LC mag_inst series using the per-frame ``lc_flux_method`` column (b.4 adaptive).

    Requires ``compute_lc_flux_method`` to have populated ``lc_flux_method``. Falls back to
    aperture (``mag_inst``) for any frame not selected as ``psf`` or with non-finite psf_flux.
    """
    sub = all_frames[all_frames["catalog_id"] == cid]
    if sub.empty:
        return np.array([], dtype=float)
    mag_inst = sub["mag_inst"].to_numpy(dtype=float)
    if "lc_flux_method" not in sub.columns or "psf_flux" not in sub.columns:
        return mag_inst
    psf_flux = pd.to_numeric(sub["psf_flux"], errors="coerce").to_numpy(dtype=float)
    if "psf_ac_applied" in sub.columns:
        ac_ok = sub["psf_ac_applied"].map(_coerce_bool_cell).to_numpy(dtype=bool)
    else:
        ac_ok = np.zeros(len(sub), dtype=bool)
    use_psf = (
        (sub["lc_flux_method"].astype(str).to_numpy() == "psf")
        & ac_ok
        & np.isfinite(psf_flux)
        & (psf_flux > 0)
    )
    psf_mag = np.where(use_psf, -2.5 * np.log10(np.where(psf_flux > 0, psf_flux, np.nan)), np.nan)
    return np.where(np.isfinite(psf_mag), psf_mag, mag_inst)


def load_epsf_metrics_for_draft(
    per_frame_csv_dir: Path,
    active_targets_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate PSF fit metrics from ``proc_*.csv`` (vectorized groupby).

    Returns per-``catalog_id`` stats: frame counts, PSF OK %, χ², mean fluxes, PSF/DAO ratio.
    """
    proc_dir = Path(per_frame_csv_dir)
    proc_files = sorted(proc_dir.glob("proc_*.csv"))
    if not proc_files:
        return pd.DataFrame()

    usecols = ["catalog_id", "psf_flux", "psf_fit_ok", "psf_chi2", "dao_flux"]
    _cid_dtype = {"catalog_id": str}
    chunks: list[pd.DataFrame] = []
    for csv_path in proc_files:
        try:
            df = pd.read_csv(
                csv_path, usecols=usecols, low_memory=False, dtype=_cid_dtype
            )
            chunks.append(df)
        except Exception as exc:  # noqa: BLE001
            logging.error("[EXC-0163] One frame's psf_fit CSV unreadable - ePSF metrics aggregate omits that frame's stars: %s", exc)
            try:
                df = pd.read_csv(csv_path, low_memory=False)
                if "psf_fit_ok" not in df.columns:
                    continue
                keep = [c for c in usecols if c in df.columns]
                chunks.append(df[keep])
            except Exception:  # noqa: BLE001
                continue

    if not chunks:
        return pd.DataFrame()

    combined = pd.concat(chunks, ignore_index=True)
    if "catalog_id" not in combined.columns or "psf_fit_ok" not in combined.columns:
        return pd.DataFrame()

    try:
        from gaia_catalog_id import normalize_gaia_source_id_series  # noqa: PLC0415

        combined["catalog_id"] = normalize_gaia_source_id_series(combined["catalog_id"])
    except Exception:  # noqa: BLE001
        combined["catalog_id"] = combined["catalog_id"].astype(str).str.strip()
    combined = combined[
        combined["catalog_id"].astype(bool)
        & ~combined["catalog_id"].str.lower().isin(("nan", "none", ""))
    ]
    if combined.empty:
        return pd.DataFrame()

    combined["psf_fit_ok"] = combined["psf_fit_ok"].fillna(False).astype(bool)
    combined["psf_chi2"] = pd.to_numeric(combined["psf_chi2"], errors="coerce")
    combined["psf_flux"] = pd.to_numeric(combined["psf_flux"], errors="coerce")
    combined["dao_flux"] = pd.to_numeric(combined["dao_flux"], errors="coerce")

    grp = combined.groupby("catalog_id", sort=False)
    result = pd.DataFrame(
        {
            "n_frames": grp["psf_fit_ok"].count(),
            "n_psf_ok": grp["psf_fit_ok"].sum(),
            "mean_chi2": grp["psf_chi2"].mean(),
            "median_chi2": grp["psf_chi2"].median(),
            "min_chi2": grp["psf_chi2"].min(),
            "mean_psf_flux": grp["psf_flux"].mean(),
            "mean_dao_flux": grp["dao_flux"].mean(),
        }
    ).reset_index()

    result["pct_psf_ok"] = (100.0 * result["n_psf_ok"] / result["n_frames"]).round(1)
    for col in ("mean_chi2", "median_chi2", "min_chi2"):
        result[col] = pd.to_numeric(result[col], errors="coerce").round(2)
    result["psf_dao_ratio"] = (
        result["mean_psf_flux"] / result["mean_dao_flux"].replace(0, np.nan)
    ).round(4)

    if (
        not active_targets_df.empty
        and "catalog_id" in active_targets_df.columns
        and "vsx_name" in active_targets_df.columns
    ):
        meta = active_targets_df[["catalog_id", "vsx_name"]].copy()
        try:
            meta["catalog_id"] = normalize_gaia_source_id_series(meta["catalog_id"])
        except Exception:  # noqa: BLE001
            meta["catalog_id"] = meta["catalog_id"].astype(str).str.strip()
        meta = meta.drop_duplicates(subset=["catalog_id"], keep="first")
        result = result.merge(meta, on="catalog_id", how="left")

    # Sort: known targets (vsx_name not null) first, then by pct_psf_ok desc
    if "vsx_name" in result.columns:
        result["_has_name"] = result["vsx_name"].notna() & (
            result["vsx_name"].astype(str).str.strip() != ""
        )
        result = (
            result.sort_values(["_has_name", "pct_psf_ok"], ascending=[False, False])
            .drop(columns="_has_name")
            .reset_index(drop=True)
        )
    else:
        result = result.sort_values("pct_psf_ok", ascending=False).reset_index(drop=True)
    return result


def _apply_role_aware_aperture_scaling(
    apertures_px: dict[str, float],
    at_df: pd.DataFrame,
    cfg: AppConfig,
) -> None:
    """TODO-44: Scale SNR-optimal radii by star role (variable vs comp/check)."""
    _var_factor = float(cfg.aperture_variable_factor)
    _comp_factor = float(cfg.aperture_comp_factor)
    if _var_factor == 1.0 and _comp_factor == 1.0:
        return
    _target_cids = set(
        _normalize_gaia_id(str(r.get("catalog_id", "")))
        for _, r in at_df.iterrows()
    )
    _n_var_scaled = 0
    _n_comp_scaled = 0
    for _cid in list(apertures_px.keys()):
        if _cid in _target_cids:
            if _var_factor != 1.0:
                apertures_px[_cid] = round(apertures_px[_cid] * _var_factor, 3)
                _n_var_scaled += 1
        else:
            if _comp_factor != 1.0:
                apertures_px[_cid] = round(apertures_px[_cid] * _comp_factor, 3)
                _n_comp_scaled += 1
    LOGGER.info(
        "[TODO-44] Role-aware aperture: var_factor=%.2f (%d targets), "
        "comp_factor=%.2f (%d comps/checks)",
        _var_factor,
        _n_var_scaled,
        _comp_factor,
        _n_comp_scaled,
    )


def _preserve_nondetection_flags_helper(
    out_flags_local: list[str], target_frames: pd.DataFrame
) -> None:
    if "flag" not in target_frames.columns:
        return
    _rf_nd = target_frames["flag"].astype(str).str.strip().str.lower().reset_index(drop=True)
    for i in range(min(len(out_flags_local), len(_rf_nd))):
        if str(_rf_nd.iloc[i]) == "nondetection":
            out_flags_local[i] = "nondetection"


def _frame_quality_gate_select(
    csv_files: list[Path],
    cfg: AppConfig | None,
    proc_frame_store: ProcFrameStore | None,
) -> tuple[list[Path], list[str]]:
    """Round-2 B.2: whole-frame transparency / PSF-collapse gate.

    Default OFF -> returns ``(list(csv_files), [])`` unchanged (byte-identical baseline).

    When enabled, rejects frames whose PSF concentration -- the per-frame median of
    ``flux_large / flux`` over bright, unsaturated sources -- is a robust outlier
    ``z = (ratio - median) / (1.4826*MAD) > cfg.frame_quality_ratio_k`` (the decisive primary
    signal) guarded by ``fwhm_estimate_px > cfg.frame_quality_fwhm_factor * median-FWHM`` so a
    spurious ratio outlier on a better-than-median (sharp) frame is spared.
    A collapsed/heavily-blurred frame pushes flux out of the fixed science aperture so the
    large/small aperture ratio spikes; a clear-but-faint frame keeps a normal concentration
    (flux falls equally in both apertures) and is spared. Frames with no usable photometry are
    also dropped. Safety floor: if the gate would keep < ``cfg.frame_quality_min_keep_frames``
    frames it is skipped (returns the input unchanged) to avoid nuking a marginal night.

    Returns ``(kept_csv_files, rejected_basenames)``.
    """
    if cfg is None or not getattr(cfg, "frame_quality_gate_enabled", False):
        return list(csv_files), []
    cols = ["flux", "flux_large", "fwhm_estimate_px", "likely_saturated", "mag"]
    ratios: list[float] = []
    fwhms: list[float] = []
    for p in csv_files:
        df = proc_frame_store.get_frame(p, cols=cols) if proc_frame_store is not None else None
        if df is None:
            try:
                _want = set(cols)
                df = pd.read_csv(p, usecols=lambda c: c in _want, low_memory=False)
            except Exception:  # noqa: BLE001
                df = None
        if df is None or "flux" not in df.columns or "flux_large" not in df.columns:
            ratios.append(np.nan)
            fwhms.append(np.nan)
            continue
        fs = pd.to_numeric(df["flux"], errors="coerce")
        fl = pd.to_numeric(df["flux_large"], errors="coerce")
        sat = (
            pd.to_numeric(df["likely_saturated"], errors="coerce").fillna(0) > 0
            if "likely_saturated" in df.columns
            else pd.Series(False, index=fs.index)
        )
        m = (fs > 0) & (fl > 0) & ~sat
        if "mag" in df.columns:
            mg = pd.to_numeric(df["mag"], errors="coerce")
            mm = m & np.isfinite(mg) & (mg >= 10.0) & (mg <= 14.5)
            if int(mm.sum()) >= 5:
                m = mm
        ratios.append(
            float(np.nanmedian((fl[m] / fs[m]).to_numpy())) if int(m.sum()) >= 3 else np.nan
        )
        fwhms.append(
            float(np.nanmedian(pd.to_numeric(df["fwhm_estimate_px"], errors="coerce")))
            if "fwhm_estimate_px" in df.columns
            else np.nan
        )
    rr = np.asarray(ratios, dtype=float)
    fw = np.asarray(fwhms, dtype=float)
    good = np.isfinite(rr)
    if int(good.sum()) < max(int(cfg.frame_quality_min_keep_frames), 5):
        return list(csv_files), []
    med = float(np.nanmedian(rr[good]))
    mad = float(np.nanmedian(np.abs(rr[good] - med)))
    scale = 1.4826 * mad if mad > 0 else (float(np.nanstd(rr[good])) or 1.0)
    fwhm_med = float(np.nanmedian(fw[np.isfinite(fw)])) if np.isfinite(fw).any() else np.inf
    z = (rr - med) / scale
    reject = (z > float(cfg.frame_quality_ratio_k)) & (
        (~np.isfinite(fw)) | (fw > float(cfg.frame_quality_fwhm_factor) * fwhm_med)
    )
    reject = reject | (~good)
    if int((~reject).sum()) < int(cfg.frame_quality_min_keep_frames):
        LOGGER.warning(
            "[FRAME-QC] gate would keep only %d < floor %d frames -> SKIPPING gate",
            int((~reject).sum()),
            int(cfg.frame_quality_min_keep_frames),
        )
        return list(csv_files), []
    kept = [p for p, rj in zip(csv_files, reject) if not rj]
    rejected = [Path(p).name for p, rj in zip(csv_files, reject) if rj]
    return kept, rejected


def _proc_stem(name: str) -> str:
    """``proc_<stem>.csv`` -> ``<stem>`` (matches alignment_report.csv ``file`` minus ``.fits``)."""
    if name.startswith("proc_") and name.endswith(".csv"):
        return name[5:-4]
    return name


def _compute_frame_align_residuals(
    csv_files: list[Path],
    proc_frame_store: ProcFrameStore | None,
) -> tuple[dict[Path, float], float]:
    """Fix B (QC, always-on): per-frame alignment residual (px) + science aperture radius (px).

    ``residual[frame]`` = the median, over bright matched sources (``10 <= mag <= 13`` and
    ``flux > 0``), of the Euclidean deviation of each source's ``(x, y)`` from that source's
    robust across-night median position. This is method-agnostic and reproduces the run-414
    diagnostic separation (astroalign ~0.36 px vs phase_correlation ~2.13 px). The reference
    (across-night median) is dominated by the well-aligned majority, so a translation-mis-aligned
    frame stands out by ~its full shift. Also returns the field-median science aperture radius
    (``aperture_r_px``) as the rig-agnostic scale for the gate threshold.

    Pure QC: does NOT change ``csv_files`` selection (and so leaves photometry byte-identical);
    its only side effect (recording the column) is in ``_record_align_residuals_to_report``.
    """
    cols = ["catalog_id", "x", "y", "mag", "flux", "aperture_r_px"]
    per_frame: list[tuple[Path, pd.DataFrame]] = []
    frames: list[pd.DataFrame] = []
    ap_vals: list[float] = []
    for p in csv_files:
        df = proc_frame_store.get_frame(p, cols=cols) if proc_frame_store is not None else None
        if df is None:
            try:
                _want = set(cols)
                df = pd.read_csv(p, usecols=lambda c: c in _want, low_memory=False)
            except Exception:  # noqa: BLE001
                df = None
        if df is None or not {"catalog_id", "x", "y"}.issubset(df.columns):
            per_frame.append((p, pd.DataFrame(columns=["catalog_id", "x", "y"])))
            continue
        cid = df["catalog_id"].astype(str)
        x = pd.to_numeric(df["x"], errors="coerce")
        y = pd.to_numeric(df["y"], errors="coerce")
        mg = pd.to_numeric(df["mag"], errors="coerce") if "mag" in df.columns else pd.Series(np.nan, index=df.index)
        fl = pd.to_numeric(df["flux"], errors="coerce") if "flux" in df.columns else pd.Series(1.0, index=df.index)
        bright = np.isfinite(x) & np.isfinite(y) & (fl > 0)
        if mg.notna().any():
            _b = bright & (mg >= 10.0) & (mg <= 13.0)
            if int(_b.sum()) >= 5:
                bright = _b
        sub = pd.DataFrame({"catalog_id": cid[bright], "x": x[bright], "y": y[bright]})
        per_frame.append((p, sub))
        if not sub.empty:
            frames.append(sub.assign(_fi=len(per_frame) - 1))
        if "aperture_r_px" in df.columns:
            apr = pd.to_numeric(df["aperture_r_px"], errors="coerce")
            apr = apr[np.isfinite(apr) & (apr > 0)]
            if len(apr):
                ap_vals.append(float(np.nanmedian(apr.to_numpy())))
    residuals: dict[Path, float] = {p: float("nan") for p, _ in per_frame}
    if frames:
        allpos = pd.concat(frames, ignore_index=True)
        ref = allpos.groupby("catalog_id")[["x", "y"]].median().rename(columns={"x": "mx", "y": "my"})
        for fi, (p, sub) in enumerate(per_frame):
            if sub.empty:
                continue
            j = sub.join(ref, on="catalog_id")
            dr = np.hypot(j["x"] - j["mx"], j["y"] - j["my"]).to_numpy()
            dr = dr[np.isfinite(dr)]
            if len(dr):
                residuals[p] = float(np.median(dr))
    aperture_r_px = float(np.nanmedian(ap_vals)) if ap_vals else float("nan")
    return residuals, aperture_r_px


def _record_align_residuals_to_report(report_path: Path, residuals: dict[Path, float]) -> None:
    """Fix B (QC, always-on): add/refresh ``align_residual_px`` in ``alignment_report.csv``.

    Additive metadata only — does not affect photometry (baseline stays byte-identical). Matches by
    frame stem (``alignment_report.file`` minus ``.fits`` == proc basename minus ``proc_``/``.csv``).
    Best-effort: any failure is logged and ignored so QC never breaks a run.
    """
    try:
        if not Path(report_path).is_file():
            return
        stem_resid = {_proc_stem(Path(p).name): v for p, v in residuals.items()}
        rep = pd.read_csv(report_path)
        if "file" not in rep.columns:
            return
        stems = rep["file"].astype(str).str.replace(".fits", "", regex=False)
        rep["align_residual_px"] = [stem_resid.get(s, float("nan")) for s in stems]
        rep.to_csv(report_path, index=False)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[ALIGN-QC] failed to record align_residual_px: %s", exc)


def _frame_align_residual_gate_select(
    csv_files: list[Path],
    cfg: AppConfig | None,
    residuals: dict[Path, float],
    aperture_r_px: float,
) -> tuple[list[Path], list[str], float]:
    """Fix B: reject-on-alignment-residual frame gate (default OFF -> input unchanged).

    Rejects frames whose recorded alignment residual exceeds
    ``cfg.frame_align_residual_max_frac * science-aperture-radius-px`` (rig-agnostic: a fraction of
    the aperture radius, never a fixed pixel value). Frames with no measurable residual (NaN) are
    KEPT (a missing QC value must not silently drop data). Safety floor: if the gate would keep
    fewer than ``cfg.frame_align_residual_min_keep_frames`` frames it is skipped (no-op).

    Cause-correct counterpart to the B.2 aperture-integrity gate; self-deactivating once alignment
    (Fix C) succeeds. Returns ``(kept, rejected_basenames, threshold_px)``.
    """
    if cfg is None or not getattr(cfg, "frame_align_residual_gate_enabled", False):
        return list(csv_files), [], float("nan")
    if not (math.isfinite(aperture_r_px) and aperture_r_px > 0):
        LOGGER.warning("[ALIGN-QC] residual gate: no valid aperture radius -> SKIPPING gate")
        return list(csv_files), [], float("nan")
    thr = float(cfg.frame_align_residual_max_frac) * float(aperture_r_px)
    reject = []
    for p in csv_files:
        r = residuals.get(p, float("nan"))
        reject.append(bool(math.isfinite(r) and r > thr))
    if int(len(csv_files) - sum(reject)) < int(cfg.frame_align_residual_min_keep_frames):
        LOGGER.warning(
            "[ALIGN-QC] residual gate would keep only %d < floor %d frames -> SKIPPING gate",
            int(len(csv_files) - sum(reject)),
            int(cfg.frame_align_residual_min_keep_frames),
        )
        return list(csv_files), [], thr
    kept = [p for p, rj in zip(csv_files, reject) if not rj]
    rejected = [Path(p).name for p, rj in zip(csv_files, reject) if rj]
    return kept, rejected, thr


def _phase2a_prepare_shared_state(
    output_dir: Path,
    lc_dir: Path,
    masterstar_fits_path: Path,
    comparison_stars_csv: Path,
    per_frame_csv_dir: Path,
    progress_cb: Any,
    *,
    active_targets_csv: Path,
    detrended_aligned_dir: Path,
    fwhm_px: float,
    annulus_inner_fwhm: float = 4.0,
    annulus_outer_fwhm: float = 6.0,
    aperture_fwhm_factor: float | None = None,
    sat_limit_adu: float | None = None,
    force_aperture_px: float | None = None,
    cfg: AppConfig | None = None,
    db: Any | None = None,
    draft_id: int | None = None,
    proc_frame_store: ProcFrameStore | None = None,
) -> _Phase2AState:
    def _p2(msg: str) -> None:
        if progress_cb is None:
            return
        try:
            progress_cb(str(msg))
        except UnicodeEncodeError:
            progress_cb(str(msg).encode("ascii", "backslashreplace").decode("ascii"))

    _aligned_dir_2a = Path(detrended_aligned_dir)
    output_dir = Path(output_dir)
    lc_dir = output_dir / "lightcurves"
    lc_dir.mkdir(parents=True, exist_ok=True)

    _cfg = cfg or AppConfig()
    _save_png = bool(_cfg.save_lightcurve_png)
    # Obs_group / filter name for decisions (e.g. color term). Prefer FITS header, fallback to directory name.
    obs_group = str(Path(per_frame_csv_dir).name)
    if aperture_fwhm_factor is not None:
        try:
            _apt_fw = float(aperture_fwhm_factor)
            if not math.isfinite(_apt_fw) or _apt_fw <= 0:
                _apt_fw = float(_cfg.aperture_fwhm_factor)
            else:
                _apt_fw = max(0.5, min(6.0, _apt_fw))
        except (TypeError, ValueError):
            _apt_fw = float(_cfg.aperture_fwhm_factor)
    else:
        _apt_fw = float(_cfg.aperture_fwhm_factor)

    # Načítaj vstupy (Gaia ID ako string — float64 stráca cifry)
    if not Path(active_targets_csv).is_file():
        raise FileNotFoundError(f"active_targets_csv not found: {active_targets_csv}")
    if not Path(comparison_stars_csv).is_file():
        raise FileNotFoundError(f"comparison_stars_csv not found: {comparison_stars_csv}")
    at_df = pd.read_csv(active_targets_csv, low_memory=False, dtype=_GAIA_ID_DTYPE)
    comp_df = pd.read_csv(
        comparison_stars_csv,
        low_memory=False,
        dtype={**_GAIA_ID_DTYPE, "target_catalog_id": str},
    )

    # Normalizuj catalog_id
    for df in (at_df, comp_df):
        for col in ("catalog_id", "name"):
            if col in df.columns:
                df[col] = df[col].apply(_normalize_gaia_id)

    at_df["skip_photometry"] = _phase2a_coerce_skip_photometry(at_df)
    _require_comparison_stars_per_target_schema(comp_df, Path(comparison_stars_csv))

    # _phase2a_load_star_list (inline): chip dims, comp index, BP-RP map, CSV cache — through target loop below.

    # Open MASTERSTAR.fits once — reuse header + data throughout Phase 2A.
    _ms_header: Any = None
    _ms_data: np.ndarray | None = None
    _ms_path = Path(masterstar_fits_path)
    if _ms_path.is_file():
        try:
            with astrofits.open(_ms_path, memmap=False) as _hdul_ms:
                _ms_header = _hdul_ms[0].header.copy()
                if _hdul_ms[0].data is not None:
                    _ms_data = np.asarray(_hdul_ms[0].data, dtype=np.float64)
            LOGGER.debug("[PHASE 2A] MASTERSTAR.fits opened once (header + data cached)")
            logging.info(
                "[PERF-2] MASTERSTAR.fits loaded once: data shape=%s, header keys=%d",
                _ms_data.shape if _ms_data is not None else None,
                len(_ms_header) if _ms_header is not None else 0,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("[PHASE 2A] MASTERSTAR.fits open failed: %s", exc)

    # Chip dims for edge checks: prefer MASTERSTAR.fits NAXIS (full detector); CSV max(x,y) underestimates.
    chip_fw: int | None = None
    chip_fh: int | None = None
    try:
        if _ms_header is not None:
            _naxis1 = int(_ms_header.get("NAXIS1", 0) or 0)
            _naxis2 = int(_ms_header.get("NAXIS2", 0) or 0)
        else:
            _naxis1 = _naxis2 = 0
        if _naxis1 > 0 and _naxis2 > 0:
            chip_fw = _naxis1
            chip_fh = _naxis2
            logging.info("[PHOT] chip dims from MASTERSTAR: %d x %d px", int(chip_fw), int(chip_fh))
    except Exception:  # noqa: BLE001
        chip_fw, chip_fh = None, None
        logging.warning("[PHOT] chip dims: MASTERSTAR read failed, fallback to CSV max(x/y)")

    # Fallback: estimate from star positions in CSV (any axis still unknown after MASTERSTAR).
    if chip_fw is None or chip_fh is None:
        try:
            xm = float(
                pd.to_numeric(
                    pd.concat([at_df.get("x", pd.Series(dtype=float)), comp_df.get("x", pd.Series(dtype=float))]),
                    errors="coerce",
                ).max()
            )
            ym = float(
                pd.to_numeric(
                    pd.concat([at_df.get("y", pd.Series(dtype=float)), comp_df.get("y", pd.Series(dtype=float))]),
                    errors="coerce",
                ).max()
            )
            if chip_fw is None and math.isfinite(xm) and xm > 0:
                chip_fw = int(math.ceil(xm)) + 2
            if chip_fh is None and math.isfinite(ym) and ym > 0:
                chip_fh = int(math.ceil(ym)) + 2
        except Exception as exc:  # noqa: BLE001
            # EXC-0165: T4 -- Non-numeric bp_rp on one masterstars row skipped when building target_bp_rp_by_cid (EXCEPT-BULK-2 2026-07-08)
            logging.error('[EXC-0164] Chip height/width inference from target/comp xy max fails - chip margin filter may use ...: %s', exc)
            pass

    if "x" not in comp_df.columns or "y" not in comp_df.columns:
        raise ValueError("comparison_stars_per_target.csv musí obsahovať stĺpce x, y pre Fázu 2A")

    # Pre-index comp_df podľa target catalog_id — raz pre celý cyklus (O(1) lookup per target).
    _id_col_comp = "target_catalog_id" if "target_catalog_id" in comp_df.columns else "catalog_id"
    _comp_index: dict[str, pd.DataFrame] = {}
    if comp_df is not None and not comp_df.empty and _id_col_comp in comp_df.columns:
        for _tid, _grp in comp_df.groupby(comp_df[_id_col_comp].apply(_normalize_gaia_id), sort=False, dropna=False):
            _tid_s = str(_tid).strip()
            if _tid_s:
                _comp_index[_tid_s] = _grp.reset_index(drop=True)
    else:
        _comp_index = {}

    # masterstars_full_match.csv — BP-RP map + comp pool pre catalog_only Phase 2A.
    target_bp_rp_by_cid: dict[str, float] = {}
    masterstars_df = pd.DataFrame()
    try:
        ms_full = Path(masterstar_fits_path).resolve().parent / "masterstars_full_match.csv"
        if ms_full.is_file():
            masterstars_df = pd.read_csv(ms_full, low_memory=False, dtype=_GAIA_ID_DTYPE)
            masterstars_df = masterstars_df.copy()
            for _id_col in ("catalog_id", "name"):
                if _id_col in masterstars_df.columns:
                    masterstars_df[_id_col] = masterstars_df[_id_col].apply(_normalize_gaia_id)
            if "bp_rp" in masterstars_df.columns:
                masterstars_df["bp_rp"] = pd.to_numeric(masterstars_df["bp_rp"], errors="coerce")
                for _, r in masterstars_df.iterrows():
                    cid0 = str(r.get("catalog_id") or "").strip()
                    if not cid0:
                        continue
                    v0 = r.get("bp_rp")
                    try:
                        vv = float(v0)
                    except Exception:  # noqa: BLE001
                        continue
                    if math.isfinite(vv):
                        target_bp_rp_by_cid[cid0] = float(vv)
    except Exception:  # noqa: BLE001
        target_bp_rp_by_cid = {}
        masterstars_df = pd.DataFrame()

    # Gated: generate crowding_targets.csv for full LC star set (adaptive blend map input).
    if bool(getattr(_cfg, "psf_adaptive_enabled", False)) and draft_id is not None and db is not None:
        try:
            from crowding_index import ensure_crowding_targets_for_lc
            from database import get_gaia_db_max_g_mag

            _setup_name = str(Path(per_frame_csv_dir).name)
            _draft_dir = Path(masterstar_fits_path).resolve().parent.parent.parent
            _gaia_max = float(get_gaia_db_max_g_mag(str(_cfg.gaia_db_path)))
            _crowd_csv = ensure_crowding_targets_for_lc(
                _draft_dir,
                _setup_name,
                db,
                int(draft_id),
                gaia_db_max_g=_gaia_max,
            )
            if _crowd_csv is not None:
                _ADAPTIVE_BLEND_CACHE.pop(str(Path(masterstar_fits_path)), None)
                _p2(f"Fáza 2A: crowding_targets.csv ({_crowd_csv.name}) — adaptive blend map")
        except Exception as _cr_exc:  # noqa: BLE001
            LOGGER.warning(
                "[ePSF] crowding_targets generation failed (adaptive rule 2 disabled): %s",
                _cr_exc,
            )

    # Nájdi per-frame CSV (FITS sa nepoužíva)
    if proc_frame_store is not None and len(proc_frame_store) > 0:
        csv_files = sorted(Path(k) for k in proc_frame_store.keys())
    else:
        csv_files = sorted(Path(per_frame_csv_dir).glob("proc_*.csv"))
    _ms_epoch_drop = [p for p in csv_files if is_masterstar_proc_name(p)]
    if _ms_epoch_drop:
        logging.warning(
            "[MASTERSTAR-EPOCH] phase2a filtered %d masterstar proc from epoch set "
            "(canonical list_proc_csvs / ProcFrameStore filter was bypassed): %s",
            len(_ms_epoch_drop),
            ", ".join(p.name for p in _ms_epoch_drop[:3]),
        )
        csv_files = [p for p in csv_files if not is_masterstar_proc_name(p)]
    # Round-2 B.2: optional whole-frame transparency/PSF-collapse gate (default OFF -> no-op).
    if getattr(_cfg, "frame_quality_gate_enabled", False):
        _kept_csv, _rejected_csv = _frame_quality_gate_select(csv_files, _cfg, proc_frame_store)
        if _rejected_csv:
            csv_files = _kept_csv
            logging.info(
                "[FRAME-QC] transparency/PSF-collapse gate: rejected %d/%d frames (%s%s)",
                len(_rejected_csv),
                len(_rejected_csv) + len(_kept_csv),
                ", ".join(_rejected_csv[:5]),
                " ..." if len(_rejected_csv) > 5 else "",
            )
            _p2(
                f"Frame-QC gate: {len(_rejected_csv)} collapsed frames rejected, "
                f"{len(_kept_csv)} kept"
            )
    # Fix B: per-frame alignment residual (px). Always-on QC: compute on the full frame set and
    # record into alignment_report.csv (additive metadata -> photometry stays byte-identical).
    # Then, only if the gate is enabled, drop frames whose residual exceeds the rig-agnostic
    # threshold (fraction of the science aperture radius). Cause-correct counterpart to B.2.
    try:
        _align_resid, _align_apr = _compute_frame_align_residuals(csv_files, proc_frame_store)
        _align_report_path = Path(masterstar_fits_path).resolve().parent / "alignment_report.csv"
        _record_align_residuals_to_report(_align_report_path, _align_resid)
    except Exception as _ar_exc:  # noqa: BLE001
        LOGGER.warning("[ALIGN-QC] residual compute/record failed (gate disabled this run): %s", _ar_exc)
        _align_resid, _align_apr = {}, float("nan")
    if getattr(_cfg, "frame_align_residual_gate_enabled", False):
        _kept_ar, _rejected_ar, _ar_thr = _frame_align_residual_gate_select(
            csv_files, _cfg, _align_resid, _align_apr
        )
        if _rejected_ar:
            csv_files = _kept_ar
            logging.info(
                "[ALIGN-QC] alignment-residual gate: rejected %d/%d frames "
                "(thr=%.2f px = %.2f*%.2f apr; %s%s)",
                len(_rejected_ar),
                len(_rejected_ar) + len(_kept_ar),
                _ar_thr,
                float(_cfg.frame_align_residual_max_frac),
                float(_align_apr),
                ", ".join(_rejected_ar[:5]),
                " ..." if len(_rejected_ar) > 5 else "",
            )
            _p2(
                f"Align-residual gate: {len(_rejected_ar)} mis-aligned frames rejected, "
                f"{len(_kept_ar)} kept"
            )
    # Len CSV — bez FITS (flux sa číta z dao_flux v CSV)
    n_frames = len(csv_files)
    _n_total = int(len(at_df))
    logging.info("[PHASE 2A] %d targets (DAO+Gaia matched)", _n_total)
    _p2(f"Phase 2A: {_n_total} targets, {n_frames} frames - loading CSV cache...")

    # Načítaj CSV cache raz pre celú Fázu 2A (read_flux_from_csv per target inak 82× na target).
    if proc_frame_store is not None and len(proc_frame_store) > 0:
        _phase2a_csv_cache = proc_frame_store
        logging.info(
            "[PERF-5] run_phase2a: using ProcFrameStore (%d frames, 0 disk reads)",
            len(proc_frame_store),
        )
        _p2(f"Fáza 2A: ProcFrameStore {len(proc_frame_store)} CSV — výpočet FWHM / apertúr…")
    else:
        logging.info("[FÁZA 2A] Načítavam CSV cache...")
        _t_cache = time.time()
        _phase2a_csv_cache: dict[str, pd.DataFrame] = {}
        _needed_cols_2a = list(
            dict.fromkeys(
                [
                    "catalog_id",
                    "name",
                    "bjd_tdb_mid",
                    "hjd_mid",
                    "jd_mid",
                    "dao_flux",
                    "noise_floor_adu",
                    "sky_adu_per_px_annulus",
                    "aperture_r_px",
                    "peak_max_adu",
                    "airmass",
                    "x",
                    "y",
                    "flux_small",
                    "flux_large",
                    # Variability / auto-export (TODO-PERF-6) — same cache, no second disk read
                    "mag",
                    "bp_rp",
                    "b_v",
                    "zone",
                    "source_type",
                    "vsx_known_variable",
                    "gaia_dr3_variable_catalog",
                    "ra_deg",
                    "dec_deg",
                    "photometry_ok",
                    "edge_safe_10px",
                    "edge_fail",
                    "snr50_ok",
                    "is_saturated",
                    "likely_saturated",
                    "is_usable",
                    "psf_flux",
                    "psf_flux_err",
                    "psf_fit_ok",
                    "psf_chi2",
                    "psf_quality",
                    "psf_quality_fallback",
                    "psf_snr",
                    "psf_ac_factor",
                    "psf_ac_n_used",
                    "psf_ac_applied",
                    "catalog_match_mode",
                ]
            )
        )
        for _csv_path in csv_files:
            try:
                _hdr = pd.read_csv(_csv_path, nrows=0)
                _cols = [c for c in _needed_cols_2a if c in _hdr.columns]
                if not _cols:
                    continue
                # Gaia ID musí byť str — float64 stráca cifry
                _dtype_2a: dict[str, type] = {}
                if "catalog_id" in _cols:
                    _dtype_2a["catalog_id"] = str
                if "name" in _cols:
                    _dtype_2a["name"] = str
                _kw: dict[str, Any] = {"usecols": _cols, "low_memory": False}
                if _dtype_2a:
                    _kw["dtype"] = _dtype_2a
                _phase2a_csv_cache[str(_csv_path)] = pd.read_csv(_csv_path, **_kw)
            except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
                from except_fix_counters import get_except_fix_counters

                get_except_fix_counters().phase2a_csv_cache_skip += 1
                logging.error("[PHASE 2A] proc CSV cache skip %s: %s", _csv_path, exc)
                continue
        logging.info(
            f"[FÁZA 2A] CSV cache: {len(_phase2a_csv_cache)} súborov "
            f"({time.time() - _t_cache:.1f}s)"
        )
        _p2(f"Fáza 2A: cache {len(_phase2a_csv_cache)} CSV — výpočet FWHM / apertúr…")

    # Lookup (id_map + xy_df) raz na snímku — inak _build_csv_lookup 82× na target.
    _phase2a_lookup_cache: dict[str, tuple[dict[str, pd.Series], pd.DataFrame]] = {}
    for _cp in csv_files:
        _key = str(_cp)
        _df_lu = _phase2a_csv_cache.get(_key)
        if _df_lu is None or _df_lu.empty:
            continue
        _id_col_lu = "catalog_id" if "catalog_id" in _df_lu.columns else "name"
        _phase2a_lookup_cache[_key] = _build_csv_lookup(_df_lu, _id_col_lu)

    # Čas + airmass (+ flip flag z alignment_report) z prvého platného riadku každého per-frame CSV
    # (podľa stem FITS).
    frame_time_lookup: dict[str, dict[str, float]] = {}
    for csv_path in csv_files:
        stem = csv_path.stem
        _csv_tmp = _phase2a_csv_cache.get(str(csv_path))
        if _csv_tmp is None or _csv_tmp.empty:
            continue
        _cmm_frame = ""
        if "catalog_match_mode" in _csv_tmp.columns:
            _cmm_s = _csv_tmp["catalog_match_mode"].dropna()
            if len(_cmm_s) > 0:
                _cmm_frame = normalize_catalog_match_mode(str(_cmm_s.iloc[0]))
        try:
            for col_bjd, col_hjd, col_jd in (("bjd_tdb_mid", "hjd_mid", "jd_mid"),):
                if not all(c in _csv_tmp.columns for c in (col_bjd, col_hjd, col_jd)):
                    continue
                vals = _csv_tmp[[col_bjd, col_hjd, col_jd]].dropna()
                if len(vals) == 0:
                    continue
                am_val = float("nan")
                for am_col in ("airmass", "AIRMASS", "air_mass"):
                    if am_col not in _csv_tmp.columns:
                        continue
                    am_series = pd.to_numeric(_csv_tmp[am_col], errors="coerce").dropna()
                    if len(am_series) > 0:
                        am_val = float(am_series.iloc[0])
                    break
                frame_time_lookup[stem] = {
                    "bjd": float(vals[col_bjd].iloc[0]),
                    "hjd": float(vals[col_hjd].iloc[0]),
                    "jd": float(vals[col_jd].iloc[0]),
                    "airmass": am_val,
                }
                if _cmm_frame:
                    frame_time_lookup[stem]["catalog_match_mode"] = _cmm_frame
                break
            if stem not in frame_time_lookup and _cmm_frame:
                frame_time_lookup[stem] = {
                    "bjd": float("nan"),
                    "hjd": float("nan"),
                    "jd": float("nan"),
                    "airmass": float("nan"),
                    "catalog_match_mode": _cmm_frame,
                }
        except Exception as exc:  # noqa: BLE001
            logging.error('[EXC-0167] catalog_match_mode not stored in frame_time_lookup for one frame stem: %s', exc)
            pass

    # Propagate meridian-flip / rotation-change flag from alignment_report.csv when present.
    try:
        align_rep = Path(masterstar_fits_path).resolve().parent / "alignment_report.csv"
        if align_rep.is_file():
            rep = pd.read_csv(align_rep, low_memory=False)
            if "file" in rep.columns:
                for _, r in rep.iterrows():
                    fn = str(r.get("file", "")).strip()
                    if not fn:
                        continue
                    stem = Path(fn).stem
                    if stem not in frame_time_lookup:
                        continue
                    if "is_flipped" in rep.columns:
                        try:
                            frame_time_lookup[stem]["is_flipped"] = bool(r.get("is_flipped", False))
                        except Exception:  # noqa: BLE001
                            frame_time_lookup[stem]["is_flipped"] = False
                    if "aligned" in rep.columns:
                        try:
                            _al = r.get("aligned", True)
                            if isinstance(_al, bool):
                                _aligned = bool(_al)
                            else:
                                s = str(_al).strip().lower()
                                _aligned = s not in ("false", "0", "no", "n")
                            frame_time_lookup[stem]["alignment_failed"] = not _aligned
                        except Exception:  # noqa: BLE001
                            frame_time_lookup[stem]["alignment_failed"] = False
                    if "reason" in rep.columns:
                        _rs = str(r.get("reason", "") or "").strip()
                        if _rs:
                            frame_time_lookup[stem]["alignment_reason"] = _rs
    except Exception as exc:  # noqa: BLE001
        logging.error('[EXC-0168] alignment_report.csv alignment_failed/reason not propagated into frame_time_lookup: %s', exc)
        pass

    # Krok 1: Globálna fixná apertúra — všetky hviezdy (target + comp), faktor × FWHM
    # Ciele so skip_photometry (saturované) nepatria do výpočtu apertúr / FWHM z targetov.
    _at_cols = [c for c in ("catalog_id", "x", "y", "mag") if c in at_df.columns]
    _comp_cols = [c for c in ("catalog_id", "x", "y", "mag") if c in comp_df.columns]
    _at_use = at_df.loc[~at_df["skip_photometry"], _at_cols].copy() if _at_cols else at_df.iloc[0:0].copy()
    _at_part = _at_use.copy()
    _comp_part = comp_df[_comp_cols].drop_duplicates("catalog_id").copy()
    if "mag" not in _at_part.columns:
        _at_part["mag"] = float("nan")
    if "mag" not in _comp_part.columns:
        _comp_part["mag"] = float("nan")
    all_stars = pd.concat(
        [
            _at_part[["catalog_id", "x", "y", "mag"]],
            _comp_part[["catalog_id", "x", "y", "mag"]],
        ],
        ignore_index=True,
    ).drop_duplicates("catalog_id")

    # Priorita: 1. VY_FWHM_GAUSS (2D fit v hlavičke), 2. VY_FWHM (DAO), 3. fit fallback
    _fwhm_from_header: float | None = None
    try:
        hdr = _ms_header
        if hdr is None and _ms_path.is_file():
            with astrofits.open(_ms_path, memmap=False) as _hdul:
                hdr = _hdul[0].header
        if hdr is not None:
            try:
                _obsg = hdr.get("VY_OBSG", None) or hdr.get("OBSG", None) or hdr.get("FILTER", None)
                if _obsg is not None:
                    _s = str(_obsg).strip()
                    if _s:
                        obs_group = _s
            except Exception as exc:  # noqa: BLE001
                logging.error('[EXC-0169] VY_OBSG/OBSG/FILTER header tag not parsed - obs_group label missing from frame metadata: %s', exc)
                pass
            vy_fwhm_gauss = hdr.get("VY_FWHM_GAUSS", None)
            vy_fwhm_dao = hdr.get("VY_FWHM", None)
            if vy_fwhm_gauss is not None:
                _fvg = float(vy_fwhm_gauss)
                if 0.5 < _fvg < 30.0:
                    _fwhm_from_header = _fvg
                    logging.info(
                        f"[FÁZA 2A] FWHM z VY_FWHM_GAUSS (2D fit): {_fwhm_from_header:.3f} px"
                    )
            if _fwhm_from_header is None and vy_fwhm_dao is not None:
                _fvd = float(vy_fwhm_dao)
                if 0.5 < _fvd < 30.0:
                    _fwhm_from_header = _fvd
                    logging.info(
                        f"[FÁZA 2A] FWHM z VY_FWHM (DAO): {_fwhm_from_header:.3f} px"
                    )
    except Exception as _e:  # noqa: BLE001
        logging.error('[EXC-0170] MASTERSTAR header FWHM read throws - measured FWHM from stars used instead (logged warn...: %s', exc)
        logging.warning(f"[FÁZA 2A] Nemôžem čítať FWHM z hlavičky: {_e}")

    if _fwhm_from_header is not None:
        fwhm_px = _fwhm_from_header
    else:
        _fallback_hint = float(fwhm_px) if math.isfinite(fwhm_px) and fwhm_px > 0 else 3.5
        fwhm_px = measure_fwhm_from_masterstar(
            Path(masterstar_fits_path),
            all_stars,
            dao_fwhm_hint=_fallback_hint,
            ms_data=_ms_data,
        )
        logging.info(f"[FÁZA 2A] FWHM z Gaussian fit: {fwhm_px:.3f} px")

    _p2(f"Fáza 2A: FWHM={float(fwhm_px):.3f} px — mapa poľa a svetelné krivky…")

    # Gain / read noise for photometric errors and SNR aperture table.
    # Gain: header-first (e-/ADU or index-mapped) -> DB -> config. RN: DB-first.
    from param_resolver import resolve_gain, resolve_read_noise  # noqa: PLC0415

    _equipment_id = _resolve_phase2a_equipment_id(
        db,
        draft_id=draft_id,
        output_dir=output_dir,
        masterstar_fits_path=Path(masterstar_fits_path),
    )
    _gain_res = resolve_gain(_ms_header, db=db, equipment_id=_equipment_id, cfg=_cfg)
    _rn_res = resolve_read_noise(_ms_header, db=db, equipment_id=_equipment_id, cfg=_cfg)
    _gain_phot = float(_gain_res.value) if _gain_res.ok else 1.0
    _rn_phot = float(_rn_res.value) if _rn_res.ok else 10.0
    logging.info(
        "[PHASE 2A] Photometric errors: gain=%.3f e-/ADU (source: %s), RN=%.1f e- (source: %s)",
        float(_gain_phot),
        _gain_res.source if _gain_res.ok else "default",
        float(_rn_phot),
        _rn_res.source if _rn_res.ok else "default",
    )

    _median_sky = _median_sky_from_phase2a_csv_cache(_phase2a_csv_cache)
    _bkg_var_px = _median_bkg_var_adu2_per_px_from_proc_cache(_phase2a_csv_cache)
    _snr_ap_table: dict[str, Any] | None = None
    if force_aperture_px is None and bool(_cfg.aperture_photometry_enabled):
        _snr_ap_table = compute_snr_optimal_aperture_table(
            fwhm_px=float(fwhm_px),
            sky_adu_per_px=float(_median_sky),
            gain=float(_gain_phot),
            read_noise=float(_rn_phot),
            bkg_var_adu2_per_px=_bkg_var_px,
        )
        _tbl = _snr_ap_table.get("table") or {}
        logging.info(
            "[PHASE 2A] SNR-optimal aperture table built: "
            "mag 7→%.2fpx mag 11→%.2fpx mag 14→%.2fpx mag 17→%.2fpx",
            float(_tbl.get(7.0, float("nan"))),
            float(_tbl.get(11.0, float("nan"))),
            float(_tbl.get(14.0, float("nan"))),
            float(_tbl.get(17.0, float("nan"))),
        )
        try:
            _draft_dir = _draft_dir_from_phase2a_paths(output_dir, Path(masterstar_fits_path))
            _ap_table_path = Path(_draft_dir) / "aperture_snr_table.json"
            with _ap_table_path.open("w", encoding="utf-8") as _f:
                json.dump(_snr_ap_table, _f, indent=2)
            logging.info("[PHASE 2A] Aperture SNR table saved: %s", _ap_table_path)
        except Exception as _ap_exc:  # noqa: BLE001
            logging.error('[EXC-0171] aperture_snr_table.json write to draft dir fails - table used in-memory but not persist...: %s', exc)
            logging.warning("[PHASE 2A] Could not save aperture_snr_table.json: %s", _ap_exc)

    _star_mag_by_cid = _phase2a_star_mag_lookup(at_df, comp_df, Path(masterstar_fits_path))
    _variable_target_cids = frozenset(
        c
        for _, row in at_df.iterrows()
        for c in [_normalize_gaia_id(row.get("catalog_id", ""))]
        if c
    )

    if force_aperture_px is not None and force_aperture_px > 0:
        # Fixná apertura pre všetky hviezdy — debug/kalibrácia
        apertures_px = {
            _normalize_gaia_id(row.get("catalog_id", "")): float(force_aperture_px)
            for _, row in all_stars.iterrows()
            if _normalize_gaia_id(row.get("catalog_id", ""))
        }
        logging.info(
            f"[FÁZA 2A] FORCE apertura: {force_aperture_px:.2f}px pre všetky hviezdy"
        )
    elif _snr_ap_table is not None:
        apertures_px = {}
        for _, row in all_stars.iterrows():
            cid = _normalize_gaia_id(row.get("catalog_id", ""))
            if not cid:
                continue
            _star_mag = float(_star_mag_by_cid.get(cid, float("nan")))
            if not math.isfinite(_star_mag):
                try:
                    _star_mag = float(pd.to_numeric(row.get("mag"), errors="coerce"))
                except Exception:  # noqa: BLE001
                    _star_mag = float("nan")
            apertures_px[cid] = _aperture_radius_from_snr_table(
                _star_mag,
                _snr_ap_table,
                aperture_fwhm_factor=_apt_fw,
                fwhm_px=float(fwhm_px),
            )
        if apertures_px:
            _rvals = list(apertures_px.values())
            logging.info(
                "[FÁZA 2A] SNR per-star apertures: min=%.3fpx median=%.3fpx max=%.3fpx (N=%d)",
                float(min(_rvals)),
                float(np.median(_rvals)),
                float(max(_rvals)),
                len(_rvals),
            )
    else:
        apertures_px = compute_optimal_apertures(
            Path(masterstar_fits_path),
            all_stars,
            fwhm_px,
            aperture_fwhm_factor=_apt_fw,
            annulus_inner_fwhm=annulus_inner_fwhm,
            annulus_outer_fwhm=annulus_outer_fwhm,
        )

    _apply_role_aware_aperture_scaling(apertures_px, at_df, _cfg)

    star_xy: dict[str, tuple[float, float]] = {}
    for _, row in all_stars.iterrows():
        cid = _normalize_gaia_id(row.get("catalog_id", ""))
        if not cid:
            continue
        try:
            star_xy[cid] = (float(row["x"]), float(row["y"]))
        except (KeyError, TypeError, ValueError):
            pass

    sat_limit_resolved = sat_limit_adu if sat_limit_adu is not None else _sat_limit_peak_adu()

    # Field map PNG (raz pre celé pole) — vždy; UI potrebuje mapu aj bez PNG kriviek
    field_map_path = output_dir / "field_map.png"
    save_field_map_png(
        field_map_path,
        Path(masterstar_fits_path),
        at_df,
        comp_df,
        ms_data=_ms_data,
    )

    # PERF-8: collect all star IDs needed across all targets (union targets + comps).
    _all_lc_ids: set[str] = set()
    for _, _trow in at_df.iterrows():
        _tcid = _normalize_gaia_id(_trow.get("catalog_id", ""))
        if _tcid:
            _all_lc_ids.add(_tcid)
        _tcomps = _comp_index.get(_tcid, pd.DataFrame())
        if not _tcomps.empty and "catalog_id" in _tcomps.columns:
            for _ccid in _tcomps["catalog_id"].astype(str):
                _n = _normalize_gaia_id(_ccid)
                if _n:
                    _all_lc_ids.add(_n)
    _all_lc_ids_list = sorted(_all_lc_ids)
    logging.info(
        "[PERF-8] Flux matrix: %d unique star IDs across %d targets",
        len(_all_lc_ids_list),
        len(at_df),
    )

    # PERF-8: one read_flux_from_csv pass per frame for all LC stars (not per target).
    _flux_matrix_rows: list[pd.DataFrame] = []
    _t_flux_matrix = time.perf_counter()
    for _csv_path in csv_files:
        _cached_df = _phase2a_csv_cache.get(str(_csv_path))
        _lookup_row = _phase2a_lookup_cache.get(str(_csv_path))
        if _cached_df is None:
            continue
        _ft = frame_time_lookup.get(_csv_path.stem)
        _df_all = read_flux_from_csv(
            _csv_path,
            _all_lc_ids_list,
            apertures_px,
            sat_limit_adu=sat_limit_resolved,
            star_xy=star_xy,
            xy_tol_px=18.0,
            frame_times=_ft,
            csv_df=_cached_df,
            lookup=_lookup_row,
            gain=float(_gain_phot),
            read_noise=float(_rn_phot),
            use_apcorr_flux=bool(getattr(_cfg, "cog_aperture_correction_enabled", False)),
            variable_target_catalog_ids=_variable_target_cids,
            err_background_mode=str(getattr(_cfg, "err_background_mode", ERR_BKG_MODE_EMPIRICAL)),
        )
        if not _df_all.empty:
            _flux_matrix_rows.append(_df_all)
    _flux_matrix: pd.DataFrame = pd.DataFrame()
    if _flux_matrix_rows:
        _flux_matrix = pd.concat(_flux_matrix_rows, ignore_index=True)
        logging.info(
            "[PERF-8] Flux matrix built: %d rows (%d stars × %d frames) in %.2fs",
            len(_flux_matrix),
            len(_all_lc_ids_list),
            len(csv_files),
            time.perf_counter() - _t_flux_matrix,
        )
    else:
        logging.warning("[PERF-8] Flux matrix empty — per-target per-frame fallback")
    logging.info(
        "[PERF-8] Flux matrix build time: included in photometry timing"
    )

    if not _flux_matrix.empty and (chip_fw is None or chip_fh is None):
        if "x" in _flux_matrix.columns and "y" in _flux_matrix.columns:
            try:
                _xm = float(pd.to_numeric(_flux_matrix["x"], errors="coerce").max())
                _ym = float(pd.to_numeric(_flux_matrix["y"], errors="coerce").max())
            except Exception:  # noqa: BLE001
                _xm, _ym = float("nan"), float("nan")
            if chip_fw is None and math.isfinite(_xm) and _xm > 0:
                chip_fw = int(math.ceil(_xm)) + 2
            if chip_fh is None and math.isfinite(_ym) and _ym > 0:
                chip_fh = int(math.ceil(_ym)) + 2

    _nt = int(len(at_df))
    _plate_scale_arcsec = _resolve_plate_scale_arcsec_per_px(
        _cfg, _ms_path, ms_header=_ms_header
    )
    _gaia_db_path = str(_cfg.gaia_db_path or "").strip() or None
    if _plate_scale_arcsec is not None:
        logging.info(
            "[PHASE 2A] GS11 context: plate_scale=%.4f arcsec/px gaia_db=%s",
            float(_plate_scale_arcsec),
            "set" if _gaia_db_path else "none",
        )
    else:
        logging.warning(
            "[PHASE 2A] GS11 context: plate_scale unknown (derive-or-None) gaia_db=%s",
            "set" if _gaia_db_path else "none",
        )

    # Observer site resolved ONCE per draft (param_resolver): draft ID_LOCATION ->
    # header SITELAT -> flagged config. Threaded into BJD/HJD recompute + meta so
    # Phase 2A no longer silently reads cfg.observer_* (config-drift trap).
    from param_resolver import resolve_site as _resolve_site  # noqa: PLC0415

    _site = _resolve_site(_ms_header, db=db, draft_id=draft_id, cfg=_cfg)
    logging.info(
        "[PHASE 2A] Observer site: source=%s lat=%s lon=%s alt=%s ok=%s",
        _site.source,
        f"{_site.lat:.4f}" if _site.lat is not None else "None",
        f"{_site.lon:.4f}" if _site.lon is not None else "None",
        f"{_site.elev:.0f}" if _site.elev is not None else "None",
        _site.ok,
    )

    _apply_ct = resolve_apply_color_term(_cfg, obs_group)
    _group_ct: _ColorTermGroupFit | None = None
    if _apply_ct:
        _pool_csv = _ensure_group_comp_pool_csv(
            platesolve_dir=Path(masterstar_fits_path).resolve().parent,
            masterstar_fits=Path(masterstar_fits_path),
            masterstars_csv=Path(comparison_stars_csv).resolve().parent / "masterstars_full_match.csv",
            cfg=_cfg,
            draft_id=draft_id,
        )
        _group_ct = _compute_group_color_term_fit(
            comparison_stars_csv=_pool_csv,
            flux_matrix=_flux_matrix,
            csv_files=csv_files,
            obs_group=obs_group,
            cfg=_cfg,
        )
        if _group_ct is not None:
            log_event(f"[COLOR TERM] group pool fit ({obs_group}): {_group_ct.gate_reason}")
        else:
            logging.warning("[COLOR TERM] group pool fit unavailable for %s", obs_group)
    else:
        log_event(f"[COLOR TERM] disabled for {obs_group} (apply_color_term toggle / filter type)")

    from k2_extinction import resolve_k2_bprp_value  # noqa: PLC0415

    _k2_bprp, _k2_src_enum = resolve_k2_bprp_value(_cfg, obs_group)
    if _k2_src_enum.value != "none" and math.isfinite(float(_k2_bprp)):
        log_event(f"[K2] obs_group {obs_group}: k2={float(_k2_bprp):.6f} source={_k2_src_enum.value}")

    # Run-effective resolved facts for the honest full-config report (metadata only).
    _resolved_facts = _build_phase2a_resolved_facts(
        cfg=_cfg,
        gain_res=_gain_res,
        rn_res=_rn_res,
        gain_value=_gain_phot,
        rn_value=_rn_phot,
        site=_site,
        sat_limit=sat_limit_resolved,
        plate_scale_arcsec=_plate_scale_arcsec,
        frame_width_px=chip_fw,
        frame_height_px=chip_fh,
        ms_header=_ms_header,
        obs_group=obs_group,
    )

    return _Phase2AState(
        at_df=at_df,
        comp_df=comp_df,
        _comp_index=_comp_index,
        target_bp_rp_by_cid=target_bp_rp_by_cid,
        csv_files=csv_files,
        n_frames=n_frames,
        _phase2a_csv_cache=_phase2a_csv_cache,
        _phase2a_lookup_cache=_phase2a_lookup_cache,
        frame_time_lookup=frame_time_lookup,
        fwhm_px=fwhm_px,
        apertures_px=apertures_px,
        star_xy=star_xy,
        chip_fw=chip_fw,
        chip_fh=chip_fh,
        _ms_header=_ms_header,
        _ms_data=_ms_data,
        _flux_matrix=_flux_matrix,
        _all_lc_ids_list=_all_lc_ids_list,
        field_map_path=field_map_path,
        obs_group=obs_group,
        _gain_phot=_gain_phot,
        _rn_phot=_rn_phot,
        equipment_id=_equipment_id,
        sat_limit_resolved=sat_limit_resolved,
        _aligned_dir_2a=_aligned_dir_2a,
        _cfg=_cfg,
        _nt=_nt,
        plate_scale_arcsec=(
            float(_plate_scale_arcsec)
            if _plate_scale_arcsec is not None
            and math.isfinite(float(_plate_scale_arcsec))
            and float(_plate_scale_arcsec) > 0
            else float("nan")
        ),
        gaia_db_path=_gaia_db_path,
        masterstars_df=masterstars_df,
        site_lat=_site.lat,
        site_lon=_site.lon,
        site_alt=_site.elev,
        site_source=_site.source,
        site_ok=bool(_site.ok),
        group_color_term=_group_ct,
        apply_color_term=bool(_apply_ct),
        k2_bprp=float(_k2_bprp),
        k2_source=str(_k2_src_enum.value),
        variable_target_catalog_ids=_variable_target_cids,
        snr_ap_table=_snr_ap_table,
        resolved_facts=_resolved_facts,
    )


def _recompute_bjd_hjd_with_status(
    jd_array: np.ndarray,
    ra_deg: float,
    dec_deg: float,
    cfg: AppConfig,
    site: tuple[float | None, float | None, float | None] | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Recompute per-target BJD(TDB) and HJD from frame JD values.

    Returns ``(bjd, hjd, time_base)`` where ``time_base`` is ``BJD_TDB`` on the astropy
    success path or ``JD_FALLBACK`` when raw JD is copied for both BJD and HJD.

    See ``_recompute_bjd_hjd_per_target`` for full docstring / references.
    """
    jd_arr = np.asarray(jd_array, dtype=float)
    bjd_out = np.full_like(jd_arr, float("nan"), dtype=float)
    hjd_out = np.full_like(jd_arr, float("nan"), dtype=float)

    if site is not None and site[0] is not None and site[1] is not None:
        lat = float(site[0])
        lon = float(site[1])
        alt = float(site[2]) if site[2] is not None else 0.0
    else:
        lat = float(cfg.observer_lat)
        lon = float(cfg.observer_lon)
        alt = float(cfg.observer_alt_m)

    from param_resolver import is_null_island_coords  # noqa: PLC0415

    if not math.isfinite(ra_deg) or not math.isfinite(dec_deg):
        LOGGER.warning(
            "BJD-PERTARGET: invalid coords ra=%s dec=%s — using frame JD fallback",
            ra_deg,
            dec_deg,
        )
        return jd_arr.copy(), jd_arr.copy(), TIME_BASE_JD_FALLBACK

    if is_null_island_coords(lat, lon):
        LOGGER.warning("BJD-PERTARGET: observer location not set — using frame JD fallback")
        return jd_arr.copy(), jd_arr.copy(), TIME_BASE_JD_FALLBACK

    try:
        import astropy.units as u
        from astropy.coordinates import EarthLocation, SkyCoord
        from astropy.time import Time

        location = EarthLocation(
            lat=lat * u.deg,
            lon=lon * u.deg,
            height=alt * u.m,
        )
        target = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
        finite = np.isfinite(jd_arr)
        if not np.any(finite):
            return bjd_out, hjd_out, TIME_BASE_BJD_TDB

        t = Time(jd_arr[finite], format="jd", scale="utc", location=location)
        ltt_bary = t.light_travel_time(target, "barycentric")
        bjd_finite = (t.tdb + ltt_bary).jd
        ltt_helio = t.light_travel_time(target, "heliocentric")
        hjd_finite = (t + ltt_helio).jd

        bjd_out[finite] = np.asarray(bjd_finite, dtype=float)
        hjd_out[finite] = np.asarray(hjd_finite, dtype=float)
        n_ok = int(np.sum(np.isfinite(bjd_out[finite])))
        LOGGER.debug(
            "BJD-PERTARGET: %d/%d frames recomputed (batch) for ra=%.4f dec=%.4f",
            n_ok,
            int(finite.sum()),
            ra_deg,
            dec_deg,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("BJD-PERTARGET: batch recompute failed (%s) — frame JD fallback", exc)
        return jd_arr.copy(), jd_arr.copy(), TIME_BASE_JD_FALLBACK

    return bjd_out, hjd_out, TIME_BASE_BJD_TDB


def _recompute_bjd_hjd_per_target(
    jd_array: np.ndarray,
    ra_deg: float,
    dec_deg: float,
    cfg: AppConfig,
    site: tuple[float | None, float | None, float | None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Recompute per-target BJD(TDB) and HJD from frame JD values.

    Uses target's own RA/Dec instead of field-center coordinates.
    Eliminates Roemer LTT error of up to ~12s for targets at field edge.

    Batch astropy Time() over all frames (scalar compute_hjd_bjd is ~12 ms/call).

    ``site`` (lat, lon, alt) is the per-draft resolved observer location
    (param_resolver: draft ID_LOCATION -> header SITELAT -> flagged config).
    When provided it OVERRIDES ``cfg.observer_*`` so Phase 2A is independent of
    config drift between sessions. ``cfg`` is used only as a legacy fallback.

    References:
        Eastman, Siverd & Gaudi (2010) PASP 122, 935 — BJD standards
        time_utils.compute_hjd_bjd() for scalar equivalence
    """
    bjd, hjd, _ = _recompute_bjd_hjd_with_status(jd_array, ra_deg, dec_deg, cfg, site=site)
    return bjd, hjd


def _phase2a_process_one_target(
    target_row: Any,
    *,
    ti: int,
    state: _Phase2AState,
    summary_rows: list,
    n_lc: int,
    lc_dir: Path,
    output_dir: Path,
    progress_cb: Any,
    masterstar_fits_path: Path,
    annulus_inner_fwhm: float,
    annulus_outer_fwhm: float,
    outlier_sigma: float,
    stability_sigma: float,
    _apt_fw: float,
    _save_png: bool,
    ac_sign_logged: list[bool],
) -> tuple[list, int]:
    """Process one target through the full Phase 2A photometry pipeline.

    Returns updated (summary_rows, n_lc).
    """
    def _p2(msg: str) -> None:
        if progress_cb is None:
            return
        try:
            progress_cb(str(msg))
        except UnicodeEncodeError:
            progress_cb(str(msg).encode("ascii", "backslashreplace").decode("ascii"))

    _comp_index = state._comp_index
    target_bp_rp_by_cid = state.target_bp_rp_by_cid
    csv_files = state.csv_files
    _phase2a_csv_cache = state._phase2a_csv_cache
    _phase2a_lookup_cache = state._phase2a_lookup_cache
    frame_time_lookup = state.frame_time_lookup
    fwhm_px = state.fwhm_px
    apertures_px = state.apertures_px
    star_xy = state.star_xy
    chip_fw = state.chip_fw
    chip_fh = state.chip_fh
    _ms_data = state._ms_data
    _flux_matrix = state._flux_matrix
    obs_group = state.obs_group
    _gain_phot = state._gain_phot
    _rn_phot = state._rn_phot
    sat_limit_resolved = state.sat_limit_resolved
    _aligned_dir_2a = state._aligned_dir_2a
    _cfg = state._cfg
    _nt = state._nt
    comp_df = state.comp_df
    _lunar = state.lunar_context

    target_cid = _normalize_gaia_id(target_row.get("catalog_id", ""))
    target_name = _target_display_name(target_row, fallback_cid=target_cid)
    target_vsx_type = str(target_row.get("vsx_type", "") or "").strip()
    _sp = target_row.get("skip_photometry", False)
    if isinstance(_sp, (bool, np.bool_)):
        skip_photo = bool(_sp)
    else:
        skip_photo = str(_sp).strip().lower() in ("1", "true", "yes", "t")
    _zf_row = str(target_row.get("zone_flag", "")).strip()
    _zf_low = _zf_row.lower()
    if _zf_low == "saturated":
        skip_photo = True
    if progress_cb is not None and (
        ti == 1 or ti == _nt or (_nt > 1 and ti % max(1, _nt // 12) == 0)
    ):
        _p2(f"Fáza 2A: cieľ {ti}/{_nt}: {target_name[:50]}")
    if skip_photo:
        _skip_reason = "saturovaný cieľ"
        logging.info(f"[FÁZA 2A] Preskakujem fotometriu ({_skip_reason}): {target_name}")
        summary_rows.append(
            {
                "catalog_id": target_cid,
                "vsx_name": target_name,
                "zone_flag": _zf_row,
                "n_frames": 0,
                "n_good_comp": 0,
                "n_saturated": 0,
                "lc_rms": float("nan"),
                "lc_median_mag": float("nan"),
                "aperture_px": float("nan"),
                "am_slope": float("nan"),
                "am_detrended": False,
                "lc_csv": "",
                "lc_png": "",
            }
        )
        return summary_rows, n_lc
    logging.info(
        f"[FÁZA 2A] Spúšťam: target={target_name}, "
        f"frames={len(csv_files)}, "
        f"apertura={_apt_fw * float(fwhm_px):.2f}px "
        f"(FWHM={float(fwhm_px):.3f}px × {_apt_fw:.2f})"
    )

    # Comp hviezdy pre tento target
    target_comps = _comp_index.get(target_cid, pd.DataFrame()).copy()
    _star_xy = dict(star_xy)

    if target_comps.empty:
        summary_rows = _phase2a_skip_empty_comps_target(
            target_cid=target_cid,
            target_name=target_name,
            zone_flag=_zf_row,
            summary_rows=summary_rows,
        )
        return summary_rows, n_lc

    comp_ids: list[str] = []
    _seen_comp: set[str] = set()
    for c in target_comps["catalog_id"].tolist():
        nc = _normalize_gaia_id(c)
        if nc and nc not in _seen_comp:
            _seen_comp.add(nc)
            comp_ids.append(nc)
    all_ids = [target_cid] + comp_ids

    # Katalógové magnitúdy comp hviezd
    comp_catalog_mag = {
        _normalize_gaia_id(r["catalog_id"]): float(r.get("mag", float("nan")))
        for _, r in target_comps.iterrows()
    }
    tier_weights = {
        1: float(_cfg.comp_tier1_weight),
        2: float(_cfg.comp_tier2_weight),
        3: float(_cfg.comp_tier3_weight),
        4: float(_cfg.comp_tier4_weight),
    }
    for _k in list(tier_weights.keys()):
        try:
            _v = float(tier_weights[_k])
        except Exception:  # noqa: BLE001
            _v = float("nan")
        if not math.isfinite(_v) or _v <= 0:
            tier_weights[_k] = 0.01
        else:
            tier_weights[_k] = max(0.01, float(_v))

    comp_tier_map: dict[str, int] = {}
    for _, r in target_comps.iterrows():
        cid0 = _normalize_gaia_id(r["catalog_id"])
        try:
            t0 = int(pd.to_numeric(r.get("comp_tier", 4), errors="coerce") or 4)
        except Exception:  # noqa: BLE001
            t0 = 4
        comp_tier_map[cid0] = int(max(1, min(4, t0)))

    comp_rms_map: dict[str, float] = {}
    for _, r in target_comps.iterrows():
        cid0 = _normalize_gaia_id(r["catalog_id"])
        try:
            rms_raw = float(r.get("comp_rms", float("nan")))
        except Exception:  # noqa: BLE001
            rms_raw = float("nan")
        tier = int(comp_tier_map.get(cid0, 4))
        tw = float(tier_weights.get(int(tier), 0.25))
        if math.isfinite(rms_raw) and rms_raw > 1e-6 and math.isfinite(tw) and tw > 0:
            comp_rms_map[cid0] = float(rms_raw) / math.sqrt(float(tw))
        else:
            comp_rms_map[cid0] = float(rms_raw)

    _chk_cid_pref: str | None = None
    try:
        from check_star_kmag import (  # noqa: PLC0415
            field_check_star_candidate_pool,
            select_check_star,
        )

        _chk_pool_pref = field_check_star_candidate_pool(
            state.comp_df,
            target_comps=target_comps,
        )
        if not _chk_pool_pref.empty:
            _chk_row_pref = select_check_star(
                _chk_pool_pref,
                ensemble_ids=set(comp_ids),
                n_comp_min=max(1, min(3, len(_chk_pool_pref))),
                cfg=_cfg,
            )
            if _chk_row_pref is not None:
                _chk_cid_pref = _normalize_gaia_id(_chk_row_pref.get("catalog_id", ""))
                if (
                    _chk_cid_pref
                    and _chk_cid_pref not in comp_ids
                    and _chk_cid_pref != target_cid
                ):
                    all_ids.append(_chk_cid_pref)
                    for _mk in ("mag", "phot_g_mean_mag"):
                        try:
                            _cm = float(pd.to_numeric(_chk_row_pref.get(_mk), errors="coerce"))
                        except Exception:  # noqa: BLE001
                            _cm = float("nan")
                        if math.isfinite(_cm):
                            comp_catalog_mag[_chk_cid_pref] = _cm
                            break
                    try:
                        _cx = float(pd.to_numeric(_chk_row_pref.get("x"), errors="coerce"))
                        _cy = float(pd.to_numeric(_chk_row_pref.get("y"), errors="coerce"))
                    except Exception:  # noqa: BLE001
                        _cx, _cy = float("nan"), float("nan")
                    if math.isfinite(_cx) and math.isfinite(_cy):
                        _star_xy[_chk_cid_pref] = (_cx, _cy)
    except (ImportError, KeyError, TypeError, ValueError, AttributeError) as _ck_pref_exc:
        logging.debug("[CHECK-KMAG] preselect skipped for %s: %s", target_cid, _ck_pref_exc)

    # Krok 2: Fotometria per snímka (PERF-8: slice shared flux matrix when built)
    frame_results: list[pd.DataFrame] = []
    if not _flux_matrix.empty:
        _id_set = set(all_ids)
        _target_slice = _flux_matrix[_flux_matrix["catalog_id"].isin(_id_set)]
        for csv_path in csv_files:
            _sf = csv_path.name
            _df_sub = _target_slice[_target_slice["source_file"] == _sf]
            if _df_sub.empty:
                continue
            df_frame = _df_sub.copy()
            _ft = frame_time_lookup.get(csv_path.stem)
            _cached_df = _phase2a_csv_cache.get(str(csv_path))
            if (chip_fw is None or chip_fh is None) and ("x" in df_frame.columns and "y" in df_frame.columns):
                try:
                    _xm = float(pd.to_numeric(df_frame["x"], errors="coerce").max())
                    _ym = float(pd.to_numeric(df_frame["y"], errors="coerce").max())
                except Exception:  # noqa: BLE001
                    _xm, _ym = float("nan"), float("nan")
                if chip_fw is None and math.isfinite(_xm) and _xm > 0:
                    chip_fw = int(math.ceil(_xm)) + 2
                if chip_fh is None and math.isfinite(_ym) and _ym > 0:
                    chip_fh = int(math.ceil(_ym)) + 2
            if chip_fw is not None and chip_fh is not None and int(chip_fw) > 0 and int(chip_fh) > 0:
                tmask = df_frame["catalog_id"].astype(str).str.strip().eq(target_cid)
                if bool(tmask.any()):
                    tr = df_frame.loc[tmask].iloc[0]
                    try:
                        x_t = float(pd.to_numeric(tr.get("x"), errors="coerce"))
                        y_t = float(pd.to_numeric(tr.get("y"), errors="coerce"))
                    except Exception:  # noqa: BLE001
                        x_t, y_t = float("nan"), float("nan")
                    try:
                        r_out_t = float(pd.to_numeric(tr.get("sky_annulus_r_out_px", 30.0), errors="coerce"))
                    except Exception:  # noqa: BLE001
                        r_out_t = 30.0
                    if not (math.isfinite(r_out_t) and r_out_t > 0):
                        r_out_t = 30.0
                    if math.isfinite(x_t) and math.isfinite(y_t):
                        edge_ok = (
                            (x_t - r_out_t >= 0)
                            and (x_t + r_out_t <= float(chip_fw))
                            and (y_t - r_out_t >= 0)
                            and (y_t + r_out_t <= float(chip_fh))
                        )
                        if not edge_ok:
                            df_frame = df_frame.copy()
                            df_frame.loc[tmask, "mag_inst"] = float("nan")
                            df_frame.loc[tmask, "flag"] = "edge_fail"
                            if "edge_fail" in df_frame.columns:
                                df_frame.loc[tmask, "edge_fail"] = True
                            logging.info(
                                "[TARGET EDGE] %s: frame %s vyradený — annulus mimo čip (x=%.0f, y=%.0f, r_out=%.1fpx)",
                                str(target_name),
                                str(csv_path.name),
                                float(x_t),
                                float(y_t),
                                float(r_out_t),
                            )
            frame_results.append(df_frame)
    else:
        for csv_path in csv_files:
            _ft = frame_time_lookup.get(csv_path.stem)
            _key_csv = str(csv_path)
            _cached_df = _phase2a_csv_cache.get(_key_csv)
            _lookup_row = _phase2a_lookup_cache.get(_key_csv)

            df_frame = read_flux_from_csv(
                csv_path,
                all_ids,
                apertures_px,
                sat_limit_adu=sat_limit_resolved,
                star_xy=_star_xy,
                xy_tol_px=18.0,
                frame_times=_ft,
                csv_df=_cached_df,
                lookup=_lookup_row,
                gain=float(_gain_phot),
                read_noise=float(_rn_phot),
                use_apcorr_flux=bool(getattr(_cfg, "cog_aperture_correction_enabled", False)),
                variable_target_catalog_ids=state.variable_target_catalog_ids,
                err_background_mode=str(getattr(_cfg, "err_background_mode", ERR_BKG_MODE_EMPIRICAL)),
            )
            if not df_frame.empty:
                if (chip_fw is None or chip_fh is None) and ("x" in df_frame.columns and "y" in df_frame.columns):
                    try:
                        _xm = float(pd.to_numeric(df_frame["x"], errors="coerce").max())
                        _ym = float(pd.to_numeric(df_frame["y"], errors="coerce").max())
                    except Exception:  # noqa: BLE001
                        _xm, _ym = float("nan"), float("nan")
                    if chip_fw is None and math.isfinite(_xm) and _xm > 0:
                        chip_fw = int(math.ceil(_xm)) + 2
                    if chip_fh is None and math.isfinite(_ym) and _ym > 0:
                        chip_fh = int(math.ceil(_ym)) + 2

                if chip_fw is not None and chip_fh is not None and int(chip_fw) > 0 and int(chip_fh) > 0:
                    tmask = df_frame["catalog_id"].astype(str).str.strip().eq(target_cid)
                    if bool(tmask.any()):
                        tr = df_frame.loc[tmask].iloc[0]
                        try:
                            x_t = float(pd.to_numeric(tr.get("x"), errors="coerce"))
                            y_t = float(pd.to_numeric(tr.get("y"), errors="coerce"))
                        except Exception:  # noqa: BLE001
                            x_t, y_t = float("nan"), float("nan")
                        try:
                            r_out_t = float(pd.to_numeric(tr.get("sky_annulus_r_out_px", 30.0), errors="coerce"))
                        except Exception:  # noqa: BLE001
                            r_out_t = 30.0
                        if not (math.isfinite(r_out_t) and r_out_t > 0):
                            r_out_t = 30.0
                        if math.isfinite(x_t) and math.isfinite(y_t):
                            edge_ok = (
                                (x_t - r_out_t >= 0)
                                and (x_t + r_out_t <= float(chip_fw))
                                and (y_t - r_out_t >= 0)
                                and (y_t + r_out_t <= float(chip_fh))
                            )
                            if not edge_ok:
                                df_frame = df_frame.copy()
                                df_frame.loc[tmask, "mag_inst"] = float("nan")
                                df_frame.loc[tmask, "flag"] = "edge_fail"
                                if "edge_fail" in df_frame.columns:
                                    df_frame.loc[tmask, "edge_fail"] = True
                                logging.info(
                                    "[TARGET EDGE] %s: frame %s vyradený — annulus mimo čip (x=%.0f, y=%.0f, r_out=%.1fpx)",
                                    str(target_name),
                                    str(csv_path.name),
                                    float(x_t),
                                    float(y_t),
                                    float(r_out_t),
                                )
                frame_results.append(df_frame)

    if not frame_results:
        return summary_rows, n_lc

    ac_result: dict[str, Any] = {
        "ok": False,
        "delta_m_corr": None,
        "scatter_mag": None,
        "n_ref_stars": 0,
        "ref_star_ids": [],
        "reason": "disabled",
    }
    if bool(_cfg.aperture_correction_enabled):
        try:
            ac_result = compute_aperture_correction(
                comp_df=target_comps,
                frame_results=frame_results,
                min_ref_stars=int(_cfg.aperture_correction_min_ref_stars),
                max_contamination=float(_cfg.aperture_correction_max_contamination),
                max_scatter_mag=float(_cfg.aperture_correction_max_scatter_mag),
            )
            if bool(ac_result.get("ok")):
                log_event(
                    f"[AC] ΔM_corr={float(ac_result['delta_m_corr']):.4f} "
                    f"scatter={float(ac_result['scatter_mag']):.4f} "
                    f"n_ref={int(ac_result['n_ref_stars'])}"
                )
            else:
                log_event(f"[AC] skipped: {ac_result.get('reason', '')}")
        except Exception as _ac_exc:  # noqa: BLE001
            log_event(f"[AC] skipped: exception {_ac_exc!s}")
            ac_result = {
                "ok": False,
                "delta_m_corr": None,
                "scatter_mag": None,
                "n_ref_stars": 0,
                "ref_star_ids": [],
                "reason": "exception",
            }
    _ = ac_result  # Krokom 3: aplikácia na mag_calib / CSV

    all_frames = pd.concat(frame_results, ignore_index=True)

    # Zostav časové rady per hviezda
    target_lc = _get_lc(target_cid, all_frames)
    comp_lc = {cid: _get_lc(cid, all_frames) for cid in comp_ids}

    # Flux sources for method-keyed LC outputs (aperture always primary/default).
    _psf_enabled = bool(_cfg.psf_photometry_enabled)
    _adaptive = bool(getattr(_cfg, "psf_adaptive_enabled", False))
    _have_psf_cols = "psf_flux" in all_frames.columns and "psf_fit_ok" in all_frames.columns
    if _have_psf_cols and (_adaptive or _psf_enabled):
        _blend_map = _load_adaptive_blend_map(masterstar_fits_path)
        all_frames["lc_flux_method"] = compute_lc_flux_method(
            all_frames,
            _blend_map,
            resolve_fwhm=float(getattr(_cfg, "psf_adaptive_resolve_fwhm", 2.0)),
            snr_lo=float(getattr(_cfg, "psf_adaptive_snr_lo", 15.0)),
        )
    # Primary published LC is always aperture (target_lc / comp_lc from _get_lc above).
    _lc_export_method = "aperture"

    # ALG-3: Temporal binning of comp ensemble (MNRAS 2023)
    comp_lc = temporal_bin_comp_lc(
        comp_lc=comp_lc,
        comp_quality={},
        all_frames=all_frames,
        window=int(_cfg.temporal_bin_window),
        enabled=bool(_cfg.temporal_binning_enabled),
    )

    # Krok 3: Stability check
    comp_bjd = {cid: _get_comp_bjd_series(cid, all_frames) for cid in comp_ids}
    comp_quality = check_comparison_stability(
        comp_lc,
        comp_rms_map=comp_rms_map,
        comp_bjd=comp_bjd,
        n_comp_min=3,
        outlier_sigma=stability_sigma,
        max_comp_slope_mmag_hr=float(_cfg.comp_max_slope_mmag_hr),
        comp_slope_significance_k=float(getattr(_cfg, "comp_slope_significance_k", 3.0)),
        common_mode_detrend=True,
        stability_run_flags=state.stability_run_flags,
    )

    # ALG-5: PyTICS iterative comp star intercalibration (RASTI 2026)
    comp_rms_map = pytics_iterative_weights(
        comp_lc=comp_lc,
        comp_quality=comp_quality,
        comp_rms_map=comp_rms_map,
        n_iter=int(_cfg.pytics_n_iter),
        enabled=bool(_cfg.pytics_enabled),
    )

    # Krok 4: Ensemble normalizácia
    mag_calib, delta_mag, ensemble_scatter = ensemble_normalize(
        target_lc,
        comp_lc,
        comp_catalog_mag,
        comp_quality,
        comp_rms_map=comp_rms_map,
        comp_tier_map=comp_tier_map,
        tier_weights=tier_weights,
        n_comp_min=max(1, int(getattr(_cfg, "phase01_comparison_n_comp_min", 3))),
        n_comp_max=int(_cfg.phase01_comparison_n_comp_max),
    )
    _ensemble_scatter_by_file = _ensemble_scatter_by_source_file(
        all_frames, target_cid, ensemble_scatter
    )

    _dilution_result: dict[str, Any] = {
        "dilution_factor": 1.0,
        "dilution_delta_mag": 0.0,
        "n_neighbors": 0,
        "neighbor_flux_sum": 0.0,
        "aperture_arcsec": float("nan"),
        "search_radius_arcsec": float("nan"),
    }
    if bool(_cfg.gs11_dilution_enabled) and state.gaia_db_path:
        from dilution import apply_target_dilution_to_mag_calib, compute_dilution_factor  # noqa: PLC0415

        try:
            _target_ra = float(
                pd.to_numeric(
                    target_row.get("ra_deg", target_row.get("ra", float("nan"))),
                    errors="coerce",
                )
            )
            _target_dec = float(
                pd.to_numeric(
                    target_row.get("dec_deg", target_row.get("dec", float("nan"))),
                    errors="coerce",
                )
            )
        except (TypeError, ValueError):
            _target_ra = _target_dec = float("nan")
        _target_g_mag = float("nan")
        for _gk in ("mag", "phot_g_mean_mag", "catalog_mag"):
            try:
                _gv = float(pd.to_numeric(target_row.get(_gk, float("nan")), errors="coerce"))
            except (TypeError, ValueError):
                _gv = float("nan")
            if math.isfinite(_gv):
                _target_g_mag = _gv
                break
        _ap_cfg = float(_cfg.gs11_dilution_aperture_arcsec)
        _dilution_skipped_ap = False
        if math.isfinite(_ap_cfg) and _ap_cfg > 0:
            _ap_arcsec = _ap_cfg
        else:
            _ap_px, _ap_src = _resolve_photometric_aperture_px_for_gs11(
                target_cid,
                apertures_px,
                _target_g_mag,
                state.snr_ap_table,
                aperture_fwhm_factor=float(_apt_fw),
                fwhm_px=float(fwhm_px),
            )
            if _ap_px is None:
                logging.warning(
                    "[GS11] target %s: photometric aperture unavailable — dilution skipped",
                    target_cid or "?",
                )
                log_event(
                    f"[GS11] target {target_cid or '?'}: photometric aperture unavailable — dilution skipped"
                )
                _dilution_skipped_ap = True
                _ap_arcsec = float("nan")
            else:
                _ap_arcsec = float(_ap_px) * float(state.plate_scale_arcsec)
        _cid_int = None
        try:
            from dilution import _normalize_exclude_source_id  # noqa: PLC0415

            _cid_int = _normalize_exclude_source_id(target_cid)
        except Exception:  # noqa: BLE001
            _cid_int = None
        if _dilution_skipped_ap:
            _dilution_result = {
                "dilution_factor": 1.0,
                "dilution_delta_mag": 0.0,
                "n_neighbors": 0,
                "neighbor_flux_sum": 0.0,
                "aperture_arcsec": float("nan"),
                "search_radius_arcsec": float("nan"),
                "dilution_skipped": True,
                "dilution_skip_reason": "photometric_aperture_unavailable",
            }
        else:
            _dilution_result = compute_dilution_factor(
                _target_ra,
                _target_dec,
                _target_g_mag,
                _ap_arcsec,
                str(state.gaia_db_path),
                catalog_id=_cid_int,
                mag_limit_delta=float(_cfg.gs11_dilution_mag_limit_delta),
            )
        _mag_pre_gs11 = float("nan")
        _finite_pre = mag_calib[np.isfinite(mag_calib)]
        if len(_finite_pre) > 0:
            _mag_pre_gs11 = float(np.median(_finite_pre))
        mag_calib, _dilution_result = apply_target_dilution_to_mag_calib(
            mag_calib,
            _dilution_result,
            _cfg,
            target_cid=str(target_cid),
        )
        _mag_post_gs11 = float("nan")
        _finite_post = mag_calib[np.isfinite(mag_calib)]
        if len(_finite_post) > 0:
            _mag_post_gs11 = float(np.median(_finite_post))
    else:
        _mag_pre_gs11 = float("nan")
        _mag_post_gs11 = float("nan")

    # ── Aperture correction (AC) ──
    ac_ok = bool(ac_result.get("ok", False)) if isinstance(ac_result, dict) else False
    delta_m_corr = ac_result.get("delta_m_corr") if isinstance(ac_result, dict) else None
    if ac_ok and delta_m_corr is not None and np.isfinite(float(delta_m_corr)):
        mag_calib_ac = mag_calib + float(delta_m_corr)
    else:
        mag_calib_ac = np.full_like(mag_calib, float("nan"))

    # Sanity log znamienka: pri delta_m_corr < 0 má byť mag_calib_ac < mag_calib.
    if (not ac_sign_logged[0]) and ac_ok and delta_m_corr is not None and np.isfinite(float(delta_m_corr)):
        if len(mag_calib) > 0 and math.isfinite(float(mag_calib[0])) and math.isfinite(float(mag_calib_ac[0])):
            log_event(
                f"[AC SIGN] mag_calib0={float(mag_calib[0]):.4f} "
                f"delta_m_corr={float(delta_m_corr):.4f} "
                f"mag_calib_ac0={float(mag_calib_ac[0]):.4f}"
            )
            ac_sign_logged[0] = True

    # ── Color term (BP-RP) — global comp-pool fit; toggle controls correction only ──
    target_bp_rp = float(target_bp_rp_by_cid.get(target_cid, float("nan")))
    comp_bp_rp: dict[str, float] = {}
    if "bp_rp" in target_comps.columns:
        for _, rr in target_comps.iterrows():
            cidc = _normalize_gaia_id(rr.get("catalog_id", ""))
            if not cidc:
                continue
            v = pd.to_numeric(rr.get("bp_rp"), errors="coerce")
            try:
                fv = float(v)
            except Exception:  # noqa: BLE001
                fv = float("nan")
            if math.isfinite(fv):
                comp_bp_rp[cidc] = float(fv)

    from k2_extinction import K2Source, apply_k2_per_frame, bp_rp_comp_median  # noqa: PLC0415

    k2_value_lc = float("nan")
    k2_colour_ref = float("nan")
    k2_source_rows = [K2Source.NONE.value] * len(mag_calib)
    _k2_val = float(getattr(state, "k2_bprp", float("nan")))
    _k2_src = str(getattr(state, "k2_source", K2Source.NONE.value))
    if _k2_src == K2Source.LITERATURE_DEFAULT.value and math.isfinite(_k2_val):
        _tf_k2 = all_frames[all_frames["catalog_id"] == target_cid]
        if "airmass" in _tf_k2.columns:
            _airmass_k2 = _tf_k2["airmass"].to_numpy(dtype=float)
        else:
            _airmass_k2 = np.full(len(mag_calib), float("nan"), dtype=float)
        _bp_med_k2 = bp_rp_comp_median(comp_bp_rp, comp_quality)
        mag_calib, _k2_delta, k2_source_rows = apply_k2_per_frame(
            mag_calib,
            _airmass_k2,
            object_bp_rp=float(target_bp_rp),
            bp_rp_comp_med=_bp_med_k2,
            k2_value=_k2_val,
            k2_source=K2Source.LITERATURE_DEFAULT,
        )
        k2_value_lc = _k2_val
        k2_colour_ref = _bp_med_k2

    c1 = 0.0
    ct_n_comp = 0
    mag_calib_ct = mag_calib.copy()
    ct_corr = 0.0
    bp_rp_comp_med = float("nan")
    ct_ok = False
    _group_ct = state.group_color_term
    if state.apply_color_term and _group_ct is not None and _group_ct.apply_gate:
        c1 = float(_group_ct.c1)
        ct_n_comp = int(_group_ct.n_comp)
        _ct_in_range = _check_color_term_extrapolation(
            target_bp_rp=float(target_bp_rp),
            comp_bp_rp_values=[float(v) for v in _group_ct.comp_bp_rp.values()],
            target_name=str(target_name),
            extrapolation_tol=float(_cfg.phase01_ct_extrapolation_tol),
        )
        if _ct_in_range:
            mag_calib_ct, ct_corr, bp_rp_comp_med = apply_color_term(
                mag_calib,
                target_bp_rp,
                _group_ct.comp_bp_rp,
                _group_ct.comp_quality,
                c1,
            )
            ct_ok = (
                bool(math.isfinite(float(target_bp_rp)))
                and float(c1) != 0.0
                and math.isfinite(float(bp_rp_comp_med))
            )
        else:
            logging.info(
                "[COLOR TERM] extrapolation → CT skipped (target kept, uncorrected)"
            )
            mag_calib_ct = mag_calib.copy()
            ct_corr = 0.0
            bp_rp_comp_med = float("nan")
            ct_ok = False

    if _ct_prototype_enabled():
        _proto_c1 = 0.0
        _proto_c1_stderr = float("nan")
        _proto_n_comp = 0
        if comp_bp_rp:
            _proto_c1, _proto_c1_stderr, _proto_n_comp = fit_color_term_c1(
                comp_lc,
                comp_catalog_mag,
                comp_bp_rp,
                comp_quality,
                min_comp=5,
                sigma_clip_sigma=3.0,
            )
        _proto_corr = 0.0
        _proto_comp_med = float("nan")
        if comp_bp_rp and float(_proto_c1) != 0.0:
            _, _proto_corr, _proto_comp_med = apply_color_term(
                mag_calib,
                float(target_bp_rp),
                comp_bp_rp,
                comp_quality,
                float(_proto_c1),
            )
        _proto_scatter, _proto_scatter_resid = (
            _color_term_cat_inst_scatter_pair(
                comp_lc,
                comp_catalog_mag,
                comp_bp_rp,
                comp_quality,
                float(_proto_c1),
                min_comp=5,
                sigma_clip_sigma=3.0,
            )
            if comp_bp_rp
            else (float("nan"), float("nan"))
        )
        _proto_stderr_ratio = float("nan")
        if float(_proto_c1) != 0.0 and math.isfinite(float(_proto_c1_stderr)):
            _proto_stderr_ratio = abs(float(_proto_c1_stderr) / float(_proto_c1))
        _proto_gate = (
            int(_proto_n_comp) >= int(_cfg.phase01_ct_min_comp)
            and float(_proto_c1) != 0.0
            and math.isfinite(_proto_stderr_ratio)
            and float(_proto_stderr_ratio) <= 0.5
        )
        _append_ct_prototype_row(
            _draft_dir_from_phase2a_paths(output_dir, Path(masterstar_fits_path)),
            {
                "catalog_id": target_cid,
                "vsx_name": target_name,
                "obs_group": str(obs_group),
                "n_comp_used": int(_proto_n_comp),
                "c1": float(_proto_c1),
                "c1_stderr": float(_proto_c1_stderr),
                "stderr_ratio": _proto_stderr_ratio,
                "target_bp_rp": float(target_bp_rp),
                "comp_med_bp_rp": float(_proto_comp_med),
                "ct_corr": float(_proto_corr),
                "cat_inst_scatter": _proto_scatter,
                "cat_inst_scatter_resid": _proto_scatter_resid,
                "gate_would_pass": bool(_proto_gate),
            },
        )

    # Časové hodnoty targetu — sort by source_file so ensemble_scatter index aligns
    # with ``_get_lc`` / ``_ensemble_scatter_by_source_file`` (LABBE-DET / SEM determinism).
    target_frames = all_frames[all_frames["catalog_id"] == target_cid]
    if not target_frames.empty and "source_file" in target_frames.columns:
        target_frames = target_frames.sort_values(["source_file"], kind="mergesort")
    _measured_ap_target = _measured_aperture_from_proc_cache(target_cid, state._phase2a_csv_cache)
    if math.isfinite(_measured_ap_target) and _measured_ap_target > 0 and not target_frames.empty:
        target_frames = target_frames.copy()
        target_frames["aperture_r_px"] = float(_measured_ap_target)
    bjd = target_frames["bjd"].to_numpy(dtype=float)
    hjd = target_frames["hjd"].to_numpy(dtype=float)
    jd = target_frames["jd"].to_numpy(dtype=float)

    # BJD-PERTARGET: recompute with target's own RA/Dec (not field-center LTT)
    _target_ra = float(pd.to_numeric(target_row.get("ra_deg", target_row.get("ra", float("nan"))), errors="coerce"))
    _target_dec = float(
        pd.to_numeric(target_row.get("dec_deg", target_row.get("dec", float("nan"))), errors="coerce")
    )
    bjd, hjd, time_base = _recompute_bjd_hjd_with_status(
        jd,
        _target_ra,
        _target_dec,
        _cfg,
        site=(state.site_lat, state.site_lon, state.site_alt) if state.site_ok else None,
    )

    err = target_frames["err"].to_numpy(dtype=float)
    err, err_method_rows = _route_lc_per_frame_err(target_frames, err)
    # Per-point uncertainty = photon/SNR base error (term-1) ⊕ ensemble zeropoint uncertainty
    # (term-3, ``ensemble_scatter``). Joined by EXACT ``source_file`` (G2-F004), not positional index.
    _src_for_err = target_frames["source_file"].astype(str).tolist()
    from sigma_floor_core import resolve_sigma_sys_mag  # noqa: PLC0415

    _sigma_sys_mag = resolve_sigma_sys_mag(
        state.equipment_id,
        _cfg,
        rig_label=str(state.obs_group or ""),
    )
    err, err_scatter_unmatched_arr = _combine_err_with_ensemble_scatter_keyed(
        err,
        _src_for_err,
        _ensemble_scatter_by_file,
        sigma_sys_mag=_sigma_sys_mag,
        target_name=str(target_name),
    )
    ap_arr = target_frames["aperture_r_px"].to_numpy(dtype=float)
    src_files = target_frames["source_file"].tolist()
    sat_flags = (target_frames["flag"] == "saturated").to_numpy(dtype=bool)

    # Airmass / flip arrays for export + the democratic detrender (no per-target airmass detrend here:
    # airmass is handled by the differential comp ensemble).
    if "airmass" in target_frames.columns:
        airmass_arr = target_frames["airmass"].to_numpy(dtype=float)
    else:
        airmass_arr = np.full_like(bjd, float("nan"), dtype=float)
    flip_arr = (
        target_frames["is_flipped"].fillna(False).astype(bool).to_numpy()
        if "is_flipped" in target_frames.columns
        else np.zeros_like(bjd, dtype=bool)
    )
    align_fail_arr = (
        target_frames["alignment_failed"].fillna(False).astype(bool).to_numpy()
        if "alignment_failed" in target_frames.columns
        else np.zeros_like(bjd, dtype=bool)
    )
    n_alignment_failed = int(np.count_nonzero(align_fail_arr))
    alignment_failed_frac = float(n_alignment_failed) / max(int(len(bjd)), 1)
    if "catalog_match_mode" in target_frames.columns:
        catalog_match_mode_list = [
            normalize_catalog_match_mode(v) for v in target_frames["catalog_match_mode"].tolist()
        ]
    else:
        catalog_match_mode_list = [""] * len(bjd)
    if "wcs_untrusted" in target_frames.columns:
        wcs_untrusted_arr = target_frames["wcs_untrusted"].fillna(False).astype(bool).to_numpy()
    else:
        wcs_untrusted_arr = np.array(
            [is_wcs_untrusted_catalog_match_mode(m) for m in catalog_match_mode_list],
            dtype=bool,
        )
    n_wcs_untrusted = int(np.count_nonzero(wcs_untrusted_arr))
    wcs_untrusted_frac = float(n_wcs_untrusted) / max(int(len(bjd)), 1)

    if "flag" in target_frames.columns:
        _raw_tf = target_frames["flag"].astype(str).str.strip().str.lower().reset_index(drop=True)
    else:
        _raw_tf = pd.Series(["__none__"] * len(mag_calib))
    base_flags: list[str] = []
    for i in range(len(mag_calib)):
        if bool(sat_flags[i]):
            base_flags.append("saturated")
        elif i < len(_raw_tf) and str(_raw_tf.iloc[i]) == "nondetection":
            base_flags.append("nondetection")
        elif math.isfinite(mag_calib[i]):
            base_flags.append("normal")
        else:
            base_flags.append("no_data")

    # Reporting path (Workstream B): see ``apply_reporting_postprocess``.
    mag_calib_raw, mag_calib, mag_calib_ct, mag_calib_ac, out_flags = apply_reporting_postprocess(
        mag_calib,
        mag_calib_ct,
        target_row=target_row,
        target_name=target_name,
        sat_flags=sat_flags,
        target_frames=target_frames,
        outlier_sigma=outlier_sigma,
        ct_ok=bool(ct_ok),
        ac_ok=bool(ac_ok),
        delta_m_corr=(float(delta_m_corr) if delta_m_corr is not None else None),
        cfg=_cfg,
    )

    # ALG-2: Savitzky-Golay non-linear detrending (Savitzky & Golay 1964)
    # Removes slow systematic trends (airmass is handled by the differential comp ensemble).
    _sg_enabled = bool(_cfg.savgol_detrend_enabled)
    if _sg_enabled:
        mag_calib = savgol_detrend_lc(
            mag_calib=mag_calib,
            bjd=bjd,
            flags=list(out_flags) if out_flags is not None else ["normal"] * len(mag_calib),
            window_frac=float(_cfg.savgol_window_frac),
            polyorder=int(_cfg.savgol_polyorder),
            enabled=True,
        )
        if ac_ok and delta_m_corr is not None and np.isfinite(float(delta_m_corr)):
            mag_calib_ac = mag_calib + float(delta_m_corr)

    # ALG-4: Democratic Detrender (arXiv:2411.09753v2, 2026)
    _dem_enabled = bool(_cfg.democratic_detrend_enabled)
    _mag_democratic: np.ndarray | None = None
    _err_inflation: np.ndarray | None = None
    if _dem_enabled:
        _mag_democratic, _err_inflation = democratic_detrend_lc(
            mag_calib=mag_calib,
            bjd=bjd,
            airmass=airmass_arr,
            flags=list(out_flags) if out_flags is not None else ["normal"] * len(mag_calib),
            window_frac=float(_cfg.democratic_sg_window_frac),
            enabled=True,
        )

    try:
        from check_star_kmag import (  # noqa: PLC0415
            build_comp_photon_mag_from_frames,
            check_kmag_sidecar_path,
            compute_check_ensemble_mag_calib,
            save_check_kmag_sidecar,
        )

        _chk_cid = _chk_cid_pref
        if _chk_cid:
            _ext_lc = dict(comp_lc)
            if _chk_cid not in _ext_lc:
                _chk_series = _get_lc(_chk_cid, all_frames)
                if _chk_series is not None and np.isfinite(_chk_series).any():
                    _ext_lc[_chk_cid] = _chk_series
            if _chk_cid in _ext_lc:
                _phot_ids = list(dict.fromkeys(list(comp_ids) + [_chk_cid]))
                _comp_photon = build_comp_photon_mag_from_frames(all_frames, _phot_ids, src_files)
                _chk_result = compute_check_ensemble_mag_calib(
                    _chk_cid,
                    list(comp_ids),
                    _ext_lc,
                    comp_catalog_mag,
                    comp_quality,
                    comp_rms_map=comp_rms_map,
                    comp_tier_map=comp_tier_map,
                    tier_weights=tier_weights,
                    cfg=_cfg,
                    n_comp_min=2,
                    n_comp_max=int(_cfg.phase01_comparison_n_comp_max),
                    comp_photon_mag=_comp_photon,
                    sigma_sys_mag=_sigma_sys_mag,
                )
                if _chk_result is not None and np.isfinite(_chk_result.kmag).any():
                    save_check_kmag_sidecar(
                        check_kmag_sidecar_path(lc_dir, target_cid),
                        check_cid=_chk_cid,
                        bjd=bjd,
                        source_files=src_files,
                        kmag=_chk_result.kmag,
                        ensemble=_chk_result,
                    )
    except (ImportError, KeyError, TypeError, ValueError, AttributeError, OSError) as _ck_exc:
        logging.debug("[CHECK-KMAG] sidecar skipped for %s: %s", target_cid, _ck_exc)

    # Krok 6: Uloženie výstupov
    lc_csv = lc_dir / f"lightcurve_{target_cid}.csv"
    if isinstance(_lunar, dict):
        _lc_lunar_phase = float(_lunar.get("lunar_phase_pct", float("nan")))
        _lc_lunar_sep = float(_lunar.get("lunar_separation_deg", float("nan")))
        _lc_lunar_risk = str(_lunar.get("lunar_risk", "UNKNOWN") or "UNKNOWN")
    else:
        _lc_lunar_phase = float("nan")
        _lc_lunar_sep = float("nan")
        _lc_lunar_risk = "UNKNOWN"
    save_lightcurve_csv(
        lc_csv,
        bjd,
        hjd,
        jd,
        airmass_arr,
        flip_arr,
        target_lc,
        mag_calib_raw,
        mag_calib,
        np.asarray(mag_calib_ct, dtype=np.float64),
        mag_calib_ac,
        delta_mag,
        err,
        ap_arr,
        out_flags,
        src_files,
        ct_correction=(float(ct_corr) if bool(ct_ok) else float("nan")),
        ct_c1=(float(c1) if bool(ct_ok) else float("nan")),
        ct_bp_rp_target=(float(target_bp_rp) if bool(ct_ok) else float("nan")),
        ct_bp_rp_comp_med=(float(bp_rp_comp_med) if bool(ct_ok) else float("nan")),
        ct_n_comp=(int(ct_n_comp) if bool(ct_ok) else None),
        ct_ok=bool(ct_ok),
        k2_source=k2_source_rows,
        k2_value=(float(k2_value_lc) if math.isfinite(float(k2_value_lc)) else float("nan")),
        k2_colour_ref=(float(k2_colour_ref) if math.isfinite(float(k2_colour_ref)) else float("nan")),
        ac_result=(ac_result if isinstance(ac_result, dict) else None),
        mag_democratic=_mag_democratic,
        err_inflation=_err_inflation,
        lunar_phase_pct=_lc_lunar_phase,
        lunar_separation_deg=_lc_lunar_sep,
        lunar_risk=_lc_lunar_risk,
        dilution_factor=float(_dilution_result.get("dilution_factor", 1.0)),
        method=_lc_export_method,
        alignment_failed=align_fail_arr,
        err_scatter_unmatched=err_scatter_unmatched_arr,
        catalog_match_mode=catalog_match_mode_list,
        wcs_untrusted=wcs_untrusted_arr,
        time_base=time_base,
        err_method=err_method_rows,
        sigma_sys_mag=_sigma_sys_mag,
    )
    if _have_psf_cols:
        try:
            from method_lc_output import MethodLcWriteContext, save_method_variant_lightcurve  # noqa: PLC0415

            _alt_methods: list[str] = []
            if bool(_cfg.psf_photometry_enabled):
                _alt_methods.append("psf")
            if bool(getattr(_cfg, "psf_adaptive_enabled", False)):
                _alt_methods.append("adaptive")
            if _alt_methods:
                _mctx_base = MethodLcWriteContext(
                    method="psf",
                    target_cid=target_cid,
                    comp_ids=list(comp_ids),
                    all_frames=all_frames,
                    lc_dir=lc_dir,
                    cfg=_cfg,
                    stability_sigma=stability_sigma,
                    outlier_sigma=outlier_sigma,
                    comp_catalog_mag=comp_catalog_mag,
                    comp_rms_map=comp_rms_map,
                    comp_tier_map=comp_tier_map,
                    tier_weights=tier_weights,
                    target_row=target_row,
                    state=state,
                    apertures_px=apertures_px,
                    ac_result=(ac_result if isinstance(ac_result, dict) else None),
                    comp_bp_rp=comp_bp_rp,
                    target_bp_rp=float(target_bp_rp),
                    bjd=bjd,
                    hjd=hjd,
                    jd=jd,
                    airmass_arr=airmass_arr,
                    flip_arr=flip_arr,
                    err=err,
                    ap_arr=ap_arr,
                    src_files=src_files,
                    sat_flags=sat_flags,
                    target_frames=target_frames,
                    lunar_phase_pct=_lc_lunar_phase,
                    lunar_separation_deg=_lc_lunar_sep,
                    lunar_risk=_lc_lunar_risk,
                    time_base=time_base,
                )
                for _alt_m in _alt_methods:
                    try:
                        _mctx = MethodLcWriteContext(**{**_mctx_base.__dict__, "method": _alt_m})
                        save_method_variant_lightcurve(_mctx)
                    except Exception as _alt_exc:  # noqa: BLE001
                        logging.error('[EXC-0172] Alternate method-variant LC file (e.g. detrended) not written for one target/method: %s', exc)
                        logging.warning(
                            "[METHOD-LC] %s %s failed: %s",
                            target_cid,
                            _alt_m,
                            _alt_exc,
                        )
        except Exception as _meth_exc:  # noqa: BLE001
            logging.error('[EXC-0173] All method-variant LC exports for target skipped when init block fails: %s', exc)
            logging.warning("[METHOD-LC] init failed for %s: %s", target_cid, _meth_exc)
    # Kvalita comp pre UI (tabuľka „Porovnávacie hviezdy“)
    _cq_path = lc_dir / f"comp_quality_{target_cid}.json"
    try:
        selected_tier = ""
        tier4_warning = False
        n_t1 = n_t2 = n_t3 = n_t4 = 0
        try:
            if "selected_tier" in comp_df.columns:
                _sub = _comp_index.get(target_cid, pd.DataFrame())
                if not _sub.empty:
                    stv = str(_sub.iloc[0].get("selected_tier", "") or "").strip()
                    selected_tier = stv
                    tier4_warning = bool(_sub.iloc[0].get("tier4_warning", False))
                    try:
                        n_t1 = int(pd.to_numeric(_sub.iloc[0].get("n_tier1", 0), errors="coerce") or 0)
                        n_t2 = int(pd.to_numeric(_sub.iloc[0].get("n_tier2", 0), errors="coerce") or 0)
                        n_t3 = int(pd.to_numeric(_sub.iloc[0].get("n_tier3", 0), errors="coerce") or 0)
                        n_t4 = int(pd.to_numeric(_sub.iloc[0].get("n_tier4", 0), errors="coerce") or 0)
                    except Exception:  # noqa: BLE001
                        n_t1 = n_t2 = n_t3 = n_t4 = 0
        except Exception:  # noqa: BLE001
            selected_tier = ""

        _cq_payload: dict[str, Any] = {}
        for cid, info in comp_quality.items():
            nk = _normalize_gaia_id(cid)
            q = str(info.get("quality", "") or "").strip()
            note = str(info.get("note", "") or "").strip()
            if q == "good" and not note:
                _cq_payload[nk] = "good"
            else:
                _cq_payload[nk] = {"quality": q, "note": note}
        _cq_payload["selected_tier"] = str(selected_tier)
        _cq_payload["tier4_warning"] = bool(tier4_warning)
        _cq_payload["n_tier1"] = int(n_t1)
        _cq_payload["n_tier2"] = int(n_t2)
        _cq_payload["n_tier3"] = int(n_t3)
        _cq_payload["n_tier4"] = int(n_t4)
        _cq_payload["aperture_correction"] = {
            "ok": (bool(ac_result.get("ok", False)) if isinstance(ac_result, dict) else False),
            "delta_m_corr": (ac_result.get("delta_m_corr") if isinstance(ac_result, dict) else None),
            "scatter_mag": (ac_result.get("scatter_mag") if isinstance(ac_result, dict) else None),
            "n_ref_stars": (int(ac_result.get("n_ref_stars", 0)) if isinstance(ac_result, dict) else 0),
            "ref_star_ids": (ac_result.get("ref_star_ids", []) if isinstance(ac_result, dict) else []),
            "reason": (str(ac_result.get("reason", "disabled")) if isinstance(ac_result, dict) else "disabled"),
        }
        _cq_path.write_text(json.dumps(_cq_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[PHASE 2A] Optional artifact write failed (comp_quality.json): %s", exc)

    lc_png = lc_dir / f"lightcurve_{target_cid}.png"
    if _save_png:
        try:
            save_lightcurve_png(
                lc_png,
                bjd,
                mag_calib,
                err,
                out_flags,
                target_name,
                comp_quality,
                delta_mag_mode=False,
                delta_mag=delta_mag,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("[PHASE 2A] Optional artifact write failed (lightcurve PNG): %s", exc)

    cutout_png = lc_dir / f"cutout_{target_cid}.png"
    if _save_png:
        try:
            save_cutout_png(
                cutout_png,
                Path(masterstar_fits_path),
                float(target_row["x"]),
                float(target_row["y"]),
                target_name,
                ms_data=_ms_data,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("[PHASE 2A] Optional artifact write failed (cutout PNG): %s", exc)

    # Per-target field map s číslovanými comp hviezdami — vždy (UI)
    try:
        _target_comp = _comp_index.get(target_cid, pd.DataFrame()).copy()
        _fm_target_path = lc_dir / f"field_map_{target_cid}.png"
        save_target_field_map_png(
            _fm_target_path,
            Path(masterstar_fits_path),
            target_row,
            _target_comp,
            ms_data=_ms_data,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[PHASE 2A] Optional artifact write failed (field map PNG): %s", exc)

    # Summary riadok
    finite_calib = mag_calib[np.isfinite(mag_calib)]
    n_good_comp = sum(
        1 for q in comp_quality.values() if q.get("quality") in ("good", "suspect")
    )
    n_stability_good = sum(1 for q in comp_quality.values() if q.get("quality") == "good")
    n_stability_suspect = sum(1 for q in comp_quality.values() if q.get("quality") == "suspect")
    n_sat = sum(1 for f in out_flags if f == "saturated")

    _measured_ap = (
        float(_measured_ap_target)
        if math.isfinite(_measured_ap_target) and _measured_ap_target > 0
        else float("nan")
    )
    if not math.isfinite(_measured_ap) and not target_frames.empty and "aperture_r_px" in target_frames.columns:
        _ap_meas = pd.to_numeric(target_frames["aperture_r_px"], errors="coerce").dropna()
        if not _ap_meas.empty:
            _measured_ap = float(np.median(_ap_meas.to_numpy(dtype=float)))
    _lc_rms_full = float(np.std(finite_calib)) if len(finite_calib) > 1 else float("nan")
    _lc_rms_ooe = compute_lc_rms_ooe(mag_calib, out_flags)

    _comp_path = "default"
    _n_tier12 = 0
    if not target_comps.empty:
        if "comp_path" in target_comps.columns:
            _cpaths = target_comps["comp_path"].astype(str).str.strip().str.lower()
            if (_cpaths == "sparse_fallback").any():
                _comp_path = "sparse_fallback"
        if "comp_tier" in target_comps.columns:
            _tiers = pd.to_numeric(target_comps["comp_tier"], errors="coerce")
            _n_tier12 = int(_tiers.isin([1, 2]).sum())

    summary_rows.append(
        {
            "catalog_id": target_cid,
            "vsx_name": target_name,
            "vsx_type": target_vsx_type,
            "zone_flag": str(target_row.get("zone_flag", "")).strip(),
            "n_frames": len(bjd),
            "n_good_comp": n_good_comp,
            "n_tier12": _n_tier12,
            "comp_path": _comp_path,
            "n_stability_good": n_stability_good,
            "n_stability_suspect": n_stability_suspect,
            "n_saturated": n_sat,
            "n_alignment_failed": n_alignment_failed,
            "alignment_failed_frac": alignment_failed_frac,
            "n_wcs_untrusted": n_wcs_untrusted,
            "wcs_untrusted_frac": wcs_untrusted_frac,
            "lc_rms": _lc_rms_full,
            "lc_rms_ooe": _lc_rms_ooe,
            "lc_median_mag": float(np.median(finite_calib)) if len(finite_calib) > 0 else float("nan"),
            "aperture_px": _measured_ap if math.isfinite(_measured_ap) else float(apertures_px.get(target_cid, float("nan"))),
            "aperture_px_planned": float(apertures_px.get(target_cid, float("nan"))),
            "am_slope": float("nan"),
            "am_detrended": False,
            "dilution_factor": float(_dilution_result.get("dilution_factor", 1.0)),
            "dilution_delta_mag": float(_dilution_result.get("dilution_delta_mag", 0.0)),
            "n_neighbors_aperture": int(_dilution_result.get("n_neighbors", 0)),
            "gs11_aperture_arcsec": float(_dilution_result.get("aperture_arcsec", float("nan"))),
            "gs11_dilution_skipped": bool(_dilution_result.get("dilution_skipped", False)),
            "gs11_dilution_skip_reason": str(_dilution_result.get("dilution_skip_reason", "") or ""),
            "mag_median_pre_gs11": _mag_pre_gs11,
            "mag_median_post_gs11": _mag_post_gs11,
            "lc_csv": str(lc_csv),
            "lc_png": str(lc_png),
            "ct_ok": bool(ct_ok),
            "ct_corr": float(ct_corr) if bool(ct_ok) and math.isfinite(float(ct_corr)) else float("nan"),
            "ct_c1": float(c1) if bool(ct_ok) and math.isfinite(float(c1)) else float("nan"),
            "ct_n_comp": int(ct_n_comp) if bool(ct_ok) else 0,
            **_ac_summary_fields(ac_result if bool(_cfg.aperture_correction_enabled) else {"ok": False, "reason": "disabled"}),
        }
    )
    n_lc += 1
    lc_rms = float(summary_rows[-1]["lc_rms"])
    lc_rms_ooe = float(summary_rows[-1].get("lc_rms_ooe", float("nan")))
    r_ap = float(summary_rows[-1]["aperture_px"])
    logging.info(
        f"[FÁZA 2A] {target_name}: "
        f"lc_rms={lc_rms:.4f}, lc_rms_ooe={lc_rms_ooe:.4f}, "
        f"n_comp={n_good_comp} (stability_good={n_stability_good}), "
        f"apertura={r_ap:.2f}px (measured)"
    )


    state.chip_fw = chip_fw
    state.chip_fh = chip_fh
    return summary_rows, n_lc

def _phase2a_finalize_exports(
    *,
    summary_rows: list,
    lc_dir: Path,
    output_dir: Path,
    _cfg: Any,
    n_lc: int,
    n_frames: int,
    at_df: pd.DataFrame,
    field_map_path: Path,
    _comp_index: dict,
    _phase2a_csv_cache: dict,
    masterstar_fits_path: Path,
    comparison_stars_csv: Path,
    per_frame_csv_dir: Path,
    detrended_aligned_dir: Path,
    fwhm_px: float = 3.0,
    _ms_header: object,
    _ms_data: object,
    progress_cb: Any = None,
    lunar_context: dict[str, Any] | None = None,
    plate_scale_arcsec: float = 1.3,
) -> dict[str, Any]:
    def _p2(msg: str) -> None:
        if progress_cb is None:
            return
        try:
            progress_cb(str(msg))
        except UnicodeEncodeError:
            progress_cb(str(msg).encode("ascii", "backslashreplace").decode("ascii"))

    summary_csv, _sum_df = _phase2a_write_summary(
        summary_rows,
        output_dir,
        lunar_context=lunar_context,
        cfg=_cfg,
        plate_scale_arcsec=float(plate_scale_arcsec),
    )

    if summary_rows:
        from collections import Counter

        _ac_applied = sum(1 for r in summary_rows if r.get("ac_applied"))
        _ac_skipped = len(summary_rows) - _ac_applied
        _ac_reasons = Counter(
            str(r.get("ac_skip_reason") or "unknown")
            for r in summary_rows
            if not r.get("ac_applied")
        )
        logging.info(
            "[AC] run summary: applied=%d skipped=%d (%s)",
            _ac_applied,
            _ac_skipped,
            dict(_ac_reasons),
        )
        log_event(
            f"[AC] run summary: applied={_ac_applied} skipped={_ac_skipped} ({dict(_ac_reasons)})"
        )

    # Draft-level comp QA (read-only w.r.t. photometry; before AAVSO/VarAstro export).
    if bool(getattr(_cfg, "comp_qa_enabled", True)):
        try:
            from comp_qa_core import run_comp_qa_for_photometry_dir  # noqa: PLC0415

            _p2("Comp QA (Sokolovsky locus)…")
            run_comp_qa_for_photometry_dir(
                photometry_dir=output_dir,
                proc_dir=per_frame_csv_dir,
                lc_dir=lc_dir,
                update_summary=True,
                min_comps=int(getattr(_cfg, "phase01_comparison_n_comp_min", 3)),
                max_comps=int(getattr(_cfg, "phase01_comparison_n_comp_max", 8)),
            )
        except Exception as exc:  # noqa: BLE001
            logging.error('[EXC-0174] COMP_QA stage failure - comp_quality JSON/summary not updated post Phase 2A: %s', exc)
            logging.warning("[COMP_QA] Stage failed (non-fatal): %s", exc)

    # Draft-level trust flag (read-only w.r.t. photometry; before AAVSO/VarAstro/PDF).
    if bool(getattr(_cfg, "trust_flag_enabled", True)):
        try:
            from trust_flag_core import run_trust_flag_for_photometry_dir  # noqa: PLC0415

            _p2("Trust flag (GREEN/YELLOW/RED)…")
            run_trust_flag_for_photometry_dir(
                photometry_dir=output_dir,
                lc_dir=lc_dir,
                update_summary=True,
                cfg=_cfg,
            )
            _sum_df = pd.read_csv(output_dir / "photometry_summary.csv", low_memory=False)
        except Exception as exc:  # noqa: BLE001
            logging.error('[EXC-0175] Trust-flag stage failure - photometry_summary trust columns not refreshed: %s', exc)
            logging.warning("[TRUST] Stage failed (non-fatal): %s", exc)

    logging.info(f"[FÁZA 2A] Hotovo: {n_lc} svetelných kriviek → {output_dir}")
    logging.info(
        f"[FÁZA 2A] Targety bez comp hviezd: "
        f"{len(at_df) - n_lc}/{len(at_df)} "
        f"(žiadne vhodné comp podľa aktuálnych filtrov)"
    )
    _p2(f"Fáza 2A hotovo: {n_lc} kriviek z {n_frames} snímok → {output_dir.name}")

    # Export lightcurve reports (AAVSO + VAR.ASTRO.CZ) — best effort, non-fatal.
    try:
        from citations import build_run_citation_context, load_pipeline_meta  # noqa: PLC0415
        from export_reports import (  # noqa: PLC0415
            export_all_method_lightcurve_reports,
            log_export_batch_summary,
            record_export_failure,
        )
        from report_methods import active_report_methods, lc_csv_path  # noqa: PLC0415

        reports_dir = output_dir / "lightcurves_reports"
        (reports_dir / "aavso").mkdir(parents=True, exist_ok=True)
        (reports_dir / "varastro").mkdir(parents=True, exist_ok=True)
        _setup_obs_group = str(Path(per_frame_csv_dir).name)
        _run_cite = build_run_citation_context(
            _cfg,
            pipeline_meta=load_pipeline_meta(output_dir),
            targets_df=at_df,
        )
        _have_psf_lc = any(lc_dir.glob("lightcurve_*_psf.csv")) or any(
            lc_dir.glob("lightcurve_*_adaptive.csv")
        )
        _active_methods = active_report_methods(
            _cfg,
            have_psf_cols=_have_psf_lc
            or bool(getattr(_cfg, "psf_photometry_enabled", False))
            or bool(getattr(_cfg, "psf_adaptive_enabled", False)),
        )

        # Build summary lookup by target catalog_id
        _sum_by = {}
        if "catalog_id" in _sum_df.columns:
            for _, r in _sum_df.iterrows():
                cid = str(r.get("catalog_id") or "").strip()
                if cid:
                    _sum_by[cid] = r

        n_export_ok = 0
        n_export_skip = 0
        _export_failures: list[dict[str, str]] = []
        for _, trow in at_df.iterrows():
            target_cid = _normalize_gaia_id(trow.get("catalog_id", ""))
            if not target_cid:
                continue
            lc_csv = lc_csv_path(lc_dir, target_cid, "aperture")
            if not lc_csv.is_file():
                # F-435-EXPORT-GHOSTS: active_targets may include stars that never got an LC
                # (no comps / empty_comp_drop). Do not enqueue as export failure.
                logging.info(
                    "[EXPORT] skip %s aperture: no LC CSV (not a photometry product; "
                    "typically no comps / dropped)",
                    target_cid,
                )
                n_export_skip += 1
                continue
            try:
                pd.read_csv(lc_csv, low_memory=False)
            except Exception as exc:  # noqa: BLE001
                record_export_failure(
                    _export_failures,
                    target_cid,
                    "aperture",
                    f"LC CSV read error: {exc}",
                )
                continue

            comp_target = _comp_index.get(target_cid, pd.DataFrame()).copy()
            srow = _sum_by.get(target_cid, pd.Series(dtype=object))

            _cq_path = lc_dir / f"comp_quality_{target_cid}.json"
            _comp_qmap: dict[str, str] = {}
            if _cq_path.is_file():
                try:
                    _raw_cq = json.loads(_cq_path.read_text(encoding="utf-8"))
                    for _qk, _qv in parse_comp_quality_json_map(_raw_cq).items():
                        _nk = _normalize_gaia_id(_qk)
                        _qv2 = str(_qv.get("quality", "")).strip().lower()
                        if _qv2 == "excluded":
                            continue
                        _comp_qmap[_nk] = _qv2
                except Exception:  # noqa: BLE001
                    _comp_qmap = {}

            try:
                _method_paths = export_all_method_lightcurve_reports(
                    reports_dir,
                    trow,
                    lc_dir=lc_dir,
                    target_cid=target_cid,
                    comp_df=comp_target,
                    summary_row=srow,
                    observer_code=str(_cfg.observer_code or ""),
                    observer_name=str(_cfg.observer_name or "Unknown Observer"),
                    comp_quality_map=_comp_qmap if _comp_qmap else None,
                    cfg=_cfg,
                    obs_group=_setup_obs_group,
                    targets_df=at_df,
                    run_citation_ctx=_run_cite,
                    export_failures=_export_failures,
                )
                if _method_paths:
                    n_export_ok += 1
                else:
                    n_export_skip += 1
            except Exception as exc:  # noqa: BLE001
                record_export_failure(
                    _export_failures,
                    target_cid,
                    "all",
                    f"export batch error: {exc}",
                )
                n_export_skip += 1

        log_export_batch_summary(_export_failures)
        logging.info(
            "[EXPORT] lightcurves_reports: %d targets exported, %d skipped (methods=%s)",
            int(n_export_ok),
            int(n_export_skip),
            ",".join(_active_methods),
        )
    except Exception as exc:  # noqa: BLE001
        logging.error('[EXC-0176] AAVSO/VarAstro lightcurve export batch fails - external report files missing: %s', exc)
        logging.warning("[EXPORT] init failed: %s", exc)

    # Build flux pivot once — reuse in variability detection (TODO-PERF-6)
    _flux_pivot: pd.DataFrame | None = None
    try:
        LOGGER.info("[PHASE 2A] Building flux pivot for variability reuse...")
        _t_pivot = time.perf_counter()
        _long_frames: list[pd.DataFrame] = []
        for _csv_path, _df_cache in _phase2a_csv_cache.items():
            if "dao_flux" not in _df_cache.columns or "catalog_id" not in _df_cache.columns:
                continue
            _meta_cols = [
                "catalog_id",
                "dao_flux",
                "bjd_tdb_mid",
                "mag",
                "bp_rp",
                "b_v",
                "zone",
                "source_type",
                "vsx_known_variable",
                "gaia_dr3_variable_catalog",
                "snr50_ok",
                "edge_safe_10px",
                "photometry_ok",
                "x",
                "y",
                "ra_deg",
                "dec_deg",
            ]
            _subset = _df_cache[[c for c in _meta_cols if c in _df_cache.columns]].copy()
            _subset["_frame"] = Path(_csv_path).stem
            _long_frames.append(_subset)
        if _long_frames:
            _flux_long = pd.concat(_long_frames, ignore_index=True)
            _flux_pivot = _flux_long.pivot_table(
                index="catalog_id",
                columns="_frame",
                values="dao_flux",
                aggfunc="first",
            )
            LOGGER.info(
                "[PHASE 2A] Flux pivot built: %s in %.2fs",
                str(_flux_pivot.shape),
                time.perf_counter() - _t_pivot,
            )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[PHASE 2A] Flux pivot build failed (will use disk): %s", exc)
        _flux_pivot = None

    # Variability detection (pipeline, no UI): auto-export variability_candidates.csv
    try:
        auto_export_variability_candidates_csv(
            masterstar_fits_path=masterstar_fits_path,
            comparison_stars_csv=comparison_stars_csv,
            per_frame_csv_dir=per_frame_csv_dir,
            output_dir=output_dir,
            cfg=_cfg,
            flux_pivot=_flux_pivot,
            csv_cache=_phase2a_csv_cache,
            ms_header=_ms_header,
            ms_data=_ms_data,
            platesolve_dir=Path(masterstar_fits_path).resolve().parent,
        )
    except Exception as exc:  # noqa: BLE001
        logging.error('[EXC-0177] Auto variability-candidates CSV export fails - rms_candidates.csv not produced: %s', exc)
        logging.warning("[VARIABILITY] Auto-export failed: %s", exc)

    # Auto crossmatch + TESS verify for candidates before Summary Measure Report (best effort).
    try:
        from crossmatch_runner import auto_crossmatch_candidates  # noqa: PLC0415
        from tess_runner import auto_tess_verify_candidates  # noqa: PLC0415

        cand_candidates = [
            output_dir / "variability_candidates.csv",  # preferred (auto-export from variability detection)
            output_dir / "suspected_variables.csv",  # fallback legacy
            output_dir / "candidates.csv",
            output_dir / "rms_candidates.csv",
        ]
        candidates_csv = next((p for p in cand_candidates if p.exists()), None)
        if candidates_csv is not None:
            auto_crossmatch_candidates(candidates_csv=candidates_csv, output_dir=output_dir, cfg=_cfg)
            auto_tess_verify_candidates(candidates_csv=candidates_csv, output_dir=output_dir, cfg=_cfg)
        else:
            logging.info("[AUTO] No candidates CSV found for crossmatch/TESS in %s", str(output_dir))
    except Exception as exc:  # noqa: BLE001
        logging.error('[EXC-0178] Auto crossmatch/TESS verify on candidates fails - enrichment columns absent from export: %s', exc)
        logging.warning("[AUTO] crossmatch/TESS zlyhalo: %s", exc)

    from except_fix_counters import get_except_fix_counters

    _ef_snap = get_except_fix_counters().snapshot()
    merge_photometry_pipeline_meta(output_dir, {"except_fix_summary": _ef_snap})
    if any(v > 0 for v in _ef_snap.values()):
        logging.error("[EXCEPT-FIX] Phase 2A terminal-failure counters: %s", _ef_snap)

    return {
        "n_targets": len(at_df),
        "n_frames": n_frames,
        "n_lightcurves": n_lc,
        "summary_csv": str(summary_csv),
        "summary_report_pdf": "",
        "field_map_png": str(field_map_path),
        "output_dir": str(output_dir),
        "lunar_context": lunar_context,
    }

def run_phase2a(
    masterstar_fits_path: Path,
    active_targets_csv: Path,
    comparison_stars_csv: Path,
    per_frame_csv_dir: Path,
    detrended_aligned_dir: Path,
    output_dir: Path,
    fwhm_px: float,
    *,
    annulus_inner_fwhm: float = 4.0,
    annulus_outer_fwhm: float = 6.0,
    aperture_fwhm_factor: float | None = None,
    sat_limit_adu: float | None = None,
    outlier_sigma: float = 3.0,
    stability_sigma: float = 3.0,
    force_aperture_px: float | None = None,
    cfg: AppConfig | None = None,
    progress_cb: Any = None,
    db: Any | None = None,
    draft_id: int | None = None,
    proc_frame_store: ProcFrameStore | None = None,
) -> dict[str, Any]:
    """Hlavný wrapper pre Fázu 2A.

    Globálny FWHM pre apertúru: ``VY_FWHM_GAUSS`` (2D fit z pipeline), inak ``VY_FWHM``
    (DAO, pre apertúru porovnateľné s Gaussian FWHM), inak 2D Gaussian fit
    (``measure_fwhm_from_masterstar``) s nápovedou z ``fwhm_px``.
    Apertúrny polomer = ``aperture_fwhm_factor × FWHM`` (predvolene z ``cfg``).

    Returns:
        dict: n_targets, n_frames, n_lightcurves, summary_csv, field_map_png
    """
    output_dir = Path(output_dir)
    lc_dir = output_dir / "lightcurves"
    lc_dir.mkdir(parents=True, exist_ok=True)

    from except_fix_counters import reset_except_fix_counters

    reset_except_fix_counters()

    def _p2(msg: str) -> None:
        if progress_cb is None:
            return
        try:
            progress_cb(str(msg))
        except UnicodeEncodeError:
            progress_cb(str(msg).encode("ascii", "backslashreplace").decode("ascii"))

    _p2("Initializing shared state…")
    state = _phase2a_prepare_shared_state(
        output_dir=output_dir,
        lc_dir=lc_dir,
        masterstar_fits_path=masterstar_fits_path,
        comparison_stars_csv=comparison_stars_csv,
        per_frame_csv_dir=per_frame_csv_dir,
        progress_cb=progress_cb,
        active_targets_csv=active_targets_csv,
        detrended_aligned_dir=detrended_aligned_dir,
        fwhm_px=fwhm_px,
        annulus_inner_fwhm=annulus_inner_fwhm,
        annulus_outer_fwhm=annulus_outer_fwhm,
        aperture_fwhm_factor=aperture_fwhm_factor,
        sat_limit_adu=sat_limit_adu,
        force_aperture_px=force_aperture_px,
        cfg=cfg,
        db=db,
        draft_id=draft_id,
        proc_frame_store=proc_frame_store,
    )
    at_df = state.at_df
    _comp_index = state._comp_index
    csv_files = state.csv_files
    n_frames = state.n_frames
    _phase2a_csv_cache = state._phase2a_csv_cache
    _phase2a_lookup_cache = state._phase2a_lookup_cache
    fwhm_px = state.fwhm_px
    _ms_header = state._ms_header
    _ms_data = state._ms_data
    _flux_matrix = state._flux_matrix
    _all_lc_ids_list = state._all_lc_ids_list
    field_map_path = state.field_map_path
    _gain_phot = state._gain_phot
    _aligned_dir_2a = state._aligned_dir_2a
    _cfg = state._cfg
    _nt = state._nt
    _save_png = bool(_cfg.save_lightcurve_png)
    if aperture_fwhm_factor is not None:
        try:
            _apt_fw = float(aperture_fwhm_factor)
            if not math.isfinite(_apt_fw) or _apt_fw <= 0:
                _apt_fw = float(_cfg.aperture_fwhm_factor)
            else:
                _apt_fw = max(0.5, min(6.0, _apt_fw))
        except (TypeError, ValueError):
            _apt_fw = float(_cfg.aperture_fwhm_factor)
    else:
        _apt_fw = float(_cfg.aperture_fwhm_factor)

    summary_rows: list[dict[str, Any]] = []
    n_lc = 0
    _ac_sign_logged_ref: list[bool] = [False]

    state.lunar_context = _phase2a_compute_lunar_context(state)
    _dyn = _build_phase2a_dynamic_params(state, output_dir, aperture_fwhm_factor=_apt_fw)
    if _dyn.get("plate_scale_arcsec_px") is None and _cfg is not None:
        try:
            _cfg_ps = float(_cfg.plate_scale_arcsec_per_px)
            if math.isfinite(_cfg_ps) and _cfg_ps > 0:
                _dyn["plate_scale_arcsec_px"] = _cfg_ps
        except (TypeError, ValueError):
            pass
    _cal_meta: dict[str, Any] = {}
    try:
        from draft_provenance import calibration_mode_report_line, resolve_calibration_mode

        _cal_mode = resolve_calibration_mode(draft_id=draft_id, db=db)
        _cal_meta = {
            "calibration_mode": _cal_mode,
            "calibration_report_line": calibration_mode_report_line(_cal_mode),
        }
    except Exception as exc:  # noqa: BLE001
        logging.error('[EXC-0179] calibration_mode not merged into pipeline_meta after Phase 2A: %s', exc)
        pass
    _cal_diag_meta: dict[str, Any] = {}
    try:
        from cal_diag import load_cal_diag_json_for_meta

        _dd = _draft_dir_from_phase2a_paths(output_dir, Path(masterstar_fits_path))
        _cd = load_cal_diag_json_for_meta(_dd)
        if _cd is not None:
            _cal_diag_meta = {"cal_diag": _cd}
    except Exception as exc:  # noqa: BLE001
        logging.error('[EXC-0180] cal_diag block not merged into pipeline_meta: %s', exc)
        pass
    merge_photometry_pipeline_meta(
        output_dir,
        {
            "lunar_context": state.lunar_context,
            "observer_location": _phase2a_observer_location_dict(
                _cfg,
                site=(state.site_lat, state.site_lon, state.site_alt) if state.site_ok else None,
                site_source=state.site_source,
            ),
            "dynamic_params": _dyn,
            "resolved_facts": state.resolved_facts,
            "common_mode_stability_detrend": bool(
                state.stability_run_flags.get("common_mode_detrend_applied")
            ),
            **_cal_meta,
            **_cal_diag_meta,
            **_sky_surface_meta_from_qc(_draft_dir_from_phase2a_paths(output_dir, Path(masterstar_fits_path))),
        },
        _cfg,
        entry_point="run_phase2a",
    )

    # Per target loop
    # _phase2a_process_single_target (inline): ZP → CT → (outlier → airmass | airmass → outlier) → export.
    for ti, (_, target_row) in enumerate(at_df.iterrows(), start=1):
        summary_rows, n_lc = _phase2a_process_one_target(
            target_row=target_row,
            ti=ti,
            state=state,
            summary_rows=summary_rows,
            n_lc=n_lc,
            lc_dir=lc_dir,
            output_dir=output_dir,
            progress_cb=progress_cb,
            masterstar_fits_path=masterstar_fits_path,
            annulus_inner_fwhm=annulus_inner_fwhm,
            annulus_outer_fwhm=annulus_outer_fwhm,
            outlier_sigma=outlier_sigma,
            stability_sigma=stability_sigma,
            _apt_fw=_apt_fw,
            _save_png=_save_png,
            ac_sign_logged=_ac_sign_logged_ref,
        )

    if not _flux_matrix.empty:
        logging.info(
            "[PERF-8] Per-target frame loops eliminated: %d targets × %d frames = %d calls saved",
            _nt,
            len(csv_files),
            _nt * len(csv_files),
        )

    return _phase2a_finalize_exports(
        summary_rows=summary_rows,
        lc_dir=lc_dir,
        output_dir=output_dir,
        _cfg=_cfg,
        n_lc=n_lc,
        n_frames=n_frames,
        at_df=at_df,
        field_map_path=field_map_path,
        _comp_index=_comp_index,
        _phase2a_csv_cache=_phase2a_csv_cache,
        masterstar_fits_path=masterstar_fits_path,
        comparison_stars_csv=comparison_stars_csv,
        per_frame_csv_dir=per_frame_csv_dir,
        detrended_aligned_dir=state._aligned_dir_2a,
        fwhm_px=float(state.fwhm_px),
        _ms_header=_ms_header,
        _ms_data=_ms_data,
        progress_cb=progress_cb,
        lunar_context=state.lunar_context,
        plate_scale_arcsec=float(state.plate_scale_arcsec),
    )




# ======================================================================
# photometry.py (zlúčené do photometry_core)
# ======================================================================

from utils import (
    fits_binning_xy_from_header,
    plate_scale_arcsec_per_pixel,
    plate_solve_fov_deg_diagonal_from_scale,
)


def _get_plate_scale_from_cfg(
    cfg: Any,
    db: Any = None,
    draft_id: int | None = None,
    *,
    fits_path: Path | None = None,
    ms_header: Any | None = None,
) -> float | None:
    """
    Plate scale (arcsec/px) for FOV / max_dist_deg.
    Priority:
    1. Solved WCS/CD matrix from the FITS (authoritative)
    2. DB EQUIPMENTS+TELESCOPE+SCANNING (pixel/focal/binning)
    3. cfg.phase01_plate_scale_arcsec_per_px (last resort)
    4. plate_scale_arcsec_per_pixel(cfg); None if unavailable
    """
    result: float | None = None

    # 1. Authoritative: solved WCS/CD from the frame's FITS.
    if fits_path is not None or ms_header is not None:
        try:
            _fp = Path(fits_path) if fits_path is not None else Path(".")
            _wcs_ps = _read_plate_scale_from_fits_path(_fp, ms_header=ms_header)
        except Exception:  # noqa: BLE001
            _wcs_ps = None
        if _wcs_ps is not None and math.isfinite(float(_wcs_ps)) and float(_wcs_ps) > 0:
            logging.info(
                "[FOV] _get_plate_scale_from_cfg → %.4f arcsec/px (solved WCS/CD)",
                float(_wcs_ps),
            )
            return float(_wcs_ps)

    # 2. DB: derive plate scale from EQUIPMENTS + TELESCOPE + SCANNING binning (if available).
    if db is not None and draft_id is not None:
        try:
            did = int(draft_id)
        except Exception:  # noqa: BLE001
            did = 0
        if did > 0:
            try:
                dr = None
                try:
                    dr = db.fetch_obs_draft_by_id(did) if hasattr(db, "fetch_obs_draft_by_id") else None
                except Exception:  # noqa: BLE001
                    dr = None
                id_eq = int(dr.get("ID_EQUIPMENTS") or 0) if isinstance(dr, dict) else 0
                id_tel = int(dr.get("ID_TELESCOPE") or 0) if isinstance(dr, dict) else 0

                binning = 1
                try:
                    cur = db.conn.execute(
                        """
                        SELECT ID_SCANNING
                        FROM OBS_FILES
                        WHERE DRAFT_ID = ?
                          AND LOWER(COALESCE(IMAGETYP,'')) = 'light'
                          AND ID_SCANNING IS NOT NULL
                        ORDER BY FILE_PATH
                        LIMIT 1;
                        """,
                        (did,),
                    )
                    r0 = cur.fetchone()
                    sid = int(r0["ID_SCANNING"]) if r0 and r0["ID_SCANNING"] is not None else 0
                    if sid > 0:
                        cur2 = db.conn.execute("SELECT BINNING FROM SCANNING WHERE ID = ? LIMIT 1;", (sid,))
                        r2 = cur2.fetchone()
                        if r2 and r2["BINNING"] is not None:
                            b0 = int(r2["BINNING"])
                            if 1 <= b0 <= 16:
                                binning = b0
                except Exception:  # noqa: BLE001
                    # EXC-0181: T4 -- DB plate-scale lookup failure falls through to config phase01_plate_scale_arcsec_per_px (EXCEPT-BULK-2 2026-07-08)
                    binning = 1

                pix_um = None
                foc_mm = None
                try:
                    pix_um = (
                        float(db.get_equipment_pixel_size_um(id_eq))
                        if (hasattr(db, "get_equipment_pixel_size_um") and id_eq > 0)
                        else None
                    )
                except Exception:  # noqa: BLE001
                    pix_um = None
                try:
                    foc_mm = (
                        float(db.get_equipment_focal_mm(id_eq))
                        if (hasattr(db, "get_equipment_focal_mm") and id_eq > 0)
                        else None
                    )
                except Exception:  # noqa: BLE001
                    foc_mm = None
                if foc_mm is None or not (math.isfinite(float(foc_mm)) and float(foc_mm) > 0):
                    try:
                        foc_mm = (
                            float(db.get_telescope_focal_mm(id_tel if id_tel > 0 else None))
                            if hasattr(db, "get_telescope_focal_mm")
                            else None
                        )
                    except Exception:  # noqa: BLE001
                        foc_mm = None

                if (
                    pix_um is not None
                    and foc_mm is not None
                    and math.isfinite(float(pix_um))
                    and float(pix_um) > 0
                    and math.isfinite(float(foc_mm))
                    and float(foc_mm) > 0
                ):
                    eff_um = float(pix_um) * float(max(1, int(binning)))
                    sc = plate_scale_arcsec_per_pixel(pixel_pitch_um=float(eff_um), focal_length_mm=float(foc_mm))
                    if sc is not None and math.isfinite(float(sc)) and float(sc) > 0:
                        result = float(sc)
                        logging.info(
                            "[FOV] _get_plate_scale_from_cfg → %.4f arcsec/px (DB: eq/tel/bin=%s)",
                            float(result),
                            int(binning),
                        )
                        return result
            except Exception:  # noqa: BLE001
                pass
    # 3. Config phase01_plate_scale_arcsec_per_px (last resort).
    try:
        val = float(cfg.phase01_plate_scale_arcsec_per_px)
        if val > 0:
            result = val
            logging.warning(
                "[FOV] _get_plate_scale_from_cfg → %.4f arcsec/px (config last-resort; no WCS/DB)",
                float(result),
            )
            return result
    except (TypeError, ValueError):
        pass

    try:
        val = plate_scale_arcsec_per_pixel(cfg)
        if val and float(val) > 0:
            result = float(val)
            logging.info(
                "[FOV] _get_plate_scale_from_cfg → %.4f arcsec/px (None = fallback na max_dist_deg)",
                float(result),
            )
            return result
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("[PHASE 2A] Plate scale from config failed (non-critical): %s", exc)
    logging.info(
        "[FOV] _get_plate_scale_from_cfg → %.4f arcsec/px (None = fallback na max_dist_deg)",
        -1.0,
    )
    return None


def _compute_fov_max_dist(
    frame_w_px: int,
    frame_h_px: int,
    plate_scale: float | None,
    fov_fraction: float,
    fallback_deg: float,
) -> float:
    """
    max_dist_deg = (FOV_diagonal / 2) * fov_fraction

    Použi plate_solve_fov_deg_diagonal_from_scale z utils.
    Ak plate_scale je None → vráť fallback_deg.
    """
    logging.info(
        "[FOV] compute: w=%d h=%d scale=%s fraction=%.2f fallback=%.3f",
        int(frame_w_px),
        int(frame_h_px),
        (f"{float(plate_scale):.4f}" if plate_scale else "None"),
        float(fov_fraction),
        float(fallback_deg),
    )
    if not plate_scale or float(plate_scale) <= 0:
        logging.debug(
            "[FÁZA 0+1] plate_scale neznámy → max_dist fallback=%.3f°",
            float(fallback_deg),
        )
        return float(fallback_deg)
    try:
        diag_deg = plate_solve_fov_deg_diagonal_from_scale(
            int(frame_w_px), int(frame_h_px), float(plate_scale)
        )
        if diag_deg is None or not math.isfinite(float(diag_deg)) or float(diag_deg) <= 0:
            raise ValueError(f"invalid diag_deg={diag_deg!r}")
        result = (float(diag_deg) / 2.0) * float(fov_fraction)
        logging.info(
            "[FÁZA 0+1] FOV max_dist: scale=%.3f\"/px, "
            "diag=%.3f°, fraction=%.2f → max_dist=%.3f°",
            float(plate_scale),
            float(diag_deg),
            float(fov_fraction),
            float(result),
        )
        return float(result)
    except Exception as exc:  # noqa: BLE001
        logging.error('[EXC-0182] FOV max_dist degree calculation fails - comp/target cone uses hardcoded fallback radius: %s', exc)
        logging.warning(
            "[FÁZA 0+1] FOV max_dist výpočet zlyhal (%s) → fallback=%.3f°",
            exc,
            float(fallback_deg),
        )
        return float(fallback_deg)


def _resolve_plate_scale_arcsec_per_px(
    cfg: Any,
    fits_path: Path | None = None,
    *,
    ms_header: Any | None = None,
) -> float | None:
    """Plate scale (arcsec/px) for GS11 + aperture arcsec conversion.

    Priority: (1) solved WCS/CD matrix from the FITS; (2) config
    ``phase01_plate_scale_arcsec_per_px`` (last resort with warning).
    Returns None when nothing derivable (derive-or-None; no magic default).
    Clamp [0.1, 30.0] when a value is returned.
    """
    _lo, _hi = 0.1, 30.0
    _fits_ps: float | None = None
    if fits_path is not None or ms_header is not None:
        try:
            _fp = Path(fits_path) if fits_path is not None else Path(".")
            _fits_ps = _read_plate_scale_from_fits_path(_fp, ms_header=ms_header)
        except Exception:  # noqa: BLE001
            # EXC-0183: T4 -- WCS pixel_scale_from_wcs path fails - CD-matrix fallback attempted next (EXCEPT-BULK 2026-07-08)
            _fits_ps = None
    if _fits_ps is not None and math.isfinite(_fits_ps) and _lo <= float(_fits_ps) <= _hi:
        return float(_fits_ps)
    # Config — last resort only (no usable WCS/CD in the FITS).
    try:
        cfg_ps = float(cfg.phase01_plate_scale_arcsec_per_px)
    except (TypeError, ValueError):
        cfg_ps = 0.0
    if math.isfinite(cfg_ps) and _lo <= cfg_ps <= _hi:
        logging.warning(
            "[PLATE SCALE] no usable WCS/CD scale — falling back to config %.3f arcsec/px",
            float(cfg_ps),
        )
        return float(cfg_ps)
    logging.warning(
        "[PLATE SCALE] plate scale not derivable (WCS/CD + config exhausted) — returning None"
    )
    return None


def _cd_matrix_scale_arcsec_per_px(hdr: Any) -> float | None:
    """Plate scale (arcsec/px) from the SOLVED astrometric WCS (CD/PC matrix → CDELT).

    This is the authoritative source: it reflects the actual sky-to-pixel solution,
    independent of stale VY_PLTS / config values. Returns None if no usable WCS.
    """
    if hdr is None:
        return None
    # Full WCS handles CD, PC+CDELT, SIP, etc.
    try:
        import warnings  # noqa: PLC0415

        import numpy as _np  # noqa: PLC0415
        from astropy.wcs import WCS as _WCS  # noqa: PLC0415
        from astropy.wcs import FITSFixedWarning as _FFW  # noqa: PLC0415
        from astropy.wcs.utils import proj_plane_pixel_scales as _pps  # noqa: PLC0415

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", _FFW)
            _w = _WCS(hdr)
        if _w.has_celestial:
            sc = float(_np.mean(_pps(_w))) * 3600.0
            if math.isfinite(sc) and 0.01 < sc < 200.0:
                return float(sc)
    except Exception:  # noqa: BLE001
        pass
    # Raw CD matrix fallback.
    try:
        cd11 = hdr.get("CD1_1")
        cd12 = hdr.get("CD1_2", 0.0)
        if cd11 is not None:
            sc = math.sqrt(float(cd11) ** 2 + float(cd12) ** 2) * 3600.0
            if math.isfinite(sc) and 0.01 < sc < 200.0:
                return float(sc)
    except (TypeError, ValueError):
        pass
    # CDELT1 fallback.
    try:
        cdelt1 = hdr.get("CDELT1")
        if cdelt1 is not None:
            sc = abs(float(cdelt1)) * 3600.0
            if math.isfinite(sc) and 0.01 < sc < 200.0:
                return float(sc)
    except (TypeError, ValueError):
        pass
    return None


def _read_plate_scale_from_fits_path(
    fits_path: Path,
    *,
    ms_header: Any | None = None,
) -> float | None:
    """Plate scale (arcsec/px) from FITS, CD/WCS-FIRST.

    Priority: (1) solved WCS/CD matrix; (2) VY_PLTS header — only if it agrees with
    the CD value within 5% (else ignored, logged); (3) other header keywords, only
    when no usable CD/WCS exists. Clamp [0.1, 30.0] (covers fine ~0.3 and wide-field ~10).
    """
    _MIN, _MAX = 0.1, 30.0
    try:
        from astropy.io import fits as astrofits  # noqa: PLC0415

        if ms_header is not None:
            hdr = ms_header
        else:
            fp = Path(fits_path)
            if not fp.is_file():
                return None
            with astrofits.open(fp, memmap=False) as hdul:
                hdr = hdul[0].header
    except Exception as exc:  # noqa: BLE001
        # EXC-0185: T4 -- Header keyword plate-scale scan fails - returns None after trying CD/WCS paths (EXCEPT-BULK-2 2026-07-08)
        logging.error('[EXC-0184] FITS open/header read for plate scale fails - returns None, caller uses config/default ...: %s', exc)
        return None

    # (1) Authoritative: solved WCS / CD matrix.
    cd_scale = _cd_matrix_scale_arcsec_per_px(hdr)
    if cd_scale is not None and _MIN <= cd_scale <= _MAX:
        # (2) Cross-check VY_PLTS: warn and ignore if it disagrees > 5%.
        vy = hdr.get("VY_PLTS")
        if vy is None:
            vy = hdr.get("VY_PLATESCALE")
        try:
            vyf = float(vy) if vy is not None else None
        except (TypeError, ValueError):
            vyf = None
        if vyf is not None and vyf > 0 and abs(vyf - cd_scale) / cd_scale > 0.05:
            logging.warning(
                "[PLATE SCALE] VY_PLTS=%.3f disagrees with CD-derived %.3f arcsec/px (>5%%) — using CD.",
                vyf,
                cd_scale,
            )
        return float(cd_scale)

    # (3) No usable WCS/CD — fall back to header keywords (still header, above config).
    try:
        for key in ("VY_PLTS", "VY_PLATESCALE", "PIXSCALE", "SECPIX", "SECPIX1", "SCALE", "CDELT1"):
            v = hdr.get(key)
            if v is None:
                continue
            try:
                f = float(v)
                if key == "CDELT1":
                    f = abs(f) * 3600.0
            except (TypeError, ValueError):
                continue
            if math.isfinite(f) and _MIN <= f <= _MAX:
                return float(f)
    except Exception:  # noqa: BLE001
        # EXC-0186: T4 -- Non-numeric catalog_id string returned unchanged instead of int-normalized form (EXCEPT-BULK 2026-07-08)
        return None
    return None

# Zlúčený modul: jedna Gaia/katalóg ID normalizácia (alias pre legacy kód v tomto súbore)
_normalize_id_value = _normalize_gaia_id  # noqa: E402

# Stĺpce načítavané z per-frame CSV pre bootstrap (78 % úspora pamäte)
_PHASE_USECOLS_PERFRAME: list[str] = [
    "name",
    "catalog_id",
    "bjd_tdb_mid",
    "flux",
    "dao_flux",
    "noise_floor_adu",
    "sky_adu_per_px_annulus",
    "aperture_r_px",
    "is_usable",
    "is_saturated",
    "is_noisy",
    "snr50_ok",
    "vsx_known_variable",
    "likely_saturated",
]


def _angular_distance_deg(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """Uhlová vzdialenosť v stupňoch (haversine)."""
    r1, d1, r2, d2 = map(math.radians, [ra1, dec1, ra2, dec2])
    a = (
        math.sin((d2 - d1) / 2) ** 2
        + math.cos(d1) * math.cos(d2) * math.sin((r2 - r1) / 2) ** 2
    )
    return math.degrees(2 * math.asin(min(1.0, math.sqrt(a))))


def _normalize_id_value(x: Any) -> str:
    """Normalize Gaia-like IDs loaded as floats; keep non-numeric strings."""
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return ""
    try:
        return str(int(float(s)))
    except Exception:  # noqa: BLE001
        return s


def _normalize_id_series(s: pd.Series) -> pd.Series:
    return s.apply(_normalize_id_value)


def _bool_col(series: pd.Series) -> pd.Series:
    """Normalizuje stĺpec na bool bez ohľadu na True/False/'true'/'false'/1/0."""
    return series.astype(str).str.strip().str.lower().isin(("true", "1", "yes", "y"))


@dataclass(frozen=True)
class StressTestResult:
    per_source_rms: dict[str, float]
    frames_sampled: int
    frames_used: int


def stress_test_relative_rms_from_sidecars(
    *,
    frames_root: Path,
    source_ids: list[str],
    sample_frac: float = 0.10,
    seed: int = 42,
    flux_col: str = "flux",
    name_col: str = "name",
    min_stars_per_frame: int = 3,
) -> StressTestResult:
    """Compute relative RMS for many sources on a random frame sample.

    For each sampled frame with a sidecar CSV, compute per-frame ensemble median among present sources,
    then record relative flux for each star: f_i / median(f_all). Returns RMS over time for each star.
    """
    root = Path(frames_root)
    files = [
        fp
        for fp in _iter_fits_recursive(root)
        if proc_csv_path_for_aligned_fits(fp).is_file()
    ]
    if not files or not source_ids:
        return StressTestResult(per_source_rms={}, frames_sampled=0, frames_used=0)

    frac = float(sample_frac)
    frac = 0.10 if not math.isfinite(frac) else max(0.01, min(1.0, frac))
    k = max(1, int(round(len(files) * frac)))
    rnd = random.Random(int(seed))
    sample = rnd.sample(files, k=min(k, len(files)))

    want = [str(x).strip() for x in source_ids if str(x).strip()]
    want_set = set(want)
    rel_lists: dict[str, list[float]] = {nm: [] for nm in want}

    frames_used = 0
    _sidecar_cache: dict[str, pd.DataFrame] = {}
    for fp in sample:
        sidecar = proc_csv_path_for_aligned_fits(fp)
        _sidecar_key = str(sidecar)
        if _sidecar_key not in _sidecar_cache:
            if Path(sidecar).is_file():
                try:
                    # Sidecar per-frame catalogs often carry Gaia IDs; preserve as strings when present.
                    _sidecar_cache[_sidecar_key] = pd.read_csv(
                        sidecar, low_memory=False, dtype=_GAIA_ID_DTYPE
                    )
                except Exception as exc:  # noqa: BLE001
                    # EXC-0187: T4 -- astroquery/Vizier import failure returns empty VSX-neighbor set - VSX comp exclusion sk... (EXCEPT-BULK-2 2026-07-08)
                    LOGGER.debug("[CSV] Skipping row due to parse error: %s", exc)
                    _sidecar_cache[_sidecar_key] = pd.DataFrame()
            else:
                _sidecar_cache[_sidecar_key] = pd.DataFrame()
        dff = _sidecar_cache[_sidecar_key]
        if dff.empty:
            continue
        if name_col not in dff.columns or flux_col not in dff.columns:
            continue
        names = dff[name_col].astype(str).str.strip()
        flux = pd.to_numeric(dff[flux_col], errors="coerce")
        mask = names.isin(want_set) & flux.notna() & (flux.astype(float) > 0)
        if not bool(mask.any()):
            continue
        sub = dff.loc[mask, [name_col, flux_col]].copy()
        sub[name_col] = sub[name_col].astype(str).str.strip()
        sub[flux_col] = pd.to_numeric(sub[flux_col], errors="coerce").astype(float)
        sub = sub.dropna()
        if len(sub) < int(min_stars_per_frame):
            continue
        med = float(sub[flux_col].median())
        if not math.isfinite(med) or med <= 0:
            continue
        frames_used += 1
        for _, row in sub.iterrows():
            nm = str(row[name_col]).strip()
            if nm in rel_lists:
                rel_lists[nm].append(float(row[flux_col]) / med)

    out: dict[str, float] = {}
    for nm, arr in rel_lists.items():
        if len(arr) < 3:
            continue
        mu = 1.0
        rms = math.sqrt(sum((x - mu) ** 2 for x in arr) / float(len(arr)))
        if math.isfinite(rms):
            out[nm] = float(rms)
    return StressTestResult(per_source_rms=out, frames_sampled=int(len(sample)), frames_used=int(frames_used))


def vsx_is_known_variable_top3_per_bin(
    *,
    rows: list[dict[str, Any]],
    phot_category_key: str = "phot_category",
    rms_key: str = "stress_rms",
    ra_key: str = "ra",
    dec_key: str = "dec",
    max_per_bin: int = 3,
    radius_arcsec: float = 2.0,
) -> set[str]:
    """Return set of Gaia source_id strings that are present in VSX near the best (lowest RMS) stars per bin."""
    try:
        from astroquery.vizier import Vizier  # type: ignore
        import astropy.units as u
        from astropy.coordinates import SkyCoord
    except Exception:  # noqa: BLE001
        # EXC-0188: T4 -- numpy/fits import failure returns None intersection bbox - alignment crop not applied (EXCEPT-BULK-2 2026-07-08)
        return set()

    by_bin: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        b = str(r.get(phot_category_key) or "").strip()
        sid = str(r.get("source_id_gaia") or "").strip()
        if not b or not sid:
            continue
        v = r.get(rms_key)
        try:
            rms = float(v)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(rms):
            continue
        by_bin.setdefault(b, []).append(r)

    viz = Vizier(row_limit=50)
    flagged: set[str] = set()
    for _, items in by_bin.items():
        items_sorted = sorted(items, key=lambda x: float(x.get(rms_key)))
        for r in items_sorted[: int(max_per_bin)]:
            sid = str(r.get("source_id_gaia") or "").strip()
            try:
                ra = float(r.get(ra_key))
                de = float(r.get(dec_key))
            except (TypeError, ValueError):
                continue
            if not (math.isfinite(ra) and math.isfinite(de)):
                continue
            c = SkyCoord(ra=ra * u.deg, dec=de * u.deg, frame="icrs")
            try:
                t = viz.query_region(c, radius=float(radius_arcsec) * u.arcsec, catalog="B/vsx")
            except Exception as exc:  # noqa: BLE001
                # EXC-0189: T4 -- One aligned frame skipped in common-field bbox - intersection computed from remaining f... (EXCEPT-BULK-2 2026-07-08)
                LOGGER.debug("[CSV] Skipping row due to parse error: %s", exc)
                continue
            if t and len(t) > 0 and len(t[0]) > 0:
                flagged.add(sid)
    return flagged


def common_field_intersection_bbox_px(
    *,
    frame_paths: list[Path],
    finite_stride: int = 16,
) -> tuple[float, float, float, float] | None:
    """Compute intersection bbox of finite pixels across frames (x0,y0,x1,y1).

    Intended for WCS-reprojected aligned frames where uncovered regions are NaN.
    Uses strided sampling for speed.
    """
    try:
        import numpy as np
        from astropy.io import fits
    except Exception:  # noqa: BLE001
        return None

    fps = [Path(p) for p in frame_paths if Path(p).is_file()]
    if len(fps) < 2:
        return None

    x0_i, y0_i = 0.0, 0.0
    x1_i, y1_i = float("inf"), float("inf")
    stride = max(1, int(finite_stride))

    for fp in fps:
        try:
            with fits.open(fp, memmap=False) as hdul:
                data = np.asarray(hdul[0].data, dtype=np.float32)
        except Exception:  # noqa: BLE001
            continue
        if data.ndim != 2:
            continue
        samp = data[::stride, ::stride]
        fin = np.isfinite(samp)
        if not bool(fin.any()):
            continue
        ys, xs = np.where(fin)
        x0 = float(xs.min() * stride)
        y0 = float(ys.min() * stride)
        x1 = float(min(data.shape[1] - 1, xs.max() * stride + (stride - 1)))
        y1 = float(min(data.shape[0] - 1, ys.max() * stride + (stride - 1)))
        x0_i = max(x0_i, x0)
        y0_i = max(y0_i, y0)
        x1_i = min(x1_i, x1)
        y1_i = min(y1_i, y1)

    if not (math.isfinite(x0_i) and math.isfinite(y0_i) and math.isfinite(x1_i) and math.isfinite(y1_i)):
        return None
    if x1_i <= x0_i or y1_i <= y0_i:
        return None
    return (x0_i, y0_i, x1_i, y1_i)


def recommended_aperture_by_color(
    *,
    bp_rp: float | None,
    median_fwhm_blue: float | None,
    median_fwhm_neutral: float | None,
    median_fwhm_red: float | None,
) -> float | None:
    """Return 2.5× median FWHM for the star's coarse color category."""
    if bp_rp is None:
        return None
    try:
        c = float(bp_rp)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(c):
        return None
    if c < 0.5:
        f = median_fwhm_blue
    elif c <= 1.5:
        f = median_fwhm_neutral
    else:
        f = median_fwhm_red
    if f is None:
        return None
    try:
        fv = float(f)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(fv) or fv <= 0:
        return None
    return 2.5 * fv



def bad_columns_for_light_frame(
    bpm: dict[str, Any] | None,
    *,
    light_header: Any,
) -> set[int]:
    """Map native ``bad_x`` from BPM JSON to 0-based integer column indices in the light frame."""
    if not bpm or not isinstance(bpm, dict):
        return set()
    raw = bpm.get("bad_x")
    if not raw:
        return set()
    try:
        lb_x, _ = fits_binning_xy_from_header(light_header)
    except Exception:  # noqa: BLE001
        lb_x = 1
    lb_x = max(1, int(lb_x))
    mb = int(bpm.get("native_binning") or 1)
    mb = max(1, mb)
    factor = max(1, lb_x // mb)
    out: set[int] = set()
    for x in raw:
        try:
            xi = int(x)
        except (TypeError, ValueError):
            continue
        out.add(int(xi // factor))
    return out


def _fwhm_moment_at(arr: np.ndarray, xc: float, yc: float, *, half: int = 6) -> float:
    """2D Gaussian moment FWHM estimate (same recipe as pipeline MASTERSTAR block)."""
    if not (math.isfinite(xc) and math.isfinite(yc)):
        return float("nan")
    xi = int(round(float(xc)))
    yi = int(round(float(yc)))
    h, w = int(arr.shape[0]), int(arr.shape[1])
    x0 = max(0, xi - half)
    x1 = min(w - 1, xi + half)
    y0 = max(0, yi - half)
    y1 = min(h - 1, yi + half)
    if x1 <= x0 or y1 <= y0:
        return float("nan")
    patch = arr[y0 : y1 + 1, x0 : x1 + 1].astype(np.float64, copy=False)
    if patch.size < 9:
        return float("nan")
    medp = float(np.nanmedian(patch))
    wgt = patch - medp
    wgt[~np.isfinite(wgt)] = 0.0
    wgt[wgt < 0] = 0.0
    s = float(wgt.sum())
    if not math.isfinite(s) or s <= 0:
        return float("nan")
    yy, xx = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
    mx = float((wgt * xx).sum() / s)
    my = float((wgt * yy).sum() / s)
    vx = float((wgt * (xx - mx) ** 2).sum() / s)
    vy = float((wgt * (yy - my) ** 2).sum() / s)
    if not (vx > 0 and vy > 0 and math.isfinite(vx) and math.isfinite(vy)):
        return float("nan")
    sigx = math.sqrt(vx)
    sigy = math.sqrt(vy)
    fwhm = 2.355 * 0.5 * (sigx + sigy)
    return float(fwhm) if math.isfinite(fwhm) else float("nan")


def compute_auto_fwhm_limit(
    fwhm_values: np.ndarray | Sequence[float],
    k: float = 1.5,
) -> dict[str, Any]:
    """
    Vypočíta automatický FWHM limit pomocou MAD štatistiky.

    Vracia dict:
        median_fwhm, mad, sigma_mad, auto_limit, k, n_total, n_kept, n_cut
    (``auto_limit`` môže byť ``None`` pri príliš málo bodoch.)
    """
    arr = np.asarray(fwhm_values, dtype=np.float64)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if len(arr) < 3:
        return {
            "median_fwhm": None,
            "mad": None,
            "sigma_mad": None,
            "auto_limit": None,
            "k": float(k),
            "n_total": int(len(arr)),
            "n_kept": int(len(arr)),
            "n_cut": 0,
        }
    median_f = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median_f)))
    sigma_mad = mad * 1.4826
    auto_limit = median_f + float(k) * sigma_mad
    n_kept = int(np.sum(arr <= auto_limit))
    return {
        "median_fwhm": round(median_f, 3),
        "mad": round(mad, 4),
        "sigma_mad": round(sigma_mad, 4),
        "auto_limit": round(float(auto_limit), 3),
        "k": float(k),
        "n_total": int(len(arr)),
        "n_kept": n_kept,
        "n_cut": int(len(arr) - n_kept),
    }


def compute_fwhm_gaussian_for_aperture_catalog(
    df: pd.DataFrame,
    data: np.ndarray,
    hdr: Any,
    *,
    gaussian_fwhm_px_override: float | None,
    aperture_fwhm_factor: float,
) -> tuple[np.ndarray, float, float]:
    """Vráti (fwhm_per_row, fwhm_moment_med, fwhm_gaussian) — rovnaký výpočet ako v ``enhance_catalog_dataframe_aperture_bpm``.

    Používa sa v ``pipeline._apply_aperture_catalog_enhancements_from_st`` pre multi-apertúru (r_small / r_large),
    aby polomery zodpovedali hlavnej apertúre.
    """
    arr = np.asarray(data, dtype=np.float32)
    x = pd.to_numeric(df.get("x"), errors="coerce").to_numpy(dtype=np.float64)
    y = pd.to_numeric(df.get("y"), errors="coerce").to_numpy(dtype=np.float64)
    n = len(df)
    if n == 0:
        return np.array([], dtype=np.float64), float("nan"), float("nan")

    fwhm_per = np.array(
        [_fwhm_moment_at(arr, float(x[i]), float(y[i])) for i in range(n)],
        dtype=np.float64,
    )

    fwhm_moment_med = float(np.nanmedian(fwhm_per[np.isfinite(fwhm_per) & (fwhm_per > 0)]))
    if not math.isfinite(fwhm_moment_med) or fwhm_moment_med <= 0:
        fwhm_moment_med = float("nan")

    DAO_TO_GAUSSIAN = 1.0 / 1.5  # 0.667 — fyzikálne odvodené, setup-nezávislé
    fwhm_gaussian: float | None = None

    if gaussian_fwhm_px_override is not None:
        try:
            _ov = float(gaussian_fwhm_px_override)
            if math.isfinite(_ov) and 0.5 < _ov < 30.0:
                fwhm_gaussian = _ov
        except (TypeError, ValueError):
            pass

    if fwhm_gaussian is None and hdr is not None:
        try:
            _vy = hdr.get("VY_FWHM", None)
            if _vy is not None:
                _vy_f = float(_vy)
                if math.isfinite(_vy_f) and 0.5 < _vy_f < 30.0:
                    fwhm_gaussian = _vy_f * DAO_TO_GAUSSIAN
                    if not bool(getattr(enhance_catalog_dataframe_aperture_bpm, "_did_log_fwhm", False)):
                        logging.info(
                            f"[PHOT] FWHM z VY_FWHM (DAO): {_vy_f:.3f}px × {DAO_TO_GAUSSIAN:.3f} = "
                            f"{float(fwhm_gaussian):.3f}px → apertura = "
                            f"{float(fwhm_gaussian) * float(aperture_fwhm_factor):.3f}px"
                        )
                        enhance_catalog_dataframe_aperture_bpm._did_log_fwhm = True
        except (TypeError, ValueError):
            pass

    if fwhm_gaussian is None:
        if math.isfinite(fwhm_moment_med) and fwhm_moment_med > 0:
            fwhm_gaussian = fwhm_moment_med * 0.619
            if not bool(getattr(enhance_catalog_dataframe_aperture_bpm, "_did_log_fwhm", False)):
                logging.info(
                    f"[PHOT] FWHM fallback moment×0.619: {fwhm_gaussian:.3f}px → "
                    f"apertura = {float(fwhm_gaussian) * float(aperture_fwhm_factor):.3f}px"
                )
                enhance_catalog_dataframe_aperture_bpm._did_log_fwhm = True
        else:
            fwhm_gaussian = float("nan")

    r_ap_test = float(aperture_fwhm_factor) * float(fwhm_gaussian) if math.isfinite(float(fwhm_gaussian)) else float("nan")
    if not math.isfinite(r_ap_test) or r_ap_test < 3.0 or r_ap_test > 20.0:
        fwhm_gaussian = float(fwhm_moment_med)
        logging.warning(
            f"[PHOT] Gaussian FWHM fallback na moment: {fwhm_gaussian:.2f}px "
            f"(r_ap={r_ap_test:.2f}px mimo rozsahu)"
        )

    return fwhm_per, fwhm_moment_med, float(fwhm_gaussian) if math.isfinite(float(fwhm_gaussian)) else float("nan")


def _sky_pp_from_annulus_image(d: np.ndarray, ann_img: np.ndarray) -> float:
    """Local sky (ADU/px) from annulus mask image — matches batch annulus logic."""
    sky_pixels = d[ann_img > 0]
    if sky_pixels.size >= 5:
        sky_med = float(np.median(sky_pixels))
        sky_std = float(np.std(sky_pixels))
        clipped = sky_pixels[sky_pixels < sky_med + 2.0 * sky_std]
        if clipped.size >= 5:
            return float(np.median(clipped))
        return sky_med
    return float(np.median(d))


def _aperture_flux_sky_per_star(
    d: np.ndarray,
    pos: np.ndarray,
    r_ap_arr: np.ndarray,
    r_in_arr: np.ndarray,
    r_out_arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-star circular aperture + annulus sky (photutils 2.3 requires scalar ``r`` per aperture)."""
    from photutils.aperture import CircularAnnulus, CircularAperture
    from photutils.aperture import aperture_photometry as _aphot

    n = int(len(pos))
    flux_arr = np.full(n, np.nan, dtype=np.float64)
    sky_pp_arr = np.full(n, np.nan, dtype=np.float64)
    n_fail = 0
    for idx in range(n):
        try:
            r_ap = float(r_ap_arr[idx])
            r_in = float(r_in_arr[idx])
            r_out = float(r_out_arr[idx])
            if not (
                math.isfinite(r_ap)
                and r_ap > 0
                and math.isfinite(r_in)
                and r_in > 0
                and math.isfinite(r_out)
                and r_out > r_in
            ):
                n_fail += 1
                continue
            xy = (float(pos[idx, 0]), float(pos[idx, 1]))
            ap_i = CircularAperture([xy], r=r_ap)
            an_i = CircularAnnulus([xy], r_in=r_in, r_out=r_out)
            phot_i = _aphot(d, ap_i)
            ann_masks = an_i.to_mask(method="center")
            if not isinstance(ann_masks, (list, tuple)):
                ann_masks = [ann_masks]
            ann_img = ann_masks[0].to_image(d.shape)
            sky_pp = _sky_pp_from_annulus_image(d, ann_img)
            area = float(ap_i.area)
            flux_arr[idx] = float(phot_i["aperture_sum"][0]) - sky_pp * area
            sky_pp_arr[idx] = sky_pp
        except Exception:  # noqa: BLE001
            n_fail += 1
    if n_fail > 0:
        logging.warning(
            "[FÁZA 2A] Per-star aperture: %d/%d positions failed or skipped",
            n_fail,
            n,
        )
    return flux_arr, sky_pp_arr


def compute_per_frame_cog_correction(
    data: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    dao_flux: np.ndarray,
    aperture_r_px: np.ndarray,
    sky_pp: np.ndarray,
    *,
    fwhm_px: float,
    peak_max_adu: np.ndarray | None = None,
    sat_limit_adu: np.ndarray | None = None,
    ref_fwhm: float = 4.5,
    ladder_step_px: float = 0.5,
    min_stars: int = 8,
    isolation_fwhm: float = 6.0,
    snr_min: float = 50.0,
    sat_frac: float = 0.85,
    gain: float = 1.0,
    read_noise: float = 10.0,
    ac_factor_max: float = 5.0,
    max_stars: int = 60,
    fallback_ee: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Per-frame curve-of-growth (encircled-energy) aperture correction.

    Builds an EE(r) curve (normalised to ``ref_fwhm × FWHM``) from bright, isolated,
    unsaturated, high-SNR stars and returns a per-star multiplicative correction
    ``ac_factor = 1 / EE(r_star)`` that puts every star on the common ref-radius
    enclosed-flux scale (removing the per-star SNR-radius differential bias).

    Returns dict: ``ac_factor`` (len n, ≥1.0), ``cog_ok``, ``n_cog``, ``ref_r_px``,
    ``ee_radii``, ``ee_curve``. When fewer than ``min_stars`` COG stars are found and
    no ``fallback_ee`` is given, ``cog_ok=False`` and every ``ac_factor=1.0``.
    """
    from photutils.aperture import CircularAperture
    from photutils.aperture import aperture_photometry as _aphot

    n = int(len(x))
    out: dict[str, Any] = {
        "ac_factor": np.ones(n, dtype=np.float64),
        "cog_ok": False,
        "n_cog": 0,
        "ref_r_px": float(ref_fwhm) * float(fwhm_px) if math.isfinite(fwhm_px) else float("nan"),
        "ee_radii": None,
        "ee_curve": None,
    }
    if n == 0 or not (math.isfinite(fwhm_px) and fwhm_px > 0):
        return out

    d = np.asarray(data, dtype=np.float64)
    if np.any(~np.isfinite(d)):
        fill = float(np.nanmedian(d)) if np.any(np.isfinite(d)) else 0.0
        d = np.where(np.isfinite(d), d, fill)
    height, width = d.shape

    ref_r = float(ref_fwhm) * float(fwhm_px)
    iso_r = max(ref_r, float(isolation_fwhm) * float(fwhm_px))
    step = float(ladder_step_px) if math.isfinite(ladder_step_px) and ladder_step_px > 0 else 0.5
    radii = np.arange(step, ref_r + 1e-6, step, dtype=np.float64)
    if radii.size == 0 or radii[-1] < ref_r - 1e-6:
        radii = np.append(radii, ref_r)
    radii[-1] = ref_r  # ensure the reference radius is the final ladder point

    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    flux = np.asarray(dao_flux, dtype=np.float64)
    rap = np.asarray(aperture_r_px, dtype=np.float64)
    skp = np.asarray(sky_pp, dtype=np.float64)

    finite_xy = np.isfinite(xx) & np.isfinite(yy)
    nn = np.full(n, np.inf, dtype=np.float64)
    try:
        from scipy.spatial import cKDTree

        pts = np.column_stack([xx[finite_xy], yy[finite_xy]])
        if pts.shape[0] >= 2:
            tree = cKDTree(pts)
            dist, _ = tree.query(pts, k=2)
            nn[finite_xy] = dist[:, 1]
    except (ImportError, AttributeError, ValueError, TypeError):
        logging.debug("[COG] cKDTree unavailable — isolation check skipped")

    g = float(gain) if math.isfinite(gain) and gain > 0 else 1.0
    rn = float(read_noise) if math.isfinite(read_noise) and read_noise >= 0 else 10.0
    area_rap = math.pi * np.square(np.where(np.isfinite(rap) & (rap > 0), rap, np.nan))
    var = flux / g + np.maximum(0.0, skp) / g * area_rap + (rn / g) ** 2 * area_rap
    snr = np.where((flux > 0) & np.isfinite(var) & (var > 0), flux / np.sqrt(var), 0.0)

    if peak_max_adu is not None and sat_limit_adu is not None:
        pk = np.asarray(peak_max_adu, dtype=np.float64)
        sl = np.asarray(sat_limit_adu, dtype=np.float64)
        unsat = ~(np.isfinite(pk) & np.isfinite(sl) & (pk > float(sat_frac) * sl))
    else:
        unsat = np.ones(n, dtype=bool)

    margin = ref_r + 1.0
    in_bounds = (xx > margin) & (xx < (width - margin)) & (yy > margin) & (yy < (height - margin))

    sel = (
        finite_xy
        & (flux > 0)
        & np.isfinite(skp)
        & unsat
        & in_bounds
        & (nn > iso_r)
        & (snr >= float(snr_min))
    )
    cog_idx = np.where(sel)[0]
    # Cap to the highest-SNR subset — a robust median EE needs only a few dozen stars.
    if int(max_stars) > 0 and cog_idx.size > int(max_stars):
        order = np.argsort(snr[cog_idx])[::-1][: int(max_stars)]
        cog_idx = cog_idx[order]

    fracs: list[np.ndarray] = []
    for i in cog_idx:
        try:
            xy = [(float(xx[i]), float(yy[i]))]
            sums = np.array(
                [
                    float(_aphot(d, CircularAperture(xy, r=float(rr)), method="exact")["aperture_sum"][0])
                    for rr in radii
                ],
                dtype=np.float64,
            )
            ee = sums - float(skp[i]) * math.pi * np.square(radii)
            ref_val = float(ee[-1])
            if math.isfinite(ref_val) and ref_val > 0:
                fr = ee / ref_val
                if np.all(np.isfinite(fr)):
                    fracs.append(fr)
        except Exception as exc:  # noqa: BLE001
            logging.error('[EXC-0190] One bright star skipped in COG encircled-energy curve - aperture COG correction biased: %s', exc)
            continue

    n_cog = len(fracs)
    out["n_cog"] = n_cog
    ee_radii = radii
    ee_curve: np.ndarray | None = None
    if n_cog >= int(min_stars):
        ee_curve = np.median(np.vstack(fracs), axis=0)
        ee_curve = np.clip(ee_curve, 1e-3, 1.0)
        ee_curve[-1] = 1.0
        out["cog_ok"] = True
    elif fallback_ee is not None:
        ee_radii, ee_curve = fallback_ee
        ee_radii = np.asarray(ee_radii, dtype=np.float64)
        ee_curve = np.asarray(ee_curve, dtype=np.float64)
        out["cog_ok"] = False  # fallback used; flag as not-fresh
    else:
        return out  # too few COG stars and no fallback → no correction (ac_factor=1)

    out["ee_radii"] = ee_radii
    out["ee_curve"] = ee_curve

    ee_at = np.interp(np.clip(rap, ee_radii[0], ee_radii[-1]), ee_radii, ee_curve)
    acf = np.where((ee_at > 0) & np.isfinite(ee_at), 1.0 / ee_at, 1.0)
    acf = np.clip(acf, 1.0, float(ac_factor_max))
    acf = np.where(np.isfinite(rap) & (rap > 0), acf, 1.0)
    out["ac_factor"] = acf
    return out


def enhance_catalog_dataframe_aperture_bpm(
    df: pd.DataFrame,
    data: np.ndarray,
    hdr: Any,
    *,
    aperture_enabled: bool,
    aperture_fwhm_factor: float,
    annulus_inner_fwhm: float,
    annulus_outer_fwhm: float,
    nonlinearity_peak_percentile: float,
    nonlinearity_fwhm_ratio: float,
    master_dark_path: Path | str | None,
    gaussian_fwhm_px_override: float | None = None,
    r_small_px: float | None = None,
    r_large_px: float | None = None,
    snr_aperture_table: dict[str, Any] | None = None,
    cog_params: dict[str, Any] | None = None,
    err_background_mode: str = ERR_BKG_MODE_EMPIRICAL,
    err_empty_apertures_n: int = 64,
    err_empty_apertures_min: int = 16,
) -> pd.DataFrame:
    """Replace DAO ``flux`` with aperture photometry when enabled; add linearity/BPM flags.

    When ``cog_params`` is given (curve-of-growth aperture correction enabled), also
    emits ``dao_flux_apcorr`` / ``ac_factor`` / ``cog_ok`` without overwriting ``dao_flux``.
    """
    out = df.copy()
    arr = np.asarray(data, dtype=np.float32)

    x = pd.to_numeric(out.get("x"), errors="coerce").to_numpy(dtype=np.float64)
    y = pd.to_numeric(out.get("y"), errors="coerce").to_numpy(dtype=np.float64)
    n = len(out)
    if n == 0:
        return out

    # Pôvodný DAO flux z detect_stars_and_match_catalog (historicky v stĺpci ``flux``).
    # ``dao_flux``: sky-subtrahovaný flux (po aperturnej fotometrii, ak je zapnutá).
    flux_dao = pd.to_numeric(out.get("flux"), errors="coerce").to_numpy(dtype=np.float64)
    if "dao_flux" not in out.columns:
        out["dao_flux"] = flux_dao

    fwhm_per, fwhm_moment_med, fwhm_gaussian_f = compute_fwhm_gaussian_for_aperture_catalog(
        out,
        arr,
        hdr,
        gaussian_fwhm_px_override=gaussian_fwhm_px_override,
        aperture_fwhm_factor=aperture_fwhm_factor,
    )
    out["fwhm_estimate_px"] = fwhm_per

    if aperture_enabled and math.isfinite(float(fwhm_gaussian_f)) and float(fwhm_gaussian_f) > 0:
        try:
            # Lokálna implementácia: sky-subtracted flux cez CircularAperture + CircularAnnulus.
            from photutils.aperture import CircularAnnulus, CircularAperture
            from photutils.aperture import aperture_photometry as _aphot

            fw = float(fwhm_gaussian_f)
            global_aperture_r_px = max(0.5, float(aperture_fwhm_factor) * fw)

            pos = np.column_stack([np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)])

            if snr_aperture_table is not None:
                r_ap_arr = np.empty(n, dtype=np.float64)
                _apertures_used: list[float] = []
                _cid_series = out.get("catalog_id")
                for i in range(n):
                    _cid = ""
                    if _cid_series is not None:
                        try:
                            _cid = _normalize_gaia_id(_cid_series.iloc[i])
                        except Exception:  # noqa: BLE001
                            _cid = ""
                    _star_mag = _star_mag_for_aperture_sizing(out.iloc[i])
                    _r_ap_i = _get_star_aperture_px(
                        _cid,
                        _star_mag,
                        snr_aperture_table,
                        fallback_r=global_aperture_r_px,
                    )
                    r_ap_arr[i] = float(_r_ap_i)
                    _apertures_used.append(float(_r_ap_i))
                r_in_arr = np.maximum(r_ap_arr + 0.5, float(annulus_inner_fwhm) * fw)
                r_out_arr = np.maximum(r_in_arr + 0.5, float(annulus_outer_fwhm) * fw)
                r_ap = float(global_aperture_r_px)
                r_in = float(r_in_arr[0]) if n else float("nan")
                r_out = float(r_out_arr[0]) if n else float("nan")
                if not bool(getattr(enhance_catalog_dataframe_aperture_bpm, "_snr_ap_stats_logged", False)):
                    if _apertures_used:
                        logging.info(
                            "[FÁZA 2A] SNR per-star apertures: min=%.2fpx median=%.2fpx max=%.2fpx (N=%d)",
                            min(_apertures_used),
                            float(np.median(_apertures_used)),
                            max(_apertures_used),
                            len(_apertures_used),
                        )
                    enhance_catalog_dataframe_aperture_bpm._snr_ap_stats_logged = True
            else:
                r_ap = global_aperture_r_px
                r_in = max(r_ap + 0.5, float(annulus_inner_fwhm) * fw)
                r_out = max(r_in + 0.5, float(annulus_outer_fwhm) * fw)
                r_ap_arr = None

            d = np.asarray(arr, dtype=np.float64)
            if np.any(~np.isfinite(d)):
                fill = float(np.nanmedian(d)) if np.any(np.isfinite(d)) else 0.0
                d = np.where(np.isfinite(d), d, fill)

            if r_ap_arr is not None:
                # photutils 2.3: CircularAperture.r must be scalar — one aperture per star.
                flux_arr, sky_pp_arr = _aperture_flux_sky_per_star(d, pos, r_ap_arr, r_in_arr, r_out_arr)
                out["flux"] = flux_arr.astype(np.float64)
                out["dao_flux"] = out["flux"]
                out["aperture_r_px"] = r_ap_arr.astype(np.float64)
                out["sky_annulus_r_out_px"] = r_out_arr.astype(np.float64)
                out["noise_floor_adu"] = sky_pp_arr.astype(np.float64)
                out[SKY_ADU_PER_PX_ANNULUS_COL] = sky_pp_arr.astype(np.float64)
            else:
                ap = CircularAperture(pos, r=r_ap)
                an = CircularAnnulus(pos, r_in=r_in, r_out=r_out)
                phot_ap = _aphot(d, ap)
                sum_ap = np.asarray(phot_ap["aperture_sum"], dtype=np.float64)
                area_ap_per = float(ap.area)
                sky_pp_arr = np.zeros(n, dtype=np.float64)
                ann_masks = an.to_mask(method="center")
                if not isinstance(ann_masks, (list, tuple)):
                    ann_masks = [ann_masks]
                for i, amask in enumerate(ann_masks):
                    try:
                        ann_img = amask.to_image(d.shape)
                        sky_pp_arr[i] = _sky_pp_from_annulus_image(d, ann_img)
                    except Exception:  # noqa: BLE001
                        sky_pp_arr[i] = float(np.median(d))
                flux_arr = sum_ap - sky_pp_arr * area_ap_per
                out["flux"] = flux_arr.astype(np.float64)
                out["dao_flux"] = out["flux"]
                out["aperture_r_px"] = float(r_ap)
                out["sky_annulus_r_out_px"] = float(r_out)
                out["noise_floor_adu"] = sky_pp_arr.astype(np.float64)
                out[SKY_ADU_PER_PX_ANNULUS_COL] = sky_pp_arr.astype(np.float64)

            # Multi-apertúra: rovnaký sky_pp_arr (ADU/px²) × plocha apertúry ako sky odčítanie.
            if r_small_px is not None and r_large_px is not None:
                try:
                    _rs = float(r_small_px)
                    _rl = float(r_large_px)
                except (TypeError, ValueError):
                    _rs, _rl = float("nan"), float("nan")
                if (
                    math.isfinite(_rs)
                    and math.isfinite(_rl)
                    and _rs > 0
                    and _rl > 0
                    and int(sky_pp_arr.shape[0]) == n
                ):
                    try:
                        ap_sm = CircularAperture(pos, r=_rs)
                        ap_lg = CircularAperture(pos, r=_rl)
                        phot_sm = _aphot(d, ap_sm)
                        phot_lg = _aphot(d, ap_lg)
                        sum_sm = np.asarray(phot_sm["aperture_sum"], dtype=np.float64).ravel()
                        sum_lg = np.asarray(phot_lg["aperture_sum"], dtype=np.float64).ravel()
                        if sum_sm.size != n or sum_lg.size != n:
                            raise ValueError(
                                f"multi-aperture sum size mismatch: n={n} small={sum_sm.size} large={sum_lg.size}"
                            )
                        area_sm = math.pi * _rs * _rs
                        area_lg = math.pi * _rl * _rl
                        flux_sm = sum_sm - sky_pp_arr * area_sm
                        flux_lg = sum_lg - sky_pp_arr * area_lg
                        flux_sm = np.where(np.isfinite(flux_sm), flux_sm, np.nan)
                        flux_lg = np.where(np.isfinite(flux_lg), flux_lg, np.nan)
                        out["flux_small"] = flux_sm.astype(np.float64)
                        out["flux_large"] = flux_lg.astype(np.float64)
                    except (ValueError, TypeError) as _ma_exc:
                        logging.debug("[PHOT] multi-aperture flux_small/flux_large skipped: %s", _ma_exc)

            # F-BINGAIN-1: per-frame empirical background noise at production aperture radii.
            _bkg_mode = _normalize_err_background_mode(err_background_mode)
            _n_empty = _clamp_err_empty_apertures_n(err_empty_apertures_n)
            _n_empty_min = _clamp_err_empty_apertures_min(err_empty_apertures_min)
            _sigma_by_r: dict[float, tuple[float, str]] = {}
            if _bkg_mode == ERR_BKG_MODE_EMPIRICAL:
                if r_ap_arr is not None:
                    _unique_r = np.unique(np.round(r_ap_arr[np.isfinite(r_ap_arr) & (r_ap_arr > 0)], 4))
                else:
                    _unique_r = np.array([float(r_ap)], dtype=np.float64)
                for _r_u in _unique_r:
                    if not math.isfinite(float(_r_u)) or float(_r_u) <= 0:
                        continue
                    _ri = max(float(_r_u) + 0.5, float(annulus_inner_fwhm) * fw)
                    _ro = max(_ri + 0.5, float(annulus_outer_fwhm) * fw)
                    _seed = _labbe_content_seed_from_header(hdr, r_ap=float(_r_u))
                    _frame_id = str(
                        hdr.get("DATE-OBS")
                        or hdr.get("FILENAME")
                        or hdr.get("FRAME")
                        or ""
                    )
                    _sig, _nv, _reason = measure_empty_aperture_sigma_bkg(
                        d,
                        np.asarray(x, dtype=np.float64),
                        np.asarray(y, dtype=np.float64),
                        float(_r_u),
                        float(_ri),
                        float(_ro),
                        n_apertures=_n_empty,
                        min_valid=_n_empty_min,
                        seed=int(_seed),
                        frame_id=_frame_id,
                        star_list_source="catalog_df_in_memory",
                    )
                    if not hasattr(enhance_catalog_dataframe_aperture_bpm, "_labbe_seeds"):
                        enhance_catalog_dataframe_aperture_bpm._labbe_seeds = []
                    enhance_catalog_dataframe_aperture_bpm._labbe_seeds.append(
                        {"r_ap": float(_r_u), "seed": int(_seed), "n_valid": int(_nv)}
                    )
                    if math.isfinite(_sig) and _sig >= 0:
                        _sigma_by_r[float(_r_u)] = (float(_sig), ERR_BKG_SOURCE_EMPIRICAL)
                    else:
                        _sigma_by_r[float(_r_u)] = (float("nan"), ERR_BKG_SOURCE_HOWELL_FALLBACK)
                        if not hasattr(enhance_catalog_dataframe_aperture_bpm, "_err_bkg_logged"):
                            enhance_catalog_dataframe_aperture_bpm._err_bkg_logged = set()
                        _log_id = str(hdr.get("FRAME") or hdr.get("VY_FRAME") or id(hdr))
                        if _log_id not in enhance_catalog_dataframe_aperture_bpm._err_bkg_logged:
                            log_event(
                                f"[PHOT] err_bkg empirical fallback (howell): r_ap={float(_r_u):.2f}px "
                                f"n_valid={_nv} reason={_reason or 'unknown'}"
                            )
                            enhance_catalog_dataframe_aperture_bpm._err_bkg_logged.add(_log_id)

                _sigma_col = np.full(n, np.nan, dtype=np.float64)
                _src_col = np.full(n, ERR_BKG_SOURCE_HOWELL_FALLBACK, dtype=object)
                if r_ap_arr is not None:
                    for _i in range(n):
                        _r_key = round(float(r_ap_arr[_i]), 4)
                        _sig_v, _src_v = _sigma_by_r.get(
                            _r_key,
                            (float("nan"), ERR_BKG_SOURCE_HOWELL_FALLBACK),
                        )
                        _sigma_col[_i] = _sig_v
                        _src_col[_i] = _src_v
                else:
                    _sig_v, _src_v = _sigma_by_r.get(
                        round(float(r_ap), 4),
                        (float("nan"), ERR_BKG_SOURCE_HOWELL_FALLBACK),
                    )
                    _sigma_col[:] = _sig_v
                    _src_col[:] = _src_v
                out[SIGMA_BKG_AP_COL] = _sigma_col
                out[ERR_BKG_SOURCE_COL] = _src_col
            elif _bkg_mode == ERR_BKG_MODE_HOWELL:
                out[SIGMA_BKG_AP_COL] = np.full(n, np.nan, dtype=np.float64)
                out[ERR_BKG_SOURCE_COL] = np.full(n, ERR_BKG_MODE_HOWELL, dtype=object)

            # Per-frame curve-of-growth aperture correction (gated; never overwrites dao_flux).
            if cog_params is not None:
                try:
                    _peak = pd.to_numeric(out.get("peak_max_adu"), errors="coerce").to_numpy(dtype=np.float64) \
                        if "peak_max_adu" in out.columns else None
                    _sat = pd.to_numeric(out.get("saturate_limit_adu"), errors="coerce").to_numpy(dtype=np.float64) \
                        if "saturate_limit_adu" in out.columns else None
                    _rap_for_cog = (
                        r_ap_arr if r_ap_arr is not None
                        else np.full(n, float(r_ap), dtype=np.float64)
                    )
                    _cog = compute_per_frame_cog_correction(
                        d,
                        np.asarray(x, dtype=np.float64),
                        np.asarray(y, dtype=np.float64),
                        pd.to_numeric(out["dao_flux"], errors="coerce").to_numpy(dtype=np.float64),
                        np.asarray(_rap_for_cog, dtype=np.float64),
                        sky_pp_arr,
                        fwhm_px=fw,
                        peak_max_adu=_peak,
                        sat_limit_adu=_sat,
                        ref_fwhm=float(cog_params.get("ref_fwhm", 4.5)),
                        ladder_step_px=float(cog_params.get("ladder_step_px", 0.5)),
                        min_stars=int(cog_params.get("min_stars", 8)),
                        isolation_fwhm=float(cog_params.get("isolation_fwhm", 6.0)),
                        snr_min=float(cog_params.get("snr_min", 50.0)),
                        sat_frac=float(cog_params.get("sat_frac", 0.85)),
                        gain=float(cog_params.get("gain", 1.0)),
                        read_noise=float(cog_params.get("read_noise", 10.0)),
                        ac_factor_max=float(cog_params.get("ac_factor_max", 5.0)),
                        fallback_ee=cog_params.get("fallback_ee"),
                    )
                    _acf = np.asarray(_cog["ac_factor"], dtype=np.float64)
                    _dao = pd.to_numeric(out["dao_flux"], errors="coerce").to_numpy(dtype=np.float64)
                    out["ac_factor"] = _acf
                    out["dao_flux_apcorr"] = (_dao * _acf).astype(np.float64)
                    out["cog_ok"] = bool(_cog["cog_ok"])
                    if not bool(getattr(enhance_catalog_dataframe_aperture_bpm, "_cog_logged", False)):
                        logging.info(
                            "[COG] per-frame aperture correction: n_cog=%d cog_ok=%s ref_r=%.2fpx ac_factor median=%.4f",
                            int(_cog["n_cog"]),
                            bool(_cog["cog_ok"]),
                            float(_cog["ref_r_px"]),
                            float(np.nanmedian(_acf)),
                        )
                        enhance_catalog_dataframe_aperture_bpm._cog_logged = True
                except Exception as _cog_exc:  # noqa: BLE001
                    logging.warning("[COG] per-frame aperture correction skipped: %s", _cog_exc)
                    out["ac_factor"] = np.ones(n, dtype=np.float64)
                    out["dao_flux_apcorr"] = pd.to_numeric(out["dao_flux"], errors="coerce").to_numpy(dtype=np.float64)
                    out["cog_ok"] = False
        except Exception as _ap_exc:  # noqa: BLE001
            logging.warning(
                "[FÁZA 2A] Aperture photometry failed — restoring pre-aperture flux: %s",
                _ap_exc,
                exc_info=True,
            )
            out["dao_flux"] = flux_dao
            out["flux"] = flux_dao
    else:
        out["dao_flux"] = flux_dao
        out["flux"] = flux_dao

    peak = pd.to_numeric(out.get("peak_max_adu"), errors="coerce").to_numpy(dtype=np.float64)
    finite_pk = peak[np.isfinite(peak)]
    thr_pk = float("nan")
    if finite_pk.size > 0:
        pct = min(100.0, max(0.0, 100.0 - float(nonlinearity_peak_percentile)))
        thr_pk = float(np.percentile(finite_pk, pct))

    ratio = float(nonlinearity_fwhm_ratio)
    likely_nl = np.zeros(n, dtype=bool)
    for i in range(n):
        if not (math.isfinite(fwhm_per[i]) and math.isfinite(fwhm_moment_med) and fwhm_moment_med > 0):
            continue
        if not (math.isfinite(peak[i]) and math.isfinite(thr_pk) and peak[i] >= thr_pk):
            continue
        if fwhm_per[i] > ratio * fwhm_moment_med:
            likely_nl[i] = True
    out["likely_nonlinear"] = likely_nl

    bpm_path = None
    bpm: dict[str, Any] | None = None
    if master_dark_path:
        mp = Path(str(master_dark_path))
        bpm_path = mp.parent / f"{mp.stem}_dark_bpm.json"
        if bpm_path.is_file():
            try:
                bpm = json.loads(bpm_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                bpm = None

    bad_x = bad_columns_for_light_frame(bpm, light_header=hdr)
    on_bad = np.zeros(n, dtype=bool)
    if bad_x:
        for i in range(n):
            if not np.isfinite(x[i]):
                continue
            xi = int(round(float(x[i])))
            if xi in bad_x:
                on_bad[i] = True
    out["on_bad_column"] = on_bad

    if "photometry_ok" in out.columns:
        base_ok = out["photometry_ok"].fillna(True).astype(bool).to_numpy()
        out["photometry_ok"] = base_ok & (~likely_nl) & (~on_bad)
    else:
        out["photometry_ok"] = ~(likely_nl | on_bad)

    if "source_type" in out.columns and "dao_flux" in out.columns:
        _forced_mask = (
            out["source_type"].fillna("").astype(str).str.strip().eq("FORCED_APERTURE")
        )
        _dao_num = pd.to_numeric(out["dao_flux"], errors="coerce")
        _has_flux = _dao_num.notna() & (_dao_num != 0)
        out.loc[_forced_mask & _has_flux, "photometry_ok"] = True

    # Multi-apertúra: stĺpce vždy existujú (NaN ak meranie neprebehlo alebo bolo vypnuté).
    if n > 0:
        _nan_vec = np.full(n, np.nan, dtype=np.float64)
        if "flux_small" not in out.columns:
            out["flux_small"] = _nan_vec
        if "flux_large" not in out.columns:
            out["flux_large"] = _nan_vec.copy()

    return out


def _phase0_effective_frame_hw_px(
    vt: pd.DataFrame,
    ms: pd.DataFrame,
    *,
    frame_w_px: int,
    frame_h_px: int,
    edge_margin_px: int,
) -> tuple[int, int]:
    """``frame_w_px`` / ``frame_h_px`` z volania alebo väčšie — podľa max. x,y v VT a masterstars.

    Predvolené 2082×1397 často nezodpovedajú veľkému čipu; inak sa VSX ciele s veľkými pixelmi
    (napr. DY Peg) vylúčia ešte pred cross-matchom, **bez** ohľadu na ``vsx_type`` (žiadny filter na SXPHE).
    """
    xs: list[float] = []
    ys: list[float] = []
    for df in (vt, ms):
        if "x" in df.columns and "y" in df.columns:
            xs.extend(pd.to_numeric(df["x"], errors="coerce").dropna().astype(float).tolist())
            ys.extend(pd.to_numeric(df["y"], errors="coerce").dropna().astype(float).tolist())
    if not xs or not ys:
        return int(frame_w_px), int(frame_h_px)
    em = int(edge_margin_px)
    need_w = int(math.ceil(float(max(xs)))) + em + 2
    need_h = int(math.ceil(float(max(ys)))) + em + 2
    return max(int(frame_w_px), need_w), max(int(frame_h_px), need_h)


def _active_target_zone_flag(ms_row: pd.Series, zone_val_raw: str) -> str:
    """Mapovanie masterstars ``zone`` (+ legacy ``is_saturated``) na ``zone_flag`` pre active_targets."""
    z = str(zone_val_raw or "").strip().lower()
    if z in ("linear", "noisy1", "noisy2", "noisy3", "saturated"):
        return z
    try:
        sat = bool(ms_row.get("is_saturated", False))
    except Exception:  # noqa: BLE001
        # EXC-0191: T3 -- Nested log_event inside catalog_id auto-repair failure also fails - repair error messag... (EXCEPT-BULK-2 2026-07-08)
        sat = False
    if sat:
        return "saturated"
    if not z:
        return "neznáma_zóna"
    return z


def _auto_repair_catalog_ids(
    *,
    vt_path: Path,
    gaia_db_path: str | None,
    log_fn: Any = None,
    max_sep_arcsec: float = 10.0,
) -> dict[str, Any]:
    """Auto-repair poškodené Gaia catalog_id v variable_targets.csv podľa RA/DEC.

    Bezpečnostné pravidlá:
    - Ak `gaia_db_path` nie je nastavená alebo DB neexistuje → nič nerob.
    - Ak `variable_targets.csv` nemá `catalog_id` alebo RA/DEC → nič nerob.
    - Opravuj iba vtedy, keď najbližší Gaia zdroj je dostatočne blízko (`max_sep_arcsec`).
    - Vytvor `.bak` zálohu iba ak sa niečo reálne opravilo.
    """
    try:
        from repair_catalog_ids import repair_catalog_ids_from_gaia_db  # noqa: PLC0415

        _log = log_fn or log_event
        if not gaia_db_path:
            return {"ok": False, "reason": "no_gaia_db_path"}
        dbp = Path(str(gaia_db_path))
        if not dbp.is_file():
            return {"ok": False, "reason": "gaia_db_missing", "gaia_db_path": str(dbp)}
        if not Path(vt_path).is_file():
            return {"ok": False, "reason": "vt_missing", "vt_path": str(vt_path)}
        res = repair_catalog_ids_from_gaia_db(
            variable_targets_csv=Path(vt_path),
            gaia_db_path=dbp,
            backup=True,
            max_sep_arcsec=float(max_sep_arcsec),
            log_fn=_log,
        )
        if int(res.get("repaired") or 0) > 0:
            _log(f"[COMP] auto-repair variable_targets.csv: repaired={res.get('repaired')} warnings={res.get('warnings')}")
        return res
    except Exception as exc:  # noqa: BLE001
        try:
            (log_fn or log_event)(f"[COMP] auto-repair variable_targets.csv FAILED: {exc!s}")
        except Exception:  # noqa: BLE001
            pass
        return {"ok": False, "reason": "exception", "error": str(exc)}


def _enrich_active_targets_bp_rp(
    targets_df: pd.DataFrame,
    *,
    gaia_db_path: str | Path | None,
) -> pd.DataFrame:
    """Doplň ``bp_rp`` pre active targets z Gaia DR3 podľa ``catalog_id``."""
    if targets_df is None or getattr(targets_df, "empty", True):
        return targets_df

    df = targets_df.copy()
    if "bp_rp" not in df.columns:
        df["bp_rp"] = float("nan")
    df["bp_rp"] = pd.to_numeric(df["bp_rp"], errors="coerce")
    if "ra_deg" in df.columns:
        df["ra_deg"] = pd.to_numeric(df["ra_deg"], errors="coerce")
    if "dec_deg" in df.columns:
        df["dec_deg"] = pd.to_numeric(df["dec_deg"], errors="coerce")

    gaia_path = str(gaia_db_path or "").strip()
    con = None
    gaia_cols: set[str] = set()
    if gaia_path and os.path.exists(gaia_path):
        try:
            import sqlite3  # noqa: PLC0415

            con = sqlite3.connect(gaia_path)
            con.row_factory = sqlite3.Row
            gaia_cols = {
                str(r[1]).strip().lower() for r in con.execute("PRAGMA table_info('gaia_dr3')").fetchall()
            }
        except Exception:  # noqa: BLE001
            # EXC-0193: T2 -- sqlite con.close() after active-target bp_rp enrichment ignored (EXCEPT-BULK-2 2026-07-08)
            con = None
            gaia_cols = set()
    sel_bp = "bp_rp" in gaia_cols
    gaia_cache: dict[int, float] = {}

    def _gaia_bp(sid_i: int) -> float:
        if sid_i in gaia_cache:
            return gaia_cache[sid_i]
        bp_r = float("nan")
        if con is not None and sel_bp:
            try:
                rw = con.execute(
                    "SELECT bp_rp FROM gaia_dr3 WHERE source_id=? LIMIT 1;",
                    (int(sid_i),),
                ).fetchone()
                if rw is not None and rw["bp_rp"] is not None:
                    bp_r = float(rw["bp_rp"])
            except Exception as exc:  # noqa: BLE001
                logging.error('[EXC-0192] Gaia DB bp_rp fetch for active targets fails - target row keeps NaN bp_rp: %s', exc)
                pass
        gaia_cache[int(sid_i)] = bp_r if math.isfinite(bp_r) else float("nan")
        return gaia_cache[int(sid_i)]

    try:
        for idx, row in df.iterrows():
            try:
                bp_now = float(pd.to_numeric(row.get("bp_rp"), errors="coerce"))
            except Exception:  # noqa: BLE001
                bp_now = float("nan")
            if math.isfinite(bp_now):
                continue
            sid_i = _sid_int(row.get("catalog_id"))
            if sid_i is None:
                continue
            gaia_bp = _gaia_bp(sid_i)
            if math.isfinite(gaia_bp):
                df.at[idx, "bp_rp"] = float(gaia_bp)
    finally:
        try:
            if con is not None:
                con.close()
        except Exception:  # noqa: BLE001
            # EXC-0195: T4 -- DB OBS_FILES NAXIS query fails - returns caller-supplied default frame width/height (EXCEPT-BULK-2 2026-07-08)
            pass

    return df


def _resolve_frame_hw_px_from_masterstar(
    ms_fits: Path,
    *,
    frame_w_px: int,
    frame_h_px: int,
    db: Any = None,
    draft_id: int | None = None,
) -> tuple[int, int, str]:
    """Authoritative chip width/height for Phase 0+1 spatial culling.

    Priority: (1) MASTERSTAR FITS ``NAXIS1``/``NAXIS2``; (2) DB ``SCANNING`` via draft;
    (3) caller defaults (global cfg knob / hardcoded 2082×1397).
    """
    w_def, h_def = int(frame_w_px), int(frame_h_px)
    if ms_fits.is_file():
        try:
            with astrofits.open(ms_fits, memmap=False) as hdul:
                hdr = hdul[0].header
                w = int(hdr.get("NAXIS1", 0) or 0)
                h = int(hdr.get("NAXIS2", 0) or 0)
                if w > 0 and h > 0:
                    return w, h, "fits_naxis"
        except Exception:  # noqa: BLE001
            pass
    if db is not None and draft_id is not None:
        try:
            did = int(draft_id)
        except (TypeError, ValueError):
            did = 0
        if did > 0 and hasattr(db, "conn"):
            try:
                cur = db.conn.execute(
                    """
                    SELECT s.NAXIS1, s.NAXIS2
                    FROM OBS_FILES f
                    JOIN SCANNING s ON s.ID = f.ID_SCANNING
                    WHERE f.DRAFT_ID = ?
                      AND LOWER(COALESCE(f.IMAGETYP, '')) = 'light'
                      AND s.NAXIS1 > 0 AND s.NAXIS2 > 0
                    ORDER BY f.FILE_PATH
                    LIMIT 1
                    """,
                    (did,),
                )
                row = cur.fetchone()
                if row is not None:
                    w = int(row["NAXIS1"] or 0)
                    h = int(row["NAXIS2"] or 0)
                    if w > 0 and h > 0:
                        return w, h, "db_scanning"
            except Exception:  # noqa: BLE001
                pass
    return w_def, h_def, "caller_default"


def _read_field_density_inputs(
    ms_fits: Path,
    masterstars_csv: Path,
    frame_w_px: int,
    frame_h_px: int,
) -> tuple[int, int, int, str, int | None]:
    """Vráti ``(n_stars, chip_w, chip_h, source, n_stars_dao_raw)`` pre hustotu poľa.

    ``n_stars``: počet riadkov v ``masterstars_csv`` s neprázdnym ``catalog_id`` (Gaia-matched).
    ``n_stars_dao_raw``: ``VY_NDAO`` z MASTERSTAR FITS (iba referencia / JSON, nie klasifikácia).
    ``source``: ``masterstars_gaia_matched`` | ``VY_NDAO_fallback`` | ``defaults``.
    """
    cw, ch = int(frame_w_px), int(frame_h_px)
    n_stars = 0
    src = "defaults"
    vy_ndao_raw: int | None = None
    cw, ch, _hw_src = _resolve_frame_hw_px_from_masterstar(
        ms_fits, frame_w_px=cw, frame_h_px=ch
    )
    if _hw_src == "fits_naxis":
        src = "fits_naxis"
    if ms_fits.is_file():
        try:
            with astrofits.open(ms_fits, memmap=False) as hdul:
                hdr = hdul[0].header
                v = hdr.get("VY_NDAO")
                if v is not None and str(v).strip() != "":
                    try:
                        vy_ndao_raw = int(float(v))
                    except (TypeError, ValueError):
                        vy_ndao_raw = None
        except Exception as exc:  # noqa: BLE001
            logging.error('[EXC-0196] VY_NDAO header read from MASTERSTAR fails - field density uses masterstars count fallback: %s', exc)
            pass
    msc_path = Path(masterstars_csv)
    if msc_path.is_file():
        try:
            _msc_df = pd.read_csv(
                msc_path,
                usecols=["catalog_id"],
                low_memory=False,
                dtype={"catalog_id": str},
            )
            _cid = _msc_df["catalog_id"].astype(str).str.strip()
            _n_gaia = int((~_cid.isin(["", "nan", "None"])).sum())
            if _n_gaia > 0:
                n_stars = _n_gaia
                src = "masterstars_gaia_matched"
        except Exception as exc:  # noqa: BLE001
            logging.error('[EXC-0197] masterstars_full_match.csv star-count read fails - field density n_stars may use VY_NDA...: %s', exc)
            pass
    if n_stars <= 0 and vy_ndao_raw is not None and vy_ndao_raw > 0:
        n_stars = int(vy_ndao_raw)
        src = "VY_NDAO_fallback"
    return int(max(0, n_stars)), cw, ch, src, vy_ndao_raw


def _refresh_variable_targets_xy(
    variable_targets_csv: Path,
    wcs: Any,
    chip_w: int,
    chip_h: int,
) -> None:
    """Prepočíta x/y stĺpce variable_targets.csv z aktuálneho MASTERSTAR WCS."""
    from astropy.wcs import WCS

    if wcs is None or not isinstance(wcs, WCS):
        return
    vt_path = Path(variable_targets_csv)
    if not vt_path.is_file():
        return

    logging.debug("[VT REFRESH] frame %s×%s px (MASTERSTAR → VT x,y)", chip_w, chip_h)

    df = pd.read_csv(vt_path, low_memory=False, dtype=_GAIA_ID_DTYPE)
    if "ra_deg" in df.columns:
        ra = pd.to_numeric(df["ra_deg"], errors="coerce").to_numpy(dtype=np.float64)
    elif "ra" in df.columns:
        ra = pd.to_numeric(df["ra"], errors="coerce").to_numpy(dtype=np.float64)
    else:
        logging.warning("[VT REFRESH] chýbajú stĺpce ra_deg / ra — x/y neaktualizované")
        return
    if "dec_deg" in df.columns:
        dec = pd.to_numeric(df["dec_deg"], errors="coerce").to_numpy(dtype=np.float64)
    elif "dec" in df.columns:
        dec = pd.to_numeric(df["dec"], errors="coerce").to_numpy(dtype=np.float64)
    else:
        logging.warning("[VT REFRESH] chýbajú stĺpce dec_deg / dec — x/y neaktualizované")
        return

    try:
        ok = np.isfinite(ra) & np.isfinite(dec)
        xy = np.full((len(df), 2), np.nan, dtype=np.float64)
        if bool(ok.any()):
            pts = np.column_stack([ra[ok], dec[ok]])
            xy[ok, :] = wcs.all_world2pix(pts, 0)
        df["x"] = xy[:, 0]
        df["y"] = xy[:, 1]
        df.to_csv(vt_path, index=False)
    except (ValueError, TypeError, AttributeError) as e:
        from except_fix_counters import get_except_fix_counters

        get_except_fix_counters().vt_wcs_refresh_fail += 1
        logging.error("[VT REFRESH] WCS prepočet zlyhal: %s — x/y ostávajú stale", e)
        return

    logging.info("[VT REFRESH] x/y súradnice variable_targets.csv aktualizované z MASTERSTAR WCS")
    xv = df["x"].to_numpy(dtype=np.float64, copy=False)
    yv = df["y"].to_numpy(dtype=np.float64, copy=False)
    if np.isfinite(xv).any() and np.isfinite(yv).any():
        logging.info(
            "[VT REFRESH] %d riadkov, x=[%.0f,%.0f] y=[%.0f,%.0f]",
            len(df),
            float(np.nanmin(xv)),
            float(np.nanmax(xv)),
            float(np.nanmin(yv)),
            float(np.nanmax(yv)),
        )
    else:
        logging.info("[VT REFRESH] %d riadkov (žiadne platné x/y po prepočte)", len(df))


def select_active_targets(
    variable_targets_csv: Path,
    masterstars_csv: Path,
    *,
    frame_w_px: int = 2082,
    frame_h_px: int = 1397,
    edge_margin_px: int = 50,
    safe_bbox: tuple[float, float, float, float] | None = None,
    match_radius_arcsec: float = 15.0,
    gaia_db_path: str | None = None,
    vsx_local_db_path: str | None = None,
    masterstar_fits_path: Path | str | None = None,
    plate_scale_arcsec_px: float | None = None,
    cfg: Any | None = None,
) -> pd.DataFrame:
    """Fáza 0: Filtruj VSX premenné → active_targets.

    Pravidlá:
    - Hviezda musí byť v snímke (``x,y`` aspoň ``edge_margin_px`` od okraja efektívneho poľa; to isté číslo
      ako ``chip_interior_margin_px`` vo Fáze 0+1 — jednotné s porovnávačkami a suspected).
    - Šírka/výška sa zväčší z dát ak treba
    - Must match masterstars_full_match.csv (cross-match < match_radius_arcsec).
      VSX without masterstar (DAO+Gaia) match is excluded from active_targets.
    - ``catalog_id`` z masterstars musí byť neprázdny (inak sa cieľ vynechá).
    - **Žiadny filter na zónu** (linear / noisy / saturated všetky prejdú); kvalita je v ``zone_flag``,
      saturované ciele majú ``skip_photometry=True`` pre Fázu 2A.
    - **Žiadny filter na ``vsx_type``** (SXPHE, DSCT, … sa nevyhadzujú samé o sebe).

    Returns:
        DataFrame s active targets — stĺpce z variable_targets + pridané zo masterstars:
        [name, catalog_id, ra_deg, dec_deg, vsx_name, vsx_type, vsx_period,
         x, y, mag, b_v, bp_rp, zone_flag, skip_photometry]
    """
    global LAST_EXCLUDED_TARGETS
    # Auto-repair poškodených Gaia ID pred načítaním (ak je dostupná lokálna Gaia DB).
    _auto_repair_catalog_ids(vt_path=Path(variable_targets_csv), gaia_db_path=gaia_db_path, log_fn=log_event)

    # variable_targets.csv môže prísť zvonka a často má catalog_id ako float/scientific — čítaj ako string
    vt = pd.read_csv(variable_targets_csv, low_memory=False, dtype=_GAIA_ID_DTYPE)
    # masterstars_full_match.csv často nesie presný Gaia source_id v "name" aj keď catalog_id je poškodený floatom
    ms = pd.read_csv(masterstars_csv, low_memory=False, dtype=_GAIA_ID_DTYPE)
    if "catalog_id" in vt.columns:
        vt["catalog_id"] = vt["catalog_id"].apply(_normalize_gaia_id)
    # Normalizuj Gaia ID na string.
    # POZOR: "name" v masterstars často obsahuje presný Gaia source_id; nesmieme ho prehnať cez float().
    if "catalog_id" in ms.columns:
        ms["catalog_id"] = _normalize_id_series(ms["catalog_id"])
    if "name" in ms.columns:
        ms["name"] = ms["name"].fillna("").astype(str).str.strip()

    # Normalizuj bool stĺpce v masterstars
    for col in ("is_usable", "is_saturated", "is_noisy", "snr50_ok", "likely_saturated"):
        if col in ms.columns:
            ms[col] = _bool_col(ms[col])

    fw, fh = _phase0_effective_frame_hw_px(
        vt, ms, frame_w_px=int(frame_w_px), frame_h_px=int(frame_h_px), edge_margin_px=int(edge_margin_px)
    )
    if fw != int(frame_w_px) or fh != int(frame_h_px):
        logging.info(
            "[FÁZA 0] Rozmer čipu zväčšený z %s×%s na %s×%s px (max x,y z variable_targets/masterstars + okraj)",
            int(frame_w_px),
            int(frame_h_px),
            fw,
            fh,
        )

    # Filter: v snímke (annulus-aware safe bbox, else fixed edge margin)
    vt["x"] = pd.to_numeric(vt["x"], errors="coerce")
    vt["y"] = pd.to_numeric(vt["y"], errors="coerce")
    if safe_bbox is not None:
        try:
            x0b, y0b, x1b, y1b = safe_bbox
            before = int(len(vt))
            in_frame = vt["x"].between(float(x0b), float(x1b)) & vt["y"].between(float(y0b), float(y1b))
            removed = before - int(in_frame.sum())
            if removed > 0:
                logging.info(
                    f"[BORDER] Active targets: removed {removed} rows outside safe bbox "
                    f"(annulus-aware intersection)"
                )
        except Exception:  # noqa: BLE001
            in_frame = (
                vt["x"].between(edge_margin_px, fw - edge_margin_px)
                & vt["y"].between(edge_margin_px, fh - edge_margin_px)
            )
    else:
        in_frame = (
            vt["x"].between(edge_margin_px, fw - edge_margin_px)
            & vt["y"].between(edge_margin_px, fh - edge_margin_px)
        )
    vt_in = vt[in_frame].copy()

    _cfg = cfg if cfg is not None else AppConfig()

    # Cross-match s masterstars (RA/Dec haversine, or pixel distance when WCS scale is bad)
    ms["ra_deg"] = pd.to_numeric(ms["ra_deg"], errors="coerce")
    ms["dec_deg"] = pd.to_numeric(ms["dec_deg"], errors="coerce")
    vt_in["ra_deg"] = pd.to_numeric(vt_in["ra_deg"], errors="coerce")
    vt_in["dec_deg"] = pd.to_numeric(vt_in["dec_deg"], errors="coerce")

    ms_arr = ms[["ra_deg", "dec_deg"]].to_numpy(dtype=float)
    ms_x_arr = pd.to_numeric(ms.get("x"), errors="coerce").to_numpy(dtype=np.float64)
    ms_y_arr = pd.to_numeric(ms.get("y"), errors="coerce").to_numpy(dtype=np.float64)

    _plate_nominal = (
        float(plate_scale_arcsec_px)
        if plate_scale_arcsec_px is not None
        and math.isfinite(float(plate_scale_arcsec_px))
        and float(plate_scale_arcsec_px) > 0
        else float(_cfg.phase01_plate_scale_arcsec_per_px or 1.3)
    )

    _use_pixel_dist = False
    _ms_fits_chk = (
        Path(str(masterstar_fits_path)).expanduser().resolve()
        if masterstar_fits_path
        else Path(masterstars_csv).resolve().parent / "MASTERSTAR.fits"
    )
    if _ms_fits_chk.is_file():
        try:
            import warnings as _warnings  # noqa: PLC0415

            from astropy.wcs import FITSFixedWarning, WCS as _WCS_chk  # noqa: PLC0415

            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore", FITSFixedWarning)
                with astrofits.open(_ms_fits_chk, memmap=False) as _hd_w:
                    _wcs_chk = _WCS_chk(_hd_w[0].header)
            _psm_chk = np.asarray(_wcs_chk.pixel_scale_matrix, dtype=np.float64)
            _actual_scale = float(np.sqrt(np.abs(np.linalg.det(_psm_chk))) * 3600.0)
            _scale_ratio = abs(_actual_scale - _plate_nominal) / max(_plate_nominal, 1e-9)
            if _scale_ratio > 0.20:
                _use_pixel_dist = True
                log_event(
                    f"[SELECT TARGETS] WCS scale {_actual_scale:.3f}\"/px deviates "
                    f"{_scale_ratio * 100.0:.1f}% from nominal {_plate_nominal:.3f}\"/px "
                    f"— pixel-distance matching"
                )
        except Exception as _wcs_exc:  # noqa: BLE001
            # EXC-0200: T4 -- Non-numeric Gaia id string returned as-is in select_active_targets helper (EXCEPT-BULK-2 2026-07-08)
            logging.error('[EXC-0199] WCS scale sanity check failure logged - distance matching may stay on ra/dec instead of...: %s', exc)
            logging.warning("[SELECT TARGETS] WCS sanity check failed: %s", _wcs_exc)
    if _use_pixel_dist:
        log_event("[SELECT TARGETS] Distance mode: pixel-fallback")

    # TODO-23: adaptive matching radius — VSX/Gaia catalog → masterstars (not 1-pixel centroid match)
    # Fixed: respect caller/config floor, adaptive with generous minimum (5× plate scale)
    if _plate_nominal > 0:
        _adaptive = _plate_nominal * 5.0
        _cfg_floor = float(_cfg.phase01_match_radius_arcsec)
        match_radius_arcsec = max(_adaptive, _cfg_floor)
    else:
        match_radius_arcsec = float(_cfg.phase01_match_radius_arcsec)
    LOGGER.debug(
        '[TODO-23] match_radius=%.2f" (plate_scale=%s"/px, adaptive=5×, floor=%s")',
        match_radius_arcsec,
        f"{float(plate_scale_arcsec_px):.3f}" if plate_scale_arcsec_px else "unknown",
        f"{float(_cfg.phase01_match_radius_arcsec):.2f}",
    )

    def _gaia_id_str(x: Any) -> str:
        s = str(x).strip()
        if s in ("", "nan"):
            return ""
        try:
            return str(int(float(s)))
        except Exception:  # noqa: BLE001
            return s

    from comp_selection_per_target import (  # noqa: PLC0415
        _angular_distance_deg_vectorized,
        _pixel_distance_deg_vectorized,
    )

    out_of_frame = int(len(vt) - int(in_frame.sum()))
    no_catalog_id = 0
    matched_rows: list[dict] = []
    matched_vt_idx: set[Any] = set()
    excluded_rows: list[dict[str, Any]] = []

    def _excluded_target_row(vrow: pd.Series, reason: str, *, mag: float | None = None) -> dict[str, Any]:
        mag_val = mag
        if mag_val is None:
            mag_val = float(pd.to_numeric(vrow.get("mag", float("nan")), errors="coerce"))
        return {
            "name": str(vrow.get("name", "") or ""),
            "vsx_name": str(vrow.get("vsx_name", "") or ""),
            "vsx_type": str(vrow.get("vsx_type", "") or ""),
            "ra_deg": float(vrow.get("ra_deg", float("nan"))),
            "dec_deg": float(vrow.get("dec_deg", float("nan"))),
            "mag": mag_val,
            "reason": reason,
        }

    for _, vrow_off in vt.loc[~in_frame].iterrows():
        excluded_rows.append(_excluded_target_row(vrow_off, "out_of_frame"))

    for vidx, vrow in vt_in.iterrows():
        ra_v = float(vrow["ra_deg"])
        dec_v = float(vrow["dec_deg"])
        if not (math.isfinite(ra_v) and math.isfinite(dec_v)):
            continue
        # Nájdi najbližší záznam v masterstars
        if _use_pixel_dist:
            x_v = float(vrow["x"])
            y_v = float(vrow["y"])
            if not (math.isfinite(x_v) and math.isfinite(y_v)):
                continue
            dists = _pixel_distance_deg_vectorized(
                x_v,
                y_v,
                ms_x_arr,
                ms_y_arr,
                plate_scale_arcsec=_plate_nominal,
            )
        else:
            dists = _angular_distance_deg_vectorized(ra_v, dec_v, ms_arr[:, 0], ms_arr[:, 1])
        best_idx = int(np.argmin(dists))
        best_dist_arcsec = dists[best_idx] * 3600.0
        if best_dist_arcsec > match_radius_arcsec:
            continue
        ms_row = ms.iloc[best_idx]
        zone_val_raw = str(ms_row.get("zone", "")).strip()
        zone_flag = _active_target_zone_flag(ms_row, zone_val_raw)
        # Preferuj "name" (často obsahuje presný Gaia source_id aj keď catalog_id je poškodený float64).
        name_raw = ms_row.get("name", "")
        name_norm = normalize_gaia_source_id(name_raw)
        if name_norm and re.fullmatch(r"\d{12,22}", str(name_norm)):
            catalog_id_norm = str(name_norm)
            cid_raw = str(name_raw).strip()
        else:
            cid_raw = str(ms_row.get("catalog_id", ms_row.get("name", ""))).strip()
            catalog_id_norm = _normalize_gaia_id(ms_row.get("catalog_id", ms_row.get("name")))
        if not catalog_id_norm:
            # Fallback na textový reťazec ak _normalize vráti prázdny ale máme nečíselný id
            catalog_id_norm = _gaia_id_str(cid_raw)
        if not catalog_id_norm:
            no_catalog_id += 1
            excluded_rows.append(_excluded_target_row(vrow, "no_catalog_id"))
            continue
        mag_for_skip = float(
            pd.to_numeric(
                ms_row.get("mag", ms_row.get("phot_g_mean_mag", float("nan"))),
                errors="coerce",
            )
        )
        _snr_raw = ms_row.get("snr50_ok", True)
        if _snr_raw is None or (isinstance(_snr_raw, float) and not math.isfinite(float(_snr_raw))):
            snr50_ok_for_skip = True
        else:
            snr50_ok_for_skip = bool(_bool_col(pd.Series([_snr_raw])).iloc[0])
        if (not snr50_ok_for_skip) and math.isfinite(mag_for_skip) and mag_for_skip < 8.0:
            logging.info(
                "[SKIP] %s: mag=%.1f snr50_ok=False "
                "— pravdepodobne saturovaná, skip",
                catalog_id_norm,
                mag_for_skip,
            )
            excluded_rows.append(_excluded_target_row(vrow, "saturated", mag=mag_for_skip))
            continue
        skip_ph = zone_flag == "saturated"
        rec = {
            "name": vrow.get("name", ""),
            "vsx_name": vrow.get("vsx_name", ""),
            "vsx_type": vrow.get("vsx_type", ""),
            "vsx_period": vrow.get("vsx_period", ""),
            "priority": vrow.get("priority", 1),
            "ra_deg": ra_v,
            "dec_deg": dec_v,
            "x": float(vrow["x"]),
            "y": float(vrow["y"]),
            "catalog_id": catalog_id_norm,
            # Prefer image-matched magnitude; fallback to Gaia G if masterstars carries it.
            "mag": float(
                pd.to_numeric(
                    ms_row.get("mag", ms_row.get("phot_g_mean_mag", float("nan"))),
                    errors="coerce",
                )
            ),
            "b_v": float(ms_row.get("b_v", float("nan"))),
            "bp_rp": float(ms_row.get("bp_rp", float("nan"))),
            "zone_flag": zone_flag,
            "skip_photometry": bool(skip_ph),
        }
        for _exo_col in (
            "exo_host_obj_id",
            "exo_host_name",
            "exo_cat_source",
            "exo_disposition",
            "exo_match_sep_arcsec",
            "target_origin",
        ):
            if _exo_col in vrow.index:
                rec[_exo_col] = vrow.get(_exo_col, "")
        matched_vt_idx.add(vidx)
        matched_rows.append(rec)

    _empty_cols = [
        "name",
        "vsx_name",
        "vsx_type",
        "vsx_period",
        "priority",
        "ra_deg",
        "dec_deg",
        "x",
        "y",
        "catalog_id",
        "mag",
        "b_v",
        "bp_rp",
        "zone_flag",
        "skip_photometry",
    ]
    n_excluded_no_dao_match = int((~vt_in.index.isin(matched_vt_idx)).sum())
    for vidx, vrow in vt_in.loc[~vt_in.index.isin(matched_vt_idx)].iterrows():
        excluded_rows.append(_excluded_target_row(vrow, "no_dao_gaia_match"))
    if n_excluded_no_dao_match:
        _ex_names = [
            str(vt_in.loc[i].get("vsx_name") or vt_in.loc[i].get("name") or "")
            for i in vt_in.index
            if i not in matched_vt_idx
        ]
        _ex_names = [n for n in _ex_names if n]
        _preview = ", ".join(_ex_names[:30])
        if len(_ex_names) > 30:
            _preview += ", ..."
        logging.info(
            "[Faza 0] Excluded %d VSX targets without masterstar (DAO+Gaia) match - not in active_targets: %s",
            n_excluded_no_dao_match,
            _preview,
        )
    if not matched_rows:
        LAST_EXCLUDED_TARGETS = (
            pd.DataFrame(excluded_rows)
            if excluded_rows
            else pd.DataFrame(columns=["name", "vsx_name", "vsx_type", "ra_deg", "dec_deg", "mag", "reason"])
        )
        log_event(
            "select_active_targets: linear=0 noisy1=0 noisy2=0 noisy3=0 saturated=0 "
            f"no_catalog_id={no_catalog_id} out_of_frame={out_of_frame}"
        )
        return pd.DataFrame(columns=_empty_cols)

    result = pd.DataFrame(matched_rows) if matched_rows else pd.DataFrame(columns=_empty_cols)
    if "catalog_id" in result.columns:
        # NEPOUŽÍVAŤ float() (precision loss). Použi robustnú normalizáciu.
        result["catalog_id"] = result["catalog_id"].apply(_normalize_gaia_id)
    # Gaia BP-RP + rovnaká B-V hierarchia ako pre comp (nesmie prepísať všetky b_v z NaN bp_rp v masterstars).
    result = _enrich_active_targets_bp_rp(
        result,
        gaia_db_path=gaia_db_path,
    )
    # Deduplicate by catalog_id — keep row with real VSX name over
    # Gaia-placeholder (e.g. "V0842 Her" preferred over
    # "Gaia DR3 1400549806859236864")
    if "catalog_id" in result.columns:
        # Prefer rows where vsx_name / name does NOT start with "Gaia DR3"
        _is_gaia_placeholder = (
            result.get("vsx_name", result.get("name", pd.Series(dtype=str)))
            .astype(str)
            .str.startswith("Gaia DR3")
        )
        # Sort: non-Gaia-placeholder first, then keep first per catalog_id
        result = (
            result
            .assign(_gaia_placeholder=_is_gaia_placeholder.astype(int))
            .sort_values("_gaia_placeholder")
            .drop_duplicates(subset=["catalog_id"], keep="first")
            .drop(columns=["_gaia_placeholder"])
            .reset_index(drop=True)
        )
        log_event(
            f"select_active_targets: deduped to {len(result)} unique "
            f"catalog_ids (prefer real VSX name over Gaia placeholder)"
        )
    n_lin = int((result["zone_flag"] == "linear").sum())
    n_n1 = int((result["zone_flag"] == "noisy1").sum())
    n_n2 = int((result["zone_flag"] == "noisy2").sum())
    n_n3 = int((result["zone_flag"] == "noisy3").sum())
    n_sat = int((result["zone_flag"] == "saturated").sum())
    log_event(
        f"select_active_targets: linear={n_lin} noisy1={n_n1} noisy2={n_n2} noisy3={n_n3} "
        f"saturated={n_sat} no_catalog_id={no_catalog_id} out_of_frame={out_of_frame} "
        f"excluded_no_dao_match={n_excluded_no_dao_match}"
    )
    logging.info(
        f"[FÁZA 0] active_targets: {len(result)} / {len(vt)} VSX hviezd "
        f"(in_frame={int(in_frame.sum())}, masterstar_matched={len(matched_rows)}, excluded_no_dao_match={n_excluded_no_dao_match})"
    )
    LAST_EXCLUDED_TARGETS = (
        pd.DataFrame(excluded_rows)
        if excluded_rows
        else pd.DataFrame(columns=["name", "vsx_name", "vsx_type", "ra_deg", "dec_deg", "mag", "reason"])
    )
    result = _ensure_active_target_display_names(result)
    return result.reset_index(drop=True)


def _batch_enrich_targets_bp_rp_from_gaia_db(
    target_cids: list[str],
    gaia_db_path: str,
) -> dict[str, dict[str, Any]]:
    """Prefetch Gaia ``bp_rp`` / ``teff_gspphot`` for Phase 1 targets (batched SQL)."""
    gdb = str(gaia_db_path or "").strip()
    if not target_cids or not gdb:
        return {}
    try:
        gp = Path(gdb).expanduser().resolve()
        if not gp.is_file():
            return {}
    except OSError:
        return {}

    ids_norm: list[str] = []
    seen: set[str] = set()
    for raw in target_cids:
        g = normalize_gaia_source_id(raw)
        if not g or not g.isdigit() or g in seen:
            continue
        seen.add(g)
        ids_norm.append(g)
    if not ids_norm:
        return {}

    out: dict[str, dict[str, Any]] = {}
    try:
        base = query_local_gaia_by_source_ids(gp, ids_norm)
        for k, v in base.items():
            out[k] = {
                "bp_rp": v.get("bp_rp"),
                "teff_gspphot": None,
            }
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[PHASE 1] Batch Gaia bp_rp lookup failed: %s", exc)
        return {}

    try:
        import sqlite3  # noqa: PLC0415

        con = sqlite3.connect(str(gp))
        con.row_factory = sqlite3.Row
        try:
            cols = {
                str(r[1]).strip().lower()
                for r in con.execute("PRAGMA table_info('gaia_dr3')").fetchall()
            }
            if "teff_gspphot" not in cols:
                return out
            ids_int = [int(x) for x in ids_norm]
            bs = 500
            for i0 in range(0, len(ids_int), bs):
                chunk = ids_int[i0 : i0 + bs]
                ph = ",".join("?" * len(chunk))
                q = f"SELECT source_id, teff_gspphot FROM gaia_dr3 WHERE source_id IN ({ph});"
                for row in con.execute(q, chunk):
                    key = normalize_gaia_source_id(row["source_id"])
                    if not key or key not in out:
                        continue
                    te = row["teff_gspphot"]
                    if te is not None:
                        try:
                            out[key]["teff_gspphot"] = float(te)
                        except (TypeError, ValueError):
                            pass
        finally:
            con.close()
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[PHASE 1] Batch Gaia teff lookup failed: %s", exc)

    return out


def _enrich_target_bp_rp_from_gaia_db(
    target: pd.Series,
    *,
    gaia_db_path: str,
    vsx_local_db_path: str | None = None,
    gaia_prefetch: dict[str, dict[str, Any]] | None = None,
) -> pd.Series:
    """Doplň ``bp_rp`` pre jeden active target (Fáza 1) z Gaia podľa ``source_id``."""
    out = target.copy()
    vsx = str(out.get("vsx_name", "") or "").strip() or str(out.get("name", "") or "").strip() or "?"

    def _fscalar(key: str) -> float:
        try:
            v = float(pd.to_numeric(out.get(key), errors="coerce"))
        except Exception:  # noqa: BLE001
            return float("nan")
        return v if math.isfinite(v) else float("nan")

    bpr_ms = _fscalar("bp_rp")

    gid = normalize_gaia_source_id(out.get("catalog_id"))
    gdb = str(gaia_db_path or "").strip()
    try:
        gp = Path(gdb).expanduser().resolve()
        gdb_ok = bool(gdb) and gp.is_file()
    except OSError:
        gdb_ok = False

    bpr_nf = float("nan")
    _prefetched = bool(gaia_prefetch and gid and gid in gaia_prefetch)
    if _prefetched:
        pf = gaia_prefetch[gid]  # type: ignore[index]
        try:
            vbp = pf.get("bp_rp")
            if vbp is not None and math.isfinite(float(vbp)):
                bpr_nf = float(vbp)
        except (TypeError, ValueError):
            pass
    elif gid and gdb_ok and gid.isdigit():
        try:
            import sqlite3  # noqa: PLC0415

            con = sqlite3.connect(str(gp))
            con.row_factory = sqlite3.Row
            try:
                cols = {str(r[1]).strip().lower() for r in con.execute("PRAGMA table_info('gaia_dr3')").fetchall()}
                parts = [c for c in ("bp_rp", "teff_gspphot") if c in cols]
                if parts:
                    rw = con.execute(
                        f"SELECT {', '.join(parts)} FROM gaia_dr3 WHERE source_id=? LIMIT 1;",
                        (int(gid),),
                    ).fetchone()
                    if rw is not None:
                        if "bp_rp" in parts:
                            try:
                                vbp = rw["bp_rp"]
                                if vbp is not None:
                                    bpr_nf = float(vbp)
                            except (TypeError, ValueError, KeyError, IndexError):
                                pass
            finally:
                con.close()
        except Exception as exc:  # noqa: BLE001
            logging.error('[EXC-0202] Outer Gaia SQL for target bp_rp logs failure - target keeps CSV bp_rp or NaN: %s', exc)
            log_event(f"TARGET Gaia SQL: {vsx} — {exc!s}")

    if math.isfinite(bpr_nf):
        out["bp_rp"] = float(bpr_nf)
    elif math.isfinite(bpr_ms):
        out["bp_rp"] = float(bpr_ms)
    elif not gid:
        log_event(f"TARGET bp_rp: {vsx} — bez platného Gaia catalog_id")
    return out


def _bprp_tier_ladder_for_selection(
    cfg: AppConfig | None,
    max_delta_bprp: float,
) -> list[float]:
    """Tier-ladder colour windows: tier1 -> tier2 -> tier3 -> comp_max_delta_bprp cap."""
    if cfg is not None:
        raw = [
            float(getattr(cfg, "comp_tier1_bprp_limit", 0.15)),
            float(getattr(cfg, "comp_tier2_bprp_limit", 0.30)),
            float(getattr(cfg, "comp_tier3_bprp_limit", 0.55)),
            float(getattr(cfg, "comp_max_delta_bprp", max_delta_bprp)),
        ]
    else:
        raw = [float(max_delta_bprp)]
    out: list[float] = []
    for v in raw:
        if math.isfinite(v) and v > 0 and v not in out:
            out.append(float(v))
    return out or [float(max_delta_bprp)]


def _select_comps_by_color_then_rms(
    candidates: pd.DataFrame,
    target_bprp: float,
    n_comp_min: int,
    n_comp_max: int,
    max_delta_bprp: float = 0.5,
    *,
    cfg: AppConfig | None = None,
) -> pd.DataFrame:
    """
    Stupeň 1: farebný filter (|ΔBP-RP|) — tier ladder widen ak < n_comp_min
    Stupeň 2: rank by comp_rms ASC (Broeg 1/rms equivalent)
    Drop comp_rms < comp_select_rms_floor (isolated_bin artefact).
    """
    if candidates is None or getattr(candidates, "empty", True):
        return pd.DataFrame()

    if "comp_rms" not in candidates.columns:
        raise ValueError("_select_comps_by_color_then_rms requires comp_rms column")

    _floor = 1e-6
    if cfg is not None:
        try:
            _floor = float(getattr(cfg, "comp_select_rms_floor", 1e-6) or 1e-6)
        except (TypeError, ValueError):
            _floor = 1e-6

    def _apply_rms_floor(df: pd.DataFrame) -> pd.DataFrame:
        rms = pd.to_numeric(df["comp_rms"], errors="coerce")
        return df[rms >= _floor].copy()

    def _rank_by_rms(df: pd.DataFrame) -> pd.DataFrame:
        ranked = df.copy()
        ranked["_broeg_score"] = ranked["comp_rms"].apply(
            lambda r: 1.0 / r if np.isfinite(r) and r > 0 else 0.0
        )
        id_col = "catalog_id" if "catalog_id" in ranked.columns else ranked.columns[0]
        return ranked.sort_values(
            ["_broeg_score", id_col], ascending=[False, True], kind="mergesort"
        )

    if not np.isfinite(float(target_bprp)):
        ranked = _rank_by_rms(_apply_rms_floor(candidates.copy()))
        return ranked.head(int(n_comp_max))

    out = candidates.copy()
    out["_delta_bprp_abs"] = (pd.to_numeric(out.get("bp_rp"), errors="coerce") - float(target_bprp)).abs()
    out = _apply_rms_floor(out)

    thresholds = _bprp_tier_ladder_for_selection(cfg, max_delta_bprp)
    first_thr = float(thresholds[0]) if thresholds else float(max_delta_bprp)

    selected = pd.DataFrame()
    used_threshold: float | None = None

    for thr in thresholds:
        pool = out[out["_delta_bprp_abs"] <= float(thr)]
        if int(len(pool)) >= int(n_comp_min):
            selected = pool
            used_threshold = float(thr)
            break

    if selected.empty:
        selected = out
        used_threshold = float(thresholds[-1]) if thresholds else float(max_delta_bprp)

    result = _rank_by_rms(selected).head(int(n_comp_max))

    if used_threshold is not None and used_threshold > first_thr:
        log_event(
            f"[COMP] color filter relaxed to delta_bprp<={float(used_threshold):.2f} "
            f"(target_bprp={float(target_bprp):.3f}, n={int(len(result))})"
        )

    return result


def _select_comps_tiered(
    candidates: pd.DataFrame,
    n_comp_min: int,
    n_comp_max: int,
    tier_weights: dict[int, float],
) -> tuple[pd.DataFrame, str]:
    """
    Vracia (selected_df, selection_note)

    Greedy tier-based výber:
    T1 → T2 → T3 → T4 (len ak treba)
    Nikdy nemiešaj T3/T4 ak T1+T2 >= n_comp_min.
    Sort: comp_tier ASC, comp_rms ASC, catalog_id (proximity only via max_dist_deg gate).
    """
    _ = tier_weights  # reserved for future (selection is tier/rms-only; weights affect Phase 2A)
    if candidates is None or getattr(candidates, "empty", True):
        return pd.DataFrame(), "no_candidates"

    if "comp_tier" not in candidates.columns or "comp_rms" not in candidates.columns:
        return pd.DataFrame(), "missing_cols"

    selected = pd.DataFrame()
    note = "ok"

    for max_tier in [1, 2, 3, 4]:
        pool = candidates[candidates["comp_tier"] <= max_tier].copy()
        pool["comp_tier"] = pd.to_numeric(pool["comp_tier"], errors="coerce").fillna(4).astype(int)
        pool["comp_rms"] = pd.to_numeric(pool["comp_rms"], errors="coerce")

        # Zoraď: tier ASC, potom comp_rms ASC, potom catalog_id (stable tiebreak)
        pool = pool.sort_values(
            ["comp_tier", "comp_rms", "catalog_id"],
            ascending=[True, True, True],
            kind="mergesort",
        )

        # Ber max n_comp_max (vždy)
        selected = pool.head(int(n_comp_max))

        n_t1t2 = len(selected[selected["comp_tier"] <= 2])

        if len(selected) >= int(n_comp_min):
            # Máme dostatok — ale over:
            # ak máme >= n_comp_min z T1+T2, odober T3/T4 z výberu
            if n_t1t2 >= int(n_comp_min):
                selected = (
                    selected[selected["comp_tier"] <= 2]
                    .sort_values(
                        ["comp_tier", "comp_rms", "catalog_id"],
                        ascending=[True, True, True],
                        kind="mergesort",
                    )
                    .head(int(n_comp_max))
                )
                if max_tier == 1:
                    note = "t1_only"
                elif max_tier == 2:
                    note = "t1t2"
                else:
                    note = "t1t2"  # T3/T4 boli odobrané
            else:
                # T3/T4 boli potrebné — selected už obsahuje až n_comp_max
                if max_tier == 3:
                    note = "t3_fallback"
                else:
                    note = "t4_fallback"
            break
    else:
        note = "sparse"

    if len(selected) == 0:
        note = "sparse_no_comps"

    return selected.reset_index(drop=True), note


def build_global_comp_pool(
    masterstars_df: pd.DataFrame,
    per_frame_csv_paths: list[Path],
    csv_cache: dict[str, pd.DataFrame],
    variable_target_catalog_ids: AbstractSet[str] | None,
    safe_bbox: tuple[float, float, float, float] | None,
    chip_fw: int,
    chip_fh: int,
    chip_interior_margin_px: int,
    max_comp_rms: float,
    cfg: AppConfig,
    *,
    flux_col: str = "dao_flux",
    min_frames_frac: float = 0.3,
    fwhm_px: float = 3.7,
    max_psf_chi2: float = 3.0,
    max_fwhm_factor: float = 1.5,
    edge_bad_frame_frac_max: float = 0.10,
) -> pd.DataFrame:
    """Zostav globálny comp pool — statické filtre + RMS naprieč framami (raz pre pole)."""
    pool = masterstars_df.copy()
    for _id_col in ("catalog_id", "name"):
        if _id_col in pool.columns:
            pool[_id_col] = _normalize_id_series(pool[_id_col])
    for col in (
        "is_usable",
        "is_saturated",
        "is_noisy",
        "snr50_ok",
        "vsx_known_variable",
        "likely_saturated",
    ):
        if col in pool.columns:
            pool[col] = _bool_col(pool[col])

    margin = int(chip_interior_margin_px)
    if "x" not in pool.columns or "y" not in pool.columns:
        logging.warning("[GLOBAL COMP POOL] chýbajú x/y — prázdny pool")
        return pd.DataFrame()
    xn = pd.to_numeric(pool["x"], errors="coerce")
    yn = pd.to_numeric(pool["y"], errors="coerce")

    if safe_bbox is not None:
        # safe_bbox already shrinks by alignment intersection + sky annulus (r_out); do not inset again.
        x0, y0, x1, y1 = safe_bbox
        if float(x1) > float(x0) and float(y1) > float(y0):
            pool = pool.loc[
                xn.between(float(x0), float(x1)) & yn.between(float(y0), float(y1))
            ].copy()
        else:
            pool = pool.iloc[0:0].copy()
    else:
        fw, fh = int(chip_fw), int(chip_fh)
        if margin > 0 and fw > 2 * margin and fh > 2 * margin:
            pool = pool.loc[
                xn.between(float(margin), float(fw - margin)) & yn.between(float(margin), float(fh - margin))
            ].copy()

    _vt_gaia_ids: frozenset[str] | None = None
    if variable_target_catalog_ids:
        from gaia_catalog_id import normalize_gaia_id_set  # noqa: PLC0415

        _vt_gaia_ids = normalize_gaia_id_set(
            variable_target_catalog_ids,
            log_label="variable_target_catalog_ids (global comp pool)",
        ) or None
    if _vt_gaia_ids:
        nid = pool.get("catalog_id", pool.get("name", pd.Series("", index=pool.index))).map(_normalize_gaia_id)
        pool = pool.loc[~nid.isin(_vt_gaia_ids)].copy()

    if "zone" in pool.columns:
        z = pool["zone"].astype(str).str.strip().str.lower()
        pool = pool.loc[~z.isin(["saturated", "nonlinear"])].copy()

    cand_mask = (
        _bool_col(pool.get("is_usable", pd.Series(True, index=pool.index)))
        & ~_bool_col(pool.get("is_saturated", pd.Series(False, index=pool.index)))
        & ~_bool_col(pool.get("is_noisy", pd.Series(False, index=pool.index)))
        & ~_bool_col(pool.get("vsx_known_variable", pd.Series(False, index=pool.index)))
        & ~_bool_col(pool.get("likely_saturated", pd.Series(False, index=pool.index)))
    )
    pool = pool.loc[cand_mask].copy()

    if bool(cfg.phase01_comparison_exclude_gaia_nss) and "gaia_nss" in pool.columns:
        pool = pool.loc[~_bool_col(pool["gaia_nss"])].copy()
    if bool(cfg.phase01_comparison_exclude_gaia_extobj):
        for _ext_col in ("gaia_qso", "gaia_gal"):
            if _ext_col in pool.columns:
                pool = pool.loc[~_bool_col(pool[_ext_col])].copy()

    if pool.empty:
        logging.warning("[GLOBAL COMP POOL] po statických filtroch 0 riadkov")
        return pool.reset_index(drop=True)

    id_col = "name" if "name" in pool.columns else "catalog_id"
    cand_ids = {str(x).strip() for x in pool[id_col].tolist() if str(x).strip()}
    if not cand_ids:
        return pool.reset_index(drop=True)

    rms_map = compute_global_pool_rms_map(
        cand_ids=cand_ids,
        _masterstars_df=masterstars_df,
        per_frame_csv_paths=per_frame_csv_paths,
        csv_cache=csv_cache,
        flux_col=flux_col,
        min_frames_frac=min_frames_frac,
        edge_bad_frame_frac_max=edge_bad_frame_frac_max,
        max_psf_chi2=max_psf_chi2,
        max_fwhm_factor=max_fwhm_factor,
        fwhm_px=fwhm_px,
        chip_fw=chip_fw,
        chip_fh=chip_fh,
        max_comp_rms=max_comp_rms,
        apply_rms_prefilter=True,
    )
    pool = attach_comp_rms_to_pool_rows(pool, rms_map, id_col=id_col)
    _before_dedupe = int(len(pool))
    pool = _dedupe_comp_pool_by_gaia_key(pool)
    if int(len(pool)) < _before_dedupe:
        logging.info(
            "[GLOBAL COMP POOL] deduped Gaia catalog_id: %d → %d rows",
            _before_dedupe,
            int(len(pool)),
        )
    logging.info(
        "[GLOBAL COMP POOL] %d kandidátov (z %d masterstars, po filtroch)",
        len(pool),
        len(masterstars_df),
    )
    # P2 determinism: canonical row order
    if "catalog_id" in pool.columns:
        pool = pool.sort_values("catalog_id", kind="mergesort").reset_index(drop=True)
    return pool


def _dedupe_comp_pool_by_gaia_key(pool: pd.DataFrame) -> pd.DataFrame:
    """One row per Gaia ``catalog_id`` (fallback ``name``); keep lowest ``comp_rms`` when duplicated."""
    if pool is None or getattr(pool, "empty", True):
        return pool if pool is not None else pd.DataFrame()
    out = pool.copy()
    id_src = out.get("catalog_id", out.get("name", pd.Series("", index=out.index)))
    out["_gaia_key"] = id_src.map(_normalize_gaia_id)
    out = out[out["_gaia_key"].astype(str).str.strip() != ""]
    if out.empty:
        return out.reset_index(drop=True)
    sort_cols = ["_gaia_key"]
    ascending = [True]
    if "comp_rms" in out.columns:
        sort_cols.append("comp_rms")
        ascending.append(True)
    if "catalog_id" in out.columns:
        sort_cols.append("catalog_id")
        ascending.append(True)
    out = out.sort_values(sort_cols, ascending=ascending, kind="mergesort")
    out = out.drop_duplicates(subset=["_gaia_key"], keep="first").drop(columns=["_gaia_key"])
    return out.reset_index(drop=True)


def _warn_zero_compstars_edge(
    *,
    target_cid: str,
    target: pd.Series,
    chip_fw: int | None,
    chip_fh: int | None,
    chip_interior_margin_px: int,
) -> None:
    """Pri neúspešnom výbere comp (0 riadkov) — ak je cieľ blízko vnútorného okraja čipu, doplní kontext."""
    try:
        tx = float(pd.to_numeric(target.get("x"), errors="coerce"))
        ty = float(pd.to_numeric(target.get("y"), errors="coerce"))
    except Exception:  # noqa: BLE001
        tx = ty = float("nan")
    if not (math.isfinite(tx) and math.isfinite(ty)):
        logging.warning("[COMP] %s: 0 comp stars", target_cid)
        return
    m = int(chip_interior_margin_px)
    if chip_fw is None or chip_fh is None or m <= 0:
        logging.warning("[COMP] %s: 0 comp stars", target_cid)
        return
    wf = float(int(chip_fw))
    hf = float(int(chip_fh))
    if wf <= 2.0 * float(m) or hf <= 2.0 * float(m):
        logging.warning("[COMP] %s: 0 comp stars", target_cid)
        return
    xmin = float(m)
    ymin = float(m)
    xmax = wf - float(m)
    ymax = hf - float(m)
    dist = min(tx - xmin, xmax - tx, ty - ymin, ymax - ty)
    if math.isfinite(dist) and dist < 100.0:
        logging.warning(
            "[COMP] %s: 0 comp stars, target je %.0fpx od okraja bbox "
            "(edge position — geometricky obmedzené pole)",
            target_cid,
            float(dist),
        )
    else:
        logging.warning("[COMP] %s: 0 comp stars", target_cid)


def _count_gate_passing_comps(
    result: pd.DataFrame | None,
    per_target_rms_map: dict[str, float] | None,
    max_comp_rms: float,
    id_col: str,
) -> int:
    """Count comps in ``result`` whose per-target ``comp_rms`` passes the gate.

    N_good = comps passing the colour ladder + per-target ``max_comp_rms`` gate.
    The per-target gate is authoritative: a comp with ``comp_rms > max_comp_rms``
    is never counted as good, so routing never treats an above-gate comp as a
    usable default comp (known-issue (b) fix). When the gate is disabled
    (non-finite / <= 0) fall back to the raw row count.
    """
    if result is None or getattr(result, "empty", True):
        return 0
    if not (math.isfinite(max_comp_rms) and max_comp_rms > 0):
        return int(len(result))
    if id_col not in result.columns:
        return int(len(result))
    _map = per_target_rms_map or {}
    n_good = 0
    for _rid in result[id_col].astype(str).str.strip():
        _v = _map.get(_rid, _map.get(str(_rid), float("nan")))
        try:
            _vf = float(_v)
        except (TypeError, ValueError):
            _vf = float("nan")
        if math.isfinite(_vf) and _vf <= float(max_comp_rms):
            n_good += 1
    return n_good


def select_comparison_stars_per_target(
    target: pd.Series,
    masterstars_df: pd.DataFrame,
    per_frame_csv_paths: list[Path],
    *,
    csv_cache: dict[str, pd.DataFrame] | None = None,
    global_comp_pool_df: pd.DataFrame | None = None,
    fwhm_px: float = 3.7,
    max_dist_deg: float = 1.0,
    max_mag_diff: float = 0.25,  # ±0.25 mag od targetu (základ; pri jasnom ciele viď ``mag_tol`` nižšie)
    max_mag_diff_t1: float = 0.50,
    max_mag_diff_t2: float = 1.00,
    max_mag_diff_t3: float = 1.50,
    max_mag_diff_t4: float = 2.00,
    n_comp_min: int = 3,
    n_comp_max: int = 7,
    max_comp_rms: float = 0.05,
    min_dist_arcsec: float = 60.0,
    min_frames_frac: float = 0.3,
    rms_outlier_sigma: float = 3.0,
    exclude_gaia_nss: bool = True,
    exclude_gaia_extobj: bool = True,
    mag_bright_threshold: float = 12.0,
    max_mag_diff_bright_floor: float = 0.0,
    max_psf_chi2: float = 3.0,
    max_fwhm_factor: float = 1.5,
    isolation_radius_px: float = 25.0,
    flux_col: str = "dao_flux",
    chip_fw: int | None = None,
    chip_fh: int | None = None,
    chip_interior_margin_px: int = 0,
    edge_bad_frame_frac_max: float = 0.10,
    max_delta_bprp: float = 0.5,
    vsx_local_db_path: str | None = None,
    gaia_db_path: str | None = None,
    gaia_prefetch: dict[str, dict[str, Any]] | None = None,
    variable_target_catalog_ids: AbstractSet[str] | None = None,
    cfg: AppConfig | None = None,
    plate_scale_arcsec: float = 1.3,
    use_pixel_dist: bool = False,
    gs11_comp_rejects_acc: list[int] | None = None,
    _selection_mode: str = "auto",
) -> pd.DataFrame:
    """Fáza 1: Pre jeden target vyber najstabilnejšie porovnávacie hviezdy.

    Postup (Možnosť D = B + C):
    1. Priestorový + fotometrický filter kandidátov z masterstars
    2. Načítaj flux zo všetkých per-frame CSV (len _PHASE_USECOLS_PERFRAME)
    3. Normalizuj flux voči ensemble mediánu per snímka
    4. Vypočítaj RMS scatter pre každého kandidáta
    5. Iteratívny ensemble filter — vyraď top outlierov kým RMS neklesá
    6. Vráť top n_comp_max najstabilnejších (min n_comp_min)

    Args:
        exclude_gaia_nss: Vylúč Gaia non-single stars (binárky, vizuálne dvojhviezdy).
            Tieto majú variabilný flux nezávislý od počasia → scatter comp hviezdy.
        exclude_gaia_extobj: Vylúč Gaia QSO a galaxie (gaia_qso, gaia_gal).
            Nie sú bodové zdroje → systematické chyby v aperturnej fotometrii.
        max_psf_chi2: Maximálny mediánový PSF chi² kandidáta cez všetky snímky.
            Vysoké chi² = profil nie je čistý Gaussian = blend alebo rozšírený zdroj.
            Použije sa len ak je stĺpec psf_chi2 dostupný v per-frame CSV.
            Nastavenie na float("inf") filter vypne.
        max_fwhm_factor: Maximálny pomer fwhm_estimate_px kandidáta voči mediánu
            všetkých hviezd na snímke. Hodnota > 1.5 indikuje blend dvoch blízkych
            hviezd. Použije sa len ak je stĺpec fwhm_estimate_px dostupný.
            Nastavenie na float("inf") filter vypne.
        isolation_radius_px: Polomer v pixeloch pre výpočet contamination indexu.
            Súčet flux susedov / flux kandidáta v tomto polomere = contamination.
            Výsledok vstupuje do combined score (soft penalizácia, nie hard exclusion).
            Nastavenie na 0.0 vypne crowding penalizáciu úplne.
        max_comp_rms: Maximálny povolený p2p RMS scatter comp hviezdy (mag).
            Hviezdy s RMS > max_comp_rms sú odmietnuté bez ohľadu na ranking.
            Default 0.05 mag (50 ppt) — štandardná fotometrická stabilita.
        min_dist_arcsec: Minimálna vzdialenosť comp hviezdy od targetu v oblúkových
            sekundách. Zabraňuje PSF overlap pri veľmi blízkych hviezdach.
            Default 60 arcsec (ochrana aj proti lokálnym artefaktom okolo targetu).
        mag_bright_threshold: Hranica ``mag`` cieľa (rovnaký systém ako ``target["mag"]``),
            pod ktorou sa uplatní ``max_mag_diff_bright_floor`` (typicky jasné hviezdy ~9 mag).
        max_mag_diff_bright_floor: Minimálna šírka |Δmag| pri jasných cieľoch; ``0`` vypne.
        chip_fw / chip_fh / chip_interior_margin_px: spolu orežú kandidátov na comp hviezdy
            blízko okraja čipu (rovnaká logika ako Fáza 0 a suspected). ``chip_interior_margin_px=0`` = vypnuté.
        variable_target_catalog_ids: Gaia ``catalog_id`` zo ``variable_targets.csv`` — tieto hviezdy
            sa nikdy neponúknu ako porovnávačky (VSX premenné vrátane ``catalog_only``).

    Returns:
        DataFrame s porovnávacími hviezdami pre tento target, zoradený podľa RMS ASC.
        Prázdny DataFrame ak sa nenájde dostatok stabilných hviezd.
    """
    _ = fwhm_px
    from comp_selection_per_target import (  # noqa: PLC0415
        _accumulate_per_frame_comp_metrics,
        _apply_comp_metric_hard_filters,
        _assemble_comp_selection_result_rows,
        _assign_comp_tiers_to_pool,
        _bootstrap_phase1_csv_cache,
        _build_candidates_pre_adaptive_mag,
        _compute_comp_contamination_map,
        _detrend_and_compute_comp_rms_map,
        _ensemble_mad_filter_rms,
        _filter_comp_candidates_spatial_static,
        _iterative_ensemble_clip_cm_residual,
        _resolve_target_color_for_comp_selection,
        _score_comp_candidates_broeg,
    )

    from config import (  # noqa: PLC0415
        resolve_comp_sparse_fallback_enabled,
        resolve_comp_sparse_fallback_min,
    )

    _cfg_p1 = cfg if cfg is not None else AppConfig()
    _mode = str(_selection_mode or "auto").strip().lower()
    if _mode not in ("auto", "default", "sparse_fallback"):
        _mode = "auto"
    sparse_fallback = _mode == "sparse_fallback"

    def _retry_sparse_fallback() -> pd.DataFrame:
        if sparse_fallback or _mode != "auto":
            return pd.DataFrame()
        if not resolve_comp_sparse_fallback_enabled(_cfg_p1):
            return pd.DataFrame()
        return select_comparison_stars_per_target(
            target,
            masterstars_df,
            per_frame_csv_paths,
            csv_cache=csv_cache,
            global_comp_pool_df=global_comp_pool_df,
            fwhm_px=fwhm_px,
            max_dist_deg=max_dist_deg,
            max_mag_diff=max_mag_diff,
            max_mag_diff_t1=max_mag_diff_t1,
            max_mag_diff_t2=max_mag_diff_t2,
            max_mag_diff_t3=max_mag_diff_t3,
            max_mag_diff_t4=max_mag_diff_t4,
            n_comp_min=n_comp_min,
            n_comp_max=n_comp_max,
            max_comp_rms=max_comp_rms,
            min_dist_arcsec=min_dist_arcsec,
            min_frames_frac=min_frames_frac,
            rms_outlier_sigma=rms_outlier_sigma,
            exclude_gaia_nss=exclude_gaia_nss,
            exclude_gaia_extobj=exclude_gaia_extobj,
            mag_bright_threshold=mag_bright_threshold,
            max_mag_diff_bright_floor=max_mag_diff_bright_floor,
            max_psf_chi2=max_psf_chi2,
            max_fwhm_factor=max_fwhm_factor,
            isolation_radius_px=isolation_radius_px,
            flux_col=flux_col,
            chip_fw=chip_fw,
            chip_fh=chip_fh,
            chip_interior_margin_px=chip_interior_margin_px,
            edge_bad_frame_frac_max=edge_bad_frame_frac_max,
            max_delta_bprp=max_delta_bprp,
            vsx_local_db_path=vsx_local_db_path,
            gaia_db_path=gaia_db_path,
            gaia_prefetch=gaia_prefetch,
            variable_target_catalog_ids=variable_target_catalog_ids,
            cfg=cfg,
            plate_scale_arcsec=plate_scale_arcsec,
            use_pixel_dist=use_pixel_dist,
            gs11_comp_rejects_acc=gs11_comp_rejects_acc,
            _selection_mode="sparse_fallback",
        )

    if sparse_fallback:
        ms = masterstars_df.copy()
    elif global_comp_pool_df is not None and not getattr(global_comp_pool_df, "empty", True):
        ms = global_comp_pool_df.copy()
        if "comp_rms" in ms.columns:
            ms = ms.drop(columns=["comp_rms"])
    else:
        ms = masterstars_df.copy()
    for _id_col in ("catalog_id", "name"):
        if _id_col in ms.columns:
            ms[_id_col] = _normalize_id_series(ms[_id_col])
    for col in (
        "is_usable",
        "is_saturated",
        "is_noisy",
        "snr50_ok",
        "vsx_known_variable",
        "likely_saturated",
    ):
        if col in ms.columns:
            ms[col] = _bool_col(ms[col])

    ctx = _resolve_target_color_for_comp_selection(
        target,
        vsx_local_db_path=vsx_local_db_path,
        gaia_db_path=gaia_db_path,
        cfg=_cfg_p1,
    )
    ra_t = float(ctx["ra_t"])
    dec_t = float(ctx["dec_t"])
    mag_t = float(ctx["mag_t"])
    target_cid = str(ctx["target_cid"])
    t_bp_tgt = float(ctx["t_bp_tgt"])
    target_bprp_eff = float(ctx["target_bprp_eff"])
    max_delta_bprp_cfg = float(ctx["max_delta_bprp_cfg"])
    _individual_tier = ctx["_individual_tier"]
    _target_name = str(ctx["_target_name"])

    mag_tol = float(max_mag_diff)
    if (
        math.isfinite(mag_t)
        and float(max_mag_diff_bright_floor) > 0.0
        and mag_t < float(mag_bright_threshold)
    ):
        mag_tol = max(mag_tol, float(max_mag_diff_bright_floor))
        if mag_tol > float(max_mag_diff):
            logging.debug(
                "[FÁZA 1] Target %s: jasný cieľ (mag=%.2f < %.2f) → |Δmag| pás "
                "max(%.3f, floor %.3f) = %.3f",
                target_cid or "?",
                mag_t,
                float(mag_bright_threshold),
                float(max_mag_diff),
                float(max_mag_diff_bright_floor),
                mag_tol,
            )

    _x_t = float(pd.to_numeric(target.get("x"), errors="coerce"))
    _y_t = float(pd.to_numeric(target.get("y"), errors="coerce"))
    ms, _base_mask, det_mask = _filter_comp_candidates_spatial_static(
        ms,
        ra_t=ra_t,
        dec_t=dec_t,
        mag_t=mag_t,
        target_cid=target_cid,
        target_bprp_eff=target_bprp_eff,
        max_delta_bprp_cfg=max_delta_bprp_cfg,
        max_dist_deg=max_dist_deg,
        min_dist_arcsec=min_dist_arcsec,
        exclude_gaia_nss=exclude_gaia_nss,
        exclude_gaia_extobj=exclude_gaia_extobj,
        chip_fw=chip_fw,
        chip_fh=chip_fh,
        chip_interior_margin_px=int(chip_interior_margin_px),
        variable_target_catalog_ids=variable_target_catalog_ids,
        use_pixel_dist=bool(use_pixel_dist),
        x_t=_x_t if math.isfinite(_x_t) else None,
        y_t=_y_t if math.isfinite(_y_t) else None,
        plate_scale_arcsec=float(plate_scale_arcsec),
    )

    built = _build_candidates_pre_adaptive_mag(
        ms,
        _base_mask=_base_mask,
        det_mask=det_mask,
        mag_t=mag_t,
        target_cid=target_cid,
        mag_tol=mag_tol,
        max_mag_diff=max_mag_diff,
        n_comp_min=n_comp_min,
        chip_fw=chip_fw,
        chip_fh=chip_fh,
        chip_interior_margin_px=int(chip_interior_margin_px),
        target=target,
        cfg=_cfg_p1,
        sparse_fallback_mode=sparse_fallback,
    )
    if built is None:
        return _retry_sparse_fallback()
    candidates_pre, used_mag_tol = built

    if str(target_cid).strip() == "1498613634033133184":
        try:
            from comp_selection_per_target import BO_CVN_STEP_COUNTS  # noqa: PLC0415

            BO_CVN_STEP_COUNTS["C_mag_diff"] = int(len(candidates_pre))
        except Exception:  # noqa: BLE001
            # EXC-0203: T3 -- BO CVn BO_CVN_STEP_COUNTS debug counter not updated (EXCEPT-BULK-2 2026-07-08)
            pass

    if str(target_cid).strip() == "1498613634033133184":
        try:
            _dbg = candidates_pre.copy()
            if "_dist_deg" in _dbg.columns and "dist_arcsec" not in _dbg.columns:
                _dbg["dist_arcsec"] = pd.to_numeric(_dbg["_dist_deg"], errors="coerce") * 3600.0
            if "mag" not in _dbg.columns and "_mag" in _dbg.columns:
                _dbg["mag"] = pd.to_numeric(_dbg["_mag"], errors="coerce")
            # Limit columns to the requested view if available
            _cols = [c for c in ["catalog_id", "bp_rp", "mag", "dist_arcsec"] if c in _dbg.columns]
            print(
                f"[DEBUG BO CVn] candidates entering PERF-4B: {int(len(_dbg))} "
                f"(used_mag_tol={float(used_mag_tol):.2f})"
            )
            if _cols:
                print(_dbg[_cols].head(200).to_string(index=False))
        except Exception:  # noqa: BLE001
            # EXC-0204: T3 -- BO CVn candidates debug table print suppressed (EXCEPT-BULK-2 2026-07-08)
            pass

    _r_ap_iso = 7.0
    try:
        _fw = float(fwhm_px)
        if math.isfinite(_fw) and _fw > 0:
            _r_ap_iso = float(2.75 * _fw)
    except (TypeError, ValueError):
        _r_ap_iso = 7.0
    if not (math.isfinite(_r_ap_iso) and _r_ap_iso > 0):
        _r_ap_iso = 7.0
    try:
        ms_arr_x = pd.to_numeric(ms.get("x", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
        ms_arr_y = pd.to_numeric(ms.get("y", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
        if "_mag" in ms.columns:
            ms_arr_mag = pd.to_numeric(ms["_mag"], errors="coerce").to_numpy(dtype=float)
        else:
            ms_arr_mag = pd.to_numeric(ms.get("mag", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
    except Exception as _iso_exc:  # noqa: BLE001
        logging.warning(f"[FÁZA 1] Aperture izolácia preskočená (chyba): {_iso_exc!s}")
        ms_arr_x = ms_arr_y = ms_arr_mag = np.array([], dtype=float)

    id_col = (
        "name"
        if "name" in candidates_pre.columns
        else ("catalog_id" if "catalog_id" in candidates_pre.columns else "name")
    )
    cand_ids = set(candidates_pre[id_col].astype(str).str.strip())

    avail_cols = _PHASE_USECOLS_PERFRAME.copy()
    csv_cache = _bootstrap_phase1_csv_cache(
        per_frame_csv_paths,
        csv_cache,
        flux_col=flux_col,
        avail_cols=avail_cols,
    )
    metrics = _accumulate_per_frame_comp_metrics(
        per_frame_csv_paths,
        csv_cache,
        cand_ids,
        flux_col=flux_col,
        chip_fw=chip_fw,
        chip_fh=chip_fh,
    )
    flux_map = metrics["flux_map"]
    bjd_map = metrics.get("bjd_map") or {}
    n_frames_loaded = int(metrics["n_frames_loaded"])
    psf_chi2_map = metrics["psf_chi2_map"]
    fwhm_map = metrics["fwhm_map"]
    frame_fwhm_medians = metrics["frame_fwhm_medians"]
    peak_over_map = metrics["peak_over_map"]
    peak_total_map = metrics["peak_total_map"]
    snr_map = metrics["snr_map"]
    edge_bad_map = metrics["edge_bad_map"]
    edge_total_map = metrics["edge_total_map"]

    min_frames = max(3, int(n_frames_loaded * min_frames_frac))

    _dilution_map: dict[str, dict[str, Any]] | None = None
    _comp_gs11_notes: dict[str, str] = {}
    if bool(_cfg_p1.gs11_dilution_enabled) and gaia_db_path:
        from dilution import compute_dilution_factor  # noqa: PLC0415

        _ap_cfg = float(_cfg_p1.gs11_dilution_aperture_arcsec)
        if math.isfinite(_ap_cfg) and _ap_cfg > 0:
            _ap_arcsec = _ap_cfg
        else:
            _ap_r_px = 2.75 * float(fwhm_px) if math.isfinite(float(fwhm_px)) and float(fwhm_px) > 0 else 7.0
            _ap_arcsec = float(_ap_r_px) * float(plate_scale_arcsec)
        _dilution_map = {}
        for _, crow in candidates_pre.iterrows():
            _cid_d = str(crow.get(id_col, crow.get("catalog_id", "")) or "").strip()
            if not _cid_d:
                continue
            try:
                _ra_d = float(pd.to_numeric(crow.get("ra_deg", crow.get("ra")), errors="coerce"))
                _dec_d = float(pd.to_numeric(crow.get("dec_deg", crow.get("dec")), errors="coerce"))
            except (TypeError, ValueError):
                continue
            _gm_d = float("nan")
            for _gcol in ("phot_g_mean_mag", "mag", "_mag"):
                if _gcol in crow.index:
                    try:
                        _gv = float(pd.to_numeric(crow[_gcol], errors="coerce"))
                    except (TypeError, ValueError):
                        _gv = float("nan")
                    if math.isfinite(_gv):
                        _gm_d = _gv
                        break
            from dilution import _normalize_exclude_source_id  # noqa: PLC0415

            _dilution_map[_cid_d] = compute_dilution_factor(
                _ra_d,
                _dec_d,
                _gm_d,
                _ap_arcsec,
                str(gaia_db_path),
                catalog_id=_normalize_exclude_source_id(_cid_d),
                mag_limit_delta=float(_cfg_p1.gs11_dilution_mag_limit_delta),
            )

    flux_map, _b_rejected = _apply_comp_metric_hard_filters(
        flux_map,
        peak_over_map,
        peak_total_map,
        snr_map,
        psf_chi2_map,
        fwhm_map,
        frame_fwhm_medians,
        edge_bad_map,
        edge_total_map,
        target_cid=target_cid,
        edge_bad_frame_frac_max=edge_bad_frame_frac_max,
        max_psf_chi2=max_psf_chi2,
        max_fwhm_factor=max_fwhm_factor,
        dilution_map=_dilution_map,
        cfg=_cfg_p1,
        comp_quality_notes=_comp_gs11_notes,
    )
    if gs11_comp_rejects_acc is not None and _dilution_map:
        _max_d_gs11 = float(_cfg_p1.gs11_comp_max_dilution)
        for _cid_r in _b_rejected:
            _ent = _dilution_map.get(str(_cid_r), {})
            try:
                _d_r = float(_ent.get("dilution_factor", 1.0))
            except (TypeError, ValueError):
                _d_r = 1.0
            if math.isfinite(_d_r) and _d_r < _max_d_gs11:
                gs11_comp_rejects_acc[0] += 1

    contamination_map = _compute_comp_contamination_map(
        flux_map,
        ms,
        target_cid=target_cid,
        isolation_radius_px=isolation_radius_px,
    )

    _use_iter_clip = bool(sparse_fallback)
    _clip_sigma = float(getattr(_cfg_p1, "comp_clip_sigma", 5.0))

    rms_result = _detrend_and_compute_comp_rms_map(
        flux_map,
        min_frames=min_frames,
        max_comp_rms=max_comp_rms,
        n_comp_min=n_comp_min,
        target_cid=target_cid,
        target=target,
        chip_fw=chip_fw,
        chip_fh=chip_fh,
        chip_interior_margin_px=int(chip_interior_margin_px),
        skip_apriori_rms=_use_iter_clip,
    )
    if rms_result[0] is None:
        return _retry_sparse_fallback()
    rms_map, sorted_rms_map = rms_result

    def _apply_aperture_isolation_safe(cands: pd.DataFrame) -> pd.DataFrame:
        if cands.empty:
            return cands
        try:
            ms_arr_x2 = ms_arr_x
            ms_arr_y2 = ms_arr_y
            ms_arr_mag2 = ms_arr_mag
        except Exception as exc:  # noqa: BLE001
            # EXC-0206: T3 -- BO CVn RMS-rejection funnel debug list not built (EXCEPT-BULK-2 2026-07-08)
            logging.error('[EXC-0205] Aperture isolation filter skipped when ms_arr arrays unavailable - crowded comps not re...: %s', exc)
            return cands
        rej: set[Any] = set()
        for idx2, crow2 in cands.iterrows():
            cx2 = float(crow2.get("x", float("nan")))
            cy2 = float(crow2.get("y", float("nan")))
            cm2 = float(crow2.get("_mag", float("nan"))) if "_mag" in cands.columns else float("nan")
            if not (math.isfinite(cx2) and math.isfinite(cy2) and math.isfinite(cm2)):
                continue
            d2 = np.sqrt((ms_arr_x2 - cx2) ** 2 + (ms_arr_y2 - cy2) ** 2)
            in_ap2 = (d2 < float(_r_ap_iso)) & (d2 > 1e-6)
            if not bool(np.any(in_ap2)):
                continue
            nm2 = ms_arr_mag2[in_ap2]
            sig2 = nm2[np.isfinite(nm2) & (np.abs(nm2 - cm2) < 3.0)]
            if int(sig2.size) > 0:
                rej.add(idx2)
        if not rej:
            return cands
        after = int(len(cands) - len(rej))
        if after >= int(n_comp_min):
            return cands[~cands.index.isin(rej)]
        return cands

    candidates = ms[_base_mask | det_mask].copy()
    if candidates.empty:
        logging.warning(f"[FÁZA 1] {target_cid}: žiadni kandidáti po hard filtroch")
        _warn_zero_compstars_edge(
            target_cid=target_cid,
            target=target,
            chip_fw=chip_fw,
            chip_fh=chip_fh,
            chip_interior_margin_px=int(chip_interior_margin_px),
        )
        return _retry_sparse_fallback()
    candidates = _apply_aperture_isolation_safe(candidates)

    clip_meta: dict[str, int] | None = None
    if _use_iter_clip:
        _clip_out = _iterative_ensemble_clip_cm_residual(
            flux_map,
            bjd_map,
            sorted_rms_map,
            clip_sigma=_clip_sigma,
            n_comp_min=n_comp_min,
            min_final=1 if sparse_fallback else None,
        )
        if _clip_out is None:
            return pd.DataFrame()
        active, clip_meta = _clip_out
    else:
        active = _ensemble_mad_filter_rms(
            rms_map,
            candidates,
            target_cid=target_cid,
            target=target,
            n_comp_min=n_comp_min,
            rms_outlier_sigma=rms_outlier_sigma,
            chip_fw=chip_fw,
            chip_fh=chip_fh,
            chip_interior_margin_px=int(chip_interior_margin_px),
        )
    if active is None:
        return _retry_sparse_fallback()

    _bo_funnel: dict[str, int] = {}
    _bo_rms_rejected: list[tuple[str, float]] = []
    if str(target_cid).strip() == "1498613634033133184":
        try:
            _bo_funnel["F_perf4b"] = int(len(candidates_pre))
            _bo_funnel["G_after_rms"] = int(len(active))
            for _cid_r, _rv in sorted(
                (sorted_rms_map or {}).items(), key=lambda kv: (float(kv[1]), str(kv[0]))
            ):
                if _cid_r not in active:
                    if math.isfinite(float(_rv)) and float(_rv) > float(max_comp_rms):
                        _bo_rms_rejected.append((str(_cid_r), float(_rv)))
        except Exception:  # noqa: BLE001
            # EXC-0207: T3 -- BO CVn comp funnel summary log not emitted (EXCEPT-BULK-2 2026-07-08)
            pass

    id_col_cand = (
        "name"
        if "name" in candidates.columns
        else ("catalog_id" if "catalog_id" in candidates.columns else "name")
    )
    score_map, tier_map = _score_comp_candidates_broeg(
        active,
        candidates,
        contamination_map,
        id_col_cand=id_col_cand,
        mag_t=mag_t,
        target_bprp_eff=target_bprp_eff,
        t_bp_tgt=t_bp_tgt,
        _individual_tier=_individual_tier,
        cfg=_cfg_p1,
    )

    tier_out = _assign_comp_tiers_to_pool(
        candidates,
        active,
        id_col_cand=id_col_cand,
        target=target,
        target_cid=target_cid,
        target_bprp_eff=target_bprp_eff,
        t_bp_tgt=t_bp_tgt,
        mag_t=mag_t,
        _individual_tier=_individual_tier,
        _target_name=_target_name,
        max_mag_diff_t1=max_mag_diff_t1,
        max_mag_diff=max_mag_diff,
        gaia_db_path=gaia_db_path,
        vsx_local_db_path=vsx_local_db_path,
        gaia_prefetch=gaia_prefetch,
        n_comp_min=n_comp_min,
        n_comp_max=n_comp_max,
        chip_fw=chip_fw,
        chip_fh=chip_fh,
        chip_interior_margin_px=int(chip_interior_margin_px),
        cfg=_cfg_p1,
    )
    final_comps = tier_out["final_comps"]
    if final_comps is None or getattr(final_comps, "empty", True):
        return _retry_sparse_fallback()

    if str(target_cid).strip() == "1498613634033133184":
        try:
            from comp_selection_per_target import (  # noqa: PLC0415
                _log_bo_cvn_comp_funnel,
                bo_cvn_funnel_snapshot,
            )

            _bo_funnel.update(bo_cvn_funnel_snapshot())
            _bo_funnel["H_after_n_comp_max"] = int(len(final_comps))
            _bo_funnel["final_selected"] = int(len(final_comps))
            _log_bo_cvn_comp_funnel(
                step_counts=_bo_funnel,
                max_comp_rms=float(max_comp_rms),
                n_comp_max=int(n_comp_max),
                rms_rejected=_bo_rms_rejected,
            )
        except Exception:  # noqa: BLE001
            pass

    try:
        final_lookup = final_comps.copy()
        final_lookup[id_col_cand] = final_lookup[id_col_cand].astype(str).str.strip()
        final_lookup = final_lookup.set_index(id_col_cand, drop=False)
    except Exception:  # noqa: BLE001
        final_lookup = None

    result = _assemble_comp_selection_result_rows(
        tier_out["selected_ids"],
        final_comps,
        id_col_cand=id_col_cand,
        active=active,
        score_map=score_map,
        contamination_map=contamination_map,
        flux_map=flux_map,
        target_cid=target_cid,
        target=target,
        target_bprp_eff=target_bprp_eff,
        t_bp_tgt=t_bp_tgt,
        sel_note=str(tier_out["sel_note"]),
        used_mag_tol=float(used_mag_tol),
        best_tier=str(tier_out["best_tier"]),
        tier4_warning=bool(tier_out["tier4_warning"]),
        n_t1=int(tier_out["n_t1"]),
        n_t2=int(tier_out["n_t2"]),
        n_t3=int(tier_out["n_t3"]),
        n_t4=int(tier_out["n_t4"]),
        comp_bprp_map=tier_out["comp_bprp_map"],
        comp_tier_final_map=tier_out["comp_tier_final_map"],
        comp_delta_bprp_map=tier_out["comp_delta_bprp_map"],
        comp_color_tier_src_map=tier_out["comp_color_tier_src_map"],
        _b_rejected=_b_rejected,
        final_lookup=final_lookup,
        dilution_map=_dilution_map,
        comp_gs11_notes=_comp_gs11_notes,
        cfg=_cfg_p1,
        clip_meta=clip_meta,
        comp_path="sparse_fallback" if sparse_fallback else "default",
        per_target_rms_map=rms_map,
    )

    if _mode == "auto":
        # Route on the count of comps passing the per-target comp_rms gate, not raw
        # len(result): zero gate-passers -> sparse_fallback (known-issue (b) fix).
        _n_good = _count_gate_passing_comps(result, rms_map, max_comp_rms, id_col_cand)
        if _n_good >= 1:
            return result
        if resolve_comp_sparse_fallback_enabled(_cfg_p1):
            fb = select_comparison_stars_per_target(
                target,
                masterstars_df,
                per_frame_csv_paths,
                csv_cache=csv_cache,
                global_comp_pool_df=global_comp_pool_df,
                fwhm_px=fwhm_px,
                max_dist_deg=max_dist_deg,
                max_mag_diff=max_mag_diff,
                max_mag_diff_t1=max_mag_diff_t1,
                max_mag_diff_t2=max_mag_diff_t2,
                max_mag_diff_t3=max_mag_diff_t3,
                max_mag_diff_t4=max_mag_diff_t4,
                n_comp_min=n_comp_min,
                n_comp_max=n_comp_max,
                max_comp_rms=max_comp_rms,
                min_dist_arcsec=min_dist_arcsec,
                min_frames_frac=min_frames_frac,
                rms_outlier_sigma=rms_outlier_sigma,
                exclude_gaia_nss=exclude_gaia_nss,
                exclude_gaia_extobj=exclude_gaia_extobj,
                mag_bright_threshold=mag_bright_threshold,
                max_mag_diff_bright_floor=max_mag_diff_bright_floor,
                max_psf_chi2=max_psf_chi2,
                max_fwhm_factor=max_fwhm_factor,
                isolation_radius_px=isolation_radius_px,
                flux_col=flux_col,
                chip_fw=chip_fw,
                chip_fh=chip_fh,
                chip_interior_margin_px=chip_interior_margin_px,
                edge_bad_frame_frac_max=edge_bad_frame_frac_max,
                max_delta_bprp=max_delta_bprp,
                vsx_local_db_path=vsx_local_db_path,
                gaia_db_path=gaia_db_path,
                gaia_prefetch=gaia_prefetch,
                variable_target_catalog_ids=variable_target_catalog_ids,
                cfg=cfg,
                plate_scale_arcsec=plate_scale_arcsec,
                use_pixel_dist=use_pixel_dist,
                gs11_comp_rejects_acc=gs11_comp_rejects_acc,
                _selection_mode="sparse_fallback",
            )
            _n_fb = int(len(fb)) if fb is not None and not getattr(fb, "empty", True) else 0
            if _n_fb >= 1:
                return fb
        return pd.DataFrame()

    return result


def run_phase0_and_phase1(
    variable_targets_csv: Path,
    masterstars_csv: Path,
    per_frame_csv_dir: Path,
    output_dir: Path,
    *,
    fwhm_px: float = 3.7,
    frame_w_px: int = 2082,
    frame_h_px: int = 1397,
    chip_interior_margin_px: int = 100,
    match_radius_arcsec: float = 15.0,
    plate_scale_arcsec_px: float | None = None,
    max_dist_deg: float = 1.0,
    max_mag_diff: float = 0.25,
    max_mag_diff_t1: float = 0.50,
    max_mag_diff_t2: float = 1.00,
    max_mag_diff_t3: float = 1.50,
    max_mag_diff_t4: float = 2.00,
    n_comp_min: int = 3,
    n_comp_max: int = 7,
    max_comp_rms: float = 0.05,
    min_dist_arcsec: float = 60.0,
    min_frames_frac: float = 0.3,
    rms_outlier_sigma: float = 3.0,
    exclude_gaia_nss: bool = True,
    exclude_gaia_extobj: bool = True,
    mag_bright_threshold: float = 12.0,
    max_mag_diff_bright_floor: float = 0.0,
    max_psf_chi2: float = 3.0,
    max_fwhm_factor: float = 1.5,
    isolation_radius_px: float = 25.0,
    flux_col: str = "dao_flux",
    comp_max_delta_bprp: float = 0.5,
    cfg: AppConfig | None = None,
    progress_cb: Any = None,
    draft_id: int | None = None,
    db: Any = None,
) -> dict[str, Any]:
    """Spusti Fázu 0 + Fázu 1 a uloží výstupy.

    Výstupy (uložené do output_dir):
      active_targets.csv              — VSX ciele + ``zone_flag`` / ``skip_photometry`` (saturované)
      comparison_stars_per_target.csv — porovnávacie hviezdy pre každý cieľ
      suspected_variables.csv         — kandidáti na nové premenné (vysoký RMS, nie VSX)

    Returns:
        dict s kľúčmi:
          n_active_targets, n_comparison_pairs,
          active_targets_csv, comparison_stars_csv, suspected_variables_csv,
          targets_without_comps (list catalog_id)

    Args:
        chip_interior_margin_px: Min. počet pixelov od okraja čipu pre **všetky** kroky Fázy 0+1
            (aktívne ciele, porovnávačky, suspected). ``0`` = bez priestorového orezania.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _p(msg: str) -> None:
        if progress_cb is None:
            return
        try:
            progress_cb(str(msg))
        except UnicodeEncodeError:
            # Some Windows consoles use legacy encodings (cp1252) and crash on diacritics.
            # Use ASCII escapes so printing never raises again.
            progress_cb(str(msg).encode("ascii", "backslashreplace").decode("ascii"))

    _cfg_base = cfg if cfg is not None else AppConfig()
    _ms_density = Path(variable_targets_csv).resolve().parent / "MASTERSTAR.fits"
    _fw_in, _fh_in = int(frame_w_px), int(frame_h_px)
    frame_w_px, frame_h_px, _frame_hw_src = _resolve_frame_hw_px_from_masterstar(
        _ms_density,
        frame_w_px=_fw_in,
        frame_h_px=_fh_in,
        db=db,
        draft_id=draft_id,
    )
    if _frame_hw_src != "caller_default":
        logging.info(
            "[PHASE 0+1] Frame dimensions %d×%d px from %s (caller default %d×%d)",
            int(frame_w_px),
            int(frame_h_px),
            _frame_hw_src,
            _fw_in,
            _fh_in,
        )
    _n_field, _cw_fd, _ch_fd, _nsrc_fd, _vy_ndao_raw = _read_field_density_inputs(
        _ms_density,
        Path(masterstars_csv),
        int(frame_w_px),
        int(frame_h_px),
    )
    _density = compute_field_density(_n_field, _cw_fd, _ch_fd)
    _d_class = classify_field_density(
        float(_density),
        float(_cfg_base.field_density_sparse_threshold),
        float(_cfg_base.field_density_dense_threshold),
    )
    logging.info(
        "[FIELD DENSITY] %.0f hviezd/Mpx → trieda: %s (n_stars=%d, chip=%d×%dpx, n_src=%s)",
        float(_density),
        _d_class,
        int(_n_field),
        int(_cw_fd),
        int(_ch_fd),
        _nsrc_fd,
    )
    _adaptive_on = bool(_cfg_base.field_density_adaptive_enabled)
    _cfg_for_2a = copy.copy(_cfg_base)
    # ── [CROWDING-CLASSIFIER] signal-based comp overrides (gated, default OFF) ──
    # Replaces the detection/scale-locked stars/Mpx class with detection-independent
    # crowding_index signals. Additive sidecar (crowding_index.json); never overwrites
    # field_density.json. Falls back to the stars/Mpx path on any failure.
    _crowding_applied = False
    if bool(getattr(_cfg_base, "crowding_classifier_enabled", False)) and db is not None and draft_id is not None:
        try:
            from crowding_index import compute_crowding_index as _compute_ci  # noqa: PLC0415
            from database import get_gaia_db_max_g_mag as _get_gmax  # noqa: PLC0415

            _ps_dir = _ms_density.parent
            _gmax = float(_get_gmax(_cfg_base.gaia_db_path))
            _ci_res, _ = _compute_ci(
                _ps_dir.parent.parent, _ps_dir.name, db, int(draft_id), gaia_db_max_g=_gmax
            )
            _blend = _ci_res.get("blend_frac_1fwhm")
            _avail = _ci_res.get("n_gaia_below_eff_limit")
            _bottleneck = bool(_ci_res.get("catalog_is_bottleneck"))
            # SAMPLING GATE: a high comp-RMS only signals real contamination when the PSF
            # is resolved. On under-sampled fields the comp-RMS tail is the field floor,
            # so tightening max_comp_rms there thins the ensemble and worsens the LC.
            _min_fwhm = float(getattr(_cfg_base, "crowding_tighten_min_fwhm_px", 3.0))
            _well_sampled = float(fwhm_px) >= _min_fwhm
            _blend_high = _blend is not None and float(_blend) >= float(
                _cfg_base.crowding_blend_tighten_threshold
            )
            _tighten = bool(_blend_high and _well_sampled)
            _loosen = _avail is not None and float(_avail) < float(
                _cfg_base.crowding_comp_availability_loosen_count
            )
            _cfg_for_2a, _md_delta = apply_crowding_overrides(
                copy.copy(_cfg_base),
                loosen=bool(_loosen),
                tighten=bool(_tighten),
                suppress_mag_loosen=_bottleneck,
            )
            max_mag_diff = float(_cfg_for_2a.phase01_comparison_max_mag_diff)
            n_comp_min = int(_cfg_for_2a.phase01_comparison_n_comp_min)
            comp_max_delta_bprp = float(_cfg_for_2a.comp_max_delta_bprp)
            max_comp_rms = float(_cfg_for_2a.phase01_comparison_max_comp_rms)
            min_dist_arcsec = float(_cfg_for_2a.phase01_comparison_min_dist_arcsec)
            if _loosen:
                max_dist_deg = float(max_dist_deg) + float(_md_delta)
            _crowding_applied = True
            logging.info(
                "[CROWDING CLASSIFIER] blend=%.4f (th=%.3f) fwhm=%.2fpx (gate>=%.1f→sampled=%s) "
                "avail=%s (th=%.0f) bottleneck=%s → loosen=%s tighten=%s | legacy stars/Mpx class=%s",
                float(_blend) if _blend is not None else float("nan"),
                float(_cfg_base.crowding_blend_tighten_threshold),
                float(fwhm_px),
                _min_fwhm,
                bool(_well_sampled),
                _avail,
                float(_cfg_base.crowding_comp_availability_loosen_count),
                _bottleneck,
                bool(_loosen),
                bool(_tighten),
                _d_class,
            )
            try:
                (output_dir / "crowding_index.json").write_text(
                    json.dumps(
                        {
                            **_ci_res,
                            "classifier": {
                                "enabled": True,
                                "loosen": bool(_loosen),
                                "tighten": bool(_tighten),
                                "blend_high": bool(_blend_high),
                                "well_sampled": bool(_well_sampled),
                                "fwhm_px": float(fwhm_px),
                                "tighten_min_fwhm_px": float(_min_fwhm),
                                "suppress_mag_loosen": bool(_bottleneck),
                                "blend_tighten_threshold": float(
                                    _cfg_base.crowding_blend_tighten_threshold
                                ),
                                "comp_availability_loosen_count": float(
                                    _cfg_base.crowding_comp_availability_loosen_count
                                ),
                                "legacy_stars_mpx_class": _d_class,
                                "eff_max_mag_diff": float(max_mag_diff),
                                "eff_n_comp_min": int(n_comp_min),
                                "eff_comp_max_delta_bprp": float(comp_max_delta_bprp),
                                "eff_max_comp_rms": float(max_comp_rms),
                                "eff_min_dist_arcsec": float(min_dist_arcsec),
                                "eff_max_dist_deg": float(max_dist_deg),
                            },
                        },
                        indent=2,
                        ensure_ascii=False,
                        default=str,
                    ),
                    encoding="utf-8",
                )
            except Exception as exc:  # noqa: BLE001
                logging.error('[EXC-0208] crowding_signal.json diagnostic write fails during field-density adaptive step: %s', exc)
                pass
        except Exception as _exc:  # noqa: BLE001
            logging.warning(
                "[CROWDING CLASSIFIER] signal computation failed (%s) — falling back to stars/Mpx",
                _exc,
            )
            _crowding_applied = False
    if not _crowding_applied and _adaptive_on:
        _cfg_for_2a = apply_density_overrides(copy.copy(_cfg_base), _d_class)
        max_mag_diff = float(_cfg_for_2a.phase01_comparison_max_mag_diff)
        n_comp_min = int(_cfg_for_2a.phase01_comparison_n_comp_min)
        comp_max_delta_bprp = float(_cfg_for_2a.comp_max_delta_bprp)
        max_comp_rms = float(_cfg_for_2a.phase01_comparison_max_comp_rms)
        min_dist_arcsec = float(_cfg_for_2a.phase01_comparison_min_dist_arcsec)
        _md_extra = DENSITY_OVERRIDES.get(_d_class, {}).get("phase01_comparison_max_dist_deg")
        if _md_extra is not None:
            max_dist_deg = float(max_dist_deg) + float(_md_extra)
    try:
        _fd_adaptive_applied = bool(
            not _crowding_applied and _adaptive_on and _d_class in ("sparse", "dense")
        )
        (output_dir / "field_density.json").write_text(
            json.dumps(
                {
                    "density_h_star_per_mpx": round(float(_density), 4),
                    "density_class": _d_class,
                    "n_stars": int(_n_field),
                    "n_stars_dao_raw": int(_vy_ndao_raw)
                    if _vy_ndao_raw is not None and int(_vy_ndao_raw) > 0
                    else None,
                    "n_stars_source": _nsrc_fd,
                    "chip_w_px": int(_cw_fd),
                    "chip_h_px": int(_ch_fd),
                    "field_density_adaptive_applied": _fd_adaptive_applied,
                    "crowding_classifier_applied": bool(_crowding_applied),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001
        logging.error('[EXC-0209] field_density.json write fails - downstream Phase 2A lacks stored density metadata: %s', exc)
        pass

    _wcs_scale_ok = True
    _expected_scale = (
        float(plate_scale_arcsec_px)
        if plate_scale_arcsec_px is not None
        and math.isfinite(float(plate_scale_arcsec_px))
        and float(plate_scale_arcsec_px) > 0
        else float(_cfg_base.phase01_plate_scale_arcsec_per_px or 1.3)
    )
    if _ms_density.is_file():
        try:
            from astropy.wcs import WCS as _WCS_check  # noqa: PLC0415

            with astrofits.open(_ms_density, memmap=False) as _hdul_wcs:
                _wcs_check = _WCS_check(_hdul_wcs[0].header)
            _psm_chk = np.asarray(_wcs_check.pixel_scale_matrix, dtype=np.float64)
            _actual_scale = float(np.sqrt(np.abs(np.linalg.det(_psm_chk))) * 3600.0)
            _scale_ratio = abs(_actual_scale - _expected_scale) / max(_expected_scale, 1e-9)
            if _scale_ratio > 0.20:
                log_event(
                    f"[WCS SANITY] Scale {_actual_scale:.3f}\"/px deviates "
                    f"{_scale_ratio * 100.0:.1f}% from expected "
                    f"{_expected_scale:.3f}\"/px — using pixel-distance fallback"
                )
                _wcs_scale_ok = False
        except Exception as _wcs_exc:  # noqa: BLE001
            logging.error('[EXC-0210] WCS scale sanity exception assumes scale OK - comp matching uses ra/dec haversine when ...: %s', exc)
            logging.warning(
                "[WCS SANITY] check failed (non-fatal): %s — skipping check, assuming WCS scale OK "
                "(radec-haversine distance mode).",
                _wcs_exc,
            )
    log_event(
        f"[COMP SELECT] Distance mode: "
        f"{'pixel-fallback' if not _wcs_scale_ok else 'radec-haversine'}"
    )

    # ── FÁZA 0 ──
    _p("Fáza 0: výber aktívnych cieľov z VSX…")
    logging.info("[FÁZA 0] Výber aktívnych cieľov...")
    _cfg_p01 = _cfg_base
    # Load annulus-aware safe bbox from photometry_plan.json (if available).
    _safe_bbox: tuple[float, float, float, float] | None = None
    try:
        plan_path = Path(variable_targets_csv).parent / "photometry_plan.json"
        if plan_path.is_file():
            import json as _json  # noqa: PLC0415

            _plan = _json.loads(plan_path.read_text(encoding="utf-8"))
            sb = _plan.get("safe_bbox_px")
            if isinstance(sb, (list, tuple)) and len(sb) == 4:
                x0b, y0b, x1b, y1b = sb
                _safe_bbox = (float(x0b), float(y0b), float(x1b), float(y1b))
    except Exception:  # noqa: BLE001
        _safe_bbox = None
    _ms_for_catalog_only = Path(variable_targets_csv).resolve().parent / "MASTERSTAR.fits"
    _masterstar_wcs: Any = None
    if _ms_for_catalog_only.is_file():
        try:
            import warnings

            from astropy.wcs import FITSFixedWarning, WCS

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FITSFixedWarning)
                with astrofits.open(_ms_for_catalog_only, memmap=False) as hdul:
                    _masterstar_wcs = WCS(hdul[0].header)
        except Exception as exc:  # noqa: BLE001
            logging.warning("[VT REFRESH] MASTERSTAR WCS sa nepodarilo načítať: %s — x/y v variable_targets.csv bez zmeny", exc)
            _masterstar_wcs = None

    _vt_p01 = Path(variable_targets_csv)
    if _masterstar_wcs is not None and _vt_p01.is_file():
        _refresh_variable_targets_xy(
            variable_targets_csv=_vt_p01,
            wcs=_masterstar_wcs,
            chip_w=int(frame_w_px),
            chip_h=int(frame_h_px),
        )

    if (
        plate_scale_arcsec_px is not None
        and math.isfinite(float(plate_scale_arcsec_px))
        and float(plate_scale_arcsec_px) > 0
    ):
        _plate_scale_p01 = float(plate_scale_arcsec_px)
    elif _ms_for_catalog_only.is_file():
        _plate_scale_p01 = _resolve_plate_scale_arcsec_per_px(_cfg_p01, _ms_for_catalog_only)
    else:
        _plate_scale_p01 = _resolve_plate_scale_arcsec_per_px(_cfg_p01)

    active = select_active_targets(
        variable_targets_csv,
        masterstars_csv,
        frame_w_px=frame_w_px,
        frame_h_px=frame_h_px,
        edge_margin_px=int(chip_interior_margin_px),
        safe_bbox=_safe_bbox,
        match_radius_arcsec=match_radius_arcsec,
        gaia_db_path=str(_cfg_p01.gaia_db_path or ""),
        vsx_local_db_path=str(_cfg_p01.vsx_local_db_path or "").strip() or None,
        masterstar_fits_path=_ms_for_catalog_only if _ms_for_catalog_only.is_file() else None,
        plate_scale_arcsec_px=_plate_scale_p01,  # TODO-23
        cfg=_cfg_p01,
    )
    active_csv = output_dir / "active_targets.csv"
    try:
        from gaia_catalog_id import normalize_gaia_source_id_series  # noqa: PLC0415

        if "catalog_id" in active.columns:
            active = active.copy()
            active["catalog_id"] = normalize_gaia_source_id_series(active["catalog_id"])
    except Exception as exc:  # noqa: BLE001
        # EXC-0212: T3 -- ProcFrameStore not stored in Streamlit session_state - UI perf cache miss only (EXCEPT-BULK-2 2026-07-08)
        logging.error('[EXC-0211] active_targets.csv catalog_id normalization fails - float-truncated IDs written to disk: %s', exc)
        pass
    active.to_csv(active_csv, index=False)
    logging.info(f"[FÁZA 0] Uložené: {active_csv} ({len(active)} cieľov)")
    _excluded = LAST_EXCLUDED_TARGETS
    if _excluded is not None and not _excluded.empty:
        excluded_csv = output_dir / "excluded_targets.csv"
        _excluded.to_csv(excluded_csv, index=False)
        logging.info(f"[FÁZA 0] Uložené: {excluded_csv} ({len(_excluded)} excluded)")
    _p(f"Fáza 0 hotová: {len(active)} aktívnych cieľov")

    if active.empty:
        return {
            "n_active_targets": 0,
            "n_comparison_pairs": 0,
            "active_targets_csv": str(active_csv),
            "comparison_stars_csv": None,
            "suspected_variables_csv": None,
            "targets_without_comps": [],
            "field_density_h_star_per_mpx": float(_density),
            "field_density_class": str(_d_class),
            "field_density_adaptive_applied": bool(_adaptive_on and _d_class in ("sparse", "dense")),
            "field_density_n_stars": int(_n_field),
            "cfg_effective_for_photometry": _cfg_for_2a if _adaptive_on else None,
        }

    # Read as strings to prevent Gaia ID precision loss (float64/scientific notation).
    ms_df = pd.read_csv(masterstars_csv, low_memory=False, dtype=_GAIA_ID_DTYPE)
    # Normalizuj Gaia ID na string
    for _id_col in ("catalog_id", "name"):
        if _id_col in ms_df.columns:
            ms_df[_id_col] = _normalize_id_series(ms_df[_id_col])

    # ── FÁZA 1 — per target ──
    all_comp_rows: list[pd.DataFrame] = []
    targets_without_comps: list[str] = []

    # PERF-5: unified ProcFrameStore — one disk read per proc_*.csv frame
    _pfc_dir = Path(per_frame_csv_dir)
    _proc_glob = PROC_CSV_GLOB
    if not any(_pfc_dir.glob(_proc_glob)):
        if any(_pfc_dir.glob("*_cal.csv")):
            _proc_glob = "*_cal.csv"
    _proc_store = ProcFrameStore.build(
        _pfc_dir,
        glob_pattern=_proc_glob,
        extra_cols=[flux_col] if flux_col not in PROC_STORE_COLS else None,
    )
    shared_csv_cache = _proc_store
    csv_paths = [Path(k) for k in _proc_store.keys()]
    _p(f"Fáza 1: ProcFrameStore {len(_proc_store)} per-frame CSV — výber porovnávačiek ({len(active)} cieľov)…")

    try:
        import streamlit as st  # noqa: PLC0415

        if hasattr(st, "session_state"):
            st.session_state["proc_frame_store"] = _proc_store
            logging.debug("[PERF-6] ProcFrameStore stored in st.session_state")
    except Exception:  # noqa: BLE001
        pass

    _cfg_gaia_targets = _cfg_base
    _gaia_db_targets = str(_cfg_gaia_targets.gaia_db_path or "").strip()
    _vsx_db_targets = str(_cfg_gaia_targets.vsx_local_db_path or "").strip() or None

    _vt_chip = pd.read_csv(variable_targets_csv, low_memory=False, dtype=_GAIA_ID_DTYPE)
    _vt_cid_exclude: frozenset[str] | None = None
    try:
        if "catalog_id" in _vt_chip.columns:
            from gaia_catalog_id import normalize_gaia_id_set  # noqa: PLC0415

            _vx = normalize_gaia_id_set(
                _vt_chip["catalog_id"].tolist(),
                log_label="variable_targets.csv (phase1 exclude)",
            )
            _vt_cid_exclude = _vx or None
    except Exception:  # noqa: BLE001
        _vt_cid_exclude = None
    _fw_chip, _fh_chip = _phase0_effective_frame_hw_px(
        _vt_chip,
        ms_df,
        frame_w_px=int(frame_w_px),
        frame_h_px=int(frame_h_px),
        edge_margin_px=int(chip_interior_margin_px),
    )

    _cfg_gp = _cfg_for_2a if _adaptive_on else _cfg_base
    _global_pool_df: pd.DataFrame | None = None
    if bool(_cfg_gp.global_comp_pool_enabled):
        try:
            _global_pool_df = build_global_comp_pool(
                masterstars_df=ms_df,
                per_frame_csv_paths=csv_paths,
                csv_cache=shared_csv_cache,
                variable_target_catalog_ids=_vt_cid_exclude or frozenset(),
                safe_bbox=_safe_bbox,
                chip_fw=int(_fw_chip),
                chip_fh=int(_fh_chip),
                chip_interior_margin_px=int(chip_interior_margin_px),
                max_comp_rms=float(max_comp_rms),
                cfg=_cfg_gp,
                flux_col=flux_col,
                min_frames_frac=float(min_frames_frac),
                fwhm_px=float(fwhm_px),
                max_psf_chi2=float("inf"),  # global pool: skip PSF chi² (per-target filter unchanged)
                max_fwhm_factor=float(max_fwhm_factor),
            )
            if _global_pool_df is None or getattr(_global_pool_df, "empty", True):
                _global_pool_df = None
        except Exception as _gcp_exc:  # noqa: BLE001
            logging.warning(
                "[GLOBAL COMP POOL] zostavenie zlyhalo: %s — fallback na per-target masterstars",
                _gcp_exc,
            )
            _global_pool_df = None

    if _global_pool_df is not None and "catalog_id" in _global_pool_df.columns:
        _global_pool_df = _global_pool_df.sort_values("catalog_id", kind="mergesort").reset_index(
            drop=True
        )

    _gaia_batch: dict[str, dict[str, Any]] = {}
    if _gaia_db_targets:
        _cids_batch = [
            str(normalize_gaia_source_id(r.get("catalog_id") or ""))
            for _, r in active.iterrows()
        ]
        _gaia_batch = _batch_enrich_targets_bp_rp_from_gaia_db(_cids_batch, _gaia_db_targets)
        logging.info(
            "[PHASE 1] Gaia batch lookup: %d/%d targets enriched",
            len(_gaia_batch),
            int(len(active)),
        )

    # PERF-3: prefetch Gaia bp_rp + teff for masterstars comp pool (before per-target loop).
    _comp_gaia_prefetch: dict[str, dict[str, Any]] = {}
    _comp_source_ids_n = 0
    try:
        _comp_id_seen: set[str] = set()
        _comp_source_ids: list[str] = []
        for _pool_df in (ms_df, _global_pool_df):
            if _pool_df is None or getattr(_pool_df, "empty", True):
                continue
            for _id_col in ("catalog_id", "name"):
                if _id_col not in _pool_df.columns:
                    continue
                for raw in _pool_df[_id_col].dropna().unique():
                    g = normalize_gaia_source_id(raw)
                    if not g or not g.isdigit() or g in _comp_id_seen:
                        continue
                    _comp_id_seen.add(g)
                    _comp_source_ids.append(g)
        _comp_source_ids_n = len(_comp_source_ids)
        if _comp_source_ids and _gaia_db_targets:
            _comp_gaia_prefetch = _batch_enrich_targets_bp_rp_from_gaia_db(
                _comp_source_ids,
                _gaia_db_targets,
            )
            logging.info(
                "[PERF-3] Comp Gaia prefetch: %d source_ids → %d hits",
                _comp_source_ids_n,
                len(_comp_gaia_prefetch),
            )
    except Exception as exc:  # noqa: BLE001
        logging.error('[EXC-0213] Comp Gaia bp_rp prefetch batch fails - phase1 comp selection hits DB per-star instead o...: %s', exc)
        logging.warning("[PERF-3] Comp Gaia prefetch failed (non-fatal): %s", exc)

    _t_phase1 = time.time()
    _gs11_comp_rejects_acc: list[int] = [0]
    _n_active = int(len(active))
    for _i_active, (active_idx, target_row) in enumerate(active.iterrows(), start=1):
        try:
            if progress_cb is not None and (
                _i_active == 1 or _i_active == _n_active or (_n_active > 1 and _i_active % max(1, _n_active // 12) == 0)
            ):
                _tid = str(target_row.get("vsx_name") or target_row.get("catalog_id", ""))[:48]
                _p(f"Phase 1: target {_i_active}/{_n_active}: {_tid}")
            tr_enriched = _enrich_target_bp_rp_from_gaia_db(
                target_row,
                gaia_db_path=_gaia_db_targets,
                vsx_local_db_path=_vsx_db_targets,
                gaia_prefetch=_gaia_batch,
            )
            if "bp_rp" in active.columns:
                active.loc[active_idx, "bp_rp"] = tr_enriched.get("bp_rp", active.loc[active_idx, "bp_rp"])
            comps = select_comparison_stars_per_target(
                tr_enriched,
                ms_df,
                csv_paths,
                csv_cache=shared_csv_cache,
                global_comp_pool_df=_global_pool_df,
                fwhm_px=fwhm_px,
                max_dist_deg=max_dist_deg,
                max_mag_diff=max_mag_diff,
                max_mag_diff_t1=max_mag_diff_t1,
                max_mag_diff_t2=max_mag_diff_t2,
                max_mag_diff_t3=max_mag_diff_t3,
                max_mag_diff_t4=max_mag_diff_t4,
                n_comp_min=n_comp_min,
                n_comp_max=n_comp_max,
                max_comp_rms=max_comp_rms,
                min_dist_arcsec=min_dist_arcsec,
                min_frames_frac=min_frames_frac,
                rms_outlier_sigma=rms_outlier_sigma,
                exclude_gaia_nss=exclude_gaia_nss,
                exclude_gaia_extobj=exclude_gaia_extobj,
                mag_bright_threshold=mag_bright_threshold,
                max_mag_diff_bright_floor=max_mag_diff_bright_floor,
                max_psf_chi2=float("inf"),  # DAO-era proc CSV: chi² not ePSF yet
                max_fwhm_factor=max_fwhm_factor,
                isolation_radius_px=isolation_radius_px,
                flux_col=flux_col,
                chip_fw=_fw_chip,
                chip_fh=_fh_chip,
                chip_interior_margin_px=int(chip_interior_margin_px),
                max_delta_bprp=float(comp_max_delta_bprp),
                vsx_local_db_path=str(_cfg_gaia_targets.vsx_local_db_path or "").strip() or None,
                gaia_db_path=str(_cfg_gaia_targets.gaia_db_path or "").strip() or None,
                gaia_prefetch=_comp_gaia_prefetch,
                variable_target_catalog_ids=_vt_cid_exclude,
                cfg=_cfg_gp,
                plate_scale_arcsec=float(
                    plate_scale_arcsec_px
                    if plate_scale_arcsec_px is not None
                    and math.isfinite(float(plate_scale_arcsec_px))
                    and float(plate_scale_arcsec_px) > 0
                    else (
                        float(_cfg_gaia_targets.phase01_plate_scale_arcsec_per_px)
                        or 1.3
                    )
                ),
                use_pixel_dist=not _wcs_scale_ok,
                gs11_comp_rejects_acc=_gs11_comp_rejects_acc,
            )
            if comps is None or comps.empty:
                targets_without_comps.append(str(tr_enriched.get("catalog_id", "")))
            else:
                all_comp_rows.append(comps)
        except Exception as exc:  # noqa: BLE001
            # EXC-0215: T3 -- Prefetch coverage stats log after comp selection suppressed (EXCEPT-BULK-2 2026-07-08)
            logging.warning(
                "[PHASE1] %s: neočakávaná chyba, preskakujem: %s",
                str(target_row.get("catalog_id", "?")),
                exc,
            )
            targets_without_comps.append(str(target_row.get("catalog_id", "") or ""))
            continue

    try:
        active.to_csv(active_csv, index=False)
        logging.info("[FÁZA 0–1] active_targets.csv prepísané po doplnení bp_rp targetov (Gaia DB).")
    except Exception as exc:  # noqa: BLE001
        logging.error('[EXC-0214] active_targets.csv rewrite after phase1 bp_rp enrichment fails - disk copy lacks update...: %s', exc)
        log_event(f"active_targets.csv zápis po Fáze 1 zlyhal: {exc!s}")

    comp_df = pd.concat(all_comp_rows, ignore_index=True) if all_comp_rows else pd.DataFrame()
    if "target_catalog_id" in comp_df.columns and "catalog_id" in comp_df.columns:
        _before = len(comp_df)
        comp_df = comp_df.drop_duplicates(
            subset=["target_catalog_id", "catalog_id"], keep="first"
        )
        _after = len(comp_df)
        if _before != _after:
            log_event(
                f"comparison_stars_per_target: removed {_before - _after} "
                f"duplicate (target_catalog_id, catalog_id) rows"
            )
    if _comp_gaia_prefetch and not comp_df.empty and "catalog_id" in comp_df.columns:
        try:
            _sel_cids = {
                normalize_gaia_source_id(x)
                for x in comp_df["catalog_id"].tolist()
                if normalize_gaia_source_id(x)
            }
            _n_pref_hit = sum(1 for c in _sel_cids if c in _comp_gaia_prefetch)
            logging.info(
                "[PERF-3] Selected comp stars covered by prefetch: %d/%d "
                "(pool prefetch %d ids, %d DB hits)",
                _n_pref_hit,
                len(_sel_cids),
                _comp_source_ids_n,
                len(_comp_gaia_prefetch),
            )
        except Exception:  # noqa: BLE001
            pass
    # Safety: even when no comps found (or all targets failed), keep a stable schema so CSV isn't empty.
    if comp_df is None or len(list(comp_df.columns)) == 0:
        comp_df = pd.DataFrame(
            columns=[
                "catalog_id",
                "name",
                "ra_deg",
                "dec_deg",
                "x",
                "y",
                "mag",
                "bp_rp",
                "comp_rms",
                "comp_score",
                "contamination_idx",
                "comp_n_frames",
                "target_catalog_id",
                "target_vsx_name",
                "target_bp_rp",
                "delta_bprp_abs",
                "comp_tier",
                "color_tier_src",
                "comp_weight",
                "selection_note",
                "used_mag_tol",
                "selected_tier",
                "tier4_warning",
                "n_tier1",
                "n_tier2",
                "n_tier3",
                "n_tier4",
            ]
        )

    # Fallback: doplň bp_rp pre COMP hviezdy bez Gaia farby pomocou lokálnej Gaia DB (sky-box okolo RA/Dec).
    try:
        if (
            not comp_df.empty
            and "bp_rp" in comp_df.columns
            and "ra_deg" in comp_df.columns
            and "dec_deg" in comp_df.columns
        ):
            gaia_db_path = str(_cfg_base.gaia_db_path or "").strip() or None

            bp_nan = pd.to_numeric(comp_df["bp_rp"], errors="coerce").isna()
            ra_ok = pd.to_numeric(comp_df["ra_deg"], errors="coerce").apply(lambda v: math.isfinite(float(v)))
            dec_ok = pd.to_numeric(comp_df["dec_deg"], errors="coerce").apply(lambda v: math.isfinite(float(v)))
            needs = comp_df[bp_nan & ra_ok & dec_ok].copy()

            n_nan = int(len(needs))
            n_found = 0
            if n_nan > 0 and gaia_db_path:
                if "gaia_bp_rp_source" not in comp_df.columns:
                    comp_df["gaia_bp_rp_source"] = ""

                # Magnitude column for matching Gaia photometry (prefer "mag", fallback to "phot_g_mean_mag").
                mag_col = "mag" if "mag" in comp_df.columns else ("phot_g_mean_mag" if "phot_g_mean_mag" in comp_df.columns else None)

                radius_deg = 0.001  # ~3.6 arcsec
                for i, row in needs.iterrows():
                    ra0 = float(pd.to_numeric(row.get("ra_deg"), errors="coerce"))
                    dec0 = float(pd.to_numeric(row.get("dec_deg"), errors="coerce"))
                    if not (math.isfinite(ra0) and math.isfinite(dec0)):
                        continue

                    mag_comp = float("nan")
                    if mag_col is not None:
                        try:
                            mag_comp = float(pd.to_numeric(row.get(mag_col), errors="coerce"))
                        except Exception:  # noqa: BLE001
                            # EXC-0216: T4 -- Bad ra/dec on one Gaia fallback row skipped in comp bp_rp nearest-neighbor search (EXCEPT-BULK-2 2026-07-08)
                            mag_comp = float("nan")
                    if not math.isfinite(mag_comp):
                        continue

                    dec_min = max(-90.0, dec0 - radius_deg)
                    dec_max = min(90.0, dec0 + radius_deg)

                    # Handle RA wrap at 0/360 for tiny windows.
                    ra_min = ra0 - radius_deg
                    ra_max = ra0 + radius_deg
                    gaia_rows: list[dict[str, Any]] = []
                    if ra_min < 0.0:
                        gaia_rows.extend(
                            query_local_gaia(
                                ra_min=360.0 + ra_min,
                                ra_max=360.0,
                                dec_min=dec_min,
                                dec_max=dec_max,
                                db_path=gaia_db_path,
                                mag_limit=max(20.0, mag_comp + 2.0),
                                max_rows=200,
                            )
                        )
                        gaia_rows.extend(
                            query_local_gaia(
                                ra_min=0.0,
                                ra_max=ra_max,
                                dec_min=dec_min,
                                dec_max=dec_max,
                                db_path=gaia_db_path,
                                mag_limit=max(20.0, mag_comp + 2.0),
                                max_rows=200,
                            )
                        )
                    elif ra_max > 360.0:
                        gaia_rows.extend(
                            query_local_gaia(
                                ra_min=ra_min,
                                ra_max=360.0,
                                dec_min=dec_min,
                                dec_max=dec_max,
                                db_path=gaia_db_path,
                                mag_limit=max(20.0, mag_comp + 2.0),
                                max_rows=200,
                            )
                        )
                        gaia_rows.extend(
                            query_local_gaia(
                                ra_min=0.0,
                                ra_max=ra_max - 360.0,
                                dec_min=dec_min,
                                dec_max=dec_max,
                                db_path=gaia_db_path,
                                mag_limit=max(20.0, mag_comp + 2.0),
                                max_rows=200,
                            )
                        )
                    else:
                        gaia_rows = query_local_gaia(
                            ra_min=ra_min,
                            ra_max=ra_max,
                            dec_min=dec_min,
                            dec_max=dec_max,
                            db_path=gaia_db_path,
                            mag_limit=max(20.0, mag_comp + 2.0),
                            max_rows=200,
                        )

                    if not gaia_rows:
                        continue

                    best = None
                    best_d = float("inf")
                    for gr in gaia_rows:
                        try:
                            g_mag = float(gr.get("g_mag"))
                        except Exception:  # noqa: BLE001
                            g_mag = float("nan")
                        if not (math.isfinite(g_mag) and abs(g_mag - mag_comp) < 1.0):
                            continue
                        try:
                            ra_g = float(gr.get("ra"))
                            dec_g = float(gr.get("dec"))
                        except Exception:  # noqa: BLE001
                            continue
                        if not (math.isfinite(ra_g) and math.isfinite(dec_g)):
                            continue
                        d = _angular_distance_deg(ra0, dec0, ra_g, dec_g)
                        if math.isfinite(d) and d < best_d:
                            best_d = d
                            best = gr

                    if best is None:
                        continue
                    bprp = best.get("bp_rp")
                    try:
                        bprp_f = float(bprp)
                    except Exception:  # noqa: BLE001
                        bprp_f = float("nan")
                    if not math.isfinite(bprp_f):
                        continue

                    comp_df.loc[i, "bp_rp"] = bprp_f
                    comp_df.loc[i, "gaia_bp_rp_source"] = "gaia_db_fallback"
                    n_found += 1

            if n_nan > 0:
                log_event(f"COMP bp_rp fallback: {n_found}/{n_nan} hviezd doplnených z Gaia DB")
    except Exception as exc:  # noqa: BLE001
        logging.error('[EXC-0217] Whole comp bp_rp Gaia-DB fallback block fails - comps export with NaN bp_rp and wrong t...: %s', exc)
        pass

    comp_csv = output_dir / "comparison_stars_per_target.csv"
    try:
        from gaia_catalog_id import normalize_gaia_source_id_series  # noqa: PLC0415

        if "catalog_id" in comp_df.columns:
            comp_df = comp_df.copy()
            comp_df["catalog_id"] = normalize_gaia_source_id_series(comp_df["catalog_id"])
        if "target_catalog_id" in comp_df.columns:
            comp_df = comp_df.copy()
            comp_df["target_catalog_id"] = normalize_gaia_source_id_series(comp_df["target_catalog_id"])
    except Exception as exc:  # noqa: BLE001
        logging.error('[EXC-0218] comparison_stars_per_target.csv catalog_id normalization fails - corrupted IDs in comp ...: %s', exc)
        pass
    comp_df.to_csv(comp_csv, index=False)
    _sparse_target_n = 0
    if "comp_path" in comp_df.columns and "target_catalog_id" in comp_df.columns:
        try:
            _cp = (
                comp_df.groupby(comp_df["target_catalog_id"].astype(str).str.strip())["comp_path"]
                .first()
                .astype(str)
                .str.strip()
                .str.lower()
            )
            _sparse_target_n = int((_cp == "sparse_fallback").sum())
        except Exception:  # noqa: BLE001
            _sparse_target_n = 0
    merge_photometry_pipeline_meta(
        output_dir,
        {"comp_sparse_fallback_target_count": int(_sparse_target_n)},
    )
    logging.info(
        f"[FÁZA 1] Uložené: {comp_csv} "
        f"({len(comp_df)} riadkov, {len(all_comp_rows)} targetov s porovnávačkami)"
    )
    logging.info(f"[FÁZA 1] Čas (comp selection): {time.time() - _t_phase1:.1f}s")
    _ps_p1 = float(
        plate_scale_arcsec_px
        if plate_scale_arcsec_px is not None
        and math.isfinite(float(plate_scale_arcsec_px))
        and float(plate_scale_arcsec_px) > 0
        else (float(_cfg_base.phase01_plate_scale_arcsec_per_px) or 1.3)
    )
    _gs11_p1 = build_gs11_summary(
        [],
        _cfg_base,
        comps_gs11_rejected=int(_gs11_comp_rejects_acc[0]),
        plate_scale_arcsec=_ps_p1,
    )
    merge_photometry_pipeline_meta(output_dir, {"gs11_summary": _gs11_p1})

    # ── Suspected variables ──
    # Hviezdy s vysokým RMS (>3σ nad mediánom) ktoré nie sú VSX ani active targets
    _p("Fáza 1: suspected variables (nové kandidáty)…")
    suspected_csv = output_dir / "suspected_variables.csv"
    _active_ids: set[str] = set()
    for _ax in active["catalog_id"].tolist():
        _nx = _normalize_id_value(_ax)
        if _nx:
            _active_ids.add(_nx)

    _margin_sus: int | None = None if int(chip_interior_margin_px) <= 0 else int(chip_interior_margin_px)

    _write_suspected_variables(
        ms_df=ms_df,
        csv_paths=csv_paths,
        active_target_ids=_active_ids,
        output_path=suspected_csv,
        min_frames_frac=min_frames_frac,
        outlier_sigma=3.0,
        interior_fw=_fw_chip,
        interior_fh=_fh_chip,
        interior_margin_px=_margin_sus,
        csv_cache=shared_csv_cache,
    )
    # Best-effort: repair Gaia IDs in suspected_variables.csv via RA/DEC + local Gaia DB.
    try:
        from repair_catalog_ids import repair_csv_catalog_ids_from_gaia_db  # noqa: PLC0415

        _gdb = str(_cfg_base.gaia_db_path or "").strip()
        if _gdb:
            gdbp = Path(_gdb)
            if gdbp.is_file() and suspected_csv.is_file():
                _ = repair_csv_catalog_ids_from_gaia_db(
                    csv_path=suspected_csv,
                    gaia_db_path=gdbp,
                    id_col="catalog_id",
                    backup=False,
                    max_sep_arcsec=10.0,
                    log_fn=lambda _m: None,
                )
    except Exception as exc:  # noqa: BLE001
        logging.error('[EXC-0219] suspected_variables.csv catalog_id auto-repair from Gaia DB fails silently: %s', exc)
        pass

    _p(f"Fáza 0+1 hotovo: {int(len(active))} cieľov, {int(len(comp_df))} párov porovnávačiek")
    return {
        "n_active_targets": int(len(active)),
        "n_comparison_pairs": int(len(comp_df)),
        "active_targets_csv": str(active_csv),
        "comparison_stars_csv": str(comp_csv),
        "suspected_variables_csv": str(suspected_csv),
        "targets_without_comps": targets_without_comps,
        "field_density_h_star_per_mpx": float(_density),
        "field_density_class": str(_d_class),
        "field_density_adaptive_applied": bool(_adaptive_on and _d_class in ("sparse", "dense")),
        "field_density_n_stars": int(_n_field),
        "cfg_effective_for_photometry": _cfg_for_2a if _adaptive_on else None,
        "proc_store": _proc_store,
    }


def run_sysrem_field(
    lc_dir: Path,
    *,
    n_iter: int = 3,
    flag_col: str = "flag",
    delta_col: str = "delta_mag",
    err_col: str = "err",
    out_col: str = "delta_mag_sysrem",
) -> dict[str, Any]:
    """Run SysRem systematic noise removal on all exported light curves.

    Implements the iterative algorithm of Tamuz, Mazeh & Zucker (2005),
    MNRAS 356, 1466. Removes common systematic trends from a set of
    differential light curves using inverse-variance weighting.

    Algorithm (one iteration):
        r_ij  = residuals (star i, frame j), NaN where flag != 'normal'
        w_ij  = 1 / err_ij^2  (weight; 0 where NaN or err <= 0)
        c_j   = sum_i(a_i * r_ij * w_ij) / sum_i(a_i^2 * w_ij)
        a_i   = sum_j(c_j * r_ij * w_ij) / sum_j(c_j^2 * w_ij)
        r_ij -= a_i * c_j

    Writes column ``delta_mag_sysrem`` into each lightcurve_*.csv.
    Stars with fewer than 10 valid frames are excluded from the
    system-vector fit but still corrected using the derived c_j vector.

    Args:
        lc_dir: Directory containing lightcurve_*.csv files.
        n_iter: Number of SysRem iterations (default 3).
        flag_col: Column name for quality flag.
        delta_col: Input column (default 'delta_mag').
        err_col: Uncertainty column (default 'err').
        out_col: Output column written to CSV (default 'delta_mag_sysrem').

    Returns:
        dict with keys:
            'n_stars': int — number of LC files processed
            'n_frames': int — number of frames (columns in matrix)
            'n_iter': int — iterations applied
            'rms_before': float — median RMS across stars before SysRem
            'rms_after': float — median RMS across stars after SysRem
            'rms_improvement_pct': float — (rms_before - rms_after) / rms_before * 100
            'skipped': list[str] — catalog_ids skipped (missing columns etc.)
    """
    lc_dir = Path(lc_dir)
    lc_files = sorted(lc_dir.glob("lightcurve_*.csv"))
    _empty: dict[str, Any] = {
        "n_stars": 0,
        "n_frames": 0,
        "n_iter": 0,
        "rms_before": float("nan"),
        "rms_after": float("nan"),
        "rms_improvement_pct": float("nan"),
        "skipped": [],
    }
    if not lc_files:
        logging.warning("[SysRem] No lightcurve_*.csv found in %s", lc_dir)
        return _empty

    dfs: dict[str, pd.DataFrame] = {}
    skipped: list[str] = []

    for f in lc_files:
        try:
            df = pd.read_csv(f, low_memory=False)
            required = {delta_col, err_col, flag_col}
            if not required.issubset(df.columns):
                skipped.append(f.stem)
                logging.warning("[SysRem] Missing columns in %s, skipping", f.name)
                continue
            cid = f.stem.replace("lightcurve_", "")
            dfs[cid] = df
        except Exception as exc:  # noqa: BLE001
            skipped.append(f.stem)
            logging.warning("[SysRem] Cannot read %s: %s", f.name, exc)

    if not dfs:
        logging.warning("[SysRem] No valid LC files loaded")
        return {**_empty, "skipped": skipped}

    n_frames = max(len(df) for df in dfs.values())
    star_ids = list(dfs.keys())
    n_stars = len(star_ids)

    R = np.full((n_stars, n_frames), np.nan)
    W = np.zeros((n_stars, n_frames))

    for i, cid in enumerate(star_ids):
        df = dfs[cid]
        nf = len(df)
        delta = pd.to_numeric(df[delta_col], errors="coerce").to_numpy()
        err = pd.to_numeric(df[err_col], errors="coerce").to_numpy()
        flags = df[flag_col].astype(str).to_numpy()

        valid = (flags == "normal") & np.isfinite(delta) & np.isfinite(err) & (err > 0)
        if valid.any():
            R[i, :nf] = np.where(valid, delta - float(np.nanmedian(delta[valid])), np.nan)
        W[i, :nf] = np.where(valid, 1.0 / (err**2), 0.0)

    min_valid_frames = 10
    fit_mask = np.sum(np.isfinite(R), axis=1) >= min_valid_frames

    rms_before_arr = np.array(
        [
            np.nanstd(R[i][np.isfinite(R[i])]) if np.isfinite(R[i]).sum() > 2 else np.nan
            for i in range(n_stars)
        ]
    )
    rms_before = float(np.nanmedian(rms_before_arr))

    a = np.ones(n_stars)
    fit_mask_f = fit_mask.astype(np.float64)

    for iteration in range(n_iter):
        R_filled = np.where(np.isfinite(R), R, 0.0)
        numer_c = np.einsum("i,ij,ij->j", a * fit_mask_f, R_filled, W)
        denom_c = np.einsum("i,ij->j", (a**2) * fit_mask_f, W)
        c = np.where(denom_c > 0, numer_c / denom_c, 0.0)

        numer_a = np.einsum("j,ij,ij->i", c, R_filled, W)
        denom_a = np.einsum("j,ij->i", c**2, W)
        a = np.where(denom_a > 0, numer_a / denom_a, 0.0)

        correction = np.outer(a, c)
        R = R - np.where(np.isfinite(R), correction, 0.0)

        logging.info(
            "[SysRem] Iteration %d/%d — median |c_j|=%.5f, median |a_i|=%.4f",
            iteration + 1,
            n_iter,
            float(np.median(np.abs(c))),
            float(np.median(np.abs(a))),
        )

    rms_after_arr = np.array(
        [
            np.nanstd(R[i][np.isfinite(R[i])]) if np.isfinite(R[i]).sum() > 2 else np.nan
            for i in range(n_stars)
        ]
    )
    rms_after = float(np.nanmedian(rms_after_arr))
    rms_improvement = (
        (rms_before - rms_after) / rms_before * 100.0 if rms_before > 0 else float("nan")
    )

    logging.info(
        "[SysRem] Done: %d stars × %d frames × %d iter | "
        "RMS before=%.4f after=%.4f improvement=%.1f%%",
        n_stars,
        n_frames,
        n_iter,
        rms_before,
        rms_after,
        rms_improvement,
    )

    for i, cid in enumerate(star_ids):
        df = dfs[cid]
        nf = len(df)
        delta_orig = pd.to_numeric(df[delta_col], errors="coerce").to_numpy()
        flags = df[flag_col].astype(str).to_numpy()
        valid = (flags == "normal") & np.isfinite(delta_orig)
        median_orig = float(np.nanmedian(delta_orig[valid])) if valid.any() else 0.0
        sysrem_col = np.full(nf, np.nan)
        sysrem_col[:nf] = R[i, :nf] + median_orig
        non_normal = flags != "normal"
        sysrem_col[non_normal] = delta_orig[non_normal]
        df = df.copy()
        df[out_col] = np.round(sysrem_col, 6)
        lc_path = lc_dir / f"lightcurve_{cid}.csv"
        df.to_csv(lc_path, index=False)

    return {
        "n_stars": n_stars,
        "n_frames": n_frames,
        "n_iter": n_iter,
        "rms_before": rms_before,
        "rms_after": rms_after,
        "rms_improvement_pct": rms_improvement,
        "skipped": skipped,
    }


def run_full_photometry_pipeline(
    *,
    masterstar_fits_path: Path,
    variable_targets_csv: Path,
    masterstars_csv: Path,
    per_frame_csv_dir: Path,
    detrended_aligned_dir: Path,
    output_dir: Path,
    cfg: AppConfig | None = None,
    db: Any = None,
    draft_id: int | None = None,
    progress_cb: Any = None,
) -> dict[str, Any]:
    """Jedno-krokový wrapper: Fáza 0+1 + Fáza 2A ako jeden celok.

    UI to používa ako jednu akciu „RUN Aperture Photometry“ pre daný obs_group.
    """
    _cfg = cfg or AppConfig()

    ensure_full_variable_targets_if_presel_stub(
        variable_targets_csv=Path(variable_targets_csv),
        masterstars_csv=Path(masterstars_csv),
        masterstar_fits=Path(masterstar_fits_path),
        cfg=_cfg,
        draft_id=draft_id,
    )

    def _p(msg: str) -> None:
        if progress_cb is None:
            return
        try:
            progress_cb(str(msg))
        except UnicodeEncodeError:
            progress_cb(str(msg).encode("ascii", "backslashreplace").decode("ascii"))

    # FWHM: prefer header (VY_FWHM_GAUSS/VY_FWHM), inak default z configu.
    fwhm_px = float(_cfg.sips_dao_fwhm_px)
    _ms_header_shared: Any | None = None
    _ms_path_shared = Path(masterstar_fits_path)
    if _ms_path_shared.is_file():
        try:
            from astropy.io import fits as astrofits  # noqa: PLC0415

            with astrofits.open(_ms_path_shared, memmap=False) as _hdul:
                _ms_header_shared = _hdul[0].header.copy()
        except Exception as exc:  # noqa: BLE001
            logging.error('[EXC-0220] Shared MASTERSTAR header cache load fails - repeated FITS opens (perf), not science num...: %s', exc)
            logging.warning("[PERF-2] Cannot open MASTERSTAR.fits for header: %s", exc)
    if _ms_header_shared is not None:
        try:
            for key in ("VY_FWHM_GAUSS", "VY_FWHM_GAUSSIAN", "VY_FWHM"):
                v = _ms_header_shared.get(key)
                if v is None:
                    continue
                fv = float(v)
                if 0.5 < fv < 30.0:
                    fwhm_px = fv
                    break
        except Exception as exc:  # noqa: BLE001
            logging.error('[EXC-0221] VY_FWHM/VY_FWHM_GAUSS header parse fails - pipeline uses default/config FWHM for phase0+1: %s', exc)
            pass

    # ── FÁZA 0+1 ──
    _p("Fáza 0+1: select targets + comparison stars…")
    _plate_scale = _get_plate_scale_from_cfg(
        _cfg,
        db=db,
        draft_id=draft_id,
        fits_path=Path(masterstar_fits_path),
        ms_header=_ms_header_shared,
    )
    if _plate_scale is None:
        _plate_scale = _read_plate_scale_from_fits_path(
            Path(masterstar_fits_path),
            ms_header=_ms_header_shared,
        )
        if _plate_scale is not None and math.isfinite(float(_plate_scale)) and float(_plate_scale) > 0:
            logging.info(
                "[FOV] plate_scale from MASTERSTAR.fits header → %.4f arcsec/px",
                float(_plate_scale),
            )
    _fw_pipe, _fh_pipe, _frame_hw_src = _resolve_frame_hw_px_from_masterstar(
        Path(masterstar_fits_path),
        frame_w_px=int(_cfg.frame_width_px),
        frame_h_px=int(_cfg.frame_height_px),
        db=db,
        draft_id=draft_id,
    )
    if _frame_hw_src != "caller_default":
        logging.info(
            "[PHASE 0+1] Pipeline frame dimensions %d×%d px from %s",
            int(_fw_pipe),
            int(_fh_pipe),
            _frame_hw_src,
        )
    p01 = run_phase0_and_phase1(
        variable_targets_csv=Path(variable_targets_csv),
        masterstars_csv=Path(masterstars_csv),
        per_frame_csv_dir=Path(per_frame_csv_dir),
        output_dir=Path(output_dir),
        fwhm_px=float(fwhm_px),
        frame_w_px=int(_fw_pipe),
        frame_h_px=int(_fh_pipe),
        chip_interior_margin_px=int(_cfg.phase01_chip_interior_margin_px),
        match_radius_arcsec=float(_cfg.phase01_match_radius_arcsec),
        plate_scale_arcsec_px=_plate_scale,
        max_dist_deg=_compute_fov_max_dist(
            frame_w_px=int(_fw_pipe),
            frame_h_px=int(_fh_pipe),
            plate_scale=_plate_scale,
            fov_fraction=float(_cfg.phase01_comparison_fov_fraction),
            fallback_deg=float(_cfg.phase01_comparison_max_dist_deg),
        ),
        max_mag_diff=float(_cfg.phase01_comparison_max_mag_diff),
        comp_max_delta_bprp=float(_cfg.comp_max_delta_bprp),
        max_mag_diff_t1=float(_cfg.phase01_tier1_mag),
        max_mag_diff_t2=float(_cfg.phase01_tier2_mag),
        max_mag_diff_t3=float(_cfg.phase01_tier3_mag),
        max_mag_diff_t4=float(_cfg.phase01_tier4_mag),
        n_comp_min=int(_cfg.phase01_comparison_n_comp_min),
        n_comp_max=int(_cfg.phase01_comparison_n_comp_max),
        max_comp_rms=float(_cfg.phase01_comparison_max_comp_rms),
        min_dist_arcsec=float(_cfg.phase01_comparison_min_dist_arcsec),
        min_frames_frac=float(_cfg.phase01_comparison_min_frames_frac),
        rms_outlier_sigma=float(_cfg.phase01_comparison_rms_outlier_sigma),
        exclude_gaia_nss=bool(_cfg.phase01_comparison_exclude_gaia_nss),
        exclude_gaia_extobj=bool(_cfg.phase01_comparison_exclude_gaia_extobj),
        mag_bright_threshold=float(_cfg.phase01_comparison_mag_bright_threshold),
        max_mag_diff_bright_floor=float(
            _cfg.phase01_comparison_max_mag_diff_bright_floor or 0.0
        ),
        max_psf_chi2=float(_cfg.phase01_comparison_max_psf_chi2),
        max_fwhm_factor=float(_cfg.phase01_comparison_max_fwhm_factor),
        isolation_radius_px=float(_cfg.phase01_comparison_isolation_radius_px),
        flux_col=_cfg.phase01_flux_col,
        cfg=_cfg,
        progress_cb=progress_cb,
        draft_id=draft_id,
        db=db,
    )

    active_targets_csv = Path(str(p01.get("active_targets_csv") or ""))
    comparison_stars_csv = Path(str(p01.get("comparison_stars_csv") or ""))
    if not active_targets_csv.is_file() or not comparison_stars_csv.is_file():
        return {
            "phase01": p01,
            "phase2a": None,
            "output_dir": str(Path(output_dir)),
            "error": "Fáza 0+1 nevygenerovala active_targets/comparison_stars CSV.",
        }

    # ── FÁZA 2A ──
    _p("Fáza 2A: aperture photometry + lightcurves…")
    _cfg2a = p01.get("cfg_effective_for_photometry") or _cfg
    p2a = run_phase2a(
        masterstar_fits_path=Path(masterstar_fits_path),
        active_targets_csv=active_targets_csv,
        comparison_stars_csv=comparison_stars_csv,
        per_frame_csv_dir=Path(per_frame_csv_dir),
        detrended_aligned_dir=Path(detrended_aligned_dir),
        output_dir=Path(output_dir),
        fwhm_px=float(fwhm_px),
        cfg=_cfg2a,
        progress_cb=progress_cb,
        db=db,
        draft_id=draft_id,
        proc_frame_store=p01.get("proc_store"),
    )

    sysrem_result: dict[str, Any] | None = None
    if bool(_cfg.sysrem_enabled):
        _p("SysRem: removing systematic trends…")
        _sysrem_lc_dir = Path(output_dir) / "lightcurves"
        sysrem_result = run_sysrem_field(
            _sysrem_lc_dir,
            n_iter=int(_cfg.sysrem_n_iter),
        )
        logging.info(
            "[SysRem] %d stars | RMS improvement %.1f%% (%d iter)",
            int(sysrem_result.get("n_stars", 0)),
            float(sysrem_result.get("rms_improvement_pct", float("nan"))),
            int(sysrem_result.get("n_iter", 0)),
        )

    return {
        "phase01": p01,
        "phase2a": p2a,
        "sysrem": sysrem_result,
        "output_dir": str(Path(output_dir)),
        "proc_frame_store": p01.get("proc_store"),
    }


def _write_suspected_variables(
    ms_df: pd.DataFrame,
    csv_paths: list[Path],
    active_target_ids: set[str],
    output_path: Path,
    *,
    flux_col: str = "dao_flux",
    min_frames_frac: float = 0.5,
    outlier_sigma: float = 3.0,
    interior_fw: int | None = None,
    interior_fh: int | None = None,
    interior_margin_px: int | None = None,
    csv_cache: dict[str, pd.DataFrame] | None = None,
) -> None:
    """Detekuj hviezdy s vysokým RMS scatter ktoré nie sú v VSX — suspected new variables.

    Zapíše suspected_variables.csv s kolumnami:
    catalog_id, ra_deg, dec_deg, mag, comp_rms, n_frames, zone

    Ak sú zadané ``interior_*``, vyhodí sa pool aj per-frame body pri okrajoch čipu
    (rovnaký okraj ako pri aktívnych cieľoch a porovnávačkách vo ``run_phase0_and_phase1``).
    """
    # Usable hviezdy ktoré nie sú VSX ani active targets
    ms = ms_df.copy()
    for col in ("is_usable", "is_saturated", "is_noisy", "vsx_known_variable"):
        if col in ms.columns:
            ms[col] = _bool_col(ms[col])

    id_col = "catalog_id" if "catalog_id" in ms.columns else "name"
    base_mask = (
        _bool_col(ms.get("is_usable", pd.Series(True, index=ms.index)))
        & ~_bool_col(ms.get("is_saturated", pd.Series(False, index=ms.index)))
        & ~_bool_col(ms.get("is_noisy", pd.Series(False, index=ms.index)))
        & ~_bool_col(ms.get("vsx_known_variable", pd.Series(False, index=ms.index)))
    )
    pool = ms[base_mask].copy()
    pool["_nid"] = pool[id_col].map(_normalize_id_value)
    pool = pool[pool["_nid"] != ""].drop_duplicates(subset=["_nid"], keep="first")

    _m = int(interior_margin_px) if interior_margin_px is not None else 0
    _fw = int(interior_fw) if interior_fw is not None else 0
    _fh = int(interior_fh) if interior_fh is not None else 0
    if (
        _m > 0
        and _fw > 2 * _m
        and _fh > 2 * _m
        and "x" in pool.columns
        and "y" in pool.columns
    ):
        _xn = pd.to_numeric(pool["x"], errors="coerce")
        _yn = pd.to_numeric(pool["y"], errors="coerce")
        _ok = _xn.between(_m, _fw - _m) & _yn.between(_m, _fh - _m)
        _n_pool0 = int(len(pool))
        pool = pool[_ok].copy()
        logging.info(
            "[SUSPECTED] Orezanie okrajov (rovnaké ako Fáza 0/1, MASTERSTAR x,y): %s → %s hviezd (margin %s px, pole %s×%s)",
            _n_pool0,
            len(pool),
            _m,
            _fw,
            _fh,
        )

    pool_ids = set(pool["_nid"]) - active_target_ids

    if not pool_ids:
        pd.DataFrame().to_csv(output_path, index=False)
        return

    # Načítaj flux pre všetky hviezdy z poolu
    flux_map: dict[str, list[float]] = {cid: [] for cid in pool_ids}
    n_frames = 0
    _cache_hits = 0
    _cache_misses = 0

    for csv_path in csv_paths:
        try:
            _cache_key = str(csv_path)
            _cached = csv_cache.get(_cache_key) if csv_cache else None
            if _cached is not None and not _cached.empty:
                header_cols = _cached.columns
            else:
                header_cols = pd.read_csv(csv_path, nrows=0).columns
            actual_flux = flux_col if flux_col in header_cols else "flux"
            name_c = "catalog_id" if "catalog_id" in header_cols else "name"
            use = [name_c, actual_flux]
            if "mag" in header_cols and "mag" not in use:
                use.append("mag")
            _use_xy = _m > 0 and _fw > 2 * _m and _fh > 2 * _m
            if _use_xy and "x" in header_cols and "y" in header_cols:
                use.extend([c for c in ("x", "y") if c not in use])
            if _cached is not None and not _cached.empty:
                df = _cached[[c for c in use if c in _cached.columns]].copy()
                _cache_hits += 1
            else:
                df = read_vyvar_csv(csv_path, usecols=use, low_memory=False)
                _cache_misses += 1
            if name_c not in df.columns:
                continue
            df[name_c] = _normalize_id_series(df[name_c])
            df[actual_flux] = pd.to_numeric(df[actual_flux], errors="coerce")
            sub = df[df[name_c].isin(pool_ids) & df[actual_flux].gt(0)]
            if _use_xy and "x" in sub.columns and "y" in sub.columns:
                _xs = pd.to_numeric(sub["x"], errors="coerce")
                _ys = pd.to_numeric(sub["y"], errors="coerce")
                sub = sub[_xs.between(_m, _fw - _m) & _ys.between(_m, _fh - _m)]
            if sub.empty:
                continue

            # Mag-bin normalizácia: medián zvlášť pre každý mag bin (0.5 mag šírka)
            mag_col_frame = "mag" if "mag" in df.columns else None
            if mag_col_frame and mag_col_frame in sub.columns:
                sub = sub.copy()
                sub["_mag_num"] = pd.to_numeric(sub[mag_col_frame], errors="coerce")
                sub["_mag_bin"] = (sub["_mag_num"] / 0.5).apply(
                    lambda x: int(x) if math.isfinite(x) else -1
                )
                bin_meds: dict[int, float] = {}
                for b, grp in sub.groupby("_mag_bin"):
                    bmed = float(grp[actual_flux].median())
                    if math.isfinite(bmed) and bmed > 0:
                        bin_meds[int(b)] = bmed
                if not bin_meds:
                    continue
            else:
                # Fallback: globálny medián
                frame_med = float(sub[actual_flux].median())
                if not math.isfinite(frame_med) or frame_med <= 0:
                    continue
                bin_meds = {}

            n_frames += 1
            # Jedna vzorka na hviezdu na snímok (CSV môže mať duplicitné riadky).
            _agg: dict[str, dict[str, float]] = {}
            for _, row in sub.iterrows():
                cid = str(row[name_c])
                if cid not in pool_ids:
                    continue
                raw_flux = float(row[actual_flux])
                if not math.isfinite(raw_flux) or raw_flux <= 0:
                    continue
                mag_num = (
                    float(row.get("_mag_num", float("nan")))
                    if "_mag_num" in row.index
                    else float("nan")
                )
                ent = _agg.setdefault(cid, {"fluxes": [], "mags": []})
                ent["fluxes"].append(raw_flux)
                if math.isfinite(mag_num):
                    ent["mags"].append(mag_num)
            for cid, ent in _agg.items():
                fluxes = ent["fluxes"]
                if not fluxes:
                    continue
                raw_flux = float(np.median(np.asarray(fluxes, dtype=np.float64)))
                if not math.isfinite(raw_flux) or raw_flux <= 0:
                    continue
                mags = ent["mags"]
                mag_num = float(np.median(np.asarray(mags, dtype=np.float64))) if mags else float("nan")
                if bin_meds:
                    b = int(mag_num / 0.5) if math.isfinite(mag_num) else -1
                    norm_med = bin_meds.get(b)
                    if norm_med is None:
                        closest = min(bin_meds.keys(), key=lambda k: abs(k - b))
                        norm_med = bin_meds[closest]
                else:
                    norm_med = frame_med  # type: ignore[assignment]
                rel = raw_flux / norm_med
                if math.isfinite(rel) and rel > 0:
                    flux_map[cid].append(rel)
        except Exception as exc:  # noqa: BLE001
            logging.error('[EXC-0222] One frame skipped in suspected-variables flux accumulation - star RMS computed without ...: %s', exc)
            continue

    logging.info(
        "[PERF-1] _write_suspected_variables: %d cache hits, %d disk reads (of %d frames)",
        _cache_hits,
        _cache_misses,
        len(csv_paths),
    )
    if _cache_misses > 0:
        logging.warning(
            "[PERF-1] %d frames read from disk (not in shared_csv_cache) — "
            "check if csv_cache is populated before calling _write_suspected_variables",
            _cache_misses,
        )

    # Airmass detrending pre suspected variables
    for cid in list(flux_map.keys()):
        vals = flux_map[cid]
        if len(vals) < 6:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        t = np.linspace(0.0, 1.0, len(arr))
        try:
            coeffs = _safe_polyfit(t, arr, 2)
            if coeffs is None:
                continue
            trend_fit = np.polyval(coeffs, t)
            safe_trend = np.where(np.abs(trend_fit) > 1e-9, trend_fit, 1.0)
            detrended = arr / safe_trend
            med_dt = float(np.median(detrended))
            if math.isfinite(med_dt) and med_dt > 0:
                flux_map[cid] = (detrended / med_dt).tolist()
        except Exception as exc:  # noqa: BLE001
            logging.error('[EXC-0223] Detrend fit failure in suspected-variables leaves raw flux - false variable candidates ...: %s', exc)
            pass

    min_f = max(3, int(n_frames * min_frames_frac))
    rms_map: dict[str, float] = {}
    nframes_map: dict[str, int] = {}
    for cid, vals in flux_map.items():
        if len(vals) < min_f:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        rms = float(np.sqrt(np.mean((arr - 1.0) ** 2)))
        if math.isfinite(rms):
            rms_map[cid] = rms
            nframes_map[cid] = len(vals)

    if not rms_map:
        pd.DataFrame().to_csv(output_path, index=False)
        return

    _MAD_CONSISTENCY = 0.6745
    rms_arr = np.asarray(list(rms_map.values()), dtype=np.float64)
    med = float(np.median(rms_arr))
    mad_raw = float(np.median(np.abs(rms_arr - med)))
    if not math.isfinite(mad_raw) or mad_raw <= 0:
        # Fallback: ak MAD=0, použi normalizovanú std ako estimátor
        mad_sigma = float(np.std(rms_arr)) / _MAD_CONSISTENCY or 1e-9
    else:
        mad_sigma = mad_raw / _MAD_CONSISTENCY
    threshold = med + outlier_sigma * mad_sigma

    suspected = {cid: rms for cid, rms in rms_map.items() if rms > threshold}

    if not suspected:
        pd.DataFrame().to_csv(output_path, index=False)
        return

    rows = []
    pool_idx = pool.set_index("_nid", drop=False)
    for cid, rms in sorted(suspected.items(), key=lambda x: -x[1]):
        if cid not in pool_idx.index:
            continue
        r = pool_idx.loc[cid]
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        rows.append(
            {
                "catalog_id": cid,
                "ra_deg": r.get("ra_deg", float("nan")),
                "dec_deg": r.get("dec_deg", float("nan")),
                "mag": r.get("mag", float("nan")),
                "comp_rms": rms,
                "n_frames": nframes_map.get(cid, 0),
                "zone": r.get("zone", ""),
            }
        )

    out_df = pd.DataFrame(rows)
    try:
        from gaia_catalog_id import normalize_gaia_source_id_series  # noqa: PLC0415

        if "catalog_id" in out_df.columns:
            out_df["catalog_id"] = normalize_gaia_source_id_series(out_df["catalog_id"])
    except Exception as exc:  # noqa: BLE001
        logging.error('[EXC-0224] suspected_variables.csv catalog_id normalization fails - exported IDs may be float-corr...: %s', exc)
        pass
    out_df.to_csv(output_path, index=False)
    logging.info(
        f"[SUSPECTED] {len(out_df)} kandidátov na nové premenné → {output_path.name} "
        f"(threshold RMS > {threshold:.4f})"
    )


__all__ = [
    # photometry (legacy)
    "StressTestResult",
    "_get_lc_adaptive",
    "apply_reporting_postprocess",
    "check_comparison_stability",
    "common_field_intersection_bbox_px",
    "compute_aperture_correction",
    "compute_fwhm_gaussian_for_aperture_catalog",
    "compute_lc_rms_ooe",
    "compute_mag_calib_final",
    "compute_optimal_apertures",
    "compute_snr_optimal_aperture_table",
    "detect_outliers",
    "empirical_feature_mask_mag",
    "enhance_catalog_dataframe_aperture_bpm",
    "ensemble_normalize",
    "ensure_full_variable_targets_if_presel_stub",
    "load_epsf_metrics_for_draft",
    # photometry_phase2a (legacy)
    "measure_fwhm_from_masterstar",
    "pytics_iterative_weights",
    "read_flux_from_csv",
    "recommended_aperture_by_color",
    "resolve_apply_color_term",
    "run_full_photometry_pipeline",
    "run_phase0_and_phase1",
    "run_phase2a",
    "run_sysrem_field",
    "save_cutout_png",
    "save_field_map_png",
    "save_lightcurve_csv",
    "save_lightcurve_png",
    "save_target_field_map_png",
    "select_active_targets",
    "select_comparison_stars_per_target",
    "stress_test_relative_rms_from_sidecars",
    "vsx_is_known_variable_top3_per_bin",
]

