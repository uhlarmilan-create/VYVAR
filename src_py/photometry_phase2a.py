"""Moved from photometry_core.py (CONSOLIDATE-01E4). Facade re-exports these names."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence
import json
import logging
import math
import re
import time
from astropy.io import fits as astrofits
import numpy as np
import pandas as pd
from catalog_match_trust import is_wcs_untrusted_catalog_match_mode, normalize_catalog_match_mode
from config import AppConfig
from gaia_catalog_id import masterstar_row_gaia_key, normalize_gaia_source_id
from infolog import log_event
from proc_frame_store import ProcFrameStore
from stats_core import _flux_to_mag
from unit_resolver import plate_scale_arcsec_per_px_from_header, resolve_px_from_arcsec
from photometry_core import (
    ERR_BKG_MODE_EMPIRICAL,
    ERR_BKG_SOURCE_COL,
    ERR_BKG_SOURCE_EMPIRICAL,
    ERR_BKG_SOURCE_HOWELL_FALLBACK,
    LOGGER,
    SIGMA_BKG_AP_COL,
    SKY_ADU_PER_PX_ANNULUS_COL,
    SKY_SURFACE_BG_MEDIAN_ADU_COL,
    _COMP_QUALITY_JSON_META_KEYS,
    _EDGE_FILTER_NOTE_FAILED,
    _EDGE_FILTER_NOTE_OK,
    _GAIA_ID_DTYPE,
    _LC_QUALITY_FLAGS,
    _MAD_CONSISTENCY,
    _phase2a_process_one_target,
)
from photometry_lightcurve import _coerce_bool_cell


def parse_comp_quality_json_map(raw: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Return ``catalog_id`` -> ``{quality, note}`` from ``comp_quality_*.json`` (strip metadata keys).

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

def _build_csv_lookup(
    csv_df: pd.DataFrame,
    id_col: str,
) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    """Vytvori dva lookup mechanizmy:
    1. Primarny: dict {normalized_id -> row}
    2. Zalozny: riadky s numerickymi x,y pre nearest-neighbor match (plne stlpce CSV).

    Proc CSV z pipeline ma ``catalog_id`` casto ako float / vedecku notaciu (strata presnosti),
    zatial co ``name`` obsahuje presny Gaia ``source_id`` - indexujeme oboje (``setdefault``),
    aby Faza 2A netrafila NN na suseda namiesto spravnej porovnavacky.
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
    # Plna kopia: NN fallback musi vratit Series so vsetkymi stlpcami (dao_flux, casy, ...).
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
    """Hladaj hviezdu v CSV - primarne cez ID, fallback cez x,y."""
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
        "[FAZA 2A] CSV NN fallback ok: requested_cid=%s matched_csv_id=%s dist_px=%.2f tol=%.1f",
        cid,
        _mid,
        float(dists[j]),
        tol,
    )
    return _hit

def _sat_limit_peak_adu(cfg: AppConfig | None = None) -> float | None:
    """Hranica peak_max_adu z configu (volitelne). Bez globalneho fallbacku - saturacia z FITS/DB v pipeline."""
    _ = cfg
    return None

def _mad_sigma_or_std_floor(arr: np.ndarray) -> float:
    """MAD/0.6745; no finite filter. Zero MAD falls back to std/0.6745 or 1e-9."""
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    if not math.isfinite(mad) or mad <= 0:
        return float(np.std(arr)) / _MAD_CONSISTENCY or 1e-9
    return mad / _MAD_CONSISTENCY

def measure_fwhm_from_masterstar(
    masterstar_fits_path: Path,
    star_positions: pd.DataFrame,
    *,
    n_stars: int = 20,
    fit_box_fwhm: float = 8.0,
    dao_fwhm_hint: float = 3.5,
    ms_data: np.ndarray | None = None,
) -> float:
    """Zmeria skutocne Gaussian FWHM z MASTERSTAR FITS.

    Fituje 2D Gaussian na izolovane, nesaturovane hviezdy z ``star_positions``
    a vracia medianove FWHM v pixeloch. Toto je fyzikalne spravne FWHM
    (zodpoveda AIJ/IRAF definicii), na rozdiel od DAO odhadu ktory
    systematicky precenuje FWHM.

    Args:
        masterstar_fits_path: Cesta k MASTERSTAR.fits
        star_positions: DataFrame so stlpcami x, y, mag (catalog_id volitelny)
        n_stars: Pocet hviezd na fit (vyberie izolovane, stredne jasne)
        fit_box_fwhm: Velkost okna pre fit v jednotkach dao_fwhm_hint
        dao_fwhm_hint: Hruby DAO odhad pre urcenie velkosti okna

    Returns:
        Medianove Gaussian FWHM v pixeloch.
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
            "[FAZA 2A] Gaussian FWHM fit: prazdne star_positions, fallback dao_fwhm_hint=%.2f px",
            float(dao_fwhm_hint),
        )
        return float(dao_fwhm_hint)
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["x", "y"])
    if len(df) < 3:
        logging.warning(
            "[FAZA 2A] Gaussian FWHM fit: malo riadkov s x,y, fallback dao_fwhm_hint=%.2f px",
            float(dao_fwhm_hint),
        )
        return float(dao_fwhm_hint)

    box = int(fit_box_fwhm * dao_fwhm_hint)
    margin = box + 5
    if box < 3 or margin * 2 >= min(h, w):
        logging.warning(
            "[FAZA 2A] Gaussian FWHM fit: prilis maly/obrovsky box=%s, fallback dao_fwhm_hint=%.2f px",
            box,
            float(dao_fwhm_hint),
        )
        return float(dao_fwhm_hint)

    df = df[(df["x"] > margin) & (df["x"] < w - margin) & (df["y"] > margin) & (df["y"] < h - margin)].copy()
    if len(df) < 3:
        logging.warning(
            "[FAZA 2A] Gaussian FWHM fit: malo hviezd po okrajovom filtri, fallback dao_fwhm_hint=%.2f px",
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
            "[FAZA 2A] Gaussian FWHM fit: prazdny vyber po mag, fallback dao_fwhm_hint=%.2f px",
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
            "[FAZA 2A] Gaussian FWHM fit: len menej ako 3 hviezd (%s), fallback dao_fwhm_hint=%.2f px",
            len(fwhm_values),
            float(dao_fwhm_hint),
        )
        return float(dao_fwhm_hint)

    result = float(np.median(fwhm_values))
    logging.info(
        "[FAZA 2A] Gaussian FWHM z MASTERSTAR: %.3f px (z %s hviezd, DAO hint %.3f px)",
        result,
        len(fwhm_values),
        float(dao_fwhm_hint),
    )
    return result

def compute_optimal_apertures(
    masterstar_fits_path: Path,
    star_positions: pd.DataFrame,
    fwhm_px: float,
    *,
    aperture_fwhm_factor: float = 1.75,
    annulus_inner_fwhm: float = 4.5,
    annulus_outer_fwhm: float = 6.0,
) -> dict[str, float]:
    """Globalna fixna apertura = aperture_fwhm_factor x FWHM.

    Fyzikalne zdovodnenie:
    - PSF FWHM (typicky ``VY_FWHM`` DAO z MASTERSTAR): r ~ 1.75x FWHM zachyti vacsinu fluxu
    - Konzistentna fixna apertura je robustnejsia ako per-hviezda
      metody v hustom poli (kontaminacia susedmi)
    - Zodpoveda AIJ metodike: fixna apertura z FWHM

    Args:
        masterstar_fits_path: Nepouziva sa - zachovane pre kompatibilitu.
        star_positions: DataFrame so stlpcami catalog_id (volitelne name).
        fwhm_px: FWHM v pixeloch (Faza 2A: ``VY_FWHM`` z hlavicky alebo Gaussian fit).
        aperture_fwhm_factor: Nasobok FWHM. Default 1.75.
        annulus_inner_fwhm: Zachovane pre kompatibilitu signatury.
        annulus_outer_fwhm: Zachovane pre kompatibilitu signatury.

    Returns:
        dict {catalog_id: apertura_px} - vsetky hviezdy maju rovnaku hodnotu.
    """
    _ = masterstar_fits_path
    _ = annulus_inner_fwhm
    _ = annulus_outer_fwhm

    global_ap = float(aperture_fwhm_factor * fwhm_px)

    logging.info(
        f"[FAZA 2A] Globalna apertura: {global_ap:.3f}px "
        f"({aperture_fwhm_factor:.2f}x FWHM={fwhm_px:.3f}px)"
    )

    result: dict[str, float] = {}
    for _, row in star_positions.iterrows():
        cid = _normalize_gaia_id(row.get("catalog_id", row.get("name", "")))
        if cid:
            result[cid] = global_ap

    return result

def _howell_variance_adu2(
    flux: float,
    sky_pp: float,
    area: float,
    gain: float = 1.0,
    read_noise: float = 10.0,
) -> float:
    """Total variance [ADU^2] from a reduced Howell (1989) eq. 2 form.

    Implemented terms: source Poisson (``flux/gain``), sky Poisson on the aperture
    (``sky_pp/gain * area``), and read noise (``(read_noise/gain)^2 * area``).

    Omitted (not in this helper): dark-current shot noise, the ``(1 + n_pix/n_B)``
    sky-estimation factor, flat-field noise, and digitisation/quantisation noise.
    """
    if not math.isfinite(flux) or flux <= 0:
        return float("nan")
    if not math.isfinite(sky_pp) or sky_pp < 0:
        sky_pp = 0.0
    if not math.isfinite(area) or area <= 0:
        return float("nan")
    g = float(gain) if math.isfinite(gain) and gain > 0 else 1.0
    rn = float(read_noise) if math.isfinite(read_noise) and read_noise >= 0 else 10.0
    return flux / g + max(0.0, sky_pp) / g * area + (rn / g) ** 2 * area

def _photometric_error(
    flux: float,
    sky_pp: float,
    area: float,
    gain: float = 1.0,
    read_noise: float = 10.0,
) -> float:
    """Relative photometric error from Poisson + read-noise variance (Howell 1989 eq. 2).

    Units at boundary: ``flux`` and ``sky_pp`` in ADU (per px for sky); ``gain`` in e-/ADU;
    ``read_noise`` in e-; internal variance in ADU^2. Returns dimensionless err/flux.
    Legacy ``howell`` mode - byte-identical to pre F-BINGAIN-1 behaviour.
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
    """Relative err/flux with empirical background term (F-BINGAIN-1).

    Empirical (production): ``var = F/g + sigma_bkg_ap^2`` - photutils/SExtractor
    pattern with measured ``sigma_bkg`` at aperture scale (Labbe et al. 2003).
    When ``sigma_bkg_ap`` is missing, Howell variance is the data-conditioned
    fallback. ``err_background_mode`` is ignored (CONSOLIDATE-01D; key removed).
    """
    _ = err_background_mode
    if not math.isfinite(flux) or flux <= 0:
        return float("nan"), ERR_BKG_SOURCE_HOWELL_FALLBACK
    g = float(gain) if math.isfinite(gain) and gain > 0 else 1.0
    sig_ap = float(sigma_bkg_ap) if sigma_bkg_ap is not None else float("nan")
    if math.isfinite(sig_ap) and sig_ap >= 0:
        variance = flux / g + sig_ap * sig_ap
        if math.isfinite(variance) and variance >= 0:
            return math.sqrt(variance) / flux, ERR_BKG_SOURCE_EMPIRICAL
    err = _photometric_error(flux, sky_pp, area, gain=gain, read_noise=read_noise)
    return err, ERR_BKG_SOURCE_HOWELL_FALLBACK

def _phase2a_proc_column_requirements() -> dict[str, list[str]]:
    """Named proc-CSV column requirements for headless Phase 2A.

    This is the contract that the reduced Phase-2A CSV cache must satisfy. The
    UI/full-pipeline path reads from ``ProcFrameStore``; the headless path must
    project every column that any selectable Phase-2A branch can consume.
    """
    return {
        "lookup_identity": [
            "catalog_id",
            "name",
            "x",
            "y",
        ],
        "frame_times_and_trust": [
            "bjd_tdb_mid",
            "hjd_mid",
            "jd_mid",
            "airmass",
            "catalog_match_mode",
        ],
        "photometry_core": [
            "dao_flux",
            "aperture_r_px",
            "noise_floor_adu",
            "sky_adu_per_px_annulus",
            SKY_SURFACE_BG_MEDIAN_ADU_COL,
            "flux_small",
            "flux_large",
        ],
        "err_empirical": [
            SIGMA_BKG_AP_COL,
            ERR_BKG_SOURCE_COL,
        ],
        "pfs_and_sat": [
            "peak_max_adu",
            "is_saturated",
            "likely_saturated",
        ],
        "edge_gating": [
            "sky_annulus_r_out_px",
            "edge_fail",
            "edge_safe_10px",
        ],
        "variability_and_export": [
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
            "snr50_ok",
            "is_usable",
        ],
        "psf_branch": [
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
            "psf_ac_policy",
        ],
        "cog_branch": [
            "ac_factor",
            "dao_flux_apcorr",
            "cog_ok",
        ],
    }

def _phase2a_cache_columns() -> list[str]:
    """Union of all required proc-CSV columns for headless Phase 2A."""
    cols: list[str] = []
    for required in _phase2a_proc_column_requirements().values():
        cols.extend(required)
    return list(dict.fromkeys(cols))

def _phase2a_empirical_sigma_bkg_ap(
    row_csv: pd.Series,
    *,
    err_background_mode: str,
    source_file: str,
    catalog_id: str,
) -> float | None:
    """Return per-row empirical ``sigma_bkg_ap`` or raise on missing input.

    INV-ERR-MODE-01: Phase 2A empirical background errors are the only policy
    (CONSOLIDATE-01D); missing required proc-CSV inputs are a hard failure, not a
    silent Howell fallback caused by a starved cache projection.
    """
    sig = float(pd.to_numeric(row_csv.get(SIGMA_BKG_AP_COL), errors="coerce"))
    if math.isfinite(sig):
        return sig
    _ = err_background_mode
    src_raw = row_csv.get(ERR_BKG_SOURCE_COL, "")
    src = str(src_raw).strip().lower() if src_raw is not None else ""
    raise ValueError(
        "[INV-ERR-MODE-01] err_background_mode=empirical requires "
        f"'{SIGMA_BKG_AP_COL}' in Phase 2A input; missing/NaN for catalog_id={catalog_id} "
        f"in {source_file} (err_bkg_source={src or 'missing'})."
    )

def _sky_pp_for_photometric_error(row: Any) -> float:
    """Sky level (ADU/px) for Howell ``_photometric_error`` from a proc-CSV row.

    I-11: prefer pre-subtraction ``sky_surface_bg_median_adu`` for the Howell sky Poisson
    term (photons that arrived before the 2D sky surface was subtracted). Post-subtraction
    annulus sky collapses toward zero and under-quotes the sky term on the legacy path.

    Fall back to ``sky_adu_per_px_annulus`` then ``noise_floor_adu`` for older proc CSVs.
    """
    pre_sub = float(pd.to_numeric(row.get(SKY_SURFACE_BG_MEDIAN_ADU_COL), errors="coerce"))
    if math.isfinite(pre_sub) and pre_sub >= 0:
        return pre_sub
    ann = float(pd.to_numeric(row.get(SKY_ADU_PER_PX_ANNULUS_COL), errors="coerce"))
    if math.isfinite(ann) and ann >= 0:
        return ann
    legacy = float(pd.to_numeric(row.get("noise_floor_adu"), errors="coerce"))
    if math.isfinite(legacy) and legacy >= 0:
        return legacy
    return 0.0

def _resolve_phase2a_equipment_id(
    db: Any | None,
    *,
    draft_id: int | None,
    output_dir: Path,
    masterstar_fits_path: Path,
) -> int | None:
    """``draft manifest.ID_EQUIPMENTS`` from ``draft_id`` or path segment ``draft_NNN``."""
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
        if hasattr(db, "get_draft_equipment_id"):
            return db.get_draft_equipment_id(int(did))
    except Exception as exc:  # noqa: BLE001
        logging.error('[EXC-0125] draft manifest equipment_id DB read fails - gain/RN resolver falls back without equipment scope: %s', exc)
        return None
    return None

def _draft_dir_from_phase2a_paths(output_dir: Path, masterstar_fits_path: Path) -> Path:
    for base in (Path(output_dir), Path(masterstar_fits_path)):
        for parent in [base, *base.parents]:
            if re.match(r"draft_\d+$", parent.name, re.IGNORECASE):
                return parent
            if re.match(r"draft_\d+_snapshot", parent.name, re.IGNORECASE):
                return parent
            if (parent / "calibrated" / "lights" / "qc_metrics.csv").is_file():
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

def _resolve_photometric_aperture_px_for_gs11(
    target_cid: str,
    apertures_px: dict[str, float],
    target_g_mag: float,
    snr_ap_table: dict[str, Any] | None,
    *,
    aperture_fwhm_factor: float,
    fwhm_px: float,
) -> tuple[float | None, str]:
    """Photometric aperture for GS11 dilution: Phase 2A per-star map, else skip.

    CONSOLIDATE-01B C2: the SNR mag-bin table is gone. No fixed-pixel fallback.
    """
    _ = (target_g_mag, snr_ap_table, aperture_fwhm_factor, fwhm_px)
    cid = _normalize_gaia_id(target_cid) if target_cid else ""
    if cid and cid in apertures_px:
        ap = float(apertures_px[cid])
        if math.isfinite(ap) and ap > 0:
            return ap, "map"
    return None, "unavailable"

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
    """Krok 2: Nacitaj flux z per-frame CSV (dao_flux).

    Namiesto citania FITS a vlastnej aperturnej fotometrie pouziva
    dao_flux ktory pipeline vypocitala pocas DAO detekcie.
    dao_flux je sky-subtrahovany flux zmerany s aperture_r_px z CSV.

    Returns:
        DataFrame: catalog_id, bjd, hjd, jd, airmass, mag_inst, err,
                   aperture_r_px, sky_pp, flag, source_file

    Args:
        lookup: Volitelny vystup z ``_build_csv_lookup`` pre zdielany ``csv_df``
            (Faza 2A - jedna vystavba lookupu na snimku namiesto 1x na target).
    """
    if csv_df is None:
        if not Path(frame_csv_path).is_file():
            return pd.DataFrame()
        try:
            # Keep Gaia IDs stable (avoid float/scientific precision loss in per-frame CSV).
            csv_df = pd.read_csv(frame_csv_path, low_memory=False, dtype=_GAIA_ID_DTYPE)
        except (OSError, ValueError) as exc:
            logging.warning(f"[FAZA 2A] Nemozem citat CSV {frame_csv_path}: {exc}")
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
            # PSF photometry (b.5) columns - carried through so Phase 2A star-method / adaptive
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
                            "[FAZA 2A] XY fallback wrong star: cid=%s, fallback_mag=%.2f > -8.0, "
                            "nastavujem NaN",
                            cid,
                            fallback_mag,
                        )
                        rows.append(base)
                        continue

        # PSF photometry (b.5) - read per-star/per-frame PSF flux + quality if present.
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

        # Casove znacky
        base["bjd"] = float(row_csv.get("bjd_tdb_mid", float("nan")))
        base["hjd"] = float(row_csv.get("hjd_mid", float("nan")))
        base["jd"] = float(row_csv.get("jd_mid", float("nan")))

        # Airmass fallback: ak frame_times nebolo dostupne, citaj priamo z CSV riadku
        if not math.isfinite(am_frame):
            am_csv = float(row_csv.get("airmass", float("nan")))
            if math.isfinite(am_csv):
                base["airmass"] = am_csv

        # dao_flux - sky-subtrahovany flux z DAO fotometrie
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

        # Apertura z CSV (ta co pipeline pouzila pri DAO)
        ap_csv = float(row_csv.get("aperture_r_px", float("nan")))
        if math.isfinite(ap_csv) and ap_csv > 0:
            base["aperture_r_px"] = ap_csv

        # Sky per pixel for Howell err: explicit annulus column, legacy noise_floor fallback.
        sky_pp = _sky_pp_for_photometric_error(row_csv)
        if math.isfinite(sky_pp):
            base["sky_pp"] = sky_pp

        # Saturacia
        peak = float(row_csv.get("peak_max_adu", float("nan")))
        is_sat = math.isfinite(peak) and math.isfinite(_sat_lim) and peak > _sat_lim

        if flux <= 0:
            base["flag"] = "no_data"
            rows.append(base)
            continue

        # Instrumentalna magnituda
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

        # Chyba - fotonovy sum + background (empirical empty-aperture or Howell legacy)
        r_ap = base["aperture_r_px"]
        area = math.pi * r_ap * r_ap if math.isfinite(r_ap) and r_ap > 0 else float("nan")
        _sig_bkg = _phase2a_empirical_sigma_bkg_ap(
            row_csv,
            err_background_mode=err_background_mode,
            source_file=source_file,
            catalog_id=str(cid),
        )
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

def compute_aperture_correction(
    comp_df: pd.DataFrame,
    frame_results: Sequence[pd.DataFrame],
    min_ref_stars: int = 3,
    max_contamination: float = 0.15,
    max_scatter_mag: float = 0.03,
) -> dict[str, Any]:
    """Metoda B: median DeltaM_corr = mag_large - mag_small medzi referencnymi comp cez framy.

    ``frame_results``: jeden DataFrame na snimku (vystup ``read_flux_from_csv``), riadky = hviezdy,
    kluc hviezdy je ``catalog_id`` (zhodne s Fazou 1).

    Ref hviezdy pre DeltaM_corr: preferuj T1, ak < min_ref_stars dopln T2; potom contamination filter.
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

    # --- KROK A: referencne hviezdy ---
    df = comp_df.copy()
    if "comp_tier" not in df.columns:
        return _fail("no_comp_tier")
    df["comp_tier"] = pd.to_numeric(df["comp_tier"], errors="coerce").fillna(4).astype(int)
    ref_t1 = df[df["comp_tier"] == 1].copy()
    ref_t2 = df[df["comp_tier"] == 2].copy()

    # Prefer T1, then T1+T2. If still empty (all TIER3/4 under COMP-ADMIT-03),
    # use the full candidate set for AC refs - colour tiers are display-only now.
    ref_stars = ref_t1
    if int(len(ref_stars)) < int(min_ref_stars):
        ref_stars = pd.concat([ref_t1, ref_t2], ignore_index=True)
    if int(len(ref_stars)) < int(min_ref_stars):
        ref_stars = df.copy()

    # Column presence must be checked on a non-empty schema source: filtering an
    # empty frame can drop columns (pandas), which falsely reported no_comp_rms
    # on draft 514 when all comps were TIER4.
    if "contamination_idx" not in df.columns:
        return _fail("no_contamination_idx")
    if "comp_rms" not in df.columns:
        return _fail("no_comp_rms")

    # Apply contamination filter
    ref_stars = ref_stars[
        ref_stars["contamination_idx"].apply(
            lambda x: float(x) <= float(max_contamination) if pd.notna(x) else False
        )
    ]

    # Apply comp_rms filter
    cr = pd.to_numeric(ref_stars["comp_rms"], errors="coerce")
    ref_stars = ref_stars[np.isfinite(cr.to_numpy(dtype=float)) & (cr.to_numpy(dtype=float) > 0)].copy()

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

    # --- KROK B: median Deltam per ref hviezda cez framy ---
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

    Pre kazdu good/suspect COMP hviezdu:
      x_i = bp_rp_i - median(bp_rp vsetkych pouzitych COMP)
      y_i = median(cat_mag_i - inst_mag_i)  [cez vsetky framy]

    Linearny fit: y = c1 * x + ZP_offset

    Returns: (c1, c1_stderr, n_comp_used)
    Pri chybe alebo malo COMP: (0.0, nan, 0)
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

    _ = (sigma_clip_sigma, c1_init, zp_init)  # no residual sigma-clip (zero-clipping 2026-08-12)
    fit_cl = _safe_polyfit(x, y, 1, cov=True)
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
        "[COLOR TERM] c1=%.4f +- %.4f, bp_rp_range=[%.2f, %.2f], n_comp=%s, sigma_clip_removed=0",
        c1,
        c1_stderr,
        bp_min,
        bp_max,
        int(x.size),
    )
    return c1, c1_stderr, int(x.size)

def should_apply_color_term(
    obs_group: str,
    c1: float,
    c1_stderr: float,
    n_comp: int,
    *,
    min_comp_for_ct: int = 7,
    max_stderr_ratio: float = 0.5,
    cfg: Any | None = None,
) -> tuple[bool, str]:
    """
    Auto-rozhodnutie ci aplikovat color term korekciu.

    Returns: (apply: bool, reason: str)
    reason = kratky popis preco sa CT aplikuje alebo nie
    """
    from band_classify import (
        band_failsafe_clear,
        classify_photometric_band,
        color_term_auto_from_band,
    )

    filter_raw = str(obs_group or "").split("|")[0].strip()
    band = classify_photometric_band(obs_group)

    # Clear / unfiltered: fixed per-rig level coefficient (export-only), not a night fit.
    if band_failsafe_clear(band):
        k = getattr(cfg, "color_level_k_mag_per_bprp", None) if cfg is not None else None
        k_se = (
            getattr(cfg, "color_level_k_stderr_mag_per_bprp", None) if cfg is not None else None
        )
        if k is not None and math.isfinite(float(k)) and abs(float(k)) > 1e-6:
            se = float(k_se) if k_se is not None and math.isfinite(float(k_se)) else float("nan")
            return True, (
                f"{band.value} ({filter_raw}) - clear level k={float(k):+.4f}"
                + (f" +- {se:.4f}" if math.isfinite(se) else "")
                + " mag/BP-RP (export-only)"
            )
        return False, f"{band.value} ({filter_raw}) - CT nie je potrebny (no color_level_k)"

    if not color_term_auto_from_band(band):
        return False, f"{band.value} ({filter_raw}) - CT nie je potrebny"

    try:
        n_comp_i = int(n_comp)
    except Exception:  # noqa: BLE001
        n_comp_i = 0
    if n_comp_i < int(min_comp_for_ct):
        return (
            False,
            (
                f"Filter {filter_raw} - CT preskoceny: "
                f"malo COMP ({n_comp_i} < {int(min_comp_for_ct)})"
            ),
        )

    if not (float(c1) != 0.0 and abs(float(c1)) > 1e-6):
        return False, f"Filter {filter_raw} - CT preskoceny: c1 ~ 0"

    stderr_ratio = abs(float(c1_stderr) / float(c1)) if float(c1) != 0.0 else float("inf")
    if not math.isfinite(stderr_ratio):
        return False, f"Filter {filter_raw} - CT nespolahlivy: stderr/c1=NaN"
    if float(stderr_ratio) > float(max_stderr_ratio):
        return (
            False,
            (
                f"Filter {filter_raw} - CT nespolahlivy: "
                f"stderr/c1={stderr_ratio:.2f} > {float(max_stderr_ratio):.2f}"
            ),
        )

    return True, (
        f"Filter {filter_raw} - CT aplikovany: "
        f"c1={float(c1):+.4f} +- {float(c1_stderr):.4f} "
        f"(stderr/c1={stderr_ratio:.2f}, n_comp={n_comp_i})"
    )

def _obs_group_filter_key(obs_group: str) -> str:
    raw = str(obs_group or "").split("|")[0].strip()
    part = raw.split("_")[0].strip()
    return part.lower() if part else raw.lower()

def resolve_apply_color_term(
    cfg: Any | None,
    obs_group: str,
    *,
    fits_filter: str | None = None,
    aavso_code: str | None = None,
) -> bool:
    """User/config toggle: CT applies correction only - never limits the target set."""
    from band_classify import (
        band_failsafe_clear,
        classify_photometric_band,
        color_term_auto_from_band,
    )

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
    if color_term_auto_from_band(band):
        return True
    # Clear/unfiltered auto: enable only when a measured per-rig level k is set.
    if band_failsafe_clear(band):
        k = getattr(cfg, "color_level_k_mag_per_bprp", None)
        try:
            return k is not None and math.isfinite(float(k)) and abs(float(k)) > 1e-6
        except (TypeError, ValueError):
            return False
    return False

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
    mode: str = "fit"  # "fit" | "clear_level"

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
    """Inst magnitudes for global comp pool - one array per comp across all frames."""
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

def _phase2a_attempt_k2_night_fit(
    *,
    cfg: Any,
    obs_group: str,
    flux_matrix: pd.DataFrame,
    csv_files: list[Path],
    comparison_stars_csv: Path,
    masterstar_fits_path: Path,
) -> Any:
    """Build flux-derived Honeycutt inputs and run NIGHT_FIT (S6). Returns None if not enabled."""
    from k2_extinction import (  # noqa: PLC0415
        airmass_from_proc_csvs,
        attempt_k2_night_fit_from_arrays,
        select_k2_fit_frames_readonly,
    )

    if flux_matrix is None or getattr(flux_matrix, "empty", True):
        return attempt_k2_night_fit_from_arrays(
            cfg,
            obs_group,
            mag_inst=np.array([], dtype=np.float64),
            colour=np.array([], dtype=np.float64),
            airmass=np.array([], dtype=np.float64),
            star_index=np.array([], dtype=np.int64),
            frame_index=np.array([], dtype=np.int64),
        )
    comp_bp_rp, _, comp_quality = _comp_maps_from_comparison_stars_csv(Path(comparison_stars_csv))
    comp_ids = [
        cid
        for cid, q in comp_quality.items()
        if q.get("quality") != "excluded" and cid in comp_bp_rp
    ]
    if len(comp_ids) < 3:
        comp_ids = sorted(comp_bp_rp.keys())
    mag_map = _group_comp_mag_inst_from_flux_matrix(flux_matrix, comp_ids, csv_files)
    am = airmass_from_proc_csvs(csv_files)
    # READ-ONLY fit-frame subset from always-on align_residual_px (when present).
    align_res = np.full(len(csv_files), float("nan"), dtype=np.float64)
    try:
        align_rep = Path(masterstar_fits_path).resolve().parent / "alignment_report.csv"
        if align_rep.is_file():
            rep = pd.read_csv(align_rep, low_memory=False)
            if "file" in rep.columns and "align_residual_px" in rep.columns:
                by_stem = {
                    Path(str(r["file"])).stem: float(
                        pd.to_numeric(r.get("align_residual_px"), errors="coerce")
                    )
                    for _, r in rep.iterrows()
                    if str(r.get("file", "")).strip()
                }
                for i, p in enumerate(csv_files):
                    align_res[i] = by_stem.get(p.stem, float("nan"))
    except Exception as exc:  # noqa: BLE001
        logging.debug("[K2-FIT] align_residual read failed: %s", exc)
    _apr = float(getattr(cfg, "aperture_fwhm_factor", 1.9) or 1.9) * 3.0
    _max_res = float(getattr(cfg, "frame_align_residual_max_frac", 0.25) or 0.25) * _apr
    frame_ok = select_k2_fit_frames_readonly(
        len(csv_files),
        align_residual_px=align_res,
        align_residual_max_px=_max_res,
    )

    mag_l: list[float] = []
    col_l: list[float] = []
    am_l: list[float] = []
    si_l: list[int] = []
    fi_l: list[int] = []
    br_l: list[float] = []
    for si, cid in enumerate(comp_ids):
        series = mag_map.get(cid)
        if series is None:
            continue
        bp = float(comp_bp_rp.get(cid, float("nan")))
        for fi in range(len(csv_files)):
            if not frame_ok[fi]:
                continue
            mv = float(series[fi])
            av = float(am[fi]) if fi < len(am) else float("nan")
            if not math.isfinite(mv) or not math.isfinite(av) or not math.isfinite(bp):
                continue
            mag_l.append(mv)
            col_l.append(bp)
            am_l.append(av)
            si_l.append(si)
            fi_l.append(fi)
            br_l.append(mv)
    return attempt_k2_night_fit_from_arrays(
        cfg,
        obs_group,
        mag_inst=np.asarray(mag_l, dtype=np.float64),
        colour=np.asarray(col_l, dtype=np.float64),
        airmass=np.asarray(am_l, dtype=np.float64),
        star_index=np.asarray(si_l, dtype=np.int64),
        frame_index=np.asarray(fi_l, dtype=np.int64),
        brightness=np.asarray(br_l, dtype=np.float64),
        frame_ok=None,  # already filtered
    )

def _compute_group_color_term_fit(
    *,
    comparison_stars_csv: Path,
    flux_matrix: pd.DataFrame,
    csv_files: list[Path],
    obs_group: str,
    cfg: Any,
    k2_value: float | None = None,
    k2_source: Any | None = None,
) -> _ColorTermGroupFit | None:
    from band_classify import band_failsafe_clear, classify_photometric_band  # noqa: PLC0415

    comp_csv = Path(comparison_stars_csv)
    if not comp_csv.is_file():
        return None
    comp_bp_rp, comp_catalog_mag, comp_quality = _comp_maps_from_comparison_stars_csv(comp_csv)
    comp_ids = sorted(comp_bp_rp.keys())
    if not comp_ids:
        return None

    band = classify_photometric_band(obs_group)
    k_level = getattr(cfg, "color_level_k_mag_per_bprp", None)
    k_se = getattr(cfg, "color_level_k_stderr_mag_per_bprp", None)
    if band_failsafe_clear(band) and k_level is not None:
        try:
            k_ok = math.isfinite(float(k_level)) and abs(float(k_level)) > 1e-6
        except (TypeError, ValueError):
            k_ok = False
        if k_ok:
            se = float("nan")
            try:
                if k_se is not None and math.isfinite(float(k_se)):
                    se = float(k_se)
            except (TypeError, ValueError):
                se = float("nan")
            reason = (
                f"{band.value} clear_level k={float(k_level):+.4f}"
                + (f" +- {se:.4f}" if math.isfinite(se) else "")
                + " mag/BP-RP (export-only; shape term null)"
            )
            logging.info("[COLOR TERM] %s", reason)
            return _ColorTermGroupFit(
                c1=float(k_level),
                c1_stderr=float(se),
                n_comp=int(len(comp_ids)),
                comp_bp_rp=comp_bp_rp,
                comp_quality=comp_quality,
                comp_catalog_mag=comp_catalog_mag,
                apply_gate=True,
                gate_reason=reason,
                mode="clear_level",
            )

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

    if k2_value is not None and k2_source is not None:
        _k2_val, _k2_src = float(k2_value), k2_source
    else:
        _k2_val, _k2_src = resolve_k2_bprp_value(cfg, obs_group)
    if _k2_src in (K2Source.LITERATURE_DEFAULT, K2Source.NIGHT_FIT) and math.isfinite(
        float(_k2_val)
    ):
        _bp_med = bp_rp_comp_median(comp_bp_rp, comp_quality)
        if math.isfinite(_bp_med):
            comp_mag_inst = apply_k2_to_comp_mag_inst(
                comp_mag_inst,
                comp_bp_rp,
                comp_quality,
                airmass_from_proc_csvs(csv_files),
                float(_k2_val),
                _bp_med,
                k2_source=_k2_src,
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
        cfg=cfg,
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
        mode="fit",
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
            n_comparison_stars=int(getattr(cfg, "comparison_stars_pool_n", 0) or 0),
            require_non_variable=bool(getattr(cfg, "phase01_comparison_require_non_variable", True)),
            draft_id=int(draft_id) if draft_id is not None else None,
            database_path=getattr(cfg, "database_path", None),
        )
        log_event(f"[PHOT] Refreshed comparison_stars.csv pool ({n_pool}->spatial grid) for CT fit in {ps_dir.name}")
    except Exception as exc:  # noqa: BLE001
        logging.error('[EXC-0137] comparison_stars.csv spatial pool refresh fails - CT fit uses stale/sparse comp pool: %s', exc)
        logging.warning("[PHOT] comparison_stars pool refresh failed: %s", exc)
    return comp_csv

def _target_row_is_vsx_known_variable(target_row: pd.Series) -> bool:
    """True when target is a catalogued variable (VSX name/type), not a Gaia-only label."""
    vn = str(target_row.get("vsx_name", target_row.get("name", "")) or "").strip()
    if vn and vn.lower() not in ("nan", "none", "-", "-"):
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
    sigma = _mad_sigma_or_std_floor(finite_vals)
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
    """Democratic Detrender - ensemble multi-model detrending (arXiv:2411.09753v2, 2026).

    Reference: Caballero-Nieves et al. (2026) arXiv:2411.09753v2 -
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
        enabled:      False -> return (mag_calib copy, zeros)

    Returns:
        (mag_democratic, err_inflation)
        mag_democratic: marginalized detrended mag (length = n_frames)
        err_inflation:  per-frame MAD across models (0 if only 1 model)
    """
    if not enabled:
        return mag_calib.copy(), np.zeros(len(mag_calib))

    from scipy.signal import savgol_filter  # lazy import - scipy already in deps

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
    """Ulozi prehladovy PNG celeho pola - cervene=target, zelene=comp."""
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

    # Zelene stvorceky - comp hviezdy (unikatne pozicie)
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
        logging.info(f"[FIELD MAP] Skipped {int(skipped_as_target)} comp markers - star is a known target")

    # Target hviezdy - DAO-matched only (exclude catalog_only from field map)
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
        "VYVAR - Field Map (red=VSX target, green=comp star)",
        fontsize=10,
    )
    ax.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)

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
        _arc_margin = cfg_dict.get("phase01_chip_interior_margin_arcsec")
        _legacy_margin = float(cfg_dict.get("phase01_chip_interior_margin_px", 100))
        base_margin = resolve_px_from_arcsec(
            _arc_margin,
            _legacy_margin,
            plate_scale_arcsec_per_px_from_header(hdr),
            param_name="phase01_chip_interior_margin_px",
        )
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
        mag_limit = float(cfg.variability_mag_limit)
    except Exception:  # noqa: BLE001
        mag_limit = 14.5

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

    results_df["detection_method"] = "-"
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
    zone_filter: tuple[str, ...] = ("linear",),
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

    if zf in ("noisy1", "noisy2", "noisy3"):
        zf = "noise"

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

    noisy_from_zone = zf == "noise"
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
    location_id: int | None = None,
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
    out_loc_id = int(location_id) if location_id is not None and int(location_id) > 0 else loc_id
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
        "location_id": out_loc_id,
        "source": out_source,
    }

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
    site_location_id: int | None = None
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
    #: Night-level COG gate (APCORR-MIXEDFRAME-ALLORNOTHING). When True, Phase 2A
    #: routes ``dao_flux_apcorr`` into mag_inst; when False, every row uses
    #: standard ``dao_flux`` (Metoda B AC chain). Default False matches
    #: ``cog_aperture_correction_enabled`` default OFF.
    use_apcorr_flux: bool = False
    #: Provenance: True when COG was enabled but any frame lacked ``cog_ok`` and
    #: the whole night fell back to the standard AC path.
    cog_night_fallback: bool = False
    cog_night_fallback_n_without_ok: int = 0
    cog_night_fallback_n_frames: int = 0
    #: PER-FRAME-SAT-GATED night meta (empty when flag OFF - INV-CFG-01).
    per_frame_sat_meta: dict[str, Any] = field(default_factory=dict)
    #: NIGHT_FIT v2 meta (empty when k2_fit_enabled=False).
    k2_fit_meta: dict[str, Any] = field(default_factory=dict)
    #: APERTURE-01 policy record (mode, f, night FWHM, r_ap/r_in/r_out).
    aperture_policy: dict[str, Any] | None = None

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
        logging.warning("Observer location not set - lunar context skipped")
        return None
    jd_vals = _phase2a_collect_session_jd_values(state.frame_time_lookup)
    if not jd_vals:
        logging.warning("[PHASE 2A] No frame JD values - lunar context skipped")
        return None
    try:
        jd_mid = get_jd_midpoint(jd_vals)
    except ValueError:
        logging.warning("[PHASE 2A] JD midpoint unavailable - lunar context skipped")
        return None
    ra_field, dec_field, src = _phase2a_resolve_field_center_ra_dec(state._ms_header, state.at_df)
    if not (math.isfinite(ra_field) and math.isfinite(dec_field)):
        logging.warning("[PHASE 2A] Field center RA/Dec unavailable - lunar context skipped")
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
        "Lunar context: %s risk - phase %.1f%%, separation %.1f deg, altitude %.1f deg",
        lunar["lunar_risk"],
        float(lunar["lunar_phase_pct"]),
        float(lunar["lunar_separation_deg"]),
        float(lunar["lunar_altitude_deg"]),
    )
    return lunar

def _preserve_nondetection_flags_helper(
    out_flags_local: list[str], target_frames: pd.DataFrame
) -> None:
    if "flag" not in target_frames.columns:
        return
    _rf_nd = target_frames["flag"].astype(str).str.strip().str.lower().reset_index(drop=True)
    for i in range(min(len(out_flags_local), len(_rf_nd))):
        if str(_rf_nd.iloc[i]) == "nondetection":
            out_flags_local[i] = "nondetection"

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

    Additive metadata only - does not affect photometry (baseline stays byte-identical). Matches by
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

def _propagate_phase2a_skip_reason_to_active(
    active_csv: Path,
    summary_rows: list[dict[str, Any]],
) -> None:
    """Write Phase 2A drop reasons (e.g. no_comps) into active_targets skip_reason."""
    if not active_csv.is_file() or not summary_rows:
        return
    try:
        active = pd.read_csv(active_csv, low_memory=False, dtype={"catalog_id": str})
    except Exception:  # noqa: BLE001
        return
    if "catalog_id" not in active.columns or "skip_reason" not in active.columns:
        return
    by_cid: dict[str, str] = {}
    for r in summary_rows:
        cid = str(r.get("catalog_id") or "").strip()
        if not cid:
            continue
        if int(r.get("n_frames") or 0) != 0:
            continue
        reason = str(r.get("ac_skip_reason") or r.get("skip_reason") or "").strip()
        if reason:
            by_cid[cid] = reason
    if not by_cid:
        return
    changed = False
    if active["skip_reason"].dtype != object:
        active["skip_reason"] = active["skip_reason"].astype(object)
    for idx in active.index:
        cid = str(active.at[idx, "catalog_id"] or "").strip()
        if cid not in by_cid:
            continue
        sr = active.at[idx, "skip_reason"]
        if pd.notna(sr) and str(sr).strip():
            continue
        active.at[idx, "skip_reason"] = by_cid[cid]
        changed = True
    if changed:
        active.to_csv(active_csv, index=False)

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
    _propagate_phase2a_skip_reason_to_active(output_dir / "active_targets.csv", summary_rows)

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

            _p2("Comp QA (Sokolovsky locus)...")
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

            _p2("Trust flag (GREEN/YELLOW/RED)...")
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

    logging.info(f"[FAZA 2A] Hotovo: {n_lc} svetelnych kriviek -> {output_dir}")
    logging.info(
        f"[FAZA 2A] Targety bez comp hviezd: "
        f"{len(at_df) - n_lc}/{len(at_df)} "
        f"(ziadne vhodne comp podla aktualnych filtrov)"
    )
    _p2(f"Faza 2A hotovo: {n_lc} kriviek z {n_frames} snimok -> {output_dir.name}")

    # Export lightcurve reports (AAVSO + VAR.ASTRO.CZ) - best effort, non-fatal.
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
        _export_stats: dict[str, int] = {}
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
                    export_stats=_export_stats,
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

        log_export_batch_summary(_export_failures, _export_stats)
        logging.info(
            "[EXPORT] lightcurves_reports: %d targets exported, %d skipped (methods=%s)",
            int(n_export_ok),
            int(n_export_skip),
            ",".join(_active_methods),
        )
        if _export_stats.get("err_scatter_unmatched_epochs"):
            logging.info(
                "[EXPORT] run summary: err_scatter_unmatched_epochs=%d",
                int(_export_stats["err_scatter_unmatched_epochs"]),
            )
        if _export_stats.get("time_base_refused"):
            logging.error(
                "[EXPORT] run summary: time_base_refused=%d",
                int(_export_stats["time_base_refused"]),
            )
    except Exception as exc:  # noqa: BLE001
        logging.error('[EXC-0176] AAVSO/VarAstro lightcurve export batch fails - external report files missing: %s', exc)
        logging.warning("[EXPORT] init failed: %s", exc)

    # Build flux pivot once - reuse in variability detection (TODO-PERF-6)
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

    # INV-DAG-01 + INV-FLAT-01 + INV-PROV-01 / INV-CFG-01 end-of-run gates.
    try:
        from invariants_runtime import (  # noqa: PLC0415
            FLATNESS_P99_WARN_ADU,
            inv_check,
            load_pipeline_meta,
            run_end_of_run_invariants,
            save_pipeline_meta,
            stamp_pipeline_stage,
        )

        _inv_meta = load_pipeline_meta(output_dir)
        stamp_pipeline_stage(_inv_meta, "phase2a", enforce_upstream=True)
        # INV-FLAT-01 from preprocess residual flatness column in qc_metrics (WARN).
        try:
            _dd = _draft_dir_from_phase2a_paths(output_dir, Path(masterstar_fits_path))
            from pipeline import find_qc_metrics_csv  # noqa: PLC0415

            _qc_p = find_qc_metrics_csv(_dd) if _dd is not None else None
            if _qc_p is not None and _qc_p.is_file():
                _qdf = pd.read_csv(_qc_p, low_memory=False)
                if "residual_flatness_p99_adu" in _qdf.columns:
                    _p99s = pd.to_numeric(
                        _qdf["residual_flatness_p99_adu"], errors="coerce"
                    ).to_numpy(dtype=float)
                    _p99s = _p99s[np.isfinite(_p99s)]
                    if _p99s.size:
                        _p99_max = float(np.nanmax(_p99s))
                        _ok_fl = _p99_max <= float(FLATNESS_P99_WARN_ADU)
                        inv_check(
                            _inv_meta,
                            "INV-FLAT-01",
                            _ok_fl,
                            policy="WARN",
                            detail=(
                                f"max residual_flatness_p99={_p99_max:.1f} ADU "
                                f"(band={FLATNESS_P99_WARN_ADU:g}; n={int(_p99s.size)})"
                            ),
                        )
        except Exception as _flat_exc:  # noqa: BLE001
            logging.debug("[INV-FLAT-01] skipped: %s", _flat_exc)
        save_pipeline_meta(output_dir, _inv_meta)
        run_end_of_run_invariants(output_dir, stamp_postprocess=True)
    except Exception as _inv_end_exc:  # noqa: BLE001
        from invariants_runtime import InvariantViolation  # noqa: PLC0415

        if isinstance(_inv_end_exc, InvariantViolation):
            raise
        logging.warning("[INV] end-of-run validation skipped: %s", _inv_end_exc)

    # PRE-IMPL-01: persist Phase-2A sigma_eff weights into comparison_stars_per_target.csv
    try:
        from comp_weights import rewrite_comparison_stars_weights_csv  # noqa: PLC0415

        _cw = Path(comparison_stars_csv)
        if _cw.is_file():
            _stats = rewrite_comparison_stars_weights_csv(_cw)
            logging.info(
                "[PRE-IMPL-01] Rewrote comp_weight/sigma_eff_mag on %s: %s",
                _cw.name,
                _stats,
            )
    except Exception as _w_exc:  # noqa: BLE001
        logging.error("[PRE-IMPL-01] comp_weight rewrite failed: %s", _w_exc)

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
    """Hlavny wrapper pre Fazu 2A.

    Globalny FWHM pre aperturu: ``VY_FWHM_GAUSS`` (2D fit z pipeline), inak ``VY_FWHM``
    (DAO, pre aperturu porovnatelne s Gaussian FWHM), inak 2D Gaussian fit
    (``measure_fwhm_from_masterstar``) s napovedou z ``fwhm_px``.
    Aperturny polomer = ``aperture_fwhm_factor x FWHM`` (predvolene z ``cfg``).

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

    _p2("Initializing shared state...")
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
                _apt_fw = max(0.25, min(6.0, _apt_fw))
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
                location_id=state.site_location_id,
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
    # INV-CFG-01: cog meta keys present only when COG is enabled.
    if bool(getattr(_cfg, "cog_aperture_correction_enabled", False)):
        merge_photometry_pipeline_meta(
            output_dir,
            {
                "cog_night_fallback": bool(state.cog_night_fallback),
                "cog_night_fallback_n_without_ok": int(state.cog_night_fallback_n_without_ok),
                "cog_night_fallback_n_frames": int(state.cog_night_fallback_n_frames),
            },
        )
    # INV-CFG-01: per-frame sat markers present only when enabled.
    if bool(getattr(_cfg, "per_frame_saturation_enabled", False)) and state.per_frame_sat_meta:
        merge_photometry_pipeline_meta(output_dir, dict(state.per_frame_sat_meta))
    # NIGHT_FIT meta only when fit path was enabled (default OFF -> absent).
    if bool(getattr(_cfg, "k2_fit_enabled", False)) and state.k2_fit_meta:
        merge_photometry_pipeline_meta(
            output_dir,
            {
                **dict(state.k2_fit_meta),
                "k2_source": str(state.k2_source),
                "k2_value": float(state.k2_bprp) if math.isfinite(float(state.k2_bprp)) else None,
            },
        )

    # Per target loop
    # _phase2a_process_single_target (inline): ZP -> CT -> (outlier -> airmass | airmass -> outlier) -> export.
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
            "[PERF-8] Per-target frame loops eliminated: %d targets x %d frames = %d calls saved",
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


# Already-extracted siblings after bodies: importing them in the header
# re-enters photometry_core while this module is incomplete (V1 stub path).
from photometry_provenance import merge_photometry_pipeline_meta  # noqa: E402,F401
from photometry_shared import (  # noqa: E402,F401
    _normalize_gaia_id,
    _safe_polyfit,
    _target_display_name,
    build_gs11_summary,
)

# When this module is imported first (pipeline.py / ui), photometry_core skips the
# late from-import. Bind the moved names onto the facade (and shared) here.
import photometry_core as _photometry_core  # noqa: E402
import photometry_shared as _photometry_shared  # noqa: E402

_E4_PHASE2A_NAMES = (
    "parse_comp_quality_json_map",
    "_build_csv_lookup",
    "_lookup_star_in_csv",
    "_sat_limit_peak_adu",
    "_mad_sigma_or_std_floor",
    "measure_fwhm_from_masterstar",
    "compute_optimal_apertures",
    "_howell_variance_adu2",
    "_photometric_error",
    "_photometric_error_with_bkg_mode",
    "_phase2a_proc_column_requirements",
    "_phase2a_cache_columns",
    "_phase2a_empirical_sigma_bkg_ap",
    "_sky_pp_for_photometric_error",
    "_resolve_phase2a_equipment_id",
    "_draft_dir_from_phase2a_paths",
    "_require_comparison_stars_per_target_schema",
    "_median_sky_from_phase2a_csv_cache",
    "_measured_aperture_from_proc_cache",
    "_resolve_photometric_aperture_px_for_gs11",
    "read_flux_from_csv",
    "compute_aperture_correction",
    "fit_color_term_c1",
    "should_apply_color_term",
    "_obs_group_filter_key",
    "resolve_apply_color_term",
    "_ColorTermGroupFit",
    "_group_comp_mag_inst_from_flux_matrix",
    "_group_comp_mag_inst_from_proc_csvs",
    "_comp_maps_from_comparison_stars_csv",
    "_phase2a_attempt_k2_night_fit",
    "_compute_group_color_term_fit",
    "_ensure_group_comp_pool_csv",
    "_target_row_is_vsx_known_variable",
    "empirical_feature_mask_mag",
    "detect_outliers",
    "apply_reporting_postprocess",
    "democratic_detrend_lc",
    "save_field_map_png",
    "_edge_ok_from_masterstar_pipeline",
    "resolve_variable_targets_csv",
    "auto_export_variability_candidates_csv",
    "_phase2a_coerce_skip_photometry",
    "build_rms_mag_model",
    "expected_rms_from_model",
    "classify_lc_quality",
    "build_lc_quality_summary",
    "_phase2a_write_summary",
    "_phase2a_observer_location_dict",
    "_sky_surface_meta_from_qc",
    "_phase2a_resolve_field_center_ra_dec",
    "_phase2a_collect_session_jd_values",
    "_Phase2AState",
    "_build_phase2a_dynamic_params",
    "_phase2a_compute_lunar_context",
    "_preserve_nondetection_flags_helper",
    "_proc_stem",
    "_compute_frame_align_residuals",
    "_record_align_residuals_to_report",
    "_frame_align_residual_gate_select",
    "_propagate_phase2a_skip_reason_to_active",
    "_phase2a_finalize_exports",
    "run_phase2a",
)
for _n in _E4_PHASE2A_NAMES:
    setattr(_photometry_core, _n, globals()[_n])
_photometry_shared._sky_pp_for_photometric_error = _sky_pp_for_photometric_error

from phase2a_state import _phase2a_prepare_shared_state  # noqa: E402
_photometry_core._phase2a_prepare_shared_state = _phase2a_prepare_shared_state
