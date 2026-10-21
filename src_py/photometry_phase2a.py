"""Moved from photometry_core.py (CONSOLIDATE-01E4). Facade re-exports these names."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json
import logging
import math
import re
from astropy.io import fits as astrofits
import numpy as np
import pandas as pd
from catalog_match_trust import is_wcs_untrusted_catalog_match_mode, normalize_catalog_match_mode
from config import AppConfig
from gaia_catalog_id import masterstar_row_gaia_key, normalize_gaia_source_id
from infolog import log_event
from proc_frame_store import ProcFrameStore
from stats_core import _flux_to_mag
from photometry_core import (
    ERR_BKG_SOURCE_COL,
    ERR_BKG_SOURCE_EMPIRICAL,
    ERR_BKG_SOURCE_HOWELL_FALLBACK,
    LOGGER,
    SIGMA_BKG_AP_COL,
    SKY_ADU_PER_PX_ANNULUS_COL,
    SKY_SURFACE_BG_MEDIAN_ADU_COL,
    _COMP_QUALITY_JSON_META_KEYS,
    _GAIA_ID_DTYPE,
    _LC_QUALITY_FLAGS,
    _MAD_CONSISTENCY,
)


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

def _sat_limit_peak_adu() -> float | None:
    """Hranica peak_max_adu z configu (volitelne). Bez globalneho fallbacku - saturacia z FITS/DB v pipeline."""
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
    star_positions: pd.DataFrame,
    fwhm_px: float,
    *,
    aperture_fwhm_factor: float = 1.75,
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
    fallback.
    """
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
            "sky_adu_per_px_annulus",
            SKY_SURFACE_BG_MEDIAN_ADU_COL,
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
            "vsx_known_variable",
            "gaia_dr3_variable_catalog",
            "ra_deg",
            "dec_deg",
            "photometry_ok",
            "snr50_ok",
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
    src_raw = row_csv.get(ERR_BKG_SOURCE_COL, "")
    src = str(src_raw).strip().lower() if src_raw is not None else ""
    raise ValueError(
        "[INV-ERR-MODE-01] empirical background error requires "
        f"'{SIGMA_BKG_AP_COL}' in Phase 2A input; missing/NaN for catalog_id={catalog_id} "
        f"in {source_file} (err_bkg_source={src or 'missing'})."
    )

def _sky_pp_for_photometric_error(row: Any) -> float:
    """Sky level (ADU/px) for Howell ``_photometric_error`` from a proc-CSV row.

    I-11: prefer pre-subtraction ``sky_surface_bg_median_adu`` for the Howell sky Poisson
    term (photons that arrived before the 2D sky surface was subtracted). Post-subtraction
    annulus sky collapses toward zero and under-quotes the sky term on the legacy path.

    Otherwise ``sky_adu_per_px_annulus`` (the per-star annulus sky of the proc CSV).
    """
    pre_sub = float(pd.to_numeric(row.get(SKY_SURFACE_BG_MEDIAN_ADU_COL), errors="coerce"))
    if math.isfinite(pre_sub) and pre_sub >= 0:
        return pre_sub
    ann = float(pd.to_numeric(row.get(SKY_ADU_PER_PX_ANNULUS_COL), errors="coerce"))
    if math.isfinite(ann) and ann >= 0:
        return ann
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
        if df is None or df.empty or SKY_ADU_PER_PX_ANNULUS_COL not in df.columns:
            continue
        s = pd.to_numeric(df[SKY_ADU_PER_PX_ANNULUS_COL], errors="coerce").dropna()
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
) -> tuple[float | None, str]:
    """Photometric aperture for GS11 dilution: Phase 2A per-star map, else skip.

    CONSOLIDATE-01B C2: the SNR mag-bin table is gone. No fixed-pixel fallback.
    """
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
    variable_target_catalog_ids: frozenset[str] | None = None,
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
            from proc_frame_store import read_proc_csv  # noqa: PLC0415

            csv_df = read_proc_csv(frame_csv_path, low_memory=False, dtype=_GAIA_ID_DTYPE)
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
            "alignment_failed": align_failed_frame,
            "catalog_match_mode": frame_catalog_match_mode,
            "wcs_untrusted": is_wcs_untrusted_catalog_match_mode(frame_catalog_match_mode),
            "mag_inst": float("nan"),
            "err": float("nan"),
            "aperture_r_px": apertures_px.get(cid, float("nan")),
            "x": float("nan"),
            "y": float("nan"),
            "sky_annulus_r_out_px": float("nan"),
            "edge_fail": False,
            "sky_pp": float("nan"),
            "flux_raw": float("nan"),
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

        # Apertura z CSV (ta co pipeline pouzila pri DAO)
        ap_csv = float(row_csv.get("aperture_r_px", float("nan")))
        if math.isfinite(ap_csv) and ap_csv > 0:
            base["aperture_r_px"] = ap_csv

        # Sky per pixel for Howell err: the annulus column.
        sky_pp = _sky_pp_for_photometric_error(row_csv)
        if math.isfinite(sky_pp):
            base["sky_pp"] = sky_pp

        # Saturacia
        # SAT-LIN-2026-10: per-frame verdict from the proc CSV (raw peak >= LINEARITY_FRAC x SAT,
        # sat_diag); an explicit sat_limit_adu argument still overrides.
        peak = float(pd.to_numeric(row_csv.get("peak_max_adu", float("nan")), errors="coerce"))
        if math.isfinite(_sat_lim):
            is_sat = math.isfinite(peak) and peak >= _sat_lim
        else:
            _ls = row_csv.get("likely_saturated", False)
            is_sat = str(_ls).strip().lower() in ("true", "1", "1.0", "yes")
            if not is_sat:
                _lim85 = float(pd.to_numeric(row_csv.get("saturate_limit_adu_85pct", float("nan")), errors="coerce"))
                is_sat = math.isfinite(peak) and math.isfinite(_lim85) and _lim85 > 0 and peak >= _lim85

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
            source_file=source_file,
            catalog_id=str(cid),
        )
        _err, _err_src = _photometric_error_with_bkg_mode(
            flux,
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

def fit_color_term_c1(
    comp_mag_inst: dict[str, np.ndarray],
    comp_catalog_mag: dict[str, float],
    comp_bp_rp: dict[str, float],
    comp_quality: dict[str, dict],
    *,
    min_comp: int = 5,
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
    from band_classify import classify_photometric_band, color_term_auto_from_band

    filter_raw = str(obs_group or "").split("|")[0].strip()
    band = classify_photometric_band(obs_group)

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
    mode: str = "fit"  # "fit" | "band"
    #: CT-2026-10: observed band of the comparison catalogue magnitudes ("G" = Gaia G).
    band: str = "G"
    #: CT-2026-10: polynomial colour term (order by BIC) in the observed band.
    poly_fit: Any = None

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
            from proc_frame_store import read_proc_csv  # noqa: PLC0415

            df = read_proc_csv(csv_path, low_memory=False, dtype=_GAIA_ID_DTYPE)
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

def _comp_maps_from_comparison_stars_csv(
    comp_csv: Path, band: str = "G"
) -> tuple[dict[str, float], dict[str, float], dict[str, dict]]:
    """bp_rp, catalogue magnitude and quality maps of the comparison pool.

    With ``band`` != "G" the catalogue magnitude is Gaia G + BP-RP transformed into the
    observed band (:mod:`gaia_bands`); stars outside the valid colour range get NaN.
    """
    from gaia_bands import band_mag_from_gaia  # noqa: PLC0415

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
        if band != "G" and cid in comp_catalog_mag:
            comp_catalog_mag[cid] = band_mag_from_gaia(
                comp_catalog_mag[cid], comp_bp_rp.get(cid, float("nan")), band
            )[0]
        # the pool is built from usable, photometry_ok stars only (select_comparison_stars_spatial_grid)
        comp_quality[cid] = {"quality": "good"}
    return comp_bp_rp, comp_catalog_mag, comp_quality

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
    from gaia_bands import filter_band_for_obs_group  # noqa: PLC0415

    comp_csv = Path(comparison_stars_csv)
    if not comp_csv.is_file():
        return None
    # BANDAUTH-2026-10: a colour term exists only for a real standard filter. Clear /
    # luminance / tri-colour / unknown filters are reported untransformed (TRANS=NO).
    if not filter_band_for_obs_group(obs_group).transformable:
        logging.info("[COLOR TERM] %s: filter not transformable -> no colour term", obs_group)
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

    if k2_value is not None and k2_source is not None:
        _k2_val, _k2_src = float(k2_value), k2_source
    else:
        _k2_val, _k2_src = resolve_k2_bprp_value(cfg, obs_group)
    if _k2_src is K2Source.LITERATURE_DEFAULT and math.isfinite(
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
    from gaia_bands import band_for_obs_group  # noqa: PLC0415

    _band = band_for_obs_group(obs_group)
    if _band != "G":
        # CT-2026-10: comparison magnitudes in the observed band; colour term order by BIC.
        from color_transform import fit_colour_term  # noqa: PLC0415

        _cbp, _ccat, _cq = _comp_maps_from_comparison_stars_csv(comp_csv, band=_band)
        _xs, _ys = [], []
        for cid in comp_ids:
            if (_cq.get(cid) or {}).get("quality") not in ("good", "suspect"):
                continue
            inst = np.asarray(comp_mag_inst.get(cid, []), dtype=np.float64)
            inst = inst[np.isfinite(inst)]
            cat = float(_ccat.get(cid, float("nan")))
            if inst.size >= 3 and math.isfinite(cat) and math.isfinite(float(_cbp.get(cid, float("nan")))):
                _xs.append(float(_cbp[cid]))
                _ys.append(cat - float(np.median(inst)))
        _pf = fit_colour_term(np.asarray(_xs), np.asarray(_ys))
        if _pf is not None:
            _ok = _pf.order >= 1 and _pf.n >= int(getattr(cfg, "phase01_ct_min_comp", 7))
            _reason = (
                f"Filter {_band} - band transform Gaia->{_band} (DR3 5.5.1); colour term order "
                f"{_pf.order} by BIC, coeffs={[round(c, 4) for c in _pf.coeffs]}, rms={_pf.rms:.4f}, "
                f"n_comp={_pf.n}, BP-RP {_pf.c_min:.2f}..{_pf.c_max:.2f}"
                + ("" if _ok else " - no significant colour term")
            )
            return _ColorTermGroupFit(
                c1=float(_pf.coeffs[0]) if _pf.coeffs else 0.0,
                c1_stderr=float("nan"),
                n_comp=int(_pf.n),
                comp_bp_rp=_cbp,
                comp_quality=_cq,
                comp_catalog_mag=_ccat,
                apply_gate=True,
                gate_reason=_reason,
                mode="band",
                band=_band,
                poly_fit=_pf if _ok else None,
            )
    c1, c1_stderr, n_comp = fit_color_term_c1(
        comp_mag_inst,
        comp_catalog_mag,
        comp_bp_rp,
        comp_quality,
        min_comp=5,
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
    err: np.ndarray | None = None,
    bjd: np.ndarray | None = None,
) -> list[str]:
    """LC-OUTLIER-01 wrapper: isolated-spike flags without image evidence.

    Legacy args ``feature_mask`` / ``skip_sigma_clip`` / ``outlier_sigma`` are
    accepted for call-site compatibility. VSX skip is retired: flare/eclipse
    protection is the adjacent same-sign rule in ``lc_outlier.isolated_spike_mask``.
    Global night MAD clip (``outlier_hi``/``outlier_lo``) is retired.

    Returns:
        flags: normal / saturated / spike_unconfirmed / no_data
    """
    from lc_outlier import assign_lc_flags  # noqa: PLC0415

    m = np.asarray(mag_calib, dtype=float)
    n = len(m)
    e = np.asarray(err, dtype=float) if err is not None else np.full(n, float("nan"))
    t = np.asarray(bjd, dtype=float) if bjd is not None else np.arange(n, dtype=float)
    result = assign_lc_flags(
        m,
        e,
        t,
        sat_flags=np.asarray(flags_saturated, dtype=bool),
        evidence_for_index=None,
        enabled=True,
    )
    return list(result.flags)

def apply_reporting_postprocess(
    mag_calib: np.ndarray,
    mag_calib_ct: np.ndarray,
    *,
    target_name: str,
    sat_flags: np.ndarray,
    target_frames: pd.DataFrame,
    ct_ok: bool,
    cfg: AppConfig | None = None,
    err: np.ndarray | None = None,
    bjd: np.ndarray | None = None,
    source_files: list[str] | None = None,
    frame_qc_reasons: dict[str, str] | None = None,
    evidence_for_index: Any | None = None,
    err_photon: np.ndarray | None = None,
    err_sem_rel: np.ndarray | None = None,
    err_scint_rel: np.ndarray | None = None,
    err_sigma_sys_rel: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Workstream B: ship ensemble-calibrated mag; LC-OUTLIER / LC-FLAG-ERR flags; no target airmass LSQ.

    Photometry columns are never altered by flagging. Returns ``(mag_calib, mag_calib_ct, flags,
    flag_reasons)`` (LC-OUTLIER-01 / LC-FLAG-ERR-01); ``mag_calib_ct`` = ``mag_calib`` when not ct_ok.
    """
    from lc_outlier import assign_lc_flags  # noqa: PLC0415

    _cfg = cfg or AppConfig()
    mag_out = np.asarray(mag_calib, dtype=np.float64).copy()
    mag_for_report = np.asarray(mag_calib_ct, dtype=np.float64)
    n = len(mag_for_report)
    e = np.asarray(err, dtype=float) if err is not None else np.full(n, float("nan"))
    t = np.asarray(bjd, dtype=float) if bjd is not None else np.arange(n, dtype=float)
    _src = source_files
    if _src is None and "source_file" in target_frames.columns:
        _src = target_frames["source_file"].astype(str).tolist()
    _enabled = bool(getattr(_cfg, "lc_outlier_enabled", True))
    _n_sigma = float(getattr(_cfg, "lc_outlier_n_sigma", 5.0) or 5.0)
    _adj = float(getattr(_cfg, "lc_outlier_adjacent_sigma", 3.0) or 3.0)
    _he_ns = float(getattr(_cfg, "lc_high_err_nsigma", 5.0) or 5.0)
    # Keep empirical eclipse mask as soft log only; isolation rule is the hard guard.
    _feature_mask = empirical_feature_mask_mag(mag_for_report)
    if int(_feature_mask.sum()) > 0:
        logging.info(
            "[OUTLIER] Feature mask (info): %d/%d frames look like eclipse runs "
            "(adjacent-sign rule protects them; arXiv:2402.16018)",
            int(_feature_mask.sum()),
            len(_feature_mask),
        )
    result = assign_lc_flags(
        mag_for_report,
        e,
        t,
        sat_flags=np.asarray(sat_flags, dtype=bool),
        source_files=_src,
        frame_qc_reasons=frame_qc_reasons,
        evidence_for_index=evidence_for_index,
        n_sigma=_n_sigma,
        adjacent_sigma=_adj,
        high_err_n_sigma=_he_ns,
        err_photon=err_photon,
        err_sem_rel=err_sem_rel,
        err_scint_rel=err_scint_rel,
        err_sigma_sys_rel=err_sigma_sys_rel,
        enabled=_enabled,
    )
    out_flags = list(result.flags)
    out_reasons = list(result.reasons)
    _preserve_nondetection_flags_helper(out_flags, target_frames)
    for i in range(min(len(out_flags), len(out_reasons))):
        if out_flags[i] == "nondetection" and not out_reasons[i]:
            out_reasons[i] = "nondetection"
    if result.n_artifact or result.n_spike_unconfirmed or result.n_frame_qc or result.n_high_err:
        logging.info(
            "[LC-OUTLIER] %s: artifact=%d spike_unconfirmed=%d frame_qc=%d "
            "high_err=%d saturated=%d",
            target_name,
            result.n_artifact,
            result.n_spike_unconfirmed,
            result.n_frame_qc,
            result.n_high_err,
            result.n_saturated,
        )
    mag_calib_ct_out = np.asarray(mag_calib_ct, dtype=np.float64).copy()
    if not ct_ok:
        mag_calib_ct_out = mag_out.copy()
    return mag_out, mag_calib_ct_out, out_flags, out_reasons

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

    _gs11 = build_gs11_summary(
        _sum_df.to_dict("records"),
        _cfg_summary,
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
        from pipeline_catalog import find_qc_metrics_csv  # noqa: PLC0415

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
    #: PER-FRAME-SAT-GATED night meta (empty when flag OFF - INV-CFG-01).
    per_frame_sat_meta: dict[str, Any] = field(default_factory=dict)
    #: APERTURE-01 policy record (mode, f, night FWHM, r_ap/r_in/r_out).
    aperture_policy: dict[str, Any] | None = None
    #: LC-OUTLIER-01: frame_key -> reason (empty = OK) from night MAD-sigma QC.
    frame_qc_reasons: dict[str, str] = field(default_factory=dict)
    #: APERTURE-PERTARGET-01: night flux grid + per-target f* choices (None when off).
    aperture_grid_night: Any | None = None
    per_target_aperture: dict[str, Any] = field(default_factory=dict)
    #: TARGET-BLEND-2026-10: neighbour light in the target aperture (fraction of own flux)
    target_blend: dict[str, float] = field(default_factory=dict)
    #: COMP-SMART-2026-10: reserve comparison candidates per target / selection log
    reserve_index: dict[str, Any] = field(default_factory=dict)
    smart_log: dict[str, Any] = field(default_factory=dict)
    #: COMP-CATOFF-2026-10: comparison candidates excluded for their catalogue offset (any target)
    catoff_excluded: set = field(default_factory=set)
    #: OUTPUT-REVIEW-2026-10 step 7: MASTERSTAR background of the per-target field maps (drawn once)
    target_field_map: Any = None
    #: PSF-LOCAL-2026-10: own moment FWHM / field median per target
    psf_ratio: dict[str, float] = field(default_factory=dict)
    #: SIGMA-SYS-NIGHT-2026-10: measured systematic floor of the night (mag); None = config
    sigma_sys_night: float | None = None
    #: PROGRESS-MEASURED-2026-10: progress counters (measured targets only)
    n_to_measure: int = 0
    n_final_skip: int = 0
    i_measured: int = 0
    #: PSF-LOCAL-2026-10: aperture radius scale actually applied (absent = 1.0)
    psf_scale: dict[str, float] = field(default_factory=dict)
    #: GSPC-ZP-2026-10: Gaia DR3 GSPC rows of the group's comps (source_id -> row) and status.
    gspc_rows: dict[str, Any] = field(default_factory=dict)
    gspc_status: str = "not used"
    # ALIGN-VERIFY-2026-10: frames of this group whose registration was not verified (not measured)
    n_align_skipped: int = 0

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
    _meta_path = Path(output_dir) / "pipeline_meta.json"
    if _meta_path.is_file():
        try:
            _fd = json.loads(_meta_path.read_text(encoding="utf-8")).get("field_density") or {}
            _dc = _fd.get("density_class")
            if _dc is not None and str(_dc).strip():
                density_class = str(_dc).strip()
        except Exception as exc:  # noqa: BLE001
            logging.error('[EXC-0160] pipeline_meta field_density not loaded - density_class left empty: %s', exc)

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
        reason = str(r.get("skip_reason") or "").strip()
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
    masterstar_fits_path: Path,
    per_frame_csv_dir: Path,
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

    # Draft-level trust flag (read-only w.r.t. photometry; before AAVSO/VarAstro/PDF).
    if bool(getattr(_cfg, "trust_flag_enabled", True)):
        try:
            from trust_flag_core import run_trust_flag_for_photometry_dir  # noqa: PLC0415

            _p2("Trust flag (GREEN/YELLOW/RED)...")
            run_trust_flag_for_photometry_dir(
                photometry_dir=output_dir,
                update_summary=True,
                cfg=_cfg,
            )
            _sum_df = pd.read_csv(output_dir / "photometry_summary.csv", low_memory=False, dtype={"catalog_id": str})
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
        from export_reports import export_group_reports  # noqa: PLC0415

        export_group_reports(
            output_dir,
            obs_group=str(Path(per_frame_csv_dir).name),
            cfg=_cfg,
            at_df=at_df,
            summary_df=_sum_df,
            comp_index=_comp_index,
        )
    except Exception as exc:  # noqa: BLE001
        logging.error('[EXC-0176] AAVSO/VarAstro lightcurve export batch fails - external report files missing: %s', exc)

    # VAR-2026-10: search for new variable stars in the whole field (same function the UI
    # reads), then - only with tess_enabled - TESS analysis of every candidate.
    try:
        from variability_field import run_field_variability  # noqa: PLC0415

        run_field_variability(
            Path(per_frame_csv_dir),
            output_dir,
            _cfg,
            platesolve_dir=Path(masterstar_fits_path).resolve().parent,
        )
    except Exception as exc:  # noqa: BLE001
        logging.error("[VARIABILITY] field search failed: %s", exc)
    try:
        from tess_runner import auto_tess_for_candidates  # noqa: PLC0415

        auto_tess_for_candidates(output_dir, _cfg)
    except Exception as exc:  # noqa: BLE001
        logging.error("[TESS] automatic analysis failed: %s", exc)

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
            from pipeline_catalog import find_qc_metrics_csv  # noqa: PLC0415

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

    return {
        "n_targets": len(at_df),
        "n_frames": n_frames,
        "n_lightcurves": n_lc,
        "summary_csv": str(summary_csv),
        "summary_report_pdf": "",
        "field_map": str(field_map_path),
        "output_dir": str(output_dir),
        "lunar_context": lunar_context,
    }


def _night_sigma_sys(state: Any, grid: Any, frame_order: list, output_dir: Path,
                     annulus_inner_fwhm: float, annulus_outer_fwhm: float, *, choices: dict | None = None) -> None:
    """SIGMA-SYS-NIGHT-2026-10 (Milan 2026-10-12): measure the night's systematic floor on the
    union of the final comparison stars (k = 1.25) and use it instead of the config constant
    (draft 526 S4 Cyg: config 18 mmag made the error bars 2x the real scatter)."""
    from aperture_pertarget import series_from_grid, sky_series_from_grid  # noqa: PLC0415
    from sigma_floor_core import night_sigma_sys_from_comps  # noqa: PLC0415

    comps: set[str] = set()
    for _df in (getattr(state, "_comp_index", None) or {}).values():
        if _df is not None and "catalog_id" in getattr(_df, "columns", []):
            comps |= {str(c).strip() for c in _df["catalog_id"].tolist()}
    # aperture: the grid k nearest to the median chosen k* of the night's targets
    _ks = [float(c.f_star) for c in (choices or {}).values() if math.isfinite(float(getattr(c, "f_star", float("nan"))))]
    _kt = float(np.median(_ks)) if _ks else 1.25
    k = min((float(f) for f in grid.f_grid), key=lambda f: abs(f - _kt)) if grid.f_grid else 1.25
    flux = {c: series_from_grid(grid, c, k, frame_order) for c in sorted(comps)}
    sky = {c: sky_series_from_grid(grid, c, k, frame_order) for c in sorted(comps)}
    fw = [float(grid.fwhm_by_frame.get(st, grid.fwhm_night_px)) for st in frame_order]
    _g = float(state._gain_phot) if state._gain_phot is not None and math.isfinite(float(state._gain_phot)) else 1.0
    _rn = float(state._rn_phot) if state._rn_phot is not None and math.isfinite(float(state._rn_phot)) else 10.0
    r = night_sigma_sys_from_comps(
        flux, sky, fw, k=k, gain=_g, read_noise=_rn,
        annulus_inner_fwhm=float(annulus_inner_fwhm), annulus_outer_fwhm=float(annulus_outer_fwhm),
    )
    r["k"] = float(k)
    v = float(r.get("sigma_sys_mag", float("nan")))
    state.sigma_sys_night = v if math.isfinite(v) else None
    r["applied"] = "night" if state.sigma_sys_night is not None else "config"
    try:
        from photometry_provenance import merge_photometry_pipeline_meta as _mpm  # noqa: PLC0415

        _mpm(Path(output_dir), {"sigma_sys_night": r})
    except Exception as exc:  # noqa: BLE001
        logging.warning("[SIGMA-SYS] pipeline_meta not updated: %s", exc)
    logging.info("[SIGMA-SYS] night floor %s mag from %s comps (%s)",
                 f"{v:.4f}" if math.isfinite(v) else "n/a", r.get("n_comps"), r.get("reason", "ok"))


def _smart_prepass(state: Any, grid: Any, choices: dict, frame_order: list, cfg: Any) -> None:
    """COMP-SMART-2026-10: replace suspect comparison stars before the per-target ensemble.

    Uses the aperture-grid fluxes at each target's k*; rewrites ``state._comp_index``.
    Log per target in ``state.smart_log`` (comp_quality json / PDF / provenance)."""
    from aperture_pertarget import blend_k_column, series_from_grid  # noqa: PLC0415
    from comp_blend import BLEND_FLUX_FRAC_MAX  # noqa: PLC0415
    from comp_smart import smart_select  # noqa: PLC0415

    res_idx = getattr(state, "reserve_index", None) or {}
    # BJD per frame stem from the per-frame CSV cache
    bjd_by_stem: dict[str, float] = {}
    for _k, _df in (getattr(state, "_phase2a_csv_cache", None) or {}).items():
        try:
            _stem = Path(str(_k)).name
            _stem = _stem[5:-4] if _stem.startswith("proc_") and _stem.endswith(".csv") else Path(_stem).stem
            _col = "bjd_tdb_mid" if "bjd_tdb_mid" in _df.columns else ("jd_mid" if "jd_mid" in _df.columns else None)
            if _col is not None and len(_df):
                bjd_by_stem[_stem] = float(pd.to_numeric(_df[_col], errors="coerce").dropna().iloc[0])
        except Exception:  # noqa: BLE001
            continue
    bjd = np.array([bjd_by_stem.get(s, float("nan")) for s in frame_order], dtype=float)
    stab_kw = dict(
        max_comp_p2p=float(cfg.phase01_comparison_max_comp_rms),
        max_comp_slope_mmag_hr=float(cfg.comp_max_slope_mmag_hr),
        comp_slope_significance_k=float(getattr(cfg, "comp_slope_significance_k", 3.0)),
        common_mode_detrend=True,
    )
    n_min = int(getattr(cfg, "phase01_comparison_n_comp_min", 3) or 3)
    n_max = int(getattr(cfg, "phase01_comparison_n_comp_max", 8) or 8)
    # COMP-CATOFF-2026-10: catalogue magnitudes in the zero-point band (same as the target's ZP)
    catoff = float(getattr(cfg, "comp_catalog_offset_max_mag", 0.20) or 0.0)
    from gaia_bands import filter_band_for_obs_group, zp_mag_from_gaia  # noqa: PLC0415

    _fb = filter_band_for_obs_group(str(getattr(state, "obs_group", "") or ""))

    def _band_mags(*dfs: Any) -> dict[str, float]:
        out: dict[str, float] = {}
        for _d in dfs:
            if _d is None or _d.empty:
                continue
            for _, r in _d.iterrows():
                _g = float(pd.to_numeric(r.get("phot_g_mean_mag", r.get("mag")), errors="coerce"))
                if not math.isfinite(_g):
                    _g = float(pd.to_numeric(r.get("mag"), errors="coerce"))
                out[str(r.get("catalog_id", "")).strip()] = zp_mag_from_gaia(
                    _g, float(pd.to_numeric(r.get("bp_rp"), errors="coerce")), _fb)
        return out

    state.catoff_excluded = set()
    for tcid, prim_df in list((state._comp_index or {}).items()):
        ch = choices.get(str(tcid))
        if ch is None or prim_df is None or prim_df.empty:
            continue
        k = float(ch.f_star)
        resv_df = res_idx.get(str(tcid), pd.DataFrame())
        if resv_df is not None and not resv_df.empty:
            bcol = blend_k_column(k)
            if bcol in resv_df.columns:  # a replacement obeys the comp blend rule at k*
                _b = pd.to_numeric(resv_df[bcol], errors="coerce")
                resv_df = resv_df[~(_b.abs() > BLEND_FLUX_FRAC_MAX)]
        prim_ids = [str(c).strip() for c in prim_df["catalog_id"].astype(str)]
        res_ids = [str(c).strip() for c in resv_df.get("catalog_id", pd.Series(dtype=str)).astype(str)] \
            if resv_df is not None and not resv_df.empty else []
        lc: dict[str, np.ndarray] = {}
        for c in prim_ids + res_ids:
            f = np.asarray(series_from_grid(grid, c, k, frame_order), dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                lc[c] = np.where(np.isfinite(f) & (f > 0), -2.5 * np.log10(f), np.nan)
        rms = {}
        for _d in (prim_df, resv_df):
            if _d is not None and not _d.empty and "comp_rms" in _d.columns:
                for c, v in zip(_d["catalog_id"].astype(str).str.strip(), pd.to_numeric(_d["comp_rms"], errors="coerce")):
                    rms[c] = float(v)
        final, log = smart_select(prim_ids, res_ids, lc, {c: bjd for c in lc}, n_comp_min=n_min,
                                  n_comp_max=n_max, stability_kwargs=stab_kw, comp_rms_map=rms,
                                  cat_mag=_band_mags(prim_df, resv_df), catalog_offset_max_mag=catoff)
        state.smart_log[str(tcid)] = log
        state.catoff_excluded.update((log.get("catalog_offset") or {}).get("excluded") or [])
        if final == prim_ids:
            continue
        pool = pd.concat([prim_df.assign(comp_role="primary"),
                          (resv_df if resv_df is not None else pd.DataFrame()).assign(comp_role="replacement")],
                         ignore_index=True)
        pool["catalog_id"] = pool["catalog_id"].astype(str).str.strip()
        pool = pool.drop_duplicates("catalog_id").set_index("catalog_id", drop=False)
        new = pool.loc[[c for c in final if c in pool.index]].reset_index(drop=True)
        new.loc[new["catalog_id"].isin(log.get("kept_suspect", [])), "comp_role"] = "kept_suspect"
        state._comp_index[str(tcid)] = new
        logging.info("[COMP-SMART] %s: %d clean of %d candidates; dropped %s, added %s, kept suspect %s",
                     tcid, log["n_clean"], len(lc), log["replaced"], log["added"], log["kept_suspect"])

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
        dict: n_targets, n_frames, n_lightcurves, summary_csv, field_map
    """
    output_dir = Path(output_dir)
    lc_dir = output_dir / "lightcurves"
    lc_dir.mkdir(parents=True, exist_ok=True)
    # OUTPUT-REVIEW-2026-10 step 7: a target skipped now must not keep (and export) its old products
    from run_outputs import clear_phase2a_products  # noqa: PLC0415

    clear_phase2a_products(output_dir)

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
    fwhm_px = state.fwhm_px
    _ms_header = state._ms_header
    _ms_data = state._ms_data
    _flux_matrix = state._flux_matrix
    field_map_path = state.field_map_path
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

    state.lunar_context = _phase2a_compute_lunar_context(state)
    # LC-OUTLIER-01: night-level frame QC (MAD-sigma vs draft_manifest inspection).
    try:
        from lc_outlier import (  # noqa: PLC0415
            frame_qc_mask_from_night_table,
            load_frame_metrics_from_manifest,
        )

        _dd_qc = _draft_dir_from_phase2a_paths(output_dir, Path(masterstar_fits_path))
        _man = _dd_qc / "draft_manifest.json"
        _fq_ns = float(getattr(_cfg, "lc_outlier_frame_qc_n_sigma", 5.0) or 5.0)
        _fm = load_frame_metrics_from_manifest(_man)
        state.frame_qc_reasons = frame_qc_mask_from_night_table(_fm, n_sigma=_fq_ns)
        _n_bad = sum(1 for v in state.frame_qc_reasons.values() if v)
        if _n_bad:
            logging.info(
                "[LC-OUTLIER] frame_qc: %d/%d frames flagged (n_sigma=%.1f)",
                _n_bad,
                len(state.frame_qc_reasons),
                _fq_ns,
            )
    except Exception as _fq_exc:  # noqa: BLE001
        logging.warning("[LC-OUTLIER] frame_qc setup failed (non-fatal): %s", _fq_exc)
        state.frame_qc_reasons = {}
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
            # SIGMA-SYS-NIGHT-2026-10: reset; overwritten when the night floor is measured
            "sigma_sys_night": {"applied": "config"},
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
    # INV-CFG-01: per-frame sat markers present only when enabled.
    if state.per_frame_sat_meta:
        merge_photometry_pipeline_meta(output_dir, dict(state.per_frame_sat_meta))
    # APERTURE-PERTARGET-01: measure f-grid for targets+comps and pick f* per target.
    state.aperture_grid_night = None
    state.per_target_aperture = {}
    try:
        from aperture_pertarget import (  # noqa: PLC0415
            choose_f_for_target,
            measure_night_grid,
            DEFAULT_APERTURE_F_GRID,
            write_per_target_choices,
        )

        _p2("APERTURE: measuring k grid (1.0-2.0 x FWHM) for targets, comps and check stars...")
        _ids: set[str] = set()
        for _, _tr in at_df.iterrows():
            _cid = str(_tr.get("catalog_id", "") or "").strip()
            if _cid:
                _ids.add(_cid)
        for _cid_key, _sub in (state._comp_index or {}).items():
            _ids.add(str(_cid_key))
            if hasattr(_sub, "iterrows"):
                for _, _cr in _sub.iterrows():
                    _cc = str(_cr.get("catalog_id", "") or "").strip()
                    if _cc:
                        _ids.add(_cc)
        # COMP-SMART-2026-10: the reserve candidates are measured too (replacements)
        for _rsub in (getattr(state, "reserve_index", None) or {}).values():
            if hasattr(_rsub, "iterrows"):
                for _cc in _rsub.get("catalog_id", pd.Series(dtype=str)).astype(str).str.strip():
                    if _cc:
                        _ids.add(_cc)
        # APERTURE-K-2026-10: the whole comparison pool (check-star candidates
        # included) is measured so the check star uses the same aperture as its comps.
        if getattr(state, "comp_df", None) is not None and "catalog_id" in state.comp_df.columns:
            for _cc in state.comp_df["catalog_id"].astype(str).str.strip():
                if _cc:
                    _ids.add(_cc)
        _xy_ms: dict[str, tuple[float, float]] = {}
        _msdf = getattr(state, "masterstars_df", None)
        if _msdf is not None and {"catalog_id", "x", "y"}.issubset(_msdf.columns):
            for _c, _x, _y in zip(
                _msdf["catalog_id"].astype(str).str.strip(),
                pd.to_numeric(_msdf["x"], errors="coerce"),
                pd.to_numeric(_msdf["y"], errors="coerce"),
            ):
                if _c in _ids:
                    _xy_ms[_c] = (float(_x), float(_y))
        _fwhm_pt = float(
            (state.aperture_policy or {}).get("fwhm_night_median_px")
            or (state.aperture_policy or {}).get("fwhm_used_px")
            or state.fwhm_px
            or 5.0
        )
        # PSF-LOCAL-2026-10 (Milan 2026-10-11, draft 526 ASASSN-V J152756.18: own PSF 1.35 x
        # the field, aperture 3.4 px): targets with a broader own profile get r scaled.
        _psf_scale: dict[str, float] = {}
        state.psf_ratio = {}
        try:
            from aperture_pertarget import psf_scale_for_targets  # noqa: PLC0415

            _tgt_ids = {str(c).strip() for c in at_df.get("catalog_id", pd.Series(dtype=str)).tolist()}
            _txy = {c: v for c, v in _xy_ms.items() if c in _tgt_ids}
            _ref_xym: list[tuple[float, float, float]] = []
            _tmag_by: dict[str, float] = {}
            _excl: set[str] = set()
            if _msdf is not None and {"x", "y", "mag"}.issubset(_msdf.columns):
                _mx = pd.to_numeric(_msdf["x"], errors="coerce").to_numpy(float)
                _my = pd.to_numeric(_msdf["y"], errors="coerce").to_numpy(float)
                _mm = pd.to_numeric(_msdf["mag"], errors="coerce").to_numpy(float)
                _W, _H = np.nanmax(_mx), np.nanmax(_my)
                _okm = np.isfinite(_mx) & np.isfinite(_my) & np.isfinite(_mm)
                # references: interior, unsaturated, not a known variable, isolated (no
                # catalogue neighbour within 4 FWHM brighter than mag + 2.5)
                _sel = _okm & (_mx > 100) & (_my > 100) & (_mx < _W - 100) & (_my < _H - 100)
                for _flag in ("vsx_known_variable", "likely_saturated", "is_saturated"):
                    if _flag in _msdf.columns:
                        _sel &= ~_msdf[_flag].astype(str).str.strip().str.lower().isin(["true", "1", "1.0"]).to_numpy()
                if "peak_max_adu" in _msdf.columns:
                    _pk = pd.to_numeric(_msdf["peak_max_adu"], errors="coerce").to_numpy(float)
                    _sel &= ~(_pk >= 50000.0)
                from scipy.spatial import cKDTree  # noqa: PLC0415

                _ii = np.where(_okm)[0]
                _tree = cKDTree(np.c_[_mx[_ii], _my[_ii]])
                for _k in np.where(_sel)[0]:
                    for _j in _tree.query_ball_point([_mx[_k], _my[_k]], 4.0 * _fwhm_pt):
                        _jj = _ii[_j]
                        if _jj != _k and _mm[_jj] < _mm[_k] + 2.5:
                            _sel[_k] = False
                            break
                _ref_xym = [(float(_mx[i]), float(_my[i]), float(_mm[i])) for i in np.where(_sel)[0]]
                # a catalogue neighbour within 3 FWHM not fainter than target + 1 mag: the
                # profile includes the neighbour -> no scaling (TARGET-BLEND judges it)
                for _c, (_x, _y) in _txy.items():
                    _tm = _msdf.loc[_msdf["catalog_id"].astype(str).str.strip() == _c, "mag"]
                    _tmag = float(pd.to_numeric(_tm, errors="coerce").iloc[0]) if len(_tm) else float("nan")
                    _tmag_by[_c] = _tmag
                    # a flat-topped (saturated / nonlinear) core looks broad -> never scaled
                    _trow = _msdf.loc[_msdf["catalog_id"].astype(str).str.strip() == _c]
                    if len(_trow):
                        _tr0 = _trow.iloc[0]
                        if any(str(_tr0.get(_f, "")).strip().lower() in ("true", "1", "1.0")
                               for _f in ("is_saturated", "likely_saturated", "likely_nonlinear")) \
                                or float(pd.to_numeric(_tr0.get("peak_max_adu"), errors="coerce")) >= 50000.0:
                            _excl.add(_c)
                    _d = np.hypot(_mx - _x, _my - _y)
                    _nb = (_d > 0.5) & (_d < 3.0 * _fwhm_pt) & (_mm < (_tmag + 1.0 if math.isfinite(_tmag) else 99))
                    if bool(_nb.any()):
                        _excl.add(_c)
            _psf_scale, state.psf_ratio = psf_scale_for_targets(
                Path(masterstar_fits_path), _txy, _ref_xym, _fwhm_pt, target_mag=_tmag_by, exclude=_excl)
            state.psf_scale = dict(_psf_scale)
            if _psf_scale:
                logging.info("[PSF-LOCAL] broader own PSF -> scaled aperture: %s",
                             {k: round(v, 2) for k, v in _psf_scale.items()})
        except Exception as _ps_exc:  # noqa: BLE001
            logging.warning("[PSF-LOCAL] not computed: %s", _ps_exc)
            _psf_scale = {}
        _grid = measure_night_grid(
            frames_dir=Path(state._aligned_dir_2a),
            catalog_ids=sorted(_ids),
            f_grid=list(DEFAULT_APERTURE_F_GRID),
            fwhm_night_px=_fwhm_pt,
            annulus_inner_fwhm=float(annulus_inner_fwhm),
            annulus_outer_fwhm=float(annulus_outer_fwhm),
            xy_by_cid=_xy_ms,
            scale_by_cid=_psf_scale,
        )
        state.aperture_grid_night = _grid
        _frame_order = sorted(_grid.frames.keys())
        _choices = {}
        for _, _tr in at_df.iterrows():
            _tcid = str(_tr.get("catalog_id", "") or "").strip()
            if not _tcid:
                continue
            _comps_sub = state._comp_index.get(_tcid, pd.DataFrame())
            if _comps_sub is None or getattr(_comps_sub, "empty", True):
                _comp_ids_pt: list[str] = []
            else:
                _comp_ids_pt = [
                    str(x).strip()
                    for x in _comps_sub.get("catalog_id", pd.Series(dtype=str)).tolist()
                    if str(x).strip()
                ]
            # BLEND-K-2026-10: only apertures at which all the target's comps keep
            # <= 1 % neighbour light (per-k values from the Phase-1 pool).
            from aperture_pertarget import allowed_f_from_comp_blend  # noqa: PLC0415
            from comp_blend import BLEND_FLUX_FRAC_MAX  # noqa: PLC0415

            _allowed_f = allowed_f_from_comp_blend(_comps_sub, _grid.f_grid, BLEND_FLUX_FRAC_MAX)
            _none_allowed = _allowed_f is not None and not _allowed_f
            if _none_allowed:
                logging.warning("[APERTURE-DYNAMIC] %s: no aperture keeps every comp <= %.0f%% "
                                "neighbour light - whole grid used", _tcid, 100 * BLEND_FLUX_FRAC_MAX)
                _allowed_f = None
            _ch = choose_f_for_target(
                _grid,
                target_cid=_tcid,
                comp_ids=_comp_ids_pt,
                frame_order=_frame_order,
                gain=float(state._gain_phot) if state._gain_phot is not None else 1.0,
                read_noise=float(state._rn_phot) if state._rn_phot is not None else 10.0,
                allowed_f=_allowed_f,
            )
            if _none_allowed:
                _ch.reason += "; no k keeps all comps <= 1 % neighbour light - whole grid used"
            _choices[_tcid] = _ch
            state.per_target_aperture[_tcid] = _ch
        write_per_target_choices(
            Path(output_dir) / "aperture_per_target.json",
            _choices,
            f_grid=_grid.f_grid,
            fwhm_night_px=_fwhm_pt,
            elapsed_s=_grid.elapsed_s,
            gain=float(state._gain_phot) if state._gain_phot is not None else None,
            read_noise=float(state._rn_phot) if state._rn_phot is not None else None,
        )
        # TARGET-BLEND-2026-10 (Milan 2026-10-11): neighbour light in the TARGET aperture at
        # its chosen k (same Moffat model as the comp gate, comp_blend), e.g. draft 526
        # CSS J153101.6 with G 10.2 at 5 px. Used by trust (soft > 1 %, hard > 5 %).
        try:
            from comp_blend import blend_contamination_for_pool  # noqa: PLC0415
            from gaia_bands import filter_band_for_obs_group  # noqa: PLC0415

            _msp = Path(masterstar_fits_path)
            _fb_tb = filter_band_for_obs_group(_msp.parent.name)
            _tpool = at_df.copy()
            _tpool["catalog_id"] = _tpool["catalog_id"].astype(str).str.strip()
            _psf_tb = None
            # one evaluation per (k, PSF-LOCAL scale): the aperture actually measured
            _sc_tb = {str(c): round(float(v), 4) for c, v in (_psf_scale or {}).items()}
            _tb_by_ks: dict[tuple[float, float], pd.Series] = {}
            _pairs = sorted({(float(_c.f_star), _sc_tb.get(str(_cid_c), 1.0)) for _cid_c, _c in _choices.items()})
            for _k, _s in _pairs:
                _cont, _bm = blend_contamination_for_pool(
                    _tpool, field_catalog_csv=_msp.parent / "field_catalog_cone.csv", masterstar_fits=_msp,
                    zp_band=_fb_tb.blend_band, fwhm_px=_fwhm_pt, r_ap_px=_k * _fwhm_pt * _s, sat_adu=None,
                    annulus_px=(float(annulus_inner_fwhm) * _fwhm_pt * _s, float(annulus_outer_fwhm) * _fwhm_pt * _s),
                    profile_df=getattr(state, "masterstars_df", None), psf=_psf_tb,
                )
                if _bm.get("status") != "ok":
                    logging.info("[TARGET-BLEND] skipped: %s", _bm.get("status"))
                    break
                _psf_tb = _bm.get("psf")
                _tb_by_ks[(_k, _s)] = _cont
            state.target_blend = {}
            for _i, _tr in _tpool.iterrows():
                _ch_t = _choices.get(_tr["catalog_id"])
                _key = (float(_ch_t.f_star), _sc_tb.get(str(_tr["catalog_id"]), 1.0)) if _ch_t is not None else None
                if _key is not None and _key in _tb_by_ks:
                    state.target_blend[_tr["catalog_id"]] = float(_tb_by_ks[_key].loc[_i])
        except Exception as _tb_exc:  # noqa: BLE001
            logging.warning("[TARGET-BLEND] not computed: %s", _tb_exc)
            state.target_blend = {}
        # COMP-SMART-2026-10 (Milan 2026-10-11): suspect comps are replaced by clean reserve
        # candidates (rank order RMS -> colour -> distance); 3..8 clean, suspects only when
        # fewer than 3 are clean. Same stability test as the Phase-2A verdict.
        try:
            _smart_prepass(state, _grid, _choices, _frame_order, _cfg)
        except Exception as _sm_exc:  # noqa: BLE001
            logging.warning("[COMP-SMART] pre-pass failed - Phase-1 comps used: %s", _sm_exc)
        # SIGMA-SYS-NIGHT-2026-10: systematic floor of this night from the final comps
        try:
            _night_sigma_sys(state, _grid, _frame_order, output_dir, annulus_inner_fwhm, annulus_outer_fwhm,
                             choices=_choices)
        except Exception as _ss_exc:  # noqa: BLE001
            logging.warning("[SIGMA-SYS] night floor not measured - config value used: %s", _ss_exc)
        _n_edge = sum(1 for _c in _choices.values() if bool(getattr(_c, "f_edge", False)))
        logging.info(
            "[APERTURE-DYNAMIC] chose f* for %d targets (grid %.1fs, n_f_edge=%d, "
            "fwhm_fallback_frames=%d)",
            len(_choices),
            float(_grid.elapsed_s),
            int(_n_edge),
            int(getattr(_grid, "n_fwhm_fallback_night", 0)),
        )
    except Exception as _pt_exc:  # noqa: BLE001
        logging.error("[APERTURE-DYNAMIC] setup failed - falling back to CSV fluxes: %s", _pt_exc)
        state.aperture_grid_night = None
        state.per_target_aperture = {}

    # PROGRESS-MEASURED-2026-10 (Milan 2026-10-12): the status line counts only the targets
    # that are measured; final skips (S/N, VSX type, depth/noise) are counted separately.
    from photometry_core import FINAL_SKIP_REASONS  # noqa: PLC0415

    _sr_all = at_df.get("skip_reason", pd.Series("", index=at_df.index)).fillna("").astype(str).str.strip()
    state.n_final_skip = int(_sr_all.isin(list(FINAL_SKIP_REASONS) + ["per_frame_saturation"]).sum())
    state.n_to_measure = int(len(at_df)) - state.n_final_skip
    state.i_measured = 0
    # Per target loop
    # _phase2a_process_single_target (inline): ZP -> CT -> (outlier -> airmass | airmass -> outlier) -> export.
    from field_map import TargetFieldMap  # noqa: PLC0415

    try:
        _fm_data = _ms_data if _ms_data is not None else astrofits.getdata(Path(masterstar_fits_path))
        _fm_hdr = _ms_header if _ms_header is not None else astrofits.getheader(Path(masterstar_fits_path))
        state.target_field_map = TargetFieldMap(_fm_data, _fm_hdr)
    except (OSError, ValueError, TypeError, ImportError) as _fm_exc:
        LOGGER.warning("[PHASE 2A] target field maps not drawn: %s", _fm_exc)
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
        )
    if state.target_field_map is not None:
        state.target_field_map.close()
        state.target_field_map = None

    if not _flux_matrix.empty:
        logging.info(
            "[PERF-8] Per-target frame loops eliminated: %d targets x %d frames = %d calls saved",
            _nt,
            len(csv_files),
            _nt * len(csv_files),
        )

    # OUTPUT-REVIEW-2026-10 step 5: mark the comparison stars of the final ensembles in the one
    # comparison table (used_role / used_rank, every target); the Phase-1 rows stay as they are.
    try:
        from comp_table import USED_PRIMARY, load_comp_table, with_used_sets, write_comp_table  # noqa: PLC0415

        _used: dict[str, list[tuple[str, str]]] = {}
        for _tid, _d in (state._comp_index or {}).items():
            if _d is None or _d.empty:
                continue
            _roles = (_d["comp_role"].fillna("").astype(str).str.strip() if "comp_role" in _d.columns
                      else pd.Series("", index=_d.index)).replace("", USED_PRIMARY)
            _used[str(_tid)] = list(zip(_d["catalog_id"].astype(str).str.strip(), _roles))
        _ct = with_used_sets(load_comp_table(Path(comparison_stars_csv)), _used)
        write_comp_table(_ct, Path(comparison_stars_csv))
        logging.info("[COMP-SMART] comparison table: %d used of %d rows", int((_ct["used_role"] != "").sum()), len(_ct))
    except Exception as _cw_exc:  # noqa: BLE001
        logging.warning("[COMP-SMART] used sets not written into the comparison table: %s", _cw_exc)
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
        masterstar_fits_path=masterstar_fits_path,
        per_frame_csv_dir=per_frame_csv_dir,
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

_E4_PHASE2A_NAMES = (
    "parse_comp_quality_json_map",
    "_mad_sigma_or_std_floor",
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
    "_resolve_photometric_aperture_px_for_gs11",
    "read_flux_from_csv",
    "fit_color_term_c1",
    "should_apply_color_term",
    "resolve_apply_color_term",
    "_group_comp_mag_inst_from_proc_csvs",
    "build_rms_mag_model",
    "expected_rms_from_model",
    "classify_lc_quality",
    "build_lc_quality_summary",
    "_Phase2AState",
    "_propagate_phase2a_skip_reason_to_active",
    "run_phase2a",
)
for _n in _E4_PHASE2A_NAMES:
    setattr(_photometry_core, _n, globals()[_n])

from phase2a_state import _phase2a_prepare_shared_state  # noqa: E402
_photometry_core._phase2a_prepare_shared_state = _phase2a_prepare_shared_state

from phase2a_target import _phase2a_process_one_target  # noqa: E402
_photometry_core._phase2a_process_one_target = _phase2a_process_one_target
