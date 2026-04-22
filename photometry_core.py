"""Photometry core — zlúčený modul (photometry + photometry_phase2a)."""
from __future__ import annotations

import json
import logging
import math
import re
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits as astrofits

from config import AppConfig
from gaia_catalog_id import normalize_gaia_source_id
from jd_axis_format import jd_axis_title, jd_series_relative

_MAD_CONSISTENCY = 0.6745  # normalizačný faktor MAD → σ ekvivalent


# ---------------------------------------------------------------------------
# Pomocné funkcie
# ---------------------------------------------------------------------------


def _normalize_gaia_id(x: Any) -> str:
    """Normalizuj Gaia ID na string integer.

    Pandas často načíta veľké Gaia ID ako float64 → ``str`` môže byť vedecká notácia
    a ``int(float(...))`` stráca presnosť. Pre textové vstupy preferujeme ``Decimal``.
    """
    if x is None:
        return ""
    if isinstance(x, (bool, np.bool_)):
        return ""
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, float):
        if not math.isfinite(x):
            return ""
        # Celé čísla v bezpečnom rozsahu float64
        if x.is_integer() and abs(x) < 2**53:
            return str(int(x))
        try:
            return str(int(Decimal(str(x))))
        except (InvalidOperation, ValueError, OverflowError):
            return str(x).strip()
    s = str(x).strip()
    if not s or s.lower() in ("nan", "none", ""):
        return ""
    try:
        return str(int(Decimal(s)))
    except (InvalidOperation, ValueError, TypeError, OverflowError):
        try:
            return str(int(float(s)))
        except (ValueError, TypeError, OverflowError):
            return s


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
        if "name" in csv_df.columns:
            raw_name = row.get("name")
            if raw_name is not None and str(raw_name).strip():
                nk = normalize_gaia_source_id(raw_name)
                if nk and re.fullmatch(r"\d{12,22}", nk):
                    id_map.setdefault(nk, row)
        cid = _normalize_gaia_id(_id_series.iloc[i])
        if cid:
            id_map.setdefault(cid, row)
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
    if cid in id_map:
        return id_map[cid]

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


def _photometric_error(flux: float, sky_pp: float, area: float, gain: float = 1.0) -> float:
    """Kvadratický súčet fotónový šum + sky šum.

    err = sqrt(flux/gain + sky_pp * area) / flux
    """
    if not math.isfinite(flux) or flux <= 0:
        return float("nan")
    if not math.isfinite(sky_pp) or sky_pp < 0:
        sky_pp = 0.0
    variance = flux / gain + max(0.0, sky_pp) * area
    return math.sqrt(variance) / flux if flux > 0 else float("nan")


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
        try:
            csv_df = pd.read_csv(frame_csv_path, low_memory=False)
        except Exception as exc:
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
    if frame_times:
        try:
            _am = float(frame_times.get("airmass", float("nan")))
            if math.isfinite(_am):
                am_frame = _am
        except (TypeError, ValueError):
            pass

    rows: list[dict] = []

    for cid in star_ids:
        base: dict[str, Any] = {
            "catalog_id": cid,
            "source_file": source_file,
            "bjd": float("nan"),
            "hjd": float("nan"),
            "jd": float("nan"),
            "airmass": am_frame,
            "mag_inst": float("nan"),
            "err": float("nan"),
            "aperture_r_px": apertures_px.get(cid, float("nan")),
            "sky_pp": float("nan"),
            "flux_raw": float("nan"),
            "flag": "no_data",
        }

        ref_x, ref_y = None, None
        if star_xy and cid in star_xy:
            rx, ry = star_xy[cid]
            ref_x = float(rx) if math.isfinite(float(rx)) else None
            ref_y = float(ry) if math.isfinite(float(ry)) else None

        row_csv = _lookup_star_in_csv(
            cid, id_map, xy_df_lookup, ref_x, ref_y, xy_tol_px=xy_tol_px
        )
        if row_csv is None:
            rows.append(base)
            continue

        # XY fallback (nie priamy ID hit): odmietni príliš jasnú hviezdu (zlá NN zhoda).
        if cid not in id_map:
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

        # Apertura z CSV (tá čo pipeline použila pri DAO)
        ap_csv = float(row_csv.get("aperture_r_px", float("nan")))
        if math.isfinite(ap_csv) and ap_csv > 0:
            base["aperture_r_px"] = ap_csv

        # Sky per pixel z noise_floor_adu (ak je k dispozícii)
        sky_pp = float(row_csv.get("noise_floor_adu", float("nan")))
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

        # Chyba — fotónový šum + sky šum
        r_ap = base["aperture_r_px"]
        area = math.pi * r_ap * r_ap if math.isfinite(r_ap) and r_ap > 0 else float("nan")
        base["err"] = _photometric_error(
            flux, sky_pp if math.isfinite(sky_pp) else 0.0, area
        )
        base["flag"] = "saturated" if is_sat else "normal"

        rows.append(base)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# KROK 3: Stability check porovnávačiek (Abbeho p2p scatter + MAD)
# ---------------------------------------------------------------------------


def check_comparison_stability(
    comp_lc: dict[str, np.ndarray],
    *,
    n_comp_min: int = 3,
    outlier_sigma: float = 3.0,
) -> dict[str, dict]:
    """Krok 3: Stability check porovnávačiek.

    Abbeho point-to-point scatter:
        rms_p2p = std(diff(mag_inst)) / sqrt(2)

    MAD filter na rms_p2p hodnoty.

    Returns:
        dict {catalog_id: {"rms_p2p": float, "lc_rms": float, "quality": str, "p2p_threshold": float}}
        quality: "good" / "suspect" / "excluded"; záznamy sú zoradené (good → suspect → excluded, v rámci good podľa rms_p2p).
    """
    result: dict[str, dict[str, Any]] = {}

    # Vypočítaj metriky
    for cid, lc in comp_lc.items():
        finite = lc[np.isfinite(lc)]
        if len(finite) < 3:
            result[cid] = {"rms_p2p": float("nan"), "lc_rms": float("nan"), "quality": "excluded"}
            continue
        lc_rms = float(np.std(finite))
        diff = np.diff(finite)
        rms_p2p = float(np.std(diff) / math.sqrt(2)) if len(diff) > 1 else float("nan")
        result[cid] = {"rms_p2p": rms_p2p, "lc_rms": lc_rms, "quality": "good"}

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
                else:
                    result[cid]["quality"] = "excluded"

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


# ---------------------------------------------------------------------------
# KROK 4: Ensemble normalizácia
# ---------------------------------------------------------------------------


def ensemble_normalize(
    target_mag_inst: np.ndarray,
    comp_mag_inst: dict[str, np.ndarray],
    comp_catalog_mag: dict[str, float],
    comp_quality: dict[str, dict],
    *,
    comp_rms_map: dict[str, float] | None = None,
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

    p2p_thr = float("nan")
    for q in comp_quality.values():
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
        ),
    )

    selected: list[str] = []
    for cid in usable_sorted:
        if len(selected) >= n_comp_max:
            break
        p2p = float(comp_quality[cid].get("rms_p2p", float("nan")))
        if len(selected) < n_comp_min:
            selected.append(cid)
        elif math.isfinite(p2p_thr) and math.isfinite(p2p) and p2p < p2p_thr:
            selected.append(cid)
        elif not math.isfinite(p2p_thr):
            selected.append(cid)

    good_ids = selected[:n_comp_max]
    if not good_ids:
        return mag_calib, delta_mag, ensemble_scatter

    cat_mags = np.asarray([comp_catalog_mag.get(cid, float("nan")) for cid in good_ids])
    cat_offset = float(np.nanmedian(cat_mags))
    logging.debug(
        f"[FÁZA 2A] Ensemble: {len(good_ids)} comps (good+suspect), "
        f"catalog_mag median={cat_offset:.3f} (mag_calib zeropoint = median(cat−inst) per frame)"
    )

    for i in range(n_frames):
        comp_pairs: list[tuple[str, float]] = []
        for cid in good_ids:
            if cid not in comp_mag_inst:
                continue
            try:
                mv = float(comp_mag_inst[cid][i])
            except Exception:  # noqa: BLE001
                continue
            if math.isfinite(mv):
                comp_pairs.append((cid, mv))

        if (not comp_pairs) or not math.isfinite(target_mag_inst[i]):
            continue

        comp_vals = np.asarray([m for _, m in comp_pairs], dtype=np.float64)

        # Priamy súčet fluxov — rovnaká metóda ako AIJ (tot_C_cnts = C2+C3+C4).
        # Váhovaný priemer 1/rms² deformuje extinkčný slope ensemble → záporný slope.
        comp_fluxes_list: list[float] = [10 ** (-0.4 * m) for _, m in comp_pairs]
        f_arr = np.asarray(comp_fluxes_list, dtype=np.float64)

        ens_flux_sum = float(np.sum(f_arr))
        if math.isfinite(ens_flux_sum) and ens_flux_sum > 0:
            ens_med = float(-2.5 * math.log10(ens_flux_sum))
        else:
            ens_med = float(np.median(comp_vals))

        ensemble_scatter[i] = float(np.std(comp_vals)) if comp_vals.size > 1 else 0.0
        delta_mag[i] = target_mag_inst[i] - ens_med

        # Zeropoint: median(cat − inst) na comps — zladí absolútnu mag s katalógom.
        # ``delta_mag + median(cat)`` by bolo nesúladné s ``ens_med`` zo súčtu fluxov (−2.5 log ΣF).
        zp_offs: list[float] = []
        for cid_j, m_j in comp_pairs:
            cm_j = float(comp_catalog_mag.get(cid_j, float("nan")))
            if math.isfinite(cm_j) and math.isfinite(m_j):
                zp_offs.append(cm_j - m_j)
        if zp_offs:
            mag_calib[i] = target_mag_inst[i] + float(np.nanmedian(np.asarray(zp_offs, dtype=np.float64)))
        else:
            mag_calib[i] = delta_mag[i] + cat_offset

    return mag_calib, delta_mag, ensemble_scatter


# ---------------------------------------------------------------------------
# KROK 5: Outlier detekcia
# ---------------------------------------------------------------------------


def detect_outliers(
    mag_calib: np.ndarray,
    flags_saturated: np.ndarray,
    *,
    outlier_sigma: float = 3.0,
) -> list[str]:
    """Krok 5: Outlier detekcia v svetelnej krivke.

    Returns:
        list flagov: "normal" / "saturated" / "outlier_hi" / "outlier_lo" / "no_data"
    """
    n = len(mag_calib)
    flags = ["no_data"] * n
    finite_mask = np.isfinite(mag_calib)

    if finite_mask.sum() < 3:
        return flags

    finite_vals = mag_calib[finite_mask]
    med = float(np.median(finite_vals))
    sigma = _mad_sigma(finite_vals)
    thr = outlier_sigma * sigma

    for i in range(n):
        if not math.isfinite(mag_calib[i]):
            flags[i] = "no_data"
        elif bool(flags_saturated[i]):
            flags[i] = "saturated"
        elif mag_calib[i] < med - thr:
            # Spike nahor (flux nahor = mag nižšie = lietadlo/kozmický lúč)
            flags[i] = "outlier_hi"
        elif mag_calib[i] > med + thr:
            # Pokles flux (zakrytie, potenciálne zaujímavé)
            flags[i] = "outlier_lo"
        else:
            flags[i] = "normal"

    return flags


def airmass_detrend_lc(
    mag_calib: np.ndarray,
    airmass: np.ndarray,
    flags: list[str],
    *,
    min_points: int = 10,
) -> tuple[np.ndarray, float, float]:
    """Lineárny airmass detrending svetelnej krivky.

    Fituje: mag_calib = a * airmass + b (na normal bodoch).
    Vracia: detrended mag_calib, slope a, intercept b.

    Detrending odstraňuje atmosferický trend ale zachováva
    astrofyzikálnu variabilitu (transit, zákryt).
    """
    mask = np.array(
        [
            f == "normal" and math.isfinite(float(m)) and math.isfinite(float(am))
            for f, m, am in zip(flags, mag_calib, airmass)
        ],
        dtype=bool,
    )

    if int(mask.sum()) < min_points:
        return mag_calib.copy(), float("nan"), float("nan")

    am_fit = airmass[mask]
    mag_fit = mag_calib[mask]

    coeffs = np.polyfit(am_fit, mag_fit, 1)
    slope, intercept = float(coeffs[0]), float(coeffs[1])

    # Guard: fyzikálne nereálny slope → preskočiť detrending
    # Hustejšie polia s farebným diferenciálom môžu mať vyšší slope.
    _MAX_REALISTIC_SLOPE = 5.0
    if abs(slope) > _MAX_REALISTIC_SLOPE:
        logging.warning(
            "[FÁZA 2A] Airmass slope=%.3f mag/airmass prekračuje fyzikálny limit "
            "(%.1f). Detrending preskočený — krivka zostáva neupravená.",
            slope,
            _MAX_REALISTIC_SLOPE,
        )
        return mag_calib.copy(), float("nan"), float("nan")

    am_ref = float(np.median(am_fit))

    detrended = mag_calib.copy()
    finite_mask = np.isfinite(mag_calib) & np.isfinite(airmass)
    detrended[finite_mask] = mag_calib[finite_mask] - slope * (airmass[finite_mask] - am_ref)

    logging.debug(
        f"[FÁZA 2A] Airmass detrend: slope={slope:.4f} mag/airmass, "
        f"am_ref={am_ref:.3f}, n_points={int(mask.sum())}"
    )

    return detrended, slope, intercept


# ---------------------------------------------------------------------------
# KROK 6: Výstup — lightcurve CSV
# ---------------------------------------------------------------------------


def save_lightcurve_csv(
    output_path: Path,
    bjd: np.ndarray,
    hjd: np.ndarray,
    jd: np.ndarray,
    airmass: np.ndarray,
    mag_inst: np.ndarray,
    mag_calib_raw: np.ndarray,
    mag_calib: np.ndarray,
    delta_mag: np.ndarray,
    err: np.ndarray,
    aperture_r_px: np.ndarray,
    flags: list[str],
    source_files: list[str],
    *,
    method: str = "aperture",
) -> None:
    """Uloží lightcurve CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "bjd": bjd,
            "hjd": hjd,
            "jd": jd,
            "airmass": airmass,
            "mag_inst": np.round(mag_inst, 6),
            "mag_calib_raw": np.round(mag_calib_raw, 6),
            "mag_calib": np.round(mag_calib, 6),
            "delta_mag": np.round(delta_mag, 6),
            "err": np.round(err, 6),
            "aperture_r_px": np.round(aperture_r_px, 3),
            "flag": flags,
            "method": method,
            "source_file": source_files,
        }
    )
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
    for i, (cid, info) in enumerate(comp_quality.items(), 1):
        q = str(info["quality"])
        p2p = info.get("rms_p2p", float("nan"))
        icon = "[OK]" if q == "good" else ("[??]" if q == "suspect" else "[X]")
        p2p_str = f"{p2p:.4f}" if math.isfinite(float(p2p)) else "N/A"
        comp_lines.append(f"{icon} C{i:02d}  {p2p_str}  {q}")

    comp_text = "\n".join(comp_lines[:15])  # max 15 riadkov
    ax_comp.text(
        0.05,
        0.95,
        "Comparison Stars\n(p2p RMS | quality)\n\n" + comp_text,
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
        with astrofits.open(masterstar_fits_path, memmap=False) as hdul:
            data = np.asarray(hdul[0].data, dtype=np.float64)
    except Exception:
        return

    h, w = data.shape
    half = size_px // 2
    x0 = max(0, int(xc) - half)
    y0 = max(0, int(yc) - half)
    x1 = min(w, x0 + size_px)
    y1 = min(h, y0 + size_px)
    cutout = data[y0:y1, x0:x1]

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
        with astrofits.open(masterstar_fits_path, memmap=False) as hdul:
            data = np.asarray(hdul[0].data, dtype=np.float64)
    except Exception:
        return

    vmin = float(np.percentile(data[np.isfinite(data)], percentile_lo))
    vmax = float(np.percentile(data[np.isfinite(data)], percentile_hi))

    fig, ax = plt.subplots(figsize=(18, 12))
    ax.imshow(data, origin="lower", cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")

    # Zelené štvorčeky — comp hviezdy (unikátne pozície)
    comp_plotted: set[str] = set()
    for _, row in comp_df.iterrows():
        cid = _normalize_gaia_id(row.get("catalog_id", ""))
        if not cid or cid in comp_plotted:
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

    # Červené štvorčeky — target hviezdy
    for _, row in active_targets.iterrows():
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
        name = str(row.get("vsx_name", row.get("catalog_id", "")))[:20]
        ax.text(xc + 14, yc, name, color="red", fontsize=5, va="center")

    ax.set_title("VYVAR — Field Map (červené=target, zelené=comp)", fontsize=10)
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
        with astrofits.open(masterstar_fits_path, memmap=False) as hdul:
            data = np.asarray(hdul[0].data, dtype=np.float64)
    except Exception:
        return

    vmin = float(np.percentile(data[np.isfinite(data)], percentile_lo))
    vmax = float(np.percentile(data[np.isfinite(data)], percentile_hi))

    fig, ax = plt.subplots(figsize=(18, 12))
    ax.imshow(data, origin="lower", cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")

    # Target — červený štvorec
    try:
        tx, ty = float(target_row["x"]), float(target_row["y"])
        rect_t = mpatches.Rectangle(
            (tx - 15, ty - 15),
            30,
            30,
            linewidth=2.0,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(rect_t)
        tname = str(target_row.get("vsx_name", target_row.get("catalog_id", "T")))[:20]
        ax.text(
            tx + 18,
            ty,
            f"T: {tname}",
            color="red",
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
        f"VYVAR — {target_name}\n(červený štvorec=target, zelené krúžky=comp)",
        fontsize=10,
    )
    ax.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Hlavný wrapper — run_phase2a
# ---------------------------------------------------------------------------


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
) -> dict[str, Any]:
    """Hlavný wrapper pre Fázu 2A.

    Globálny FWHM pre apertúru: ``VY_FWHM_GAUSS`` (2D fit z pipeline), inak ``VY_FWHM``
    (DAO, pre apertúru porovnateľné s Gaussian FWHM), inak 2D Gaussian fit
    (``measure_fwhm_from_masterstar``) s nápovedou z ``fwhm_px``.
    Apertúrny polomer = ``aperture_fwhm_factor × FWHM`` (predvolene z ``cfg``).

    Returns:
        dict: n_targets, n_frames, n_lightcurves, summary_csv, field_map_png
    """
    _ = detrended_aligned_dir  # FITS sa v Fáze 2A nepotrebujú — flux z dao_flux v CSV
    output_dir = Path(output_dir)
    lc_dir = output_dir / "lightcurves"
    lc_dir.mkdir(parents=True, exist_ok=True)

    def _p2(msg: str) -> None:
        if progress_cb is not None:
            progress_cb(str(msg))

    _cfg = cfg or AppConfig()
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

    # Načítaj vstupy
    at_df = pd.read_csv(active_targets_csv, low_memory=False)
    comp_df = pd.read_csv(comparison_stars_csv, low_memory=False)

    # Normalizuj catalog_id
    for df in (at_df, comp_df):
        for col in ("catalog_id", "name"):
            if col in df.columns:
                df[col] = df[col].apply(_normalize_gaia_id)

    if "x" not in comp_df.columns or "y" not in comp_df.columns:
        raise ValueError("comparison_stars_per_target.csv musí obsahovať stĺpce x, y pre Fázu 2A")

    # Nájdi per-frame CSV (FITS sa nepoužíva)
    csv_files = sorted(Path(per_frame_csv_dir).glob("proc_*.csv"))
    # Len CSV — bez FITS (flux sa číta z dao_flux v CSV)
    n_frames = len(csv_files)
    logging.info(f"[FÁZA 2A] {len(at_df)} targetov, {n_frames} snímok (CSV only)")
    _p2(f"Fáza 2A: {len(at_df)} cieľov, {n_frames} snímok — načítavam CSV cache…")

    # Načítaj CSV cache raz pre celú Fázu 2A (read_flux_from_csv per target inak 82× na target).
    logging.info("[FÁZA 2A] Načítavam CSV cache...")
    _t_cache = time.time()
    _phase2a_csv_cache: dict[str, pd.DataFrame] = {}
    _needed_cols_2a = [
        "catalog_id",
        "name",
        "bjd_tdb_mid",
        "hjd_mid",
        "jd_mid",
        "dao_flux",
        "noise_floor_adu",
        "aperture_r_px",
        "peak_max_adu",
        "airmass",
        "x",
        "y",
    ]
    for _csv_path in csv_files:
        try:
            _hdr = pd.read_csv(_csv_path, nrows=0)
            _cols = [c for c in _needed_cols_2a if c in _hdr.columns]
            if not _cols:
                continue
            _phase2a_csv_cache[str(_csv_path)] = pd.read_csv(
                _csv_path, usecols=_cols, low_memory=False
            )
        except Exception:  # noqa: BLE001
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

    # Čas + airmass z prvého platného riadku každého per-frame CSV (podľa stem FITS)
    frame_time_lookup: dict[str, dict[str, float]] = {}
    for csv_path in csv_files:
        stem = csv_path.stem
        _csv_tmp = _phase2a_csv_cache.get(str(csv_path))
        if _csv_tmp is None or _csv_tmp.empty:
            continue
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
                break
        except Exception:  # noqa: BLE001
            pass

    # Krok 1: Globálna fixná apertúra — všetky hviezdy (target + comp), faktor × FWHM
    _at_cols = [c for c in ("catalog_id", "x", "y", "mag") if c in at_df.columns]
    _comp_cols = [c for c in ("catalog_id", "x", "y", "mag") if c in comp_df.columns]
    _at_part = at_df[_at_cols].copy()
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
        with astrofits.open(Path(masterstar_fits_path), memmap=False) as _hdul:
            hdr = _hdul[0].header
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
    except Exception as _e:
        logging.warning(f"[FÁZA 2A] Nemôžem čítať FWHM z hlavičky: {_e}")

    if _fwhm_from_header is not None:
        fwhm_px = _fwhm_from_header
    else:
        _fallback_hint = float(fwhm_px) if math.isfinite(fwhm_px) and fwhm_px > 0 else 3.5
        fwhm_px = measure_fwhm_from_masterstar(
            Path(masterstar_fits_path),
            all_stars,
            dao_fwhm_hint=_fallback_hint,
        )
        logging.info(f"[FÁZA 2A] FWHM z Gaussian fit: {fwhm_px:.3f} px")

    _p2(f"Fáza 2A: FWHM={float(fwhm_px):.3f} px — mapa poľa a svetelné krivky…")

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
    else:
        apertures_px = compute_optimal_apertures(
            Path(masterstar_fits_path),
            all_stars,
            fwhm_px,
            aperture_fwhm_factor=_apt_fw,
            annulus_inner_fwhm=annulus_inner_fwhm,
            annulus_outer_fwhm=annulus_outer_fwhm,
        )

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
    save_field_map_png(field_map_path, Path(masterstar_fits_path), at_df, comp_df)

    summary_rows: list[dict[str, Any]] = []
    n_lc = 0

    # Per target loop
    _nt = int(len(at_df))
    for ti, (_, target_row) in enumerate(at_df.iterrows(), start=1):
        target_cid = _normalize_gaia_id(target_row.get("catalog_id", ""))
        target_name = str(target_row.get("vsx_name", target_cid))
        if progress_cb is not None and (
            ti == 1 or ti == _nt or (_nt > 1 and ti % max(1, _nt // 12) == 0)
        ):
            _p2(f"Fáza 2A: cieľ {ti}/{_nt}: {target_name[:50]}")
        logging.info(
            f"[FÁZA 2A] Spúšťam: target={target_name}, "
            f"frames={len(csv_files)}, "
            f"apertura={_apt_fw * float(fwhm_px):.2f}px "
            f"(FWHM={float(fwhm_px):.3f}px × {_apt_fw:.2f})"
        )

        # Comp hviezdy pre tento target
        if "target_catalog_id" in comp_df.columns:
            tc = comp_df["target_catalog_id"].apply(_normalize_gaia_id)
        else:
            tc = pd.Series([""] * len(comp_df), index=comp_df.index)
        target_comps = comp_df[tc == target_cid].copy()

        if target_comps.empty:
            logging.warning(f"[FÁZA 2A] Target {target_name}: žiadne comp hviezdy")
            continue

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
        comp_rms_map = {
            _normalize_gaia_id(r["catalog_id"]): float(r.get("comp_rms", float("nan")))
            for _, r in target_comps.iterrows()
        }

        # Krok 2: Fotometria per snímka
        frame_results: list[pd.DataFrame] = []
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
                star_xy=star_xy,
                xy_tol_px=18.0,
                frame_times=_ft,
                csv_df=_cached_df,
                lookup=_lookup_row,
            )
            if not df_frame.empty:
                frame_results.append(df_frame)

        if not frame_results:
            continue

        all_frames = pd.concat(frame_results, ignore_index=True)

        # Zostav časové rady per hviezda
        def _get_lc(cid: str) -> np.ndarray:
            sub = all_frames[all_frames["catalog_id"] == cid]["mag_inst"].to_numpy(dtype=float)
            return sub

        target_lc = _get_lc(target_cid)
        comp_lc = {cid: _get_lc(cid) for cid in comp_ids}

        # Krok 3: Stability check
        comp_quality = check_comparison_stability(
            comp_lc,
            n_comp_min=3,
            outlier_sigma=stability_sigma,
        )

        # Krok 4: Ensemble normalizácia
        mag_calib, delta_mag, _ = ensemble_normalize(
            target_lc,
            comp_lc,
            comp_catalog_mag,
            comp_quality,
            comp_rms_map=comp_rms_map,
            n_comp_min=3,
            n_comp_max=10,
        )

        # Časové hodnoty targetu
        target_frames = all_frames[all_frames["catalog_id"] == target_cid]
        bjd = target_frames["bjd"].to_numpy(dtype=float)
        hjd = target_frames["hjd"].to_numpy(dtype=float)
        jd = target_frames["jd"].to_numpy(dtype=float)
        err = target_frames["err"].to_numpy(dtype=float)
        ap_arr = target_frames["aperture_r_px"].to_numpy(dtype=float)
        src_files = target_frames["source_file"].tolist()
        sat_flags = (target_frames["flag"] == "saturated").to_numpy(dtype=bool)

        # Airmass detrending (ak je dosť bodov a airmass k dispozícii)
        if "airmass" in target_frames.columns:
            airmass_arr = target_frames["airmass"].to_numpy(dtype=float)
        else:
            airmass_arr = np.full_like(bjd, float("nan"), dtype=float)

        # Najprv detrend na airmass iba podľa saturácie/NaN (outliere detekujeme až po detrende).
        base_flags = [
            "saturated" if bool(sat_flags[i]) else ("normal" if math.isfinite(mag_calib[i]) else "no_data")
            for i in range(len(mag_calib))
        ]
        mag_cal_am, am_slope, _ = airmass_detrend_lc(mag_calib, airmass_arr, base_flags)
        mag_calib_raw = mag_calib.copy()
        mag_calib = mag_cal_am

        # Krok 5: Outlier detekcia (po detrendingu) — outliere pri airmass trende inak nafukujú MAD.
        out_flags = detect_outliers(mag_calib, sat_flags, outlier_sigma=outlier_sigma)

        # Krok 6: Uloženie výstupov
        lc_csv = lc_dir / f"lightcurve_{target_cid}.csv"
        save_lightcurve_csv(
            lc_csv,
            bjd,
            hjd,
            jd,
            airmass_arr,
            target_lc,
            mag_calib_raw,
            mag_calib,
            delta_mag,
            err,
            ap_arr,
            out_flags,
            src_files,
        )

        # Kvalita comp pre UI (tabuľka „Porovnávacie hviezdy“)
        _cq_path = lc_dir / f"comp_quality_{target_cid}.json"
        try:
            _cq_payload = {
                _normalize_gaia_id(cid): str(info.get("quality", ""))
                for cid, info in comp_quality.items()
            }
            _cq_path.write_text(json.dumps(_cq_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

        lc_png = lc_dir / f"lightcurve_{target_cid}.png"
        if _save_png:
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

        cutout_png = lc_dir / f"cutout_{target_cid}.png"
        if _save_png:
            try:
                save_cutout_png(
                    cutout_png,
                    Path(masterstar_fits_path),
                    float(target_row["x"]),
                    float(target_row["y"]),
                    target_name,
                )
            except Exception:
                pass

        # Per-target field map s číslovanými comp hviezdami — vždy (UI)
        try:
            _id_col_comp = "target_catalog_id" if "target_catalog_id" in comp_df.columns else "catalog_id"
            _target_comp = comp_df[comp_df[_id_col_comp].apply(_normalize_gaia_id) == target_cid].copy()
            _fm_target_path = lc_dir / f"field_map_{target_cid}.png"
            save_target_field_map_png(
                _fm_target_path,
                Path(masterstar_fits_path),
                target_row,
                _target_comp,
            )
        except Exception:
            pass

        # Summary riadok
        finite_calib = mag_calib[np.isfinite(mag_calib)]
        n_good_comp = sum(
            1 for q in comp_quality.values() if q.get("quality") in ("good", "suspect")
        )
        n_out = sum(1 for f in out_flags if f.startswith("outlier"))
        n_sat = sum(1 for f in out_flags if f == "saturated")

        summary_rows.append(
            {
                "catalog_id": target_cid,
                "vsx_name": target_name,
                "n_frames": len(bjd),
                "n_good_comp": n_good_comp,
                "n_outliers": n_out,
                "n_saturated": n_sat,
                "lc_rms": float(np.std(finite_calib)) if len(finite_calib) > 1 else float("nan"),
                "lc_median_mag": float(np.median(finite_calib)) if len(finite_calib) > 0 else float("nan"),
                "aperture_px": float(apertures_px.get(target_cid, float("nan"))),
                "am_slope": am_slope,
                "am_detrended": bool(math.isfinite(am_slope)),
                "lc_csv": str(lc_csv),
                "lc_png": str(lc_png),
            }
        )
        n_lc += 1
        lc_rms = float(summary_rows[-1]["lc_rms"])
        r_ap = float(apertures_px.get(target_cid, float("nan")))
        logging.info(
            f"[FÁZA 2A] {target_name}: "
            f"lc_rms={lc_rms:.4f}, "
            f"n_comp={n_good_comp}, "
            f"apertura={r_ap:.2f}px, "
            f"am_slope={float(am_slope):.4f} mag/am"
        )

    # Uloži summary
    summary_csv = output_dir / "photometry_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    logging.info(f"[FÁZA 2A] Hotovo: {n_lc} svetelných kriviek → {output_dir}")
    logging.info(
        f"[FÁZA 2A] Targety bez comp hviezd: "
        f"{len(at_df) - n_lc}/{len(at_df)} "
        f"(žiadne vhodné comp podľa aktuálnych filtrov)"
    )
    _p2(f"Fáza 2A hotovo: {n_lc} kriviek z {n_frames} snímok → {output_dir.name}")

    return {
        "n_targets": len(at_df),
        "n_frames": n_frames,
        "n_lightcurves": n_lc,
        "summary_csv": str(summary_csv),
        "field_map_png": str(field_map_path),
        "output_dir": str(output_dir),
    }


# ======================================================================
# photometry.py (zlúčené do photometry_core)
# ======================================================================

import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils import fits_binning_xy_from_header

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


def _iter_fits_recursive(root: Path) -> list[Path]:
    root = Path(root)
    out: list[Path] = []
    for p in root.rglob("*.fits"):
        if p.is_file():
            out.append(p)
    for p in root.rglob("*.fit"):
        if p.is_file():
            out.append(p)
    out.sort()
    return out


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
    files = [fp for fp in _iter_fits_recursive(root) if fp.with_suffix(".csv").is_file()]
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
    for fp in sample:
        sidecar = fp.with_suffix(".csv")
        try:
            dff = pd.read_csv(sidecar)
        except Exception:
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
    except Exception:
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
    for b, items in by_bin.items():
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
            except Exception:
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
) -> pd.DataFrame:
    """Replace DAO ``flux`` with aperture photometry when enabled; add linearity/BPM flags."""
    out = df.copy()
    arr = np.asarray(data, dtype=np.float32)

    x = pd.to_numeric(out.get("x"), errors="coerce").to_numpy(dtype=np.float64)
    y = pd.to_numeric(out.get("y"), errors="coerce").to_numpy(dtype=np.float64)
    n = len(out)
    if n == 0:
        return out

    # Pôvodný DAO flux z detect_stars_and_match_catalog (historicky v stĺpci ``flux``).
    # Pre ďalšie fázy chceme mať konzistentne:
    # - ``flux_raw``: pôvodný flux (čo prišlo z DAO bloku)
    # - ``dao_flux``: sky-subtrahovaný flux (po aperturnej fotometrii, ak je zapnutá)
    flux_raw = pd.to_numeric(out.get("flux"), errors="coerce").to_numpy(dtype=np.float64)
    if "flux_raw" not in out.columns:
        out["flux_raw"] = flux_raw
    if "dao_flux" not in out.columns:
        out["dao_flux"] = flux_raw

    fwhm_per = np.array(
        [_fwhm_moment_at(arr, float(x[i]), float(y[i])) for i in range(n)],
        dtype=np.float64,
    )
    out["fwhm_estimate_px"] = fwhm_per

    fwhm_moment_med = float(np.nanmedian(fwhm_per[np.isfinite(fwhm_per) & (fwhm_per > 0)]))
    if not math.isfinite(fwhm_moment_med) or fwhm_moment_med <= 0:
        fwhm_moment_med = float("nan")

    DAO_TO_GAUSSIAN = 1.0 / 1.5  # 0.667 — fyzikálne odvodené, setup-nezávislé
    fwhm_gaussian: float | None = None

    # Override pre testovanie
    if gaussian_fwhm_px_override is not None:
        try:
            _ov = float(gaussian_fwhm_px_override)
            if math.isfinite(_ov) and 0.5 < _ov < 30.0:
                fwhm_gaussian = _ov
        except (TypeError, ValueError):
            pass

    # Priorita 1: VY_FWHM z FITS hlavičky (DAO mediánový FWHM z celého runu)
    if fwhm_gaussian is None and hdr is not None:
        try:
            _vy = hdr.get("VY_FWHM", None)
            if _vy is not None:
                _vy_f = float(_vy)
                if math.isfinite(_vy_f) and 0.5 < _vy_f < 30.0:
                    fwhm_gaussian = _vy_f * DAO_TO_GAUSSIAN
                    # Viditeľný log len raz (inak 82× za run).
                    if not bool(getattr(enhance_catalog_dataframe_aperture_bpm, "_did_log_fwhm", False)):
                        logging.info(
                            f"[PHOT] FWHM z VY_FWHM (DAO): {_vy_f:.3f}px × {DAO_TO_GAUSSIAN:.3f} = "
                            f"{float(fwhm_gaussian):.3f}px → apertura = "
                            f"{float(fwhm_gaussian) * float(aperture_fwhm_factor):.3f}px"
                        )
                        setattr(enhance_catalog_dataframe_aperture_bpm, "_did_log_fwhm", True)
        except (TypeError, ValueError):
            pass

    # Fallback: moment FWHM z aktuálnej snímky
    if fwhm_gaussian is None:
        if math.isfinite(fwhm_moment_med) and fwhm_moment_med > 0:
            fwhm_gaussian = fwhm_moment_med * 0.619
            # Viditeľný log len raz (inak 82× za run).
            if not bool(getattr(enhance_catalog_dataframe_aperture_bpm, "_did_log_fwhm", False)):
                logging.info(
                    f"[PHOT] FWHM fallback moment×0.619: {fwhm_gaussian:.3f}px → "
                    f"apertura = {float(fwhm_gaussian) * float(aperture_fwhm_factor):.3f}px"
                )
                setattr(enhance_catalog_dataframe_aperture_bpm, "_did_log_fwhm", True)
        else:
            fwhm_gaussian = float("nan")

    # Sanity check: if computed aperture is out of a reasonable range, fallback to moment median directly.
    r_ap_test = float(aperture_fwhm_factor) * float(fwhm_gaussian) if math.isfinite(float(fwhm_gaussian)) else float("nan")
    if not math.isfinite(r_ap_test) or r_ap_test < 3.0 or r_ap_test > 20.0:
        fwhm_gaussian = float(fwhm_moment_med)
        logging.warning(
            f"[PHOT] Gaussian FWHM fallback na moment: {fwhm_gaussian:.2f}px "
            f"(r_ap={r_ap_test:.2f}px mimo rozsahu)"
        )

    out["fwhm_gaussian_px"] = float(fwhm_gaussian) if math.isfinite(float(fwhm_gaussian)) else float("nan")

    if aperture_enabled and math.isfinite(float(fwhm_gaussian)) and float(fwhm_gaussian) > 0:
        try:
            # Lokálna implementácia: sky-subtracted flux cez CircularAperture + CircularAnnulus.
            from photutils.aperture import CircularAnnulus, CircularAperture
            from photutils.aperture import aperture_photometry as _aphot

            fw = float(fwhm_gaussian)
            r_ap = max(0.5, float(aperture_fwhm_factor) * fw)
            r_in = max(r_ap + 0.5, float(annulus_inner_fwhm) * fw)
            r_out = max(r_in + 0.5, float(annulus_outer_fwhm) * fw)

            pos = np.column_stack([np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)])
            ap = CircularAperture(pos, r=r_ap)
            an = CircularAnnulus(pos, r_in=r_in, r_out=r_out)

            d = np.asarray(arr, dtype=np.float64)
            if np.any(~np.isfinite(d)):
                fill = float(np.nanmedian(d)) if np.any(np.isfinite(d)) else 0.0
                d = np.where(np.isfinite(d), d, fill)

            # Len aperture sum (annulus sky cez medián pixelov, nie sum/area).
            phot_ap = _aphot(d, ap)
            sum_ap = np.asarray(phot_ap["aperture_sum"], dtype=np.float64)
            area_ap = float(ap.area)

            sky_pp_arr = np.zeros(n, dtype=np.float64)
            ann_masks = an.to_mask(method="center")
            if not isinstance(ann_masks, (list, tuple)):
                ann_masks = [ann_masks]
            for i, amask in enumerate(ann_masks):
                try:
                    ann_img = amask.to_image(d.shape)
                    sky_pixels = d[ann_img > 0]
                    if sky_pixels.size >= 5:
                        # Sigma clip: odstráň pixely > median + 2σ (hviezdy v annuluse)
                        sky_med = float(np.median(sky_pixels))
                        sky_std = float(np.std(sky_pixels))
                        clipped = sky_pixels[sky_pixels < sky_med + 2.0 * sky_std]
                        if clipped.size >= 5:
                            sky_pp_arr[i] = float(np.median(clipped))
                        else:
                            sky_pp_arr[i] = sky_med
                    else:
                        sky_pp_arr[i] = float(np.median(d))
                except Exception:  # noqa: BLE001
                    sky_pp_arr[i] = float(np.median(d))

            flux_arr = sum_ap - sky_pp_arr * area_ap
            out["flux"] = flux_arr.astype(np.float64)
            out["dao_flux"] = out["flux"]
            out["aperture_r_px"] = float(r_ap)
            out["sky_annulus_r_in_px"] = float(r_in)
            out["sky_annulus_r_out_px"] = float(r_out)
            # Uložíme sky_pp per hviezda (nie globálna konštanta)
            out["noise_floor_adu"] = sky_pp_arr.astype(np.float64)
        except Exception:  # noqa: BLE001
            out["dao_flux"] = flux_raw
            out["flux"] = flux_raw
    else:
        out["dao_flux"] = flux_raw
        out["flux"] = flux_raw

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


def select_active_targets(
    variable_targets_csv: Path,
    masterstars_csv: Path,
    *,
    frame_w_px: int = 2082,
    frame_h_px: int = 1397,
    edge_margin_px: int = 50,
    match_radius_arcsec: float = 15.0,
) -> pd.DataFrame:
    """Fáza 0: Filtruj VSX premenné → active_targets.

    Pravidlá:
    - Hviezda musí byť v snímke (``x,y`` aspoň ``edge_margin_px`` od okraja efektívneho poľa; to isté číslo
      ako ``chip_interior_margin_px`` vo Fáze 0+1 — jednotné s porovnávačkami a suspected).
    - Šírka/výška sa zväčší z dát ak treba
    - Musí byť nájdená v masterstars_full_match.csv (cross-match < match_radius_arcsec)
    - masterstars záznam musí mať is_usable=True
    - Nesmie byť saturovaná (is_saturated=False)
    - **Žiadny filter na ``vsx_type``** (SXPHE, DSCT, … sa nevyhadzujú samé o sebe).

    Returns:
        DataFrame s active targets — stĺpce z variable_targets + pridané zo masterstars:
        [name, catalog_id, ra_deg, dec_deg, vsx_name, vsx_type, vsx_period,
         x, y, mag, b_v, bp_rp, snr50_ok, is_usable, zone]
    """
    vt = pd.read_csv(variable_targets_csv, low_memory=False)
    ms = pd.read_csv(masterstars_csv, low_memory=False)
    # Normalizuj Gaia ID na string (pandas číta veľké int ako float)
    for _id_col in ("catalog_id", "name"):
        if _id_col in ms.columns:
            ms[_id_col] = _normalize_id_series(ms[_id_col])

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

    # Filter: v snímke s okrajom
    vt["x"] = pd.to_numeric(vt["x"], errors="coerce")
    vt["y"] = pd.to_numeric(vt["y"], errors="coerce")
    in_frame = (
        vt["x"].between(edge_margin_px, fw - edge_margin_px)
        & vt["y"].between(edge_margin_px, fh - edge_margin_px)
    )
    vt_in = vt[in_frame].copy()

    # Cross-match s masterstars cez ra/dec
    ms["ra_deg"] = pd.to_numeric(ms["ra_deg"], errors="coerce")
    ms["dec_deg"] = pd.to_numeric(ms["dec_deg"], errors="coerce")
    vt_in["ra_deg"] = pd.to_numeric(vt_in["ra_deg"], errors="coerce")
    vt_in["dec_deg"] = pd.to_numeric(vt_in["dec_deg"], errors="coerce")

    ms_arr = ms[["ra_deg", "dec_deg"]].to_numpy(dtype=float)

    matched_rows: list[dict] = []
    for _, vrow in vt_in.iterrows():
        ra_v = float(vrow["ra_deg"])
        dec_v = float(vrow["dec_deg"])
        if not (math.isfinite(ra_v) and math.isfinite(dec_v)):
            continue
        # Nájdi najbližší záznam v masterstars
        dists = [
            _angular_distance_deg(ra_v, dec_v, float(ms_arr[i, 0]), float(ms_arr[i, 1]))
            for i in range(len(ms_arr))
        ]
        best_idx = int(np.argmin(dists))
        best_dist_arcsec = dists[best_idx] * 3600.0
        if best_dist_arcsec > match_radius_arcsec:
            continue
        ms_row = ms.iloc[best_idx]
        zone_val = str(ms_row.get("zone", "")).strip().lower()
        # Zahrnúť: linear (plne použiteľná) + noisy1 (slabší signál, možná premenná)
        # Vylúčiť: saturated, noisy2, noisy3, prázdna zóna
        if zone_val in ("noisy2", "noisy3", "saturated", ""):
            continue
        if zone_val not in ("linear", "noisy1"):
            # Fallback pre staré CSV bez noisy sub-kategórií
            if not bool(ms_row.get("is_usable", False)):
                continue
            if bool(ms_row.get("is_saturated", False)):
                continue
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
            "catalog_id": str(ms_row.get("catalog_id", ms_row.get("name", ""))).strip(),
            "mag": float(ms_row.get("mag", float("nan"))),
            "b_v": float(ms_row.get("b_v", float("nan"))),
            "bp_rp": float(ms_row.get("bp_rp", float("nan"))),
            "snr50_ok": bool(ms_row.get("snr50_ok", False)),
            "zone": str(ms_row.get("zone", "")),
            "is_usable": bool(ms_row.get("is_usable", False)),
            "match_dist_arcsec": best_dist_arcsec,
        }
        matched_rows.append(rec)

    if not matched_rows:
        return pd.DataFrame()

    result = pd.DataFrame(matched_rows)
    # Zabezpeč string formát Gaia ID v output CSV
    if "catalog_id" in result.columns:
        def _gaia_id_str(x: Any) -> str:
            s = str(x).strip()
            if s in ("", "nan"):
                return ""
            try:
                return str(int(float(s)))
            except Exception:  # noqa: BLE001
                return s

        result["catalog_id"] = result["catalog_id"].apply(_gaia_id_str)
    logging.info(
        f"[FÁZA 0] active_targets: {len(result)} / {len(vt)} VSX hviezd "
        f"(in_frame={int(in_frame.sum())}, matched+usable={len(result)})"
    )
    return result.reset_index(drop=True)


def select_comparison_stars_per_target(
    target: pd.Series,
    masterstars_df: pd.DataFrame,
    per_frame_csv_paths: list[Path],
    *,
    csv_cache: dict[str, pd.DataFrame] | None = None,
    fwhm_px: float = 3.7,
    max_dist_deg: float = 1.0,
    max_mag_diff: float = 0.25,  # ±0.25 mag od targetu (základ; pri jasnom ciele viď ``mag_tol`` nižšie)
    max_bv_diff: float = 0.15,  # ±0.15 B-V od targetu
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

    Returns:
        DataFrame s porovnávacími hviezdami pre tento target, zoradený podľa RMS ASC.
        Prázdny DataFrame ak sa nenájde dostatok stabilných hviezd.
    """
    _ = fwhm_px
    ms = masterstars_df.copy()
    # Normalizuj Gaia ID na string
    for _id_col in ("catalog_id", "name"):
        if _id_col in ms.columns:
            ms[_id_col] = _normalize_id_series(ms[_id_col])

    # Normalizuj bool stĺpce
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

    ra_t = float(target["ra_deg"])
    dec_t = float(target["dec_deg"])
    mag_t = float(target.get("mag", float("nan")))
    target_cid = str(target.get("catalog_id", ""))

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

    # ── Krok 1: Filter kandidátov ──
    ms["_dist_deg"] = ms.apply(
        lambda r: _angular_distance_deg(
            ra_t,
            dec_t,
            float(r["ra_deg"]) if math.isfinite(float(r["ra_deg"])) else 999.0,
            float(r["dec_deg"]) if math.isfinite(float(r["dec_deg"])) else 999.0,
        ),
        axis=1,
    )
    cand_mask = (
        ms["_dist_deg"].le(max_dist_deg)
        & _bool_col(ms.get("is_usable", pd.Series(True, index=ms.index)))
        & ~_bool_col(ms.get("is_saturated", pd.Series(False, index=ms.index)))
        & ~_bool_col(ms.get("is_noisy", pd.Series(False, index=ms.index)))
        & ~_bool_col(ms.get("vsx_known_variable", pd.Series(False, index=ms.index)))
        & ~_bool_col(ms.get("likely_saturated", pd.Series(False, index=ms.index)))
    )
    # Vylúč samotný target
    if target_cid:
        cand_mask &= ms.get("catalog_id", ms.get("name", pd.Series("", index=ms.index))).astype(str) != target_cid

    # Jednotný vnútorný okraj čipu (premenné / comp / suspected rovnaké pravidlá)
    _cm = int(chip_interior_margin_px)
    if (
        _cm > 0
        and chip_fw is not None
        and chip_fh is not None
        and int(chip_fw) > 2 * _cm
        and int(chip_fh) > 2 * _cm
        and "x" in ms.columns
        and "y" in ms.columns
    ):
        _xn = pd.to_numeric(ms["x"], errors="coerce")
        _yn = pd.to_numeric(ms["y"], errors="coerce")
        cand_mask &= _xn.between(_cm, int(chip_fw) - _cm) & _yn.between(_cm, int(chip_fh) - _cm)

    # Hard filter: minimálna vzdialenosť od targetu
    if math.isfinite(min_dist_arcsec) and min_dist_arcsec > 0:
        min_dist_deg = float(min_dist_arcsec) / 3600.0
        cand_mask &= ms["_dist_deg"].ge(min_dist_deg)

    # Hard filter: |ΔMag| <= mag_tol (základ max_mag_diff; pri jasnom ciele zväčší floor)
    if math.isfinite(mag_t):
        ms["_mag"] = pd.to_numeric(ms.get("mag", ms.get("phot_g_mean_mag")), errors="coerce")
        cand_mask &= ms["_mag"].sub(mag_t).abs().le(mag_tol)

    # Hard filter: |ΔB-V| <= max_bv_diff (len ak poznáme B-V targetu aj kandidáta)
    target_bv_pre = float(target.get("b_v", float("nan")))
    if math.isfinite(target_bv_pre) and math.isfinite(max_bv_diff):
        ms["_bv"] = pd.to_numeric(ms.get("b_v", pd.Series(dtype=float)), errors="coerce")
        bv_known_mask = ms["_bv"].notna()
        bv_filter = ~bv_known_mask | ms["_bv"].sub(target_bv_pre).abs().le(max_bv_diff)
        cand_mask &= bv_filter
        n_bv_filtered = int((bv_known_mask & ~bv_filter).sum())
        if n_bv_filtered > 0:
            logging.debug(
                f"[FÁZA 1] Target {target_cid}: |ΔB-V| filter odstránil "
                f"{n_bv_filtered} kandidátov (threshold={max_bv_diff:.2f})"
            )

    # Filter A: Gaia objektové flagy
    _n_before_a = int(cand_mask.sum())

    # gaia_nss=True → non-single star (binárka/dvojhviezda) → variabilný flux
    if exclude_gaia_nss and "gaia_nss" in ms.columns:
        _nss_rej = cand_mask & _bool_col(ms["gaia_nss"])
        cand_mask &= ~_bool_col(ms["gaia_nss"])
        _n_rej = int(_nss_rej.sum())
        if _n_rej > 0:
            logging.info(
                f"[FÁZA 1] Target {target_cid}: Filter A (gaia_nss) vylúčil {_n_rej} kandidátov"
            )

    # gaia_qso, gaia_gal → nie bodový zdroj → systematické chyby
    if exclude_gaia_extobj:
        _rej_ext_total = 0
        for _ext_col in ("gaia_qso", "gaia_gal"):
            if _ext_col in ms.columns:
                _rej_mask = cand_mask & _bool_col(ms[_ext_col])
                cand_mask &= ~_bool_col(ms[_ext_col])
                _rej = int(_rej_mask.sum())
                _rej_ext_total += _rej
                if _rej > 0:
                    logging.info(
                        f"[FÁZA 1] Target {target_cid}: Filter A ({_ext_col}) vylúčil {_rej} kandidátov"
                    )

        if _rej_ext_total == 0:
            _ = _rej_ext_total  # noqa: B018

    _n_after_a = int(cand_mask.sum())
    _rej_a_total = _n_before_a - _n_after_a
    if _rej_a_total > 0:
        logging.debug(
            f"[FÁZA 1] Target {target_cid}: Filter A celkom vylúčil {_rej_a_total} kandidátov "
            f"({_n_before_a} → {_n_after_a})"
        )

    # Zahrň DET hviezdy (bez Gaia ID) ak majú snr50_ok a nie sú saturované.
    # Tieto môžu byť stabilné comp hviezdy aj bez katalógového záznamu.
    det_mask = (
        ms.get("catalog_id", ms.get("name", pd.Series("", index=ms.index)))
        .astype(str)
        .str.startswith("DET")
        & ms["_dist_deg"].le(max_dist_deg)
        & ~_bool_col(ms.get("is_saturated", pd.Series(False, index=ms.index)))
        & ~_bool_col(ms.get("likely_saturated", pd.Series(False, index=ms.index)))
        & _bool_col(ms.get("snr50_ok", pd.Series(False, index=ms.index)))
        & ~_bool_col(ms.get("vsx_known_variable", pd.Series(False, index=ms.index)))
    )
    if target_cid:
        det_mask &= (
            ms.get("catalog_id", ms.get("name", pd.Series("", index=ms.index))).astype(str)
            != target_cid
        )
    if math.isfinite(min_dist_arcsec) and min_dist_arcsec > 0:
        det_mask &= ms["_dist_deg"].ge(float(min_dist_arcsec) / 3600.0)

    cand_mask = cand_mask | det_mask

    # Base mask for tiered Δmag/ΔB-V selection (keeps all non-photometric filters).
    # NOTE: cand_mask already includes many filters + DET; we rebuild explicitly for clarity.
    _base_mask = (
        ms["_dist_deg"].le(max_dist_deg)
        & _bool_col(ms.get("is_usable", pd.Series(True, index=ms.index)))
        & ~_bool_col(ms.get("is_saturated", pd.Series(False, index=ms.index)))
        & ~_bool_col(ms.get("is_noisy", pd.Series(False, index=ms.index)))
        & ~_bool_col(ms.get("vsx_known_variable", pd.Series(False, index=ms.index)))
        & ~_bool_col(ms.get("likely_saturated", pd.Series(False, index=ms.index)))
    )
    if target_cid:
        _base_mask &= (
            ms.get("catalog_id", ms.get("name", pd.Series("", index=ms.index))).astype(str) != target_cid
        )
    if math.isfinite(min_dist_arcsec) and min_dist_arcsec > 0:
        _base_mask &= ms["_dist_deg"].ge(float(min_dist_arcsec) / 3600.0)
    if exclude_gaia_nss and "gaia_nss" in ms.columns:
        _base_mask &= ~_bool_col(ms["gaia_nss"])
    if exclude_gaia_extobj:
        for _ext_col in ("gaia_qso", "gaia_gal"):
            if _ext_col in ms.columns:
                _base_mask &= ~_bool_col(ms[_ext_col])

    # Prepare numeric mag/BV once (used by tiers)
    if "_mag" not in ms.columns:
        ms["_mag"] = pd.to_numeric(ms.get("mag", ms.get("phot_g_mean_mag")), errors="coerce")
    if "_bv" not in ms.columns:
        ms["_bv"] = pd.to_numeric(ms.get("b_v", pd.Series(dtype=float)), errors="coerce")

    # Start with a broad candidate set (emergency tier) for one-pass per-frame metrics.
    candidates_pre = ms[_base_mask | det_mask].copy()
    if len(candidates_pre) < n_comp_min:
        logging.warning(
            f"[FÁZA 1] Target {target_cid}: len {len(candidates_pre)} kandidátov "
            f"< n_comp_min={n_comp_min} — preskakujem."
        )
        return pd.DataFrame()

    # Izolácia v aperture — hard filter (pred RMS výpočtom)
    # Kandidát nesmie mať významného suseda priamo v jeho apertúre (r < r_ap).
    # r_ap ≈ aperture_fwhm_factor × Gaussian FWHM (ak fwhm_px dostupné), inak 7 px.
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

    # Identifikátory kandidátov
    # ``catalog_id`` býva v CSV často uložené ako float (scientific) a môže stratiť presnosť;
    # ``name`` obsahuje presný identifikátor (Gaia ID ako string alebo DET_*). Preferuj preto ``name``.
    id_col = "name" if "name" in candidates_pre.columns else ("catalog_id" if "catalog_id" in candidates_pre.columns else "name")
    cand_ids = set(candidates_pre[id_col].astype(str).str.strip())

    # ── Krok 2: Načítaj flux z per-frame CSV ──
    # Načítaj len potrebné stĺpce — úspora 78 % pamäte
    avail_cols = _PHASE_USECOLS_PERFRAME.copy()
    flux_map: dict[str, list[float]] = {cid: [] for cid in cand_ids}
    n_frames_loaded = 0

    # Filter C → Contamination index (soft penalizácia v scoringu)
    # Namiesto hard-exclusion vypočítaj contamination ratio per kandidát.
    # Hustá oblasť neba: hard filter by vylúčil väčšinu kandidátov.
    # Riešenie: crowding sa prejaví ako penalizácia v combined score (Krok 5).
    contamination_map: dict[str, float] = {}  # cid → sum flux ratio od susedov

    # Pre filter B: zbieraj psf_chi2 a fwhm_estimate_px per kandidát
    psf_chi2_map: dict[str, list[float]] = {cid: [] for cid in cand_ids}
    fwhm_map: dict[str, list[float]] = {cid: [] for cid in cand_ids}
    frame_fwhm_medians: list[float] = []  # mediánový FWHM všetkých hviezd per snímka
    # Saturácia naprieč framami: peak_max_adu > saturate_limit_adu_85pct
    peak_over_map: dict[str, int] = {cid: 0 for cid in cand_ids}
    peak_total_map: dict[str, int] = {cid: 0 for cid in cand_ids}
    # Skutočný 5σ SNR (median cez framy)
    snr_map: dict[str, list[float]] = {cid: [] for cid in cand_ids}

    # ── CSV cache: ak príde zvonka (Fáza 1 shared), použi ju; inak vybuduj lokálne
    if csv_cache is None:
        csv_cache = {}
        for _csv_path in per_frame_csv_paths:
            try:
                _hdr = pd.read_csv(_csv_path, nrows=0)
                _use_cols = [c for c in avail_cols if c in _hdr.columns]
                _actual_flux = flux_col if flux_col in _hdr.columns else "flux"
                if _actual_flux not in _use_cols:
                    _use_cols.append(_actual_flux)
                _name_col = (
                    "name"
                    if "name" in _hdr.columns
                    else ("catalog_id" if "catalog_id" in _hdr.columns else "name")
                )
                if "mag" not in _use_cols and "mag" in _hdr.columns:
                    _use_cols.append("mag")
                if "psf_chi2" in _hdr.columns and "psf_chi2" not in _use_cols:
                    _use_cols.append("psf_chi2")
                if "fwhm_estimate_px" in _hdr.columns and "fwhm_estimate_px" not in _use_cols:
                    _use_cols.append("fwhm_estimate_px")
                if "peak_max_adu" in _hdr.columns and "peak_max_adu" not in _use_cols:
                    _use_cols.append("peak_max_adu")
                if (
                    "saturate_limit_adu_85pct" in _hdr.columns
                    and "saturate_limit_adu_85pct" not in _use_cols
                ):
                    _use_cols.append("saturate_limit_adu_85pct")
                _df0 = pd.read_csv(_csv_path, usecols=_use_cols, low_memory=False)
                _df0[_name_col] = _normalize_id_series(_df0[_name_col])
                _df0[_actual_flux] = pd.to_numeric(_df0[_actual_flux], errors="coerce")
                if "peak_max_adu" in _df0.columns:
                    _df0["peak_max_adu"] = pd.to_numeric(_df0["peak_max_adu"], errors="coerce")
                if "saturate_limit_adu_85pct" in _df0.columns:
                    _df0["saturate_limit_adu_85pct"] = pd.to_numeric(
                        _df0["saturate_limit_adu_85pct"], errors="coerce"
                    )
                csv_cache[str(_csv_path)] = _df0
            except Exception:  # noqa: BLE001
                continue

    for csv_path in per_frame_csv_paths:
        df = csv_cache.get(str(csv_path))
        if df is None or df.empty:
            continue
        try:
            name_col = "name" if "name" in df.columns else ("catalog_id" if "catalog_id" in df.columns else "name")
            actual_flux_col = flux_col if flux_col in df.columns else "flux"

            # Saturácia naprieč framami (nezávisle od flux>0):
            # peak_max_adu > saturate_limit_adu_85pct
            if "peak_max_adu" in df.columns and "saturate_limit_adu_85pct" in df.columns:
                for _, _row in df[df[name_col].isin(cand_ids)].iterrows():
                    _cid = str(_row[name_col])
                    peak = float(_row.get("peak_max_adu", float("nan")))
                    limit = float(_row.get("saturate_limit_adu_85pct", float("nan")))
                    if math.isfinite(peak) and math.isfinite(limit) and limit > 0:
                        peak_total_map[_cid] = int(peak_total_map.get(_cid, 0)) + 1
                        if peak > limit:
                            peak_over_map[_cid] = int(peak_over_map.get(_cid, 0)) + 1

            # Zbieraj psf_chi2 a fwhm_estimate_px pre Filter B
            if "psf_chi2" in df.columns:
                for _, _row in df[df[name_col].isin(cand_ids)].iterrows():
                    _cid = str(_row[name_col])
                    _chi2 = float(_row.get("psf_chi2", float("nan")))
                    if math.isfinite(_chi2) and _chi2 > 0:
                        psf_chi2_map[_cid].append(_chi2)

            if "fwhm_estimate_px" in df.columns:
                _fwhm_col = pd.to_numeric(df["fwhm_estimate_px"], errors="coerce")
                # Mediánový FWHM všetkých hviezd na tejto snímke
                _frame_fwhm_med = float(_fwhm_col.median())
                if math.isfinite(_frame_fwhm_med) and _frame_fwhm_med > 0:
                    frame_fwhm_medians.append(_frame_fwhm_med)
                # FWHM per kandidát
                for _, _row in df[df[name_col].isin(cand_ids)].iterrows():
                    _cid = str(_row[name_col])
                    _fwhm = float(_row.get("fwhm_estimate_px", float("nan")))
                    if math.isfinite(_fwhm) and _fwhm > 0:
                        fwhm_map[_cid].append(_fwhm)

            # Len kandidáti s platným flux
            sub = df[df[name_col].isin(cand_ids) & df[actual_flux_col].gt(0)].copy()
            if sub.empty:
                continue

            # Mag-bin normalizácia: medián zvlášť pre každý mag bin (0.5 mag šírka)
            # Eliminuje sky drift ktorý závisí od jasnosti hviezdy
            mag_col_frame = "mag" if "mag" in df.columns else None
            if mag_col_frame and mag_col_frame in sub.columns:
                sub = sub.copy()
                sub["_mag_num"] = pd.to_numeric(sub[mag_col_frame], errors="coerce")
                sub["_mag_bin"] = (sub["_mag_num"] / 0.5).apply(
                    lambda x: int(x) if math.isfinite(x) else -1
                )
                bin_meds: dict[int, float] = {}
                for b, grp in sub.groupby("_mag_bin"):
                    bmed = float(grp[actual_flux_col].median())
                    if math.isfinite(bmed) and bmed > 0:
                        bin_meds[int(b)] = bmed
                if not bin_meds:
                    continue
            else:
                # Fallback: globálny medián
                frame_med = float(sub[actual_flux_col].median())
                if not math.isfinite(frame_med) or frame_med <= 0:
                    continue
                bin_meds = {}

            n_frames_loaded += 1
            for _, row in sub.iterrows():
                cid = str(row[name_col])
                raw_flux = float(row[actual_flux_col])
                if not math.isfinite(raw_flux) or raw_flux <= 0:
                    continue

                # Skutočný 5σ SNR:
                # SNR = dao_flux / sqrt(dao_flux + noise_floor_adu * aperture_area)
                flux_snr = float(row.get("dao_flux", raw_flux))
                sky = float(row.get("noise_floor_adu", 0.0))
                r_ap = float(row.get("aperture_r_px", 7.0))
                area = math.pi * r_ap * r_ap if math.isfinite(r_ap) and r_ap > 0 else float("nan")
                if math.isfinite(flux_snr) and flux_snr > 0 and math.isfinite(sky) and math.isfinite(area):
                    denom = flux_snr + max(0.0, sky) * area
                    if denom > 0:
                        snr = flux_snr / math.sqrt(denom)
                        if math.isfinite(snr):
                            snr_map[cid].append(float(snr))
                # Normalizácia voči mediánu hviezd rovnakej jasnosti
                if bin_meds:
                    mag_num = (
                        float(row.get("_mag_num", float("nan")))
                        if "_mag_num" in row.index
                        else float("nan")
                    )
                    b = int(mag_num / 0.5) if math.isfinite(mag_num) else -1
                    norm_med = bin_meds.get(b)
                    if norm_med is None:
                        # Fallback na najbližší bin
                        closest = min(bin_meds.keys(), key=lambda k: abs(k - b))
                        norm_med = bin_meds[closest]
                else:
                    norm_med = frame_med  # type: ignore[assignment]
                rel = raw_flux / norm_med
                if math.isfinite(rel) and rel > 0:
                    flux_map[cid].append(rel)

        except Exception:  # noqa: BLE001
            continue

    min_frames = max(3, int(n_frames_loaded * min_frames_frac))

    # Filter SAT: vylúč kandidátov, ktorí sú nad 85% sat limitu vo viac než 10% framov
    _sat_rejected: set[str] = set()
    for cid in list(flux_map.keys()):
        total = int(peak_total_map.get(cid, 0) or 0)
        over = int(peak_over_map.get(cid, 0) or 0)
        if total >= 10 and total > 0 and (float(over) / float(total)) > 0.10:
            flux_map.pop(cid, None)
            _sat_rejected.add(cid)
            logging.info(
                f"[FÁZA 1] Saturácia filter: vylúčený {cid} "
                f"({over}/{total} framov nad 85% limitom)"
            )
    if _sat_rejected:
        logging.info(f"[FÁZA 1] Celkom vylúčených kvôli saturácii: {len(_sat_rejected)}")

    # Filter SNR: vylúč kandidátov s median SNR < 5σ
    _snr_rejected: set[str] = set()
    for cid in list(flux_map.keys()):
        snrs = snr_map.get(cid, [])
        if len(snrs) >= 5:
            snr_median = float(np.median(np.asarray(snrs, dtype=np.float64)))
            if math.isfinite(snr_median) and snr_median < 5.0:
                flux_map.pop(cid, None)
                _snr_rejected.add(cid)
                logging.info(
                    f"[FÁZA 1] SNR filter: vylúčený {cid} "
                    f"(median SNR={snr_median:.1f} < 5)"
                )

    # Filter B: PSF chi² a FWHM blend detekcia
    _global_fwhm_med = float(np.median(frame_fwhm_medians)) if frame_fwhm_medians else float("nan")
    _b_rejected: set[str] = set()

    if math.isfinite(max_psf_chi2):
        for _cid, _chi2_vals in psf_chi2_map.items():
            if len(_chi2_vals) < 3:
                continue
            _med_chi2 = float(np.median(_chi2_vals))
            if _med_chi2 > max_psf_chi2:
                _b_rejected.add(_cid)
                logging.debug(
                    f"[FÁZA 1] Blend filter (PSF chi²): vylúčený {_cid} "
                    f"(median chi²={_med_chi2:.2f} > {max_psf_chi2:.2f})"
                )

    if math.isfinite(max_fwhm_factor) and math.isfinite(_global_fwhm_med) and _global_fwhm_med > 0:
        for _cid, _fwhm_vals in fwhm_map.items():
            if len(_fwhm_vals) < 3:
                continue
            _med_fwhm = float(np.median(_fwhm_vals))
            _fwhm_ratio = _med_fwhm / _global_fwhm_med
            if _fwhm_ratio > max_fwhm_factor:
                _b_rejected.add(_cid)
                logging.debug(
                    f"[FÁZA 1] Blend filter (FWHM): vylúčený {_cid} "
                    f"(median FWHM={_med_fwhm:.2f}px, ratio={_fwhm_ratio:.2f} > {max_fwhm_factor:.2f})"
                )

    if _b_rejected:
        logging.info(
            f"[FÁZA 1] Target {target_cid}: Filter B (PSF/FWHM) vylúčil "
            f"{len(_b_rejected)} kandidátov: {sorted(_b_rejected)}"
        )
        for _cid in _b_rejected:
            flux_map.pop(_cid, None)

    # Filter C → Contamination index (soft penalizácia v scoringu)
    # Namiesto hard-exclusion vypočítaj contamination ratio per kandidát.
    # Hustá oblasť neba: hard filter by vylúčil väčšinu kandidátov.
    # Riešenie: crowding sa prejaví ako penalizácia v combined score (Krok 5).
    if isolation_radius_px > 0 and "x" in ms.columns and "y" in ms.columns:
        ms_reset = ms.reset_index(drop=True)
        _id_col_ms = "catalog_id" if "catalog_id" in ms_reset.columns else "name"

        # Flux proxy: dao_flux > flux > phot_g_mean_mag (mag → relatívny flux)
        _flux_col_ms = next((fc for fc in ("dao_flux", "flux") if fc in ms_reset.columns), None)
        _mag_col_ms = next(
            (mc for mc in ("phot_g_mean_mag", "catalog_mag", "mag") if mc in ms_reset.columns),
            None,
        )

        # Zostavíme vektory pre rýchly výpočet vzdialeností
        _ms_x_all = pd.to_numeric(ms_reset["x"], errors="coerce").to_numpy(dtype=np.float64)
        _ms_y_all = pd.to_numeric(ms_reset["y"], errors="coerce").to_numpy(dtype=np.float64)

        if _flux_col_ms:
            _ms_flux_all = pd.to_numeric(ms_reset[_flux_col_ms], errors="coerce").to_numpy(dtype=np.float64)
        elif _mag_col_ms:
            _mags_all = pd.to_numeric(ms_reset[_mag_col_ms], errors="coerce").to_numpy(dtype=np.float64)
            _ms_flux_all = np.where(np.isfinite(_mags_all), 10 ** (-0.4 * _mags_all), np.nan)
        else:
            _ms_flux_all = np.ones(len(ms_reset))

        _ms_mag_all = (
            pd.to_numeric(ms_reset[_mag_col_ms], errors="coerce").to_numpy(dtype=np.float64)
            if _mag_col_ms
            else np.full(len(ms_reset), np.nan, dtype=np.float64)
        )

        # Lookup: catalog_id → riadok index v ms_reset
        _cid_to_idx: dict[str, int] = {}
        for _ri, _rrow in ms_reset.iterrows():
            _rcid = _normalize_id_value(_rrow.get(_id_col_ms, ""))
            if _rcid:
                _cid_to_idx[_rcid] = int(_ri)

        for _cid in flux_map:
            _ci = _cid_to_idx.get(_cid)
            if _ci is None:
                continue
            _cx = _ms_x_all[_ci]
            _cy = _ms_y_all[_ci]
            _cflux = _ms_flux_all[_ci]
            if not (math.isfinite(_cx) and math.isfinite(_cy)):
                continue
            if not math.isfinite(_cflux) or _cflux <= 0:
                continue

            _dx = _ms_x_all - _cx
            _dy = _ms_y_all - _cy
            _dists = np.sqrt(_dx * _dx + _dy * _dy)
            _neighbor_mask = (
                (_dists > 0.5)
                & (_dists <= isolation_radius_px)
                & np.isfinite(_ms_flux_all)
                & (_ms_flux_all > 0)
            )
            # Zahrnúť len susedov do 3 mag od kandidáta
            mag_cand = float(_ms_mag_all[_ci]) if _ci < len(_ms_mag_all) else float("nan")
            if math.isfinite(mag_cand):
                _neighbor_mask = _neighbor_mask & (
                    ~np.isfinite(_ms_mag_all) | ((_ms_mag_all - mag_cand) <= 3.0)
                )
            if not np.any(_neighbor_mask):
                contamination_map[_cid] = 0.0
                continue

            # Contamination = súčet flux susedov / flux kandidáta
            # (súčet, nie maximum — viac slabých susedov = väčší efekt)
            _neighbor_flux_sum = float(np.sum(_ms_flux_all[_neighbor_mask]))
            contamination_map[_cid] = min(_neighbor_flux_sum / _cflux, 2.0)  # cap na 2.0 (200%)

        if contamination_map:
            _cont_vals = list(contamination_map.values())
            logging.debug(
                f"[FÁZA 1] Target {target_cid}: contamination index "
                f"median={float(np.median(_cont_vals)):.3f}, "
                f"max={max(_cont_vals):.3f} "
                f"(isolation_radius={isolation_radius_px:.0f}px)"
            )

    # ── Krok 2b: Airmass detrending ──
    # Polynomický fit (stupeň 2) na časový rad relatívneho flux odstráni
    # systematický airmass trend. Residuály = skutočná fotometrická variabilita.
    for cid in list(flux_map.keys()):
        vals = flux_map[cid]
        if len(vals) < 6:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        t = np.linspace(0.0, 1.0, len(arr))
        try:
            coeffs = np.polyfit(t, arr, 2)
            trend_fit = np.polyval(coeffs, t)
            safe_trend = np.where(np.abs(trend_fit) > 1e-9, trend_fit, 1.0)
            detrended = arr / safe_trend
            med_dt = float(np.median(detrended))
            if math.isfinite(med_dt) and med_dt > 0:
                flux_map[cid] = (detrended / med_dt).tolist()
        except Exception:  # noqa: BLE001
            pass  # Ponechaj pôvodné hodnoty ak fit zlyhá

    # ── Krok 3: RMS scatter per kandidát ──
    rms_map: dict[str, float] = {}
    for cid, vals in flux_map.items():
        if len(vals) < min_frames:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        rms = float(np.sqrt(np.mean((arr - 1.0) ** 2)))
        if math.isfinite(rms):
            rms_map[cid] = rms

    # Zoradené RMS pre fallback kroky
    sorted_rms_map: dict[str, float] = dict(sorted(rms_map.items(), key=lambda kv: float(kv[1])))

    # Tvrdý RMS limit — odmietni nestabilné hviezdy bez ohľadu na ranking
    if math.isfinite(max_comp_rms) and max_comp_rms > 0:
        n_before = len(rms_map)
        rms_map = {cid: rms for cid, rms in sorted_rms_map.items() if rms <= max_comp_rms}
        n_rejected = n_before - len(rms_map)
        if n_rejected > 0:
            logging.info(
                f"[FÁZA 1] Target {target_cid}: tvrdý RMS filter (>{max_comp_rms:.3f}) "
                f"odmietol {n_rejected} kandidátov, zostáva {len(rms_map)}"
            )

    # Fallback na uvoľnený RMS limit ak stále <n_comp_min
    if len(rms_map) < n_comp_min and math.isfinite(max_comp_rms) and max_comp_rms > 0:
        _good: dict[str, float] = dict(rms_map)
        _rms_fallback_steps = [float(max_comp_rms), 0.08, 0.15]
        for _rms_limit in _rms_fallback_steps:
            _good = {cid: rms for cid, rms in sorted_rms_map.items() if rms <= float(_rms_limit)}
            if len(_good) >= n_comp_min:
                if float(_rms_limit) > float(max_comp_rms):
                    logging.warning(
                        f"[FÁZA 1] Target {target_cid}: RMS fallback "
                        f"max_comp_rms {float(max_comp_rms):.3f} → {float(_rms_limit):.3f}, "
                        f"nájdených {len(_good)} comp"
                    )
                break
        rms_map = _good

    if len(rms_map) < n_comp_min:
        logging.warning(
            f"[FÁZA 1] Target {target_cid}: len {len(rms_map)} kandidátov "
            f"s dostatkom snímok < n_comp_min={n_comp_min}."
        )
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Adaptive tier selection (Δmag/ΔB-V) based on how many GOOD comp remain
    # ------------------------------------------------------------------
    TIERS = [
        {"key": "TIER1", "mag": 0.25, "bv": 0.15, "name": "TIER1_ideal"},
        {"key": "TIER2", "mag": 0.50, "bv": 0.30, "name": "TIER2_standard"},
        {"key": "TIER3", "mag": 1.00, "bv": 0.50, "name": "TIER3_relaxed"},
        {"key": "TIER4", "mag": 2.00, "bv": float("inf"), "name": "TIER4_emergency"},
    ]

    def _apply_aperture_isolation_safe(cands: pd.DataFrame) -> pd.DataFrame:
        """Apply aperture isolation; if it would drop below n_comp_min, keep original."""
        if cands.empty:
            return cands
        try:
            ms_arr_x2 = ms_arr_x
            ms_arr_y2 = ms_arr_y
            ms_arr_mag2 = ms_arr_mag
        except Exception:  # noqa: BLE001
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

    selected_tier: str | None = None
    used_tier: dict[str, Any] | None = None
    candidates = pd.DataFrame()

    # At this point we only have the hard RMS filter map (`rms_map`).
    # Use it to evaluate tiers; the iterative ensemble filtering happens later.
    active_keys = set(rms_map.keys())
    for tier in TIERS:
        _mask = _base_mask.copy()
        if math.isfinite(mag_t):
            # If mag is known, enforce tier; if mag is unknown, keep as fallback.
            _tier_dm = max(float(tier["mag"]), float(mag_tol))
            _mask &= ms["_mag"].isna() | ms["_mag"].sub(mag_t).abs().le(_tier_dm)
        if math.isfinite(target_bv_pre) and math.isfinite(float(tier["bv"])) and float(tier["bv"]) < float("inf"):
            bv_known = ms["_bv"].notna()
            _mask &= ~bv_known | ms["_bv"].sub(target_bv_pre).abs().le(float(tier["bv"]))

        _cands = ms[_mask | det_mask].copy()
        if _cands.empty:
            continue
        _cands = _apply_aperture_isolation_safe(_cands)

        id_col_t = "name" if "name" in _cands.columns else ("catalog_id" if "catalog_id" in _cands.columns else "name")
        ids_t = set(_cands[id_col_t].astype(str).str.strip())
        good_count = int(len([cid for cid in active_keys if cid in ids_t]))
        if good_count >= int(n_comp_min):
            selected_tier = str(tier["key"])
            used_tier = dict(tier)
            if math.isfinite(mag_t):
                used_tier["mag"] = max(float(tier["mag"]), float(mag_tol))
            candidates = _cands
            break

    if selected_tier is None or used_tier is None or candidates.empty:
        logging.warning(f"[FÁZA 1] {target_cid}: žiadny tier nedal ≥{n_comp_min} comp")
        return pd.DataFrame()

    logging.info(
        f"[FÁZA 1] {target_cid}: {str(used_tier.get('name', selected_tier))} "
        f"(Δmag≤{float(used_tier['mag']):.2f}, ΔB-V≤{used_tier['bv']}, "
        f"target={len(candidates)} cand)"
    )

    # ── Krok 4: Iteratívny ensemble filter (robustný MAD) ──
    # Prah = median + k × (MAD / 0.6745)
    # MAD / 0.6745 = konzistentný estimátor σ robustný voči outlierom
    # k = rms_outlier_sigma (default 3.0)
    _MAD_CONSISTENCY = 0.6745  # normalizačný faktor MAD → σ ekvivalent
    # Restrict to selected tier IDs before ensemble outlier filtering.
    id_col_cand = "name" if "name" in candidates.columns else ("catalog_id" if "catalog_id" in candidates.columns else "name")
    tier_ids = set(candidates[id_col_cand].astype(str).str.strip())
    active = {cid: rms for cid, rms in rms_map.items() if cid in tier_ids}
    if len(active) < n_comp_min:
        logging.warning(
            f"[FÁZA 1] {target_cid}: {selected_tier} má len {len(active)} comp po RMS filtre "
            f"< n_comp_min={n_comp_min}."
        )
        return pd.DataFrame()
    for _iter in range(10):
        if len(active) <= n_comp_min:
            break
        vals_arr = np.asarray(list(active.values()), dtype=np.float64)
        med = float(np.median(vals_arr))
        mad_raw = float(np.median(np.abs(vals_arr - med)))
        if not math.isfinite(mad_raw) or mad_raw <= 0:
            # MAD = 0 znamená že všetky hodnoty sú rovnaké → konvergencia
            break
        mad_sigma = mad_raw / _MAD_CONSISTENCY  # robustný σ estimátor
        threshold = med + rms_outlier_sigma * mad_sigma
        new_active = {cid: rms for cid, rms in active.items() if rms <= threshold}
        if len(new_active) == len(active):
            break  # Konvergencia — žiadne ďalšie výrazy
        if len(new_active) < n_comp_min:
            break  # Neprekroč minimum
        active = new_active

    # ── Krok 5: Finálny výber ──
    # Score: stabilita (RMS) + vzdialenosť + izolácia (contamination)
    # (nižší = lepší kandidát)
    score_map: dict[str, float] = {}
    for cid, rms in active.items():
        row = candidates[candidates[id_col_cand].astype(str).str.strip() == cid]
        if row.empty:
            continue
        dist_deg = float(row.iloc[0].get("_dist_deg", float("nan")))
        # Preferuj optimálnu vzdialenosť (nie "čím bližšie tým lepšie"):
        # príliš blízko → lokálne artefakty (optika/flat/gradients) korelujú s targetom,
        # príliš ďaleko → iná atmosférická bunka.
        # Default optimum: ~150 px at ~9.7"/px ≈ 1455" (~24 arcmin).
        # This helps avoid local/optical artifacts near the target while staying within the same airmass cell.
        optimal_dist_arcsec = 1455.0
        dist_arcsec = float(dist_deg) * 3600.0 if math.isfinite(dist_deg) else float("nan")
        dist_score = (
            abs(dist_arcsec - optimal_dist_arcsec) / optimal_dist_arcsec
            if math.isfinite(dist_arcsec) and optimal_dist_arcsec > 0
            else 1.0
        )
        contamination = float(contamination_map.get(cid, 0.0)) if contamination_map else 0.0
        rms_score = float(rms)
        score_map[cid] = rms_score * 0.5 + dist_score * 0.3 + contamination * 0.2

    scored = sorted(score_map.items(), key=lambda x: float(x[1]))
    # Preferuj Gaia-matched (číselné ID) pred DET_* ak máme dostatok možností.
    scored_non_det = [(cid, sc) for cid, sc in scored if not str(cid).startswith("DET")]
    scored_det = [(cid, sc) for cid, sc in scored if str(cid).startswith("DET")]

    # Dynamický výber 3–7 comp:
    # - prvé 3 vždy
    # - 4.–7. len ak comp_rms <= median(rms prvých 3) × 1.5
    selected_ids: list[str] = []
    sorted_ids = [cid for cid, _ in scored_non_det] + [cid for cid, _ in scored_det]
    for cid in sorted_ids:
        if len(selected_ids) >= 7:
            break
        if len(selected_ids) < 3:
            selected_ids.append(cid)
            continue
        rms_so_far = [float(active.get(x, float("nan"))) for x in selected_ids[:3]]
        rms_so_far = [x for x in rms_so_far if math.isfinite(x) and x > 0]
        if not rms_so_far:
            selected_ids.append(cid)
            continue
        rms_median = float(np.median(np.asarray(rms_so_far, dtype=np.float64)))
        r_c = float(active.get(cid, float("nan")))
        if math.isfinite(r_c) and r_c <= rms_median * 1.5:
            selected_ids.append(cid)

    if len(selected_ids) < n_comp_min:
        logging.warning(
            f"[FÁZA 1] Target {target_cid}: po filtrácii len {len(selected_ids)} "
            f"< n_comp_min={n_comp_min}."
        )
        return pd.DataFrame()

    # Zostav výstupný DataFrame
    result_rows = []
    target_bv = float(target.get("b_v", float("nan")))
    for cid in selected_ids:
        row = candidates[candidates[id_col_cand].astype(str).str.strip() == cid]
        if row.empty:
            continue
        r = row.iloc[0].to_dict()
        r["comp_rms"] = active.get(cid, float("nan"))
        r["comp_score"] = score_map.get(cid, float("nan"))
        r["comp_n_frames"] = len(flux_map.get(cid, []))
        r["target_catalog_id"] = target_cid
        r["target_vsx_name"] = str(target.get("vsx_name", ""))
        r["comp_tier"] = selected_tier
        result_rows.append(r)

    if not result_rows:
        return pd.DataFrame()

    _total_rejected_b = len(_b_rejected) if "_b_rejected" in dir() else 0
    if _total_rejected_b > 0:
        logging.info(
            f"[FÁZA 1] Target {target_cid}: blend filter B celkom vylúčil "
            f"{_total_rejected_b} kandidátov"
        )

    out = pd.DataFrame(result_rows).sort_values("comp_score").reset_index(drop=True)
    if "b_v" in out.columns and math.isfinite(target_bv):
        out_bv = pd.to_numeric(out["b_v"], errors="coerce")
        dbv_out = (out_bv - target_bv).abs()
        bv_info = f"ΔB-V median={float(dbv_out.median()):.3f} max={float(dbv_out.max()):.3f}"
    else:
        bv_info = "ΔB-V N/A"

    logging.info(
        f"[FÁZA 1] Target {target_cid} ({target.get('vsx_name','')}): "
        f"{len(out)} porovnávačiek | RMS min={out['comp_rms'].min():.4f} "
        f"max={out['comp_rms'].max():.4f} | {bv_info}"
    )
    return out


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
    max_dist_deg: float = 1.0,
    max_mag_diff: float = 0.25,
    max_bv_diff: float = 0.15,
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
    progress_cb: Any = None,
) -> dict[str, Any]:
    """Spusti Fázu 0 + Fázu 1 a uloží výstupy.

    Výstupy (uložené do output_dir):
      active_targets.csv              — filtrované VSX ciele
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
        if progress_cb is not None:
            progress_cb(str(msg))

    # ── FÁZA 0 ──
    _p("Fáza 0: výber aktívnych cieľov z VSX…")
    logging.info("[FÁZA 0] Výber aktívnych cieľov...")
    active = select_active_targets(
        variable_targets_csv,
        masterstars_csv,
        frame_w_px=frame_w_px,
        frame_h_px=frame_h_px,
        edge_margin_px=int(chip_interior_margin_px),
        match_radius_arcsec=match_radius_arcsec,
    )
    active_csv = output_dir / "active_targets.csv"
    active.to_csv(active_csv, index=False)
    logging.info(f"[FÁZA 0] Uložené: {active_csv} ({len(active)} cieľov)")
    _p(f"Fáza 0 hotová: {len(active)} aktívnych cieľov")

    if active.empty:
        return {
            "n_active_targets": 0,
            "n_comparison_pairs": 0,
            "active_targets_csv": str(active_csv),
            "comparison_stars_csv": None,
            "suspected_variables_csv": None,
            "targets_without_comps": [],
        }

    # ── Per-frame CSV súbory ──
    csv_paths = sorted(per_frame_csv_dir.rglob("*.csv"))
    # Vylúč výstupné súbory (active_targets, comparison_stars atď.)
    csv_paths = [p for p in csv_paths if p.stem.startswith("proc_") or p.stem.startswith("ali_")]
    _p(f"Fáza 1: načítavam {len(csv_paths)} per-frame CSV do cache…")
    logging.info(f"[FÁZA 1] Načítavam flux z {len(csv_paths)} per-frame CSV súborov...")

    ms_df = pd.read_csv(masterstars_csv, low_memory=False)
    # Normalizuj Gaia ID na string
    for _id_col in ("catalog_id", "name"):
        if _id_col in ms_df.columns:
            ms_df[_id_col] = _normalize_id_series(ms_df[_id_col])

    # ── FÁZA 1 — per target ──
    all_comp_rows: list[pd.DataFrame] = []
    targets_without_comps: list[str] = []

    # Načítaj CSV cache raz pre všetky targety (zabráni 81× opakovaniu I/O).
    _needed_cols = [
        "catalog_id",
        "name",
        "dao_flux",
        "flux",
        "noise_floor_adu",
        "peak_max_adu",
        "saturate_limit_adu_85pct",
        "fwhm_estimate_px",
        "psf_chi2",
        "aperture_r_px",
        "is_usable",
        "is_saturated",
        "is_noisy",
        "snr50_ok",
        "vsx_known_variable",
        "likely_saturated",
        "mag",
        "bjd_tdb_mid",
        "hjd_mid",
        "jd_mid",
        "airmass",
    ]
    if flux_col not in _needed_cols:
        _needed_cols.append(flux_col)
    logging.info("[FÁZA 1] Načítavam CSV cache...")
    _t_cache0 = time.time()
    shared_csv_cache: dict[str, pd.DataFrame] = {}
    for _csv_path in csv_paths:
        try:
            _hdr = pd.read_csv(_csv_path, nrows=0)
            _cols = [c for c in _needed_cols if c in _hdr.columns]
            _df0 = pd.read_csv(_csv_path, usecols=_cols, low_memory=False)
            _name_col = (
                "name"
                if "name" in _df0.columns
                else ("catalog_id" if "catalog_id" in _df0.columns else "name")
            )
            if _name_col in _df0.columns:
                _df0[_name_col] = _normalize_id_series(_df0[_name_col])
            for _num_col in ("flux", "dao_flux", "peak_max_adu", "saturate_limit_adu_85pct", "psf_chi2", "fwhm_estimate_px"):
                if _num_col in _df0.columns:
                    _df0[_num_col] = pd.to_numeric(_df0[_num_col], errors="coerce")
            shared_csv_cache[str(_csv_path)] = _df0
        except Exception:  # noqa: BLE001
            continue
    logging.info(
        f"[FÁZA 1] CSV cache: {len(shared_csv_cache)} súborov "
        f"({time.time() - _t_cache0:.1f}s)"
    )
    _p(f"Fáza 1: cache {len(shared_csv_cache)} súborov — výber porovnávačiek ({len(active)} cieľov)…")

    _vt_chip = pd.read_csv(variable_targets_csv, low_memory=False)
    _fw_chip, _fh_chip = _phase0_effective_frame_hw_px(
        _vt_chip,
        ms_df,
        frame_w_px=int(frame_w_px),
        frame_h_px=int(frame_h_px),
        edge_margin_px=int(chip_interior_margin_px),
    )

    _t_phase1 = time.time()
    n_act0 = int(len(active))
    for idx, (_, target_row) in enumerate(active.iterrows(), start=1):
        if progress_cb is not None and (
            idx == 1 or idx == n_act0 or (n_act0 > 1 and idx % max(1, n_act0 // 12) == 0)
        ):
            _tid = str(target_row.get("vsx_name") or target_row.get("catalog_id", ""))[:48]
            _p(f"Fáza 1: cieľ {idx}/{n_act0}: {_tid}")
        comps = select_comparison_stars_per_target(
            target_row,
            ms_df,
            csv_paths,
            csv_cache=shared_csv_cache,
            fwhm_px=fwhm_px,
            max_dist_deg=max_dist_deg,
            max_mag_diff=max_mag_diff,
            max_bv_diff=max_bv_diff,
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
            chip_fw=_fw_chip,
            chip_fh=_fh_chip,
            chip_interior_margin_px=int(chip_interior_margin_px),
        )
        if comps.empty:
            targets_without_comps.append(str(target_row.get("catalog_id", "")))
        else:
            all_comp_rows.append(comps)

    comp_df = pd.concat(all_comp_rows, ignore_index=True) if all_comp_rows else pd.DataFrame()
    comp_csv = output_dir / "comparison_stars_per_target.csv"
    comp_df.to_csv(comp_csv, index=False)
    logging.info(
        f"[FÁZA 1] Uložené: {comp_csv} "
        f"({len(comp_df)} riadkov, {len(all_comp_rows)} targetov s porovnávačkami)"
    )
    logging.info(f"[FÁZA 1] Čas (comp selection): {time.time() - _t_phase1:.1f}s")

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
    )

    _p(f"Fáza 0+1 hotovo: {int(len(active))} cieľov, {int(len(comp_df))} párov porovnávačiek")
    return {
        "n_active_targets": int(len(active)),
        "n_comparison_pairs": int(len(comp_df)),
        "active_targets_csv": str(active_csv),
        "comparison_stars_csv": str(comp_csv),
        "suspected_variables_csv": str(suspected_csv),
        "targets_without_comps": targets_without_comps,
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

    for csv_path in csv_paths:
        try:
            header = pd.read_csv(csv_path, nrows=0)
            actual_flux = flux_col if flux_col in header.columns else "flux"
            name_c = "catalog_id" if "catalog_id" in header.columns else "name"
            use = [name_c, actual_flux]
            if "mag" in header.columns and "mag" not in use:
                use.append("mag")
            _use_xy = _m > 0 and _fw > 2 * _m and _fh > 2 * _m
            if _use_xy and "x" in header.columns and "y" in header.columns:
                use.extend([c for c in ("x", "y") if c not in use])
            df = pd.read_csv(csv_path, usecols=use, low_memory=False)
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
        except Exception:  # noqa: BLE001
            continue

    # Airmass detrending pre suspected variables
    for cid in list(flux_map.keys()):
        vals = flux_map[cid]
        if len(vals) < 6:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        t = np.linspace(0.0, 1.0, len(arr))
        try:
            coeffs = np.polyfit(t, arr, 2)
            trend_fit = np.polyval(coeffs, t)
            safe_trend = np.where(np.abs(trend_fit) > 1e-9, trend_fit, 1.0)
            detrended = arr / safe_trend
            med_dt = float(np.median(detrended))
            if math.isfinite(med_dt) and med_dt > 0:
                flux_map[cid] = (detrended / med_dt).tolist()
        except Exception:  # noqa: BLE001
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
    out_df.to_csv(output_path, index=False)
    logging.info(
        f"[SUSPECTED] {len(out_df)} kandidátov na nové premenné → {output_path.name} "
        f"(threshold RMS > {threshold:.4f})"
    )


__all__ = [
    # photometry (legacy)
    "StressTestResult",
    "stress_test_relative_rms_from_sidecars",
    "vsx_is_known_variable_top3_per_bin",
    "common_field_intersection_bbox_px",
    "recommended_aperture_by_color",
    "enhance_catalog_dataframe_aperture_bpm",
    "select_active_targets",
    "select_comparison_stars_per_target",
    "run_phase0_and_phase1",
    # photometry_phase2a (legacy)
    "measure_fwhm_from_masterstar",
    "compute_optimal_apertures",
    "read_flux_from_csv",
    "check_comparison_stability",
    "ensemble_normalize",
    "detect_outliers",
    "airmass_detrend_lc",
    "save_lightcurve_csv",
    "save_lightcurve_png",
    "save_cutout_png",
    "save_field_map_png",
    "save_target_field_map_png",
    "run_phase2a",
]

