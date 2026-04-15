"""Fáza 2A — Aperturná fotometria premenných hviezd.

Postup:
1. Globálna fixná apertura = aperture_fwhm_factor × Gaussian FWHM (default 1.75×)
2. Aperturná fotometria per snímka (FITS z detrended_aligned)
   - x, y, bjd/hjd/jd z per-frame CSV (single source of truth)
   - mediánový sky z CircularAnnulus (nie priemer)
3. Stability check porovnávačiek (Abbeho p2p scatter + MAD)
4. Ensemble normalizácia (len good comp)
5. Outlier detekcia (saturated / spike / dip)
6. Výstup: lightcurve CSV + PNG + cutout + field_map + summary
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits as astrofits

from config import AppConfig

_MAD_CONSISTENCY = 0.6745  # normalizačný faktor MAD → σ ekvivalent


# ---------------------------------------------------------------------------
# Pomocné funkcie
# ---------------------------------------------------------------------------


def _normalize_gaia_id(x: Any) -> str:
    """Normalizuj Gaia ID na string integer."""
    s = str(x).strip()
    if not s or s.lower() in ("nan", "none", ""):
        return ""
    try:
        return str(int(float(s)))
    except (ValueError, TypeError):
        return s


def _build_csv_lookup(
    csv_df: pd.DataFrame,
    id_col: str,
) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    """Vytvorí dva lookup mechanizmy:
    1. Primárny: dict {normalized_id → row}
    2. Záložný: riadky s numerickými x,y pre nearest-neighbor match (plné stĺpce CSV).
    """
    id_map: dict[str, pd.Series] = {}
    for _, row in csv_df.iterrows():
        cid = _normalize_gaia_id(row.get(id_col, ""))
        if cid:
            id_map[cid] = row
    xy_df = csv_df.copy()
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
    xy_tol_px: float = 50.0,
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
    j = int(np.argmin(dists))
    if float(dists[j]) <= float(xy_tol_px):
        _hit = xy_df.iloc[j]
        _mid = str(_hit.get("_cid_norm", ""))
        logging.debug(
            "[FÁZA 2A] CSV NN fallback ok: requested_cid=%s matched_csv_id=%s dist_px=%.2f tol=%.1f",
            cid,
            _mid,
            float(dists[j]),
            float(xy_tol_px),
        )
        return _hit
    return None


def _sat_limit_peak_adu(cfg: AppConfig | None = None) -> float:
    """Hranica peak_max_adu pre saturáciu — zladená s pipeline (fallback × saturate_limit_fraction)."""
    c = cfg or AppConfig()
    fb = c.photometry_fallback_saturate_adu
    frac = max(0.0, min(1.0, float(c.saturate_limit_fraction)))
    if fb is not None and math.isfinite(float(fb)) and float(fb) > 0:
        return float(fb) * frac
    return 65535.0 * frac


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
# KROK 1: Globálna fixná apertura z Gaussian FWHM na MASTERSTAR
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
    """Globálna fixná apertura = aperture_fwhm_factor × Gaussian FWHM.

    Fyzikálne zdôvodnenie:
    - Gaussian PSF: r = 1.75× FWHM zachytí ~98% flux
    - Konzistentná fixná apertura je robustnejšia ako per-hviezda
      metódy v hustom poli (kontaminácia susedmi)
    - Zodpovedá AIJ metodike: fixná apertura z FWHM

    Args:
        masterstar_fits_path: Nepoužíva sa — zachované pre kompatibilitu.
        star_positions: DataFrame so stĺpcami catalog_id (voliteľne name).
        fwhm_px: Gaussian FWHM v pixeloch (z measure_fwhm_from_masterstar).
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
        f"({aperture_fwhm_factor:.2f}× Gaussian FWHM={fwhm_px:.3f}px)"
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


def _measure_flux_median_sky(
    data: np.ndarray,
    xc: float,
    yc: float,
    r_ap: float,
    r_in: float,
    r_out: float,
    *,
    neighbor_positions: list[tuple[float, float]] | None = None,
    neighbor_r_excl: float | None = None,
) -> tuple[float, float, float]:
    """Zmeraj sky-subtracted flux s mediánovým sky.

    Susedné hviezdy v sky annuluse sú maskované (ako AIJ 'Remove stars from background').

    Args:
        neighbor_positions: Zoznam (x, y) susedných hviezd ktoré treba maskovať v sky.
        neighbor_r_excl: Polomer maskovania susedov (px). Default = r_ap.

    Returns:
        (flux_sky_sub, sky_pp, aperture_area)
    """
    from photutils.aperture import CircularAnnulus, CircularAperture, aperture_photometry

    h, w = data.shape
    if not (r_out < xc < w - r_out and r_out < yc < h - r_out):
        return float("nan"), float("nan"), float("nan")

    pos = np.array([[xc, yc]])
    ap = CircularAperture(pos, r=r_ap)
    ann = CircularAnnulus(pos, r_in=r_in, r_out=r_out)

    # Sky pixely z annulusu (vždy v súradniciach plného rastra — multiply() môže vrátiť len výrez)
    ann_mask = _aperture_to_mask_single(ann)
    ann_on = ann_mask.to_image(data.shape) > 0
    sky_pixels = data[ann_on].copy()

    # Maskuj susedné hviezdy v sky annuluse
    if neighbor_positions and sky_pixels.size > 0:
        _excl_r = neighbor_r_excl if neighbor_r_excl is not None else r_ap
        yy, xx = np.mgrid[0:h, 0:w]
        neighbor_full = np.zeros((h, w), dtype=bool)
        for nx_pos, ny_pos in neighbor_positions:
            neighbor_full |= (xx - nx_pos) ** 2 + (yy - ny_pos) ** 2 <= _excl_r**2
        valid = ann_on & ~neighbor_full
        sky_pixels_masked = data[valid]
        if sky_pixels_masked.size >= 5:
            sky_pixels = sky_pixels_masked

    if sky_pixels.size < 5:
        return float("nan"), float("nan"), float("nan")

    sky_pp = float(np.median(sky_pixels))

    phot = aperture_photometry(data, ap)
    raw_flux = float(phot["aperture_sum"][0])
    flux_sub = raw_flux - sky_pp * float(ap.area)

    return flux_sub, sky_pp, float(ap.area)


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
    xy_tol_px: float = 50.0,
    frame_times: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Krok 2: Načítaj flux z per-frame CSV (dao_flux).

    Namiesto čítania FITS a vlastnej aperturnej fotometrie používa
    dao_flux ktorý pipeline vypočítala počas DAO detekcie.
    dao_flux je sky-subtrahovaný flux zmeraný s aperture_r_px z CSV.

    Returns:
        DataFrame: catalog_id, bjd, hjd, jd, airmass, mag_inst, err,
                   aperture_r_px, sky_pp, flag, source_file
    """
    try:
        csv_df = pd.read_csv(frame_csv_path, low_memory=False)
    except Exception as exc:
        logging.warning(f"[FÁZA 2A] Nemôžem čítať CSV {frame_csv_path}: {exc}")
        return pd.DataFrame()

    _sat_lim = float(sat_limit_adu) if sat_limit_adu is not None else _sat_limit_peak_adu()
    source_file = frame_csv_path.name

    # Normalizuj catalog_id
    id_col = "catalog_id" if "catalog_id" in csv_df.columns else "name"
    csv_df["_cid"] = csv_df[id_col].apply(_normalize_gaia_id)
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
        is_sat = math.isfinite(peak) and peak > _sat_lim

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

    mag_calib = mag_inst(target) - weighted_median(mag_inst(good_comp))
              + median(catalog_mag(good_comp))
    delta_mag  = mag_inst(target) - weighted_median(mag_inst(good_comp))

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

    good_all = [cid for cid, q in comp_quality.items() if q["quality"] == "good"]
    good_sorted = sorted(
        good_all,
        key=lambda c: float(comp_rms_map.get(c, float("inf"))),
    )

    selected: list[str] = []
    for cid in good_sorted:
        if len(selected) >= n_comp_max:
            break
        p2p = float(comp_quality[cid].get("rms_p2p", float("nan")))
        if len(selected) < n_comp_min:
            selected.append(cid)
        elif math.isfinite(p2p_thr) and math.isfinite(p2p) and p2p < p2p_thr:
            selected.append(cid)
        elif not math.isfinite(p2p_thr):
            selected.append(cid)

    if len(selected) < n_comp_min:
        suspect_sorted = sorted(
            [cid for cid, q in comp_quality.items() if q["quality"] == "suspect"],
            key=lambda c: float(comp_rms_map.get(c, float("inf"))),
        )
        for cid in suspect_sorted:
            if len(selected) >= n_comp_min:
                break
            if cid not in selected:
                selected.append(cid)
            if len(selected) >= n_comp_max:
                break

    good_ids = selected[:n_comp_max]
    if not good_ids:
        return mag_calib, delta_mag, ensemble_scatter

    cat_mags = np.asarray([comp_catalog_mag.get(cid, float("nan")) for cid in good_ids])
    cat_offset = float(np.nanmedian(cat_mags))
    # cat_offset = medián katalógových mag porovnávačiek
    # Toto je referenčná úroveň pre mag_calib — ensemble normalizácia
    # predpokladá že target aj comp majú rovnaký aperture fraction (EE)
    # Pre jasné hviezdy s veľkým PSF môže byť systematický offset
    # → riešenie je v apertúre na MASTERSTAR (faktor × FWHM), nie tu
    logging.debug(
        f"[FÁZA 2A] Ensemble: {len(good_ids)} good comp, "
        f"catalog_mag median={cat_offset:.3f}"
    )

    for i in range(n_frames):
        comp_vals = np.asarray(
            [
                comp_mag_inst[cid][i]
                for cid in good_ids
                if cid in comp_mag_inst and math.isfinite(comp_mag_inst[cid][i])
            ],
            dtype=np.float64,
        )
        if comp_vals.size == 0 or not math.isfinite(target_mag_inst[i]):
            continue
        # Flux-weighted ensemble: comp_vals sú mag_inst → konvertuj na relatívny flux,
        # spriemeruj a späť na mag. Jasnejšie comp majú väčšiu váhu (AIJ-like).
        comp_fluxes = np.asarray(
            [10 ** (-0.4 * m) for m in comp_vals if math.isfinite(m)],
            dtype=np.float64,
        )
        if comp_fluxes.size > 0:
            ens_flux_sum = float(np.sum(comp_fluxes))
            ens_med = float(-2.5 * math.log10(ens_flux_sum / comp_fluxes.size))
        else:
            ens_med = float(np.median(comp_vals))
        ensemble_scatter[i] = float(np.std(comp_vals)) if comp_vals.size > 1 else 0.0
        delta_mag[i] = target_mag_inst[i] - ens_med
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

    # Svetelná krivka
    for flag, color in flag_colors.items():
        mask = np.array([f == flag for f in flags])
        if not mask.any():
            continue
        bjd_f = bjd[mask]
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

    ax_lc.set_xlabel("BJD (TDB)", fontsize=9)
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
    sat_limit_adu: float | None = None,
    outlier_sigma: float = 3.0,
    stability_sigma: float = 3.0,
    force_aperture_px: float | None = None,
    cfg: AppConfig | None = None,
) -> dict[str, Any]:
    """Hlavný wrapper pre Fázu 2A.

    FWHM pre globálnu fixnú apertúru (faktor × FWHM) sa meria 2D Gaussian fitom na MASTERSTAR
    (``measure_fwhm_from_masterstar``). Parameter ``fwhm_px`` sa použije len ako záložný
    odhad veľkosti fit okna, ak v hlavičke MASTERSTAR chýba alebo je neplatný ``VY_FWHM``.

    Returns:
        dict: n_targets, n_frames, n_lightcurves, summary_csv, field_map_png
    """
    _ = detrended_aligned_dir  # FITS sa v Fáze 2A nepotrebujú — flux z dao_flux v CSV
    output_dir = Path(output_dir)
    lc_dir = output_dir / "lightcurves"
    lc_dir.mkdir(parents=True, exist_ok=True)
    _cfg = cfg or AppConfig()
    _save_png = bool(_cfg.save_lightcurve_png)

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

    # Čas + airmass z prvého platného riadku každého per-frame CSV (podľa stem FITS)
    frame_time_lookup: dict[str, dict[str, float]] = {}
    for csv_path in csv_files:
        stem = csv_path.stem
        try:
            _csv_tmp = pd.read_csv(csv_path, low_memory=False)
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
        except Exception:
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

    _fallback_hint = float(fwhm_px) if math.isfinite(fwhm_px) and fwhm_px > 0 else 3.5
    _dao_hint = _fallback_hint
    try:
        with astrofits.open(Path(masterstar_fits_path), memmap=False) as _hdul:
            hdr = _hdul[0].header
            if "VY_FWHM" in hdr:
                v = float(hdr["VY_FWHM"])
                if not (math.isfinite(v) and v > 0):
                    _dao_hint = _fallback_hint
                else:
                    _dao_hint = v
    except Exception:
        _dao_hint = _fallback_hint

    fwhm_px = measure_fwhm_from_masterstar(
        Path(masterstar_fits_path),
        all_stars,
        dao_fwhm_hint=_dao_hint,
    )

    if force_aperture_px is not None and force_aperture_px > 0:
        # Fixná apertura pre všetky hviezdy — debug/kalibrácia
        apertures_px = {
            str(row.get("catalog_id", "")): float(force_aperture_px)
            for _, row in all_stars.iterrows()
        }
        logging.info(
            f"[FÁZA 2A] FORCE apertura: {force_aperture_px:.2f}px pre všetky hviezdy"
        )
    else:
        apertures_px = compute_optimal_apertures(
            Path(masterstar_fits_path),
            all_stars,
            fwhm_px,
            aperture_fwhm_factor=2.0,
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

    sat_limit_resolved = float(sat_limit_adu) if sat_limit_adu is not None else _sat_limit_peak_adu()

    # Field map PNG (raz pre celé pole)
    field_map_path = output_dir / "field_map.png"
    if _save_png:
        save_field_map_png(field_map_path, Path(masterstar_fits_path), at_df, comp_df)

    summary_rows: list[dict[str, Any]] = []
    n_lc = 0

    # Per target loop
    for _, target_row in at_df.iterrows():
        target_cid = _normalize_gaia_id(target_row.get("catalog_id", ""))
        target_name = str(target_row.get("vsx_name", target_cid))

        # Comp hviezdy pre tento target
        if "target_catalog_id" in comp_df.columns:
            tc = comp_df["target_catalog_id"].apply(_normalize_gaia_id)
        else:
            tc = pd.Series([""] * len(comp_df), index=comp_df.index)
        target_comps = comp_df[tc == target_cid].copy()

        if target_comps.empty:
            logging.warning(f"[FÁZA 2A] Target {target_name}: žiadne comp hviezdy")
            continue

        comp_ids = [str(c) for c in target_comps["catalog_id"].tolist() if str(c).strip()]
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

            df_frame = read_flux_from_csv(
                csv_path,
                all_ids,
                apertures_px,
                sat_limit_adu=sat_limit_resolved,
                star_xy=star_xy,
                xy_tol_px=50.0,
                frame_times=_ft,
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
        am_detrended, am_slope, _ = airmass_detrend_lc(mag_calib, airmass_arr, base_flags)
        mag_calib_raw = mag_calib.copy()
        mag_calib = am_detrended

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

        # Per-target field map s číslovanými comp hviezdami
        if _save_png:
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
        n_good_comp = sum(1 for q in comp_quality.values() if q["quality"] == "good")
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
        logging.info(
            f"[FÁZA 2A] {target_name}: {len(bjd)} snímok, "
            f"{n_good_comp} good comp, lc_rms={summary_rows[-1]['lc_rms']:.4f}"
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

    return {
        "n_targets": len(at_df),
        "n_frames": n_frames,
        "n_lightcurves": n_lc,
        "summary_csv": str(summary_csv),
        "field_map_png": str(field_map_path),
        "output_dir": str(output_dir),
    }


__all__ = [
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
