"""
dilution.py - light of catalogue neighbours inside the target aperture (GS11-2026-10).

For every Gaia neighbour within aperture + 3 FWHM the fraction of its PSF that falls
into the target aperture is computed (circular Gaussian of the frame FWHM; Rice /
non-central chi-square CDF) and its flux is taken in the observed band (Gaia DR3
G + BP-RP -> g/r/i/V/B/R, DR3 documentation 5.5.1). The total is removed from the
target light curve in FLUX, frame by frame:

    m_corr(t) = -2.5 log10( 10^(-0.4 m(t)) - F_contam )

A constant magnitude offset (the former implementation) only shifts the curve and
leaves the amplitude compressed; subtracting the constant blend flux restores it.
D = F_star / (F_star + F_contam) at catalogue brightness is kept as a blend metric
(comparison-star gate, reports).

References:
    Howell (2006) Handbook of CCD Astronomy - aperture contamination
    Gaia DR3 documentation release 1.3, section 5.5.1 - photometric relationships
    Gaia Collaboration (2023) A&A 674, A1 - neighbour catalogue
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from catalog_db import query_local_gaia
from gaia_catalog_id import normalize_gaia_source_id

LOGGER = logging.getLogger(__name__)


def _haversine_arcsec(ra1_deg: float, dec1_deg: float, ra2_deg: float, dec2_deg: float) -> float:
    """Great-circle separation in arcseconds (scalar haversine)."""
    ra1, dec1, ra2, dec2 = map(math.radians, (ra1_deg, dec1_deg, ra2_deg, dec2_deg))
    dra = ra2 - ra1
    ddec = dec2 - dec1
    a = math.sin(ddec / 2) ** 2 + math.cos(dec1) * math.cos(dec2) * math.sin(dra / 2) ** 2
    return math.degrees(2 * math.asin(min(1.0, math.sqrt(max(0.0, a))))) * 3600.0


def _normalize_exclude_source_id(catalog_id: int | str | None) -> int | None:
    if catalog_id is None:
        return None
    s = normalize_gaia_source_id(catalog_id)
    if not s or not s.isdigit():
        try:
            return int(catalog_id)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
    try:
        return int(s)
    except (TypeError, ValueError, OverflowError):
        return None


def query_gaia_neighbors(
    ra_deg: float,
    dec_deg: float,
    radius_arcsec: float,
    gaia_db_path: str,
    *,
    mag_limit: float = 99.0,
    exclude_source_id: int | None = None,
) -> list[dict[str, Any]]:
    """
    Return Gaia DR3 stars within radius_arcsec of (ra_deg, dec_deg).

    Uses query_local_gaia() bounding box + haversine post-filter.
    Excludes the star itself via exclude_source_id (catalog_id).

    Returns list of dicts with keys:
        source_id, ra, dec, g_mag, bp_mag, rp_mag, sep_arcsec
    Sorted by sep_arcsec ascending.
    Returns [] if DB unreachable or no neighbors found.
    """
    try:
        ra = float(ra_deg)
        dec = float(dec_deg)
        r_arcsec = float(radius_arcsec)
    except (TypeError, ValueError):
        return []

    if not (math.isfinite(ra) and math.isfinite(dec) and math.isfinite(r_arcsec)):
        return []
    if r_arcsec <= 0:
        return []

    pad_deg = r_arcsec / 3600.0
    pad_ra_deg = pad_deg / max(1e-6, math.cos(math.radians(dec)))
    db_path = Path(str(gaia_db_path)).expanduser()
    if not db_path.is_file():
        LOGGER.warning("[DILUTION] Gaia DB not found: %s", db_path)
        return []

    try:
        rows = query_local_gaia(
            db_path,
            ra_min=ra - pad_ra_deg,
            ra_max=ra + pad_ra_deg,
            dec_min=dec - pad_deg,
            dec_max=dec + pad_deg,
            mag_limit=float(mag_limit),
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[DILUTION] query_local_gaia failed: %s", exc)
        return []

    exclude_id = exclude_source_id
    out: list[dict[str, Any]] = []
    for row in rows:
        try:
            sid_raw = row.get("source_id")
            sid_int = _normalize_exclude_source_id(sid_raw)
            if exclude_id is not None and sid_int is not None and sid_int == int(exclude_id):
                continue
            ra_n = float(row["ra"])
            dec_n = float(row["dec"])
            g_mag = float(row["g_mag"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (math.isfinite(ra_n) and math.isfinite(dec_n) and math.isfinite(g_mag)):
            continue
        sep = _haversine_arcsec(ra, dec, ra_n, dec_n)
        if sep > r_arcsec:
            continue
        out.append(
            {
                "source_id": sid_int if sid_int is not None else sid_raw,
                "ra": ra_n,
                "dec": dec_n,
                "g_mag": g_mag,
                "bp_mag": row.get("bp_mag"),
                "rp_mag": row.get("rp_mag"),
                "bp_rp": row.get("bp_rp"),
                "var_flag": row.get("var_flag") or row.get("phot_variable_flag"),
                "sep_arcsec": float(sep),
            }
        )

    out.sort(key=lambda d: float(d.get("sep_arcsec", float("inf"))))
    return out


def flux_from_gmag(g_mag: float) -> float:
    """
    Convert Gaia G magnitude to relative linear flux.
    f = 10^(-g_mag / 2.5)
    Used only for flux ratios - absolute zero point cancels out.
    """
    return float(10.0 ** (-float(g_mag) / 2.5))


def _no_blend_result(
    *,
    aperture_arcsec: float,
    search_radius_arcsec: float,
) -> dict[str, Any]:
    return {
        "dilution_factor": 1.0,
        "dilution_delta_mag": 0.0,
        "n_neighbors": 0,
        "neighbor_flux_sum": 0.0,
        "aperture_arcsec": float(aperture_arcsec),
        "search_radius_arcsec": float(search_radius_arcsec),
    }


def enclosed_fraction(sep: float, r_ap: float, sigma: float) -> float:
    """Fraction of a circular Gaussian PSF (sigma) centred ``sep`` away that falls in a
    circular aperture of radius ``r_ap`` (same units).

    |X| for X ~ N(mu, sigma^2 I_2), |mu| = sep, is Rice distributed, so the fraction is the
    non-central chi-square CDF with 2 dof: P((r/sigma)^2; k=2, lambda=(sep/sigma)^2).
    """
    from scipy.stats import ncx2  # noqa: PLC0415

    if not (sigma > 0 and r_ap > 0):
        return 0.0
    return float(ncx2.cdf((r_ap / sigma) ** 2, 2, (sep / sigma) ** 2))


def compute_dilution_factor(
    ra_deg: float,
    dec_deg: float,
    g_mag: float,
    aperture_arcsec: float,
    gaia_db_path: str,
    *,
    catalog_id: int | None = None,
    search_radius_arcsec: float | None = None,
    mag_limit_delta: float = 5.0,
    fwhm_arcsec: float | None = None,
    band: str = "G",
    target_bp_rp: float | None = None,
) -> dict[str, Any]:
    """Light of Gaia neighbours inside the target aperture (GS11-2026-10).

    Each neighbour contributes F_nb * f_enc(sep), with f_enc the enclosed fraction of a
    Gaussian PSF of the frame FWHM centred at the neighbour (:func:`enclosed_fraction`).
    Fluxes are taken in the observed ``band`` (Gaia G + BP-RP -> g/r/i/V/B/R via
    :mod:`gaia_bands`); a neighbour without a valid colour falls back to G (flagged).
    The search radius is aperture + 3 FWHM (the PSF wing beyond that is < 1e-5).

    Returns ``contam_flux_total`` - neighbour light in the aperture expressed as a
    total-flux equivalent on the catalogue scale (divided by the target's own enclosed
    fraction, because the ensemble zero point puts aperture fluxes on total magnitudes) -
    which :func:`apply_target_dilution_to_mag_calib` subtracts in flux, frame by frame.
    ``dilution_factor`` D = F_target / (F_target + contamination) at catalogue brightness.
    """
    from gaia_bands import band_mag_from_gaia  # noqa: PLC0415

    try:
        ap_arcsec = float(aperture_arcsec)
    except (TypeError, ValueError):
        ap_arcsec = float("nan")
    try:
        fw = float(fwhm_arcsec) if fwhm_arcsec is not None else float("nan")
    except (TypeError, ValueError):
        fw = float("nan")
    if not (math.isfinite(fw) and fw > 0):
        fw = ap_arcsec / 1.35 if math.isfinite(ap_arcsec) and ap_arcsec > 0 else float("nan")
    if search_radius_arcsec is not None:
        r_search = float(search_radius_arcsec)
    else:
        r_search = ap_arcsec + 3.0 * fw if math.isfinite(ap_arcsec) and math.isfinite(fw) else 0.0
    base = _no_blend_result(aperture_arcsec=ap_arcsec, search_radius_arcsec=r_search)
    try:
        g = float(g_mag)
    except (TypeError, ValueError):
        return base
    if not (math.isfinite(g) and math.isfinite(ap_arcsec) and ap_arcsec > 0 and math.isfinite(fw)):
        return base
    sigma = fw / (2.0 * math.sqrt(2.0 * math.log(2.0)))

    m_t, _ = band_mag_from_gaia(g, target_bp_rp if target_bp_rp is not None else float("nan"), band)
    target_band_ok = math.isfinite(m_t)
    if not target_band_ok:
        m_t = g
    f_t = flux_from_gmag(m_t)
    f0 = enclosed_fraction(0.0, ap_arcsec, sigma)

    neighbors = query_gaia_neighbors(
        float(ra_deg),
        float(dec_deg),
        r_search,
        gaia_db_path,
        mag_limit=g + float(mag_limit_delta),
        exclude_source_id=_normalize_exclude_source_id(catalog_id),
    )
    contam_ap = 0.0
    n_used = 0
    n_g_fallback = 0
    variable_ids: list[str] = []
    for n in neighbors:
        try:
            gm = float(n["g_mag"])
            sep = float(n["sep_arcsec"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (math.isfinite(gm) and math.isfinite(sep)):
            continue
        m_n, _ = band_mag_from_gaia(gm, n.get("bp_rp") if n.get("bp_rp") is not None else float("nan"), band)
        if not math.isfinite(m_n):
            m_n = gm
            n_g_fallback += 1 if band != "G" else 0
        frac = enclosed_fraction(sep, ap_arcsec, sigma)
        if frac <= 1e-6:
            continue
        contam_ap += flux_from_gmag(m_n) * frac
        n_used += 1
        if str(n.get("var_flag") or "").strip().upper() == "VARIABLE":
            variable_ids.append(str(n.get("source_id")))
    if contam_ap <= 0 or f0 <= 0:
        return {**base, "n_neighbors": 0}
    contam_total = contam_ap / f0
    d = float(f_t / (f_t + contam_total))
    return {
        "dilution_factor": d,
        "dilution_delta_mag": float(-2.5 * math.log10(d)) if d < 1.0 else 0.0,
        "n_neighbors": int(n_used),
        "neighbor_flux_sum": float(contam_total / f_t),
        "contam_flux_total": float(contam_total),
        "contam_mag_total": float(-2.5 * math.log10(contam_total)),
        "band": str(band),
        "target_band_mag_ok": bool(target_band_ok),
        "n_neighbors_g_fallback": int(n_g_fallback),
        "variable_neighbors": variable_ids,
        "aperture_arcsec": float(ap_arcsec),
        "fwhm_arcsec": float(fw),
        "search_radius_arcsec": float(r_search),
    }


def apply_target_dilution_to_mag_calib(
    mag_calib: np.ndarray,
    dilution_result: dict[str, Any],
    cfg: Any,
    *,
    target_cid: str = "",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Remove neighbour light from the target light curve in flux (GS11-2026-10).

    m_corr(t) = -2.5 log10(10^(-0.4 m(t)) - F_contam). Blend light is a constant flux, so
    the correction is larger at minimum light and restores the true amplitude (a constant
    magnitude offset would only shift the curve and keep the amplitude compressed).
    Skipped and flagged when D < ``gs11_target_min_dilution`` (contamination too large to
    trust) or a contaminating neighbour is a known variable; frames where the measured
    flux does not exceed the contamination become NaN.
    """
    out = np.asarray(mag_calib, dtype=np.float64).copy()
    res = dict(dilution_result)
    if not bool(getattr(cfg, "gs11_dilution_enabled", False)):
        return out, res
    try:
        d = float(res.get("dilution_factor", 1.0))
        contam = float(res.get("contam_flux_total", 0.0))
    except (TypeError, ValueError):
        return out, res
    if not (math.isfinite(d) and math.isfinite(contam)) or d >= 1.0 or contam <= 0:
        return out, res
    min_d = float(getattr(cfg, "gs11_target_min_dilution", 0.50) or 0.50)
    if d < min_d:
        res.update({"dilution_skipped": True, "dilution_skip_reason": f"D={d:.3f} < {min_d:.2f}"})
        LOGGER.warning("GS11: %s D=%.4f too low - correction skipped, flagged", target_cid or "?", d)
        return out, res
    if res.get("variable_neighbors"):
        res.update({"dilution_skipped": True, "dilution_skip_reason": "variable neighbour in aperture"})
        LOGGER.warning("GS11: %s contaminating neighbour is variable %s - skipped",
                       target_cid or "?", res.get("variable_neighbors"))
        return out, res
    f = np.power(10.0, -0.4 * out)
    rest = f - contam
    good = np.isfinite(rest) & (rest > 0)
    corr = np.full_like(out, np.nan)
    corr[good] = -2.5 * np.log10(rest[good])
    n_bad = int(np.count_nonzero(np.isfinite(out) & ~good))
    dm = corr - out
    res.update({
        "dilution_applied": True,
        "dilution_delta_mag": float(np.nanmedian(dm)) if np.isfinite(dm).any() else 0.0,
        "dilution_delta_mag_min": float(np.nanmin(dm)) if np.isfinite(dm).any() else 0.0,
        "dilution_delta_mag_max": float(np.nanmax(dm)) if np.isfinite(dm).any() else 0.0,
        "dilution_frames_nan": n_bad,
    })
    LOGGER.info(
        "GS11 dilution (flux): %s D=%.4f contam=%.3f mag, correction %.1f..%.1f mmag (%d neighbours)",
        target_cid or "?", d, float(res.get("contam_mag_total", float("nan"))),
        1000.0 * res["dilution_delta_mag_min"], 1000.0 * res["dilution_delta_mag_max"],
        int(res.get("n_neighbors", 0)),
    )
    return corr, res


def gs11_dilution_from_summary(summary_row: Any, cfg: Any) -> tuple[float, float] | None:
    """(D, delta_mag) of a target when GS11 is enabled, was applied and the blend is significant
    (D < 0.99); else None. ``delta_mag`` is the correction applied to the magnitudes."""
    if cfg is None or not bool(getattr(cfg, "gs11_dilution_enabled", False)) or summary_row is None:
        return None
    try:
        get = summary_row.get
        if bool(get("gs11_dilution_skipped", False)) and str(get("gs11_dilution_skipped", "")).lower() not in ("false", "0", ""):
            return None
        d = float(get("dilution_factor", float("nan")))
        dm = float(get("dilution_delta_mag", float("nan")))
    except (TypeError, ValueError, AttributeError):
        return None
    if not math.isfinite(d) or d >= 0.99 or d <= 0.0:
        return None
    return d, (dm if math.isfinite(dm) else float("nan"))

