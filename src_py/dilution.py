"""
dilution.py - Flux dilution factor computation for VYVAR (TODO-GS11).

Computes per-star dilution factor D = F_star / (F_star + SigmaF_neighbors)
from Gaia DR3 catalog neighbors within the photometric aperture.

D = 1.0  -> no blend (isolated star)
D < 1.0  -> blended; neighbor flux dilutes the target -> observed mag is too bright
           (too small numerically). Correction adds a positive offset:
           delta_mag = -2.5 * log10(D)  (> 0 when D < 1)
           mag_corrected = mag_observed + delta_mag

References:
    Seager & Mallen-Ornelas (2003) ApJ 585, 1038  - dilution factor definition
    Ciardi et al. (2015) ApJ 805, 16              - blend correction formula
    Howell (2006) Handbook of CCD Astronomy       - aperture contamination
    Gaia Collaboration (2023) A&A 674, A1         - neighbor catalog
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from database import query_local_gaia
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
    db_path = Path(str(gaia_db_path)).expanduser()
    if not db_path.is_file():
        LOGGER.warning("[DILUTION] Gaia DB not found: %s", db_path)
        return []

    try:
        rows = query_local_gaia(
            db_path,
            ra_min=ra - pad_deg,
            ra_max=ra + pad_deg,
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
) -> dict[str, Any]:
    """
    Compute flux dilution factor for a star given its aperture size.

    Parameters:
        ra_deg, dec_deg     : star position
        g_mag               : Gaia G magnitude of the star
        aperture_arcsec     : photometric aperture radius in arcsec
        gaia_db_path        : path to local Gaia SQLite DB
        catalog_id          : Gaia source_id to exclude (the star itself)
        search_radius_arcsec: neighbor search radius; default = aperture_arcsec
        mag_limit_delta     : only include neighbors fainter by at most this
                              (default 5.0 mag -> neighbors contribute > 1% flux)

    Returns dict:
        {
            "dilution_factor":    float,   # D = F_star / (F_star + SigmaF_neighbors); 1.0 = no blend
            "dilution_delta_mag": float,   # -2.5 * log10(D); 0.0 = no blend
            "n_neighbors":        int,     # neighbors found within aperture
            "neighbor_flux_sum":  float,   # SigmaF_neighbors / F_star (relative)
            "aperture_arcsec":    float,   # input aperture
            "search_radius_arcsec": float, # actual search radius used
        }
    """
    try:
        ap_arcsec = float(aperture_arcsec)
    except (TypeError, ValueError):
        ap_arcsec = float("nan")

    r_search = float(search_radius_arcsec) if search_radius_arcsec is not None else ap_arcsec
    if not math.isfinite(r_search) or r_search <= 0:
        r_search = ap_arcsec if math.isfinite(ap_arcsec) and ap_arcsec > 0 else 0.0

    base = _no_blend_result(aperture_arcsec=ap_arcsec, search_radius_arcsec=r_search)

    try:
        g = float(g_mag)
    except (TypeError, ValueError):
        return base

    if not math.isfinite(g):
        return base

    f_star = flux_from_gmag(g)
    if not (math.isfinite(f_star) and f_star > 0):
        return base

    neighbor_mag_limit = g + float(mag_limit_delta)
    exclude_id = _normalize_exclude_source_id(catalog_id)

    neighbors = query_gaia_neighbors(
        float(ra_deg),
        float(dec_deg),
        r_search,
        gaia_db_path,
        mag_limit=neighbor_mag_limit,
        exclude_source_id=exclude_id,
    )

    f_neighbors = 0.0
    for n in neighbors:
        try:
            gm = float(n["g_mag"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(gm):
            f_neighbors += flux_from_gmag(gm)

    neighbor_flux_ratio = float(f_neighbors / f_star) if f_star > 0 else 0.0

    if f_neighbors <= 0:
        return {
            **base,
            "n_neighbors": 0,
            "neighbor_flux_sum": 0.0,
        }

    d = float(f_star / (f_star + f_neighbors))
    delta_mag = float(-2.5 * math.log10(d)) if d < 1.0 else 0.0

    return {
        "dilution_factor": d,
        "dilution_delta_mag": delta_mag,
        "n_neighbors": int(len(neighbors)),
        "neighbor_flux_sum": neighbor_flux_ratio,
        "aperture_arcsec": float(ap_arcsec),
        "search_radius_arcsec": float(r_search),
    }


def _star_g_mag(star: dict[str, Any]) -> float:
    for key in ("g_mag", "phot_g_mean_mag", "mag"):
        if key in star:
            try:
                v = float(star[key])
            except (TypeError, ValueError):
                continue
            if math.isfinite(v):
                return v
    return float("nan")


def compute_dilution_batch(
    stars: list[dict[str, Any]],
    aperture_arcsec: float,
    gaia_db_path: str,
    *,
    mag_limit_delta: float = 5.0,
) -> list[dict[str, Any]]:
    """
    Compute dilution for a list of stars (targets or comps).

    Each star dict must have: ra_deg, dec_deg, g_mag (or phot_g_mean_mag),
    and optionally catalog_id.

    Returns same list with dilution fields added to each dict.
    Logs progress at INFO level every 50 stars.
    """
    out: list[dict[str, Any]] = []
    n_total = len(stars)
    for i, star in enumerate(stars):
        row = dict(star)
        if (i + 1) % 50 == 0 or i == 0:
            LOGGER.info("[DILUTION] batch progress %d/%d", i + 1, n_total)
        try:
            g = _star_g_mag(row)
            cid = row.get("catalog_id")
            if cid is not None:
                cid = _normalize_exclude_source_id(cid)
            dil = compute_dilution_factor(
                float(row["ra_deg"]),
                float(row["dec_deg"]),
                g,
                float(aperture_arcsec),
                gaia_db_path,
                catalog_id=cid,
                mag_limit_delta=float(mag_limit_delta),
            )
            row.update(dil)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "[DILUTION] star %s failed: %s",
                row.get("catalog_id", "?"),
                exc,
            )
            row.update(
                _no_blend_result(
                    aperture_arcsec=float(aperture_arcsec),
                    search_radius_arcsec=float(aperture_arcsec),
                )
            )
        out.append(row)
    return out


def apply_target_dilution_to_mag_calib(
    mag_calib: np.ndarray,
    dilution_result: dict[str, Any],
    cfg: Any,
    *,
    target_cid: str = "",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply GS11 dilution correction to per-frame mag_calib (Phase 2A, post-ensemble).

    Returns (corrected_mag_calib, dilution_result) - result dict is unchanged when skipped.
    """
    out = np.asarray(mag_calib, dtype=np.float64).copy()
    res = dict(dilution_result)
    if not bool(getattr(cfg, "gs11_dilution_enabled", False)):
        return out, res
    try:
        d = float(res.get("dilution_factor", 1.0))
    except (TypeError, ValueError):
        d = 1.0
    try:
        delta = float(res.get("dilution_delta_mag", 0.0))
    except (TypeError, ValueError):
        delta = 0.0
    min_d = float(getattr(cfg, "gs11_target_min_dilution", 0.50) or 0.50)
    if d >= min_d and d < 1.0 and math.isfinite(delta) and delta != 0.0:
        # Blend makes star too bright -> mag too small -> add positive delta_mag.
        out = out + float(delta)
        LOGGER.info(
            "GS11 dilution correction: %s D=%.4f Deltam=%.1f mmag (%s neighbors)",
            target_cid or "?",
            d,
            delta * 1000.0,
            int(res.get("n_neighbors", 0)),
        )
    elif d < min_d and d < 1.0:
        LOGGER.warning(
            "GS11: %s D=%.4f too low - skipping correction, flagging",
            target_cid or "?",
            d,
        )
    return out, res
