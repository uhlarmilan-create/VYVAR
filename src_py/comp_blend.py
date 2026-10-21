# -*- coding: ascii -*-
"""Light of catalogue neighbours inside a comparison-star aperture (BLEND-2026-10).

A comparison star calibrates the zero point with its catalogue magnitude, so the flux
measured in its aperture must be its own flux. Gaia neighbours that are not resolved
from the ground add light and bias the zero point; in draft 524 (g) 8 of 76 comps had
more than 0.01 mag of neighbour light and the worst ~1 mag.

Contamination of a star (all fluxes in the zero-point band, G when the colour relation
is not valid):

    c = sum_n F_n * E(s_n; r_ap) / (F_star * E(0; r_ap))

E(s; r) is the fraction of a PSF centred s pixels from the aperture centre that falls
inside the aperture of radius r. The PSF is a circular Moffat profile (Moffat 1969)
with the night FWHM; its wing exponent beta is measured on the MASTERSTAR from
isolated unsaturated stars (growth curve fit). Ground-based profiles have wings much
heavier than a Gaussian (measured on draft 524: beta = 2.7; at r = 1.5 FWHM the
Gaussian encloses 0.997, the data 0.937), so a Gaussian would under-estimate the light
of neighbours 1-3 FWHM away.

A star with c > BLEND_FLUX_FRAC_MAX (1 %, 0.011 mag - below the 0.03 mag scatter of
the Gaia -> band relations) is not used as a comparison star. The value is computed for
every k of the aperture grid (BLEND-K-2026-10): the pool keeps a star that passes at the
smallest k, and a target's aperture is chosen only among the k at which all its comps
pass (aperture_pertarget.allowed_f_from_comp_blend).

References:
    Moffat (1969) A&A 3, 455 - stellar profile
    Trujillo et al. (2001) MNRAS 328, 977 - Moffat profiles of seeing-limited images
    Gaia Collaboration, Vallenari et al. (2023) A&A 674, A1 - neighbour catalogue
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

#: Maximum neighbour light in a comparison / check aperture, fraction of the star's own flux.
BLEND_FLUX_FRAC_MAX: Final[float] = 0.01
#: Moffat beta used when the MASTERSTAR has too few isolated stars to measure it.
MOFFAT_BETA_FALLBACK: Final[float] = 2.5
_MIN_PROFILE_STARS: Final[int] = 5

# Quadrature for the off-centre enclosed fraction (Gauss-Legendre in radius, uniform in angle).
_GL_X, _GL_W = np.polynomial.legendre.leggauss(48)
_N_THETA: Final[int] = 96
_THETA = (np.arange(_N_THETA) + 0.5) * (2.0 * math.pi / _N_THETA)


def moffat_alpha(fwhm: float, beta: float) -> float:
    """Moffat core radius alpha for a given FWHM and beta."""
    return float(fwhm) / (2.0 * math.sqrt(2.0 ** (1.0 / float(beta)) - 1.0))


def moffat_ee(r: float | np.ndarray, alpha: float, beta: float) -> float | np.ndarray:
    """Encircled energy of a centred, normalised Moffat profile within radius r."""
    return 1.0 - (1.0 + (np.asarray(r, dtype=float) / alpha) ** 2) ** (1.0 - beta)


def measure_moffat_beta(
    image: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    peaks: np.ndarray,
    sat_adu: float,
    fwhm_px: float,
    neighbour_xy: np.ndarray,
    neighbour_mag: np.ndarray,
    star_mag: np.ndarray,
) -> dict[str, Any]:
    """Moffat beta from the growth curves of isolated, unsaturated stars.

    Isolation: no catalogue source brighter than (star + 5 mag) within 5 FWHM.
    Growth curve to 4 FWHM, sky = median in 4.4..5.6 FWHM. Fit of EE(r)/EE(4 FWHM).
    Returns {"beta", "fwhm_fit_px", "n_stars", "source"}.
    """
    from scipy.optimize import least_squares  # noqa: PLC0415
    from scipy.spatial import cKDTree  # noqa: PLC0415

    out = {"beta": MOFFAT_BETA_FALLBACK, "fwhm_fit_px": float("nan"), "n_stars": 0, "source": "fallback"}
    if image is None or not (math.isfinite(fwhm_px) and fwhm_px > 0):
        return out
    h, w = image.shape
    r_out = 4.0 * fwhm_px
    r_s0, r_s1 = 4.4 * fwhm_px, 5.6 * fwhm_px
    half = int(math.ceil(r_s1)) + 1
    tree = cKDTree(neighbour_xy) if len(neighbour_xy) else None
    radii = np.linspace(0.25, 4.0, 16) * fwhm_px
    curves = []
    for x0, y0, pk, m0 in zip(xs, ys, peaks, star_mag, strict=False):
        if not (math.isfinite(x0) and math.isfinite(y0) and math.isfinite(pk) and math.isfinite(m0)):
            continue
        if not (0.05 * sat_adu < pk < 0.6 * sat_adu):
            continue
        if not (half < x0 < w - half - 1 and half < y0 < h - half - 1):
            continue
        if tree is not None:
            idx = tree.query_ball_point([x0, y0], 5.0 * fwhm_px)
            bad = False
            for i in idx:
                dx = neighbour_xy[i, 0] - x0
                dy = neighbour_xy[i, 1] - y0
                if dx * dx + dy * dy < 1.5**2:
                    continue  # the star itself
                if neighbour_mag[i] < m0 + 5.0:
                    bad = True
                    break
            if bad:
                continue
        yi, xi = int(round(y0)), int(round(x0))
        sub = image[yi - half : yi + half + 1, xi - half : xi + half + 1].astype(float)
        yy, xx = np.mgrid[yi - half : yi + half + 1, xi - half : xi + half + 1]
        rr = np.hypot(xx - x0, yy - y0)
        sky = float(np.median(sub[(rr > r_s0) & (rr < r_s1)]))
        ee = np.array([np.sum(sub[rr < r] - sky) for r in radii])
        if not (np.isfinite(ee).all() and ee[-1] > 0):
            continue
        curves.append(ee / ee[-1])
    out["n_stars"] = len(curves)
    if len(curves) < _MIN_PROFILE_STARS:
        return out
    prof = np.median(np.asarray(curves), axis=0)

    def _res(p: np.ndarray) -> np.ndarray:
        a, b = p
        return moffat_ee(radii, a, b) / moffat_ee(r_out, a, b) - prof

    try:
        fit = least_squares(_res, [fwhm_px / 1.2, 3.0], bounds=([0.1, 1.05], [50.0, 20.0]))
        a, b = float(fit.x[0]), float(fit.x[1])
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[BLEND] Moffat fit failed: %s", exc)
        return out
    out.update(
        beta=b,
        fwhm_fit_px=2.0 * a * math.sqrt(2.0 ** (1.0 / b) - 1.0),
        source="masterstar_growth_curve",
    )
    return out


def _moffat_psf(d2: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Normalised Moffat surface brightness at squared distance d2."""
    return (beta - 1.0) / (math.pi * alpha**2) * (1.0 + d2 / alpha**2) ** (-beta)


def _aperture_nodes(r_ap: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quadrature nodes (x, y, weight) covering the aperture disc."""
    rho = 0.5 * r_ap * (_GL_X + 1.0)
    wr = 0.5 * r_ap * _GL_W * rho * (2.0 * math.pi / _N_THETA)
    xx = (rho[:, None] * np.cos(_THETA)[None, :]).ravel()
    yy = (rho[:, None] * np.sin(_THETA)[None, :]).ravel()
    ww = np.repeat(wr, _N_THETA)
    return xx, yy, ww


def _annulus_samples(r_in: float, r_out: float, n_r: int = 24, n_t: int = 72) -> tuple[np.ndarray, np.ndarray]:
    """Equal-area sample points of the sky annulus (for the median)."""
    u = (np.arange(n_r) + 0.5) / n_r
    rr = np.sqrt(r_in**2 + u * (r_out**2 - r_in**2))
    tt = (np.arange(n_t) + 0.5) * (2.0 * math.pi / n_t)
    return (rr[:, None] * np.cos(tt)[None, :]).ravel(), (rr[:, None] * np.sin(tt)[None, :]).ravel()


def contamination_fractions(
    star_xy: np.ndarray,
    star_flux: np.ndarray,
    neighbour_xy: np.ndarray,
    neighbour_flux: np.ndarray,
    *,
    r_ap_px: float,
    fwhm_px: float,
    beta: float,
    annulus_px: tuple[float, float] | None = None,
    star_ids: list[str] | None = None,
    neighbour_ids: list[str] | None = None,
        star_flux_g: np.ndarray | None = None,
    neighbour_flux_g: np.ndarray | None = None,
    self_match_px: float = 0.05,
) -> np.ndarray:
    """Extra light measured in a star's aperture because of its neighbours, / own light.

    The measurement is modelled as the pipeline does it: aperture sum minus aperture
    area x median of the sky annulus. With ``annulus_px`` the median is taken over the
    total model light (star + neighbours) in the annulus, so a far neighbour whose wing
    is flat over aperture and annulus cancels, while a near one does not:

        meas(with nb) = F_s E(0) + sum F_n E(s_n) - A * median_ann(F_s P_s + sum F_n P_n)
        meas(alone)   = F_s E(0) - A * median_ann(F_s P_s)
        c = meas(with nb) / meas(alone) - 1

    Flux ratios neighbour/star are taken in the zero-point band when both stars have a
    valid band magnitude, else in Gaia G (``*_flux_g``), so a neighbour without colour
    is never compared on a different scale. The star itself is skipped by catalogue id.
    """
    from scipy.spatial import cKDTree  # noqa: PLC0415

    n = len(star_xy)
    out = np.zeros(n, dtype=float)
    if n == 0 or len(neighbour_xy) == 0:
        return out
    alpha = moffat_alpha(fwhm_px, beta)
    ax, ay, aw = _aperture_nodes(float(r_ap_px))
    area = math.pi * float(r_ap_px) ** 2
    if annulus_px is not None:
        sx_, sy_ = _annulus_samples(float(annulus_px[0]), float(annulus_px[1]))
        r_search = float(annulus_px[1]) + 6.0 * fwhm_px
    else:
        sx_ = sy_ = None
        r_search = float(r_ap_px) + 6.0 * fwhm_px
    e0 = float(moffat_ee(r_ap_px, alpha, beta))
    own_ann = 0.0
    if sx_ is not None:
        own_ann = float(np.median(_moffat_psf(sx_**2 + sy_**2, alpha, beta)))
    own_meas = e0 - area * own_ann
    tree = cKDTree(neighbour_xy)
    for i in range(n):
        fs = float(star_flux[i])
        fs_g = float(star_flux_g[i]) if star_flux_g is not None else float("nan")
        band_ok = math.isfinite(fs) and fs > 0
        if (not band_ok and not (math.isfinite(fs_g) and fs_g > 0)) or own_meas <= 0:
            out[i] = float("nan")
            continue
        sid = star_ids[i] if star_ids is not None else None
        rel, dx, dy = [], [], []
        for j in tree.query_ball_point(star_xy[i], r_search):
            if sid is not None and neighbour_ids is not None and neighbour_ids[j] == sid:
                continue
            ddx = float(neighbour_xy[j, 0] - star_xy[i, 0])
            ddy = float(neighbour_xy[j, 1] - star_xy[i, 1])
            if ddx * ddx + ddy * ddy < self_match_px**2:
                continue
            fn = float(neighbour_flux[j])
            if band_ok and math.isfinite(fn) and fn > 0:
                ratio = fn / fs
            elif neighbour_flux_g is not None and math.isfinite(fs_g) and fs_g > 0:
                ratio = float(neighbour_flux_g[j]) / fs_g
            else:
                continue
            if not (math.isfinite(ratio) and ratio > 0):
                continue
            rel.append(ratio)
            dx.append(ddx)
            dy.append(ddy)
        if not rel:
            continue
        rel_a = np.asarray(rel)
        dxa, dya = np.asarray(dx), np.asarray(dy)
        # neighbour light inside the aperture (in units of the star's total flux)
        d2 = (ax[None, :] - dxa[:, None]) ** 2 + (ay[None, :] - dya[:, None]) ** 2
        in_ap = float(np.sum(rel_a * (_moffat_psf(d2, alpha, beta) @ aw)))
        sky_shift = 0.0
        if sx_ is not None:
            d2s = (sx_[None, :] - dxa[:, None]) ** 2 + (sy_[None, :] - dya[:, None]) ** 2
            tot = _moffat_psf(sx_**2 + sy_**2, alpha, beta) + rel_a @ _moffat_psf(d2s, alpha, beta)
            sky_shift = area * (float(np.median(tot)) - own_ann)
        out[i] = (in_ap - sky_shift) / own_meas
    return out


def blend_contamination_for_pool(
    pool: pd.DataFrame,
    *,
    field_catalog_csv: Path | None,
    masterstar_fits: Path | None,
    zp_band: str,
    fwhm_px: float,
    r_ap_px: float,
    sat_adu: float | None,
    annulus_px: tuple[float, float] | None = None,
    profile_df: pd.DataFrame | None = None,
    psf: dict[str, Any] | None = None,
) -> tuple[pd.Series, dict[str, Any]]:
    """Contamination fraction for every pool row (index-aligned) and a metadata dict.

    ``psf``: a profile from an earlier call (``meta["psf"]``) - reused instead of measuring
    beta again, so a grid of apertures uses one PSF.

    Neighbours: the field Gaia cone (``field_catalog_cone.csv``) projected with the
    MASTERSTAR WCS onto the reference frame. Fluxes in ``zp_band`` (G when the relation
    is not valid for a star). Missing inputs -> all NaN and ``meta["status"]`` explains.
    """
    from astropy.io import fits  # noqa: PLC0415
    from astropy.wcs import WCS  # noqa: PLC0415

    from gaia_bands import band_mag_from_gaia  # noqa: PLC0415

    meta: dict[str, Any] = {
        "rule_id": "COMP_BLEND_NEIGHBOUR_LIGHT",
        "threshold_flux_frac": BLEND_FLUX_FRAC_MAX,
        "r_ap_px": float(r_ap_px),
        "fwhm_px": float(fwhm_px),
        "annulus_px": list(annulus_px) if annulus_px is not None else None,
        "zp_band": str(zp_band),
    }
    nan = pd.Series(np.nan, index=pool.index, dtype=float)
    if field_catalog_csv is None or not Path(field_catalog_csv).is_file():
        meta["status"] = "skipped: field_catalog_cone.csv missing"
        return nan, meta
    if masterstar_fits is None or not Path(masterstar_fits).is_file():
        meta["status"] = "skipped: MASTERSTAR missing"
        return nan, meta
    if not (math.isfinite(fwhm_px) and fwhm_px > 0 and math.isfinite(r_ap_px) and r_ap_px > 0):
        meta["status"] = "skipped: FWHM / aperture unknown"
        return nan, meta
    cat = pd.read_csv(field_catalog_csv, low_memory=False)
    with fits.open(masterstar_fits, memmap=False) as hdul:
        hdr = hdul[0].header
        img = np.asarray(hdul[0].data, dtype=np.float32)
    wcs = WCS(hdr)
    cx, cy = wcs.all_world2pix(
        pd.to_numeric(cat["ra_deg"], errors="coerce").to_numpy(),
        pd.to_numeric(cat["dec_deg"], errors="coerce").to_numpy(),
        0,
    )
    cg = pd.to_numeric(cat["mag"], errors="coerce").to_numpy(dtype=float)
    cbp = pd.to_numeric(cat.get("bp_rp"), errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(cx) & np.isfinite(cy) & np.isfinite(cg)
    cx, cy, cg, cbp = cx[ok], cy[ok], cg[ok], cbp[ok]

    def _band_flux(g: np.ndarray, bp: np.ndarray) -> np.ndarray:
        """Band flux, NaN when the Gaia -> band relation is not valid for the star."""
        m = np.array([band_mag_from_gaia(a, b, zp_band)[0] for a, b in zip(g, bp, strict=False)])
        return 10.0 ** (-0.4 * m)

    from gaia_catalog_id import normalize_gaia_source_id  # noqa: PLC0415

    nb_ids = [str(normalize_gaia_source_id(v) or "") for v in cat["catalog_id"].to_numpy()[ok]]
    nb_xy = np.c_[cx, cy]
    nb_flux = _band_flux(cg, cbp)
    meta["neighbour_catalog_mag_limit"] = float(np.nanmax(cg)) if cg.size else float("nan")
    meta["n_neighbour_catalog"] = int(cg.size)

    px = pd.to_numeric(pool.get("x"), errors="coerce").to_numpy(dtype=float)
    py = pd.to_numeric(pool.get("y"), errors="coerce").to_numpy(dtype=float)
    pg = pd.to_numeric(pool.get("phot_g_mean_mag", pool.get("mag")), errors="coerce").to_numpy(dtype=float)
    if "mag" in pool.columns:
        pg = np.where(np.isfinite(pg), pg, pd.to_numeric(pool["mag"], errors="coerce").to_numpy(dtype=float))
    pbp = pd.to_numeric(pool.get("bp_rp"), errors="coerce").to_numpy(dtype=float)

    # PSF wing (beta) from all detected stars of the MASTERSTAR (``profile_df``), not only the pool.
    _pf = profile_df if profile_df is not None and not profile_df.empty else pool
    _ppx = pd.to_numeric(_pf.get("x"), errors="coerce").to_numpy(dtype=float)
    _ppy = pd.to_numeric(_pf.get("y"), errors="coerce").to_numpy(dtype=float)
    _ppg = pd.to_numeric(_pf.get("mag"), errors="coerce").to_numpy(dtype=float)
    _pk = _pf.get("peak_max_adu")
    peaks = pd.to_numeric(_pk, errors="coerce").to_numpy(dtype=float) if _pk is not None else np.full(len(_pf), np.nan)
    sat = float(sat_adu) if sat_adu is not None and math.isfinite(float(sat_adu)) else float("nan")
    if psf is not None and math.isfinite(float(psf.get("beta", float("nan")))):
        prof = dict(psf)
    elif math.isfinite(sat) and sat > 0 and np.isfinite(peaks).any():
        prof = measure_moffat_beta(img, _ppx, _ppy, peaks, sat, fwhm_px, nb_xy, cg, _ppg)
    else:
        prof = {"beta": MOFFAT_BETA_FALLBACK, "fwhm_fit_px": float("nan"), "n_stars": 0, "source": "fallback"}
    meta["psf"] = prof
    # Geometry from catalogue positions (star and neighbours on the same WCS projection).
    pra = pd.to_numeric(pool.get("ra_deg"), errors="coerce").to_numpy(dtype=float)
    pdec = pd.to_numeric(pool.get("dec_deg"), errors="coerce").to_numpy(dtype=float)
    sx, sy = wcs.all_world2pix(pra, pdec, 0)
    sx = np.where(np.isfinite(sx), sx, px)
    sy = np.where(np.isfinite(sy), sy, py)
    _idc = "catalog_id" if "catalog_id" in pool.columns else "name"
    s_ids = [str(normalize_gaia_source_id(v) or "") for v in pool[_idc].to_numpy()]
    c = contamination_fractions(
        np.c_[sx, sy], _band_flux(pg, pbp), nb_xy, nb_flux,
        r_ap_px=float(r_ap_px), fwhm_px=float(fwhm_px), beta=float(prof["beta"]),
        annulus_px=annulus_px,
        star_ids=s_ids, neighbour_ids=nb_ids,
        star_flux_g=10.0 ** (-0.4 * pg), neighbour_flux_g=10.0 ** (-0.4 * cg),
    )
    meta["status"] = "ok"
    return pd.Series(c, index=pool.index, dtype=float), meta
