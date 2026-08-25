"""Effective PSF (ePSF) construction on MASTERSTAR and per-star PSF photometry.

Uses Photutils EPSFBuilder / PSFPhotometry. Does not import ``pipeline`` (avoid cycles).
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.models import Const2D, Moffat2D
from astropy.nddata import NDData
from astropy.table import Table
from masterstar_context import header_core_fwhm_px
from photutils.psf import (
    EPSFBuilder,
    EPSFStars,
    ImagePSF,
    IterativePSFPhotometry,
    PSFPhotometry,
    SourceGrouper,
    extract_stars,
    grid_from_epsfs,
)

from database import VyvarDatabase
from gaia_catalog_id import normalize_gaia_source_id as _norm_catalog_id
from infolog import log_event

LOGGER = logging.getLogger(__name__)

_MASTERSTAR_EPSF_NAME = "masterstar_epsf.fits"
_MASTERSTAR_EPSF_META_NAME = "masterstar_epsf_meta.json"

# INTERIM cap: DAOPHOT/allstar practice uses tens to low hundreds of PSF stars
# (Stetson 1987; Harris et al. DAOPHOT manual). Part D will tune empirically.
_EPSF_BUILD_INTERIM_TOP_N = 200
_EPSF_BUILD_CLEAN_SOURCE_STATES = frozenset({"DETECTED_P1", "DETECTED_P2", "FORCED_SEED"})
_EPSF_BUILD_GUARD_MAX_DROP_FRAC = 0.10
_EPSF_BUILD_GUARD_REASON = "epsf_build_non_finite_guard"


class InstrumentedEPSFBuilder(EPSFBuilder):
    """EPSFBuilder subclass recording per-iteration photutils _fit_error_status counts."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.iteration_failure_curve: list[dict[str, int]] = []

    def _process_iteration(self, stars: Any, epsf: Any, iter_num: int) -> Any:
        result = super()._process_iteration(stars, epsf, iter_num)
        hist = {0: 0, 1: 0, 2: 0, 3: 0}
        for star in stars:
            st = int(getattr(star, "_fit_error_status", 0) or 0)
            hist[st] = hist.get(st, 0) + 1
        n_fail = hist[1] + hist[2] + hist[3]
        self.iteration_failure_curve.append(
            {
                "iteration": len(self.iteration_failure_curve) + 1,
                "n_stars": len(stars),
                "n_fail": n_fail,
                "n_status_0": hist[0],
                "n_status_1": hist[1],
                "n_status_2": hist[2],
                "n_status_3": hist[3],
            }
        )
        return result


def _epsf_apply_build_selection_gates(
    df: pd.DataFrame,
    *,
    platesolve_dir: Path,
    cutout_size: int,
    image_shape: tuple[int, int],
    funnel: dict[str, Any],
) -> pd.DataFrame:
    """Part C build-star gates: zone linear, clean source_state, science scope, edge-safe, interim top-N."""
    from epsf_science_set import build_epsf_science_set

    work = df.copy()
    funnel["n_build_input"] = int(len(work))

    if "zone" in work.columns:
        work = work[work["zone"].astype(str).str.strip().str.lower() == "linear"]
    funnel["n_after_zone_linear"] = int(len(work))

    if "source_state" in work.columns:
        ss = work["source_state"].astype(str).str.strip().str.upper()
        work = work[ss.isin(_EPSF_BUILD_CLEAN_SOURCE_STATES)]
    funnel["n_after_clean_source_state"] = int(len(work))

    sci = build_epsf_science_set(platesolve_dir)
    if sci.catalog_ids:
        cids = {_norm_catalog_id(x) for x in sci.catalog_ids}
        if "_cid" not in work.columns:
            work["_cid"] = work["catalog_id"].map(_norm_catalog_id)
        work = work[work["_cid"].isin(cids)]
    else:
        work = work.iloc[0:0]
    funnel["n_after_science_scope"] = int(len(work))

    h, w = image_shape
    half = int(cutout_size) // 2
    margin = half + 1
    xs = pd.to_numeric(work["x"], errors="coerce")
    ys = pd.to_numeric(work["y"], errors="coerce")
    edge_ok = (
        xs.notna()
        & ys.notna()
        & (xs >= margin)
        & (xs < w - margin)
        & (ys >= margin)
        & (ys < h - margin)
    )
    work = work.loc[edge_ok]
    funnel["n_after_edge_safe_cutout"] = int(len(work))

    if len(work) > _EPSF_BUILD_INTERIM_TOP_N:
        mag_col = next(
            (c for c in ("mag", "catalog_mag", "phot_g_mean_mag") if c in work.columns),
            None,
        )
        if mag_col is not None:
            work = work.copy()
            work["_sort_mag"] = pd.to_numeric(work[mag_col], errors="coerce")
            work = work.sort_values("_sort_mag", ascending=True, na_position="last")
        work = work.head(_EPSF_BUILD_INTERIM_TOP_N)
    funnel["n_after_interim_top_n"] = int(len(work))
    funnel["build_interim_top_n"] = int(_EPSF_BUILD_INTERIM_TOP_N)
    funnel["build_interim_top_n_mark"] = "INTERIM"

    return work


def _epsf_noop_finder(_data: np.ndarray, mask: np.ndarray | None = None) -> None:
    """Finder for IterativePSFPhotometry: no residual sources (catalog positions only)."""
    _ = _data, mask
    return None


def _clamp_fwhm_px(v: float) -> float:
    return float(max(2.0, min(12.0, v)))


def _median_fwhm_obs_files(db: VyvarDatabase, draft_id: int) -> float | None:
    rows = db.fetch_draft_light_rows_for_quality(int(draft_id))
    vals: list[float] = []
    for row in rows:
        try:
            x = float(row.get("FWHM"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(x) and x > 0.0:
            vals.append(x)
    if not vals:
        return None
    return float(np.median(np.asarray(vals, dtype=np.float64)))


def _epsf_allowed_catalog_ids(platesolve_dir: Path) -> tuple[set[str], int, int]:
    """Catalog IDs for ePSF build: active_targets + comparison_stars pool."""
    root = Path(platesolve_dir)
    phot = root / "photometry"
    ms_ids: set[str] = set()
    comp_ids: set[str] = set()

    def _add_cid(raw: Any, bucket: set[str]) -> None:
        if raw is None:
            return
        s = str(raw).strip()
        if not s or s.lower() in ("nan", "none"):
            return
        try:
            bucket.add(str(_norm_catalog_id(s)).strip())
        except Exception:  # noqa: BLE001
            bucket.add(s)

    at_p = phot / "active_targets.csv"
    if at_p.is_file():
        try:
            at = pd.read_csv(at_p, low_memory=False, dtype={"catalog_id": str})
            for _, row in at.iterrows():
                _add_cid(row.get("catalog_id"), ms_ids)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("[ePSF] active_targets load failed: %s", exc)

    for comp_p in (phot / "comparison_stars.csv", root / "comparison_stars.csv"):
        if not comp_p.is_file():
            continue
        try:
            cs = pd.read_csv(comp_p, low_memory=False, dtype={"catalog_id": str})
            for _, row in cs.iterrows():
                _add_cid(row.get("catalog_id"), comp_ids)
            break
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("[ePSF] comparison_stars load failed (%s): %s", comp_p.name, exc)

    all_ids = ms_ids | comp_ids
    return all_ids, len(ms_ids), len(comp_ids)


def _epsf_positions_from_csvs(
    platesolve_dir: Path,
    allowed_ids: set[str],
    existing: set[str],
) -> list[dict[str, Any]]:
    """Add x/y positions for allowed IDs missing from the primary CSV candidate set."""
    root = Path(platesolve_dir)
    phot = root / "photometry"
    extra: list[dict[str, Any]] = []
    need = allowed_ids - existing
    if not need:
        return extra

    def _try_csv(path: Path) -> None:
        nonlocal need, extra
        if not path.is_file() or not need:
            return
        try:
            df = pd.read_csv(path, low_memory=False, dtype={"catalog_id": str})
        except Exception as exc:  # noqa: BLE001
            logging.error('[EXC-0446] Auxiliary CSV read for PSF star x/y fails - those catalog_ids missing from PSF star_pos...: %s', exc)
            return
        for _, row in df.iterrows():
            cid = str(_norm_catalog_id(row.get("catalog_id", "")) or "").strip()
            if not cid or cid not in need:
                continue
            try:
                x_f = float(pd.to_numeric(row.get("x"), errors="coerce"))
                y_f = float(pd.to_numeric(row.get("y"), errors="coerce"))
            except (TypeError, ValueError):
                continue
            if not (math.isfinite(x_f) and math.isfinite(y_f)):
                continue
            ra_v = float(pd.to_numeric(row.get("ra_deg"), errors="coerce"))
            de_v = float(pd.to_numeric(row.get("dec_deg"), errors="coerce"))
            extra.append(
                {
                    "catalog_id": cid,
                    "x": x_f,
                    "y": y_f,
                    "ra_deg": ra_v if math.isfinite(ra_v) else float("nan"),
                    "dec_deg": de_v if math.isfinite(de_v) else float("nan"),
                }
            )
            need.discard(cid)

    _try_csv(phot / "active_targets.csv")
    for comp_p in (phot / "comparison_stars.csv", root / "comparison_stars.csv"):
        _try_csv(comp_p)
    return extra


def get_epsf_fwhm_from_context(
    masterstar_fits_path: Path,
    db: VyvarDatabase,
    draft_id: int,
) -> float:
    """Return FWHM in pixels for EPSFBuilder (VY_FWHM_GAUSS -> VY_FWHM_GAUSSIAN -> VY_FWHM header
    -> manifest files[] median -> 4.5), clamped to [2, 12]."""
    fwhm: float | None = None
    p = Path(masterstar_fits_path)
    try:
        if p.is_file():
            with fits.open(p, memmap=True) as hdul:
                fwhm = header_core_fwhm_px(hdul[0].header)
    except Exception:  # noqa: BLE001
        fwhm = None

    if fwhm is None:
        fwhm = _median_fwhm_obs_files(db, draft_id)

    if fwhm is None:
        fwhm = 4.5

    return _clamp_fwhm_px(fwhm)


def _to_odd_cutout(n: int) -> int:
    n = max(15, int(n))
    if n % 2 == 0:
        n += 1
    return n


def _scalar_is_explicit_false(v: Any) -> bool:
    """True only for explicit false (not unknown / empty)."""
    if isinstance(v, np.bool_):
        return not bool(v)
    if v is False:
        return True
    if v is True or v is None:
        return False
    if isinstance(v, float) and math.isnan(v):
        return False
    s = str(v).strip().lower()
    if s == "":
        return False
    return s in ("false", "0", "no")


def _scalar_is_explicit_true(v: Any) -> bool:
    """True only for explicit true (not unknown / empty)."""
    if isinstance(v, np.bool_):
        return bool(v)
    if v is True:
        return True
    if v is False or v is None:
        return False
    if isinstance(v, float) and math.isnan(v):
        return False
    s = str(v).strip().lower()
    if s == "":
        return False
    return s in ("true", "1", "yes", "y")


def _read_plate_scale_arcsec_px_from_fits(fits_path: Path) -> float | None:
    """Read plate scale in arcsec/px from FITS header, CD/WCS-FIRST.

    Priority: (1) solved WCS CD matrix / CDELT; (2) VY_PLTS - ONLY if it agrees with
    the CD value within 5% (else ignored); (3) SECPIX/PIXSCALE/SCALE only when no CD.
    Returns None if no reliable value. Sanity range: 0.1-30.0 arcsec/px (wide-field safe).
    """
    _sane_min = 0.1
    _sane_max = 30.0
    try:
        with fits.open(Path(fits_path), memmap=False) as hdul:
            hdr = hdul[0].header

            # (1) Authoritative: solved WCS (handles CD, PC+CDELT, SIP), then raw CD, then CDELT.
            cd_scale: float | None = None
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
                    v = float(_np.mean(_pps(_w))) * 3600.0
                    if _sane_min <= v <= _sane_max:
                        cd_scale = v
            except Exception:  # noqa: BLE001
                cd_scale = None
            if cd_scale is None:
                cd11 = hdr.get("CD1_1")
                cd12 = hdr.get("CD1_2", 0.0)
                if cd11 is not None:
                    try:
                        v = math.sqrt(float(cd11) ** 2 + float(cd12) ** 2) * 3600.0
                        if _sane_min <= v <= _sane_max:
                            cd_scale = v
                    except (TypeError, ValueError):
                        pass
            if cd_scale is None:
                cdelt1 = hdr.get("CDELT1")
                if cdelt1 is not None:
                    try:
                        v = abs(float(cdelt1)) * 3600.0
                        if _sane_min <= v <= _sane_max:
                            cd_scale = v
                    except (TypeError, ValueError):
                        pass

            if cd_scale is not None:
                # (2) VY_PLTS cross-check: warn and ignore if it disagrees > 5%.
                vy = hdr.get("VY_PLTS")
                if vy is not None:
                    try:
                        vyf = float(vy)
                        if vyf > 0 and abs(vyf - cd_scale) / cd_scale > 0.05:
                            logging.warning(
                                "[ePSF] VY_PLTS=%.3f disagrees with CD-derived %.3f arcsec/px (>5%%) - using CD.",
                                vyf,
                                cd_scale,
                            )
                    except (TypeError, ValueError):
                        pass
                return cd_scale

            # (3) No usable CD/WCS - header keyword fallbacks (incl. VY_PLTS).
            vy = hdr.get("VY_PLTS")
            if vy is not None:
                try:
                    v = float(vy)
                    if _sane_min <= v <= _sane_max:
                        return v
                except (TypeError, ValueError):
                    pass

            for key in ("SECPIX", "PIXSCALE", "PIXSCAL1", "PLTSCALE"):
                val = hdr.get(key)
                if val is not None:
                    try:
                        v = float(val)
                        if _sane_min <= v <= _sane_max:
                            return v
                    except (TypeError, ValueError):
                        pass

            scale = hdr.get("SCALE")
            if scale is not None:
                try:
                    v = float(scale)
                    if _sane_min <= v <= _sane_max:
                        return v
                except (TypeError, ValueError):
                    pass

            return None
    except Exception as exc:  # noqa: BLE001
        logging.error('[EXC-0447] Plate scale from FITS header fails - PSF cutout/fit sizing uses defaults: %s', exc)
        return None


def _fit_shape_for_cutout(cutout_size: int, fwhm_px: float | None = None) -> tuple[int, int]:
    """PSF fit window size (odd). Uses global ``fwhm_px`` from ePSF meta (uniform per star)."""
    if fwhm_px is not None and fwhm_px > 0:
        # 2xFWHM+1, rounded up to nearest odd integer, minimum 5
        fs = int(math.ceil(2.0 * float(fwhm_px) + 1.0))
        if fs % 2 == 0:
            fs += 1
        fs = max(5, fs)
    else:
        # legacy fallback
        fs = cutout_size - 4
        if fs % 2 == 0:
            fs += 1
        fs = max(3, fs)
    return (fs, fs)


def _moffat_fwhm_px(gamma: float, alpha: float) -> float:
    """FWHM in pixels from Moffat gamma and alpha (beta) parameters.

    FWHM = 2 * gamma * sqrt(2^(1/alpha) - 1)
    """
    try:
        val = 2.0 * gamma * math.sqrt(2.0 ** (1.0 / alpha) - 1.0)
        return float(val) if math.isfinite(val) and val > 0 else float("nan")
    except (ValueError, ZeroDivisionError, OverflowError):
        return float("nan")


def _compute_aperture_correction(
    psf_fluxes: np.ndarray,
    ref_fluxes: np.ndarray,
    chi2_vals: np.ndarray,
    *,
    chi2_limit: float = 5.0,
    min_ref_stars: int = 5,
) -> tuple[float, int]:
    """Compute aperture correction factor from clean reference stars.

    Returns ``(correction_factor, n_used)``. ``correction_factor`` is 1.0 if not enough clean stars.

    This chi2<5 per-star DAO/PSF median is the ``chi2_lt5_legacy`` fallback only
    (EPSF-AC-01 A2: the gate is a brightness cut). Absolute PSF scale, when
    wanted, is a DAOGROW/DOLPHOT-style growth-curve total (Stetson 1990), not
    this ensemble.
    """
    mask = (
        np.isfinite(psf_fluxes)
        & np.isfinite(ref_fluxes)
        & np.isfinite(chi2_vals)
        & (psf_fluxes > 0)
        & (ref_fluxes > 0)
        & (chi2_vals < chi2_limit)
    )
    n_clean = int(mask.sum())
    if n_clean < min_ref_stars:
        return 1.0, n_clean
    ratios = ref_fluxes[mask] / psf_fluxes[mask]
    med = float(np.median(ratios))
    mad = float(np.median(np.abs(ratios - med)))
    if mad > 0:
        inliers = np.abs(ratios - med) < 3.0 * mad
        ratios = ratios[inliers]
    if len(ratios) < min_ref_stars:
        return 1.0, len(ratios)
    return float(np.median(ratios)), len(ratios)


PSF_AC_POLICY_P4_NONE = "p4_none"
PSF_AC_POLICY_CHI2_LT5_LEGACY = "chi2_lt5_legacy"
PSF_AC_POLICIES = frozenset({PSF_AC_POLICY_P4_NONE, PSF_AC_POLICY_CHI2_LT5_LEGACY})


def resolve_psf_ac_policy(
    raw: Any = None,
    *,
    apply_aperture_correction: bool | None = None,
) -> str:
    """Return ``p4_none`` or ``chi2_lt5_legacy``. Explicit policy wins over the bool."""
    s = str(raw or "").strip().lower()
    if s in PSF_AC_POLICIES:
        return s
    if apply_aperture_correction is False:
        return PSF_AC_POLICY_P4_NONE
    if apply_aperture_correction is True:
        return PSF_AC_POLICY_CHI2_LT5_LEGACY
    return PSF_AC_POLICY_P4_NONE


def invert_applied_ac(
    psf_flux: float,
    psf_flux_err: float,
    ac_factor: float,
    ac_applied: bool,
) -> tuple[float, float]:
    """Recover uncorrected fit flux from a stored AC multiply."""
    flux = float(psf_flux) if psf_flux is not None else float("nan")
    err = float(psf_flux_err) if psf_flux_err is not None else float("nan")
    fac = float(ac_factor) if ac_factor is not None else float("nan")
    if (
        bool(ac_applied)
        and math.isfinite(flux)
        and flux > 0
        and math.isfinite(fac)
        and fac > 0
    ):
        flux = flux / fac
        if math.isfinite(err):
            err = err / fac
    return flux, err


def _compute_moffat_aperture_correction(
    moffat_results: pd.DataFrame,
    dao_fluxes: np.ndarray,
    *,
    chi2_limit: float = 5.0,
    min_flux_snr: float = 50000.0,
    min_ref_stars: int = 5,
) -> tuple[float, int]:
    """Compute Moffat aperture correction from bright isolated stars.

    Uses stars with moffat_fit_ok=True, chi2<chi2_limit, dao_flux>min_flux_snr.
    Returns (correction_factor, n_used) where correction_factor =
    median(dao_flux / moffat_flux) for clean reference stars.
    Apply: moffat_flux_corrected = moffat_flux * correction_factor
    """
    try:
        m_flux = pd.to_numeric(
            moffat_results.get("moffat_flux", pd.Series(dtype=float)),
            errors="coerce",
        ).values
        m_chi2 = pd.to_numeric(
            moffat_results.get("moffat_chi2", pd.Series(dtype=float)),
            errors="coerce",
        ).values
        m_ok = (
            moffat_results.get("moffat_fit_ok", pd.Series([False] * len(moffat_results)))
            .astype(str)
            .str.lower()
            .str.strip()
            == "true"
        )
        d_flux = np.asarray(dao_fluxes, dtype=float)

        mask = (
            m_ok
            & np.isfinite(m_flux)
            & (m_flux > 0)
            & np.isfinite(d_flux)
            & (d_flux > min_flux_snr)
            & np.isfinite(m_chi2)
            & (m_chi2 < chi2_limit)
        )
        n_clean = int(mask.sum())
        if n_clean < min_ref_stars:
            return 1.0, n_clean

        ratios = d_flux[mask] / m_flux[mask]
        med = float(np.median(ratios))
        mad = float(np.median(np.abs(ratios - med)))
        if mad > 0:
            inliers = np.abs(ratios - med) < 3.0 * mad
            ratios = ratios[inliers]
        if len(ratios) < min_ref_stars:
            return 1.0, len(ratios)
        return float(np.median(ratios)), len(ratios)
    except Exception:  # noqa: BLE001
        return 1.0, 0


# ePSF/input FWHM ratio warning band (diagnostic only; set from V3e harness scatter).
_EPSF_FWHM_RATIO_WARN_LO = 0.80
_EPSF_FWHM_RATIO_WARN_HI = 1.25


def _epsf_fwhm_native_legacy_px(epsf_data: np.ndarray, *, osamp: int) -> float:
    """Legacy half-max: first radius-sorted pixel below 0.5*peak (EPSF-1 diagnostic baseline)."""
    z = np.asarray(epsf_data, dtype=np.float64)
    cy, cx = np.array(z.shape) // 2
    y_idx, x_idx = np.indices(z.shape)
    r = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2).ravel()
    v = z.ravel()
    finite_mask = np.isfinite(v)
    r = r[finite_mask]
    v = v[finite_mask]
    if len(r) == 0 or v.max() <= 0:
        return float("nan")
    v = v / v.max()
    sort_idx = np.argsort(r)
    r_s = r[sort_idx]
    v_s = v[sort_idx]
    below_half = np.where(v_s < 0.5)[0]
    if len(below_half) == 0:
        return float("nan")
    return float(2.0 * float(r_s[below_half[0]]) / max(1, int(osamp)))


def _epsf_fwhm_native_from_profile(epsf_data: np.ndarray, *, osamp: int) -> float:
    """Azimuthally-binned radial profile half-max FWHM (native px)."""
    z = np.asarray(epsf_data, dtype=np.float64)
    cy, cx = np.array(z.shape) // 2
    yy, xx = np.indices(z.shape)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).ravel()
    v = z.ravel()
    ok = np.isfinite(v) & np.isfinite(r)
    r = r[ok]
    v = v[ok]
    if r.size < 10:
        return float("nan")
    peak = float(np.max(v))
    if peak <= 0:
        return float("nan")
    v = v / peak
    h, w = z.shape
    rmax = min(h, w) * 0.45
    bin_w = 0.5  # oversampled px
    edges = np.arange(0.0, rmax + bin_w * 0.5, bin_w)
    centers: list[float] = []
    means: list[float] = []
    for i in range(len(edges) - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        sel = (r >= lo) & (r < hi)
        if int(sel.sum()) < 3:
            continue
        centers.append(0.5 * (lo + hi))
        means.append(float(np.mean(v[sel])))
    if len(centers) < 4:
        return float("nan")
    c_arr = np.asarray(centers, dtype=np.float64)
    m_arr = np.asarray(means, dtype=np.float64)
    cross = None
    for i in range(len(c_arr) - 1):
        a, b = m_arr[i], m_arr[i + 1]
        if a >= 0.5 >= b and a != b:
            frac = (a - 0.5) / (a - b)
            cross = float(c_arr[i] + frac * (c_arr[i + 1] - c_arr[i]))
            break
    if cross is None:
        return float("nan")
    return float(2.0 * cross / max(1, int(osamp)))


def _epsf_build_imagepsf_from_stars(
    stars: Any,
    *,
    osamp: int,
    fwhm_px: float,
    cutout_size: int,
    smoothing_kernel: str | None = None,
    builder_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run EPSFBuilder on a (sky-subtracted) EPSFStars set; return normalized array + QC.

    Shared by the single global build and the spatially-varying grid build so the ePSF
    construction, QC, and ImagePSF flux normalization are identical in both paths.
    """
    osamp = max(1, int(osamp))
    if smoothing_kernel is None:
        if osamp <= 2:
            _smoothing = "quadratic"
        else:
            _smoothing = "quartic"
    else:
        _smoothing = str(smoothing_kernel).strip().lower() or ("quadratic" if osamp <= 2 else "quartic")

    _bkw = dict(builder_kwargs or {})
    _maxiters = int(_bkw.pop("maxiters", 15))
    builder = InstrumentedEPSFBuilder(
        oversampling=osamp,
        maxiters=_maxiters,
        progress_bar=False,
        smoothing_kernel=_smoothing,
        **_bkw,
    )
    epsf_model, _fitted = builder(stars)
    iteration_failure_curve = list(builder.iteration_failure_curve)
    epsf_data = epsf_model.data

    nan_frac = float(np.sum(~np.isfinite(epsf_data)) / epsf_data.size)

    # ePSF FWHM via azimuthally-binned radial profile (oversampled -> native)
    try:
        epsf_fwhm_native = _epsf_fwhm_native_from_profile(epsf_data, osamp=osamp)
    except Exception:  # noqa: BLE001
        epsf_fwhm_native = float("nan")

    # symmetry / asymmetry
    try:
        cy, cx = np.array(epsf_data.shape) // 2
        q1 = epsf_data[:cy, :cx]
        q2 = epsf_data[:cy, cx + (epsf_data.shape[1] % 2):]
        q3 = epsf_data[cy + (epsf_data.shape[0] % 2):, :cx]
        q4 = epsf_data[cy + (epsf_data.shape[0] % 2):, cx + (epsf_data.shape[1] % 2):]
        min_r = min(q1.shape[0], q2.shape[0], q3.shape[0], q4.shape[0])
        min_c = min(q1.shape[1], q2.shape[1], q3.shape[1], q4.shape[1])
        quads = np.stack([q[:min_r, :min_c] for q in [q1, q2[:, ::-1], q3[::-1, :], q4[::-1, ::-1]]])
        finite_q = np.all(np.isfinite(quads), axis=0)
        peak = float(np.nanmax(epsf_data))
        asymmetry = (
            float(np.nanstd(quads[:, finite_q], axis=0).mean()) / peak
            if (finite_q.any() and peak > 0)
            else float("nan")
        )
    except Exception:  # noqa: BLE001
        asymmetry = float("nan")

    _qc = {
        "epsf_fwhm_native_px": round(epsf_fwhm_native, 3) if math.isfinite(epsf_fwhm_native) else None,
        "epsf_vs_input_fwhm_ratio": (
            round(epsf_fwhm_native / fwhm_px, 3)
            if (math.isfinite(epsf_fwhm_native) and fwhm_px > 0)
            else None
        ),
        "epsf_nan_fraction": round(nan_frac, 4),
        "epsf_asymmetry": round(asymmetry, 4) if math.isfinite(asymmetry) else None,
    }

    fit_shape = _fit_shape_for_cutout(cutout_size, fwhm_px=fwhm_px)
    _epsf_raw = np.asarray(epsf_model.data, dtype=np.float64).copy()
    _norm_factor = float(_epsf_raw.sum() / (osamp**2))
    if _norm_factor > 0 and np.isfinite(_norm_factor):
        _epsf_normalized = _epsf_raw / _norm_factor
    else:
        _epsf_normalized = _epsf_raw
    arr = np.asarray(_epsf_normalized, dtype=np.float32)
    return {
        "arr": arr,
        "qc": _qc,
        "norm_factor": _norm_factor,
        "smoothing": _smoothing,
        "fit_shape": fit_shape,
        "epsf_sum_native": float(_epsf_normalized.sum() / osamp**2),
        "iteration_failure_curve": iteration_failure_curve,
    }


def _epsf_is_non_finite_build_error(exc: BaseException) -> bool:
    if not isinstance(exc, ValueError):
        return False
    msg = str(exc).lower()
    return "finite" in msg or "non-finite" in msg


def _epsf_dist_edge_px(x: float, y: float, *, height: int, width: int) -> float:
    return min(float(x), float(y), width - 1 - float(x), height - 1 - float(y))


def _epsf_guard_pick_drop(psf_stars_df: pd.DataFrame, *, image_shape: tuple[int, int]) -> dict[str, Any]:
    """Deterministic edge-nearest star drop for non-finite ePSF build recovery."""
    h, w = int(image_shape[0]), int(image_shape[1])
    work = psf_stars_df.copy()
    xs = pd.to_numeric(work["x"], errors="coerce")
    ys = pd.to_numeric(work["y"], errors="coerce")
    dists = [
        _epsf_dist_edge_px(x, y, height=h, width=w) if math.isfinite(x) and math.isfinite(y) else float("inf")
        for x, y in zip(xs, ys, strict=True)
    ]
    work["_dist_edge"] = dists
    work["_cid"] = work["catalog_id"].map(_norm_catalog_id)
    work = work.sort_values(["_dist_edge", "_cid"], ascending=[True, True], kind="mergesort")
    row = work.iloc[0]
    return {
        "catalog_id": str(row["_cid"]),
        "x": float(row["x"]),
        "y": float(row["y"]),
        "dist_edge_px": float(row["_dist_edge"]),
        "reason": _EPSF_BUILD_GUARD_REASON,
    }


_CONE_CATALOG_NAME = "field_catalog_cone.csv"


def _load_cone_catalog(epsf_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Load the FULL Gaia cone catalogue (ra_deg, dec_deg, mag) for ePSF isolation.

    Returns ``None`` if the cone CSV is absent/unreadable so the caller can fall back to
    the legacy candidate-vs-candidate test.
    """
    p = Path(epsf_dir) / _CONE_CATALOG_NAME
    if not p.is_file():
        return None
    try:
        c = pd.read_csv(
            p, low_memory=False, usecols=lambda col: col in ("ra_deg", "dec_deg", "mag")
        )
        ra = pd.to_numeric(c["ra_deg"], errors="coerce").to_numpy(dtype=float)
        dec = pd.to_numeric(c["dec_deg"], errors="coerce").to_numpy(dtype=float)
        mag = pd.to_numeric(c.get("mag"), errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(ra) & np.isfinite(dec)
        return ra[ok], dec[ok], mag[ok]
    except Exception as e:  # noqa: BLE001
        LOGGER.warning("[ePSF] cone catalogue load failed (%s) - isolation uses candidate set", e)
        return None


def _epsf_augment_candidates_from_detected_pool(
    *,
    mpath: Path,
    csv_ok: pd.DataFrame,
    star_rows: list[dict[str, Any]],
    fwhm_px: float,
    db: VyvarDatabase,
    cfg: Any,
    funnel: dict[str, Any],
) -> tuple[list[dict[str, Any]], bool]:
    """Augment sparse safe-comp joins with COG-style per-frame detected stars.

    Reuses ``scripts/diagnose_psf_elongation_362._select_frame_stars_from_proc`` so
    dense-field ePSF builds share the same bright/isolated star pick as the elongation
    diagnostic (cone isolation, SNR, saturation, per-frame cap).
    """
    diag_path = Path(__file__).resolve().parent.parent / "dev" / "scripts" / "diagnose_psf_elongation_362.py"
    if not diag_path.is_file():
        LOGGER.warning("[ePSF] broad-pool augment: diagnostic script missing at %s", diag_path)
        funnel["n_broad_pool_reason"] = "diagnostic script missing"
        return star_rows, False

    import importlib.util

    spec = importlib.util.spec_from_file_location("vyvar_epsf_diag", diag_path)
    if spec is None or spec.loader is None:
        funnel["n_broad_pool_reason"] = "diagnostic import failed"
        return star_rows, False
    diag = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diag)

    setup_name = mpath.parent.name
    draft_dir = mpath.parent.parent.parent
    aligned_dir = draft_dir / "detrended_aligned" / "lights" / setup_name
    if not aligned_dir.is_dir():
        LOGGER.warning("[ePSF] broad-pool augment: aligned dir missing: %s", aligned_dir)
        funnel["n_broad_pool_reason"] = "aligned dir missing"
        return star_rows, False

    plate_scale = _read_plate_scale_arcsec_px_from_fits(mpath)
    if plate_scale is None or not math.isfinite(plate_scale) or plate_scale <= 0:
        funnel["n_broad_pool_reason"] = "no plate scale"
        return star_rows, False

    from param_resolver import resolve_gain, resolve_read_noise

    with fits.open(mpath, memmap=True) as hd:
        mhdr = hd[0].header
    gain = float(resolve_gain(mhdr, db=db, equipment_id=None, cfg=cfg).value or 1.0)
    rn = float(resolve_read_noise(mhdr, db=db, equipment_id=None, cfg=cfg).value or 10.0)

    half_scaled = int(round(3.0 * float(fwhm_px)))
    fit_size = 2 * half_scaled + 1
    if fit_size % 2 == 0:
        fit_size += 1
    fit_shape = (fit_size, fit_size)

    xy_lookup: dict[str, dict[str, Any]] = {}
    _mag_col = next((c for c in ("mag", "catalog_mag", "phot_g_mean_mag") if c in csv_ok.columns), None)
    for _, r in csv_ok.iterrows():
        cid = str(r.get("_cid", "") or "").strip()
        if not cid:
            continue
        try:
            x_f = float(pd.to_numeric(r.get("x"), errors="coerce"))
            y_f = float(pd.to_numeric(r.get("y"), errors="coerce"))
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(x_f) and math.isfinite(y_f)):
            continue
        ra_v = float(pd.to_numeric(r.get("ra_deg"), errors="coerce"))
        de_v = float(pd.to_numeric(r.get("dec_deg"), errors="coerce"))
        mag_v = float("nan")
        if _mag_col is not None:
            mag_v = float(pd.to_numeric(r.get(_mag_col), errors="coerce"))
        xy_lookup[cid] = {
            "catalog_id": cid,
            "x": x_f,
            "y": y_f,
            "ra_deg": ra_v if math.isfinite(ra_v) else float("nan"),
            "dec_deg": de_v if math.isfinite(de_v) else float("nan"),
            "mag": mag_v if math.isfinite(mag_v) else float("nan"),
        }

    have = {str(r.get("catalog_id", "")).strip() for r in star_rows if str(r.get("catalog_id", "")).strip()}
    broad: dict[str, dict[str, Any]] = {}
    n_picked_raw = 0
    pairs: list[tuple[Path, Path]] = []
    for fits_path in sorted(aligned_dir.glob("proc_*.fits")):
        csv_path = aligned_dir / f"{fits_path.stem}.csv"
        if csv_path.is_file():
            pairs.append((fits_path, csv_path))
    funnel["n_broad_pool_frames"] = int(len(pairs))

    for fits_path, csv_path in pairs:
        try:
            frame_df = pd.read_csv(csv_path, low_memory=False, dtype={"catalog_id": str})
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("[ePSF] broad-pool: skip %s (%s)", csv_path.name, exc)
            continue
        try:
            with fits.open(fits_path, memmap=True) as hd:
                img_shape = hd[0].data.shape
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("[ePSF] broad-pool: skip %s (%s)", fits_path.name, exc)
            continue
        picked = diag._select_frame_stars_from_proc(
            frame_df,
            mpath.parent,
            fwhm_px=float(fwhm_px),
            plate_scale_arcsec_px=float(plate_scale),
            fit_shape=fit_shape,
            gain=gain,
            rn=rn,
            img_shape=img_shape,
        )
        n_picked_raw += int(len(picked))
        for _, st in picked.iterrows():
            cid = str(st.get("_cid", st.get("catalog_id", "")) or "").strip()
            if not cid or cid in have or cid in broad:
                continue
            if cid in xy_lookup:
                broad[cid] = dict(xy_lookup[cid])
            else:
                try:
                    x_f = float(pd.to_numeric(st.get("x"), errors="coerce"))
                    y_f = float(pd.to_numeric(st.get("y"), errors="coerce"))
                    ra_v = float(pd.to_numeric(st.get("ra_deg"), errors="coerce"))
                    de_v = float(pd.to_numeric(st.get("dec_deg"), errors="coerce"))
                except (TypeError, ValueError):
                    continue
                if not (math.isfinite(x_f) and math.isfinite(y_f)):
                    continue
                mag_v = float("nan")
                if _mag_col and _mag_col in st.index:
                    mag_v = float(pd.to_numeric(st.get(_mag_col), errors="coerce"))
                broad[cid] = {
                    "catalog_id": cid,
                    "x": x_f,
                    "y": y_f,
                    "ra_deg": ra_v if math.isfinite(ra_v) else float("nan"),
                    "dec_deg": de_v if math.isfinite(de_v) else float("nan"),
                    "mag": mag_v if math.isfinite(mag_v) else float("nan"),
                }

    funnel["n_broad_pool_picked_raw"] = int(n_picked_raw)
    funnel["n_broad_pool_unique"] = int(len(broad))
    if not broad:
        funnel["n_broad_pool_reason"] = "no stars picked"
        return star_rows, False

    merged = list(star_rows) + list(broad.values())
    funnel["n_after_broad_pool"] = int(len(merged))
    log_event(
        f"[ePSF] Broad detected-star pool: +{len(broad)} unique "
        f"({n_picked_raw} frame picks across {len(pairs)} frames) -> {len(merged)} total"
    )
    LOGGER.info(
        "[ePSF funnel] broad pool: frames=%s picked_raw=%s unique=%s total=%s",
        funnel.get("n_broad_pool_frames"),
        n_picked_raw,
        len(broad),
        len(merged),
    )
    return merged, True


def _epsf_prepare_stars(
    masterstar_fits_path: Path,
    masterstars_csv_path: Path,
    db: VyvarDatabase,
    draft_id: int,
    *,
    min_stars: int | None = None,
    moffat_centroids: pd.DataFrame | None = None,
    exclude_catalog_ids: frozenset[str] | set[str] | None = None,
) -> dict[str, Any]:
    """Select clean, isolated ePSF candidate stars and return sky-subtracted EPSFStars + meta.

    Candidate positions come only from ``masterstars_full_match.csv`` (quality-filtered
    rows with finite ``x``/``y``). The ``db`` parameter is used for FWHM manifest fallback and
    gain/read-noise resolution elsewhere in the ePSF pipeline - not for star selection.

    Conscious widening (MS-SOURCES-RETIRE, 2026-08-21): production ePSF uses the full
    CSV-quality pool instead of the retired MASTER_SOURCES safe-comp subset, targeting, and
    broad-pool augment. PSF-star selection refinement belongs to ePSF-VALID-01.
    """
    mpath = Path(masterstar_fits_path)
    csvpath = Path(masterstars_csv_path)
    if not mpath.is_file():
        raise FileNotFoundError(f"MASTERSTAR FITS not found: {mpath}")
    if not csvpath.is_file():
        raise FileNotFoundError(f"masterstars_full_match.csv not found: {csvpath}")

    try:
        from config import AppConfig

        cfg = AppConfig()
    except Exception:  # noqa: BLE001
        cfg = None

    if min_stars is None:
        min_stars = int(getattr(cfg, "epsf_min_stars", 30) if cfg is not None else 30)
    min_stars = max(10, int(min_stars))

    fwhm_px = get_epsf_fwhm_from_context(mpath, db, draft_id)
    cutout_size = _to_odd_cutout(int(fwhm_px * 5))
    log_event(f"PSF ePSF: FWHM={fwhm_px:.3f} px (clamped context), cutout_size={cutout_size}")

    def _csv_catalog_id_cell(raw: Any) -> str:
        if raw is None:
            return ""
        s = str(raw).strip()
        if not s or s.lower() in ("nan", "none"):
            return ""
        return _norm_catalog_id(raw)

    df = pd.read_csv(csvpath, low_memory=False, converters={"catalog_id": _csv_catalog_id_cell})
    funnel: dict[str, Any] = {"n_csv_input": int(len(df))}
    if "catalog_known_variable" not in df.columns:
        _vsx = (
            df["vsx_known_variable"].map(_scalar_is_explicit_true)
            if "vsx_known_variable" in df.columns
            else pd.Series(False, index=df.index)
        )
        _gvar = (
            df["gaia_dr3_variable_catalog"].map(_scalar_is_explicit_true)
            if "gaia_dr3_variable_catalog" in df.columns
            else pd.Series(False, index=df.index)
        )
        df = df.copy()
        df["catalog_known_variable"] = _vsx | _gvar

    req = ("catalog_id", "catalog_known_variable", "likely_saturated", "photometry_ok", "x", "y")
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(
            f"masterstars_full_match.csv missing required columns {missing} in {csvpath}"
        )

    _funnel_mask = pd.Series(True, index=df.index)
    _funnel_mask &= df["catalog_known_variable"].map(_scalar_is_explicit_false)
    funnel["n_after_variable_excluded"] = int(_funnel_mask.sum())
    _funnel_mask &= df["likely_saturated"].map(_scalar_is_explicit_false)
    funnel["n_after_peak_sat_likely"] = int(_funnel_mask.sum())
    if "is_saturated" in df.columns:
        _funnel_mask &= ~df["is_saturated"].fillna(False).astype(bool)
        funnel["n_after_peak_sat_is_saturated"] = int(_funnel_mask.sum())
    else:
        funnel["n_after_peak_sat_is_saturated"] = funnel["n_after_peak_sat_likely"]
    _funnel_mask &= df["photometry_ok"].map(_scalar_is_explicit_true)
    funnel["n_after_photometry_ok"] = int(_funnel_mask.sum())
    if "is_noisy" in df.columns:
        _funnel_mask &= ~df["is_noisy"].fillna(False).astype(bool)
        funnel["n_after_not_noisy"] = int(_funnel_mask.sum())
    else:
        funnel["n_after_not_noisy"] = funnel["n_after_photometry_ok"]
    if "is_usable" in df.columns:
        _funnel_mask &= df["is_usable"].fillna(False).astype(bool)
        funnel["n_after_usable"] = int(_funnel_mask.sum())
    else:
        funnel["n_after_usable"] = funnel["n_after_not_noisy"]
    funnel["n_after_snr_cut"] = None  # SNR not filtered in _epsf_prepare_stars
    funnel["n_after_snr_cut_reason"] = "not applied in this function"

    csv_mask = (
        df["catalog_known_variable"].map(_scalar_is_explicit_false)
        & df["likely_saturated"].map(_scalar_is_explicit_false)
        & df["photometry_ok"].map(_scalar_is_explicit_true)
    )
    if "is_saturated" in df.columns:
        csv_mask &= ~df["is_saturated"].fillna(False).astype(bool)
    if "is_noisy" in df.columns:
        csv_mask &= ~df["is_noisy"].fillna(False).astype(bool)
    if "is_usable" in df.columns:
        csv_mask &= df["is_usable"].fillna(False).astype(bool)
    csv_ok = df.loc[csv_mask].copy()
    with fits.open(mpath, memmap=True) as _hd_gate:
        _gate_shape = np.asarray(_hd_gate[0].data).shape
    csv_ok = _epsf_apply_build_selection_gates(
        csv_ok,
        platesolve_dir=mpath.parent,
        cutout_size=cutout_size,
        image_shape=(int(_gate_shape[0]), int(_gate_shape[1])),
        funnel=funnel,
    )
    csv_ok["_cid"] = csv_ok["catalog_id"].map(_norm_catalog_id)
    csv_ok = csv_ok[csv_ok["_cid"] != ""]
    funnel["n_csv_with_catalog_id"] = int(len(csv_ok))
    log_event(f"PSF ePSF: CSV filter -> {len(csv_ok)} rows with non-empty catalog_id")
    LOGGER.info(
        "[ePSF funnel] after CSV quality: n_input=%s n_after_sat=%s n_after_photometry_ok=%s n_csv_with_id=%s",
        funnel.get("n_csv_input"),
        funnel.get("n_after_peak_sat_is_saturated"),
        funnel.get("n_after_photometry_ok"),
        funnel.get("n_csv_with_catalog_id"),
    )

    _ra_by_cid: dict[str, float] = {}
    _dec_by_cid: dict[str, float] = {}
    _mag_by_cid: dict[str, float] = {}
    _mag_col = next((c for c in ("mag", "catalog_mag", "phot_g_mean_mag") if c in csv_ok.columns), None)
    if "ra_deg" in csv_ok.columns and "dec_deg" in csv_ok.columns:
        for _, r in csv_ok.iterrows():
            cid = str(r.get("_cid", "") or "").strip()
            if not cid:
                continue
            try:
                ra_v = float(pd.to_numeric(r.get("ra_deg"), errors="coerce"))
                de_v = float(pd.to_numeric(r.get("dec_deg"), errors="coerce"))
            except (TypeError, ValueError):
                continue
            if math.isfinite(ra_v) and math.isfinite(de_v):
                _ra_by_cid[cid] = ra_v
                _dec_by_cid[cid] = de_v
            if _mag_col is not None:
                mg = float(pd.to_numeric(r.get(_mag_col), errors="coerce"))
                if math.isfinite(mg):
                    _mag_by_cid[cid] = mg

    star_rows: list[dict[str, Any]] = []
    for _, r in csv_ok.iterrows():
        cid = str(r.get("_cid", "") or "").strip()
        if not cid:
            continue
        try:
            x_f = float(pd.to_numeric(r.get("x"), errors="coerce"))
            y_f = float(pd.to_numeric(r.get("y"), errors="coerce"))
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(x_f) and math.isfinite(y_f)):
            continue
        star_rows.append(
            {
                "catalog_id": cid,
                "x": x_f,
                "y": y_f,
                "ra_deg": _ra_by_cid.get(cid, float("nan")),
                "dec_deg": _dec_by_cid.get(cid, float("nan")),
                "mag": _mag_by_cid.get(cid, float("nan")),
            }
        )

    funnel["used_broad_pool"] = False
    funnel["n_safe_comp_csv_join"] = int(len(star_rows))

    if len(star_rows) == 0:
        funnel["n_after_csv_select"] = 0
        funnel["n_final"] = 0
        _fmsg = (
            "ePSF candidate funnel: "
            + ", ".join(f"{k}={v}" for k, v in funnel.items())
            + f" - no ePSF candidates after CSV quality filter ({csvpath.name})"
        )
        LOGGER.info("[ePSF funnel] %s", _fmsg)
        raise ValueError(_fmsg)

    psf_stars_df = pd.DataFrame(star_rows)
    n_join = int(len(psf_stars_df))
    funnel["n_after_csv_select"] = n_join
    funnel["n_after_db_join"] = n_join
    funnel["n_after_targeting"] = n_join
    log_event(
        f"PSF ePSF: masterstars_full_match.csv candidates -> {n_join} star positions "
        f"(file: {csvpath.name})"
    )
    LOGGER.info("[ePSF funnel] csv path: n_after_csv_select=%s", n_join)

    if n_join < min_stars:
        funnel["n_final"] = n_join
        _fmsg = (
            f"EPSF build needs at least {min_stars} clean stars from {csvpath.name}; found {n_join}. "
            f"Funnel: {funnel}"
        )
        LOGGER.info("[ePSF funnel] insufficient after CSV select: %s", _fmsg)
        raise ValueError(_fmsg)

    # Phase 2: override centroid positions with Moffat-fitted values
    _n_overridden = 0
    if moffat_centroids is not None and len(moffat_centroids) > 0:
        try:
            _mc = moffat_centroids.copy()
            _mc["_cid"] = _mc["catalog_id"].astype(str).str.strip()
            _mc_ok = _mc[
                (pd.to_numeric(_mc["moffat_fit_ok"], errors="coerce") == 1)
                & (
                    pd.to_numeric(
                        _mc.get("moffat_chi2", pd.Series([999] * len(_mc))), errors="coerce"
                    )
                    < 20.0
                )
            ]
            if len(_mc_ok) >= 10:
                _centroid_override = {
                    str(r["_cid"]): (float(r["moffat_x_fit"]), float(r["moffat_y_fit"]))
                    for _, r in _mc_ok.iterrows()
                    if math.isfinite(float(r.get("moffat_x_fit", float("nan"))))
                    and math.isfinite(float(r.get("moffat_y_fit", float("nan"))))
                }
                for idx, row in psf_stars_df.iterrows():
                    _cid_s = str(row.get("catalog_id", "")).strip()
                    if _cid_s in _centroid_override:
                        psf_stars_df.at[idx, "x"] = _centroid_override[_cid_s][0]
                        psf_stars_df.at[idx, "y"] = _centroid_override[_cid_s][1]
                        _n_overridden += 1
                log_event(
                    f"ePSF Phase 2: {_n_overridden}/{len(psf_stars_df)} "
                    f"centroid positions overridden from Moffat fit"
                )
            else:
                log_event(
                    f"ePSF Phase 2: only {len(_mc_ok)} Moffat-ok stars "
                    f"(need>=10) - using DAO centroids"
                )
        except Exception as _mc_exc:  # noqa: BLE001
            LOGGER.warning("[ePSF] Phase 2 moffat centroid override failed: %s", _mc_exc)

    # Isolation: reject any candidate with a Gaia *cone-catalogue* neighbour within
    # NxFWHM (correct plate scale) that is within ~Deltamag (brighter, or up to Deltamag fainter).
    # Compared against the FULL cone - not just the candidate set - so a bright NON-candidate
    # neighbour correctly disqualifies a candidate (the previous candidate-vs-candidate test
    # let such stars pass). Falls back to candidate-vs-candidate if the cone CSV is missing.
    _isolation_fwhm_mult = 3.0
    _isolation_delta_mag = 2.5
    _isolation_radius_px = float(fwhm_px) * _isolation_fwhm_mult
    plate_scale_arcsec_px = _read_plate_scale_arcsec_px_from_fits(mpath)
    n_before_iso = int(len(psf_stars_df))
    funnel["n_before_isolation"] = n_before_iso
    if plate_scale_arcsec_px is None or not math.isfinite(plate_scale_arcsec_px) or plate_scale_arcsec_px <= 0:
        LOGGER.warning(
            "[ePSF] plate scale unavailable - skipping isolation filter (%.1fxFWHM=%.1fpx)",
            _isolation_fwhm_mult,
            _isolation_radius_px,
        )
        funnel["n_after_isolation"] = n_before_iso
        funnel["n_after_isolation_reason"] = "skipped (no plate scale)"
    elif n_join > 0 and "ra_deg" in psf_stars_df.columns and "dec_deg" in psf_stars_df.columns:
        from sky_separation import angular_distance_deg_vectorized as _angular_distance_deg_vectorized

        radius_deg = _isolation_radius_px * float(plate_scale_arcsec_px) / 3600.0
        self_deg = 0.5 * float(plate_scale_arcsec_px) / 3600.0  # <0.5px counts as the star itself
        _cand_mag = (
            pd.to_numeric(psf_stars_df.get("mag"), errors="coerce").to_numpy(dtype=float)
            if "mag" in psf_stars_df.columns
            else np.full(len(psf_stars_df), np.nan)
        )
        _cone = _load_cone_catalog(mpath.parent)
        _isolated: list[bool] = []
        if _cone is not None:
            cone_ra, cone_dec, cone_mag = _cone
            _n_rej = 0
            for i, row in psf_stars_df.iterrows():
                ra_i = float(row["ra_deg"])
                de_i = float(row["dec_deg"])
                if not (math.isfinite(ra_i) and math.isfinite(de_i)):
                    _isolated.append(False)
                    continue
                cosd = max(math.cos(math.radians(de_i)), 0.2)
                box = (np.abs(cone_dec - de_i) <= radius_deg * 1.5) & (
                    np.abs(cone_ra - ra_i) <= radius_deg * 1.5 / cosd
                )
                if not box.any():
                    _isolated.append(True)
                    continue
                d_deg = _angular_distance_deg_vectorized(ra_i, de_i, cone_ra[box], cone_dec[box])
                m_box = cone_mag[box]
                cand_m = float(_cand_mag[i]) if (i < len(_cand_mag) and math.isfinite(_cand_mag[i])) else float("nan")
                near = (d_deg > self_deg) & (d_deg <= radius_deg)
                if math.isfinite(cand_m):
                    contaminating = near & ((m_box - cand_m) <= _isolation_delta_mag)
                else:
                    contaminating = near  # no candidate mag -> any neighbour disqualifies (conservative)
                iso = not bool(np.any(contaminating))
                _isolated.append(iso)
                if not iso:
                    _n_rej += 1
            psf_stars_df = psf_stars_df.loc[_isolated].reset_index(drop=True)
            funnel["n_after_isolation"] = int(len(psf_stars_df))
            log_event(
                f"[ePSF] Cone isolation ({_isolation_fwhm_mult}xFWHM={_isolation_radius_px:.1f}px, "
                f"Deltamag<={_isolation_delta_mag}): rejected {_n_rej}/{n_before_iso}, kept {len(psf_stars_df)}"
            )
            LOGGER.info(
                "[ePSF funnel] after isolation: n_before=%s n_after=%s rejected=%s",
                n_before_iso,
                funnel["n_after_isolation"],
                _n_rej,
            )
        else:
            _ra_arr = psf_stars_df["ra_deg"].to_numpy(dtype=float)
            _dec_arr = psf_stars_df["dec_deg"].to_numpy(dtype=float)
            for _, row in psf_stars_df.iterrows():
                ra_i = float(row["ra_deg"])
                de_i = float(row["dec_deg"])
                if not (math.isfinite(ra_i) and math.isfinite(de_i)):
                    _isolated.append(False)
                    continue
                dists_px = (
                    _angular_distance_deg_vectorized(ra_i, de_i, _ra_arr, _dec_arr)
                    * 3600.0
                    / float(plate_scale_arcsec_px)
                )
                neighbor_dists = dists_px[dists_px > 0.01]
                _isolated.append(
                    len(neighbor_dists) == 0 or float(neighbor_dists.min()) > _isolation_radius_px
                )
            psf_stars_df = psf_stars_df.loc[_isolated].reset_index(drop=True)
            funnel["n_after_isolation"] = int(len(psf_stars_df))
            LOGGER.info(
                "[ePSF] Isolation (candidate-set fallback, %.1fxFWHM=%.1fpx): %d PSF stars",
                _isolation_fwhm_mult,
                _isolation_radius_px,
                len(psf_stars_df),
            )
            LOGGER.info(
                "[ePSF funnel] after isolation (fallback): n_before=%s n_after=%s",
                n_before_iso,
                funnel["n_after_isolation"],
            )
        if len(psf_stars_df) < min_stars:
            funnel["n_final"] = int(len(psf_stars_df))
            _fmsg = (
                f"EPSF build needs at least {min_stars} isolated stars; found {len(psf_stars_df)}. "
                f"Funnel: {funnel}"
            )
            LOGGER.info("[ePSF funnel] insufficient after isolation: %s", _fmsg)
            raise ValueError(_fmsg)

    if exclude_catalog_ids:
        _excl = {_norm_catalog_id(x) for x in exclude_catalog_ids if str(x).strip()}
        if _excl:
            _n_before_guard = int(len(psf_stars_df))
            psf_stars_df = psf_stars_df[
                ~psf_stars_df["catalog_id"].map(_norm_catalog_id).isin(_excl)
            ].reset_index(drop=True)
            funnel["n_after_guard_exclude"] = int(len(psf_stars_df))
            funnel["n_guard_excluded_this_pass"] = _n_before_guard - int(len(psf_stars_df))
            if len(psf_stars_df) < min_stars:
                funnel["n_final"] = int(len(psf_stars_df))
                _fmsg = (
                    f"EPSF build needs at least {min_stars} stars after guard exclusions; "
                    f"found {len(psf_stars_df)}. Funnel: {funnel}"
                )
                LOGGER.info("[ePSF funnel] insufficient after guard exclude: %s", _fmsg)
                raise ValueError(_fmsg)

    xs = psf_stars_df["x"].to_numpy(dtype=float).tolist()
    ys = psf_stars_df["y"].to_numpy(dtype=float).tolist()
    cat = Table([xs, ys], names=("x", "y"))
    with fits.open(mpath, memmap=True) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float32)

    nd = NDData(data)
    stars = extract_stars(nd, cat, size=cutout_size)
    n_ext = int(getattr(stars, "n_stars", 0))
    funnel["n_after_extract_stars"] = n_ext
    funnel["n_final"] = n_ext
    log_event(f"PSF ePSF: extract_stars retained {n_ext} cutouts (size={cutout_size})")
    LOGGER.info("[ePSF funnel] final: %s", funnel)
    if n_ext < min_stars:
        _fmsg = (
            f"EPSF build needs at least {min_stars} extractable star cutouts; got {n_ext} "
            f"(many positions may lie outside the frame or overlap borders). Funnel: {funnel}"
        )
        LOGGER.info("[ePSF funnel] insufficient extract_stars: %s", _fmsg)
        raise ValueError(_fmsg)

    # Sky-subtract each star cutout to remove background pedestal
    # (EPSFBuilder does not subtract sky - must be done manually)
    try:
        _nd_list: list[NDData] = []
        _x_list: list[float] = []
        _y_list: list[float] = []
        for _s in stars:
            _d = np.array(_s.data, dtype=np.float64)
            _border_mask = np.ones(_d.shape, dtype=bool)
            if _d.shape[0] > 4 and _d.shape[1] > 4:
                _border_mask[2:-2, 2:-2] = False
            _border_vals = _d[_border_mask]
            _finite_border = _border_vals[np.isfinite(_border_vals)]
            _sky = float(np.median(_finite_border)) if len(_finite_border) >= 4 else 0.0
            _d_sub = _d - _sky
            _nd_list.append(NDData(data=_d_sub.astype(np.float32)))

            # photutils EPSFStar.center is (x, y) - verified against the installed photutils.
            _x_list.append(float(_s.center[0]))
            _y_list.append(float(_s.center[1]))

        _cat2 = Table([_x_list, _y_list], names=("x", "y"))
        stars = extract_stars(NDData(data=np.asarray(data, dtype=np.float32)), _cat2, size=cutout_size)
        n_ext2 = int(getattr(stars, "n_stars", 0))
        log_event(f"PSF ePSF: re-extracted {n_ext2} stars for sky-sub processing")

        # Replace each extracted star's data with its sky-subtracted version
        for _i, _s in enumerate(stars):
            try:
                _s._data = _nd_list[_i].data  # noqa: SLF001
            except (AttributeError, IndexError, TypeError, ValueError) as exc:
                from except_fix_counters import get_except_fix_counters

                get_except_fix_counters().psf_epsf_sky_inject_fail += 1
                logging.error(
                    "[ePSF] sky-sub data inject failed star %d: %s",
                    _i,
                    exc,
                )
                raise RuntimeError(f"ePSF sky-sub inject failed for star {_i}") from exc
        log_event("PSF ePSF: applied per-cutout sky subtraction before EPSFBuilder")
    except Exception as _sky_e:  # noqa: BLE001
        LOGGER.warning("[ePSF] per-cutout sky subtraction failed; proceeding without it: %s", _sky_e)

    return {
        "cfg": cfg,
        "mpath": mpath,
        "min_stars": int(min_stars),
        "fwhm_px": float(fwhm_px),
        "cutout_size": int(cutout_size),
        "data": data,
        "stars": stars,
        "n_ext": int(n_ext),
        "n_join": int(n_join),
        "n_before_isolation": int(n_before_iso),
        "n_after_isolation": int(len(psf_stars_df)),
        "n_overridden": int(_n_overridden),
        "isolation_radius_px": float(_isolation_radius_px),
        "plate_scale_arcsec_px": plate_scale_arcsec_px,
        "funnel": funnel,
        "psf_stars_df": psf_stars_df,
    }


def build_epsf_model(
    masterstar_fits_path: Path,
    masterstars_csv_path: Path,
    db: VyvarDatabase,
    draft_id: int,
    *,
    oversampling: int = 2,
    min_stars: int | None = None,
    spatial_order: int | None = None,
    moffat_centroids: pd.DataFrame | None = None,
    sandbox_output_dir: Path | str | None = None,
    meta_extra: dict[str, Any] | None = None,
    smoothing_kernel: str | None = None,
    builder_kwargs: dict[str, Any] | None = None,
) -> Path:
    """Build ePSF from clean comparison stars and write ``masterstar_epsf.fits`` + meta JSON."""
    osamp = max(1, int(oversampling))
    try:
        from config import AppConfig

        cfg = AppConfig()
    except Exception:  # noqa: BLE001
        cfg = None
    if spatial_order is None:
        spatial_order = int(getattr(cfg, "psf_spatial_order", 0) or 0) if cfg is not None else 0
    spatial_order = max(0, min(2, int(spatial_order)))
    _spatial_enabled = bool(getattr(cfg, "psf_spatial_enabled", False)) if cfg is not None else False
    if not _spatial_enabled:
        spatial_order = 0

    guard_dropped: list[dict[str, Any]] = []
    exclude: set[str] = set()
    n_pool_baseline: int | None = None
    prep: dict[str, Any] | None = None
    _built: dict[str, Any] | None = None

    while True:
        prep = _epsf_prepare_stars(
            masterstar_fits_path,
            masterstars_csv_path,
            db,
            draft_id,
            min_stars=min_stars,
            moffat_centroids=moffat_centroids,
            exclude_catalog_ids=frozenset(exclude) if exclude else None,
        )
        if n_pool_baseline is None:
            n_pool_baseline = int(prep.get("n_after_isolation") or prep.get("n_ext") or 0)
        try:
            _built = _epsf_build_imagepsf_from_stars(
                prep["stars"],
                osamp=osamp,
                fwhm_px=float(prep["fwhm_px"]),
                cutout_size=int(prep["cutout_size"]),
                smoothing_kernel=smoothing_kernel,
                builder_kwargs=builder_kwargs,
            )
            break
        except ValueError as exc:
            if not _epsf_is_non_finite_build_error(exc):
                raise
            if n_pool_baseline <= 0:
                raise
            next_drop_n = len(guard_dropped) + 1
            if next_drop_n / float(n_pool_baseline) > _EPSF_BUILD_GUARD_MAX_DROP_FRAC:
                raise RuntimeError(
                    f"ePSF build guard: non-finite build would require dropping "
                    f"{next_drop_n}/{n_pool_baseline} stars "
                    f"(>{_EPSF_BUILD_GUARD_MAX_DROP_FRAC:.0%} of gated pool). "
                    f"Already dropped: {guard_dropped}"
                ) from exc
            drop = _epsf_guard_pick_drop(
                prep["psf_stars_df"],
                image_shape=tuple(np.asarray(prep["data"]).shape),
            )
            guard_dropped.append(drop)
            exclude.add(str(drop["catalog_id"]))
            log_event(
                "ePSF guard: dropping "
                f"{drop['catalog_id']} at ({drop['x']:.1f},{drop['y']:.1f}) "
                f"dist_edge={drop['dist_edge_px']:.1f}px ({drop['reason']})"
            )

    assert prep is not None and _built is not None

    cfg = prep["cfg"]
    mpath = prep["mpath"]
    fwhm_px = prep["fwhm_px"]
    cutout_size = prep["cutout_size"]
    stars = prep["stars"]
    n_ext = prep["n_ext"]
    n_join = prep["n_join"]
    _n_overridden = prep["n_overridden"]
    _isolation_radius_px = prep["isolation_radius_px"]
    plate_scale_arcsec_px = prep["plate_scale_arcsec_px"]
    _build_funnel = dict(prep.get("funnel") or {})
    _qc = _built["qc"]
    _smoothing = _built["smoothing"]
    fit_shape = _built["fit_shape"]
    _norm_factor = _built["norm_factor"]
    _epsf_sum_native = _built["epsf_sum_native"]
    arr = _built["arr"]
    _iteration_curve = list(_built.get("iteration_failure_curve") or [])
    if _qc.get("epsf_nan_fraction") and _qc["epsf_nan_fraction"] > 0.05:
        log_event(f"ePSF QC WARNING: {_qc['epsf_nan_fraction']:.1%} non-finite pixels in ePSF model")
    _ratio = _qc.get("epsf_vs_input_fwhm_ratio")
    if _ratio is not None and (
        _ratio < _EPSF_FWHM_RATIO_WARN_LO or _ratio > _EPSF_FWHM_RATIO_WARN_HI
    ):
        log_event(
            f"ePSF QC WARNING: ePSF/input FWHM ratio={_ratio:.2f} "
            f"(expect {_EPSF_FWHM_RATIO_WARN_LO:.2f}-{_EPSF_FWHM_RATIO_WARN_HI:.2f}) - possible bad ePSF build"
        )
    if _qc.get("epsf_asymmetry") and _qc["epsf_asymmetry"] > 0.1:
        log_event(f"ePSF QC WARNING: ePSF asymmetry={_qc['epsf_asymmetry']:.3f} (>0.1) - coma/tracking")

    out_dir = Path(sandbox_output_dir) if sandbox_output_dir else mpath.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    epsf_path = out_dir / _MASTERSTAR_EPSF_NAME
    meta_path = out_dir / _MASTERSTAR_EPSF_META_NAME

    fits.PrimaryHDU(data=arr).writeto(epsf_path, overwrite=True)

    created = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    meta = {
        "fwhm_px": float(fwhm_px),
        "cutout_size": int(cutout_size),
        "oversampling": int(osamp),
        "spatial_order": int(spatial_order),
        "smoothing_kernel": _smoothing,
        "fit_shape": list(fit_shape),
        "epsf_qc": _qc,
        "epsf_norm_factor": round(float(_norm_factor), 6) if math.isfinite(_norm_factor) else None,
        "epsf_sum_native": round(float(_epsf_sum_native), 6),
        "phase2_moffat_centroids_used": int(_n_overridden) if moffat_centroids is not None else 0,
        "isolation_radius_px": float(_isolation_radius_px),
        "plate_scale_arcsec_px": (
            float(plate_scale_arcsec_px)
            if plate_scale_arcsec_px is not None and math.isfinite(float(plate_scale_arcsec_px))
            else None
        ),
        "n_stars_used": int(n_ext),
        "n_stars_after_join": int(n_join),
        "draft_id": int(draft_id),
        "created_utc": created,
        "build_funnel": _build_funnel,
        "build_selection": {
            "gates": [
                "zone_linear",
                "non_variable",
                "non_saturated",
                "photometry_ok",
                "clean_source_state",
                "science_scope",
                "edge_safe_cutout",
                "isolated",
                "interim_top_n",
            ],
            "interim_top_n": _EPSF_BUILD_INTERIM_TOP_N,
            "interim_top_n_mark": "INTERIM",
            "instrumented_builder": "InstrumentedEPSFBuilder._process_iteration",
        },
        "iteration_failure_curve": _iteration_curve,
        "sandbox_output": bool(sandbox_output_dir),
        "build_guard": {
            "enabled": True,
            "max_drop_frac": _EPSF_BUILD_GUARD_MAX_DROP_FRAC,
            "n_pool_baseline": int(n_pool_baseline or 0),
            "requested_n": int(n_pool_baseline or 0),
            "n_dropped": int(len(guard_dropped)),
            "n_stars_used_after_guard": int(n_ext),
            "dropped": guard_dropped,
        },
        "n_policy": {
            "production_pool": "full Part C gated science-comp pool",
            "certificate_metric": (
                "scale-aligned per-star RMS delta < 3x median ERR budget of compared stars; "
                "raw inter-model offsets are bookkeeping"
            ),
            "interim_top_n": "disabled",
            "split_half_note": (
                "S5b D1b PASS: odd/even gated split-half aligned RMS 30.3 mmag vs 15.3 mmag ERR budget"
            ),
        },
    }
    if meta_extra:
        meta.update(meta_extra)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    log_event(
        f"PSF ePSF: saved {epsf_path.name} shape={arr.shape}, n_stars_used={n_ext}, meta -> {meta_path.name}"
    )
    return epsf_path


def _parse_psf_grid(grid: str) -> tuple[int, int]:
    """Parse a grid spec like ``"3x3"`` -> (nx, ny). Falls back to (3, 3)."""
    try:
        a, b = str(grid).lower().split("x")
        nx = max(1, int(a.strip()))
        ny = max(1, int(b.strip()))
        return nx, ny
    except Exception:  # noqa: BLE001
        return 3, 3


def build_epsf_grid_model(
    masterstar_fits_path: Path,
    masterstars_csv_path: Path,
    db: VyvarDatabase,
    draft_id: int,
    *,
    grid: str = "3x3",
    oversampling: int = 2,
    min_stars_per_cell: int = 25,
    min_stars: int | None = None,
    moffat_centroids: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Build a spatially-varying ePSF over an NxM grid of detector cells (gated, offline).

    Reuses the *same* isolated ePSF-candidate selection as :func:`build_epsf_model`, then
    bins those stars into an ``nx x ny`` grid of detector regions and builds one ePSF per
    cell. Cells with fewer than ``min_stars_per_cell`` isolated stars fall back to the single
    global ePSF (flagged). Returns a dict with:

    - ``gridded_model``: photutils ``GriddedPSFModel`` (or ``None`` if assembly failed)
    - ``cell_arrays``: list of per-cell normalized oversampled ePSF arrays (row-major j*nx+i)
    - ``cell_centers``: list of (x, y) cell centres in detector pixels
    - ``cell_x_centers`` / ``cell_y_centers``: unique grid axis centres (len nx / ny)
    - ``cell_n_stars`` / ``cell_fallback``: per-cell star count and fallback flag
    - ``global_arr``: the single global ePSF (fallback model)
    - ``grid_nx`` / ``grid_ny`` / ``oversampling`` / ``cutout_size`` / ``fwhm_px``

    Does NOT write any production files (validation / gated use only).
    """
    nx, ny = _parse_psf_grid(grid)
    osamp = max(1, int(oversampling))

    prep = _epsf_prepare_stars(
        masterstar_fits_path,
        masterstars_csv_path,
        db,
        draft_id,
        min_stars=min_stars,
        moffat_centroids=moffat_centroids,
    )
    stars = prep["stars"]
    data = prep["data"]
    fwhm_px = prep["fwhm_px"]
    cutout_size = prep["cutout_size"]

    ny_img, nx_img = data.shape[0], data.shape[1]
    star_list = list(stars)
    # photutils EPSFStar.center is (x, y) - verified against the installed photutils.
    centers_xy = [(float(s.center[0]), float(s.center[1])) for s in star_list]

    # Global (single) ePSF - used as fallback for sparse cells.
    global_built = _epsf_build_imagepsf_from_stars(
        stars, osamp=osamp, fwhm_px=fwhm_px, cutout_size=cutout_size
    )
    global_arr = global_built["arr"]

    x_edges = np.linspace(0.0, float(nx_img), nx + 1)
    y_edges = np.linspace(0.0, float(ny_img), ny + 1)
    cell_x_centers = [0.5 * (x_edges[i] + x_edges[i + 1]) for i in range(nx)]
    cell_y_centers = [0.5 * (y_edges[j] + y_edges[j + 1]) for j in range(ny)]

    cell_centers: list[tuple[float, float]] = []
    cell_arrays: list[np.ndarray] = []
    cell_n_stars: list[int] = []
    cell_fallback: list[bool] = []
    cell_qc: list[dict[str, Any]] = []

    for j in range(ny):
        cy0, cy1 = y_edges[j], y_edges[j + 1]
        for i in range(nx):
            cx0, cx1 = x_edges[i], x_edges[i + 1]
            idx = [
                k
                for k, (sx, sy) in enumerate(centers_xy)
                if (cx0 <= sx < cx1 and cy0 <= sy < cy1)
            ]
            cell_centers.append((float(cell_x_centers[i]), float(cell_y_centers[j])))
            built_ok = False
            if len(idx) >= int(min_stars_per_cell):
                try:
                    sub = EPSFStars([star_list[k] for k in idx])
                    cb = _epsf_build_imagepsf_from_stars(
                        sub, osamp=osamp, fwhm_px=fwhm_px, cutout_size=cutout_size
                    )
                    if cb["arr"].shape == global_arr.shape and np.isfinite(cb["arr"]).any():
                        cell_arrays.append(cb["arr"])
                        cell_qc.append(cb["qc"])
                        built_ok = True
                except Exception as _cell_e:  # noqa: BLE001
                    LOGGER.warning("[ePSF-grid] cell (%d,%d) build failed: %s", i, j, _cell_e)
            if not built_ok:
                cell_arrays.append(global_arr)
                cell_qc.append(global_built["qc"])
            cell_n_stars.append(len(idx))
            cell_fallback.append(not built_ok)

    n_fallback = int(sum(cell_fallback))
    log_event(
        f"[ePSF-grid] {nx}x{ny} grid, {len(star_list)} isolated stars; "
        f"per-cell n={cell_n_stars}; fallback cells={n_fallback}/{nx * ny}"
    )

    gridded_model = None
    try:
        imagepsfs = [ImagePSF(np.asarray(a, dtype=np.float64), oversampling=osamp) for a in cell_arrays]
        gridded_model = grid_from_epsfs(imagepsfs, grid_xypos=cell_centers)
    except Exception as _grid_e:  # noqa: BLE001
        LOGGER.warning("[ePSF-grid] grid_from_epsfs failed (interpolation still available): %s", _grid_e)

    return {
        "gridded_model": gridded_model,
        "cell_arrays": cell_arrays,
        "cell_centers": cell_centers,
        "cell_x_centers": cell_x_centers,
        "cell_y_centers": cell_y_centers,
        "cell_n_stars": cell_n_stars,
        "cell_fallback": cell_fallback,
        "cell_qc": cell_qc,
        "global_arr": global_arr,
        "global_qc": global_built["qc"],
        "grid_nx": nx,
        "grid_ny": ny,
        "n_fallback": n_fallback,
        "oversampling": osamp,
        "cutout_size": int(cutout_size),
        "fwhm_px": float(fwhm_px),
        "n_isolated": int(len(star_list)),
        "img_shape": (int(ny_img), int(nx_img)),
    }


def interp_gridded_epsf_array(grid: dict[str, Any], x: float, y: float) -> np.ndarray:
    """Bilinearly interpolate the per-cell ePSF stamps at detector position (x, y).

    This reproduces photutils ``GriddedPSFModel`` bilinear interpolation directly on the
    oversampled cell stamps so a position-appropriate ImagePSF can be used in the existing
    per-cutout fitting architecture. Positions outside the cell-centre hull clamp to the edge.
    """
    xs = grid["cell_x_centers"]
    ys = grid["cell_y_centers"]
    nx = int(grid["grid_nx"])
    arrs = grid["cell_arrays"]

    def _bracket(vals: list[float], q: float) -> tuple[int, int, float]:
        n = len(vals)
        if n == 1:
            return 0, 0, 0.0
        if q <= vals[0]:
            return 0, 0, 0.0
        if q >= vals[-1]:
            return n - 1, n - 1, 0.0
        hi = 1
        while hi < n and vals[hi] < q:
            hi += 1
        lo = hi - 1
        span = vals[hi] - vals[lo]
        t = (q - vals[lo]) / span if span > 0 else 0.0
        return lo, hi, float(t)

    i0, i1, tx = _bracket(list(xs), float(x))
    j0, j1, ty = _bracket(list(ys), float(y))
    A = arrs[j0 * nx + i0]
    B = arrs[j0 * nx + i1]
    C = arrs[j1 * nx + i0]
    D = arrs[j1 * nx + i1]
    top = A * (1.0 - tx) + B * tx
    bot = C * (1.0 - tx) + D * tx
    return np.asarray(top * (1.0 - ty) + bot * ty, dtype=np.float32)


def fit_moffat_psf_stars(
    frame_data: np.ndarray,
    frame_hdr: Any,
    star_positions: pd.DataFrame,
    *,
    fwhm_guess_px: float = 3.5,
    cutout_size: int | None = None,
    error: np.ndarray | None = None,
    fix_alpha: float | None = None,
    alpha_bounds: tuple[float, float] = (2.0, 8.0),
    chi2_limit: float = 50.0,
    saturate_limit_adu: float | None = None,
    peak_col: str = "peak_dao",
) -> pd.DataFrame:
    """Per-star Moffat PSF fit on frame cutouts.

    Step 1 of two-step PSF pipeline. Returns one row per input star with:
      catalog_id, x, y,
      moffat_flux, moffat_flux_err,
      moffat_gamma, moffat_alpha, moffat_fwhm_px,
      moffat_sky, moffat_x_fit, moffat_y_fit,
      moffat_chi2, moffat_fit_ok,
      moffat_x_err, moffat_y_err

    Parameters
    ----------
    frame_data      : 2-D float array (full frame, sky NOT subtracted)
    frame_hdr       : FITS header (unused currently, reserved)
    star_positions  : DataFrame with columns catalog_id, x, y
    fwhm_guess_px   : initial FWHM guess in pixels (from VY_FWHM or ePSF meta)
    cutout_size     : odd int; if None -> max(15, int(fwhm_guess_px * 6 + 1))
                      (larger than ePSF cutout to capture wings)
    error           : per-pixel error map (same shape as frame_data); if None
                      -> estimated from cutout border std per star
    fix_alpha       : if float, fix Moffat alpha (beta) at this value;
                      if None -> fit alpha as free parameter
    alpha_bounds    : (min, max) for alpha when fitting freely
    chi2_limit      : reduced chi2 threshold for moffat_fit_ok=True
    """
    _ = frame_hdr  # reserved for future metadata

    _MOFFAT_COLS = [
        "catalog_id",
        "x",
        "y",
        "moffat_flux",
        "moffat_flux_err",
        "moffat_gamma",
        "moffat_alpha",
        "moffat_fwhm_px",
        "moffat_sky",
        "moffat_sky_resid",
        "moffat_saturated",
        "moffat_x_fit",
        "moffat_y_fit",
        "moffat_x_err",
        "moffat_y_err",
        "moffat_chi2",
        "moffat_fit_ok",
    ]
    _NAN_ROW_EXTRA = {
        c: float("nan") for c in _MOFFAT_COLS if c not in ("catalog_id", "x", "y", "moffat_fit_ok", "moffat_saturated")
    }

    if star_positions is None or len(star_positions) == 0:
        return pd.DataFrame(columns=_MOFFAT_COLS)

    frame_data = np.asarray(frame_data, dtype=np.float64)
    h, w = frame_data.shape

    if cutout_size is None:
        _cs = max(15, int(math.ceil(float(fwhm_guess_px) * 6.0 + 1.0)))
        if _cs % 2 == 0:
            _cs += 1
        cutout_size = _cs

    _alpha_init = float(fix_alpha) if fix_alpha is not None else 3.5
    try:
        _gamma_init = float(
            float(fwhm_guess_px) / (2.0 * math.sqrt(2.0 ** (1.0 / _alpha_init) - 1.0))
        )
        _gamma_init = max(0.5, _gamma_init)
    except Exception:  # noqa: BLE001
        _gamma_init = float(fwhm_guess_px) / 2.0

    fitter = LevMarLSQFitter()

    out_rows: list[dict[str, Any]] = []
    half = int(cutout_size) // 2

    for _, srow in star_positions.iterrows():
        cid = str(srow.get("catalog_id", ""))
        sx = float(srow.get("x", float("nan")))
        sy = float(srow.get("y", float("nan")))

        base: dict[str, Any] = {
            "catalog_id": cid,
            "x": sx,
            "y": sy,
            "moffat_fit_ok": False,
            "moffat_saturated": False,
        }
        base.update(_NAN_ROW_EXTRA)

        if not (math.isfinite(sx) and math.isfinite(sy)):
            out_rows.append(base)
            continue

        # Skip saturated stars - Moffat profile invalid for saturated cores
        if saturate_limit_adu is not None:
            _peak = float(srow.get(peak_col, float("nan")))
            if math.isfinite(_peak) and _peak >= float(saturate_limit_adu) * 0.80:
                base["moffat_saturated"] = True
                base["moffat_fit_ok"] = False
                base["moffat_chi2"] = float("nan")
                out_rows.append(base)
                continue

        x0 = max(0, int(sx) - half)
        x1 = min(w, int(sx) + half + 1)
        y0 = max(0, int(sy) - half)
        y1 = min(h, int(sy) + half + 1)
        cut = frame_data[y0:y1, x0:x1]

        if cut.size == 0 or cut.shape[0] < 5 or cut.shape[1] < 5:
            out_rows.append(base)
            continue

        _border_mask = np.ones(cut.shape, dtype=bool)
        if cut.shape[0] > 4 and cut.shape[1] > 4:
            _border_mask[2:-2, 2:-2] = False
        _border_vals = cut[_border_mask]
        _finite_border = _border_vals[np.isfinite(_border_vals)]
        _sky = float(np.median(_finite_border)) if len(_finite_border) >= 4 else 0.0
        cut_sub = cut - _sky

        if float(np.nanmax(cut_sub)) <= 0:
            out_rows.append(base)
            continue

        if error is not None:
            err_cut = np.asarray(error, dtype=np.float64)[y0:y1, x0:x1]
        else:
            _noise = float(np.std(_finite_border)) if len(_finite_border) >= 8 else 1.0
            _noise = max(1.0, _noise)
            err_cut = np.full(cut_sub.shape, _noise, dtype=np.float64)

        _pos = err_cut > 0
        if bool(np.any(_pos)):
            _med_pos = float(np.nanmedian(err_cut[_pos]))
            if not math.isfinite(_med_pos) or _med_pos <= 0:
                _med_pos = 1.0
        else:
            _med_pos = 1.0
        err_cut = np.where(err_cut > 0, err_cut, _med_pos).astype(np.float64, copy=False)

        xc = sx - x0
        yc = sy - y0
        flux_guess = float(np.nansum(cut_sub.clip(min=0)))
        if not math.isfinite(flux_guess) or flux_guess <= 0:
            flux_guess = float(np.nanmax(cut_sub)) * float(cutout_size)

        try:
            # Compound model: Moffat + residual local sky constant
            _amp_init = float(np.nanmax(cut_sub)) if float(np.nanmax(cut_sub)) > 0 else 1.0
            moffat_component = Moffat2D(
                amplitude=_amp_init,
                x_0=xc,
                y_0=yc,
                gamma=_gamma_init,
                alpha=_alpha_init,  # power index
            )
            sky_component = Const2D(amplitude=0.0)
            sky_component.amplitude.min = -3000.0
            sky_component.amplitude.max = 3000.0

            if fix_alpha is not None:
                moffat_component.alpha.fixed = True
            else:
                moffat_component.alpha.min = float(alpha_bounds[0])
                moffat_component.alpha.max = float(alpha_bounds[1])

            moffat_component.gamma.min = 0.1
            moffat_component.gamma.max = float(cutout_size)
            moffat_component.amplitude.min = 0.0

            compound_model = moffat_component + sky_component

            _ny, _nx = cut_sub.shape
            _y_grid, _x_grid = np.mgrid[0:_ny, 0:_nx]

            _sigma = np.clip(err_cut, 1e-6, None)
            _weights = 1.0 / _sigma

            fitted_compound = fitter(
                compound_model,
                _x_grid,
                _y_grid,
                cut_sub,
                weights=_weights,
                maxiter=300,
            )

            fitted_moffat = fitted_compound[0]
            fitted_sky = fitted_compound[1]

            amp_fit = float(fitted_moffat.amplitude.value)
            x_fit_cut = float(fitted_moffat.x_0.value)
            y_fit_cut = float(fitted_moffat.y_0.value)
            gamma_fit = float(abs(fitted_moffat.gamma.value))
            alpha_fit = float(fitted_moffat.alpha.value)
            sky_resid = float(fitted_sky.amplitude.value)
            fwhm_fit = _moffat_fwhm_px(gamma_fit, alpha_fit)

            # Use numerical sum on cutout grid (avoids integral blow-up for low alpha)
            # Subtract sky component from model evaluation
            _moffat_only = fitted_moffat(_x_grid, _y_grid)
            flux_fit = float(np.sum(np.clip(_moffat_only, 0.0, None)))
            # Store also the analytical integral for reference (in meta only)
            if alpha_fit > 1.1 and gamma_fit > 0 and amp_fit > 0:
                _flux_analytic = float(math.pi * amp_fit * gamma_fit**2 / (alpha_fit - 1.0))
            else:
                _flux_analytic = float("nan")

            _model_vals = fitted_compound(_x_grid, _y_grid)
            _residuals = cut_sub - _model_vals
            _chi2_raw = float(np.nansum((_residuals / _sigma) ** 2))
            _n_free = max(1, int(cut_sub.size) - 6)
            chi2 = float(_chi2_raw / _n_free)

            fit_info = fitter.fit_info
            if (
                fit_info is not None
                and "param_cov" in fit_info
                and fit_info["param_cov"] is not None
                and hasattr(fit_info["param_cov"], "shape")
            ):
                try:
                    _cov = fit_info["param_cov"]
                    _amp_var = float(_cov[0, 0]) if _cov.shape[0] > 0 else 0.0
                    _gam_var = float(_cov[3, 3]) if _cov.shape[0] > 3 else 0.0
                    _den = max(alpha_fit - 1.0, 0.01)
                    _dFdA = math.pi * gamma_fit**2 / _den
                    _dFdG = 2.0 * math.pi * amp_fit * gamma_fit / _den
                    flux_err = float(
                        math.sqrt(max(0.0, _dFdA**2 * _amp_var + _dFdG**2 * _gam_var))
                    )
                except Exception:  # noqa: BLE001
                    flux_err = float("nan")
            else:
                flux_err = float("nan")

            chi2_ok = math.isfinite(chi2) and chi2 < float(chi2_limit)
            _alpha_at_bound = abs(alpha_fit - float(alpha_bounds[0])) < 0.05
            fit_ok = bool(
                chi2_ok
                and math.isfinite(flux_fit)
                and flux_fit > 0
                and math.isfinite(gamma_fit)
                and gamma_fit > 0
                and math.isfinite(alpha_fit)
                and alpha_fit > 1.0
                and not _alpha_at_bound
                and math.isfinite(fwhm_fit)
                and 0.5 < fwhm_fit < float(cutout_size)
            )

            out_rows.append(
                {
                    "catalog_id": cid,
                    "x": sx,
                    "y": sy,
                    "moffat_flux": flux_fit,
                    "moffat_flux_err": flux_err,
                    "moffat_gamma": gamma_fit,
                    "moffat_alpha": alpha_fit,
                    "moffat_fwhm_px": fwhm_fit,
                    "moffat_sky": _sky + sky_resid,
                    "moffat_sky_resid": sky_resid,
                    "moffat_x_fit": x_fit_cut + x0,
                    "moffat_y_fit": y_fit_cut + y0,
                    "moffat_x_err": float("nan"),
                    "moffat_y_err": float("nan"),
                    "moffat_chi2": chi2,
                    "moffat_fit_ok": fit_ok,
                }
            )
        except Exception as exc:  # noqa: BLE001
            log_event(f"Moffat fit failed for {cid} at ({sx:.1f},{sy:.1f}): {exc}")
            out_rows.append(base)

    df = pd.DataFrame(out_rows, columns=_MOFFAT_COLS)

    n_ok = int(df["moffat_fit_ok"].sum()) if "moffat_fit_ok" in df.columns else 0
    if len(df) > 0:
        fwhm_med = df.loc[df["moffat_fit_ok"] == True, "moffat_fwhm_px"].median()  # noqa: E712
        alpha_med = df.loc[df["moffat_fit_ok"] == True, "moffat_alpha"].median()  # noqa: E712
        log_event(
            f"Moffat PSF fit: {n_ok}/{len(df)} ok, "
            f"FWHM_median={float(fwhm_med):.2f}px, "
            f"alpha_median={float(alpha_med):.2f}"
        )

    return df


def _aperture_annulus_radii_px(fwhm_px: float) -> tuple[float, float, float]:
    """``r_ap``, ``r_in``, ``r_out`` matching ``photometry_core`` catalog-only aperture path."""
    try:
        from config import AppConfig

        cfg = AppConfig()
        af = float(cfg.aperture_fwhm_factor)
        ai = float(cfg.annulus_inner_fwhm)
        ao = float(cfg.annulus_outer_fwhm)
    except Exception:  # noqa: BLE001
        af, ai, ao = 1.9, 4.75, 9.0
    fw = float(fwhm_px)
    r_ap = max(0.5, af * fw)
    r_in = max(r_ap + 0.5, ai * fw)
    r_out = max(r_in + 0.5, ao * fw)
    return r_ap, r_in, r_out


def _border_median_sky_from_cutout(cut: np.ndarray) -> float:
    """Legacy 2-pixel border median on a fit cutout (fallback when annulus is infeasible)."""
    border_mask = np.ones(cut.shape, dtype=bool)
    if cut.shape[0] > 4 and cut.shape[1] > 4:
        border_mask[2:-2, 2:-2] = False
    border_vals = cut[border_mask]
    finite = border_vals[np.isfinite(border_vals)]
    if len(finite) >= 8:
        return float(np.median(finite))
    return float(np.nanmedian(cut))


def _psf_annulus_radii_px(
    fwhm_px: float,
    *,
    inner_fwhm: float | None = None,
    outer_fwhm: float | None = None,
) -> tuple[float, float, float]:
    """Annulus radii for PSF sky (defaults match aperture path; overrides PSF-only)."""
    r_ap, r_in, r_out = _aperture_annulus_radii_px(fwhm_px)
    fw = float(fwhm_px)
    if inner_fwhm is not None and fw > 0:
        r_in = max(r_ap + 0.5, float(inner_fwhm) * fw)
    if outer_fwhm is not None and fw > 0:
        r_out = max(r_in + 0.5, float(outer_fwhm) * fw)
    return r_ap, r_in, r_out


def _annulus_median_per_px(
    frame_data: np.ndarray,
    x: float,
    y: float,
    *,
    r_in: float,
    r_out: float,
) -> float:
    """Median ADU/px in a circular annulus (PSF path; does not touch aperture photometry)."""
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(r_in) and math.isfinite(r_out)):
        return float("nan")
    if r_out <= r_in:
        return float("nan")
    try:
        from photutils.aperture import CircularAnnulus
    except ImportError:
        return float("nan")
    d = np.asarray(frame_data, dtype=np.float64)
    if np.any(~np.isfinite(d)):
        fill = float(np.nanmedian(d)) if np.any(np.isfinite(d)) else 0.0
        d = np.where(np.isfinite(d), d, fill)
    pos = np.array([[float(x), float(y)]], dtype=np.float64)
    an = CircularAnnulus(pos, r_in=float(r_in), r_out=float(r_out))
    ann_masks = an.to_mask(method="center")
    if not isinstance(ann_masks, (list, tuple)):
        ann_masks = [ann_masks]
    for amask in ann_masks:
        try:
            cut = np.asarray(amask.get_values(d), dtype=np.float64).ravel()
            cut = cut[np.isfinite(cut)]
            if cut.size >= 8:
                return float(np.median(cut))
        except Exception:  # noqa: BLE001
            continue
    return float("nan")


def _annulus_sky_per_px_custom(
    frame_data: np.ndarray,
    x: float,
    y: float,
    *,
    fwhm_px: float,
    inner_fwhm: float,
    outer_fwhm: float,
) -> tuple[float, str]:
    """PSF-only annulus sky with explicit inner/outer FWHM multipliers (option A sweep)."""
    if not (math.isfinite(fwhm_px) and fwhm_px > 0):
        return float("nan"), "border_fallback"
    h, w = frame_data.shape
    _, r_in, r_out = _psf_annulus_radii_px(
        fwhm_px, inner_fwhm=inner_fwhm, outer_fwhm=outer_fwhm
    )
    margin = 0.5
    if not (
        x - r_out >= margin
        and y - r_out >= margin
        and x + r_out <= w - 1 - margin
        and y + r_out <= h - 1 - margin
    ):
        return float("nan"), "border_fallback"
    sky = _annulus_median_per_px(frame_data, x, y, r_in=r_in, r_out=r_out)
    if math.isfinite(sky):
        return sky, f"annulus_r{inner_fwhm:.1f}_{outer_fwhm:.1f}fwhm"
    return float("nan"), "border_fallback"


def _subtract_psf_models(
    frame_data: np.ndarray,
    psf_model: Any,
    sources: list[tuple[float, float, float]],
) -> np.ndarray:
    """Return ``frame - sum(flux_i * PSF_i)`` on the full image."""
    d = np.asarray(frame_data, dtype=np.float64)
    residual = d.copy()
    if not sources:
        return residual
    h, w = d.shape
    yy, xx = np.mgrid[0:h, 0:w]
    xg = xx.astype(np.float64)
    yg = yy.astype(np.float64)
    for sx, sy, sflux in sources:
        if not (math.isfinite(sx) and math.isfinite(sy) and math.isfinite(sflux) and sflux > 0):
            continue
        shape = (h, w)
        model = psf_model.evaluate(
            xg,
            yg,
            np.full(shape, float(sflux), dtype=np.float64),
            np.full(shape, float(sx), dtype=np.float64),
            np.full(shape, float(sy), dtype=np.float64),
        )
        residual -= np.asarray(model, dtype=np.float64)
    return residual


def _residual_annulus_sky_per_px(
    frame_data: np.ndarray,
    x: float,
    y: float,
    *,
    fwhm_px: float,
    psf_model: Any,
    sources: list[tuple[float, float, float]],
    inner_fwhm: float | None = None,
    outer_fwhm: float | None = None,
) -> tuple[float, str]:
    """Sky from annulus on (data - fitted PSF models); wing- and neighbour-clean (option C)."""
    if not (math.isfinite(fwhm_px) and fwhm_px > 0):
        return float("nan"), "border_fallback"
    _, r_in, r_out = _psf_annulus_radii_px(
        fwhm_px, inner_fwhm=inner_fwhm, outer_fwhm=outer_fwhm
    )
    h, w = frame_data.shape
    margin = 0.5
    if not (
        x - r_out >= margin
        and y - r_out >= margin
        and x + r_out <= w - 1 - margin
        and y + r_out <= h - 1 - margin
    ):
        return float("nan"), "border_fallback"
    residual = _subtract_psf_models(frame_data, psf_model, sources)
    sky = _annulus_median_per_px(residual, x, y, r_in=r_in, r_out=r_out)
    if math.isfinite(sky):
        return sky, "residual_annulus"
    return float("nan"), "border_fallback"


def _annulus_sky_per_px_full_frame(
    frame_data: np.ndarray,
    x: float,
    y: float,
    *,
    fwhm_px: float,
) -> tuple[float, str]:
    """Aperture-consistent local sky on the full image (same annulus as catalog aperture path)."""
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(fwhm_px) and fwhm_px > 0):
        return float("nan"), "border_fallback"
    h, w = frame_data.shape
    r_ap, r_in, r_out = _aperture_annulus_radii_px(fwhm_px)
    margin = 0.5
    if not (
        x - r_out >= margin
        and y - r_out >= margin
        and x + r_out <= w - 1 - margin
        and y + r_out <= h - 1 - margin
    ):
        return float("nan"), "border_fallback"
    try:
        from photometry_core import _annulus_sky_subtracted_flux

        _, sky_pp, _ = _annulus_sky_subtracted_flux(
            np.asarray(frame_data, dtype=np.float64),
            float(x),
            float(y),
            r_ap,
            r_in,
            r_out,
        )
        if math.isfinite(sky_pp):
            return float(sky_pp), "annulus_local"
    except (ImportError, ValueError, TypeError, IndexError) as exc:
        from except_fix_counters import get_except_fix_counters

        get_except_fix_counters().psf_local_sky_fail += 1
        logging.error(
            "[PSF] local annulus sky failed x=%.2f y=%.2f: %s",
            float(x),
            float(y),
            exc,
        )


def _psf_resolve_gain_read_noise(frame_hdr: Any) -> tuple[float, float]:
    """Gain (e-/ADU) and read noise (e-) for PSF fit-weight construction."""
    try:
        from config import AppConfig
        from param_resolver import resolve_gain, resolve_read_noise

        cfg = AppConfig()
        gain = float(resolve_gain(frame_hdr, cfg=cfg).value or 1.0)
        rn = float(resolve_read_noise(frame_hdr, cfg=cfg).value or 10.0)
    except Exception:  # noqa: BLE001
        gain, rn = 1.0, 10.0
    if not math.isfinite(gain) or gain <= 0:
        gain = 1.0
    if not math.isfinite(rn) or rn < 0:
        rn = 10.0
    return gain, rn


def _psf_sky_only_sigma_per_px(sky_per_px_adu: float, gain: float, read_noise_e: float) -> float:
    """Brightness-independent per-pixel sigma (ADU): sky Poisson + read noise only."""
    sky = max(0.0, float(sky_per_px_adu)) if math.isfinite(sky_per_px_adu) else 0.0
    g = max(1e-6, float(gain))
    rn = max(0.0, float(read_noise_e))
    var = sky / g + (rn / g) ** 2
    return float(math.sqrt(max(var, 1e-12)))


def _psf_variance_per_px_adu2(
    f_model_per_px: np.ndarray,
    *,
    sky_per_px: float,
    gain: float,
    read_noise_e: float,
) -> np.ndarray:
    """Full CCD variance per pixel (ADU^2): F_model/g + sky/g + (RN/g)^2."""
    g = max(1e-6, float(gain))
    rn = max(0.0, float(read_noise_e))
    sky = max(0.0, float(sky_per_px)) if math.isfinite(sky_per_px) else 0.0
    f = np.maximum(np.asarray(f_model_per_px, dtype=np.float64), 0.0)
    return np.maximum(f / g + sky / g + (rn / g) ** 2, 1e-12)


def _psf_model_prediction_cutout(
    psf_model: Any,
    cut_shape: tuple[int, ...],
    flux: float,
    x_0: float,
    y_0: float,
) -> np.ndarray:
    """Per-pixel model prediction (ADU) for one-pass error-map construction."""
    if not (math.isfinite(flux) and flux > 0 and math.isfinite(x_0) and math.isfinite(y_0)):
        return np.zeros(cut_shape, dtype=np.float64)
    h, w = int(cut_shape[0]), int(cut_shape[1])
    yy, xx = np.mgrid[0:h, 0:w]
    xg = xx.astype(np.float64).ravel()
    yg = yy.astype(np.float64).ravel()
    npx = xg.size
    try:
        model = psf_model.evaluate(
            xg,
            yg,
            np.full(npx, float(flux), dtype=np.float64),
            np.full(npx, float(x_0), dtype=np.float64),
            np.full(npx, float(y_0), dtype=np.float64),
        )
    except Exception as exc:  # noqa: BLE001
        logging.error("[PSF] model evaluate for error map failed: %s", exc)
        return np.zeros(cut_shape, dtype=np.float64)
    return np.maximum(np.asarray(model, dtype=np.float64).reshape(cut_shape), 0.0)


def _psf_flux_init_for_error_map(dao_flux: float, flux_guess: float) -> float:
    """Prefer DAO aperture flux for one-pass F_model; fall back to cutout guess."""
    if math.isfinite(dao_flux) and dao_flux > 0:
        return float(dao_flux)
    if math.isfinite(flux_guess) and flux_guess > 0:
        return float(flux_guess)
    return 1.0


_PSF_WEIGHT_MODE_FULL = "full_ccd"
_PSF_ERR_MODE_FULL = "sandwich_full_ccd"


def _psf_fit_error_cutout(
    cut_shape: tuple[int, ...],
    *,
    sky_per_px: float,
    gain: float,
    read_noise_e: float,
    err_full_cut: np.ndarray | None = None,
) -> np.ndarray:
    """Uniform sky+read-noise error map for photutils PSF fit weights."""
    sigma = _psf_sky_only_sigma_per_px(sky_per_px, gain, read_noise_e)
    err_fit = np.full(cut_shape, sigma, dtype=np.float64)
    if err_full_cut is not None and err_full_cut.shape == cut_shape:
        pos = err_fit > 0
        if np.any(pos):
            err_fit = np.where(pos, err_fit, sigma).astype(np.float64, copy=False)
    return err_fit


def _psf_fit_error_cutout_full_ccd(
    cut_shape: tuple[int, ...],
    *,
    psf_model: Any,
    flux_init: float,
    x_0: float,
    y_0: float,
    sky_per_px: float,
    gain: float,
    read_noise_e: float,
    err_full_cut: np.ndarray | None = None,
) -> np.ndarray:
    """Per-pixel sigma for photutils PSF fit (full CCD variance, one-pass F_model from DAO flux)."""
    f_map = _psf_model_prediction_cutout(psf_model, cut_shape, flux_init, x_0, y_0)
    var = _psf_variance_per_px_adu2(
        f_map, sky_per_px=sky_per_px, gain=gain, read_noise_e=read_noise_e
    )
    err_fit = np.sqrt(var)
    if err_full_cut is not None and err_full_cut.shape == cut_shape:
        pos = np.isfinite(err_full_cut) & (err_full_cut > 0)
        if np.any(pos):
            err_fit = np.where(pos, err_fit, np.sqrt(var)).astype(np.float64, copy=False)
    return err_fit.astype(np.float64, copy=False)


def _psf_fit_region_mask(
    shape: tuple[int, ...],
    cy: float,
    cx: float,
    fit_shape: tuple[int, int],
) -> np.ndarray:
    """Boolean mask for the PSF fit window centered on (cx, cy) in cutout coords."""
    mask = np.zeros(shape, dtype=bool)
    fh = max(int(fit_shape[0]), int(fit_shape[1])) // 2
    iy = int(round(cy))
    ix = int(round(cx))
    y0 = max(0, iy - fh)
    y1 = min(shape[0], iy + fh + 1)
    x0 = max(0, ix - fh)
    x1 = min(shape[1], ix + fh + 1)
    mask[y0:y1, x0:x1] = True
    return mask


def _psf_sandwich_flux_err(
    flux_fit: float,
    psf_model: Any,
    x_fit: float,
    y_fit: float,
    cut_shape: tuple[int, ...],
    *,
    sky_per_px: float,
    gain: float,
    read_noise_e: float,
    fit_shape: tuple[int, int],
) -> float:
    """Sandwich SE for full-CCD-weighted PSF flux (weights match per-pixel variance)."""
    if not (math.isfinite(flux_fit) and flux_fit > 0 and math.isfinite(x_fit) and math.isfinite(y_fit)):
        return float("nan")
    fh_y = int(fit_shape[0]) // 2
    fh_x = int(fit_shape[1]) // 2
    iy = int(round(y_fit))
    ix = int(round(x_fit))
    y0 = max(0, iy - fh_y)
    y1 = min(int(cut_shape[0]), iy + fh_y + 1)
    x0 = max(0, ix - fh_x)
    x1 = min(int(cut_shape[1]), ix + fh_x + 1)
    if y1 <= y0 or x1 <= x0:
        return float("nan")
    f_map = _psf_model_prediction_cutout(psf_model, cut_shape, flux_fit, x_fit, y_fit)
    p_map = _psf_model_prediction_cutout(psf_model, cut_shape, 1.0, x_fit, y_fit)
    var = _psf_variance_per_px_adu2(
        f_map, sky_per_px=sky_per_px, gain=gain, read_noise_e=read_noise_e
    )
    p = p_map[y0:y1, x0:x1].ravel()
    v = var[y0:y1, x0:x1].ravel()
    pos = (p > 0) & (v > 0) & np.isfinite(p) & np.isfinite(v)
    if not np.any(pos):
        return float("nan")
    wp2 = (p[pos] * p[pos]) / v[pos]
    denom = float(np.sum(wp2))
    if denom <= 0 or not math.isfinite(denom):
        return float("nan")
    return float(1.0 / math.sqrt(denom))


def _apply_psf_fixed_position(phot: Any, *, fix: bool) -> None:
    """Fix PSF centroid to init (forced photometry; Guy et al. 2010 / Lacroix et al. 2025)."""
    if not fix:
        return
    try:
        phot.psf_model.x_0.fixed = True
        phot.psf_model.y_0.fixed = True
    except Exception as exc:  # noqa: BLE001
        logging.error('[EXC-0454] PSF centroid fixed-position flags not set - forced photometry may drift centroid and ch...: %s', exc)
        pass


def _resolve_psf_fit_sky(
    frame_data: np.ndarray,
    cut: np.ndarray,
    x: float,
    y: float,
    *,
    fwhm_px: float,
) -> tuple[float, str]:
    """Prefer aperture-geometry annulus sky; fall back to cutout border median."""
    if fwhm_px > 0:
        sky, method = _annulus_sky_per_px_full_frame(frame_data, x, y, fwhm_px=fwhm_px)
        if method == "annulus_local" and math.isfinite(sky):
            return sky, method
    return _border_median_sky_from_cutout(cut), "border_fallback"


def _grouped_psf_fit(
    frame_data: np.ndarray,
    err_full: np.ndarray | None,
    x: float,
    y: float,
    *,
    fwhm_px: float,
    fit_shape: tuple[int, int],
    psf_model: Any,
    neighbor_xy: np.ndarray,
    neighbor_flux: np.ndarray,
    group_sep_fwhm: float,
    neighbor_include_fwhm: float,
    chi2_limit: float,
    frame_hdr: Any = None,
) -> dict[str, Any] | None:
    """Joint (deblended) PSF fit of a target plus its close neighbours.

    Builds an init-params table containing the target and every neighbour within
    ``neighbor_include_fwhm x fwhm_px``, fits them simultaneously via a
    ``SourceGrouper`` (``min_separation = group_sep_fwhm x fwhm_px``), then returns
    ONLY the target's flux. Returns ``None`` to signal the caller to fall back to the
    single-star path (no neighbours, out of frame, or fit failure/divergence).
    """
    if neighbor_xy is None or len(neighbor_xy) == 0 or not math.isfinite(fwhm_px) or fwhm_px <= 0:
        return None
    h, w = frame_data.shape
    r_inc = float(neighbor_include_fwhm) * float(fwhm_px)
    sep = float(group_sep_fwhm) * float(fwhm_px)

    # Neighbours within the inclusion radius (exclude the target itself).
    dx = neighbor_xy[:, 0] - float(x)
    dy = neighbor_xy[:, 1] - float(y)
    dist = np.hypot(dx, dy)
    sel = (dist <= r_inc) & (dist > 0.5 * float(fwhm_px))
    if not np.any(sel):
        return None  # isolated -> let caller use the single-star path

    src_x = [float(x)]
    src_y = [float(y)]
    src_f: list[float] = [float("nan")]  # target init flux filled after cutout
    for j in np.where(sel)[0]:
        src_x.append(float(neighbor_xy[j, 0]))
        src_y.append(float(neighbor_xy[j, 1]))
        fj = float(neighbor_flux[j]) if neighbor_flux is not None else float("nan")
        src_f.append(fj if math.isfinite(fj) and fj > 0 else float("nan"))

    # Dynamic cutout covering all grouped sources + a fit-shape margin.
    fit_half = max(int(fit_shape[0]), int(fit_shape[1])) // 2
    pad = fit_half + 2
    x_lo = int(math.floor(min(src_x))) - pad
    x_hi = int(math.ceil(max(src_x))) + pad
    y_lo = int(math.floor(min(src_y))) - pad
    y_hi = int(math.ceil(max(src_y))) + pad
    if x_lo < 0 or y_lo < 0 or x_hi >= w or y_hi >= h:
        return None
    cut = np.asarray(frame_data[y_lo : y_hi + 1, x_lo : x_hi + 1], dtype=np.float64)
    if cut.size == 0 or cut.shape[0] < int(fit_shape[1]) or cut.shape[1] < int(fit_shape[0]):
        return None

    sky, _sky_method = _resolve_psf_fit_sky(frame_data, cut, float(x), float(y), fwhm_px=float(fwhm_px))
    cut_sub = cut - sky
    _bmask = np.ones(cut.shape, dtype=bool)
    if cut.shape[0] > 4 and cut.shape[1] > 4:
        _bmask[2:-2, 2:-2] = False
    _bvals = cut[_bmask]
    _bfin = _bvals[np.isfinite(_bvals)]

    # Target init flux: positive sum within ~1 FWHM of its position.
    tx_l, ty_l = float(x) - x_lo, float(y) - y_lo
    yy, xx = np.mgrid[0 : cut.shape[0], 0 : cut.shape[1]]
    near = np.hypot(xx - tx_l, yy - ty_l) <= max(2.0, float(fwhm_px))
    tflux = float(np.nansum(cut_sub[near].clip(min=0)))
    if not math.isfinite(tflux) or tflux <= 0:
        tflux = float(np.nanmax(cut_sub)) if math.isfinite(float(np.nanmax(cut_sub))) else 1.0
        if tflux <= 0:
            tflux = 1.0
    src_f[0] = tflux
    # Fill neighbour init fluxes that were unknown with a modest positive guess.
    _fallback_f = max(1.0, tflux)
    init_x = [sx - x_lo for sx in src_x]
    init_y = [sy - y_lo for sy in src_y]
    init_f = [f if (math.isfinite(f) and f > 0) else _fallback_f for f in src_f]

    err_full_cut = None
    if err_full is not None:
        ec = np.asarray(err_full[y_lo : y_hi + 1, x_lo : x_hi + 1], dtype=np.float64)
        if ec.shape == cut.shape and np.any(np.isfinite(ec)) and float(np.nanmax(ec)) > 0:
            err_full_cut = ec
    _grp_gain, _grp_rn = _psf_resolve_gain_read_noise(frame_hdr)
    _flux_init = _psf_flux_init_for_error_map(float("nan"), tflux)
    err_cut = _psf_fit_error_cutout_full_ccd(
        cut.shape,
        psf_model=psf_model,
        flux_init=_flux_init,
        x_0=tx_l,
        y_0=ty_l,
        sky_per_px=sky,
        gain=_grp_gain,
        read_noise_e=_grp_rn,
        err_full_cut=err_full_cut,
    )
    _weight_mode = _PSF_WEIGHT_MODE_FULL
    _err_mode = _PSF_ERR_MODE_FULL

    try:
        grouper = SourceGrouper(min_separation=float(sep))
        phot = PSFPhotometry(
            psf_model,
            fit_shape,
            grouper=grouper,
            aperture_radius=max(3, int(round(float(fwhm_px) * 1.5))),
            progress_bar=False,
        )
        init = Table([init_x, init_y, init_f], names=("x_0", "y_0", "flux_0"))
        res = phot(data=cut_sub, init_params=init, error=err_cut)
        _flux_arr = np.asarray(res["flux_fit"], dtype=float)
        _sources = [
            (float(x_lo) + float(res["x_fit"][i]), float(y_lo) + float(res["y_fit"][i]), float(_flux_arr[i]))
            for i in range(len(_flux_arr))
            if math.isfinite(_flux_arr[i]) and _flux_arr[i] > 0
        ]
        if _sources:
            _sky_new, _meth_new = _residual_annulus_sky_per_px(
                frame_data,
                float(x),
                float(y),
                fwhm_px=float(fwhm_px),
                psf_model=psf_model,
                sources=_sources,
            )
            if _meth_new == "residual_annulus" and math.isfinite(_sky_new):
                sky = _sky_new
                _sky_method = _meth_new
                cut_sub = cut - sky
                init_f2 = [float(f) for f in _flux_arr]
                init = Table([init_x, init_y, init_f2], names=("x_0", "y_0", "flux_0"))
                _flux_init2 = _psf_flux_init_for_error_map(float("nan"), float(_flux_arr[0]))
                err_cut = _psf_fit_error_cutout_full_ccd(
                    cut.shape,
                    psf_model=psf_model,
                    flux_init=_flux_init2,
                    x_0=tx_l,
                    y_0=ty_l,
                    sky_per_px=sky,
                    gain=_grp_gain,
                    read_noise_e=_grp_rn,
                    err_full_cut=err_full_cut,
                )
                res = phot(data=cut_sub, init_params=init, error=err_cut)
        # Identify the target row: nearest fit position to the target init xy.
        xf = np.asarray(res["x_fit"], dtype=float)
        yf = np.asarray(res["y_fit"], dtype=float)
        d2 = (xf - tx_l) ** 2 + (yf - ty_l) ** 2
        k = int(np.argmin(d2))
        if math.sqrt(float(d2[k])) > max(2.0, 0.75 * float(fwhm_px)):
            return None  # target not recovered in the joint fit -> fall back
        flux_fit = float(res["flux_fit"][k])
        flux_err = _psf_sandwich_flux_err(
            flux_fit,
            psf_model,
            float(xf[k]),
            float(yf[k]),
            cut.shape,
            sky_per_px=sky,
            gain=_grp_gain,
            read_noise_e=_grp_rn,
            fit_shape=fit_shape,
        )
        chi2 = float(res["reduced_chi2"][k]) if "reduced_chi2" in res.colnames else float("nan")
        flags = int(res["flags"][k]) if "flags" in res.colnames else 0
    except (ValueError, TypeError, RuntimeError, ImportError) as exc:
        from except_fix_counters import get_except_fix_counters

        get_except_fix_counters().psf_grouped_fit_fail += 1
        logging.error(
            "[PSF] grouped fit failed x=%.2f y=%.2f: %s",
            float(x),
            float(y),
            exc,
        )
        return None

    if not math.isfinite(flux_fit) or flux_fit <= 0:
        return None
    converged = (flags & 8) == 0
    chi2_ok = (not math.isfinite(chi2)) or (chi2 < float(chi2_limit))
    return {
        "psf_flux": flux_fit,
        "psf_flux_err": flux_err,
        "psf_chi2": chi2,
        "psf_fit_ok": bool(converged and chi2_ok),
        "n_group": int(len(init_f)),
        "x_fit": float(x_lo) + float(xf[k]),
        "y_fit": float(y_lo) + float(yf[k]),
        "psf_sky_method": _sky_method,
        "psf_weight_mode": _weight_mode,
        "psf_err_mode": _err_mode,
    }


_PSF_QUALITY_THRESH = {
    "chi2_marginal": 5.0,
    "snr_marginal": 10.0,
    "snr_bad": 5.0,
    "shift_marginal_fwhm": 0.5,
    "shift_bad_fwhm": 1.0,
    "nn_bad_fwhm": 1.0,        # any neighbour this close -> unresolved blend, bad
    "nn_blend_fwhm": 1.5,      # contaminating (<=Deltamag) neighbour this close -> bad
    "nn_marginal_fwhm": 2.0,   # any neighbour this close -> marginal
    "nn_contam_dmag": 2.5,     # neighbour <= this many mag fainter counts as contaminating
}


def assess_psf_quality(
    chi2: float | None,
    snr: float | None,
    pos_shift_px: float | None,
    fwhm_px: float | None,
    nn_dist_fwhm: float | None,
    *,
    nn_delta_mag: float | None = None,
    chi2_bad: float = 50.0,
) -> str:
    """Grade a per-star PSF fit as ``good`` / ``marginal`` / ``bad``.

    Combines reduced chi^2, fit SNR (flux/flux_err, Howell-style detection significance),
    fitted-position shift (in FWHM), and nearest-neighbour proximity (``nn_dist_fwhm``,
    with the neighbour's relative brightness ``nn_delta_mag`` = ``mag_neighbour - mag_star``).
    A close neighbour that is comparably bright or brighter (``nn_delta_mag <= nn_contam_dmag``)
    is treated as a contaminating blend - that is the CSS_J161519.8 case (neighbour 3.5 mag
    *brighter* at 1.46 FWHM). Severity is the WORST of the available criteria; a non-finite
    chi^2/shift is itself ``bad``. Missing inputs (NaN/None) are skipped (graceful degradation).
    """
    sev = 0  # 0 good, 1 marginal, 2 bad
    t = _PSF_QUALITY_THRESH
    if chi2 is None or not math.isfinite(chi2) or chi2 >= chi2_bad:
        sev = max(sev, 2)
    elif chi2 >= t["chi2_marginal"]:
        sev = max(sev, 1)
    if snr is not None and math.isfinite(snr):
        if snr < t["snr_bad"]:
            sev = max(sev, 2)
        elif snr < t["snr_marginal"]:
            sev = max(sev, 1)
    if pos_shift_px is not None and not math.isfinite(pos_shift_px):
        sev = max(sev, 2)
    elif pos_shift_px is not None and fwhm_px and fwhm_px > 0:
        frac = pos_shift_px / fwhm_px
        if frac >= t["shift_bad_fwhm"]:
            sev = max(sev, 2)
        elif frac >= t["shift_marginal_fwhm"]:
            sev = max(sev, 1)
    if nn_dist_fwhm is not None and math.isfinite(nn_dist_fwhm):
        _contam = (nn_delta_mag is None) or (
            math.isfinite(nn_delta_mag) and nn_delta_mag <= t["nn_contam_dmag"]
        )
        if nn_dist_fwhm < t["nn_bad_fwhm"] or _contam and nn_dist_fwhm < t["nn_blend_fwhm"]:
            sev = max(sev, 2)
        elif nn_dist_fwhm < t["nn_marginal_fwhm"] or _contam and nn_dist_fwhm < (t["nn_marginal_fwhm"] + 0.5):
            sev = max(sev, 1)
    return ("good", "marginal", "bad")[sev]


def psf_photometry_stars(
    frame_data: np.ndarray,
    frame_hdr: fits.Header,
    star_positions: pd.DataFrame,
    epsf_model_path: Path,
    *,
    cutout_size: int | None = None,
    error: np.ndarray | None = None,
    use_iterative: bool = True,
    max_fit_iters: int = 3,
    ref_fluxes: np.ndarray | None = None,
    apply_aperture_correction: bool = True,
    psf_ac_policy: str | None = None,
    grouper_enabled: bool | None = None,
    neighbor_catalog: pd.DataFrame | None = None,
    group_sep_fwhm: float | None = None,
    neighbor_include_fwhm: float | None = None,
    gridded_model: dict[str, Any] | None = None,
    nn_dist_fwhm_map: dict[str, float] | None = None,
    nn_delta_mag_map: dict[str, float] | None = None,
    quality_fallback_enabled: bool | None = None,
) -> pd.DataFrame:
    """Run iterative (or single-pass) PSF photometry on cutouts per star; never fails per row.

    If ``error`` is provided (same shape as ``frame_data``), per-cutout slices are passed to
    photutils for per-pixel uncertainties (enables finite reduced chi^2 when the model fits).

    If ``gridded_model`` (the dict from :func:`build_epsf_grid_model`) is provided, the
    spatially-varying path is used: each star is fit with the ePSF bilinearly interpolated at
    its (x, y) detector position instead of one global ePSF. The single-ePSF path (``epsf_model_path``)
    is untouched when ``gridded_model`` is ``None``.
    """
    _ = frame_hdr  # reserved for future metadata (WCS, gain, ...)

    cols_req = ("x", "y", "catalog_id", "name")
    for c in cols_req:
        if c not in star_positions.columns:
            raise ValueError(f"star_positions missing required column: {c!r}")

    ep = Path(epsf_model_path)
    if not ep.is_file():
        raise FileNotFoundError(f"EPSF FITS not found: {ep}")

    meta: dict[str, Any] = {}
    meta_fp = ep.parent / _MASTERSTAR_EPSF_META_NAME
    if meta_fp.is_file():
        try:
            meta = json.loads(meta_fp.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            meta = {}

    if cutout_size is None:
        if not meta_fp.is_file():
            raise FileNotFoundError(f"cutout_size=None requires {meta_fp}")
        cutout_size = int(meta["cutout_size"])
        os_meta = meta.get("oversampling", 2)
        if isinstance(os_meta, list):
            osamp: Any = int(os_meta[0]) if len(os_meta) else 2
        else:
            osamp = int(os_meta)
    else:
        osamp = 2
        try:
            om = meta.get("oversampling", 2)
            osamp = int(om[0]) if isinstance(om, list) and len(om) else int(om)
        except Exception:  # noqa: BLE001
            pass

    cutout_size = int(cutout_size)
    if cutout_size % 2 == 0 or cutout_size < 3:
        raise ValueError(f"cutout_size must be odd and >= 3, got {cutout_size}")

    fwhm_px_meta = float(meta.get("fwhm_px", 0.0))
    _psf_gain, _psf_rn = _psf_resolve_gain_read_noise(frame_hdr)
    _weight_mode = _PSF_WEIGHT_MODE_FULL
    _err_mode = _PSF_ERR_MODE_FULL

    _ref_flux_by_cid: dict[str, float] = {}
    if ref_fluxes is not None:
        _ref_arr = np.asarray(ref_fluxes, dtype=float)
        if len(_ref_arr) == len(star_positions):
            for _j, (_, _row) in enumerate(star_positions.iterrows()):
                _v = float(_ref_arr[_j])
                if math.isfinite(_v) and _v > 0:
                    _ref_flux_by_cid[str(_row["catalog_id"])] = _v

    err_full: np.ndarray | None = None
    if error is not None:
        err_full = np.asarray(error, dtype=np.float64)
        if err_full.shape != frame_data.shape:
            raise ValueError(
                f"error map shape {err_full.shape} != frame_data shape {frame_data.shape}"
            )

    psf_data = np.asarray(fits.getdata(ep), dtype=np.float64)
    psf_model = ImagePSF(psf_data, oversampling=osamp)
    fit_shape = _fit_shape_for_cutout(
        cutout_size,
        fwhm_px=fwhm_px_meta if fwhm_px_meta > 0 else None,
    )
    _aperture_radius = max(3, int(round(fwhm_px_meta * 1.5))) if fwhm_px_meta > 0 else 5
    _used_iterative = False
    if use_iterative:
        try:
            phot = IterativePSFPhotometry(
                psf_model,
                fit_shape,
                _epsf_noop_finder,
                aperture_radius=_aperture_radius,
                sub_shape=fit_shape,
                maxiters=max_fit_iters,
                mode="new",
                progress_bar=False,
            )
            _used_iterative = True
        except Exception as exc_iter:  # noqa: BLE001
            log_event(
                f"IterativePSFPhotometry failed ({exc_iter}), falling back to PSFPhotometry"
            )
            phot = PSFPhotometry(psf_model, fit_shape=fit_shape, progress_bar=False)
            _used_iterative = False
    else:
        phot = PSFPhotometry(psf_model, fit_shape=fit_shape, progress_bar=False)
        _used_iterative = False

    # -- Spatially-varying ePSF (gated): per-star position-interpolated model ------
    _spatial_active = bool(
        gridded_model is not None
        and isinstance(gridded_model, dict)
        and gridded_model.get("cell_arrays")
    )

    if _spatial_active:
        log_event(
            f"PSF spatial ePSF ACTIVE: {gridded_model.get('grid_nx')}x{gridded_model.get('grid_ny')} grid, "
            f"{gridded_model.get('n_fallback', 0)} fallback cell(s)"
        )

    # -- Gated SourceGrouper (joint/deblended fit) configuration ------------------
    try:
        from config import AppConfig as _AppCfg  # noqa: PLC0415

        _cfg_grp = _AppCfg()
    except Exception:  # noqa: BLE001
        _cfg_grp = None
    _fix_position = bool(getattr(_cfg_grp, "psf_fix_position_enabled", False)) if _cfg_grp else False
    _apply_psf_fixed_position(phot, fix=_fix_position)

    def _make_phot(_pm: ImagePSF) -> tuple[Any, bool]:
        """Build a photometry object for a given (per-star) PSF model, mirroring global config."""
        if use_iterative:
            try:
                _p = IterativePSFPhotometry(
                    _pm,
                    fit_shape,
                    _epsf_noop_finder,
                    aperture_radius=_aperture_radius,
                    sub_shape=fit_shape,
                    maxiters=max_fit_iters,
                    mode="new",
                    progress_bar=False,
                )
                _apply_psf_fixed_position(_p, fix=_fix_position)
                return _p, True
            except Exception:  # noqa: BLE001
                _p = PSFPhotometry(_pm, fit_shape=fit_shape, progress_bar=False)
                _apply_psf_fixed_position(_p, fix=_fix_position)
                return _p, False
        _p = PSFPhotometry(_pm, fit_shape=fit_shape, progress_bar=False)
        _apply_psf_fixed_position(_p, fix=_fix_position)
        return _p, False
    _grp_enabled = (
        bool(getattr(_cfg_grp, "psf_grouper_enabled", False))
        if grouper_enabled is None
        else bool(grouper_enabled)
    )
    _grp_sep = (
        float(getattr(_cfg_grp, "psf_group_sep_fwhm", 1.5))
        if group_sep_fwhm is None
        else float(group_sep_fwhm)
    )
    _grp_inc = (
        float(getattr(_cfg_grp, "psf_neighbor_include_fwhm", 3.0))
        if neighbor_include_fwhm is None
        else float(neighbor_include_fwhm)
    )
    try:
        _grp_chi2_limit = float(getattr(_cfg_grp, "psf_chi2_threshold", 50.0))
    except Exception:  # noqa: BLE001
        _grp_chi2_limit = 50.0
    if not math.isfinite(_grp_chi2_limit) or _grp_chi2_limit <= 0:
        _grp_chi2_limit = 50.0
    # Neighbour arrays (positions + init flux) for joint fitting.
    _nb_xy: np.ndarray | None = None
    _nb_flux: np.ndarray | None = None
    _grp_active = bool(_grp_enabled and neighbor_catalog is not None and fwhm_px_meta > 0)
    if _grp_active:
        try:
            _nb = neighbor_catalog
            if {"x", "y"}.issubset(_nb.columns) and len(_nb) > 0:
                _nb_xy = _nb[["x", "y"]].to_numpy(dtype=float)
                if "flux_init" in _nb.columns:
                    _nb_flux = pd.to_numeric(_nb["flux_init"], errors="coerce").to_numpy(dtype=float)
                else:
                    _nb_flux = np.full(len(_nb_xy), float("nan"))
            else:
                _grp_active = False
        except Exception:  # noqa: BLE001
            _grp_active = False
            _nb_xy = None

    h, w = frame_data.shape
    half = cutout_size // 2

    out_rows: list[dict[str, Any]] = []
    _cols = [
        "catalog_id",
        "name",
        "x",
        "y",
        "psf_flux",
        "psf_flux_err",
        "psf_chi2",
        "psf_fit_ok",
        "psf_iterative",
        "psf_ac_factor",
        "psf_ac_n_used",
        "psf_ac_applied",
        "psf_ac_policy",
        "psf_group_used",
        "psf_group_n",
        "psf_group_fallback",
        "psf_snr",
        "psf_pos_shift",
        "psf_sky_method",
        "psf_weight_mode",
        "psf_err_mode",
        "psf_nn_dist_fwhm",
        "psf_quality",
        "psf_quality_fallback",
        "x_fit",
        "y_fit",
    ]
    _quality_fallback_on = (
        bool(getattr(_cfg_grp, "psf_quality_fallback_enabled", True))
        if quality_fallback_enabled is None
        else bool(quality_fallback_enabled)
    )
    _nn_map = nn_dist_fwhm_map or {}
    _nn_dmag_map = nn_delta_mag_map or {}
    _ac_policy = resolve_psf_ac_policy(
        psf_ac_policy, apply_aperture_correction=apply_aperture_correction
    )
    _do_ac = _ac_policy == PSF_AC_POLICY_CHI2_LT5_LEGACY
    if star_positions.empty:
        return pd.DataFrame(columns=_cols)

    for _, row in star_positions.iterrows():
        cid = row["catalog_id"]
        name = row["name"]
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
                    "psf_fit_ok": False,
                    "psf_iterative": False,
                    "psf_ac_factor": 1.0,
                    "psf_ac_n_used": 0,
                    "psf_ac_applied": False,
                    "psf_group_used": False,
                    "psf_group_n": 0,
                    "psf_group_fallback": False,
                    "x_fit": float("nan"),
                    "y_fit": float("nan"),
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
            "psf_fit_ok": False,
            "psf_iterative": False,
            "psf_ac_factor": 1.0,
            "psf_ac_n_used": 0,
            "psf_ac_applied": False,
            "psf_group_used": False,
            "psf_group_n": 0,
            "psf_group_fallback": False,
            "x_fit": float("nan"),
            "y_fit": float("nan"),
        }

        # -- Gated joint/deblended fit: target + close neighbours via SourceGrouper --
        _group_fallback = False
        if _grp_active and _nb_xy is not None:
            _gres = _grouped_psf_fit(
                frame_data,
                err_full,
                x,
                y,
                fwhm_px=fwhm_px_meta,
                fit_shape=fit_shape,
                psf_model=psf_model,
                neighbor_xy=_nb_xy,
                neighbor_flux=_nb_flux if _nb_flux is not None else np.array([]),
                group_sep_fwhm=_grp_sep,
                neighbor_include_fwhm=_grp_inc,
                chi2_limit=_grp_chi2_limit,
                frame_hdr=frame_hdr,
            )
            if _gres is not None:
                row_out = dict(base)
                row_out.update(_gres)
                row_out["psf_group_used"] = True
                row_out["psf_group_n"] = int(_gres.get("n_group", 0))
                row_out.pop("n_group", None)
                out_rows.append(row_out)
                continue
            # neighbours existed but joint fit failed -> flag and fall through.
            # (No neighbours returns None too; that is a normal isolated star, not a
            #  fallback. Distinguish via a quick neighbour check.)
            _dxq = _nb_xy[:, 0] - float(x)
            _dyq = _nb_xy[:, 1] - float(y)
            _distq = np.hypot(_dxq, _dyq)
            _has_nb = bool(
                np.any((_distq <= _grp_inc * float(fwhm_px_meta)) & (_distq > 0.5 * float(fwhm_px_meta)))
            )
            _group_fallback = _has_nb
        base["psf_group_fallback"] = _group_fallback

        xi, yi = int(round(x)), int(round(y))
        if xi < half or yi < half or xi >= w - half or yi >= h - half:
            out_rows.append(base)
            continue

        x1 = xi - half
        y1 = yi - half
        x2 = x1 + cutout_size
        y2 = y1 + cutout_size

        try:
            cut = np.asarray(frame_data[y1:y2, x1:x2], dtype=np.float64)
            if cut.shape != (cutout_size, cutout_size):
                out_rows.append(base)
                continue

            _sky_per_px, _sky_method = _resolve_psf_fit_sky(
                frame_data,
                cut,
                float(x),
                float(y),
                fwhm_px=fwhm_px_meta if fwhm_px_meta > 0 else 0.0,
            )
            _border_mask = np.ones(cut.shape, dtype=bool)
            _border_mask[2:-2, 2:-2] = False
            _border_vals = cut[_border_mask]
            _border_finite = _border_vals[np.isfinite(_border_vals)]
            cut_sky_sub = cut - _sky_per_px

            xc = x - x1
            yc = y - y1
            flux_guess = float(np.nansum(cut_sky_sub.clip(min=0)))
            if not math.isfinite(flux_guess) or flux_guess <= 0.0:
                flux_guess = float(np.nanmax(cut)) * 0.5 * cutout_size * cutout_size
                if not math.isfinite(flux_guess) or flux_guess <= 0.0:
                    flux_guess = 1.0

            init = Table([[xc], [yc], [flux_guess]], names=("x_0", "y_0", "flux_0"))
            err_full_cut = None
            if err_full is not None:
                err_full_cut = np.asarray(err_full[y1:y2, x1:x2], dtype=np.float64)
                if err_full_cut.shape != cut.shape:
                    raise ValueError("error cutout shape mismatch")
            _dao_flux = _ref_flux_by_cid.get(str(cid), float("nan"))
            _flux_init = _psf_flux_init_for_error_map(_dao_flux, flux_guess)
            _star_iterative = _used_iterative
            _psf_model_use = psf_model
            _phot_use = phot
            if _spatial_active:
                _local_arr = interp_gridded_epsf_array(gridded_model, x, y)
                _psf_model_use = ImagePSF(np.asarray(_local_arr, dtype=np.float64), oversampling=osamp)
                _phot_use, _star_iterative = _make_phot(_psf_model_use)
            err_cut = _psf_fit_error_cutout_full_ccd(
                cut.shape,
                psf_model=_psf_model_use,
                flux_init=_flux_init,
                x_0=xc,
                y_0=yc,
                sky_per_px=_sky_per_px,
                gain=_psf_gain,
                read_noise_e=_psf_rn,
                err_full_cut=err_full_cut,
            )
            try:
                res = _phot_use(data=cut_sky_sub, init_params=init, error=err_cut)
            except Exception as exc_call:  # noqa: BLE001
                if _star_iterative:
                    log_event(
                        f"IterativePSFPhotometry fit failed ({exc_call}), "
                        "falling back to PSFPhotometry for this star"
                    )
                    _phot_fb = PSFPhotometry(_psf_model_use, fit_shape=fit_shape, progress_bar=False)
                    res = _phot_fb(data=cut_sky_sub, init_params=init, error=err_cut)
                    _star_iterative = False
                else:
                    raise

            # Residual-annulus sky: subtract fitted PSF wing from annulus (1 refine pass).
            if fwhm_px_meta > 0:
                _ff0 = float(res["flux_fit"][0])
                _xf0 = float(res["x_fit"][0]) + float(x1)
                _yf0 = float(res["y_fit"][0]) + float(y1)
                if math.isfinite(_ff0) and _ff0 > 0:
                    _sky_new, _meth_new = _residual_annulus_sky_per_px(
                        frame_data,
                        float(x),
                        float(y),
                        fwhm_px=float(fwhm_px_meta),
                        psf_model=_psf_model_use,
                        sources=[(_xf0, _yf0, _ff0)],
                    )
                    if _meth_new == "residual_annulus" and math.isfinite(_sky_new):
                        _sky_per_px = _sky_new
                        _sky_method = _meth_new
                        cut_sky_sub = cut - _sky_per_px
                        _fg2 = float(np.nansum(cut_sky_sub.clip(min=0)))
                        if math.isfinite(_fg2) and _fg2 > 0:
                            init = Table([[xc], [yc], [_fg2]], names=("x_0", "y_0", "flux_0"))
                            _flux_init = _psf_flux_init_for_error_map(_dao_flux, _fg2)
                            err_cut = _psf_fit_error_cutout_full_ccd(
                                cut.shape,
                                psf_model=_psf_model_use,
                                flux_init=_flux_init,
                                x_0=xc,
                                y_0=yc,
                                sky_per_px=_sky_per_px,
                                gain=_psf_gain,
                                read_noise_e=_psf_rn,
                                err_full_cut=err_full_cut,
                            )
                            try:
                                res = _phot_use(data=cut_sky_sub, init_params=init, error=err_cut)
                            except Exception as exc_refine:  # noqa: BLE001
                                if _star_iterative:
                                    _phot_fb = PSFPhotometry(
                                        _psf_model_use, fit_shape=fit_shape, progress_bar=False
                                    )
                                    res = _phot_fb(data=cut_sky_sub, init_params=init, error=err_cut)
                                    _star_iterative = False
                                else:
                                    log_event(f"PSF residual-sky refit failed ({exc_refine})")

            flux_fit = float(res["flux_fit"][0])
            _xf_fit = float(res["x_fit"][0])
            _yf_fit = float(res["y_fit"][0])
            flux_err = _psf_sandwich_flux_err(
                flux_fit,
                _psf_model_use,
                _xf_fit,
                _yf_fit,
                cut.shape,
                sky_per_px=_sky_per_px,
                gain=_psf_gain,
                read_noise_e=_psf_rn,
                fit_shape=fit_shape,
            )
            chi2 = float(res["reduced_chi2"][0])
            flags = int(res["flags"][0])
            converged = (flags & 8) == 0
            try:
                _xf = float(res["x_fit"][0])
                _yf = float(res["y_fit"][0])
                _pos_shift = float(math.hypot(_xf - xc, _yf - yc))
            except Exception:  # noqa: BLE001
                _pos_shift = float("nan")
            _snr = (
                float(flux_fit / flux_err)
                if (math.isfinite(flux_fit) and math.isfinite(flux_err) and flux_err > 0)
                else float("nan")
            )
            try:
                from config import AppConfig as _AppConfig

                _chi2_limit = float(getattr(_AppConfig(), "psf_chi2_threshold", 50.0))
            except Exception:  # noqa: BLE001
                _chi2_limit = 50.0
            if not math.isfinite(_chi2_limit) or _chi2_limit <= 0:
                _chi2_limit = 50.0
            chi2_ok = math.isfinite(chi2) and chi2 < _chi2_limit
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
                    "psf_fit_ok": fit_ok,
                    "psf_iterative": _star_iterative,
                    "psf_group_used": False,
                    "psf_group_n": 0,
                    "psf_group_fallback": _group_fallback,
                    "psf_snr": _snr,
                    "psf_pos_shift": _pos_shift,
                    "psf_sky_method": _sky_method,
                    "psf_weight_mode": _weight_mode,
                    "psf_err_mode": _err_mode,
                    "x_fit": float(res["x_fit"][0]) + float(x1),
                    "y_fit": float(res["y_fit"][0]) + float(y1),
                }
            )
        except Exception:  # noqa: BLE001
            out_rows.append(base)

    # --- Aperture correction ---
    _ac_factor = 1.0
    _ac_n_used = 0
    if _do_ac and ref_fluxes is not None:
        _psf_flux_arr = np.array([r.get("psf_flux", np.nan) for r in out_rows], dtype=float)
        _ref_flux_arr = np.asarray(ref_fluxes, dtype=float)
        _chi2_arr = np.array([r.get("psf_chi2", np.nan) for r in out_rows], dtype=float)
        if len(_psf_flux_arr) == len(_ref_flux_arr):
            _ac_factor, _ac_n_used = _compute_aperture_correction(
                _psf_flux_arr,
                _ref_flux_arr,
                _chi2_arr,
                chi2_limit=5.0,
                min_ref_stars=5,
            )
            if _ac_factor != 1.0:
                log_event(
                    f"ePSF aperture correction: factor={_ac_factor:.4f}, "
                    f"n_ref_stars={_ac_n_used}"
                )
                for r in out_rows:
                    if np.isfinite(r.get("psf_flux", np.nan)) and r["psf_flux"] > 0:
                        r["psf_flux"] = r["psf_flux"] * _ac_factor
                        if np.isfinite(r.get("psf_flux_err", np.nan)):
                            r["psf_flux_err"] = r["psf_flux_err"] * _ac_factor
            else:
                log_event(
                    f"ePSF aperture correction: not applied "
                    f"(n_clean_stars={_ac_n_used} < min_ref_stars=5)"
                )
    # --- end aperture correction ---

    for r in out_rows:
        r["psf_ac_factor"] = _ac_factor
        r["psf_ac_n_used"] = _ac_n_used
        r["psf_ac_policy"] = _ac_policy
        if _do_ac:
            r["psf_ac_applied"] = bool(_ac_n_used >= 5)
        else:
            r["psf_ac_applied"] = False
        # Per-star quality grade (always computed) + auto-fallback (default on).
        _cid_s = str(r.get("catalog_id", "")).strip()
        _nn = _nn_map.get(_cid_s, float("nan"))
        _nn_dm = _nn_dmag_map.get(_cid_s, None)
        r["psf_nn_dist_fwhm"] = _nn
        _q = assess_psf_quality(
            r.get("psf_chi2", float("nan")),
            r.get("psf_snr", float("nan")),
            r.get("psf_pos_shift", float("nan")),
            fwhm_px_meta if fwhm_px_meta > 0 else None,
            _nn,
            nn_delta_mag=_nn_dm,
            chi2_bad=_grp_chi2_limit,
        )
        r["psf_quality"] = _q
        # A bad PSF fit must never silently become the reported value: drop usability and
        # signal the caller to substitute aperture for this star (the RMS-20.4 lesson).
        if _q == "bad" and _quality_fallback_on:
            r["psf_quality_fallback"] = True
            r["psf_fit_ok"] = False
        else:
            r["psf_quality_fallback"] = False

    return pd.DataFrame(out_rows, columns=_cols)


__all__ = [
    "build_epsf_model",
    "fit_moffat_psf_stars",
    "get_epsf_fwhm_from_context",
    "psf_photometry_stars",
]
