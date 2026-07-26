"""
Density-aware VSX -> Gaia DR3 cross-match (Marrese et al. 2017/2019; Sutherland & Saunders 1992).

Design A: VSX is matched against the deep local Gaia DR3 catalogue (not masterstars).
True-match separations are modelled as a two-component Rayleigh mixture (heterogeneous VSX).
Acceptance radius r_max follows from the 1% contamination budget; mixture reliability ranks
multi-candidate cases only. Phase 0 separately tests DAO detection membership.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, search_around_sky

from database import get_gaia_db_max_g_mag, query_local_gaia
from gaia_catalog_id import normalize_gaia_source_id
from vyvar_platesolver import GAIA_EPOCH, _apply_proper_motion

LOGGER = logging.getLogger(__name__)

MAX_EXPECTED_CONTAMINATION = 0.01
PM_SIGMA_BROADEN_ARCSEC = 0.05
VSX_MATCH_EPOCH = 2000.0  # VSX positions treated as J2000 when epoch not recorded per entry
DEGENERACY_SIGMA_FACTOR = 0.25  # flag when sigma_broad > DEGENERACY_SIGMA_FACTOR / sqrt(rho) [deg]
MASTERSTARS_RECOVERY_WARN_FRAC = 0.80  # WARN if accepted masterstars < 80% of in-frame DAO IDs


class VsxGaiaCrossmatchError(RuntimeError):
    """Cross-match fit failed or field too sparse to estimate density."""


class VsxGaiaCrossmatchDegenerateError(VsxGaiaCrossmatchError):
    """Mixture fit locked onto the chance component (sigma_broad ~ mean NN separation)."""


@dataclass
class RayleighMixtureFit:
    q_fit: float
    w_fit: float
    sigma_narrow_arcsec: float
    sigma_broad_arcsec: float


@dataclass
class VsxGaiaCrossmatchDiagnostics:
    q_fit: float
    w_fit: float
    sigma_narrow_arcsec: float
    sigma_broad_arcsec: float
    rho_per_deg2: float
    mean_nn_arcsec: float
    pm_path: str
    pm_columns_present: bool
    n_pm_finite: int
    n_vsx: int
    n_gaia: int
    n_accepted: int
    expected_contamination_fraction: float
    r_max_arcsec: float
    converged: bool
    field_area_deg2: float
    gaia_db_max_g: float
    vsx_fainter_than_gaia_db: int
    candidate_multiplicity: dict[str, int] = field(default_factory=dict)
    multi_candidate_fraction: float = float("nan")
    fit_degenerate_warn: bool = False
    sep_quantiles_before_pm: dict[str, float] = field(default_factory=dict)
    sep_quantiles_after_pm: dict[str, float] = field(default_factory=dict)
    sep_quantiles_accepted: dict[str, float] = field(default_factory=dict)
    masterstars_in_frame: int = 0
    masterstars_eligible: int = 0
    masterstars_accepted: int = 0
    outcome_check: str = "ok"
    notes: list[str] = field(default_factory=list)

    @property
    def sigma_arcsec(self) -> float:
        """Backward-compatible alias (narrow component)."""
        return float(self.sigma_narrow_arcsec)


@dataclass
class VsxGaiaMatchRow:
    catalog_id: str
    sep_arcsec: float
    reliability: float
    quality: str
    accepted: bool


def _rayleigh_pdf(r_arcsec: float | np.ndarray, sigma_arcsec: float) -> float | np.ndarray:
    r = np.asarray(r_arcsec, dtype=float)
    sig = float(sigma_arcsec)
    if sig <= 0:
        return 0.0 if np.isscalar(r_arcsec) else np.zeros_like(r)
    out = (r / (sig * sig)) * np.exp(-(r * r) / (2.0 * sig * sig))
    out = np.where(r >= 0, out, 0.0)
    if np.isscalar(r_arcsec):
        return float(out)
    return out


def _true_match_pdf(
    r_arcsec: float | np.ndarray,
    w: float,
    sigma_narrow: float,
    sigma_broad: float,
) -> float | np.ndarray:
    w_f = float(w)
    return w_f * _rayleigh_pdf(r_arcsec, sigma_narrow) + (1.0 - w_f) * _rayleigh_pdf(
        r_arcsec, sigma_broad
    )


def _mean_nn_arcsec(rho_per_deg2: float) -> float:
    if rho_per_deg2 <= 0:
        return float("nan")
    return float(3600.0 / (2.0 * math.sqrt(float(rho_per_deg2))))


def _degeneracy_threshold_arcsec(rho_per_deg2: float) -> float:
    if rho_per_deg2 <= 0:
        return float("inf")
    return float(DEGENERACY_SIGMA_FACTOR * 3600.0 / math.sqrt(float(rho_per_deg2)))


def _check_fit_degeneracy(sigma_broad_arcsec: float, rho_per_deg2: float) -> None:
    thr = _degeneracy_threshold_arcsec(rho_per_deg2)
    mean_nn = _mean_nn_arcsec(rho_per_deg2)
    if math.isfinite(sigma_broad_arcsec) and math.isfinite(thr) and sigma_broad_arcsec > thr:
        raise VsxGaiaCrossmatchDegenerateError(
            "VSX->Gaia mixture fit degenerate: fitted sigma_broad "
            f"{sigma_broad_arcsec:.2f}\" exceeds chance-scale guard {thr:.2f}\" "
            f"(mean NN separation ~ {mean_nn:.1f}\" at rho={rho_per_deg2:.1f} deg^-2) - "
            "fit locked onto random neighbours, not astrometric scatter"
        )


def _check_fit_narrow_on_wide_tail(
    seps: np.ndarray,
    fit: RayleighMixtureFit,
    rho_per_deg2: float,
) -> None:
    """Reject fits whose narrow component cannot explain astrometric separations."""
    mean_nn = _mean_nn_arcsec(rho_per_deg2)
    p50_all = float(np.percentile(seps, 50))
    scale = max(fit.sigma_narrow_arcsec, fit.sigma_broad_arcsec)
    if fit.q_fit > 0.5 and p50_all > 0.35 * mean_nn and p50_all > 3.0 * scale:
        raise VsxGaiaCrossmatchDegenerateError(
            "VSX->Gaia mixture fit degenerate: p50 separation "
            f"{p50_all:.1f}\" >> 3 x fitted scale {scale:.2f}\" at rho={rho_per_deg2:.1f} deg^-2 - "
            "narrow fit on chance-scale separations"
        )
    astrometric_cap = min(mean_nn * 0.15, 10.0)
    likely_true = seps[seps <= astrometric_cap]
    if likely_true.size < 10:
        return
    p90_true = float(np.percentile(likely_true, 90))
    if fit.q_fit > 0.5 and p90_true > 5.0 * fit.sigma_broad_arcsec:
        raise VsxGaiaCrossmatchDegenerateError(
            "VSX->Gaia mixture fit degenerate: p90 astrometric separation "
            f"{p90_true:.1f}\" >> 5 x sigma_broad {fit.sigma_broad_arcsec:.2f}\" - "
            "tail not captured by true-match mixture"
        )


def _mixture_pdf(
    r_arcsec: float | np.ndarray,
    q: float,
    w: float,
    sigma_narrow: float,
    sigma_broad: float,
    rho_per_deg2: float,
) -> float | np.ndarray:
    r_deg = np.asarray(r_arcsec, dtype=float) / 3600.0
    f_true = _true_match_pdf(r_arcsec, w, sigma_narrow, sigma_broad)
    f_chance = 2.0 * math.pi * r_deg * float(rho_per_deg2)
    return float(q) * f_true + f_chance


def _acceptance_radius_arcsec(rho_per_deg2: float) -> float:
    """Pre-registered 1%% contamination budget: rho * pi * r_max^2 = 0.01 (r in deg)."""
    if rho_per_deg2 <= 0:
        return float("nan")
    r_deg = math.sqrt(MAX_EXPECTED_CONTAMINATION / (math.pi * float(rho_per_deg2)))
    return float(r_deg * 3600.0)


def _assess_fit_degeneracy(
    seps: np.ndarray,
    fit: RayleighMixtureFit,
    rho_per_deg2: float,
) -> str | None:
    try:
        _check_fit_degeneracy(fit.sigma_broad_arcsec, rho_per_deg2)
        _check_fit_narrow_on_wide_tail(seps, fit, rho_per_deg2)
    except VsxGaiaCrossmatchDegenerateError as exc:
        return str(exc)
    return None


def _reliability_from_sep(
    sep_arcsec: float,
    q: float,
    w: float,
    sigma_narrow: float,
    sigma_broad: float,
    rho_per_deg2: float,
) -> float:
    if sep_arcsec < 0 or rho_per_deg2 <= 0 or q <= 0:
        return 0.0
    if sigma_narrow <= 0 or sigma_broad <= 0:
        return 0.0
    r_deg = float(sep_arcsec) / 3600.0
    f_true = float(_true_match_pdf(float(sep_arcsec), w, sigma_narrow, sigma_broad))
    f_chance = 2.0 * math.pi * r_deg * float(rho_per_deg2)
    if f_chance <= 0:
        return 1.0 if f_true > 0 else 0.0
    likelihood = float(q) * f_true / f_chance
    if not math.isfinite(likelihood) or likelihood <= 0:
        return 0.0
    return float(likelihood / (likelihood + 1.0))


def _chance_contamination_fraction(seps_arcsec: np.ndarray, rho_per_deg2: float) -> float:
    """Expected false-match fraction: mean of rho * pi * r^2 (r in deg) over candidates."""
    seps = np.asarray(seps_arcsec, dtype=float)
    seps = seps[np.isfinite(seps)]
    if seps.size == 0:
        return 0.0
    r_deg = seps / 3600.0
    return float(np.mean(float(rho_per_deg2) * np.pi * r_deg * r_deg))


def _sep_quantiles(seps_arcsec: np.ndarray) -> dict[str, float]:
    seps = np.asarray(seps_arcsec, dtype=float)
    seps = seps[np.isfinite(seps) & (seps >= 0)]
    if seps.size == 0:
        return {}
    return {
        "p50": float(np.percentile(seps, 50)),
        "p90": float(np.percentile(seps, 90)),
        "p95": float(np.percentile(seps, 95)),
    }


def _quality_from_reliability(r: float) -> str:
    if r >= 0.99:
        return "high"
    if r >= 0.95:
        return "good"
    if r >= 0.80:
        return "uncertain"
    return "poor"


def fit_rayleigh_mixture_from_separations(
    seps_arcsec: np.ndarray,
    rho_per_deg2: float,
    *,
    strict_degeneracy: bool = True,
) -> RayleighMixtureFit:
    """Fit Q, w, sigma_narrow, sigma_broad from nearest-neighbour separations."""
    seps = np.asarray(seps_arcsec, dtype=float)
    seps = seps[np.isfinite(seps) & (seps > 0)]
    if seps.size < 5:
        raise VsxGaiaCrossmatchError(
            f"VSX->Gaia cross-match refused: need >=5 separations to fit mixture, got {seps.size}"
        )
    if rho_per_deg2 <= 0 or not math.isfinite(rho_per_deg2):
        raise VsxGaiaCrossmatchError(f"VSX->Gaia cross-match refused: invalid rho={rho_per_deg2}")

    mean_nn = _mean_nn_arcsec(rho_per_deg2)
    p20 = float(np.percentile(seps, 20))
    p50 = float(np.percentile(seps, 50))
    p80 = float(np.percentile(seps, 80))
    p90 = float(np.percentile(seps, 90))
    p95 = float(np.percentile(seps, 95))
    seps_core = seps[seps <= p95]
    p80_core = float(np.percentile(seps_core, 80)) if seps_core.size else p80
    astrometric_cap = min(mean_nn * 0.15, 10.0)
    likely_true = seps[seps <= astrometric_cap]

    sn_seed = max(0.05, p20 / 1.177)
    if likely_true.size >= 10:
        p80_true = float(np.percentile(likely_true, 80))
        p90_true = float(np.percentile(likely_true, 90))
        sb_seed = max(sn_seed * 1.4, p80_true / 1.177, p90_true / 2.15)
        sb_floor = max(sn_seed * 1.3, p90_true / 2.15)
    else:
        sb_seed = max(sn_seed * 1.4, p80_core / 1.177, p90 / 2.15)
        sb_floor = sn_seed * 1.3
    w_seed = float(np.clip(np.mean(seps <= max(0.5, p50)), 0.05, 0.98))
    q_seed = float(np.clip(0.5 + 0.45 * (1.0 - min(1.0, p50 / max(mean_nn, 1.0))), 0.5, 0.99))

    def _eval_fit(q: float, w: float, sn: float, sb: float) -> float | None:
        if sn <= 0 or sb <= sn * 1.1 or q <= 0 or w <= 0 or w >= 1:
            return None
        pdf = _mixture_pdf(seps, q, w, sn, sb, rho_per_deg2)
        if np.any(pdf <= 0) or not np.all(np.isfinite(pdf)):
            return None
        return float(-np.sum(np.log(pdf)))

    best = RayleighMixtureFit(q_fit=q_seed, w_fit=w_seed, sigma_narrow_arcsec=sn_seed, sigma_broad_arcsec=sb_seed)
    best_nll = _eval_fit(q_seed, w_seed, sn_seed, sb_seed)
    if best_nll is None:
        best_nll = float("inf")

    sigma_n_grid = sn_seed * np.geomspace(0.35, 2.5, 10)
    sigma_b_grid = np.maximum(sb_seed * np.geomspace(0.5, 2.0, 12), sb_floor)
    w_grid = np.linspace(max(0.05, w_seed - 0.4), min(0.98, w_seed + 0.4), 10)
    q_grid = np.linspace(max(0.5, q_seed - 0.15), min(0.99, q_seed + 0.04), 8)

    for sn in sigma_n_grid:
        for sb in sigma_b_grid:
            if float(sb) <= float(sn) * 1.1:
                continue
            for w in w_grid:
                for q in q_grid:
                    nll = _eval_fit(float(q), float(w), float(sn), float(sb))
                    if nll is not None and nll < best_nll:
                        best_nll = nll
                        best = RayleighMixtureFit(
                            q_fit=float(q),
                            w_fit=float(w),
                            sigma_narrow_arcsec=float(sn),
                            sigma_broad_arcsec=float(sb),
                        )

    if not math.isfinite(best_nll):
        raise VsxGaiaCrossmatchError("VSX->Gaia cross-match refused: mixture fit did not converge")

    deg_msg = _assess_fit_degeneracy(seps, best, rho_per_deg2)
    if deg_msg:
        if strict_degeneracy:
            raise VsxGaiaCrossmatchDegenerateError(deg_msg)
        LOGGER.warning(
            "VSX-GAIA XM fit degenerate (ranking only; acceptance unaffected): %s",
            deg_msg,
        )
    return best


def fit_sigma_q_from_separations(
    seps_arcsec: np.ndarray,
    rho_per_deg2: float,
) -> tuple[float, float]:
    """Backward-compatible wrapper returning (sigma_narrow, q)."""
    fit = fit_rayleigh_mixture_from_separations(seps_arcsec, rho_per_deg2)
    return fit.sigma_narrow_arcsec, fit.q_fit


def field_area_deg2_from_wcs(wcs: Any, width_px: int, height_px: int) -> float:
    try:
        corners_x = np.array([0.0, float(width_px), float(width_px), 0.0])
        corners_y = np.array([0.0, 0.0, float(height_px), float(height_px)])
        ra_c, dec_c = wcs.all_pix2world(corners_x, corners_y, 0)
        coords = SkyCoord(ra=ra_c * u.deg, dec=dec_c * u.deg, frame="icrs")
        sep01 = coords[0].separation(coords[1]).deg
        sep12 = coords[1].separation(coords[2]).deg
        sep23 = coords[2].separation(coords[3]).deg
        sep30 = coords[3].separation(coords[0]).deg
        width_deg = (sep01 + sep23) / 2.0
        height_deg = (sep12 + sep30) / 2.0
        return float(max(width_deg * height_deg, 1e-12))
    except Exception:  # noqa: BLE001
        return float("nan")


def frame_bbox_radec_limits(
    wcs: Any,
    width_px: int,
    height_px: int,
    margin_px: float = 50.0,
    center: SkyCoord | None = None,
) -> tuple[float, float, float, float]:
    """Return (ra_min, ra_max, dec_min, dec_max) for frame bbox + margin (same geometry as VSX query)."""
    m = float(margin_px)
    w = float(width_px)
    h = float(height_px)
    xs_px = np.asarray([-m, w * 0.5, w + m, -m, w + m, -m, w * 0.5, w + m], dtype=np.float64)
    ys_px = np.asarray([-m, -m, -m, h * 0.5, h * 0.5, h + m, h + m, h + m], dtype=np.float64)
    world = wcs.all_pix2world(xs_px, ys_px, 0)
    ras = np.asarray(world[0], dtype=np.float64)
    decs = np.asarray(world[1], dtype=np.float64)
    ok = np.isfinite(ras) & np.isfinite(decs)
    if not bool(ok.any()):
        raise VsxGaiaCrossmatchError("VSX->Gaia cross-match refused: frame bbox WCS failed")
    ras = ras[ok]
    decs = decs[ok]
    de_min = float(np.min(decs))
    de_max = float(np.max(decs))
    if center is not None:
        try:
            ra0 = float(center.icrs.ra.deg)
        except Exception:  # noqa: BLE001
            ra0 = float(np.median(ras))
    else:
        ra0 = float(np.median(ras))
    dra = ((ras - ra0 + 180.0) % 360.0) - 180.0
    ra_min = ra0 + float(np.min(dra))
    ra_max = ra0 + float(np.max(dra))
    pad = 1.0 / 3600.0
    return ra_min - pad, ra_max + pad, de_min - pad, de_max + pad


def query_gaia_for_frame_bbox(
    gaia_db_path: str | Path,
    wcs: Any,
    width_px: int,
    height_px: int,
    *,
    margin_px: float = 50.0,
    center: SkyCoord | None = None,
) -> list[dict[str, Any]]:
    """Load Gaia DR3 rows over the frame footprint (same bbox as VSX frame query)."""
    ra_min, ra_max, dec_min, dec_max = frame_bbox_radec_limits(
        wcs, width_px, height_px, margin_px=margin_px, center=center
    )
    gmax = get_gaia_db_max_g_mag(gaia_db_path)
    mag_limit = float(gmax) if gmax > 0 else None
    return query_local_gaia(
        gaia_db_path,
        ra_min=ra_min,
        ra_max=ra_max,
        dec_min=dec_min,
        dec_max=dec_max,
        mag_limit=mag_limit,
    )


def _pm_stats(pmra: np.ndarray | None, pmdec: np.ndarray | None) -> tuple[bool, int]:
    if pmra is None or pmdec is None:
        return False, 0
    ok = np.isfinite(pmra) & np.isfinite(pmdec)
    return True, int(np.sum(ok))


def _apply_pm_to_gaia_catalog(
    ga_ra: np.ndarray,
    ga_dec: np.ndarray,
    pmra: np.ndarray | None,
    pmdec: np.ndarray | None,
    target_epoch: float,
) -> tuple[np.ndarray, np.ndarray, str, float, int]:
    """Propagate Gaia DR3 positions from GAIA_EPOCH to target_epoch (VSX assumed J2000)."""
    ra_out = np.array(ga_ra, dtype=float, copy=True)
    dec_out = np.array(ga_dec, dtype=float, copy=True)
    pm_cols, n_pm_finite = _pm_stats(pmra, pmdec)
    if not pm_cols or n_pm_finite == 0:
        return ra_out, dec_out, "broadened", PM_SIGMA_BROADEN_ARCSEC, n_pm_finite
    n_prop = 0
    for i in range(len(ra_out)):
        ra_c, dec_c = _apply_proper_motion(
            float(ra_out[i]),
            float(dec_out[i]),
            float(pmra[i]) if pmra is not None else None,
            float(pmdec[i]) if pmdec is not None else None,
            obs_year=float(target_epoch),
        )
        if math.isfinite(pmra[i]) and math.isfinite(pmdec[i]):  # type: ignore[index]
            n_prop += 1
        ra_out[i] = ra_c
        dec_out[i] = dec_c
    if n_prop > 0:
        return ra_out, dec_out, "propagated", 0.0, n_pm_finite
    return ra_out, dec_out, "broadened", PM_SIGMA_BROADEN_ARCSEC, n_pm_finite


def _nn_separations_arcsec(
    vsx_ra_deg: np.ndarray,
    vsx_dec_deg: np.ndarray,
    ga_ra_deg: np.ndarray,
    ga_dec_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    vcoo = SkyCoord(ra=vsx_ra_deg * u.deg, dec=vsx_dec_deg * u.deg, frame="icrs")
    gcoo = SkyCoord(ra=ga_ra_deg * u.deg, dec=ga_dec_deg * u.deg, frame="icrs")
    idx, sep2d, _ = vcoo.match_to_catalog_sky(gcoo)
    return idx.astype(int), sep2d.to(u.arcsec).value.astype(float)


def match_vsx_to_gaia_density_aware(
    vsx_ra_deg: np.ndarray,
    vsx_dec_deg: np.ndarray,
    gaia_source_id: np.ndarray,
    gaia_ra_deg: np.ndarray,
    gaia_dec_deg: np.ndarray,
    *,
    field_area_deg2: float,
    pmra: np.ndarray | None = None,
    pmdec: np.ndarray | None = None,
    vsx_match_epoch: float = VSX_MATCH_EPOCH,
    gaia_db_max_g: float = 0.0,
    vsx_mag_max: np.ndarray | None = None,
    masterstars_ids: set[str] | frozenset[str] | None = None,
) -> tuple[list[VsxGaiaMatchRow], VsxGaiaCrossmatchDiagnostics]:
    """
    Match each VSX row to the nearest Gaia DR3 neighbour using a two-Rayleigh mixture fit.

    Raises VsxGaiaCrossmatchError / VsxGaiaCrossmatchDegenerateError on failure.
    """
    n_vsx = int(len(vsx_ra_deg))
    n_gaia = int(len(gaia_ra_deg))
    notes: list[str] = []
    empty_rows = [VsxGaiaMatchRow("", float("nan"), 0.0, "", False) for _ in range(n_vsx)]
    ms_ids = {str(x).strip() for x in (masterstars_ids or set()) if str(x).strip()}
    pm_columns_present, n_pm_finite = _pm_stats(pmra, pmdec)

    n_vsx_faint = 0
    if vsx_mag_max is not None and gaia_db_max_g > 0:
        vm = np.asarray(vsx_mag_max, dtype=float)
        n_vsx_faint = int(np.sum(np.isfinite(vm) & (vm > float(gaia_db_max_g))))

    if n_vsx == 0:
        return empty_rows, VsxGaiaCrossmatchDiagnostics(
            q_fit=float("nan"),
            w_fit=float("nan"),
            sigma_narrow_arcsec=float("nan"),
            sigma_broad_arcsec=float("nan"),
            rho_per_deg2=float("nan"),
            mean_nn_arcsec=float("nan"),
            pm_path="n/a",
            pm_columns_present=pm_columns_present,
            n_pm_finite=n_pm_finite,
            n_vsx=0,
            n_gaia=n_gaia,
            n_accepted=0,
            expected_contamination_fraction=0.0,
            r_max_arcsec=float("nan"),
            converged=False,
            field_area_deg2=float(field_area_deg2),
            gaia_db_max_g=float(gaia_db_max_g),
            vsx_fainter_than_gaia_db=n_vsx_faint,
            masterstars_in_frame=len(ms_ids),
            notes=["no VSX rows"],
        )

    if n_gaia == 0 or not math.isfinite(field_area_deg2) or field_area_deg2 <= 0:
        raise VsxGaiaCrossmatchError(
            "VSX->Gaia cross-match refused: no Gaia sources or invalid field area "
            f"(n_gaia={n_gaia}, area={field_area_deg2})"
        )

    idx_before, seps_before = _nn_separations_arcsec(
        vsx_ra_deg, vsx_dec_deg, gaia_ra_deg, gaia_dec_deg
    )
    ga_ra, ga_dec, pm_path, sigma_pm, _n_pm_used = _apply_pm_to_gaia_catalog(
        gaia_ra_deg, gaia_dec_deg, pmra, pmdec, vsx_match_epoch
    )
    _idx_after, seps_arcsec = _nn_separations_arcsec(vsx_ra_deg, vsx_dec_deg, ga_ra, ga_dec)
    idx = _idx_after

    rho = float(n_gaia) / float(field_area_deg2)
    mean_nn = _mean_nn_arcsec(rho)
    if rho <= 0 or not math.isfinite(rho):
        raise VsxGaiaCrossmatchError(f"VSX->Gaia cross-match refused: invalid rho={rho}")

    r_max = _acceptance_radius_arcsec(rho)

    fit_degenerate_warn = False
    try:
        fit = fit_rayleigh_mixture_from_separations(seps_arcsec, rho, strict_degeneracy=False)
    except VsxGaiaCrossmatchError as exc:
        raise VsxGaiaCrossmatchError(f"VSX->Gaia cross-match refused: mixture fit failed: {exc}") from exc
    deg_msg = _assess_fit_degeneracy(seps_arcsec, fit, rho)
    if deg_msg:
        fit_degenerate_warn = True
        notes.append(f"fit degenerate (ranking only): {deg_msg[:120]}")

    sn_eff = math.sqrt(fit.sigma_narrow_arcsec**2 + sigma_pm**2)
    sb_eff = math.sqrt(fit.sigma_broad_arcsec**2 + sigma_pm**2)

    vcoo = SkyCoord(ra=vsx_ra_deg * u.deg, dec=vsx_dec_deg * u.deg, frame="icrs")
    gcoo = SkyCoord(ra=ga_ra * u.deg, dec=ga_dec * u.deg, frame="icrs")
    idx_vsx, idx_gaia, sep2d, _ = search_around_sky(vcoo, gcoo, r_max * u.arcsec)
    seps_cand = sep2d.to(u.arcsec).value.astype(float)

    candidates: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for iv, ig, sep in zip(idx_vsx, idx_gaia, seps_cand, strict=False):
        candidates[int(iv)].append((int(ig), float(sep)))

    mult: dict[str, int] = {"0": 0, "1": 0, "2": 0, "3plus": 0}
    rows: list[VsxGaiaMatchRow] = []
    masterstars_accepted = 0
    accepted_seps: list[float] = []

    for i in range(n_vsx):
        cands = candidates.get(i, [])
        n_c = len(cands)
        if n_c == 0:
            mult["0"] += 1
            rows.append(VsxGaiaMatchRow("", float("nan"), 0.0, "", False))
            continue
        if n_c == 1:
            mult["1"] += 1
        elif n_c == 2:
            mult["2"] += 1
        else:
            mult["3plus"] += 1

        if n_c == 1:
            j, sep = cands[0]
        else:
            j, sep = max(
                cands,
                key=lambda t: _reliability_from_sep(
                    float(t[1]), fit.q_fit, fit.w_fit, sn_eff, sb_eff, rho
                ),
            )

        cid = normalize_gaia_source_id(gaia_source_id[j]) if 0 <= j < n_gaia else ""
        rel = float(
            _reliability_from_sep(float(sep), fit.q_fit, fit.w_fit, sn_eff, sb_eff, rho)
        )
        if cid:
            if cid in ms_ids:
                masterstars_accepted += 1
            accepted_seps.append(float(sep))
            rows.append(
                VsxGaiaMatchRow(
                    catalog_id=cid,
                    sep_arcsec=float(sep),
                    reliability=rel,
                    quality=_quality_from_reliability(rel),
                    accepted=True,
                )
            )
        else:
            rows.append(
                VsxGaiaMatchRow(
                    catalog_id="",
                    sep_arcsec=float(sep),
                    reliability=rel,
                    quality=_quality_from_reliability(rel),
                    accepted=False,
                )
            )

    accepted_mask = np.array([r.accepted for r in rows], dtype=bool)
    realized_contam = _chance_contamination_fraction(
        np.asarray(accepted_seps, dtype=float), rho
    ) if accepted_seps else 0.0
    multi_candidate_n = int(mult["2"] + mult["3plus"])
    multi_candidate_frac = float(multi_candidate_n / n_vsx) if n_vsx > 0 else float("nan")

    outcome_check = "ok"
    ms_eligible = 0
    if ms_ids:
        for i in range(n_vsx):
            for j, _sep in candidates.get(i, []):
                if j < 0 or j >= n_gaia:
                    continue
                cid = str(normalize_gaia_source_id(gaia_source_id[j])).strip()
                if cid in ms_ids:
                    ms_eligible += 1
                    break
        if ms_eligible > 0:
            frac = float(masterstars_accepted) / float(ms_eligible)
            if frac < MASTERSTARS_RECOVERY_WARN_FRAC:
                outcome_check = "warn_masterstars_low"
                notes.append(
                    f"G3 outcome WARN: masterstars_accepted={masterstars_accepted} "
                    f"masterstars_eligible={ms_eligible} "
                    f"({100.0 * frac:.1f}% < {100.0 * MASTERSTARS_RECOVERY_WARN_FRAC:.0f}% threshold)"
                )
                LOGGER.warning(
                    "VSX-GAIA XM outcome WARN: accepted %d/%d eligible DAO cross-IDs (%.1f%%)",
                    masterstars_accepted,
                    ms_eligible,
                    100.0 * frac,
                )
        elif masterstars_accepted == 0:
            outcome_check = "warn_masterstars_low"
            notes.append("G3 outcome WARN: no eligible masterstar cross-IDs within contamination radius")

    q_before = _sep_quantiles(seps_before)
    q_after = _sep_quantiles(seps_arcsec)
    q_accepted = _sep_quantiles(np.asarray(accepted_seps, dtype=float))

    pm_log = "propagated" if pm_path == "propagated" else "broadened"
    LOGGER.info(
        "VSX-GAIA XM: n_vsx=%d n_gaia=%d rho=%.1f deg^-2 mean_nn=%.1f\" r_max=%.2f\" "
        "Q=%.2f w=%.2f sigma_n=%.2f\" sigma_b=%.2f\" accepted=%d contamination=%.3f%% "
        "cand_mult=0:%d 1:%d 2:%d 3+:%d multi=%.2f%% "
        "pm_path=%s pm_cols=%s pm_finite=%d vsx_epoch=%.1f gaia_epoch=%.1f "
        "masterstars=%d/%d outcome=%s gaia_db_max_g=%.1f",
        n_vsx,
        n_gaia,
        rho,
        mean_nn,
        r_max,
        fit.q_fit,
        fit.w_fit,
        sn_eff,
        sb_eff,
        int(np.sum(accepted_mask)),
        100.0 * realized_contam,
        mult["0"],
        mult["1"],
        mult["2"],
        mult["3plus"],
        100.0 * multi_candidate_frac if math.isfinite(multi_candidate_frac) else float("nan"),
        pm_log,
        pm_columns_present,
        n_pm_finite,
        float(vsx_match_epoch),
        float(GAIA_EPOCH),
        masterstars_accepted,
        ms_eligible if ms_ids else len(ms_ids),
        outcome_check,
        float(gaia_db_max_g),
    )
    if pm_path == "broadened":
        notes.append(
            f"PM broadened: pm_columns_present={pm_columns_present} n_pm_finite={n_pm_finite}; "
            f"Gaia propagated GAIA_EPOCH={GAIA_EPOCH} -> VSX epoch={vsx_match_epoch:.1f} assumed J2000"
        )
    elif q_before and q_after:
        notes.append(
            f"PM propagated {n_pm_finite} stars: sep p50 {q_before.get('p50', float('nan')):.2f}\" "
            f"-> {q_after.get('p50', float('nan')):.2f}\""
        )
    if n_vsx_faint > 0:
        notes.append(f"{n_vsx_faint} VSX rows have mag_max > gaia_db_max_g={gaia_db_max_g:.1f}")

    diag = VsxGaiaCrossmatchDiagnostics(
        q_fit=float(fit.q_fit),
        w_fit=float(fit.w_fit),
        sigma_narrow_arcsec=float(sn_eff),
        sigma_broad_arcsec=float(sb_eff),
        rho_per_deg2=float(rho),
        mean_nn_arcsec=float(mean_nn),
        pm_path=str(pm_path),
        pm_columns_present=bool(pm_columns_present),
        n_pm_finite=int(n_pm_finite),
        n_vsx=n_vsx,
        n_gaia=n_gaia,
        n_accepted=int(np.sum(accepted_mask)),
        expected_contamination_fraction=float(realized_contam),
        r_max_arcsec=float(r_max),
        candidate_multiplicity=dict(mult),
        multi_candidate_fraction=float(multi_candidate_frac),
        fit_degenerate_warn=bool(fit_degenerate_warn),
        converged=True,
        field_area_deg2=float(field_area_deg2),
        gaia_db_max_g=float(gaia_db_max_g),
        vsx_fainter_than_gaia_db=n_vsx_faint,
        sep_quantiles_before_pm=q_before,
        sep_quantiles_after_pm=q_after,
        sep_quantiles_accepted=q_accepted,
        masterstars_in_frame=len(ms_ids),
        masterstars_eligible=ms_eligible,
        masterstars_accepted=masterstars_accepted,
        outcome_check=outcome_check,
        notes=notes,
    )
    return rows, diag


# Backward-compatible alias for imports
_field_area_deg2_from_wcs = field_area_deg2_from_wcs

