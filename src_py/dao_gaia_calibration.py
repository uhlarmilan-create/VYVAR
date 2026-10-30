"""DAO-Gaia matching tolerances of a setup (DAO-GAIA-ERA-01).

Identity radius from the plate scale, the FWHM and the plate-solve RMS; pixel tolerances of the
MASTERSTAR pass-2 / census matching from the astrometric residuals of the pass-1 Gaia-DAO pairs
(``catalog_match``). The per-setup certificate with empty-sky audits (draft 516 sandbox) was
development scaffolding and is gone (CODE-REVIEW-2026-10 step 6).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


DEFAULT_MATCH_K = 1.7
DEFAULT_CENTROID_FLOOR_PX = 1.0
DEFAULT_CENTROID_CAP_PX = 3.0
PX_ROUND_STEP = 0.5


@dataclass(frozen=True)
class PopulationStats:
    """Measured residual population for one tolerance derivation (D-A)."""

    name: str
    n: int
    p50_px: float
    p95_px: float
    g_band: str | None = None
    states: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DerivedTolerances:
    """Pixel tolerances derived from measured setup inputs (D-A)."""

    residual_p95_px: float
    match_radius_px: float
    pass2_center_tol_px: float
    lock_pair_tol_px: float
    lock_leftover_radius_px: float
    plate_scale_arcsec_per_px: float
    fwhm_px: float
    pass1_sigma: float
    pass2_sigma: float
    detection_identity: PopulationStats | None = None
    seed_centroid: PopulationStats | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.seed_centroid is None:
            d.pop("seed_centroid", None)
        return d


def _round_px(value: float, *, step: float = PX_ROUND_STEP) -> float:
    if not math.isfinite(value):
        return float("nan")
    return round(float(value) / step) * step


def _clamp_lo_hi(value: float, lo: float, hi: float) -> float:
    """Clamp ``value`` into ``[lo, hi]``. Not param_resolver SANITY lookup."""
    return max(lo, min(hi, float(value)))


def plate_scale_arcsec_per_px_from_wcs_nan(wcs_obj: Any) -> float:
    """:func:`optics.plate_scale_arcsec_per_px_from_wcs` with NaN instead of None (G04-30: one formula)."""
    from optics import plate_scale_arcsec_per_px_from_wcs  # noqa: PLC0415

    v = plate_scale_arcsec_per_px_from_wcs(wcs_obj)
    return float("nan") if v is None else float(v)


def solve_rms_px_from_fits_header(hdr: Any) -> float | None:
    """Fit RMS [px] of the astrosolve WCS (``VY_SOLRM``; WCS-CONFIRM-2026-10: every MASTERSTAR
    WCS comes from astrosolve). None when absent."""
    try:
        v = float(hdr.get("VY_SOLRM"))
    except (TypeError, ValueError, AttributeError):
        return None
    return v if math.isfinite(v) and v >= 0 else None


def catalog_identity_radius_arcsec(
    *,
    fwhm_dao_px: float,
    plate_scale_arcsec_per_px: float,
    solve_rms_px: float | None = None,
) -> tuple[float, dict[str, Any]]:
    """IDENTITY-2026-10: DAO -> Gaia identity radius = max(1 px, 1 x FWHM), in arcsec.

    Pixel based, so it is the same physical tolerance on every rig (before: max(12", 3 FWHM),
    then a fixed 4.5" tightening = 0.46 px at 9.8"/px and 8 px at 0.56"/px). The solved WCS is
    good to ~0.4 px; within one FWHM a detection is assigned the brightest Gaia star (see
    catalog_match), so faint catalogue neighbours cannot steal a bright star's identity."""
    fw = float(fwhm_dao_px) if math.isfinite(float(fwhm_dao_px)) else 1.0
    r_px = max(1.0, fw)
    try:
        scale = float(plate_scale_arcsec_per_px)
    except (TypeError, ValueError):
        scale = float("nan")
    used = r_px * scale if math.isfinite(scale) and scale > 0 else float("nan")
    try:
        rms = float(solve_rms_px) if solve_rms_px is not None else float("nan")
    except (TypeError, ValueError):
        rms = float("nan")
    inputs = {
        "radius_px": float(r_px),
        "fwhm_dao_px": float(fw),
        "plate_scale_arcsec_per_px": float(scale) if math.isfinite(scale) else None,
        "solve_rms_px": float(rms) if math.isfinite(rms) else None,
        "used_arcsec": float(used) if math.isfinite(used) else None,
    }
    return float(used), inputs


def compute_pass1_astrometric_residuals_px(
    dao_x: np.ndarray,
    dao_y: np.ndarray,
    gaia_x: np.ndarray,
    gaia_y: np.ndarray,
    *,
    coarse_match_px: float,
) -> np.ndarray:
    """Radial residuals [px] for pass-1 DAO matched to nearest on-chip Gaia (locked-identity proxy)."""
    dx = np.asarray(dao_x, dtype=np.float64).ravel()
    dy = np.asarray(dao_y, dtype=np.float64).ravel()
    gx = np.asarray(gaia_x, dtype=np.float64).ravel()
    gy = np.asarray(gaia_y, dtype=np.float64).ravel()
    ok_d = np.isfinite(dx) & np.isfinite(dy)
    ok_g = np.isfinite(gx) & np.isfinite(gy)
    if not (bool(ok_d.any()) and bool(ok_g.any())):
        return np.asarray([], dtype=np.float64)
    tree = cKDTree(np.column_stack([gx[ok_g], gy[ok_g]]))
    dist, _ = tree.query(
        np.column_stack([dx[ok_d], dy[ok_d]]),
        distance_upper_bound=float(coarse_match_px),
    )
    fin = np.isfinite(dist) & (dist <= float(coarse_match_px))
    return np.asarray(dist[fin], dtype=np.float64)


def _population_stats(name: str, dr: np.ndarray, *, g_band: str | None = None, states: str | None = None) -> PopulationStats:
    if dr.size == 0:
        return PopulationStats(name=name, n=0, p50_px=float("nan"), p95_px=float("nan"), g_band=g_band, states=states)
    return PopulationStats(
        name=name,
        n=int(dr.size),
        p50_px=float(np.percentile(dr, 50)),
        p95_px=float(np.percentile(dr, 95)),
        g_band=g_band,
        states=states,
    )


def derive_tolerances_from_residuals(
    identity_dr_px: np.ndarray,
    seed_centroid_dr_px: np.ndarray,
    *,
    fwhm_px: float,
    plate_scale_arcsec_per_px: float,
    pass1_sigma: float,
    pass2_sigma: float,
    match_k: float = DEFAULT_MATCH_K,
    centroid_floor_px: float = DEFAULT_CENTROID_FLOOR_PX,
    centroid_cap_px: float = DEFAULT_CENTROID_CAP_PX,
    identity_states: str = "pass1+pass2|all_G|matched",
    seed_states: str = "DETECTED_P2|Gaia_seed_position",
) -> DerivedTolerances:
    """Derive match radius from full matched identity p95; centroid QA from seed acceptances p95."""
    if identity_dr_px.size == 0:
        identity_p95 = float("nan")
    else:
        identity_p95 = float(np.percentile(identity_dr_px, 95))

    if seed_centroid_dr_px.size == 0:
        seed_p95 = float("nan")
    else:
        seed_p95 = float(np.percentile(seed_centroid_dr_px, 95))

    if not math.isfinite(identity_p95):
        identity_p95 = 1.78
    if not math.isfinite(seed_p95):
        seed_p95 = 1.9

    match_px = _round_px(float(match_k) * float(identity_p95))
    centroid_raw = _clamp_lo_hi(seed_p95, centroid_floor_px, centroid_cap_px)
    centroid_px = _round_px(centroid_raw)
    match_px = max(match_px, centroid_px)

    pop_id = _population_stats("detection_identity", identity_dr_px, states=identity_states)
    pop_seed = _population_stats("seed_centroid", seed_centroid_dr_px, states=seed_states)

    return DerivedTolerances(
        residual_p95_px=float(identity_p95),
        match_radius_px=float(match_px),
        pass2_center_tol_px=float(centroid_px),
        lock_pair_tol_px=float(match_px),
        lock_leftover_radius_px=float(match_px),
        plate_scale_arcsec_per_px=float(plate_scale_arcsec_per_px),
        fwhm_px=float(fwhm_px),
        pass1_sigma=float(pass1_sigma),
        pass2_sigma=float(pass2_sigma),
        detection_identity=pop_id,
        seed_centroid=pop_seed,
    )


