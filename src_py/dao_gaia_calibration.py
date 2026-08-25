"""DAO-GAIA-ERA-01: per-setup self-calibration certificate (D-A).

Derives pixel tolerances from measured plate scale, FWHM, and astrometric
residuals of pass-1 Gaia-DAO pairs; auto-generates empty-sky audit positions;
evaluates INV-DET-FALSEFILL-01 / INV-SEED-FALSEFILL-01 at MS build time.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from masterstar_gaia_accounting import (
    ForcedSeedAcceptParams,
    Pass2AcceptParams,
    SOURCE_FORCED_SEED,
    dao_pass2_try_at_position,
    forced_seed_accept,
    forced_seed_measure_at_position,
    lock_existing_and_leftover_assign,
)

CERT_FILENAME = "dao_gaia_calibration.json"
INV_DET_FALSEFILL = "INV-DET-FALSEFILL-01"
INV_SEED_FALSEFILL = "INV-SEED-FALSEFILL-01"
DEFAULT_FALSE_ACCEPT_MAX = 0.01
DEFAULT_EMPTY_SKY_N = 2200
DEFAULT_MATCH_K = 1.7
DEFAULT_CENTROID_FLOOR_PX = 1.0
DEFAULT_CENTROID_CAP_PX = 3.0
DIAGNOSTIC_CENTROID_CAP_PX = 3.0
PX_ROUND_STEP = 0.5


@dataclass(frozen=True)
class DiagnosticPopulationStats:
    """Auditable diagnostic-mode residual population (A-fix 4)."""

    name: str
    n: int
    p50_px: float
    p95_px: float
    n_raw: int
    p50_raw_px: float
    p95_raw_px: float
    tail_estimate_px: float
    tail_method: str
    measurement_mode: str
    diagnostic_radius_px: float | None = None
    g_band: str | None = None
    states: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    forced_seed_centroid_max_px: float
    plate_scale_arcsec_per_px: float
    fwhm_px: float
    pass1_sigma: float
    pass2_sigma: float
    detection_identity: PopulationStats | None = None
    seed_centroid: PopulationStats | None = None
    faint_star_centroid: PopulationStats | None = None  # legacy alias
    diagnostic: DiagnosticTolerances | None = None

    def to_dict(self) -> dict[str, float]:
        d = asdict(self)
        if self.detection_identity is not None:
            d["detection_identity"] = self.detection_identity.to_dict()
        seed = self.seed_centroid or self.faint_star_centroid
        if seed is not None:
            d["seed_centroid"] = seed.to_dict()
            d["faint_star_centroid"] = seed.to_dict()
        else:
            d.pop("faint_star_centroid", None)
            d.pop("seed_centroid", None)
        if self.diagnostic is not None:
            d["diagnostic"] = {
                "detection_identity": self.diagnostic.detection_identity.to_dict(),
                "seed_centroid": self.diagnostic.seed_centroid.to_dict(),
            }
        return d


def effective_tol_stamps(
    derived: dict[str, Any] | None,
    cfg: Any,
    *,
    fwhm_px: float,
    census_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """S2: effective DAO-Gaia tols vs config defaults for pipeline_meta."""
    import math

    d = derived if isinstance(derived, dict) else {}
    cm = census_meta if isinstance(census_meta, dict) else {}
    fw = float(fwhm_px) if math.isfinite(float(fwhm_px)) and float(fwhm_px) > 0 else 3.5
    lock_def = float(getattr(cfg, "masterstar_lock_pair_tol_px", 3.0))
    p2_def = float(getattr(cfg, "masterstar_dao_pass2_center_tol_px", 2.0))
    leftover_def = float(getattr(cfg, "masterstar_lock_leftover_radius_px", 3.0))

    def _eff(key: str, default: float) -> float:
        raw = d.get(key)
        if raw is None:
            raw = cm.get(f"{key}_effective", cm.get(key))
        try:
            val = float(raw)
        except (TypeError, ValueError):
            return float(default)
        return val if math.isfinite(val) else float(default)

    ident = cm.get("identity_fail_px")
    try:
        ident_f = float(ident) if ident is not None else 3.0 * fw
    except (TypeError, ValueError):
        ident_f = 3.0 * fw
    if not math.isfinite(ident_f) or ident_f <= 0:
        ident_f = 3.0 * fw
    return {
        "lock_pair_tol_px": _eff("lock_pair_tol_px", lock_def),
        "lock_pair_tol_px_config_default": lock_def,
        "pass2_center_tol_px": _eff("pass2_center_tol_px", p2_def),
        "pass2_center_tol_px_config_default": p2_def,
        "identity_fail_px": float(ident_f),
        "identity_fail_px_formula": "3 * FWHM_dao_px",
        "match_radius_px": _eff("match_radius_px", leftover_def),
        "match_radius_px_config_default": leftover_def,
        "lock_leftover_radius_px": _eff("lock_leftover_radius_px", leftover_def),
        "lock_leftover_radius_px_config_default": leftover_def,
    }


@dataclass(frozen=True)
class DiagnosticTolerances:
    """Diagnostic distributions recorded in certificate (A-fix 4)."""

    detection_identity: DiagnosticPopulationStats
    seed_centroid: DiagnosticPopulationStats


@dataclass
class EmptySkyAudit:
    n_positions: int
    pass2_accept: int
    pass2_rate: float
    seed_accept: int
    seed_rate: float
    inv_det: str
    inv_seed: str


@dataclass
class ValidationGateResult:
    status: str
    fail_reason: str | None
    max_regression_pp: float
    hand_scores: dict[str, Any]
    derived_scores: dict[str, Any]
    regressions: dict[str, float]
    g2_pass: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DaoGaiaCalibrationCertificate:
    setup: str
    built_utc: str
    status: str
    fail_reason: str | None
    derived: DerivedTolerances
    empty_sky: EmptySkyAudit
    validation: ValidationGateResult | None = None
    inputs: dict[str, Any] = field(default_factory=dict)
    identity: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "setup": self.setup,
            "built_utc": self.built_utc,
            "status": self.status,
            "fail_reason": self.fail_reason,
            "derived": self.derived.to_dict(),
            "empty_sky": asdict(self.empty_sky),
            "validation": self.validation.to_dict() if self.validation is not None else None,
            "inputs": self.inputs,
        }
        if self.identity:
            payload.update(self.identity)
        return payload


def _round_px(value: float, *, step: float = PX_ROUND_STEP) -> float:
    if not math.isfinite(value):
        return float("nan")
    return round(float(value) / step) * step


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def plate_scale_arcsec_per_px_from_wcs(wcs_obj: Any) -> float:
    from astropy.wcs.utils import proj_plane_pixel_scales

    try:
        scales_deg = proj_plane_pixel_scales(wcs_obj)
        arcsec = float(np.mean(scales_deg)) * 3600.0
        if math.isfinite(arcsec) and arcsec > 0:
            return arcsec
    except Exception:  # noqa: BLE001
        pass
    return float("nan")


def solve_rms_px_from_fits_header(hdr: Any) -> float | None:
    """Best-effort solve RMS [px] from FITS HISTORY (SIP lin= or NN refine line)."""
    import re

    first_lin: float | None = None
    nn_rms: float | None = None
    try:
        cards = list(hdr.cards)
    except Exception:  # noqa: BLE001
        return None
    for c in cards:
        if str(getattr(c, "keyword", "")) != "HISTORY":
            continue
        s = str(getattr(c, "value", "") or "")
        if first_lin is None:
            m = re.search(r"lin=([0-9]+(?:\.[0-9]+)?)", s)
            if m:
                try:
                    first_lin = float(m.group(1))
                except ValueError:
                    first_lin = None
        if nn_rms is None:
            m = re.search(r"Mean residual error\s*=\s*([0-9]+(?:\.[0-9]+)?)", s, re.I)
            if m:
                try:
                    nn_rms = float(m.group(1))
                except ValueError:
                    nn_rms = None
    if nn_rms is not None and math.isfinite(nn_rms):
        return float(nn_rms)
    if first_lin is not None and math.isfinite(first_lin):
        return float(first_lin)
    return None


def catalog_match_radius_d1_arcsec(
    *,
    solve_rms_px: float | None,
    fwhm_dao_px: float,
    plate_scale_arcsec_per_px: float,
    floor_arcsec: float = 12.0,
) -> tuple[float, dict[str, Any]]:
    """D1: one-pass catalog match radius. No match-rate widening.

    radius = max(floor, 3 x FWHM_dao_px x plate_scale).

    ``solve_rms_px`` is stamped as a diagnostic only. It belongs to the
    identity-gate WCS, not to a radius the 3xFWHM gate strips anyway.
    """
    fw = max(0.5, float(fwhm_dao_px))
    try:
        rms = float(solve_rms_px) if solve_rms_px is not None else float("nan")
    except (TypeError, ValueError):
        rms = float("nan")
    inner_px = fw
    try:
        scale = float(plate_scale_arcsec_per_px)
    except (TypeError, ValueError):
        scale = float("nan")
    formula = 3.0 * inner_px * scale if math.isfinite(scale) and scale > 0 else float("nan")
    used = float(floor_arcsec)
    if math.isfinite(formula):
        used = max(float(floor_arcsec), float(formula))
    inputs = {
        "solve_rms_px": (float(rms) if math.isfinite(rms) else None),
        "fwhm_dao_px": float(fw),
        "plate_scale_arcsec_per_px": (float(scale) if math.isfinite(scale) else None),
        "inner_px": float(inner_px),
        "formula_arcsec": (float(formula) if math.isfinite(formula) else None),
        "floor_arcsec": float(floor_arcsec),
        "used_arcsec": float(used),
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


def diagnostic_crossmatch_radius_px(fwhm_px: float) -> float:
    """Wide-open astrometric crossmatch radius for calibration measurement."""
    return float(max(10.0, 2.0 * float(fwhm_px)))


def _chance_match_scale_px(n_sources: int, *, wpx: int, h: int) -> float:
    """Characteristic random-alignment scale sqrt(1/(pi*rho)) [px]."""
    area = max(float(wpx) * float(h), 1.0)
    rho = float(n_sources) / area
    if rho <= 0:
        return float("nan")
    return float(math.sqrt(1.0 / (math.pi * rho)))


def _random_match_scale_px(n_gaia: int, *, wpx: int, h: int) -> float:
    """Legacy alias for chance-match scale."""
    return _chance_match_scale_px(n_gaia, wpx=wpx, h=h)


def _tail_correct_core_separations(
    dr_raw: np.ndarray,
    *,
    tail_scale_px: float,
    astrometric_cap_px: float = 4.0,
) -> tuple[np.ndarray, str, float]:
    """Keep astrometric core; report random-match scale as tail estimate."""
    dr = np.asarray(dr_raw, dtype=np.float64)
    dr = dr[np.isfinite(dr)]
    if dr.size == 0:
        return dr, "empty", float("nan")
    tail_est = float(tail_scale_px) if math.isfinite(tail_scale_px) else float("nan")
    cap = float(astrometric_cap_px)
    core = dr[dr <= cap]
    method = (
        f"core=sep<= {cap:.1f} px astrometric envelope; "
        f"random_match_scale={tail_est:.3f} px"
    )
    if core.size == 0:
        return dr, method + "; fallback=all_raw", tail_est
    return core, method, tail_est


def _mutual_nearest_separations_px(
    ax: np.ndarray,
    ay: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
    *,
    radius_px: float,
) -> np.ndarray:
    """Mutual nearest-neighbour separations within ``radius_px``."""
    ax = np.asarray(ax, dtype=np.float64).ravel()
    ay = np.asarray(ay, dtype=np.float64).ravel()
    bx = np.asarray(bx, dtype=np.float64).ravel()
    by = np.asarray(by, dtype=np.float64).ravel()
    ok_a = np.isfinite(ax) & np.isfinite(ay)
    ok_b = np.isfinite(bx) & np.isfinite(by)
    if not (bool(ok_a.any()) and bool(ok_b.any())):
        return np.asarray([], dtype=np.float64)
    pts_a = np.column_stack([ax[ok_a], ay[ok_a]])
    pts_b = np.column_stack([bx[ok_b], by[ok_b]])
    tree_b = cKDTree(pts_b)
    tree_a = cKDTree(pts_a)
    dist_ab, idx_ab = tree_b.query(pts_a, distance_upper_bound=float(radius_px))
    dist_ba, idx_ba = tree_a.query(pts_b, distance_upper_bound=float(radius_px))
    seps: list[float] = []
    for i in range(len(pts_a)):
        j = int(idx_ab[i])
        d = float(dist_ab[i])
        if not (np.isfinite(d) and j >= 0 and d <= float(radius_px)):
            continue
        if int(idx_ba[j]) == i:
            seps.append(d)
    return np.asarray(seps, dtype=np.float64)


def _diagnostic_identity_separations_px(
    data0: np.ndarray,
    gaia_x: np.ndarray,
    gaia_y: np.ndarray,
    *,
    fwhm_px: float,
    pass1_sigma: float,
    pass2_sigma: float,
    dao_x: np.ndarray,
    dao_y: np.ndarray,
) -> np.ndarray:
    """Greedy diagnostic assign: pass1 DAO + pass2 accepted at wide-open radius."""
    h, wpx = int(data0.shape[0]), int(data0.shape[1])
    r_diag = diagnostic_crossmatch_radius_px(fwhm_px)
    gx = np.asarray(gaia_x, dtype=np.float64).ravel()
    gy = np.asarray(gaia_y, dtype=np.float64).ravel()
    dx = np.asarray(dao_x, dtype=np.float64).ravel()
    dy = np.asarray(dao_y, dtype=np.float64).ravel()
    gaia_df = pd.DataFrame({"x_gaia": gx, "y_gaia": gy, "catalog_id": [""] * len(gx)})

    seps: list[float] = []
    det_to_g, gaia_owner, _, _ = lock_existing_and_leftover_assign(
        dx, dy, gaia_df, leftover_radius_px=r_diag
    )
    for i in range(len(dx)):
        g = int(det_to_g[i])
        if g >= 0:
            seps.append(float(math.hypot(dx[i] - gx[g], dy[i] - gy[g])))

    p2_params = Pass2AcceptParams(
        sigma=float(pass2_sigma),
        center_tol_px=float(DIAGNOSTIC_CENTROID_CAP_PX),
        fwhm_px=float(fwhm_px),
    )
    for j in range(len(gx)):
        if int(gaia_owner[j]) >= 0:
            continue
        hit = dao_pass2_try_at_position(data0, float(gx[j]), float(gy[j]), wpx=wpx, h=h, params=p2_params)
        if hit.get("accepted"):
            cp = float(hit.get("centroid_px", float("nan")))
            if math.isfinite(cp):
                seps.append(cp)
    return np.asarray(seps, dtype=np.float64)


def _run_pass1_dao_detections(
    data0: np.ndarray,
    *,
    fwhm_px: float,
    pass1_sigma: float,
    gaia_x: np.ndarray,
    gaia_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Pass-1 DAOStarFinder detections (diagnostic; no production match gate)."""
    import warnings

    from astropy.stats import sigma_clipped_stats
    from photutils.detection import DAOStarFinder

    from utils import DAO_STAR_FINDER_NO_ROUNDNESS_FILTER

    h, w = int(data0.shape[0]), int(data0.shape[1])
    gx = np.asarray(gaia_x, dtype=np.float64).ravel()
    gy = np.asarray(gaia_y, dtype=np.float64).ravel()
    ok_g = np.isfinite(gx) & np.isfinite(gy)
    mask = np.ones((h, w), dtype=bool)
    if bool(ok_g.any()):
        r = max(int(round(2.0 * float(fwhm_px))), 3)
        for x, y in zip(gx[ok_g], gy[ok_g], strict=False):
            ix, iy = int(round(x)), int(round(y))
            y0, y1 = max(0, iy - r), min(h, iy + r + 1)
            x0, x1 = max(0, ix - r), min(w, ix + r + 1)
            mask[y0:y1, x0:x1] = False
    bg = data0[mask]
    bg_fin = bg[np.isfinite(bg)]
    if bg_fin.size < 1000:
        bg_fin = data0.ravel()
    _, _, sig = sigma_clipped_stats(bg_fin, sigma=3.0, maxiters=3)
    thr = max(float(pass1_sigma) * float(sig), 1e-6)
    finder = DAOStarFinder(
        fwhm=float(fwhm_px),
        threshold=float(thr),
        **DAO_STAR_FINDER_NO_ROUNDNESS_FILTER,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tbl = finder(data0)
    if tbl is None or len(tbl) == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    return (
        np.asarray(tbl["x_centroid"], dtype=np.float64),
        np.asarray(tbl["y_centroid"], dtype=np.float64),
    )


def compute_diagnostic_identity_residuals_px(
    data0: np.ndarray,
    gaia_x: np.ndarray,
    gaia_y: np.ndarray,
    *,
    fwhm_px: float,
    pass1_sigma: float,
    pass2_sigma: float,
    target_depth_g: float = 15.0,
    edge_margin_px: float = 10.0,
    dao_x: np.ndarray | None = None,
    dao_y: np.ndarray | None = None,
) -> DiagnosticPopulationStats:
    """Diagnostic-mode identity residuals: greedy DAO<->Gaia at wide radius, tail-corrected core p95."""
    h, wpx = int(data0.shape[0]), int(data0.shape[1])
    r_diag = diagnostic_crossmatch_radius_px(fwhm_px)
    gx = np.asarray(gaia_x, dtype=np.float64).ravel()
    gy = np.asarray(gaia_y, dtype=np.float64).ravel()
    ok_g = np.isfinite(gx) & np.isfinite(gy)
    gx, gy = gx[ok_g], gy[ok_g]

    if dao_x is None or dao_y is None:
        dx, dy = _run_pass1_dao_detections(
            data0,
            fwhm_px=float(fwhm_px),
            pass1_sigma=float(pass1_sigma),
            gaia_x=gx,
            gaia_y=gy,
        )
    else:
        dx = np.asarray(dao_x, dtype=np.float64).ravel()
        dy = np.asarray(dao_y, dtype=np.float64).ravel()

    seps_all = _diagnostic_identity_separations_px(
        data0,
        gx,
        gy,
        fwhm_px=float(fwhm_px),
        pass1_sigma=float(pass1_sigma),
        pass2_sigma=float(pass2_sigma),
        dao_x=dx,
        dao_y=dy,
    )

    tail_scale = _chance_match_scale_px(len(gx), wpx=wpx, h=h)
    core, tail_method, tail_est = _tail_correct_core_separations(seps_all, tail_scale_px=tail_scale)

    def _pct(arr: np.ndarray, q: float) -> float:
        return float(np.percentile(arr, q)) if arr.size else float("nan")

    return DiagnosticPopulationStats(
        name="detection_identity",
        n=int(core.size),
        p50_px=_pct(core, 50),
        p95_px=_pct(core, 95),
        n_raw=int(seps_all.size),
        p50_raw_px=_pct(seps_all, 50),
        p95_raw_px=_pct(seps_all, 95),
        tail_estimate_px=float(tail_est),
        tail_method=tail_method,
        measurement_mode="diagnostic|greedy_assign+pass2_accepted|tail_core",
        diagnostic_radius_px=float(r_diag),
        states="pass1+pass2|all_G|DAO_gaia_crossmatch",
    )


def compute_diagnostic_seed_centroid_offsets_px(
    data0: np.ndarray,
    gaia_x: np.ndarray,
    gaia_y: np.ndarray,
    gaia_g: np.ndarray,
    *,
    pass1_dao_x: np.ndarray,
    pass1_dao_y: np.ndarray,
    fwhm_px: float,
    pass2_sigma: float,
    seed_snr_min: float,
    target_depth_g: float = 15.0,
    edge_margin_px: float = 10.0,
) -> DiagnosticPopulationStats:
    """Diagnostic seed truth: pass2/forced-seed acceptances (SNR gate only; centroid cap 3 px)."""
    h, wpx = int(data0.shape[0]), int(data0.shape[1])
    r_diag = diagnostic_crossmatch_radius_px(fwhm_px)
    gx = np.asarray(gaia_x, dtype=np.float64).ravel()
    gy = np.asarray(gaia_y, dtype=np.float64).ravel()
    gg = np.asarray(gaia_g, dtype=np.float64).ravel()
    dx = np.asarray(pass1_dao_x, dtype=np.float64).ravel()
    dy = np.asarray(pass1_dao_y, dtype=np.float64).ravel()

    owned = np.zeros(len(gx), dtype=bool)
    if dx.size and gx.size:
        tree_g = cKDTree(np.column_stack([gx, gy]))
        dist, idx = tree_g.query(np.column_stack([dx, dy]), distance_upper_bound=r_diag)
        for j in idx[np.isfinite(dist) & (dist <= r_diag)]:
            if j >= 0:
                owned[int(j)] = True

    p2_params = Pass2AcceptParams(
        sigma=float(pass2_sigma),
        center_tol_px=float(DIAGNOSTIC_CENTROID_CAP_PX),
        fwhm_px=float(fwhm_px),
    )
    seed_params = ForcedSeedAcceptParams(
        centroid_max_px=float(DIAGNOSTIC_CENTROID_CAP_PX),
        snr_min=float(seed_snr_min),
    )
    offsets: list[float] = []
    for j in range(len(gx)):
        if owned[j]:
            continue
        if not (math.isfinite(gg[j]) and float(gg[j]) <= float(target_depth_g)):
            continue
        xg, yg = float(gx[j]), float(gy[j])
        if xg < edge_margin_px or yg < edge_margin_px or xg > wpx - edge_margin_px or yg > h - edge_margin_px:
            continue
        hit = dao_pass2_try_at_position(data0, xg, yg, wpx=wpx, h=h, params=p2_params)
        if hit.get("accepted"):
            cp = float(hit.get("centroid_px", float("nan")))
            if math.isfinite(cp):
                offsets.append(cp)
                continue
        meas = forced_seed_measure_at_position(data0, xg, yg, fwhm_px=float(fwhm_px), params=seed_params)
        snr = float(meas.get("snr", float("nan")))
        cp = float(meas.get("centroid_px", float("nan")))
        if math.isfinite(snr) and snr >= float(seed_snr_min) and math.isfinite(cp):
            offsets.append(cp)

    dr = np.asarray(offsets, dtype=np.float64)
    tail_scale = _random_match_scale_px(int(np.sum(np.isfinite(gx))), wpx=wpx, h=h)
    core, tail_method, tail_est = _tail_correct_core_separations(dr, tail_scale_px=tail_scale)

    def _pct(arr: np.ndarray, q: float) -> float:
        return float(np.percentile(arr, q)) if arr.size else float("nan")

    return DiagnosticPopulationStats(
        name="seed_centroid",
        n=int(core.size),
        p50_px=_pct(core, 50),
        p95_px=_pct(core, 95),
        n_raw=int(dr.size),
        p50_raw_px=_pct(dr, 50),
        p95_raw_px=_pct(dr, 95),
        tail_estimate_px=float(tail_est),
        tail_method=tail_method,
        measurement_mode="diagnostic|snr_only_before_centroid_gate",
        states="pass2+forced_seed|unowned_Gaia_seed_position",
    )


def compute_matched_detection_identity_offsets_px(
    ms_df: pd.DataFrame,
    census_df: pd.DataFrame,
) -> np.ndarray:
    """GAIA-00-style identity residuals: all matched detections (pass1+pass2, all G).

    Sampled without selection on the gated quantity (catalog_id match only).
    """
    if ms_df is None or census_df is None or ms_df.empty or census_df.empty:
        return np.asarray([], dtype=np.float64)
    ms = ms_df.copy()
    ms["catalog_id"] = ms.get("catalog_id", pd.Series([""] * len(ms))).astype(str).str.strip()
    ms = ms.loc[ms["catalog_id"].ne("")].copy()
    if ms.empty:
        return np.asarray([], dtype=np.float64)
    cens = census_df.copy()
    cens["catalog_id"] = cens.get("catalog_id", pd.Series([""] * len(cens))).astype(str).str.strip()
    j = ms.merge(cens[["catalog_id", "x_gaia", "y_gaia"]], on="catalog_id", how="inner")
    if j.empty:
        return np.asarray([], dtype=np.float64)
    dx = pd.to_numeric(j.get("x"), errors="coerce") - pd.to_numeric(j["x_gaia"], errors="coerce")
    dy = pd.to_numeric(j.get("y"), errors="coerce") - pd.to_numeric(j["y_gaia"], errors="coerce")
    dr = np.hypot(dx.to_numpy(dtype=np.float64), dy.to_numpy(dtype=np.float64))
    return dr[np.isfinite(dr)]


def compute_seed_acceptance_centroid_offsets_px(
    ms_df: pd.DataFrame,
    census_df: pd.DataFrame,
    *,
    data0: np.ndarray | None = None,
    fwhm_px: float | None = None,
    pass2_sigma: float = 4.0,
    pass2_center_tol_px: float = 2.0,
    seed_snr_min: float = 4.0,
    seed_centroid_cap_px: float = 3.0,
    faint_g_lo: float = 13.0,
    faint_g_hi: float = 15.0,
) -> tuple[np.ndarray, str]:
    """Seed QA population: pass2/seed acceptances at Gaia seed positions (GAIA-01 truth).

    Primary: DETECTED_P2 + FORCED_SEED census rows. Fallback when empty: pass2-only
    faint-end matched detections (proxy for fresh setups without seed history).
    Returns (offsets_px, population_label).
    """
    if ms_df is None or census_df is None or census_df.empty:
        return np.asarray([], dtype=np.float64), "empty"

    ms = ms_df.copy()
    ms["catalog_id"] = ms.get("catalog_id", pd.Series([""] * len(ms))).astype(str).str.strip()
    cens = census_df.copy()
    cens["catalog_id"] = cens.get("catalog_id", pd.Series([""] * len(cens))).astype(str).str.strip()
    cens["source_state"] = cens.get("source_state", pd.Series([""] * len(cens))).astype(str).str.strip()

    seed_states = {"DETECTED_P2", SOURCE_FORCED_SEED}
    seed_cens = cens.loc[cens["source_state"].isin(seed_states)].copy()
    offsets: list[float] = []

    h = wpx = 0
    if data0 is not None:
        h, wpx = int(data0.shape[0]), int(data0.shape[1])
    p2_params = Pass2AcceptParams(
        sigma=float(pass2_sigma),
        center_tol_px=float(pass2_center_tol_px),
        fwhm_px=float(fwhm_px or 3.5),
    )
    seed_params = ForcedSeedAcceptParams(
        centroid_max_px=float(seed_centroid_cap_px),
        snr_min=float(seed_snr_min),
    )

    ms_by_id = ms.set_index("catalog_id", drop=False) if not ms.empty else pd.DataFrame()

    if not seed_cens.empty:
        for _, row in seed_cens.iterrows():
            st = str(row["source_state"])
            cid = str(row["catalog_id"])
            xg = float(row["x_gaia"])
            yg = float(row["y_gaia"])
            if st == SOURCE_FORCED_SEED:
                sc = pd.to_numeric(row.get("seed_centroid_px"), errors="coerce")
                if math.isfinite(float(sc)):
                    offsets.append(float(sc))
                    continue
                if data0 is not None and h > 0 and wpx > 0:
                    meas = forced_seed_measure_at_position(
                        data0, xg, yg, fwhm_px=float(fwhm_px or 3.5), params=seed_params
                    )
                    offsets.append(
                        float(
                            math.hypot(
                                float(meas["cx"]) - xg,
                                float(meas["cy"]) - yg,
                            )
                        )
                    )
                continue
            # DETECTED_P2: MS detection offset when owned; else live pass2 at Gaia seed.
            if cid in ms_by_id.index:
                mrow = ms_by_id.loc[cid]
                if isinstance(mrow, pd.DataFrame):
                    mrow = mrow.iloc[0]
                dx = pd.to_numeric(mrow.get("x"), errors="coerce") - xg
                dy = pd.to_numeric(mrow.get("y"), errors="coerce") - yg
                off = float(math.hypot(float(dx), float(dy))) if math.isfinite(dx) and math.isfinite(dy) else float("nan")
                if math.isfinite(off):
                    offsets.append(off)
                    continue
            if data0 is not None and h > 0 and wpx > 0:
                hit = dao_pass2_try_at_position(data0, xg, yg, wpx=wpx, h=h, params=p2_params)
                if hit.get("accepted"):
                    offsets.append(
                        float(
                            math.hypot(
                                float(hit["x_det"]) - xg,
                                float(hit["y_det"]) - yg,
                            )
                        )
                    )

    label = "DETECTED_P2|FORCED_SEED|Gaia_seed_position"
    if offsets:
        return np.asarray(offsets, dtype=np.float64), label

    # Fresh-setup proxy: pass2-only faint-end matched detections.
    g = pd.to_numeric(cens.get("g_mag"), errors="coerce")
    faint = cens.loc[
        cens["source_state"].eq("DETECTED_P2") & g.ge(float(faint_g_lo)) & g.le(float(faint_g_hi))
    ]
    if faint.empty:
        return np.asarray([], dtype=np.float64), "empty"
    j = ms.loc[pd.to_numeric(ms.get("vy_dao_pass"), errors="coerce") == 2].merge(
        faint[["catalog_id", "x_gaia", "y_gaia"]], on="catalog_id", how="inner"
    )
    if j.empty:
        return np.asarray([], dtype=np.float64), "empty"
    dx = pd.to_numeric(j.get("x"), errors="coerce") - pd.to_numeric(j["x_gaia"], errors="coerce")
    dy = pd.to_numeric(j.get("y"), errors="coerce") - pd.to_numeric(j["y_gaia"], errors="coerce")
    dr = np.hypot(dx.to_numpy(dtype=np.float64), dy.to_numpy(dtype=np.float64))
    fin = dr[np.isfinite(dr)]
    return fin, f"pass2_only_faint_proxy|G={faint_g_lo:.1f}-{faint_g_hi:.1f}"


def compute_faint_star_centroid_offsets_px(
    ms_df: pd.DataFrame,
    census_df: pd.DataFrame,
    *,
    g_lo: float = 13.0,
    g_hi: float = 15.0,
    detected_states: frozenset[str] | None = None,
) -> np.ndarray:
    """Deprecated alias: use compute_seed_acceptance_centroid_offsets_px."""
    dr, _ = compute_seed_acceptance_centroid_offsets_px(
        ms_df,
        census_df,
        faint_g_lo=float(g_lo),
        faint_g_hi=float(g_hi),
    )
    return dr


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


def derive_tolerances_from_diagnostic(
    identity_diag: DiagnosticPopulationStats,
    seed_diag: DiagnosticPopulationStats,
    *,
    fwhm_px: float,
    plate_scale_arcsec_per_px: float,
    pass1_sigma: float,
    pass2_sigma: float,
    match_k: float = DEFAULT_MATCH_K,
    centroid_floor_px: float = DEFAULT_CENTROID_FLOOR_PX,
    centroid_cap_px: float = DEFAULT_CENTROID_CAP_PX,
) -> DerivedTolerances:
    """Derive production tolerances from diagnostic-mode population p95 (A-fix 4)."""
    identity_p95 = float(identity_diag.p95_px)
    seed_p95 = float(seed_diag.p95_px)
    if not math.isfinite(identity_p95):
        identity_p95 = 1.78
    if not math.isfinite(seed_p95):
        seed_p95 = 1.9

    match_px = _round_px(float(match_k) * float(identity_p95))
    centroid_raw = _clamp(seed_p95, centroid_floor_px, centroid_cap_px)
    centroid_px = _round_px(centroid_raw)
    match_px = max(match_px, centroid_px)

    pop_id = PopulationStats(
        name="detection_identity",
        n=int(identity_diag.n),
        p50_px=float(identity_diag.p50_px),
        p95_px=float(identity_p95),
        states=identity_diag.states,
    )
    pop_seed = PopulationStats(
        name="seed_centroid",
        n=int(seed_diag.n),
        p50_px=float(seed_diag.p50_px),
        p95_px=float(seed_p95),
        states=seed_diag.states,
    )

    return DerivedTolerances(
        residual_p95_px=float(identity_p95),
        match_radius_px=float(match_px),
        pass2_center_tol_px=float(centroid_px),
        lock_pair_tol_px=float(match_px),
        lock_leftover_radius_px=float(match_px),
        forced_seed_centroid_max_px=float(centroid_px),
        plate_scale_arcsec_per_px=float(plate_scale_arcsec_per_px),
        fwhm_px=float(fwhm_px),
        pass1_sigma=float(pass1_sigma),
        pass2_sigma=float(pass2_sigma),
        detection_identity=pop_id,
        seed_centroid=pop_seed,
        faint_star_centroid=pop_seed,
        diagnostic=DiagnosticTolerances(detection_identity=identity_diag, seed_centroid=seed_diag),
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
    seed_states: str = "DETECTED_P2|FORCED_SEED|Gaia_seed_position",
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
    centroid_raw = _clamp(seed_p95, centroid_floor_px, centroid_cap_px)
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
        forced_seed_centroid_max_px=float(centroid_px),
        plate_scale_arcsec_per_px=float(plate_scale_arcsec_per_px),
        fwhm_px=float(fwhm_px),
        pass1_sigma=float(pass1_sigma),
        pass2_sigma=float(pass2_sigma),
        detection_identity=pop_id,
        seed_centroid=pop_seed,
        faint_star_centroid=pop_seed,
    )


def generate_empty_sky_positions(
    *,
    wpx: int,
    h: int,
    gaia_x: np.ndarray,
    gaia_y: np.ndarray,
    gaia_g: np.ndarray,
    det_x: np.ndarray,
    det_y: np.ndarray,
    fwhm_px: float,
    target_depth_g: float,
    match_radius_px: float,
    edge_margin_px: float,
    target_n: int = DEFAULT_EMPTY_SKY_N,
    rng: np.random.Generator | None = None,
    max_attempts: int = 80_000,
) -> pd.DataFrame:
    """GAIA-01 empty-sky recipe: no Gaia within 2x FWHM, no det within match radius, edge margin."""
    rng = rng or np.random.default_rng(516)
    gx = np.asarray(gaia_x, dtype=np.float64).ravel()
    gy = np.asarray(gaia_y, dtype=np.float64).ravel()
    gg = np.asarray(gaia_g, dtype=np.float64).ravel()
    dx = np.asarray(det_x, dtype=np.float64).ravel()
    dy = np.asarray(det_y, dtype=np.float64).ravel()

    em = float(edge_margin_px)
    xlo, xhi = em, float(wpx) - em
    ylo, yhi = em, float(h) - em
    if xhi <= xlo or yhi <= ylo:
        return pd.DataFrame(columns=["x", "y", "frame", "is_corner"])

    gaia_r = max(2.0 * float(fwhm_px), 4.0)
    depth_mask = np.isfinite(gg) & (gg <= float(target_depth_g))
    gtree: cKDTree | None = None
    if bool(depth_mask.any()):
        gtree = cKDTree(np.column_stack([gx[depth_mask], gy[depth_mask]]))

    dtree: cKDTree | None = None
    ok_d = np.isfinite(dx) & np.isfinite(dy)
    if bool(ok_d.any()):
        dtree = cKDTree(np.column_stack([dx[ok_d], dy[ok_d]]))

    rows: list[dict[str, float | str | bool]] = []
    attempts = 0
    while len(rows) < int(target_n) and attempts < int(max_attempts):
        attempts += 1
        x = float(rng.uniform(xlo, xhi))
        y = float(rng.uniform(ylo, yhi))
        if gtree is not None:
            gd, _ = gtree.query([x, y], distance_upper_bound=gaia_r)
            if np.isfinite(gd) and float(gd) <= gaia_r:
                continue
        if dtree is not None:
            dd, _ = dtree.query([x, y], distance_upper_bound=float(match_radius_px))
            if np.isfinite(dd) and float(dd) <= float(match_radius_px):
                continue
        rows.append({"x": x, "y": y, "frame": "MASTERSTAR", "is_corner": False})
    return pd.DataFrame(rows)


def run_empty_sky_audits(
    empty_df: pd.DataFrame,
    data0: np.ndarray,
    *,
    wpx: int,
    h: int,
    fwhm_px: float,
    pass2_sigma: float,
    pass2_center_tol_px: float,
    seed_centroid_max_px: float,
    seed_snr_min: float,
    false_accept_max: float = DEFAULT_FALSE_ACCEPT_MAX,
) -> EmptySkyAudit:
    """Evaluate INV-DET-FALSEFILL-01 and INV-SEED-FALSEFILL-01 on empty-sky positions."""
    n = int(len(empty_df))
    if n == 0:
        return EmptySkyAudit(
            n_positions=0,
            pass2_accept=0,
            pass2_rate=1.0,
            seed_accept=0,
            seed_rate=1.0,
            inv_det="FAIL",
            inv_seed="FAIL",
        )

    p2 = Pass2AcceptParams(
        sigma=float(pass2_sigma),
        center_tol_px=float(pass2_center_tol_px),
        fwhm_px=float(fwhm_px),
    )
    sp = ForcedSeedAcceptParams(
        centroid_max_px=float(seed_centroid_max_px),
        snr_min=float(seed_snr_min),
    )
    p2_acc = 0
    seed_acc = 0
    for _, row in empty_df.iterrows():
        x0, y0 = float(row["x"]), float(row["y"])
        if dao_pass2_try_at_position(data0, x0, y0, wpx=wpx, h=h, params=p2)["accepted"]:
            p2_acc += 1
        meas = forced_seed_measure_at_position(data0, x0, y0, fwhm_px=fwhm_px, params=sp)
        if forced_seed_accept(meas, params=sp)[0]:
            seed_acc += 1

    p2_rate = p2_acc / n
    seed_rate = seed_acc / n
    inv_det = "PASS" if p2_rate <= float(false_accept_max) else "FAIL"
    inv_seed = "PASS" if seed_rate <= float(false_accept_max) else "FAIL"
    return EmptySkyAudit(
        n_positions=n,
        pass2_accept=int(p2_acc),
        pass2_rate=float(p2_rate),
        seed_accept=int(seed_acc),
        seed_rate=float(seed_rate),
        inv_det=inv_det,
        inv_seed=inv_seed,
    )


def build_calibration_certificate(
    *,
    setup: str,
    wcs_obj: Any,
    data0: np.ndarray,
    dao_x: np.ndarray,
    dao_y: np.ndarray,
    gaia_x: np.ndarray,
    gaia_y: np.ndarray,
    gaia_g: np.ndarray,
    fwhm_px: float,
    pass1_sigma: float,
    pass2_sigma: float,
    seed_snr_min: float,
    target_depth_g: float,
    edge_margin_px: float,
    cfg: Any | None = None,
    coarse_match_px: float | None = None,
    ms_df: pd.DataFrame | None = None,
    census_df: pd.DataFrame | None = None,
    run_validation: bool = True,
    repo_root: Path | str | None = None,
) -> DaoGaiaCalibrationCertificate:
    """Full per-setup calibration: derive tolerances + empty-sky audits."""
    from config import AppConfig

    _cfg = cfg if cfg is not None else AppConfig()
    match_k = float(getattr(_cfg, "masterstar_dao_match_radius_k", DEFAULT_MATCH_K))
    centroid_floor = float(getattr(_cfg, "masterstar_dao_centroid_qa_floor_px", DEFAULT_CENTROID_FLOOR_PX))
    centroid_cap = float(getattr(_cfg, "masterstar_dao_centroid_qa_cap_px", DEFAULT_CENTROID_CAP_PX))
    false_max = float(getattr(_cfg, "masterstar_dao_empty_sky_false_accept_max", DEFAULT_FALSE_ACCEPT_MAX))
    empty_n = int(getattr(_cfg, "masterstar_dao_empty_sky_target_n", DEFAULT_EMPTY_SKY_N))

    h, wpx = int(data0.shape[0]), int(data0.shape[1])
    plate_scale = plate_scale_arcsec_per_px_from_wcs(wcs_obj)

    if coarse_match_px is None:
        coarse_match_px = max(10.0, float(getattr(_cfg, "masterstar_lock_pair_tol_px", 3.0)) * 2.0)

    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent

    identity_diag: DiagnosticPopulationStats | None = None
    seed_diag: DiagnosticPopulationStats | None = None
    identity_dr = np.asarray([], dtype=np.float64)
    seed_dr = np.asarray([], dtype=np.float64)
    seed_pop_label = "empty"
    if data0 is not None and census_df is not None:
        p1x, p1y = _run_pass1_dao_detections(
            data0,
            fwhm_px=float(fwhm_px),
            pass1_sigma=float(pass1_sigma),
            gaia_x=gaia_x,
            gaia_y=gaia_y,
        )
        identity_diag = compute_diagnostic_identity_residuals_px(
            data0,
            gaia_x,
            gaia_y,
            fwhm_px=float(fwhm_px),
            pass1_sigma=float(pass1_sigma),
            pass2_sigma=float(pass2_sigma),
            target_depth_g=float(target_depth_g),
            edge_margin_px=float(edge_margin_px),
            dao_x=p1x,
            dao_y=p1y,
        )
        seed_diag = compute_diagnostic_seed_centroid_offsets_px(
            data0,
            gaia_x,
            gaia_y,
            gaia_g,
            pass1_dao_x=p1x,
            pass1_dao_y=p1y,
            fwhm_px=float(fwhm_px),
            pass2_sigma=float(pass2_sigma),
            seed_snr_min=float(seed_snr_min),
            target_depth_g=float(target_depth_g),
            edge_margin_px=float(edge_margin_px),
        )
        derived = derive_tolerances_from_diagnostic(
            identity_diag,
            seed_diag,
            fwhm_px=float(fwhm_px),
            plate_scale_arcsec_per_px=plate_scale,
            pass1_sigma=float(pass1_sigma),
            pass2_sigma=float(pass2_sigma),
            match_k=match_k,
            centroid_floor_px=centroid_floor,
            centroid_cap_px=centroid_cap,
        )
        seed_pop_label = seed_diag.states or "diagnostic"
    else:
        if ms_df is not None and census_df is not None:
            identity_dr = compute_matched_detection_identity_offsets_px(ms_df, census_df)
            seed_dr, seed_pop_label = compute_seed_acceptance_centroid_offsets_px(
                ms_df,
                census_df,
                data0=data0,
                fwhm_px=float(fwhm_px),
                pass2_sigma=float(pass2_sigma),
                pass2_center_tol_px=float(getattr(_cfg, "masterstar_dao_pass2_center_tol_px", 2.0)),
                seed_snr_min=float(seed_snr_min),
                seed_centroid_cap_px=float(centroid_cap),
            )
        elif dao_x is not None and gaia_x is not None:
            identity_dr = compute_pass1_astrometric_residuals_px(
                dao_x, dao_y, gaia_x, gaia_y, coarse_match_px=float(coarse_match_px)
            )

        derived = derive_tolerances_from_residuals(
            identity_dr,
            seed_dr,
            fwhm_px=float(fwhm_px),
            plate_scale_arcsec_per_px=plate_scale,
            pass1_sigma=float(pass1_sigma),
            pass2_sigma=float(pass2_sigma),
            match_k=match_k,
            centroid_floor_px=centroid_floor,
            centroid_cap_px=centroid_cap,
            seed_states=seed_pop_label,
        )

    empty_df = generate_empty_sky_positions(
        wpx=wpx,
        h=h,
        gaia_x=gaia_x,
        gaia_y=gaia_y,
        gaia_g=gaia_g,
        det_x=dao_x,
        det_y=dao_y,
        fwhm_px=float(fwhm_px),
        target_depth_g=float(target_depth_g),
        match_radius_px=float(derived.match_radius_px),
        edge_margin_px=float(edge_margin_px),
        target_n=empty_n,
    )
    empty_audit = run_empty_sky_audits(
        empty_df,
        data0,
        wpx=wpx,
        h=h,
        fwhm_px=float(fwhm_px),
        pass2_sigma=float(pass2_sigma),
        pass2_center_tol_px=float(derived.pass2_center_tol_px),
        seed_centroid_max_px=float(derived.forced_seed_centroid_max_px),
        seed_snr_min=float(seed_snr_min),
        false_accept_max=false_max,
    )

    fail_reason: str | None = None
    if empty_audit.inv_det != "PASS":
        fail_reason = f"{INV_DET_FALSEFILL}: pass2 false-accept {empty_audit.pass2_rate:.4f} > {false_max}"
    elif empty_audit.inv_seed != "PASS":
        fail_reason = f"{INV_SEED_FALSEFILL}: seed false-accept {empty_audit.seed_rate:.4f} > {false_max}"

    validation_result: ValidationGateResult | None = None
    if run_validation:
        from dao_gaia_stage_validation import run_validation_gate  # noqa: PLC0415

        validation_result = run_validation_gate(
            derived,
            pass1_sigma=float(pass1_sigma),
            pass2_sigma=float(pass2_sigma),
            seed_snr_min=float(seed_snr_min),
            repo_root=repo_root,
        )
        if validation_result.status != "PASS":
            fail_reason = validation_result.fail_reason or "STAGE-01 validation gate FAIL"

    census_accounting: dict[str, Any] | None = None
    detection_completeness: dict[str, Any] | None = None
    if census_df is not None:
        n_chip = int(len(census_df))
        if gaia_x is not None and len(gaia_x):
            n_chip = int(len(gaia_x))
        census_accounting = census_accounting_report(census_df, n_on_chip=n_chip)
        detection_completeness = census_completeness_above_depth(
            census_df, depth_g=float(ITER4_SANDBOX_DETECTION_COMPLETENESS_REF["depth_g"])
        )
        if detection_completeness is not None:
            detection_completeness = dict(detection_completeness)
            detection_completeness["reference"] = dict(ITER4_SANDBOX_DETECTION_COMPLETENESS_REF)
            ref_pct = float(ITER4_SANDBOX_DETECTION_COMPLETENESS_REF["completeness_pct"])
            cur_pct = float(detection_completeness.get("completeness_pct", float("nan")))
            if math.isfinite(cur_pct) and math.isfinite(ref_pct):
                detection_completeness["delta_vs_ref_pp"] = cur_pct - ref_pct
            detection_completeness["diagnostic_only"] = True

    cert_inputs = {
            "measurement_mode": "diagnostic" if identity_diag is not None else "legacy",
            "diagnostic_radius_px": (
                float(identity_diag.diagnostic_radius_px) if identity_diag is not None else None
            ),
            "n_identity_core": int(identity_diag.n) if identity_diag is not None else int(identity_dr.size),
            "n_identity_raw": int(identity_diag.n_raw) if identity_diag is not None else int(identity_dr.size),
            "n_seed_core": int(seed_diag.n) if seed_diag is not None else int(seed_dr.size),
            "n_seed_raw": int(seed_diag.n_raw) if seed_diag is not None else int(seed_dr.size),
            "seed_population": seed_pop_label,
            "coarse_match_px": float(coarse_match_px),
            "match_k": match_k,
            "centroid_floor_px": centroid_floor,
            "centroid_cap_px": centroid_cap,
            "target_depth_g": float(target_depth_g),
            "edge_margin_px": float(edge_margin_px),
            "empty_sky_target_n": empty_n,
            "false_accept_max": false_max,
            "seed_snr_min": float(seed_snr_min),
        }
    if census_accounting is not None:
        cert_inputs["census_accounting"] = census_accounting
    if detection_completeness is not None:
        cert_inputs["detection_completeness"] = detection_completeness

    return DaoGaiaCalibrationCertificate(
        setup=str(setup),
        built_utc=datetime.now(timezone.utc).isoformat(),
        status="PASS" if fail_reason is None else "FAIL",
        fail_reason=fail_reason,
        derived=derived,
        empty_sky=empty_audit,
        validation=validation_result,
        inputs=cert_inputs,
    )


def write_calibration_certificate(
    cert: DaoGaiaCalibrationCertificate,
    platesolve_dir: Path | str,
    *,
    fail_closed: bool = True,
    repo_root: Path | str | None = None,
) -> Path:
    """Write ``dao_gaia_calibration.json``; optionally block catalog acceptance on FAIL.

    XFER-01: identity stamps are mandatory (catalog fp, sandbox SHAs, hand CSV,
    lock-rig plate scale/FWHM, production-scope derived tols). Missing stamp
    source fails loud (DAO-GAIA-IDENTITY). Drift WARN is informational.
    """
    from invariants_runtime import InvariantViolation
    from dao_gaia_stage_validation import (  # noqa: PLC0415
        IDENTITY_STAMP_KEYS,
        build_certificate_identity_stamps,
    )

    stamps = build_certificate_identity_stamps(cert.derived, repo_root=repo_root)
    for key in IDENTITY_STAMP_KEYS:
        if stamps.get(key) in (None, "", {}):
            raise InvariantViolation("DAO-GAIA-IDENTITY", f"identity stamp {key} empty")
    prod = stamps["production_tolerances"]
    stamps["derived_pass2_center_tol_px"] = prod["derived_pass2_center_tol_px"]
    stamps["derived_forced_seed_centroid_max_px"] = prod["derived_forced_seed_centroid_max_px"]
    cert.identity = stamps

    out_dir = Path(platesolve_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / CERT_FILENAME
    path.write_text(json.dumps(cert.to_dict(), indent=2), encoding="utf-8")
    warn = stamps.get("tol_drift_warn") or {}
    if warn.get("status") == "WARN":
        msg = f"DAO-GAIA-XFER-01 {warn.get('message')}"
        try:
            from pipeline import log_event  # noqa: PLC0415

            log_event(msg)
        except Exception:  # noqa: BLE001
            import logging

            logging.getLogger("vyvar").warning(msg)
    if fail_closed and cert.status != "PASS":
        raise InvariantViolation(
            "DAO-GAIA-CALIBRATION",
            cert.fail_reason or "calibration certificate FAIL",
        )
    return path


def load_calibration_certificate(platesolve_dir: Path | str) -> dict[str, Any] | None:
    path = Path(platesolve_dir) / CERT_FILENAME
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def census_completeness_above_depth(
    census_df: pd.DataFrame,
    *,
    depth_g: float = 15.0,
) -> dict[str, Any]:
    """Honest completeness (D-B / M4 diagnostic): (DETECTED_P1+P2+FORCED_SEED) / (G<=depth on-chip minus EDGE)."""
    if census_df is None or census_df.empty:
        return {"ok": False, "reason": "empty census"}
    g = pd.to_numeric(census_df.get("g_mag"), errors="coerce")
    st = census_df.get("source_state", pd.Series([""] * len(census_df))).astype(str).str.strip()
    eligible = census_df.loc[g.le(float(depth_g)) & st.ne("EDGE")].copy()
    detected_states = {"DETECTED_P1", "DETECTED_P2", "FORCED_SEED"}
    num = int(eligible["source_state"].isin(detected_states).sum())
    den = int(len(eligible))
    pct = (100.0 * num / den) if den > 0 else float("nan")
    bins: list[dict[str, Any]] = []
    if den > 0:
        eligible = eligible.assign(_g=g.loc[eligible.index])
        bin_edges = np.arange(8.0, float(depth_g) + 0.51, 0.5)
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:], strict=False):
            mask = eligible["_g"].ge(lo) & eligible["_g"].lt(hi)
            sub = eligible.loc[mask]
            if len(sub) == 0:
                continue
            n_ok = int(sub["source_state"].isin(detected_states).sum())
            bins.append(
                {
                    "g_lo": float(lo),
                    "g_hi": float(hi),
                    "n_eligible": int(len(sub)),
                    "n_detected_seed": n_ok,
                    "completeness_pct": 100.0 * n_ok / len(sub),
                }
            )
    return {
        "ok": True,
        "numerator": num,
        "denominator": den,
        "completeness_pct": pct,
        "depth_g": float(depth_g),
        "denominator_label": f"on-chip Gaia G<={depth_g:.1f} excluding EDGE",
        "numerator_label": "DETECTED_P1 + DETECTED_P2 + FORCED_SEED",
        "bins": bins,
        "state_counts": st.value_counts().to_dict(),
    }


ITER4_SANDBOX_DETECTION_COMPLETENESS_REF = {
    "iter_id": "win_p1_4.5_p2_4.0_i6_i7",
    "depth_g": 14.5,
    "completeness_pct": 95.77897160399079,  # g1_eye_seed_le145 from wire01 iter4 sandbox
    "label": "g1_eye_seed_le145",
}


def census_accounting_report(
    census_df: pd.DataFrame,
    *,
    n_on_chip: int,
) -> dict[str, Any]:
    """L5-new: every on-chip Gaia star named; census rows sum exactly to on-chip count."""
    from masterstar_gaia_accounting import verify_gaia_census_complete

    ok, detail = verify_gaia_census_complete(census_df, int(n_on_chip))
    st = (
        census_df.get("source_state", pd.Series([""] * len(census_df))).astype(str).str.strip()
        if census_df is not None and len(census_df)
        else pd.Series([], dtype=object)
    )
    return {
        "ok": bool(ok),
        "detail": detail,
        "n_census": int(len(census_df)) if census_df is not None else 0,
        "n_on_chip": int(n_on_chip),
        "accounting_pct": 100.0 if ok and n_on_chip > 0 else (0.0 if n_on_chip > 0 else 100.0),
        "state_counts": st.value_counts().to_dict() if len(st) else {},
    }
