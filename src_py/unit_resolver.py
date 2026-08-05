"""Unit-normalisation resolvers for group (b) parameters (Task D1).

When a companion normalised field is None, the legacy pixel/value is returned verbatim
(byte-identity guarantee). When set, convert using resolved plate scale or measured FWHM
at the call site -- never a config constant for plate scale.
"""
from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from utils import plate_solve_fov_deg_diagonal_from_scale

logger = logging.getLogger(__name__)

_LOGGED: set[str] = set()


def _log_once(key: str, message: str) -> None:
    if key in _LOGGED:
        return
    _LOGGED.add(key)
    logger.info(message)


def reset_unit_resolver_logs() -> None:
    """Clear one-shot log guard (tests)."""
    _LOGGED.clear()


def plate_scale_arcsec_per_px_from_wcs(wcs: Any) -> float | None:
    """Median plate scale [arcsec/px] from a celestial WCS, or None."""
    if wcs is None:
        return None
    try:
        from astropy.wcs.utils import proj_plane_pixel_scales

        sc_deg = proj_plane_pixel_scales(wcs)
        ps = float(np.mean(np.abs(np.asarray(sc_deg, dtype=np.float64))) * 3600.0)
        if math.isfinite(ps) and ps > 0:
            return ps
    except Exception:  # noqa: BLE001
        pass
    return None


def plate_scale_arcsec_per_px_from_header(hdr: Any) -> float | None:
    """Best-effort plate scale from a FITS header WCS."""
    if hdr is None:
        return None
    try:
        from astropy.wcs import WCS

        return plate_scale_arcsec_per_px_from_wcs(WCS(hdr))
    except Exception:  # noqa: BLE001
        return None


def _finite_pos(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v) or v <= 0:
        return None
    return v


def resolve_px_from_arcsec(
    value_arcsec: float | None,
    legacy_px: float,
    arcsec_per_px: float | None,
    *,
    param_name: str,
) -> float:
    """Return pixel value. None arcsec -> legacy_px unchanged."""
    if value_arcsec is None:
        _log_once(f"{param_name}:legacy", f"[UNIT-RESOLVE] {param_name}: legacy px={legacy_px}")
        return float(legacy_px)
    arcsec = _finite_pos(value_arcsec)
    scale = _finite_pos(arcsec_per_px)
    if arcsec is None or scale is None:
        _log_once(
            f"{param_name}:legacy-no-scale",
            f"[UNIT-RESOLVE] {param_name}: normalised set but plate scale missing -> legacy px={legacy_px}",
        )
        return float(legacy_px)
    px = arcsec / scale
    _log_once(
        f"{param_name}:arcsec",
        f"[UNIT-RESOLVE] {param_name}: {arcsec:.4f} arcsec / {scale:.4f} arcsec/px -> {px:.4f} px",
    )
    return float(px)


def resolve_int_px_from_arcsec(
    value_arcsec: float | None,
    legacy_px: int,
    arcsec_per_px: float | None,
    *,
    param_name: str,
) -> int:
    px = resolve_px_from_arcsec(value_arcsec, float(legacy_px), arcsec_per_px, param_name=param_name)
    return int(round(px))


def resolve_px_from_fwhm_factor(
    value_fwhm_factor: float | None,
    legacy_px: float,
    fwhm_px: float | None,
    *,
    param_name: str,
) -> float:
    """Return pixel value. None factor -> legacy_px unchanged."""
    if value_fwhm_factor is None:
        _log_once(f"{param_name}:legacy", f"[UNIT-RESOLVE] {param_name}: legacy px={legacy_px}")
        return float(legacy_px)
    factor = _finite_pos(value_fwhm_factor)
    fwhm = _finite_pos(fwhm_px)
    if factor is None or fwhm is None:
        _log_once(
            f"{param_name}:legacy-no-fwhm",
            f"[UNIT-RESOLVE] {param_name}: normalised set but FWHM missing -> legacy px={legacy_px}",
        )
        return float(legacy_px)
    px = factor * fwhm
    _log_once(
        f"{param_name}:fwhm",
        f"[UNIT-RESOLVE] {param_name}: {factor:.4f} x FWHM {fwhm:.4f} px -> {px:.4f} px",
    )
    return float(px)


def resolve_hfr_limit_px(
    cfg: Any,
    *,
    fwhm_px: float | None,
) -> float:
    """qc_max_hfr: legacy is px HFR cap; normalised is ratio x FWHM."""
    legacy = float(getattr(cfg, "qc_max_hfr", 5.0))
    ratio = getattr(cfg, "qc_max_hfr_fwhm_ratio", None)
    if ratio is None:
        _log_once("qc_max_hfr:legacy", f"[UNIT-RESOLVE] qc_max_hfr: legacy px={legacy}")
        return legacy
    r = _finite_pos(ratio)
    fwhm = _finite_pos(fwhm_px)
    if r is None or fwhm is None:
        _log_once(
            "qc_max_hfr:legacy-no-fwhm",
            f"[UNIT-RESOLVE] qc_max_hfr_fwhm_ratio set but FWHM missing -> legacy px={legacy}",
        )
        return legacy
    val = r * fwhm
    _log_once(
        "qc_max_hfr:ratio",
        f"[UNIT-RESOLVE] qc_max_hfr: ratio {r:.4f} x FWHM {fwhm:.4f} px -> {val:.4f} px",
    )
    return float(val)


def resolve_max_dist_fallback_deg(
    cfg: Any,
    *,
    frame_w_px: int,
    frame_h_px: int,
    plate_scale_arcsec_px: float | None,
) -> float:
    """Fallback/additive max_dist_deg when FOV fraction companion is inactive or scale missing."""
    legacy = float(getattr(cfg, "phase01_comparison_max_dist_deg", 1.5))
    frac = getattr(cfg, "phase01_comparison_max_dist_fov_frac", None)
    if frac is None:
        _log_once(
            "phase01_comparison_max_dist_deg:legacy",
            f"[UNIT-RESOLVE] phase01_comparison_max_dist_deg: legacy fallback={legacy:.4f} deg",
        )
        return legacy
    scale = _finite_pos(plate_scale_arcsec_px)
    if scale is None:
        _log_once(
            "phase01_comparison_max_dist_deg:legacy-no-scale",
            f"[UNIT-RESOLVE] phase01_comparison_max_dist_fov_frac set but plate scale missing "
            f"-> legacy fallback={legacy:.4f} deg",
        )
        return legacy
    try:
        diag_deg = plate_solve_fov_deg_diagonal_from_scale(int(frame_w_px), int(frame_h_px), scale)
    except Exception:  # noqa: BLE001
        diag_deg = None
    if diag_deg is None or not math.isfinite(float(diag_deg)) or float(diag_deg) <= 0:
        _log_once(
            "phase01_comparison_max_dist_deg:legacy-bad-diag",
            f"[UNIT-RESOLVE] phase01_comparison_max_dist_fov_frac: invalid FOV diag -> legacy={legacy:.4f} deg",
        )
        return legacy
    f = float(frac)
    result = (float(diag_deg) / 2.0) * f
    _log_once(
        "phase01_comparison_max_dist_deg:fov-frac",
        f"[UNIT-RESOLVE] phase01_comparison_max_dist_fov_frac: diag={float(diag_deg):.4f} deg, "
        f"frac={f:.4f} -> {result:.4f} deg",
    )
    return float(result)


# --- cfg-backed convenience wrappers (legacy field names preserved) ---


def blind_verify_match_tol_px(cfg: Any, *, arcsec_per_px: float | None) -> float:
    return resolve_px_from_arcsec(
        getattr(cfg, "blind_verify_match_tol_arcsec", None),
        float(getattr(cfg, "blind_verify_match_tol_px", 2.5)),
        arcsec_per_px,
        param_name="blind_verify_match_tol_px",
    )


def cog_ladder_step_px(cfg: Any, *, fwhm_px: float | None) -> float:
    return resolve_px_from_fwhm_factor(
        getattr(cfg, "cog_ladder_step_fwhm", None),
        float(getattr(cfg, "cog_ladder_step_px", 0.5)),
        fwhm_px,
        param_name="cog_ladder_step_px",
    )


def hrd_color_bg_box_px(cfg: Any, *, arcsec_per_px: float | None) -> int:
    val = resolve_int_px_from_arcsec(
        getattr(cfg, "hrd_color_bg_box_arcsec", None),
        int(getattr(cfg, "hrd_color_bg_box_px", 96)),
        arcsec_per_px,
        param_name="hrd_color_bg_box_px",
    )
    return int(max(32, min(512, val)))


def masterstar_centre_rms_max_px(cfg: Any, *, arcsec_per_px: float | None) -> float:
    return resolve_px_from_arcsec(
        getattr(cfg, "masterstar_centre_rms_max_arcsec", None),
        float(getattr(cfg, "masterstar_centre_rms_max_px", 1.20)),
        arcsec_per_px,
        param_name="masterstar_centre_rms_max_px",
    )


def masterstar_sibling_rms_max_px(cfg: Any, *, arcsec_per_px: float | None) -> float:
    return resolve_px_from_arcsec(
        getattr(cfg, "masterstar_sibling_rms_max_arcsec", None),
        float(getattr(cfg, "masterstar_sibling_rms_max_px", 2.0)),
        arcsec_per_px,
        param_name="masterstar_sibling_rms_max_px",
    )


def phase01_chip_interior_margin_px(cfg: Any, *, arcsec_per_px: float | None) -> int:
    return resolve_int_px_from_arcsec(
        getattr(cfg, "phase01_chip_interior_margin_arcsec", None),
        int(getattr(cfg, "phase01_chip_interior_margin_px", 50)),
        arcsec_per_px,
        param_name="phase01_chip_interior_margin_px",
    )


def phase01_comparison_isolation_radius_px(
    cfg: Any,
    *,
    arcsec_per_px: float | None,
) -> float:
    return resolve_px_from_arcsec(
        getattr(cfg, "phase01_comparison_isolation_radius_arcsec", None),
        float(getattr(cfg, "phase01_comparison_isolation_radius_px", 25.0)),
        arcsec_per_px,
        param_name="phase01_comparison_isolation_radius_px",
    )


def sips_dao_fwhm_px(cfg: Any, *, fwhm_px: float | None) -> float:
    return resolve_px_from_fwhm_factor(
        getattr(cfg, "sips_dao_fwhm_fwhm_factor", None),
        float(getattr(cfg, "sips_dao_fwhm_px", 2.5)),
        fwhm_px,
        param_name="sips_dao_fwhm_px",
    )
