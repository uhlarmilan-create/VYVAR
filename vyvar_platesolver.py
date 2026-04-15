"""VYVAR plate solving: ICRS hints from FITS/UI and optional in-process solver (local Gaia DR3).

RA/Dec must come from headers (``VY_TARG*``, object keywords, …) or mandatory user input.
The optional ``solve_wcs_with_local_gaia`` path matches DAO detections to a local Gaia DR3 cone/box and
fits a **TAN** (gnomonic / tangent-plane) WCS via ``fit_wcs_from_points``, then optionally **SIP**
(Simple Imaging Polynomial) distortion up to 3rd order so wide-field optics match Gaia DR3 across the
full chip—not only near the centre (pure CD+CRPIX is only linear on the tangent plane).
"""

from __future__ import annotations

import itertools
import logging
import math
import numbers
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import Angle, SkyCoord
from astropy.io import fits
from astropy.wcs import WCS, Sip
from astropy.wcs.utils import fit_wcs_from_points

from config import AppConfig
from database import get_gaia_db_max_g_mag
from infolog import log_event, log_gaia_query
from utils import (
    MIN_GAIA_CONE_RADIUS_DEG,
    catalog_cone_radius_deg_from_optics,
    catalog_cone_radius_from_fov_diameter_deg,
    dao_detection_fwhm_pixels,
    effective_binned_pixel_pitch_um,
    estimate_field_diameter_deg_diagonal,
    fits_header_has_celestial_wcs,
    fits_binning_xy_from_header,
    get_optimal_params,
    maybe_rescale_linear_wcs_cd_to_target_arcsec_per_pixel,
    normalize_telescope_focal_mm_for_plate_scale,
    plate_scale_arcsec_per_pixel,
    strip_celestial_wcs_keys,
)

__all__ = [
    "PointingRequiredError",
    "ResolvedPointing",
    "parse_user_dec_string_to_deg",
    "parse_user_ra_string_to_deg",
    "pointing_hint_from_header",
    "resolve_pointing_for_vyvar",
    "solve_wcs_with_local_gaia",
]

LOGGER = logging.getLogger(__name__)


class PointingRequiredError(ValueError):
    """Raised when neither FITS header nor user text provides a valid RA/Dec pair."""


@dataclass(frozen=True)
class ResolvedPointing:
    """Field center in ICRS degrees and where it came from."""

    ra_icrs_deg: float
    dec_icrs_deg: float
    source: str


_DECIMAL_HDR_RE = re.compile(r"^[-+]?[0-9]+(\.[0-9]*)?([eE][-+]?[0-9]+)?$")


def _fits_header_pick(header: fits.Header, *keys: str) -> Any:
    for key in keys:
        if key in header and header[key] not in (None, ""):
            return header[key]
    return None


def _fits_header_parse_ra_deg(value: Any) -> float | None:
    """RA in degrees (ICRS-style); HMS strings allowed."""
    if value is None:
        return None
    if isinstance(value, numbers.Real):
        x = float(value)
        return x if math.isfinite(x) else None
    s = str(value).strip()
    if not s or s.upper() in {"NAN", "NONE"}:
        return None
    if _DECIMAL_HDR_RE.fullmatch(s):
        try:
            x = float(s)
            return x if math.isfinite(x) else None
        except ValueError:
            return None
    try:
        x = float(Angle(s, unit=u.hourangle).degree)
        return x if math.isfinite(x) else None
    except (ValueError, TypeError):
        pass
    try:
        x = float(Angle(s, unit=u.deg).degree)
        return x if math.isfinite(x) else None
    except (ValueError, TypeError):
        return None


def _fits_header_parse_dec_deg(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, numbers.Real):
        x = float(value)
        return x if math.isfinite(x) else None
    s = str(value).strip()
    if not s or s.upper() in {"NAN", "NONE"}:
        return None
    if _DECIMAL_HDR_RE.fullmatch(s):
        try:
            x = float(s)
            return x if math.isfinite(x) else None
        except ValueError:
            return None
    try:
        x = float(Angle(s, unit=u.deg).degree)
        return x if math.isfinite(x) else None
    except (ValueError, TypeError):
        return None


def pointing_hint_from_header(header: fits.Header) -> tuple[float | None, float | None, str]:
    """Field-center RA/Dec in degrees for hints when WCS is not yet solved.

    Returns ``(ra_deg, dec_deg, source)`` where source is ``VY_TARG``, ``RA_DEC_deg``, ``OBJCTRA_DEC_hms``,
    or ``none``. CRVAL is intentionally not used — the FITS is expected to carry explicit target keywords.
    """
    vy1 = _fits_header_pick(header, "VYTARGRA", "VY_TARGRA")
    vy2 = _fits_header_pick(header, "VYTARGDE", "VY_TARGDEC")
    if vy1 is not None and vy2 is not None:
        try:
            r = float(vy1)
            d = float(vy2)
            if math.isfinite(r) and math.isfinite(d):
                return r, d, "VY_TARG"
        except (TypeError, ValueError):
            pass

    # 2) RA/DEC as floats in degrees (do not interpret HMS here)
    rv = _fits_header_pick(header, "RA")
    dv = _fits_header_pick(header, "DEC")
    if rv is not None and dv is not None:
        try:
            r = float(str(rv).strip())
            d = float(str(dv).strip())
            if math.isfinite(r) and math.isfinite(d):
                return r, d, "RA_DEC_deg"
        except (TypeError, ValueError):
            pass

    # 3) OBJCTRA/OBJCTDEC as HMS/DMS strings
    rv = _fits_header_pick(header, "OBJCTRA")
    dv = _fits_header_pick(header, "OBJCTDEC")
    ra = _fits_header_parse_ra_deg(rv)
    dec = _fits_header_parse_dec_deg(dv)
    if ra is not None and dec is not None:
        return ra, dec, "OBJCTRA_DEC_hms"

    return None, None, "none"


def parse_user_ra_string_to_deg(text: str) -> float:
    """Parse user-entered RA to decimal degrees.

    Accepted: decimal degrees (e.g. ``8.598354``, ``08.598354``); HMS with spaces or colons
    (``08 39 06``, ``8:39:6``); compact ``HHMMSS`` / ``HHMMSS.ss`` (at least 6 digit body).
    """
    raw = (text or "").strip()
    if not raw:
        raise ValueError("RA je prázdne — zadaj hodnotu alebo spusti krok 1 (Analyze).")
    s_norm = " ".join(raw.split())
    sn = raw.replace(" ", "")

    dec_only = re.compile(r"^[-+]?\d+\.\d+([eE][-+]?\d+)?$")
    if dec_only.match(sn):
        x = float(sn)
        if not math.isfinite(x):
            raise ValueError("RA nie je platné číslo")
        return x

    if ":" in raw or (s_norm.count(" ") >= 1 and not dec_only.match(sn)):
        try:
            return float(Angle(s_norm.replace(":", " "), unit=u.hourangle).degree)
        except (ValueError, TypeError) as e:
            raise ValueError(f"RA (HMS) sa nepodarilo rozparsovať: {e}") from e

    if re.fullmatch(r"\d+(\.\d+)?", sn):
        if "." in sn:
            x = float(sn)
            if math.isfinite(x) and -0.001 <= x <= 360.001:
                return x
            raise ValueError(f"RA „{raw}“ — neplatné desatinné stupne.")
        m6 = re.fullmatch(r"(\d{6})(\.\d+)?", sn)
        if m6:
            body = m6.group(1)
            frac = m6.group(2) or ""
            h = int(body[:2])
            mi = int(body[2:4])
            sec = float(body[4:6] + frac)
            if h < 24 and mi < 60 and sec < 60.0001:
                hms = f"{h} {mi} {sec}"
                return float(Angle(hms, unit=u.hourangle).degree)
        if len(sn) <= 3 and sn.isdigit():
            iv = int(sn)
            if 0 <= iv <= 360:
                return float(iv)
        raise ValueError(f"RA „{raw}“ — očakávam HMS (HH MM SS / HHMMSS[.ss]) alebo stupne 0–360.")

    try:
        return float(Angle(s_norm.replace(":", " "), unit=u.hourangle).degree)
    except (ValueError, TypeError) as e:
        raise ValueError(f"RA sa nepodarilo rozparsovať: {e}") from e


def parse_user_dec_string_to_deg(text: str) -> float:
    """Parse user-entered Dec to decimal degrees.

    Accepted: decimal degrees with optional leading ``+`` (``68.598354``, ``+68.598354``, ``-22.5``);
    DMS with spaces/colons (``+68 35 48``, ``-22 30 00``); compact ``[+-]DDMMSS`` / ``DDMMSS.ss``.
    """
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Dec je prázdne — zadaj hodnotu alebo spusti krok 1 (Analyze).")
    s_norm = " ".join(raw.split())

    if ":" in raw or s_norm.count(" ") >= 1:
        try:
            return float(Angle(s_norm.replace(":", " "), unit=u.deg).degree)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Dec (DMS) sa nepodarilo rozparsovať: {e}") from e

    sn = raw.replace(" ", "")
    if sn.startswith("+"):
        sn = sn[1:]
    dec_float = re.compile(r"^-?\d+\.\d+([eE][-+]?\d+)?$")
    if dec_float.match(sn):
        x = float(sn)
        if not math.isfinite(x) or x < -90.001 or x > 90.001:
            raise ValueError("Dec mimo rozsahu −90…+90°")
        return x

    m = re.fullmatch(r"(-?)(\d{2})(\d{2})(\d{2})(\.\d+)?", sn)
    if m:
        sign = -1.0 if m.group(1) == "-" else 1.0
        d = int(m.group(2))
        mi = int(m.group(3))
        se = float(m.group(4) + (m.group(5) or ""))
        if d > 90 or mi >= 60 or se >= 60.0001:
            raise ValueError(f"Dec „{raw}“ — neplatné DMS v kompaktnom tvare DDMMSS.")
        v = sign * (d + mi / 60.0 + se / 3600.0)
        if v < -90 or v > 90:
            raise ValueError("Dec mimo rozsahu −90…+90°")
        return v

    if re.fullmatch(r"-?\d+", sn):
        iv = int(sn)
        if -90 <= iv <= 90:
            return float(iv)
        raise ValueError("Dec ako celé číslo musí byť v rozsahu −90…+90.")

    try:
        return float(Angle(s_norm.replace(":", " "), unit=u.deg).degree)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Dec sa nepodarilo rozparsovať: {e}") from e


def resolve_pointing_for_vyvar(
    header: fits.Header | None,
    *,
    user_ra_text: str = "",
    user_dec_text: str = "",
) -> ResolvedPointing:
    """Return ICRS field center: header first, otherwise **mandatory** user RA/Dec strings.

    Raises:
        PointingRequiredError: if the header has no usable pair and user text does not parse.
    """
    if header is not None:
        ra, dec, src = pointing_hint_from_header(header)
        if ra is not None and dec is not None:
            return ResolvedPointing(float(ra), float(dec), src)
    try:
        ra_u = parse_user_ra_string_to_deg(user_ra_text)
        dec_u = parse_user_dec_string_to_deg(user_dec_text)
    except ValueError as exc:
        raise PointingRequiredError(
            "Pre pokračovanie (plate solve / katalóg) chýba platná RA a Dec: "
            "buď ich má FITS v hlavičke (VY_TARG*, OBJECT, …), alebo ich zadaj v UI."
        ) from exc
    return ResolvedPointing(ra_u, dec_u, "user_ui")


def _triangle_sorted_sides_pixel(xa: float, ya: float, xb: float, yb: float, xc: float, yc: float) -> tuple[float, float, float]:
    d12 = float(np.hypot(xa - xb, ya - yb))
    d23 = float(np.hypot(xb - xc, yb - yc))
    d13 = float(np.hypot(xa - xc, ya - yc))
    s1, s2, s3 = sorted((d12, d23, d13))
    return s1, s2, s3


def _ratios(s1: float, s2: float, s3: float) -> tuple[float, float] | None:
    if s1 < 2.0:
        return None
    return (s2 / s1, s3 / s1)


def _scale_consistent(s_img: tuple[float, float, float], s_arc: tuple[float, float, float], rtol: float) -> bool:
    scales = [s_img[i] / max(s_arc[i], 1e-6) for i in range(3)]
    mx, mn = max(scales), min(scales)
    return mx <= mn * (1.0 + rtol) if mn > 0 else False


def _empirical_median_plate_scale_arcsec_per_px(
    xs: np.ndarray,
    ys: np.ndarray,
    ra_deg: np.ndarray,
    de_deg: np.ndarray,
    *,
    max_stars: int = 42,
    max_pairs: int = 450,
) -> float | None:
    """Median sky_separation[arcsec] / pixel_distance from Gaia–DAO pairs (robust plate scale check)."""
    n = min(int(len(xs)), int(len(ys)), int(len(ra_deg)), int(len(de_deg)), int(max_stars))
    if n < 8:
        return None
    xs = np.asarray(xs[:n], dtype=np.float64)
    ys = np.asarray(ys[:n], dtype=np.float64)
    ra = np.asarray(ra_deg[:n], dtype=np.float64)
    de = np.asarray(de_deg[:n], dtype=np.float64)
    c = SkyCoord(ra=ra * u.deg, dec=de * u.deg, frame="icrs")
    scales: list[float] = []
    npairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            dsky = float(c[i].separation(c[j]).to(u.arcsec).value)
            dpx = float(np.hypot(xs[i] - xs[j], ys[i] - ys[j]))
            if dpx < 8.0 or dsky < 2.0:
                continue
            scales.append(dsky / dpx)
            npairs += 1
            if npairs >= int(max_pairs):
                break
        if npairs >= int(max_pairs):
            break
    if len(scales) < 18:
        return None
    return float(np.median(np.asarray(scales, dtype=np.float64)))


def _triangle_angles_sorted_from_sides(s1: float, s2: float, s3: float) -> tuple[float, float, float] | None:
    """Return internal angles (rad) sorted ascending; rotation/translation invariant."""
    a, b, c = float(s1), float(s2), float(s3)
    if a <= 0 or b <= 0 or c <= 0:
        return None
    # Law of cosines; clamp for numerical robustness.
    def _ang(opposite: float, u: float, v: float) -> float | None:
        den = 2.0 * u * v
        if den <= 0:
            return None
        cosv = (u * u + v * v - opposite * opposite) / den
        cosv = float(max(-1.0, min(1.0, cosv)))
        return float(math.acos(cosv))

    A = _ang(a, b, c)
    B = _ang(b, a, c)
    C = _ang(c, a, b)
    if A is None or B is None or C is None:
        return None
    if not (math.isfinite(A) and math.isfinite(B) and math.isfinite(C)):
        return None
    return tuple(sorted((A, B, C)))


def _linear_tan_predict_pixels(wcs_obj: WCS, ra_deg: np.ndarray, dec_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """TAN+CD pixel prediction ignoring SIP (for fitting distortion on top of a linear plate model)."""
    w = wcs_obj.deepcopy()
    w.sip = None
    return w.all_world2pix(np.asarray(ra_deg, dtype=np.float64), np.asarray(dec_deg, dtype=np.float64), 0)


def _wcs_pixel_rms_linear(wcs_obj: WCS, x_obs: np.ndarray, y_obs: np.ndarray, world: SkyCoord) -> float:
    px, py = _linear_tan_predict_pixels(wcs_obj, world.ra.deg, world.dec.deg)
    return float(np.sqrt(np.mean((px - x_obs) ** 2 + (py - y_obs) ** 2)))


def _wcs_pixel_rms_full(wcs_obj: WCS, x_obs: np.ndarray, y_obs: np.ndarray, world: SkyCoord) -> float:
    px, py = wcs_obj.all_world2pix(world.ra.deg, world.dec.deg, 0)
    return float(np.sqrt(np.mean((px - x_obs) ** 2 + (py - y_obs) ** 2)))


def _filter_catalog_to_fov(df: pd.DataFrame, *, naxis1: int, naxis2: int) -> pd.DataFrame:
    """Wide-field keep-mask for catalog pixels; preserve off-frame stars for corner distortion recovery."""
    if df is None or df.empty or "x" not in df.columns or "y" not in df.columns:
        return df
    x = pd.to_numeric(df.get("x"), errors="coerce")
    y = pd.to_numeric(df.get("y"), errors="coerce")
    mask = (x >= -500.0) & (x < float(naxis1) + 500.0) & (y >= -400.0) & (y < float(naxis2) + 400.0)
    return df.loc[mask].copy()


def _sip_uv_term_indices(max_order: int, min_total_degree: int = 2) -> list[tuple[int, int]]:
    """Triangular SIP-style indices with ``min_total_degree <= i+j <= max_order``."""
    idxs: list[tuple[int, int]] = []
    for i in range(max_order + 1):
        for j in range(max_order + 1 - i):
            s = i + j
            if s < min_total_degree or s > max_order:
                continue
            idxs.append((i, j))
    return idxs


def _sip_fill_ab(coefx: np.ndarray, coefy: np.ndarray, idxs: list[tuple[int, int]], max_order: int) -> tuple[np.ndarray, np.ndarray]:
    n = max_order + 1
    a = np.zeros((n, n), dtype=np.float64)
    b = np.zeros((n, n), dtype=np.float64)
    for k, (i, j) in enumerate(idxs):
        a[i, j] = float(coefx[k])
        b[i, j] = float(coefy[k])
    return a, b


def _fit_sip_on_matches(
    w_lin: WCS,
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    world: SkyCoord,
    *,
    max_order: int = 3,
    ridge: float = 1e-9,
    force_apply: bool = False,
    sip_force_rms_guard_ratio: float | None = 1.15,
) -> tuple[WCS | None, dict[str, Any]]:
    """Attach SIP (forward A/B only) if it reduces RMS vs linear TAN. Returns (new_wcs or None, meta)."""
    meta: dict[str, Any] = {"sip_tried": True, "sip_order": int(max_order)}
    if max_order < 2:
        meta["sip_applied"] = False
        meta["reason"] = "order<2"
        return None, meta

    idxs = _sip_uv_term_indices(max_order, min_total_degree=2)
    npar = len(idxs)
    npts = len(x_obs)
    if npts < npar + 3:
        meta["sip_applied"] = False
        meta["reason"] = "underdetermined"
        return None, meta

    crpix1 = float(w_lin.wcs.crpix[0])
    crpix2 = float(w_lin.wcs.crpix[1])
    xp, yp = _linear_tan_predict_pixels(w_lin, world.ra.deg, world.dec.deg)
    u = xp - crpix1
    v = yp - crpix2
    dx = x_obs - xp
    dy = y_obs - yp

    M = np.column_stack([(u**i) * (v**j) for i, j in idxs])
    if not np.all(np.isfinite(M)) or not np.all(np.isfinite(dx)) or not np.all(np.isfinite(dy)):
        meta["sip_applied"] = False
        meta["reason"] = "non_finite"
        return None, meta

    # One sigma-clip on |Δpix| before fitting (rejects bad pairings).
    r0 = np.hypot(dx, dy)
    med = float(np.median(r0))
    mad = float(np.median(np.abs(r0 - med))) + 1e-9
    clip = med + 5.0 * 1.4826 * mad
    good = r0 <= max(clip, 12.0)
    if int(good.sum()) < npar + 3:
        good = np.ones_like(good, dtype=bool)
    Mg, dxg, dyg = M[good], dx[good], dy[good]

    MtM = Mg.T @ Mg
    dim = MtM.shape[0]
    try:
        if float(np.linalg.cond(MtM)) > 1e13:
            # SIP fallback: high-order SIP often ill-conditioned on wide fields — step down 5→4→3→2.
            if int(max_order) > 2:
                for fo in range(int(max_order) - 1, 1, -1):
                    w2, m2 = _fit_sip_on_matches(
                        w_lin,
                        x_obs,
                        y_obs,
                        world,
                        max_order=int(fo),
                        ridge=ridge,
                        force_apply=force_apply,
                        sip_force_rms_guard_ratio=sip_force_rms_guard_ratio,
                    )
                    if w2 is not None and bool(m2.get("sip_applied", False)):
                        m2 = dict(m2)
                        m2["sip_fallback_from_order"] = int(max_order)
                        m2["sip_fallback_to_order"] = int(fo)
                        m2["sip_fallback_reason"] = "ill_conditioned"
                        return w2, m2
            meta["sip_applied"] = False
            meta["reason"] = "ill_conditioned"
            return None, meta
    except np.linalg.LinAlgError:
        meta["sip_applied"] = False
        meta["reason"] = "cond_failed"
        return None, meta
    reg = ridge * np.eye(dim, dtype=np.float64)
    try:
        coefx = np.linalg.solve(MtM + reg, Mg.T @ dxg)
        coefy = np.linalg.solve(MtM + reg, Mg.T @ dyg)
    except np.linalg.LinAlgError:
        coefx, _, _, _ = np.linalg.lstsq(Mg, dxg, rcond=None)
        coefy, _, _, _ = np.linalg.lstsq(Mg, dyg, rcond=None)

    a, b = _sip_fill_ab(coefx, coefy, idxs, max_order)
    w_sip = w_lin.deepcopy()
    w_sip.wcs.ctype = ["RA---TAN-SIP", "DEC--TAN-SIP"]
    w_sip.sip = Sip(a, b, None, None, (crpix1, crpix2))

    rms_lin = _wcs_pixel_rms_linear(w_lin, x_obs, y_obs, world)
    rms_sip = _wcs_pixel_rms_full(w_sip, x_obs, y_obs, world)
    meta["rms_linear_px"] = rms_lin
    meta["rms_sip_px"] = rms_sip

    if rms_sip < rms_lin * 0.97 or rms_sip < min(rms_lin - 0.08, rms_lin * 0.99):
        meta["sip_applied"] = True
        return w_sip, meta

    if force_apply and w_sip.sip is not None:
        _rms_sip_f = float(meta.get("rms_sip_px") or 999.0)
        _rms_lin_f = float(meta.get("rms_linear_px") or 999.0)
        _guard = float(sip_force_rms_guard_ratio) if sip_force_rms_guard_ratio is not None else None
        if _guard is not None and _rms_lin_f > 0 and _rms_sip_f > _rms_lin_f * _guard:
            meta["sip_applied"] = False
            meta["reason"] = "force_apply_blocked_rms_regression"
            meta["rms_guard_ratio"] = round(_rms_sip_f / _rms_lin_f, 4)
            _hist_msg = (
                f"VYVAR: SIP rejected by RMS guard "
                f"(lin={_rms_lin_f:.3f} sip={_rms_sip_f:.3f} "
                f"ratio={_rms_sip_f / _rms_lin_f:.3f} guard={_guard:.2f})"
            )
            log_event(_hist_msg)
            meta["sip_rms_guard_history"] = _hist_msg
            return None, meta
        meta["sip_applied"] = True
        meta["reason"] = "forced_distortion_model"
        return w_sip, meta

    meta["sip_applied"] = False
    meta["reason"] = "no_rms_gain"
    return None, meta


def _fit_sip_on_matches_masterstar_try_orders(
    w_lin: WCS,
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    world: SkyCoord,
    *,
    sip_max_order: int,
    sip_min_order: int,
    force_apply: bool,
    sip_force_rms_guard_ratio: float | None = 1.15,
) -> tuple[WCS | None, dict[str, Any]]:
    """MASTERSTAR: skúšaj SIP od ``sip_max_order`` nadol po ``sip_min_order`` (typicky 5→4→3), prvý úspešný."""
    metaacc: dict[str, Any] = {"sip_orders_tried_masterstar": []}
    hi = max(2, min(5, int(sip_max_order)))
    lo = max(2, min(5, int(sip_min_order)))
    if lo > hi:
        lo = hi
    last_m: dict[str, Any] = {}
    chosen: int | None = None
    for ord in range(hi, lo - 1, -1):
        metaacc["sip_orders_tried_masterstar"].append(int(ord))
        w_sip, m = _fit_sip_on_matches(
            w_lin,
            x_obs,
            y_obs,
            world,
            max_order=int(ord),
            force_apply=force_apply,
            sip_force_rms_guard_ratio=sip_force_rms_guard_ratio,
        )
        last_m = dict(m) if isinstance(m, dict) else {}
        if w_sip is not None and bool(last_m.get("sip_applied", False)):
            chosen = int(ord)
            out = {**last_m, **metaacc, "sip_chosen_order": chosen}
            if chosen != hi:
                log_event(
                    f"VYVAR MASTERSTAR: SIP stupeň {chosen} (najvyšší úspešný; v config max={hi}, "
                    f"min={lo})."
                )
            return w_sip, out
    return None, {**last_m, **metaacc}


def _fit_sip_for_solver(
    is_masterstar: bool,
    w_lin: WCS,
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    world: SkyCoord,
    *,
    sip_max_order: int,
    sip_min_order: int,
    force_apply: bool,
    sip_force_rms_guard_ratio: float | None = 1.15,
) -> tuple[WCS | None, dict[str, Any]]:
    if is_masterstar and int(sip_max_order) >= 2:
        return _fit_sip_on_matches_masterstar_try_orders(
            w_lin,
            x_obs,
            y_obs,
            world,
            sip_max_order=int(sip_max_order),
            sip_min_order=int(sip_min_order),
            force_apply=force_apply,
            sip_force_rms_guard_ratio=sip_force_rms_guard_ratio,
        )
    return _fit_sip_on_matches(
        w_lin,
        x_obs,
        y_obs,
        world,
        max_order=int(sip_max_order),
        force_apply=force_apply,
        sip_force_rms_guard_ratio=sip_force_rms_guard_ratio,
    )


def _ransac_fit_wcs_tan(
    x: np.ndarray,
    y: np.ndarray,
    world: SkyCoord,
    *,
    rng: np.random.Generator,
    n_iter: int = 90,
    min_sample: int = 8,
    inlier_thresh_px: float = 4.0,
) -> WCS:
    """Robust linear TAN fit: random minimal sets, keep model with most pixel inliers, then refit on inliers."""
    n = int(len(x))
    if n < min_sample:
        return fit_wcs_from_points((x, y), world, projection="TAN")

    ms = min(min_sample, n)
    best_mask = np.ones(n, dtype=bool)
    best_count = -1

    for _ in range(n_iter):
        idx = rng.choice(n, size=ms, replace=False)
        try:
            w_trial = fit_wcs_from_points((x[idx], y[idx]), world[idx], projection="TAN")
        except Exception:  # noqa: BLE001
            continue
        px, py = w_trial.all_world2pix(world.ra.deg, world.dec.deg, 0)
        dist = np.hypot(px - x, py - y)
        mask = dist < float(inlier_thresh_px)
        n_in = int(mask.sum())
        if n_in > best_count:
            best_count = n_in
            best_mask = mask

    if best_count < ms:
        return fit_wcs_from_points((x, y), world, projection="TAN")
    return fit_wcs_from_points((x[best_mask], y[best_mask]), world[best_mask], projection="TAN")


def _greedy_match_pairs_pixel_wcs(
    wcs_for_pred: WCS,
    ra_cat_deg: np.ndarray,
    dec_cat_deg: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    max_px: float,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Match each detection to at most one catalog row by predicted pixel distance (greedy, flux order via sort)."""
    pred_x, pred_y = wcs_for_pred.all_world2pix(
        np.asarray(ra_cat_deg, dtype=np.float64),
        np.asarray(dec_cat_deg, dtype=np.float64),
        0,
    )
    max_px = float(max_px)
    max_px2 = max_px * max_px
    n_img = int(len(xs))
    n_cat = int(len(ra_cat_deg))
    pairs_xy: list[tuple[float, int, int]] = []
    try:
        from scipy.spatial import cKDTree

        cat_xy = np.column_stack(
            [np.asarray(pred_x, dtype=np.float64), np.asarray(pred_y, dtype=np.float64)]
        )
        det_xy = np.column_stack([xs, ys])
        tree_px = cKDTree(cat_xy)
        # Wide fields can have many catalog candidates near a detection; k must be large enough
        # that the true Gaia neighbor survives projection/SIP residuals.
        nk = max(1, min(160, n_cat))
        dist, ind = tree_px.query(det_xy, k=nk, distance_upper_bound=max_px)
        dist = np.atleast_2d(np.asarray(dist, dtype=np.float64))
        ind = np.atleast_2d(np.asarray(ind, dtype=np.int64))
        for k in range(n_img):
            rowd, rowi = dist[k], ind[k]
            for t in range(rowd.shape[0]):
                di = float(rowd[t])
                ji = int(rowi[t])
                if not np.isfinite(di) or ji < 0 or ji >= n_cat:
                    continue
                d2 = di * di
                if d2 <= max_px2:
                    pairs_xy.append((d2, k, ji))
    except Exception:  # noqa: BLE001
        pred_xa = np.asarray(pred_x, dtype=np.float64)
        pred_ya = np.asarray(pred_y, dtype=np.float64)
        for k in range(n_img):
            for j in range(n_cat):
                dx = float(pred_xa[j] - xs[k])
                dy = float(pred_ya[j] - ys[k])
                d2 = dx * dx + dy * dy
                if d2 <= max_px2:
                    pairs_xy.append((d2, k, j))
    pairs_xy.sort(key=lambda t: t[0])
    seen_k: set[int] = set()
    seen_j: set[int] = set()
    pairs_x: list[float] = []
    pairs_y: list[float] = []
    pairs_ra: list[float] = []
    pairs_de: list[float] = []
    for _d2, k, j in pairs_xy:
        if k in seen_k or j in seen_j:
            continue
        seen_k.add(k)
        seen_j.add(j)
        pairs_x.append(float(xs[k]))
        pairs_y.append(float(ys[k]))
        pairs_ra.append(float(ra_cat_deg[j]))
        pairs_de.append(float(dec_cat_deg[j]))
    return pairs_x, pairs_y, pairs_ra, pairs_de


def _greedy_pixel_nn_one_to_one(
    xs: np.ndarray,
    ys: np.ndarray,
    cat_x: np.ndarray,
    cat_y: np.ndarray,
    ra_cat: np.ndarray,
    dec_cat: np.ndarray,
    max_px: float,
    *,
    order_idx: "np.ndarray | None" = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Nearest unused catalog position in pixel space (within ``max_px``).

    By default processes detections in array order; pass ``order_idx`` (e.g. argsort(-flux)) so
    brighter sources claim Gaia neighbours first — critical for robust WCS refits on crowded fields.
    """
    from scipy.spatial import cKDTree

    n_c = int(len(cat_x))
    n_d = int(len(xs))
    if n_c < 5 or n_d < 5:
        return (
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
        )
    xy_c = np.column_stack(
        [np.asarray(cat_x, dtype=np.float64), np.asarray(cat_y, dtype=np.float64)]
    )
    tree = cKDTree(xy_c)
    max_px = float(max_px)
    mx = max(1, min(64, n_c))
    used: set[int] = set()
    px_l: list[float] = []
    py_l: list[float] = []
    ra_l: list[float] = []
    de_l: list[float] = []
    if order_idx is None:
        iter_k = range(n_d)
    else:
        iter_k = np.asarray(order_idx, dtype=np.int64).ravel()
    for k in iter_k:
        dists, idxs = tree.query([float(xs[int(k)]), float(ys[int(k)])], k=mx, distance_upper_bound=max_px)
        dists = np.atleast_1d(np.asarray(dists, dtype=np.float64))
        idxs = np.atleast_1d(np.asarray(idxs, dtype=np.int64))
        for t in np.argsort(dists):
            di = float(dists[t])
            ji = int(idxs[t])
            if ji < 0 or ji >= n_c or not math.isfinite(di):
                continue
            if di > max_px:
                break
            if ji in used:
                continue
            used.add(ji)
            ki = int(k)
            px_l.append(float(xs[ki]))
            py_l.append(float(ys[ki]))
            ra_l.append(float(ra_cat[ji]))
            de_l.append(float(dec_cat[ji]))
            break
    return (
        np.asarray(px_l, dtype=np.float64),
        np.asarray(py_l, dtype=np.float64),
        np.asarray(ra_l, dtype=np.float64),
        np.asarray(de_l, dtype=np.float64),
    )


def _refine_wcs_tan_nn_gaia(
    wcs_in: WCS,
    *,
    xs_det: np.ndarray,
    ys_det: np.ndarray,
    ra_cat_full_deg: np.ndarray,
    dec_cat_full_deg: np.ndarray,
    max_match_px: float,
    min_pairs: int = 12,
    det_order_idx: "np.ndarray | None" = None,
) -> tuple[WCS | None, dict[str, Any]]:
    """Many-star linear TAN refit: greedy NN in pixel space, then ``fit_wcs_from_points``."""
    out: dict[str, Any] = {"n_pairs": 0, "rms_px": None, "mean_dx": None, "mean_dy": None}
    finite_c = np.isfinite(ra_cat_full_deg) & np.isfinite(dec_cat_full_deg)
    ra_c = np.asarray(ra_cat_full_deg[finite_c], dtype=np.float64)
    de_c = np.asarray(dec_cat_full_deg[finite_c], dtype=np.float64)
    if len(ra_c) < min_pairs:
        return None, out
    try:
        px_c, py_c = wcs_in.all_world2pix(ra_c, de_c, 0)
    except Exception:  # noqa: BLE001
        return None, out
    fin_m = np.isfinite(px_c) & np.isfinite(py_c)
    px_c = np.asarray(px_c[fin_m], dtype=np.float64)
    py_c = np.asarray(py_c[fin_m], dtype=np.float64)
    ra_c = ra_c[fin_m]
    de_c = de_c[fin_m]
    if len(ra_c) < min_pairs:
        return None, out
    pxa, pya, rra, dde = _greedy_pixel_nn_one_to_one(
        np.asarray(xs_det, dtype=np.float64),
        np.asarray(ys_det, dtype=np.float64),
        px_c,
        py_c,
        ra_c,
        de_c,
        max_match_px,
        order_idx=det_order_idx,
    )
    out["n_pairs"] = int(len(pxa))
    if len(pxa) < min_pairs:
        return None, out
    world = SkyCoord(ra=rra * u.deg, dec=dde * u.deg, frame="icrs")
    try:
        w_new = fit_wcs_from_points((pxa, pya), world, projection="TAN")
        prx, pry = w_new.all_world2pix(rra, dde, 0)
        dx = np.asarray(prx, dtype=np.float64) - pxa
        dy = np.asarray(pry, dtype=np.float64) - pya
        rms = float(np.sqrt(np.mean(dx * dx + dy * dy)))
        out["rms_px"] = rms
        out["mean_dx"] = float(np.mean(dx))
        out["mean_dy"] = float(np.mean(dy))
        out["pxa"] = pxa
        out["pya"] = pya
        out["world"] = world
        return w_new, out
    except Exception:  # noqa: BLE001
        return None, out


def _log_wcs_orientation_header_hints(wcs_obj: WCS, hdr: fits.Header) -> None:
    """Heuristics for mirrored / flipped acquisition (SIPS etc.)."""
    try:
        xb, yb = fits_binning_xy_from_header(hdr)
        log_event(f"WCS diag: FITS binning ≈ {int(xb)}×{int(yb)} (XBINNING/YBINNING) — over zhodu s efektívnym pixelom pri mierke.")
    except Exception:  # noqa: BLE001
        pass
    try:
        scales = wcs_obj.celestial.proj_plane_pixel_scales()
        sx = abs(float(scales[0].to(u.arcsec).value))
        sy = abs(float(scales[1].to(u.arcsec).value))
        log_event(f"WCS diag: riešená mierka ≈ {sx:.4f} × {sy:.4f} arcsec/pix (celestná projekčná rovina).")
        if sy > 0 and abs(sx / sy - 1.0) > 0.15:
            log_event(
                f"VAROVANIE: Anizotropná mierka ({sx:.3f} × {sy:.3f} arcsec/px) "
                f"— WCS je pravdepodobne nesprávny. Skontroluj plate-solve v MASTERSTAR QA tabe."
            )
    except Exception:  # noqa: BLE001
        pass
    try:
        det = float(np.linalg.det(np.asarray(wcs_obj.wcs.get_pc(), dtype=np.float64)))
        log_event(f"WCS diag: det(PC) ≈ {det:.6f} (záporné ⇒ možné zrkadlenie osí v CD/PC).")
        if det < 0 and abs(det) > 1e-4:
            log_event(
                "WCS orientácia: determinant PC < 0 — možné zrkadlenie osí (SIPS / kamera); "
                "ak stred sedí a okraj nie, skontroluj rotáciu alebo mierku."
            )
        elif det < 0:
            log_event(
                "WCS diag: det(PC) je záporný len numericky (~0) — ignoruj ako signál zrkadlenia; "
                "over radšej pomer mierky sx/sy a zhodu s optikou."
            )
    except Exception:  # noqa: BLE001
        pass
    for key in ("FLIPSTAT", "FLIPPED", "MIRRORED"):
        if key not in hdr:
            continue
        v = hdr[key]
        if v in (None, "", 0, "0", False):
            continue
        log_event(f"WCS orientácia: FITS {key}={v!r} — over zhodu katalógu s obrázkom.")
        break


def _mirror_detections_xy(
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    naxis1: int,
    naxis2: int,
    flip_x: bool,
    flip_y: bool,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(xs, dtype=np.float64, copy=True)
    y = np.asarray(ys, dtype=np.float64, copy=True)
    if flip_x:
        x = (float(naxis1) - 1.0) - x
    if flip_y:
        y = (float(naxis2) - 1.0) - y
    return x, y


def _gaia_triangle_greedy_orientation_probe(
    cat_df_in: pd.DataFrame,
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    naxis1: int,
    naxis2: int,
    w: float,
    h: float,
    simple_mode: bool,
    exp_scale: float | None,
    silent_catalog_crop_log: bool = False,
    max_px_coarse_override: float | None = None,
    expected_scale_rel_tol_override: float | None = None,
) -> dict[str, Any] | None:
    """Triangle match → TAN init → optional catalog crop → coarse greedy match (one orientation)."""
    cat_df = cat_df_in.copy().reset_index(drop=True)
    # Performance: triangle search only uses the brightest ~16–24 Gaia rows, so avoid building a SkyCoord for
    # the full catalog (can be 5k–20k rows). Build it only for the subset we will actually combine.
    ra_all = cat_df["ra_deg"].to_numpy(dtype=np.float64, copy=False)
    de_all = cat_df["dec_deg"].to_numpy(dtype=np.float64, copy=False)
    n_cat = int(len(cat_df))
    n_img = int(len(xs))
    if n_img < 6 or n_cat < 6:
        return None

    n_choose_img = min(n_img, 12 if simple_mode else 20)
    n_choose_cat = min(n_cat, 16 if simple_mode else 24)
    idx_img = list(range(n_choose_img))
    idx_cat = list(range(n_choose_cat))
    # Precompute small Gaia pairwise separations in arcsec without Astropy (hot path in profiling).
    ra_s = np.asarray(ra_all[:n_choose_cat], dtype=np.float64)
    de_s = np.asarray(de_all[:n_choose_cat], dtype=np.float64)
    ra_r = np.deg2rad(ra_s)
    de_r = np.deg2rad(de_s)
    sin_de = np.sin(de_r)
    cos_de = np.cos(de_r)
    # cos(angle) = sin d1 sin d2 + cos d1 cos d2 cos(Δra)
    dra = ra_r[:, None] - ra_r[None, :]
    cosang = (sin_de[:, None] * sin_de[None, :]) + (cos_de[:, None] * cos_de[None, :] * np.cos(dra))
    cosang = np.clip(cosang, -1.0, 1.0)
    sep_arcsec = np.rad2deg(np.arccos(cosang)) * 3600.0

    ratio_tol = 0.040
    scale_rtol = 0.12
    expected_scale_rel_tol = (
        float(expected_scale_rel_tol_override)
        if expected_scale_rel_tol_override is not None and math.isfinite(float(expected_scale_rel_tol_override)) and float(expected_scale_rel_tol_override) > 0
        else 0.08  # striktnejší — zamietne 13.7% odchýlku
    )
    ang_tol_rad = 0.08
    best: tuple[float, tuple[int, int, int], tuple[int, int, int]] | None = None

    for ia, ib, ic in itertools.combinations(idx_img, 3):
        si = _triangle_sorted_sides_pixel(xs[ia], ys[ia], xs[ib], ys[ib], xs[ic], ys[ic])
        ri = _ratios(*si)
        ai = _triangle_angles_sorted_from_sides(*si)
        if ri is None:
            continue
        if ai is None:
            continue
        for ca, cb, cc in itertools.combinations(idx_cat, 3):
            dab = float(sep_arcsec[ca, cb])
            dac = float(sep_arcsec[ca, cc])
            dbc = float(sep_arcsec[cb, cc])
            sc = tuple(sorted((dab, dac, dbc)))
            rc = _ratios(*sc)
            if rc is None:
                continue
            ac = _triangle_angles_sorted_from_sides(*sc)
            if ac is None:
                continue
            dr1 = abs(ri[0] - rc[0])
            dr2 = abs(ri[1] - rc[1])
            if dr1 > ratio_tol or dr2 > ratio_tol:
                continue
            if max(abs(ai[k] - ac[k]) for k in range(3)) > ang_tol_rad:
                continue
            if not _scale_consistent(si, sc, scale_rtol):
                continue
            if exp_scale is not None:
                imp = sum(sc[i] / max(si[i], 1e-12) for i in range(3)) / 3.0
                if not math.isfinite(imp) or imp <= 0:
                    continue
                if abs(imp / exp_scale - 1.0) > expected_scale_rel_tol:
                    continue
            err = dr1 + dr2
            trip_cat = tuple(sorted((ca, cb, cc)))
            trip_img = (ia, ib, ic)
            if best is None or err < best[0]:
                best = (err, trip_img, trip_cat)

    if best is None:
        return None

    _, trip_img, trip_cat = best
    ii = list(trip_img)
    cc = list(trip_cat)

    wcs_init = None
    best_rms = float("inf")
    for perm in itertools.permutations((0, 1, 2), 3):
        px = np.array([xs[ii[0]], xs[ii[1]], xs[ii[2]]], dtype=np.float64)
        py = np.array([ys[ii[0]], ys[ii[1]], ys[ii[2]]], dtype=np.float64)
        ra_l = [float(cat_df.iloc[cc[perm[0]]]["ra_deg"]), float(cat_df.iloc[cc[perm[1]]]["ra_deg"]), float(cat_df.iloc[cc[perm[2]]]["ra_deg"])]
        de_l = [float(cat_df.iloc[cc[perm[0]]]["dec_deg"]), float(cat_df.iloc[cc[perm[1]]]["dec_deg"]), float(cat_df.iloc[cc[perm[2]]]["dec_deg"])]
        world = SkyCoord(ra=np.array(ra_l) * u.deg, dec=np.array(de_l) * u.deg, frame="icrs")
        try:
            w_try = fit_wcs_from_points((px, py), world, projection="TAN")
            pxp, pyp = w_try.all_world2pix(ra_l, de_l, 0)
            rms = float(np.sqrt(np.mean((pxp - px) ** 2 + (pyp - py) ** 2)))
        except Exception:  # noqa: BLE001
            continue
        if rms < best_rms:
            best_rms = rms
            wcs_init = w_try

    if wcs_init is None:
        return None

    try:
        px_all0, py_all0 = wcs_init.all_world2pix(
            cat_df["ra_deg"].to_numpy(dtype=np.float64),
            cat_df["dec_deg"].to_numpy(dtype=np.float64),
            0,
        )
        keep_df = pd.DataFrame({"x": np.asarray(px_all0, dtype=np.float64), "y": np.asarray(py_all0, dtype=np.float64)})
        keep_df = _filter_catalog_to_fov(keep_df, naxis1=int(naxis1), naxis2=int(naxis2))
        keep_cat = np.zeros(len(cat_df), dtype=bool)
        if not keep_df.empty:
            keep_cat[np.asarray(keep_df.index, dtype=np.int64)] = True
        if int(np.count_nonzero(keep_cat)) >= 16:
            cat_df = cat_df.loc[keep_cat].copy().reset_index(drop=True)
            if not silent_catalog_crop_log:
                log_event(
                    f"CATALOG CROP(wide): kept {len(cat_df)} Gaia stars in expanded envelope "
                    f"({int(naxis1)}x{int(naxis2)} px, margin=+500px/-500px X, +400px/-400px Y)."
                )
                log_event(f"CATALOG BOUNDS PX: X[-500,{int(naxis1) + 500}] Y[-400,{int(naxis2) + 400}]")
    except Exception:  # noqa: BLE001
        pass

    ra_all = cat_df["ra_deg"].to_numpy(dtype=np.float64)
    de_all = cat_df["dec_deg"].to_numpy(dtype=np.float64)
    max_px_coarse = max(18.0, min(42.0, 0.014 * float(math.hypot(float(w), float(h)))))
    if max_px_coarse_override is not None:
        try:
            m = float(max_px_coarse_override)
            if math.isfinite(m) and m > 0:
                max_px_coarse = m
        except (TypeError, ValueError):
            pass
    pairs_x, pairs_y, pairs_ra, pairs_de = _greedy_match_pairs_pixel_wcs(
        wcs_init,
        ra_all,
        de_all,
        xs,
        ys,
        max_px=max_px_coarse,
    )
    rate = float(len(pairs_x)) / float(max(1, int(n_img)))
    return {
        "match_rate": rate,
        "wcs_init": wcs_init,
        "cat_df": cat_df,
        "ra_all": ra_all,
        "de_all": de_all,
        "pairs_x": pairs_x,
        "pairs_y": pairs_y,
        "pairs_ra": pairs_ra,
        "pairs_de": pairs_de,
        "max_px_coarse": float(max_px_coarse),
        "best_rms": float(best_rms),
        "n_img": int(n_img),
    }


def _fits_header_strip_sip(hdr: fits.Header) -> fits.Header:
    """Remove SIP polynomial keys so WCS is linear TAN only."""
    h = hdr.copy()
    for k in list(h.keys()):
        ku = str(k).upper()
        if ku.startswith(("A_", "B_", "AP_", "BP_")):
            del h[k]
        if ku in ("A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER"):
            del h[k]
    for i in (1, 2):
        ck = f"CTYPE{i}"
        if ck not in h:
            continue
        v = str(h[ck])
        if "-SIP" in v.upper():
            h[ck] = v.upper().replace("-SIP", "")
    return h


def _wcs_linear_without_sip(wcs_in: WCS) -> WCS | None:
    try:
        from astropy.wcs import FITSFixedWarning
        import warnings

        h = _fits_header_strip_sip(wcs_in.to_header(relax=True))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            w = WCS(h)
        return w if getattr(w, "has_celestial", False) else None
    except Exception:  # noqa: BLE001
        return None


def _equalize_wcs_cd_axes_to_target_arcsec(
    wcs_lin: WCS, target_arcsec_per_px: float
) -> tuple[WCS | None, dict[str, Any]]:
    """Scale CD columns separately so proj-plane scales approach ``target`` (square pixels / optika)."""
    tgt = float(target_arcsec_per_px)
    meta: dict[str, Any] = {}
    if not math.isfinite(tgt) or tgt <= 0:
        return None, meta
    try:
        w = wcs_lin.deepcopy()
        scales = w.celestial.proj_plane_pixel_scales()
        sx = abs(float(scales[0].to(u.arcsec).value))
        sy = abs(float(scales[1].to(u.arcsec).value))
    except Exception:  # noqa: BLE001
        return None, meta
    if min(sx, sy) <= 0:
        return None, meta
    ratio = max(sx, sy) / min(sx, sy)
    meta["plate_scale_sx_arcsec_before"] = float(sx)
    meta["plate_scale_sy_arcsec_before"] = float(sy)
    meta["plate_scale_axis_ratio_before"] = float(ratio)
    facx = tgt / sx
    facy = tgt / sy
    try:
        cd = w.wcs.cd
        if cd is not None:
            arr = np.asarray(cd, dtype=np.float64)
            if arr.shape != (2, 2) or not np.any(arr != 0):
                return None, meta
            arr = arr.copy()
            arr[:, 0] *= facx
            arr[:, 1] *= facy
            w.wcs.cd = arr
        elif w.wcs.cdelt is not None:
            cdlt = np.asarray(w.wcs.cdelt, dtype=np.float64).ravel()
            if cdlt.size < 2:
                return None, meta
            w.wcs.cdelt = np.array([float(cdlt[0]) * facx, float(cdlt[1]) * facy], dtype=np.float64)
        else:
            return None, meta
        scales2 = w.celestial.proj_plane_pixel_scales()
        sx2 = abs(float(scales2[0].to(u.arcsec).value))
        sy2 = abs(float(scales2[1].to(u.arcsec).value))
        meta["plate_scale_sx_arcsec_after"] = float(sx2)
        meta["plate_scale_sy_arcsec_after"] = float(sy2)
        meta["plate_scale_axis_ratio_after"] = float(max(sx2, sy2) / min(sx2, sy2))
        return w, meta
    except Exception:  # noqa: BLE001
        return None, meta


def _maybe_repair_masterstar_anisotropic_plate_scale(
    wcs_in: WCS,
    *,
    target_arcsec_per_px: float,
    pairs_x: np.ndarray,
    pairs_y: np.ndarray,
    pairs_ra: np.ndarray,
    pairs_de: np.ndarray,
    enable_sip: bool,
    sip_max_order: int,
    sip_min_order: int,
    is_masterstar: bool,
    axis_ratio_trigger: float = 1.10,
    sip_force_rms_guard_ratio: float | None = 1.15,
) -> tuple[WCS | None, dict[str, Any]]:
    """If sx/sy plate scales are inconsistent (bad linear fit), re-linearize, equalize CD, refit SIP."""
    meta: dict[str, Any] = {"plate_scale_aniso_repair": False}
    if not is_masterstar:
        return None, meta
    tgt = float(target_arcsec_per_px)
    if not math.isfinite(tgt) or tgt <= 0:
        return None, meta
    n = int(len(pairs_x))
    if n < 12 or n != len(pairs_y) or n != len(pairs_ra) or n != len(pairs_de):
        return None, meta
    try:
        scales = wcs_in.celestial.proj_plane_pixel_scales()
        sx = abs(float(scales[0].to(u.arcsec).value))
        sy = abs(float(scales[1].to(u.arcsec).value))
    except Exception:  # noqa: BLE001
        return None, meta
    if min(sx, sy) <= 0:
        return None, meta
    ratio = max(sx, sy) / min(sx, sy)
    meta["plate_scale_axis_ratio"] = float(ratio)
    if ratio < float(axis_ratio_trigger):
        return None, meta

    pxa = np.asarray(pairs_x, dtype=np.float64)
    pya = np.asarray(pairs_y, dtype=np.float64)
    world_m = SkyCoord(ra=np.asarray(pairs_ra, dtype=np.float64) * u.deg, dec=np.asarray(pairs_de, dtype=np.float64) * u.deg, frame="icrs")
    rms_before = _wcs_pixel_rms_full(wcs_in, pxa, pya, world_m)

    w_lin = _wcs_linear_without_sip(wcs_in)
    if w_lin is None:
        return None, meta

    w_eq, eq_meta = _equalize_wcs_cd_axes_to_target_arcsec(w_lin, tgt)
    if w_eq is None:
        return None, meta
    meta.update(eq_meta)

    w_try = w_eq
    if enable_sip and int(sip_max_order) >= 2:
        w_sip, sip_pass = _fit_sip_for_solver(
            True,
            w_eq,
            pxa,
            pya,
            world_m,
            sip_max_order=int(sip_max_order),
            sip_min_order=int(sip_min_order),
            force_apply=True,
            sip_force_rms_guard_ratio=sip_force_rms_guard_ratio,
        )
        meta.update(sip_pass)
        if w_sip is not None:
            w_try = w_sip

    rms_after = _wcs_pixel_rms_full(w_try, pxa, pya, world_m)
    meta["rms_pairs_before_aniso_repair"] = float(rms_before)
    meta["rms_pairs_after_aniso_repair"] = float(rms_after)

    _force = ratio >= 1.18
    if (not _force) and math.isfinite(rms_before) and rms_after > rms_before * 1.12:
        log_event(
            f"VYVAR MASTERSTAR: anizotropná mierka (pomer osí {ratio:.3f}) — oprava CD zamieta "
            f"(RMS {rms_after:.2f}px > {rms_before:.2f}px × 1.12)."
        )
        return None, meta

    log_event(
        f"VYVAR MASTERSTAR: anizotropná lineárna mierka sx/sy (pomer {ratio:.3f}) vs očakávaná ~{tgt:.3f}″/px — "
        f"CD stĺpce zosúladené na cieľ a SIP znovu prepočítané (RMS na pároch {rms_before:.2f} → {rms_after:.2f} px). "
        f"Modrá projekcia Gaia v QA používa túto WCS; predtým „zlá Gaia“ mohla byť len z deformovaného CD so SIP."
    )
    meta["plate_scale_aniso_repair"] = True
    return w_try, meta


def solve_wcs_with_local_gaia(
    fits_path: Path | str,
    *,
    hint_ra_deg: float | None,
    hint_dec_deg: float | None,
    fov_diameter_deg: float,
    gaia_db_path: Path | str,
    dao_threshold_sigma: float = 3.5,
    max_cat_mag: float = 15.8,
    enable_sip: bool = True,
    sip_max_order: int = 3,
    ransac_refinement: bool = True,
    ransac_min_pairs: int = 14,
    effective_pixel_um: float | None = None,
    focal_length_mm: float | None = None,
    expected_plate_scale_arcsec_per_px: float | None = None,
    max_catalog_rows: int | None = None,
    faintest_mag_limit: float | None = None,
    masterstar_prewrite_rms_max_px: float | None = None,
    masterstar_prewrite_relaxed_rms_max_px: float | None = None,
    masterstar_nn_refine_max_rms_px: float | None = None,
    masterstar_sip_min_order: int | None = None,
    masterstar_sip_force_rms_guard_ratio: float | None = None,
) -> dict[str, Any]:
    """Plate-solve by matching DAO stars to **local Gaia DR3** (SQLite); writes WCS into the FITS primary HDU.

    Očakáva platné hinty RA/Dec (``VY_TARG*``, ``RA``/``DEC``, …) a **platnú mierku** v hlavičke alebo v
    argumentoch: FOCALLEN+PIXSIZE alebo SECPIX/PIXSCALE/SCALE. Parameter
    ``expected_plate_scale_arcsec_per_px`` môže byť nastavený aj pre ``MASTERSTAR.fits`` (prepíše odvodzovanie
    z hlavičky, ak je zadaný).

    ``fit_wcs_from_points(..., projection=\"TAN\")`` len zostaví **lineárny** TAN; SIP sa dopočíta po zhode hviezd.
    """
    from astropy.stats import sigma_clipped_stats
    from photutils.detection import DAOStarFinder

    from database import query_local_gaia

    fp = Path(fits_path).resolve()
    if not fp.is_file():
        return {"solved": False, "reason": f"File not found: {fp}"}
    _is_masterstar = fp.name.strip().upper() == "MASTERSTAR.FITS"

    # SIP order: 2–5; MASTERSTAR skúša nadol po masterstar_sip_min_order (napr. 5→4→3).
    if enable_sip:
        try:
            _smo = int(sip_max_order)
            sip_max_order = max(2, min(5, _smo)) if _smo >= 0 else 3
        except Exception:  # noqa: BLE001
            sip_max_order = 3
    else:
        sip_max_order = 0

    _sip_min_ms = 3
    if _is_masterstar and enable_sip and int(sip_max_order) >= 2:
        if masterstar_sip_min_order is not None:
            try:
                _sip_min_ms = max(2, min(5, int(masterstar_sip_min_order)))
            except (TypeError, ValueError):
                _sip_min_ms = 3
        _sip_min_ms = min(int(sip_max_order), max(2, _sip_min_ms))

    _ms_sip_guard_r: float | None = masterstar_sip_force_rms_guard_ratio
    if _is_masterstar:
        if _ms_sip_guard_r is None:
            _ms_sip_guard_r = AppConfig().masterstar_sip_force_rms_guard_ratio
    else:
        _ms_sip_guard_r = None

    root = Path(gaia_db_path).expanduser().resolve()
    if not root.is_file():
        return {"solved": False, "reason": "VYVAR solver: nastav platnú cestu GAIA_DB_PATH (.db) v Settings."}

    with fits.open(fp, memmap=False) as hdul:
        hdr0 = hdul[0].header.copy()
        data = np.asarray(hdul[0].data, dtype=np.float32)
    if data.ndim != 2:
        return {"solved": False, "reason": "VYVAR solver: očakávam 2D primary image."}

    h, w = int(data.shape[0]), int(data.shape[1])
    naxis1 = int(hdr0.get("NAXIS1", 0) or 0) or w
    naxis2 = int(hdr0.get("NAXIS2", 0) or 0) or h

    # 1) Self-correction for hints (when caller passes None)
    if hint_ra_deg is None or hint_dec_deg is None:
        ra_h, dec_h, _src = pointing_hint_from_header(hdr0)
        hint_ra_deg = ra_h
        hint_dec_deg = dec_h

    # 2) MASTERSTAR-first: trust coordinates embedded in MASTERSTAR.fits over any UI/global settings
    ra0: float | None = None
    de0: float | None = None
    if hint_ra_deg is not None:
        try:
            _r = float(hint_ra_deg)
            if math.isfinite(_r):
                ra0 = _r
        except (TypeError, ValueError):
            pass
    if hint_dec_deg is not None:
        try:
            _d = float(hint_dec_deg)
            if math.isfinite(_d):
                de0 = _d
        except (TypeError, ValueError):
            pass

    _coord_src = "UI/args"
    if _is_masterstar:
        ra_m, de_m, src_m = pointing_hint_from_header(hdr0)
        if ra_m is not None and de_m is not None:
            ra0, de0 = ra_m, de_m
            _coord_src = f"MASTERSTAR header ({src_m})"
    if ra0 is None or de0 is None:
        ra_h2, dec_h2, src_h2 = pointing_hint_from_header(hdr0)
        if ra_h2 is not None and dec_h2 is not None:
            ra0, de0 = ra_h2, dec_h2
            _coord_src = f"header ({src_h2})"

    if ra0 is None or de0 is None or not (math.isfinite(float(ra0)) and math.isfinite(float(de0))):
        return {"solved": False, "reason": "VYVAR solver: neplatný RA/Dec hint (missing/invalid)."}

    log_event(f"INFO: Solver using center hint from {_coord_src}: RA={float(ra0)}, Dec={float(de0)}.")

    # 3) MASTERSTAR: vyžadujeme platný VY_FWHM v hlavičke (žiadne dopĺňanie).
    if _is_masterstar:
        _vyf_raw = hdr0.get("VY_FWHM")
        _vyf_ok = False
        if _vyf_raw is not None:
            try:
                _vyf = float(_vyf_raw)
                _vyf_ok = math.isfinite(_vyf) and _vyf > 0
            except (TypeError, ValueError):
                _vyf_ok = False
        if not _vyf_ok:
            return {
                "solved": False,
                "reason": "VYVAR solver: MASTERSTAR.fits musí mať v hlavičke platný VY_FWHM (px).",
            }

    _ep_um: float | None = None
    if effective_pixel_um is not None:
        try:
            _v = float(effective_pixel_um)
            if math.isfinite(_v) and _v > 0:
                _ep_um = _v
        except (TypeError, ValueError):
            _ep_um = None

    _foc_mm: float | None = None
    if focal_length_mm is not None:
        try:
            _fv = float(focal_length_mm)
            if math.isfinite(_fv) and _fv > 0:
                _foc_mm, _ = normalize_telescope_focal_mm_for_plate_scale(_fv)
        except (TypeError, ValueError):
            _foc_mm = None
    if _foc_mm is None:
        _fh_arg = _fits_header_pick(hdr0, "FOCALLEN", "FOCALLENGTH", "FOCAL", "FOC_LEN")
        if _fh_arg is not None:
            try:
                _fv_h = float(_fh_arg)
                if math.isfinite(_fv_h) and _fv_h > 0:
                    _foc_mm, _ = normalize_telescope_focal_mm_for_plate_scale(_fv_h)
            except (TypeError, ValueError):
                pass
    if _ep_um is None:
        _ph_arg = _fits_header_pick(hdr0, "PIXSIZE", "XPIXSZ", "PIXSZ", "PIXELSIZE", "PIX_SIZE")
        if _ph_arg is not None:
            try:
                _pv_h = float(_ph_arg)
                if math.isfinite(_pv_h) and _pv_h > 0:
                    _ep_um = _pv_h
            except (TypeError, ValueError):
                pass

    # MASTERSTAR-first: if the file embeds pixel pitch / focal length, override any UI/global inputs.
    if _is_masterstar:
        _ph_arg_ms = _fits_header_pick(hdr0, "PIXSIZE", "XPIXSZ", "PIXSZ", "PIXELSIZE", "PIX_SIZE")
        if _ph_arg_ms is not None:
            try:
                _pv_h_ms = float(_ph_arg_ms)
                if math.isfinite(_pv_h_ms) and _pv_h_ms > 0:
                    _ep_um = _pv_h_ms
            except (TypeError, ValueError):
                pass
        _fh_arg_ms = _fits_header_pick(hdr0, "FOCALLEN", "FOCALLENGTH", "FOCAL", "FOC_LEN")
        if _fh_arg_ms is not None:
            try:
                _fv_h_ms = float(_fh_arg_ms)
                if math.isfinite(_fv_h_ms) and _fv_h_ms > 0:
                    _foc_mm, _ = normalize_telescope_focal_mm_for_plate_scale(_fv_h_ms)
            except (TypeError, ValueError):
                pass

    if _ep_um is not None:
        log_event(
            f"VYVAR platesolve: efektívny pixel pre mierku / odvodenia = {_ep_um:.4g} um (súbor {fp.name})"
        )

    _exp_scale: float | None = None
    if expected_plate_scale_arcsec_per_px is not None:
        try:
            _es = float(expected_plate_scale_arcsec_per_px)
            if math.isfinite(_es) and _es > 0:
                _exp_scale = _es
                if _is_masterstar:
                    log_event(
                        f"MASTERSTAR: očakávaná mierka z config/UI = {_es:.4f} arcsec/px "
                        f"(prepíše odvodzovanie z FOCALLEN×PIXSIZE v hlavičke pre filter trojuholníkov)."
                    )
        except (TypeError, ValueError):
            _exp_scale = None
    if _exp_scale is None:
        _hdr_foc_c = _fits_header_pick(hdr0, "FOCALLEN", "FOCALLENGTH", "FOCAL", "FOC_LEN")
        _hdr_pix_c = _fits_header_pick(hdr0, "PIXSIZE", "XPIXSZ", "PIXSZ", "PIXELSIZE", "PIX_SIZE")
        _has_hdr_foc_pix = False
        try:
            if _hdr_foc_c is not None and _hdr_pix_c is not None:
                _hf = float(_hdr_foc_c)
                _hp = float(_hdr_pix_c)
                _has_hdr_foc_pix = math.isfinite(_hf) and _hf > 0 and math.isfinite(_hp) and _hp > 0
        except (TypeError, ValueError):
            _has_hdr_foc_pix = False
        _scale_hdr_kw: float | None = None
        for _sk in ("SECPIX", "PIXSCALE", "SCALE", "SECPIXEL"):
            if _sk not in hdr0:
                continue
            try:
                _sv = float(hdr0[_sk])
                if math.isfinite(_sv) and _sv > 0:
                    _scale_hdr_kw = _sv
                    break
            except (TypeError, ValueError):
                pass
        if _scale_hdr_kw is not None:
            _exp_scale = float(_scale_hdr_kw)
        elif _has_hdr_foc_pix:
            try:
                xb_h, _yb_h = fits_binning_xy_from_header(hdr0)
                eff_um_h = effective_binned_pixel_pitch_um(base_pixel_um_1x1=float(_hdr_pix_c), binning=int(xb_h))
                foc_mm_h, _ = normalize_telescope_focal_mm_for_plate_scale(float(_hdr_foc_c))
                _es_h = plate_scale_arcsec_per_pixel(pixel_pitch_um=float(eff_um_h), focal_length_mm=float(foc_mm_h))
                if _es_h is not None and math.isfinite(float(_es_h)) and float(_es_h) > 0:
                    _exp_scale = float(_es_h)
            except Exception:  # noqa: BLE001
                pass
    if _exp_scale is None:
        return {
            "solved": False,
            "reason": (
                "VYVAR solver: chýba platná mierka — v hlavičke FOCALLEN+PIXSIZE alebo SECPIX/PIXSCALE/SCALE "
                "(arcsec/px); pre MASTERSTAR musí byť mierka v súbore, pre ostatné snímky môže pomôcť "
                "expected plate scale z konfigurácie."
            ),
        }
    log_event(
        f"VYVAR platesolve: očakávaná mierka z pixel×ohnisko ≈ {_exp_scale:.3f} arcsec/px — "
        "filtrujem trojuholníky mimo tejto mierky (proti 10× omylom zhody)."
    )

    _xbin = 1
    try:
        _xbin = max(1, int(float(hdr0.get("XBINNING", hdr0.get("BINNING", 1)) or 1)))
    except Exception:  # noqa: BLE001
        _xbin = 1
    _pix_native_um = (float(_ep_um) / float(_xbin)) if _ep_um is not None and _xbin > 0 else None
    _opt = get_optimal_params(
        focal_length_mm=float(_foc_mm) if _foc_mm is not None else None,
        pixel_size_um=float(_pix_native_um) if _pix_native_um is not None else None,
        binning=int(_xbin),
        naxis1=int(naxis1),
        naxis2=int(naxis2),
        fov_diameter_deg=float(fov_diameter_deg),
    )
    _f_um = float(_ep_um) if _ep_um is not None else 0.0
    _f_mm_u = float(_foc_mm) if _foc_mm is not None else 0.0
    try:
        _scale_arcsec = plate_scale_arcsec_per_pixel(pixel_pitch_um=float(_f_um), focal_length_mm=float(_f_mm_u))
    except Exception:  # noqa: BLE001
        _scale_arcsec = None
    if _scale_arcsec is not None and math.isfinite(float(_scale_arcsec)):
        log_event(
            f"INFO: Starting solve with Scale={float(_scale_arcsec):.3f} arcsec/px "
            f"(F={_foc_mm}mm, Px={_ep_um}um, Bin={_xbin}x)."
        )
    else:
        log_event(
            f"INFO: Starting solve with Scale=nan arcsec/px "
            f"(F={_foc_mm}mm, Px={_ep_um}um, Bin={_xbin}x)."
        )
    cone_r = catalog_cone_radius_deg_from_optics(
        naxis1=naxis1,
        naxis2=naxis2,
        pixel_pitch_um=_f_um,
        focal_length_mm=_f_mm_u,
        margin=0.85,
        fov_diameter_fallback_deg=float(fov_diameter_deg),
    )
    # Ensure cone covers chip diagonal from header optics when present.
    try:
        foc_h = _fits_header_pick(hdr0, "FOCALLEN", "FOCALLENGTH", "FOCAL", "FOC_LEN")
        pix_h = _fits_header_pick(hdr0, "PIXSIZE", "XPIXSZ", "PIXSZ", "PIXELSIZE", "PIX_SIZE")
        foc_mm_h = float(foc_h) if foc_h is not None else float("nan")
        pix_um_h = float(pix_h) if pix_h is not None else float("nan")
        if math.isfinite(foc_mm_h) and foc_mm_h > 0 and math.isfinite(pix_um_h) and pix_um_h > 0:
            foc_mm_h, _ = normalize_telescope_focal_mm_for_plate_scale(float(foc_mm_h))
            xb_h, _yb_h = fits_binning_xy_from_header(hdr0)
            eff_um_h = effective_binned_pixel_pitch_um(base_pixel_um_1x1=float(pix_um_h), binning=int(xb_h))
            sc = plate_scale_arcsec_per_pixel(pixel_pitch_um=float(eff_um_h), focal_length_mm=float(foc_mm_h))
            if sc is not None and math.isfinite(float(sc)) and float(sc) > 0:
                diag_deg = estimate_field_diameter_deg_diagonal(
                    naxis1=int(naxis1),
                    naxis2=int(naxis2),
                    scale_x_arcsec_per_px=float(sc),
                    scale_y_arcsec_per_px=float(sc),
                )
                cone_diag = 0.5 * float(diag_deg) * 1.1
                if math.isfinite(cone_diag) and cone_diag > 0:
                    cone_r = max(float(cone_r), float(cone_diag))
    except Exception:  # noqa: BLE001
        pass
    cone_r = max(float(cone_r), float(MIN_GAIA_CONE_RADIUS_DEG))
    required_corners_radius = catalog_cone_radius_deg_from_optics(
        naxis1=naxis1,
        naxis2=naxis2,
        pixel_pitch_um=_f_um,
        focal_length_mm=_f_mm_u,
        margin=0.85,
        fov_diameter_fallback_deg=float(fov_diameter_deg),
    )
    cone_r = max(float(cone_r), float(required_corners_radius))
    _r_fov = catalog_cone_radius_from_fov_diameter_deg(float(fov_diameter_deg))
    if _r_fov > 0:
        cone_r = max(float(cone_r), _r_fov)
    try:
        cone_r = max(float(cone_r), float(_opt.get("search_radius", 0.0)))
    except Exception:  # noqa: BLE001
        pass
    _foc_log = f"{_foc_mm:g}" if _foc_mm is not None else "?"
    ra_deg = float(ra0)
    dec_deg = float(de0)
    calc_radius = float(cone_r)
    calculated_radius = calc_radius
    log_event(f"📐 FOV Check: Center={ra_deg:.3f},{dec_deg:.3f} | REQUIRED RADIUS for corners: {calc_radius:.3f} deg")
    log_gaia_query(float(ra0), float(de0), calculated_radius)
    log_event(
        f"CATALOG SEARCH: Ra={ra0}, Dec={de0}, Radius={cone_r:.2f} deg (vypočítané pre {_foc_log}mm)"
    )

    center = SkyCoord(ra=ra0 * u.deg, dec=de0 * u.deg, frame="icrs")
    # Gaia rectangular prefilter around the cone (fast idx_ra/idx_dec); then filter by angular radius.
    ra0f, de0f = float(ra0), float(de0)
    ra_min = ra0f - float(cone_r)
    ra_max = ra0f + float(cone_r)
    de_min = de0f - float(cone_r)
    de_max = de0f + float(cone_r)
    if _exp_scale is not None:
        _fov_area_deg2 = (float(naxis1) * float(_exp_scale) / 3600.0) * (
            float(naxis2) * float(_exp_scale) / 3600.0
        )
    else:
        _fov_area_deg2 = 1.0
    _mag_cap: float | None = 15.8
    if _foc_mm is not None and math.isfinite(float(_foc_mm)) and float(_foc_mm) > 0:
        # Dynamický mag cap: cieľ je ~50–200 jasných hviezd vo FOV
        # Odhad hustoty: g<11.5 ~1/deg², g<13 ~10/deg², g<15 ~100/deg² (orientačne pri b~60°)
        # Galaktický pás (b~0°) je hustejší — konzervatívnejšie prahy podľa plochy FOV
        if _fov_area_deg2 > 5.0:
            _mag_cap = 11.5
        elif _fov_area_deg2 > 0.5:
            _mag_cap = 13.0
        elif _fov_area_deg2 > 0.05:
            _mag_cap = 15.0
        else:
            _gmax_db = float(get_gaia_db_max_g_mag(root))
            _mag_cap = float(_gmax_db) if _gmax_db > 0.0 else 16.0
        log_event(
            f"VYVAR platesolve: dynamický mag_cap={float(_mag_cap):.1f} "
            f"(FOV≈{_fov_area_deg2:.3f} deg², focal={float(_foc_mm):.0f}mm)"
        )
    else:
        log_event(
            f"VYVAR platesolve: applying Gaia mag cap g<={float(_mag_cap):.1f} (focal={_foc_mm}mm)."
        )
    if _fov_area_deg2 < 10.0:
        # Pre všetky zostavy: obmedzí kužeľ na FOV+20 % (nie veľký default ~7°+)
        _sc_fov: float | None = float(_exp_scale) if _exp_scale is not None else None
        if (
            _sc_fov is None
            or (not math.isfinite(float(_sc_fov)))
            or float(_sc_fov) <= 0.0
        ) and _scale_arcsec is not None:
            try:
                _sa = float(_scale_arcsec)
                if math.isfinite(_sa) and _sa > 0.0:
                    _sc_fov = _sa
            except (TypeError, ValueError):
                _sc_fov = None
        if (
            _sc_fov is not None
            and math.isfinite(float(_sc_fov))
            and float(_sc_fov) > 0.0
            and int(naxis1) > 0
            and int(naxis2) > 0
        ):
            _fov_x = (float(naxis1) * float(_sc_fov)) / 3600.0  # deg
            _fov_y = (float(naxis2) * float(_sc_fov)) / 3600.0  # deg
            _fov_r = 0.5 * math.hypot(_fov_x, _fov_y) * 1.2  # s 20% okrajom
            cone_r = min(float(cone_r), float(_fov_r))
            log_event(
                f"VYVAR platesolve: FOV={_fov_area_deg2:.3f} deg² < 10 → "
                f"cone_r obmedzený na {cone_r:.3f}° (FOV+20%)"
            )
            ra_min = ra0f - float(cone_r)
            ra_max = ra0f + float(cone_r)
            de_min = de0f - float(cone_r)
            de_max = de0f + float(cone_r)
    # VŽDY obmedz cone_r na FOV+20% ak je _exp_scale k dispozícii; potom SQL box (RA šírka podľa |dec|)
    if _exp_scale is not None and naxis1 is not None and naxis2 is not None:
        _fov_x_deg = float(naxis1) * float(_exp_scale) / 3600.0
        _fov_y_deg = float(naxis2) * float(_exp_scale) / 3600.0
        _fov_r_deg = 0.5 * math.hypot(_fov_x_deg, _fov_y_deg) * 1.2
        if float(cone_r) > _fov_r_deg:
            cone_r = _fov_r_deg
            log_event(f"VYVAR: cone_r clipped to FOV+20% = {cone_r:.3f}°")
        ra_min = ra0f - float(cone_r) / math.cos(math.radians(abs(de0f)))
        ra_max = ra0f + float(cone_r) / math.cos(math.radians(abs(de0f)))
        de_min = de0f - float(cone_r)
        de_max = de0f + float(cone_r)
    rows_g = query_local_gaia(
        root,
        ra_min=ra_min,
        ra_max=ra_max,
        dec_min=de_min,
        dec_max=de_max,
        mag_limit=_mag_cap,
        max_rows=int(max_catalog_rows) if max_catalog_rows is not None else None,
    )
    if not rows_g:
        return {"solved": False, "reason": "VYVAR solver: Gaia query v okolí hintu vrátil 0 hviezd."}
    cat_df = pd.DataFrame(rows_g)
    # Normalize to the solver's expected catalog schema.
    cat_df = cat_df.rename(columns={"source_id": "catalog_id", "ra": "ra_deg", "dec": "dec_deg", "g_mag": "mag"})
    cat_df["catalog"] = "GAIA_DR3"
    # Color index as a stand-in (optional)
    if "bp_rp" not in cat_df.columns:
        cat_df["bp_rp"] = None
    cat_df["b_v"] = pd.to_numeric(cat_df.get("bp_rp"), errors="coerce")
    # Filter by magnitude if available (deeper for MASTERSTAR diagnostic step).
    _ = max_cat_mag
    _ = faintest_mag_limit
    eff_max_cat_mag = float(_mag_cap)
    if "mag" in cat_df.columns:
        m = pd.to_numeric(cat_df["mag"], errors="coerce")
        cat_df = (
            cat_df[(m.notna()) & (m <= float(eff_max_cat_mag))].copy()
            if math.isfinite(float(eff_max_cat_mag))
            else cat_df
        )
    _n_cat_raw = int(len(cat_df))
    log_event(f"SQL GAIA: Nájdených {_n_cat_raw} hviezd pre box okolo hintu (≈ r {float(cone_r):.3f}°).")
    if len(cat_df) < 8:
        return {"solved": False, "reason": f"VYVAR solver: v Gaia výreze málo hviezd ({len(cat_df)})."}

    cat_df_cone_full = cat_df.sort_values("mag", na_position="last").reset_index(drop=True)
    try:
        _scale_arcsec = plate_scale_arcsec_per_pixel(pixel_pitch_um=float(_f_um), focal_length_mm=float(_f_mm_u))
    except Exception:  # noqa: BLE001
        _scale_arcsec = None
    if _scale_arcsec is not None and math.isfinite(float(_scale_arcsec)) and float(_scale_arcsec) > 0:
        _fov_x_deg = (float(naxis1) * float(_scale_arcsec)) / 3600.0
        _fov_y_deg = (float(naxis2) * float(_scale_arcsec)) / 3600.0
        _fov_area = max(1e-6, float(_fov_x_deg) * float(_fov_y_deg))
    else:
        _d = max(0.05, float(fov_diameter_deg))
        _fov_area = max(1e-6, float(_d) * float(_d) * 0.5)
    n_max = int(max(5_000, min(100_000, float(_fov_area) * 5_000.0)))
    if max_catalog_rows is not None:
        try:
            n_max = min(int(n_max), int(max_catalog_rows))
        except (TypeError, ValueError):
            pass
    log_event(f"GAIA dynamic cap: fov_area≈{_fov_area:.4f} deg² -> max_catalog_rows={n_max}")
    _cat_pool = min(int(n_max), len(cat_df_cone_full))
    cat_df = cat_df_cone_full.head(int(n_max)).copy().reset_index(drop=True)
    c_cat = SkyCoord(ra=cat_df["ra_deg"].to_numpy() * u.deg, dec=cat_df["dec_deg"].to_numpy() * u.deg, frame="icrs")
    n_cat = len(c_cat)
    if n_cat < 6:
        return {"solved": False, "reason": "VYVAR solver: po zoradení podľa mag je v kuželi príliš málo hviezd."}

    # Dynamic normalization for per-frame noise adaptation before DAO.
    working_data = np.nan_to_num(data).astype("float32")
    _, med_w, clipped_std = sigma_clipped_stats(working_data, sigma=3.0, maxiters=5)
    clipped_std = float(clipped_std) if np.isfinite(clipped_std) else 0.0
    if clipped_std <= 0:
        clipped_std = 1.0
    working_data = np.clip(working_data - float(med_w), 0.0, None).astype(np.float32, copy=False)

    finite = np.isfinite(working_data)
    if not np.any(finite):
        return {"solved": False, "reason": "VYVAR solver: prázdne dáta."}

    std = float(clipped_std)

    img2 = np.nan_to_num(working_data, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    try:
        _cfg_dao = AppConfig()
        _sips_fb = float(_cfg_dao.sips_dao_fwhm_px)
        if not math.isfinite(_sips_fb) or _sips_fb <= 0:
            _sips_fb = 2.5
        _sig_cfg = float(_cfg_dao.sips_dao_threshold_sigma)
        if not math.isfinite(_sig_cfg) or _sig_cfg <= 0:
            _sig_cfg = 3.5
    except Exception:  # noqa: BLE001
        _sips_fb = 2.5
        _sig_cfg = 3.5
    # Auto-FWHM (DAO centroid kernel).
    _dao_fw = dao_detection_fwhm_pixels(hdr0, configured_fallback=3.0)
    if _dao_fw is None:
        _dao_fw = 3.5
        log_event("VYVAR: VY_FWHM sa nepodarilo získať — DAO FWHM fallback=3.5 px.")
    # Adaptive to per-frame noise via sigma-clipped std; sigma comes from explicit arg or AppConfig.
    try:
        _sig_in = float(dao_threshold_sigma)
    except (TypeError, ValueError):
        _sig_in = float(_sig_cfg)
    sig_req = float(_sig_in if math.isfinite(_sig_in) and _sig_in > 0 else _sig_cfg)
    log_event(f"DEBUG: Threshold set to {sig_req * std:.2f} (using clipped_std={std:.2f})")
    log_event(
        f"Detekcia hviezd: Použité FWHM={float(_dao_fw):.2f}, Sigma={float(sig_req):.2f}"
    )
    sig_try: list[float] = []
    for s in (sig_req, 2.0, 1.2, 1.0):
        ss = max(float(s), 1e-6)
        if not any(abs(ss - t) < 1e-9 for t in sig_try):
            sig_try.append(ss)
    tbl = None
    used_sig = sig_try[0]
    best_tbl = None
    best_n = -1
    best_sig = used_sig
    for s in sig_try:
        finder = DAOStarFinder(
            fwhm=float(_dao_fw),
            threshold=max(float(s) * std, 1e-6),
            brightest=None,
            roundlo=-1.0,
            roundhi=1.0,
        )
        tbl_try = finder(img2)
        n_try = int(len(tbl_try)) if tbl_try is not None else 0
        if n_try > best_n:
            best_n = n_try
            best_tbl = tbl_try
            best_sig = float(s)
        if tbl_try is not None and n_try >= 50:
            tbl = tbl_try
            used_sig = float(s)
            break
    if tbl is None:
        tbl = best_tbl
        used_sig = float(best_sig)
    if used_sig < sig_req - 1e-9:
        log_event(
            f"VYVAR platesolve: DAO fallback sigma {sig_req:.2f} -> {used_sig:.2f} (pre slabé/šumové pole)."
        )
    if tbl is None or len(tbl) < 6:
        return {"solved": False, "reason": "VYVAR solver: málo DAO detekcií (skús nižší prah σ)."}

    tbl = tbl[np.isfinite(tbl["xcentroid"]) & np.isfinite(tbl["ycentroid"]) & np.isfinite(tbl["flux"])]
    flux_arr = np.asarray(tbl["flux"], dtype=np.float64)
    order_full = np.argsort(-flux_arr)
    tbl_sorted = tbl[order_full]
    _simple_mode = not bool(enable_sip)
    top = min(250, len(tbl_sorted))
    tbl = tbl_sorted[:top]
    xs = np.asarray(tbl["xcentroid"], dtype=np.float64)
    ys = np.asarray(tbl["ycentroid"], dtype=np.float64)
    n_img = len(xs)
    if n_img < 6:
        return {"solved": False, "reason": "VYVAR solver: po orezaní málo hviezd na snímke."}

    xs_native = np.asarray(xs, dtype=np.float64, copy=True)
    ys_native = np.asarray(ys, dtype=np.float64, copy=True)

    # Ignore CROTA: triangle matching is rotation-invariant by construction.

    probe0 = _gaia_triangle_greedy_orientation_probe(
        cat_df,
        xs_native,
        ys_native,
        naxis1=int(naxis1),
        naxis2=int(naxis2),
        w=float(w),
        h=float(h),
        simple_mode=bool(_simple_mode),
        exp_scale=_exp_scale,
        silent_catalog_crop_log=False,
        max_px_coarse_override=None,
        expected_scale_rel_tol_override=None,
    )
    if probe0 is None:
        return {
            "solved": False,
            "reason": "VYVAR solver: nenašiel som zhodný trojuholník (skús iný FOV alebo presnejší RA/Dec).",
        }

    ori_candidates: list[tuple[str, bool, bool, dict[str, Any]]] = [("native", False, False, probe0)]
    _probe_rate0 = float(probe0["match_rate"])
    # Slabý native match: vždy otestovať zrkadlá. MASTERSTAR: vždy porovnať (rohy / parity),
    # lebo vysoký globálny match_rate ešte nemusí znamenať správnu orientáciu pixel↔sky.
    _mirror_sweep = bool(_probe_rate0 < 0.10) or bool(_is_masterstar)
    if _mirror_sweep:
        for name, fx, fy in (("mirror_x", True, False), ("mirror_y", False, True), ("mirror_xy", True, True)):
            xs_t, ys_t = _mirror_detections_xy(
                xs_native,
                ys_native,
                naxis1=int(naxis1),
                naxis2=int(naxis2),
                flip_x=fx,
                flip_y=fy,
            )
            pr = _gaia_triangle_greedy_orientation_probe(
                cat_df,
                xs_t,
                ys_t,
                naxis1=int(naxis1),
                naxis2=int(naxis2),
                w=float(w),
                h=float(h),
                simple_mode=bool(_simple_mode),
                exp_scale=_exp_scale,
                silent_catalog_crop_log=True,
                max_px_coarse_override=None,
                expected_scale_rel_tol_override=None,
            )
            if pr is not None:
                ori_candidates.append((name, fx, fy, pr))

    _best_name, best_fx, best_fy, best = max(
        ori_candidates,
        key=lambda t: (float(t[3]["match_rate"]), 1 if (not t[1] and not t[2]) else 0),
    )
    if _is_masterstar and len(ori_candidates) > 1:
        log_event(
            f"VYVAR MASTERSTAR mirror sweep: native={_probe_rate0 * 100.0:.1f}% → "
            f"výber={_best_name} ({float(best['match_rate']) * 100.0:.1f}%)."
        )

    cat_df = best["cat_df"]
    ra_all = best["ra_all"]
    de_all = best["de_all"]
    max_px_coarse = float(best["max_px_coarse"])
    best_rms = float(best["best_rms"])
    wcs_init = best["wcs_init"]
    pairs_x, pairs_y, pairs_ra, pairs_de = best["pairs_x"], best["pairs_y"], best["pairs_ra"], best["pairs_de"]

    if best_fx or best_fy:
        log_event(
            f"VYVAR mirror probe: native match_rate={_probe_rate0 * 100.0:.1f}% → "
            f"winner={_best_name} ({float(best['match_rate']) * 100.0:.1f}%) → "
            "native-pixel WCS refit (CD/PC vs. DAO/SIPS frame)."
        )
        pxa_m = np.asarray(pairs_x, dtype=np.float64)
        pya_m = np.asarray(pairs_y, dtype=np.float64)
        pxa_n, pya_n = _mirror_detections_xy(
            pxa_m,
            pya_m,
            naxis1=int(naxis1),
            naxis2=int(naxis2),
            flip_x=best_fx,
            flip_y=best_fy,
        )
        pra_keep = np.asarray(best["pairs_ra"], dtype=np.float64)
        pde_keep = np.asarray(best["pairs_de"], dtype=np.float64)
        world_m0 = SkyCoord(ra=pra_keep * u.deg, dec=pde_keep * u.deg, frame="icrs")
        try:
            wcs_init = fit_wcs_from_points((pxa_n, pya_n), world_m0, projection="TAN")
            pxv, pyv = wcs_init.all_world2pix(pra_keep, pde_keep, 0)
            best_rms = float(np.sqrt(np.mean((pxv - pxa_n) ** 2 + (pyv - pya_n) ** 2)))
        except Exception:  # noqa: BLE001
            return {"solved": False, "reason": "VYVAR solver: refit WCS po mirror probe zlyhal."}
        xs = xs_native
        ys = ys_native
        pairs_x, pairs_y, pairs_ra, pairs_de = _greedy_match_pairs_pixel_wcs(
            wcs_init,
            ra_all,
            de_all,
            xs,
            ys,
            max_px=max_px_coarse,
        )
        if len(pairs_x) < 5:
            pairs_x, pairs_y, pairs_ra, pairs_de = pxa_n, pya_n, pra_keep, pde_keep
    else:
        xs = xs_native
        ys = ys_native

    pairs_x = np.asarray(pairs_x, dtype=np.float64).tolist()
    pairs_y = np.asarray(pairs_y, dtype=np.float64).tolist()
    pairs_ra = np.asarray(pairs_ra, dtype=np.float64).tolist()
    pairs_de = np.asarray(pairs_de, dtype=np.float64).tolist()
    _n_pairs_post_orientation = int(len(pairs_x))

    # One-shot global offset search (coarse): if initial pairing is very weak, test +/- 1-2 arcmin RA/Dec shifts.
    _initial_match_rate = float(len(pairs_x)) / float(max(1, int(n_img)))
    _coarse_offset_px: float | None = None
    if _initial_match_rate < 0.10:
        try:
            xs_seed = np.asarray(tbl_sorted[: min(50, len(tbl_sorted))]["xcentroid"], dtype=np.float64)
            ys_seed = np.asarray(tbl_sorted[: min(50, len(tbl_sorted))]["ycentroid"], dtype=np.float64)
            if len(xs_seed) >= 8:
                base_n = int(len(pairs_x))
                best_n = base_n
                best_dxdy: tuple[float, float] | None = None
                cos_dec = max(1e-6, abs(math.cos(math.radians(float(de0)))))
                for d_ra_m in (-2.0, -1.0, 1.0, 2.0):
                    for d_de_m in (-2.0, -1.0, 1.0, 2.0):
                        ra_try = ra_all + (float(d_ra_m) / 60.0) / cos_dec
                        de_try = np.clip(de_all + (float(d_de_m) / 60.0), -89.999999, 89.999999)
                        px_t, py_t, pra_t, pde_t = _greedy_match_pairs_pixel_wcs(
                            wcs_init,
                            ra_try,
                            de_try,
                            xs_seed,
                            ys_seed,
                            max_px=max_px_coarse * 1.35,
                        )
                        n_t = int(len(px_t))
                        if n_t <= best_n:
                            continue
                        try:
                            xp_t, yp_t = wcs_init.all_world2pix(
                                np.asarray(pra_t, dtype=np.float64),
                                np.asarray(pde_t, dtype=np.float64),
                                0,
                            )
                            dx_med = float(np.nanmedian(np.asarray(px_t, dtype=np.float64) - np.asarray(xp_t, dtype=np.float64)))
                            dy_med = float(np.nanmedian(np.asarray(py_t, dtype=np.float64) - np.asarray(yp_t, dtype=np.float64)))
                            if math.isfinite(dx_med) and math.isfinite(dy_med):
                                best_n = n_t
                                best_dxdy = (dx_med, dy_med)
                        except Exception:  # noqa: BLE001
                            continue
                if best_dxdy is not None and best_n >= base_n + 4:
                    dx_med, dy_med = best_dxdy
                    off_pix = float(math.hypot(dx_med, dy_med))
                    _coarse_offset_px = float(off_pix)
                    log_event(
                        f"DEBUG: Initial WCS offset detected: {off_pix:.2f} pixels. Applying coarse correction..."
                    )
                    w_tmp = wcs_init.deepcopy()
                    w_tmp.wcs.crpix[0] = float(w_tmp.wcs.crpix[0]) + float(dx_med)
                    w_tmp.wcs.crpix[1] = float(w_tmp.wcs.crpix[1]) + float(dy_med)
                    wcs_init = w_tmp
                    try:
                        _cx0 = 0.5 * float(naxis1)
                        _cy0 = 0.5 * float(naxis2)
                        _ra_c0, _de_c0 = wcs_init.all_pix2world([_cx0], [_cy0], 0)
                        ra0 = float(_ra_c0[0])
                        de0 = float(_de_c0[0])
                    except Exception:  # noqa: BLE001
                        pass
                    pairs_x, pairs_y, pairs_ra, pairs_de = _greedy_match_pairs_pixel_wcs(
                        wcs_init,
                        ra_all,
                        de_all,
                        xs,
                        ys,
                        max_px=max_px_coarse,
                    )
        except Exception:  # noqa: BLE001
            pass

    sip_meta: dict[str, Any] = {
        "max_px_coarse": float(max_px_coarse),
    }
    if best_fx or best_fy:
        sip_meta["det_mirror_orientation"] = str(_best_name)
        sip_meta["n_pairs_after_mirror_native"] = int(_n_pairs_post_orientation)
        # Diagnostic: parity should be negative when mirrored.
        try:
            det_pc = float(np.linalg.det(np.asarray(wcs_init.wcs.get_pc(), dtype=np.float64)))
            sip_meta["wcs_pc_det_after_mirror"] = det_pc
        except Exception:  # noqa: BLE001
            pass
    if _coarse_offset_px is not None:
        sip_meta["initial_wcs_offset_px"] = float(_coarse_offset_px)
    wcs_final = wcs_init
    if len(pairs_x) >= 5:
        pxa = np.asarray(pairs_x, dtype=np.float64)
        pya = np.asarray(pairs_y, dtype=np.float64)
        world_m = SkyCoord(
            ra=np.asarray(pairs_ra, dtype=np.float64) * u.deg,
            dec=np.asarray(pairs_de, dtype=np.float64) * u.deg,
            frame="icrs",
        )
        try:
            if ransac_refinement and len(pairs_x) >= int(ransac_min_pairs):
                rng = np.random.default_rng((hash(str(fp)) & 0xFFFFFFFF) ^ (len(pairs_x) << 12))
                w_lin = _ransac_fit_wcs_tan(pxa, pya, world_m, rng=rng)
            else:
                w_lin = fit_wcs_from_points((pxa, pya), world_m, projection="TAN")
            wcs_final = w_lin
            sip_pass1: dict[str, Any] = {}
            if enable_sip and int(sip_max_order) >= 2:
                w_sip, sip_pass1 = _fit_sip_for_solver(
                    bool(_is_masterstar),
                    w_lin,
                    pxa,
                    pya,
                    world_m,
                    sip_max_order=int(sip_max_order),
                    sip_min_order=int(_sip_min_ms),
                    force_apply=bool(_is_masterstar),
                    sip_force_rms_guard_ratio=_ms_sip_guard_r,
                )
                if w_sip is not None:
                    wcs_final = w_sip
            sip_meta.update(sip_pass1)

            # Refine pass 2: tighter max_px after a better WCS (incl. SIP) → cleaner pairs, refit TAN+SIP.
            max_px_tight = max(6.5, min(13.5, max_px_coarse * 0.40))
            sip_meta["max_px_tight"] = float(max_px_tight)
            prx, pry, prra, prde = _greedy_match_pairs_pixel_wcs(
                wcs_final,
                ra_all,
                de_all,
                xs,
                ys,
                max_px=max_px_tight,
            )
            n_coarse = len(pairs_x)
            sip_meta["n_pairs_coarse"] = int(n_coarse)
            sip_meta["n_pairs_tight"] = int(len(prx))
            min_tight = max(8, int(0.55 * n_coarse))
            if len(prx) >= min_tight and len(prx) >= 5:
                pxa2 = np.asarray(prx, dtype=np.float64)
                pya2 = np.asarray(pry, dtype=np.float64)
                world_m2 = SkyCoord(
                    ra=np.asarray(prra, dtype=np.float64) * u.deg,
                    dec=np.asarray(prde, dtype=np.float64) * u.deg,
                    frame="icrs",
                )
                try:
                    if ransac_refinement and len(prx) >= int(ransac_min_pairs):
                        rng2 = np.random.default_rng((hash(str(fp)) & 0xFFFFFFFF) ^ 0xA5A51234)
                        w_lin2 = _ransac_fit_wcs_tan(pxa2, pya2, world_m2, rng=rng2)
                    else:
                        w_lin2 = fit_wcs_from_points((pxa2, pya2), world_m2, projection="TAN")
                    w_try = w_lin2
                    sip_pass2: dict[str, Any] = {}
                    if enable_sip and int(sip_max_order) >= 2:
                        w_sip2, sip_pass2 = _fit_sip_for_solver(
                            bool(_is_masterstar),
                            w_lin2,
                            pxa2,
                            pya2,
                            world_m2,
                            sip_max_order=int(sip_max_order),
                            sip_min_order=int(_sip_min_ms),
                            force_apply=bool(_is_masterstar),
                            sip_force_rms_guard_ratio=_ms_sip_guard_r,
                        )
                        if w_sip2 is not None:
                            w_try = w_sip2
                    rms_prev = _wcs_pixel_rms_full(wcs_final, pxa2, pya2, world_m2)
                    rms_new = _wcs_pixel_rms_full(w_try, pxa2, pya2, world_m2)
                    if rms_new <= rms_prev * 1.08:
                        wcs_final = w_try
                        sip_meta["refine_tight_applied"] = True
                        sip_meta.update(sip_pass2)
                        pairs_x, pairs_y, pairs_ra, pairs_de = prx, pry, prra, prde
                    else:
                        sip_meta["refine_tight_applied"] = False
                        sip_meta["refine_tight_rejected"] = "rms_regression"
                except Exception:  # noqa: BLE001
                    sip_meta["refine_tight_applied"] = False
                    sip_meta["refine_tight_error"] = True
            else:
                sip_meta["refine_tight_applied"] = False
                sip_meta["refine_tight_skipped"] = "too_few_pairs"
        except Exception:  # noqa: BLE001
            wcs_final = wcs_init
            sip_meta["refine_error"] = True

    if wcs_final.sip is None:
        _cd_rescaled_any = False
        if len(pairs_x) >= 14:
            try:
                _emp_s = _empirical_median_plate_scale_arcsec_per_px(
                    np.asarray(pairs_x, dtype=np.float64),
                    np.asarray(pairs_y, dtype=np.float64),
                    np.asarray(pairs_ra, dtype=np.float64),
                    np.asarray(pairs_de, dtype=np.float64),
                )
                if _emp_s is not None and math.isfinite(_emp_s) and float(_emp_s) > 0:
                    # Ak sa empiria líši od optickej mierky z hlavičky >10 %, never jej (zlé páry / konfúzia
                    # dávali napr. ~12"/px namiesto ~9.55"/px a rozbíjali FITS WCS).
                    if _exp_scale is not None:
                        try:
                            _rel_hdr = abs(float(_emp_s) / float(_exp_scale) - 1.0)
                        except (TypeError, ValueError, ZeroDivisionError):
                            _rel_hdr = 1.0
                        if _rel_hdr > 0.10:
                            log_event(
                                f"VYVAR: empirická mierka z párov {float(_emp_s):.3f} arcsec/px vs hlavička "
                                f"{float(_exp_scale):.3f} (Δ {_rel_hdr*100:.1f}%) — CD škálovanie z párov preskočené."
                            )
                            _emp_s = None
                if _emp_s is not None and math.isfinite(_emp_s) and float(_emp_s) > 0:
                    w_e, _ok_e = maybe_rescale_linear_wcs_cd_to_target_arcsec_per_pixel(
                        wcs_final,
                        float(_emp_s),
                        trigger_relative_mismatch=0.007,
                    )
                    if _ok_e:
                        wcs_final = w_e
                        _cd_rescaled_any = True
                        sip_meta["cd_rescaled_to_empirical_scale"] = True
                        sip_meta["plate_scale_empirical_arcsec_per_px"] = float(_emp_s)
                        log_event(
                            f"VYVAR WCS: CD/PC škálované podľa empirie z párov hviezd ≈ {float(_emp_s):.3f} arcsec/px"
                        )
            except Exception:  # noqa: BLE001
                pass
        if not _cd_rescaled_any and _exp_scale is not None:
            w_adj, _cd_rescaled = maybe_rescale_linear_wcs_cd_to_target_arcsec_per_pixel(
                wcs_final, float(_exp_scale)
            )
            if _cd_rescaled:
                wcs_final = w_adj
                sip_meta["cd_rescaled_to_expected_scale"] = True
                log_event(
                    f"VYVAR WCS: CD/PC škálované podľa optickej mierky {float(_exp_scale):.3f} arcsec/px"
                )

    # Critical: if match rate is low but solution is not rejected, force SIP4 refit to fix edge residuals.
    # This intentionally ignores any legacy CDELT/CROTA in the incoming header: we refit distortion from Gaia pairs.
    try:
        _n_det_total0 = max(1, int(n_img))
        _mr0 = float(int(len(pairs_x))) / float(_n_det_total0)
    except Exception:  # noqa: BLE001
        _mr0 = float("nan")
    if (
        bool(enable_sip)
        and wcs_final.sip is None
        and math.isfinite(_mr0)
        and _mr0 < 0.10
        and int(len(pairs_x)) >= max(12, int(ransac_min_pairs))
    ):
        try:
            pxa_f = np.asarray(pairs_x, dtype=np.float64)
            pya_f = np.asarray(pairs_y, dtype=np.float64)
            world_f = SkyCoord(
                ra=np.asarray(pairs_ra, dtype=np.float64) * u.deg,
                dec=np.asarray(pairs_de, dtype=np.float64) * u.deg,
                frame="icrs",
            )
            if _is_masterstar:
                _fo_hi = max(2, min(5, max(4, int(sip_max_order))))
                w_sip_force, sip_force = _fit_sip_for_solver(
                    True,
                    wcs_final,
                    pxa_f,
                    pya_f,
                    world_f,
                    sip_max_order=int(_fo_hi),
                    sip_min_order=int(_sip_min_ms),
                    force_apply=True,
                    sip_force_rms_guard_ratio=_ms_sip_guard_r,
                )
            else:
                w_sip_force, sip_force = _fit_sip_on_matches(
                    wcs_final,
                    pxa_f,
                    pya_f,
                    world_f,
                    max_order=4,
                    force_apply=False,
                )
            if w_sip_force is not None and bool(sip_force.get("sip_applied", False)):
                wcs_final = w_sip_force
                sip_meta.update(sip_force)
                sip_meta["sip_force_low_match_rate"] = True
                log_event(
                    "VYVAR: Low match_rate → forcing SIP refit (TAN-SIP) for edge correction."
                )
        except Exception:  # noqa: BLE001
            pass

    # So SIP: CD/PC škálovanie podľa optiky sa v bloku vyššie preskočí (sip is not None). Zlý lineárny fit potom
    # môže dať nefyzikálnu anizotropiu sx≠sy (napr. 7.7×12.4″/px) — QA potom „posúva“ modrú Gaia vs. raster.
    if _is_masterstar and _exp_scale is not None and len(pairs_x) >= max(12, int(ransac_min_pairs)):
        try:
            w_rep, rep_meta = _maybe_repair_masterstar_anisotropic_plate_scale(
                wcs_final,
                target_arcsec_per_px=float(_exp_scale),
                pairs_x=np.asarray(pairs_x, dtype=np.float64),
                pairs_y=np.asarray(pairs_y, dtype=np.float64),
                pairs_ra=np.asarray(pairs_ra, dtype=np.float64),
                pairs_de=np.asarray(pairs_de, dtype=np.float64),
                enable_sip=bool(enable_sip),
                sip_max_order=int(sip_max_order),
                sip_min_order=int(_sip_min_ms),
                is_masterstar=True,
                sip_force_rms_guard_ratio=_ms_sip_guard_r,
            )
            if w_rep is not None and bool(rep_meta.get("plate_scale_aniso_repair")):
                wcs_final = w_rep
                sip_meta.update(rep_meta)
        except Exception:  # noqa: BLE001
            pass

    # Fast-path / triangle solve may use a bright-only Gaia slice (e.g. g<=11.5). For association QA we need a
    # deeper cone catalog, otherwise faint DAO peaks look "unmatched" and match% is misleadingly low.
    cat_df_assoc = cat_df_cone_full
    try:
        _mlim_assoc = float(get_gaia_db_max_g_mag(root))
        if _mlim_assoc <= 0.0:
            _mlim_assoc = 11.5
        if faintest_mag_limit is not None:
            try:
                _mlim_assoc = min(float(_mlim_assoc), float(faintest_mag_limit))
            except (TypeError, ValueError):
                pass
        _mr_cap = int(max_catalog_rows) if max_catalog_rows is not None else 20000
        rows_assoc = query_local_gaia(
            root,
            ra_min=float(ra_min),
            ra_max=float(ra_max),
            dec_min=float(de_min),
            dec_max=float(de_max),
            mag_limit=float(_mlim_assoc),
            max_rows=max(5000, min(50000, _mr_cap)),
        )
        if rows_assoc:
            _dfa = pd.DataFrame(rows_assoc)
            _dfa = _dfa.rename(
                columns={"source_id": "catalog_id", "ra": "ra_deg", "dec": "dec_deg", "g_mag": "mag"}
            )
            _dfa["catalog"] = "GAIA_DR3"
            if "bp_rp" not in _dfa.columns:
                _dfa["bp_rp"] = None
            _dfa["b_v"] = pd.to_numeric(_dfa.get("bp_rp"), errors="coerce")
            _dfa = _dfa.sort_values("mag", na_position="last").reset_index(drop=True)
            cat_df_assoc = _dfa
            log_event(
                f"VYVAR: association Gaia slice for QA: {len(cat_df_assoc)} rows (g<={float(_mlim_assoc):.1f})"
            )
    except Exception:  # noqa: BLE001
        pass

    # QA rematch on the same detections used for solving (``n_img`` brightest), not ``len(tbl_sorted)`` (can be 5k+).
    # Use ``cat_df_assoc`` (deep cone), not the triangle-probe crop in ``ra_all``/``de_all``.
    try:
        _ra_cat = cat_df_assoc["ra_deg"].to_numpy(dtype=np.float64)
        _de_cat = cat_df_assoc["dec_deg"].to_numpy(dtype=np.float64)
        _mtq = float(sip_meta.get("max_px_tight", 0.0) or 0.0)
        if not (math.isfinite(_mtq) and _mtq > 0):
            _mtq = float(max_px_coarse)
        _qa_px = max(15.0, min(48.0, float(_mtq) * 1.22))
        qx, qy, qra, qde = _greedy_match_pairs_pixel_wcs(
            wcs_final,
            _ra_cat,
            _de_cat,
            np.asarray(xs, dtype=np.float64),
            np.asarray(ys, dtype=np.float64),
            max_px=float(_qa_px),
        )
        pairs_x, pairs_y, pairs_ra, pairs_de = list(qx), list(qy), list(qra), list(qde)
        sip_meta["qa_rematch_max_px"] = float(_qa_px)
    except Exception:  # noqa: BLE001
        pass

    # Pre-write validation: reject weak/shifted solutions and retry with simpler TAN model.
    _n_det_total = max(1, int(n_img))
    _matched_n = int(len(pairs_x))
    _match_rate = float(_matched_n) / float(_n_det_total)
    sip_meta["match_rate_n_used"] = int(n_img)
    sip_meta["match_rate_n_matched"] = int(_matched_n)
    _rms_px = None
    for _k in ("wcs_refine_rms_px", "rms_sip_px", "rms_linear_px"):
        _v = sip_meta.get(_k)
        try:
            _vf = float(_v)
            if math.isfinite(_vf):
                _rms_px = _vf
                break
        except (TypeError, ValueError):
            continue
    if _rms_px is None:
        try:
            _rms_px = float(best_rms)
        except Exception:  # noqa: BLE001
            _rms_px = float("inf")

    try:
        _cx = 0.5 * float(naxis1)
        _cy = 0.5 * float(naxis2)
        _ra_c, _de_c = wcs_final.all_pix2world([_cx], [_cy], 0)
        _hint_sc = SkyCoord(ra=float(ra0) * u.deg, dec=float(de0) * u.deg, frame="icrs")
        _sol_sc = SkyCoord(ra=float(_ra_c[0]) * u.deg, dec=float(_de_c[0]) * u.deg, frame="icrs")
        _hint_sep_deg = float(_hint_sc.separation(_sol_sc).deg)
    except Exception:  # noqa: BLE001
        _hint_sep_deg = float("nan")
    log_event(
        f"VYVAR platesolve QA: match_rate={_match_rate * 100.0:.1f}% "
        f"rms={float(_rms_px):.2f}px hint_vs_solved={_hint_sep_deg:.3f}deg"
    )

    _sip_reason = str(sip_meta.get("reason", "") or "").strip().lower()
    if (not bool(sip_meta.get("sip_applied", False))) and _sip_reason == "ill_conditioned":
        log_event(
            "VYVAR platesolve: SIP zlyhal (ill_conditioned) — opakujem s jednoduchším lineárnym WCS (očakávané pri niektorých stackoch)."
        )

    # MASTERSTAR stack: lineárny TAN + široké pole často dáva RMS > 5 px na prvom párovaní, ale match_rate je dobrý;
    # astrometry_optimizer a širší katalógový match to potom stiahnu. Bežné snímky: prísnych 5 px.
    _rms_max_accept = (
        float(masterstar_prewrite_rms_max_px)
        if (_is_masterstar and masterstar_prewrite_rms_max_px is not None)
        else (14.0 if _is_masterstar else 5.0)
    )
    _rms_relaxed_cap = (
        float(masterstar_prewrite_relaxed_rms_max_px)
        if (_is_masterstar and masterstar_prewrite_relaxed_rms_max_px is not None)
        else 22.0
    )
    sip_meta["prewrite_rms_threshold_px"] = float(_rms_max_accept)
    sip_meta["prewrite_rms_relaxed_cap_px"] = float(_rms_relaxed_cap)
    _rms_bad = float(_rms_px) > float(_rms_max_accept)
    if (
        _is_masterstar
        and _rms_bad
        and float(_match_rate) >= 0.45
        and math.isfinite(float(_rms_px))
        and float(_rms_px) <= float(_rms_relaxed_cap)
    ):
        _rms_bad = False
        log_event(
            f"VYVAR MASTERSTAR: RMS {float(_rms_px):.2f}px > {_rms_max_accept:.0f}px, "
            f"ale match_rate={_match_rate * 100.0:.1f}% — akceptujem do {_rms_relaxed_cap:.0f} px pred ďalšími krokmi."
        )
        sip_meta["prewrite_rms_relaxed_for_masterstar"] = True
    _invalid = (_match_rate < 0.02) or (not math.isfinite(float(_rms_px))) or _rms_bad
    if _invalid:
        return {
            "solved": False,
            "match_rate": float(_match_rate),
            "rms_px": float(_rms_px),
            "reason": (
                f"VYVAR solver: invalid solution (match_rate={_match_rate * 100.0:.1f}%, "
                f"rms={float(_rms_px):.2f}px, hint_sep={_hint_sep_deg:.3f}deg)."
            ),
        }

    _log_wcs_orientation_header_hints(wcs_final, hdr0)

    _nref = min(200, max(100, len(tbl_sorted)))
    _max_mpx = max(22.0, min(95.0, 0.026 * float(math.hypot(float(w), float(h)))))
    xs_ref = np.asarray(tbl_sorted[:_nref]["xcentroid"], dtype=np.float64)
    ys_ref = np.asarray(tbl_sorted[:_nref]["ycentroid"], dtype=np.float64)
    _nn_cat_n = min(int(len(cat_df_cone_full)), 8000)
    ra_full = cat_df_cone_full["ra_deg"].to_numpy(dtype=np.float64)[:_nn_cat_n]
    de_full = cat_df_cone_full["dec_deg"].to_numpy(dtype=np.float64)[:_nn_cat_n]
    w_nn: WCS | None = None
    meta_nn: dict[str, Any] = {}
    w_nn, meta_nn = _refine_wcs_tan_nn_gaia(
        wcs_final,
        xs_det=xs_ref,
        ys_det=ys_ref,
        ra_cat_full_deg=ra_full,
        dec_cat_full_deg=de_full,
        max_match_px=_max_mpx,
        min_pairs=12,
    )
    if w_nn is not None and meta_nn.get("rms_px") is not None:
        rms_nn = float(meta_nn["rms_px"])
        log_event(f"WCS Refined: Mean residual error = {rms_nn:.2f} pixels")
        # NN refit can latch onto wrong Gaia neighbours when max_match_px is large; a high RMS means
        # the new TAN is worse than triangle+SIP — applying it destroys downstream catalog matching (~0% Gaia).
        _rms_nn_max = (
            float(masterstar_nn_refine_max_rms_px)
            if (_is_masterstar and masterstar_nn_refine_max_rms_px is not None)
            else 7.5
        )
        sip_meta["wcs_nn_max_rms_px_threshold"] = float(_rms_nn_max)
        accept_nn = math.isfinite(rms_nn) and float(rms_nn) <= float(_rms_nn_max)
        mdx = meta_nn.get("mean_dx")
        mdy = meta_nn.get("mean_dy")
        if mdx is not None and mdy is not None and (abs(float(mdx)) > 0.35 or abs(float(mdy)) > 0.35):
            log_event(
                f"WCS refine hint: stredný posun dx={float(mdx):.2f}, dy={float(mdy):.2f} px "
                "(jednotný offset); rozdielny posun po poli → rotácia / mierka / flip."
            )
        if not accept_nn:
            log_event(
                f"VYVAR: NN WCS refine zamietnutý (rms={rms_nn:.2f}px > {_rms_nn_max:.1f}px) — "
                "ponechávam WCS pred NN (inak často kolaps zhody s Gaia v MASTERSTAR kroku)."
            )
            sip_meta["wcs_nn_refined"] = False
            sip_meta["wcs_nn_rejected"] = True
            sip_meta["wcs_nn_rejected_rms_px"] = float(rms_nn)
            sip_meta["wcs_nn_rejected_max_rms_px"] = float(_rms_nn_max)
        else:
            wcs_final = w_nn
            sip_meta["wcs_nn_refined"] = True
            sip_meta["wcs_refine_rms_px"] = rms_nn
            sip_meta["wcs_refine_n_pairs"] = int(meta_nn.get("n_pairs", 0))
            pxa_r = meta_nn.get("pxa")
            pya_r = meta_nn.get("pya")
            world_r = meta_nn.get("world")
            if (
                enable_sip
                and int(sip_max_order) >= 2
                and pxa_r is not None
                and pya_r is not None
                and world_r is not None
                and len(pxa_r) >= int(ransac_min_pairs)
            ):
                try:
                    w_sip_r, sip_r3 = _fit_sip_for_solver(
                        bool(_is_masterstar),
                        wcs_final,
                        np.asarray(pxa_r, dtype=np.float64),
                        np.asarray(pya_r, dtype=np.float64),
                        world_r,
                        sip_max_order=int(sip_max_order),
                        sip_min_order=int(_sip_min_ms),
                        force_apply=bool(_is_masterstar),
                        sip_force_rms_guard_ratio=_ms_sip_guard_r,
                    )
                    if w_sip_r is not None:
                        wcs_final = w_sip_r
                        sip_meta.update(sip_r3)
                        sip_meta["sip_after_nn_refine"] = True
                except Exception:  # noqa: BLE001
                    pass
            pairs_x = np.asarray(pxa_r, dtype=np.float64).tolist()
            pairs_y = np.asarray(pya_r, dtype=np.float64).tolist()
            pairs_ra = np.asarray(world_r.ra.deg, dtype=np.float64).tolist()
            pairs_de = np.asarray(world_r.dec.deg, dtype=np.float64).tolist()

    # Po NN/SIP ešte raz zrátaj páry na rovnakom súbore detekcií (``n_img``) oproti asociačnému Gaia výrezu.
    try:
        _ra_cat2 = cat_df_assoc["ra_deg"].to_numpy(dtype=np.float64)
        _de_cat2 = cat_df_assoc["dec_deg"].to_numpy(dtype=np.float64)
        _mtq3 = float(sip_meta.get("max_px_tight", 0.0) or 0.0)
        if not (math.isfinite(_mtq3) and _mtq3 > 0):
            _mtq3 = float(max_px_coarse)
        _post_px = max(15.0, min(52.0, float(_mtq3) * 1.28))
        fx3, fy3, fra3, fde3 = _greedy_match_pairs_pixel_wcs(
            wcs_final,
            _ra_cat2,
            _de_cat2,
            np.asarray(xs, dtype=np.float64),
            np.asarray(ys, dtype=np.float64),
            max_px=float(_post_px),
        )
        pairs_x, pairs_y, pairs_ra, pairs_de = list(fx3), list(fy3), list(fra3), list(fde3)
        sip_meta["post_nn_rematch_max_px"] = float(_post_px)
        _matched_all = int(len(pairs_x))
        sip_meta["match_rate_n_matched_all"] = int(_matched_all)
        sip_meta["match_rate_full_frame"] = float(_matched_all) / float(max(1, int(n_img)))
        # User-facing match%: najjasnejšie hviezdy (Gaia je tu takmer úplná); slabé DAO špičky bez Gaia by inak znižili %.
        _nrate = min(200, int(n_img))
        if _nrate >= 6:
            _bx = np.asarray(xs, dtype=np.float64)[: int(_nrate)]
            _by = np.asarray(ys, dtype=np.float64)[: int(_nrate)]
            fb_x, fb_y, _, _ = _greedy_match_pairs_pixel_wcs(
                wcs_final,
                _ra_cat2,
                _de_cat2,
                _bx,
                _by,
                max_px=float(_post_px),
            )
            _matched_n = int(len(fb_x))
            _match_rate = float(_matched_n) / float(int(_nrate))
            sip_meta["match_rate_n_used"] = int(_nrate)
            sip_meta["match_rate_n_matched"] = int(_matched_n)
            sip_meta["match_rate_scope"] = "brightest_n"
        else:
            _matched_n = int(_matched_all)
            _match_rate = float(_matched_n) / float(max(1, int(n_img)))
            sip_meta["match_rate_n_used"] = int(n_img)
            sip_meta["match_rate_n_matched"] = int(_matched_n)
            sip_meta["match_rate_scope"] = "all_det"
        sip_meta["match_rate_final"] = float(_match_rate)
        log_event(
            f"VYVAR platesolve final: Gaia match_rate={_match_rate * 100.0:.1f}% "
            f"({int(_matched_n)}/{int(sip_meta.get('match_rate_n_used', n_img))} "
            f"{str(sip_meta.get('match_rate_scope'))}) | all-frame≈{float(sip_meta.get('match_rate_full_frame', 0.0)) * 100.0:.1f}%"
        )
    except Exception:  # noqa: BLE001
        pass

    try:
        wh = wcs_final.to_header(relax=True)
    except Exception as exc:  # noqa: BLE001
        return {"solved": False, "reason": f"VYVAR solver: WCS header: {exc}"}
    # Update a local header (hdr0) with the complete solved WCS (incl. SIP).
    # This ensures the file stays consistent for downstream photometry.
    strip_celestial_wcs_keys(hdr0)
    hdr0.update(wh)
    try:
        hdr0["VY_MRATE"] = (
            float(_match_rate * 100.0),
            "VYVAR: Gaia match % (brightest-N subset; see VY_MRN/VY_MSCOPE)",
        )
        hdr0["VY_MSCP"] = (
            str(sip_meta.get("match_rate_scope", "all_det")),
            "VYVAR: match-rate scope: brightest_n vs all_det",
        )
        hdr0["VY_MRN"] = (
            int(sip_meta.get("match_rate_n_used", n_img) or n_img),
            "VYVAR: DAO stars in match-rate denominator",
        )
        hdr0["VY_MRM"] = (
            int(sip_meta.get("match_rate_n_matched", len(pairs_x)) or 0),
            "VYVAR: DAO stars matched (bright-N metric if brightest_n)",
        )
    except Exception:  # noqa: BLE001
        pass

    # FOCALLEN from solved pixel scale + known pixel size (if available).
    vy_platescale_arcsec_per_px: float | None = None
    try:
        if _ep_um is not None and math.isfinite(float(_ep_um)) and float(_ep_um) > 0:
            sc_deg = np.asarray(wcs_final.proj_plane_pixel_scales(), dtype=np.float64)
            if sc_deg.size > 0 and np.all(np.isfinite(sc_deg)):
                sc_arcsec_per_px = float(np.nanmean(sc_deg)) * 3600.0
                vy_platescale_arcsec_per_px = float(sc_arcsec_per_px)
                if math.isfinite(sc_arcsec_per_px) and sc_arcsec_per_px > 0:
                    foc_mm_est = float(_ep_um) * 206.265 / sc_arcsec_per_px
                    foc_mm_norm, _src = normalize_telescope_focal_mm_for_plate_scale(float(foc_mm_est))
                    if math.isfinite(float(foc_mm_norm)) and float(foc_mm_norm) > 0:
                        hdr0["FOCALLEN"] = (
                            float(foc_mm_norm),
                            "VYVAR: FOCALLEN estimated from solved WCS pixel scale and PIXSIZE/EPUM",
                        )
    except Exception:  # noqa: BLE001
        pass

    # Save measured DAO centroid kernel FWHM.
    try:
        hdr0["VY_FWHM"] = (float(_dao_fw), "VYVAR: DAO kernel FWHM [px] used by plate-solver")
    except Exception:  # noqa: BLE001
        pass

    # Save solved plate scale (arcsec/pixel).
    try:
        if vy_platescale_arcsec_per_px is not None and math.isfinite(float(vy_platescale_arcsec_per_px)) and float(vy_platescale_arcsec_per_px) > 0:
            hdr0["VY_PLTS"] = (float(vy_platescale_arcsec_per_px), "VYVAR: solved plate scale [arcsec/px]")
            hdr0["VY_PLATESCALE"] = (float(vy_platescale_arcsec_per_px), "VYVAR: solved plate scale [arcsec/px]")
    except Exception:  # noqa: BLE001
        pass

    # Write rotation hint for legacy tools: derive CROTA from WCS PC matrix (degrees).
    try:
        pc = np.asarray(wcs_final.wcs.get_pc(), dtype=np.float64)
        if pc.shape == (2, 2) and np.all(np.isfinite(pc)):
            crota = math.degrees(math.atan2(float(pc[0, 1]), float(pc[0, 0])))
            if math.isfinite(crota):
                hdr0["CROTA1"] = (float(crota), "VYVAR: derived rotation from solved WCS (deg)")
                hdr0["CROTA2"] = (float(crota), "VYVAR: derived rotation from solved WCS (deg)")
    except Exception:  # noqa: BLE001
        pass

    hdr0["VY_PSOLV"] = (True, "Plate solved by VYVAR (Gaia DR3 match)")
    hdr0["VY_GAIR"] = (float(cone_r), "Gaia query cone radius [deg] used by VYVAR")
    if isinstance(sip_meta.get("det_mirror_orientation"), str) and sip_meta.get("det_mirror_orientation") != "":
        try:
            hdr0["VY_MIRR"] = (str(sip_meta.get("det_mirror_orientation")), "VYVAR: mirror orientation winner (x/y/xy)")
        except Exception:  # noqa: BLE001
            pass
    hdr0.add_history("VYVAR: Plate solved via vyvar_platesolver (local Gaia DR3 + DAO)")
    _hist_guard = sip_meta.get("sip_rms_guard_history")
    if isinstance(_hist_guard, str) and _hist_guard:
        hdr0.add_history(_hist_guard)
    if sip_meta.get("wcs_nn_refined") and sip_meta.get("wcs_refine_rms_px") is not None:
        hdr0.add_history(
            f"VYVAR: WCS NN refinement mean residual "
            f"{float(sip_meta['wcs_refine_rms_px']):.3f} px, "
            f"n_pairs={int(sip_meta.get('wcs_refine_n_pairs', 0))}"
        )
    if _ep_um is not None:
        hdr0["VY_EPUM"] = (float(_ep_um), "Effective pixel pitch [um] used for plate-scale metadata")
    if sip_meta.get("sip_applied"):
        _lr = sip_meta.get("rms_linear_px")
        _sr = sip_meta.get("rms_sip_px")
        _rms_s = (
            f"rms_px lin={float(_lr):.3f} -> sip={float(_sr):.3f}"
            if _lr is not None and _sr is not None
            else "rms_px n/a"
        )
        hdr0.add_history(
            f"VYVAR: SIP distortion applied (order {sip_meta.get('sip_order', 3)}; {_rms_s})"
        )

    # Physical synchronization + forced disk write.
    # 1) Update HDU header in memory and flush.
    _hdr_written: fits.Header | None = None
    _data_written: np.ndarray | None = None
    try:
        with fits.open(fp, mode="update", memmap=False) as hdul_w:
            h_w = hdul_w[0].header
            strip_celestial_wcs_keys(h_w)
            h_w.update(wcs_final.to_header(relax=True))
            _hg_disk = sip_meta.get("sip_rms_guard_history")
            if isinstance(_hg_disk, str) and _hg_disk:
                h_w.add_history(_hg_disk)
            # Explicit metadata writes
            if "FOCALLEN" in hdr0:
                h_w["FOCALLEN"] = hdr0["FOCALLEN"]
            h_w["VY_FWHM"] = hdr0.get("VY_FWHM", (float(_dao_fw), "VYVAR: DAO kernel FWHM [px] used by plate-solver"))
            if "VY_PLTS" in hdr0:
                h_w["VY_PLTS"] = hdr0["VY_PLTS"]
            if "VY_PLATESCALE" in hdr0:
                h_w["VY_PLATESCALE"] = hdr0["VY_PLATESCALE"]
            hdul_w.flush()
            _hdr_written = h_w.copy()
            _data_written = np.asarray(hdul_w[0].data, dtype=np.float32)
    except Exception:  # noqa: BLE001
        _hdr_written = None
        _data_written = None

    # 2) For MASTERSTAR on disk: hard overwrite to guarantee file is physically updated.
    try:
        if _is_masterstar and _hdr_written is not None and _data_written is not None:
            fits.writeto(fp, _data_written, _hdr_written, overwrite=True)
    except Exception:  # noqa: BLE001
        pass

    if not fits_header_has_celestial_wcs(_hdr_written or hdr0):
        return {"solved": False, "reason": "VYVAR solver: WCS po zápise stále neplatný."}

    LOGGER.info(
        "VYVAR plate solve OK: %s n_match=%s sip=%s max_px_coarse/tight=%s/%s tight_pass=%s rms_lin=%s rms_sip=%s",
        fp.name,
        len(pairs_x),
        sip_meta.get("sip_applied", False),
        sip_meta.get("max_px_coarse"),
        sip_meta.get("max_px_tight"),
        sip_meta.get("refine_tight_applied"),
        sip_meta.get("rms_linear_px"),
        sip_meta.get("rms_sip_px"),
    )
    pairs_x_out = np.asarray(pairs_x, dtype=np.float64).tolist()
    pairs_y_out = np.asarray(pairs_y, dtype=np.float64).tolist()
    pairs_ra_out = np.asarray(pairs_ra, dtype=np.float64).tolist()
    pairs_de_out = np.asarray(pairs_de, dtype=np.float64).tolist()
    pairs_catalog_id: list[str] = []
    try:
        if len(pairs_x_out) > 0 and len(cat_df_cone_full) > 0 and "catalog_id" in cat_df_cone_full.columns:
            ps = SkyCoord(
                ra=np.asarray(pairs_ra_out, dtype=np.float64) * u.deg,
                dec=np.asarray(pairs_de_out, dtype=np.float64) * u.deg,
                frame="icrs",
            )
            ras = np.asarray(cat_df_cone_full["ra_deg"].to_numpy(dtype=np.float64), dtype=np.float64)
            des = np.asarray(cat_df_cone_full["dec_deg"].to_numpy(dtype=np.float64), dtype=np.float64)
            cs = SkyCoord(ra=ras * u.deg, dec=des * u.deg, frame="icrs")
            idx, sep2d, _ = ps.match_to_catalog_sky(cs)
            cids = cat_df_cone_full["catalog_id"].to_numpy()
            for k in range(len(pairs_x_out)):
                ik = int(idx[k])
                if 0 <= ik < len(cids) and sep2d[k] < 2.0 * u.arcsec:
                    pairs_catalog_id.append(str(cids[ik]))
                else:
                    pairs_catalog_id.append("")
        else:
            pairs_catalog_id = [""] * len(pairs_x_out)
    except Exception:  # noqa: BLE001
        pairs_catalog_id = [""] * len(pairs_x_out)

    return {
        "solved": True,
        "method": "vyvar_gaia_sip" if sip_meta.get("sip_applied") else "vyvar_gaia",
        "n_matched_approx": int(len(pairs_x_out)),
        "match_rate": float(_match_rate),
        "rms_px": float(_rms_px),
        "attempt": f"cone_r={cone_r:.3f}deg,mag<={eff_max_cat_mag}",
        "sip_meta": sip_meta,
        "effective_pixel_um": _ep_um,
        "vy_fwhm_px": float(_dao_fw),
        "vy_plate_scale_arcsec_per_px": float(vy_platescale_arcsec_per_px) if vy_platescale_arcsec_per_px is not None else None,
        "vy_focallen_mm": float(hdr0.get("FOCALLEN")) if hdr0.get("FOCALLEN") is not None else None,
        "pairs_x": pairs_x_out,
        "pairs_y": pairs_y_out,
        "pairs_ra": pairs_ra_out,
        "pairs_de": pairs_de_out,
        "pairs_catalog_id": pairs_catalog_id,
    }
