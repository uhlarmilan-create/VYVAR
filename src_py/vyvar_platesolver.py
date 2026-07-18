"""VYVAR plate solving: ICRS hints from FITS/UI and optional in-process solver (local Gaia DR3).

RA/Dec come from the FITS header (``VY_TARG*``, object keywords, …) or, if missing, from the in-process blind solver.
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
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import Angle, SkyCoord
from astropy.io import fits
from astropy.time import Time
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
    plate_solve_fov_deg_diagonal_from_scale,
    strip_celestial_wcs_keys,
    strip_vendor_platesolve_metadata,
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
GAIA_EPOCH = 2016.0
PM_CORRECTION_MIN_MASYR = 10.0

# WAVE-B STEP 6 (HARDCODE): plate-solve / odds-verification internals, formerly AppConfig knobs.
# Fixed to their long-standing defaults (never tuned in config history); solver mechanics.
_BLIND_PREFILTER_MIN = 4                 # was cfg.blind_prefilter_min (4); prefilter floor
_MASTERSTAR_ODDS_MATCH_FLOOR = 30        # was cfg.masterstar_odds_match_floor (30)
_MASTERSTAR_ODDS_K = 12.0                # was cfg.masterstar_odds_k (12.0)
_MASTERSTAR_ODDS_MIN_QUADRANTS = 3       # was cfg.masterstar_odds_min_quadrants (3)
_MASTERSTAR_FALSE_ALARM_P_MAX = 1e-6     # was cfg.masterstar_false_alarm_p_max (1e-6)
_MASTERSTAR_SIP_FORCE_RMS_GUARD_RATIO = 1.15  # was cfg.masterstar_sip_force_rms_guard_ratio (1.15)


def _apply_proper_motion(
    ra: float,
    dec: float,
    pmra: float | None,
    pmdec: float | None,
    obs_year: float,
) -> tuple[float, float]:
    """Propagate Gaia DR3 coordinates from epoch 2016.0 to observation year."""
    try:
        ra_f = float(ra)
        dec_f = float(dec)
    except (TypeError, ValueError):
        return ra, dec
    if not (math.isfinite(ra_f) and math.isfinite(dec_f)):
        return ra, dec
    try:
        pmra_f = float(pmra) if pmra is not None else float("nan")
        pmdec_f = float(pmdec) if pmdec is not None else float("nan")
    except (TypeError, ValueError):
        return ra_f, dec_f
    if not (math.isfinite(pmra_f) and math.isfinite(pmdec_f)):
        return ra_f, dec_f
    dt = float(obs_year) - float(GAIA_EPOCH)
    if not math.isfinite(dt) or abs(dt) < 1e-9:
        return ra_f, dec_f
    cos_dec = math.cos(math.radians(dec_f))
    if not math.isfinite(cos_dec) or abs(cos_dec) < 1e-6:
        return ra_f, dec_f
    delta_ra = (pmra_f / 3_600_000.0) * dt / cos_dec
    delta_dec = (pmdec_f / 3_600_000.0) * dt
    ra_corr = (ra_f + delta_ra) % 360.0
    dec_corr = max(-90.0, min(90.0, dec_f + delta_dec))
    return ra_corr, dec_corr


def _obs_year_from_header(header: fits.Header | None) -> float:
    """Best-effort observation year from FITS DATE-OBS, fallback to current UTC year."""
    date_obs = str((header or {}).get("DATE-OBS", "")).strip()
    if date_obs:
        try:
            return float(Time(date_obs, format="isot", scale="utc").decimalyear)
        except Exception:  # noqa: BLE001
            # EXC-0590: T4 -- DATE-OBS isot parse fail -> next format in ladder (EXCEPT-BULK 2026-07-08)
            pass
        try:
            return float(Time(date_obs, format="fits", scale="utc").decimalyear)
        except Exception:  # noqa: BLE001
            # EXC-0591: T4 -- DATE-OBS fits parse fail -> now()-year fallback (EXCEPT-BULK 2026-07-08)
            pass
    return float(datetime.utcnow().year)


def _apply_pm_to_gaia_rows(rows: list[dict[str, Any]], *, obs_year: float) -> tuple[list[dict[str, Any]], int]:
    """Apply PM correction to Gaia rows where PM is significant."""
    out: list[dict[str, Any]] = []
    n_corrected = 0
    for row in rows:
        rr = dict(row)
        try:
            pmra = rr.get("pmra")
            pmdec = rr.get("pmdec")
            pmra_f = float(pmra) if pmra is not None else 0.0
            pmdec_f = float(pmdec) if pmdec is not None else 0.0
        except (TypeError, ValueError):
            pmra_f = 0.0
            pmdec_f = 0.0
        if abs(pmra_f) > PM_CORRECTION_MIN_MASYR or abs(pmdec_f) > PM_CORRECTION_MIN_MASYR:
            ra_corr, dec_corr = _apply_proper_motion(
                rr.get("ra", float("nan")),
                rr.get("dec", float("nan")),
                rr.get("pmra"),
                rr.get("pmdec"),
                obs_year=float(obs_year),
            )
            rr["ra"] = float(ra_corr)
            rr["dec"] = float(dec_corr)
            n_corrected += 1
        out.append(rr)
    return out, n_corrected


def _get_masterstar_wcs_parity(masterstar_fits_path: Path) -> str | None:
    """Detect WCS parity from MASTERSTAR.fits.

    Returns:
        Preferred mirror orientation from MASTERSTAR (VY_MIRR) if available; otherwise
        "native" for positive determinant, "mirror_x" for negative determinant, or None.
    """
    try:
        import numpy as np
        from astropy.io import fits as _fits
        from astropy.wcs import WCS

        with _fits.open(str(masterstar_fits_path), memmap=False) as hdul:
            hdr = hdul[0].header
            vy_mirr = hdr.get("VY_MIRR", None)
            if vy_mirr is not None:
                v = str(vy_mirr).strip()
                if v in {"native", "mirror_x", "mirror_y", "mirror_xy"}:
                    log_event(f"INFO: MASTERSTAR preferred mirror z VY_MIRR={v}")
                    return v
            wcs = WCS(hdr)
        det: float | None = None
        try:
            pc = wcs.wcs.get_pc()
            det = float(pc[0][0] * pc[1][1] - pc[0][1] * pc[1][0])
        except Exception:  # noqa: BLE001
            try:
                pm = getattr(wcs, "pixel_scale_matrix", None)
                if pm is not None:
                    det = float(np.linalg.det(np.asarray(pm, dtype=float)))
            except Exception:  # noqa: BLE001
                det = None
        if det is None or (not math.isfinite(float(det))):
            return None
        if float(det) < 0:
            log_event(f"INFO: MASTERSTAR WCS parity = mirror (det={float(det):.2e})")
            return "mirror_x"
        log_event(f"INFO: MASTERSTAR WCS parity = native (det={float(det):.2e})")
        return "native"
    except Exception as e:  # noqa: BLE001
        # EXC-0592: T4 -- WCS parity detection fail surfaced as WARNING + None (EXCEPT-BULK 2026-07-08)
        log_event(f"WARNING: Nemôžem zistiť WCS paritu z MASTERSTAR: {e}")
        return None


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
    or ``none``.
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

    # 2) Common target keys: allow degrees or HMS/DMS strings.
    # Many capture apps store RA/DEC as sexagesimal strings even when the keyword is "RA".
    rv = _fits_header_pick(header, "RA", "OBJRA", "TELRA", "MNT_RA", "MOUNT_RA")
    dv = _fits_header_pick(header, "DEC", "OBJDEC", "TELDEC", "MNT_DEC", "MOUNT_DEC")
    ra = _fits_header_parse_ra_deg(rv)
    dec = _fits_header_parse_dec_deg(dv)
    if ra is not None and dec is not None:
        return ra, dec, "RA_DEC"

    # 3) OBJCTRA/OBJCTDEC as HMS/DMS strings
    rv = _fits_header_pick(header, "OBJCTRA")
    dv = _fits_header_pick(header, "OBJCTDEC")
    ra = _fits_header_parse_ra_deg(rv)
    dec = _fits_header_parse_dec_deg(dv)
    if ra is not None and dec is not None:
        return ra, dec, "OBJCTRA_DEC_hms"

    # 4) If the FITS already carries a celestial WCS, CRVAL often encodes the approximate field center.
    # We treat this only as a weak hint (not as an "accepted solve").
    try:
        cr1 = header.get("CRVAL1")
        cr2 = header.get("CRVAL2")
        if cr1 is not None and cr2 is not None:
            r = float(cr1)
            d = float(cr2)
            if math.isfinite(r) and math.isfinite(d):
                return r, d, "CRVAL"
    except (TypeError, ValueError):
        pass

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


def _adaptive_ridge(n_matches: int, sip_order: int) -> float:
    """Adaptive ridge to stabilize SIP fit when match count is low."""
    n_params = max(1, (int(sip_order) + 1) * (int(sip_order) + 2) // 2)
    ratio = float(n_matches) / float(n_params)
    if ratio >= 10.0:
        return 1e-6
    if ratio >= 5.0:
        return 1e-4
    if ratio >= 3.0:
        return 1e-2
    return 1e-1


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
    ridge_eff = _adaptive_ridge(n_matches=npts, sip_order=int(max_order))
    try:
        ridge_base = float(ridge)
    except (TypeError, ValueError):
        ridge_base = 1e-9
    if math.isfinite(ridge_base) and ridge_base > 0:
        ridge_eff = max(ridge_eff, ridge_base)
    meta["sip_ridge"] = float(ridge_eff)
    if ridge_eff >= 1e-2:
        logging.debug(
            "[SIP] Nizky pocet matched hviezd (%d) pre SIP rad %d -> adaptivny ridge=%.0e",
            int(npts),
            int(max_order),
            float(ridge_eff),
        )
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
    reg = float(ridge_eff) * np.eye(dim, dtype=np.float64)
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
    try:
        from wcs_invertibility import ensure_sip_inverse_coefficients

        w_sip = ensure_sip_inverse_coefficients(w_sip)
    except Exception:  # noqa: BLE001
        pass

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
        if _rms_sip_f > _rms_lin_f:
            meta["sip_applied"] = False
            meta["reason"] = "force_apply_blocked_rms_regression"
            meta["rms_guard_ratio"] = round(_rms_sip_f / _rms_lin_f, 4) if _rms_lin_f > 0 else None
            _hist_msg = (
                f"VYVAR: SIP rejected by RMS guard "
                f"(lin={_rms_lin_f:.3f} sip={_rms_sip_f:.3f} "
                f"ratio={_rms_sip_f / _rms_lin_f:.3f}; SIP must not exceed linear RMS)"
            )
            log_event(_hist_msg)
            meta["sip_rms_guard_history"] = _hist_msg
            return None, meta
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
            # EXC-0593: T4 -- RANSAC trial fit fail -> continue (expected-iteration failure) (EXCEPT-BULK 2026-07-08)
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


_VERIFY_GAIA_KDTREE_CACHE: dict[tuple[str, float], "_VerifyGaiaBrightCatalog"] = {}


def _radec_to_unit_xyz(ra_deg: float, dec_deg: float) -> np.ndarray:
    ra = math.radians(float(ra_deg))
    dec = math.radians(float(dec_deg))
    cos_d = math.cos(dec)
    return np.array([cos_d * math.cos(ra), cos_d * math.sin(ra), math.sin(dec)], dtype=np.float64)


def _radec_array_to_unit_xyz(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra = np.radians(np.asarray(ra_deg, dtype=np.float64))
    dec = np.radians(np.asarray(dec_deg, dtype=np.float64))
    cos_d = np.cos(dec)
    return np.column_stack([cos_d * np.cos(ra), cos_d * np.sin(ra), np.sin(dec)])


@dataclass
class _VerifyGaiaBrightCatalog:
    """All-sky bright Gaia rows + unit-sphere cKDTree for in-memory blind verify cones."""

    ra: np.ndarray
    dec: np.ndarray
    g_mag: np.ndarray
    _xyz: np.ndarray
    _tree: Any

    @classmethod
    def load(cls, db_path: Path | str, *, mag_limit: float) -> "_VerifyGaiaBrightCatalog":
        from database import load_verify_gaia_bright_stars

        p = Path(db_path).expanduser().resolve()
        ml = float(mag_limit)
        key = (str(p), ml)
        cached = _VERIFY_GAIA_KDTREE_CACHE.get(key)
        if cached is not None:
            return cached
        ra_l, dec_l, g_l = load_verify_gaia_bright_stars(p, mag_limit=ml)
        ra = np.asarray(ra_l, dtype=np.float64)
        dec = np.asarray(dec_l, dtype=np.float64)
        g_mag = np.asarray(g_l, dtype=np.float64)
        xyz = _radec_array_to_unit_xyz(ra, dec)
        from scipy.spatial import cKDTree

        obj = cls(ra=ra, dec=dec, g_mag=g_mag, _xyz=xyz, _tree=cKDTree(xyz))
        _VERIFY_GAIA_KDTREE_CACHE[key] = obj
        return obj

    def cone_indices(
        self,
        ra0: float,
        dec0: float,
        cone_r_deg: float,
        *,
        max_rows: int = 15000,
        use_box: bool = True,
    ) -> np.ndarray:
        cone_r = max(0.0, float(cone_r_deg))
        ra_half = cone_r / max(math.cos(math.radians(abs(float(dec0)))), 1e-6)
        dec_half = cone_r
        ball_r_deg = math.hypot(ra_half, dec_half)
        theta = math.radians(ball_r_deg)
        chord_r = 2.0 * math.sin(theta / 2.0)
        uv = _radec_to_unit_xyz(ra0, dec0)
        idx = self._tree.query_ball_point(uv, r=float(chord_r))
        if not idx:
            return np.zeros(0, dtype=np.int64)
        idx_arr = np.asarray(idx, dtype=np.int64)
        if use_box:
            ra_min = float(ra0) - ra_half
            ra_max = float(ra0) + ra_half
            de_min = float(dec0) - dec_half
            de_max = float(dec0) + dec_half
            ra_s = self.ra[idx_arr]
            de_s = self.dec[idx_arr]
            box = (ra_s >= ra_min) & (ra_s <= ra_max) & (de_s >= de_min) & (de_s <= de_max)
            idx_arr = idx_arr[box]
        cap = int(max_rows) if max_rows and max_rows > 0 else 0
        if cap > 0 and len(idx_arr) > cap:
            gm = self.g_mag[idx_arr]
            keep = np.argsort(gm)[:cap]
            idx_arr = idx_arr[keep]
        return idx_arr

    def cone_arrays(
        self,
        ra0: float,
        dec0: float,
        cone_r_deg: float,
        *,
        max_rows: int = 15000,
    ) -> tuple[np.ndarray, np.ndarray]:
        idx = self.cone_indices(ra0, dec0, cone_r_deg, max_rows=max_rows)
        if len(idx) == 0:
            return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
        return self.ra[idx], self.dec[idx]


def _catalog_pixel_kdtree(
    wcs_for_pred: WCS,
    ra_cat_deg: np.ndarray,
    dec_cat_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, Any]:
    pred_x, pred_y = wcs_for_pred.all_world2pix(
        np.asarray(ra_cat_deg, dtype=np.float64),
        np.asarray(dec_cat_deg, dtype=np.float64),
        0,
    )
    cat_xy = np.column_stack(
        [np.asarray(pred_x, dtype=np.float64), np.asarray(pred_y, dtype=np.float64)]
    )
    from scipy.spatial import cKDTree

    return (
        np.asarray(pred_x, dtype=np.float64),
        np.asarray(pred_y, dtype=np.float64),
        cKDTree(cat_xy),
    )


def _blind_verify_prefilter_pass(
    wcs_for_pred: WCS,
    ra_cat_deg: np.ndarray,
    dec_cat_deg: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    max_px: float,
    min_count: int,
    n_bright: int = 20,
) -> bool:
    """Cheap rough-match count before full greedy verify."""
    n_c = min(int(n_bright), int(len(ra_cat_deg)))
    n_d = min(int(n_bright), int(len(xs)))
    if n_c < 3 or n_d < 3:
        return False
    try:
        px, py = wcs_for_pred.all_world2pix(
            np.asarray(ra_cat_deg[:n_c], dtype=np.float64),
            np.asarray(dec_cat_deg[:n_c], dtype=np.float64),
            0,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("[SOLVER] blind prefilter exception: %s", exc)
        return False
    tol2 = float(max_px) * float(max_px)
    count = 0
    for k in range(n_d):
        dx = np.asarray(px, dtype=np.float64) - float(xs[k])
        dy = np.asarray(py, dtype=np.float64) - float(ys[k])
        if float(np.min(dx * dx + dy * dy)) <= tol2:
            count += 1
    return count >= int(min_count)


def _greedy_match_pairs_pixel_wcs(
    wcs_for_pred: WCS,
    ra_cat_deg: np.ndarray,
    dec_cat_deg: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    max_px: float,
    cat_pred_xy: tuple[np.ndarray, np.ndarray] | None = None,
    cat_kdtree: Any | None = None,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Match each detection to at most one catalog row by predicted pixel distance (greedy, flux order via sort)."""
    if cat_pred_xy is not None:
        pred_x, pred_y = cat_pred_xy
    else:
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

        if cat_kdtree is not None:
            tree_px = cat_kdtree
        else:
            cat_xy = np.column_stack(
                [np.asarray(pred_x, dtype=np.float64), np.asarray(pred_y, dtype=np.float64)]
            )
            tree_px = cKDTree(cat_xy)
        det_xy = np.column_stack([xs, ys])
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
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("[SOLVER] world2pix before NN refine failed: %s", exc)
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
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("[SOLVER] WCS NN refine failed: %s", exc)
        return None, out


def _sky_sep_arcsec(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    c1 = SkyCoord(ra=float(ra1) * u.deg, dec=float(dec1) * u.deg, frame="icrs")
    c2 = SkyCoord(ra=float(ra2) * u.deg, dec=float(dec2) * u.deg, frame="icrs")
    return float(c1.separation(c2).arcsec)


def _img_triangle_cyclic_sides_arcsec(
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    plate_scale_arcsec_per_px: float | None,
    x_cen: float,
    y_cen: float,
    use_gnomonic: bool,
) -> tuple[float, float, float]:
    from vyvar_blind_solver import _side_arcsec_flat, _side_arcsec_gnomonic

    p0 = np.array([float(xs[0]), float(ys[0])], dtype=np.float64)
    p1 = np.array([float(xs[1]), float(ys[1])], dtype=np.float64)
    p2 = np.array([float(xs[2]), float(ys[2])], dtype=np.float64)
    ps = float(plate_scale_arcsec_per_px or 1.0)
    if use_gnomonic:
        fn = lambda a, b: _side_arcsec_gnomonic(  # noqa: E731
            a, b, x_cen=x_cen, y_cen=y_cen, plate_scale_arcsec_per_px=ps
        )
    else:
        fn = lambda a, b: _side_arcsec_flat(a, b, plate_scale_arcsec_per_px=ps)  # noqa: E731
    return fn(p0, p1), fn(p1, p2), fn(p2, p0)


def _triangle_perm_side_rms(
    perm: tuple[int, ...],
    *,
    cat_ra: np.ndarray,
    cat_dec: np.ndarray,
    img_sides: tuple[float, float, float],
) -> float:
    i0, i1, i2 = (int(perm[0]), int(perm[1]), int(perm[2]))
    cat_s = (
        _sky_sep_arcsec(cat_ra[i0], cat_dec[i0], cat_ra[i1], cat_dec[i1]),
        _sky_sep_arcsec(cat_ra[i1], cat_dec[i1], cat_ra[i2], cat_dec[i2]),
        _sky_sep_arcsec(cat_ra[i2], cat_dec[i2], cat_ra[i0], cat_dec[i0]),
    )
    err = 0.0
    for a, b in zip(img_sides, cat_s, strict=True):
        err += (float(a) - float(b)) ** 2
    return math.sqrt(err / 3.0)


def _best_perm_for_triangle(
    xs: np.ndarray,
    ys: np.ndarray,
    cat_ra: np.ndarray,
    cat_dec: np.ndarray,
    *,
    plate_scale_arcsec_per_px: float | None,
    x_cen: float,
    y_cen: float,
    use_gnomonic: bool,
    naxis1: int | None = None,
    naxis2: int | None = None,
) -> tuple[int, int, int]:
    img_sides = _img_triangle_cyclic_sides_arcsec(
        xs,
        ys,
        plate_scale_arcsec_per_px=plate_scale_arcsec_per_px,
        x_cen=x_cen,
        y_cen=y_cen,
        use_gnomonic=use_gnomonic,
    )
    perms = sorted(
        itertools.permutations(range(3)),
        key=lambda p: _triangle_perm_side_rms(
            p, cat_ra=cat_ra, cat_dec=cat_dec, img_sides=img_sides
        ),
    )
    if naxis1 is None or naxis2 is None:
        return perms[0]
    best_perm = perms[0]
    best_score = -float("inf")
    for perm in perms:
        cra = cat_ra[list(perm)]
        cde = cat_dec[list(perm)]
        side_rms = _triangle_perm_side_rms(
            perm, cat_ra=cat_ra, cat_dec=cat_dec, img_sides=img_sides
        )
        try:
            world = SkyCoord(ra=cra * u.deg, dec=cde * u.deg, frame="icrs")
            w_try = fit_wcs_from_points((xs, ys), world, projection="TAN")
            w_try.array_shape = (int(naxis2), int(naxis1))
            px_c, py_c = w_try.all_world2pix(cra, cde, 0)
            resid = float(np.sqrt(np.mean((px_c - xs) ** 2 + (py_c - ys) ** 2)))
            score = -resid - 0.05 * side_rms
        except Exception:  # noqa: BLE001
            score = -1e6 - side_rms
        if score > best_score:
            best_score = score
            best_perm = perm
    return best_perm


def _paired_triangle_vertices(
    xs: np.ndarray,
    ys: np.ndarray,
    cat_ra: np.ndarray,
    cat_dec: np.ndarray,
    *,
    plate_scale_arcsec_per_px: float | None,
    x_cen: float,
    y_cen: float,
    use_gnomonic: bool,
    naxis1: int | None = None,
    naxis2: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    perm = _best_perm_for_triangle(
        xs,
        ys,
        cat_ra,
        cat_dec,
        plate_scale_arcsec_per_px=plate_scale_arcsec_per_px,
        x_cen=x_cen,
        y_cen=y_cen,
        use_gnomonic=use_gnomonic,
        naxis1=naxis1,
        naxis2=naxis2,
    )
    cra = cat_ra[list(perm)]
    cde = cat_dec[list(perm)]
    return xs, ys, cra, cde


_CLUSTER_RANSAC_MIN_PAIRS = 5
_CLUSTER_RANSAC_ITER = 80


def _pool_cluster_correspondences(
    members: list[Any],
    *,
    plate_scale_arcsec_per_px: float | None,
    x_cen: float,
    y_cen: float,
    use_gnomonic: bool,
    naxis1: int,
    naxis2: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Side-order paired vertices from cluster triangles; dedup by img/cat star."""
    img_used: set[tuple[float, float]] = set()
    cat_used: set[tuple[float, float]] = set()
    xs_out: list[float] = []
    ys_out: list[float] = []
    ra_out: list[float] = []
    dec_out: list[float] = []
    for h in sorted(members, key=lambda m: float(getattr(m, "hash_dist", 0.0))):
        if getattr(h, "cat_sky", None) is None:
            continue
        px = np.asarray(h.img_px, dtype=np.float64)
        sky = np.asarray(h.cat_sky, dtype=np.float64)
        xs, ys, cra, cde = _paired_triangle_vertices(
            px[:, 0],
            px[:, 1],
            sky[:, 0],
            sky[:, 1],
            plate_scale_arcsec_per_px=plate_scale_arcsec_per_px,
            x_cen=x_cen,
            y_cen=y_cen,
            use_gnomonic=use_gnomonic,
            naxis1=naxis1,
            naxis2=naxis2,
        )
        for j in range(3):
            ik = (round(float(xs[j]), 1), round(float(ys[j]), 1))
            ck = (round(float(cra[j]), 5), round(float(cde[j]), 5))
            if ik in img_used or ck in cat_used:
                continue
            img_used.add(ik)
            cat_used.add(ck)
            xs_out.append(float(xs[j]))
            ys_out.append(float(ys[j]))
            ra_out.append(float(cra[j]))
            dec_out.append(float(cde[j]))
    return (
        np.asarray(xs_out, dtype=np.float64),
        np.asarray(ys_out, dtype=np.float64),
        np.asarray(ra_out, dtype=np.float64),
        np.asarray(dec_out, dtype=np.float64),
    )


def _count_wcs_inliers(
    wcs: WCS,
    xs: np.ndarray,
    ys: np.ndarray,
    ra: np.ndarray,
    dec: np.ndarray,
    *,
    tol_px: float,
) -> np.ndarray:
    px, py = wcs.all_world2pix(ra, dec, 0)
    dist = np.hypot(px - xs, py - ys)
    return np.asarray(dist < float(tol_px), dtype=bool)


def _fit_cluster_ransac_wcs(
    xs: np.ndarray,
    ys: np.ndarray,
    ra: np.ndarray,
    dec: np.ndarray,
    *,
    naxis1: int,
    naxis2: int,
    tol_px: float,
    n_iter: int = _CLUSTER_RANSAC_ITER,
    rng: np.random.Generator | None = None,
) -> tuple[WCS | None, float, float, int]:
    n = int(len(xs))
    if n < 3:
        return None, float("nan"), float("nan"), 0
    gen = rng or np.random.default_rng(42)
    best_inlier_idx: np.ndarray | None = None
    best_n = 0
    for _ in range(int(n_iter)):
        if n == 3:
            sample = np.arange(3, dtype=np.int64)
        else:
            sample = gen.choice(n, size=3, replace=False)
        try:
            world = SkyCoord(ra=ra[sample] * u.deg, dec=dec[sample] * u.deg, frame="icrs")
            w_try = fit_wcs_from_points((xs[sample], ys[sample]), world, projection="TAN")
            w_try.array_shape = (int(naxis2), int(naxis1))
        except Exception:  # noqa: BLE001
            # EXC-0594: T4 -- RANSAC triangle trial fail -> continue (EXCEPT-BULK 2026-07-08)
            continue
        inliers = _count_wcs_inliers(w_try, xs, ys, ra, dec, tol_px=tol_px)
        n_inl = int(np.count_nonzero(inliers))
        if n_inl > best_n:
            best_n = n_inl
            best_inlier_idx = inliers
    if best_inlier_idx is None or best_n < 3:
        return None, float("nan"), float("nan"), 0
    idx = np.flatnonzero(best_inlier_idx)
    try:
        world = SkyCoord(ra=ra[idx] * u.deg, dec=dec[idx] * u.deg, frame="icrs")
        w_final = fit_wcs_from_points((xs[idx], ys[idx]), world, projection="TAN")
        w_final.array_shape = (int(naxis2), int(naxis1))
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("[SOLVER] cluster RANSAC WCS fit failed: %s", exc)
        return None, float("nan"), float("nan"), 0
    x_cen = float(naxis1) / 2.0
    y_cen = float(naxis2) / 2.0
    fc = w_final.all_pix2world(x_cen, y_cen, 0)
    return w_final, float(fc[0]), float(fc[1]), int(len(idx))


def _wcs_scale_gate(
    wcs: WCS,
    *,
    known_ps: float,
    scale_tol: float,
) -> tuple[bool, float]:
    from astropy.wcs.utils import proj_plane_pixel_scales

    sc_deg = proj_plane_pixel_scales(wcs)
    fitted_ps = float(np.mean(np.abs(np.asarray(sc_deg, dtype=np.float64))) * 3600.0)
    ok = abs(fitted_ps / known_ps - 1.0) <= scale_tol
    return ok, fitted_ps


def _triangle_wcs_from_candidate(
    xs: np.ndarray,
    ys: np.ndarray,
    cat_ra: np.ndarray,
    cat_dec: np.ndarray,
    *,
    naxis1: int,
    naxis2: int,
    plate_scale_arcsec_per_px: float | None,
    x_cen: float,
    y_cen: float,
    use_gnomonic: bool,
) -> tuple[WCS | None, float, float]:
    perm = _best_perm_for_triangle(
        xs,
        ys,
        cat_ra,
        cat_dec,
        plate_scale_arcsec_per_px=plate_scale_arcsec_per_px,
        x_cen=x_cen,
        y_cen=y_cen,
        use_gnomonic=use_gnomonic,
        naxis1=naxis1,
        naxis2=naxis2,
    )
    cra = cat_ra[list(perm)]
    cde = cat_dec[list(perm)]
    world = SkyCoord(ra=cra * u.deg, dec=cde * u.deg, frame="icrs")
    wcs_seed = fit_wcs_from_points((xs, ys), world, projection="TAN")
    wcs_seed.array_shape = (int(naxis2), int(naxis1))
    fc = wcs_seed.all_pix2world(x_cen, y_cen, 0)
    return wcs_seed, float(fc[0]), float(fc[1])


def _best_triangle_wcs_in_cluster(
    members: list[Any],
    *,
    naxis1: int,
    naxis2: int,
    plate_scale_arcsec_per_px: float | None,
    x_cen: float,
    y_cen: float,
    use_gnomonic: bool,
    known_ps: float | None,
    scale_tol: float,
    rig_prior: bool,
    max_try: int = 12,
) -> tuple[WCS | None, float, float]:
    """Best scale-valid per-triangle TAN WCS among cluster members (by hash_dist)."""
    best_w: WCS | None = None
    best_ra = best_de = float("nan")
    best_score = -float("inf")
    for h in sorted(members, key=lambda m: float(getattr(m, "hash_dist", 0.0)))[: int(max_try)]:
        if getattr(h, "cat_sky", None) is None:
            continue
        px = np.asarray(h.img_px, dtype=np.float64)
        sky = np.asarray(h.cat_sky, dtype=np.float64)
        try:
            w_try, ra0, de0 = _triangle_wcs_from_candidate(
                px[:, 0],
                px[:, 1],
                sky[:, 0],
                sky[:, 1],
                naxis1=int(naxis1),
                naxis2=int(naxis2),
                plate_scale_arcsec_per_px=plate_scale_arcsec_per_px,
                x_cen=x_cen,
                y_cen=y_cen,
                use_gnomonic=use_gnomonic,
            )
        except Exception:  # noqa: BLE001
            # EXC-0595: T4 -- candidate triangle WCS fail -> continue (EXCEPT-BULK 2026-07-08)
            continue
        if rig_prior and known_ps is not None:
            ok, _ = _wcs_scale_gate(w_try, known_ps=known_ps, scale_tol=scale_tol)
            if not ok:
                continue
        xs, ys, cra, cde = _paired_triangle_vertices(
            px[:, 0],
            px[:, 1],
            sky[:, 0],
            sky[:, 1],
            plate_scale_arcsec_per_px=plate_scale_arcsec_per_px,
            x_cen=x_cen,
            y_cen=y_cen,
            use_gnomonic=use_gnomonic,
            naxis1=int(naxis1),
            naxis2=int(naxis2),
        )
        px_c, py_c = w_try.all_world2pix(cra, cde, 0)
        resid = float(np.sqrt(np.mean((px_c - xs) ** 2 + (py_c - ys) ** 2)))
        score = -resid - 0.01 * float(getattr(h, "hash_dist", 0.0))
        if score > best_score:
            best_score = score
            best_w = w_try
            best_ra = ra0
            best_de = de0
    return best_w, best_ra, best_de


def _cluster_wcs_seed(
    members: list[Any],
    *,
    naxis1: int,
    naxis2: int,
    plate_scale_arcsec_per_px: float | None,
    x_cen: float,
    y_cen: float,
    use_gnomonic: bool,
    known_ps: float | None,
    scale_tol: float,
    rig_prior: bool,
    tol_px: float,
) -> tuple[WCS | None, float, float, str, int, int]:
    """RANSAC on pooled pairs when consistent; else best per-triangle in cluster."""
    px_a, py_a, ra_a, de_a = _pool_cluster_correspondences(
        members,
        plate_scale_arcsec_per_px=plate_scale_arcsec_per_px,
        x_cen=x_cen,
        y_cen=y_cen,
        use_gnomonic=use_gnomonic,
        naxis1=int(naxis1),
        naxis2=int(naxis2),
    )
    n_pairs = int(len(px_a))
    if n_pairs >= _CLUSTER_RANSAC_MIN_PAIRS:
        w_r, ra0, de0, n_inl = _fit_cluster_ransac_wcs(
            px_a,
            py_a,
            ra_a,
            de_a,
            naxis1=int(naxis1),
            naxis2=int(naxis2),
            tol_px=tol_px,
        )
        ransac_ok = w_r is not None and n_inl >= max(5, n_pairs // 6)
        if ransac_ok and rig_prior and known_ps is not None:
            ok, _ = _wcs_scale_gate(w_r, known_ps=known_ps, scale_tol=scale_tol)
            ransac_ok = ok
        if ransac_ok and w_r is not None:
            return w_r, ra0, de0, "cluster_ransac", n_pairs, int(n_inl)
    w_t, ra_t, de_t = _best_triangle_wcs_in_cluster(
        members,
        naxis1=int(naxis1),
        naxis2=int(naxis2),
        plate_scale_arcsec_per_px=plate_scale_arcsec_per_px,
        x_cen=x_cen,
        y_cen=y_cen,
        use_gnomonic=use_gnomonic,
        known_ps=known_ps,
        scale_tol=scale_tol,
        rig_prior=rig_prior,
    )
    if w_t is not None:
        mode = "triangle_fallback" if n_pairs >= _CLUSTER_RANSAC_MIN_PAIRS else "triangle"
        return w_t, ra_t, de_t, mode, n_pairs, 0
    return None, float("nan"), float("nan"), "triangle", n_pairs, 0


def _verify_blind_candidates(
    candidates: list[Any],
    *,
    dao_df: pd.DataFrame,
    gaia_db_path: Path | str,
    fov_deg: float,
    naxis1: int,
    naxis2: int,
    pixel_pitch_um: float | None,
    focal_length_mm: float | None,
    max_cat_mag: float,
    known_plate_scale_arcsec_per_px: float | None = None,
    app_config: Any | None = None,
    debug_sink: dict | None = None,
) -> tuple[float, float] | None:
    """Verify blind vote hypotheses by back-projecting Gaia and greedy pixel matching."""
    import time

    from database import query_local_gaia
    from vyvar_blind_solver import BlindCandidate, _dao_flux_column, _rig_prior_enabled, _scale_tol_frac

    if not candidates:
        return None

    _cfg = app_config or AppConfig()
    top_tol = float(getattr(_cfg, "blind_verify_match_tol_px", 2.5))
    min_matches = int(getattr(_cfg, "blind_verify_min_matches", 12))
    min_fraction = float(getattr(_cfg, "blind_verify_min_fraction", 0.30))
    _dbg = bool(getattr(_cfg, "debug_platesolver", False))
    _rig_prior = _rig_prior_enabled(_cfg)
    _scale_tol = _scale_tol_frac(_cfg)
    _use_inmem = bool(getattr(_cfg, "blind_verify_inmemory_catalog", True))
    _verify_mag = float(getattr(_cfg, "verify_mag_limit", 14.0))
    _prefilter_min = _BLIND_PREFILTER_MIN  # WAVE-B STEP 6: hardcoded solver internal
    _early_accept = int(getattr(_cfg, "blind_verify_early_accept", 30))
    _early_floor_cfg = int(getattr(_cfg, "blind_verify_early_floor", 0))
    _early_fraction = float(getattr(_cfg, "blind_verify_early_fraction", 0.0))
    _query_max_rows = int(getattr(_cfg, "catalog_query_max_rows", 15000) or 15000)
    _x_cen = float(naxis1) / 2.0
    _y_cen = float(naxis2) / 2.0
    _known_ps: float | None = None
    if known_plate_scale_arcsec_per_px is not None:
        try:
            _kp = float(known_plate_scale_arcsec_per_px)
            if math.isfinite(_kp) and _kp > 0:
                _known_ps = _kp
                if _rig_prior and _kp >= 5.0:
                    min_matches = max(6, int(min_matches) - 4)
        except (TypeError, ValueError):
            pass

    if _early_floor_cfg > 0:
        _early_floor = max(_early_accept, _early_floor_cfg)
    else:
        _early_floor = max(_early_accept, int(min_matches) * 6)

    xs_dao = np.asarray(dao_df["x"].values, dtype=np.float64)
    ys_dao = np.asarray(dao_df["y"].values, dtype=np.float64)
    fov_f = max(0.05, float(fov_deg))
    # Match-fraction uses brightest detections only — full 5σ lists can be 5000+ and collapse fraction.
    _dao_mult = 30 if (_rig_prior and fov_f >= 2.0) else 4
    _cap = max(int(min_matches) * _dao_mult, 60)
    if _rig_prior and fov_f >= 2.0:
        _cap = min(int(_cap), 240)
    if len(xs_dao) > _cap:
        _flux_col = _dao_flux_column(dao_df)
        _sub = dao_df.sort_values(_flux_col, ascending=False).head(_cap)
        xs_dao = np.asarray(_sub["x"].values, dtype=np.float64)
        ys_dao = np.asarray(_sub["y"].values, dtype=np.float64)
    n_dao = int(len(xs_dao))
    if n_dao < 3:
        return None

    root = Path(gaia_db_path).expanduser().resolve()
    _use_gnomonic_sides = bool(_rig_prior) and fov_f >= 2.0
    if _rig_prior and _known_ps is not None and math.isfinite(fov_f) and fov_f > 0:
        cone_r = max(0.5, float(fov_f) * 0.55)
    elif pixel_pitch_um is not None and focal_length_mm is not None:
        try:
            _p_um = float(pixel_pitch_um)
            _f_mm = float(focal_length_mm)
        except (TypeError, ValueError):
            _p_um, _f_mm = 0.0, 0.0
        if math.isfinite(_p_um) and _p_um > 0 and math.isfinite(_f_mm) and _f_mm > 0:
            cone_r = catalog_cone_radius_deg_from_optics(
                naxis1=int(naxis1),
                naxis2=int(naxis2),
                pixel_pitch_um=_p_um,
                focal_length_mm=focal_length_mm,
                margin=0.85,
                fov_diameter_fallback_deg=fov_f,
            )
        else:
            cone_r = catalog_cone_radius_from_fov_diameter_deg(fov_f)
    else:
        cone_r = catalog_cone_radius_from_fov_diameter_deg(fov_f)
    if not (_rig_prior and _known_ps is not None):
        cone_r = max(float(cone_r), max(2.0, float(fov_f) * 1.5))

    deduped: list[BlindCandidate] = []
    for cand in candidates:
        if not isinstance(cand, BlindCandidate):
            continue
        dup = False
        for kept in deduped:
            dra = (cand.center_ra - kept.center_ra) * math.cos(math.radians(cand.center_dec))
            ddec = cand.center_dec - kept.center_dec
            if math.hypot(dra, ddec) < 0.3:
                dup = True
                break
        if not dup:
            deduped.append(cand)

    _inmem_cat: _VerifyGaiaBrightCatalog | None = None
    _load_mag = float(max_cat_mag)
    catalog_load_s = 0.0
    if _use_inmem:
        t_cat = time.monotonic()
        try:
            _load_mag = float(max_cat_mag)
            _db_cap = float(get_gaia_db_max_g_mag(root))
            if math.isfinite(_db_cap) and _db_cap > 0.0:
                _load_mag = min(_load_mag, _db_cap)
            if float(_verify_mag) > 0.0:
                _load_mag = min(_load_mag, float(_verify_mag))
            _inmem_cat = _VerifyGaiaBrightCatalog.load(root, mag_limit=_load_mag)
            catalog_load_s = time.monotonic() - t_cat
            log_event(
                f"INFO: Blind verify in-memory Gaia: {len(_inmem_cat.ra)} stars "
                f"(g<={_load_mag:.1f}), ready in {catalog_load_s:.2f}s"
            )
        except Exception as exc:  # noqa: BLE001
            catalog_load_s = time.monotonic() - t_cat
            log_event(
                f"WARN: in-memory verify catalog failed ({exc!s}); using per-cone DB queries"
            )
            _inmem_cat = None

    verified_rows: list[dict[str, Any]] = []
    best: tuple[float, float, float, int, int, int, int] | None = None
    # ra, dec, frac, n_match, n_cat_in_frame, n_dao, cone_n_cat
    best_hash = float("inf")
    best_cand_idx = -1
    early_exit_fired = False
    early_exit_cand_idx = -1
    t0 = time.monotonic()

    for i, cand in enumerate(deduped):
        vote_ra = float(cand.center_ra)
        vote_dec = float(cand.center_dec)
        row: dict[str, Any] = {
            "idx": i,
            "center_ra": vote_ra,
            "center_dec": vote_dec,
            "hash_dist": float(cand.hash_dist),
            "vote_count": int(cand.vote_count),
            "accepted": False,
        }
        xs = np.asarray(cand.img_px[:, 0], dtype=np.float64)
        ys = np.asarray(cand.img_px[:, 1], dtype=np.float64)
        cat_ra = np.asarray(cand.cat_sky[:, 0], dtype=np.float64)
        cat_dec = np.asarray(cand.cat_sky[:, 1], dtype=np.float64)
        wcs_seed: WCS | None = None
        ra0 = de0 = float("nan")
        try:
            cluster_members = getattr(cand, "cluster_members", None)
            if cluster_members:
                wcs_seed, ra0, de0, v_mode, n_cp, n_inl = _cluster_wcs_seed(
                    cluster_members,
                    naxis1=int(naxis1),
                    naxis2=int(naxis2),
                    plate_scale_arcsec_per_px=_known_ps,
                    x_cen=_x_cen,
                    y_cen=_y_cen,
                    use_gnomonic=_use_gnomonic_sides,
                    known_ps=_known_ps,
                    scale_tol=_scale_tol,
                    rig_prior=_rig_prior,
                    tol_px=top_tol,
                )
                row["cluster_pairs"] = int(n_cp)
                row["verify_mode"] = v_mode
                row["ransac_inliers"] = int(n_inl)
                if wcs_seed is None:
                    raise ValueError("cluster WCS failed")
            if wcs_seed is None:
                row["verify_mode"] = "triangle"
                wcs_seed, ra0, de0 = _triangle_wcs_from_candidate(
                    xs,
                    ys,
                    cat_ra,
                    cat_dec,
                    naxis1=int(naxis1),
                    naxis2=int(naxis2),
                    plate_scale_arcsec_per_px=_known_ps,
                    x_cen=_x_cen,
                    y_cen=_y_cen,
                    use_gnomonic=_use_gnomonic_sides,
                )
            row["field_center_ra"] = ra0
            row["field_center_dec"] = de0
            if _rig_prior and _known_ps is not None:
                ok, fitted_ps = _wcs_scale_gate(
                    wcs_seed, known_ps=_known_ps, scale_tol=_scale_tol
                )
                row["fitted_plate_scale_arcsec_px"] = fitted_ps
                if not ok:
                    row["error"] = (
                        f"scale_mismatch: fitted={fitted_ps:.3f} known={_known_ps:.3f}"
                    )
                    verified_rows.append(row)
                    continue
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"wcs_seed:{exc!s}"
            verified_rows.append(row)
            continue

        try:
            if _inmem_cat is not None:
                ra_cat, de_cat = _inmem_cat.cone_arrays(
                    ra0,
                    de0,
                    cone_r,
                    max_rows=_query_max_rows,
                )
            else:
                ra_min = ra0 - float(cone_r) / max(math.cos(math.radians(abs(de0))), 1e-6)
                ra_max = ra0 + float(cone_r) / max(math.cos(math.radians(abs(de0))), 1e-6)
                de_min = de0 - float(cone_r)
                de_max = de0 + float(cone_r)
                rows_g = query_local_gaia(
                    root,
                    ra_min=ra_min,
                    ra_max=ra_max,
                    dec_min=de_min,
                    dec_max=de_max,
                    mag_limit=float(max_cat_mag),
                    max_rows=_query_max_rows,
                )
                if not rows_g:
                    ra_cat = np.zeros(0, dtype=np.float64)
                    de_cat = np.zeros(0, dtype=np.float64)
                else:
                    ra_cat = np.asarray([float(r["ra"]) for r in rows_g], dtype=np.float64)
                    de_cat = np.asarray([float(r["dec"]) for r in rows_g], dtype=np.float64)
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"gaia_query:{exc!s}"
            verified_rows.append(row)
            continue
        if len(ra_cat) == 0:
            row.update(
                {
                    "n_dao": n_dao,
                    "n_cat": 0,
                    "n_cat_in_frame": 0,
                    "n_matched": 0,
                    "fraction": 0.0,
                }
            )
            verified_rows.append(row)
            continue

        wcs_use: WCS = wcs_seed
        _cl_mem = getattr(cand, "cluster_members", None)
        if _cl_mem and len(_cl_mem) > 1:
            best_n_prev = -1
            best_w_prev: WCS | None = None
            best_hd_prev = float("inf")
            for h in sorted(_cl_mem, key=lambda m: float(getattr(m, "hash_dist", 0.0)))[
                : min(12, len(_cl_mem))
            ]:
                if getattr(h, "cat_sky", None) is None:
                    continue
                try:
                    px = np.asarray(h.img_px, dtype=np.float64)
                    sky = np.asarray(h.cat_sky, dtype=np.float64)
                    w_t, _ra_t, _de_t = _triangle_wcs_from_candidate(
                        px[:, 0],
                        px[:, 1],
                        sky[:, 0],
                        sky[:, 1],
                        naxis1=int(naxis1),
                        naxis2=int(naxis2),
                        plate_scale_arcsec_per_px=_known_ps,
                        x_cen=_x_cen,
                        y_cen=_y_cen,
                        use_gnomonic=_use_gnomonic_sides,
                    )
                    if _rig_prior and _known_ps is not None:
                        ok_t, _ = _wcs_scale_gate(
                            w_t, known_ps=_known_ps, scale_tol=_scale_tol
                        )
                        if not ok_t:
                            continue
                    px_pred, py_pred, tree_t = _catalog_pixel_kdtree(
                        w_t, ra_cat, de_cat
                    )
                    px_p, _, _, _ = _greedy_match_pairs_pixel_wcs(
                        w_t,
                        ra_cat,
                        de_cat,
                        xs_dao,
                        ys_dao,
                        max_px=top_tol,
                        cat_pred_xy=(px_pred, py_pred),
                        cat_kdtree=tree_t,
                    )
                    n_prev = int(len(px_p))
                    hd = float(getattr(h, "hash_dist", 0.0))
                    if n_prev > best_n_prev or (
                        n_prev == best_n_prev and hd < best_hd_prev
                    ):
                        best_n_prev = n_prev
                        best_hd_prev = hd
                        best_w_prev = w_t
                except Exception:  # noqa: BLE001
                    # EXC-0596: T4 -- candidate refine loop fail -> continue (EXCEPT-BULK 2026-07-08)
                    continue
            if best_w_prev is not None:
                wcs_use = best_w_prev
                wcs_seed = best_w_prev
                fc = wcs_seed.all_pix2world(_x_cen, _y_cen, 0)
                ra0, de0 = float(fc[0]), float(fc[1])
                row["field_center_ra"] = ra0
                row["field_center_dec"] = de0
                row["triangle_sweep_n"] = int(best_n_prev)
        elif not _blind_verify_prefilter_pass(
            wcs_use,
            ra_cat,
            de_cat,
            xs_dao,
            ys_dao,
            max_px=top_tol,
            min_count=_prefilter_min,
        ):
            row.update(
                {
                    "n_dao": n_dao,
                    "n_cat": int(len(ra_cat)),
                    "n_cat_in_frame": int(len(ra_cat)),
                    "n_matched": 0,
                    "fraction": 0.0,
                    "prefilter_skip": True,
                }
            )
            verified_rows.append(row)
            continue
        w_ref, _ref_meta = _refine_wcs_tan_nn_gaia(
            wcs_use,
            xs_det=xs_dao,
            ys_det=ys_dao,
            ra_cat_full_deg=ra_cat,
            dec_cat_full_deg=de_cat,
            max_match_px=top_tol,
            min_pairs=min(12, min_matches),
        )
        if w_ref is not None:
            wcs_use = w_ref

        cat_px_x, cat_px_y, cat_tree = _catalog_pixel_kdtree(
            wcs_use, ra_cat, de_cat
        )
        px_m, py_m, _, _ = _greedy_match_pairs_pixel_wcs(
            wcs_use,
            ra_cat,
            de_cat,
            xs_dao,
            ys_dao,
            max_px=top_tol,
            cat_pred_xy=(cat_px_x, cat_px_y),
            cat_kdtree=cat_tree,
        )
        n_matched = int(len(px_m))
        n_cat = int(len(ra_cat))
        if _rig_prior and fov_f >= 2.0:
            n_cat_in_frame = int(n_dao)
            denom = max(1, n_dao)
            min_fraction_eff = min(
                float(min_fraction),
                float(min_matches) / float(max(1, n_dao)),
            )
        else:
            n_cat_in_frame = n_cat
            denom = max(1, n_cat_in_frame)
            min_fraction_eff = float(min_fraction)
        fraction = float(n_matched) / float(denom)
        vote_ra_g = float(cand.center_ra)
        vote_dec_g = float(cand.center_dec)
        _cl_mem = getattr(cand, "cluster_members", None)
        if _cl_mem:
            vote_ra_g = float(np.median([float(h.center_ra) for h in _cl_mem]))
            vote_dec_g = float(np.median([float(h.center_dec) for h in _cl_mem]))
        dra_v = (ra0 - vote_ra_g) * math.cos(math.radians((de0 + vote_dec_g) / 2.0))
        ddec_v = de0 - vote_dec_g
        vote_fc_sep = float(math.hypot(dra_v, ddec_v))
        row["vote_fc_sep_deg"] = vote_fc_sep
        vote_fc_ok = True
        if _rig_prior and fov_f >= 2.0:
            vote_fc_ok = vote_fc_sep <= 2.0
        row.update(
            {
                "n_dao": n_dao,
                "n_cat": n_cat,
                "n_cat_in_frame": n_cat_in_frame,
                "n_matched": n_matched,
                "fraction": fraction,
            }
        )
        accepted = (
            n_matched >= min_matches
            and fraction >= min_fraction_eff
            and vote_fc_ok
        )
        row["accepted"] = accepted
        verified_rows.append(row)
        if _dbg:
            log_event(
                f"DEBUG: Blind verify cand[{i}]: RA={ra0:.3f} Dec={de0:.3f} "
                f"matched={n_matched}/{denom} fraction={fraction:.3f} "
                f"hash_dist={cand.hash_dist:.4f} votes={cand.vote_count} "
                f"{'ACCEPT' if accepted else 'reject'}"
            )
        if accepted and (
            best is None
            or n_matched > best[3]
            or (
                n_matched == best[3]
                and float(cand.hash_dist) < best_hash - 1e-15
            )
        ):
            best_hash = float(cand.hash_dist)
            best = (ra0, de0, fraction, n_matched, n_cat_in_frame, n_dao, n_cat)
            best_cand_idx = int(i)
            _early_exit = False
            if _early_fraction > 0.0:
                _early_exit = fraction >= _early_fraction
            elif n_matched >= _early_floor:
                _early_exit = True
            if _early_exit:
                early_exit_fired = True
                early_exit_cand_idx = int(i)
                if _dbg:
                    log_event(
                        f"DEBUG: Blind verify early-exit at cand[{i}] "
                        f"n_matched={n_matched} fraction={fraction:.3f} "
                        f"early_frac={_early_fraction:.3f} floor={_early_floor}"
                    )
                break

    verify_s = time.monotonic() - t0
    _n_checked = sum(1 for r in verified_rows if not r.get("prefilter_skip"))
    log_event(
        f"INFO: Blind verify: {len(deduped)} kandidátov, "
        f"{sum(1 for r in verified_rows if r.get('accepted'))} akceptovaných, "
        f"{_n_checked} plne overených, verify={verify_s:.2f}s load={catalog_load_s:.2f}s"
    )
    max_false_n_matched = max(
        (int(r.get("n_matched", 0)) for r in verified_rows if not r.get("accepted")),
        default=0,
    )
    if debug_sink is not None:
        debug_sink["verified_candidates"] = verified_rows
        debug_sink["verify_elapsed_s"] = float(verify_s)
        debug_sink["verify_inmemory_catalog"] = _inmem_cat is not None
        debug_sink["verify_metrics"] = {
            "catalog_load_s": round(float(catalog_load_s), 3),
            "verify_s": round(float(verify_s), 3),
            "verify_mag_limit": float(_load_mag),
            "early_exit_fired": bool(early_exit_fired),
            "early_exit_cand_idx": int(early_exit_cand_idx),
            "max_false_n_matched": int(max_false_n_matched),
            "winner_cand_idx": int(best_cand_idx),
            "n_candidates": int(len(deduped)),
            "n_verified_rows": int(len(verified_rows)),
        }
        if best is not None:
            debug_sink["verify_winner"] = {
                "ra": best[0],
                "dec": best[1],
                "fraction": best[2],
                "n_matched": best[3],
                "n_cat_in_frame": best[4],
                "n_dao": best[5],
                "cone_n_cat": best[6],
            }

    if best is None:
        return None
    log_event(
        f"INFO: Blind verify winner: RA={best[0]:.4f} Dec={best[1]:.4f} "
        f"fraction={best[2]:.3f} matched={best[3]}"
    )
    return best[0], best[1]


def _log_wcs_orientation_header_hints(wcs_obj: WCS, hdr: fits.Header) -> None:
    """Heuristics for mirrored / flipped acquisition (SIPS etc.)."""
    try:
        xb, yb = fits_binning_xy_from_header(hdr)
        log_event(f"WCS diag: FITS binning ≈ {int(xb)}×{int(yb)} (XBINNING/YBINNING) — over zhodu s efektívnym pixelom pri mierke.")
    except Exception:  # noqa: BLE001
        # EXC-0597: T3 -- WCS diag block (wraps fits_binning_xy_from_header) skipped (EXCEPT-BULK 2026-07-08)
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
        # EXC-0598: T3 -- WCS diag block (wraps proj_plane_pixel_scales) skipped (EXCEPT-BULK 2026-07-08)
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
        # EXC-0599: T3 -- WCS diag block (wraps det computation) skipped (EXCEPT-BULK 2026-07-08)
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


def _fits_roworder_yflip_applied(hdr: Any) -> bool:
    ro = str(hdr.get("ROWORDER", "") or "").strip().upper().replace("_", "-")
    return ro == "BOTTOM-UP"


def _apply_fits_roworder_to_detections(
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    hdr: Any,
    naxis2: int,
) -> tuple[np.ndarray, np.ndarray, str | None]:
    """Map DAO centroids to top-down image frame when FITS ``ROWORDER=BOTTOM-UP``."""
    x = np.asarray(xs, dtype=np.float64, copy=True)
    y = np.asarray(ys, dtype=np.float64, copy=True)
    if _fits_roworder_yflip_applied(hdr):
        y = (float(naxis2) - 1.0) - y
        return x, y, "bottom_up_yflip"
    return x, y, None


def _sip_match_max_px(max_px_coarse: float) -> float:
    """Match radius for SIP fitting — wide enough to include edge stars (distortion constraints)."""
    return max(15.0, min(48.0, float(max_px_coarse) * 0.55))


def _compute_masterstar_catalog_recovery(
    wcs: WCS,
    cat_ra: np.ndarray,
    cat_de: np.ndarray,
    xs_det: np.ndarray,
    ys_det: np.ndarray,
    *,
    naxis1: int,
    naxis2: int,
    qa_px: float,
    tight_px: float = 2.5,
) -> dict[str, Any]:
    """Catalog-denominated recovery at ``wcs_final`` (Gaia-in-frame vs DAO matches)."""
    out: dict[str, Any] = {
        "n_cat_in_frame": 0,
        "n_matched_coarse": 0,
        "n_matched_tight": 0,
        "catalog_recovery_coarse": 0.0,
        "catalog_recovery_tight": 0.0,
    }
    ra_a = np.asarray(cat_ra, dtype=np.float64)
    de_a = np.asarray(cat_de, dtype=np.float64)
    xs_a = np.asarray(xs_det, dtype=np.float64)
    ys_a = np.asarray(ys_det, dtype=np.float64)
    if len(ra_a) == 0 or len(xs_a) == 0:
        return out
    try:
        xp, yp = wcs.all_world2pix(ra_a, de_a, 0)
        xp = np.asarray(xp, dtype=np.float64)
        yp = np.asarray(yp, dtype=np.float64)
        in_frame = (
            np.isfinite(xp)
            & np.isfinite(yp)
            & (xp >= 0.0)
            & (xp < float(naxis1))
            & (yp >= 0.0)
            & (yp < float(naxis2))
        )
        ra_if = ra_a[in_frame]
        de_if = de_a[in_frame]
        n_cat = int(len(ra_if))
        out["n_cat_in_frame"] = n_cat
        if n_cat <= 0:
            return out
        qx_c, _, _, _ = _greedy_match_pairs_pixel_wcs(
            wcs,
            ra_if,
            de_if,
            xs_a,
            ys_a,
            max_px=float(qa_px),
        )
        qx_t, qy_t, _, _ = _greedy_match_pairs_pixel_wcs(
            wcs,
            ra_if,
            de_if,
            xs_a,
            ys_a,
            max_px=float(tight_px),
        )
        n_coarse = int(len(qx_c))
        n_tight = int(len(qx_t))
        n_det = int(len(xs_a))
        out["n_detections_used"] = n_det
        out["n_matched_coarse"] = n_coarse
        out["n_matched_tight"] = n_tight
        out["catalog_recovery_coarse"] = float(n_coarse) / float(n_cat)
        out["catalog_recovery_tight"] = float(n_tight) / float(n_cat)
        # Gate fraction (legacy QA flag only under odds acceptance): at most one Gaia star per DAO peak.
        n_denom = int(min(n_cat, n_det)) if n_det > 0 else n_cat
        out["catalog_recovery_denom"] = n_denom
        out["catalog_recovery_tight_gate"] = float(n_tight) / float(max(1, n_denom))
        out["catalog_recovery_coarse_gate"] = float(n_coarse) / float(max(1, n_denom))
        out["quadrants_with_match"] = _sibling_quadrant_count(
            np.asarray(qx_t, dtype=np.float64),
            np.asarray(qy_t, dtype=np.float64),
            int(naxis1),
            int(naxis2),
        )
        area = float(max(1, int(naxis1) * int(naxis2)))
        p_one = min(1.0, float(n_cat) * math.pi * float(tight_px) ** 2 / area)
        out["expected_random"] = float(n_det) * p_one
        out["false_alarm_p"] = _sibling_false_alarm_p(
            n_tight, n_det, n_cat, int(naxis1), int(naxis2), r_px=float(tight_px)
        )
    except Exception as exc:  # noqa: BLE001
        out["catalog_recovery_error"] = repr(exc)
    return out


def _masterstar_quality_flags(
    *,
    catalog_recovery_tight_gate: float,
    recovery_min: float,
    n_cat_in_frame: int,
    centre_rms: float | None,
    centre_rms_max: float,
    dist_benign: bool,
    crowded_n_cat_min: int = 800,
) -> dict[str, Any]:
    """Non-gating quality metadata for MASTERSTAR (trust / photometry QA)."""
    tags: list[str] = []
    if float(catalog_recovery_tight_gate) < float(recovery_min):
        tags.append("low_recovery")
    if int(n_cat_in_frame) >= int(crowded_n_cat_min):
        tags.append("crowded")
    if centre_rms is not None and math.isfinite(float(centre_rms)) and float(centre_rms) > float(centre_rms_max):
        tags.append("blurred")
    if not bool(dist_benign):
        tags.append("distorted")
    if "crowded" in tags:
        qflag = "crowded"
    elif "blurred" in tags:
        qflag = "blurred"
    elif "distorted" in tags:
        qflag = "distorted"
    elif "low_recovery" in tags:
        qflag = "low_recovery"
    else:
        qflag = "ok"
    return {
        "quality_flags": tags,
        "quality_flag_primary": qflag,
    }


def _masterstar_solve_acceptance(
    *,
    accept_mode: str = "odds",
    catalog_recovery_tight: float,
    catalog_recovery_tight_gate: float | None = None,
    n_matched_tight: int,
    n_det: int = 0,
    n_cat_in_frame: int = 0,
    quadrants_with_match: int = 0,
    expected_random: float | None = None,
    false_alarm_p: float | None = None,
    dist_benign: bool,
    centre_rms: float | None,
    edge_rms: float | None = None,
    recovery_min: float,
    matched_floor: int,
    centre_rms_max: float,
    hint_sep_deg: float,
    hint_sep_limit: float,
    fov_diameter_deg: float,
    odds_k: float = 12.0,
    odds_min_quadrants: int = 3,
    false_alarm_p_max: float = 1e-6,
    crowded_n_cat_min: int = 800,
) -> dict[str, Any]:
    """MASTERSTAR verified-solve gate: odds-based acceptance (default) or legacy fraction."""
    _centre_ok = False
    if centre_rms is not None and math.isfinite(float(centre_rms)):
        _centre_ok = float(centre_rms) <= float(centre_rms_max)
    _dist_ok = bool(dist_benign) or _centre_ok
    _mode = str(accept_mode or "odds").strip().lower()
    _gate_frac = (
        float(catalog_recovery_tight_gate)
        if catalog_recovery_tight_gate is not None
        else float(catalog_recovery_tight)
    )
    exp_r = float(expected_random) if expected_random is not None else 0.0
    p_false = float(false_alarm_p) if false_alarm_p is not None else 1.0
    if _mode == "odds":
        k_thr = max(float(matched_floor), float(odds_k) * max(0.0, exp_r))
        _verified = (
            int(n_matched_tight) >= int(math.ceil(k_thr))
            and int(quadrants_with_match) >= int(odds_min_quadrants)
            and p_false <= float(false_alarm_p_max)
        )
    else:
        _verified = (
            float(_gate_frac) >= float(recovery_min)
            and int(n_matched_tight) >= int(matched_floor)
            and bool(_dist_ok)
        )
    qmeta = _masterstar_quality_flags(
        catalog_recovery_tight_gate=float(_gate_frac),
        recovery_min=float(recovery_min),
        n_cat_in_frame=int(n_cat_in_frame),
        centre_rms=centre_rms,
        centre_rms_max=float(centre_rms_max),
        dist_benign=bool(dist_benign),
        crowded_n_cat_min=int(crowded_n_cat_min),
    )
    _fov_d = float(fov_diameter_deg)
    _tripwire = max(1.5, _fov_d) if math.isfinite(_fov_d) and _fov_d > 0.0 else 1.5
    _hint_sep_bad_hard = (
        (not _verified)
        and math.isfinite(float(hint_sep_deg))
        and float(hint_sep_deg) > float(_tripwire)
    )
    _hint_sep_warn = (
        _verified
        and math.isfinite(float(hint_sep_deg))
        and float(hint_sep_deg) > float(hint_sep_limit)
    )
    return {
        "masterstar_verified": bool(_verified),
        "accept_mode": _mode,
        "expected_random": float(exp_r),
        "false_alarm_p": float(p_false),
        "odds_match_threshold": float(max(float(matched_floor), float(odds_k) * max(0.0, exp_r))),
        "hint_sep_warn": bool(_hint_sep_warn),
        "hint_sep_bad_hard": bool(_hint_sep_bad_hard),
        "hint_sep_tripwire_deg": float(_tripwire),
        "distortion_ok": bool(_dist_ok),
        "quality_flag_primary": qmeta["quality_flag_primary"],
        "quality_flags": qmeta["quality_flags"],
    }


def _assess_masterstar_distortion_limited_linear(
    wcs: WCS,
    px: np.ndarray,
    py: np.ndarray,
    pra: np.ndarray,
    pde: np.ndarray,
    *,
    naxis1: int,
    naxis2: int,
    benign_ratio_max: float = 3.20,
) -> dict[str, Any]:
    """Overlay-style check: centre-tight / edge-drift residuals (benign Newton distortion)."""
    meta: dict[str, Any] = {"distortion_limited_benign": False}
    pxa = np.asarray(px, dtype=np.float64)
    pya = np.asarray(py, dtype=np.float64)
    ra_a = np.asarray(pra, dtype=np.float64)
    de_a = np.asarray(pde, dtype=np.float64)
    if len(pxa) < 20:
        meta["distortion_assess_skipped"] = "too_few_pairs"
        return meta
    try:
        xp, yp = wcs.all_world2pix(ra_a, de_a, 0)
        res = np.hypot(pxa - np.asarray(xp, dtype=np.float64), pya - np.asarray(yp, dtype=np.float64))
    except Exception as exc:  # noqa: BLE001
        meta["distortion_assess_error"] = repr(exc)
        return meta
    cx = 0.5 * float(naxis1)
    cy = 0.5 * float(naxis2)
    r_norm = np.hypot(pxa - cx, pya - cy) / max(1.0, 0.5 * min(float(naxis1), float(naxis2)))
    centre = r_norm < 0.35
    edge = r_norm > 0.65
    if not bool(np.any(centre)) or not bool(np.any(edge)):
        meta["distortion_assess_skipped"] = "insufficient_radial_bins"
        return meta
    centre_rms = float(np.sqrt(np.mean(np.square(res[centre]))))
    edge_rms = float(np.sqrt(np.mean(np.square(res[edge]))))
    overall_rms = float(np.sqrt(np.mean(np.square(res))))
    ratio = float(edge_rms / centre_rms) if centre_rms > 1e-6 else float("inf")
    meta.update(
        {
            "distortion_centre_rms_px": centre_rms,
            "distortion_edge_rms_px": edge_rms,
            "distortion_edge_centre_ratio": ratio,
            "distortion_overall_rms_px": overall_rms,
        }
    )
    benign = (
        math.isfinite(centre_rms)
        and math.isfinite(edge_rms)
        and centre_rms <= 1.60
        and edge_rms <= 3.20
        and ratio <= float(benign_ratio_max)
        and overall_rms <= 2.50
    )
    meta["distortion_limited_benign"] = bool(benign)
    meta["distortion_benign_ratio_max"] = float(benign_ratio_max)
    return meta


def _greedy_match_pairs_for_sip(
    wcs: WCS,
    ra_all: np.ndarray,
    de_all: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    max_px_coarse: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    max_px = _sip_match_max_px(max_px_coarse)
    px, py, pra, pde = _greedy_match_pairs_pixel_wcs(
        wcs,
        ra_all,
        de_all,
        xs,
        ys,
        max_px=max_px,
    )
    return (
        np.asarray(px, dtype=np.float64),
        np.asarray(py, dtype=np.float64),
        np.asarray(pra, dtype=np.float64),
        np.asarray(pde, dtype=np.float64),
        float(max_px),
    )


def _fit_linear_wcs_from_pairs(
    px: np.ndarray,
    py: np.ndarray,
    pra: np.ndarray,
    pde: np.ndarray,
    *,
    ransac_refinement: bool,
    ransac_min_pairs: int,
    rng_seed: int,
) -> WCS:
    world_m = SkyCoord(ra=np.asarray(pra, dtype=np.float64) * u.deg, dec=np.asarray(pde, dtype=np.float64) * u.deg, frame="icrs")
    if ransac_refinement and len(px) >= int(ransac_min_pairs):
        rng = np.random.default_rng(int(rng_seed))
        return _ransac_fit_wcs_tan(px, py, world_m, rng=rng)
    return fit_wcs_from_points((px, py), world_m, projection="TAN")


def _refit_linear_and_sip_on_full_pairs(
    w_lin: WCS,
    ra_all: np.ndarray,
    de_all: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    max_px_coarse: float,
    enable_sip: bool,
    sip_max_order: int,
    sip_min_order: int,
    is_masterstar: bool,
    sip_force_rms_guard_ratio: float | None,
    ransac_refinement: bool,
    ransac_min_pairs: int,
    rng_seed: int,
) -> tuple[WCS, list[float], list[float], list[float], list[float], dict[str, Any]]:
    """Greedy wide match → linear refit → optional SIP on **all** matched pairs (not tight subset)."""
    meta: dict[str, Any] = {}
    px, py, pra, pde, max_px_sip = _greedy_match_pairs_for_sip(
        w_lin, ra_all, de_all, xs, ys, max_px_coarse=max_px_coarse
    )
    meta["max_px_sip"] = float(max_px_sip)
    meta["n_pairs_sip_input"] = int(len(px))
    if len(px) < 5:
        meta["sip_skipped"] = "too_few_full_pairs"
        return (
            w_lin,
            np.asarray(px, dtype=np.float64).tolist(),
            np.asarray(py, dtype=np.float64).tolist(),
            np.asarray(pra, dtype=np.float64).tolist(),
            np.asarray(pde, dtype=np.float64).tolist(),
            meta,
        )
    w_lin2 = _fit_linear_wcs_from_pairs(
        px, py, pra, pde,
        ransac_refinement=ransac_refinement,
        ransac_min_pairs=ransac_min_pairs,
        rng_seed=rng_seed,
    )
    wcs_out: WCS = w_lin2
    world_m = SkyCoord(ra=pra * u.deg, dec=pde * u.deg, frame="icrs")
    if enable_sip and int(sip_max_order) >= 2:
        w_sip, sip_m = _fit_sip_for_solver(
            bool(is_masterstar),
            w_lin2,
            px,
            py,
            world_m,
            sip_max_order=int(sip_max_order),
            sip_min_order=int(sip_min_order),
            force_apply=bool(is_masterstar),
            sip_force_rms_guard_ratio=sip_force_rms_guard_ratio,
        )
        meta.update(sip_m)
        if w_sip is not None:
            wcs_out = w_sip
    return (
        wcs_out,
        px.tolist(),
        py.tolist(),
        pra.tolist(),
        pde.tolist(),
        meta,
    )


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
    # Keep a small set of best triangle candidates by ratio error. This reduces the chance that a
    # single lucky-but-wrong triangle dominates the solution (common in dense fields).
    best_k = 25
    candidates: list[tuple[float, tuple[int, int, int], tuple[int, int, int]]] = []

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
            if len(candidates) < best_k:
                candidates.append((err, trip_img, trip_cat))
            else:
                # replace worst
                iw = max(range(len(candidates)), key=lambda i: candidates[i][0])
                if err < candidates[iw][0]:
                    candidates[iw] = (err, trip_img, trip_cat)

    if not candidates:
        return None
    # Evaluate candidates: build TAN for each and pick best by match_rate (then RMS).
    candidates = sorted(candidates, key=lambda t: t[0])
    wcs_init_best: WCS | None = None
    best_rms = float("inf")
    best_rate = -1.0
    best_pairs: tuple[Any, Any, Any, Any] | None = None
    for _, trip_img, trip_cat in candidates:
        ii = list(trip_img)
        cc = list(trip_cat)
        wcs_init = None
        rms0 = float("inf")
        for perm in itertools.permutations((0, 1, 2), 3):
            px = np.array([xs[ii[0]], xs[ii[1]], xs[ii[2]]], dtype=np.float64)
            py = np.array([ys[ii[0]], ys[ii[1]], ys[ii[2]]], dtype=np.float64)
            ra_l = [
                float(cat_df.iloc[cc[perm[0]]]["ra_deg"]),
                float(cat_df.iloc[cc[perm[1]]]["ra_deg"]),
                float(cat_df.iloc[cc[perm[2]]]["ra_deg"]),
            ]
            de_l = [
                float(cat_df.iloc[cc[perm[0]]]["dec_deg"]),
                float(cat_df.iloc[cc[perm[1]]]["dec_deg"]),
                float(cat_df.iloc[cc[perm[2]]]["dec_deg"]),
            ]
            world = SkyCoord(ra=np.array(ra_l) * u.deg, dec=np.array(de_l) * u.deg, frame="icrs")
            try:
                w_try = fit_wcs_from_points((px, py), world, projection="TAN")
                pxp, pyp = w_try.all_world2pix(ra_l, de_l, 0)
                rms = float(np.sqrt(np.mean((pxp - px) ** 2 + (pyp - py) ** 2)))
            except Exception:  # noqa: BLE001
                # EXC-0600: T4 -- permutation trial fit fail -> continue (EXCEPT-BULK 2026-07-08)
                continue
            if rms < rms0:
                rms0 = rms
                wcs_init = w_try
        if wcs_init is None:
            continue

        # Coarse greedy match on full cat
        try:
            ra_all0 = cat_df["ra_deg"].to_numpy(dtype=np.float64)
            de_all0 = cat_df["dec_deg"].to_numpy(dtype=np.float64)
            max_px_coarse0 = max(18.0, min(42.0, 0.014 * float(math.hypot(float(w), float(h)))))
            px_m, py_m, pra_m, pde_m = _greedy_match_pairs_pixel_wcs(
                wcs_init,
                ra_all0,
                de_all0,
                xs,
                ys,
                max_px=max_px_coarse0,
            )
            rate0 = float(len(px_m)) / float(max(1, int(n_img)))
        except Exception:  # noqa: BLE001
            # EXC-0601: T4 -- candidate coarse-match-rate fail -> continue (EXCEPT-BULK 2026-07-08)
            continue
        if (rate0 > best_rate + 1e-9) or (abs(rate0 - best_rate) < 1e-9 and rms0 < best_rms):
            best_rate = float(rate0)
            best_rms = float(rms0)
            wcs_init_best = wcs_init
            best_pairs = (px_m, py_m, pra_m, pde_m)

    if wcs_init_best is None or best_pairs is None:
        return None
    wcs_init = wcs_init_best
    pairs_x, pairs_y, pairs_ra, pairs_de = best_pairs

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
        # EXC-0602: T4 -- catalog crop fail -> full catalog kept (fail-safe, slower) (EXCEPT-BULK 2026-07-08)
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
    # Re-match on cropped catalog if it was reduced above.
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
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("[SOLVER] linear-WCS (SIP strip) failed: %s", exc)
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
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("[SOLVER] CD-axis equalize failed: %s", exc)
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
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("[SOLVER] CD-axis equalize failed: %s", exc)
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
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("[SOLVER] anisotropic plate-scale repair failed: %s", exc)
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


class _SolveWcsCatalogError(Exception):
    """Raised when Gaia catalog build fails; carries the solver failure dict."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.result: dict[str, Any] = {"solved": False, "reason": reason}


def _solve_wcs_build_catalog(
    pointing_ra: float,
    pointing_dec: float,
    fov_diameter_deg_eff: float,
    exp_scale: float | None,
    chip_fw: int,
    chip_fh: int,
    gaia_db_path: Path | str,
    eff_max_cat_mag: float,
    obs_epoch: float,
    logger: Any | None,
    *,
    hdr0: fits.Header,
    fov_diameter_deg: float,
    pixel_pitch_um: float,
    focal_length_mm: float | None,
    scale_arcsec: float | None,
    optimal_params: dict[str, Any],
    max_catalog_rows: int | None,
    max_cat_mag: float,
    faintest_mag_limit: float | None,
    coord_src: str,
    exp_scale_from_expected_arg: bool,
    app_config: Any | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, SkyCoord, float, pd.DataFrame, float]:
    """Query Gaia catalog and build matched star lists for plate solving.

    Returns:
        (cat_df, cat_df_tri, c_cat, cone_r, cat_df_cone_full, eff_max_cat_mag)
    """
    from database import query_local_gaia

    _cfg = app_config or AppConfig()
    _ = logger
    _ = eff_max_cat_mag
    ra0 = float(pointing_ra)
    de0 = float(pointing_dec)
    naxis1 = int(chip_fw)
    naxis2 = int(chip_fh)
    root = Path(gaia_db_path)
    _exp_scale = exp_scale
    _f_um = float(pixel_pitch_um)
    _ep_um = (
        float(pixel_pitch_um)
        if math.isfinite(float(pixel_pitch_um)) and float(pixel_pitch_um) > 0
        else None
    )
    _f_mm_u = float(focal_length_mm) if focal_length_mm is not None else 0.0
    _foc_mm = focal_length_mm
    _scale_arcsec = scale_arcsec
    _opt = optimal_params
    _coord_src = coord_src
    _exp_scale_from_expected_arg = bool(exp_scale_from_expected_arg)
    _obs_year = float(obs_epoch)

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
        # EXC-0603: T2 -- optics-based cone floor fail -> pass -> possibly undersized Gaia cone (solve failure is... (EXCEPT-BULK 2026-07-08)
        pass
    # Global minimum cone radius is only needed when optics are unknown; for narrow-field optics it is harmful
    # (explodes Gaia rows and makes triangle matching intractable).
    _min_cone = float(MIN_GAIA_CONE_RADIUS_DEG)
    try:
        if (
            _scale_arcsec is not None
            and math.isfinite(float(_scale_arcsec))
            and float(_scale_arcsec) > 0
            and _foc_mm is not None
            and math.isfinite(float(_foc_mm))
            and float(_foc_mm) > 0
            and _ep_um is not None
            and math.isfinite(float(_ep_um))
            and float(_ep_um) > 0
        ):
            _min_cone = 0.08
    except Exception:  # noqa: BLE001
        _min_cone = float(MIN_GAIA_CONE_RADIUS_DEG)
    cone_r = max(float(cone_r), float(_min_cone))
    required_corners_radius = catalog_cone_radius_deg_from_optics(
        naxis1=naxis1,
        naxis2=naxis2,
        pixel_pitch_um=_f_um,
        focal_length_mm=_f_mm_u,
        margin=0.85,
        fov_diameter_fallback_deg=float(fov_diameter_deg),
    )
    cone_r = max(float(cone_r), float(required_corners_radius))
    # FOV fallback from caller/config can be wildly wrong (e.g. default 7°) and must not override
    # optics-derived cone when we have a plausible focal length + pixel pitch.
    _r_fov = catalog_cone_radius_from_fov_diameter_deg(float(fov_diameter_deg))
    _has_optics = (
        _scale_arcsec is not None
        and math.isfinite(float(_scale_arcsec))
        and float(_scale_arcsec) > 0
        and _foc_mm is not None
        and math.isfinite(float(_foc_mm))
        and float(_foc_mm) > 0
        and _ep_um is not None
        and math.isfinite(float(_ep_um))
        and float(_ep_um) > 0
    )
    if (not bool(_has_optics)) and _r_fov > 0:
        cone_r = max(float(cone_r), float(_r_fov))
    try:
        cone_r = max(float(cone_r), float(_opt.get("search_radius", 0.0)))
    except (TypeError, ValueError) as exc:  # noqa: BLE001
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
    _focal_for_mag: float | None = None
    if _foc_mm is not None:
        try:
            _fv = float(_foc_mm)
            if math.isfinite(_fv) and _fv > 0:
                _focal_for_mag = float(_fv)
        except (TypeError, ValueError):
            _focal_for_mag = None

    # Dynamický mag cap (per-frame aj MASTERSTAR): malé FOV + dlhé ohnisko potrebuje hlbší katalóg.
    if _focal_for_mag is not None:
        focal = float(_focal_for_mag)
        if focal >= 800:
            _mag_cap = min(float(max_cat_mag), 15.8)
        elif focal >= 400:
            _mag_cap = min(float(max_cat_mag), 14.5)
        else:
            _mag_cap = min(float(max_cat_mag), 13.0)
    else:
        # Neznáme ohnisko → konzervatívne podľa FOV plochy
        if _fov_area_deg2 < 2.0:
            _mag_cap = min(float(max_cat_mag), 15.0)
        elif _fov_area_deg2 < 10.0:
            _mag_cap = min(float(max_cat_mag), 13.5)
        else:
            _mag_cap = min(float(max_cat_mag), 12.0)

    log_event(
        f"INFO: Dynamický mag_cap={float(_mag_cap):.1f} "
        f"(FOV={_fov_area_deg2:.3f} deg², focal={_foc_mm}mm)"
    )
    _hint_is_blind_cone = "blind solver" in str(_coord_src or "").strip().lower()
    # If expected scale came from DB/config (expected_plate_scale_arcsec_per_px) and hint is blind,
    # avoid clipping the Gaia cone to an (often wrong) FOV derived from that scale.
    _allow_cone_clip_to_fov = not (bool(_hint_is_blind_cone) and bool(_exp_scale_from_expected_arg))

    if _fov_area_deg2 < 10.0 and bool(_allow_cone_clip_to_fov):
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
    # (but skip this when blind hint + expected scale from DB/config).
    if bool(_allow_cone_clip_to_fov) and _exp_scale is not None and naxis1 is not None and naxis2 is not None:
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
    elif (not bool(_allow_cone_clip_to_fov)) and _exp_scale is not None:
        log_event(
            "VYVAR platesolve: vynechávam cone_r clip na FOV+20% (blind hint + expected scale z DB/config môže byť nesprávna)."
        )
    rows_g = query_local_gaia(
        root,
        ra_min=ra_min,
        ra_max=ra_max,
        dec_min=de_min,
        dec_max=de_max,
        mag_limit=_mag_cap,
        max_rows=int(max_catalog_rows) if max_catalog_rows is not None else None,
    )
    rows_g, _pm_corr_count = _apply_pm_to_gaia_rows(rows_g, obs_year=float(_obs_year))
    if _pm_corr_count > 0:
        log_event(
            f"VYVAR platesolve: proper motion correction applied to {_pm_corr_count} Gaia stars (epoch {GAIA_EPOCH:.1f} -> {_obs_year:.2f})."
        )
    if not rows_g:
        raise _SolveWcsCatalogError("VYVAR solver: Gaia query v okolí hintu vrátil 0 hviezd.")
    cat_df = pd.DataFrame(rows_g)
    # Normalize to the solver's expected catalog schema.
    cat_df = cat_df.rename(columns={"source_id": "catalog_id", "ra": "ra_deg", "dec": "dec_deg", "g_mag": "mag"})
    cat_df["catalog"] = "GAIA_DR3"
    # Color index as a stand-in (optional)
    if "bp_rp" not in cat_df.columns:
        cat_df["bp_rp"] = None
    # Gaia provides BP-RP; keep B-V separate (do not overwrite with BP-RP).
    if "b_v" not in cat_df.columns:
        cat_df["b_v"] = float("nan")
    cat_df["bp_rp"] = pd.to_numeric(cat_df.get("bp_rp"), errors="coerce")
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
        raise _SolveWcsCatalogError(f"VYVAR solver: v Gaia výreze málo hviezd ({len(cat_df)}).")

    cat_df_cone_full = cat_df.sort_values("mag", na_position="last").reset_index(drop=True)
    # Brighter catalog subset for triangle matching to reduce ambiguity.
    cat_df_tri = cat_df_cone_full
    try:
        tri_cap_default = 13.5 if (_foc_mm is not None and math.isfinite(float(_foc_mm)) and float(_foc_mm) >= 800.0) else 12.5
    except Exception:  # noqa: BLE001
        tri_cap_default = 13.5
    try:
        tri_cap = float(getattr(_cfg, "platesolve_triangle_mag_cap", tri_cap_default))
    except (TypeError, ValueError):
        tri_cap = float(tri_cap_default)
    if not math.isfinite(tri_cap) or tri_cap <= 0:
        tri_cap = float(tri_cap_default)
    try:
        if "mag" in cat_df_cone_full.columns:
            mm = pd.to_numeric(cat_df_cone_full["mag"], errors="coerce")
            cand = cat_df_cone_full[(mm.notna()) & (mm <= float(tri_cap))].copy()
            if len(cand) >= 24:
                cat_df_tri = cand.sort_values("mag", na_position="last").reset_index(drop=True)
                log_event(
                    f"VYVAR platesolve: triangle catalog mag_cap={float(tri_cap):.1f} → {len(cat_df_tri)} hviezd (z {_n_cat_raw})."
                )
    except Exception:  # noqa: BLE001
        cat_df_tri = cat_df_cone_full
    try:
        _scale_arcsec = plate_scale_arcsec_per_pixel(pixel_pitch_um=float(_f_um), focal_length_mm=float(_f_mm_u))
    except Exception:  # noqa: BLE001
        _scale_arcsec = None
    if (
        _scale_arcsec is None
        or (not math.isfinite(float(_scale_arcsec)))
        or float(_scale_arcsec) <= 0.0
    ):
        if _exp_scale is not None and math.isfinite(float(_exp_scale)) and float(_exp_scale) > 0.0:
            _scale_arcsec = float(_exp_scale)
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
        raise _SolveWcsCatalogError("VYVAR solver: po zoradení podľa mag je v kuželi príliš málo hviezd.")

    return cat_df, cat_df_tri, c_cat, float(cone_r), cat_df_cone_full, float(eff_max_cat_mag)


class _SolveWcsValidationError(Exception):
    def __init__(self, result: dict) -> None:
        super().__init__(str(result.get("reason", "validation failed")))
        self.result = result


def _solve_wcs_validate_and_refine(
    wcs_final: WCS,
    pairs_final: list[float],
    cat_df: pd.DataFrame,
    cat_df_assoc: pd.DataFrame,
    xs_native: np.ndarray,
    ys_native: np.ndarray,
    hdr0: fits.Header,
    fp: Path,
    is_masterstar: bool,
    hint_ra: float,
    hint_dec: float,
    sip_meta: dict[str, Any],
    gaia_db_path: str,
    logger: Any | None,
    *,
    pairs_y: list[float],
    pairs_ra: list[float],
    pairs_de: list[float],
    n_img: int,
    naxis1: int,
    naxis2: int,
    fov_diameter_deg: float,
    coord_src: str | None,
    best_rms: float,
    dao_fw: float,
    tbl_sorted: Any,
    cat_df_cone_full: pd.DataFrame,
    max_px_coarse: float,
    w: int,
    h: int,
    enable_sip: bool,
    sip_max_order: int,
    ransac_min_pairs: int,
    sip_min_order: int,
    sip_force_rms_guard_ratio: float | None,
    masterstar_prewrite_rms_max_px: float | None,
    masterstar_prewrite_relaxed_rms_max_px: float | None,
    masterstar_nn_refine_max_rms_px: float | None,
    fits_header_hint_sep_escape: bool = True,
    app_config: Any | None = None,
) -> tuple[
    WCS,
    fits.Header,
    list[float],
    list[float],
    list[float],
    list[float],
    dict[str, Any],
    float,
    float | None,
    float,
]:
    """Validate WCS solution quality and apply NN refinement if needed.

    Returns:
        (wcs_final, hdr0, pairs_x, pairs_y, pairs_ra, pairs_de, sip_meta, match_rate, rms_px, dao_fw)
    Or raises _SolveWcsValidationError on invalid solution.
    """
    _ = logger
    _ = cat_df
    _ = gaia_db_path
    _ = xs_native
    _ = ys_native
    pairs_x = pairs_final
    _is_masterstar = bool(is_masterstar)
    ra0 = float(hint_ra)
    de0 = float(hint_dec)
    _coord_src = coord_src
    _dao_fw = float(dao_fw)
    _sip_min_ms = int(sip_min_order)
    _ms_sip_guard_r = sip_force_rms_guard_ratio
    xs = xs_native
    ys = ys_native

    # QA rematch on the same detections used for solving (``n_img`` brightest), not ``len(tbl_sorted)`` (can be 5k+).
    # Use ``cat_df_assoc`` (deep cone), not the triangle-probe crop in ``ra_all``/``de_all``.
    _ra_cat = cat_df_assoc["ra_deg"].to_numpy(dtype=np.float64)
    _de_cat = cat_df_assoc["dec_deg"].to_numpy(dtype=np.float64)
    _mtq = float(sip_meta.get("max_px_sip", 0.0) or 0.0)
    if not (math.isfinite(_mtq) and _mtq > 0):
        _mtq = float(max_px_coarse)
    _qa_px = max(15.0, min(48.0, float(_mtq) * 1.22))
    try:
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
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("[SOLVER] QA rematch pair update skipped: %s", exc)
        pass

    # Pre-write validation: reject weak/shifted solutions and retry with simpler TAN model.
    _n_det_total = max(1, int(n_img))
    _matched_n = int(len(pairs_x))
    _match_rate = float(_matched_n) / float(_n_det_total)
    sip_meta["match_rate_n_used"] = int(n_img)
    sip_meta["match_rate_n_matched"] = int(_matched_n)
    # Brightest-N scope (same as final user-facing match%) for validation gates.
    _nrate_qa = min(200, int(n_img))
    _match_rate_bright = _match_rate
    if _nrate_qa >= 6 and int(n_img) > _nrate_qa:
        try:
            _bxq = np.asarray(xs, dtype=np.float64)[: int(_nrate_qa)]
            _byq = np.asarray(ys, dtype=np.float64)[: int(_nrate_qa)]
            _bqx, _bqy, _, _ = _greedy_match_pairs_pixel_wcs(
                wcs_final,
                _ra_cat,
                _de_cat,
                _bxq,
                _byq,
                max_px=float(_qa_px),
            )
            _match_rate_bright = float(len(_bqx)) / float(int(_nrate_qa))
            sip_meta["match_rate_bright_n"] = int(_nrate_qa)
            sip_meta["match_rate_bright"] = float(_match_rate_bright)
            sip_meta["match_rate_bright_matched"] = int(len(_bqx))
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("[SOLVER] QA brightest-N rematch skipped: %s", exc)
    _match_rate_gate = float(_match_rate_bright if _is_masterstar else _match_rate)
    _cfg_val = app_config or AppConfig()
    _benign_ratio_max = 3.20
    _recovery_min = 0.65
    _matched_floor = 40
    _centre_rms_max = 1.20
    if _is_masterstar:
        try:
            _benign_ratio_max = float(
                getattr(_cfg_val, "masterstar_distortion_benign_ratio_max", 3.20)
            )
        except (TypeError, ValueError):
            _benign_ratio_max = 3.20
        _benign_ratio_max = max(2.0, min(5.0, _benign_ratio_max))
        try:
            _recovery_min = float(getattr(_cfg_val, "masterstar_catalog_recovery_min", 0.65))
        except (TypeError, ValueError):
            _recovery_min = 0.65
        _recovery_min = max(0.40, min(0.95, _recovery_min))
        try:
            _matched_floor = int(getattr(_cfg_val, "masterstar_min_matched_floor", 40))
        except (TypeError, ValueError):
            _matched_floor = 40
        _matched_floor = max(1, min(500, _matched_floor))
        try:
            _centre_rms_max = float(getattr(_cfg_val, "masterstar_centre_rms_max_px", 1.20))
        except (TypeError, ValueError):
            _centre_rms_max = 1.20
        _centre_rms_max = max(0.5, min(5.0, _centre_rms_max))
    _dist_assess: dict[str, Any] = {}
    if _is_masterstar and int(_matched_n) >= 20:
        try:
            _dist_assess = _assess_masterstar_distortion_limited_linear(
                wcs_final,
                np.asarray(pairs_x, dtype=np.float64),
                np.asarray(pairs_y, dtype=np.float64),
                np.asarray(pairs_ra, dtype=np.float64),
                np.asarray(pairs_de, dtype=np.float64),
                naxis1=int(naxis1),
                naxis2=int(naxis2),
                benign_ratio_max=float(_benign_ratio_max),
            )
            sip_meta.update(_dist_assess)
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("[SOLVER] distortion-limited assess skipped: %s", exc)
    _dist_benign = bool(_dist_assess.get("distortion_limited_benign", False))
    _catalog_recovery: dict[str, Any] = {}
    if _is_masterstar:
        try:
            _catalog_recovery = _compute_masterstar_catalog_recovery(
                wcs_final,
                _ra_cat,
                _de_cat,
                np.asarray(xs, dtype=np.float64),
                np.asarray(ys, dtype=np.float64),
                naxis1=int(naxis1),
                naxis2=int(naxis2),
                qa_px=float(_qa_px),
                tight_px=2.5,
            )
            sip_meta.update(_catalog_recovery)
            sip_meta["catalog_recovery_verification"] = True
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("[SOLVER] catalog-recovery assess skipped: %s", exc)
    _rms_px = None
    _rms_keys = (
        ("wcs_refine_rms_px", "rms_sip_px", "rms_linear_px")
        if bool(sip_meta.get("sip_applied", False))
        else ("wcs_refine_rms_px", "rms_linear_px", "rms_sip_px")
    )
    for _k in _rms_keys:
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
    _qa_log = (
        f"VYVAR platesolve QA: match_rate={_match_rate * 100.0:.1f}% "
        f"gate={_match_rate_gate * 100.0:.1f}% (info) "
        f"rms={float(_rms_px):.2f}px hint_vs_solved={_hint_sep_deg:.3f}deg "
        f"distortion_benign={bool(_dist_benign)} "
        f"centre/edge_rms="
        f"{_dist_assess.get('distortion_centre_rms_px', 'n/a')}/"
        f"{_dist_assess.get('distortion_edge_rms_px', 'n/a')}px"
    )
    if _is_masterstar and _catalog_recovery:
        _qa_log += (
            f" catalog_recovery_tight={float(_catalog_recovery.get('catalog_recovery_tight', 0.0)) * 100.0:.1f}%"
            f" gate={float(_catalog_recovery.get('catalog_recovery_tight_gate', 0.0)) * 100.0:.1f}%"
            f" coarse={float(_catalog_recovery.get('catalog_recovery_coarse', 0.0)) * 100.0:.1f}%"
            f" n_cat_in_frame={int(_catalog_recovery.get('n_cat_in_frame', 0))}"
            f" n_det={int(_catalog_recovery.get('n_detections_used', 0))}"
        )
    log_event(_qa_log)

    # Hint separation guard: adapt to hint source + field size.
    # - Strong hint (mount/object header): keep strict.
    # - Weak hint (blind solver): allow larger separation, but only when match metrics are strong.
    try:
        _fov_d = float(fov_diameter_deg)
        _fov_r = 0.5 * _fov_d if math.isfinite(_fov_d) and _fov_d > 0 else None
    except (TypeError, ValueError):
        _fov_r = None
    _coord_src_l = str(_coord_src or "").strip().lower()
    _hint_is_weak = "blind solver" in _coord_src_l
    # Base limits (deg)
    _base_strict = 0.15 if _is_masterstar else 0.50
    _base_relaxed = 0.50 if _is_masterstar else 1.00
    if _fov_r is not None:
        # Allow up to ~35% of FOV radius for weak hints, but cap to prevent wrong-field solves.
        _rel_by_fov = max(_base_relaxed, min(1.50, 0.35 * float(_fov_r)))
        _strict_by_fov = max(_base_strict, min(0.50, 0.20 * float(_fov_r)))
    else:
        _rel_by_fov = _base_relaxed
        _strict_by_fov = _base_strict
    hint_sep_limit = float(_rel_by_fov if _hint_is_weak else _strict_by_fov)
    # Anti-false-solve: only relax hint guard when match is reasonably strong (non-MASTERSTAR).
    _relax_ok = (float(_match_rate) >= (0.20 if _is_masterstar else 0.10)) and (int(_matched_n) >= (20 if _is_masterstar else 12))
    if _hint_is_weak and not _relax_ok:
        hint_sep_limit = float(_base_strict)
    _ = fits_header_hint_sep_escape  # legacy API; MASTERSTAR escape blocks removed (TASK 2)
    _verified = False
    _hint_sep_warn = False
    _hint_sep_bad_hard = False
    _hint_sep_bad = False
    if _is_masterstar:
        _centre_rms_val = _dist_assess.get("distortion_centre_rms_px")
        try:
            _centre_rms_f = float(_centre_rms_val) if _centre_rms_val is not None else None
            if _centre_rms_f is not None and not math.isfinite(_centre_rms_f):
                _centre_rms_f = None
        except (TypeError, ValueError):
            _centre_rms_f = None
        _accept = _masterstar_solve_acceptance(
            accept_mode=str(getattr(_cfg_val, "masterstar_accept_mode", "odds")),
            catalog_recovery_tight=float(
                _catalog_recovery.get("catalog_recovery_tight", 0.0)
            ),
            catalog_recovery_tight_gate=float(
                _catalog_recovery.get(
                    "catalog_recovery_tight_gate",
                    _catalog_recovery.get("catalog_recovery_tight", 0.0),
                )
            ),
            n_matched_tight=int(_catalog_recovery.get("n_matched_tight", 0)),
            n_det=int(_catalog_recovery.get("n_detections_used", 0)),
            n_cat_in_frame=int(_catalog_recovery.get("n_cat_in_frame", 0)),
            quadrants_with_match=int(_catalog_recovery.get("quadrants_with_match", 0)),
            expected_random=_catalog_recovery.get("expected_random"),
            false_alarm_p=_catalog_recovery.get("false_alarm_p"),
            dist_benign=_dist_benign,
            centre_rms=_centre_rms_f,
            edge_rms=_dist_assess.get("distortion_edge_rms_px"),
            recovery_min=float(_recovery_min),
            matched_floor=int(
                _MASTERSTAR_ODDS_MATCH_FLOOR
                if str(getattr(_cfg_val, "masterstar_accept_mode", "odds")).strip().lower() == "odds"
                else _matched_floor
            ),
            centre_rms_max=float(_centre_rms_max),
            hint_sep_deg=float(_hint_sep_deg),
            hint_sep_limit=float(hint_sep_limit),
            fov_diameter_deg=float(fov_diameter_deg),
            odds_k=float(_MASTERSTAR_ODDS_K),
            odds_min_quadrants=int(_MASTERSTAR_ODDS_MIN_QUADRANTS),
            false_alarm_p_max=float(_MASTERSTAR_FALSE_ALARM_P_MAX),
            crowded_n_cat_min=int(getattr(_cfg_val, "masterstar_quality_crowded_n_cat_min", 800)),
        )
        _verified = bool(_accept.get("masterstar_verified", False))
        _hint_sep_warn = bool(_accept.get("hint_sep_warn", False))
        _hint_sep_bad_hard = bool(_accept.get("hint_sep_bad_hard", False))
        sip_meta["masterstar_verified"] = bool(_verified)
        sip_meta["masterstar_accept_mode"] = str(_accept.get("accept_mode", "odds"))
        sip_meta["expected_random"] = _accept.get("expected_random")
        sip_meta["false_alarm_p"] = _accept.get("false_alarm_p")
        sip_meta["odds_match_threshold"] = _accept.get("odds_match_threshold")
        sip_meta["quadrants_with_match"] = int(_catalog_recovery.get("quadrants_with_match", 0))
        sip_meta["quality_flag_primary"] = _accept.get("quality_flag_primary", "ok")
        sip_meta["quality_flags"] = _accept.get("quality_flags", [])
        sip_meta["masterstar_catalog_recovery_min"] = float(_recovery_min)
        sip_meta["masterstar_min_matched_floor"] = int(_matched_floor)
        sip_meta["masterstar_centre_rms_max_px"] = float(_centre_rms_max)
        if _is_masterstar and _accept.get("accept_mode") == "odds":
            log_event(
                "VYVAR MASTERSTAR odds: "
                f"n_tight={int(_catalog_recovery.get('n_matched_tight', 0))} "
                f"expected_random={float(_accept.get('expected_random', 0.0)):.2f} "
                f"p_false={float(_accept.get('false_alarm_p', 1.0)):.2e} "
                f"quads={int(_catalog_recovery.get('quadrants_with_match', 0))} "
                f"verified={bool(_verified)} "
                f"qflag={_accept.get('quality_flag_primary', 'ok')}"
            )
        if _hint_sep_warn:
            sip_meta["hint_sep_warn"] = True
            sip_meta["hint_sep_deg"] = float(_hint_sep_deg)
            log_event(
                f"VYVAR MASTERSTAR: hint_sep warning (non-fatal): "
                f"{float(_hint_sep_deg):.3f}deg > limit {float(hint_sep_limit):.2f}deg "
                f"(verified catalog_recovery_gate={float(_catalog_recovery.get('catalog_recovery_tight_gate', _catalog_recovery.get('catalog_recovery_tight', 0.0))) * 100.0:.1f}%)"
            )
        if _hint_sep_bad_hard:
            log_event(
                f"VYVAR MASTERSTAR: hint_sep hard tripwire: "
                f"{float(_hint_sep_deg):.3f}deg > {float(_accept.get('hint_sep_tripwire_deg', 1.5)):.2f}deg "
                f"(not verified)"
            )
    else:
        _hint_sep_bad = math.isfinite(float(_hint_sep_deg)) and float(_hint_sep_deg) > float(hint_sep_limit)
    log_event(
        f"INFO: hint_sep guard: {float(_hint_sep_deg):.3f}deg "
        f"(limit={float(hint_sep_limit):.2f}deg, is_masterstar={bool(_is_masterstar)}, "
        f"hint_src={_coord_src or 'unknown'}, relax_ok={bool(_relax_ok)})"
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
    if _is_masterstar:
        _invalid = (
            (not _verified)
            or (not math.isfinite(float(_rms_px)))
            or _rms_bad
            or _hint_sep_bad_hard
        )
    else:
        _invalid = (_match_rate < 0.02) or (not math.isfinite(float(_rms_px))) or _rms_bad or bool(_hint_sep_bad)
    if _invalid:
        _reason_extra = ""
        if _is_masterstar:
            _reason_extra = (
                f", verified={bool(_verified)}, "
                f"catalog_recovery_tight={float(_catalog_recovery.get('catalog_recovery_tight', 0.0)) * 100.0:.1f}%, "
                f"catalog_recovery_gate={float(_catalog_recovery.get('catalog_recovery_tight_gate', 0.0)) * 100.0:.1f}%, "
                f"n_matched_tight={int(_catalog_recovery.get('n_matched_tight', 0))}"
            )
        raise _SolveWcsValidationError(
            {
                "solved": False,
                "match_rate": float(_match_rate),
                "rms_px": float(_rms_px),
                "reason": (
                    f"VYVAR solver: invalid solution (match_rate={_match_rate * 100.0:.1f}%, "
                    f"rms={float(_rms_px):.2f}px, hint_sep={_hint_sep_deg:.3f}deg{_reason_extra})."
                ),
            }
        )

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
                except Exception as exc:  # noqa: BLE001
                    LOGGER.debug("[SOLVER] SIP NN-refine apply skipped: %s", exc)
                    pass
            if pxa_r is not None and pya_r is not None and world_r is not None:
                pairs_x = np.asarray(pxa_r, dtype=np.float64).tolist()
                pairs_y = np.asarray(pya_r, dtype=np.float64).tolist()
                pairs_ra = np.asarray(world_r.ra.deg, dtype=np.float64).tolist()
                pairs_de = np.asarray(world_r.dec.deg, dtype=np.float64).tolist()
            else:
                log_event(
                    "WARNING: NN-refine accepted but pairs/world missing — skipping refine-pairs export."
                )

    # Po NN/SIP ešte raz zrátaj páry na rovnakom súbore detekcií (``n_img``) oproti asociačnému Gaia výrezu.
    try:
        _ra_cat2 = cat_df_assoc["ra_deg"].to_numpy(dtype=np.float64)
        _de_cat2 = cat_df_assoc["dec_deg"].to_numpy(dtype=np.float64)
        _mtq3 = float(sip_meta.get("max_px_sip", 0.0) or 0.0)
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
            f"{sip_meta.get('match_rate_scope')!s}) | all-frame≈{float(sip_meta.get('match_rate_full_frame', 0.0)) * 100.0:.1f}%"
        )
    except Exception as exc:  # noqa: BLE001
        # EXC-0605 / EXCEPT-FIX-3 #10: surface the final match-rate QA computation failure and
        # write explicit sentinels so downstream solve QA sees "error" instead of a silently
        # missing match_rate_final key. See docs/VYVAR_EXCEPT_CENSUS.md (EXC-0605).
        from except_fix_counters import get_except_fix_counters

        get_except_fix_counters().platesolve_match_rate_meta_fail += 1
        LOGGER.error("[PLATE-SOLVE] final match-rate meta computation failed: %s", exc)
        sip_meta["match_rate_final"] = float("nan")
        sip_meta["match_rate_scope"] = "error"

    return (
        wcs_final,
        hdr0,
        pairs_x,
        pairs_y,
        pairs_ra,
        pairs_de,
        sip_meta,
        float(_match_rate),
        _rms_px,
        float(_dao_fw),
    )


class _SolveWcsWriteError(Exception):
    """Raised when FITS write / header sync fails; carries the solver failure dict."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.result: dict[str, Any] = {"solved": False, "reason": reason}


def _solve_wcs_write_results(
    fp: Path,
    hdr0: fits.Header,
    wcs_final: WCS,
    sip_meta: dict[str, Any],
    pairs_final: list[float],
    match_rate: float,
    rms_px: float,
    dao_fw: float,
    platescale_arcsec_per_px: float | None,
    is_masterstar: bool,
    logger: Any | None,
    *,
    cone_r: float,
    ep_um: float | None,
    n_img: int,
) -> None:
    """Write solved WCS + VYVAR metadata to FITS file on disk."""
    _ = logger
    _ = platescale_arcsec_per_px
    _ = rms_px
    pairs_x = pairs_final
    _match_rate = float(match_rate)
    _dao_fw = float(dao_fw)
    _is_masterstar = bool(is_masterstar)
    _ep_um = ep_um

    try:
        wh = wcs_final.to_header(relax=True)
    except Exception as exc:  # noqa: BLE001
        raise _SolveWcsWriteError(f"VYVAR solver: WCS header: {exc}") from exc
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
        if sip_meta.get("catalog_recovery_tight") is not None:
            _crt_gate = sip_meta.get("catalog_recovery_tight_gate")
            _crt_pct = (
                float(_crt_gate) * 100.0
                if _crt_gate is not None
                else float(sip_meta.get("catalog_recovery_tight", 0.0)) * 100.0
            )
            hdr0["VY_CRT"] = (
                _crt_pct,
                "VYVAR: catalog recovery gate [%] (QA flag; not accept gate under odds mode)",
            )
        if sip_meta.get("quality_flag_primary") is not None:
            hdr0["VY_QFLAG"] = (
                str(sip_meta.get("quality_flag_primary", "ok")),
                "VYVAR: field quality flag (ok|crowded|blurred|distorted|low_recovery)",
            )
        if sip_meta.get("catalog_recovery_tight_gate") is not None:
            hdr0["VY_QFRAC"] = (
                float(sip_meta.get("catalog_recovery_tight_gate", 0.0)) * 100.0,
                "VYVAR: catalog recovery gate [%] (quality metadata)",
            )
        _qrms = sip_meta.get("distortion_centre_rms_px")
        if _qrms is None:
            _qrms = sip_meta.get("wcs_refine_rms_px", sip_meta.get("rms_linear_px"))
        if _qrms is not None:
            try:
                hdr0["VY_QRMS"] = (
                    float(_qrms),
                    "VYVAR: centre/residual RMS [px] (quality metadata)",
                )
            except (TypeError, ValueError):
                pass
        if sip_meta.get("n_cat_in_frame") is not None:
            hdr0["VY_QCRWD"] = (
                int(sip_meta.get("n_cat_in_frame", 0)),
                "VYVAR: Gaia-in-frame count (crowding proxy)",
            )
            hdr0["VY_CNF"] = (
                int(sip_meta.get("n_cat_in_frame", 0)),
                "VYVAR: Gaia catalog stars predicted in frame",
            )
        if bool(sip_meta.get("hint_sep_warn", False)):
            hdr0["VY_HSWN"] = (
                True,
                "VYVAR: hint separation warning (solve verified; stale pointing hint)",
            )
            if sip_meta.get("hint_sep_deg") is not None:
                hdr0["VY_HSEP"] = (
                    float(sip_meta.get("hint_sep_deg", 0.0)),
                    "VYVAR: hint vs solved separation [deg]",
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
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("[SOLVER] VY_PLTS plate-scale header write skipped: %s", exc)
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
        # EXC-0610: T3 -- VY_MIRR header write skipped (EXCEPT-BULK 2026-07-08)
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
            strip_vendor_platesolve_metadata(h_w)
            hdul_w.flush()
            _hdr_written = h_w.copy()
            _data_written = np.asarray(hdul_w[0].data, dtype=np.float32)
    except Exception as exc:  # noqa: BLE001
        _hdr_written = None
        _data_written = None
        if _is_masterstar:
            LOGGER.error(
                "[SOLVER] MASTERSTAR WCS persist failed (read/update) for %s: %s",
                fp,
                exc,
            )
            raise _SolveWcsWriteError(
                f"VYVAR solver: MASTERSTAR WCS persist failed (read/update): {fp}: {exc}"
            ) from exc

    # 2) For MASTERSTAR on disk: hard overwrite to guarantee file is physically updated.
    if _is_masterstar:
        if _hdr_written is None or _data_written is None:
            LOGGER.error("[SOLVER] MASTERSTAR WCS persist failed: missing header/data for %s", fp)
            raise _SolveWcsWriteError(
                f"VYVAR solver: MASTERSTAR WCS persist failed (missing header/data): {fp}"
            )
        try:
            fits.writeto(fp, _data_written, _hdr_written, overwrite=True)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "[SOLVER] failed to persist solved MASTERSTAR WCS to %s: %s",
                fp,
                exc,
            )
            raise _SolveWcsWriteError(
                f"VYVAR solver: MASTERSTAR WCS persist failed (writeto): {fp}: {exc}"
            ) from exc

    if not fits_header_has_celestial_wcs(_hdr_written or hdr0):
        raise _SolveWcsWriteError("VYVAR solver: WCS po zápise stále neplatný.")



def _try_blind_series_hint(
    data: np.ndarray,
    hdr0: fits.Header,
    *,
    plate_scale_arcsec_per_px: float | None,
    fov_deg: float | None,
    max_cat_mag: float,
    app_config: Any,
) -> tuple[float, float, str] | None:
    """Run scale-constrained blind triangle solver; return (ra, dec, tier) or None."""
    from astropy.stats import sigma_clipped_stats
    from photutils.detection import DAOStarFinder

    try:
        from vyvar_blind_series import solve_blind_with_series

        _, med, std = sigma_clipped_stats(data, sigma=3.0)
        finder = DAOStarFinder(fwhm=3.0, threshold=5.0 * float(std))
        srcs = finder(data - float(med))
        if srcs is None or len(srcs) < 3:
            return None
        bdf = srcs.to_pandas().rename(columns={"xcentroid": "x", "ycentroid": "y"})
        if "peak" in bdf.columns and "flux" not in bdf.columns:
            bdf["flux"] = bdf["peak"]
        elif "flux" not in bdf.columns:
            bdf["flux"] = 1.0
        bdf = bdf.sort_values("flux", ascending=False)
        blind_sink: dict[str, Any] = {}
        series = solve_blind_with_series(
            bdf,
            hdr0,
            plate_scale_arcsec_per_px=plate_scale_arcsec_per_px,
            fov_deg=fov_deg,
            max_cat_mag=float(max_cat_mag),
            debug_sink=blind_sink if bool(getattr(app_config, "debug_platesolver", False)) else None,
        )
        if series is None:
            return None
        ra_b, de_b, tier = float(series[0]), float(series[1]), str(series[2])
        if not (math.isfinite(ra_b) and math.isfinite(de_b)):
            return None
        return ra_b, de_b, tier
    except Exception as exc:  # noqa: BLE001
        log_event(f"WARNING: Blind series hint failed: {exc}")
        return None


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
    preferred_mirror: str | None = None,
    masterstar_prewrite_rms_max_px: float | None = None,
    masterstar_prewrite_relaxed_rms_max_px: float | None = None,
    masterstar_nn_refine_max_rms_px: float | None = None,
    masterstar_sip_min_order: int | None = None,
    masterstar_sip_force_rms_guard_ratio: float | None = None,
    app_config: Any | None = None,
    solver_use_cone_for_sip: bool = True,
    solver_apply_roworder_yflip: bool = False,
    solver_legacy_masterstar_mirror_sweep: bool = True,
    solver_fits_header_hint_sep_escape: bool = True,
    solver_skip_header_coords: bool = False,
    solver_blind_fallback_attempted: bool = False,
) -> dict[str, Any]:
    """Plate-solve by matching DAO stars to **local Gaia DR3** (SQLite); writes WCS into the FITS primary HDU.

    **RA/Dec:** najprv z FITS hlavičky (``VY_TARG*``, ``RA``/``DEC``, …); ak chýbajú, použije sa hint z argumentov
    ``hint_ra_deg`` / ``hint_dec_deg``; ak stále chýba, spustí sa **blind** trojuholníkový solver
    (``blind_index_path``).

    **Mierka:** FOCALLEN+PIXSIZE alebo SECPIX/PIXSCALE/SCALE v hlavičke alebo argument
    ``expected_plate_scale_arcsec_per_px`` (napr. MASTERSTAR).

    ``fit_wcs_from_points(..., projection=\"TAN\")`` len zostaví **lineárny** TAN; SIP sa dopočíta po zhode hviezd.
    """
    from astropy.stats import sigma_clipped_stats
    from photutils.detection import DAOStarFinder


    _caller_hint_ra = hint_ra_deg
    _caller_hint_dec = hint_dec_deg

    fp = Path(fits_path).resolve()
    if not fp.is_file():
        return {"solved": False, "reason": f"File not found: {fp}"}
    _cfg_ps = app_config or AppConfig()
    _is_masterstar = fp.name.strip().upper() == "MASTERSTAR.FITS"
    log_event(
        "VYVAR solver flags: "
        f"cone_sip={bool(solver_use_cone_for_sip)} "
        f"hint_sep_escape={bool(solver_fits_header_hint_sep_escape)} "
        f"legacy_mirror={bool(solver_legacy_masterstar_mirror_sweep)} "
        f"roworder_yflip={bool(solver_apply_roworder_yflip)} "
        f"app_config={'passed' if app_config is not None else 'default'}"
    )

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
            # WAVE-B STEP 6: hardcoded solver internal (was cfg.masterstar_sip_force_rms_guard_ratio).
            _ms_sip_guard_r = _MASTERSTAR_SIP_FORCE_RMS_GUARD_RATIO
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

    # Derive a reliable FOV diameter from optics when available. This protects against a wrong
    # config/UI `fov_diameter_deg` (often a wide default like 7°) which would break blind hints
    # and Gaia cone sizing for narrow-field telescopes.
    fov_diameter_deg_eff = float(fov_diameter_deg)
    try:
        if effective_pixel_um is not None and focal_length_mm is not None:
            _eff_um0 = float(effective_pixel_um)
            _foc0 = float(focal_length_mm)
            if math.isfinite(_eff_um0) and _eff_um0 > 0 and math.isfinite(_foc0) and _foc0 > 0:
                _sc0 = plate_scale_arcsec_per_pixel(pixel_pitch_um=float(_eff_um0), focal_length_mm=float(_foc0))
                if _sc0 is not None and math.isfinite(float(_sc0)) and float(_sc0) > 0:
                    _diag = plate_solve_fov_deg_diagonal_from_scale(int(naxis1), int(naxis2), float(_sc0))
                    if _diag is not None and math.isfinite(float(_diag)) and float(_diag) > 0:
                        # Use diagonal FOV as a robust scale for "full-frame" solve.
                        # If caller FOV is wildly different, override it.
                        _diag_eff = float(_diag)
                        try:
                            _caller = float(fov_diameter_deg)
                        except (TypeError, ValueError):
                            # EXC-0612: T4 -- optics FOV override fail surfaced as WARNING, falls back to caller FOV (EXCEPT-BULK 2026-07-08)
                            _caller = float("nan")
                        if (not math.isfinite(_caller)) or (_caller <= 0) or (_caller > _diag_eff * 3.0) or (_caller < _diag_eff / 3.0):
                            fov_diameter_deg_eff = _diag_eff
                            log_event(
                                f"INFO: FOV override from optics: caller={_caller if math.isfinite(_caller) else 'n/a'}° "
                                f"→ diag_from_optics={_diag_eff:.3f}° (F={_foc0:g}mm, Px_eff={_eff_um0:g}µm)."
                            )
    except Exception as _fov_exc:  # noqa: BLE001
        log_event(
            f"WARNING: optics-based FOV override failed ({_fov_exc}) — falling back to caller "
            f"fov_diameter_deg={float(fov_diameter_deg):g}°."
        )

    # RA/Dec: FITS header first, then caller hint, then blind solver.
    ra0: float | None = None
    de0: float | None = None
    _coord_src = ""

    ra_h, dec_h, src_h = (None, None, "")
    if not bool(solver_skip_header_coords):
        ra_h, dec_h, src_h = pointing_hint_from_header(hdr0)
    if (ra_h is None or dec_h is None) and (_caller_hint_ra is not None) and (_caller_hint_dec is not None):
        try:
            _rr = float(_caller_hint_ra)
            _dd = float(_caller_hint_dec)
            if math.isfinite(_rr) and math.isfinite(_dd):
                ra_h, dec_h, src_h = _rr, _dd, "caller_hint"
        except (TypeError, ValueError):
            pass
    if bool(solver_skip_header_coords) and ra_h is not None and dec_h is not None:
        src_h = src_h or "blind solver (fallback)"
    # Optional debug: trace why hint is missing (per-frame should have VYTARG* injected).
    try:
        if bool(getattr(_cfg_ps, "debug_platesolver", False)):
            _vra = hdr0.get("VYTARGRA")
            _vde = hdr0.get("VYTARGDE")
            _ra_kw = hdr0.get("RA")
            _de_kw = hdr0.get("DEC", hdr0.get("DE"))
            _cr1 = hdr0.get("CRVAL1")
            _cr2 = hdr0.get("CRVAL2")
            log_event(
                "DEBUG: pointing_hint_from_header="
                f"(ra={ra_h}, dec={dec_h}, src={src_h}); "
                f"hdr[VYTARGRA]={_vra} hdr[VYTARGDE]={_vde} "
                f"hdr[RA]={_ra_kw} hdr[DEC/DE]={_de_kw} "
                f"hdr[CRVAL1]={_cr1} hdr[CRVAL2]={_cr2}"
            )
    except Exception:  # noqa: BLE001
        pass
    if ra_h is not None and dec_h is not None:
        try:
            _rf, _df = float(ra_h), float(dec_h)
            if math.isfinite(_rf) and math.isfinite(_df):
                ra0, de0 = _rf, _df
                if bool(solver_skip_header_coords):
                    _coord_src = str(src_h or "blind solver (fallback)")
                else:
                    _coord_src = f"FITS header ({src_h})"
                log_event(f"INFO: Solver hint: RA={ra0:.4f} Dec={de0:.4f} ({_coord_src})")
        except (TypeError, ValueError):
            ra0, de0 = None, None

    _blind_plate_scale: float | None = None
    if expected_plate_scale_arcsec_per_px is not None:
        try:
            _bps = float(expected_plate_scale_arcsec_per_px)
            if math.isfinite(_bps) and _bps > 0:
                _blind_plate_scale = _bps
        except (TypeError, ValueError):
            pass
    _blind_fov_deg: float | None = None
    try:
        _bfd = float(fov_diameter_deg_eff)
        if math.isfinite(_bfd) and _bfd > 0:
            _blind_fov_deg = _bfd
    except (TypeError, ValueError):
        pass

    if ra0 is None or de0 is None:
        try:
            if bool(getattr(_cfg_ps, "debug_platesolver", False)):
                log_event(
                    f"DEBUG: entering blind-solver fallback (ra0={ra0}, de0={de0}, src={src_h}); "
                    f"VYTARGRA={hdr0.get('VYTARGRA')} VYTARGDE={hdr0.get('VYTARGDE')}"
                )
        except Exception:  # noqa: BLE001
            pass
        log_event("INFO: FITS nemá RA/Dec — spúšťam blind solver.")
        try:
            from vyvar_blind_series import solve_blind_with_series

            _, _bmed, _bstd = sigma_clipped_stats(data, sigma=3.0)
            _bfinder = DAOStarFinder(fwhm=3.0, threshold=5.0 * _bstd)
            _bsrcs = _bfinder(data - _bmed)
            if _bsrcs is not None and len(_bsrcs) >= 3:
                _bdf = _bsrcs.to_pandas().rename(
                    columns={"xcentroid": "x", "ycentroid": "y"}
                )
                if "peak" in _bdf.columns and "flux" not in _bdf.columns:
                    _bdf["flux"] = _bdf["peak"]
                elif "flux" not in _bdf.columns:
                    _bdf["flux"] = 1.0
                _bdf = _bdf.sort_values("flux", ascending=False)
                _blind_sink: dict[str, Any] = {}
                _series = solve_blind_with_series(
                    _bdf,
                    app_config=_cfg_ps,
                    plate_scale_arcsec_per_px=_blind_plate_scale,
                    fov_deg=_blind_fov_deg,
                    gaia_db_path=root,
                    naxis1=int(naxis1),
                    naxis2=int(naxis2),
                    pixel_pitch_um=effective_pixel_um,
                    focal_length_mm=focal_length_mm,
                    max_cat_mag=float(max_cat_mag),
                    debug_sink=_blind_sink if bool(getattr(_cfg_ps, "debug_platesolver", False)) else None,
                )
                if _series is not None:
                    ra0, de0, _tier = _series[0], _series[1], _series[2]
                    _verify_on = bool(getattr(_cfg_ps, "blind_verify_enabled", True))
                    _coord_src = (
                        f"blind solver (series tier={_tier}, verify)"
                        if _verify_on
                        else f"blind solver (series tier={_tier}, vote-only)"
                    )
                    log_event(f"INFO: Blind solver hint: RA={ra0:.4f} Dec={de0:.4f}")
        except Exception as _be:  # noqa: BLE001
            log_event(f"WARNING: Blind solver exception: {_be}")

    if ra0 is None or de0 is None:
        return {
            "solved": False,
            "reason": "VYVAR solver: FITS nemá RA/Dec a blind solver nenašiel zhodu.",
        }

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
    _exp_scale_from_expected_arg = False
    if expected_plate_scale_arcsec_per_px is not None:
        try:
            _es = float(expected_plate_scale_arcsec_per_px)
            if math.isfinite(_es) and _es > 0:
                _exp_scale = _es
                _exp_scale_from_expected_arg = True
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
    _coord_src_l2 = str(_coord_src or "").strip().lower()
    _hint_is_blind = "blind solver" in _coord_src_l2
    if _hint_is_blind:
        log_event(
            f"VYVAR platesolve: očakávaná mierka ≈ {_exp_scale:.3f} arcsec/px — "
            "blind hint: mierku použijem len ako slabý hint; pri slabom matchi povolím fallback bez scale filtra."
        )
    else:
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
        fov_diameter_deg=float(fov_diameter_deg_eff),
    )
    _f_um = float(_ep_um) if _ep_um is not None else 0.0
    _f_mm_u = float(_foc_mm) if _foc_mm is not None else 0.0
    try:
        _scale_arcsec = plate_scale_arcsec_per_pixel(pixel_pitch_um=float(_f_um), focal_length_mm=float(_f_mm_u))
    except Exception:  # noqa: BLE001
        _scale_arcsec = None
    if (
        _scale_arcsec is None
        or (not math.isfinite(float(_scale_arcsec)))
        or float(_scale_arcsec) <= 0.0
    ):
        if _exp_scale is not None and math.isfinite(float(_exp_scale)) and float(_exp_scale) > 0.0:
            _scale_arcsec = float(_exp_scale)
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
    try:
        cat_df, cat_df_tri, c_cat, cone_r, cat_df_cone_full, eff_max_cat_mag = _solve_wcs_build_catalog(
            pointing_ra=float(ra0),
            pointing_dec=float(de0),
            fov_diameter_deg_eff=float(fov_diameter_deg_eff),
            exp_scale=_exp_scale,
            chip_fw=int(naxis1),
            chip_fh=int(naxis2),
            gaia_db_path=root,
            eff_max_cat_mag=float(max_cat_mag),
            obs_epoch=float(_obs_year_from_header(hdr0)),
            logger=LOGGER,
            hdr0=hdr0,
            fov_diameter_deg=float(fov_diameter_deg),
            pixel_pitch_um=float(_f_um),
            focal_length_mm=_foc_mm,
            scale_arcsec=_scale_arcsec,
            optimal_params=_opt,
            max_catalog_rows=max_catalog_rows,
            max_cat_mag=float(max_cat_mag),
            faintest_mag_limit=faintest_mag_limit,
            coord_src=str(_coord_src),
            exp_scale_from_expected_arg=bool(_exp_scale_from_expected_arg),
            app_config=_cfg_ps,
        )
    except _SolveWcsCatalogError as exc:
        return exc.result


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
        _sips_fb = float(_cfg_ps.sips_dao_fwhm_px)
        if not math.isfinite(_sips_fb) or _sips_fb <= 0:
            _sips_fb = 2.5
        _sig_cfg = float(_cfg_ps.sips_dao_threshold_sigma)
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
    _cap_adaptive = bool(getattr(_cfg_ps, "masterstar_detection_cap_adaptive", True))
    _cap_min = int(getattr(_cfg_ps, "masterstar_detection_cap_min", 250))
    _cap_max = int(getattr(_cfg_ps, "masterstar_detection_cap_max", 800))
    _cap_k = float(getattr(_cfg_ps, "masterstar_detection_cap_k", 0.08))
    _n_cat_est = int(len(cat_df_tri)) if cat_df_tri is not None else 0
    if _is_masterstar and _cap_adaptive and _n_cat_est > 0:
        top = min(len(tbl_sorted), max(_cap_min, min(_cap_max, int(_cap_k * _n_cat_est))))
        log_event(
            f"VYVAR MASTERSTAR: adaptive detection cap {top} "
            f"(n_cat_tri={_n_cat_est}, k={_cap_k}, bounds [{_cap_min},{_cap_max}])"
        )
    else:
        top = min(250, len(tbl_sorted))
    tbl = tbl_sorted[:top]
    xs = np.asarray(tbl["xcentroid"], dtype=np.float64)
    ys = np.asarray(tbl["ycentroid"], dtype=np.float64)
    _roworder_ori: str | None = None
    if bool(solver_apply_roworder_yflip):
        xs, ys, _roworder_ori = _apply_fits_roworder_to_detections(
            xs, ys, hdr=hdr0, naxis2=int(naxis2)
        )
    n_img = len(xs)
    if n_img < 6:
        return {"solved": False, "reason": "VYVAR solver: po orezaní málo hviezd na snímke."}

    xs_native = np.asarray(xs, dtype=np.float64, copy=True)
    ys_native = np.asarray(ys, dtype=np.float64, copy=True)

    # Ignore CROTA: triangle matching is rotation-invariant by construction.

    probe0 = _gaia_triangle_greedy_orientation_probe(
        cat_df_tri,
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
    # If we only have a blind hint, the DB/config plate scale can be wrong (common after equipment DB edits).
    # When the initial probe match is weak, retry without scale filtering / with wide scale tolerance.
    try:
        _probe_rate0 = float(probe0.get("match_rate", 0.0) or 0.0)
    except (TypeError, ValueError):
        _probe_rate0 = 0.0
    _coord_src_l3 = str(_coord_src or "").strip().lower()
    _hint_is_blind2 = "blind solver" in _coord_src_l3
    if _hint_is_blind2 and _probe_rate0 < 0.06:
        log_event(
            f"INFO: Blind hint + weak initial probe ({_probe_rate0 * 100.0:.1f}%) — retry triangle probe "
            "bez scale filtra (wide tolerance)."
        )
        probe_relaxed = _gaia_triangle_greedy_orientation_probe(
            cat_df_tri,
            xs_native,
            ys_native,
            naxis1=int(naxis1),
            naxis2=int(naxis2),
            w=float(w),
            h=float(h),
            simple_mode=bool(_simple_mode),
            exp_scale=None,
            silent_catalog_crop_log=False,
            max_px_coarse_override=None,
            expected_scale_rel_tol_override=1.0,
        )
        if probe_relaxed is not None:
            try:
                _pr2 = float(probe_relaxed.get("match_rate", 0.0) or 0.0)
            except (TypeError, ValueError):
                _pr2 = 0.0
            if _pr2 > _probe_rate0 + 0.02:
                log_event(
                    f"INFO: Relaxed probe improved match_rate {(_probe_rate0 * 100.0):.1f}% → {(_pr2 * 100.0):.1f}%."
                )
                probe0 = probe_relaxed

    ori_candidates: list[tuple[str, bool, bool, dict[str, Any]]] = [("native", False, False, probe0)]
    _probe_rate0 = float(probe0["match_rate"])
    # Slabý native match: vždy otestovať zrkadlá. MASTERSTAR: legacy path vždy porovná zrkadlá
    # (anchor-validated); ROWORDER skip len keď explicitne vypnutý legacy režim.
    _preferred = str(preferred_mirror or "").strip().lower() or None
    if _preferred not in {"native", "mirror_x", "mirror_y", "mirror_xy"}:
        _preferred = None

    if bool(solver_legacy_masterstar_mirror_sweep):
        _mirror_sweep = (
            bool(_preferred and _preferred != "native")
            or float(_probe_rate0) < 0.10
            or bool(_is_masterstar)
        )
    else:
        _roworder_native_ok = bool(_roworder_ori) and float(_probe_rate0) >= 0.10
        _mirror_sweep = bool(_preferred and _preferred != "native") or (
            float(_probe_rate0) < 0.10
            or (bool(_is_masterstar) and not _roworder_native_ok)
        )
        if _roworder_native_ok:
            log_event(
                f"INFO: FITS ROWORDER native parity {float(_probe_rate0) * 100.0:.1f}% — "
                "mirror sweep preskočený (fallback len pri native < 10%)."
            )
    if _mirror_sweep:
        mirrors = [("mirror_x", True, False), ("mirror_y", False, True), ("mirror_xy", True, True)]
        if _preferred and _preferred != "native":
            mirrors = sorted(mirrors, key=lambda t: (0 if t[0] == _preferred else 1))
        for name, fx, fy in mirrors:
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
                try:
                    log_event(f"Mirror probe {name}: match_rate={float(pr.get('match_rate', 0.0)) * 100.0:.1f}%")
                except Exception:  # noqa: BLE001
                    pass
                ori_candidates.append((name, fx, fy, pr))
                if _preferred and name == _preferred:
                    try:
                        if float(pr.get("match_rate", 0.0) or 0.0) > 0.50:
                            log_event(
                                f"INFO: Preferred mirror '{name}' potvrdený ({float(pr.get('match_rate', 0.0)) * 100.0:.1f}%) — skracujem probe."
                            )
                            break
                    except Exception:  # noqa: BLE001
                        pass

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

    # One-shot global offset search (coarse): if initial pairing is very weak, test +/- arcmin RA/Dec shifts.
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
                # Expand search to handle manual re-center after meridian flip.
                # Keep it bounded to limit false matches.
                _max_off_m = 12.0 if _is_masterstar else 6.0
                _steps = [1.0, 2.0, 3.0, 5.0, float(_max_off_m)]
                deltas: list[float] = []
                for s in _steps:
                    if s not in deltas:
                        deltas.append(float(s))
                delta_grid = [-d for d in reversed(deltas)] + deltas
                for d_ra_m in delta_grid:
                    for d_de_m in delta_grid:
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
                    except Exception as exc:  # noqa: BLE001
                        LOGGER.debug("[SOLVER] center RA/Dec from WCS failed: %s", exc)
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
    if _roworder_ori:
        sip_meta["fits_roworder_applied"] = str(_roworder_ori)
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
            sip_meta["sip_pass1_deferred"] = True

            # Refine pass 2: wide greedy match → refit TAN+SIP (deep cone when enabled).
            if bool(solver_use_cone_for_sip):
                _ra_sip = cat_df_cone_full["ra_deg"].to_numpy(dtype=np.float64)
                _de_sip = cat_df_cone_full["dec_deg"].to_numpy(dtype=np.float64)
            else:
                _ra_sip = np.asarray(ra_all, dtype=np.float64)
                _de_sip = np.asarray(de_all, dtype=np.float64)
            rng_seed = (hash(str(fp)) & 0xFFFFFFFF) ^ 0xA5A51234
            _mr_before_refine = float(len(pairs_x)) / float(max(1, int(n_img)))
            log_event(
                "VYVAR full-pair refit: entering "
                f"n_coarse={len(pairs_x)} match_rate={_mr_before_refine * 100.0:.1f}% "
                f"cone_sip={bool(solver_use_cone_for_sip)} "
                f"gaia_cone_n={len(_ra_sip)} detections={int(n_img)}"
            )
            w_ref, prx, pry, prra, prde, sip_pass2 = _refit_linear_and_sip_on_full_pairs(
                wcs_final,
                _ra_sip,
                _de_sip,
                xs,
                ys,
                max_px_coarse=max_px_coarse,
                enable_sip=enable_sip,
                sip_max_order=int(sip_max_order),
                sip_min_order=int(_sip_min_ms),
                is_masterstar=bool(_is_masterstar),
                sip_force_rms_guard_ratio=_ms_sip_guard_r,
                ransac_refinement=ransac_refinement,
                ransac_min_pairs=ransac_min_pairs,
                rng_seed=rng_seed,
            )
            n_coarse = len(pairs_x)
            sip_meta["n_pairs_coarse"] = int(n_coarse)
            sip_meta["n_pairs_full_sip"] = int(len(prx))
            _mr_full_pairs = float(len(prx)) / float(max(1, int(n_img)))
            log_event(
                "VYVAR full-pair refit: greedy result "
                f"n_pairs={len(prx)} match_rate={_mr_full_pairs * 100.0:.1f}% "
                f"max_px_sip={sip_pass2.get('max_px_sip')} "
                f"sip_skipped={sip_pass2.get('sip_skipped', '')!s}"
            )
            if len(prx) >= 5:
                world_m2 = SkyCoord(
                    ra=np.asarray(prra, dtype=np.float64) * u.deg,
                    dec=np.asarray(prde, dtype=np.float64) * u.deg,
                    frame="icrs",
                )
                pxa2 = np.asarray(prx, dtype=np.float64)
                pya2 = np.asarray(pry, dtype=np.float64)
                try:
                    rms_prev = _wcs_pixel_rms_full(wcs_final, pxa2, pya2, world_m2)
                    rms_new = _wcs_pixel_rms_full(w_ref, pxa2, pya2, world_m2)
                    sip_meta["refine_full_pairs_rms_prev"] = float(rms_prev)
                    sip_meta["refine_full_pairs_rms_new"] = float(rms_new)
                    if rms_new <= rms_prev * 1.08:
                        wcs_final = w_ref
                        sip_meta["refine_full_pairs_applied"] = True
                        sip_meta.update(sip_pass2)
                        pairs_x, pairs_y, pairs_ra, pairs_de = prx, pry, prra, prde
                        log_event(
                            "VYVAR full-pair refit: ADOPTED "
                            f"rms {rms_prev:.2f}→{rms_new:.2f}px "
                            f"pairs {n_coarse}→{len(prx)} "
                            f"match_rate {_mr_before_refine * 100.0:.1f}%→{_mr_full_pairs * 100.0:.1f}%"
                        )
                    else:
                        sip_meta["refine_full_pairs_applied"] = False
                        sip_meta["refine_full_pairs_rejected"] = "rms_regression"
                        log_event(
                            "VYVAR full-pair refit: REJECTED rms_regression "
                            f"rms {rms_prev:.2f}→{rms_new:.2f}px (limit {rms_prev * 1.08:.2f}px) "
                            f"pairs={len(prx)} match_rate={_mr_full_pairs * 100.0:.1f}%"
                        )
                except Exception as _ref_exc:  # noqa: BLE001
                    sip_meta["refine_full_pairs_applied"] = False
                    sip_meta["refine_full_pairs_error"] = True
                    log_event(f"VYVAR full-pair refit: ERROR {_ref_exc!r}")
            else:
                sip_meta["refine_full_pairs_applied"] = False
                sip_meta["refine_full_pairs_skipped"] = "too_few_pairs"
                log_event(
                    f"VYVAR full-pair refit: SKIPPED too_few_pairs n={len(prx)} "
                    f"(gate ≥5, coarse had {n_coarse})"
                )
        except Exception:  # noqa: BLE001
            wcs_final = wcs_init
            sip_meta["refine_error"] = True

    # MASTERSTAR cone recenter: when linear WCS center disagrees with header hint, the Gaia cone
    # queried at VY_TARG can skew full-pair matching (stale mount pointing). Re-query at solved center.
    if (
        _is_masterstar
        and bool(solver_use_cone_for_sip)
        and len(pairs_x) >= 5
    ):
        try:
            _cx_r = 0.5 * float(naxis1)
            _cy_r = 0.5 * float(naxis2)
            _ra_w, _de_w = wcs_final.all_pix2world([_cx_r], [_cy_r], 0)
            _sc_h = SkyCoord(ra=float(ra0) * u.deg, dec=float(de0) * u.deg, frame="icrs")
            _sc_w = SkyCoord(ra=float(_ra_w[0]) * u.deg, dec=float(_de_w[0]) * u.deg, frame="icrs")
            _off_deg = float(_sc_h.separation(_sc_w).deg)
            sip_meta["cone_hint_vs_wcs_center_deg"] = _off_deg
            if math.isfinite(_off_deg) and _off_deg >= 0.05:
                log_event(
                    f"VYVAR MASTERSTAR cone recenter: header hint vs WCS center = {_off_deg:.3f}° "
                    "— Gaia re-query at solved center + full-pair refit pass 3."
                )
                (
                    _cat_df_rc,
                    _cat_df_tri_rc,
                    _c_cat_rc,
                    _cone_r_rc,
                    cat_df_cone_full,
                    _eff_mag_rc,
                ) = _solve_wcs_build_catalog(
                    pointing_ra=float(_ra_w[0]),
                    pointing_dec=float(_de_w[0]),
                    fov_diameter_deg_eff=float(fov_diameter_deg_eff),
                    exp_scale=_exp_scale,
                    chip_fw=int(naxis1),
                    chip_fh=int(naxis2),
                    gaia_db_path=root,
                    eff_max_cat_mag=float(max_cat_mag),
                    obs_epoch=float(_obs_year_from_header(hdr0)),
                    logger=LOGGER,
                    hdr0=hdr0,
                    fov_diameter_deg=float(fov_diameter_deg),
                    pixel_pitch_um=float(_f_um),
                    focal_length_mm=_foc_mm,
                    scale_arcsec=_scale_arcsec,
                    optimal_params=_opt,
                    max_catalog_rows=max_catalog_rows,
                    max_cat_mag=float(max_cat_mag),
                    faintest_mag_limit=faintest_mag_limit,
                    coord_src=f"{_coord_src}; cone recentered on WCS",
                    exp_scale_from_expected_arg=bool(_exp_scale_from_expected_arg),
                    app_config=_cfg_ps,
                )
                cat_df = _cat_df_rc
                cat_df_tri = _cat_df_tri_rc
                cone_r = float(_cone_r_rc)
                _ra_sip_rc = cat_df_cone_full["ra_deg"].to_numpy(dtype=np.float64)
                _de_sip_rc = cat_df_cone_full["dec_deg"].to_numpy(dtype=np.float64)
                _mr_before_rc = float(len(pairs_x)) / float(max(1, int(n_img)))
                w_ref3, prx3, pry3, prra3, prde3, sip_pass3 = _refit_linear_and_sip_on_full_pairs(
                    wcs_final,
                    _ra_sip_rc,
                    _de_sip_rc,
                    xs,
                    ys,
                    max_px_coarse=max_px_coarse,
                    enable_sip=enable_sip,
                    sip_max_order=int(sip_max_order),
                    sip_min_order=int(_sip_min_ms),
                    is_masterstar=True,
                    sip_force_rms_guard_ratio=_ms_sip_guard_r,
                    ransac_refinement=ransac_refinement,
                    ransac_min_pairs=ransac_min_pairs,
                    rng_seed=(hash(str(fp)) & 0xFFFFFFFF) ^ 0xC0E12345,
                )
                _mr_rc = float(len(prx3)) / float(max(1, int(n_img)))
                log_event(
                    f"VYVAR cone recenter refit: n_pairs={len(prx3)} "
                    f"match_rate {_mr_before_rc * 100.0:.1f}%→{_mr_rc * 100.0:.1f}%"
                )
                if len(prx3) >= 5:
                    world_m3 = SkyCoord(
                        ra=np.asarray(prra3, dtype=np.float64) * u.deg,
                        dec=np.asarray(prde3, dtype=np.float64) * u.deg,
                        frame="icrs",
                    )
                    pxa3 = np.asarray(prx3, dtype=np.float64)
                    pya3 = np.asarray(pry3, dtype=np.float64)
                    rms_prev3 = _wcs_pixel_rms_full(wcs_final, pxa3, pya3, world_m3)
                    rms_new3 = _wcs_pixel_rms_full(w_ref3, pxa3, pya3, world_m3)
                    _adopt_rc = (
                        rms_new3 <= rms_prev3 * 1.08
                        and (
                            _mr_rc > _mr_before_rc + 0.03
                            or len(prx3) >= len(pairs_x) + 8
                        )
                    )
                    if _adopt_rc:
                        wcs_final = w_ref3
                        pairs_x, pairs_y, pairs_ra, pairs_de = prx3, pry3, prra3, prde3
                        sip_meta["cone_recenter_refit_applied"] = True
                        sip_meta["cone_recenter_match_rate"] = float(_mr_rc)
                        sip_meta.update(sip_pass3)
                        log_event(
                            f"VYVAR cone recenter refit: ADOPTED rms {rms_prev3:.2f}→{rms_new3:.2f}px "
                            f"pairs {len(prx3)} match_rate={_mr_rc * 100.0:.1f}%"
                        )
                    else:
                        sip_meta["cone_recenter_refit_applied"] = False
                        log_event(
                            f"VYVAR cone recenter refit: REJECTED "
                            f"rms {rms_prev3:.2f}→{rms_new3:.2f}px "
                            f"match {_mr_before_rc * 100.0:.1f}%→{_mr_rc * 100.0:.1f}%"
                        )
        except Exception as _rc_exc:  # noqa: BLE001
            sip_meta["cone_recenter_error"] = repr(_rc_exc)
            log_event(f"VYVAR cone recenter: skipped ({_rc_exc!r})")

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
        # Rescale-to-expected is helpful when the expected scale comes from FITS optics;
        # but can be harmful when the expected scale comes from DB/config and the hint is blind.
        _hint_is_blind_cd = "blind solver" in str(_coord_src or "").strip().lower()
        _allow_expected_cd_rescale = not (bool(_hint_is_blind_cd) and bool(_exp_scale_from_expected_arg))
        if (not _cd_rescaled_any) and (_exp_scale is not None) and bool(_allow_expected_cd_rescale):
            w_adj, _cd_rescaled = maybe_rescale_linear_wcs_cd_to_target_arcsec_per_pixel(
                wcs_final, float(_exp_scale)
            )
            if _cd_rescaled:
                wcs_final = w_adj
                sip_meta["cd_rescaled_to_expected_scale"] = True
                log_event(
                    f"VYVAR WCS: CD/PC škálované podľa optickej mierky {float(_exp_scale):.3f} arcsec/px"
                )
        elif (not _cd_rescaled_any) and (_exp_scale is not None) and (not bool(_allow_expected_cd_rescale)):
            log_event(
                f"VYVAR WCS: vynechávam CD/PC škálovanie podľa očakávanej mierky {float(_exp_scale):.3f} arcsec/px "
                "(blind hint + expected scale z DB/config môže byť nesprávna)."
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
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("[SOLVER] anisotropic repair apply skipped: %s", exc)
            pass

    # Association QA + the post-solve NN refine (``_solve_wcs_validate_and_refine``) use the FULL
    # deep cone catalog ``cat_df_cone_full`` — not the bright-only triangle slice — otherwise faint
    # DAO peaks look "unmatched" and match% is misleadingly low.
    # NOTE (robustness audit): ``cat_df_assoc`` feeds the NN pair re-matching that can refine the
    # final WCS/SIP, so it MUST remain the deep cone. An earlier slice-rebuild here referenced
    # out-of-scope bbox names (``ra_min``/``de_min``/``_obs_year`` — locals of
    # ``_solve_wcs_build_catalog``) and therefore NameError'd on every call, silently falling back
    # to this exact assignment. Removed the dead block to drop the latent NameError WITHOUT changing
    # the catalog that drives the solve (guaranteed no-op vs prior runtime behaviour).
    cat_df_assoc = cat_df_cone_full

    try:
        (
            wcs_final,
            hdr0,
            pairs_x,
            pairs_y,
            pairs_ra,
            pairs_de,
            sip_meta,
            _match_rate,
            _rms_px,
            _dao_fw,
        ) = _solve_wcs_validate_and_refine(
            wcs_final=wcs_final,
            pairs_final=pairs_x,
            cat_df=cat_df,
            cat_df_assoc=cat_df_assoc,
            xs_native=xs_native,
            ys_native=ys_native,
            hdr0=hdr0,
            fp=fp,
            is_masterstar=_is_masterstar,
            hint_ra=ra0,
            hint_dec=de0,
            sip_meta=sip_meta,
            gaia_db_path=str(root),
            logger=LOGGER,
            pairs_y=pairs_y,
            pairs_ra=pairs_ra,
            pairs_de=pairs_de,
            n_img=int(n_img),
            naxis1=int(naxis1),
            naxis2=int(naxis2),
            fov_diameter_deg=float(fov_diameter_deg),
            coord_src=_coord_src,
            best_rms=float(best_rms),
            dao_fw=float(_dao_fw),
            tbl_sorted=tbl_sorted,
            cat_df_cone_full=cat_df_cone_full,
            max_px_coarse=float(max_px_coarse),
            w=int(w),
            h=int(h),
            enable_sip=bool(enable_sip),
                        sip_max_order=int(sip_max_order),
            ransac_min_pairs=int(ransac_min_pairs),
                        sip_min_order=int(_sip_min_ms),
                        sip_force_rms_guard_ratio=_ms_sip_guard_r,
            masterstar_prewrite_rms_max_px=masterstar_prewrite_rms_max_px,
            masterstar_prewrite_relaxed_rms_max_px=masterstar_prewrite_relaxed_rms_max_px,
            masterstar_nn_refine_max_rms_px=masterstar_nn_refine_max_rms_px,
            fits_header_hint_sep_escape=bool(solver_fits_header_hint_sep_escape),
            app_config=_cfg_ps,
        )
    except _SolveWcsValidationError as exc:
        if (
            _is_masterstar
            and not bool(solver_blind_fallback_attempted)
            and "blind solver" not in str(_coord_src or "").lower()
        ):
            _bl_fb = _try_blind_series_hint(
                data,
                hdr0,
                plate_scale_arcsec_per_px=_blind_plate_scale,
                fov_deg=_blind_fov_deg,
                max_cat_mag=float(max_cat_mag),
                app_config=_cfg_ps,
            )
            if _bl_fb is not None:
                log_event(
                    "INFO: MASTERSTAR validation failed — blind fallback retry "
                    f"(prior hint {float(ra0):.4f},{float(de0):.4f} → blind {float(_bl_fb[0]):.4f},{float(_bl_fb[1]):.4f})."
                )
                return solve_wcs_with_local_gaia(
                    fits_path,
                    hint_ra_deg=float(_bl_fb[0]),
                    hint_dec_deg=float(_bl_fb[1]),
                    fov_diameter_deg=fov_diameter_deg,
                    gaia_db_path=gaia_db_path,
                    dao_threshold_sigma=dao_threshold_sigma,
                    max_cat_mag=max_cat_mag,
                    enable_sip=enable_sip,
                    sip_max_order=sip_max_order,
                    ransac_refinement=ransac_refinement,
                    ransac_min_pairs=ransac_min_pairs,
                    effective_pixel_um=effective_pixel_um,
                    focal_length_mm=focal_length_mm,
                    expected_plate_scale_arcsec_per_px=expected_plate_scale_arcsec_per_px,
                    max_catalog_rows=max_catalog_rows,
                    faintest_mag_limit=faintest_mag_limit,
                    preferred_mirror=preferred_mirror,
                    masterstar_prewrite_rms_max_px=masterstar_prewrite_rms_max_px,
                    masterstar_prewrite_relaxed_rms_max_px=masterstar_prewrite_relaxed_rms_max_px,
                    masterstar_nn_refine_max_rms_px=masterstar_nn_refine_max_rms_px,
                    masterstar_sip_min_order=masterstar_sip_min_order,
                    masterstar_sip_force_rms_guard_ratio=masterstar_sip_force_rms_guard_ratio,
                    app_config=app_config,
                    solver_use_cone_for_sip=solver_use_cone_for_sip,
                    solver_apply_roworder_yflip=solver_apply_roworder_yflip,
                    solver_legacy_masterstar_mirror_sweep=solver_legacy_masterstar_mirror_sweep,
                    solver_fits_header_hint_sep_escape=solver_fits_header_hint_sep_escape,
                    solver_skip_header_coords=True,
                    solver_blind_fallback_attempted=True,
                )
        return exc.result


    try:
        _solve_wcs_write_results(
            fp=fp,
            hdr0=hdr0,
            wcs_final=wcs_final,
            sip_meta=sip_meta,
            pairs_final=pairs_x,
            match_rate=_match_rate,
            rms_px=_rms_px,
            dao_fw=_dao_fw,
            platescale_arcsec_per_px=None,
            is_masterstar=_is_masterstar,
            logger=LOGGER,
            cone_r=float(cone_r),
            ep_um=_ep_um,
            n_img=int(n_img),
        )
    except _SolveWcsWriteError as exc:
        return exc.result

    vy_platescale_arcsec_per_px: float | None = None
    try:
        if hdr0.get("VY_PLTS") is not None:
            vy_platescale_arcsec_per_px = float(hdr0["VY_PLTS"][0])
    except Exception:  # noqa: BLE001
        vy_platescale_arcsec_per_px = None


    LOGGER.info(
        "VYVAR plate solve OK: %s n_match=%s sip=%s max_px_coarse/sip=%s/%s full_pairs=%s rms_lin=%s rms_sip=%s",
        fp.name,
        len(pairs_x),
        sip_meta.get("sip_applied", False),
        sip_meta.get("max_px_coarse"),
        sip_meta.get("max_px_sip"),
        sip_meta.get("refine_full_pairs_applied"),
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


# ---------------------------------------------------------------------------
# Pass 2 — sibling-WCS recovery (validated in sandbox/sibling_wcs_recovery_test.py)
# ---------------------------------------------------------------------------

SIBLING_WCS_TIGHT_PX: float = 2.5

FILTER_EFFECTIVE_WAVELENGTH_NM: dict[str, float] = {
    "u": 354.0,
    "g": 477.0,
    "r": 623.0,
    "i": 762.0,
    "z": 913.0,
    "y": 1035.0,
    "b": 440.0,
    "v": 551.0,
    "R": 623.0,
    "I": 762.0,
    "B": 440.0,
    "V": 551.0,
    "clear": 550.0,
    "none": 550.0,
    "nofilter": 550.0,
}


def filter_code_from_setup_name(setup: str) -> str:
    """Extract filter token from setup folder name (e.g. ``g_60_4`` → ``g``)."""
    s = str(setup or "").strip()
    if not s or s.casefold() == "(root)":
        return ""
    return s.split("_")[0].strip().lower()


def _sibling_cfg_thresholds(cfg: AppConfig) -> dict[str, float | int]:
    try:
        min_matched = int(cfg.masterstar_sibling_min_matched)
    except (TypeError, ValueError):
        min_matched = 40
    min_matched = max(1, min(500, int(min_matched)))
    try:
        rms_max = float(cfg.masterstar_sibling_rms_max_px)
    except (TypeError, ValueError):
        rms_max = 2.0
    if not math.isfinite(rms_max) or rms_max <= 0:
        rms_max = 2.0
    try:
        min_quads = int(cfg.masterstar_sibling_min_quadrants)
    except (TypeError, ValueError):
        min_quads = 3
    min_quads = max(1, min(4, int(min_quads)))
    try:
        stack_n = int(cfg.masterstar_sibling_stack_n)
    except (TypeError, ValueError):
        stack_n = 10
    stack_n = max(2, min(50, int(stack_n)))
    return {
        "min_matched": min_matched,
        "rms_max_px": rms_max,
        "min_quadrants": min_quads,
        "stack_n": stack_n,
    }


def _sibling_quadrant_count(
    xs: np.ndarray, ys: np.ndarray, naxis1: int, naxis2: int
) -> int:
    if len(xs) == 0:
        return 0
    cx, cy = float(naxis1) * 0.5, float(naxis2) * 0.5
    quads: set[int] = set()
    for x, y in zip(xs, ys, strict=False):
        q = (0 if float(x) < cx else 1) + (0 if float(y) < cy else 2)
        quads.add(int(q))
    return len(quads)


def _sibling_false_alarm_p(
    n_matched: int,
    n_det: int,
    n_cat: int,
    naxis1: int,
    naxis2: int,
    *,
    r_px: float,
) -> float:
    area = float(max(1, int(naxis1) * int(naxis2)))
    p_one = min(1.0, float(n_cat) * math.pi * float(r_px) ** 2 / area)
    if n_det <= 0 or n_matched <= 0:
        return 1.0
    try:
        from scipy.stats import binom

        return float(binom.sf(n_matched - 1, n_det, p_one))
    except Exception:  # noqa: BLE001
        lam = n_det * p_one
        if lam <= 0:
            return 1.0
        return float(min(1.0, lam ** n_matched))


def _sibling_odds_confirmed(
    metrics: dict[str, Any],
    *,
    min_matched: int,
    rms_max_px: float,
    min_quadrants: int,
    false_alarm_p_max: float = 1e-6,
) -> bool:
    """Sibling odds gate: binomial false-alarm path OR strong geometric evidence on crowded fields."""
    n_tight = int(metrics.get("n_matched_tight") or 0)
    rms_d = metrics.get("rms_px")
    quads = int(metrics.get("quadrants_with_match") or 0)
    p_false = float(metrics.get("false_alarm_p") or 1.0)
    rms_finite = rms_d is not None and math.isfinite(float(rms_d))
    rms_val = float(rms_d) if rms_finite else float("inf")
    quads_ok = quads >= int(min_quadrants)
    odds_ok = (
        n_tight >= int(min_matched)
        and rms_finite
        and rms_val <= float(rms_max_px)
        and quads_ok
        and p_false < float(false_alarm_p_max)
    )
    # Crowded fields inflate p_one -> p_false; sub-px RMS with many tight matches is not random.
    strong_n = max(int(min_matched) * 2, int(min_matched) + 20)
    strong_rms = float(rms_max_px) * 0.5
    geometric_ok = (
        n_tight >= strong_n
        and rms_finite
        and rms_val <= strong_rms
        and quads_ok
    )
    return odds_ok or geometric_ok


def _sibling_match_metrics(
    wcs_use: WCS,
    ra_cat: np.ndarray,
    de_cat: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    naxis1: int,
    naxis2: int,
    *,
    thresholds: dict[str, float | int],
    cat_pred_flip: str | None = None,
) -> dict[str, Any]:
    qa_px = max(15.0, min(48.0, 15.0 * 1.22))
    rec = _compute_masterstar_catalog_recovery(
        wcs_use,
        ra_cat,
        de_cat,
        xs,
        ys,
        naxis1=int(naxis1),
        naxis2=int(naxis2),
        qa_px=float(qa_px),
        tight_px=float(SIBLING_WCS_TIGHT_PX),
    )
    pred_x, pred_y = wcs_use.all_world2pix(ra_cat, de_cat, 0)
    pred_x = np.asarray(pred_x, dtype=np.float64)
    pred_y = np.asarray(pred_y, dtype=np.float64)
    if cat_pred_flip == "flip_x":
        pred_x = float(naxis1) - 1.0 - pred_x
    elif cat_pred_flip == "rot180":
        pred_x = float(naxis1) - 1.0 - pred_x
        pred_y = float(naxis2) - 1.0 - pred_y
    qx, qy, qra, qde = _greedy_match_pairs_pixel_wcs(
        wcs_use,
        ra_cat,
        de_cat,
        xs,
        ys,
        max_px=float(SIBLING_WCS_TIGHT_PX),
        cat_pred_xy=(pred_x, pred_y),
    )
    n_tight = int(len(qx))
    if n_tight > 0:
        res: list[float] = []
        for xi, yi, ra_i, de_i in zip(qx, qy, qra, qde, strict=False):
            px, py = wcs_use.all_world2pix(float(ra_i), float(de_i), 0)
            px, py = float(px), float(py)
            if cat_pred_flip == "flip_x":
                px = float(naxis1) - 1.0 - px
            elif cat_pred_flip == "rot180":
                px = float(naxis1) - 1.0 - px
                py = float(naxis2) - 1.0 - py
            res.append(float(math.hypot(px - float(xi), py - float(yi))))
        med_d = float(np.median(res))
        rms_d = float(math.sqrt(np.mean(np.square(res))))
    else:
        med_d = float("nan")
        rms_d = float("inf")
    n_cat = int(rec.get("n_cat_in_frame", 0))
    n_det = int(rec.get("n_detections_used", len(xs)))
    quads = _sibling_quadrant_count(np.asarray(qx), np.asarray(qy), naxis1, naxis2)
    p_false = _sibling_false_alarm_p(
        n_tight, n_det, n_cat, naxis1, naxis2, r_px=float(SIBLING_WCS_TIGHT_PX)
    )
    metrics = {
        "n_matched_tight": n_tight,
        "median_dpx": med_d,
        "rms_px": rms_d,
        "quadrants_with_match": quads,
        "false_alarm_p": p_false,
        "catalog_recovery_tight": rec.get("catalog_recovery_tight"),
        "catalog_recovery_tight_gate": rec.get("catalog_recovery_tight_gate"),
        "n_cat_in_frame": n_cat,
        "n_detections_used": n_det,
    }
    metrics["confirmed"] = _sibling_odds_confirmed(
        metrics,
        min_matched=int(thresholds["min_matched"]),
        rms_max_px=float(thresholds["rms_max_px"]),
        min_quadrants=int(thresholds["min_quadrants"]),
    )
    return metrics


def _sibling_match_offset_median(
    wcs: WCS,
    ra_cat: np.ndarray,
    de_cat: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    cat_pred_flip: str | None,
    naxis1: int,
    naxis2: int,
) -> tuple[float, float]:
    pred_x, pred_y = wcs.all_world2pix(ra_cat, de_cat, 0)
    pred_x = np.asarray(pred_x, dtype=np.float64)
    pred_y = np.asarray(pred_y, dtype=np.float64)
    if cat_pred_flip == "flip_x":
        pred_x = float(naxis1) - 1.0 - pred_x
    elif cat_pred_flip == "rot180":
        pred_x = float(naxis1) - 1.0 - pred_x
        pred_y = float(naxis2) - 1.0 - pred_y
    qx, qy, qra, qde = _greedy_match_pairs_pixel_wcs(
        wcs,
        ra_cat,
        de_cat,
        xs,
        ys,
        max_px=float(SIBLING_WCS_TIGHT_PX),
        cat_pred_xy=(pred_x, pred_y),
    )
    if len(qx) < 4:
        return float("nan"), float("nan")
    dxs: list[float] = []
    dys: list[float] = []
    for xi, yi, ra_i, de_i in zip(qx, qy, qra, qde, strict=False):
        px, py = wcs.all_world2pix(float(ra_i), float(de_i), 0)
        px, py = float(px), float(py)
        if cat_pred_flip == "flip_x":
            px = float(naxis1) - 1.0 - px
        elif cat_pred_flip == "rot180":
            px = float(naxis1) - 1.0 - px
            py = float(naxis2) - 1.0 - py
        dxs.append(float(xi) - px)
        dys.append(float(yi) - py)
    return float(np.median(dxs)), float(np.median(dys))


def _sibling_apply_bulk_shift_crpix(
    wcs: WCS, dx: float, dy: float, *, sx: int = -1, sy: int = -1
) -> WCS:
    w2 = wcs.deepcopy()
    w2.wcs.crpix[0] = float(w2.wcs.crpix[0]) + float(sx) * float(dx)
    w2.wcs.crpix[1] = float(w2.wcs.crpix[1]) + float(sy) * float(dy)
    return w2


def _sibling_best_bulk_shift(
    w_adopt: WCS,
    ra_cat: np.ndarray,
    de_cat: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    naxis1: int,
    naxis2: int,
    *,
    thresholds: dict[str, float | int],
    cat_pred_flip: str | None = None,
) -> tuple[WCS, dict[str, Any], dict[str, Any]]:
    before = _sibling_match_metrics(
        w_adopt,
        ra_cat,
        de_cat,
        xs,
        ys,
        naxis1,
        naxis2,
        thresholds=thresholds,
        cat_pred_flip=cat_pred_flip,
    )
    if before.get("confirmed"):
        return (
            w_adopt,
            {"dx": 0.0, "dy": 0.0, "applied": False, "reason": "already confirmed"},
            before,
        )
    mdx, mdy = _sibling_match_offset_median(
        w_adopt,
        ra_cat,
        de_cat,
        xs,
        ys,
        cat_pred_flip=cat_pred_flip,
        naxis1=naxis1,
        naxis2=naxis2,
    )
    if not (math.isfinite(mdx) and math.isfinite(mdy)):
        return (
            w_adopt,
            {"dx": mdx, "dy": mdy, "applied": False, "reason": "no offset"},
            before,
        )
    best_w = w_adopt
    best_after = before
    best_bulk: dict[str, Any] = {"dx": mdx, "dy": mdy, "applied": False, "sign": (0, 0)}
    for sx in (-1, 1):
        for sy in (-1, 1):
            w_try = _sibling_apply_bulk_shift_crpix(w_adopt, mdx, mdy, sx=sx, sy=sy)
            after_try = _sibling_match_metrics(
                w_try,
                ra_cat,
                de_cat,
                xs,
                ys,
                naxis1,
                naxis2,
                thresholds=thresholds,
                cat_pred_flip=cat_pred_flip,
            )
            score = (
                int(after_try.get("confirmed", False)),
                int(after_try.get("n_matched_tight") or 0),
                -float(after_try.get("median_dpx") or 99),
            )
            best_score = (
                int(best_after.get("confirmed", False)),
                int(best_after.get("n_matched_tight") or 0),
                -float(best_after.get("median_dpx") or 99),
            )
            if score > best_score:
                best_w, best_after = w_try, after_try
                best_bulk = {"dx": mdx, "dy": mdy, "applied": True, "sign": (sx, sy)}
    if not best_bulk.get("applied"):
        best_bulk["reason"] = "no improving shift"
    return best_w, best_bulk, best_after


def _sibling_adopt_and_confirm(
    donor_wcs: WCS,
    ra_cat: np.ndarray,
    de_cat: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    naxis1: int,
    naxis2: int,
    *,
    thresholds: dict[str, float | int],
    cat_pred_flip: str | None = None,
) -> dict[str, Any]:
    w_adopt = donor_wcs.deepcopy()
    before = _sibling_match_metrics(
        w_adopt,
        ra_cat,
        de_cat,
        xs,
        ys,
        naxis1,
        naxis2,
        thresholds=thresholds,
        cat_pred_flip=cat_pred_flip,
    )
    w_after, bulk, after = _sibling_best_bulk_shift(
        w_adopt,
        ra_cat,
        de_cat,
        xs,
        ys,
        naxis1,
        naxis2,
        thresholds=thresholds,
        cat_pred_flip=cat_pred_flip,
    )
    return {
        "wcs": w_after,
        "before": before,
        "bulk_shift": bulk,
        "after": after,
        "confirmed": bool(after.get("confirmed")),
    }


def pick_sibling_donor_filter(
    recipient_filter: str,
    verified_filters: "set[str] | dict[str, Any]",
) -> str | None:
    """Pick spectrally-nearest verified donor filter (preference, not hard rule)."""
    if isinstance(verified_filters, dict):
        candidates = {str(k) for k in verified_filters if str(k)}
    else:
        candidates = {str(f) for f in verified_filters if str(f)}
    candidates.discard(str(recipient_filter))
    if not candidates:
        return None
    lam_r = float(FILTER_EFFECTIVE_WAVELENGTH_NM.get(str(recipient_filter).lower(), 550.0))
    return min(
        candidates,
        key=lambda f: abs(
            float(FILTER_EFFECTIVE_WAVELENGTH_NM.get(str(f).lower(), 550.0)) - lam_r
        ),
    )


def _sibling_detect_dao_on_image(
    data: np.ndarray,
    hdr: fits.Header,
    *,
    dao_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    from astropy.stats import sigma_clipped_stats
    from photutils.detection import DAOStarFinder

    working = np.nan_to_num(np.asarray(data, dtype=np.float32))
    _, med_w, clipped_std = sigma_clipped_stats(working, sigma=3.0, maxiters=5)
    std = float(clipped_std) if np.isfinite(clipped_std) and clipped_std > 0 else 1.0
    img2 = np.clip(working - float(med_w), 0.0, None).astype(np.float32, copy=False)
    dao_fw = float(dao_detection_fwhm_pixels(hdr, configured_fallback=3.0) or 3.5)
    sig_try: list[float] = []
    for s in (float(dao_sigma), 2.0, 1.2, 1.0):
        ss = max(float(s), 1e-6)
        if not any(abs(ss - t) < 1e-9 for t in sig_try):
            sig_try.append(ss)
    tbl = None
    best_tbl, best_n = None, -1
    for s in sig_try:
        finder = DAOStarFinder(
            fwhm=dao_fw,
            threshold=max(float(s) * std, 1e-6),
            brightest=None,
            roundlo=-1.0,
            roundhi=1.0,
        )
        tbl_try = finder(img2)
        n_try = int(len(tbl_try)) if tbl_try is not None else 0
        if n_try > best_n:
            best_n, best_tbl = n_try, tbl_try
        if tbl_try is not None and n_try >= 50:
            tbl = tbl_try
            break
    if tbl is None:
        tbl = best_tbl
    if tbl is None or len(tbl) < 6:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    tbl = tbl[np.isfinite(tbl["xcentroid"]) & np.isfinite(tbl["ycentroid"]) & np.isfinite(tbl["flux"])]
    flux_arr = np.asarray(tbl["flux"], dtype=np.float64)
    order = np.argsort(-flux_arr)
    tbl = tbl[order[: min(250, len(tbl))]]
    xs = np.asarray(tbl["xcentroid"], dtype=np.float64)
    ys = np.asarray(tbl["ycentroid"], dtype=np.float64)
    return xs, ys


def _sibling_median_stack_into_fits(
    recipient_path: Path,
    frame_paths: "list[Path]",
    *,
    n_stack: int,
) -> int:
    paths = sorted(frame_paths, key=lambda p: str(p).casefold())
    if len(paths) <= 1:
        return 0
    n = min(max(2, int(n_stack)), len(paths))
    start = max(0, (len(paths) - n) // 2)
    picked = paths[start : start + n]
    stacks: list[np.ndarray] = []
    for fp in picked:
        with fits.open(fp, memmap=False) as hd:
            stacks.append(np.asarray(hd[0].data, dtype=np.float32))
    med = np.median(np.stack(stacks, axis=0), axis=0).astype(np.float32)
    with fits.open(recipient_path, mode="update", memmap=False) as hd:
        hd[0].data = med
        hd.flush()
    return int(n)


def _sibling_load_gaia_catalog(
    wcs_ref: WCS,
    hdr: fits.Header,
    naxis1: int,
    naxis2: int,
    *,
    gaia_db_path: Path,
    fov_diameter_deg: float,
    expected_plate_scale_arcsec_per_px: float | None,
    effective_pixel_um: float | None,
    focal_length_mm: float | None,
    app_config: AppConfig | None,
) -> tuple[np.ndarray, np.ndarray]:
    ra0 = float(wcs_ref.wcs.crval[0])
    de0 = float(wcs_ref.wcs.crval[1])
    try:
        obs_year = float(_obs_year_from_header(hdr))
    except Exception:  # noqa: BLE001
        obs_year = 2026.0
    _opt = get_optimal_params(
        focal_length_mm=float(focal_length_mm) if focal_length_mm is not None else None,
        pixel_size_um=float(effective_pixel_um) if effective_pixel_um is not None else None,
        fov_diameter_deg=float(fov_diameter_deg),
    )
    exp_scale = float(expected_plate_scale_arcsec_per_px) if expected_plate_scale_arcsec_per_px else None
    _cat = _solve_wcs_build_catalog(
        pointing_ra=ra0,
        pointing_dec=de0,
        fov_diameter_deg_eff=float(fov_diameter_deg),
        exp_scale=exp_scale,
        chip_fw=int(naxis1),
        chip_fh=int(naxis2),
        gaia_db_path=Path(gaia_db_path),
        eff_max_cat_mag=18.0,
        obs_epoch=obs_year,
        logger=None,
        hdr0=hdr,
        fov_diameter_deg=float(fov_diameter_deg),
        pixel_pitch_um=float(effective_pixel_um) if effective_pixel_um is not None else None,
        focal_length_mm=float(focal_length_mm) if focal_length_mm is not None else None,
        scale_arcsec=exp_scale,
        optimal_params=_opt,
        max_catalog_rows=30000,
        max_cat_mag=18.0,
        faintest_mag_limit=18.0,
        coord_src="sibling recovery",
        exp_scale_from_expected_arg=True,
        app_config=app_config,
    )
    cat_df_cone_full = _cat[4]
    ra = np.asarray(cat_df_cone_full["ra_deg"].to_numpy(dtype=np.float64))
    de = np.asarray(cat_df_cone_full["dec_deg"].to_numpy(dtype=np.float64))
    return ra, de


class _SiblingWcsCopyError(Exception):
    """Raised when a core WCS key cannot be copied during sibling recovery (EXC-0625).

    Signals the caller to ABORT recovery for that frame so no half-written WCS is persisted.
    """


def _write_sibling_recovered_wcs_to_fits(
    recipient_path: Path,
    wcs_final: WCS,
    *,
    donor_filter: str,
    bulk_shift: dict[str, Any],
    metrics: dict[str, Any],
    stack_n: int,
) -> None:
    from astropy.io.fits import Header
    from wcs_header_io import copy_wcs_header_keys

    wcs_hdr = wcs_final.to_header(relax=True)
    # EXC-0625 / EXCEPT-FIX-3 #8: validate core-key copyability BEFORE opening/mutating the
    # recipient FITS. If any core celestial key is uncopyable we abort here (nothing is
    # opened or stripped), so the frame stays unrecovered instead of getting a broken WCS.
    ctx = f"sibling recovery {recipient_path.name}"
    failed_core = copy_wcs_header_keys(Header(), wcs_hdr, context=ctx)
    if failed_core:
        raise _SiblingWcsCopyError(f"core WCS keys uncopyable: {failed_core}")
    with fits.open(recipient_path, mode="update", memmap=False) as hdul:
        h = hdul[0].header
        strip_celestial_wcs_keys(h)
        copy_wcs_header_keys(h, wcs_hdr, context=ctx)
        h["VY_CRT"] = ("sibling_recovered", "VYVAR creation route")
        h["VY_SIBL"] = (str(donor_filter), "Sibling donor filter")
        h["VY_SSHX"] = (float(bulk_shift.get("dx") or 0.0), "Sibling bulk-shift dx [px]")
        h["VY_SSHY"] = (float(bulk_shift.get("dy") or 0.0), "Sibling bulk-shift dy [px]")
        h["VY_SODD"] = (int(metrics.get("n_matched_tight") or 0), "Sibling odds n_matched_tight")
        h["VY_SRMS"] = (float(metrics.get("rms_px") or 0.0), "Sibling odds RMS [px]")
        h["VY_SSTK"] = (int(stack_n), "Sibling stack N frames (0=single)")
        h.add_history(
            "VYVAR: sibling-WCS Pass 2 recovery - donor geometry + bulk-shift + odds confirm"
        )
        hdul.flush()


def try_recover_masterstar_sibling_wcs(
    *,
    recipient_masterstar_fits: Path,
    donor_masterstar_fits: Path,
    recipient_filter: str,
    donor_filter: str,
    frame_paths: "list[Path] | None" = None,
    app_config: AppConfig | None = None,
    plate_solve_fov_deg: float | None = None,
    expected_plate_scale_arcsec_per_px: float | None = None,
    effective_pixel_um: float | None = None,
    focal_length_mm: float | None = None,
) -> dict[str, Any]:
    """Adopt donor WCS, bulk-shift on recipient detections, odds-confirm; optional median stack."""
    cfg = app_config or AppConfig()
    thresholds = _sibling_cfg_thresholds(cfg)
    recipient_path = Path(recipient_masterstar_fits)
    donor_path = Path(donor_masterstar_fits)
    if not recipient_path.is_file():
        return {"confirmed": False, "reason": f"recipient MASTERSTAR missing: {recipient_path}"}
    if not donor_path.is_file():
        return {"confirmed": False, "reason": f"donor MASTERSTAR missing: {donor_path}"}

    with fits.open(donor_path, memmap=False) as hd_d:
        donor_wcs = WCS(hd_d[0].header)
    with fits.open(recipient_path, memmap=False) as hd_r:
        hdr = hd_r[0].header.copy()
        data = np.asarray(hd_r[0].data, dtype=np.float32)
        naxis1 = int(hdr.get("NAXIS1", data.shape[1]))
        naxis2 = int(hdr.get("NAXIS2", data.shape[0]))

    if not donor_wcs.has_celestial:
        return {"confirmed": False, "reason": "donor WCS not celestial"}

    try:
        dao_sigma = float(cfg.masterstar_dao_threshold_sigma)
    except (TypeError, ValueError):
        dao_sigma = 2.1
    if not math.isfinite(dao_sigma) or dao_sigma <= 0:
        dao_sigma = 2.1

    fov_deg = float(plate_solve_fov_deg) if plate_solve_fov_deg is not None else float(cfg.plate_solve_fov_deg)
    gaia_db = Path(str(cfg.gaia_db_path or "").strip())
    if not gaia_db.is_file():
        return {"confirmed": False, "reason": "gaia_db_path missing"}

    ra_cat, de_cat = _sibling_load_gaia_catalog(
        donor_wcs,
        hdr,
        naxis1,
        naxis2,
        gaia_db_path=gaia_db,
        fov_diameter_deg=fov_deg,
        expected_plate_scale_arcsec_per_px=expected_plate_scale_arcsec_per_px,
        effective_pixel_um=effective_pixel_um,
        focal_length_mm=focal_length_mm,
        app_config=cfg,
    )

    # Validated sandbox path: no auto-flip; flip/0-match -> conservative skip (T5).
    cat_pred_flip: str | None = None

    xs, ys = _sibling_detect_dao_on_image(data, hdr, dao_sigma=dao_sigma)
    stack_n = 0
    adopt = _sibling_adopt_and_confirm(
        donor_wcs,
        ra_cat,
        de_cat,
        xs,
        ys,
        naxis1,
        naxis2,
        thresholds=thresholds,
        cat_pred_flip=cat_pred_flip,
    )
    n_after = int((adopt.get("after") or {}).get("n_matched_tight") or 0)
    flip_guard = int(thresholds["min_matched"]) // 4
    if n_after < flip_guard:
        return {
            "confirmed": False,
            "donor_filter": donor_filter,
            "reason": f"flip/0-match guard: n_tight={n_after} < {flip_guard}",
            "before": adopt.get("before"),
            "after": adopt.get("after"),
            "stack_n": 0,
        }

    if not adopt.get("confirmed") and frame_paths:
        stack_n = _sibling_median_stack_into_fits(
            recipient_path,
            list(frame_paths),
            n_stack=int(thresholds["stack_n"]),
        )
        if stack_n > 0:
            with fits.open(recipient_path, memmap=False) as hd_r2:
                data2 = np.asarray(hd_r2[0].data, dtype=np.float32)
            xs, ys = _sibling_detect_dao_on_image(data2, hdr, dao_sigma=dao_sigma)
            adopt = _sibling_adopt_and_confirm(
                donor_wcs,
                ra_cat,
                de_cat,
                xs,
                ys,
                naxis1,
                naxis2,
                thresholds=thresholds,
                cat_pred_flip=cat_pred_flip,
            )
            n_after = int((adopt.get("after") or {}).get("n_matched_tight") or 0)
            if n_after < flip_guard:
                return {
                    "confirmed": False,
                    "donor_filter": donor_filter,
                    "reason": f"flip/0-match after stack: n_tight={n_after} < {flip_guard}",
                    "before": adopt.get("before"),
                    "after": adopt.get("after"),
                    "stack_n": stack_n,
                }

    confirmed = bool(adopt.get("confirmed"))
    bulk = adopt.get("bulk_shift") if isinstance(adopt.get("bulk_shift"), dict) else {}
    after = adopt.get("after") if isinstance(adopt.get("after"), dict) else {}
    if confirmed:
        w_final = adopt.get("wcs")
        if isinstance(w_final, WCS):
            try:
                _write_sibling_recovered_wcs_to_fits(
                    recipient_path,
                    w_final,
                    donor_filter=str(donor_filter),
                    bulk_shift=bulk,
                    metrics=after,
                    stack_n=stack_n,
                )
            except _SiblingWcsCopyError as exc:
                # EXCEPT-FIX-3 #8: core WCS-key copy failed -> recovery aborted before any FITS
                # mutation; the frame stays unrecovered (pre-existing loud path) rather than
                # carrying a half-written WCS. See census EXC-0625.
                LOGGER.error(
                    "[SIBLING-WCS] recovery aborted for %s (%s)", recipient_path, exc
                )
                return {
                    "confirmed": False,
                    "donor_filter": donor_filter,
                    "reason": f"wcs header copy aborted: {exc}",
                    "before": adopt.get("before"),
                    "after": after,
                    "stack_n": stack_n,
                }

    log_event(
        f"SIBLING-WCS Pass2: donor={donor_filter} recipient={recipient_filter} "
        f"dx={bulk.get('dx', 0):.3f} dy={bulk.get('dy', 0):.3f} "
        f"n_tight={after.get('n_matched_tight', 0)} rms={after.get('rms_px', 'nan')} "
        f"quads={after.get('quadrants_with_match', 0)} "
        f"p_false={after.get('false_alarm_p', 'nan')} "
        f"stack={stack_n} CONFIRMED={confirmed}"
    )
    return {
        "confirmed": confirmed,
        "donor_filter": donor_filter,
        "recipient_filter": recipient_filter,
        "bulk_shift": bulk,
        "before": adopt.get("before"),
        "after": after,
        "stack_n": stack_n,
    }
