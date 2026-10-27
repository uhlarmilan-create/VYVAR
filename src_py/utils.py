"""Utility helpers shared across the project."""

from __future__ import annotations

import contextlib
import glob
import math
import os
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.wcs import FITSFixedWarning, WCS

from fits_suffixes import path_suffix_is_fits

# Shared deterministic RNG seed (solver RANSAC uses 42; astroalign patched below).
VYVAR_RANDOM_SEED: int = 42


@contextlib.contextmanager
def seeded_numpy_default_rng(seed: int | None = None):
    """Pin ``numpy.random.default_rng()`` for libraries that call it without a seed.

    astroalign 2.6.x RANSAC uses ``np.random.default_rng().shuffle(...)``, which is
    nondeterministic across runs. Patch only the no-argument call used by astroalign.
    """
    s = int(seed if seed is not None else VYVAR_RANDOM_SEED)
    rng = np.random.default_rng(s)
    orig = np.random.default_rng

    def _patched(*args: Any, **kwargs: Any) -> np.random.Generator:
        if len(args) == 0 and not kwargs:
            return rng
        return orig(*args, **kwargs)

    np.random.default_rng = _patched  # type: ignore[method-assign, assignment]
    try:
        yield rng
    finally:
        np.random.default_rng = orig  # type: ignore[method-assign, assignment]


def iter_fits_paths_recursive(root: Path | str) -> list[Path]:
    """Return sorted FITS paths under ``root`` (recursive), including nested layouts (e.g. ``filter_exp_binning``).

    Uses :func:`glob.glob` with ``**`` so behavior matches explicit recursive globbing (not only top-level).
    """
    root = Path(root)
    if not root.exists():
        return []
    root_s = os.path.normpath(str(root.resolve()))
    patterns = (
        os.path.join(root_s, "**", "*.fits"),
        os.path.join(root_s, "**", "*.fit"),
        os.path.join(root_s, "**", "*.fts"),
        os.path.join(root_s, "**", "*.FITS"),
        os.path.join(root_s, "**", "*.FIT"),
        os.path.join(root_s, "**", "*.FTS"),
    )
    seen: set[str] = set()
    out: list[Path] = []
    for pat in patterns:
        for p in glob.glob(pat, recursive=True):
            fp = Path(p)
            if not fp.is_file():
                continue
            if not path_suffix_is_fits(fp):
                continue
            key = str(fp.resolve()).casefold()
            if key in seen:
                continue
            seen.add(key)
            out.append(fp.resolve())
    return sorted(out)


# Prefixes for celestial WCS keywords (avoid bare "PC" - would match PCOUNT).
_CELESTIAL_WCS_PREFIXES: tuple[str, ...] = (
    "CRPIX",
    "CRVAL",
    "CDELT",
    "CTYPE",
    "CUNIT",
    "CD1_",
    "CD2_",
    "PC1_",
    "PC2_",
    "PC3_",
    "PV1_",
    "PV2_",
    "A_",
    "B_",
    "AP_",
    "BP_",
    "RADESYS",
    "EQUINOX",
    "LONPOLE",
    "LATPOLE",
)


def set_float_image_data_in_place(hdu: Any, data: Any) -> None:
    """FITS-SCALE-2026-10 (Milan 2026-10-13): replace the pixel data of an HDU opened with
    ``mode="update"`` by physical float32 values.

    astropy keeps BZERO/BSCALE of an integer original when float data are assigned in update
    mode and stores the values unscaled, so every in-place write adds BZERO again (16-bit
    lights with BZERO=32768 -> +32768 ADU per write; draft 532: sky 1664 -> 34 100 ADU, the
    second run 66 800). The scaling keywords are removed: the stored floats ARE the values."""
    import numpy as _np  # noqa: PLC0415

    for _k in ("BZERO", "BSCALE", "BLANK"):
        if _k in hdu.header:
            del hdu.header[_k]
    hdu.data = _np.asarray(data, dtype=_np.float32)


def strip_celestial_wcs_keys(hdr: fits.Header) -> None:
    """Remove common celestial WCS keywords before merging a new solution."""
    for k in list(hdr.keys()):
        if any(k.startswith(p) for p in _CELESTIAL_WCS_PREFIXES):
            try:
                del hdr[k]
            except KeyError:
                pass


# Leftovers from ASTAP / other tools - not part of the FITS WCS standard; safe to drop after VYVAR solve.
_VENDOR_PLATESOLVE_METADATA_KEYS: tuple[str, ...] = (
    "PLTSOLVD",  # ASTAP: plate-solved flag
)


def strip_vendor_platesolve_metadata(hdr: fits.Header) -> None:
    """Remove third-party plate-solve tool keywords (does **not** remove celestial WCS keys)."""
    for _k in _VENDOR_PLATESOLVE_METADATA_KEYS:
        try:
            del hdr[_k]
        except KeyError:
            pass


def header_key_is_celestial_wcs(k: str) -> bool:
    """True if FITS keyword ``k`` belongs to the celestial WCS block we strip/copy."""
    return any(k.startswith(p) for p in _CELESTIAL_WCS_PREFIXES)


def fits_header_has_celestial_wcs(header: fits.Header) -> bool:
    """True if ``header`` defines a usable celestial WCS (Astropy)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            w = WCS(header)
        return bool(w.has_celestial)
    except Exception:  # noqa: BLE001
        # EXC-0563: T4 -- intent unclear (w = WCS(header) / return bool(w.has_celestial) / except Exception:  # n... (EXCEPT-BULK 2026-07-08)
        return False


def wcs_rotation_angle_deg(header: fits.Header) -> float | None:
    """Approximate rotation angle [deg] of the image WCS.

    Computed from the linear pixel-scale matrix of the celestial WCS. Robust to meridian flip / 180 deg rotation
    without relying on FITS "FLIP" metadata.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            w = WCS(header)
        if not w.has_celestial:
            return None
        wc = w.celestial
        m = np.asarray(wc.pixel_scale_matrix, dtype=np.float64)
        if m.shape != (2, 2) or (not np.all(np.isfinite(m))):
            return None
        ang = math.degrees(math.atan2(float(m[0, 1]), float(m[0, 0])))
        a = float(ang % 360.0)
        return a
    except Exception:  # noqa: BLE001
        # EXC-0564: T4 -- intent unclear (a = float(ang % 360.0) / return a / except Exception:  # noqa: BLE001 /... (EXCEPT-BULK 2026-07-08)
        return None


def circular_angle_diff_deg(a_deg: float, b_deg: float) -> float:
    """Smallest absolute difference between angles on a circle, in degrees (0..180]."""
    a = float(a_deg) % 360.0
    b = float(b_deg) % 360.0
    d = (a - b + 180.0) % 360.0 - 180.0
    return abs(float(d))


def effective_binned_pixel_pitch_um(*, base_pixel_um_1x1: float, binning: int) -> float:
    """Physical pixel pitch of one **image** pixel after symmetric binning.

    ``base_pixel_um_1x1`` is the camera datasheet pitch at 1x1 (e.g. from EQUIPMENTS.PIXELSIZE).
    For 2x2 binning each output pixel spans 2x2 sensor pixels -> pitch doubles.
    """
    b = max(1, int(binning))
    return float(base_pixel_um_1x1) * float(b)


def parse_fits_binning_int(raw: object, default: int = 1) -> int:
    """Parse FITS binning: ``2``, ``2.0``, ``"2x2"``, ``"2 x 2"`` -> int; else *default*."""
    if raw is None:
        return max(1, int(default))
    if isinstance(raw, (int, np.integer)):
        return max(1, int(raw))
    if isinstance(raw, (float, np.floating)):
        if math.isfinite(float(raw)) and float(raw) > 0:
            return max(1, int(round(float(raw))))
        return max(1, int(default))
    s = str(raw).strip().lower().replace(" ", "")
    m = re.match(r"^(\d+)[xx](\d+)$", s)
    if m:
        return max(1, int(m.group(1)))
    try:
        return max(1, int(round(float(raw))))
    except (TypeError, ValueError):
        return max(1, int(default))


def parse_sensor_naxis_from_text(sensor_size: str | None) -> tuple[int, int] | None:
    """Parse ``EQUIPMENTS.SENSORSIZE`` like ``6280x4176`` -> (width, height)."""
    if not sensor_size or not str(sensor_size).strip():
        return None
    m = re.search(r"(\d+)\s*[xx]\s*(\d+)", str(sensor_size), flags=re.IGNORECASE)
    if not m:
        return None
    w, h = int(m.group(1)), int(m.group(2))
    return (w, h) if w > 0 and h > 0 else None


def infer_binning_xy_from_sensor_shape(
    naxis1: int,
    naxis2: int,
    native_size: tuple[int, int] | None,
    header_xy: tuple[int, int],
) -> tuple[int, int, bool]:
    """If header says 1x1 but image size ~ native/k, infer integer binning *k* (2, 3, 4)."""
    hx, hy = max(1, int(header_xy[0])), max(1, int(header_xy[1]))
    if hx > 1 or hy > 1:
        return hx, hy, False
    if native_size is None or naxis1 <= 0 or naxis2 <= 0:
        return hx, hy, False
    nw, nh = int(native_size[0]), int(native_size[1])
    if nw <= 0 or nh <= 0:
        return hx, hy, False
    tol = max(64, int(0.02 * max(nw, nh)))
    for k in (4, 3, 2):
        ew, eh = nw // k, nh // k
        if abs(int(naxis1) - ew) <= tol and abs(int(naxis2) - eh) <= tol:
            return k, k, True
    return hx, hy, False


def fits_binning_xy_from_header(header: fits.Header) -> tuple[int, int]:
    """Read ``XBINNING`` / ``YBINNING`` (fallback ``BINNING``, symmetric Y=X) from a primary header."""
    raw_x = header.get("XBINNING", header.get("BINNING", 1))
    xb = parse_fits_binning_int(raw_x, 1)
    yb = parse_fits_binning_int(header.get("YBINNING", xb), xb)
    return xb, yb


def normalize_telescope_focal_mm_for_plate_scale(raw_mm: float) -> tuple[float, bool]:
    """Correct a common data-entry error: focal length stored as **2000** instead of **200** mm.

    Only a narrow band is adjusted so real ~2 m systems (e.g. 2000 mm) are not rewritten blindly.
    Returns ``(focal_mm, was_corrected)``.
    """
    if not math.isfinite(raw_mm) or raw_mm <= 0:
        return float(raw_mm), False
    if 1750.0 <= raw_mm <= 2250.0:
        v10 = raw_mm / 10.0
        if 120.0 <= v10 <= 380.0:
            return float(v10), True
    return float(raw_mm), False


# Plate scale [arcsec/pixel] = (pixel_pitch_um / focal_length_mm) * K; K = 206.265 (~ rad-> arcsec / 1000 for um/mm).
PLATE_SCALE_ARCSEC_PER_UM_OVER_MM: float = 206.265

# Minimum great-circle cone radius [deg] for Gaia queries and overlays. Undersized cones (often from a
# bad WCS scale that collapses corner separations) produce a visible 'catalog disc' in QA; wide chips at ~200 mm
# need several degrees to cover the diagonal.
MIN_GAIA_CONE_RADIUS_DEG: float = 3.5


# photutils ``DAOStarFinder`` (>=3): ``roundness_range`` / ``sharpness_range`` disable shape
# filtering when set to extreme values; ``min_separation=0`` freezes 2.x detection (DEPS-CYCLE-2).
# Comatic / corner stars are non-round and would be rejected with default roundness (-1, 1).
DAO_STAR_FINDER_NO_ROUNDNESS_FILTER: dict[str, float | tuple[float, float]] = {
    "roundness_range": (-1.0e9, 1.0e9),
    "sharpness_range": (0.0, 2.0),
    "min_separation": 0.0,
}


def dao_detection_fwhm_pixels(header: fits.Header | None, *, configured_fallback: float | None) -> float | None:
    """DAO centroid kernel from header FWHM, or fallback.

    - If ``VY_FWHM`` exists: returns ``VY_FWHM`` (clamped to a sensible range).
    - Else: returns ``max(3, configured_fallback)`` when fallback is provided.
    - If fallback is ``None`` and header has no usable value: returns ``None``.
    """
    measured: float | None = None
    # Safe: header key checked with `in header` before access.
    if header is not None and "VY_FWHM" in header:
        try:
            v = float(header["VY_FWHM"])
            if math.isfinite(v) and 0.5 < v < 80.0:
                measured = v
        except (TypeError, ValueError):
            pass
    if measured is not None:
        return float(max(1.2, min(20.0, measured)))
    if configured_fallback is None:
        return None
    cf = float(configured_fallback)
    if not math.isfinite(cf) or cf <= 0:
        cf = 4.5
    return float(max(3.0, cf))


def catalog_cone_radius_from_fov_diameter_deg(fov_diameter_deg: float) -> float:
    """Cone radius [deg] = (FOV diameter / 2) x 1.2 (20 % margin over half-angle to edge)."""
    d = float(fov_diameter_deg)
    if not math.isfinite(d) or d <= 0:
        return 0.0
    return float(0.5 * d * 1.2)


def plate_scale_arcsec_per_pixel(*, pixel_pitch_um: float, focal_length_mm: float) -> float | None:
    """Small-angle plate scale [arcsec/pixel] from pixel pitch [um] and focal length [mm].

    ``scale = (pixel_pitch_um / focal_mm) * 206.265``. Focal length is passed through
    :func:`normalize_telescope_focal_mm_for_plate_scale` so a common ``2000`` vs ``200`` mm typo
    cannot slip through on this path.
    """
    if pixel_pitch_um <= 0:
        return None
    foc_n, _ = normalize_telescope_focal_mm_for_plate_scale(float(focal_length_mm))
    if foc_n <= 0:
        return None
    return (float(pixel_pitch_um) / float(foc_n)) * PLATE_SCALE_ARCSEC_PER_UM_OVER_MM


def per_frame_catalog_match_sep_arcsec_for_scale(
    arcsec_per_px: float | None,
    *,
    floor_arcsec: float = 3.0,
    fallback_arcsec: float = 20.0,
) -> float:
    """~2.5 px matching tolerance in arcseconds; ``fallback_arcsec`` when scale is unknown."""
    if arcsec_per_px is None or not math.isfinite(arcsec_per_px) or arcsec_per_px <= 0:
        return float(fallback_arcsec)
    return max(float(floor_arcsec), float(arcsec_per_px) * 2.5)


def get_optimal_params(
    *,
    focal_length_mm: float | None,
    pixel_size_um: float | None,
    binning: int = 1,
    naxis1: int | None = None,
    naxis2: int | None = None,
    fov_diameter_deg: float | None = None,
) -> dict[str, float]:
    """Heuristic defaults for star detection / catalog search by optics.

    Returns:
    - ``fwhm``: DAO kernel FWHM in pixels (wide-field / bigger effective pixels -> 4.5)
    - ``detection_threshold``: sigma threshold (default 1.5)
    - ``search_radius``: catalog cone radius [deg] with +20 % margin
    """
    b = max(1, int(binning or 1))
    px_native = float(pixel_size_um) if pixel_size_um is not None else float("nan")
    eff_um = (
        effective_binned_pixel_pitch_um(base_pixel_um_1x1=px_native, binning=b)
        if math.isfinite(px_native) and px_native > 0
        else float("nan")
    )
    foc = float(focal_length_mm) if focal_length_mm is not None else float("nan")

    # Wide-field / larger pixels: broader PSF footprint in pixels -> use 4.5 as requested baseline.
    if (math.isfinite(foc) and foc <= 300.0) or (math.isfinite(eff_um) and eff_um >= 8.0):
        fwhm = 4.5
    else:
        fwhm = 3.5
    detection_threshold = 1.5

    if (
        naxis1 is not None
        and naxis2 is not None
        and math.isfinite(foc)
        and foc > 0
        and math.isfinite(eff_um)
        and eff_um > 0
    ):
        sr = catalog_cone_radius_deg_from_optics(
            naxis1=int(naxis1),
            naxis2=int(naxis2),
            pixel_pitch_um=float(eff_um),
            focal_length_mm=float(foc),
            margin=1.2,
            fov_diameter_fallback_deg=float(fov_diameter_deg or 1.0),
        )
    else:
        sr = catalog_cone_radius_from_fov_diameter_deg(float(fov_diameter_deg or 1.0))
    return {
        "fwhm": float(fwhm),
        "detection_threshold": float(detection_threshold),
        "search_radius": float(sr),
    }


def catalog_cone_radius_deg_from_optics(
    *,
    naxis1: int,
    naxis2: int,
    pixel_pitch_um: float,
    focal_length_mm: float,
    margin: float = 1.3,
    fov_diameter_fallback_deg: float = 1.0,
) -> float:
    """Great-circle radius [deg] from field centre to chip corner (diagonal), then ``margin`` (default +30%).

    Uses chip diagonal directly: ``hypot(naxis1, naxis2)`` with plate scale [arcsec/px], then half-diagonal
    converted to degrees and padded by ``margin``. This guarantees query cone >= full rectangular sensor field.

    If focal length or pixel pitch are non-positive, falls back to ``fov_diameter_fallback_deg`` with a
    sensible floor (>= ``MIN_GAIA_CONE_RADIUS_DEG``, default 3.5 deg).
    """
    wpx = max(1, int(naxis1))
    hpx = max(1, int(naxis2))
    f_mm = float(focal_length_mm)
    p_um = float(pixel_pitch_um)
    r_floor = float(MIN_GAIA_CONE_RADIUS_DEG)
    if math.isfinite(f_mm) and f_mm > 0 and math.isfinite(p_um) and p_um > 0:
        sc = plate_scale_arcsec_per_pixel(pixel_pitch_um=p_um, focal_length_mm=f_mm)
        if sc is None or (not math.isfinite(float(sc))) or float(sc) <= 0:
            sc = 0.0
        fov_x_deg = (float(wpx) * float(sc)) / 3600.0
        fov_y_deg = (float(hpx) * float(sc)) / 3600.0
        # Required formula: radius = 0.5 * hypot(FOV_x, FOV_y) * padding.
        r = 0.5 * float(math.hypot(fov_x_deg, fov_y_deg)) * float(margin)
        # Dynamic floor: wide-field (short focal) needs a larger Gaia cone by default,
        # but narrow-field optics must not be forced to the global MIN_GAIA_CONE_RADIUS_DEG (3.5 deg),
        # otherwise the catalog becomes too large and triangle matching becomes intractable.
        #
        # For long focal lengths: floor ~ (600 / focal_mm) deg, with a small absolute minimum.
        dyn_floor = max(0.08, min(float(r_floor), 600.0 / f_mm))
        return float(max(0.08, max(r, float(dyn_floor))))
    fb = max(0.05, float(fov_diameter_fallback_deg))
    return float(max(r_floor, fb * 0.65))


def estimate_field_diameter_deg_diagonal(
    *,
    naxis1: int,
    naxis2: int,
    scale_x_arcsec_per_px: float,
    scale_y_arcsec_per_px: float,
) -> float:
    """Rough field diameter along chip diagonal (different X/Y scales allowed)."""
    import math

    w = max(1, int(naxis1))
    h = max(1, int(naxis2))
    diag_arcsec = math.hypot(w * float(scale_x_arcsec_per_px), h * float(scale_y_arcsec_per_px))
    return float(diag_arcsec / 3600.0)


