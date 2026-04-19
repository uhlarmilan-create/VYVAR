"""Utility helpers shared across the project."""

from __future__ import annotations

import glob
import math
import os
import warnings
from datetime import datetime, timezone
from pathlib import Path
from secrets import token_hex
from typing import Any

import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.wcs import FITSFixedWarning, WCS

from fits_suffixes import path_suffix_is_fits

# Optional acceleration: bottleneck nanmedian (if installed).
try:  # pragma: no cover
    import bottleneck as bn  # type: ignore
except Exception:  # noqa: BLE001
    bn = None


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


# Prefixes for celestial WCS keywords (avoid bare "PC" — would match PCOUNT).
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


def strip_celestial_wcs_keys(hdr: fits.Header) -> None:
    """Remove common celestial WCS keywords before merging a new solution."""
    for k in list(hdr.keys()):
        if any(k.startswith(p) for p in _CELESTIAL_WCS_PREFIXES):
            try:
                del hdr[k]
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
        return False


def wcs_distortion_log_suffix(header: fits.Header) -> str:
    """Short Slovak hint for diagnostics (SIP vs linear TAN)."""
    try:
        ao = header.get("A_ORDER")
        bo = header.get("B_ORDER")
        if ao is not None and int(ao) > 0:
            return f" WCS má SIP (A_ORDER={int(ao)})."
        if bo is not None and int(bo) > 0:
            return f" WCS má SIP (B_ORDER={int(bo)})."
        return (
            " WCS vyzerá ako lineárny TAN bez SIP — pri širokom poli skontroluj plate solve "
            "alebo riešenie so SIP (napr. Astrometry.net)."
        )
    except (TypeError, ValueError):
        return ""


def wcs_rotation_angle_deg(header: fits.Header) -> float | None:
    """Approximate rotation angle [deg] of the image WCS.

    Computed from the linear pixel-scale matrix of the celestial WCS. Robust to meridian flip / 180° rotation
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
        return None


def circular_angle_diff_deg(a_deg: float, b_deg: float) -> float:
    """Smallest absolute difference between angles on a circle, in degrees (0..180]."""
    a = float(a_deg) % 360.0
    b = float(b_deg) % 360.0
    d = (a - b + 180.0) % 360.0 - 180.0
    return abs(float(d))


def robust_stacking(
    file_paths: "list[str] | tuple[str, ...] | list[Path] | tuple[Path, ...]",
    *,
    sigma: float = 3.0,
    max_iters: int = 5,
    chunk_rows: int | None = None,
) -> np.ndarray:
    """Robust sigma-clipped median stack for aligned FITS frames.

    - Loads data as float32.
    - Asserts all frames have identical (H, W) shape.
    - Sigma-clips along the stack axis (axis=0), then aggregates with nanmedian.
    - Uses row-chunks to avoid MemoryError.
    """
    from astropy.stats import sigma_clip

    paths = [Path(p) for p in file_paths]
    if not paths:
        raise ValueError("robust_stacking: no file_paths")

    # Determine shape from first frame and assert identical for all.
    with fits.open(paths[0], memmap=False) as hdul0:
        arr0 = np.asarray(hdul0[0].data)
    if arr0.ndim != 2:
        raise ValueError("robust_stacking: first FITS is not 2D")
    h0, w0 = int(arr0.shape[0]), int(arr0.shape[1])
    for p in paths[1:]:
        with fits.open(p, memmap=False) as hdul:
            a = np.asarray(hdul[0].data)
        assert a.ndim == 2, f"robust_stacking: non-2D FITS: {p}"
        assert int(a.shape[0]) == h0 and int(a.shape[1]) == w0, (
            f"robust_stacking: shape mismatch for {p}: {a.shape} != {(h0, w0)}"
        )

    n = int(len(paths))
    if chunk_rows is None:
        # Heuristic: keep a chunk cube under ~256 MiB (float32).
        bytes_per_row = n * w0 * 4
        target = 256 * 1024 * 1024
        chunk_rows = int(max(8, min(h0, target // max(bytes_per_row, 1))))

    out = np.zeros((h0, w0), dtype=np.float32)
    eps = 1e-12
    s = float(sigma)
    it = int(max_iters)

    for y0 in range(0, h0, int(chunk_rows)):
        y1 = int(min(h0, y0 + int(chunk_rows)))
        cube = np.empty((n, y1 - y0, w0), dtype=np.float32)
        for i, p in enumerate(paths):
            with fits.open(p, memmap=False) as hdul:
                cube[i] = np.asarray(hdul[0].data[y0:y1, :], dtype=np.float32)

        clipped = sigma_clip(
            cube,
            sigma=s,
            maxiters=it,
            axis=0,
            cenfunc="median",
            stdfunc="std",
            masked=True,
            copy=False,
        )
        data = clipped.filled(np.nan).astype(np.float32, copy=False)
        if bn is not None:
            out[y0:y1, :] = bn.nanmedian(data, axis=0).astype(np.float32, copy=False)
        else:
            out[y0:y1, :] = np.nanmedian(data, axis=0).astype(np.float32, copy=False)
        # Explicitly free large temporaries each chunk (helps long stacks on Windows).
        try:
            del cube, clipped, data
        except Exception:  # noqa: BLE001
            pass
        try:
            import gc

            gc.collect()
        except Exception:  # noqa: BLE001
            pass

    return out


def mean_stacking(
    file_paths: "list[str] | tuple[str, ...] | list[Path] | tuple[Path, ...]",
    *,
    chunk_rows: int | None = None,
) -> np.ndarray:
    """Plain mean (average) stack of aligned FITS frames along the stack axis.

    - Loads data as float32.
    - Asserts all frames have identical (H, W) shape.
    - Aggregates with ``nanmean`` along axis 0 (NaNs ignored per pixel).
    - Uses row-chunks to avoid MemoryError (same heuristic as :func:`robust_stacking`).
    """
    paths = [Path(p) for p in file_paths]
    if not paths:
        raise ValueError("mean_stacking: no file_paths")

    with fits.open(paths[0], memmap=False) as hdul0:
        arr0 = np.asarray(hdul0[0].data)
    if arr0.ndim != 2:
        raise ValueError("mean_stacking: first FITS is not 2D")
    h0, w0 = int(arr0.shape[0]), int(arr0.shape[1])
    for p in paths[1:]:
        with fits.open(p, memmap=False) as hdul:
            a = np.asarray(hdul[0].data)
        assert a.ndim == 2, f"mean_stacking: non-2D FITS: {p}"
        assert int(a.shape[0]) == h0 and int(a.shape[1]) == w0, (
            f"mean_stacking: shape mismatch for {p}: {a.shape} != {(h0, w0)}"
        )

    n = int(len(paths))
    if chunk_rows is None:
        bytes_per_row = n * w0 * 4
        target = 256 * 1024 * 1024
        chunk_rows = int(max(8, min(h0, target // max(bytes_per_row, 1))))

    out = np.zeros((h0, w0), dtype=np.float32)

    for y0 in range(0, h0, int(chunk_rows)):
        y1 = int(min(h0, y0 + int(chunk_rows)))
        cube = np.empty((n, y1 - y0, w0), dtype=np.float32)
        for i, p in enumerate(paths):
            with fits.open(p, memmap=False) as hdul:
                cube[i] = np.asarray(hdul[0].data[y0:y1, :], dtype=np.float32)

        if bn is not None:
            out[y0:y1, :] = bn.nanmean(cube, axis=0).astype(np.float32, copy=False)
        else:
            out[y0:y1, :] = np.nanmean(cube, axis=0).astype(np.float32, copy=False)
        try:
            del cube
        except Exception:  # noqa: BLE001
            pass
        try:
            import gc

            gc.collect()
        except Exception:  # noqa: BLE001
            pass

    return out


def effective_binned_pixel_pitch_um(*, base_pixel_um_1x1: float, binning: int) -> float:
    """Physical pixel pitch of one **image** pixel after symmetric binning.

    ``base_pixel_um_1x1`` is the camera datasheet pitch at 1×1 (e.g. from EQUIPMENTS.PIXELSIZE).
    For 2×2 binning each output pixel spans 2×2 sensor pixels → pitch doubles.
    """
    b = max(1, int(binning))
    return float(base_pixel_um_1x1) * float(b)


def fits_binning_xy_from_header(header: fits.Header) -> tuple[int, int]:
    """Read ``XBINNING`` / ``YBINNING`` (fallback ``BINNING``, symmetric Y=X) from a primary header."""
    def _one(raw: object, default: int = 1) -> int:
        try:
            return max(1, int(float(raw)))
        except (TypeError, ValueError):
            return max(1, int(default))

    xb = _one(header.get("XBINNING", header.get("BINNING", 1)), 1)
    yb = _one(header.get("YBINNING", xb), xb)
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


# Plate scale [arcsec/pixel] = (pixel_pitch_um / focal_length_mm) * K; K = 206.265 (≈ rad→″ / 1000 for µm/mm).
PLATE_SCALE_ARCSEC_PER_UM_OVER_MM: float = 206.265

# Minimum great-circle cone radius [deg] for Gaia queries and overlays. Undersized cones (often from a
# bad WCS scale that collapses corner separations) produce a visible „catalog disc“ in QA; wide chips at ~200 mm
# need several degrees to cover the diagonal.
MIN_GAIA_CONE_RADIUS_DEG: float = 3.5

# Astrometry.net SIP / distortion: ``tweak_order`` in the nova API and ``--tweak-order`` for ``solve-field``.
# Order 3 helps wide-field refractors (e.g. 200 mm) where a pure TAN/CD solution leaves edge residuals vs SIPS.
ASTROMETRY_NET_TWEAK_ORDER: int = 3

# Local ``solve-field`` (ANSVR / astrometry.net index bundle): ``--cpulimit`` in seconds per field.
ASTROMETRY_SOLVE_FIELD_CPULIMIT_SEC: int = 30


def effective_astrometry_net_tweak_order() -> int:
    """Effective SIP ``tweak_order`` / ``--tweak-order``. Default 3; env ``VYVAR_ASTROMETRY_TWEAK_ORDER`` may raise it (3–6), never below 3."""
    v = (os.environ.get("VYVAR_ASTROMETRY_TWEAK_ORDER") or "").strip()
    if not v:
        return int(ASTROMETRY_NET_TWEAK_ORDER)
    try:
        o = int(v)
    except ValueError:
        return int(ASTROMETRY_NET_TWEAK_ORDER)
    return max(3, min(6, o))


# photutils ``DAOStarFinder`` has no ``roundness_limit=(None, None)``; extreme ``roundlo``/``roundhi`` disable filtering.
# Comatic / corner stars are non-round and would be rejected with defaults (-1, 1).
DAO_STAR_FINDER_NO_ROUNDNESS_FILTER: dict[str, float] = {"roundlo": -1.0e9, "roundhi": 1.0e9}


def dao_detection_fwhm_pixels(header: fits.Header | None, *, configured_fallback: float | None) -> float | None:
    """DAO centroid kernel from header FWHM, or fallback.

    - If ``VY_FWHM`` exists: returns ``VY_FWHM`` (clamped to a sensible range).
    - Else: returns ``max(3, configured_fallback)`` when fallback is provided.
    - If fallback is ``None`` and header has no usable value: returns ``None``.
    """
    measured: float | None = None
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
    """Cone radius [deg] = (FOV diameter / 2) × 1.2 (20 % margin over half-angle to edge)."""
    d = float(fov_diameter_deg)
    if not math.isfinite(d) or d <= 0:
        return 0.0
    return float(0.5 * d * 1.2)


def plate_scale_arcsec_per_pixel(*, pixel_pitch_um: float, focal_length_mm: float) -> float | None:
    """Small-angle plate scale [arcsec/pixel] from pixel pitch [µm] and focal length [mm].

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
    sensible floor (≥ ``MIN_GAIA_CONE_RADIUS_DEG``, default 3.5°).
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
        dyn_floor = max(r_floor, min(6.0, 600.0 / f_mm))
        return float(max(0.08, max(r, dyn_floor)))
    fb = max(0.05, float(fov_diameter_fallback_deg))
    return float(max(r_floor, fb * 0.65))


def astrometry_net_scale_bounds_arcsec_per_pix(scale_arcsec_per_px: float) -> tuple[float, float]:
    """Upper/lower scale for Astrometry.net (``scale_lower`` / ``scale_upper``), arcsec/pixel.

    Band is proportional to the nominal scale so that e.g. ``s ≈ 9.55`` maps to about ``8 … 11``
    (same relative width as ``8/9.55`` and ``11/9.55``). Falls back to a wider band if needed.
    """
    s = float(scale_arcsec_per_px)
    if not math.isfinite(s) or s <= 0:
        return 0.03, 180.0
    lo = s * (8.0 / 9.55)
    hi = s * (11.0 / 9.55)
    lo = max(0.03, lo)
    hi = min(180.0, hi)
    if hi <= lo:
        lo = max(0.03, s * 0.52)
        hi = min(180.0, s * 1.75)
        if hi <= lo:
            hi = min(180.0, lo * 1.25)
    return float(lo), float(hi)


def masterstar_wcs_quality(
    wcs: WCS,
    expected_scale_arcsec: float,
    *,
    anisotropy_limit: float | None = None,
) -> dict[str, Any]:
    """Heuristic WCS sanity vs expected plate scale [arcsec/pix] (MASTERSTAR diagnostics)."""
    _aniso_default = 1.3
    try:
        pm = wcs.pixel_scale_matrix
        sx = abs(pm[0, 0]) * 3600
        sy = abs(pm[1, 1]) * 3600
        ratio = max(sx, sy) / max(min(sx, sy), 0.001)
        mean_scale = (sx + sy) / 2.0
        scale_err = abs(mean_scale - expected_scale_arcsec) / max(expected_scale_arcsec, 1e-9) * 100.0
        _lim = float(anisotropy_limit) if anisotropy_limit is not None else float(_aniso_default)
        if not math.isfinite(_lim) or _lim <= 0:
            _lim = float(_aniso_default)
        ok = ratio <= _lim and scale_err < 20.0
        return {
            "ok": ok,
            "ratio": ratio,
            "scale_x": sx,
            "scale_y": sy,
            "mean_scale": mean_scale,
            "scale_err_pct": scale_err,
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "ratio": 999.0, "error": str(exc)}


def maybe_rescale_linear_wcs_cd_to_target_arcsec_per_pixel(
    wcs_in: WCS,
    target_arcsec_per_pixel: float,
    *,
    trigger_relative_mismatch: float = 0.22,
    min_scale_factor: float = 1.0 / 15.0,
    max_scale_factor: float = 15.0,
) -> tuple[WCS, bool]:
    """Scale CD (or CDELT) uniformly so mean proj-plane scale matches ``target``; position angle unchanged.

    Only for **linear** WCS (no SIP on ``wcs_in``). Use when a wrong index solution yields ~N× plate
    scale vs optics, without changing CRVAL/CRPIX/rotation.
    """
    tgt = float(target_arcsec_per_pixel)
    if not math.isfinite(tgt) or tgt <= 0:
        return wcs_in, False
    if wcs_in.sip is not None:
        return wcs_in, False
    try:
        w = wcs_in.deepcopy()
        scales = w.proj_plane_pixel_scales()
        # Angles per pixel (effectively arcsec/px); ``to(u.arcsec)`` works from deg-based quantities.
        sx = abs(float(scales[0].to(u.arcsec).value))
        sy = abs(float(scales[1].to(u.arcsec).value))
        mean_s = 0.5 * (sx + sy)
    except Exception:  # noqa: BLE001
        return wcs_in, False
    if not math.isfinite(mean_s) or mean_s <= 0:
        return wcs_in, False
    rel = abs(mean_s / tgt - 1.0)
    if rel <= trigger_relative_mismatch:
        return wcs_in, False
    fac = tgt / mean_s
    if not math.isfinite(fac) or fac <= 0 or fac < min_scale_factor or fac > max_scale_factor:
        return wcs_in, False
    try:
        cd = w.wcs.cd
        if cd is not None:
            arr = np.asarray(cd, dtype=np.float64)
            if arr.size >= 4 and np.any(arr != 0):
                w.wcs.cd = arr * fac
                return w, True
    except Exception:  # noqa: BLE001
        pass
    try:
        if w.wcs.cdelt is not None:
            w.wcs.cdelt = np.asarray(w.wcs.cdelt, dtype=np.float64) * fac
            return w, True
    except Exception:  # noqa: BLE001
        pass
    return wcs_in, False


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


def generate_session_id(ts: datetime | None = None, random_bytes: int = 3) -> str:
    """Generate a unique session id in form YYYYMMDD_#a1b2c3."""
    timestamp = ts or datetime.now(timezone.utc)
    day_str = timestamp.strftime("%Y%m%d")
    suffix = token_hex(random_bytes)
    return f"{day_str}_#{suffix}"


def session_paths(archive_root: Path, session_id: str) -> dict[str, Path]:
    """Return canonical directories for one observing session."""
    base = archive_root / session_id
    return {
        "base": base,
        "raw": base / "Raw",
        "masters": base / "Masters",
        "output": base / "Output",
    }


