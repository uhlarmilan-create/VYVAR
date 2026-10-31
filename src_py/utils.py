"""Utility helpers shared across the project."""

from __future__ import annotations

import contextlib
import math
import os
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.wcs import FITSFixedWarning, WCS

from aperture_policy import fwhm_from_header_vy_fwhm
from fits_suffixes import list_fits_files

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
    """Return sorted, resolved FITS paths under ``root`` (recursive), including nested layouts (e.g. ``filter_exp_binning``).

    G13-31: one walk with the shared suffix rule (:func:`fits_suffixes.list_fits_files`); paths that resolve to
    the same file (symlinks) are listed once, keyed by ``os.path.normcase`` (case-insensitive only where the OS
    is), never by case-folding the names.
    """
    root = Path(root)
    if not root.exists():
        return []
    seen: set[str] = set()
    out: list[Path] = []
    for fp in list_fits_files(root.resolve(), recursive=True):
        rp = fp.resolve()
        key = os.path.normcase(str(rp))
        if key in seen:
            continue
        seen.add(key)
        out.append(rp)
    return sorted(out)


# G03-21: the ONE list of WCS keywords (astrosolve, utils, pipeline_astrometry, alignment, OSC).
# Exact patterns, not prefixes: the former "A_" / "B_" prefixes also deleted user keys such as B_V or A_MAG,
# and bare "PC" would match PCOUNT. [A-Z]? = alternate-WCS letter (Greisen & Calabretta 2002, Paper I).
# Celestial solution keys: copied from a reference header and stripped before a new solution; SIP
# (A/B/AP/BP_ORDER, _DMAX, _p_q; Shupe et al. 2005) and TPV / PV distortion terms included.
_CELESTIAL_WCS_KEY_RE = re.compile(
    r"^(?:(?:CRPIX|CRVAL|CDELT|CTYPE|CUNIT)\d+[A-Z]?|(?:CD|PC|PV|PS)\d+_\d+[A-Z]?"
    r"|(?:A|B|AP|BP)_(?:ORDER|DMAX|\d+_\d+)|(?:RADESYS|RADECSYS|EQUINOX|LONPOLE|LATPOLE)[A-Z]?)$"
)
# Structural / legacy keys of a solution: stripped with it but never copied (WCSAXES must precede the other
# WCS keys of a header, so it is not appended to another header; a stale CROTA2 next to a new CD matrix makes
# the WCS ambiguous). Time keys (MJDREF, DATEREF, DATE-OBS, MJD-OBS, ...) are not WCS solution keys and stay
# (Rots et al. 2015, WCS Paper IV).
_WCS_STRUCTURAL_KEY_RE = re.compile(r"^(?:WCSAXES[A-Z]?|WCSDIM|WCSNAME[A-Z]?|CROTA\d+)$")


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
    """Remove the WCS solution (celestial + structural keys, :func:`header_key_is_wcs`) before a new one."""
    for k in list(hdr.keys()):
        if header_key_is_wcs(k):
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
    """True if FITS keyword ``k`` belongs to the celestial WCS solution block (copied between headers)."""
    return bool(_CELESTIAL_WCS_KEY_RE.match(str(k)))


def header_key_is_wcs(k: str) -> bool:
    """True if FITS keyword ``k`` is stripped before a new solution: celestial + structural keys."""
    return header_key_is_celestial_wcs(k) or bool(_WCS_STRUCTURAL_KEY_RE.match(str(k)))


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
    m = re.match("^(\\d+)[*x\u00d7](\\d+)$", s)  # G13-07: "2x2", "2*2", "2 X 2"
    if m:
        return max(1, int(m.group(1)))
    try:
        return max(1, int(round(float(raw))))
    except (TypeError, ValueError):
        return max(1, int(default))


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


# photutils ``DAOStarFinder`` (>=3): ``roundness_range`` / ``sharpness_range`` disable shape
# filtering when set to extreme values; ``min_separation=0`` freezes 2.x detection (DEPS-CYCLE-2).
# Comatic / corner stars are non-round and would be rejected with default roundness (-1, 1).
DAO_STAR_FINDER_NO_ROUNDNESS_FILTER: dict[str, float | tuple[float, float]] = {
    "roundness_range": (-1.0e9, 1.0e9),
    "sharpness_range": (0.0, 2.0),
    "min_separation": 0.0,
}


def dao_detection_fwhm_pixels(header: fits.Header | None, *, configured_fallback: float | None) -> float | None:
    """DAO centroid kernel FWHM [px]: header ``VY_FWHM`` (the one reader,
    :func:`aperture_policy.fwhm_from_header_vy_fwhm`, G03-29) clamped to 1.2..20 px, else
    ``max(3, configured_fallback)``; None when neither is usable (no 4.5 px default).
    """
    measured = fwhm_from_header_vy_fwhm(header)
    if measured is not None:
        return float(max(1.2, min(20.0, measured)))
    if configured_fallback is None:
        return None
    cf = float(configured_fallback)
    if not math.isfinite(cf) or cf <= 0:
        return None
    return float(max(3.0, cf))


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


def norm_fits_path_key(path: Path | str) -> str:
    """Normalized absolute path key for qc_metrics / allowlist joins (casefold)."""
    try:
        return str(Path(path).resolve()).casefold()
    except OSError:
        return str(path).casefold()


def load_qc_metrics_status_by_path(qc_csv: Path | str) -> dict[str, str]:
    """Map normalized absolute FITS path -> ``status`` from ``qc_metrics.csv``."""
    import pandas as pd  # noqa: PLC0415

    p = Path(qc_csv)
    if not p.is_file():
        raise FileNotFoundError(f"qc_metrics.csv not found: {p}")
    df = pd.read_csv(p)
    if "status" not in df.columns:
        raise ValueError(f"qc_metrics.csv missing status column: {p}")
    src_col = "src" if "src" in df.columns else "dst"
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        raw = row.get(src_col)
        if raw is None or (isinstance(raw, float) and not math.isfinite(float(raw))):
            continue
        out[norm_fits_path_key(str(raw))] = str(row["status"]).strip()
    return out


#: EXPOSURE-ONE-2026-10: header keywords of the exposure duration, in priority order (one authority).
EXPOSURE_KEYWORDS: tuple[str, ...] = ("EXPTIME", "EXPOSURE", "EXPOSURE0", "EXP_TIME", "EXPTIM")


def exposure_seconds(hdr: Any) -> float | None:
    """Exposure duration [s] of one frame from its FITS header (EXPOSURE_KEYWORDS), None when absent.

    The single reader of the exposure (code review 2 G06-03 / G08-02): never parsed from a setup or
    folder name such as ``g_120_4`` or ``g|120|4``."""
    if hdr is None:
        return None
    for k in EXPOSURE_KEYWORDS:
        v = hdr.get(k)
        if v is None:
            continue
        try:
            sec = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(sec) and sec >= 0.0:
            return sec
    return None


def json_finite(o: Any) -> Any:
    """``o`` with numpy scalars as Python numbers and non-finite floats as None (strict JSON)."""
    if isinstance(o, dict):
        return {k: json_finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_finite(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o) if math.isfinite(float(o)) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def write_json_finite(path: Path | str, obj: Any) -> None:
    """Write ``obj`` as indented JSON (UTF-8) after :func:`json_finite`. G10-35: the one meta writer of
    the variability and transient field products."""
    import json  # noqa: PLC0415

    Path(path).write_text(json.dumps(json_finite(obj), indent=2), encoding="utf-8")
