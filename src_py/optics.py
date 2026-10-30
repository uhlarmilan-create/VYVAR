"""Optics facts of a frame - the ONE module (code review 2 T08, HDR-ONE-2026-10).

Focal length, pixel size, sensor size and plate scale are read and combined here only.

Conventions
-----------
* ``EQUIPMENTS.PIXELSIZE`` is the native (1x1) pixel pitch [um]; ``TELESCOPE.FOCAL`` is [mm].
* Header ``XPIXSZ`` / ``YPIXSZ`` follow the MaxIm DL / SBIG convention: the BINNED pixel pitch
  [um] (QHY / N.I.N.A. write the comment "X binned pixel size in microns"). ``PIXSIZE1`` /
  ``PIXSIZE2`` / ``PIXSIZE`` / ``PIXSZLX`` / ``PIXSZLY`` are the native pitch. A card comment that
  says otherwise ("binned" / "unbinned") wins.
* Effective (on-sky) pixel = native pitch x XBINNING.
* Precedence (param_resolver rule for equipment-intrinsic values): DB (valid) -> header, with a
  logged cross-check when both exist and disagree.
* Plate scale [arcsec/px] = effective pixel [um] / focal [mm] x 206.265. No value is rewritten
  ("2000 mm looks like a typo for 200 mm" was removed, G13-06): a wrong DB entry is reported when
  the solved WCS disagrees (:func:`report_scale_mismatch`), never guessed.
* A solved celestial WCS is the authority for the scale of a solved frame
  (:func:`plate_scale_arcsec_per_px_from_wcs`, :func:`plate_scale_from_header`).
"""
from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from infolog import log_event
from utils import fits_binning_xy_from_header

#: arcsec per (um / mm): 206264.8 arcsec per radian / 1000.
PLATE_SCALE_ARCSEC_PER_UM_OVER_MM: float = 206.265

#: Header keywords of the focal length, in priority order.
FOCAL_KEYS: tuple[str, ...] = ("FOCALLEN", "FOCLEN", "TELFOCA", "FOCAL_LEN", "FOCALL", "FOC_LEN")
#: Native pixel pitch keywords (X, Y) and binned pixel pitch keywords (MaxIm convention).
NATIVE_PIXEL_KEYS: tuple[tuple[str, ...], tuple[str, ...]] = (("PIXSIZE1", "PIXSIZE", "PIXSZLX"), ("PIXSIZE2", "PIXSZLY"))
BINNED_PIXEL_KEYS: tuple[tuple[str, ...], tuple[str, ...]] = (("XPIXSZ",), ("YPIXSZ",))

#: Physical limits only (no rig values): all-sky lenses start near 1.4 mm, the largest telescopes
#: reach ~120 m effective focal length; pixels are below 300 um.
FOCAL_MM_MIN: float = 1.0
FOCAL_MM_MAX: float = 120_000.0
PIXEL_UM_MAX: float = 300.0

# Card comments that state a unit or a binning state.
_COMMENT_MM_RE = re.compile(r"\b(mm|millimet(er|re)s?)\b", re.IGNORECASE)
_COMMENT_M_RE = re.compile(r"(\[m\]|\(m\)|\bmet(er|re)s?\b|\bin m\b)", re.IGNORECASE)
_COMMENT_UNBINNED_RE = re.compile(r"\b(unbinned|native|1x1)\b", re.IGNORECASE)
_COMMENT_BINNED_RE = re.compile(r"\bbinned\b", re.IGNORECASE)
# Sensor size "W*H", "WxH", "W X H", "W<multiplication sign>H" (G13-07: the DB writes "*").
_SENSOR_SIZE_RE = re.compile("(\\d+)\\s*[*xX\u00d7]\\s*(\\d+)")


def _finite_positive(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) and x > 0 else None


def _comment(header: Any, key: str) -> str:
    try:
        return str(header.comments[key] or "")
    except (AttributeError, KeyError, IndexError):
        return ""


# --------------------------------------------------------------------------------------------
# focal length
# --------------------------------------------------------------------------------------------
def focal_mm_plausible(mm: Any) -> bool:
    v = _finite_positive(mm)
    return v is not None and FOCAL_MM_MIN <= v <= FOCAL_MM_MAX


def focal_length_mm_from_value(value: Any, comment: str = "") -> float | None:
    """Focal length [mm] of one header value; None when not plausible.

    The card comment decides the unit when it names one ("mm", "[m]", "meters"). Without a unit a
    value below 0.5 is metres (no lens is shorter than 0.5 mm; G02-41: 14-24 mm lenses stay mm).
    """
    v = _finite_positive(value)
    if v is None:
        return None
    if _COMMENT_MM_RE.search(comment or ""):
        mm = v
    elif _COMMENT_M_RE.search(comment or ""):
        mm = v * 1000.0
    else:
        mm = v * 1000.0 if v < 0.5 else v
    return float(mm) if focal_mm_plausible(mm) else None


def header_focal_length_mm(header: Any) -> float | None:
    """Focal length [mm] from the first plausible :data:`FOCAL_KEYS` card."""
    if header is None:
        return None
    for key in FOCAL_KEYS:
        if key not in header:
            continue
        mm = focal_length_mm_from_value(header[key], _comment(header, key))
        if mm is not None:
            return mm
    return None


# --------------------------------------------------------------------------------------------
# pixel size and sensor size
# --------------------------------------------------------------------------------------------
def pixel_size_um_from_value(value: Any) -> float | None:
    """Pixel pitch [um] of one header value written in m, mm or um (None when not usable).

    Below 5e-5 the value is metres (3.76e-6), below 0.2 millimetres (0.00376); a pitch in um is
    never below 0.2. Pitches above :data:`PIXEL_UM_MAX` are rejected.
    """
    v = _finite_positive(value)
    if v is None:
        return None
    if v < 5e-5:
        um = v * 1e6
    elif v < 0.2:
        um = v * 1000.0
    else:
        um = v
    return float(um) if um <= PIXEL_UM_MAX else None


def header_native_pixel_um(header: Any) -> tuple[float | None, float | None]:
    """Native (1x1) pixel pitch [um] (x, y) from a header (see the module conventions)."""
    if header is None:
        return None, None
    xb, yb = fits_binning_xy_from_header(header)
    out: list[float | None] = []
    for axis, binning in ((0, xb), (1, yb)):
        val: float | None = None
        for key in NATIVE_PIXEL_KEYS[axis]:
            if key in header:
                p = pixel_size_um_from_value(header[key])
                if p is not None:
                    # "binned" as a word: "unbinned" does not match
                    val = p / binning if _COMMENT_BINNED_RE.search(_comment(header, key)) else p
                    break
        if val is None:
            for key in BINNED_PIXEL_KEYS[axis]:
                if key in header:
                    p = pixel_size_um_from_value(header[key])
                    if p is not None:
                        val = p if _COMMENT_UNBINNED_RE.search(_comment(header, key)) else p / binning
                        break
        out.append(val)
    x, y = out
    return (x if x is not None else y), (y if y is not None else x)


def header_native_pixel_um_mean(header: Any) -> float | None:
    x, y = header_native_pixel_um(header)
    if x is None or y is None:
        return None
    return (float(x) + float(y)) / 2.0


def parse_sensor_size(text: Any) -> tuple[int, int] | None:
    """``EQUIPMENTS.SENSORSIZE`` / header text ``"6280*4176"``, ``"6280x4176"``, ``"6280 X 4176"``
    -> (width, height); None otherwise (G13-07: one parser)."""
    if text is None or not str(text).strip():
        return None
    m = _SENSOR_SIZE_RE.search(str(text))
    if not m:
        return None
    w, h = int(m.group(1)), int(m.group(2))
    return (w, h) if w > 0 and h > 0 else None


# --------------------------------------------------------------------------------------------
# plate scale
# --------------------------------------------------------------------------------------------
def plate_scale_arcsec_per_pixel(*, pixel_pitch_um: float, focal_length_mm: float) -> float | None:
    """Small-angle plate scale [arcsec/px] = pitch [um] / focal [mm] x 206.265 (no rewriting)."""
    p = _finite_positive(pixel_pitch_um)
    f = _finite_positive(focal_length_mm)
    if p is None or f is None:
        return None
    return p / f * PLATE_SCALE_ARCSEC_PER_UM_OVER_MM


def plate_scale_arcsec_per_px_from_wcs(wcs_obj: Any) -> float | None:
    """Mean of ``proj_plane_pixel_scales`` [arcsec/px] of a celestial WCS (any rotation, G04-30)."""
    from astropy.wcs.utils import proj_plane_pixel_scales  # noqa: PLC0415

    if wcs_obj is None:
        return None
    try:
        w = wcs_obj.celestial if getattr(wcs_obj, "has_celestial", False) else wcs_obj
        arcsec = float(np.mean(proj_plane_pixel_scales(w))) * 3600.0
    except (ValueError, TypeError, AttributeError):
        return None
    return arcsec if math.isfinite(arcsec) and arcsec > 0 else None


_HEADER_SCALE_KEYS: tuple[str, ...] = ("VY_PLTS", "PIXSCALE", "SECPIX", "SECPIX1", "PIXSCAL1", "PLTSCALE", "SCALE")


def plate_scale_from_header(header: Any) -> float | None:
    """Plate scale [arcsec/px] of a frame from its header - the one FITS reader (G08-29 / G09-24).

    1. the solved celestial WCS (CD, PC + CDELT, SIP); ``VY_PLTS`` is only cross-checked (> 5 %
       disagreement is logged);
    2. without a celestial WCS: ``VY_PLTS``, then the capture-software scale keywords.
    None when nothing is usable. No sanity window: a solved 0.05 "/px or 60 "/px is real.
    """
    if header is None:
        return None
    wcs_scale: float | None = None
    if any(str(header.get(f"CTYPE{i}", "")).upper().startswith(("RA", "DEC")) for i in (1, 2)):
        from astropy.wcs import WCS, FITSFixedWarning  # noqa: PLC0415

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            try:
                w = WCS(header)
            except (ValueError, KeyError, MemoryError) as exc:
                log_event(f"WARNING: plate scale: header WCS not readable ({exc})")
                w = None
        if w is not None and w.has_celestial:
            wcs_scale = plate_scale_arcsec_per_px_from_wcs(w)
    if wcs_scale is not None:
        vy = _finite_positive(header.get("VY_PLTS"))
        if vy is not None and abs(vy - wcs_scale) / wcs_scale > 0.05:
            log_event(
                f"WARNING: VY_PLTS={vy:.4f} disagrees with the solved WCS scale {wcs_scale:.4f} arcsec/px "
                "(> 5 %) - using the WCS."
            )
        return wcs_scale
    for key in _HEADER_SCALE_KEYS:
        v = _finite_positive(header.get(key))
        if v is not None:
            return v
    return None


def plate_scale_from_fits(fits_path: Path | str | None = None, *, header: Any = None) -> float | None:
    """:func:`plate_scale_from_header` of ``header`` or of the primary header of ``fits_path``."""
    if header is None:
        if fits_path is None or not Path(fits_path).is_file():
            return None
        from astropy.io import fits  # noqa: PLC0415

        header = fits.getheader(Path(fits_path), 0)
    return plate_scale_from_header(header)


def frame_plate_scale(
    fits_path: Path | str | None = None,
    *,
    header: Any = None,
    db: Any = None,
    draft_id: int | None = None,
) -> float | None:
    """Plate scale [arcsec/px] of a (MASTERSTAR) frame: its solved WCS / scale keywords
    (:func:`plate_scale_from_fits`), else the draft rig from the DB with the frame binning
    (:func:`optics_hint`). None when neither resolves (derive-or-None, no default)."""
    if header is None and fits_path is not None and Path(fits_path).is_file():
        from astropy.io import fits  # noqa: PLC0415

        header = fits.getheader(Path(fits_path), 0)
    scale = plate_scale_from_header(header)
    if scale is not None:
        return scale
    if db is None or draft_id is None:
        return None
    row = db.fetch_obs_draft_by_id(int(draft_id)) or {}
    eq = int(row["ID_EQUIPMENTS"]) if row.get("ID_EQUIPMENTS") else None
    tel = int(row["ID_TELESCOPE"]) if row.get("ID_TELESCOPE") else None
    scale = optics_hint(header, db, equipment_id=eq, telescope_id=tel).arcsec_per_px
    if scale is not None:
        log_event(f"PLATE SCALE: no solved WCS - draft rig (DB) gives {scale:.4f} arcsec/px")
    return scale


@dataclass(frozen=True)
class OpticsHint:
    """Expected optics of a frame before it is solved (G03-19: one place)."""

    focal_mm: float | None
    focal_source: str
    pixel_native_um: float | None
    pixel_source: str
    binning_x: int
    binning_y: int

    @property
    def pixel_effective_um(self) -> float | None:
        return None if self.pixel_native_um is None else float(self.pixel_native_um) * float(self.binning_x)

    @property
    def arcsec_per_px(self) -> float | None:
        if self.pixel_effective_um is None or self.focal_mm is None:
            return None
        return plate_scale_arcsec_per_pixel(pixel_pitch_um=self.pixel_effective_um, focal_length_mm=self.focal_mm)


def _cross_check(label: str, db_value: float, hdr_value: float | None, unit: str) -> None:
    if hdr_value is not None and abs(hdr_value - db_value) / db_value > 0.05:
        log_event(
            f"WARNING: {label}: database {db_value:g} {unit} vs FITS header {hdr_value:g} {unit} (> 5 %) - "
            "using the database value; check the equipment / telescope entry."
        )


def optics_hint(
    header: Any,
    db: Any = None,
    *,
    equipment_id: int | None = None,
    telescope_id: int | None = None,
    binning: tuple[int, int] | None = None,
) -> OpticsHint:
    """Focal length, native pixel and binning of a frame: DB (valid) -> header (one rule).

    ``TELESCOPE.FOCAL`` of ``telescope_id`` and ``EQUIPMENTS.PIXELSIZE`` of ``equipment_id`` win;
    the header (:func:`header_focal_length_mm`, :func:`header_native_pixel_um_mean`) is used when
    the DB has no valid value and is cross-checked otherwise. Binning: ``binning`` when given, else
    the header (``XBINNING`` / ``YBINNING``). No "first active telescope" guess without an id.
    """
    hdr_focal = header_focal_length_mm(header)
    hdr_pixel = header_native_pixel_um_mean(header)
    focal, focal_src = None, "none"
    if db is not None and telescope_id is not None:
        tf = db.get_telescope_focal_mm(int(telescope_id))
        if focal_mm_plausible(tf):
            focal, focal_src = float(tf), "database_telescope"
            _cross_check(f"focal length (telescope {int(telescope_id)})", focal, hdr_focal, "mm")
    if focal is None and hdr_focal is not None:
        focal, focal_src = hdr_focal, "fits_header"
    pixel, pixel_src = None, "none"
    if db is not None and equipment_id is not None:
        pv = _finite_positive(db.get_equipment_pixel_size_um(int(equipment_id)))
        if pv is not None and pv <= PIXEL_UM_MAX:
            pixel, pixel_src = pv, "database_equipment"
            _cross_check(f"native pixel (equipment {int(equipment_id)})", pixel, hdr_pixel, "um")
    if pixel is None and hdr_pixel is not None:
        pixel, pixel_src = hdr_pixel, "fits_header"
    if binning is not None:
        bx, by = max(1, int(binning[0])), max(1, int(binning[1]))
    elif header is not None:
        bx, by = fits_binning_xy_from_header(header)
    else:
        bx, by = 1, 1
    return OpticsHint(focal, focal_src, pixel, pixel_src, int(bx), int(by))


def report_scale_mismatch(expected_arcsec: float | None, solved_arcsec: float | None, *, context: str) -> float | None:
    """Compare the expected scale (DB / header optics) with a solved WCS scale (G13-06).

    Returns solved / expected; logs a WARNING naming the likely cause when they differ by more
    than 10 % (a ~10x ratio is typically a focal length typed in the wrong unit or with an extra
    digit, a ratio equal to the binning a binned / native pixel mix-up). The solved scale is never
    replaced by the expected one.
    """
    e = _finite_positive(expected_arcsec)
    s = _finite_positive(solved_arcsec)
    if e is None or s is None:
        return None
    ratio = s / e
    if abs(ratio - 1.0) > 0.10:
        log_event(
            f"WARNING: {context}: solved plate scale {s:.4f} arcsec/px vs expected {e:.4f} from the "
            f"telescope focal length / camera pixel / binning (ratio {ratio:.3f}). Check TELESCOPE.FOCAL, "
            "EQUIPMENTS.PIXELSIZE and the binning of this draft."
        )
    return ratio
