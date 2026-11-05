"""SKY-ORIENT-2026-11: every displayed sky image of VYVAR is shown North up, East left (Milan 2026-11-05).

North up, East left is the standard sky-chart convention (AAVSO VSP charts; SAOImage DS9 "Align to WCS";
Calabretta & Greisen 2002, A&A 395, 1077 for the CD / PC matrix). DISPLAY ONLY: FITS pixels, the WCS in the
files, pixel coordinates in the CSVs and the photometry are never changed. The image is never resampled: the
display transform is one of the 8 lossless pixel-grid symmetries (transpose and / or row / column flips, the
dihedral group of the square). It is chosen from the WCS at the frame centre so that North is within 45 deg of
up and East is on the LEFT; a mirrored (parity-flipped) optical train needs a flip for that, which is chosen
by the sign of the pixel -> sky Jacobian. The remaining rotation (|residual| <= 45 deg) is shown honestly with a
small N / E compass on every sky image (also when it is ~0). Without a celestial WCS the image is shown in the
FITS-natural view (row 0 at the bottom, as DS9 without WCS alignment) with the visible note
``orientation unknown (no WCS)`` - the orientation is never guessed.

Coordinates (explicit):

* pixel ``(x, y)``: 0-based, x = column index (FITS axis 1), y = row index (FITS axis 2) of the data array
  ``a[y, x]``; this is the astropy / numpy origin-0 convention used by every VYVAR CSV (``x``, ``y``) and by
  ``WCS.world_to_pixel``. FITS 1-based pixel = 0-based + 1.
* display ``(c, r)``: 0-based column / row of the display raster ``apply_image(a)[r, c]``; row 0 is the TOP of
  the shown image (PIL, ``st.image``, plotly with a reversed y axis, matplotlib ``origin="upper"``). Pixel
  centres map to pixel centres (integer -> integer), so ``inverse_xy(apply_xy(x, y)) == (x, y)`` exactly.

One code path: the GUI, the night run and the PDF report all call this module.
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)

NO_WCS_NOTE = "orientation unknown (no WCS)"
#: compass colour (dark orange: readable on the negative white sky and on a dark sky)
COMPASS_COLOR = "#c2410c"
#: bump when the display convention changes, so image render caches (HRD field JPEG) are redrawn
ORIENT_VERSION = 1


def _celestial(wcs_or_header: Any) -> Any | None:
    """The 2-D celestial astropy WCS of a WCS / FITS header, or None (no celestial WCS)."""
    if wcs_or_header is None:
        return None
    try:
        from astropy.wcs import WCS, FITSFixedWarning  # noqa: PLC0415

        if isinstance(wcs_or_header, WCS):
            w = wcs_or_header
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FITSFixedWarning)
                w = WCS(wcs_or_header)
        if not w.has_celestial:
            return None
        cel = w.celestial
        return cel if cel.naxis == 2 else None
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[SKY ORIENT] WCS not usable, image shown with '%s': %s", NO_WCS_NOTE, exc)
        return None


def _sky_axes_in_pixels(cel: Any, nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    """Unit vectors (dx, dy) in pixel space pointing North and East at the frame centre (full WCS incl. SIP)."""
    import astropy.units as u  # noqa: PLC0415
    from astropy.wcs.utils import proj_plane_pixel_scales  # noqa: PLC0415

    cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
    c0 = cel.pixel_to_world(cx, cy)
    step = 10.0 * float(np.mean(np.abs(proj_plane_pixel_scales(cel)))) * 3600.0  # 10 px, arcsec
    out = []
    for pa in (0.0, 90.0):  # position angle: North 0, East 90
        px, py = cel.world_to_pixel(c0.directional_offset_by(pa * u.deg, step * u.arcsec))
        v = np.array([float(px) - cx, float(py) - cy])
        n = float(np.hypot(*v))
        if not (math.isfinite(n) and n > 0):
            raise ValueError("WCS gives no direction at the frame centre")
        out.append(v / n)
    return out[0], out[1]


def _matrix(transpose: bool, flip_c: bool, flip_r: bool) -> np.ndarray:
    """2x2 integer matrix: display offset (dc, dr) = M @ pixel offset (dx, dy)."""
    p = np.array([[0, 1], [1, 0]]) if transpose else np.eye(2, dtype=int)
    return np.diag([-1 if flip_c else 1, -1 if flip_r else 1]) @ p


def _angle_from_up_ccw(v: np.ndarray) -> float:
    """Screen angle of a display vector (dc, dr; r down) from up, counter-clockwise (towards left) positive."""
    return float(np.degrees(np.arctan2(-v[0], -v[1])))


@dataclass(frozen=True)
class DisplayOrient:
    """A lossless display transform of a (ny, nx) sky image; see the module docstring for coordinates."""

    nx: int
    ny: int
    transpose: bool
    flip_c: bool
    flip_r: bool
    has_wcs: bool
    #: angle of North from display up, counter-clockwise (towards East = left) positive; |..| <= 45; nan w/o WCS
    residual_deg: float = float("nan")
    #: True when the raw chip is mirrored against the sky (FITS-natural view shows East right with North up)
    mirrored: bool = False
    #: unit display vectors (dc, dr) of North and East (None without WCS)
    north: tuple[float, float] | None = None
    east: tuple[float, float] | None = None

    # ---------------------------------------------------------------- geometry
    @property
    def out_shape(self) -> tuple[int, int]:
        """(rows, columns) of the display raster."""
        return (self.nx, self.ny) if self.transpose else (self.ny, self.nx)

    @property
    def matrix(self) -> np.ndarray:
        return _matrix(self.transpose, self.flip_c, self.flip_r)

    @property
    def note(self) -> str:
        """Text for a caption: the convention and the residual rotation, or the no-WCS note."""
        if not self.has_wcs:
            return NO_WCS_NOTE
        return f"North up, East left (residual rotation {self.residual_deg:+.1f}\u00b0)"

    def for_shape(self, shape: tuple[int, ...]) -> "DisplayOrient":
        """The same display transform for a crop (box) of the image with ``shape`` (ny, nx): a sub-box keeps
        the sky orientation of its frame."""
        return replace(self, ny=int(shape[0]), nx=int(shape[1]))

    def apply_image(self, arr: np.ndarray) -> np.ndarray:
        """Display raster of ``arr[y, x]`` (2-D, or 3-D with trailing channels); a view, pixels unchanged."""
        a = np.asarray(arr)
        if a.shape[:2] != (self.ny, self.nx):
            raise ValueError(f"image shape {a.shape[:2]} is not the oriented shape {(self.ny, self.nx)}")
        if self.transpose:
            a = np.swapaxes(a, 0, 1)
        if self.flip_r:
            a = a[::-1]
        if self.flip_c:
            a = a[:, ::-1]
        return a

    def apply_xy(self, x: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
        """0-based pixel (x, y) -> 0-based display (c, r); arrays or scalars (returned as float arrays)."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        u, v = (y, x) if self.transpose else (x, y)
        h, w = self.out_shape
        c = (w - 1) - u if self.flip_c else u
        r = (h - 1) - v if self.flip_r else v
        return np.asarray(c, dtype=np.float64), np.asarray(r, dtype=np.float64)

    def inverse_xy(self, c: Any, r: Any) -> tuple[np.ndarray, np.ndarray]:
        """0-based display (c, r) -> 0-based pixel (x, y): for clicks / hover on a displayed image."""
        c = np.asarray(c, dtype=np.float64)
        r = np.asarray(r, dtype=np.float64)
        h, w = self.out_shape
        u = (w - 1) - c if self.flip_c else c
        v = (h - 1) - r if self.flip_r else r
        x, y = (v, u) if self.transpose else (u, v)
        return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)

    def apply_xy_array(self, xy: np.ndarray) -> np.ndarray:
        """(n, 2) pixel array -> (n, 2) display array."""
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        c, r = self.apply_xy(xy[:, 0], xy[:, 1])
        return np.column_stack([c, r])

    # ------------------------------------------------ matplotlib (WCSAxes allows only origin="lower")
    def mpl_image(self, arr: np.ndarray) -> np.ndarray:
        """The display raster for ``imshow(..., origin="lower")``: the same picture on screen, rows stored
        bottom-up (matplotlib data y = ``out_shape[0] - 1 - r``)."""
        return self.apply_image(arr)[::-1]

    def apply_xy_mpl(self, x: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
        """0-based pixel (x, y) -> matplotlib data (c, y_up) of :meth:`mpl_image` shown with origin lower."""
        c, r = self.apply_xy(x, y)
        return c, (self.out_shape[0] - 1) - r

    def inverse_xy_mpl(self, c: Any, y_up: Any) -> tuple[np.ndarray, np.ndarray]:
        """matplotlib data (c, y_up) -> 0-based pixel (x, y)."""
        return self.inverse_xy(c, (self.out_shape[0] - 1) - np.asarray(y_up, dtype=np.float64))

    def mpl_wcs(self, cel: Any) -> Any:
        """Low-level APE-14 WCS of :meth:`mpl_image` (matplotlib WCSAxes RA / Dec grid); in memory only."""
        return _display_wcs_class()(cel, self)


def display_transform(wcs_or_header: Any, shape: tuple[int, ...]) -> DisplayOrient:
    """The North-up / East-left display transform of an image of ``shape`` (ny, nx[, ...]).

    ``wcs_or_header``: an astropy WCS, a FITS header or None. Without a celestial WCS: the FITS-natural
    view (row 0 at the bottom), ``has_wcs`` False (show :data:`NO_WCS_NOTE`).
    """
    ny, nx = int(shape[0]), int(shape[1])
    cel = _celestial(wcs_or_header)
    natural = DisplayOrient(nx=nx, ny=ny, transpose=False, flip_c=False, flip_r=True, has_wcs=False)
    if cel is None:
        return natural
    try:
        n_pix, e_pix = _sky_axes_in_pixels(cel, nx, ny)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[SKY ORIENT] no sky direction from the WCS, image shown with '%s': %s", NO_WCS_NOTE, exc)
        return natural
    sky_det = float(e_pix[0] * n_pix[1] - e_pix[1] * n_pix[0])
    best: tuple[tuple[float, int], DisplayOrient] | None = None
    for transpose in (False, True):
        for flip_r in (True, False):
            for flip_c in (False, True):
                m = _matrix(transpose, flip_c, flip_r)
                # display East-left / North-up has det([e, n]) > 0 in (c, r-down); M keeps or flips it
                if float(np.linalg.det(m)) * sky_det <= 0:
                    continue
                nd, ed = m @ n_pix, m @ e_pix
                res = _angle_from_up_ccw(nd)
                # smallest residual; an exact 45 deg tie goes to the fewest operations (deterministic)
                key = (round(abs(res), 9), int(transpose) * 4 + int(flip_c) * 2 + int(not flip_r))
                if best is None or key < best[0]:
                    best = (key, DisplayOrient(
                        nx=nx, ny=ny, transpose=transpose, flip_c=flip_c, flip_r=flip_r, has_wcs=True,
                        residual_deg=res, mirrored=sky_det > 0,
                        north=(float(nd[0]), float(nd[1])), east=(float(ed[0]), float(ed[1])),
                    ))
    assert best is not None  # the 4 symmetries of each parity cover all directions
    return best[1]


def header_from_string(text: str | None) -> Any | None:
    """A FITS header from its string form (e.g. a WCS stored in a JSON plot file), or None."""
    if not text:
        return None
    from astropy.io import fits  # noqa: PLC0415

    return fits.Header.fromstring(str(text))


def orient_from_fits(fits_path: Path | str, shape: tuple[int, ...] | None = None) -> DisplayOrient:
    """Display transform from a FITS file's primary header (header read only; the file is not changed)."""
    from astropy.io import fits  # noqa: PLC0415

    hdr = fits.getheader(Path(fits_path), 0)
    if shape is None:
        shape = (int(hdr["NAXIS2"]), int(hdr["NAXIS1"]))
    return display_transform(hdr, shape)


# ---------------------------------------------------------------------------------------------- WCS wrapper
def _display_wcs_class() -> type:
    from astropy.wcs.wcsapi.wrappers.base import BaseWCSWrapper  # noqa: PLC0415

    class DisplayWCS(BaseWCSWrapper):
        """APE-14 low-level WCS of the matplotlib display: (c, y_up) -> pixel (x, y) -> world."""

        def __init__(self, wcs: Any, orient: DisplayOrient) -> None:
            super().__init__(wcs)
            self._orient = orient

        def pixel_to_world_values(self, *pixel_arrays: Any) -> Any:
            x, y = self._orient.inverse_xy_mpl(pixel_arrays[0], pixel_arrays[1])
            return self._wcs.pixel_to_world_values(x, y)

        def world_to_pixel_values(self, *world_arrays: Any) -> Any:
            x, y = self._wcs.world_to_pixel_values(*world_arrays)
            return self._orient.apply_xy_mpl(x, y)

        @property
        def pixel_shape(self) -> tuple[int, int]:
            h, w = self._orient.out_shape
            return (w, h)

        @property
        def pixel_bounds(self) -> None:
            return None

        @property
        def axis_correlation_matrix(self) -> np.ndarray:
            return np.ones((2, 2), dtype=bool)

    return DisplayWCS


# ---------------------------------------------------------------------------------------------- compasses
def _rgb(color: str) -> tuple[int, int, int]:
    s = color.lstrip("#")
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))


def draw_compass_pil(img: Any, orient: DisplayOrient, *, size_px: int | None = None, color: str = COMPASS_COLOR) -> None:
    """Draw the N / E compass (upper-right corner) or the no-WCS note (lower-left corner) onto a PIL image of
    the display raster (any scale: arrows are directions only)."""
    from PIL import ImageDraw, ImageFont  # noqa: PLC0415

    draw = ImageDraw.Draw(img)
    w, h = img.size
    L = float(size_px or max(18, int(round(0.06 * min(w, h)))))
    fsize = max(11, int(round(L * 0.45)))
    try:
        font = ImageFont.load_default(size=fsize)
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    col = _rgb(color)
    if not orient.has_wcs:
        x0, y1 = 4, h - 4
        box = draw.textbbox((0, 0), NO_WCS_NOTE, font=font)
        tw, th = box[2] - box[0], box[3] - box[1]
        draw.rectangle([x0, y1 - th - 10, x0 + tw + 8, y1], fill=(255, 255, 255))
        draw.text((x0 + 4, y1 - th - 6 - box[1]), NO_WCS_NOTE, fill=col, font=font)
        return
    pad = 0.6 * L + fsize
    ox, oy = w - pad - L * 0.2, pad + L * 0.2
    lw = max(2, int(round(L / 14)))
    for lab, v in (("N", orient.north), ("E", orient.east)):
        dx, dy = float(v[0]), float(v[1])
        tx, ty = ox + L * dx, oy + L * dy
        draw.line([(ox, oy), (tx, ty)], fill=col, width=lw)
        a = 0.28 * L  # arrow head
        for s in (+1, -1):
            ang = math.atan2(dy, dx) + math.pi + s * 0.45
            draw.line([(tx, ty), (tx + a * math.cos(ang), ty + a * math.sin(ang))], fill=col, width=lw)
        lx, ly = ox + 1.32 * L * dx, oy + 1.32 * L * dy
        draw.text((lx, ly), lab, fill=col, font=font, anchor="mm")


def draw_compass_mpl(ax: Any, orient: DisplayOrient, *, anchor: tuple[float, float] = (0.93, 0.88),
                     length_pt: float = 26.0, color: str = COMPASS_COLOR, fontsize: float = 11.0) -> list[Any]:
    """N / E compass on matplotlib axes that show the display picture (:meth:`DisplayOrient.mpl_image` with
    ``origin="lower"``, y axis not inverted); ``anchor`` is the compass origin in axes fraction. Without WCS the note is written instead. Returns the
    artists."""
    from matplotlib.text import OffsetFrom  # noqa: PLC0415

    if not orient.has_wcs:
        return [ax.text(0.01, 0.99, NO_WCS_NOTE, transform=ax.transAxes, ha="left", va="top", color=color,
                        fontsize=fontsize, bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))]
    ref = OffsetFrom(ax, anchor, unit="points")
    out = []
    for lab, v in (("N", orient.north), ("E", orient.east)):
        dx, dy = float(v[0]), -float(v[1])  # screen points: y up
        out.append(ax.annotate("", xy=(length_pt * dx, length_pt * dy), xycoords=ref, xytext=(0, 0), textcoords=ref,
                               arrowprops=dict(arrowstyle="-|>", color=color, lw=2), annotation_clip=False))
        out.append(ax.annotate(lab, xy=(1.4 * length_pt * dx, 1.4 * length_pt * dy), xycoords=ref, ha="center",
                               va="center", color=color, fontsize=fontsize, fontweight="bold", annotation_clip=False))
    return out


def compass_plotly_annotations(orient: DisplayOrient, *, xref: str = "x", yref: str = "y",
                               color: str = COMPASS_COLOR) -> list[dict[str, Any]]:
    """Plotly annotations of the compass in DATA coordinates of the display raster (equal-aspect axes with the
    y axis reversed, upper-right corner); the no-WCS note in paper coordinates."""
    if not orient.has_wcs:
        return [dict(text=NO_WCS_NOTE, xref="paper", yref="paper", x=0.01, y=0.99, xanchor="left", yanchor="top",
                     showarrow=False, font=dict(color=color, size=13), bgcolor="rgba(255,255,255,0.85)")]
    h, w = orient.out_shape
    L = 0.07 * min(w, h)
    ox, oy = w - 1.0 - 2.2 * L, 2.2 * L
    ax_ref = xref
    ay_ref = yref
    out: list[dict[str, Any]] = []
    for lab, v in (("N", orient.north), ("E", orient.east)):
        dx, dy = float(v[0]), float(v[1])
        out.append(dict(x=ox + L * dx, y=oy + L * dy, ax=ox, ay=oy, xref=xref, yref=yref, axref=ax_ref, ayref=ay_ref,
                        text="", showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=2, arrowcolor=color))
        out.append(dict(x=ox + 1.45 * L * dx, y=oy + 1.45 * L * dy, xref=xref, yref=yref, text=f"<b>{lab}</b>",
                        showarrow=False, font=dict(color=color, size=14)))
    return out
