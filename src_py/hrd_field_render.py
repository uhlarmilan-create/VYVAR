"""Field images of the HRD view (HRD-TAB-2026-10): inverted, catalogue colour (vivid) and physical blackbody.

Visualization only, no photometry impact. Luminance comes from the MASTERSTAR FITS with an asinh
stretch; colour comes from the Gaia BP-RP of each matched star, splatted around its position and
applied with a hue-preserving stretch (Lupton et al. 2004, PASP 116, 133): a pixel whose brightest
channel would exceed 1 is scaled down as a whole, so bright stars keep their colour instead of
burning to white. Colour is applied only on detected light (3 sigma above the sky), so the sky
background stays neutral grey.

* ``vivid``: a display palette along BP-RP (blue - white - yellow - orange - red); the same palette
  colours the stars of the HR diagram (``bp_rp_rgb``).
* ``physical``: the sRGB colour of a blackbody at the star's Teff (BP-RP -> Teff, Pecaut & Mamajek
  2013; CIE 1931 colour matching functions, Wyman, Sloan & Shirley 2013; D65 white). Honest and pale.
* ``inverted``: black stars on white, no colour.

Catalogue colours are not measured by this camera and include interstellar reddening.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

FieldMode = Literal["inverted", "vivid", "physical"]
FIELD_MODES: tuple[FieldMode, ...] = ("inverted", "vivid", "physical")

TEFF_MIN_K = 2500.0
TEFF_MAX_K = 40000.0
# Pecaut & Mamajek (2013, ApJS 208, 9) Gaia-era BP-RP vs Teff anchors (2024 online table).
_BP_RP_ANCHORS = np.array([-0.40, -0.25, -0.05, 0.65, 1.00, 1.35, 1.80, 2.15, 2.85, 3.60, 4.50])
_TEFF_ANCHORS = np.array([40000.0, 30000.0, 9600.0, 5772.0, 5250.0, 4400.0, 3800.0, 3400.0, 3000.0, 2700.0, 2500.0])
_BP_RP_TO_TEFF = PchipInterpolator(_BP_RP_ANCHORS, _TEFF_ANCHORS, extrapolate=True)

# display palette along BP-RP (sRGB 0..1)
_PAL_BP_RP = np.array([-0.3, 0.2, 0.6, 0.85, 1.2, 1.6, 2.2, 3.2])
_PAL_RGB = np.array([[0.45, 0.62, 1.0], [0.65, 0.78, 1.0], [1.0, 1.0, 1.0], [1.0, 0.96, 0.80],
                     [1.0, 0.86, 0.55], [1.0, 0.68, 0.35], [1.0, 0.48, 0.22], [1.0, 0.32, 0.15]])

_WL_NM = np.arange(380.0, 781.0, 2.0)
_H, _C, _K = 6.62607015e-34, 299792458.0, 1.380649e-23

_ASINH_SOFT_SIGMA = 3.0
_TOP_PERCENTILE = 99.95
_CHROMA_BOOST = {"vivid": 1.6, "physical": 1.0}


def teff_from_bp_rp(bp_rp: float | np.ndarray) -> np.ndarray:
    """BP-RP -> Teff (K), monotonic, clamped to 2500..40000 K."""
    x = np.atleast_1d(np.asarray(bp_rp, dtype=float))
    return np.clip(_BP_RP_TO_TEFF(x), TEFF_MIN_K, TEFF_MAX_K)


def bp_rp_rgb(bp_rp: float | np.ndarray) -> np.ndarray:
    """Display colour (sRGB 0..1, shape (n, 3)) of the vivid palette."""
    b = np.clip(np.atleast_1d(np.asarray(bp_rp, dtype=float)), _PAL_BP_RP[0], _PAL_BP_RP[-1])
    return np.stack([np.interp(b, _PAL_BP_RP, _PAL_RGB[:, i]) for i in range(3)], axis=1)


def bp_rp_css(bp_rp: float) -> str:
    """CSS colour of one BP-RP (grey when unknown)."""
    if bp_rp is None or not math.isfinite(float(bp_rp)):
        return "rgb(140,145,155)"
    r, g, b = (bp_rp_rgb(float(bp_rp))[0] * 255).round().astype(int)
    return f"rgb({r},{g},{b})"


def _gauss(x: np.ndarray, mean: float, s1: float, s2: float) -> np.ndarray:
    a = (x - mean) / np.where(x < mean, s1, s2)
    return np.exp(-0.5 * a * a)


def blackbody_rgb(teff_k: np.ndarray) -> np.ndarray:
    """sRGB (gamma, brightest channel 1) of a blackbody, D65 white."""
    t = np.clip(np.atleast_1d(np.asarray(teff_k, dtype=float)), TEFF_MIN_K, TEFF_MAX_K)[:, None]
    wl = _WL_NM
    xbar = 1.056 * _gauss(wl, 599.8, 37.9, 31.0) + 0.362 * _gauss(wl, 442.0, 16.0, 26.7) - 0.065 * _gauss(wl, 501.1, 20.4, 26.2)
    ybar = 0.821 * _gauss(wl, 568.8, 46.9, 40.5) + 0.286 * _gauss(wl, 530.9, 16.3, 31.1)
    zbar = 1.217 * _gauss(wl, 437.0, 11.8, 36.0) + 0.681 * _gauss(wl, 459.0, 26.0, 13.8)
    lam = wl * 1e-9
    spd = (2 * _H * _C**2) / (lam**5 * np.expm1(_H * _C / (_K * lam * t)))
    xyz = np.stack([np.trapezoid(spd * c, wl, axis=1) for c in (xbar, ybar, zbar)], axis=1)
    m = np.array([[3.2404542, -1.5371385, -0.4985314], [-0.9692660, 1.8760108, 0.0415560], [0.0556434, -0.2040259, 1.0572252]])
    lin = np.clip(xyz @ m.T, 0.0, None)
    lin = lin / np.maximum(lin.max(axis=1, keepdims=True), 1e-30)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.power(lin, 1 / 2.4) - 0.055)


def field_luminance(data: np.ndarray) -> np.ndarray:
    """asinh-stretched luminance 0..1 of a frame (sky median 0, softening 3 sigma)."""
    d = np.asarray(data, dtype=float)
    ok = np.isfinite(d)
    sky = float(np.median(d[ok]))
    sigma = 1.4826 * float(np.median(np.abs(d[ok] - sky))) or 1.0
    x = np.where(ok, (d - sky) / sigma, 0.0)
    top = max(float(np.percentile(x[ok], _TOP_PERCENTILE)), 10.0)
    return np.clip(np.arcsinh(np.clip(x, 0.0, None) / _ASINH_SOFT_SIGMA) / np.arcsinh(top / _ASINH_SOFT_SIGMA), 0.0, 1.0)


def _chroma_map(shape: tuple[int, int], stars: pd.DataFrame, rgb: np.ndarray, fwhm_px: float) -> np.ndarray:
    """Per-pixel colour (mean 1 per pixel) around each star, flux-weighted where stars overlap."""
    h, w = shape
    chroma = rgb / rgb.mean(axis=1, keepdims=True)
    flux = np.clip(10 ** (-0.4 * (stars["phot_g_mean_mag"].to_numpy(float) - 12.0)), 1e-3, None)
    acc = np.zeros((h, w, 3))
    wsum = np.zeros((h, w))
    sig = max(float(fwhm_px), 1.0)
    rad = int(math.ceil(6 * sig))
    yy, xx = np.mgrid[-rad:rad + 1, -rad:rad + 1]
    for x, y, c, f in zip(stars["x"].to_numpy(float), stars["y"].to_numpy(float), chroma, flux):
        ix, iy = int(round(x)), int(round(y))
        y0, y1, x0, x1 = max(0, iy - rad), min(h, iy + rad + 1), max(0, ix - rad), min(w, ix + rad + 1)
        if y0 >= y1 or x0 >= x1:
            continue
        r = (3.0 + 1.5 * math.log10(f + 1.0)) * sig  # brighter stars have wider light
        k = np.exp(-((xx - (x - ix)) ** 2 + (yy - (y - iy)) ** 2) / (2 * r * r))
        k = k[y0 - (iy - rad):y1 - (iy - rad), x0 - (ix - rad):x1 - (ix - rad)]
        acc[y0:y1, x0:x1] += k[..., None] * c * f
        wsum[y0:y1, x0:x1] += k * f
    out = np.ones((h, w, 3))
    hit = wsum > 1e-12
    out[hit] = acc[hit] / wsum[hit, None]
    return out


def render_field(data: np.ndarray, stars: pd.DataFrame, mode: FieldMode, *, fwhm_px: float = 2.2) -> np.ndarray:
    """RGB uint8 image (rows = FITS y, columns = FITS x) of the field in one of ``FIELD_MODES``.

    ``stars``: ``x``, ``y`` (0-based pixels of the frame), ``bp_rp``, ``phot_g_mean_mag``.
    """
    lum = field_luminance(data)
    if mode == "inverted":
        g = (255 * (1.0 - lum)).astype(np.uint8)
        return np.repeat(g[..., None], 3, axis=2)
    s = stars[np.isfinite(stars["x"]) & np.isfinite(stars["y"]) & np.isfinite(stars["bp_rp"])
              & np.isfinite(stars["phot_g_mean_mag"])]
    rgb = bp_rp_rgb(s["bp_rp"]) if mode == "vivid" else blackbody_rgb(teff_from_bp_rp(s["bp_rp"]))
    cm = _chroma_map(lum.shape, s, rgb, fwhm_px)
    low = lum[lum < 0.15]
    sig = float(np.std(low)) if low.size > 16 else 0.02
    gate = np.clip((lum - float(np.median(lum)) - 3 * sig) / (4 * sig), 0.0, 1.0)[..., None]
    col = np.clip(1.0 + _CHROMA_BOOST[mode] * gate * (cm - 1.0), 0.0, None)
    img = lum[..., None] * col
    peak = img.max(axis=2, keepdims=True)
    np.divide(img, peak, out=img, where=peak > 1.0)  # hue-preserving clip (Lupton et al. 2004)
    return (255 * np.clip(img, 0.0, 1.0)).astype(np.uint8)


def field_image_jpeg(fits_path: Path, stars: pd.DataFrame, mode: FieldMode, out_jpg: Path, *, fwhm_px: float = 2.2) -> Path:
    """Render the field of one mode to a JPEG file."""
    from astropy.io import fits  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    out_jpg = Path(out_jpg)
    data = np.asarray(fits.getdata(fits_path), dtype=float)
    Image.fromarray(render_field(data, stars, mode, fwhm_px=fwhm_px)).save(out_jpg, quality=88)
    return out_jpg
