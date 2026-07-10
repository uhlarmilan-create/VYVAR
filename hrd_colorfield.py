"""Catalog-derived color tinting for monochrome MASTERSTAR fields (Gaia BP-RP / Teff).

Visualization only -- no photometry impact. Luminance from the science FITS; chrominance from
Gaia catalog colors splatted at matched star positions.

Physics:
  - Teff: ``teff_gspphot`` from HRD enrichment cache when present (extreme candidates); else
    BP-RP -> Teff via monotonic Pecaut & Mamajek (2013) Gaia BP-RP anchor points.
  - Teff -> sRGB chromaticity: Planckian locus with Wyman, Sloan & Shirley (2013, JCGT 2, 2)
    analytic CIE 1931 CMFs -> linear sRGB (D65) -> gamma; unit max channel; desaturate toward white.
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from gaia_catalog_id import normalize_gaia_source_id, normalize_gaia_source_id_series, read_vyvar_csv
from infolog import log_event

logger = logging.getLogger(__name__)

# Mandatory honesty caption (English, on-image).
HRD_COLORFIELD_CAPTION = (
    "Catalog-derived colors (Gaia BP-RP / Teff); not measured by this camera. "
    "BP-RP includes interstellar reddening -- extinguished stars appear redder."
)

TEFF_MIN_K = 2500.0
TEFF_MAX_K = 40000.0
BP_RP_DOMAIN = (-0.4, 4.5)

# Pecaut & Mamajek (2013, ApJS 208, 9) Gaia-era BP-RP vs Teff anchors (2024 online table).
_BP_RP_ANCHORS = np.array([-0.40, -0.25, -0.05, 0.65, 1.00, 1.35, 1.80, 2.15, 2.85, 3.60, 4.50])
_TEFF_ANCHORS = np.array([40000.0, 30000.0, 9600.0, 5772.0, 5250.0, 4400.0, 3800.0, 3400.0, 3000.0, 2700.0, 2500.0])
_BP_RP_TO_TEFF = PchipInterpolator(_BP_RP_ANCHORS, _TEFF_ANCHORS, extrapolate=True)

# Planck + CMF integration grid (nm).
_WL_NM = np.arange(380.0, 781.0, 2.0)
_H_PLANCK = 6.62607015e-34
_C_LIGHT = 299792458.0
_K_BOLTZ = 1.380649e-23

_DEFAULT_KERNEL_SIGMA_PX = 2.5
_STRETCH_LO_PCT = 5.0
_STRETCH_HI_PCT = 99.5


def _piecewise_gaussian(x: np.ndarray, mean: float, stddev_1: float, stddev_2: float) -> np.ndarray:
    """Wyman et al. (2013) piecewise Gaussian lobe."""
    stddev = np.where(x < mean, stddev_1, stddev_2)
    a = (x - mean) / stddev
    return np.exp(-0.5 * a * a)


def _cie_cmfs_wyman2013(wavelength_nm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CIE 1931 x-bar, y-bar, z-bar via Wyman, Sloan & Shirley (2013) multi-lobe Gaussian fit."""
    wl = np.asarray(wavelength_nm, dtype=np.float64)
    xbar = (
        1.056 * _piecewise_gaussian(wl, 599.8, 37.9, 31.0)
        + 0.362 * _piecewise_gaussian(wl, 442.0, 16.0, 26.7)
        - 0.065 * _piecewise_gaussian(wl, 501.1, 20.4, 26.2)
    )
    ybar = 0.821 * _piecewise_gaussian(wl, 568.8, 46.9, 40.5) + 0.286 * _piecewise_gaussian(
        wl, 530.9, 16.3, 31.1
    )
    zbar = 1.217 * _piecewise_gaussian(wl, 437.0, 11.8, 36.0) + 0.681 * _piecewise_gaussian(
        wl, 459.0, 26.0, 13.8
    )
    return xbar, ybar, zbar


def _planck_spd(wavelength_nm: np.ndarray, teff_k: np.ndarray) -> np.ndarray:
    """Relative Planck spectral radiance; teff_k shape (N,) -> (N, n_wl)."""
    wl_m = np.asarray(wavelength_nm, dtype=np.float64) * 1e-9
    t = np.asarray(teff_k, dtype=np.float64)[..., np.newaxis]
    c1 = 2.0 * _H_PLANCK * _C_LIGHT**2
    c2 = _H_PLANCK * _C_LIGHT / _K_BOLTZ
    denom = wl_m**5 * np.expm1(c2 / (wl_m * t))
    return c1 / denom


def _xyz_from_spd_batch(spd: np.ndarray) -> np.ndarray:
    """Integrate SPD against Wyman CMFs -> XYZ, shape (N, 3)."""
    xbar, ybar, zbar = _cie_cmfs_wyman2013(_WL_NM)
    cmf = np.stack([xbar, ybar, zbar], axis=0)
    integrand = spd[..., np.newaxis, :] * cmf[np.newaxis, :, :]
    return np.trapezoid(integrand, _WL_NM, axis=-1)


def _srgb_linear_from_xyz(xyz: np.ndarray) -> np.ndarray:
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    r = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z
    g = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
    b = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z
    return np.stack([r, g, b], axis=-1)


def _srgb_gamma_encode(rgb_lin: np.ndarray) -> np.ndarray:
    out = np.asarray(rgb_lin, dtype=np.float64)
    lo = out <= 0.0031308
    out = out.copy()
    out[lo] = 12.92 * out[lo]
    out[~lo] = 1.055 * np.power(out[~lo], 1.0 / 2.4) - 0.055
    return out


def teff_from_bp_rp(bp_rp: float | np.ndarray) -> float | np.ndarray:
    """Monotonic BP-RP -> Teff (K); domain [-0.4, 4.5], clamp [2500, 40000].

    Anchor points from Pecaut & Mamajek (2013, ApJS 208, 9) Gaia BP-RP scale; PCHIP interpolation.
    """
    scalar = np.isscalar(bp_rp)
    x = np.atleast_1d(np.asarray(bp_rp, dtype=np.float64))
    teff = _BP_RP_TO_TEFF(x)
    teff = np.clip(teff, TEFF_MIN_K, TEFF_MAX_K)
    if scalar:
        return float(teff[0])
    return teff


def teff_to_srgb_chroma(
    teff_k: float | np.ndarray,
    *,
    saturation: float = 0.7,
) -> np.ndarray:
    """Planckian locus -> unit-max sRGB chromaticity with desaturation toward white.

    Wyman, Sloan & Shirley (2013, JCGT 2, 2) CMFs; sRGB D65 gamma encoding.
    """
    sat = float(np.clip(saturation, 0.0, 1.0))
    scalar = np.isscalar(teff_k)
    t = np.atleast_1d(np.asarray(teff_k, dtype=np.float64))
    t = np.clip(t, TEFF_MIN_K, TEFF_MAX_K)
    spd = _planck_spd(_WL_NM, t)
    xyz = _xyz_from_spd_batch(spd)
    rgb_lin = _srgb_linear_from_xyz(xyz)
    mx = np.max(rgb_lin, axis=-1, keepdims=True)
    mx = np.where(mx > 0, mx, 1.0)
    rgb_lin = rgb_lin / mx
    rgb = _srgb_gamma_encode(rgb_lin)
    rgb = sat * rgb + (1.0 - sat) * 1.0
    if scalar:
        return rgb[0]
    return rgb


def hrd_color_saturation_from_cfg(cfg: Any | None) -> float:
    """Resolve ``hrd_color_saturation`` (default 0.7, clamp 0..1)."""
    default = 0.7
    if cfg is None:
        return default
    try:
        val = float(getattr(cfg, "hrd_color_saturation", default))
    except (TypeError, ValueError):
        return default
    return float(np.clip(val, 0.0, 1.0))


def hrd_color_field_enabled(cfg: Any | None) -> bool:
    if cfg is None:
        return True
    return bool(getattr(cfg, "hrd_color_field_enabled", True))


def _resolve_masterstars_csv(platesolve_dir: Path, photometry_dir: Path) -> Path | None:
    for base in (Path(platesolve_dir), Path(photometry_dir)):
        p = base / "masterstars_full_match.csv"
        if p.is_file():
            return p
    return None


def _resolve_flux_column(df: pd.DataFrame) -> str | None:
    for col in ("dao_flux", "flux", "peak_dao"):
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            if (vals > 0).any():
                return col
    return None


def _load_enrich_teff_cache(cache_path: Path) -> dict[str, float]:
    if not cache_path.is_file():
        return {}
    try:
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    entries = raw.get("entries", raw)
    out: dict[str, float] = {}
    if not isinstance(entries, dict):
        return out
    for sid, hit in entries.items():
        if not isinstance(hit, dict):
            continue
        t = hit.get("teff_gspphot")
        if t is None:
            continue
        try:
            tf = float(t)
        except (TypeError, ValueError):
            continue
        if math.isfinite(tf):
            out[normalize_gaia_source_id(sid)] = tf
    return out


def _load_luminance_from_fits(
    fits_path: Path,
    *,
    lo_pct: float = _STRETCH_LO_PCT,
    hi_pct: float = _STRETCH_HI_PCT,
) -> np.ndarray | None:
    """Percentile-stretched luminance L in [0, 1] (same logic as ``fits_first_image_to_png``)."""
    try:
        from astropy.io import fits
    except ImportError:
        return None
    try:
        with fits.open(fits_path, memmap=False) as hdul:
            data = np.asarray(hdul[0].data, dtype=np.float64)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to read MASTERSTAR FITS %s", fits_path)
        return None
    if data.size == 0:
        return None
    ok = np.isfinite(data)
    if not ok.any():
        return None
    lo = float(np.nanpercentile(data[ok], float(lo_pct)))
    hi = float(np.nanpercentile(data[ok], float(hi_pct)))
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        lo = float(np.nanmin(data[ok]))
        hi = float(np.nanmax(data[ok]))
    if hi <= lo:
        hi = lo + 1e-6
    scaled = np.clip((data - lo) / (hi - lo), 0.0, 1.0)
    scaled[~ok] = 0.0
    return scaled


def _estimate_kernel_sigma_px(platesolve_dir: Path, photometry_dir: Path) -> float:
    """Median QC ``fwhm_px`` for this setup, else 2.5 px."""
    try:
        from hrd_analysis import _draft_dir_from_photometry, _obs_group_from_photometry
        from pipeline import find_qc_metrics_csv

        draft_dir = _draft_dir_from_photometry(Path(photometry_dir))
        obs_group = _obs_group_from_photometry(Path(photometry_dir))
        qc_csv = find_qc_metrics_csv(draft_dir, app_config=None)
        if qc_csv is None:
            return _DEFAULT_KERNEL_SIGMA_PX
        dfq = pd.read_csv(qc_csv, low_memory=False)
        if dfq.empty or "dst" not in dfq.columns:
            return _DEFAULT_KERNEL_SIGMA_PX
        m = dfq["dst"].astype(str).str.contains(str(obs_group), regex=False)
        sub = dfq.loc[m]
        if sub.empty:
            return _DEFAULT_KERNEL_SIGMA_PX
        fwhm = pd.to_numeric(sub.get("fwhm_px"), errors="coerce")
        med = float(fwhm.median()) if fwhm.notna().any() else math.nan
        if math.isfinite(med) and med > 0:
            return max(0.8, min(8.0, med))
    except Exception:  # noqa: BLE001
        logger.debug("FWHM estimate failed; using default sigma", exc_info=True)
    return _DEFAULT_KERNEL_SIGMA_PX


def _prepare_color_stars(
    ms_csv: Path,
    enrich_cache: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (all_dao_rows, colorable_rows with teff_k and flux)."""
    ms = read_vyvar_csv(ms_csv, low_memory=False)
    if ms.empty:
        return ms, ms
    flux_col = _resolve_flux_column(ms)
    if flux_col is not None:
        ms = ms[pd.to_numeric(ms[flux_col], errors="coerce") > 0].copy()
    ms["catalog_id"] = normalize_gaia_source_id_series(ms["catalog_id"])
    ms = ms[ms["catalog_id"].astype(str).str.len() > 0].copy()
    ms = ms[ms["catalog_id"].astype(str) != "0"].copy()

    teff_cache = _load_enrich_teff_cache(enrich_cache)
    bp = pd.to_numeric(ms.get("bp_rp"), errors="coerce")

    teff_out: list[float] = []
    for cid, bpr in zip(ms["catalog_id"].astype(str), bp, strict=False):
        t: float | None = teff_cache.get(cid)
        if t is None and pd.notna(bpr) and math.isfinite(float(bpr)):
            t = float(teff_from_bp_rp(float(bpr)))
        teff_out.append(float(t) if t is not None and math.isfinite(t) else math.nan)

    ms["teff_k"] = teff_out
    ms["bp_rp_num"] = bp
    if flux_col:
        ms["flux_use"] = pd.to_numeric(ms[flux_col], errors="coerce")
    else:
        ms["flux_use"] = 1.0

    colorable = ms[np.isfinite(ms["teff_k"]) & np.isfinite(ms["bp_rp_num"])].copy()
    return ms, colorable


def splat_chroma_layer(
    shape: tuple[int, int],
    xs: np.ndarray,
    ys: np.ndarray,
    rgbs: np.ndarray,
    amplitudes: np.ndarray,
    *,
    sigma_px: float,
) -> np.ndarray:
    """Amplitude-weighted Gaussian splat of per-star RGB chromaticities; neutral (1,1,1) elsewhere."""
    h, w = int(shape[0]), int(shape[1])
    rgb_acc = np.zeros((h, w, 3), dtype=np.float64)
    w_acc = np.zeros((h, w), dtype=np.float64)
    sigma = max(0.5, float(sigma_px))
    radius = int(math.ceil(3.0 * sigma))
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    stamp = np.exp(-0.5 * (xx * xx + yy * yy) / (sigma * sigma))

    for x0, y0, rgb, amp in zip(xs, ys, rgbs, amplitudes, strict=False):
        if not (math.isfinite(x0) and math.isfinite(y0) and math.isfinite(amp) and amp > 0):
            continue
        xi = int(round(x0))
        yi = int(round(y0))
        x_lo = xi - radius
        x_hi = xi + radius + 1
        y_lo = yi - radius
        y_hi = yi + radius + 1
        sx0 = max(0, x_lo)
        sy0 = max(0, y_lo)
        sx1 = min(w, x_hi)
        sy1 = min(h, y_hi)
        if sx0 >= sx1 or sy0 >= sy1:
            continue
        st_x0 = sx0 - x_lo
        st_y0 = sy0 - y_lo
        st_x1 = st_x0 + (sx1 - sx0)
        st_y1 = st_y0 + (sy1 - sy0)
        patch = stamp[st_y0:st_y1, st_x0:st_x1]
        wt = patch * float(amp)
        rgb_acc[sy0:sy1, sx0:sx1, :] += wt[..., np.newaxis] * rgb
        w_acc[sy0:sy1, sx0:sx1] += wt

    chroma = np.ones((h, w, 3), dtype=np.float64)
    mask = w_acc > 0
    chroma[mask] = rgb_acc[mask] / w_acc[mask, np.newaxis]
    return chroma


def _draw_caption(img_rgb: np.ndarray, caption: str) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil)
    font = None
    for path in (
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ):
        try:
            font = ImageFont.truetype(path, 14)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()
    margin = 8
    max_w = pil.width - 2 * margin
    words = caption.split()
    lines: list[str] = []
    line = ""
    for word in words:
        trial = f"{line} {word}".strip()
        bbox = draw.textbbox((0, 0), trial, font=font)
        if bbox[2] - bbox[0] <= max_w:
            line = trial
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    line_h = draw.textbbox((0, 0), "Ag", font=font)[3]
    bar_h = margin + len(lines) * (line_h + 2) + margin
    overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    odraw.rectangle([0, pil.height - bar_h, pil.width, pil.height], fill=(0, 0, 0, 170))
    pil = pil.convert("RGBA")
    pil.alpha_composite(overlay)
    draw = ImageDraw.Draw(pil)
    y = pil.height - bar_h + margin
    for ln in lines:
        draw.text((margin, y), ln, fill=(255, 255, 255, 255), font=font)
        y += line_h + 2
    return np.asarray(pil.convert("RGB"))


def render_catalog_color_field(
    platesolve_dir: Path,
    photometry_dir: Path,
    cfg: Any,
    out_png: Path,
) -> Path | None:
    """Render mono luminance x Gaia chrominance field PNG. Fail-open on missing inputs."""
    if not hrd_color_field_enabled(cfg):
        return None

    ps = Path(platesolve_dir)
    pt = Path(photometry_dir)
    out_png = Path(out_png)

    from hrd_analysis import _resolve_masterstar_fits_path, field_annotation_pixel_scale

    fits_path = _resolve_masterstar_fits_path(ps)
    if fits_path is None:
        log_event("HRD color field: MASTERSTAR FITS missing -- skipped")
        return None

    ms_csv = _resolve_masterstars_csv(ps, pt)
    if ms_csv is None:
        log_event("HRD color field: masterstars_full_match.csv missing -- skipped")
        return None

    luminance = _load_luminance_from_fits(fits_path)
    if luminance is None:
        log_event("HRD color field: could not load MASTERSTAR luminance -- skipped")
        return None

    enrich_cache = ps / "_hrd_cache" / "hrd_enrich.json"
    _all, colorable = _prepare_color_stars(ms_csv, enrich_cache)
    if colorable.empty:
        log_event("HRD color field: no matched stars with finite BP-RP -- skipped")
        return None

    saturation = hrd_color_saturation_from_cfg(cfg)
    teffs = colorable["teff_k"].to_numpy(dtype=np.float64)
    rgbs = teff_to_srgb_chroma(teffs, saturation=saturation)
    flux = colorable["flux_use"].to_numpy(dtype=np.float64)
    flux = np.where(np.isfinite(flux) & (flux > 0), flux, 1.0)
    amps = np.sqrt(flux)

    h, w = luminance.shape
    sx, sy, ok_scale = field_annotation_pixel_scale(ps, w, h, png_from_fits=True)
    if not ok_scale:
        log_event("HRD color field: pixel scale unknown vs MASTERSTAR FITS -- skipped")
        return None

    xs = pd.to_numeric(colorable["x"], errors="coerce").to_numpy(dtype=np.float64) * sx
    ys = pd.to_numeric(colorable["y"], errors="coerce").to_numpy(dtype=np.float64) * sy

    sigma = _estimate_kernel_sigma_px(ps, pt)
    chroma = splat_chroma_layer((h, w), xs, ys, rgbs, amps, sigma_px=sigma)

    rgb = np.clip(luminance[..., np.newaxis] * chroma, 0.0, 1.0)
    img_u8 = (rgb * 255.0).astype(np.uint8)
    img_u8 = _draw_caption(img_u8, HRD_COLORFIELD_CAPTION)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image

        Image.fromarray(img_u8, mode="RGB").save(str(out_png))
    except Exception:  # noqa: BLE001
        logger.exception("HRD color field: failed to save %s", out_png)
        log_event(f"HRD color field: failed to save PNG -- skipped ({out_png.name})")
        return None

    if not out_png.is_file():
        log_event("HRD color field: PNG not written -- skipped")
        return None
    return out_png


def color_field_stats(
    platesolve_dir: Path,
    photometry_dir: Path,
    *,
    render_seconds: float | None = None,
) -> dict[str, Any]:
    """Diagnostics for validation: colored fraction, BP-RP range, render time."""
    ps = Path(platesolve_dir)
    pt = Path(photometry_dir)
    ms_csv = _resolve_masterstars_csv(ps, pt)
    if ms_csv is None:
        return {"error": "no masterstars csv"}
    enrich_cache = ps / "_hrd_cache" / "hrd_enrich.json"
    all_dao, colorable = _prepare_color_stars(ms_csv, enrich_cache)
    n_dao = len(all_dao)
    n_color = len(colorable)
    bp = colorable["bp_rp_num"] if not colorable.empty else pd.Series(dtype=float)
    stats: dict[str, Any] = {
        "n_dao_matched": n_dao,
        "n_colored": n_color,
        "pct_colored": (100.0 * n_color / n_dao) if n_dao else 0.0,
        "bp_rp_min": float(bp.min()) if not bp.empty else None,
        "bp_rp_max": float(bp.max()) if not bp.empty else None,
        "bp_rp_median": float(bp.median()) if not bp.empty else None,
    }
    if render_seconds is not None:
        stats["render_seconds"] = float(render_seconds)
    return stats


def timed_render_catalog_color_field(
    platesolve_dir: Path,
    photometry_dir: Path,
    cfg: Any,
    out_png: Path,
) -> tuple[Path | None, float]:
    """Wrapper returning (path, elapsed_seconds) for validation scripts."""
    t0 = time.perf_counter()
    path = render_catalog_color_field(platesolve_dir, photometry_dir, cfg, out_png)
    return path, time.perf_counter() - t0
