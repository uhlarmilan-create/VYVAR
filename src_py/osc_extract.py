# -*- coding: ascii -*-
"""OSC Bayer channel extraction (plane split, average semantics, no demosaic).

Channels: oneRGGB (internal/diagnostics), R, G, B. Each is a half-size mono plane
from the calibrated CFA mosaic, optionally NxN average-binned (``osc_channel_binning``).

Effective gain/RN after plane averaging and bin averaging (Poisson + read noise):

    n_avg = plane_count * bin_n**2
    gain_eff = gain_raw * n_avg
    rn_eff = rn_raw * sqrt(n_avg)

where plane_count is 1 (R/B), 2 (G), or 4 (oneRGGB). ADU scale is preserved by
AVERAGE semantics (not SUM), so Poisson variance per output ADU scales as 1/gain_eff.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from fits_suffixes import path_suffix_is_fits

OSC_CHANNELS: tuple[str, ...] = ("oneRGGB", "R", "G", "B")
CANONICAL_BAYERMASKS: frozenset[str] = frozenset({"RGGB", "BGGR", "GBRG", "GRBG"})
PLANE_AVG_COUNT: dict[str, int] = {"R": 1, "G": 2, "B": 1, "oneRGGB": 4}

COPY_HEADER_KEYS: tuple[str, ...] = (
    "DATE-OBS",
    "EXPTIME",
    "EXPOSURE",
    "GAIN",
    "EGAIN",
    "CCD-TEMP",
    "INSTRUME",
    "TELESCOP",
    "FOCALLEN",
    "FOCLEN",
    "OBJCTRA",
    "OBJCTDEC",
    "RA",
    "DEC",
    "IMAGETYP",
    "SITELAT",
    "SITELONG",
    "SITELON",
    "LAT-OBS",
    "LONG-OBS",
    "ALT-OBS",
    "XORGSUBF",
    "YORGSUBF",
    "JD",
    "JD-OBS",
    "MJD-OBS",
    "FILTER",
)


def normalize_bayermask(value: str | None) -> str | None:
    """Return canonical Bayer mask, ``mono``, or None (empty == mono)."""
    if value is None:
        return None
    s = str(value).strip().upper()
    if not s or s == "MONO":
        return None
    if s in CANONICAL_BAYERMASKS:
        return s
    raise ValueError(
        f"Invalid BAYERMASK {value!r}; allowed: RGGB, BGGR, GBRG, GRBG, mono, or empty."
    )


def is_osc_bayermask(bayermask: str | None) -> bool:
    return normalize_bayermask(bayermask) is not None


def valid_bayer_pattern_4(s: str | None) -> str | None:
    if not s:
        return None
    p = "".join(str(s).upper().split())
    if len(p) < 4:
        return None
    p = p[:4]
    if not all(c in "RGB" for c in p):
        return None
    if p.count("R") != 1 or p.count("B") != 1 or p.count("G") != 2:
        return None
    return p


def bayer_planes_from_mosaic(data: np.ndarray, pattern: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return R, G1, G2, B planes each (H/2, W/2) for any 4-char Bayer pattern."""
    pat = valid_bayer_pattern_4(pattern)
    if pat is None:
        raise ValueError(f"Unsupported Bayer pattern: {pattern!r}")
    d = np.asarray(data, dtype=np.float32)
    if d.ndim != 2:
        raise ValueError(f"Expected 2D mosaic, got shape {d.shape}")
    h, w = d.shape
    if h % 2 or w % 2:
        d = d[: h - (h % 2), : w - (w % 2)]
    coords = ((0, 0, pat[0]), (0, 1, pat[1]), (1, 0, pat[2]), (1, 1, pat[3]))
    r = g1 = g2 = b = None
    g_seen = 0
    for row, col, ch in coords:
        sl = d[row::2, col::2]
        if ch == "R":
            r = sl
        elif ch == "B":
            b = sl
        elif ch == "G":
            if g_seen == 0:
                g1 = sl
                g_seen = 1
            else:
                g2 = sl
    if r is None or g1 is None or g2 is None or b is None:
        raise ValueError(f"Pattern {pat!r} did not yield R/G/B planes")
    return r, g1, g2, b


def derive_channel_planes(
    r: np.ndarray, g1: np.ndarray, g2: np.ndarray, b: np.ndarray
) -> dict[str, np.ndarray]:
    g = (g1 + g2) * 0.5
    one = (r + g1 + g2 + b) * 0.25
    return {
        "R": np.asarray(r, dtype=np.float32),
        "G": np.asarray(g, dtype=np.float32),
        "B": np.asarray(b, dtype=np.float32),
        "oneRGGB": np.asarray(one, dtype=np.float32),
    }


def average_bin_2d(arr: np.ndarray, bin_n: int) -> np.ndarray:
    """NxN average binning preserving ADU scale."""
    n = max(1, int(bin_n))
    a = np.asarray(arr, dtype=np.float32)
    if n == 1:
        return a.copy()
    h, w = a.shape
    h2 = (h // n) * n
    w2 = (w // n) * n
    if h2 < n or w2 < n:
        return a.copy()
    a = a[:h2, :w2]
    return a.reshape(h2 // n, n, w2 // n, n).mean(axis=(1, 3)).astype(np.float32)


def effective_gain_rn(
    gain_e_per_adu: float,
    read_noise_e: float,
    channel: str,
    osc_bin: int,
) -> tuple[float, float]:
    """Effective (gain [e-/ADU], read_noise [e-]) for extracted+binned channel pixels."""
    ch = str(channel)
    if ch not in PLANE_AVG_COUNT:
        raise ValueError(f"Unknown channel {channel!r}")
    plane_n = int(PLANE_AVG_COUNT[ch])
    bin_n = max(1, int(osc_bin))
    n_avg = float(plane_n * bin_n * bin_n)
    g = float(gain_e_per_adu)
    rn = float(read_noise_e)
    if not math.isfinite(g) or g <= 0:
        raise ValueError(f"Invalid gain_e_per_adu={gain_e_per_adu}")
    if not math.isfinite(rn) or rn < 0:
        raise ValueError(f"Invalid read_noise_e={read_noise_e}")
    gain_eff = g * n_avg
    rn_eff = rn * math.sqrt(n_avg)
    return gain_eff, rn_eff


def superpixel_scale_factor(osc_bin: int) -> float:
    """Total linear scale vs raw mosaic: 2 (Bayer superpixel) * osc_channel_binning."""
    return 2.0 * max(1, int(osc_bin))


def checkerboard_column_delta(data: np.ndarray) -> float:
    """Mean abs difference between even/odd columns (Bayer checkerboard diagnostic)."""
    d = np.asarray(data, dtype=np.float64)
    if d.ndim != 2 or d.shape[1] < 2:
        return 0.0
    even = d[:, 0::2]
    odd = d[:, 1::2]
    n = min(even.shape[1], odd.shape[1])
    if n == 0:
        return 0.0
    return float(np.mean(np.abs(even[:, :n] - odd[:, :n])))


def channel_obs_group_folder(base_folder: str, channel: str) -> str:
    return f"{base_folder}_{channel}"


def is_channel_obs_group_folder(name: str) -> bool:
    n = str(name)
    for ch in OSC_CHANNELS:
        if n.endswith(f"_{ch}"):
            return True
    return False


def build_channel_header(
    src_hdr: fits.Header,
    *,
    channel: str,
    bayermask: str,
    osc_bin: int,
    gain_eff: float,
    rn_eff: float,
    src_name: str,
) -> fits.Header:
    hdr = fits.Header()
    for key in COPY_HEADER_KEYS:
        if key in src_hdr:
            hdr[key] = src_hdr[key]
    scale = superpixel_scale_factor(osc_bin)
    xpix = src_hdr.get("XPIXSZ") or src_hdr.get("PIXSIZE") or src_hdr.get("PIXSIZE1")
    try:
        pix_native = float(xpix) if xpix not in (None, "") else None
    except (TypeError, ValueError):
        pix_native = None
    if pix_native is not None and math.isfinite(pix_native) and pix_native > 0:
        eff_pix = pix_native * scale
        hdr["XPIXSZ"] = eff_pix
        hdr["YPIXSZ"] = eff_pix
    bin_out = max(1, int(src_hdr.get("XBINNING") or src_hdr.get("BINNING") or 1)) * int(scale)
    hdr["XBINNING"] = int(bin_out)
    hdr["YBINNING"] = int(bin_out)
    hdr["BINNING"] = int(bin_out)
    hdr["VY_CHANNEL"] = (channel, "OSC extracted channel token")
    hdr["VY_BAYERMASK"] = (bayermask, "Equipment Bayer mask used for extraction")
    hdr["VY_OSC_BIN"] = (int(osc_bin), "Post-extraction average binning factor N")
    hdr["EGAIN"] = (float(gain_eff), "Effective gain e-/ADU after OSC extract+bin")
    hdr["VY_EGAIN"] = (float(gain_eff), "Effective gain e-/ADU (VYVAR OSC)")
    hdr["VY_RDNOIS"] = (float(rn_eff), "Effective read noise e- after OSC extract+bin")
    hdr["VY_OSCSRC"] = (str(src_name), "Source CFA mosaic FITS")
    if "BAYERPAT" in hdr:
        del hdr["BAYERPAT"]
    return hdr


def extract_one_light_to_channels(
    src_path: Path,
    *,
    out_dirs: dict[str, Path],
    bayermask: str,
    osc_bin: int,
    gain_e_per_adu: float,
    read_noise_e: float,
) -> dict[str, Path]:
    """Extract one calibrated mosaic FITS into four channel outputs."""
    with fits.open(src_path, memmap=False) as hdul:
        src_hdr = hdul[0].header
        data = np.asarray(hdul[0].data, dtype=np.float32)
        hdr_pat = valid_bayer_pattern_4(str(src_hdr.get("BAYERPAT") or "")) or bayermask
    r, g1, g2, b = bayer_planes_from_mosaic(data, hdr_pat)
    planes = derive_channel_planes(r, g1, g2, b)
    written: dict[str, Path] = {}
    for ch in OSC_CHANNELS:
        out_dir = out_dirs[ch]
        out_dir.mkdir(parents=True, exist_ok=True)
        ch_data = average_bin_2d(planes[ch], osc_bin)
        g_eff, rn_eff = effective_gain_rn(gain_e_per_adu, read_noise_e, ch, osc_bin)
        ch_hdr = build_channel_header(
            src_hdr,
            channel=ch,
            bayermask=bayermask,
            osc_bin=osc_bin,
            gain_eff=g_eff,
            rn_eff=rn_eff,
            src_name=src_path.name,
        )
        out_path = out_dir / src_path.name
        fits.writeto(out_path, ch_data, ch_hdr, overwrite=True)
        written[ch] = out_path
    return written


def validate_bayer_crosscheck(
    *,
    fits_bayerpat: str | None,
    equipment_bayermask: str | None,
) -> tuple[str, str | None]:
    """Return (verdict, message) where verdict is ok|warn|fail."""
    hdr_pat = valid_bayer_pattern_4(fits_bayerpat)
    eq_pat = normalize_bayermask(equipment_bayermask)
    if hdr_pat and not eq_pat:
        return (
            "fail",
            "OSC mosaic detected (FITS BAYERPAT="
            f"{hdr_pat}); set EQUIPMENTS.BAYERMASK to RGGB/BGGR/GBRG/GRBG "
            "(not mono/empty) before science import.",
        )
    if hdr_pat and eq_pat and hdr_pat != eq_pat:
        return (
            "warn",
            f"BAYERPAT header ({hdr_pat}) differs from EQUIPMENTS.BAYERMASK ({eq_pat}); "
            "extraction uses equipment mask.",
        )
    return ("ok", None)


def iter_mosaic_light_fits(group_dir: Path) -> list[Path]:
    out: list[Path] = []
    for fp in sorted(group_dir.iterdir()):
        if not fp.is_file() or not path_suffix_is_fits(fp):
            continue
        try:
            with fits.open(fp, memmap=False) as hdul:
                if hdul[0].header.get("VY_CHANNEL"):
                    continue
                if valid_bayer_pattern_4(str(hdul[0].header.get("BAYERPAT") or "")):
                    out.append(fp)
        except OSError:
            continue
    return out
