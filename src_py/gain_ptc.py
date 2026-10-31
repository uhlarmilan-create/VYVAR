"""Gain and read noise from raw flat and dark pairs (photon transfer; CAMERA-PROFILE-2026-10).

For two flats F1, F2 taken one after the other with the same settings and exposure, the difference
removes the fixed pattern (vignetting, dust, PRNU) and leaves only temporal noise:

    var(F1 - F2) / 2 = S / g + RN^2 / g^2           (S = mean level above the dark, ADU)

so the variance against the level is a line with slope 1/g (Janesick 2007, "Photon Transfer";
Howell 2006, "Handbook of CCD Astronomy", sec. 4.3). Read noise comes from two short darks:
RN = g * std(D1 - D2) / sqrt(2). The flat spread is the MAD (x 1.4826) of the difference image (robust
to cosmic rays; at flat levels the noise is many ADU, so the integer steps do not bias it); the dark
spread is the plain standard deviation (read noise is only a few ADU there and the MAD of integer
data would be biased, while hot pixels cancel in the difference). No clipping, no iteration; central
half of the frame.

Works only on raw, unmastered frames: a master is an average and has lost the noise this method
measures. The result is in the ADU of the files (for a 14-bit camera writing 16-bit files: container
ADU); ``native_gain`` converts it to the camera's native ADU with ``ADC_BITS``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

#: Pairs above this fraction of the saturation level are left out (non-linearity near full well).
MAX_LEVEL_FRACTION = 0.7
#: Two frames form a pair when their mean levels differ by at most this fraction.
PAIR_LEVEL_TOL = 0.05
#: Central part of the frame used (fraction of width / height) - avoids vignetting and edge glow.
ROI_FRACTION = 0.5
MAD_TO_SIGMA = 1.4826


@dataclass
class PairPoint:
    files: tuple[str, str]
    exptime: float
    level_adu: float
    var_half: float
    used: bool = True
    note: str = ""


@dataclass
class PtcResult:
    ok: bool
    gain: float | None = None  # e-/ADU of the files
    gain_err: float | None = None
    read_noise_e: float | None = None
    read_noise_adu: float | None = None
    r2: float | None = None
    n_levels: int = 0
    binning: int = 1
    gain_setting: float | None = None
    points: list[PairPoint] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)


def _roi(a: np.ndarray) -> np.ndarray:
    h, w = a.shape[-2:]
    dy, dx = int(h * (1 - ROI_FRACTION) / 2), int(w * (1 - ROI_FRACTION) / 2)
    return a[..., dy:h - dy, dx:w - dx]


def _mad_sigma(d: np.ndarray) -> float:
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("nan")
    return float(MAD_TO_SIGMA * np.median(np.abs(d - np.median(d))))


def pair_variance(f1: np.ndarray, f2: np.ndarray, dark: np.ndarray | float = 0.0) -> tuple[float, float]:
    """(mean level above the dark, var(F1 - F2) / 2) on the central region, in ADU."""
    a, b = _roi(np.asarray(f1, dtype=np.float64)), _roi(np.asarray(f2, dtype=np.float64))
    dk = _roi(np.asarray(dark, dtype=np.float64)) if np.ndim(dark) else float(dark)
    return _pair_variance_roi(a, b, dk)


def _pair_variance_roi(a: np.ndarray, b: np.ndarray, dk: np.ndarray | float) -> tuple[float, float]:
    """:func:`pair_variance` on frames already cut to the central region (float64)."""
    level = float(np.median((a + b) / 2.0 - dk))
    sig = _mad_sigma(a - b)
    return level, sig * sig / 2.0


def read_noise_adu(d1: np.ndarray, d2: np.ndarray) -> float:
    """Read noise in ADU from two darks of the same (short) exposure."""
    return _read_noise_adu_roi(_roi(np.asarray(d1, dtype=np.float64)), _roi(np.asarray(d2, dtype=np.float64)))


def _read_noise_adu_roi(a: np.ndarray, b: np.ndarray) -> float:
    """:func:`read_noise_adu` on darks already cut to the central region (float64)."""
    d = (a - b)[np.isfinite(a - b)]
    return float(np.std(d)) / math.sqrt(2.0) if d.size else float("nan")


def fit_ptc(points: list[PairPoint], rn_adu: float | None) -> tuple[float, float, float, float]:
    """Gain, its 1-sigma error, intercept and R^2 from the used pair points.

    Two or more levels: least-squares line var = S / g + c. One level: g = S / (var - RN_adu^2)
    (needs the dark read noise); its error is not defined (NaN)."""
    use = [p for p in points if p.used]
    x = np.array([p.level_adu for p in use], dtype=np.float64)
    y = np.array([p.var_half for p in use], dtype=np.float64)
    if len(use) >= 2 and np.ptp(x) > 0:
        A = np.vstack([x, np.ones_like(x)]).T
        coef, res, _, _ = np.linalg.lstsq(A, y, rcond=None)
        slope, icpt = float(coef[0]), float(coef[1])
        yhat = A @ coef
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        if len(use) > 2:
            s2 = ss_res / (len(use) - 2)
            slope_err = math.sqrt(s2 / float(np.sum((x - x.mean()) ** 2)))
        else:
            slope_err = float("nan")
        if slope <= 0:
            return float("nan"), float("nan"), icpt, r2
        return 1.0 / slope, slope_err / (slope * slope), icpt, r2
    if len(use) == 1 and rn_adu is not None and math.isfinite(rn_adu):
        den = float(y[0]) - rn_adu * rn_adu
        if den > 0:
            return float(x[0]) / den, float("nan"), rn_adu * rn_adu, float("nan")
    return float("nan"), float("nan"), float("nan"), float("nan")


def _hdr_num(h: Any, *keys: str) -> float | None:
    for k in keys:
        try:
            v = float(h[k])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(v):
            return v
    return None


def _load_header(path: Path) -> Any:
    from astropy.io import fits  # noqa: PLC0415

    with fits.open(path, memmap=False) as hd:
        return hd[0].header.copy()


def _load_roi(path: Path) -> np.ndarray:
    """The central region (:func:`_roi`) of the frame as float64 - G01-35: only the region the
    measurement uses is kept (the full frames were all held in memory as float64)."""
    from astropy.io import fits  # noqa: PLC0415

    with fits.open(path, memmap=False) as hd:
        return np.array(_roi(np.asarray(hd[0].data)), dtype=np.float64)


def measure_from_files(
    flat_files: list[Path],
    dark_files: list[Path],
    *,
    saturation_adu: float,
) -> PtcResult:
    """Photon transfer from raw flats (pairs of equal exposure) and darks (pairs of equal exposure).

    ``saturation_adu`` is required (NEW-4: no assumed 16-bit level; the GUI takes the camera's SATURATE_ADU
    or asks for it).

    Every flat exposure needs darks of the same exposure (dark level and amp glow); the shortest dark
    pair gives the read noise. All frames must share binning and gain setting."""
    res = PtcResult(ok=False)
    # G01-35: headers first (the binning / gain check needs no pixels), then the central region only
    flat_hdrs = [(p, _load_header(p)) for p in sorted({Path(f) for f in flat_files})]
    dark_hdrs = [(p, _load_header(p)) for p in sorted({Path(f) for f in dark_files})]
    if len(flat_hdrs) < 2:
        res.messages.append("At least two raw flats of the same exposure are needed (better 5 levels x 2).")
        return res
    keys = {(_hdr_num(h, "XBINNING") or 1, _hdr_num(h, "GAIN")) for _, h in flat_hdrs + dark_hdrs}
    if len(keys) > 1:
        res.messages.append(f"Frames differ in binning / gain setting: {sorted(keys, key=str)} - use one setting.")
        return res
    frames: list[tuple[Path, np.ndarray, Any]] = [(p, _load_roi(p), h) for p, h in flat_hdrs]
    darks: list[tuple[Path, np.ndarray, Any]] = [(p, _load_roi(p), h) for p, h in dark_hdrs]
    res.binning = int((next(iter(keys))[0]) or 1)
    res.gain_setting = next(iter(keys))[1]

    def _exp(h: Any) -> float:
        return float(_hdr_num(h, "EXPTIME", "EXPOSURE") or 0.0)

    dark_by_exp: dict[float, list[np.ndarray]] = {}
    for _, a, h in darks:
        dark_by_exp.setdefault(round(_exp(h), 4), []).append(a)
    rn_adu = None
    if dark_by_exp:
        e_min = min(dark_by_exp)
        if len(dark_by_exp[e_min]) >= 2:
            rn_adu = _read_noise_adu_roi(dark_by_exp[e_min][0], dark_by_exp[e_min][1])
            res.read_noise_adu = rn_adu
        else:
            res.messages.append("Only one dark at the shortest exposure - read noise needs a dark pair.")
    else:
        res.messages.append("No darks: the dark level is not removed and read noise is not measured.")

    by_exp: dict[float, list[tuple[Path, np.ndarray]]] = {}
    for p, a, h in frames:
        by_exp.setdefault(round(_exp(h), 4), []).append((p, a))
    for e, fl in sorted(by_exp.items()):
        dks = dark_by_exp.get(e)
        if dks:
            dark = np.mean(np.stack(dks), axis=0) if len(dks) > 1 else dks[0]
        elif dark_by_exp:
            near = min(dark_by_exp, key=lambda k: abs(k - e))
            dark = np.mean(np.stack(dark_by_exp[near]), axis=0)
            res.messages.append(f"Flats {e:g} s: no dark of that exposure, dark of {near:g} s used.")
        else:
            dark = 0.0
        for (p1, a1), (p2, a2) in zip(fl[0::2], fl[1::2]):
            lev, var_h = _pair_variance_roi(a1, a2, dark)
            m1, m2 = float(np.median(a1)), float(np.median(a2))
            pt = PairPoint((p1.name, p2.name), e, lev, var_h)
            if abs(m1 - m2) > PAIR_LEVEL_TOL * max(abs(m1), abs(m2), 1.0):
                pt.used, pt.note = False, "levels of the two flats differ (light changed)"
            elif max(m1, m2) > MAX_LEVEL_FRACTION * float(saturation_adu):
                pt.used, pt.note = False, f"above {MAX_LEVEL_FRACTION:.0%} of saturation"
            elif lev <= 0:
                pt.used, pt.note = False, "level not above the dark"
            res.points.append(pt)
        if len(fl) % 2:
            res.messages.append(f"Flats {e:g} s: odd number of frames, the last one is not paired.")
    g, g_err, _, r2 = fit_ptc(res.points, rn_adu)
    res.n_levels = sum(p.used for p in res.points)
    if not (math.isfinite(g) and g > 0):
        res.messages.append("No usable gain: check that the flats are raw, unsaturated and of steady light.")
        return res
    res.ok, res.gain, res.r2 = True, float(g), (float(r2) if math.isfinite(r2) else None)
    res.gain_err = float(g_err) if math.isfinite(g_err) else None
    if rn_adu is not None:
        res.read_noise_e = float(g * rn_adu)
    if res.n_levels < 3:
        res.messages.append(f"Only {res.n_levels} usable level(s): take flats at 3 or more levels for a reliable line.")
    return res


#: FITS integer container of the frames (bits).
CONTAINER_BITS = 16


def container_scale(adc_bits: int | None) -> float:
    """Factor between native ADU and the values in the 16-bit files: an N-bit camera writes ADU x 2^(16-N)
    (1.0 when the bit depth is unknown or outside 1..16). G14-18: the one definition."""
    if adc_bits is None or not (1 <= int(adc_bits) <= CONTAINER_BITS):
        return 1.0
    return float(2 ** (CONTAINER_BITS - int(adc_bits)))


def file_gain(gain_native: float, adc_bits: int | None) -> float:
    """Native ADC gain (e-/ADU) -> gain of the 16-bit file values (e-/file ADU)."""
    return float(gain_native) / container_scale(adc_bits)


def native_gain(gain_files: float, adc_bits: int | None) -> float:
    """Gain of the files -> native ADC gain (files of an N-bit camera hold values x 2^(16-N))."""
    return float(gain_files) * container_scale(adc_bits)
