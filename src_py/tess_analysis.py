# -*- coding: ascii -*-
"""TESS full-frame-image analysis of one star (TESS-2026-10).

Pipeline per star (no sigma clipping, no detrending filter):

1. TessCut cutouts of every sector (lightkurve), cadences with TESS quality flags
   removed (``quality_bitmask="hard"``: includes scattered light, momentum dumps...).
2. Gaia DR3 neighbours (local DB) projected on the cutout; T-band flux from G_RP
   (proxy used by TESSILATOR, Binks & Guenther 2024).
3. PRF width measured on the sector's median image (Gaussian fit to the brightest
   isolated star); every source modelled with that PRF.
4. Aperture: pixels ordered by target / (target + neighbours) light, grown while the
   target S/N increases. Contamination = neighbour light / target light in it.
5. Background per cadence: median of pixels where the model puts < 1 % of the
   brightest source's peak (catalogue mask, no clipping).
6. Light curve in magnitudes, each orbit segment (gaps > 0.5 d) on its own median.
7. Periodograms on all sectors together:
   - multi-harmonic AoV (Schwarzenberg-Czerny 1996), primary - best for non-sinusoidal
     curves (Paunzen & Vanmunster 2016);
   - Lomb-Scargle (VanderPlas 2018) with Baluev (2008) false-alarm probability;
   - PDM (Stellingwerf 1978) at the strongest AoV peaks;
   - BLS (Kovacs et al. 2002) for eclipses.
   False-alarm probability of AoV from its F distribution with the number of
   independent frequencies; period error after Schwarzenberg-Czerny (1991): half width
   of the peak at (peak - mean noise power level), measured on a grid that resolves it.
8. Period = k x (strongest LS peak): k = 1..4 by BIC of Fourier models with the
   harmonics of f/k (Schwarz 1978) - a narrow eclipse puts its power at 2-4 f -, and
   k = 2 also when the two minima or maxima at 2P differ significantly. Equal minima
   and maxima: P with the note that an EW/ELL orbital period would be 2P.
9. Who varies: per-pixel amplitude at the best frequency fitted with the PRF placed
   on every Gaia source (as TESS_localize, Higgins & Bell 2023) -> relative likelihood.
10. Verdict: AMBIGUOUS when FAP >= 0.01 or the amplitude S/N against the local
    residual spectrum (+-1 c/d, red noise included) is < 4 (Breger et al. 1993) - nothing
    to localise; RELIABLE when the target is the most likely source (relative
    likelihood >= 0.5), contamination (neighbours / target in the aperture) <
    ``TESS_CONTAM_MAX`` and the period of >= 2 sectors (and most) agrees within 3 sigma of
    each sector's own error; NEIGHBOUR when another star is the most likely source;
    otherwise AMBIGUOUS with the reasons. A period is printed in the report only
    for RELIABLE.

The analysis is always available from the UI; ``tess_enabled`` in config.json only
decides whether the night run starts it automatically (tess_runner).
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

LOGGER = logging.getLogger(__name__)

#: RELIABLE needs the contamination (neighbours / target in the aperture) below this fraction.
TESS_CONTAM_MAX = 0.10
_SEGMENT_GAP_D = 0.5  # gap separating TESS orbits / downlinks
_N_HARM = 3  # harmonics of the multi-harmonic AoV
_OVERSAMPLE = 5
_BKG_MODEL_FRAC = 0.01  # background pixels: model light < 1 % of the brightest peak


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------
@dataclass
class PeriodSolution:
    method: str
    period_d: float
    period_err_d: float
    fap: float
    power: float
    amplitude_mag: float = float("nan")
    epoch_min_bjd: float = float("nan")
    note: str = ""


@dataclass
class TessSector:
    sector: int
    cadence_s: float
    n_points: int
    t_start: float
    t_end: float
    contamination: float
    n_aperture_px: int
    prf_sigma_px: float
    period: PeriodSolution | None = None
    error: str | None = None
    # sector period brought to the multiple (P, 2P, ...) chosen for all sectors
    period_common_d: float = float("nan")


@dataclass
class TessAnalysis:
    catalog_id: str
    ra: float
    dec: float
    sectors: list[TessSector] = field(default_factory=list)
    periods: dict[str, PeriodSolution] = field(default_factory=dict)
    best: PeriodSolution | None = None
    two_p_preferred: bool = False
    period_multiplier: int = 1  # period = multiplier / f(strongest Lomb-Scargle peak)
    localization: dict[str, float] = field(default_factory=dict)
    target_likelihood: float = float("nan")
    snr_amplitude: float = float("nan")  # Breger et al. (1993) S/N of the strongest harmonic
    at_long_period_limit: bool = False
    contamination: float = float("nan")
    verdict: str = "NO_DATA"
    reasons: list[str] = field(default_factory=list)
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        def _clean(o: Any) -> Any:
            if isinstance(o, float) and not math.isfinite(o):
                return None
            if isinstance(o, dict):
                return {k: _clean(v) for k, v in o.items()}
            if isinstance(o, list):
                return [_clean(v) for v in o]
            return o

        return _clean(asdict(self))


# ---------------------------------------------------------------------------
# PRF, aperture, background
# ---------------------------------------------------------------------------
def _erf_box(x0: float, sigma: float, n: int) -> np.ndarray:
    from scipy.special import erf  # noqa: PLC0415

    e = np.arange(n + 1) - 0.5
    c = 0.5 * (1.0 + erf((e - x0) / (math.sqrt(2.0) * sigma)))
    return np.diff(c)


def prf_image(shape: tuple[int, int], x: float, y: float, sigma: float) -> np.ndarray:
    """Fraction of a Gaussian PRF (sigma px) centred at (x, y) falling in each pixel."""
    ny, nx = shape
    return np.outer(_erf_box(y, sigma, ny), _erf_box(x, sigma, nx))


def measure_prf_sigma(img: np.ndarray, x: float, y: float) -> float:
    """Gaussian sigma of a star at (x, y) on a median image (least squares)."""
    from scipy.optimize import least_squares  # noqa: PLC0415

    ny, nx = img.shape
    xi, yi = int(round(x)), int(round(y))
    h = 3
    y0, y1, x0, x1 = max(0, yi - h), min(ny, yi + h + 1), max(0, xi - h), min(nx, xi + h + 1)
    sub = img[y0:y1, x0:x1]
    if sub.size < 9 or not np.isfinite(sub).all():
        return float("nan")

    def _res(p: np.ndarray) -> np.ndarray:
        a, b, s, dx, dy = p
        m = a * prf_image(sub.shape, x - x0 + dx, y - y0 + dy, s) + b
        return (m - sub).ravel()

    p0 = [float(np.nansum(sub) - sub.size * np.nanmin(sub)), float(np.nanmin(sub)), 0.8, 0.0, 0.0]
    try:
        r = least_squares(_res, p0, bounds=([0, -np.inf, 0.3, -1, -1], [np.inf, np.inf, 3.0, 1, 1]))
    except Exception:  # noqa: BLE001
        return float("nan")
    return float(r.x[2])


def choose_aperture(
    target_img: np.ndarray, contam_img: np.ndarray, bkg_var_px: float, t_exp_s: float = 1.0
) -> tuple[np.ndarray, float]:
    """Aperture maximising target S/N; returns (mask, contamination = neighbours/target,
    the TIC 'contratio' definition, Stassun et al. 2018).

    Images in e-/s; ``bkg_var_px`` is the measured per-pixel variance of the rate
    ((e-/s)^2) and the source shot noise of a rate R is R / t_exp ((e-/s)^2)."""
    tot = target_img + contam_img
    ratio = np.where(tot > 0, target_img / np.where(tot > 0, tot, 1.0), 0.0)
    order = np.argsort(-(ratio * target_img).ravel())
    mask = np.zeros(target_img.size, bool)
    best_snr, best_n = -1.0, 0
    s_t = s_tot = 0.0
    for k, i in enumerate(order, start=1):
        s_t += target_img.ravel()[i]
        s_tot += tot.ravel()[i]
        snr = s_t / math.sqrt(max(s_tot / max(t_exp_s, 1e-9) + k * bkg_var_px, 1e-30))
        if snr > best_snr:
            best_snr, best_n = snr, k
    mask[order[:best_n]] = True
    mask = mask.reshape(target_img.shape)
    t_in = float(target_img[mask].sum())
    c_in = float(contam_img[mask].sum())
    return mask, (c_in / t_in if t_in > 0 else float("nan"))


# ---------------------------------------------------------------------------
# periodograms
# ---------------------------------------------------------------------------
def _segments(t: np.ndarray) -> np.ndarray:
    seg = np.zeros(t.size, int)
    if t.size > 1:
        seg[1:] = np.cumsum(np.diff(t) > _SEGMENT_GAP_D)
    return seg


def frequency_grid(t: np.ndarray, cadence_d: float) -> np.ndarray:
    """From 2 / (median segment length) to the Nyquist frequency, oversampled 5x."""
    seg = _segments(t)
    lens = [float(np.ptp(t[seg == s])) for s in np.unique(seg) if (seg == s).sum() > 10]
    t_seg = float(np.median(lens)) if lens else float(np.ptp(t))
    f_min = 2.0 / max(t_seg, 1e-3)
    f_max = 0.5 / max(cadence_d, 1e-4)
    df = 1.0 / (_OVERSAMPLE * max(float(np.ptp(t)), 1e-3))
    n = int(min(2_000_000, max(100, (f_max - f_min) / df)))
    return np.linspace(f_min, f_max, n)


def _zoom_peak(
    t: np.ndarray, m: np.ndarray, e: np.ndarray, f_c: float, half: float,
    f_lo: float, f_hi: float, nterms: int = _N_HARM, levels: int = 8,
) -> tuple[float, float, float]:
    """Best frequency near ``f_c`` of an ``nterms`` Fourier fit and its 1-sigma error.

    Returns (f, AoV Theta at f, sigma_f). The error is the least-squares one: the
    half width of chi^2(f) at chi^2_min + 1 with the errors rescaled to chi^2_red = 1,
    times sqrt(D) where D is the correlation length of the residuals from their number
    of sign runs (Schwarzenberg-Czerny 1991). chi^2(f) = chi^2_0 (1 - p(f)) with p the
    'standard' power, so the level is p_max - (1 - p_max) / (N - 2 nterms - 1). The grid is
    refined until >= 20 samples cross the peak; the search stays inside [f_lo, f_hi].
    """
    from astropy.timeseries import LombScargle  # noqa: PLC0415

    n = t.size
    dof = max(n - 2 * nterms - 1, 1)
    ls = LombScargle(t, m, e, nterms=nterms, normalization="standard")
    lo_f, hi_f = f_c - half, f_c + half
    fz = pz = None
    k = lo = hi = 0
    if not (f_lo < f_hi):
        return float(f_c), float("nan"), float("nan")
    f_c = min(max(f_c, f_lo), f_hi)
    for _ in range(levels):
        lo_f, hi_f = max(lo_f, f_lo), min(hi_f, f_hi)
        if not (hi_f - lo_f > 1e-9 * max(abs(hi_f), 1e-12)):
            # window collapsed (peak on a range edge) or below float resolution
            if fz is not None:
                break
            lo_f, hi_f = max(f_lo, f_c - half), min(f_hi, f_c + half)
            if not (hi_f - lo_f > 1e-9 * max(abs(hi_f), 1e-12)):
                break
        fz = np.linspace(lo_f, hi_f, 401)
        pz = np.clip(np.asarray(ls.power(fz, method="fastchi2" if nterms > 1 else "slow"), dtype=float), 0.0, 1.0 - 1e-12)
        k = int(np.nanargmax(pz))
        lvl = pz[k] - (1.0 - pz[k]) / dof
        lo = k
        while lo > 0 and pz[lo] > lvl:
            lo -= 1
        hi = k
        while hi < fz.size - 1 and pz[hi] > lvl:
            hi += 1
        at_edge = (lo == 0 and lo_f > f_lo) or (hi == fz.size - 1 and hi_f < f_hi)
        if at_edge:
            span = hi_f - lo_f
            lo_f, hi_f = fz[k] - 1.5 * span, fz[k] + 1.5 * span
            continue
        if hi - lo >= 20 or lo == 0 or hi == fz.size - 1:
            break
        hw = max(hi - lo, 2) * (fz[1] - fz[0]) * 3.0
        lo_f, hi_f = fz[k] - hw, fz[k] + hw
    if fz is None or pz is None:
        return float(f_c), float("nan"), float("nan")
    f_best = float(fz[k])
    sig_f = 0.5 * float(fz[hi] - fz[lo])
    try:
        r = m - ls.model(t, f_best)
        runs = 1 + int(np.sum(np.diff(np.sign(r - np.median(r))) != 0))
        d_corr = max(1.0, 0.5 * (n + 1) / max(runs, 1))
    except Exception:  # noqa: BLE001
        d_corr = 1.0
    sig_f *= math.sqrt(d_corr)
    theta = dof / (2.0 * nterms) * pz[k] / (1.0 - pz[k])
    return f_best, float(theta), float(sig_f)


def _harmonic_bic(t: np.ndarray, m: np.ndarray, e: np.ndarray, f0: float, k: int, n_h: int = _N_HARM) -> float:
    """BIC (Schwarz 1978) of a Fourier model with the harmonics j f0 / k, j = 1..n_h k
    (Gaussian likelihood, scale from the weighted residuals)."""
    w = 1.0 / np.maximum(e, 1e-9)
    cols = [np.ones_like(t)]
    for j in range(1, n_h * k + 1):
        ph = 2.0 * math.pi * (j * f0 / k) * t
        cols += [np.sin(ph), np.cos(ph)]
    X = np.c_[tuple(cols)] * w[:, None]
    coef, *_ = np.linalg.lstsq(X, m * w, rcond=None)
    rss = float(np.sum((m * w - X @ coef) ** 2))
    n = t.size
    return n * math.log(max(rss, 1e-300) / n) + X.shape[1] * math.log(n)


def amplitude_snr(t: np.ndarray, m: np.ndarray, period: float, f_sig: float, f_lo: float, half_win: float = 1.0) -> float:
    """Breger et al. (1993) S/N: amplitude at ``f_sig`` over the mean amplitude of the
    residual spectrum (after removing the n-harmonic model at ``period``) within
    +-``half_win`` c/d of it - the local noise, red noise included. Signal when >= 4."""
    from astropy.timeseries import LombScargle  # noqa: PLC0415

    n = t.size
    if n < 20 or not (period > 0 and f_sig > 0):
        return float("nan")
    a_sig = math.sqrt(max(4.0 * float(LombScargle(t, m, normalization="psd").power(np.array([f_sig]))[0]) / n, 0.0))
    # residuals of the curve itself: harmonics of 1/period up to 10, number by BIC, so a
    # non-sinusoidal shape (eclipse) is not counted as noise
    best_r, best_b = None, float("inf")
    for nh in range(1, 11):
        if 2 * nh + 1 >= n // 4:
            break
        r_try = m - LombScargle(t, m, nterms=nh).model(t, 1.0 / period)
        b = n * math.log(max(float(np.sum(r_try**2)), 1e-300) / n) + (2 * nh + 1) * math.log(n)
        if b < best_b:
            best_r, best_b = r_try, b
    r = best_r if best_r is not None else m - float(np.median(m))
    # window centred on the signal, cut (not shifted) at the lowest searched frequency
    fz = np.linspace(max(f_lo, f_sig - half_win), f_sig + half_win, 2000)
    a_noise = float(np.mean(np.sqrt(np.maximum(4.0 * LombScargle(t, r, normalization="psd").power(fz) / n, 0.0))))
    return float(a_sig / a_noise) if a_noise > 0 else float("nan")


def aov_multiharmonic(t: np.ndarray, m: np.ndarray, e: np.ndarray, freq: np.ndarray) -> tuple[np.ndarray, int]:
    """Multi-harmonic AoV statistic Theta(f) (Schwarzenberg-Czerny 1996) via the
    n-term least-squares periodogram: Theta = (N-2n-1)/(2n) * p/(1-p)."""
    from astropy.timeseries import LombScargle  # noqa: PLC0415

    ls = LombScargle(t, m, e, nterms=_N_HARM, normalization="standard")
    p = np.clip(ls.power(freq, method="fastchi2"), 0.0, 1.0 - 1e-12)
    n = t.size
    theta = (n - 2 * _N_HARM - 1) / (2.0 * _N_HARM) * p / (1.0 - p)
    return theta, n


def aov_fap(theta: float, n: int, freq: np.ndarray, t: np.ndarray) -> float:
    """FAP of the highest of M independent frequencies, M = frequency range x baseline."""
    from scipy.stats import f as fdist  # noqa: PLC0415

    p1 = float(fdist.sf(theta, 2 * _N_HARM, n - 2 * _N_HARM - 1))
    m_ind = max(1.0, (freq[-1] - freq[0]) * float(np.ptp(t)))
    return float(-np.expm1(m_ind * np.log1p(-min(p1, 1 - 1e-16))))


def pdm_theta(t: np.ndarray, m: np.ndarray, period: float, nbins: int = 10) -> float:
    """Stellingwerf (1978) PDM statistic (binned variance / total variance)."""
    ph = (t / period) % 1.0
    b = np.minimum((ph * nbins).astype(int), nbins - 1)
    s_tot = float(np.var(m, ddof=1))
    num = den = 0.0
    for k in range(nbins):
        x = m[b == k]
        if x.size > 1:
            num += float(np.var(x, ddof=1)) * (x.size - 1)
            den += x.size - 1
    return float((num / den) / s_tot) if den > 0 and s_tot > 0 else float("nan")


def _fold_minima(
    t: np.ndarray, m: np.ndarray, period: float, nbins: int = 40, extreme: str = "min"
) -> tuple[float, float, float, float]:
    """Levels and errors of the two minima (``extreme="max"``: maxima, sign flipped) of a
    curve folded at ``period``; the second is searched half a cycle from the first."""
    ph = (t / period) % 1.0
    b = np.minimum((ph * nbins).astype(int), nbins - 1)
    med = np.full(nbins, np.nan)
    err = np.full(nbins, np.nan)
    for k in range(nbins):
        x = m[b == k]
        if x.size >= 3:
            med[k] = np.median(x)
            err[k] = 1.2533 * np.std(x, ddof=1) / math.sqrt(x.size)
    if not np.isfinite(med).any():
        return (float("nan"),) * 4
    if extreme == "max":
        med = -med
    k1 = int(np.nanargmax(med))
    k2 = (k1 + nbins // 2) % nbins
    win = [(k2 + d) % nbins for d in range(-nbins // 8, nbins // 8 + 1)]
    k2 = max(win, key=lambda k: med[k] if np.isfinite(med[k]) else -np.inf)
    return float(med[k1]), float(err[k1]), float(med[k2]), float(err[k2])


def analyse_periods(t: np.ndarray, m: np.ndarray, e: np.ndarray, cadence_d: float) -> tuple[dict[str, PeriodSolution], bool, dict[str, Any]]:
    """All periodograms on one light curve (t in BJD, m in mag). Returns solutions,
    whether 2P is preferred over the AoV period, and the arrays for plotting."""
    from astropy.timeseries import BoxLeastSquares, LombScargle  # noqa: PLC0415

    ok = np.isfinite(t) & np.isfinite(m) & np.isfinite(e) & (e > 0)
    t, m, e = t[ok], m[ok], e[ok]
    out: dict[str, PeriodSolution] = {}
    plots: dict[str, Any] = {}
    if t.size < 50:
        return out, False, plots
    freq = frequency_grid(t, cadence_d)
    plots["freq"] = freq

    theta, n = aov_multiharmonic(t, m, e, freq)
    # the n-th harmonic must stay below the Nyquist frequency (else the fit degenerates)
    theta = np.where(freq * _N_HARM <= freq[-1], theta, 0.0)
    i = int(np.argmax(theta))
    df = float(freq[1] - freq[0])

    f_lo, f_hi = float(freq[0]), float(freq[-1])
    fa, tha, sfa = _zoom_peak(t, m, e, float(freq[i]), 3.0 * df, f_lo, f_hi)
    p_aov = 1.0 / fa
    out["AoV"] = PeriodSolution(
        "AoV (multi-harmonic, n=3)", p_aov, sfa / (fa * fa), aov_fap(float(tha), n, freq, t), float(tha)
    )
    plots["AoV"] = theta

    ls = LombScargle(t, m, e)
    pw = ls.power(freq, method="fast")
    j = int(np.argmax(pw))
    fl, _thl, sfl = _zoom_peak(t, m, e, float(freq[j]), 3.0 * df, f_lo, f_hi, nterms=1)
    pwl = float(ls.power(np.array([fl]))[0])
    out["LS"] = PeriodSolution(
        "Lomb-Scargle", 1.0 / fl, sfl / (fl * fl),
        float(ls.false_alarm_probability(pwl, method="baluev", minimum_frequency=freq[0], maximum_frequency=freq[-1])),
        float(pwl),
    )
    plots["LS"] = pw

    # PDM at the strongest AoV peaks and their halves / doubles
    cand = set()
    for k in np.argsort(-theta)[:200]:
        cand.add(round(1.0 / freq[k], 8))
    cand |= {round(2 * p_aov, 8), round(0.5 * p_aov, 8)}
    pdm = {p: pdm_theta(t, m, p) for p in cand}
    p_pdm = min(pdm, key=lambda p: pdm[p] if math.isfinite(pdm[p]) else 9)
    out["PDM"] = PeriodSolution("PDM (10 bins)", float(p_pdm), float("nan"), float("nan"), float(pdm[p_pdm]))

    # BLS for eclipses (durations 1-15 % of the period range used)
    try:
        # eclipsing binaries: P >= 0.2 d; astropy needs every duration < the shortest
        # period, so short and long eclipses are searched in two passes
        bls = BoxLeastSquares(t, m, e)
        p_all = 1.0 / freq[:: max(1, freq.size // 20000)]
        best_b = None
        for p_min, dur in ((0.2, np.array([0.02, 0.04, 0.08])), (0.6, np.array([0.16, 0.3]))):
            periods = p_all[p_all >= p_min]
            if periods.size < 10:
                continue
            r = bls.power(periods, dur, objective="snr")
            kb = int(np.argmax(r.power))
            if best_b is None or float(r.power[kb]) > best_b[1]:
                best_b = (float(r.period[kb]), float(r.power[kb]))
        if best_b is not None:
            out["BLS"] = PeriodSolution("BLS (eclipses)", best_b[0], float("nan"), float("nan"), best_b[1], note="power = SNR")
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("[TESS] BLS skipped: %s", exc)

    # Fundamental frequency: the Lomb-Scargle peak (dominant harmonic) - for eclipsing
    # curves the strongest BLS peak when LS is not significant. A multi-harmonic AoV peak
    # alone can sit on a sub-harmonic (3 harmonics at f/3 contain f), so AoV refines and
    # tests the period found this way instead of choosing it.
    p0 = out["LS"].period_d
    if out["LS"].fap >= 0.01 and "BLS" in out and out["BLS"].power >= 7.0:
        p0 = out["BLS"].period_d
    # The LS peak can be a harmonic: a narrow eclipse puts most power at 2, 3, 4 x f.
    # Fourier models with the harmonics of f0/k (k = 1..4) are compared by BIC
    # (Schwarz 1978); the k-model contains the 1-model, so k > 1 wins only when the
    # sub-harmonics really fit the curve (one eclipse per cycle, unequal minima).
    # only multiples that stay inside the searched range (>= 2 cycles per segment)
    bic = {kk: _harmonic_bic(t, m, e, 1.0 / p0, kk) for kk in (1, 2, 3, 4) if kk == 1 or 1.0 / (kk * p0) >= f_lo}
    mult = min(bic, key=bic.get)
    # P or 2P: two minima / maxima of significantly different level at 2P -> 2P
    # bins fine enough for a narrow eclipse, >= ~8 points per bin
    nb = int(min(400, max(40, t.size // 8)))
    d1, s1, d2, s2 = _fold_minima(t, m, 2.0 * p0, nbins=nb)
    x1, u1, x2, u2 = _fold_minima(t, m, 2.0 * p0, nbins=nb, extreme="max")
    diff_min = bool(math.isfinite(d1) and math.isfinite(d2) and abs(d1 - d2) > 3.0 * math.hypot(s1, s2))
    diff_max = bool(math.isfinite(x1) and math.isfinite(x2) and abs(x1 - x2) > 3.0 * math.hypot(u1, u2))
    if mult == 1 and (diff_min or diff_max) and 0.5 / p0 >= f_lo:  # 2P inside the range
        mult = 2
    two_p = mult == 2
    p_sel = mult * p0
    # AoV refinement around the selected frequency, error on the resolved peak
    fr, thr, sfr = _zoom_peak(t, m, e, 1.0 / p_sel, 5.0 * df / mult, f_lo, f_hi)
    p_ref = 1.0 / fr
    if mult == 1:
        why = "equal minima and maxima at 2P: P or 2P (EW/ELL orbital period would be 2P)"
    elif mult == 2:
        why = "unequal minima at 2P" if diff_min else "unequal maxima at 2P" if diff_max else "2P preferred by BIC"
    else:
        why = f"{mult} x the strongest Lomb-Scargle peak (one eclipse per cycle; BIC)"
    best = PeriodSolution(
        "AoV refined at LS" + (f" x{mult}" if mult > 1 else "") + ("/BLS" if p0 != out["LS"].period_d else ""),
        float(p_ref),
        float(sfr / (fr * fr)),
        aov_fap(float(thr), n, freq, t),
        float(thr),
        note=(
            why
            + (f"; minima {d1:.4f}/{d2:.4f}, maxima {-x1:.4f}/{-x2:.4f} mag at 2 x LS" if math.isfinite(d1) and math.isfinite(x1) else "")
        ),
    )
    # amplitude and epoch of minimum from a 3-harmonic fit at the chosen period
    lsb = LombScargle(t, m, e, nterms=_N_HARM)
    tt = np.linspace(t.min(), t.min() + best.period_d, 400)
    model = lsb.model(tt, 1.0 / best.period_d)
    best.amplitude_mag = float(np.ptp(model))
    best.epoch_min_bjd = float(tt[int(np.argmax(model))])
    # a peak on the long-period limit is red noise / trend, not a resolved period
    if 1.0 / (best.period_d / mult) <= f_lo * 1.05 or fr <= f_lo * 1.05:
        best.note = "at the long-period limit of the search (>= half a TESS segment); " + best.note
    out["best"] = best
    plots["t"], plots["m"] = t, m
    plots["mult"] = mult
    return out, two_p, plots


# ---------------------------------------------------------------------------
# localisation
# ---------------------------------------------------------------------------
def localise(
    cube: np.ndarray,
    t: np.ndarray,
    freq: float,
    sources_xy: dict[str, tuple[float, float]],
    sigma: float,
) -> tuple[dict[str, float], np.ndarray]:
    """Relative likelihood that each source produces the signal at ``freq``.

    Per-pixel semi-amplitude at the frequency (sin + cos fit), then for each source
    the PRF placed on it scaled by least squares; chi^2 -> likelihood, normalised.
    """
    nt, ny, nx = cube.shape
    w = 2.0 * math.pi * freq
    X = np.c_[np.ones(nt), np.sin(w * t), np.cos(w * t)]
    seg = _segments(t)
    flat = cube.reshape(nt, -1).astype(float)
    for s in np.unique(seg):
        k = seg == s
        flat[k] -= np.nanmedian(flat[k], axis=0)
    good = np.isfinite(flat).all(axis=1)
    coef, *_ = np.linalg.lstsq(X[good], flat[good], rcond=None)
    cs, cc = coef[1].reshape(ny, nx), coef[2].reshape(ny, nx)
    amp = np.hypot(cs, cc)
    resid = flat[good] - X[good] @ coef
    # error of one sin / cos coefficient: sigma_res sqrt(2 / N)
    sig_c = (np.std(resid, axis=0) * math.sqrt(2.0 / max(good.sum(), 1))).reshape(ny, nx)
    sig_c = np.where(sig_c > 0, sig_c, np.nanmedian(sig_c))
    w2 = 1.0 / sig_c**2
    chi: dict[str, float] = {}
    for sid, (x, y) in sources_xy.items():
        if not (-1 <= x <= nx and -1 <= y <= ny):
            continue
        prf = prf_image((ny, nx), x, y, sigma)
        # the varying star changes every pixel in phase: cs = a_s PRF, cc = a_c PRF
        # (linear, unbiased - |amplitude| has a positive noise bias that favours bright stars)
        nn = max(float(np.sum(prf**2 * w2)), 1e-30)
        a_s = float(np.sum(prf * cs * w2)) / nn
        a_c = float(np.sum(prf * cc * w2)) / nn
        chi[sid] = float(np.sum((cs - a_s * prf) ** 2 * w2) + np.sum((cc - a_c * prf) ** 2 * w2))
    if not chi:
        return {}, amp
    c0 = min(chi.values())
    L = {k: math.exp(-0.5 * (v - c0)) for k, v in chi.items()}
    s = sum(L.values())
    return {k: v / s for k, v in L.items()}, amp


# ---------------------------------------------------------------------------
# one sector from a lightkurve TargetPixelFile
# ---------------------------------------------------------------------------
def sector_light_curve(
    tpf: Any, target_id: str, sources: list[dict[str, Any]]
) -> tuple[TessSector, dict[str, Any]]:
    """Aperture light curve of one TessCut sector. ``sources``: dicts with
    source_id, ra, dec, rp_mag (target included)."""
    sector = int(getattr(tpf, "sector", 0) or 0)
    flux = np.asarray(tpf.flux.value, dtype=float)
    ferr = np.asarray(tpf.flux_err.value, dtype=float)
    t = np.asarray(tpf.time.value, dtype=float) + 2457000.0  # BTJD -> BJD_TDB
    nt, ny, nx = flux.shape
    cad = float(np.nanmedian(np.diff(t))) * 86400.0 if nt > 1 else float("nan")
    med = np.nanmedian(flux, axis=0)
    xy: dict[str, tuple[float, float]] = {}
    rp: dict[str, float] = {}
    for s in sources:
        try:
            px, py = tpf.wcs.all_world2pix(float(s["ra"]), float(s["dec"]), 0)
        except Exception:  # noqa: BLE001
            continue
        xy[str(s["source_id"])] = (float(px), float(py))
        _mg = s.get("rp_mag") if s.get("rp_mag") is not None else s.get("g_mag")
        rp[str(s["source_id"])] = float(_mg) if _mg is not None else float("nan")
    if target_id not in xy:
        return TessSector(sector, cad, 0, float("nan"), float("nan"), float("nan"), 0, float("nan"), error="target not on cutout"), {}
    if not math.isfinite(rp[target_id]):
        return TessSector(sector, cad, 0, float("nan"), float("nan"), float("nan"), 0, float("nan"),
                          error="target magnitude unknown (not in the Gaia DB)"), {}
    # PRF width from the brightest source inside the cutout (data-derived)
    inside = [k for k, (x, y) in xy.items() if 1 < x < nx - 2 and 1 < y < ny - 2 and math.isfinite(rp[k])]
    bright = min(inside, key=lambda k: rp[k]) if inside else target_id
    sigma = measure_prf_sigma(med, *xy[bright])
    if not (math.isfinite(sigma) and 0.3 < sigma < 3.0):
        sigma = 0.9
    fl = {k: 10 ** (-0.4 * (rp[k] - 10.0)) for k in xy if math.isfinite(rp[k])}
    timg = fl[target_id] * prf_image((ny, nx), *xy[target_id], sigma)
    cimg = np.zeros((ny, nx))
    for k, f in fl.items():
        if k != target_id:
            cimg += f * prf_image((ny, nx), *xy[k], sigma)
    tot = timg + cimg
    bkg_px = tot < _BKG_MODEL_FRAC * float(np.max(tot))
    if bkg_px.sum() < 5:
        bkg_px = tot <= np.percentile(tot, 20)
    bkg = np.nanmedian(flux[:, bkg_px], axis=1)
    # scale the model to e-/s using the target's measured flux on the median image
    scale = float(np.nansum((med - np.nanmedian(med[bkg_px])) * timg) / max(np.sum(timg * timg), 1e-30))
    bvar = float(np.nanmedian(np.nanvar(flux[:, bkg_px], axis=0)))
    # effective exposure per cadence: cadence x DEADC (TESS FFI 0.792: 2 s frames, readout
    # and cosmic-ray mitigation; TESS Instrument Handbook)
    try:
        deadc = float(tpf.hdu[1].header.get("DEADC", 0.792))
    except Exception:  # noqa: BLE001
        deadc = 0.792
    t_exp = cad * deadc if math.isfinite(cad) else 1425.6
    mask, contam = choose_aperture(max(scale, 1e-9) * timg, max(scale, 1e-9) * cimg, bvar, t_exp)
    f_ap = np.nansum(flux[:, mask], axis=1) - bkg * mask.sum()
    e_ap = np.sqrt(np.nansum(ferr[:, mask] ** 2, axis=1))
    ok = np.isfinite(f_ap) & (f_ap > 0) & np.isfinite(e_ap)
    t, f_ap, e_ap = t[ok], f_ap[ok], e_ap[ok]
    m = -2.5 * np.log10(f_ap)
    e = 1.0857 * e_ap / f_ap
    seg = _segments(t)
    for s in np.unique(seg):
        k = seg == s
        m[k] -= np.median(m[k])
    sec = TessSector(sector, cad, int(t.size), float(t.min()) if t.size else float("nan"),
                     float(t.max()) if t.size else float("nan"), float(contam), int(mask.sum()), float(sigma))
    aux = {"t": t, "m": m, "e": e, "cube": flux[ok] - bkg[ok, None, None], "mask": mask, "xy": xy, "sigma": sigma, "median": med}
    return sec, aux


# ---------------------------------------------------------------------------
# verdict
# ---------------------------------------------------------------------------
def verdict(res: TessAnalysis, contam_max: float) -> None:
    reasons: list[str] = []
    if res.best is None:
        res.verdict, res.reasons = "NO_DATA", ["no usable TESS data"]
        return
    if not (res.best.fap < 0.01):
        # no significant signal: nothing to localise
        res.verdict, res.reasons = "AMBIGUOUS", [f"no significant period (FAP {res.best.fap:.2g} >= 0.01)"]
        return
    if res.at_long_period_limit:
        res.verdict, res.reasons = "AMBIGUOUS", ["period at the long-period limit of a TESS segment (trend / red noise, not resolved)"]
        return
    if math.isfinite(res.snr_amplitude) and res.snr_amplitude < 4.0:
        # red noise passes a white-noise FAP; the local amplitude S/N does not
        res.verdict, res.reasons = "AMBIGUOUS", [f"amplitude S/N {res.snr_amplitude:.1f} < 4 against the local noise (Breger et al. 1993)"]
        return
    if res.localization:
        top = max(res.localization, key=res.localization.get)
        if top != res.catalog_id:
            res.verdict = "NEIGHBOUR"
            res.reasons = [f"signal most likely from Gaia DR3 {top} (L={res.localization[top]:.2f})"]
            return
    if not (res.target_likelihood >= 0.5):
        reasons.append(f"target likelihood {res.target_likelihood:.2f} < 0.5")
    if not math.isfinite(res.contamination):
        reasons.append("neighbour light unknown (no Gaia neighbours available)")
    elif not (res.contamination < contam_max):
        reasons.append(f"contamination {100 * res.contamination:.0f} % >= {100 * contam_max:.0f} %")
    per = [s.period.period_d for s in res.sectors if s.period is not None and math.isfinite(s.period.period_d)]
    if len(per) < 2:
        reasons.append("period from a single sector")
    else:
        p = res.best.period_d
        agree = 0
        for s in res.sectors:
            if s.period is None or not math.isfinite(s.period.period_d):
                continue
            q = s.period_common_d if math.isfinite(s.period_common_d) else s.period.period_d
            # the sector's own error only (scaled with the multiple): the combined error
            # is inflated by the very disagreement this test has to catch
            sq = s.period.period_err_d * q / s.period.period_d if math.isfinite(s.period.period_err_d) else float("nan")
            tol = 3.0 * sq if math.isfinite(sq) and sq > 0 else 1e-3 * p
            if abs(q - p) <= tol:
                agree += 1
        if agree < 2 or 2 * agree < len(per):
            reasons.append(f"sectors disagree ({agree}/{len(per)} agree within 3 sigma)")
    res.verdict = "RELIABLE" if not reasons else "AMBIGUOUS"
    res.reasons = reasons


def combine_sectors(
    t: np.ndarray, m: np.ndarray, e: np.ndarray, auxs: list[tuple[TessSector, dict[str, Any]]],
    best_all: PeriodSolution | None,
) -> tuple[PeriodSolution | None, int]:
    """Period of all sectors together; returns (solution, period multiplier).

    Within one sector the multiple (P, 2P, 3P ... of the strongest LS peak) and the
    period are unambiguous; across sectors years apart the combined periodogram is a
    comb of aliases (cycle count unknown). So: the multiple by majority of the sectors,
    the sector periods brought to it and averaged (inverse variance), and the combined
    data only refine the period inside +-3 sigma of that mean. When that window holds
    more than one alias tooth the sector-mean error is kept and noted.
    """
    secs = [(s, a) for s, a in auxs if s.period is not None and math.isfinite(s.period.period_d)]
    if not secs:
        return best_all, 1
    # the final sector periods vote: the one most others agree with (within 2 %,
    # tie -> the longer sector) sets the period; the rest are brought to it by the
    # nearest ratio of small integers (1/4 .. 4)
    pf = [s.period.period_d for s, _ in secs]
    support = [sum(abs(q / p - 1.0) < 0.02 for q in pf) for p in pf]
    # tie: the sector whose own period has the smaller relative error
    # tie: the more significant sector (relative period error would favour harmonics)
    i_ref = max(range(len(secs)), key=lambda i: (support[i], secs[i][0].n_points,
                                                 -math.log10(max(secs[i][0].period.fap, 1e-300)) if math.isfinite(secs[i][0].period.fap) else 0.0))
    p_ref0 = pf[i_ref]
    mult = int(secs[i_ref][1].get("mult", 1))
    # only whole multiples (a harmonic of the same signal); 3/2 etc. are different signals
    ratios = sorted({float(k) for k in range(1, 6)} | {1.0 / k for k in range(2, 6)})
    ps, sg = [], []
    n_same = 0
    for s, a in secs:
        r = min(ratios, key=lambda x: abs(x * s.period.period_d / p_ref0 - 1.0))
        n_same += int(abs(r - 1.0) < 1e-9)
        p = r * s.period.period_d
        s.period_common_d = float(p)
        sig = s.period.period_err_d * p / s.period.period_d if math.isfinite(s.period.period_err_d) and s.period.period_err_d > 0 else 1e-3 * p
        ps.append(p)
        sg.append(sig)
    ps_a, sg_a = np.array(ps), np.array(sg)
    ws_a = 1.0 / sg_a**2
    p_mean = float(np.sum(ps_a * ws_a) / np.sum(ws_a))
    sig_w = float(math.sqrt(1.0 / np.sum(ws_a)))
    consistent = bool(np.all(np.abs(ps_a - p_mean) <= 3.0 * np.hypot(sg_a, sig_w)))
    sig_mean = sig_w if consistent or len(ps_a) < 2 else float(max(sig_w, np.std(ps_a, ddof=1) / math.sqrt(len(ps_a))))
    span = float(np.ptp(t)) if t.size else 0.0
    f0 = 1.0 / p_mean
    df_win = min(3.0 * sig_mean / (p_mean * p_mean), 0.5 * f0)  # never down to f <= 0
    n_teeth = 2.0 * df_win * span
    n_grid = int(min(200_000, max(2001, 20.0 * n_teeth)))
    fz = np.linspace(f0 - df_win, f0 + df_win, n_grid)
    thz, n = aov_multiharmonic(t, m, e, fz)
    k = int(np.argmax(thz))
    f_ref, th_ref, sf_ref = _zoom_peak(t, m, e, float(fz[k]), 2.0 * float(fz[1] - fz[0]),
                                       float(fz[0]), float(fz[-1]))
    p_ref = 1.0 / f_ref
    note_alias = ""
    if not consistent:
        err = sig_mean
        note_alias = "; sector periods disagree beyond 3 sigma"
    elif n_teeth > 1.5:
        err = sig_mean
        note_alias = f"; sectors {span:.0f} d apart: cycle count ambiguous within +-3 sigma ({n_teeth:.0f} alias peaks)"
    else:
        err = float(sf_ref / (f_ref * f_ref))
    from scipy.stats import f as fdist  # noqa: PLC0415

    p1 = float(fdist.sf(float(th_ref), 2 * _N_HARM, n - 2 * _N_HARM - 1))
    m_ind = max(1.0, (1.0 / (2 * np.median([s.cadence_s for s, _ in secs]) / 86400.0)) * span)
    fap = float(-np.expm1(m_ind * np.log1p(-min(p1, 1 - 1e-16))))
    amps = [s.period.amplitude_mag for s, _ in secs if math.isfinite(s.period.amplitude_mag)]
    base_note = secs[0][0].period.note if secs[0][0].period is not None else ""
    best = PeriodSolution(
        f"sectors ({len(secs)}) + AoV refinement" + (f" x{mult}" if mult > 1 else ""),
        p_ref, err, fap, float(th_ref),
        amplitude_mag=float(np.median(amps)) if amps else float("nan"),
        note=(f"period agreed by {n_same}/{len(secs)} sectors without rescaling" + note_alias + (f"; {base_note}" if base_note else "")),
    )
    # epoch of minimum from the longest sector folded on the final period
    s0, a0 = max(secs, key=lambda sa: sa[0].n_points)
    try:
        from astropy.timeseries import LombScargle  # noqa: PLC0415

        ls = LombScargle(a0["t"], a0["m"], a0["e"], nterms=_N_HARM)
        tt = np.linspace(a0["t"].min(), a0["t"].min() + p_ref, 400)
        best.epoch_min_bjd = float(tt[int(np.argmax(ls.model(tt, 1.0 / p_ref)))])
    except Exception:  # noqa: BLE001
        pass
    return best, int(mult)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def gaia_sources_for(ra: float, dec: float, cfg: Any, radius_arcsec: float = 210.0) -> list[dict[str, Any]]:
    """Gaia neighbours (+ target) for the TESS cutout from the local DB."""
    from dilution import query_gaia_neighbors  # noqa: PLC0415

    db = str(getattr(cfg, "gaia_db_path", "") or "")
    rows = query_gaia_neighbors(ra, dec, radius_arcsec, db) if db else []
    out = [
        {"source_id": str(r.get("source_id")), "ra": r["ra"], "dec": r["dec"],
         "rp_mag": r.get("rp_mag"), "g_mag": r.get("g_mag")}
        for r in rows
    ]
    return out


def run_tess(
    catalog_id: str,
    ra: float,
    dec: float,
    out_dir: Path,
    *,
    cfg: Any = None,
    sources: list[dict[str, Any]] | None = None,
    vyvar_lc: tuple[np.ndarray, np.ndarray] | None = None,
    progress: Callable[[str, float], None] | None = None,
    tpfs: list[Any] | None = None,
    target_mag: float | None = None,
) -> TessAnalysis:
    """Download (unless ``tpfs`` given) and analyse all TESS sectors of one star.

    ``target_mag``: Gaia G of the target, used when the star is not in the local Gaia DB.
    A failed download never overwrites an earlier result in ``out_dir``."""
    from config import AppConfig  # noqa: PLC0415

    cfg = cfg or AppConfig()
    cid = str(catalog_id).strip()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = TessAnalysis(cid, float(ra), float(dec))

    def _p(msg: str, v: float) -> None:
        if progress is not None:
            try:
                progress(msg, v)
            except Exception:  # noqa: BLE001
                pass

    if tpfs is None:
        try:
            import lightkurve as lk  # noqa: PLC0415

            _p("Searching TESS sectors", 0.05)
            sr = lk.search_tesscut(f"{float(ra)} {float(dec)}")
            if len(sr) == 0:
                res.error = "no TESS sector covers this position"
                verdict(res, TESS_CONTAM_MAX)
                _write(res, out_dir)
                return res
            _p(f"Downloading {len(sr)} sector(s)", 0.1)
            tpfs = list(sr.download_all(cutout_size=15, quality_bitmask="hard"))
        except Exception as exc:  # noqa: BLE001
            res.error = f"TESS download failed: {exc}"
            res.verdict, res.reasons = "NO_DATA", [res.error]
            if not (out_dir / "tess_result.json").is_file():
                _write(res, out_dir)
            return res
    if sources is None:
        sources = gaia_sources_for(ra, dec, cfg)
    has_neighbours = any(str(s["source_id"]) != cid for s in sources)
    if not any(str(s["source_id"]) == cid for s in sources):
        sources = list(sources) + [{"source_id": cid, "ra": ra, "dec": dec, "rp_mag": None,
                                    "g_mag": float(target_mag) if target_mag is not None and math.isfinite(float(target_mag)) else None}]

    all_t, all_m, all_e, auxs = [], [], [], []
    for i, tpf in enumerate(tpfs):
        _p(f"Sector {getattr(tpf, 'sector', '?')}", 0.3 + 0.4 * i / max(1, len(tpfs)))
        try:
            sec, aux = sector_light_curve(tpf, cid, sources)
        except Exception as exc:  # noqa: BLE001
            res.sectors.append(TessSector(int(getattr(tpf, "sector", 0) or 0), float("nan"), 0, float("nan"), float("nan"),
                                          float("nan"), 0, float("nan"), error=str(exc)))
            continue
        if aux:
            sols, tp_sec, pl_sec = analyse_periods(aux["t"], aux["m"], aux["e"], sec.cadence_s / 86400.0)
            sec.period = sols.get("best")
            aux["mult"] = int(pl_sec.get("mult", 2 if tp_sec else 1))
            all_t.append(aux["t"]); all_m.append(aux["m"]); all_e.append(aux["e"]); auxs.append((sec, aux))
        res.sectors.append(sec)
    if not all_t:
        res.error = res.error or "no usable sector"
        verdict(res, TESS_CONTAM_MAX)
        _write(res, out_dir)
        return res
    t = np.concatenate(all_t); m = np.concatenate(all_m); e = np.concatenate(all_e)
    cad = float(np.nanmin([s.cadence_s for s, _ in auxs])) / 86400.0
    _p("Period analysis", 0.75)
    sols, two_p_all, plots = analyse_periods(t, m, e, cad)
    res.periods = {k: v for k, v in sols.items() if k != "best"}
    res.best, mult = combine_sectors(t, m, e, auxs, sols.get("best"))
    res.period_multiplier = int(mult)
    res.two_p_preferred = mult == 2
    w = np.array([max(s.n_points, 1) for s, _ in auxs], float)
    res.contamination = float(np.nansum(w * np.array([s.contamination for s, _ in auxs])) / w.sum())
    if not has_neighbours:
        res.contamination = float("nan")
    if res.best is not None:
        _p("Who is varying", 0.85)
        sec0, aux0 = max(auxs, key=lambda sa: sa[0].n_points)
        # strongest harmonic = the LS frequency
        f_loc = mult / res.best.period_d
        res.localization, amp = localise(aux0["cube"], aux0["t"], f_loc, aux0["xy"], aux0["sigma"])
        res.target_likelihood = float(res.localization.get(cid, float("nan")))
        try:
            f_lo_sec = float(frequency_grid(aux0["t"], sec0.cadence_s / 86400.0)[0])
            res.at_long_period_limit = bool(
                f_loc <= 1.05 * f_lo_sec
                or any(s.period is not None and s.period.note.startswith("at the long-period limit") for s, _ in auxs)
            )
            res.snr_amplitude = amplitude_snr(aux0["t"], aux0["m"], res.best.period_d, f_loc, f_lo_sec)
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("[TESS] S/N failed: %s", exc)
        plots["amp_map"], plots["mask"], plots["xy"] = amp, aux0["mask"], aux0["xy"]
    verdict(res, TESS_CONTAM_MAX)
    try:
        _plots(res, plots, out_dir, vyvar_lc)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[TESS] plots failed: %s", exc)
    np.savetxt(out_dir / "tess_lc.csv", np.c_[t, m, e], delimiter=",", header="bjd_tdb,dmag,err", comments="", fmt="%.6f")
    _write_plot_data(res, plots, out_dir)
    _write(res, out_dir)
    return res


def _write(res: TessAnalysis, out_dir: Path) -> None:
    (out_dir / "tess_result.json").write_text(json.dumps(res.to_json(), indent=2), encoding="utf-8")


def _write_plot_data(res: TessAnalysis, plots: dict[str, Any], out_dir: Path) -> None:
    """VAR-TAB-2026-10: data of the periodogram and the pixel map for the interactive UI plots."""
    if "freq" in plots:
        np.savetxt(out_dir / "periodogram.csv", np.c_[plots["freq"], plots["LS"], plots["AoV"]], delimiter=",",
                   header="frequency_per_d,ls_power,aov_theta", comments="", fmt="%.8g")
    if "amp_map" in plots:
        pix = {
            "amplitude": np.asarray(plots["amp_map"], float).round(6).tolist(),
            "mask": np.asarray(plots["mask"], bool).astype(int).tolist(),
            "stars": {str(k): [float(v[0]), float(v[1])] for k, v in plots["xy"].items()},
            "target_likelihood": res.target_likelihood if math.isfinite(res.target_likelihood) else None,
        }
        (out_dir / "pixel_map.json").write_text(json.dumps(pix), encoding="utf-8")


def _plots(res: TessAnalysis, plots: dict[str, Any], out_dir: Path, vyvar_lc: tuple[np.ndarray, np.ndarray] | None) -> None:
    """PNG figures of the PDF report (periodogram, phase curve, pixel map)."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    if "freq" in plots:
        fig, axs = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
        axs[0].plot(plots["freq"], plots["AoV"], lw=0.7, c="k"); axs[0].set_ylabel("AoV Theta")
        axs[1].plot(plots["freq"], plots["LS"], lw=0.7, c="k"); axs[1].set_ylabel("LS power")
        if res.best is not None:
            for a in axs:
                a.axvline(res.period_multiplier / res.best.period_d, c="#D9480F", ls="--", lw=1)
        axs[1].set_xlabel("frequency [1/d]")
        fig.tight_layout(); p = out_dir / "periodogram.png"; fig.savefig(p, dpi=110); plt.close(fig)
    if res.best is not None and "t" in plots:
        P = res.best.period_d
        fig, ax = plt.subplots(figsize=(9, 4))
        ph = ((plots["t"] - res.best.epoch_min_bjd) / P) % 1.0
        ax.plot(np.r_[ph, ph + 1], np.r_[plots["m"], plots["m"]], ".", ms=1, c="#9AA3AE", label="TESS")
        if vyvar_lc is not None:
            tv, mv = np.asarray(vyvar_lc[0], float), np.asarray(vyvar_lc[1], float)
            mv = mv - np.nanmedian(mv)
            pv = ((tv - res.best.epoch_min_bjd) / P) % 1.0
            ax.plot(np.r_[pv, pv + 1], np.r_[mv, mv], "o", ms=3, c="#D9480F", label="VYVAR")
        ax.invert_yaxis(); ax.set_xlabel(f"phase (P = {P:.6f} d)"); ax.set_ylabel("dmag"); ax.legend(fontsize=8)
        fig.tight_layout(); p = out_dir / "phased.png"; fig.savefig(p, dpi=110); plt.close(fig)
    if "amp_map" in plots:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(plots["amp_map"], origin="lower", cmap="Greys")
        my, mx = np.where(plots["mask"])
        ax.plot(mx, my, "s", mfc="none", mec="#1F5FBF", ms=12)
        for sid, (x, y) in plots["xy"].items():
            c = "#D9480F" if sid == res.catalog_id else "#5A6370"
            ax.plot(x, y, "o", c=c, ms=5 if sid == res.catalog_id else 3)
        ax.set_title(f"amplitude per pixel; target L = {res.target_likelihood:.2f}", fontsize=9)
        fig.tight_layout(); p = out_dir / "pixel_map.png"; fig.savefig(p, dpi=110); plt.close(fig)
