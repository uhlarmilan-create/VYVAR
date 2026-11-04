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

STEP7 C2 (EMVA 1288 R4.0 sec. 6.5-6.9; Astier et al. 2019): per level the temporal dark variance of the dark
pair of the same exposure is subtracted, var(F1 - F2)/2 - var(D1 - D2)/2, and the line is fitted through the
origin (slope 1/g) when every level has its dark pair; the report holds the local slopes (EMVA: within 3 % of the
global slope), the mean-vs-exposure linearity (5-95 % of saturation) and the nearest-neighbour covariance of the
difference image (|rho| above 1 %: the covariances are summed into the variance, Astier et al. 2019).
``PtcResult.to_json`` is stored with the measured EQUIPMENT_CURVE point (PTC_JSON).

``sky_pair_gain`` is the per-night CAL-DIAG diagnostic from consecutive raw lights (sky as the light source):
gain only (the read noise is an extrapolation there), never an authority.
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
#: EMVA 1288 R4.0 eq. 51: local slopes of the photon transfer within this fraction of the global slope.
LOCAL_SLOPE_TOL = 0.03
#: EMVA 1288 linearity range (fraction of saturation) for the mean-vs-exposure check.
LINEARITY_RANGE = (0.05, 0.95)
#: Nearest-neighbour correlation of the difference image above which the covariances are summed into the
#: variance (Astier et al. 2019; lit_cal C2: "< 1 %, else sum covariances"); the effect-size gate of
#: ``_covariance_check``, which also requires 3 standard errors of significance.
COV_RHO_MAX = 0.01


@dataclass
class PairPoint:
    files: tuple[str, str]
    exptime: float
    level_adu: float
    var_half: float  # temporal variance used in the fit (dark pair subtracted, covariances summed when needed)
    used: bool = True
    note: str = ""
    var_half_raw: float | None = None  # var(F1 - F2) / 2 as measured
    dark_var_half: float | None = None  # var(D1 - D2) / 2 of the dark pair of this exposure
    rho_x: float | None = None  # nearest-neighbour correlation of F1 - F2 along x / y
    rho_y: float | None = None
    cov_half: float | None = None  # 2 (cov_x + cov_y) / 2, added when |rho| > COV_RHO_MAX
    n_px: int = 0  # pixels of the difference image (for the significance of rho)


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
    saturation_adu: float | None = None
    through_origin: bool = False
    linearity: dict[str, Any] = field(default_factory=dict)
    covariance: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """The photon-transfer report stored with the measured point (EQUIPMENT_CURVE.PTC_JSON)."""
        import json  # noqa: PLC0415
        from dataclasses import asdict  # noqa: PLC0415

        d = asdict(self)
        d["method"] = "EMVA 1288 R4.0 photon transfer (gain_ptc)"
        return json.dumps(d, default=lambda o: None if isinstance(o, float) and not math.isfinite(o) else str(o))


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


def _neighbour_cov(d: np.ndarray) -> tuple[float, float, float]:
    """(variance, covariance at lag x = 1, covariance at lag y = 1) of a difference image (finite pixels)."""
    d = np.asarray(d, dtype=np.float64)
    m = np.isfinite(d)
    mu = float(d[m].mean()) if m.any() else 0.0
    x = np.where(m, d - mu, 0.0)
    var = float((x[m] ** 2).mean()) if m.any() else float("nan")
    mx = m[:, 1:] & m[:, :-1]
    my = m[1:, :] & m[:-1, :]
    cx = float((x[:, 1:] * x[:, :-1])[mx].mean()) if mx.any() else float("nan")
    cy = float((x[1:, :] * x[:-1, :])[my].mean()) if my.any() else float("nan")
    return var, cx, cy


def read_noise_adu(d1: np.ndarray, d2: np.ndarray) -> float:
    """Read noise in ADU from two darks of the same (short) exposure."""
    return _read_noise_adu_roi(_roi(np.asarray(d1, dtype=np.float64)), _roi(np.asarray(d2, dtype=np.float64)))


def _read_noise_adu_roi(a: np.ndarray, b: np.ndarray) -> float:
    """:func:`read_noise_adu` on darks already cut to the central region (float64)."""
    d = (a - b)[np.isfinite(a - b)]
    return float(np.std(d)) / math.sqrt(2.0) if d.size else float("nan")


def fit_ptc(
    points: list[PairPoint], rn_adu: float | None, *, through_origin: bool = False
) -> tuple[float, float, float, float]:
    """Gain, its 1-sigma error, intercept and R^2 from the used pair points.

    Two or more levels: least-squares line var = S / g + c. ``through_origin`` (every level dark-pair
    subtracted, EMVA 1288): var = S / g. One level: g = S / (var - RN_adu^2) (needs the dark read noise; with
    the dark variance already subtracted g = S / var); its error is not defined (NaN)."""
    use = [p for p in points if p.used]
    x = np.array([p.level_adu for p in use], dtype=np.float64)
    y = np.array([p.var_half for p in use], dtype=np.float64)
    if through_origin and len(use) >= 2 and np.all(x > 0):
        slope = float(np.sum(x * y) / np.sum(x * x))
        resid = y - slope * x
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        slope_err = math.sqrt(ss_res / (len(use) - 1) / float(np.sum(x * x))) if len(use) > 1 else float("nan")
        if slope <= 0:
            return float("nan"), float("nan"), 0.0, r2
        return 1.0 / slope, slope_err / (slope * slope), 0.0, r2
    if through_origin and len(use) == 1 and x[0] > 0 and y[0] > 0:
        return float(x[0]) / float(y[0]), float("nan"), 0.0, float("nan")
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


def ptc_frames_from_folder(folder: Path, *, db: Any = None) -> tuple[list[Path], list[Path]]:
    """(raw flats, raw darks) of a folder for :func:`measure_from_files` - the one selection used by the
    Calibration Library page and dev/tools/ptc_from_folder.py. Subfolders included; frames are recognised by
    IMAGETYP (FRAME / IMTYPE); masters (MD_ / MF_ / MASTER names, NCOMBINE, MASTER history) are left out, and a
    flat typed as a dark-flat counts as neither. ``db`` only speeds up repeated scans (FITS_HEADER_CACHE)."""
    from calib_plan import is_dark_typed  # noqa: PLC0415
    from importer import _collect_fits_by_type, _looks_like_master  # noqa: PLC0415

    p = Path(folder)
    if not p.is_dir():
        raise FileNotFoundError(f"PTC folder not found: {p}")
    by = _collect_fits_by_type(p, db=db)
    flats = [f for f in by.get("flat", []) if not _looks_like_master(f) and not is_dark_typed(f)]
    darks = [f for f in by.get("dark", []) if not _looks_like_master(f)]
    return flats, darks


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
    res.saturation_adu = float(saturation_adu)
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
    all_dark_pairs = True
    for e, fl in sorted(by_exp.items()):
        dks = dark_by_exp.get(e)
        dvar_h: float | None = None
        if dks:
            dark = np.mean(np.stack(dks), axis=0) if len(dks) > 1 else dks[0]
            if len(dks) >= 2:  # C2: temporal dark variance of this exposure (EMVA eq. 16-18)
                dvar_h = _read_noise_adu_roi(dks[0], dks[1]) ** 2
        elif dark_by_exp:
            near = min(dark_by_exp, key=lambda k: abs(k - e))
            dark = np.mean(np.stack(dark_by_exp[near]), axis=0)
            res.messages.append(f"Flats {e:g} s: no dark of that exposure, dark of {near:g} s used.")
        else:
            dark = 0.0
        if dvar_h is None:
            all_dark_pairs = False
        for (p1, a1), (p2, a2) in zip(fl[0::2], fl[1::2]):
            lev, var_h = _pair_variance_roi(a1, a2, dark)
            m1, m2 = float(np.median(a1)), float(np.median(a2))
            v_d, c_x, c_y = _neighbour_cov(a1 - a2)
            rho_x, rho_y = (c_x / v_d, c_y / v_d) if v_d > 0 else (float("nan"), float("nan"))
            pt = PairPoint((p1.name, p2.name), e, lev, var_h - (dvar_h or 0.0), var_half_raw=var_h,
                           dark_var_half=dvar_h, rho_x=rho_x, rho_y=rho_y, n_px=int(np.isfinite(a1 - a2).sum()))
            if abs(m1 - m2) > PAIR_LEVEL_TOL * max(abs(m1), abs(m2), 1.0):
                pt.used, pt.note = False, "levels of the two flats differ (light changed)"
            elif max(m1, m2) > MAX_LEVEL_FRACTION * float(saturation_adu):
                pt.used, pt.note = False, f"above {MAX_LEVEL_FRACTION:.0%} of saturation"
            elif lev <= 0:
                pt.used, pt.note = False, "level not above the dark"
            res.points.append(pt)
        if len(fl) % 2:
            res.messages.append(f"Flats {e:g} s: odd number of frames, the last one is not paired.")
    _covariance_check(res)
    res.through_origin = bool(all_dark_pairs and dark_by_exp)
    if not res.through_origin:
        res.messages.append("Not every flat exposure has a dark pair: the dark variance stays in the fit intercept "
                            "(take two darks per flat exposure).")
    g, g_err, _, r2 = fit_ptc(res.points, rn_adu, through_origin=res.through_origin)
    res.n_levels = sum(p.used for p in res.points)
    if math.isfinite(g) and g > 0:
        _linearity_report(res, 1.0 / g)
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


def _covariance_check(res: PtcResult) -> None:
    """C2: nearest-neighbour correlation of the flat difference images, averaged over the used levels; when it
    is both physically relevant and statistically significant, the covariances (both lags, both directions) are
    added to the variance of every level (Astier et al. 2019).

    Two gates, both required (Milan 2026-11-04 kept the 3 sigma guard, V10-CAL-PTC-COV):

    - Effect size, ``|rho| > COV_RHO_MAX`` (1 %): inter-pixel capacitance and brighter-fatter covariances move
      shot-noise variance into the neighbours, so var(F1 - F2) alone underestimates it and the PTC gain comes
      out high by about the summed neighbour correlation, 2 (rho_x + rho_y) for the four nearest neighbours
      (Astier et al. 2019, A&A 629, A36; the 1 % threshold is lit_cal C2: "< 1 %, else sum covariances").
    - Significance, ``|rho| > 3 x SE``: per level the estimator is the lag-1 sample autocorrelation of the
      difference image, r = mean(x_i x_i+1) / mean(x_i^2) over n pixel pairs (n = rows x (cols - 1) for x; the
      pixel count n_px differs from it by 1 / cols, negligible). Under the null of uncorrelated noise its
      standard error is 1 / sqrt(n) (Bartlett 1946 for lag-k autocorrelations of white noise; the same 1 / sqrt(n)
      as Fisher's large-n result for a sample correlation coefficient). The mean of L independent levels has
      SE = sqrt(sum_l 1 / n_l) / L, which is 1 / sqrt(sum_l n_l) when every level has the same pixel count (the
      usual case: one ROI). 3 SE is a two-sided false trigger rate of 0.27 % per statistic, about 0.54 % for the
      larger of rho_x and rho_y. It only matters for small frames: with >= 9e4 pixels in total 3 SE < 1 % and the
      effect-size gate decides alone.

    Averaging rho over the levels is right for a detector property: the IPC correlation does not depend on the
    level. A brighter-fatter correlation grows with the level; the mean is then the ladder-average correlation
    applied to every level (first order; Astier's full per-level model is not fitted)."""
    use = [p for p in res.points if p.used and p.rho_x is not None and math.isfinite(p.rho_x) and p.n_px > 0]
    if not use:
        res.covariance = {"rho_x": None, "rho_y": None, "limit": COV_RHO_MAX, "summed": False}
        return
    rx = float(np.mean([p.rho_x for p in use]))
    ry = float(np.mean([p.rho_y for p in use]))
    se = math.sqrt(sum(1.0 / float(p.n_px) for p in use)) / len(use)  # SE of the unweighted mean (see above)
    rmax = max(abs(rx), abs(ry))
    res.covariance = {"rho_x": rx, "rho_y": ry, "max_abs_rho": rmax, "rho_stderr": se, "limit": COV_RHO_MAX,
                      "summed": False}
    if rmax > COV_RHO_MAX and rmax > 3.0 * se:
        for p in res.points:
            v_raw = float(p.var_half_raw if p.var_half_raw is not None else p.var_half)
            p.cov_half = 2.0 * v_raw * (rx + ry)
            p.var_half = p.var_half + p.cov_half
        res.covariance["summed"] = True
        res.messages.append(f"Neighbour correlation of the flat differences {rmax:.1%} (> {COV_RHO_MAX:.0%}): "
                            "covariances summed into the variance (inter-pixel capacitance / brighter-fatter).")


def _linearity_report(res: PtcResult, slope: float) -> None:
    """C2: EMVA 1288 local slopes of the used levels against the global slope, and the mean level vs exposure
    time (LINEARITY_RANGE of saturation): maximum relative deviation from the straight line."""
    use = sorted((p for p in res.points if p.used), key=lambda p: p.level_adu)
    loc: list[dict[str, float]] = []
    for a, b in zip(use, use[1:]):
        if b.level_adu > a.level_adu:
            s_loc = (b.var_half - a.var_half) / (b.level_adu - a.level_adu)
            loc.append({"level_adu": 0.5 * (a.level_adu + b.level_adu), "local_slope": s_loc,
                        "deviation": s_loc / slope - 1.0})
    max_dev = max((abs(d["deviation"]) for d in loc), default=float("nan"))
    res.linearity = {"global_slope": slope, "local_slopes": loc, "max_local_slope_dev": max_dev,
                     "local_slope_tol": LOCAL_SLOPE_TOL}
    if loc and max_dev > LOCAL_SLOPE_TOL:
        res.messages.append(f"Local photon-transfer slopes deviate up to {max_dev:.1%} from the global slope "
                            f"(EMVA 1288: within {LOCAL_SLOPE_TOL:.0%}) - non-linearity or unsteady light.")
    sat = res.saturation_adu
    lo, hi = LINEARITY_RANGE
    pts = [p for p in res.points if p.exptime > 0 and sat and lo * sat <= p.level_adu <= hi * sat]
    if len({p.exptime for p in pts}) >= 3:
        t = np.array([p.exptime for p in pts], dtype=np.float64)
        mu = np.array([p.level_adu for p in pts], dtype=np.float64)
        a, b = np.polyfit(t, mu, 1)
        dev = (mu - (a * t + b)) / mu
        res.linearity.update({"mean_vs_time_rate_adu_s": float(a), "mean_vs_time_offset_adu": float(b),
                              "linearity_error_max": float(np.max(np.abs(dev))), "n_linearity_levels": int(len(pts))})
        res.messages.append(f"Linearity (mean level vs exposure, {lo:.0%}-{hi:.0%} of saturation): "
                            f"max deviation {float(np.max(np.abs(dev))):.2%} over {len(pts)} pairs.")


#: FITS integer container of the frames (bits).
CONTAINER_BITS = 16


def container_scale(adc_bits: int | None) -> float:
    """Factor between native ADU and the values in the 16-bit files: an N-bit camera writes ADU x 2^(16-N)
    (1.0 when the bit depth is unknown or outside 1..16). G14-18: the one definition."""
    if adc_bits is None or not (1 <= int(adc_bits) <= CONTAINER_BITS):
        return 1.0
    return float(2 ** (CONTAINER_BITS - int(adc_bits)))


class ContainerScaleError(ValueError):
    """G06-34: the raw data's LSB granularity contradicts the camera's ADC_BITS (container scale)."""


def lsb_granularity(data: Any, saturation_adu: float | None = None) -> int | None:
    """G06-34: the largest power of 2 dividing every finite, unsaturated raw pixel value (exact test, no
    threshold): 4 for a 14-bit camera writing 16-bit files, 1 for 16-bit data or digitally averaged bins.

    Pixels at or above ``saturation_adu`` are left out (else the 16-bit container ceiling 2^16 - 1).
    None when the values are not integers (not raw data) or no pixel is left / all are zero."""
    a = np.asarray(data).ravel()
    if a.dtype.kind == "f":
        a = a[np.isfinite(a)]
    ceiling = float(saturation_adu) if saturation_adu is not None and math.isfinite(float(saturation_adu)) \
        else float(2 ** CONTAINER_BITS - 1)
    a = a[np.abs(a) < ceiling]
    if a.size == 0:
        return None
    if a.dtype.kind == "f" and not np.array_equal(a, np.rint(a)):
        return None
    o = int(np.bitwise_or.reduce(np.abs(a.astype(np.int64 if a.dtype.itemsize > 4 else np.int32))))
    return (o & -o) if o else None


def container_scale_check(granularity: int | None, adc_bits: int | None, binning: int) -> tuple[str, str]:
    """G06-34: (status, message) of the measured LSB granularity against ``container_scale(adc_bits)``.

    ok: granularity == scale. contradiction (stop): granularity > scale, or bin 1 and granularity != scale.
    ambiguous (WARNING): bin > 1 and granularity < scale - in-camera average binning erases the step.
    unmeasured (WARNING): non-integer data; unknown_adc (WARNING): ADC_BITS not known, scale 1 assumed."""
    if granularity is None:
        return "unmeasured", "raw values are not integers - the container scale (ADC_BITS) cannot be checked"
    if adc_bits is None:
        if granularity > 1:
            return "unknown_adc", (f"raw data move in steps of {granularity} ADU but the camera's ADC_BITS is unknown "
                                   "- the gain is used as is (container scale 1); set ADC_BITS of the camera")
        return "ok", ""
    scale = int(container_scale(adc_bits))
    if granularity == scale:
        return "ok", ""
    if granularity > scale or int(binning) <= 1:
        return "contradiction", (f"raw data granularity {granularity} ADU, ADC_BITS {int(adc_bits)} implies "
                                 f"{scale} ADU (bin {int(binning)}) - correct ADC_BITS of the camera")
    return "ambiguous", (f"raw data granularity {granularity} ADU at bin {int(binning)} vs {scale} ADU for ADC_BITS "
                         f"{int(adc_bits)}: in-camera average binning hides the step - container scale not verified")


def file_gain(gain_native: float, adc_bits: int | None) -> float:
    """Native ADC gain (e-/ADU) -> gain of the 16-bit file values (e-/file ADU)."""
    return float(gain_native) / container_scale(adc_bits)


def native_gain(gain_files: float, adc_bits: int | None) -> float:
    """Gain of the files -> native ADC gain (files of an N-bit camera hold values x 2^(16-N))."""
    return float(gain_files) * container_scale(adc_bits)


# --------------------------------------------------------------------------- #
# C2: per-night sky-pair gain (CAL-DIAG diagnostic, never an authority)
# --------------------------------------------------------------------------- #
#: Tile edge [px] of the sky-pair estimate: 4096 pixels per tile give the variance to ~2 % (sampling only).
SKY_PAIR_TILE_PX = 64
#: Sky-pair method band (lit_cal C2: the treatments of the 527 audit span about +-7 % around the best value);
#: a ratio to the profile gain outside 1 +- this band is a WARNING.
SKY_PAIR_BAND = 0.07
#: Bootstrap resamples over the pairs for the gain sigma (seeded: same input, same numbers).
SKY_PAIR_BOOTSTRAP = 300


def _sky_pair_tiles(a: np.ndarray, b: np.ndarray, dark_level: float, tile: int) -> tuple[np.ndarray, np.ndarray]:
    """Per tile: (median sky level above the dark, var(A - B) / 2) of two consecutive raw lights. The plain
    variance (the MAD of 14-bit data in 4-ADU steps is too coarse); tiles holding star residuals are outliers
    that the decile medians of :func:`_decile_slope` leave out - no pixel is clipped."""
    h, w = (a.shape[0] // tile) * tile, (a.shape[1] // tile) * tile
    if h == 0 or w == 0:
        return np.array([]), np.array([])

    def _tiles(x: np.ndarray) -> np.ndarray:
        return x[:h, :w].reshape(h // tile, tile, w // tile, tile).swapaxes(1, 2).reshape(-1, tile * tile)

    ta, td = _tiles((a + b) / 2.0), _tiles(a - b)
    lev = np.nanmedian(ta, axis=1) - float(dark_level)
    return lev, np.nanvar(td, axis=1) / 2.0


def _decile_slope(lev: np.ndarray, var: np.ndarray) -> float:
    """Slope of var vs level over the medians of the ten level deciles (robust to star tiles)."""
    q = np.percentile(lev, np.linspace(0, 100, 11))
    xs, ys = [], []
    for lo, hi in zip(q[:-1], q[1:]):
        s = (lev >= lo) & (lev <= hi)
        if s.any():
            xs.append(float(np.median(lev[s])))
            ys.append(float(np.median(var[s])))
    if len(xs) < 2 or np.ptp(xs) <= 0:
        return float("nan")
    return float(np.polyfit(np.array(xs), np.array(ys), 1)[0])


def sky_pair_gain(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    *,
    dark_level: float,
    profile_gain_file: float | None,
    profile_read_noise_e: float | None,
    seed: int = 0,
    tile: int = SKY_PAIR_TILE_PX,
) -> dict[str, Any]:
    """C2: gain of the file values from consecutive raw lights of one night (photon transfer with the sky as the
    light source: var(A - B)/2 vs sky level over tiles; slope 1/g, independent of the read noise and the
    pedestal). Diagnostic only: the result is the ratio to the profile gain with a bootstrap sigma over the
    pairs; ``status`` ok / warning (ratio outside 1 +- SKY_PAIR_BAND) / skipped.

    Skipped when the sky does not dominate the read noise in the fit: the shot-noise variation over the tiles'
    sky range (p5-p95), dS / g, must exceed the read-noise variance (RN / g)^2 of the profile (else the slope is
    an extrapolation; Brno 534: 28 ADU sky at 12.5 e-/ADU), or when there are fewer than two pairs."""
    out: dict[str, Any] = {"method": "sky-pair photon transfer (diagnostic)", "n_pairs": len(pairs), "status": "skipped"}
    if len(pairs) < 2:
        out["reason"] = "fewer than two consecutive light pairs"
        return out
    per_pair = [_sky_pair_tiles(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64), dark_level, tile)
                for a, b in pairs]
    lev = np.concatenate([p[0] for p in per_pair])
    var = np.concatenate([p[1] for p in per_pair])
    ok = np.isfinite(lev) & np.isfinite(var) & (lev > 0)
    if int(ok.sum()) < 20:
        out["reason"] = "too few sky tiles above the dark level"
        return out
    lo, hi = np.percentile(lev[ok], [5, 95])
    out.update({"n_tiles": int(ok.sum()), "sky_p5_adu": float(lo), "sky_p95_adu": float(hi)})
    if profile_gain_file is None or profile_read_noise_e is None or not (profile_gain_file > 0):
        out["reason"] = "profile gain / read noise unknown"
        return out
    g_p = float(profile_gain_file)
    rn_var = (float(profile_read_noise_e) / g_p) ** 2
    shot_span = float(hi - lo) / g_p
    out.update({"shot_variance_span_adu2": shot_span, "read_noise_variance_adu2": rn_var})
    if shot_span <= rn_var:
        out["reason"] = (f"sky variance does not dominate the read noise (shot-noise span {shot_span:.3g} ADU^2 <= "
                         f"RN^2 {rn_var:.3g} ADU^2)")
        return out
    slope = _decile_slope(lev[ok], var[ok])
    if not (math.isfinite(slope) and slope > 0):
        out["reason"] = "no positive slope"
        return out
    g = 1.0 / slope
    rng = np.random.default_rng(int(seed))
    sizes = [len(p[0]) for p in per_pair]
    starts = np.cumsum([0] + sizes[:-1])
    bs: list[float] = []
    for _ in range(SKY_PAIR_BOOTSTRAP):
        pick = rng.integers(0, len(pairs), len(pairs))
        idx = np.concatenate([np.arange(starts[k], starts[k] + sizes[k]) for k in pick])
        m = ok[idx]
        sl = _decile_slope(lev[idx][m], var[idx][m])
        if math.isfinite(sl) and sl > 0:
            bs.append(1.0 / sl)
    sig = float(np.std(bs)) if len(bs) > 1 else float("nan")
    ratio = g / g_p
    out.update({"gain_file": g, "gain_file_sigma": sig, "profile_gain_file": g_p, "ratio_to_profile": ratio,
                "band": SKY_PAIR_BAND,
                "status": "ok" if abs(ratio - 1.0) <= SKY_PAIR_BAND else "warning"})
    out.pop("reason", None)
    return out
