"""APERTURE-DYNAMIC-02: per-target aperture via Howell S/N (default).

Physics (D5-1): target and its comps share the same enclosed-energy fraction
(same f = r / FWHM) for one differential measurement.

Selection (APERTURE-DYNAMIC-02): f* = argmax of predicted S/N(f) from the
night-median growth curve F(f) and Howell (1989) variance
(``photometry_phase2a._howell_variance_adu2``). Flat top (within 1% of max)
-> largest f. Abbe p2p remains a reported diagnostic only.

Production radius: r = f* x FWHM_frame (FWHM-AUTH-01); night median only as
fallback when a frame has no QC FWHM.

Default production mode is ``per_target``. Fixed one-r for the draft remains
available as ``f_fixed_night``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence
import json
import logging
import math
import time

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

# APERTURE-DYNAMIC-02: fine sampling (step 0.05 FWHM), not a discrete threshold.
DEFAULT_APERTURE_F_GRID: tuple[float, ...] = tuple(
    round(0.4 + 0.05 * i, 2) for i in range(int(round((3.0 - 0.4) / 0.05)) + 1)
)

# Documented fallback when S/N cannot be evaluated (no finite fluxes / sky).
# Midpoint of the Howell/Naylor sky-limited optimum band ~0.6-0.75 FWHM.
# Replaces the undocumented midpoint-of-grid fallback (1.35 on the old grid)
# that produced the 521 spike of 32 stars at exactly 1.35.
FALLBACK_F_STAR: float = 0.70

# Flat-top tolerance: choose largest f within this fraction of max S/N.
SNR_FLAT_FRAC: float = 0.01

MODE_PER_TARGET = "per_target"


def normalize_f_grid(raw: Any) -> list[float]:
    """Parse ``aperture_f_grid`` from config; fall back to DEFAULT_APERTURE_F_GRID."""
    out: list[float] = []
    if isinstance(raw, (list, tuple)):
        for x in raw:
            try:
                v = float(x)
            except (TypeError, ValueError):
                continue
            if math.isfinite(v) and v > 0:
                out.append(float(v))
    if len(out) < 2:
        return list(DEFAULT_APERTURE_F_GRID)
    return sorted(set(out))


def abbe_p2p_scatter(mag: np.ndarray) -> float:
    """Abbe / von Neumann point-to-point scatter: std(diff(mag)) / sqrt(2)."""
    m = np.asarray(mag, dtype=float)
    finite = m[np.isfinite(m)]
    if finite.size < 3:
        return float("nan")
    d = np.diff(finite)
    if d.size < 2:
        return float("nan")
    return float(np.std(d) / math.sqrt(2.0))


def flux_to_mag(flux: np.ndarray | float) -> np.ndarray:
    f = np.asarray(flux, dtype=float)
    out = np.full(f.shape, float("nan"), dtype=float)
    ok = np.isfinite(f) & (f > 0)
    out[ok] = -2.5 * np.log10(f[ok])
    return out


def equal_weight_ensemble_delta(
    target_mag: np.ndarray,
    comp_mags: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Differential LC: target - equal-weight flux-sum ensemble of comps."""
    t = np.asarray(target_mag, dtype=float)
    n = len(t)
    if n == 0:
        return t.copy()
    series = [np.asarray(v, dtype=float) for v in comp_mags.values() if len(v) == n]
    if not series:
        return np.full(n, float("nan"), dtype=float)
    stack = np.vstack(series)
    delta = np.full(n, float("nan"), dtype=float)
    for i in range(n):
        if not math.isfinite(t[i]):
            continue
        cols = stack[:, i]
        ok = np.isfinite(cols)
        if int(ok.sum()) < 1:
            continue
        flux_sum = float(np.sum(np.power(10.0, -0.4 * cols[ok])))
        if not (math.isfinite(flux_sum) and flux_sum > 0):
            continue
        ens = -2.5 * math.log10(flux_sum)
        delta[i] = t[i] - ens
    return delta


def pick_f_star(p2p_by_f: Mapping[float, float]) -> float | None:
    """argmin p2p; ties / flat minimum -> larger f. None if all non-finite.

    Kept for diagnostics / tests; production selection uses ``pick_f_star_snr``.
    """
    best_f: float | None = None
    best_p2p = float("inf")
    for f in sorted(float(k) for k in p2p_by_f.keys()):
        p = float(p2p_by_f[f])
        if not math.isfinite(p):
            continue
        if p < best_p2p - 1e-12:
            best_p2p = p
            best_f = f
        elif abs(p - best_p2p) <= 1e-12:
            if best_f is None or f > best_f:
                best_f = f
    return best_f


def pick_f_star_snr(
    snr_by_f: Mapping[float, float],
    *,
    flat_frac: float = SNR_FLAT_FRAC,
) -> float | None:
    """argmax S/N; among f within flat_frac of the max, pick the largest f."""
    finite: list[tuple[float, float]] = []
    for f, s in snr_by_f.items():
        ff = float(f)
        ss = float(s)
        if math.isfinite(ff) and math.isfinite(ss) and ss > 0:
            finite.append((ff, ss))
    if not finite:
        return None
    max_snr = max(s for _, s in finite)
    thresh = float(max_snr) * (1.0 - float(flat_frac))
    candidates = [f for f, s in finite if s >= thresh - 1e-15]
    return float(max(candidates))


def f_star_is_grid_edge(f_star: float, f_grid: Sequence[float], *, tol: float = 1e-9) -> bool:
    """True if f* equals the minimum or maximum of the sampling grid."""
    vals = sorted(float(x) for x in f_grid if math.isfinite(float(x)) and float(x) > 0)
    if len(vals) < 2 or not math.isfinite(float(f_star)):
        return False
    fs = float(f_star)
    return abs(fs - vals[0]) <= tol or abs(fs - vals[-1]) <= tol


def howell_variance_adu2(
    flux: float,
    sky_pp: float,
    area: float,
    *,
    gain: float = 1.0,
    read_noise: float = 10.0,
) -> float:
    """Total variance [ADU^2] - mirror of ``photometry_phase2a._howell_variance_adu2``.

    Cite: ``src_py/photometry_phase2a.py`` lines ~378-401 (Howell 1989 eq. 2 form).
    Terms: source Poisson ``flux/g``, sky Poisson ``sky_pp/g * area``,
    read noise ``(RN/g)^2 * area``. Kept local to avoid circular import with
    photometry_phase2a.
    """
    if not math.isfinite(flux) or flux <= 0:
        return float("nan")
    if not math.isfinite(sky_pp) or sky_pp < 0:
        sky_pp = 0.0
    if not math.isfinite(area) or area <= 0:
        return float("nan")
    g = float(gain) if math.isfinite(gain) and gain > 0 else 1.0
    rn = float(read_noise) if math.isfinite(read_noise) and read_noise >= 0 else 10.0
    return flux / g + max(0.0, sky_pp) / g * area + (rn / g) ** 2 * area


def howell_snr(
    flux: float,
    sky_pp: float,
    area: float,
    *,
    gain: float = 1.0,
    read_noise: float = 10.0,
) -> float:
    """Predicted S/N = F / sqrt(var) using VYVAR Howell variance terms."""
    f = float(flux)
    if not math.isfinite(f) or f <= 0:
        return float("nan")
    var = howell_variance_adu2(
        f, float(sky_pp), float(area), gain=float(gain), read_noise=float(read_noise)
    )
    if not math.isfinite(var) or var <= 0:
        return float("nan")
    return float(f / math.sqrt(var))


def _aperture_flux_uniform(
    image: np.ndarray,
    pos: np.ndarray,
    r_ap: float,
    r_in: float,
    r_out: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Sky-subtracted exact aperture sums + annulus sky_pp at one shared radius."""
    from photutils.aperture import CircularAnnulus, CircularAperture
    from photutils.aperture import aperture_photometry as _aphot
    from sky_estimation import sky_median_mask  # noqa: PLC0415

    pos = np.asarray(pos, dtype=np.float64)
    n = int(pos.shape[0])
    flux_arr = np.full(n, np.nan, dtype=np.float64)
    sky_arr = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return flux_arr, sky_arr
    r0 = float(r_ap)
    rin = float(r_in)
    rout = float(r_out)
    if not (math.isfinite(r0) and r0 > 0 and math.isfinite(rin) and rin > 0 and rout > rin):
        return flux_arr, sky_arr
    try:
        ap = CircularAperture(pos, r=r0)
        phot = _aphot(image, ap, method="exact")
        sums = np.asarray(phot["aperture_sum"], dtype=np.float64)
        area = float(ap.area)
        an = CircularAnnulus(pos, r_in=rin, r_out=rout)
        masks = an.to_mask(method="center")
        if not isinstance(masks, (list, tuple)):
            masks = [masks]
        for i, m in enumerate(masks):
            try:
                ann_img = m.to_image(image.shape)
                sky_arr[i] = float(sky_median_mask(image, ann_img))
            except Exception:  # noqa: BLE001
                sky_arr[i] = float("nan")
        flux_arr = sums - sky_arr * area
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("[APERTURE-DYNAMIC] uniform aperture failed: %s", exc)
    return flux_arr, sky_arr


def measure_flux_on_f_grid(
    image: np.ndarray,
    xy: np.ndarray,
    *,
    f_grid: Sequence[float],
    fwhm_px: float,
    annulus_inner_fwhm: float,
    annulus_outer_fwhm: float,
) -> tuple[dict[float, np.ndarray], dict[float, np.ndarray]]:
    """Measure sky-subtracted aperture flux and sky_pp for every star at each f."""
    from aperture_policy import resolve_aperture_geometry  # noqa: PLC0415

    pos = np.asarray(xy, dtype=float)
    n = int(pos.shape[0]) if pos.ndim == 2 else 0
    out_f: dict[float, np.ndarray] = {}
    out_s: dict[float, np.ndarray] = {}
    if n == 0:
        for f in f_grid:
            out_f[float(f)] = np.array([], dtype=float)
            out_s[float(f)] = np.array([], dtype=float)
        return out_f, out_s
    img = np.asarray(image, dtype=float)
    for f in f_grid:
        ff = float(f)
        r_ap, r_in, r_out = resolve_aperture_geometry(
            f=ff,
            fwhm_px=float(fwhm_px),
            annulus_inner_fwhm=float(annulus_inner_fwhm),
            annulus_outer_fwhm=float(annulus_outer_fwhm),
        )
        flux_arr, sky_arr = _aperture_flux_uniform(img, pos, r_ap, r_in, r_out)
        out_f[ff] = flux_arr
        out_s[ff] = sky_arr
    return out_f, out_s


@dataclass
class PerTargetChoice:
    catalog_id: str
    f_star: float
    r_ap_px: float
    p2p_by_f: dict[str, float] = field(default_factory=dict)
    snr_by_f: dict[str, float] = field(default_factory=dict)
    n_comps: int = 0
    n_frames: int = 0
    f_edge: bool = False
    reason: str = ""
    snr_max: float = float("nan")
    growth_F_by_f: dict[str, float] = field(default_factory=dict)
    sky_pp_by_f: dict[str, float] = field(default_factory=dict)


@dataclass
class ApertureGridNight:
    """In-memory night grid: frame_key -> catalog_id -> f -> flux / sky_pp."""

    f_grid: list[float]
    fwhm_night_px: float
    # frame_stem -> {cid: {f: flux}}
    frames: dict[str, dict[str, dict[float, float]]] = field(default_factory=dict)
    # frame_stem -> {cid: {f: sky_pp ADU/px}}
    sky_pp: dict[str, dict[str, dict[float, float]]] = field(default_factory=dict)
    # frame_stem -> FWHM used for that frame (AUTH-01)
    fwhm_by_frame: dict[str, float] = field(default_factory=dict)
    elapsed_s: float = 0.0
    n_stars: int = 0
    n_frames_measured: int = 0
    n_fwhm_fallback_night: int = 0


def frame_stem_from_source(source_file: str) -> str:
    name = Path(str(source_file)).name
    if name.startswith("proc_") and name.endswith(".csv"):
        return name[5:-4]
    if name.lower().endswith(".fits"):
        return Path(name).stem
    return Path(name).stem


def _fwhm_from_fits_header(hdr: Any, *, night_fallback: float) -> tuple[float, bool]:
    """Return (fwhm_px, used_night_fallback). Prefer VY_FWHM (FWHM-AUTH-01)."""
    from aperture_policy import clamp_fwhm_px  # noqa: PLC0415

    raw = None
    try:
        raw = hdr.get("VY_FWHM")
    except Exception:  # noqa: BLE001
        raw = None
    frame = clamp_fwhm_px(raw)
    if frame is not None:
        return float(frame), False
    night = clamp_fwhm_px(night_fallback)
    if night is not None:
        return float(night), True
    return 5.0, True


def measure_night_grid(
    *,
    frames_dir: Path,
    catalog_ids: Sequence[str],
    f_grid: Sequence[float],
    fwhm_night_px: float,
    annulus_inner_fwhm: float = 2.7,
    annulus_outer_fwhm: float = 5.2,
    max_frames: int | None = None,
) -> ApertureGridNight:
    """Measure targets+comps on the f-grid; r = f x FWHM_frame per frame."""
    from astropy.io import fits  # noqa: PLC0415

    want = {str(c) for c in catalog_ids if str(c).strip()}
    night = ApertureGridNight(
        f_grid=[float(f) for f in f_grid],
        fwhm_night_px=float(fwhm_night_px),
        n_stars=len(want),
    )
    if not want or not Path(frames_dir).is_dir():
        return night
    t0 = time.perf_counter()
    fits_files = sorted(Path(frames_dir).glob("*.fits"))
    if max_frames is not None:
        fits_files = fits_files[: int(max_frames)]
    for fp in fits_files:
        stem = fp.stem
        if stem.upper() == "MASTERSTAR":
            continue
        proc = Path(frames_dir) / f"proc_{stem}.csv"
        if not proc.is_file():
            continue
        try:
            df = pd.read_csv(
                proc,
                usecols=lambda c: c in {"catalog_id", "x", "y"},
                low_memory=False,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("[APERTURE-DYNAMIC] proc read fail %s: %s", proc.name, exc)
            continue
        if "catalog_id" not in df.columns:
            continue
        df = df[df["catalog_id"].astype(str).isin(want)].copy()
        if df.empty:
            continue
        ids = df["catalog_id"].astype(str).tolist()
        xy = np.column_stack(
            [
                pd.to_numeric(df["x"], errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(df["y"], errors="coerce").to_numpy(dtype=float),
            ]
        )
        try:
            with fits.open(fp, memmap=True) as hdul:
                img = np.asarray(hdul[0].data, dtype=float)
                fwhm_frame, used_fb = _fwhm_from_fits_header(
                    hdul[0].header, night_fallback=float(fwhm_night_px)
                )
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("[APERTURE-DYNAMIC] FITS open fail %s: %s", fp.name, exc)
            continue
        if used_fb:
            night.n_fwhm_fallback_night += 1
            LOGGER.info(
                "[APERTURE-DYNAMIC] frame %s: no VY_FWHM - using night median %.4f px",
                stem,
                float(fwhm_frame),
            )
        night.fwhm_by_frame[stem] = float(fwhm_frame)
        fluxes, skies = measure_flux_on_f_grid(
            img,
            xy,
            f_grid=night.f_grid,
            fwhm_px=float(fwhm_frame),
            annulus_inner_fwhm=float(annulus_inner_fwhm),
            annulus_outer_fwhm=float(annulus_outer_fwhm),
        )
        per_cid: dict[str, dict[float, float]] = {cid: {} for cid in ids}
        per_sky: dict[str, dict[float, float]] = {cid: {} for cid in ids}
        for f, arr in fluxes.items():
            sky_arr = skies.get(f, np.full(len(ids), np.nan))
            for i, cid in enumerate(ids):
                v = float(arr[i]) if i < len(arr) else float("nan")
                per_cid[cid][float(f)] = v
                s = float(sky_arr[i]) if i < len(sky_arr) else float("nan")
                per_sky[cid][float(f)] = s
        night.frames[stem] = per_cid
        night.sky_pp[stem] = per_sky
        night.n_frames_measured += 1
    night.elapsed_s = float(time.perf_counter() - t0)
    LOGGER.info(
        "[APERTURE-DYNAMIC] grid measured: n_frames=%d n_stars=%d n_f=%d "
        "fwhm_fallback=%d elapsed=%.1fs",
        night.n_frames_measured,
        night.n_stars,
        len(night.f_grid),
        night.n_fwhm_fallback_night,
        night.elapsed_s,
    )
    return night


def series_from_grid(
    night: ApertureGridNight,
    catalog_id: str,
    f: float,
    frame_order: Sequence[str],
) -> np.ndarray:
    """Flux series for ``catalog_id`` at factor ``f`` in ``frame_order``."""
    out = np.full(len(frame_order), float("nan"), dtype=float)
    ff = float(f)
    cid = str(catalog_id)
    for i, stem in enumerate(frame_order):
        ent = night.frames.get(str(stem), {}).get(cid)
        if not ent:
            continue
        v = ent.get(ff)
        if v is None:
            for k, val in ent.items():
                if abs(float(k) - ff) < 1e-9:
                    v = val
                    break
        if v is not None and math.isfinite(float(v)):
            out[i] = float(v)
    return out


def sky_series_from_grid(
    night: ApertureGridNight,
    catalog_id: str,
    f: float,
    frame_order: Sequence[str],
) -> np.ndarray:
    """Annulus sky_pp (ADU/px) series for ``catalog_id`` at factor ``f``."""
    out = np.full(len(frame_order), float("nan"), dtype=float)
    ff = float(f)
    cid = str(catalog_id)
    for i, stem in enumerate(frame_order):
        ent = night.sky_pp.get(str(stem), {}).get(cid)
        if not ent:
            continue
        v = ent.get(ff)
        if v is None:
            for k, val in ent.items():
                if abs(float(k) - ff) < 1e-9:
                    v = val
                    break
        if v is not None and math.isfinite(float(v)):
            out[i] = float(v)
    return out


def _night_median_positive(arr: np.ndarray) -> float:
    a = np.asarray(arr, dtype=float)
    ok = a[np.isfinite(a) & (a > 0)]
    if ok.size == 0:
        ok = a[np.isfinite(a) & (a >= 0)]
    if ok.size == 0:
        return float("nan")
    return float(np.median(ok))


def choose_f_for_target(
    night: ApertureGridNight,
    *,
    target_cid: str,
    comp_ids: Sequence[str],
    frame_order: Sequence[str],
    gain: float = 1.0,
    read_noise: float = 10.0,
) -> PerTargetChoice:
    """Pick f* for one target from Howell S/N of the night-median growth curve.

    Comps still share the chosen f* (D5-1). Abbe p2p of the differential LC at
    each f is recorded as a diagnostic only.
    """
    snr_by_f: dict[float, float] = {}
    growth: dict[float, float] = {}
    sky_med: dict[float, float] = {}
    p2p_by_f: dict[float, float] = {}
    fwhm = float(night.fwhm_night_px) if math.isfinite(float(night.fwhm_night_px)) else 5.0
    if fwhm <= 0:
        fwhm = 5.0

    for f in night.f_grid:
        ff = float(f)
        t_flux = series_from_grid(night, target_cid, ff, frame_order)
        t_sky = sky_series_from_grid(night, target_cid, ff, frame_order)
        F_med = _night_median_positive(t_flux)
        sky_pp = _night_median_positive(t_sky)
        growth[ff] = F_med
        sky_med[ff] = sky_pp
        r_ap = max(0.5, ff * fwhm)
        area = math.pi * r_ap * r_ap
        snr_by_f[ff] = howell_snr(
            F_med, sky_pp if math.isfinite(sky_pp) else 0.0, area,
            gain=float(gain), read_noise=float(read_noise),
        )
        # Diagnostic: Abbe p2p of equal-weight differential LC at this f.
        t_mag = flux_to_mag(t_flux)
        comp_mags = {
            str(c): flux_to_mag(series_from_grid(night, str(c), ff, frame_order))
            for c in comp_ids
        }
        delta = equal_weight_ensemble_delta(t_mag, comp_mags)
        p2p_by_f[ff] = abbe_p2p_scatter(delta)

    f_star = pick_f_star_snr(snr_by_f)
    reason = "snr_argmax"
    if f_star is None:
        f_star = float(FALLBACK_F_STAR)
        reason = (
            f"fallback_f={FALLBACK_F_STAR:.2f}_howell_sky_limited_midband"
            "_no_finite_snr"
        )
    else:
        # Note flat-top selection in reason when multiple f within 1%.
        max_snr = max(
            (float(s) for s in snr_by_f.values() if math.isfinite(float(s))),
            default=float("nan"),
        )
        if math.isfinite(max_snr) and max_snr > 0:
            n_flat = sum(
                1
                for s in snr_by_f.values()
                if math.isfinite(float(s))
                and float(s) >= max_snr * (1.0 - SNR_FLAT_FRAC) - 1e-15
            )
            if n_flat > 1:
                reason = f"snr_argmax_flat_top_largest_f_n={n_flat}"

    edge = f_star_is_grid_edge(float(f_star), night.f_grid)
    r_ap = float(f_star) * float(fwhm)
    snr_max = float(snr_by_f.get(float(f_star), float("nan")))
    if not math.isfinite(snr_max):
        for k, v in snr_by_f.items():
            if abs(float(k) - float(f_star)) < 1e-9:
                snr_max = float(v)
                break
    return PerTargetChoice(
        catalog_id=str(target_cid),
        f_star=float(f_star),
        r_ap_px=float(r_ap),
        p2p_by_f={f"{k:.4g}": float(v) for k, v in sorted(p2p_by_f.items())},
        snr_by_f={f"{k:.4g}": float(v) for k, v in sorted(snr_by_f.items())},
        n_comps=len(list(comp_ids)),
        n_frames=len(list(frame_order)),
        f_edge=bool(edge),
        reason=str(reason),
        snr_max=float(snr_max) if math.isfinite(snr_max) else float("nan"),
        growth_F_by_f={f"{k:.4g}": float(v) for k, v in sorted(growth.items())},
        sky_pp_by_f={f"{k:.4g}": float(v) for k, v in sorted(sky_med.items())},
    )


def apply_grid_fluxes_to_frames(
    all_frames: pd.DataFrame,
    night: ApertureGridNight,
    *,
    catalog_ids: Sequence[str],
    f_star: float,
) -> pd.DataFrame:
    """Replace mag_inst / aperture_r_px with grid values at f* (r = f* x FWHM_frame)."""
    if all_frames is None or all_frames.empty or "source_file" not in all_frames.columns:
        return all_frames
    out = all_frames.copy()
    want = {str(c) for c in catalog_ids}
    for i, row in out.iterrows():
        cid = str(row.get("catalog_id", ""))
        if cid not in want:
            continue
        stem = frame_stem_from_source(str(row.get("source_file", "")))
        ent = night.frames.get(stem, {}).get(cid)
        if not ent:
            continue
        flux = ent.get(float(f_star))
        if flux is None:
            for k, val in ent.items():
                if abs(float(k) - float(f_star)) < 1e-9:
                    flux = val
                    break
        if flux is None or not math.isfinite(float(flux)) or float(flux) <= 0:
            continue
        fwhm = float(night.fwhm_by_frame.get(stem, night.fwhm_night_px))
        r_ap = float(f_star) * fwhm
        out.at[i, "mag_inst"] = float(-2.5 * math.log10(float(flux)))
        out.at[i, "aperture_r_px"] = float(r_ap)
        out.at[i, "aperture_f"] = float(f_star)
    return out


def write_per_target_choices(
    path: Path,
    choices: Mapping[str, PerTargetChoice],
    *,
    f_grid: Sequence[float],
    fwhm_night_px: float,
    elapsed_s: float,
    gain: float | None = None,
    read_noise: float | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_edge = sum(1 for c in choices.values() if bool(c.f_edge))
    payload = {
        "policy": "APERTURE-DYNAMIC-02",
        "mode": MODE_PER_TARGET,
        "criterion": "howell_snr_argmax",
        "fallback_f_star": float(FALLBACK_F_STAR),
        "snr_flat_frac": float(SNR_FLAT_FRAC),
        "f_grid": [float(f) for f in f_grid],
        "fwhm_night_px": float(fwhm_night_px),
        "elapsed_s": float(elapsed_s),
        "n_targets": len(choices),
        "n_f_edge": int(n_edge),
        "gain": float(gain) if gain is not None and math.isfinite(float(gain)) else None,
        "read_noise": (
            float(read_noise)
            if read_noise is not None and math.isfinite(float(read_noise))
            else None
        ),
        "targets": {
            cid: {
                "f_star": c.f_star,
                "r_ap_px": c.r_ap_px,
                "aperture_f_edge": bool(c.f_edge),
                "reason": c.reason,
                "snr_max": c.snr_max if math.isfinite(float(c.snr_max)) else None,
                "snr_by_f": c.snr_by_f,
                "p2p_by_f": c.p2p_by_f,
                "growth_F_by_f": c.growth_F_by_f,
                "sky_pp_by_f": c.sky_pp_by_f,
                "n_comps": c.n_comps,
                "n_frames": c.n_frames,
            }
            for cid, c in choices.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


__all__ = [
    "DEFAULT_APERTURE_F_GRID",
    "FALLBACK_F_STAR",
    "MODE_PER_TARGET",
    "SNR_FLAT_FRAC",
    "ApertureGridNight",
    "PerTargetChoice",
    "abbe_p2p_scatter",
    "apply_grid_fluxes_to_frames",
    "choose_f_for_target",
    "equal_weight_ensemble_delta",
    "f_star_is_grid_edge",
    "flux_to_mag",
    "frame_stem_from_source",
    "howell_snr",
    "howell_variance_adu2",
    "measure_flux_on_f_grid",
    "measure_night_grid",
    "normalize_f_grid",
    "pick_f_star",
    "pick_f_star_snr",
    "series_from_grid",
    "sky_series_from_grid",
    "write_per_target_choices",
]
