"""Scatter-optimal aperture radius selection for differential time series.

Howell SNR maximises single-measurement SNR. For VYVAR differential light curves
the relevant figure of merit is LC scatter of non-variables (Kepler CDPP /
eleanor min-CDPP / C-Munipack min Std.Dev). Measure a flux ladder once, then
evaluate scatter offline per candidate radius.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

LOGGER = logging.getLogger(__name__)

# Diagnostic ladder only (APERTURE-01c): production radii are f x FWHM.
# r_min = 0.75 FWHM. Pixel defaults below are fallbacks when FWHM is unknown.
DEFAULT_R_MIN_FWHM = 0.75
DEFAULT_R_MIN_PX = 1.5
DEFAULT_R_MAX_PX = 12.0
DEFAULT_R_STEP_PX = 0.5
DEFAULT_K_FWHM = (0.75, 1.0, 1.2, 1.35, 1.5, 1.7, 2.0, 2.5)


@dataclass
class LadderSpec:
    """Diagnostic radius ladder. Production radii are APERTURE-01 f x FWHM."""

    r_min_px: float = DEFAULT_R_MIN_PX
    r_max_px: float = DEFAULT_R_MAX_PX
    r_step_px: float = DEFAULT_R_STEP_PX
    r_min_fwhm: float = DEFAULT_R_MIN_FWHM
    why: str = (
        "Diagnostic only (APERTURE-01c). r_min = 0.75 FWHM; pixel 1.5-12 "
        "step 0.5 is the FWHM-unknown fallback. Production r = f x FWHM."
    )

    def radii_from_fwhm(self, fwhm_px: float) -> np.ndarray:
        fw = float(fwhm_px)
        if not math.isfinite(fw) or fw <= 0:
            return self.radii_px()
        r0 = max(0.5, float(self.r_min_fwhm) * fw)
        r1 = max(r0 + float(self.r_step_px), float(self.r_max_px))
        return np.arange(r0, r1 + 1e-9, float(self.r_step_px), dtype=np.float64)

    def radii_px(self) -> np.ndarray:
        step = float(self.r_step_px)
        if not math.isfinite(step) or step <= 0:
            step = DEFAULT_R_STEP_PX
        r0 = float(self.r_min_px)
        r1 = float(self.r_max_px)
        if not math.isfinite(r0) or r0 <= 0:
            r0 = DEFAULT_R_MIN_PX
        if not math.isfinite(r1) or r1 <= r0:
            r1 = DEFAULT_R_MAX_PX
        out = np.arange(r0, r1 + 0.5 * step, step, dtype=np.float64)
        if out.size == 0:
            out = np.array([r0], dtype=np.float64)
        return out


@dataclass
class ScatterCurve:
    """Scatter-versus-radius result for one evaluation set."""

    radii_px: list[float]
    scatter_mmag: list[float]
    n_stars: list[int]
    policy: str
    set_name: str
    best_r_px: float = float("nan")
    best_scatter_mmag: float = float("nan")
    shape: str = "unknown"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "radii_px": [round(float(r), 4) for r in self.radii_px],
            "scatter_mmag": [
                (round(float(s), 4) if math.isfinite(float(s)) else None)
                for s in self.scatter_mmag
            ],
            "n_stars": [int(n) for n in self.n_stars],
            "policy": self.policy,
            "set_name": self.set_name,
            "best_r_px": (
                round(float(self.best_r_px), 4) if math.isfinite(self.best_r_px) else None
            ),
            "best_scatter_mmag": (
                round(float(self.best_scatter_mmag), 4)
                if math.isfinite(self.best_scatter_mmag)
                else None
            ),
            "shape": self.shape,
            "notes": list(self.notes),
        }


def classify_scatter_curve_shape(
    radii: Sequence[float],
    scatter_mmag: Sequence[float],
    *,
    flat_tol_frac: float = 0.05,
) -> str:
    """Classify min region: sharp_min | flat_min | monotonic_down | monotonic_up | noisy."""
    r = np.asarray(radii, dtype=np.float64)
    s = np.asarray(scatter_mmag, dtype=np.float64)
    ok = np.isfinite(r) & np.isfinite(s)
    if int(ok.sum()) < 3:
        return "insufficient"
    r = r[ok]
    s = s[ok]
    i_min = int(np.argmin(s))
    s_min = float(s[i_min])
    if s_min <= 0:
        return "noisy"
    near = s <= s_min * (1.0 + float(flat_tol_frac))
    n_near = int(np.sum(near))
    if n_near >= max(3, int(0.35 * len(s))):
        return "flat_min"
    # Monotonic trends
    diffs = np.diff(s)
    if np.all(diffs <= 0):
        return "monotonic_down"
    if np.all(diffs >= 0):
        return "monotonic_up"
    # Sharp if neighbours rise > flat_tol
    left = float(s[i_min - 1]) if i_min > 0 else s_min
    right = float(s[i_min + 1]) if i_min + 1 < len(s) else s_min
    if left > s_min * (1.0 + flat_tol_frac) and right > s_min * (1.0 + flat_tol_frac):
        return "sharp_min"
    return "broad_min"


def robust_scatter_mmag(mag: np.ndarray) -> float:
    """1.4826 * MAD in mmag (finite samples only)."""
    x = np.asarray(mag, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 5:
        return float("nan")
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)) * 1.4826 * 1000.0)


def split_selection_holdout(
    star_ids: Sequence[str],
    *,
    seed: int = 51403,
    selection_frac: float = 0.5,
) -> tuple[list[str], list[str]]:
    """Disjoint selection / held-out split (P1)."""
    ids = [str(s).strip() for s in star_ids if str(s).strip()]
    ids = sorted(set(ids))
    rng = np.random.default_rng(int(seed))
    rng.shuffle(ids)
    n = len(ids)
    if n < 4:
        # Too few: put half (at least 1) in each when possible.
        mid = max(1, n // 2)
        return ids[:mid], ids[mid:] if mid < n else ids[:mid]
    n_sel = max(2, int(round(n * float(selection_frac))))
    n_sel = min(n_sel, n - 2)
    return ids[:n_sel], ids[n_sel:]


def flux_to_inst_mag(flux: np.ndarray) -> np.ndarray:
    f = np.asarray(flux, dtype=np.float64)
    out = np.full(f.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(f) & (f > 0)
    out[ok] = -2.5 * np.log10(f[ok])
    return out


def differential_mag_series(
    target_flux: np.ndarray,
    comp_flux: dict[str, np.ndarray],
    *,
    ac_delta_m: float | None = None,
) -> np.ndarray:
    """Simple ensemble differential mag: target - median(comps). Optional AC offset."""
    t = flux_to_inst_mag(target_flux)
    if not comp_flux:
        out = t.copy()
    else:
        stack = np.vstack([flux_to_inst_mag(v) for v in comp_flux.values()])
        ens = np.nanmedian(stack, axis=0)
        out = t - ens
    if ac_delta_m is not None and math.isfinite(float(ac_delta_m)):
        out = out + float(ac_delta_m)
    return out


def evaluate_scatter_at_radius(
    flux_by_star: dict[str, np.ndarray],
    eval_ids: Sequence[str],
    comp_ids: Sequence[str],
    *,
    ac_delta_m: float | None = None,
) -> tuple[float, int]:
    """Median robust scatter across eval stars; each uses comps excluding itself."""
    scatters: list[float] = []
    for sid in eval_ids:
        sid = str(sid)
        if sid not in flux_by_star:
            continue
        comps = {
            c: flux_by_star[c]
            for c in comp_ids
            if c != sid and c in flux_by_star
        }
        if len(comps) < 2:
            continue
        series = differential_mag_series(
            flux_by_star[sid], comps, ac_delta_m=ac_delta_m
        )
        sc = robust_scatter_mmag(series)
        if math.isfinite(sc):
            scatters.append(sc)
    if not scatters:
        return float("nan"), 0
    return float(np.median(scatters)), int(len(scatters))


def build_scatter_curve(
    radii_px: Sequence[float],
    flux_ladder: dict[float, dict[str, np.ndarray]],
    eval_ids: Sequence[str],
    comp_ids: Sequence[str],
    *,
    policy: str,
    set_name: str,
    ac_delta_m_by_r: dict[float, float] | None = None,
) -> ScatterCurve:
    """Evaluate scatter at each radius; pick argmin."""
    sc_list: list[float] = []
    n_list: list[int] = []
    r_list = [float(r) for r in radii_px]
    for r in r_list:
        key = float(r)
        # tolerate float key mismatch
        flux_map = flux_ladder.get(key)
        if flux_map is None:
            for k in flux_ladder:
                if abs(float(k) - key) < 1e-6:
                    flux_map = flux_ladder[k]
                    key = float(k)
                    break
        if flux_map is None:
            sc_list.append(float("nan"))
            n_list.append(0)
            continue
        ac = None
        if ac_delta_m_by_r is not None:
            ac = ac_delta_m_by_r.get(key)
            if ac is None:
                for k, v in ac_delta_m_by_r.items():
                    if abs(float(k) - key) < 1e-6:
                        ac = v
                        break
        sc, n = evaluate_scatter_at_radius(
            flux_map, eval_ids, comp_ids, ac_delta_m=ac
        )
        sc_list.append(sc)
        n_list.append(n)

    best_r = float("nan")
    best_sc = float("nan")
    finite = [(r, s) for r, s in zip(r_list, sc_list) if math.isfinite(s)]
    if finite:
        best_r, best_sc = min(finite, key=lambda t: t[1])
    shape = classify_scatter_curve_shape(r_list, sc_list)
    notes: list[str] = []
    if shape == "flat_min":
        notes.append(
            "Flat minimum: any radius in the flat band is equivalent within 5%; "
            "prefer the most robust (mid-band / larger EE) rather than 0.1 mmag winner."
        )
    return ScatterCurve(
        radii_px=r_list,
        scatter_mmag=sc_list,
        n_stars=n_list,
        policy=policy,
        set_name=set_name,
        best_r_px=best_r,
        best_scatter_mmag=best_sc,
        shape=shape,
        notes=notes,
    )


def measure_flux_ladder_frame(
    data: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    radii_px: Sequence[float],
    *,
    annulus_inner_px: float,
    annulus_outer_px: float,
    method: str = "exact",
) -> tuple[dict[float, np.ndarray], np.ndarray]:
    """One-pass multi-radius sky-subtracted fluxes via production aperture sum.

    IMPL-04: uses ``photometry_core._aperture_flux_sky_batch`` (photutils
    ``method='exact'``) - the same path as production ``enhance_catalog``. The old
    harness default ``method='center'`` produced integer/half-integer parity sawtooth
    and must not be used for radius selection. ``method`` is accepted only for the
    fire-proof test; production/scan callers leave the default.
    """
    pos = np.column_stack(
        [np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)]
    )
    n = int(pos.shape[0])
    d = np.asarray(data, dtype=np.float64)
    r_list = [float(r) for r in radii_px if math.isfinite(float(r)) and float(r) > 0]
    r_in = float(annulus_inner_px)
    r_out = float(annulus_outer_px)
    sky_pp = np.full(n, np.nan, dtype=np.float64)
    out: dict[float, np.ndarray] = {}

    if str(method).strip().lower() != "exact":
        # Fire-proof path only: intentional broken masking (parity sawtooth).
        from photutils.aperture import CircularAnnulus, CircularAperture
        from photutils.aperture import aperture_photometry as _aphot

        if (
            math.isfinite(r_in)
            and math.isfinite(r_out)
            and r_out > r_in > 0
            and n > 0
        ):
            try:
                ann = CircularAnnulus(pos, r_in=r_in, r_out=r_out)
                masks = ann.to_mask(method="center")
                if not isinstance(masks, (list, tuple)):
                    masks = [masks]
                for i, m in enumerate(masks):
                    try:
                        img = m.get_values(d)
                        img = np.asarray(img, dtype=np.float64)
                        img = img[np.isfinite(img)]
                        if img.size >= 8:
                            sky_pp[i] = float(np.median(img))
                    except Exception:  # noqa: BLE001
                        continue
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("[SCATTER-AP] annulus sky failed: %s", exc)
        for r in r_list:
            flux = np.full(n, np.nan, dtype=np.float64)
            try:
                ap = CircularAperture(pos, r=float(r))
                phot = _aphot(d, ap, method=method)
                area = float(math.pi * float(r) ** 2)
                sums = np.asarray(phot["aperture_sum"], dtype=np.float64)
                flux = sums - sky_pp * area
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("[SCATTER-AP] r=%.2f photometry failed: %s", r, exc)
            out[float(r)] = flux
        return out, sky_pp

    # Production path: shared batch/per-star exact aperture sum.
    from photometry_core import _aperture_flux_sky_batch

    for r in r_list:
        flux, sky = _aperture_flux_sky_batch(
            d, pos, float(r), float(annulus_inner_px), float(annulus_outer_px)
        )
        out[float(r)] = flux
        sky_pp = sky
    return out, sky_pp


def ac_delta_m_from_ee(
    r_px: float,
    ee_radii: Sequence[float],
    ee_curve: Sequence[float],
    *,
    ref_r_px: float | None = None,
) -> float:
    """Approximate Method-B-like AC as -2.5 log10(EE(r)/EE(ref))."""
    rr = np.asarray(ee_radii, dtype=np.float64)
    ee = np.asarray(ee_curve, dtype=np.float64)
    if rr.size < 2 or ee.size != rr.size:
        return 0.0
    ref = float(ref_r_px) if ref_r_px is not None and math.isfinite(float(ref_r_px)) else float(rr[-1])
    ee_r = float(np.interp(float(r_px), rr, ee))
    ee_ref = float(np.interp(ref, rr, ee))
    if not (math.isfinite(ee_r) and math.isfinite(ee_ref) and ee_r > 0 and ee_ref > 0):
        return 0.0
    return float(-2.5 * math.log10(ee_r / ee_ref))


def flat_aperture_table_from_radius(
    r_px: float,
    *,
    fwhm_px: float,
    meta: dict[str, Any] | None = None,
    mag_range: tuple[float, float] = (7.0, 18.0),
    mag_step: float = 0.5,
) -> dict[str, Any]:
    """SNR-table-shaped artifact with every mag bin at the scatter-chosen radius."""
    r = float(r_px)
    table = {
        round(float(m), 1): round(r, 3)
        for m in np.arange(float(mag_range[0]), float(mag_range[1]) + float(mag_step), float(mag_step))
    }
    out: dict[str, Any] = {
        "table": table,
        "fwhm_px": float(fwhm_px),
        "r_min_px": r,
        "r_max_px": r,
        "fixed_radius_px": r,
        "selection_criterion": "scatter",
        "ee_path": "scatter_optimal_fixed_radius",
        "bound_hit_by_mag": {round(float(m), 1): "none" for m in table},
        "n_bound_hits": 0,
        "ee_at_opt_by_mag": {k: float("nan") for k in table},
    }
    if meta:
        out.update(meta)
    return out


def calibrate_snr_zero_point_from_fluxes(
    mag: Sequence[float],
    flux_adu: Sequence[float],
    aperture_r_px: Sequence[float] | None = None,
    *,
    ee_radii: Sequence[float] | None = None,
    ee_curve: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Calibrate instrumental ZP so Ftot = 10**((ZP-G)/2.5) matches measured fluxes.

    If EE curve is given, convert aperture flux to total via Ftot = F(r)/EE(r).
    """
    m = np.asarray(mag, dtype=np.float64)
    f = np.asarray(flux_adu, dtype=np.float64)
    if aperture_r_px is not None and ee_radii is not None and ee_curve is not None:
        rr = np.asarray(aperture_r_px, dtype=np.float64)
        ee_r = np.asarray(ee_radii, dtype=np.float64)
        ee_c = np.asarray(ee_curve, dtype=np.float64)
        ee = np.interp(rr, ee_r, ee_c, left=np.nan, right=np.nan)
        ok = np.isfinite(m) & np.isfinite(f) & (f > 0) & np.isfinite(ee) & (ee > 0.05)
        ftot = np.where(ok, f / ee, np.nan)
    else:
        ok = np.isfinite(m) & np.isfinite(f) & (f > 0)
        ftot = np.where(ok, f, np.nan)
    zp = m + 2.5 * np.log10(ftot)
    zp = zp[np.isfinite(zp)]
    if zp.size < 5:
        return {
            "ok": False,
            "zero_point": 25.0,
            "n": int(zp.size),
            "reason": "too_few_stars",
        }
    med = float(np.median(zp))
    mad = float(np.median(np.abs(zp - med)) * 1.4826)
    return {
        "ok": True,
        "zero_point": med,
        "zero_point_mad": mad,
        "n": int(zp.size),
        "p16": float(np.percentile(zp, 16)),
        "p84": float(np.percentile(zp, 84)),
        "reason": "median_G_plus_2p5log10_Ftot",
    }
