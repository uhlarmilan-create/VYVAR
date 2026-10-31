"""Moved from photometry_core.py (CONSOLIDATE-01E1). Facade re-exports these names."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any
import json
import logging
import math
import numpy as np

from pipeline_constants import BKG_SCALE_R_CLAMP_HI, BKG_SCALE_R_CLAMP_LO, MAD_CONSISTENCY
from photometry_defs import ERR_BKG_SOURCE_EMPIRICAL, ERR_BKG_SOURCE_HOWELL_FALLBACK

if TYPE_CHECKING:
    pass

def _clamp_err_empty_apertures_min(n: int) -> int:
    try:
        v = int(n)
    except (TypeError, ValueError):
        v = 16
    return max(1, min(256, v))


def _robust_scatter_mad(values: np.ndarray) -> float:
    """Plain MAD scatter, 1.4826 x MAD, no clipping (Labbe et al. 2003 empty apertures)."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 3:
        return float("nan")
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    if mad <= 0:
        return 0.0
    return mad / MAD_CONSISTENCY


def _build_star_exclusion_mask(
    shape: tuple[int, ...],
    star_x: np.ndarray,
    star_y: np.ndarray,
    exclusion_radius_px: float,
    edge_margin_px: float,
) -> np.ndarray:
    """Boolean mask: True where empty apertures must not be placed."""
    ny, nx = int(shape[0]), int(shape[1])
    blocked = np.zeros((ny, nx), dtype=bool)
    em = int(math.ceil(max(0.0, float(edge_margin_px))))
    if em > 0:
        blocked[:em, :] = True
        blocked[-em:, :] = True
        blocked[:, :em] = True
        blocked[:, -em:] = True
    ex_r = float(exclusion_radius_px)
    if ex_r <= 0:
        return blocked
    ex_r2 = ex_r * ex_r
    xs = np.asarray(star_x, dtype=np.float64)
    ys = np.asarray(star_y, dtype=np.float64)
    ok = np.isfinite(xs) & np.isfinite(ys)
    # Canonical order: mask is OR-commutative, but keep draw/debug paths order-stable.
    order = np.lexsort((xs[ok], ys[ok]))
    xs_s = xs[ok][order]
    ys_s = ys[ok][order]
    for xi, yi in zip(xs_s, ys_s):
        x0 = max(0, int(math.floor(float(xi) - ex_r)) - 1)
        x1 = min(nx, int(math.ceil(float(xi) + ex_r)) + 2)
        y0 = max(0, int(math.floor(float(yi) - ex_r)) - 1)
        y1 = min(ny, int(math.ceil(float(yi) + ex_r)) + 2)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        dist2 = (xx - float(xi)) ** 2 + (yy - float(yi)) ** 2
        blocked[y0:y1, x0:x1] |= dist2 <= ex_r2
    return blocked


def _off_footprint_band(covered: np.ndarray | None, shape: tuple[int, ...], radius_px: float) -> np.ndarray:
    """True on pixels within ``radius_px`` of a pixel outside the data footprint (all False without one)."""
    if covered is None:
        return np.zeros(shape[:2], dtype=bool)
    cov = np.asarray(covered, dtype=bool)
    if cov.shape != tuple(shape[:2]):
        raise ValueError(f"footprint shape {cov.shape} != frame shape {tuple(shape[:2])}")
    if cov.all():
        return np.zeros(cov.shape, dtype=bool)
    from scipy.ndimage import distance_transform_edt  # noqa: PLC0415

    return distance_transform_edt(cov) <= float(radius_px)


def _canonicalize_star_xy(
    star_x: np.ndarray,
    star_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Sort finite (x,y) pairs and return (xs, ys, sha256 hex of canonical list)."""
    import hashlib

    xs = np.asarray(star_x, dtype=np.float64).ravel()
    ys = np.asarray(star_y, dtype=np.float64).ravel()
    n = min(xs.size, ys.size)
    xs, ys = xs[:n], ys[:n]
    ok = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[ok], ys[ok]
    if xs.size == 0:
        return xs, ys, hashlib.sha256(b"").hexdigest()
    order = np.lexsort((xs, ys))
    xs, ys = xs[order], ys[order]
    # Deduplicate exact duplicates after sort (stable membership).
    if xs.size > 1:
        keep = np.empty(xs.size, dtype=bool)
        keep[0] = True
        keep[1:] = (xs[1:] != xs[:-1]) | (ys[1:] != ys[:-1])
        xs, ys = xs[keep], ys[keep]
    blob = np.column_stack([xs, ys]).astype("<f8", copy=False).tobytes()
    return xs, ys, hashlib.sha256(blob).hexdigest()


def _labbe_debug_dump_enabled() -> bool:
    import os

    return str(os.environ.get("VYVAR_LABBE_DEBUG_DUMP", "")).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _labbe_debug_dump_path() -> Path:
    import os

    raw = str(os.environ.get("VYVAR_LABBE_DEBUG_DUMP_PATH", "")).strip()
    if raw:
        return Path(raw)
    return Path("tmp") / "labbe_debug_dump.jsonl"


def _labbe_append_debug_record(record: dict[str, Any]) -> None:
    if not _labbe_debug_dump_enabled():
        return
    path = _labbe_debug_dump_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    except OSError:
        pass


def _sigma_bkg_r_key(r: float) -> float:
    """Canonical dict key for per-radius ``sigma_bkg_ap`` transport.

    Every ``_sigma_by_r`` store AND lookup must go through this function
    (ERR-518-01: unrounded store key vs rounded lookup key lost 100%
    of empirical measurements in global_fixed mode).
    """
    return round(float(r), 4)

def _assert_inv_err_sigma_acct_01(
    sigma_by_r: dict[float, tuple[float, str]],
    src_col: np.ndarray,
    *,
    n: int,
    r_ap_arr: np.ndarray | None,
    r_ap: float,
) -> None:
    """INV-ERR-SIGMA-ACCT-01: measured empirical radii must project to rows."""
    from invariants_runtime import InvariantViolation  # noqa: PLC0415

    n_empirical_measured = sum(
        1 for _sig, _src in sigma_by_r.values() if _src == ERR_BKG_SOURCE_EMPIRICAL
    )
    n_empirical_assigned = int(np.sum(src_col == ERR_BKG_SOURCE_EMPIRICAL))
    if n_empirical_measured <= 0 or n_empirical_assigned > 0:
        return
    if r_ap_arr is not None:
        radii_requested = sorted(
            {
                _sigma_bkg_r_key(float(v))
                for v in r_ap_arr[np.isfinite(r_ap_arr) & (r_ap_arr > 0)]
            }
        )
    else:
        radii_requested = [_sigma_bkg_r_key(float(r_ap))]
    radii_measured = sorted(
        _sigma_bkg_r_key(r)
        for r, (_sig, src) in sigma_by_r.items()
        if src == ERR_BKG_SOURCE_EMPIRICAL
    )
    raise InvariantViolation(
        "INV-ERR-SIGMA-ACCT-01",
        "Labbe measurement succeeded for "
        f"{n_empirical_measured} radius value(s) but 0 of {n} rows received "
        "empirical sigma_bkg_ap; key projection is broken. "
        f"radii_measured={radii_measured} radii_requested={radii_requested}",
    )


def _clamp_err_empty_apertures_n(n: int) -> int:
    """Clamp ``err_empty_apertures_n`` to registry range 16..256."""
    try:
        v = int(n)
    except (TypeError, ValueError):
        v = 64
    return max(16, min(256, v))


def _labbe_content_seed_from_header(hdr: Any, *, r_ap: float) -> int:
    """Stable Labbe RNG seed from frame identity + aperture radius (F-431 / LABBE-DET)."""
    import hashlib

    def _hget(key: str) -> str:
        try:
            return str(hdr.get(key) or "")
        except Exception:  # noqa: BLE001
            return ""

    parts = [
        _hget("DATE-OBS"),
        _hget("FILENAME"),
        _hget("FRAME"),
        _hget("NAXIS1"),
        _hget("NAXIS2"),
        f"{float(r_ap):.4f}",
    ]
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False) % (2**63 - 1)

def _empty_aperture_net(
    d: np.ndarray, sky: np.ndarray, x_c: float, y_c: float, r_ap: float, r_in: float, r_out: float
) -> float:
    """Aperture sum of ``d`` minus area x the science local sky (annulus median of ``sky``); NaN without sky."""
    from photutils.aperture import CircularAperture  # noqa: PLC0415
    from photutils.aperture import aperture_photometry as _aphot  # noqa: PLC0415

    from sky_estimation import annulus_sky_medians  # noqa: PLC0415

    pos = np.array([[float(x_c), float(y_c)]], dtype=np.float64)
    ap = CircularAperture(pos, r=float(r_ap))
    sum_ap = float(np.asarray(_aphot(d, ap, method="exact")["aperture_sum"], dtype=np.float64).ravel()[0])
    sky_pp = float(annulus_sky_medians(sky, pos, float(r_in), float(r_out))[0])
    return float(sum_ap - sky_pp * float(ap.area)) if math.isfinite(sky_pp) else float("nan")


def measure_empty_aperture_sigma_bkg(
    data: np.ndarray,
    star_x: np.ndarray,
    star_y: np.ndarray,
    r_ap: float,
    r_in: float,
    r_out: float,
    *,
    sky: np.ndarray,
    covered: np.ndarray | None,
    n_apertures: int = 64,
    min_valid: int = 16,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    frame_id: str | None = None,
    star_list_source: str = "in_memory",
) -> tuple[float, int, str]:
    """Empirical aperture background noise via random empty apertures (Labbe et al. 2003).

    Each placement uses the same local sky as production science apertures: ``sky`` is the
    frame for local-sky medians (``sky_estimation.sky_image``: NaN off the aligned footprint and on
    detected sources) and ``sky_estimation.annulus_sky_medians`` takes the annulus median
    (SKY-ONE-2026-10, Milan 2026-10-28: one sky rule; it used the plain median of the whole annulus).
    G08-11: ``covered`` is the aligned frame's data footprint (``frame_footprint.aligned_footprint``;
    None = the whole frame is data). No empty aperture or its annulus may touch the aligner's constant
    fill: a placement there has a net sum of exactly 0 and pulls the scatter down.
    The robust scatter of net sums is ``sigma_bkg_ap``
    [ADU] and already includes background Poisson, read noise, resampling covariance,
    pedestal offsets, and the Merline & Howell (1995) sky-estimation term - do **not**
    add a separate RN or annulus term (would double-count).

    Determinism (LABBE-DET): star list is canonicalized (sorted by x,y); when ``rng`` is
    None a child Generator is derived via ``SeedSequence`` from ``seed`` (and r_ap) so
    draws are independent of call order / shared parent RNG.

    Args:
        seed: When ``rng`` is None, seed the Generator (F-431). Prefer a content-derived
            seed from the caller so same-draft re-photometry is byte-stable.
        frame_id / star_list_source: optional debug-dump metadata
            (``VYVAR_LABBE_DEBUG_DUMP=1``).

    Returns:
        (sigma_bkg_ap, n_valid, reason) - reason non-empty when measurement failed.
    """
    import hashlib

    xs_c, ys_c, star_list_hash = _canonicalize_star_xy(star_x, star_y)
    seed_value = int(seed) if seed is not None else None
    if rng is None:
        if seed_value is None:
            rng = np.random.default_rng(None)
        else:
            # Independent child RNG keyed on content seed + r_ap (order-independent).
            ss = np.random.SeedSequence(
                [int(seed_value), int(round(float(r_ap) * 10000.0)) & 0xFFFFFFFF]
            )
            rng = np.random.default_rng(ss)
    n_target = _clamp_err_empty_apertures_n(n_apertures)
    n_min = _clamp_err_empty_apertures_min(min_valid)
    if not (
        math.isfinite(r_ap)
        and r_ap > 0
        and math.isfinite(r_in)
        and r_in > 0
        and math.isfinite(r_out)
        and r_out > r_in
    ):
        return float("nan"), 0, "invalid_annulus_geometry"

    d = np.asarray(data, dtype=np.float64)
    if d.ndim != 2 or d.size == 0:
        return float("nan"), 0, "empty_image"
    sky_img = np.asarray(sky, dtype=np.float64)
    if sky_img.shape != d.shape:
        raise ValueError(f"sky image shape {sky_img.shape} != frame shape {d.shape}")
    d_fill = d
    if np.any(~np.isfinite(d)):
        d_fill = np.where(np.isfinite(d), d, float(np.nanmedian(d)) if np.any(np.isfinite(d)) else 0.0)

    margin_px = max(2.0, float(r_out) - float(r_in))
    excl_r = float(r_out) + margin_px
    edge_margin = float(r_out) + float(r_ap) + 1.0
    off_fp = _off_footprint_band(covered, d.shape, float(r_out) + 1.0)
    blocked = _build_star_exclusion_mask(d.shape, xs_c, ys_c, excl_r, edge_margin) | off_fp
    free_y, free_x = np.nonzero(~blocked)
    if free_x.size < n_min:
        # APERTURE-01: FWHM-scaled sky annulus (~9 x FWHM) around every catalog
        # star can fill the chip, so Labbe has nowhere to sit. Keep the aperture
        # star-free (4 x r_ap) and still require the empty aperture+annulus on-chip.
        excl_r = max(4.0 * float(r_ap), 8.0)
        blocked = _build_star_exclusion_mask(d.shape, xs_c, ys_c, excl_r, edge_margin) | off_fp
        free_y, free_x = np.nonzero(~blocked)
    mask_hash = hashlib.sha256(np.ascontiguousarray(blocked).view(np.uint8)).hexdigest()
    labbe_input_hash = hashlib.sha256(
        f"{star_list_hash}|{mask_hash}|{float(r_ap):.4f}|{seed_value}".encode("utf-8")
    ).hexdigest()

    if free_x.size < n_min:
        _labbe_append_debug_record(
            {
                "frame_id": frame_id,
                "r_ap": float(r_ap),
                "seed_value": seed_value,
                "star_list_source": star_list_source,
                "n_stars": int(xs_c.size),
                "star_list_hash": star_list_hash,
                "mask_hash": mask_hash,
                "labbe_input_hash": labbe_input_hash,
                "n_attempted": 0,
                "n_valid_apertures": 0,
                "first5_aperture_xy": [],
                "sigma_result": None,
                "reason": f"crowding: only {int(free_x.size)} candidate pixels (< {n_min})",
            }
        )
        return float("nan"), 0, f"crowding: only {int(free_x.size)} candidate pixels (< {n_min})"

    n_try = min(int(free_x.size), max(n_target * 8, n_target))
    idx = rng.choice(free_x.size, size=n_try, replace=False)
    net_sums: list[float] = []
    first5: list[list[float]] = []
    for j in idx:
        xc = float(free_x[j]) + 0.5
        yc = float(free_y[j]) + 0.5
        flux_net = _empty_aperture_net(d_fill, sky_img, xc, yc, float(r_ap), float(r_in), float(r_out))
        if math.isfinite(flux_net):
            net_sums.append(float(flux_net))
            if len(first5) < 5:
                first5.append([xc, yc])
        if len(net_sums) >= n_target:
            break

    n_valid = len(net_sums)
    if n_valid < n_min:
        _labbe_append_debug_record(
            {
                "frame_id": frame_id,
                "r_ap": float(r_ap),
                "seed_value": seed_value,
                "star_list_source": star_list_source,
                "n_stars": int(xs_c.size),
                "star_list_hash": star_list_hash,
                "mask_hash": mask_hash,
                "labbe_input_hash": labbe_input_hash,
                "n_attempted": int(n_try),
                "n_valid_apertures": int(n_valid),
                "first5_aperture_xy": first5,
                "sigma_result": None,
                "reason": f"crowding: {n_valid} valid empty apertures (< {n_min})",
            }
        )
        return float("nan"), n_valid, f"crowding: {n_valid} valid empty apertures (< {n_min})"
    sigma = _robust_scatter_mad(np.asarray(net_sums, dtype=np.float64))
    if not math.isfinite(sigma) or sigma < 0:
        return float("nan"), n_valid, "non_finite_scatter"
    _labbe_append_debug_record(
        {
            "frame_id": frame_id,
            "r_ap": float(r_ap),
            "seed_value": seed_value,
            "star_list_source": star_list_source,
            "n_stars": int(xs_c.size),
            "star_list_hash": star_list_hash,
            "mask_hash": mask_hash,
            "labbe_input_hash": labbe_input_hash,
            "n_attempted": int(n_try),
            "n_valid_apertures": int(n_valid),
            "first5_aperture_xy": first5,
            "sigma_result": float(sigma),
            "reason": "",
        }
    )
    return float(sigma), n_valid, ""


def _clamp_bkg_scale_r(r: float) -> float:
    if not math.isfinite(r):
        return float("nan")
    return float(max(BKG_SCALE_R_CLAMP_LO, min(BKG_SCALE_R_CLAMP_HI, float(r))))

def bkg_scale_ratio_empirical_over_howell(
    sigma_bkg_ap: float,
    sky_pp: float,
    area: float,
    n_sky: float,
    *,
    gain: float,
    read_noise: float,
) -> float:
    """Per-measurement r = sigma_bkg_ap^2 / CCD-equation background variance (``ccd_noise.ccd_sky_variance_adu2``,
    with the sky-estimation term; CCD-ONE-2026-10) for hybrid fallback calibration."""
    from ccd_noise import ccd_sky_variance_adu2  # noqa: PLC0415

    sig = float(sigma_bkg_ap)
    if not math.isfinite(sig) or sig < 0:
        return float("nan")
    hb = ccd_sky_variance_adu2(sky_pp, area, n_sky, gain=gain, read_noise=read_noise)
    if not math.isfinite(hb) or hb <= 0:
        return float("nan")
    return float(sig * sig / hb)

def compute_setup_bkg_scale_r(ratios: list[float]) -> tuple[float, int]:
    """Median empirical/Howell background variance ratio; clamped to [0.05, 2.0]."""
    ok = [float(r) for r in ratios if math.isfinite(float(r)) and float(r) > 0]
    if not ok:
        return float("nan"), 0
    return _clamp_bkg_scale_r(float(np.median(np.asarray(ok, dtype=np.float64)))), len(ok)

def scaled_sigma_bkg_ap_from_howell(
    sky_pp: float,
    area: float,
    n_sky: float,
    *,
    gain: float,
    read_noise: float,
    r_setup: float,
) -> float:
    """Calibrated fallback: sqrt(r_setup * CCD-equation background variance) [ADU] at aperture scale."""
    from ccd_noise import ccd_sky_variance_adu2  # noqa: PLC0415

    r_c = _clamp_bkg_scale_r(float(r_setup))
    hb = ccd_sky_variance_adu2(sky_pp, area, n_sky, gain=gain, read_noise=read_noise)
    if not math.isfinite(r_c) or not math.isfinite(hb) or hb < 0:
        return float("nan")
    return float(math.sqrt(r_c * hb))


def _annulus_sky_subtracted_flux(
    data: np.ndarray,
    x_c: float,
    y_c: float,
    r_ap: float,
    r_in: float,
    r_out: float,
) -> tuple[float, float, float]:
    """Sky-subtracted aperture sum, annulus sky median, peak in aperture (shared DAO/PSF path)."""
    if not (math.isfinite(x_c) and math.isfinite(y_c) and math.isfinite(r_ap) and r_ap > 0):
        return float("nan"), float("nan"), float("nan")
    try:
        from photutils.aperture import CircularAnnulus, CircularAperture
        from photutils.aperture import aperture_photometry as _aphot
    except ImportError:
        return float("nan"), float("nan"), float("nan")

    d = np.asarray(data, dtype=np.float64)
    if np.any(~np.isfinite(d)):
        fill = float(np.nanmedian(d)) if np.any(np.isfinite(d)) else 0.0
        d = np.where(np.isfinite(d), d, fill)

    pos = np.array([[float(x_c), float(y_c)]], dtype=np.float64)
    ap = CircularAperture(pos, r=float(r_ap))
    an = CircularAnnulus(pos, r_in=float(r_in), r_out=float(r_out))

    phot_ap = _aphot(d, ap, method="exact")
    sum_ap = float(np.asarray(phot_ap["aperture_sum"], dtype=np.float64).ravel()[0])
    area_ap = float(ap.area)

    sky_pp = float("nan")
    sky_ok = False
    ann_masks = an.to_mask(method="center")
    if not isinstance(ann_masks, (list, tuple)):
        ann_masks = [ann_masks]
    for amask in ann_masks:
        try:
            sky_pp = _sky_pp_from_annulus_mask(d, amask)
            sky_ok = math.isfinite(sky_pp)
            if sky_ok:
                break
        except (ValueError, TypeError, IndexError, AttributeError) as exc:
            from except_fix_counters import get_except_fix_counters

            get_except_fix_counters().sky_annulus_mask_fail += 1
            logging.error(
                "[PHOT] annulus sky mask failed x=%.2f y=%.2f: %s",
                float(x_c),
                float(y_c),
                exc,
            )
            continue

    if not sky_ok:
        from except_fix_counters import get_except_fix_counters

        get_except_fix_counters().sky_annulus_invalid += 1
        logging.error(
            "[PHOT] annulus sky invalid (no usable pixels) x=%.2f y=%.2f r_ap=%.2f",
            float(x_c),
            float(y_c),
            float(r_ap),
        )
        peak_local = float("nan")
        try:
            m_ap = ap.to_mask(method="center")
            if isinstance(m_ap, (list, tuple)):
                m0 = m_ap[0]
            else:
                m0 = m_ap
            vals = m0.get_values(d)
            peak_local = (
                float(np.nanmax(np.asarray(vals, dtype=np.float64)))
                if vals is not None
                else float("nan")
            )
        except (ValueError, TypeError, IndexError):
            peak_local = float("nan")
        return float("nan"), float("nan"), peak_local

    flux_net = float(sum_ap - sky_pp * area_ap)
    try:
        m_ap = ap.to_mask(method="center")
        if isinstance(m_ap, (list, tuple)):
            m0 = m_ap[0]
        else:
            m0 = m_ap
        vals = m0.get_values(d)
        peak_local = float(np.nanmax(np.asarray(vals, dtype=np.float64))) if vals is not None else float("nan")
    except Exception:  # noqa: BLE001
        peak_local = float("nan")

    return flux_net, sky_pp, peak_local


def _sky_pp_from_annulus_mask(d: np.ndarray, amask: Any) -> float:
    """Local sky (ADU/px): plain median of the annulus pixels (``ApertureMask.get_values``).

    Same pixels and estimator as ``_sky_pp_from_annulus_image(d, amask.to_image(d.shape))``
    but without building a full-frame mask image per star (5.6 ms -> 0.05 ms per star on a
    9.5 Mpx frame; PERF-2026-10).

    An annulus with no pixels on the frame has no local sky: NaN (the caller reports
    it as invalid) instead of substituting the whole-frame median.
    """
    from sky_estimation import sky_median_mask  # noqa: PLC0415

    return sky_median_mask(values=np.asarray(amask.get_values(d), dtype=np.float64))


def _aperture_flux_sky_per_star(
    d: np.ndarray,
    pos: np.ndarray,
    r_ap_arr: np.ndarray,
    r_in_arr: np.ndarray,
    r_out_arr: np.ndarray,
    *,
    sky: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-star circular aperture + annulus sky (photutils 2.3 requires scalar ``r`` per aperture).

    ``sky``: image for the annulus medians (NaN = not sky, SKY-SRC-MASK-2026-10); default ``d``.

    Aperture sum uses photutils ``method='exact'`` (fractional pixel overlap). Binary
    ``center`` masking produces radius-parity sawtooth in growth/scatter ladders
    (IMPL-04). Annulus sky still uses ``center`` masks for median sampling only.
    """
    from photutils.aperture import CircularAperture
    from photutils.aperture import aperture_photometry as _aphot

    from sky_estimation import annulus_sky_medians  # noqa: PLC0415

    n = int(len(pos))
    flux_arr = np.full(n, np.nan, dtype=np.float64)
    sky_pp_arr = np.full(n, np.nan, dtype=np.float64)
    n_fail = 0
    for idx in range(n):
        try:
            r_ap = float(r_ap_arr[idx])
            r_in = float(r_in_arr[idx])
            r_out = float(r_out_arr[idx])
            if not (
                math.isfinite(r_ap)
                and r_ap > 0
                and math.isfinite(r_in)
                and r_in > 0
                and math.isfinite(r_out)
                and r_out > r_in
            ):
                n_fail += 1
                continue
            xy = (float(pos[idx, 0]), float(pos[idx, 1]))
            ap_i = CircularAperture([xy], r=r_ap)
            phot_i = _aphot(d, ap_i, method="exact")
            sky_pp = float(annulus_sky_medians(d if sky is None else sky, np.array([xy]), r_in, r_out)[0])
            area = float(ap_i.area)
            flux_arr[idx] = float(phot_i["aperture_sum"][0]) - sky_pp * area
            sky_pp_arr[idx] = sky_pp
        except Exception:  # noqa: BLE001
            n_fail += 1
    if n_fail > 0:
        logging.warning(
            "[APERTURE] Per-star aperture: %d/%d positions failed or skipped",
            n_fail,
            n,
        )
    return flux_arr, sky_pp_arr


# --- moved from aperture_photometry (LAYER-2026-10) ---


def _photometric_error(
    flux: float,
    sky_pp: float,
    area: float,
    n_sky: float,
    *,
    gain: float,
    read_noise: float,
) -> float:
    """Relative photometric error err/flux from the CCD equation (``ccd_noise.ccd_rel_error``: Merline &
    Howell 1995 with the sky-estimation term, median sky; CCD-ONE-2026-10). ``flux`` and ``sky_pp`` in ADU
    (per px for sky), ``n_sky`` the number of sky pixels, ``gain`` e-/ADU, ``read_noise`` e-; NaN for flux <= 0."""
    from ccd_noise import ccd_rel_error  # noqa: PLC0415

    return float(ccd_rel_error(flux, sky_pp, area, n_sky, gain=gain, read_noise=read_noise))


def _photometric_error_with_bkg_mode(
    flux: float,
    *,
    sky_pp: float,
    area: float,
    n_sky: float,
    gain: float,
    read_noise: float,
    sigma_bkg_ap: float | None = None,
) -> tuple[float, str]:
    """Relative err/flux with empirical background term (F-BINGAIN-1).

    Empirical (production): ``var = F/g + sigma_bkg_ap^2`` - photutils/SExtractor
    pattern with measured ``sigma_bkg`` at aperture scale (Labbe et al. 2003).
    When ``sigma_bkg_ap`` is missing, the CCD equation (``_photometric_error``) is the
    data-conditioned fallback. Gain and read noise are required (ValueError).
    """
    from ccd_noise import check_gain_read_noise  # noqa: PLC0415

    g, _rn = check_gain_read_noise(gain, read_noise)
    if not math.isfinite(flux) or flux <= 0:
        return float("nan"), ERR_BKG_SOURCE_HOWELL_FALLBACK
    sig_ap = float(sigma_bkg_ap) if sigma_bkg_ap is not None else float("nan")
    if math.isfinite(sig_ap) and sig_ap >= 0:
        variance = flux / g + sig_ap * sig_ap
        if math.isfinite(variance) and variance >= 0:
            return math.sqrt(variance) / flux, ERR_BKG_SOURCE_EMPIRICAL
    err = _photometric_error(flux, sky_pp, area, n_sky, gain=g, read_noise=_rn)
    return err, ERR_BKG_SOURCE_HOWELL_FALLBACK
