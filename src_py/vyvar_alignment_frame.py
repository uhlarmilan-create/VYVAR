"""Per-frame astrometry alignment (DAO + WCS reproject / astroalign) for ProcessPoolExecutor workers.

Kept separate from ``pipeline.py`` to avoid circular imports while remaining picklable top-level
entry points for multiprocessing (Windows spawn).
"""

from __future__ import annotations

import logging
import math
import warnings
from pathlib import Path
from typing import Any, NamedTuple

import astropy.units as u
import numpy as np
import pandas as pd

from gaia_catalog_id import read_vyvar_csv
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs import FITSFixedWarning

from infolog import log_event
from plain_stats import plain_mean_med_std
from utils import (
    DAO_STAR_FINDER_NO_ROUNDNESS_FILTER,
    circular_angle_diff_deg,
    dao_detection_fwhm_pixels,
    fits_header_has_celestial_wcs,
    header_key_is_celestial_wcs,
    seeded_numpy_default_rng,
    strip_celestial_wcs_keys,
    VYVAR_RANDOM_SEED,
    wcs_rotation_angle_deg,
)

logger = logging.getLogger(__name__)



def _roundtrip_needed(w_in: WCS, w_out: WCS, shape_out: tuple[int, ...], step: int = 16) -> bool:
    """SPEED-2026-10: does reproject's coordinate round trip (out -> in -> out, > 1 px = NaN) reset
    any pixel? Checked on a grid every ``step`` px plus the last row / column; when none fails the
    full-image round trip is skipped (the same output, half the WCS transforms: 4 s of 5 s per
    frame). A failure anywhere on the grid keeps the full round trip."""
    from astropy.wcs.utils import pixel_to_pixel  # noqa: PLC0415

    ny, nx = int(shape_out[0]), int(shape_out[1])
    xs = np.unique(np.r_[np.arange(0, nx, step), nx - 1]).astype(float)
    ys = np.unique(np.r_[np.arange(0, ny, step), ny - 1]).astype(float)
    gx, gy = np.meshgrid(xs, ys)
    gx, gy = gx.ravel(), gy.ravel()
    ix, iy = pixel_to_pixel(w_out, w_in, gx, gy)
    bx, by = pixel_to_pixel(w_in, w_out, ix, iy)
    with np.errstate(invalid="ignore"):  # same test as reproject (NaN never resets)
        bad = (np.abs(bx - gx) > 1) | (np.abs(by - gy) > 1)
    return bool(np.any(bad))

def _as_fits_float32_image(data: Any) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(data, dtype=np.float32))


def _hdr_has_wcs(header: fits.Header) -> bool:
    return bool(fits_header_has_celestial_wcs(header))


def _alignment_emit_log(log_sink: list[str] | None, msg: str) -> None:
    if log_sink is not None:
        log_sink.append(msg)
    else:
        log_event(msg)


def _alignment_n_unique_spread_sample(
    arr_in: np.ndarray,
    *,
    fp_name: str = "",
    log_sink: list[str] | None = None,
    max_samples: int = 10_000,
) -> int:
    """Count unique values on a spread sample across the whole frame.

    Used by the constant-frame QC gate. Using the first N finite pixels is biased toward
    borders (often constant due to padding), which can incorrectly mark a valid aligned
    frame as 'constant'; hence the spread sampling.

    Returns the unique count (>= 0) or ``-1`` when the check itself is unavailable
    (EXC-0586 / EXCEPT-FIX-3 #9): on an internal error we surface it and return the ``-1``
    sentinel so callers do NOT reject a good frame just because the diagnostic failed. The
    previous ``return 0`` made ``n_unique <= 3`` reject the frame outright.
    """
    try:
        a = np.asarray(arr_in, dtype=np.float32)
        flat = a.ravel()
        if flat.size == 0:
            return 0
        finite = np.isfinite(flat)
        if not np.any(finite):
            return 0
        idx_all = np.flatnonzero(finite)
        n = int(min(int(max_samples), int(idx_all.size)))
        if n <= 0:
            return 0
        if idx_all.size <= n:
            samp = flat[idx_all]
        else:
            take = np.linspace(0, idx_all.size - 1, num=n, dtype=np.int64)
            samp = flat[idx_all[take]]
        return int(len(np.unique(samp)))
    except Exception as exc:  # noqa: BLE001
        from except_fix_counters import get_except_fix_counters

        get_except_fix_counters().align_unique_sample_fail += 1
        _alignment_emit_log(
            log_sink, f"WARNING: unique-spread check unavailable for {fp_name}: {exc}"
        )
        logger.error("[ALIGN] unique-spread sample failed for %s: %s", fp_name, exc)
        return -1


def _alignment_as_alignment_points(
    sources: Any, *, label: str, log_sink: list[str] | None
) -> np.ndarray:
    pts: np.ndarray
    try:
        if hasattr(sources, "colnames") and "x_centroid" in getattr(sources, "colnames", []) and "y_centroid" in getattr(
            sources, "colnames", []
        ):
            pts = np.transpose((sources["x_centroid"], sources["y_centroid"]))  # type: ignore[index]
        elif hasattr(sources, "dtype") and getattr(sources.dtype, "names", None):
            names = set(sources.dtype.names or ())
            if "x_centroid" in names and "y_centroid" in names:
                pts = np.transpose((sources["x_centroid"], sources["y_centroid"]))  # type: ignore[index]
            else:
                arr = np.asarray(sources, dtype=np.float64)
                pts = arr.reshape((-1, 2)) if arr.ndim == 1 and arr.size % 2 == 0 else arr
        else:
            arr = np.asarray(sources, dtype=np.float64)
            pts = arr.reshape((-1, 2)) if arr.ndim == 1 and arr.size % 2 == 0 else arr
    except Exception:  # noqa: BLE001
        pts = np.zeros((0, 2), dtype=np.float32)
    if pts.ndim != 2:
        pts = np.zeros((0, 2), dtype=np.float32)
    elif pts.shape[1] > 2:
        pts = np.asarray(pts[:, :2], dtype=np.float64)
    elif pts.shape[1] < 2:
        pts = np.zeros((0, 2), dtype=np.float32)
    pts = np.asarray(pts, dtype=np.float32)
    _alignment_emit_log(
        log_sink,
        f"DEBUG: Alignment input ({label}) type: {type(sources).__name__} and shape: {tuple(pts.shape)}",
    )
    return pts


def _alignment_detect_xy(
    img: np.ndarray,
    want_max: int,
    *,
    det_sigma: float,
    fwhm_px: float,
    label: str = "",
    log_sink: list[str] | None = None,
) -> np.ndarray:
    from photutils.detection import DAOStarFinder

    if int(want_max) <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    data = np.ascontiguousarray(img, dtype=np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    sky_pixels = data[data > 10.0]
    if sky_pixels.size > 0:
        m = float(np.nanmedian(sky_pixels))
    else:
        m = float(np.nanmedian(data))
    data -= np.float32(m)
    data = np.clip(data, -100.0, None).astype(np.float32, copy=False)
    _frame_name = label or "frame"
    _alignment_emit_log(
        log_sink,
        f"[alert] [ALIGNMENT] Frame: {_frame_name} | Offset Removed: {m:.2f} | Final Mean: {np.nanmean(data):.4f}",
    )
    arr = data
    finite = np.isfinite(data)
    if not np.any(finite):
        return np.zeros((0, 2), dtype=np.float32)
    try:
        _vals = arr[finite].astype(np.float64, copy=False)
        _mean = float(np.mean(_vals)) if _vals.size else 0.0
        _std = float(np.std(_vals)) if _vals.size else 0.0
        _max = float(np.max(_vals)) if _vals.size else 0.0
        _nm = label or "frame"
        _alignment_emit_log(
            log_sink,
            f"DEBUG: Image stats for {_nm} - Mean: {_mean:.6g}, Std: {_std:.6g}, Max: {_max:.6g}",
        )
    except Exception:  # noqa: BLE001
        pass
    arr_bg = arr
    used_bg_sub = False
    arr_bg = np.nan_to_num(arr_bg, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    _med_bg = float(np.nanmedian(arr_bg))
    _std_bg = float(np.nanstd(arr_bg))
    _floor_bg = _med_bg - 3.0 * _std_bg if np.isfinite(_std_bg) and _std_bg > 0 else _med_bg
    arr_bg = np.clip(arr_bg, _floor_bg, None).astype(np.float32, copy=False)
    _, med, bg_std = plain_mean_med_std(arr_bg[np.isfinite(arr_bg)])
    bg_std = float(bg_std) if np.isfinite(bg_std) else 0.0
    if bg_std <= 0:
        bg_std = 1.0
    if bg_std < 1e-6:
        _nm = label or "frame"
        _alignment_emit_log(
            log_sink,
            f"DEBUG: Image stats for {_nm} - suspiciously low Std after BG ({bg_std:.3e}); "
            "background subtraction may be too aggressive.",
        )
    data = np.nan_to_num((arr_bg - float(med)).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    sig = max(0.5, min(20.0, float(det_sigma)))
    threshold = float(sig) * float(bg_std)
    _nm = label or "frame"
    try:
        data_chk = np.asarray(data, dtype=np.float32)
        nan_count = int(np.isnan(data_chk).sum())
        inf_count = int(np.isinf(data_chk).sum())
        min_val = float(np.min(data_chk))
        max_val = float(np.max(data_chk))
        mean_val = float(np.mean(data_chk))
        _alignment_emit_log(log_sink, f"[DEBUG DATA CHECK] File: {_nm}")
        _alignment_emit_log(
            log_sink,
            f"[DEBUG DATA CHECK] Stats -> Min: {min_val:.4f}, Max: {max_val:.4f}, Mean: {mean_val:.4f}",
        )
        _alignment_emit_log(
            log_sink,
            f"[DEBUG DATA CHECK] Bad Pixels -> NaNs: {nan_count}, Infs: {inf_count}",
        )
        _alignment_emit_log(log_sink, f"[DEBUG DATA CHECK] Current Threshold in use: {threshold:.2f}")
    except Exception:  # noqa: BLE001
        pass
    _alignment_emit_log(
        log_sink,
        f"DEBUG: Using FWHM={float(fwhm_px):.2f} and Sigma={float(sig):.2f} (from alignment det_sigma) for detection.",
    )
    _alignment_emit_log(log_sink, f"DEBUG: Threshold set to {threshold:.2f} (using bg_std={bg_std:.2f})")
    fw = max(1.2, float(fwhm_px))
    sig_try: list[float] = []
    s_cur = float(sig)
    while s_cur >= 0.5 - 1e-9:
        ss = max(float(s_cur), 0.5)
        if not any(abs(ss - t) < 1e-9 for t in sig_try):
            sig_try.append(ss)
        s_cur -= 0.2

    def _run_dao_on(img_in: np.ndarray, sigma_list: list[float], *, nm: str) -> tuple[Any, float, int]:
        _best_tbl = None
        _best_n = -1
        _best_sigma = float(sigma_list[0] if sigma_list else sig)
        _used_sigma = _best_sigma
        for idx_s, s in enumerate(sigma_list):
            if idx_s > 0:
                _alignment_emit_log(log_sink, f"Trying lower threshold {float(s):.2f} for {nm}...")
            finder = DAOStarFinder(
                fwhm=fw,
                threshold=max(float(s) * float(bg_std), 1e-6),
                n_brightest=None,
                **DAO_STAR_FINDER_NO_ROUNDNESS_FILTER,
            )
            _tbl_try = finder(img_in)
            _n_try = int(len(_tbl_try)) if _tbl_try is not None else 0
            if _n_try > _best_n:
                _best_n = _n_try
                _best_tbl = _tbl_try
                _best_sigma = float(s)
            if _tbl_try is not None and _n_try >= 50:
                _used_sigma = float(s)
                return _tbl_try, _used_sigma, _n_try
        return _best_tbl, _best_sigma, max(0, _best_n)

    _nm = label or "frame"
    tbl, used_sigma, best_n = _run_dao_on(data, sig_try, nm=_nm)
    if (tbl is None or best_n <= 0) and used_bg_sub:
        _med_r = float(np.nanmedian(arr))
        _std_r = float(np.nanstd(arr))
        _floor_r = _med_r - 3.0 * _std_r if np.isfinite(_std_r) and _std_r > 0 else _med_r
        arr_raw2 = np.nan_to_num((arr - _med_r).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        arr_raw2 = np.clip(arr_raw2, _floor_r - _med_r, None).astype(np.float32, copy=False)
        _alignment_emit_log(log_sink, f"DEBUG: {_nm} - fallback without background subtraction (median-only).")
        tbl2, used_sigma2, best_n2 = _run_dao_on(arr_raw2, sig_try, nm=_nm)
        if best_n2 > best_n:
            tbl, used_sigma, best_n = tbl2, used_sigma2, best_n2
    if float(used_sigma) < float(sig) - 1e-9:
        _alignment_emit_log(
            log_sink,
            f"Star detection: used FWHM={fw:.2f}, Sigma={float(used_sigma):.2f} (fallback from {float(sig):.2f})",
        )
    if tbl is None or len(tbl) == 0:
        _alignment_emit_log(log_sink, f"DEBUG: Frame {_nm} - attempting detection at extreme sensitivity (threshold=1.0)")
        try:
            finder_ext = DAOStarFinder(
                fwhm=5.0,
                threshold=max(1.0 * float(bg_std), 1e-6),
                n_brightest=None,
                **DAO_STAR_FINDER_NO_ROUNDNESS_FILTER,
            )
            tbl_ext = finder_ext(data)
            if tbl_ext is not None and len(tbl_ext) > 0:
                tbl = tbl_ext
        except Exception:  # noqa: BLE001
            pass
    if tbl is None or len(tbl) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    k = int(min(len(tbl), int(want_max)))
    if len(tbl) > k:
        flux_np = np.asarray(tbl["flux"], dtype=np.float64)
        take = np.argpartition(flux_np, -k)[-k:]
        tbl = tbl[take]
    tbl.sort("flux")
    tbl = tbl[::-1]
    x = np.asarray(tbl["x_centroid"], dtype=np.float32)
    y = np.asarray(tbl["y_centroid"], dtype=np.float32)
    return np.stack([x, y], axis=1)


def _alignment_run_astroalign_points(
    *,
    source_pts: Any,
    target_pts: Any,
    image_source: np.ndarray,
    image_target: np.ndarray,
    max_control_points: int,
) -> tuple[np.ndarray | None, str | None, dict[str, Any] | None]:
    try:
        import astroalign  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return None, str(exc), None
    try:
        src = np.array(source_pts, dtype="float32")
        tgt = np.array(target_pts, dtype="float32")
        if src.ndim != 2 or src.shape[1] != 2:
            return None, f"Invalid source point format: shape={tuple(src.shape)}", None
        if tgt.ndim != 2 or tgt.shape[1] != 2:
            return None, f"Invalid target point format: shape={tuple(tgt.shape)}", None
        if src.shape == (0, 2) or tgt.shape == (0, 2):
            return None, "Alignment cannot start, empty coordinates!", None
        if len(src) < 10 or len(tgt) < 10:
            return None, "Insufficient stars", None
        mcp = max(12, min(int(max_control_points), int(min(len(src), len(tgt)))))
        with seeded_numpy_default_rng(VYVAR_RANDOM_SEED):
            t, _ = astroalign.find_transform(
                source=np.asarray(src, dtype=np.float32),
                target=np.asarray(tgt, dtype=np.float32),
                max_control_points=mcp,
            )
        aligned_data, _ = astroalign.apply_transform(t, image_source, image_target)
        handoff = {
            "method": "astroalign",
            "matrix": [float(x) for x in np.asarray(t.params, dtype=np.float64).reshape(-1)],
        }
        return _as_fits_float32_image(aligned_data), None, handoff
    except Exception as e:  # noqa: BLE001
        return None, str(e), None


class RegistrationCheck(NamedTuple):
    """ALIGN-VERIFY-2026-10: star-pair check of one frame->reference registration (px)."""

    resid_px: float
    p90_px: float
    pairs: int
    possible: int


_NO_CHECK = RegistrationCheck(float("nan"), float("nan"), 0, 0)


def _finite_xy(xy: Any) -> np.ndarray:
    a = np.asarray(xy, dtype=np.float64).reshape(-1, 2) if np.size(xy) else np.zeros((0, 2))
    return a[np.all(np.isfinite(a), axis=1)]


def _apply_matrix_3x3(matrix: Any, xy: np.ndarray) -> np.ndarray:
    """Map frame (x, y) through a row-major 3x3 frame->reference matrix."""
    m = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    p = np.c_[np.asarray(xy, dtype=np.float64)[:, :2], np.ones(len(xy))]
    q = p @ m.T
    w = np.where(np.abs(q[:, 2]) > 1e-12, q[:, 2], 1.0)
    return q[:, :2] / w[:, None]


def _mutual_pairs(src_ref: np.ndarray, ref_xy: np.ndarray, r_match: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mutual nearest neighbours within ``r_match`` px: (src_idx, ref_idx, distance). Non-finite
    points never pair."""
    from scipy.spatial import cKDTree  # noqa: PLC0415

    empty = (np.zeros(0, dtype=int), np.zeros(0, dtype=int), np.zeros(0))
    ok_s = np.all(np.isfinite(src_ref), axis=1) if len(src_ref) else np.zeros(0, dtype=bool)
    ok_r = np.all(np.isfinite(ref_xy), axis=1) if len(ref_xy) else np.zeros(0, dtype=bool)
    s_idx, r_idx = np.flatnonzero(ok_s), np.flatnonzero(ok_r)
    if len(s_idx) == 0 or len(r_idx) == 0 or not math.isfinite(float(r_match)):
        return empty
    s_xy, r_xy = src_ref[s_idx], ref_xy[r_idx]
    d_sr, i_sr = cKDTree(r_xy).query(s_xy, k=1, distance_upper_bound=float(r_match))
    _d_rs, i_rs = cKDTree(s_xy).query(r_xy, k=1, distance_upper_bound=float(r_match))
    si = np.flatnonzero(np.isfinite(d_sr))
    ri = i_sr[si]
    keep = i_rs[ri] == si
    return s_idx[si[keep]], r_idx[ri[keep]], d_sr[si][keep]


def registration_check(src_xy: Any, ref_xy: Any, matrix: Any, *, fwhm_px: float) -> RegistrationCheck:
    """ALIGN-VERIFY-2026-10: the frame's stars mapped through ``matrix`` (frame -> reference) and
    paired with the reference stars (mutual nearest neighbour within ALIGN_MATCH_FWHM x FWHM)."""
    from pipeline_constants import ALIGN_MATCH_FWHM  # noqa: PLC0415

    src, ref = _finite_xy(src_xy), _finite_xy(ref_xy)
    if matrix is None or len(src) == 0 or len(ref) == 0:
        return _NO_CHECK
    m = np.asarray(matrix, dtype=np.float64)
    if m.size != 9 or not np.all(np.isfinite(m)):
        return _NO_CHECK
    _si, _ri, d = _mutual_pairs(_apply_matrix_3x3(m, src), ref, ALIGN_MATCH_FWHM * float(fwhm_px))
    possible = int(min(len(src), len(ref)))
    if len(d) == 0:
        return RegistrationCheck(float("nan"), float("nan"), 0, possible)
    return RegistrationCheck(float(np.median(d)), float(np.percentile(d, 90)), int(len(d)), possible)


def registration_ok(chk: RegistrationCheck, *, fwhm_px: float) -> bool:
    from pipeline_constants import (  # noqa: PLC0415
        ALIGN_MIN_PAIR_FRAC,
        ALIGN_MIN_PAIRS,
        ALIGN_P90_FLOOR_PX,
        ALIGN_P90_MAX_FWHM,
    )

    return bool(
        math.isfinite(chk.p90_px)
        and chk.pairs >= max(ALIGN_MIN_PAIRS, math.ceil(ALIGN_MIN_PAIR_FRAC * chk.possible))
        and chk.p90_px <= max(ALIGN_P90_MAX_FWHM * float(fwhm_px), ALIGN_P90_FLOOR_PX)
    )


def phase_correlation_shift(data: np.ndarray, ref_data: np.ndarray) -> np.ndarray | None:
    """Translation-only first guess (frame -> reference) by windowed phase correlation.
    Used only to seed the star-matched fit, never as a registration on its own."""
    src = np.asarray(data, dtype=np.float32)
    ref = np.asarray(ref_data, dtype=np.float32)
    if src.shape != ref.shape or src.size == 0:
        return None
    win = (np.hanning(src.shape[0])[:, None] * np.hanning(src.shape[1])[None, :]).astype(np.float32)
    src0 = np.nan_to_num(src - np.nanmedian(src), nan=0.0, posinf=0.0, neginf=0.0) * win
    ref0 = np.nan_to_num(ref - np.nanmedian(ref), nan=0.0, posinf=0.0, neginf=0.0) * win
    try:
        from skimage.registration import phase_cross_correlation  # type: ignore  # noqa: PLC0415

        shift, _e, _p = phase_cross_correlation(ref0, src0, upsample_factor=10)
    except Exception:  # noqa: BLE001
        return None
    dy, dx = float(shift[0]), float(shift[1])
    if not (math.isfinite(dx) and math.isfinite(dy)):
        return None
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])


def rotation_180_guess(shape_hw: tuple[int, int]) -> np.ndarray:
    """180 deg about the image centre (meridian flip) - a first guess for the star-matched fit."""
    h, w = int(shape_hw[0]), int(shape_hw[1])
    return np.array([[-1.0, 0.0, w - 1.0], [0.0, -1.0, h - 1.0], [0.0, 0.0, 1.0]])


def _similarity_estimate(src: np.ndarray, dst: np.ndarray) -> Any:
    from skimage.transform import SimilarityTransform  # noqa: PLC0415

    if hasattr(SimilarityTransform, "from_estimate"):  # scikit-image >= 0.26
        tt = SimilarityTransform.from_estimate(src, dst)
        ok = bool(tt)
    else:
        tt = SimilarityTransform()
        ok = bool(tt.estimate(src, dst))
    if not ok or not np.all(np.isfinite(np.asarray(tt.params, dtype=np.float64))):
        return None
    return tt


def fit_similarity_from_stars(
    src_xy: Any, ref_xy: Any, guesses: list[Any], *, fwhm_px: float
) -> tuple[Any, RegistrationCheck]:
    """ALIGN-VERIFY-2026-10: similarity transform (shift + rotation + scale) frame -> reference
    fitted on the stars themselves. Each guess (3x3) is refined by pairing within shrinking radii
    (10, 4, 2 FWHM; at least 20 px first) and a least-squares fit on the mutual pairs; a fit from
    a wider radius is kept when a tighter radius finds too few pairs. Returns the best
    (skimage SimilarityTransform | None, RegistrationCheck)."""
    src, ref = _finite_xy(src_xy), _finite_xy(ref_xy)
    fw = float(fwhm_px)
    best: tuple[Any, RegistrationCheck] = (None, _NO_CHECK)
    if len(src) < 6 or len(ref) < 6 or not (math.isfinite(fw) and fw > 0):
        return best
    for g in guesses:
        if g is None:
            continue
        m = np.asarray(g, dtype=np.float64).reshape(3, 3)
        if not np.all(np.isfinite(m)):
            continue
        t = None
        for r in (max(20.0, 10.0 * fw), 4.0 * fw, 2.0 * fw):
            si, ri, _d = _mutual_pairs(_apply_matrix_3x3(m, src), ref, r)
            if len(si) < 6:
                break
            tt = _similarity_estimate(src[si], ref[ri])
            if tt is None:
                break
            t, m = tt, np.asarray(tt.params, dtype=np.float64)
        if t is None:
            continue
        chk = registration_check(src, ref, m, fwhm_px=fw)
        if math.isfinite(chk.resid_px) and (
            best[0] is None or (chk.pairs, -chk.resid_px) > (best[1].pairs, -best[1].resid_px)
        ):
            best = (t, chk)
    return best


def _estimate(cls: Any, src: np.ndarray, dst: np.ndarray, **kw: Any) -> Any:
    """skimage transform ``cls`` estimated src -> dst (API of scikit-image < and >= 0.26); None on failure."""
    if hasattr(cls, "from_estimate"):
        tt = cls.from_estimate(src, dst, **kw)
        ok = bool(tt)
    else:
        tt = cls()
        ok = bool(tt.estimate(src, dst, **kw))
    if not ok or not np.all(np.isfinite(np.asarray(tt.params, dtype=np.float64))):
        return None
    return tt


def fit_polynomial_from_stars(
    src_xy: Any, ref_xy: Any, seed_matrix: Any, *, fwhm_px: float
) -> tuple[Any, Any, RegistrationCheck] | None:
    """ALIGN-POLY-2026-10: polynomial frame -> reference transform (order ALIGN_POLY_ORDER) fitted on
    the star pairs, seeded by a 3x3 matrix (the best similarity). Pairs are mutual nearest
    neighbours within ALIGN_MATCH_FWHM x FWHM, refined three times. Returns (forward, inverse,
    RegistrationCheck) or None (too few pairs: at least 3 per coefficient)."""
    from skimage.transform import PolynomialTransform  # noqa: PLC0415

    from pipeline_constants import ALIGN_MATCH_FWHM, ALIGN_POLY_ORDER  # noqa: PLC0415

    src, ref = _finite_xy(src_xy), _finite_xy(ref_xy)
    if seed_matrix is None or len(src) == 0 or len(ref) == 0:
        return None
    m = np.asarray(seed_matrix, dtype=np.float64).reshape(3, 3)
    if not np.all(np.isfinite(m)):
        return None
    order = int(ALIGN_POLY_ORDER)
    n_min = 3 * (order + 1) * (order + 2) // 2
    r = ALIGN_MATCH_FWHM * float(fwhm_px)
    mapped = _apply_matrix_3x3(m, src)
    si = ri = np.zeros(0, dtype=int)
    for _ in range(3):
        si, ri, _d = _mutual_pairs(mapped, ref, r)
        if len(si) < n_min:
            return None
        fwd = _estimate(PolynomialTransform, src[si], ref[ri], order=order)
        if fwd is None:
            return None
        mapped = np.asarray(fwd(src), dtype=np.float64)
    si, ri, d = _mutual_pairs(mapped, ref, r)
    if len(si) < n_min:
        return None
    fwd = _estimate(PolynomialTransform, src[si], ref[ri], order=order)
    inv = _estimate(PolynomialTransform, ref[ri], src[si], order=order)
    if fwd is None or inv is None:
        return None
    mapped = np.asarray(fwd(src), dtype=np.float64)
    _si, _ri, d = _mutual_pairs(mapped, ref, r)
    possible = int(min(len(src), len(ref)))
    if len(d) == 0:
        return None
    return fwd, inv, RegistrationCheck(float(np.median(d)), float(np.percentile(d, 90)), int(len(d)), possible)


def registration_model_dbic(src_xy: Any, ref_xy: Any, seed_matrix: Any, *, fwhm_px: float) -> float | None:
    """ALIGN-MODEL-2026-10: BIC(similarity) - BIC(polynomial) of the frame -> reference mapping, both
    least-squares fits on the SAME star pairs (mutual nearest neighbours of the seed registration within
    ALIGN_MATCH_FWHM x FWHM). BIC = n ln(RSS / n) + k ln n, n = 2 x pairs (x and y residuals), k = 4 for
    the similarity, 6 per axis for the 2nd-order polynomial. Positive = the polynomial is preferred.
    None when the pairs are too few for the polynomial."""
    from skimage.transform import PolynomialTransform  # noqa: PLC0415

    from pipeline_constants import ALIGN_MATCH_FWHM, ALIGN_POLY_ORDER  # noqa: PLC0415

    src, ref = _finite_xy(src_xy), _finite_xy(ref_xy)
    if seed_matrix is None or len(src) == 0 or len(ref) == 0:
        return None
    m = np.asarray(seed_matrix, dtype=np.float64).reshape(3, 3)
    if not np.all(np.isfinite(m)):
        return None
    order = int(ALIGN_POLY_ORDER)
    k_poly = (order + 1) * (order + 2)  # coefficients of x and y together
    si, ri, _d = _mutual_pairs(_apply_matrix_3x3(m, src), ref, ALIGN_MATCH_FWHM * float(fwhm_px))
    if len(si) < 3 * k_poly // 2:
        return None
    a, b = src[si], ref[ri]
    sim = _similarity_estimate(a, b)
    poly = _estimate(PolynomialTransform, a, b, order=order)
    if sim is None or poly is None:
        return None
    n = 2 * len(a)
    rss_s = float(np.sum((np.asarray(sim(a)) - b) ** 2))
    rss_p = float(np.sum((np.asarray(poly(a)) - b) ** 2))
    # a frame identical to the reference (residual ~1e-13 px) has nothing to gain: ln(0) is no evidence
    if not (rss_s > 0 and rss_p > 0) or rss_s / n < 1e-4:
        return None
    bic_s = n * math.log(rss_s / n) + 4 * math.log(n)
    bic_p = n * math.log(rss_p / n) + k_poly * math.log(n)
    return float(bic_s - bic_p)


def projective_approximation(fwd: Any, shape_hw: tuple[int, int]) -> tuple[list[float], float]:
    """3x3 projective matrix closest to the polynomial ``fwd`` on a 9x9 grid over the frame, and the
    largest grid difference (px). Stored as VY_AM0..8 for the steps that use a matrix (data
    footprint, saturation back-mapping, field corners)."""
    from skimage.transform import ProjectiveTransform  # noqa: PLC0415

    h, w = int(shape_hw[0]), int(shape_hw[1])
    gx, gy = np.meshgrid(np.linspace(0, w - 1, 9), np.linspace(0, h - 1, 9))
    pts = np.c_[gx.ravel(), gy.ravel()]
    dst = np.asarray(fwd(pts), dtype=np.float64)
    pt = _estimate(ProjectiveTransform, pts, dst)
    if pt is None:
        raise ValueError("projective approximation of the polynomial registration failed")
    err = float(np.max(np.hypot(*(np.asarray(pt(pts)) - dst).T)))
    return [float(x) for x in np.asarray(pt.params, dtype=np.float64).reshape(-1)], err


def warp_to_reference(data: Any, ref_shape: tuple[int, int], inverse_map: Any) -> np.ndarray:
    """Resample a frame onto the reference grid exactly as astroalign.apply_transform does (cubic,
    constant fill with the frame median) - for transforms astroalign cannot apply (polynomial)."""
    from skimage.transform import warp  # noqa: PLC0415

    src = np.asarray(data)
    return warp(
        src,
        inverse_map=inverse_map,
        output_shape=tuple(ref_shape),
        order=3,
        mode="constant",
        cval=float(np.median(src)),
        clip=True,
        preserve_range=True,
    )


def _alignment_load_masterstar_catalog_points_for_frame(
    hdr_frame: fits.Header,
    *,
    shape_hw: tuple[int, int],
    platesolve_dir: Path,
    align_star_cap: int,
) -> np.ndarray:
    try:
        ms_csv = Path(platesolve_dir) / "masterstars.csv"
        if not ms_csv.is_file():
            return np.zeros((0, 2), dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            w = WCS(hdr_frame)
        if not getattr(w, "has_celestial", False):
            return np.zeros((0, 2), dtype=np.float32)
        mdf = read_vyvar_csv(
            ms_csv,
            low_memory=False,
            dtype={"catalog_id": str, "name": str},  # Gaia ID must be str - float64 loses digits
        )
        if "ra_deg" not in mdf.columns or "dec_deg" not in mdf.columns:
            return np.zeros((0, 2), dtype=np.float32)
        ra = pd.to_numeric(mdf["ra_deg"], errors="coerce").to_numpy(dtype=np.float64)
        de = pd.to_numeric(mdf["dec_deg"], errors="coerce").to_numpy(dtype=np.float64)
        ok = np.isfinite(ra) & np.isfinite(de)
        if not np.any(ok):
            return np.zeros((0, 2), dtype=np.float32)
        sky = SkyCoord(ra=ra[ok] * u.deg, dec=de[ok] * u.deg, frame="icrs")
        xp, yp = w.world_to_pixel(sky)
        xy = np.stack([np.asarray(xp, dtype=np.float32), np.asarray(yp, dtype=np.float32)], axis=1)
        h, wpx = int(shape_hw[0]), int(shape_hw[1])
        inb = (
            np.isfinite(xy[:, 0])
            & np.isfinite(xy[:, 1])
            & (xy[:, 0] >= 0)
            & (xy[:, 1] >= 0)
            & (xy[:, 0] < float(wpx))
            & (xy[:, 1] < float(h))
        )
        xy = xy[inb]
        if len(xy) > int(align_star_cap):
            xy = xy[: int(align_star_cap)]
        return np.asarray(xy, dtype=np.float32)
    except Exception:  # noqa: BLE001
        return np.zeros((0, 2), dtype=np.float32)


def _alignment_compute_one_frame(
    fp: Path,
    frame_index_1based: int,
    ctx: dict[str, Any],
    log_sink: list[str] | None,
) -> dict[str, Any]:
    """Align one light frame to reference; no disk I/O. ``log_sink`` collects messages for parent replay in workers."""
    ref_data = ctx["ref_data"]
    ref_hdr = ctx["ref_hdr"]
    ref_fp_name = str(ctx["ref_fp_name"])
    fixed_target_pts: np.ndarray = ctx["fixed_target_pts"]
    reference_list: list[Any] = ctx["reference_list"]
    platesolve_dir = Path(ctx["platesolve_dir"])
    align_star_cap = int(ctx["align_star_cap"])
    max_control_points = int(ctx.get("max_control_points", 80))
    min_detected_stars = int(ctx["min_detected_stars"])
    fb_align = float(ctx["fb_align"])
    rotation_ref_angle_deg = ctx.get("rotation_ref_angle_deg")

    ref_wcs_obj = None
    if bool(ctx.get("has_ref_wcs", False)):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FITSFixedWarning)
                ref_wcs_obj = WCS(ref_hdr) if _hdr_has_wcs(ref_hdr) else None
        except Exception:  # noqa: BLE001
            ref_wcs_obj = None

    with fits.open(fp, memmap=False) as hdul:
        hdr = hdul[0].header.copy()
        data = _as_fits_float32_image(hdul[0].data).astype(np.float32, copy=False)

    is_flipped = False
    try:
        if rotation_ref_angle_deg is not None:
            a_i = wcs_rotation_angle_deg(hdr)
            if a_i is not None:
                diff = circular_angle_diff_deg(float(a_i), float(rotation_ref_angle_deg))
                if diff > 175.0:
                    is_flipped = True
    except Exception:  # noqa: BLE001
        is_flipped = False

    fw_i = float(dao_detection_fwhm_pixels(hdr, configured_fallback=fb_align) or fb_align)
    data_to_detect = np.asarray(data, dtype=np.float32)
    try:
        _alignment_emit_log(
            log_sink,
            "DEBUG: Data stats for alignment - "
            f"Min: {np.min(data_to_detect):.2f}, "
            f"Max: {np.max(data_to_detect):.2f}, "
            f"Mean: {np.mean(data_to_detect):.2f}, "
            f"NaN count: {np.isnan(data_to_detect).sum()}",
        )
    except Exception:  # noqa: BLE001
        pass

    _attempts = [
        {"dao_sigma": 3.5, "max_stars": 200},
        {"dao_sigma": 2.5, "max_stars": 300},
        {"dao_sigma": 2.0, "max_stars": 400},
        {"dao_sigma": 1.5, "max_stars": 500},
    ]
    aligned_data: np.ndarray | None = None
    aligned_method = "astroalign"
    registration_handoff: dict[str, Any] | None = None
    xy_used = np.zeros((0, 2), dtype=np.float32)
    # Keep best available detection across attempts so that non-astroalign fallbacks
    # (phase correlation / WCS shift) still report a meaningful detected star count.
    xy_best = np.zeros((0, 2), dtype=np.float32)
    _attempt_ok = False
    _chk_wcs: RegistrationCheck | None = None  # set when the frame WCS failed the star check

    def _n_unique_spread_sample(arr_in: np.ndarray, *, max_samples: int = 10_000) -> int:
        # Thin closure over the module-level helper so the constant-frame gate can log via the
        # per-frame log_sink / name. See _alignment_n_unique_spread_sample (EXC-0586 / FIX-3 #9).
        return _alignment_n_unique_spread_sample(
            arr_in, fp_name=fp.name, log_sink=log_sink, max_samples=max_samples
        )

    for i_att, att in enumerate(_attempts, start=1):
        dao_sig = float(att["dao_sigma"])
        max_st = int(att["max_stars"])
        try:
            xy = _alignment_detect_xy(
                data_to_detect,
                int(max(100, max_st)),
                det_sigma=dao_sig,
                fwhm_px=fw_i,
                label=fp.name,
                log_sink=log_sink,
            )
        except Exception as _det_exc:  # noqa: BLE001
            # ALIGN-VERIFY-2026-10: never swallow a detection failure silently (526: 5 frames
            # failed every attempt on Milan's PC with no trace in the log)
            _alignment_emit_log(
                log_sink,
                f"WARNING: {fp.name} attempt {i_att} star detection failed "
                f"({type(_det_exc).__name__}: {_det_exc})",
            )
            xy = np.zeros((0, 2), dtype=np.float32)

        if len(xy) == 0:
            xy_ms = _alignment_load_masterstar_catalog_points_for_frame(
                hdr, shape_hw=tuple(data.shape), platesolve_dir=platesolve_dir, align_star_cap=align_star_cap
            )
            if len(xy_ms) > 0:
                xy = np.asarray(xy_ms[:max_st], dtype=np.float32)
                _alignment_emit_log(
                    log_sink,
                    f"DEBUG: {fp.name} attempt {i_att} - DAO=0, using fallback MASTERSTAR catalog ({len(xy)} points).",
                )
            else:
                _alignment_emit_log(
                    log_sink,
                    f"WARNING: {fp.name} attempt {i_att} failed (0 stars detected, no MASTERSTAR fallback), "
                    "retrying with relaxed parameters",
                )
                continue

        if len(xy) < int(min_detected_stars):
            _alignment_emit_log(
                log_sink,
                f"WARNING: {fp.name} attempt {i_att} failed ({len(xy)} stars < {int(min_detected_stars)}), "
                "retrying with relaxed parameters",
            )
            continue
        # Track best detection even if transform fails / produces constant frame.
        if len(xy) > len(xy_best):
            xy_best = np.asarray(xy, dtype=np.float32)

        try:
            aligned_data = None
            aligned_method = "astroalign"
            if ref_wcs_obj is not None and _hdr_has_wcs(hdr):
                try:
                    from reproject import reproject_interp  # type: ignore

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", FITSFixedWarning)
                        w_i = WCS(hdr)
                    if getattr(w_i, "has_celestial", False):
                        # ALIGN-MATRIX-2026-10: affine frame->reference fit through both WCS
                        _wcs_matrix = None
                        try:
                            _hh, _ww = np.asarray(data).shape[:2]
                            _gx, _gy = np.meshgrid(np.linspace(0, _ww - 1, 7), np.linspace(0, _hh - 1, 7))
                            _fx, _fy = _gx.ravel(), _gy.ravel()
                            _ra, _de = w_i.celestial.all_pix2world(_fx, _fy, 0)
                            _rx, _ry = ref_wcs_obj.celestial.all_world2pix(_ra, _de, 0)
                            _A = np.c_[_fx, _fy, np.ones_like(_fx)]
                            _cx, *_ = np.linalg.lstsq(_A, _rx, rcond=None)
                            _cy, *_ = np.linalg.lstsq(_A, _ry, rcond=None)
                            _wcs_matrix = [
                                float(_cx[0]), float(_cx[1]), float(_cx[2]),
                                float(_cy[0]), float(_cy[1]), float(_cy[2]),
                                0.0, 0.0, 1.0,
                            ]
                        except Exception:  # noqa: BLE001
                            _wcs_matrix = None
                        # ALIGN-WCS-2026-10: check the frame's WCS on its stars BEFORE reprojecting.
                        # A capture-program WCS (NINA: mount pointing + rotation, no distortion) is
                        # typically 2 px off - every frame of a night was reprojected, failed the star
                        # check and was warped again by the star-matched fit. Same outcome, no wasted
                        # reprojection: go straight to the star-matched fit with the WCS as first guess.
                        if _wcs_matrix is not None:
                            _chk_w = registration_check(xy, fixed_target_pts, _wcs_matrix, fwhm_px=fw_i)
                            if not registration_ok(_chk_w, fwhm_px=fw_i):
                                _alignment_emit_log(
                                    log_sink,
                                    f"WARNING: {fp.name} wcs_reproject not verified on stars (residual "
                                    f"{_chk_w.resid_px:.2f} px, p90 {_chk_w.p90_px:.2f} px, "
                                    f"{_chk_w.pairs}/{_chk_w.possible} pairs, FWHM {fw_i:.2f}) "
                                    "- trying the star-matched fit",
                                )
                                aligned_method = "wcs_reproject"
                                registration_handoff = {"method": "wcs_reproject", "matrix": _wcs_matrix}
                                xy_used = np.asarray(xy, dtype=np.float32)
                                _chk_wcs = _chk_w
                                break
                        aligned_data, _foot = reproject_interp(
                            (np.asarray(data, dtype=np.float32), w_i.celestial),
                            ref_wcs_obj.celestial,
                            shape_out=ref_data.shape,
                            roundtrip_coords=_roundtrip_needed(
                                w_i.celestial, ref_wcs_obj.celestial, ref_data.shape
                            ),
                        )
                        aligned_data = _as_fits_float32_image(aligned_data)
                        aligned_method = "wcs_reproject"
                        registration_handoff = {"method": "wcs_reproject"}
                        if _wcs_matrix is not None:
                            registration_handoff["matrix"] = _wcs_matrix
                except Exception:  # noqa: BLE001
                    aligned_data = None

            if aligned_data is None:
                _alignment_emit_log(
                    log_sink,
                    f"DEBUG: Alignment detection {fp.name} (attempt {i_att}) - DAO stars(source)={len(xy)}, "
                    f"DAO stars(target_ref)={len(fixed_target_pts)}",
                )
                if fixed_target_pts.shape[0] == 0:
                    raise RuntimeError(
                        "FATAL ERROR: reference points were cleared from memory before alignment started!"
                    )
                current_target = np.array(reference_list, dtype="float32")
                n_fit = int(min(len(current_target), len(xy)))
                xy_fit = _alignment_as_alignment_points(xy[:n_fit], label="source", log_sink=log_sink)
                ref_fit = _alignment_as_alignment_points(current_target[:n_fit], label="target", log_sink=log_sink)
                n_fit = int(min(len(xy_fit), len(ref_fit)))
                if n_fit > 0:
                    xy_fit = np.asarray(xy_fit[:n_fit], dtype=np.float32)
                    ref_fit = np.asarray(ref_fit[:n_fit], dtype=np.float32)
                mcp = max(12, min(int(max_control_points), n_fit))
                aligned_data, aa_err, reg_handoff = _alignment_run_astroalign_points(
                    source_pts=xy_fit,
                    target_pts=ref_fit,
                    image_source=data,
                    image_target=ref_data,
                    max_control_points=mcp,
                )
                if aligned_data is None:
                    raise RuntimeError(str(aa_err or "astroalign failed"))
                registration_handoff = reg_handoff

            try:
                arr_check = np.asarray(aligned_data, dtype=np.float32)
                n_unique = _n_unique_spread_sample(arr_check, max_samples=10_000)
                # EXCEPT-FIX-3 #9: only 0 <= n_unique <= 3 is truly constant; n_unique < 0
                # means the check was unavailable -> accept the frame (do not reject on a
                # diagnostic's own failure). See census EXC-0586.
                if 0 <= n_unique <= 3:
                    _alignment_emit_log(
                        log_sink,
                        f"WARNING: {fp.name} aligned frame is constant (n_unique={n_unique}), attempt {i_att} failed",
                    )
                    aligned_data = None
                    continue
            except Exception:  # noqa: BLE001
                pass

            m_al = float(np.nanmean(aligned_data)) if aligned_data is not None else float("nan")
            s_al = float(np.nanstd(aligned_data)) if aligned_data is not None else float("nan")
            if math.isfinite(m_al) and math.isfinite(s_al) and (m_al < 50.0) and (s_al < 20.0):
                _alignment_emit_log(
                    log_sink,
                    f"WARNING: {fp.name} attempt {i_att} failed (aligned frame black: mean {m_al:.1f}, "
                    f"std {s_al:.1f}), retrying with relaxed parameters",
                )
                aligned_data = None
                continue
            if float(np.nansum(np.abs(aligned_data))) < 1.0:
                _alignment_emit_log(
                    log_sink,
                    f"WARNING: {fp.name} attempt {i_att} failed (aligned frame empty), "
                    "retrying with relaxed parameters",
                )
                aligned_data = None
                continue

            xy_used = np.asarray(xy, dtype=np.float32)
            _attempt_ok = True
            break
        except Exception as _att_exc:  # noqa: BLE001
            aligned_data = None
            _alignment_emit_log(
                log_sink,
                f"WARNING: {fp.name} attempt {i_att} failed ({type(_att_exc).__name__}: {_att_exc}), "
                "retrying with relaxed parameters",
            )
            continue

    # ALIGN-VERIFY-2026-10 (Milan 2026-10-12): a registration is accepted only when the frame's
    # own stars land on the reference stars. Draft 526 (BO CVn): astroalign failed on 5 frames and
    # the translation-only phase-correlation fallback was accepted although the frames were rotated
    # 0.3-0.4 deg against the post-flip reference -> stars 1-7 px off the apertures, all flagged
    # aligned. The translation-only, WCS-shift and identity fallbacks are removed: the only
    # fallback is a similarity transform fitted on star pairs (phase correlation = first guess),
    # and a frame that cannot be verified is not aligned and not measured.
    src_v = np.asarray(xy_used if len(xy_used) else xy_best, dtype=np.float64).reshape(-1, 2)
    ref_v = np.asarray(fixed_target_pts, dtype=np.float64).reshape(-1, 2)
    chk = _NO_CHECK if _chk_wcs is None else _chk_wcs
    if _attempt_ok and aligned_data is not None:
        _mat = (registration_handoff or {}).get("matrix") if isinstance(registration_handoff, dict) else None
        chk = registration_check(src_v, ref_v, _mat, fwhm_px=fw_i)
        if not registration_ok(chk, fwhm_px=fw_i):
            _alignment_emit_log(
                log_sink,
                f"WARNING: {fp.name} {aligned_method} not verified on stars (residual {chk.resid_px:.2f} px, "
                f"p90 {chk.p90_px:.2f} px, {chk.pairs}/{chk.possible} pairs, FWHM {fw_i:.2f}) "
                "- trying the star-matched fit",
            )
            _attempt_ok, aligned_data = False, None

    _t = None
    if not _attempt_ok or aligned_data is None:
        _guesses: list[Any] = [
            phase_correlation_shift(data, ref_data), np.eye(3), rotation_180_guess(np.asarray(data).shape[:2]),
        ]
        if isinstance(registration_handoff, dict) and registration_handoff.get("matrix") is not None:
            _guesses.insert(0, registration_handoff.get("matrix"))
        try:
            _t, chk = fit_similarity_from_stars(src_v, ref_v, _guesses, fwhm_px=fw_i)
        except Exception as _fit_exc:  # noqa: BLE001
            _alignment_emit_log(log_sink, f"WARNING: {fp.name} star-matched fit failed: {_fit_exc}")
            _t, chk = None, _NO_CHECK
        if _t is not None and registration_ok(chk, fwhm_px=fw_i):
            try:
                import astroalign  # type: ignore  # noqa: PLC0415

                _warped, _ = astroalign.apply_transform(_t, data, ref_data)
                aligned_data = _as_fits_float32_image(_warped)
                aligned_method = "star_similarity"
                registration_handoff = {
                    "method": "star_similarity",
                    "matrix": [float(x) for x in np.asarray(_t.params, dtype=np.float64).reshape(-1)],
                }
                _attempt_ok = True
                if len(xy_used) == 0:
                    xy_used = np.asarray(src_v, dtype=np.float32)
                _alignment_emit_log(
                    log_sink,
                    f"INFO: {fp.name} star-matched similarity fit: rotation "
                    f"{math.degrees(float(_t.rotation)):.3f} deg, scale {float(_t.scale):.5f}, "
                    f"residual {chk.resid_px:.2f} px (p90 {chk.p90_px:.2f}) on {chk.pairs} pairs",
                )
            except Exception as _sim_exc:  # noqa: BLE001
                _alignment_emit_log(log_sink, f"WARNING: {fp.name} star-matched warp failed: {_sim_exc}")
                _attempt_ok, aligned_data = False, None

    # ALIGN-POLY-2026-10 (Milan 2026-10-24): a wide field shifted on the chip carries the lens
    # distortion the similarity cannot follow (MN Boo 538: p90 0.59-0.62 px, 101 of 139 frames
    # refused). Fit the polynomial on the star pairs of the best similarity and verify it with the
    # same gate (IRAF geomap "general" geometry: linear part + distortion).
    # ALIGN-MODEL-2026-10 (Milan 2026-10-24): a VERIFIED similarity (astroalign / star_similarity) is
    # also replaced when the frame's own stars prefer the polynomial by BIC (ALIGN_MODEL_DBIC).
    from pipeline_constants import ALIGN_MODEL_DBIC  # noqa: PLC0415

    _seed = None
    _model_dbic: float | None = None
    _verified_sim = bool(_attempt_ok and aligned_data is not None and aligned_method in ("astroalign", "star_similarity"))
    if _verified_sim and isinstance(registration_handoff, dict) and registration_handoff.get("matrix") is not None:
        _seed = np.asarray(registration_handoff["matrix"], dtype=np.float64)
        try:
            _model_dbic = registration_model_dbic(src_v, ref_v, _seed, fwhm_px=fw_i)
        except Exception as _bic_exc:  # noqa: BLE001
            _alignment_emit_log(log_sink, f"WARNING: {fp.name} model selection failed: {_bic_exc}")
            _model_dbic = None
        if _model_dbic is None or _model_dbic <= ALIGN_MODEL_DBIC:
            _seed = None  # the similarity stays
        else:
            _alignment_emit_log(
                log_sink,
                f"INFO: {fp.name} {aligned_method} verified, but the stars prefer the polynomial "
                f"(dBIC {_model_dbic:.1f} > {ALIGN_MODEL_DBIC:g})",
            )
    elif (not _attempt_ok or aligned_data is None) and _t is not None:
        _seed = np.asarray(_t.params, dtype=np.float64)
    if _seed is not None:
        try:
            _poly = fit_polynomial_from_stars(src_v, ref_v, _seed, fwhm_px=fw_i)
        except Exception as _poly_exc:  # noqa: BLE001
            _alignment_emit_log(log_sink, f"WARNING: {fp.name} polynomial fit failed: {_poly_exc}")
            _poly = None
        if _poly is not None and registration_ok(_poly[2], fwhm_px=fw_i):
            _fwd, _inv, chk = _poly
            from pipeline_constants import ALIGN_POLY_ORDER  # noqa: PLC0415

            try:
                _am, _am_err = projective_approximation(_fwd, np.asarray(data).shape[:2])
                aligned_data = _as_fits_float32_image(warp_to_reference(data, ref_data.shape, _inv))
                aligned_method = "star_polynomial"
                registration_handoff = {
                    "method": "star_polynomial",
                    "matrix": _am,
                    "matrix_max_err_px": _am_err,
                    "poly_order": int(ALIGN_POLY_ORDER),
                    "poly_fwd": [float(x) for x in np.asarray(_fwd.params, dtype=np.float64).reshape(-1)],
                    "poly_inv": [float(x) for x in np.asarray(_inv.params, dtype=np.float64).reshape(-1)],
                }
                _attempt_ok = True
                if len(xy_used) == 0:
                    xy_used = np.asarray(src_v, dtype=np.float32)
                _alignment_emit_log(
                    log_sink,
                    f"INFO: {fp.name} star-matched polynomial fit: residual {chk.resid_px:.2f} px "
                    f"(p90 {chk.p90_px:.2f}) on {chk.pairs} pairs; matrix approximation <= {_am_err:.2f} px",
                )
            except Exception as _pw_exc:  # noqa: BLE001
                _alignment_emit_log(log_sink, f"WARNING: {fp.name} polynomial warp failed: {_pw_exc}")
                if not _verified_sim:
                    _attempt_ok, aligned_data = False, None

    if not _attempt_ok or aligned_data is None:
        _alignment_emit_log(
            log_sink,
            f"ERROR: {fp.name} alignment not verified (residual {chk.resid_px:.2f} px, {chk.pairs} pairs) "
            "- frame skipped, not measured",
        )
        return {
            "kind": "failed_skip",
            "fp": str(fp.resolve()),
            "frame_index_1based": int(frame_index_1based),
            "is_flipped": bool(is_flipped),
            "star_count": {
                "file": fp.name,
                "frame_index": int(frame_index_1based),
                "detected_stars": int(len(src_v)),
                "aligned": False,
                "reason": "alignment_not_verified",
                "is_flipped": bool(is_flipped),
                "align_residual_px": float(chk.resid_px),
                "align_p90_px": float(chk.p90_px),
                "align_pairs": int(chk.pairs),
            },
        }

    strip_celestial_wcs_keys(hdr)
    for k, v in ref_hdr.items():
        if header_key_is_celestial_wcs(k):
            hdr[k] = v
    hdr["VYALGOK"] = (True, "Aligned to reference (passed black frame guard)")
    hdr["VY_ALGN"] = (True, "Aligned to reference")
    hdr["VY_REF"] = (ref_fp_name[:60], "Reference frame for alignment")
    hdr["VYALGM"] = (aligned_method[:30], "Alignment method (astroalign or WCS reproj.)")
    return {
        "kind": "aligned",
        "fp": str(fp.resolve()),
        "frame_index_1based": int(frame_index_1based),
        "is_flipped": bool(is_flipped),
        "hdr": hdr.copy(),
        "aligned_data": np.copy(aligned_data),
        "aligned_method": aligned_method,
        "registration_handoff": registration_handoff,
        "xy_used": xy_used,
        "fw_i": fw_i,
        "star_count": {
            "file": fp.name,
            "frame_index": int(frame_index_1based),
            "detected_stars": int(len(xy_used)),
            "aligned": True,
            "alignment_method": aligned_method,
            "is_flipped": bool(is_flipped),
            "align_residual_px": float(chk.resid_px),
            "align_p90_px": float(chk.p90_px),
            "align_pairs": int(chk.pairs),
            # ALIGN-MODEL-2026-10: BIC(similarity) - BIC(polynomial) of a verified similarity (> 10 -> polynomial)
            "align_model_dbic": float(_model_dbic) if _model_dbic is not None else float("nan"),
        },
    }


_ASTROMETRY_ALIGN_MP_CTX: dict[str, Any] | None = None


def _astrometry_align_mp_init(ctx: dict[str, Any]) -> None:
    global _ASTROMETRY_ALIGN_MP_CTX
    _ASTROMETRY_ALIGN_MP_CTX = ctx


def _astrometry_align_mp_task(item: tuple[str, int]) -> dict[str, Any]:
    assert _ASTROMETRY_ALIGN_MP_CTX is not None
    fp_s, idx = item
    logs: list[str] = []
    res = _alignment_compute_one_frame(Path(fp_s), int(idx), _ASTROMETRY_ALIGN_MP_CTX, logs)
    out = dict(res)
    out["log_events"] = logs
    return out
