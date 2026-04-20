"""Per-frame astrometry alignment (DAO + WCS reproject / astroalign) for ProcessPoolExecutor workers.

Kept separate from ``pipeline.py`` to avoid circular imports while remaining picklable top-level
entry points for multiprocessing (Windows spawn).
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs import FITSFixedWarning

from infolog import log_event
from utils import (
    DAO_STAR_FINDER_NO_ROUNDNESS_FILTER,
    circular_angle_diff_deg,
    dao_detection_fwhm_pixels,
    fits_header_has_celestial_wcs,
    header_key_is_celestial_wcs,
    strip_celestial_wcs_keys,
    wcs_rotation_angle_deg,
)


def _as_fits_float32_image(data: Any) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(data, dtype=np.float32))


def _hdr_has_wcs(header: fits.Header) -> bool:
    return bool(fits_header_has_celestial_wcs(header))


def _alignment_emit_log(log_sink: list[str] | None, msg: str) -> None:
    if log_sink is not None:
        log_sink.append(msg)
    else:
        log_event(msg)


def _alignment_as_alignment_points(
    sources: Any, *, label: str, log_sink: list[str] | None
) -> np.ndarray:
    pts: np.ndarray
    try:
        if hasattr(sources, "colnames") and "xcentroid" in getattr(sources, "colnames", []) and "ycentroid" in getattr(
            sources, "colnames", []
        ):
            pts = np.transpose((sources["xcentroid"], sources["ycentroid"]))  # type: ignore[index]
        elif hasattr(sources, "dtype") and getattr(sources.dtype, "names", None):
            names = set(getattr(sources.dtype, "names") or ())
            if "xcentroid" in names and "ycentroid" in names:
                pts = np.transpose((sources["xcentroid"], sources["ycentroid"]))  # type: ignore[index]
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
        f"DEBUG: Alignment vstup ({label}) type: {type(sources).__name__} a shape: {tuple(pts.shape)}",
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
    from astropy.stats import sigma_clipped_stats
    from photutils.background import Background2D, MedianBackground
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
        f"🚨 [ALIGNMENT] Frame: {_frame_name} | Offset Removed: {m:.2f} | Final Mean: {np.nanmean(data):.4f}",
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
            f"DEBUG: Image stats pre {_nm} - Mean: {_mean:.6g}, Std: {_std:.6g}, Max: {_max:.6g}",
        )
    except Exception:  # noqa: BLE001
        pass
    arr_bg = arr
    used_bg_sub = False
    try:
        _h, _w = int(arr.shape[0]), int(arr.shape[1])
        _bx = max(16, min(50, _w))
        _by = max(16, min(50, _h))
        bkg = Background2D(
            arr,
            box_size=(_by, _bx),
            filter_size=(3, 3),
            bkg_estimator=MedianBackground(),
        )
        arr_bg = np.asarray(arr - bkg.background, dtype=np.float32)
        used_bg_sub = True
    except Exception:  # noqa: BLE001
        arr_bg = arr
        used_bg_sub = False
    arr_bg = np.nan_to_num(arr_bg, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    _med_bg = float(np.nanmedian(arr_bg))
    _std_bg = float(np.nanstd(arr_bg))
    _floor_bg = _med_bg - 3.0 * _std_bg if np.isfinite(_std_bg) and _std_bg > 0 else _med_bg
    arr_bg = np.clip(arr_bg, _floor_bg, None).astype(np.float32, copy=False)
    _, med, clipped_std = sigma_clipped_stats(arr_bg[np.isfinite(arr_bg)], sigma=3.0, maxiters=5)
    clipped_std = float(clipped_std) if np.isfinite(clipped_std) else 0.0
    if clipped_std <= 0:
        clipped_std = 1.0
    if clipped_std < 1e-6:
        _nm = label or "frame"
        _alignment_emit_log(
            log_sink,
            f"DEBUG: Image stats pre {_nm} - podozrivo nízke Std po BG ({clipped_std:.3e}); "
            "background subtraction môže byť príliš agresívna.",
        )
    data = np.nan_to_num((arr_bg - float(med)).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    sig = max(0.5, min(20.0, float(det_sigma)))
    threshold = float(sig) * float(clipped_std)
    _nm = label or "frame"
    try:
        data_chk = np.asarray(data, dtype=np.float32)
        nan_count = int(np.isnan(data_chk).sum())
        inf_count = int(np.isinf(data_chk).sum())
        min_val = float(np.min(data_chk))
        max_val = float(np.max(data_chk))
        mean_val = float(np.mean(data_chk))
        _alignment_emit_log(log_sink, f"[DEBUG DATA CHECK] Súbor: {_nm}")
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
        f"DEBUG: Používam FWHM={float(fwhm_px):.2f} a Sigma={float(sig):.2f} (z alignment det_sigma) pre detekciu.",
    )
    _alignment_emit_log(log_sink, f"DEBUG: Threshold set to {threshold:.2f} (using clipped_std={clipped_std:.2f})")
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
                _alignment_emit_log(log_sink, f"Skúšam znížiť threshold na {float(s):.2f} pre {nm}...")
            finder = DAOStarFinder(
                fwhm=fw,
                threshold=max(float(s) * float(clipped_std), 1e-6),
                brightest=None,
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
        _alignment_emit_log(log_sink, f"DEBUG: {_nm} - fallback bez background subtraction (median-only).")
        tbl2, used_sigma2, best_n2 = _run_dao_on(arr_raw2, sig_try, nm=_nm)
        if best_n2 > best_n:
            tbl, used_sigma, best_n = tbl2, used_sigma2, best_n2
    if float(used_sigma) < float(sig) - 1e-9:
        _alignment_emit_log(
            log_sink,
            f"Detekcia hviezd: Použité FWHM={fw:.2f}, Sigma={float(used_sigma):.2f} (fallback z {float(sig):.2f})",
        )
    if tbl is None or len(tbl) == 0:
        _alignment_emit_log(log_sink, f"DEBUG: Snímka {_nm} - pokus o detekciu s extrémnou citlivosťou (threshold=1.0)")
        try:
            finder_ext = DAOStarFinder(
                fwhm=5.0,
                threshold=max(1.0 * float(clipped_std), 1e-6),
                brightest=None,
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
    x = np.asarray(tbl["xcentroid"], dtype=np.float32)
    y = np.asarray(tbl["ycentroid"], dtype=np.float32)
    return np.stack([x, y], axis=1)


def _alignment_run_astroalign_points(
    *,
    source_pts: Any,
    target_pts: Any,
    image_source: np.ndarray,
    image_target: np.ndarray,
    max_control_points: int,
) -> tuple[np.ndarray | None, str | None]:
    try:
        import astroalign  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)
    try:
        src = np.array(source_pts, dtype="float32")
        tgt = np.array(target_pts, dtype="float32")
        if src.ndim != 2 or src.shape[1] != 2:
            return None, f"Nesprávny formát source bodov: shape={tuple(src.shape)}"
        if tgt.ndim != 2 or tgt.shape[1] != 2:
            return None, f"Nesprávny formát target bodov: shape={tuple(tgt.shape)}"
        if src.shape == (0, 2) or tgt.shape == (0, 2):
            return None, "Alignment nemôže začať, prázdne súradnice!"
        if len(src) < 10 or len(tgt) < 10:
            return None, "Insufficient stars"
        mcp = max(12, min(int(max_control_points), int(min(len(src), len(tgt)))))
        t, _ = astroalign.find_transform(
            source=np.asarray(src, dtype=np.float32),
            target=np.asarray(tgt, dtype=np.float32),
            max_control_points=mcp,
        )
        aligned_data, _ = astroalign.apply_transform(t, image_source, image_target)
        return _as_fits_float32_image(aligned_data), None
    except Exception as e:  # noqa: BLE001
        return None, str(e)


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
        mdf = pd.read_csv(ms_csv)
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
    min_detected_stars = int(ctx["min_detected_stars"])
    max_detected_stars = int(ctx["max_detected_stars"])
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
    xy_used = np.zeros((0, 2), dtype=np.float32)
    _attempt_ok = False

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
        except Exception:
            xy = np.zeros((0, 2), dtype=np.float32)

        if len(xy) == 0:
            xy_ms = _alignment_load_masterstar_catalog_points_for_frame(
                hdr, shape_hw=tuple(data.shape), platesolve_dir=platesolve_dir, align_star_cap=align_star_cap
            )
            if len(xy_ms) > 0:
                xy = np.asarray(xy_ms[:max_st], dtype=np.float32)
                _alignment_emit_log(
                    log_sink,
                    f"DEBUG: {fp.name} attempt {i_att} - DAO=0, používam fallback MASTERSTAR katalóg ({len(xy)} bodov).",
                )
            else:
                _alignment_emit_log(
                    log_sink, f"WARNING: {fp.name} attempt {i_att} zlyhalo, skúšam s uvoľnenými parametrami"
                )
                continue

        if len(xy) < int(min_detected_stars):
            _alignment_emit_log(
                log_sink, f"WARNING: {fp.name} attempt {i_att} zlyhalo, skúšam s uvoľnenými parametrami"
            )
            continue

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
                        aligned_data, _foot = reproject_interp(
                            (np.asarray(data, dtype=np.float32), w_i.celestial),
                            ref_wcs_obj.celestial,
                            shape_out=ref_data.shape,
                        )
                        aligned_data = _as_fits_float32_image(aligned_data)
                        aligned_method = "wcs_reproject"
                except Exception:  # noqa: BLE001
                    aligned_data = None

            if aligned_data is None:
                _alignment_emit_log(
                    log_sink,
                    f"DEBUG: Alignment detekcia {fp.name} (attempt {i_att}) - DAO hviezd(source)={len(xy)}, "
                    f"DAO hviezd(target_ref)={len(fixed_target_pts)}",
                )
                if fixed_target_pts.shape[0] == 0:
                    raise RuntimeError(
                        "FATÁLNA CHYBA: Referenčné body boli vymazané z pamäte pred štartom alignmentu!"
                    )
                current_target = np.array(reference_list, dtype="float32")
                n_fit = int(min(len(current_target), len(xy)))
                xy_fit = _alignment_as_alignment_points(xy[:n_fit], label="source", log_sink=log_sink)
                ref_fit = _alignment_as_alignment_points(current_target[:n_fit], label="target", log_sink=log_sink)
                n_fit = int(min(len(xy_fit), len(ref_fit)))
                if n_fit > 0:
                    xy_fit = np.asarray(xy_fit[:n_fit], dtype=np.float32)
                    ref_fit = np.asarray(ref_fit[:n_fit], dtype=np.float32)
                mcp = max(12, min(max_st, n_fit))
                aligned_data, aa_err = _alignment_run_astroalign_points(
                    source_pts=xy_fit,
                    target_pts=ref_fit,
                    image_source=data,
                    image_target=ref_data,
                    max_control_points=mcp,
                )
                if aligned_data is None:
                    raise RuntimeError(str(aa_err or "astroalign failed"))

            try:
                arr_check = np.asarray(aligned_data, dtype=np.float32)
                finite_chk = arr_check[np.isfinite(arr_check)]
                if finite_chk.size:
                    samp = finite_chk[: min(10_000, int(finite_chk.size))]
                    n_unique = int(len(np.unique(samp)))
                else:
                    n_unique = 0
                if n_unique <= 3:
                    _alignment_emit_log(
                        log_sink,
                        f"WARNING: {fp.name} aligned frame je konštantný (n_unique={n_unique}), pokus {i_att} zlyhal",
                    )
                    aligned_data = None
                    continue
            except Exception:  # noqa: BLE001
                pass

            m_al = float(np.nanmean(aligned_data)) if aligned_data is not None else float("nan")
            s_al = float(np.nanstd(aligned_data)) if aligned_data is not None else float("nan")
            if math.isfinite(m_al) and math.isfinite(s_al) and (m_al < 50.0) and (s_al < 20.0):
                _alignment_emit_log(
                    log_sink, f"WARNING: {fp.name} attempt {i_att} zlyhalo, skúšam s uvoľnenými parametrami"
                )
                aligned_data = None
                continue
            if float(np.nansum(np.abs(aligned_data))) < 1.0:
                _alignment_emit_log(
                    log_sink, f"WARNING: {fp.name} attempt {i_att} zlyhalo, skúšam s uvoľnenými parametrami"
                )
                aligned_data = None
                continue

            xy_used = np.asarray(xy, dtype=np.float32)
            _attempt_ok = True
            break
        except Exception:
            aligned_data = None
            _alignment_emit_log(
                log_sink, f"WARNING: {fp.name} attempt {i_att} zlyhalo, skúšam s uvoľnenými parametrami"
            )
            continue

    # Fallback: if astroalign failed and identity would be used, try a simple WCS-based shift (dx, dy)
    # computed from CRVAL mapping between frame and reference WCS. This avoids producing constant frames.
    if (not _attempt_ok or aligned_data is None) and ref_wcs_obj is not None and _hdr_has_wcs(hdr):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FITSFixedWarning)
                wcs_frame = WCS(hdr)
            if getattr(wcs_frame, "has_celestial", False) and getattr(ref_wcs_obj, "has_celestial", False):
                crval_frame = wcs_frame.wcs.crval  # [ra, dec]
                px_ref = ref_wcs_obj.all_world2pix([[crval_frame[0], crval_frame[1]]], 0)[0]
                crpix_ref = ref_wcs_obj.wcs.crpix  # [x, y]
                dx = float(crpix_ref[0] - px_ref[0])
                dy = float(crpix_ref[1] - px_ref[1])
                naxis2, naxis1 = int(ref_data.shape[0]), int(ref_data.shape[1])
                if abs(dx) < naxis1 * 0.5 and abs(dy) < naxis2 * 0.5:
                    from scipy.ndimage import shift as ndimage_shift

                    shifted = ndimage_shift(
                        np.asarray(data, dtype=np.float32),
                        shift=[dy, dx],  # (y, x)
                        mode="constant",
                        cval=0.0,
                        order=1,
                        prefilter=False,
                    )
                    shifted = _as_fits_float32_image(shifted)
                    # Sanity: avoid constant/empty frames.
                    finite_chk = shifted[np.isfinite(shifted)]
                    n_unique = int(len(np.unique(finite_chk[: min(10_000, int(finite_chk.size))]))) if finite_chk.size else 0
                    if n_unique > 3 and float(np.nansum(np.abs(shifted))) >= 1.0:
                        aligned_data = shifted
                        aligned_method = "wcs_shift"
                        _attempt_ok = True
                        _alignment_emit_log(
                            log_sink,
                            f"INFO: WCS alignment fallback: dx={dx:.1f}px dy={dy:.1f}px pre {fp.name}",
                        )
        except Exception as _we:  # noqa: BLE001
            _alignment_emit_log(log_sink, f"WARNING: WCS alignment fallback zlyhal: {_we}")

    if not _attempt_ok or aligned_data is None:
        m_id = float(np.nanmean(data_to_detect)) if data_to_detect is not None else float("nan")
        s_id = float(np.nanstd(data_to_detect)) if data_to_detect is not None else float("nan")
        if math.isfinite(m_id) and math.isfinite(s_id) and (m_id < 50.0) and (s_id < 20.0):
            _alignment_emit_log(
                log_sink,
                f"ERROR: alignment zlyhal pre {fp.name}, identity nepoužitá (black frame guard), preskakujem.",
            )
            return {
                "kind": "failed_skip",
                "fp": str(fp.resolve()),
                "frame_index_1based": int(frame_index_1based),
                "is_flipped": bool(is_flipped),
                "star_count": {
                    "file": fp.name,
                    "frame_index": int(frame_index_1based),
                    "detected_stars": int(len(xy_used)),
                    "aligned": False,
                    "reason": "alignment_failed_black_guard",
                    "is_flipped": bool(is_flipped),
                },
            }

        aligned_data = _as_fits_float32_image(data_to_detect)
        aligned_method = "identity"
        hdr["VYALGOK"] = (False, "Alignment failed; identity fallback used")
        hdr["VY_ALGN"] = (False, "Not aligned to reference (identity fallback)")
        hdr["VY_REF"] = (ref_fp_name[:60], "Reference frame for alignment")
        hdr["VYALGM"] = (aligned_method[:30], "Alignment method (astroalign or WCS reprojection)")
        _alignment_emit_log(log_sink, f"ERROR: alignment zlyhal pre {fp.name}, použitá identity")
        return {
            "kind": "identity",
            "fp": str(fp.resolve()),
            "frame_index_1based": int(frame_index_1based),
            "is_flipped": bool(is_flipped),
            "hdr": hdr.copy(),
            "aligned_data": np.copy(aligned_data),
            "aligned_method": aligned_method,
            "xy_used": xy_used,
            "fw_i": fw_i,
            "star_count": {
                "file": fp.name,
                "frame_index": int(frame_index_1based),
                "detected_stars": int(len(xy_used)),
                "aligned": False,
                "reason": "identity_fallback",
                "alignment_method": aligned_method,
                "is_flipped": bool(is_flipped),
            },
        }

    strip_celestial_wcs_keys(hdr)
    for k, v in ref_hdr.items():
        if header_key_is_celestial_wcs(k):
            hdr[k] = v
    hdr["VYALGOK"] = (True, "Aligned to reference (passed black frame guard)")
    hdr["VY_ALGN"] = (True, "Aligned to reference")
    hdr["VY_REF"] = (ref_fp_name[:60], "Reference frame for alignment")
    hdr["VYALGM"] = (aligned_method[:30], "Alignment method (astroalign or WCS reprojection)")
    return {
        "kind": "aligned",
        "fp": str(fp.resolve()),
        "frame_index_1based": int(frame_index_1based),
        "is_flipped": bool(is_flipped),
        "hdr": hdr.copy(),
        "aligned_data": np.copy(aligned_data),
        "aligned_method": aligned_method,
        "xy_used": xy_used,
        "fw_i": fw_i,
        "star_count": {
            "file": fp.name,
            "frame_index": int(frame_index_1based),
            "detected_stars": int(len(xy_used)),
            "aligned": True,
            "alignment_method": aligned_method,
            "is_flipped": bool(is_flipped),
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
