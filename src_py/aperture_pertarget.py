"""APERTURE-K-2026-10: one aperture r = k x FWHM_frame for the targets and their comps.

Physics: target, comparison stars and check star share the same enclosed-light
fraction (same k = r / FWHM in each frame), so the fraction cancels in the
differential light curve even when seeing changes.

Grid (C-Munipack "Find best aperture", AIJ / EXOTIC practice): every star of the
comparison pool is measured on k in {1.0, 1.25, 1.5, 1.75, 2.0} x FWHM of each frame
(exact apertures, one sky per star and frame). For each target the curve "comp scatter
vs k" is computed - each comp against the flux sum of the other comps (leave-one-out),
1.4826 x MAD over the night, median over comps. The target itself never takes part (its
variability must not steer the aperture); the check star stays an independent test.

AP-BRIGHT-K-2026-10 (Milan 2026-10-27, replaces the one-k-per-night rule AP-NIGHT-K of 2026-10-24):
k follows the target's brightness. A single target's own comp curve is decided by noise (one comp
more or less changed its k for ~half the targets on 537 / MN Boo), so k is not taken from it but
from the constant field stars of the target's brightness (``field_k_by_brightness``: LOO scatter at
each k, hundreds of stars per magnitude group). One k per night chose 1.5 FWHM on Brno 534 r from
the scatter of the bright comps and made the faint targets 20-60 % noisier. The scatter over the
whole night includes the slow drifts (seeing, centring, pixel phase) that grow at small apertures on
undersampled images - the CCD-equation S/N optimum (~0.67 FWHM, Howell 1989, Naylor 1998) ignores
them; bright stars therefore keep a larger k, faint stars a smaller one (Irwin / MEarth, HATNet: apertures
by magnitude). The Howell (1989)
predicted S/N is kept as a diagnostic.
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
from stats_core import coerce_bool_series
from photometry_defs import DEFAULT_APERTURE_F_GRID, blend_k_column, flux_k_column  # LAYER-2026-10: moved there
# COMP-CEIL-FIELDK-2026-10: the k rule and its bootstrap count live in the lower module (shared with the field k)
from comp_rms_loo import BRIGHT_K_BOOT, k_within_one_se, ks_within_one_se

LOGGER = logging.getLogger(__name__)


# Documented fallback when S/N cannot be evaluated (no finite fluxes / sky).
# Midpoint of the Howell/Naylor sky-limited optimum band ~0.6-0.75 FWHM.
# Replaces the undocumented midpoint-of-grid fallback (1.35 on the old grid)
# that produced the 521 spike of 32 stars at exactly 1.35.
FALLBACK_F_STAR: float = 0.70

# Flat-top tolerance: choose largest f within this fraction of max S/N.
SNR_FLAT_FRAC: float = 0.01

# AP-BRIGHT-K-2026-10 (Milan 2026-10-27): the aperture factor k follows the target's brightness.
# For constant field stars the comparison-free LOO scatter is measured at every k; stars are grouped
# by instrumental magnitude (BRIGHT_K_BIN_MAG bins merged until BRIGHT_K_MIN_STARS stars), and each
# group takes the largest k whose median scatter ratio is within one bootstrap standard error of the
# group minimum (larger k: less sensitive to seeing, centring and pixel phase - AP-NIGHT-K study).
BRIGHT_K_BIN_MAG: float = 0.5
#: A target without a brightness group (no magnitude, or too few constant field stars) keeps the largest
#: allowed k <= BRIGHT_K_NO_GROUP: apertures below 1.0 FWHM are chosen only from measured field scatter
#: (AP-SMALL-K-2026-10; 1.0 was the smallest k of the grid before the grid went down to 0.6).
BRIGHT_K_NO_GROUP: float = 1.0
BRIGHT_K_MIN_STARS: int = 30

MODE_PER_TARGET = "per_target"


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


def howell_snr(
    flux: float,
    sky_pp: float,
    area: float,
    n_sky: float,
    *,
    gain: float,
    read_noise: float,
) -> float:
    """Predicted S/N = F / sqrt(var), var from the CCD equation (``ccd_noise.ccd_variance_adu2``,
    Merline & Howell 1995, median sky; CCD-ONE-2026-10)."""
    from ccd_noise import ccd_variance_adu2  # noqa: PLC0415

    f = float(flux)
    if not math.isfinite(f) or f <= 0:
        return float("nan")
    var = ccd_variance_adu2(
        f, float(sky_pp), float(area), float(n_sky), gain=float(gain), read_noise=float(read_noise)
    )
    if not math.isfinite(var) or var <= 0:
        return float("nan")
    return float(f / math.sqrt(var))


@dataclass
class FrameImages:
    """One frame prepared once for every aperture call (SPEED-2026-10 S2).

    ``filled``: the image with NaN replaced by its NaN-median (the image itself when all pixels are
    finite - ``nan_to_num`` would return an equal copy); ``sky``: NaN off the aligned footprint;
    ``off``: pixels without data - off the aligned footprint or NaN in the frame (G01-29: bad flat
    pixels are NaN after calibration), None when there are none to check. A star whose aperture
    touches an ``off`` pixel is flagged (``edge_fail``), so a filled value never enters its flux.
    """

    img: np.ndarray
    filled: np.ndarray
    sky: np.ndarray
    off: np.ndarray | None


def prepare_frame_images(image: np.ndarray, fwhm_px: float, covered: np.ndarray | None = None) -> FrameImages:
    """``sky``: the frame for annulus medians - NaN off the footprint and on detected sources (SKY-SRC-MASK-2026-10)."""
    from sky_estimation import sky_image  # noqa: PLC0415

    img = np.asarray(image, dtype=float)
    finite = np.isfinite(img)
    off = None if covered is None else ~np.asarray(covered, dtype=bool)
    if bool(finite.all()):
        filled = img
    else:
        filled = np.nan_to_num(img, nan=float(np.nanmedian(img)))
        off = ~finite if off is None else (off | ~finite)
    return FrameImages(img=img, filled=filled, sky=sky_image(img, fwhm_px, covered=covered), off=off)


def measure_flux_on_f_grid(
    image: np.ndarray,
    xy: np.ndarray,
    *,
    f_grid: Sequence[float],
    fwhm_px: float,
    annulus_inner_fwhm: float,
    annulus_outer_fwhm: float,
    covered: np.ndarray | None = None,
    prepared: FrameImages | None = None,
    return_n_sky: bool = False,
) -> tuple[Any, ...]:
    """Sky-subtracted exact aperture fluxes for every k in ``f_grid`` (r = k x FWHM).

    One annulus per star (``sky_estimation.annulus_sky_medians``: plain median of its sky pixels, no
    clipping) placed outside the largest aperture; all radii in a single ``aperture_photometry`` call.

    AP-FOOTPRINT-2026-10: ``covered`` (``frame_footprint.aligned_footprint``) - the sky median uses
    only annulus pixels with data of the aligned frame (not the aligner's constant fill); the third
    return value marks stars whose largest aperture reaches the fill or a missing (NaN) pixel, or whose
    annulus keeps less than half of its pixels (the light curve drops them as ``edge_fail``).

    ``return_n_sky`` (CCD-ONE-2026-10): a fourth return value, the number of sky pixels of each star's
    sky median (the n_sky of the CCD equation, ``ccd_noise``; 0 without a sky).
    """
    from photutils.aperture import CircularAperture  # noqa: PLC0415
    from photutils.aperture import aperture_photometry as _aphot  # noqa: PLC0415

    from aperture_policy import resolve_aperture_geometry  # noqa: PLC0415

    pos = np.asarray(xy, dtype=float)
    n = int(pos.shape[0]) if pos.ndim == 2 else 0
    ks = [float(f) for f in f_grid]
    out_f = {k: np.full(n, np.nan) for k in ks}
    out_s = {k: np.full(n, np.nan) for k in ks}
    edge = np.zeros(n, dtype=bool)
    n_sky = np.zeros(n, dtype=np.int64)

    def _ret() -> tuple[Any, ...]:
        return (out_f, out_s, edge, n_sky) if return_n_sky else (out_f, out_s, edge)

    if n == 0 or not ks:
        return _ret()
    fr = prepared if prepared is not None else prepare_frame_images(image, float(fwhm_px), covered)
    _r, r_in, r_out = resolve_aperture_geometry(
        f=max(ks), fwhm_px=float(fwhm_px),
        annulus_inner_fwhm=float(annulus_inner_fwhm), annulus_outer_fwhm=float(annulus_outer_fwhm),
    )
    ok = np.all(np.isfinite(pos), axis=1)
    if not ok.any():
        return _ret()
    p = pos[ok]
    from frame_footprint import off_footprint  # noqa: PLC0415

    # G06-07: one edge test for every frame - a frame without a transform (not resampled) covers the whole
    # chip, and the chip bounds still count
    _off = fr.off if fr.off is not None else np.zeros(np.asarray(fr.img).shape, dtype=bool)
    edge[ok] = off_footprint(_off, p[:, 0], p[:, 1], max(ks) * float(fwhm_px), r_in, r_out)
    from sky_estimation import annulus_sky_medians_n  # noqa: PLC0415

    sky, n_sky[ok] = annulus_sky_medians_n(fr.sky, p, r_in, r_out)
    aps = [CircularAperture(p, r=max(0.5, k * float(fwhm_px))) for k in ks]
    tbl = _aphot(fr.filled, aps, method="exact")
    for j, (k, ap) in enumerate(zip(ks, aps)):
        sums = np.asarray(tbl[f"aperture_sum_{j}"], dtype=float)
        out_f[k][ok] = sums - sky * float(ap.area)
        out_s[k][ok] = sky
    return _ret()


def allowed_f_from_comp_blend(
    comps: Any, f_grid: Sequence[float], max_frac: float
) -> list[float] | None:
    """k of the grid at which every comp of the target has |neighbour light| <= max_frac.

    ``comps``: the target's rows of comparison_stars_per_target.csv. A comp without a
    value at some k (unknown neighbour light) does not block that k - as in the pool gate.
    None when the per-k columns are absent (older drafts): no restriction.
    """
    import pandas as pd  # noqa: PLC0415

    if comps is None or getattr(comps, "empty", True):
        return None
    cols = [blend_k_column(k) for k in f_grid]
    if not any(c in comps.columns for c in cols):
        return None
    out: list[float] = []
    for k, c in zip(f_grid, cols, strict=False):
        if c not in comps.columns:
            continue
        v = pd.to_numeric(comps[c], errors="coerce").to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0 or bool(np.all(np.abs(v) <= float(max_frac))):
            out.append(float(k))
    return out


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
    f_allowed: list[float] | None = None  # BLEND-K-2026-10: k where all comps <= 1 % neighbour light
    f_own: float = float("nan")  # this target's own comp-curve choice before AP-BRIGHT-K (diagnostic)


@dataclass
class ApertureGridNight:
    """In-memory night grid: frame_key -> catalog_id -> f -> flux / sky_pp."""

    f_grid: list[float]
    fwhm_night_px: float
    # frame_stem -> {cid: {f: flux}}
    frames: dict[str, dict[str, dict[float, float]]] = field(default_factory=dict)
    # frame_stem -> {cid: {f: sky_pp ADU/px}}
    sky_pp: dict[str, dict[str, dict[float, float]]] = field(default_factory=dict)
    # CCD-ONE-2026-10: frame_stem -> {cid: number of sky pixels of the sky median} (one annulus for all k)
    sky_n: dict[str, dict[str, int]] = field(default_factory=dict)
    # frame_stem -> FWHM used for that frame (AUTH-01)
    fwhm_by_frame: dict[str, float] = field(default_factory=dict)
    elapsed_s: float = 0.0
    n_stars: int = 0
    n_frames_measured: int = 0
    n_fwhm_fallback_night: int = 0
    # PSF-LOCAL-2026-10: per-star radius scale (own PSF broader than the field), cid -> s
    scale_by_cid: dict[str, float] = field(default_factory=dict)
    # AP-FOOTPRINT-2026-10: frame_stem -> stars whose aperture / annulus is off the aligned frame's data
    edge_by_frame: dict[str, set[str]] = field(default_factory=dict)
    n_frames_no_transform: int = 0
    # BADPIX-REPAIR-2026-10: frame_stem -> repaired pixels on the reference grid (x, y, is_star_model)
    bp_by_frame: dict[str, Any] = field(default_factory=dict)


def frame_stem_from_source(source_file: str) -> str:
    name = Path(str(source_file)).name
    if name.startswith("proc_") and name.endswith(".csv"):
        return name[5:-4]
    return Path(name).stem


def _fwhm_from_fits_header(hdr: Any, *, night_fallback: float) -> tuple[float, bool]:
    """Return (fwhm_px, used_night_fallback). Prefer VY_FWHM (FWHM-AUTH-01), else the night median; neither
    -> error (no fixed FWHM, Milan 2026-10-29)."""
    from aperture_policy import clamp_fwhm_px, fwhm_from_header_vy_fwhm, header_shape  # noqa: PLC0415

    frame = fwhm_from_header_vy_fwhm(hdr)
    if frame is not None:
        return float(frame), False
    night = clamp_fwhm_px(night_fallback, shape=header_shape(hdr))
    if night is not None:
        return float(night), True
    raise ValueError("aperture grid: frame without a valid VY_FWHM and no night median FWHM")


_GRID_CTX: dict[str, Any] = {}


def _grid_worker_init(ctx: dict[str, Any]) -> None:
    _GRID_CTX.clear()
    _GRID_CTX.update(ctx)


def _grid_measure_frame(fp: Path) -> tuple[Any, ...] | None:
    """Measure one aligned frame on the k grid (SPEED-2026-10 S2: frames are independent).

    Returns ``(stem, fwhm_frame, used_night_fallback, has_transform, per_cid, per_sky, edge_ids, per_nsky)``
    or None when the frame has no usable proc CSV / FITS.
    """
    from astropy.io import fits  # noqa: PLC0415

    from frame_footprint import aligned_footprint, has_alignment_transform  # noqa: PLC0415

    c = _GRID_CTX
    stem = fp.stem
    proc = Path(c["frames_dir"]) / f"proc_{stem}.csv"
    if not proc.is_file():
        return None
    from proc_frame_store import read_proc_csv  # noqa: PLC0415

    # ids as text (one blank id must not turn the column into float64 and drop the frame, G06-04);
    # an unreadable proc file is an error, not a silently missing frame
    df = read_proc_csv(proc, usecols=lambda col: col in {"catalog_id", "x", "y"}, low_memory=False)
    if "catalog_id" not in df.columns:
        return None
    df = df[df["catalog_id"].astype(str).isin(c["want"])]
    if df.empty:
        return None
    ids = df["catalog_id"].astype(str).tolist()
    xy = np.column_stack(
        [
            pd.to_numeric(df["x"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(df["y"], errors="coerce").to_numpy(dtype=float),
        ]
    )
    xy_by_cid = c["xy_by_cid"]
    if xy_by_cid:
        for _j, _c in enumerate(ids):
            _p = xy_by_cid.get(_c)
            if _p is not None and all(math.isfinite(float(v)) for v in _p):
                xy[_j] = (float(_p[0]), float(_p[1]))
    try:
        with fits.open(fp, memmap=True) as hdul:
            img = np.asarray(hdul[0].data, dtype=float)
            fwhm_frame, used_fb = _fwhm_from_fits_header(hdul[0].header, night_fallback=float(c["fwhm_night_px"]))
            hdr_fp = hdul[0].header.copy()
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("[APERTURE-DYNAMIC] FITS open fail %s: %s", fp.name, exc)
        return None
    f_grid = c["f_grid"]
    scale_by_cid = c["scale_by_cid"]
    _sc = np.round(np.array([float(scale_by_cid.get(cc, 1.0)) for cc in ids]), 4)
    fluxes = {float(f): np.full(len(ids), np.nan) for f in f_grid}
    skies = {float(f): np.full(len(ids), np.nan) for f in f_grid}
    edges = np.zeros(len(ids), dtype=bool)
    nsky = np.zeros(len(ids), dtype=np.int64)
    # AP-FOOTPRINT-2026-10: only pixels the aligned frame covers (no transform: not resampled).
    has_tr = has_alignment_transform(hdr_fp)
    covered = aligned_footprint(hdr_fp, img.shape) if has_tr else None
    prepared = prepare_frame_images(img, float(fwhm_frame), covered)
    for _s in sorted(set(np.round(_sc, 4))):
        _m = _sc == _s
        _fl, _sk, _ed, _ns = measure_flux_on_f_grid(
            img,
            xy[_m],
            f_grid=f_grid,
            fwhm_px=float(fwhm_frame) * float(_s),
            annulus_inner_fwhm=float(c["annulus_inner_fwhm"]),
            annulus_outer_fwhm=float(c["annulus_outer_fwhm"]),
            prepared=prepared,
            return_n_sky=True,
        )
        edges[_m] = _ed
        nsky[_m] = _ns
        for f in f_grid:
            fluxes[float(f)][_m] = _fl[float(f)]
            skies[float(f)][_m] = _sk[float(f)]
    per_cid: dict[str, dict[float, float]] = {cid: {} for cid in ids}
    per_sky: dict[str, dict[float, float]] = {cid: {} for cid in ids}
    for f, arr in fluxes.items():
        sky_l = skies[f].tolist()
        for cid, v, s_ in zip(ids, arr.tolist(), sky_l, strict=True):
            per_cid[cid][float(f)] = v
            per_sky[cid][float(f)] = s_
    edge_ids = {cid for cid, e in zip(ids, edges, strict=False) if bool(e)}
    per_nsky = {cid: int(v) for cid, v in zip(ids, nsky.tolist(), strict=True)}
    return stem, float(fwhm_frame), bool(used_fb), bool(has_tr), per_cid, per_sky, edge_ids, per_nsky


def measure_night_grid(
    *,
    frames_dir: Path,
    catalog_ids: Sequence[str],
    f_grid: Sequence[float],
    fwhm_night_px: float,
    annulus_inner_fwhm: float = 2.7,
    annulus_outer_fwhm: float = 5.2,
    max_frames: int | None = None,
    xy_by_cid: Mapping[str, tuple[float, float]] | None = None,
    scale_by_cid: Mapping[str, float] | None = None,
    workers: int | None = None,
) -> ApertureGridNight:
    """Measure targets, comps and check-star candidates on the k grid; r = k x FWHM_frame.

    ``xy_by_cid``: sub-pixel MASTERSTAR positions (the frames are registered to the
    MASTERSTAR grid). When given they replace the per-frame CSV positions, which are
    snapped to integer pixels.

    ``workers`` > 1: frames are measured in a process pool (spawn); results are merged in frame
    order, so the grid is identical to the serial one.
    """
    want = {str(c) for c in catalog_ids if str(c).strip()}
    night = ApertureGridNight(
        f_grid=[float(f) for f in f_grid],
        fwhm_night_px=float(fwhm_night_px),
        n_stars=len(want),
        scale_by_cid={str(k): round(float(v), 4) for k, v in (scale_by_cid or {}).items()},
    )
    if not want or not Path(frames_dir).is_dir():
        return night
    from badpix_sidecar import load_repaired_sidecar  # noqa: PLC0415

    night.bp_by_frame = load_repaired_sidecar(frames_dir)
    t0 = time.perf_counter()
    from fits_suffixes import list_fits_files  # noqa: PLC0415

    fits_files = list_fits_files(frames_dir)
    if max_frames is not None:
        fits_files = fits_files[: int(max_frames)]
    fits_files = [fp for fp in fits_files if fp.stem.upper() != "MASTERSTAR"]
    ctx = {
        "frames_dir": str(frames_dir),
        "want": want,
        "f_grid": night.f_grid,
        "fwhm_night_px": float(fwhm_night_px),
        "annulus_inner_fwhm": float(annulus_inner_fwhm),
        "annulus_outer_fwhm": float(annulus_outer_fwhm),
        "xy_by_cid": {str(k): (float(v[0]), float(v[1])) for k, v in (xy_by_cid or {}).items()},
        "scale_by_cid": night.scale_by_cid,
    }
    from parallel_frames import frame_bytes_of_fits, map_frames, stage_workers  # noqa: PLC0415

    if workers is None:
        nw = stage_workers(len(fits_files), frame_bytes_of_fits(fits_files[0]), stage="aperture grid") if fits_files else 1
    else:
        nw = max(1, min(int(workers), len(fits_files)))
    results = list(map_frames(_grid_measure_frame, fits_files, workers=nw, initializer=_grid_worker_init, ctx=ctx,
                              stage="aperture grid"))
    for res in results:
        if res is None:
            continue
        stem, fwhm_frame, used_fb, has_tr, per_cid, per_sky, edge_ids, per_nsky = res
        if used_fb:
            night.n_fwhm_fallback_night += 1
            LOGGER.info(
                "[APERTURE-DYNAMIC] frame %s: no VY_FWHM - using night median %.4f px",
                stem,
                float(fwhm_frame),
            )
        night.fwhm_by_frame[stem] = float(fwhm_frame)
        if not has_tr:
            night.n_frames_no_transform += 1
            LOGGER.warning("[AP-FOOTPRINT] frame %s: no VY_AM transform - aligned footprint not checked", stem)
        night.frames[stem] = per_cid
        night.sky_pp[stem] = per_sky
        night.sky_n[stem] = per_nsky
        night.edge_by_frame[stem] = edge_ids
        night.n_frames_measured += 1
    night.elapsed_s = float(time.perf_counter() - t0)
    LOGGER.info(
        "[APERTURE-DYNAMIC] grid measured: n_frames=%d n_stars=%d n_f=%d "
        "fwhm_fallback=%d workers=%d elapsed=%.1fs",
        night.n_frames_measured,
        night.n_stars,
        len(night.f_grid),
        night.n_fwhm_fallback_night,
        nw,
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


def sky_n_series_from_grid(
    night: ApertureGridNight,
    catalog_id: str,
    frame_order: Sequence[str],
) -> np.ndarray:
    """Number of sky pixels of ``catalog_id``'s sky median per frame (CCD-ONE-2026-10); NaN when not measured."""
    out = np.full(len(frame_order), float("nan"), dtype=float)
    cid = str(catalog_id)
    for i, stem in enumerate(frame_order):
        v = night.sky_n.get(str(stem), {}).get(cid)
        if v is not None and int(v) > 0:
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


def comp_loo_scatter(comp_flux: Mapping[str, np.ndarray]) -> float:
    """Median over comps of the 1.4826 MAD scatter of m_comp - m(sum of the other comps).

    Leave-one-out: a comp is never compared with itself. Frames where any star is missing
    are skipped for that comparison. NaN when fewer than 2 comps or 5 usable frames.
    """
    ids = [c for c, v in comp_flux.items() if np.isfinite(np.asarray(v, dtype=float)).sum() >= 5]
    if len(ids) < 2:
        return float("nan")
    F = np.vstack([np.asarray(comp_flux[c], dtype=float) for c in ids])
    F = np.where(np.isfinite(F) & (F > 0), F, np.nan)
    out: list[float] = []
    for j in range(F.shape[0]):
        rest = np.delete(F, j, axis=0)
        good = np.isfinite(F[j]) & np.all(np.isfinite(rest), axis=0)
        if int(good.sum()) < 5:
            continue
        d = -2.5 * np.log10(F[j, good]) + 2.5 * np.log10(np.sum(rest[:, good], axis=0))
        out.append(1.4826 * float(np.median(np.abs(d - np.median(d)))))
    return float(np.median(out)) if out else float("nan")


def choose_f_for_target(
    night: ApertureGridNight,
    *,
    target_cid: str,
    comp_ids: Sequence[str],
    frame_order: Sequence[str],
    gain: float,
    read_noise: float,
    allowed_f: Sequence[float] | None = None,
) -> PerTargetChoice:
    """k* = argmin over the grid of the comparison-star leave-one-out scatter.

    ``allowed_f`` (BLEND-K-2026-10): only these k may be chosen - those at which every
    comp has <= 1 % neighbour light; None = the whole grid.

    Ties within 1 % of the minimum -> the larger k (less sensitive to PSF changes).
    Without a measurable comp scatter (fewer than 2 comps) the Howell S/N maximum of the
    target is used. Predicted S/N per k is reported as a diagnostic.
    """
    snr_by_f: dict[float, float] = {}
    growth: dict[float, float] = {}
    sky_med: dict[float, float] = {}
    scat_by_f: dict[float, float] = {}
    fwhm = float(night.fwhm_night_px) if math.isfinite(float(night.fwhm_night_px)) else 5.0
    if fwhm <= 0:
        fwhm = 5.0
    # PSF-LOCAL-2026-10: the target's aperture is k x FWHM x own-PSF scale
    fwhm *= float(night.scale_by_cid.get(str(target_cid), 1.0))
    n_sky_med = _night_median_positive(sky_n_series_from_grid(night, target_cid, frame_order))
    for f in night.f_grid:
        ff = float(f)
        t_flux = series_from_grid(night, target_cid, ff, frame_order)
        t_sky = sky_series_from_grid(night, target_cid, ff, frame_order)
        F_med = _night_median_positive(t_flux)
        sky_pp = _night_median_positive(t_sky)
        growth[ff] = F_med
        sky_med[ff] = sky_pp
        r_ap = max(0.5, ff * fwhm)
        snr_by_f[ff] = howell_snr(
            F_med, sky_pp, math.pi * r_ap * r_ap, n_sky_med,
            gain=float(gain), read_noise=float(read_noise),
        )
        scat_by_f[ff] = comp_loo_scatter(
            {str(c): series_from_grid(night, str(c), ff, frame_order) for c in comp_ids}
        )
    _allowed = None if allowed_f is None else {round(float(k), 6) for k in allowed_f}
    finite = {k: v for k, v in scat_by_f.items() if math.isfinite(v) and (_allowed is None or round(k, 6) in _allowed)}
    if finite:
        best = min(finite.values())
        f_star = max(k for k, v in finite.items() if v <= best * (1.0 + SNR_FLAT_FRAC))
        reason = "comp_loo_scatter_min"
    else:
        f_star = pick_f_star_snr(
            snr_by_f if _allowed is None else {k: v for k, v in snr_by_f.items() if round(k, 6) in _allowed}
        )
        reason = "howell_snr_argmax_no_comp_scatter"
        if f_star is None:
            f_star = float(FALLBACK_F_STAR)
            reason = "fallback_no_finite_scatter_or_snr"
    # edge of the grid the choice was made on (the allowed k when restricted)
    edge = f_star_is_grid_edge(
        float(f_star), sorted(_allowed) if _allowed else night.f_grid
    )
    snr_max = float(snr_by_f.get(float(f_star), float("nan")))
    return PerTargetChoice(
        catalog_id=str(target_cid),
        f_star=float(f_star),
        r_ap_px=float(f_star) * float(fwhm),
        p2p_by_f={f"{k:.4g}": float(v) for k, v in sorted(scat_by_f.items())},
        snr_by_f={f"{k:.4g}": float(v) for k, v in sorted(snr_by_f.items())},
        n_comps=len(list(comp_ids)),
        n_frames=len(list(frame_order)),
        f_edge=bool(edge),
        reason=str(reason),
        snr_max=float(snr_max) if math.isfinite(snr_max) else float("nan"),
        growth_F_by_f={f"{k:.4g}": float(v) for k, v in sorted(growth.items())},
        sky_pp_by_f={f"{k:.4g}": float(v) for k, v in sorted(sky_med.items())},
        f_allowed=None if allowed_f is None else sorted(float(k) for k in allowed_f),
    )


@dataclass
class BrightnessK:
    """Aperture factor by instrumental magnitude (AP-BRIGHT-K-2026-10)."""

    mag_lo: list[float]
    mag_hi: list[float]
    k: list[float]
    n_stars: list[int]
    curves: list[dict[str, float]]
    se: list[float]
    #: AP-BRIGHT-K diagnostic (Milan 2026-10-30): per group every k within one SE of the curve minimum
    #: (``comp_rms_loo.ks_within_one_se``); empty for groups read from a file written before it
    k_within_se: list[list[float]] = field(default_factory=list)

    def k_for(self, mag: float) -> float | None:
        """k of the group whose magnitude range holds ``mag`` (outside the range: the end group)."""
        if not self.k or not math.isfinite(float(mag)):
            return None
        for lo, hi, k in zip(self.mag_lo, self.mag_hi, self.k, strict=False):
            if lo <= float(mag) < hi:
                return float(k)
        return float(self.k[0]) if float(mag) < self.mag_lo[0] else float(self.k[-1])

    def to_dict(self) -> dict[str, Any]:
        groups = [
            {"mag_lo": round(lo, 3), "mag_hi": round(hi, 3), "k": k, "n_stars": n, "se": round(se, 4),
             "curve": {kk: round(v, 4) for kk, v in c.items()}}
            for lo, hi, k, n, c, se in zip(self.mag_lo, self.mag_hi, self.k, self.n_stars, self.curves, self.se,
                                          strict=False)
        ]
        if len(self.k_within_se) == len(groups):
            for g, kw in zip(groups, self.k_within_se, strict=True):
                g["k_within_se"] = [float(v) for v in kw]
        return {"groups": groups}


def field_k_by_brightness(
    proc_csv_paths: Sequence[Path],
    f_grid: Sequence[float],
    *,
    frames: Mapping[str, pd.DataFrame] | None = None,
) -> BrightnessK | None:
    """AP-BRIGHT-K-2026-10: k per instrumental-magnitude group from constant field stars.

    Constant field stars: never a known variable (VSX / Gaia DR3 variability) or saturated on any frame.
    Scatter at each k: the LOO magnitude RMS of ``comp_rms_loo`` (each star on its night level against the
    median of all other stars in the frame; MAD sigma, no clipping) on the proc columns ``flux_kNNN``.
    Magnitude: -2.5 log10 of the star's night median flux at the smallest k. None without enough stars.
    """
    from comp_rms_loo import loo_mag_rms_maps  # noqa: PLC0415
    from proc_frame_store import read_proc_csv  # noqa: PLC0415

    ks = sorted(float(k) for k in f_grid)
    cols = [flux_k_column(k) for k in ks]
    cache: dict[str, pd.DataFrame] = {}
    bad: set[str] = set()
    seen: set[str] = set()
    for p in proc_csv_paths:
        # G07-31: the frames come from the ProcFrameStore (PROC_STORE_COLS hold the grid and the flags);
        # a file only on disk is read
        df = frames.get(str(p)) if frames is not None else None
        if df is None:
            try:
                df = read_proc_csv(p, low_memory=False, dtype={"catalog_id": str})
            except (OSError, ValueError) as exc:
                LOGGER.warning("[APERTURE-BRIGHT-K] skip unreadable %s: %s", Path(p).name, exc)
                continue
        if "catalog_id" not in df.columns or not all(c in df.columns for c in cols):
            continue
        cid = df["catalog_id"].astype(str).str.strip()
        flag = np.zeros(len(df), dtype=bool)
        for f in ("vsx_known_variable", "gaia_dr3_variable_catalog", "likely_saturated", "is_saturated"):
            if f in df.columns:
                flag |= coerce_bool_series(df[f]).to_numpy()
        bad.update(cid[flag].tolist())
        seen.update(cid.tolist())
        cache[str(p)] = pd.DataFrame({"catalog_id": cid, **{c: pd.to_numeric(df[c], errors="coerce") for c in cols}})
    paths = [Path(p) for p in cache]
    ids = {c for c in seen - bad if c}
    if len(ids) < BRIGHT_K_MIN_STARS or not paths:
        return None
    _by_col = loo_mag_rms_maps(ids, paths, cache, cols, strict_flux_col=True)
    rms = {k: _by_col[c] for k, c in zip(ks, cols, strict=True)}
    common = sorted(set.intersection(*(set(m) for m in rms.values())))
    if len(common) < BRIGHT_K_MIN_STARS:
        return None
    allf = pd.concat([d[["catalog_id", cols[0]]] for d in cache.values()], ignore_index=True)
    med_f = allf[allf[cols[0]] > 0].groupby("catalog_id")[cols[0]].median()
    mag = {c: float(-2.5 * np.log10(med_f[c])) for c in common if c in med_f.index}
    stars = sorted((m, c) for c, m in mag.items() if math.isfinite(m))
    ratios = []
    for _m, c in stars:
        v = np.array([rms[k][c] for k in ks], dtype=float)
        mn = float(np.min(v))
        ratios.append(v / mn if mn > 0 else np.full(len(ks), np.nan))
    R = np.array(ratios)
    M = np.array([m for m, _c in stars])
    # groups: BRIGHT_K_BIN_MAG bins from bright to faint, merged until BRIGHT_K_MIN_STARS stars
    b = np.floor((M - M.min()) / BRIGHT_K_BIN_MAG).astype(int)
    groups: list[np.ndarray] = []
    cur: list[np.ndarray] = []
    for v in np.unique(b):
        cur.append(np.flatnonzero(b == v))
        if sum(len(x) for x in cur) >= BRIGHT_K_MIN_STARS:
            groups.append(np.concatenate(cur))
            cur = []
    if cur:
        if not groups:
            return None
        groups[-1] = np.concatenate([groups[-1]] + cur)
    rng = np.random.default_rng(0)
    out = BrightnessK([], [], [], [], [], [])
    for g in groups:
        r = R[g]
        r = r[np.all(np.isfinite(r), axis=1)]
        if len(r) < BRIGHT_K_MIN_STARS:
            continue
        k_g, curve, se = k_within_one_se(ks, r, rng)
        out.k_within_se.append([float(v) for v in ks_within_one_se(ks, curve, se)])
        out.mag_lo.append(float(M[g].min()))
        out.mag_hi.append(float(M[g].max()))
        out.k.append(float(k_g))
        out.n_stars.append(int(len(r)))
        out.curves.append({f"{k:.4g}": float(v) for k, v in zip(ks, curve, strict=False)})
        out.se.append(se)
    if not out.k:
        return None
    # contiguous ranges: each group runs up to the next group's first magnitude
    for i in range(len(out.k) - 1):
        out.mag_hi[i] = out.mag_lo[i + 1]
    out.mag_hi[-1] = float("inf")
    out.mag_lo[0] = float("-inf")
    return out


def target_inst_mag(night: "ApertureGridNight", cid: str) -> float:
    """-2.5 log10 of the target's night median flux at the smallest k of the grid (NaN if none)."""
    k0 = min(float(k) for k in night.f_grid)
    v = [fr[cid][k0] for fr in night.frames.values() if cid in fr and k0 in fr[cid]]
    v = [x for x in v if math.isfinite(float(x)) and float(x) > 0]
    return float(-2.5 * np.log10(np.median(v))) if v else float("nan")


def apply_brightness_k(
    choices: Mapping[str, PerTargetChoice],
    f_grid: Sequence[float],
    target_mag: Mapping[str, float],
    bk: BrightnessK | None,
) -> dict[str, Any]:
    """AP-BRIGHT-K-2026-10: each target gets the k of its brightness group (changes ``choices`` in place).

    Blend limit: the largest k the target may use that is <= its group k, else its smallest allowed k.
    A target without a brightness group gets the largest allowed k <= BRIGHT_K_NO_GROUP. Target, its comps and
    its check star are measured with that k (one enclosed-light fraction per light curve).
    """
    grid = sorted(float(k) for k in f_grid)
    info: dict[str, Any] = {"brightness_k": bk.to_dict() if bk else None, "k_by_target": {}}
    for cid, c in choices.items():
        al = sorted(float(k) for k in (c.f_allowed if c.f_allowed else grid))
        m = float(target_mag.get(cid, float("nan")))
        k_want = bk.k_for(m) if bk else None
        if k_want is None:
            below = [k for k in al if k <= BRIGHT_K_NO_GROUP + 1e-9]
            k_t, why = (max(below) if below else min(al)), f"no brightness group - k {BRIGHT_K_NO_GROUP:g}"
        else:
            below = [k for k in al if k <= k_want + 1e-9]
            k_t = max(below) if below else min(al)
            why = "brightness_k" + ("" if k_t == k_want else "; blend limit below the group k")
        c.f_own = float(c.f_star)
        scale = c.r_ap_px / c.f_star if c.f_star > 0 else float("nan")  # FWHM x PSF-LOCAL scale
        c.f_star = float(k_t)
        c.r_ap_px = float(k_t) * scale
        c.f_edge = f_star_is_grid_edge(float(k_t), al)
        c.snr_max = float(c.snr_by_f.get(f"{k_t:.4g}", float("nan")))
        c.reason = why
        info["k_by_target"][cid] = {"inst_mag": round(m, 3) if math.isfinite(m) else None, "k": float(k_t)}
    return info


def apply_grid_fluxes_to_frames(
    all_frames: pd.DataFrame,
    night: ApertureGridNight,
    *,
    catalog_ids: Sequence[str],
    f_star: float,
    gain: float,
    read_noise: float,
) -> pd.DataFrame:
    """Set mag_inst / aperture_r_px (r = k* x FWHM_frame) / err for target, comps and check star.

    A star without a grid flux in a frame gets mag_inst = NaN (never a different aperture
    in the same frame). ``err`` (relative) is the CCD equation at the k* aperture with the grid's
    sky and the real number of sky pixels (``ccd_noise.ccd_rel_error``, Merline & Howell 1995, median
    sky; CCD-ONE-2026-10). G08-28: a row whose grid flux or grid sky is missing gets err = NaN with
    its mag_inst = NaN - it never keeps the per-frame proc error, which belongs to another aperture
    (and another error model); a grid sky is missing only together with the grid flux
    (flux = sum - sky x area), so there is no flux to give an error to. A star whose aperture or
    annulus is off the aligned frame's data in that frame (``night.edge_by_frame``) gets
    mag_inst = NaN, err = NaN and flag ``edge_fail``. Gain and read noise are required (ValueError).
    """
    from ccd_noise import check_gain_read_noise, ccd_rel_error  # noqa: PLC0415

    g, rn = check_gain_read_noise(gain, read_noise)
    if all_frames is None or all_frames.empty or "source_file" not in all_frames.columns:
        return all_frames
    out = all_frames.copy()
    want = {str(c) for c in catalog_ids}
    k = float(f_star)
    # SPEED-2026-10: visit only the rows of the wanted stars (column lists, same order as iterrows).
    _n = len(out)
    _cid_l = [str(v) for v in out["catalog_id"].tolist()] if "catalog_id" in out.columns else [""] * _n
    _src_l = [str(v) for v in out["source_file"].tolist()]
    _labels = out.index.tolist()
    from badpix_sidecar import repaired_in_aperture  # noqa: PLC0415

    _x_l = out["x"].to_numpy(float) if "x" in out.columns else np.full(_n, np.nan)
    _y_l = out["y"].to_numpy(float) if "y" in out.columns else np.full(_n, np.nan)
    if "bp_repaired_px" not in out.columns:
        out["bp_repaired_px"] = 0
    for _p in [j for j, c in enumerate(_cid_l) if c in want]:
        i = _labels[_p]
        cid = _cid_l[_p]
        stem = frame_stem_from_source(_src_l[_p])
        ent = night.frames.get(stem, {}).get(cid) or {}
        flux = ent.get(k)
        if flux is None:
            for kk, val in ent.items():
                if abs(float(kk) - k) < 1e-9:
                    flux = val
                    break
        fwhm = float(night.fwhm_by_frame.get(stem, night.fwhm_night_px)) * float(night.scale_by_cid.get(cid, 1.0))
        r_ap = k * fwhm
        out.at[i, "aperture_r_px"] = float(r_ap)
        out.at[i, "aperture_f"] = k
        # BADPIX-REPAIR-2026-10: repaired (model-filled) pixels inside this aperture - a note, not a flag
        out.at[i, "bp_repaired_px"] = repaired_in_aperture(night.bp_by_frame.get(stem), _x_l[_p], _y_l[_p], r_ap)
        out.at[i, "err"] = float("nan")
        if cid in night.edge_by_frame.get(stem, set()):
            out.at[i, "mag_inst"] = float("nan")
            out.at[i, "flag"] = "edge_fail"
            out.at[i, "edge_fail"] = True
            continue
        if flux is None or not math.isfinite(float(flux)) or float(flux) <= 0:
            out.at[i, "mag_inst"] = float("nan")
            continue
        sky = (night.sky_pp.get(stem, {}).get(cid) or {}).get(k, float("nan"))
        n_sky = night.sky_n.get(stem, {}).get(cid, 0)
        if sky is None or not math.isfinite(float(sky)) or not n_sky:
            out.at[i, "mag_inst"] = float("nan")
            out.at[i, "flag"] = "no_data"
            continue
        out.at[i, "mag_inst"] = float(-2.5 * math.log10(float(flux)))
        out.at[i, "err"] = float(
            ccd_rel_error(float(flux), float(sky), math.pi * r_ap * r_ap, float(n_sky), gain=g, read_noise=rn)
        )
    return out


def load_photometric_gain(photometry_dir: Path | str) -> tuple[float, float]:
    """(gain e-/ADU container, read noise e-) the run's aperture photometry used (GAIN-ONE-2026-10; written to
    aperture_per_target.json); NaN when the file or a value is missing."""
    try:
        d = json.loads((Path(photometry_dir) / "aperture_per_target.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return float("nan"), float("nan")

    def _f(v: Any) -> float:
        try:
            x = float(v)
        except (TypeError, ValueError):
            return float("nan")
        return x if math.isfinite(x) else float("nan")

    return _f(d.get("gain")), _f(d.get("read_noise"))


def load_field_k(photometry_dir: Path | str) -> tuple[BrightnessK | None, str]:
    """C1-2026-10 (G10-07): the aperture k the light curves use, for the field searches.

    Returns ``(brightness groups, status)``: the AP-BRIGHT-K groups of ``aperture_per_target.json``
    (``None`` with status ``"no_group"`` when the run had too few constant field stars - every star then
    takes BRIGHT_K_NO_GROUP, as the targets do). Status ``"missing"`` when the file is absent or unreadable
    (aperture photometry did not run) - the caller stops instead of guessing a k.
    """
    try:
        d = json.loads((Path(photometry_dir) / "aperture_per_target.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, "missing"
    groups = ((d.get("brightness_k") or {}).get("brightness_k") or {}).get("groups") or []
    if not groups:
        return None, "no_group"
    bk = BrightnessK([], [], [], [], [], [])
    for g in groups:
        bk.mag_lo.append(float(g["mag_lo"]))
        bk.mag_hi.append(float(g["mag_hi"]))
        bk.k.append(float(g["k"]))
        bk.n_stars.append(int(g.get("n_stars") or 0))
        bk.curves.append(dict(g.get("curve") or {}))
        bk.se.append(float(g.get("se") or float("nan")))
        if "k_within_se" in g:
            bk.k_within_se.append([float(v) for v in g["k_within_se"]])
    return bk, "brightness_k"


def field_k_plan(photometry_dir: Path, f_grid: Any) -> dict[str, Any] | None:
    """C1-2026-10 (Milan 2026-10-29, option a): the aperture k of every star of the field search.

    The AP-BRIGHT-K groups of the run (aperture_per_target.json): a star takes the k of its
    instrumental-magnitude group (magnitude at the smallest grid k, as the groups were built);
    k_ref = the largest group k (the scale all magnitudes are corrected to); transients are measured
    at the k of the faintest group (new sources are near the detection limit). A run without groups
    (too few constant field stars) uses BRIGHT_K_NO_GROUP for every star, as its targets do.
    None when aperture_per_target.json is missing - no k is guessed.
    """
    bk, status = load_field_k(photometry_dir)
    if status == "missing":
        return None
    grid = sorted(float(v) for v in f_grid)
    k_default = max(k for k in grid if k <= BRIGHT_K_NO_GROUP + 1e-9)
    ks = sorted({float(k) for k in bk.k}) if bk is not None else [k_default]
    for kk in ks:
        if not any(abs(kk - g) < 1e-9 for g in grid):
            raise ValueError(f"brightness-group k {kk} is not on the aperture grid {grid}")
    return {
        "rule": "brightness_k" if bk is not None else "no_group",
        "groups": bk,
        "k_default": k_default,
        "k0": grid[0],
        "k_ref": max(ks),
        "k_transient": float(bk.k[-1]) if bk is not None else k_default,
        "k_all": sorted(set(ks) | {grid[0], k_default}),
    }


def group_k_fluxes(D: pd.DataFrame, base: pd.Series, kplan: Mapping[str, Any]) -> dict[str, Any]:
    """C1-2026-10 / G06-18: flux of every row of a night table on the aperture k of its star's brightness group,
    and the per-frame aperture correction to k_ref (one rule for the field search and the colour-term fit).

    ``D``: one row per star and frame (``catalog_id``, frame index ``t``, the grid columns ``flux_kNNN`` of
    ``kplan["k_all"]``); ``base``: rows usable at all (not saturated, photometry ok). A star's group is that of
    its night-median instrumental magnitude at the smallest grid k (as the AP-BRIGHT-K groups are built).
    Aperture correction m(k_ref) - m(k): median per frame over the stars of the k_ref group (bright, high S/N;
    Stetson 1990). Returns {k_star, k_row, flux, apcorr (k -> Series by t), apcorr_row}.
    """
    k0, k_ref = kplan["k0"], kplan["k_ref"]
    bk, k_default = kplan["groups"], kplan["k_default"]
    fk = {kk: pd.to_numeric(D[flux_k_column(kk)], errors="coerce") for kk in kplan["k_all"]}
    f0 = fk[k0].where(base & (fk[k0] > 0))
    m0 = -2.5 * np.log10(f0.groupby(D["catalog_id"]).median())
    k_star = {c: (bk.k_for(float(m)) if bk is not None else None) or k_default for c, m in m0.items()}
    k_row = D["catalog_id"].map(k_star).fillna(k_default).astype(float)
    flux = pd.Series(np.nan, index=D.index)
    for kk in kplan["k_all"]:
        sel = k_row == kk
        flux[sel] = fk[kk][sel]
    apc: dict[float, pd.Series] = {}
    in_ref = base & (k_row == k_ref) & (fk[k_ref] > 0)
    for kk in kplan["k_all"]:
        if kk == k_ref or not (k_row == kk).any():
            continue
        ok = in_ref & (fk[kk] > 0)
        apc[kk] = (-2.5 * np.log10(fk[k_ref][ok] / fk[kk][ok])).groupby(D.loc[ok, "t"]).median()
    apcorr_row = pd.Series(0.0, index=D.index)
    for kk, ser in apc.items():
        sel = k_row == kk
        apcorr_row[sel] = D.loc[sel, "t"].map(ser).to_numpy(float)
    return {"k_star": k_star, "k_row": k_row, "flux": flux, "apcorr": apc, "apcorr_row": apcorr_row}


def write_per_target_choices(
    path: Path,
    choices: Mapping[str, PerTargetChoice],
    *,
    f_grid: Sequence[float],
    fwhm_night_px: float,
    elapsed_s: float,
    gain: float | None = None,
    read_noise: float | None = None,
    brightness_k: Mapping[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_edge = sum(1 for c in choices.values() if bool(c.f_edge))
    payload = {
        "policy": "APERTURE-K-2026-10",
        "mode": MODE_PER_TARGET,
        "criterion": "brightness_k_field_scatter",
        "brightness_k": dict(brightness_k) if brightness_k else None,
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
                "f_allowed_neighbour_light": c.f_allowed,
                "f_own_curve_min": c.f_own if math.isfinite(float(c.f_own)) else None,
            }
            for cid, c in choices.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


__all__ = [
    "DEFAULT_APERTURE_F_GRID",
    "FALLBACK_F_STAR",
    "MODE_PER_TARGET",
    "BRIGHT_K_BIN_MAG",
    "BRIGHT_K_BOOT",
    "BRIGHT_K_MIN_STARS",
    "k_within_one_se",
    "BrightnessK",
    "load_field_k",
    "field_k_plan",
    "group_k_fluxes",
    "SNR_FLAT_FRAC",
    "ApertureGridNight",
    "PerTargetChoice",
    "apply_grid_fluxes_to_frames",
    "choose_f_for_target",
    "allowed_f_from_comp_blend",
    "apply_brightness_k",
    "field_k_by_brightness",
    "target_inst_mag",
    "blend_k_column",
    "flux_k_column",
    "f_star_is_grid_edge",
    "frame_stem_from_source",
    "howell_snr",
    "measure_flux_on_f_grid",
    "measure_night_grid",
    "pick_f_star_snr",
    "series_from_grid",
    "sky_series_from_grid",
    "sky_n_series_from_grid",
    "write_per_target_choices",
]


#: PSF-LOCAL-2026-10: a target whose own PSF (moment FWHM on MASTERSTAR) is broader than the
#: field median by more than this ratio gets r = k x FWHM_frame x ratio (Howell 1989: the
#: optimal radius scales with the star's own profile). Capped at PSF_SCALE_MAX.
PSF_BROAD_RATIO = 1.25
PSF_SCALE_MAX = 2.5


def moment_fwhm(img: np.ndarray, x: float, y: float, fwhm_ref: float) -> float:
    """Second-moment FWHM (2.355 sigma) of the background-subtracted profile within 2 FWHM_ref;
    background = median of 3-4 FWHM_ref. NaN when not measurable."""
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(fwhm_ref) and fwhm_ref > 0):
        return float("nan")
    R = int(math.ceil(4.0 * fwhm_ref))
    xi, yi = int(round(x)), int(round(y))
    y0, y1, x0, x1 = max(0, yi - R), min(img.shape[0], yi + R + 1), max(0, xi - R), min(img.shape[1], xi + R + 1)
    c = np.asarray(img[y0:y1, x0:x1], dtype=float)
    if c.size == 0:
        return float("nan")
    yy, xx = np.mgrid[y0:y1, x0:x1]
    r = np.hypot(xx - x, yy - y)
    ann = c[(r > 3.0 * fwhm_ref) & (r < 4.0 * fwhm_ref) & np.isfinite(c)]
    if ann.size < 10:
        return float("nan")
    I = c - float(np.median(ann))
    m = (r <= 2.0 * fwhm_ref) & np.isfinite(I) & (I > 0)
    if int(m.sum()) < 5 or float(I[m].sum()) <= 0:
        return float("nan")
    s2 = float((I[m] * r[m] ** 2).sum() / (2.0 * I[m].sum()))
    return 2.3548 * math.sqrt(s2)


#: PSF-LOCAL magnitude matching: reference stars within +-PSF_REF_DMAG of the target (widened
#: once to 2x when fewer than PSF_REF_MIN_N); the moment FWHM grows with magnitude (noise
#: wings: draft 526, isolated constant stars ratio 0.95 at G 9 -> 1.37 at G 14), so a fixed
#: bright reference would scale every faint target.
PSF_REF_DMAG = 0.5
PSF_REF_MIN_N = 10


def psf_scale_for_targets(
    masterstar_fits: Path,
    target_xy: Mapping[str, tuple[float, float]],
    reference_xym: Sequence[tuple[float, float, float]],
    fwhm_ref: float,
    *,
    target_mag: Mapping[str, float] | None = None,
    exclude: set[str] | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """(scale_by_cid, ratio_by_cid). ratio = target moment FWHM / median moment FWHM of the
    isolated constant reference stars of the same magnitude (same method and same S/N, so the
    method's bias cancels); scale = min(ratio, PSF_SCALE_MAX) when ratio > PSF_BROAD_RATIO.
    ``reference_xym``: (x, y, mag); ``exclude``: blended targets (their profile includes the
    neighbour). A target without >= PSF_REF_MIN_N references of its magnitude is not judged."""
    from astropy.io import fits  # noqa: PLC0415

    img = np.asarray(fits.getdata(masterstar_fits), dtype=float)
    ref_v: list[float] = []
    ref_m: list[float] = []
    for x, y, m in reference_xym:
        v = moment_fwhm(img, float(x), float(y), fwhm_ref)
        if math.isfinite(v) and math.isfinite(float(m)):
            ref_v.append(v)
            ref_m.append(float(m))
    rv, rm = np.asarray(ref_v), np.asarray(ref_m)
    ratio: dict[str, float] = {}
    scale: dict[str, float] = {}
    for cid, (x, y) in target_xy.items():
        tm = float((target_mag or {}).get(str(cid), float("nan")))
        if not math.isfinite(tm) or rv.size == 0:
            continue
        sel = np.abs(rm - tm) <= PSF_REF_DMAG
        if int(sel.sum()) < PSF_REF_MIN_N:
            sel = np.abs(rm - tm) <= 2.0 * PSF_REF_DMAG
        if int(sel.sum()) < PSF_REF_MIN_N:
            continue
        med = float(np.median(rv[sel]))
        v = moment_fwhm(img, float(x), float(y), fwhm_ref)
        if not math.isfinite(v) or med <= 0:
            continue
        ratio[str(cid)] = v / med
        if v / med > PSF_BROAD_RATIO and str(cid) not in (exclude or set()):
            scale[str(cid)] = float(min(v / med, PSF_SCALE_MAX))
    return scale, ratio

