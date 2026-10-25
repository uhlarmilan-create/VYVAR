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

AP-NIGHT-K-2026-10 (Milan 2026-10-24): ONE k per night and observation group, from the
median over targets of the normalised curves (``apply_night_k``). A single target's curve
is decided by noise (one comp more or less changed its k for ~half the targets on 537 /
MN Boo), the night median is not. The comp scatter over the whole night includes the slow
drifts (seeing, centring, pixel phase) that grow at small apertures on undersampled
images - the CCD-equation S/N optimum (~0.67 FWHM, Howell 1989, Naylor 1998) ignores them
and was rejected (COMP-SELECT-2026-10, AP-NIGHT-K). One factor x FWHM for every star is the
common practice (AAVSO VPhot 1.5, AstroImageJ 1.7, HiPERCAM 1.5-2.5). The Howell (1989)
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

LOGGER = logging.getLogger(__name__)

# APERTURE-K-2026-10: small grid of apertures in units of the frame FWHM.
DEFAULT_APERTURE_F_GRID: tuple[float, ...] = (1.0, 1.25, 1.5, 1.75, 2.0)

# Documented fallback when S/N cannot be evaluated (no finite fluxes / sky).
# Midpoint of the Howell/Naylor sky-limited optimum band ~0.6-0.75 FWHM.
# Replaces the undocumented midpoint-of-grid fallback (1.35 on the old grid)
# that produced the 521 spike of 32 stars at exactly 1.35.
FALLBACK_F_STAR: float = 0.70

# Flat-top tolerance: choose largest f within this fraction of max S/N.
SNR_FLAT_FRAC: float = 0.01

# AP-NIGHT-K-2026-10: k within this fraction of the night curve's minimum count as equal and
# the larger k is taken (less sensitive to seeing / centring). 0.03 = the bootstrap standard
# error (over targets) of the night median curve measured on 537 and MN Boo 538 (0.02-0.03).
NIGHT_K_TIE_FRAC: float = 0.03
# a k enters the night curve only when this fraction of the targets may use it (blend limit)
NIGHT_K_MIN_TARGET_FRAC: float = 0.75

MODE_PER_TARGET = "per_target"


def flux_to_mag(flux: np.ndarray | float) -> np.ndarray:
    f = np.asarray(flux, dtype=float)
    out = np.full(f.shape, float("nan"), dtype=float)
    ok = np.isfinite(f) & (f > 0)
    out[ok] = -2.5 * np.log10(f[ok])
    return out


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
    """Total variance [ADU^2] - mirror of ``aperture_photometry._howell_variance_adu2``.

    Cite: ``src_py/aperture_photometry.py`` lines ~378-401 (Howell 1989 eq. 2 form).
    Terms: source Poisson ``flux/g``, sky Poisson ``sky_pp/g * area``,
    read noise ``(RN/g)^2 * area``. Kept local to avoid circular import with
    aperture_photometry.
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


@dataclass
class FrameImages:
    """One frame prepared once for every aperture call (SPEED-2026-10 S2).

    ``filled``: the image with NaN replaced by its NaN-median (the image itself when all pixels are
    finite - ``nan_to_num`` would return an equal copy); ``sky``: NaN off the aligned footprint;
    ``off``: pixels without data of the aligned frame (None when the footprint is not checked).
    """

    img: np.ndarray
    filled: np.ndarray
    sky: np.ndarray
    off: np.ndarray | None


def prepare_frame_images(image: np.ndarray, covered: np.ndarray | None = None) -> FrameImages:
    img = np.asarray(image, dtype=float)
    if bool(np.isfinite(img).all()):
        filled = img
    else:
        filled = np.nan_to_num(img, nan=float(np.nanmedian(img)))
    if covered is None:
        return FrameImages(img=img, filled=filled, sky=img, off=None)
    off = ~np.asarray(covered, dtype=bool)
    return FrameImages(img=img, filled=filled, sky=np.where(off, np.nan, img), off=off)


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
) -> tuple[dict[float, np.ndarray], dict[float, np.ndarray], np.ndarray]:
    """Sky-subtracted exact aperture fluxes for every k in ``f_grid`` (r = k x FWHM).

    One annulus per star (plain median of its pixels, no clipping) placed outside the
    largest aperture; all radii in a single ``aperture_photometry`` call.

    AP-FOOTPRINT-2026-10: ``covered`` (``frame_footprint.aligned_footprint``) - the sky median uses
    only annulus pixels with data of the aligned frame (not the aligner's constant fill); the third
    return value marks stars whose largest aperture reaches the fill or whose annulus keeps less than
    half of its pixels (the light curve drops them as ``edge_fail``).
    """
    from photutils.aperture import CircularAnnulus, CircularAperture  # noqa: PLC0415
    from photutils.aperture import aperture_photometry as _aphot  # noqa: PLC0415

    from aperture_policy import resolve_aperture_geometry  # noqa: PLC0415

    pos = np.asarray(xy, dtype=float)
    n = int(pos.shape[0]) if pos.ndim == 2 else 0
    ks = [float(f) for f in f_grid]
    out_f = {k: np.full(n, np.nan) for k in ks}
    out_s = {k: np.full(n, np.nan) for k in ks}
    edge = np.zeros(n, dtype=bool)
    if n == 0 or not ks:
        return out_f, out_s, edge
    fr = prepared if prepared is not None else prepare_frame_images(image, covered)
    _r, r_in, r_out = resolve_aperture_geometry(
        f=max(ks), fwhm_px=float(fwhm_px),
        annulus_inner_fwhm=float(annulus_inner_fwhm), annulus_outer_fwhm=float(annulus_outer_fwhm),
    )
    ok = np.all(np.isfinite(pos), axis=1)
    if not ok.any():
        return out_f, out_s, edge
    p = pos[ok]
    if fr.off is not None:
        from frame_footprint import off_footprint  # noqa: PLC0415

        edge[ok] = off_footprint(fr.off, p[:, 0], p[:, 1], max(ks) * float(fwhm_px), r_in, r_out)
    sky = np.full(int(ok.sum()), np.nan)
    for i, m in enumerate(CircularAnnulus(p, r_in=r_in, r_out=r_out).to_mask(method="center")):
        v = np.asarray(m.get_values(fr.sky), dtype=float)
        v = v[np.isfinite(v)]
        if v.size:
            sky[i] = float(np.median(v))
    aps = [CircularAperture(p, r=max(0.5, k * float(fwhm_px))) for k in ks]
    tbl = _aphot(fr.filled, aps, method="exact")
    for j, (k, ap) in enumerate(zip(ks, aps)):
        sums = np.asarray(tbl[f"aperture_sum_{j}"], dtype=float)
        out_f[k][ok] = sums - sky * float(ap.area)
        out_s[k][ok] = sky
    return out_f, out_s, edge


def blend_k_column(k: float) -> str:
    """Column of the neighbour-light fraction at r = k x FWHM (BLEND-K-2026-10)."""
    return f"blend_k{int(round(float(k) * 100)):03d}"


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
    f_own: float = float("nan")  # AP-NIGHT-K-2026-10: this target's own curve minimum (diagnostic)


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
    # PSF-LOCAL-2026-10: per-star radius scale (own PSF broader than the field), cid -> s
    scale_by_cid: dict[str, float] = field(default_factory=dict)
    # AP-FOOTPRINT-2026-10: frame_stem -> stars whose aperture / annulus is off the aligned frame's data
    edge_by_frame: dict[str, set[str]] = field(default_factory=dict)
    n_frames_no_transform: int = 0


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


_GRID_CTX: dict[str, Any] = {}


def _grid_worker_init(ctx: dict[str, Any]) -> None:
    _GRID_CTX.clear()
    _GRID_CTX.update(ctx)


def _grid_measure_frame(fp: Path) -> tuple[Any, ...] | None:
    """Measure one aligned frame on the k grid (SPEED-2026-10 S2: frames are independent).

    Returns ``(stem, fwhm_frame, used_night_fallback, has_transform, per_cid, per_sky, edge_ids)``
    or None when the frame has no usable proc CSV / FITS.
    """
    from astropy.io import fits  # noqa: PLC0415

    from frame_footprint import aligned_footprint, has_alignment_transform  # noqa: PLC0415

    c = _GRID_CTX
    stem = fp.stem
    proc = Path(c["frames_dir"]) / f"proc_{stem}.csv"
    if not proc.is_file():
        return None
    try:
        df = pd.read_csv(
            proc,
            usecols=lambda col: col in {"catalog_id", "x", "y"},
            low_memory=False,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("[APERTURE-DYNAMIC] proc read fail %s: %s", proc.name, exc)
        return None
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
    # AP-FOOTPRINT-2026-10: only pixels the aligned frame covers (no transform: not resampled).
    has_tr = has_alignment_transform(hdr_fp)
    covered = aligned_footprint(hdr_fp, img.shape) if has_tr else None
    prepared = prepare_frame_images(img, covered)
    for _s in sorted(set(np.round(_sc, 4))):
        _m = _sc == _s
        _fl, _sk, _ed = measure_flux_on_f_grid(
            img,
            xy[_m],
            f_grid=f_grid,
            fwhm_px=float(fwhm_frame) * float(_s),
            annulus_inner_fwhm=float(c["annulus_inner_fwhm"]),
            annulus_outer_fwhm=float(c["annulus_outer_fwhm"]),
            prepared=prepared,
        )
        edges[_m] = _ed
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
    return stem, float(fwhm_frame), bool(used_fb), bool(has_tr), per_cid, per_sky, edge_ids


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
    workers: int = 1,
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
    t0 = time.perf_counter()
    fits_files = sorted(Path(frames_dir).glob("*.fits"))
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
    nw = max(1, min(int(workers or 1), len(fits_files)))
    if nw > 1:
        import multiprocessing  # noqa: PLC0415
        from concurrent.futures import ProcessPoolExecutor  # noqa: PLC0415

        with ProcessPoolExecutor(
            max_workers=nw,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_grid_worker_init,
            initargs=(ctx,),
        ) as ex:
            results = list(ex.map(_grid_measure_frame, fits_files, chunksize=max(1, len(fits_files) // (4 * nw))))
    else:
        _grid_worker_init(ctx)
        results = [_grid_measure_frame(fp) for fp in fits_files]
        _GRID_CTX.clear()
    for res in results:
        if res is None:
            continue
        stem, fwhm_frame, used_fb, has_tr, per_cid, per_sky, edge_ids = res
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
    gain: float = 1.0,
    read_noise: float = 10.0,
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
            F_med, sky_pp if math.isfinite(sky_pp) else 0.0, math.pi * r_ap * r_ap,
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


def apply_night_k(
    choices: Mapping[str, PerTargetChoice],
    f_grid: Sequence[float],
    *,
    tie_frac: float = NIGHT_K_TIE_FRAC,
    min_target_frac: float = NIGHT_K_MIN_TARGET_FRAC,
) -> dict[str, Any]:
    """AP-NIGHT-K-2026-10: one k for every target of the group (changes ``choices`` in place).

    Night curve: for each k the median over targets of scatter(k) / min_k scatter, using the
    targets with a finite comp scatter at k inside their blend limit; a k enters only when at
    least ``min_target_frac`` of those targets may use it. k_night = the largest k within
    ``tie_frac`` of the curve's minimum. Each target then gets the largest k it may use that
    is <= k_night (else its smallest allowed k), targets without comp scatter included, so
    target, comps and check star keep one enclosed-light fraction across the group.
    No target with a curve -> nothing changes (the per-target fallbacks stay).
    """
    grid = sorted(float(k) for k in f_grid)
    rows: list[dict[float, float]] = []
    for c in choices.values():
        al = None if c.f_allowed is None else {round(float(k), 6) for k in c.f_allowed}
        sc = {float(k): float(v) for k, v in c.p2p_by_f.items()
              if math.isfinite(float(v)) and float(v) > 0 and (al is None or round(float(k), 6) in al)}
        if sc:
            mn = min(sc.values())
            rows.append({k: v / mn for k, v in sc.items()})
    info: dict[str, Any] = {"k_night": None, "curve": {}, "n_targets_curve": len(rows), "tie_frac": float(tie_frac)}
    if not rows:
        return info
    need = max(1.0, float(min_target_frac) * len(rows))
    curve = {k: float(np.median([r[k] for r in rows if k in r])) for k in grid if sum(k in r for r in rows) >= need}
    if not curve:
        return info
    best = min(curve.values())
    k_night = max(k for k, v in curve.items() if v <= best * (1.0 + float(tie_frac)))
    info.update(k_night=float(k_night), curve={f"{k:.4g}": v for k, v in curve.items()})
    for c in choices.values():
        al = sorted(float(k) for k in (c.f_allowed if c.f_allowed else grid))
        below = [k for k in al if k <= k_night + 1e-9]
        k_t = max(below) if below else min(al)
        c.f_own = float(c.f_star)
        scale = c.r_ap_px / c.f_star if c.f_star > 0 else float("nan")  # FWHM x PSF-LOCAL scale
        c.f_star = float(k_t)
        c.r_ap_px = float(k_t) * scale
        c.f_edge = f_star_is_grid_edge(float(k_t), al)
        c.snr_max = float(c.snr_by_f.get(f"{k_t:.4g}", float("nan")))
        c.reason = "night_k_comp_scatter" + ("" if k_t == k_night else "; blend limit below the night k")
    return info


def apply_grid_fluxes_to_frames(
    all_frames: pd.DataFrame,
    night: ApertureGridNight,
    *,
    catalog_ids: Sequence[str],
    f_star: float,
    gain: float | None = None,
    read_noise: float | None = None,
    annulus_inner_fwhm: float | None = None,
    annulus_outer_fwhm: float | None = None,
) -> pd.DataFrame:
    """Set mag_inst / aperture_r_px (r = k* x FWHM_frame) for target, comps and check star.

    A star without a grid flux in a frame gets mag_inst = NaN (never a different aperture
    in the same frame). With ``gain`` given, ``err`` (relative) is recomputed for the k*
    aperture from the CCD equation (Merline & Howell 1995):
        var = F/g + A (1 + A/n_sky) (sky/g + (RN/g)^2)   [ADU^2],  err = sqrt(var) / F
    where n_sky is the annulus area. A star whose aperture or annulus is off the aligned frame's data
    in that frame (``night.edge_by_frame``) gets mag_inst = NaN and flag ``edge_fail``.
    """
    if all_frames is None or all_frames.empty or "source_file" not in all_frames.columns:
        return all_frames
    out = all_frames.copy()
    want = {str(c) for c in catalog_ids}
    k = float(f_star)
    g = float(gain) if gain is not None and math.isfinite(float(gain)) and float(gain) > 0 else None
    rn = float(read_noise) if read_noise is not None and math.isfinite(float(read_noise)) else 0.0
    # SPEED-2026-10: visit only the rows of the wanted stars (column lists, same order as iterrows).
    _n = len(out)
    _cid_l = [str(v) for v in out["catalog_id"].tolist()] if "catalog_id" in out.columns else [""] * _n
    _src_l = [str(v) for v in out["source_file"].tolist()]
    _labels = out.index.tolist()
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
        if cid in night.edge_by_frame.get(stem, set()):
            out.at[i, "mag_inst"] = float("nan")
            out.at[i, "flag"] = "edge_fail"
            out.at[i, "edge_fail"] = True
            continue
        if flux is None or not math.isfinite(float(flux)) or float(flux) <= 0:
            out.at[i, "mag_inst"] = float("nan")
            continue
        out.at[i, "mag_inst"] = float(-2.5 * math.log10(float(flux)))
        if g is not None and annulus_inner_fwhm and annulus_outer_fwhm:
            sky = (night.sky_pp.get(stem, {}).get(cid) or {}).get(k, float("nan"))
            if sky is not None and math.isfinite(float(sky)):
                area = math.pi * r_ap * r_ap
                r_in = max(float(annulus_inner_fwhm) * fwhm, r_ap)
                n_sky = math.pi * ((float(annulus_outer_fwhm) * fwhm) ** 2 - r_in**2)
                var = float(flux) / g + area * (1.0 + area / max(n_sky, 1.0)) * (
                    max(float(sky), 0.0) / g + (rn / g) ** 2
                )
                if var > 0:
                    out.at[i, "err"] = float(math.sqrt(var) / float(flux))
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


def write_per_target_choices(
    path: Path,
    choices: Mapping[str, PerTargetChoice],
    *,
    f_grid: Sequence[float],
    fwhm_night_px: float,
    elapsed_s: float,
    gain: float | None = None,
    read_noise: float | None = None,
    night_k: Mapping[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_edge = sum(1 for c in choices.values() if bool(c.f_edge))
    payload = {
        "policy": "APERTURE-K-2026-10",
        "mode": MODE_PER_TARGET,
        "criterion": "night_k_comp_scatter",
        "night_k": dict(night_k) if night_k else None,
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
    "NIGHT_K_MIN_TARGET_FRAC",
    "NIGHT_K_TIE_FRAC",
    "SNR_FLAT_FRAC",
    "ApertureGridNight",
    "PerTargetChoice",
    "apply_grid_fluxes_to_frames",
    "choose_f_for_target",
    "allowed_f_from_comp_blend",
    "apply_night_k",
    "blend_k_column",
    "f_star_is_grid_edge",
    "flux_to_mag",
    "frame_stem_from_source",
    "howell_snr",
    "howell_variance_adu2",
    "measure_flux_on_f_grid",
    "measure_night_grid",
    "pick_f_star_snr",
    "series_from_grid",
    "sky_series_from_grid",
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

