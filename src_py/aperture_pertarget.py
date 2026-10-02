"""APERTURE-DYNAMIC-01 / APERTURE-PERTARGET-01: per-target aperture (default).

Physics (D5-1): target and its comps share the same enclosed-energy fraction
(same f = r / FWHM) for one differential measurement.

Selection: f* = argmin of Abbe / von Neumann point-to-point scatter of the
differential LC (comps at the same f). Ties / flat minimum -> larger f.
Production radius: r = f* x FWHM_frame (FWHM-AUTH-01); night median only as
fallback when a frame has no QC FWHM.

Default production mode is ``per_target`` (APERTURE-DYNAMIC-01). Fixed one-r
for the draft remains available as ``f_fixed_night``.
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

# APERTURE-DYNAMIC-01: extended below 0.75 so f* is not forced to the edge.
DEFAULT_APERTURE_F_GRID: tuple[float, ...] = (
    0.5,
    0.6,
    0.75,
    1.0,
    1.25,
    1.35,
    1.5,
    1.75,
    2.0,
    2.5,
)

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
    """argmin p2p; ties / flat minimum -> larger f. None if all non-finite."""
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


def f_star_is_grid_edge(f_star: float, f_grid: Sequence[float], *, tol: float = 1e-9) -> bool:
    """True if f* equals the minimum or maximum of the sampling grid."""
    vals = sorted(float(x) for x in f_grid if math.isfinite(float(x)) and float(x) > 0)
    if len(vals) < 2 or not math.isfinite(float(f_star)):
        return False
    fs = float(f_star)
    return abs(fs - vals[0]) <= tol or abs(fs - vals[-1]) <= tol


def _aperture_flux_uniform(
    image: np.ndarray,
    pos: np.ndarray,
    r_ap: float,
    r_in: float,
    r_out: float,
) -> np.ndarray:
    """Sky-subtracted exact aperture sums at one shared radius (photutils)."""
    from photutils.aperture import CircularAnnulus, CircularAperture
    from photutils.aperture import aperture_photometry as _aphot
    from sky_estimation import sky_median_mask  # noqa: PLC0415

    pos = np.asarray(pos, dtype=np.float64)
    n = int(pos.shape[0])
    flux_arr = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return flux_arr
    r0 = float(r_ap)
    rin = float(r_in)
    rout = float(r_out)
    if not (math.isfinite(r0) and r0 > 0 and math.isfinite(rin) and rin > 0 and rout > rin):
        return flux_arr
    try:
        ap = CircularAperture(pos, r=r0)
        phot = _aphot(image, ap, method="exact")
        sums = np.asarray(phot["aperture_sum"], dtype=np.float64)
        area = float(ap.area)
        an = CircularAnnulus(pos, r_in=rin, r_out=rout)
        masks = an.to_mask(method="center")
        if not isinstance(masks, (list, tuple)):
            masks = [masks]
        sky_pp = np.full(n, np.nan, dtype=np.float64)
        for i, m in enumerate(masks):
            try:
                ann_img = m.to_image(image.shape)
                sky_pp[i] = float(sky_median_mask(image, ann_img))
            except Exception:  # noqa: BLE001
                sky_pp[i] = float("nan")
        flux_arr = sums - sky_pp * area
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("[APERTURE-DYNAMIC] uniform aperture failed: %s", exc)
    return flux_arr


def measure_flux_on_f_grid(
    image: np.ndarray,
    xy: np.ndarray,
    *,
    f_grid: Sequence[float],
    fwhm_px: float,
    annulus_inner_fwhm: float,
    annulus_outer_fwhm: float,
) -> dict[float, np.ndarray]:
    """Measure sky-subtracted aperture flux for every star at each f in the grid."""
    from aperture_policy import resolve_aperture_geometry  # noqa: PLC0415

    pos = np.asarray(xy, dtype=float)
    n = int(pos.shape[0]) if pos.ndim == 2 else 0
    out: dict[float, np.ndarray] = {}
    if n == 0:
        for f in f_grid:
            out[float(f)] = np.array([], dtype=float)
        return out
    img = np.asarray(image, dtype=float)
    for f in f_grid:
        ff = float(f)
        r_ap, r_in, r_out = resolve_aperture_geometry(
            f=ff,
            fwhm_px=float(fwhm_px),
            annulus_inner_fwhm=float(annulus_inner_fwhm),
            annulus_outer_fwhm=float(annulus_outer_fwhm),
        )
        out[ff] = _aperture_flux_uniform(img, pos, r_ap, r_in, r_out)
    return out


@dataclass
class PerTargetChoice:
    catalog_id: str
    f_star: float
    r_ap_px: float
    p2p_by_f: dict[str, float] = field(default_factory=dict)
    n_comps: int = 0
    n_frames: int = 0
    f_edge: bool = False


@dataclass
class ApertureGridNight:
    """In-memory night grid: frame_key -> catalog_id -> f -> flux."""

    f_grid: list[float]
    fwhm_night_px: float
    # frame_stem -> {cid: {f: flux}}
    frames: dict[str, dict[str, dict[float, float]]] = field(default_factory=dict)
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
        fluxes = measure_flux_on_f_grid(
            img,
            xy,
            f_grid=night.f_grid,
            fwhm_px=float(fwhm_frame),
            annulus_inner_fwhm=float(annulus_inner_fwhm),
            annulus_outer_fwhm=float(annulus_outer_fwhm),
        )
        per_cid: dict[str, dict[float, float]] = {cid: {} for cid in ids}
        for f, arr in fluxes.items():
            for i, cid in enumerate(ids):
                v = float(arr[i]) if i < len(arr) else float("nan")
                per_cid[cid][float(f)] = v
        night.frames[stem] = per_cid
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


def choose_f_for_target(
    night: ApertureGridNight,
    *,
    target_cid: str,
    comp_ids: Sequence[str],
    frame_order: Sequence[str],
) -> PerTargetChoice:
    """Pick f* for one target from Abbe p2p of equal-weight differential LCs."""
    p2p_by_f: dict[float, float] = {}
    for f in night.f_grid:
        t_flux = series_from_grid(night, target_cid, f, frame_order)
        t_mag = flux_to_mag(t_flux)
        comp_mags = {
            str(c): flux_to_mag(series_from_grid(night, str(c), f, frame_order))
            for c in comp_ids
        }
        delta = equal_weight_ensemble_delta(t_mag, comp_mags)
        p2p_by_f[float(f)] = abbe_p2p_scatter(delta)
    f_star = pick_f_star(p2p_by_f)
    if f_star is None:
        f_star = float(night.f_grid[len(night.f_grid) // 2]) if night.f_grid else 1.35
    edge = f_star_is_grid_edge(float(f_star), night.f_grid)
    # Representative r_ap for audit (night median scale); per-frame r set at apply.
    r_ap = float(f_star) * float(night.fwhm_night_px)
    return PerTargetChoice(
        catalog_id=str(target_cid),
        f_star=float(f_star),
        r_ap_px=float(r_ap),
        p2p_by_f={f"{k:.4g}": float(v) for k, v in sorted(p2p_by_f.items())},
        n_comps=len(list(comp_ids)),
        n_frames=len(list(frame_order)),
        f_edge=bool(edge),
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
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_edge = sum(1 for c in choices.values() if bool(c.f_edge))
    payload = {
        "policy": "APERTURE-DYNAMIC-01",
        "mode": MODE_PER_TARGET,
        "f_grid": [float(f) for f in f_grid],
        "fwhm_night_px": float(fwhm_night_px),
        "elapsed_s": float(elapsed_s),
        "n_targets": len(choices),
        "n_f_edge": int(n_edge),
        "targets": {
            cid: {
                "f_star": c.f_star,
                "r_ap_px": c.r_ap_px,
                "aperture_f_edge": bool(c.f_edge),
                "p2p_by_f": c.p2p_by_f,
                "n_comps": c.n_comps,
                "n_frames": c.n_frames,
            }
            for cid, c in choices.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


__all__ = [
    "DEFAULT_APERTURE_F_GRID",
    "MODE_PER_TARGET",
    "ApertureGridNight",
    "PerTargetChoice",
    "abbe_p2p_scatter",
    "apply_grid_fluxes_to_frames",
    "choose_f_for_target",
    "equal_weight_ensemble_delta",
    "f_star_is_grid_edge",
    "flux_to_mag",
    "frame_stem_from_source",
    "measure_flux_on_f_grid",
    "measure_night_grid",
    "normalize_f_grid",
    "pick_f_star",
    "series_from_grid",
    "write_per_target_choices",
]
