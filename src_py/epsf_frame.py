"""Per-frame ePSF models and frame-level PSF photometry (EPSF-SPATIAL-2026-10).

Model of a frame = the frame's own ePSF (built from the MASTERSTAR build stars, all over the field)
plus a night-averaged spatial perturbation on a 3x3 (or 2x2) grid: the PSF core changes from frame
to frame with the seeing, while its variation across the field (optics) is stable over the night.
This is the Anderson & King (2006, ACS ISR 2006-01) library + per-exposure perturbation scheme
turned around for a ground-based night; a grid built independently on every frame has only ~60
stars per cell and its noise jumps from frame to frame (draft 535: comparison-star RMS 12.7 mmag
vs 7.8 mmag with the night-averaged perturbation).

Photometry fits the fit set together with its catalogue neighbours (Gaia down to the frame limit)
in one photutils ``PSFPhotometry`` call on the full frame (global pixel coordinates, so the gridded
model is evaluated at each star's position; stars closer than two fit boxes are fitted as a group).
Positions are fixed at Gaia through the MASTERSTAR WCS, only fluxes are fitted (EPSF-BLEND-2026-10):
a faint star next to a bright one keeps its own light. Local sky from an annulus, two passes with
per-pixel errors from the model (sky + model) / gain + RN^2.
``psf_flux_err`` is the fit (Fisher) error; on draft 535 (28 constant comparison stars, 146 frames,
gain 0.7925 e-/ADU) the scatter / error ratio is 1.17 - the bright stars' excess (~4 mmag) is
scintillation, which the internal PSF light curve adds from the aperture light curve. Inflating by
sqrt(chi2_red) would overstate the errors 2.5x: chi2 > 1 on bright stars is model-shape mismatch,
which does not translate into flux scatter (EPSF-ERR-2026-10).
"""

from __future__ import annotations

import hashlib
import io
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.nddata import NDData
from astropy.table import Table
from photutils.psf import GriddedPSFModel, ImagePSF, PSFPhotometry, SourceGrouper

from frame_footprint import aligned_footprint, off_footprint
from sky_estimation import annulus_sky_medians, sky_image
from psf_photometry import _epsf_build_imagepsf_from_stars, _epsf_sky_subtracted_stars

OVERSAMPLING = 2
# At least ~50 stars per PSF (Anderson et al. 2006, A&A 454, 1029); fewer build stars -> coarser grid.
_MIN_STARS_PER_CELL = 50
# The night perturbation is the median over every k-th frame, at most this many frames.
_DELTA_MAX_FRAMES = 30
# photutils flags that make a fit unusable: 2 outside image, 4 non-positive flux,
# 8 fit did not converge, 16 covariance unavailable.
_BAD_FIT_FLAGS = 2 | 4 | 8 | 16
# Blend error (EPSF-BLEND-ERR2-2026-10): the model-shape error of the neighbours' light inside the fit box
# is a fixed fraction of that light, sigma_F = k * F_nb, i.e. sigma_rel = k * f_nb (f_nb = neighbour light /
# own light in the box). Injections 2-5 mag fainter, 3-12 px from bright stars: k fitted on draft 535
# (wide rig, 9.8"/px) 0.0406, on Brno 533 g / 534 r (0.56"/px) 0.0399; each tested on the other rig:
# pull 0.78-1.22 per separation bin. The earlier 0.113 * sqrt(f_nb) (535 only) gave 0.45-1.35 on Brno.
_BLEND_ERR_K = 0.040
# EPSF-MERGE-2026-10: a fainter Gaia neighbour closer than 0.5 x night FWHM is merged with the fit star
# (injections on Brno 533 g at 0.25 / 0.5 / 0.75 / 1.0 FWHM, 2-5 mag fainter: joint fit 110-250 / 55-70 /
# 20-33 / 8-19 mmag, merged 3-11 / 3-12 / 2-15 / 3-36 mmag; the crossover is at 0.75-1 FWHM of the frame,
# 0.5 x the night median frame FWHM keeps the night's best frames (533 g 3.77 / 4.50, 534 g 3.34 / 5.48 px)
# below it).
_MERGE_FWHM = 0.5


def grid_shape_for(n_build_stars: int) -> tuple[int, int]:
    """(nx, ny) of the spatial grid: 3x3 or 2x2 with >= 50 build stars per cell, else (1, 1)."""
    for n in (3, 2):
        if n_build_stars >= _MIN_STARS_PER_CELL * n * n:
            return n, n
    return 1, 1


def grid_cells(shape: tuple[int, int], nx: int, ny: int) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float]]]:
    """Cell edges in x and y and the cell centres (x, y) in pixel coordinates."""
    h, w = shape
    xe, ye = np.linspace(0.0, float(w), nx + 1), np.linspace(0.0, float(h), ny + 1)
    centres = [((xe[i] + xe[i + 1]) / 2.0, (ye[j] + ye[j + 1]) / 2.0) for j in range(ny) for i in range(nx)]
    return xe, ye, centres


def _xy_table(df: pd.DataFrame) -> Table:
    return Table([df["x"].to_numpy(dtype=float), df["y"].to_numpy(dtype=float)], names=("x", "y"))


def _build_epsf(data: np.ndarray, stars_df: pd.DataFrame, cutout_size: int, fwhm_px: float) -> dict[str, Any]:
    stars = _epsf_sky_subtracted_stars(data, _xy_table(stars_df), int(cutout_size))
    built = _epsf_build_imagepsf_from_stars(stars, osamp=OVERSAMPLING, fwhm_px=float(fwhm_px), cutout_size=int(cutout_size))
    built["n_stars"] = int(len(stars))
    return built


def _read(frame_fits: Path) -> tuple[np.ndarray, np.ndarray]:
    """Frame data and its aligned footprint (``aligned_footprint``)."""
    with fits.open(frame_fits, memmap=True) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float32)
        return data, aligned_footprint(hdul[0].header, data.shape)


def _covered_stars(stars_df: pd.DataFrame, covered: np.ndarray, cutout_size: int) -> pd.DataFrame:
    """Build stars whose whole cutout lies on covered pixels of the frame."""
    half = int(cutout_size) // 2
    h, w = covered.shape
    keep = []
    for x, y in zip(stars_df["x"].to_numpy(float), stars_df["y"].to_numpy(float), strict=False):
        xi, yi = int(round(x)), int(round(y))
        y0, y1, x0, x1 = yi - half, yi + half + 1, xi - half, xi + half + 1
        keep.append(bool(y0 >= 0 and x0 >= 0 and y1 <= h and x1 <= w and covered[y0:y1, x0:x1].all()))
    return stars_df[np.asarray(keep, dtype=bool)]


_DELTA_CTX: dict[str, Any] = {}


def _delta_worker_init(ctx: dict[str, Any]) -> None:
    _DELTA_CTX.clear()
    _DELTA_CTX.update(ctx)


def _frame_cell_deltas(frame_fits: Path) -> tuple[np.ndarray, list[int], float, tuple[int, int]]:
    """(cell - field) ePSFs of one frame, the build stars per cell, the field ePSF peak, frame shape."""
    c = _DELTA_CTX
    nx, ny, cutout_size, fwhm_px = c["nx"], c["ny"], c["cutout_size"], c["fwhm_px"]
    data, covered = _read(frame_fits)
    stars_fp = _covered_stars(c["build_stars"], covered, cutout_size)
    xe, ye, _ = grid_cells(data.shape, nx, ny)
    field = _build_epsf(data, stars_fp, cutout_size, fwhm_px)["arr"].astype(np.float64)
    cells = []
    n_cell = []
    for j in range(ny):
        for i in range(nx):
            m = (
                (stars_fp["x"] >= xe[i]) & (stars_fp["x"] < xe[i + 1])
                & (stars_fp["y"] >= ye[j]) & (stars_fp["y"] < ye[j + 1])
            )
            cell = _build_epsf(data, stars_fp[m], cutout_size, fwhm_px)
            cells.append(cell["arr"].astype(np.float64) - field)
            n_cell.append(cell["n_stars"])
    return np.array(cells), n_cell, float(np.max(field)), tuple(data.shape)


def night_spatial_delta(
    frames: list[Path],
    build_stars: pd.DataFrame,
    *,
    cutout_size: int,
    fwhm_px: float,
    out_path: Path,
    workers: int = 1,
) -> dict[str, Any]:
    """Night-averaged perturbation of every grid cell from the frame ePSF; written to ``out_path``.

    For a subset of frames (every k-th, at most 30) the cell ePSFs and the whole-field ePSF are built
    from the same frame, from the build stars whose cutout the frame covers (``aligned_footprint``);
    delta_cell = median over frames of (cell - field). (1, 1) grid -> no file. ``workers`` > 1 builds
    the frames in a process pool; the result does not depend on it.
    """
    from parallel_frames import map_frames  # noqa: PLC0415

    nx, ny = grid_shape_for(len(build_stars))
    out = {"grid": [nx, ny], "n_build_stars": int(len(build_stars)), "file": None, "n_frames": 0}
    if nx == 1 or not frames:
        return out
    step = max(1, math.ceil(len(frames) / _DELTA_MAX_FRAMES))
    subset = frames[::step]
    ctx = {"nx": nx, "ny": ny, "cutout_size": int(cutout_size), "fwhm_px": float(fwhm_px), "build_stars": build_stars}
    res = list(map_frames(_frame_cell_deltas, subset, workers=workers, initializer=_delta_worker_init, ctx=ctx,
                          stage="ePSF spatial delta"))
    deltas = [r[0] for r in res]
    n_cell, peak = res[-1][1], res[-1][2]
    _, _, centres = grid_cells(res[-1][3], nx, ny)
    delta = np.median(np.array(deltas), axis=0)
    hdr = fits.Header()
    hdr["GRID_NX"], hdr["GRID_NY"], hdr["NFRAMES"], hdr["OVERSAMP"] = nx, ny, len(subset), OVERSAMPLING
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(delta.astype(np.float32), header=hdr).writeto(out_path, overwrite=True)
    out.update(
        {
            "file": out_path.name,
            "n_frames": len(subset),
            "frames": [p.name for p in subset],
            "n_build_stars_per_cell": n_cell,
            "cell_centres": centres,
            "max_abs_delta_over_peak": [round(float(np.max(np.abs(d)) / peak), 4) for d in delta],
            "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
        }
    )
    return out


def build_frame_model(
    frame_fits: Path,
    build_stars: pd.DataFrame,
    *,
    cutout_size: int,
    fwhm_px: float,
    delta: np.ndarray | None,
    grid: tuple[int, int],
) -> tuple[Any, dict[str, Any]]:
    """The frame's ePSF (+ night perturbation per cell); returns the photutils model and its record.

    The model is used in memory only (OUTPUT-REVIEW-2026-10: no per-frame model files); the record's
    ``sha256`` is the hash of the model as a FITS (primary: the whole-field ePSF; extension CELLS:
    the cell models), so the manifest still identifies the model set.
    """
    data, covered = _read(frame_fits)
    built = _build_epsf(data, _covered_stars(build_stars, covered, cutout_size), cutout_size, fwhm_px)
    field = built["arr"].astype(np.float64)
    nx, ny = grid
    hdus = [fits.PrimaryHDU(field.astype(np.float32))]
    if delta is not None and nx > 1:
        cells = np.array([field + d for d in delta])
        cells = np.array([c / (c.sum() / OVERSAMPLING**2) for c in cells])
        _, _, centres = grid_cells(data.shape, nx, ny)
        model: Any = GriddedPSFModel(NDData(cells, meta={"grid_xypos": centres, "oversampling": OVERSAMPLING}))
        hdus.append(fits.ImageHDU(cells.astype(np.float32), name="CELLS"))
    else:
        model = ImagePSF(field, oversampling=OVERSAMPLING)
    buf = io.BytesIO()
    fits.HDUList(hdus).writeto(buf)
    fwhm_frame = built["qc"].get("epsf_fwhm_native_px")
    record = {
        "frame": Path(frame_fits).name,
        "sha256": hashlib.sha256(buf.getvalue()).hexdigest(),
        "fwhm_px": float(fwhm_frame) if fwhm_frame is not None else float(fwhm_px),
        "grid": [nx, ny],
        "n_stars_used": built["n_stars"],
        "converged": bool(built["build_converged"]),
        "iterations": len(built["build_iterations"]),
        "smoothing_kernel": built["smoothing"],
    }
    return model, record


def frame_fit_sources(
    platesolve_dir: Path, fit_ids: set[str] | list[str], *, cutout_size: int, fwhm_px: float, merge_fwhm_px: float
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fixed positions of the fit set and of the catalogue neighbours fitted with them.

    Positions: Gaia DR3 (``field_catalog_cone.csv``) through the MASTERSTAR WCS - the frames are
    aligned to the MASTERSTAR, and a DAO centroid of a faint star is pulled towards a bright
    neighbour (535: 0.6 px). Neighbours: catalogue stars down to the frame limit
    (``frame_depth.g_lim_frame``) whose model reaches a fit star's fit box
    (cutout/2 + fit box/2 + 1 px). A fainter neighbour closer than ``_MERGE_FWHM`` x ``merge_fwhm_px``
    (the night median frame FWHM, not the MASTERSTAR: 534 g MASTERSTAR 3.44 px, frames 3.34-9.55 px,
    median 5.48) is not fitted: such a pair is not separable (Brno injections at 0.25-0.5 FWHM: joint fit 55-250 mmag
    scatter with pull 2-3, merged 3-12 mmag), so its light stays with the fit star
    (``merged_light_frac``, Gaia G flux ratio). Returns (stars[catalog_id, x, y, merged_light_frac],
    neighbours[catalog_id, x, y, mag], meta).
    """
    from astropy.wcs import WCS  # noqa: PLC0415
    from scipy.spatial import cKDTree  # noqa: PLC0415

    from gaia_catalog_id import normalize_gaia_source_id  # noqa: PLC0415
    from psf_photometry import frame_mag_limit  # noqa: PLC0415

    ps = Path(platesolve_dir)
    cone = pd.read_csv(ps / "field_catalog_cone.csv", low_memory=False, dtype={"catalog_id": str})
    cone["catalog_id"] = cone["catalog_id"].map(lambda v: str(normalize_gaia_source_id(v) or "").strip())
    cone = cone.drop_duplicates("catalog_id")
    wcs = WCS(fits.getheader(ps / "MASTERSTAR.fits"))
    cone["x"], cone["y"] = wcs.all_world2pix(
        pd.to_numeric(cone["ra_deg"], errors="coerce").to_numpy(float),
        pd.to_numeric(cone["dec_deg"], errors="coerce").to_numpy(float),
        0,
    )
    cone["mag"] = pd.to_numeric(cone["mag"], errors="coerce")
    cone = cone[np.isfinite(cone["x"]) & np.isfinite(cone["y"])]
    ids = {str(c) for c in fit_ids}
    stars = cone.loc[cone["catalog_id"].isin(ids), ["catalog_id", "x", "y"]].reset_index(drop=True)
    missing = sorted(ids - set(stars["catalog_id"]))
    mag_limit = frame_mag_limit(ps)
    fsz = max(5, 2 * int(round(float(fwhm_px))) + 1)
    reach = int(cutout_size) / 2.0 + fsz / 2.0 + 1.0
    cand = cone[(cone["mag"] <= mag_limit) & ~cone["catalog_id"].isin(ids)]
    tree = cKDTree(np.c_[cand["x"].to_numpy(float), cand["y"].to_numpy(float)])
    hit: set[int] = set()
    for x0, y0 in zip(stars["x"], stars["y"], strict=False):
        hit.update(tree.query_ball_point([x0, y0], reach))
    nb = cand.iloc[sorted(hit)][["catalog_id", "x", "y", "mag"]].reset_index(drop=True)
    # Unresolved companions (EPSF-MERGE-2026-10): a fainter neighbour closer than _MERGE_FWHM x FWHM is not
    # fitted - its light stays with the fit star as a constant dilution (as in the aperture). One decision
    # per night (night median frame FWHM, ``merge_fwhm_px``), so the light curve has no dilution jumps.
    merge_r = _MERGE_FWHM * float(merge_fwhm_px)
    star_mag = cone.set_index("catalog_id")["mag"]
    stars["merged_light_frac"] = 0.0
    merged: list[dict[str, Any]] = []
    drop: set[int] = set()
    for i, s in stars.iterrows():
        sep = np.hypot(nb["x"].to_numpy(float) - s["x"], nb["y"].to_numpy(float) - s["y"])
        m_s = float(star_mag.get(s["catalog_id"], np.nan))
        for j in np.flatnonzero(sep < merge_r):
            m_n = float(nb.at[j, "mag"])
            if np.isfinite(m_s) and np.isfinite(m_n) and m_n > m_s:
                drop.add(int(j))
                stars.at[i, "merged_light_frac"] += 10.0 ** (-0.4 * (m_n - m_s))
                merged.append({"catalog_id": str(s["catalog_id"]), "neighbour_id": str(nb.at[j, "catalog_id"]),
                               "sep_px": round(float(sep[j]), 3), "dmag": round(m_n - m_s, 3)})
    nb = nb.drop(index=sorted(drop)).reset_index(drop=True)
    meta = {
        "position_source": "gaia_dr3_masterstar_wcs",
        "n_fit_stars": int(len(stars)),
        "fit_ids_not_in_catalogue": missing,
        "n_neighbours": int(len(nb)),
        "neighbour_mag_limit": float(mag_limit),
        "neighbour_reach_px": float(reach),
        "merge_fwhm_px": round(float(merge_fwhm_px), 3),
        "merge_radius_px": round(merge_r, 3),
        "merged_neighbours": merged,
    }
    return stars, nb, meta


def fit_frame_psf(
    data: np.ndarray,
    model: Any,
    stars: pd.DataFrame,
    neighbours: pd.DataFrame | None = None,
    *,
    fwhm_px: float,
    gain: float,
    read_noise_e: float,
    sat_limit_adu: float | None = None,
    covered: np.ndarray | None = None,
    fit_box_fwhm: float = 1.0,
) -> pd.DataFrame:
    """PSF photometry of ``stars`` (catalog_id, x, y) on one frame, fitted together with their
    catalogue ``neighbours`` (x, y); one row per star of ``stars`` (EPSF-BLEND-2026-10).

    Positions are fixed (Gaia through the MASTERSTAR WCS; the frames are aligned to it): only the
    fluxes are fitted. Stars closer than two fit boxes form one group and are fitted jointly
    (photutils ``SourceGrouper``), so a faint star next to a bright one keeps its own light - fitted
    alone with a free position it slid onto the bright neighbour (draft 535: up to 3 px, 2-4 mag
    too bright). Sky: plain median of an annulus 4-6 FWHM without detected sources (SKY-SRC-MASK-2026-10). Pass 1 uses per-pixel errors from the data,
    pass 2 from the pass-1 model image ((sky + model) / gain + (RN / gain)^2, ADU): weights from
    the data bias faint stars low (Stetson 1987; ALLSTAR uses the model). The flux error is the
    photutils covariance error; chi^2 is kept as ``psf_chi2`` for the night locus.

    Pixels at or above ``sat_limit_adu`` (the frame's non-linearity / saturation peak limit, the
    catalogue ``saturate_limit_adu_85pct``) and non-finite pixels are masked: a saturated core is
    fitted from its unsaturated wings (DAOPHOT "high good datum"); ``psf_sat_masked`` marks a star
    whose fit box lost pixels to the mask.

    ``covered`` (``aligned_footprint``): pixels the aligned frame does not cover hold a constant fill and
    are masked too (also in the sky annulus: photutils passes the mask to the local background);
    ``psf_edge_masked`` marks a star whose fit box reaches them or whose sky annulus keeps less than
    half of its pixels.
    """
    d = np.asarray(data, dtype=np.float64)
    mask = ~np.isfinite(d)
    if sat_limit_adu is not None and math.isfinite(float(sat_limit_adu)):
        mask |= d >= float(sat_limit_adu)
    sat_mask = mask.copy()
    off = np.zeros(d.shape, dtype=bool) if covered is None else ~np.asarray(covered, dtype=bool)
    mask |= off
    fw = float(fwhm_px)
    fsz = max(5, 2 * int(round(fit_box_fwhm * fw)) + 1)
    sky0 = float(np.nanmedian(d))
    psf = model.copy()
    psf.x_0.fixed = True
    psf.y_0.fixed = True
    phot = PSFPhotometry(
        psf, (fsz, fsz), aperture_radius=1.5 * fw, grouper=SourceGrouper(min_separation=2.0 * fsz),
    )
    n_st = len(stars)
    nb = neighbours if neighbours is not None else pd.DataFrame({"x": [], "y": []})
    xs = np.r_[stars["x"].to_numpy(float), nb["x"].to_numpy(float)]
    ys = np.r_[stars["y"].to_numpy(float), nb["y"].to_numpy(float)]
    # Local sky (SKY-SRC-MASK-2026-10): plain median of the 4-6 FWHM annulus pixels that are on the footprint,
    # unsaturated and not on a detected source - the aperture photometry's estimator; replaces photutils
    # MedianBackground, whose default is a 3-sigma clip with 10 iterations (zero-clipping policy).
    lbkg = annulus_sky_medians(sky_image(np.where(mask, np.nan, d), fw), np.c_[xs, ys], 4.0 * fw, 6.0 * fw)
    rn_adu2 = (float(read_noise_e) / float(gain)) ** 2
    err1 = np.sqrt(np.clip(np.nan_to_num(d, nan=sky0), sky0, None) / float(gain) + rn_adu2)
    init = Table({"id": np.arange(1, len(xs) + 1), "x": xs, "y": ys, "local_bkg": lbkg})
    r1 = phot(d, mask=mask, init_params=init, error=err1)
    model_img = phot.make_model_image(d.shape, include_local_bkg=False)
    err2 = np.sqrt(np.clip(sky0 + model_img, 0.0, None) / float(gain) + rn_adu2)
    r1 = r1[np.argsort(np.asarray(r1["id"]))]
    init2 = Table({"id": np.asarray(r1["id"]), "x": xs, "y": ys, "flux": np.asarray(r1["flux_fit"], float),
                   "local_bkg": lbkg})
    r2 = phot(d, mask=mask, init_params=init2, error=err2)
    all_img = phot.make_model_image(d.shape, include_local_bkg=False)
    r2 = r2[np.argsort(np.asarray(r2["id"]))][:n_st]
    flux = np.asarray(r2["flux_fit"], float)
    nb_frac = _neighbour_light_in_box(psf, all_img, xs[:n_st], ys[:n_st], flux, fsz // 2)
    chi2 = np.asarray(r2["reduced_chi2"], float)
    flags = np.asarray(r2["flags"], int)
    ferr_fit = np.asarray(r2["flux_err"], float)
    # Blend term (EPSF-BLEND-2026-10): model-shape error of the neighbours' light inside the fit box,
    # not in the Fisher error; relative sigma = k * (neighbour light / own light).
    blend_rel = _BLEND_ERR_K * np.clip(np.nan_to_num(nb_frac, nan=0.0), 0.0, None)
    ferr = np.sqrt(ferr_fit**2 + (blend_rel * np.abs(flux)) ** 2)
    good = ((flags & _BAD_FIT_FLAGS) == 0) & np.isfinite(flux) & (flux > 0) & np.isfinite(ferr)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.where(good & (ferr > 0), flux / ferr, np.nan)
    return pd.DataFrame(
        {
            "catalog_id": stars["catalog_id"].astype(str).to_numpy(),
            "psf_flux": flux,
            "psf_flux_err": ferr,
            "psf_flux_err_fit": ferr_fit,
            "psf_chi2": chi2,
            "psf_flags": flags,
            "psf_converged": good,
            "psf_snr": snr,
            "psf_sky": np.asarray(r2["local_bkg"], float),
            "psf_group_size": np.asarray(r2["group_size"], int),
            "psf_nb_light_frac": nb_frac,
            "psf_sat_masked": _box_has_mask(sat_mask, stars["x"].to_numpy(float), stars["y"].to_numpy(float), fsz // 2),
            "psf_edge_masked": off_footprint(off, stars["x"].to_numpy(float), stars["y"].to_numpy(float), fsz // 2, 4.0 * fw, 6.0 * fw, square_core=True),
        }
    )


def _neighbour_light_in_box(
    psf: Any, all_img: np.ndarray, xs: np.ndarray, ys: np.ndarray, flux: np.ndarray, half: int
) -> np.ndarray:
    """Model light of the other fitted sources inside each star's fit box, fraction of its own."""
    h, w = all_img.shape
    out = np.full(len(xs), np.nan)
    m = psf.copy()
    for i, (x, y, f) in enumerate(zip(xs, ys, flux, strict=False)):
        if not (np.isfinite(f) and f > 0):
            continue
        xi, yi = int(round(x)), int(round(y))
        y0, y1, x0, x1 = max(0, yi - half), min(h, yi + half + 1), max(0, xi - half), min(w, xi + half + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        m.x_0, m.y_0, m.flux = x, y, f
        own = m(xx, yy)
        tot_own = float(own.sum())
        if tot_own > 0:
            out[i] = float((all_img[y0:y1, x0:x1] - own).sum()) / tot_own
    return out


def _box_has_mask(mask: np.ndarray, xs: np.ndarray, ys: np.ndarray, half: int) -> np.ndarray:
    """True where the (2*half+1)^2 fit box around (x, y) contains a masked pixel."""
    h, w = mask.shape
    out = np.zeros(len(xs), dtype=bool)
    for i, (x, y) in enumerate(zip(xs, ys, strict=False)):
        xi, yi = int(round(x)), int(round(y))
        y0, y1, x0, x1 = max(0, yi - half), min(h, yi + half + 1), max(0, xi - half), min(w, xi + half + 1)
        out[i] = bool(mask[y0:y1, x0:x1].any()) if y0 < y1 and x0 < x1 else True
    return out
