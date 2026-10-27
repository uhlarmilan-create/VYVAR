# -*- coding: ascii -*-
"""Search for new sources in the field of one night (TRANSIENT-2026-10, Milan 2026-10-25).

Part of the field variability search (:func:`variability_field.run_field_variability` calls it when
``transient_search_enabled`` is on; off by default). Method as Gaia Science Alerts (Hodgkin et al.
2021) and NMW / VaST (Sokolovsky et al. 2014, 2018): detections compared with the catalogue, no
image subtraction; the many frames of one night are the filter against artefacts (ATLAS: >= 3
detections, Smith et al. 2020).

1. Sources: DAOStarFinder on every aligned frame (5 sigma_pix, the frame FWHM, inside the frame
   footprint); a detection within max(1 FWHM, 2 px) of any MASTERSTAR source (Gaia-matched or not)
   is known. The rest is linked across frames (friends-of-friends, 1 FWHM): a group seen on >= 3
   frames is a stationary source; a group whose position runs with time by more than 1.5 FWHM is a
   moving object. MASTERSTAR detections without a Gaia star brighter than the catalogue depth
   (``unmatched_in_range``) are sources too.
2. Every stationary source is measured on every frame at its position: aperture k x FWHM_frame
   (the field k), sky = median of the annulus, the frame zero point of the variability search
   (Honeycutt), CCD-equation error; magnitude on the Gaia G scale of the field.
3. Kept as a candidate (T1, T2, ...): S/N >= 5 on >= 3 consecutive frames; centroid within 0.75 FWHM;
   not fixed on the detector while the field moves (hot pixel); no Gaia star within
   max(2", 1.5 FWHM) that is less than 1 mag fainter than the source; no G < 12 star within
   max(10", 4 FWHM); aperture and annulus inside every frame. Kind: ``appeared`` (the first frames
   are below 3 sigma, then >= 3 consecutive detections), ``faded`` (the reverse), ``present``.
   ``extended`` when its moment FWHM on MASTERSTAR exceeds 1.5 x that of field stars (a galaxy:
   a supernova on top of it is not found without image subtraction).
4. Minor planets and comets (``minor_bodies``, offline MPC orbits in the catalogue folder): bodies
   brighter than V 20 in the field, whether they are detected, and which moving object they are.

No sigma clipping. Products: ``transient_candidates.csv``, ``moving_objects.csv``,
``minor_bodies_in_field.csv``, ``transient_meta.json``, ``lightcurves/transient_<id>.csv``.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gaia_catalog_id import read_vyvar_csv

from parallel_frames import map_frames

LOGGER = logging.getLogger(__name__)

TRANSIENT_CSV = "transient_candidates.csv"
MOVING_CSV = "moving_objects.csv"
BODIES_CSV = "minor_bodies_in_field.csv"
META_JSON = "transient_meta.json"

DETECT_NSIGMA = 5.0
SNR_DETECT = 5.0
N_CONSEC_MIN = 3
MATCH_FWHM = 1.0
LINK_FWHM = 1.0
MOVE_FWHM = 1.5
CENTROID_MAX_FWHM = 0.75
NEIGHBOUR_ARCSEC, NEIGHBOUR_FWHM, NEIGHBOUR_DMAG = 2.0, 2.5, 1.0
#: bright stars: their halo, spikes and reflections reach r = 4 FWHM x 10**(0.15 (12 - G)) (G 12: 4 FWHM,
#: G 7: 23 FWHM, G 4.6: 52 FWHM - V842 Her 537: three "sources" in the halo of a G 4.6 star 9 FWHM away)
# halo of a very bright star: ghosts, spikes and the wings fool the detector; only stars that are
# much brighter than the source count, and the radius grows with the star's brightness
BRIGHT_G, BRIGHT_ARCSEC, BRIGHT_FWHM, BRIGHT_SLOPE, BRIGHT_DMAG = 8.0, 10.0, 4.0, 0.2, 3.0
EDGE_MIN_FRAC = 0.8  # aperture and sky annulus inside the frame on at least this fraction of frames
#: hot pixel of the run's master dark (in detector pixels): above this many sigma of the dark
DARK_HOT_SIGMA = 50.0
EXTENDED_RATIO = 1.5
#: a new source must be this much brighter than the catalogue depth (else a star below Gaia's limit)
DEPTH_MARGIN = 1.0
#: OpenNGC galaxies (catalogs/NGC.csv): a source on a galaxy's core is the galaxy, off the core but
#: inside it a possible supernova (kept, ``in_galaxy``)
OPENNGC_NAME = "NGC.csv"
GALAXY_CORE_ARCSEC = 5.0
#: "elsewhere on the sky": the detector pixel maps this far from the source (an extended galaxy stays clear)
DETECTOR_AWAY_FWHM = 10.0
#: frames where the source's detector pixel looks elsewhere on the sky, needed for the detector test
DETECTOR_TEST_MIN = 5
#: a sky source leaves its detector pixel empty elsewhere (noise alone: ~0 % at S/N 5); a hot or warm
#: pixel is bright there too, also when it flickers
DETECTOR_FRAC_MAX = 0.15


# ---------------------------------------------------------------- per-frame detection
def _detect_one(args: tuple) -> pd.DataFrame:
    fp, t, fwhm = args
    from astropy.io import fits  # noqa: PLC0415
    from photutils.detection import DAOStarFinder  # noqa: PLC0415

    from frame_footprint import aligned_footprint  # noqa: PLC0415
    from frame_metrics import pixel_noise_sigma  # noqa: PLC0415
    from utils import DAO_STAR_FINDER_NO_ROUNDNESS_FILTER  # noqa: PLC0415

    with fits.open(fp, memmap=False) as h:
        img = np.asarray(h[0].data, dtype=np.float64)
        hdr = h[0].header
    cov = aligned_footprint(hdr, img.shape)
    sky = float(np.median(img[cov]))
    sig = pixel_noise_sigma(np.where(cov, img, np.nan))
    if not sig:
        return pd.DataFrame()
    tbl = DAOStarFinder(fwhm=float(fwhm), threshold=DETECT_NSIGMA * sig, n_brightest=None,
                        **DAO_STAR_FINDER_NO_ROUNDNESS_FILTER)(img - sky, mask=~cov)
    if tbl is None or len(tbl) == 0:
        return pd.DataFrame()
    return pd.DataFrame({"t": t, "x": np.asarray(tbl["x_centroid"], float), "y": np.asarray(tbl["y_centroid"], float),
                         "flux": np.asarray(tbl["flux"], float), "sharp": np.asarray(tbl["sharpness"], float)})


def detect_all_frames(frame_paths: list[Path], fwhm: list[float], workers: int) -> pd.DataFrame:
    jobs = [(str(p), t, f) for t, (p, f) in enumerate(zip(frame_paths, fwhm, strict=True))]
    parts = list(map_frames(_detect_one, jobs, workers=workers, stage="transient detection"))
    parts = [p for p in parts if not p.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["t", "x", "y", "flux", "sharp"])


def unmatched_detections(det: pd.DataFrame, known_xy: np.ndarray, fwhm_by_t: np.ndarray) -> pd.DataFrame:
    """Detections farther than max(MATCH_FWHM x FWHM, 2 px) from every known source."""
    from scipy.spatial import cKDTree  # noqa: PLC0415

    if det.empty:
        return det
    tree = cKDTree(known_xy)
    d, _ = tree.query(det[["x", "y"]].to_numpy(float))
    lim = np.maximum(MATCH_FWHM * fwhm_by_t[det["t"].to_numpy(int)], 2.0)
    return det[d > lim].reset_index(drop=True)


def link_detections(un: pd.DataFrame, link_px: float) -> np.ndarray:
    """Friends-of-friends groups (label per detection) within ``link_px``."""
    from scipy.sparse import coo_matrix  # noqa: PLC0415
    from scipy.sparse.csgraph import connected_components  # noqa: PLC0415
    from scipy.spatial import cKDTree  # noqa: PLC0415

    if un.empty:
        return np.zeros(0, int)
    xy = un[["x", "y"]].to_numpy(float)
    pairs = cKDTree(xy).query_pairs(link_px, output_type="ndarray")
    n = len(xy)
    if len(pairs) == 0:
        return np.arange(n)
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    return connected_components(g, directed=False)[1]


def classify_groups(un: pd.DataFrame, labels: np.ndarray, times: np.ndarray, fwhm_med: float) -> tuple[list[dict], list[dict]]:
    """(stationary sources, moving objects) from linked detections."""
    stat, mov = [], []
    if un.empty:
        return stat, mov
    un = un.assign(g=labels)
    for _g, d in un.groupby("g"):
        fr = d["t"].nunique()
        if fr < N_CONSEC_MIN:
            continue
        d = d.sort_values("t")
        tt = times[d["t"].to_numpy(int)]
        x, y = d["x"].to_numpy(float), d["y"].to_numpy(float)
        span = float(tt.max() - tt.min())
        if span > 0 and fr >= 4:
            cx, cy = np.polyfit(tt, x, 1), np.polyfit(tt, y, 1)
            drift = math.hypot(cx[0], cy[0]) * span
            rx, ry = x - np.polyval(cx, tt), y - np.polyval(cy, tt)
            resid = float(np.sqrt(np.median(rx * rx + ry * ry)))
        else:
            drift, resid, cx, cy = 0.0, 0.0, (0.0, float(np.median(x))), (0.0, float(np.median(y)))
        if drift > MOVE_FWHM * fwhm_med and resid < 0.5 * fwhm_med:
            mov.append({"_g": int(_g), "n_frames": int(fr), "t0": int(d["t"].min()), "t1": int(d["t"].max()),
                        "x_t0": float(np.polyval(cx, tt.min())), "y_t0": float(np.polyval(cy, tt.min())),
                        "x_t1": float(np.polyval(cx, tt.max())), "y_t1": float(np.polyval(cy, tt.max())),
                        "rate_px_day": float(math.hypot(cx[0], cy[0])), "resid_px": resid,
                        "jd0": float(tt.min()), "jd1": float(tt.max()), "_vx": float(cx[0]), "_vy": float(cy[0]),
                        "_x0": float(cx[1]), "_y0": float(cy[1])})
        elif drift <= MOVE_FWHM * fwhm_med:
            stat.append({"x": float(np.median(x)), "y": float(np.median(y)), "n_det_frames": int(fr), "origin": "frames"})
    return stat, mov


# ---------------------------------------------------------------- forced photometry
def _measure_one(args: tuple) -> np.ndarray:
    fp, xy, fwhm, k, a_in, a_out, gain, rn = args
    from astropy.io import fits  # noqa: PLC0415
    from photutils.aperture import ApertureStats, CircularAnnulus, CircularAperture, aperture_photometry  # noqa: PLC0415

    from frame_footprint import aligned_footprint  # noqa: PLC0415

    with fits.open(fp, memmap=False) as h:
        img = np.asarray(h[0].data, dtype=np.float64)
        hdr = h[0].header
    cov = aligned_footprint(hdr, img.shape)
    r = k * fwhm
    ap = CircularAperture(xy, r)
    an = CircularAnnulus(xy, a_in * fwhm, a_out * fwhm)
    sky = np.asarray(ApertureStats(img, an, mask=~cov).median, float)
    s = aperture_photometry(img, ap)["aperture_sum"]
    F = np.asarray(s, float) - sky * ap.area
    var = np.clip(F, 0, None) / gain + ap.area * (1 + ap.area / an.area) * (np.clip(sky, 0, None) / gain + (rn / gain) ** 2)
    # centroid offset (intensity-weighted, inside 1 FWHM)
    off = np.full(len(xy), np.nan)
    R = int(math.ceil(fwhm))
    for i, (x, y) in enumerate(xy):
        xi, yi = int(round(x)), int(round(y))
        if not (R <= xi < img.shape[1] - R and R <= yi < img.shape[0] - R):
            continue
        c = img[yi - R: yi + R + 1, xi - R: xi + R + 1] - sky[i]
        w = np.clip(c, 0, None)
        if w.sum() > 0:
            yy, xx = np.mgrid[yi - R: yi + R + 1, xi - R: xi + R + 1]
            off[i] = math.hypot(float((w * xx).sum() / w.sum()) - x, float((w * yy).sum() / w.sum()) - y)
    inside = np.array([bool(cov[int(round(y)), int(round(x))]) if (0 <= int(round(y)) < cov.shape[0] and 0 <= int(round(x)) < cov.shape[1]) else False
                       for x, y in xy])
    return np.c_[F, np.sqrt(var), off, inside.astype(float)]


def forced_photometry_tracks(frame_paths: list[Path], fwhm: np.ndarray, xy_t: np.ndarray, *, k: float, a_in: float,
                             a_out: float, gain: float, rn: float, workers: int) -> np.ndarray:
    """As :func:`forced_photometry` but with positions per frame: ``xy_t`` (n_frames, n_sources, 2)."""
    jobs = [(str(p), xy_t[t], float(f), k, a_in, a_out, gain, rn) for t, (p, f) in enumerate(zip(frame_paths, fwhm, strict=True))]
    res = list(map_frames(_measure_one, jobs, workers=workers, stage="transient photometry"))
    return np.stack(res)


def forced_photometry(frame_paths: list[Path], fwhm: np.ndarray, xy: np.ndarray, *, k: float, a_in: float, a_out: float,
                      gain: float, rn: float, workers: int) -> np.ndarray:
    """(n_frames, n_sources, 4): flux, error, centroid offset (px), inside footprint."""
    jobs = [(str(p), xy, float(f), k, a_in, a_out, gain, rn) for p, f in zip(frame_paths, fwhm, strict=True)]
    res = list(map_frames(_measure_one, jobs, workers=workers, stage="transient photometry"))
    return np.stack(res)


def _runs(mask: np.ndarray) -> tuple[int, int]:
    """(longest run of True, index where it starts)."""
    best, start, cur, cs = 0, -1, 0, 0
    for i, v in enumerate(mask):
        if v:
            if cur == 0:
                cs = i
            cur += 1
            if cur > best:
                best, start = cur, cs
        else:
            cur = 0
    return best, start


def onset_kind(snr: np.ndarray) -> tuple[str, int]:
    """``appeared`` / ``faded`` / ``present`` and the first frame of the longest detected run."""
    det = snr >= SNR_DETECT
    n_run, s0 = _runs(det)
    if n_run < N_CONSEC_MIN:
        return "", -1
    first = int(np.argmax(det))
    last = int(len(det) - 1 - np.argmax(det[::-1]))
    before = snr[:first]
    after = snr[last + 1:]
    nb = max(N_CONSEC_MIN, int(0.1 * len(snr)))
    if before.size >= nb and _absent(before):
        return "appeared", first
    if after.size >= nb and _absent(after):
        return "faded", first
    return "present", first


def _absent(snr: np.ndarray) -> bool:
    """No source in these frames: median S/N < 1.5 and at most 5 % of them at S/N >= 3 (noise alone
    gives 0.1 %; a fixed 'all below 3' fails on a long stretch by chance)."""
    v = snr[np.isfinite(snr)]
    return bool(v.size and float(np.median(v)) < 1.5 and float(np.mean(v >= 3.0)) <= 0.05)


# ---------------------------------------------------------------- main
def run_field_transients(
    *,
    per_frame_dir: Path,
    photometry_dir: Path,
    platesolve_dir: Path,
    cfg: Any,
    frame: pd.DataFrame,
    E: pd.Series,
    zp_g: float,
    k: float,
    gain: float,
    read_noise: float,
    check_catalogs: bool = True,
) -> dict[str, Any]:
    """Run the search for one observation group. ``frame`` (index = time order t; stem, time,
    fwhm, airmass, jd_utc), ``E`` (frame zero point by t) and ``zp_g`` (G = m_inst - E - zp_g for a
    star of solar colour) come from the variability search of the same run."""
    from astropy.io import fits  # noqa: PLC0415
    from astropy.wcs import WCS  # noqa: PLC0415
    from scipy.spatial import cKDTree  # noqa: PLC0415

    phot, ps = Path(photometry_dir), Path(platesolve_dir)
    meta: dict[str, Any] = {"method": "TRANSIENT-2026-10", "n_frames": int(len(frame))}
    stems = list(frame["stem"])
    paths = [Path(per_frame_dir) / f"{s}.fits" for s in stems]
    if not all(p.is_file() for p in paths):
        meta["status"] = "aligned frames missing"
        _write_meta(phot, meta)
        return meta
    fwhm = frame["fwhm"].to_numpy(float)
    fwhm_med = float(np.nanmedian(fwhm))
    times = frame["time"].to_numpy(float)
    from parallel_frames import frame_bytes_of_fits, stage_workers  # noqa: PLC0415

    workers = stage_workers(len(paths), frame_bytes_of_fits(paths[0]), stage="transient search")
    a_in, a_out = float(cfg.annulus_inner_fwhm), float(cfg.annulus_outer_fwhm)

    msh = fits.getheader(ps / "MASTERSTAR.fits")
    wcs = WCS(msh)
    from astropy.wcs.utils import proj_plane_pixel_scales  # noqa: PLC0415

    scale = float(np.mean(np.abs(proj_plane_pixel_scales(wcs)))) * 3600.0  # arcsec / px
    ms = read_vyvar_csv(ps / "masterstars_full_match.csv", low_memory=False)
    ms = ms[np.isfinite(pd.to_numeric(ms["x"], errors="coerce")) & np.isfinite(pd.to_numeric(ms["y"], errors="coerce"))]
    known_xy = ms[["x", "y"]].to_numpy(float)
    cat = read_vyvar_csv(ps / "field_catalog_cone.csv", low_memory=False)
    gx, gy = wcs.all_world2pix(pd.to_numeric(cat["ra_deg"]), pd.to_numeric(cat["dec_deg"]), 0)
    gmag = pd.to_numeric(cat["mag"], errors="coerce").to_numpy(float)
    gtree = cKDTree(np.c_[gx, gy])

    # 1. sources
    det = detect_all_frames(paths, list(fwhm), workers)
    un = unmatched_detections(det, known_xy, fwhm)
    labels = link_detections(un, LINK_FWHM * fwhm_med)
    stat, mov = classify_groups(un, labels, times, fwhm_med)
    mats = [np.array([float(fits.getheader(p)[f"VY_AM{j}"]) for j in range(9)]).reshape(3, 3) for p in paths]
    n_mov0 = len(mov)
    mov = [m for m in mov if not _detector_fixed(un, labels, m, mats, fwhm_med)]
    meta["n_moving_detector_fixed"] = n_mov0 - len(mov)
    meta.update(n_detections=int(len(det)), n_unmatched=int(len(un)), n_stationary_groups=len(stat), n_moving=len(mov))
    if "dao_only_class" in ms.columns:
        uir = ms[ms["dao_only_class"].astype(str) == "unmatched_in_range"]
        for _, r in uir.iterrows():
            stat.append({"x": float(r["x"]), "y": float(r["y"]), "n_det_frames": np.nan, "origin": "masterstar"})
    # merge sources closer than 1 FWHM (a MASTERSTAR source seen on the frames too)
    if stat:
        S = pd.DataFrame(stat)
        lab = link_detections(S, MATCH_FWHM * fwhm_med)
        S = S.assign(g=lab).groupby("g").agg(x=("x", "median"), y=("y", "median"), n_det_frames=("n_det_frames", "max"),
                                            origin=("origin", lambda s: "+".join(sorted(set(s))))).reset_index(drop=True)
    else:
        S = pd.DataFrame(columns=["x", "y", "n_det_frames", "origin"])

    # 2. forced photometry
    rows = []
    lc_dir = phot / "lightcurves"
    lc_dir.mkdir(parents=True, exist_ok=True)
    from run_outputs import remove_files  # noqa: PLC0415

    remove_files(lc_dir, ("transient_*.csv",))
    depth_g = float(np.nanpercentile(gmag, 99.5)) if np.isfinite(gmag).any() else float("nan")
    meta["catalogue_depth_g"] = depth_g
    if len(S):
        xy = S[["x", "y"]].to_numpy(float)
        # empty-sky apertures: the real noise of a forced measurement on these frames (interpolated
        # aligned frames, faint stars below the catalogue), per frame
        rnd = _empty_positions(msh, ms, fwhm_med, a_out, n=300)
        allxy = np.vstack([xy, rnd])
        Fm = forced_photometry(paths, fwhm, allxy, k=k, a_in=a_in, a_out=a_out, gain=gain, rn=read_noise, workers=workers)
        Fe = Fm[:, len(xy):, 0]
        sig_sky = 1.4826 * np.nanmedian(np.abs(Fe - np.nanmedian(Fe, axis=1, keepdims=True)), axis=1)
        meta["empty_sky_noise_ratio"] = float(np.nanmedian(sig_sky / np.nanmedian(Fm[:, len(xy):, 1], axis=1)))
        Fm = Fm[:, : len(xy)]
        Ev = E.reindex(frame.index).to_numpy(float)
        ms_img = fits.getdata(ps / "MASTERSTAR.fits").astype(float)
        from aperture_pertarget import moment_fwhm  # noqa: PLC0415

        ref_fw = _field_moment_fwhm(ms_img, ms, fwhm_med)
        from variability_field import _frame_footprints, _inside_all  # noqa: PLC0415

        quads = _frame_footprints(Path(per_frame_dir), stems)
        r_ann = a_out * float(np.nanmax(fwhm))
        ins_ann = (np.array([_inside_all(xy[:, 0], xy[:, 1], r_ann, [q]) for q in quads]) if len(quads) == len(paths)
                   else np.ones((len(paths), len(xy)), bool))
        dark = _master_dark(ps, per_frame_dir.name if hasattr(per_frame_dir, "name") else Path(per_frame_dir).name, paths[0])
        meta["master_dark"] = None if dark is None else dark.get("file")
        gal = load_galaxies(_catalog_dir(cfg))
        meta["galaxy_catalogue"] = None if gal is None else {"file": OPENNGC_NAME, "n": int(len(gal))}
        ra, dec = wcs.all_pix2world(xy[:, 0], xy[:, 1], 0)
        # detector test: the same DETECTOR pixel on the frames where it looks at another part of the
        # sky - a hot pixel or a reflection fixed in the optics is bright there too, a sky source is not
        snr_all = Fm[:, :, 0] / np.sqrt(sig_sky[:, None] ** 2 + np.clip(Fm[:, :, 0], 0, None) / gain)
        # reference frame: the detected frame of median S/N (the brightest frame may hold a hot pixel
        # that happens to fall on the source)
        peak = np.empty(len(xy), int)
        for i in range(len(xy)):
            si = np.where(Fm[:, i, 3] > 0.5, snr_all[:, i], np.nan)
            dt_ = np.flatnonzero(si >= SNR_DETECT)
            if dt_.size == 0:
                peak[i] = int(np.nanargmax(np.nan_to_num(si, nan=-np.inf)))
            else:
                peak[i] = int(dt_[np.argsort(si[dt_])[dt_.size // 2]])
        inv = [np.linalg.inv(M) for M in mats]
        xy_t = np.empty((len(paths), len(xy), 2))
        for i in range(len(xy)):
            dpx = _apply(inv[peak[i]], xy[i])
            for t in range(len(paths)):
                xy_t[t, i] = _apply(mats[t], dpx)
        Fd = forced_photometry_tracks(paths, fwhm, xy_t, k=k, a_in=a_in, a_out=a_out, gain=gain, rn=read_noise, workers=workers)
        snr_d = Fd[:, :, 0] / np.sqrt(sig_sky[:, None] ** 2 + np.clip(Fd[:, :, 0], 0, None) / gain)
        away = (np.hypot(xy_t[:, :, 0] - xy[None, :, 0], xy_t[:, :, 1] - xy[None, :, 1]) > DETECTOR_AWAY_FWHM * fwhm_med) & (Fd[:, :, 3] > 0.5)
        ktree = cKDTree(known_xy)
        dk, _ = ktree.query(xy_t.reshape(-1, 2))
        away &= (dk.reshape(xy_t.shape[:2]) > 2.0 * fwhm_med)  # not on a known star there
        for i in range(len(S)):
            F, off = Fm[:, i, 0], Fm[:, i, 2]
            ins = (Fm[:, i, 3] > 0.5) & ins_ann[:, i]
            eF = np.sqrt(sig_sky**2 + np.clip(F, 0, None) / gain)
            snr = np.where(ins & (eF > 0), F / eF, np.nan)
            # onset on the frames where the source is measurable (an edge source drifting out of the
            # field must not look like a fading one)
            okf = np.flatnonzero(np.isfinite(snr))
            kind, first = onset_kind(snr[okf]) if okf.size else ("", -1)
            first = int(okf[first]) if first >= 0 else -1
            det_i = snr >= SNR_DETECT
            m_inst = np.where(F > 0, -2.5 * np.log10(np.clip(F, 1e-30, None)), np.nan)
            mag = m_inst - Ev - zp_g
            mag_med = float(np.nanmedian(mag[det_i])) if det_i.any() else float("nan")
            dmin = gtree.query_ball_point(xy[i], max(NEIGHBOUR_ARCSEC / scale, NEIGHBOUR_FWHM * fwhm_med))
            nb = [j for j in dmin if math.isfinite(gmag[j]) and (not math.isfinite(mag_med) or gmag[j] < mag_med + NEIGHBOUR_DMAG)]
            dd_b = np.hypot(gx - xy[i, 0], gy - xy[i, 1])
            r_b = np.maximum(BRIGHT_ARCSEC / scale, BRIGHT_FWHM * fwhm_med * 10.0 ** (BRIGHT_SLOPE * (BRIGHT_G - np.nan_to_num(gmag, nan=99.0))))
            brighter = gmag < (mag_med - BRIGHT_DMAG if math.isfinite(mag_med) else BRIGHT_G)
            bright = list(np.flatnonzero((gmag < BRIGHT_G) & brighter & (dd_b < r_b)))
            hot_dark = dark_hot_pixel(dark, mats, xy[i], peak[i])
            fw_c = moment_fwhm(ms_img, float(xy[i, 0]), float(xy[i, 1]), fwhm_med)
            ext = bool(math.isfinite(fw_c) and math.isfinite(ref_fw) and fw_c > EXTENDED_RATIO * ref_fw)
            off_med = float(np.nanmedian(off[det_i])) if det_i.any() else float("nan")
            n_away = int(away[:, i].sum())
            det_frac = float(np.mean(snr_d[away[:, i], i] >= SNR_DETECT)) if n_away >= DETECTOR_TEST_MIN else float("nan")
            gname, gdist, gsemi = galaxy_near(gal, float(ra[i]), float(dec[i]))
            reasons = []
            if gname and gdist <= max(GALAXY_CORE_ARCSEC, 1.5 * fwhm_med * scale):
                reasons.append(f"galaxy {gname} (OpenNGC)")
            if not kind:
                reasons.append(f"< {N_CONSEC_MIN} consecutive frames at S/N >= {SNR_DETECT:g}")
            if math.isfinite(off_med) and off_med > CENTROID_MAX_FWHM * fwhm_med:
                reasons.append("centroid off the position")
            if math.isfinite(det_frac) and det_frac >= DETECTOR_FRAC_MAX:
                reasons.append("on the same detector pixel elsewhere on the sky (hot pixel / reflection)")
            if math.isfinite(mag_med) and math.isfinite(depth_g) and mag_med > depth_g - DEPTH_MARGIN:
                reasons.append(f"fainter than the catalogue depth - {DEPTH_MARGIN:g} mag")
            if nb:
                reasons.append(f"Gaia star within {max(NEIGHBOUR_ARCSEC, NEIGHBOUR_FWHM * fwhm_med * scale):.1f}\"")
            if bright:
                reasons.append(f"in the halo of a G < {BRIGHT_G:g} star")
            if hot_dark:
                reasons.append("hot pixel of the master dark")
            if ins.mean() < EDGE_MIN_FRAC:
                reasons.append(f"aperture or sky annulus outside the frame on > {100 * (1 - EDGE_MIN_FRAC):.0f} % of frames")
            rows.append({"x": float(xy[i, 0]), "y": float(xy[i, 1]), "ra_deg": float(ra[i]), "dec_deg": float(dec[i]),
                         "origin": S.loc[i, "origin"], "kind": kind, "first_frame": int(first),
                         "first_time": float(times[first]) if first >= 0 else float("nan"),
                         "n_frames_snr5": int(det_i.sum()), "longest_run": int(_runs(det_i)[0]),
                         "snr_max": float(np.nanmax(snr)) if np.isfinite(snr).any() else float("nan"),
                         "mag_g_est": mag_med, "centroid_off_px": off_med, "extended": ext,
                         "fwhm_ratio": float(fw_c / ref_fw) if math.isfinite(fw_c) and ref_fw > 0 else float("nan"),
                         "n_gaia_close": len(nb), "n_bright_near": len(bright),
                         "detector_pixel_frac": det_frac, "detector_test_frames": n_away,
                         "galaxy": gname, "galaxy_dist_arcsec": gdist,
                         "in_galaxy": bool(gname) and gdist <= max(gsemi, GALAXY_CORE_ARCSEC),
                         "candidate": not reasons, "reject_reason": "; ".join(reasons)})
            if not reasons:
                rows[-1]["_lc"] = pd.DataFrame({"time": times, "mag": mag, "err": 1.0857 * eF / np.where(F > 0, F, np.nan),
                                                "snr": snr, "fwhm": fwhm, "airmass": frame["airmass"].to_numpy(float)})
    T = pd.DataFrame(rows)
    if not T.empty:
        c = T[T["candidate"]].sort_values("snr_max", ascending=False).index
        T["label"] = ""
        T.loc[c, "label"] = [f"T{j + 1}" for j in range(len(c))]
        for j in c:
            T.at[j, "_lc"].to_csv(lc_dir / f"transient_{T.at[j, 'label']}.csv", index=False)
        T = T.drop(columns=["_lc"], errors="ignore")
        T["catalogs"] = ""
        if check_catalogs and len(c):
            from variability_field import _catalog_check  # noqa: PLC0415

            T.loc[c, "catalogs"] = [_catalog_check(pd.Series({"ra_deg": T.at[j, "ra_deg"], "dec_deg": T.at[j, "dec_deg"],
                                                              "mag": T.at[j, "mag_g_est"]}), cfg) for j in c]
    T.to_csv(phot / TRANSIENT_CSV, index=False)

    # 3. minor bodies
    Mv = pd.DataFrame(mov)
    bodies = _minor_bodies(cfg, wcs, msh, frame, ps)
    if not Mv.empty:
        Mv["ra_t0"], Mv["dec_t0"] = wcs.all_pix2world(Mv["x_t0"], Mv["y_t0"], 0)
        Mv["rate_arcsec_h"] = Mv["rate_px_day"] * scale / 24.0
        Mv["minor_body"] = ""
    B = bodies.get("table", pd.DataFrame())
    if not B.empty:
        B["detected_frames"] = 0
        B["moving_object"] = -1
        dtree = cKDTree(det[["x", "y"]].to_numpy(float)) if not det.empty else None
        tol = max(3.0, 60.0 / scale)
        for bi, br in B.iterrows():
            px = np.asarray(br["_x"], float)
            py = np.asarray(br["_y"], float)
            n = 0
            for t in range(len(frame)):
                if dtree is None or not (0 <= px[t] < msh["NAXIS1"] and 0 <= py[t] < msh["NAXIS2"]):
                    continue
                hit = dtree.query_ball_point((px[t], py[t]), tol)
                n += int(any(int(det.iloc[h]["t"]) == t for h in hit))
            B.at[bi, "detected_frames"] = n
            if not Mv.empty:
                tm = 0.5 * (times[0] + times[-1])
                xb, yb = np.interp(tm, times, px), np.interp(tm, times, py)
                xm = Mv["_x0"] + Mv["_vx"] * tm
                ym = Mv["_y0"] + Mv["_vy"] * tm
                dd = np.hypot(xm - xb, ym - yb)
                j = int(np.argmin(dd))
                if dd.iloc[j] < tol:
                    B.at[bi, "moving_object"] = j
                    Mv.at[j, "minor_body"] = str(br["name"])
        B = B.drop(columns=["_x", "_y"])
    B.to_csv(phot / BODIES_CSV, index=False)
    (Mv.drop(columns=[c for c in Mv.columns if c.startswith("_")]) if not Mv.empty else Mv).to_csv(phot / MOVING_CSV, index=False)
    meta.update(n_sources=int(len(T)), n_candidates=int(T["candidate"].sum()) if not T.empty else 0,
                n_minor_bodies=int(len(B)), minor_bodies=bodies.get("info"), status="ok",
                pixel_scale_arcsec=scale, fwhm_px=fwhm_med)
    _write_meta(phot, meta)
    LOGGER.info("[TRANSIENT] %d detections, %d unmatched, %d sources, %d candidates, %d moving, %d minor bodies V<=20 in field",
                meta["n_detections"], meta["n_unmatched"], meta["n_sources"], meta["n_candidates"], len(Mv), len(B))
    return meta


def _detector_fixed(un: pd.DataFrame, labels: np.ndarray, m: dict, mats: list[np.ndarray], fwhm: float) -> bool:
    """A track that stays on one detector pixel while the field moves: hot pixel or a reflection
    fixed in the optics, not an object on the sky."""
    d = un[labels == m["_g"]]
    pts = np.array([_apply(np.linalg.inv(mats[int(t)]), (x, y)) for t, x, y in zip(d["t"], d["x"], d["y"], strict=True)])
    return bool(np.hypot(*pts.std(axis=0)) < 0.5 * fwhm)


def _empty_positions(msh: Any, ms: pd.DataFrame, fwhm: float, a_out: float, n: int) -> np.ndarray:
    """``n`` positions >= a_out FWHM from every MASTERSTAR source and the frame edge (fixed seed)."""
    from scipy.spatial import cKDTree  # noqa: PLC0415

    nx, ny = int(msh["NAXIS1"]), int(msh["NAXIS2"])
    rng = np.random.default_rng(20261025)
    tree = cKDTree(ms[["x", "y"]].to_numpy(float))
    m = (a_out + 1.0) * fwhm
    out: list[np.ndarray] = []
    for _ in range(50):
        c = np.c_[rng.uniform(m, nx - m, 4 * n), rng.uniform(m, ny - m, 4 * n)]
        d, _i = tree.query(c)
        out.extend(c[d > 3.0 * fwhm])
        if len(out) >= n:
            break
    return np.asarray(out[:n], float).reshape(-1, 2)


def _master_dark(ps: Path, obs_group: str, frame0: Path) -> dict[str, Any] | None:
    """The run's master dark (draft manifest ``dark_by_obs_key``), binned to the lights' pixels the way
    the calibration did (sum), with its median and robust sigma. None when not found."""
    import re  # noqa: PLC0415

    from astropy.io import fits  # noqa: PLC0415

    man = None
    for p in (ps.parent.parent / "draft_manifest.json", ps.parent / "draft_manifest.json"):
        if p.is_file():
            man = p
            break
    if man is None:
        return None
    try:
        calib = str(json.loads(man.read_text(encoding="utf-8")).get("paths", {}).get("calib", ""))
    except (OSError, ValueError):
        return None
    m = re.search(r"dark_by_obs_key=\{([^}]*)\}", calib)
    if not m:
        return None
    key = obs_group.replace("_", "|")
    cands = re.findall(r"'([^']+)':\s*'([^']+)'", m.group(1))
    fp = next((Path(v) for k, v in cands if k == key), None) or (Path(cands[0][1]) if cands else None)
    if fp is None or not fp.is_file():
        return None
    try:
        dk = np.asarray(fits.getdata(fp), dtype=np.float64)
        h0 = fits.getheader(frame0)
        ny, nx = int(h0["NAXIS2"]), int(h0["NAXIS1"])  # aligned frames keep the raw frame size
    except (OSError, ValueError):
        return None
    if dk.ndim != 2:
        return None
    return {"file": str(fp), "data": dk, "raw_shape": (ny, nx)}


def dark_hot_pixel(dark: dict[str, Any] | None, mats: list[np.ndarray], xy: np.ndarray, t_ref: int) -> bool:
    """True when the source's detector pixel (inverse alignment of frame ``t_ref``) is a hot pixel of the
    master dark: > DARK_HOT_SIGMA robust sigma within 1 px. The dark is binned to the lights (sum) when it
    is larger by an integer factor."""
    if dark is None:
        return False
    dk = dark["data"]
    if "binned" not in dark:
        ny, nx = dark["raw_shape"]
        f = int(round(dk.shape[0] / ny)) if ny else 1
        if f > 1 and abs(dk.shape[1] / f - nx) < f + 1:
            h, w = (dk.shape[0] // f) * f, (dk.shape[1] // f) * f
            dk = dk[:h, :w].reshape(h // f, f, w // f, f).sum(axis=(1, 3))
        med = float(np.median(dk))
        dark["binned"] = dk
        dark["med"] = med
        dark["sig"] = 1.4826 * float(np.median(np.abs(dk - med))) or 1.0
    dk = dark["binned"]
    xd, yd = _apply(np.linalg.inv(mats[int(t_ref)]), xy)
    xi, yi = int(round(xd)), int(round(yd))
    if not (1 <= xi < dk.shape[1] - 1 and 1 <= yi < dk.shape[0] - 1):
        return False
    return bool((dk[yi - 1:yi + 2, xi - 1:xi + 2].max() - dark["med"]) / dark["sig"] > DARK_HOT_SIGMA)


def _apply(M: np.ndarray, xy) -> np.ndarray:
    v = M @ np.array([float(xy[0]), float(xy[1]), 1.0])
    return v[:2] / v[2]


def _field_moment_fwhm(img: np.ndarray, ms: pd.DataFrame, fwhm: float) -> float:
    from aperture_pertarget import moment_fwhm  # noqa: PLC0415

    st = ms[pd.to_numeric(ms.get("catalog_id"), errors="coerce").notna()] if "catalog_id" in ms.columns else ms
    st = st.sample(n=min(200, len(st)), random_state=0) if len(st) else st
    v = [moment_fwhm(img, float(r["x"]), float(r["y"]), fwhm) for _, r in st.iterrows()]
    v = [x for x in v if math.isfinite(x)]
    return float(np.median(v)) if v else float("nan")


def _minor_bodies(cfg: Any, wcs: Any, msh: Any, frame: pd.DataFrame, ps: Path) -> dict[str, Any]:
    """Minor planets / comets brighter than V 20 in the frame (empty without the MPC files)."""
    from minor_bodies import bodies_in_field, load_orbits  # noqa: PLC0415

    cat_dir = _catalog_dir(cfg)
    try:
        mp, cm, info = load_orbits(cat_dir)
    except Exception as exc:  # noqa: BLE001
        return {"info": {"status": f"orbits not read: {exc}"}}
    if mp is None and cm is None:
        return {"info": {"status": f"no MPCORB.DAT(.gz) / CometEls.txt in {cat_dir}"}, **info}
    nx, ny = int(msh["NAXIS1"]), int(msh["NAXIS2"])
    c = wcs.all_pix2world([[nx / 2, ny / 2]], 0)[0]
    corners = wcs.all_pix2world([[0, 0], [nx, 0], [0, ny], [nx, ny]], 0)
    from minor_bodies import _sep_deg  # noqa: PLC0415

    rad = float(np.max(_sep_deg(corners[:, 0], corners[:, 1], c[0], c[1])))
    site = _site(ps)
    jd = frame["jd_utc"].to_numpy(float) if "jd_utc" in frame else frame["time"].to_numpy(float)
    step = max(1, len(jd) // 24)
    jd_s = jd[::step] if jd[::step][-1] == jd[-1] else np.r_[jd[::step], jd[-1]]
    b = bodies_in_field(mp, cm, jd_utc=jd_s, ra_c=float(c[0]), dec_c=float(c[1]), radius_deg=rad, site=site)
    out = []
    for name, g in b.groupby("name"):
        x, y = wcs.all_world2pix(g["ra_deg"], g["dec_deg"], 0)
        xi, yi = np.interp(jd, g["jd"], x), np.interp(jd, g["jd"], y)
        inside = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
        if not inside.any():
            continue
        out.append({"name": name, "kind": g["kind"].iloc[0], "mag_v": float(g["mag"].median()),
                    "rate_arcsec_h": float(g["rate_arcsec_h"].iloc[0]), "r_au": float(g["r_au"].median()),
                    "delta_au": float(g["delta_au"].median()), "ra_mid": float(np.interp(np.median(jd), g["jd"], g["ra_deg"])),
                    "dec_mid": float(np.interp(np.median(jd), g["jd"], g["dec_deg"])), "frames_in_field": int(inside.sum()),
                    "_x": xi, "_y": yi})
    tab = pd.DataFrame(out)
    if not tab.empty:
        tab = tab.sort_values("mag_v").reset_index(drop=True)
    return {"table": tab, "info": {"status": "ok", "radius_deg": rad, "site": site, **info}}


def _catalog_dir(cfg: Any) -> Path:
    """The local catalogue folder (the one that holds ``objects_db_path``)."""
    return Path(str(getattr(cfg, "objects_db_path", "") or "catalogs/vyvar_objects.db")).parent


def load_galaxies(cat_dir: Path) -> pd.DataFrame | None:
    """Galaxies of OpenNGC (``NGC.csv``, https://github.com/mattiaverga/OpenNGC, CC-BY-SA 4.0):
    ra, dec (deg), semi-major axis (arcsec), name. None when the file is not there."""
    fp = Path(cat_dir) / OPENNGC_NAME
    if not fp.is_file():
        return None
    g = pd.read_csv(fp, sep=";", usecols=["Name", "Type", "RA", "Dec", "MajAx"], low_memory=False)
    g = g[g["Type"].astype(str).str.startswith("G") & g["RA"].notna() & g["Dec"].notna()].copy()

    def _sex(v: str, hours: bool) -> float:
        sgn = -1.0 if str(v).strip().startswith("-") else 1.0
        a, b, c = (abs(float(x)) for x in str(v).replace("+", "").replace("-", "").split(":"))
        d = a + b / 60.0 + c / 3600.0
        return sgn * d * (15.0 if hours else 1.0)

    g["ra"] = [_sex(v, True) for v in g["RA"]]
    g["dec"] = [_sex(v, False) for v in g["Dec"]]
    g["semi_arcsec"] = pd.to_numeric(g["MajAx"], errors="coerce").fillna(0.5) * 30.0
    return g[["Name", "ra", "dec", "semi_arcsec"]].reset_index(drop=True)


def galaxy_near(gal: pd.DataFrame | None, ra: float, dec: float) -> tuple[str, float, float]:
    """(name, distance arcsec, semi-major axis arcsec) of the nearest catalogued galaxy within
    3 of its semi-major axes; ("", nan, nan) otherwise."""
    if gal is None or gal.empty:
        return "", float("nan"), float("nan")
    from minor_bodies import _sep_deg  # noqa: PLC0415

    d = _sep_deg(gal["ra"].to_numpy(float), gal["dec"].to_numpy(float), ra, dec) * 3600.0
    j = int(np.argmin(d))
    semi = float(gal.at[j, "semi_arcsec"])
    if d[j] > max(3.0 * semi, GALAXY_CORE_ARCSEC):
        return "", float("nan"), float("nan")
    return str(gal.at[j, "Name"]), float(d[j]), semi


def _site(ps: Path) -> tuple[float | None, float | None, float | None]:
    """Observer latitude, longitude, height of the run (draft manifest)."""
    for p in (ps.parent / "draft_manifest.json", ps.parent.parent / "draft_manifest.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        loc = d.get("observer_location") or d.get("site") or {}
        lat = loc.get("lat") if isinstance(loc, dict) else None
        lon = loc.get("lon") if isinstance(loc, dict) else None
        h = loc.get("height_m", loc.get("alt_m")) if isinstance(loc, dict) else None
        if lat is not None and lon is not None:
            return float(lat), float(lon), float(h) if h is not None else 0.0
    return None, None, None


def load_transient_candidates(photometry_dir: Path) -> pd.DataFrame:
    """The kept candidates (T1, T2, ...) of a setup; empty without the search."""
    fp = Path(photometry_dir) / TRANSIENT_CSV
    if not fp.is_file() or fp.stat().st_size <= 2:
        return pd.DataFrame()
    d = pd.read_csv(fp, low_memory=False)
    return d[d["candidate"].astype(str).str.lower() == "true"].reset_index(drop=True) if "candidate" in d else d


def _write_meta(phot: Path, meta: dict[str, Any]) -> None:
    def _c(o: Any) -> Any:
        if isinstance(o, dict):
            return {kk: _c(v) for kk, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_c(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return float(o) if math.isfinite(float(o)) else None
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return o

    (Path(phot) / META_JSON).write_text(json.dumps(_c(meta), indent=2), encoding="utf-8")
