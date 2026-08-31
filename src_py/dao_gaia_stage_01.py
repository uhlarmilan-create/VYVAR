#!/usr/bin/env python3
"""DAO-GAIA-STAGE-01 sandbox: single-pass detection + 3 px greedy match + overlay/metrics.

Read-only on draft 516 platesolved products. Not imported by production.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from photutils.detection import DAOStarFinder
from scipy.ndimage import binary_dilation
try:
    from scipy.ndimage import disk  # type: ignore[attr-defined]
except ImportError:
    from skimage.morphology import disk
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src_py"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import AppConfig  # noqa: E402
from dao_gaia_common import _is_corner  # noqa: E402
from masterstar_gaia_accounting import (  # noqa: E402
    SOURCE_BLENDED,
    SOURCE_EDGE,
    SOURCE_SATURATED,
    SOURCE_TOO_FAINT,
    _dao_pass2_annulus_stats,
    annotate_blended_groups,
    lock_existing_and_leftover_assign,
)
from plain_stats import plain_mean_med_std, sky_mad_sigma_adu  # noqa: E402
from pipeline import _query_gaia_local  # noqa: E402

DRAFT = REPO / "Archive" / "Drafts" / "draft_000516"
PS_DIR = DRAFT / "platesolve" / "NoFilter_60_2"
LIGHTS_DIR = DRAFT / "detrended_aligned" / "lights" / "NoFilter_60_2"
EMPTY_SKY_CSV = REPO / "dev" / "results" / "context" / "session_20260819_msgaia01" / "empty_positions_main.csv"
DEFAULT_CTX = REPO / "dev" / "results" / "context" / "session_20260819_daostage01"

FRAMES: list[tuple[str, Path]] = [
    ("MASTERSTAR", PS_DIR / "MASTERSTAR.fits"),
    ("Light_001", LIGHTS_DIR / "BO_CVn_Light_001.fits"),
    ("Light_076", LIGHTS_DIR / "BO_CVn_Light_076.fits"),
    ("Light_148", LIGHTS_DIR / "BO_CVn_Light_148.fits"),
]

THRESHOLD_SIGMAS = [3.0, 3.5, 3.8, 4.5, 5.0]
MATCH_RADIUS_PX = 3.0
EDGE_MARGIN_PX = 50.0
CORNER_MARGIN_PX = 120.0
TARGET_DEPTH_G = 15.0
OVERLAY_G_MAX = 16.0
RED_X_G_MAX = 14.0
GAIA_QUERY_G = 17.5

SHARPNESS_PRODUCTION = (0.0, 2.0)
SHARPNESS_OPEN = (0.0, 10.0)


@dataclass
class SkyEstimate:
    sky_sigma_clipped: float
    sky_median_clipped: float
    sky_mad_sigma: float
    sky_median_mad: float
    local_annulus_std_p50: float
    local_annulus_std_p16: float
    local_annulus_std_p84: float
    rms_conv: float


@dataclass
class IterationConfig:
    iter_id: str
    threshold_sigma: float
    sharpness_range: tuple[float, float]
    fwhm_px: float


@dataclass
class FrameMetrics:
    frame: str
    n_det: int
    n_gaia_onchip_le16: int
    g1_le13: float
    g1_le145: float
    g1_curve: dict[str, float]
    g2_empty_false_accept: float
    g3_spurious_frac: float
    n_spurious: int
    g4_unnamed: int
    state_counts: dict[str, int]
    n_bright_killed_by_sharpness: int
    n_bright_gaia_le13: int
    sky: SkyEstimate
    threshold_adu: float
    runtime_s: float


@dataclass
class SweepRow:
    iter_id: str
    threshold_sigma: float
    sharpness_lo: float
    sharpness_hi: float
    frame: str
    g1_le13: float
    g1_le145: float
    g2: float
    g3: float
    g4_unnamed: int
    n_det: int
    sky_sigma: float
    local_annulus_p50: float
    threshold_adu: float
    runtime_s: float
    verdict: str


def _wcs_from_hdr(hdr: fits.Header) -> WCS:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return WCS(hdr)


def load_frame(path: Path) -> tuple[np.ndarray, np.ndarray, fits.Header, WCS, float, int, int]:
    with fits.open(path, memmap=False) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float32)
        hdr = hdul[0].header
    fwhm = float(hdr.get("VY_FWHM", 5.3))
    if not math.isfinite(fwhm) or fwhm <= 0:
        fwhm = 5.3
    _, med, _ = plain_mean_med_std(data, sigma=3.0, maxiters=3)
    data0 = np.nan_to_num((data - med).astype(np.float32), nan=0.0)
    wcs = _wcs_from_hdr(hdr)
    h, w = data0.shape
    return data, data0, hdr, wcs, fwhm, w, h


def _gaia_on_chip(wcs: WCS, wpx: int, h: int, gaia_db: Path) -> pd.DataFrame:
    import astropy.units as u

    center = wcs.pixel_to_world(wpx / 2.0, h / 2.0)
    radius = max(float(wcs.proj_plane_pixel_scales().max()) * max(wpx, h) / 2.0 * 1.2, 0.25)
    radius_deg = float(radius.to(u.deg)) if hasattr(radius, "to") else float(radius)

    if not hasattr(radius, "to"):
        radius_deg = max(
            float(wcs.proj_plane_pixel_scales()[0]) * wpx / 2.0 / 3600.0 * 1.5,
            0.3,
        )
    df = _query_gaia_local(
        center=center,
        radius_deg=radius_deg,
        gaia_db_path=gaia_db,
        max_mag=GAIA_QUERY_G,
        max_rows=200_000,
    )
    if df.empty:
        return df
    ra = pd.to_numeric(df["ra_deg"], errors="coerce").to_numpy(dtype=np.float64)
    de = pd.to_numeric(df["dec_deg"], errors="coerce").to_numpy(dtype=np.float64)
    ok = np.isfinite(ra) & np.isfinite(de)
    x, y = wcs.world_to_pixel_values(ra[ok], de[ok])
    sub = df.loc[ok].copy().reset_index(drop=True)
    sub["x_gaia"] = x
    sub["y_gaia"] = y
    sub["g_mag"] = pd.to_numeric(sub.get("mag"), errors="coerce")
    inb = (
        (sub["x_gaia"] >= 0)
        & (sub["x_gaia"] < float(wpx))
        & (sub["y_gaia"] >= 0)
        & (sub["y_gaia"] < float(h))
    )
    return sub.loc[inb].reset_index(drop=True)


def _star_mask_from_gaia(data0: np.ndarray, gaia_df: pd.DataFrame, fwhm_px: float) -> np.ndarray:
    h, w = data0.shape
    mask = np.zeros((h, w), dtype=bool)
    if gaia_df is None or gaia_df.empty:
        return mask
    r = max(int(round(2.0 * float(fwhm_px))), 3)
    d = disk(r)
    dh, dw = d.shape
    gx = pd.to_numeric(gaia_df["x_gaia"], errors="coerce").to_numpy(dtype=np.float64)
    gy = pd.to_numeric(gaia_df["y_gaia"], errors="coerce").to_numpy(dtype=np.float64)
    for x, y in zip(gx, gy, strict=False):
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        ix, iy = int(round(x)), int(round(y))
        y0, y1 = max(0, iy - dh // 2), min(h, iy + dh // 2 + 1)
        x0, x1 = max(0, ix - dw // 2), min(w, ix + dw // 2 + 1)
        sy0 = max(0, dh // 2 - iy)
        sx0 = max(0, dw // 2 - ix)
        sy1 = sy0 + (y1 - y0)
        sx1 = sx0 + (x1 - x0)
        mask[y0:y1, x0:x1] |= d[sy0:sy1, sx0:sx1].astype(bool)
    return mask


def estimate_sky(
    data0: np.ndarray,
    *,
    gaia_df: pd.DataFrame,
    fwhm_px: float,
    rng: np.random.Generator,
) -> SkyEstimate:
    from pipeline import _dao_convolved_background_rms_adu

    star_mask = _star_mask_from_gaia(data0, gaia_df, fwhm_px)
    bg = data0[~star_mask]
    bg_fin = bg[np.isfinite(bg)]
    if bg_fin.size < 1000:
        bg_fin = data0.ravel()
    med_clip, _, sig_clip = sigma_clipped_stats(bg_fin, sigma=3.0, maxiters=3)
    sky_med_mad, sky_sig_mad = sky_mad_sigma_adu(data0, mask=star_mask)
    rms_conv, _ = _dao_convolved_background_rms_adu(data0, fwhm_px=float(fwhm_px))

    ann_stds: list[float] = []
    if gaia_df is not None and not gaia_df.empty:
        idx = rng.choice(len(gaia_df), size=min(400, len(gaia_df)), replace=False)
        for j in idx:
            xg = float(gaia_df.iloc[j]["x_gaia"])
            yg = float(gaia_df.iloc[j]["y_gaia"])
            _, sd = _dao_pass2_annulus_stats(data0, xg, yg)
            if math.isfinite(sd) and sd > 0:
                ann_stds.append(float(sd))
    if ann_stds:
        p16, p50, p84 = np.percentile(ann_stds, [16, 50, 84]).tolist()
    else:
        p16 = p50 = p84 = float("nan")

    return SkyEstimate(
        sky_sigma_clipped=float(sig_clip),
        sky_median_clipped=float(med_clip),
        sky_mad_sigma=float(sky_sig_mad),
        sky_median_mad=float(sky_med_mad),
        local_annulus_std_p50=float(p50),
        local_annulus_std_p16=float(p16),
        local_annulus_std_p84=float(p84),
        rms_conv=float(rms_conv),
    )


def run_dao(
    data0: np.ndarray,
    *,
    fwhm_px: float,
    threshold_adu: float,
    sharpness_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finder = DAOStarFinder(
        fwhm=float(fwhm_px),
        threshold=float(threshold_adu),
        roundness_range=(-1.0e9, 1.0e9),
        sharpness_range=sharpness_range,
        min_separation=0.0,
        n_brightest=None,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tbl = finder(data0)
    if tbl is None or len(tbl) == 0:
        return (
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
        )
    x = np.asarray(tbl["x_centroid"], dtype=np.float64)
    y = np.asarray(tbl["y_centroid"], dtype=np.float64)
    peak = (
        np.asarray(tbl["peak"], dtype=np.float64)
        if "peak" in tbl.colnames
        else np.asarray(tbl["flux"], dtype=np.float64)
    )
    return x, y, peak


def _saturation_limit(hdr: fits.Header) -> float:
    for key in ("SATURATE", "VY_SATURATE", "HISTCUTLO"):
        if key in hdr:
            try:
                v = float(hdr[key])
                if math.isfinite(v) and v > 0:
                    return v
            except (TypeError, ValueError):
                pass
    return 60000.0


def _peak_at(data0: np.ndarray, x: float, y: float, r: int = 3) -> float:
    h, w = data0.shape
    ix, iy = int(round(x)), int(round(y))
    x0, x1 = max(0, ix - r), min(w, ix + r + 1)
    y0, y1 = max(0, iy - r), min(h, iy + r + 1)
    patch = data0[y0:y1, x0:x1]
    return float(np.max(patch)) if patch.size else float("nan")


def assign_states(
    gaia_df: pd.DataFrame,
    gaia_owner: np.ndarray,
    *,
    data0: np.ndarray,
    hdr: fits.Header,
    fwhm_px: float,
    wpx: int,
    h: int,
    depth_g: float,
) -> pd.DataFrame:
    gdf = gaia_df.copy()
    gdf = annotate_blended_groups(gdf, gaia_owner, fwhm_px=fwhm_px)
    sat_lim = _saturation_limit(hdr) * 0.999
    states: list[str] = []
    for j in range(len(gdf)):
        xg = float(gdf.iloc[j]["x_gaia"])
        yg = float(gdf.iloc[j]["y_gaia"])
        gmag = float(gdf.iloc[j]["g_mag"]) if pd.notna(gdf.iloc[j]["g_mag"]) else float("nan")
        owner = int(gaia_owner[j]) if j < len(gaia_owner) else -1
        blend_gid = str(gdf.iloc[j].get("blend_group_id", "") or "")
        if (
            xg < EDGE_MARGIN_PX
            or yg < EDGE_MARGIN_PX
            or xg >= float(wpx) - EDGE_MARGIN_PX
            or yg >= float(h) - EDGE_MARGIN_PX
        ):
            states.append(SOURCE_EDGE)
        elif math.isfinite(gmag) and gmag > float(depth_g):
            states.append(SOURCE_TOO_FAINT)
        elif owner >= 0:
            states.append("DETECTED")
        elif blend_gid:
            states.append(SOURCE_BLENDED)
        elif _peak_at(data0, xg, yg) >= sat_lim:
            states.append(SOURCE_SATURATED)
        elif math.isfinite(gmag) and gmag <= float(depth_g):
            states.append(SOURCE_TOO_FAINT)
        else:
            states.append(SOURCE_TOO_FAINT)
    gdf["source_state"] = states
    return gdf


def completeness_curve(gaia_df: pd.DataFrame, gaia_owner: np.ndarray, g_max: float = 16.0) -> dict[str, float]:
    gm = pd.to_numeric(gaia_df["g_mag"], errors="coerce").to_numpy(dtype=np.float64)
    out: dict[str, float] = {}
    bins = np.arange(8.0, g_max + 0.25, 0.5)
    for lo in bins:
        hi = lo + 0.5
        sel = np.isfinite(gm) & (gm >= lo) & (gm < hi)
        n = int(sel.sum())
        if n == 0:
            continue
        matched = sum(1 for j in np.where(sel)[0] if int(gaia_owner[j]) >= 0)
        out[f"{lo:.1f}-{hi:.1f}"] = matched / n
    return out


def g1_at_mag(gaia_df: pd.DataFrame, gaia_owner: np.ndarray, mag_cut: float) -> float:
    gm = pd.to_numeric(gaia_df["g_mag"], errors="coerce").to_numpy(dtype=np.float64)
    sel = np.isfinite(gm) & (gm <= float(mag_cut))
    n = int(sel.sum())
    if n == 0:
        return float("nan")
    matched = sum(1 for j in np.where(sel)[0] if int(gaia_owner[j]) >= 0)
    return matched / n


def g2_empty_false_accept(
    det_x: np.ndarray,
    det_y: np.ndarray,
    empty_df: pd.DataFrame,
    frame_label: str,
    *,
    match_radius_px: float | None = None,
) -> float:
    match_r = float(MATCH_RADIUS_PX if match_radius_px is None else match_radius_px)
    if empty_df is None or empty_df.empty or "frame" not in empty_df.columns:
        return float("nan")
    sub = empty_df[empty_df["frame"].astype(str).str.contains(frame_label.split("_")[-1] if "Light" in frame_label else "MASTERSTAR")]
    if sub.empty and frame_label == "MASTERSTAR":
        sub = empty_df[empty_df["frame"].astype(str) == "MASTERSTAR"]
    elif sub.empty:
        sub = empty_df[empty_df["frame"].astype(str).str.contains(frame_label.replace("Light_", "BO_CVn_Light_"))]
    if sub.empty:
        return float("nan")
    if det_x.size == 0:
        return 0.0
    tree = cKDTree(np.column_stack([det_x, det_y]))
    hit = 0
    for _, row in sub.iterrows():
        d, _ = tree.query([float(row["x"]), float(row["y"])], distance_upper_bound=match_r)
        if math.isfinite(float(d)) and float(d) <= match_r:
            hit += 1
    return hit / len(sub)


def g3_spurious(
    det_x: np.ndarray,
    det_y: np.ndarray,
    gaia_df: pd.DataFrame,
    *,
    wpx: int,
    h: int,
    match_radius_px: float | None = None,
) -> tuple[float, int]:
    match_r = float(MATCH_RADIUS_PX if match_radius_px is None else match_radius_px)
    if det_x.size == 0:
        return 0.0, 0
    gx = pd.to_numeric(gaia_df["x_gaia"], errors="coerce").to_numpy(dtype=np.float64)
    gy = pd.to_numeric(gaia_df["y_gaia"], errors="coerce").to_numpy(dtype=np.float64)
    tree = cKDTree(np.column_stack([gx, gy]))
    spurious = 0
    considered = 0
    for x, y in zip(det_x, det_y, strict=False):
        if _is_corner(float(x), float(y), wpx, h):
            continue
        considered += 1
        d, _ = tree.query([float(x), float(y)], distance_upper_bound=match_r)
        if not math.isfinite(float(d)) or float(d) > match_r:
            spurious += 1
    frac = spurious / considered if considered else 0.0
    return frac, spurious


def bright_sharpness_kill_count(
    data0: np.ndarray,
    gaia_df: pd.DataFrame,
    *,
    fwhm_px: float,
    sky_sigma: float,
    threshold_sigma: float,
) -> tuple[int, int]:
    gm = pd.to_numeric(gaia_df["g_mag"], errors="coerce").to_numpy(dtype=np.float64)
    sel = np.isfinite(gm) & (gm <= 13.0)
    idx = np.where(sel)[0]
    if idx.size == 0:
        return 0, 0
    thr = max(float(threshold_sigma) * float(sky_sigma), 1e-6)
    open_x, open_y, _ = run_dao(
        data0, fwhm_px=fwhm_px, threshold_adu=thr, sharpness_range=SHARPNESS_OPEN
    )
    prod_x, prod_y, _ = run_dao(
        data0, fwhm_px=fwhm_px, threshold_adu=thr, sharpness_range=SHARPNESS_PRODUCTION
    )
    if open_x.size == 0:
        return 0, int(idx.size)
    tree_open = cKDTree(np.column_stack([open_x, open_y]))
    tree_prod = cKDTree(np.column_stack([prod_x, prod_y])) if prod_x.size else None
    killed = 0
    for j in idx:
        xg = float(gaia_df.iloc[j]["x_gaia"])
        yg = float(gaia_df.iloc[j]["y_gaia"])
        d_open, _ = tree_open.query([xg, yg], distance_upper_bound=MATCH_RADIUS_PX)
        if not math.isfinite(float(d_open)) or float(d_open) > MATCH_RADIUS_PX:
            continue
        if tree_prod is None:
            killed += 1
            continue
        d_prod, _ = tree_prod.query([xg, yg], distance_upper_bound=MATCH_RADIUS_PX)
        if not math.isfinite(float(d_prod)) or float(d_prod) > MATCH_RADIUS_PX:
            killed += 1
    return killed, int(idx.size)


def asinh_rgb(data0: np.ndarray) -> np.ndarray:
    pos = np.clip(data0, 0, None)
    scale = float(np.percentile(pos[pos > 0], 99.5)) if np.any(pos > 0) else 1.0
    scale = max(scale, 1.0)
    return np.arcsinh(pos / scale) / np.arcsinh(1.0)


def render_overlay(
    data0: np.ndarray,
    gaia_df: pd.DataFrame,
    gaia_owner: np.ndarray,
    det_x: np.ndarray,
    det_y: np.ndarray,
    *,
    out_path: Path,
    title: str,
    crop: tuple[int, int, int, int] | None = None,
) -> None:
    rgb = asinh_rgb(data0)
    if crop is not None:
        x0, y0, cw, ch = crop
        x1, y1 = x0 + cw, y0 + ch
        rgb = rgb[y0:y1, x0:x1]
        ox, oy = x0, y0
    else:
        ox = oy = 0

    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    ax.imshow(rgb, cmap="gray", origin="lower", interpolation="nearest")

    gm = pd.to_numeric(gaia_df["g_mag"], errors="coerce").to_numpy(dtype=np.float64)
    gx = pd.to_numeric(gaia_df["x_gaia"], errors="coerce").to_numpy(dtype=np.float64)
    gy = pd.to_numeric(gaia_df["y_gaia"], errors="coerce").to_numpy(dtype=np.float64)

    blend_col = gaia_df.get("blend_group_id", pd.Series([""] * len(gaia_df)))
    for j in range(len(gaia_df)):
        if not (math.isfinite(gx[j]) and math.isfinite(gy[j])):
            continue
        x = gx[j] - ox
        y = gy[j] - oy
        if crop is not None and (x < 0 or y < 0 or x >= crop[2] or y >= crop[3]):
            continue
        g = gm[j] if math.isfinite(gm[j]) else 99.0
        if g > OVERLAY_G_MAX:
            continue
        gid = str(blend_col.iloc[j] if j < len(blend_col) else "")
        if gid:
            ax.plot(x, y, "o", ms=5, mfc="violet", mec="white", mew=0.3, alpha=0.9)
        else:
            ax.plot(x, y, "o", ms=3, mfc="dodgerblue", mec="none", alpha=0.55)
        if g <= RED_X_G_MAX and int(gaia_owner[j]) < 0:
            ax.plot(x, y, "x", ms=7, mfc="red", mec="red", mew=1.2)

    for x, y in zip(det_x, det_y, strict=False):
        xp, yp = float(x) - ox, float(y) - oy
        if crop is not None and (xp < 0 or yp < 0 or xp >= crop[2] or yp >= crop[3]):
            continue
        circ = plt.Circle((xp, yp), 4.0, fill=False, edgecolor="lime", linewidth=0.8, alpha=0.85)
        ax.add_patch(circ)

    ax.set_title(title, fontsize=10)
    ax.set_axis_off()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def crop_boxes(wpx: int, h: int, size: int = 500) -> dict[str, tuple[int, int, int, int]]:
    size = min(size, wpx, h)
    return {
        "center": ((wpx - size) // 2, (h - size) // 2, size, size),
        "mid": ((wpx - size) // 2, max(0, h // 3 - size // 2), size, size),
        "corner": (0, max(0, h - size), size, size),
    }


def verdict_for(m: FrameMetrics) -> str:
    g1a = m.g1_le13 >= 0.99
    g1b = m.g1_le145 >= 0.95
    g2 = m.g2_empty_false_accept <= 0.01 if math.isfinite(m.g2_empty_false_accept) else True
    g3 = m.g3_spurious_frac <= 0.01
    g4 = m.g4_unnamed == 0
    if g1a and g1b and g2 and g3 and g4:
        return "PASS"
    parts = []
    if not g1a:
        parts.append("G1<=13")
    if not g1b:
        parts.append("G1<=14.5")
    if not g2:
        parts.append("G2")
    if not g3:
        parts.append("G3")
    if not g4:
        parts.append("G4")
    return "FAIL:" + "+".join(parts)


def process_frame(
    frame_label: str,
    fits_path: Path,
    cfg: IterationConfig,
    gaia_db: Path,
    empty_df: pd.DataFrame,
    rng: np.random.Generator,
    out_dir: Path,
    *,
    write_overlay: bool,
) -> FrameMetrics:
    t0 = time.perf_counter()
    _raw, data0, hdr, wcs, fwhm_hdr, wpx, h = load_frame(fits_path)
    fwhm = float(cfg.fwhm_px) if cfg.fwhm_px > 0 else fwhm_hdr
    gaia = _gaia_on_chip(wcs, wpx, h, gaia_db)
    gaia_le16 = gaia[pd.to_numeric(gaia["g_mag"], errors="coerce") <= OVERLAY_G_MAX].copy().reset_index(drop=True)

    sky = estimate_sky(data0, gaia_df=gaia, fwhm_px=fwhm, rng=rng)
    thr_adu = max(float(cfg.threshold_sigma) * sky.sky_sigma_clipped, 1e-6)

    det_x, det_y, _ = run_dao(
        data0,
        fwhm_px=fwhm,
        threshold_adu=thr_adu,
        sharpness_range=cfg.sharpness_range,
    )
    det_to_g, gaia_owner, _, _ = lock_existing_and_leftover_assign(
        det_x, det_y, gaia_le16, locked_pairs=None, leftover_radius_px=MATCH_RADIUS_PX
    )
    census = assign_states(
        gaia_le16, gaia_owner, data0=data0, hdr=hdr, fwhm_px=fwhm, wpx=wpx, h=h, depth_g=TARGET_DEPTH_G
    )

    g4_unnamed = 0
    state_counts: dict[str, int] = {}
    gm = pd.to_numeric(census["g_mag"], errors="coerce").to_numpy(dtype=np.float64)
    for j in range(len(census)):
        st = str(census.iloc[j]["source_state"])
        state_counts[st] = state_counts.get(st, 0) + 1
        g = gm[j]
        if math.isfinite(g) and g <= TARGET_DEPTH_G and st != "DETECTED":
            if st not in (SOURCE_BLENDED, SOURCE_SATURATED, SOURCE_EDGE, SOURCE_TOO_FAINT):
                g4_unnamed += 1

    g2 = g2_empty_false_accept(det_x, det_y, empty_df, frame_label)
    g3, n_sp = g3_spurious(det_x, det_y, gaia_le16, wpx=wpx, h=h)
    killed, n_bright = bright_sharpness_kill_count(
        data0, gaia_le16, fwhm_px=fwhm, sky_sigma=sky.sky_sigma_clipped, threshold_sigma=cfg.threshold_sigma
    )

    if write_overlay:
        od = out_dir / cfg.iter_id / frame_label
        census = annotate_blended_groups(census, gaia_owner, fwhm_px=fwhm)
        render_overlay(
            data0,
            census,
            gaia_owner,
            det_x,
            det_y,
            out_path=od / "overlay_full.png",
            title=f"{frame_label} {cfg.iter_id} thr={cfg.threshold_sigma}",
        )
        for name, box in crop_boxes(wpx, h).items():
            render_overlay(
                data0,
                census,
                gaia_owner,
                det_x,
                det_y,
                out_path=od / f"overlay_crop_{name}.png",
                title=f"{frame_label} {cfg.iter_id} {name}",
                crop=box,
            )

    rt = time.perf_counter() - t0
    return FrameMetrics(
        frame=frame_label,
        n_det=int(det_x.size),
        n_gaia_onchip_le16=int(len(gaia_le16)),
        g1_le13=g1_at_mag(gaia_le16, gaia_owner, 13.0),
        g1_le145=g1_at_mag(gaia_le16, gaia_owner, 14.5),
        g1_curve=completeness_curve(gaia_le16, gaia_owner, OVERLAY_G_MAX),
        g2_empty_false_accept=float(g2),
        g3_spurious_frac=float(g3),
        n_spurious=int(n_sp),
        g4_unnamed=int(g4_unnamed),
        state_counts=state_counts,
        n_bright_killed_by_sharpness=int(killed),
        n_bright_gaia_le13=int(n_bright),
        sky=sky,
        threshold_adu=float(thr_adu),
        runtime_s=float(rt),
    )


def run_sweep(ctx: Path, *, best_only: bool = False, threshold: float | None = None) -> None:
    cfg_app = AppConfig()
    gaia_db = Path(cfg_app.gaia_db_path)
    empty_df = pd.read_csv(EMPTY_SKY_CSV) if EMPTY_SKY_CSV.is_file() else pd.DataFrame()
    ctx.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(51601)

    sweep_rows: list[SweepRow] = []
    all_metrics: dict[str, Any] = {"iterations": []}

    sigmas = [float(threshold)] if threshold is not None else THRESHOLD_SIGMAS
    for sig in sigmas:
        icfg = IterationConfig(
            iter_id=f"thr{sig:.1f}_sharp_open",
            threshold_sigma=float(sig),
            sharpness_range=SHARPNESS_OPEN,
            fwhm_px=5.3,
        )
        iter_metrics: dict[str, Any] = {"config": asdict(icfg), "frames": {}}
        write_overlay = not best_only or sig == min(sigmas, key=lambda s: abs(s - 3.8))
        if threshold is not None:
            write_overlay = True

        for frame_label, fpath in FRAMES:
            if not fpath.is_file():
                continue
            fm = process_frame(
                frame_label,
                fpath,
                icfg,
                gaia_db,
                empty_df,
                rng,
                ctx,
                write_overlay=write_overlay,
            )
            iter_metrics["frames"][frame_label] = {
                k: v for k, v in asdict(fm).items() if k != "sky"
            }
            iter_metrics["frames"][frame_label]["sky"] = asdict(fm.sky)
            v = verdict_for(fm)
            sweep_rows.append(
                SweepRow(
                    iter_id=icfg.iter_id,
                    threshold_sigma=sig,
                    sharpness_lo=SHARPNESS_OPEN[0],
                    sharpness_hi=SHARPNESS_OPEN[1],
                    frame=frame_label,
                    g1_le13=fm.g1_le13,
                    g1_le145=fm.g1_le145,
                    g2=fm.g2_empty_false_accept,
                    g3=fm.g3_spurious_frac,
                    g4_unnamed=fm.g4_unnamed,
                    n_det=fm.n_det,
                    sky_sigma=fm.sky.sky_sigma_clipped,
                    local_annulus_p50=fm.sky.local_annulus_std_p50,
                    threshold_adu=fm.threshold_adu,
                    runtime_s=fm.runtime_s,
                    verdict=v,
                )
            )
        all_metrics["iterations"].append(iter_metrics)

    sweep_df = pd.DataFrame([asdict(r) for r in sweep_rows])
    sweep_df.to_csv(ctx / "iteration_log.csv", index=False)
    with open(ctx / "metrics_all.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    # Production sharpness diagnostic on MASTERSTAR @ 3.8 sigma
    icfg_prod = IterationConfig(
        iter_id="thr3.8_sharp_prod",
        threshold_sigma=3.8,
        sharpness_range=SHARPNESS_PRODUCTION,
        fwhm_px=5.3,
    )
    fm_prod = process_frame(
        "MASTERSTAR",
        PS_DIR / "MASTERSTAR.fits",
        icfg_prod,
        gaia_db,
        empty_df,
        rng,
        ctx,
        write_overlay=False,
    )
    with open(ctx / "sharpness_kill_report.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "threshold_sigma": 3.8,
                "sharpness_production": SHARPNESS_PRODUCTION,
                "sharpness_open": SHARPNESS_OPEN,
                "masterstar_bright_le13_total": fm_prod.n_bright_gaia_le13,
                "killed_by_production_sharpness": fm_prod.n_bright_killed_by_sharpness,
                "production_g1_le13": fm_prod.g1_le13,
                "production_g1_le145": fm_prod.g1_le145,
                "open_reference_g1_le13": next(
                    (
                        r.g1_le13
                        for r in sweep_rows
                        if r.frame == "MASTERSTAR" and abs(r.threshold_sigma - 3.8) < 0.01
                    ),
                    None,
                ),
            },
            f,
            indent=2,
        )

    # Best config summary (MASTERSTAR median rank)
    ms = sweep_df[sweep_df["frame"] == "MASTERSTAR"].copy()
    if not ms.empty:
        ms["score"] = ms["g1_le145"] - 10 * ms["g3"] - 5 * ms["g2"].fillna(0)
        best = ms.sort_values("score", ascending=False).iloc[0]
        with open(ctx / "best_config.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "best_iter_id": str(best["iter_id"]),
                    "threshold_sigma": float(best["threshold_sigma"]),
                    "masterstar_g1_le13": float(best["g1_le13"]),
                    "masterstar_g1_le145": float(best["g1_le145"]),
                    "masterstar_g2": float(best["g2"]),
                    "masterstar_g3": float(best["g3"]),
                    "verdict": str(best["verdict"]),
                    "pass2_needed": bool(
                        float(best["g1_le145"]) < 0.95
                        or float(best["g2"]) > 0.01
                        or float(best["g3"]) > 0.01
                    ),
                },
                f,
                indent=2,
            )


def main() -> None:
    ap = argparse.ArgumentParser(description="DAO-GAIA-STAGE-01 sandbox harness")
    ap.add_argument("--ctx", type=Path, default=DEFAULT_CTX)
    ap.add_argument("--threshold", type=float, default=None, help="Single threshold sigma (skip sweep)")
    args = ap.parse_args()
    t0 = time.perf_counter()
    run_sweep(args.ctx, threshold=args.threshold)
    print(f"Done in {time.perf_counter() - t0:.1f}s -> {args.ctx}")


if __name__ == "__main__":
    main()
