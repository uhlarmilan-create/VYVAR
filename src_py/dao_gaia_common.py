"""Family-local helpers for the dao_gaia CLI family (iter4 + validation).

Not imported by the production pipeline (night_run / pipeline / app).
Do not merge helpers from this family into global homes (CONSOLIDATE-01C R4).

Flattened 2026-09-01 CONSOLIDATE-01D P2-6 from stage_01 / iter2 / iter3.
Follow-up: rename dao_gaia_stage_01_iter4.py to dao_gaia_stage.py only if Milan asks.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from photutils.detection import DAOStarFinder
from scipy.spatial import cKDTree

try:
    from scipy.ndimage import disk  # type: ignore[attr-defined]
except ImportError:
    from skimage.morphology import disk

from masterstar_gaia_accounting import (
    SOURCE_BLENDED,
    SOURCE_EDGE,
    SOURCE_SATURATED,
    SOURCE_TOO_FAINT,
    _dao_pass2_annulus_stats,
)
from pipeline import _query_gaia_local
from plain_stats import plain_mean_med_std, sky_mad_sigma_adu
from vyvar_platesolver import _apply_pm_to_gaia_rows, _obs_year_from_header

REPO = Path(__file__).resolve().parents[1]
DRAFT = REPO / "Archive" / "Drafts" / "draft_000516"
PS_DIR = DRAFT / "platesolve" / "NoFilter_60_2"
LIGHTS_DIR = DRAFT / "detrended_aligned" / "lights" / "NoFilter_60_2"

# from stage_01, 2026-08
FRAMES: list[tuple[str, Path]] = [
    ("MASTERSTAR", PS_DIR / "MASTERSTAR.fits"),
    ("Light_001", LIGHTS_DIR / "BO_CVn_Light_001.fits"),
    ("Light_076", LIGHTS_DIR / "BO_CVn_Light_076.fits"),
    ("Light_148", LIGHTS_DIR / "BO_CVn_Light_148.fits"),
]

SHARPNESS_OPEN = (0.0, 10.0)
MATCH_RADIUS_PX = 3.0
CORNER_MARGIN_PX = 120.0

# from iter2, 2026-08 -- production masterstar_gaia_census_edge_margin_px consumer
EDGE_MARGIN_PX = 10.0
OVERLAY_G_MAX = 16.0
G3_GAIA_MAX = 18.0
GAIA_QUERY_G = 18.0

# from iter3, 2026-08
SOURCE_CROWDED_MISS = "CROWDED_MISS"


def _is_corner(x: float, y: float, wpx: int, h: int) -> bool:
    m = float(CORNER_MARGIN_PX)
    return x < m or y < m or x >= float(wpx) - m or y >= float(h) - m


def _peak_at(data0: np.ndarray, x: float, y: float, r: int = 3) -> float:
    h, w = data0.shape
    ix, iy = int(round(x)), int(round(y))
    x0, x1 = max(0, ix - r), min(w, ix + r + 1)
    y0, y1 = max(0, iy - r), min(h, iy + r + 1)
    patch = data0[y0:y1, x0:x1]
    return float(np.max(patch)) if patch.size else float("nan")


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


def asinh_rgb(data0: np.ndarray) -> np.ndarray:
    pos = np.clip(data0, 0, None)
    scale = float(np.percentile(pos[pos > 0], 99.5)) if np.any(pos > 0) else 1.0
    scale = max(scale, 1.0)
    return np.arcsinh(pos / scale) / np.arcsinh(1.0)

# --- from stage_01, 2026-08 ---
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

def _wcs_from_hdr(hdr: fits.Header) -> WCS:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return WCS(hdr)


def load_frame(path: Path) -> tuple[np.ndarray, np.ndarray, fits.Header, WCS, float, int, int]:
    """from stage_01, 2026-08."""
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
    """from stage_01, 2026-08."""
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
    """from stage_01, 2026-08."""
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

def g2_empty_false_accept(
    det_x: np.ndarray,
    det_y: np.ndarray,
    empty_df: pd.DataFrame,
    frame_label: str,
    *,
    match_radius_px: float | None = None,
) -> float:
    """from stage_01, 2026-08."""
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

def crop_boxes(wpx: int, h: int, size: int = 500) -> dict[str, tuple[int, int, int, int]]:
    """from stage_01, 2026-08."""
    size = min(size, wpx, h)
    return {
        "center": ((wpx - size) // 2, (h - size) // 2, size, size),
        "mid": ((wpx - size) // 2, max(0, h // 3 - size // 2), size, size),
        "corner": (0, max(0, h - size), size, size),
    }


# --- from iter2, 2026-08 ---
def _is_edge(x: float, y: float, wpx: int, h: int, margin: float = EDGE_MARGIN_PX) -> bool:
    """from iter2, 2026-08. Consumes EDGE_MARGIN_PX (masterstar_gaia_census_edge_margin_px)."""
    m = float(margin)
    return x < m or y < m or x >= float(wpx) - m or y >= float(h) - m


def _local_snr(data0: np.ndarray, x: float, y: float, fwhm_px: float) -> float:
    peak = _peak_at(data0, x, y, r=3)
    _, local_std = _dao_pass2_annulus_stats(data0, x, y)
    if not (math.isfinite(peak) and math.isfinite(local_std) and local_std > 0):
        return float("nan")
    return float(peak / local_std)


def _nn_gaia_px(j: int, gaia_df: pd.DataFrame) -> float:
    gx = pd.to_numeric(gaia_df["x_gaia"], errors="coerce").to_numpy(dtype=np.float64)
    gy = pd.to_numeric(gaia_df["y_gaia"], errors="coerce").to_numpy(dtype=np.float64)
    x0, y0 = gx[j], gy[j]
    if not (math.isfinite(x0) and math.isfinite(y0)):
        return float("nan")
    dmin = float("inf")
    for k in range(len(gaia_df)):
        if k == j:
            continue
        if math.isfinite(gx[k]) and math.isfinite(gy[k]):
            d = float(math.hypot(gx[k] - x0, gy[k] - y0))
            if d < dmin:
                dmin = d
    return dmin if math.isfinite(dmin) else float("nan")

def g3_spurious(
    det_x: np.ndarray,
    det_y: np.ndarray,
    gaia_g18: pd.DataFrame,
    *,
    wpx: int,
    h: int,
    match_radius_px: float | None = None,
) -> tuple[float, int]:
    """from iter2, 2026-08."""
    match_r = float(MATCH_RADIUS_PX if match_radius_px is None else match_radius_px)
    if det_x.size == 0:
        return 0.0, 0
    gx = pd.to_numeric(gaia_g18["x_gaia"], errors="coerce").to_numpy(dtype=np.float64)
    gy = pd.to_numeric(gaia_g18["y_gaia"], errors="coerce").to_numpy(dtype=np.float64)
    ok = np.isfinite(gx) & np.isfinite(gy)
    tree = cKDTree(np.column_stack([gx[ok], gy[ok]]))
    spurious = 0
    considered = 0
    for x, y in zip(det_x, det_y, strict=False):
        if _is_corner(float(x), float(y), wpx, h):
            continue
        considered += 1
        d, _ = tree.query([float(x), float(y)], distance_upper_bound=match_r)
        if not math.isfinite(float(d)) or float(d) > match_r:
            spurious += 1
    return (spurious / considered if considered else 0.0), spurious

def decompose_holes_le13(
    census: pd.DataFrame,
    gaia_df: pd.DataFrame,
    data0: np.ndarray,
    fwhm_px: float,
) -> pd.DataFrame:
    """from iter2, 2026-08."""
    gm = pd.to_numeric(census["g_mag"], errors="coerce").to_numpy(dtype=np.float64)
    st = census["source_state"].astype(str).to_numpy()
    rows: list[dict[str, Any]] = []
    summary: dict[str, int] = {}
    for j in range(len(census)):
        if not (math.isfinite(gm[j]) and gm[j] <= 13.0):
            continue
        state = str(st[j])
        if state == "DETECTED":
            continue
        if state == SOURCE_TOO_FAINT:
            hole_class = "true_miss"
        elif state in (SOURCE_BLENDED, SOURCE_EDGE, SOURCE_SATURATED):
            hole_class = state
        else:
            hole_class = "true_miss"
        summary[hole_class] = summary.get(hole_class, 0) + 1
        if hole_class == "true_miss":
            xg = float(census.iloc[j]["x_gaia"])
            yg = float(census.iloc[j]["y_gaia"])
            rows.append(
                {
                    "catalog_id": str(census.iloc[j]["catalog_id"]),
                    "x": xg,
                    "y": yg,
                    "g_mag": float(gm[j]),
                    "state": state,
                    "nn_px": _nn_gaia_px(j, gaia_df),
                    "local_snr": _local_snr(data0, xg, yg, fwhm_px),
                }
            )
    out = pd.DataFrame(rows)
    return out


# --- from iter3, 2026-08 ---
def _gaia_on_chip_pm(
    wcs: WCS, wpx: int, h: int, gaia_db: Path, hdr, *, max_mag: float
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """from iter3, 2026-08. Return (pm-corrected on-chip, no-pm on-chip, n_pm_corrected)."""
    corners = wcs.pixel_to_world([0, wpx, 0, wpx], [0, 0, h, h])
    ra_corners = [float(c.icrs.ra.deg) for c in corners]
    de_corners = [float(c.icrs.dec.deg) for c in corners]
    center = wcs.pixel_to_world(wpx / 2.0, h / 2.0)
    radius_deg = max(max(ra_corners) - min(ra_corners), max(de_corners) - min(de_corners)) / 2.0 * 1.25
    radius_deg = max(float(radius_deg), 0.6)
    df0 = _query_gaia_local(
        center=center,
        radius_deg=radius_deg,
        gaia_db_path=gaia_db,
        max_mag=float(max_mag),
        max_rows=800_000,
    )
    if df0.empty:
        return df0, df0, 0

    obs_year = _obs_year_from_header(hdr)
    rows = []
    for _, r in df0.iterrows():
        rows.append(
            {
                "source_id": r.get("catalog_id", r.get("source_id")),
                "ra": float(r["ra_deg"]),
                "dec": float(r["dec_deg"]),
                "g_mag": r.get("mag"),
                "pmra": r.get("pmra") if "pmra" in df0.columns else None,
                "pmdec": r.get("pmdec") if "pmdec" in df0.columns else None,
            }
        )
    rows_pm, n_pm = _apply_pm_to_gaia_rows(rows, obs_year=float(obs_year))

    def _build(rows_in: list[dict[str, Any]], *, suffix: str) -> pd.DataFrame:
        ra = np.array([float(r["ra"]) for r in rows_in], dtype=np.float64)
        de = np.array([float(r["dec"]) for r in rows_in], dtype=np.float64)
        x, y = wcs.world_to_pixel_values(ra, de)
        out = df0.copy()
        out["ra_deg"] = ra
        out["dec_deg"] = de
        out["x_gaia"] = x
        out["y_gaia"] = y
        out["g_mag"] = pd.to_numeric(out.get("mag"), errors="coerce")
        inb = (out["x_gaia"] >= 0) & (out["x_gaia"] < float(wpx)) & (out["y_gaia"] >= 0) & (out["y_gaia"] < float(h))
        return out.loc[inb].reset_index(drop=True)

    # no-pm uses original ra_deg/dec_deg from query
    ra0 = pd.to_numeric(df0["ra_deg"], errors="coerce").to_numpy(dtype=np.float64)
    de0 = pd.to_numeric(df0["dec_deg"], errors="coerce").to_numpy(dtype=np.float64)
    x0, y0 = wcs.world_to_pixel_values(ra0, de0)
    nopm = df0.copy()
    nopm["x_gaia"] = x0
    nopm["y_gaia"] = y0
    nopm["g_mag"] = pd.to_numeric(nopm.get("mag"), errors="coerce")
    inb0 = (nopm["x_gaia"] >= 0) & (nopm["x_gaia"] < float(wpx)) & (nopm["y_gaia"] >= 0) & (nopm["y_gaia"] < float(h))
    nopm = nopm.loc[inb0].reset_index(drop=True)

    pm = _build(rows_pm, suffix="pm")
    return pm, nopm, int(n_pm)

