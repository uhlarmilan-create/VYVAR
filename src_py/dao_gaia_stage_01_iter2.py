#!/usr/bin/env python3
"""DAO-GAIA-STAGE-01 iteration 2: metric fixes + combined pass2."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from photutils.detection import DAOStarFinder
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src_py"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import AppConfig  # noqa: E402
from dao_gaia_common import _is_corner, _peak_at, _saturation_limit, asinh_rgb  # noqa: E402
from masterstar_gaia_accounting import (  # noqa: E402
    SOURCE_BLENDED,
    SOURCE_EDGE,
    SOURCE_SATURATED,
    SOURCE_TOO_FAINT,
    Pass2AcceptParams,
    _dao_pass2_annulus_stats,
    annotate_blended_groups,
    dao_pass2_try_at_position,
    lock_existing_and_leftover_assign,
)
from plain_stats import plain_mean_med_std  # noqa: E402
from pipeline import _query_gaia_local  # noqa: E402

# Import shared helpers from iter1 harness
sys.path.insert(0, str(REPO / "tmp"))
from dao_gaia_stage_01 import (  # noqa: E402
    FRAMES,
    LIGHTS_DIR,
    PS_DIR,
    SHARPNESS_OPEN,
    crop_boxes,
    estimate_sky,
    g2_empty_false_accept,
    load_frame,
    run_dao,
)

DRAFT = REPO / "Archive" / "Drafts" / "draft_000516"
EMPTY_SKY_CSV = REPO / "dev" / "results" / "context" / "session_20260819_msgaia01" / "empty_positions_main.csv"
ITER1_CTX = REPO / "dev" / "results" / "context" / "session_20260819_daostage01"
DEFAULT_CTX = REPO / "dev" / "results" / "context" / "session_20260819_daostage01_iter2"

SWEEP_SIGMAS = [3.0, 3.5, 3.8, 4.5, 5.0]
COMBINED_SIGMAS = [3.5, 4.5]
MATCH_RADIUS_PX = 3.0
EDGE_MARGIN_PX = 10.0  # production masterstar_gaia_census_edge_margin_px
CORNER_MARGIN_PX = 120.0
TARGET_DEPTH_G = 15.0
OVERLAY_G_MAX = 16.0
G3_GAIA_MAX = 18.0
GAIA_QUERY_G = 18.0
PASS2_G_LO = 13.0
PASS2_G_HI = 15.0
PASS2_PARAMS = Pass2AcceptParams(sigma=5.0, center_tol_px=2.0, fwhm_px=5.3)

EYE_OK_STATES = frozenset({"DETECTED", SOURCE_BLENDED, SOURCE_SATURATED})


@dataclass
class ScoreBundle:
    g1_strict_le13: float
    g1_strict_le145: float
    g1_eye_le13: float
    g1_eye_le145: float
    g2: float
    g3_g18: float
    g3_g16: float
    n_spurious_g18: int
    g4_unnamed: int
    state_counts: dict[str, int]
    n_det: int
    n_edge: int
    edge_margin_px: float
    n_gaia_onchip_le16: int
    n_gaia_eligible_le16: int


def _gaia_on_chip(wcs: WCS, wpx: int, h: int, gaia_db: Path, *, max_mag: float) -> pd.DataFrame:
    corners = wcs.pixel_to_world([0, wpx, 0, wpx], [0, 0, h, h])
    ra_corners = [float(c.icrs.ra.deg) for c in corners]
    de_corners = [float(c.icrs.dec.deg) for c in corners]
    center = wcs.pixel_to_world(wpx / 2.0, h / 2.0)
    radius_deg = max(max(ra_corners) - min(ra_corners), max(de_corners) - min(de_corners)) / 2.0 * 1.25
    radius_deg = max(float(radius_deg), 0.6)
    df = _query_gaia_local(
        center=center,
        radius_deg=radius_deg,
        gaia_db_path=gaia_db,
        max_mag=float(max_mag),
        max_rows=800_000,
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


def _is_edge(x: float, y: float, wpx: int, h: int, margin: float = EDGE_MARGIN_PX) -> bool:
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


def edge_audit(wpx: int, h: int, n_onchip: int, *, margin: float) -> dict[str, Any]:
    """Geometric expectation for uniform Gaia density in rectangular footprint."""
    inner_w = max(float(wpx) - 2.0 * margin, 0.0)
    inner_h = max(float(h) - 2.0 * margin, 0.0)
    frac_inner = (inner_w / float(wpx)) * (inner_h / float(h)) if wpx > 0 and h > 0 else 0.0
    frac_edge = 1.0 - frac_inner
    return {
        "edge_margin_px": float(margin),
        "frame_w_px": int(wpx),
        "frame_h_px": int(h),
        "n_onchip_le16": int(n_onchip),
        "geometric_edge_fraction": float(frac_edge),
        "geometric_edge_count_expected": float(frac_edge * n_onchip),
        "note": "Uniform-density approximation; real edge excess comes from WCS distortion + border crowding.",
    }


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
    edge_margin_px: float = EDGE_MARGIN_PX,
) -> pd.DataFrame:
    gdf = annotate_blended_groups(gaia_df.copy(), gaia_owner, fwhm_px=fwhm_px)
    sat_lim = _saturation_limit(hdr) * 0.999
    states: list[str] = []
    for j in range(len(gdf)):
        xg = float(gdf.iloc[j]["x_gaia"])
        yg = float(gdf.iloc[j]["y_gaia"])
        gmag = float(gdf.iloc[j]["g_mag"]) if pd.notna(gdf.iloc[j]["g_mag"]) else float("nan")
        owner = int(gaia_owner[j]) if j < len(gaia_owner) else -1
        blend_gid = str(gdf.iloc[j].get("blend_group_id", "") or "")
        if _is_edge(xg, yg, wpx, h, edge_margin_px):
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


def g3_spurious(
    det_x: np.ndarray,
    det_y: np.ndarray,
    gaia_g18: pd.DataFrame,
    *,
    wpx: int,
    h: int,
    match_radius_px: float | None = None,
) -> tuple[float, int]:
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


def _eligible_mask(gaia_df: pd.DataFrame, wpx: int, h: int, edge_margin: float) -> np.ndarray:
    gx = pd.to_numeric(gaia_df["x_gaia"], errors="coerce").to_numpy(dtype=np.float64)
    gy = pd.to_numeric(gaia_df["y_gaia"], errors="coerce").to_numpy(dtype=np.float64)
    return ~np.array([_is_edge(float(x), float(y), wpx, h, edge_margin) for x, y in zip(gx, gy, strict=False)])


def g1_strict(gaia_df: pd.DataFrame, census: pd.DataFrame, mag_cut: float, eligible: np.ndarray) -> float:
    gm = pd.to_numeric(gaia_df["g_mag"], errors="coerce").to_numpy(dtype=np.float64)
    sel = eligible & np.isfinite(gm) & (gm <= float(mag_cut))
    n = int(sel.sum())
    if n == 0:
        return float("nan")
    det = (census["source_state"].astype(str) == "DETECTED").to_numpy()
    return float(det[sel].sum() / n)


def g1_eye(census: pd.DataFrame, mag_cut: float, eligible: np.ndarray) -> float:
    gm = pd.to_numeric(census["g_mag"], errors="coerce").to_numpy(dtype=np.float64)
    st = census["source_state"].astype(str).to_numpy()
    sel = eligible & np.isfinite(gm) & (gm <= float(mag_cut))
    n = int(sel.sum())
    if n == 0:
        return float("nan")
    ok = np.isin(st, list(EYE_OK_STATES))
    return float(ok[sel].sum() / n)


def run_pass2_seeds(
    data0: np.ndarray,
    gaia_df: pd.DataFrame,
    gaia_owner: np.ndarray,
    *,
    wpx: int,
    h: int,
    fwhm_px: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    extra_x: list[float] = []
    extra_y: list[float] = []
    n_try = 0
    gm = pd.to_numeric(gaia_df["g_mag"], errors="coerce").to_numpy(dtype=np.float64)
    params = Pass2AcceptParams(
        sigma=float(PASS2_PARAMS.sigma),
        center_tol_px=float(PASS2_PARAMS.center_tol_px),
        fwhm_px=float(fwhm_px),
    )
    for j in range(len(gaia_df)):
        if int(gaia_owner[j]) >= 0:
            continue
        g = float(gm[j]) if math.isfinite(gm[j]) else float("nan")
        if not (math.isfinite(g) and float(PASS2_G_LO) < g <= float(PASS2_G_HI)):
            continue
        xg = float(gaia_df.iloc[j]["x_gaia"])
        yg = float(gaia_df.iloc[j]["y_gaia"])
        if _is_edge(xg, yg, wpx, h, EDGE_MARGIN_PX):
            continue
        n_try += 1
        hit = dao_pass2_try_at_position(data0, xg, yg, wpx=wpx, h=h, params=params)
        if hit.get("accepted"):
            extra_x.append(float(hit["x_det"]))
            extra_y.append(float(hit["y_det"]))
    if not extra_x:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64), n_try
    return np.asarray(extra_x, dtype=np.float64), np.asarray(extra_y, dtype=np.float64), n_try


def score_detections(
    det_x: np.ndarray,
    det_y: np.ndarray,
    *,
    gaia_le16: pd.DataFrame,
    gaia_g18: pd.DataFrame,
    gaia_match_g18: pd.DataFrame,
    data0: np.ndarray,
    hdr: fits.Header,
    fwhm_px: float,
    wpx: int,
    h: int,
    empty_df: pd.DataFrame,
    frame_label: str,
) -> tuple[ScoreBundle, pd.DataFrame]:
    det_to_g, gaia_owner_full, _, _ = lock_existing_and_leftover_assign(
        det_x, det_y, gaia_match_g18, locked_pairs=None, leftover_radius_px=MATCH_RADIUS_PX
    )
    cid_full = gaia_match_g18["catalog_id"].astype(str).str.strip()
    cid_le16 = gaia_le16["catalog_id"].astype(str).str.strip()
    cid_to_j = {str(cid_le16.iloc[j]).strip(): j for j in range(len(gaia_le16))}
    gaia_owner = np.full(len(gaia_le16), -1, dtype=np.int64)
    for i in range(len(det_x)):
        gi = int(det_to_g[i])
        if gi < 0:
            continue
        cid = str(cid_full.iloc[gi]).strip()
        lj = cid_to_j.get(cid)
        if lj is not None and gaia_owner[lj] < 0:
            gaia_owner[lj] = i

    census = assign_states(
        gaia_le16, gaia_owner, data0=data0, hdr=hdr, fwhm_px=fwhm_px, wpx=wpx, h=h, depth_g=TARGET_DEPTH_G
    )
    eligible = _eligible_mask(gaia_le16, wpx, h, EDGE_MARGIN_PX)
    g2 = g2_empty_false_accept(det_x, det_y, empty_df, frame_label)
    g3_18, n_sp_18 = g3_spurious(det_x, det_y, gaia_g18, wpx=wpx, h=h)
    gaia_g16 = gaia_g18[pd.to_numeric(gaia_g18["g_mag"], errors="coerce") <= OVERLAY_G_MAX]
    g3_16, _ = g3_spurious(det_x, det_y, gaia_g16, wpx=wpx, h=h)

    state_counts: dict[str, int] = {}
    g4_unnamed = 0
    gm = pd.to_numeric(census["g_mag"], errors="coerce").to_numpy(dtype=np.float64)
    for j in range(len(census)):
        st = str(census.iloc[j]["source_state"])
        state_counts[st] = state_counts.get(st, 0) + 1
        g = gm[j]
        if math.isfinite(g) and g <= TARGET_DEPTH_G and st not in (
            "DETECTED",
            SOURCE_BLENDED,
            SOURCE_SATURATED,
            SOURCE_EDGE,
            SOURCE_TOO_FAINT,
        ):
            g4_unnamed += 1

    bundle = ScoreBundle(
        g1_strict_le13=g1_strict(gaia_le16, census, 13.0, eligible),
        g1_strict_le145=g1_strict(gaia_le16, census, 14.5, eligible),
        g1_eye_le13=g1_eye(census, 13.0, eligible),
        g1_eye_le145=g1_eye(census, 14.5, eligible),
        g2=float(g2),
        g3_g18=float(g3_18),
        g3_g16=float(g3_16),
        n_spurious_g18=int(n_sp_18),
        g4_unnamed=int(g4_unnamed),
        state_counts=state_counts,
        n_det=int(det_x.size),
        n_edge=int(state_counts.get(SOURCE_EDGE, 0)),
        edge_margin_px=float(EDGE_MARGIN_PX),
        n_gaia_onchip_le16=int(len(gaia_le16)),
        n_gaia_eligible_le16=int(eligible.sum()),
    )
    return bundle, census


def decompose_holes_le13(
    census: pd.DataFrame,
    gaia_df: pd.DataFrame,
    data0: np.ndarray,
    fwhm_px: float,
) -> pd.DataFrame:
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


def save_detections(path: Path, det_x: np.ndarray, det_y: np.ndarray, meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, det_x=det_x, det_y=det_y, meta=json.dumps(meta))


def load_detections(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    z = np.load(path)
    meta = json.loads(str(z["meta"]))
    return np.asarray(z["det_x"], dtype=np.float64), np.asarray(z["det_y"], dtype=np.float64), meta


def render_overlay_v2(
    data0: np.ndarray,
    census: pd.DataFrame,
    det_x: np.ndarray,
    det_y: np.ndarray,
    *,
    out_path: Path,
    title: str,
    crop: tuple[int, int, int, int] | None = None,
) -> None:
    rgb = asinh_rgb(data0)
    ox = oy = 0
    if crop is not None:
        x0, y0, cw, ch = crop
        rgb = rgb[y0 : y0 + ch, x0 : x0 + cw]
        ox, oy = x0, y0

    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    ax.imshow(rgb, cmap="gray", origin="lower", interpolation="nearest")

    gm = pd.to_numeric(census["g_mag"], errors="coerce").to_numpy(dtype=np.float64)
    gx = pd.to_numeric(census["x_gaia"], errors="coerce").to_numpy(dtype=np.float64)
    gy = pd.to_numeric(census["y_gaia"], errors="coerce").to_numpy(dtype=np.float64)
    st = census["source_state"].astype(str).to_numpy()
    blend_col = census.get("blend_group_id", pd.Series([""] * len(census)))

    for j in range(len(census)):
        if not (math.isfinite(gx[j]) and math.isfinite(gy[j])):
            continue
        g = gm[j] if math.isfinite(gm[j]) else 99.0
        if g > OVERLAY_G_MAX:
            continue
        x, y = gx[j] - ox, gy[j] - oy
        if crop is not None and (x < 0 or y < 0 or x >= crop[2] or y >= crop[3]):
            continue
        state = str(st[j])
        gid = str(blend_col.iloc[j] if j < len(blend_col) else "")
        if state == SOURCE_TOO_FAINT:
            ax.plot(x, y, "o", ms=2.5, mfc="0.65", mec="none", alpha=0.45)
        elif state == SOURCE_SATURATED:
            ax.plot(x, y, "o", ms=6, mfc="orange", mec="white", mew=0.3, alpha=0.95)
        elif gid or state == SOURCE_BLENDED:
            ax.plot(x, y, "o", ms=5, mfc="violet", mec="white", mew=0.3, alpha=0.9)
        elif state == "DETECTED":
            pass  # green circle drawn below
        elif state == SOURCE_EDGE:
            pass
        elif g <= 13.0 and state == SOURCE_TOO_FAINT:
            ax.plot(x, y, "x", ms=8, mfc="red", mec="red", mew=1.4)
        elif state == SOURCE_TOO_FAINT and g <= 14.0:
            # true unexplained miss at bright end
            ax.plot(x, y, "x", ms=8, mfc="red", mec="red", mew=1.4)

    # red X only for true misses: G<=14, eligible, TOO_FAINT (not edge/blend/sat)
    for j in range(len(census)):
        if not (math.isfinite(gx[j]) and math.isfinite(gy[j])):
            continue
        g = gm[j] if math.isfinite(gm[j]) else 99.0
        state = str(st[j])
        if state != SOURCE_TOO_FAINT or g > 14.0:
            continue
        x, y = gx[j] - ox, gy[j] - oy
        if crop is not None and (x < 0 or y < 0 or x >= crop[2] or y >= crop[3]):
            continue
        ax.plot(x, y, "x", ms=8, mfc="red", mec="red", mew=1.4)

    for x, y in zip(det_x, det_y, strict=False):
        xp, yp = float(x) - ox, float(y) - oy
        if crop is not None and (xp < 0 or yp < 0 or xp >= crop[2] or yp >= crop[3]):
            continue
        ax.add_patch(plt.Circle((xp, yp), 4.0, fill=False, edgecolor="lime", linewidth=0.8, alpha=0.85))

    ax.set_title(title, fontsize=10)
    ax.set_axis_off()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def detect_and_save(
    ctx: Path,
    gaia_db: Path,
    sigmas: list[float],
    *,
    combined: bool = False,
) -> None:
    det_root = ctx / "detections"
    rng = np.random.default_rng(51602)
    for frame_label, fpath in FRAMES:
        _raw, data0, hdr, wcs, fwhm, wpx, h = load_frame(fpath)
        gaia_all = _gaia_on_chip(wcs, wpx, h, gaia_db, max_mag=GAIA_QUERY_G)
        for sig in sigmas:
            tag = f"thr{sig:.1f}_sharp_open"
            if combined:
                tag = f"comb_thr{sig:.1f}_p2"
            out_npz = det_root / tag / f"{frame_label}.npz"
            if out_npz.is_file() and not combined:
                continue
            sky = estimate_sky(data0, gaia_df=gaia_all, fwhm_px=fwhm, rng=rng)
            thr = max(float(sig) * sky.sky_sigma_clipped, 1e-6)
            det_x, det_y, _ = run_dao(
                data0, fwhm_px=fwhm, threshold_adu=thr, sharpness_range=SHARPNESS_OPEN
            )
            n_p2 = 0
            if combined:
                gaia_le16 = gaia_all[pd.to_numeric(gaia_all["g_mag"], errors="coerce") <= OVERLAY_G_MAX].copy()
                _, owner_pre, _, _ = lock_existing_and_leftover_assign(
                    det_x, det_y, gaia_all, locked_pairs=None, leftover_radius_px=MATCH_RADIUS_PX
                )
                # map to le16 indices
                cid_all = gaia_all["catalog_id"].astype(str).str.strip()
                cid_le16 = gaia_le16["catalog_id"].astype(str).str.strip()
                cid_to_j = {str(cid_le16.iloc[j]).strip(): j for j in range(len(gaia_le16))}
                owner_le16 = np.full(len(gaia_le16), -1, dtype=np.int64)
                for i in range(len(det_x)):
                    gi = int(owner_pre[i]) if i < len(owner_pre) else -1
                    if gi < 0:
                        continue
                    lj = cid_to_j.get(str(cid_all.iloc[gi]).strip())
                    if lj is not None:
                        owner_le16[lj] = i
                ex, ey, n_p2 = run_pass2_seeds(
                    data0, gaia_le16, owner_le16, wpx=wpx, h=h, fwhm_px=fwhm
                )
                if ex.size:
                    det_x = np.concatenate([det_x, ex])
                    det_y = np.concatenate([det_y, ey])
            save_detections(
                out_npz,
                det_x,
                det_y,
                {
                    "frame": frame_label,
                    "threshold_sigma": sig,
                    "combined": combined,
                    "n_pass2_added": n_p2,
                    "threshold_adu": thr,
                    "sky_sigma": sky.sky_sigma_clipped,
                },
            )


def rescore_all(ctx: Path, gaia_db: Path, empty_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    det_root = ctx / "detections"
    for npz in sorted(det_root.glob("**/*.npz")):
        rel = npz.relative_to(det_root)
        iter_id = rel.parts[0]
        frame_label = npz.stem
        fpath = dict(FRAMES)[frame_label]
        det_x, det_y, meta = load_detections(npz)
        _raw, data0, hdr, wcs, fwhm, wpx, h = load_frame(fpath)
        gaia_all = _gaia_on_chip(wcs, wpx, h, gaia_db, max_mag=GAIA_QUERY_G)
        gaia_g18 = gaia_all[pd.to_numeric(gaia_all["g_mag"], errors="coerce") <= G3_GAIA_MAX].copy()
        gaia_le16 = gaia_all[pd.to_numeric(gaia_all["g_mag"], errors="coerce") <= OVERLAY_G_MAX].copy()
        bundle, census = score_detections(
            det_x,
            det_y,
            gaia_le16=gaia_le16,
            gaia_g18=gaia_g18,
            gaia_match_g18=gaia_g18,
            data0=data0,
            hdr=hdr,
            fwhm_px=fwhm,
            wpx=wpx,
            h=h,
            empty_df=empty_df,
            frame_label=frame_label,
        )
        rows.append(
            {
                "iter_id": iter_id,
                "frame": frame_label,
                "threshold_sigma": meta.get("threshold_sigma"),
                "combined": bool(meta.get("combined", False)),
                **{k: getattr(bundle, k) for k in bundle.__dataclass_fields__},
            }
        )
        if frame_label == "MASTERSTAR" and iter_id == "thr4.5_sharp_open":
            holes = decompose_holes_le13(census, gaia_le16, data0, fwhm)
            holes.to_csv(ctx / "holes_le13_true_miss_thr45.csv", index=False)
            _vc = census.loc[
                (census.g_mag <= 13) & (census.source_state != "DETECTED"), "source_state"
            ].value_counts()
            pd.DataFrame([{"state": str(k), "n": int(n)} for k, n in _vc.items()]).to_csv(
                ctx / "holes_le13_summary_thr45.csv", index=False
            )
            # proper summary
            gm = pd.to_numeric(census["g_mag"], errors="coerce")
            sub = census[(gm <= 13) & (census.source_state != "DETECTED")]
            summ = sub["source_state"].value_counts().reset_index()
            summ.columns = ["state", "n"]
            summ["true_miss"] = summ["state"].eq(SOURCE_TOO_FAINT)
            summ.to_csv(ctx / "holes_le13_decompose_thr45.csv", index=False)
            audit = edge_audit(wpx, h, len(gaia_le16), margin=EDGE_MARGIN_PX)
            audit["observed_edge_count_thr45"] = bundle.n_edge
            audit["observed_edge_count_iter1_margin50"] = 714
            with open(ctx / "edge_audit.json", "w", encoding="utf-8") as f:
                json.dump(audit, f, indent=2)
    df = pd.DataFrame(rows)
    df.to_csv(ctx / "rescore_iteration_log.csv", index=False)
    return df


def regenerate_overlays(ctx: Path, gaia_db: Path, iter_id: str) -> None:
    npz_root = ctx / "detections" / iter_id
    for frame_label, fpath in FRAMES:
        npz = npz_root / f"{frame_label}.npz"
        if not npz.is_file():
            continue
        det_x, det_y, meta = load_detections(npz)
        _raw, data0, hdr, wcs, fwhm, wpx, h = load_frame(fpath)
        gaia_all = _gaia_on_chip(wcs, wpx, h, gaia_db, max_mag=GAIA_QUERY_G)
        gaia_g18 = gaia_all[pd.to_numeric(gaia_all["g_mag"], errors="coerce") <= G3_GAIA_MAX]
        gaia_le16 = gaia_all[pd.to_numeric(gaia_all["g_mag"], errors="coerce") <= OVERLAY_G_MAX]
        _, census = score_detections(
            det_x,
            det_y,
            gaia_le16=gaia_le16,
            gaia_g18=gaia_g18,
            gaia_match_g18=gaia_g18,
            data0=data0,
            hdr=hdr,
            fwhm_px=fwhm,
            wpx=wpx,
            h=h,
            empty_df=pd.DataFrame(),
            frame_label=frame_label,
        )
        od = ctx / "overlays" / iter_id / frame_label
        title = f"{frame_label} {iter_id} thr={meta.get('threshold_sigma')}"
        render_overlay_v2(data0, census, det_x, det_y, out_path=od / "overlay_full.png", title=title)
        for name, box in crop_boxes(wpx, h).items():
            render_overlay_v2(
                data0, census, det_x, det_y, out_path=od / f"overlay_crop_{name}.png", title=f"{title} {name}", crop=box
            )


def pick_best_combined(df: pd.DataFrame) -> str:
    sub = df[(df["frame"] == "MASTERSTAR") & (df["combined"] == True)].copy()  # noqa: E712
    if sub.empty:
        return "comb_thr4.5_p2"
    sub["score"] = sub["g1_eye_le145"] - 10 * sub["g3_g18"] - 5 * sub["g2"].fillna(0)
    return str(sub.sort_values("score", ascending=False).iloc[0]["iter_id"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=Path, default=DEFAULT_CTX)
    ap.add_argument("--skip-detect", action="store_true", help="Only rescore from saved detections")
    args = ap.parse_args()
    ctx = args.ctx
    ctx.mkdir(parents=True, exist_ok=True)
    cfg = AppConfig()
    gaia_db = Path(cfg.gaia_db_path)
    empty_df = pd.read_csv(EMPTY_SKY_CSV) if EMPTY_SKY_CSV.is_file() else pd.DataFrame()
    t0 = time.perf_counter()

    if not args.skip_detect:
        detect_and_save(ctx, gaia_db, SWEEP_SIGMAS, combined=False)
        detect_and_save(ctx, gaia_db, COMBINED_SIGMAS, combined=True)

    df = rescore_all(ctx, gaia_db, empty_df)

    # M1: add g3 rescored column vs iter1 log
    iter1 = ITER1_CTX / "iteration_log_full_sweep.csv"
    if iter1.is_file():
        old = pd.read_csv(iter1)
        old_ms = old[old.frame == "MASTERSTAR"][["threshold_sigma", "g3"]].rename(columns={"g3": "g3_iter1_g16"})
        new_ms = df[(df.frame == "MASTERSTAR") & (~df.combined)][["threshold_sigma", "g3_g16", "g3_g18"]]
        cmp_df = old_ms.merge(new_ms, on="threshold_sigma", how="outer")
        cmp_df.to_csv(ctx / "g3_rescore_comparison.csv", index=False)

    best = pick_best_combined(df)
    regenerate_overlays(ctx, gaia_db, best)
    with open(ctx / "best_combined.json", "w", encoding="utf-8") as f:
        sub = df[(df.iter_id == best) & (df.frame == "MASTERSTAR")].iloc[0]
        json.dump(
            {
                "best_iter_id": best,
                "masterstar_g1_eye_le13": float(sub["g1_eye_le13"]),
                "masterstar_g1_eye_le145": float(sub["g1_eye_le145"]),
                "masterstar_g1_strict_le13": float(sub["g1_strict_le13"]),
                "masterstar_g2": float(sub["g2"]),
                "masterstar_g3_g18": float(sub["g3_g18"]),
                "edge_margin_px": EDGE_MARGIN_PX,
                "n_edge": int(sub["n_edge"]),
            },
            f,
            indent=2,
        )
    print(f"Done in {time.perf_counter() - t0:.1f}s -> {ctx} best={best}")


if __name__ == "__main__":
    main()
