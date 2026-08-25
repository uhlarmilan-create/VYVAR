#!/usr/bin/env python3
"""DAO-GAIA-STAGE-01 iteration 3: pass2 window fix, PM, I2-I5 audits."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.wcs import WCS
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src_py"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REPO / "tmp"))

from config import AppConfig  # noqa: E402
from masterstar_gaia_accounting import (  # noqa: E402
    SOURCE_BLENDED,
    SOURCE_EDGE,
    SOURCE_SATURATED,
    SOURCE_TOO_FAINT,
    Pass2AcceptParams,
    dao_pass2_try_at_position,
    lock_existing_and_leftover_assign,
)
from pipeline import _query_gaia_local  # noqa: E402
from vyvar_platesolver import _apply_pm_to_gaia_rows, _obs_year_from_header  # noqa: E402

import dao_gaia_stage_01_iter2 as i2  # noqa: E402
from dao_gaia_stage_01 import FRAMES, SHARPNESS_OPEN, crop_boxes, estimate_sky, load_frame, run_dao  # noqa: E402

DEFAULT_CTX = REPO / "dev" / "results" / "context" / "session_20260819_daostage01_iter3"
ITER2_CTX = REPO / "dev" / "results" / "context" / "session_20260819_daostage01_iter2"
EMPTY_SKY_CSV = REPO / "dev" / "results" / "context" / "session_20260819_msgaia01" / "empty_positions_main.csv"

SOURCE_CROWDED_MISS = "CROWDED_MISS"
PASS1_SIGMA = 4.5
PASS2_SIGMA_CANDIDATES = [4.0, 4.5, 5.0]
PASS2_CENTER_TOL = 2.0
TARGET_DEPTH_G = 15.0


def _gaia_on_chip_pm(
    wcs: WCS, wpx: int, h: int, gaia_db: Path, hdr, *, max_mag: float
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Return (pm-corrected on-chip, no-pm on-chip, n_pm_corrected)."""
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


def _nn_gaia_info(j: int, gaia_df: pd.DataFrame) -> tuple[float, float]:
    gx = pd.to_numeric(gaia_df["x_gaia"], errors="coerce").to_numpy(dtype=np.float64)
    gy = pd.to_numeric(gaia_df["y_gaia"], errors="coerce").to_numpy(dtype=np.float64)
    gm = pd.to_numeric(gaia_df["g_mag"], errors="coerce").to_numpy(dtype=np.float64)
    x0, y0 = gx[j], gy[j]
    best_d, best_g = float("inf"), float("nan")
    for k in range(len(gaia_df)):
        if k == j or not (math.isfinite(gx[k]) and math.isfinite(gy[k])):
            continue
        d = float(math.hypot(gx[k] - x0, gy[k] - y0))
        if d < best_d:
            best_d, best_g = d, float(gm[k]) if math.isfinite(gm[k]) else float("nan")
    return (best_d if math.isfinite(best_d) else float("nan")), best_g


def run_pass2_seeds_le15(
    data0: np.ndarray,
    gaia_df: pd.DataFrame,
    gaia_owner: np.ndarray,
    *,
    wpx: int,
    h: int,
    fwhm_px: float,
    pass2_sigma: float,
) -> tuple[np.ndarray, np.ndarray, int, list[dict[str, Any]]]:
    extra_x: list[float] = []
    extra_y: list[float] = []
    audits: list[dict[str, Any]] = []
    n_try = 0
    gm = pd.to_numeric(gaia_df["g_mag"], errors="coerce").to_numpy(dtype=np.float64)
    params = Pass2AcceptParams(
        sigma=float(pass2_sigma),
        center_tol_px=float(PASS2_CENTER_TOL),
        fwhm_px=float(fwhm_px),
    )
    for j in range(len(gaia_df)):
        if int(gaia_owner[j]) >= 0:
            continue
        g = float(gm[j]) if math.isfinite(gm[j]) else float("nan")
        if not (math.isfinite(g) and g <= float(TARGET_DEPTH_G)):
            continue
        xg = float(gaia_df.iloc[j]["x_gaia"])
        yg = float(gaia_df.iloc[j]["y_gaia"])
        if i2._is_edge(xg, yg, wpx, h, i2.EDGE_MARGIN_PX):
            continue
        n_try += 1
        hit = dao_pass2_try_at_position(data0, xg, yg, wpx=wpx, h=h, params=params)
        nn_px, nn_g = _nn_gaia_info(j, gaia_df)
        audits.append(
            {
                "catalog_id": str(gaia_df.iloc[j]["catalog_id"]),
                "g_mag": g,
                "x_seed": xg,
                "y_seed": yg,
                "accepted": bool(hit.get("accepted")),
                "reason": str(hit.get("reason", "")),
                "peak": hit.get("peak"),
                "local_std": hit.get("local_std"),
                "threshold_adu": hit.get("threshold_adu"),
                "centroid_px": hit.get("centroid_px"),
                "x_det": hit.get("x_det"),
                "y_det": hit.get("y_det"),
                "nn_px": nn_px,
                "nn_G": nn_g,
                "local_snr_seed": i2._local_snr(data0, xg, yg, fwhm_px),
            }
        )
        if hit.get("accepted"):
            extra_x.append(float(hit["x_det"]))
            extra_y.append(float(hit["y_det"]))
    ex = np.asarray(extra_x, dtype=np.float64)
    ey = np.asarray(extra_y, dtype=np.float64)
    return ex, ey, n_try, audits


def pass2_empty_sky_audit(
    data0: np.ndarray,
    empty_df: pd.DataFrame,
    *,
    wpx: int,
    h: int,
    fwhm_px: float,
    sigmas: list[float],
) -> pd.DataFrame:
    sub = empty_df[empty_df["frame"].astype(str) == "MASTERSTAR"]
    rows = []
    for sig in sigmas:
        params = Pass2AcceptParams(sigma=float(sig), center_tol_px=PASS2_CENTER_TOL, fwhm_px=fwhm_px)
        accept = 0
        for _, r in sub.iterrows():
            hit = dao_pass2_try_at_position(
                data0, float(r["x"]), float(r["y"]), wpx=wpx, h=h, params=params
            )
            if hit.get("accepted"):
                accept += 1
        n = len(sub)
        rows.append({"pass2_sigma": sig, "n_empty": n, "n_accept": accept, "false_accept_rate": accept / n if n else float("nan")})
    return pd.DataFrame(rows)


def assign_states_v3(
    gaia_df: pd.DataFrame,
    gaia_owner: np.ndarray,
    *,
    data0: np.ndarray,
    hdr,
    fwhm_px: float,
    wpx: int,
    h: int,
    pass2_audit_by_cid: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    census = i2.assign_states(
        gaia_df, gaia_owner, data0=data0, hdr=hdr, fwhm_px=fwhm_px, wpx=wpx, h=h, depth_g=TARGET_DEPTH_G
    )
    pass2_audit_by_cid = pass2_audit_by_cid or {}
    new_states = []
    for j in range(len(census)):
        st = str(census.iloc[j]["source_state"])
        cid = str(census.iloc[j]["catalog_id"]).strip()
        g = float(census.iloc[j]["g_mag"]) if pd.notna(census.iloc[j]["g_mag"]) else float("nan")
        if st != SOURCE_TOO_FAINT:
            new_states.append(st)
            continue
        snr = i2._local_snr(data0, float(census.iloc[j]["x_gaia"]), float(census.iloc[j]["y_gaia"]), fwhm_px)
        aud = pass2_audit_by_cid.get(cid, {})
        if math.isfinite(snr) and snr > 5.0 and math.isfinite(g) and g <= TARGET_DEPTH_G:
            reason = str(aud.get("reason", ""))
            nn_px = float(aud.get("nn_px", float("nan")))
            if reason == "centroid_tol" or (math.isfinite(nn_px) and nn_px < float(fwhm_px)):
                new_states.append(SOURCE_CROWDED_MISS)
            elif reason == "no_detection" and math.isfinite(nn_px) and nn_px < 1.5 * float(fwhm_px):
                new_states.append(SOURCE_CROWDED_MISS)
            else:
                new_states.append(st)
        else:
            new_states.append(st)
    census = census.copy()
    census["source_state"] = new_states
    return census


def g3_anatomy(
    det_x: np.ndarray,
    det_y: np.ndarray,
    gaia_pm: pd.DataFrame,
    gaia_nopm: pd.DataFrame,
    *,
    wpx: int,
    h: int,
) -> pd.DataFrame:
    rows = []
    gx = pd.to_numeric(gaia_pm["x_gaia"], errors="coerce").to_numpy(dtype=np.float64)
    gy = pd.to_numeric(gaia_pm["y_gaia"], errors="coerce").to_numpy(dtype=np.float64)
    gm = pd.to_numeric(gaia_pm["g_mag"], errors="coerce").to_numpy(dtype=np.float64)
    tree_pm = cKDTree(np.column_stack([gx, gy]))
    gx0 = pd.to_numeric(gaia_nopm["x_gaia"], errors="coerce").to_numpy(dtype=np.float64)
    gy0 = pd.to_numeric(gaia_nopm["y_gaia"], errors="coerce").to_numpy(dtype=np.float64)
    tree_nopm = cKDTree(np.column_stack([gx0, gy0]))

    for i, (x, y) in enumerate(zip(det_x, det_y, strict=False)):
        if i2._is_corner(float(x), float(y), wpx, h):
            continue
        d3, idx3 = tree_pm.query([float(x), float(y)], distance_upper_bound=3.0)
        d5, idx5 = tree_pm.query([float(x), float(y)], distance_upper_bound=5.0)
        d3n, _ = tree_nopm.query([float(x), float(y)], distance_upper_bound=3.0)
        matched_3 = math.isfinite(float(d3)) and float(d3) <= 3.0
        if matched_3:
            continue
        ng = float(gm[int(idx3)]) if matched_3 and int(idx3) < len(gm) else float("nan")
        if math.isfinite(float(d3)) and int(idx3) < len(gm):
            ng = float(gm[int(idx3)])
        elif math.isfinite(float(d5)) and int(idx5) < len(gm):
            ng = float(gm[int(idx5)])
        else:
            ng = float("nan")
        nd = float(d3) if math.isfinite(float(d3)) else float(d5) if math.isfinite(float(d5)) else float("nan")
        ndn = float(d3n) if math.isfinite(float(d3n)) else float("nan")
        if math.isfinite(nd) and nd <= 5.0:
            klass = "poor_centroid_3to5px"
        elif math.isfinite(ng) and ng > 17.5:
            klass = "catalog_faint_neighbor"
        elif math.isfinite(ndn) and math.isfinite(nd) and ndn <= 3.0 and nd > 3.0:
            klass = "pm_offset_resolved"
        elif math.isfinite(nd) and nd > 5.0:
            klass = "no_gaia_g18_artifact_or_real"
        else:
            klass = "unmatched_other"
        rows.append(
            {
                "det_i": i,
                "x": float(x),
                "y": float(y),
                "corner_excluded": False,
                "nearest_gaia_dist_px": nd,
                "nearest_gaia_dist_px_nopm": ndn,
                "nearest_gaia_G": ng,
                "match_5px": bool(math.isfinite(float(d5)) and float(d5) <= 5.0),
                "pm_shifts_match": bool(math.isfinite(ndn) and ndn <= 3.0 and (not math.isfinite(nd) or nd > 3.0)),
                "class": klass,
            }
        )
    return pd.DataFrame(rows)


def render_overlay_v3(
    data0: np.ndarray,
    census: pd.DataFrame,
    det_x: np.ndarray,
    det_y: np.ndarray,
    *,
    out_path: Path,
    title: str,
    crop: tuple[int, int, int, int] | None = None,
) -> None:
    import matplotlib.pyplot as plt

    rgb = i2.asinh_rgb(data0)
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
        if g > i2.OVERLAY_G_MAX:
            continue
        x, y = gx[j] - ox, gy[j] - oy
        if crop is not None and (x < 0 or y < 0 or x >= crop[2] or y >= crop[3]):
            continue
        state = str(st[j])
        gid = str(blend_col.iloc[j] if j < len(blend_col) else "")
        if state == SOURCE_EDGE:
            continue
        if gid or state == SOURCE_BLENDED:
            ax.plot(x, y, "o", ms=5, mfc="violet", mec="white", mew=0.3, alpha=0.9)
        elif state == SOURCE_SATURATED:
            ax.plot(x, y, "o", ms=6, mfc="orange", mec="white", mew=0.3, alpha=0.95)
        elif state == SOURCE_CROWDED_MISS:
            ax.plot(x, y, "o", ms=4, mfc="gold", mec="k", mew=0.2, alpha=0.85)
        elif state == SOURCE_TOO_FAINT and g > 14.0:
            ax.plot(x, y, "o", ms=2.5, mfc="0.65", mec="none", alpha=0.45)
        elif state == SOURCE_TOO_FAINT and g <= 14.0:
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


def run_combined_frame(
    frame_label: str,
    fpath: Path,
    gaia_db: Path,
    *,
    pass1_sigma: float,
    pass2_sigma: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    _raw, data0, hdr, wcs, fwhm, wpx, h = load_frame(fpath)
    gaia_pm, gaia_nopm, n_pm = _gaia_on_chip_pm(wcs, wpx, h, gaia_db, hdr, max_mag=i2.GAIA_QUERY_G)
    gaia_g18 = gaia_pm[pd.to_numeric(gaia_pm["g_mag"], errors="coerce") <= i2.G3_GAIA_MAX].copy()
    gaia_le16 = gaia_pm[pd.to_numeric(gaia_pm["g_mag"], errors="coerce") <= i2.OVERLAY_G_MAX].copy()
    sky = estimate_sky(data0, gaia_df=gaia_pm, fwhm_px=fwhm, rng=rng)
    thr = max(float(pass1_sigma) * sky.sky_sigma_clipped, 1e-6)
    det_x, det_y, _ = run_dao(data0, fwhm_px=fwhm, threshold_adu=thr, sharpness_range=SHARPNESS_OPEN)
    _, owner_pre, _, _ = lock_existing_and_leftover_assign(
        det_x, det_y, gaia_g18, locked_pairs=None, leftover_radius_px=i2.MATCH_RADIUS_PX
    )
    cid_all = gaia_g18["catalog_id"].astype(str).str.strip()
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
    ex, ey, n_try, audits = run_pass2_seeds_le15(
        data0, gaia_le16, owner_le16, wpx=wpx, h=h, fwhm_px=fwhm, pass2_sigma=pass2_sigma
    )
    if ex.size:
        det_x = np.concatenate([det_x, ex])
        det_y = np.concatenate([det_y, ey])
    audit_map = {str(a["catalog_id"]): a for a in audits}
    bundle, census = i2.score_detections(
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
        empty_df=pd.read_csv(EMPTY_SKY_CSV) if EMPTY_SKY_CSV.is_file() else pd.DataFrame(),
        frame_label=frame_label,
    )
    census = assign_states_v3(
        gaia_le16,
        np.array(
            [
                next(
                    (
                        i
                        for i in range(len(det_x))
                        if i2.MATCH_RADIUS_PX
                        and math.hypot(
                            det_x[i] - float(gaia_le16.iloc[j]["x_gaia"]),
                            det_y[i] - float(gaia_le16.iloc[j]["y_gaia"]),
                        )
                        <= i2.MATCH_RADIUS_PX
                    ),
                    -1,
                )
                for j in range(len(gaia_le16))
            ],
            dtype=np.int64,
        ),
        data0=data0,
        hdr=hdr,
        fwhm_px=fwhm,
        wpx=wpx,
        h=h,
        pass2_audit_by_cid=audit_map,
    )
    # Recompute owner for census v3
    det_to_g, _, _, _ = lock_existing_and_leftover_assign(
        det_x, det_y, gaia_g18, locked_pairs=None, leftover_radius_px=i2.MATCH_RADIUS_PX
    )
    cid_full = gaia_g18["catalog_id"].astype(str).str.strip()
    cid_to_j2 = {str(gaia_le16.iloc[j]["catalog_id"]).strip(): j for j in range(len(gaia_le16))}
    owner = np.full(len(gaia_le16), -1, dtype=np.int64)
    for i in range(len(det_x)):
        gi = int(det_to_g[i])
        if gi < 0:
            continue
        lj = cid_to_j2.get(str(cid_full.iloc[gi]).strip())
        if lj is not None and owner[lj] < 0:
            owner[lj] = i
    census = assign_states_v3(
        gaia_le16, owner, data0=data0, hdr=hdr, fwhm_px=fwhm, wpx=wpx, h=h, pass2_audit_by_cid=audit_map
    )
    eligible = i2._eligible_mask(gaia_le16, wpx, h, i2.EDGE_MARGIN_PX)
    g1_eye_13 = i2.g1_eye(census, 13.0, eligible)
    g1_eye_145 = i2.g1_eye(census, 14.5, eligible)
    return {
        "frame": frame_label,
        "det_x": det_x,
        "det_y": det_y,
        "census": census,
        "data0": data0,
        "hdr": hdr,
        "wpx": wpx,
        "h": h,
        "fwhm": fwhm,
        "gaia_pm": gaia_pm,
        "gaia_nopm": gaia_nopm,
        "gaia_g18": gaia_g18,
        "gaia_le16": gaia_le16,
        "n_pm": n_pm,
        "n_pass2_try": n_try,
        "n_pass2_added": int(ex.size),
        "pass2_audits": audits,
        "g1_eye_le13": g1_eye_13,
        "g1_eye_le145": g1_eye_145,
        "g3_g18": bundle.g3_g18,
        "g2": bundle.g2,
        "state_counts": census["source_state"].value_counts().to_dict(),
    }


def rescore_iter2_g3_pm(ctx: Path, gaia_db: Path) -> pd.DataFrame:
    """M1 follow-up + I3 prep: rescore iter2 saved dets with PM."""
    rows = []
    det_root = ITER2_CTX / "detections" / "thr4.5_sharp_open"
    fpath = dict(FRAMES)["MASTERSTAR"]
    _raw, data0, hdr, wcs, fwhm, wpx, h = load_frame(fpath)
    gaia_pm, gaia_nopm, n_pm = _gaia_on_chip_pm(wcs, wpx, h, gaia_db, hdr, max_mag=i2.GAIA_QUERY_G)
    gaia_g18 = gaia_pm[pd.to_numeric(gaia_pm["g_mag"], errors="coerce") <= i2.G3_GAIA_MAX]
    gaia_g18_nopm = gaia_nopm[pd.to_numeric(gaia_nopm["g_mag"], errors="coerce") <= i2.G3_GAIA_MAX]
    npz = det_root / "MASTERSTAR.npz"
    if npz.is_file():
        det_x, det_y, _ = i2.load_detections(npz)
        g3_pm, nsp_pm = i2.g3_spurious(det_x, det_y, gaia_g18, wpx=wpx, h=h)
        g3_nopm, nsp_nopm = i2.g3_spurious(det_x, det_y, gaia_g18_nopm, wpx=wpx, h=h)
        rows.append({"g3_pm": g3_pm, "n_sp_pm": nsp_pm, "g3_nopm": g3_nopm, "n_sp_nopm": nsp_nopm, "n_pm_corrected": n_pm})
        anatomy = g3_anatomy(det_x, det_y, gaia_g18, gaia_g18_nopm, wpx=wpx, h=h)
        anatomy.to_csv(ctx / "g3_anatomy_thr45.csv", index=False)
    return pd.DataFrame(rows)


def audit_true_misses_i2(ctx: Path, gaia_db: Path) -> pd.DataFrame:
    miss_path = ITER2_CTX / "holes_le13_true_miss_thr45.csv"
    if not miss_path.is_file():
        return pd.DataFrame()
    misses = pd.read_csv(miss_path)
    fpath = dict(FRAMES)["MASTERSTAR"]
    _raw, data0, hdr, wcs, fwhm, wpx, h = load_frame(fpath)
    gaia_pm, _, _ = _gaia_on_chip_pm(wcs, wpx, h, gaia_db, hdr, max_mag=i2.GAIA_QUERY_G)
    cid_to_j = {str(gaia_pm.iloc[j]["catalog_id"]).strip(): j for j in range(len(gaia_pm))}
    params = Pass2AcceptParams(sigma=5.0, center_tol_px=PASS2_CENTER_TOL, fwhm_px=fwhm)
    rows = []
    for _, r in misses.iterrows():
        cid = str(r["catalog_id"]).strip()
        xg, yg = float(r["x"]), float(r["y"])
        hit = dao_pass2_try_at_position(data0, xg, yg, wpx=wpx, h=h, params=params)
        j = cid_to_j.get(cid)
        nn_px, nn_g = (_nn_gaia_info(j, gaia_pm) if j is not None else (float("nan"), float("nan")))
        mechanism = str(hit.get("reason", ""))
        if hit.get("accepted"):
            mechanism = "accepted"
        elif mechanism == "centroid_tol":
            mechanism = "centroid_pulled_by_neighbour_beyond_2px"
        elif mechanism == "no_detection":
            if math.isfinite(nn_px) and nn_px < fwhm:
                mechanism = "crowded_no_peak_in_cutout"
            else:
                mechanism = "no_detection_isolated"
        rows.append(
            {
                **r.to_dict(),
                "pass2_accepted": bool(hit.get("accepted")),
                "pass2_reason": hit.get("reason"),
                "mechanism": mechanism,
                "peak": hit.get("peak"),
                "local_std": hit.get("local_std"),
                "threshold_adu": hit.get("threshold_adu"),
                "centroid_px": hit.get("centroid_px"),
                "nn_G": nn_g,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(ctx / "i2_true_miss_pass2_audit.csv", index=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=Path, default=DEFAULT_CTX)
    args = ap.parse_args()
    ctx = args.ctx
    ctx.mkdir(parents=True, exist_ok=True)
    cfg = AppConfig()
    gaia_db = Path(cfg.gaia_db_path)
    t0 = time.perf_counter()
    rng = np.random.default_rng(51603)

    # I3 + PM check on iter2 detections
    pm_df = rescore_iter2_g3_pm(ctx, gaia_db)
    pm_df.to_csv(ctx / "g3_pm_rescore_thr45.csv", index=False)

    # I2 audit of iter2 11 true misses
    audit_true_misses_i2(ctx, gaia_db)

    # I4 empty-sky pass2 sigma pick
    _raw, data0, hdr, wcs, fwhm, wpx, h = load_frame(dict(FRAMES)["MASTERSTAR"])
    empty_df = pd.read_csv(EMPTY_SKY_CSV) if EMPTY_SKY_CSV.is_file() else pd.DataFrame()
    i4_df = pass2_empty_sky_audit(data0, empty_df, wpx=wpx, h=h, fwhm_px=fwhm, sigmas=PASS2_SIGMA_CANDIDATES)
    i4_df.to_csv(ctx / "i4_pass2_empty_sky_audit.csv", index=False)
    ok = i4_df[i4_df["false_accept_rate"] <= 0.01]
    pass2_sigma = float(ok["pass2_sigma"].min()) if not ok.empty else 5.0

    # I1 + D: winning combined config
    win_tag = f"win_p1_{PASS1_SIGMA}_p2_{pass2_sigma}"
    results = []
    all_audits: list[dict[str, Any]] = []
    for frame_label, fpath in FRAMES:
        res = run_combined_frame(
            frame_label, fpath, gaia_db, pass1_sigma=PASS1_SIGMA, pass2_sigma=pass2_sigma, rng=rng
        )
        results.append({k: v for k, v in res.items() if k not in ("det_x", "det_y", "census", "data0", "hdr")})
        all_audits.extend(res.get("pass2_audits", []))
        od = ctx / "overlays" / win_tag / frame_label
        for name, box in crop_boxes(res["wpx"], res["h"]).items():
            render_overlay_v3(
                res["data0"],
                res["census"],
                res["det_x"],
                res["det_y"],
                out_path=od / f"overlay_crop_{name}.png",
                title=f"{frame_label} {win_tag} {name}",
                crop=box,
            )
        render_overlay_v3(
            res["data0"],
            res["census"],
            res["det_x"],
            res["det_y"],
            out_path=od / "overlay_full.png",
            title=f"{frame_label} {win_tag}",
        )
        i2.save_detections(
            ctx / "detections" / win_tag / f"{frame_label}.npz",
            res["det_x"],
            res["det_y"],
            {"pass1_sigma": PASS1_SIGMA, "pass2_sigma": pass2_sigma, "frame": frame_label},
        )
        if frame_label == "MASTERSTAR":
            holes, summ = i2.decompose_holes_le13(res["census"], res["gaia_le16"], res["data0"], res["fwhm"])
            holes.to_csv(ctx / "holes_le13_after_win.csv", index=False)
            summ.to_csv(ctx / "holes_le13_decompose_after_win.csv", index=False)

    pd.DataFrame(results).to_csv(ctx / "winning_config_scores.csv", index=False)
    pd.DataFrame(all_audits).to_csv(ctx / "pass2_seed_audit_all.csv", index=False)

    ms = next(r for r in results if r["frame"] == "MASTERSTAR")
    with open(ctx / "winning_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "iter_id": win_tag,
                "pass1_sigma": PASS1_SIGMA,
                "pass2_sigma": pass2_sigma,
                "pass2_window": "G<=15",
                "pass2_center_tol_px": PASS2_CENTER_TOL,
                "pm_corrected": True,
                "n_pm_corrected": ms["n_pm"],
                "masterstar_g1_eye_le13": ms["g1_eye_le13"],
                "masterstar_g1_eye_le145": ms["g1_eye_le145"],
                "masterstar_g2": ms["g2"],
                "masterstar_g3_g18": ms["g3_g18"],
                "state_counts": ms["state_counts"],
            },
            f,
            indent=2,
        )
    print(f"Done in {time.perf_counter() - t0:.1f}s pass2_sigma={pass2_sigma} -> {ctx}")


if __name__ == "__main__":
    main()
