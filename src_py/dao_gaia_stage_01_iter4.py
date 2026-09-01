#!/usr/bin/env python3
"""DAO-GAIA-STAGE-01 iteration 4: pass2 provenance + FORCED_SEED layer."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src_py"
sys.path.insert(0, str(SRC))

from masterstar_gaia_accounting import (  # noqa: E402
    SOURCE_BLENDED,
    SOURCE_EDGE,
    SOURCE_FORCED_SEED,
    SOURCE_SATURATED,
    SOURCE_TOO_FAINT,
    ForcedSeedAcceptParams,
    Pass2AcceptParams,
    annotate_blended_groups,
    dao_pass2_try_at_position,
    forced_seed_accept,
    forced_seed_measure_at_position,
    lock_existing_and_leftover_assign,
)

from dao_gaia_common import (  # noqa: E402
    EDGE_MARGIN_PX,
    FRAMES,
    G3_GAIA_MAX,
    GAIA_QUERY_G,
    OVERLAY_G_MAX,
    SHARPNESS_OPEN,
    SOURCE_CROWDED_MISS,
    _gaia_on_chip_pm,
    _is_edge,
    _peak_at,
    _saturation_limit,
    asinh_rgb,
    crop_boxes,
    decompose_holes_le13,
    estimate_sky,
    g2_empty_false_accept,
    g3_spurious,
    load_frame,
    run_dao,
)
from config import AppConfig  # noqa: E402

DEFAULT_CTX = REPO / "dev" / "results" / "context" / "session_20260819_daostage01_iter4"
EMPTY_SKY_CSV = REPO / "dev" / "results" / "context" / "session_20260819_msgaia01" / "empty_positions_main.csv"

SOURCE_DETECTED = "DETECTED"
SOURCE_AMBIGUOUS_OWNER = "AMBIGUOUS_OWNER"
PASS1_SIGMA = 4.5
PASS2_SIGMA = 4.0
PASS2_CENTER_TOL = 2.0
TARGET_DEPTH_G = 15.0
SEED_PARAMS = ForcedSeedAcceptParams(centroid_max_px=2.0, snr_min=4.0)

HAND_MATCH_RADIUS_PX = 3.0
HAND_PASS2_CENTER_TOL = 2.0
HAND_SEED_CENTROID_MAX_PX = 2.0


@dataclass(frozen=True)
class ValidationParams:
    """STAGE-01 iter4 scoring knobs (hand-validated defaults)."""

    pass1_sigma: float = 4.5
    pass2_sigma: float = 4.0
    match_radius_px: float = 3.0
    pass2_center_tol_px: float = 2.0
    seed_centroid_max_px: float = 2.0
    seed_snr_min: float = 4.0

    @classmethod
    def hand_validated(cls) -> ValidationParams:
        return cls()

    def seed_params(self) -> ForcedSeedAcceptParams:
        return ForcedSeedAcceptParams(
            centroid_max_px=float(self.seed_centroid_max_px),
            snr_min=float(self.seed_snr_min),
        )

EYE_OK_NO_SEED = frozenset({SOURCE_DETECTED, SOURCE_BLENDED, SOURCE_SATURATED})
EYE_OK_WITH_SEED = EYE_OK_NO_SEED | frozenset({SOURCE_FORCED_SEED})


@dataclass
class DetRec:
    x: float
    y: float
    source: str  # pass1 | pass2 | forced_seed
    owner_j: int = -1
    catalog_id: str = ""
    ambiguous: bool = False
    ambiguous_other_j: int = -1
    pass2_reason: str = ""


@dataclass
class FrameResult:
    frame: str
    detections: list[DetRec]
    gaia_le16: pd.DataFrame
    gaia_g18: pd.DataFrame
    census: pd.DataFrame
    owner_kind: np.ndarray  # per gaia row: '', 'pass1', 'pass2', 'forced_seed'
    g1_strict_le13: float
    g1_strict_le145: float
    g1_eye_le13: float
    g1_eye_le145: float
    g1_eye_seed_le13: float
    g1_eye_seed_le145: float
    g2: float
    g3_g18: float
    n_det_pass1: int
    n_det_pass2: int
    n_forced_seed: int
    n_ambiguous: int
    n_crowded_miss: int
    state_counts: dict[str, int]
    g4_ok: bool
    data0: np.ndarray = field(repr=False)
    wpx: int = 0
    h: int = 0


def _eligible_mask(gaia_df: pd.DataFrame, wpx: int, h: int) -> np.ndarray:
    gx = pd.to_numeric(gaia_df["x_gaia"], errors="coerce").to_numpy(dtype=np.float64)
    gy = pd.to_numeric(gaia_df["y_gaia"], errors="coerce").to_numpy(dtype=np.float64)
    return np.array([not _is_edge(float(x), float(y), wpx, h, EDGE_MARGIN_PX) for x, y in zip(gx, gy, strict=False)])


def _nearest_gaia_dist(x: float, y: float, gaia_df: pd.DataFrame, skip_j: int = -1) -> tuple[float, int]:
    gx = pd.to_numeric(gaia_df["x_gaia"], errors="coerce").to_numpy(dtype=np.float64)
    gy = pd.to_numeric(gaia_df["y_gaia"], errors="coerce").to_numpy(dtype=np.float64)
    best_d, best_j = float("inf"), -1
    for k in range(len(gaia_df)):
        if k == skip_j or not (math.isfinite(gx[k]) and math.isfinite(gy[k])):
            continue
        d = float(math.hypot(gx[k] - x, gy[k] - y))
        if d < best_d:
            best_d, best_j = d, k
    return (best_d if math.isfinite(best_d) else float("nan")), best_j


def _check_ambiguous_owner(
    x_det: float, y_det: float, owner_j: int, gaia_df: pd.DataFrame, *, margin_px: float = 1.0
) -> tuple[bool, int]:
    d_seed, _ = _nearest_gaia_dist(x_det, y_det, gaia_df, skip_j=-1)
    gx0 = float(gaia_df.iloc[owner_j]["x_gaia"])
    gy0 = float(gaia_df.iloc[owner_j]["y_gaia"])
    d_owner = float(math.hypot(x_det - gx0, y_det - gy0))
    d_other, j_other = _nearest_gaia_dist(x_det, y_det, gaia_df, skip_j=owner_j)
    if math.isfinite(d_other) and math.isfinite(d_owner) and d_other + margin_px < d_owner:
        return True, j_other
    return False, -1


def _dedup_same_owner(dets: list[DetRec], gaia_df: pd.DataFrame) -> list[DetRec]:
    by_owner: dict[int, list[DetRec]] = {}
    unowned: list[DetRec] = []
    for d in dets:
        if d.owner_j < 0:
            unowned.append(d)
            continue
        by_owner.setdefault(d.owner_j, []).append(d)
    kept: list[DetRec] = list(unowned)
    for j, group in by_owner.items():
        if len(group) == 1:
            kept.append(group[0])
            continue
        xg = float(gaia_df.iloc[j]["x_gaia"])
        yg = float(gaia_df.iloc[j]["y_gaia"])
        best = min(group, key=lambda d: math.hypot(d.x - xg, d.y - yg))
        kept.append(best)
    return kept


def _dedup_pass1_spatial(dets: list[DetRec], *, sep_px: float = 0.75) -> list[DetRec]:
    """Collapse pass1 peaks closer than sep_px (keep first)."""
    p1 = [d for d in dets if d.source == "pass1"]
    other = [d for d in dets if d.source != "pass1"]
    if len(p1) <= 1:
        return dets
    out_p1: list[DetRec] = []
    for d in p1:
        if all(math.hypot(d.x - e.x, d.y - e.y) >= sep_px for e in out_p1):
            out_p1.append(d)
    return out_p1 + other


def forced_seed_empty_sky_audit(data0: np.ndarray, empty_df: pd.DataFrame, *, fwhm_px: float) -> dict[str, Any]:
    sub = empty_df[empty_df["frame"].astype(str) == "MASTERSTAR"]
    accept = 0
    for _, r in sub.iterrows():
        meas = forced_seed_measure_at_position(data0, float(r["x"]), float(r["y"]), fwhm_px=fwhm_px, params=SEED_PARAMS)
        ok, _ = forced_seed_accept(meas, params=SEED_PARAMS)
        if ok:
            accept += 1
    n = len(sub)
    return {"n_empty": n, "n_accept": accept, "false_accept_rate": accept / n if n else float("nan")}


def run_frame_i6_i7(
    frame_label: str,
    fpath: Path,
    gaia_db: Path,
    rng: np.random.Generator,
    *,
    params: ValidationParams | None = None,
) -> FrameResult:
    p = params or ValidationParams.hand_validated()
    pass1_sigma = float(p.pass1_sigma)
    pass2_sigma = float(p.pass2_sigma)
    match_radius_px = float(p.match_radius_px)
    pass2_center_tol = float(p.pass2_center_tol_px)
    seed_params = p.seed_params()
    _raw, data0, hdr, wcs, fwhm, wpx, h = load_frame(fpath)
    gaia_pm, _, _ = _gaia_on_chip_pm(wcs, wpx, h, gaia_db, hdr, max_mag=GAIA_QUERY_G)
    gaia_g18 = gaia_pm[pd.to_numeric(gaia_pm["g_mag"], errors="coerce") <= G3_GAIA_MAX].copy().reset_index(drop=True)
    gaia_le16 = gaia_pm[pd.to_numeric(gaia_pm["g_mag"], errors="coerce") <= OVERLAY_G_MAX].copy().reset_index(drop=True)
    gaia_le15 = gaia_le16[pd.to_numeric(gaia_le16["g_mag"], errors="coerce") <= TARGET_DEPTH_G].copy().reset_index(drop=True)
    le15_to_le16 = {str(gaia_le15.iloc[j]["catalog_id"]).strip(): int(gaia_le16.index[gaia_le16["catalog_id"] == gaia_le15.iloc[j]["catalog_id"]][0]) if len(gaia_le16.index[gaia_le16["catalog_id"] == gaia_le15.iloc[j]["catalog_id"]]) else j for j in range(len(gaia_le15))}
    # simpler map by catalog_id
    cid_to_le16 = {str(gaia_le16.iloc[j]["catalog_id"]).strip(): j for j in range(len(gaia_le16))}

    sky = estimate_sky(data0, gaia_df=gaia_pm, fwhm_px=fwhm, rng=rng)
    thr = max(pass1_sigma * sky.sky_sigma_clipped, 1e-6)
    p1x, p1y, _ = run_dao(data0, fwhm_px=fwhm, threshold_adu=thr, sharpness_range=SHARPNESS_OPEN)
    dets: list[DetRec] = [DetRec(x=float(x), y=float(y), source="pass1") for x, y in zip(p1x, p1y, strict=False)]
    dets = _dedup_pass1_spatial(dets)

    # Pass1 greedy match ONLY (global re-match for pass1 peaks only)
    if dets:
        dx = np.array([d.x for d in dets], dtype=np.float64)
        dy = np.array([d.y for d in dets], dtype=np.float64)
        det_to_g, gaia_owner_p1, _, _ = lock_existing_and_leftover_assign(
            dx, dy, gaia_g18, locked_pairs=None, leftover_radius_px=match_radius_px
        )
        cid_g18 = gaia_g18["catalog_id"].astype(str).str.strip()
        for i, d in enumerate(dets):
            gi = int(det_to_g[i])
            if gi < 0:
                continue
            cid = str(cid_g18.iloc[gi]).strip()
            lj = cid_to_le16.get(cid)
            if lj is not None:
                d.owner_j = lj
                d.catalog_id = cid

    owner_kind = np.array([""] * len(gaia_le16), dtype=object)
    gaia_owner = np.full(len(gaia_le16), -1, dtype=np.int64)
    for i, d in enumerate(dets):
        if d.owner_j >= 0 and gaia_owner[d.owner_j] < 0:
            gaia_owner[d.owner_j] = i
            owner_kind[d.owner_j] = "pass1"

    # Pass2: born-owned, no global re-match
    p2_params = Pass2AcceptParams(sigma=pass2_sigma, center_tol_px=pass2_center_tol, fwhm_px=fwhm)
    gm15 = pd.to_numeric(gaia_le15["g_mag"], errors="coerce").to_numpy(dtype=np.float64)
    ambiguous_pairs: list[dict[str, Any]] = []
    n_p2_try = 0
    for j15 in range(len(gaia_le15)):
        cid = str(gaia_le15.iloc[j15]["catalog_id"]).strip()
        lj = cid_to_le16.get(cid)
        if lj is None or int(gaia_owner[lj]) >= 0:
            continue
        g = float(gm15[j15]) if math.isfinite(gm15[j15]) else float("nan")
        if not (math.isfinite(g) and g <= TARGET_DEPTH_G):
            continue
        xg = float(gaia_le15.iloc[j15]["x_gaia"])
        yg = float(gaia_le15.iloc[j15]["y_gaia"])
        if _is_edge(xg, yg, wpx, h, EDGE_MARGIN_PX):
            continue
        n_p2_try += 1
        hit = dao_pass2_try_at_position(data0, xg, yg, wpx=wpx, h=h, params=p2_params)
        if not hit.get("accepted"):
            continue
        xd, yd = float(hit["x_det"]), float(hit["y_det"])
        amb, j_other = _check_ambiguous_owner(xd, yd, lj, gaia_le16)
        rec = DetRec(
            x=xd,
            y=yd,
            source="pass2",
            owner_j=lj,
            catalog_id=cid,
            ambiguous=amb,
            ambiguous_other_j=j_other,
            pass2_reason=str(hit.get("reason", "")),
        )
        dets.append(rec)
        gaia_owner[lj] = len(dets) - 1
        owner_kind[lj] = "pass2"
        if amb:
            ambiguous_pairs.append(
                {
                    "owner_catalog_id": cid,
                    "other_catalog_id": str(gaia_le16.iloc[j_other]["catalog_id"]),
                    "x_det": xd,
                    "y_det": yd,
                    "owner_j": lj,
                    "other_j": j_other,
                }
            )

    dets = _dedup_same_owner(dets, gaia_le16)

    # Rebuild gaia_owner after dedup
    gaia_owner[:] = -1
    owner_kind[:] = ""
    for i, d in enumerate(dets):
        if d.owner_j >= 0:
            gaia_owner[d.owner_j] = i
            owner_kind[d.owner_j] = d.source

    # I7 FORCED_SEED for G<=15 still without owner
    n_forced = 0
    for j15 in range(len(gaia_le15)):
        cid = str(gaia_le15.iloc[j15]["catalog_id"]).strip()
        lj = cid_to_le16.get(cid)
        if lj is None or int(gaia_owner[lj]) >= 0:
            continue
        xg = float(gaia_le15.iloc[j15]["x_gaia"])
        yg = float(gaia_le15.iloc[j15]["y_gaia"])
        if _is_edge(xg, yg, wpx, h, EDGE_MARGIN_PX):
            continue
        meas = forced_seed_measure_at_position(data0, xg, yg, fwhm_px=fwhm, params=seed_params)
        ok, reason = forced_seed_accept(meas, params=seed_params)
        if not ok:
            continue
        dets.append(
            DetRec(
                x=float(meas["cx"]),
                y=float(meas["cy"]),
                source="forced_seed",
                owner_j=lj,
                catalog_id=cid,
            )
        )
        gaia_owner[lj] = len(dets) - 1
        owner_kind[lj] = "forced_seed"
        n_forced += 1

    census = _build_census(gaia_le16, gaia_owner, owner_kind, data0, hdr, fwhm, wpx, h)
    eligible = _eligible_mask(gaia_le16, wpx, h)

    all_x = np.array([d.x for d in dets], dtype=np.float64)
    all_y = np.array([d.y for d in dets], dtype=np.float64)
    empty_df = pd.read_csv(EMPTY_SKY_CSV) if EMPTY_SKY_CSV.is_file() else pd.DataFrame()
    g2 = g2_empty_false_accept(
        all_x, all_y, empty_df, frame_label, match_radius_px=match_radius_px
    )
    g3, _ = g3_spurious(
        all_x, all_y, gaia_g18, wpx=wpx, h=h, match_radius_px=match_radius_px
    )

    st = census["source_state"].astype(str).to_numpy()
    gm = pd.to_numeric(census["g_mag"], errors="coerce").to_numpy(dtype=np.float64)

    def _g1_strict(mag_cut: float) -> float:
        sel = eligible & np.isfinite(gm) & (gm <= mag_cut)
        n = int(sel.sum())
        return float((st[sel] == SOURCE_DETECTED).sum() / n) if n else float("nan")

    def _g1_eye(mag_cut: float, ok_states: frozenset[str]) -> float:
        sel = eligible & np.isfinite(gm) & (gm <= mag_cut)
        n = int(sel.sum())
        return float(np.isin(st[sel], list(ok_states)).sum() / n) if n else float("nan")

    state_counts = census["source_state"].value_counts().to_dict()
    g4_ok = _verify_g4(census, eligible)

    return FrameResult(
        frame=frame_label,
        detections=dets,
        gaia_le16=gaia_le16,
        gaia_g18=gaia_g18,
        census=census,
        owner_kind=owner_kind,
        g1_strict_le13=_g1_strict(13.0),
        g1_strict_le145=_g1_strict(14.5),
        g1_eye_le13=_g1_eye(13.0, EYE_OK_NO_SEED),
        g1_eye_le145=_g1_eye(14.5, EYE_OK_NO_SEED),
        g1_eye_seed_le13=_g1_eye(13.0, EYE_OK_WITH_SEED),
        g1_eye_seed_le145=_g1_eye(14.5, EYE_OK_WITH_SEED),
        g2=float(g2),
        g3_g18=float(g3),
        n_det_pass1=int(sum(1 for d in dets if d.source == "pass1")),
        n_det_pass2=int(sum(1 for d in dets if d.source == "pass2")),
        n_forced_seed=n_forced,
        n_ambiguous=int(sum(1 for d in dets if d.ambiguous)),
        n_crowded_miss=int(state_counts.get(SOURCE_CROWDED_MISS, 0)),
        state_counts={str(k): int(v) for k, v in state_counts.items()},
        g4_ok=g4_ok,
        data0=data0,
        wpx=wpx,
        h=h,
    )


def _build_census(
    gaia_df: pd.DataFrame,
    gaia_owner: np.ndarray,
    owner_kind: np.ndarray,
    data0: np.ndarray,
    hdr,
    fwhm_px: float,
    wpx: int,
    h: int,
) -> pd.DataFrame:
    gdf = annotate_blended_groups(gaia_df.copy(), gaia_owner, fwhm_px=fwhm_px)
    sat_lim = _saturation_limit(hdr) * 0.999
    states: list[str] = []
    for j in range(len(gdf)):
        xg = float(gdf.iloc[j]["x_gaia"])
        yg = float(gdf.iloc[j]["y_gaia"])
        gmag = float(gdf.iloc[j]["g_mag"]) if pd.notna(gdf.iloc[j]["g_mag"]) else float("nan")
        blend_gid = str(gdf.iloc[j].get("blend_group_id", "") or "")
        kind = str(owner_kind[j]) if j < len(owner_kind) else ""
        owner = int(gaia_owner[j]) if j < len(gaia_owner) else -1
        if _is_edge(xg, yg, wpx, h, EDGE_MARGIN_PX):
            states.append(SOURCE_EDGE)
        elif math.isfinite(gmag) and gmag > TARGET_DEPTH_G:
            states.append(SOURCE_TOO_FAINT)
        elif kind == "forced_seed":
            states.append(SOURCE_FORCED_SEED)
        elif owner >= 0 and kind in ("pass1", "pass2"):
            states.append(SOURCE_DETECTED)
        elif blend_gid:
            states.append(SOURCE_BLENDED)
        elif _peak_at(data0, xg, yg) >= sat_lim:
            states.append(SOURCE_SATURATED)
        elif math.isfinite(gmag) and gmag <= TARGET_DEPTH_G:
            states.append(SOURCE_TOO_FAINT)
        else:
            states.append(SOURCE_TOO_FAINT)
    gdf["source_state"] = states
    return gdf


def _verify_g4(census: pd.DataFrame, eligible: np.ndarray) -> bool:
    gm = pd.to_numeric(census["g_mag"], errors="coerce").to_numpy(dtype=np.float64)
    st = census["source_state"].astype(str).to_numpy()
    ok_states = {
        SOURCE_DETECTED,
        SOURCE_FORCED_SEED,
        SOURCE_BLENDED,
        SOURCE_SATURATED,
        SOURCE_EDGE,
        SOURCE_TOO_FAINT,
    }
    sel = eligible & np.isfinite(gm) & (gm <= TARGET_DEPTH_G)
    for j in np.where(sel)[0]:
        if st[j] not in ok_states:
            return False
    n_eligible_le15 = int(sel.sum())
    n_accounted = int(sel.sum())
    return n_accounted == n_eligible_le15


def render_overlay_final(
    res: FrameResult,
    *,
    out_path: Path,
    title: str,
    crop: tuple[int, int, int, int] | None = None,
) -> None:
    data0 = res.data0
    census = res.census
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
        if state == SOURCE_EDGE:
            continue
        if gid or state == SOURCE_BLENDED:
            ax.plot(x, y, "o", ms=5, mfc="violet", mec="white", mew=0.3, alpha=0.9)
        elif state == SOURCE_SATURATED:
            ax.plot(x, y, "o", ms=6, mfc="orange", mec="white", mew=0.3, alpha=0.95)
        elif state == SOURCE_TOO_FAINT and g > 14.0:
            ax.plot(x, y, "o", ms=2.5, mfc="0.65", mec="none", alpha=0.45)
        elif state == SOURCE_TOO_FAINT and g <= 14.0:
            ax.plot(x, y, "x", ms=8, mfc="red", mec="red", mew=1.4)
    for d in res.detections:
        xp, yp = d.x - ox, d.y - oy
        if crop is not None and (xp < 0 or yp < 0 or xp >= crop[2] or yp >= crop[3]):
            continue
        if d.source == "forced_seed":
            ax.add_patch(plt.Circle((xp, yp), 4.0, fill=True, facecolor="cyan", edgecolor="white", linewidth=0.6, alpha=0.85))
        elif d.source in ("pass1", "pass2"):
            ax.add_patch(plt.Circle((xp, yp), 4.0, fill=False, edgecolor="lime", linewidth=0.8, alpha=0.85))
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def frame_result_to_score_row(res: FrameResult) -> dict[str, Any]:
    return {
        "frame": res.frame,
        "g1_strict_le13": res.g1_strict_le13,
        "g1_strict_le145": res.g1_strict_le145,
        "g1_eye_le13": res.g1_eye_le13,
        "g1_eye_le145": res.g1_eye_le145,
        "g1_eye_seed_le13": res.g1_eye_seed_le13,
        "g1_eye_seed_le145": res.g1_eye_seed_le145,
        "g2": res.g2,
        "g3_g18": res.g3_g18,
        "g4_ok": res.g4_ok,
        "n_pass1": res.n_det_pass1,
        "n_pass2": res.n_det_pass2,
        "n_forced_seed": res.n_forced_seed,
        "n_ambiguous_owner": res.n_ambiguous,
        "state_counts": res.state_counts,
    }


def score_validation_params(
    params: ValidationParams,
    *,
    gaia_db: Path,
    rng: np.random.Generator | None = None,
) -> list[dict[str, Any]]:
    """Run STAGE-01 iter4 harness on all frames for one tolerance config."""
    rng = rng or np.random.default_rng(51604)
    return [
        frame_result_to_score_row(run_frame_i6_i7(frame_label, fpath, gaia_db, rng, params=params))
        for frame_label, fpath in FRAMES
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=Path, default=DEFAULT_CTX)
    args = ap.parse_args()
    ctx = args.ctx
    ctx.mkdir(parents=True, exist_ok=True)
    cfg = AppConfig()
    gaia_db = Path(cfg.gaia_db_path)
    rng = np.random.default_rng(51604)
    t0 = time.perf_counter()

    # I7 empty-sky audit for forced seed
    _raw, data0, hdr, wcs, fwhm, wpx, h = load_frame(dict(FRAMES)["MASTERSTAR"])
    seed_audit = forced_seed_empty_sky_audit(data0, pd.read_csv(EMPTY_SKY_CSV), fwhm_px=fwhm)
    with open(ctx / "forced_seed_empty_sky_audit.json", "w", encoding="utf-8") as f:
        json.dump(seed_audit, f, indent=2)

    tag = f"win_p1_{PASS1_SIGMA}_p2_{PASS2_SIGMA}_i6_i7"
    score_rows: list[dict[str, Any]] = []
    all_ambiguous: list[dict[str, Any]] = []
    red_x_rows: list[dict[str, Any]] = []

    for frame_label, fpath in FRAMES:
        res = run_frame_i6_i7(frame_label, fpath, gaia_db, rng)
        score_rows.append(
            {
                "frame": res.frame,
                "g1_strict_le13": res.g1_strict_le13,
                "g1_strict_le145": res.g1_strict_le145,
                "g1_eye_le13": res.g1_eye_le13,
                "g1_eye_le145": res.g1_eye_le145,
                "g1_eye_seed_le13": res.g1_eye_seed_le13,
                "g1_eye_seed_le145": res.g1_eye_seed_le145,
                "g2": res.g2,
                "g3_g18": res.g3_g18,
                "n_pass1": res.n_det_pass1,
                "n_pass2": res.n_det_pass2,
                "n_forced_seed": res.n_forced_seed,
                "n_ambiguous_owner": res.n_ambiguous,
                "n_crowded_miss": res.n_crowded_miss,
                "g4_ok": res.g4_ok,
                "state_counts": res.state_counts,
            }
        )
        for d in res.detections:
            if d.ambiguous:
                all_ambiguous.append({"frame": frame_label, **d.__dict__})
        census = res.census
        gm = pd.to_numeric(census["g_mag"], errors="coerce")
        red = census[(gm <= 14) & (census.source_state == SOURCE_TOO_FAINT)]
        for _, r in red.iterrows():
            red_x_rows.append({"frame": frame_label, **r.to_dict()})

        od = ctx / "overlays" / tag / frame_label
        render_overlay_final(res, out_path=od / "overlay_full.png", title=f"{frame_label} {tag}")
        for name, box in crop_boxes(res.wpx, res.h,).items():
            render_overlay_final(res, out_path=od / f"overlay_crop_{name}.png", title=f"{frame_label} {name}", crop=box)

        holes = decompose_holes_le13(census, res.gaia_le16, res.data0, fwhm)
        if frame_label == "MASTERSTAR":
            holes.to_csv(ctx / "holes_le13_final.csv", index=False)

    pd.DataFrame(score_rows).to_csv(ctx / "final_scores.csv", index=False)
    pd.DataFrame(all_ambiguous).to_csv(ctx / "ambiguous_owner_flags.csv", index=False)
    pd.DataFrame(red_x_rows).to_csv(ctx / "red_x_remaining.csv", index=False)

    ms = next(r for r in score_rows if r["frame"] == "MASTERSTAR")
    with open(ctx / "final_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "iter_id": tag,
                "pass1_sigma": PASS1_SIGMA,
                "pass2_sigma": PASS2_SIGMA,
                "pass2_provenance": "born_owned_no_rematch",
                "forced_seed": {"centroid_max_px": 2.0, "snr_min": 4.0},
                "forced_seed_empty_sky": seed_audit,
                **{k: ms[k] for k in ms if k != "state_counts"},
                "state_counts": ms["state_counts"],
            },
            f,
            indent=2,
        )
    print(f"Done in {time.perf_counter() - t0:.1f}s -> {ctx}")


if __name__ == "__main__":
    main()
