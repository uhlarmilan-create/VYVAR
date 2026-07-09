"""Gaia<->DAO field accounting: G_lim, cone decomposition, unmatched-DAO classification.

Shared by ``scripts/dao_reconcile_diag.py`` and ``pipeline.py`` (``pipeline_meta.json``).
Blend radius follows ``crowding_index._build_blend_targets_df``: 1.5  FWHM [px].
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from gaia_catalog_id import normalize_gaia_source_id_series

BLEND_FWHM_FACTOR = 1.5
DEFAULT_G_LIM_PERCENTILE = 95.0
COLLINEAR_MIN_POINTS = 3
COLLINEAR_MAX_PERP_PX = 2.0


def _finite_mag_series(df: pd.DataFrame, cols: tuple[str, ...]) -> pd.Series:
    for col in cols:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            if s.notna().any():
                return s
    return pd.Series(dtype=float)


def _normalize_ids(df: pd.DataFrame, col: str = "catalog_id") -> pd.Series:
    if col not in df.columns:
        return pd.Series([""] * len(df), dtype=str)
    return normalize_gaia_source_id_series(df[col]).fillna("").astype(str).str.strip()


def estimate_g_lim(
    mags: pd.Series | np.ndarray,
    *,
    percentile: float = DEFAULT_G_LIM_PERCENTILE,
) -> float | None:
    """Frame detection limit from matched-star Gaia G distribution (default p95)."""
    arr = pd.to_numeric(pd.Series(mags), errors="coerce").to_numpy(dtype=np.float64)
    ok = np.isfinite(arr)
    if int(np.count_nonzero(ok)) < 3:
        return None
    return float(np.percentile(arr[ok], float(percentile)))


def blend_radius_px(fwhm_px: float, *, factor: float = BLEND_FWHM_FACTOR) -> float:
    return float(factor) * float(fwhm_px)


def blend_radius_arcsec(fwhm_px: float, plate_scale_arcsec: float, *, factor: float = BLEND_FWHM_FACTOR) -> float:
    return blend_radius_px(fwhm_px, factor=factor) * float(plate_scale_arcsec)


def project_ra_dec_to_pixel(
    wcs: Any,
    ra_deg: np.ndarray | pd.Series,
    dec_deg: np.ndarray | pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    ra = np.asarray(ra_deg, dtype=np.float64)
    de = np.asarray(dec_deg, dtype=np.float64)
    ok = np.isfinite(ra) & np.isfinite(de)
    xp = np.full(len(ra), np.nan, dtype=np.float64)
    yp = np.full(len(de), np.nan, dtype=np.float64)
    if not np.any(ok):
        return xp, yp
    pix = wcs.all_world2pix(ra[ok], de[ok], 0)
    xp[ok] = np.asarray(pix[0], dtype=np.float64)
    yp[ok] = np.asarray(pix[1], dtype=np.float64)
    return xp, yp


def _matched_detection_mask(detections_df: pd.DataFrame) -> np.ndarray:
    cid = _normalize_ids(detections_df)
    return (cid != "").to_numpy(dtype=bool)


def _matched_catalog_id_set(detections_df: pd.DataFrame) -> set[str]:
    cid = _normalize_ids(detections_df)
    return set(cid[cid != ""].unique())


def _build_matched_xy(detections_df: pd.DataFrame) -> np.ndarray:
    m = _matched_detection_mask(detections_df)
    if not np.any(m):
        return np.empty((0, 2), dtype=np.float64)
    sub = detections_df.loc[m]
    x = pd.to_numeric(sub["x"], errors="coerce").to_numpy(dtype=np.float64)
    y = pd.to_numeric(sub["y"], errors="coerce").to_numpy(dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    return np.column_stack([x[ok], y[ok]])


def is_blended_with_matched(
    x: float,
    y: float,
    matched_xy: np.ndarray,
    *,
    blend_r_px: float,
) -> bool:
    if matched_xy.size == 0 or not (math.isfinite(x) and math.isfinite(y)):
        return False
    d2 = np.sum((matched_xy - np.array([x, y], dtype=np.float64)) ** 2, axis=1)
    return bool(np.min(d2) <= float(blend_r_px) ** 2)


def decompose_undetected_cone(
    cone_df: pd.DataFrame,
    detections_df: pd.DataFrame,
    *,
    g_lim: float,
    fwhm_px: float,
    wcs: Any | None = None,
    blend_factor: float = BLEND_FWHM_FACTOR,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Label each cone row: matched | below_limit | blended | genuinely_missed."""
    if cone_df.empty:
        return pd.DataFrame(), {
            "n_gaia_matched": 0,
            "n_gaia_below_limit": 0,
            "n_gaia_blended": 0,
            "n_gaia_missed": 0,
        }

    cone = cone_df.copy()
    cone["_cid"] = _normalize_ids(cone)
    cone["_mag"] = pd.to_numeric(cone.get("mag"), errors="coerce")
    matched_ids = _matched_catalog_id_set(detections_df)
    cone["_bucket"] = np.where(cone["_cid"].isin(matched_ids), "matched", "undetected")

    matched_xy = _build_matched_xy(detections_df)
    blend_r = blend_radius_px(fwhm_px, factor=blend_factor)

    if wcs is not None and {"ra_deg", "dec_deg"}.issubset(cone.columns):
        cx, cy = project_ra_dec_to_pixel(wcs, cone["ra_deg"], cone["dec_deg"])
        cone["_x_pix"] = cx
        cone["_y_pix"] = cy
    else:
        cone["_x_pix"] = np.nan
        cone["_y_pix"] = np.nan

    und = cone["_bucket"] == "undetected"
    below = und & cone["_mag"].gt(float(g_lim))
    cone.loc[below, "_bucket"] = "below_limit"

    rest = und & ~below
    blended_mask = np.zeros(len(cone), dtype=bool)
    if np.any(rest) and matched_xy.size > 0:
        from scipy.spatial import cKDTree

        tree = cKDTree(matched_xy)
        idxs = np.where(rest.to_numpy())[0]
        pts = np.column_stack(
            [
                cone.loc[rest, "_x_pix"].to_numpy(dtype=np.float64),
                cone.loc[rest, "_y_pix"].to_numpy(dtype=np.float64),
            ]
        )
        ok_pt = np.isfinite(pts).all(axis=1)
        if np.any(ok_pt):
            dists, _ = tree.query(pts[ok_pt], k=1)
            blended_local = dists <= blend_r
            blended_mask[idxs[ok_pt]] = blended_local
    cone.loc[blended_mask, "_bucket"] = "blended"
    missed = und & ~below & ~blended_mask
    cone.loc[missed, "_bucket"] = "genuinely_missed"

    counts = {
        "n_gaia_matched": int((cone["_bucket"] == "matched").sum()),
        "n_gaia_below_limit": int((cone["_bucket"] == "below_limit").sum()),
        "n_gaia_blended": int((cone["_bucket"] == "blended").sum()),
        "n_gaia_missed": int((cone["_bucket"] == "genuinely_missed").sum()),
    }
    return cone, counts


def corrected_completeness_pct(n_matched: int, n_missed: int) -> float | None:
    denom = int(n_matched) + int(n_missed)
    if denom <= 0:
        return None
    return round(100.0 * float(n_matched) / float(denom), 2)


def raw_completeness_pct(n_matched_unique: int, catalog_rows: int) -> float | None:
    if int(catalog_rows) <= 0:
        return None
    return round(100.0 * float(n_matched_unique) / float(catalog_rows), 2)


def _perp_distances_to_line(xy: np.ndarray, p0: np.ndarray, direction: np.ndarray) -> np.ndarray:
    v = direction / (np.linalg.norm(direction) + 1e-12)
    rel = xy - p0
    along = np.dot(rel, v)
    perp = rel - np.outer(along, v)
    return np.linalg.norm(perp, axis=1)


def find_largest_collinear_group(
    xy: np.ndarray,
    *,
    min_points: int = COLLINEAR_MIN_POINTS,
    max_perp_px: float = COLLINEAR_MAX_PERP_PX,
) -> dict[str, Any]:
    """Return the largest subset consistent with a straight line (satellite-trail probe)."""
    pts = np.asarray(xy, dtype=np.float64)
    ok = np.isfinite(pts).all(axis=1)
    pts = pts[ok]
    n = len(pts)
    if n < min_points:
        return {"n_collinear": 0, "inlier_indices": [], "consistent_with_line": False}

    best_n = 0
    best_mask: np.ndarray | None = None
    for i in range(n):
        for j in range(i + 1, n):
            direction = pts[j] - pts[i]
            if float(np.linalg.norm(direction)) < 1e-6:
                continue
            perp = _perp_distances_to_line(pts, pts[i], direction)
            mask = perp <= float(max_perp_px)
            cnt = int(mask.sum())
            if cnt > best_n:
                best_n = cnt
                best_mask = mask

    if best_mask is None or best_n < min_points:
        return {"n_collinear": 0, "inlier_indices": [], "consistent_with_line": False}

    inlier_idx = np.where(ok)[0][best_mask].tolist()
    return {
        "n_collinear": best_n,
        "inlier_indices": inlier_idx,
        "consistent_with_line": bool(best_n >= min_points),
    }


def classify_unmatched_dao(
    detections_df: pd.DataFrame,
    *,
    collinear_min_points: int = COLLINEAR_MIN_POINTS,
    collinear_max_perp_px: float = COLLINEAR_MAX_PERP_PX,
) -> dict[str, Any]:
    """DAO detections without ``catalog_id``: flux/peak stats + collinearity artifact probe."""
    cid = _normalize_ids(detections_df)
    unmatched = detections_df.loc[cid == ""].copy()
    n_total = int(len(unmatched))
    if n_total == 0:
        return {
            "n_dao_unmatched": 0,
            "n_artifact_candidates": 0,
            "n_unexplained": 0,
            "collinearity": find_largest_collinear_group(np.empty((0, 2))),
            "flux": {},
            "peak_dao": {},
            "sharpness": {},
            "roundness": {},
            "classification": "none",
        }

    flux = _finite_mag_series(unmatched, ("flux", "dao_flux"))
    peak = _finite_mag_series(unmatched, ("peak_dao", "peak_max_adu"))
    sharp = _finite_mag_series(unmatched, ("sharpness",))
    roundness = _finite_mag_series(unmatched, ("roundness_mean", "roundness"))

    xy = np.column_stack(
        [
            pd.to_numeric(unmatched.get("x"), errors="coerce").to_numpy(dtype=np.float64),
            pd.to_numeric(unmatched.get("y"), errors="coerce").to_numpy(dtype=np.float64),
        ]
    )
    collin = find_largest_collinear_group(
        xy,
        min_points=collinear_min_points,
        max_perp_px=collinear_max_perp_px,
    )

    artifact_mask = np.zeros(n_total, dtype=bool)
    if collin["consistent_with_line"] and collin["inlier_indices"]:
        artifact_mask[collin["inlier_indices"]] = True

    # Isolated hot-pixel / cosmic candidates: peak well above median unmatched peak.
    if peak.notna().sum() >= 5:
        med_peak = float(peak.median())
        p90_peak = float(peak.quantile(0.9))
        hot_thr = max(med_peak * 3.0, p90_peak * 1.5)
        artifact_mask |= peak.to_numpy(dtype=np.float64) >= hot_thr

    n_artifact = int(artifact_mask.sum())
    n_unexplained = int(n_total - n_artifact)
    classification = "artifact_dominant" if n_artifact >= max(1, n_total // 2) else "mixed"

    def _dist_stats(s: pd.Series) -> dict[str, float | None]:
        arr = pd.to_numeric(s, errors="coerce").dropna()
        if arr.empty:
            return {"n": 0, "p50": None, "p90": None, "max": None}
        return {
            "n": int(len(arr)),
            "p50": round(float(arr.quantile(0.5)), 4),
            "p90": round(float(arr.quantile(0.9)), 4),
            "max": round(float(arr.max()), 4),
        }

    return {
        "n_dao_unmatched": n_total,
        "n_artifact_candidates": n_artifact,
        "n_unexplained": n_unexplained,
        "classification": classification,
        "collinearity": collin,
        "flux": _dist_stats(flux),
        "peak_dao": _dist_stats(peak),
        "sharpness": _dist_stats(sharp),
        "roundness": _dist_stats(roundness),
    }


def compute_gaia_dao_reconcile(
    cone_df: pd.DataFrame,
    detections_df: pd.DataFrame,
    *,
    fwhm_px: float,
    plate_scale_arcsec: float | None = None,
    wcs: Any | None = None,
    g_lim_percentile: float = DEFAULT_G_LIM_PERCENTILE,
    blend_factor: float = BLEND_FWHM_FACTOR,
) -> dict[str, Any]:
    """Full Gaia<->DAO reconciliation report (diagnostic + pipeline meta source)."""
    matched_mask = _matched_detection_mask(detections_df)
    matched_df = detections_df.loc[matched_mask]
    matched_mags = _finite_mag_series(matched_df, ("phot_g_mean_mag", "mag", "catalog_mag"))
    g_lim = estimate_g_lim(matched_mags, percentile=g_lim_percentile)

    if g_lim is None:
        g_lim = estimate_g_lim(pd.to_numeric(cone_df.get("mag"), errors="coerce"), percentile=g_lim_percentile)

    if g_lim is None:
        raise ValueError("Cannot estimate G_lim: insufficient matched-star magnitudes")

    labeled, counts = decompose_undetected_cone(
        cone_df,
        detections_df,
        g_lim=float(g_lim),
        fwhm_px=float(fwhm_px),
        wcs=wcs,
        blend_factor=blend_factor,
    )

    cid = _normalize_ids(detections_df)
    n_matched_unique = int(cid[cid != ""].nunique())
    catalog_rows = int(len(cone_df))
    raw_pct = raw_completeness_pct(n_matched_unique, catalog_rows)
    corr_pct = corrected_completeness_pct(counts["n_gaia_matched"], counts["n_gaia_missed"])
    unmatched_dao = classify_unmatched_dao(detections_df)

    if plate_scale_arcsec is None and wcs is not None:
        try:
            from astropy.wcs.utils import proj_plane_pixel_scales

            scales = proj_plane_pixel_scales(wcs) * 3600.0
            plate_scale_arcsec = float(np.mean(scales))
        except Exception:  # noqa: BLE001
            plate_scale_arcsec = None

    blend_r_px = blend_radius_px(fwhm_px, factor=blend_factor)
    blend_r_arcsec = (
        blend_radius_arcsec(fwhm_px, plate_scale_arcsec, factor=blend_factor)
        if plate_scale_arcsec is not None
        else None
    )

    matched_mag_arr = pd.to_numeric(matched_mags, errors="coerce").dropna()
    g_lim_stats = {
        "percentile": float(g_lim_percentile),
        "p50": round(float(matched_mag_arr.quantile(0.5)), 4) if not matched_mag_arr.empty else None,
        "p90": round(float(matched_mag_arr.quantile(0.9)), 4) if not matched_mag_arr.empty else None,
        "p95": round(float(matched_mag_arr.quantile(0.95)), 4) if not matched_mag_arr.empty else None,
        "max": round(float(matched_mag_arr.max()), 4) if not matched_mag_arr.empty else None,
        "n_matched_stars": int(len(matched_mag_arr)),
    }

    return {
        "g_lim_est": round(float(g_lim), 4),
        "g_lim_stats": g_lim_stats,
        "fwhm_px": round(float(fwhm_px), 4),
        "blend_factor_fwhm": float(blend_factor),
        "blend_radius_px": round(blend_r_px, 4),
        "blend_radius_arcsec": round(blend_r_arcsec, 4) if blend_r_arcsec is not None else None,
        "plate_scale_arcsec": round(float(plate_scale_arcsec), 6) if plate_scale_arcsec is not None else None,
        "catalog_rows": catalog_rows,
        "n_gaia_matched_unique": n_matched_unique,
        **counts,
        "gaia_dao_completeness_pct": corr_pct,
        "gaia_dao_completeness_raw_pct": raw_pct,
        "unmatched_dao": unmatched_dao,
        "n_dao_unmatched": int(unmatched_dao["n_dao_unmatched"]),
        "labeled_cone": labeled,
    }


def reconcile_to_pipeline_meta(report: dict[str, Any]) -> dict[str, Any]:
    """Extract flat keys for ``merge_photometry_pipeline_meta``."""
    return {
        "g_lim_est": report.get("g_lim_est"),
        "n_gaia_matched": report.get("n_gaia_matched"),
        "n_gaia_below_limit": report.get("n_gaia_below_limit"),
        "n_gaia_blended": report.get("n_gaia_blended"),
        "n_gaia_missed": report.get("n_gaia_missed"),
        "gaia_dao_completeness_pct": report.get("gaia_dao_completeness_pct"),
        "gaia_dao_completeness_raw_pct": report.get("gaia_dao_completeness_raw_pct"),
        "n_dao_unmatched": report.get("n_dao_unmatched"),
        "g_lim_percentile": report.get("g_lim_stats", {}).get("percentile"),
        "blend_radius_px": report.get("blend_radius_px"),
        "blend_radius_arcsec": report.get("blend_radius_arcsec"),
    }
