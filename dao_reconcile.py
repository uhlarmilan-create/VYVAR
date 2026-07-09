"""Gaia<->DAO field accounting on the frame footprint reference population.

Completeness curve and G_lim_50 / G_lim_90 use the Fleming et al. (1995) error-function
model (see ``CITATIONS.bib`` key ``fleming1995``). Reference stars come from a direct
local-Gaia DB query over the MASTERSTAR WCS bounding box at detect-time depth (no
``field_catalog_cone.csv`` row cap). Blend radius: 1.5 x FWHM [px] (``crowding_index``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import erf, erfinv

from database import get_gaia_db_max_g_mag, query_local_gaia
from gaia_catalog_id import normalize_gaia_source_id_series

BLEND_FWHM_FACTOR = 1.5
DEFAULT_EDGE_MARGIN_FWHM = 2.0
DEFAULT_BIN_WIDTH_MAG = 0.5
COLLINEAR_MIN_POINTS = 3
COLLINEAR_MAX_PERP_PX = 2.0
DEFAULT_MATCH_SEP_ARCSEC = 8.0


class ReferencePopulationMismatch(RuntimeError):
    """Matched DAO catalog_ids or magnitudes inconsistent with the reference query."""


@dataclass
class FlemingFitResult:
    g_lim_50: float
    g_lim_90: float | None
    sigma_mag: float | None
    fit_method: str
    fit_params: dict[str, float | None]
    curve_bins: list[dict[str, Any]]


def _normalize_ids(df: pd.DataFrame, col: str = "catalog_id") -> pd.Series:
    if col not in df.columns:
        return pd.Series([""] * len(df), dtype=str)
    return normalize_gaia_source_id_series(df[col]).fillna("").astype(str).str.strip()


def blend_radius_px(fwhm_px: float, *, factor: float = BLEND_FWHM_FACTOR) -> float:
    return float(factor) * float(fwhm_px)


def blend_radius_arcsec(fwhm_px: float, plate_scale_arcsec: float, *, factor: float = BLEND_FWHM_FACTOR) -> float:
    return blend_radius_px(fwhm_px, factor=factor) * float(plate_scale_arcsec)


def fleming_completeness(mag: np.ndarray | float, g_lim_50: float, sigma_mag: float) -> np.ndarray:
    """Fleming et al. (1995): C(G) = 0.5 * (1 + erf((G_50 - G) / (sqrt(2) sigma)))."""
    m = np.asarray(mag, dtype=np.float64)
    sig = max(float(sigma_mag), 1e-6)
    return 0.5 * (1.0 + erf((float(g_lim_50) - m) / (math.sqrt(2.0) * sig)))


def frame_sky_bbox_deg(wcs: Any, naxis1: int, naxis2: int) -> tuple[float, float, float, float]:
    """RA/Dec extrema from the four chip corners (ICRS deg)."""
    xs = [0.0, float(naxis1 - 1), float(naxis1 - 1), 0.0]
    ys = [0.0, 0.0, float(naxis2 - 1), float(naxis2 - 1)]
    ra, dec = wcs.all_pix2world(xs, ys, 0)
    ra = np.asarray(ra, dtype=np.float64)
    dec = np.asarray(dec, dtype=np.float64)
    return float(np.nanmin(ra)), float(np.nanmax(ra)), float(np.nanmin(dec)), float(np.nanmax(dec))


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


def gaia_db_rows_to_reference_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["catalog_id", "ra_deg", "dec_deg", "mag"])
    df = pd.DataFrame(rows)
    rename = {"source_id": "catalog_id", "ra": "ra_deg", "dec": "dec_deg", "g_mag": "mag"}
    for old, new in rename.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})
    df["catalog_id"] = _normalize_ids(df)
    df["mag"] = pd.to_numeric(df.get("mag"), errors="coerce")
    df["ra_deg"] = pd.to_numeric(df.get("ra_deg"), errors="coerce")
    df["dec_deg"] = pd.to_numeric(df.get("dec_deg"), errors="coerce")
    return df


def query_reference_population(
    gaia_db_path: str | Any,
    wcs: Any,
    naxis1: int,
    naxis2: int,
    *,
    mag_limit: float | None = None,
) -> pd.DataFrame:
    """Direct Gaia DB query over the frame WCS bbox at full detect-time depth (no row cap)."""
    ra_min, ra_max, dec_min, dec_max = frame_sky_bbox_deg(wcs, naxis1, naxis2)
    ml = float(mag_limit) if mag_limit is not None and math.isfinite(float(mag_limit)) else None
    if ml is None:
        _gmax = get_gaia_db_max_g_mag(gaia_db_path)
        ml = float(_gmax) if _gmax > 0 else None
    rows = query_local_gaia(
        gaia_db_path,
        ra_min=ra_min,
        ra_max=ra_max,
        dec_min=dec_min,
        dec_max=dec_max,
        mag_limit=ml,
        max_rows=None,
    )
    return gaia_db_rows_to_reference_df(rows)


def apply_footprint_filter(
    reference_df: pd.DataFrame,
    wcs: Any,
    naxis1: int,
    naxis2: int,
    *,
    fwhm_px: float,
    edge_margin_fwhm: float = DEFAULT_EDGE_MARGIN_FWHM,
) -> tuple[pd.DataFrame, int]:
    """Tag in-frame (chip interior minus edge margin) vs off-frame."""
    if reference_df.empty:
        return reference_df.copy(), 0
    ref = reference_df.copy()
    xp, yp = project_ra_dec_to_pixel(wcs, ref["ra_deg"], ref["dec_deg"])
    ref["_x_pix"] = xp
    ref["_y_pix"] = yp
    margin = float(edge_margin_fwhm) * float(fwhm_px)
    on_chip = (
        np.isfinite(xp)
        & np.isfinite(yp)
        & (xp >= 0.0)
        & (xp < float(naxis1))
        & (yp >= 0.0)
        & (yp < float(naxis2))
    )
    in_frame = (
        on_chip
        & (xp >= margin)
        & (xp < float(naxis1) - margin)
        & (yp >= margin)
        & (yp < float(naxis2) - margin)
    )
    ref["_in_frame"] = in_frame
    n_off = int((~in_frame).sum())
    return ref, n_off


def _matched_catalog_id_set(detections_df: pd.DataFrame) -> set[str]:
    cid = _normalize_ids(detections_df)
    return set(cid[cid != ""].unique())


def _build_matched_xy(detections_df: pd.DataFrame) -> np.ndarray:
    cid = _normalize_ids(detections_df)
    m = cid != ""
    if not m.any():
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


def check_reference_population_consistency(
    detections_df: pd.DataFrame,
    reference_df: pd.DataFrame,
) -> dict[str, Any]:
    """Assert matched stars are covered by the reference query (exact catalog_id + G depth)."""
    matched_mask = _normalize_ids(detections_df) != ""
    matched_df = detections_df.loc[matched_mask]
    matched_ids = _matched_catalog_id_set(detections_df)
    ref_ids = set(reference_df["catalog_id"].astype(str).str.strip()) if len(reference_df) else set()
    missing_ids = sorted(matched_ids - ref_ids)
    mags = pd.to_numeric(matched_df.get("phot_g_mean_mag", matched_df.get("mag")), errors="coerce").dropna()
    ref_mags = pd.to_numeric(reference_df.get("mag"), errors="coerce").dropna()
    max_matched = float(mags.max()) if not mags.empty else float("nan")
    max_ref = float(ref_mags.max()) if not ref_mags.empty else float("nan")
    ok = not missing_ids and (not math.isfinite(max_matched) or not math.isfinite(max_ref) or max_matched <= max_ref + 1e-3)
    out = {
        "ok": bool(ok),
        "n_matched_ids": int(len(matched_ids)),
        "n_missing_from_reference": int(len(missing_ids)),
        "missing_ids_sample": missing_ids[:12],
        "max_matched_g": round(max_matched, 4) if math.isfinite(max_matched) else None,
        "max_reference_g": round(max_ref, 4) if math.isfinite(max_ref) else None,
    }
    if not ok:
        raise ReferencePopulationMismatch(
            f"Reference population mismatch: missing={len(missing_ids)} "
            f"max_matched_G={max_matched} max_ref_G={max_ref}"
        )
    return out


def bin_completeness_curve(
    reference_in_frame: pd.DataFrame,
    matched_ids: set[str],
    *,
    bin_width: float = DEFAULT_BIN_WIDTH_MAG,
) -> list[dict[str, Any]]:
    """Per-bin n_ref, n_matched, completeness fraction (0.5 mag bins by default)."""
    if reference_in_frame.empty:
        return []
    ref = reference_in_frame.copy()
    ref["_matched"] = ref["catalog_id"].astype(str).str.strip().isin(matched_ids)
    mags = pd.to_numeric(ref["mag"], errors="coerce")
    ref = ref.loc[mags.notna()].copy()
    if ref.empty:
        return []
    bw = float(bin_width)
    lo = float(math.floor(float(mags.min()) / bw) * bw)
    hi = float(math.ceil(float(mags.max()) / bw) * bw)
    bins: list[dict[str, Any]] = []
    edge = lo
    while edge < hi + 1e-9:
        center = edge + 0.5 * bw
        mask = (mags >= edge) & (mags < edge + bw)
        n_ref = int(mask.sum())
        if n_ref > 0:
            n_mat = int(ref.loc[mask, "_matched"].sum())
            frac = float(n_mat) / float(n_ref)
            bins.append(
                {
                    "bin_lo": round(edge, 3),
                    "bin_hi": round(edge + bw, 3),
                    "bin_center": round(center, 3),
                    "n_ref": n_ref,
                    "n_matched": n_mat,
                    "completeness_frac": round(frac, 4),
                }
            )
        edge += bw
    return bins


def _crossing_mag(bins: list[dict[str, Any]], level: float) -> float | None:
    if not bins:
        return None
    pts = sorted(bins, key=lambda b: b["bin_center"])
    for i in range(len(pts) - 1):
        f0 = float(pts[i]["completeness_frac"])
        f1 = float(pts[i + 1]["completeness_frac"])
        m0 = float(pts[i]["bin_center"])
        m1 = float(pts[i + 1]["bin_center"])
        if f0 <= level <= f1 or f1 <= level <= f0:
            if abs(f1 - f0) < 1e-9:
                return m0
            t = (level - f0) / (f1 - f0)
            return float(m0 + t * (m1 - m0))
    return None


def fit_fleming_completeness(
    curve_bins: list[dict[str, Any]],
) -> FlemingFitResult:
    """Fit Fleming erf completeness; fallback to linear interpolation crossings."""
    if not curve_bins:
        return FlemingFitResult(
            g_lim_50=float("nan"),
            g_lim_90=None,
            sigma_mag=None,
            fit_method="none",
            fit_params={},
            curve_bins=[],
        )

    mags = np.array([b["bin_center"] for b in curve_bins], dtype=np.float64)
    fracs = np.array([b["completeness_frac"] for b in curve_bins], dtype=np.float64)
    weights = np.sqrt(np.maximum([b["n_ref"] for b in curve_bins], 1)).astype(np.float64)

    g50_i = _crossing_mag(curve_bins, 0.5)
    g90_i = _crossing_mag(curve_bins, 0.9)

    fit_method = "interpolation"
    g50 = g50_i
    g90 = g90_i
    sigma: float | None = None
    params: dict[str, float | None] = {"g_lim_50_interp": g50_i, "g_lim_90_interp": g90_i}

    if len(curve_bins) >= 4 and np.any(fracs < 0.95) and np.any(fracs > 0.05):
        try:
            p0 = (float(g50_i or np.median(mags)), 0.4)
            bounds = ([float(mags.min()) - 2.0, 0.05], [float(mags.max()) + 2.0, 3.0])

            def _model(m: np.ndarray, m50: float, sig: float) -> np.ndarray:
                return fleming_completeness(m, m50, sig)

            popt, _ = curve_fit(
                _model,
                mags,
                fracs,
                p0=p0,
                bounds=bounds,
                sigma=1.0 / weights,
                absolute_sigma=False,
                maxfev=8000,
            )
            m50_f, sig_f = float(popt[0]), float(popt[1])
            g50 = m50_f
            sigma = sig_f
            g90 = m50_f - math.sqrt(2.0) * sig_f * float(erfinv(0.8))
            fit_method = "fleming1995_erf"
            params = {"g_lim_50": m50_f, "sigma_mag": sig_f, "g_lim_90": g90}
        except Exception:  # noqa: BLE001
            pass

    if g50 is None or not math.isfinite(float(g50)):
        g50 = float(np.median(mags))
    return FlemingFitResult(
        g_lim_50=float(g50),
        g_lim_90=float(g90) if g90 is not None and math.isfinite(float(g90)) else g90_i,
        sigma_mag=sigma,
        fit_method=fit_method,
        fit_params={k: (round(float(v), 4) if v is not None and math.isfinite(float(v)) else None) for k, v in params.items()},
        curve_bins=curve_bins,
    )


def decompose_reference_population(
    reference_df: pd.DataFrame,
    detections_df: pd.DataFrame,
    *,
    g_lim_50: float,
    fwhm_px: float,
    blend_factor: float = BLEND_FWHM_FACTOR,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Bucket each reference star: matched / off_frame / below_limit / blended / genuinely_missed."""
    if reference_df.empty:
        return pd.DataFrame(), {
            "n_gaia_matched": 0,
            "n_gaia_off_frame": 0,
            "n_gaia_below_limit": 0,
            "n_gaia_blended": 0,
            "n_gaia_missed": 0,
            "n_ref_in_frame": 0,
        }

    ref = reference_df.copy()
    matched_ids = _matched_catalog_id_set(detections_df)
    ref["_mag"] = pd.to_numeric(ref.get("mag"), errors="coerce")
    ref["_bucket"] = "off_frame"
    in_frame = ref.get("_in_frame", pd.Series(False, index=ref.index)).astype(bool)
    ref.loc[in_frame, "_bucket"] = np.where(
        ref.loc[in_frame, "catalog_id"].astype(str).str.strip().isin(matched_ids),
        "matched",
        "undetected",
    )

    matched_xy = _build_matched_xy(detections_df)
    blend_r = blend_radius_px(fwhm_px, factor=blend_factor)
    glim = float(g_lim_50)

    und = in_frame & (ref["_bucket"] == "undetected")
    below = und & ref["_mag"].gt(glim)
    ref.loc[below, "_bucket"] = "below_limit"

    rest = und & ~below
    blended_mask = np.zeros(len(ref), dtype=bool)
    if np.any(rest) and matched_xy.size > 0:
        from scipy.spatial import cKDTree

        tree = cKDTree(matched_xy)
        idxs = np.where(rest.to_numpy())[0]
        pts = np.column_stack(
            [
                ref.loc[rest, "_x_pix"].to_numpy(dtype=np.float64),
                ref.loc[rest, "_y_pix"].to_numpy(dtype=np.float64),
            ]
        )
        ok_pt = np.isfinite(pts).all(axis=1)
        if np.any(ok_pt):
            dists, _ = tree.query(pts[ok_pt], k=1)
            blended_local = dists <= blend_r
            blended_mask[idxs[ok_pt]] = blended_local
    ref.loc[blended_mask, "_bucket"] = "blended"
    missed = und & ~below & ~blended_mask
    ref.loc[missed, "_bucket"] = "genuinely_missed"

    in_frame_mask = in_frame.to_numpy()
    detectable = in_frame_mask & ref["_mag"].le(glim).to_numpy()
    n_matched_detectable = int((detectable & (ref["_bucket"] == "matched").to_numpy()).sum())
    n_missed = int((ref["_bucket"] == "genuinely_missed").sum())

    counts = {
        "n_gaia_matched": int((ref["_bucket"] == "matched").sum()),
        "n_gaia_matched_detectable": n_matched_detectable,
        "n_gaia_off_frame": int((ref["_bucket"] == "off_frame").sum()),
        "n_gaia_below_limit": int((ref["_bucket"] == "below_limit").sum()),
        "n_gaia_blended": int((ref["_bucket"] == "blended").sum()),
        "n_gaia_missed": n_missed,
        "n_ref_in_frame": int(in_frame.sum()),
    }
    return ref, counts


def completeness_50_pct(n_matched_detectable: int, n_missed: int) -> float | None:
    denom = int(n_matched_detectable) + int(n_missed)
    if denom <= 0:
        return None
    return round(100.0 * float(n_matched_detectable) / float(denom), 2)


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


def _positional_match_to_reference(
    unmatched: pd.DataFrame,
    ref_in_frame: pd.DataFrame,
    *,
    match_sep_arcsec: float,
    plate_scale_arcsec: float | None,
    fwhm_px: float,
) -> np.ndarray:
    if unmatched.empty or ref_in_frame.empty:
        return np.zeros(len(unmatched), dtype=bool)
    if plate_scale_arcsec is not None and plate_scale_arcsec > 0:
        r_px = float(match_sep_arcsec) / float(plate_scale_arcsec)
    else:
        r_px = float(match_sep_arcsec) / 9.78
    from scipy.spatial import cKDTree

    ref_xy = np.column_stack(
        [
            pd.to_numeric(ref_in_frame["_x_pix"], errors="coerce").to_numpy(dtype=np.float64),
            pd.to_numeric(ref_in_frame["_y_pix"], errors="coerce").to_numpy(dtype=np.float64),
        ]
    )
    ok_ref = np.isfinite(ref_xy).all(axis=1)
    if not np.any(ok_ref):
        return np.zeros(len(unmatched), dtype=bool)
    tree = cKDTree(ref_xy[ok_ref])
    ux = pd.to_numeric(unmatched["x"], errors="coerce").to_numpy(dtype=np.float64)
    uy = pd.to_numeric(unmatched["y"], errors="coerce").to_numpy(dtype=np.float64)
    pts = np.column_stack([ux, uy])
    ok_u = np.isfinite(pts).all(axis=1)
    hit = np.zeros(len(unmatched), dtype=bool)
    if np.any(ok_u):
        dists, _ = tree.query(pts[ok_u], k=1)
        hit[np.where(ok_u)[0]] = dists <= r_px
    return hit


def classify_unmatched_dao(
    detections_df: pd.DataFrame,
    *,
    ref_in_frame: pd.DataFrame | None = None,
    matched_ids: set[str] | None = None,
    match_sep_arcsec: float = DEFAULT_MATCH_SEP_ARCSEC,
    plate_scale_arcsec: float | None = None,
    fwhm_px: float = 3.5,
    collinear_min_points: int = COLLINEAR_MIN_POINTS,
    collinear_max_perp_px: float = COLLINEAR_MAX_PERP_PX,
) -> dict[str, Any]:
    cid = _normalize_ids(detections_df)
    unmatched = detections_df.loc[cid == ""].copy()
    n_total = int(len(unmatched))
    if n_total == 0:
        return {
            "n_dao_unmatched": 0,
            "n_now_matched_to_faint": 0,
            "n_artifact_candidates": 0,
            "n_unexplained": 0,
            "collinearity": find_largest_collinear_group(np.empty((0, 2))),
            "flux": {},
            "peak_dao": {},
            "classification": "none",
        }

    peak = pd.to_numeric(unmatched.get("peak_dao", unmatched.get("peak_max_adu")), errors="coerce")
    flux = pd.to_numeric(unmatched.get("flux", unmatched.get("dao_flux")), errors="coerce")
    xy = np.column_stack(
        [
            pd.to_numeric(unmatched.get("x"), errors="coerce").to_numpy(dtype=np.float64),
            pd.to_numeric(unmatched.get("y"), errors="coerce").to_numpy(dtype=np.float64),
        ]
    )
    collin = find_largest_collinear_group(xy, min_points=collinear_min_points, max_perp_px=collinear_max_perp_px)

    faint_hit = np.zeros(n_total, dtype=bool)
    if ref_in_frame is not None and not ref_in_frame.empty:
        faint_hit = _positional_match_to_reference(
            unmatched,
            ref_in_frame,
            match_sep_arcsec=match_sep_arcsec,
            plate_scale_arcsec=plate_scale_arcsec,
            fwhm_px=fwhm_px,
        )

    artifact_mask = faint_hit.copy()
    if collin["consistent_with_line"] and collin["inlier_indices"]:
        artifact_mask[collin["inlier_indices"]] = True
    if peak.notna().sum() >= 5:
        med_peak = float(peak.median())
        p90_peak = float(peak.quantile(0.9))
        hot_thr = max(med_peak * 3.0, p90_peak * 1.5)
        artifact_mask |= peak.to_numpy(dtype=np.float64) >= hot_thr

    n_faint = int(faint_hit.sum())
    n_artifact = int(artifact_mask.sum())
    n_unexplained = int(n_total - n_artifact)

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
        "n_now_matched_to_faint": n_faint,
        "n_artifact_candidates": n_artifact,
        "n_unexplained": n_unexplained,
        "classification": "artifact_dominant" if n_artifact >= max(1, n_total // 2) else "mixed",
        "collinearity": collin,
        "flux": _dist_stats(flux),
        "peak_dao": _dist_stats(peak),
    }


def compute_gaia_dao_reconcile(
    detections_df: pd.DataFrame,
    *,
    gaia_db_path: str | Any,
    wcs: Any,
    naxis1: int,
    naxis2: int,
    fwhm_px: float,
    plate_scale_arcsec: float | None = None,
    mag_limit: float | None = None,
    edge_margin_fwhm: float = DEFAULT_EDGE_MARGIN_FWHM,
    bin_width_mag: float = DEFAULT_BIN_WIDTH_MAG,
    blend_factor: float = BLEND_FWHM_FACTOR,
    match_sep_arcsec: float = DEFAULT_MATCH_SEP_ARCSEC,
    cone_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Full footprint-based Gaia<->DAO reconciliation (R-2 methodology)."""
    reference_df = query_reference_population(
        gaia_db_path,
        wcs,
        naxis1,
        naxis2,
        mag_limit=mag_limit,
    )
    pop_check = check_reference_population_consistency(detections_df, reference_df)
    reference_df, n_off = apply_footprint_filter(
        reference_df,
        wcs,
        naxis1,
        naxis2,
        fwhm_px=fwhm_px,
        edge_margin_fwhm=edge_margin_fwhm,
    )
    matched_ids = _matched_catalog_id_set(detections_df)
    ref_in_frame = reference_df.loc[reference_df.get("_in_frame", False).astype(bool)].copy()

    curve_bins = bin_completeness_curve(ref_in_frame, matched_ids, bin_width=bin_width_mag)
    fleming = fit_fleming_completeness(curve_bins)
    g_lim_50 = float(fleming.g_lim_50)
    g_lim_90 = fleming.g_lim_90

    labeled, counts = decompose_reference_population(
        reference_df,
        detections_df,
        g_lim_50=g_lim_50,
        fwhm_px=float(fwhm_px),
        blend_factor=blend_factor,
    )

    corr_pct = completeness_50_pct(counts["n_gaia_matched_detectable"], counts["n_gaia_missed"])
    cid = _normalize_ids(detections_df)
    n_matched_unique = int(cid[cid != ""].nunique())
    catalog_rows = int(len(cone_df)) if cone_df is not None else int(len(reference_df))
    raw_pct = raw_completeness_pct(n_matched_unique, catalog_rows) if cone_df is not None else None

    if plate_scale_arcsec is None and wcs is not None:
        try:
            from astropy.wcs.utils import proj_plane_pixel_scales

            scales = proj_plane_pixel_scales(wcs) * 3600.0
            plate_scale_arcsec = float(np.mean(scales))
        except Exception:  # noqa: BLE001
            plate_scale_arcsec = None

    unmatched_dao = classify_unmatched_dao(
        detections_df,
        ref_in_frame=ref_in_frame,
        matched_ids=matched_ids,
        match_sep_arcsec=match_sep_arcsec,
        plate_scale_arcsec=plate_scale_arcsec,
        fwhm_px=float(fwhm_px),
    )

    blend_r_px = blend_radius_px(fwhm_px, factor=blend_factor)
    blend_r_arcsec = (
        blend_radius_arcsec(fwhm_px, plate_scale_arcsec, factor=blend_factor)
        if plate_scale_arcsec is not None
        else None
    )

    return {
        "methodology": "footprint_reference_fleming1995",
        "population_check": pop_check,
        "g_lim_50": round(g_lim_50, 4),
        "g_lim_90": round(float(g_lim_90), 4) if g_lim_90 is not None and math.isfinite(float(g_lim_90)) else None,
        "g_lim_est": round(g_lim_50, 4),
        "fit_method": fleming.fit_method,
        "fleming_fit_params": fleming.fit_params,
        "completeness_curve": fleming.curve_bins,
        "fwhm_px": round(float(fwhm_px), 4),
        "edge_margin_px": round(float(edge_margin_fwhm) * float(fwhm_px), 4),
        "blend_factor_fwhm": float(blend_factor),
        "blend_radius_px": round(blend_r_px, 4),
        "blend_radius_arcsec": round(blend_r_arcsec, 4) if blend_r_arcsec is not None else None,
        "plate_scale_arcsec": round(float(plate_scale_arcsec), 6) if plate_scale_arcsec is not None else None,
        "n_ref_total": int(len(reference_df)),
        "n_ref_in_frame": int(counts["n_ref_in_frame"]),
        "n_gaia_off_frame": int(counts["n_gaia_off_frame"]),
        "catalog_rows": catalog_rows,
        "n_gaia_matched_unique": n_matched_unique,
        **counts,
        "gaia_dao_completeness_pct": corr_pct,
        "gaia_dao_completeness_raw_pct": raw_pct,
        "unmatched_dao": unmatched_dao,
        "n_dao_unmatched": int(unmatched_dao["n_dao_unmatched"]),
        "labeled_reference": labeled,
    }


def reconcile_to_pipeline_meta(report: dict[str, Any]) -> dict[str, Any]:
    """Extract flat keys for ``merge_photometry_pipeline_meta``."""
    curve = report.get("completeness_curve") or []
    compact_curve = [
        {
            "bin_center": c.get("bin_center"),
            "n_ref": c.get("n_ref"),
            "n_matched": c.get("n_matched"),
            "completeness_frac": c.get("completeness_frac"),
        }
        for c in curve[:80]
    ]
    return {
        "g_lim_50": report.get("g_lim_50"),
        "g_lim_90": report.get("g_lim_90"),
        "g_lim_est": report.get("g_lim_50"),
        "fit_method": report.get("fit_method"),
        "completeness_curve": compact_curve,
        "n_ref_in_frame": report.get("n_ref_in_frame"),
        "n_gaia_matched": report.get("n_gaia_matched"),
        "n_gaia_off_frame": report.get("n_gaia_off_frame"),
        "n_gaia_below_limit": report.get("n_gaia_below_limit"),
        "n_gaia_blended": report.get("n_gaia_blended"),
        "n_gaia_missed": report.get("n_gaia_missed"),
        "gaia_dao_completeness_pct": report.get("gaia_dao_completeness_pct"),
        "gaia_dao_completeness_raw_pct": report.get("gaia_dao_completeness_raw_pct"),
        "n_dao_unmatched": report.get("n_dao_unmatched"),
        "n_dao_matched_to_faint": (report.get("unmatched_dao") or {}).get("n_now_matched_to_faint"),
        "blend_radius_px": report.get("blend_radius_px"),
        "blend_radius_arcsec": report.get("blend_radius_arcsec"),
        "dao_reconcile_methodology": report.get("methodology"),
    }
