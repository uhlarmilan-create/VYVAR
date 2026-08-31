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

from database import get_gaia_db_max_g_mag, query_local_gaia, query_local_gaia_by_source_ids
from gaia_catalog_id import normalize_gaia_source_id_series

BLEND_FWHM_FACTOR = 1.5
DEFAULT_EDGE_MARGIN_FWHM = 2.0
DEFAULT_BIN_WIDTH_MAG = 0.5
COLLINEAR_MIN_POINTS = 3
COLLINEAR_MAX_PERP_PX = 2.0
DEFAULT_MATCH_SEP_ARCSEC = 8.0
CENSOR_MARGIN_MAG = 0.1
MASTERSTAR_FAINTEST_MAG_FLOOR = 18.0


class ReferencePopulationMismatch(RuntimeError):
    """Matched DAO catalog_ids or magnitudes inconsistent with the reference query."""


@dataclass
class FlemingFitResult:
    g_lim_50: float | None
    g_lim_90: float | None
    sigma_mag: float | None
    fit_method: str
    fit_params: dict[str, float | None]
    curve_bins: list[dict[str, Any]]
    no_crossing_50: bool = False
    no_crossing_90: bool = False


@dataclass
class CensoredLimit:
    """G limit after right-censoring against reference-population depth."""

    value_g: float | None
    raw_fit_g: float | None
    censored: bool
    reference_depth_g: float | None
    display: str


def reference_population_depth_g(
    reference_df: pd.DataFrame,
    *,
    mag_limit: float | None = None,
    gaia_db_path: str | Any | None = None,
) -> tuple[float | None, float | None]:
    """Return (max G in reference, query mag_limit actually used)."""
    mags = pd.to_numeric(reference_df.get("mag"), errors="coerce").dropna()
    max_ref = float(mags.max()) if not mags.empty else None
    ml: float | None
    if mag_limit is not None and math.isfinite(float(mag_limit)):
        ml = float(mag_limit)
    elif gaia_db_path is not None:
        _gmax = get_gaia_db_max_g_mag(gaia_db_path)
        ml = float(_gmax) if _gmax > 0 else None
    else:
        ml = None
    depth = max_ref
    if ml is not None and (depth is None or ml < depth):
        depth = ml
    return (
        round(depth, 4) if depth is not None and math.isfinite(depth) else None,
        round(ml, 4) if ml is not None and math.isfinite(ml) else None,
    )


def apply_limit_censoring(
    fit_g: float | None,
    reference_depth_g: float | None,
    *,
    label: str,
    no_crossing: bool = False,
) -> CensoredLimit:
    """Clamp extrapolated G_lim when fit exceeds reference depth (right-censored)."""
    raw = float(fit_g) if fit_g is not None and math.isfinite(float(fit_g)) else None
    depth = float(reference_depth_g) if reference_depth_g is not None and math.isfinite(float(reference_depth_g)) else None
    if no_crossing and depth is not None:
        return CensoredLimit(
            value_g=depth,
            raw_fit_g=None,
            censored=True,
            reference_depth_g=depth,
            display=f">= {depth:.1f} (no crossing)",
        )
    if raw is None:
        return CensoredLimit(
            value_g=depth,
            raw_fit_g=None,
            censored=False,
            reference_depth_g=depth,
            display=f"{label}: ?",
        )
    if depth is None:
        return CensoredLimit(
            value_g=raw,
            raw_fit_g=raw,
            censored=False,
            reference_depth_g=None,
            display=f"{label}={raw:.2f}",
        )
    censored = raw > depth - CENSOR_MARGIN_MAG
    if censored:
        return CensoredLimit(
            value_g=depth,
            raw_fit_g=raw,
            censored=True,
            reference_depth_g=depth,
            display=f">= {depth:.1f} (censored)",
        )
    return CensoredLimit(
        value_g=raw,
        raw_fit_g=raw,
        censored=False,
        reference_depth_g=depth,
        display=f"{label}={raw:.2f}",
    )


def split_missed_by_g90(
    labeled: pd.DataFrame,
    *,
    g_lim_90: float | None,
    g_lim_50: float | None,
    g_lim_90_censored: bool,
    reference_depth_g: float | None,
) -> dict[str, int | bool | None]:
    """Split genuinely-missed into fade-zone vs below-G90 (2-pass decision metric)."""
    if labeled.empty or "_bucket" not in labeled.columns:
        return {
            "n_missed_below_g90": 0,
            "n_missed_fadezone": 0,
            "missed_below_g90_uses_censored_depth": False,
        }
    missed = labeled.loc[labeled["_bucket"] == "genuinely_missed"].copy()
    if missed.empty:
        return {
            "n_missed_below_g90": 0,
            "n_missed_fadezone": 0,
            "missed_below_g90_uses_censored_depth": False,
        }
    mags = pd.to_numeric(missed.get("_mag", missed.get("mag")), errors="coerce")
    g90_thr = g_lim_90
    uses_censored = False
    if g_lim_90_censored or g90_thr is None:
        g90_thr = reference_depth_g
        uses_censored = g90_thr is not None
    g50_thr = g_lim_50 if g_lim_50 is not None else reference_depth_g
    below = 0
    fade = 0
    for g in mags.dropna():
        gv = float(g)
        if g90_thr is not None and gv < float(g90_thr):
            below += 1
        elif g50_thr is not None and float(g90_thr or g50_thr) <= gv <= float(g50_thr):
            fade += 1
        elif g90_thr is not None and g50_thr is not None and gv > float(g50_thr):
            pass
        elif g90_thr is None and g50_thr is not None and gv <= float(g50_thr):
            fade += 1
    return {
        "n_missed_below_g90": int(below),
        "n_missed_fadezone": int(fade),
        "missed_below_g90_uses_censored_depth": bool(uses_censored),
    }


def resolve_effective_match_depth(
    pipeline_meta: dict[str, Any] | None,
    *,
    is_masterstar: bool = True,
) -> dict[str, Any]:
    """Resolve faintest_mag_limit that governed detect-time catalog matching."""
    meta = pipeline_meta or {}
    config_val = meta.get("faintest_mag_limit")
    provenance = meta.get("provenance") if isinstance(meta.get("provenance"), dict) else {}
    snap = provenance.get("config_snapshot") if isinstance(provenance.get("config_snapshot"), dict) else {}
    setup_val = snap.get("faintest_mag_limit")
    raw = config_val if config_val is not None else setup_val
    if is_masterstar:
        if raw is None:
            eff = MASTERSTAR_FAINTEST_MAG_FLOOR
            source = (
                "MASTERSTAR _ms_faintest_mag_eff=18.0 "
                "(faintest_mag_limit unset in setup + pipeline_meta)"
            )
        else:
            eff = max(float(raw), MASTERSTAR_FAINTEST_MAG_FLOOR)
            source = f"max(faintest_mag_limit={float(raw)}, {MASTERSTAR_FAINTEST_MAG_FLOOR}) MASTERSTAR floor"
    elif raw is not None and math.isfinite(float(raw)):
        eff = float(raw)
        source = f"faintest_mag_limit={eff} (detect path)"
    else:
        eff = None
        source = "no faintest_mag_limit (detect path)"
    return {
        "match_depth": round(float(eff), 4) if eff is not None and math.isfinite(float(eff)) else None,
        "match_depth_source": source,
        "faintest_mag_limit_config": float(raw) if raw is not None and math.isfinite(float(raw)) else None,
    }


def _normalize_id_series(df: pd.DataFrame, col: str = "catalog_id") -> pd.Series:
    """Per-row Gaia canonical ids from a DataFrame column. Not a unique-list helper."""
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
    df["catalog_id"] = _normalize_id_series(df)
    df["mag"] = pd.to_numeric(df.get("mag"), errors="coerce")
    df["ra_deg"] = pd.to_numeric(df.get("ra_deg"), errors="coerce")
    df["dec_deg"] = pd.to_numeric(df.get("dec_deg"), errors="coerce")
    return df


def augment_reference_with_matched_ids(
    reference_df: pd.DataFrame,
    detections_df: pd.DataFrame,
    gaia_db_path: str | Any,
) -> pd.DataFrame:
    """Add bbox-missed but DAO-matched stars from per-id Gaia lookup + MASTERSTAR astrometry."""
    matched_ids = _matched_catalog_id_set(detections_df)
    ref_ids = set(_normalize_id_series(reference_df)) - {""}
    missing = sorted(matched_ids - ref_ids)
    if not missing:
        return reference_df

    got = query_local_gaia_by_source_ids(gaia_db_path, missing)
    det = detections_df.copy()
    det["_cid"] = _normalize_id_series(det)
    rows: list[dict[str, Any]] = []
    for cid in missing:
        sub = det.loc[det["_cid"] == cid]
        if sub.empty:
            continue
        r0 = sub.iloc[0]
        ra = pd.to_numeric(r0.get("ra_deg"), errors="coerce")
        de = pd.to_numeric(r0.get("dec_deg"), errors="coerce")
        g_row = got.get(cid, {})
        g_db = g_row.get("g_mag") if isinstance(g_row, dict) else None
        mag = pd.to_numeric(g_db if g_db is not None else r0.get("phot_g_mean_mag", r0.get("mag")), errors="coerce")
        rows.append(
            {
                "catalog_id": cid,
                "ra_deg": float(ra) if np.isfinite(ra) else np.nan,
                "dec_deg": float(de) if np.isfinite(de) else np.nan,
                "mag": float(mag) if np.isfinite(mag) else np.nan,
                "_augmented_match": True,
            }
        )
    if not rows:
        return reference_df
    extra = pd.DataFrame(rows)
    return pd.concat([reference_df, extra], ignore_index=True)


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
    cid = _normalize_id_series(detections_df)
    return set(cid[cid != ""].unique())


def _build_matched_xy(detections_df: pd.DataFrame) -> np.ndarray:
    cid = _normalize_id_series(detections_df)
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
    matched_mask = _normalize_id_series(detections_df) != ""
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


def _deepest_bin(curve_bins: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not curve_bins:
        return None
    return max(curve_bins, key=lambda b: float(b["bin_center"]))


def _stays_above_level_to_edge(curve_bins: list[dict[str, Any]], level: float) -> bool:
    """True when the faintest bin completeness stays at or above ``level`` (no crossing)."""
    deepest = _deepest_bin(curve_bins)
    if deepest is None:
        return False
    return float(deepest["completeness_frac"]) >= float(level) - 1e-6


def fit_fleming_completeness(
    curve_bins: list[dict[str, Any]],
) -> FlemingFitResult:
    """Fit Fleming erf completeness; fallback to linear interpolation crossings."""
    if not curve_bins:
        return FlemingFitResult(
            g_lim_50=None,
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
    no_cross_50 = g50_i is None and _stays_above_level_to_edge(curve_bins, 0.5)
    no_cross_90 = g90_i is None and _stays_above_level_to_edge(curve_bins, 0.9)

    fit_method = "interpolation"
    g50: float | None = g50_i
    g90: float | None = g90_i
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

    fleming_ok = fit_method == "fleming1995_erf"

    if len(curve_bins) < 3:
        fit_method = "degenerate"
        g50 = float(np.median(mags))
    elif not fleming_ok and no_cross_50:
        fit_method = "no_crossing"
        g50 = None
        if no_cross_90:
            g90 = None
    elif g50 is None or not math.isfinite(float(g50)):
        if no_cross_50:
            fit_method = "no_crossing"
            g50 = None
            if no_cross_90:
                g90 = None

    if g90 is not None and not math.isfinite(float(g90)):
        g90 = g90_i if g90_i is not None and math.isfinite(float(g90_i)) else None
    if no_cross_90 and g90 is None and fit_method != "degenerate":
        pass

    return FlemingFitResult(
        g_lim_50=g50,
        g_lim_90=g90,
        sigma_mag=sigma,
        fit_method=fit_method,
        fit_params={k: (round(float(v), 4) if v is not None and math.isfinite(float(v)) else None) for k, v in params.items()},
        curve_bins=curve_bins,
        no_crossing_50=bool(no_cross_50 and g50 is None and fit_method == "no_crossing"),
        no_crossing_90=bool(no_cross_90 and g90 is None and fit_method in ("no_crossing", "interpolation", "fleming1995_erf")),
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
    cid = _normalize_id_series(detections_df)
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
    collin = find_largest_collinear_group(
        xy[: min(n_total, 400)],
        min_points=collinear_min_points,
        max_perp_px=collinear_max_perp_px,
    )

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
    reference_df = augment_reference_with_matched_ids(reference_df, detections_df, gaia_db_path)
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
    ref_depth, query_mag_limit = reference_population_depth_g(
        reference_df,
        mag_limit=mag_limit,
        gaia_db_path=gaia_db_path,
    )

    lim50 = apply_limit_censoring(
        fleming.g_lim_50,
        ref_depth,
        label="G_lim_50",
        no_crossing=fleming.no_crossing_50,
    )
    lim90 = apply_limit_censoring(
        fleming.g_lim_90,
        ref_depth,
        label="G_lim_90",
        no_crossing=fleming.no_crossing_90,
    )
    g_lim_50 = float(lim50.value_g) if lim50.value_g is not None else None
    g_lim_90 = float(lim90.value_g) if lim90.value_g is not None else fleming.g_lim_90

    labeled, counts = decompose_reference_population(
        reference_df,
        detections_df,
        g_lim_50=float(g_lim_50) if g_lim_50 is not None else float(ref_depth or 0.0),
        fwhm_px=float(fwhm_px),
        blend_factor=blend_factor,
    )

    missed_split = split_missed_by_g90(
        labeled,
        g_lim_90=g_lim_90,
        g_lim_50=g_lim_50,
        g_lim_90_censored=lim90.censored,
        reference_depth_g=ref_depth,
    )

    corr_pct = completeness_50_pct(counts["n_gaia_matched_detectable"], counts["n_gaia_missed"])
    if (lim50.censored or fleming.no_crossing_50) and ref_depth is not None:
        completeness_label = f"measured to G <= {ref_depth:.1f}"
    elif g_lim_50 is not None:
        completeness_label = f"measured to G <= {g_lim_50:.2f}"
    else:
        completeness_label = "completeness_50 unavailable"
    cid = _normalize_id_series(detections_df)
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
        "reference_depth_g": ref_depth,
        "reference_query_mag_limit": query_mag_limit,
        "g_lim_50": round(g_lim_50, 4) if g_lim_50 is not None and math.isfinite(g_lim_50) else None,
        "g_lim_90": round(float(g_lim_90), 4) if g_lim_90 is not None and math.isfinite(float(g_lim_90)) else None,
        "g_lim_50_raw_fit": round(lim50.raw_fit_g, 4) if lim50.raw_fit_g is not None else None,
        "g_lim_90_raw_fit": round(lim90.raw_fit_g, 4) if lim90.raw_fit_g is not None else None,
        "g_lim_50_censored": bool(lim50.censored),
        "g_lim_90_censored": bool(lim90.censored),
        "g_lim_50_display": lim50.display,
        "g_lim_90_display": lim90.display,
        "g_lim_est": round(g_lim_50, 4) if g_lim_50 is not None and math.isfinite(g_lim_50) else None,
        "fit_method": fleming.fit_method,
        "fleming_fit_params": fleming.fit_params,
        "completeness_curve": fleming.curve_bins,
        "completeness_50_label": completeness_label,
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
        "n_missed_below_g90": missed_split["n_missed_below_g90"],
        "n_missed_fadezone": missed_split["n_missed_fadezone"],
        "missed_below_g90_uses_censored_depth": missed_split["missed_below_g90_uses_censored_depth"],
        "unmatched_dao": unmatched_dao,
        "n_dao_unmatched": int(unmatched_dao["n_dao_unmatched"]),
        "labeled_reference": labeled,
    }


DAO_ONLY_CLASS_ARTIFACT_NEGATIVE = "artifact_negative"
DAO_ONLY_CLASS_UNMATCHED_IN_RANGE = "unmatched_in_range"
DAO_ONLY_CLASS_AMBIGUOUS_DEPTH = "ambiguous_depth"
DAO_ONLY_CLASS_BEYOND_CATALOGUE = "beyond_catalogue"
DAO_ONLY_CLASS_INDETERMINATE = "indeterminate"
SIGMA_G_UNMEASURABLE_THRESHOLD_MAG = 1.0

DAO_ONLY_CLASS_LABELS = (
    DAO_ONLY_CLASS_ARTIFACT_NEGATIVE,
    DAO_ONLY_CLASS_UNMATCHED_IN_RANGE,
    DAO_ONLY_CLASS_AMBIGUOUS_DEPTH,
    DAO_ONLY_CLASS_BEYOND_CATALOGUE,
    DAO_ONLY_CLASS_INDETERMINATE,
)

# err_mag ~ 1.0857 / SNR (photometry_core aperture convention)
SIGMA_G_SNR_TERM = 1.0857362


@dataclass
class FluxToGFitResult:
    ok: bool
    zp: float | None
    n_matched: int
    residual_mad: float | None
    residual_rms: float | None
    method: str
    reason: str = ""


@dataclass
class ConfirmableDepthResult:
    confirmable_depth_g: float | None
    winning_input: str | None
    inputs: dict[str, float | None]
    depth_resolvable: bool
    reason: str = ""


def derive_confirmable_depth_g(
    *,
    gaia_db_max_g_mag: float | None,
    effective_match_depth: float | None,
    cone_query_mag_limit: float | None,
) -> ConfirmableDepthResult:
    """Minimum of every limit that constrains what the local catalogue could return."""
    inputs: dict[str, float | None] = {
        "gaia_db_max_g_mag": (
            float(gaia_db_max_g_mag)
            if gaia_db_max_g_mag is not None and math.isfinite(float(gaia_db_max_g_mag))
            else None
        ),
        "effective_match_depth": (
            float(effective_match_depth)
            if effective_match_depth is not None and math.isfinite(float(effective_match_depth))
            else None
        ),
        "cone_query_mag_limit": (
            float(cone_query_mag_limit)
            if cone_query_mag_limit is not None and math.isfinite(float(cone_query_mag_limit))
            else None
        ),
    }
    if inputs["gaia_db_max_g_mag"] is None or inputs["gaia_db_max_g_mag"] <= 0:
        return ConfirmableDepthResult(
            None,
            None,
            inputs,
            False,
            "gaia_db_max_g_mag absent or non-finite; confirmable depth undefined",
        )
    candidates = {k: v for k, v in inputs.items() if v is not None and v > 0}
    if not candidates:
        return ConfirmableDepthResult(None, None, inputs, False, "no finite depth inputs")
    winner = min(candidates, key=candidates.get)
    return ConfirmableDepthResult(
        float(candidates[winner]),
        winner,
        inputs,
        True,
        "",
    )


def row_snr_from_flux(flux: Any, *, noise_adu: float | None) -> float:
    """Detection SNR = flux / noise_adu (consistent with snr50_ok = flux >= 50 * noise)."""
    f = float(flux)
    n = float(noise_adu) if noise_adu is not None else float("nan")
    if not math.isfinite(f) or f <= 0 or not math.isfinite(n) or n <= 0:
        return float("nan")
    return float(f / n)


def sigma_g_row(*, zp_residual_rms: float, snr: float) -> float:
    """Per-row implied-G uncertainty: hypot(ZP fit RMS, 1.0857/SNR)."""
    rms = float(zp_residual_rms) if math.isfinite(float(zp_residual_rms)) else 0.0
    s = float(snr)
    if not math.isfinite(s) or s <= 0:
        return float("nan")
    return float(math.hypot(rms, SIGMA_G_SNR_TERM / s))


def fit_instrumental_flux_to_g(
    matched_df: pd.DataFrame,
    *,
    flux_col: str = "flux",
    g_col: str = "phot_g_mean_mag",
) -> FluxToGFitResult:
    """Robust ZP from Gaia-matched rows: G = zp - 2.5*log10(flux)."""
    if matched_df.empty:
        return FluxToGFitResult(False, None, 0, None, None, "none", "empty matched population")
    flux = pd.to_numeric(matched_df.get(flux_col), errors="coerce")
    gmag = pd.to_numeric(matched_df.get(g_col, matched_df.get("mag")), errors="coerce")
    ok = flux.gt(0) & gmag.notna() & np.isfinite(flux.to_numpy(dtype=float)) & np.isfinite(gmag.to_numpy(dtype=float))
    n_ok = int(ok.sum())
    if n_ok < 5:
        return FluxToGFitResult(
            False,
            None,
            n_ok,
            None,
            None,
            "insufficient",
            f"need >=5 matched rows with positive flux and G, got {n_ok}",
        )
    logf = np.log10(flux.loc[ok].to_numpy(dtype=np.float64))
    g_arr = gmag.loc[ok].to_numpy(dtype=np.float64)
    zp_samples = g_arr + 2.5 * logf
    zp = float(np.median(zp_samples))
    residuals = g_arr - (zp - 2.5 * logf)
    mad = float(np.median(np.abs(residuals - np.median(residuals))))
    rms = float(np.sqrt(np.mean(np.square(residuals))))
    return FluxToGFitResult(True, zp, n_ok, mad, rms, "median_zp")


def implied_g_from_flux(flux: Any, *, zp: float) -> float:
    f = float(flux)
    if not math.isfinite(f) or f <= 0:
        return float("nan")
    return float(zp - 2.5 * math.log10(f))


def _dao_only_row_mask(df: pd.DataFrame) -> pd.Series:
    if "source_type" in df.columns:
        st = df["source_type"].fillna("").astype(str).str.strip().str.upper()
        return st.eq("DAO_ONLY")
    cid = _normalize_id_series(df)
    return cid.eq("")


def _estimate_frame_noise_adu(df: pd.DataFrame, *, frame_noise_adu: float | None) -> float | None:
    if frame_noise_adu is not None and math.isfinite(float(frame_noise_adu)) and float(frame_noise_adu) > 0:
        return float(frame_noise_adu)
    if "noise_floor_adu" in df.columns:
        nf = pd.to_numeric(df["noise_floor_adu"], errors="coerce")
        ok = nf.notna() & (nf > 0)
        if ok.any():
            return float(nf.loc[ok].median())
    flux = pd.to_numeric(df.get("flux"), errors="coerce")
    snr_ok = df.get("snr50_ok")
    if snr_ok is not None:
        mask = snr_ok.fillna(False).astype(bool) & flux.gt(0)
        if mask.any():
            return float((flux.loc[mask] / 50.0).median())
    matched = ~_dao_only_row_mask(df)
    f = flux.loc[matched & flux.gt(0)]
    if len(f) >= 10:
        return float(f.median() / 100.0)
    return None


def _row_noise_adu(row: pd.Series, *, frame_noise_adu: float | None) -> float | None:
    for col in ("noise_floor_adu", "bg_std_adu"):
        if col in row.index:
            v = pd.to_numeric(row.get(col), errors="coerce")
            if pd.notna(v) and float(v) > 0:
                return float(v)
    if frame_noise_adu is not None and math.isfinite(float(frame_noise_adu)) and float(frame_noise_adu) > 0:
        return float(frame_noise_adu)
    return None


def _implied_g_deciles(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {}
    out: dict[str, float] = {}
    for p in (10, 20, 30, 40, 50, 60, 70, 80, 90):
        out[f"p{p}"] = float(np.percentile(arr, p))
    return out


def classify_dao_only_dataframe(
    df: pd.DataFrame,
    *,
    depth: ConfirmableDepthResult,
    flux_fit: FluxToGFitResult,
    fleming_sigma_mag: float | None = None,
    frame_noise_adu: float | None = None,
    gaia_db_identity: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add implied-G columns and ``dao_only_class`` (additive; non-DAO rows blank)."""
    out = df.copy()
    out["implied_g_mag"] = np.nan
    out["implied_g_minus_depth"] = np.nan
    out["sigma_g_row"] = np.nan
    out["dao_only_class"] = ""
    dao_mask = _dao_only_row_mask(out)
    meta: dict[str, Any] = {
        "confirmable_depth_g": depth.confirmable_depth_g,
        "confirmable_depth_inputs": dict(depth.inputs),
        "confirmable_depth_winner": depth.winning_input,
        "depth_resolvable": bool(depth.depth_resolvable),
        "depth_unresolvable_reason": depth.reason if not depth.depth_resolvable else "",
        "flux_fit": {
            "ok": flux_fit.ok,
            "zp": flux_fit.zp,
            "n_matched": flux_fit.n_matched,
            "residual_mad": flux_fit.residual_mad,
            "residual_rms": flux_fit.residual_rms,
            "method": flux_fit.method,
            "reason": flux_fit.reason,
        },
        "fleming_sigma_mag_population": (
            float(fleming_sigma_mag)
            if fleming_sigma_mag is not None and math.isfinite(float(fleming_sigma_mag))
            else None
        ),
        "sigma_g_formula": "sigma_g(row) = hypot(zp_residual_rms, 1.0857 / SNR(row))",
        "gaia_db_identity": dict(gaia_db_identity or {}),
    }
    if gaia_db_identity and gaia_db_identity.get("max_g_mag") is not None:
        meta["max_g_mag"] = float(gaia_db_identity["max_g_mag"])
    elif depth.inputs.get("gaia_db_max_g_mag") is not None:
        meta["max_g_mag"] = float(depth.inputs["gaia_db_max_g_mag"])

    counts = {k: 0 for k in DAO_ONLY_CLASS_LABELS}
    if not dao_mask.any():
        meta["counts"] = counts
        meta["n_dao_only"] = 0
        return out, meta

    flux = pd.to_numeric(out.get("flux"), errors="coerce")
    peak = pd.to_numeric(out.get("peak_max_adu"), errors="coerce")
    sat = out.get("likely_saturated")
    edge = out.get("edge_safe_10px")
    zp_rms = float(flux_fit.residual_rms) if flux_fit.residual_rms is not None and flux_fit.ok else float("nan")
    sigma_samples: list[float] = []
    implied_samples: list[float] = []

    depth_unresolvable = not depth.depth_resolvable or depth.confirmable_depth_g is None
    confirmable_depth = float(depth.confirmable_depth_g) if not depth_unresolvable else float("nan")
    noise_frame = _estimate_frame_noise_adu(out, frame_noise_adu=frame_noise_adu)
    meta["frame_noise_adu_used"] = noise_frame

    for idx in out.index[dao_mask]:
        f = flux.loc[idx] if idx in flux.index else float("nan")
        cls = DAO_ONLY_CLASS_INDETERMINATE
        ig = float("nan")
        sig_g = float("nan")
        ig_minus = float("nan")

        if pd.notna(f) and float(f) <= 0:
            cls = DAO_ONLY_CLASS_ARTIFACT_NEGATIVE
        elif depth_unresolvable:
            cls = DAO_ONLY_CLASS_INDETERMINATE
            meta.setdefault("indeterminate_reasons", {})
            meta["indeterminate_reasons"]["depth_unresolvable"] = (
                int(meta["indeterminate_reasons"].get("depth_unresolvable", 0)) + 1
            )
        else:
            indet = False
            if flux_fit.ok and flux_fit.zp is not None and pd.notna(f) and float(f) > 0:
                ig = implied_g_from_flux(float(f), zp=float(flux_fit.zp))
                out.at[idx, "implied_g_mag"] = ig
                implied_samples.append(ig)
                ig_minus = float(ig) - confirmable_depth
                out.at[idx, "implied_g_minus_depth"] = ig_minus
            else:
                indet = True
            if sat is not None:
                sv = str(sat.loc[idx]).strip().lower()
                if sv in ("true", "1", "t", "yes"):
                    indet = True
            if edge is not None:
                ev = str(edge.loc[idx]).strip().lower()
                if ev in ("false", "0", "f", "no"):
                    indet = True
            if peak is not None and (not pd.notna(peak.loc[idx])):
                indet = True
            noise = _row_noise_adu(out.loc[idx], frame_noise_adu=noise_frame)
            snr = row_snr_from_flux(f, noise_adu=noise)
            if math.isfinite(zp_rms):
                sig_g = sigma_g_row(zp_residual_rms=zp_rms, snr=snr)
                if math.isfinite(sig_g):
                    out.at[idx, "sigma_g_row"] = sig_g
                    sigma_samples.append(sig_g)
            if indet or not math.isfinite(ig):
                cls = DAO_ONLY_CLASS_INDETERMINATE
            elif not math.isfinite(sig_g):
                cls = DAO_ONLY_CLASS_INDETERMINATE
            elif ig < confirmable_depth - sig_g:
                cls = DAO_ONLY_CLASS_UNMATCHED_IN_RANGE
            elif ig > confirmable_depth + sig_g:
                cls = DAO_ONLY_CLASS_BEYOND_CATALOGUE
            else:
                cls = DAO_ONLY_CLASS_AMBIGUOUS_DEPTH
        out.at[idx, "dao_only_class"] = cls
        counts[cls] = int(counts.get(cls, 0)) + 1

    if sigma_samples:
        sig_arr = np.asarray(sigma_samples, dtype=np.float64)
        meta["sigma_g_row_median"] = float(np.median(sig_arr))
        meta["sigma_g_row_mean"] = float(np.mean(sig_arr))
        meta["sigma_g_row_deciles"] = _implied_g_deciles(sig_arr)
        n_unmeas = int(np.sum(sig_arr > SIGMA_G_UNMEASURABLE_THRESHOLD_MAG))
        meta["sigma_g_unmeasurable_threshold_mag"] = SIGMA_G_UNMEASURABLE_THRESHOLD_MAG
        meta["sigma_g_unmeasurable_n"] = n_unmeas
        meta["sigma_g_unmeasurable_fraction"] = float(n_unmeas / sig_arr.size)
    meta["dao_only_implied_g_deciles"] = _implied_g_deciles(np.asarray(implied_samples, dtype=np.float64))
    meta["counts"] = counts
    meta["n_dao_only"] = int(dao_mask.sum())
    return out, meta


def format_dao_only_census_log(meta: dict[str, Any], *, n_total: int) -> str:
    """Informational census line with per-class breakdown (not a gate)."""
    counts = meta.get("counts") or {}
    n_dao = int(meta.get("n_dao_only") or sum(int(v) for v in counts.values()))
    frac = float(n_dao) / float(n_total) if n_total > 0 else 0.0
    parts = [
        f"artifact_negative={int(counts.get(DAO_ONLY_CLASS_ARTIFACT_NEGATIVE, 0))}",
        f"unmatched_in_range={int(counts.get(DAO_ONLY_CLASS_UNMATCHED_IN_RANGE, 0))}",
        f"ambiguous_depth={int(counts.get(DAO_ONLY_CLASS_AMBIGUOUS_DEPTH, 0))}",
        f"beyond_catalogue={int(counts.get(DAO_ONLY_CLASS_BEYOND_CATALOGUE, 0))}",
        f"indeterminate={int(counts.get(DAO_ONLY_CLASS_INDETERMINATE, 0))}",
    ]
    depth = meta.get("confirmable_depth_g")
    winner = meta.get("confirmable_depth_winner") or "?"
    depth_s = f"{float(depth):.2f}" if depth is not None and math.isfinite(float(depth)) else "unresolved"
    if not meta.get("depth_resolvable", True):
        depth_s = f"unresolved ({meta.get('depth_unresolvable_reason', '?')})"
    ff = meta.get("flux_fit") or {}
    zp_rms = ff.get("residual_rms")
    extras: list[str] = []
    if zp_rms is not None and math.isfinite(float(zp_rms)):
        extras.append(f"flux-to-G RMS={float(zp_rms):.3f} mag")
    un_frac = meta.get("sigma_g_unmeasurable_fraction")
    if un_frac is not None:
        thr = meta.get("sigma_g_unmeasurable_threshold_mag", SIGMA_G_UNMEASURABLE_THRESHOLD_MAG)
        extras.append(
            f"unmeasurable(sigma_g>{float(thr):.1f})={float(un_frac):.3f}"
        )
    tail = f" | {'; '.join(extras)}" if extras else ""
    return (
        f"MASTERSTAR DAO_ONLY census: {n_dao}/{n_total} (fraction={frac:.3f}) "
        f"[{', '.join(parts)}] | confirmable_depth G={depth_s} (from {winner}) | "
        f"informational, not a gate{tail}"
    )



def dao_only_class_meta_flat(meta: dict[str, Any]) -> dict[str, Any]:
    """Flat keys for pipeline_meta."""
    counts = meta.get("counts") or {}
    gdb = meta.get("gaia_db_identity") if isinstance(meta.get("gaia_db_identity"), dict) else {}
    flat: dict[str, Any] = {
        "dao_only_class_counts": dict(counts),
        "dao_only_n_artifact_negative": int(counts.get(DAO_ONLY_CLASS_ARTIFACT_NEGATIVE, 0)),
        "dao_only_n_unmatched_in_range": int(counts.get(DAO_ONLY_CLASS_UNMATCHED_IN_RANGE, 0)),
        "dao_only_n_ambiguous_depth": int(counts.get(DAO_ONLY_CLASS_AMBIGUOUS_DEPTH, 0)),
        "dao_only_n_beyond_catalogue": int(counts.get(DAO_ONLY_CLASS_BEYOND_CATALOGUE, 0)),
        "dao_only_n_indeterminate": int(counts.get(DAO_ONLY_CLASS_INDETERMINATE, 0)),
        "confirmable_depth_g": meta.get("confirmable_depth_g"),
        "confirmable_depth_inputs": meta.get("confirmable_depth_inputs"),
        "confirmable_depth_winner": meta.get("confirmable_depth_winner"),
        "dao_only_depth_resolvable": meta.get("depth_resolvable"),
        "dao_only_implied_g_deciles": meta.get("dao_only_implied_g_deciles"),
        "dao_only_flux_fit_zp": meta.get("flux_fit", {}).get("zp"),
        "dao_only_flux_fit_residual_mad": meta.get("flux_fit", {}).get("residual_mad"),
        "dao_only_flux_fit_residual_rms": meta.get("flux_fit", {}).get("residual_rms"),
        "dao_only_sigma_g_row_median": meta.get("sigma_g_row_median"),
        "dao_only_sigma_g_row_mean": meta.get("sigma_g_row_mean"),
        "dao_only_sigma_g_formula": meta.get("sigma_g_formula"),
        "dao_only_sigma_g_unmeasurable_threshold_mag": meta.get("sigma_g_unmeasurable_threshold_mag"),
        "dao_only_sigma_g_unmeasurable_n": meta.get("sigma_g_unmeasurable_n"),
        "dao_only_sigma_g_unmeasurable_fraction": meta.get("sigma_g_unmeasurable_fraction"),
        "fleming_sigma_mag_population": meta.get("fleming_sigma_mag_population"),
        "gaia_db_max_g_mag": meta.get("max_g_mag") or gdb.get("max_g_mag"),
        "gaia_db_fingerprint_sha256": gdb.get("fingerprint_sha256"),
        "gaia_db_row_count": gdb.get("row_count"),
    }
    return flat


def annotate_dao_only_magnitude_classes(
    df: pd.DataFrame,
    *,
    gaia_db_path: str | Any,
    effective_match_depth: float | None = None,
    cone_query_mag_limit: float | None = None,
    fleming_sigma_mag: float | None = None,
    frame_noise_adu: float | None = None,
    gaia_db_identity: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Classify DAO_ONLY rows by implied G vs derived confirmable depth; additive columns only."""
    if gaia_db_identity is None:
        try:
            from catalog_provenance import fingerprint_gaia_db  # noqa: PLC0415

            gaia_db_identity = fingerprint_gaia_db(gaia_db_path)
        except Exception:  # noqa: BLE001
            gaia_db_identity = {}
    max_g: float | None = None
    try:
        max_g = float(get_gaia_db_max_g_mag(gaia_db_path))
        if not math.isfinite(max_g) or max_g <= 0:
            max_g = None
    except Exception:  # noqa: BLE001
        max_g = None
    if gaia_db_identity and gaia_db_identity.get("max_g_mag") is not None:
        try:
            max_g = float(gaia_db_identity["max_g_mag"])
        except (TypeError, ValueError):
            pass
    depth = derive_confirmable_depth_g(
        gaia_db_max_g_mag=max_g,
        effective_match_depth=effective_match_depth,
        cone_query_mag_limit=cone_query_mag_limit,
    )
    matched = df.loc[~_dao_only_row_mask(df)].copy()
    fit = fit_instrumental_flux_to_g(matched)
    return classify_dao_only_dataframe(
        df,
        depth=depth,
        flux_fit=fit,
        fleming_sigma_mag=fleming_sigma_mag,
        frame_noise_adu=frame_noise_adu,
        gaia_db_identity=gaia_db_identity,
    )


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
    flat = {
        "g_lim_50": report.get("g_lim_50"),
        "g_lim_90": report.get("g_lim_90"),
        "g_lim_50_raw_fit": report.get("g_lim_50_raw_fit"),
        "g_lim_90_raw_fit": report.get("g_lim_90_raw_fit"),
        "g_lim_50_censored": report.get("g_lim_50_censored"),
        "g_lim_90_censored": report.get("g_lim_90_censored"),
        "g_lim_50_display": report.get("g_lim_50_display"),
        "g_lim_90_display": report.get("g_lim_90_display"),
        "g_lim_est": report.get("g_lim_50"),
        "fit_method": report.get("fit_method"),
        "completeness_curve": compact_curve,
        "completeness_50_label": report.get("completeness_50_label"),
        "reference_depth_g": report.get("reference_depth_g"),
        "match_depth": report.get("match_depth"),
        "match_depth_source": report.get("match_depth_source"),
        "n_ref_in_frame": report.get("n_ref_in_frame"),
        "n_gaia_matched": report.get("n_gaia_matched"),
        "n_gaia_off_frame": report.get("n_gaia_off_frame"),
        "n_gaia_below_limit": report.get("n_gaia_below_limit"),
        "n_gaia_blended": report.get("n_gaia_blended"),
        "n_gaia_missed": report.get("n_gaia_missed"),
        "n_missed_below_g90": report.get("n_missed_below_g90"),
        "n_missed_fadezone": report.get("n_missed_fadezone"),
        "missed_below_g90_uses_censored_depth": report.get("missed_below_g90_uses_censored_depth"),
        "gaia_dao_completeness_pct": report.get("gaia_dao_completeness_pct"),
        "gaia_dao_completeness_raw_pct": report.get("gaia_dao_completeness_raw_pct"),
        "n_dao_unmatched": report.get("n_dao_unmatched"),
        "n_dao_matched_to_faint": (report.get("unmatched_dao") or {}).get("n_now_matched_to_faint"),
        "blend_radius_px": report.get("blend_radius_px"),
        "blend_radius_arcsec": report.get("blend_radius_arcsec"),
        "dao_reconcile_methodology": report.get("methodology"),
    }
    if report.get("dao_only_class_meta"):
        flat.update(dao_only_class_meta_flat(report["dao_only_class_meta"]))
    return flat
