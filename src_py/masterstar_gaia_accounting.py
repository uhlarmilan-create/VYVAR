"""MASTERSTAR-GAIA-01: Gaia-complete membership, honest source_state census, lock assignment.

Shared pass-2 acceptance (Part A audit + production), lock-existing assignment (Part B),
source_state census (Part C), BLENDED accounting (Part D).

G04-05 (code review 2, Milan 2026-10-29): the FORCED_SEED mechanism is gone. Catalogue
membership (DAO-GAIA-ERA-01 M1, ``expand_detection_to_catalog_membership``) gives every
on-chip Gaia star to the matching depth a row, so a seed never admitted anything. Only DAO
detections (own ``vy_dao_pass`` in {1, 2} and a finite ``peak_dao``) own a Gaia star.
"""
from __future__ import annotations

import copy
import math
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pipeline_constants import (
    MS_DAO_PASS1_DEDUP_PX,
    MS_GAIA_CENSUS_EDGE_MARGIN_PX,
    MS_LOCK_LEFTOVER_RADIUS_PX,
    MS_LOCK_PAIR_TOL_PX,
)
from photutils.detection import DAOStarFinder
from photutils.utils.exceptions import NoDetectionsWarning
from scipy.spatial import cKDTree

from plain_stats import plain_mean_med_std
from utils import DAO_STAR_FINDER_NO_ROUNDNESS_FILTER

# --- source_state values (Part C) ---
SOURCE_DETECTED_P1 = "DETECTED_P1"
SOURCE_DETECTED_P2 = "DETECTED_P2"
SOURCE_BLENDED = "BLENDED"
SOURCE_TOO_FAINT = "TOO_FAINT"
SOURCE_SATURATED = "SATURATED"
SOURCE_EDGE = "EDGE"
SOURCE_CATALOG_ONLY = "CATALOG_ONLY"
SOURCE_CATALOG_MEMBERSHIP = "catalog_membership"

ALL_SOURCE_STATES = frozenset(
    {
        SOURCE_DETECTED_P1,
        SOURCE_DETECTED_P2,
        SOURCE_BLENDED,
        SOURCE_TOO_FAINT,
        SOURCE_SATURATED,
        SOURCE_EDGE,
        SOURCE_CATALOG_ONLY,
        SOURCE_CATALOG_MEMBERSHIP,
    }
)

# Census states that may label a non-detection row (INV-SOURCE-STATE-01).
_NONDET_CENSUS_STATES = frozenset(
    {
        SOURCE_CATALOG_MEMBERSHIP,
        SOURCE_CATALOG_ONLY,
        SOURCE_TOO_FAINT,
        SOURCE_SATURATED,
        SOURCE_EDGE,
        SOURCE_BLENDED,
    }
)


# --- POINT-SOURCE-SHAPE-01 (F1, RUN522-CRASH-01) ---------------------------
# Shape statistic q = (sum of the 8 neighbours of the peak pixel) / peak, after
# local background subtraction. A star spreads light over neighbouring pixels
# (q follows the PSF of this image); a hot pixel or cosmic-ray hit does not
# (q ~ 0). The acceptance floor is MEASURED on the same image from Gaia-matched
# pass-1 stars, so it adapts to every rig (well or under-sampled) - no
# rig-specific number. Constants below are statistical, not instrumental.
#: Reference stars need peak S/N >= this so their q is measured almost noise-free.
SHAPE_REF_PEAK_SNR_MIN = 20.0
#: Lower envelope of the reference q distribution (1st percentile).
SHAPE_REF_QUANTILE = 0.01
#: Minimum reference stars for a usable floor; fewer -> no shape gate (logged).
SHAPE_REF_N_MIN = 30
#: Reject a candidate only when its q is below the floor by this many sigma.
SHAPE_REJECT_NSIGMA = 3.0


@dataclass(frozen=True)
class Pass2AcceptParams:
    sigma: float = 4.0
    center_tol_px: float = 2.0
    cutout_half_width: int = 10
    fwhm_px: float = 5.3
    #: POINT-SOURCE-SHAPE-01 floor on q (None = gate off).
    shape_q_floor: float | None = None


def point_source_shape_q(
    data0: np.ndarray,
    x: float,
    y: float,
    *,
    local_bg: float,
    local_sigma: float,
) -> tuple[float, float]:
    """Return (q, peak_snr) at the brightest pixel within +-1 px of (x, y).

    q = sum(8 neighbours) / peak after subtracting ``local_bg``;
    peak_snr = peak / ``local_sigma``. NaN when undefined (edge, peak <= 0).
    """
    h, w = data0.shape
    ix, iy = int(round(float(x))), int(round(float(y)))
    if not (2 <= ix < w - 2 and 2 <= iy < h - 2):
        return float("nan"), float("nan")
    bg = float(local_bg) if math.isfinite(float(local_bg)) else 0.0
    box = np.asarray(data0[iy - 1 : iy + 2, ix - 1 : ix + 2], dtype=np.float64) - bg
    if not np.all(np.isfinite(box)):
        return float("nan"), float("nan")
    py, px = np.unravel_index(int(np.argmax(box)), box.shape)
    ix2, iy2 = ix + int(px) - 1, iy + int(py) - 1
    b = np.asarray(data0[iy2 - 1 : iy2 + 2, ix2 - 1 : ix2 + 2], dtype=np.float64) - bg
    pk = float(b[1, 1])
    if not (math.isfinite(pk) and pk > 0):
        return float("nan"), float("nan")
    q = (float(np.sum(b)) - pk) / pk
    sd = float(local_sigma)
    snr = pk / sd if math.isfinite(sd) and sd > 0 else float("nan")
    return float(q), float(snr)


def point_source_shape_ok(q: float, peak_snr: float, q_floor: float | None) -> bool:
    """False only when q is significantly below the star floor (single-pixel event).

    Noise on q from 9 pixels of sigma s: sigma_q = sqrt(8 + q_floor^2) / peak_snr.
    Undefined q or S/N -> not rejected here (other gates decide).
    """
    if q_floor is None or not math.isfinite(float(q_floor)):
        return True
    if not (math.isfinite(q) and math.isfinite(peak_snr) and peak_snr > 0):
        return True
    qf = float(q_floor)
    sig_q = math.sqrt(8.0 + qf * qf) / float(peak_snr)
    return bool(float(q) >= qf - SHAPE_REJECT_NSIGMA * sig_q)


def derive_point_source_q_floor(
    data0: np.ndarray,
    det_x: np.ndarray,
    det_y: np.ndarray,
    gaia_x: np.ndarray,
    gaia_y: np.ndarray,
    *,
    match_r_px: float,
) -> dict[str, Any]:
    """Measure the q floor from pass-1 detections matched to Gaia on this image.

    Returns dict with ``q_floor`` (None if too few references), ``q_p50``,
    ``n_ref`` and ``method``. Matching to Gaia excludes hot pixels / CRs from
    the reference set; peak S/N >= SHAPE_REF_PEAK_SNR_MIN keeps q noise-free.
    """
    out: dict[str, Any] = {
        "q_floor": None,
        "q_p50": None,
        "n_ref": 0,
        "method": (
            f"POINT-SOURCE-SHAPE-01: q=sum8/peak, Gaia-matched pass-1, "
            f"peak_snr>={SHAPE_REF_PEAK_SNR_MIN:g}, p{100 * SHAPE_REF_QUANTILE:g}"
        ),
    }
    dx = np.asarray(det_x, dtype=np.float64).ravel()
    dy = np.asarray(det_y, dtype=np.float64).ravel()
    gx = np.asarray(gaia_x, dtype=np.float64).ravel()
    gy = np.asarray(gaia_y, dtype=np.float64).ravel()
    okd = np.isfinite(dx) & np.isfinite(dy)
    okg = np.isfinite(gx) & np.isfinite(gy)
    if not (okd.any() and okg.any()):
        return out
    dx, dy = dx[okd], dy[okd]
    tree = cKDTree(np.column_stack([gx[okg], gy[okg]]))
    dist, _ = tree.query(np.column_stack([dx, dy]), distance_upper_bound=float(match_r_px))
    sel = np.isfinite(dist)
    qs: list[float] = []
    for x0, y0 in zip(dx[sel], dy[sel], strict=False):
        md, sd = _dao_pass2_annulus_stats(data0, float(x0), float(y0))
        q, snr = point_source_shape_q(data0, x0, y0, local_bg=md, local_sigma=sd)
        if math.isfinite(q) and math.isfinite(snr) and snr >= SHAPE_REF_PEAK_SNR_MIN:
            qs.append(q)
    out["n_ref"] = int(len(qs))
    if len(qs) < SHAPE_REF_N_MIN:
        return out
    arr = np.asarray(qs, dtype=np.float64)
    out["q_floor"] = float(np.quantile(arr, SHAPE_REF_QUANTILE))
    out["q_p50"] = float(np.median(arr))
    return out


def _dao_pass2_annulus_stats(data0: np.ndarray, cx: float, cy: float) -> tuple[float, float]:
    """Local background median and std on annulus r=8-12 px (bg-subtracted image)."""
    h, w = data0.shape
    rmax = 13
    ix, iy = int(round(cx)), int(round(cy))
    x0, x1 = max(0, ix - rmax), min(w, ix + rmax + 1)
    y0, y1 = max(0, iy - rmax), min(h, iy + rmax + 1)
    if x1 <= x0 or y1 <= y0:
        return float("nan"), float("nan")
    yy, xx = np.mgrid[y0:y1, x0:x1]
    rr = np.hypot(xx - cx, yy - cy)
    ann = data0[y0:y1, x0:x1][(rr >= 8.0) & (rr <= 12.0)]
    if ann.size < 10:
        return float("nan"), float("nan")
    _, md, sd = plain_mean_med_std(ann)
    return float(md), float(sd)


#: SPEED-2026-10: one DAOStarFinder (kernel) per pass-2 FWHM, reused for every cutout.
_PASS2_FINDERS: dict[float, DAOStarFinder] = {}


def _pass2_catalog(cutout: np.ndarray, fwhm: float, threshold: float) -> Any:
    """``DAOStarFinder(fwhm, threshold)(cutout)`` without the output table: the same catalogue
    (photutils ``find_stars`` = raw catalogue -> ``apply_all_filters`` -> ``to_table``; the table
    only copies the columns). The finder of a FWHM is built once (its kernel) and the per-cutout
    threshold is set as its __init__ sets it. 67k cutouts per night: the table build and the
    kernel were 45 % of the pass-2 time."""
    base = _PASS2_FINDERS.get(fwhm)
    if base is None:
        base = DAOStarFinder(fwhm=fwhm, threshold=1.0, n_brightest=None, **DAO_STAR_FINDER_NO_ROUNDNESS_FILTER)
        _PASS2_FINDERS[fwhm] = base
    # G04-38: the cached finder is never mutated; each call works on its own shallow copy (the kernel is
    # shared read-only), so pass-2 stays correct if it is ever run in threads. The private
    # ``_get_raw_catalog`` is pinned by test_pass2_catalog_equals_public_finder (photutils < 4).
    f = copy.copy(base)
    f.threshold = threshold
    f.threshold_eff = threshold * f.kernel.rel_err if f.scale_threshold else threshold
    cat = f._get_raw_catalog(cutout)
    return None if cat is None else cat.apply_all_filters()


def dao_pass2_try_at_position(
    data0: np.ndarray,
    x0: float,
    y0: float,
    *,
    wpx: int,
    h: int,
    params: Pass2AcceptParams | None = None,
) -> dict[str, Any]:
    """Run production pass-2 acceptance at one position. Returns accept flag + diagnostics."""
    p = params or Pass2AcceptParams()
    hw = int(p.cutout_half_width)
    ix, iy = int(round(x0)), int(round(y0))
    xlo, xhi = max(0, ix - hw), min(int(wpx), ix + hw + 1)
    ylo, yhi = max(0, iy - hw), min(int(h), iy + hw + 1)
    out: dict[str, Any] = {
        "accepted": False,
        "x_seed": float(x0),
        "y_seed": float(y0),
        "x_det": float("nan"),
        "y_det": float("nan"),
        "centroid_px": float("nan"),
        "local_std": float("nan"),
        "threshold_adu": float("nan"),
        "flux": float("nan"),
        "peak": float("nan"),
        "reason": "",
    }
    if xhi - xlo < 7 or yhi - ylo < 7:
        out["reason"] = "cutout_too_small"
        return out
    local_bg, local_std = _dao_pass2_annulus_stats(data0, float(x0), float(y0))
    out["local_std"] = float(local_std) if math.isfinite(local_std) else float("nan")
    if not (math.isfinite(local_std) and local_std > 0):
        out["reason"] = "bad_local_std"
        return out
    sigma_p2 = max(1.5, min(20.0, float(p.sigma)))
    thr2 = max(sigma_p2 * float(local_std), 1e-6)
    out["threshold_adu"] = float(thr2)
    fwhm_cut = max(1.2, min(20.0, float(p.fwhm_px)))
    cutout = data0[ylo:yhi, xlo:xhi]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NoDetectionsWarning)
            cat = _pass2_catalog(cutout, float(fwhm_cut), float(thr2))
    except Exception:  # noqa: BLE001
        out["reason"] = "finder_error"
        return out
    if cat is None or len(cat) == 0:
        out["reason"] = "no_detection"
        return out
    xc = np.asarray(cat.x_centroid, dtype=np.float64)
    yc = np.asarray(cat.y_centroid, dtype=np.float64)
    x_full = xlo + xc
    y_full = ylo + yc
    dctr = np.hypot(x_full - x0, y_full - y0)
    j = int(np.argmin(dctr))
    dmin = float(dctr[j])
    out["x_det"] = float(x_full[j])
    out["y_det"] = float(y_full[j])
    out["centroid_px"] = dmin
    if dmin > float(p.center_tol_px):
        out["reason"] = "centroid_tol"
        return out
    if p.shape_q_floor is not None:
        q, psnr = point_source_shape_q(
            data0, out["x_det"], out["y_det"], local_bg=local_bg, local_sigma=local_std
        )
        out["shape_q"] = float(q)
        if not point_source_shape_ok(q, psnr, p.shape_q_floor):
            out["reason"] = "shape_not_point_source"
            return out
    flux_np = np.asarray(cat.flux, dtype=np.float64)
    peak_np = np.asarray(cat.peak, dtype=np.float64)
    out["flux"] = float(flux_np[j])
    out["peak"] = float(peak_np[j])
    out["accepted"] = True
    out["reason"] = "ok"
    return out


def _star_mask_disk(r: int) -> np.ndarray:
    y, x = np.ogrid[-r : r + 1, -r : r + 1]
    return (x * x + y * y) <= r * r


def star_mask_from_gaia_xy(
    gx: np.ndarray,
    gy: np.ndarray,
    *,
    wpx: int,
    h: int,
    fwhm_px: float,
) -> np.ndarray:
    """True = stellar pixel (exclude from global sky estimate)."""
    mask = np.zeros((int(h), int(wpx)), dtype=bool)
    r = max(int(round(2.0 * float(fwhm_px))), 3)
    d = _star_mask_disk(r)
    dh, dw = d.shape
    for x, y in zip(gx, gy, strict=False):
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        ix, iy = int(round(x)), int(round(y))
        y0, y1 = max(0, iy - dh // 2), min(int(h), iy + dh // 2 + 1)
        x0, x1 = max(0, ix - dw // 2), min(int(wpx), ix + dw // 2 + 1)
        sy0 = max(0, dh // 2 - iy)
        sx0 = max(0, dw // 2 - ix)
        sy1 = sy0 + (y1 - y0)
        sx1 = sx0 + (x1 - x0)
        mask[y0:y1, x0:x1] |= d[sy0:sy1, sx0:sx1]
    return mask


def estimate_star_masked_sky_sigma(
    data0: np.ndarray,
    *,
    star_mask: np.ndarray | None = None,
) -> tuple[float, float]:
    """Star-masked sky sigma (1.4826 MAD, no clipping) on bg-subtracted image (DAO-GAIA iter4 pass1 basis)."""
    from plain_stats import sky_mad_sigma_adu

    bg = data0[~star_mask] if star_mask is not None else data0.ravel()
    bg_fin = bg[np.isfinite(bg)]
    if bg_fin.size < 1000:
        bg_fin = data0.ravel()
    med_clip, sig_clip = sky_mad_sigma_adu(bg_fin)  # median / 1.4826 MAD, no clipping
    sig = float(sig_clip) if math.isfinite(sig_clip) and sig_clip > 0 else float("nan")
    med = float(med_clip) if math.isfinite(med_clip) else float("nan")
    return sig, med


def check_ambiguous_owner(
    x_det: float,
    y_det: float,
    owner_j: int,
    gaia_df: pd.DataFrame,
    *,
    margin_px: float = 1.0,
    gxy: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[bool, int]:
    """True when centroid is >margin_px closer to a different Gaia star than the seed.

    ``gxy``: the Gaia pixel positions of ``gaia_df`` already converted by the caller (G04-20: was a
    ``pd.to_numeric`` of both columns on every call)."""
    if gxy is not None:
        gx, gy = gxy
        if len(gx) != len(gaia_df) or len(gy) != len(gaia_df):
            raise ValueError(
                f"check_ambiguous_owner: gxy has {len(gx)}/{len(gy)} positions for {len(gaia_df)} Gaia rows"
            )
    else:
        gx = pd.to_numeric(gaia_df.get("x_gaia", gaia_df.get("x")), errors="coerce").to_numpy(dtype=np.float64)
        gy = pd.to_numeric(gaia_df.get("y_gaia", gaia_df.get("y")), errors="coerce").to_numpy(dtype=np.float64)
    if owner_j < 0 or owner_j >= len(gaia_df):
        return False, -1
    xg0 = float(gx[owner_j])
    yg0 = float(gy[owner_j])
    if not (math.isfinite(xg0) and math.isfinite(yg0)):
        return False, -1
    d_owner = float(math.hypot(x_det - xg0, y_det - yg0))
    d_all = np.hypot(x_det - gx, y_det - gy)
    d_all[owner_j] = np.inf
    d_all[~np.isfinite(d_all)] = np.inf
    best_j = int(np.argmin(d_all)) if d_all.size else -1
    best_d = float(d_all[best_j]) if best_j >= 0 else float("inf")
    if not math.isfinite(best_d):
        best_j = -1
    if math.isfinite(best_d) and math.isfinite(d_owner) and best_d + float(margin_px) < d_owner:
        return True, best_j
    return False, -1


def dedup_pass1_spatial(tbl_pass1: Any, *, sep_px: float = 0.75) -> Any:
    """Collapse pass1 peaks closer than sep_px (keep brightest flux)."""
    import numpy as np
    from astropy.table import Table

    if tbl_pass1 is None or len(tbl_pass1) <= 1:
        return tbl_pass1
    t = Table(tbl_pass1, copy=True)
    flux = np.asarray(t["flux"], dtype=np.float64)
    xb = np.asarray(t["x_centroid"], dtype=np.float64)
    yb = np.asarray(t["y_centroid"], dtype=np.float64)
    order = np.argsort(-flux)
    # Greedy brightest-first (unchanged rule); neighbours from a KD-tree instead of a scan of
    # every kept peak (O(n^2) Python: ~1 min for 14 000 MASTERSTAR peaks).
    tree = cKDTree(np.column_stack([xb, yb]))
    kept = np.zeros(len(flux), dtype=bool)
    keep: list[int] = []
    for idx in order:
        nb = tree.query_ball_point([xb[idx], yb[idx]], r=float(sep_px))
        if any(kept[j] and float(math.hypot(xb[idx] - xb[j], yb[idx] - yb[j])) < float(sep_px) for j in nb):
            continue
        kept[idx] = True
        keep.append(int(idx))
    keep.sort()
    return t[keep]


def merge_dao_pass1_pass2_born_owned(
    tbl_pass1: Any,
    pass2_rows: list[dict[str, Any]],
    *,
    gaia_chip: pd.DataFrame | None = None,
) -> Any:
    """Append born-owned pass2; dedup only same-owner duplicates (not global 3 px pass1 collision)."""
    import numpy as np
    from astropy.table import Table, vstack

    if tbl_pass1 is not None and len(tbl_pass1) > 0 and "vy_dao_pass" not in tbl_pass1.colnames:
        tbl_pass1 = Table(tbl_pass1, copy=True)
        tbl_pass1["vy_dao_pass"] = np.ones(len(tbl_pass1), dtype=np.int16)
    if not pass2_rows:
        return tbl_pass1

    cid_to_xy: dict[str, tuple[float, float]] = {}
    if gaia_chip is not None and not gaia_chip.empty:
        gx = pd.to_numeric(gaia_chip.get("x_gaia"), errors="coerce").to_numpy(dtype=np.float64)
        gy = pd.to_numeric(gaia_chip.get("y_gaia"), errors="coerce").to_numpy(dtype=np.float64)
        cids = gaia_chip.get("catalog_id", pd.Series([""] * len(gaia_chip))).map(_norm_cid_strip_dotzero)
        for j in range(len(gaia_chip)):
            cid = _norm_cid_strip_dotzero(cids.iloc[j])
            if cid and math.isfinite(gx[j]) and math.isfinite(gy[j]):
                cid_to_xy[cid] = (float(gx[j]), float(gy[j]))

    by_owner: dict[str, dict[str, Any]] = {}
    for row in pass2_rows:
        cid = _norm_cid_strip_dotzero(row.get("vy_seed_catalog_id", ""))
        if not cid:
            continue
        prev = by_owner.get(cid)
        if prev is None:
            by_owner[cid] = row
            continue
        xg, yg = cid_to_xy.get(cid, (float("nan"), float("nan")))
        d_new = float(math.hypot(float(row["x_full"]) - xg, float(row["y_full"]) - yg)) if math.isfinite(xg) else 0.0
        d_old = float(math.hypot(float(prev["x_full"]) - xg, float(prev["y_full"]) - yg)) if math.isfinite(xg) else 0.0
        if d_new < d_old:
            by_owner[cid] = row

    kept: list[dict[str, Any]] = []
    for row in _dedup_pass2_by_position(list(by_owner.values()), gaia_chip):
        xb, yb = float(row["x_binned"]), float(row["y_binned"])
        kept.append(
            {
                "x_centroid": xb,
                "y_centroid": yb,
                "flux": float(row["flux"]),
                "peak": float(row.get("peak", row["flux"])),
                "vy_dao_pass": 2,
                "vy_seed_catalog_id": str(row.get("vy_seed_catalog_id", "")),
                "vy_ambiguous_owner": bool(row.get("vy_ambiguous_owner", False)),
            }
        )
    if not kept:
        return tbl_pass1
    t2 = Table(kept)
    if tbl_pass1 is None or len(tbl_pass1) == 0:
        return t2
    return vstack([tbl_pass1, t2])


def _dedup_pass2_by_position(
    rows: list[dict[str, Any]], gaia_chip: pd.DataFrame | None, *, sep_px: float = MS_DAO_PASS1_DEDUP_PX
) -> list[dict[str, Any]]:
    """G04-07: one pass-2 detection per centroid.

    Two Gaia seeds closer than the pass-2 centre tolerance can both accept the same DAO centroid; each
    seed then made its own MASTERSTAR row at one pixel (full flux twice, two Gaia ids). Detections
    whose full-resolution centroids lie within ``sep_px`` (the pass-1 duplicate radius) are one
    detection: the brightest seed (smallest Gaia G; ties by catalog id) owns it. The other seed keeps
    no row, so the census marks it BLENDED next to the owner (annotate_blended_groups)."""
    if len(rows) <= 1:
        return rows
    gmag: dict[str, float] = {}
    if gaia_chip is not None and not gaia_chip.empty and "catalog_id" in gaia_chip.columns:
        gm = pd.to_numeric(gaia_chip.get("g_mag", gaia_chip.get("mag")), errors="coerce").to_numpy(dtype=np.float64)
        for cid, g in zip(gaia_chip["catalog_id"].map(_norm_cid_strip_dotzero), gm):
            if cid and math.isfinite(float(g)):
                gmag[cid] = float(g)

    def _key(r: dict[str, Any]) -> tuple[float, str]:
        cid = _norm_cid_strip_dotzero(r.get("vy_seed_catalog_id", ""))
        return (gmag.get(cid, math.inf), cid)

    xy = np.array([[float(rows[i]["x_full"]), float(rows[i]["y_full"])] for i in range(len(rows))], dtype=np.float64)
    kept = one_detection_per_centroid(xy, [_key(r) for r in rows], sep_px=float(sep_px))
    return [rows[i] for i in range(len(rows)) if kept[i]]


def one_detection_per_centroid(xy_full: np.ndarray, owner_keys: list[Any], *, sep_px: float) -> np.ndarray:
    """G04-07 / G04-26: the rule shared by the MASTERSTAR and the per-frame DAO pass 2 - pass-2 detections whose
    full-resolution centroids lie within ``sep_px`` of an already kept one are the same detection.

    Rows are visited in the order of ``owner_keys`` (smallest first: brightest seed, ties by catalogue id;
    a stable sort keeps the input order for equal keys); the first one keeps the centroid. Returns the
    boolean mask of kept rows."""
    n = len(owner_keys)
    kept = np.zeros(n, dtype=bool)
    if n == 0:
        return kept
    xy = np.asarray(xy_full, dtype=np.float64)
    order = sorted(range(n), key=lambda i: owner_keys[i])
    tree = cKDTree(xy)
    for i in order:
        if any(kept[j] for j in tree.query_ball_point(xy[i], r=float(sep_px))):
            continue
        kept[i] = True
    return kept


def _dao_xy_binned_to_full(x: np.ndarray, y: np.ndarray, f: int) -> tuple[np.ndarray, np.ndarray]:
    if f <= 1:
        return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    s = float(f)
    off = float(f - 1) * 0.5
    return np.asarray(x, dtype=np.float64) * s + off, np.asarray(y, dtype=np.float64) * s + off


def _dao_full_to_binned_xy(x_full: float, y_full: float, bfac: int) -> tuple[float, float]:
    if int(bfac) <= 1:
        return float(x_full), float(y_full)
    s = float(bfac)
    off = float(bfac - 1) * 0.5
    return (float(x_full) - off) / s, (float(y_full) - off) / s


def dao_pass2_born_owned_rows(
    data0: np.ndarray,
    tbl_pass1: Any,
    *,
    gaia_chip: pd.DataFrame,
    bfac: int,
    pass2_params: Pass2AcceptParams,
    target_depth_g: float,
    edge_margin_px: float,
    match_r_px: float,
    wpx: int,
    h: int,
) -> tuple[list[dict[str, Any]], int, int, list[dict[str, Any]]]:
    """Born-owned pass2 for on-chip Gaia G<=depth without pass1 neighbor. No global rematch."""
    import numpy as np

    if gaia_chip is None or gaia_chip.empty:
        return [], 0, 0, []

    gx = pd.to_numeric(gaia_chip["x_gaia"], errors="coerce").to_numpy(dtype=np.float64)
    gy = pd.to_numeric(gaia_chip["y_gaia"], errors="coerce").to_numpy(dtype=np.float64)
    gm = pd.to_numeric(gaia_chip.get("g_mag", gaia_chip.get("mag")), errors="coerce").to_numpy(dtype=np.float64)
    cids = gaia_chip["catalog_id"].map(_norm_cid_strip_dotzero).to_numpy(dtype=object)

    dao_x = np.asarray([], dtype=np.float64)
    dao_y = np.asarray([], dtype=np.float64)
    if tbl_pass1 is not None and len(tbl_pass1) > 0:
        xb = np.asarray(tbl_pass1["x_centroid"], dtype=np.float64)
        yb = np.asarray(tbl_pass1["y_centroid"], dtype=np.float64)
        dao_x, dao_y = _dao_xy_binned_to_full(xb, yb, int(bfac))

    _fin_d = np.isfinite(dao_x) & np.isfinite(dao_y)
    _dao_tree = cKDTree(np.column_stack([dao_x[_fin_d], dao_y[_fin_d]])) if bool(_fin_d.any()) else None
    unmatched: list[int] = []
    for j in range(len(gaia_chip)):
        if not (math.isfinite(gx[j]) and math.isfinite(gy[j])):
            continue
        g = float(gm[j]) if math.isfinite(gm[j]) else float("nan")
        if not (math.isfinite(g) and g <= float(target_depth_g)):
            continue
        if (
            gx[j] < edge_margin_px
            or gy[j] < edge_margin_px
            or gx[j] >= float(wpx) - edge_margin_px
            or gy[j] >= float(h) - edge_margin_px
        ):
            continue
        if _dao_tree is not None:
            d_nn, _ = _dao_tree.query([gx[j], gy[j]], k=1)
            if float(d_nn) <= float(match_r_px):
                continue
        unmatched.append(j)

    pass2_rows: list[dict[str, Any]] = []
    ambiguous_pairs: list[dict[str, Any]] = []
    for j in unmatched:
        x0, y0 = float(gx[j]), float(gy[j])
        cid = _norm_cid_strip_dotzero(cids[j])
        hit = dao_pass2_try_at_position(data0, x0, y0, wpx=int(wpx), h=int(h), params=pass2_params)
        if not hit.get("accepted"):
            continue
        xd, yd = float(hit["x_det"]), float(hit["y_det"])
        amb, j_other = check_ambiguous_owner(xd, yd, j, gaia_chip, gxy=(gx, gy))
        xb, yb = _dao_full_to_binned_xy(xd, yd, int(bfac))
        pass2_rows.append(
            {
                "x_full": xd,
                "y_full": yd,
                "x_binned": float(xb),
                "y_binned": float(yb),
                "flux": float(hit["flux"]),
                "peak": float(hit.get("peak", hit["flux"])),
                "vy_seed_catalog_id": cid,
                "vy_ambiguous_owner": amb,
            }
        )
        if amb and j_other >= 0:
            ambiguous_pairs.append(
                {
                    "owner_catalog_id": cid,
                    "other_catalog_id": _norm_cid_strip_dotzero(cids[j_other]),
                    "x_det": xd,
                    "y_det": yd,
                }
            )
    return pass2_rows, len(unmatched), len(pass2_rows), ambiguous_pairs


def _row_is_dao_detected(peak_dao: object, vy_dao_pass: object) -> tuple[bool, int]:
    """INV-SOURCE-STATE-01: DETECTED_Pn requires this row's own pass and peak.

    Column presence is not a detection. ``vy_dao_pass`` fillna(1) is forbidden here.
    Returns (is_detected, pass_number) with pass_number in {1, 2} when detected.
    """
    try:
        peak = float(peak_dao)
    except (TypeError, ValueError):
        peak = float("nan")
    if not math.isfinite(peak) or peak <= 0.0:
        return False, 0
    try:
        vp = int(float(vy_dao_pass))
    except (TypeError, ValueError):
        return False, 0
    if vp not in (1, 2):
        return False, 0
    return True, vp


def _norm_cid_strip_dotzero(v: object) -> str:
    """Strip whitespace and a trailing ``.0`` on all-digit strings. Not Gaia-canonical."""
    s = str(v or "").strip()
    if not s or s.lower() in {"nan", "none"}:
        return ""
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return s


def _require_unique_index(df: pd.DataFrame, what: str) -> None:
    """Fail loud when ``df`` has a non-unique index (G04-20): the label writes (``.loc[df.index[i]]``) and
    the positional label decisions of the MASTERSTAR Gaia accounting address one row per label."""
    if not df.index.is_unique:
        raise ValueError(f"MASTERSTAR Gaia accounting: the {what} table index is not unique")


def lock_existing_and_leftover_assign(
    det_x: np.ndarray,
    det_y: np.ndarray,
    gaia_df: pd.DataFrame,
    *,
    locked_pairs: dict[str, tuple[float, float]] | None = None,
    leftover_radius_px: float = 3.0,
    lock_tol_px: float = 3.0,
    identity_fail_px: float | None = None,
    det_catalog_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Return (det_to_gaia_row, gaia_owner_det, vy_match_mode per det, geometry_reject_det_indices).

    det_to_gaia_row[i] = row index in gaia_df or -1
    gaia_owner_det[j] = det index or -1
    vy_match_mode[i] = 'locked' | 'leftover_promotion' | ''
    geometry_reject_det_indices: born-owned detections farther than
    identity_fail_px from Gaia xy (D4: reject threshold is the identity-gate
    fail radius 3 x FWHM_dao, not the derived lock preference). When
    identity_fail_px is omitted, reject uses lock_tol_px (STAGE-01 callers).
    """
    n_d = int(len(det_x))
    n_g = int(len(gaia_df))
    det_to_g = np.full(n_d, -1, dtype=np.int64)
    gaia_owner = np.full(n_g, -1, dtype=np.int64)
    match_mode = np.array([""] * n_d, dtype=object)

    gx = pd.to_numeric(gaia_df.get("x_gaia", gaia_df.get("x")), errors="coerce").to_numpy(
        dtype=np.float64
    )
    gy = pd.to_numeric(gaia_df.get("y_gaia", gaia_df.get("y")), errors="coerce").to_numpy(
        dtype=np.float64
    )
    cids = gaia_df["catalog_id"].map(_norm_cid_strip_dotzero).to_numpy(dtype=object) if "catalog_id" in gaia_df.columns else np.array([""] * n_g, dtype=object)
    cid_to_row = {_norm_cid_strip_dotzero(cids[i]): i for i in range(n_g) if _norm_cid_strip_dotzero(cids[i])}

    used_det: set[int] = set()
    used_g: set[int] = set()
    geometry_reject_dets: list[int] = []
    fail_px = float(identity_fail_px) if identity_fail_px is not None else float(lock_tol_px)
    if not math.isfinite(fail_px) or fail_px <= 0:
        fail_px = float(lock_tol_px)

    if locked_pairs:
        det_cids = (
            np.asarray([_norm_cid_strip_dotzero(c) for c in det_catalog_ids], dtype=object)
            if det_catalog_ids is not None
            else np.array([""] * n_d, dtype=object)
        )
        # G04-20: detections by their own catalog_id (ascending index) and a KD-tree over the
        # detections, built once; the per-pair rules below are unchanged (were two scans of every
        # detection per locked pair: O(n_lock x n_det)).
        dets_by_cid: dict[str, list[int]] = {}
        for i in range(min(n_d, len(det_cids))):
            dc = _norm_cid_strip_dotzero(det_cids[i])
            if dc:
                dets_by_cid.setdefault(dc, []).append(i)
        dx_a = np.asarray(det_x, dtype=np.float64)
        dy_a = np.asarray(det_y, dtype=np.float64)
        fin_d = np.flatnonzero(np.isfinite(dx_a) & np.isfinite(dy_a))
        det_tree = cKDTree(np.column_stack([dx_a[fin_d], dy_a[fin_d]])) if fin_d.size else None
        r_lock = float(lock_tol_px) * (1.0 + 1e-9) + 1e-9
        for cid, (_lx, _ly) in locked_pairs.items():
            gr = cid_to_row.get(_norm_cid_strip_dotzero(cid))
            if gr is None or gr in used_g:
                continue
            nc = _norm_cid_strip_dotzero(cid)
            gx_g = float(gx[gr]) if gr < len(gx) else float("nan")
            gy_g = float(gy[gr]) if gr < len(gy) else float("nan")
            best_i = -1
            best_d = float("inf")
            # Preference: born-owned within lock_tol_px. Keep (still lock) when
            # lock_tol < d <= identity_fail. Reject only past identity_fail_px (D4).
            for i in dets_by_cid.get(nc, ()):
                if i in used_det:
                    continue
                d_gaia = float(math.hypot(det_x[i] - gx_g, det_y[i] - gy_g))
                if not math.isfinite(d_gaia):
                    continue
                if d_gaia <= float(lock_tol_px):
                    best_i = i
                    best_d = d_gaia
                    break
                if d_gaia > float(fail_px):
                    geometry_reject_dets.append(i)
                elif d_gaia < best_d:
                    best_i = i
                    best_d = d_gaia
            if (best_i < 0 and det_tree is not None and math.isfinite(gx_g) and math.isfinite(gy_g)
                    and math.isfinite(float(lock_tol_px)) and float(lock_tol_px) >= 0):
                # nearest unused detection within lock_tol_px, ties to the lowest index (the scan's
                # strict '<'); the tree gives candidates, the test is the same math.hypot
                for i in sorted(int(fin_d[c]) for c in det_tree.query_ball_point((gx_g, gy_g), r_lock)):
                    if i in used_det:
                        continue
                    d = float(math.hypot(det_x[i] - gx_g, det_y[i] - gy_g))
                    if d <= float(lock_tol_px) and d < best_d:
                        best_d = d
                        best_i = i
            if best_i >= 0:
                used_det.add(best_i)
                used_g.add(gr)
                det_to_g[best_i] = gr
                gaia_owner[gr] = best_i
                match_mode[best_i] = "locked"

    pairs: list[tuple[float, int, int]] = []
    if n_d and n_g:
        free_d = [i for i in range(n_d) if i not in used_det]
        free_g = [j for j in range(n_g) if j not in used_g]
        if free_d and free_g:
            tree = cKDTree(np.column_stack([gx[free_g], gy[free_g]]))
            dist, idx = tree.query(
                np.column_stack([det_x[free_d], det_y[free_d]]),
                k=min(32, len(free_g)),
                distance_upper_bound=float(leftover_radius_px),
            )
            dist = np.atleast_2d(np.asarray(dist, dtype=np.float64))
            idx = np.atleast_2d(np.asarray(idx, dtype=np.int64))
            for k, i in enumerate(free_d):
                for kk in range(idx.shape[1]):
                    jloc = int(idx[k, kk])
                    d = float(dist[k, kk])
                    if jloc < 0 or jloc >= len(free_g) or not math.isfinite(d):
                        continue
                    j = free_g[jloc]
                    if d <= float(leftover_radius_px):
                        pairs.append((d, i, j))
    pairs.sort(key=lambda t: t[0])
    for _d, i, j in pairs:
        if i in used_det or j in used_g:
            continue
        used_det.add(i)
        used_g.add(j)
        det_to_g[i] = j
        gaia_owner[j] = i
        if not match_mode[i]:
            match_mode[i] = "leftover_promotion"

    return det_to_g, gaia_owner, match_mode, geometry_reject_dets


def annotate_blended_groups(
    gaia_df: pd.DataFrame,
    gaia_owner: np.ndarray,
    *,
    fwhm_px: float,
    blend_sep_px: float | None = None,
) -> pd.DataFrame:
    """Mark BLENDED non-owners in Gaia pairs closer than FWHM."""
    sep = float(blend_sep_px) if blend_sep_px is not None else float(fwhm_px)
    out = gaia_df.copy()
    n = len(out)
    if n < 2:
        out["blend_group_id"] = np.array([""] * n, dtype=object)
        out["blend_flux_ratio"] = np.full(n, np.nan, dtype=np.float64)
        return out
    gx = pd.to_numeric(out.get("x_gaia", out.get("x")), errors="coerce").to_numpy(dtype=np.float64)
    gy = pd.to_numeric(out.get("y_gaia", out.get("y")), errors="coerce").to_numpy(dtype=np.float64)
    gm = pd.to_numeric(out.get("g_mag", out.get("mag")), errors="coerce").to_numpy(dtype=np.float64)
    blend_gid = np.array([""] * n, dtype=object)
    blend_fr = np.full(n, np.nan, dtype=np.float64)
    tree = cKDTree(np.column_stack([gx, gy]))
    pairs = tree.query_pairs(float(sep))
    group_serial = 0
    for i, j in sorted(pairs):
        owner_i = int(gaia_owner[i]) >= 0
        owner_j = int(gaia_owner[j]) >= 0
        if owner_i and owner_j:
            continue
        if not owner_i and not owner_j:
            continue
        group_serial += 1
        gid = f"BLEND_{group_serial:05d}"
        if owner_i and not owner_j:
            blend_gid[j] = gid
            if math.isfinite(gm[i]) and math.isfinite(gm[j]) and gm[j] > 0:
                blend_fr[j] = float(10 ** (-0.4 * (gm[i] - gm[j])))
        elif owner_j and not owner_i:
            blend_gid[i] = gid
            if math.isfinite(gm[i]) and math.isfinite(gm[j]) and gm[i] > 0:
                blend_fr[i] = float(10 ** (-0.4 * (gm[j] - gm[i])))
    out["blend_group_id"] = blend_gid
    out["blend_flux_ratio"] = blend_fr
    return out


def build_gaia_census_rows(
    gaia_on_chip: pd.DataFrame,
    gaia_owner: np.ndarray,
    det_pass: np.ndarray,
    *,
    target_depth_g: float,
    edge_margin_px: float,
    wpx: int,
    h: int,
) -> pd.DataFrame:
    """One row per on-chip Gaia with source_state (does not alter MS detection table).

    ``gaia_owner[j]`` is the MASTERSTAR row index of the DAO detection that owns Gaia star ``j``
    (-1: none); ``det_pass[i]`` is that row's own DAO pass (1 or 2; 0 = not a detection).
    An owned star is DETECTED_P1/P2 from the owner's own pass. An unowned star inside the chip
    margin and to the depth is CATALOG_ONLY (G04-05: no forced seed probe any more).
    """
    rows: list[dict[str, Any]] = []
    gx = pd.to_numeric(gaia_on_chip.get("x_gaia"), errors="coerce").to_numpy(dtype=np.float64)
    gy = pd.to_numeric(gaia_on_chip.get("y_gaia"), errors="coerce").to_numpy(dtype=np.float64)
    gm = pd.to_numeric(gaia_on_chip.get("g_mag", gaia_on_chip.get("mag")), errors="coerce").to_numpy(
        dtype=np.float64
    )
    for j in range(len(gaia_on_chip)):
        cid = _norm_cid_strip_dotzero(gaia_on_chip.iloc[j].get("catalog_id"))
        xg, yg = float(gx[j]), float(gy[j])
        gmag = float(gm[j]) if math.isfinite(gm[j]) else float("nan")
        state = SOURCE_TOO_FAINT
        owner = int(gaia_owner[j]) if j < len(gaia_owner) else -1
        if (
            xg < edge_margin_px
            or yg < edge_margin_px
            or xg >= float(wpx) - edge_margin_px
            or yg >= float(h) - edge_margin_px
        ):
            state = SOURCE_EDGE
        elif math.isfinite(gmag) and gmag > float(target_depth_g):
            state = SOURCE_TOO_FAINT
        elif owner >= 0:
            p = int(det_pass[owner])
            if p not in (1, 2):
                raise ValueError(f"Gaia {cid} owned by row {owner} that is not a DAO detection (pass={p})")
            state = SOURCE_DETECTED_P2 if p == 2 else SOURCE_DETECTED_P1
        elif math.isfinite(gmag):
            state = SOURCE_CATALOG_ONLY
        rows.append(
            {
                "catalog_id": cid,
                "x_gaia": xg,
                "y_gaia": yg,
                "g_mag": gmag,
                "source_state": state,
                "ambiguous_owner": False,
            }
        )
    return pd.DataFrame(rows)


def verify_gaia_census_complete(census: pd.DataFrame, n_on_chip: int) -> tuple[bool, str]:
    """INV-MS-CENSUS-01: census rows sum to on-chip Gaia count."""
    n = int(len(census)) if census is not None else 0
    if n != int(n_on_chip):
        return False, f"census {n} != on_chip {n_on_chip}"
    if n == 0:
        return True, "ok n=0"
    bad = census[~census["source_state"].isin(ALL_SOURCE_STATES)]
    if len(bad):
        return False, f"invalid states {len(bad)}"
    return True, f"ok n={n}"


def verify_gaia_census(census: pd.DataFrame, *, n_on_chip: int) -> dict[str, Any]:
    """FAIL-loud INV-MS-CENSUS-01 on the in-memory census (OUTPUT-REVIEW-2026-10: no census file).

    Mismatch vs on-chip count raises.
    """
    from invariants_runtime import inv_check

    out = census if census is not None else pd.DataFrame()
    ok, det = verify_gaia_census_complete(out, int(n_on_chip))
    meta: dict[str, Any] = {"invariants": []}
    inv_check(meta, "INV-MS-CENSUS-01", ok, policy="FAIL", detail=det)
    return {"ok": True, "detail": det, "n": int(len(out)), "invariants": meta.get("invariants") or []}


def gaia_on_chip_from_cone(
    cone_df: pd.DataFrame,
    *,
    gx: np.ndarray,
    gy: np.ndarray,
    ok_mask: np.ndarray,
    wpx: int,
    h: int,
) -> pd.DataFrame:
    """Subset cone rows with finite sky coords whose projected pixel is on-chip."""
    inb = (
        (gx >= 0)
        & (gx < float(wpx))
        & (gy >= 0)
        & (gy < float(h))
    )
    chip = cone_df.loc[ok_mask].iloc[inb].copy()
    chip["x_gaia"] = gx[inb]
    chip["y_gaia"] = gy[inb]
    chip["g_mag"] = pd.to_numeric(chip.get("mag"), errors="coerce")
    return chip


def expand_detection_to_catalog_membership(
    df_ms: pd.DataFrame,
    gaia_on_chip: pd.DataFrame,
    *,
    membership_depth_g: float,
    wpx: int,
    h: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """DAO-GAIA-ERA-01 M1: additive catalog membership; row existence never depends on detection.

    Every on-chip Gaia star with G <= ``membership_depth_g`` receives a MASTERSTAR row at the
    propagated Gaia position when missing from the detection table. All DAO_ONLY detection rows
    are preserved. Detection rows are never dropped.
    """
    meta: dict[str, Any] = {
        "membership_depth_g": float(membership_depth_g),
        "n_detection_rows_in": int(len(df_ms)) if df_ms is not None else 0,
        "n_catalog_rows_added": 0,
        "n_dao_only_preserved": 0,
    }
    out = df_ms.copy() if df_ms is not None else pd.DataFrame()
    if gaia_on_chip is None or gaia_on_chip.empty:
        return out, meta

    present: set[str] = set()
    if "catalog_id" in out.columns:
        for raw in out["catalog_id"].tolist():
            cid = _norm_cid_strip_dotzero(raw)
            if cid:
                present.add(cid)
    if "catalog_id" in out.columns:
        cid_col = out["catalog_id"].map(_norm_cid_strip_dotzero)
        meta["n_dao_only_preserved"] = int((cid_col.eq("")).sum())

    gdf = gaia_on_chip.copy()
    gx = pd.to_numeric(gdf.get("x_gaia"), errors="coerce").to_numpy(dtype=np.float64)
    gy = pd.to_numeric(gdf.get("y_gaia"), errors="coerce").to_numpy(dtype=np.float64)
    gm = pd.to_numeric(
        gdf.get("g_mag", gdf.get("mag")),
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    depth = float(membership_depth_g)
    new_rows: list[dict[str, Any]] = []

    for j in range(len(gdf)):
        cid = _norm_cid_strip_dotzero(gdf.iloc[j].get("catalog_id"))
        if not cid or cid in present:
            continue
        xg, yg = float(gx[j]), float(gy[j])
        if not (math.isfinite(xg) and math.isfinite(yg)):
            continue
        if xg < 0 or yg < 0 or xg >= float(wpx) or yg >= float(h):
            continue
        gmag = float(gm[j]) if j < len(gm) and math.isfinite(gm[j]) else float("nan")
        # G04-37: saturated detections are already in ``present``; only the depth decides here
        if not (math.isfinite(gmag) and gmag <= depth):
            continue
        row = {
            "name": cid,
            "catalog_id": cid,
            "catalog": "Gaia",
            "source_type": "GAIA_MATCHED",
            "x": xg,
            "y": yg,
            "ra_deg": gdf.iloc[j].get("ra_deg", gdf.iloc[j].get("ra")),
            "dec_deg": gdf.iloc[j].get("dec_deg", gdf.iloc[j].get("dec")),
            "mag": gmag,
            "flux": float("nan"),
            "forced_photometry": False,
            "is_usable": False,
            "photometry_ok": False,
            "vy_match_mode": "catalog_membership",
        }
        if "bp_rp" in gdf.columns:
            row["bp_rp"] = gdf.iloc[j].get("bp_rp")
        # GAIA-VAR-ID-2026-10: the Gaia DR3 variability flag of the catalogue row, as for detections
        row["gaia_dr3_variable_catalog"] = str(gdf.iloc[j].get("var_flag") or "").strip().upper() == "VARIABLE"
        new_rows.append(row)
        present.add(cid)

    if new_rows:
        out = pd.concat([out, pd.DataFrame(new_rows)], ignore_index=True)
        meta["n_catalog_rows_added"] = int(len(new_rows))
    meta["n_rows_out"] = int(len(out))
    return out, meta


def verify_ms_expand_guard(expand_meta: dict[str, Any]) -> tuple[bool, str]:
    """INV-MS-EXPAND-01: catalog expand ran and reported rows_added (census + certificate are checked in memory)."""
    if not expand_meta:
        return False, "missing catalog_derived_membership meta"
    n_in = int(expand_meta.get("n_detection_rows_in", -1))
    n_out = int(expand_meta.get("n_rows_out", -1))
    if n_in < 0 or n_out < 0:
        return False, f"expand row counts missing n_in={n_in} n_out={n_out}"
    if n_out < n_in:
        return False, f"membership shrink n_in={n_in} n_out={n_out}"
    if "n_catalog_rows_added" not in expand_meta:
        return False, "n_catalog_rows_added not reported"
    return True, (
        f"ok n_in={n_in} n_out={n_out} added={int(expand_meta.get('n_catalog_rows_added', 0))}"
    )


def dao_detection_pass(df: pd.DataFrame) -> np.ndarray:
    """Per row: its own DAO pass (1 or 2) when the row is a DAO detection, else 0.

    INV-SOURCE-STATE-01 / G04-05: one rule (``_row_is_dao_detected``) - own ``vy_dao_pass`` in
    {1, 2} and a finite positive ``peak_dao``; no fillna. Catalogue-membership rows are never
    detections.
    """
    n = len(df)
    out = np.zeros(n, dtype=np.int16)
    if n == 0 or "peak_dao" not in df.columns or "vy_dao_pass" not in df.columns:
        return out
    peak = df["peak_dao"].to_numpy(dtype=object)
    vpass = df["vy_dao_pass"].to_numpy(dtype=object)
    for i in range(n):
        ok, p = _row_is_dao_detected(peak[i], vpass[i])
        if ok:
            out[i] = p
    return out


def _acc_identity_locks(
    *,
    catalog_derived_membership,
    fwhm_px,
    gaia_on_chip,
    identity_lock_only,
    locked_pairs,
    meta,
    out,
    tolerance_overrides,
):
    """Gaia stars on the chip and their identity locks to DAO detections (owner map).

    Only DAO detection rows (``dao_detection_pass`` > 0) take part: a catalogue-membership row
    sits at its own Gaia position and would otherwise lock its own star (G04-05).
    Returns (gaia_owner as MASTERSTAR row index or -1, gdf, identity_lock_only, det_pass, n_g).
    """
    gdf = gaia_on_chip.copy()
    if "x_gaia" not in gdf.columns:
        gdf["x_gaia"] = pd.to_numeric(gdf.get("x"), errors="coerce")
        gdf["y_gaia"] = pd.to_numeric(gdf.get("y"), errors="coerce")
    if "g_mag" not in gdf.columns and "mag" in gdf.columns:
        gdf["g_mag"] = pd.to_numeric(gdf["mag"], errors="coerce")

    n_g = len(gdf)
    det_pass = dao_detection_pass(out)
    det_rows = np.flatnonzero(det_pass > 0)
    det_x = pd.to_numeric(out.get("x"), errors="coerce").to_numpy(dtype=np.float64)[det_rows]
    det_y = pd.to_numeric(out.get("y"), errors="coerce").to_numpy(dtype=np.float64)[det_rows]
    det_cids = (
        out["catalog_id"].map(_norm_cid_strip_dotzero).to_numpy(dtype=object)[det_rows]
        if "catalog_id" in out.columns
        else np.array([""] * len(det_rows), dtype=object)
    )

    if locked_pairs is None:
        locked_pairs = {}
        for k, i in enumerate(det_rows):
            cid = det_cids[k]
            if cid and math.isfinite(det_x[k]) and math.isfinite(det_y[k]):
                # Born-owned pass2 wins over pass1 when both carry the same catalog_id.
                if cid not in locked_pairs or int(det_pass[i]) == 2:
                    locked_pairs[cid] = (float(det_x[k]), float(det_y[k]))

    if catalog_derived_membership:
        identity_lock_only = True
    leftover_r = (
        0.0
        if identity_lock_only
        else float(
            (tolerance_overrides or {}).get(
                "lock_leftover_radius_px",
                MS_LOCK_LEFTOVER_RADIUS_PX,
            )
        )
    )
    lock_tol = float(
        (tolerance_overrides or {}).get(
            "lock_pair_tol_px",
            MS_LOCK_PAIR_TOL_PX,
        )
    )
    # D4: reject radius is the identity-gate fail threshold (3 x FWHM_dao), not lock_tol.
    _fwhm_lock = float(fwhm_px) if math.isfinite(float(fwhm_px)) and float(fwhm_px) > 0 else 3.5
    identity_fail_px = float(
        (tolerance_overrides or {}).get("identity_fail_px", 3.0 * _fwhm_lock)
    )
    det_to_g, owner_k, vy_modes, _geom_rej = lock_existing_and_leftover_assign(
        det_x,
        det_y,
        gdf,
        locked_pairs=locked_pairs,
        leftover_radius_px=leftover_r,
        lock_tol_px=lock_tol,
        identity_fail_px=identity_fail_px,
        det_catalog_ids=det_cids,
    )
    # Owner as MASTERSTAR row index (the assignment works on the detection subset).
    gaia_owner = np.full(n_g, -1, dtype=np.int64)
    _owned = owner_k >= 0
    gaia_owner[_owned] = det_rows[owner_k[_owned]]
    meta["n_lock_geometry_reject"] = int(len({int(k) for k in _geom_rej}))
    meta["identity_fail_px"] = float(identity_fail_px)
    meta["lock_pair_tol_px_effective"] = float(lock_tol)
    if _geom_rej:
        from wcs_invertibility import clear_row_match_identity, det_fallback_name

        for k in _geom_rej:
            if int(det_to_g[k]) >= 0:
                continue
            i = int(det_rows[k])
            clear_row_match_identity(out, out.index[i], det_name=det_fallback_name(i + 1))

    if "vy_match_mode" not in out.columns:
        out["vy_match_mode"] = ""
    for k, i in enumerate(det_rows):
        idx = out.index[int(i)]
        if vy_modes[k]:
            out.loc[idx, "vy_match_mode"] = str(vy_modes[k])
        gr = int(det_to_g[k])
        if (not identity_lock_only) and gr >= 0 and gr < n_g:
            cid = _norm_cid_strip_dotzero(gdf.iloc[gr].get("catalog_id"))
            if cid:
                out.loc[idx, "catalog_id"] = cid
                out.loc[idx, "source_type"] = "GAIA_MATCHED"
                if str(vy_modes[k]) == "leftover_promotion":
                    meta["n_leftover_promotions"] = int(meta["n_leftover_promotions"]) + 1

    return gaia_owner, gdf, identity_lock_only, det_pass, n_g


def _acc_census(
    *,
    det_pass,
    edge_m,
    fwhm_px,
    gaia_owner,
    gdf,
    h,
    meta,
    n_g,
    out,
    target_depth_g,
    wpx,
):
    """Blended groups and the MASTERSTAR Gaia census rows (detected / blended / catalogue only / below limit)."""
    gdf = annotate_blended_groups(gdf, gaia_owner, fwhm_px=fwhm_px)
    census = build_gaia_census_rows(
        gdf,
        gaia_owner,
        det_pass,
        target_depth_g=float(target_depth_g),
        edge_margin_px=edge_m,
        wpx=int(wpx),
        h=int(h),
    )
    _require_unique_index(census, "Gaia census")  # G04-20: census.loc label writes below
    _bg_col = gdf["blend_group_id"] if "blend_group_id" in gdf.columns else pd.Series([""] * n_g, index=gdf.index)
    _bg_arr = _bg_col.fillna("").astype(str).str.strip().to_numpy(dtype=object)
    if len(census) and "catalog_id" in census.columns and bool((_bg_arr != "").any()):
        _census_rows: dict[str, list[Any]] = {}
        for _ci, _cc in zip(census.index, census["catalog_id"]):
            _census_rows.setdefault(_cc, []).append(_ci)
        _bfr = gdf["blend_flux_ratio"].to_numpy() if "blend_flux_ratio" in gdf.columns else None
        _gcid = gdf["catalog_id"] if "catalog_id" in gdf.columns else pd.Series([""] * n_g, index=gdf.index)
        for j in np.nonzero(_bg_arr != "")[0]:
            cid = _norm_cid_strip_dotzero(_gcid.iloc[j])
            if cid and int(gaia_owner[j]) < 0:
                _rows = _census_rows.get(cid)
                if _rows:
                    census.loc[_rows, "source_state"] = SOURCE_BLENDED
                    census.loc[_rows, "blend_group_id"] = _bg_arr[j]
                    census.loc[_rows, "blend_flux_ratio"] = _bfr[j] if _bfr is not None else None

    if "source_state" not in out.columns:
        out["source_state"] = ""
    cid_out = out.get("catalog_id", pd.Series([""] * len(out))).map(_norm_cid_strip_dotzero)
    census_st: dict[str, str] = {}
    if len(census):
        # G04-20: column-wise (was census.iloc[ci] per row); the last row of an id wins, as before
        _c_st = census["source_state"].tolist() if "source_state" in census.columns else [""] * len(census)
        _c_id = census["catalog_id"].tolist() if "catalog_id" in census.columns else [None] * len(census)
        for _cid_c, _st_c in zip(_c_id, _c_st, strict=True):
            cc = _norm_cid_strip_dotzero(_cid_c)
            if cc:
                census_st[cc] = str(_st_c).strip()
    # G04-20: the per-row state is decided on lists and written once (was out.loc per row: ~7 s for
    # 9 000 rows). Row i is read and written only at step i, so the values are the same as the
    # former row-by-row label writes - for a unique index, which the MASTERSTAR table has.
    _require_unique_index(out, "masterstars")
    _st_cur = out["source_state"].tolist()
    _mode_cur = out["vy_match_mode"].tolist() if "vy_match_mode" in out.columns else None
    _cid_cur = cid_out.tolist()
    _new_pos: list[int] = []
    _new_val: list[str] = []
    for i in range(len(out)):
        cid = _norm_cid_strip_dotzero(_cid_cur[i])
        if not cid:
            _new_pos.append(i)
            _new_val.append("DAO_ONLY")
            continue
        st = _st_cur[i]
        if str(st).strip() in (SOURCE_DETECTED_P1, SOURCE_DETECTED_P2):
            # Pre-set DETECTED_* is only kept when this row itself is a DAO detection.
            pass
        elif str(st).strip():
            continue
        vp = int(det_pass[i])
        if vp in (1, 2):
            _new_pos.append(i)
            _new_val.append(SOURCE_DETECTED_P2 if vp == 2 else SOURCE_DETECTED_P1)
            continue
        mode = str(_mode_cur[i] or "").strip() if _mode_cur is not None else ""
        if mode == "catalog_membership":
            _new_pos.append(i)
            _new_val.append(SOURCE_CATALOG_MEMBERSHIP)
            continue
        cst = str(census_st.get(cid, "") or "").strip()
        _new_pos.append(i)
        _new_val.append(cst if cst in _NONDET_CENSUS_STATES else SOURCE_CATALOG_ONLY)
    if _new_pos:
        _st_col = out.columns.get_loc("source_state")
        out.iloc[_new_pos, _st_col] = np.asarray(_new_val, dtype=object)

    meta["n_on_chip_gaia"] = n_g
    meta["census_by_state"] = (
        census["source_state"].value_counts().to_dict() if len(census) else {}
    )
    if "ambiguous_owner" in out.columns and len(census):
        amb_map: dict[str, bool] = {}
        _o_id = out["catalog_id"].tolist() if "catalog_id" in out.columns else [None] * len(out)
        for _cid_o, raw_amb in zip(_o_id, out["ambiguous_owner"].tolist(), strict=True):
            cid = _norm_cid_strip_dotzero(_cid_o)
            if not cid:
                continue
            amb_map[cid] = (
                bool(raw_amb)
                if raw_amb is not None and not (isinstance(raw_amb, float) and not math.isfinite(raw_amb))
                else False
            )
        _c_id2 = census["catalog_id"].tolist() if "catalog_id" in census.columns else [None] * len(census)
        _amb_rows = [ci for ci, _cid_c in enumerate(_c_id2) if amb_map.get(_norm_cid_strip_dotzero(_cid_c))]
        for ci in _amb_rows:
            census.loc[census.index[ci], "ambiguous_owner"] = True
    return census


def enrich_masterstar_gaia_complete(
    df_ms: pd.DataFrame,
    *,
    gaia_on_chip: pd.DataFrame,
    wpx: int,
    h: int,
    fwhm_px: float,
    target_depth_g: float,
    locked_pairs: dict[str, tuple[float, float]] | None = None,
    identity_lock_only: bool = False,
    catalog_derived_membership: bool = False,
    tolerance_overrides: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Apply lock-leftover assignment, BLENDED + census (Parts B-D). Never adds rows.

    ``identity_lock_only=True`` keeps existing MS row identities (no leftover
    catalog_id rewrite).

    ``catalog_derived_membership=True`` (DAO-GAIA-ERA-01 M1): membership rows were
    pre-expanded from the catalog; do not rewrite catalog_id via leftover promotion.
    """
    meta: dict[str, Any] = {
        "n_leftover_promotions": 0,
        "n_lock_geometry_reject": 0,
    }
    out = df_ms.copy() if df_ms is not None else pd.DataFrame()

    if gaia_on_chip is None or gaia_on_chip.empty:
        return out, pd.DataFrame(), meta
    # G04-20: every write below addresses one masterstars row per label - check before the first one
    _require_unique_index(out, "masterstars")

    gaia_owner, gdf, identity_lock_only, det_pass, n_g = _acc_identity_locks(
        catalog_derived_membership=catalog_derived_membership,
        fwhm_px=fwhm_px,
        gaia_on_chip=gaia_on_chip,
        identity_lock_only=identity_lock_only,
        locked_pairs=locked_pairs,
        meta=meta,
        out=out,
        tolerance_overrides=tolerance_overrides,
    )
    census = _acc_census(
        det_pass=det_pass,
        edge_m=MS_GAIA_CENSUS_EDGE_MARGIN_PX,
        fwhm_px=fwhm_px,
        gaia_owner=gaia_owner,
        gdf=gdf,
        h=h,
        meta=meta,
        n_g=n_g,
        out=out,
        target_depth_g=target_depth_g,
        wpx=wpx,
    )
    meta["identity_lock_only"] = bool(identity_lock_only)
    meta["catalog_derived_membership"] = bool(catalog_derived_membership)
    return out, census, meta
