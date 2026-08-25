"""MASTERSTAR-GAIA-01: Gaia-complete membership, honest source_state census, lock assignment.

Shared pass-2 / forced-seed acceptance (Part A audit + production), lock-existing
assignment (Part B), FORCED_SEED admission (Part C), BLENDED accounting (Part D).
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from photutils.aperture import CircularAnnulus, CircularAperture, aperture_photometry
from photutils.detection import DAOStarFinder
from photutils.utils.exceptions import NoDetectionsWarning
from scipy.spatial import cKDTree

from plain_stats import plain_mean_med_std
from utils import DAO_STAR_FINDER_NO_ROUNDNESS_FILTER

# --- source_state values (Part C) ---
SOURCE_DETECTED_P1 = "DETECTED_P1"
SOURCE_DETECTED_P2 = "DETECTED_P2"
SOURCE_FORCED_SEED = "FORCED_SEED"
SOURCE_BLENDED = "BLENDED"
SOURCE_TOO_FAINT = "TOO_FAINT"
SOURCE_SATURATED = "SATURATED"
SOURCE_EDGE = "EDGE"
SOURCE_SEED_REJECTED = "SEED_REJECTED"
SOURCE_CATALOG_ONLY = "CATALOG_ONLY"
SOURCE_CATALOG_MEMBERSHIP = "catalog_membership"

ALL_SOURCE_STATES = frozenset(
    {
        SOURCE_DETECTED_P1,
        SOURCE_DETECTED_P2,
        SOURCE_FORCED_SEED,
        SOURCE_BLENDED,
        SOURCE_TOO_FAINT,
        SOURCE_SATURATED,
        SOURCE_EDGE,
        SOURCE_SEED_REJECTED,
        SOURCE_CATALOG_ONLY,
        SOURCE_CATALOG_MEMBERSHIP,
    }
)

# Census states that may label a non-detection row (INV-SOURCE-STATE-01).
_NONDET_CENSUS_STATES = frozenset(
    {
        SOURCE_FORCED_SEED,
        SOURCE_SEED_REJECTED,
        SOURCE_CATALOG_MEMBERSHIP,
        SOURCE_CATALOG_ONLY,
        SOURCE_TOO_FAINT,
        SOURCE_SATURATED,
        SOURCE_EDGE,
        SOURCE_BLENDED,
    }
)


@dataclass(frozen=True)
class Pass2AcceptParams:
    sigma: float = 4.0
    center_tol_px: float = 2.0
    cutout_half_width: int = 10
    fwhm_px: float = 5.3


@dataclass(frozen=True)
class ForcedSeedAcceptParams:
    centroid_max_px: float = 2.0
    snr_min: float = 4.0
    aperture_r_px: float | None = None  # None -> 1.5 * fwhm


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
    _, md, sd = plain_mean_med_std(ann, sigma=3.0, maxiters=2)
    return float(md), float(sd)


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
    _, local_std = _dao_pass2_annulus_stats(data0, float(x0), float(y0))
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
            finder2 = DAOStarFinder(
                fwhm=float(fwhm_cut),
                threshold=float(thr2),
                n_brightest=None,
                **DAO_STAR_FINDER_NO_ROUNDNESS_FILTER,
            )
            tloc = finder2(cutout)
    except Exception:  # noqa: BLE001
        out["reason"] = "finder_error"
        return out
    if tloc is None or len(tloc) == 0:
        out["reason"] = "no_detection"
        return out
    xc = np.asarray(tloc["x_centroid"], dtype=np.float64)
    yc = np.asarray(tloc["y_centroid"], dtype=np.float64)
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
    flux_np = np.asarray(tloc["flux"], dtype=np.float64)
    peak_np = (
        np.asarray(tloc["peak"], dtype=np.float64) if "peak" in tloc.colnames else flux_np
    )
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
    """Star-masked sigma-clipped sky sigma on bg-subtracted image (DAO-GAIA iter4 pass1 basis)."""
    from astropy.stats import sigma_clipped_stats

    bg = data0[~star_mask] if star_mask is not None else data0.ravel()
    bg_fin = bg[np.isfinite(bg)]
    if bg_fin.size < 1000:
        bg_fin = data0.ravel()
    med_clip, _, sig_clip = sigma_clipped_stats(bg_fin, sigma=3.0, maxiters=3)
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
) -> tuple[bool, int]:
    """True when centroid is >margin_px closer to a different Gaia star than the seed."""
    gx = pd.to_numeric(gaia_df.get("x_gaia", gaia_df.get("x")), errors="coerce").to_numpy(dtype=np.float64)
    gy = pd.to_numeric(gaia_df.get("y_gaia", gaia_df.get("y")), errors="coerce").to_numpy(dtype=np.float64)
    if owner_j < 0 or owner_j >= len(gaia_df):
        return False, -1
    xg0 = float(gx[owner_j])
    yg0 = float(gy[owner_j])
    if not (math.isfinite(xg0) and math.isfinite(yg0)):
        return False, -1
    d_owner = float(math.hypot(x_det - xg0, y_det - yg0))
    best_d, best_j = float("inf"), -1
    for k in range(len(gaia_df)):
        if k == owner_j or not (math.isfinite(gx[k]) and math.isfinite(gy[k])):
            continue
        d = float(math.hypot(x_det - gx[k], y_det - gy[k]))
        if d < best_d:
            best_d, best_j = d, k
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
    keep: list[int] = []
    for idx in order:
        ok = True
        for j in keep:
            if float(math.hypot(xb[idx] - xb[j], yb[idx] - yb[j])) < float(sep_px):
                ok = False
                break
        if ok:
            keep.append(int(idx))
    keep.sort()
    return t[keep]


def merge_dao_pass1_pass2_born_owned(
    tbl_pass1: Any,
    pass2_rows: list[dict[str, Any]],
    *,
    bfac: int,
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
        cids = gaia_chip.get("catalog_id", pd.Series([""] * len(gaia_chip))).map(_norm_cid)
        for j in range(len(gaia_chip)):
            cid = _norm_cid(cids.iloc[j])
            if cid and math.isfinite(gx[j]) and math.isfinite(gy[j]):
                cid_to_xy[cid] = (float(gx[j]), float(gy[j]))

    by_owner: dict[str, dict[str, Any]] = {}
    for row in pass2_rows:
        cid = _norm_cid(row.get("vy_seed_catalog_id", ""))
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
    for row in by_owner.values():
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
    fwhm_px: float,
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
    cids = gaia_chip["catalog_id"].map(_norm_cid).to_numpy(dtype=object)

    dao_x = np.asarray([], dtype=np.float64)
    dao_y = np.asarray([], dtype=np.float64)
    if tbl_pass1 is not None and len(tbl_pass1) > 0:
        xb = np.asarray(tbl_pass1["x_centroid"], dtype=np.float64)
        yb = np.asarray(tbl_pass1["y_centroid"], dtype=np.float64)
        dao_x, dao_y = _dao_xy_binned_to_full(xb, yb, int(bfac))

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
        if dao_x.size:
            d = np.hypot(dao_x - gx[j], dao_y - gy[j])
            if float(np.min(d)) <= float(match_r_px):
                continue
        unmatched.append(j)

    pass2_rows: list[dict[str, Any]] = []
    ambiguous_pairs: list[dict[str, Any]] = []
    n_empty = 0
    for j in unmatched:
        x0, y0 = float(gx[j]), float(gy[j])
        cid = _norm_cid(cids[j])
        hit = dao_pass2_try_at_position(data0, x0, y0, wpx=int(wpx), h=int(h), params=pass2_params)
        if not hit.get("accepted"):
            if str(hit.get("reason", "")) == "no_detection":
                n_empty += 1
            continue
        xd, yd = float(hit["x_det"]), float(hit["y_det"])
        amb, j_other = check_ambiguous_owner(xd, yd, j, gaia_chip)
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
                    "other_catalog_id": _norm_cid(cids[j_other]),
                    "x_det": xd,
                    "y_det": yd,
                }
            )
    return pass2_rows, len(unmatched), len(pass2_rows), ambiguous_pairs


def forced_seed_measure_at_position(
    data0: np.ndarray,
    x: float,
    y: float,
    *,
    fwhm_px: float,
    params: ForcedSeedAcceptParams | None = None,
) -> dict[str, float]:
    """Forced aperture + COM centroid at propagated Gaia position."""
    p = params or ForcedSeedAcceptParams()
    r_px = float(p.aperture_r_px) if p.aperture_r_px is not None else max(1.5 * float(fwhm_px), 3.0)
    h, w = data0.shape
    if not (math.isfinite(x) and math.isfinite(y)):
        return {
            "flux": float("nan"),
            "sigma": float("nan"),
            "snr": float("nan"),
            "peak": float("nan"),
            "cx": x,
            "cy": y,
            "centroid_px": float("nan"),
            "local_sigma": float("nan"),
        }
    ap = CircularAperture((x, y), r=float(r_px))
    ann = CircularAnnulus((x, y), r_in=float(r_px) + 4.0, r_out=float(r_px) + 8.0)
    phot = aperture_photometry(data0, ap)
    flux_raw = float(phot["aperture_sum"][0])
    try:
        ann_mask = ann.to_mask(method="center")
        vals = ann_mask.multiply(data0)
        vals = vals[ann_mask.data > 0]
        vals = vals[np.isfinite(vals)]
        _, md, sd = (
            plain_mean_med_std(vals, sigma=3.0, maxiters=2)
            if vals.size >= 10
            else (float("nan"), float("nan"), float("nan"))
        )
    except Exception:  # noqa: BLE001
        md, sd = float("nan"), float("nan")
    area = math.pi * float(r_px) ** 2
    flux = flux_raw - float(md) * area if math.isfinite(md) else flux_raw
    snr = flux / (sd * math.sqrt(area)) if math.isfinite(sd) and sd > 0 else float("nan")
    ix, iy = int(round(x)), int(round(y))
    hw = 3
    x0, x1 = max(0, ix - hw), min(w, ix + hw + 1)
    y0, y1 = max(0, iy - hw), min(h, iy + hw + 1)
    stamp = data0[y0:y1, x0:x1]
    peak = float(np.nanmax(stamp)) if stamp.size else float("nan")
    yy, xx = np.mgrid[y0:y1, x0:x1]
    wgt = np.clip(stamp - (md if math.isfinite(md) else 0.0), 0, None)
    s = float(np.nansum(wgt))
    if s > 0:
        cx = float(np.nansum(xx * wgt) / s)
        cy = float(np.nansum(yy * wgt) / s)
    else:
        cx, cy = x, y
    dcent = float(math.hypot(cx - x, cy - y))
    return {
        "flux": float(flux),
        "sigma": float(sd) if math.isfinite(sd) else float("nan"),
        "snr": float(snr),
        "peak": float(peak),
        "cx": float(cx),
        "cy": float(cy),
        "centroid_px": dcent,
        "local_sigma": float(sd) if math.isfinite(sd) else float("nan"),
    }


def forced_seed_accept(
    meas: dict[str, float],
    *,
    params: ForcedSeedAcceptParams | None = None,
) -> tuple[bool, str]:
    p = params or ForcedSeedAcceptParams()
    dcent = float(meas.get("centroid_px", float("nan")))
    snr = float(meas.get("snr", float("nan")))
    if not math.isfinite(dcent):
        return False, "bad_centroid"
    if dcent > float(p.centroid_max_px):
        return False, "centroid_tol"
    if not math.isfinite(snr) or snr < float(p.snr_min):
        return False, "snr_low"
    return True, "ok"


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


def _norm_cid(v: object) -> str:
    s = str(v or "").strip()
    if not s or s.lower() in {"nan", "none"}:
        return ""
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return s


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
    cids = gaia_df["catalog_id"].map(_norm_cid).to_numpy(dtype=object) if "catalog_id" in gaia_df.columns else np.array([""] * n_g, dtype=object)
    cid_to_row = {_norm_cid(cids[i]): i for i in range(n_g) if _norm_cid(cids[i])}

    used_det: set[int] = set()
    used_g: set[int] = set()
    geometry_reject_dets: list[int] = []
    fail_px = float(identity_fail_px) if identity_fail_px is not None else float(lock_tol_px)
    if not math.isfinite(fail_px) or fail_px <= 0:
        fail_px = float(lock_tol_px)

    if locked_pairs:
        det_cids = (
            np.asarray([_norm_cid(c) for c in det_catalog_ids], dtype=object)
            if det_catalog_ids is not None
            else np.array([""] * n_d, dtype=object)
        )
        for cid, (_lx, _ly) in locked_pairs.items():
            gr = cid_to_row.get(_norm_cid(cid))
            if gr is None or gr in used_g:
                continue
            nc = _norm_cid(cid)
            gx_g = float(gx[gr]) if gr < len(gx) else float("nan")
            gy_g = float(gy[gr]) if gr < len(gy) else float("nan")
            best_i = -1
            best_d = float("inf")
            # Preference: born-owned within lock_tol_px. Keep (still lock) when
            # lock_tol < d <= identity_fail. Reject only past identity_fail_px (D4).
            for i in range(n_d):
                if i in used_det:
                    continue
                dc = _norm_cid(det_cids[i]) if i < len(det_cids) else ""
                if dc and dc == nc:
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
            if best_i < 0:
                for i in range(n_d):
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
    for d, i, j in pairs:
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
    det_pass: np.ndarray | None,
    *,
    target_depth_g: float,
    edge_margin_px: float,
    sat_limit_adu: float | None,
    fwhm_px: float,
    wpx: int,
    h: int,
    forced_results: dict[int, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """One row per on-chip Gaia with source_state (does not alter MS detection table)."""
    forced_results = forced_results or {}
    rows: list[dict[str, Any]] = []
    gx = pd.to_numeric(gaia_on_chip.get("x_gaia"), errors="coerce").to_numpy(dtype=np.float64)
    gy = pd.to_numeric(gaia_on_chip.get("y_gaia"), errors="coerce").to_numpy(dtype=np.float64)
    gm = pd.to_numeric(gaia_on_chip.get("g_mag", gaia_on_chip.get("mag")), errors="coerce").to_numpy(
        dtype=np.float64
    )
    for j in range(len(gaia_on_chip)):
        cid = _norm_cid(gaia_on_chip.iloc[j].get("catalog_id"))
        xg, yg = float(gx[j]), float(gy[j])
        gmag = float(gm[j]) if math.isfinite(gm[j]) else float("nan")
        state = SOURCE_TOO_FAINT
        reason = ""
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
            p = int(det_pass[owner]) if det_pass is not None and owner < len(det_pass) else 1
            state = SOURCE_DETECTED_P2 if p == 2 else SOURCE_DETECTED_P1
        elif j in forced_results:
            fr = forced_results[j]
            if fr.get("accepted"):
                state = SOURCE_FORCED_SEED
            else:
                state = SOURCE_SEED_REJECTED
                reason = str(fr.get("reason", ""))
        elif math.isfinite(gmag) and gmag <= float(target_depth_g):
            state = SOURCE_SEED_REJECTED
            reason = "no_owner_no_seed"
        row = {
            "catalog_id": cid,
            "x_gaia": xg,
            "y_gaia": yg,
            "g_mag": gmag,
            "source_state": state,
            "seed_reject_reason": reason,
            "ambiguous_owner": False,
        }
        if j in forced_results:
            for k in ("centroid_px", "snr", "local_sigma", "flux"):
                if k in forced_results[j]:
                    row[f"seed_{k}"] = forced_results[j][k]
        rows.append(row)
    return pd.DataFrame(rows)


def verify_ms_identity(
    baseline_pairs: dict[str, tuple[float, float]],
    result_pairs: dict[str, tuple[float, float]],
    *,
    tol_px: float = 0.01,
) -> tuple[bool, str]:
    """INV-MS-IDENTITY-01: every baseline catalog_id maps to same detection xy."""
    missing = []
    remapped = []
    for cid, (bx, by) in baseline_pairs.items():
        if cid not in result_pairs:
            missing.append(cid)
            continue
        rx, ry = result_pairs[cid]
        if math.hypot(bx - rx, by - ry) > tol_px:
            remapped.append(cid)
    if missing or remapped:
        return False, f"missing={len(missing)} remapped={len(remapped)}"
    return True, f"ok n={len(baseline_pairs)}"


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


def write_gaia_census_and_verify(
    census: pd.DataFrame,
    *,
    n_on_chip: int,
    census_path: Any,
) -> dict[str, Any]:
    """Always write census CSV, then FAIL-loud INV-MS-CENSUS-01.

    Empty census is still written; mismatch vs on-chip count raises.
    """
    from pathlib import Path

    from invariants_runtime import inv_check

    path = Path(census_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = census if census is not None else pd.DataFrame()
    try:
        from pipeline import _vyvar_df_to_csv  # noqa: PLC0415

        _vyvar_df_to_csv(out, path)
    except Exception:  # noqa: BLE001
        out.to_csv(path, index=False)
    ok, det = verify_gaia_census_complete(out, int(n_on_chip))
    meta: dict[str, Any] = {"invariants": []}
    inv_check(meta, "INV-MS-CENSUS-01", ok, policy="FAIL", detail=det)
    return {"ok": True, "detail": det, "path": str(path), "n": int(len(out)), "invariants": meta.get("invariants") or []}


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
            cid = _norm_cid(raw)
            if cid:
                present.add(cid)
    if "catalog_id" in out.columns:
        cid_col = out["catalog_id"].map(_norm_cid)
        meta["n_dao_only_preserved"] = int((cid_col.eq("")).sum())

    gdf = gaia_on_chip.copy()
    gx = pd.to_numeric(gdf.get("x_gaia"), errors="coerce").to_numpy(dtype=np.float64)
    gy = pd.to_numeric(gdf.get("y_gaia"), errors="coerce").to_numpy(dtype=np.float64)
    gm = pd.to_numeric(
        gdf.get("g_mag", gdf.get("mag")),
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    depth = float(membership_depth_g)
    sat_cids: set[str] = set()
    if "catalog_id" in out.columns:
        _sat = pd.Series(False, index=out.index)
        if "likely_saturated" in out.columns:
            _sat |= out["likely_saturated"].fillna(False).astype(bool)
        if "is_saturated" in out.columns:
            _sat |= out["is_saturated"].fillna(False).astype(bool)
        if "zone" in out.columns:
            _sat |= out["zone"].astype(str).str.strip().str.lower().isin(["saturated", "nonlinear"])
        for _cid in out.loc[_sat, "catalog_id"].tolist():
            _c = _norm_cid(_cid)
            if _c:
                sat_cids.add(_c)
    new_rows: list[dict[str, Any]] = []

    for j in range(len(gdf)):
        cid = _norm_cid(gdf.iloc[j].get("catalog_id"))
        if not cid or cid in present:
            continue
        xg, yg = float(gx[j]), float(gy[j])
        if not (math.isfinite(xg) and math.isfinite(yg)):
            continue
        if xg < 0 or yg < 0 or xg >= float(wpx) or yg >= float(h):
            continue
        gmag = float(gm[j]) if j < len(gm) and math.isfinite(gm[j]) else float("nan")
        in_depth = math.isfinite(gmag) and gmag <= depth
        in_sat = cid in sat_cids
        if not in_depth and not in_sat:
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
        new_rows.append(row)
        present.add(cid)

    if new_rows:
        out = pd.concat([out, pd.DataFrame(new_rows)], ignore_index=True)
        meta["n_catalog_rows_added"] = int(len(new_rows))
    meta["n_rows_out"] = int(len(out))
    return out, meta


def verify_ms_expand_guard(
    expand_meta: dict[str, Any],
    *,
    census_path: Any,
    cert_path: Any,
) -> tuple[bool, str]:
    """INV-MS-EXPAND-01: catalog expand ran, reported rows_added, census+cert exist."""
    from pathlib import Path

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
    cens_p = Path(census_path)
    cert_p = Path(cert_path)
    if not cens_p.is_file():
        return False, f"census missing: {cens_p}"
    if not cert_p.is_file():
        return False, f"certificate missing: {cert_p}"
    return True, (
        f"ok n_in={n_in} n_out={n_out} added={int(expand_meta.get('n_catalog_rows_added', 0))}"
    )


def enrich_masterstar_gaia_complete(
    df_ms: pd.DataFrame,
    *,
    data0: np.ndarray,
    gaia_on_chip: pd.DataFrame,
    cfg: Any,
    wpx: int,
    h: int,
    fwhm_px: float,
    target_depth_g: float,
    sat_limit_adu: float | None = None,
    locked_pairs: dict[str, tuple[float, float]] | None = None,
    identity_lock_only: bool = False,
    catalog_derived_membership: bool = False,
    tolerance_overrides: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Apply lock-leftover assignment, FORCED_SEED admission, BLENDED + census (Parts B-D).

    ``identity_lock_only=True`` keeps existing MS row identities (no leftover
    catalog_id rewrite). Unowned on-chip Gaia at depth may still append FORCED_SEED.

    ``catalog_derived_membership=True`` (DAO-GAIA-ERA-01 M1): membership rows were
    pre-expanded from the catalog; do not rewrite catalog_id via leftover promotion and
    do not append duplicate FORCED_SEED membership rows (census seed probe still runs).
    """
    from config import AppConfig

    _cfg = cfg if isinstance(cfg, AppConfig) else AppConfig()
    meta: dict[str, Any] = {
        "n_forced_seed": 0,
        "n_leftover_promotions": 0,
        "n_seed_rejected": 0,
        "n_lock_geometry_reject": 0,
    }
    out = df_ms.copy() if df_ms is not None else pd.DataFrame()

    if gaia_on_chip is None or gaia_on_chip.empty:
        return out, pd.DataFrame(), meta

    gdf = gaia_on_chip.copy()
    if "x_gaia" not in gdf.columns:
        gdf["x_gaia"] = pd.to_numeric(gdf.get("x"), errors="coerce")
        gdf["y_gaia"] = pd.to_numeric(gdf.get("y"), errors="coerce")
    if "g_mag" not in gdf.columns and "mag" in gdf.columns:
        gdf["g_mag"] = pd.to_numeric(gdf["mag"], errors="coerce")

    n_g = len(gdf)
    gaia_owner = np.full(n_g, -1, dtype=np.int64)

    det_x = pd.to_numeric(out.get("x"), errors="coerce").to_numpy(dtype=np.float64)
    det_y = pd.to_numeric(out.get("y"), errors="coerce").to_numpy(dtype=np.float64)
    n_d = len(out)

    if locked_pairs is None:
        locked_pairs = {}
        if "catalog_id" in out.columns:
            cids = out["catalog_id"].map(_norm_cid)
            _pass_col = (
                pd.to_numeric(out.get("vy_dao_pass"), errors="coerce").fillna(1).to_numpy(dtype=np.int16)
                if "vy_dao_pass" in out.columns
                else np.ones(n_d, dtype=np.int16)
            )
            for i in range(n_d):
                cid = _norm_cid(cids.iloc[i])
                if cid and math.isfinite(det_x[i]) and math.isfinite(det_y[i]):
                    # Born-owned pass2 wins over pass1 when both carry the same catalog_id.
                    if cid not in locked_pairs or int(_pass_col[i]) == 2:
                        locked_pairs[cid] = (float(det_x[i]), float(det_y[i]))

    if catalog_derived_membership:
        identity_lock_only = True
    leftover_r = (
        0.0
        if identity_lock_only
        else float(
            (tolerance_overrides or {}).get(
                "lock_leftover_radius_px",
                getattr(_cfg, "masterstar_lock_leftover_radius_px", 3.0),
            )
        )
    )
    lock_tol = float(
        (tolerance_overrides or {}).get(
            "lock_pair_tol_px",
            getattr(_cfg, "masterstar_lock_pair_tol_px", 3.0),
        )
    )
    # D4: reject radius is the identity-gate fail threshold (3 x FWHM_dao), not lock_tol.
    _fwhm_lock = float(fwhm_px) if math.isfinite(float(fwhm_px)) and float(fwhm_px) > 0 else 3.5
    identity_fail_px = float(
        (tolerance_overrides or {}).get("identity_fail_px", 3.0 * _fwhm_lock)
    )
    det_to_g, gaia_owner, vy_modes, _geom_rej = lock_existing_and_leftover_assign(
        det_x,
        det_y,
        gdf,
        locked_pairs=locked_pairs,
        leftover_radius_px=leftover_r,
        lock_tol_px=lock_tol,
        identity_fail_px=identity_fail_px,
        det_catalog_ids=(
            out["catalog_id"].map(_norm_cid).to_numpy(dtype=object)
            if "catalog_id" in out.columns
            else None
        ),
    )
    meta["n_lock_geometry_reject"] = int(len({int(i) for i in _geom_rej}))
    meta["identity_fail_px"] = float(identity_fail_px)
    meta["lock_pair_tol_px_effective"] = float(lock_tol)
    if _geom_rej:
        from wcs_invertibility import clear_row_match_identity, det_fallback_name

        for i in _geom_rej:
            if int(det_to_g[i]) >= 0:
                continue
            idx = out.index[i]
            clear_row_match_identity(out, idx, det_name=det_fallback_name(i + 1))

    if "vy_match_mode" not in out.columns:
        out["vy_match_mode"] = ""
    for i in range(n_d):
        if vy_modes[i]:
            idx = out.index[i]
            cur = str(out.loc[idx, "vy_match_mode"] or "").strip()
            # Expand rows stay catalog_membership; lock must not relabel them (INV-SOURCE-STATE-01).
            if cur != "catalog_membership":
                out.loc[idx, "vy_match_mode"] = str(vy_modes[i])
        gr = int(det_to_g[i])
        if (not identity_lock_only) and gr >= 0 and gr < n_g:
            cid = _norm_cid(gdf.iloc[gr].get("catalog_id"))
            if cid:
                out.loc[out.index[i], "catalog_id"] = cid
                out.loc[out.index[i], "source_type"] = "GAIA_MATCHED"
                if str(vy_modes[i]) == "leftover_promotion":
                    meta["n_leftover_promotions"] = int(meta["n_leftover_promotions"]) + 1

    seed_params = ForcedSeedAcceptParams(
        centroid_max_px=float(
            (tolerance_overrides or {}).get(
                "forced_seed_centroid_max_px",
                getattr(_cfg, "masterstar_forced_seed_centroid_max_px", 2.0),
            )
        ),
        snr_min=float(getattr(_cfg, "masterstar_forced_seed_snr_min", 4.0)),
    )
    edge_m = float(getattr(_cfg, "masterstar_gaia_census_edge_margin_px", 10.0))
    forced_results: dict[int, dict[str, Any]] = {}
    new_rows: list[dict[str, Any]] = []

    det_pass = (
        pd.to_numeric(out.get("vy_dao_pass"), errors="coerce").fillna(1).to_numpy(dtype=np.int16)
        if "vy_dao_pass" in out.columns
        else np.ones(n_d, dtype=np.int16)
    )

    for j in range(n_g):
        if int(gaia_owner[j]) >= 0:
            continue
        gmag = float(gdf.iloc[j].get("g_mag", np.nan))
        if not (math.isfinite(gmag) and gmag <= float(target_depth_g)):
            continue
        xg = float(gdf.iloc[j]["x_gaia"])
        yg = float(gdf.iloc[j]["y_gaia"])
        if (
            xg < edge_m
            or yg < edge_m
            or xg >= float(wpx) - edge_m
            or yg >= float(h) - edge_m
        ):
            continue
        meas = forced_seed_measure_at_position(data0, xg, yg, fwhm_px=fwhm_px, params=seed_params)
        ok, reason = forced_seed_accept(meas, params=seed_params)
        fr: dict[str, Any] = {
            "accepted": ok,
            "reason": reason,
            "centroid_px": meas.get("centroid_px"),
            "snr": meas.get("snr"),
            "local_sigma": meas.get("local_sigma"),
            "flux": meas.get("flux"),
        }
        forced_results[j] = fr
        if ok and not catalog_derived_membership:
            cid = _norm_cid(gdf.iloc[j].get("catalog_id"))
            new_rows.append(
                {
                    "name": cid or f"SEED_{j:05d}",
                    "catalog_id": cid,
                    "catalog": "Gaia",
                    "source_type": "GAIA_MATCHED",
                    "source_state": SOURCE_FORCED_SEED,
                    "forced_photometry": True,
                    "x": xg,
                    "y": yg,
                    "ra_deg": gdf.iloc[j].get("ra_deg"),
                    "dec_deg": gdf.iloc[j].get("dec_deg"),
                    "mag": gmag,
                    "flux": meas.get("flux"),
                    "seed_centroid_px": meas.get("centroid_px"),
                    "seed_snr": meas.get("snr"),
                    "seed_local_sigma": meas.get("local_sigma"),
                    "vy_match_mode": "forced_seed",
                    "is_usable": False,
                    "photometry_ok": False,
                }
            )
            meta["n_forced_seed"] = int(meta["n_forced_seed"]) + 1
        elif ok and catalog_derived_membership:
            cid = _norm_cid(gdf.iloc[j].get("catalog_id"))
            if cid and "catalog_id" in out.columns:
                _cid_ms = out["catalog_id"].map(_norm_cid)
                hit = _cid_ms.eq(cid)
                if hit.any():
                    idx = out.index[hit][0]
                    out.loc[idx, "source_state"] = SOURCE_FORCED_SEED
                    out.loc[idx, "forced_photometry"] = True
                    out.loc[idx, "flux"] = meas.get("flux")
                    out.loc[idx, "seed_centroid_px"] = meas.get("centroid_px")
                    out.loc[idx, "seed_snr"] = meas.get("snr")
                    out.loc[idx, "seed_local_sigma"] = meas.get("local_sigma")
                    out.loc[idx, "vy_match_mode"] = "forced_seed"
            meta["n_forced_seed"] = int(meta["n_forced_seed"]) + 1
        else:
            meta["n_seed_rejected"] = int(meta["n_seed_rejected"]) + 1

    # When identity_lock_only=True (overlay path), FORCED_SEED rows must NOT be appended
    # to the masterstars table because extra rows change Phase-1 per-target comp selection
    # even when excluded by is_usable=False. They are recorded in the census only.
    if new_rows and not identity_lock_only:
        out = pd.concat([out, pd.DataFrame(new_rows)], ignore_index=True)

    gdf = annotate_blended_groups(gdf, gaia_owner, fwhm_px=fwhm_px)
    census = build_gaia_census_rows(
        gdf,
        gaia_owner,
        det_pass,
        target_depth_g=float(target_depth_g),
        edge_margin_px=edge_m,
        sat_limit_adu=sat_limit_adu,
        fwhm_px=fwhm_px,
        wpx=int(wpx),
        h=int(h),
        forced_results=forced_results,
    )
    for j in range(n_g):
        bg = str(gdf.iloc[j].get("blend_group_id", "") or "").strip()
        if bg:
            cid = _norm_cid(gdf.iloc[j].get("catalog_id"))
            if cid and int(gaia_owner[j]) < 0:
                census.loc[census["catalog_id"] == cid, "source_state"] = SOURCE_BLENDED
                census.loc[census["catalog_id"] == cid, "blend_group_id"] = bg
                census.loc[census["catalog_id"] == cid, "blend_flux_ratio"] = gdf.iloc[j].get(
                    "blend_flux_ratio"
                )

    if "source_state" not in out.columns:
        out["source_state"] = ""
    cid_out = out.get("catalog_id", pd.Series([""] * len(out))).map(_norm_cid)
    census_st: dict[str, str] = {}
    if len(census):
        for ci in range(len(census)):
            cc = _norm_cid(census.iloc[ci].get("catalog_id"))
            if cc:
                census_st[cc] = str(census.iloc[ci].get("source_state", "")).strip()
    peak_dao = pd.to_numeric(out.get("peak_dao"), errors="coerce")
    for i in range(len(out)):
        cid = _norm_cid(cid_out.iloc[i])
        _fp_raw = out.loc[out.index[i], "forced_photometry"] if "forced_photometry" in out.columns else False
        _fp_ok = False
        if _fp_raw is not None and not (isinstance(_fp_raw, float) and not math.isfinite(_fp_raw)):
            _fp_ok = str(_fp_raw).strip().lower() in {"true", "1", "yes"}
        if _fp_ok:
            out.loc[out.index[i], "source_state"] = SOURCE_FORCED_SEED
            continue
        if not cid:
            out.loc[out.index[i], "source_state"] = "DAO_ONLY"
            continue
        st = out.loc[out.index[i], "source_state"] if "source_state" in out.columns else ""
        if str(st).strip() in (SOURCE_DETECTED_P1, SOURCE_DETECTED_P2):
            # Pre-set DETECTED_* is only kept when this row itself is a DAO detection.
            pass
        elif str(st).strip():
            continue
        peak_v = peak_dao.iloc[i] if i < len(peak_dao) else float("nan")
        vp_raw = out.loc[out.index[i], "vy_dao_pass"] if "vy_dao_pass" in out.columns else float("nan")
        is_det, vp = _row_is_dao_detected(peak_v, vp_raw)
        if is_det:
            out.loc[out.index[i], "source_state"] = (
                SOURCE_DETECTED_P2 if vp == 2 else SOURCE_DETECTED_P1
            )
            continue
        mode = str(out.loc[out.index[i], "vy_match_mode"] or "").strip() if "vy_match_mode" in out.columns else ""
        if mode == "catalog_membership":
            out.loc[out.index[i], "source_state"] = SOURCE_CATALOG_MEMBERSHIP
            continue
        cst = str(census_st.get(cid, "") or "").strip()
        if cst in _NONDET_CENSUS_STATES:
            out.loc[out.index[i], "source_state"] = cst
        else:
            out.loc[out.index[i], "source_state"] = SOURCE_CATALOG_ONLY

    meta["n_on_chip_gaia"] = n_g
    meta["census_by_state"] = (
        census["source_state"].value_counts().to_dict() if len(census) else {}
    )
    if "ambiguous_owner" in out.columns and len(census):
        amb_map: dict[str, bool] = {}
        for i in range(len(out)):
            cid = _norm_cid(out.iloc[i].get("catalog_id"))
            if not cid:
                continue
            raw_amb = out.iloc[i].get("ambiguous_owner")
            amb_map[cid] = (
                bool(raw_amb)
                if raw_amb is not None and not (isinstance(raw_amb, float) and not math.isfinite(raw_amb))
                else False
            )
        for ci in range(len(census)):
            cid = _norm_cid(census.iloc[ci].get("catalog_id"))
            if amb_map.get(cid):
                census.loc[census.index[ci], "ambiguous_owner"] = True
    meta["identity_lock_only"] = bool(identity_lock_only)
    meta["catalog_derived_membership"] = bool(catalog_derived_membership)
    return out, census, meta


def overlay_gaia_complete_on_existing_masterstar(
    platesolve_dir: Any,
    *,
    cfg: Any,
    masterstar_fits: Any,
    ms_csv: Any,
    identity_lock_only: bool = True,
    target_depth_g: float | None = None,
    sat_limit_adu: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Recompute source_state + additive FORCED_SEED on an existing MS table.

    Does not redetect. Writes ``gaia_source_state_census.csv`` and runs INV-MS-CENSUS-01.
    """
    import warnings
    from pathlib import Path

    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.wcs.wcs import FITSFixedWarning

    ps = Path(platesolve_dir)
    ms_path = Path(ms_csv)
    fits_path = Path(masterstar_fits)
    cone_path = ps / "field_catalog_cone.csv"
    if not ms_path.is_file():
        raise FileNotFoundError(str(ms_path))
    if not fits_path.is_file():
        raise FileNotFoundError(str(fits_path))
    if not cone_path.is_file():
        raise FileNotFoundError(str(cone_path))

    df_ms = pd.read_csv(ms_path, low_memory=False, dtype={"catalog_id": str, "name": str})
    cone = pd.read_csv(cone_path, low_memory=False, dtype={"catalog_id": str})
    with fits.open(fits_path, memmap=False) as hdul:
        hdr = hdul[0].header
        raw = np.asarray(hdul[0].data, dtype=np.float32)
        fwhm = float(hdr.get("VY_FWHM") or 5.3)
        wpx = int(hdr.get("NAXIS1") or raw.shape[1])
        h = int(hdr.get("NAXIS2") or raw.shape[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        wcs = WCS(hdr)
    ra = pd.to_numeric(cone["ra_deg"], errors="coerce").to_numpy(dtype=np.float64)
    de = pd.to_numeric(cone["dec_deg"], errors="coerce").to_numpy(dtype=np.float64)
    ok = np.isfinite(ra) & np.isfinite(de)
    gx, gy = wcs.world_to_pixel_values(ra[ok], de[ok])
    chip = gaia_on_chip_from_cone(cone, gx=gx, gy=gy, ok_mask=ok, wpx=wpx, h=h)
    _, med, _ = plain_mean_med_std(raw, sigma=3.0, maxiters=3)
    data0 = np.nan_to_num((raw - med).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    depth = float(target_depth_g) if target_depth_g is not None else 17.5
    n_base = int(len(df_ms))
    locked: dict[str, tuple[float, float]] = {}
    cids = df_ms["catalog_id"].map(_norm_cid) if "catalog_id" in df_ms.columns else pd.Series([""] * n_base)
    xs = pd.to_numeric(df_ms.get("x"), errors="coerce")
    ys = pd.to_numeric(df_ms.get("y"), errors="coerce")
    for i in range(n_base):
        cid = _norm_cid(cids.iloc[i])
        if cid and math.isfinite(float(xs.iloc[i])) and math.isfinite(float(ys.iloc[i])):
            locked[cid] = (float(xs.iloc[i]), float(ys.iloc[i]))
    out, census, meta = enrich_masterstar_gaia_complete(
        df_ms,
        data0=data0,
        gaia_on_chip=chip,
        cfg=cfg,
        wpx=wpx,
        h=h,
        fwhm_px=fwhm,
        target_depth_g=depth,
        sat_limit_adu=sat_limit_adu,
        locked_pairs=locked,
        identity_lock_only=bool(identity_lock_only),
    )
    inv = write_gaia_census_and_verify(
        census,
        n_on_chip=len(chip),
        census_path=ps / "gaia_source_state_census.csv",
    )
    meta["census_inv"] = inv
    meta["n_base_rows"] = n_base
    meta["n_rows_after"] = int(len(out))
    return out, census, meta
