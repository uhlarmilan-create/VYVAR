"""Depth-aware, detection-INDEPENDENT crowding index (PARALLEL diagnostic).

This module is standalone and side-effect-free (``compute_crowding_index`` only reads
artifacts and returns scalars). As of 2026-05 it is consumed by the optional, gated
signal-based comp classifier in ``run_phase0_and_phase1`` (AppConfig
``crowding_classifier_enabled``, default OFF). It still does NOT modify field_density /
classify_field_density / apply_density_overrides, and it does NOT touch DAO detection or
DAO<->Gaia matching; when the flag is OFF the legacy stars/Mpx path is used unchanged.

It reads only EXISTING artifacts written by a prior run:
    - field_catalog_cone.csv        (full local-Gaia footprint, mag-cut at DB max)
    - masterstars_full_match.csv    (DAO detections matched to Gaia: flux, catalog_mag)
    - MASTERSTAR.fits               (WCS + core FWHM: VY_FWHM_GAUSS preferred)
    - photometry/active_targets.csv (variable-target worklist)
    - EQUIPMENTS gain/read-noise    (from the project DB)

Outputs (parallel, never overwrites field_density.json):
    - crowding_index.json     (Parts A/B/C scalars + metadata)
    - crowding_targets.csv    (Part D per-target PSF-deblend worklist)

Method notes / caveats:
    * Frame limiting magnitude (Part A, empirical) uses the masterstars `flux`
      column, which is the DAO *instrumental* flux. We calibrate an instrumental
      zero point ZP = median(catalog_mag + 2.5*log10(flux)) so flux<->mag is
      self-consistent, then build a Howell (1989) SNR for each matched star and
      locate the catalog_mag where the binned median SNR crosses 5. Because DAO
      flux is not a calibrated aperture sum, treat frame_limit_mag as a *relative*
      depth proxy that is consistent across drafts (identical method everywhere).
    * Plate scale is taken from the MASTERSTAR WCS (proj_plane_pixel_scales), NOT
      from VY_PLTS (which is missing on 311 and was stale on 360/361).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from masterstar_context import header_core_fwhm_px
from photometry_core import _photometric_error

SNR_LIMIT = 5.0
APERTURE_FWHM_FACTOR = 1.9  # AppConfig default; aperture radius = factor * FWHM
GAIN_FALLBACK = 3.17        # eq 1 (QHY294MM) — used when a draft has no equipment row
RN_FALLBACK = 7.6


# ---------------------------------------------------------------- metadata I/O
def _load_wcs_meta(ms_fits: Path) -> dict[str, Any]:
    """WCS, frame size, FWHM(px), plate scale(arcsec/px) from MASTERSTAR.fits."""
    import warnings

    from astropy.io import fits
    from astropy.wcs import WCS, FITSFixedWarning
    from astropy.wcs.utils import proj_plane_pixel_scales

    with fits.open(ms_fits, memmap=False) as hdul:
        hdr = hdul[0].header
        naxis1 = int(hdr.get("NAXIS1", 0) or 0)
        naxis2 = int(hdr.get("NAXIS2", 0) or 0)
        fwhm_px = header_core_fwhm_px(hdr) or 0.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            wcs = WCS(hdr)
        scales_deg = proj_plane_pixel_scales(wcs)
        plate_scale_arcsec = float(np.mean(scales_deg)) * 3600.0
    if not (math.isfinite(fwhm_px) and fwhm_px > 0):
        fwhm_px = 3.5
    return {
        "wcs": wcs,
        "naxis1": naxis1,
        "naxis2": naxis2,
        "fwhm_px": fwhm_px,
        "plate_scale_arcsec": plate_scale_arcsec,
    }


def _gain_rn_for_draft(db: Any, draft_id: int) -> tuple[float, float, str]:
    """(gain e-/ADU, read_noise e-, source) from EQUIPMENTS via OBS_DRAFT."""
    try:
        row = db.conn.execute(
            "SELECT ID_EQUIPMENTS FROM OBS_DRAFT WHERE ID = ?;", (int(draft_id),)
        ).fetchone()
        eid = int(row["ID_EQUIPMENTS"]) if row and row["ID_EQUIPMENTS"] else None
        if eid:
            from param_resolver import resolve_gain, resolve_read_noise  # noqa: PLC0415

            g_res = resolve_gain(None, db=db, equipment_id=eid)
            rn_res = resolve_read_noise(None, db=db, equipment_id=eid)
            if g_res.ok and rn_res.ok:
                return float(g_res.value), float(rn_res.value), f"{g_res.source}_eq{eid}"
    except Exception:  # noqa: BLE001
        pass
    return GAIN_FALLBACK, RN_FALLBACK, "fallback_eq1"


# ---------------------------------------------------------------- SNR helpers
def _howell_snr(flux: float, sky_pp: float, area: float, gain: float, rn: float) -> float:
    rel = _photometric_error(float(flux), float(sky_pp), float(area), gain=float(gain), read_noise=float(rn))
    if rel is None or not math.isfinite(rel) or rel <= 0:
        return float("nan")
    return 1.0 / rel


def _interp_snr5_crossing(mag_bins: np.ndarray, snr_bins: np.ndarray) -> float:
    """Magnitude where median SNR crosses SNR_LIMIT (faint side), interp in log10(SNR)."""
    ok = np.isfinite(mag_bins) & np.isfinite(snr_bins) & (snr_bins > 0)
    m = mag_bins[ok]
    s = snr_bins[ok]
    if m.size < 2:
        return float("nan")
    order = np.argsort(m)
    m, s = m[order], s[order]
    log_s = np.log10(s)
    target = math.log10(SNR_LIMIT)
    # walk faintward; find first bin pair bracketing the crossing (log_s goes target+ -> target-)
    for i in range(len(m) - 1):
        a, b = log_s[i], log_s[i + 1]
        if a >= target >= b and a != b:
            frac = (a - target) / (a - b)
            return float(m[i] + frac * (m[i + 1] - m[i]))
    if log_s[-1] >= target:
        return float(m[-1])  # never drops below 5 within sampled range
    if log_s[0] < target:
        return float(m[0])   # already below 5 at the bright end (degenerate)
    return float("nan")


def _analytic_snr5(fwhm_px: float, sky_pp: float, gain: float, rn: float, zero_point: float) -> float:
    """SNR=5 crossing from the same model as compute_snr_optimal_aperture_table (SNR it discards)."""
    fw = float(fwhm_px) if math.isfinite(fwhm_px) and fwhm_px > 0 else 3.5
    sky = max(0.0, float(sky_pp))
    g = float(gain) if gain > 0 else 1.0
    rnv = float(rn) if rn >= 0 else 10.0
    sigma = fw / 2.355
    r_values = np.arange(0.8 * fw, 2.5 * fw, 0.05)
    if r_values.size == 0:
        r_values = np.array([max(0.5, 0.8 * fw)])
    mags = np.arange(7.0, 22.01, 0.1)
    snr_curve = []
    for mag in mags:
        flux_total = 10.0 ** ((zero_point - mag) / 2.5)
        best = -1.0
        for r in r_values:
            enclosed = flux_total * (1.0 - math.exp(-(r ** 2) / (2.0 * sigma ** 2)))
            area = math.pi * r ** 2
            noise = math.sqrt(max(enclosed / g + area * sky / g + area * (rnv / g) ** 2, 1e-12))
            snr = (enclosed / g) / noise if noise > 0 else 0.0
            best = max(best, snr)
        snr_curve.append(best)
    return _interp_snr5_crossing(mags, np.asarray(snr_curve))


def _load_lc_star_table(platesolve_dir: Path) -> pd.DataFrame:
    """Union LC stars: active targets + comp pool + per-target comps (deduped by catalog_id)."""
    phot = Path(platesolve_dir) / "photometry"
    chunks: list[pd.DataFrame] = []

    def _read(path: Path) -> pd.DataFrame | None:
        if not path.is_file():
            return None
        df = pd.read_csv(path, low_memory=False, dtype={"catalog_id": str})
        if "catalog_id" not in df.columns or df.empty:
            return None
        df = df.copy()
        df["catalog_id"] = df["catalog_id"].fillna("").astype(str).str.strip()
        df = df[df["catalog_id"].ne("") & ~df["catalog_id"].str.lower().isin(["nan", "none"])]
        keep = [c for c in ("catalog_id", "name", "vsx_type", "ra_deg", "dec_deg", "mag") if c in df.columns]
        return df[keep] if not df.empty else None

    for p in (phot / "active_targets.csv",):
        part = _read(p)
        if part is not None:
            chunks.append(part)
    for comp_p in (phot / "comparison_stars.csv", Path(platesolve_dir) / "comparison_stars.csv"):
        part = _read(comp_p)
        if part is not None:
            chunks.append(part)
            break
    part = _read(phot / "comparison_stars_per_target.csv")
    if part is not None:
        chunks.append(part)

    if not chunks:
        return pd.DataFrame()
    out = pd.concat(chunks, ignore_index=True)
    return out.drop_duplicates(subset=["catalog_id"], keep="first")


def _build_blend_targets_df(
    stars_df: pd.DataFrame,
    *,
    wcs: Any,
    cone_f: pd.DataFrame,
    fwhm_px: float,
    sky_pp: float,
    area: float,
    gain: float,
    rn: float,
    zp: float,
    frame_limit_mag: float,
) -> pd.DataFrame:
    """Per-star blend metrics (``is_blended``, ``nn_dist_fwhm``) for adaptive PSF routing."""
    from scipy.spatial import cKDTree

    if stars_df.empty:
        return pd.DataFrame()

    cone_pts = cone_f[["x", "y"]].to_numpy(dtype=float)
    cone_mag = cone_f["mag_g"].to_numpy(dtype=float)
    cone_ids = cone_f["catalog_id"].to_numpy()
    ctree = cKDTree(cone_pts) if len(cone_pts) >= 2 else None
    fl = frame_limit_mag
    rows: list[dict[str, Any]] = []
    for _, t in stars_df.iterrows():
        tra = pd.to_numeric(t.get("ra_deg"), errors="coerce")
        tde = pd.to_numeric(t.get("dec_deg"), errors="coerce")
        tmag = pd.to_numeric(t.get("mag"), errors="coerce")
        if not (np.isfinite(tra) and np.isfinite(tde)):
            continue
        tx, ty = wcs.all_world2pix([[float(tra), float(tde)]], 0)[0]
        rec: dict[str, Any] = {
            "name": t.get("name"),
            "vsx_type": t.get("vsx_type"),
            "catalog_id": t.get("catalog_id"),
            "mag": float(tmag) if np.isfinite(tmag) else float("nan"),
        }
        if ctree is not None and len(cone_pts) >= 2:
            dists, idxs = ctree.query([tx, ty], k=min(50, len(cone_pts)))
            dists = np.atleast_1d(dists)
            idxs = np.atleast_1d(idxs)
            nn_i = None
            tid = str(t.get("catalog_id") or "").strip()
            for d_, j_ in zip(dists, idxs, strict=True):
                if str(cone_ids[j_]).strip() == tid or d_ < 1e-6:
                    continue
                nn_i = int(j_)
                nn_d = float(d_)
                break
            if nn_i is not None:
                rec["nn_dist_fwhm"] = round(nn_d / fwhm_px, 3)
                rec["nn_catalog_id"] = cone_ids[nn_i]
                rec["nn_mag"] = round(float(cone_mag[nn_i]), 3)
                rec["delta_mag_nn"] = (
                    round(float(cone_mag[nn_i] - tmag), 3) if np.isfinite(tmag) else float("nan")
                )
                rec["is_blended"] = bool(nn_d <= 1.5 * fwhm_px)
            rec["n_neigh_2fwhm"] = int(len(ctree.query_ball_point([tx, ty], r=2.0 * fwhm_px)) - 1)
            rec["n_neigh_3fwhm"] = int(len(ctree.query_ball_point([tx, ty], r=3.0 * fwhm_px)) - 1)
        if np.isfinite(tmag) and math.isfinite(zp):
            tflux = 10.0 ** ((zp - float(tmag)) / 2.5)
            tsnr = _howell_snr(tflux, sky_pp, area, gain, rn)
        else:
            tsnr = float("nan")
        rec["snr"] = round(tsnr, 2) if math.isfinite(tsnr) else float("nan")
        rec["is_drowning"] = bool(
            (math.isfinite(tsnr) and tsnr < 10.0)
            or (np.isfinite(tmag) and math.isfinite(fl) and float(tmag) > fl - 1.0)
        )
        rows.append(rec)
    return pd.DataFrame(rows)


def ensure_crowding_targets_for_lc(
    draft_dir: Path | str,
    setup: str,
    db: Any,
    draft_id: int,
    *,
    gaia_db_max_g: float,
    force: bool = False,
) -> Path | None:
    """Write ``crowding_targets.csv`` for the full LC star set (adaptive blend map input).

    Gated caller: invoke only when ``psf_adaptive_enabled`` is true. Does not touch
    ``field_density.json`` or other legacy crowding artifacts.
    """
    draft_dir = Path(draft_dir)
    ps = draft_dir / "platesolve" / setup
    out_csv = ps / "crowding_targets.csv"
    if out_csv.is_file() and not force:
        return out_csv
    _res, targets_df = compute_crowding_index(
        draft_dir,
        setup,
        db,
        draft_id,
        gaia_db_max_g=gaia_db_max_g,
        lc_star_set=True,
    )
    if targets_df.empty:
        return None
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    targets_df.to_csv(out_csv, index=False)
    return out_csv


# ---------------------------------------------------------------- main compute
def compute_crowding_index(
    draft_dir: Path | str,
    setup: str,
    db: Any,
    draft_id: int,
    *,
    gaia_db_max_g: float,
    lc_star_set: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame]:
    draft_dir = Path(draft_dir)
    ps = draft_dir / "platesolve" / setup
    phot = ps / "photometry"
    ms_fits = ps / "MASTERSTAR.fits"

    meta = _load_wcs_meta(ms_fits)
    wcs = meta["wcs"]
    naxis1, naxis2 = meta["naxis1"], meta["naxis2"]
    fwhm_px = meta["fwhm_px"]
    plate_scale = meta["plate_scale_arcsec"]
    fwhm_arcsec = fwhm_px * plate_scale
    gain, rn, gain_src = _gain_rn_for_draft(db, draft_id)

    # ----- load matched (masterstars) + cone catalog
    mt = pd.read_csv(ps / "masterstars_full_match.csv", low_memory=False, dtype={"catalog_id": str})
    mt["catalog_id"] = mt["catalog_id"].fillna("").astype(str).str.strip()
    matched = mt[mt["catalog_id"].ne("") & ~mt["catalog_id"].str.lower().isin(["nan", "none"])].copy()
    matched["mag_cat"] = pd.to_numeric(matched["mag"], errors="coerce")
    matched["flux_v"] = pd.to_numeric(matched["flux"], errors="coerce")
    sky_pp = float(pd.to_numeric(mt.get("noise_floor_adu"), errors="coerce").median())
    if not math.isfinite(sky_pp) or sky_pp < 0:
        sky_pp = 0.0

    cone = pd.read_csv(ps / "field_catalog_cone.csv", low_memory=False, dtype={"catalog_id": str})
    cone["catalog_id"] = cone["catalog_id"].fillna("").astype(str).str.strip()
    cone["mag_g"] = pd.to_numeric(cone["mag"], errors="coerce")
    cone = cone[np.isfinite(cone["mag_g"])].copy()
    # The cone is row-capped (ORDER BY g_mag LIMIT ~100k), so its faint edge is
    # set by field density, not by the DB. The cone is COMPLETE brightward of this
    # edge; the per-draft catalog availability limit is therefore the cone's max g.
    n_cone_rows = int(len(cone))
    cone_max_g = float(cone["mag_g"].max()) if n_cone_rows else float("nan")
    cone_is_row_capped = bool(n_cone_rows >= 99_000 and cone_max_g < gaia_db_max_g - 0.05)

    aperture_r = APERTURE_FWHM_FACTOR * fwhm_px
    area = math.pi * aperture_r ** 2

    # ----- PART A: instrumental zero point + empirical frame limit
    good = matched[(matched["flux_v"] > 0) & np.isfinite(matched["mag_cat"])].copy()
    zp = float("nan")
    if len(good) >= 5:
        zp = float(np.median(good["mag_cat"] + 2.5 * np.log10(good["flux_v"])))

    snr = np.array([
        _howell_snr(f, sky_pp, area, gain, rn) for f in good["flux_v"].to_numpy(dtype=float)
    ])
    good = good.assign(snr=snr)
    bin_edges = np.arange(8.0, 16.51, 0.5)
    centers, med_snr = [], []
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        sel = good[(good["mag_cat"] >= lo) & (good["mag_cat"] < hi)]
        s = sel["snr"].to_numpy(dtype=float)
        s = s[np.isfinite(s)]
        if s.size >= 3:
            centers.append((lo + hi) / 2.0)
            med_snr.append(float(np.median(s)))
    frame_limit_mag = _interp_snr5_crossing(np.asarray(centers), np.asarray(med_snr))

    analytic_snr5 = _analytic_snr5(fwhm_px, sky_pp, gain, rn, zp) if math.isfinite(zp) else float("nan")

    # Per-draft catalog availability = cone's own faint edge (row-cap induced),
    # capped at the DB max. This is the real ceiling on what we can verify.
    catalog_limit = float(min(gaia_db_max_g, cone_max_g)) if math.isfinite(cone_max_g) else float(gaia_db_max_g)
    fl = frame_limit_mag if math.isfinite(frame_limit_mag) else catalog_limit
    catalog_is_bottleneck = bool(math.isfinite(frame_limit_mag) and frame_limit_mag > catalog_limit)
    effective_limit = float(min(catalog_limit, fl))

    # ----- project cone to pixels (detection-independent footprint)
    ra = cone["ra_deg"].to_numpy(dtype=float)
    de = cone["dec_deg"].to_numpy(dtype=float)
    xpix, ypix = wcs.all_world2pix(np.column_stack([ra, de]), 0).T
    cone = cone.assign(x=xpix, y=ypix)
    in_frame = (xpix >= 0) & (xpix < naxis1) & (ypix >= 0) & (ypix < naxis2)
    cone_f = cone[in_frame].copy()

    # ----- PART B: detection-independent crowding down to effective_limit
    from scipy.spatial import cKDTree

    footprint_area_arcmin2 = (naxis1 * naxis2) * (plate_scale / 60.0) ** 2
    sub_b = cone_f[cone_f["mag_g"] <= effective_limit]
    n_gaia_eff = int(len(sub_b))
    gaia_density_per_arcmin2 = n_gaia_eff / footprint_area_arcmin2 if footprint_area_arcmin2 > 0 else float("nan")
    blend_frac_1fwhm = blend_frac_2fwhm = float("nan")
    if n_gaia_eff >= 2:
        pts = sub_b[["x", "y"]].to_numpy(dtype=float)
        tree = cKDTree(pts)
        for radius, key in ((fwhm_px, "1"), (2.0 * fwhm_px, "2")):
            counts = tree.query_ball_point(pts, r=radius, return_length=True)
            frac = float(np.mean(np.asarray(counts) > 1))  # >1 because self is included
            if key == "1":
                blend_frac_1fwhm = frac
            else:
                blend_frac_2fwhm = frac

    # ----- PART C: Gaia->DAO miss decomposition down to frame_limit_mag
    detected_ids = set(matched["catalog_id"])
    on_frame = cone_f[cone_f["mag_g"] <= fl].copy()
    n_on_frame = int(len(on_frame))
    n_below_depth = int((cone_f["mag_g"] > fl).sum())
    of_ids = on_frame["catalog_id"].to_numpy()
    is_det = np.array([cid in detected_ids for cid in of_ids])
    detected = int(is_det.sum())
    completeness_on_frame = detected / n_on_frame if n_on_frame > 0 else float("nan")

    n_blend_miss = n_threshold_miss = 0
    if n_on_frame > 0:
        of_pts = on_frame[["x", "y"]].to_numpy(dtype=float)
        of_mag = on_frame["mag_g"].to_numpy(dtype=float)
        of_tree = cKDTree(of_pts)
        undet_idx = np.where(~is_det)[0]
        for i in undet_idx:
            neigh = of_tree.query_ball_point(of_pts[i], r=fwhm_px)
            has_bright = False
            for j in neigh:
                if j == i:
                    continue
                if of_mag[j] <= of_mag[i] + 0.5:  # brighter or within 0.5 mag
                    has_bright = True
                    break
            if has_bright:
                n_blend_miss += 1
            else:
                n_threshold_miss += 1
    n_undetected = n_on_frame - detected
    achievable_ceiling = (1.0 - n_blend_miss / n_on_frame) if n_on_frame > 0 else float("nan")
    blend_miss_frac = n_blend_miss / n_on_frame if n_on_frame > 0 else float("nan")
    threshold_miss_frac = n_threshold_miss / n_on_frame if n_on_frame > 0 else float("nan")

    # ----- PART D: per-target PSF-deblend worklist (active targets, or full LC when gated)
    if lc_star_set:
        lc = _load_lc_star_table(ps)
        stars_for_d = lc if not lc.empty else pd.DataFrame()
    else:
        at_path = phot / "active_targets.csv"
        stars_for_d = pd.DataFrame()
        if at_path.is_file():
            at = pd.read_csv(at_path, low_memory=False, dtype={"catalog_id": str})
            at["catalog_id"] = at["catalog_id"].fillna("").astype(str).str.strip()
            stars_for_d = at
    targets_df = _build_blend_targets_df(
        stars_for_d,
        wcs=wcs,
        cone_f=cone_f,
        fwhm_px=fwhm_px,
        sky_pp=sky_pp,
        area=area,
        gain=gain,
        rn=rn,
        zp=zp,
        frame_limit_mag=fl,
    )

    result = {
        "draft_id": int(draft_id),
        "setup": setup,
        # metadata
        "fwhm_px": round(fwhm_px, 4),
        "plate_scale_arcsec_px": round(plate_scale, 5),
        "fwhm_arcsec": round(fwhm_arcsec, 4),
        "gain_e_per_adu": gain,
        "read_noise_e": rn,
        "gain_source": gain_src,
        "sky_adu_per_px_proxy": round(sky_pp, 3),
        "instrumental_zero_point": round(zp, 4) if math.isfinite(zp) else None,
        "n_matched_gaia": int(len(matched)),
        "n_cone_total": int(len(cone)),
        "n_cone_in_frame": int(len(cone_f)),
        "footprint_area_arcmin2": round(footprint_area_arcmin2, 4),
        # cone depth (row-cap induced = a density signal in itself)
        "cone_max_g_100k": round(cone_max_g, 3) if math.isfinite(cone_max_g) else None,
        "cone_is_row_capped": cone_is_row_capped,
        "gaia_db_max_g": round(float(gaia_db_max_g), 3),
        # Part A
        "frame_limit_mag": round(frame_limit_mag, 3) if math.isfinite(frame_limit_mag) else None,
        "frame_limit_mag_analytic": round(analytic_snr5, 3) if math.isfinite(analytic_snr5) else None,
        "catalog_limit_g": round(catalog_limit, 3),
        "effective_limit": round(effective_limit, 3),
        "catalog_is_bottleneck": catalog_is_bottleneck,
        # Part B
        "gaia_density_per_arcmin2": round(gaia_density_per_arcmin2, 4) if math.isfinite(gaia_density_per_arcmin2) else None,
        "n_gaia_below_eff_limit": n_gaia_eff,
        "blend_frac_1fwhm": round(blend_frac_1fwhm, 4) if math.isfinite(blend_frac_1fwhm) else None,
        "blend_frac_2fwhm": round(blend_frac_2fwhm, 4) if math.isfinite(blend_frac_2fwhm) else None,
        # Part C
        "n_on_frame": n_on_frame,
        "detected": detected,
        "n_undetected": n_undetected,
        "completeness_on_frame": round(completeness_on_frame, 4) if math.isfinite(completeness_on_frame) else None,
        "completeness_is_lower_bound": catalog_is_bottleneck,
        "n_blend_miss": n_blend_miss,
        "n_threshold_miss": n_threshold_miss,
        "blend_miss_frac": round(blend_miss_frac, 4) if math.isfinite(blend_miss_frac) else None,
        "threshold_miss_frac": round(threshold_miss_frac, 4) if math.isfinite(threshold_miss_frac) else None,
        "achievable_ceiling": round(achievable_ceiling, 4) if math.isfinite(achievable_ceiling) else None,
        "n_below_depth": n_below_depth,
    }
    return result, targets_df
