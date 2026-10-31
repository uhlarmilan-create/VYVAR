"""Moved from photometry_core.py (CONSOLIDATE-01E3). Facade re-exports these names."""
from __future__ import annotations

from pathlib import Path
from typing import AbstractSet, Any
import json
import logging
import math
from astropy.io import fits as astrofits
import numpy as np
import pandas as pd
from stats_core import coerce_bool_series
from comp_pool_rms import _lookup_rms_map, attach_comp_rms_to_pool_rows, comp_frame_stats
from comp_rms_loo import (
    COMP_PHOTON_MAG_COL,
    COMP_RMS_FRAMES_BASIS,
    COMP_RMS_LOO_PHOTON_K_DEFAULT,
    COMP_SCINT_MAG_COL,
    COMP_SIGMA_REF_COL,
    compute_loo_mag_rms_map,
)
from config import AppConfig
from pipeline_constants import COMP_SELECT_RMS_FLOOR
from catalog_db import query_local_gaia_by_source_ids
from gaia_catalog_id import normalize_gaia_source_id, normalize_gaia_source_id_series
from infolog import log_event
from photometry_shared import _normalize_gaia_id
from photometry_defs import LOGGER
from gaia_catalog_id import VYVAR_CSV_DTYPE

LAST_EXCLUDED_TARGETS: pd.DataFrame = pd.DataFrame(
    columns=["name", "vsx_name", "vsx_type", "ra_deg", "dec_deg", "mag", "reason"]
)

def _sid_int(v: Any) -> int | None:
    sid = normalize_gaia_source_id(v)
    if sid and sid.isdigit():
        try:
            return int(sid)
        except Exception:  # noqa: BLE001
            # EXC-0121: T4 -- Non-integer Gaia source_id returns None - downstream treats star as uncatalogued (EXCEPT-BULK 2026-07-08)
            return None
    return None

def gaia_bp_rp(ids: Any, gaia_db_path: str | Path | None) -> dict[str, float | None]:
    """Gaia DR3 BP-RP by source_id from the local Gaia DB: normalised source_id -> bp_rp (None when the
    star has no colour), one batched query (``catalog_db.query_local_gaia_by_source_ids``).
    G07-07: the one Gaia colour lookup of target / comparison-star selection. Ids that are not Gaia
    source_ids are skipped; no DB file -> {}; a failed query logs a WARNING and returns {}."""
    gdb = str(gaia_db_path or "").strip()
    if not gdb:
        return {}
    try:
        gp = Path(gdb).expanduser().resolve()
        if not gp.is_file():
            return {}
    except OSError:
        return {}
    ids_norm: list[str] = []
    seen: set[str] = set()
    for raw in ids:
        g = normalize_gaia_source_id(raw)
        if not g or not g.isdigit() or g in seen:
            continue
        seen.add(g)
        ids_norm.append(g)
    if not ids_norm:
        return {}
    try:
        base = query_local_gaia_by_source_ids(gp, ids_norm)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[GAIA-BPRP] Batch Gaia bp_rp lookup failed (%d ids): %s", len(ids_norm), exc)
        return {}
    return {k: v.get("bp_rp") for k, v in base.items()}


def _finite_or_nan(v: Any) -> float:
    try:
        f = float(v) if v is not None else float("nan")
    except (TypeError, ValueError):
        return float("nan")
    return f if math.isfinite(f) else float("nan")


def _db_bp_rp(gids: list[str], gaia_db_path: Any,
              gaia_prefetch: dict[str, float | None] | None = None) -> dict[str, float]:
    """Gaia DB colour (finite or NaN) of normalised Gaia ids: ``gaia_prefetch`` (an earlier ``gaia_bp_rp``
    result of the same DB) for the ids it holds, one batched ``gaia_bp_rp`` query for the rest."""
    pf = gaia_prefetch or {}
    rest = [g for g in gids if g not in pf]
    got = gaia_bp_rp(rest, gaia_db_path) if rest else {}
    return {g: _finite_or_nan(pf[g] if g in pf else got.get(g)) for g in gids}


def _apply_gaia_bp_rp(df: pd.DataFrame, sid_of_row: Any, gaia_db_path: Any,
                      gaia_prefetch: dict[str, float | None] | None = None) -> pd.DataFrame:
    """Set ``bp_rp`` from the local Gaia DB (G07-07, Milan 2026-10-30: "DB wins"): a finite DB value
    replaces any other value (CSV column); rows whose id the DB lacks (or has without a colour) keep theirs."""
    sids: dict[int, str] = {}
    for _pos, row in enumerate(df.to_dict("records")):
        sid_i = _sid_int(sid_of_row(row))
        if sid_i is not None:
            sids[_pos] = normalize_gaia_source_id(sid_i)
    if not sids:
        return df
    vals = _db_bp_rp(list(dict.fromkeys(sids.values())), gaia_db_path, gaia_prefetch)
    for _pos, gid in sids.items():
        v = vals[gid]
        if math.isfinite(v):
            df.at[df.index[_pos], "bp_rp"] = v
    return df


def _enrich_comp_bp_rp(
    candidates: pd.DataFrame,
    gaia_db_path: str | None,
    *,
    gaia_prefetch: dict[str, float | None] | None = None,
) -> pd.DataFrame:
    """``bp_rp`` of comp stars from Gaia DR3 by ``source_id`` (the DB value wins, ``_apply_gaia_bp_rp``)."""
    if candidates is None or getattr(candidates, "empty", True):
        return pd.DataFrame()

    df = candidates.copy()
    if "bp_rp" not in df.columns:
        df["bp_rp"] = float("nan")

    df["bp_rp"] = pd.to_numeric(df.get("bp_rp"), errors="coerce")
    if "ra_deg" in df.columns:
        df["ra_deg"] = pd.to_numeric(df.get("ra_deg"), errors="coerce")
    if "dec_deg" in df.columns:
        df["dec_deg"] = pd.to_numeric(df.get("dec_deg"), errors="coerce")
    return _apply_gaia_bp_rp(
        df, lambda row: row.get("source_id") or row.get("catalog_id") or row.get("name"), gaia_db_path, gaia_prefetch
    )

def _ensure_active_target_names(df: pd.DataFrame) -> pd.DataFrame:
    """``vsx_name`` holds the VSX name only (empty for manual / exoplanet / candidate targets - Milan
    2026-10-29, G11-22); a blank ``name`` takes the catalog id. Displays use
    ``star_page_m3.target_display_name``."""
    if df is None or getattr(df, "empty", True):
        return df
    out = df.copy()
    if "vsx_name" not in out.columns:
        out["vsx_name"] = ""
    if "name" in out.columns:
        blank_name = out["name"].astype(str).str.strip().str.lower().isin(("", "nan", "none"))
        out.loc[blank_name, "name"] = out.loc[blank_name, "catalog_id"]
    return out

# GAIA-ID-2026-10: one exact normaliser. The former local version did str(int(float(id))), which
# changes about half of all 19-digit Gaia ids (float64 keeps 15-16 digits): the comparison pool then
# kept corrupted ids and lost every star whose id no longer matched the per-frame catalogs
# (draft 537: 2154 of 4265 MASTERSTAR ids; derived pool 545 -> 273).
_normalize_id_value = normalize_gaia_source_id
_normalize_id_series = normalize_gaia_source_id_series

def _target_selection_effective_frame_hw_px(
    vt: pd.DataFrame,
    ms: pd.DataFrame,
    *,
    frame_w_px: int,
    frame_h_px: int,
    edge_margin_px: int,
) -> tuple[int, int]:
    """``frame_w_px`` / ``frame_h_px`` from the call or larger - from the max x,y in VT and masterstars.

    The default 2082x1397 often does not match a large chip; otherwise VSX targets at large pixel
    coordinates (e.g. DY Peg) are excluded before the cross-match, **regardless** of ``vsx_type`` (no SXPHE filter).
    """
    xs: list[float] = []
    ys: list[float] = []
    for df in (vt, ms):
        if "x" in df.columns and "y" in df.columns:
            xs.extend(pd.to_numeric(df["x"], errors="coerce").dropna().astype(float).tolist())
            ys.extend(pd.to_numeric(df["y"], errors="coerce").dropna().astype(float).tolist())
    if not xs or not ys:
        return int(frame_w_px), int(frame_h_px)
    em = int(edge_margin_px)
    need_w = int(math.ceil(float(max(xs)))) + em + 2
    need_h = int(math.ceil(float(max(ys)))) + em + 2
    return max(int(frame_w_px), need_w), max(int(frame_h_px), need_h)

def _enrich_active_targets_bp_rp(
    targets_df: pd.DataFrame,
    *,
    gaia_db_path: str | Path | None,
) -> pd.DataFrame:
    """``bp_rp`` of active targets from Gaia DR3 by ``catalog_id`` (the DB value wins, ``_apply_gaia_bp_rp``)."""
    if targets_df is None or getattr(targets_df, "empty", True):
        return targets_df

    df = targets_df.copy()
    if "bp_rp" not in df.columns:
        df["bp_rp"] = float("nan")
    df["bp_rp"] = pd.to_numeric(df["bp_rp"], errors="coerce")
    if "ra_deg" in df.columns:
        df["ra_deg"] = pd.to_numeric(df["ra_deg"], errors="coerce")
    if "dec_deg" in df.columns:
        df["dec_deg"] = pd.to_numeric(df["dec_deg"], errors="coerce")

    return _apply_gaia_bp_rp(df, lambda row: row.get("catalog_id"), gaia_db_path)

def _resolve_frame_hw_px_from_masterstar(
    ms_fits: Path,
    *,
    db: Any = None,
    draft_id: int | None = None,
) -> tuple[int, int, str]:
    """Chip width / height for target and comparison-star selection: MASTERSTAR ``NAXIS1/2``, else a light
    FITS of the draft. CONFIG-ONE-2026-10 (G07-21): no caller default (it was 2082 x 1397 px, one camera);
    an unknown size is an error."""
    if ms_fits.is_file():
        try:
            with astrofits.open(ms_fits, memmap=False) as hdul:
                hdr = hdul[0].header
                w = int(hdr.get("NAXIS1", 0) or 0)
                h = int(hdr.get("NAXIS2", 0) or 0)
            if w > 0 and h > 0:
                return w, h, "fits_naxis"
        except Exception as exc:  # noqa: BLE001 - next source: a light of the draft
            logging.warning("[TARGET-SELECT] MASTERSTAR size not read from %s: %s", ms_fits.name, exc)
    if db is not None and draft_id is not None and int(draft_id) > 0:
        for lr in db.fetch_draft_light_rows_for_quality(int(draft_id)):
            fp0 = lr.get("FILE_PATH")
            if fp0:
                with astrofits.open(str(fp0), memmap=False) as _hdul:
                    w = int(_hdul[0].header.get("NAXIS1") or 0)
                    h = int(_hdul[0].header.get("NAXIS2") or 0)
                if w > 0 and h > 0:
                    return w, h, "fits_naxis_light"
                break
    raise RuntimeError(f"frame size unknown: no NAXIS in {ms_fits} and no light FITS of draft {draft_id}")


def field_density_inputs(
    ms_header: Any,
    ms_catalog_ids: pd.Series | None,
    frame_w_px: int,
    frame_h_px: int,
) -> tuple[int, int, int, str, int | None]:
    """Return ``(n_stars, chip_w, chip_h, source, n_stars_dao_raw)`` for the field density (diagnostic).

    G07-19 (Milan 2026-10-30, option b): from data the selection already holds - the MASTERSTAR header
    (``VY_NDAO``) and the ``catalog_id`` column of masterstars_full_match.csv as read (text).
    ``n_stars``: rows with a non-empty ``catalog_id`` (Gaia-matched).
    ``n_stars_dao_raw``: ``VY_NDAO`` (reference / JSON only, not classification).
    ``source``: ``masterstars_gaia_matched`` | ``VY_NDAO_fallback`` | ``defaults``.
    """
    cw, ch = int(frame_w_px), int(frame_h_px)
    n_stars = 0
    src = "defaults"
    vy_ndao_raw: int | None = None
    v = ms_header.get("VY_NDAO") if ms_header is not None else None
    if v is not None and str(v).strip() != "":
        try:
            vy_ndao_raw = int(float(v))
        except (TypeError, ValueError):
            vy_ndao_raw = None
    if ms_catalog_ids is not None:
        _cid = ms_catalog_ids.astype(str).str.strip()
        _n_gaia = int((~_cid.isin(["", "nan", "None"])).sum())
        if _n_gaia > 0:
            n_stars = _n_gaia
            src = "masterstars_gaia_matched"
    if n_stars <= 0 and vy_ndao_raw is not None and vy_ndao_raw > 0:
        n_stars = int(vy_ndao_raw)
        src = "VY_NDAO_fallback"
    return int(max(0, n_stars)), cw, ch, src, vy_ndao_raw

def _refresh_variable_targets_xy(
    variable_targets_csv: Path,
    wcs: Any,
    chip_w: int,
    chip_h: int,
) -> None:
    """Recompute the x/y columns of variable_targets.csv from the current MASTERSTAR WCS."""
    from astropy.wcs import WCS

    if wcs is None or not isinstance(wcs, WCS):
        return
    vt_path = Path(variable_targets_csv)
    if not vt_path.is_file():
        return

    logging.debug("[VT REFRESH] frame %sx%s px (MASTERSTAR -> VT x,y)", chip_w, chip_h)

    df = pd.read_csv(vt_path, low_memory=False, dtype=VYVAR_CSV_DTYPE)
    if "ra_deg" in df.columns:
        ra = pd.to_numeric(df["ra_deg"], errors="coerce").to_numpy(dtype=np.float64)
    elif "ra" in df.columns:
        ra = pd.to_numeric(df["ra"], errors="coerce").to_numpy(dtype=np.float64)
    else:
        logging.warning("[VT REFRESH] missing columns ra_deg / ra - x/y not updated")
        return
    if "dec_deg" in df.columns:
        dec = pd.to_numeric(df["dec_deg"], errors="coerce").to_numpy(dtype=np.float64)
    elif "dec" in df.columns:
        dec = pd.to_numeric(df["dec"], errors="coerce").to_numpy(dtype=np.float64)
    else:
        logging.warning("[VT REFRESH] missing columns dec_deg / dec - x/y not updated")
        return

    try:
        ok = np.isfinite(ra) & np.isfinite(dec)
        xy = np.full((len(df), 2), np.nan, dtype=np.float64)
        if bool(ok.any()):
            pts = np.column_stack([ra[ok], dec[ok]])
            xy[ok, :] = wcs.all_world2pix(pts, 0)
        df["x"] = xy[:, 0]
        df["y"] = xy[:, 1]
        df.to_csv(vt_path, index=False)
    except (ValueError, TypeError, AttributeError) as e:
        from except_fix_counters import get_except_fix_counters

        get_except_fix_counters().vt_wcs_refresh_fail += 1
        logging.error("[VT REFRESH] WCS recompute failed: %s - x/y stay stale", e)
        return

    logging.info("[VT REFRESH] x/y coordinates of variable_targets.csv updated from MASTERSTAR WCS")
    xv = df["x"].to_numpy(dtype=np.float64, copy=False)
    yv = df["y"].to_numpy(dtype=np.float64, copy=False)
    if np.isfinite(xv).any() and np.isfinite(yv).any():
        logging.info(
            "[VT REFRESH] %d rows, x=[%.0f,%.0f] y=[%.0f,%.0f]",
            len(df),
            float(np.nanmin(xv)),
            float(np.nanmax(xv)),
            float(np.nanmin(yv)),
            float(np.nanmax(yv)),
        )
    else:
        logging.info("[VT REFRESH] %d rows (no valid x/y after recompute)", len(df))

def _masterstar_row_is_dao_detection(ms_row: pd.Series) -> bool:
    """TARGET-RULES-2026-10: True for a MASTERSTAR row that is a DAO detection (finite DAO flux).

    Catalogue-membership rows (DAO-GAIA-ERA-01 M1: a Gaia position added without a
    detection) carry ``source_state='catalog_membership'`` and no flux."""
    if str(ms_row.get("source_state", "") or "").strip().lower() == "catalog_membership":
        return False
    f = float(pd.to_numeric(ms_row.get("flux", float("nan")), errors="coerce"))
    return math.isfinite(f)


def _at_in_frame_targets(
    *,
    cfg,
    edge_margin_px,
    frame_h_px,
    frame_w_px,
    masterstars_csv,
    safe_bbox,
    variable_targets_csv,
):
    """Variable targets inside the frame / safe bbox and the MASTERSTAR rows by catalog id."""
    vt = pd.read_csv(variable_targets_csv, low_memory=False, dtype=VYVAR_CSV_DTYPE)
    # masterstars_full_match.csv often carries the exact Gaia source_id in "name" even when catalog_id is float-corrupted
    ms = pd.read_csv(masterstars_csv, low_memory=False, dtype=VYVAR_CSV_DTYPE)
    if "catalog_id" in vt.columns:
        vt["catalog_id"] = vt["catalog_id"].apply(_normalize_gaia_id)
    # Normalize Gaia IDs to string.
    # CAUTION: "name" in masterstars often holds the exact Gaia source_id; it must not go through float().
    if "catalog_id" in ms.columns:
        ms["catalog_id"] = _normalize_id_series(ms["catalog_id"])
    if "name" in ms.columns:
        ms["name"] = ms["name"].fillna("").astype(str).str.strip()

    fw, fh = _target_selection_effective_frame_hw_px(
        vt, ms, frame_w_px=int(frame_w_px), frame_h_px=int(frame_h_px), edge_margin_px=int(edge_margin_px)
    )
    if fw != int(frame_w_px) or fh != int(frame_h_px):
        logging.info(
            "[TARGET-SELECT] Chip size enlarged from %sx%s to %sx%s px (max x,y from variable_targets/masterstars + margin)",
            int(frame_w_px),
            int(frame_h_px),
            fw,
            fh,
        )

    # Filter: in the frame (annulus-aware safe bbox, else fixed edge margin)
    vt["x"] = pd.to_numeric(vt["x"], errors="coerce")
    vt["y"] = pd.to_numeric(vt["y"], errors="coerce")
    if safe_bbox is not None:
        try:
            from aperture_policy import stars_fit_on_chip  # noqa: PLC0415

            x0b, y0b, x1b, y1b = safe_bbox
            before = int(len(vt))
            in_frame = stars_fit_on_chip(
                vt["x"], vt["y"], (0.0, 0.0, 0.0), (float(x0b), float(y0b), float(x1b), float(y1b))
            )
            in_frame = pd.Series(np.asarray(in_frame, dtype=bool), index=vt.index)
            removed = before - int(in_frame.sum())
            if removed > 0:
                logging.info(
                    f"[BORDER] Active targets: removed {removed} rows outside safe bbox "
                    f"(annulus-aware intersection)"
                )
        except Exception:  # noqa: BLE001
            in_frame = (
                vt["x"].between(edge_margin_px, fw - edge_margin_px)
                & vt["y"].between(edge_margin_px, fh - edge_margin_px)
            )
    else:
        in_frame = (
            vt["x"].between(edge_margin_px, fw - edge_margin_px)
            & vt["y"].between(edge_margin_px, fh - edge_margin_px)
        )
    vt_in = vt[in_frame].copy()

    _cfg = cfg


    # Identity join index: normalized Gaia source_id -> masterstar row.
    ms_by_cid: dict[str, pd.Series] = {}
    for _, ms_row in ms.iterrows():
        cid_ms = _normalize_gaia_id(ms_row.get("catalog_id", ""))  # identity = catalog_id only (G04-17)
        if cid_ms and cid_ms not in ms_by_cid:
            ms_by_cid[str(cid_ms)] = ms_row

    out_of_frame = int(len(vt) - int(in_frame.sum()))
    return _cfg, in_frame, ms_by_cid, out_of_frame, vt, vt_in


def _at_match_targets(*, _cfg, in_frame, ms_by_cid, vt, vt_in):
    """Match each in-frame target to its MASTERSTAR detection; count the exclusion reasons."""
    from vsx_type_scope import is_vsx_auto_selected_target  # noqa: PLC0415
    from vsx_type_scope import vsx_type_is_out_of_scope  # noqa: PLC0415

    no_catalog_id = 0
    no_gaia_id = 0
    no_dao_detection = 0
    not_target_eligible = 0
    matched_rows: list[dict] = []
    matched_vt_idx: set[Any] = set()
    excluded_rows: list[dict[str, Any]] = []

    def _excluded_target_row(vrow: pd.Series, reason: str, *, mag: float | None = None) -> dict[str, Any]:
        mag_val = mag
        if mag_val is None:
            mag_val = float(pd.to_numeric(vrow.get("mag", float("nan")), errors="coerce"))
        return {
            "name": str(vrow.get("name", "") or ""),
            "vsx_name": str(vrow.get("vsx_name", "") or ""),
            "vsx_type": str(vrow.get("vsx_type", "") or ""),
            "ra_deg": float(vrow.get("ra_deg", float("nan"))),
            "dec_deg": float(vrow.get("dec_deg", float("nan"))),
            "mag": mag_val,
            "reason": reason,
        }

    for _, vrow_off in vt.loc[~in_frame].iterrows():
        excluded_rows.append(_excluded_target_row(vrow_off, "out_of_frame"))

    for vidx, vrow in vt_in.iterrows():
        ra_v = float(pd.to_numeric(vrow.get("ra_deg"), errors="coerce"))
        dec_v = float(pd.to_numeric(vrow.get("dec_deg"), errors="coerce"))
        if not (math.isfinite(ra_v) and math.isfinite(dec_v)):
            continue

        is_vsx_auto = is_vsx_auto_selected_target(vrow)
        planner_cid = _normalize_gaia_id(vrow.get("catalog_id", ""))
        gaia_src = str(vrow.get("gaia_match_source", "") or "").strip().lower()

        if is_vsx_auto:
            if gaia_src not in ("masterstars",):
                not_target_eligible += 1
                excluded_rows.append(
                    _excluded_target_row(vrow, "no_gaia_id" if not planner_cid else "not_target_eligible")
                )
                continue
            if not planner_cid:
                no_gaia_id += 1
                excluded_rows.append(_excluded_target_row(vrow, "no_gaia_id"))
                continue
        elif not planner_cid:
            no_catalog_id += 1
            excluded_rows.append(_excluded_target_row(vrow, "no_catalog_id"))
            continue

        ms_row = ms_by_cid.get(str(planner_cid))
        # TARGET-RULES-2026-10: only a DAO detection on MASTERSTAR is image data. A catalogue
        # membership row (Gaia position added without a detection, flux NaN) is not.
        if ms_row is None or not _masterstar_row_is_dao_detection(ms_row):
            no_dao_detection += 1
            excluded_rows.append(_excluded_target_row(vrow, "no_dao_detection"))
            continue

        catalog_id_norm = str(planner_cid)
        zone_flag = str(ms_row.get("zone", "") or "").strip().lower()
        mag_for_skip = float(
            pd.to_numeric(
                ms_row.get("mag", ms_row.get("phot_g_mean_mag", float("nan"))),
                errors="coerce",
            )
        )
        skip_ph = False
        skip_reason = ""
        # upper cut: saturated on MASTERSTAR (final - never measured)
        if zone_flag == "saturated":
            skip_ph = True
            skip_reason = "saturated"
        # lower cut: drowned in noise - aperture S/N on MASTERSTAR below cfg.snr_min. A DAO
        # detection without a positive aperture flux has no S/N -> no signal -> below.
        _snr_t = float(pd.to_numeric(ms_row.get("snr_ap_pixscaled", float("nan")), errors="coerce"))
        _snr_min = float(getattr(_cfg, "snr_min", 10.0))
        if (not skip_ph) and not (math.isfinite(_snr_t) and _snr_t >= _snr_min):
            skip_ph = True
            skip_reason = "below_snr_min"
        _voos = list(getattr(_cfg, "vsx_out_of_scope_types", []) or [])
        if (
            (not skip_ph)
            and _voos
            and is_vsx_auto
            and vsx_type_is_out_of_scope(str(vrow.get("vsx_type", "") or ""), _voos)
        ):
            skip_ph = True
            skip_reason = "vsx_type_out_of_scope"
        rec = {
            "name": vrow.get("name", ""),
            "vsx_name": vrow.get("vsx_name", ""),
            "vsx_type": vrow.get("vsx_type", ""),
            "vsx_period": vrow.get("vsx_period", ""),
            "ra_deg": ra_v,
            "dec_deg": dec_v,
            "x": float(pd.to_numeric(ms_row.get("x", vrow.get("x")), errors="coerce")),
            "y": float(pd.to_numeric(ms_row.get("y", vrow.get("y")), errors="coerce")),
            "catalog_id": catalog_id_norm,
            "mag": mag_for_skip,
            "bp_rp": float(ms_row.get("bp_rp", float("nan"))),
            "zone_flag": zone_flag,
            "skip_photometry": bool(skip_ph),
            "skip_reason": str(skip_reason),
        }
        if "catalog" in vrow.index:
            rec["catalog"] = vrow.get("catalog", "")
        for _exo_col in (
            "exo_host_name",
            "exo_cat_source",
            "exo_disposition",
            "target_origin",
        ):
            if _exo_col in vrow.index:
                rec[_exo_col] = vrow.get(_exo_col, "")
        matched_vt_idx.add(vidx)
        matched_rows.append(rec)
    return excluded_rows, matched_rows, no_catalog_id, no_dao_detection, no_gaia_id, not_target_eligible


def _at_assemble(
    *,
    _cfg,
    _empty_cols,
    gaia_db_path,
    matched_rows,
    n_gaia_id_assigned,
    no_catalog_id,
    no_dao_detection,
    no_gaia_id,
    not_target_eligible,
    out_of_frame,
    vt,
    vt_in,
):
    """Active target table: ids, Gaia colours, funnel log and the excluded rows."""
    result = pd.DataFrame(matched_rows) if matched_rows else pd.DataFrame(columns=_empty_cols)
    if "catalog_id" in result.columns:
        # DO NOT use float() (precision loss). Use the robust normalization.
        result["catalog_id"] = result["catalog_id"].apply(_normalize_gaia_id)
    # Gaia BP-RP + the same B-V hierarchy as for comps (must not overwrite all b_v from NaN bp_rp in masterstars).
    result = _enrich_active_targets_bp_rp(
        result,
        gaia_db_path=gaia_db_path,
    )
    # Deduplicate by catalog_id - keep row with real VSX name over
    # Gaia-placeholder (e.g. "V0842 Her" preferred over
    # "Gaia DR3 1400549806859236864")
    if "catalog_id" in result.columns:
        # Prefer rows where vsx_name / name does NOT start with "Gaia DR3"
        _is_gaia_placeholder = (
            result.get("vsx_name", result.get("name", pd.Series(dtype=str)))
            .astype(str)
            .str.startswith("Gaia DR3")
        )
        # Sort: non-Gaia-placeholder first, then keep first per catalog_id
        result = (
            result
            .assign(_gaia_placeholder=_is_gaia_placeholder.astype(int))
            .sort_values("_gaia_placeholder")
            .drop_duplicates(subset=["catalog_id"], keep="first")
            .drop(columns=["_gaia_placeholder"])
            .reset_index(drop=True)
        )
        log_event(
            f"select_active_targets: deduped to {len(result)} unique "
            f"catalog_ids (prefer real VSX name over Gaia placeholder)"
        )
    _sr_res = result["skip_reason"].astype(str) if "skip_reason" in result.columns else pd.Series(dtype=str)
    n_masked_sat = int((_sr_res == "saturated").sum())
    n_masked_vsx_type = int((_sr_res == "vsx_type_out_of_scope").sum())
    n_masked_snr = int((_sr_res == "below_snr_min").sum())
    _n_meas = int((~result["skip_photometry"].astype(bool)).sum()) if "skip_photometry" in result.columns else 0
    _funnel_msg = (
        f"[TARGET-SELECT] funnel: vsx_bbox={int(len(vt))} -> in_frame={int(len(vt_in))} -> "
        f"gaia_id_assigned={n_gaia_id_assigned} -> dao_detected={len(result)} -> to_measure={_n_meas} | "
        f"excluded: no_dao_detection={no_dao_detection} no_gaia_id={no_gaia_id} "
        f"no_catalog_id={no_catalog_id} not_target_eligible={not_target_eligible} "
        f"out_of_frame={out_of_frame} | masked: saturated={n_masked_sat} "
        f"below_snr_min={n_masked_snr} vsx_type_out_of_scope={n_masked_vsx_type}"
    )
    logging.info(_funnel_msg)
    log_event(_funnel_msg)
    _voos_cfg = list(getattr(_cfg, "vsx_out_of_scope_types", []) or [])
    if _voos_cfg and n_masked_vsx_type == 0 and "catalog" in vt.columns:
        _vt_vsx = vt[vt["catalog"].astype(str).str.upper() == "VSX"]
        _types = set()
        # fillna first: under pandas 3 astype(str) keeps a missing type as float NaN
        for _t in _vt_vsx.get("vsx_type", pd.Series(dtype=str)).fillna("").astype(str):
            for _tok in _t.replace(":", " ").replace("/", " ").split():
                _types.add(_tok.strip().upper())
        if any(str(t).strip().upper() in _types for t in _voos_cfg):
            logging.warning(
                "[TARGET-SELECT] vsx_out_of_scope_types=%s but zero rows masked - check VSX type tokens",
                _voos_cfg,
            )
    if n_gaia_id_assigned > 0 and len(vt_in) > 0:
        _assign_frac = float(n_gaia_id_assigned) / float(len(vt_in))
        if _assign_frac < 0.50:
            logging.warning(
                "[TARGET-SELECT] gaia_id_assigned=%d far below in_frame=%d (%.0f%%) - sparse Gaia cross-match",
                n_gaia_id_assigned,
                int(len(vt_in)),
                100.0 * _assign_frac,
            )
    return result,


def select_active_targets(
    variable_targets_csv: Path,
    masterstars_csv: Path,
    *,
    frame_w_px: int,
    frame_h_px: int,
    edge_margin_px: int = 50,
    safe_bbox: tuple[float, float, float, float] | None = None,
    gaia_db_path: str | None = None,
    cfg: AppConfig,
) -> pd.DataFrame:
    """Target selection: filter VSX variables -> active_targets.

    Rules:
    - The star must be in the frame (``x,y`` at least ``edge_margin_px`` from the edge of the effective field; the
      same number as ``chip_interior_margin_px`` in target and comparison-star selection - consistent with the
      comparison stars and suspected variables).
    - Width/height are enlarged from the data if needed
    - VSX auto rows: identity join on planner ``catalog_id`` in masterstars (no positional matching).
      Only ``gaia_match_source=masterstars`` rows are promotable; ``gaia_dr3_direct`` / ``no_match`` stay
      in variable_targets (neighbour light at maximum, B4-2026-10) but are excluded here.
    - Manual / exoplanet targets: identity join on ``catalog_id`` when present (not subject to VSX gate).
    - ``catalog_id`` from masterstars must be non-empty (otherwise the target is skipped).
    - TARGET-RULES-2026-10 (Milan 2026-10-13): a target is a star DAO detected on MASTERSTAR.
      Catalogue-membership rows (Gaia position, no DAO detection) are catalogue rows, not image
      data -> excluded (``no_dao_detection``). Then exactly three final cuts, in this order:
      ``saturated`` (MASTERSTAR zone saturated = peak above the linear limit 0.90 x SAT-DIAG),
      ``below_snr_min`` (MASTERSTAR aperture S/N < cfg.snr_min, star drowned in noise) and
      ``vsx_type_out_of_scope`` (cfg.vsx_out_of_scope_types, VSX auto-selected targets only).
      Masked targets stay in active_targets with ``skip_photometry=True`` (mask-first).
    Returns:
        DataFrame of active targets - columns from variable_targets + added from masterstars:
        [name, catalog_id, ra_deg, dec_deg, vsx_name, vsx_type, vsx_period,
         x, y, mag, b_v, bp_rp, zone_flag, skip_photometry, skip_reason]
    """
    global LAST_EXCLUDED_TARGETS
    # Auto-repair corrupted Gaia IDs before loading (if a local Gaia DB is available).

    # variable_targets.csv may come from outside and often has catalog_id as float/scientific - read as string
    _cfg, in_frame, ms_by_cid, out_of_frame, vt, vt_in = _at_in_frame_targets(
        cfg=cfg,
        edge_margin_px=edge_margin_px,
        frame_h_px=frame_h_px,
        frame_w_px=frame_w_px,
        masterstars_csv=masterstars_csv,
        safe_bbox=safe_bbox,
        variable_targets_csv=variable_targets_csv,
    )
    excluded_rows, matched_rows, no_catalog_id, no_dao_detection, no_gaia_id, not_target_eligible = _at_match_targets(
        _cfg=_cfg,
        in_frame=in_frame,
        ms_by_cid=ms_by_cid,
        vt=vt,
        vt_in=vt_in,
    )

    _empty_cols = [
        "name",
        "vsx_name",
        "vsx_type",
        "vsx_period",
        "ra_deg",
        "dec_deg",
        "x",
        "y",
        "catalog_id",
        "mag",
        "bp_rp",
        "zone_flag",
        "skip_photometry",
        "skip_reason",
    ]
    # (a plain sum: on an empty string column pandas 3 sums to '' and int('') raises)
    n_gaia_id_assigned = sum(bool(_normalize_gaia_id(x)) for x in vt_in["catalog_id"]) if "catalog_id" in vt_in.columns else 0
    if not matched_rows:
        LAST_EXCLUDED_TARGETS = (
            pd.DataFrame(excluded_rows)
            if excluded_rows
            else pd.DataFrame(columns=["name", "vsx_name", "vsx_type", "ra_deg", "dec_deg", "mag", "reason"])
        )
        _funnel0 = (
            f"[TARGET-SELECT] funnel: vsx_bbox={int(len(vt))} -> in_frame={int(len(vt_in))} -> "
            f"gaia_id_assigned={n_gaia_id_assigned} -> dao_detected=0 -> active=0 | excluded: "
            f"no_dao_detection={no_dao_detection} no_gaia_id={no_gaia_id} no_catalog_id={no_catalog_id} "
            f"not_target_eligible={not_target_eligible} out_of_frame={out_of_frame}"
        )
        logging.info(_funnel0)
        log_event(_funnel0)
        return pd.DataFrame(columns=_empty_cols)

    result, = _at_assemble(
        _cfg=_cfg,
        _empty_cols=_empty_cols,
        gaia_db_path=gaia_db_path,
        matched_rows=matched_rows,
        n_gaia_id_assigned=n_gaia_id_assigned,
        no_catalog_id=no_catalog_id,
        no_dao_detection=no_dao_detection,
        no_gaia_id=no_gaia_id,
        not_target_eligible=not_target_eligible,
        out_of_frame=out_of_frame,
        vt=vt,
        vt_in=vt_in,
    )
    LAST_EXCLUDED_TARGETS = (
        pd.DataFrame(excluded_rows)
        if excluded_rows
        else pd.DataFrame(columns=["name", "vsx_name", "vsx_type", "ra_deg", "dec_deg", "mag", "reason"])
    )
    result = _ensure_active_target_names(result)
    return result.reset_index(drop=True)

def _enrich_target_bp_rp_from_gaia_db(
    target: pd.Series,
    *,
    gaia_db_path: str,
    gaia_prefetch: dict[str, float | None] | None = None,
) -> pd.Series:
    """``bp_rp`` of one active target (comparison-star selection) from Gaia by ``source_id``: the DB value wins
    (``_db_bp_rp``, G07-07), else the value the row holds."""
    out = target.copy()
    vsx = str(out.get("vsx_name", "") or "").strip() or str(out.get("name", "") or "").strip() or "?"

    def _fscalar(key: str) -> float:
        try:
            v = float(pd.to_numeric(out.get(key), errors="coerce"))
        except Exception:  # noqa: BLE001
            return float("nan")
        return v if math.isfinite(v) else float("nan")

    bpr_ms = _fscalar("bp_rp")

    gid = normalize_gaia_source_id(out.get("catalog_id"))
    bpr_nf = _db_bp_rp([gid], gaia_db_path, gaia_prefetch)[gid] if gid else float("nan")

    if math.isfinite(bpr_nf):
        out["bp_rp"] = float(bpr_nf)
    elif math.isfinite(bpr_ms):
        out["bp_rp"] = float(bpr_ms)
    elif not gid:
        log_event(f"TARGET bp_rp: {vsx} - no valid Gaia catalog_id")
    return out

def _bprp_tier_ladder_for_selection(
    cfg: AppConfig | None,
    max_delta_bprp: float,
) -> list[float]:
    """Tier-ladder colour windows: tier1 -> tier2 -> tier3 -> comp_max_delta_bprp cap."""
    if cfg is not None:
        _tier_lims = cfg.comp_tier_bprp_limits()
        raw = [
            float(_tier_lims[0]),
            float(_tier_lims[1]),
            float(_tier_lims[2]),
            float(getattr(cfg, "comp_max_delta_bprp", max_delta_bprp)),
        ]
    else:
        raw = [float(max_delta_bprp)]
    # G07-35: a widening ladder that ends at the cap (a tier above comp_max_delta_bprp is clipped)
    cap = raw[-1]
    if not (math.isfinite(cap) and cap > 0):
        raise ValueError(f"comp_max_delta_bprp must be a positive number, got {cap!r}")
    return sorted({min(float(v), cap) for v in raw if math.isfinite(v) and v > 0})

def _sel_colour_ladder(*, _n_min, cfg, max_delta_bprp, out, target_bprp):
    """Colour ladder: candidates within the widening |Delta BP-RP| window until n_comp_min remain."""
    id_col = "catalog_id" if "catalog_id" in out.columns else out.columns[0]
    tb = float(target_bprp)
    if "bp_rp" in out.columns and math.isfinite(tb):
        out["_delta_bprp_abs"] = (
            pd.to_numeric(out.get("bp_rp"), errors="coerce") - tb
        ).abs()
    else:
        out["_delta_bprp_abs"] = 0.0

    if "_dist_deg" in out.columns:
        out["_dist_sort"] = pd.to_numeric(out["_dist_deg"], errors="coerce")
    elif "dist_deg" in out.columns:
        out["_dist_sort"] = pd.to_numeric(out["dist_deg"], errors="coerce")
    else:
        out["_dist_sort"] = 0.0
    out["_dist_sort"] = out["_dist_sort"].fillna(float("inf"))
    out["_rms_sort"] = pd.to_numeric(out["comp_rms"], errors="coerce").fillna(float("inf"))
    out["_delta_bprp_abs"] = pd.to_numeric(out["_delta_bprp_abs"], errors="coerce").fillna(
        float("inf")
    )

    ladder = _bprp_tier_ladder_for_selection(cfg, float(max_delta_bprp))
    selected = pd.DataFrame()
    used_lim = float("nan")
    ladder_step = 0
    for step_i, lim in enumerate(ladder, start=1):
        pool = out[out["_delta_bprp_abs"] <= float(lim)].copy()
        if len(pool) >= _n_min:
            selected = pool
            used_lim = float(lim)
            ladder_step = int(step_i)
            logging.info(
                "[COMP] color filter at delta_bprp<=%.3f (n=%d >= n_comp_min=%d) "
                "ladder_step=%d (RMS-ordered within)",
                float(lim),
                len(pool),
                _n_min,
                ladder_step,
            )
            break
        selected = pool
        used_lim = float(lim)
        ladder_step = int(step_i)
    return id_col, ladder, ladder_step, selected, tb, used_lim


def _sel_rank_and_cap(
    *,
    _ceil,
    _iso_fwhm,
    _n_max,
    color_fallback,
    id_col,
    ladder_step,
    out,
    selected,
    used_lim,
):
    """Log the colour-window rejections, single-source isolation gate, rank by RMS and cap at n_comp_max."""
    n_pre_colour = int(len(out))
    n_colour_pool = int(len(selected))
    if n_pre_colour > n_colour_pool and math.isfinite(float(used_lim)):
        _col_dropped = out.loc[~out.index.isin(selected.index)]
        if not _col_dropped.empty:
            _bcol = _col_dropped.sort_values("_rms_sort", kind="mergesort").iloc[0]
            _db = float(pd.to_numeric(_bcol.get("_delta_bprp_abs"), errors="coerce"))
            logging.info(
                "[COMP] best colour-rejected cid=%s rms=%s d_bprp=%s colour_lim=%.3f",
                str(_bcol.get(id_col, "")),
                str(_bcol.get("comp_rms", "")),
                f"{_db:.3f}" if math.isfinite(_db) else "nan",
                float(used_lim),
            )
    _sort_cols = (
        ["_delta_bprp_abs", "_rms_sort", "_dist_sort", id_col]
        if color_fallback
        else ["_rms_sort", "_delta_bprp_abs", "_dist_sort", id_col]
    )
    selected = selected.sort_values(
        _sort_cols,
        ascending=[True, True, True, True],
        kind="mergesort",
    )
    n_clean_pool = int(len(selected))
    _ceil_s = f"{float(_ceil):.4f}" if math.isfinite(_ceil) and _ceil > 0 else "nan"
    if n_clean_pool > _n_max:
        _nxt = selected.iloc[_n_max]
        _nr = float(pd.to_numeric(_nxt.get("comp_rms"), errors="coerce"))
        _nd = float(pd.to_numeric(_nxt.get("_delta_bprp_abs"), errors="coerce"))
        _nds = float(pd.to_numeric(_nxt.get("_dist_sort"), errors="coerce"))
        logging.info(
            "[COMP] clean pool n=%d after ceiling/isolation/colour; "
            "n_comp_max=%d admitted=%d; best not-admitted (n_comp_max/distance sort) "
            "cid=%s rms=%s d_bprp=%s dist_deg=%s "
            "(thresholds: ceiling=%s isolation=%.2f FWHM colour_lim=%s)",
            n_clean_pool,
            _n_max,
            _n_max,
            str(_nxt.get(id_col, "")),
            f"{_nr:.4f}" if math.isfinite(_nr) else "nan",
            f"{_nd:.3f}" if math.isfinite(_nd) else "nan",
            f"{_nds:.5f}" if math.isfinite(_nds) else "nan",
            _ceil_s,
            float(_iso_fwhm),
            f"{float(used_lim):.3f}" if math.isfinite(float(used_lim)) else "nan",
        )
    else:
        logging.info(
            "[COMP] clean pool n=%d after ceiling/isolation/colour; "
            "n_comp_max=%d admitted=%d; nothing better in the pool "
            "(thresholds: ceiling=%s isolation=%.2f FWHM colour_lim=%s)",
            n_clean_pool,
            _n_max,
            n_clean_pool,
            _ceil_s,
            float(_iso_fwhm),
            f"{float(used_lim):.3f}" if math.isfinite(float(used_lim)) else "nan",
        )
    selected = selected.head(_n_max).copy()
    _max_d_used = float(pd.to_numeric(selected.get("_delta_bprp_abs"), errors="coerce").max())
    if not math.isfinite(_max_d_used):
        _max_d_used = float(used_lim) if math.isfinite(float(used_lim)) else float("nan")
    selected.attrs["color_ladder_lim"] = used_lim
    selected.attrs["color_ladder_step"] = int(ladder_step)
    selected.attrs["color_fallback"] = bool(color_fallback)
    selected.attrs["max_delta_bprp_used"] = float(_max_d_used)
    selected.attrs["max_comp_rms_ceiling"] = (
        float(_ceil) if math.isfinite(_ceil) and _ceil > 0 else float("nan")
    )
    selected.attrs["single_source_isolation_fwhm"] = float(_iso_fwhm)
    return selected,


def _select_comps_by_rms_then_color(
    candidates: pd.DataFrame,
    target_bprp: float,
    n_comp_min: int,
    n_comp_max: int,
    max_delta_bprp: float,
    *,
    cfg: AppConfig | None = None,
    max_comp_rms: float | None = None,
    fwhm_px: float | None = None,
) -> pd.DataFrame:
    """COMP-ASSIGN-03: RMS -> colour ladder -> distance; single-source only.

    ``comp_rms`` from the global pool (step 1). ``phase01_comparison_max_comp_rms``
    is a hard ceiling before the colour ladder and ``head(n_comp_max)``
    (COMP-ASSIGN-02). Colour window widens until ``n_comp_min`` under-ceiling
    single-source comps exist; within a colour step, order is ``comp_rms``, then
    ``|delta BP-RP|``, then ``_dist_deg``, then ``catalog_id``. Blends (nearest
    catalogue neighbour closer than ``comp_isolation_fwhm`` x FWHM) are
    excluded from candidacy. ``n_comp_max`` remains a ceiling, not a pad target.
    """
    if candidates is None or getattr(candidates, "empty", True):
        return pd.DataFrame()

    if "comp_rms" not in candidates.columns:
        raise ValueError("_select_comps_by_rms_then_color requires comp_rms column")

    _n_min = max(1, int(n_comp_min))
    _n_max = max(_n_min, int(n_comp_max))
    _floor = COMP_SELECT_RMS_FLOOR
    _ceil = float("nan")
    _iso_fwhm = 3.0
    if cfg is not None:
        try:
            _ceil = float(
                getattr(cfg, "phase01_comparison_max_comp_rms", float("nan")) or float("nan")
            )
        except (TypeError, ValueError):
            _ceil = float("nan")
        try:
            _iso_fwhm = float(getattr(cfg, "comp_isolation_fwhm", 3.0) or 3.0)
        except (TypeError, ValueError):
            _iso_fwhm = 3.0
    if max_comp_rms is not None:
        try:
            _mc = float(max_comp_rms)
        except (TypeError, ValueError):
            _mc = float("nan")
        if math.isfinite(_mc) and _mc > 0:
            _ceil = _mc
    if not (math.isfinite(_iso_fwhm) and _iso_fwhm > 0):
        _iso_fwhm = 3.0

    out = candidates.copy()
    rms = pd.to_numeric(out["comp_rms"], errors="coerce")
    out = out[rms >= _floor].copy()
    if out.empty:
        return out

    # Authoritative per-target RMS gate (INV-COMP-RMS-01): LOO mag MAD vs k * sigma_ref and an
    # absolute 0.1 mag cap. COMP-CEIL-FIELDK-2026-10 (Milan 2026-10-30): the reference is that of the
    # comp_rms aperture and frames, not the MASTERSTAR S/N (one frame, 1.9 FWHM). COMP-CEIL-NOISEMODEL-2026-10
    # (Milan 2026-10-30): sigma_ref = sqrt((a e_photon)^2 + e_scint^2 + floor^2), a / floor fitted on the
    # night's constant pool stars (pool column COMP_SIGMA_REF_COL, comp_rms_loo.comp_ceiling_reference) - the
    # photon error alone rejected good bright comps (Brno 534 g: 41 constant stars, all bright).
    _k_loo = COMP_RMS_LOO_PHOTON_K_DEFAULT
    _abs_ceil = _ceil if math.isfinite(_ceil) and _ceil > 0 else 0.1
    if COMP_SIGMA_REF_COL not in out.columns:
        raise ValueError(
            f"INV-COMP-RMS-01: {COMP_SIGMA_REF_COL} missing on comparison candidates "
            "(the comparison pool carries it, build_global_comp_pool)"
        )
    _ref = pd.to_numeric(out[COMP_SIGMA_REF_COL], errors="coerce")
    _rms_c = pd.to_numeric(out["comp_rms"], errors="coerce")
    _star_ceil = np.minimum(float(_abs_ceil), float(_k_loo) * _ref)
    _n0 = int(len(out))
    _pre_ceil = out.copy()
    out = out[_rms_c.notna() & _ref.gt(0) & _rms_c.le(_star_ceil)].copy()
    logging.info(
        "[COMP] loo ceiling k=%.3f abs=%.4f: %d -> %d under-ceiling candidates",
        float(_k_loo),
        float(_abs_ceil),
        _n0,
        int(len(out)),
    )
    if _n0 > int(len(out)):
        _ceil_dropped = _pre_ceil.loc[~_pre_ceil.index.isin(out.index)]
        if not _ceil_dropped.empty:
            _bc = _ceil_dropped.sort_values("comp_rms", kind="mergesort").iloc[0]
            _cidc = "catalog_id" if "catalog_id" in _ceil_dropped.columns else _ceil_dropped.columns[0]
            logging.info(
                "[COMP] best ceiling-rejected cid=%s rms=%s threshold=%.4f",
                str(_bc.get(_cidc, "")),
                str(_bc.get("comp_rms", "")),
                float(_abs_ceil),
            )
    if out.empty:
        logging.warning(
            "[COMP] no candidates under loo ceiling k=%.3f abs=%.4f after floor filter",
            float(_k_loo),
            float(_abs_ceil),
        )
        return out

    # Single-source (COMP-ASSIGN-03): exclude blends inside CoG isolation radius.
    _nn_fwhm = None
    if "_nn_dist_fwhm" in out.columns:
        _nn_fwhm = pd.to_numeric(out["_nn_dist_fwhm"], errors="coerce")
    elif "_nn_px" in out.columns and fwhm_px is not None:
        try:
            _fw = float(fwhm_px)
        except (TypeError, ValueError):
            _fw = float("nan")
        if math.isfinite(_fw) and _fw > 0:
            _nn_fwhm = pd.to_numeric(out["_nn_px"], errors="coerce") / _fw
    if _nn_fwhm is not None:
        out = out.copy()
        if "_nn_dist_fwhm" not in out.columns:
            out["_nn_dist_fwhm"] = _nn_fwhm
        _n_pre = int(len(out))
        _pre_iso = out.copy()
        _ok = _nn_fwhm.notna() & (_nn_fwhm >= float(_iso_fwhm))
        # Missing NN: keep (unknown), only drop measured blends.
        _keep = _nn_fwhm.isna() | _ok
        out = out[_keep].copy()
        logging.info(
            "[COMP] single-source isolation >=%.2f FWHM: %d -> %d candidates",
            float(_iso_fwhm),
            _n_pre,
            int(len(out)),
        )
        if _n_pre > int(len(out)):
            _iso_dropped = _pre_iso.loc[~_pre_iso.index.isin(out.index)]
            if not _iso_dropped.empty:
                _nn_d = pd.to_numeric(_iso_dropped.get("_nn_dist_fwhm"), errors="coerce")
                if _nn_d is not None and _nn_d.notna().any():
                    _iso_dropped = _iso_dropped.assign(_nn_sort=_nn_d)
                    _bi = _iso_dropped.sort_values(
                        ["comp_rms", "_nn_sort"],
                        ascending=[True, True],
                        kind="mergesort",
                    ).iloc[0]
                    _nnv = float(pd.to_numeric(_bi.get("_nn_dist_fwhm"), errors="coerce"))
                else:
                    _bi = _iso_dropped.sort_values("comp_rms", kind="mergesort").iloc[0]
                    _nnv = float("nan")
                _cid_iso = (
                    "catalog_id" if "catalog_id" in _iso_dropped.columns else _iso_dropped.columns[0]
                )
                logging.info(
                    "[COMP] best isolation-rejected cid=%s rms=%s nn_fwhm=%s threshold=%.2f FWHM",
                    str(_bi.get(_cid_iso, "")),
                    str(_bi.get("comp_rms", "")),
                    f"{_nnv:.3f}" if math.isfinite(_nnv) else "nan",
                    float(_iso_fwhm),
                )
        if out.empty:
            logging.warning(
                "[COMP] no candidates after single-source isolation (%.2f FWHM)",
                float(_iso_fwhm),
            )
            return out

    id_col, ladder, ladder_step, selected, tb, used_lim = _sel_colour_ladder(
        _n_min=_n_min,
        cfg=cfg,
        max_delta_bprp=max_delta_bprp,
        out=out,
        target_bprp=target_bprp,
    )

    # D-RED-TARGET-T4-01 (i-a): last rung (cap) n < n_comp_min and finite
    # target BP-RP -> T4_FALLBACK on the same post-ceiling / post-isolation
    # set, ranked by |dBP-RP|. Quality chain is the SNR floor (abs RMS
    # ceiling 0.080 mag ~ SNR 13.6 / epoch). NaN-BP-RP keeps the legacy
    # relax-to-full-set path (_delta_bprp_abs=0.0 bypass above).
    color_fallback = False
    if math.isfinite(tb) and len(selected) < _n_min:
        n_ladder = int(len(selected))
        if len(out) >= _n_min:
            selected = out.copy()
            color_fallback = True
            used_lim = float("inf")
            ladder_step = int(len(ladder)) + 1
            logging.info(
                "[COMP] T4_FALLBACK: last rung n=%d < n_comp_min=%d; "
                "quality set n=%d ranked by |dBP-RP| (SNR floor = RMS ceiling)",
                n_ladder,
                _n_min,
                int(len(selected)),
            )
        else:
            logging.warning(
                "[COMP] T4_FALLBACK still n=%d < n_comp_min=%d -> no_comps",
                int(len(out)),
                _n_min,
            )
            empty = out.iloc[0:0].copy()
            empty.attrs["color_fallback"] = True
            empty.attrs["color_ladder_lim"] = float("nan")
            empty.attrs["color_ladder_step"] = int(len(ladder)) + 1
            empty.attrs["max_delta_bprp_used"] = float("nan")
            empty.attrs["max_comp_rms_ceiling"] = (
                float(_ceil) if math.isfinite(_ceil) and _ceil > 0 else float("nan")
            )
            empty.attrs["single_source_isolation_fwhm"] = float(_iso_fwhm)
            return empty.drop(columns=["_rms_sort", "_dist_sort"], errors="ignore")
    elif not math.isfinite(tb):
        if selected.empty:
            selected = out
            used_lim = float(ladder[-1]) if ladder else float(max_delta_bprp)
            ladder_step = int(len(ladder)) if ladder else 0
            logging.warning(
                "[COMP] color filter relaxed to full under-ceiling single-source set "
                "(no ladder step reached n_comp_min=%d; lim=%.3f; n=%d)",
                _n_min,
                used_lim,
                int(len(selected)),
            )
        elif len(selected) < _n_min and len(out) >= _n_min:
            n_ladder = int(len(selected))
            selected = out
            ladder_step = int(len(ladder)) if ladder else ladder_step
            logging.info(
                "[COMP] color filter relaxed to full under-ceiling single-source set "
                "after ladder (n_ladder=%d < n_comp_min=%d; n_full=%d)",
                n_ladder,
                _n_min,
                int(len(out)),
            )

    # COMP-ASSIGN-03: RMS first, then colour, then distance.
    # T4_FALLBACK: |dBP-RP| first (D-RED-TARGET-T4-01).
    selected, = _sel_rank_and_cap(
        _ceil=_ceil,
        _iso_fwhm=_iso_fwhm,
        _n_max=_n_max,
        color_fallback=color_fallback,
        id_col=id_col,
        ladder_step=ladder_step,
        out=out,
        selected=selected,
        used_lim=used_lim,
    )
    return selected.drop(columns=["_rms_sort", "_dist_sort"], errors="ignore")

def night_frame_fwhms_px(per_frame_csv_paths: Any, store: Any = None) -> list[float]:
    """Per-frame FWHM (px) of THIS observation group: median of ``fwhm_px_for_aperture`` of each per-frame
    CSV (the per-frame QC FWHM stamped on the frame, FWHM-AUTH-01). ``store``: the in-memory proc tables
    (ProcFrameStore / path-keyed cache, G07-16); a file it lacks, or without the column, is read."""
    vals = []
    for p in list(per_frame_csv_paths or []):
        try:
            from proc_frame_store import read_proc_csv  # noqa: PLC0415

            d = store.get(str(p)) if store is not None else None
            if d is None or "fwhm_px_for_aperture" not in d.columns:
                d = read_proc_csv(p, usecols=["fwhm_px_for_aperture"])
            v = float(pd.to_numeric(d["fwhm_px_for_aperture"], errors="coerce").median())
            if math.isfinite(v) and v > 0:
                vals.append(v)
        except Exception:  # noqa: BLE001
            continue
    return vals


def night_median_fwhm_px(
    per_frame_csv_paths: Any, fallback: float, *, store: Any = None, frame_fwhms: list[float] | None = None
) -> tuple[float, str]:
    """Night median frame FWHM (px) of THIS observation group (``night_frame_fwhms_px``; ``frame_fwhms``:
    its result when the caller already has it, G07-16).
    The draft-level qc_metrics.csv is not used here - it mixes the filters of the draft
    (524: g 5.31, r 4.91, mixed 5.14 px). Fallback: ``fallback`` (MASTERSTAR). Returns (fwhm, source)."""
    vals = frame_fwhms if frame_fwhms is not None else night_frame_fwhms_px(per_frame_csv_paths, store)
    if vals:
        return float(np.median(vals)), f"night median of {len(vals)} frames (fwhm_px_for_aperture)"
    logging.warning(
        "[NIGHT-FWHM] no per-frame fwhm_px_for_aperture - fallback FWHM %.2f px used "
        "(MASTERSTAR / QC: best frame or mixed filters, not the group night median)", float(fallback)
    )
    return float(fallback), "fallback (no per-frame FWHM)"


def apply_comp_quality_gates(pool: pd.DataFrame, *, comp_peak_sat_frac: float, label: str = "COMP") -> pd.DataFrame:
    """COMP-SAT-80-2026-10 / COMP-SNR-20-2026-10 (Milan 2026-10-12): comps must be safely linear
    (MASTERSTAR raw peak < 0.80 x SAT; targets keep 0.90) and well measured (aperture S/N on
    MASTERSTAR >= COMP_SNR_MIN; AAVSO CCD/CMOS guide: S/N 100 excellent, < 10-20 poor).
    Applied to the global pool AND to the masterstars fallback (one rule)."""
    from pipeline_constants import COMP_SNR_MIN, LINEARITY_FRAC  # noqa: PLC0415

    _n0 = int(len(pool))
    if "peak_max_adu" in pool.columns and "saturate_limit_adu_85pct" in pool.columns:
        _pk = pd.to_numeric(pool["peak_max_adu"], errors="coerce")
        # saturate_limit_adu_85pct == LINEARITY_FRAC x SAT -> the comp limit comp_peak_sat_frac x SAT
        _lim = pd.to_numeric(pool["saturate_limit_adu_85pct"], errors="coerce") * (
            float(comp_peak_sat_frac) / float(LINEARITY_FRAC))
        pool = pool.loc[~((_lim > 0) & (_pk >= _lim))].copy()
    _n1 = int(len(pool))
    if "snr_ap_pixscaled" in pool.columns:
        _snr = pd.to_numeric(pool["snr_ap_pixscaled"], errors="coerce")
        pool = pool.loc[_snr >= float(COMP_SNR_MIN)].copy()
    logging.info(
        "[%s] linearity %.2f x SAT removed %d, S/N < %.0f removed %d (of %d)",
        label, float(comp_peak_sat_frac), _n0 - _n1, float(COMP_SNR_MIN), _n1 - int(len(pool)), _n0,
    )
    return pool


def _stamp_comp_pool_admission(photometry_dir: Path | str, rules: list[dict[str, Any]]) -> None:
    """The per-rule (n_in, n_out) record of the comparison pool in ``pipeline_meta["comp_pool_admission"]``
    (the only copy; variability_field reads the field aperture k from it)."""
    from photometry_provenance import merge_photometry_pipeline_meta  # noqa: PLC0415

    def _default(o: Any) -> Any:
        if isinstance(o, float) and not math.isfinite(o):
            return None
        if hasattr(o, "item"):
            try:
                return o.item()
            except (TypeError, ValueError):
                pass
        return str(o)

    clean = json.loads(json.dumps({"rules": list(rules)}, ensure_ascii=True, default=_default))
    merge_photometry_pipeline_meta(Path(photometry_dir), {"comp_pool_admission": clean})


def _pool_geometry_gates(
    *,
    cfg,
    chip_fh,
    chip_fw,
    fwhm_px,
    margin,
    photometry_dir_for_meta,
    pool,
    safe_bbox,
    variable_target_catalog_ids,
):
    """Static pool gates: safe bbox / chip margin, variable targets, saturated / VSX rows, D3 candidacy."""
    xn = pd.to_numeric(pool["x"], errors="coerce")
    yn = pd.to_numeric(pool["y"], errors="coerce")

    if safe_bbox is not None:
        # safe_bbox already shrinks by alignment intersection + sky annulus (r_out); do not inset again.
        from aperture_policy import stars_fit_on_chip  # noqa: PLC0415

        x0, y0, x1, y1 = safe_bbox
        if float(x1) > float(x0) and float(y1) > float(y0):
            _on = stars_fit_on_chip(
                xn, yn, (0.0, 0.0, 0.0), (float(x0), float(y0), float(x1), float(y1))
            )
            pool = pool.loc[np.asarray(_on, dtype=bool)].copy()
        else:
            pool = pool.iloc[0:0].copy()
    else:
        fw, fh = int(chip_fw), int(chip_fh)
        if margin > 0 and fw > 2 * margin and fh > 2 * margin:
            pool = pool.loc[
                xn.between(float(margin), float(fw - margin)) & yn.between(float(margin), float(fh - margin))
            ].copy()

    _vt_gaia_ids: frozenset[str] | None = None
    if variable_target_catalog_ids:
        from gaia_catalog_id import normalize_gaia_id_set  # noqa: PLC0415

        _vt_gaia_ids = normalize_gaia_id_set(
            variable_target_catalog_ids,
            log_label="variable_target_catalog_ids (global comp pool)",
        ) or None
    if _vt_gaia_ids:
        nid = pool["catalog_id"].map(_normalize_gaia_id)
        pool = pool.loc[~nid.isin(_vt_gaia_ids)].copy()

    if "zone" in pool.columns:
        z = pool["zone"].astype(str).str.strip().str.lower()
        pool = pool.loc[~z.eq("saturated")].copy()

    # FORCED-PHOT / COMP-ADMIT-03 review: is_noisy is not a gate; is_usable may
    # still encode zone=linear for diagnostics but noisy stars enter via scatter.
    cand_mask = (
        ~coerce_bool_series(pool.get("is_saturated", pd.Series(False, index=pool.index)))
        & ~coerce_bool_series(pool.get("vsx_known_variable", pd.Series(False, index=pool.index)))
        & ~coerce_bool_series(pool.get("likely_saturated", pd.Series(False, index=pool.index)))
    )
    pool = pool.loc[cand_mask].copy()
    pool = apply_comp_quality_gates(pool, comp_peak_sat_frac=float(cfg.comp_peak_sat_frac), label="GLOBAL COMP POOL")

    from d3_comparison_candidacy import apply_d3_comparison_candidacy  # noqa: PLC0415

    _solve_rms_d3: float | None = None
    if photometry_dir_for_meta is not None:
        try:
            _pm = Path(photometry_dir_for_meta)
            _meta_p = _pm if _pm.name == "pipeline_meta.json" else _pm / "pipeline_meta.json"
            if _meta_p.is_file():
                _pj = json.loads(_meta_p.read_text(encoding="utf-8"))
                _inp = _pj.get("match_sep_formula_inputs") or {}
                if isinstance(_inp, dict) and _inp.get("solve_rms_px") is not None:
                    _solve_rms_d3 = float(_inp["solve_rms_px"])
        except Exception:  # noqa: BLE001
            _solve_rms_d3 = None
    _d3_mask, _ = apply_d3_comparison_candidacy(
        pool,
        fwhm_dao_px=float(fwhm_px),
        solve_rms_px=_solve_rms_d3,
        log_label="global_comp_pool",
        snr_min=float(getattr(cfg, "snr_min", 10.0)),
    )
    pool = pool.loc[_d3_mask].copy()
    return pool,


def _pool_admission_gates(
    *,
    _fstats,
    gain,
    read_noise,
    scint_params,
    admission_rules_band,
    admission_rules_blend,
    admission_rules_sat,
    cand_ids,
    csv_cache,
    flux_col,
    id_col,
    min_frames_frac,
    per_frame_csv_paths,
    photometry_dir_for_meta,
    pool,
):
    """Aperture-k RMS (LOO), frames per comp, Gaia DR3 known variables."""
    admission_rules: list[dict[str, Any]] = []
    if admission_rules_sat is not None:
        admission_rules.append(admission_rules_sat)
    for _r_extra in (admission_rules_band, admission_rules_blend):
        if _r_extra is not None:
            admission_rules.append(_r_extra)

    # PHASE1-ONE-PASS-2026-10: the per-frame pass of the saturation rule also gives the frames measured per comp
    # APERTURE-K-2026-10: comparison RMS on the light-curve aperture grid (field k);
    # falls back to ``flux_col`` for per-frame catalogs written before the grid columns.
    from comp_rms_loo import choose_field_aperture_k  # noqa: PLC0415

    from comp_rms_loo import photon_mag_err_map  # noqa: PLC0415
    from photometry_defs import flux_k_column  # noqa: PLC0415

    _k_field, _loo_k, _k_scores, _k_info = choose_field_aperture_k(
        cand_ids, per_frame_csv_paths, csv_cache, min_frames_frac=min_frames_frac
    )
    if _k_field is not None:
        loo_map = _loo_k
        _ph_col = flux_k_column(_k_field)
        admission_rules.append(
            {
                "rule_id": "COMP_RMS_APERTURE_K",
                "n_in": int(len(pool)),
                "n_out": int(len(pool)),
                "threshold_value": float(_k_field),
                "unit": "k (r = k x FWHM)",
                "scores": {f"{k:.2f}": float(v) for k, v in _k_scores.items()},
                # COMP-CEIL-FIELDK-2026-10: the k rule (AP-BRIGHT-K) and its bootstrap SE
                "rule": _k_info["rule"],
                "ratio_curve": _k_info["ratio_curve"],
                "se": _k_info["se"],
                "n_stars": _k_info["n_stars"],
                # AP-BRIGHT-K diagnostic (Milan 2026-10-30): every k within one SE of the minimum
                "k_within_se": _k_info["k_within_se"],
            }
        )
    else:
        _ph_col = flux_col
        loo_map, _basis = compute_loo_mag_rms_map(
            cand_ids=cand_ids,
            per_frame_csv_paths=per_frame_csv_paths,
            csv_cache=csv_cache,
            flux_col=flux_col,
            min_frames_frac=min_frames_frac,
        )
    pool = attach_comp_rms_to_pool_rows(
        pool,
        loo_map,
        id_col=id_col,
        frames_basis=COMP_RMS_FRAMES_BASIS,
    )
    # COMP-CEIL-FIELDK-2026-10: photon reference of the comp_rms ceiling, same aperture and frames as comp_rms
    _ph_map = photon_mag_err_map(
        cand_ids, per_frame_csv_paths, csv_cache, _ph_col, k=_k_field, gain=gain, read_noise=read_noise
    )
    pool[COMP_PHOTON_MAG_COL] = (
        _lookup_rms_map(_ph_map, pool[id_col].astype(str).str.strip()).astype(float)
        if _ph_map else np.nan
    )
    _n_no_ph = int((pool["comp_rms"].notna() & pool[COMP_PHOTON_MAG_COL].isna()).sum())
    if _n_no_ph:
        logging.warning(
            "[GLOBAL COMP POOL] %d stars with comp_rms have no CCD-equation photon error at %s "
            "(no sky / aperture in their frames) - they cannot pass the comp_rms ceiling", _n_no_ph, _ph_col,
        )
    admission_rules.append(
        {
            "rule_id": "COMP_PHOTON_REF",
            "n_in": int(len(pool)),
            "n_out": int(len(pool)),
            "threshold_value": _ph_col,
            "unit": "flux column of the ceiling photon error (median over frames, CCD equation)",
            "n_without_photon_err": _n_no_ph,
            "gain": float(gain),
            "read_noise": float(read_noise),
        }
    )
    # COMP-CEIL-NOISEMODEL-2026-10 (Milan 2026-10-30): scintillation of each star on its comp_rms frames and the
    # ceiling reference sigma_ref with a / floor fitted on the constant pool stars at the field k
    from comp_rms_loo import comp_ceiling_reference, scint_mag_err_map  # noqa: PLC0415

    _sc_map, _sc_info = scint_mag_err_map(cand_ids, per_frame_csv_paths, csv_cache, _ph_col, rig=scint_params)
    if _sc_info["n_frames_no_airmass"]:
        logging.warning(
            "[GLOBAL COMP POOL] %d of %d frames without an airmass: scintillation 0 there (as in the "
            "light-curve errors)", _sc_info["n_frames_no_airmass"], _sc_info["n_frames"],
        )
    pool[COMP_SCINT_MAG_COL] = (
        _lookup_rms_map(_sc_map, pool[id_col].astype(str).str.strip()).astype(float)
        if _sc_map else np.nan
    )
    _const = np.ones(len(pool), dtype=bool)
    for _f in ("vsx_known_variable", "gaia_dr3_variable_catalog", "is_saturated", "likely_saturated"):
        if _f in pool.columns:
            _const &= ~coerce_bool_series(pool[_f]).to_numpy()
    _sref, _ref_info = comp_ceiling_reference(
        pd.to_numeric(pool["comp_rms"], errors="coerce").to_numpy(dtype=np.float64),
        pd.to_numeric(pool[COMP_PHOTON_MAG_COL], errors="coerce").to_numpy(dtype=np.float64),
        pd.to_numeric(pool[COMP_SCINT_MAG_COL], errors="coerce").to_numpy(dtype=np.float64),
        _const,
    )
    pool[COMP_SIGMA_REF_COL] = _sref
    admission_rules.append(
        {
            "rule_id": "COMP_CEIL_NOISE_MODEL",
            "n_in": int(len(pool)),
            "n_out": int(len(pool)),
            "threshold_value": float(COMP_RMS_LOO_PHOTON_K_DEFAULT),
            "unit": f"comp_rms <= min(abs cap, multiplier x {COMP_SIGMA_REF_COL}) (per-target selection)",
            "rule": "COMP-CEIL-NOISEMODEL-2026-10",
            **_ref_info,
            # rig values as the light-curve errors use them; None = not known (scintillation 0)
            "scintillation": {
                **_sc_info,
                **{_n: (float(_v) if math.isfinite(float(_v)) else None) for _n, _v in (
                    ("telescope_diameter_m", scint_params.telescope_diameter_m),
                    ("altitude_m", scint_params.altitude_m),
                    ("exposure_s", scint_params.exposure_s),
                    ("c_y", scint_params.c_y),
                )},
            },
        }
    )
    _keys_fs = pool[id_col].astype(str).str.strip()
    pool["comp_n_frames"] = _keys_fs.map(lambda c: int((_fstats.get(c) or {}).get("n_frames", 0)))

    # COMP-ADMIT-2026-10: known variables are never comparison stars - VSX (coordinate match,
    # removed above with the static gates) and the Gaia DR3 variability catalogue. This replaces the
    # derived admission (COMP-POOL-01), whose only decision was this same catalogue test: its noise
    # curve, thresholds and dilution were diagnostics nobody read, computed with gain 1 / read
    # noise 0 / plate scale 1.3"/px from the raw config, and cost about 40 s per setup.
    _before_var = int(len(pool))
    _var = coerce_bool_series(pool.get("gaia_dr3_variable_catalog", pd.Series(False, index=pool.index)))
    pool = pool.loc[~_var].copy()
    admission_rules.append(
        {
            "rule_id": "COMP_KNOWN_VARIABLE",
            "n_in": _before_var,
            "n_out": int(len(pool)),
            "threshold_value": "vsx_known_variable or gaia_dr3_variable_catalog",
            "unit": "catalogue flag",
        }
    )
    if photometry_dir_for_meta is not None:
        _stamp_comp_pool_admission(photometry_dir_for_meta, admission_rules)

    _before_dedupe = int(len(pool))
    pool = _dedupe_comp_pool_by_gaia_key(pool)
    if int(len(pool)) < _before_dedupe:
        logging.info(
            "[GLOBAL COMP POOL] deduped Gaia catalog_id: %d -> %d rows",
            _before_dedupe,
            int(len(pool)),
        )
    return pool,


def build_global_comp_pool(
    masterstars_df: pd.DataFrame,
    per_frame_csv_paths: list[Path],
    csv_cache: dict[str, pd.DataFrame],
    variable_target_catalog_ids: AbstractSet[str] | None,
    safe_bbox: tuple[float, float, float, float] | None,
    chip_fw: int,
    chip_fh: int,
    chip_interior_margin_px: int,
    cfg: AppConfig,
    *,
    flux_col: str = "dao_flux",
    min_frames_frac: float,
    fwhm_px: float,
    photometry_dir_for_meta: Path | str | None = None,
    obs_group: str | None = None,
    masterstar_fits: Path | str | None = None,
    field_catalog_csv: Path | str | None = None,
    sat_may_exclude: bool = True,
    gain: float,
    read_noise: float,
    scint_params: Any,
) -> pd.DataFrame:
    """Build the global comp pool - static filters + RMS across frames (once per field).

    ``gain`` [e-/ADU] / ``read_noise`` [e-]: the run's photometric gain authority
    (``rig_photometry.photometric_gain_read_noise``), for the CCD-equation photon error of the
    comp_rms ceiling (COMP-CEIL-FIELDK-2026-10); required, no default. ``scint_params``: the run's
    ``sigma_budget.RigScintillationParams`` (``aperture_photometry.rig_scint_params_for_run``), for the
    scintillation term of the ceiling reference (COMP-CEIL-NOISEMODEL-2026-10); required, no default."""
    pool = masterstars_df.copy()
    if "catalog_id" in pool.columns:
        pool["catalog_id"] = _normalize_id_series(pool["catalog_id"])
    for col in (
        "is_usable",
        "is_saturated",
        "is_noisy",
        "snr50_ok",
        "vsx_known_variable",
        "likely_saturated",
    ):
        if col in pool.columns:
            pool[col] = coerce_bool_series(pool[col])

    margin = int(chip_interior_margin_px)
    if "x" not in pool.columns or "y" not in pool.columns:
        logging.warning("[GLOBAL COMP POOL] missing x/y - empty pool")
        return pd.DataFrame()
    pool, = _pool_geometry_gates(
        cfg=cfg,
        chip_fh=chip_fh,
        chip_fw=chip_fw,
        fwhm_px=fwhm_px,
        margin=margin,
        photometry_dir_for_meta=photometry_dir_for_meta,
        pool=pool,
        safe_bbox=safe_bbox,
        variable_target_catalog_ids=variable_target_catalog_ids,
    )

    if pool.empty:
        logging.warning("[GLOBAL COMP POOL] 0 rows after static filters")
        return pool.reset_index(drop=True)

    id_col = "catalog_id" if "catalog_id" in pool.columns else "name"  # identity = catalog_id (G07-05)
    cand_ids = {str(x).strip() for x in pool[id_col].tolist() if str(x).strip()}
    if not cand_ids:
        return pool.reset_index(drop=True)

    # COMP-SAT-ONE-2026-10 (Milan 2026-10-29, code review 2 G07-27): one saturation rule for comparison
    # stars - raw peak >= comp_peak_sat_frac x SAT (or flagged likely_saturated) on ANY frame -> not a
    # comparison star and not a LOO reference (COMP-SAT-80-2026-10: "< 0.80 x SAT per frame"). Replaces
    # the >= 0.90 any-frame rule (ungated) and the "> 10 % of frames" rule. Applied only with a measured
    # saturation level (SAT-LIMIT-01: never exclude against an unverified level).
    from comp_pool_rms import _norm_id_val  # noqa: PLC0415

    _frac_c = float(cfg.comp_peak_sat_frac)
    _fstats = comp_frame_stats(cand_ids, per_frame_csv_paths, csv_cache, flux_col=flux_col,
                               comp_peak_sat_frac=_frac_c)
    admission_rules_sat = None
    if sat_may_exclude:
        _over = {_norm_id_val(c) for c, v in _fstats.items() if int(v.get("n_over", 0)) > 0}
        _nid_pool = pool[id_col].map(_norm_id_val)
        _drop = _nid_pool.isin(_over)
        logging.info("[GLOBAL COMP POOL] raw peak >= %.2f x SAT on >= 1 frame: %d stars removed",
                     _frac_c, int(_drop.sum()))
        admission_rules_sat = {
            "rule_id": "COMP_SAT_ANY_FRAME",
            "n_in": int(len(pool)),
            "n_out": int(len(pool) - int(_drop.sum())),
            "threshold_value": _frac_c,
            "unit": "fraction_of_SAT (raw peak, any frame)",
        }
        pool = pool.loc[~_drop].copy()
        cand_ids = {str(x).strip() for x in pool[id_col].tolist() if str(x).strip()}
        if not cand_ids:
            return pool.reset_index(drop=True)
    else:
        logging.info("[SAT-DIAG] comparison saturation exclusion disabled (saturation level unverified)")

    # COMP-BAND-2026-10: no band-validity gate on comparison stars. The light-curve shape
    # comes from the whole ensemble; only the absolute level (zero point) needs catalogue
    # magnitudes in the filter band, and ensemble_normalize takes it from the comps whose
    # Gaia DR3 -> band relation is valid (Honeycutt 1992 ensemble + calibrating subset).
    # The former gate removed 35 of the 76 comps of draft 524 g (BP-RP > 2.0, reddened field).
    admission_rules_band = None
    admission_rules_blend = None
    if obs_group is not None:
        from gaia_bands import filter_band_for_obs_group  # noqa: PLC0415

        _fb = filter_band_for_obs_group(obs_group)

        # BLEND-2026-10: neighbour light in the comparison aperture <= 1 % of the star's flux.
        # BLEND-K-2026-10: the light is computed for every aperture of the grid
        # (r = k x FWHM, columns blend_k060 ... blend_k200, one measured PSF). The target's
        # aperture is chosen later (aperture photometry) only among the k at which all its comps stay
        # <= 1 % (aperture_pertarget.allowed_f_from_comp_blend); the pool keeps a star that
        # passes at the smallest k, so that k is always available. Contamination is defined
        # for the aperture actually used, as Kepler / TESS CROWDSAP.
        from photometry_defs import blend_k_column  # noqa: PLC0415
        from photometry_defs import DEFAULT_APERTURE_F_GRID  # noqa: PLC0415
        from comp_blend import BLEND_FLUX_FRAC_MAX, blend_contamination_for_pool  # noqa: PLC0415

        # BLEND-FWHM-2026-10: the apertures are r = k x FWHM(frame).
        # BLEND-SEEING-2026-10 (Milan 2026-10-28): the light is evaluated at the night's bad-seeing
        # frames (quantile BLEND_FWHM_QUANTILE of the group's frame FWHM): a neighbour outside the
        # aperture at the median seeing moves into it when the FWHM grows (Brno 534: 3.9 -> 8.7 px),
        # which put seeing-shaped trends of 6-100 mmag into "clean" light curves. The Moffat beta is
        # measured once on the MASTERSTAR at its own (night median) FWHM.
        from comp_blend import BLEND_FWHM_QUANTILE, blend_fwhm_px  # noqa: PLC0415

        _fw_frames = night_frame_fwhms_px(per_frame_csv_paths, csv_cache)  # G07-16: from the store, once
        _fw_med, _fw_src = night_median_fwhm_px(per_frame_csv_paths, float(fwhm_px), frame_fwhms=_fw_frames)
        _fw, _fw_src_b = blend_fwhm_px(_fw_frames, _fw_med)
        _ks = sorted(float(k) for k in DEFAULT_APERTURE_F_GRID)
        _ann = (
            float(getattr(cfg, "annulus_inner_fwhm", 2.7)) * _fw,
            float(getattr(cfg, "annulus_outer_fwhm", 5.2)) * _fw,
        )
        _sat_b = float("nan")
        if "saturate_limit_adu" in masterstars_df.columns:
            _sat_b = float(pd.to_numeric(masterstars_df["saturate_limit_adu"], errors="coerce").median())
        _bmeta: dict = {}
        _psf = None
        # G07-12: catalogue, WCS and neighbour fluxes loaded once for the whole k grid
        from comp_blend import BlendField  # noqa: PLC0415

        _bfield = (
            BlendField(Path(field_catalog_csv), Path(masterstar_fits), _fb.blend_band)
            if field_catalog_csv and masterstar_fits and Path(field_catalog_csv).is_file()
            and Path(masterstar_fits).is_file() else None
        )
        for _k in _ks:
            _cont, _bmeta = blend_contamination_for_pool(
                pool,
                field_catalog_csv=Path(field_catalog_csv) if field_catalog_csv else None,
                masterstar_fits=Path(masterstar_fits) if masterstar_fits else None,
                zp_band=_fb.blend_band,
                fwhm_px=_fw,
                profile_fwhm_px=_fw_med,
                r_ap_px=_k * _fw,
                sat_adu=_sat_b,
                annulus_px=_ann,
                profile_df=masterstars_df,
                psf=_psf,
                field=_bfield,
            )
            if _bmeta.get("status") != "ok":
                break
            _psf = _bmeta.get("psf")
            pool[blend_k_column(_k)] = _cont.to_numpy()
        if _bmeta.get("status") == "ok":
            _k_gate = _ks[0]
            _cg = pool[blend_k_column(_k_gate)].to_numpy()
            pool["blend_contamination"] = _cg
            _drop_b = np.abs(_cg) > BLEND_FLUX_FRAC_MAX
            _bmeta.update(
                n_in=int(len(pool)), n_out=int(len(pool) - int(_drop_b.sum())),
                k_grid=_ks, gate_k=_k_gate, r_ap_px=_k_gate * _fw,
                fwhm_px=_fw, fwhm_source=_fw_src_b, fwhm_median_px=_fw_med, fwhm_median_source=_fw_src,
                fwhm_quantile=BLEND_FWHM_QUANTILE,
            )
            logging.info(
                "[GLOBAL COMP POOL] |neighbour light| > %.0f%% at the smallest aperture k=%.2f "
                "(r=%.1f px, Moffat beta=%.2f, %s): %d stars removed; per-k values kept for the "
                "target aperture choice",
                100.0 * BLEND_FLUX_FRAC_MAX, _k_gate, _k_gate * _fw, float(_bmeta["psf"]["beta"]),
                _bmeta["psf"]["source"], int(_drop_b.sum()),
            )
            pool = pool.loc[~_drop_b].copy()
        else:
            # FAIL-LOUD-2026-10 (code review 2 G07-14): the 1 % neighbour-light gate is a comp rule; a pool
            # without it can admit blends, so the selection stops instead of continuing ungated
            raise RuntimeError(f"comparison pool: blend gate not applied ({_bmeta.get('status')})")
        admission_rules_blend = _bmeta
        cand_ids = {str(x).strip() for x in pool[id_col].tolist() if str(x).strip()}
        if not cand_ids:
            return pool.reset_index(drop=True)

    pool, = _pool_admission_gates(
        _fstats=_fstats,
        gain=gain,
        read_noise=read_noise,
        scint_params=scint_params,
        admission_rules_band=admission_rules_band,
        admission_rules_blend=admission_rules_blend,
        admission_rules_sat=admission_rules_sat,
        cand_ids=cand_ids,
        csv_cache=csv_cache,
        flux_col=flux_col,
        id_col=id_col,
        min_frames_frac=min_frames_frac,
        per_frame_csv_paths=per_frame_csv_paths,
        photometry_dir_for_meta=photometry_dir_for_meta,
        pool=pool,
    )
    logging.info(
        "[GLOBAL COMP POOL] %d candidates (of %d masterstars, after filters)",
        len(pool),
        len(masterstars_df),
    )
    # P2 determinism: canonical row order
    if "catalog_id" in pool.columns:
        pool = pool.sort_values("catalog_id", kind="mergesort").reset_index(drop=True)
    return pool

def _dedupe_comp_pool_by_gaia_key(pool: pd.DataFrame) -> pd.DataFrame:
    """One row per Gaia ``catalog_id``; keep lowest ``comp_rms`` when duplicated."""
    if pool is None or getattr(pool, "empty", True):
        return pool if pool is not None else pd.DataFrame()
    out = pool.copy()
    id_src = out["catalog_id"]
    out["_gaia_key"] = id_src.map(_normalize_gaia_id)
    out = out[out["_gaia_key"].astype(str).str.strip() != ""]
    if out.empty:
        return out.reset_index(drop=True)
    sort_cols = ["_gaia_key"]
    ascending = [True]
    if "comp_rms" in out.columns:
        sort_cols.append("comp_rms")
        ascending.append(True)
    if "catalog_id" in out.columns:
        sort_cols.append("catalog_id")
        ascending.append(True)
    out = out.sort_values(sort_cols, ascending=ascending, kind="mergesort")
    out = out.drop_duplicates(subset=["_gaia_key"], keep="first").drop(columns=["_gaia_key"])
    return out.reset_index(drop=True)

def _warn_zero_compstars_edge(
    *,
    target_cid: str,
    target: pd.Series,
    chip_fw: int | None,
    chip_fh: int | None,
    chip_interior_margin_px: int,
) -> None:
    """On a failed comp selection (0 rows) - if the target is near the inner chip edge, add context."""
    try:
        tx = float(pd.to_numeric(target.get("x"), errors="coerce"))
        ty = float(pd.to_numeric(target.get("y"), errors="coerce"))
    except Exception:  # noqa: BLE001
        tx = ty = float("nan")
    if not (math.isfinite(tx) and math.isfinite(ty)):
        logging.warning("[COMP] %s: 0 comp stars", target_cid)
        return
    m = int(chip_interior_margin_px)
    if chip_fw is None or chip_fh is None or m <= 0:
        logging.warning("[COMP] %s: 0 comp stars", target_cid)
        return
    wf = float(int(chip_fw))
    hf = float(int(chip_fh))
    if wf <= 2.0 * float(m) or hf <= 2.0 * float(m):
        logging.warning("[COMP] %s: 0 comp stars", target_cid)
        return
    xmin = float(m)
    ymin = float(m)
    xmax = wf - float(m)
    ymax = hf - float(m)
    dist = min(tx - xmin, xmax - tx, ty - ymin, ymax - ty)
    if math.isfinite(dist) and dist < 100.0:
        logging.warning(
            "[COMP] %s: 0 comp stars, target is %.0fpx from the bbox edge "
            "(edge position - geometrically limited field)",
            target_cid,
            float(dist),
        )
    else:
        logging.warning("[COMP] %s: 0 comp stars", target_cid)

def _count_gate_passing_comps(
    result: pd.DataFrame | None,
    per_target_rms_map: dict[str, float] | None,
    max_comp_rms: float,
    id_col: str,
) -> int:
    """Count comps in ``result`` whose per-target ``comp_rms`` passes the gate.

    N_good = comps passing the colour ladder + per-target ``max_comp_rms`` gate.
    The per-target gate is authoritative: a comp with ``comp_rms > max_comp_rms``
    is never counted as good (known-issue (b) fix). When the gate is disabled
    (non-finite / <= 0) fall back to the raw row count.
    """
    if result is None or getattr(result, "empty", True):
        return 0
    if not (math.isfinite(max_comp_rms) and max_comp_rms > 0):
        return int(len(result))
    if id_col not in result.columns:
        return int(len(result))
    _map = per_target_rms_map or {}
    n_good = 0
    for _rid in result[id_col].astype(str).str.strip():
        _v = _map.get(_rid, _map.get(str(_rid), float("nan")))
        try:
            _vf = float(_v)
        except (TypeError, ValueError):
            _vf = float("nan")
        if math.isfinite(_vf) and _vf <= float(max_comp_rms):
            n_good += 1
    return n_good

def select_comparison_stars_per_target(
    target: pd.Series,
    masterstars_df: pd.DataFrame,
    *,
    global_comp_pool_df: pd.DataFrame | None = None,
    fwhm_px: float,
    n_comp_min: int,
    n_comp_max: int,
    max_comp_rms: float,
    min_dist_arcsec: float = 60.0,
    chip_fw: int | None = None,
    chip_fh: int | None = None,
    chip_interior_margin_px: int = 0,
    gaia_db_path: str | None = None,
    gaia_prefetch: dict[str, dict[str, Any]] | None = None,
    variable_target_catalog_ids: AbstractSet[str] | None = None,
    cfg: AppConfig,
    n_reserve: int = 0,
) -> pd.DataFrame:
    """Comparison-star selection: pick the most stable comparison stars for one target.

    COMP-SMART-2026-10: ``n_reserve`` > 0 additionally returns the next candidates of a
    separate selection with ``n_comp_max + n_reserve`` (rows with ``comp_role="reserve"`` and
    ``reserve_rank``); the admitted comps are those of the unchanged ``n_comp_max`` selection.

    Procedure (option D = B + C):
    1. Spatial + photometric filter of candidates from masterstars
    2. Load flux from all per-frame CSVs (only PHASE_USECOLS_PERFRAME)
    3. Normalize flux to the ensemble median per frame
    4. Compute the RMS scatter of each candidate
    5. Iterative ensemble filter - drop the top outliers until RMS stops decreasing
    6. Return the top n_comp_max most stable (min n_comp_min)

    Args:
        max_comp_rms: Maximum allowed p2p RMS scatter of a comp star (mag).
            Stars with RMS > max_comp_rms are rejected regardless of ranking.
            Default 0.05 mag (50 ppt) - standard photometric stability.
        min_dist_arcsec: Minimum distance of a comp star from the target in arcseconds.
            Prevents PSF overlap for very close stars.
            Default 60 arcsec (also protects against local artefacts around the target).
        chip_fw / chip_fh / chip_interior_margin_px: together they cut comp candidates
            near the chip edge (same logic as target selection and suspected variables).
            ``chip_interior_margin_px=0`` = off.
        variable_target_catalog_ids: Gaia ``catalog_id`` from ``variable_targets.csv`` - these stars
            are never offered as comparison stars (VSX variables including ``catalog_only``).

    Returns:
        DataFrame of comparison stars for this target, sorted by RMS ASC.
        Empty DataFrame if not enough stable stars are found.
    """
    from comp_selection_per_target import (  # noqa: PLC0415
        _assemble_comp_selection_result_rows,
        _assign_comp_tiers_to_pool,
        _build_candidates_pre,
        _ensemble_mad_filter_rms,
        _filter_comp_candidates_spatial_static,
        _resolve_target_color_for_comp_selection,
        _score_comp_candidates_broeg,
    )

    _cfg_p1 = cfg

    if global_comp_pool_df is None or getattr(global_comp_pool_df, "empty", True):
        # COMP-POOL-ONLY-2026-10 (Milan 2026-10-26, D5): comparison stars come only from the pool
        # that passed every rule (Gaia/VSX variables, blend, saturation, D3, frames); an empty pool
        # means no comparison star for any target - never ungated MASTERSTAR stars.
        logging.warning("[COMP] %s: no comparison star passed the pool rules - no comps", target.get("catalog_id", ""))
        return pd.DataFrame()
    # COMP-ASSIGN-01 D2: keep step-1 pool ``comp_rms`` (do not drop / re-derive).
    ms = global_comp_pool_df.copy()
    if "catalog_id" in ms.columns:
        ms["catalog_id"] = _normalize_id_series(ms["catalog_id"])
    for col in ("is_saturated", "vsx_known_variable", "likely_saturated"):
        if col in ms.columns:
            ms[col] = coerce_bool_series(ms[col])

    ctx = _resolve_target_color_for_comp_selection(target, cfg=_cfg_p1)
    ra_t = float(ctx["ra_t"])
    dec_t = float(ctx["dec_t"])
    mag_t = float(ctx["mag_t"])
    target_cid = str(ctx["target_cid"])
    target_bprp_eff = float(ctx["target_bprp_eff"])
    _individual_tier = ctx["_individual_tier"]
    _target_name = str(ctx["_target_name"])

    ms, _base_mask = _filter_comp_candidates_spatial_static(
        ms,
        ra_t=ra_t,
        dec_t=dec_t,
        target_cid=target_cid,
        min_dist_arcsec=min_dist_arcsec,
        variable_target_catalog_ids=variable_target_catalog_ids,
    )

    candidates_pre = _build_candidates_pre(
        ms,
        _base_mask=_base_mask,
        mag_t=mag_t,
        target_cid=target_cid,
        n_comp_min=n_comp_min,
        chip_fw=chip_fw,
        chip_fh=chip_fh,
        chip_interior_margin_px=int(chip_interior_margin_px),
        target=target,
    )
    if candidates_pre is None:
        return pd.DataFrame()

    _r_ap_iso = 7.0
    try:
        _fw = float(fwhm_px)
        if math.isfinite(_fw) and _fw > 0:
            _r_ap_iso = float(2.75 * _fw)
    except (TypeError, ValueError):
        _r_ap_iso = 7.0
    if not (math.isfinite(_r_ap_iso) and _r_ap_iso > 0):
        _r_ap_iso = 7.0
    try:
        # Full catalogue for single-source NN (global pool may omit the neighbour).
        _field_src = masterstars_df if masterstars_df is not None else ms
        ms_arr_x = pd.to_numeric(_field_src.get("x", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
        ms_arr_y = pd.to_numeric(_field_src.get("y", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
        if "_mag" in _field_src.columns:
            ms_arr_mag = pd.to_numeric(_field_src["_mag"], errors="coerce").to_numpy(dtype=float)
        else:
            ms_arr_mag = pd.to_numeric(_field_src.get("mag", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
    except Exception as _iso_exc:  # noqa: BLE001
        logging.warning(f"[COMP-SELECT] Aperture isolation skipped (error): {_iso_exc!s}")
        ms_arr_x = ms_arr_y = ms_arr_mag = np.array([], dtype=float)

    # PHASE1-ONE-PASS-2026-10 / COMP-ASSIGN-01 D2: the global pool carries the one comp RMS (LOO, comp_rms),
    # frames per comp and the per-frame gates. G07-01 residue: no per-target re-derivation from the proc
    # CSVs with another statistic - a pool without a finite comp_rms gives no comparison stars.
    _idc_pool = "catalog_id" if "catalog_id" in ms.columns else "name"
    _pool_ids = ms[_idc_pool].map(lambda v: str(v).strip() if pd.notna(v) else "").tolist()
    _rms_vals = pd.to_numeric(ms["comp_rms"], errors="coerce").tolist() if "comp_rms" in ms.columns else []
    rms_map: dict[str, float] = {
        c: float(v) for c, v in zip(_pool_ids, _rms_vals, strict=False) if c and math.isfinite(float(v))
    }
    if not rms_map:
        logging.warning("[COMP-ASSIGN] Target %s: no pool comp has a comp_rms - no comparison stars", target_cid)
        return pd.DataFrame()
    _nf = pd.to_numeric(ms["comp_n_frames"], errors="coerce").fillna(0).astype(int).tolist() if "comp_n_frames" in ms.columns else []
    comp_n_frames_map = {c: int(v) for c, v in zip(_pool_ids, _nf, strict=False) if c}
    logging.info("[COMP-ASSIGN] Target %s: pool comp_rms for %d comps", target_cid, len(rms_map))

    def _apply_aperture_isolation_safe(cands: pd.DataFrame) -> pd.DataFrame:
        if cands.empty:
            return cands
        try:
            ms_arr_x2 = ms_arr_x
            ms_arr_y2 = ms_arr_y
            ms_arr_mag2 = ms_arr_mag
        except Exception as exc:  # noqa: BLE001
            logging.error('[EXC-0205] Aperture isolation filter skipped when ms_arr arrays unavailable - crowded comps not re...: %s', exc)
            return cands
        rej: set[Any] = set()
        # SPEED-2026-10: column lists instead of iterrows (same values, same order).
        _nc = len(cands)
        _nan_l = [float("nan")] * _nc
        _x_l = cands["x"].tolist() if "x" in cands.columns else _nan_l
        _y_l = cands["y"].tolist() if "y" in cands.columns else _nan_l
        _m_l = cands["_mag"].tolist() if "_mag" in cands.columns else _nan_l
        for idx2, _xv, _yv, _mv in zip(cands.index.tolist(), _x_l, _y_l, _m_l, strict=True):
            cx2 = float(_xv)
            cy2 = float(_yv)
            cm2 = float(_mv)
            if not (math.isfinite(cx2) and math.isfinite(cy2) and math.isfinite(cm2)):
                continue
            d2 = np.sqrt((ms_arr_x2 - cx2) ** 2 + (ms_arr_y2 - cy2) ** 2)
            in_ap2 = (d2 < float(_r_ap_iso)) & (d2 > 1e-6)
            if not bool(np.any(in_ap2)):
                continue
            nm2 = ms_arr_mag2[in_ap2]
            sig2 = nm2[np.isfinite(nm2) & (np.abs(nm2 - cm2) < 3.0)]
            if int(sig2.size) > 0:
                rej.add(idx2)
        if not rej:
            return cands
        after = int(len(cands) - len(rej))
        if after >= int(n_comp_min):
            return cands[~cands.index.isin(rej)]
        return cands

    candidates = ms[_base_mask].copy()
    if candidates.empty:
        logging.warning(f"[COMP-SELECT] {target_cid}: no candidates after hard filters")
        _warn_zero_compstars_edge(
            target_cid=target_cid,
            target=target,
            chip_fw=chip_fw,
            chip_fh=chip_fh,
            chip_interior_margin_px=int(chip_interior_margin_px),
        )
        return pd.DataFrame()
    candidates = _apply_aperture_isolation_safe(candidates)

    active = _ensemble_mad_filter_rms(
        rms_map,
        candidates,
        target_cid=target_cid,
        target=target,
        chip_fw=chip_fw,
        chip_fh=chip_fh,
        chip_interior_margin_px=int(chip_interior_margin_px),
    )
    if active is None:
        return pd.DataFrame()

    id_col_cand = "catalog_id" if "catalog_id" in candidates.columns else "name"  # identity = catalog_id (G07-05)
    score_map, tier_map = _score_comp_candidates_broeg(
        active,
        candidates,
        id_col_cand=id_col_cand,
        target_bprp_eff=target_bprp_eff,
        _individual_tier=_individual_tier,
        cfg=_cfg_p1,
    )

    tier_out = _assign_comp_tiers_to_pool(
        candidates,
        active,
        id_col_cand=id_col_cand,
        target=target,
        target_cid=target_cid,
        target_bprp_eff=target_bprp_eff,
        mag_t=mag_t,
        _individual_tier=_individual_tier,
        _target_name=_target_name,
        gaia_db_path=gaia_db_path,
        gaia_prefetch=gaia_prefetch,
        n_comp_min=n_comp_min,
        # G07-09: one ranking for the admitted set and the reserve (COMP-SMART-2026-10); the admitted set
        # is its head (the n_comp_max selection), the reserve the next n_reserve candidates
        n_comp_max=int(n_comp_max) + max(0, int(n_reserve)),
        n_primary=int(n_comp_max) if int(n_reserve) > 0 else None,
        chip_fw=chip_fw,
        chip_fh=chip_fh,
        chip_interior_margin_px=int(chip_interior_margin_px),
        cfg=_cfg_p1,
        max_comp_rms=float(max_comp_rms),
        fwhm_px=float(fwhm_px) if fwhm_px is not None else None,
        field_x=ms_arr_x,
        field_y=ms_arr_y,
    )
    tier_wide = tier_out
    if tier_out.get("primary") is not None:
        tier_out = tier_out["primary"]
    final_comps = tier_out["final_comps"]
    if final_comps is None or getattr(final_comps, "empty", True):
        return pd.DataFrame()

    result = _assemble_comp_selection_result_rows(
        tier_out["selected_ids"],
        final_comps,
        id_col_cand=id_col_cand,
        active=active,
        score_map=score_map,
        comp_n_frames_map=comp_n_frames_map,
        target_cid=target_cid,
        target=target,
        target_bprp_eff=target_bprp_eff,
        sel_note=str(tier_out["sel_note"]),
        best_tier=str(tier_out["best_tier"]),
        tier4_warning=bool(tier_out["tier4_warning"]),
        n_t1=int(tier_out["n_t1"]),
        n_t2=int(tier_out["n_t2"]),
        n_t3=int(tier_out["n_t3"]),
        n_t4=int(tier_out["n_t4"]),
        comp_bprp_map=tier_out["comp_bprp_map"],
        comp_tier_final_map=tier_out["comp_tier_final_map"],
        comp_delta_bprp_map=tier_out["comp_delta_bprp_map"],
        comp_color_tier_src_map=tier_out["comp_color_tier_src_map"],
        per_target_rms_map=rms_map,
    )

    # Zero comps passing the per-target comp_rms gate -> no comparison set for this target.
    _n_good = _count_gate_passing_comps(result, rms_map, max_comp_rms, id_col_cand)
    if _n_good < 1:
        return pd.DataFrame()
    if tier_wide is not tier_out:
        result = _append_reserve_rows(
            result,
            tier_out,
            tier_wide,
            id_col_cand=id_col_cand,
            active=active,
            score_map=score_map,
            comp_n_frames_map=comp_n_frames_map,
            target_cid=target_cid,
            target=target,
            target_bprp_eff=target_bprp_eff,
            per_target_rms_map=rms_map,
        )
    return result


def _append_reserve_rows(
    result: pd.DataFrame,
    tier_primary: dict[str, Any],
    wide: dict[str, Any],
    *,
    id_col_cand: str,
    active: dict[str, float],
    score_map: dict[str, float],
    comp_n_frames_map: dict[str, int],
    target_cid: str,
    target: pd.Series,
    target_bprp_eff: float,
    per_target_rms_map: dict[str, float],
) -> pd.DataFrame:
    """COMP-SMART-2026-10: reserve = the candidates the wide ranking (n_comp_max + n_reserve) admits
    beyond the primary set, in rank order. The primary rows are untouched."""
    from comp_selection_per_target import _assemble_comp_selection_result_rows  # noqa: PLC0415

    prim_ids = {str(c).strip() for c in tier_primary.get("selected_ids") or []}
    res_ids = [str(c).strip() for c in (wide.get("selected_ids") or []) if str(c).strip() not in prim_ids]
    fc = wide.get("final_comps")
    if not res_ids or fc is None or getattr(fc, "empty", True):
        return result
    id_col = id_col_cand
    res = _assemble_comp_selection_result_rows(
        res_ids,
        fc,
        id_col_cand=id_col,
        active=active,
        score_map=score_map,
        comp_n_frames_map=comp_n_frames_map,
        target_cid=target_cid,
        target=target,
        target_bprp_eff=target_bprp_eff,
        sel_note=str(wide.get("sel_note", "")),
        best_tier=str(wide.get("best_tier", "")),
        tier4_warning=bool(wide.get("tier4_warning", False)),
        n_t1=int(wide.get("n_t1", 0)),
        n_t2=int(wide.get("n_t2", 0)),
        n_t3=int(wide.get("n_t3", 0)),
        n_t4=int(wide.get("n_t4", 0)),
        comp_bprp_map=wide.get("comp_bprp_map") or {},
        comp_tier_final_map=wide.get("comp_tier_final_map") or {},
        comp_delta_bprp_map=wide.get("comp_delta_bprp_map") or {},
        comp_color_tier_src_map=wide.get("comp_color_tier_src_map") or {},
        per_target_rms_map=per_target_rms_map,
    )
    if res is None or res.empty:
        return result
    _rank = {c: i for i, c in enumerate(res_ids)}
    res = res.copy()
    res["comp_role"] = "reserve"
    res["reserve_rank"] = [int(_rank.get(str(c).strip(), len(_rank))) for c in res[id_col].astype(str)]
    res = res.sort_values("reserve_rank", kind="mergesort")
    # the primary rows keep the selection note / tier counts of the primary selection
    for c in ("selection_note", "selected_tier", "tier4_warning", "n_tier1", "n_tier2", "n_tier3", "n_tier4"):
        if c in result.columns and c in res.columns and len(result):
            res[c] = result[c].iloc[0]
    return pd.concat([result, res], ignore_index=True, sort=False)
