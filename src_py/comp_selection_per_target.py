"""Per-target comparison star selection (CQ-3 / PERF-4B / PERF-9).

Extracted from ``photometry_core.select_comparison_stars_per_target``.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import AbstractSet, Any, Callable

import numpy as np
import pandas as pd

from comp_frame_normalize import (
    assign_relative_flux,
    build_frame_bin_medians,
    robust_comp_rms,
)
from comp_pool_rms import sort_per_frame_csv_paths
from catalog_db import connect_ro
from config import AppConfig
from gaia_catalog_id import normalize_gaia_id_set, normalize_gaia_source_id
from infolog import log_event
from pipeline_constants import COMP_LINEARITY_FRAC, COMP_SAT_LIMIT_SCALE
from photometry_core import (
    _PHASE_USECOLS_PERFRAME,
    _bool_col,
    _enrich_comp_bp_rp,
    _normalize_gaia_id,
    _normalize_id_series,
    _normalize_id_value,
    _select_comps_by_rms_then_color,
    _warn_zero_compstars_edge,
)
from sky_separation import (
    angular_distance_deg_vectorized as _angular_distance_deg_vectorized,
    pixel_distance_deg_vectorized as _pixel_distance_deg_vectorized,
    quantize_dist_deg,
)

LOGGER = logging.getLogger(__name__)

# COMP-POOL-01 Stage 3: pair-criterion relaxation order (assignment only).
# Pool admission never relaxes these. Each firing is recorded in selection_note / provenance.
COMP_ASSIGNMENT_RELAX_ORDER: tuple[str, ...] = ("colour_tier_widen_T1_to_T4",)


def format_assignment_relax_provenance(
    *,
    best_tier: str,
    n_t1: int = 0,
    n_t2: int = 0,
    n_t3: int = 0,
    n_t4: int = 0,
) -> str:
    """ASCII provenance fragment: fixed order plus which steps actually fired."""
    fired: list[str] = []
    bt = str(best_tier or "").strip().upper()
    if bt and bt not in ("TIER1", "1", ""):
        fired.append(f"colour_tier->{bt}")
    order = ">".join(COMP_ASSIGNMENT_RELAX_ORDER)
    fired_s = ",".join(fired) if fired else "none"
    return (
        f"relax_order={order}; fired={fired_s}; "
        f"n_tier1={int(n_t1)},n_tier2={int(n_t2)},n_tier3={int(n_t3)},n_tier4={int(n_t4)}"
    )


# Distance helpers live in sky_separation (one formula for greedy + pin paths).


def _resolve_target_color_for_comp_selection(
    target: pd.Series,
    *,
    gaia_db_path: str | None,
    cfg: AppConfig | None = None,
) -> dict[str, Any]:
    _cfg = cfg or AppConfig()
    ra_t = float(target["ra_deg"])
    dec_t = float(target["dec_deg"])
    mag_t = float(
        pd.to_numeric(
            target.get(
                "mag",
                target.get(
                    "phot_g_mean_mag",
                    target.get("g_mag", target.get("gaia_g_mag", float("nan"))),
                ),
            ),
            errors="coerce",
        )
    )
    target_cid = str(target.get("catalog_id", ""))
    t_bp_tgt = float(pd.to_numeric(target.get("bp_rp"), errors="coerce"))

    _gdb = str(gaia_db_path or "").strip() or str(_cfg.gaia_db_path or "").strip()
    _sid = normalize_gaia_source_id(target.get("catalog_id", target.get("name")))

    # If bp_rp is missing in the row, try to fetch it (and teff) from Gaia SQLite by source_id.
    g_bp_rp = float("nan")
    if _sid and _sid.isdigit() and _gdb and os.path.exists(_gdb):
        try:
            import sqlite3  # noqa: PLC0415

            con = connect_ro(_gdb)
            try:
                con.row_factory = sqlite3.Row
                cols = {str(r[1]).strip().lower() for r in con.execute("PRAGMA table_info('gaia_dr3')").fetchall()}
                want = []
                if "bp_rp" in cols:
                    want.append("bp_rp")
                if want:
                    row = con.execute(
                        f"SELECT {', '.join(want)} FROM gaia_dr3 WHERE source_id=? LIMIT 1;",
                        (int(_sid),),
                    ).fetchone()
                    if row is not None:
                        if "bp_rp" in want and row["bp_rp"] is not None:
                            try:
                                g_bp_rp = float(row["bp_rp"])
                            except (TypeError, ValueError):
                                g_bp_rp = float("nan")
            finally:
                con.close()
        except Exception as exc:  # noqa: BLE001
            logging.error('[EXC-0030] Gaia source_id BP-RP SQL lookup failure leaves target bp_rp NaN until NN fallback may m...: %s', exc)
            pass

    # If source_id lookup failed (often due to catalog_id float precision loss upstream),
    # fall back to nearest-neighbor by sky position in a small radius.
    if (not math.isfinite(g_bp_rp)) and _gdb and os.path.exists(_gdb):
        try:
            import sqlite3  # noqa: PLC0415

            ra0 = float(ra_t)
            de0 = float(dec_t)
            if math.isfinite(ra0) and math.isfinite(de0):
                r_deg = 10.0 / 3600.0
                cos_dec = math.cos(math.radians(abs(de0)))
                cos_dec = cos_dec if abs(cos_dec) > 1e-6 else 1e-6
                con = connect_ro(_gdb)
                try:
                    con.row_factory = sqlite3.Row
                    cols = {str(r[1]).strip().lower() for r in con.execute("PRAGMA table_info('gaia_dr3')").fetchall()}
                    sel = ["source_id"]
                    if "bp_rp" in cols:
                        sel.append("bp_rp")
                    if sel:
                        q = (
                            f"SELECT {', '.join(sel)} FROM gaia_dr3 "
                            "WHERE ra BETWEEN ? AND ? AND dec BETWEEN ? AND ? "
                            "ORDER BY ((ra-?)*(ra-?)*?)+((dec-?)*(dec-?)) "
                            "LIMIT 1;"
                        )
                        row = con.execute(
                            q,
                            (
                                ra0 - r_deg / cos_dec,
                                ra0 + r_deg / cos_dec,
                                de0 - r_deg,
                                de0 + r_deg,
                                ra0,
                                ra0,
                                cos_dec * cos_dec,
                                de0,
                                de0,
                            ),
                        ).fetchone()
                        if row is not None:
                            if "bp_rp" in row and row["bp_rp"] is not None:
                                try:
                                    g_bp_rp = float(row["bp_rp"])
                                except (TypeError, ValueError):
                                    g_bp_rp = float("nan")
                finally:
                    con.close()
        except Exception as exc:  # noqa: BLE001
            logging.error('[EXC-0031] Same BP-RP SQL lookup failure logged via log_event but target still proceeds without bp_rp: %s', exc)
            log_event(f"[COMP] Gaia BP-RP lookup failed for {target_cid}: {exc}")

    if not math.isfinite(t_bp_tgt) and math.isfinite(g_bp_rp):
        t_bp_tgt = float(g_bp_rp)

    max_delta_bprp_cfg = float(_cfg.comp_max_delta_bprp or 0.79)
    target_bprp_eff = float(t_bp_tgt) if math.isfinite(float(t_bp_tgt)) else float("nan")

    _target_name = (
        str(target.get("vsx_name", "") or target.get("name", "") or target_cid or "?").strip() or "?"
    )
    try:
        if math.isfinite(target_bprp_eff):
            log_event(f"[COMP] BP-RP={float(target_bprp_eff):.3f} for target {target_cid}")
        else:
            log_event(f"[COMP] BP-RP=NaN for target {target_cid} -> T4 / mag proxy")
    except Exception:  # noqa: BLE001
        # EXC-0032: T3 -- BP-RP status log_event line for target never prints (EXCEPT-BULK-2 2026-07-08)
        pass
    if not math.isfinite(target_bprp_eff):
        _vnm = str(target.get("vsx_name", "") or target.get("name", "") or "").strip() or (target_cid or "?")
        log_event(f"TARGET {_vnm} has no BP-RP -> TIER filter by mag only")

    _tier_lims = _cfg.comp_tier_bprp_limits()
    TIER_DEFS = [
        (1, float(_tier_lims[0] or 0.15)),
        (2, float(_tier_lims[1] or 0.30)),
        (3, float(_tier_lims[2] or 0.55)),
        (4, 999.0),
    ]

    def _individual_tier(delta_bprp: float) -> int:
        """Tier by |DeltaBP-RP|; NaN -> T4 (unknown colour)."""
        if not np.isfinite(delta_bprp):
            return 4
        for tier, limit in TIER_DEFS:
            if float(delta_bprp) <= float(limit):
                return int(tier)
        return 4

    return {
        "ra_t": ra_t, "dec_t": dec_t, "mag_t": mag_t, "target_cid": target_cid,
        "t_bp_tgt": t_bp_tgt, "target_bprp_eff": target_bprp_eff,
        "max_delta_bprp_cfg": max_delta_bprp_cfg,
        "TIER_DEFS": TIER_DEFS, "_individual_tier": _individual_tier, "_target_name": _target_name,
    }


def _filter_comp_candidates_spatial_static(
    ms: pd.DataFrame,
    *,
    ra_t: float,
    dec_t: float,
    target_cid: str,
    min_dist_arcsec: float,
    variable_target_catalog_ids: AbstractSet[str] | None,
    use_pixel_dist: bool = False,
    x_t: float | None = None,
    y_t: float | None = None,
    plate_scale_arcsec: float = 1.3,
) -> tuple[pd.DataFrame, pd.Series]:
    """Per-target admission mask of the comparison pool: returns (ms with ``_dist_deg`` / ``_mag``,
    mask). Pool gates (S/N, saturation, edge, blend, identity) are applied once in the global pool;
    here: not saturated, not a known variable, not the target, minimum separation (outside the
    target's sky annulus), not a variable target. Distance is only a tie-break."""
    if use_pixel_dist and x_t is not None and y_t is not None and "x" in ms.columns and "y" in ms.columns:
        x_arr = pd.to_numeric(ms["x"], errors="coerce").to_numpy(dtype=np.float64)
        y_arr = pd.to_numeric(ms["y"], errors="coerce").to_numpy(dtype=np.float64)
        ms["_dist_deg"] = quantize_dist_deg(
            _pixel_distance_deg_vectorized(
                float(x_t),
                float(y_t),
                x_arr,
                y_arr,
                plate_scale_arcsec=float(plate_scale_arcsec),
            )
        )
    else:
        ra_arr = pd.to_numeric(ms["ra_deg"], errors="coerce").to_numpy(dtype=np.float64)
        dec_arr = pd.to_numeric(ms["dec_deg"], errors="coerce").to_numpy(dtype=np.float64)
        ms["_dist_deg"] = quantize_dist_deg(
            _angular_distance_deg_vectorized(ra_t, dec_t, ra_arr, dec_arr)
        )
    ms["_norm_cid_vt"] = ms.get("catalog_id", ms.get("name", pd.Series("", index=ms.index))).map(_normalize_gaia_id)
    _vt_gaia_ids: frozenset[str] | None = None
    if variable_target_catalog_ids:
        _vt_gaia_ids = normalize_gaia_id_set(
            variable_target_catalog_ids,
            log_label="variable_target_catalog_ids (comp spatial filter)",
        ) or None

    _base_mask = (
        ~_bool_col(ms.get("is_saturated", pd.Series(False, index=ms.index)))
        & ~_bool_col(ms.get("vsx_known_variable", pd.Series(False, index=ms.index)))
        & ~_bool_col(ms.get("likely_saturated", pd.Series(False, index=ms.index)))
    )
    if "zone" in ms.columns:
        _base_mask &= ~ms["zone"].astype(str).str.strip().str.lower().eq("saturated")
    if target_cid:
        _base_mask &= (
            ms.get("catalog_id", ms.get("name", pd.Series("", index=ms.index))).astype(str) != target_cid
        )
    if math.isfinite(min_dist_arcsec) and min_dist_arcsec > 0:
        _base_mask &= ms["_dist_deg"].ge(float(min_dist_arcsec) / 3600.0)
    if _vt_gaia_ids:
        _base_mask &= ~ms["_norm_cid_vt"].isin(_vt_gaia_ids)

    if "_mag" not in ms.columns:
        ms["_mag"] = pd.to_numeric(ms.get("mag", ms.get("phot_g_mean_mag")), errors="coerce")
    return ms, _base_mask

def _build_candidates_pre(
    ms: pd.DataFrame,
    *,
    _base_mask: pd.Series,
    mag_t: float,
    target_cid: str,
    n_comp_min: int,
    chip_fw: int | None,
    chip_fh: int | None,
    chip_interior_margin_px: int,
    target: pd.Series,
    cfg: AppConfig | None = None,
    fwhm_dao_px: float = 3.5,
    solve_rms_px: float | None = None,
) -> pd.DataFrame | None:
    """Returns ``candidates_pre`` or None if too few candidates (no magnitude window)."""
    candidates_pre = ms[_base_mask].copy()
    from d3_comparison_candidacy import apply_d3_comparison_candidacy  # noqa: PLC0415

    _d3_mask, _ = apply_d3_comparison_candidacy(
        candidates_pre,
        fwhm_dao_px=float(fwhm_dao_px),
        solve_rms_px=solve_rms_px,
        log_label=str(target_cid or "comp"),
        snr_min=float(getattr(cfg, "snr_min", 10.0)) if cfg is not None else None,
    )
    candidates_pre = candidates_pre.loc[_d3_mask].copy()
    # P3 determinism: sort candidates by catalog_id before any filtering
    if "catalog_id" in candidates_pre.columns:
        candidates_pre = candidates_pre.sort_values("catalog_id", kind="mergesort").reset_index(
            drop=True
        )
    if not candidates_pre.empty and math.isfinite(mag_t):
        candidates_pre = candidates_pre.copy()
        candidates_pre["mag"] = pd.to_numeric(candidates_pre.get("_mag", candidates_pre.get("mag")), errors="coerce")
    if len(candidates_pre) < n_comp_min:
        logging.warning(
            f"[COMP-SELECT] Target {target_cid}: only {len(candidates_pre)} candidates "
            f"< n_comp_min={n_comp_min} - skipping."
        )
        _warn_zero_compstars_edge(
            target_cid=target_cid,
            target=target,
            chip_fw=chip_fw,
            chip_fh=chip_fh,
            chip_interior_margin_px=int(chip_interior_margin_px),
        )
        return None
    return candidates_pre

def _bootstrap_comp_selection_csv_cache(
    per_frame_csv_paths: list[Path],
    csv_cache: dict[str, pd.DataFrame] | None,
    *,
    flux_col: str,
    avail_cols: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    if avail_cols is None:
        avail_cols = list(_PHASE_USECOLS_PERFRAME)
    if csv_cache is None:
        # Note: csv_cache should be passed from the comparison-star selection entry to avoid duplicate reads
        #       when multiple code paths call with csv_cache=None. TODO-CACHE-CENTRAL.
        csv_cache = {}
        for _csv_path in per_frame_csv_paths:
            try:
                from proc_frame_store import read_proc_csv  # noqa: PLC0415

                _hdr = read_proc_csv(_csv_path, nrows=0)
                _use_cols = [c for c in avail_cols if c in _hdr.columns]
                _actual_flux = flux_col if flux_col in _hdr.columns else "dao_flux"
                if _actual_flux not in _use_cols:
                    _use_cols.append(_actual_flux)
                _name_col = "catalog_id" if "catalog_id" in _hdr.columns else "name"
                if "mag" not in _use_cols and "mag" in _hdr.columns:
                    _use_cols.append("mag")
                if "peak_max_adu" in _hdr.columns and "peak_max_adu" not in _use_cols:
                    _use_cols.append("peak_max_adu")
                if (
                    "saturate_limit_adu_85pct" in _hdr.columns
                    and "saturate_limit_adu_85pct" not in _use_cols
                ):
                    _use_cols.append("saturate_limit_adu_85pct")
                # Gaia ID must be str - float64 loses digits
                _dtype_pf: dict[str, type] = {}
                if "catalog_id" in _use_cols:
                    _dtype_pf["catalog_id"] = str
                if "name" in _use_cols:
                    _dtype_pf["name"] = str
                _df0 = read_proc_csv(
                    _csv_path,
                    usecols=_use_cols,
                    low_memory=False,
                    dtype=_dtype_pf or None,
                )
                _df0[_name_col] = _normalize_id_series(_df0[_name_col])
                _df0[_actual_flux] = pd.to_numeric(_df0[_actual_flux], errors="coerce")
                if "peak_max_adu" in _df0.columns:
                    _df0["peak_max_adu"] = pd.to_numeric(_df0["peak_max_adu"], errors="coerce")
                if "saturate_limit_adu_85pct" in _df0.columns:
                    _df0["saturate_limit_adu_85pct"] = pd.to_numeric(
                        _df0["saturate_limit_adu_85pct"], errors="coerce"
                    )
                csv_cache[str(_csv_path)] = _df0
            except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
                from except_fix_counters import get_except_fix_counters

                get_except_fix_counters().comp_selection_csv_cache_skip += 1
                logging.error("[COMP] comp_selection CSV cache skip %s: %s", _csv_path, exc)
                continue
    return csv_cache

def _accumulate_per_frame_comp_metrics(
    per_frame_csv_paths: list[Path],
    csv_cache: dict[str, pd.DataFrame],
    cand_ids: set[str],
    *,
    flux_col: str,
    chip_fw: int | None,
    chip_fh: int | None,
) -> dict[str, Any]:
    """Masterstars-fallback path: relative flux per candidate and per-frame saturation / edge
    counts (one vectorized implementation; S/N, PSF chi2 and FWHM maps were collected and never
    used)."""
    _sorted_cids = sorted(cand_ids)
    flux_map: dict[str, list[float]] = {cid: [] for cid in _sorted_cids}
    bjd_map: dict[str, list[float]] = {cid: [] for cid in _sorted_cids}
    peak_over_map: dict[str, int] = {cid: 0 for cid in _sorted_cids}
    peak_total_map: dict[str, int] = {cid: 0 for cid in _sorted_cids}
    edge_bad_map: dict[str, int] = {cid: 0 for cid in _sorted_cids}
    edge_total_map: dict[str, int] = {cid: 0 for cid in _sorted_cids}
    n_frames_loaded = 0
    _w = int(chip_fw) if chip_fw is not None else None
    _h = int(chip_fh) if chip_fh is not None else None
    for csv_path in sort_per_frame_csv_paths(per_frame_csv_paths, csv_cache):
        df = csv_cache.get(str(csv_path))
        if df is None or df.empty:
            continue
        try:
            name_col = "catalog_id" if "catalog_id" in df.columns else "name"
            actual_flux_col = flux_col if flux_col in df.columns else "dao_flux"
            if (_w is None or _h is None) and "x" in df.columns and "y" in df.columns:
                _xm = float(pd.to_numeric(df["x"], errors="coerce").max())
                _ym = float(pd.to_numeric(df["y"], errors="coerce").max())
                if math.isfinite(_xm) and _xm > 0:
                    _w = max(int(_w or 0), int(math.ceil(_xm)) + 2)
                if math.isfinite(_ym) and _ym > 0:
                    _h = max(int(_h or 0), int(math.ceil(_ym)) + 2)
            _cand = df[df[name_col].isin(cand_ids)]
            if _cand.empty:
                continue
            if "peak_max_adu" in _cand.columns and "saturate_limit_adu_85pct" in _cand.columns:
                pk = pd.to_numeric(_cand["peak_max_adu"], errors="coerce")
                # COMP-SAT-80-2026-10: comps use 0.80 x SAT (column == LINEARITY_FRAC x SAT)
                lim = pd.to_numeric(_cand["saturate_limit_adu_85pct"], errors="coerce") * COMP_SAT_LIMIT_SCALE
                okp = pk.notna() & lim.gt(0)
                for cid, n in _cand.loc[okp, name_col].astype(str).value_counts().items():
                    peak_total_map[cid] = peak_total_map.get(cid, 0) + int(n)
                for cid, n in _cand.loc[okp & (pk >= lim), name_col].astype(str).value_counts().items():
                    peak_over_map[cid] = peak_over_map.get(cid, 0) + int(n)
            sub = _cand[pd.to_numeric(_cand[actual_flux_col], errors="coerce").gt(0)].copy()
            if sub.empty:
                continue
            bin_meds, frame_med, used_mag_bins = build_frame_bin_medians(df, flux_col=actual_flux_col)
            if (used_mag_bins and not bin_meds) or (not used_mag_bins and not (math.isfinite(frame_med) and frame_med > 0)):
                continue
            n_frames_loaded += 1
            sw = assign_relative_flux(sub, flux_col=actual_flux_col, bin_meds=bin_meds, frame_med=frame_med, id_col=name_col)
            rel = pd.to_numeric(sw["_rel"], errors="coerce")
            ok_rel = rel.notna() & np.isfinite(rel.to_numpy(dtype=np.float64)) & rel.gt(0)
            if _w and _h and {"x", "y", "sky_annulus_r_out_px"} <= set(sw.columns):
                x0 = pd.to_numeric(sw["x"], errors="coerce")
                y0 = pd.to_numeric(sw["y"], errors="coerce")
                ro = pd.to_numeric(sw["sky_annulus_r_out_px"], errors="coerce")
                valid = x0.notna() & y0.notna() & ro.gt(0)
                bad = valid & ~((x0 - ro >= 0) & (x0 + ro <= _w) & (y0 - ro >= 0) & (y0 + ro <= _h))
                for cid, n in sw.loc[valid, name_col].astype(str).value_counts().items():
                    edge_total_map[cid] = edge_total_map.get(cid, 0) + int(n)
                for cid, n in sw.loc[bad, name_col].astype(str).value_counts().items():
                    edge_bad_map[cid] = edge_bad_map.get(cid, 0) + int(n)
            for cid, grp in sw.loc[ok_rel].groupby(name_col, sort=True):
                flux_map[str(cid)].extend(grp["_rel"].astype(float).tolist())
                if "bjd_tdb_mid" in grp.columns:
                    bjd_map[str(cid)].extend(pd.to_numeric(grp["bjd_tdb_mid"], errors="coerce").astype(float).tolist())
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            from except_fix_counters import get_except_fix_counters

            get_except_fix_counters().comp_frame_accumulation_skip += 1
            logging.error("[COMP] frame metrics accumulation failed %s: %s", csv_path, exc)
            continue
    logging.info("[PERF-4B] _accumulate_per_frame_comp_metrics: %d frames x %d candidates", n_frames_loaded, len(cand_ids))
    return {
        "flux_map": flux_map,
        "bjd_map": bjd_map,
        "n_frames_loaded": n_frames_loaded,
        "peak_over_map": peak_over_map,
        "peak_total_map": peak_total_map,
        "edge_bad_map": edge_bad_map,
        "edge_total_map": edge_total_map,
    }


def _apply_comp_metric_hard_filters(
    flux_map: dict[str, list[float]],
    peak_over_map: dict[str, int],
    peak_total_map: dict[str, int],
    edge_bad_map: dict[str, int],
    edge_total_map: dict[str, int],
    *,
    edge_bad_frame_frac_max: float,
    sat_may_exclude: bool = True,
) -> dict[str, list[float]]:
    """Masterstars-fallback path only (no global pool): per-frame saturation / edge gates before
    the per-target RMS (the global pool applies the same rules once for the field)."""
    # Filter SAT: reject candidates above admission gate (Tier 1 only; INV-SAT-01)
    _sat_rejected: set[str] = set()
    if sat_may_exclude:
        for cid in sorted(flux_map.keys()):
            total = int(peak_total_map.get(cid, 0) or 0)
            over = int(peak_over_map.get(cid, 0) or 0)
            if total >= 10 and total > 0 and (float(over) / float(total)) > 0.10:
                flux_map.pop(cid, None)
                _sat_rejected.add(cid)
                logging.info(
                    f"[COMP-SELECT] Saturation filter: excluded {cid} "
                    f"({over}/{total} frames above {COMP_LINEARITY_FRAC:.0%} of saturation)"
                )
    else:
        logging.info("[SAT-DIAG] saturation exclusion disabled (Tier 3 / unverified limit)")
    if _sat_rejected:
        logging.info(
            "[BATCH-E E.5] saturation gate excluded %d comps (peak >= COMP_LINEARITY_FRAC x SAT)",
            len(_sat_rejected),
        )
        logging.info(f"[COMP-SELECT] Total excluded for saturation: {len(_sat_rejected)}")

    # Filter EDGE: exclude candidates whose sky annulus often extends beyond the chip
    _edge_rejected: set[str] = set()
    try:
        bad_thr = float(edge_bad_frame_frac_max)
    except (TypeError, ValueError):
        bad_thr = 0.10
    if not math.isfinite(bad_thr) or bad_thr < 0:
        bad_thr = 0.10
    # Apply only when we actually collected edge stats for that star (edge_total_map > 0).
    for cid in sorted(flux_map.keys()):
        total_e = int(edge_total_map.get(cid, 0) or 0)
        bad_e = int(edge_bad_map.get(cid, 0) or 0)
        if total_e > 0:
            bad_frac = float(bad_e) / float(total_e) if total_e > 0 else 0.0
            if bad_frac > bad_thr:
                flux_map.pop(cid, None)
                _edge_rejected.add(cid)
                logging.info(
                    f"[COMP-SELECT] Edge/annulus filter: excluded {cid} "
                    f"({bad_e}/{total_e} frames off chip, frac={bad_frac:.2f} > {bad_thr:.2f})"
                )
    if _edge_rejected:
        logging.info(f"[COMP-SELECT] Total excluded for edge/annulus: {len(_edge_rejected)}")

    return flux_map


def _detrend_and_compute_comp_rms_map(
    flux_map: dict[str, list[float]],
    *,
    min_frames: int,
    max_comp_rms: float,
    n_comp_min: int,
    target_cid: str,
    target: pd.Series,
    chip_fw: int | None,
    chip_fh: int | None,
    chip_interior_margin_px: int,
) -> Any:
    # -- Step 2b: Airmass detrending --
    # A polynomial fit (degree 2) to the relative-flux time series removes the
    # systematic airmass trend. Residuals = true photometric variability.
    for cid in sorted(flux_map.keys()):
        vals = flux_map[cid]
        if len(vals) < 6:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        t = np.linspace(0.0, 1.0, len(arr))
        try:
            coeffs = np.polyfit(t, arr, 2)
            trend_fit = np.polyval(coeffs, t)
            safe_trend = np.where(np.abs(trend_fit) > 1e-9, trend_fit, 1.0)
            detrended = arr / safe_trend
            med_dt = float(np.median(detrended))
            if math.isfinite(med_dt) and med_dt > 0:
                flux_map[cid] = (detrended / med_dt).tolist()
        except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
            from except_fix_counters import get_except_fix_counters

            get_except_fix_counters().comp_detrend_skip += 1
            logging.error(
                "[COMP] detrend fit failed for %s (%d frames): %s",
                cid,
                len(vals),
                exc,
            )

    # -- Step 3: RMS scatter per candidate --
    rms_map: dict[str, float] = {}
    for cid, vals in sorted(flux_map.items(), key=lambda kv: str(kv[0])):
        if len(vals) < min_frames:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        rms = robust_comp_rms(arr)
        if math.isfinite(rms):
            rms_map[cid] = rms

    # Detect "isolated bin" artefacts:
    # if comp_rms comes out unrealistically low (typically ~0) while the star has many measurements,
    # it is often a result of normalisation in a too-sparse brightness bin (norm_med ~ raw_flux).
    ISOLATED_BIN_RMS_FLOOR = 1e-4  # 0.1 mmag
    ISOLATED_BIN_MIN_FRAMES = 50
    for cid in sorted(rms_map.keys()):
        try:
            rms_v = float(rms_map.get(cid, float("nan")))
        except Exception:  # noqa: BLE001
            rms_v = float("nan")
        nfr = int(len(flux_map.get(cid, [])))
        if math.isfinite(rms_v) and rms_v < float(ISOLATED_BIN_RMS_FLOOR) and nfr >= int(ISOLATED_BIN_MIN_FRAMES):
            rms_map.pop(cid, None)
            logging.debug(
                "[COMP SELECT] %s: comp_rms < %.1e at %d frames -> isolated bin -> excluded",
                str(cid),
                float(ISOLATED_BIN_RMS_FLOOR),
                int(nfr),
            )

    # Sorted RMS for the fallback steps
    sorted_rms_map: dict[str, float] = dict(
        sorted(rms_map.items(), key=lambda kv: (float(kv[1]), str(kv[0])))
    )

    # COMP-ADMIT-03: max_comp_rms is no longer an admission hard cut. Scatter
    # enters continuous weights (sigma_rms). Keep all stars with a finite RMS.
    if math.isfinite(max_comp_rms) and max_comp_rms > 0:
        logging.info(
            "[COMP-SELECT] Target %s: COMP-ADMIT-03 skips hard RMS cut "
            "(max_comp_rms=%.3f retained as diagnostic only); n=%d",
            target_cid,
            float(max_comp_rms),
            len(rms_map),
        )

    # Previously: relax-above-gate was forbidden; now there is no RMS gate.
    if len(rms_map) < n_comp_min:
        logging.warning(
            f"[COMP-SELECT] Target {target_cid}: only {len(rms_map)} candidates "
            f"with enough frames < n_comp_min={n_comp_min}."
        )
        _warn_zero_compstars_edge(
            target_cid=target_cid,
            target=target,
            chip_fw=chip_fw,
            chip_fh=chip_fh,
            chip_interior_margin_px=int(chip_interior_margin_px),
        )
        return None, None
    return rms_map, sorted_rms_map


def _ensemble_mad_filter_rms(
    rms_map: dict[str, float],
    candidates: pd.DataFrame,
    *,
    target_cid: str,
    target: pd.Series,
    chip_fw: int | None,
    chip_fh: int | None,
    chip_interior_margin_px: int,
) -> dict[str, float] | None:
    # -- Step 4: no iterative MAD outlier rejection (zero-clipping policy 2026-08-12) --
    id_col_cand = "name" if "name" in candidates.columns else ("catalog_id" if "catalog_id" in candidates.columns else "name")
    cand_ids = set(candidates[id_col_cand].astype(str).str.strip())
    active = {
        cid: rms
        for cid, rms in sorted(rms_map.items(), key=lambda kv: (float(kv[1]), str(kv[0])))
        if cid in cand_ids
    }
    if len(active) < 2:
        logging.warning(
            f"[COMP-SELECT] {target_cid}: only {len(active)} comp after the RMS filter "
            f"< 2 (minimum for QA)."
        )
        _warn_zero_compstars_edge(
            target_cid=target_cid,
            target=target,
            chip_fw=chip_fw,
            chip_fh=chip_fh,
            chip_interior_margin_px=int(chip_interior_margin_px),
        )
        return None
    return {
        cid: rms
        for cid, rms in sorted(active.items(), key=lambda kv: (float(kv[1]), str(kv[0])))
    }


def _score_comp_candidates_broeg(
    active: dict[str, float],
    candidates: pd.DataFrame,
    *,
    id_col_cand: str,
    target_bprp_eff: float,
    _individual_tier: Callable[[float], int],
    cfg: AppConfig | None = None,
) -> Any:
    # -- Step 5: Final selection --
    # Ranking: Broeg (1/comp_rms). Tier is an informative column only.
    #
    # Literature for comp_score:
    # - Broeg, Fernandez & Neuhauser (2005): Astron. Nachr. 326, 134
    #   DOI: 10.1002/asna.200410350
    # - Young, A.T. (1967): AJ 72, 747 - scintillation noise
    # - Hardie, R.H. (1962): in Astronomical Techniques (Stars & Stellar Systems vol. II)
    #   - color-term corrections in differential photometry
    # - AAVSO CCD Photometry Guide (2023), ch. 5.5 and 6
    #   https://www.aavso.org/ccd-photometry-guide
    score_map: dict[str, float] = {}
    tier_map: dict[str, str] = {}
    # SPEED-2026-10 (S3): one id -> first-row bp_rp map instead of a full-column scan per comp.
    _ids = candidates[id_col_cand].astype(str).str.strip().tolist()
    _bprp_raw = candidates["bp_rp"].tolist() if "bp_rp" in candidates.columns else [None] * len(_ids)
    _first_bprp: dict[str, Any] = {}
    for _c, _v in zip(_ids, _bprp_raw, strict=True):
        _first_bprp.setdefault(_c, _v)
    _debug = logging.getLogger().isEnabledFor(logging.DEBUG)
    for cid, rms in sorted(active.items(), key=lambda kv: (float(kv[1]), str(kv[0]))):
        if cid not in _first_bprp:
            continue
        comp_rms = float(rms) if math.isfinite(float(rms)) else float("nan")
        try:
            comp_bprp = float(pd.to_numeric(_first_bprp[cid], errors="coerce"))
        except Exception:  # noqa: BLE001
            comp_bprp = float("nan")
        delta_bprp_tier = (
            abs(float(comp_bprp) - float(target_bprp_eff))
            if (math.isfinite(float(comp_bprp)) and math.isfinite(float(target_bprp_eff)))
            else float("nan")
        )
        _tier = int(_individual_tier(float(delta_bprp_tier)))
        tier_map[str(cid)] = int(_tier)

        # Broeg score 1 / rms^2 (neighbour light is handled by the pool's aperture blend model).
        rms_f = float(comp_rms) if math.isfinite(float(comp_rms)) else float("nan")
        # Safe: rms_f > 1e-6 guard prevents division by zero in weight computation.
        comp_score = (1.0 / (rms_f**2)) if (math.isfinite(rms_f) and rms_f > 1e-6) else 0.0

        if _debug:
            logging.debug(
                "[COMP] %s: tier=%s, rms=%.4f, score=%.0f",
                str(cid),
                str(_tier),
                float(rms_f) if math.isfinite(rms_f) else float("nan"),
                float(comp_score),
            )

        score_map[str(cid)] = float(comp_score)
    return score_map, tier_map


def _assign_comp_tiers_to_pool(
    candidates: pd.DataFrame,
    active: dict[str, float],
    *,
    id_col_cand: str,
    target: pd.Series,
    target_cid: str,
    target_bprp_eff: float,
    mag_t: float,
    _individual_tier: Callable[[float], int],
    _target_name: str,
    gaia_db_path: str | None,
    gaia_prefetch: dict[str, dict[str, Any]] | None,
    n_comp_min: int,
    n_comp_max: int,
    chip_fw: int | None,
    chip_fh: int | None,
    chip_interior_margin_px: int,
    cfg: AppConfig | None = None,
    max_comp_rms: float | None = None,
    fwhm_px: float | None = None,
    field_x: np.ndarray | None = None,
    field_y: np.ndarray | None = None,
) -> dict[str, Any]:
    # Build candidate pool DF with comp_rms, then select by tier + RMS.
    _active_keys = sorted(active.keys(), key=str)
    candidate_pool_df = candidates[
        candidates[id_col_cand].astype(str).str.strip().isin(_active_keys)
    ].copy()
    if id_col_cand in candidate_pool_df.columns and not candidate_pool_df.empty:
        candidate_pool_df = candidate_pool_df.drop_duplicates(subset=[id_col_cand], keep="first")
    # P3 determinism: sort candidates by catalog_id before tier / top-N selection
    if "catalog_id" in candidate_pool_df.columns:
        candidate_pool_df = candidate_pool_df.sort_values("catalog_id", kind="mergesort").reset_index(
            drop=True
        )
    if candidate_pool_df.empty:
        _warn_zero_compstars_edge(
            target_cid=target_cid,
            target=target,
            chip_fw=chip_fw,
            chip_fh=chip_fh,
            chip_interior_margin_px=int(chip_interior_margin_px),
        )
        return {"final_comps": None, "sel_note": "", "selected_ids": [], "n_t1": 0, "n_t2": 0, "n_t3": 0, "n_t4": 0, "n_good": 0, "tier4_warning": False, "best_tier": "TIER4", "comp_bprp_map": {}, "comp_tier_final_map": {}, "comp_delta_bprp_map": {}, "comp_color_tier_src_map": {}}
    candidate_pool_df["comp_rms"] = candidate_pool_df[id_col_cand].astype(str).str.strip().map(active).astype(float)

    _bprp_nan_before = int(pd.to_numeric(candidate_pool_df.get("bp_rp"), errors="coerce").isna().sum())
    candidate_pool_df = _enrich_comp_bp_rp(
        candidates=candidate_pool_df,
        gaia_db_path=str(gaia_db_path or "").strip() or None,
        gaia_prefetch=gaia_prefetch,
    )
    _bprp_nan_after = int(pd.to_numeric(candidate_pool_df.get("bp_rp"), errors="coerce").isna().sum())
    if _bprp_nan_after < _bprp_nan_before:
        log_event(
            f"[COMP] {target_cid}: comp bp_rp enriched NaN {_bprp_nan_before} -> {_bprp_nan_after}"
        )

    def _tier_from_delta_bprp_cell(x: Any) -> int:
        try:
            if pd.isna(x):
                return 4
            xf = float(x)
        except (TypeError, ValueError):
            return 4
        if not np.isfinite(xf):
            return 4
        return _individual_tier(xf)

    _bpr_col = pd.to_numeric(candidate_pool_df.get("bp_rp"), errors="coerce")
    if math.isfinite(float(target_bprp_eff)):
        _delta_bprp = (_bpr_col - float(target_bprp_eff)).abs()
        candidate_pool_df["delta_bprp_abs"] = _delta_bprp
        candidate_pool_df["comp_tier"] = _delta_bprp.apply(_tier_from_delta_bprp_cell).astype(int)
        candidate_pool_df["color_tier_src"] = np.where(_bpr_col.notna(), "bprp", "unknown").astype(str)
    else:
        _mag_col_proxy = candidate_pool_df.get(
            "_mag", candidate_pool_df.get("mag", candidate_pool_df.get("phot_g_mean_mag"))
        )
        _mag_co_proxy = pd.to_numeric(_mag_col_proxy, errors="coerce")
        if math.isfinite(float(mag_t)):
            _delta_mag_col = (_mag_co_proxy - float(mag_t)).abs()
        else:
            _delta_mag_col = pd.Series(np.nan, index=candidate_pool_df.index)
        candidate_pool_df["delta_bprp_abs"] = float("nan")

        def _tier_from_delta_mag_proxy(dm: Any) -> int:
            try:
                if pd.isna(dm):
                    return 4
                dmf = float(dm)
            except (TypeError, ValueError):
                return 4
            if not math.isfinite(dmf):
                return 4
            if dmf <= 0.5:
                return 2
            if dmf <= 1.5:
                return 3
            return 4

        candidate_pool_df["comp_tier"] = _delta_mag_col.apply(_tier_from_delta_mag_proxy).astype(int)
        candidate_pool_df["color_tier_src"] = "mag_proxy"
        logging.info(
            "[COMP] Target %s: magnitude-proxy tiers assigned (no Gaia BP-RP)",
            _target_name,
        )

    if logging.getLogger(__name__).isEnabledFor(logging.DEBUG):
        _tbpr_dbg = float(target_bprp_eff) if math.isfinite(float(target_bprp_eff)) else float("nan")
        for _, prow in candidate_pool_df.iterrows():
            try:
                cid_dbg = str(prow.get(id_col_cand, "") or "").strip()
                dbpr_dbg = float(pd.to_numeric(prow.get("delta_bprp_abs"), errors="coerce"))
                ct_dbg = int(pd.to_numeric(prow.get("comp_tier"), errors="coerce") or 4)
                bpr_dbg = float(pd.to_numeric(prow.get("bp_rp"), errors="coerce"))
            except Exception:  # noqa: BLE001
                continue
            logging.debug(
                "[TIER] target=%s target_bprp=%.4f comp=%s delta_bprp=%.4f tier=%d bp_rp=%.4f",
                target_cid,
                _tbpr_dbg,
                cid_dbg,
                dbpr_dbg if math.isfinite(dbpr_dbg) else float("nan"),
                ct_dbg,
                bpr_dbg if math.isfinite(bpr_dbg) else float("nan"),
            )

    try:
        _id_ser = candidate_pool_df[id_col_cand].astype(str).str.strip()
        _bprp_map = pd.to_numeric(candidate_pool_df.get("bp_rp"), errors="coerce")
        _tier_map_final = pd.to_numeric(candidate_pool_df.get("comp_tier"), errors="coerce").fillna(4).astype(int)
        _delta_bprp_map = pd.to_numeric(candidate_pool_df.get("delta_bprp_abs"), errors="coerce")
        _cts = (
            candidate_pool_df.get("color_tier_src", pd.Series([""] * len(candidate_pool_df)))
            .astype(str)
            .str.strip()
        )
        comp_bprp_map = dict(zip(_id_ser.tolist(), _bprp_map.tolist(), strict=True))
        comp_tier_final_map = dict(zip(_id_ser.tolist(), _tier_map_final.tolist(), strict=True))
        comp_delta_bprp_map = dict(zip(_id_ser.tolist(), _delta_bprp_map.tolist(), strict=True))
        comp_color_tier_src_map = dict(zip(_id_ser.tolist(), _cts.tolist(), strict=True))
    except Exception:  # noqa: BLE001
        # EXC-0046: T3 -- DEBUG tier-assignment log line for one comp candidate skipped (EXCEPT-BULK-2 2026-07-08)
        comp_bprp_map = {}
        comp_tier_final_map = {}
        comp_delta_bprp_map = {}
        comp_color_tier_src_map = {}

    _cfg_w = cfg or AppConfig()
    _max_d = float(_cfg_w.comp_max_delta_bprp or 0.79)

    # COMP-ASSIGN-03 single-source: nearest-neighbour distance in FWHM units
    # against the full field (ms arrays), CoG isolation criterion.
    try:
        _fw_nn = float(fwhm_px) if fwhm_px is not None else float("nan")
    except (TypeError, ValueError):
        _fw_nn = float("nan")
    if (
        field_x is not None
        and field_y is not None
        and "x" in candidate_pool_df.columns
        and "y" in candidate_pool_df.columns
        and math.isfinite(_fw_nn)
        and _fw_nn > 0
    ):
        fx = np.asarray(field_x, dtype=float)
        fy = np.asarray(field_y, dtype=float)
        okf = np.isfinite(fx) & np.isfinite(fy)
        fx, fy = fx[okf], fy[okf]
        if fx.size >= 2:
            from scipy.spatial import cKDTree

            tree = cKDTree(np.column_stack([fx, fy]))
            cx = pd.to_numeric(candidate_pool_df["x"], errors="coerce").to_numpy(dtype=float)
            cy = pd.to_numeric(candidate_pool_df["y"], errors="coerce").to_numpy(dtype=float)
            nn_px = np.full(len(candidate_pool_df), np.nan, dtype=float)
            okc = np.isfinite(cx) & np.isfinite(cy)
            if bool(okc.any()):
                # k=2: if the query point sits on a field star (self), d0~0 and
                # true NN is d1; if slightly offset, d0 is the true NN.
                d_nn, _ = tree.query(np.column_stack([cx[okc], cy[okc]]), k=2)
                self_eps = max(0.25, 0.05 * float(_fw_nn))
                nn_px[okc] = np.where(
                    d_nn[:, 0] <= self_eps, d_nn[:, 1], d_nn[:, 0]
                )
            candidate_pool_df = candidate_pool_df.copy()
            candidate_pool_df["_nn_px"] = nn_px
            candidate_pool_df["_nn_dist_fwhm"] = nn_px / _fw_nn
            logging.info(
                "[COMP] single-source NN attached: fwhm=%.3f iso>=%.2f FWHM "
                "(%.1f px); n_cand=%d n_field=%d",
                float(_fw_nn),
                float(getattr(_cfg_w, "comp_isolation_fwhm", 3.0) or 3.0),
                float(_fw_nn) * float(getattr(_cfg_w, "comp_isolation_fwhm", 3.0) or 3.0),
                int(len(candidate_pool_df)),
                int(fx.size),
            )

    logging.info(
        "[COMP] target=%s rms-then-color begin n_cand=%d",
        str(target_cid),
        int(len(candidate_pool_df)),
    )
    final_comps = _select_comps_by_rms_then_color(
        candidates=candidate_pool_df,
        target_bprp=float(target_bprp_eff),
        n_comp_min=int(n_comp_min),
        n_comp_max=int(n_comp_max),
        max_delta_bprp=_max_d,
        cfg=_cfg_w,
        max_comp_rms=max_comp_rms,
        fwhm_px=_fw_nn if math.isfinite(_fw_nn) else None,
    )
    logging.info(
        "[COMP] target=%s rms-then-color end n_admitted=%d",
        str(target_cid),
        int(0 if final_comps is None else len(final_comps)),
    )

    def _color_rms_sel_note(fc: pd.DataFrame) -> str:
        if fc is None or getattr(fc, "empty", True):
            return "no_candidates"
        if "_delta_bprp_abs" in fc.columns:
            _col = "_delta_bprp_abs"
        elif "delta_bprp_abs" in fc.columns:
            _col = "delta_bprp_abs"
        else:
            return "color_rms"
        mx = float(pd.to_numeric(fc[_col], errors="coerce").max())
        _wlims = _cfg_w.comp_tier_bprp_limits()
        t1 = float(_wlims[0])
        t2 = float(_wlims[1])
        t3 = float(_wlims[2])
        cap = float(_cfg_w.comp_max_delta_bprp)
        if mx <= t1:
            return "color_rms_t1"
        if mx <= t2:
            return "color_rms_t2"
        if mx <= t3:
            return "color_rms_t3"
        if mx <= cap:
            return "color_rms_cap"
        return "color_rms_wide"

    sel_note = _color_rms_sel_note(final_comps)
    if final_comps is not None and bool(getattr(final_comps, "attrs", {}).get("color_fallback")):
        sel_note = "color_rms_t4_fallback"

    if final_comps is None or getattr(final_comps, "empty", True):
        _warn_zero_compstars_edge(
            target_cid=target_cid,
            target=target,
            chip_fw=chip_fw,
            chip_fh=chip_fh,
            chip_interior_margin_px=int(chip_interior_margin_px),
        )
        return {
            "final_comps": None,
            "sel_note": "",
            "selected_ids": [],
            "n_t1": 0,
            "n_t2": 0,
            "n_t3": 0,
            "n_t4": 0,
            "n_good": 0,
            "tier4_warning": False,
            "best_tier": "TIER4",
            "comp_bprp_map": {},
            "comp_tier_final_map": {},
            "comp_delta_bprp_map": {},
            "comp_color_tier_src_map": {},
        }

    # Logging (selection summary)
    try:
        _t = pd.to_numeric(final_comps["comp_tier"], errors="coerce").fillna(4).astype(int)
        log_event(
            f"[COMP] {target_cid}: selected={len(final_comps)} "
            f"T1={int((_t==1).sum())} "
            f"T2={int((_t==2).sum())} "
            f"T3={int((_t==3).sum())} "
            f"T4={int((_t==4).sum())} "
            f"note={sel_note}"
        )
        if str(sel_note).startswith("color_rms") and sel_note not in ("color_rms_t1",):
            log_event(
                f"[COMP] WARNING {target_cid}: {sel_note} - widened colour window / sparse pool"
            )
    except Exception:  # noqa: BLE001
        # EXC-0047: T3 -- Comp selection summary log_event (widened colour window warning) never emitted (EXCEPT-BULK-2 2026-07-08)
        pass

    selected_ids = final_comps[id_col_cand].astype(str).str.strip().tolist()

    # Tier counts must reflect the FINAL (enriched) color-based tiers (BP-RP primary or legacy B-V).
    _t_final = pd.to_numeric(final_comps.get("comp_tier"), errors="coerce").fillna(4).astype(int)
    n_t1 = int((_t_final == 1).sum())
    n_t2 = int((_t_final == 2).sum())
    n_t3 = int((_t_final == 3).sum())
    n_t4 = int((_t_final == 4).sum())
    n_good = int(n_t1 + n_t2)

    if len(selected_ids) == 0:
        if n_good == 0:
            logging.warning(
                "[COMP] %s: no T1/T2 comp, "
                "only %d TIER3/4 - LC will be skipped",
                target_cid,
                len(selected_ids),
            )
        _warn_zero_compstars_edge(
            target_cid=target_cid,
            target=target,
            chip_fw=chip_fw,
            chip_fh=chip_fh,
            chip_interior_margin_px=int(chip_interior_margin_px),
        )
        return {
            "final_comps": None,
            "sel_note": str(sel_note),
            "selected_ids": [],
            "n_t1": 0,
            "n_t2": 0,
            "n_t3": 0,
            "n_t4": 0,
            "n_good": 0,
            "tier4_warning": False,
            "best_tier": "TIER4",
            "comp_bprp_map": {},
            "comp_tier_final_map": {},
            "comp_delta_bprp_map": {},
            "comp_color_tier_src_map": {},
        }
    if len(selected_ids) < int(n_comp_min):
        logging.warning(
            f"[COMP-SELECT] Target {target_cid}: only {len(selected_ids)} after filtering "
            f"< n_comp_min={n_comp_min} - keeping thin set (graceful degradation)."
        )

    # Tier distribution + warning logic (after final selection)
    if n_good == 0 and n_t4 > 0:
        log_event(
            f"WARNING: [COMP] {target_cid}: SPARSE FIELD - "
            f"all {len(selected_ids)} comp stars TIER3/4 "
            f"(color mismatch, penalty applied). "
            f"LC quality may be reduced.",
        )
        tier4_warning = True
    elif n_t4 > 0:
        log_event(
            f"INFO: [COMP] {target_cid}: mixed ensemble - "
            f"{n_t1}xT1 {n_t2}xT2 {n_t3}xT3 {n_t4}xT4 "
            f"(TIER3/4 penalty applied).",
        )
        tier4_warning = False
    else:
        tier4_warning = False

    # selected_tier = best tier present (not majority).
    tier_counts = {"TIER1": int(n_t1), "TIER2": int(n_t2), "TIER3": int(n_t3), "TIER4": int(n_t4)}
    TIER_PRIORITY = ["TIER1", "TIER2", "TIER3", "TIER4"]
    best_tier = "TIER4"
    for t in TIER_PRIORITY:
        if int(tier_counts.get(t, 0) or 0) > 0:
            best_tier = str(t)
            break
    return {
        "final_comps": final_comps,
        "sel_note": sel_note,
        "selected_ids": selected_ids,
        "n_t1": n_t1, "n_t2": n_t2, "n_t3": n_t3, "n_t4": n_t4,
        "n_good": n_good,
        "tier4_warning": tier4_warning,
        "best_tier": best_tier,
        "comp_bprp_map": comp_bprp_map,
        "comp_tier_final_map": comp_tier_final_map,
        "comp_delta_bprp_map": comp_delta_bprp_map,
        "comp_color_tier_src_map": comp_color_tier_src_map,
    }

def _assemble_comp_selection_result_rows(
    selected_ids: list[str],
    final_comps: pd.DataFrame,
    *,
    id_col_cand: str,
    active: dict[str, float],
    score_map: dict[str, float],
    comp_n_frames_map: dict[str, int],
    target_cid: str,
    target: pd.Series,
    target_bprp_eff: float,
    sel_note: str,
    best_tier: str,
    tier4_warning: bool,
    n_t1: int,
    n_t2: int,
    n_t3: int,
    n_t4: int,
    comp_bprp_map: dict[str, float],
    comp_tier_final_map: dict[str, int],
    comp_delta_bprp_map: dict[str, float],
    comp_color_tier_src_map: dict[str, str],
    final_lookup: pd.DataFrame | None,
    dilution_map: dict[str, dict[str, Any]] | None = None,
    per_target_rms_map: dict[str, float] | None = None,
) -> pd.DataFrame:
    # Build the output DataFrame
    # IMPORTANT: rows must come from final_comps (enriched bp_rp), not from masterstars-derived `candidates`.
    try:
        final_lookup = final_comps.copy()
        final_lookup[id_col_cand] = final_lookup[id_col_cand].astype(str).str.strip()
        final_lookup = final_lookup.set_index(id_col_cand, drop=False)
    except Exception:  # noqa: BLE001
        final_lookup = None
    result_rows = []
    _seen_selected: set[str] = set()
    for cid in selected_ids:
        cid_s = str(cid).strip()
        if not cid_s or cid_s in _seen_selected:
            continue
        _seen_selected.add(cid_s)
        if final_lookup is not None:
            try:
                hit = final_lookup.loc[cid_s]
                if isinstance(hit, pd.DataFrame):
                    hit = hit.iloc[0]
                r = hit.to_dict()
            except Exception:  # noqa: BLE001
                r = {}
        else:
            r = {}
        if not r:
            continue
        if "catalog_id" in r:
            r["catalog_id"] = normalize_gaia_source_id(r.get("catalog_id"))
        if "name" in r:
            r["name"] = normalize_gaia_source_id(r.get("name")) or str(r.get("name", "") or "")
        _cid_key = cid_s
        _pt_map = per_target_rms_map or {}
        _pt_rms = float(_pt_map.get(_cid_key, _pt_map.get(cid, float("nan"))))
        _fw_rms = float(active.get(cid, active.get(_cid_key, float("nan"))))
        r["comp_rms"] = _fw_rms if math.isfinite(_fw_rms) else _pt_rms
        r["comp_rms_fieldwide"] = float("nan")
        r["comp_score"] = score_map.get(cid, float("nan"))
        r["comp_n_frames"] = int(comp_n_frames_map.get(cid_s, comp_n_frames_map.get(str(cid), 0)))
        r["target_catalog_id"] = normalize_gaia_source_id(target_cid) or str(target_cid)
        r["target_vsx_name"] = str(target.get("vsx_name", ""))
        r["selected_tier"] = str(best_tier)
        r["tier4_warning"] = bool(tier4_warning)
        r["n_tier1"] = int(n_t1)
        r["n_tier2"] = int(n_t2)
        r["n_tier3"] = int(n_t3)
        r["n_tier4"] = int(n_t4)
        try:
            comp_bprp = float(pd.to_numeric(comp_bprp_map.get(cid, r.get("bp_rp", float("nan"))), errors="coerce"))
        except Exception:  # noqa: BLE001
            comp_bprp = float("nan")
        r["bp_rp"] = float(comp_bprp) if math.isfinite(comp_bprp) else float("nan")
        try:
            r["delta_bprp_abs"] = float(
                pd.to_numeric(comp_delta_bprp_map.get(cid, float("nan")), errors="coerce")
            )
        except Exception:  # noqa: BLE001
            r["delta_bprp_abs"] = float("nan")
        if not math.isfinite(float(r.get("delta_bprp_abs", float("nan")))):
            delta_bprp = (
                abs(float(comp_bprp) - float(target_bprp_eff))
                if (math.isfinite(comp_bprp) and math.isfinite(float(target_bprp_eff)))
                else float("nan")
            )
            r["delta_bprp_abs"] = float(delta_bprp) if math.isfinite(delta_bprp) else float("nan")
        try:
            r["comp_tier"] = int(pd.to_numeric(comp_tier_final_map.get(cid, r.get("comp_tier")), errors="coerce") or 4)
        except Exception:  # noqa: BLE001
            r["comp_tier"] = 4
        try:
            r["color_tier_src"] = str(comp_color_tier_src_map.get(cid, "") or "").strip()
        except Exception:  # noqa: BLE001
            r["color_tier_src"] = ""
        r["target_bp_rp"] = float(target_bprp_eff) if math.isfinite(float(target_bprp_eff)) else float("nan")
        if dilution_map and cid in dilution_map:
            try:
                r["dilution_factor"] = float(dilution_map[cid].get("dilution_factor", float("nan")))
            except (TypeError, ValueError):
                r["dilution_factor"] = float("nan")
            try:
                r["dilution_delta_mag"] = float(dilution_map[cid].get("dilution_delta_mag", float("nan")))
            except (TypeError, ValueError):
                r["dilution_delta_mag"] = float("nan")
        else:
            r["dilution_factor"] = float("nan")
            r["dilution_delta_mag"] = float("nan")
        _sel = str(sel_note)
        _relax = format_assignment_relax_provenance(
            best_tier=str(best_tier),
            n_t1=int(n_t1),
            n_t2=int(n_t2),
            n_t3=int(n_t3),
            n_t4=int(n_t4),
        )
        _sel = f"{_sel}; {_relax}" if _sel else _relax
        r["selection_note"] = _sel
        r["comp_path"] = "default"
        rms_val = r.get("comp_rms", float("nan"))
        try:
            rms_f = float(pd.to_numeric(rms_val, errors="coerce"))
        except Exception:  # noqa: BLE001
            rms_f = float("nan")
        # COMP-ADMIT-03 / PRE-IMPL-01: persist the same sigma_eff weight aperture photometry uses.
        # Previous 1/rms^2-only column was identical across targets (no colour term) and
        # did not describe the run. Weights are the selection mechanism after gate removal.
        try:
            from comp_weights import (  # noqa: PLC0415
                C_COL_PSF_REFRACTIVE_MAG_PER_BPRP,
                sigma_eff_mag,
                weight_from_sigma_eff,
            )

            bpr = float(pd.to_numeric(r.get("bp_rp"), errors="coerce"))
            tb = float(pd.to_numeric(r.get("target_bp_rp"), errors="coerce"))
            db = abs(bpr - tb) if math.isfinite(bpr) and math.isfinite(tb) else 0.0
            # Separation optional at the comparison-star selection write; c_dist=0 on wide refractive.
            se = sigma_eff_mag(
                sigma_rms_mag=rms_f if math.isfinite(rms_f) else float("nan"),
                delta_bprp=db,
                r_deg=0.0,
                c_col_mag_per_bprp=float(C_COL_PSF_REFRACTIVE_MAG_PER_BPRP),
                c_dist_mag_per_deg=0.0,
            )
            r["comp_weight"] = weight_from_sigma_eff(se)
            r["sigma_eff_mag"] = se
        except Exception:  # noqa: BLE001
            if math.isfinite(rms_f) and rms_f > 1e-6:
                r["comp_weight"] = 1.0 / (rms_f**2)
            else:
                r["comp_weight"] = float("nan")
        result_rows.append(r)

    if not result_rows:
        # Return empty with stable schema so downstream CSV read never crashes.
        return pd.DataFrame(
            columns=[
                "catalog_id",
                "name",
                "ra_deg",
                "dec_deg",
                "x",
                "y",
                "mag",
                "bp_rp",
                "comp_rms",
                "comp_rms_fieldwide",
                "comp_score",
                "comp_n_frames",
                "target_catalog_id",
                "target_vsx_name",
                "target_bp_rp",
                "delta_bprp_abs",
                "comp_tier",
                "color_tier_src",
                "comp_weight",
                "dilution_factor",
                "dilution_delta_mag",
                "selection_note",
                "comp_path",
                "selected_tier",
                "tier4_warning",
                "n_tier1",
                "n_tier2",
                "n_tier3",
                "n_tier4",
            ]
        )

    out = pd.DataFrame(result_rows)
    if "comp_rms" in out.columns:
        out = out.sort_values(
            ["comp_rms", "catalog_id"], ascending=[True, True], kind="mergesort"
        ).reset_index(drop=True)
    color_extra = ""
    dpb = pd.to_numeric(out.get("delta_bprp_abs"), errors="coerce")
    if dpb.notna().any():
        color_info = f"DeltaBPRP median={float(dpb.median()):.3f} max={float(dpb.max()):.3f}"
    else:
        color_info = "DeltaBPRP N/A"
    try:
        if final_comps is not None and "color_tier_src" in final_comps.columns:
            _cs = final_comps["color_tier_src"].astype(str)
            if len(_cs) > 0:
                if int(_cs.nunique(dropna=False)) > 1:
                    color_extra = "color_src=mixed"
                else:
                    color_extra = f"color_src={_cs.iloc[0]}"
    except Exception:  # noqa: BLE001
        color_extra = ""

    _log_line = (
        f"[COMP-SELECT] Target {target_cid} ({target.get('vsx_name','')}): "
        f"{len(out)} comparison stars | RMS min={out['comp_rms'].min():.4f} "
        f"max={out['comp_rms'].max():.4f} | {color_info}"
    )
    if color_extra:
        _log_line += f" | {color_extra}"
    logging.info(_log_line)
    return out
