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
    norm_med_for_bin,
    robust_comp_rms,
)
from comp_pool_rms import sort_per_frame_csv_paths
from config import AppConfig
from gaia_catalog_id import normalize_gaia_id_set, normalize_gaia_source_id
from infolog import log_event
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

LOGGER = logging.getLogger(__name__)

_BO_CVN_CID = "1498613634033133184"
BO_CVN_STEP_COUNTS: dict[str, int] = {}

# COMP-POOL-01 Stage 3: pair-criterion relaxation order (assignment only).
# Pool admission never relaxes these. Sparse fields widen in this stated order;
# each firing is recorded in selection_note / provenance.
COMP_ASSIGNMENT_RELAX_ORDER: tuple[str, ...] = (
    "colour_tier_widen_T1_to_T4",
    "adaptive_delta_mag",
    "sparse_fallback_path",
)


def format_assignment_relax_provenance(
    *,
    used_mag_tol: float,
    mag_tol_start: float | None,
    best_tier: str,
    comp_path: str,
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
    if mag_tol_start is not None and math.isfinite(float(mag_tol_start)):
        if float(used_mag_tol) > float(mag_tol_start) + 1e-9:
            fired.append(f"delta_mag->{float(used_mag_tol):.2f}")
    if str(comp_path or "").strip().lower() == "sparse_fallback":
        fired.append("sparse_fallback")
    order = ">".join(COMP_ASSIGNMENT_RELAX_ORDER)
    fired_s = ",".join(fired) if fired else "none"
    return (
        f"relax_order={order}; fired={fired_s}; "
        f"n_tier1={int(n_t1)},n_tier2={int(n_t2)},n_tier3={int(n_t3)},n_tier4={int(n_t4)}"
    )


def bo_cvn_funnel_snapshot() -> dict[str, int]:
    return dict(BO_CVN_STEP_COUNTS)


def _log_bo_cvn_comp_funnel(
    *,
    step_counts: dict[str, int],
    max_comp_rms: float,
    n_comp_max: int,
    rms_rejected: list[tuple[str, float]] | None = None,
) -> None:
    """Structured debug funnel for BO CVn comp selection (TASK audit)."""
    order = (
        "A_spatial_max_dist",
        "B_bp_rp",
        "C_mag_diff",
        "D_min_dist",
        "E_is_usable_zone",
        "F_perf4b",
        "G_after_rms",
        "H_after_n_comp_max",
        "final_selected",
    )
    labels = {
        "A_spatial_max_dist": "A: after max_dist_deg (pixel-fallback)",
        "B_bp_rp": "B: after bp_rp filter",
        "C_mag_diff": "C: after mag_diff (adaptive)",
        "D_min_dist": "D: after min_dist_arcsec",
        "E_is_usable_zone": "E: after is_usable / zone",
        "F_perf4b": "F: entering PERF-4B",
        "G_after_rms": "G: after RMS filter (+ MAD)",
        "H_after_n_comp_max": "H: after n_comp_max truncation",
        "final_selected": "Final: comp stars selected",
    }
    print(f"[DEBUG BO CVn] max_comp_rms={float(max_comp_rms):.4f} n_comp_max={int(n_comp_max)}")
    for key in order:
        if key in step_counts:
            print(f"[DEBUG BO CVn] {labels.get(key, key)} -> {int(step_counts[key])}")
    if rms_rejected:
        print(f"[DEBUG BO CVn] Step G REJECTED (rms > {float(max_comp_rms):.4f}):")
        for cid, rv in rms_rejected[:30]:
            print(f"  catalog_id={cid} comp_rms={float(rv):.6f}")


def _pixel_distance_deg_vectorized(
    x_t: float,
    y_t: float,
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    *,
    plate_scale_arcsec: float,
) -> np.ndarray:
    """Euclidean pixel distance converted to degrees via plate scale; invalid -> 999.0."""
    x2 = np.asarray(x_arr, dtype=np.float64)
    y2 = np.asarray(y_arr, dtype=np.float64)
    scale = float(plate_scale_arcsec)
    if not math.isfinite(scale) or scale <= 0:
        return np.full(x2.shape, 999.0, dtype=np.float64)
    _bad_xt = not math.isfinite(float(x_t))
    _bad_yt = not math.isfinite(float(y_t))
    bad = ~np.isfinite(x2) | ~np.isfinite(y2) | _bad_xt | _bad_yt
    dist_px = np.hypot(x2 - float(x_t), y2 - float(y_t))
    dist_deg = dist_px * scale / 3600.0
    return np.where(bad, 999.0, dist_deg)


def _angular_distance_deg_vectorized(
    ra_t: float, dec_t: float, ra_arr: np.ndarray, dec_arr: np.ndarray
) -> np.ndarray:
    """Haversine distance (deg); invalid coords -> 999.0 (PERF-9)."""
    ra2 = np.asarray(ra_arr, dtype=np.float64)
    de2 = np.asarray(dec_arr, dtype=np.float64)
    ra1 = float(ra_t)
    de1 = float(dec_t)
    bad = ~np.isfinite(ra2) | ~np.isfinite(de2)
    r1, d1 = math.radians(ra1), math.radians(de1)
    r2 = np.radians(ra2)
    d2 = np.radians(de2)
    a = (
        np.sin((d2 - d1) / 2.0) ** 2
        + math.cos(d1) * np.cos(d2) * np.sin((r2 - r1) / 2.0) ** 2
    )
    dist = np.degrees(2.0 * np.arcsin(np.minimum(1.0, np.sqrt(np.clip(a, 0.0, 1.0)))))
    dist = np.where(bad, 999.0, dist)
    logging.debug("[PERF-9] vectorized haversine on %d candidates", int(len(dist)))
    return dist


def _resolve_target_color_for_comp_selection(
    target: pd.Series,
    *,
    vsx_local_db_path: str | None,
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

            con = sqlite3.connect(_gdb)
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
                con = sqlite3.connect(_gdb)
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
        log_event(f"TARGET {_vnm} nema BP-RP -> TIER filter len podla mag")

    _tier_lims = _cfg.comp_tier_bprp_limits()
    TIER_DEFS = [
        (1, float(_tier_lims[0] or 0.25)),
        (2, float(_tier_lims[1] or 0.48)),
        (3, float(_tier_lims[2] or 0.79)),
        (4, 999.0),
    ]

    def _individual_tier(delta_bprp: float) -> int:
        """Tier podla |DeltaBP-RP|; NaN -> T4 (neznama farba)."""
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


def _adaptive_mag_filter(
    all_candidates: pd.DataFrame,
    target_mag: float,
    mag_diff_start: float,
    mag_diff_absolute: float,
    n_comp_min: int,
    *,
    max_mag_diff: float,
    mag_diff_step: float = 0.25,
) -> tuple[pd.DataFrame, float]:
    """COMP-ADMIT-03: |dmag| is not an admission cut (faint -> large sigma_rms weight)."""
    _ = (target_mag, mag_diff_absolute, n_comp_min, max_mag_diff, mag_diff_step)
    if all_candidates is None or getattr(all_candidates, "empty", True):
        return pd.DataFrame(), float(mag_diff_start)
    return all_candidates.copy(), float(mag_diff_start)


def _filter_comp_candidates_spatial_static(
    ms: pd.DataFrame,
    *,
    ra_t: float,
    dec_t: float,
    mag_t: float,
    target_cid: str,
    target_bprp_eff: float,
    max_delta_bprp_cfg: float,
    max_dist_deg: float,
    min_dist_arcsec: float,
    exclude_gaia_nss: bool,
    exclude_gaia_extobj: bool,
    chip_fw: int | None,
    chip_fh: int | None,
    chip_interior_margin_px: int,
    variable_target_catalog_ids: AbstractSet[str] | None,
    use_pixel_dist: bool = False,
    x_t: float | None = None,
    y_t: float | None = None,
    plate_scale_arcsec: float = 1.3,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Returns (ms, _base_mask, det_mask)."""
    # -- Krok 1: Filter kandidatov --
    _debug_bo = str(target_cid).strip() == "1498613634033133184"
    if _debug_bo:
                print(f"[DEBUG BO CVn] Step A: global_comp_pool size = {int(len(ms))}")
    if use_pixel_dist and x_t is not None and y_t is not None and "x" in ms.columns and "y" in ms.columns:
        x_arr = pd.to_numeric(ms["x"], errors="coerce").to_numpy(dtype=np.float64)
        y_arr = pd.to_numeric(ms["y"], errors="coerce").to_numpy(dtype=np.float64)
        ms["_dist_deg"] = _pixel_distance_deg_vectorized(
            float(x_t),
            float(y_t),
            x_arr,
            y_arr,
            plate_scale_arcsec=float(plate_scale_arcsec),
        )
    else:
        ra_arr = pd.to_numeric(ms["ra_deg"], errors="coerce").to_numpy(dtype=np.float64)
        dec_arr = pd.to_numeric(ms["dec_deg"], errors="coerce").to_numpy(dtype=np.float64)
        ms["_dist_deg"] = _angular_distance_deg_vectorized(ra_t, dec_t, ra_arr, dec_arr)
    ms["_norm_cid_vt"] = ms.get("catalog_id", ms.get("name", pd.Series("", index=ms.index))).map(_normalize_gaia_id)
    _vt_gaia_ids: frozenset[str] | None = None
    if variable_target_catalog_ids:
        _vt_gaia_ids = normalize_gaia_id_set(
            variable_target_catalog_ids,
            log_label="variable_target_catalog_ids (comp spatial filter)",
        ) or None

    # COMP-ADMIT-03 / FORCED-PHOT-01: distance is a weight term, not a hard cut.
    # Gates: measurability (sat/nonlinear; Gaia QSO/GAL), known variable (VSX + Gaia
    # NSS), geometry (chip margin). is_noisy is NOT a gate (DAO significance -> weight).
    _ = (max_dist_deg, min_dist_arcsec, max_delta_bprp_cfg, target_bprp_eff, mag_t)
    cand_mask = (
        ~_bool_col(ms.get("is_saturated", pd.Series(False, index=ms.index)))
        & ~_bool_col(ms.get("vsx_known_variable", pd.Series(False, index=ms.index)))
        & ~_bool_col(ms.get("likely_saturated", pd.Series(False, index=ms.index)))
    )
    if "zone" in ms.columns:
        _z = ms["zone"].astype(str).str.strip().str.lower()
        cand_mask &= ~_z.isin(["saturated", "nonlinear"])
    if _debug_bo:
        try:
            _n_a = int(cand_mask.sum())
            BO_CVN_STEP_COUNTS["A_spatial_max_dist"] = _n_a
            BO_CVN_STEP_COUNTS["E_is_usable_zone"] = _n_a
            print(
                f"[DEBUG BO CVn] Step B+F: after max_dist_deg={float(max_dist_deg):.3f} "
                f"+ is_usable/zone flags -> {_n_a}"
            )
        except Exception:  # noqa: BLE001
            pass
    # Vyluc samotny target
    if target_cid:
        cand_mask &= ms.get("catalog_id", ms.get("name", pd.Series("", index=ms.index))).astype(str) != target_cid
        if _debug_bo:
                        print(f"[DEBUG BO CVn] (exclude self) -> {int(cand_mask.sum())}")

    # Jednotny vnutorny okraj cipu (premenne / comp / suspected rovnake pravidla)
    _cm = int(chip_interior_margin_px)
    if (
        _cm > 0
        and chip_fw is not None
        and chip_fh is not None
        and int(chip_fw) > 2 * _cm
        and int(chip_fh) > 2 * _cm
        and "x" in ms.columns
        and "y" in ms.columns
    ):
        _xn = pd.to_numeric(ms["x"], errors="coerce")
        _yn = pd.to_numeric(ms["y"], errors="coerce")
        cand_mask &= _xn.between(_cm, int(chip_fw) - _cm) & _yn.between(_cm, int(chip_fh) - _cm)
        if _debug_bo:
                        print(f"[DEBUG BO CVn] (chip margin {int(_cm)} px) -> {int(cand_mask.sum())}")

    # COMP-ADMIT-03: min_dist / colour are weight terms, not admission cuts.
    if math.isfinite(min_dist_arcsec) and min_dist_arcsec > 0 and _debug_bo:
        print(
            f"[DEBUG BO CVn] min_dist_arcsec={float(min_dist_arcsec):.1f} "
            f"skipped as hard cut (COMP-ADMIT-03)"
        )

    # Ziadna hviezda zo zoznamu VSX cielov (variable_targets) ako comp.
    if _vt_gaia_ids:
        cand_mask &= ~ms["_norm_cid_vt"].isin(_vt_gaia_ids)
        if _debug_bo:
            print(f"[DEBUG BO CVn] (exclude variable_targets IDs) -> {int(cand_mask.sum())}")

    if "_mag" not in ms.columns:
        ms["_mag"] = pd.to_numeric(ms.get("mag", ms.get("phot_g_mean_mag")), errors="coerce")

    if math.isfinite(max_delta_bprp_cfg) and max_delta_bprp_cfg > 0 and _debug_bo:
        print(
            f"[DEBUG BO CVn] max_delta_bprp={float(max_delta_bprp_cfg):.2f} "
            f"skipped as hard cut (COMP-ADMIT-03)"
        )

    # Filter A: Gaia objektove flagy
    _n_before_a = int(cand_mask.sum())

    # Gaia NSS -> known variable (non-single / binary flux). QSO/GAL -> measurability
    # (extended source; aperture photometry invalid). Both remain among the three
    # permitted gate classes (COMP-ADMIT-03 review).
    if exclude_gaia_nss and "gaia_nss" in ms.columns:
        _nss_rej = cand_mask & _bool_col(ms["gaia_nss"])
        cand_mask &= ~_bool_col(ms["gaia_nss"])
        _n_rej = int(_nss_rej.sum())
        if _n_rej > 0:
            logging.info(
                f"[FAZA 1] Target {target_cid}: Filter A (gaia_nss) vylucil {_n_rej} kandidatov"
            )

    # gaia_qso / gaia_gal -> extended; measurability gate (aperture invalid)
    if exclude_gaia_extobj:
        _rej_ext_total = 0
        for _ext_col in ("gaia_qso", "gaia_gal"):
            if _ext_col in ms.columns:
                _rej_mask = cand_mask & _bool_col(ms[_ext_col])
                cand_mask &= ~_bool_col(ms[_ext_col])
                _rej = int(_rej_mask.sum())
                _rej_ext_total += _rej
                if _rej > 0:
                    logging.info(
                        f"[FAZA 1] Target {target_cid}: Filter A ({_ext_col}) vylucil {_rej} kandidatov"
                    )

        if _rej_ext_total == 0:
            _ = _rej_ext_total  # noqa: B018

    _n_after_a = int(cand_mask.sum())
    _rej_a_total = _n_before_a - _n_after_a
    if _rej_a_total > 0:
        logging.debug(
            f"[FAZA 1] Target {target_cid}: Filter A celkom vylucil {_rej_a_total} kandidatov "
            f"({_n_before_a} -> {_n_after_a})"
        )
    if _debug_bo:
        try:
            BO_CVN_STEP_COUNTS["E_is_usable_zone"] = int(cand_mask.sum())
            print(f"[DEBUG BO CVn] Step A': after Filter A (nss/extobj) -> {int(cand_mask.sum())}")
        except Exception:  # noqa: BLE001
            # EXC-0040: T3 -- BO CVn debug print after Filter A (nss/extobj) suppressed (EXCEPT-BULK-2 2026-07-08)
            pass

    # Include DET stars (no Gaia ID) if snr50_ok and not saturated.
    # COMP-ADMIT-03: distance is not a hard cut (weight term in Phase-2A).
    det_mask = (
        ms.get("catalog_id", ms.get("name", pd.Series("", index=ms.index)))
        .astype(str)
        .str.startswith("DET")
        & ~_bool_col(ms.get("is_saturated", pd.Series(False, index=ms.index)))
        & ~_bool_col(ms.get("likely_saturated", pd.Series(False, index=ms.index)))
        & _bool_col(ms.get("snr50_ok", pd.Series(False, index=ms.index)))
        & ~_bool_col(ms.get("vsx_known_variable", pd.Series(False, index=ms.index)))
    )
    if target_cid:
        det_mask &= (
            ms.get("catalog_id", ms.get("name", pd.Series("", index=ms.index))).astype(str)
            != target_cid
        )
    _ = min_dist_arcsec  # retained config; not an admission cut

    cand_mask = cand_mask | det_mask

    # Base mask: measurability + known variable only (COMP-ADMIT-03 review).
    # is_noisy omitted (weight via scatter). Gaia NSS/QSO/GAL handled below.
    _base_mask = (
        ~_bool_col(ms.get("is_saturated", pd.Series(False, index=ms.index)))
        & ~_bool_col(ms.get("vsx_known_variable", pd.Series(False, index=ms.index)))
        & ~_bool_col(ms.get("likely_saturated", pd.Series(False, index=ms.index)))
    )
    if "zone" in ms.columns:
        _z2 = ms["zone"].astype(str).str.strip().str.lower()
        _base_mask &= ~_z2.isin(["saturated", "nonlinear"])
    if target_cid:
        _base_mask &= (
            ms.get("catalog_id", ms.get("name", pd.Series("", index=ms.index))).astype(str) != target_cid
        )
    if math.isfinite(min_dist_arcsec) and min_dist_arcsec > 0:
        _base_mask &= ms["_dist_deg"].ge(float(min_dist_arcsec) / 3600.0)
    if exclude_gaia_nss and "gaia_nss" in ms.columns:
        _base_mask &= ~_bool_col(ms["gaia_nss"])
    if exclude_gaia_extobj:
        for _ext_col in ("gaia_qso", "gaia_gal"):
            if _ext_col in ms.columns:
                _base_mask &= ~_bool_col(ms[_ext_col])
    if _vt_gaia_ids:
        _base_mask &= ~ms["_norm_cid_vt"].isin(_vt_gaia_ids)

    if "_mag" not in ms.columns:
        ms["_mag"] = pd.to_numeric(ms.get("mag", ms.get("phot_g_mean_mag")), errors="coerce")
    return ms, _base_mask, det_mask

def _build_candidates_pre_adaptive_mag(
    ms: pd.DataFrame,
    *,
    _base_mask: pd.Series,
    det_mask: pd.Series,
    mag_t: float,
    target_cid: str,
    mag_tol: float,
    max_mag_diff: float,
    n_comp_min: int,
    chip_fw: int | None,
    chip_fh: int | None,
    chip_interior_margin_px: int,
    target: pd.Series,
    cfg: AppConfig | None = None,
    sparse_fallback_mode: bool = False,
) -> tuple[pd.DataFrame, float] | None:
    """Returns (candidates_pre, used_mag_tol) or None if too few candidates."""
    # Start with a broad candidate set (emergency tier) for one-pass per-frame metrics.
    # Apply adaptive Deltamag filter here (changes only the INPUT to later filters).
    candidates_pre = ms[_base_mask | det_mask].copy()
    _debug_bo = str(target_cid).strip() == "1498613634033133184"
    if _debug_bo:
        print(f"[DEBUG BO CVn] Step G0: candidates before adaptive delta mag = {int(len(candidates_pre))}")
    # P3 determinism: sort candidates by catalog_id before any filtering
    if "catalog_id" in candidates_pre.columns:
        candidates_pre = candidates_pre.sort_values("catalog_id", kind="mergesort").reset_index(
            drop=True
        )
    used_mag_tol = float(mag_tol)
    if not candidates_pre.empty and math.isfinite(mag_t):
        try:
            _cfg_mag = cfg or AppConfig()
            mag_abs = float(_cfg_mag.phase01_comparison_max_mag_diff_absolute or 3.0)
            if sparse_fallback_mode:
                mag_abs = min(mag_abs, 2.0)
        except Exception:  # noqa: BLE001
            mag_abs = 3.0
        candidates_pre = candidates_pre.copy()
        candidates_pre["mag"] = pd.to_numeric(candidates_pre.get("_mag", candidates_pre.get("mag")), errors="coerce")
        _before_mag = int(len(candidates_pre))
        candidates_pre, used_mag_tol = _adaptive_mag_filter(
            all_candidates=candidates_pre,
            target_mag=float(mag_t),
            mag_diff_start=float(mag_tol),
            mag_diff_absolute=float(mag_abs),
            n_comp_min=int(n_comp_min),
            max_mag_diff=float(max_mag_diff),
            mag_diff_step=0.25,
        )
        if float(used_mag_tol) > float(mag_tol) + 1e-9:
            log_event(
                f"[COMP] {target_cid}: Deltamag uvolneny {float(mag_tol):.2f} -> {float(used_mag_tol):.2f} "
                f"(pole ma malo kandidatov)"
            )
        logging.debug(
            f"[FAZA 1] Target {target_cid}: adaptive Deltamag filter "
            f"{_before_mag} -> {int(len(candidates_pre))} (used_mag_tol={float(used_mag_tol):.2f})"
        )
        if _debug_bo:
            print(
                f"[DEBUG BO CVn] Step C: after adaptive |delta mag| (start={float(mag_tol):.2f}, "
                f"used={float(used_mag_tol):.2f}) -> {int(len(candidates_pre))}"
            )
        if "catalog_id" in candidates_pre.columns:
            candidates_pre = candidates_pre.sort_values("catalog_id", kind="mergesort").reset_index(
                drop=True
            )
    if len(candidates_pre) < n_comp_min:
        if sparse_fallback_mode and len(candidates_pre) > 0:
            logging.info(
                f"[FAZA 1] Target {target_cid}: sparse fallback - "
                f"{len(candidates_pre)} kandidatov (< n_comp_min={n_comp_min}), pokracujem."
            )
        else:
            logging.warning(
                f"[FAZA 1] Target {target_cid}: len {len(candidates_pre)} kandidatov "
                f"< n_comp_min={n_comp_min} - preskakujem."
            )
            _warn_zero_compstars_edge(
                target_cid=target_cid,
                target=target,
                chip_fw=chip_fw,
                chip_fh=chip_fh,
                chip_interior_margin_px=int(chip_interior_margin_px),
            )
            return None
    if "catalog_id" in candidates_pre.columns:
        candidates_pre = candidates_pre.sort_values("catalog_id", kind="mergesort").reset_index(
            drop=True
        )
    return candidates_pre, float(used_mag_tol)

def _bootstrap_phase1_csv_cache(
    per_frame_csv_paths: list[Path],
    csv_cache: dict[str, pd.DataFrame] | None,
    *,
    flux_col: str,
    avail_cols: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    if avail_cols is None:
        avail_cols = list(_PHASE_USECOLS_PERFRAME)
    if csv_cache is None:
        # Note: csv_cache should be passed from Phase 1 entry to avoid duplicate reads
        #       when multiple code paths call with csv_cache=None. TODO-CACHE-CENTRAL.
        csv_cache = {}
        for _csv_path in per_frame_csv_paths:
            try:
                _hdr = pd.read_csv(_csv_path, nrows=0)
                _use_cols = [c for c in avail_cols if c in _hdr.columns]
                _actual_flux = flux_col if flux_col in _hdr.columns else "flux"
                if _actual_flux not in _use_cols:
                    _use_cols.append(_actual_flux)
                _name_col = (
                    "name"
                    if "name" in _hdr.columns
                    else ("catalog_id" if "catalog_id" in _hdr.columns else "name")
                )
                if "mag" not in _use_cols and "mag" in _hdr.columns:
                    _use_cols.append("mag")
                if "psf_chi2" in _hdr.columns and "psf_chi2" not in _use_cols:
                    _use_cols.append("psf_chi2")
                if "fwhm_estimate_px" in _hdr.columns and "fwhm_estimate_px" not in _use_cols:
                    _use_cols.append("fwhm_estimate_px")
                if "peak_max_adu" in _hdr.columns and "peak_max_adu" not in _use_cols:
                    _use_cols.append("peak_max_adu")
                if (
                    "saturate_limit_adu_85pct" in _hdr.columns
                    and "saturate_limit_adu_85pct" not in _use_cols
                ):
                    _use_cols.append("saturate_limit_adu_85pct")
                # Gaia ID musi byt str - float64 straca cifry
                _dtype_pf: dict[str, type] = {}
                if "catalog_id" in _use_cols:
                    _dtype_pf["catalog_id"] = str
                if "name" in _use_cols:
                    _dtype_pf["name"] = str
                _df0 = pd.read_csv(
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

                get_except_fix_counters().phase1_csv_cache_skip += 1
                logging.error("[COMP] phase1 CSV cache skip %s: %s", _csv_path, exc)
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
    _sorted_cids = sorted(cand_ids)
    flux_map: dict[str, list[float]] = {cid: [] for cid in _sorted_cids}
    bjd_map: dict[str, list[float]] = {cid: [] for cid in _sorted_cids}
    n_frames_loaded = 0
    contamination_map: dict[str, float] = {}
    psf_chi2_map: dict[str, list[float]] = {cid: [] for cid in _sorted_cids}
    fwhm_map: dict[str, list[float]] = {cid: [] for cid in _sorted_cids}
    frame_fwhm_medians: list[float] = []
    peak_over_map: dict[str, int] = {cid: 0 for cid in _sorted_cids}
    peak_total_map: dict[str, int] = {cid: 0 for cid in _sorted_cids}
    snr_map: dict[str, list[float]] = {cid: [] for cid in _sorted_cids}
    edge_bad_map: dict[str, int] = {cid: 0 for cid in _sorted_cids}
    edge_total_map: dict[str, int] = {cid: 0 for cid in _sorted_cids}
    _chip_w_eff: int | None = int(chip_fw) if chip_fw is not None else None
    _chip_h_eff: int | None = int(chip_fh) if chip_fh is not None else None
    _edge_log_done = False
    _use_vectorized = len(cand_ids) >= 50
    from config import AppConfig

    _cfg_sat = AppConfig()

    for csv_path in sort_per_frame_csv_paths(per_frame_csv_paths, csv_cache):
        df = csv_cache.get(str(csv_path))
        if df is None or df.empty:
            continue
        try:
            name_col = "name" if "name" in df.columns else ("catalog_id" if "catalog_id" in df.columns else "name")
            actual_flux_col = flux_col if flux_col in df.columns else "flux"

            if (_chip_w_eff is None or _chip_h_eff is None) and ("x" in df.columns and "y" in df.columns):
                try:
                    _xmax = float(pd.to_numeric(df["x"], errors="coerce").max())
                    _ymax = float(pd.to_numeric(df["y"], errors="coerce").max())
                except Exception:  # noqa: BLE001
                    _xmax, _ymax = float("nan"), float("nan")
                if math.isfinite(_xmax) and _xmax > 0:
                    _chip_w_eff = max(int(_chip_w_eff or 0), int(math.ceil(_xmax)) + 2)
                if math.isfinite(_ymax) and _ymax > 0:
                    _chip_h_eff = max(int(_chip_h_eff or 0), int(math.ceil(_ymax)) + 2)

            have_edge_cols = (
                "x" in df.columns
                and "y" in df.columns
                and "sky_annulus_r_out_px" in df.columns
                and _chip_w_eff is not None
                and _chip_h_eff is not None
                and int(_chip_w_eff) > 0
                and int(_chip_h_eff) > 0
            )
            if have_edge_cols and not _edge_log_done:
                logging.info(
                    f"[EDGE CHECK] chip={int(_chip_w_eff)}x{int(_chip_h_eff)}px, "
                    "annulus outer pouzity per-frame z sky_annulus_r_out_px"
                )
                _edge_log_done = True

            _cand = df[df[name_col].isin(cand_ids)]

            if "peak_max_adu" in df.columns and "saturate_limit_adu_85pct" in df.columns and not _cand.empty:
                if _use_vectorized:
                    sp = _cand[[name_col, "peak_max_adu", "saturate_limit_adu_85pct"]].copy()
                    sp["_peak"] = pd.to_numeric(sp["peak_max_adu"], errors="coerce")
                    sp["_limit"] = pd.to_numeric(sp["saturate_limit_adu_85pct"], errors="coerce")
                    sp = sp[sp["_limit"].gt(0) & sp["_peak"].notna() & sp["_limit"].notna()]
                    if not sp.empty:
                        _adm_frac = 0.70
                        _sat_frac_col = 0.85
                        try:
                            _adm_frac = float(getattr(_cfg_sat, "admission_sat_peak_frac", 0.70))
                        except (TypeError, ValueError):
                            _adm_frac = 0.70
                        try:
                            _sat_frac_col = float(getattr(_cfg_sat, "saturate_limit_fraction", 0.85))
                        except (TypeError, ValueError):
                            _sat_frac_col = 0.85
                        _adm_mult = _adm_frac / _sat_frac_col if _sat_frac_col > 0 else (0.70 / 0.85)
                        sp["_over"] = sp["_peak"] > sp["_limit"] * _adm_mult
                        for cid, n_tot in sp.groupby(name_col, sort=True).size().items():
                            cid_s = str(cid)
                            peak_total_map[cid_s] = int(peak_total_map.get(cid_s, 0)) + int(n_tot)
                        for cid, n_over in sp.loc[sp["_over"]].groupby(name_col, sort=True).size().items():
                            cid_s = str(cid)
                            peak_over_map[cid_s] = int(peak_over_map.get(cid_s, 0)) + int(n_over)
                else:
                    for _, _row in _cand.iterrows():
                        _cid = str(_row[name_col])
                        peak = float(_row.get("peak_max_adu", float("nan")))
                        limit = float(_row.get("saturate_limit_adu_85pct", float("nan")))
                        if math.isfinite(peak) and math.isfinite(limit) and limit > 0:
                            peak_total_map[_cid] = int(peak_total_map.get(_cid, 0)) + 1
                            _adm_frac = 0.70
                            _sat_frac_col = 0.85
                            try:
                                _adm_frac = float(getattr(_cfg_sat, "admission_sat_peak_frac", 0.70))
                            except (TypeError, ValueError):
                                _adm_frac = 0.70
                            try:
                                _sat_frac_col = float(getattr(_cfg_sat, "saturate_limit_fraction", 0.85))
                            except (TypeError, ValueError):
                                _sat_frac_col = 0.85
                            _adm_mult = _adm_frac / _sat_frac_col if _sat_frac_col > 0 else (0.70 / 0.85)
                            if peak > limit * _adm_mult:
                                peak_over_map[_cid] = int(peak_over_map.get(_cid, 0)) + 1

            if "psf_chi2" in df.columns and not _cand.empty:
                if _use_vectorized:
                    sp = _cand[[name_col, "psf_chi2"]].copy()
                    sp["_chi2"] = pd.to_numeric(sp["psf_chi2"], errors="coerce")
                    sp = sp[sp["_chi2"].gt(0)]
                    for cid, vals in sp.groupby(name_col, sort=True)["_chi2"]:
                        psf_chi2_map[str(cid)].extend(vals.astype(float).tolist())
                else:
                    for _, _row in _cand.iterrows():
                        _cid = str(_row[name_col])
                        _chi2 = float(_row.get("psf_chi2", float("nan")))
                        if math.isfinite(_chi2) and _chi2 > 0:
                            psf_chi2_map[_cid].append(_chi2)

            if "fwhm_estimate_px" in df.columns:
                _fwhm_col = pd.to_numeric(df["fwhm_estimate_px"], errors="coerce")
                _frame_fwhm_med = float(_fwhm_col.median())
                if math.isfinite(_frame_fwhm_med) and _frame_fwhm_med > 0:
                    frame_fwhm_medians.append(_frame_fwhm_med)
                if not _cand.empty:
                    if _use_vectorized:
                        sp = _cand[[name_col, "fwhm_estimate_px"]].copy()
                        sp["_fwhm"] = pd.to_numeric(sp["fwhm_estimate_px"], errors="coerce")
                        sp = sp[sp["_fwhm"].gt(0)]
                        for cid, vals in sp.groupby(name_col, sort=True)["_fwhm"]:
                            fwhm_map[str(cid)].extend(vals.astype(float).tolist())
                    else:
                        for _, _row in _cand.iterrows():
                            _cid = str(_row[name_col])
                            _fwhm = float(_row.get("fwhm_estimate_px", float("nan")))
                            if math.isfinite(_fwhm) and _fwhm > 0:
                                fwhm_map[_cid].append(_fwhm)

            sub = df[df[name_col].isin(cand_ids) & df[actual_flux_col].gt(0)].copy()
            if sub.empty:
                continue

            bin_meds, frame_med, used_mag_bins = build_frame_bin_medians(
                df, flux_col=actual_flux_col
            )
            if used_mag_bins:
                if not bin_meds:
                    continue
            elif not math.isfinite(frame_med) or frame_med <= 0:
                continue

            n_frames_loaded += 1

            if _use_vectorized:
                sub_work = assign_relative_flux(
                    sub,
                    flux_col=actual_flux_col,
                    bin_meds=bin_meds,
                    frame_med=frame_med,
                    id_col=name_col,
                )
                _rel_ok = sub_work["_rel"].notna() & np.isfinite(sub_work["_rel"].to_numpy(dtype=np.float64))
                _rel_ok = _rel_ok & sub_work["_rel"].gt(0)

                if have_edge_cols:
                    x0 = pd.to_numeric(sub_work["x"], errors="coerce")
                    y0 = pd.to_numeric(sub_work["y"], errors="coerce")
                    r_out = pd.to_numeric(sub_work["sky_annulus_r_out_px"], errors="coerce")
                    w = float(int(_chip_w_eff))
                    h = float(int(_chip_h_eff))
                    _edge_valid = (
                        x0.notna()
                        & y0.notna()
                        & r_out.notna()
                        & r_out.gt(0)
                        & np.isfinite(x0.to_numpy(dtype=np.float64))
                        & np.isfinite(y0.to_numpy(dtype=np.float64))
                        & np.isfinite(r_out.to_numpy(dtype=np.float64))
                    )
                    _edge_ok = (
                        _edge_valid
                        & (x0 - r_out >= 0.0)
                        & (x0 + r_out <= w)
                        & (y0 - r_out >= 0.0)
                        & (y0 + r_out <= h)
                    )
                    sub_work["_edge_count"] = _edge_valid.astype(np.int64)
                    sub_work["_edge_bad"] = (_edge_valid & ~_edge_ok).astype(np.int64)
                else:
                    sub_work["_edge_count"] = 0
                    sub_work["_edge_bad"] = 0

                if "dao_flux" in sub_work.columns:
                    flux_snr = pd.to_numeric(sub_work["dao_flux"], errors="coerce")
                    flux_snr = flux_snr.where(flux_snr.notna(), sub_work["_raw_flux"])
                else:
                    flux_snr = sub_work["_raw_flux"].copy()
                sky = pd.to_numeric(
                    sub_work.get("noise_floor_adu", pd.Series(0.0, index=sub_work.index)),
                    errors="coerce",
                )
                r_ap = pd.to_numeric(
                    sub_work.get("aperture_r_px", pd.Series(7.0, index=sub_work.index)),
                    errors="coerce",
                )
                area = np.pi * r_ap * r_ap
                denom = flux_snr + np.maximum(0.0, sky) * area
                _snr_ok = (
                    flux_snr.gt(0)
                    & sky.notna()
                    & area.notna()
                    & np.isfinite(flux_snr.to_numpy(dtype=np.float64))
                    & np.isfinite(sky.to_numpy(dtype=np.float64))
                    & np.isfinite(area.to_numpy(dtype=np.float64))
                    & denom.gt(0)
                )
                sub_work["_snr"] = np.where(_snr_ok, flux_snr / np.sqrt(denom), np.nan)

                if have_edge_cols:
                    for cid, grp in sub_work.groupby(name_col, sort=True):
                        cid_s = str(cid)
                        edge_total_map[cid_s] = int(edge_total_map.get(cid_s, 0)) + int(grp["_edge_count"].sum())
                        edge_bad_map[cid_s] = int(edge_bad_map.get(cid_s, 0)) + int(grp["_edge_bad"].sum())

                for cid, grp in sub_work.loc[_rel_ok].groupby(name_col, sort=True):
                    cid_s = str(cid)
                    flux_map[cid_s].extend(grp["_rel"].astype(float).tolist())
                    if "bjd_tdb_mid" in grp.columns:
                        bjd_map[cid_s].extend(
                            pd.to_numeric(grp["bjd_tdb_mid"], errors="coerce").astype(float).tolist()
                        )

                for cid, grp in sub_work.groupby(name_col, sort=True):
                    cid_s = str(cid)
                    snr_vals = grp["_snr"].to_numpy(dtype=np.float64)
                    snr_vals = snr_vals[np.isfinite(snr_vals)]
                    if snr_vals.size > 0:
                        snr_map[cid_s].extend(snr_vals.astype(float).tolist())
            else:
                sub_work = assign_relative_flux(
                    sub,
                    flux_col=actual_flux_col,
                    bin_meds=bin_meds,
                    frame_med=frame_med,
                    id_col=name_col,
                )
                _bin_keys = np.fromiter(bin_meds.keys(), dtype=np.int64) if bin_meds else np.array([], dtype=np.int64)

                for _, row in sub_work.iterrows():
                    cid = str(row[name_col])
                    if cid not in cand_ids:
                        continue
                    raw_flux = float(row.get("_raw_flux", row[actual_flux_col]))
                    if not math.isfinite(raw_flux) or raw_flux <= 0:
                        continue

                    if have_edge_cols:
                        try:
                            x0 = float(pd.to_numeric(row.get("x"), errors="coerce"))
                            y0 = float(pd.to_numeric(row.get("y"), errors="coerce"))
                            r_out = float(pd.to_numeric(row.get("sky_annulus_r_out_px"), errors="coerce"))
                        except Exception:  # noqa: BLE001
                            x0, y0, r_out = float("nan"), float("nan"), float("nan")
                        if math.isfinite(x0) and math.isfinite(y0) and math.isfinite(r_out) and r_out > 0:
                            edge_total_map[cid] = int(edge_total_map.get(cid, 0)) + 1
                            w = float(int(_chip_w_eff))
                            h = float(int(_chip_h_eff))
                            edge_ok = (
                                (x0 - r_out >= 0.0)
                                and (x0 + r_out <= w)
                                and (y0 - r_out >= 0.0)
                                and (y0 + r_out <= h)
                            )
                            if not edge_ok:
                                edge_bad_map[cid] = int(edge_bad_map.get(cid, 0)) + 1

                    flux_snr = float(row.get("dao_flux", raw_flux))
                    if not math.isfinite(flux_snr):
                        flux_snr = raw_flux
                    sky = float(row.get("noise_floor_adu", 0.0))
                    r_ap = float(row.get("aperture_r_px", 7.0))
                    area = math.pi * r_ap * r_ap if math.isfinite(r_ap) and r_ap > 0 else float("nan")
                    if math.isfinite(flux_snr) and flux_snr > 0 and math.isfinite(sky) and math.isfinite(area):
                        denom = flux_snr + max(0.0, sky) * area
                        if denom > 0:
                            snr = flux_snr / math.sqrt(denom)
                            if math.isfinite(snr):
                                snr_map[cid].append(float(snr))

                    rel = float(row.get("_rel", float("nan")))
                    if math.isfinite(rel) and rel > 0:
                        flux_map[cid].append(rel)
                        if "bjd_tdb_mid" in row.index:
                            _bjd_v = float(pd.to_numeric(row.get("bjd_tdb_mid"), errors="coerce"))
                            if math.isfinite(_bjd_v):
                                bjd_map[cid].append(_bjd_v)

        except (ValueError, TypeError, KeyError, IndexError) as exc:
            from except_fix_counters import get_except_fix_counters

            get_except_fix_counters().comp_frame_accumulation_skip += 1
            logging.error("[COMP] frame metrics accumulation failed %s: %s", csv_path, exc)
            continue

    logging.info(
        "[PERF-4B] _accumulate_per_frame_comp_metrics: %d frames x %d candidates (%s)",
        n_frames_loaded,
        len(cand_ids),
        "vectorized" if _use_vectorized else "iterrows (small N)",
    )
    return {
        "flux_map": flux_map,
        "bjd_map": bjd_map,
        "n_frames_loaded": n_frames_loaded,
        "contamination_map": contamination_map,
        "psf_chi2_map": psf_chi2_map,
        "fwhm_map": fwhm_map,
        "frame_fwhm_medians": frame_fwhm_medians,
        "peak_over_map": peak_over_map,
        "peak_total_map": peak_total_map,
        "snr_map": snr_map,
        "edge_bad_map": edge_bad_map,
        "edge_total_map": edge_total_map,
        "_chip_w_eff": _chip_w_eff,
        "_chip_h_eff": _chip_h_eff,
    }


def _apply_comp_metric_hard_filters(
    flux_map: dict[str, list[float]],
    peak_over_map: dict[str, int],
    peak_total_map: dict[str, int],
    snr_map: dict[str, list[float]],
    psf_chi2_map: dict[str, list[float]],
    fwhm_map: dict[str, list[float]],
    frame_fwhm_medians: list[float],
    edge_bad_map: dict[str, int],
    edge_total_map: dict[str, int],
    *,
    target_cid: str,
    edge_bad_frame_frac_max: float,
    max_psf_chi2: float,
    max_fwhm_factor: float,
    dilution_map: dict[str, dict[str, Any]] | None = None,
    cfg: Any | None = None,
    comp_quality_notes: dict[str, str] | None = None,
    sat_may_exclude: bool = True,
) -> Any:
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
                    f"[FAZA 1] Saturacia filter: vyluceny {cid} "
                    f"({over}/{total} framov nad 85% limitom)"
                )
    else:
        logging.info("[SAT-DIAG] saturation exclusion disabled (Tier 3 / unverified limit)")
    if _sat_rejected:
        logging.info(
            "[BATCH-E E.5] saturation gate excluded %d comps (admission_sat_peak_frac gate)",
            len(_sat_rejected),
        )
        logging.info(f"[FAZA 1] Celkom vylucenych kvoli saturacii: {len(_sat_rejected)}")

    # Filter EDGE: vyluc kandidatov, ktorych sky annulus casto vycnieva mimo cip
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
                    f"[FAZA 1] Edge/annulus filter: vyluceny {cid} "
                    f"({bad_e}/{total_e} framov mimo cip, frac={bad_frac:.2f} > {bad_thr:.2f})"
                )
    if _edge_rejected:
        logging.info(f"[FAZA 1] Celkom vylucenych kvoli edge/annulus: {len(_edge_rejected)}")

    # COMP-ADMIT-03: SNR / PSF / FWHM / GS11 dilution are not admission cuts.
    # Low SNR and blends raise sigma_rms (weight); dilution is not a gate.
    _ = (snr_map, max_psf_chi2, max_fwhm_factor, frame_fwhm_medians, dilution_map, cfg, comp_quality_notes)
    _snr_rejected: set[str] = set()
    _b_rejected: set[str] = set()
    _gs11_rejected: set[str] = set()
    return flux_map, _b_rejected


def _compute_comp_contamination_map(
    flux_map: dict[str, list[float]],
    ms: pd.DataFrame,
    *,
    target_cid: str,
    isolation_radius_px: float,
) -> dict[str, float]:
    # Filter C -> Contamination index (soft penalizacia v scoringu)
    # Namiesto hard-exclusion vypocitaj contamination ratio per kandidat.
    # Husta oblast neba: hard filter by vylucil vacsinu kandidatov.
    # Riesenie: crowding sa prejavi ako penalizacia v combined score (Krok 5).
    contamination_map: dict[str, float] = {}
    if isolation_radius_px > 0 and "x" in ms.columns and "y" in ms.columns:
        ms_reset = ms.reset_index(drop=True)
        _id_col_ms = "catalog_id" if "catalog_id" in ms_reset.columns else "name"

        # Flux proxy: dao_flux > flux > phot_g_mean_mag (mag -> relativny flux)
        _flux_col_ms = next((fc for fc in ("dao_flux", "flux") if fc in ms_reset.columns), None)
        _mag_col_ms = next(
            (mc for mc in ("phot_g_mean_mag", "catalog_mag", "mag") if mc in ms_reset.columns),
            None,
        )

        # Zostavime vektory pre rychly vypocet vzdialenosti
        _ms_x_all = pd.to_numeric(ms_reset["x"], errors="coerce").to_numpy(dtype=np.float64)
        _ms_y_all = pd.to_numeric(ms_reset["y"], errors="coerce").to_numpy(dtype=np.float64)

        if _flux_col_ms:
            _ms_flux_all = pd.to_numeric(ms_reset[_flux_col_ms], errors="coerce").to_numpy(dtype=np.float64)
        elif _mag_col_ms:
            _mags_all = pd.to_numeric(ms_reset[_mag_col_ms], errors="coerce").to_numpy(dtype=np.float64)
            _ms_flux_all = np.where(np.isfinite(_mags_all), 10 ** (-0.4 * _mags_all), np.nan)
        else:
            _ms_flux_all = np.ones(len(ms_reset))

        _ms_mag_all = (
            pd.to_numeric(ms_reset[_mag_col_ms], errors="coerce").to_numpy(dtype=np.float64)
            if _mag_col_ms
            else np.full(len(ms_reset), np.nan, dtype=np.float64)
        )

        # Lookup: catalog_id -> riadok index v ms_reset
        _cid_to_idx: dict[str, int] = {}
        for _ri, _rrow in ms_reset.iterrows():
            _rcid = _normalize_id_value(_rrow.get(_id_col_ms, ""))
            if _rcid:
                _cid_to_idx[_rcid] = int(_ri)

        for _cid in sorted(flux_map.keys()):
            _ci = _cid_to_idx.get(_cid)
            if _ci is None:
                continue
            _cx = _ms_x_all[_ci]
            _cy = _ms_y_all[_ci]
            _cflux = _ms_flux_all[_ci]
            if not (math.isfinite(_cx) and math.isfinite(_cy)):
                continue
            if not math.isfinite(_cflux) or _cflux <= 0:
                continue

            _dx = _ms_x_all - _cx
            _dy = _ms_y_all - _cy
            _dists = np.sqrt(_dx * _dx + _dy * _dy)
            _neighbor_mask = (
                (_dists > 0.5)
                & (_dists <= isolation_radius_px)
                & np.isfinite(_ms_flux_all)
                & (_ms_flux_all > 0)
            )
            # Zahrnut len susedov do 3 mag od kandidata
            mag_cand = float(_ms_mag_all[_ci]) if _ci < len(_ms_mag_all) else float("nan")
            if math.isfinite(mag_cand):
                _neighbor_mask = _neighbor_mask & (
                    ~np.isfinite(_ms_mag_all) | ((_ms_mag_all - mag_cand) <= 3.0)
                )
            if not np.any(_neighbor_mask):
                contamination_map[_cid] = 0.0
                continue

            # Contamination = sucet flux susedov / flux kandidata
            # (sucet, nie maximum - viac slabych susedov = vacsi efekt)
            _neighbor_flux_sum = float(np.sum(_ms_flux_all[_neighbor_mask]))
            contamination_map[_cid] = min(_neighbor_flux_sum / _cflux, 2.0)  # cap na 2.0 (200%)

        if contamination_map:
            _cont_vals = list(contamination_map.values())
            logging.debug(
                f"[FAZA 1] Target {target_cid}: contamination index "
                f"median={float(np.median(_cont_vals)):.3f}, "
                f"max={max(_cont_vals):.3f} "
                f"(isolation_radius={isolation_radius_px:.0f}px)"
            )
    return contamination_map


_MAD_SIGMA_SCALE = 1.4826


def _mad_sigma(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if int(v.size) < 3:
        return float(np.std(v)) if int(v.size) > 0 else float("inf")
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    if mad > 0:
        return float(_MAD_SIGMA_SCALE * mad)
    return float(np.std(v)) if float(np.std(v)) > 0 else float("inf")


def _flux_series_to_mag_bjd(
    flux_map: dict[str, list[float]],
    bjd_map: dict[str, list[float]],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Per-candidate mag LC and BJD from relative flux series (Phase-1 frame norm)."""
    mag_lc: dict[str, np.ndarray] = {}
    bjd_lc: dict[str, np.ndarray] = {}
    for cid in sorted(flux_map.keys()):
        fluxes = flux_map.get(cid) or []
        if len(fluxes) < 3:
            continue
        arr = np.asarray(fluxes, dtype=np.float64)
        ok = np.isfinite(arr) & (arr > 0)
        if int(ok.sum()) < 3:
            continue
        bjds = bjd_map.get(cid) or []
        if len(bjds) == len(fluxes):
            b = np.asarray(bjds, dtype=np.float64)
        else:
            b = np.arange(len(arr), dtype=np.float64)
        mag_lc[cid] = -2.5 * np.log10(arr)
        bjd_lc[cid] = b
    return mag_lc, bjd_lc


def _common_mode_detrend_mag_lcs(
    mag_lc: dict[str, np.ndarray],
    bjd_lc: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Sorted-BJD linear common-mode detrend (Honeycutt 1992 em(e); stability-check style)."""
    from scipy.stats import linregress as _lr

    cids = sorted(mag_lc.keys())
    stacks_bjd: list[np.ndarray] = []
    stacks_mag: list[np.ndarray] = []
    for cid in cids:
        b = bjd_lc.get(cid)
        m = mag_lc.get(cid)
        if b is None or m is None:
            continue
        ok = np.isfinite(b) & np.isfinite(m)
        if int(ok.sum()) < 20:
            continue
        bo, mo = b[ok], m[ok]
        order = np.argsort(bo, kind="mergesort")
        stacks_bjd.append(bo[order])
        stacks_mag.append(mo[order])
    if len(stacks_mag) < 2:
        return {cid: mag_lc[cid].copy() for cid in cids}
    ref_bjd = stacks_bjd[int(np.argmax([len(x) for x in stacks_bjd]))]
    stack = [np.interp(ref_bjd, b, m) for b, m in zip(stacks_bjd, stacks_mag, strict=True)]
    common = np.median(np.vstack(stack), axis=0)
    lr = _lr(ref_bjd, common)
    out: dict[str, np.ndarray] = {}
    for cid in cids:
        b = bjd_lc.get(cid)
        m = mag_lc.get(cid)
        if b is None or m is None:
            continue
        ok = np.isfinite(b) & np.isfinite(m)
        md = m.copy()
        md[ok] = m[ok] - (lr.slope * b[ok] + lr.intercept) + float(common.mean())
        out[cid] = md
    return out


def _iterative_ensemble_clip_cm_residual(
    flux_map: dict[str, list[float]],
    bjd_map: dict[str, list[float]],
    provisional_rms: dict[str, float],
    *,
    clip_sigma: float,
    n_comp_min: int,
    max_iter: int = 5,
    min_final: int | None = None,
) -> tuple[dict[str, float], dict[str, int]] | None:
    """Passthrough: no ensemble sigma-clip (zero-clipping policy 2026-08-12)."""
    _ = (flux_map, bjd_map, clip_sigma, n_comp_min, max_iter)
    _min_final = max(1, int(min_final if min_final is not None else n_comp_min))
    active_rms = {
        cid: float(rms)
        for cid, rms in provisional_rms.items()
        if math.isfinite(float(rms))
    }
    if len(active_rms) < _min_final:
        return None
    n_candidates = int(len(active_rms))
    meta = {
        "comp_pool_n_candidates": n_candidates,
        "comp_pool_n_clipped": 0,
        "comp_pool_n_final": n_candidates,
        "comp_clip_iterations": 0,
    }
    logging.info(
        "[COMP] Ensemble clip disabled (zero-clipping): keeping %d comps",
        n_candidates,
    )
    return active_rms, meta


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
    skip_apriori_rms: bool = False,
) -> Any:
    # -- Krok 2b: Airmass detrending --
    # Polynomicky fit (stupen 2) na casovy rad relativneho flux odstrani
    # systematicky airmass trend. Residualy = skutocna fotometricka variabilita.
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

    # -- Krok 3: RMS scatter per kandidat --
    rms_map: dict[str, float] = {}
    for cid, vals in sorted(flux_map.items(), key=lambda kv: str(kv[0])):
        if len(vals) < min_frames:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        rms = robust_comp_rms(arr)
        if math.isfinite(rms):
            rms_map[cid] = rms

    # Detekuj "isolated bin" artefakty:
    # ak comp_rms vyjde nerealne nizke (typicky ~0), ale hviezda ma vela merani,
    # je to casto dosledok normalizacie v prilis riedkom brightness bine (norm_med ~ raw_flux).
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
                "[COMP SELECT] %s: comp_rms < %.1e pri %d framoch -> isolated bin -> excluded",
                str(cid),
                float(ISOLATED_BIN_RMS_FLOOR),
                int(nfr),
            )

    # Zoradene RMS pre fallback kroky
    sorted_rms_map: dict[str, float] = dict(
        sorted(rms_map.items(), key=lambda kv: (float(kv[1]), str(kv[0])))
    )

    if skip_apriori_rms:
        if not rms_map:
            logging.warning(
                f"[FAZA 1] Target {target_cid}: iterative clip - no candidates with enough frames."
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

    # COMP-ADMIT-03: max_comp_rms is no longer an admission hard cut. Scatter
    # enters continuous weights (sigma_rms). Keep all stars with a finite RMS.
    if math.isfinite(max_comp_rms) and max_comp_rms > 0:
        logging.info(
            "[FAZA 1] Target %s: COMP-ADMIT-03 skips hard RMS cut "
            "(max_comp_rms=%.3f retained as diagnostic only); n=%d",
            target_cid,
            float(max_comp_rms),
            len(rms_map),
        )

    # Previously: relax-above-gate was forbidden; now there is no RMS gate.
    if len(rms_map) < n_comp_min:
        logging.warning(
            f"[FAZA 1] Target {target_cid}: len {len(rms_map)} kandidatov "
            f"s dostatkom snimok < n_comp_min={n_comp_min}."
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
    n_comp_min: int,
    rms_outlier_sigma: float,
    chip_fw: int | None,
    chip_fh: int | None,
    chip_interior_margin_px: int,
) -> dict[str, float] | None:
    # -- Krok 4: no iterative MAD outlier rejection (zero-clipping policy 2026-08-12) --
    _ = (rms_outlier_sigma, n_comp_min)
    id_col_cand = "name" if "name" in candidates.columns else ("catalog_id" if "catalog_id" in candidates.columns else "name")
    cand_ids = set(candidates[id_col_cand].astype(str).str.strip())
    active = {
        cid: rms
        for cid, rms in sorted(rms_map.items(), key=lambda kv: (float(kv[1]), str(kv[0])))
        if cid in cand_ids
    }
    if len(active) < 2:
        logging.warning(
            f"[FAZA 1] {target_cid}: len {len(active)} comp po RMS filtre "
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
    contamination_map: dict[str, float],
    *,
    id_col_cand: str,
    mag_t: float,
    target_bprp_eff: float,
    t_bp_tgt: float,
    _individual_tier: Callable[[float], int],
    cfg: AppConfig | None = None,
) -> Any:
    # -- Krok 5: Finalny vyber --
    # Ranking: Broeg (1/comp_rms). Tier je len informativny stlpec.
    #
    # Literatura pre comp_score:
    # - Broeg, Fernandez & Neuhauser (2005): Astron. Nachr. 326, 134
    #   DOI: 10.1002/asna.200410350
    # - Young, A.T. (1967): AJ 72, 747 - scintilacny sum
    # - Hardie, R.H. (1962): in Astronomical Techniques (Stars & Stellar Systems vol. II)
    #   - color term korekcie pri diferencialnej fotometrii
    # - AAVSO CCD Photometry Guide (2023), kap. 5.5 a 6
    #   https://www.aavso.org/ccd-photometry-guide
    score_map: dict[str, float] = {}
    tier_map: dict[str, str] = {}
    try:
        _contam_penalty_k = float((cfg or AppConfig()).comp_contamination_penalty_k)
    except Exception:  # noqa: BLE001
        _contam_penalty_k = 3.0
    if not math.isfinite(_contam_penalty_k) or _contam_penalty_k < 0:
        _contam_penalty_k = 3.0
    for cid, rms in sorted(active.items(), key=lambda kv: (float(kv[1]), str(kv[0]))):
        row = candidates[candidates[id_col_cand].astype(str).str.strip() == cid]
        if row.empty:
            continue
        r0 = row.iloc[0]
        contamination = float(contamination_map.get(cid, 0.0)) if contamination_map else 0.0
        if not math.isfinite(contamination) or contamination < 0.0:
            contamination = 0.0

        comp_rms = float(rms) if math.isfinite(float(rms)) else float("nan")
        try:
            comp_bprp = float(pd.to_numeric(r0.get("bp_rp"), errors="coerce"))
        except Exception:  # noqa: BLE001
            comp_bprp = float("nan")
        delta_bprp_tier = (
            abs(float(comp_bprp) - float(target_bprp_eff))
            if (math.isfinite(float(comp_bprp)) and math.isfinite(float(target_bprp_eff)))
            else float("nan")
        )
        _tier = int(_individual_tier(float(delta_bprp_tier)))
        tier_map[str(cid)] = int(_tier)

        # Broeg base score + tier penalty.
        rms_f = float(comp_rms) if math.isfinite(float(comp_rms)) else float("nan")
        # Safe: rms_f > 1e-6 guard prevents division by zero in weight computation.
        broeg_score = (1.0 / (rms_f**2)) if (math.isfinite(rms_f) and rms_f > 1e-6) else 0.0
        cont_penalty = math.exp(-float(_contam_penalty_k) * float(contamination))
        comp_score = float(broeg_score) * float(cont_penalty)

        logging.debug(
            "[COMP] %s: tier=%s, rms=%.4f, broeg=%.0f, contamination=%.3f, penalty=%.3f, score=%.0f",
            str(cid),
            str(_tier),
            float(rms_f) if math.isfinite(rms_f) else float("nan"),
            float(broeg_score),
            float(contamination),
            float(cont_penalty),
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
    t_bp_tgt: float,
    mag_t: float,
    _individual_tier: Callable[[float], int],
    _target_name: str,
    max_mag_diff_t1: float,
    max_mag_diff: float,
    gaia_db_path: str | None,
    vsx_local_db_path: str | None,
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
                float(getattr(_cfg_w, "snr_cog_isolation_fwhm", 3.0) or 3.0),
                float(_fw_nn) * float(getattr(_cfg_w, "snr_cog_isolation_fwhm", 3.0) or 3.0),
                int(len(candidate_pool_df)),
                int(fx.size),
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
                "[COMP] %s: ziadne T1/T2 comp, "
                "len %d TIER3/4 - LC bude vynechana",
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
            f"[FAZA 1] Target {target_cid}: po filtracii len {len(selected_ids)} "
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
    contamination_map: dict[str, float],
    flux_map: dict[str, list[float]],
    target_cid: str,
    target: pd.Series,
    target_bprp_eff: float,
    t_bp_tgt: float,
    sel_note: str,
    used_mag_tol: float,
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
    _b_rejected: set[str],
    final_lookup: pd.DataFrame | None,
    dilution_map: dict[str, dict[str, Any]] | None = None,
    comp_gs11_notes: dict[str, str] | None = None,
    cfg: AppConfig | None = None,
    clip_meta: dict[str, int] | None = None,
    comp_path: str = "default",
    per_target_rms_map: dict[str, float] | None = None,
) -> pd.DataFrame:
    _cfg_asm = cfg or AppConfig()
    # Zostav vystupny DataFrame
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
        if str(comp_path or "default").strip().lower() == "sparse_fallback":
            r["comp_rms"] = _pt_rms if math.isfinite(_pt_rms) else float("nan")
            r["comp_rms_fieldwide"] = _fw_rms if math.isfinite(_fw_rms) else float("nan")
        else:
            r["comp_rms"] = _fw_rms if math.isfinite(_fw_rms) else _pt_rms
            r["comp_rms_fieldwide"] = float("nan")
        r["comp_score"] = score_map.get(cid, float("nan"))
        # Ranking columns (new selection philosophy)
        try:
            _cont_v = contamination_map.get(cid) if contamination_map else None
            r["contamination_idx"] = (
                float(_cont_v)
                if _cont_v is not None and math.isfinite(float(_cont_v))
                else float("nan")
            )
        except (TypeError, ValueError):
            r["contamination_idx"] = float("nan")
        r["comp_n_frames"] = len(flux_map.get(cid_s, flux_map.get(cid, [])))
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
        if comp_gs11_notes and cid in comp_gs11_notes:
            _gs11n = str(comp_gs11_notes[cid])
            _sel = f"{_sel}; {_gs11n}" if _sel else _gs11n
        try:
            _mag_start = float(getattr(_cfg_asm, "phase01_comparison_max_mag_diff", float("nan")))
        except (TypeError, ValueError):
            _mag_start = float("nan")
        _relax = format_assignment_relax_provenance(
            used_mag_tol=float(used_mag_tol) if math.isfinite(float(used_mag_tol)) else float("nan"),
            mag_tol_start=_mag_start if math.isfinite(_mag_start) else None,
            best_tier=str(best_tier),
            comp_path=str(comp_path or "default"),
            n_t1=int(n_t1),
            n_t2=int(n_t2),
            n_t3=int(n_t3),
            n_t4=int(n_t4),
        )
        _sel = f"{_sel}; {_relax}" if _sel else _relax
        r["selection_note"] = _sel
        r["used_mag_tol"] = float(used_mag_tol) if math.isfinite(float(used_mag_tol)) else float("nan")
        r["comp_path"] = str(comp_path or "default").strip() or "default"
        if clip_meta:
            for _mk in (
                "comp_pool_n_candidates",
                "comp_pool_n_clipped",
                "comp_pool_n_final",
                "comp_clip_iterations",
            ):
                if _mk in clip_meta:
                    r[_mk] = int(clip_meta[_mk])
        rms_val = r.get("comp_rms", float("nan"))
        try:
            rms_f = float(pd.to_numeric(rms_val, errors="coerce"))
        except Exception:  # noqa: BLE001
            rms_f = float("nan")
        # COMP-ADMIT-03 / PRE-IMPL-01: persist the same sigma_eff weight Phase 2A uses.
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
            # Separation optional at Phase-1 write; c_dist=0 on wide refractive.
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
                "contamination_idx",
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
                "used_mag_tol",
                "comp_path",
                "comp_pool_n_candidates",
                "comp_pool_n_clipped",
                "comp_pool_n_final",
                "comp_clip_iterations",
                "selected_tier",
                "tier4_warning",
                "n_tier1",
                "n_tier2",
                "n_tier3",
                "n_tier4",
            ]
        )

    _total_rejected_b = len(_b_rejected) if "_b_rejected" in dir() else 0
    if _total_rejected_b > 0:
        logging.info(
            f"[FAZA 1] Target {target_cid}: blend filter B celkom vylucil "
            f"{_total_rejected_b} kandidatov"
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
        f"[FAZA 1] Target {target_cid} ({target.get('vsx_name','')}): "
        f"{len(out)} porovnavaciek | RMS min={out['comp_rms'].min():.4f} "
        f"max={out['comp_rms'].max():.4f} | {color_info}"
    )
    if color_extra:
        _log_line += f" | {color_extra}"
    logging.info(_log_line)
    return out
