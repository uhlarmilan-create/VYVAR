"""Moved from photometry_core.py (CONSOLIDATE-01E4). Facade re-exports these names."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import logging
import math
import numpy as np
import pandas as pd
from catalog_match_trust import is_wcs_untrusted_catalog_match_mode, normalize_catalog_match_mode
from infolog import log_event
from photometry_shared import _angular_distance_deg, _normalize_gaia_id, _target_display_name

from photometry_phase2a import (
    _Phase2AState,
    _draft_dir_from_phase2a_paths,
    _measured_aperture_from_proc_cache,
    _resolve_photometric_aperture_px_for_gs11,
    apply_reporting_postprocess,
    compute_aperture_correction,
    democratic_detrend_lc,
    fit_color_term_c1,
    read_flux_from_csv,
)
from photometry_lightcurve import (
    _ac_summary_fields,
    _append_ct_prototype_row,
    _check_color_term_extrapolation,
    _color_term_cat_inst_scatter_pair,
    _combine_err_with_ensemble_scatter_keyed,
    _ct_prototype_enabled,
    _ensemble_scatter_by_source_file,
    _err_budget_components_keyed,
    _exclude_err_scatter_unmatched_epochs,
    _get_comp_bjd_series,
    _get_lc,
    _load_adaptive_blend_map,
    _phase2a_skip_empty_comps_target,
    _recompute_bjd_hjd_with_status,
    _route_lc_per_frame_err,
    apply_color_term,
    check_comparison_stability,
    compute_lc_flux_method,
    compute_lc_rms_ooe,
    ct_ensemble_reference_maps,
    ensemble_normalize,
    pytics_iterative_weights,
    save_cutout_png,
    save_lightcurve_csv,
    save_lightcurve_png,
    save_target_field_map_png,
    savgol_detrend_lc,
    temporal_bin_comp_lc,
)

from photometry_core import (
    ERR_BKG_MODE_EMPIRICAL,
    LOGGER,
)


def _phase2a_process_one_target(
    target_row: Any,
    *,
    ti: int,
    state: _Phase2AState,
    summary_rows: list,
    n_lc: int,
    lc_dir: Path,
    output_dir: Path,
    progress_cb: Any,
    masterstar_fits_path: Path,
    annulus_inner_fwhm: float,
    annulus_outer_fwhm: float,
    outlier_sigma: float,
    stability_sigma: float,
    _apt_fw: float,
    _save_png: bool,
    ac_sign_logged: list[bool],
) -> tuple[list, int]:
    """Process one target through the full Phase 2A photometry pipeline.

    Returns updated (summary_rows, n_lc).
    """
    def _p2(msg: str) -> None:
        if progress_cb is None:
            return
        try:
            progress_cb(str(msg))
        except UnicodeEncodeError:
            progress_cb(str(msg).encode("ascii", "backslashreplace").decode("ascii"))

    _comp_index = state._comp_index
    target_bp_rp_by_cid = state.target_bp_rp_by_cid
    csv_files = state.csv_files
    _phase2a_csv_cache = state._phase2a_csv_cache
    _phase2a_lookup_cache = state._phase2a_lookup_cache
    frame_time_lookup = state.frame_time_lookup
    fwhm_px = state.fwhm_px
    apertures_px = state.apertures_px
    star_xy = state.star_xy
    chip_fw = state.chip_fw
    chip_fh = state.chip_fh
    _ms_data = state._ms_data
    _flux_matrix = state._flux_matrix
    obs_group = state.obs_group
    _gain_phot = state._gain_phot
    _rn_phot = state._rn_phot
    sat_limit_resolved = state.sat_limit_resolved
    _aligned_dir_2a = state._aligned_dir_2a
    _cfg = state._cfg
    _nt = state._nt
    comp_df = state.comp_df
    _lunar = state.lunar_context

    target_cid = _normalize_gaia_id(target_row.get("catalog_id", ""))
    target_name = _target_display_name(target_row, fallback_cid=target_cid)
    target_vsx_type = str(target_row.get("vsx_type", "") or "").strip()
    _sp = target_row.get("skip_photometry", False)
    if isinstance(_sp, (bool, np.bool_)):
        skip_photo = bool(_sp)
    else:
        skip_photo = str(_sp).strip().lower() in ("1", "true", "yes", "t")
    _zf_row = str(target_row.get("zone_flag", "")).strip()
    _zf_low = _zf_row.lower()
    # When per-frame sat is ON, skip_photometry already encodes the decision;
    # do not re-force whole-star skip from master zone_flag.
    _pfs_on = bool(getattr(_cfg, "per_frame_saturation_enabled", False))
    # TARGET-DEPTH-02 outranks PFS: noise never enters photometry.
    # Saturation-zone skip is re-forced only when PFS is OFF (PFS already
    # encoded the saturation decision). Do not exempt the whole {saturated, noise} set.
    if _zf_low == "noise":
        skip_photo = True
    elif (not _pfs_on) and _zf_low == "saturated":
        skip_photo = True
    if progress_cb is not None and (
        ti == 1 or ti == _nt or (_nt > 1 and ti % max(1, _nt // 12) == 0)
    ):
        _p2(f"Faza 2A: ciel {ti}/{_nt}: {target_name[:50]}")
    if skip_photo:
        _sr_col = str(target_row.get("skip_reason", "") or "").strip()
        if _sr_col:
            _skip_reason = _sr_col
        elif _zf_low == "noise":
            _skip_reason = "zone_noise"
        else:
            _skip_reason = "saturovany ciel"
        logging.info(f"[FAZA 2A] Preskakujem fotometriu ({_skip_reason}): {target_name}")
        _skip_sum: dict[str, Any] = {
            "catalog_id": target_cid,
            "vsx_name": target_name,
            "zone_flag": _zf_row,
            "n_frames": 0,
            "n_good_comp": 0,
            "n_saturated": 0,
            "lc_rms": float("nan"),
            "lc_median_mag": float("nan"),
            "aperture_px": float("nan"),
            "am_slope": float("nan"),
            "am_detrended": False,
            "lc_csv": "",
            "lc_png": "",
        }
        if _pfs_on:
            _skip_sum["skip_reason"] = _skip_reason
            _scf = float(pd.to_numeric(target_row.get("sat_clean_frac"), errors="coerce"))
            _skip_sum["sat_clean_frac"] = _scf
            _skip_sum["per_frame_sat_fallback"] = bool(
                target_row.get("per_frame_sat_fallback", False)
            )
        summary_rows.append(_skip_sum)
        return summary_rows, n_lc
    logging.info(
        f"[FAZA 2A] Spustam: target={target_name}, "
        f"frames={len(csv_files)}, "
        f"apertura={_apt_fw * float(fwhm_px):.2f}px "
        f"(FWHM={float(fwhm_px):.3f}px x {_apt_fw:.2f})"
    )

    # Comp hviezdy pre tento target
    target_comps = _comp_index.get(target_cid, pd.DataFrame()).copy()
    _star_xy = dict(star_xy)

    if target_comps.empty:
        summary_rows = _phase2a_skip_empty_comps_target(
            target_cid=target_cid,
            target_name=target_name,
            zone_flag=_zf_row,
            summary_rows=summary_rows,
        )
        return summary_rows, n_lc

    comp_ids: list[str] = []
    _seen_comp: set[str] = set()
    for c in target_comps["catalog_id"].tolist():
        nc = _normalize_gaia_id(c)
        if nc and nc not in _seen_comp:
            _seen_comp.add(nc)
            comp_ids.append(nc)
    all_ids = [target_cid] + comp_ids

    # Katalogove magnitudy comp hviezd
    comp_catalog_mag = {
        _normalize_gaia_id(r["catalog_id"]): float(r.get("mag", float("nan")))
        for _, r in target_comps.iterrows()
    }
    _cfg_tw = _cfg.comp_tier_weights()
    tier_weights = {
        1: float(_cfg_tw[0]),
        2: float(_cfg_tw[1]),
        3: float(_cfg_tw[2]),
        4: float(_cfg_tw[3]),
    }
    for _k in list(tier_weights.keys()):
        try:
            _v = float(tier_weights[_k])
        except Exception:  # noqa: BLE001
            _v = float("nan")
        if not math.isfinite(_v) or _v <= 0:
            tier_weights[_k] = 0.01
        else:
            tier_weights[_k] = max(0.01, float(_v))

    comp_tier_map: dict[str, int] = {}
    for _, r in target_comps.iterrows():
        cid0 = _normalize_gaia_id(r["catalog_id"])
        try:
            t0 = int(pd.to_numeric(r.get("comp_tier", 4), errors="coerce") or 4)
        except Exception:  # noqa: BLE001
            t0 = 4
        comp_tier_map[cid0] = int(max(1, min(4, t0)))

    comp_rms_map: dict[str, float] = {}
    for _, r in target_comps.iterrows():
        cid0 = _normalize_gaia_id(r["catalog_id"])
        try:
            rms_raw = float(r.get("comp_rms", float("nan")))
        except Exception:  # noqa: BLE001
            rms_raw = float("nan")
        # COMP-ADMIT-03: do not bake tier into rms; colour/distance enter sigma_eff.
        comp_rms_map[cid0] = float(rms_raw)

    # Continuous weights: sigma_eff^2 = rms^2 + (c_col*|dBP-RP|)^2 + (c_dist*r)^2
    from comp_weights import resolve_comp_weight_coeffs, sigma_eff_mag, weight_from_sigma_eff  # noqa: PLC0415

    _tx = float(pd.to_numeric(target_row.get("x"), errors="coerce"))
    _ty = float(pd.to_numeric(target_row.get("y"), errors="coerce"))
    _tra = float(pd.to_numeric(target_row.get("ra_deg", target_row.get("ra")), errors="coerce"))
    _tde = float(pd.to_numeric(target_row.get("dec_deg", target_row.get("dec")), errors="coerce"))
    _tbpr = float(pd.to_numeric(target_row.get("bp_rp"), errors="coerce"))
    _plate = float(getattr(_cfg, "plate_scale_arcsec_per_px", 0.0) or 0.0)
    _c_col_ov = getattr(_cfg, "comp_weight_c_col_mag_per_bprp", None)
    _c_dist_ov = getattr(_cfg, "comp_weight_c_dist_mag_per_deg", None)
    try:
        _c_col_ov_f = float(_c_col_ov) if _c_col_ov is not None else None
    except (TypeError, ValueError):
        _c_col_ov_f = None
    try:
        _c_dist_ov_f = float(_c_dist_ov) if _c_dist_ov is not None else None
    except (TypeError, ValueError):
        _c_dist_ov_f = None
    _k2 = None
    try:
        from k2_extinction import resolve_k2_bprp_value  # noqa: PLC0415

        _k2, _ = resolve_k2_bprp_value(_cfg, str(getattr(_cfg, "active_obs_group", "") or ""))
    except Exception:  # noqa: BLE001
        _k2 = None
    _am_span = float(getattr(_cfg, "comp_weight_airmass_span", float("nan")) or float("nan"))
    if not math.isfinite(_am_span):
        _am_span = 0.0
    _r_list: list[float] = []
    _sc_list: list[float] = []
    for _, r in target_comps.iterrows():
        try:
            _rr = float(pd.to_numeric(r.get("ra_deg", r.get("ra")), errors="coerce"))
            _dd = float(pd.to_numeric(r.get("dec_deg", r.get("dec")), errors="coerce"))
            _rms = float(pd.to_numeric(r.get("comp_rms"), errors="coerce"))
        except Exception:  # noqa: BLE001
            continue
        if math.isfinite(_rr) and math.isfinite(_dd) and math.isfinite(_tra) and math.isfinite(_tde):
            dra = math.radians(_rr - _tra) * math.cos(math.radians(0.5 * (_dd + _tde)))
            dde = math.radians(_dd - _tde)
            _r_list.append(float(math.degrees(math.hypot(dra, dde))))
            if math.isfinite(_rms):
                _sc_list.append(_rms)
        elif math.isfinite(_tx) and math.isfinite(_ty) and _plate > 0:
            try:
                _cx = float(pd.to_numeric(r.get("x"), errors="coerce"))
                _cy = float(pd.to_numeric(r.get("y"), errors="coerce"))
            except Exception:  # noqa: BLE001
                continue
            if math.isfinite(_cx) and math.isfinite(_cy):
                _r_list.append(float(math.hypot(_cx - _tx, _cy - _ty) * _plate / 3600.0))
                if math.isfinite(_rms):
                    _sc_list.append(_rms)
    _optics = str(getattr(_cfg, "comp_weight_optics_kind", "") or "").strip()
    if not _optics:
        try:
            from comp_weights import infer_optics_kind_from_header_or_name  # noqa: PLC0415

            _optics = infer_optics_kind_from_header_or_name(
                telescop=str(getattr(_cfg, "telescope_name", "") or ""),
                telescope_name=str(getattr(_cfg, "telescope_name", "") or ""),
            )
        except Exception:  # noqa: BLE001
            _optics = "unknown"
    if not math.isfinite(_am_span) or _am_span <= 0:
        # Best-effort airmass span from frame table if present on comps flux cache later; keep 0.
        _am_span = 0.0
    _coeffs = resolve_comp_weight_coeffs(
        k2_bprp=_k2,
        airmass_span=_am_span,
        optics_kind=_optics,
        r_deg=_r_list,
        residual_scatter_mag=_sc_list,
        c_col_override=_c_col_ov_f,
        c_dist_override=_c_dist_ov_f,
    )
    comp_weight_map: dict[str, float] = {}
    for _, r in target_comps.iterrows():
        cid0 = _normalize_gaia_id(r["catalog_id"])
        rms0 = float(comp_rms_map.get(cid0, float("nan")))
        try:
            bpr0 = float(pd.to_numeric(r.get("bp_rp"), errors="coerce"))
        except Exception:  # noqa: BLE001
            bpr0 = float("nan")
        db = abs(bpr0 - _tbpr) if math.isfinite(bpr0) and math.isfinite(_tbpr) else 0.0
        rdeg = 0.0
        try:
            _rr = float(pd.to_numeric(r.get("ra_deg", r.get("ra")), errors="coerce"))
            _dd = float(pd.to_numeric(r.get("dec_deg", r.get("dec")), errors="coerce"))
            if math.isfinite(_rr) and math.isfinite(_dd) and math.isfinite(_tra) and math.isfinite(_tde):
                dra = math.radians(_rr - _tra) * math.cos(math.radians(0.5 * (_dd + _tde)))
                dde = math.radians(_dd - _tde)
                rdeg = float(math.degrees(math.hypot(dra, dde)))
            elif math.isfinite(_tx) and math.isfinite(_ty) and _plate > 0:
                _cx = float(pd.to_numeric(r.get("x"), errors="coerce"))
                _cy = float(pd.to_numeric(r.get("y"), errors="coerce"))
                if math.isfinite(_cx) and math.isfinite(_cy):
                    rdeg = float(math.hypot(_cx - _tx, _cy - _ty) * _plate / 3600.0)
        except Exception:  # noqa: BLE001
            rdeg = 0.0
        se = sigma_eff_mag(
            sigma_rms_mag=rms0,
            delta_bprp=db,
            r_deg=rdeg,
            c_col_mag_per_bprp=_coeffs.c_col_mag_per_bprp,
            c_dist_mag_per_deg=_coeffs.c_dist_mag_per_deg,
        )
        comp_weight_map[cid0] = weight_from_sigma_eff(se)

    _chk_cid_pref: str | None = None
    try:
        from pinned_ensembles import (  # noqa: PLC0415
            get_pinned_check_for_target,
            is_pinned_target,
            validate_pinned_check_member,
        )

        if is_pinned_target(str(target_cid)):
            _pin_chk = get_pinned_check_for_target(str(target_cid))
            if _pin_chk is not None:
                _chk_ms = state.masterstars_df.loc[
                    state.masterstars_df["catalog_id"].astype(str).str.strip().eq(_pin_chk.check_catalog_id)
                ]
                if not _chk_ms.empty:
                    _chk_row_ms = _chk_ms.iloc[0]
                    _chk_dist = float("nan")
                    if "_dist_deg" in target_comps.columns:
                        _sub = target_comps.loc[
                            target_comps["catalog_id"].astype(str).str.strip().eq(_pin_chk.check_catalog_id)
                        ]
                        if not _sub.empty and "_dist_deg" in _sub.columns:
                            _chk_dist = float(pd.to_numeric(_sub["_dist_deg"].iloc[0], errors="coerce")) * 3600.0
                    if not math.isfinite(_chk_dist):
                        try:
                            _cra = float(
                                pd.to_numeric(
                                    _chk_row_ms.get("ra_deg", _chk_row_ms.get("ra")),
                                    errors="coerce",
                                )
                            )
                            _cde = float(
                                pd.to_numeric(
                                    _chk_row_ms.get("dec_deg", _chk_row_ms.get("dec")),
                                    errors="coerce",
                                )
                            )
                            if (
                                math.isfinite(_cra)
                                and math.isfinite(_cde)
                                and math.isfinite(_tra)
                                and math.isfinite(_tde)
                            ):
                                _chk_dist = _angular_distance_deg(_tra, _tde, _cra, _cde) * 3600.0
                            elif math.isfinite(_tx) and math.isfinite(_ty) and _plate > 0:
                                _cx = float(pd.to_numeric(_chk_row_ms.get("x"), errors="coerce"))
                                _cy = float(pd.to_numeric(_chk_row_ms.get("y"), errors="coerce"))
                                if math.isfinite(_cx) and math.isfinite(_cy):
                                    _chk_dist = float(
                                        math.hypot(_cx - _tx, _cy - _ty) * _plate / 3600.0
                                    )
                        except Exception:  # noqa: BLE001
                            _chk_dist = float("nan")
                    _chk_rms = float(
                        pd.to_numeric(
                            comp_rms_map.get(_pin_chk.check_catalog_id, float("nan")),
                            errors="coerce",
                        )
                    )
                    _ok_chk, _reason_chk = validate_pinned_check_member(
                        _chk_row_ms,
                        target_cid=str(target_cid),
                        dist_arcsec=_chk_dist,
                        comp_rms=_chk_rms,
                        min_dist_arcsec=float(_cfg.phase01_comparison_min_dist_arcsec),
                        max_comp_rms=float(_cfg.phase01_comparison_max_comp_rms),
                    )
                    if _ok_chk:
                        _chk_cid_pref = _pin_chk.check_catalog_id
                        log_event(
                            f"[PIN] check star {_pin_chk.check_catalog_id} "
                            f"kname={_pin_chk.check_kname!r} target={target_cid}"
                        )
                    else:
                        logging.warning(
                            "[PIN-DROP] check star %s for target %s: %s",
                            _pin_chk.check_catalog_id,
                            target_cid,
                            _reason_chk,
                        )
    except Exception as _pin_chk_exc:  # noqa: BLE001
        logging.debug("[PIN] check star pin skipped for %s: %s", target_cid, _pin_chk_exc)

    if _chk_cid_pref is None:
        try:
            from check_star_kmag import (  # noqa: PLC0415
                field_check_star_candidate_pool,
                select_check_star,
            )

            _chk_pool_pref = field_check_star_candidate_pool(
                state.comp_df,
                target_comps=target_comps,
            )
            if not _chk_pool_pref.empty:
                _chk_row_pref = select_check_star(
                    _chk_pool_pref,
                    ensemble_ids=set(comp_ids),
                    n_comp_min=max(1, min(3, len(_chk_pool_pref))),
                    cfg=_cfg,
                )
                if _chk_row_pref is not None:
                    _chk_cid_pref = _normalize_gaia_id(_chk_row_pref.get("catalog_id", ""))
        except (ImportError, KeyError, TypeError, ValueError, AttributeError) as _ck_pref_exc:
            logging.debug("[CHECK-KMAG] preselect skipped for %s: %s", target_cid, _ck_pref_exc)

    if _chk_cid_pref:
        if (
            _chk_cid_pref not in comp_ids
            and _chk_cid_pref != target_cid
        ):
            all_ids.append(_chk_cid_pref)
            _chk_row_pref = None
            try:
                from check_star_kmag import field_check_star_candidate_pool  # noqa: PLC0415

                _chk_pool_pref = field_check_star_candidate_pool(
                    state.comp_df,
                    target_comps=target_comps,
                )
                if not _chk_pool_pref.empty:
                    _m = _chk_pool_pref["catalog_id"].astype(str).str.strip().eq(_chk_cid_pref)
                    if bool(_m.any()):
                        _chk_row_pref = _chk_pool_pref.loc[_m].iloc[0]
            except Exception:  # noqa: BLE001
                _chk_row_pref = None
            if _chk_row_pref is None:
                _chk_ms = state.masterstars_df.loc[
                    state.masterstars_df["catalog_id"].astype(str).str.strip().eq(_chk_cid_pref)
                ]
                _chk_row_pref = _chk_ms.iloc[0] if not _chk_ms.empty else None
            if _chk_row_pref is not None:
                for _mk in ("mag", "phot_g_mean_mag"):
                    try:
                        _cm = float(pd.to_numeric(_chk_row_pref.get(_mk), errors="coerce"))
                    except Exception:  # noqa: BLE001
                        _cm = float("nan")
                    if math.isfinite(_cm):
                        comp_catalog_mag[_chk_cid_pref] = _cm
                        break
                try:
                    _cx = float(pd.to_numeric(_chk_row_pref.get("x"), errors="coerce"))
                    _cy = float(pd.to_numeric(_chk_row_pref.get("y"), errors="coerce"))
                except Exception:  # noqa: BLE001
                    _cx, _cy = float("nan"), float("nan")
                if math.isfinite(_cx) and math.isfinite(_cy):
                    _star_xy[_chk_cid_pref] = (_cx, _cy)

    # Krok 2: Fotometria per snimka (PERF-8: slice shared flux matrix when built)
    frame_results: list[pd.DataFrame] = []
    if not _flux_matrix.empty:
        _id_set = set(all_ids)
        _target_slice = _flux_matrix[_flux_matrix["catalog_id"].isin(_id_set)]
        for csv_path in csv_files:
            _sf = csv_path.name
            _df_sub = _target_slice[_target_slice["source_file"] == _sf]
            if _df_sub.empty:
                continue
            df_frame = _df_sub.copy()
            _ft = frame_time_lookup.get(csv_path.stem)
            _cached_df = _phase2a_csv_cache.get(str(csv_path))
            if (chip_fw is None or chip_fh is None) and ("x" in df_frame.columns and "y" in df_frame.columns):
                try:
                    _xm = float(pd.to_numeric(df_frame["x"], errors="coerce").max())
                    _ym = float(pd.to_numeric(df_frame["y"], errors="coerce").max())
                except Exception:  # noqa: BLE001
                    _xm, _ym = float("nan"), float("nan")
                if chip_fw is None and math.isfinite(_xm) and _xm > 0:
                    chip_fw = int(math.ceil(_xm)) + 2
                if chip_fh is None and math.isfinite(_ym) and _ym > 0:
                    chip_fh = int(math.ceil(_ym)) + 2
            if chip_fw is not None and chip_fh is not None and int(chip_fw) > 0 and int(chip_fh) > 0:
                tmask = df_frame["catalog_id"].astype(str).str.strip().eq(target_cid)
                if bool(tmask.any()):
                    tr = df_frame.loc[tmask].iloc[0]
                    try:
                        x_t = float(pd.to_numeric(tr.get("x"), errors="coerce"))
                        y_t = float(pd.to_numeric(tr.get("y"), errors="coerce"))
                    except Exception:  # noqa: BLE001
                        x_t, y_t = float("nan"), float("nan")
                    try:
                        r_out_t = float(pd.to_numeric(tr.get("sky_annulus_r_out_px", 30.0), errors="coerce"))
                    except Exception:  # noqa: BLE001
                        r_out_t = 30.0
                    if not (math.isfinite(r_out_t) and r_out_t > 0):
                        r_out_t = 30.0
                    if math.isfinite(x_t) and math.isfinite(y_t):
                        edge_ok = (
                            (x_t - r_out_t >= 0)
                            and (x_t + r_out_t <= float(chip_fw))
                            and (y_t - r_out_t >= 0)
                            and (y_t + r_out_t <= float(chip_fh))
                        )
                        if not edge_ok:
                            df_frame = df_frame.copy()
                            df_frame.loc[tmask, "mag_inst"] = float("nan")
                            df_frame.loc[tmask, "flag"] = "edge_fail"
                            if "edge_fail" in df_frame.columns:
                                df_frame.loc[tmask, "edge_fail"] = True
                            logging.info(
                                "[TARGET EDGE] %s: frame %s vyradeny - annulus mimo cip (x=%.0f, y=%.0f, r_out=%.1fpx)",
                                str(target_name),
                                str(csv_path.name),
                                float(x_t),
                                float(y_t),
                                float(r_out_t),
                            )
            frame_results.append(df_frame)
    else:
        for csv_path in csv_files:
            _ft = frame_time_lookup.get(csv_path.stem)
            _key_csv = str(csv_path)
            _cached_df = _phase2a_csv_cache.get(_key_csv)
            _lookup_row = _phase2a_lookup_cache.get(_key_csv)

            df_frame = read_flux_from_csv(
                csv_path,
                all_ids,
                apertures_px,
                sat_limit_adu=sat_limit_resolved,
                star_xy=_star_xy,
                xy_tol_px=18.0,
                frame_times=_ft,
                csv_df=_cached_df,
                lookup=_lookup_row,
                gain=float(_gain_phot),
                read_noise=float(_rn_phot),
                use_apcorr_flux=bool(state.use_apcorr_flux),
                variable_target_catalog_ids=state.variable_target_catalog_ids,
                err_background_mode=ERR_BKG_MODE_EMPIRICAL,
            )
            if not df_frame.empty:
                if (chip_fw is None or chip_fh is None) and ("x" in df_frame.columns and "y" in df_frame.columns):
                    try:
                        _xm = float(pd.to_numeric(df_frame["x"], errors="coerce").max())
                        _ym = float(pd.to_numeric(df_frame["y"], errors="coerce").max())
                    except Exception:  # noqa: BLE001
                        _xm, _ym = float("nan"), float("nan")
                    if chip_fw is None and math.isfinite(_xm) and _xm > 0:
                        chip_fw = int(math.ceil(_xm)) + 2
                    if chip_fh is None and math.isfinite(_ym) and _ym > 0:
                        chip_fh = int(math.ceil(_ym)) + 2

                if chip_fw is not None and chip_fh is not None and int(chip_fw) > 0 and int(chip_fh) > 0:
                    tmask = df_frame["catalog_id"].astype(str).str.strip().eq(target_cid)
                    if bool(tmask.any()):
                        tr = df_frame.loc[tmask].iloc[0]
                        try:
                            x_t = float(pd.to_numeric(tr.get("x"), errors="coerce"))
                            y_t = float(pd.to_numeric(tr.get("y"), errors="coerce"))
                        except Exception:  # noqa: BLE001
                            x_t, y_t = float("nan"), float("nan")
                        try:
                            r_out_t = float(pd.to_numeric(tr.get("sky_annulus_r_out_px", 30.0), errors="coerce"))
                        except Exception:  # noqa: BLE001
                            r_out_t = 30.0
                        if not (math.isfinite(r_out_t) and r_out_t > 0):
                            r_out_t = 30.0
                        if math.isfinite(x_t) and math.isfinite(y_t):
                            edge_ok = (
                                (x_t - r_out_t >= 0)
                                and (x_t + r_out_t <= float(chip_fw))
                                and (y_t - r_out_t >= 0)
                                and (y_t + r_out_t <= float(chip_fh))
                            )
                            if not edge_ok:
                                df_frame = df_frame.copy()
                                df_frame.loc[tmask, "mag_inst"] = float("nan")
                                df_frame.loc[tmask, "flag"] = "edge_fail"
                                if "edge_fail" in df_frame.columns:
                                    df_frame.loc[tmask, "edge_fail"] = True
                                logging.info(
                                    "[TARGET EDGE] %s: frame %s vyradeny - annulus mimo cip (x=%.0f, y=%.0f, r_out=%.1fpx)",
                                    str(target_name),
                                    str(csv_path.name),
                                    float(x_t),
                                    float(y_t),
                                    float(r_out_t),
                                )
                frame_results.append(df_frame)

    if not frame_results:
        return summary_rows, n_lc

    ac_result: dict[str, Any] = {
        "ok": False,
        "delta_m_corr": None,
        "scatter_mag": None,
        "n_ref_stars": 0,
        "ref_star_ids": [],
        "reason": "disabled",
    }
    if bool(_cfg.aperture_correction_enabled):
        try:
            ac_result = compute_aperture_correction(
                comp_df=target_comps,
                frame_results=frame_results,
                min_ref_stars=int(_cfg.aperture_correction_min_ref_stars),
                max_contamination=float(_cfg.aperture_correction_max_contamination),
                max_scatter_mag=float(_cfg.aperture_correction_max_scatter_mag),
            )
            if bool(ac_result.get("ok")):
                log_event(
                    f"[AC] DeltaM_corr={float(ac_result['delta_m_corr']):.4f} "
                    f"scatter={float(ac_result['scatter_mag']):.4f} "
                    f"n_ref={int(ac_result['n_ref_stars'])}"
                )
            else:
                log_event(f"[AC] skipped: {ac_result.get('reason', '')}")
        except Exception as _ac_exc:  # noqa: BLE001
            log_event(f"[AC] skipped: exception {_ac_exc!s}")
            ac_result = {
                "ok": False,
                "delta_m_corr": None,
                "scatter_mag": None,
                "n_ref_stars": 0,
                "ref_star_ids": [],
                "reason": "exception",
            }
    _ = ac_result  # Krokom 3: aplikacia na mag_calib / CSV

    all_frames = pd.concat(frame_results, ignore_index=True)

    # Zostav casove rady per hviezda
    target_lc = _get_lc(target_cid, all_frames)
    comp_lc = {cid: _get_lc(cid, all_frames) for cid in comp_ids}

    # Flux sources for method-keyed LC outputs (aperture always primary/default).
    _psf_enabled = bool(_cfg.psf_photometry_enabled)
    _adaptive = bool(getattr(_cfg, "psf_adaptive_enabled", False))
    _have_psf_cols = "psf_flux" in all_frames.columns and "psf_fit_ok" in all_frames.columns
    if _have_psf_cols and (_adaptive or _psf_enabled):
        _blend_map = _load_adaptive_blend_map(masterstar_fits_path)
        all_frames["lc_flux_method"] = compute_lc_flux_method(
            all_frames,
            _blend_map,
            resolve_fwhm=float(getattr(_cfg, "psf_adaptive_resolve_fwhm", 2.0)),
            snr_lo=float(getattr(_cfg, "psf_adaptive_snr_lo", 15.0)),
        )
    # Primary published LC is always aperture (target_lc / comp_lc from _get_lc above).
    _lc_export_method = "aperture"

    # ALG-3: Temporal binning of comp ensemble (MNRAS 2023)
    comp_lc = temporal_bin_comp_lc(
        comp_lc=comp_lc,
        comp_quality={},
        all_frames=all_frames,
        window=int(_cfg.temporal_bin_window),
        enabled=bool(_cfg.temporal_binning_enabled),
    )

    # COMP-ASSIGN-01 D4/D5: membership is fixed from Phase 1 (3-8). Stability is a
    # post-photometry verdict only - do not let it re-select before ensemble.
    comp_quality = {cid: {"quality": "good"} for cid in comp_ids}

    # ALG-5: PyTICS iterative comp star intercalibration (RASTI 2026)
    comp_rms_map = pytics_iterative_weights(
        comp_lc=comp_lc,
        comp_quality=comp_quality,
        comp_rms_map=comp_rms_map,
        n_iter=int(_cfg.pytics_n_iter),
        enabled=bool(_cfg.pytics_enabled),
    )

    # Krok 4: Ensemble normalizacia (consumes the delivered set as given)
    mag_calib, delta_mag, ensemble_scatter = ensemble_normalize(
        target_lc,
        comp_lc,
        comp_catalog_mag,
        comp_quality,
        comp_rms_map=comp_rms_map,
        comp_tier_map=comp_tier_map,
        tier_weights=tier_weights,
        comp_weight_map=comp_weight_map,
        n_comp_min=max(1, int(getattr(_cfg, "phase01_comparison_n_comp_min", 3))),
        n_comp_max=int(_cfg.phase01_comparison_n_comp_max),
    )
    _ensemble_scatter_by_file = _ensemble_scatter_by_source_file(
        all_frames, target_cid, ensemble_scatter
    )

    _dilution_result: dict[str, Any] = {
        "dilution_factor": 1.0,
        "dilution_delta_mag": 0.0,
        "n_neighbors": 0,
        "neighbor_flux_sum": 0.0,
        "aperture_arcsec": float("nan"),
        "search_radius_arcsec": float("nan"),
    }
    if bool(_cfg.gs11_dilution_enabled) and state.gaia_db_path:
        from dilution import apply_target_dilution_to_mag_calib, compute_dilution_factor  # noqa: PLC0415

        try:
            _target_ra = float(
                pd.to_numeric(
                    target_row.get("ra_deg", target_row.get("ra", float("nan"))),
                    errors="coerce",
                )
            )
            _target_dec = float(
                pd.to_numeric(
                    target_row.get("dec_deg", target_row.get("dec", float("nan"))),
                    errors="coerce",
                )
            )
        except (TypeError, ValueError):
            _target_ra = _target_dec = float("nan")
        _target_g_mag = float("nan")
        for _gk in ("mag", "phot_g_mean_mag", "catalog_mag"):
            try:
                _gv = float(pd.to_numeric(target_row.get(_gk, float("nan")), errors="coerce"))
            except (TypeError, ValueError):
                _gv = float("nan")
            if math.isfinite(_gv):
                _target_g_mag = _gv
                break
        _ap_cfg = float(_cfg.gs11_dilution_aperture_arcsec)
        _dilution_skipped_ap = False
        if math.isfinite(_ap_cfg) and _ap_cfg > 0:
            _ap_arcsec = _ap_cfg
        else:
            _ap_px, _ap_src = _resolve_photometric_aperture_px_for_gs11(
                target_cid,
                apertures_px,
                _target_g_mag,
                state.snr_ap_table,
                aperture_fwhm_factor=float(_apt_fw),
                fwhm_px=float(fwhm_px),
            )
            if _ap_px is None:
                logging.warning(
                    "[GS11] target %s: photometric aperture unavailable - dilution skipped",
                    target_cid or "?",
                )
                log_event(
                    f"[GS11] target {target_cid or '?'}: photometric aperture unavailable - dilution skipped"
                )
                _dilution_skipped_ap = True
                _ap_arcsec = float("nan")
            else:
                _ap_arcsec = float(_ap_px) * float(state.plate_scale_arcsec)
        _cid_int = None
        try:
            from dilution import _normalize_exclude_source_id  # noqa: PLC0415

            _cid_int = _normalize_exclude_source_id(target_cid)
        except Exception:  # noqa: BLE001
            _cid_int = None
        if _dilution_skipped_ap:
            _dilution_result = {
                "dilution_factor": 1.0,
                "dilution_delta_mag": 0.0,
                "n_neighbors": 0,
                "neighbor_flux_sum": 0.0,
                "aperture_arcsec": float("nan"),
                "search_radius_arcsec": float("nan"),
                "dilution_skipped": True,
                "dilution_skip_reason": "photometric_aperture_unavailable",
            }
        else:
            _dilution_result = compute_dilution_factor(
                _target_ra,
                _target_dec,
                _target_g_mag,
                _ap_arcsec,
                str(state.gaia_db_path),
                catalog_id=_cid_int,
                mag_limit_delta=float(_cfg.gs11_dilution_mag_limit_delta),
            )
        _mag_pre_gs11 = float("nan")
        _finite_pre = mag_calib[np.isfinite(mag_calib)]
        if len(_finite_pre) > 0:
            _mag_pre_gs11 = float(np.median(_finite_pre))
        mag_calib, _dilution_result = apply_target_dilution_to_mag_calib(
            mag_calib,
            _dilution_result,
            _cfg,
            target_cid=str(target_cid),
        )
        _mag_post_gs11 = float("nan")
        _finite_post = mag_calib[np.isfinite(mag_calib)]
        if len(_finite_post) > 0:
            _mag_post_gs11 = float(np.median(_finite_post))
    else:
        _mag_pre_gs11 = float("nan")
        _mag_post_gs11 = float("nan")

    # -- Aperture correction (AC) --
    ac_ok = bool(ac_result.get("ok", False)) if isinstance(ac_result, dict) else False
    delta_m_corr = ac_result.get("delta_m_corr") if isinstance(ac_result, dict) else None
    if ac_ok and delta_m_corr is not None and np.isfinite(float(delta_m_corr)):
        mag_calib_ac = mag_calib + float(delta_m_corr)
    else:
        mag_calib_ac = np.full_like(mag_calib, float("nan"))

    # Sanity log znamienka: pri delta_m_corr < 0 ma byt mag_calib_ac < mag_calib.
    if (not ac_sign_logged[0]) and ac_ok and delta_m_corr is not None and np.isfinite(float(delta_m_corr)):
        if len(mag_calib) > 0 and math.isfinite(float(mag_calib[0])) and math.isfinite(float(mag_calib_ac[0])):
            log_event(
                f"[AC SIGN] mag_calib0={float(mag_calib[0]):.4f} "
                f"delta_m_corr={float(delta_m_corr):.4f} "
                f"mag_calib_ac0={float(mag_calib_ac[0]):.4f}"
            )
            ac_sign_logged[0] = True

    # -- Color term (BP-RP) - global comp-pool fit; toggle controls correction only --
    target_bp_rp = float(target_bp_rp_by_cid.get(target_cid, float("nan")))
    comp_bp_rp: dict[str, float] = {}
    if "bp_rp" in target_comps.columns:
        for _, rr in target_comps.iterrows():
            cidc = _normalize_gaia_id(rr.get("catalog_id", ""))
            if not cidc:
                continue
            v = pd.to_numeric(rr.get("bp_rp"), errors="coerce")
            try:
                fv = float(v)
            except Exception:  # noqa: BLE001
                fv = float("nan")
            if math.isfinite(fv):
                comp_bp_rp[cidc] = float(fv)

    from k2_extinction import K2Source, apply_k2_per_frame, bp_rp_comp_median  # noqa: PLC0415

    k2_value_lc = float("nan")
    k2_colour_ref = float("nan")
    k2_source_rows = [K2Source.NONE.value] * len(mag_calib)
    _k2_val = float(getattr(state, "k2_bprp", float("nan")))
    _k2_src = str(getattr(state, "k2_source", K2Source.NONE.value))
    if _k2_src in (
        K2Source.LITERATURE_DEFAULT.value,
        K2Source.NIGHT_FIT.value,
    ) and math.isfinite(_k2_val):
        _tf_k2 = all_frames[all_frames["catalog_id"] == target_cid]
        if "airmass" in _tf_k2.columns:
            _airmass_k2 = _tf_k2["airmass"].to_numpy(dtype=float)
        else:
            _airmass_k2 = np.full(len(mag_calib), float("nan"), dtype=float)
        _bp_med_k2 = bp_rp_comp_median(comp_bp_rp, comp_quality)
        _k2_src_enum = (
            K2Source.NIGHT_FIT
            if _k2_src == K2Source.NIGHT_FIT.value
            else K2Source.LITERATURE_DEFAULT
        )
        mag_calib, _k2_delta, k2_source_rows = apply_k2_per_frame(
            mag_calib,
            _airmass_k2,
            object_bp_rp=float(target_bp_rp),
            bp_rp_comp_med=_bp_med_k2,
            k2_value=_k2_val,
            k2_source=_k2_src_enum,
        )
        k2_value_lc = _k2_val
        k2_colour_ref = _bp_med_k2

    c1 = 0.0
    c1_stderr = float("nan")
    ct_mode = ""
    ct_n_comp = 0
    mag_calib_ct = mag_calib.copy()
    ct_corr = 0.0
    bp_rp_comp_med = float("nan")
    ct_ok = False
    _group_ct = state.group_color_term
    if state.apply_color_term and _group_ct is not None and _group_ct.apply_gate:
        c1 = float(_group_ct.c1)
        c1_stderr = float(_group_ct.c1_stderr)
        ct_mode = str(getattr(_group_ct, "mode", "fit") or "fit")
        ct_n_comp = int(_group_ct.n_comp)
        _ref_bp, _ref_q = ct_ensemble_reference_maps(comp_bp_rp, comp_quality)
        _ct_in_range = _check_color_term_extrapolation(
            target_bp_rp=float(target_bp_rp),
            comp_bp_rp_values=[float(v) for v in _ref_bp.values()],
            target_name=str(target_name),
            extrapolation_tol=float(_cfg.phase01_ct_extrapolation_tol),
        )
        try:
            from pinned_ensembles import baseline_lc_ct_ok_for_target, is_pinned_target  # noqa: PLC0415

            if is_pinned_target(str(target_cid)):
                _pin_ct_ok = baseline_lc_ct_ok_for_target(str(target_cid))
                if _pin_ct_ok is False:
                    _ct_in_range = False
                elif _pin_ct_ok is True and str(ct_mode) == "clear_level":
                    _ct_in_range = True
        except Exception as _pin_ct_rng_exc:  # noqa: BLE001
            LOGGER.debug("[PIN] CT extrapolation pin gate skip: %s", _pin_ct_rng_exc)
        if _ct_in_range:
            mag_calib_ct, ct_corr, bp_rp_comp_med = apply_color_term(
                mag_calib,
                target_bp_rp,
                _ref_bp,
                _ref_q,
                c1,
                comp_weights=comp_weight_map if ct_mode == "clear_level" else None,
            )
            ct_ok = (
                bool(math.isfinite(float(target_bp_rp)))
                and float(c1) != 0.0
                and math.isfinite(float(bp_rp_comp_med))
            )
        else:
            logging.info(
                "[COLOR TERM] extrapolation -> CT skipped (target kept, uncorrected)"
            )
            mag_calib_ct = mag_calib.copy()
            ct_corr = 0.0
            bp_rp_comp_med = float("nan")
            ct_ok = False

    if _ct_prototype_enabled():
        _proto_c1 = 0.0
        _proto_c1_stderr = float("nan")
        _proto_n_comp = 0
        if comp_bp_rp:
            _proto_c1, _proto_c1_stderr, _proto_n_comp = fit_color_term_c1(
                comp_lc,
                comp_catalog_mag,
                comp_bp_rp,
                comp_quality,
                min_comp=5,
                sigma_clip_sigma=3.0,
            )
        _proto_corr = 0.0
        _proto_comp_med = float("nan")
        if comp_bp_rp and float(_proto_c1) != 0.0:
            _, _proto_corr, _proto_comp_med = apply_color_term(
                mag_calib,
                float(target_bp_rp),
                comp_bp_rp,
                comp_quality,
                float(_proto_c1),
            )
        _proto_scatter, _proto_scatter_resid = (
            _color_term_cat_inst_scatter_pair(
                comp_lc,
                comp_catalog_mag,
                comp_bp_rp,
                comp_quality,
                float(_proto_c1),
                min_comp=5,
                sigma_clip_sigma=3.0,
            )
            if comp_bp_rp
            else (float("nan"), float("nan"))
        )
        _proto_stderr_ratio = float("nan")
        if float(_proto_c1) != 0.0 and math.isfinite(float(_proto_c1_stderr)):
            _proto_stderr_ratio = abs(float(_proto_c1_stderr) / float(_proto_c1))
        _proto_gate = (
            int(_proto_n_comp) >= int(_cfg.phase01_ct_min_comp)
            and float(_proto_c1) != 0.0
            and math.isfinite(_proto_stderr_ratio)
            and float(_proto_stderr_ratio) <= 0.5
        )
        _append_ct_prototype_row(
            _draft_dir_from_phase2a_paths(output_dir, Path(masterstar_fits_path)),
            {
                "catalog_id": target_cid,
                "vsx_name": target_name,
                "obs_group": str(obs_group),
                "n_comp_used": int(_proto_n_comp),
                "c1": float(_proto_c1),
                "c1_stderr": float(_proto_c1_stderr),
                "stderr_ratio": _proto_stderr_ratio,
                "target_bp_rp": float(target_bp_rp),
                "comp_med_bp_rp": float(_proto_comp_med),
                "ct_corr": float(_proto_corr),
                "cat_inst_scatter": _proto_scatter,
                "cat_inst_scatter_resid": _proto_scatter_resid,
                "gate_would_pass": bool(_proto_gate),
            },
        )

    # Casove hodnoty targetu - sort by source_file so ensemble_scatter index aligns
    # with ``_get_lc`` / ``_ensemble_scatter_by_source_file`` (LABBE-DET / SEM determinism).
    target_frames = all_frames[all_frames["catalog_id"] == target_cid]
    if not target_frames.empty and "source_file" in target_frames.columns:
        target_frames = target_frames.sort_values(["source_file"], kind="mergesort")
    _measured_ap_target = _measured_aperture_from_proc_cache(target_cid, state._phase2a_csv_cache)
    if math.isfinite(_measured_ap_target) and _measured_ap_target > 0 and not target_frames.empty:
        target_frames = target_frames.copy()
        target_frames["aperture_r_px"] = float(_measured_ap_target)
    bjd = target_frames["bjd"].to_numpy(dtype=float)
    hjd = target_frames["hjd"].to_numpy(dtype=float)
    jd = target_frames["jd"].to_numpy(dtype=float)

    # BJD-PERTARGET: recompute with target's own RA/Dec (not field-center LTT)
    _target_ra = float(pd.to_numeric(target_row.get("ra_deg", target_row.get("ra", float("nan"))), errors="coerce"))
    _target_dec = float(
        pd.to_numeric(target_row.get("dec_deg", target_row.get("dec", float("nan"))), errors="coerce")
    )
    bjd, hjd, time_base = _recompute_bjd_hjd_with_status(
        jd,
        _target_ra,
        _target_dec,
        _cfg,
        site=(state.site_lat, state.site_lon, state.site_alt) if state.site_ok else None,
    )

    err = target_frames["err"].to_numpy(dtype=float)
    err, err_method_rows = _route_lc_per_frame_err(target_frames, err)
    err_photon_arr = np.asarray(err, dtype=np.float64).copy()
    if "airmass" in target_frames.columns:
        airmass_arr = target_frames["airmass"].to_numpy(dtype=float)
    else:
        airmass_arr = np.full(len(target_frames), float("nan"), dtype=float)
    # Per-point uncertainty = photon/SNR base error (term-1) (+) ensemble zeropoint uncertainty
    # (term-3, ``ensemble_scatter``). Joined by EXACT ``source_file`` (G2-F004), not positional index.
    _src_for_err = target_frames["source_file"].astype(str).tolist()
    from sigma_budget import resolve_rig_scintillation_params  # noqa: PLC0415
    from sigma_floor_core import resolve_sigma_sys_mag, scintillation_mag_per_epoch  # noqa: PLC0415

    _sigma_sys_mag = resolve_sigma_sys_mag(
        state.equipment_id,
        _cfg,
        rig_label=str(state.obs_group or ""),
    )
    _draft_id_lc: int | None = None
    try:
        from platesolve_ui_paths import parse_draft_id_from_text  # noqa: PLC0415

        _draft_id_lc = parse_draft_id_from_text(str(output_dir))
    except Exception:  # noqa: BLE001
        _draft_id_lc = None
    _rig_scint = resolve_rig_scintillation_params(
        draft_id=_draft_id_lc,
        setup=str(state.obs_group or ""),
        cfg=_cfg,
        pipeline_meta=(
            {"observer_location": {"alt_m": float(state.site_alt)}}
            if state.site_ok and state.site_alt is not None
            else None
        ),
    )
    _scint_mag_arr = np.array(
        [
            scintillation_mag_per_epoch(
                telescope_diameter_m=_rig_scint.telescope_diameter_m,
                airmass=float(am),
                exposure_s=_rig_scint.exposure_s,
                altitude_m=_rig_scint.altitude_m,
                c_y=_rig_scint.c_y,
            )
            if math.isfinite(float(am)) and float(am) >= 1.0
            else 0.0
            for am in airmass_arr
        ],
        dtype=np.float64,
    )
    err, err_scatter_unmatched_arr = _combine_err_with_ensemble_scatter_keyed(
        err,
        _src_for_err,
        _ensemble_scatter_by_file,
        sigma_sys_mag=_sigma_sys_mag,
        sigma_scint_mag=_scint_mag_arr,
        target_name=str(target_name),
    )
    # WIDE-ERR-03: Pont/Gillon calibration layer on combined model err.
    # CONSOLIDATE-01D: always ERR-CALIB calibrated (export_err_mode=model branch deleted).
    try:
        from err_calibration import (  # noqa: PLC0415
            ERR_CALIB_SIDECAR,
            apply_calibration_rel,
            bins_from_sidecar,
            load_sidecar,
            smooth_from_sidecar,
        )

        _cal_path = Path(output_dir) / ERR_CALIB_SIDECAR if output_dir is not None else None
        _cal = load_sidecar(_cal_path) if _cal_path is not None else None
        if _cal:
            _smooth = smooth_from_sidecar(_cal)
            _bins = bins_from_sidecar(_cal) if not _smooth else []
            _calib_obj = _smooth if _smooth is not None else _bins
            _g_tgt = float("nan")
            try:
                _g_tgt = float(
                    pd.to_numeric(
                        target_row.get("phot_g_mean_mag", target_row.get("mag", float("nan"))),
                        errors="coerce",
                    )
                )
            except Exception:  # noqa: BLE001
                _g_tgt = float("nan")
            if math.isfinite(_g_tgt) and _calib_obj:
                err = np.asarray(
                    [
                        apply_calibration_rel(float(e), _g_tgt, _calib_obj)
                        for e in np.asarray(err, dtype=np.float64)
                    ],
                    dtype=np.float64,
                )
                logging.info(
                    "[ERR-CALIB] applied export_err_mode=calibrated for G=%.3f (%s)",
                    _g_tgt,
                    "smooth" if _smooth is not None else f"{len(_bins)} bins",
                )
    except Exception as _cal_exc:  # noqa: BLE001
        logging.warning("[ERR-CALIB] skip apply: %s", _cal_exc)
    # Propagate colour-level coefficient uncertainty into exported err (constant per LC).
    if bool(ct_ok) and math.isfinite(float(c1_stderr)) and math.isfinite(float(ct_corr)):
        # corr = c1 * (target - ref) => sigma_corr = |target-ref| * sigma_c1
        _dcol = float(target_bp_rp) - float(bp_rp_comp_med) if math.isfinite(float(bp_rp_comp_med)) else float("nan")
        if math.isfinite(_dcol):
            _err_ct = abs(_dcol) * float(c1_stderr)
            if math.isfinite(_err_ct) and _err_ct > 0:
                err = np.sqrt(np.square(np.asarray(err, dtype=np.float64)) + _err_ct**2)
                logging.info(
                    "[COLOR TERM] err += %.4f mag from k uncertainty (delta_colour=%+.3f)",
                    float(_err_ct),
                    float(_dcol),
                )
    err_photon_export, err_sem_rel_export, err_scint_rel_export, err_sigma_sys_rel_export = (
        _err_budget_components_keyed(
            err_photon_arr,
            _src_for_err,
            _ensemble_scatter_by_file,
            sigma_sys_mag=_sigma_sys_mag,
            sigma_scint_mag=_scint_mag_arr,
        )
    )
    ap_arr = target_frames["aperture_r_px"].to_numpy(dtype=float)
    src_files = target_frames["source_file"].tolist()
    sat_flags = (target_frames["flag"] == "saturated").to_numpy(dtype=bool)

    # Airmass / flip arrays for export + the democratic detrender (no per-target airmass detrend here:
    # airmass is handled by the differential comp ensemble).
    flip_arr = (
        target_frames["is_flipped"].fillna(False).astype(bool).to_numpy()
        if "is_flipped" in target_frames.columns
        else np.zeros_like(bjd, dtype=bool)
    )
    align_fail_arr = (
        target_frames["alignment_failed"].fillna(False).astype(bool).to_numpy()
        if "alignment_failed" in target_frames.columns
        else np.zeros_like(bjd, dtype=bool)
    )
    n_alignment_failed = int(np.count_nonzero(align_fail_arr))
    alignment_failed_frac = float(n_alignment_failed) / max(int(len(bjd)), 1)
    if "catalog_match_mode" in target_frames.columns:
        catalog_match_mode_list = [
            normalize_catalog_match_mode(v) for v in target_frames["catalog_match_mode"].tolist()
        ]
    else:
        catalog_match_mode_list = [""] * len(bjd)
    if "wcs_untrusted" in target_frames.columns:
        wcs_untrusted_arr = target_frames["wcs_untrusted"].fillna(False).astype(bool).to_numpy()
    else:
        wcs_untrusted_arr = np.array(
            [is_wcs_untrusted_catalog_match_mode(m) for m in catalog_match_mode_list],
            dtype=bool,
        )
    n_wcs_untrusted = int(np.count_nonzero(wcs_untrusted_arr))
    wcs_untrusted_frac = float(n_wcs_untrusted) / max(int(len(bjd)), 1)

    if "flag" in target_frames.columns:
        _raw_tf = target_frames["flag"].astype(str).str.strip().str.lower().reset_index(drop=True)
    else:
        _raw_tf = pd.Series(["__none__"] * len(mag_calib))
    base_flags: list[str] = []
    for i in range(len(mag_calib)):
        if bool(sat_flags[i]):
            base_flags.append("saturated")
        elif i < len(_raw_tf) and str(_raw_tf.iloc[i]) == "nondetection":
            base_flags.append("nondetection")
        elif math.isfinite(mag_calib[i]):
            base_flags.append("normal")
        else:
            base_flags.append("no_data")

    # Reporting path (Workstream B): see ``apply_reporting_postprocess``.
    mag_calib_raw, mag_calib, mag_calib_ct, mag_calib_ac, out_flags = apply_reporting_postprocess(
        mag_calib,
        mag_calib_ct,
        target_row=target_row,
        target_name=target_name,
        sat_flags=sat_flags,
        target_frames=target_frames,
        outlier_sigma=outlier_sigma,
        ct_ok=bool(ct_ok),
        ac_ok=bool(ac_ok),
        delta_m_corr=(float(delta_m_corr) if delta_m_corr is not None else None),
        cfg=_cfg,
    )

    # ALG-2: Savitzky-Golay non-linear detrending (Savitzky & Golay 1964)
    # Removes slow systematic trends (airmass is handled by the differential comp ensemble).
    _sg_enabled = bool(_cfg.savgol_detrend_enabled)
    if _sg_enabled:
        mag_calib = savgol_detrend_lc(
            mag_calib=mag_calib,
            bjd=bjd,
            flags=list(out_flags) if out_flags is not None else ["normal"] * len(mag_calib),
            window_frac=float(_cfg.savgol_window_frac),
            polyorder=int(_cfg.savgol_polyorder),
            enabled=True,
        )
        if ac_ok and delta_m_corr is not None and np.isfinite(float(delta_m_corr)):
            mag_calib_ac = mag_calib + float(delta_m_corr)

    # ALG-4: Democratic Detrender (arXiv:2411.09753v2, 2026)
    _dem_enabled = bool(_cfg.democratic_detrend_enabled)
    _mag_democratic: np.ndarray | None = None
    _err_inflation: np.ndarray | None = None
    if _dem_enabled:
        _mag_democratic, _err_inflation = democratic_detrend_lc(
            mag_calib=mag_calib,
            bjd=bjd,
            airmass=airmass_arr,
            flags=list(out_flags) if out_flags is not None else ["normal"] * len(mag_calib),
            window_frac=float(_cfg.democratic_sg_window_frac),
            enabled=True,
        )

    try:
        from check_star_kmag import (  # noqa: PLC0415
            build_comp_photon_mag_from_frames,
            check_kmag_sidecar_path,
            compute_check_ensemble_mag_calib,
            save_check_kmag_sidecar,
        )

        _chk_cid = _chk_cid_pref
        if _chk_cid:
            _ext_lc = dict(comp_lc)
            if _chk_cid not in _ext_lc:
                _chk_series = _get_lc(_chk_cid, all_frames)
                if _chk_series is not None and np.isfinite(_chk_series).any():
                    _ext_lc[_chk_cid] = _chk_series
            if _chk_cid in _ext_lc:
                _phot_ids = list(dict.fromkeys(list(comp_ids) + [_chk_cid]))
                _comp_photon = build_comp_photon_mag_from_frames(all_frames, _phot_ids, src_files)
                _chk_result = compute_check_ensemble_mag_calib(
                    _chk_cid,
                    list(comp_ids),
                    _ext_lc,
                    comp_catalog_mag,
                    comp_quality,
                    comp_rms_map=comp_rms_map,
                    comp_tier_map=comp_tier_map,
                    tier_weights=tier_weights,
                    cfg=_cfg,
                    n_comp_min=2,
                    n_comp_max=int(_cfg.phase01_comparison_n_comp_max),
                    comp_photon_mag=_comp_photon,
                    sigma_sys_mag=_sigma_sys_mag,
                )
                if _chk_result is not None and np.isfinite(_chk_result.kmag).any():
                    save_check_kmag_sidecar(
                        check_kmag_sidecar_path(lc_dir, target_cid),
                        check_cid=_chk_cid,
                        bjd=bjd,
                        source_files=src_files,
                        kmag=_chk_result.kmag,
                        ensemble=_chk_result,
                    )
                else:
                    logging.warning(
                        "[CHECK-KMAG] ensemble returned empty for target=%s check=%s",
                        target_cid,
                        _chk_cid,
                    )
            else:
                logging.warning(
                    "[CHECK-KMAG] check star %s has no LC series for target %s",
                    _chk_cid,
                    target_cid,
                )
        else:
            logging.warning("[CHECK-KMAG] no check star selected for target %s", target_cid)
    except (ImportError, KeyError, TypeError, ValueError, AttributeError, OSError) as _ck_exc:
        logging.warning("[CHECK-KMAG] sidecar skipped for %s: %s", target_cid, _ck_exc)

    # Krok 6: Ulozenie vystupov
    lc_csv = lc_dir / f"lightcurve_{target_cid}.csv"
    if isinstance(_lunar, dict):
        _lc_lunar_phase = float(_lunar.get("lunar_phase_pct", float("nan")))
        _lc_lunar_sep = float(_lunar.get("lunar_separation_deg", float("nan")))
        _lc_lunar_risk = str(_lunar.get("lunar_risk", "UNKNOWN") or "UNKNOWN")
    else:
        _lc_lunar_phase = float("nan")
        _lc_lunar_sep = float("nan")
        _lc_lunar_risk = "UNKNOWN"
    # I-04: exclude epochs with unmatched ensemble scatter from LC export.
    if err_scatter_unmatched_arr is not None and np.any(err_scatter_unmatched_arr):
        _keep_lc = ~np.asarray(err_scatter_unmatched_arr, dtype=bool)
        if err_method_rows is not None and len(err_method_rows) == len(_keep_lc):
            err_method_rows = [err_method_rows[i] for i in range(len(_keep_lc)) if _keep_lc[i]]
        if _mag_democratic is not None and len(_mag_democratic) == len(_keep_lc):
            _mag_democratic = np.asarray(_mag_democratic, dtype=float)[_keep_lc]
        if _err_inflation is not None and len(_err_inflation) == len(_keep_lc):
            _err_inflation = np.asarray(_err_inflation, dtype=float)[_keep_lc]
        (
            bjd,
            hjd,
            jd,
            airmass_arr,
            flip_arr,
            target_lc,
            mag_calib_raw,
            mag_calib,
            mag_calib_ct,
            mag_calib_ac,
            delta_mag,
            err,
            ap_arr,
            out_flags,
            src_files,
            align_fail_arr,
            err_scatter_unmatched_arr,
            catalog_match_mode_list,
            wcs_untrusted_arr,
            err_photon_export,
            err_sem_rel_export,
            err_scint_rel_export,
            err_sigma_sys_rel_export,
        ) = _exclude_err_scatter_unmatched_epochs(
            ~_keep_lc,
            bjd,
            hjd,
            jd,
            airmass_arr,
            flip_arr,
            target_lc,
            mag_calib_raw,
            mag_calib,
            mag_calib_ct,
            mag_calib_ac,
            delta_mag,
            err,
            ap_arr,
            out_flags,
            src_files,
            align_fail_arr,
            err_scatter_unmatched_arr,
            catalog_match_mode_list,
            wcs_untrusted_arr,
            err_photon_export,
            err_sem_rel_export,
            err_scint_rel_export,
            err_sigma_sys_rel_export,
        )
    # Pinned-era LC metadata: preserve anchor ct_n_comp for byte continuity (477dc8cf).
    if bool(ct_ok):
        try:
            from pinned_ensembles import baseline_lc_ct_n_comp_for_target, is_pinned_target  # noqa: PLC0415

            if is_pinned_target(target_cid):
                _pin_ct_n = baseline_lc_ct_n_comp_for_target(target_cid)
                if _pin_ct_n is not None:
                    ct_n_comp = int(_pin_ct_n)
        except Exception as _pin_ct_exc:  # noqa: BLE001
            LOGGER.debug("[PIN] ct_n_comp overlay skip: %s", _pin_ct_exc)
    save_lightcurve_csv(
        lc_csv,
        bjd,
        hjd,
        jd,
        airmass_arr,
        flip_arr,
        target_lc,
        mag_calib_raw,
        mag_calib,
        np.asarray(mag_calib_ct, dtype=np.float64),
        mag_calib_ac,
        delta_mag,
        err,
        ap_arr,
        out_flags,
        src_files,
        ct_correction=(float(ct_corr) if bool(ct_ok) else float("nan")),
        ct_c1=(float(c1) if bool(ct_ok) else float("nan")),
        ct_c1_stderr=(float(c1_stderr) if bool(ct_ok) else float("nan")),
        ct_mode=(str(ct_mode) if bool(ct_ok) else ""),
        ct_bp_rp_target=(float(target_bp_rp) if bool(ct_ok) else float("nan")),
        ct_bp_rp_comp_med=(float(bp_rp_comp_med) if bool(ct_ok) else float("nan")),
        ct_n_comp=(int(ct_n_comp) if bool(ct_ok) else None),
        ct_ok=bool(ct_ok),
        k2_source=k2_source_rows,
        k2_value=(float(k2_value_lc) if math.isfinite(float(k2_value_lc)) else float("nan")),
        k2_colour_ref=(float(k2_colour_ref) if math.isfinite(float(k2_colour_ref)) else float("nan")),
        ac_result=(ac_result if isinstance(ac_result, dict) else None),
        mag_democratic=_mag_democratic,
        err_inflation=_err_inflation,
        lunar_phase_pct=_lc_lunar_phase,
        lunar_separation_deg=_lc_lunar_sep,
        lunar_risk=_lc_lunar_risk,
        dilution_factor=float(_dilution_result.get("dilution_factor", 1.0)),
        method=_lc_export_method,
        alignment_failed=align_fail_arr,
        err_scatter_unmatched=err_scatter_unmatched_arr,
        catalog_match_mode=catalog_match_mode_list,
        wcs_untrusted=wcs_untrusted_arr,
        time_base=time_base,
        err_method=err_method_rows,
        sigma_sys_mag=_sigma_sys_mag,
        err_photon=err_photon_export,
        err_sem_rel=err_sem_rel_export,
        err_scint_rel=err_scint_rel_export,
        err_sigma_sys_rel=err_sigma_sys_rel_export,
        aperture_policy=getattr(state, "aperture_policy", None),
    )
    # EPSF-LC-LOG-01 / INV-PSF-SUBMIT-01: PSF LC files are an internal diagnostic
    # product written by psf_internal_lc (RUN ePSF path), not Phase 2A.

    # COMP-ASSIGN-01 D4: stability AFTER photometry - verdict only (membership unchanged).
    comp_bjd = {cid: _get_comp_bjd_series(cid, all_frames) for cid in comp_ids}
    comp_quality = check_comparison_stability(
        comp_lc,
        comp_rms_map=comp_rms_map,
        comp_bjd=comp_bjd,
        n_comp_min=3,
        outlier_sigma=stability_sigma,
        max_comp_p2p=float(_cfg.phase01_comparison_max_comp_rms),
        max_comp_slope_mmag_hr=float(_cfg.comp_max_slope_mmag_hr),
        comp_slope_significance_k=float(getattr(_cfg, "comp_slope_significance_k", 3.0)),
        common_mode_detrend=True,
        stability_run_flags=state.stability_run_flags,
    )

    # Kvalita comp pre UI (tabulka 'Porovnavacie hviezdy')
    _cq_path = lc_dir / f"comp_quality_{target_cid}.json"
    try:
        selected_tier = ""
        tier4_warning = False
        n_t1 = n_t2 = n_t3 = n_t4 = 0
        try:
            if "selected_tier" in comp_df.columns:
                _sub = _comp_index.get(target_cid, pd.DataFrame())
                if not _sub.empty:
                    stv = str(_sub.iloc[0].get("selected_tier", "") or "").strip()
                    selected_tier = stv
                    tier4_warning = bool(_sub.iloc[0].get("tier4_warning", False))
                    try:
                        n_t1 = int(pd.to_numeric(_sub.iloc[0].get("n_tier1", 0), errors="coerce") or 0)
                        n_t2 = int(pd.to_numeric(_sub.iloc[0].get("n_tier2", 0), errors="coerce") or 0)
                        n_t3 = int(pd.to_numeric(_sub.iloc[0].get("n_tier3", 0), errors="coerce") or 0)
                        n_t4 = int(pd.to_numeric(_sub.iloc[0].get("n_tier4", 0), errors="coerce") or 0)
                    except Exception:  # noqa: BLE001
                        n_t1 = n_t2 = n_t3 = n_t4 = 0
        except Exception:  # noqa: BLE001
            selected_tier = ""

        _cq_payload: dict[str, Any] = {}
        for cid, info in comp_quality.items():
            nk = _normalize_gaia_id(cid)
            q = str(info.get("quality", "") or "").strip()
            note = str(info.get("note", "") or "").strip()
            if q == "good" and not note:
                _cq_payload[nk] = "good"
            else:
                _cq_payload[nk] = {"quality": q, "note": note}
        _cq_payload["selected_tier"] = str(selected_tier)
        _cq_payload["tier4_warning"] = bool(tier4_warning)
        _cq_payload["n_tier1"] = int(n_t1)
        _cq_payload["n_tier2"] = int(n_t2)
        _cq_payload["n_tier3"] = int(n_t3)
        _cq_payload["n_tier4"] = int(n_t4)
        _cq_payload["aperture_correction"] = {
            "ok": (bool(ac_result.get("ok", False)) if isinstance(ac_result, dict) else False),
            "delta_m_corr": (ac_result.get("delta_m_corr") if isinstance(ac_result, dict) else None),
            "scatter_mag": (ac_result.get("scatter_mag") if isinstance(ac_result, dict) else None),
            "n_ref_stars": (int(ac_result.get("n_ref_stars", 0)) if isinstance(ac_result, dict) else 0),
            "ref_star_ids": (ac_result.get("ref_star_ids", []) if isinstance(ac_result, dict) else []),
            "reason": (str(ac_result.get("reason", "disabled")) if isinstance(ac_result, dict) else "disabled"),
        }
        try:
            from pinned_ensembles import get_pinned_provenance_for_target  # noqa: PLC0415

            _pin_prov = get_pinned_provenance_for_target(target_cid)
            if _pin_prov:
                _cq_payload["comp_provenance"] = _pin_prov
        except Exception as _pin_cq_exc:  # noqa: BLE001
            LOGGER.debug("[PIN] comp_provenance sidecar skip: %s", _pin_cq_exc)
        _cq_path.write_text(json.dumps(_cq_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[PHASE 2A] Optional artifact write failed (comp_quality.json): %s", exc)

    lc_png = lc_dir / f"lightcurve_{target_cid}.png"
    if _save_png:
        try:
            save_lightcurve_png(
                lc_png,
                bjd,
                mag_calib,
                err,
                out_flags,
                target_name,
                comp_quality,
                delta_mag_mode=False,
                delta_mag=delta_mag,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("[PHASE 2A] Optional artifact write failed (lightcurve PNG): %s", exc)

    cutout_png = lc_dir / f"cutout_{target_cid}.png"
    if _save_png:
        try:
            save_cutout_png(
                cutout_png,
                Path(masterstar_fits_path),
                float(target_row["x"]),
                float(target_row["y"]),
                target_name,
                ms_data=_ms_data,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("[PHASE 2A] Optional artifact write failed (cutout PNG): %s", exc)

    # Per-target field map s cislovanymi comp hviezdami - vzdy (UI)
    try:
        _target_comp = _comp_index.get(target_cid, pd.DataFrame()).copy()
        _fm_target_path = lc_dir / f"field_map_{target_cid}.png"
        save_target_field_map_png(
            _fm_target_path,
            Path(masterstar_fits_path),
            target_row,
            _target_comp,
            ms_data=_ms_data,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[PHASE 2A] Optional artifact write failed (field map PNG): %s", exc)

    # Summary riadok
    finite_calib = mag_calib[np.isfinite(mag_calib)]
    n_good_comp = sum(
        1 for q in comp_quality.values() if q.get("quality") in ("good", "suspect")
    )
    n_stability_good = sum(1 for q in comp_quality.values() if q.get("quality") == "good")
    n_stability_suspect = sum(1 for q in comp_quality.values() if q.get("quality") == "suspect")
    n_sat = sum(1 for f in out_flags if f == "saturated")

    _measured_ap = (
        float(_measured_ap_target)
        if math.isfinite(_measured_ap_target) and _measured_ap_target > 0
        else float("nan")
    )
    if not math.isfinite(_measured_ap) and not target_frames.empty and "aperture_r_px" in target_frames.columns:
        _ap_meas = pd.to_numeric(target_frames["aperture_r_px"], errors="coerce").dropna()
        if not _ap_meas.empty:
            _measured_ap = float(np.median(_ap_meas.to_numpy(dtype=float)))
    _lc_rms_full = float(np.std(finite_calib)) if len(finite_calib) > 1 else float("nan")
    _lc_rms_ooe = compute_lc_rms_ooe(mag_calib, out_flags)

    _comp_path = "default"
    _n_tier12 = 0
    if not target_comps.empty:
        if "comp_path" in target_comps.columns:
            _cpaths = target_comps["comp_path"].astype(str).str.strip().str.lower()
            if (_cpaths == "sparse_fallback").any():
                _comp_path = "sparse_fallback"
        if "comp_tier" in target_comps.columns:
            _tiers = pd.to_numeric(target_comps["comp_tier"], errors="coerce")
            _n_tier12 = int(_tiers.isin([1, 2]).sum())

    _sum_row: dict[str, Any] = {
        "catalog_id": target_cid,
        "vsx_name": target_name,
        "vsx_type": target_vsx_type,
        "zone_flag": str(target_row.get("zone_flag", "")).strip(),
        "n_frames": len(bjd),
        "n_good_comp": n_good_comp,
        "n_tier12": _n_tier12,
        "comp_path": _comp_path,
        "n_stability_good": n_stability_good,
        "n_stability_suspect": n_stability_suspect,
        "n_saturated": n_sat,
        "n_alignment_failed": n_alignment_failed,
        "alignment_failed_frac": alignment_failed_frac,
        "n_wcs_untrusted": n_wcs_untrusted,
        "wcs_untrusted_frac": wcs_untrusted_frac,
        "lc_rms": _lc_rms_full,
        "lc_rms_ooe": _lc_rms_ooe,
        "lc_median_mag": float(np.median(finite_calib)) if len(finite_calib) > 0 else float("nan"),
        "aperture_px": _measured_ap if math.isfinite(_measured_ap) else float(apertures_px.get(target_cid, float("nan"))),
        "aperture_px_planned": float(apertures_px.get(target_cid, float("nan"))),
        "am_slope": float("nan"),
        "am_detrended": False,
        "dilution_factor": float(_dilution_result.get("dilution_factor", 1.0)),
        "dilution_delta_mag": float(_dilution_result.get("dilution_delta_mag", 0.0)),
        "n_neighbors_aperture": int(_dilution_result.get("n_neighbors", 0)),
        "gs11_aperture_arcsec": float(_dilution_result.get("aperture_arcsec", float("nan"))),
        "gs11_dilution_skipped": bool(_dilution_result.get("dilution_skipped", False)),
        "gs11_dilution_skip_reason": str(_dilution_result.get("dilution_skip_reason", "") or ""),
        "mag_median_pre_gs11": _mag_pre_gs11,
        "mag_median_post_gs11": _mag_post_gs11,
        "lc_csv": str(lc_csv),
        "lc_png": str(lc_png),
        "ct_ok": bool(ct_ok),
        "ct_corr": float(ct_corr) if bool(ct_ok) and math.isfinite(float(ct_corr)) else float("nan"),
        "ct_c1": float(c1) if bool(ct_ok) and math.isfinite(float(c1)) else float("nan"),
        "ct_c1_stderr": float(c1_stderr) if bool(ct_ok) and math.isfinite(float(c1_stderr)) else float("nan"),
        "ct_mode": str(ct_mode) if bool(ct_ok) else "",
        "ct_n_comp": int(ct_n_comp) if bool(ct_ok) else 0,
        **_ac_summary_fields(ac_result if bool(_cfg.aperture_correction_enabled) else {"ok": False, "reason": "disabled"}),
    }
    if _pfs_on:
        _sum_row["skip_reason"] = str(target_row.get("skip_reason", "") or "")
        _sum_row["sat_clean_frac"] = float(
            pd.to_numeric(target_row.get("sat_clean_frac"), errors="coerce")
        )
        _sum_row["per_frame_sat_fallback"] = bool(
            target_row.get("per_frame_sat_fallback", False)
        )
    summary_rows.append(_sum_row)
    n_lc += 1
    lc_rms = float(summary_rows[-1]["lc_rms"])
    lc_rms_ooe = float(summary_rows[-1].get("lc_rms_ooe", float("nan")))
    r_ap = float(summary_rows[-1]["aperture_px"])
    logging.info(
        f"[FAZA 2A] {target_name}: "
        f"lc_rms={lc_rms:.4f}, lc_rms_ooe={lc_rms_ooe:.4f}, "
        f"n_comp={n_good_comp} (stability_good={n_stability_good}), "
        f"apertura={r_ap:.2f}px (measured)"
    )


    state.chip_fw = chip_fw
    state.chip_fh = chip_fh
    return summary_rows, n_lc
