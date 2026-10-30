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
from photometry_shared import _normalize_gaia_id, _target_display_name

from aperture_photometry import (
    _AperturePhotometryState,
    _measured_aperture_from_proc_cache,
    _resolve_photometric_aperture_px_for_gs11,
    apply_reporting_postprocess,
    read_flux_from_csv,
)
from photometry_lightcurve import (
    _combine_err_with_ensemble_scatter_keyed,
    _ensemble_scatter_by_source_file,
    _err_budget_components_keyed,
    _exclude_err_scatter_unmatched_epochs,
    _get_comp_bjd_series,
    _get_lc,
    _aperture_skip_empty_comps_target,
    _recompute_bjd_hjd_with_status,
    check_comparison_stability,
    compute_lc_rms_ooe,
    ensemble_normalize,
    frame_rows_in_time_order,
    save_lightcurve_csv,
    save_lightcurve_png,
)

from mag_constants import MAG_ERR_SCALE
from photometry_defs import LOGGER



def _vsx_name_of(target_row: Any) -> str:
    """The VSX name of a target (empty for manual / exoplanet / candidate targets, G11-22)."""
    v = str(target_row.get("vsx_name", "") or "").strip()
    return "" if v.lower() in ("nan", "none") else v


def zero_point_uncertainty(
    weights: dict[str, float],
    cat_mag: dict[str, float],
    comp_level: dict[str, float],
    band: str,
) -> dict[str, Any]:
    """Uncertainty of the absolute level C of the light curve (ZP-ERR-2026-10).

    C = sum_j w_j (cat_j - M_j) / sum_j w_j over the comps with a valid band magnitude
    (photometry_lightcurve.ensemble_normalize). Error of a weighted mean of independent
    estimates, n_eff = (sum w)^2 / sum w^2:

    * model: sigma_rel / sqrt(n_eff), sigma_rel = scatter of the Gaia DR3 -> band relation
      (DR3 documentation 5.5.1, Tables 5.8 / 5.9; gaia_bands). Lower bound - stars of one
      field share reddening / population, so their relation errors are partly correlated.
    * data (n >= 3): weighted scatter of (cat_j - M_j) among those comps / sqrt(n_eff),
      M_j = the Honeycutt reference level of the comp from ensemble_normalize (the level
      C uses); unbiased weighted variance with reliability weights, factor
      n_eff / (n_eff - 1).
    """
    from gaia_bands import band_mag_from_gaia  # noqa: PLC0415

    out: dict[str, Any] = {"band": band, "n_comp": 0, "n_eff": None, "sigma_rel_mag": None,
                           "sigma_zp_model_mag": None, "sigma_zp_data_mag": None}
    if band == "NONE":
        return out
    ids = [c for c in weights if math.isfinite(float(cat_mag.get(c, float("nan"))))]
    if not ids:
        return out
    w = np.array([float(weights[c]) for c in ids], dtype=float)
    d = np.array([float(cat_mag[c]) - float(comp_level.get(c, float("nan"))) for c in ids])
    ok = np.isfinite(w) & (w > 0) & np.isfinite(d)
    w, d = w[ok], d[ok]
    n = int(w.size)
    out["n_comp"] = n
    if n == 0:
        return out
    n_eff = float(w.sum() ** 2 / np.sum(w * w))
    sig_rel = float(band_mag_from_gaia(15.0, 1.0, band)[1])
    out["n_eff"] = round(n_eff, 3)
    if math.isfinite(sig_rel) and sig_rel > 0:
        out["sigma_rel_mag"] = round(sig_rel, 5)
        out["sigma_zp_model_mag"] = round(sig_rel / math.sqrt(n_eff), 5)
    if n >= 3 and n_eff > 1.0:
        mu = float(np.sum(w * d) / w.sum())
        s = math.sqrt(float(np.sum(w * (d - mu) ** 2) / w.sum()) * n_eff / (n_eff - 1.0))
        out["sigma_zp_data_mag"] = round(s / math.sqrt(n_eff), 5)
    return out

def _tgt_comp_inputs(*, obs_group, target_cid, target_comps):
    """Comparison ids of the target, their band magnitudes (zero-point band or NONE) and pool RMS."""
    comp_ids: list[str] = []
    _seen_comp: set[str] = set()
    for c in target_comps["catalog_id"].tolist():
        nc = _normalize_gaia_id(c)
        if nc and nc not in _seen_comp:
            _seen_comp.add(nc)
            comp_ids.append(nc)
    all_ids = [target_cid] + comp_ids

    # BANDAUTH-2026-10: comparison catalogue magnitudes are ALWAYS in the zero-point band of
    # the filter (gaia_bands.filter_band), independent of the colour-term switch. A comp
    # outside the valid colour range of the Gaia -> band relation carries NaN: it still
    # shapes the light curve (ensemble) but does not enter the zero point.
    # COMP-BAND-2026-10: at least ONE comp with a valid band magnitude calibrates the LC;
    # none -> band_system "NONE" (written for inspection, export refuses it).
    from gaia_bands import filter_band_for_obs_group, zp_mag_from_gaia  # noqa: PLC0415

    _fb = filter_band_for_obs_group(obs_group)
    comp_catalog_mag = {}
    for _, r in target_comps.iterrows():
        _cidb = _normalize_gaia_id(r["catalog_id"])
        _g = float(pd.to_numeric(r.get("phot_g_mean_mag", r.get("mag")), errors="coerce"))
        if not math.isfinite(_g):
            _g = float(pd.to_numeric(r.get("mag"), errors="coerce"))
        comp_catalog_mag[_cidb] = zp_mag_from_gaia(
            _g, float(pd.to_numeric(r.get("bp_rp"), errors="coerce")), _fb
        )
    _n_band_ok = sum(1 for v in comp_catalog_mag.values() if math.isfinite(v))
    _band_system = _fb.zp_band if _n_band_ok >= 1 else "NONE"
    if _band_system == "NONE":
        # LOG-NOISE-2026-10: per target INFO, one summary WARNING after the target loop
        logging.info("[BAND] %s: no comp with a valid %s magnitude - differential LC (no zero point)",
                     target_cid, _fb.label)
    else:
        logging.info("[BAND] %s: zero point from %d of %d comps with a valid %s magnitude",
                     target_cid, _n_band_ok, len(comp_catalog_mag), _fb.label)
    comp_rms_map: dict[str, float] = {}
    for _, r in target_comps.iterrows():
        cid0 = _normalize_gaia_id(r["catalog_id"])
        try:
            rms_raw = float(r.get("comp_rms", float("nan")))
        except Exception:  # noqa: BLE001
            rms_raw = float("nan")
        comp_rms_map[cid0] = float(rms_raw)
    return _band_system, _fb, all_ids, comp_catalog_mag, comp_ids, comp_rms_map


def _tgt_choose_check_star(
    *,
    _cfg,
    _fb,
    all_ids,
    comp_catalog_mag,
    comp_ids,
    state,
    target_cid,
    target_comps,
    target_row,
):
    """Check star of the target: the comp-table check role, else the best reserve / pool star."""
    from gaia_bands import zp_mag_from_gaia  # noqa: PLC0415

    _chk_cid_pref: str | None = None
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
            # CHECK-NOT-SUSPECT-2026-10: comps dropped by COMP-SMART as unstable are no check stars
            from check_star_kmag import drop_smart_rejected_check_candidates  # noqa: PLC0415

            _chk_pool_pref = drop_smart_rejected_check_candidates(
                _chk_pool_pref, (getattr(state, "smart_log", None) or {}).get(str(target_cid)), str(target_cid),
                catalog_offset_excluded=getattr(state, "catoff_excluded", None),
            )
            # BLEND-K-2026-10: the check star is measured in the target's aperture, so it
            # obeys the comparison-star rule there: |neighbour light| <= 1 % at the chosen k
            # (AAVSO: "exclude close doubles"). Unknown (NaN) does not block, as for comps.
            _pt_chk = (getattr(state, "per_target_aperture", None) or {}).get(str(target_cid))
            if _pt_chk is not None and not _chk_pool_pref.empty:
                from check_star_kmag import drop_blended_check_candidates  # noqa: PLC0415

                _n0 = len(_chk_pool_pref)
                _chk_pool_pref = drop_blended_check_candidates(_chk_pool_pref, float(_pt_chk.f_star))
                if len(_chk_pool_pref) < _n0:
                    logging.info("[CHECK-KMAG] %s: %d check candidates with > 1%% neighbour light at "
                                 "k=%.2f excluded", target_cid, _n0 - len(_chk_pool_pref), float(_pt_chk.f_star))
            if not _chk_pool_pref.empty:
                _chk_row_pref = select_check_star(
                    _chk_pool_pref,
                    ensemble_ids=set(comp_ids),
                    n_comp_min=max(1, min(3, len(_chk_pool_pref))),
                    cfg=_cfg,
                    target_bp_rp=pd.to_numeric(target_row.get("bp_rp"), errors="coerce"),
                    target_mag=pd.to_numeric(
                        target_row.get("mag", target_row.get("phot_g_mean_mag")), errors="coerce"
                    ),
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
                # BANDAUTH-2026-10: check-star reference magnitude in the same band as the comps.
                for _mk in ("mag", "phot_g_mean_mag"):
                    try:
                        _cm = float(pd.to_numeric(_chk_row_pref.get(_mk), errors="coerce"))
                    except Exception:  # noqa: BLE001
                        _cm = float("nan")
                    if math.isfinite(_cm):
                        comp_catalog_mag[_chk_cid_pref] = zp_mag_from_gaia(
                            _cm,
                            float(pd.to_numeric(_chk_row_pref.get("bp_rp"), errors="coerce")),
                            _fb,
                        )
                        break
    return _chk_cid_pref,


def _tgt_frame_rows(
    *,
    _aperture_csv_cache,
    _aperture_lookup_cache,
    _flux_matrix,
    _gain_phot,
    _rn_phot,
    all_ids,
    apertures_px,
    chip_fh,
    chip_fw,
    csv_files,
    frame_time_lookup,
    sat_limit_resolved,
    state,
):
    """Per-frame rows (target + comps) from the night flux matrix with saturation / edge / alignment flags."""
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
            _cached_df = _aperture_csv_cache.get(str(csv_path))
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
            # G06-07: the chip / footprint edge verdict comes from the flux grid only (AP-FOOTPRINT-2026-10)
            frame_results.append(df_frame)
    else:
        for csv_path in csv_files:
            _ft = frame_time_lookup.get(csv_path.stem)
            _key_csv = str(csv_path)
            _cached_df = _aperture_csv_cache.get(_key_csv)
            _lookup_row = _aperture_lookup_cache.get(_key_csv)

            df_frame = read_flux_from_csv(
                csv_path,
                all_ids,
                apertures_px,
                sat_limit_adu=sat_limit_resolved,
                frame_times=_ft,
                csv_df=_cached_df,
                lookup=_lookup_row,
                gain=float(_gain_phot),
                read_noise=float(_rn_phot),
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

                # G06-07: edge verdict from the flux grid only
                frame_results.append(df_frame)
    return chip_fh, chip_fw, frame_results


def _tgt_apply_night_k(
    *,
    _chk_cid_pref,
    annulus_inner_fwhm,
    annulus_outer_fwhm,
    comp_ids,
    frame_results,
    state,
    target_cid,
):
    """Concatenate the frame rows and apply the target's per-target aperture k (AP-PERTARGET)."""
    all_frames = pd.concat(frame_results, ignore_index=True)

    # APERTURE-PERTARGET-01: rebuild mag_inst at f* for target + its comps (same f).
    _pt_choice = (getattr(state, "per_target_aperture", None) or {}).get(str(target_cid))
    _pt_grid = getattr(state, "aperture_grid_night", None)
    if _pt_choice is not None and _pt_grid is not None:
        from aperture_pertarget import apply_grid_fluxes_to_frames  # noqa: PLC0415

        _apply_ids = [str(target_cid)] + [str(c) for c in comp_ids]
        if _chk_cid_pref and str(_chk_cid_pref) not in _apply_ids:
            _apply_ids.append(str(_chk_cid_pref))  # check star in the same aperture
        all_frames = apply_grid_fluxes_to_frames(
            all_frames,
            _pt_grid,
            catalog_ids=_apply_ids,
            f_star=float(_pt_choice.f_star),
            gain=float(state._gain_phot) if state._gain_phot is not None else None,
            read_noise=float(state._rn_phot) if state._rn_phot is not None else None,
            annulus_inner_fwhm=float(annulus_inner_fwhm),
            annulus_outer_fwhm=float(annulus_outer_fwhm),
        )
        logging.info(
            "[APERTURE-DYNAMIC] target %s f*=%.3f r_ap(night_scale)=%.3f px (n_comp=%d)",
            target_cid,
            float(_pt_choice.f_star),
            float(_pt_choice.r_ap_px),
            len(comp_ids),
        )
    return all_frames,


def _tgt_ensemble(*, _band_system, _cfg, _fb, all_frames, comp_catalog_mag, comp_ids, target_cid):
    """Ensemble zero point (Broeg weights) of the target: calibrated and differential magnitudes, ZP uncertainty."""
    target_lc = _get_lc(target_cid, all_frames)
    comp_lc = {cid: _get_lc(cid, all_frames) for cid in comp_ids}

    # COMP-ASSIGN-01 D4/D5: membership is fixed by comparison-star selection (3-8). Stability is a
    # post-photometry verdict only - do not let it re-select before ensemble.
    comp_quality = {cid: {"quality": "good"} for cid in comp_ids}

    # Step 4: ensemble normalization (consumes the delivered set as given)
    _broeg_w: dict[str, float] = {}
    _comp_levels: dict[str, float] = {}
    mag_calib, delta_mag, ensemble_scatter = ensemble_normalize(
        target_lc,
        comp_lc,
        comp_catalog_mag,
        comp_quality,
        n_comp_min=max(1, int(getattr(_cfg, "phase01_comparison_n_comp_min", 3))),
        weights_out=_broeg_w,
        levels_out=_comp_levels,
    )
    # COMP-BAND-2026-10: calibrated only if a comp with a valid band magnitude actually
    # entered the ensemble (a valid comp without measurements cannot set the level).
    if _band_system != "NONE" and not any(
        math.isfinite(float(comp_catalog_mag.get(_c, float("nan")))) for _c in (_broeg_w or {})
    ):
        logging.info("[BAND] %s: no measured comp with a valid %s magnitude - differential LC (no zero point)",
                     target_cid, _fb.label)
        _band_system = "NONE"
    _zp_info = zero_point_uncertainty(
        _broeg_w, comp_catalog_mag, _comp_levels, _fb.zp_band if _band_system != "NONE" else "NONE"
    )
    # UNCAL-LC-2026-10: without a band zero point the light curve is the differential delta_mag.
    _calibrated = _band_system != "NONE"
    # ENSEMBLE-BROEG-2026-10: the colour-term correction sum_j w_j P(C_j) / sum_j w_j must use
    # the SAME weights as the ensemble zero point, i.e. the Broeg weights.
    comp_weight_map: dict[str, float] = {str(k): float(v) for k, v in _broeg_w.items()}
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
    return _band_system, _broeg_w, _calibrated, _comp_levels, _dilution_result, _ensemble_scatter_by_file, _zp_info, comp_lc, comp_quality, comp_weight_map, delta_mag, mag_calib, target_lc


def _tgt_dilution(
    *,
    _cfg,
    _dilution_result,
    apertures_px,
    fwhm_px,
    mag_calib,
    obs_group,
    state,
    target_bp_rp_by_cid,
    target_cid,
    target_row,
):
    """GS11 flux dilution by Gaia neighbours (only with gs11_dilution_enabled)."""
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
        _dilution_skipped_ap = False
        _ap_px, _ap_src = _resolve_photometric_aperture_px_for_gs11(
            target_cid,
            apertures_px,
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
            from gaia_bands import band_for_obs_group  # noqa: PLC0415

            _dilution_result = compute_dilution_factor(
                _target_ra,
                _target_dec,
                _target_g_mag,
                _ap_arcsec,
                str(state.gaia_db_path),
                catalog_id=_cid_int,
                fwhm_arcsec=float(fwhm_px) * float(state.plate_scale_arcsec),
                band=band_for_obs_group(obs_group),
                target_bp_rp=float(target_bp_rp_by_cid.get(target_cid, float("nan"))),
            )
        mag_calib, _dilution_result = apply_target_dilution_to_mag_calib(
            mag_calib,
            _dilution_result,
            _cfg,
            target_cid=str(target_cid),
        )
    return _dilution_result, mag_calib


def _tgt_k2(
    *,
    _calibrated,
    _cfg,
    all_frames,
    comp_quality,
    comp_weight_map,
    delta_mag,
    mag_calib,
    state,
    target_bp_rp_by_cid,
    target_cid,
    target_comps,
    target_row,
):
    """Colours, target airmass and the second-order extinction k'' correction."""
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

    # EXPORT-AUDIT-2026-10: airmass of the target itself (k'', scintillation, light curve, exports)
    from photometry_lightcurve import target_airmass  # noqa: PLC0415

    _tf_am = all_frames[all_frames["catalog_id"] == target_cid]
    _target_airmass_arr = target_airmass(
        _tf_am["jd"].to_numpy(dtype=float),
        float(pd.to_numeric(target_row.get("ra_deg", target_row.get("ra", float("nan"))), errors="coerce")),
        float(pd.to_numeric(target_row.get("dec_deg", target_row.get("dec", float("nan"))), errors="coerce")),
        _cfg,
        site=(state.site_lat, state.site_lon, state.site_alt) if state.site_ok else None,
        fallback=(_tf_am["airmass"].to_numpy(dtype=float) if "airmass" in _tf_am.columns else None),
    )

    from k2_extinction import K2Source, apply_k2_per_frame, bp_rp_comp_median, ensemble_bp_rp  # noqa: PLC0415

    k2_value_lc = float("nan")
    k2_colour_ref = float("nan")
    # EPSF-K2-2026-10: k'' x (target - comp colour) per unit airmass, written with the light curve
    # (``k2_delta_mag``) so the PSF light curve applies the same per-epoch term.
    k2_coeff_lc = float("nan")
    k2_source_rows = [K2Source.NONE.value] * len(mag_calib)
    _k2_val = float(getattr(state, "k2_bprp", float("nan")))
    _k2_src = str(getattr(state, "k2_source", K2Source.NONE.value))
    if _k2_src == K2Source.LITERATURE_DEFAULT.value and math.isfinite(_k2_val):
        _airmass_k2 = _target_airmass_arr
        # K2-ONE-2026-10 (Milan 2026-10-26, D4): the colour of the comparison is the colour of
        # the ensemble the light curve uses - the Broeg-weighted mean colour of its comps.
        _bp_med_k2 = ensemble_bp_rp(comp_bp_rp, comp_weight_map)
        if not math.isfinite(_bp_med_k2):
            _bp_med_k2 = bp_rp_comp_median(comp_bp_rp, comp_quality)
        _k2_src_enum = K2Source.LITERATURE_DEFAULT
        # k'' x airmass x (target - ensemble colour) is a differential effect: one convention,
        # applied to delta_mag and to mag_calib alike (Henden: dm0 = dm - k'' X dC).
        delta_mag, _, k2_source_rows = apply_k2_per_frame(
            delta_mag,
            _airmass_k2,
            object_bp_rp=float(target_bp_rp),
            bp_rp_comp_med=_bp_med_k2,
            k2_value=_k2_val,
            k2_source=_k2_src_enum,
        )
        if _calibrated:
            mag_calib, _, _ = apply_k2_per_frame(
                mag_calib, _airmass_k2, object_bp_rp=float(target_bp_rp), bp_rp_comp_med=_bp_med_k2,
                k2_value=_k2_val, k2_source=_k2_src_enum,
            )
        k2_value_lc = _k2_val
        k2_colour_ref = _bp_med_k2
        if any(str(v) != K2Source.NONE.value for v in k2_source_rows):
            k2_coeff_lc = float(_k2_val) * (float(target_bp_rp) - float(_bp_med_k2))
    # summary: the k'' source of the target (the tag when any epoch got the correction)
    k2_source_target = next((str(v) for v in k2_source_rows if str(v) != K2Source.NONE.value), K2Source.NONE.value)
    return _target_airmass_arr, comp_bp_rp, delta_mag, k2_coeff_lc, k2_colour_ref, k2_source_target, k2_value_lc, mag_calib, target_bp_rp


def _band_ct_correction(pf: Any, colour: float, comp_bp_rp: dict, comp_catalog_mag: dict,
                        comp_weight_map: dict) -> tuple[float, float]:
    """(correction, weighted ensemble colour) of the band colour term for a star of ``colour``:
    P(C) - sum w_j P(C_j) / sum w_j over the zero-point comparison stars (CT-2026-10). NaN outside the
    fitted colour range (no extrapolation). One implementation for the target and the check star."""
    cc, ww = [], []
    for cid, bp in comp_bp_rp.items():
        w = float(comp_weight_map.get(cid, float("nan")))
        if math.isfinite(float(comp_catalog_mag.get(cid, float("nan")))) and math.isfinite(w) and w > 0:
            cc.append(float(bp))
            ww.append(w)
    if not cc:
        return float("nan"), float("nan")
    ens = float(np.sum(np.asarray(ww) * np.asarray(cc)) / np.sum(ww))
    if pf is None:
        return 0.0, float(np.median(cc))
    if not (math.isfinite(float(colour)) and pf.c_min <= float(colour) <= pf.c_max):
        return float("nan"), ens
    return float(pf.correction(float(colour), cc, ww)), ens


def _tgt_colour_term(
    *,
    _band_system,
    comp_bp_rp,
    comp_catalog_mag,
    comp_quality,
    comp_weight_map,
    mag_calib,
    state,
    target_bp_rp,
    target_cid,
    target_name,
):
    """Group colour term applied to the target when the gate allows it (no extrapolation)."""
    c1 = 0.0
    c1_stderr = float("nan")
    ct_mode = ""
    ct_n_comp = 0
    mag_calib_ct = mag_calib.copy()
    ct_corr = 0.0
    bp_rp_comp_med = float("nan")
    ct_ok = False
    _group_ct = state.group_color_term
    if state.apply_color_term and _group_ct is not None and str(getattr(_group_ct, "mode", "")) == "band":
        # CT-2026-10: zero point already in the band; add the polynomial colour term
        # dm = P(C_t) - sum w_j P(C_j) / sum w_j with the zero-point weights.
        _pf = getattr(_group_ct, "poly_fit", None)
        ct_mode = f"band:{_band_system}:o{_pf.order if _pf is not None else 0}"
        ct_n_comp = int(_group_ct.n_comp)
        if _band_system not in ("G", "NONE"):
            ct_corr, bp_rp_comp_med = _band_ct_correction(_pf, float(target_bp_rp), comp_bp_rp, comp_catalog_mag,
                                                          comp_weight_map)
            ct_ok = math.isfinite(ct_corr)
            if _pf is not None and ct_ok:
                c1 = float(_pf.coeffs[0])
            if not ct_ok:
                logging.info("[COLOR TERM] %s: target colour outside the fitted range -> no colour term", target_cid)
                ct_corr = 0.0
            if not math.isfinite(ct_corr):
                ct_corr, ct_ok = 0.0, False
            mag_calib_ct = mag_calib + float(ct_corr)
    return _group_ct, bp_rp_comp_med, c1, c1_stderr, ct_corr, ct_mode, ct_n_comp, ct_ok, mag_calib_ct


def _tgt_time_and_errors(
    *,
    _cfg,
    _ensemble_scatter_by_file,
    _target_airmass_arr,
    all_frames,
    bp_rp_comp_med,
    c1_stderr,
    ct_corr,
    ct_ok,
    mag_calib,
    output_dir,
    state,
    target_bp_rp,
    target_cid,
    target_name,
    target_row,
):
    """Times (BJD/HJD), error budget (photon, ensemble scatter, scintillation, night floor, CT) and per-epoch flags."""
    target_frames = all_frames[all_frames["catalog_id"] == target_cid]
    target_frames = frame_rows_in_time_order(target_frames)
    _pt_active = bool((getattr(state, "per_target_aperture", None) or {}).get(str(target_cid)))
    # SPEED-2026-10 (S1): the proc-cache (comparison-star selection global) radius is read only when the target has
    # no per-target aperture; otherwise it is never used.
    _measured_ap_target = (
        float("nan") if _pt_active else _measured_aperture_from_proc_cache(target_cid, state._aperture_csv_cache)
    )
    if (
        (not _pt_active)
        and math.isfinite(_measured_ap_target)
        and _measured_ap_target > 0
        and not target_frames.empty
    ):
        target_frames = target_frames.copy()
        target_frames["aperture_r_px"] = float(_measured_ap_target)
    # APERTURE-DYNAMIC-01: when per_target, apply_grid_fluxes_to_frames already
    # stamped per-frame r = f* x FWHM_frame; do not overwrite with night median.
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
    _err_rel_for_snr = np.asarray(err, dtype=np.float64).copy()  # relative flux error (photon)
    err_photon_arr = np.asarray(err, dtype=np.float64).copy()
    if len(_target_airmass_arr) == len(target_frames):
        airmass_arr = np.asarray(_target_airmass_arr, dtype=float)
    elif "airmass" in target_frames.columns:
        airmass_arr = target_frames["airmass"].to_numpy(dtype=float)
    else:
        airmass_arr = np.full(len(target_frames), float("nan"), dtype=float)
    # Per-point uncertainty = photon/SNR base error (term-1) (+) ensemble zeropoint uncertainty
    # (term-3, ``ensemble_scatter``). Joined by EXACT ``source_file`` (G2-F004), not positional index.
    _src_for_err = target_frames["source_file"].astype(str).tolist()
    from sigma_budget import resolve_rig_scintillation_params  # noqa: PLC0415
    from utils import exposure_seconds  # noqa: PLC0415
    from sigma_floor_core import resolve_sigma_sys_mag, scintillation_mag_per_epoch  # noqa: PLC0415

    _sigma_sys_mag = resolve_sigma_sys_mag(
        state.equipment_id,
        _cfg,
        rig_label=str(state.obs_group or ""),
    )
    # SIGMA-SYS-NIGHT-2026-10: the floor measured on this night's comps replaces the constant
    _ssn = getattr(state, "sigma_sys_night", None)
    if _ssn is not None and math.isfinite(float(_ssn)):
        _sigma_sys_mag = float(_ssn)
    _draft_id_lc: int | None = None
    try:
        from processed_field import processed_field_id  # noqa: PLC0415

        _draft_id_lc = processed_field_id(output_dir)
    except Exception:  # noqa: BLE001
        _draft_id_lc = None
    _rig_scint = resolve_rig_scintillation_params(
        draft_id=_draft_id_lc,
        setup=str(state.obs_group or ""),
        exposure_s=exposure_seconds(state._ms_header),
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
    # Propagate colour-level coefficient uncertainty into exported err (constant per LC).
    if bool(ct_ok) and math.isfinite(float(c1_stderr)) and math.isfinite(float(ct_corr)):
        # corr = c1 * (target - ref) => sigma_corr = |target-ref| * sigma_c1
        _dcol = float(target_bp_rp) - float(bp_rp_comp_med) if math.isfinite(float(bp_rp_comp_med)) else float("nan")
        if math.isfinite(_dcol):
            _err_ct = abs(_dcol) * float(c1_stderr)
            if math.isfinite(_err_ct) and _err_ct > 0:
                # ``err`` is a relative flux error; the colour-term error is in mag.
                err = np.sqrt(np.square(np.asarray(err, dtype=np.float64)) + (_err_ct / 1.0857362) ** 2)
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

    # Alignment / catalogue-match counts for the summary (no per-target airmass detrend here:
    # airmass is handled by the differential comp ensemble).
    align_fail_arr = (
        target_frames["alignment_failed"].fillna(False).astype(bool).to_numpy()
        if "alignment_failed" in target_frames.columns
        else np.zeros_like(bjd, dtype=bool)
    )
    # ALIGN-VERIFY-2026-10: frames skipped at alignment (no proc CSV) count as failed epochs too
    _n_al_skip = int(getattr(state, "n_align_skipped", 0) or 0)
    n_alignment_failed = int(np.count_nonzero(align_fail_arr)) + _n_al_skip
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

    if "flag" in target_frames.columns:
        _raw_tf = target_frames["flag"].astype(str).str.strip().str.lower().reset_index(drop=True)
    else:
        _raw_tf = pd.Series(["__none__"] * len(mag_calib))
    return _err_rel_for_snr, _measured_ap_target, _pt_active, _raw_tf, _scint_mag_arr, _sigma_sys_mag, airmass_arr, ap_arr, bjd, err, err_photon_export, err_scatter_unmatched_arr, err_scint_rel_export, err_sem_rel_export, err_sigma_sys_rel_export, hjd, jd, n_alignment_failed, n_wcs_untrusted, sat_flags, src_files, target_frames, time_base


def _tgt_gspc_zero_point(
    *,
    _broeg_w,
    _calibrated,
    _comp_levels,
    _fb,
    _group_ct,
    comp_bp_rp,
    comp_catalog_mag,
    comp_weight_map,
    ct_corr,
    ct_ok,
    state,
    target_bp_rp,
    target_cid,
    target_row,
):
    """GSPC second zero point and the zero-point note of the target."""
    from gspc_zp import GSPC_BAND, ZP_NOTE_MIN_DIFF_MAG, gspc_zero_point  # noqa: PLC0415

    _gspc_zp: dict[str, Any] = {"status": str(getattr(state, "gspc_status", "not used")), "band": _fb.zp_band}
    # mag_gspc = base + _gspc_shift; base = mag_calib (before CT) when calibrated, else delta_mag
    _gspc_shift = float("nan")
    _c_std = float("nan")
    try:
        if getattr(state, "gspc_rows", None) and _fb.zp_band in GSPC_BAND and _broeg_w:
            _gspc_zp.update(gspc_zero_point(_broeg_w, _comp_levels, state.gspc_rows, _fb.zp_band))
            _cg = float(_gspc_zp.get("C", float("nan")))
            if math.isfinite(_cg):
                if _calibrated:
                    _ok_c = [c for c in _broeg_w if math.isfinite(float(comp_catalog_mag.get(c, float("nan"))))
                             and math.isfinite(float(_comp_levels.get(c, float("nan"))))]
                    if _ok_c:
                        _sw = sum(float(_broeg_w[c]) for c in _ok_c)
                        _c_std = sum(float(_broeg_w[c]) * (float(comp_catalog_mag[c]) - float(_comp_levels[c]))
                                     for c in _ok_c) / _sw
                        # colour term on the SAME comps as the GSPC zero point (band-mode CT:
                        # P(C_t) - sum w P(C_j) / sum w); other CT modes are per-target constants.
                        _ct_g = float(ct_corr) if bool(ct_ok) else 0.0
                        _pf_g = getattr(_group_ct, "poly_fit", None) if _group_ct is not None else None
                        if bool(ct_ok) and str(getattr(_group_ct, "mode", "")) == "band" and _pf_g is not None:
                            # only comps inside the colour range of the CT fit (no extrapolation)
                            _acc_g = [c for c in (_gspc_zp.get("accepted") or []) if c in comp_bp_rp
                                      and _pf_g.c_min <= float(comp_bp_rp[c]) <= _pf_g.c_max]
                            _cc_g = [float(comp_bp_rp[c]) for c in _acc_g]
                            _ww_g = [float(comp_weight_map.get(c, float("nan"))) for c in _acc_g]
                            _okw = [i for i, v in enumerate(_ww_g) if math.isfinite(v) and v > 0]
                            if _okw:
                                _v = float(_pf_g.correction(float(target_bp_rp), [_cc_g[i] for i in _okw],
                                                            [_ww_g[i] for i in _okw]))
                                if math.isfinite(_v):
                                    _ct_g = _v
                        _gspc_shift = _cg - _c_std + _ct_g
                        _gspc_zp["diff_vs_std_mag"] = round(_gspc_shift - (float(ct_corr) if bool(ct_ok) else 0.0), 5)
                else:
                    _gspc_shift = _cg + float(_gspc_zp.get("inst_level", float("nan")))
    except Exception as _gspc_exc:  # noqa: BLE001
        LOGGER.warning("[GSPC] %s: second zero point skipped (%s)", target_cid, _gspc_exc)
        _gspc_shift = float("nan")
        _gspc_zp["status"] = f"error: {_gspc_exc}"[:200]
    _gspc_zp["C"] = _gspc_zp.get("C") if math.isfinite(float(_gspc_zp.get("C", float("nan")) or float("nan"))) else None
    _gspc_zp["C_std"] = _c_std if math.isfinite(_c_std) else None
    _gspc_zp.setdefault("diff_vs_std_mag", None)
    _zp_note = ""
    if not _calibrated:
        _zp_note = "A"
    elif _gspc_zp["diff_vs_std_mag"] is not None:
        if abs(float(_gspc_zp["diff_vs_std_mag"])) >= ZP_NOTE_MIN_DIFF_MAG:
            _zp_note = "B"
    else:
        try:
            from astropy import units as _u_b  # noqa: PLC0415
            from astropy.coordinates import SkyCoord as _SC_b  # noqa: PLC0415

            _gb = float(_SC_b(float(target_row.get("ra_deg")) * _u_b.deg,
                              float(target_row.get("dec_deg")) * _u_b.deg).galactic.b.deg)
        except Exception:  # noqa: BLE001
            _gb = float("nan")
        _cbp = [float(v) for v in comp_bp_rp.values() if math.isfinite(float(v))]
        if math.isfinite(_gb) and abs(_gb) < 10.0 and _cbp and float(np.median(_cbp)) > 1.5:
            _zp_note = "B-heuristic"
    _gspc_zp["note"] = _zp_note
    try:
        from astropy import units as _u_n  # noqa: PLC0415
        from astropy.coordinates import SkyCoord as _SC_n  # noqa: PLC0415

        _gspc_zp["galactic_b_deg"] = round(float(_SC_n(float(target_row.get("ra_deg")) * _u_n.deg,
                                                        float(target_row.get("dec_deg")) * _u_n.deg).galactic.b.deg), 2)
    except Exception:  # noqa: BLE001
        _gspc_zp["galactic_b_deg"] = None
    _cbp_all = [float(v) for v in comp_bp_rp.values() if math.isfinite(float(v))]
    _gspc_zp["comp_bp_rp_median"] = round(float(np.median(_cbp_all)), 3) if _cbp_all else None
    _gspc_zp["comp_bp_rp_min"] = round(float(np.min(_cbp_all)), 3) if _cbp_all else None
    _gspc_zp["comp_bp_rp_max"] = round(float(np.max(_cbp_all)), 3) if _cbp_all else None
    return _gspc_shift, _gspc_zp


def _tgt_lc_flags(
    *,
    _calibrated,
    _cfg,
    _err_rel_for_snr,
    _raw_tf,
    airmass_arr,
    bjd,
    ct_ok,
    delta_mag,
    err,
    err_photon_export,
    mag_calib,
    mag_calib_ct,
    sat_flags,
    state,
    target_cid,
    target_frames,
    target_name,
):
    """Per-epoch flags: base flags (saturation, alignment, WCS, S/N) and the LC outlier tests with pixel evidence."""
    base_flags: list[str] = []
    # UNCAL-LC-2026-10: a frame is measured when the differential magnitude exists. Without a
    # band zero point (band_system NONE) mag_calib is NaN by design (export guard), but the
    # light curve itself (delta_mag = mag_calib - constant) is valid: the LC flags
    # (assign_lc_flags via apply_reporting_postprocess below), lc_rms and the check star use
    # it. (K2 is applied to delta_mag in that case; CT needs a band zero point.)
    _meas = np.asarray(mag_calib if _calibrated else delta_mag, dtype=np.float64)
    # FRAME-REJECT-2026-10: target SNR on the frame below cfg.snr_min -> 'low_snr' (not exported).
    _snr_min_f = float(getattr(_cfg, "snr_min", 10.0) or 10.0)
    _err_rel_arr = np.asarray(_err_rel_for_snr, dtype=np.float64).reshape(-1)
    for i in range(len(mag_calib)):
        _er = float(_err_rel_arr[i]) if i < _err_rel_arr.size else float("nan")
        if bool(sat_flags[i]):
            base_flags.append("saturated")
        elif math.isfinite(_er) and _er > 0 and (1.0 / _er) < _snr_min_f and math.isfinite(_meas[i]):
            base_flags.append("low_snr")
        elif i < len(_raw_tf) and str(_raw_tf.iloc[i]) in ("nondetection", "edge_fail"):
            base_flags.append(str(_raw_tf.iloc[i]))  # G08-05: the grid's edge verdict survives the LC flags
        elif math.isfinite(_meas[i]):  # UNCAL-LC: delta_mag when there is no band zero point
            base_flags.append("normal")
        else:
            base_flags.append("no_data")

    # Reporting path (Workstream B / LC-OUTLIER-01): see ``apply_reporting_postprocess``.
    from lc_outlier import EvidenceCache  # noqa: PLC0415

    _src_files_pre = target_frames["source_file"].astype(str).tolist()
    _aligned = getattr(state, "_aligned_dir_2a", None)
    _ev_cache = EvidenceCache(
        frames_dir=Path(_aligned) if _aligned else None,
        catalog_id=str(target_cid),
        proc_cache={},
    )
    # Normalize proc cache keys to basenames for EvidenceCache lookups.
    _proc_by_name: dict[str, pd.DataFrame] = {}
    for _k, _dfc in (getattr(state, "_aperture_csv_cache", None) or {}).items():
        _proc_by_name[Path(str(_k)).name] = _dfc
    _ev_cache.proc_cache = _proc_by_name

    def _evidence_at(_i: int):
        return _ev_cache.evidence_at(_i, _src_files_pre)

    mag_calib, mag_calib_ct, out_flags, out_flag_reasons = (
        apply_reporting_postprocess(
            mag_calib if _calibrated else _meas,
            mag_calib_ct if _calibrated else _meas,
            target_name=target_name,
            sat_flags=sat_flags,
            base_flags=base_flags,
            target_frames=target_frames,
            ct_ok=bool(ct_ok),
            err=err,
            bjd=bjd,
            source_files=_src_files_pre,
            frame_qc_reasons=getattr(state, "frame_qc_reasons", None),
            evidence_for_index=_evidence_at,
            err_photon=err_photon_export,
            airmass=airmass_arr,
        )
    )
    # G08-05: the upstream verdicts (saturated / edge_fail / nondetection / low_snr) enter assign_lc_flags as
    # base_flags (one flag merge); the S/N of a low_snr epoch is written into its reason
    if out_flag_reasons is None or len(out_flag_reasons) != len(out_flags):
        out_flag_reasons = [""] * len(out_flags)
    for _i, _fl in enumerate(out_flags):
        if _fl == "low_snr":
            _er_i = float(_err_rel_arr[_i]) if _i < _err_rel_arr.size else float("nan")
            out_flag_reasons[_i] = (f"low_snr: SNR {1.0 / _er_i:.1f} < {_snr_min_f:g}"
                                    if math.isfinite(_er_i) and _er_i > 0 else "low_snr")
    # BADPIX-REPAIR-2026-10: a repaired (ePSF-model) pixel inside the target aperture is a note, not a flag
    if "bp_repaired_px" in target_frames.columns:
        _bp = pd.to_numeric(target_frames["bp_repaired_px"], errors="coerce").fillna(0).astype(int).tolist()
        for _i, _n in enumerate(_bp[: len(out_flag_reasons)]):
            if _n > 0:
                _note = f"bad_pixel_repaired: {_n} px"
                out_flag_reasons[_i] = f"{out_flag_reasons[_i]}; {_note}" if out_flag_reasons[_i] else _note
    if not _calibrated:
        _nan_lc = np.full(len(_meas), np.nan, dtype=np.float64)
        mag_calib, mag_calib_ct = _nan_lc.copy(), _nan_lc.copy()
    return mag_calib, mag_calib_ct, out_flag_reasons, out_flags


def _tgt_check_star(
    *,
    _k_like_target,
    _calibrated,
    _chk_cid_pref,
    _ensemble_scatter_by_file,
    _scint_mag_arr,
    _sigma_sys_mag,
    all_frames,
    bjd,
    comp_catalog_mag,
    comp_ids,
    comp_lc,
    comp_quality,
    lc_dir,
    src_files,
    target_cid,
):
    """Check-star K series and its comp_quality JSON block (ensemble without the check star)."""
    _check_block: dict[str, Any] | None = None  # comp_quality JSON "check_star" (OUTPUT-REVIEW step 6)
    try:
        from check_star_kmag import (  # noqa: PLC0415
            build_comp_photon_mag_from_frames,
            check_kmag_sidecar_path,
            check_star_block,
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
                    n_comp_min=2,
                    comp_photon_mag=_comp_photon,
                    sigma_sys_mag=_sigma_sys_mag,
                    allow_relative=not _calibrated,
                )
                if _chk_result is not None and np.isfinite(_chk_result.kmag).any():
                    # B2 / G11-06 / G08-45 (Milan 2026-10-29): the check star is processed exactly like the
                    # target - the same k'' term and the same band colour term (AAVSO KMAG under TRANS=YES)
                    _k_corr, _k_note = _k_like_target(_chk_cid)
                    import dataclasses  # noqa: PLC0415

                    _chk_result = dataclasses.replace(
                        _chk_result, kmag=np.asarray(_chk_result.kmag, dtype=np.float64) + _k_corr)
                    # CHECK-RULES-2026-10: per-epoch K error = photon (check) + ensemble + sigma_sys,
                    # the same budget as the target, for the chi2 consistency test.
                    # CHECK-SCINT-2026-10 (code review 2 D10 / G08-10, Milan 2026-10-28 with O1): plus the
                    # scintillation of the epoch, as in the target's error.
                    # G06-39: the target's own error helper (one budget: matched NaN scatter = no ensemble
                    # term, unmatched epoch = NaN), so K and target errors agree epoch by epoch
                    _cf = all_frames[all_frames["catalog_id"].astype(str).map(_normalize_gaia_id) == str(_chk_cid)]
                    _err_by_src = dict(zip(_cf["source_file"].astype(str),
                                           pd.to_numeric(_cf["err"], errors="coerce").to_numpy(dtype=np.float64),
                                           strict=False))
                    _kerr_rel, _ = _combine_err_with_ensemble_scatter_keyed(
                        np.array([_err_by_src.get(str(_sf), np.nan) for _sf in src_files], dtype=np.float64),
                        src_files,
                        _ensemble_scatter_by_file,
                        sigma_sys_mag=_sigma_sys_mag,
                        sigma_scint_mag=_scint_mag_arr,
                        target_name=f"{target_cid} check {_chk_cid}",
                    )
                    _kerr = MAG_ERR_SCALE * np.asarray(_kerr_rel, dtype=np.float64)
                    save_check_kmag_sidecar(
                        check_kmag_sidecar_path(lc_dir, target_cid),
                        bjd=bjd,
                        source_files=src_files,
                        kmag=_chk_result.kmag,
                        kmag_err=_kerr,
                    )
                    _check_block = check_star_block(_chk_cid, _chk_result)
                    if isinstance(_check_block, dict):
                        _check_block["corrections"] = _k_note
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
    return _check_block,


def _tgt_write_outputs(
    *,
    _band_system,
    _broeg_w,
    _calibrated,
    _cfg,
    _check_block,
    _comp_index,
    _gspc_shift,
    _gspc_zp,
    _save_png,
    _zp_info,
    airmass_arr,
    all_frames,
    ap_arr,
    bjd,
    comp_catalog_mag,
    comp_ids,
    comp_lc,
    comp_quality,
    comp_rms_map,
    delta_mag,
    err,
    err_photon_export,
    err_scatter_unmatched_arr,
    err_scint_rel_export,
    err_sem_rel_export,
    err_sigma_sys_rel_export,
    hjd,
    jd,
    k2_coeff_lc,
    lc_dir,
    mag_calib,
    mag_calib_ct,
    out_flag_reasons,
    out_flags,
    src_files,
    state,
    target_cid,
    target_lc,
    target_name,
    target_row,
    time_base,
):
    """Write the light curve CSV, comp_quality JSON, LC PNG and the target field map."""
    lc_csv = lc_dir / f"lightcurve_{target_cid}.csv"
    # I-04: exclude epochs with unmatched ensemble scatter from LC export.
    if err_scatter_unmatched_arr is not None and np.any(err_scatter_unmatched_arr):
        _keep_lc = ~np.asarray(err_scatter_unmatched_arr, dtype=bool)
        (
            _mask_kept,  # the helper returns the keep-mask first
            bjd,
            hjd,
            jd,
            airmass_arr,
            target_lc,
            mag_calib,
            mag_calib_ct,
            delta_mag,
            err,
            ap_arr,
            out_flags,
            src_files,
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
            target_lc,
            mag_calib,
            mag_calib_ct,
            delta_mag,
            err,
            ap_arr,
            out_flags,
            src_files,
            err_photon_export,
            err_sem_rel_export,
            err_scint_rel_export,
            err_sigma_sys_rel_export,
        )
        if out_flag_reasons is not None and len(out_flag_reasons) == len(_keep_lc):
            out_flag_reasons = [out_flag_reasons[i] for i in range(len(_keep_lc)) if _keep_lc[i]]
    # GSPC-ZP-2026-10: second scale on the final arrays (same frames as the CSV).
    if math.isfinite(_gspc_shift):
        _base_g = np.asarray(mag_calib if _calibrated else delta_mag, dtype=np.float64)
        _mag_gspc = _base_g + float(_gspc_shift)
        _fin_g = _mag_gspc[np.isfinite(_mag_gspc)]
        _gspc_zp["median_mag"] = round(float(np.median(_fin_g)), 4) if _fin_g.size else None
    else:
        _mag_gspc = None
        _gspc_zp["median_mag"] = None
    # OUTPUT-REVIEW-2026-10 step 6: mag_calib_ct is written when the setup applies a colour term
    # (equal to mag_calib for a target outside the fitted colour range; ct_ok in the summary).
    save_lightcurve_csv(
        lc_csv,
        bjd=bjd,
        hjd=hjd,
        jd=jd,
        airmass=airmass_arr,
        mag_inst=target_lc,
        mag_calib=mag_calib,
        mag_calib_ct=(np.asarray(mag_calib_ct, dtype=np.float64) if bool(state.apply_color_term) else None),
        mag_gspc=_mag_gspc,
        delta_mag=delta_mag,
        err=err,
        aperture_r_px=ap_arr,
        flags=out_flags,
        flag_reasons=out_flag_reasons,
        source_files=src_files,
        time_base=time_base,
        err_photon=err_photon_export,
        err_sem_rel=err_sem_rel_export,
        err_scint_rel=err_scint_rel_export,
        err_sigma_sys_rel=err_sigma_sys_rel_export,
        k2_coeff=k2_coeff_lc,
    )
    # EPSF-LC-LOG-01 / INV-PSF-SUBMIT-01: PSF LC files are an internal diagnostic
    # product written by psf_internal_lc (RUN ePSF path), not aperture photometry.

    # COMP-ASSIGN-01 D4: stability AFTER photometry - verdict only (membership unchanged).
    comp_bjd = {cid: _get_comp_bjd_series(cid, all_frames) for cid in comp_ids}
    comp_quality = check_comparison_stability(
        comp_lc,
        comp_rms_map=comp_rms_map,
        comp_bjd=comp_bjd,
        max_comp_p2p=float(_cfg.phase01_comparison_max_comp_rms),
        max_comp_slope_mmag_hr=float(_cfg.comp_max_slope_mmag_hr),
        common_mode_detrend=True,
        stability_run_flags=state.stability_run_flags,
    )

    # Comp quality for the UI (comparison-stars table)
    _cq_path = lc_dir / f"comp_quality_{target_cid}.json"
    try:
        _cq_payload: dict[str, Any] = {}
        for cid, info in comp_quality.items():
            nk = _normalize_gaia_id(cid)
            q = str(info.get("quality", "") or "").strip()
            note = str(info.get("note", "") or "").strip()
            if q == "good" and not note:
                _cq_payload[nk] = "good"
            else:
                _cq_payload[nk] = {"quality": q, "note": note}
        _sm = (getattr(state, "smart_log", None) or {}).get(str(target_cid))
        if _sm:
            _cq_payload["comp_smart"] = {k: _sm.get(k) for k in ("replaced", "added", "kept_suspect", "n_clean")}
            # COMP-CATOFF-2026-10: candidates excluded for their catalogue offset (mag)
            _co = _sm.get("catalog_offset") or {}
            if _co.get("excluded"):
                _cq_payload["comp_smart"]["catalog_offset"] = {
                    "max_mag": _co.get("max_mag"),
                    "excluded": {c: (_co.get("offsets") or {}).get(c) for c in _co["excluded"]},
                }
        if _check_block:
            _cq_payload["check_star"] = _check_block
        # EXPORT-2026-10: the ensemble as actually used - for the exports and the PDF.
        _bw_sum = float(sum(v for v in _broeg_w.values() if math.isfinite(v)))
        if _bw_sum > 0:
            _cq_payload["broeg_weights"] = {
                _normalize_gaia_id(k): round(float(v) / _bw_sum, 6)
                for k, v in _broeg_w.items()
                if math.isfinite(v)
            }
        _cq_payload["comp_band_mag"] = {
            _normalize_gaia_id(k): (round(float(v), 4) if math.isfinite(float(v)) else None)
            for k, v in comp_catalog_mag.items()
        }
        _cq_payload["band_system"] = str(_band_system)
        _cq_payload["zero_point"] = _zp_info
        _cq_payload["gspc_zero_point"] = _gspc_zp
        _cq_path.write_text(json.dumps(_cq_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[APERTURE] Optional artifact write failed (comp_quality.json): %s", exc)

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
                delta_mag_mode=not _calibrated,
                delta_mag=delta_mag,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("[APERTURE] Optional artifact write failed (lightcurve PNG): %s", exc)

    # Per-target field map with numbered comp stars - always (UI)
    try:
        if state.target_field_map is not None:
            from field_map import target_field_map_path  # noqa: PLC0415

            state.target_field_map.save(target_field_map_path(lc_dir, target_cid), target_row,
                                        _comp_index.get(target_cid, pd.DataFrame()))
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[APERTURE] Optional artifact write failed (field map PNG): %s", exc)
    return bjd, comp_quality, delta_mag, lc_csv, lc_png, mag_calib, out_flags


def _tgt_summary_row(
    *,
    _band_system,
    _calibrated,
    _dilution_result,
    _measured_ap_target,
    _pt_active,
    apertures_px,
    bjd,
    bp_rp_comp_med,
    c1,
    c1_stderr,
    comp_quality,
    ct_corr,
    ct_mode,
    ct_n_comp,
    ct_ok,
    delta_mag,
    k2_colour_ref,
    k2_source_target,
    k2_value_lc,
    lc_csv,
    lc_png,
    mag_calib,
    n_alignment_failed,
    n_lc,
    n_wcs_untrusted,
    out_flags,
    state,
    summary_rows,
    target_bp_rp,
    target_cid,
    target_comps,
    target_frames,
    target_name,
    target_row,
    target_vsx_type,
):
    """Summary row of the target (photometry_summary.csv)."""
    finite_calib = mag_calib[np.isfinite(mag_calib)]
    _meas_lc = np.asarray(mag_calib if _calibrated else delta_mag, dtype=np.float64)
    finite_meas = _meas_lc[np.isfinite(_meas_lc)]
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
    # FIX-APREPORT-2026-10: per-target aperture (r = f* x FWHM_frame, stamped per frame by
    # apply_grid_fluxes_to_frames) is the radius the LC was measured with; the proc-cache
    # radius is the global one from comparison-star selection (draft 525: PDF said 7.2 px, LC used ~5.0 px).
    if _pt_active:
        _measured_ap = float("nan")
    if not math.isfinite(_measured_ap) and not target_frames.empty and "aperture_r_px" in target_frames.columns:
        _ap_meas = pd.to_numeric(target_frames["aperture_r_px"], errors="coerce").dropna()
        if not _ap_meas.empty:
            _measured_ap = float(np.median(_ap_meas.to_numpy(dtype=float)))
    _lc_rms_full = float(np.std(finite_meas)) if len(finite_meas) > 1 else float("nan")
    _lc_rms_ooe = compute_lc_rms_ooe(_meas_lc, out_flags)

    _n_tier12 = 0
    if not target_comps.empty:
        if "comp_tier" in target_comps.columns:
            _tiers = pd.to_numeric(target_comps["comp_tier"], errors="coerce")
            _n_tier12 = int(_tiers.isin([1, 2]).sum())

    _sum_row: dict[str, Any] = {
        "catalog_id": target_cid,
        "name": target_name,
        "vsx_name": _vsx_name_of(target_row),
        "vsx_type": target_vsx_type,
        "zone_flag": str(target_row.get("zone_flag", "")).strip(),
        "n_frames": len(bjd),
        "n_good_comp": n_good_comp,
        "n_tier12": _n_tier12,
        "n_stability_good": n_stability_good,
        "n_stability_suspect": n_stability_suspect,
        "n_saturated": n_sat,
        "n_alignment_failed": n_alignment_failed,
        "n_wcs_untrusted": n_wcs_untrusted,
        "lc_rms": _lc_rms_full,
        "lc_rms_ooe": _lc_rms_ooe,
        "lc_median_mag": float(np.median(finite_calib)) if len(finite_calib) > 0 else float("nan"),
        "aperture_px": _measured_ap if math.isfinite(_measured_ap) else float(apertures_px.get(target_cid, float("nan"))),
        "target_blend_frac": float((getattr(state, "target_blend", None) or {}).get(str(target_cid), float("nan"))),
        "psf_ratio": float((getattr(state, "psf_ratio", None) or {}).get(str(target_cid), float("nan"))),
        "psf_scale": float((getattr(state, "psf_scale", None) or {}).get(str(target_cid), 1.0)),
        "dilution_factor": float(_dilution_result.get("dilution_factor", 1.0)),
        "dilution_delta_mag": float(_dilution_result.get("dilution_delta_mag", 0.0)),
        # read by photometry_shared GS11 aggregate (pipeline_meta) when gs11_dilution_enabled
        "gs11_aperture_arcsec": float(_dilution_result.get("aperture_arcsec", float("nan"))),
        "gs11_dilution_skipped": bool(_dilution_result.get("dilution_skipped", False)),
        "gs11_dilution_skip_reason": str(_dilution_result.get("dilution_skip_reason", "") or ""),
        "lc_csv": str(lc_csv),
        "lc_png": str(lc_png),
        "ct_ok": bool(ct_ok),
        "ct_corr": float(ct_corr) if bool(ct_ok) and math.isfinite(float(ct_corr)) else float("nan"),
        "ct_c1": float(c1) if bool(ct_ok) and math.isfinite(float(c1)) else float("nan"),
        "ct_c1_stderr": float(c1_stderr) if bool(ct_ok) and math.isfinite(float(c1_stderr)) else float("nan"),
        "ct_mode": str(ct_mode) if bool(ct_ok) else "",
        "ct_n_comp": int(ct_n_comp) if bool(ct_ok) else 0,
        "ct_bp_rp_target": float(target_bp_rp) if bool(ct_ok) else float("nan"),
        "ct_bp_rp_comp_med": float(bp_rp_comp_med) if bool(ct_ok) else float("nan"),
        "k2_source": k2_source_target,
        "k2_value": float(k2_value_lc) if math.isfinite(float(k2_value_lc)) else float("nan"),
        "k2_colour_ref": float(k2_colour_ref) if math.isfinite(float(k2_colour_ref)) else float("nan"),
        "band_system": str(_band_system),
    }
    _sum_row["skip_reason"] = str(target_row.get("skip_reason", "") or "")
    summary_rows.append(_sum_row)
    n_lc += 1
    lc_rms = float(summary_rows[-1]["lc_rms"])
    lc_rms_ooe = float(summary_rows[-1].get("lc_rms_ooe", float("nan")))
    r_ap = float(summary_rows[-1]["aperture_px"])
    logging.info(
        f"[APERTURE] {target_name}: "
        f"lc_rms={lc_rms:.4f}, lc_rms_ooe={lc_rms_ooe:.4f}, "
        f"n_comp={n_good_comp} (stability_good={n_stability_good}), "
        f"aperture={r_ap:.2f}px (measured)"
    )
    return n_lc,


def _aperture_process_one_target(
    target_row: Any,
    *,
    state: _AperturePhotometryState,
    summary_rows: list,
    n_lc: int,
    lc_dir: Path,
    output_dir: Path,
    progress_cb: Any,
    annulus_inner_fwhm: float,
    annulus_outer_fwhm: float,
    _apt_fw: float,
    _save_png: bool,
) -> tuple[list, int]:
    """Process one target through the full aperture photometry pipeline.

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
    _aperture_csv_cache = state._aperture_csv_cache
    _aperture_lookup_cache = state._aperture_lookup_cache
    frame_time_lookup = state.frame_time_lookup
    fwhm_px = state.fwhm_px
    apertures_px = state.apertures_px
    chip_fw = state.chip_fw
    chip_fh = state.chip_fh
    _flux_matrix = state._flux_matrix
    obs_group = state.obs_group
    _gain_phot = state._gain_phot
    _rn_phot = state._rn_phot
    sat_limit_resolved = state.sat_limit_resolved
    _cfg = state._cfg
    _nt = state._nt

    target_cid = _normalize_gaia_id(target_row.get("catalog_id", ""))
    target_name = _target_display_name(target_row, fallback_cid=target_cid)
    target_vsx_type = str(target_row.get("vsx_type", "") or "").strip()
    _sp = target_row.get("skip_photometry", False)
    if isinstance(_sp, (bool, np.bool_)):
        skip_photo = bool(_sp)
    else:
        skip_photo = str(_sp).strip().lower() in ("1", "true", "yes", "t")
    _zf_row = str(target_row.get("zone_flag", "")).strip()
    # skip_photometry / skip_reason carry the whole decision (target-selection cuts + per-frame saturation)
    if not skip_photo:
        state.i_measured = int(getattr(state, "i_measured", 0)) + 1
        _nm = int(getattr(state, "n_to_measure", 0)) or _nt
        _im = int(state.i_measured)
        if progress_cb is not None and (_im == 1 or _im >= _nm or (_nm > 1 and _im % max(1, _nm // 12) == 0)):
            _ns = int(getattr(state, "n_final_skip", 0))
            _p2(f"Aperture photometry: target {_im}/{_nm}: {target_name[:50]}"
                + (f" ({_ns} skipped: saturation, S/N, type)" if _ns else ""))
    if skip_photo:
        _skip_reason = str(target_row.get("skip_reason", "") or "").strip()
        logging.info(f"[APERTURE] Skipping photometry ({_skip_reason}): {target_name}")
        _skip_sum: dict[str, Any] = {
            "catalog_id": target_cid,
            "name": target_name,
            "vsx_name": _vsx_name_of(target_row),
            "vsx_type": target_vsx_type,
            "zone_flag": _zf_row,
            "n_frames": 0,
            "n_good_comp": 0,
            "n_saturated": 0,
            "lc_rms": float("nan"),
            "lc_median_mag": float("nan"),
            "aperture_px": float("nan"),
            "lc_csv": "",
            "lc_png": "",
        }
        _skip_sum["skip_reason"] = _skip_reason
        summary_rows.append(_skip_sum)
        return summary_rows, n_lc
    logging.info(
        f"[APERTURE] Starting: target={target_name}, "
        f"frames={len(csv_files)}, "
        f"aperture={_apt_fw * float(fwhm_px):.2f}px "
        f"(FWHM={float(fwhm_px):.3f}px x {_apt_fw:.2f})"
    )

    # Comp stars for this target
    target_comps = _comp_index.get(target_cid, pd.DataFrame()).copy()

    if target_comps.empty:
        summary_rows = _aperture_skip_empty_comps_target(
            target_cid=target_cid,
            target_name=target_name,
            zone_flag=_zf_row,
            summary_rows=summary_rows,
            vsx_type=target_vsx_type,
        )
        return summary_rows, n_lc

    _band_system, _fb, all_ids, comp_catalog_mag, comp_ids, comp_rms_map = _tgt_comp_inputs(
        obs_group=obs_group,
        target_cid=target_cid,
        target_comps=target_comps,
    )


    _chk_cid_pref, = _tgt_choose_check_star(
        _cfg=_cfg,
        _fb=_fb,
        all_ids=all_ids,
        comp_catalog_mag=comp_catalog_mag,
        comp_ids=comp_ids,
        state=state,
        target_cid=target_cid,
        target_comps=target_comps,
        target_row=target_row,
    )

    # Step 2: photometry per frame (PERF-8: slice shared flux matrix when built)
    chip_fh, chip_fw, frame_results = _tgt_frame_rows(
        _aperture_csv_cache=_aperture_csv_cache,
        _aperture_lookup_cache=_aperture_lookup_cache,
        _flux_matrix=_flux_matrix,
        _gain_phot=_gain_phot,
        _rn_phot=_rn_phot,
        all_ids=all_ids,
        apertures_px=apertures_px,
        chip_fh=chip_fh,
        chip_fw=chip_fw,
        csv_files=csv_files,
        frame_time_lookup=frame_time_lookup,
        sat_limit_resolved=sat_limit_resolved,
        state=state,
    )

    if not frame_results:
        return summary_rows, n_lc

    all_frames, = _tgt_apply_night_k(
        _chk_cid_pref=_chk_cid_pref,
        annulus_inner_fwhm=annulus_inner_fwhm,
        annulus_outer_fwhm=annulus_outer_fwhm,
        comp_ids=comp_ids,
        frame_results=frame_results,
        state=state,
        target_cid=target_cid,
    )
    # Build time series per star
    _band_system, _broeg_w, _calibrated, _comp_levels, _dilution_result, _ensemble_scatter_by_file, _zp_info, comp_lc, comp_quality, comp_weight_map, delta_mag, mag_calib, target_lc = _tgt_ensemble(
        _band_system=_band_system,
        _cfg=_cfg,
        _fb=_fb,
        all_frames=all_frames,
        comp_catalog_mag=comp_catalog_mag,
        comp_ids=comp_ids,
        target_cid=target_cid,
    )
    _dilution_result, mag_calib = _tgt_dilution(
        _cfg=_cfg,
        _dilution_result=_dilution_result,
        apertures_px=apertures_px,
        fwhm_px=fwhm_px,
        mag_calib=mag_calib,
        obs_group=obs_group,
        state=state,
        target_bp_rp_by_cid=target_bp_rp_by_cid,
        target_cid=target_cid,
        target_row=target_row,
    )

    # -- Color term (BP-RP) - global comp-pool fit; toggle controls correction only --
    _target_airmass_arr, comp_bp_rp, delta_mag, k2_coeff_lc, k2_colour_ref, k2_source_target, k2_value_lc, mag_calib, target_bp_rp = _tgt_k2(
        _calibrated=_calibrated,
        _cfg=_cfg,
        all_frames=all_frames,
        comp_quality=comp_quality,
        comp_weight_map=comp_weight_map,
        delta_mag=delta_mag,
        mag_calib=mag_calib,
        state=state,
        target_bp_rp_by_cid=target_bp_rp_by_cid,
        target_cid=target_cid,
        target_comps=target_comps,
        target_row=target_row,
    )

    _group_ct, bp_rp_comp_med, c1, c1_stderr, ct_corr, ct_mode, ct_n_comp, ct_ok, mag_calib_ct = _tgt_colour_term(
        _band_system=_band_system,
        comp_bp_rp=comp_bp_rp,
        comp_catalog_mag=comp_catalog_mag,
        comp_quality=comp_quality,
        comp_weight_map=comp_weight_map,
        mag_calib=mag_calib,
        state=state,
        target_bp_rp=target_bp_rp,
        target_cid=target_cid,
        target_name=target_name,
    )

    # Target time values in the row order of ``_get_lc`` / ``_ensemble_scatter_by_source_file``
    # (``frame_rows_in_time_order``, G08-07).
    _err_rel_for_snr, _measured_ap_target, _pt_active, _raw_tf, _scint_mag_arr, _sigma_sys_mag, airmass_arr, ap_arr, bjd, err, err_photon_export, err_scatter_unmatched_arr, err_scint_rel_export, err_sem_rel_export, err_sigma_sys_rel_export, hjd, jd, n_alignment_failed, n_wcs_untrusted, sat_flags, src_files, target_frames, time_base = _tgt_time_and_errors(
        _cfg=_cfg,
        _ensemble_scatter_by_file=_ensemble_scatter_by_file,
        _target_airmass_arr=_target_airmass_arr,
        all_frames=all_frames,
        bp_rp_comp_med=bp_rp_comp_med,
        c1_stderr=c1_stderr,
        ct_corr=ct_corr,
        ct_ok=ct_ok,
        mag_calib=mag_calib,
        output_dir=output_dir,
        state=state,
        target_bp_rp=target_bp_rp,
        target_cid=target_cid,
        target_name=target_name,
        target_row=target_row,
    )
    # GSPC-ZP-2026-10: second zero point from Gaia DR3 synthetic photometry (GSPC) with the
    # same estimator and weights as the standard one. mag_gspc = standard light curve shifted by
    # (C_GSPC - C_std); for an uncalibrated light curve mag_gspc = delta_mag + level + C_GSPC.
    # Notes: "A" = no standard zero point (band NONE); "B" = |C_GSPC - C_std| >= 0.02 mag;
    # "B-heuristic" = GSPC not available, red comps (median BP-RP > 1.5) in the Galactic plane
    # (|b| < 10 deg).
    _gspc_shift, _gspc_zp = _tgt_gspc_zero_point(
        _broeg_w=_broeg_w,
        _calibrated=_calibrated,
        _comp_levels=_comp_levels,
        _fb=_fb,
        _group_ct=_group_ct,
        comp_bp_rp=comp_bp_rp,
        comp_catalog_mag=comp_catalog_mag,
        comp_weight_map=comp_weight_map,
        ct_corr=ct_corr,
        ct_ok=ct_ok,
        state=state,
        target_bp_rp=target_bp_rp,
        target_cid=target_cid,
        target_row=target_row,
    )

    mag_calib, mag_calib_ct, out_flag_reasons, out_flags = _tgt_lc_flags(
        _calibrated=_calibrated,
        _cfg=_cfg,
        _err_rel_for_snr=_err_rel_for_snr,
        _raw_tf=_raw_tf,
        airmass_arr=airmass_arr,
        bjd=bjd,
        ct_ok=ct_ok,
        delta_mag=delta_mag,
        err=err,
        err_photon_export=err_photon_export,
        mag_calib=mag_calib,
        mag_calib_ct=mag_calib_ct,
        sat_flags=sat_flags,
        state=state,
        target_cid=target_cid,
        target_frames=target_frames,
        target_name=target_name,
    )

    def _k_like_target(chk_cid: str) -> tuple[np.ndarray, str]:
        """Per-epoch k'' + colour-term corrections of the check star, as applied to the target."""
        _n = len(src_files)
        _corr = np.zeros(_n, dtype=np.float64)
        _row = state.masterstars_df.loc[state.masterstars_df["catalog_id"].astype(str).map(_normalize_gaia_id) == chk_cid]
        _c = float(pd.to_numeric(_row["bp_rp"], errors="coerce").iloc[0]) if len(_row) and "bp_rp" in _row else float("nan")
        _notes = []
        if k2_source_target != "none" and math.isfinite(k2_value_lc) and math.isfinite(k2_colour_ref) and math.isfinite(_c):
            _am = np.asarray(_target_airmass_arr, dtype=np.float64)[:_n]
            _corr[: len(_am)] -= np.where(np.isfinite(_am), float(k2_value_lc) * (_c - float(k2_colour_ref)) * _am, 0.0)
            _notes.append("k2")
        if ct_ok and state.group_color_term is not None:
            _cc, _ = _band_ct_correction(getattr(state.group_color_term, "poly_fit", None), _c, comp_bp_rp,
                                         comp_catalog_mag, comp_weight_map)
            if math.isfinite(_cc):
                _corr += _cc
                _notes.append("colour term")
            else:
                _notes.append("colour term not applicable (check colour outside the fitted range)")
        return _corr, ", ".join(_notes) if _notes else "none"

    _check_block, = _tgt_check_star(
        _k_like_target=_k_like_target,
        _calibrated=_calibrated,
        _chk_cid_pref=_chk_cid_pref,
        _ensemble_scatter_by_file=_ensemble_scatter_by_file,
        _scint_mag_arr=_scint_mag_arr,
        _sigma_sys_mag=_sigma_sys_mag,
        all_frames=all_frames,
        bjd=bjd,
        comp_catalog_mag=comp_catalog_mag,
        comp_ids=comp_ids,
        comp_lc=comp_lc,
        comp_quality=comp_quality,
        lc_dir=lc_dir,
        src_files=src_files,
        target_cid=target_cid,
    )

    # Step 6: save outputs
    bjd, comp_quality, delta_mag, lc_csv, lc_png, mag_calib, out_flags = _tgt_write_outputs(
        _band_system=_band_system,
        _broeg_w=_broeg_w,
        _calibrated=_calibrated,
        _cfg=_cfg,
        _check_block=_check_block,
        _comp_index=_comp_index,
        _gspc_shift=_gspc_shift,
        _gspc_zp=_gspc_zp,
        _save_png=_save_png,
        _zp_info=_zp_info,
        airmass_arr=airmass_arr,
        all_frames=all_frames,
        ap_arr=ap_arr,
        bjd=bjd,
        comp_catalog_mag=comp_catalog_mag,
        comp_ids=comp_ids,
        comp_lc=comp_lc,
        comp_quality=comp_quality,
        comp_rms_map=comp_rms_map,
        delta_mag=delta_mag,
        err=err,
        err_photon_export=err_photon_export,
        err_scatter_unmatched_arr=err_scatter_unmatched_arr,
        err_scint_rel_export=err_scint_rel_export,
        err_sem_rel_export=err_sem_rel_export,
        err_sigma_sys_rel_export=err_sigma_sys_rel_export,
        hjd=hjd,
        jd=jd,
        k2_coeff_lc=k2_coeff_lc,
        lc_dir=lc_dir,
        mag_calib=mag_calib,
        mag_calib_ct=mag_calib_ct,
        out_flag_reasons=out_flag_reasons,
        out_flags=out_flags,
        src_files=src_files,
        state=state,
        target_cid=target_cid,
        target_lc=target_lc,
        target_name=target_name,
        target_row=target_row,
        time_base=time_base,
    )

    # Summary row
    n_lc, = _tgt_summary_row(
        _band_system=_band_system,
        _calibrated=_calibrated,
        _dilution_result=_dilution_result,
        _measured_ap_target=_measured_ap_target,
        _pt_active=_pt_active,
        apertures_px=apertures_px,
        bjd=bjd,
        bp_rp_comp_med=bp_rp_comp_med,
        c1=c1,
        c1_stderr=c1_stderr,
        comp_quality=comp_quality,
        ct_corr=ct_corr,
        ct_mode=ct_mode,
        ct_n_comp=ct_n_comp,
        ct_ok=ct_ok,
        delta_mag=delta_mag,
        k2_colour_ref=k2_colour_ref,
        k2_source_target=k2_source_target,
        k2_value_lc=k2_value_lc,
        lc_csv=lc_csv,
        lc_png=lc_png,
        mag_calib=mag_calib,
        n_alignment_failed=n_alignment_failed,
        n_lc=n_lc,
        n_wcs_untrusted=n_wcs_untrusted,
        out_flags=out_flags,
        state=state,
        summary_rows=summary_rows,
        target_bp_rp=target_bp_rp,
        target_cid=target_cid,
        target_comps=target_comps,
        target_frames=target_frames,
        target_name=target_name,
        target_row=target_row,
        target_vsx_type=target_vsx_type,
    )


    state.chip_fw = chip_fw
    state.chip_fh = chip_fh
    return summary_rows, n_lc
