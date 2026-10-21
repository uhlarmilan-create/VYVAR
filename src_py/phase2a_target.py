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

from photometry_phase2a import (
    _Phase2AState,
    _draft_dir_from_phase2a_paths,
    _measured_aperture_from_proc_cache,
    _resolve_photometric_aperture_px_for_gs11,
    apply_reporting_postprocess,
    fit_color_term_c1,
    read_flux_from_csv,
)
from photometry_lightcurve import (
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
    _phase2a_skip_empty_comps_target,
    _recompute_bjd_hjd_with_status,
    apply_color_term,
    check_comparison_stability,
    compute_lc_rms_ooe,
    ct_ensemble_reference_maps,
    ensemble_normalize,
    save_lightcurve_csv,
    save_lightcurve_png,
)

from photometry_core import (
    LOGGER,
)



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
    # skip_photometry / skip_reason carry the whole decision (Phase 0 cuts + per-frame saturation)
    if not skip_photo:
        state.i_measured = int(getattr(state, "i_measured", 0)) + 1
        _nm = int(getattr(state, "n_to_measure", 0)) or _nt
        _im = int(state.i_measured)
        if progress_cb is not None and (_im == 1 or _im >= _nm or (_nm > 1 and _im % max(1, _nm // 12) == 0)):
            _ns = int(getattr(state, "n_final_skip", 0))
            _p2(f"Faza 2A: ciel {_im}/{_nm}: {target_name[:50]}"
                + (f" ({_ns} preskocenych: saturacia, S/N, typ)" if _ns else ""))
    if skip_photo:
        _skip_reason = str(target_row.get("skip_reason", "") or "").strip()
        logging.info(f"[FAZA 2A] Preskakujem fotometriu ({_skip_reason}): {target_name}")
        _skip_sum: dict[str, Any] = {
            "catalog_id": target_cid,
            "vsx_name": target_name,
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
            vsx_type=target_vsx_type,
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
        logging.warning(
            "[BAND] %s: no comp with a valid %s magnitude - LC not calibrated, not exportable",
            target_cid, _fb.label,
        )
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
                    # PSF-LOCAL-2026-10: the target's annulus scales with its own PSF too
                    r_out_t *= float((getattr(state, "psf_scale", None) or {}).get(str(target_cid), 1.0))
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
                variable_target_catalog_ids=state.variable_target_catalog_ids,
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
                        # PSF-LOCAL-2026-10: the target's annulus scales with its own PSF too
                        r_out_t *= float((getattr(state, "psf_scale", None) or {}).get(str(target_cid), 1.0))
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

    all_frames = pd.concat(frame_results, ignore_index=True)

    # APERTURE-PERTARGET-01: rebuild mag_inst at f* for target + its comps (same f).
    _pt_choice = (getattr(state, "per_target_aperture", None) or {}).get(str(target_cid))
    _pt_grid = getattr(state, "aperture_grid_night", None)
    if _pt_choice is not None and _pt_grid is not None:
        try:
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
        except Exception as _pt_apply_exc:  # noqa: BLE001
            logging.warning("[APERTURE-DYNAMIC] apply failed for %s: %s", target_cid, _pt_apply_exc)

    # Zostav casove rady per hviezda
    target_lc = _get_lc(target_cid, all_frames)
    comp_lc = {cid: _get_lc(cid, all_frames) for cid in comp_ids}

    # COMP-ASSIGN-01 D4/D5: membership is fixed from Phase 1 (3-8). Stability is a
    # post-photometry verdict only - do not let it re-select before ensemble.
    comp_quality = {cid: {"quality": "good"} for cid in comp_ids}

    # Krok 4: Ensemble normalizacia (consumes the delivered set as given)
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
        logging.warning("[BAND] %s: no measured comp with a valid %s magnitude - LC not calibrated",
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
                mag_limit_delta=float(_cfg.gs11_dilution_mag_limit_delta),
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

    from k2_extinction import K2Source, apply_k2_per_frame, bp_rp_comp_median  # noqa: PLC0415

    k2_value_lc = float("nan")
    k2_colour_ref = float("nan")
    k2_source_rows = [K2Source.NONE.value] * len(mag_calib)
    _k2_val = float(getattr(state, "k2_bprp", float("nan")))
    _k2_src = str(getattr(state, "k2_source", K2Source.NONE.value))
    if _k2_src == K2Source.LITERATURE_DEFAULT.value and math.isfinite(_k2_val):
        _airmass_k2 = _target_airmass_arr
        _bp_med_k2 = bp_rp_comp_median(comp_bp_rp, comp_quality)
        _k2_src_enum = K2Source.LITERATURE_DEFAULT
        # UNCAL-LC-2026-10: k'' x airmass x (target - comp colour) shapes the light curve; it
        # does not need a zero point, so an uncalibrated light curve (delta_mag) gets it too.
        _k2_in = mag_calib if _calibrated else delta_mag
        _k2_out, _k2_delta, k2_source_rows = apply_k2_per_frame(
            _k2_in,
            _airmass_k2,
            object_bp_rp=float(target_bp_rp),
            bp_rp_comp_med=_bp_med_k2,
            k2_value=_k2_val,
            k2_source=_k2_src_enum,
        )
        if _calibrated:
            mag_calib = _k2_out
        else:
            delta_mag = _k2_out
        k2_value_lc = _k2_val
        k2_colour_ref = _bp_med_k2
    # summary: the k'' source of the target (the tag when any epoch got the correction)
    k2_source_target = next((str(v) for v in k2_source_rows if str(v) != K2Source.NONE.value), K2Source.NONE.value)

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
            _cc, _ww = [], []
            for _cid_ct, _bp_ct in comp_bp_rp.items():
                _w = float(comp_weight_map.get(_cid_ct, float("nan")))
                if math.isfinite(float(comp_catalog_mag.get(_cid_ct, float("nan")))) and math.isfinite(_w) and _w > 0:
                    _cc.append(float(_bp_ct))
                    _ww.append(_w)
            if _pf is None:
                ct_corr, ct_ok = 0.0, True
                bp_rp_comp_med = float(np.median(_cc)) if _cc else float("nan")
            elif math.isfinite(float(target_bp_rp)) and _pf.c_min <= float(target_bp_rp) <= _pf.c_max and _cc:
                ct_corr = float(_pf.correction(float(target_bp_rp), _cc, _ww))
                ct_ok = math.isfinite(ct_corr)
                c1 = float(_pf.coeffs[0])
                bp_rp_comp_med = float(np.sum(np.asarray(_ww) * np.asarray(_cc)) / np.sum(_ww))
            else:
                logging.info("[COLOR TERM] %s: target colour outside the fitted range -> no colour term", target_cid)
                ct_corr, ct_ok = 0.0, False
            if not math.isfinite(ct_corr):
                ct_corr, ct_ok = 0.0, False
            mag_calib_ct = mag_calib + float(ct_corr)
    elif state.apply_color_term and _group_ct is not None and _group_ct.apply_gate:
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
        if _ct_in_range:
            mag_calib_ct, ct_corr, bp_rp_comp_med = apply_color_term(
                mag_calib,
                target_bp_rp,
                _ref_bp,
                _ref_q,
                c1,
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
    _pt_active = bool((getattr(state, "per_target_aperture", None) or {}).get(str(target_cid)))
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
    # GSPC-ZP-2026-10: second zero point from Gaia DR3 synthetic photometry (GSPC) with the
    # same estimator and weights as the standard one. mag_gspc = standard light curve shifted by
    # (C_GSPC - C_std); for an uncalibrated light curve mag_gspc = delta_mag + level + C_GSPC.
    # Notes: "A" = no standard zero point (band NONE); "B" = |C_GSPC - C_std| >= 0.02 mag;
    # "B-heuristic" = GSPC not available, red comps (median BP-RP > 1.5) in the Galactic plane
    # (|b| < 10 deg).
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
        elif i < len(_raw_tf) and str(_raw_tf.iloc[i]) == "nondetection":
            base_flags.append("nondetection")
        elif math.isfinite(mag_calib[i]):
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
        n_sigma=float(getattr(_cfg, "lc_outlier_evidence_n_sigma", 5.0) or 5.0),
        proc_cache={},
    )
    # Normalize proc cache keys to basenames for EvidenceCache lookups.
    _proc_by_name: dict[str, pd.DataFrame] = {}
    for _k, _dfc in (getattr(state, "_phase2a_csv_cache", None) or {}).items():
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
            target_frames=target_frames,
            ct_ok=bool(ct_ok),
            cfg=_cfg,
            err=err,
            bjd=bjd,
            source_files=_src_files_pre,
            frame_qc_reasons=getattr(state, "frame_qc_reasons", None),
            evidence_for_index=_evidence_at,
            err_photon=err_photon_export,
            err_sem_rel=err_sem_rel_export,
            err_scint_rel=err_scint_rel_export,
            err_sigma_sys_rel=err_sigma_sys_rel_export,
        )
    )
    # FIX-LOWSNR-2026-10: the FRAME-REJECT-2026-10 verdict (target SNR < snr_min -> low_snr,
    # not exported) lived only in base_flags, which nothing read; apply it to the LC flags.
    # Physical verdicts already set (saturated / no_data / nondetection / edge_fail) stay.
    if out_flag_reasons is None or len(out_flag_reasons) != len(out_flags):
        out_flag_reasons = [""] * len(out_flags)
    for _i, _bf in enumerate(base_flags):
        if _bf == "low_snr" and _i < len(out_flags) and out_flags[_i] not in (
            "saturated", "no_data", "nondetection", "edge_fail"
        ):
            _prev = (str(out_flag_reasons[_i] or out_flags[_i]) if out_flags[_i] != "normal" else "")
            out_flags[_i] = "low_snr"
            _er_i = float(_err_rel_arr[_i]) if _i < _err_rel_arr.size else float("nan")
            out_flag_reasons[_i] = (f"low_snr: SNR {1.0 / _er_i:.1f} < {_snr_min_f:g}"
                                    if math.isfinite(_er_i) and _er_i > 0 else "low_snr")
            if _prev:  # keep the earlier diagnosis (e.g. frame_qc / high_err) in the reason
                out_flag_reasons[_i] += f" (also {_prev})"
    if not _calibrated:
        _nan_lc = np.full(len(_meas), np.nan, dtype=np.float64)
        mag_calib, mag_calib_ct = _nan_lc.copy(), _nan_lc.copy()

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
                    cfg=_cfg,
                    n_comp_min=2,
                    comp_photon_mag=_comp_photon,
                    sigma_sys_mag=_sigma_sys_mag,
                    allow_relative=not _calibrated,
                )
                if _chk_result is not None and np.isfinite(_chk_result.kmag).any():
                    # CHECK-RULES-2026-10: per-epoch K error = photon (check) + ensemble + sigma_sys,
                    # the same budget as the target, for the chi2 consistency test.
                    _kerr = np.full(len(src_files), np.nan, dtype=np.float64)
                    try:
                        _cf = all_frames[
                            all_frames["catalog_id"].astype(str).map(_normalize_gaia_id) == str(_chk_cid)
                        ]
                        _err_by_src = dict(
                            zip(
                                _cf["source_file"].astype(str),
                                pd.to_numeric(_cf["err"], errors="coerce").to_numpy(dtype=np.float64),
                                strict=False,
                            )
                        )
                        _ss = float(_sigma_sys_mag) if _sigma_sys_mag is not None else 0.0
                        if not math.isfinite(_ss):
                            _ss = 0.0
                        for _ii, _sf in enumerate(src_files):
                            _er = float(_err_by_src.get(str(_sf), float("nan")))
                            _es = float(_ensemble_scatter_by_file.get(str(_sf), 0.0) or 0.0)
                            if math.isfinite(_er) and _er > 0:
                                _kerr[_ii] = math.sqrt((1.0857362 * _er) ** 2 + (_es if math.isfinite(_es) else 0.0) ** 2 + _ss**2)
                    except Exception as _kerr_exc:  # noqa: BLE001
                        logging.debug("[CHECK-KMAG] kmag_err not built for %s: %s", target_cid, _kerr_exc)
                    save_check_kmag_sidecar(
                        check_kmag_sidecar_path(lc_dir, target_cid),
                        bjd=bjd,
                        source_files=src_files,
                        kmag=_chk_result.kmag,
                        kmag_err=_kerr,
                    )
                    _check_block = check_star_block(_chk_cid, _chk_result)
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
    )
    # EPSF-LC-LOG-01 / INV-PSF-SUBMIT-01: PSF LC files are an internal diagnostic
    # product written by psf_internal_lc (RUN ePSF path), not Phase 2A.

    # COMP-ASSIGN-01 D4: stability AFTER photometry - verdict only (membership unchanged).
    comp_bjd = {cid: _get_comp_bjd_series(cid, all_frames) for cid in comp_ids}
    comp_quality = check_comparison_stability(
        comp_lc,
        comp_rms_map=comp_rms_map,
        comp_bjd=comp_bjd,
        max_comp_p2p=float(_cfg.phase01_comparison_max_comp_rms),
        max_comp_slope_mmag_hr=float(_cfg.comp_max_slope_mmag_hr),
        comp_slope_significance_k=float(getattr(_cfg, "comp_slope_significance_k", 3.0)),
        common_mode_detrend=True,
        stability_run_flags=state.stability_run_flags,
    )

    # Kvalita comp pre UI (tabulka 'Porovnavacie hviezdy')
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
                delta_mag_mode=not _calibrated,
                delta_mag=delta_mag,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("[PHASE 2A] Optional artifact write failed (lightcurve PNG): %s", exc)

    # Per-target field map s cislovanymi comp hviezdami - vzdy (UI)
    try:
        if state.target_field_map is not None:
            state.target_field_map.save(lc_dir / f"field_map_{target_cid}.jpg", target_row,
                                        _comp_index.get(target_cid, pd.DataFrame()))
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[PHASE 2A] Optional artifact write failed (field map PNG): %s", exc)

    # Summary riadok
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
    # radius is the Phase-1 global one (draft 525: PDF said 7.2 px, LC used ~5.0 px).
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
        "vsx_name": target_name,
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
        f"[FAZA 2A] {target_name}: "
        f"lc_rms={lc_rms:.4f}, lc_rms_ooe={lc_rms_ooe:.4f}, "
        f"n_comp={n_good_comp} (stability_good={n_stability_good}), "
        f"apertura={r_ap:.2f}px (measured)"
    )


    state.chip_fw = chip_fw
    state.chip_fh = chip_fh
    return summary_rows, n_lc
