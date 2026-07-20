# ruff: noqa
"""Replace select_comparison_stars_per_target body with orchestrator."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
pc = ROOT / "photometry_core.py"
lines = pc.read_text(encoding="utf-8").splitlines(keepends=True)

# Find function body start (after docstring) and end
start_def = None
for i, ln in enumerate(lines):
    if ln.startswith("def select_comparison_stars_per_target("):
        start_def = i
        break
assert start_def is not None

# docstring ends at line with closing """
body_start = start_def + 1
while body_start < len(lines) and '"""' not in lines[body_start]:
    body_start += 1
# skip to line after closing """
if body_start < len(lines):
    if lines[body_start].count('"""') == 1 and not lines[body_start].strip().endswith('"""'):
        body_start += 1
        while body_start < len(lines) and '"""' not in lines[body_start]:
            body_start += 1
body_start += 1

end_def = None
for i in range(body_start, len(lines)):
    if lines[i].startswith("def run_phase0_and_phase1("):
        end_def = i
        break
assert end_def is not None

ORCH = '''    _ = fwhm_px
    from comp_selection_per_target import (  # noqa: PLC0415
        _accumulate_per_frame_comp_metrics,
        _apply_comp_metric_hard_filters,
        _assemble_comp_selection_result_rows,
        _assign_comp_tiers_to_pool,
        _bootstrap_phase1_csv_cache,
        _build_candidates_pre_adaptive_mag,
        _compute_comp_contamination_map,
        _detrend_and_compute_comp_rms_map,
        _ensemble_mad_filter_rms,
        _filter_comp_candidates_spatial_static,
        _resolve_target_color_for_comp_selection,
        _score_comp_candidates_broeg,
    )

    if global_comp_pool_df is not None and not getattr(global_comp_pool_df, "empty", True):
        ms = global_comp_pool_df.copy()
        if "comp_rms" in ms.columns:
            ms = ms.drop(columns=["comp_rms"])
    else:
        ms = masterstars_df.copy()
    _co_mask = _is_catalog_only(ms)
    if _co_mask.any():
        ms = ms[~_co_mask].copy()
        logging.info(
            "[COMP] catalog_only excluded from comp pool: %d removed, %d remain",
            int(_co_mask.sum()),
            int(len(ms)),
        )
    for _id_col in ("catalog_id", "name"):
        if _id_col in ms.columns:
            ms[_id_col] = _normalize_id_series(ms[_id_col])
    for col in (
        "is_usable",
        "is_saturated",
        "is_noisy",
        "snr50_ok",
        "vsx_known_variable",
        "likely_saturated",
    ):
        if col in ms.columns:
            ms[col] = _bool_col(ms[col])

    ctx = _resolve_target_color_for_comp_selection(
        target,
        vsx_local_db_path=vsx_local_db_path,
        gaia_db_path=gaia_db_path,
    )
    ra_t = float(ctx["ra_t"])
    dec_t = float(ctx["dec_t"])
    mag_t = float(ctx["mag_t"])
    target_cid = str(ctx["target_cid"])
    target_bv_pre = float(ctx["target_bv_pre"])
    target_bv_source = str(ctx["target_bv_source"])
    t_bp_tgt = float(ctx["t_bp_tgt"])
    target_bprp_eff = float(ctx["target_bprp_eff"])
    use_bprp_primary = bool(ctx["use_bprp_primary"])
    max_delta_bprp_cfg = float(ctx["max_delta_bprp_cfg"])
    _individual_tier = ctx["_individual_tier"]
    _target_name = str(ctx["_target_name"])

    mag_tol = float(max_mag_diff)
    if (
        math.isfinite(mag_t)
        and float(max_mag_diff_bright_floor) > 0.0
        and mag_t < float(mag_bright_threshold)
    ):
        mag_tol = max(mag_tol, float(max_mag_diff_bright_floor))
        if mag_tol > float(max_mag_diff):
            logging.debug(
                "[FAZA 1] Target %s: jasny ciel (mag=%.2f < %.2f) -> |Deltamag| pas "
                "max(%.3f, floor %.3f) = %.3f",
                target_cid or "?",
                mag_t,
                float(mag_bright_threshold),
                float(max_mag_diff),
                float(max_mag_diff_bright_floor),
                mag_tol,
            )

    ms, _base_mask, det_mask = _filter_comp_candidates_spatial_static(
        ms,
        ra_t=ra_t,
        dec_t=dec_t,
        mag_t=mag_t,
        target_cid=target_cid,
        target_bv_pre=target_bv_pre,
        target_bprp_eff=target_bprp_eff,
        use_bprp_primary=use_bprp_primary,
        max_delta_bprp_cfg=max_delta_bprp_cfg,
        max_dist_deg=max_dist_deg,
        max_bv_diff=max_bv_diff,
        min_dist_arcsec=min_dist_arcsec,
        exclude_gaia_nss=exclude_gaia_nss,
        exclude_gaia_extobj=exclude_gaia_extobj,
        chip_fw=chip_fw,
        chip_fh=chip_fh,
        chip_interior_margin_px=int(chip_interior_margin_px),
        variable_target_catalog_ids=variable_target_catalog_ids,
    )

    built = _build_candidates_pre_adaptive_mag(
        ms,
        _base_mask=_base_mask,
        det_mask=det_mask,
        mag_t=mag_t,
        target_cid=target_cid,
        mag_tol=mag_tol,
        max_mag_diff=max_mag_diff,
        n_comp_min=n_comp_min,
        chip_fw=chip_fw,
        chip_fh=chip_fh,
        chip_interior_margin_px=int(chip_interior_margin_px),
        target=target,
    )
    if built is None:
        return pd.DataFrame()
    candidates_pre, used_mag_tol = built

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
        ms_arr_x = pd.to_numeric(ms.get("x", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
        ms_arr_y = pd.to_numeric(ms.get("y", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
        if "_mag" in ms.columns:
            ms_arr_mag = pd.to_numeric(ms["_mag"], errors="coerce").to_numpy(dtype=float)
        else:
            ms_arr_mag = pd.to_numeric(ms.get("mag", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
    except Exception as _iso_exc:  # noqa: BLE001
        logging.warning(f"[FAZA 1] Aperture izolacia preskocena (chyba): {_iso_exc!s}")
        ms_arr_x = ms_arr_y = ms_arr_mag = np.array([], dtype=float)

    id_col = (
        "name"
        if "name" in candidates_pre.columns
        else ("catalog_id" if "catalog_id" in candidates_pre.columns else "name")
    )
    cand_ids = set(candidates_pre[id_col].astype(str).str.strip())

    avail_cols = _PHASE_USECOLS_PERFRAME.copy()
    csv_cache = _bootstrap_phase1_csv_cache(
        per_frame_csv_paths,
        csv_cache,
        flux_col=flux_col,
        avail_cols=avail_cols,
    )
    metrics = _accumulate_per_frame_comp_metrics(
        per_frame_csv_paths,
        csv_cache,
        cand_ids,
        flux_col=flux_col,
        chip_fw=chip_fw,
        chip_fh=chip_fh,
    )
    flux_map = metrics["flux_map"]
    n_frames_loaded = int(metrics["n_frames_loaded"])
    psf_chi2_map = metrics["psf_chi2_map"]
    fwhm_map = metrics["fwhm_map"]
    frame_fwhm_medians = metrics["frame_fwhm_medians"]
    peak_over_map = metrics["peak_over_map"]
    peak_total_map = metrics["peak_total_map"]
    snr_map = metrics["snr_map"]
    edge_bad_map = metrics["edge_bad_map"]
    edge_total_map = metrics["edge_total_map"]

    min_frames = max(3, int(n_frames_loaded * min_frames_frac))

    flux_map, _b_rejected = _apply_comp_metric_hard_filters(
        flux_map,
        peak_over_map,
        peak_total_map,
        snr_map,
        psf_chi2_map,
        fwhm_map,
        frame_fwhm_medians,
        edge_bad_map,
        edge_total_map,
        target_cid=target_cid,
        edge_bad_frame_frac_max=edge_bad_frame_frac_max,
        max_psf_chi2=max_psf_chi2,
        max_fwhm_factor=max_fwhm_factor,
    )

    contamination_map = _compute_comp_contamination_map(
        flux_map,
        ms,
        target_cid=target_cid,
        isolation_radius_px=isolation_radius_px,
    )

    rms_result = _detrend_and_compute_comp_rms_map(
        flux_map,
        min_frames=min_frames,
        max_comp_rms=max_comp_rms,
        n_comp_min=n_comp_min,
        target_cid=target_cid,
        target=target,
        chip_fw=chip_fw,
        chip_fh=chip_fh,
        chip_interior_margin_px=int(chip_interior_margin_px),
    )
    if rms_result[0] is None:
        return pd.DataFrame()
    rms_map, sorted_rms_map = rms_result

    def _apply_aperture_isolation_safe(cands: pd.DataFrame) -> pd.DataFrame:
        if cands.empty:
            return cands
        try:
            ms_arr_x2 = ms_arr_x
            ms_arr_y2 = ms_arr_y
            ms_arr_mag2 = ms_arr_mag
        except Exception:  # noqa: BLE001
            return cands
        rej: set[Any] = set()
        for idx2, crow2 in cands.iterrows():
            cx2 = float(crow2.get("x", float("nan")))
            cy2 = float(crow2.get("y", float("nan")))
            cm2 = float(crow2.get("_mag", float("nan"))) if "_mag" in cands.columns else float("nan")
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

    candidates = ms[_base_mask | det_mask].copy()
    if candidates.empty:
        logging.warning(f"[FAZA 1] {target_cid}: ziadni kandidati po hard filtroch")
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
        n_comp_min=n_comp_min,
        rms_outlier_sigma=rms_outlier_sigma,
        chip_fw=chip_fw,
        chip_fh=chip_fh,
        chip_interior_margin_px=int(chip_interior_margin_px),
    )
    if active is None:
        return pd.DataFrame()

    id_col_cand = (
        "name"
        if "name" in candidates.columns
        else ("catalog_id" if "catalog_id" in candidates.columns else "name")
    )
    score_map, tier_map = _score_comp_candidates_broeg(
        active,
        candidates,
        contamination_map,
        id_col_cand=id_col_cand,
        mag_t=mag_t,
        target_bv_pre=target_bv_pre,
        target_bprp_eff=target_bprp_eff,
        t_bp_tgt=t_bp_tgt,
        use_bprp_primary=use_bprp_primary,
        _individual_tier=_individual_tier,
    )

    tier_out = _assign_comp_tiers_to_pool(
        candidates,
        active,
        id_col_cand=id_col_cand,
        target=target,
        target_cid=target_cid,
        target_bv_pre=target_bv_pre,
        target_bprp_eff=target_bprp_eff,
        t_bp_tgt=t_bp_tgt,
        mag_t=mag_t,
        use_bprp_primary=use_bprp_primary,
        _individual_tier=_individual_tier,
        _target_name=_target_name,
        max_mag_diff_t1=max_mag_diff_t1,
        max_mag_diff=max_mag_diff,
        gaia_db_path=gaia_db_path,
        vsx_local_db_path=vsx_local_db_path,
        gaia_prefetch=gaia_prefetch,
        n_comp_min=n_comp_min,
        n_comp_max=n_comp_max,
        chip_fw=chip_fw,
        chip_fh=chip_fh,
        chip_interior_margin_px=int(chip_interior_margin_px),
    )
    final_comps = tier_out["final_comps"]
    if final_comps is None or getattr(final_comps, "empty", True):
        return pd.DataFrame()

    try:
        final_lookup = final_comps.copy()
        final_lookup[id_col_cand] = final_lookup[id_col_cand].astype(str).str.strip()
        final_lookup = final_lookup.set_index(id_col_cand, drop=False)
    except Exception:  # noqa: BLE001
        final_lookup = None

    return _assemble_comp_selection_result_rows(
        tier_out["selected_ids"],
        final_comps,
        id_col_cand=id_col_cand,
        active=active,
        score_map=score_map,
        contamination_map=contamination_map,
        flux_map=flux_map,
        target_cid=target_cid,
        target=target,
        target_bv_pre=target_bv_pre,
        target_bv_source=target_bv_source,
        target_bprp_eff=target_bprp_eff,
        t_bp_tgt=t_bp_tgt,
        use_bprp_primary=use_bprp_primary,
        sel_note=str(tier_out["sel_note"]),
        used_mag_tol=float(used_mag_tol),
        best_tier=str(tier_out["best_tier"]),
        tier4_warning=bool(tier_out["tier4_warning"]),
        n_t1=int(tier_out["n_t1"]),
        n_t2=int(tier_out["n_t2"]),
        n_t3=int(tier_out["n_t3"]),
        n_t4=int(tier_out["n_t4"]),
        comp_bv_map=tier_out["comp_bv_map"],
        comp_bv_source_map=tier_out["comp_bv_source_map"],
        comp_tier_final_map=tier_out["comp_tier_final_map"],
        comp_delta_bv_map=tier_out["comp_delta_bv_map"],
        comp_color_tier_src_map=tier_out["comp_color_tier_src_map"],
        _b_rejected=_b_rejected,
        final_lookup=final_lookup,
    )

'''

new_lines = lines[:body_start] + [ORCH + "\n"] + lines[end_def:]
pc.write_text("".join(new_lines), encoding="utf-8")
print(f"Replaced body lines {body_start+1}-{end_def} with orchestrator ({end_def-body_start} -> {len(ORCH.splitlines())} lines)")
