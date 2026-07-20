# ruff: noqa
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "comp_selection_per_target.py"
text = path.read_text(encoding="utf-8")
start = text.index("def _accumulate_per_frame_comp_metrics(")
end = text.index("\ndef _apply_comp_metric_hard_filters(")

NEW = '''def _accumulate_per_frame_comp_metrics(
    per_frame_csv_paths: list[Path],
    csv_cache: dict[str, pd.DataFrame],
    cand_ids: set[str],
    *,
    flux_col: str,
    chip_fw: int | None,
    chip_fh: int | None,
) -> dict[str, Any]:
    flux_map: dict[str, list[float]] = {cid: [] for cid in cand_ids}
    n_frames_loaded = 0
    contamination_map: dict[str, float] = {}
    psf_chi2_map: dict[str, list[float]] = {cid: [] for cid in cand_ids}
    fwhm_map: dict[str, list[float]] = {cid: [] for cid in cand_ids}
    frame_fwhm_medians: list[float] = []
    peak_over_map: dict[str, int] = {cid: 0 for cid in cand_ids}
    peak_total_map: dict[str, int] = {cid: 0 for cid in cand_ids}
    snr_map: dict[str, list[float]] = {cid: [] for cid in cand_ids}
    edge_bad_map: dict[str, int] = {cid: 0 for cid in cand_ids}
    edge_total_map: dict[str, int] = {cid: 0 for cid in cand_ids}
    _chip_w_eff: int | None = int(chip_fw) if chip_fw is not None else None
    _chip_h_eff: int | None = int(chip_fh) if chip_fh is not None else None
    _edge_log_done = False

    for csv_path in per_frame_csv_paths:
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
                sp = _cand[[name_col, "peak_max_adu", "saturate_limit_adu_85pct"]].copy()
                sp["_peak"] = pd.to_numeric(sp["peak_max_adu"], errors="coerce")
                sp["_limit"] = pd.to_numeric(sp["saturate_limit_adu_85pct"], errors="coerce")
                sp = sp[sp["_limit"].gt(0) & sp["_peak"].notna() & sp["_limit"].notna()]
                if not sp.empty:
                    sp["_over"] = sp["_peak"] > sp["_limit"]
                    for cid, n_tot in sp.groupby(name_col, sort=False).size().items():
                        cid_s = str(cid)
                        peak_total_map[cid_s] = int(peak_total_map.get(cid_s, 0)) + int(n_tot)
                    for cid, n_over in sp.loc[sp["_over"]].groupby(name_col, sort=False).size().items():
                        cid_s = str(cid)
                        peak_over_map[cid_s] = int(peak_over_map.get(cid_s, 0)) + int(n_over)

            if "psf_chi2" in df.columns and not _cand.empty:
                sp = _cand[[name_col, "psf_chi2"]].copy()
                sp["_chi2"] = pd.to_numeric(sp["psf_chi2"], errors="coerce")
                sp = sp[sp["_chi2"].gt(0)]
                for cid, vals in sp.groupby(name_col, sort=False)["_chi2"]:
                    psf_chi2_map[str(cid)].extend(vals.astype(float).tolist())

            if "fwhm_estimate_px" in df.columns:
                _fwhm_col = pd.to_numeric(df["fwhm_estimate_px"], errors="coerce")
                _frame_fwhm_med = float(_fwhm_col.median())
                if math.isfinite(_frame_fwhm_med) and _frame_fwhm_med > 0:
                    frame_fwhm_medians.append(_frame_fwhm_med)
                if not _cand.empty:
                    sp = _cand[[name_col, "fwhm_estimate_px"]].copy()
                    sp["_fwhm"] = pd.to_numeric(sp["fwhm_estimate_px"], errors="coerce")
                    sp = sp[sp["_fwhm"].gt(0)]
                    for cid, vals in sp.groupby(name_col, sort=False)["_fwhm"]:
                        fwhm_map[str(cid)].extend(vals.astype(float).tolist())

            sub = df[df[name_col].isin(cand_ids) & df[actual_flux_col].gt(0)].copy()
            if sub.empty:
                continue

            mag_col_frame = "mag" if "mag" in sub.columns else None
            if mag_col_frame and mag_col_frame in sub.columns:
                sub = sub.copy()
                sub["_mag_num"] = pd.to_numeric(sub[mag_col_frame], errors="coerce")
                sub["_mag_bin"] = (sub["_mag_num"] / 0.5).apply(
                    lambda x: int(x) if math.isfinite(x) else -1
                )
                bin_meds: dict[int, float] = {}
                for b, grp in sub.groupby("_mag_bin"):
                    bmed = float(grp[actual_flux_col].median())
                    if math.isfinite(bmed) and bmed > 0:
                        bin_meds[int(b)] = bmed
                if not bin_meds:
                    continue
            else:
                frame_med = float(sub[actual_flux_col].median())
                if not math.isfinite(frame_med) or frame_med <= 0:
                    continue
                bin_meds = {}

            n_frames_loaded += 1
            sub_work = sub.copy()
            raw_flux = pd.to_numeric(sub_work[actual_flux_col], errors="coerce")
            sub_work["_raw_flux"] = raw_flux

            if bin_meds:
                _bin_keys = np.fromiter(bin_meds.keys(), dtype=np.int64)

                def _norm_med_for_bin(b: int) -> float:
                    bi = int(b)
                    if bi in bin_meds:
                        return float(bin_meds[bi])
                    if len(_bin_keys) == 0:
                        return float("nan")
                    ck = int(_bin_keys[int(np.argmin(np.abs(_bin_keys - bi)))])
                    return float(bin_meds[ck])

                sub_work["_norm_med"] = sub_work["_mag_bin"].map(_norm_med_for_bin)
            else:
                sub_work["_norm_med"] = float(frame_med)

            sub_work["_rel"] = sub_work["_raw_flux"] / pd.to_numeric(sub_work["_norm_med"], errors="coerce")
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
                _edge_ok = _edge_valid & (x0 - r_out >= 0.0) & (x0 + r_out <= w) & (y0 - r_out >= 0.0) & (y0 + r_out <= h)
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
            sky = pd.to_numeric(sub_work.get("noise_floor_adu", pd.Series(0.0, index=sub_work.index)), errors="coerce")
            r_ap = pd.to_numeric(sub_work.get("aperture_r_px", pd.Series(7.0, index=sub_work.index)), errors="coerce")
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
                for cid, grp in sub_work.groupby(name_col, sort=False):
                    cid_s = str(cid)
                    edge_total_map[cid_s] = int(edge_total_map.get(cid_s, 0)) + int(grp["_edge_count"].sum())
                    edge_bad_map[cid_s] = int(edge_bad_map.get(cid_s, 0)) + int(grp["_edge_bad"].sum())

            for cid, grp in sub_work.loc[_rel_ok].groupby(name_col, sort=False):
                cid_s = str(cid)
                flux_map[cid_s].extend(grp["_rel"].astype(float).tolist())

            for cid, grp in sub_work.groupby(name_col, sort=False):
                cid_s = str(cid)
                snr_vals = grp["_snr"].to_numpy(dtype=np.float64)
                snr_vals = snr_vals[np.isfinite(snr_vals)]
                if snr_vals.size > 0:
                    snr_map[cid_s].extend(snr_vals.astype(float).tolist())

        except Exception:  # noqa: BLE001
            continue

    logging.info(
        "[PERF-4B] _accumulate_per_frame_comp_metrics: %d frames x %d candidates vectorized",
        n_frames_loaded,
        len(cand_ids),
    )
    return {
        "flux_map": flux_map,
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

'''

path.write_text(text[:start] + NEW + text[end:], encoding="utf-8")
print("Patched accumulate")
