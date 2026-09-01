"""One-shot generator for VYVAR_PARAMS.md (config<->UI parity registry)."""
from __future__ import annotations

import dataclasses
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401  (repo layout: src_py + dev on sys.path)
ROOT = _bootstrap.REPO_ROOT
def _area(key: str) -> str:
    rules: list[tuple[str, str]] = [
        ("comp_qa_", "Trust / QA (comp_qa)"),
        ("trust_flag_", "Trust / QA (trust gate)"),
        ("observer_", "Observer / export"),
        ("aavso_", "Observer / export"),
        ("varastro_", "Observer / export"),
        ("export_", "Observer / export"),
        ("archive_root", "Paths"),
        ("calibration_library", "Paths / calibration library"),
        ("database_path", "Paths"),
        ("gaia_", "Paths / catalogs"),
        ("blind_index", "Paths / catalogs"),
        ("vsx_", "Paths / catalogs"),
        ("catalog_query", "Paths / catalogs"),
        ("masterdark", "Calibration validity"),
        ("masterflat", "Calibration validity"),
        ("bpm_", "Calibration / BPM"),
        ("qc_", "QC"),
        ("dao_qc", "QC"),
        ("auto_fwhm", "QC / FITS QA"),
        ("alignment_", "Alignment"),
        ("masterstar_", "MASTERSTAR / plate-solve"),
        ("platesolve_", "MASTERSTAR / plate-solve"),
        ("sips_dao", "MASTERSTAR / SIPS DAO"),
        ("debug_platesolver", "MASTERSTAR / plate-solve"),
        ("blind_verify_", "MASTERSTAR / plate-solve"),
        ("phase01_comparison", "Phase 0+1 comp selection"),
        ("phase01_", "Phase 0+1"),
        ("comp_tier", "Phase 0+1 comp selection"),
        ("comp_max", "Phase 0+1 comp selection"),
        ("comp_contamination", "Phase 0+1 comp selection"),
        ("global_comp_pool", "Phase 0+1 comp selection"),
        ("crowding_", "Crowding classifier"),
        ("field_density", "Field density"),
        ("aperture_", "Photometry / aperture"),
        ("annulus_", "Photometry / aperture"),
        ("nonlinearity_", "Photometry / aperture"),
        ("cog_", "Photometry / COG correction"),
        ("photometry_mode", "Photometry mode"),
        ("psf_", "PSF photometry"),
        ("moffat_", "PSF photometry"),
        ("epsf_", "PSF photometry"),
        ("gain", "Sensor / noise model"),
        ("read_noise", "Sensor / noise model"),
        ("frame_", "Sensor geometry"),
        ("sky_adu", "Sensor / noise model"),
        ("saturate_", "Sensor / saturation"),
        ("temporal_bin", "Phase 2A detrend / ALG"),
        ("savgol_", "Phase 2A detrend / ALG"),
        ("democratic_", "Phase 2A detrend / ALG"),
        ("pytics_", "Phase 2A detrend / ALG"),
        ("sysrem_", "Phase 2A detrend / ALG"),
        ("phase2a_", "Phase 2A"),
        ("comp_qa", "Trust / QA"),
        ("save_lightcurve", "Phase 2A output"),
        ("gs11_", "GS11 dilution"),
        ("variability_", "Variability detection"),
        ("tess_", "TESS"),
        ("per_frame_mp", "Parallelism"),
        ("qc_preprocess_workers", "Parallelism"),
        ("plate_scale", "Plate scale"),
        ("plate_solve_fov", "Plate scale / FOV"),
        ("project_root", "Internal"),
    ]
    for prefix, area in rules:
        if key.startswith(prefix) or key == prefix.rstrip("_"):
            return area
    return "Other"


def _parse_clamps(src: str) -> dict[str, str]:
    clamps: dict[str, str] = {}
    # _f01("key", default, lo, hi)
    for m in re.finditer(
        r'_f01\(\s*"([^"]+)"\s*,\s*[^,]+,\s*([^,]+),\s*([^)]+)\)',
        src,
    ):
        key, lo, hi = m.group(1), m.group(2).strip(), m.group(3).strip()
        clamps[key] = f"{lo} ... {hi}"
    # max(lo, min(hi, int(data.get("key"
    for m in re.finditer(
        r'self\.(\w+)\s*=\s*max\(\s*([^,]+),\s*min\(\s*([^,]+),\s*int\(data\.get\("([^"]+)"',
        src,
    ):
        lo, hi, key = m.group(2).strip(), m.group(3).strip(), m.group(4)
        clamps.setdefault(key, f"{lo} ... {hi}")
    for m in re.finditer(
        r'self\.(\w+)\s*=\s*max\(\s*([^,]+),\s*min\(\s*([^,]+),\s*float\(data\.get\("([^"]+)"',
        src,
    ):
        lo, hi, key = m.group(2).strip(), m.group(3).strip(), m.group(4)
        clamps.setdefault(key, f"{lo} ... {hi}")
    # explicit max/min on assignment lines
    for m in re.finditer(
        r'self\.(\w+)\s*=\s*max\(\s*([^,]+),\s*min\(\s*([^,]+),\s*float\([^)]+\)\s*\)',
        src,
    ):
        key, lo, hi = m.group(1), m.group(2).strip(), m.group(3).strip()
        clamps.setdefault(key, f"{lo} ... {hi}")
    # UI save clamps in ui_settings (document separately if found)
    manual = {
        "masterstar_prematch_peak_sigma_floor": "0.5 ... 6.0 (UI slider)",
        "masterstar_dao_threshold_sigma": "0.1 ... 6.0 (UI slider)",
        "aperture_fwhm_factor": "0.5 ... 6.0",
        "annulus_inner_fwhm": "1.0 ... 10.0",
        "annulus_outer_fwhm": "1.5 ... 12.0",
        "nonlinearity_peak_percentile": "0.0 ... 50.0",
        "nonlinearity_fwhm_ratio": "1.01 ... 3.0",
        "aperture_variable_factor": "0.5 ... 2.0",
        "aperture_comp_factor": "0.5 ... 2.0",
        "phase01_comparison_max_dist_deg": "0.05 ... 10.0",
        "phase01_comparison_max_mag_diff": "0.05 ... 5.0",
        "phase01_comparison_rms_bin_mag": "0.0001 ... 0.05",
        "sips_dao_fwhm_px": "1.0 ... 8.0",
        "calibration_library_native_binning": "1 ... 16",
        "alignment_max_stars": "10 ... 5000",
        "alignment_max_control_points": "12 ... 500",
        "catalog_query_max_rows": "1000 ... 500000",
        "psf_spatial_order": "0 ... 2",
        "psf_spatial_enabled": "master gate; spatial ePSF active iff enabled AND order>0",
    }
    clamps.update({k: v for k, v in manual.items() if k not in clamps})
    return clamps


def _scan_ui() -> dict[str, list[str]]:
    ui_hits: dict[str, list[str]] = {}
    pat_cfg = re.compile(r"\bcfg\.(\w+)\b")
    pat_getattr = re.compile(r'getattr\(cfg,\s*["\'](\w+)["\']')
    key_prefixes = (
        "phase01_",
        "comp_",
        "masterstar_",
        "qc_",
        "aperture_",
        "annulus_",
        "crowding_",
        "psf_",
        "cog_",
        "comp_qa",
        "trust_flag",
        "archive_",
        "calibration_",
        "database_",
        "gaia_",
        "blind_",
        "vsx_",
        "alignment_",
        "sips_",
        "temporal_",
        "pytics_",
        "savgol_",
        "democratic_",
        "nonlinearity_",
        "observer_",
        "aavso_",
        "varastro_",
        "photometry_",
        "save_lightcurve",
        "sysrem_",
        "auto_fwhm",
        "dao_qc",
        "field_density",
        "global_comp",
        "tess_",
        "per_frame",
    )
    for ui_path in sorted(ROOT.glob("ui*.py")):
        rel = ui_path.name
        lines = ui_path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines, 1):
            loc = f"{rel}:{i}"
            for m in pat_cfg.finditer(line):
                ui_hits.setdefault(m.group(1), []).append(loc)
            for m in pat_getattr.finditer(line):
                ui_hits.setdefault(m.group(1), []).append(loc)
            for m in re.finditer(r'["\']([a-z][a-z0-9_]{4,})["\']', line):
                key = m.group(1)
                if key.startswith(key_prefixes):
                    ui_hits.setdefault(key, []).append(loc)
            # cfg.field = assignment on save
            for m in re.finditer(r"cfg\.(\w+)\s*=", line):
                ui_hits.setdefault(m.group(1), []).append(loc)
    # dedupe preserving order
    for k, v in list(ui_hits.items()):
        seen: set[str] = set()
        deduped = []
        for x in v:
            if x not in seen:
                seen.add(x)
                deduped.append(x)
        ui_hits[k] = deduped[:3]
    return ui_hits


INTENTIONALLY_HIDDEN = {
    "comp_qa_enabled",
    "trust_flag_enabled",
    "phase01_comparison_proximity_tiebreak",
    "phase01_comparison_rms_bin_mag",
    "cog_aperture_correction_enabled",
    "cog_ref_fwhm",
    "cog_min_stars",
    "cog_isolation_fwhm",
    "cog_snr_min",
    "cog_sat_frac",
    "cog_ladder_step_px",
    "cog_ac_factor_max",
    "crowding_classifier_enabled",
    "crowding_blend_tighten_threshold",
    "crowding_comp_availability_loosen_count",
    "crowding_tighten_min_fwhm_px",
    "psf_adaptive_enabled",
    "psf_adaptive_resolve_fwhm",
    "psf_adaptive_snr_lo",
    "psf_grouper_enabled",
    "psf_group_sep_fwhm",
    "psf_neighbor_include_fwhm",
    "psf_spatial_enabled",
    "psf_spatial_order",
    "psf_chi2_threshold",
    "psf_quality_fallback_enabled",
    "phase2a_airmass_before_outlier",
    "sysrem_enabled",
    "sysrem_n_iter",
    "debug_platesolver",
    "blind_verify_enabled",
    "blind_verify_top_n",
    "blind_verify_match_tol_px",
    "blind_verify_min_matches",
    "blind_verify_min_fraction",
    "blind_verify_inmemory_catalog",
    "verify_mag_limit",
    "blind_verify_early_accept",
    "blind_verify_early_floor",
    "blind_verify_early_fraction",
    "blind_prefilter_min",
    "qc_preprocess_workers",
    "project_root",
    "plate_solve_fov_deg",
    "export_arcsec_per_px",
    "phase01_plate_scale_arcsec_per_px",
    "plate_scale_arcsec_per_px",
    "phase01_use_bprp_as_bv_fallback",
    "phase01_tier1_mag",
    "phase01_tier2_mag",
    "phase01_tier3_mag",
    "phase01_tier4_mag",
    "phase01_tier1_bv",
    "phase01_tier2_bv",
    "phase01_tier3_bv",
    "phase01_tier4_bv",
    "phase01_comparison_fov_fraction",
    "phase01_comparison_mag_bright_threshold",
    "phase01_comparison_max_mag_diff_bright_floor",
    "phase01_comparison_max_mag_diff_absolute",
    "phase01_comparison_max_psf_chi2",
    "phase01_comparison_max_fwhm_factor",
    "phase01_comparison_isolation_radius_px",
    "phase01_ct_min_comp",
    "phase01_ct_extrapolation_tol",
    "phase01_flux_col",
    "comp_max_slope_mmag_hr",
    "aperture_fwhm_factor_small",
    "aperture_fwhm_factor_medium",
    "aperture_fwhm_factor_large",
    "aperture_correction_enabled",
    "aperture_correction_min_ref_stars",
    "aperture_correction_max_contamination",
    "aperture_correction_max_scatter_mag",
    "moffat_chi2_limit",
    "epsf_min_stars",
    "masterstar_use_best_frame_fwhm",
    "masterstar_dao_pass2_sigma",
    "masterstar_best_of_n",
    "masterstar_platesolve_prewrite_rms_max_px",
    "masterstar_platesolve_prewrite_relaxed_rms_max_px",
    "masterstar_platesolve_nn_refine_max_rms_px",
    "masterstar_sip_force_rms_guard_ratio",
    "masterstar_catalog_recovery_min",
    "masterstar_min_matched_floor",
    "masterstar_centre_rms_max_px",
    "masterstar_distortion_benign_ratio_max",
    "masterstar_odds_match_floor",
    "masterstar_odds_k",
    "masterstar_odds_min_quadrants",
    "masterstar_false_alarm_p_max",
    "masterstar_quality_crowded_n_cat_min",
    "masterstar_detection_cap_adaptive",
    "masterstar_detection_cap_min",
    "masterstar_detection_cap_max",
    "masterstar_detection_cap_k",
    "masterstar_sibling_recovery_enabled",
    "masterstar_sibling_min_matched",
    "masterstar_sibling_rms_max_px",
    "masterstar_sibling_min_quadrants",
    "masterstar_sibling_stack_n",
    "masterstar_solver_use_draft_median_if_hint_sep_deg",
    "masterstar_log_astroalign",
    "masterstar_optimizer_mirror_extra_log",
    "saturate_limit_fraction",
    "bpm_dark_mad_sigma",
    "gain",
    "read_noise",
    "sky_adu_fallback",
    "frame_width_px",
    "frame_height_px",
    "gs11_dilution_enabled",
    "gs11_dilution_aperture_arcsec",
    "gs11_dilution_mag_limit_delta",
    "gs11_comp_max_dilution",
    "gs11_target_min_dilution",
    "temporal_bin_window",
    "savgol_window_frac",
    "savgol_polyorder",
    "democratic_sg_window_frac",
    "pytics_n_iter",
    "field_density_adaptive_enabled",
    "field_density_sparse_threshold",
    "field_density_dense_threshold",
    "tess_enabled",
    "variability_min_frames",
    "variability_min_frames_frac",
    "variability_p85_filter",
    "variability_slope_floor",
    "variability_sigma_threshold",
    "variability_comp_floor_factor",
    "variability_smoothness_max",
    "variability_mag_limit",
    "variability_min_rms_pct",
    "variability_min_amplitude_mag",
    "variability_clip_ratio_min",
    "variability_vdi_z_threshold",
    "variability_min_points_rms",
    "qc_max_background_rms",
    "qc_dao_detection_sigma",
    "per_frame_mp_reserve_ram_gb",
    "platesolve_anisotropy_threshold",
    "aavso_observer_code",
    "comp_contamination_penalty_k",
}


def main() -> None:
    import sys

    sys.path.insert(0, str(ROOT))
    from config import AppConfig

    cfg_src = (ROOT / "config.py").read_text(encoding="utf-8")
    clamps = _parse_clamps(cfg_src)
    ui_hits = _scan_ui()

    json_path = ROOT / "config.json"
    json_data = json.loads(json_path.read_text(encoding="utf-8")) if json_path.exists() else {}
    json_keys = set(json_data.keys())

    # Fresh config with project root only (loads config.json - use dataclass defaults via inspection)
    dc_defaults: dict[str, object] = {}
    for f in dataclasses.fields(AppConfig):
        if f.name in {"archive_root", "calibration_library_root", "database_path"}:
            continue
        if f.default is not dataclasses.MISSING:
            dc_defaults[f.name] = f.default
        elif f.default_factory is not dataclasses.MISSING:  # type: ignore[comparison-overlap]
            dc_defaults[f.name] = f.default_factory()

    cfg = AppConfig()
    runtime: dict[str, object] = {f.name: getattr(cfg, f.name) for f in dataclasses.fields(AppConfig)}

    rows: list[tuple[str, str, str, str, str, str, str]] = []
    all_keys = sorted(f.name for f in dataclasses.fields(AppConfig) if f.name != "project_root")
    config_keys = set(all_keys)

    for key in all_keys:
        if key in {"archive_root", "calibration_library_root", "database_path"}:
            continue
        default_dc = dc_defaults.get(key, "-")
        default_rt = runtime.get(key, "-")
        default_json = json_data.get(key, "-")
        if default_json != "-":
            default_str = repr(default_json)
            if default_dc != "-" and repr(default_dc) != repr(default_json):
                default_str += f" (dataclass {repr(default_dc)})"
        elif default_dc != "-":
            default_str = repr(default_dc)
            if repr(default_rt) != repr(default_dc):
                default_str += f" -> runtime {repr(default_rt)}"
        else:
            default_str = repr(default_rt)

        clamp = clamps.get(key, "-")
        ui_locs = ui_hits.get(key, [])
        ui_loc = ", ".join(ui_locs) if ui_locs else "-"
        in_json = key in json_keys

        if ui_locs:
            exposed = "yes"
        elif key in INTENTIONALLY_HIDDEN:
            exposed = "intentionally-hidden"
        else:
            exposed = "no"

        area = _area(key)
        json_note = "" if in_json else " ! not in config.json"
        rows.append((area, key, default_str, clamp, ui_loc, exposed, json_note))

    # UI-only drift (config-like keys referenced in UI but not on AppConfig)
    ui_only_noise = {
        "to_json",
        "to_dict",
        "ensure_base_dirs",
        "project_root",
        "gaia_dr3",
        "archive_path",
        "per_frame_csv",
        "per_frame_csv_dir",
        "photometry_dir",
        "masterstar_fits",
        "masterstar_fits_path",
    }
    ui_only = sorted(
        k
        for k in ui_hits
        if k not in config_keys
        and not k.startswith("vyvar_")
        and k not in ui_only_noise
        and not k.endswith(("_flux", "_flux_err", "_mag", "_mag_err", "_mag_rms", "_rms", "_px"))
        and k
        not in {
            "aperture_median_mag",
            "psf_median_mag",
            "comp_ids",
            "comp_n_frames",
            "comp_rms",
            "comp_rms_map",
            "comp_stars",
            "comp_tier",
            "comp_weight",
            "gaia_bprp",
            "gaia_dao_completeness_pct",
            "gaia_dr3_variable_catalog",
            "gaia_match_source",
            "gaia_teff",
            "masterstar_candidate_paths",
            "masterstar_candidates_n",
            "masterstar_candidates_table",
            "masterstar_processed_total",
            "psf_chi2",
            "psf_dao_ratio",
            "psf_fit_ok",
            "psf_flux",
            "psf_flux_err",
            "qc_rank",
            "tess_auto_done",
            "tess_results",
            "tess_selected_sector",
            "vsx_known_variable",
            "vsx_match",
            "vsx_name",
            "vsx_name_display",
            "vsx_period",
            "vsx_type",
        }
    )

    # Group rows
    from collections import defaultdict

    grouped: dict[str, list[tuple]] = defaultdict(list)
    for row in rows:
        grouped[row[0]].append(row)

    lines = [
        "# VYVAR - Config <-> UI parameter registry",
        "",
        "Generated **2026-06-02** from `config.py`, `config.json`, and `ui*.py`.",
        "Registry required by `docs/VYVAR_PROCESS.md` Definition of Done S4.",
        "",
        "**Legend - exposed:** `yes` = Settings or tool UI widget; `intentionally-hidden` =",
        "dev/gated flag documented here (edit via `config.json`); `no` = drift (config-only,",
        "no UI yet).",
        "",
        f"**Summary:** {sum(1 for r in rows if r[5]=='yes')} exposed . "
        f"{sum(1 for r in rows if r[5]=='intentionally-hidden')} intentionally-hidden . "
        f"{sum(1 for r in rows if r[5]=='no')} config-only (no UI) . "
        f"{len(ui_only)} UI references without config key",
        "",
        "---",
        "",
    ]

    priority_keys = [
        "comp_qa_enabled",
        "trust_flag_enabled",
        "phase01_comparison_proximity_tiebreak",
        "phase01_comparison_rms_bin_mag",
        "cog_aperture_correction_enabled",
        "crowding_classifier_enabled",
        "psf_adaptive_enabled",
        "psf_grouper_enabled",
        "psf_spatial_enabled",
    ]

    lines += [
        "## Recently-added flags (audit focus)",
        "",
        "| key | default | clamp | UI | exposed | notes |",
        "|-----|---------|-------|-----|---------|-------|",
    ]
    row_by_key = {r[1]: r for r in rows}
    for pk in priority_keys:
        if pk not in row_by_key:
            continue
        area, key, default, clamp, ui_loc, exposed, json_note = row_by_key[pk]
        note = json_note.strip() or ("OK in config.json" if pk in json_keys else "missing from shipped config.json")
        lines.append(
            f"| `{key}` | {default} | {clamp} | {ui_loc} | **{exposed}** | {note} |"
        )

    lines += ["", "---", ""]

    for area in sorted(grouped.keys()):
        lines.append(f"## {area}")
        lines.append("")
        lines.append("| key | default | clamp/range | UI location | exposed |")
        lines.append("|-----|---------|-------------|-------------|---------|")
        for _, key, default, clamp, ui_loc, exposed, json_note in sorted(grouped[area], key=lambda x: x[1]):
            d = default.replace("|", "\\|")[:80]
            lines.append(f"| `{key}` | {d} | {clamp} | {ui_loc} | {exposed}{json_note} |")
        lines.append("")

    if ui_only:
        lines += [
            "---",
            "",
            "## UI references without `AppConfig` key (reverse drift)",
            "",
            "These strings appear in UI code but are not fields on `AppConfig`:",
            "",
        ]
        for k in ui_only:
            lines.append(f"- `{k}` -> {', '.join(ui_hits[k])}")
        lines.append("")

    out = ROOT / "docs" / "VYVAR_PARAMS.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out} ({len(rows)} keys)")


if __name__ == "__main__":
    main()
