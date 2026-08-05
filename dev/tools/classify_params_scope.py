"""Scope classifier for params_registry.json (Task C).

Mechanical rules + explicit overrides. Run: python dev/tools/classify_params_scope.py --write
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src_py"))
import params_registry as pr  # noqa: E402

REG_PATH = ROOT / "dev" / "validation" / "params_registry.json"

# Explicit per-key overrides (ground-truth anchors and resolver-aligned resolved keys).
EXPLICIT: dict[str, tuple[str, str, str]] = {
    # --- resolved / fits_dynamic (param_resolver alignment) ---
    "gain": ("rig", "high", "Equipment-intrinsic e-/ADU; resolver category equipment-intrinsic."),
    "read_noise": ("rig", "high", "Equipment-intrinsic e-; resolver category equipment-intrinsic."),
    "plate_scale_arcsec_per_px": ("session", "high", "Observation-specific plate scale from solved WCS per exposure."),
    "phase01_plate_scale_arcsec_per_px": ("session", "high", "Phase-01 plate scale from WCS/header per run."),
    "export_arcsec_per_px": ("session", "high", "Export metadata plate scale resolved from WCS at run time."),
    "plate_solve_fov_deg": ("rig", "high", "Blind-solve FOV seed depends on telescope+camera field size."),
    "frame_height_px": ("rig", "high", "Detector frame height in pixels (camera ROI/binning)."),
    "frame_width_px": ("rig", "high", "Detector frame width in pixels (camera ROI/binning)."),
    "qc_preprocess_workers": ("universal", "high", "Internal worker cap derived from hardware; plumbing not science knob."),
    # --- FWHM-normalised (universal by construction) ---
    "annulus_inner_fwhm": ("universal", "high", "Sky annulus inner radius as multiple of measured FWHM."),
    "annulus_outer_fwhm": ("universal", "high", "Sky annulus outer radius as multiple of measured FWHM."),
    "aperture_fwhm_factor": ("universal", "high", "Aperture radius as multiple of measured FWHM."),
    "aperture_snr_sizing": ("universal", "high", "SNR sizing uses FWHM-normalised radii."),
    "aperture_comp_factor": ("universal", "high", "Comp aperture as multiple of target aperture (dimensionless)."),
    "aperture_variable_factor": ("universal", "high", "Variable-star aperture multiplier (dimensionless)."),
    "auto_fwhm_k_factor": ("universal", "high", "Auto-FWHM scale factor relative to measured FWHM."),
    "auto_fwhm_k_min": ("universal", "high", "Auto-FWHM lower clamp as FWHM multiple."),
    "auto_fwhm_k_max": ("universal", "high", "Auto-FWHM upper clamp as FWHM multiple."),
    "cog_isolation_fwhm": ("universal", "high", "COG isolation radius in FWHM units."),
    "cog_ref_fwhm": ("universal", "high", "COG reference FWHM multiplier."),
    "dao_centroid_max_shift_fwhm": ("universal", "high", "Centroid shift limit as FWHM fraction."),
    "frame_quality_fwhm_factor": ("universal", "high", "Frame QC FWHM ratio gate (dimensionless)."),
    "masterstar_use_best_frame_fwhm": ("universal", "high", "Boolean policy; no hardware unit."),
    "neighbor_sub_centroid_max_fwhm": ("universal", "high", "Neighbor-sub centroid limit in FWHM units."),
    "neighbor_sub_refuse_sep_fwhm": ("universal", "high", "Neighbor refusal separation in FWHM units."),
    "nonlinearity_fwhm_ratio": ("universal", "high", "Nonlinearity check uses FWHM ratio."),
    "phase01_comparison_max_fwhm_factor": ("universal", "high", "Comp FWHM ratio limit (dimensionless)."),
    "psf_adaptive_resolve_fwhm": ("universal", "high", "PSF adaptive switch threshold in FWHM units."),
    "psf_group_sep_fwhm": ("universal", "high", "PSF group separation in FWHM units."),
    "psf_neighbor_include_fwhm": ("universal", "high", "PSF neighbor inclusion radius in FWHM units."),
    # --- Ground-truth rig anchors (task brief) ---
    "masterstar_dao_threshold_sigma": ("rig", "high", "Optimal DAO sigma threshold varies with noise/calibration/rig (draft_501 anchor)."),
    "sips_dao_threshold_sigma": ("rig", "high", "Plate-solve DAO threshold varies with field depth and rig sampling."),
    "qc_dao_detection_sigma": ("rig", "high", "QC DAO sigma threshold coupled to rig noise and depth."),
    "alignment_max_control_points": ("rig", "high", "Control-point cap depends on plate scale and field star density."),
    "err_background_mode": ("rig", "high", "Empirical vs Howell term differed on Newton/bin4 vs wide rig (F-BINGAIN-1)."),
    "sigma_sys_mag": ("rig", "high", "Systematic mag floor is per-rig (PROD-SIGMA-FLOOR)."),
    "k2_coeff_v": ("rig", "high", "Second-order extinction k'' is per-rig/site (ROADMAP)."),
    "k2_coeff_r": ("rig", "high", "Second-order extinction k'' is per-rig/site (ROADMAP)."),
    "k2_coeff_i": ("rig", "high", "Second-order extinction k'' is per-rig/site (ROADMAP)."),
    "masterdark_validity_days": ("rig", "high", "Master dark shelf life depends on detector stability and storage."),
    "masterflat_validity_days": ("rig", "high", "Master flat shelf life depends on detector/optics handling."),
    "calibration_library_native_binning": ("rig", "high", "Native binning of calibration library matches detector setup."),
    "bpm_dark_mad_sigma": ("rig", "high", "Hot-pixel MAD sigma depends on dark noise structure per detector."),
    "admission_sat_peak_frac": ("rig", "high", "Saturation fraction gate relative to full-well ADU per detector."),
    "sips_dao_fwhm_px": ("rig", "high", "Initial DAO FWHM guess in pixels depends on seeing sampling."),
    "blind_use_rig_prior": ("rig", "high", "Blind index selection uses rig-specific FOV prior."),
    "osc_channel_binning": ("rig", "high", "OSC channel binning matches camera read mode."),
    "calibration_master_ccd_temp_tolerance_c": ("rig", "high", "CCD temperature tolerance for master matching depends on camera."),
    # --- site / observer ---
    "k2_mode": ("site", "high", "Extinction mode depends on site data availability and atmosphere."),
    "k2_defaults_bprp": ("universal", "high", "Literature bp-rp anchor for k'' defaults (Jordi et al.)."),
    "k2_ceiling": ("universal", "high", "Numerical ceiling on k'' fit (dimensionless guard)."),
    "k2_fit_consistency_sigma": ("universal", "high", "Sigma gate on k'' fit consistency (statistics)."),
    "k2_fit_enabled": ("universal", "high", "Boolean enable for k'' fitting."),
    "k2_fit_lit_factor": ("universal", "high", "Literature blend factor for k'' (dimensionless)."),
    "k2_fit_min_detectability": ("universal", "high", "Minimum detectability for k'' fit (statistics)."),
    "apply_color_term": ("site", "high", "Color-term correction depends on site/filter/transform."),
    # --- session-ish / per-night ---
    "blind_index_select_mode": ("session", "low", "Index set choice depends on FOV and index coverage per run."),
    "blind_img_select_mode": ("session", "low", "Image selection for blind solve depends on run layout."),
    "phase01_flux_col": ("session", "low", "Flux column choice depends on which photometry path succeeded."),
    # --- universal algorithm anchors ---
    "dao_detection_n_equiv": ("universal", "high", "DAO equivalence parameter (sigma-scaled detection statistic)."),
    "comp_max_delta_bprp": ("universal", "high", "Max Gaia bp-rp color difference for comp admission."),
    "phase01_comparison_max_comp_rms": ("universal", "high", "RMS quality gate on comp residuals."),
    "phase01_comparison_max_dist_deg": ("universal", "high", "Angular separation limit in degrees."),
    "phase01_comparison_max_mag_diff": ("universal", "high", "Magnitude difference limit for comp pairing."),
    "phase01_comparison_min_dist_arcsec": ("universal", "high", "Minimum angular separation in arcsec."),
    "phase01_comparison_n_comp_min": ("universal", "high", "Minimum comp count threshold."),
    "gs11_dilution_aperture_arcsec": ("rig", "high", "Physical dilution aperture diameter in arcsec scales with plate scale intent."),
    # --- low-confidence keys resolved to high ---
    "blind_img_select_mode": ("session", "high", "Blind-solve image pick depends on run frame layout."),
    "blind_index_select_mode": ("session", "high", "Index set depends on FOV coverage per run."),
    "blind_verify_early_accept": ("universal", "high", "Early-accept fraction threshold (dimensionless)."),
    "blind_verify_early_floor": ("universal", "high", "Early-accept floor count (integer gate)."),
    "blind_verify_top_n": ("universal", "high", "Top-N match cap (integer)."),
    "comp_color_tiers": ("universal", "high", "Structured color-tier table (algorithm policy)."),
    "comp_contamination_penalty_k": ("universal", "high", "Dimensionless contamination penalty weight."),
    "comp_select_rms_floor": ("universal", "high", "Minimum RMS floor in mmag (statistical gate)."),
    "comp_slope_significance_k": ("universal", "high", "Sigma multiplier for slope significance test."),
    "comp_sparse_fallback_min": ("universal", "high", "Minimum comp count for sparse fallback (integer)."),
    "crowding_comp_availability_loosen_count": ("universal", "high", "Integer loosen trigger for crowding comp pool."),
    "err_empty_apertures_min": ("universal", "high", "Minimum empty apertures for empirical background term (count)."),
    "err_empty_apertures_n": ("universal", "high", "Number of empty apertures sampled (count)."),
    "gs11_comp_suspect_dilution": ("universal", "high", "Dilution ratio threshold (dimensionless)."),
    "masterstar_best_of_n": ("universal", "high", "Best-of-N frame pick count (integer)."),
    "masterstar_catalog_recovery_min": ("universal", "high", "Minimum catalog recovery count (integer)."),
    "masterstar_detection_cap_k": ("universal", "high", "Adaptive detection cap scale factor (dimensionless)."),
    "masterstar_detection_cap_max": ("universal", "high", "Adaptive detection cap maximum (integer)."),
    "masterstar_detection_cap_min": ("universal", "high", "Adaptive detection cap minimum (integer)."),
    "masterstar_sibling_stack_n": ("universal", "high", "Sibling stack frame count (integer)."),
    "neighbor_sub_chi2_max": ("universal", "high", "Chi-squared gate for neighbor subtraction (dimensionless)."),
    "neighbor_sub_nn_contam_dmag": ("universal", "high", "Magnitude-difference gate for neighbor contamination."),
    "neighbor_sub_regime_dmag_min": ("universal", "high", "Regime split on delta-mag (magnitudes)."),
    "neighbor_sub_regime_sep_max": ("universal", "high", "Regime split on separation (FWHM-normalised elsewhere)."),
    "neighbor_sub_residual_rms_max": ("universal", "high", "Residual RMS gate (dimensionless/statistical)."),
    "phase01_flux_col": ("session", "high", "Flux column follows whichever photometry path succeeded this run."),
    "phase01_tiers": ("universal", "high", "Phase-01 tier table structure (algorithm policy)."),
    "temporal_bin_window": ("universal", "high", "Temporal bin width in minutes (time unit, not hardware)."),
    "variability_p85_filter": ("universal", "high", "Percentile filter on variability metric (statistics)."),
    "variability_slope_floor": ("universal", "high", "Slope floor in mag/time (statistical gate)."),
    "variability_smoothness_max": ("universal", "high", "Smoothness metric ceiling (dimensionless)."),
    "hrd_color_saturation": ("universal", "high", "HR diagram display saturation tuning (not detector full-well)."),
    "per_frame_saturation_enabled": ("universal", "high", "Boolean gate enable; not a hardware dimensional value."),
    "temporal_binning_enabled": ("universal", "high", "Boolean enable for temporal binning (not camera binning)."),
    "cal_diag_hard_sigma": ("universal", "high", "Sigma gate on calibration diagnostic statistics."),
    "cal_diag_rel_tol": ("universal", "high", "Relative tolerance on calibration diagnostic ratios."),
    "cal_diag_sat_warn_frac": ("rig", "high", "Saturation warning fraction relative to detector full-well."),
    "vsx_out_of_scope_types": ("universal", "high", "VSX type exclusion list (catalog policy)."),
}

# Substrings forcing rig (pixel/ADU/detector geometry)
_RIG_SUBSTR = (
    "_px", "_adu", "native_binning", "channel_binning",
    "frame_height", "frame_width", "plate_solve_fov",
    "ccd_temp", "_hfr", "cog_ladder_step",
)

# Substrings forcing site
_SITE_SUBSTR = ("extinction", "observer_", "aavso_", "location", "k2_coeff")

# Substrings forcing universal (dimensionless / FWHM-normalised / statistics)
_UNIV_SUBSTR = (
    "_enabled", "_sigma", "_frac", "_fraction", "_ratio", "_factor", "_fwhm",
    "_tol", "_iter", "_sigma_clip", "_clip", "_threshold", "_min_", "_max_",
    "_n_", "_n_comp", "_min_frames", "_min_epochs", "_mode", "_map",
    "_weight", "_alpha", "_beta", "_k_factor", "_polyorder", "_window_frac",
    "_percentile", "_prob", "_timeout", "_budget", "_grid", "_order",
    "_mag", "_mmag", "_pct", "_percent", "_deg", "_mas", "_snr",
)

_BOOL_SUFFIX = ("_enabled",)


def _classify(key: str, entry: dict) -> tuple[str, str, str]:
    if key in EXPLICIT:
        return EXPLICIT[key]

    owner = entry.get("owner", "")
    phase = entry.get("phase", "")
    kind = entry.get("kind", "")
    unit = (entry.get("unit") or "") or ""
    kl = key.lower()
    ftype = pr.appconfig_field_types().get(key, "")

    if owner == "internal" or phase == "paths":
        return ("universal", "high", "Machine paths/plumbing; scope axis not meaningful here.")
    if owner == "db_static":
        return ("site", "high", "DB-owned observatory/site fact.")
    if owner == "fits_dynamic":
        # fall through to explicit; if missing, session default
        return ("session", "high", "fits_dynamic: resolved from FITS/WCS at run time.")

    if kind == "resolved":
        return ("session", "high", "kind=resolved: runtime-derived per run (see param_resolver.py).")

    if unit in ("px", "pixel", "pixels", "ADU", "adu", "e-"):
        return ("rig", "high", f"Registry unit={unit!r} implies detector/scaling dependence.")

    if any(s in kl for s in _RIG_SUBSTR):
        return ("rig", "high", "Key or help implies pixel/ADU/detector geometry dependence.")

    if any(s in kl for s in _SITE_SUBSTR):
        return ("site", "high", "Observatory/site or extinction parameter.")

    if ftype == "bool" or kl.endswith(_BOOL_SUFFIX):
        return ("universal", "high", "Boolean feature flag; no hardware dimensional unit.")

    if phase == "calibration":
        return ("rig", "high", "Calibration master/gate parameters depend on detector/optics.")

    if phase == "extinction":
        return ("site", "high", "Extinction pipeline parameters depend on site atmosphere.")

    if phase == "observer":
        return ("site", "high", "Observer/site metadata.")

    if phase in ("export", "system"):
        return ("universal", "high", f"{phase} phase knobs are global policy/format.")

    if any(s in kl for s in _UNIV_SUBSTR):
        return ("universal", "high", "Dimensionless, FWHM-normalised, or statistical algorithm constant.")

    if phase == "reports":
        return ("universal", "high", "Report rendering/enrichment thresholds (non-science hardware).")

    if phase == "trust":
        return ("universal", "high", "Trust/QC statistical thresholds on light-curve quality.")

    if phase == "alignment":
        if "control_points" in kl or "max_stars" in kl:
            return ("rig", "high", "Alignment resource limits depend on plate scale and star density.")
        return ("universal", "high", "Alignment detection sigma is dimensionless.")

    if phase == "qc":
        if "fwhm" in kl or "hfr" in kl or "background" in kl:
            return ("rig", "low", "QC absolute limits in detector units may need per-rig tuning.")
        return ("universal", "high", "QC sigma/ratio/count gates are dimensionless.")

    if phase in ("detection", "photometry", "comp_selection"):
        if re.search(r"arcsec", kl):
            # angular matching tolerances on sky -> universal; physical apertures -> rig handled in EXPLICIT
            if "match" in kl or "sep" in kl or "radius" in kl or "query" in kl:
                return ("universal", "high", "Angular tolerance on sky (arcsec), not pixel sampling.")
            return ("rig", "low", "Arcsec parameter may encode physical scale tied to plate scale.")
        return ("universal", "low", "Algorithm knob without explicit pixel/ADU unit; review if rig-tuned.")

    return ("universal", "low", "No dimensional dependence identified; default universal pending review.")


def classify_all(registry: dict[str, dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for key, entry in registry.items():
        scope, conf, just = _classify(key, entry)
        e = dict(entry)
        e["scope"] = scope
        e["scope_confidence"] = conf
        out[key] = e
    return out


def main() -> int:
    raw = json.loads(REG_PATH.read_text(encoding="utf-8"))
    meta = raw.get("__meta__", {})
    reg = {k: v for k, v in raw.items() if not k.startswith("__")}
    classified = classify_all(reg)

    anchors_rig = [
        "masterstar_dao_threshold_sigma", "alignment_max_control_points",
        "err_background_mode", "sigma_sys_mag", "masterdark_validity_days",
        "calibration_library_native_binning", "bpm_dark_mad_sigma",
    ]
    for a in anchors_rig:
        if a in classified and classified[a]["scope"] != "rig":
            print(f"ERROR: anchor {a} got {classified[a]['scope']}", file=sys.stderr)
            sys.exit(1)

    c = Counter(e["scope"] for e in classified.values())
    lc = [k for k, e in classified.items() if e["scope_confidence"] == "low"]
    print("Scope distribution:", dict(c))
    print(f"Low confidence: {len(lc)}")
    if lc:
        print(" ", ", ".join(sorted(lc)))

    if "--write" in sys.argv:
        new_raw = {"__meta__": meta, **classified}
        REG_PATH.write_text(json.dumps(new_raw, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        print(f"Wrote {REG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
