"""Scope classifier for params_registry.json (Task C / C').

Mechanical rules + explicit overrides. Validates EXPLICIT keys exist in registry.
Run: python dev/tools/classify_params_scope.py --write
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src_py"))
import params_registry as pr  # noqa: E402

REG_PATH = ROOT / "dev" / "validation" / "params_registry.json"

# scope, scope_key, scope_group (n/a for non-rig), confidence, internal note (not written to JSON)
class ScopeResult(NamedTuple):
    scope: str
    scope_key: str
    scope_group: str
    confidence: str
    note: str


def validate_explicit_keys(registry: dict[str, dict]) -> None:
    """Fail loudly if EXPLICIT names keys absent from the registry."""
    dead = sorted(k for k in EXPLICIT if k not in registry)
    if dead:
        raise KeyError(
            "EXPLICIT override keys absent from registry (fix or remove):\n  "
            + "\n  ".join(dead)
        )


# Explicit overrides: ground-truth anchors, C'-3 reclassifications, resolver alignment.
# scope_group: a=genuine per-rig physics, b=unit artefact, c=operational tuning, n/a=non-rig
EXPLICIT: dict[str, ScopeResult] = {
    # --- resolved / fits_dynamic ---
    "gain": ScopeResult("rig", "rig", "a", "high", "Equipment-intrinsic e-/ADU (param_resolver)."),
    "read_noise": ScopeResult("rig", "rig", "a", "high", "Equipment-intrinsic e- (param_resolver)."),
    "plate_scale_arcsec_per_px": ScopeResult("session", "frame", "n/a", "high", "Plate scale from WCS per exposure."),
    "phase01_plate_scale_arcsec_per_px": ScopeResult("session", "frame", "n/a", "high", "Phase-01 plate scale from WCS."),
    "export_arcsec_per_px": ScopeResult("session", "frame", "n/a", "high", "Export plate scale from WCS."),
    "plate_solve_fov_deg": ScopeResult("rig", "rig", "a", "high", "Blind-solve FOV seed is telescope+camera property."),
    "frame_height_px": ScopeResult("session", "frame", "n/a", "high", "WAVE-B: measured from FITS NAXIS2 per frame; not stored per rig."),
    "frame_width_px": ScopeResult("session", "frame", "n/a", "high", "WAVE-B: measured from FITS NAXIS1 per frame; not stored per rig."),
    "qc_preprocess_workers": ScopeResult("universal", "none", "n/a", "high", "Internal plumbing worker cap."),
    # --- FWHM-normalised universal ---
    "annulus_inner_fwhm": ScopeResult("universal", "none", "n/a", "high", "FWHM multiple."),
    "annulus_outer_fwhm": ScopeResult("universal", "none", "n/a", "high", "FWHM multiple."),
    "aperture_fwhm_factor": ScopeResult("universal", "none", "n/a", "high", "FWHM multiple."),
    "aperture_snr_sizing": ScopeResult("universal", "none", "n/a", "high", "FWHM-normalised SNR sizing."),
    "aperture_comp_factor": ScopeResult("universal", "none", "n/a", "high", "Dimensionless aperture ratio."),
    "aperture_variable_factor": ScopeResult("universal", "none", "n/a", "high", "Dimensionless multiplier."),
    "auto_fwhm_k_factor": ScopeResult("universal", "none", "n/a", "high", "FWHM scale factor."),
    "auto_fwhm_k_min": ScopeResult("universal", "none", "n/a", "high", "FWHM clamp."),
    "auto_fwhm_k_max": ScopeResult("universal", "none", "n/a", "high", "FWHM clamp."),
    "cog_isolation_fwhm": ScopeResult("universal", "none", "n/a", "high", "FWHM isolation radius."),
    "cog_ref_fwhm": ScopeResult("universal", "none", "n/a", "high", "FWHM reference."),
    "dao_centroid_max_shift_fwhm": ScopeResult("universal", "none", "n/a", "high", "FWHM fraction."),
    "frame_quality_fwhm_factor": ScopeResult("universal", "none", "n/a", "high", "Dimensionless FWHM ratio."),
    "masterstar_use_best_frame_fwhm": ScopeResult("universal", "none", "n/a", "high", "Boolean policy."),
    "neighbor_sub_centroid_max_fwhm": ScopeResult("universal", "none", "n/a", "high", "FWHM units."),
    "neighbor_sub_refuse_sep_fwhm": ScopeResult("universal", "none", "n/a", "high", "FWHM separation."),
    "nonlinearity_fwhm_ratio": ScopeResult("universal", "none", "n/a", "high", "FWHM ratio."),
    "phase01_comparison_max_fwhm_factor": ScopeResult("universal", "none", "n/a", "high", "Dimensionless FWHM ratio."),
    "psf_adaptive_resolve_fwhm": ScopeResult("universal", "none", "n/a", "high", "FWHM threshold."),
    "psf_group_sep_fwhm": ScopeResult("universal", "none", "n/a", "high", "FWHM separation."),
    "psf_neighbor_include_fwhm": ScopeResult("universal", "none", "n/a", "high", "FWHM inclusion radius."),
    # --- C'-3 reclassifications ---
    "k2_defaults_bprp": ScopeResult(
        "rig", "rig_band", "a", "low",
        "Per-band k'' depends on filter x detector QE (rig) and atmosphere (site); flat dict cannot hold per-rig.",
    ),
    "apply_color_term": ScopeResult(
        "rig", "rig_band", "a", "high",
        "Color-term transform is filter x detector response, not observing location.",
    ),
    "phase01_comparison_max_dist_deg": ScopeResult(
        "rig", "rig", "b", "low",
        "Additive margin on FOV-derived base (config.py:2764,2796,2828); D1 target: fraction of resolved FOV.",
    ),
    "phase01_comparison_min_dist_arcsec": ScopeResult(
        "universal", "none", "n/a", "high",
        "Consumed directly in arcsec (photometry_core.py:14798); arcsec is rig-independent.",
    ),
    "gs11_dilution_aperture_arcsec": ScopeResult(
        "universal", "none", "n/a", "low",
        "Angular aperture on sky; 0 derives from photometric aperture -- not rig-scoped when expressed in arcsec.",
    ),
    # --- group (a) rig physics ---
    "err_background_mode": ScopeResult("rig", "rig", "a", "high", "F-BINGAIN-1: empirical vs Howell differed Newton/bin4 vs wide."),
    "sigma_sys_mag": ScopeResult("rig", "rig", "a", "high", "PROD-SIGMA-FLOOR; keyed on equipment_id in sigma_floor_core."),
    "bpm_dark_mad_sigma": ScopeResult("rig", "rig", "a", "high", "Hot-pixel MAD depends on dark noise per detector."),
    "calibration_library_native_binning": ScopeResult("rig", "rig", "a", "high", "Defines native binning; input to rig_sampling, not resolved by it."),
    "calibration_master_ccd_temp_tolerance_c": ScopeResult("rig", "rig", "a", "high", "CCD temp tolerance for master match."),
    "osc_channel_binning": ScopeResult("rig", "rig", "a", "high", "Defines OSC sampling (2 x N superpixel); input to rig_sampling."),
    "admission_sat_peak_frac": ScopeResult("rig", "rig", "a", "high", "Saturation fraction vs detector full-well."),
    "saturate_limit_fraction": ScopeResult("rig", "rig", "a", "high", "Saturation fraction vs detector full-well."),
    "cal_diag_sat_warn_frac": ScopeResult("rig", "rig", "a", "high", "Saturation warning vs full-well ADU."),
    "blind_use_rig_prior": ScopeResult("rig", "rig", "a", "high", "Blind index selection uses rig FOV prior."),
    # --- group (b) unit artefacts ---
    "blind_verify_match_tol_px": ScopeResult("rig", "rig_sampling", "b", "low", "Target unit: arcsec via resolved plate scale."),
    "cog_ladder_step_px": ScopeResult("rig", "rig_sampling", "b", "low", "Target unit: FWHM multiples for aperture ladder step."),
    "crowding_tighten_min_fwhm_px": ScopeResult(
        "rig", "rig_sampling", "a", "high",
        "Undersampling gate in px; sampling is a pixel-domain property -- not group (b).",
    ),
    "hrd_color_bg_box_px": ScopeResult("rig", "rig_sampling", "b", "low", "Target unit: arcsec or field fraction for HR diagram crop."),
    "masterstar_centre_rms_max_px": ScopeResult("rig", "rig_sampling", "b", "low", "Target unit: arcsec via plate scale."),
    "masterstar_sibling_rms_max_px": ScopeResult("rig", "rig_sampling", "b", "low", "Target unit: arcsec via plate scale."),
    "phase01_chip_interior_margin_px": ScopeResult("rig", "rig_sampling", "b", "low", "PHASE0-BORDER-MARGIN-GEOMETRY: derive from aperture+annulus FWHM."),
    "phase01_comparison_isolation_radius_px": ScopeResult("rig", "rig_sampling", "b", "low", "Target unit: arcsec; twin of min_dist_arcsec."),
    "qc_max_hfr": ScopeResult("rig", "rig_sampling", "b", "low", "Target unit: FWHM-normalised HFR ratio."),
    "sips_dao_fwhm_px": ScopeResult("rig", "rig_sampling", "b", "low", "Target unit: FWHM multiples for initial DAO guess."),
    # --- group (c) operational tuning ---
    "alignment_max_control_points": ScopeResult("rig", "rig_sampling", "c", "high", "chi/h Persei tuning; performance not science correctness."),
    "masterdark_validity_days": ScopeResult("rig", "rig", "c", "low", "Staleness warning threshold; detector storage habit."),
    "masterflat_validity_days": ScopeResult("rig", "rig", "c", "low", "Staleness warning threshold; optics handling."),
    "masterstar_dao_threshold_sigma": ScopeResult(
        "rig", "rig_sampling", "a", "low",
        "Dimensionless sigma; optimal numeric value depends on sampling/depth -- group (a) not (c) because detection completeness.",
    ),
    "sips_dao_threshold_sigma": ScopeResult("rig", "rig_sampling", "a", "low", "Same physics as masterstar_dao_threshold_sigma."),
    "qc_dao_detection_sigma": ScopeResult("rig", "rig_sampling", "a", "low", "Same physics as masterstar_dao_threshold_sigma."),
    # --- site / extinction policy ---
    "k2_mode": ScopeResult("site", "site", "n/a", "high", "Extinction mode depends on site data/atmosphere."),
    "k2_ceiling": ScopeResult("universal", "none", "n/a", "high", "Numerical ceiling on k'' fit."),
    "k2_fit_consistency_sigma": ScopeResult("universal", "none", "n/a", "high", "Sigma gate on k'' fit consistency."),
    "k2_fit_enabled": ScopeResult("universal", "none", "n/a", "high", "Boolean enable."),
    "k2_fit_lit_factor": ScopeResult("universal", "none", "n/a", "high", "Literature blend factor."),
    "k2_fit_min_detectability": ScopeResult("universal", "none", "n/a", "high", "Minimum detectability statistic."),
    # --- session ---
    "blind_index_select_mode": ScopeResult("session", "frame", "n/a", "low", "Index set depends on FOV coverage per run."),
    "blind_img_select_mode": ScopeResult("session", "frame", "n/a", "low", "Image pick depends on run frame layout."),
    "phase01_flux_col": ScopeResult("session", "frame", "n/a", "low", "Flux column follows photometry path that succeeded."),
    # --- universal algorithm anchors ---
    "dao_detection_n_equiv": ScopeResult("universal", "none", "n/a", "high", "Dimensionless DAO equivalence."),
    "comp_max_delta_bprp": ScopeResult("universal", "none", "n/a", "high", "Gaia bp-rp color gate (mag)."),
    "phase01_comparison_max_comp_rms": ScopeResult("universal", "none", "n/a", "high", "RMS quality gate."),
    "phase01_comparison_max_mag_diff": ScopeResult("universal", "none", "n/a", "high", "Magnitude difference limit."),
    "phase01_comparison_n_comp_min": ScopeResult("universal", "none", "n/a", "high", "Minimum comp count."),
    # --- keys promoted from mechanical low to explicit ---
    "blind_verify_early_accept": ScopeResult("universal", "none", "n/a", "high", "Dimensionless early-accept fraction."),
    "blind_verify_early_floor": ScopeResult("universal", "none", "n/a", "high", "Integer floor count."),
    "blind_verify_top_n": ScopeResult("universal", "none", "n/a", "high", "Integer top-N cap."),
    "comp_color_tiers": ScopeResult("universal", "none", "n/a", "high", "Tier table structure."),
    "comp_contamination_penalty_k": ScopeResult("universal", "none", "n/a", "high", "Dimensionless penalty weight."),
    "comp_select_rms_floor": ScopeResult("universal", "none", "n/a", "high", "mmag RMS floor."),
    "comp_slope_significance_k": ScopeResult("universal", "none", "n/a", "high", "Sigma multiplier."),
    "comp_sparse_fallback_min": ScopeResult("universal", "none", "n/a", "high", "Integer comp count."),
    "crowding_comp_availability_loosen_count": ScopeResult("universal", "none", "n/a", "high", "Integer trigger."),
    "err_empty_apertures_min": ScopeResult("universal", "none", "n/a", "high", "Integer count."),
    "err_empty_apertures_n": ScopeResult("universal", "none", "n/a", "high", "Integer count."),
    "gs11_comp_suspect_dilution": ScopeResult("universal", "none", "n/a", "high", "Dimensionless dilution ratio."),
    "masterstar_best_of_n": ScopeResult("universal", "none", "n/a", "high", "Integer frame count."),
    "masterstar_catalog_recovery_min": ScopeResult("universal", "none", "n/a", "high", "Integer recovery count."),
    "masterstar_detection_cap_k": ScopeResult("universal", "none", "n/a", "high", "Dimensionless cap scale."),
    "masterstar_detection_cap_max": ScopeResult("universal", "none", "n/a", "high", "Integer cap max."),
    "masterstar_detection_cap_min": ScopeResult("universal", "none", "n/a", "high", "Integer cap min."),
    "masterstar_sibling_stack_n": ScopeResult("universal", "none", "n/a", "high", "Integer stack count."),
    "neighbor_sub_chi2_max": ScopeResult("universal", "none", "n/a", "high", "Chi-squared gate."),
    "neighbor_sub_nn_contam_dmag": ScopeResult("universal", "none", "n/a", "high", "Delta-mag gate."),
    "neighbor_sub_regime_dmag_min": ScopeResult("universal", "none", "n/a", "high", "Delta-mag regime split."),
    "neighbor_sub_regime_sep_max": ScopeResult("universal", "none", "n/a", "high", "Separation regime split."),
    "neighbor_sub_residual_rms_max": ScopeResult("universal", "none", "n/a", "high", "Residual RMS gate."),
    "phase01_tiers": ScopeResult("universal", "none", "n/a", "high", "Tier table structure."),
    "temporal_bin_window": ScopeResult("universal", "none", "n/a", "high", "Time window in minutes."),
    "variability_p85_filter": ScopeResult("universal", "none", "n/a", "high", "Percentile filter."),
    "variability_slope_floor": ScopeResult("universal", "none", "n/a", "high", "Slope floor mag/time."),
    "variability_smoothness_max": ScopeResult("universal", "none", "n/a", "high", "Smoothness ceiling."),
    "hrd_color_saturation": ScopeResult("universal", "none", "n/a", "high", "Display chroma saturation."),
    "per_frame_saturation_enabled": ScopeResult("universal", "none", "n/a", "high", "Boolean gate enable."),
    "temporal_binning_enabled": ScopeResult("universal", "none", "n/a", "high", "Boolean temporal binning."),
    "cal_diag_hard_sigma": ScopeResult("universal", "none", "n/a", "high", "Sigma gate on cal diagnostic."),
    "cal_diag_rel_tol": ScopeResult("universal", "none", "n/a", "high", "Relative tolerance ratio."),
    "vsx_out_of_scope_types": ScopeResult("universal", "none", "n/a", "high", "VSX type exclusion list."),
}

_RIG_SUBSTR = (
    "_px", "_adu", "native_binning", "channel_binning",
    "frame_height", "frame_width", "plate_solve_fov", "ccd_temp", "_hfr", "cog_ladder_step",
)
_SITE_SUBSTR = ("extinction", "observer_", "aavso_", "location")
_UNIV_SUBSTR = (
    "_enabled", "_sigma", "_frac", "_fraction", "_ratio", "_factor", "_fwhm",
    "_tol", "_iter", "_clip", "_threshold", "_min_", "_max_", "_n_", "_n_comp",
    "_min_frames", "_min_epochs", "_mode", "_map", "_weight", "_alpha", "_beta",
    "_k_factor", "_polyorder", "_window_frac", "_percentile", "_prob", "_timeout",
    "_budget", "_grid", "_order", "_mag", "_mmag", "_pct", "_percent", "_deg", "_mas", "_snr",
)


def _scope_key_for(scope: str, key: str) -> str:
    if scope == "universal":
        return "none"
    if scope == "site":
        return "site"
    if scope == "session":
        return "frame"
    # rig default
    if "band" in key or key in ("apply_color_term", "k2_defaults_bprp"):
        return "rig_band"
    if any(s in key for s in ("_px", "binning", "fwhm_px", "hfr", "sampling", "fov")):
        return "rig_sampling"
    return "rig"


def _rig_group_for(key: str, scope_key: str) -> str:
    if key in EXPLICIT:
        return EXPLICIT[key].scope_group
    if scope_key == "rig_band":
        return "a"
    if scope_key == "rig_sampling":
        return "b"
    return "a"


def _classify_mechanical(key: str, entry: dict) -> ScopeResult:
    owner = entry.get("owner", "")
    phase = entry.get("phase", "")
    kind = entry.get("kind", "")
    unit = (entry.get("unit") or "") or ""
    kl = key.lower()
    ftype = pr.appconfig_field_types().get(key, "")

    if owner == "internal" or phase == "paths":
        return ScopeResult("universal", "none", "n/a", "high", "Plumbing/paths.")
    if owner == "db_static":
        return ScopeResult("site", "site", "n/a", "high", "DB site fact.")
    if owner == "fits_dynamic":
        if key in ("gain", "read_noise"):
            return ScopeResult("rig", "rig", "a", "high", "Equipment-intrinsic despite fits_dynamic owner.")
        if key == "plate_solve_fov_deg":
            return ScopeResult("rig", "rig", "a", "high", "FOV seed is rig property.")
        return ScopeResult("session", "frame", "n/a", "low", "fits_dynamic without explicit override.")
    if kind == "resolved" and key not in EXPLICIT:
        return ScopeResult("session", "frame", "n/a", "low", "kind=resolved fallback.")
    if unit in ("px", "pixel", "pixels", "ADU", "adu", "e-"):
        sk = _scope_key_for("rig", key)
        return ScopeResult("rig", sk, _rig_group_for(key, sk), "low", f"Registry unit={unit!r}.")
    if any(s in kl for s in _RIG_SUBSTR):
        sk = _scope_key_for("rig", key)
        return ScopeResult("rig", sk, _rig_group_for(key, sk), "low", "Key substring implies rig dependence.")
    if any(s in kl for s in _SITE_SUBSTR):
        return ScopeResult("site", "site", "n/a", "low", "Key substring implies site.")
    if ftype == "bool" or kl.endswith("_enabled"):
        return ScopeResult("universal", "none", "n/a", "high", "Boolean flag.")
    if phase == "calibration":
        return ScopeResult("rig", "rig", "a", "low", "Calibration phase default.")
    if phase == "extinction" and key not in EXPLICIT:
        return ScopeResult("site", "site", "n/a", "low", "Extinction phase default.")
    if phase == "observer":
        return ScopeResult("site", "site", "n/a", "high", "Observer phase.")
    if phase in ("export", "system"):
        return ScopeResult("universal", "none", "n/a", "high", f"{phase} policy.")
    if any(s in kl for s in _UNIV_SUBSTR):
        return ScopeResult("universal", "none", "n/a", "high", "Dimensionless/FWHM-normalised/statistical.")
    if phase == "reports":
        return ScopeResult("universal", "none", "n/a", "high", "Report rendering threshold.")
    if phase == "trust":
        return ScopeResult("universal", "none", "n/a", "high", "Trust/QC statistical threshold.")
    if phase == "alignment":
        if "control_points" in kl or "max_stars" in kl:
            return ScopeResult("rig", "rig_sampling", "c", "low", "Alignment resource limit.")
        return ScopeResult("universal", "none", "n/a", "low", "Alignment sigma default.")
    if phase == "qc":
        if "fwhm" in kl or "hfr" in kl:
            sk = "rig_sampling"
            return ScopeResult("rig", sk, "b", "low", "QC limit in detector units.")
        return ScopeResult("universal", "none", "n/a", "low", "QC dimensionless default.")
    if phase in ("detection", "photometry", "comp_selection"):
        if re.search(r"arcsec", kl):
            if "match" in kl or "sep" in kl or "query" in kl:
                return ScopeResult("universal", "none", "n/a", "high", "Angular tolerance on sky.")
            return ScopeResult("rig", "rig_sampling", "b", "low", "Arcsec param with possible plate-scale coupling.")
        return ScopeResult("universal", "none", "n/a", "high", "Algorithm knob without pixel/ADU unit.")
    return ScopeResult("universal", "none", "n/a", "low", "Unclassified default.")


def classify_key(key: str, entry: dict) -> ScopeResult:
    if key in EXPLICIT:
        return EXPLICIT[key]
    return _classify_mechanical(key, entry)


def classify_all(registry: dict[str, dict]) -> dict[str, dict]:
    validate_explicit_keys(registry)
    out: dict[str, dict] = {}
    for key, entry in registry.items():
        r = classify_key(key, entry)
        e = dict(entry)
        e["scope"] = r.scope
        e["scope_key"] = r.scope_key
        e["scope_group"] = r.scope_group
        e["scope_confidence"] = r.confidence
        out[key] = e
    return out


def main() -> int:
    raw = json.loads(REG_PATH.read_text(encoding="utf-8"))
    meta = raw.get("__meta__", {})
    reg = {k: v for k, v in raw.items() if not k.startswith("__")}
    classified = classify_all(reg)

    c = Counter(e["scope"] for e in classified.values())
    lc = [k for k, e in classified.items() if e["scope_confidence"] == "low"]
    print("Scope distribution:", dict(c))
    print(f"Low confidence: {len(lc)}")
    rg = Counter(e["scope_group"] for e in classified.values() if e["scope"] == "rig")
    print("Rig groups:", dict(rg))

    if "--write" in sys.argv:
        # Fix sigma_sys_mag help (C'-3)
        if "sigma_sys_mag" in classified:
            classified["sigma_sys_mag"]["help"] = (
                "Per-equipment systematic error floor (mag) added in quadrature to statistical "
                "errors; dict keys are equipment_id strings (e.g. {'3': 0.018}), not filter band "
                "numbers. Lookup: resolve_sigma_sys_mag in sigma_floor_core.py."
            )
        new_raw = {"__meta__": meta, **classified}
        REG_PATH.write_text(json.dumps(new_raw, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        print(f"Wrote {REG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
