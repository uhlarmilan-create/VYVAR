"""Add D1 normalised companion fields to params_registry.json."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REG = ROOT / "dev" / "validation" / "params_registry.json"

_UNIV = {
    "scope": "universal",
    "scope_key": "none",
    "scope_group": "n/a",
    "scope_confidence": "high",
}

NEW_FIELDS: dict[str, dict] = {
    "blind_verify_match_tol_arcsec": {
        "help": "Blind verify match tolerance in arcsec. None uses legacy blind_verify_match_tol_px.",
        "kind": "static",
        "label": "Blind Verify Match Tol Arcsec",
        "owner": "config_runtime",
        "phase": "detection",
        "range": None,
        "tier": "expert",
        "unit": "arcsec",
        "widget": "hidden",
        **_UNIV,
    },
    "cog_ladder_step_fwhm": {
        "help": "COG ladder step as multiple of measured FWHM. None uses legacy cog_ladder_step_px.",
        "kind": "static",
        "label": "COG Ladder Step FWHM",
        "owner": "config_runtime",
        "phase": "photometry",
        "range": None,
        "tier": "expert",
        "unit": None,
        "widget": "hidden",
        **_UNIV,
    },
    "hrd_color_bg_box_arcsec": {
        "help": "HR diagram local background box in arcsec. None uses legacy hrd_color_bg_box_px.",
        "kind": "static",
        "label": "HRD Color BG Box Arcsec",
        "owner": "config_runtime",
        "phase": "reports",
        "range": None,
        "tier": "expert",
        "unit": "arcsec",
        "widget": "hidden",
        **_UNIV,
    },
    "masterstar_centre_rms_max_arcsec": {
        "help": "MASTERSTAR centre RMS gate in arcsec. None uses legacy masterstar_centre_rms_max_px.",
        "kind": "static",
        "label": "Masterstar Centre RMS Max Arcsec",
        "owner": "config_runtime",
        "phase": "detection",
        "range": None,
        "tier": "expert",
        "unit": "arcsec",
        "widget": "hidden",
        **_UNIV,
    },
    "masterstar_sibling_rms_max_arcsec": {
        "help": "Sibling recovery RMS gate in arcsec. None uses legacy masterstar_sibling_rms_max_px.",
        "kind": "static",
        "label": "Masterstar Sibling RMS Max Arcsec",
        "owner": "config_runtime",
        "phase": "detection",
        "range": None,
        "tier": "expert",
        "unit": "arcsec",
        "widget": "hidden",
        **_UNIV,
    },
    "phase01_chip_interior_margin_arcsec": {
        "help": "Chip interior margin in arcsec. None uses legacy phase01_chip_interior_margin_px.",
        "kind": "static",
        "label": "Phase01 Chip Interior Margin Arcsec",
        "owner": "config_runtime",
        "phase": "comp_selection",
        "range": None,
        "tier": "expert",
        "unit": "arcsec",
        "widget": "hidden",
        **_UNIV,
    },
    "phase01_comparison_isolation_radius_arcsec": {
        "help": "Comp isolation radius in arcsec. None uses legacy phase01_comparison_isolation_radius_px.",
        "kind": "static",
        "label": "Phase01 Comparison Isolation Radius Arcsec",
        "owner": "config_runtime",
        "phase": "comp_selection",
        "range": None,
        "tier": "expert",
        "unit": "arcsec",
        "widget": "hidden",
        **_UNIV,
    },
    "phase01_comparison_max_dist_fov_frac": {
        "help": "Max comp distance as fraction of half-diagonal FOV. None uses legacy max_dist_deg fallback.",
        "kind": "static",
        "label": "Phase01 Comparison Max Dist FOV Frac",
        "owner": "config_runtime",
        "phase": "comp_selection",
        "range": None,
        "tier": "expert",
        "unit": None,
        "widget": "hidden",
        **_UNIV,
    },
    "qc_max_hfr_fwhm_ratio": {
        "help": "QC HFR limit as multiple of measured FWHM. None uses legacy qc_max_hfr px cap.",
        "kind": "static",
        "label": "QC Max HFR FWHM Ratio",
        "owner": "config_runtime",
        "phase": "qc",
        "range": None,
        "tier": "expert",
        "unit": None,
        "widget": "hidden",
        **_UNIV,
    },
    "sips_dao_fwhm_fwhm_factor": {
        "help": "Initial DAO FWHM as multiple of measured FWHM. None uses legacy sips_dao_fwhm_px.",
        "kind": "static",
        "label": "SIPS DAO FWHM Factor",
        "owner": "config_runtime",
        "phase": "detection",
        "range": None,
        "tier": "expert",
        "unit": None,
        "widget": "hidden",
        **_UNIV,
    },
}

raw = json.loads(REG.read_text(encoding="utf-8"))
for k, v in NEW_FIELDS.items():
    if k in raw:
        raise SystemExit(f"already exists: {k}")
    raw[k] = v
REG.write_text(json.dumps(raw, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
print(f"Added {len(NEW_FIELDS)} registry entries")
