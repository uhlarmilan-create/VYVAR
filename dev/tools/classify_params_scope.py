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
    "qc_preprocess_workers": ScopeResult("universal", "none", "n/a", "high", "Internal plumbing worker cap."),
    # --- FWHM-normalised universal ---
    "annulus_inner_fwhm": ScopeResult("universal", "none", "n/a", "high", "FWHM multiple."),
    "annulus_outer_fwhm": ScopeResult("universal", "none", "n/a", "high", "FWHM multiple."),
    "aperture_fwhm_factor": ScopeResult("universal", "none", "n/a", "high", "FWHM multiple."),
    "auto_fwhm_k_factor": ScopeResult("universal", "none", "n/a", "high", "FWHM scale factor."),
    # --- C'-3 reclassifications ---
    "k2_defaults_bprp": ScopeResult(
        "rig", "rig_band", "a", "low",
        "Per-band k'' depends on filter x detector QE (rig) and atmosphere (site); flat dict cannot hold per-rig.",
    ),
    "apply_color_term": ScopeResult(
        "rig", "rig_band", "a", "high",
        "Color-term transform is filter x detector response, not observing location.",
    ),
    # --- group (a) rig physics ---
    "calibration_master_ccd_temp_tolerance_c": ScopeResult("rig", "rig", "a", "high", "CCD temp tolerance for master match."),
    "osc_channel_binning": ScopeResult("rig", "rig", "a", "high", "Defines OSC sampling (2 x N superpixel); input to rig_sampling."),
    # --- group (b) pixel parameters (CFG-UNITS-2026-10: one key per meaning, pixels) ---
    # --- group (c) operational tuning ---
    "masterdark_validity_days": ScopeResult("rig", "rig", "c", "low", "Staleness warning threshold; detector storage habit."),
    "masterflat_validity_days": ScopeResult("rig", "rig", "c", "low", "Staleness warning threshold; optics handling."),
    "detection_threshold_sigma": ScopeResult(
        "rig", "rig_sampling", "a", "low",
        "Dimensionless sigma; optimal numeric value depends on sampling/depth -- group (a) not (c) because detection completeness.",
    ),
    # --- site / extinction policy ---
    "k2_mode": ScopeResult("site", "site", "n/a", "high", "Extinction mode depends on site data/atmosphere."),
    # --- session ---
    # --- universal algorithm anchors ---
    "comp_max_delta_bprp": ScopeResult("universal", "none", "n/a", "high", "Gaia bp-rp color gate (mag)."),
    "phase01_comparison_max_comp_rms": ScopeResult("universal", "none", "n/a", "high", "RMS quality gate."),
    "phase01_comparison_n_comp_min": ScopeResult("universal", "none", "n/a", "high", "Minimum comp count."),
    # --- keys promoted from mechanical low to explicit ---
    "comp_color_tiers": ScopeResult("universal", "none", "n/a", "high", "Tier table structure."),
    "masterstar_best_of_n": ScopeResult("universal", "none", "n/a", "high", "Integer frame count."),
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
        new_raw = {"__meta__": meta, **classified}
        REG_PATH.write_text(json.dumps(new_raw, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        print(f"Wrote {REG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
