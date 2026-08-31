"""G2-F002b: per-epoch catalog_match_mode trust classification (flag-only; no photometry change)."""

from __future__ import annotations

# Exported per-frame modes from pipeline export / per_frame_catalog_index.
TRUSTED_CATALOG_MATCH_MODES = frozenset(
    {
        "master_reference_sky",
        "sky",
        "full_cone",
        "master_reference_locked",
    }
)
UNTRUSTED_FLUX_CATALOG_MATCH_MODES = frozenset({"master_reference_pixel"})
NONDET_CATALOG_MATCH_MODES = frozenset(
    {
        "nondet_no_wcs",
        "nondet_unaligned_sky",
    }
)

# Internal _detect_stars_match_masterstars match_mode ? exported catalog_match_mode.
INTERNAL_MATCH_MODE_TO_EXPORT = {
    "sky": "master_reference_sky",
    "sky_unaligned_no_pixel_fallback": "nondet_unaligned_sky",
    "pixel_fallback_bad_wcs": "master_reference_pixel",
    "nondet_unaligned_no_wcs": "nondet_no_wcs",
    "pixel_fallback_no_wcs": "master_reference_pixel",
}


def normalize_catalog_match_mode(mode: str | None) -> str:
    return str(mode or "").strip()


def export_catalog_match_mode_from_internal(match_mode: str) -> str:
    """Map internal detect_stars match_mode to exported catalog_match_mode string."""
    key = str(match_mode or "").strip()
    if key in INTERNAL_MATCH_MODE_TO_EXPORT:
        return INTERNAL_MATCH_MODE_TO_EXPORT[key]
    if key.startswith("pixel"):
        return "master_reference_pixel"
    return "master_reference_sky"


def is_wcs_untrusted_catalog_match_mode(mode: str | None) -> bool:
    """True when flux was matched via pixel fallback (G2-F002b UNTRUSTED-FLUX)."""
    return normalize_catalog_match_mode(mode) in UNTRUSTED_FLUX_CATALOG_MATCH_MODES


