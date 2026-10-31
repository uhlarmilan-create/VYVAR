"""Photometry names: logger, proc / LC column names, labels and key sets (leaf module).

LAYER-2026-10 (code review 2 T13): the photometry modules import these from here, not from the
``photometry_core`` facade (that import made 12 modules unimportable on their own). Leaf rule as
``pipeline_constants``: this module imports nothing from VYVAR; anything may import it.
Physical constants live in ``pipeline_constants`` (D-CONSTANTS-LEAF-01).
"""

from __future__ import annotations

import logging

#: One logger for the photometry stage; the name is kept so log routing is unchanged.
LOGGER = logging.getLogger("photometry_core")

# Explicit annulus sky (ADU/px) for the Howell error.
SKY_ADU_PER_PX_ANNULUS_COL = "sky_adu_per_px_annulus"
#: CCD-ONE-2026-10: number of sky pixels of that sky median (n_sky of the CCD equation, ``ccd_noise``).
SKY_N_PX_COL = "sky_n_px"

# F-BINGAIN-1: empirical background noise (empty-aperture scatter) + provenance.
SIGMA_BKG_AP_COL = "sigma_bkg_ap"
ERR_BKG_SOURCE_COL = "err_bkg_source"
ERR_BKG_SOURCE_EMPIRICAL = "empirical"
ERR_BKG_SOURCE_HOWELL_FALLBACK = "howell_fallback"
ERR_BKG_SOURCE_HOWELL_SCALED = "howell_scaled"

# Per-target LC time provenance (F-BJD-1): labels the BJD recompute path, does not alter time values.
TIME_BASE_COL = "time_base"
TIME_BASE_BJD_TDB = "BJD_TDB"
TIME_BASE_JD_FALLBACK = "JD_FALLBACK"

COMP_QUALITY_JSON_META_KEYS = frozenset(
    {
        "selected_tier",
        "tier4_warning",
        "n_tier1",
        "n_tier2",
        "n_tier3",
        "n_tier4",
        "aperture_correction",  # legacy key in pre-2026-10 comp_quality JSON (AC removed)
        "qa_degraded",
        "qa_degraded_reason",
        "broeg_weights",  # EXPORT-2026-10: normalised Broeg (2005) ensemble weights
        "comp_band_mag",  # catalogue magnitudes in the zero-point band
        "band_system",
        "zp_band_label",
        "zero_point",  # ZP-ERR-2026-10: comps setting the absolute level + its uncertainty
        "gspc_zero_point",
        "comp_smart",
        "check_star",  # OUTPUT-REVIEW-2026-10 step 6: the check star and its trust statistics
        "color_fallback",  # legacy (tier copies before OUTPUT-REVIEW step 6)
        "max_delta_bprp_used",  # legacy
    }
)

LC_QUALITY_FLAGS: tuple[str, ...] = (
    "good",
    "noisy",
    "noisy_moon",
    "short_baseline",
    "no_data",
    "saturated",
)
#: G08-04: light-curve qualities that are information only (trust not lowered): "noisy" and its moonlit
#: variant are the same verdict (decisions 3632 / 3654); "noisy_moon" was a hard RED by omission.
LC_QUALITY_INFORMATIONAL: frozenset[str] = frozenset({"good", "noisy", "noisy_moon"})

#: TARGET-RULES-2026-10 (Milan 2026-10-13): target skips decided on MASTERSTAR in target
#: selection (saturated = upper cut, below_snr_min = lower cut, VSX type out of scope) are final:
#: no comparison-star selection, never measured, never revived later.
FINAL_SKIP_REASONS = frozenset({"saturated", "below_snr_min", "vsx_type_out_of_scope"})

#: Per-frame CSV columns loaded for the comparison pool.
# APERTURE-K-2026-10: small grid of apertures in units of the frame FWHM.
# AP-SMALL-K-2026-10 (Milan 2026-10-28): down to 0.6 FWHM - for faint (sky-limited) stars the field
# scatter minimum lay at the old grid edge 1.0 (Brno 534, BO CVn: 12-16 % lower scatter at 0.6-0.75;
# S/N optimum ~0.6-0.7 FWHM, Howell 1989, Naylor 1998). AP-BRIGHT-K still chooses k per brightness group.
DEFAULT_APERTURE_F_GRID: tuple[float, ...] = (0.6, 0.75, 0.875, 1.0, 1.25, 1.5, 1.75, 2.0)


def flux_k_column(k: float) -> str:
    """Per-frame proc column with the flux in r = k x FWHM (e.g. 1.25 -> ``flux_k125``, 0.6 -> ``flux_k060``)."""
    return f"flux_k{int(round(float(k) * 100)):03d}"


def flux_k_radius_px(k: float, fwhm_px: float) -> float:
    """Radius [px] of the proc grid flux ``flux_k_column(k)`` in a frame of FWHM ``fwhm_px``
    (``fwhm_px_for_aperture``): r = k x FWHM, at least 0.5 px. The one definition for the writer
    (``photometry_shared``) and for the CCD-equation error of that flux (COMP-CEIL-FIELDK-2026-10)."""
    return max(0.5, float(k) * float(fwhm_px))


def blend_k_column(k: float) -> str:
    """Column of the neighbour-light fraction at r = k x FWHM (BLEND-K-2026-10)."""
    return f"blend_k{int(round(float(k) * 100)):03d}"


#: proc columns of the aperture grid and the comparison-table neighbour-light columns, in grid order
FLUX_K_COLUMNS: tuple[str, ...] = tuple(flux_k_column(k) for k in DEFAULT_APERTURE_F_GRID)
BLEND_K_COLUMNS: tuple[str, ...] = tuple(blend_k_column(k) for k in DEFAULT_APERTURE_F_GRID)


PHASE_USECOLS_PERFRAME: tuple[str, ...] = (
    "catalog_id",
    "bjd_tdb_mid",
    "dao_flux",
    "sky_adu_per_px_annulus",
    SKY_N_PX_COL,
    "aperture_r_px",
    "is_saturated",
    "is_noisy",
    "snr50_ok",
    "vsx_known_variable",
    "likely_saturated",
    # APERTURE-K-2026-10: comparison-star RMS on the light-curve aperture grid
    *FLUX_K_COLUMNS,
)


# --- moved from aperture_pertarget (LAYER-2026-10) ---


