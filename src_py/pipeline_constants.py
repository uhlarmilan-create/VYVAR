"""Pipeline numeric and column constants.

Leaf rule: this module imports nothing from VYVAR; anything may import
it at module level.
"""
from __future__ import annotations

_MASTERSTAR_SOLVER_USE_DRAFT_MEDIAN_IF_HINT_SEP_DEG = 1.0  # was cfg.masterstar_solver_use_draft_median_if_hint_sep_deg (1.0)

_EXO_HOST_ANNOTATION_COLUMNS: tuple[str, ...] = (
    "exo_host_obj_id",
    "exo_host_name",
    "exo_cat_source",
    "exo_disposition",
    "exo_match_sep_arcsec",
)

# FRAME-QC-PARITY-02C: two-sided n_stars diagnostic (status=ok); no drop authority.
N_STARS_DIAG_K = 5.0
_N_STARS_DIAG_MAD_TO_SIGMA = 1.4826

# SAT-LIMIT-01 / GAIN-DOMAIN-01: 16-bit FITS container clip (pile-up at 65535, not 65532).
SAT_LIMIT_CONTAINER_CLIP_ADU = 65535.0
# SAT-LIN-2026-10: one linearity limit. A star whose raw peak (incl. sky) reaches
# LINEARITY_FRAC x saturation level is 'saturated' (not linear) - the top 10 % of the
# range is never used for photometry. Saturation level itself comes from the data
# (pile-up), header or bit depth (sat_diag), not from a user parameter.
LINEARITY_FRAC = 0.90
SAT_LIMIT_NO_KNEE_FRAC = LINEARITY_FRAC
# COMP-SAT-80-2026-10 (Milan 2026-10-12): comparison stars must be safely linear - a comp is
# not admitted when its raw peak reaches 0.80 x saturation level (targets: 0.90). AAVSO CCD/CMOS
# guide 3.3.7: keep peaks well below the linearity limit (60-70 %), >= 90 % is unreliable.
COMP_LINEARITY_FRAC = 0.80
#: factor on the per-row ``saturate_limit_adu_85pct`` column (== LINEARITY_FRAC x SAT)
COMP_SAT_LIMIT_SCALE = COMP_LINEARITY_FRAC / LINEARITY_FRAC
# COMP-SNR-20-2026-10 (Milan 2026-10-12): minimum MASTERSTAR aperture S/N of a comparison star
# (snr_ap_pixscaled; AAVSO CCD/CMOS guide 2.1.4: S/N 100 excellent, below 10-20 poor). 50 removed
# 2924 of 3272 pool stars on the 60 s wide-field rig (S/N 50 ~ G 11.5, draft 526), 20 keeps
# the comps up to G ~12.5 that the ensemble used before.
COMP_SNR_MIN = 20.0
# ALIGN-VERIFY-2026-10 (Milan 2026-10-12): every frame->reference registration is checked on the
# stars themselves. The frame's detections are mapped through the transform and paired with the
# reference detections (mutual nearest neighbour within ALIGN_MATCH_FWHM x FWHM). The 90th
# percentile of the pair distances must not exceed max(ALIGN_P90_MAX_FWHM x FWHM,
# ALIGN_P90_FLOOR_PX) - the percentile, not the median, because a rotation left in a shift-only
# registration grows towards the chip edge; the px floor keeps DAO centroid noise of undersampled
# rigs (FWHM 1.3-2 px) from failing correct registrations. At least max(ALIGN_MIN_PAIRS,
# ALIGN_MIN_PAIR_FRAC x possible) pairs. Draft 526: correct registrations p90 0.15 px
# (0.07 FWHM); the shift-only fallback 3.1-3.2 px median.
ALIGN_MATCH_FWHM = 2.0
ALIGN_P90_MAX_FWHM = 0.25
ALIGN_P90_FLOOR_PX = 0.5
ALIGN_MIN_PAIRS = 10
ALIGN_MIN_PAIR_FRAC = 0.3
# Provenance string for the INV-SAT-LIMIT peak-test (catalog zone + per-frame clean).
SAT_LIMIT_PEAK_TEST_SOURCE = (
    f"INV-SAT-LIMIT peak-test {SAT_LIMIT_NO_KNEE_FRAC:.2f}x "
    f"container_clip_{SAT_LIMIT_CONTAINER_CLIP_ADU:.0f}"
)

# CODE-REVIEW-2026-10 step 7: former config keys that nobody tunes (values unchanged).
# Exoplanet host label: Gaia position within this separation of the catalogue host (arcsec).
EXOPLANET_MATCH_MAX_SEP_ARCSEC = 3.0
# Comparison-star selection: RMS (mag) below this is an artefact, not a stable star.
COMP_SELECT_RMS_FLOOR = 1e-6
# ePSF build: fewest MASTERSTAR build stars for a model.
EPSF_MIN_BUILD_STARS = 30
# DAO centroid guard: a matched centroid further than this (x FWHM, capped at 1.5 px) from the
# MASTERSTAR position falls back to the WCS position.
DAO_CENTROID_MAX_SHIFT_FWHM = 1.0
# Colour term: fewest comparison stars for a fitted colour term.
CT_MIN_COMP = 7
# MASTERSTAR DAO / Gaia census (all in px of the MASTERSTAR unless named otherwise).
MS_DAO_PASS1_DEDUP_PX = 0.75  # pass-1 duplicates closer than this are merged
MS_DAO_PASS2_SIGMA = 4.0  # targeted pass-2 detection threshold (sigma)
MS_DAO_PASS2_CENTER_TOL_PX = 2.0  # pass-2 centroid within this of the Gaia position
MS_GAIA_CENSUS_EDGE_MARGIN_PX = 10.0  # Gaia stars nearer the edge are not counted
MS_DAO_MATCH_RADIUS_K = 1.7  # identity radius = k x residual scatter
MS_DAO_CENTROID_QA_FLOOR_PX = 1.0  # centroid QA tolerance floor
MS_DAO_CENTROID_QA_CAP_PX = 3.0  # centroid QA tolerance cap
MS_LOCK_LEFTOVER_RADIUS_PX = 3.0  # leftover DAO detection promoted to a Gaia star within this
MS_LOCK_PAIR_TOL_PX = 3.0  # DAO-Gaia lock pair tolerance
MS_FORCED_SEED_CENTROID_MAX_PX = 2.0  # forced seed accepted when its centroid moves less than this
MS_FORCED_SEED_SNR_MIN = 4.0  # forced seed accepted from this S/N
# Alignment: astroalign control-point cap (detection count is alignment_max_stars).
ALIGNMENT_MAX_CONTROL_POINTS = 80
# DAO kernel FWHM (px) for a frame without a measured FWHM (QC writes VY_FWHM on every processed
# light; this is reached only when that measurement failed). Was config sips_dao_fwhm_px (2.5),
# which dao_detection_fwhm_pixels already raised to 3.0.
DAO_FWHM_FALLBACK_PX = 3.0
