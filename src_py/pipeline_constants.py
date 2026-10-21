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
