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
# COMP-SAT-80-2026-10: the comparison-star peak limit (0.80 x SAT) is the config value comp_peak_sat_frac
# (Milan 2026-10-29, COMP-SAT-ONE-2026-10); targets keep LINEARITY_FRAC.
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
# ALIGN-POLY-2026-10 (Milan 2026-10-24): when the similarity transform does not verify, a
# polynomial of this order is fitted on the star pairs (the "general" geometry of IRAF
# geomap/geotran: linear part + distortion). Wide field after a shift on the chip (MN Boo 538,
# 200 mm lens, field moved 110-160 px against the reference): similarity p90 0.59-0.62 px
# (edge ~1 px, refused at 0.5), affine the same, 2nd order 0.12-0.16 px on 250 pairs.
ALIGN_POLY_ORDER = 2
# ALIGN-MODEL-2026-10 (Milan 2026-10-24): every frame chooses its registration model on its own stars.
# A verified similarity is replaced by the polynomial only when the Bayesian information criterion
# on the same star pairs prefers it by more than this (BIC_similarity - BIC_polynomial; Kass & Raftery
# 1995: > 10 = very strong evidence). The polynomial always fits a little better (12 vs 4 parameters);
# BIC charges the extra parameters, so a frame without distortion keeps the similarity.
ALIGN_MODEL_DBIC = 10.0
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
# Alignment: astroalign control-point cap (detection count is alignment_max_stars).
ALIGNMENT_MAX_CONTROL_POINTS = 80
# DAO kernel FWHM (px) for a frame without a measured FWHM (QC writes VY_FWHM on every processed
# light; this is reached only when that measurement failed). Was config sips_dao_fwhm_px (2.5),
# which dao_detection_fwhm_pixels already raised to 3.0.
DAO_FWHM_FALLBACK_PX = 3.0

# MAD -> sigma for a normal distribution: sigma = MAD / 0.6745 (= 1.4826 x MAD).
MAD_CONSISTENCY = 0.6745
# F-BINGAIN-1: clamp of the empirical / Howell background scale ratio.
BKG_SCALE_R_CLAMP_LO = 0.05
BKG_SCALE_R_CLAMP_HI = 2.0


# --- moved from frame_depth (LAYER-2026-10) ---


#: C(G90) = 0.9 on the Fleming erf: G90 = G50 - z_0.9 sigma.
Z_90 = 1.2815516

# CONFIG-ONE-2026-10 (G05-15 / G14-13, Milan 2026-10-27): one default location of the catalogue DBs, used by
# AppConfig, the catalogue builders and the Settings wizard (relative to the data directory).
GAIA_DB_DEFAULT = "catalogs/vyvar_gaia_dr3.db"
OBJECTS_DB_DEFAULT = "catalogs/vyvar_objects.db"
# G05-18: catalogue builder defaults, one place for the wizard, its job arguments and the scripts' CLI.
GAIA_BUILD_MAG_LIMIT_DEFAULT = 17.5
VSX_BUILD_MAG_LIMIT_DEFAULT = 18.0
# CONFIG-ONE-2026-10 (Milan 2026-10-27): the values runs used, now named constants (the GUI / CLI parameters
# catalog_match_max_sep_arcsec, max_catalog_rows, min/max_detected_stars had three defaults each and were floored).
#: MASTERSTAR initial Gaia match radius [arcsec]; recorded in the meta, the identity radius replaces it (IDENTITY-2026-10).
CATALOG_MATCH_SEP_ARCSEC = 10.0
#: DAO detections kept on MASTERSTAR (spatially stratified) and on each aligned frame.
MASTERSTAR_MAX_DAO_ROWS = 100000
FRAME_MAX_DAO_ROWS = 12000
#: Alignment reference frame: fewest stars to accept it, most stars detected on it.
ALIGN_MIN_DETECTED_STARS = 100
ALIGN_MAX_DETECTED_STARS = 500
