"""Pipeline numeric and column constants.

Leaf rule: this module imports nothing from VYVAR; anything may import
it at module level.
"""
from __future__ import annotations

_SKY_ADU_FALLBACK = 1581.6  # was cfg.sky_adu_fallback (1581.6)
_MASTERSTAR_SOLVER_USE_DRAFT_MEDIAN_IF_HINT_SEP_DEG = 1.0  # was cfg.masterstar_solver_use_draft_median_if_hint_sep_deg (1.0)
_MASTERSTAR_OPTIMIZER_MIRROR_EXTRA_LOG = True  # was cfg.masterstar_optimizer_mirror_extra_log (True)
_MASTERSTAR_PLATESOLVE_PREWRITE_RMS_MAX_PX = 30.0  # was cfg.masterstar_platesolve_prewrite_rms_max_px (30.0)
_MASTERSTAR_PLATESOLVE_PREWRITE_RELAXED_RMS_MAX_PX = 35.0  # was cfg.masterstar_platesolve_prewrite_relaxed_rms_max_px (35.0)
_MASTERSTAR_PLATESOLVE_NN_REFINE_MAX_RMS_PX = None  # was cfg.masterstar_platesolve_nn_refine_max_rms_px (None)
_MASTERSTAR_SIP_FORCE_RMS_GUARD_RATIO = 1.15  # was cfg.masterstar_sip_force_rms_guard_ratio (1.15)
_PLATESOLVE_ANISOTROPY_THRESHOLD = 1.3  # was cfg.platesolve_anisotropy_threshold (1.3)

_EXO_HOST_ANNOTATION_COLUMNS: tuple[str, ...] = (
    "exo_host_obj_id",
    "exo_host_name",
    "exo_cat_source",
    "exo_disposition",
    "exo_match_sep_arcsec",
)

# SAT-LIMIT-01 / GAIN-DOMAIN-01: 16-bit FITS container clip (pile-up at 65535, not 65532).
SAT_LIMIT_CONTAINER_CLIP_ADU = 65535.0
# Peak-test fraction when the linearity knee is unmeasured (D1-2 / SAT-LIMIT-01 B3).
SAT_LIMIT_NO_KNEE_FRAC = 0.80
# Provenance string for the INV-SAT-LIMIT peak-test (catalog zone + per-frame clean).
SAT_LIMIT_PEAK_TEST_SOURCE = (
    f"INV-SAT-LIMIT peak-test {SAT_LIMIT_NO_KNEE_FRAC:.2f}x "
    f"container_clip_{SAT_LIMIT_CONTAINER_CLIP_ADU:.0f}"
)
