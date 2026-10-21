"""COMP-ADMIT-03: continuous comparison-star weights (no admission rank cuts).

sigma_eff^2 = sigma_rms^2 + (c_col * |delta(BP-RP)|)^2 + (c_dist * r_deg)^2
w           = 1 / sigma_eff^2

Gates that remain elsewhere (not weights): saturation/non-linearity, known
variable, geometry (aligned footprint). See docs/VYVAR_DECISIONS.md COMP-ADMIT-03.
"""
from __future__ import annotations

import logging
import math


LOGGER = logging.getLogger(__name__)

# COMP-POOL-02 Item 4 (draft 512, Zeiss 200 mm refractive): FWHM-rescaled COG
# enclosed-flux red-minus-blue = 29.485 mmag over BP-RP span 0.5 -> 1.5 (Delta=1.0).
# c_col_psf = 0.029485 / 1.0 mag per BP-RP. Mirror/Newton: prediction ~0.
C_COL_PSF_EE_MMAG = 29.485010546318453
C_COL_PSF_BPRP_SPAN = 1.0  # BP-RP 0.5 to 1.5
C_COL_PSF_REFRACTIVE_MAG_PER_BPRP = float(C_COL_PSF_EE_MMAG / 1000.0 / C_COL_PSF_BPRP_SPAN)
C_COL_PSF_SOURCE = (
    "MEASURED COMP-POOL-02 Item4: mmag_cog_fwhm_rescaled="
    f"{C_COL_PSF_EE_MMAG:.3f} over Delta(BP-RP)={C_COL_PSF_BPRP_SPAN:.3f} "
    f"(0.5->1.5); c_col_psf={C_COL_PSF_REFRACTIVE_MAG_PER_BPRP:.6g} mag/BP-RP"
)


def sigma_eff_mag(
    *,
    sigma_rms_mag: float,
    delta_bprp: float,
    r_deg: float,
    c_col_mag_per_bprp: float,
    c_dist_mag_per_deg: float,
) -> float:
    """Effective sigma in magnitudes for one comparison star relative to one target."""
    s = float(sigma_rms_mag)
    if not math.isfinite(s) or s < 0:
        s = float("nan")
    dc = float(delta_bprp) if math.isfinite(float(delta_bprp)) else 0.0
    rd = float(r_deg) if math.isfinite(float(r_deg)) else 0.0
    cc = float(c_col_mag_per_bprp) if math.isfinite(float(c_col_mag_per_bprp)) else 0.0
    cd = float(c_dist_mag_per_deg) if math.isfinite(float(c_dist_mag_per_deg)) else 0.0
    if not math.isfinite(s):
        return float("nan")
    # Floor tiny rms so a perfect flat artefact does not dominate with infinite weight.
    s_use = max(s, 1e-6)
    se2 = s_use * s_use + (cc * abs(dc)) ** 2 + (cd * abs(rd)) ** 2
    return float(math.sqrt(se2))


def weight_from_sigma_eff(sigma_eff: float) -> float:
    if not math.isfinite(float(sigma_eff)) or float(sigma_eff) <= 0:
        return 0.0
    return float(1.0 / (float(sigma_eff) ** 2))

