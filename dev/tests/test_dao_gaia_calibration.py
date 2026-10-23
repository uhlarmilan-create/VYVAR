"""DAO-GAIA-ERA-01: matching tolerances derived from the pass-1 residuals."""
from __future__ import annotations

import numpy as np

def test_derive_match_radius_from_detection_identity_p95() -> None:
    from dao_gaia_calibration import derive_tolerances_from_residuals

    identity = np.linspace(0.2, 2.5, 200)
    faint = np.linspace(0.8, 1.6, 80)
    tol = derive_tolerances_from_residuals(
        identity,
        faint,
        fwhm_px=5.3,
        plate_scale_arcsec_per_px=1.3,
        pass1_sigma=4.5,
        pass2_sigma=4.0,
        match_k=1.7,
        centroid_floor_px=1.0,
        centroid_cap_px=3.0,
    )
    id_p95 = float(np.percentile(identity, 95))
    faint_p95 = float(np.percentile(faint, 95))
    assert abs(tol.residual_p95_px - id_p95) < 1e-6
    assert abs(tol.match_radius_px - round(1.7 * id_p95 * 2) / 2) < 1e-6
    assert abs(tol.pass2_center_tol_px - round(max(1.0, min(3.0, faint_p95)) * 2) / 2) < 1e-6
    assert tol.detection_identity is not None
    assert tol.seed_centroid is not None
    assert tol.detection_identity.n == 200
    assert tol.seed_centroid.n == 80
