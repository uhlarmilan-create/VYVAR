"""VY_FWHM = frame QC FWHM (FWHM-ONE-2026-10): robust to hot pixels, no clipping."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from pipeline import _qc_fwhm_elongation

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "robust_fwhm_frame62.npz"


def _load_frame62_fixture() -> np.ndarray:
    """Center-cropped calibrated BO CVn frame 62 (draft 435 source).

    Gaussian-core FWHM 2.1-2.7 px (photutils fit_fwhm on 20 bright stars, median 2.3);
    the old moment estimate (~5.3 px) included the PSF wings.
    """
    if not _FIXTURE.is_file():
        raise FileNotFoundError(f"missing fixture {_FIXTURE}")
    with np.load(_FIXTURE) as z:
        return np.asarray(z["data"], dtype=np.float32)


def test_robust_fwhm_on_frame62_not_cr_scale() -> None:
    """Frame 62 previously stored DB FWHM=1.45 (CR-scale); must measure ~5 px stars."""
    data = _load_frame62_fixture()
    # Inject many hot pixels / cosmic spikes
    rng = np.random.default_rng(0)
    hot = data.copy()
    h, w = hot.shape
    for _ in range(200):
        y = int(rng.integers(10, h - 10))
        x = int(rng.integers(10, w - 10))
        hot[y, x] += float(rng.uniform(8000.0, 30000.0))
    rob = _qc_fwhm_elongation(hot)
    clean = _qc_fwhm_elongation(data)
    assert rob["fwhm_px"] is not None, rob
    assert 1.9 <= float(rob["fwhm_px"]) <= 2.8
    assert abs(float(rob["fwhm_px"]) - float(clean["fwhm_px"])) < 0.1
    assert int(rob["n_sources"]) >= 12


def test_qc_fwhm_elongation_not_segmentation_cr() -> None:
    data = _load_frame62_fixture()
    qc = _qc_fwhm_elongation(data)
    assert qc["fwhm_px"] is not None
    # Elliptical Gaussian fit, same as photutils fit_fwhm (median 2.3 px).
    assert 1.9 < float(qc["fwhm_px"]) < 2.8
