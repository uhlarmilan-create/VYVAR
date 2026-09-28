# -*- coding: ascii -*-
"""FIXPOS-NOOP-01: _apply_psf_fixed_position must fix IterativePSFPhotometry."""

from __future__ import annotations

import numpy as np
import pytest
from photutils.psf import ImagePSF, IterativePSFPhotometry, PSFPhotometry

from psf_photometry import _apply_psf_fixed_position, _epsf_noop_finder


def _tiny_image_psf() -> ImagePSF:
    data = np.ones((11, 11), dtype=np.float64)
    data /= float(data.sum())
    return ImagePSF(data, oversampling=1)


def test_fixpos_noop_01_iterative_sets_x0_y0_fixed():
    """Pre-fix: IterativePSFPhotometry has no psf_model -> EXC-0454 no-op.
    Post-fix: flags land on phot._psfphot.psf_model.
    """
    model = _tiny_image_psf()
    phot = IterativePSFPhotometry(
        model,
        (5, 5),
        _epsf_noop_finder,
        aperture_radius=3,
        progress_bar=False,
    )
    assert not hasattr(phot, "psf_model") or getattr(phot, "psf_model", None) is None
    assert phot._psfphot.psf_model.x_0.fixed is False
    assert phot._psfphot.psf_model.y_0.fixed is False

    _apply_psf_fixed_position(phot, fix=True)

    assert phot._psfphot.psf_model.x_0.fixed is True
    assert phot._psfphot.psf_model.y_0.fixed is True


def test_fixpos_noop_01_plain_psfphotometry_still_works():
    model = _tiny_image_psf()
    phot = PSFPhotometry(model, fit_shape=(5, 5), progress_bar=False)
    _apply_psf_fixed_position(phot, fix=True)
    assert phot.psf_model.x_0.fixed is True
    assert phot.psf_model.y_0.fixed is True


def test_fixpos_noop_01_fix_false_is_noop():
    model = _tiny_image_psf()
    phot = IterativePSFPhotometry(
        model,
        (5, 5),
        _epsf_noop_finder,
        aperture_radius=3,
        progress_bar=False,
    )
    _apply_psf_fixed_position(phot, fix=False)
    assert phot._psfphot.psf_model.x_0.fixed is False
