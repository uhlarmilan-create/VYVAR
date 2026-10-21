"""F3 (RUN522-CRASH-01): post-solve Gaia cone follows the chip WCS - no global 3.5 deg floor
(CATALOG-FOOTPRINT-2026-10: the optics floor is gone, the WCS footprint is the only cone)."""
from __future__ import annotations

import sys
from pathlib import Path

from astropy.io import fits
from astropy.wcs import WCS

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src_py"))

from pipeline_catalog import _field_center_and_radius_from_wcs  # noqa: E402


def _hdr(scale_arcsec=0.566, nx=3552, ny=2664):
    h = fits.Header()
    h["NAXIS"] = 2
    h["NAXIS1"], h["NAXIS2"] = nx, ny
    h["CTYPE1"], h["CTYPE2"] = "RA---TAN", "DEC--TAN"
    h["CRVAL1"], h["CRVAL2"] = 320.36, 51.82
    h["CRPIX1"], h["CRPIX2"] = nx / 2, ny / 2
    h["CD1_1"], h["CD1_2"], h["CD2_1"], h["CD2_2"] = -scale_arcsec / 3600, 0.0, 0.0, scale_arcsec / 3600
    return h


def test_narrow_field_cone_not_floored_to_3p5():
    h = _hdr()
    _, r = _field_center_and_radius_from_wcs(WCS(h), h["NAXIS2"], h["NAXIS1"])
    half_diag = 0.5 * ((3552 * 0.566) ** 2 + (2664 * 0.566) ** 2) ** 0.5 / 3600
    assert half_diag < r < 2.0 * half_diag  # footprint + margins, not 3.5 deg
