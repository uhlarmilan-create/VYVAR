"""Tests for ePSF science-light frame enumeration (EPSF-VALID-02 F5)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from astropy.io import fits
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src_py"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from epsf_frame_accounting import (  # noqa: E402
    is_non_science_aligned_fits,
    list_epsf_science_light_fits,
)


def test_is_non_science_aligned_fits_masterstar() -> None:
    assert is_non_science_aligned_fits("MASTERSTAR.fits")
    assert is_non_science_aligned_fits("masterstar.fits")
    assert not is_non_science_aligned_fits("BO_CVn_Light_001.fits")


def test_list_epsf_science_light_fits_excludes_masterstar(tmp_path: Path) -> None:
    lights = tmp_path / "NoFilter_60_2"
    lights.mkdir()
    data = np.zeros((16, 16), dtype=np.float32)
    hdr = fits.Header()
    hdr["NAXIS1"] = 16
    hdr["NAXIS2"] = 16
    fits.PrimaryHDU(data=data, header=hdr).writeto(lights / "BO_CVn_Light_001.fits", overwrite=True)
    fits.PrimaryHDU(data=data, header=hdr).writeto(lights / "MASTERSTAR.fits", overwrite=True)

    found = list_epsf_science_light_fits(lights)
    names = {p.name for p in found}
    assert names == {"BO_CVn_Light_001.fits"}
    assert len(found) == 1


@pytest.mark.skipif(
    not (ROOT / "Archive/Drafts/draft_000516/detrended_aligned/lights/NoFilter_60_2").is_dir(),
    reason="draft 516 not on disk",
)
def test_draft516_science_light_count_is_134() -> None:
    root = ROOT / "Archive/Drafts/draft_000516/detrended_aligned/lights/NoFilter_60_2"
    files = list_epsf_science_light_fits(root)
    assert len(files) == 134
    assert all("Light_" in p.name for p in files)
    assert not any(p.name.upper() == "MASTERSTAR.FITS" for p in files)
