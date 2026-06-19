"""T2-2: frame dimensions derived from MASTERSTAR FITS NAXIS1/NAXIS2."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits


def _write_synthetic_ms(path: Path, naxis1: int, naxis2: int) -> None:
    data = np.zeros((naxis2, naxis1), dtype=np.float32)
    hdr = fits.Header()
    hdr["NAXIS1"] = int(naxis1)
    hdr["NAXIS2"] = int(naxis2)
    fits.PrimaryHDU(data=data, header=hdr).writeto(path, overwrite=True)


def test_resolve_frame_hw_from_fits_naxis(tmp_path: Path) -> None:
    from photometry_core import _resolve_frame_hw_px_from_masterstar

    ms = tmp_path / "MASTERSTAR.fits"
    _write_synthetic_ms(ms, 4096, 3000)
    w, h, src = _resolve_frame_hw_px_from_masterstar(
        ms,
        frame_w_px=2082,
        frame_h_px=1397,
    )
    assert src == "fits_naxis"
    assert w == 4096
    assert h == 3000
    assert (w, h) != (2082, 1397)


def test_resolve_frame_hw_falls_back_to_caller_default(tmp_path: Path) -> None:
    from photometry_core import _resolve_frame_hw_px_from_masterstar

    missing = tmp_path / "missing.fits"
    w, h, src = _resolve_frame_hw_px_from_masterstar(
        missing,
        frame_w_px=2082,
        frame_h_px=1397,
    )
    assert src == "caller_default"
    assert w == 2082
    assert h == 1397
