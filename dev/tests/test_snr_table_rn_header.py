"""RN header passthrough for SNR aperture table (RN-HEADER-NONE fix)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from photometry_core import precompute_and_save_snr_aperture_table_for_draft


def test_snr_table_read_noise_unscaled_when_header_unavailable(tmp_path: Path) -> None:
    """When MASTERSTAR header cannot be read, RN stays DB bin1 (legacy SNR-table path)."""
    missing = tmp_path / "missing_MASTERSTAR.fits"
    db_path = Path(__file__).resolve().parents[2] / "vyvar.sqlite3"
    if not db_path.is_file():
        pytest.skip("vyvar.sqlite3 not available")

    out = precompute_and_save_snr_aperture_table_for_draft(
        tmp_path,
        masterstar_fits_path=missing,
        fwhm_fallback_px=3.0,
        database_path=db_path,
        equipment_id=2,
        sky_fallback=100.0,
    )
    assert out is not None
    tbl = json.loads((tmp_path / "aperture_snr_table.json").read_text(encoding="utf-8"))
    # equipment_id=2 wide rig: DB read_noise 1.3 e- per pixel (no bin scaling without header)
    assert math.isclose(float(tbl["read_noise"]), 1.3, rel_tol=1e-6)


def test_snr_table_read_noise_doubles_with_bin2_header(tmp_path: Path) -> None:
    """Bin2 MASTERSTAR header -> RN scaled in aperture_snr_table.json (parity with Phase 2A)."""
    ms = tmp_path / "MASTERSTAR.fits"
    data = np.zeros((64, 64), dtype=np.float32)
    hdu = fits.PrimaryHDU(data=data)
    hdu.header["XBINNING"] = 2
    hdu.header["YBINNING"] = 2
    hdu.writeto(ms, overwrite=True)

    db_path = Path(__file__).resolve().parents[2] / "vyvar.sqlite3"
    if not db_path.is_file():
        pytest.skip("vyvar.sqlite3 not available")

    out = precompute_and_save_snr_aperture_table_for_draft(
        tmp_path,
        masterstar_fits_path=ms,
        fwhm_fallback_px=3.0,
        database_path=db_path,
        equipment_id=2,
        sky_fallback=100.0,
    )
    assert out is not None
    tbl = json.loads((tmp_path / "aperture_snr_table.json").read_text(encoding="utf-8"))
    assert math.isclose(float(tbl["read_noise"]), 2.6, rel_tol=1e-6)
