# -*- coding: ascii -*-
"""SAT-DIAG unit tests (INV-SAT-01 regression gate)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src_py"
sys.path.insert(0, str(SRC))

from sat_diag import (  # noqa: E402
    N_PILEUP_MIN,
    PILEUP_RATIO,
    derive_ceiling_from_paths,
    peak_self_check,
    resolve_sat_limit,
    PileupResult,
)


def _make_uint16_fits(path: Path, data: np.ndarray) -> None:
    hdu = fits.PrimaryHDU(data.astype(np.uint16))
    hdu.header["BITPIX"] = 16
    hdu.header["BZERO"] = 32768
    hdu.header["BSCALE"] = 1
    hdu.writeto(path, overwrite=True)


def test_pileup_detection_at_65535(tmp_path: Path) -> None:
    d = np.full((32, 32), 5000, dtype=np.uint16)
    d.ravel()[: N_PILEUP_MIN + 10] = 65535
    d.ravel()[N_PILEUP_MIN + 10] = 65532
    fp = tmp_path / "f.fits"
    _make_uint16_fits(fp, d)
    r = derive_ceiling_from_paths([fp])
    assert r.pileup_detected
    assert r.v_ceiling == 65535.0


def test_no_pileup_when_below_threshold(tmp_path: Path) -> None:
    d = np.full((32, 32), 5000, dtype=np.uint16)
    d[0, 0] = 65535
    fp = tmp_path / "f.fits"
    _make_uint16_fits(fp, d)
    r = derive_ceiling_from_paths([fp])
    assert not r.pileup_detected


def test_conflict_derived_refutes_equipment(tmp_path: Path) -> None:
    d = np.full((32, 32), 5000, dtype=np.uint16)
    d.ravel()[: N_PILEUP_MIN + 10] = 65535
    fp = tmp_path / "f.fits"
    _make_uint16_fits(fp, d)
    pileup = derive_ceiling_from_paths([fp])
    hdr = fits.Header()
    hdr["BITPIX"] = 16
    hdr["BZERO"] = 32768
    ctx = resolve_sat_limit(hdr=hdr, pileup=pileup, equipment_adu=16384.0)
    assert ctx.sat_source == "CONFLICT_DERIVED"
    assert ctx.sat_adu == 65535.0
    assert ctx.refuted_source == "EQUIPMENT"


def test_derived_no_pileup_bitpix(tmp_path: Path) -> None:
    d = np.full((32, 32), 5000, dtype=np.uint16)
    fp = tmp_path / "f.fits"
    _make_uint16_fits(fp, d)
    pileup = derive_ceiling_from_paths([fp])
    hdr = fits.Header()
    hdr["BITPIX"] = 16
    hdr["BZERO"] = 32768
    ctx = resolve_sat_limit(hdr=hdr, pileup=pileup, equipment_adu=None)
    assert ctx.sat_source == "DERIVED_NO_PILEUP"
    assert ctx.sat_adu == 65535.0


def test_peak_self_check_rejects_background() -> None:
    arr = np.full((64, 64), 2500.0, dtype=np.float64)
    arr[32, 32] = 2600.0
    assert not peak_self_check(arr, 32, 32, 2600.0)


def test_peak_self_check_accepts_star() -> None:
    arr = np.full((64, 64), 2500.0, dtype=np.float64)
    yy, xx = np.ogrid[:64, :64]
    dist2 = (xx - 32) ** 2 + (yy - 32) ** 2
    arr[dist2 <= 9] = 20000.0
    arr[32, 32] = 21000.0
    assert peak_self_check(arr, 32, 32, 21000.0)


def test_refuse_float_input(tmp_path: Path) -> None:
    d = np.ones((8, 8), dtype=np.float32)
    hdu = fits.PrimaryHDU(d)
    hdu.header["BITPIX"] = -32
    fp = tmp_path / "float.fits"
    hdu.writeto(fp, overwrite=True)
    r = derive_ceiling_from_paths([fp])
    assert r.refused
    assert r.refuse_reason == "REFUSE_NON_RAW"
