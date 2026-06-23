"""DB bin1 equipment intrinsics scaled by summed FITS binning."""

from __future__ import annotations

import logging
import math

import pytest
from astropy.io import fits

from param_resolver import (
    _binning_from_header,
    _scale_bin1_db_for_header,
    _scale_bin1_to_binning,
    resolve_gain,
    resolve_read_noise,
)


def test_scale_bin1_to_binning_no_op_at_bin1():
    assert math.isclose(_scale_bin1_to_binning(0.78, 1, 2), 0.78)
    assert math.isclose(_scale_bin1_to_binning(1.3, 1, 1), 1.3)


def test_scale_bin1_to_binning_bin2_gain_and_rn():
    assert math.isclose(_scale_bin1_to_binning(0.78, 2, 2), 3.12)
    assert math.isclose(_scale_bin1_to_binning(1.3, 2, 1), 2.6)


def test_scale_bin1_to_binning_bin3():
    assert math.isclose(_scale_bin1_to_binning(0.78, 3, 2), 0.78 * 9)
    assert math.isclose(_scale_bin1_to_binning(1.3, 3, 1), 1.3 * 3)


def test_binning_from_header_reads_xbinning():
    hdr = fits.Header()
    hdr["XBINNING"] = 2
    hdr["YBINNING"] = 2
    assert _binning_from_header(hdr) == 2


def test_binning_from_header_absent():
    assert _binning_from_header(None) is None
    assert _binning_from_header(fits.Header()) is None


def test_binning_from_header_asymmetric_xy():
    hdr = fits.Header()
    hdr["XBINNING"] = 2
    hdr["YBINNING"] = 3
    assert _binning_from_header(hdr) is None


def test_scale_db_for_header_warns_when_binning_absent(caplog):
    caplog.set_level(logging.WARNING)
    eff = _scale_bin1_db_for_header(
        1.3, None, exponent=1, param_label="read_noise_warn_test"
    )
    assert math.isclose(eff, 1.3)
    assert any("binning unresolved" in r.message.lower() for r in caplog.records)


def test_scale_db_for_header_info_when_bin2(caplog):
    hdr = fits.Header()
    hdr["XBINNING"] = 2
    hdr["YBINNING"] = 2
    caplog.set_level(logging.INFO)
    eff = _scale_bin1_db_for_header(1.3, hdr, exponent=1, param_label="read_noise")
    assert math.isclose(eff, 2.6)
    assert any("scaled DB bin1" in r.message for r in caplog.records)


def test_resolve_gain_db_fallback_scaled_bin2():
    hdr = fits.Header()
    hdr["XBINNING"] = 2
    hdr["YBINNING"] = 2
    res = resolve_gain(hdr, db_value=0.78, equipment_id=2)
    assert res.ok
    assert res.source == "db"
    assert math.isclose(res.value, 3.12)


def test_resolve_gain_fits_present_not_scaled():
    hdr = fits.Header()
    hdr["GAIN"] = (3.12, "e-/ADU")
    hdr["XBINNING"] = 2
    hdr["YBINNING"] = 2
    res = resolve_gain(hdr, db_value=0.78, equipment_id=2)
    assert res.ok
    assert res.source == "header"
    assert math.isclose(res.value, 3.12)


def test_resolve_read_noise_db_scaled_bin2_draft421_oracle():
    hdr = fits.Header()
    hdr["GAIN"] = (3.12, "e-/ADU")
    hdr["XBINNING"] = 2
    hdr["YBINNING"] = 2
    res = resolve_read_noise(hdr, db_value=1.3, equipment_id=2)
    assert res.ok
    assert res.source == "db"
    assert math.isclose(res.value, 2.6)
