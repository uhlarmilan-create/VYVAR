"""Header-first gain resolution (param_resolver.resolve_gain) - audit oracle cases."""

from __future__ import annotations

import math

import pytest
from astropy.io import fits

from param_resolver import CROSS_CHECK_RTOL, resolve_gain, resolve_read_noise


def _hdr(**cards: tuple[float, str]) -> fits.Header:
    """Build a FITS header from keyword -> (value, comment) pairs."""
    h = fits.Header()
    for key, (val, comment) in cards.items():
        h[key] = (val, comment)
    return h


# ---------------------------------------------------------------------------
# Moravian e-/ADU headers (C3, C5A)
# ---------------------------------------------------------------------------
def test_moravian_c3_gain_header_e_per_adu():
    hdr = _hdr(GAIN=(0.78, "e-/ADU"))
    res = resolve_gain(hdr, db_value=0.78, equipment_id=2)
    assert res.ok
    assert res.source == "header"
    assert math.isclose(res.value, 0.78)
    assert not res.warnings


def test_moravian_c5a_gain_header_cross_check_warning():
    hdr = _hdr(GAIN=(12.48, "e-/ADU"))
    res = resolve_gain(hdr, db_value=1.0, equipment_id=4)
    assert res.ok
    assert res.source == "header"
    assert math.isclose(res.value, 12.48)
    assert res.warnings
    assert "disagrees" in res.warnings[0].lower()
    assert not math.isclose(12.48, 1.0, rel_tol=CROSS_CHECK_RTOL)


def test_egain_preferred_over_gain():
    hdr = _hdr(EGAIN=(2.5, "e-/ADU"), GAIN=(99.0, "Gain"))
    res = resolve_gain(hdr, db_value=1.0, equipment_id=1)
    assert res.ok
    assert res.source == "header"
    assert math.isclose(res.value, 2.5)
    assert res.key == "EGAIN"


# ---------------------------------------------------------------------------
# QHY slider index - never use raw value as e-/ADU
# ---------------------------------------------------------------------------
def test_qhy294_setting_zero_maps_to_db_gain():
    hdr = _hdr(GAIN=(0.0, "Gain"))
    hdr["READMODE"] = 0
    res = resolve_gain(hdr, db_value=3.17, equipment_id=1)
    assert res.ok
    assert res.source == "header_index_mapped"
    assert math.isclose(res.value, 3.17)
    assert res.value != 0.0


def test_qhy294_setting_56_never_used_as_e_per_adu():
    hdr = _hdr(GAIN=(56.0, "Gain"))
    res = resolve_gain(hdr, db_value=3.17, equipment_id=1)
    assert res.ok
    assert res.value != 56.0
    assert res.source == "db"
    assert math.isclose(res.value, 3.17)
    assert res.warnings
    assert "56" in res.warnings[0]
    assert "not in gain map" in res.warnings[0].lower()


def test_qhy294_readmode_triggers_index_semantics():
    hdr = _hdr(GAIN=(5.0, ""))
    hdr["READMODE"] = 0
    res = resolve_gain(hdr, db_value=3.17, equipment_id=1)
    assert res.ok
    assert res.value != 5.0
    assert res.source in ("db", "header_index_mapped")


# ---------------------------------------------------------------------------
# No header gain -> DB
# ---------------------------------------------------------------------------
def test_no_header_gain_uses_db():
    res = resolve_gain(None, db_value=1.0, equipment_id=3)
    assert res.ok
    assert res.source == "db"
    assert math.isclose(res.value, 1.0)


def test_empty_header_uses_db():
    res = resolve_gain(fits.Header(), db_value=1.0, equipment_id=3)
    assert res.ok
    assert res.source == "db"
    assert math.isclose(res.value, 1.0)


# ---------------------------------------------------------------------------
# Read noise - DB-first (unchanged)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "eq_id,db_rn",
    [(2, 1.3), (4, 10.0), (1, 7.6), (3, 5.0)],
)
def test_read_noise_db_first_no_header(eq_id: int, db_rn: float):
    hdr = _hdr(GAIN=(0.78, "e-/ADU"))
    res = resolve_read_noise(hdr, db_value=db_rn, equipment_id=eq_id)
    assert res.ok
    assert res.source == "db"
    assert math.isclose(res.value, db_rn)


def test_read_noise_c3_oracle():
    res = resolve_read_noise(None, db_value=1.3, equipment_id=2)
    assert res.ok
    assert res.source == "db"
    assert math.isclose(res.value, 1.3)


# ---------------------------------------------------------------------------
# Fallback chain
# ---------------------------------------------------------------------------
def test_gain_config_fallback_when_no_db():
    hdr = fits.Header()
    res = resolve_gain(hdr, db_value=None, equipment_id=None, cfg=type("C", (), {"gain": 2.5})())
    assert res.ok
    assert res.source == "config"
    assert math.isclose(res.value, 2.5)
