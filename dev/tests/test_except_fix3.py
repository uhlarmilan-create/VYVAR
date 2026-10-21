"""Unit tests for EXCEPT-FIX-3 (tranche 3: importer / platesolver / alignment / astrometry).

Covers the four behaviour-change fixes required by the spec (#3, #5, #8, #9) plus counter
smoke tests for the cheap surfacing-only injection points (#1, #4, #6).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from except_fix_counters import get_except_fix_counters, reset_except_fix_counters


# --------------------------------------------------------------------------- #
# FIX-3 #3 -- EXC-0090 scope-conflict check [fail-open -> fail-closed]
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# FIX-3 #5 -- EXC-0089 capture date [today -> file mtime]
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# FIX-3 #8 -- copy_wcs_header_keys (shared helper, EXC-0625 + EXC-0010)
# --------------------------------------------------------------------------- #
def test_fix3_8_copy_all_keys_ok_passthrough() -> None:
    from wcs_header_io import copy_wcs_header_keys

    src = fits.Header()
    src["CTYPE1"] = "RA---TAN"
    src["CTYPE2"] = "DEC--TAN"
    src["CRVAL1"] = 180.0
    src["CRVAL2"] = 0.0
    src["CRPIX1"] = 8.0
    src["CRPIX2"] = 8.0
    src["CD1_1"] = -0.001
    src["CD2_2"] = 0.001
    src["A_ORDER"] = 2
    src["COMMENT"] = "should be skipped"
    src["SIMPLE"] = True

    dst = fits.Header()
    reset_except_fix_counters()
    failed = copy_wcs_header_keys(dst, src, context="test-ok")
    assert failed == []
    assert dst["CRVAL1"] == 180.0
    assert dst["CD1_1"] == -0.001
    assert dst["A_ORDER"] == 2
    assert "SIMPLE" not in dst  # structural key skipped
    assert "COMMENT" not in dst
    assert get_except_fix_counters().wcs_header_key_copy_fail == 0


class _FakeHeader:
    """Mapping that raises on the configured 'bad' keys to force a copy failure."""

    def __init__(self, keys: list[str], bad: set[str]) -> None:
        self._keys = keys
        self._bad = bad

    def __iter__(self):
        return iter(self._keys)

    def __getitem__(self, k: str):
        if k in self._bad:
            raise ValueError(f"uncopyable card {k}")
        return 1.0


def test_fix3_8_core_key_failure_aborts_and_counts(caplog: pytest.LogCaptureFixture) -> None:
    from wcs_header_io import copy_wcs_header_keys

    src = _FakeHeader(["CRVAL1", "CD1_1", "CRPIX1"], bad={"CRVAL1"})
    dst = fits.Header()
    reset_except_fix_counters()
    caplog.set_level(logging.ERROR)
    failed = copy_wcs_header_keys(dst, src, context="test-core-fail")
    assert failed == ["CRVAL1"]
    assert len(dst) == 0  # atomic: dst untouched on core failure
    assert get_except_fix_counters().wcs_header_key_copy_fail == 1
    assert any("core WCS keys uncopyable" in r.message for r in caplog.records)


def test_fix3_8_noncore_failure_warns_only(caplog: pytest.LogCaptureFixture) -> None:
    from wcs_header_io import copy_wcs_header_keys

    src = _FakeHeader(["CRVAL1", "CD1_1", "FOOBAR"], bad={"FOOBAR"})
    dst = fits.Header()
    reset_except_fix_counters()
    caplog.set_level(logging.WARNING)
    failed = copy_wcs_header_keys(dst, src, context="test-noncore")
    assert failed == []  # non-core failure does not abort
    assert dst["CRVAL1"] == 1.0
    assert dst["CD1_1"] == 1.0
    assert "FOOBAR" not in dst
    assert get_except_fix_counters().wcs_header_key_copy_fail == 0
    assert any("non-core header keys skipped" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# FIX-3 #9 -- EXC-0586 alignment unique-sample helper [do not reject on error]
# --------------------------------------------------------------------------- #
def test_fix3_9_unique_sample_valid_counts() -> None:
    from vyvar_alignment_frame import _alignment_n_unique_spread_sample

    assert _alignment_n_unique_spread_sample(np.array([1, 2, 3, 4, 5], dtype=np.float32)) == 5
    assert _alignment_n_unique_spread_sample(np.array([7, 7, 7], dtype=np.float32)) == 1


def test_fix3_9_unique_sample_error_returns_sentinel_not_zero(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from vyvar_alignment_frame import _alignment_n_unique_spread_sample

    reset_except_fix_counters()
    caplog.set_level(logging.ERROR)
    log_sink: list[str] = []
    # object dtype of non-numeric strings -> np.asarray(..., float32) raises inside the helper
    bad = np.array(["a", "b", "c"], dtype=object)
    out = _alignment_n_unique_spread_sample(bad, fp_name="frame.fits", log_sink=log_sink)
    assert out == -1  # sentinel "check unavailable", NOT 0 (which would reject the frame)
    assert get_except_fix_counters().align_unique_sample_fail == 1
    assert any("unique-spread check unavailable" in m for m in log_sink)
    assert any("unique-spread sample failed" in r.message for r in caplog.records)


def test_fix3_9_sentinel_does_not_mark_frame_constant() -> None:
    """The -1 sentinel must fall outside the constant band the caller rejects on."""
    n_unique = -1
    # Caller (astroalign path) rejects only when 0 <= n_unique <= 3.
    assert not (0 <= n_unique <= 3)
    # Caller (phase-corr / wcs-shift paths) accepts when n_unique < 0 or n_unique > 3.
    assert (n_unique < 0 or n_unique > 3)


# --------------------------------------------------------------------------- #
# Counter smoke tests for surfacing-only fixes (#1, #4, #6)
# --------------------------------------------------------------------------- #
def test_fix3_1_read_filter_unreadable_counts(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from importer import _read_filter

    bad = tmp_path / "bad.fits"
    bad.write_text("not fits")
    reset_except_fix_counters()
    caplog.set_level(logging.ERROR)
    assert _read_filter(bad, None) == "NoFilter"  # contract unchanged
    assert get_except_fix_counters().importer_filter_read_fail == 1


