"""Plate-solver SIP full-pair refine + FITS ROWORDER helpers."""
from __future__ import annotations

import numpy as np
import pytest

from vyvar_platesolver import (
    _apply_fits_roworder_to_detections,
    _fits_roworder_yflip_applied,
    _sip_match_max_px,
)


class _Hdr:
    def __init__(self, mapping: dict[str, str]) -> None:
        self._m = mapping

    def get(self, key: str, default=None):
        return self._m.get(key, default)


def test_roworder_bottom_up_yflip():
    hdr = _Hdr({"ROWORDER": "BOTTOM-UP"})
    assert _fits_roworder_yflip_applied(hdr) is True
    xs = np.array([0.0, 10.0])
    ys = np.array([0.0, 100.0])
    x2, y2, tag = _apply_fits_roworder_to_detections(xs, ys, hdr=hdr, naxis2=200)
    assert tag == "bottom_up_yflip"
    np.testing.assert_allclose(x2, xs)
    np.testing.assert_allclose(y2, [199.0, 99.0])


def test_roworder_top_down_unchanged():
    hdr = _Hdr({"ROWORDER": "TOP-DOWN"})
    xs = np.array([1.0, 2.0])
    ys = np.array([3.0, 4.0])
    x2, y2, tag = _apply_fits_roworder_to_detections(xs, ys, hdr=hdr, naxis2=100)
    assert tag is None
    np.testing.assert_allclose(x2, xs)
    np.testing.assert_allclose(y2, ys)


@pytest.mark.parametrize(
    ("coarse", "expected"),
    [
        (42.0, 23.1),
        (10.0, 15.0),
        (100.0, 48.0),
    ],
)
def test_sip_match_max_px(coarse: float, expected: float) -> None:
    assert _sip_match_max_px(coarse) == pytest.approx(expected)
