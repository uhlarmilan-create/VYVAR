# -*- coding: ascii -*-
"""C0a: one _dist_deg function; persisted column quantized to 1e-9 deg."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from sky_separation import (
    DIST_DEG_INVALID,
    DIST_DEG_QUANTUM,
    angular_distance_deg,
    angular_distance_deg_vectorized,
    persist_dist_deg_column,
    quantize_dist_deg,
)


def _math_haversine(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """Pre-C0a photometry_core scalar (math module)."""
    r1, d1, r2, d2 = map(math.radians, [ra1, dec1, ra2, dec2])
    a = (
        math.sin((d2 - d1) / 2) ** 2
        + math.cos(d1) * math.cos(d2) * math.sin((r2 - r1) / 2) ** 2
    )
    return math.degrees(2 * math.asin(min(1.0, math.sqrt(a))))


def test_scalar_wraps_vectorized() -> None:
    ra1, de1, ra2, de2 = 210.123456789, 37.987654321, 210.234567891, 38.012345678
    s = angular_distance_deg(ra1, de1, ra2, de2)
    v = float(angular_distance_deg_vectorized(ra1, de1, np.array([ra2]), np.array([de2]))[0])
    assert s == v


def test_math_vs_numpy_ulp_collapses_at_quantum() -> None:
    """The old two formulas can differ at ~1e-14; persisted column must not."""
    pairs = [
        (210.1234567890123, 37.9876543210987, 210.2345678901234, 38.0123456789012),
        (149.0, 32.5, 149.001, 32.5001),
        (0.0, 0.0, 180.0, 0.0),
    ]
    for ra1, de1, ra2, de2 in pairs:
        old_math = _math_haversine(ra1, de1, ra2, de2)
        new = angular_distance_deg(ra1, de1, ra2, de2)
        q_old = quantize_dist_deg(old_math)
        q_new = quantize_dist_deg(new)
        assert q_old == q_new
        assert abs(float(q_new) - float(new)) < DIST_DEG_QUANTUM * 0.6


def test_invalid_sentinel_not_quantized() -> None:
    out = angular_distance_deg_vectorized(1.0, 2.0, np.array([np.nan]), np.array([3.0]))
    assert float(out[0]) == DIST_DEG_INVALID
    assert float(quantize_dist_deg(out)[0]) == DIST_DEG_INVALID


def test_persist_string_tokens_byte_identical(tmp_path: Path) -> None:
    df = pd.DataFrame({"_dist_deg": [0.012345678901234, 0.012345678901235, 999.0]})
    persist_dist_deg_column(df)
    p1 = tmp_path / "a.csv"
    p2 = tmp_path / "b.csv"
    df.to_csv(p1, index=False)
    df2 = pd.read_csv(p1)
    persist_dist_deg_column(df2)
    df2.to_csv(p2, index=False)
    assert p1.read_bytes() == p2.read_bytes()
    assert df["_dist_deg"].tolist()[0] == "0.012345679"


def test_b3_snap_vs_phot_only_match_after_quantize() -> None:
    """B3 T3-P5: only _dist_deg differed; after 1e-9 quantize the files agree."""
    session = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "session_20260825_sel_ghost_01_b3"
    )
    a = session / "candidate_516_snapshot" / "platesolve" / "NoFilter_60_2" / "photometry" / "comparison_stars_per_target.csv"
    b = session / "t3_p5_full" / "platesolve" / "NoFilter_60_2" / "photometry" / "comparison_stars_per_target.csv"
    if not a.is_file() or not b.is_file():
        return
    da = pd.read_csv(a)
    db = pd.read_csv(b)
    persist_dist_deg_column(da)
    persist_dist_deg_column(db)
    pd.testing.assert_frame_equal(da, db, check_dtype=False)


def test_comp_selection_aliases_are_sky_separation() -> None:
    from comp_selection_per_target import (
        _angular_distance_deg_vectorized,
        _pixel_distance_deg_vectorized,
    )
    from sky_separation import pixel_distance_deg_vectorized

    assert _angular_distance_deg_vectorized is angular_distance_deg_vectorized
    assert _pixel_distance_deg_vectorized is pixel_distance_deg_vectorized
