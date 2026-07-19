"""PER-FRAME-SAT-GATED: synthetic decision + apply path tests (no Archive)."""

from __future__ import annotations

from pathlib import Path

import math
import pandas as pd
import pytest

from photometry_core import (
    apply_per_frame_saturation_to_active_targets,
    decide_target_saturation_policy,
)


def _flags(n_sat: int, n_total: int = 20) -> list[bool]:
    return [True] * n_sat + [False] * (n_total - n_sat)


def test_decide_off_legacy_identical() -> None:
    sat = _flags(6)
    off = decide_target_saturation_policy(
        zone_flag="saturated",
        legacy_skip=True,
        frame_saturated=sat,
        enabled=False,
        min_clean_frac=0.5,
    )
    assert off["skip_photometry"] is True
    assert off["skip_reason"] == "zone_flag"
    assert off["per_frame_sat_fallback"] is False
    assert math.isnan(off["sat_clean_frac"])


def test_decide_s1_on_rescues_partial_saturation() -> None:
    """S1: 6/20 saturated -> clean_frac=0.7 >= 0.5 -> measure."""
    sat = _flags(6)
    on = decide_target_saturation_policy(
        zone_flag="likely_saturated",
        legacy_skip=True,
        frame_saturated=sat,
        enabled=True,
        min_clean_frac=0.5,
        likely_saturated=True,
    )
    assert on["skip_photometry"] is False
    assert on["skip_reason"] == ""
    assert on["sat_clean_frac"] == pytest.approx(0.7)
    assert on["n_clean"] == 14
    assert on["per_frame_sat_fallback"] is False


def test_decide_s2_on_skips_low_clean_frac() -> None:
    """S2: 15/20 saturated -> clean_frac=0.25 < 0.5 -> skip per_frame_saturation."""
    sat = _flags(15)
    on = decide_target_saturation_policy(
        zone_flag="saturated",
        legacy_skip=True,
        frame_saturated=sat,
        enabled=True,
        min_clean_frac=0.5,
    )
    assert on["skip_photometry"] is True
    assert on["skip_reason"] == "per_frame_saturation"
    assert on["sat_clean_frac"] == pytest.approx(0.25)


def test_decide_s2_off_also_skips() -> None:
    sat = _flags(15)
    off = decide_target_saturation_policy(
        zone_flag="saturated",
        legacy_skip=True,
        frame_saturated=sat,
        enabled=False,
        min_clean_frac=0.5,
    )
    assert off["skip_photometry"] is True
    assert off["skip_reason"] == "zone_flag"


def test_decide_s3_linear_identical_both_modes() -> None:
    sat = _flags(0)
    off = decide_target_saturation_policy(
        zone_flag="linear",
        legacy_skip=False,
        frame_saturated=sat,
        enabled=False,
    )
    on = decide_target_saturation_policy(
        zone_flag="linear",
        legacy_skip=False,
        frame_saturated=sat,
        enabled=True,
    )
    assert off["skip_photometry"] is False
    assert on["skip_photometry"] is False
    assert on["sat_clean_frac"] == pytest.approx(1.0)
    assert off["skip_reason"] == ""
    assert on["skip_reason"] == ""


def test_decide_fallback_missing_frame_data() -> None:
    dec = decide_target_saturation_policy(
        zone_flag="saturated",
        legacy_skip=True,
        frame_saturated=None,
        enabled=True,
        min_clean_frac=0.5,
    )
    assert dec["skip_photometry"] is True
    assert dec["skip_reason"] == "zone_flag"
    assert dec["per_frame_sat_fallback"] is True


def test_decide_fallback_empty_flags() -> None:
    dec = decide_target_saturation_policy(
        zone_flag="saturated",
        legacy_skip=True,
        frame_saturated=[],
        enabled=True,
    )
    assert dec["per_frame_sat_fallback"] is True
    assert dec["skip_photometry"] is True


def test_flag_off_apply_is_noop() -> None:
    at = pd.DataFrame(
        {
            "catalog_id": ["1001", "1002"],
            "zone_flag": ["saturated", "linear"],
            "skip_photometry": [True, False],
        }
    )
    before = at.copy()
    meta = apply_per_frame_saturation_to_active_targets(
        at,
        csv_files=[],
        csv_cache={},
        sat_limit_adu=50000.0,
        enabled=False,
        min_clean_frac=0.5,
    )
    assert meta == {}
    assert list(at.columns) == list(before.columns)
    pd.testing.assert_frame_equal(at, before)


def _build_synthetic_night(tmp_path: Path) -> tuple[pd.DataFrame, list[Path], dict]:
    """N=20 frames; S1 6/20 sat, S2 15/20 sat, S3 0/20 sat."""
    n = 20
    s1, s2, s3 = "1001", "1002", "1003"
    sat_limit = 50000.0
    csv_files: list[Path] = []
    cache: dict[str, pd.DataFrame] = {}
    for i in range(n):
        rows = []
        for cid, n_sat in ((s1, 6), (s2, 15), (s3, 0)):
            is_sat = i < n_sat
            peak = sat_limit + 1000.0 if is_sat else sat_limit * 0.3
            rows.append(
                {
                    "catalog_id": cid,
                    "peak_max_adu": peak,
                    "is_saturated": is_sat,
                    "dao_flux": 1000.0,
                    "x": 10.0,
                    "y": 10.0,
                }
            )
        path = tmp_path / f"proc_{i:04d}.csv"
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
        csv_files.append(path)
        cache[str(path)] = df

    at = pd.DataFrame(
        {
            "catalog_id": [s1, s2, s3],
            "zone_flag": ["likely_saturated", "saturated", "linear"],
            "skip_photometry": [True, True, False],
            "likely_saturated": [True, True, False],
            "x": [10.0, 20.0, 30.0],
            "y": [10.0, 20.0, 30.0],
            "mag": [10.0, 9.5, 12.0],
        }
    )
    return at, csv_files, cache


def test_apply_synthetic_night_on_semantics(tmp_path: Path) -> None:
    at, csv_files, cache = _build_synthetic_night(tmp_path)
    meta = apply_per_frame_saturation_to_active_targets(
        at,
        csv_files=csv_files,
        csv_cache=cache,
        sat_limit_adu=50000.0,
        enabled=True,
        min_clean_frac=0.5,
    )
    by_id = at.set_index("catalog_id")
    # S1 rescued
    assert bool(by_id.loc["1001", "skip_photometry"]) is False
    assert float(by_id.loc["1001", "sat_clean_frac"]) == pytest.approx(0.7)
    assert str(by_id.loc["1001", "skip_reason"]) == ""
    # S2 skipped with per_frame reason
    assert bool(by_id.loc["1002", "skip_photometry"]) is True
    assert str(by_id.loc["1002", "skip_reason"]) == "per_frame_saturation"
    assert float(by_id.loc["1002", "sat_clean_frac"]) == pytest.approx(0.25)
    # S3 unchanged measured
    assert bool(by_id.loc["1003", "skip_photometry"]) is False
    assert float(by_id.loc["1003", "sat_clean_frac"]) == pytest.approx(1.0)
    assert meta["per_frame_sat_n_rescued"] == 1
    assert meta["per_frame_sat_n_skipped"] == 1
    assert meta["per_frame_sat_n_fallback"] == 0


def test_apply_synthetic_night_off_legacy(tmp_path: Path) -> None:
    at, csv_files, cache = _build_synthetic_night(tmp_path)
    meta = apply_per_frame_saturation_to_active_targets(
        at,
        csv_files=csv_files,
        csv_cache=cache,
        sat_limit_adu=50000.0,
        enabled=False,
        min_clean_frac=0.5,
    )
    assert meta == {}
    # OFF leaves skip_photometry as constructed (S1/S2 skipped, S3 not)
    assert list(at["skip_photometry"]) == [True, True, False]
    assert "sat_clean_frac" not in at.columns


def test_apply_fallback_when_peak_missing(tmp_path: Path) -> None:
    path = tmp_path / "proc_0000.csv"
    df = pd.DataFrame({"catalog_id": ["1001"], "dao_flux": [1.0]})  # no peak/sat
    df.to_csv(path, index=False)
    at = pd.DataFrame(
        {
            "catalog_id": ["1001"],
            "zone_flag": ["saturated"],
            "skip_photometry": [True],
        }
    )
    meta = apply_per_frame_saturation_to_active_targets(
        at,
        csv_files=[path],
        csv_cache={str(path): df},
        sat_limit_adu=50000.0,
        enabled=True,
        min_clean_frac=0.5,
    )
    assert bool(at.loc[0, "skip_photometry"]) is True
    assert str(at.loc[0, "skip_reason"]) == "zone_flag"
    assert bool(at.loc[0, "per_frame_sat_fallback"]) is True
    assert meta["per_frame_sat_n_fallback"] == 1


def test_off_equivalence_mixed_fixture_decisions() -> None:
    """Flag-OFF decide outputs match legacy for a mixed zone fixture."""
    cases = [
        ("saturated", True, _flags(6)),
        ("saturated", True, _flags(15)),
        ("linear", False, _flags(0)),
        ("noisy1", False, _flags(2)),
    ]
    for zf, legacy, flags in cases:
        off = decide_target_saturation_policy(
            zone_flag=zf,
            legacy_skip=legacy,
            frame_saturated=flags,
            enabled=False,
        )
        # Legacy: skip iff legacy_skip or zone saturated
        expect = bool(legacy) or zf == "saturated"
        assert off["skip_photometry"] is expect
        assert off["skip_reason"] == ("zone_flag" if expect else "")
