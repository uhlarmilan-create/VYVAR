"""SAT-TARGET-2026-10 / TARGET-RULES-2026-10: per-frame saturation of measurable targets.

Target-selection skips (saturated on MASTERSTAR, S/N below snr_min, VSX type out of scope) are final;
the per-frame test only removes a target linear on MASTERSTAR that is saturated in most frames.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from photometry_core import (
    _per_frame_sat_flags_for_catalog_id,
    apply_per_frame_saturation_to_active_targets,
    decide_per_frame_saturation,
)
from pipeline import (
    SAT_LIMIT_CONTAINER_CLIP_ADU,
    SAT_LIMIT_NO_KNEE_FRAC,
    _annotate_masterstars_flux_zones,
    inv_sat_limit_peak_test_adu,
)


def _flags(n_sat: int, n_total: int = 20) -> list[bool]:
    return [True] * n_sat + [False] * (n_total - n_sat)


def test_recorded_target_selection_skip_is_final() -> None:
    for reason in ("saturated", "below_snr_min", "vsx_type_out_of_scope", "per_frame_saturation"):
        d = decide_per_frame_saturation(skip_reason=reason, frame_saturated=_flags(0))
        assert d["skip_photometry"] and d["skip_reason"] == reason


def test_linear_target_saturated_in_most_frames_is_skipped() -> None:
    d = decide_per_frame_saturation(skip_reason="", frame_saturated=[True] * 71 + [False] * 51)
    assert d["skip_photometry"] and d["skip_reason"] == "per_frame_saturation"
    d = decide_per_frame_saturation(skip_reason="", frame_saturated=[True] * 5 + [False] * 117)
    assert not d["skip_photometry"] and d["skip_reason"] == ""
    d = decide_per_frame_saturation(skip_reason="", frame_saturated=None)  # no per-frame verdict
    assert not d["skip_photometry"] and math.isnan(d["sat_clean_frac"])


def _night(tmp_path: Path) -> tuple[pd.DataFrame, list[Path], dict]:
    """20 frames; 1001 saturated in 6, 1002 in 15, 1003 in none, 1004 skipped in target selection."""
    files, cache = [], {}
    for i in range(20):
        rows = [{"catalog_id": cid, "peak_max_adu": 51000.0 if i < k else 15000.0}
                for cid, k in (("1001", 6), ("1002", 15), ("1003", 0), ("1004", 20))]
        p = tmp_path / f"proc_{i:04d}.csv"
        df = pd.DataFrame(rows)
        df.to_csv(p, index=False)
        files.append(p)
        cache[str(p)] = df
    at = pd.DataFrame({
        "catalog_id": ["1001", "1002", "1003", "1004"],
        "zone_flag": ["linear", "linear", "linear", "saturated"],
        "skip_photometry": [False, False, False, True],
        "skip_reason": [float("nan"), float("nan"), float("nan"), "saturated"],
    })
    return at, files, cache


def test_apply_night(tmp_path: Path) -> None:
    at, files, cache = _night(tmp_path)
    meta = apply_per_frame_saturation_to_active_targets(
        at, csv_files=files, csv_cache=cache, min_clean_frac=0.5, peak_test_adu=50000.0)
    by = at.set_index("catalog_id")
    assert not by.loc["1001", "skip_photometry"] and "sat_clean_frac" not in at.columns  # OUTPUT-REVIEW step 6
    assert by.loc["1002", "skip_photometry"] and by.loc["1002", "skip_reason"] == "per_frame_saturation"
    assert not by.loc["1003", "skip_photometry"] and by.loc["1003", "skip_reason"] == ""
    # A target saturated in target selection is never revived, even with clean frames elsewhere
    assert by.loc["1004", "skip_photometry"] and by.loc["1004", "skip_reason"] == "saturated"
    # NaN from CSV is not written as the reason "nan"
    assert "nan" not in set(at["skip_reason"].astype(str))
    assert meta["per_frame_sat_n_skipped"] == 1


def test_per_frame_peak_test_equals_catalog_authority(tmp_path: Path) -> None:
    """Per-frame peak test == MASTERSTAR zone peak test (0.90 x clip); peak in (58981.5, 65535] is not clean."""
    peak_test, src = inv_sat_limit_peak_test_adu()
    assert peak_test == pytest.approx(SAT_LIMIT_CONTAINER_CLIP_ADU * SAT_LIMIT_NO_KNEE_FRAC)
    cat = _annotate_masterstars_flux_zones(
        pd.DataFrame({"flux": [8000.0], "peak_dao": [45.0], "peak_max_adu": [60000.0]}),
        equipment_saturate_adu=None, saturate_limit_adu_fallback=None,
        sigma_px=10.0, sky_median_adu=1955.0, dao_detection_n_equiv=3.78,
    )
    assert float(cat.loc[0, "saturate_limit_adu_85pct"]) == pytest.approx(peak_test)
    assert str(cat.loc[0, "zone"]) == "saturated"
    cid = "1497007144465726080"
    p = tmp_path / "proc_0000.csv"
    df = pd.DataFrame({"catalog_id": [cid], "peak_max_adu": [60000.0], "is_saturated": [False]})
    df.to_csv(p, index=False)
    assert _per_frame_sat_flags_for_catalog_id(cid, [p], {str(p): df}, peak_test_adu=peak_test) == [True]
    at = pd.DataFrame({"catalog_id": [cid], "zone_flag": ["linear"], "skip_photometry": [False]})
    apply_per_frame_saturation_to_active_targets(
        at, csv_files=[p], csv_cache={str(p): df}, min_clean_frac=0.5, peak_test_adu=peak_test,
        peak_test_source=src)
    assert at.loc[0, "skip_reason"] == "per_frame_saturation"


def test_aperture_rerun_of_old_active_targets(tmp_path: Path) -> None:
    """Review 2026-10-13: active_targets.csv from the old code (rescued saturated target, legacy
    reasons) and aperture-photometry reasons that are not selection skips (no_comps)."""
    at, files, cache = _night(tmp_path)
    at = pd.DataFrame({
        "catalog_id": ["1001", "1002", "1003", "1004"],
        # 1001: MASTERSTAR saturated, cleared by the old per-frame rescue; 1002 legacy depth skip;
        # 1003: no_comps written by aperture photometry (measurable); 1004 legacy zone_flag skip
        "zone_flag": ["saturated", "linear", "linear", "saturated"],
        "skip_photometry": [False, True, False, True],
        "skip_reason": ["", "below_target_depth", "no_comps", "zone_flag"],
    })
    apply_per_frame_saturation_to_active_targets(
        at, csv_files=files, csv_cache=cache, min_clean_frac=0.5, peak_test_adu=50000.0)
    by = at.set_index("catalog_id")
    assert by.loc["1001", "skip_photometry"] and by.loc["1001", "skip_reason"] == "saturated"
    assert by.loc["1002", "skip_photometry"] and by.loc["1002", "skip_reason"] == "below_snr_min"
    assert not by.loc["1003", "skip_photometry"] and by.loc["1003", "skip_reason"] == ""
    assert by.loc["1004", "skip_photometry"] and by.loc["1004", "skip_reason"] == "saturated"
