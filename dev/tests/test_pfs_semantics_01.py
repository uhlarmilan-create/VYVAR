"""PFS-SEMANTICS-01 guard tests. Order is required: hole, then post-fix, then threshold."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from photometry_core import (
    _per_frame_sat_flags_for_catalog_id,
    apply_per_frame_saturation_to_active_targets,
    decide_target_saturation_policy,
    pfs_rescue_eligible,
)
from pipeline import (
    SAT_LIMIT_CONTAINER_CLIP_ADU,
    SAT_LIMIT_NO_KNEE_FRAC,
    _annotate_masterstars_flux_zones,
    inv_sat_limit_peak_test_adu,
)


def _pre_fix_decide_target_saturation_policy(
    *,
    zone_flag: str,
    legacy_skip: bool,
    frame_saturated: list[bool] | None,
    enabled: bool,
    min_clean_frac: float = 0.5,
    likely_saturated: bool = False,
) -> dict:
    """Inlined pre-PFS-SEMANTICS-01 decide (hole proof). Do not use in production."""
    zf = str(zone_flag or "").strip().lower()
    legacy = bool(legacy_skip) or zf == "saturated"
    advisory = bool(legacy) or zf in ("saturated", "likely_saturated") or bool(likely_saturated)
    thr = max(0.1, min(1.0, float(min_clean_frac)))
    if not enabled:
        return {"skip_photometry": bool(legacy), "skip_reason": "zone_flag" if legacy else ""}
    if frame_saturated is None:
        return {"skip_photometry": bool(legacy), "skip_reason": "zone_flag" if legacy else ""}
    flags = [bool(x) for x in list(frame_saturated)]
    n = len(flags)
    if n == 0:
        return {"skip_photometry": bool(legacy), "skip_reason": "zone_flag" if legacy else ""}
    n_clean = int(sum(1 for s in flags if not s))
    clean_frac = float(n_clean) / float(n)
    if not advisory:
        return {"skip_photometry": False, "skip_reason": "", "sat_clean_frac": clean_frac}
    if clean_frac >= thr:
        return {"skip_photometry": False, "skip_reason": "", "sat_clean_frac": clean_frac}
    return {
        "skip_photometry": True,
        "skip_reason": "per_frame_saturation",
        "sat_clean_frac": clean_frac,
    }


def test_a_pre_fix_pfs_on_rescues_zone_noise() -> None:
    """(a) Pre-fix code: PFS ON + zone_noise target -> rescue fires (the hole)."""
    sat = [False] * 20
    old = _pre_fix_decide_target_saturation_policy(
        zone_flag="noise",
        legacy_skip=True,
        frame_saturated=sat,
        enabled=True,
        min_clean_frac=0.5,
    )
    assert old["skip_photometry"] is False
    assert old["skip_reason"] == ""
    assert old["sat_clean_frac"] == pytest.approx(1.0)


def test_b_post_fix_does_not_clear_zone_noise_or_depth(tmp_path: Path) -> None:
    """(b) PFS ON does not clear zone_noise or below_target_depth; sat zone is rescued."""
    assert pfs_rescue_eligible(zone_flag="noise", skip_reason="zone_noise") is False
    assert pfs_rescue_eligible(
        zone_flag="linear", skip_reason="below_target_depth"
    ) is False
    assert pfs_rescue_eligible(zone_flag="saturated", skip_reason="zone_flag") is True

    n = 10
    noise_id, depth_id, sat_id = "2001", "2002", "2003"
    peak_test, _src = inv_sat_limit_peak_test_adu()
    csv_files: list[Path] = []
    cache: dict[str, pd.DataFrame] = {}
    for i in range(n):
        rows = [
            {
                "catalog_id": noise_id,
                "peak_max_adu": 1000.0,
                "is_saturated": False,
            },
            {
                "catalog_id": depth_id,
                "peak_max_adu": 1000.0,
                "is_saturated": False,
            },
            {
                "catalog_id": sat_id,
                "peak_max_adu": 1000.0,
                "is_saturated": False,
            },
        ]
        path = tmp_path / f"proc_{i:04d}.csv"
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
        csv_files.append(path)
        cache[str(path)] = df

    at = pd.DataFrame(
        {
            "catalog_id": [noise_id, depth_id, sat_id],
            "zone_flag": ["noise", "linear", "saturated"],
            "skip_photometry": [True, True, True],
            "skip_reason": ["zone_noise", "below_target_depth", "zone_flag"],
        }
    )
    meta = apply_per_frame_saturation_to_active_targets(
        at,
        csv_files=csv_files,
        csv_cache=cache,
        sat_limit_adu=None,
        enabled=True,
        min_clean_frac=0.5,
        peak_test_adu=peak_test,
        peak_test_source=_src,
    )
    by_id = at.set_index("catalog_id")
    assert bool(by_id.loc[noise_id, "skip_photometry"]) is True
    assert str(by_id.loc[noise_id, "skip_reason"]) == "zone_noise"
    assert bool(by_id.loc[depth_id, "skip_photometry"]) is True
    assert str(by_id.loc[depth_id, "skip_reason"]) == "below_target_depth"
    assert bool(by_id.loc[sat_id, "skip_photometry"]) is False
    assert str(by_id.loc[sat_id, "skip_reason"]) == ""
    assert float(by_id.loc[sat_id, "sat_clean_frac"]) == pytest.approx(1.0)
    assert int(meta["per_frame_sat_n_rescued"]) == 1

    noise_dec = decide_target_saturation_policy(
        zone_flag="noise",
        legacy_skip=True,
        frame_saturated=[False] * 10,
        enabled=True,
        skip_reason="zone_noise",
    )
    assert noise_dec["skip_photometry"] is True
    assert noise_dec["skip_reason"] == "zone_noise"


def test_c_per_frame_peak_test_equals_catalog_authority(tmp_path: Path) -> None:
    """(c) Per-frame test value == catalog peak-test; peak in (52428, 65535] is not clean."""
    peak_test, src = inv_sat_limit_peak_test_adu()
    assert peak_test == pytest.approx(
        SAT_LIMIT_CONTAINER_CLIP_ADU * SAT_LIMIT_NO_KNEE_FRAC
    )
    assert peak_test == pytest.approx(52428.0)
    assert "INV-SAT-LIMIT" in src
    assert SAT_LIMIT_CONTAINER_CLIP_ADU == pytest.approx(65535.0)
    assert peak_test != SAT_LIMIT_CONTAINER_CLIP_ADU

    cat = _annotate_masterstars_flux_zones(
        pd.DataFrame(
            {
                "flux": [8000.0],
                "peak_dao": [45.0],
                "peak_max_adu": [55000.0],
            }
        ),
        noise_floor_adu=2105.9,
        equipment_saturate_adu=None,
        saturate_limit_adu_fallback=None,
        sigma_px=10.0,
        sky_median_adu=1955.0,
        prematch_peak_sigma_floor=1.8,
        dao_detection_n_equiv=3.78,
    )
    catalog_peak_test = float(cat.loc[0, "saturate_limit_adu_85pct"])
    assert catalog_peak_test == pytest.approx(peak_test)
    assert str(cat.loc[0, "zone"]) == "saturated"

    mid_peak = 55000.0
    assert peak_test < mid_peak < SAT_LIMIT_CONTAINER_CLIP_ADU
    cid = "1497007144465726080"
    path = tmp_path / "proc_0000.csv"
    df = pd.DataFrame(
        {
            "catalog_id": [cid],
            "peak_max_adu": [mid_peak],
            "is_saturated": [False],
        }
    )
    df.to_csv(path, index=False)
    flags = _per_frame_sat_flags_for_catalog_id(
        cid,
        [path],
        {str(path): df},
        peak_test_adu=peak_test,
    )
    assert flags == [True]

    at = pd.DataFrame(
        {
            "catalog_id": [cid],
            "zone_flag": ["saturated"],
            "skip_photometry": [True],
            "skip_reason": ["zone_flag"],
        }
    )
    apply_per_frame_saturation_to_active_targets(
        at,
        csv_files=[path],
        csv_cache={str(path): df},
        sat_limit_adu=None,
        enabled=True,
        min_clean_frac=0.5,
        peak_test_adu=peak_test,
        peak_test_source=src,
    )
    assert bool(at.loc[0, "skip_photometry"]) is True
    assert str(at.loc[0, "skip_reason"]) == "per_frame_saturation"
    assert float(at.loc[0, "sat_clean_frac"]) == pytest.approx(0.0)
    assert not math.isnan(float(at.loc[0, "sat_clean_frac"]))
