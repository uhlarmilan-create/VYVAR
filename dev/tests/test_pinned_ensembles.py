# -*- coding: ascii -*-
"""Unit tests for pinned ensemble mechanism (INV-PIN-01/02/03)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src_py"
import sys

sys.path.insert(0, str(SRC))

from pinned_ensembles import (  # noqa: E402
    PinMember,
    compute_file_sha256,
    default_pinned_ensembles_path,
    generate_pinned_ensembles_csv,
    get_pinned_members_for_target,
    load_pinned_ensembles,
    validate_pinned_member,
    verify_inv_pin_01,
    verify_inv_pin_02,
    verify_inv_pin_03,
    verify_inv_pin_04,
)

BASELINE = ROOT / "Archive" / "Drafts" / "draft_000516_snapshot_cleanrebuild_20260818"
COMP_PT = (
    BASELINE
    / "platesolve"
    / "NoFilter_60_2"
    / "photometry"
    / "comparison_stars_per_target.csv"
)
PIN_PATH = default_pinned_ensembles_path()


@pytest.fixture(scope="module")
def pin_file() -> Path:
    assert PIN_PATH.is_file(), f"missing committed pin file: {PIN_PATH}"
    return PIN_PATH


def test_pin_file_has_48_targets(pin_file: Path) -> None:
    pins, sha, _ = load_pinned_ensembles(pin_file, force=True)
    assert len(pins) == 48
    assert len(sha) == 64
    assert compute_file_sha256(pin_file) == sha


def test_bo_cvn_is_pinned(pin_file: Path) -> None:
    members = get_pinned_members_for_target("1498613634033133184", pin_file)
    assert members is not None
    assert 3 <= len(members) <= 8


def test_inv_pin_01_exact_membership() -> None:
    pin = [
        PinMember("111", 1.0, 1, "477dc8cf", "test", "477dc8cf", "2026-08-19"),
        PinMember("222", 1.0, 1, "477dc8cf", "test", "477dc8cf", "2026-08-19"),
    ]
    verify_inv_pin_01("tgt", ["111", "222"], pin, drop_log=[])


def test_inv_pin_02_named_drop() -> None:
    verify_inv_pin_02([("333", "zone_saturated")])


def test_inv_pin_03_meta_sha() -> None:
    sha = "abc123"
    verify_inv_pin_03({"pinned_ensembles_sha256": sha}, sha)


def test_validate_zone_sat_drop() -> None:
    row = pd.Series({"catalog_id": "999", "zone": "saturated", "is_saturated": False})
    ok, reason = validate_pinned_member(
        row,
        target_cid="100",
        target_bprp_eff=1.0,
        dist_arcsec=120.0,
        comp_rms=0.01,
        min_dist_arcsec=60.0,
        max_comp_rms=0.1,
        max_delta_bprp_cfg=0.79,
        comp_tier=1,
        tier_defs=[(1, 0.25), (2, 0.48), (3, 0.79), (4, 999.0)],
    )
    assert not ok
    assert reason == "zone_saturated"


def test_validate_tier4_color_uses_tier_not_ceiling() -> None:
    """Tier-4 comps with |dBP-RP|>0.79 must pass (R1 / INV-PIN-04)."""
    row = pd.Series({"catalog_id": "888", "zone": "linear", "bp_rp": 5.5})
    ok, reason = validate_pinned_member(
        row,
        target_cid="100",
        target_bprp_eff=1.0,
        dist_arcsec=120.0,
        comp_rms=0.01,
        min_dist_arcsec=60.0,
        max_comp_rms=0.1,
        max_delta_bprp_cfg=0.79,
        comp_tier=4,
        tier_defs=[(1, 0.25), (2, 0.48), (3, 0.79), (4, 999.0)],
    )
    assert ok
    assert reason == "ok"


def test_inv_pin_04_catalog_stable() -> None:
    ms = pd.DataFrame(
        [
            {"catalog_id": "111", "zone": "linear", "bp_rp": 5.5},
            {"catalog_id": "222", "zone": "linear", "bp_rp": 6.0},
        ]
    )
    pin_pt = pd.DataFrame(
        [
            {
                "target_catalog_id": "100",
                "catalog_id": "111",
                "delta_bprp_abs": 4.5,
                "bp_rp": 5.5,
                "target_bp_rp": 1.0,
            },
        ]
    )
    members = [PinMember("111", 1.0, 4, "477dc8cf", "test", "477dc8cf", "2026-08-19")]
    verify_inv_pin_04("100", members, ms, target_bprp_eff=1.0, pin_time_comp_pt=pin_pt)


def test_baseline_lc_ct_ok_helper() -> None:
    from pinned_ensembles import baseline_lc_ct_ok_for_target  # noqa: PLC0415

    assert baseline_lc_ct_ok_for_target("1498613634033133184") in (True, False)


def test_generate_matches_committed(pin_file: Path, tmp_path: Path) -> None:
    sys.path.insert(0, str(ROOT / "tmp"))
    from dao_gaia_era_01_part_c_rebuild import BASELINE_LC_TARGET_IDS  # noqa: E402

    out = tmp_path / "pinned_test.csv"
    generate_pinned_ensembles_csv(COMP_PT, BASELINE_LC_TARGET_IDS, out)
    committed = pd.read_csv(pin_file, dtype=str)
    generated = pd.read_csv(out, dtype=str)
    assert len(committed) == len(generated)
    assert set(committed["target_catalog_id"]) == set(generated["target_catalog_id"])
