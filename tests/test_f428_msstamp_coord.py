"""F-428-MS-STAMP + F-428-COORD unit tests."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from gaia_catalog_id import normalize_gaia_source_id
from photometry_core import stamp_vsx_known_variable_on_masterstars


def test_stamp_vsx_by_catalog_id_despite_coordinate_offset() -> None:
    """Shared Gaia ID must stamp True even when vt/ms sky coords disagree."""
    cid = "1400549806859236864"
    ms = pd.DataFrame(
        {
            "catalog_id": [cid],
            "name": [cid],
            "ra_deg": [202.5],
            "dec_deg": [42.5],
            "x": [500.0],
            "y": [600.0],
            "vsx_known_variable": [False],
        }
    )
    vt = pd.DataFrame(
        {
            "catalog_id": [cid],
            "vsx_name": ["BO CVn"],
            "ra_deg": [202.0],
            "dec_deg": [42.0],
        }
    )
    out, stats = stamp_vsx_known_variable_on_masterstars(ms, vt)
    assert stats["id_join"] == 1
    assert bool(out.iloc[0]["vsx_known_variable"]) is True


def test_stamp_vsx_positional_fallback_without_catalog_id() -> None:
    ms = pd.DataFrame(
        {
            "catalog_id": [""],
            "name": ["DET_0001"],
            "ra_deg": [100.0],
            "dec_deg": [20.0],
            "vsx_known_variable": [False],
        }
    )
    vt = pd.DataFrame(
        {
            "catalog_id": [""],
            "vsx_name": ["NOID"],
            "ra_deg": [100.00001],
            "dec_deg": [20.00001],
        }
    )
    out, stats = stamp_vsx_known_variable_on_masterstars(ms, vt, positional_fallback_arcsec=5.0)
    assert stats["id_join"] == 0
    assert stats["positional_fallback"] == 1
    assert bool(out.iloc[0]["vsx_known_variable"]) is True


def test_stamp_post_finalize_after_optimizer_assigns_ids() -> None:
    """F-429-STAMP-WIRE: stamp after finalize must id-join when catalog_id present."""
    cid = "1400549806859236864"
    ms_pre = pd.DataFrame(
        {
            "catalog_id": [cid],
            "name": [cid],
            "ra_deg": [202.5],
            "dec_deg": [42.5],
            "x": [500.0],
            "y": [600.0],
            "vsx_known_variable": [False],
            "coord_source": ["gaia_catalog"],
        }
    )
    vt = pd.DataFrame(
        {
            "catalog_id": [cid],
            "vsx_name": ["BO CVn"],
            "ra_deg": [202.0],
            "dec_deg": [42.0],
        }
    )
    out, stats = stamp_vsx_known_variable_on_masterstars(ms_pre, vt)
    assert stats["id_join"] == 1
    assert stats["positional_fallback"] == 0
    assert bool(out.iloc[0]["vsx_known_variable"]) is True


def test_stamp_before_optimizer_has_zero_id_join() -> None:
    """Early stamp (pre-optimizer) should not id-join when catalog_id still empty."""
    cid = "1400549806859236864"
    ms = pd.DataFrame(
        {
            "catalog_id": [""],
            "name": ["DET_0001"],
            "ra_deg": [202.5],
            "dec_deg": [42.5],
            "x": [500.0],
            "y": [600.0],
            "vsx_known_variable": [False],
        }
    )
    vt = pd.DataFrame(
        {
            "catalog_id": [cid],
            "vsx_name": ["BO CVn"],
            "ra_deg": [202.0],
            "dec_deg": [42.0],
        }
    )
    out, stats = stamp_vsx_known_variable_on_masterstars(ms, vt)
    assert stats["id_join"] == 0
    assert bool(out.iloc[0]["vsx_known_variable"]) is False


def test_draft428_dry_run_stamp_prediction() -> None:
    """Dry-run: count vt IDs present in masterstars (post-fix expectation)."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    ms_path = root / "Archive/Drafts/draft_000428/platesolve/NoFilter_60_2/masterstars_full_match.csv"
    vt_path = root / "Archive/Drafts/draft_000428/platesolve/NoFilter_60_2/variable_targets.csv"
    if not ms_path.is_file() or not vt_path.is_file():
        pytest.skip("draft_428 archive not available")

    ms = pd.read_csv(ms_path, dtype={"catalog_id": str})
    vt = pd.read_csv(vt_path, dtype={"catalog_id": str})
    ms_ids = {normalize_gaia_source_id(x) for x in ms["catalog_id"] if str(x).strip()}
    vt_ids = {normalize_gaia_source_id(x) for x in vt["catalog_id"] if str(x).strip()}
    predicted = len(ms_ids & vt_ids)
    assert predicted > 46, f"expected stamp count well above 46, got {predicted}"
