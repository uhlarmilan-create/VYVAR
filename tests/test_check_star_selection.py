"""Check-star selection hardening (CS-2 / CS-3 / CS-4)."""

from __future__ import annotations

import pandas as pd
import pytest

from check_star_kmag import select_check_star
from config import AppConfig


def _good_row(cid: str, *, rms: float, tier: int = 1) -> dict:
    return {
        "catalog_id": cid,
        "status": "good",
        "comp_rms": rms,
        "p2p_rms": rms,
        "comp_tier": tier,
        "contamination_idx": 0.0,
    }


def test_cs3_never_picks_ensemble_member() -> None:
    df = pd.DataFrame(
        [
            _good_row("100", rms=0.001),
            _good_row("200", rms=0.002),
            _good_row("300", rms=0.003),
            _good_row("400", rms=0.02),
            _good_row("500", rms=0.03),
            _good_row("600", rms=0.04),
        ]
    )
    row = select_check_star(df, ensemble_ids={"100", "200", "300"}, n_comp_min=3)
    assert row is not None
    assert str(row["catalog_id"]) == "400"


def test_cs3_all_low_rms_in_ensemble_returns_none() -> None:
    df = pd.DataFrame(
        [
            _good_row("100", rms=0.001),
            _good_row("200", rms=0.002),
            _good_row("300", rms=0.003),
        ]
    )
    assert select_check_star(df, ensemble_ids={"100", "200", "300"}, n_comp_min=3) is None


def test_cs3_independent_star_preferred_over_ensemble() -> None:
    df = pd.DataFrame(
        [
            _good_row("ens", rms=0.0001),
            _good_row("ind", rms=0.02),
            _good_row("c3", rms=0.03),
            _good_row("c4", rms=0.04),
        ]
    )
    row = select_check_star(df, ensemble_ids={"ens"}, n_comp_min=3)
    assert row is not None
    assert str(row["catalog_id"]) == "ind"


def test_cs2_artefact_rms_not_selected() -> None:
    df = pd.DataFrame(
        [
            _good_row("art", rms=0.0),
            _good_row("real", rms=0.01),
            _good_row("c3", rms=0.02),
            _good_row("c4", rms=0.03),
        ]
    )
    row = select_check_star(df, ensemble_ids=set(), n_comp_min=3, cfg=AppConfig())
    assert row is not None
    assert str(row["catalog_id"]) == "real"


def test_cs2_all_below_floor_returns_none() -> None:
    df = pd.DataFrame(
        [
            _good_row("a", rms=0.0),
            _good_row("b", rms=0.0),
            _good_row("c", rms=0.0),
        ]
    )
    assert select_check_star(df, ensemble_ids=set(), n_comp_min=3) is None


def test_cs4_high_contamination_excluded() -> None:
    hi = _good_row("crowded", rms=0.001)
    hi["contamination_idx"] = 0.5
    df = pd.DataFrame(
        [
            hi,
            _good_row("ok1", rms=0.02),
            _good_row("ok2", rms=0.03),
            _good_row("ok3", rms=0.04),
        ]
    )
    row = select_check_star(df, ensemble_ids=set(), n_comp_min=3, cfg=AppConfig())
    assert row is not None
    assert str(row["catalog_id"]) != "crowded"
