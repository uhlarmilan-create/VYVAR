"""Regression: duplicate Gaia comp IDs must not crash Phase-1 assembly."""

from __future__ import annotations

import pandas as pd

from comp_selection_per_target import _assemble_comp_selection_result_rows
from gaia_catalog_id import normalize_gaia_id_set, normalize_gaia_source_id
from photometry_core import _dedupe_comp_pool_by_gaia_key


def test_normalize_gaia_source_id_from_dict() -> None:
    assert normalize_gaia_source_id({"source_id": 1625467932661420928}) == "1625467932661420928"
    assert normalize_gaia_source_id({"catalog_id": "1625467932661420928"}) == "1625467932661420928"


def test_normalize_gaia_id_set_drops_dict_entries() -> None:
    out = normalize_gaia_id_set(
        ["1625467932661420928", {"source_id": 999}],
        log_label="test",
    )
    assert out == frozenset({"1625467932661420928"})


def test_dedupe_comp_pool_by_gaia_key_keeps_best_rms() -> None:
    pool = pd.DataFrame(
        {
            "catalog_id": ["100", "100", "200"],
            "name": ["100", "100", "200"],
            "comp_rms": [0.04, 0.02, 0.03],
        }
    )
    out = _dedupe_comp_pool_by_gaia_key(pool)
    assert len(out) == 2
    assert float(out.loc[out["catalog_id"] == "100", "comp_rms"].iloc[0]) == 0.02


def test_assemble_comp_rows_survives_duplicate_selected_ids() -> None:
    cid = "1625568744133808640"
    final_comps = pd.DataFrame(
        {
            "name": [cid, cid],
            "catalog_id": [cid, cid],
            "comp_rms": [0.043, 0.043],
            "comp_tier": [2, 2],
            "bp_rp": [1.0, 1.0],
            "color_tier_src": ["bprp", "bprp"],
        }
    )
    out = _assemble_comp_selection_result_rows(
        [cid, cid],
        final_comps,
        id_col_cand="name",
        active={cid: 0.043},
        score_map={cid: 1.0},
        contamination_map={},
        flux_map={cid: [1.0, 1.0, 1.0]},
        target_cid="1625467932661420928",
        target=pd.Series({"vsx_name": "Gaia DR3 1625467932661420928", "catalog_id": "1625467932661420928"}),
        target_bprp_eff=1.2,
        t_bp_tgt=1.2,
        sel_note="t1t2",
        used_mag_tol=0.5,
        best_tier="TIER2",
        tier4_warning=False,
        n_t1=0,
        n_t2=1,
        n_t3=0,
        n_t4=0,
        comp_bprp_map={cid: 1.0},
        comp_tier_final_map={cid: 2},
        comp_delta_bprp_map={cid: 0.2},
        comp_color_tier_src_map={cid: "bprp"},
        _b_rejected=set(),
        final_lookup=None,
    )
    assert len(out) == 1
    assert out["catalog_id"].iloc[0] == cid
