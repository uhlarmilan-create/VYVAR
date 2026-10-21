"""COMP-POOL-01 Stage 3: assignment relax provenance."""
from __future__ import annotations

from comp_selection_per_target import (
    COMP_ASSIGNMENT_RELAX_ORDER,
    format_assignment_relax_provenance,
)


def test_relax_order_tuple_stable():
    assert COMP_ASSIGNMENT_RELAX_ORDER == ("colour_tier_widen_T1_to_T4",)


def test_format_relax_none_fired():
    s = format_assignment_relax_provenance(
        best_tier="TIER1",
        n_t1=5,
    )
    assert "fired=none" in s
    assert "relax_order=" in s


def test_format_relax_records_colour_tier():
    s = format_assignment_relax_provenance(
        best_tier="TIER3",
        n_t1=0,
        n_t2=0,
        n_t3=3,
    )
    assert "colour_tier->TIER3" in s
    assert "sparse_fallback" not in s
