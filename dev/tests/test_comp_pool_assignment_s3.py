"""COMP-POOL-01 Stage 3: assignment relax provenance."""
from __future__ import annotations

from comp_selection_per_target import (
    COMP_ASSIGNMENT_RELAX_ORDER,
    format_assignment_relax_provenance,
)


def test_relax_order_tuple_stable():
    assert COMP_ASSIGNMENT_RELAX_ORDER[0].startswith("colour_tier")
    assert "sparse_fallback_path" in COMP_ASSIGNMENT_RELAX_ORDER


def test_format_relax_none_fired():
    s = format_assignment_relax_provenance(
        used_mag_tol=1.0,
        mag_tol_start=1.0,
        best_tier="TIER1",
        comp_path="default",
        n_t1=5,
    )
    assert "fired=none" in s
    assert "relax_order=" in s


def test_format_relax_records_sparse_and_mag():
    s = format_assignment_relax_provenance(
        used_mag_tol=2.5,
        mag_tol_start=1.0,
        best_tier="TIER3",
        comp_path="sparse_fallback",
        n_t1=0,
        n_t2=0,
        n_t3=3,
    )
    assert "colour_tier->TIER3" in s
    assert "delta_mag->2.50" in s
    assert "sparse_fallback" in s
