"""Tests for epsf_science_set builder (EPSF-VALID-02 F1/F3)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src_py"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from epsf_science_set import build_epsf_science_set  # noqa: E402
from pipeline import _epsf_lc_catalog_ids  # noqa: E402

DRAFT516_PS = ROOT / "Archive" / "Drafts" / "draft_000516" / "platesolve" / "NoFilter_60_2"


@pytest.mark.skipif(not DRAFT516_PS.is_dir(), reason="draft 516 platesolve not on disk")
def test_science_set_census_matches_p1_decisions() -> None:
    result = build_epsf_science_set(DRAFT516_PS)
    assert result.n_total == 333
    assert result.n_targets > 0
    assert result.n_per_target_comps > 0
    lc = _epsf_lc_catalog_ids(DRAFT516_PS) or set()
    assert result.catalog_ids.issubset(lc)
    pool_only = lc - set(result.catalog_ids)
    assert len(pool_only) == 2172


@pytest.mark.skipif(not DRAFT516_PS.is_dir(), reason="draft 516 platesolve not on disk")
def test_science_set_excludes_catalog_only_targets() -> None:
    result = build_epsf_science_set(DRAFT516_PS)
    import pandas as pd

    at = pd.read_csv(
        DRAFT516_PS / "photometry" / "active_targets.csv",
        low_memory=False,
        dtype={"catalog_id": str},
    )
    catalog_only = set()
    for _, row in at.iterrows():
        z = str(row.get("zone_flag", row.get("zone", "")) or "").strip().lower()
        if z == "catalog_only":
            catalog_only.add(str(row.get("catalog_id", "")).strip())
    for cid in result.catalog_ids:
        assert str(cid) not in catalog_only
