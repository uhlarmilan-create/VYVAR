"""MASTERSTAR-GAIA-01 fire-proof invariants (INV-DET-FALSEFILL-01, INV-SEED-FALSEFILL-01, INV-MS-*)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
CTX = ROOT / "dev" / "results" / "context" / "session_20260819_msgaia01"
def test_inv_ms_census_01_fail_loud(tmp_path: Path) -> None:
    from invariants_runtime import InvariantViolation
    from masterstar_gaia_accounting import (
        SOURCE_DETECTED_P1,
        verify_gaia_census_complete,
        verify_gaia_census,
    )

    census = pd.DataFrame(
        {"catalog_id": ["1", "2"], "source_state": [SOURCE_DETECTED_P1, SOURCE_DETECTED_P1]}
    )
    rec = verify_gaia_census(census, n_on_chip=2)
    assert rec["ok"] is True and rec["n"] == 2
    ok, _ = verify_gaia_census_complete(census, 2)
    assert ok
    assert verify_gaia_census(pd.DataFrame(), n_on_chip=0)["n"] == 0
    with pytest.raises(InvariantViolation):
        verify_gaia_census(census, n_on_chip=99)
    assert not list(tmp_path.iterdir())  # OUTPUT-REVIEW-2026-10: no census file


def test_seed_rows_excluded_from_comp_pool_when_gate_off() -> None:
    from pipeline import select_comparison_stars_spatial_grid

    rows = []
    for i in range(12):
        rows.append(
            {
                "catalog_id": str(1000 + i),
                "catalog": "Gaia",
                "x": 100.0 + 40.0 * (i % 4),
                "y": 100.0 + 40.0 * (i // 4),
                "ra_deg": 200.0,
                "dec_deg": 40.0,
                "mag": 12.0 + 0.1 * i,
                "is_usable": True,
                "photometry_ok": True,
                "source_state": "DETECTED_P1",
                "forced_photometry": False,
            }
        )
    rows.append(
        {
            "catalog_id": "999999",
            "catalog": "Gaia",
            "x": 120.0,
            "y": 120.0,
            "ra_deg": 200.0,
            "dec_deg": 40.0,
            "mag": 11.0,
            "is_usable": True,
            "photometry_ok": True,
            "source_state": "FORCED_SEED",
            "forced_photometry": True,
        }
    )
    df = pd.DataFrame(rows)
    comp, _meta = select_comparison_stars_spatial_grid(
        df, width_px=400.0, height_px=400.0, n_comp=0, require_non_variable=False
    )
    ids = set(comp["catalog_id"].astype(str)) if len(comp) else set()
    assert "999999" not in ids


def test_part_a_report_exists() -> None:
    p = CTX / "part_a_false_accept.json"
    if not p.is_file():
        pytest.skip("Part A audit not run")
    rep = json.loads(p.read_text(encoding="ascii"))
    e1 = rep.get("E1") or rep.get("E1_tightened")
    assert e1 in ("PASS", "DEVIATE")
    assert int(rep.get("n_main", 0)) >= 2000
