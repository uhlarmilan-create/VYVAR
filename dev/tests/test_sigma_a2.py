"""Tests for sigma A2 rig fixes and sigma_floor variant."""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import pytest

from sigma_budget import (
    combine_sigma_mag_quadrature,
    resolve_rig_scintillation_params,
)

ROOT = Path(__file__).resolve().parent.parent


def test_altitude_zero_pipeline_meta_falls_back_to_location(monkeypatch):
    class _FakeConn:
        def execute(self, sql, *_a, **_k):
            class _Row:
                def fetchone(self):
                    if "TELESCOPE" in sql:
                        return {
                            "diameter_mm": 200.0,
                            "telescope_name": "Carl-Zeiss",
                        }
                    if "LOCATION" in sql:
                        return {
                            "altitude_m": 250.0,
                            "place_name": "Jirny",
                        }
                    return None

            return _Row()

    class _FakeDB:
        conn = _FakeConn()

        def get_draft_telescope_id(self, draft_id: int) -> int:
            return 1

        def get_draft_location_id(self, draft_id: int) -> int:
            return 1

    monkeypatch.setattr("database.VyvarDatabase", lambda *_a, **_k: _FakeDB())
    rig = resolve_rig_scintillation_params(
        draft_id=424,
        setup="NoFilter_60_2",
        pipeline_meta={"observer_location": {"alt_m": 0.0}},
    )
    assert rig.altitude_m == pytest.approx(250.0)
    assert any("ignored (<=0)" in n for n in rig.source_notes)
    assert any("LOCATION" in n for n in rig.source_notes)


def test_combine_sigma_mag_quadrature_floor():
    total = combine_sigma_mag_quadrature(0.01, 0.005, sigma_floor_mag=0.003)
    expected = math.sqrt(0.01**2 + 0.005**2 + 0.003**2)
    assert total == pytest.approx(expected)


def test_fix_telescope_diameter_dry_run():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "fix_telescope_diameter.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "DIAMETER before=" in proc.stdout or "No matching" in proc.stdout
