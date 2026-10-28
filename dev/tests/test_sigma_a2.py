"""Tests for sigma A2 rig fixes and sigma_floor variant."""

from __future__ import annotations

from pathlib import Path

import pytest

from sigma_budget import resolve_rig_scintillation_params
from config import AppConfig  # noqa: E402

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

    class _FakeVyvarDatabase:
        def __new__(cls, *_a, **_k):
            return _FakeDB()

        @classmethod
        def for_config(cls, *_a, **_k):  # CONFIG-ONE-2026-10: the run database from the run config
            return _FakeDB()

    monkeypatch.setattr("database.VyvarDatabase", _FakeVyvarDatabase)
    rig = resolve_rig_scintillation_params(
        draft_id=424,
        setup="NoFilter_60_2",
        pipeline_meta={"observer_location": {"alt_m": 0.0}},
        cfg=AppConfig(),
    )
    assert rig.altitude_m == pytest.approx(250.0)
    assert any("ignored (<=0)" in n for n in rig.source_notes)
    assert any("LOCATION" in n for n in rig.source_notes)

