"""Tests for manifest-first UI helper functions (Phase 2.7)."""

from __future__ import annotations

import json
from pathlib import Path

from draft_provenance import (
    collect_manifest_draft_rows,
    collect_manifest_obs_file_rows,
    collect_manifest_observation_rows,
    draft_scan_summary_from_manifest,
    obs_draft_row_from_manifest,
    write_draft_manifest,
)


def test_obs_draft_row_from_manifest_maps_core_fields(tmp_path: Path) -> None:
    root = tmp_path / "draft_000001"
    root.mkdir()
    write_draft_manifest(
        root,
        draft_id=1,
        calibration_mode="vyvar_calibrated",
        rig={
            "equipment_id": 2,
            "telescope_id": 3,
            "location_id": 4,
            "scanning_id": 5,
        },
        paths={"archive": str(root), "lights": str(root / "lights")},
        status="INGESTED",
        center={"ra_deg": 12.5, "de_deg": 45.0},
        observation_start_jd=2450000.5,
        is_calibrated=1,
        files=[],
    )
    manifest = json.loads((root / "draft_manifest.json").read_text(encoding="utf-8"))
    row = obs_draft_row_from_manifest(manifest, 1)
    assert row["ID"] == 1
    assert row["ID_EQUIPMENTS"] == 2
    assert row["CENTEROFFIELDRA"] == 12.5
    assert row["STATUS"] == "INGESTED"


def test_collect_manifest_draft_rows_scans_archive(tmp_path: Path) -> None:
    drafts = tmp_path / "Drafts"
    d1 = drafts / "draft_000010"
    d1.mkdir(parents=True)
    write_draft_manifest(
        d1,
        draft_id=10,
        calibration_mode="vyvar_calibrated",
        rig={"equipment_id": 1, "telescope_id": 1, "location_id": 1, "scanning_id": 1},
        paths={"archive": str(d1)},
        status="PROCESSED",
        files=[],
    )
    rows = collect_manifest_draft_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["ID"] == 10
    assert rows[0]["STATUS"] == "PROCESSED"


def test_collect_manifest_observation_rows_finalized(tmp_path: Path) -> None:
    drafts = tmp_path / "Drafts"
    d1 = drafts / "draft_000020"
    d1.mkdir(parents=True)
    write_draft_manifest(
        d1,
        draft_id=20,
        calibration_mode="vyvar_calibrated",
        rig={"equipment_id": 1, "telescope_id": 1, "location_id": 1, "scanning_id": 1},
        paths={"archive": str(d1), "lights": str(d1 / "lights")},
        status="FINALIZED",
        final_observation_id="OBS-ABC",
        is_calibrated=0,
        files=[],
    )
    obs_rows = collect_manifest_observation_rows(tmp_path)
    assert len(obs_rows) == 1
    assert obs_rows[0]["ID"] == "OBS-ABC"
    assert obs_rows[0]["DRAFT_ID"] == 20


def test_collect_manifest_obs_file_rows(tmp_path: Path) -> None:
    drafts = tmp_path / "Drafts"
    d1 = drafts / "draft_000030"
    d1.mkdir(parents=True)
    write_draft_manifest(
        d1,
        draft_id=30,
        calibration_mode="vyvar_calibrated",
        files=[
            {
                "file_path": str(d1 / "a.fits"),
                "imagetyp": "light",
                "filter": "V",
                "obs_file_id": 99,
            }
        ],
    )
    rows = collect_manifest_obs_file_rows(tmp_path, draft_id=30)
    assert len(rows) == 1
    assert rows[0]["DRAFT_ID"] == 30
    assert rows[0]["FILTER"] == "V"


def test_draft_scan_summary_from_manifest_files() -> None:
    manifest = {
        "files": [
            {
                "imagetyp": "bias",
                "filter": "V",
            },
            {
                "imagetyp": "light",
                "filter": "R",
                "inspection": {"exptime": 60.0},
            },
        ]
    }
    scan = draft_scan_summary_from_manifest(manifest)
    assert scan is not None
    assert scan["filters"] == "R"
    assert scan["exptime"] == 60.0
