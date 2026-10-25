# -*- coding: ascii -*-
"""PROCESSED-FIELD-NAME-2026-10: run folders <TARGET>_<YYYYMMDD>_<NNN>, id in the manifest."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SRC_PY = Path(__file__).resolve().parents[2] / "src_py"
if str(_SRC_PY) not in sys.path:
    sys.path.insert(0, str(_SRC_PY))

from astropy.time import Time  # noqa: E402

import draft_provenance as dp  # noqa: E402
from platesolve_ui_paths import resolve_draft_directory  # noqa: E402
from processed_field import processed_field_id, processed_field_root  # noqa: E402

JIRNY_LON = 14.70


def test_target_name_from_object_then_folder() -> None:
    assert dp.processed_field_target_name("MN Boo", "whatever") == "MN_Boo"
    assert dp.processed_field_target_name("", "V842_Her") == "V842_Her"
    assert dp.processed_field_target_name(None, "lights") == "field"
    assert dp.processed_field_target_name("V* MN Boo (EW)", "") == "V_MN_Boo_EW"


def test_night_date_is_the_evening_the_night_began() -> None:
    assert dp.processed_field_night_date(Time("2026-04-22T19:50:26").jd, JIRNY_LON) == "20260422"
    # first frame after midnight (local) still belongs to the evening before
    assert dp.processed_field_night_date(Time("2026-04-22T23:40:00").jd, JIRNY_LON) == "20260422"
    assert dp.processed_field_night_date(Time("2026-04-23T01:10:00").jd, JIRNY_LON) == "20260422"


def _new(root: Path, **kw) -> int:
    data = {"observation_start_jd": Time("2026-04-22T19:50:26").jd, "object_name": "MN Boo",
            "site_lon_deg": JIRNY_LON, **kw}
    return dp.create_draft_manifest(root, data)


def test_repeated_run_gets_the_next_counter_and_a_new_id(tmp_path: Path) -> None:
    a, b = _new(tmp_path), _new(tmp_path)
    assert b == a + 1
    assert dp.draft_dir_for_id(tmp_path, a).name == "MN_Boo_20260422_001"
    assert dp.draft_dir_for_id(tmp_path, b).name == "MN_Boo_20260422_002"
    mf = json.loads((dp.draft_dir_for_id(tmp_path, b) / "draft_manifest.json").read_text())
    assert mf["draft_id"] == b and mf["name"] == "MN_Boo_20260422_002" and mf["target"] == "MN_Boo"


def test_ids_are_never_reused_after_folders_are_deleted(tmp_path: Path) -> None:
    did = _new(tmp_path, min_id_floor=538)
    assert did == 539  # above the largest id the database has seen


def test_legacy_draft_folder_still_resolves(tmp_path: Path) -> None:
    legacy = tmp_path / "Drafts" / "draft_000538"
    (legacy / "platesolve").mkdir(parents=True)
    assert dp.draft_dir_for_id(tmp_path, 538) == legacy.resolve()
    assert processed_field_id(legacy / "platesolve") == 538
    new_id = _new(tmp_path)
    assert new_id == 539


def test_load_processed_field_by_name_number_or_path(tmp_path: Path) -> None:
    did = _new(tmp_path)
    folder = dp.draft_dir_for_id(tmp_path, did)
    (folder / "platesolve" / "NoFilter_60_2").mkdir(parents=True)
    for text in ("MN_Boo_20260422_001", str(did), str(folder)):
        d, i, err = resolve_draft_directory(text, archive_root=tmp_path)
        assert err == "" and d == folder.resolve() and i == did
    assert processed_field_root(folder / "platesolve" / "NoFilter_60_2") == folder.resolve()
    _d, _i, err = resolve_draft_directory("Nope_20260101_001", archive_root=tmp_path)
    assert "not found" in err


def test_site_provenance_keeps_the_name_in_the_manifest(tmp_path: Path) -> None:
    """record_observer_location_provenance used to rewrite the manifest and drop name / target."""
    did = _new(tmp_path)
    folder = dp.draft_dir_for_id(tmp_path, did)
    dp.record_observer_location_provenance(
        archive_path=folder, draft_id=did,
        resolved={"location_id": 2, "name": "Jirny", "lat": 50.11, "lon": 14.70, "alt_m": 275.0},
    )
    mf = json.loads((folder / "draft_manifest.json").read_text())
    assert mf["name"] == "MN_Boo_20260422_001" and mf["target"] == "MN_Boo"
    assert mf["observer_location"]["location_id"] == 2 and mf["rig"]["location_id"] == 2
