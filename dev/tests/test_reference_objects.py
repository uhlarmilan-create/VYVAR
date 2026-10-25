"""REF-OBJECTS-2026-10: add / edit / default / hide / delete cameras, telescopes and sites."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from database import VyvarDatabase


def _db(tmp_path: Path) -> VyvarDatabase:
    (tmp_path / "Archive" / "Drafts").mkdir(parents=True)
    (tmp_path / "config.json").write_text(json.dumps({"archive_root": str(tmp_path / "Archive")}))
    db = VyvarDatabase(tmp_path / "vyvar.sqlite3")
    # One row per table first: the first object of a table becomes its default.
    for table, name_col in (("EQUIPMENTS", "CAMERANAME"), ("TELESCOPE", "TELESCOPENAME"), ("LOCATION", "PLACENAME")):
        db.save_reference_object(table, None, {name_col: "Default"})
    return db


def _draft(tmp_path: Path, name: str, **rig: int) -> None:
    d = tmp_path / "Archive" / "Drafts" / name
    d.mkdir()
    (d / "draft_manifest.json").write_text(json.dumps({"rig": rig}))


def test_add_edit_hide_delete_telescope(tmp_path: Path) -> None:
    db = _db(tmp_path)
    tid = db.save_reference_object("TELESCOPE", None, {"TELESCOPENAME": "Test scope", "FOCAL": 1000.0, "ALIAS": None})
    row = db.conn.execute("SELECT TELESCOPENAME, FOCAL, ACTIVE FROM TELESCOPE WHERE ID = ?", (tid,)).fetchone()
    assert tuple(row) == ("Test scope", 1000.0, "YES")
    db.save_reference_object("TELESCOPE", tid, {"FOCAL": 1200.0})
    assert db.conn.execute("SELECT FOCAL FROM TELESCOPE WHERE ID = ?", (tid,)).fetchone()[0] == 1200.0
    db.set_reference_active("TELESCOPE", tid, False)
    assert db.conn.execute("SELECT ACTIVE FROM TELESCOPE WHERE ID = ?", (tid,)).fetchone()[0] == "NO"
    assert db.reference_object_usage("TELESCOPE", tid) == []
    db.delete_reference_object("TELESCOPE", tid)
    assert db.conn.execute("SELECT COUNT(*) FROM TELESCOPE WHERE ID = ?", (tid,)).fetchone()[0] == 0


def test_name_is_required(tmp_path: Path) -> None:
    db = _db(tmp_path)
    with pytest.raises(ValueError, match="name is required"):
        db.save_reference_object("LOCATION", None, {"LATITUDE": 50.0})


def test_used_object_is_not_deleted_and_usage_is_listed(tmp_path: Path) -> None:
    db = _db(tmp_path)
    lid = db.save_reference_object("LOCATION", None, {"PLACENAME": "Site X", "LATITUDE": 50.0, "LONGITUDE": 14.0})
    _draft(tmp_path, "draft_000527_BO_CVn", location_id=lid)  # a renamed draft folder counts too
    used = db.reference_object_usage("LOCATION", lid)
    assert used and "draft_000527_BO_CVn" in used[0]
    with pytest.raises(ValueError, match="Still used"):
        db.delete_reference_object("LOCATION", lid)
    db.set_reference_active("LOCATION", lid, False)  # hiding is allowed


def test_calibration_masters_count_as_usage_and_curve_goes_with_the_camera(tmp_path: Path) -> None:
    db = _db(tmp_path)
    import calib_library as cl

    cl.ensure_schema(db)
    eid = db.save_reference_object("EQUIPMENTS", None, {"CAMERANAME": "Cam", "GAIN_ADU": 1.2, "READNOISE_E": 3.0})
    db.conn.execute("INSERT INTO EQUIPMENT_CURVE (EQUIPMENT_ID, GAIN_SETTING, E_PER_ADU) VALUES (?, 0, 1.2)", (eid,))
    cols = {r[1] for r in db.conn.execute("PRAGMA table_info('CALIBRATION_LIBRARY')")}
    if "ID_EQUIPMENTS" in cols:
        db.conn.execute(
            "INSERT INTO CALIBRATION_LIBRARY (KIND, FILE_PATH, XBINNING, EXPTIME, FILTER_NAME, GAIN, REGISTERED_AT, "
            "ID_EQUIPMENTS) VALUES ('dark', 'x.fits', 1, 60, 'NoFilter', 0, '2026-10-23', ?)", (eid,),
        )
        assert any("Calibration Library" in u for u in db.reference_object_usage("EQUIPMENTS", eid))
        db.conn.execute("DELETE FROM CALIBRATION_LIBRARY")
    db.delete_reference_object("EQUIPMENTS", eid)
    assert db.conn.execute("SELECT COUNT(*) FROM EQUIPMENT_CURVE WHERE EQUIPMENT_ID = ?", (eid,)).fetchone()[0] == 0


def test_default_cannot_be_hidden_or_deleted(tmp_path: Path) -> None:
    db = _db(tmp_path)
    a = db.save_reference_object("TELESCOPE", None, {"TELESCOPENAME": "A"})
    b = db.save_reference_object("TELESCOPE", None, {"TELESCOPENAME": "B"})
    db.set_reference_default("TELESCOPE", a)
    with pytest.raises(ValueError, match="default"):
        db.delete_reference_object("TELESCOPE", a)
    with pytest.raises(ValueError, match="default"):
        db.set_reference_active("TELESCOPE", a, False)
    db.set_reference_default("TELESCOPE", b)
    assert [r[0] for r in db.conn.execute("SELECT ID FROM TELESCOPE WHERE IS_DEFAULT = 1")] == [b]
    db.delete_reference_object("TELESCOPE", a)


def test_explorer_page_renders(tmp_path: Path) -> None:
    from streamlit.testing.v1 import AppTest

    db = _db(tmp_path)
    db.save_reference_object("TELESCOPE", None, {"TELESCOPENAME": "A", "FOCAL": 500.0})
    page = tmp_path / "page.py"
    page.write_text(
        "import types\nfrom database import VyvarDatabase\nfrom ui_database_explorer import render_database_explorer\n"
        f"render_database_explorer(types.SimpleNamespace(db=VyvarDatabase({str(tmp_path / 'vyvar.sqlite3')!r})))\n"
    )
    at = AppTest.from_file(str(page), default_timeout=60).run()
    assert not at.exception
    keys = {b.key for b in at.button}
    tid = int(db.conn.execute("SELECT ID FROM TELESCOPE WHERE TELESCOPENAME = 'A'").fetchone()[0])
    assert {"ref_add_TELESCOPE", f"ref_edit_TELESCOPE_{tid}", f"ref_del_TELESCOPE_{tid}"} <= keys
    at.button(key=f"ref_edit_TELESCOPE_{tid}").click().run()  # opens the edit window
    assert not at.exception
    assert f"ref_TELESCOPE_{tid}_FOCAL" in {n.key for n in at.number_input}


def test_force_delete_of_a_used_default_object(tmp_path: Path) -> None:
    db = _db(tmp_path)
    import calib_library as cl

    cl.ensure_schema(db)
    tid = db.save_reference_object("TELESCOPE", None, {"TELESCOPENAME": "Old"})
    db.set_reference_default("TELESCOPE", tid)
    _draft(tmp_path, "draft_000001", telescope_id=tid)
    db.conn.execute(
        "INSERT INTO CALIBRATION_LIBRARY (KIND, FILE_PATH, XBINNING, EXPTIME, FILTER_NAME, GAIN, REGISTERED_AT, "
        "ID_TELESCOPE) VALUES ('flat', 'f.fits', 1, 1, 'V', 0, '2026-10-23', ?)", (tid,),
    )
    db.conn.commit()
    with pytest.raises(ValueError):
        db.delete_reference_object("TELESCOPE", tid)
    res = db.delete_reference_object("TELESCOPE", tid, force=True)
    assert res["masters_unlinked"] == 1
    assert db.conn.execute("SELECT COUNT(*) FROM TELESCOPE WHERE ID = ?", (tid,)).fetchone()[0] == 0
    assert db.conn.execute("SELECT ID_TELESCOPE FROM CALIBRATION_LIBRARY").fetchone()[0] is None


def test_default_follows_lowest_active_row(tmp_path: Path) -> None:
    """No designated row IDs: the first object becomes the default, a deleted default passes on
    to the lowest-ID active row, and hidden rows are skipped."""
    db = _db(tmp_path)
    for t in ("EQUIPMENTS", "TELESCOPE", "LOCATION"):
        db.conn.execute(f"DELETE FROM {t};")
    db.conn.commit()
    a = db.save_reference_object("TELESCOPE", None, {"TELESCOPENAME": "A"})
    assert db.get_default_id("TELESCOPE") == a
    b = db.save_reference_object("TELESCOPE", None, {"TELESCOPENAME": "B"})
    c = db.save_reference_object("TELESCOPE", None, {"TELESCOPENAME": "C"})
    assert db.get_default_id("TELESCOPE") == a
    db.set_reference_active("TELESCOPE", b, False)
    db.delete_reference_object("TELESCOPE", a, force=True)
    assert db.get_default_id("TELESCOPE") == c


def test_gain_curve_points_are_replaced_and_empty_rows_skipped(tmp_path: Path) -> None:
    import pandas as pd

    db = _db(tmp_path)
    eid = db.save_reference_object("EQUIPMENTS", None, {"CAMERANAME": "Cam"})
    rows = pd.DataFrame([
        {"GAIN_SETTING": 0, "BINNING": None, "E_PER_ADU": 3.2, "READNOISE_E": 7.5, "SOURCE": "measured"},
        {"GAIN_SETTING": 100, "BINNING": 2, "E_PER_ADU": 1.1, "READNOISE_E": None, "SOURCE": "bogus"},
        {"GAIN_SETTING": 200, "BINNING": 1, "E_PER_ADU": None, "READNOISE_E": None},  # empty editor row
    ])
    assert db.save_equipment_curve(eid, rows) == 2
    got = db.conn.execute(
        "SELECT GAIN_SETTING, BINNING, E_PER_ADU, SOURCE FROM EQUIPMENT_CURVE WHERE EQUIPMENT_ID = ? ORDER BY GAIN_SETTING",
        (eid,),
    ).fetchall()
    assert [tuple(r) for r in got] == [(0, 1, 3.2, "measured"), (100, 2, 1.1, "manual")]
    assert db.save_equipment_curve(eid, rows.iloc[:1]) == 1  # replaces, does not append
    with pytest.raises(ValueError, match="does not exist"):
        db.save_equipment_curve(9999, rows)
