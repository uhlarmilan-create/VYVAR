# -*- coding: ascii -*-
"""VYVAR step 6, T19 (s6_db): database migrations, measured gain points, id allocation, job
protocol and catalogue-builder robustness (Milan 2026-10-31)."""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from pathlib import Path

import pandas as pd
import pytest

import camera_profile as cp
import draft_provenance as dp
from database import VyvarDatabase


def _mk(path: Path, sql: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.executescript(sql)
    con.commit()
    con.close()
    return path


def _curve_row(**kw) -> dict:
    row = {c: None for c in VyvarDatabase.CURVE_POINT_COLUMNS}
    row.update(kw)
    return row


# --- G13-09 / G13-10 / G13-47: migrations on old DB variants --------------------------------------

def test_g1309_focal_migration_keeps_columns_and_curve_fk(tmp_path) -> None:
    p = _mk(tmp_path / "vyvar.sqlite3", """
        CREATE TABLE EQUIPMENTS(ID INTEGER PRIMARY KEY AUTOINCREMENT, CAMERANAME TEXT, ALIAS TEXT,
            SENSORTYPE TEXT, SENSORSIZE TEXT, PIXELSIZE REAL, FOCAL REAL, ACTIVE TEXT DEFAULT 'YES',
            MODEL_KEY TEXT, SIGMA_SYS_MAG REAL);
        INSERT INTO EQUIPMENTS(CAMERANAME, PIXELSIZE, FOCAL, ACTIVE, MODEL_KEY, SIGMA_SYS_MAG)
            VALUES('QHY294M', 4.63, 0, 'YES', NULL, 0.004);
    """)
    for _ in range(2):  # open twice: the second open must not lose anything either
        db = VyvarDatabase(p)
        db.close()
    db = VyvarDatabase(p)
    cols = {r["name"] for r in db.conn.execute("PRAGMA table_info('EQUIPMENTS');").fetchall()}
    assert "FOCAL" not in cols and {"SIGMA_SYS_MAG", "MODEL_KEY", "DEFAULT_GAIN_SETTING"} <= cols
    assert db.conn.execute("SELECT SIGMA_SYS_MAG FROM EQUIPMENTS WHERE ID = 1;").fetchone()[0] == pytest.approx(0.004)
    fk = {r[2] for r in db.conn.execute("PRAGMA foreign_key_list('EQUIPMENT_CURVE');").fetchall()}
    assert fk == {"EQUIPMENTS"}
    n = db.save_equipment_curve(1, pd.DataFrame([_curve_row(GAIN_SETTING=0, BINNING=1, E_PER_ADU=1.0, READNOISE_E=2.0)]))
    assert n == 1
    db.close()


def test_g1309_heals_curve_fk_pointing_at_equipments_old(tmp_path) -> None:
    p = _mk(tmp_path / "vyvar.sqlite3", """
        CREATE TABLE EQUIPMENTS(ID INTEGER PRIMARY KEY AUTOINCREMENT, CAMERANAME TEXT, ACTIVE TEXT DEFAULT 'YES');
        INSERT INTO EQUIPMENTS(CAMERANAME) VALUES('cam');
        CREATE TABLE EQUIPMENT_CURVE (ID INTEGER PRIMARY KEY AUTOINCREMENT,
            EQUIPMENT_ID INTEGER NOT NULL REFERENCES "EQUIPMENTS_OLD"(ID), READOUT_MODE TEXT, GAIN_SETTING REAL,
            BINNING INTEGER NOT NULL DEFAULT 1, E_PER_ADU REAL, READNOISE_E REAL,
            SOURCE TEXT NOT NULL DEFAULT 'manual', MEASURED_AT TEXT, NOTE TEXT);
        INSERT INTO EQUIPMENT_CURVE(EQUIPMENT_ID, GAIN_SETTING, E_PER_ADU) VALUES(1, 100, 0.8);
    """)
    db = VyvarDatabase(p)
    assert {r[2] for r in db.conn.execute("PRAGMA foreign_key_list('EQUIPMENT_CURVE');").fetchall()} == {"EQUIPMENTS"}
    assert db.conn.execute("SELECT E_PER_ADU FROM EQUIPMENT_CURVE;").fetchone()[0] == pytest.approx(0.8)
    db.save_measured_curve_point(1, readout_mode=None, gain_setting=100, binning=1, e_per_adu=0.9, readnoise_e=None)
    db.close()


def test_g1310_integer_active_legacy_db_opens_twice(tmp_path) -> None:
    p = _mk(tmp_path / "vyvar.sqlite3", """
        CREATE TABLE EQUIPMENTS(ID INTEGER PRIMARY KEY AUTOINCREMENT, CAMERANAME TEXT, ALIAS TEXT,
            SENSORTYPE TEXT, SENSORSIZE TEXT, PIXELSIZE REAL, ACTIVE INTEGER DEFAULT 1);
        INSERT INTO EQUIPMENTS(CAMERANAME, ACTIVE) VALUES('cam', 1);
        CREATE TABLE TELESCOPE(ID INTEGER PRIMARY KEY AUTOINCREMENT, TELESCOPENAME TEXT, ALIAS TEXT,
            DIAMETER REAL, FOCAL REAL, ACTIVE INTEGER DEFAULT 1);
        INSERT INTO TELESCOPE(TELESCOPENAME, FOCAL, ACTIVE) VALUES('t', 1000, 1);
    """)
    VyvarDatabase(p).close()
    db = VyvarDatabase(p)
    assert db.conn.execute("SELECT ACTIVE FROM EQUIPMENTS WHERE ID = 1;").fetchone()[0] == "YES"
    assert db.conn.execute("SELECT ACTIVE FROM TELESCOPE WHERE ID = 1;").fetchone()[0] == "YES"
    rid = db.save_measured_curve_point(1, readout_mode="High Gain", gain_setting=0, binning=1,
                                       e_per_adu=1.0, readnoise_e=2.0, note="ptc")
    assert rid > 0
    db.close()


# --- G13-16: a half-consumed cursor never blocks another thread ------------------------------------

def test_g1316_half_consumed_cursor_does_not_hold_the_lock(tmp_path) -> None:
    db = VyvarDatabase(tmp_path / "vyvar.sqlite3")
    cur = db.conn.execute("SELECT name FROM sqlite_master;")
    assert cur.fetchone() is not None
    done = threading.Event()
    th = threading.Thread(target=lambda: (db.conn.execute("SELECT 1;").fetchone(), done.set()), daemon=True)
    th.start()
    assert done.wait(1.0)
    assert cur.fetchall()  # the rest of the rows is still there
    ins = db.conn.execute("INSERT INTO LOCATION (PLACENAME) VALUES ('x');")
    assert ins.lastrowid >= 1 and ins.rowcount == 1
    db.close()


# --- G13-14 / NEW-5: QC apply is one transaction; the per-file snapshot table is gone ---------------

def test_g1314_qc_apply_rolls_back_and_new5_file_table_dropped(tmp_path, monkeypatch) -> None:
    db = VyvarDatabase(tmp_path / "vyvar.sqlite3")
    tables = {r[0] for r in db.conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()}
    assert "OBS_QC_PROCESSING_FILE" not in tables
    db.record_qc_processing_apply(5, "h1")

    def _boom(*_a, **_k):
        raise RuntimeError("fault")

    monkeypatch.setattr(db, "_bump_id_counter", _boom)
    with pytest.raises(RuntimeError):
        db.record_qc_processing_apply(6, "h2")
    assert not db.conn.in_transaction
    db.conn.commit()  # any later commit elsewhere
    rows = sqlite3.connect(tmp_path / "vyvar.sqlite3").execute(
        "SELECT PROCESSING_HASH, DRAFT_ID FROM OBS_QC_PROCESSING_RUN ORDER BY ID;").fetchall()
    assert rows == [("h1", 5)]
    db.close()


# --- G14-15 / NEW-1: one measured point per key; the newest wins ------------------------------------

def test_g1415_save_twice_one_row_and_newest_wins(tmp_path) -> None:
    db = VyvarDatabase(tmp_path / "vyvar.sqlite3")
    eid = db.insert_equipment("c", "c", "MONO", "1x1", 3.76)
    for g, day in ((1.10, "2026-10-01"), (0.95, "2026-10-30")):
        db.save_measured_curve_point(eid, readout_mode=None, gain_setting=100, binning=1, e_per_adu=g,
                                     readnoise_e=2.0, measured_at=day, note="ptc")
    assert db.conn.execute("SELECT COUNT(*) FROM EQUIPMENT_CURVE;").fetchone()[0] == 1
    prof = cp.profile_from_connection(db.conn, eid)
    assert cp.lookup(prof, "gain", 100, 1).value == pytest.approx(0.95)
    with pytest.raises(sqlite3.IntegrityError):  # a raw duplicate cannot be written any more
        db.conn.execute(
            "INSERT INTO EQUIPMENT_CURVE (EQUIPMENT_ID, GAIN_SETTING, BINNING, E_PER_ADU, SOURCE) "
            "VALUES (?, 100, 1, 1.2, 'measured');", (eid,))
    db.conn.rollback()
    db.close()


def test_new1_lookup_takes_newest_of_equal_points(tmp_path) -> None:
    db = VyvarDatabase(tmp_path / "vyvar.sqlite3")
    eid = db.insert_equipment("c", "c", "MONO", "1x1", 3.76)
    db.save_equipment_curve(eid, pd.DataFrame([
        _curve_row(GAIN_SETTING=100, BINNING=1, E_PER_ADU=1.10, SOURCE="manual", MEASURED_AT="2026-01-01"),
        _curve_row(GAIN_SETTING=100, BINNING=1, E_PER_ADU=0.95, SOURCE="manual", MEASURED_AT="2026-10-01"),
    ]))
    prof = cp.profile_from_connection(db.conn, eid)
    assert cp.lookup(prof, "gain", 100, 1).value == pytest.approx(0.95)
    db.close()


def test_new1_migration_removes_older_duplicates_and_logs(tmp_path, caplog) -> None:
    p = tmp_path / "vyvar.sqlite3"
    VyvarDatabase(p).close()
    con = sqlite3.connect(p)
    con.execute("DROP INDEX UX_EQUIPMENT_CURVE_MEASURED;")  # a DB written before step 6
    con.execute("INSERT INTO EQUIPMENTS (CAMERANAME, ACTIVE) VALUES ('c', 'YES');")
    for g, day in ((1.10, "2026-10-01"), (0.95, "2026-10-30"), (1.00, "2026-10-15")):
        con.execute("INSERT INTO EQUIPMENT_CURVE (EQUIPMENT_ID, GAIN_SETTING, BINNING, E_PER_ADU, SOURCE, MEASURED_AT) "
                    "VALUES (1, 100, 1, ?, 'measured', ?);", (g, day))
    con.execute("INSERT INTO EQUIPMENT_CURVE (EQUIPMENT_ID, GAIN_SETTING, BINNING, E_PER_ADU, SOURCE, MEASURED_AT) "
                "VALUES (1, 200, 1, 0.5, 'measured', '2026-10-01');")
    con.commit()
    con.close()
    with caplog.at_level(logging.WARNING):
        db = VyvarDatabase(p)
    rows = db.conn.execute("SELECT GAIN_SETTING, E_PER_ADU FROM EQUIPMENT_CURVE ORDER BY GAIN_SETTING;").fetchall()
    assert [tuple(r) for r in rows] == [(100.0, 0.95), (200.0, 0.5)]
    assert sum("NEW-1 cleanup" in r.getMessage() for r in caplog.records) == 2
    db.close()


def test_new1_pristine_user_db_has_no_duplicates() -> None:
    """Read-only count on the pristine copy (Milan's question: does the cleanup change any gain?)."""
    src = Path("/root/p0env_pristine/vyvar.sqlite3")
    if not src.is_file():
        pytest.skip("pristine env not on this machine")
    con = sqlite3.connect(src.resolve().as_uri() + "?mode=ro", uri=True)
    n = con.execute(
        "SELECT COUNT(*) FROM (SELECT 1 FROM EQUIPMENT_CURVE WHERE SOURCE = 'measured' GROUP BY EQUIPMENT_ID, "
        "IFNULL(READOUT_MODE, ''), IFNULL(GAIN_SETTING, 'none'), BINNING HAVING COUNT(*) > 1);").fetchone()[0]
    con.close()
    assert n == 0


# --- G13-43 / NEW-4 / G12-22 / G12-21 / NEW-3 / G12-45: draft ids and folders -------------------------

def _rig_db(tmp_path) -> tuple[VyvarDatabase, dict]:
    db = VyvarDatabase(tmp_path / "vyvar.sqlite3")
    db._archive_root_override = tmp_path / "Archive"
    eq = db.insert_equipment("Cam", "C", "mono", "1000", 3.76)
    tel = db.insert_telescope("Tel", "T", 200.0, 1000.0)
    loc = db.insert_location("Site", 50.0, 14.0, 300.0)
    data = {"id_equipments": eq, "id_telescope": tel, "id_location": loc, "id_scanning": 1,
            "observation_start_jd": 2460970.4, "object_name": "MN Boo", "site_lon_deg": 14.7}
    return db, data


def test_g1343_create_draft_requires_rig_ids(tmp_path) -> None:
    db, data = _rig_db(tmp_path)
    with pytest.raises(KeyError):
        db.create_draft({k: v for k, v in data.items() if k != "id_equipments"})
    db.close()


def test_new4_id_floor_query_fails_loud(tmp_path) -> None:
    db, data = _rig_db(tmp_path)
    db.conn.execute("DROP TABLE OBS_QC_PROCESSING_RUN;")
    db.conn.commit()
    with pytest.raises(sqlite3.OperationalError):
        db.create_draft(data)
    db.close()


def test_g1222_ids_never_reused_after_folder_delete_or_bad_manifest(tmp_path) -> None:
    import shutil

    db, data = _rig_db(tmp_path)
    a = db.create_draft(data)
    b = db.create_draft(data)
    shutil.rmtree(dp.draft_dir_for_id(tmp_path / "Archive", b))  # newest folder deleted
    c = db.create_draft(data)
    assert c > b > a
    (dp.draft_dir_for_id(tmp_path / "Archive", c) / "draft_manifest.json").write_text("{broken", encoding="ascii")
    d = db.create_draft(data)  # the unreadable manifest no longer hides its id
    assert d > c
    db.close()


def test_new3_concurrent_create_draft_unique_ids_and_folders(tmp_path) -> None:
    db0, data = _rig_db(tmp_path)
    db0.close()
    ids: list[int] = []
    errors: list[str] = []
    start = threading.Barrier(6)

    def _one() -> None:
        db = VyvarDatabase(tmp_path / "vyvar.sqlite3")  # its own connection, like GUI + CLI
        db._archive_root_override = tmp_path / "Archive"
        try:
            start.wait()
            ids.append(db.create_draft(data))
        except Exception as exc:  # noqa: BLE001
            errors.append(repr(exc))
        finally:
            db.close()

    ths = [threading.Thread(target=_one) for _ in range(6)]
    for t in ths:
        t.start()
    for t in ths:
        t.join(60)
    assert not errors, errors
    assert len(set(ids)) == 6
    folders = [p for p in (tmp_path / "Archive" / "Drafts").iterdir() if p.is_dir()]
    assert len(folders) == 6
    got = {json.loads((p / "draft_manifest.json").read_text())["draft_id"] for p in folders}
    assert got == set(ids)


def test_g1245_folder_counter_claims_by_mkdir(tmp_path) -> None:
    out: list[Path] = []
    start = threading.Barrier(8)

    def _one() -> None:
        start.wait()
        out.append(dp.make_processed_field_folder(tmp_path, "MN_Boo", "20260422"))

    ths = [threading.Thread(target=_one) for _ in range(8)]
    for t in ths:
        t.start()
    for t in ths:
        t.join(10)
    assert len({p.name for p in out}) == 8


def test_g1221_manifest_writer_atomic_and_cleans_tmp(tmp_path, monkeypatch) -> None:
    dp.write_draft_manifest(tmp_path, draft_id=7, calibration_mode="VYVAR")
    before = (tmp_path / "draft_manifest.json").read_text()

    def _fail(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(dp.os, "replace", _fail)
    with pytest.raises(OSError):
        dp.write_draft_manifest(tmp_path, draft_id=8, calibration_mode="VYVAR")
    assert (tmp_path / "draft_manifest.json").read_text() == before
    assert [p.name for p in tmp_path.iterdir()] == ["draft_manifest.json"]


# --- G13-46: header cache compares st_mtime_ns exactly ---------------------------------------------

def test_g1346_header_cache_exact_mtime_ns(tmp_path) -> None:
    db = VyvarDatabase(tmp_path / "vyvar.sqlite3")
    fp = tmp_path / "a.fits"
    fp.write_bytes(b"x" * 2880)
    st = fp.stat()
    meta = {"exposure": 60.0, "filter": "V", "binning": 1, "naxis1": 10, "naxis2": 10, "gain": 1.0,
            "ra": 0.0, "dec": 0.0, "jd_start": 2460000.5}
    db.fits_header_cache_upsert_one(fp, file_size=st.st_size, mtime_ns=st.st_mtime_ns, meta=meta,
                                    imagetyp="LIGHT", date_obs=None)
    assert db.fits_header_cache_try_meta(fp) is not None
    os.utime(fp, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))  # rewritten 1 s later, same size
    assert db.fits_header_cache_try_meta(fp) is None
    db.close()


def test_g1346_legacy_float_mtime_cache_is_rebuilt(tmp_path) -> None:
    p = tmp_path / "vyvar.sqlite3"
    VyvarDatabase(p).close()
    con = sqlite3.connect(p)
    con.execute("DROP TABLE FITS_HEADER_CACHE;")
    con.execute("CREATE TABLE FITS_HEADER_CACHE (FILE_PATH TEXT PRIMARY KEY, FILE_SIZE INTEGER NOT NULL, "
                "MTIME REAL NOT NULL, CCD_TEMP REAL, GAIN REAL NOT NULL, READOUT_MODE TEXT);")
    con.commit()
    con.close()
    db = VyvarDatabase(p)
    cols = {r[1] for r in db.conn.execute("PRAGMA table_info('FITS_HEADER_CACHE');").fetchall()}
    assert "MTIME_NS" in cols and "MTIME" not in cols
    db.close()


# --- G14-23 / G14-34 / G13-45 -----------------------------------------------------------------------

def test_g1423_autodetect_candidates_active_only(tmp_path) -> None:
    db = VyvarDatabase(tmp_path / "vyvar.sqlite3")
    a = db.insert_equipment("A", "a", "MONO", "1x1", 3.76)
    b = db.insert_equipment("B", "b", "MONO", "1x1", 3.76)
    db.conn.execute("UPDATE EQUIPMENTS SET ACTIVE = 'NO' WHERE ID = ?;", (b,))
    db.conn.commit()
    t = db.insert_telescope("T", "t", 200.0, 1000.0)
    eq, tel = db.autodetect_candidates()
    assert [r["ID"] for r in eq] == [a] and "MODEL_KEY" in eq[0]
    assert [r["ID"] for r in tel] == [t] and tel[0]["FOCAL"] == 1000.0
    db.close()


def test_g1434_delete_row_first_file_kept_on_db_error(tmp_path) -> None:
    from ui_calibration_library import _delete_master

    f = tmp_path / "MasterDark.fits"
    f.write_bytes(b"x")

    class _Db:
        def delete_calibration_library_entry_by_path(self, _p):
            raise sqlite3.OperationalError("database is locked")

    with pytest.raises(sqlite3.Error):
        _delete_master(_Db(), f)
    assert f.is_file()


def test_g1345_calibration_library_schema_owned_by_calib_library(tmp_path) -> None:
    import inspect

    import calib_library as cl

    db = VyvarDatabase(tmp_path / "vyvar.sqlite3")
    cols = {r[1]: str(r[2]).upper() for r in db.conn.execute("PRAGMA table_info('CALIBRATION_LIBRARY');").fetchall()}
    assert cols["GAIN"] == "REAL"
    assert {"ID_EQUIPMENTS", "ID_TELESCOPE", "OFFSET_HANDLING", "GAIN_EXACT"} <= set(cols)
    cl.create_table(db.conn)  # idempotent
    assert "CREATE TABLE IF NOT EXISTS CALIBRATION_LIBRARY" not in inspect.getsource(VyvarDatabase)
    db.close()


def test_g1222_hash_failure_is_logged_not_silent() -> None:
    import inspect

    import night_run

    src = inspect.getsource(night_run)
    i = src.index("ph = generate_observation_hash(pipeline.db, draft_id)")
    block = src[i:i + 700]
    assert "LOGGER.warning" in block and "log_event" in block


def test_eager_cursor_iteration_inside_write_loop(tmp_path) -> None:
    """Migrations iterate a SELECT and UPDATE in the loop (same connection, eager rows)."""
    db = VyvarDatabase(tmp_path / "vyvar.sqlite3")
    for i in range(3):
        db.conn.execute("INSERT INTO LOCATION (PLACENAME, ACTIVE) VALUES (?, '1');", (f"s{i}",))
    db.conn.commit()
    db._normalize_active_values_in_place("LOCATION")
    assert {r[0] for r in db.conn.execute("SELECT ACTIVE FROM LOCATION;").fetchall()} == {"YES"}
    assert not db.conn.in_transaction
    db.close()


# --- G12-39 / G12-40 / G12-41 / G05-17: job protocol -------------------------------------------------

class _FakeProc:
    def __init__(self, pid: int) -> None:
        self.pid = pid


def test_g1241_two_job_dirs_in_one_second(tmp_path, monkeypatch) -> None:
    import run_job

    class _Now:
        @staticmethod
        def now(tz=None):
            from datetime import datetime as _dt
            return _dt(2026, 10, 31, 12, 0, 0, tzinfo=tz)

    monkeypatch.setattr(run_job, "datetime", _Now)
    a, b = run_job.new_job_dir(tmp_path), run_job.new_job_dir(tmp_path)
    assert a != b and a.name == "run_20261031_120000" and b.name == "run_20261031_120000_2"


def test_g1240_stdout_handle_closed_in_parent(tmp_path, monkeypatch) -> None:
    import run_job

    seen = {}

    def _popen(cmd, stdout=None, **kw):
        seen["out"] = stdout
        return _FakeProc(os.getpid())

    monkeypatch.setattr(run_job.subprocess, "Popen", _popen)
    monkeypatch.setattr(run_job, "params_to_job", lambda p: {})
    jd = run_job.launch_job(object(), tmp_path)
    assert seen["out"].closed
    assert run_job.read_status(jd)["pid"] == os.getpid()


class _PsProc:
    def __init__(self, pid):
        pass

    def children(self, recursive=True):
        return []

    def terminate(self):
        pass


def test_g1239_stop_true_after_terminate_even_if_status_write_fails(tmp_path, monkeypatch) -> None:
    import psutil

    import run_job

    run_job.write_status(tmp_path, state="running", pid=os.getpid())
    monkeypatch.setattr(psutil, "Process", _PsProc)
    monkeypatch.setattr(run_job, "pid_alive", lambda pid: True)

    def _locked(*_a, **_k):
        raise PermissionError("status file held open by the GUI")

    monkeypatch.setattr(run_job, "_write_status_locked", _locked)
    assert run_job.stop_job(tmp_path) is True


def test_g1239_stopped_state_is_final_and_no_tmp_left(tmp_path, monkeypatch) -> None:
    import psutil

    import run_job

    run_job.write_status(tmp_path, state="running", pid=os.getpid())
    monkeypatch.setattr(psutil, "Process", _PsProc)
    monkeypatch.setattr(run_job, "pid_alive", lambda pid: True)
    assert run_job.stop_job(tmp_path) is True
    run_job.write_status(tmp_path, state="running", message="frame 12/40", extra=1)  # a late child write
    st = run_job.read_status(tmp_path)
    assert st["state"] == "failed" and st["message"] == "Stopped by user" and st["extra"] == 1
    assert not list(tmp_path.glob("*.tmp")) and not (tmp_path / "run_status.lock").exists()


def test_g1239_concurrent_status_writes_keep_all_fields(tmp_path) -> None:
    import run_job

    run_job.write_status(tmp_path, state="starting")
    start = threading.Barrier(8)

    def _w(i: int) -> None:
        start.wait()
        for k in range(20):
            run_job.write_status(tmp_path, **{f"k{i}": k})

    ths = [threading.Thread(target=_w, args=(i,)) for i in range(8)]
    for t in ths:
        t.start()
    for t in ths:
        t.join(30)
    st = run_job.read_status(tmp_path)
    assert all(st.get(f"k{i}") == 19 for i in range(8))


def test_g0517_two_launches_one_job(tmp_path, monkeypatch) -> None:
    import catalog_jobs as cj

    monkeypatch.setattr(cj.subprocess, "Popen", lambda cmd, **kw: _FakeProc(os.getpid()))
    ok: list[Path] = []
    refused: list[str] = []
    start = threading.Barrier(2)

    def _go() -> None:
        start.wait()
        try:
            ok.append(cj.launch("objects_update", ["--db", "x"], tmp_path))
        except RuntimeError as exc:
            refused.append(str(exc))

    ths = [threading.Thread(target=_go) for _ in range(2)]
    for t in ths:
        t.start()
    for t in ths:
        t.join(30)
    assert len(ok) == 1 and len(refused) == 1
    assert [d.name for d in (tmp_path / "jobs").iterdir() if d.is_dir()] == [ok[0].name]


def test_g0517_starting_without_pid_counts_as_running_then_lost(tmp_path) -> None:
    from datetime import datetime, timedelta, timezone

    import run_job

    now = datetime.now(timezone.utc)
    run_job.write_status(tmp_path, state="starting", started_utc=now.isoformat())
    assert run_job.job_state(tmp_path) == "running"
    old = now - timedelta(seconds=run_job.SPAWN_TIMEOUT_S + 5)
    run_job.write_status(tmp_path, started_utc=old.isoformat())
    assert run_job.job_state(tmp_path) == "lost"


def test_g0517_stale_lock_replaced_and_child_releases(tmp_path, monkeypatch) -> None:
    import catalog_jobs as cj

    old = tmp_path / "jobs" / "catalog_objects_update_20261001_120000"
    old.mkdir(parents=True)
    (old / cj.STATUS_FILE).write_text(json.dumps({"state": "done", "catalog": "objects"}))
    (tmp_path / "jobs" / ".lock_objects").write_text(old.name)
    monkeypatch.setattr(cj.subprocess, "Popen", lambda cmd, **kw: _FakeProc(os.getpid()))
    jd = cj.launch("objects_update", ["--db", "x"], tmp_path)
    assert (tmp_path / "jobs" / ".lock_objects").read_text() == jd.name
    monkeypatch.setattr(cj, "_run_child", lambda job_dir, job: 0)
    assert cj.run_child(jd) == 0
    assert not (tmp_path / "jobs" / ".lock_objects").exists()


# --- G05-03 / NEW-2: a download is complete only when rows == COUNT(*) --------------------------------

import re as _re  # noqa: E402

import numpy as _np  # noqa: E402

_BOX_RE = _re.compile(r"ra >= (-?[\d.]+) AND ra (<=|<) (-?[\d.]+) AND dec >= (-?[\d.]+) AND dec (<=|<) (-?[\d.]+)")


class _FakeTap:
    """Archive stand-in: honest COUNT(*), row queries cut at ``row_cap`` rows (as the anonymous
    ESA async jobs do); ``lie`` adds rows to every COUNT (a server that never agrees)."""

    def __init__(self, n=4000, row_cap=None, lie=0, seed=3, dec=(0.0, 10.0), ra=(0.0, 20.0)):
        rng = _np.random.default_rng(seed)
        self.df = pd.DataFrame({"SOURCE_ID": _np.arange(1, n + 1), "ra": rng.uniform(*ra, n),
                                "dec": rng.uniform(*dec, n), "g_mag": rng.uniform(8, 17, n)})
        self.row_cap, self.lie = row_cap, lie
        self.n_row_queries = 0

    def _mask(self, adql: str):
        d = self.df
        m_box = _np.zeros(len(d), dtype=bool)
        for r0, rop, r1, d0, dop, d1 in _BOX_RE.findall(adql):
            ra_hi = d["ra"] <= float(r1) if rop == "<=" else d["ra"] < float(r1)
            de_hi = d["dec"] <= float(d1) if dop == "<=" else d["dec"] < float(d1)
            m_box |= (d["ra"] >= float(r0)) & ra_hi & (d["dec"] >= float(d0)) & de_hi
        g_hi = float(_re.search(r"phot_g_mean_mag <= (-?[\d.]+)", adql).group(1))
        lo = _re.search(r"phot_g_mean_mag > (-?[\d.]+)", adql)
        m = m_box & (d["g_mag"] <= g_hi)
        if lo:
            m &= d["g_mag"] > float(lo.group(1))
        return m

    def __call__(self, adql: str) -> pd.DataFrame:
        m = self._mask(adql)
        if "COUNT(*)" in adql:
            return pd.DataFrame({"n": [int(m.sum()) + self.lie]})
        self.n_row_queries += 1
        out = self.df.loc[m].reset_index(drop=True)
        return out.iloc[: self.row_cap] if self.row_cap is not None else out


def _builder():
    import importlib.util
    import sys

    path = Path(__file__).resolve().parents[2] / "catalogs" / "build_gaia_catalog.py"
    spec = importlib.util.spec_from_file_location("build_gaia_catalog", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_gaia_catalog"] = mod
    spec.loader.exec_module(mod)
    return mod


def _cfg(b, db: Path, **kw):
    base = dict(db_path=db, mag_limit=17.5, dec_min=0.0, dec_max=10.0, ra_step=20.0, dec_step=10.0,
                max_strips=800, use_full_source=False, skip_vacuum=True, no_strip_clamp=True)
    base.update(kw)
    return b.BuildConfig(**base)


def test_g0503_truncated_strip_is_split_until_complete(tmp_path) -> None:
    import gaia_coverage as gc

    b = _builder()
    tap = _FakeTap(n=4000, row_cap=1500)  # one 20x10 deg strip, the "server" returns at most 1500 rows
    con = sqlite3.connect(tmp_path / "g.db")
    gc.init_gaia_tables(con)
    gc.ensure_strip_columns(con)
    strip = (0.0, 20.0, 0.0, 10.0)
    state, n_rows, n_new = b._store_strip(con, strip, False, True, mag_limit=17.5, use_full_source=False, fetch=tap)
    assert state == "ok" and n_rows == n_new == len(tap.df[(tap.df["ra"] < 20.0)])
    assert con.execute("SELECT COUNT(*) FROM gaia_dr3").fetchone()[0] == n_rows
    row = con.execute("SELECT n_stars, n_expected, verified FROM strip_progress").fetchone()
    assert row == (n_rows, n_rows, 1) and tap.n_row_queries > 1


def test_g0503_strip_never_done_when_count_and_rows_disagree(tmp_path) -> None:
    import gaia_coverage as gc

    b = _builder()
    tap = _FakeTap(n=500, lie=1)  # COUNT says one row more than any query returns
    con = sqlite3.connect(tmp_path / "g.db")
    gc.init_gaia_tables(con)
    gc.ensure_strip_columns(con)
    state, _n, _new = b._store_strip(con, (0.0, 20.0, 0.0, 10.0), False, True, mag_limit=17.5,
                                     use_full_source=False, fetch=tap)
    assert state == "truncated"
    assert con.execute("SELECT COUNT(*) FROM gaia_dr3").fetchone()[0] == 0  # rolled back
    assert con.execute("SELECT verified FROM strip_progress").fetchone()[0] == 0
    assert b.get_done_strips(con) == set()


def test_g0503_build_records_coverage_only_when_verified(tmp_path) -> None:
    import gaia_coverage as gc

    b = _builder()
    db = tmp_path / "g.db"
    tap = _FakeTap(n=3000, row_cap=1000, ra=(0.0, 360.0))
    b.create_local_gaia(_cfg(b, db, ra_step=180.0), fetch=tap)
    con = sqlite3.connect(db)
    assert con.execute("SELECT COUNT(*) FROM gaia_dr3").fetchone()[0] == 3000
    assert {r[0] for r in con.execute("SELECT verified FROM strip_progress")} == {1}
    con.close()
    assert gc.covered_depth_g(db, (10.0, 11.0, 2.0, 3.0)) == pytest.approx(17.5)
    meta = __import__("catalog_db").read_catalog_meta(db)["gaia_dr3"]["params"]
    assert meta["strips_unverified"] == 0 and meta["truncated_strips"] == []


def test_g0503_detection_lists_legacy_strip_at_cap(tmp_path, capsys) -> None:
    import gaia_coverage as gc

    b = _builder()
    db = tmp_path / "g.db"
    con = sqlite3.connect(db)
    gc.init_gaia_tables(con)
    con.executemany("INSERT INTO strip_progress (strip_key, ra_min, ra_max, dec_min, dec_max, n_stars) VALUES (?,?,?,?,?,?)",
                    [("a", 258.3495, 270.0926, -34.0319, -26.0365, 3_000_000), ("b", 0, 11.7, 0, 8, 120_000)])
    con.commit()
    con.close()
    bad = b.check_strips(db)
    assert [s["strip_key"] for s in bad] == ["a", "b"]
    assert bad[0]["at_anonymous_cap"] and bad[0]["state"] == "never_checked" and not bad[1]["at_anonymous_cap"]
    assert "AT ANONYMOUS TAP ROW CAP" in capsys.readouterr().out


def test_g0503_repair_verifies_or_redownloads_resumable(tmp_path) -> None:
    import gaia_coverage as gc

    b = _builder()
    db = tmp_path / "g.db"
    tap = _FakeTap(n=2000, ra=(0.0, 360.0))
    con = sqlite3.connect(db)
    gc.init_gaia_tables(con)
    full = tap.df[tap.df["ra"] < 180.0]
    gc.insert_gaia_rows(con, full)  # strip A complete (legacy, never checked)
    gc.insert_gaia_rows(con, tap.df[tap.df["ra"] >= 180.0].iloc[:300])  # strip B cut at 300 rows
    con.executemany("INSERT INTO strip_progress (strip_key, ra_min, ra_max, dec_min, dec_max, n_stars) VALUES (?,?,?,?,?,?)",
                    [(b.strip_key((0.0, 180.0, 0.0, 10.0)), 0.0, 180.0, 0.0, 10.0, len(full)),
                     (b.strip_key((180.0, 360.0, 0.0, 10.0)), 180.0, 360.0, 0.0, 10.0, 300)])
    con.commit()
    con.close()
    out = b.repair_strips(_cfg(b, db, repair=True), fetch=tap)
    assert out["verified"] == 1 and out["downloaded"] == 1 and out["new_rows"] == 2000 - 300 - len(full)
    con = sqlite3.connect(db)
    assert con.execute("SELECT COUNT(*) FROM gaia_dr3").fetchone()[0] == 2000
    assert {r[0] for r in con.execute("SELECT verified FROM strip_progress")} == {1}
    con.close()
    again = b.repair_strips(_cfg(b, db, repair=True), fetch=tap)
    assert again == {"verified": 0, "downloaded": 0, "truncated": 0, "error": 0, "new_rows": 0}


def test_g0503_truncated_strip_has_no_coverage_claim(tmp_path) -> None:
    import gaia_coverage as gc

    db = tmp_path / "g.db"
    con = sqlite3.connect(db)
    gc.init_gaia_tables(con)
    gc.ensure_strip_columns(con)
    con.execute("INSERT INTO gaia_dr3 (source_id, ra, dec, g_mag) VALUES (1, 100, 5, 17.0)")
    con.execute("INSERT INTO strip_progress (strip_key, ra_min, ra_max, dec_min, dec_max, n_stars, verified) "
                "VALUES ('bulge', 258.35, 270.09, -34.03, -26.04, 3000000, 0)")
    con.commit()
    con.close()
    gc.record_coverage(db, box=(0.0, 360.0, -90.0, 90.0), g_max=17.5, source="build_gaia_catalog")
    assert gc.covered_depth_g(db, (264.0, 265.0, -30.0, -29.0)) is None
    assert gc.covered_depth_g(db, (100.0, 101.0, 4.0, 5.0)) == pytest.approx(17.5)


def test_new2_topup_splits_and_records_only_complete(tmp_path) -> None:
    import gaia_coverage as gc

    db = tmp_path / "g.db"
    con = sqlite3.connect(db)
    gc.init_gaia_tables(con)
    con.commit()
    con.close()
    gc.record_coverage(db, box=(0.0, 360.0, -90.0, 90.0), g_max=8.0, source="build_gaia_catalog")
    tap = _FakeTap(n=3000, row_cap=100)
    res = gc.top_up_field(db, box=(2.0, 6.0, 2.0, 6.0), g_from=8.0, g_to=17.0, fetch=tap)
    want = int(tap._mask(gc.build_topup_adql((2.0, 6.0, 2.0, 6.0), g_from=8.0, g_to=17.0)).sum())
    assert res["status"] == "ok" and res["n_fetched"] == want > 100
    assert gc.covered_depth_g(db, (3.0, 4.0, 3.0, 4.0)) == pytest.approx(17.0)
    liar = _FakeTap(n=3000, lie=1, seed=9)
    res2 = gc.top_up_field(db, box=(12.0, 16.0, 2.0, 6.0), g_from=8.0, g_to=17.0, fetch=liar)
    assert res2["status"] == "incomplete" and "COUNT" in res2["error"]
    assert gc.covered_depth_g(db, (13.0, 14.0, 3.0, 4.0)) == pytest.approx(8.0)  # no claim


# --- G04-45 / G05-55: reading coverage and provenance never write the Gaia DB --------------------------

def _legacy_gaia_db(path: Path) -> Path:
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE gaia_dr3 (source_id INTEGER PRIMARY KEY, ra REAL, dec REAL, g_mag REAL)")
    con.executemany("INSERT INTO gaia_dr3 VALUES (?,?,?,?)", [(i, i % 360, -89.95 + i * 0.01, 10 + i % 5) for i in range(1, 2000)])
    con.commit()
    con.close()
    return path


def test_g0445_load_coverage_is_read_only(tmp_path) -> None:
    import gaia_coverage as gc

    db = _legacy_gaia_db(tmp_path / "g.db")
    before = db.read_bytes()
    rec = gc.load_coverage(db)
    assert rec and rec[0]["source"] == "registered_existing_db"
    assert gc.covered_depth_g(db, (10.0, 11.0, -80.0, -79.0)) == pytest.approx(14.0)
    assert db.read_bytes() == before and not gc.coverage_is_stored(db)
    assert gc.register_existing_db(db)["g_max"] == pytest.approx(14.0)
    assert gc.coverage_is_stored(db) and gc.register_existing_db(db) is None


def test_g0555_provenance_unregistered_and_no_write(tmp_path) -> None:
    import catalog_provenance as cp

    db = _legacy_gaia_db(tmp_path / "g.db")
    before = db.read_bytes()
    fp = cp.fingerprint_gaia_db(db)
    assert fp["coverage"] == "unregistered" and db.read_bytes() == before


# --- G05-16: VACUUM blocked by a reader does not fail the objects update ------------------------------

def test_g0516_vacuum_locked_keeps_rc_zero(tmp_path) -> None:
    import argparse
    import importlib.util
    import sys

    path = Path(__file__).resolve().parents[2] / "catalogs" / "objects_make.py"
    spec = importlib.util.spec_from_file_location("objects_make", path)
    om = importlib.util.module_from_spec(spec)
    sys.modules["objects_make"] = om
    spec.loader.exec_module(om)
    src = tmp_path / "src.db"
    s = sqlite3.connect(src)
    om.ensure_schema(s)
    s.execute("INSERT INTO vsx_data VALUES (1, 'V1', 1.0, 1.0, 'EA', 1.0, 12.0, 13.0)")
    s.commit()
    s.close()

    class _Locked:
        """The objects DB while VYVAR reads it: VACUUM cannot get its lock."""

        def __init__(self, con):
            self._con = con

        def execute(self, sql, *a):
            if sql.strip().upper() == "VACUUM":
                raise sqlite3.OperationalError("database is locked")
            return self._con.execute(sql, *a)

        def __getattr__(self, name):
            return getattr(self._con, name)

    con = sqlite3.connect(tmp_path / "o.db")
    args = argparse.Namespace(import_vsx=str(src), vsx=False, import_exoplanets="", exoplanets=False)
    rc = om._run(_Locked(con), args)
    assert rc == 0
    assert con.execute("SELECT COUNT(*) FROM vsx_data").fetchone()[0] == 1
    con.close()


# --- G05-47: frames.csv cache keyed on (mtime_ns, size) ---------------------------------------------

def test_g0547_frame_table_reloaded_on_same_mtime_other_size(tmp_path) -> None:
    import proc_frame_store as pfs

    p = pfs.frame_table_path(tmp_path)
    p.write_text("proc_csv,airmass\na.csv,1.1\n", encoding="ascii")
    ns = p.stat().st_mtime_ns
    assert pfs.load_frame_table(tmp_path)["a.csv"]["airmass"] == pytest.approx(1.1)
    p.write_text("proc_csv,airmass\na.csv,1.25\n", encoding="ascii")
    os.utime(p, ns=(ns, ns))  # same mtime (coarse clock), other size
    assert pfs.load_frame_table(tmp_path)["a.csv"]["airmass"] == pytest.approx(1.25)


# --- G01-22 / G01-37: calibration library ---------------------------------------------------------

def _tcs():
    """Synthetic frame helpers of test_calibration_spec_2026 (same masters and lights)."""
    import importlib.util
    import sys

    if "_tcs_helpers" in sys.modules:
        return sys.modules["_tcs_helpers"]
    spec = importlib.util.spec_from_file_location("_tcs_helpers", Path(__file__).with_name("test_calibration_spec_2026.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_tcs_helpers"] = mod
    spec.loader.exec_module(mod)
    return mod


def _cal_env(tmp_path):
    import calib_library as cl
    import calib_masters as cm

    tcs = _tcs()

    db = VyvarDatabase(tmp_path / "t.sqlite3")
    cam = db.insert_equipment("C5A-150M", "cam", "CMOS", "x", 3.76)
    tel = db.insert_telescope("AZ800", "az", 800.0, 5480.0)
    cl.ensure_schema(db)
    rng = _np.random.default_rng(5)
    cm.build_dark(tcs._raw_darks(tmp_path / "src", 60.0, 5, tcs.T0, rng), library_root=tmp_path / "lib", db=db,
                  camera_id=cam)
    legacy = tcs._write(tmp_path / "lib" / "Flat_legacy.fits", tcs.P + 20000.0 * tcs._vign(), imagetyp="Flat Field",
                        exp=0.5, when=tcs.T0, filt="V", extra={"VYFLNRD": 1})
    cl.register_master(db, legacy, kind="flat", camera_id=cam, telescope_id=tel)
    lp = [tcs._write(tmp_path / "src" / f"L{i}.fits", _np.full((tcs.NY, tcs.NX), 2000.0), imagetyp="Light Frame",
                     exp=60.0, when=tcs.T0 + __import__("datetime").timedelta(days=1, minutes=i), filt="V")
          for i in range(2)]
    groups = {"V|60|1": {"filter": "V", "exposure_s": 60.0, "binning": 1, "gain": 0, "temp": -10.0,
                         "light_paths": [str(x) for x in lp]}}
    return db, cam, tel, legacy, groups


def test_g0122_scan_does_not_write_calibration_library(tmp_path) -> None:
    import calib_plan as cp

    db, cam, tel, legacy, groups = _cal_env(tmp_path)
    kw = dict(db=db, camera_id=cam, telescope_id=tel, dark_validity_days=90, flat_validity_days=200, temp_tol=0.5)

    def _row():
        return tuple(db.conn.execute("SELECT OFFSET_HANDLING, OFFSET_ADU FROM CALIBRATION_LIBRARY WHERE KIND = 'flat';").fetchone())

    before = _row()
    for _ in range(2):  # scan twice
        plan = cp.build_calibration_plan(groups, **kw)
        assert plan.groups["V|60|1"].flat_offset_handling == "removed-at-use"
        assert _row() == before
    cp.build_calibration_plan(groups, **kw, persist_offsets=True)  # RUN
    assert before[1] is None
    assert _row()[0] == "unknown" and _row()[1] == pytest.approx(plan.groups["V|60|1"].flat_offset_adu)


def test_g0137_import_never_overwrites_a_different_master(tmp_path) -> None:
    import calib_masters as cm

    tcs = _tcs()
    db = VyvarDatabase(tmp_path / "t.sqlite3")
    cam = db.insert_equipment("C5A-150M", "cam", "CMOS", "x", 3.76)
    lib = tmp_path / "lib"
    a = tcs._write(tmp_path / "x" / "a.fits", _np.full((tcs.NY, tcs.NX), 500.0), imagetyp="Dark Frame", exp=60.0,
                   when=tcs.T0)
    b = tcs._write(tmp_path / "y" / "a.fits", _np.full((tcs.NY, tcs.NX), 520.0), imagetyp="Dark Frame", exp=60.0,
                   when=tcs.T0)  # same header -> same library name, other pixels
    ra = cm.import_master(a, kind="dark", library_root=lib, db=db, camera_id=cam, telescope_id=None)
    rb = cm.import_master(b, kind="dark", library_root=lib, db=db, camera_id=cam, telescope_id=None)
    assert ra.path != rb.path and rb.path.name.endswith("_2.fits")
    assert float(__import__("astropy.io.fits", fromlist=["x"]).getdata(ra.path)[0, 0]) == 500.0
    again = cm.import_master(b, kind="dark", library_root=lib, db=db, camera_id=cam, telescope_id=None)
    assert again.path == rb.path and "already in the library" in again.messages[0]
    assert db.conn.execute("SELECT COUNT(*) FROM CALIBRATION_LIBRARY;").fetchone()[0] == 2
    assert not list(lib.glob("*.tmp"))
