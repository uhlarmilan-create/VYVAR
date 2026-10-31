"""CATALOG-WIZARD-2026-10: catalog builder in Settings (jobs, full-replace update, Gaia states)."""
from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
import sys
import time
import types
from pathlib import Path

import pytest

import catalog_db as cdb
import catalog_jobs as cj
import gaia_coverage as gc
import settings_model as sm

_REPO = Path(__file__).resolve().parents[2]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _REPO / "catalogs" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


om = _load("objects_make")


def _objects_db(path: Path, n: int = 100) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    om.ensure_schema(con)
    con.executemany("INSERT INTO vsx_data VALUES (?,?,?,?,?,?,?,?)",
                    [(i, f"V{i}", float(i % 360), 1.0, "EA", 1.0, 12.0, 13.0) for i in range(1, n + 1)])
    con.commit()
    return con


def _fill_changed(con: sqlite3.Connection, table: str) -> None:
    # 100 old rows: 1..10 removed, 11..15 changed period, 101..120 added
    rows = [(i, f"V{i}", float(i % 360), 1.0, "EA", 2.0 if i <= 15 else 1.0, 12.0, 13.0) for i in range(11, 121)]
    con.executemany(f"INSERT INTO {table} VALUES (?,?,?,?,?,?,?,?)", rows)
    con.commit()


def test_refresh_replaces_table_and_reports_difference(tmp_path: Path) -> None:
    con = _objects_db(tmp_path / "o.db")
    diff = om.refresh_table(con, "vsx_data", _fill_changed)
    assert diff == {"previous_rows": 100, "rows": 110, "added": 20, "removed": 10, "changed": 5}
    assert con.execute("SELECT period FROM vsx_data WHERE oid = 12").fetchone()[0] == 2.0
    assert con.execute("SELECT COUNT(*) FROM vsx_data WHERE oid <= 10").fetchone()[0] == 0
    names = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'index')")}
    assert "vsx_data_new" not in names and {"idx_vsx_ra", "idx_vsx_dec"} <= names


def test_failed_download_keeps_the_old_table(tmp_path: Path) -> None:
    con = _objects_db(tmp_path / "o.db")

    def _broken(c: sqlite3.Connection, table: str) -> None:
        c.execute(f"INSERT INTO {table} VALUES (1, 'x', 1, 1, 'EA', 1, 1, 1)")
        raise om.DownloadError("strip failed")

    with pytest.raises(om.DownloadError):
        om.refresh_table(con, "vsx_data", _broken)
    assert con.execute("SELECT COUNT(*) FROM vsx_data").fetchone()[0] == 100
    assert con.execute("SELECT 1 FROM sqlite_master WHERE name = 'vsx_data_new'").fetchone() is None


def test_truncated_download_is_refused_unless_forced(tmp_path: Path) -> None:
    con = _objects_db(tmp_path / "o.db")

    def _few(c: sqlite3.Connection, table: str) -> None:
        c.executemany(f"INSERT INTO {table} VALUES (?,?,?,?,?,?,?,?)", [(i, "x", 1, 1, "EA", 1, 1, 1) for i in range(1, 30)])

    with pytest.raises(om.DownloadError, match="truncated"):
        om.refresh_table(con, "vsx_data", _few)
    assert con.execute("SELECT COUNT(*) FROM vsx_data").fetchone()[0] == 100
    assert om.refresh_table(con, "vsx_data", _few, force=True)["rows"] == 29


def test_vsx_strip_failing_five_times_aborts(tmp_path: Path, monkeypatch) -> None:
    class _Viz:
        def __init__(self, **_k):
            pass

        def query_constraints(self, **_k):
            raise TimeoutError("VizieR timeout")

    fake = types.ModuleType("astroquery.vizier")
    fake.Vizier = _Viz
    monkeypatch.setitem(sys.modules, "astroquery.vizier", fake)
    monkeypatch.setattr(om.time, "sleep", lambda _s: None)
    con = _objects_db(tmp_path / "o.db")
    with pytest.raises(om.DownloadError, match="failed 5 times"):
        om.refresh_table(con, "vsx_data", lambda c, t: om.build_vsx(c, dec_min=0, dec_max=2, mag_limit=18,
                                                                     batch_deg=1, table=t))
    assert con.execute("SELECT COUNT(*) FROM vsx_data").fetchone()[0] == 100


def test_catalog_job_runs_as_own_process(tmp_path: Path) -> None:
    db = tmp_path / "g.db"
    con = sqlite3.connect(db)
    gc.init_gaia_tables(con)
    con.executemany("INSERT INTO gaia_dr3 (source_id, ra, dec, g_mag) VALUES (?,?,?,?)", [(i, 1, 1, 12) for i in range(1, 50)])
    con.commit()
    con.close()
    jd = cj.launch("gaia_register", cj.tool_args("gaia_register", db=db), tmp_path)
    for _ in range(100):
        if cj.job_state(jd) in ("done", "failed"):
            break
        time.sleep(0.1)
    assert cj.job_state(jd) == "done", cj.tail(jd)
    assert (cj.read_status(jd) or {}).get("returncode") == 0
    meta = cdb.read_catalog_meta(db)["gaia_dr3"]
    assert meta["row_count"] == 49 and "gaia_source_lite" in meta["source"]
    assert cj.latest_job(tmp_path, "gaia") == jd and cj.latest_job(tmp_path, "objects") is None


def test_second_job_of_a_catalog_is_refused_while_one_runs(tmp_path: Path) -> None:
    jd = tmp_path / "jobs" / "catalog_objects_update_20261020_120000"
    jd.mkdir(parents=True)
    (jd / cj.STATUS_FILE).write_text(json.dumps({"state": "running", "pid": os.getpid(), "tool": "objects_update",
                                                 "catalog": "objects", "label": "Update VSX + exoplanets"}))
    with pytest.raises(RuntimeError, match="still running"):
        cj.launch("objects_update", ["--db", "x"], tmp_path)


@pytest.mark.parametrize("tool,lines,args,expected", [
    ("gaia_build", ["  Strips:      800", "[ 200/800] RA 0.00-7.50 | ETA ~1h | inserts ~1 ok"], [], 0.25),
    ("objects_update", ["[VSX] Dec 0.0 .. 1.0 deg", "   10 rows (10 new)"], ["--vsx-dec-min", "-90", "--vsx-dec-max", "90"], 0.95 * 91 / 180),
    ("objects_update", ["[vsx_data] replaced: 10 rows", "[EXO] toi"], [], 0.97),
    ("astro_index", ["base stars: strip 5/10, 100 stars, 3 s"], [], 0.15),
])
def test_progress_from_script_output(tmp_path: Path, tool, lines, args, expected) -> None:
    jd = tmp_path / "jobs" / f"catalog_{tool}_x"
    jd.mkdir(parents=True)
    (jd / cj.STATUS_FILE).write_text(json.dumps({"tool": tool}))
    (jd / cj.JOB_FILE).write_text(json.dumps({"args": args}))
    (jd / cj.STDOUT_FILE).write_text("\n".join(lines) + "\n")
    assert cj.progress(jd) == pytest.approx(expected, abs=1e-6)


def test_tool_arguments_name_the_paths() -> None:
    a = cj.tool_args("gaia_build", out="C:/d/catalogs/g.db", mag_limit=17.5, dec_min=-30, dec_max=90)
    assert a == ["--out", "C:/d/catalogs/g.db", "--mag-limit", "17.5", "--dec-min", "-30", "--dec-max", "90"]
    o = cj.tool_args("objects_update", db="o.db", vsx=False, exoplanets=True)
    assert o == ["--db", "o.db", "--exoplanets"]
    for tool, (script, _cat, _label) in cj.TOOLS.items():
        assert (cj.catalogs_dir() / script).is_file(), tool


def test_unfinished_gaia_build_is_shown(tmp_path: Path) -> None:
    db = tmp_path / "g.db"
    con = sqlite3.connect(db)
    gc.init_gaia_tables(con)
    con.execute("INSERT INTO gaia_dr3 (source_id, ra, dec, g_mag, bp_rp) VALUES (1, 1, 1, 12, 0.5)")
    con.execute("INSERT INTO strip_progress (strip_key, ra_min, ra_max, dec_min, dec_max, n_stars) VALUES ('k', 0, 7.5, 0, 5, 1)")
    con.commit()
    con.close()
    st = sm.gaia_status(db)
    assert st.state == "unfinished" and dict(st.facts)["Strips done"] == "1"


def test_settings_page_catalog_builder(tmp_path: Path, monkeypatch) -> None:
    """Paths group renders the builder; Report runs as a job; an unsaved path edit locks the buttons."""
    from streamlit.testing.v1 import AppTest

    root = tmp_path / "data"
    (root / "catalogs").mkdir(parents=True)
    g = root / "catalogs" / "vyvar_gaia_dr3.db"
    con = sqlite3.connect(g)
    gc.init_gaia_tables(con)
    con.executemany("INSERT INTO gaia_dr3 (source_id, ra, dec, g_mag, bp_rp) VALUES (?,?,?,?,?)", [(i, 1, 1, 12, 0.5) for i in range(1, 60)])
    for sql in cdb.GAIA_INDEXES.values():
        con.execute(sql)
    con.commit()
    cdb.write_catalog_meta(con, "gaia_dr3", source="t", params={}, builder="t")
    con.close()
    gc.record_coverage(g, box=(0, 360, -90, 90), g_max=17.5, source="build_gaia_catalog")
    o = root / "catalogs" / "vyvar_objects.db"
    _objects_db(o).close()
    (root / "config.json").write_text(json.dumps({"gaia_db_path": str(g), "objects_db_path": str(o)}), encoding="utf-8")
    monkeypatch.setenv("VYVAR_DATA_DIR", str(root))
    app = tmp_path / "app.py"
    app.write_text("import sys\nfrom pathlib import Path\n"
                   f"sys.path.insert(0, {str(_REPO / 'src_py')!r})\n"
                   "from config import AppConfig\nimport ui_settings\n"
                   f"ui_settings.render_settings_dashboard(AppConfig(project_root=Path({str(root)!r})))\n")
    at = AppTest.from_file(str(app), default_timeout=60)
    at.run()
    at.radio(key="vyvar_set_group").set_value("paths").run()
    assert not at.exception, at.exception
    assert [t.label for t in at.tabs] == ["Gaia DR3", "VSX + exoplanets", "Blind index"]
    labels = {b.label: b for b in at.button}
    assert labels["Drop unused indexes"].disabled and not labels["Report"].disabled
    labels["Report"].click().run()
    for _ in range(100):
        jd = cj.latest_job(root, "gaia")
        if jd is not None and cj.job_state(jd) == "done":
            break
        time.sleep(0.1)
    assert jd is not None and cj.job_state(jd) == "done"
    at.run()
    assert any("Gaia DB report: done" in s.value for s in at.success)
    at.text_input(key="vyvar_set_w_objects_db_path").set_value(str(tmp_path / "other.db")).run()
    assert any("Save it first" in i.value for i in at.info)
    assert all(b.disabled for b in at.button if b.label.startswith(("Update", "Report", "Build")))


def test_failed_vsx_does_not_block_exoplanets(tmp_path: Path, monkeypatch, capsys) -> None:
    db = tmp_path / "o.db"
    _objects_db(db).close()

    def _vsx_down(*_a, **_k):
        raise om.DownloadError("VizieR down")

    def _exo(conn, *, mag_limit, confirmed_only, table_out):
        conn.executemany(f"INSERT INTO {table_out} VALUES (?,?,?,?,?,?,?,?,?,?)",
                         [(f"P{i}", f"P{i}", "H", 1.0, 2.0, "CONFIRMED", "CONFIRMED", 3.0, 11.0, "TESS") for i in range(5)])
        return 5

    monkeypatch.setattr(om, "build_vsx", _vsx_down)
    monkeypatch.setattr(om, "build_exoplanets", _exo)
    assert om.main(["--db", str(db), "--vsx", "--exoplanets"]) == 1
    out = capsys.readouterr().out
    assert "[vsx_data] unchanged" in out and "FAILED: vsx_data" in out
    meta = cdb.read_catalog_meta(db)
    assert meta["exoplanet_data"]["row_count"] == 5 and meta["exoplanet_data"]["params"]["update"]["added"] == 5
    con = sqlite3.connect(db)
    assert con.execute("SELECT COUNT(*) FROM vsx_data").fetchone()[0] == 100
    con.close()


def test_archive_http_error_is_retried_and_keeps_the_table(tmp_path: Path, monkeypatch, capsys) -> None:
    """2026-10-20 on Milan's PC: HTTP 526 on the TAP_SCHEMA column query crashed the update."""
    import urllib.error

    calls = []

    def _526(adql: str) -> str:
        calls.append(adql)
        raise urllib.error.HTTPError("https://exoplanetarchive.ipac.caltech.edu/TAP/sync", 526, "<none>", {}, None)

    monkeypatch.setattr(om, "_tap_csv", _526)
    monkeypatch.setattr(om.time, "sleep", lambda _s: None)
    db = tmp_path / "o.db"
    con = _objects_db(db)
    con.execute("INSERT INTO exoplanet_data VALUES ('P1','P1','H',1,2,'CONFIRMED','CONFIRMED',3,11,'TESS')")
    con.commit()
    con.close()
    assert om.main(["--db", str(db), "--exoplanets"]) == 1
    out = capsys.readouterr().out
    assert "failed 5 times" in out and "HTTP Error 526" in out and "[exoplanet_data] unchanged" in out
    assert len(calls) == 5 and all("TAP_SCHEMA" in c for c in calls)
    con = sqlite3.connect(db)
    assert con.execute("SELECT COUNT(*) FROM exoplanet_data").fetchone()[0] == 1
    assert con.execute("SELECT 1 FROM sqlite_master WHERE name = 'exoplanet_data_new'").fetchone() is None
    con.close()
