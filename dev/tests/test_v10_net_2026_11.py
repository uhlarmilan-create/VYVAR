# -*- coding: ascii -*-
"""VYVAR 1.0 item 19 (Milan 2026-11-04): Gaia archive TAP jobs never hang (NET-GAIA-TAP-DEADLINE-2026-11).

Fake job objects, a fake clock and fake fetchers stand in for the ESA archive (no network here)."""
from __future__ import annotations

import importlib.util
import re
import sqlite3
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import pytest

import gaia_tap_job as gtj
import net_call

ROOT = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------------ fakes
class _Clock:
    def __init__(self) -> None:
        self.t = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.t

    def sleep(self, dt: float) -> None:
        self.sleeps.append(dt)
        self.t += dt


class _Resp:
    def __init__(self, status: int = 200) -> None:
        self.status = status
        self.reason = "OK"


class _Conn:
    def __init__(self, status: int = 200) -> None:
        self.posts: list[tuple[str, str]] = []
        self.status = status

    def execute_tappost(self, subcontext: str, data: str) -> _Resp:
        self.posts.append((subcontext, data))
        return _Resp(self.status)


class _Job:
    """astroquery Job stand-in: phases returned one per poll (the last one repeats)."""

    def __init__(self, phases, results=None, jobid="1762000000000O", poll_error=None) -> None:
        self.jobid = jobid
        self._phases = list(phases)
        self._i = 0
        self.results = results
        self.connHandler = _Conn()
        self.poll_error = poll_error
        self.n_polls = 0

    def get_phase(self, *, update: bool = False) -> str:
        self.n_polls += 1
        if self.poll_error is not None:
            raise self.poll_error
        ph = self._phases[min(self._i, len(self._phases) - 1)]
        self._i += 1
        return ph

    def get_results(self):
        return self.results

    def get_error(self) -> str:
        return "Query syntax error"


def _direct(fn, _timeout, _what):
    return fn()


def _run(job, clock, **kw):
    lines: list[str] = []
    kw.setdefault("exec_deadline_s", 60.0)
    out = None
    try:
        out = gtj.run_tap_job("SELECT COUNT(*) AS n FROM gaiadr3.gaia_source", what="T", launch=lambda q: job,
                              clock=clock, sleep=clock.sleep, call=_direct, poll_s=5.0, report_every_s=20.0,
                              report=lines.append, **kw)
    finally:
        _run.lines = lines
    return out


# ------------------------------------------------------------------ a) the shared runner
def test_job_stuck_in_queued_is_aborted_and_raises_tap_timeout() -> None:
    clock = _Clock()
    job = _Job(["QUEUED"])
    with pytest.raises(gtj.TapTimeout) as ei:
        _run(job, clock, exec_deadline_s=60.0)
    exc = ei.value
    assert exc.phase == "QUEUED" and exc.aborted and exc.jobid == job.jobid
    assert 60.0 <= exc.elapsed_s <= 60.0 + 5.0  # within the deadline + one poll interval
    assert job.connHandler.posts == [(f"async/{job.jobid}/phase", "PHASE=ABORT")]  # UWS abort on the server
    assert "aborted on the server" in str(exc)
    # b) progress never looks frozen: phase + elapsed at the report interval
    assert any("QUEUED" in ln and "20s" in ln for ln in _run.lines)
    assert len(_run.lines) == 2  # 20 s and 40 s; at 60 s the timeout


def test_completed_job_returns_result_without_abort() -> None:
    clock = _Clock()
    job = _Job(["QUEUED", "EXECUTING", "EXECUTING", "COMPLETED"], results="TABLE")
    assert _run(job, clock) == "TABLE"
    assert job.connHandler.posts == [] and job.n_polls == 4


def test_failed_job_raises_with_archive_error_text() -> None:
    clock = _Clock()
    with pytest.raises(gtj.TapJobFailed, match="ERROR on the archive: Query syntax error"):
        _run(_Job(["EXECUTING", "ERROR"]), clock)


def test_unreadable_phase_gives_up_and_aborts() -> None:
    clock = _Clock()
    job = _Job(["QUEUED"], poll_error=OSError("connection reset"))
    with pytest.raises(gtj.TapJobFailed, match="unreadable 3 times"):
        _run(job, clock)
    assert job.n_polls == gtj.TAP_POLL_ERRORS_MAX and job.connHandler.posts


def test_ctrl_c_aborts_the_job_on_the_server() -> None:
    clock = _Clock()
    job = _Job(["EXECUTING"])

    def _interrupt(_dt: float) -> None:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        gtj.run_tap_job("SELECT 1", launch=lambda q: job, clock=clock, sleep=_interrupt, call=_direct,
                        report=lambda m: None)
    assert job.connHandler.posts == [(f"async/{job.jobid}/phase", "PHASE=ABORT")]


def test_failed_abort_is_reported_in_the_timeout() -> None:
    clock = _Clock()
    job = _Job(["EXECUTING"])
    job.connHandler.status = 500
    with pytest.raises(gtj.TapTimeout) as ei:
        _run(job, clock, exec_deadline_s=10.0)
    assert not ei.value.aborted and "abort of job" in str(ei.value)


def test_hanging_submit_and_download_raise_tap_timeout_in_real_time() -> None:
    """The real thread timeout bounds every blocking HTTP step (astroquery has no socket timeout)."""
    ev = threading.Event()
    t0 = time.monotonic()
    with pytest.raises(gtj.TapTimeout) as ei:
        gtj.run_tap_job("SELECT 1", launch=lambda q: ev.wait(30), http_timeout_s=1.0, report=lambda m: None)
    assert ei.value.phase == "SUBMIT"
    job = _Job(["COMPLETED"])
    job.get_results = lambda: ev.wait(30)
    with pytest.raises(gtj.TapTimeout) as ei2:
        gtj.run_tap_job("SELECT 1", launch=lambda q: job, download_deadline_s=1.0, report=lambda m: None)
    assert ei2.value.phase == "DOWNLOAD"
    ev.set()
    assert time.monotonic() - t0 < 10.0


def test_default_launch_submits_in_background(monkeypatch) -> None:
    from astroquery.gaia import Gaia

    seen = {}

    def _fake(query, **kw):
        seen.update(kw)
        return _Job(["COMPLETED"], results=None)

    monkeypatch.setattr(Gaia, "launch_job_async", _fake)
    assert gtj.tap_dataframe("SELECT COUNT(*) AS n FROM t", report=lambda m: None).empty
    assert seen.get("background") is True  # astroquery must not wait for the job itself


def test_tap_dataframe_deadlines_by_query_kind(monkeypatch) -> None:
    got = []
    monkeypatch.setattr(gtj, "run_tap_job", lambda adql, **kw: got.append(kw) or None)
    gtj.tap_dataframe("SELECT COUNT(*) AS n FROM t")
    gtj.tap_dataframe("SELECT source_id FROM t")
    assert got[0]["exec_deadline_s"] == gtj.TAP_COUNT_EXEC_DEADLINE_S
    assert got[0]["download_deadline_s"] == gtj.TAP_HTTP_TIMEOUT_S and got[0]["what"].endswith("COUNT")
    assert got[1]["exec_deadline_s"] == gtj.TAP_ROWS_EXEC_DEADLINE_S
    assert got[1]["download_deadline_s"] == gtj.TAP_ROWS_DOWNLOAD_DEADLINE_S
    # a healthy anonymous job (~100 s measured) is far inside every deadline
    assert min(gtj.TAP_COUNT_EXEC_DEADLINE_S, gtj.TAP_ROWS_EXEC_DEADLINE_S) >= 5 * 100


def test_sync_query_has_socket_timeout_and_deadline() -> None:
    seen = {}

    class _R:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"source_id\n1\n"

    def _opener(req, timeout):
        seen["timeout"] = timeout
        return _R()

    assert gtj.run_sync_query("https://x/sync", {"QUERY": "q"}, opener=_opener) == "source_id\n1\n"
    assert seen["timeout"] == gtj.TAP_HTTP_TIMEOUT_S
    ev = threading.Event()
    with pytest.raises(gtj.TapTimeout):
        gtj.run_sync_query("https://x/sync", {"QUERY": "q"}, deadline_s=1.0, opener=lambda r, timeout: ev.wait(30))
    ev.set()


def test_gspc_uses_the_bounded_sync_query(monkeypatch, tmp_path) -> None:
    import gspc_zp

    seen = {}

    def _fake(url, params, **kw):
        seen.update(kw, url=url)
        raise gtj.TapTimeout("stuck", what="x", phase="SYNC", elapsed_s=1, deadline_s=1)

    monkeypatch.setattr(gtj, "run_sync_query", _fake)
    with pytest.raises(gtj.TapTimeout):
        gspc_zp.fetch_gspc(["2172293994744599040"], tmp_path)
    assert seen["url"] == gspc_zp.TAP_SYNC_URL and seen["deadline_s"] == gtj.TAP_COUNT_EXEC_DEADLINE_S


# ------------------------------------------------------------------ online top-up / HRD: WARNING + meta, no hang
def _timeout(*_a, **_k):
    raise gtj.TapTimeout("job J still QUEUED after 10m00s; job J aborted on the server", what="t",
                         phase="QUEUED", elapsed_s=600, deadline_s=600, jobid="J", aborted=True)


def test_topup_timeout_is_a_warning_with_status(monkeypatch, tmp_path) -> None:
    import gaia_coverage as gc

    db = tmp_path / "g.db"
    con = sqlite3.connect(db)
    gc.init_gaia_tables(con)
    con.commit()
    con.close()
    gc.record_coverage(db, box=(0.0, 360.0, -90.0, 90.0), g_max=12.0, source="build_gaia_catalog")
    logs: list[str] = []
    monkeypatch.setattr(gc, "_log", logs.append)
    monkeypatch.setattr(gtj, "run_tap_job", _timeout)  # the production fetch (fetch=None)
    net_call.clear_unavailable()
    try:
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        res = gc.ensure_field_depth(db, center=SkyCoord(10 * u.deg, 20 * u.deg), radius_deg=0.3, g_needed=15.0)
        assert res["status"] == "timeout" and "TapTimeout" in res["error"] and "aborted" in res["error"]
        assert any(m.startswith("WARNING: GAIA TOP-UP not possible (timeout") for m in logs)
        assert gc.covered_depth_g(db, gc.field_box(SkyCoord(10 * u.deg, 20 * u.deg), 0.3)) == pytest.approx(12.0)
        assert net_call.unavailable_reason("gaia_tap_topup")  # the rest of the run does not wait again
    finally:
        net_call.clear_unavailable()


def test_hrd_timeout_is_not_retried_and_recorded(monkeypatch, tmp_path) -> None:
    import json

    import hrd_enrich

    calls = []
    events: list[str] = []
    monkeypatch.setattr(hrd_enrich, "run_tap_job", lambda *a, **k: calls.append(1) or _timeout())
    monkeypatch.setattr(hrd_enrich, "log_event", events.append)
    monkeypatch.setattr(hrd_enrich.time, "sleep", lambda s: pytest.fail("no retry wait after a timeout"))
    net_call.clear_unavailable()
    try:
        cache = tmp_path / "hrd_enrich.json"
        out = hrd_enrich.enrich_candidates(pd.DataFrame({"catalog_id": ["1234567890123456789"]}), cache,
                                           enabled=True, simbad_enabled=False)
        assert len(calls) == 1 and str(out.loc[0, "enrich_source"]) == "n/a"
        last = json.loads(cache.read_text(encoding="utf-8"))["last_enrich"]
        assert "aborted" in last["enrich_skip_reason"] and last["enrich_attempts"] == 1
        assert any(e.startswith("WARNING: HRD enrichment skipped") for e in events)
    finally:
        net_call.clear_unavailable()


# ------------------------------------------------------------------ builder / repair
def _builder():
    path = ROOT / "catalogs" / "build_gaia_catalog.py"
    spec = importlib.util.spec_from_file_location("build_gaia_catalog", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_gaia_catalog"] = mod
    spec.loader.exec_module(mod)
    return mod


def _cfg(b, db: Path, **kw):
    base = dict(db_path=db, mag_limit=17.5, dec_min=0.0, dec_max=10.0, ra_step=60.0, dec_step=10.0,
                max_strips=800, use_full_source=False, skip_vacuum=True, no_strip_clamp=True)
    base.update(kw)
    return b.BuildConfig(**base)


_RA_RE = re.compile(r"ra >= (-?[\d.]+) AND ra")


class _Tap:
    """COUNT(*) = ``n`` rows per strip (honest rows); strips whose RA start is in ``stuck`` time out."""

    def __init__(self, n=5, stuck=()) -> None:
        self.n, self.stuck = n, {float(x) for x in stuck}
        self.queries: list[tuple[float, str]] = []
        self.counts: dict[float, int] = {}

    def __call__(self, adql: str) -> pd.DataFrame:
        ra0 = float(_RA_RE.search(adql).group(1))
        kind = "COUNT" if "COUNT(*)" in adql else "rows"
        self.queries.append((ra0, kind))
        if ra0 in self.stuck:
            _timeout()
        n = self.counts.get(ra0, self.n)
        if kind == "COUNT":
            return pd.DataFrame({"n": [n]})
        base = int(ra0 * 1000) + 1
        return pd.DataFrame({"source_id": range(base, base + n), "ra": [ra0 + 0.5] * n, "dec": [5.0] * n,
                             "g_mag": [12.0] * n})


def test_tap_fetch_retries_a_timeout_once_then_raises() -> None:
    b = _builder()
    n = []
    with pytest.raises(gtj.TapTimeout):
        b._tap_fetch("SELECT COUNT(*) AS n FROM t", run=lambda q: n.append(1) or _timeout(), sleep=lambda s: None)
    assert len(n) == b.MAX_TIMEOUT_ATTEMPTS == 2
    m = []

    def _flaky(q):
        m.append(1)
        if len(m) < 4:
            raise OSError("HTTP 500")
        return pd.DataFrame({"n": [1]})

    assert int(b._tap_fetch("q", run=_flaky, sleep=lambda s: None).iloc[0, 0]) == 1 and len(m) == 4


def test_build_continues_after_timed_out_strip_and_lists_it(tmp_path, capsys) -> None:
    b = _builder()
    db = tmp_path / "g.db"
    tap = _Tap(stuck=[60.0])
    b.create_local_gaia(_cfg(b, db), fetch=tap)  # 6 strips of 60 deg; the second times out
    out = capsys.readouterr().out
    con = sqlite3.connect(db)
    done = b.get_done_strips(con)
    con.close()
    stuck_key = b.strip_key((60.0, 120.0, 0.0, 10.0))
    assert len(done) == 5 and stuck_key not in done
    assert "Failed strips: 1" in out and stuck_key in out and "timeout" in out and "Rerun the same command" in out
    assert [r for r, _k in tap.queries if r > 60.0]  # strips after the stuck one were fetched


def test_build_stops_after_consecutive_timeouts(tmp_path, capsys) -> None:
    b = _builder()
    tap = _Tap(stuck=[0.0, 60.0, 120.0, 180.0, 240.0, 300.0])
    b.create_local_gaia(_cfg(b, tmp_path / "g.db"), fetch=tap)
    out = capsys.readouterr().out
    assert len({r for r, _k in tap.queries}) == b.STOP_AFTER_TIMEOUT_STRIPS == 3
    assert "STOPPED" in out and "3 strip(s) not tried" in out


def _legacy_db(b, db: Path, strips: list[tuple[float, int]]) -> None:
    import gaia_coverage as gc

    con = sqlite3.connect(db)
    gc.init_gaia_tables(con)
    con.executemany(
        "INSERT INTO strip_progress (strip_key, ra_min, ra_max, dec_min, dec_max, n_stars) VALUES (?,?,?,?,?,?)",
        [(b.strip_key((r0, r0 + 60.0, 0.0, 10.0)), r0, r0 + 60.0, 0.0, 10.0, n) for r0, n in strips])
    con.commit()
    con.close()


def test_repair_continues_after_timed_out_strip_and_reports_it(tmp_path, capsys) -> None:
    b = _builder()
    db = tmp_path / "g.db"
    _legacy_db(b, db, [(0.0, 5), (60.0, 5), (120.0, 5)])
    tap = _Tap(stuck=[60.0])
    res = b.repair_strips(_cfg(b, db, repair=True), fetch=tap)
    out = capsys.readouterr().out
    assert res["verified"] == 2 and res["error"] == 1
    assert res["failed_strips"] == [b.strip_key((60.0, 120.0, 0.0, 10.0))] and res["not_tried"] == 0
    assert "TIMEOUT (COUNT)" in out and "Failed strips: 1" in out and "Rerun the same --repair command" in out
    again = b.repair_strips(_cfg(b, db, repair=True), fetch=_Tap())  # resumable: only the failed strip is left
    assert again["verified"] == 1 and again["failed_strips"] == []


def test_repair_orders_strips_at_the_anonymous_cap_last(tmp_path) -> None:
    import gaia_coverage as gc

    b = _builder()
    db = tmp_path / "g.db"
    cap = gc.ESA_ANON_ASYNC_ROW_CAP
    _legacy_db(b, db, [(0.0, 7), (60.0, cap), (120.0, 9), (180.0, cap), (240.0, 3)])
    tap = _Tap()
    tap.counts = {0.0: 7, 60.0: cap, 120.0: 9, 180.0: cap, 240.0: 3}  # every COUNT agrees -> verified only
    res = b.repair_strips(_cfg(b, db, repair=True), fetch=tap)
    order = [r for r, k in tap.queries if k == "COUNT"]
    assert order == [120.0, 0.0, 240.0, 180.0, 60.0]  # normal strips first (largest first), cap strips last
    assert res["verified"] == 5


# ------------------------------------------------------------------ item 20: user docs
def test_docs_show_system_python_and_the_archive_behaviour() -> None:
    """Milan's PC has no .venv: the Gaia builder / maintenance commands are `python catalogs\\...`."""
    docs = [ROOT / "docs" / n for n in ("INSTALL.md", "README_FULL.md", "README_CZ.md", "VYVAR_PIPELINE_CZ.md")]
    docs.append(ROOT / "catalogs" / "README.md")
    tool = re.compile(r"(\S*)\s*(?:catalogs|GAIA_DR3)[\\/](?:build_gaia_catalog|gaia_maintain|build_astro_index|"
                      r"objects_make)\.py")
    for p in docs:
        text = p.read_text(encoding="ascii")
        assert ".venv\\Scripts\\python.exe catalogs" not in text, p.name
        for m in re.finditer(r"^\s*(\S+) (?:catalogs|GAIA_DR3)[\\/]\w+\.py", text, re.M):
            assert m.group(1).lstrip("`(") == "python", (p.name, m.group(0))
        for m in tool.finditer(text):
            assert "GAIA_DR3" not in m.group(0), (p.name, m.group(0))
    install = (ROOT / "docs" / "INSTALL.md").read_text(encoding="ascii")
    for s in ("python catalogs\\build_gaia_catalog.py --repair", "Anonymous access is slow", "1-2 minutes",
              "Timeouts:", "Ctrl+C is safe", "Resume:", "No login is needed"):
        assert s in install, s


def test_maintain_report_prints_a_native_repair_command(tmp_path, capsys) -> None:
    spec = importlib.util.spec_from_file_location("gaia_maintain", ROOT / "catalogs" / "gaia_maintain.py")
    gm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gm)
    b = _builder()
    db = tmp_path / "g.db"
    import gaia_coverage as gc

    _legacy_db(b, db, [(0.0, gc.ESA_ANON_ASYNC_ROW_CAP)])
    gm.report(db)
    out = capsys.readouterr().out
    assert f"repair:        python {Path('catalogs') / 'build_gaia_catalog.py'} --out {db} --repair --strip" in out
