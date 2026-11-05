"""Gaia archive TAP calls with a deadline (NET-GAIA-TAP-DEADLINE-2026-11, Milan 2026-11-04).

Every call VYVAR makes to the ESA Gaia archive goes through this module: the catalogue builder and
its ``--repair`` (``catalogs/build_gaia_catalog.py``), the online per-field top-up
(``gaia_coverage``), the HRD enrichment (``hrd_enrich``) and the GSPC zero point (``gspc_zp``,
synchronous TAP).

Incident (2026-11-04): ``Gaia.launch_job_async(adql).get_results()`` (astroquery 0.4.11) never
returned - the archive job stayed stuck and the builder printed nothing for hours. astroquery
polls the job phase without any deadline and its HTTP connections have no socket timeout.

An asynchronous job here is: submit (astroquery, ``background=True``), poll the UWS phase with a
deadline, report the phase and elapsed time at a modest interval, download the result once the job
is COMPLETED. When the deadline passes the job is ABORTED on the server (UWS ``PHASE=ABORT``, a
plain POST through astroquery's connection handler: astroquery's own ``Job.abort`` refuses any job
that is not PENDING locally) and :class:`TapTimeout` is raised. Every blocking HTTP step runs under
``net_call.call_with_timeout``, so no single request can hang either.

Deadlines (measured 2026-11-04 on Milan's PC, anonymous ESA TAP; since 2026-10-14 the archive allows
5 jobs in execution and a queue of 50 per user, and anonymous users may share these slots):

* a tiny job takes 50-100 s end to end (sync COUNT of 1x1 deg 60 s; async job ~57 s to EXECUTING,
  ~86 s COMPLETED, ~99 s with its result);
* the COUNT(*) of the densest bulge strip (3,742,554 rows) COMPLETED in 98 s - the server-side
  scan time barely depends on the size of the box.

So a healthy job of any size reaches COMPLETED in ~100 s; the deadlines below give a 6x margin
(queue shared with other anonymous users) for COUNT and small jobs, twice that for a row job (the
server also writes the result table, up to the anonymous cap of 3,000,000 rows). The result
download is bounded separately: a row job returns at most the anonymous cap, ~3 M rows x ~80 bytes
(lite SELECT, VOTable binary) = ~250 MB before gzip; 30 min covers that at ~140 kB/s. The fetch
interface of the builder (``fetch(adql)``) does not know the expected rows, and the cap bounds
every job anyway, so the download deadline is not scaled per job.
"""

from __future__ import annotations

import logging
import re
import time
import urllib.parse
import urllib.request
from typing import Any, Callable

import pandas as pd

from net_call import call_with_timeout

LOGGER = logging.getLogger(__name__)

#: submit -> COMPLETED of a COUNT(*) or small (id list) job: 6x the slowest measured job (~100 s).
TAP_COUNT_EXEC_DEADLINE_S = 600.0
#: submit -> COMPLETED of a row job: twice the COUNT deadline (the server also writes the result).
TAP_ROWS_EXEC_DEADLINE_S = 1200.0
#: download + parse of a COMPLETED row job's result (<= anonymous cap of 3 M rows, see above).
TAP_ROWS_DOWNLOAD_DEADLINE_S = 1800.0
#: one HTTP request (submit, phase poll, error text, small result): 3x the longest single anonymous
#: request measured (the 60 s synchronous COUNT, which waits for the query before answering).
TAP_HTTP_TIMEOUT_S = 180.0
#: the abort POST after a deadline or Ctrl+C; it is a tiny request, and a job that cannot be
#: aborted expires on the server on its own - do not keep the user waiting for it.
TAP_ABORT_TIMEOUT_S = 30.0
#: phase poll interval (astroquery polls every 0.5 s; the archive asks clients not to hammer it).
TAP_POLL_S = 5.0
#: progress line interval while a job waits (a healthy job of ~100 s prints one or two lines).
TAP_REPORT_EVERY_S = 60.0
#: consecutive failed phase polls (HTTP errors) before the job is given up as failed.
TAP_POLL_ERRORS_MAX = 3

_WAITING_PHASES = frozenset({"PENDING", "QUEUED", "EXECUTING", "HELD", "SUSPENDED", "UNKNOWN"})
_FAILED_PHASES = frozenset({"ERROR", "ABORTED", "ARCHIVED"})


class TapTimeout(TimeoutError):
    """A Gaia archive call passed its deadline; the job was aborted on the server when possible."""

    def __init__(self, message: str, *, what: str, phase: str, elapsed_s: float, deadline_s: float,
                 jobid: str | None = None, aborted: bool = False) -> None:
        super().__init__(message)
        self.what = what
        self.phase = phase
        self.elapsed_s = float(elapsed_s)
        self.deadline_s = float(deadline_s)
        self.jobid = jobid
        self.aborted = bool(aborted)


class TapJobFailed(RuntimeError):
    """The archive reported the job as failed (ERROR / ABORTED / ARCHIVED) or stopped answering."""


class TapJobRejected(TapJobFailed):
    """The archive answered and put the job in phase ERROR: it rejected the query (ADQL it cannot
    parse, an unknown column, a limit of the service). The archive is reachable - this is not an
    outage, and retrying the same query cannot succeed (GAIA-TOPUP-ADQL-2026-11).
    ``archive_error`` is the archive's own error text (echoed query replaced by ``'<query>'``)."""

    def __init__(self, message: str, *, archive_error: str) -> None:
        super().__init__(message)
        self.archive_error = archive_error


def fmt_duration(seconds: float) -> str:
    s = max(0, int(round(float(seconds))))
    return f"{s // 60}m{s % 60:02d}s" if s >= 60 else f"{s}s"


def _launch_astroquery(adql: str) -> Any:
    """Submit an async job without waiting (astroquery would wait with no deadline)."""
    from astroquery.gaia import Gaia  # noqa: PLC0415

    return Gaia.launch_job_async(adql, background=True)


def abort_uws_job(job: Any) -> None:
    """UWS abort: POST PHASE=ABORT to <async>/<jobid>/phase (astroquery's ``Job.abort`` only
    works on a job it still holds as PENDING). Raises when the server does not accept it."""
    jobid = getattr(job, "jobid", None)
    if not jobid:
        raise TapJobFailed("job has no id; nothing to abort")
    resp = job.connHandler.execute_tappost(subcontext=f"async/{jobid}/phase",
                                           data=urllib.parse.urlencode({"PHASE": "ABORT"}))
    status = int(getattr(resp, "status", 0) or 0)
    if status not in (200, 303):
        raise TapJobFailed(f"abort of job {jobid} answered HTTP {status} {getattr(resp, 'reason', '')}".strip())


def run_tap_job(
    adql: str,
    *,
    what: str = "Gaia TAP job",
    exec_deadline_s: float = TAP_COUNT_EXEC_DEADLINE_S,
    download_deadline_s: float = TAP_HTTP_TIMEOUT_S,
    report: Callable[[str], None] | None = None,
    launch: Callable[[str], Any] | None = None,
    abort: Callable[[Any], None] | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    call: Callable[[Callable[[], Any], float, str], Any] = call_with_timeout,
    poll_s: float = TAP_POLL_S,
    report_every_s: float = TAP_REPORT_EVERY_S,
    http_timeout_s: float = TAP_HTTP_TIMEOUT_S,
) -> Any:
    """Run one asynchronous Gaia archive job with a deadline; returns its result (astropy Table).

    ``exec_deadline_s`` bounds submit -> COMPLETED; past it the job is aborted on the server and
    :class:`TapTimeout` is raised. ``download_deadline_s`` bounds the result download (TapTimeout,
    the job is finished then). A failed job raises :class:`TapJobFailed`; other errors of the
    archive / network propagate. Ctrl+C aborts the job on the server, then propagates.
    ``report`` gets a progress line every ``report_every_s`` while the job waits (default: the log).
    ``launch`` / ``abort`` / ``clock`` / ``sleep`` / ``call`` are seams for tests.
    """
    launch = launch or _launch_astroquery
    abort = abort or abort_uws_job
    report = report or LOGGER.info
    t0 = clock()

    def _elapsed() -> float:
        return clock() - t0

    def _http(fn: Callable[[], Any], label: str) -> Any:
        left = float(exec_deadline_s) - _elapsed()
        return call(fn, max(1.0, min(float(http_timeout_s), left)), f"{what} {label}")

    try:
        job = _http(lambda: launch(adql), "submit")
    except TimeoutError as exc:
        el = _elapsed()
        msg = (f"{what}: the Gaia archive did not accept the job within {fmt_duration(el)} "
               f"({exc}); if it was created it expires on the server")
        LOGGER.warning(msg)
        raise TapTimeout(msg, what=what, phase="SUBMIT", elapsed_s=el, deadline_s=exec_deadline_s) from exc
    jobid = str(getattr(job, "jobid", "") or "?")

    def _abort(reason: str) -> tuple[bool, str]:
        try:
            call(lambda: abort(job), TAP_ABORT_TIMEOUT_S, f"{what} abort")
            return True, f"job {jobid} aborted on the server ({reason})"
        except Exception as exc:  # noqa: BLE001 - reported in the message; the job expires on the server
            LOGGER.warning("%s: abort of job %s failed: %s", what, jobid, exc)
            return False, f"abort of job {jobid} failed ({type(exc).__name__}: {exc}); it expires on the server"

    phase = "QUEUED"
    last_report = t0
    poll_errors = 0
    try:
        while True:
            try:
                phase = str(_http(lambda: job.get_phase(update=True), "phase")).strip().upper()
                poll_errors = 0
            except Exception as exc:  # noqa: BLE001 - counted; a dead connection must not hang the loop
                poll_errors += 1
                LOGGER.warning("%s: phase poll %d/%d of job %s failed: %s", what, poll_errors,
                               TAP_POLL_ERRORS_MAX, jobid, exc)
                if poll_errors >= TAP_POLL_ERRORS_MAX:
                    _ok, note = _abort("phase polls failed")
                    raise TapJobFailed(f"{what}: phase of job {jobid} unreadable {poll_errors} times "
                                       f"({type(exc).__name__}: {exc}); {note}") from exc
            if phase == "COMPLETED":
                break
            if phase == "ERROR":
                err = _job_error(job, _http)
                raise TapJobRejected(f"{what}: job {jobid} {phase} on the archive: {err}", archive_error=err)
            if phase in _FAILED_PHASES:
                raise TapJobFailed(f"{what}: job {jobid} {phase} on the archive: {_job_error(job, _http)}")
            el = _elapsed()
            if el >= float(exec_deadline_s):
                aborted, note = _abort(f"deadline {fmt_duration(exec_deadline_s)}")
                msg = (f"{what}: job {jobid} still {phase} after {fmt_duration(el)} "
                       f"(deadline {fmt_duration(exec_deadline_s)}); {note}")
                LOGGER.warning(msg)
                raise TapTimeout(msg, what=what, phase=phase, elapsed_s=el, deadline_s=exec_deadline_s,
                                 jobid=jobid, aborted=aborted)
            if clock() - last_report >= float(report_every_s):
                report(f"{what}: job {jobid} {phase} {fmt_duration(el)} "
                       f"(deadline {fmt_duration(exec_deadline_s)})")
                last_report = clock()
            sleep(max(0.0, min(float(poll_s), float(exec_deadline_s) - el)))
    except KeyboardInterrupt:
        report(f"{what}: interrupted - aborting job {jobid} on the archive (Ctrl+C again to quit at once)")
        _abort("interrupted")
        raise

    t_dl = clock()
    try:
        return call(job.get_results, float(download_deadline_s), f"{what} result")
    except TimeoutError as exc:
        el = clock() - t_dl
        msg = (f"{what}: result of job {jobid} not downloaded within {fmt_duration(download_deadline_s)} "
               f"({exc})")
        LOGGER.warning(msg)
        raise TapTimeout(msg, what=what, phase="DOWNLOAD", elapsed_s=el, deadline_s=download_deadline_s,
                         jobid=jobid) from exc


#: The archive echoes the whole query in its error text ("Cannot parse query 'SELECT ...' ...");
#: the echo is replaced so that the reason after it is not cut off by the length limit. VYVAR ADQL
#: has no string literals, so the echo ends at the next single quote.
_ECHOED_QUERY = re.compile(r"'\s*SELECT\b[^']*'", re.IGNORECASE)
#: Length of the archive error text kept in messages / run meta (the reason, not the query echo).
_JOB_ERROR_MAX_CHARS = 600


def compact_archive_error(text: str) -> str:
    """The archive error text with the echoed query replaced by ``'<query>'``, one line, bounded."""
    t = _ECHOED_QUERY.sub("'<query>'", str(text))
    t = " ".join(t.split())
    return t[:_JOB_ERROR_MAX_CHARS]


def _job_error(job: Any, http: Callable[[Callable[[], Any], str], Any]) -> str:
    try:
        return compact_archive_error(str(http(lambda: job.get_error(), "error")))
    except Exception as exc:  # noqa: BLE001 - the phase already says the job failed
        return f"error text unavailable ({type(exc).__name__}: {exc})"


def is_count_query(adql: str) -> bool:
    return adql.lstrip().upper().startswith("SELECT COUNT(")


def tap_dataframe(adql: str, *, what: str = "Gaia TAP", report: Callable[[str], None] | None = None,
                  **kw: Any) -> pd.DataFrame:
    """:func:`run_tap_job` as a DataFrame (the ``fetch(adql)`` of the builder and the top-up). A
    COUNT(*) gets the COUNT deadline and a one-request download; a row query the row deadlines."""
    if is_count_query(adql):
        kw.setdefault("exec_deadline_s", TAP_COUNT_EXEC_DEADLINE_S)
        kw.setdefault("download_deadline_s", TAP_HTTP_TIMEOUT_S)
        what = f"{what} COUNT"
    else:
        kw.setdefault("exec_deadline_s", TAP_ROWS_EXEC_DEADLINE_S)
        kw.setdefault("download_deadline_s", TAP_ROWS_DOWNLOAD_DEADLINE_S)
        what = f"{what} rows"
    table = run_tap_job(adql, what=what, report=report, **kw)
    return pd.DataFrame() if table is None or len(table) == 0 else table.to_pandas()


def run_sync_query(
    url: str,
    params: dict[str, str],
    *,
    what: str = "Gaia TAP sync",
    deadline_s: float = TAP_COUNT_EXEC_DEADLINE_S,
    socket_timeout_s: float = TAP_HTTP_TIMEOUT_S,
    opener: Callable[..., Any] = urllib.request.urlopen,
    call: Callable[[Callable[[], Any], float, str], Any] = call_with_timeout,
) -> str:
    """One synchronous TAP query (POST) -> response text, bounded by ``deadline_s`` in total and by
    ``socket_timeout_s`` per blocking read. A sync query has no job to abort: the server ends it
    when the connection closes (the socket timeout closes it). Raises :class:`TapTimeout`."""
    body = urllib.parse.urlencode(params).encode()

    def _run() -> str:
        with opener(urllib.request.Request(url, data=body), timeout=float(socket_timeout_s)) as r:
            return r.read().decode("utf-8")

    t0 = time.monotonic()
    try:
        return call(_run, float(deadline_s), what)
    except TimeoutError as exc:
        el = time.monotonic() - t0
        msg = f"{what}: no answer from the Gaia archive after {fmt_duration(el)} ({exc})"
        LOGGER.warning(msg)
        raise TapTimeout(msg, what=what, phase="SYNC", elapsed_s=el, deadline_s=deadline_s) from exc
