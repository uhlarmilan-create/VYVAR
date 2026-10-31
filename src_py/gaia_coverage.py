"""Main Gaia DB: schema, coverage register and online top-up of one field (FRAME-DEPTH-2026-10 step 2).

Milan 2026-10-14: the user builds the main Gaia DB at install with his own G and Dec limits
(``catalogs/build_gaia_catalog.py``). When a MASTERSTAR reaches fainter than the DB covers in its
field, VYVAR fetches the missing stars online for that field (+-1 deg around the field edges) down
to the frame limit and writes them into the main DB. Offline, the run continues with the local DB
and the limitation is logged and reported.

``gaia_coverage`` records which sky box is complete to which G:

* a DB built before this table existed is registered on first use as RA 0-360,
  Dec MIN..MAX(dec) of the stored stars, G <= MAX(g_mag) (Milan: "b) yes, I agree");
* the builder records the box and G limit it downloaded;
* every top-up records its field box and G limit.

The depth of a field is the deepest row whose box contains the whole field box. One depth for the
whole pipeline: the frame limit (``frame_depth.py``) capped by this coverage depth.
"""
from __future__ import annotations

import logging
import math
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from net_call import call_with_timeout, mark_unavailable, unavailable_reason
from catalog_db import add_catalog_meta_rows, connect_ro, connect_rw, ra_intervals_deg, resolve_catalog_path

LOGGER = logging.getLogger(__name__)

#: Agreed margin around the field edges for a top-up (Milan 2026-10-14: "+- 1 degree around the field edges").
TOPUP_MARGIN_DEG = 1.0
_TAP_TIMEOUT_S = 600.0
_INSERT_BATCH = 8000
_EPS_DEG = 1e-6

GAIA_ROW_COLUMNS: tuple[str, ...] = (
    "source_id",
    "ra",
    "dec",
    "g_mag",
    "bp_mag",
    "rp_mag",
    "bp_rp",
    "g_flux_error_rel",
    "parallax",
    "parallax_error",
    "parallax_over_error",
    "teff_gspphot",
    "logg_gspphot",
    "mh_gspphot",
    "distance_gspphot",
    "var_flag",
    "non_single_star",
)

#: SELECT list of ``gaiadr3.gaia_source`` mapped to the ``gaia_dr3`` columns (builder and top-up).
GAIA_SOURCE_SELECT = """
            source_id,
            ra,
            dec,
            phot_g_mean_mag AS g_mag,
            phot_bp_mean_mag AS bp_mag,
            phot_rp_mean_mag AS rp_mag,
            bp_rp,
            (phot_g_mean_flux_error / NULLIF(phot_g_mean_flux, 0.0)) AS g_flux_error_rel,
            parallax,
            parallax_error,
            parallax_over_error,
            teff_gspphot,
            logg_gspphot,
            mh_gspphot,
            distance_gspphot,
            phot_variable_flag AS var_flag,
            non_single_star"""
#: SELECT list of ``gaiadr3.gaia_source_lite`` (no flux error / astrophysical parameters).
GAIA_SOURCE_LITE_SELECT = """
            source_id,
            ra,
            dec,
            phot_g_mean_mag AS g_mag,
            phot_bp_mean_mag AS bp_mag,
            phot_rp_mean_mag AS rp_mag,
            bp_rp,
            parallax,
            parallax_error,
            parallax_over_error,
            phot_variable_flag AS var_flag,
            non_single_star"""



def _db_path(db_path: str | Path) -> Path:
    """The DB path as every catalogue reader resolves it (``~``, relative to the data root)."""
    return resolve_catalog_path(db_path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log(msg: str) -> None:
    """Run log line; G04-45: a failing infolog is reported (WARNING), never swallowed."""
    try:
        from infolog import log_event  # noqa: PLC0415

        log_event(msg)
    except Exception as exc:  # noqa: BLE001 - logging must never break a catalogue operation
        LOGGER.warning("infolog failed (%s); message was: %s", exc, msg)


# ------------------------------------------------------------------ complete TAP downloads (G05-03, NEW-2)
#: G05-03 / NEW-2 guard against a TAP server whose COUNT(*) and returned rows never agree: a box is
#: split into four at most this many times (4**6 = 4096 sub-boxes of one strip). It is not a row
#: cap and decides nothing about completeness - the COUNT(*) equality does.
MAX_SPLIT_DEPTH = 6

#: The ESA archive's documented row limit of an anonymous async job (archive FAQ). Used ONLY to
#: label strips of a build made before the COUNT check (report / warning); never a decision.
ESA_ANON_ASYNC_ROW_CAP = 3_000_000


class IncompleteDownload(RuntimeError):
    """The rows of a box never matched its COUNT(*) (more rows than counted, or the split guard)."""


@dataclass(frozen=True)
class SkyBox:
    """RA/Dec box; lower edges inclusive, upper edges inclusive when ``*_closed`` (else open)."""

    ra0: float
    ra1: float
    de0: float
    de1: float
    ra_closed: bool = True
    de_closed: bool = True

    def where(self) -> str:
        ra_op = "<=" if self.ra_closed else "<"
        de_op = "<=" if self.de_closed else "<"
        return (f"ra >= {self.ra0:.8f} AND ra {ra_op} {self.ra1:.8f} "
                f"AND dec >= {self.de0:.8f} AND dec {de_op} {self.de1:.8f}")

    def split4(self) -> list[SkyBox]:
        """Four sub-boxes (RA and Dec halves); the inner edges are half-open, so every star of the
        box is in exactly one sub-box."""
        rm, dm = 0.5 * (self.ra0 + self.ra1), 0.5 * (self.de0 + self.de1)
        out = []
        for d0, d1, dc in ((self.de0, dm, False), (dm, self.de1, self.de_closed)):
            for r0, r1, rc in ((self.ra0, rm, False), (rm, self.ra1, self.ra_closed)):
                out.append(SkyBox(r0, r1, d0, d1, rc, dc))
        return out


def region_where(region: list[SkyBox]) -> str:
    return " OR ".join(f"({b.where()})" for b in region)


def region_adql(region: list[SkyBox], *, select: str, table: str, mag_where: str) -> str:
    return f"SELECT {select}\n        FROM {table}\n        WHERE ({region_where(region)})\n          AND {mag_where}"


def region_count_adql(region: list[SkyBox], *, table: str, mag_where: str) -> str:
    """COUNT(*) of exactly the rows :func:`region_adql` selects (same WHERE)."""
    return f"SELECT COUNT(*) AS n\n        FROM {table}\n        WHERE ({region_where(region)})\n          AND {mag_where}"


def _tap_count(fetch: Callable[[str], pd.DataFrame], adql: str) -> int:
    df = fetch(adql)
    if df is None or len(df) != 1 or len(df.columns) != 1:
        raise IncompleteDownload(f"COUNT(*) query returned no single value: {None if df is None else df.shape}")
    return int(df.iloc[0, 0])


def fetch_region_complete(
    fetch: Callable[[str], pd.DataFrame],
    region: list[SkyBox],
    *,
    select: str,
    table: str,
    mag_where: str,
    on_rows: Callable[[pd.DataFrame], None],
    depth: int = 0,
) -> tuple[int, int]:
    """G05-03 / NEW-2: download ``region`` completely. A box is complete only when the rows
    received equal the TAP ``COUNT(*)`` of the same query; otherwise each box is split into four
    and fetched again (recursively). ``on_rows`` gets every verified part (normalised frame).
    Returns (COUNT(*), rows delivered); raises :class:`IncompleteDownload` when it cannot be made
    complete. Network errors of ``fetch`` propagate."""
    n_expected = _tap_count(fetch, region_count_adql(region, table=table, mag_where=mag_where))
    if n_expected == 0:
        return 0, 0
    df = normalize_tap_dataframe(fetch(region_adql(region, select=select, table=table, mag_where=mag_where)))
    if len(df) == n_expected:
        on_rows(df)
        return n_expected, len(df)
    if len(df) > n_expected:
        raise IncompleteDownload(f"{len(df)} rows for COUNT(*) {n_expected} ({region_where(region)})")
    if depth >= MAX_SPLIT_DEPTH:
        raise IncompleteDownload(
            f"{len(df)} of {n_expected} rows after {depth} splits ({region_where(region)})")
    LOGGER.warning("Gaia TAP: %d of %d rows (truncated) - box split in 4 (level %d)", len(df), n_expected, depth + 1)
    del df
    got_e = got_r = 0
    for box in region:
        for sub in box.split4():
            e, r = fetch_region_complete(fetch, [sub], select=select, table=table, mag_where=mag_where,
                                         on_rows=on_rows, depth=depth + 1)
            got_e += e
            got_r += r
    if got_e != n_expected:
        raise IncompleteDownload(f"sub-box counts {got_e} != COUNT(*) {n_expected} ({region_where(region)})")
    return n_expected, got_r


# ------------------------------------------------------------------ strip completeness (G05-03)
def ensure_strip_columns(conn: sqlite3.Connection) -> None:
    """strip_progress keeps the G limit (mag_limit), the TAP COUNT(*) of the strip query
    (n_expected) and ``verified``: 1 = rows equal the COUNT, 0 = known truncated, NULL = a strip of
    a build before the COUNT check (never checked)."""
    cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(strip_progress)").fetchall()}
    for name, decl in (("mag_limit", "REAL"), ("n_expected", "INTEGER"), ("verified", "INTEGER")):
        if name not in cols:
            conn.execute(f"ALTER TABLE strip_progress ADD COLUMN {name} {decl}")


def strips_needing_repair(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """G05-03 detection: strips of ``strip_progress`` whose completeness is not verified - known
    truncated (verified 0) or never COUNT-checked (verified NULL, every strip of an older build).
    Largest first; ``at_anonymous_cap`` labels a stored count equal to the documented ESA cap."""
    if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='strip_progress'").fetchone() is None:
        return []
    cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(strip_progress)").fetchall()}
    sel = ", ".join(c if c in cols else f"NULL AS {c}" for c in ("mag_limit", "n_expected", "verified"))
    rows = conn.execute(
        f"SELECT strip_key, ra_min, ra_max, dec_min, dec_max, n_stars, {sel} FROM strip_progress "
        "ORDER BY n_stars DESC, strip_key"
    ).fetchall()
    out = []
    for key, r0, r1, d0, d1, n, ml, ne, ver in rows:
        if ver is not None and int(ver) == 1:
            continue
        out.append({
            "strip_key": str(key), "box": (float(r0), float(r1), float(d0), float(d1)),
            "n_stars": None if n is None else int(n), "mag_limit": ml, "n_expected": ne,
            "state": "truncated" if ver is not None else "never_checked",
            "at_anonymous_cap": n is not None and int(n) == ESA_ANON_ASYNC_ROW_CAP,
        })
    return out


def _truncated_strip_boxes(conn: sqlite3.Connection) -> list[tuple[float, float, float, float]]:
    if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='strip_progress'").fetchone() is None:
        return []
    cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(strip_progress)").fetchall()}
    if "verified" not in cols:
        return []
    return [tuple(float(v) for v in r) for r in conn.execute(
        "SELECT ra_min, ra_max, dec_min, dec_max FROM strip_progress WHERE verified = 0").fetchall()]


def _boxes_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    if a[3] <= b[2] or b[3] <= a[2]:
        return False
    return any(x0 < y1 and y0 < x1 for x0, x1 in ra_intervals_deg(a[0], a[1]) for y0, y1 in ra_intervals_deg(b[0], b[1]))


# ------------------------------------------------------------------ schema / insert (shared with the builder)
def init_gaia_tables(conn: sqlite3.Connection) -> None:
    """Create ``gaia_dr3``, ``strip_progress`` (builder resume) and ``gaia_coverage`` if absent."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS gaia_dr3 (
            source_id           INTEGER PRIMARY KEY,
            ra                  REAL,
            dec                 REAL,
            g_mag               REAL,
            bp_mag              REAL,
            rp_mag              REAL,
            bp_rp               REAL,
            g_flux_error_rel    REAL,
            parallax            REAL,
            parallax_error      REAL,
            parallax_over_error REAL,
            teff_gspphot        REAL,
            logg_gspphot        REAL,
            mh_gspphot          REAL,
            distance_gspphot    REAL,
            var_flag            TEXT,
            non_single_star     INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS strip_progress (
            strip_key   TEXT PRIMARY KEY,
            ra_min      REAL,
            ra_max      REAL,
            dec_min     REAL,
            dec_max     REAL,
            n_stars     INTEGER,
            finished_at TEXT
        )
        """
    )
    _create_coverage_table(conn)
    conn.commit()


def _create_coverage_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS gaia_coverage (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ra_min      REAL NOT NULL,
            ra_max      REAL NOT NULL,
            dec_min     REAL NOT NULL,
            dec_max     REAL NOT NULL,
            g_max       REAL NOT NULL,
            source      TEXT NOT NULL,
            n_stars     INTEGER,
            created_utc TEXT NOT NULL
        )
        """
    )


def normalize_tap_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """ADQL result -> exactly the ``gaia_dr3`` columns (lower case; absent columns as NULL)."""
    if df is None or df.empty:
        return pd.DataFrame(columns=list(GAIA_ROW_COLUMNS))
    out = df.rename(columns={c: str(c).strip().lower() for c in df.columns})
    for c in GAIA_ROW_COLUMNS:
        if c not in out.columns:
            out[c] = None if c == "var_flag" else float("nan")
    return out[list(GAIA_ROW_COLUMNS)].copy()


def _sql_value(col: str, v: Any) -> Any:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if col == "source_id":
        from gaia_catalog_id import normalize_gaia_source_id  # noqa: PLC0415

        s = normalize_gaia_source_id(v)
        return int(s) if s and s.isdigit() else None
    if col == "var_flag":
        s = str(v).strip()
        return s if s and s.lower() != "nan" else None
    if col == "non_single_star":
        try:
            return int(v)
        except (TypeError, ValueError):
            return None
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return None
    return fv if math.isfinite(fv) else None


def insert_gaia_rows(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """INSERT OR IGNORE the rows of a normalised TAP frame; returns the number of new stars."""
    df = normalize_tap_dataframe(df)
    if df.empty:
        return 0
    rows = [
        tuple(_sql_value(c, v) for c, v in zip(GAIA_ROW_COLUMNS, rec, strict=True))
        for rec in df.itertuples(index=False, name=None)
    ]
    rows = [r for r in rows if r[0] is not None]
    if not rows:
        return 0
    cols = ", ".join(GAIA_ROW_COLUMNS)
    ph = ", ".join("?" * len(GAIA_ROW_COLUMNS))
    before = conn.total_changes
    for i in range(0, len(rows), _INSERT_BATCH):
        conn.executemany(f"INSERT OR IGNORE INTO gaia_dr3 ({cols}) VALUES ({ph})", rows[i : i + _INSERT_BATCH])
    return int(conn.total_changes - before)


# ------------------------------------------------------------------ boxes
def field_box(center: Any, radius_deg: float, *, margin_deg: float = 0.0) -> tuple[float, float, float, float]:
    """Sky box (ra_min, ra_max, dec_min, dec_max) of the field cone grown by ``margin_deg``."""
    from catalog_db import cone_sky_box  # noqa: PLC0415

    return cone_sky_box(center, float(radius_deg) + max(0.0, float(margin_deg)))


def _box_inside(
    inner: tuple[float, float, float, float], outer: tuple[float, float, float, float]
) -> bool:
    ira0, ira1, ide0, ide1 = inner
    ora0, ora1, ode0, ode1 = outer
    if ide0 < ode0 - _EPS_DEG or ide1 > ode1 + _EPS_DEG:
        return False
    outer_iv = ra_intervals_deg(ora0, ora1)
    for a, b in ra_intervals_deg(ira0, ira1):
        if not any(c - _EPS_DEG <= a and b <= d + _EPS_DEG for c, d in outer_iv):
            return False
    return True


# ------------------------------------------------------------------ coverage
def _coverage_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    cur = conn.execute(
        "SELECT ra_min, ra_max, dec_min, dec_max, g_max, source, n_stars, created_utc "
        "FROM gaia_coverage ORDER BY id"
    )
    keys = ("ra_min", "ra_max", "dec_min", "dec_max", "g_max", "source", "n_stars", "created_utc")
    return [dict(zip(keys, r, strict=True)) for r in cur.fetchall()]


def _register_existing(conn: sqlite3.Connection) -> dict[str, Any] | None:
    """A DB built before the coverage record: all RA, Dec and G as far as its stars reach
    (a Dec extreme within 0.1 deg of a pole counts as the pole)."""
    row = conn.execute(
        "SELECT MIN(dec), MAX(dec), MAX(g_mag), COUNT(*) FROM gaia_dr3 WHERE g_mag IS NOT NULL"
    ).fetchone()
    if row is None or row[2] is None or not int(row[3] or 0):
        return None
    de0 = -90.0 if float(row[0]) < -89.9 else float(row[0])
    de1 = 90.0 if float(row[1]) > 89.9 else float(row[1])
    rec = {
        "ra_min": 0.0, "ra_max": 360.0,
        "dec_min": de0, "dec_max": de1, "g_max": float(row[2]),
        "source": "registered_existing_db", "n_stars": int(row[3]), "created_utc": _utc_now(),
    }
    return rec


def _insert_coverage(conn: sqlite3.Connection, rec: dict[str, Any]) -> None:
    conn.execute(
        "INSERT INTO gaia_coverage (ra_min, ra_max, dec_min, dec_max, g_max, source, n_stars, created_utc) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (rec["ra_min"], rec["ra_max"], rec["dec_min"], rec["dec_max"], rec["g_max"], rec["source"],
         rec.get("n_stars"), rec["created_utc"]),
    )


def load_coverage(db_path: str | Path) -> list[dict[str, Any]]:
    """Coverage rows of the DB (read-only, G04-45). A DB built before the coverage record has its
    registration computed in memory (:func:`register_existing_db` stores it).

    An empty ``gaia_coverage`` table is a build by ``build_gaia_catalog.py`` that has not finished
    (it records its box only after the last strip): no coverage is claimed for it.
    """
    p = _db_path(db_path)
    if not p.is_file():
        raise FileNotFoundError(f"Gaia DB not found: {p}")
    conn = connect_ro(p)
    try:
        if _has_coverage_table(conn):
            return _coverage_rows(conn)
        rec = _register_existing(conn)
        return [] if rec is None else [rec]
    finally:
        conn.close()


def _has_coverage_table(conn: sqlite3.Connection) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='gaia_coverage'"
    ).fetchone() is not None


def coverage_is_stored(db_path: str | Path) -> bool:
    """True when the DB holds a ``gaia_coverage`` table (registered or built with it)."""
    conn = connect_ro(_db_path(db_path))
    try:
        return _has_coverage_table(conn)
    finally:
        conn.close()


def register_existing_db(db_path: str | Path) -> dict[str, Any] | None:
    """G04-45: store the coverage registration of a DB built before the coverage record (explicit
    write; ``gaia_maintain.py --register`` and the online top-up). Returns the stored row, None
    when the DB already has a coverage table or holds no stars. Raises sqlite3.Error when the DB
    is not writable."""
    p = _db_path(db_path)
    conn = connect_rw(p)
    try:
        if _has_coverage_table(conn):
            return None
        rec = _register_existing(conn)
        if rec is None:
            return None
        _create_coverage_table(conn)
        _insert_coverage(conn, rec)
        conn.commit()
    finally:
        conn.close()
    _log(
        f"GAIA COVERAGE: {p.name} registered as RA 0-360, Dec {rec['dec_min']:.2f}..{rec['dec_max']:.2f}, "
        f"G <= {rec['g_max']:.2f} ({rec['n_stars']} stars)"
    )
    return rec


def record_coverage(
    db_path: str | Path,
    *,
    box: tuple[float, float, float, float],
    g_max: float,
    source: str,
    n_stars: int | None = None,
) -> None:
    """Add one coverage row (the builder after a complete build, a top-up after its insert)."""
    conn = connect_rw(_db_path(db_path))
    try:
        _create_coverage_table(conn)
        _insert_coverage(conn, {
            "ra_min": float(box[0]), "ra_max": float(box[1]), "dec_min": float(box[2]), "dec_max": float(box[3]),
            "g_max": float(g_max), "source": str(source), "n_stars": n_stars, "created_utc": _utc_now(),
        })
        conn.commit()
    finally:
        conn.close()


def covered_depth_g(db_path: str | Path, box: tuple[float, float, float, float]) -> float | None:
    """Faintest G to which the DB is complete over the whole ``box``; None when no record covers it
    (a field outside the Dec range of the DB: TODO-GAIA-COVERAGE-DEC).

    G05-03: a box touching a strip the COUNT check found truncated (strip_progress verified 0) is
    not covered by the whole-DB records (build / registration); only a top-up record (complete by
    its own COUNT check) can cover it."""
    conn = connect_ro(_db_path(db_path))
    try:
        holes = [h for h in _truncated_strip_boxes(conn) if _boxes_overlap(box, h)]
    finally:
        conn.close()
    best: float | None = None
    for rec in load_coverage(db_path):
        if holes and rec["source"] != "gaia_tap_topup":
            continue
        outer = (rec["ra_min"], rec["ra_max"], rec["dec_min"], rec["dec_max"])
        if _box_inside(box, outer):
            g = float(rec["g_max"])
            if best is None or g > best:
                best = g
    return best


def coverage_summary(db_path: str | Path) -> list[dict[str, Any]]:
    """Coverage rows for provenance / reports (rounded)."""
    return [
        {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}
        for rec in load_coverage(db_path)
    ]


# ------------------------------------------------------------------ online top-up
def _topup_region(box: tuple[float, float, float, float]) -> list[SkyBox]:
    return [SkyBox(a, b, box[2], box[3]) for a, b in ra_intervals_deg(box[0], box[1])]


def _topup_mag_where(g_from: float, g_to: float) -> str:
    return f"phot_g_mean_mag > {float(g_from):.4f}\n          AND phot_g_mean_mag <= {float(g_to):.4f}"


def build_topup_adql(box: tuple[float, float, float, float], *, g_from: float, g_to: float) -> str:
    """ADQL for the stars of ``box`` with g_from < G <= g_to (``gaiadr3.gaia_source``)."""
    return region_adql(_topup_region(box), select=GAIA_SOURCE_SELECT, table="gaiadr3.gaia_source",
                       mag_where=_topup_mag_where(g_from, g_to))


def _fetch_tap(adql: str) -> pd.DataFrame:
    """Run one ADQL job on the Gaia archive (astroquery), bounded by a timeout."""

    def _run() -> pd.DataFrame:
        from astroquery.gaia import Gaia  # noqa: PLC0415

        table = Gaia.launch_job_async(adql).get_results()
        return pd.DataFrame() if table is None or len(table) == 0 else table.to_pandas()

    return call_with_timeout(_run, _TAP_TIMEOUT_S, "Gaia TAP top-up")


def top_up_field(
    db_path: str | Path,
    *,
    box: tuple[float, float, float, float],
    g_from: float,
    g_to: float,
    fetch: Callable[[str], pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Fetch the field box from the Gaia archive for g_from < G <= g_to into the main DB.

    Fail-open: without the archive the result says why (``status='offline'``) and nothing changes.
    Coverage is recorded only after every row is stored, so a broken download never claims depth.
    NEW-2: the download is complete only when its rows equal the TAP COUNT(*) of the same query
    (split in four and retried otherwise, as the builder does); when it cannot be made complete the
    verified rows are kept but no coverage is recorded (``status='incomplete'``, WARNING).
    """
    out: dict[str, Any] = {
        "status": "skipped", "g_from": round(float(g_from), 3), "g_to": round(float(g_to), 3),
        "box": [round(float(v), 5) for v in box], "n_fetched": 0, "n_new": 0, "error": None,
    }
    if not (math.isfinite(float(g_from)) and math.isfinite(float(g_to))) or float(g_to) <= float(g_from):
        out["error"] = "nothing to fetch"
        return out
    _down = unavailable_reason("gaia_tap_topup")
    if fetch is None and _down is not None:
        out.update({"status": "offline", "error": f"Gaia archive unavailable a moment ago ({_down})"})
        return out
    if not _coverage_table_stored(db_path):
        # The record of the whole DB must be stored before deeper stars go in (MAX(g_mag) would
        # then claim the new depth for the whole sky); a DB that cannot take it is not extended.
        out.update({"status": "db_not_writable", "error": "coverage record of the DB cannot be written"})
        return out
    t0 = time.monotonic()
    parts: list[pd.DataFrame] = []
    complete_error: str | None = None
    try:
        fetch_region_complete(fetch or _fetch_tap, _topup_region(box), select=GAIA_SOURCE_SELECT,
                              table="gaiadr3.gaia_source", mag_where=_topup_mag_where(g_from, g_to),
                              on_rows=parts.append)
    except IncompleteDownload as exc:
        complete_error = str(exc)
    except Exception as exc:  # noqa: BLE001 - offline / archive down: continue with the local DB
        if fetch is None:
            mark_unavailable("gaia_tap_topup", exc)
        out.update({"status": "offline", "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
        return out
    df = normalize_tap_dataframe(pd.concat(parts, ignore_index=True)) if parts else normalize_tap_dataframe(None)
    try:
        conn = connect_rw(_db_path(db_path))
        try:
            init_gaia_tables(conn)
            n_new = insert_gaia_rows(conn, df)
            add_catalog_meta_rows(conn, "gaia_dr3", n_new)
            conn.commit()
        finally:
            conn.close()
        if complete_error is None:
            record_coverage(db_path, box=box, g_max=float(g_to), source="gaia_tap_topup", n_stars=int(len(df)))
    except sqlite3.Error as exc:  # read-only / locked main DB: the run continues with what it has
        out.update({"status": "db_not_writable", "n_fetched": int(len(df)),
                    "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
        return out
    out.update({
        "status": "ok" if complete_error is None else "incomplete", "n_fetched": int(len(df)),
        "n_new": int(n_new), "seconds": round(time.monotonic() - t0, 1),
    })
    if complete_error is not None:
        out["error"] = f"download not complete (COUNT check): {complete_error[:300]}"
        LOGGER.warning("Gaia top-up incomplete, no coverage recorded: %s", complete_error)
        _log(f"WARNING: GAIA TOP-UP incomplete ({complete_error[:300]}); {len(df)} verified rows kept, "
             "no coverage recorded")
    return out


def _coverage_table_stored(db_path: str | Path) -> bool:
    """The DB has its coverage table, registering an older DB first (G04-45: one explicit write)."""
    try:
        register_existing_db(db_path)
        return coverage_is_stored(db_path)
    except sqlite3.Error:
        return False


def _warn_unverified_strips(db_path: str | Path, box: tuple[float, float, float, float]) -> None:
    """G05-03: say so when the field lies in a strip of an older build that is known truncated or
    whose stored count equals the anonymous TAP row cap (repair: build_gaia_catalog.py --repair)."""
    conn = connect_ro(_db_path(db_path))
    try:
        bad = [s for s in strips_needing_repair(conn)
               if (s["state"] == "truncated" or s["at_anonymous_cap"]) and _boxes_overlap(box, s["box"])]
    finally:
        conn.close()
    for s in bad:
        _log(f"WARNING: GAIA DB strip {s['strip_key']} ({s['n_stars']} rows, {s['state']}"
             f"{', at the anonymous TAP row cap' if s['at_anonymous_cap'] else ''}) overlaps this field: the "
             "local Gaia DB may miss stars here; run catalogs/build_gaia_catalog.py --repair")


def ensure_field_depth(
    db_path: str | Path,
    *,
    center: Any,
    radius_deg: float,
    g_needed: float,
    fetch: Callable[[str], pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Make the main DB complete to ``g_needed`` over the field (+ TOPUP_MARGIN_DEG) if possible.

    Returns ``{"status": "covered"|"ok"|"offline"|"no_coverage"|..., "depth_before", "depth_after", ...}``.
    """
    field = field_box(center, radius_deg)
    _warn_unverified_strips(db_path, field_box(center, radius_deg, margin_deg=TOPUP_MARGIN_DEG))
    before = covered_depth_g(db_path, field)
    res: dict[str, Any] = {"depth_before": before, "g_needed": round(float(g_needed), 3)}
    if before is None:
        # Field outside the Dec range of the DB (TODO-GAIA-COVERAGE-DEC): no base to extend.
        res.update({"status": "no_coverage", "depth_after": None})
        return res
    if float(g_needed) <= float(before) + 1e-3:
        res.update({"status": "covered", "depth_after": before})
        return res
    box = field_box(center, radius_deg, margin_deg=TOPUP_MARGIN_DEG)
    box_depth = covered_depth_g(db_path, box)
    g_from = float(box_depth) if box_depth is not None else float(before)
    if box_depth is None:
        # The grown box leaves the DB Dec range: fetch from the bottom so the box is complete.
        g_from = -10.0
    top = top_up_field(db_path, box=box, g_from=g_from, g_to=float(g_needed), fetch=fetch)
    res.update(top)
    res["depth_after"] = covered_depth_g(db_path, field)
    if top["status"] == "ok":
        _log(
            f"GAIA TOP-UP: field box RA {box[0]:.3f}..{box[1]:.3f} Dec {box[2]:.3f}..{box[3]:.3f}, "
            f"G {g_from:.2f}..{float(g_needed):.2f}: {top['n_fetched']} stars fetched, {top['n_new']} new "
            f"-> field depth G {res['depth_after']}"
        )
    else:
        _log(
            f"GAIA TOP-UP: not possible ({top.get('error')}); continuing with the local DB "
            f"complete to G {before:.2f} (frame reaches G {float(g_needed):.2f})"
        )
    return res
