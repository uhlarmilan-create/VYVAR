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

import math
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from net_call import call_with_timeout, mark_unavailable, unavailable_reason
from catalog_db import add_catalog_meta_rows, connect_rw, ra_intervals_deg, resolve_catalog_path

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
    try:
        from infolog import log_event  # noqa: PLC0415

        log_event(msg)
    except Exception:  # noqa: BLE001 - logging must never break a catalogue operation
        pass


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
    """Coverage rows of the DB; registers an older DB on first use (in memory when read-only).

    An empty ``gaia_coverage`` table is a build by ``build_gaia_catalog.py`` that has not finished
    (it records its box only after the last strip): no coverage is claimed for it.
    """
    p = _db_path(db_path)
    if not p.is_file():
        raise FileNotFoundError(f"Gaia DB not found: {p}")
    conn = connect_rw(p)
    try:
        has = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='gaia_coverage'"
        ).fetchone()
        if has:
            return _coverage_rows(conn)
        rec = _register_existing(conn)
        if rec is None:
            return []
        try:
            _create_coverage_table(conn)
            _insert_coverage(conn, rec)
            conn.commit()
            _log(
                f"GAIA COVERAGE: {p.name} registered as RA 0-360, Dec {rec['dec_min']:.2f}..{rec['dec_max']:.2f}, "
                f"G <= {rec['g_max']:.2f} ({rec['n_stars']} stars)"
            )
        except sqlite3.Error as exc:
            _log(f"GAIA COVERAGE: {p.name} is not writable ({exc}); coverage used from its contents only")
        return [rec]
    finally:
        conn.close()


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
    (a field outside the Dec range of the DB: TODO-GAIA-COVERAGE-DEC)."""
    best: float | None = None
    for rec in load_coverage(db_path):
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
def build_topup_adql(box: tuple[float, float, float, float], *, g_from: float, g_to: float) -> str:
    """ADQL for the stars of ``box`` with g_from < G <= g_to (``gaiadr3.gaia_source``)."""
    ra_terms = " OR ".join(
        f"(ra >= {a:.8f} AND ra <= {b:.8f})" for a, b in ra_intervals_deg(box[0], box[1])
    )
    return (
        f"SELECT {GAIA_SOURCE_SELECT}\n        FROM gaiadr3.gaia_source\n"
        f"        WHERE ({ra_terms})\n"
        f"          AND dec >= {box[2]:.8f} AND dec <= {box[3]:.8f}\n"
        f"          AND phot_g_mean_mag > {float(g_from):.4f}\n"
        f"          AND phot_g_mean_mag <= {float(g_to):.4f}"
    )


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
    adql = build_topup_adql(box, g_from=g_from, g_to=g_to)
    t0 = time.monotonic()
    try:
        df = (fetch or _fetch_tap)(adql)
    except Exception as exc:  # noqa: BLE001 - offline / archive down: continue with the local DB
        if fetch is None:
            mark_unavailable("gaia_tap_topup", exc)
        out.update({"status": "offline", "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
        return out
    df = normalize_tap_dataframe(df)
    try:
        conn = connect_rw(_db_path(db_path))
        try:
            init_gaia_tables(conn)
            n_new = insert_gaia_rows(conn, df)
            add_catalog_meta_rows(conn, "gaia_dr3", n_new)
            conn.commit()
        finally:
            conn.close()
        record_coverage(db_path, box=box, g_max=float(g_to), source="gaia_tap_topup", n_stars=int(len(df)))
    except sqlite3.Error as exc:  # read-only / locked main DB: the run continues with what it has
        out.update({"status": "db_not_writable", "n_fetched": int(len(df)),
                    "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
        return out
    out.update({
        "status": "ok", "n_fetched": int(len(df)), "n_new": int(n_new),
        "seconds": round(time.monotonic() - t0, 1),
    })
    return out


def _coverage_table_stored(db_path: str | Path) -> bool:
    load_coverage(db_path)  # registers an older DB
    try:
        conn = connect_rw(_db_path(db_path))
        try:
            return conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='gaia_coverage'"
            ).fetchone() is not None
        finally:
            conn.close()
    except sqlite3.Error:
        return False


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
