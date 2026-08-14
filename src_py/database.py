"""SQLite database layer for the VYVAR project."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar, Iterable

from astropy.io import fits

from gaia_catalog_id import normalize_gaia_source_id
from infolog import log_event
from utils import normalize_telescope_focal_mm_for_plate_scale

_GAIA_INDEX_CHECK_DONE = False

# Resolved (db_path, mag_limit) -> (ra[], dec[], g_mag[]) for blind verify in-memory catalog.
_VERIFY_GAIA_BRIGHT_CACHE: dict[tuple[str, float], tuple[list[float], list[float], list[float]]] = {}

# Resolved DB path -> MAX(g_mag) from ``gaia_dr3`` (discovered at runtime).
_GAIA_DB_GMAG_MAX_CACHE: dict[str, float] = {}

LOGGER = logging.getLogger(__name__)

_SQLITE_CONNECT_TIMEOUT_S = 30.0


def open_sqlite_connection(
    db_path: str | Path,
    *,
    timeout: float = _SQLITE_CONNECT_TIMEOUT_S,
) -> sqlite3.Connection:
    """Open ``vyvar.sqlite3`` with WAL + busy timeout (Streamlit-safe concurrent access).

    ``check_same_thread=False`` allows a cached connection to be used across Streamlit
    rerun threads; :class:`ThreadSafeSQLiteConnection` serializes access with an RLock.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        str(path),
        timeout=float(timeout),
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute(f"PRAGMA busy_timeout = {int(float(timeout) * 1000)};")
    return conn


class _LockedCursor:
    """Hold the connection RLock until this cursor is fully consumed or closed."""

    __slots__ = ("_cursor", "_lock", "_released")

    def __init__(self, cursor: sqlite3.Cursor, lock: threading.RLock) -> None:
        self._cursor = cursor
        self._lock = lock
        self._released = False

    def _release(self) -> None:
        if not self._released:
            self._released = True
            self._lock.release()

    def fetchone(self) -> sqlite3.Row | None:
        try:
            row = self._cursor.fetchone()
            if row is None:
                self._release()
            return row
        except Exception:
            self._release()
            raise

    def fetchmany(self, size: int | None = None) -> list[sqlite3.Row]:
        try:
            if size is None:
                rows = self._cursor.fetchmany()
            else:
                rows = self._cursor.fetchmany(size)
            if not rows:
                self._release()
            return rows
        except Exception:
            self._release()
            raise

    def fetchall(self) -> list[sqlite3.Row]:
        try:
            return self._cursor.fetchall()
        finally:
            self._release()

    def __iter__(self) -> _LockedCursor:
        return self

    def __next__(self) -> sqlite3.Row:
        row = self.fetchone()
        if row is None:
            raise StopIteration
        return row

    def close(self) -> None:
        try:
            self._cursor.close()
        finally:
            self._release()

    @property
    def lastrowid(self) -> int:
        return int(self._cursor.lastrowid)

    @property
    def rowcount(self) -> int:
        return int(self._cursor.rowcount)

    @property
    def description(self):
        return self._cursor.description

    def __del__(self) -> None:
        self._release()


class ThreadSafeSQLiteConnection:
    """RLock-serialized wrapper around a shared ``vyvar.sqlite3`` connection.

    Rejected alternative: thread-local connections per ``VyvarDatabase`` -- would
    multiply migration/schema work and complicate commit visibility across threads.
    """

    __slots__ = ("_conn", "_lock")

    def __init__(self, conn: sqlite3.Connection, lock: threading.RLock | None = None) -> None:
        self._conn = conn
        self._lock = lock or threading.RLock()

    def execute(
        self,
        sql: str,
        parameters: Iterable[Any] = (),
    ) -> _LockedCursor:
        self._lock.acquire()
        try:
            cur = self._conn.execute(sql, parameters)
            return _LockedCursor(cur, self._lock)
        except Exception:
            self._lock.release()
            raise

    def executemany(self, sql: str, seq_of_parameters: Iterable[Iterable[Any]]) -> None:
        with self._lock:
            self._conn.executemany(sql, seq_of_parameters)

    def executescript(self, sql_script: str) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.executescript(sql_script)

    def commit(self) -> None:
        with self._lock:
            self._conn.commit()

    def rollback(self) -> None:
        with self._lock:
            self._conn.rollback()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

_EDITABLE_EDITOR_TABLES = frozenset({"EQUIPMENTS", "TELESCOPE", "LOCATION", "DRAFT_MANIFEST"})
_EDITABLE_DEFAULT_TABLES = frozenset({"EQUIPMENTS", "TELESCOPE", "LOCATION"})
_REBUILD_MIGRATION_TABLES = frozenset({
    "EQUIPMENTS",
    "TELESCOPE",
    "LOCATION",
    "OBS_QC_PROCESSING_RUN",
    "OBS_QC_PROCESSING_FILE",
})


def get_gaia_db_max_g_mag(db_path: str | Path) -> float:
    """Return the maximum ``g_mag`` stored in this Gaia SQLite (``SELECT MAX(g_mag) FROM gaia_dr3``).

    Cached per resolved path. If the table is empty or the query fails, returns ``0.0`` (caller should treat as
    'no photometric depth' / empty DB).
    """
    p = _resolve_catalog_db_path(db_path) if str(db_path or "").strip() else Path()
    if not p:
        raise GaiaCatalogError("Gaia local database path is empty.")
    key = str(p)
    if key in _GAIA_DB_GMAG_MAX_CACHE:
        return float(_GAIA_DB_GMAG_MAX_CACHE[key])
    if not p.is_file():
        raise FileNotFoundError(f"Gaia DB not found: {p}")
    out = 0.0
    con = sqlite3.connect(str(p))
    try:
        row = con.execute("SELECT MAX(g_mag) AS m FROM gaia_dr3 WHERE g_mag IS NOT NULL").fetchone()
        if row is not None and row[0] is not None:
            try:
                v = float(row[0])
                if math.isfinite(v) and v > 0:
                    out = float(v)
            except (TypeError, ValueError):
                out = 0.0
    except Exception:  # noqa: BLE001
        out = 0.0
        try:
            log_event(f"GAIA DB: MAX(g_mag) sa nepodarilo nacitat z {p.name} - predpokladam 0.0")
        except Exception:  # noqa: BLE001
            pass
    finally:
        con.close()
    _GAIA_DB_GMAG_MAX_CACHE[key] = out
    try:
        if out > 0:
            log_event(f"GAIA DB: MAX(g_mag) v gaia_dr3 = {out:.3f} ({p.name})")
        else:
            log_event(f"GAIA DB: MAX(g_mag) = 0 alebo prazdna tabulka ({p.name})")
    except Exception:  # noqa: BLE001
        pass
    return out


def load_verify_gaia_bright_stars(
    db_path: str | Path,
    *,
    mag_limit: float = 14.0,
) -> tuple[list[float], list[float], list[float]]:
    """Load all-sky Gaia stars brighter than ``mag_limit`` for in-memory blind verify.

    Returns ``(ra_deg, dec_deg, g_mag)`` sorted by ``g_mag`` ascending. Cached per DB path + limit.
    """
    p = _resolve_catalog_db_path(db_path)
    if not p.is_file():
        raise FileNotFoundError(f"Gaia DB not found: {p}")
    try:
        ml = float(mag_limit)
    except (TypeError, ValueError):
        ml = 14.0
    if not math.isfinite(ml) or ml <= 0:
        ml = 14.0
    _gmax_db = get_gaia_db_max_g_mag(p)
    if _gmax_db > 0.0 and ml > float(_gmax_db):
        ml = float(_gmax_db)
    key = (str(p), float(ml))
    if key in _VERIFY_GAIA_BRIGHT_CACHE:
        return _VERIFY_GAIA_BRIGHT_CACHE[key]
    t0 = __import__("time").monotonic()
    conn = sqlite3.connect(str(p))
    try:
        cur = conn.execute(
            "SELECT ra, dec, g_mag FROM gaia_dr3 "
            "WHERE g_mag <= ? AND ra IS NOT NULL AND dec IS NOT NULL "
            "ORDER BY g_mag ASC;",
            (float(ml),),
        )
        ra_l: list[float] = []
        dec_l: list[float] = []
        g_l: list[float] = []
        for row in cur.fetchall():
            try:
                ra_l.append(float(row[0]))
                dec_l.append(float(row[1]))
                g_l.append(float(row[2]))
            except (TypeError, ValueError, IndexError):
                continue
    finally:
        conn.close()
    _VERIFY_GAIA_BRIGHT_CACHE[key] = (ra_l, dec_l, g_l)
    elapsed = __import__("time").monotonic() - t0
    try:
        log_event(
            f"GAIA verify catalog: {len(ra_l)} stars (g<={float(ml):.1f}) "
            f"loaded in {elapsed:.2f}s ({p.name})"
        )
    except Exception:  # noqa: BLE001
        pass
    return ra_l, dec_l, g_l


def query_local_gaia(
    db_path: str | Path,
    *,
    ra_min: float,
    ra_max: float,
    dec_min: float,
    dec_max: float,
    mag_limit: float | None = None,
    max_rows: int | None = None,
) -> list[dict[str, Any]]:
    """Query local Gaia DR3 SQLite for a rectangular sky window (ICRS deg).

    The Gaia DB schema can evolve; this function auto-detects optional columns and returns them when present.
    """
    p = _resolve_catalog_db_path(db_path)
    if not p.is_file():
        raise FileNotFoundError(f"Gaia DB not found: {p}")

    mag_cap: float | None = None
    if mag_limit is not None:
        try:
            ml = float(mag_limit)
        except (TypeError, ValueError):
            ml = float("nan")
        if not math.isfinite(ml) or ml <= 0:
                        log_event(
                            f"GAIA SQL: invalid mag_limit={mag_limit!r} - no g_mag cap applied."
                        )
        else:
            _gmax_db = get_gaia_db_max_g_mag(p)
            if _gmax_db > 0.0 and ml > float(_gmax_db):
                try:
                    log_event(
                        f"GAIA SQL: mag_limit {float(ml):.2f} > MAX(g_mag) v DB ({float(_gmax_db):.3f}) - orezavam."
                    )
                except Exception:  # noqa: BLE001
                    # EXC-0060: T3 -- pure log_event guard (EXCEPT-BULK-2 2026-07-08)
                    pass
                ml = float(_gmax_db)
            mag_cap = float(ml)

    conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    global _GAIA_INDEX_CHECK_DONE
    try:
        conn.execute("PRAGMA automatic_index = ON;")
        try:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ra ON gaia_dr3 (ra);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_dec ON gaia_dr3 (dec);")
            conn.commit()
        except sqlite3.Error as exc:  # noqa: BLE001
            pass

        # Discover optional columns for forward-compatible queries.
        cur_cols = conn.execute("PRAGMA table_info('gaia_dr3');")
        cols = {str(r[1]).strip().lower() for r in cur_cols.fetchall()}
        if not _GAIA_INDEX_CHECK_DONE:
            try:
                idx_rows = conn.execute("PRAGMA index_list('gaia_dr3');").fetchall()
                idx_names = [str(r[1]) for r in idx_rows if len(r) > 1]
                has_ra_dec_idx = False
                for nm in idx_names:
                    row_sql = conn.execute(
                        "SELECT sql FROM sqlite_master WHERE type='index' AND name=? LIMIT 1;",
                        (nm,),
                    ).fetchone()
                    sql_txt = str(row_sql[0]).lower() if row_sql is not None and row_sql[0] is not None else ""
                    if "ra" in sql_txt and "dec" in sql_txt:
                        has_ra_dec_idx = True
                        break
                    nml = nm.lower()
                    if "ra" in nml and "dec" in nml:
                        has_ra_dec_idx = True
                        break
                if not has_ra_dec_idx:
                    log_event(
                        "GAIA DB: upozornenie - nenasiel sa zjavny index na (ra, dec); "
                        "dotazy mozu byt pomale."
                    )
            except sqlite3.Error as exc:  # noqa: BLE001
                pass
            _GAIA_INDEX_CHECK_DONE = True
        base_cols = ["source_id", "ra", "dec", "g_mag", "bp_mag", "rp_mag", "bp_rp", "var_flag"]
        opt_cols: list[str] = []
        for c in ("g_flux_error_rel", "non_single_star", "phot_variable_flag", "pmra", "pmdec"):
            if c in cols:
                opt_cols.append(c)
        # Backward/alternate schema: some DBs carry a string catalog id.
        if "catalog_id" in cols:
            opt_cols.append("catalog_id")
        sel = ", ".join(base_cols + opt_cols)
        mag_clause = f" AND g_mag <= {float(mag_cap)}" if mag_cap is not None else ""
        # Prefer ORDER BY+LIMIT when max_rows is set to avoid fetching huge boxes from full Gaia DB.
        lim = None
        if max_rows is not None:
            try:
                lim_i = int(max_rows)
                if lim_i > 0:
                    lim = lim_i
            except (TypeError, ValueError):
                lim = None
        if lim is not None:
            # With proper indexes, ORDER BY g_mag gives a stable "brightest-first" subset.
            query = (
                f"SELECT {sel} FROM gaia_dr3 "
                f"WHERE ra >= ? AND ra <= ? AND dec >= ? AND dec <= ?{mag_clause} "
                f"ORDER BY g_mag ASC LIMIT {int(lim)};"
            )
        else:
            query = (
                f"SELECT {sel} FROM gaia_dr3 "
                f"WHERE ra >= ? AND ra <= ? AND dec >= ? AND dec <= ?{mag_clause};"
            )
        cur = conn.execute(query, (float(ra_min), float(ra_max), float(dec_min), float(dec_max)))
        rows = [dict(r) for r in cur.fetchall()]
        try:
            if mag_cap is not None:
                log_event(f"GAIA SQL: Found {len(rows)} stars (Mag <= {float(mag_cap)})")
            else:
                log_event(f"GAIA SQL: Found {len(rows)} stars (no mag cap)")
        except Exception:  # noqa: BLE001
            # EXC-0063: T3 -- pure log_event guard (EXCEPT-BULK-2 2026-07-08)
            pass
        return rows
    finally:
        conn.close()


def _normalize_gaia_source_id_for_sql(v: Any) -> int | None:
    s = normalize_gaia_source_id(v)
    if not s or not s.isdigit():
        return None
    try:
        return int(s)
    except (TypeError, ValueError, OverflowError):
        return None


def query_local_gaia_by_source_ids(
    db_path: str | Path,
    source_ids: Iterable[Any],
    *,
    batch_size: int = 500,
) -> dict[str, dict[str, Any]]:
    """Fetch ``bp_rp`` / magnitudes for specific Gaia ``source_id`` rows (no sky box, no LIMIT sort).

    Used when a frame-wide Gaia query (rectangle + ``ORDER BY g_mag LIMIT``) omits stars that are
    already catalog-matched on the image - those stars still need ``bp_rp`` for photometry / color terms.
    """
    p = _resolve_catalog_db_path(db_path)
    if not p.is_file():
        raise FileNotFoundError(f"Gaia DB not found: {p}")
    try:
        bs = max(50, min(2000, int(batch_size)))
    except (TypeError, ValueError):
        bs = 500
    ids_u: list[int] = []
    seen: set[int] = set()
    for raw in source_ids:
        sid = _normalize_gaia_source_id_for_sql(raw)
        if sid is None or sid in seen:
            continue
        seen.add(sid)
        ids_u.append(sid)
    if not ids_u:
        return {}
    out: dict[str, dict[str, Any]] = {}
    conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    try:
        cur_cols = conn.execute("PRAGMA table_info('gaia_dr3');")
        cols = {str(r[1]).strip().lower() for r in cur_cols.fetchall()}
        sel_parts = ["source_id"]
        for c in ("bp_rp", "g_mag", "bp_mag", "rp_mag"):
            if c in cols:
                sel_parts.append(c)
        if "source_id" not in cols:
            return {}
        sel = ", ".join(sel_parts)
        for i0 in range(0, len(ids_u), bs):
            chunk = ids_u[i0 : i0 + bs]
            ph = ",".join("?" * len(chunk))
            q = f"SELECT {sel} FROM gaia_dr3 WHERE source_id IN ({ph});"
            for row in conn.execute(q, chunk):
                d = dict(row)
                sid0 = d.get("source_id")
                key = normalize_gaia_source_id(sid0)
                if not key:
                    continue
                bpr = d.get("bp_rp")
                if bpr is None and "bp_mag" in d and "rp_mag" in d:
                    try:
                        bpm = float(d["bp_mag"])
                        rpm = float(d["rp_mag"])
                        if math.isfinite(bpm) and math.isfinite(rpm):
                            bpr = bpm - rpm
                    except (TypeError, ValueError):
                        bpr = None
                out[key] = {
                    "bp_rp": float(bpr) if bpr is not None and math.isfinite(float(bpr)) else None,
                    "g_mag": float(d["g_mag"])
                    if d.get("g_mag") is not None and math.isfinite(float(d["g_mag"]))
                    else None,
                }
    finally:
        conn.close()
        log_event(f"GAIA SQL by source_id: fetched {len(out)} / {len(ids_u)} unique ids (batched).")
    return out


def validate_gaia_db_schema(db_path: str | Path) -> tuple[bool, str]:
    """Validate local Gaia DB has table/columns required by VYVAR."""
    p = _resolve_catalog_db_path(db_path)
    if not str(db_path).strip() or not p.is_file():
        return False, "missing_file"
    con = sqlite3.connect(str(p))
    try:
        cur = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='gaia_dr3' LIMIT 1;"
        )
        if cur.fetchone() is None:
            return False, "missing_table_gaia_dr3"
        cur2 = con.execute("PRAGMA table_info('gaia_dr3');")
        cols = {str(r[1]).strip().lower() for r in cur2.fetchall()}
        required = {"source_id", "ra", "dec", "g_mag", "bp_rp"}
        missing = sorted([c for c in required if c not in cols])
        if missing:
            return False, f"missing_columns:{','.join(missing)}"
        return True, "ok"
    finally:
        con.close()


def _location_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    """Map ``LOCATION`` SQL row to observer-location dict keys."""
    keys = set(row.keys())
    return {
        "id": int(row["ID"]),
        "name": str(row["PLACENAME"] or ""),
        "lat": float(row["LATITUDE"]) if row["LATITUDE"] is not None else 0.0,
        "lon": float(row["LONGITUDE"]) if row["LONGITUDE"] is not None else 0.0,
        "alt_m": float(row["ALTITUDE"]) if row["ALTITUDE"] is not None else 0.0,
        "is_default": int(row["IS_DEFAULT"] or 0) if "IS_DEFAULT" in keys else 0,
        "active": VyvarDatabase.normalize_active_db_value(row["ACTIVE"]) if "ACTIVE" in keys else 1,
    }


def get_observer_locations(db_path: str | Path, *, active_only: bool = False) -> list[dict[str, Any]]:
    """Return rows from the ``LOCATION`` table as list of dicts.

    Keys: ``id``, ``name``, ``lat``, ``lon``, ``alt_m``, ``is_default``, ``active``.
    Returns ``[]`` if the DB file is missing, the table is empty, or the query fails.
    The ``IS_DEFAULT``/``ACTIVE`` columns are read defensively (older DBs lack them).
    """
    p = Path(db_path).expanduser().resolve()
    if not p.is_file():
        return []
    conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    try:
        loc_cols = {r["name"] for r in conn.execute("PRAGMA table_info('LOCATION');").fetchall()}
        extra = ", ".join(c for c in ("IS_DEFAULT", "ACTIVE") if c in loc_cols)
        sel = "SELECT ID, PLACENAME, LATITUDE, LONGITUDE, ALTITUDE"
        if extra:
            sel += ", " + extra
        sql = f"{sel} FROM LOCATION"
        if active_only and "ACTIVE" in loc_cols:
            sql += f" WHERE {VyvarDatabase.sql_expr_active_is_true('ACTIVE')} "
        sql += " ORDER BY ID;"
        cur = conn.execute(sql)
        return [_location_row_to_dict(r) for r in cur.fetchall()]
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


def get_observer_location_by_id(db_path: str | Path, location_id: int) -> dict[str, Any] | None:
    """Return a single ``LOCATION`` row as dict, or ``None`` if not found."""
    try:
        lid = int(location_id)
    except (TypeError, ValueError):
        return None
    if lid <= 0:
        return None
    p = Path(db_path).expanduser().resolve()
    if not p.is_file():
        return None
    conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT ID, PLACENAME, LATITUDE, LONGITUDE, ALTITUDE FROM LOCATION WHERE ID = ? LIMIT 1;",
            (lid,),
        ).fetchone()
        if row is None:
            return None
        return _location_row_to_dict(row)
    except sqlite3.OperationalError:
        return None
    finally:
        conn.close()


def validate_vsx_local_db_schema(db_path: str | Path) -> tuple[bool, str]:
    """Validate local VSX subset SQLite (``vsx_data`` from VizieR B/vsx/vsx import)."""
    p = _resolve_catalog_db_path(db_path)
    if not str(db_path).strip() or not p.is_file():
        return False, "missing_file"
    con = sqlite3.connect(str(p))
    try:
        cur = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='vsx_data' LIMIT 1;"
        )
        if cur.fetchone() is None:
            return False, "missing_table_vsx_data"
        cur2 = con.execute("PRAGMA table_info('vsx_data');")
        cols = {str(r[1]).strip().lower() for r in cur2.fetchall()}
        required = {"oid", "ra_deg", "dec_deg"}
        missing = sorted([c for c in required if c not in cols])
        if missing:
            return False, f"missing_columns:{','.join(missing)}"
        return True, "ok"
    finally:
        con.close()


def count_vsx_local_rows(db_path: str | Path) -> int:
    """Return row count in ``vsx_data`` (0 when table missing or unreadable)."""
    p = _resolve_catalog_db_path(db_path)
    if not p.is_file():
        return 0
    con = sqlite3.connect(str(p))
    try:
        cur = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='vsx_data' LIMIT 1;"
        )
        if cur.fetchone() is None:
            return 0
        row = con.execute("SELECT COUNT(*) FROM vsx_data;").fetchone()
        return int(row[0] if row is not None else 0)
    except sqlite3.Error:
        return 0
    finally:
        con.close()


class VSXCatalogError(RuntimeError):
    """VSX local catalog path/schema/availability failure (fail-loud, like Gaia)."""


class ExoplanetCatalogError(RuntimeError):
    """Exoplanet local host DB path/schema/availability failure (fail-loud when configured)."""


class GaiaCatalogError(RuntimeError):
    """Gaia local DB path/schema/availability failure (fail-loud when configured)."""


def _resolve_catalog_db_path(raw: str | Path) -> Path:
    """Resolve catalog DB path using data_root (not CWD)."""
    from config import resolve_config_path, resolve_data_root  # noqa: PLC0415

    install_root = Path(__file__).resolve().parent.parent
    data_root = resolve_data_root(install_root)
    resolved = resolve_config_path(raw, data_root)
    if not resolved:
        return Path()
    return Path(resolved)


def require_gaia_db_path(gaia_db_path: str | Path | None) -> Path:
    """Resolve Gaia SQLite path or raise when photometry needs the local catalog."""
    raw = str(gaia_db_path or "").strip()
    if not raw:
        raise GaiaCatalogError(
            "Gaia local database path is not set (config key gaia_db_path). "
            "Open Settings -> Catalogs and set gaia_db_path to your Gaia DR3 SQLite."
        )
    p = _resolve_catalog_db_path(raw)
    if not p.is_file():
        raise GaiaCatalogError(
            f"Gaia local database file not found: {p}. "
            "Set gaia_db_path in Settings -> Catalogs."
        )
    return p


def require_exoplanet_local_db_path(exoplanet_local_db_path: str | Path | None) -> Path:
    """Resolve exoplanet SQLite path or raise when the feature is configured."""
    raw = str(exoplanet_local_db_path or "").strip()
    if not raw:
        raise ExoplanetCatalogError(
            "Exoplanet local database path is not set (config key exoplanet_local_db_path)."
        )
    p = _resolve_catalog_db_path(raw)
    if not p.is_file():
        raise ExoplanetCatalogError(
            f"Exoplanet local database file not found: {p}. "
            "Set exoplanet_local_db_path in Settings -> Catalogs to "
            "exoplanets/vyvar_exoplanet_local.db under the VYVAR data directory."
        )
    ok, code = validate_exoplanet_local_db_schema(p)
    if not ok:
        raise ExoplanetCatalogError(
            f"Exoplanet local database invalid ({code}): {p}. "
            "Rebuild the snapshot and verify table exoplanet_data."
        )
    return p


def require_vsx_local_db_path(vsx_local_db_path: str | Path | None) -> Path:
    """Resolve VSX SQLite path or raise with Settings-tab actionable message."""
    raw = str(vsx_local_db_path or "").strip()
    if not raw:
        raise VSXCatalogError(
            "VSX local database path is not set (config key vsx_local_db_path). "
            "Open Settings -> Catalogs, set VSX local DB to your built SQLite "
            "(scripts/catalogs/vsx_make.py output under <data_dir>/VSX/, table vsx_data)."
        )
    p = _resolve_catalog_db_path(raw)
    if not p.is_file():
        raise VSXCatalogError(
            f"VSX local database file not found: {p}. "
            "Build it with ./vyvar.sh --tool build_vsx -- (or VYVAR.bat) and set "
            "vsx_local_db_path in Settings -> Catalogs."
        )
    ok, code = validate_vsx_local_db_schema(p)
    if not ok:
        raise VSXCatalogError(
            f"VSX local database invalid ({code}): {p}. "
            "Rebuild with build_vsx and verify table vsx_data in Settings -> Catalogs."
        )
    n_rows = count_vsx_local_rows(p)
    if n_rows <= 0:
        raise VSXCatalogError(
            f"VSX local database has zero rows in vsx_data: {p}. "
            "Rebuild the VSX catalog (build_vsx) before running photometry."
        )
    return p


def _vsx_ra_intervals_deg(ra_min: float, ra_max: float) -> list[tuple[float, float]]:
    """Split an RA range (deg) into sub-intervals within [0, 360) when the box crosses the meridian."""
    rm, rM = float(ra_min), float(ra_max)
    if rm >= 0.0 and rM <= 360.0 and rm <= rM:
        return [(rm, rM)]
    out: list[tuple[float, float]] = []
    if rm < 0.0:
        out.append((360.0 + rm, 360.0))
        rm = 0.0
    if rM > 360.0:
        out.append((0.0, rM - 360.0))
        rM = 360.0
    if rm < rM:
        out.append((rm, rM))
    return out


def query_local_vsx(
    db_path: str | Path,
    *,
    ra_min: float,
    ra_max: float,
    dec_min: float,
    dec_max: float,
    max_rows: int | None = None,
) -> list[dict[str, Any]]:
    """Query local VSX SQLite (``vsx_data``) for a rectangular RA/Dec window (deg, ICRS).

    Uses the same bounding box as ``query_local_gaia``; RA wrap at 0 deg is handled via split intervals.
    Rows are de-duplicated by ``oid`` when present, else by (ra_deg, dec_deg).
    """
    p = _resolve_catalog_db_path(db_path)
    if not p.is_file():
        return []
    de0 = float(dec_min)
    de1 = float(dec_max)
    if de1 < de0:
        de0, de1 = de1, de0
    de0 = max(-90.0, min(90.0, de0))
    de1 = max(-90.0, min(90.0, de1))
    intervals = _vsx_ra_intervals_deg(float(ra_min), float(ra_max))
    lim: int | None = None
    if max_rows is not None:
        try:
            lim_i = int(max_rows)
            lim = lim_i if lim_i > 0 else None
        except (TypeError, ValueError):
            lim = None

    conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    try:
        try:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_vsx_ra ON vsx_data (ra_deg);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_vsx_dec ON vsx_data (dec_deg);")
            conn.commit()
        except sqlite3.Error as exc:  # noqa: BLE001
            pass
        cur_cols = conn.execute("PRAGMA table_info('vsx_data');")
        cols = {str(r[1]).strip().lower() for r in cur_cols.fetchall()}
        if not cols or "ra_deg" not in cols or "dec_deg" not in cols:
            return []
        # VSX schema differs by source. Prefer period columns when present.
        want = [
            "oid",
            "name",
            "ra_deg",
            "dec_deg",
            "var_type",
            # Optional period columns (if present in vsx_data)
            "period",
            "varperiod",
            "var_period",
            "mag_max",
            "mag_min",
        ]
        sel_cols = [c for c in want if c in cols]
        sel = ", ".join(sel_cols)
        seen: set[Any] = set()
        rows_out: list[dict[str, Any]] = []
        for rlo, rhi in intervals:
            if lim is not None and len(rows_out) >= lim:
                break
            q = (
                f"SELECT {sel} FROM vsx_data "
                "WHERE ra_deg >= ? AND ra_deg <= ? AND dec_deg >= ? AND dec_deg <= ?;"
            )
            cur = conn.execute(q, (float(rlo), float(rhi), de0, de1))
            for r in cur.fetchall():
                d = dict(r)
                oid = d.get("oid")
                key: Any
                if oid is not None:
                    key = oid
                else:
                    key = (d.get("ra_deg"), d.get("dec_deg"))
                if key in seen:
                    continue
                seen.add(key)
                rows_out.append(d)
                if lim is not None and len(rows_out) >= lim:
                    break
                log_event(
                    f"VSX SQL: {len(rows_out)} riadkov (obdlznik Dec=[{de0:.3f},{de1:.3f}], RA intervaly={len(intervals)})"
                )
        return rows_out
    finally:
        conn.close()


def validate_exoplanet_local_db_schema(db_path: str | Path) -> tuple[bool, str]:
    """Validate local exoplanet host SQLite (``exoplanet_data`` from NASA Exoplanet Archive snapshot)."""
    p = _resolve_catalog_db_path(db_path)
    if not str(db_path).strip() or not p.is_file():
        return False, "missing_file"
    con = sqlite3.connect(str(p))
    try:
        cur = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='exoplanet_data' LIMIT 1;"
        )
        if cur.fetchone() is None:
            return False, "missing_table_exoplanet_data"
        cur2 = con.execute("PRAGMA table_info('exoplanet_data');")
        cols = {str(r[1]).strip().lower() for r in cur2.fetchall()}
        required = {"obj_id", "ra_deg", "dec_deg"}
        missing = sorted([c for c in required if c not in cols])
        if missing:
            return False, f"missing_columns:{','.join(missing)}"
        return True, "ok"
    finally:
        con.close()


def query_local_exoplanet(
    db_path: str | Path,
    *,
    ra_min: float,
    ra_max: float,
    dec_min: float,
    dec_max: float,
    max_rows: int | None = None,
) -> list[dict[str, Any]]:
    """Query local exoplanet host SQLite (``exoplanet_data``) for a rectangular RA/Dec window (deg).

    Same bounding-box pattern as ``query_local_vsx``; RA wrap via ``_vsx_ra_intervals_deg``.
    Rows de-duplicated by ``obj_id`` when present, else by (ra_deg, dec_deg).
    """
    p = _resolve_catalog_db_path(db_path)
    if not p.is_file():
        return []
    de0 = float(dec_min)
    de1 = float(dec_max)
    if de1 < de0:
        de0, de1 = de1, de0
    de0 = max(-90.0, min(90.0, de0))
    de1 = max(-90.0, min(90.0, de1))
    intervals = _vsx_ra_intervals_deg(float(ra_min), float(ra_max))
    lim: int | None = None
    if max_rows is not None:
        try:
            lim_i = int(max_rows)
            lim = lim_i if lim_i > 0 else None
        except (TypeError, ValueError):
            lim = None

    conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    try:
        try:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_exo_ra ON exoplanet_data (ra_deg);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_exo_dec ON exoplanet_data (dec_deg);")
            conn.commit()
        except sqlite3.Error as exc:  # noqa: BLE001
            pass
        cur_cols = conn.execute("PRAGMA table_info('exoplanet_data');")
        cols = {str(r[1]).strip().lower() for r in cur_cols.fetchall()}
        if not cols or "ra_deg" not in cols or "dec_deg" not in cols:
            return []
        want = [
            "obj_id",
            "name",
            "host_name",
            "ra_deg",
            "dec_deg",
            "cat_source",
            "disposition",
            "period",
            "mag",
            "mag_band",
        ]
        sel_cols = [c for c in want if c in cols]
        sel = ", ".join(sel_cols)
        seen: set[Any] = set()
        rows_out: list[dict[str, Any]] = []
        for rlo, rhi in intervals:
            if lim is not None and len(rows_out) >= lim:
                break
            q = (
                f"SELECT {sel} FROM exoplanet_data "
                "WHERE ra_deg >= ? AND ra_deg <= ? AND dec_deg >= ? AND dec_deg <= ?;"
            )
            cur = conn.execute(q, (float(rlo), float(rhi), de0, de1))
            for r in cur.fetchall():
                d = dict(r)
                oid = d.get("obj_id")
                key: Any
                if oid is not None and str(oid).strip():
                    key = str(oid).strip()
                else:
                    key = (d.get("ra_deg"), d.get("dec_deg"))
                if key in seen:
                    continue
                seen.add(key)
                rows_out.append(d)
                if lim is not None and len(rows_out) >= lim:
                    break
                log_event(
                    f"EXO SQL: {len(rows_out)} riadkov (obdlznik Dec=[{de0:.3f},{de1:.3f}], RA intervaly={len(intervals)})"
                )
        return rows_out
    finally:
        conn.close()


class DraftTechnicalMetadataError(RuntimeError):
    """Focal length and/or effective pixel pitch missing after FITS + draft manifest SQL merge."""

    def __init__(self, draft_id: int) -> None:
        self.draft_id = int(draft_id)
        super().__init__(
            f"Kriticka chyba: Chybaju technicke parametre pre Draft {self.draft_id}. "
            "Skontrolujte tabulky EQUIPMENTS a TELESCOPES."
        )


def _db_fits_pixel_raw_to_micrometres(value: float) -> float:
    """Map raw FITS pixel-size keywords to micrometres (same semantics as ``pipeline``)."""
    if not math.isfinite(value) or value <= 0:
        return 0.0
    v = float(value)
    if v < 5e-5:
        return v * 1e6
    if v < 0.2:
        return v * 1000.0
    return v


def _db_to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _db_pick_header(header: fits.Header, *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in header and header[key] not in (None, ""):
            return header[key]
    return default


def _db_xbinning_strict(header: fits.Header) -> int:
    """X axis binning from FITS (``XBINNING``, else ``BINNING``; supports ``2x2`` strings)."""
    from utils import fits_binning_xy_from_header

    return fits_binning_xy_from_header(header)[0]


def _db_ybinning_header(header: fits.Header, x_fallback: int) -> int:
    from utils import fits_binning_xy_from_header

    return fits_binning_xy_from_header(header)[1]


def _db_focal_plausible_mm(mm: float) -> bool:
    return math.isfinite(mm) and 40.0 <= mm <= 120_000.0


def _db_header_focal_length_mm(header: fits.Header) -> float | None:
    """Focal length [mm] from FITS; ``FOCALLEN`` / ``FOCLEN`` often metres."""
    for key in ("FOCALLEN", "FOCLEN", "TELFOCA", "FOCAL_LEN", "FOCALL", "FOC_LEN"):
        if key not in header or header[key] in (None, "", " ", "0", 0):
            continue
        try:
            v = float(header[key])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(v) or v <= 0:
            continue
        mm = v * 1000.0 if v < 25.0 else v
        if _db_focal_plausible_mm(mm):
            return float(mm)
    return None


def _db_header_pixel_native_um_mean(header: fits.Header) -> float | None:
    p1 = _db_fits_pixel_raw_to_micrometres(
        _db_to_float(_db_pick_header(header, "PIXSIZE1", "XPIXSZ", "PIXSZLX", "PIXSIZE", default=0.0))
    )
    p2 = _db_fits_pixel_raw_to_micrometres(
        _db_to_float(_db_pick_header(header, "PIXSIZE2", "YPIXSZ", "PIXSZLY", default=0.0))
    )
    px = p1 if p1 > 0 else None
    py = p2 if p2 > 0 else None
    if px is not None and py is not None:
        return (float(px) + float(py)) / 2.0
    if px is not None:
        return float(px)
    if py is not None:
        return float(py)
    return None


def fits_header_cache_row_to_meta(row: sqlite3.Row) -> dict[str, Any]:
    """Rebuild ``extract_fits_metadata``-style dict from ``FITS_HEADER_CACHE`` row."""
    pu = row["PIXEL_UM"]
    tel, cam = row["TELESCOPE"], row["CAMERA"]
    pitch: float | None
    if pu is None:
        pitch = None
    else:
        try:
            pf = float(pu)
            pitch = pf if math.isfinite(pf) else None
        except (TypeError, ValueError):
            pitch = None
    bx = max(1, int(row["BINNING"] or 1))
    by = int(row["BINNING_Y"] if row["BINNING_Y"] is not None else row["BINNING"] or 1)
    by = max(1, by)
    phys_approx: float | None = None
    if pitch is not None and bx > 0:
        try:
            phys_approx = float(pitch) / float(bx)
            if not math.isfinite(phys_approx):
                phys_approx = None
        except (TypeError, ValueError):
            phys_approx = None
    return {
        "exposure": float(row["EXPTIME"] if row["EXPTIME"] is not None else 0.0),
        "filter": str(row["FILTER"] or "NoFilter"),
        "binning": bx,
        "binning_y": by,
        "naxis1": int(row["NAXIS1"] or 0),
        "naxis2": int(row["NAXIS2"] or 0),
        "pixel_size_um_physical": phys_approx,
        "pixel_size_um_header": pitch,
        "effective_pixel_um_plate_scale": pitch,
        "temp": float(row["CCD_TEMP"] if row["CCD_TEMP"] is not None else 0.0),
        "gain": int(row["GAIN"] or 0),
        "ra": float(row["RA_DEG"] if row["RA_DEG"] is not None else 0.0),
        "dec": float(row["DEC_DEG"] if row["DEC_DEG"] is not None else 0.0),
        "jd_start": float(row["JD_START"] if row["JD_START"] is not None else 0.0),
        "telescope": None if tel in (None, "") else str(tel),
        "camera": None if cam in (None, "") else str(cam),
    }


def _fits_header_cache_pack_row(
    fp: Path,
    sz: int,
    mt: float,
    meta: dict[str, Any],
    imagetyp: str,
    date_obs: str | None,
) -> tuple[Any, ...]:
    bx = int(meta.get("binning", 1) or 1)
    by = int(meta.get("binning_y", bx) or bx)
    bx = max(1, bx)
    by = max(1, by)
    # ``PIXEL_UM`` stores **effective** on-sky pitch [um] (physical header pixel x binning).
    pu = meta.get("pixel_size_um_header")
    if pu is None:
        pphys = meta.get("pixel_size_um_physical")
        if pphys is not None:
            try:
                ppv = float(pphys)
                if math.isfinite(ppv) and ppv > 0:
                    pu = ppv * float(bx)
            except (TypeError, ValueError):
                pu = None
    pu_sql: float | None
    if pu is None:
        pu_sql = None
    else:
        try:
            pfv = float(pu)
            pu_sql = pfv if math.isfinite(pfv) else None
        except (TypeError, ValueError):
            pu_sql = None
    tel = meta.get("telescope")
    cam = meta.get("camera")
    return (
        str(fp.resolve()),
        int(sz),
        float(mt),
        float(meta.get("exposure", 0.0)),
        str(meta.get("filter", "NoFilter")),
        bx,
        by,
        int(meta.get("naxis1", 0) or 0),
        int(meta.get("naxis2", 0) or 0),
        pu_sql,
        float(meta.get("temp", 0.0)),
        int(meta.get("gain", 0) or 0),
        float(meta.get("ra", 0.0)),
        float(meta.get("dec", 0.0)),
        float(meta.get("jd_start", 0.0)),
        None if tel in (None, "") else str(tel),
        None if cam in (None, "") else str(cam),
        date_obs,
        str(imagetyp or ""),
    )


class VyvarDatabase:
    """Database manager for variable-star observation metadata."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._archive_root_override: Path | None = None
        self.conn = ThreadSafeSQLiteConnection(open_sqlite_connection(self.db_path))
        self._create_tables()
        self.initialize_database()
        self._drop_vestigial_tables()
        self._drop_equipments_focal_column()

    def resolve_archive_root(self) -> Path:
        """Resolve Archive root for manifest I/O (config.json or data_root/Archive)."""
        if self._archive_root_override is not None:
            return Path(self._archive_root_override).expanduser().resolve()
        dbp = Path(self.db_path).expanduser().resolve()
        data_root = dbp.parent
        for cfg_path in (data_root / "config.json", dbp.parent.parent / "config.json"):
            if not cfg_path.is_file():
                continue
            try:
                raw = json.loads(cfg_path.read_text(encoding="utf-8"))
                ar = str(raw.get("archive_root") or "").strip()
                if ar:
                    p = Path(ar).expanduser()
                    if not p.is_absolute():
                        p = data_root / p
                    return p.resolve()
            except Exception:  # noqa: BLE001
                continue
        return (data_root / "Archive").resolve()

    def _enable_foreign_keys(self) -> None:
        self.conn.execute("PRAGMA foreign_keys = ON;")

    def _create_tables(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS EQUIPMENTS (
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                CAMERANAME TEXT,
                ALIAS TEXT,
                SENSORTYPE TEXT,
                SENSORSIZE TEXT,
                PIXELSIZE REAL,
                ACTIVE TEXT DEFAULT 'YES'
            );

            CREATE TABLE IF NOT EXISTS TELESCOPE (
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                TELESCOPENAME TEXT,
                ALIAS TEXT,
                DIAMETER REAL,
                FOCAL REAL,
                ACTIVE TEXT DEFAULT 'YES'
            );

            CREATE TABLE IF NOT EXISTS LOCATION (
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                PLACENAME TEXT,
                LATITUDE REAL,
                LONGITUDE REAL,
                ALTITUDE REAL
            );

            """
        )
        self.conn.commit()
        self._ensure_active_columns()
        self._ensure_is_default_columns()
        self._ensure_equipments_saturate_adu_column()
        self._migrate_qhy294mm_saturate_adu_null()
        self._ensure_equipments_cosmic_columns()
        self._ensure_equipments_bayermask_column()
        self._ensure_calibration_library_table()
        self._ensure_fits_header_cache_table()
        self._ensure_qc_processing_tables()
        self._ensure_master_sources_table()
        self._normalize_active_columns_to_text()

    def _column_sql_type(self, table: str, column: str) -> str | None:
        for row in self.conn.execute(f"PRAGMA table_info('{table}');"):
            if str(row["name"]) == column:
                return str(row["type"] or "").upper()
        return None

    def _normalize_active_values_in_place(self, table: str) -> None:
        for row in self.conn.execute(f"SELECT ID, ACTIVE FROM {table};"):
            want = self.normalize_active_text(row["ACTIVE"])
            have = row["ACTIVE"]
            if have is None or str(have).strip().upper() != want:
                self.conn.execute(
                    f"UPDATE {table} SET ACTIVE = ? WHERE ID = ?;",
                    (want, int(row["ID"])),
                )

    def _rebuild_table_safely(self, table: str, create_sql: str, insert_sql: str) -> None:
        """Crash-safe RENAME/CREATE/COPY/DROP table rebuild for one-time migrations."""
        if table not in _REBUILD_MIGRATION_TABLES:
            raise ValueError(f"unsafe table name for rebuild migration: {table!r}")
        old = f"{table}_OLD"
        self.conn.execute("PRAGMA foreign_keys = OFF;")
        try:
            self.conn.execute("BEGIN;")
            try:
                self.conn.execute(f"DROP TABLE IF EXISTS {old};")
                self.conn.execute(f"ALTER TABLE {table} RENAME TO {old};")
                self.conn.execute(create_sql.strip())
                self.conn.execute(insert_sql.strip())
                self.conn.execute(f"DROP TABLE {old};")
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise
        finally:
            self.conn.execute("PRAGMA foreign_keys = ON;")

    def _rebuild_table_active_column_to_text(self, table: str, create_sql: str, copy_sql: str) -> None:
        self._rebuild_table_safely(table, create_sql, copy_sql)

    def _normalize_active_columns_to_text(self) -> None:
        """One-time migration: ACTIVE is physically ``YES``/``NO`` on EQUIPMENTS/TELESCOPE/LOCATION."""
        self._normalize_active_values_in_place("EQUIPMENTS")

        tel_type = self._column_sql_type("TELESCOPE", "ACTIVE")
        if tel_type and "INT" in tel_type:
            self._rebuild_table_active_column_to_text(
                "TELESCOPE",
                """
                CREATE TABLE TELESCOPE (
                    ID INTEGER PRIMARY KEY AUTOINCREMENT,
                    TELESCOPENAME TEXT,
                    ALIAS TEXT,
                    DIAMETER REAL,
                    FOCAL REAL,
                    ACTIVE TEXT DEFAULT 'YES',
                    IS_DEFAULT INTEGER DEFAULT 0
                );
                """,
                """
                INSERT INTO TELESCOPE (
                    ID, TELESCOPENAME, ALIAS, DIAMETER, FOCAL, ACTIVE, IS_DEFAULT
                )
                SELECT
                    ID, TELESCOPENAME, ALIAS, DIAMETER, FOCAL,
                    CASE
                        WHEN ACTIVE IS NULL THEN 'YES'
                        WHEN typeof(ACTIVE) IN ('integer', 'real') AND CAST(ACTIVE AS INTEGER) = 0 THEN 'NO'
                        WHEN UPPER(TRIM(CAST(ACTIVE AS TEXT))) IN ('NO', 'N', 'FALSE', '0', '0.0') THEN 'NO'
                        ELSE 'YES'
                    END,
                    COALESCE(IS_DEFAULT, 0)
                FROM TELESCOPE_OLD;
                """,
            )
        else:
            self._normalize_active_values_in_place("TELESCOPE")

        loc_type = self._column_sql_type("LOCATION", "ACTIVE")
        if loc_type and "INT" in loc_type:
            self._rebuild_table_active_column_to_text(
                "LOCATION",
                """
                CREATE TABLE LOCATION (
                    ID INTEGER PRIMARY KEY AUTOINCREMENT,
                    PLACENAME TEXT,
                    LATITUDE REAL,
                    LONGITUDE REAL,
                    ALTITUDE REAL,
                    ACTIVE TEXT DEFAULT 'YES',
                    IS_DEFAULT INTEGER DEFAULT 0
                );
                """,
                """
                INSERT INTO LOCATION (
                    ID, PLACENAME, LATITUDE, LONGITUDE, ALTITUDE, ACTIVE, IS_DEFAULT
                )
                SELECT
                    ID, PLACENAME, LATITUDE, LONGITUDE, ALTITUDE,
                    CASE
                        WHEN ACTIVE IS NULL THEN 'YES'
                        WHEN typeof(ACTIVE) IN ('integer', 'real') AND CAST(ACTIVE AS INTEGER) = 0 THEN 'NO'
                        WHEN UPPER(TRIM(CAST(ACTIVE AS TEXT))) IN ('NO', 'N', 'FALSE', '0', '0.0') THEN 'NO'
                        ELSE 'YES'
                    END,
                    COALESCE(IS_DEFAULT, 0)
                FROM LOCATION_OLD;
                """,
            )
        elif loc_type:
            self._normalize_active_values_in_place("LOCATION")

        self.conn.commit()

    def _drop_final_data_view(self) -> None:
        """Retire SQL FINAL_DATA view; integrity counts use manifest scan."""
        self.conn.execute("DROP VIEW IF EXISTS FINAL_DATA;")
        self.conn.commit()

    def _ensure_master_sources_table(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS MASTER_SOURCES (
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                DRAFT_ID INTEGER,
                SOURCE_ID_GAIA TEXT,
                X_MASTER REAL,
                Y_MASTER REAL,
                RA REAL,
                DE REAL,
                G_MAG REAL,
                BP_RP REAL,
                G_FLUX_ERROR_REL REAL,
                NON_SINGLE_STAR INTEGER DEFAULT 0,
                PHOT_VARIABLE_FLAG TEXT,
                FILTER_NAME TEXT,
                PHOT_CATEGORY TEXT,
                RECOMMENDED_APERTURE REAL,
                IS_VAR INTEGER DEFAULT 0,
                IS_SATURATED INTEGER DEFAULT 0,
                IS_SAFE_COMP INTEGER DEFAULT 1,
                EXCLUSION_REASON TEXT,
                STRESS_RMS REAL,
                SAFE_OVERRIDE INTEGER DEFAULT 0,
                CREATED_AT TEXT NOT NULL
            );
            """
        )
        # Migrations for older DBs: add missing columns.
        cur = self.conn.execute("PRAGMA table_info('MASTER_SOURCES');")
        cols = {str(r[1]).upper() for r in cur.fetchall()}
        add_cols: list[tuple[str, str]] = []
        if "FILTER_NAME" not in cols:
            add_cols.append(("FILTER_NAME", "TEXT"))
        if "PHOT_CATEGORY" not in cols:
            add_cols.append(("PHOT_CATEGORY", "TEXT"))
        if "G_FLUX_ERROR_REL" not in cols:
            add_cols.append(("G_FLUX_ERROR_REL", "REAL"))
        if "NON_SINGLE_STAR" not in cols:
            add_cols.append(("NON_SINGLE_STAR", "INTEGER DEFAULT 0"))
        if "PHOT_VARIABLE_FLAG" not in cols:
            add_cols.append(("PHOT_VARIABLE_FLAG", "TEXT"))
        if "RECOMMENDED_APERTURE" not in cols:
            add_cols.append(("RECOMMENDED_APERTURE", "REAL"))
        if "IS_SAFE_COMP" not in cols:
            add_cols.append(("IS_SAFE_COMP", "INTEGER DEFAULT 1"))
        if "EXCLUSION_REASON" not in cols:
            add_cols.append(("EXCLUSION_REASON", "TEXT"))
        if "STRESS_RMS" not in cols:
            add_cols.append(("STRESS_RMS", "REAL"))
        if "SAFE_OVERRIDE" not in cols:
            add_cols.append(("SAFE_OVERRIDE", "INTEGER DEFAULT 0"))
        if "LIKELY_NONLINEAR" not in cols:
            add_cols.append(("LIKELY_NONLINEAR", "INTEGER DEFAULT 0"))
        if "ON_BAD_COLUMN" not in cols:
            add_cols.append(("ON_BAD_COLUMN", "INTEGER DEFAULT 0"))
        for name, sql_type in add_cols:
            self.conn.execute(f"ALTER TABLE MASTER_SOURCES ADD COLUMN {name} {sql_type};")
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS IDX_MASTER_SOURCES_DRAFT ON MASTER_SOURCES (DRAFT_ID);"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS IDX_MASTER_SOURCES_GAIA ON MASTER_SOURCES (SOURCE_ID_GAIA);"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS IDX_MASTER_SOURCES_PHOTCAT ON MASTER_SOURCES (DRAFT_ID, PHOT_CATEGORY);"
        )
        self.conn.commit()

    def replace_master_sources_for_draft(self, draft_id: int, rows: list[dict[str, Any]]) -> int:
        """Replace MASTER_SOURCES rows for a draft (delete+insert)."""
        did = int(draft_id)
        self.conn.execute("DELETE FROM MASTER_SOURCES WHERE DRAFT_ID = ?;", (did,))
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        n = 0
        for r in rows:
            self.conn.execute(
                """
                INSERT INTO MASTER_SOURCES (
                    DRAFT_ID,
                    SOURCE_ID_GAIA,
                    X_MASTER,
                    Y_MASTER,
                    RA,
                    DE,
                    G_MAG,
                    BP_RP,
                    G_FLUX_ERROR_REL,
                    NON_SINGLE_STAR,
                    PHOT_VARIABLE_FLAG,
                    FILTER_NAME,
                    PHOT_CATEGORY,
                    RECOMMENDED_APERTURE,
                    IS_VAR,
                    IS_SATURATED,
                    IS_SAFE_COMP,
                    EXCLUSION_REASON,
                    STRESS_RMS,
                    SAFE_OVERRIDE,
                    LIKELY_NONLINEAR,
                    ON_BAD_COLUMN,
                    CREATED_AT
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    did,
                    str(r.get("source_id_gaia") or ""),
                    float(r.get("x_master")) if r.get("x_master") is not None else None,
                    float(r.get("y_master")) if r.get("y_master") is not None else None,
                    float(r.get("ra")) if r.get("ra") is not None else None,
                    float(r.get("dec")) if r.get("dec") is not None else None,
                    float(r.get("g_mag")) if r.get("g_mag") is not None else None,
                    float(r.get("bp_rp")) if r.get("bp_rp") is not None else None,
                    float(r.get("g_flux_error_rel")) if r.get("g_flux_error_rel") is not None else None,
                    1 if int(r.get("non_single_star") or 0) else 0,
                    str(r.get("phot_variable_flag") or "") or None,
                    str(r.get("filter_name") or ""),
                    str(r.get("phot_category") or ""),
                    float(r.get("recommended_aperture")) if r.get("recommended_aperture") is not None else None,
                    1 if int(r.get("is_var") or 0) else 0,
                    1 if int(r.get("is_saturated") or 0) else 0,
                    1 if int(r.get("is_safe_comp") if r.get("is_safe_comp") is not None else 1) else 0,
                    str(r.get("exclusion_reason") or "") or None,
                    float(r.get("stress_rms")) if r.get("stress_rms") is not None else None,
                    1 if int(r.get("safe_override") or 0) else 0,
                    1 if int(r.get("likely_nonlinear") or 0) else 0,
                    1 if int(r.get("on_bad_column") or 0) else 0,
                    now,
                ),
            )
            n += 1
        self.conn.commit()
        return n

    def fetch_master_sources_for_draft(self, draft_id: int) -> list[dict[str, Any]]:
        cur = self.conn.execute(
            """
            SELECT *
            FROM MASTER_SOURCES
            WHERE DRAFT_ID = ?
            ORDER BY
              COALESCE(IS_SAFE_COMP, 0) DESC,
              COALESCE(STRESS_RMS, 1e9) ASC,
              COALESCE(G_MAG, 99) ASC,
              ID ASC;
            """,
            (int(draft_id),),
        )
        return [dict(r) for r in cur.fetchall()]

    def update_master_source_safety(
        self,
        source_row_id: int,
        *,
        is_safe_comp: bool,
        exclusion_reason: str | None = None,
        safe_override: bool = True,
    ) -> None:
        self.conn.execute(
            """
            UPDATE MASTER_SOURCES
            SET IS_SAFE_COMP = ?,
                EXCLUSION_REASON = ?,
                SAFE_OVERRIDE = ?
            WHERE ID = ?;
            """,
            (
                1 if bool(is_safe_comp) else 0,
                (str(exclusion_reason).strip() if exclusion_reason is not None else None),
                1 if bool(safe_override) else 0,
                int(source_row_id),
            ),
        )
        self.conn.commit()

    @staticmethod
    def sql_expr_active_is_true(column_ref: str) -> str:
        """SQL predicate: row is *active* (soft-delete off). Storage is ``YES``/``NO`` text."""
        c = column_ref.strip()
        return f"""(
            ({c}) IS NULL
            OR UPPER(TRIM(CAST({c} AS TEXT))) NOT IN ('NO', 'N', 'FALSE', '0', '0.0')
        )"""

    def count_final_data_for_equipment_id(self, equipment_id: int) -> int:
        from draft_provenance import count_manifest_final_data_for_equipment

        return int(count_manifest_final_data_for_equipment(self, int(equipment_id)))

    def count_final_data_for_telescope_id(self, telescope_id: int) -> int:
        from draft_provenance import count_manifest_final_data_for_telescope

        return int(count_manifest_final_data_for_telescope(self, int(telescope_id)))

    def count_references_to_location_id(self, location_id: int) -> int:
        from draft_provenance import count_manifest_references_to_location_id

        return int(count_manifest_references_to_location_id(self, int(location_id)))

    @staticmethod
    def normalize_active_db_value(raw: Any) -> int:
        """Normalize UI / DB values to **1** = active, **0** = inactive (soft-delete). Legacy ``YES``/``NO`` supported."""
        if raw is None:
            return 1
        try:
            if isinstance(raw, float) and math.isnan(raw):
                return 1
        except TypeError:
            pass
        if raw is True:
            return 1
        if raw is False:
            return 0
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            try:
                return 0 if int(raw) == 0 else 1
            except (TypeError, ValueError):
                return 1
        s = str(raw).strip().upper()
        if s in ("NO", "N", "FALSE", "0", "0.0"):
            return 0
        if s in ("YES", "Y", "TRUE", "1", "1.0", ""):
            return 1
        return 1

    @staticmethod
    def normalize_active_text(raw: Any) -> str:
        """Normalize ACTIVE column storage to ``YES`` or ``NO`` (text only)."""
        return "NO" if VyvarDatabase.normalize_active_db_value(raw) == 0 else "YES"

    def _coerce_sql_param(self, table: str, col: str, raw: Any) -> Any:
        if col == "ACTIVE":
            return self.normalize_active_text(raw)
        if col == "IS_DEFAULT":
            return 0 if int(self.normalize_active_db_value(raw)) == 0 else 1
        if table == "EQUIPMENTS" and col == "BAYERMASK":
            if raw is None:
                return None
            s = str(raw).strip()
            if not s:
                return None
            from osc_extract import normalize_bayermask

            pat = normalize_bayermask(s)
            return pat if pat is not None else "mono"
        if raw is None:
            return None
        try:
            if isinstance(raw, float) and math.isnan(raw):
                return None
        except TypeError:
            pass
        try:
            import pandas as pd_na

            if pd_na.isna(raw):
                return None
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("[DATABASE] Editor value coerce failed (non-critical): %s", exc)
        try:
            import numpy as np

            if isinstance(raw, np.generic):
                return raw.item()
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("[DATABASE] Editor numpy coerce failed (non-critical): %s", exc)
        return raw

    def apply_main_table_editor_save(
        self,
        table: str,
        pk_col: str,
        original_df: Any,
        edited_df: Any,
        *,
        editable_cols: list[str],
    ) -> dict[str, int]:
        """Apply INSERT/UPDATE/DELETE from a Streamlit ``data_editor`` diff.

        Only ``EQUIPMENTS``, ``TELESCOPE``, ``LOCATION``, ``DRAFT_MANIFEST`` are allowed.

        **EQUIPMENTS / TELESCOPE:** a row removed from the editor is **not** ``DELETE``-d; ``ACTIVE`` is set
        to **0** (soft-delete). Physical ``DELETE`` for those tables is never performed from this API.
        **DRAFT_MANIFEST:** manifest-direct save (no SQL staging table).
        **LOCATION:** physical delete with reference checks (manifest rig ``location_id`` usage).
        """
        import pandas as pd_local

        if table not in _EDITABLE_EDITOR_TABLES:
            raise ValueError(f"Refusing to edit non-allowlisted table: {table!r}")
        if pk_col != "ID":
            raise ValueError("Expected primary key column ID.")

        def _parse_pk(val: Any) -> int | None:
            if val is None:
                return None
            try:
                if isinstance(val, float) and math.isnan(val):
                    return None
            except TypeError:
                pass
            if pd_local.isna(val):
                return None
            s = str(val).strip()
            if not s:
                return None
            try:
                return int(float(s))
            except (TypeError, ValueError):
                return None

        orig = original_df.copy()
        edit = edited_df.copy()
        if pk_col not in orig.columns or pk_col not in edit.columns:
            raise ValueError(f"Missing primary key column {pk_col!r}.")

        # Track an explicit IS_DEFAULT=1 toggle so we can enforce exactly-one after save.
        _new_default_pid: int | None = None
        if "IS_DEFAULT" in editable_cols and "IS_DEFAULT" in orig.columns and "IS_DEFAULT" in edit.columns:
            for _, _r in edit.iterrows():
                _pid = _parse_pk(_r.get(pk_col))
                if _pid is None:
                    continue
                if int(self.normalize_active_db_value(_r.get("IS_DEFAULT"))) == 1:
                    _hits = orig.loc[orig[pk_col].apply(_parse_pk) == _pid]
                    _was = (
                        int(self.normalize_active_db_value(_hits.iloc[0].get("IS_DEFAULT")))
                        if not _hits.empty
                        else 0
                    )
                    if _was != 1:
                        _new_default_pid = _pid  # last explicit check wins

        orig_ids: set[int] = set()
        for v in orig[pk_col].tolist():
            pid = _parse_pk(v)
            if pid is not None:
                orig_ids.add(pid)

        edited_by_id: dict[int, Any] = {}
        new_rows: list[Any] = []
        for _, row in edit.iterrows():
            pid = _parse_pk(row.get(pk_col))
            if pid is None:
                new_rows.append(row)
            else:
                edited_by_id[pid] = row

        deleted_ids = orig_ids - set(edited_by_id.keys())
        inserted = 0
        updated = 0
        deleted = 0
        soft_deactivated = 0

        if table == "DRAFT_MANIFEST":
            from draft_provenance import (
                _optional_float,
                _optional_int,
                clear_manifest_shadow_load_cache,
                load_or_init_manifest,
                patch_draft_manifest,
                resolve_draft_dir_for_id,
            )

            inserted = 0
            updated = 0
            deleted = 0
            for pid, row in edited_by_id.items():
                root = resolve_draft_dir_for_id(self, int(pid))
                if root is None:
                    continue
                manifest = load_or_init_manifest(root, int(pid))
                rig = dict(manifest.get("rig") or {})
                paths = dict(manifest.get("paths") or {})
                center = dict(manifest.get("center") or {"ra_deg": None, "de_deg": None})
                scalar_map = {
                    "ID_EQUIPMENTS": ("rig", "equipment_id"),
                    "ID_TELESCOPE": ("rig", "telescope_id"),
                    "ID_LOCATION": ("rig", "location_id"),
                    "ID_SCANNING": ("rig", "scanning_id"),
                    "LIGHTS_PATH": ("paths", "lights"),
                    "CALIB_PATH": ("paths", "calib"),
                    "ARCHIVE_PATH": ("paths", "archive"),
                    "MASTERSTAR_PATH": ("paths", "masterstar"),
                    "MASTERSTAR_FITS_PATH": ("paths", "masterstar_fits"),
                    "STATUS": ("status", None),
                    "CENTEROFFIELDRA": ("center", "ra_deg"),
                    "CENTEROFFIELDDE": ("center", "de_deg"),
                    "OBSERVATIONSTARTJD": ("observation_start_jd", None),
                    "IS_CALIBRATED": ("is_calibrated", None),
                }
                changed = False
                for col in editable_cols:
                    if col not in row.index:
                        continue
                    val = self._coerce_sql_param(table, col, row[col])
                    spec = scalar_map.get(col)
                    if spec is None:
                        continue
                    bucket, key = spec
                    if bucket == "rig" and key:
                        rig[key] = _optional_int(val) if val is not None else None
                        changed = True
                    elif bucket == "paths" and key:
                        paths[key] = str(val).strip() if val is not None else None
                        changed = True
                    elif bucket == "center" and key:
                        center[key] = float(val) if val is not None else None
                        changed = True
                    elif bucket == "status":
                        patch_draft_manifest(root, int(pid), status=str(val) if val is not None else None)
                        changed = True
                    elif bucket == "observation_start_jd":
                        patch_draft_manifest(root, int(pid), observation_start_jd=_optional_float(val))
                        changed = True
                    elif bucket == "is_calibrated":
                        patch_draft_manifest(
                            root,
                            int(pid),
                            is_calibrated=1 if int(val) != 0 else 0 if val is not None else None,
                        )
                        changed = True
                if changed:
                    patch_draft_manifest(
                        root,
                        int(pid),
                        rig=rig,
                        paths=paths,
                        center=center,
                    )
                    updated += 1
            clear_manifest_shadow_load_cache()
            return {
                "inserted": inserted,
                "updated": updated,
                "deleted": deleted,
                "soft_deactivated": 0,
            }

        pragma_cols = [r[1] for r in self.conn.execute(f"PRAGMA table_info({table});").fetchall()]
        insert_colnames = [c for c in pragma_cols if c != pk_col]

        try:
            self.conn.execute("BEGIN;")
            for did in sorted(deleted_ids):
                if table in ("EQUIPMENTS", "TELESCOPE"):
                    if "ACTIVE" not in pragma_cols:
                        raise ValueError(f"Tabulka {table} nema stlpec ACTIVE - soft-delete nie je mozny.")
                    self.conn.execute(
                        f"UPDATE {table} SET ACTIVE = 'NO' WHERE {pk_col} = ?;",
                        (did,),
                    )
                    soft_deactivated += 1
                    continue
                if table == "LOCATION":
                    n = self.count_references_to_location_id(did)
                    if n > 0:
                        raise ValueError(
                            f"Lokalitu ID={did} nie je mozne zmazat: {n} odkazov v draft manifestoch."
                        )
                self.conn.execute(f"DELETE FROM {table} WHERE {pk_col} = ?;", (did,))
                deleted += 1

            for pid, row in edited_by_id.items():
                if pid not in orig_ids:
                    continue
                hits = orig.loc[orig[pk_col].apply(_parse_pk) == pid]
                if hits.empty:
                    continue
                orig_row = hits.iloc[0]
                changes: dict[str, Any] = {}
                for col in editable_cols:
                    if col == pk_col or col not in pragma_cols:
                        continue
                    if col not in row.index or col not in orig_row.index:
                        continue
                    nv = self._coerce_sql_param(table, col, row[col])
                    ov = self._coerce_sql_param(table, col, orig_row[col])
                    if nv != ov:
                        changes[col] = nv
                if not changes:
                    continue
                set_sql = ", ".join(f"{k} = ?" for k in changes)
                params = list(changes.values()) + [pid]
                self.conn.execute(f"UPDATE {table} SET {set_sql} WHERE {pk_col} = ?;", params)
                updated += 1

            for row in new_rows:
                vals: list[Any] = []
                for c in insert_colnames:
                    raw = row[c] if c in row.index else None
                    vals.append(self._coerce_sql_param(table, c, raw))
                placeholders = ", ".join("?" * len(insert_colnames))
                cols_sql = ", ".join(insert_colnames)
                ins_cur = self.conn.execute(
                    f"INSERT INTO {table} ({cols_sql}) VALUES ({placeholders});",
                    vals,
                )
                inserted += 1

            # Enforce exactly-one IS_DEFAULT when the user explicitly checked a new row.
            if (
                _new_default_pid is not None
                and table in ("EQUIPMENTS", "TELESCOPE", "LOCATION")
                and "IS_DEFAULT" in pragma_cols
            ):
                self.conn.execute(
                    f"UPDATE {table} SET IS_DEFAULT = CASE WHEN {pk_col} = ? THEN 1 ELSE 0 END;",
                    (int(_new_default_pid),),
                )

            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

        return {
            "inserted": inserted,
            "updated": updated,
            "deleted": deleted,
            "soft_deactivated": soft_deactivated,
        }

    def _ensure_qc_processing_tables(self) -> None:
        """Append-only QC Apply snapshots: hashtag + accepted manifest light rows (IS_REJECTED=0)."""
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS OBS_QC_PROCESSING_RUN (
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                PROCESSING_HASH TEXT NOT NULL UNIQUE,
                DRAFT_ID INTEGER NOT NULL,
                CREATED_AT TEXT NOT NULL
            );
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS OBS_QC_PROCESSING_FILE (
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                RUN_ID INTEGER NOT NULL,
                SOURCE_OBS_FILE_ID INTEGER NOT NULL,
                FILE_PATH TEXT,
                FILTER TEXT,
                EXPTIME REAL,
                INSPECTION_JD REAL,
                FWHM REAL,
                DRIFT REAL,
                FOREIGN KEY (RUN_ID) REFERENCES OBS_QC_PROCESSING_RUN (ID) ON DELETE CASCADE
            );
            """
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS IDX_QC_PROC_RUN_DRAFT ON OBS_QC_PROCESSING_RUN (DRAFT_ID);"
        )
        self.conn.commit()

    def fetch_obs_draft_by_id(self, draft_id: int) -> dict[str, Any] | None:
        from draft_provenance import fetch_obs_draft_row_manifest

        return fetch_obs_draft_row_manifest(self, int(draft_id))

    def _draft_rig_resolve_row(self, draft_id: int) -> dict[str, Any] | None:
        """Rig FK columns + path hints from manifest."""
        row = self.fetch_obs_draft_by_id(int(draft_id))
        if row is None:
            return None
        return {
            "ID_EQUIPMENTS": row.get("ID_EQUIPMENTS"),
            "ID_TELESCOPE": row.get("ID_TELESCOPE"),
            "ID_LOCATION": row.get("ID_LOCATION"),
            "ID_SCANNING": row.get("ID_SCANNING"),
            "ARCHIVE_PATH": row.get("ARCHIVE_PATH"),
            "LIGHTS_PATH": row.get("LIGHTS_PATH"),
        }

    def _resolve_draft_rig_id(self, draft_id: int, field: str, db_col: str) -> int | None:
        from draft_provenance import resolve_rig_id_manifest_first

        row = self._draft_rig_resolve_row(draft_id)
        if row is None:
            return None
        raw = row.get(db_col)
        db_val: int | None
        if raw is None:
            db_val = None
        else:
            try:
                db_val = int(raw)
            except (TypeError, ValueError):
                db_val = None
        return resolve_rig_id_manifest_first(
            int(draft_id),
            field,
            db_val,
            draft_row=row,
            db=self,
        )

    def get_draft_equipment_id(self, draft_id: int) -> int | None:
        """Manifest-first ``equipment_id``; DB fallback when manifest absent."""
        return self._resolve_draft_rig_id(int(draft_id), "equipment_id", "ID_EQUIPMENTS")

    def get_draft_telescope_id(self, draft_id: int) -> int | None:
        """Manifest-first ``telescope_id``; DB fallback when manifest absent."""
        return self._resolve_draft_rig_id(int(draft_id), "telescope_id", "ID_TELESCOPE")

    def get_draft_location_id(self, draft_id: int) -> int | None:
        """Manifest-first ``location_id``; DB fallback when manifest absent."""
        return self._resolve_draft_rig_id(int(draft_id), "location_id", "ID_LOCATION")

    def get_draft_scanning_id(self, draft_id: int) -> int | None:
        """Manifest-first ``scanning_id``; DB fallback when manifest absent."""
        return self._resolve_draft_rig_id(int(draft_id), "scanning_id", "ID_SCANNING")

    def _try_refresh_draft_manifest(self, draft_id: int) -> None:
        """No-op: manifest is written directly (Phase 2.8)."""
        return

    def update_obs_draft_center(self, draft_id: int, center_ra_deg: float, center_de_deg: float) -> None:
        """Persist draft field center (ICRS degrees) in manifest."""
        from draft_provenance import patch_draft_manifest, resolve_draft_dir_for_id

        root = resolve_draft_dir_for_id(self, int(draft_id))
        if root is None:
            raise ValueError(f"Draft '{draft_id}' not found for center update.")
        patch_draft_manifest(
            root,
            int(draft_id),
            center={"ra_deg": float(center_ra_deg), "de_deg": float(center_de_deg)},
        )

    def update_obs_draft_status_panel_values(
        self,
        draft_id: int,
        *,
        center_ra_deg: float | None = None,
        center_de_deg: float | None = None,
        focal_mm: float | None = None,
        pixel_um: float | None = None,
    ) -> None:
        """Persist status-panel values for a draft (center + optional focal/pixel) in manifest."""
        from draft_provenance import patch_draft_manifest, resolve_draft_dir_for_id

        _ = focal_mm, pixel_um
        if center_ra_deg is None and center_de_deg is None:
            return
        root = resolve_draft_dir_for_id(self, int(draft_id))
        if root is None:
            raise ValueError(f"Draft '{draft_id}' not found for status-panel update.")
        center: dict[str, float | None] = {}
        if center_ra_deg is not None:
            center["ra_deg"] = float(center_ra_deg)
        if center_de_deg is not None:
            center["de_deg"] = float(center_de_deg)
        patch_draft_manifest(root, int(draft_id), center=center)

    def get_obs_draft_masterstar_path(self, draft_id: int) -> str | None:
        """Return persisted MASTERSTAR path for a draft (manifest-first)."""
        row = self.fetch_obs_draft_by_id(int(draft_id))
        if row is None:
            return None
        v = row.get("MASTERSTAR_FITS_PATH") or row.get("MASTERSTAR_PATH")
        s = str(v).strip() if v is not None else ""
        return s or None

    # ------------------------------------------------------------------
    # MASTERSTAR selection vs MASTERSTAR product (separate persistence)
    # ------------------------------------------------------------------
    def set_obs_draft_masterstar_source_path(self, draft_id: int, source_path: str | None) -> None:
        """Persist the selected source frame path in manifest ``paths.masterstar``."""
        from draft_provenance import load_or_init_manifest, patch_draft_manifest, resolve_draft_dir_for_id

        _p = (str(source_path).strip() if source_path is not None else "") or None
        root = resolve_draft_dir_for_id(self, int(draft_id))
        if root is None:
            raise ValueError(f"Draft '{draft_id}' not found for MASTERSTAR source update.")
        manifest = load_or_init_manifest(root, int(draft_id))
        paths = dict(manifest.get("paths") or {})
        paths["masterstar"] = _p
        patch_draft_manifest(root, int(draft_id), paths=paths)

    def get_obs_draft_masterstar_source_path(self, draft_id: int) -> str | None:
        """Return persisted MASTERSTAR *source* (selected frame) path (manifest-first)."""
        row = self.fetch_obs_draft_by_id(int(draft_id))
        if row is None:
            return None
        v = row.get("MASTERSTAR_PATH")
        s = str(v).strip() if v is not None else ""
        return s or None

    def set_obs_draft_masterstar_fits_path(self, draft_id: int, fits_path: str | None) -> None:
        """Persist produced MASTERSTAR.fits path in manifest."""
        from draft_provenance import load_or_init_manifest, patch_draft_manifest, resolve_draft_dir_for_id

        _p = (str(fits_path).strip() if fits_path is not None else "") or None
        root = resolve_draft_dir_for_id(self, int(draft_id))
        if root is None:
            raise ValueError(f"Draft '{draft_id}' not found for MASTERSTAR FITS path update.")
        manifest = load_or_init_manifest(root, int(draft_id))
        paths = dict(manifest.get("paths") or {})
        paths["masterstar_fits"] = _p
        patch_draft_manifest(root, int(draft_id), paths=paths)

    def record_qc_processing_apply(
        self,
        draft_id: int,
        processing_hash: str,
        *,
        overwrite: bool = False,
    ) -> int:
        """Idempotent upsert run + copy light rows with ``IS_REJECTED`` 0."""
        h = str(processing_hash).strip()
        if not h:
            raise ValueError("processing_hash is empty")
        _ = overwrite  # backward compatible arg; run is always upserted
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.conn.execute(
            """
            INSERT INTO OBS_QC_PROCESSING_RUN (PROCESSING_HASH, DRAFT_ID, CREATED_AT)
            VALUES (?, ?, ?)
            ON CONFLICT(PROCESSING_HASH) DO UPDATE SET
                DRAFT_ID = excluded.DRAFT_ID,
                CREATED_AT = excluded.CREATED_AT;
            """,
            (h, int(draft_id), now),
        )
        run_row = self.conn.execute(
            "SELECT ID FROM OBS_QC_PROCESSING_RUN WHERE PROCESSING_HASH = ?;",
            (h,),
        ).fetchone()
        if run_row is None:
            raise ValueError(f"Cannot resolve QC processing run for hash: {h}")
        run_id = int(run_row["ID"])
        self.conn.execute("DELETE FROM OBS_QC_PROCESSING_FILE WHERE RUN_ID = ?;", (run_id,))
        light_rows = self.fetch_draft_light_rows_for_quality(int(draft_id))
        for frow in light_rows:
            if frow.get("IS_REJECTED") not in (None, 0):
                continue
            ex_v: float | None = None
            if frow["EXPTIME"] is not None:
                try:
                    ex_f = float(frow["EXPTIME"])
                    ex_v = ex_f if math.isfinite(ex_f) else None
                except (TypeError, ValueError):
                    ex_v = None
            jd_v: float | None = None
            if frow["INSPECTION_JD"] is not None:
                try:
                    jd_f = float(frow["INSPECTION_JD"])
                    jd_v = jd_f if math.isfinite(jd_f) else None
                except (TypeError, ValueError):
                    jd_v = None
            fwhm_v: float | None = None
            if frow["FWHM"] is not None:
                try:
                    fw_f = float(frow["FWHM"])
                    fwhm_v = fw_f if math.isfinite(fw_f) else None
                except (TypeError, ValueError):
                    fwhm_v = None
            drift_v: float | None = None
            if frow["DRIFT"] is not None:
                try:
                    dr_f = float(frow["DRIFT"])
                    drift_v = dr_f if math.isfinite(dr_f) else None
                except (TypeError, ValueError):
                    drift_v = None
            self.conn.execute(
                """
                INSERT INTO OBS_QC_PROCESSING_FILE (
                    RUN_ID, SOURCE_OBS_FILE_ID, FILE_PATH, FILTER, EXPTIME, INSPECTION_JD, FWHM, DRIFT
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    run_id,
                    int(frow["ID"]),
                    str(frow["FILE_PATH"] or ""),
                    str(frow["FILTER"]) if frow["FILTER"] is not None else None,
                    ex_v,
                    jd_v,
                    fwhm_v,
                    drift_v,
                ),
            )
        self.conn.commit()
        return run_id

    def update_obs_draft_status(self, draft_id: int, status: str) -> None:
        from draft_provenance import patch_draft_manifest, resolve_draft_dir_for_id

        root = resolve_draft_dir_for_id(self, int(draft_id))
        if root is None:
            raise ValueError(f"Draft '{draft_id}' not found for status update.")
        patch_draft_manifest(root, int(draft_id), status=str(status))

    def _ensure_fits_header_cache_table(self) -> None:
        """Primary-header cache for fast smart_scan (path + size + mtime invalidation)."""
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS FITS_HEADER_CACHE (
                FILE_PATH TEXT PRIMARY KEY,
                FILE_SIZE INTEGER NOT NULL,
                MTIME REAL NOT NULL,
                EXPTIME REAL NOT NULL,
                FILTER TEXT NOT NULL,
                BINNING INTEGER NOT NULL,
                BINNING_Y INTEGER NOT NULL,
                NAXIS1 INTEGER NOT NULL,
                NAXIS2 INTEGER NOT NULL,
                PIXEL_UM REAL,
                CCD_TEMP REAL NOT NULL,
                GAIN INTEGER NOT NULL,
                RA_DEG REAL NOT NULL,
                DEC_DEG REAL NOT NULL,
                JD_START REAL NOT NULL,
                TELESCOPE TEXT,
                CAMERA TEXT,
                DATE_OBS TEXT,
                IMAGETYP TEXT NOT NULL DEFAULT ''
            );
            """
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS IDX_FITS_HDR_CACHE_STALE ON FITS_HEADER_CACHE (FILE_PATH, FILE_SIZE, MTIME);"
        )
        self.conn.commit()

    @staticmethod
    def fits_header_cache_file_key(path: str | Path) -> str:
        return str(Path(path).resolve())

    def fits_header_cache_get_if_fresh(
        self,
        path: str | Path,
        *,
        file_size: int,
        mtime: float,
    ) -> sqlite3.Row | None:
        """Return cache row if it exists and matches current ``file_size`` and ``mtime``."""
        key = self.fits_header_cache_file_key(path)
        row = self.conn.execute(
            "SELECT * FROM FITS_HEADER_CACHE WHERE FILE_PATH = ?;",
            (key,),
        ).fetchone()
        if row is None:
            return None
        if int(row["FILE_SIZE"]) != int(file_size):
            return None
        # Filesystems may round mtime; allow tiny float noise (and ~2s coarse stamps).
        if abs(float(row["MTIME"]) - float(mtime)) > 2.0:
            return None
        return row

    def fits_header_cache_try_meta(self, path: str | Path) -> dict[str, Any] | None:
        """If file exists and cache row matches size+mtime, return metadata dict (``extract_fits_metadata`` shape)."""
        p = Path(path)
        try:
            st = p.stat()
        except OSError:
            return None
        row = self.fits_header_cache_get_if_fresh(p, file_size=int(st.st_size), mtime=float(st.st_mtime))
        if row is None:
            return None
        return fits_header_cache_row_to_meta(row)

    def fits_header_cache_try_filter(self, path: str | Path) -> str | None:
        """Return normalized filter string from cache if fresh; else ``None``."""
        p = Path(path)
        try:
            st = p.stat()
        except OSError:
            return None
        row = self.fits_header_cache_get_if_fresh(p, file_size=int(st.st_size), mtime=float(st.st_mtime))
        if row is None:
            return None
        flt = str(row["FILTER"] or "").strip()
        if not flt or flt.lower() in {"unknown", "none", "nan"}:
            return "NoFilter"
        return flt

    def fits_header_cache_upsert_batch(
        self,
        items: list[tuple[Path, int, float, dict[str, Any], str, str | None]],
    ) -> None:
        """Insert or replace many rows in one transaction. Each item:
        ``(path, file_size, mtime, meta_dict, imagetyp_raw, date_obs_str_or_none)``.
        """
        if not items:
            return
        sql = """
            INSERT INTO FITS_HEADER_CACHE (
                FILE_PATH, FILE_SIZE, MTIME, EXPTIME, FILTER, BINNING, BINNING_Y,
                NAXIS1, NAXIS2, PIXEL_UM, CCD_TEMP, GAIN, RA_DEG, DEC_DEG, JD_START,
                TELESCOPE, CAMERA, DATE_OBS, IMAGETYP
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(FILE_PATH) DO UPDATE SET
                FILE_SIZE = excluded.FILE_SIZE,
                MTIME = excluded.MTIME,
                EXPTIME = excluded.EXPTIME,
                FILTER = excluded.FILTER,
                BINNING = excluded.BINNING,
                BINNING_Y = excluded.BINNING_Y,
                NAXIS1 = excluded.NAXIS1,
                NAXIS2 = excluded.NAXIS2,
                PIXEL_UM = excluded.PIXEL_UM,
                CCD_TEMP = excluded.CCD_TEMP,
                GAIN = excluded.GAIN,
                RA_DEG = excluded.RA_DEG,
                DEC_DEG = excluded.DEC_DEG,
                JD_START = excluded.JD_START,
                TELESCOPE = excluded.TELESCOPE,
                CAMERA = excluded.CAMERA,
                DATE_OBS = excluded.DATE_OBS,
                IMAGETYP = excluded.IMAGETYP;
        """
        self.conn.execute("BEGIN IMMEDIATE;")
        try:
            for fp, sz, mt, meta, imagetyp, date_obs in items:
                self.conn.execute(sql, _fits_header_cache_pack_row(fp, sz, mt, meta, imagetyp, date_obs))
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def fits_header_cache_upsert_one(
        self,
        path: Path,
        *,
        file_size: int,
        mtime: float,
        meta: dict[str, Any],
        imagetyp: str,
        date_obs: str | None,
    ) -> None:
        self.fits_header_cache_upsert_batch([(path, file_size, mtime, meta, imagetyp, date_obs)])

    def _ensure_calibration_library_table(self) -> None:
        """Master Dark/Flat registry for fast matching on import (path + XBINNING, EXPTIME, TEMP, FILTER)."""
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS CALIBRATION_LIBRARY (
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                KIND TEXT NOT NULL,
                FILE_PATH TEXT NOT NULL UNIQUE,
                XBINNING INTEGER NOT NULL,
                EXPTIME REAL NOT NULL,
                CCD_TEMP REAL,
                FILTER_NAME TEXT NOT NULL DEFAULT '',
                GAIN INTEGER NOT NULL DEFAULT 0,
                NCOMBINE INTEGER,
                REGISTERED_AT TEXT NOT NULL
            );
            """
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS IDX_CAL_LIB_LOOKUP ON CALIBRATION_LIBRARY "
            "(KIND, XBINNING, EXPTIME, FILTER_NAME, GAIN);"
        )
        self.conn.commit()
        self._migrate_calibration_library_scope_columns()

    def _migrate_calibration_library_scope_columns(self) -> None:
        """Add optional equipment/telescope scope (set = kamera + dalekohlad). Legacy rows: NULL,NULL = vseobecne."""
        cur = self.conn.execute("PRAGMA table_info('CALIBRATION_LIBRARY');")
        cols = {str(r[1]).upper() for r in cur.fetchall()}
        if "ID_EQUIPMENTS" not in cols:
            self.conn.execute("ALTER TABLE CALIBRATION_LIBRARY ADD COLUMN ID_EQUIPMENTS INTEGER;")
        if "ID_TELESCOPE" not in cols:
            self.conn.execute("ALTER TABLE CALIBRATION_LIBRARY ADD COLUMN ID_TELESCOPE INTEGER;")
        self.conn.commit()

    @staticmethod
    def calibration_scopes_match(
        row_eq: int | None,
        row_tel: int | None,
        req_eq: int | None,
        req_tel: int | None,
    ) -> bool:
        """True when two equipment/telescope scopes are the same registration target."""
        if row_eq is None and row_tel is None and req_eq is None and req_tel is None:
            return True
        if row_eq is None or row_tel is None or req_eq is None or req_tel is None:
            return False
        return int(row_eq) == int(req_eq) and int(row_tel) == int(req_tel)

    def get_calibration_library_row_by_path(
        self, file_path: str | Path
    ) -> dict[str, Any] | None:
        """Return the registry row for an absolute master path, if any."""
        self._ensure_calibration_library_table()
        fp = str(Path(file_path).resolve())
        cur = self.conn.execute(
            "SELECT * FROM CALIBRATION_LIBRARY WHERE FILE_PATH = ? LIMIT 1;",
            (fp,),
        )
        row = cur.fetchone()
        return dict(row) if row is not None else None

    def calibration_library_scope_conflicts(
        self,
        file_path: str | Path,
        id_equipments: int | None,
        id_telescope: int | None,
    ) -> bool:
        """True if ``file_path`` is registered to a different equipment/telescope set."""
        row = self.get_calibration_library_row_by_path(file_path)
        if row is None:
            return False
        return not self.calibration_scopes_match(
            row.get("ID_EQUIPMENTS"),
            row.get("ID_TELESCOPE"),
            id_equipments,
            id_telescope,
        )

    def register_calibration_library_entry(
        self,
        *,
        kind: str,
        file_path: str | Path,
        xbinning: int,
        exptime: float,
        ccd_temp: float | None,
        filter_name: str = "",
        gain: int = 0,
        ncombine: int | None = None,
        id_equipments: int | None = None,
        id_telescope: int | None = None,
    ) -> bool:
        """Insert or update one master calibration row (keyed by absolute ``FILE_PATH``).

        Returns False without writing when the path is already registered to a
        different equipment/telescope set (caller should use a new file path).
        """
        k = str(kind or "").strip().lower()
        if k not in ("dark", "flat"):
            raise ValueError("kind must be 'dark' or 'flat'")
        fp = str(Path(file_path).resolve())
        flt = "" if k == "dark" else str(filter_name or "").strip()
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _eq = int(id_equipments) if id_equipments is not None else None
        _tel = int(id_telescope) if id_telescope is not None else None
        if _eq is None or _tel is None:
            try:
                log_event(
                    f"CALIB LIB: refused registration without equipment/telescope scope: {fp}"
                )
            except Exception:  # noqa: BLE001
                pass
            return False
        if k == "dark" and (
            ccd_temp is None
            or not math.isfinite(float(ccd_temp))
        ):
            try:
                log_event(
                    f"CALIB LIB: refused dark registration without finite CCD_TEMP: {fp}"
                )
            except Exception:  # noqa: BLE001
                # EXC-0070: T3 -- pure log_event guard (EXCEPT-BULK-2 2026-07-08)
                pass
            return False
        existing = self.get_calibration_library_row_by_path(fp)
        if existing is not None and not self.calibration_scopes_match(
            existing.get("ID_EQUIPMENTS"),
            existing.get("ID_TELESCOPE"),
            _eq,
            _tel,
        ):
            return False
        self.conn.execute(
            """
            INSERT INTO CALIBRATION_LIBRARY (
                KIND, FILE_PATH, XBINNING, EXPTIME, CCD_TEMP, FILTER_NAME, GAIN, NCOMBINE,
                REGISTERED_AT, ID_EQUIPMENTS, ID_TELESCOPE
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(FILE_PATH) DO UPDATE SET
                KIND = excluded.KIND,
                XBINNING = excluded.XBINNING,
                EXPTIME = excluded.EXPTIME,
                CCD_TEMP = excluded.CCD_TEMP,
                FILTER_NAME = excluded.FILTER_NAME,
                GAIN = excluded.GAIN,
                NCOMBINE = excluded.NCOMBINE,
                REGISTERED_AT = excluded.REGISTERED_AT,
                ID_EQUIPMENTS = excluded.ID_EQUIPMENTS,
                ID_TELESCOPE = excluded.ID_TELESCOPE;
            """,
            (
                k,
                fp,
                int(xbinning),
                float(exptime),
                float(ccd_temp) if ccd_temp is not None and math.isfinite(float(ccd_temp)) else None,
                flt,
                int(gain),
                int(ncombine) if ncombine is not None else None,
                now,
                _eq,
                _tel,
            ),
        )
        self.conn.commit()
        return True

    def delete_calibration_library_entry_by_path(self, file_path: str | Path) -> int:
        """Remove registry row for a master file (after deleting the FITS on disk). Returns deleted row count."""
        fp = str(Path(file_path).resolve())
        cur = self.conn.execute("DELETE FROM CALIBRATION_LIBRARY WHERE FILE_PATH = ?", (fp,))
        self.conn.commit()
        return int(cur.rowcount or 0)

    def calibration_library_path_tag_map(self) -> dict[str, dict[str, Any]]:
        """``Path.resolve().casefold()`` -> ``{id_equipments, id_telescope, camera, telescope, ...}`` for UI."""
        self._ensure_calibration_library_table()
        cur = self.conn.execute(
            """
            SELECT c.FILE_PATH, c.ID_EQUIPMENTS, c.ID_TELESCOPE,
                   e.CAMERANAME, e.ALIAS AS EQ_ALIAS,
                   t.TELESCOPENAME, t.ALIAS AS TEL_ALIAS
            FROM CALIBRATION_LIBRARY c
            LEFT JOIN EQUIPMENTS e ON c.ID_EQUIPMENTS = e.ID
            LEFT JOIN TELESCOPE t ON c.ID_TELESCOPE = t.ID
            """
        )
        out: dict[str, dict[str, Any]] = {}
        for row in cur.fetchall():
            try:
                key = str(Path(str(row["FILE_PATH"])).resolve()).casefold()
            except OSError:
                key = str(row["FILE_PATH"]).casefold()
            out[key] = {
                "id_equipments": row["ID_EQUIPMENTS"],
                "id_telescope": row["ID_TELESCOPE"],
                "camera": row["CAMERANAME"],
                "eq_alias": row["EQ_ALIAS"],
                "telescope": row["TELESCOPENAME"],
                "tel_alias": row["TEL_ALIAS"],
            }
        return out

    def find_best_calibration_library_path(
        self,
        *,
        kind: str,
        xbinning: int,
        exptime: float,
        ccd_temp: float | None,
        filter_name: str = "",
        gain: int = 0,
        temp_tolerance: float = 0.5,
        prefer_unbinned_master: bool = True,
        id_equipments: int | None = None,
        id_telescope: int | None = None,
    ) -> str | None:
        """Return best scoped calibration master path, or None.

        Scoped model (no global NULL,NULL rows):
        - **dark:** equipment+telescope, XBINNING, EXPTIME, GAIN, CCD_TEMP within tol (required).
        - **flat:** equipment+telescope, XBINNING, GAIN, FILTER_NAME (no EXPTIME gate).

        Prefers smallest |DeltaT| for dark, then newest mtime. When ``prefer_unbinned_master`` and
        ``xbinning`` > 1, tries XBINNING=1 first for on-the-fly resample.
        """
        k = str(kind or "").strip().lower()
        if k not in ("dark", "flat"):
            return None
        try:
            tol = float(temp_tolerance)
        except (TypeError, ValueError):
            tol = 0.5
        if not math.isfinite(tol) or tol < 0:
            tol = 0.5
        try:
            eq_id = int(id_equipments)
            tel_id = int(id_telescope)
        except (TypeError, ValueError):
            try:
                log_event(f"CALIB LIB: no scoped master for {k} - missing equipment/telescope ids")
            except Exception:  # noqa: BLE001
                # EXC-0072: T3 -- pure log_event guard (EXCEPT-BULK-2 2026-07-08)
                pass
            return None
        if k == "dark":
            if ccd_temp is None or not math.isfinite(float(ccd_temp)):
                try:
                    log_event(
                        f"CALIB LIB: no scoped dark - light CCD_TEMP unknown "
                        f"(eq={eq_id} tel={tel_id})"
                    )
                except Exception:  # noqa: BLE001
                    # EXC-0071: T3 -- pure log_event guard (EXCEPT-BULK-2 2026-07-08)
                    pass
                return None
        flt = "" if k == "dark" else str(filter_name or "").strip()
        light_temp = float(ccd_temp) if k == "dark" else float("nan")

        def _query_rows(xb: int) -> list[sqlite3.Row]:
            if k == "dark":
                params: list[Any] = [
                    int(xb),
                    float(exptime),
                    int(gain),
                    float(light_temp),
                    float(tol),
                    eq_id,
                    tel_id,
                ]
                query = """
                    SELECT FILE_PATH, CCD_TEMP
                    FROM CALIBRATION_LIBRARY
                    WHERE KIND = 'dark'
                      AND XBINNING = ?
                      AND EXPTIME = ?
                      AND COALESCE(GAIN, 0) = ?
                      AND COALESCE(FILTER_NAME, '') = ''
                      AND CCD_TEMP IS NOT NULL
                      AND ABS(CCD_TEMP - ?) <= ?
                      AND ID_EQUIPMENTS = ?
                      AND ID_TELESCOPE = ?
                """
            else:
                params = [int(xb), int(gain), flt, eq_id, tel_id]
                query = """
                    SELECT FILE_PATH, CCD_TEMP
                    FROM CALIBRATION_LIBRARY
                    WHERE KIND = 'flat'
                      AND XBINNING = ?
                      AND COALESCE(GAIN, 0) = ?
                      AND FILTER_NAME = ?
                      AND ID_EQUIPMENTS = ?
                      AND ID_TELESCOPE = ?
                """
            cur = self.conn.execute(query, tuple(params))
            return cur.fetchall()

        def _score(rows_in: list[sqlite3.Row]) -> str | None:
            scored: list[tuple[float, float, str]] = []
            for row in rows_in:
                p = Path(str(row["FILE_PATH"]))
                if not p.is_file():
                    continue
                if k == "dark":
                    tdb = row["CCD_TEMP"]
                    if tdb is None:
                        continue
                    try:
                        tdelta = abs(float(tdb) - float(light_temp))
                    except (TypeError, ValueError):
                        continue
                else:
                    tdelta = 0.0
                try:
                    mtime = -float(p.stat().st_mtime)
                except OSError:
                    mtime = 0.0
                scored.append((tdelta, mtime, str(p)))
            if not scored:
                return None
            scored.sort(key=lambda x: (x[0], x[1]))
            return scored[0][2]

        def _pick(xb: int) -> str | None:
            return _score(_query_rows(xb))

        if prefer_unbinned_master and int(xbinning) > 1:
            hit1 = _pick(1)
            if hit1 is not None:
                return hit1
        hit = _pick(int(xbinning))
        if hit is None:
            try:
                if k == "dark":
                    log_event(
                        f"CALIB LIB: no scoped master for dark eq={eq_id} tel={tel_id} "
                        f"bin={int(xbinning)} exp={float(exptime):g} gain={int(gain)} "
                        f"temp={float(light_temp):g}+-{tol:g}C"
                    )
                else:
                    log_event(
                        f"CALIB LIB: no scoped master for flat eq={eq_id} tel={tel_id} "
                        f"bin={int(xbinning)} gain={int(gain)} filter={flt!r}"
                    )
            except (TypeError, ValueError) as exc:  # noqa: BLE001
                pass
        return hit

    def fetch_obs_draft_telescope_equipment(self, draft_id: int) -> dict[str, Any] | None:
        """Telescope + sensor summary for UI (manifest rig ids)."""
        row = self.fetch_obs_draft_by_id(int(draft_id))
        if row is None:
            return None
        eq_id = self.get_draft_equipment_id(int(draft_id))
        tel_id = self.get_draft_telescope_id(int(draft_id))
        if tel_id is None:
            return {
                "draft_id": int(draft_id),
                "telescope_name": None,
                "telescope_focal_mm": None,
                "equipment_name": None,
                "pixel_um": None,
            }
        names = self.conn.execute(
            """
            SELECT
                t.TELESCOPENAME AS telescope_name,
                t.FOCAL AS telescope_focal_mm,
                e.CAMERANAME AS equipment_name,
                e.PIXELSIZE AS pixel_um
            FROM TELESCOPE t
            LEFT JOIN EQUIPMENTS e ON e.ID = ?
            WHERE t.ID = ?;
            """,
            (eq_id, tel_id),
        ).fetchone()
        if names is None:
            return {
                "draft_id": int(draft_id),
                "telescope_name": None,
                "telescope_focal_mm": None,
                "equipment_name": None,
                "pixel_um": None,
            }
        return {
            "draft_id": int(draft_id),
            "telescope_name": names["telescope_name"],
            "telescope_focal_mm": names["telescope_focal_mm"],
            "equipment_name": names["equipment_name"],
            "pixel_um": names["pixel_um"],
        }

    @staticmethod
    def _normalize_calibration_library_filter_name(flt: str | None) -> str:
        s = str(flt or "").strip()
        if not s or s.lower() in {"unknown", "none", "nan"}:
            return "NoFilter"
        return s

    def calibration_library_has_flat_for_filter(self, filter_name: str | None) -> bool:
        """True if CALIBRATION_LIBRARY has at least one flat row for this filter name."""
        flt = self._normalize_calibration_library_filter_name(filter_name)
        row = self.conn.execute(
            """
            SELECT 1 FROM CALIBRATION_LIBRARY
            WHERE KIND = 'flat' AND FILTER_NAME = ?
            LIMIT 1;
            """,
            (flt,),
        ).fetchone()
        return row is not None

    def set_obs_draft_calibration_mode(self, draft_id: int, calibration_mode: str) -> None:
        from draft_provenance import patch_draft_manifest, resolve_draft_dir_for_id

        root = resolve_draft_dir_for_id(self, int(draft_id))
        if root is None:
            raise ValueError(f"Draft '{draft_id}' not found for calibration mode update.")
        patch_draft_manifest(root, int(draft_id), calibration_mode=str(calibration_mode))

    def _ensure_active_columns(self) -> None:
        """Schema migration: add ACTIVE columns to EQUIPMENTS/TELESCOPE."""
        # EQUIPMENTS
        cursor = self.conn.execute("PRAGMA table_info('EQUIPMENTS');")
        eq_cols = {row["name"] for row in cursor.fetchall()}
        if "ACTIVE" not in eq_cols:
            self.conn.execute("ALTER TABLE EQUIPMENTS ADD COLUMN ACTIVE TEXT DEFAULT 'YES';")
            self.conn.execute("UPDATE EQUIPMENTS SET ACTIVE = 'YES' WHERE ACTIVE IS NULL;")

        # TELESCOPE
        cursor = self.conn.execute("PRAGMA table_info('TELESCOPE');")
        tel_cols = {row["name"] for row in cursor.fetchall()}
        if "ACTIVE" not in tel_cols:
            self.conn.execute("ALTER TABLE TELESCOPE ADD COLUMN ACTIVE TEXT DEFAULT 'YES';")
            self.conn.execute("UPDATE TELESCOPE SET ACTIVE = 'YES' WHERE ACTIVE IS NULL;")

        self.conn.commit()

    #: Designated default rows for the reference tables (Phase 2). These are the
    #: EXPLICIT user markers seeded once; they are NOT a silent ``id=1`` fallback.
    #: Equipment=QHY294MM(1), Telescope=Carl-Zeiss 200mm(1), Location=Jirny(2).
    _DEFAULT_SEED_IDS: ClassVar[dict[str, int]] = {
        "EQUIPMENTS": 1,
        "TELESCOPE": 1,
        "LOCATION": 2,
    }

    def _ensure_is_default_columns(self) -> None:
        """Schema migration (Phase 2): ``IS_DEFAULT`` on EQUIPMENTS/TELESCOPE/LOCATION
        (exactly one TRUE each) and an ``ACTIVE`` column on LOCATION.

        ``IS_DEFAULT`` is an explicit user marker that pre-selects a row in the Scan
        Source UI - it does not silently override per-draft / header resolution.
        """
        for table in ("EQUIPMENTS", "TELESCOPE", "LOCATION"):
            cols = {r["name"] for r in self.conn.execute(f"PRAGMA table_info('{table}');").fetchall()}
            if "IS_DEFAULT" not in cols:
                self.conn.execute(f"ALTER TABLE {table} ADD COLUMN IS_DEFAULT INTEGER DEFAULT 0;")
                self.conn.execute(f"UPDATE {table} SET IS_DEFAULT = 0 WHERE IS_DEFAULT IS NULL;")

        loc_cols = {r["name"] for r in self.conn.execute("PRAGMA table_info('LOCATION');").fetchall()}
        if "ACTIVE" not in loc_cols:
            self.conn.execute("ALTER TABLE LOCATION ADD COLUMN ACTIVE TEXT DEFAULT 'YES';")
            self.conn.execute("UPDATE LOCATION SET ACTIVE = 'YES' WHERE ACTIVE IS NULL;")

        self.conn.commit()
        self._seed_table_defaults()

    def _seed_table_defaults(self) -> None:
        """Seed exactly one ``IS_DEFAULT = 1`` row per reference table (only if none set)."""
        for table, seed_id in self._DEFAULT_SEED_IDS.items():
            n_default = int(
                self.conn.execute(f"SELECT COUNT(*) FROM {table} WHERE IS_DEFAULT = 1;").fetchone()[0]
            )
            if n_default == 1:
                continue
            if n_default > 1:
                # Collapse to a single default (keep the lowest ID that is already marked).
                keep = self.conn.execute(
                    f"SELECT MIN(ID) FROM {table} WHERE IS_DEFAULT = 1;"
                ).fetchone()[0]
                self.set_table_default(table, int(keep))
                continue
            # No default yet: prefer the explicitly designated seed row, else lowest ID.
            row = self.conn.execute(
                f"SELECT ID FROM {table} WHERE ID = ?;", (int(seed_id),)
            ).fetchone()
            if row is None:
                row = self.conn.execute(f"SELECT MIN(ID) AS ID FROM {table};").fetchone()
                if row is None or row["ID"] is None:
                    LOGGER.warning("[DATABASE] %s empty - cannot seed IS_DEFAULT.", table)
                    continue
                LOGGER.warning(
                    "[DATABASE] %s has no row id=%s for IS_DEFAULT seed; using id=%s instead.",
                    table,
                    seed_id,
                    int(row["ID"]),
                )
            self.set_table_default(table, int(row["ID"]))

    def set_table_default(self, table: str, row_id: int) -> None:
        """Mark ``row_id`` as the single ``IS_DEFAULT = 1`` row in ``table`` (exclusive)."""
        if table not in _EDITABLE_DEFAULT_TABLES:
            raise ValueError(f"Refusing to edit non-allowlisted table: {table!r}")
        try:
            self.conn.execute("BEGIN;")
            self.conn.execute(
                f"UPDATE {table} SET IS_DEFAULT = CASE WHEN ID = ? THEN 1 ELSE 0 END;",
                (int(row_id),),
            )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def get_default_id(self, table: str) -> int | None:
        """Return the ``IS_DEFAULT = 1`` row id for ``table``, or ``None`` if unset."""
        allowed = {"EQUIPMENTS", "TELESCOPE", "LOCATION"}
        if table not in allowed:
            raise ValueError(f"get_default_id: table not allowed: {table}")
        try:
            row = self.conn.execute(
                f"SELECT ID FROM {table} WHERE IS_DEFAULT = 1 ORDER BY ID LIMIT 1;"
            ).fetchone()
        except sqlite3.Error as exc:  # noqa: BLE001
            return None
        return int(row["ID"]) if row is not None and row["ID"] is not None else None

    def _ensure_equipments_saturate_adu_column(self) -> None:
        """Schema migration: linear saturation ceiling (ADU) per camera/equipment row."""
        cursor = self.conn.execute("PRAGMA table_info('EQUIPMENTS');")
        cols = {row["name"] for row in cursor.fetchall()}
        if "SATURATE_ADU" in cols:
            return
        self.conn.execute("ALTER TABLE EQUIPMENTS ADD COLUMN SATURATE_ADU REAL;")
        self.conn.commit()

    def _migrate_qhy294mm_saturate_adu_null(self) -> None:
        """Idempotent: QHY294MM (id=1) wrong binned SATURATE_ADU=16384 -> NULL (SAT-DIAG derives)."""
        try:
            row = self.conn.execute(
                "SELECT SATURATE_ADU FROM EQUIPMENTS WHERE ID = 1;"
            ).fetchone()
        except sqlite3.Error:
            return
        if row is None:
            return
        v = row["SATURATE_ADU"]
        if v is None:
            return
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return
        if abs(fv - 16384.0) < 0.5:
            self.conn.execute(
                "UPDATE EQUIPMENTS SET SATURATE_ADU = NULL WHERE ID = 1 AND SATURATE_ADU = ?;",
                (fv,),
            )
            self.conn.commit()

    def _ensure_equipments_cosmic_columns(self) -> None:
        """Detector gain and read noise (e-/ADU, read noise e-) for photometric error / SNR."""
        cursor = self.conn.execute("PRAGMA table_info('EQUIPMENTS');")
        cols = {row["name"] for row in cursor.fetchall()}
        if "GAIN_ADU" not in cols:
            self.conn.execute("ALTER TABLE EQUIPMENTS ADD COLUMN GAIN_ADU REAL;")
        if "READNOISE_E" not in cols:
            self.conn.execute("ALTER TABLE EQUIPMENTS ADD COLUMN READNOISE_E REAL;")
        self.conn.commit()

    def _ensure_equipments_bayermask_column(self) -> None:
        """OSC Bayer mask authority: RGGB/BGGR/GBRG/GRBG, mono, or empty (mono)."""
        cursor = self.conn.execute("PRAGMA table_info('EQUIPMENTS');")
        cols = {row["name"] for row in cursor.fetchall()}
        if "BAYERMASK" not in cols:
            self.conn.execute("ALTER TABLE EQUIPMENTS ADD COLUMN BAYERMASK TEXT;")
            self.conn.commit()
        try:
            row = self.conn.execute(
                "SELECT ID, BAYERMASK FROM EQUIPMENTS WHERE ID = 5;"
            ).fetchone()
            if row is not None and (row["BAYERMASK"] is None or str(row["BAYERMASK"]).strip() == ""):
                self.conn.execute(
                    "UPDATE EQUIPMENTS SET BAYERMASK = 'RGGB' WHERE ID = 5;"
                )
                self.conn.commit()
        except sqlite3.Error:
            pass

    def get_equipment_bayermask(self, equipment_id: int | None) -> str | None:
        """Return canonical Bayer mask or None (mono/empty)."""
        if equipment_id is None:
            return None
        try:
            row = self.conn.execute(
                "SELECT BAYERMASK FROM EQUIPMENTS WHERE ID = ?;",
                (int(equipment_id),),
            ).fetchone()
        except sqlite3.Error:
            return None
        if not row or row["BAYERMASK"] in (None, ""):
            return None
        from osc_extract import normalize_bayermask

        try:
            return normalize_bayermask(str(row["BAYERMASK"]))
        except ValueError:
            return None

    def _table_exists(self, table: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1;",
            (str(table),),
        ).fetchone()
        return row is not None

    def _table_has_fk_to(self, table: str, ref_table: str) -> bool:
        try:
            for row in self.conn.execute(f"PRAGMA foreign_key_list('{table}');"):
                if str(row[2]).upper() == str(ref_table).upper():
                    return True
        except sqlite3.Error:
            return False
        return False

    def _heal_qc_orphan_old_tables(self) -> None:
        """Drop stale *_OLD QC tables or recover when only *_OLD survived a crash."""
        for base in ("OBS_QC_PROCESSING_RUN", "OBS_QC_PROCESSING_FILE"):
            old = f"{base}_OLD"
            if not self._table_exists(old):
                continue
            if self._table_exists(base):
                self.conn.execute(f"DROP TABLE IF EXISTS {old};")
            else:
                self.conn.execute(f"ALTER TABLE {old} RENAME TO {base};")

    def _qc_file_fk_parent(self) -> str | None:
        """Return the parent table name referenced by OBS_QC_PROCESSING_FILE.RUN_ID, if any."""
        if not self._table_exists("OBS_QC_PROCESSING_FILE"):
            return None
        try:
            for row in self.conn.execute("PRAGMA foreign_key_list('OBS_QC_PROCESSING_FILE');"):
                # row: id, seq, table, from, to, on_update, on_delete, match
                if str(row[3]).upper() == "RUN_ID":
                    return str(row[2])
        except sqlite3.Error:
            return None
        return None

    def _qc_file_needs_run_fk_rebuild(self) -> bool:
        """True when FILE FK points at a missing/wrong parent (e.g. OBS_QC_PROCESSING_RUN_OLD)."""
        parent = self._qc_file_fk_parent()
        if parent is None:
            # No FK recorded: still rebuild if orphan *_OLD naming is present historically
            return False
        return str(parent).upper() != "OBS_QC_PROCESSING_RUN"

    def _drop_vestigial_tables(self) -> None:
        # WAVE-B STEP 5: SETTINGS held only masterdark/masterflat validity days (config.json-authoritative).
        # PHOTOMETRY_LIGHT_CURVE was never read; light curves are file-based CSVs.
        self.conn.execute("PRAGMA foreign_keys = OFF;")
        try:
            self.conn.execute("DROP TABLE IF EXISTS SCANNING;")
            self.conn.execute("DROP TABLE IF EXISTS OBSERVATION;")
            self.conn.execute("DROP TABLE IF EXISTS OBS_FILES;")
            self._rebuild_qc_tables_without_obs_draft_fk()
            self._drop_final_data_view()
            self.conn.execute("DROP TABLE IF EXISTS OBS_DRAFT;")
            self.conn.execute("DROP TABLE IF EXISTS SETTINGS;")
            self.conn.execute("DROP TABLE IF EXISTS PHOTOMETRY_LIGHT_CURVE;")
            self.conn.commit()
        finally:
            self.conn.execute("PRAGMA foreign_keys = ON;")

    def _qc_run_has_obs_draft_fk(self) -> bool:
        legacy_draft = "OBS" + "_DRAFT"
        if not self._table_exists("OBS_QC_PROCESSING_RUN"):
            return False
        return self._table_has_fk_to("OBS_QC_PROCESSING_RUN", legacy_draft)

    def _qc_file_has_obs_draft_fk(self) -> bool:
        legacy_draft = "OBS" + "_DRAFT"
        if not self._table_exists("OBS_QC_PROCESSING_FILE"):
            return False
        return self._table_has_fk_to("OBS_QC_PROCESSING_FILE", legacy_draft)

    def _rebuild_qc_tables_without_obs_draft_fk(self) -> None:
        """Remove legacy draft-table FK from QC tables on older databases.

        Also rebuilds OBS_QC_PROCESSING_FILE when its RUN_ID FK still points at
        OBS_QC_PROCESSING_RUN_OLD (half-finished migration; TARGET-DEPTH-01).
        """
        self._heal_qc_orphan_old_tables()
        run_needs = self._qc_run_has_obs_draft_fk()
        file_needs = self._qc_file_has_obs_draft_fk() or self._qc_file_needs_run_fk_rebuild()
        if not run_needs and not file_needs:
            return
        if run_needs:
            self._rebuild_table_safely(
                "OBS_QC_PROCESSING_RUN",
                """
                CREATE TABLE OBS_QC_PROCESSING_RUN (
                    ID INTEGER PRIMARY KEY AUTOINCREMENT,
                    PROCESSING_HASH TEXT NOT NULL UNIQUE,
                    DRAFT_ID INTEGER NOT NULL,
                    CREATED_AT TEXT NOT NULL
                );
                """,
                """
                INSERT INTO OBS_QC_PROCESSING_RUN (ID, PROCESSING_HASH, DRAFT_ID, CREATED_AT)
                SELECT ID, PROCESSING_HASH, DRAFT_ID, CREATED_AT FROM OBS_QC_PROCESSING_RUN_OLD;
                """,
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS IDX_QC_PROC_RUN_DRAFT ON OBS_QC_PROCESSING_RUN (DRAFT_ID);"
            )
            file_needs = True
        if file_needs and self._table_exists("OBS_QC_PROCESSING_FILE"):
            self._rebuild_table_safely(
                "OBS_QC_PROCESSING_FILE",
                """
                CREATE TABLE OBS_QC_PROCESSING_FILE (
                    ID INTEGER PRIMARY KEY AUTOINCREMENT,
                    RUN_ID INTEGER NOT NULL,
                    SOURCE_OBS_FILE_ID INTEGER NOT NULL,
                    FILE_PATH TEXT,
                    FILTER TEXT,
                    EXPTIME REAL,
                    INSPECTION_JD REAL,
                    FWHM REAL,
                    DRIFT REAL,
                    FOREIGN KEY (RUN_ID) REFERENCES OBS_QC_PROCESSING_RUN (ID) ON DELETE CASCADE
                );
                """,
                """
                INSERT INTO OBS_QC_PROCESSING_FILE (
                    ID, RUN_ID, SOURCE_OBS_FILE_ID, FILE_PATH, FILTER, EXPTIME,
                    INSPECTION_JD, FWHM, DRIFT
                )
                SELECT
                    ID, RUN_ID, SOURCE_OBS_FILE_ID, FILE_PATH, FILTER, EXPTIME,
                    INSPECTION_JD, FWHM, DRIFT
                FROM OBS_QC_PROCESSING_FILE_OLD;
                """,
            )

    def _drop_equipments_focal_column(self) -> None:
        """One-time migration: focal length belongs on TELESCOPE, not EQUIPMENTS."""
        cursor = self.conn.execute("PRAGMA table_info('EQUIPMENTS');")
        cols = {row["name"] for row in cursor.fetchall()}
        if "FOCAL" not in cols:
            return
        self._rebuild_table_safely(
            "EQUIPMENTS",
            """
            CREATE TABLE EQUIPMENTS (
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                CAMERANAME TEXT,
                ALIAS TEXT,
                SENSORTYPE TEXT,
                SENSORSIZE TEXT,
                PIXELSIZE REAL,
                ACTIVE TEXT DEFAULT 'YES',
                SATURATE_ADU REAL,
                GAIN_ADU REAL,
                READNOISE_E REAL,
                BAYERMASK TEXT,
                IS_DEFAULT INTEGER DEFAULT 0
            );
            """,
            """
            INSERT INTO EQUIPMENTS (
                ID, CAMERANAME, ALIAS, SENSORTYPE, SENSORSIZE, PIXELSIZE, ACTIVE,
                SATURATE_ADU, GAIN_ADU, READNOISE_E, BAYERMASK, IS_DEFAULT
            )
            SELECT
                ID, CAMERANAME, ALIAS, SENSORTYPE, SENSORSIZE, PIXELSIZE, ACTIVE,
                SATURATE_ADU, GAIN_ADU, READNOISE_E, BAYERMASK, IS_DEFAULT
            FROM EQUIPMENTS_OLD;
            """,
        )

    def initialize_database(self) -> None:
        """Reference tables are created (schema) but intentionally EMPTY on a fresh DB.

        Product decision (DB-SEED-SPLIT, 2026-07-18): a new user's fresh database must
        be empty -- they create their own Location / Telescope / Equipment in Settings.
        The author's observatory rows are a REFERENCE FIXTURE for the anchor/test
        machinery, not product content; they were moved out of the runtime package to
        ``dev/tools/reference_seed.py::seed_reference_observatory(db)``. The ``--full``
        anchor harness and pytest fixtures that need the anchor context call that helper
        explicitly (it uses ``INSERT OR IGNORE``, so it is a no-op on an already
        populated DB such as the author's production DB).

        The schema itself is created by ``_create_tables`` (run just before this in
        ``__init__``); this method is deliberately a no-op kept for API stability and
        as the documented seam where seeding used to live.
        """
        # Intentionally no rows: fresh reference tables stay empty.
        return

    @staticmethod
    def _jd_to_yyyymmdd(jd: float) -> str:
        """Convert Julian date to UTC date string YYYYMMDD.

        Robust against missing/invalid JD values (falls back to today's UTC date).
        """
        try:
            jd_f = float(jd)
        except (TypeError, ValueError):
            return datetime.now(timezone.utc).strftime("%Y%m%d")

        # JD sanity bounds (very permissive); guard against 0/NaN/inf.
        if (not math.isfinite(jd_f)) or jd_f < 2_000_000 or jd_f > 3_500_000:
            return datetime.now(timezone.utc).strftime("%Y%m%d")

        unix_seconds = (jd_f - 2440587.5) * 86400.0
        try:
            dt_utc = datetime.fromtimestamp(unix_seconds, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return datetime.now(timezone.utc).strftime("%Y%m%d")
        return dt_utc.strftime("%Y%m%d")

    @staticmethod
    def generate_hashtag(
        id_equipments: int,
        id_telescope: int,
        id_location: int,
        id_scanning: int,
        center_of_field_ra: float,
        center_of_field_de: float,
        observation_start_jd: float,
    ) -> str:
        """Create deterministic observation id in format YYYYMMDD_HASH."""
        payload = (
            f"{id_equipments}|{id_telescope}|{id_location}|{id_scanning}|"
            f"{center_of_field_ra:.8f}|{center_of_field_de:.8f}|{observation_start_jd:.8f}"
        )
        digest = hashlib.md5(payload.encode("utf-8"), usedforsecurity=False).hexdigest()[:6]
        date_prefix = VyvarDatabase._jd_to_yyyymmdd(observation_start_jd)
        return f"{date_prefix}_{digest}"

    def insert_equipment(
        self,
        camera_name: str,
        alias: str,
        sensor_type: str,
        sensor_size: str,
        pixel_size: float,
    ) -> int:
        cursor = self.conn.execute(
            """
            INSERT INTO EQUIPMENTS (CAMERANAME, ALIAS, SENSORTYPE, SENSORSIZE, PIXELSIZE)
            VALUES (?, ?, ?, ?, ?);
            """,
            (camera_name, alias, sensor_type, sensor_size, pixel_size),
        )
        self.conn.commit()
        return int(cursor.lastrowid)

    def get_equipment_saturation_adu(self, equipment_id: int) -> float | None:
        """Return EQUIPMENTS.SATURATE_ADU if set and positive; else None."""
        cursor = self.conn.execute(
            "SELECT SATURATE_ADU FROM EQUIPMENTS WHERE ID = ?;",
            (int(equipment_id),),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        v = row["SATURATE_ADU"]
        if v is None:
            return None
        try:
            f = float(v)
            return f if f > 0 and math.isfinite(f) else None
        except (TypeError, ValueError):
            return None

    def get_equipment_pixel_size_um(self, equipment_id: int) -> float | None:
        """``EQUIPMENTS.PIXELSIZE``: native (1x1) pixel pitch [um], if set and positive.

        With ``TELESCOPE.FOCAL`` and X/Y binning from the FITS header, the pipeline derives
        ``expected_scale`` (arcsec/px) for plate solve and catalog geometry.
        """
        cursor = self.conn.execute(
            "SELECT PIXELSIZE FROM EQUIPMENTS WHERE ID = ?;",
            (int(equipment_id),),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        v = row["PIXELSIZE"]
        if v is None:
            return None
        try:
            f = float(v)
            return f if f > 0 and math.isfinite(f) else None
        except (TypeError, ValueError):
            return None

    def get_equipment_cosmic_params(self, equipment_id: int) -> tuple[float | None, float | None]:
        """Return ``(GAIN_ADU, READNOISE_E)`` from EQUIPMENTS when set (positive, finite)."""
        cursor = self.conn.execute(
            "SELECT GAIN_ADU, READNOISE_E FROM EQUIPMENTS WHERE ID = ?;",
            (int(equipment_id),),
        )
        row = cursor.fetchone()
        if row is None:
            return None, None

        def _pos(x: Any) -> float | None:
            if x is None:
                return None
            try:
                f = float(x)
                return f if f > 0 and math.isfinite(f) else None
            except (TypeError, ValueError):
                return None

        return _pos(row["GAIN_ADU"]), _pos(row["READNOISE_E"])

    def set_equipment_cosmic_params(
        self,
        equipment_id: int,
        gain: float,
        read_noise: float,
    ) -> None:
        """Persist detector gain [e-/ADU] and read noise [e-] on EQUIPMENTS.

        ``READNOISE_E`` is per-pixel at bin1; param_resolver scales *bin for software-summed binning.
        """
        try:
            g = float(gain)
        except (TypeError, ValueError):
            g = float("nan")
        try:
            rn = float(read_noise)
        except (TypeError, ValueError):
            rn = float("nan")
        g_store = g if math.isfinite(g) and g > 0 else None
        rn_store = rn if math.isfinite(rn) and rn > 0 else None
        self.conn.execute(
            "UPDATE EQUIPMENTS SET GAIN_ADU = ?, READNOISE_E = ? WHERE ID = ?;",
            (g_store, rn_store, int(equipment_id)),
        )
        self.conn.commit()

    def get_telescope_focal_mm(self, telescope_id: int | None = None) -> float | None:
        """Positive ``TELESCOPE.FOCAL`` [mm]. If ``telescope_id`` is None, first active row with FOCAL > 0."""
        if telescope_id is not None:
            cursor = self.conn.execute(
                "SELECT FOCAL FROM TELESCOPE WHERE ID = ?;",
                (int(telescope_id),),
            )
        else:
            act = self.sql_expr_active_is_true("ACTIVE")
            cursor = self.conn.execute(
                f"""
                SELECT FOCAL FROM TELESCOPE
                WHERE {act}
                  AND FOCAL IS NOT NULL AND FOCAL > 0
                ORDER BY ID
                LIMIT 1;
                """
            )
        row = cursor.fetchone()
        if row is None:
            return None
        v = row["FOCAL"]
        if v is None:
            return None
        try:
            f = float(v)
            return f if f > 0 and math.isfinite(f) else None
        except (TypeError, ValueError):
            return None

    def get_combined_metadata(self, file_path: str | Path, draft_id: int) -> dict[str, Any]:
        """Merge FITS primary header with ``draft manifest`` -> ``TELESCOPE`` / ``EQUIPMENTS`` SQL fallbacks.

        - Focal: plausible FITS keywords first; else ``TELESCOPE.FOCAL`` for the draft telescope.
        - Native pixel [um]: FITS ``PIXSIZE*`` / ``XPIXSZ`` ...; else ``EQUIPMENTS.PIXELSIZE`` via draft join
          (same as ``get_equipment_pixel_size_um`` on ``draft manifest.ID_EQUIPMENTS``).
        - **Binning:** ``XBINNING`` / ``BINNING`` (supports ``2x2`` strings); if header says 1x1 but
          ``NAXIS`` matches ``EQUIPMENTS.SENSORSIZE`` at 2x/3x/4x, binning is inferred. Effective pixel
          is ``native_pixel_um * XBINNING``.
        - ``SATURATE_ADU`` from ``EQUIPMENTS`` for the draft camera when set.

        Schema note: this project uses tables ``TELESCOPE`` / ``FOCAL`` and ``EQUIPMENTS`` / ``PIXELSIZE``,
        not ``TELESCOPES`` / ``FOCAL_LENGTH_MM`` / ``PIXEL_UM``.
        """
        fp = Path(file_path)
        did = int(draft_id)
        if self._draft_rig_resolve_row(did) is None:
            raise ValueError(f"Draft id={did} not found.")

        id_eq = self.get_draft_equipment_id(did)
        id_tel = self.get_draft_telescope_id(did)

        with fits.open(fp, memmap=False) as hdul:
            header = hdul[0].header.copy()

        x_bin = _db_xbinning_strict(header)
        y_bin = _db_ybinning_header(header, x_bin)

        from utils import infer_binning_xy_from_sensor_shape, parse_sensor_naxis_from_text

        naxis1 = int(header.get("NAXIS1") or 0)
        naxis2 = int(header.get("NAXIS2") or 0)
        native_size: tuple[int, int] | None = None
        if id_eq is not None:
            row_ss = self.conn.execute(
                "SELECT SENSORSIZE FROM EQUIPMENTS WHERE ID = ?;",
                (int(id_eq),),
            ).fetchone()
            if row_ss is not None:
                native_size = parse_sensor_naxis_from_text(row_ss["SENSORSIZE"])
        x_inf, y_inf, inferred = infer_binning_xy_from_sensor_shape(
            naxis1, naxis2, native_size, (x_bin, y_bin)
        )
        if inferred:
            log_event(
                f"BINNING: FITS hlavicka {x_bin}x{y_bin}, NAXIS {naxis1}x{naxis2}, "
                f"senzor {native_size} -> pouzite {x_inf}x{y_inf} (odvodene z rozmeru)."
            )
            x_bin, y_bin = x_inf, y_inf

        focal_mm = _db_header_focal_length_mm(header)
        _f_hdr_ok = focal_mm is not None and _db_focal_plausible_mm(float(focal_mm))
        focal_src = "fits_header" if _f_hdr_ok else "none"
        if not _f_hdr_ok:
            focal_mm = None
            if id_tel is not None:
                try:
                    tf = self.get_telescope_focal_mm(int(id_tel))
                except Exception:  # noqa: BLE001
                    tf = None
                if tf is not None:
                    n2, _ = normalize_telescope_focal_mm_for_plate_scale(float(tf))
                    if _db_focal_plausible_mm(n2):
                        focal_mm = float(n2)
                        focal_src = "telescope_focal_sql"

        native_um = _db_header_pixel_native_um_mean(header)
        pix_src = "fits_header"
        if native_um is None or not math.isfinite(native_um) or native_um <= 0 or native_um > 300.0:
            native_um = None
            pix_src = "none"
            if id_eq is not None:
                try:
                    native_um = self.get_equipment_pixel_size_um(int(id_eq))
                except Exception:  # noqa: BLE001
                    native_um = None
                if native_um is not None:
                    pix_src = "equipment_pixelsize_sql"

        pixel_effective_um: float | None = None
        if native_um is not None and math.isfinite(float(native_um)) and float(native_um) > 0:
            pixel_effective_um = float(native_um) * float(x_bin)

        sat: float | None = None
        if id_eq is not None:
            try:
                sat = self.get_equipment_saturation_adu(int(id_eq))
            except Exception:  # noqa: BLE001
                sat = None

        if focal_mm is not None:
            n_f, _ = normalize_telescope_focal_mm_for_plate_scale(float(focal_mm))
            if _db_focal_plausible_mm(n_f):
                focal_mm = float(n_f)

        return {
            "draft_id": did,
            "id_equipments": None if id_eq is None else int(id_eq),
            "id_telescope": None if id_tel is None else int(id_tel),
            "xbinning": int(x_bin),
            "ybinning": int(y_bin),
            "focal_length_mm": focal_mm,
            "focal_source": focal_src,
            "pixel_native_um": float(native_um) if native_um is not None else None,
            "pixel_effective_um": pixel_effective_um,
            "pixel_source": pix_src,
            "saturate_adu": sat,
        }

    def insert_telescope(
        self,
        telescope_name: str,
        alias: str,
        diameter: float,
        focal: float,
    ) -> int:
        cursor = self.conn.execute(
            """
            INSERT INTO TELESCOPE (TELESCOPENAME, ALIAS, DIAMETER, FOCAL)
            VALUES (?, ?, ?, ?);
            """,
            (telescope_name, alias, diameter, focal),
        )
        self.conn.commit()
        return int(cursor.lastrowid)

    def insert_location(
        self,
        place_name: str,
        latitude: float,
        longitude: float,
        altitude: float,
    ) -> int:
        cursor = self.conn.execute(
            """
            INSERT INTO LOCATION (PLACENAME, LATITUDE, LONGITUDE, ALTITUDE)
            VALUES (?, ?, ?, ?);
            """,
            (place_name, latitude, longitude, altitude),
        )
        self.conn.commit()
        return int(cursor.lastrowid)

    def derive_scanning_id(self, metadata: dict[str, Any]) -> int:
        """Stable scanning id from FITS metadata (no SCANNING SQL table)."""
        from draft_provenance import derive_scanning_id as _derive_scanning_id

        return int(_derive_scanning_id(metadata))

    def _fk_row_exists(self, table: str, row_id: int) -> bool:
        if table not in ("EQUIPMENTS", "TELESCOPE", "LOCATION"):
            raise ValueError(f"unsupported FK table: {table!r}")
        row = self.conn.execute(
            f"SELECT 1 FROM {table} WHERE ID = ? LIMIT 1;",
            (int(row_id),),
        ).fetchone()
        return row is not None

    def _validate_observation_foreign_keys(
        self,
        *,
        id_equipments: int,
        id_telescope: int,
        id_location: int,
        id_scanning: int,
    ) -> None:
        checks = (
            ("EQUIPMENTS", int(id_equipments), "equipment/camera"),
            ("TELESCOPE", int(id_telescope), "telescope"),
            ("LOCATION", int(id_location), "observatory location"),
        )
        missing = [
            f"{label} (id={rid})"
            for table, rid, label in checks
            if not self._fk_row_exists(table, rid)
        ]
        if missing:
            raise ValueError(
                "Cannot create observation draft: missing database row(s): "
                + ", ".join(missing)
                + ". Add the required rows in Database Explorer or import with valid FITS metadata."
            )

    def resolve_import_location_id(
        self,
        *,
        id_location: int | None,
        cfg_location_id: int,
    ) -> tuple[int, str | None]:
        """Return a valid ``LOCATION.ID`` for import (delegates to unified resolver)."""
        from observer_location import resolve_observer_location_for_run

        class _CfgShim:
            observer_location_id = int(cfg_location_id)

        resolved = resolve_observer_location_for_run(
            self.db_path,
            explicit_location_id=id_location,
            cfg=_CfgShim(),
            source_hint="cli_arg" if id_location is not None else None,
        )
        return int(resolved.location_id), None

    def get_equipments(self, *, active_only: bool = False) -> list[dict[str, Any]]:
        sql = """
            SELECT ID, CAMERANAME, ALIAS, ACTIVE, IS_DEFAULT
            FROM EQUIPMENTS
        """
        params: tuple[object, ...] = ()
        if active_only:
            sql += f" WHERE {self.sql_expr_active_is_true('ACTIVE')} "
        sql += " ORDER BY ID; "
        cursor = self.conn.execute(sql, params)
        return [dict(row) for row in cursor.fetchall()]

    def get_telescopes(self, *, active_only: bool = False) -> list[dict[str, Any]]:
        sql = """
            SELECT ID, TELESCOPENAME, ALIAS, ACTIVE, IS_DEFAULT
            FROM TELESCOPE
        """
        params: tuple[object, ...] = ()
        if active_only:
            sql += f" WHERE {self.sql_expr_active_is_true('ACTIVE')} "
        sql += " ORDER BY ID; "
        cursor = self.conn.execute(sql, params)
        return [dict(row) for row in cursor.fetchall()]

    def count_obs_files_for_observation(self, observation_id: str) -> int:
        from draft_provenance import count_manifest_files_for_observation

        return int(count_manifest_files_for_observation(self.resolve_archive_root(), str(observation_id)))

    def count_obs_files_for_draft(self, draft_id: int) -> int:
        from draft_provenance import count_manifest_files_for_draft

        return int(count_manifest_files_for_draft(self, int(draft_id)))

    def insert_draft_files(self, draft_id: int, files: list[dict[str, Any]]) -> None:
        """Insert per-file evidence into manifest ``files[]``."""
        from draft_provenance import set_draft_manifest_files

        set_draft_manifest_files(self, int(draft_id), files)

    def update_obs_file_qc_by_raw_light_path(
        self,
        raw_light_path: str | Path,
        *,
        draft_id: int | None = None,
        observation_id: str | None = None,
        qc_hfr: float | None = None,
        qc_stars: int | None = None,
        qc_background: float | None = None,
        qc_bg_rms: float | None = None,
        qc_passed: bool | None = None,
    ) -> int:
        """Update QC fields in manifest ``files[]`` for matching ``file_path``."""
        from draft_provenance import update_manifest_file_entry

        if draft_id is None:
            return 0
        p = str(Path(raw_light_path).resolve())
        qc: dict[str, Any] = {}
        if qc_hfr is not None and math.isfinite(float(qc_hfr)):
            qc["hfr"] = float(qc_hfr)
        if qc_stars is not None:
            qc["stars"] = int(qc_stars)
        if qc_background is not None and math.isfinite(float(qc_background)):
            qc["background"] = float(qc_background)
        if qc_bg_rms is not None and math.isfinite(float(qc_bg_rms)):
            qc["bg_rms"] = float(qc_bg_rms)
        if qc_passed is not None:
            qc["passed"] = 1 if qc_passed else 0
        if not qc:
            return 0
        _ = observation_id
        ok = update_manifest_file_entry(self, int(draft_id), file_path=p, qc=qc)
        return 1 if ok else 0

    def fetch_draft_light_rows_for_quality(self, draft_id: int) -> list[dict[str, Any]]:
        """Light frames for a draft (manifest-only)."""
        from draft_provenance import light_rows_from_manifest

        rows = light_rows_from_manifest(self, int(draft_id), imagetyp="light")
        return rows if rows is not None else []

    def update_obs_file_calibration_state_by_raw_light_path(
        self,
        raw_light_path: str | Path,
        *,
        draft_id: int | None = None,
        observation_id: str | None = None,
        is_calibrated: int | None = None,
        calib_type: str | None = None,
        calib_flags: str | None = None,
    ) -> int:
        """Update calibration fields in manifest ``files[]``."""
        from draft_provenance import update_manifest_file_entry

        if draft_id is None:
            return 0
        p = str(Path(raw_light_path).resolve())
        _ = observation_id
        ok = update_manifest_file_entry(
            self,
            int(draft_id),
            file_path=p,
            is_calibrated=1 if is_calibrated is not None and int(is_calibrated) else is_calibrated,
            calib_type=str(calib_type) if calib_type is not None else None,
            calib_flags=str(calib_flags) if calib_flags is not None else None,
        )
        return 1 if ok else 0

    def update_obs_file_cal_stage_by_raw_light_path(
        self,
        raw_light_path: str | Path,
        *,
        draft_id: int | None = None,
        observation_id: str | None = None,
        cal_stage: str,
        cal_datasum: str,
        cal_pstbg: float | None = None,
    ) -> int:
        """Update INV-CAL-02 stage fields in manifest ``files[]``."""
        from datetime import datetime, timezone

        from draft_provenance import update_manifest_file_entry

        if draft_id is None:
            return 0
        p = str(Path(raw_light_path).resolve())
        _ = observation_id
        payload: dict[str, Any] = {
            "cal_stage": str(cal_stage),
            "cal_datasum": str(cal_datasum),
            "cal_stage_ut": datetime.now(timezone.utc).isoformat(),
        }
        if cal_pstbg is not None:
            payload["cal_pstbg"] = float(cal_pstbg)
        ok = update_manifest_file_entry(
            self,
            int(draft_id),
            file_path=p,
            cal_stage=payload,
        )
        return 1 if ok else 0

    def update_obs_file_quality_by_id(
        self,
        draft_id: int,
        row_id: int,
        *,
        fwhm: float | None = None,
        sky_level: float | None = None,
        star_count: int | None = None,
        rejected_auto: int | None = None,
        inspection_jd: float | None = None,
        is_rejected: int | None = None,
        ra_deg: float | None = None,
        de_deg: float | None = None,
        exptime_sec: float | None = None,
        drift_arcmin: float | None = None,
        clear_drift: bool = False,
        drift_dra_deg: float | None = None,
        drift_dde_deg: float | None = None,
        roundness_mean: float | None = None,
        elongation_mean: float | None = None,
    ) -> None:
        """Update quality-inspection fields for one manifest ``files[]`` entry."""
        from draft_provenance import update_manifest_file_entry

        insp: dict[str, Any] = {}
        if clear_drift:
            insp.update({"drift": None, "drift_dra": None, "drift_dde": None})
        else:
            if drift_arcmin is not None and math.isfinite(float(drift_arcmin)) and float(drift_arcmin) >= 0.0:
                insp["drift"] = float(drift_arcmin)
            if drift_dra_deg is not None and math.isfinite(float(drift_dra_deg)):
                insp["drift_dra"] = float(drift_dra_deg)
            if drift_dde_deg is not None and math.isfinite(float(drift_dde_deg)):
                insp["drift_dde"] = float(drift_dde_deg)
        if fwhm is not None and math.isfinite(float(fwhm)):
            insp["fwhm"] = float(fwhm)
        if sky_level is not None and math.isfinite(float(sky_level)):
            insp["sky_level"] = float(sky_level)
        if star_count is not None:
            insp["star_count"] = int(star_count)
        if rejected_auto is not None:
            insp["rejected_auto"] = 1 if int(rejected_auto) else 0
        if inspection_jd is not None and math.isfinite(float(inspection_jd)):
            insp["inspection_jd"] = float(inspection_jd)
        if is_rejected is not None:
            insp["is_rejected"] = 1 if int(is_rejected) else 0
        if ra_deg is not None and math.isfinite(float(ra_deg)):
            insp["ra"] = float(ra_deg)
        if de_deg is not None and math.isfinite(float(de_deg)):
            insp["de"] = float(de_deg)
        if exptime_sec is not None and math.isfinite(float(exptime_sec)) and float(exptime_sec) >= 0:
            insp["exptime"] = float(exptime_sec)
        if roundness_mean is not None and math.isfinite(float(roundness_mean)) and float(roundness_mean) >= 0.0:
            insp["roundness_mean"] = float(roundness_mean)
        if elongation_mean is not None and math.isfinite(float(elongation_mean)) and float(elongation_mean) > 0.0:
            insp["elongation_mean"] = float(elongation_mean)
        if not insp:
            return
        update_manifest_file_entry(self, int(draft_id), obs_file_id=int(row_id), inspection=insp)

    def bulk_update_obs_file_is_rejected(
        self,
        draft_id: int,
        updates: list[tuple[int, int]],
    ) -> None:
        """``updates`` = list of (obs_file_id, is_rejected 0/1) in manifest."""
        from draft_provenance import bulk_update_manifest_is_rejected

        bulk_update_manifest_is_rejected(self, int(draft_id), updates)

    def _normalize_obs_file_path_key(self, p: str | Path) -> str:
        try:
            return str(Path(p).resolve()).casefold()
        except OSError:
            return str(p).casefold()

    def fetch_light_file_paths_not_rejected_for_draft(self, draft_id: int) -> set[str]:
        """Normalized FILE_PATH keys for draft lights with IS_REJECTED 0 or NULL (manifest)."""
        rows = self.fetch_draft_light_rows_for_quality(int(draft_id))
        out: set[str] = set()
        for row in rows:
            rej = row.get("IS_REJECTED")
            if rej not in (None, 0):
                continue
            fp = row.get("FILE_PATH")
            if fp:
                out.add(self._normalize_obs_file_path_key(str(fp)))
        return out

    def fetch_light_file_paths_not_rejected_for_observation(self, observation_id: str) -> set[str]:
        """Light paths for a finalized observation (manifest ``final_observation_id``)."""
        from draft_provenance import collect_manifest_obs_file_rows, iter_draft_archive_dirs

        out: set[str] = set()
        archive_root = self.resolve_archive_root()
        rows = collect_manifest_obs_file_rows(archive_root, observation_id=str(observation_id))
        for row in rows:
            if str(row.get("IMAGETYP") or "").strip().lower() != "light":
                continue
            rej = row.get("IS_REJECTED")
            if rej not in (None, 0):
                continue
            fp = row.get("FILE_PATH")
            if fp:
                out.add(self._normalize_obs_file_path_key(str(fp)))
        if out:
            return out
        for did, apath in iter_draft_archive_dirs(archive_root):
            _ = apath
            rows2 = self.fetch_draft_light_rows_for_quality(int(did))
            for row in rows2:
                rej = row.get("IS_REJECTED")
                if rej not in (None, 0):
                    continue
                fp = row.get("FILE_PATH")
                if fp:
                    out.add(self._normalize_obs_file_path_key(str(fp)))
        return out

    def create_draft(self, data: dict[str, Any]) -> int:
        """Create an ingestion draft (manifest-direct; no draft manifest SQL)."""
        id_equipments = int(data.get("id_equipments", 1))
        id_telescope = int(data.get("id_telescope", 1))
        id_location = int(data.get("id_location", 1))
        id_scanning = int(data.get("id_scanning", 1))
        self._validate_observation_foreign_keys(
            id_equipments=id_equipments,
            id_telescope=id_telescope,
            id_location=id_location,
            id_scanning=id_scanning,
        )
        from draft_provenance import create_draft_manifest

        archive_root = self.resolve_archive_root()
        return int(create_draft_manifest(self, archive_root, data))

    def update_draft_import_log(
        self,
        draft_id: int,
        *,
        lights_path: str,
        calib_path: str,
        imported_at: str,
        import_warnings: str | None = None,
        is_calibrated: bool | None = None,
        archive_path: str | None = None,
    ) -> None:
        from draft_provenance import patch_draft_manifest, resolve_draft_dir_for_id

        root = resolve_draft_dir_for_id(self, int(draft_id))
        if root is None and archive_path:
            root = Path(str(archive_path)).expanduser().resolve()
        if root is None:
            raise ValueError(f"Draft '{draft_id}' not found for import log update.")
        is_cal_int = None if is_calibrated is None else (1 if is_calibrated else 0)
        paths = {
            "lights": str(lights_path),
            "calib": str(calib_path),
            "archive": str(archive_path or root),
        }
        extra: dict[str, Any] = {"imported_at": imported_at}
        if import_warnings is not None:
            extra["import_warnings"] = import_warnings
        patch_draft_manifest(
            root,
            int(draft_id),
            paths=paths,
            is_calibrated=is_cal_int,
            extra=extra,
        )

    def finalize_draft_to_observation(
        self,
        draft_id: int,
        *,
        approved_by: str | None = None,
        notes: str | None = None,
    ) -> str:
        """Persist UI approval in manifest (``final_observation_id``, ``FINALIZED``)."""
        from astropy.time import Time
        from draft_provenance import patch_draft_manifest, resolve_draft_dir_for_id

        draft = self.fetch_obs_draft_by_id(int(draft_id))
        if draft is None:
            raise ValueError(f"Draft {draft_id} not found")
        st = str(draft.get("STATUS") or "").strip().upper()
        if st == "FINALIZED":
            existing = draft.get("FINAL_OBSERVATION_ID")
            if existing:
                return str(existing)
            raise ValueError(f"Draft {draft_id} is already finalized")

        approval_jd = float(Time.now().jd)
        ra_c = float(draft.get("CENTEROFFIELDRA") or 0.0)
        de_c = float(draft.get("CENTEROFFIELDDE") or 0.0)
        obs_id = self.generate_hashtag(
            id_equipments=int(draft.get("ID_EQUIPMENTS") or 0),
            id_telescope=int(draft.get("ID_TELESCOPE") or 0),
            id_location=int(draft.get("ID_LOCATION") or 0),
            id_scanning=int(draft.get("ID_SCANNING") or 0),
            center_of_field_ra=ra_c,
            center_of_field_de=de_c,
            observation_start_jd=float(draft.get("OBSERVATIONSTARTJD") or 0.0),
        )
        root = resolve_draft_dir_for_id(self, int(draft_id))
        if root is None:
            raise ValueError(f"Draft {draft_id}: manifest archive missing")
        patch_draft_manifest(
            root,
            int(draft_id),
            status="FINALIZED",
            final_observation_id=str(obs_id),
            extra={
                "approved_by": approved_by,
                "approval_notes": notes,
                "approval_jd": approval_jd,
            },
        )
        return str(obs_id)

    def _migrate_comp_library_tables(self) -> None:
        """Create FIELD_REGISTRY / COMP_STAR_LIBRARY and indexes (idempotent)."""
        if getattr(self, "_comp_library_tables_ready", False):
            return
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS FIELD_REGISTRY (
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                CENTER_RA_DEG REAL NOT NULL,
                CENTER_DE_DEG REAL NOT NULL,
                RADIUS_DEG REAL NOT NULL DEFAULT 1.0,
                OBJECT_NAME TEXT,
                MASTERSTAR_PATH TEXT,
                GRID_PATH TEXT,
                COMPARISON_CSV_PATH TEXT,
                VARIABLE_TARGETS_CSV_PATH TEXT,
                FIRST_OBSERVATION_ID TEXT,
                LAST_OBSERVATION_ID TEXT,
                N_OBSERVATIONS INTEGER DEFAULT 1,
                CREATED_JD REAL,
                LAST_UPDATED_JD REAL
            );

            CREATE TABLE IF NOT EXISTS COMP_STAR_LIBRARY (
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                FIELD_ID INTEGER REFERENCES FIELD_REGISTRY(ID),
                CATALOG_ID TEXT NOT NULL,
                NAME TEXT,
                RA_DEG REAL,
                DEC_DEG REAL,
                G_MAG REAL,
                BP_RP REAL,
                APERTURE_MEDIAN_MAG REAL,
                APERTURE_RMS REAL,
                PSF_MEDIAN_MAG REAL,
                PSF_RMS REAL,
                N_OBSERVATIONS INTEGER DEFAULT 1,
                N_FRAMES_TOTAL INTEGER DEFAULT 0,
                VSX_KNOWN_VARIABLE INTEGER DEFAULT 0,
                CATALOG_KNOWN_VARIABLE INTEGER DEFAULT 0,
                IS_SAFE_COMP INTEGER DEFAULT 1,
                VERDICT TEXT DEFAULT 'Approved',
                FIRST_USED_JD REAL,
                LAST_USED_JD REAL,
                LAST_OBSERVATION_ID TEXT,
                NOTES TEXT
            );

            CREATE INDEX IF NOT EXISTS IX_COMP_STAR_LIBRARY_CATALOG_ID
                ON COMP_STAR_LIBRARY(CATALOG_ID);

            CREATE INDEX IF NOT EXISTS IX_COMP_STAR_LIBRARY_FIELD_ID
                ON COMP_STAR_LIBRARY(FIELD_ID);

            CREATE INDEX IF NOT EXISTS IX_FIELD_REGISTRY_COORDS
                ON FIELD_REGISTRY(CENTER_RA_DEG, CENTER_DE_DEG);
            """
        )
        self.conn.commit()
        self._comp_library_tables_ready = True

    @staticmethod
    def _haversine_deg(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
        ra1, dec1, ra2, dec2 = map(math.radians, [ra1, dec1, ra2, dec2])
        dra = ra2 - ra1
        ddec = dec2 - dec1
        a = math.sin(ddec / 2) ** 2 + math.cos(dec1) * math.cos(dec2) * math.sin(dra / 2) ** 2
        return math.degrees(2 * math.asin(min(1.0, math.sqrt(max(0.0, a)))))

    def find_matching_field(
        self,
        ra_deg: float,
        dec_deg: float,
        *,
        match_radius_deg: float = 0.5,
    ) -> dict[str, Any] | None:
        self._migrate_comp_library_tables()
        cur = self.conn.execute("SELECT * FROM FIELD_REGISTRY;")
        rows = cur.fetchall()
        if not rows:
            return None
        best: dict[str, Any] | None = None
        best_sep = float("inf")
        for r in rows:
            d = dict(r)
            try:
                cra = float(d["CENTER_RA_DEG"])
                cde = float(d["CENTER_DE_DEG"])
            except (KeyError, TypeError, ValueError):
                continue
            sep = self._haversine_deg(float(ra_deg), float(dec_deg), cra, cde)
            if sep < best_sep:
                best_sep = sep
                best = d
        if best is not None and best_sep < float(match_radius_deg):
            return best
        return None

    def register_or_update_field(
        self,
        *,
        ra_deg: float,
        dec_deg: float,
        object_name: str | None = None,
        masterstar_path: str | None = None,
        grid_path: str | None = None,
        comparison_csv_path: str | None = None,
        variable_targets_csv_path: str | None = None,
        observation_id: str | None = None,
        radius_deg: float = 1.0,
        match_radius_deg: float = 0.5,
    ) -> int:
        from astropy.time import Time

        now_jd = float(Time.now().jd)
        match = self.find_matching_field(float(ra_deg), float(dec_deg), match_radius_deg=match_radius_deg)
        if match is not None:
            fid = int(match["ID"])
            sets = [
                "N_OBSERVATIONS = N_OBSERVATIONS + 1",
                "LAST_OBSERVATION_ID = ?",
                "LAST_UPDATED_JD = ?",
            ]
            params: list[Any] = [observation_id, now_jd]
            if object_name is not None:
                sets.append("OBJECT_NAME = ?")
                params.append(object_name)
            if masterstar_path is not None:
                sets.append("MASTERSTAR_PATH = ?")
                params.append(masterstar_path)
            if grid_path is not None:
                sets.append("GRID_PATH = ?")
                params.append(grid_path)
            if comparison_csv_path is not None:
                sets.append("COMPARISON_CSV_PATH = ?")
                params.append(comparison_csv_path)
            if variable_targets_csv_path is not None:
                sets.append("VARIABLE_TARGETS_CSV_PATH = ?")
                params.append(variable_targets_csv_path)
            params.append(fid)
            self.conn.execute(
                f"UPDATE FIELD_REGISTRY SET {', '.join(sets)} WHERE ID = ?;",
                tuple(params),
            )
            self.conn.commit()
            return fid

        self.conn.execute(
            """
            INSERT INTO FIELD_REGISTRY (
                CENTER_RA_DEG, CENTER_DE_DEG, RADIUS_DEG, OBJECT_NAME, MASTERSTAR_PATH, GRID_PATH,
                COMPARISON_CSV_PATH, VARIABLE_TARGETS_CSV_PATH,
                FIRST_OBSERVATION_ID, LAST_OBSERVATION_ID, N_OBSERVATIONS,
                CREATED_JD, LAST_UPDATED_JD
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?);
            """,
            (
                float(ra_deg),
                float(dec_deg),
                float(radius_deg),
                object_name,
                masterstar_path,
                grid_path,
                comparison_csv_path,
                variable_targets_csv_path,
                observation_id,
                observation_id,
                now_jd,
                now_jd,
            ),
        )
        self.conn.commit()
        return int(self.conn.execute("SELECT last_insert_rowid();").fetchone()[0])

    def upsert_comp_star_library(
        self,
        field_id: int,
        comp_stars: list[dict[str, Any]],
        *,
        observation_id: str | None = None,
    ) -> int:
        from astropy.time import Time

        self._migrate_comp_library_tables()
        now_jd = float(Time.now().jd)
        n_done = 0
        for star in comp_stars:
            cid = str(star.get("catalog_id") or "").strip()
            if not cid:
                continue
            verdict = str(star.get("verdict") or "Approved")
            is_safe = 1 if verdict == "Approved" else 0
            vsx = 1 if bool(star.get("vsx_known_variable")) else 0
            ckv = 1 if bool(star.get("catalog_known_variable")) else 0
            n_frames = int(star.get("n_frames") or 0)

            row = self.conn.execute(
                """
                SELECT ID, N_OBSERVATIONS, N_FRAMES_TOTAL, APERTURE_RMS, PSF_RMS,
                       APERTURE_MEDIAN_MAG, PSF_MEDIAN_MAG,
                       VSX_KNOWN_VARIABLE, CATALOG_KNOWN_VARIABLE
                FROM COMP_STAR_LIBRARY
                WHERE FIELD_ID = ? AND CATALOG_ID = ?;
                """,
                (int(field_id), cid),
            ).fetchone()
            if row is None:
                self.conn.execute(
                    """
                    INSERT INTO COMP_STAR_LIBRARY (
                        FIELD_ID, CATALOG_ID, NAME, RA_DEG, DEC_DEG, G_MAG, BP_RP,
                        APERTURE_MEDIAN_MAG, APERTURE_RMS, PSF_MEDIAN_MAG, PSF_RMS,
                        N_OBSERVATIONS, N_FRAMES_TOTAL,
                        VSX_KNOWN_VARIABLE, CATALOG_KNOWN_VARIABLE, IS_SAFE_COMP, VERDICT,
                        FIRST_USED_JD, LAST_USED_JD, LAST_OBSERVATION_ID
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        int(field_id),
                        cid,
                        star.get("name"),
                        star.get("ra_deg"),
                        star.get("dec_deg"),
                        star.get("g_mag"),
                        star.get("bp_rp"),
                        star.get("aperture_median_mag"),
                        star.get("aperture_rms"),
                        star.get("psf_median_mag"),
                        star.get("psf_rms"),
                        n_frames,
                        vsx,
                        ckv,
                        is_safe,
                        verdict,
                        now_jd,
                        now_jd,
                        observation_id,
                    ),
                )
                n_done += 1
                continue

            sid = int(row["ID"])
            old_n_obs = int(row["N_OBSERVATIONS"] or 1)
            old_n_fr = int(row["N_FRAMES_TOTAL"] or 0)
            old_ar = row["APERTURE_RMS"]
            old_pr = row["PSF_RMS"]
            old_am = row["APERTURE_MEDIAN_MAG"]
            old_pm = row["PSF_MEDIAN_MAG"]
            vsx_u = max(int(row["VSX_KNOWN_VARIABLE"] or 0), vsx)
            ckv_u = max(int(row["CATALOG_KNOWN_VARIABLE"] or 0), ckv)

            new_ar = star.get("aperture_rms")
            new_pr = star.get("psf_rms")
            w_old = max(1, old_n_fr if old_n_fr > 0 else old_n_obs)
            w_new = max(1, n_frames if n_frames > 0 else 1)

            def _pool_rms(old_v: Any, new_v: Any, wo: int, wn: int) -> float | None:
                try:
                    if new_v is None or (isinstance(new_v, float) and not math.isfinite(new_v)):
                        return float(old_v) if old_v is not None and math.isfinite(float(old_v)) else None
                    nv = float(new_v)
                    if old_v is None or (isinstance(old_v, float) and not math.isfinite(float(old_v))):
                        return nv
                    ov = float(old_v)
                    wt = wo + wn
                    if wt <= 0:
                        return nv
                    return math.sqrt(max(0.0, (wo * ov * ov + wn * nv * nv) / wt))
                except (TypeError, ValueError):
                    return None

            comb_ar = _pool_rms(old_ar, new_ar, w_old, w_new)
            comb_pr = _pool_rms(old_pr, new_pr, w_old, w_new)

            new_am = star.get("aperture_median_mag")
            comb_am = old_am
            if new_am is not None:
                try:
                    nv = float(new_am)
                    if old_am is None:
                        comb_am = nv
                    else:
                        comb_am = (float(old_am) * old_n_obs + nv) / float(old_n_obs + 1)
                except (TypeError, ValueError):
                    comb_am = old_am

            new_pm = star.get("psf_median_mag")
            comb_pm = old_pm
            if new_pm is not None:
                try:
                    nv = float(new_pm)
                    if old_pm is None:
                        comb_pm = nv
                    else:
                        comb_pm = (float(old_pm) * old_n_obs + nv) / float(old_n_obs + 1)
                except (TypeError, ValueError):
                    comb_pm = old_pm

            self.conn.execute(
                """
                UPDATE COMP_STAR_LIBRARY SET
                    NAME = COALESCE(?, NAME),
                    RA_DEG = COALESCE(?, RA_DEG),
                    DEC_DEG = COALESCE(?, DEC_DEG),
                    G_MAG = COALESCE(?, G_MAG),
                    BP_RP = COALESCE(?, BP_RP),
                    APERTURE_MEDIAN_MAG = ?,
                    APERTURE_RMS = ?,
                    PSF_MEDIAN_MAG = ?,
                    PSF_RMS = ?,
                    N_OBSERVATIONS = N_OBSERVATIONS + 1,
                    N_FRAMES_TOTAL = N_FRAMES_TOTAL + ?,
                    VSX_KNOWN_VARIABLE = ?,
                    CATALOG_KNOWN_VARIABLE = ?,
                    IS_SAFE_COMP = ?,
                    VERDICT = ?,
                    LAST_USED_JD = ?,
                    LAST_OBSERVATION_ID = ?
                WHERE ID = ?;
                """,
                (
                    star.get("name"),
                    star.get("ra_deg"),
                    star.get("dec_deg"),
                    star.get("g_mag"),
                    star.get("bp_rp"),
                    comb_am,
                    comb_ar,
                    comb_pm,
                    comb_pr,
                    n_frames,
                    vsx_u,
                    ckv_u,
                    is_safe,
                    verdict,
                    now_jd,
                    observation_id,
                    sid,
                ),
            )
            n_done += 1
        self.conn.commit()
        return n_done

    def get_comp_stars_for_field(
        self,
        ra_deg: float,
        dec_deg: float,
        *,
        match_radius_deg: float = 0.5,
        only_approved: bool = True,
    ) -> list[dict[str, Any]]:
        fld = self.find_matching_field(float(ra_deg), float(dec_deg), match_radius_deg=match_radius_deg)
        if fld is None:
            return []
        fid = int(fld["ID"])
        flt = 1 if only_approved else 0
        cur = self.conn.execute(
            """
            SELECT * FROM COMP_STAR_LIBRARY
            WHERE FIELD_ID = ?
              AND (? = 0 OR IS_SAFE_COMP = 1)
              AND (? = 0 OR VERDICT = 'Approved')
            ORDER BY G_MAG ASC;
            """,
            (fid, flt, flt),
        )
        return [dict(r) for r in cur.fetchall()]

    def close(self) -> None:
        try:
            self.conn.commit()
        except sqlite3.Error:
            pass
        self.conn.close()

