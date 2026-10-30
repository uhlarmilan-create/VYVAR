"""Star catalog access: the Gaia DR3 DB and the objects DB (VSX + exoplanets).

One module for every catalog reader (CATALOG-DB-2026-10, Milan 2026-10-20):

- Gaia DR3: ``gaia_db_path`` (table ``gaia_dr3``; ``gaia_coverage`` writes the online top-up).
- Objects: ``objects_db_path`` (tables ``vsx_data`` and ``exoplanet_data``), built by
  ``catalogs/objects_make.py``.
- ``catalog_meta`` in each catalog DB: one row per table (source, build date, limits, row count,
  content fingerprint of the objects tables). The builders write it (``catalogs/``), the Gaia
  top-up keeps its row count current; provenance and Settings read it.

Readers open the catalogs read-only (``connect_ro``). Only the Gaia top-up (``gaia_coverage``) and
the builders write (``connect_rw``).
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from astropy.coordinates import SkyCoord

from gaia_catalog_id import normalize_gaia_source_id
from infolog import log_event

#: default catalog DBs under the VYVAR data directory (one folder for every catalog)
from pipeline_constants import (  # noqa: E402,F401  (one default)
    GAIA_BUILD_MAG_LIMIT_DEFAULT,
    GAIA_DB_DEFAULT,
    OBJECTS_DB_DEFAULT,
    VSX_BUILD_MAG_LIMIT_DEFAULT,
)

#: the indexes of ``gaia_dr3`` VYVAR queries use (EXPLAIN QUERY PLAN, 2026-10-20): sky box
#: (ra, dec), Dec strips of the blind index build, magnitude cuts. ``catalogs/gaia_maintain.py``
#: keeps the same list (standard library only) and drops any other index.
GAIA_INDEXES: dict[str, str] = {
    "idx_ra_dec": "CREATE INDEX IF NOT EXISTS idx_ra_dec ON gaia_dr3 (ra, dec)",
    "idx_dec": "CREATE INDEX IF NOT EXISTS idx_dec ON gaia_dr3 (dec)",
    "idx_g_mag": "CREATE INDEX IF NOT EXISTS idx_g_mag ON gaia_dr3 (g_mag)",
}

_GAIA_INDEX_CHECK_DONE = False


# ---------------------------------------------------------------------------
# paths and connections
# ---------------------------------------------------------------------------
def resolve_catalog_path(raw: str | Path) -> Path:
    """Resolve a catalog DB path against the data root (not the CWD)."""
    from config import resolve_config_path, resolve_data_root  # noqa: PLC0415

    install_root = Path(__file__).resolve().parent.parent
    data_root = resolve_data_root(install_root)
    resolved = resolve_config_path(raw, data_root)
    if not resolved:
        return Path()
    return Path(resolved)


def configured_catalog_path(key: str, default: str) -> Path:
    """Catalog path of config.json key ``key`` (Settings -> Paths and catalogs), else ``default``;
    relative paths resolve against the data directory. The builders write where VYVAR reads."""
    from config import load_config_json, resolve_config_path, resolve_data_root  # noqa: PLC0415

    data_root = Path(resolve_data_root(Path(__file__).resolve().parent.parent))
    raw = str(load_config_json(data_root).get(key) or "").strip() or default
    return Path(resolve_config_path(raw, data_root))


def connect_ro(db_path: str | Path) -> sqlite3.Connection:
    """Read-only connection to a catalog DB (no index creation or other writes by readers)."""
    p = Path(db_path).expanduser().resolve()
    if not p.is_file():
        raise sqlite3.OperationalError(f"unable to open database file: {p}")
    # canonical SQLite URI: file:///C:/... on Windows, file:///home/... on POSIX (percent-encoded)
    return sqlite3.connect(p.as_uri() + "?mode=ro", uri=True)


def connect_rw(db_path: str | Path) -> sqlite3.Connection:
    """Writable connection: only the Gaia top-up and the catalog builders."""
    return sqlite3.connect(str(Path(db_path).expanduser()))


def _table_columns(con: sqlite3.Connection, table: str) -> set[str]:
    return {str(r[1]).strip().lower() for r in con.execute(f"PRAGMA table_info('{table}');").fetchall()}


GAIA_TABLE = "gaia_dr3"


def gaia_table(con: sqlite3.Connection) -> str:
    """The Gaia source table of a VYVAR Gaia DB (``gaia_dr3``); RuntimeError naming the tables found otherwise."""
    if _has_table(con, GAIA_TABLE):
        return GAIA_TABLE
    names = sorted(str(r[0]) for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall())
    raise RuntimeError(f"Gaia DB: no table {GAIA_TABLE!r} (found: {names[:20]})")


def _has_table(con: sqlite3.Connection, table: str) -> bool:
    return con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=? LIMIT 1;", (table,)
    ).fetchone() is not None


def ra_intervals_deg(ra_min: float, ra_max: float) -> list[tuple[float, float]]:
    """Split an RA range (deg) into sub-intervals within [0, 360) when the box crosses RA 0."""
    rm, rM = float(ra_min), float(ra_max)
    if rM - rm >= 360.0:
        return [(0.0, 360.0)]
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


def _row_limit(max_rows: int | None) -> int | None:
    if max_rows is None:
        return None
    try:
        lim = int(max_rows)
    except (TypeError, ValueError):
        return None
    return lim if lim > 0 else None


# ---------------------------------------------------------------------------
# catalog_meta and table fingerprints
# ---------------------------------------------------------------------------
CATALOG_META_SCHEMA = """
CREATE TABLE IF NOT EXISTS catalog_meta (
    table_name     TEXT PRIMARY KEY,
    source         TEXT,
    built_utc      TEXT,
    row_count      INTEGER,
    content_sha256 TEXT,
    params_json    TEXT,
    builder        TEXT
);
"""

#: primary key of each objects table (fingerprint row order)
TABLE_KEYS = {"vsx_data": "oid", "exoplanet_data": "obj_id"}


def table_content_sha256(con: sqlite3.Connection, table: str) -> str:
    """sha256 over all rows of ``table`` ordered by its key (one line ``repr(row)`` per row)."""
    key = TABLE_KEYS[table]
    h = hashlib.sha256()
    cur = con.execute(f"SELECT * FROM {table} ORDER BY {key};")
    while True:
        rows = cur.fetchmany(100_000)
        if not rows:
            break
        h.update("".join(repr(tuple(r)) + "\n" for r in rows).encode("utf-8"))
    return h.hexdigest()


def write_catalog_meta(
    con: sqlite3.Connection,
    table: str,
    *,
    source: str,
    params: dict[str, Any],
    builder: str,
    built_utc: str | None = None,
) -> dict[str, Any]:
    """(Re)write the ``catalog_meta`` row of ``table`` from its current content.

    The content fingerprint is written for the objects tables (``TABLE_KEYS``); ``gaia_dr3``
    (200 million rows) is identified by its row count, build parameters and coverage instead."""
    con.execute(CATALOG_META_SCHEMA)
    n = int(con.execute(f"SELECT COUNT(*) FROM {table};").fetchone()[0])
    rec = {
        "table_name": table,
        "source": str(source),
        "built_utc": built_utc or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "row_count": n,
        "content_sha256": table_content_sha256(con, table) if table in TABLE_KEYS else None,
        "params_json": json.dumps(params, sort_keys=True),
        "builder": str(builder),
    }
    con.execute(
        "INSERT OR REPLACE INTO catalog_meta "
        "(table_name, source, built_utc, row_count, content_sha256, params_json, builder) "
        "VALUES (:table_name, :source, :built_utc, :row_count, :content_sha256, :params_json, :builder);",
        rec,
    )
    con.commit()
    return rec


def add_catalog_meta_rows(con: sqlite3.Connection, table: str, n_new: int) -> None:
    """Keep ``catalog_meta.row_count`` current after rows were added (Gaia online top-up).
    No row (a DB never registered) stays without one."""
    if int(n_new) <= 0 or not _has_table(con, "catalog_meta"):
        return
    con.execute("UPDATE catalog_meta SET row_count = row_count + ? WHERE table_name = ?;", (int(n_new), table))


def read_catalog_meta(db_path: str | Path) -> dict[str, dict[str, Any]]:
    """``catalog_meta`` rows by table name ({} when the DB or the table is missing)."""
    p = resolve_catalog_path(db_path)
    if not str(db_path).strip() or not p.is_file():
        return {}
    con = connect_ro(p)
    con.row_factory = sqlite3.Row
    try:
        if not _has_table(con, "catalog_meta"):
            return {}
        out: dict[str, dict[str, Any]] = {}
        for r in con.execute("SELECT * FROM catalog_meta;"):
            d = dict(r)
            try:
                d["params"] = json.loads(d.get("params_json") or "{}")
            except ValueError:
                d["params"] = {}
            out[str(d["table_name"])] = d
        return out
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Gaia DR3
# ---------------------------------------------------------------------------
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
    p = resolve_catalog_path(db_path)
    if not p.is_file():
        raise FileNotFoundError(f"Gaia DB not found: {p}")

    mag_cap: float | None = None
    if mag_limit is not None:
        try:
            ml = float(mag_limit)
        except (TypeError, ValueError):
            ml = float("nan")
        if not math.isfinite(ml) or ml <= 0:
            log_event(f"GAIA SQL: invalid mag_limit={mag_limit!r} - no g_mag cap applied.")
        else:
            mag_cap = float(ml)

    conn = connect_ro(p)
    conn.row_factory = sqlite3.Row
    global _GAIA_INDEX_CHECK_DONE  # noqa: PLW0603
    try:
        cols = _table_columns(conn, "gaia_dr3")
        if not _GAIA_INDEX_CHECK_DONE:
            idx_names = [str(r[1]) for r in conn.execute("PRAGMA index_list('gaia_dr3');").fetchall()]
            has_ra_dec = any(
                {"ra", "dec"} <= {str(c[2]).lower() for c in conn.execute(f"PRAGMA index_info('{nm}');").fetchall()}
                for nm in idx_names
            )
            if not has_ra_dec:
                log_event("GAIA DB: no (ra, dec) index found; sky-box queries will be slow.")
            _GAIA_INDEX_CHECK_DONE = True
        base_cols = ["source_id", "ra", "dec", "g_mag", "bp_mag", "rp_mag", "bp_rp", "var_flag"]
        opt_cols = [c for c in ("g_flux_error_rel", "non_single_star", "pmra", "pmdec") if c in cols]
        # Alternate schema: some DBs carry a string catalog id.
        if "catalog_id" in cols:
            opt_cols.append("catalog_id")
        sel = ", ".join(base_cols + opt_cols)
        mag_clause = f" AND g_mag <= {float(mag_cap)}" if mag_cap is not None else ""
        lim = _row_limit(max_rows)
        # RA wrap at 0 deg: one query per RA interval (CATALOG-FOOTPRINT-2026-10).
        intervals = ra_intervals_deg(float(ra_min), float(ra_max))
        order = f" ORDER BY g_mag ASC LIMIT {int(lim)}" if lim is not None else ""
        query = (
            f"SELECT {sel} FROM gaia_dr3 "
            f"WHERE ra >= ? AND ra <= ? AND dec >= ? AND dec <= ?{mag_clause}{order};"
        )
        rows = []
        for _rlo, _rhi in intervals:
            cur = conn.execute(query, (float(_rlo), float(_rhi), float(dec_min), float(dec_max)))
            rows.extend(dict(r) for r in cur.fetchall())
        if lim is not None and len(intervals) > 1:
            rows.sort(key=lambda r: (r.get("g_mag") is None, r.get("g_mag") or 0.0))
            rows = rows[: int(lim)]
        if mag_cap is not None:
            log_event(f"GAIA SQL: Found {len(rows)} stars (Mag <= {float(mag_cap)})")
        else:
            log_event(f"GAIA SQL: Found {len(rows)} stars (no mag cap)")
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
    p = resolve_catalog_path(db_path)
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
    conn = connect_ro(p)
    conn.row_factory = sqlite3.Row
    try:
        cols = _table_columns(conn, "gaia_dr3")
        if "source_id" not in cols:
            return {}
        sel = ", ".join(["source_id"] + [c for c in ("bp_rp", "g_mag", "bp_mag", "rp_mag") if c in cols])
        for i0 in range(0, len(ids_u), bs):
            chunk = ids_u[i0 : i0 + bs]
            ph = ",".join("?" * len(chunk))
            q = f"SELECT {sel} FROM gaia_dr3 WHERE source_id IN ({ph});"
            for row in conn.execute(q, chunk):
                d = dict(row)
                key = normalize_gaia_source_id(d.get("source_id"))
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


def _validate_table(db_path: str | Path, table: str, required: set[str]) -> tuple[bool, str]:
    p = resolve_catalog_path(db_path)
    if not str(db_path).strip() or not p.is_file():
        return False, "missing_file"
    con = connect_ro(p)
    try:
        if not _has_table(con, table):
            return False, f"missing_table_{table}"
        missing = sorted(required - _table_columns(con, table))
        if missing:
            return False, f"missing_columns:{','.join(missing)}"
        return True, "ok"
    finally:
        con.close()


def validate_gaia_db_schema(db_path: str | Path) -> tuple[bool, str]:
    """Validate local Gaia DB has table/columns required by VYVAR."""
    return _validate_table(db_path, "gaia_dr3", {"source_id", "ra", "dec", "g_mag", "bp_rp"})


# ---------------------------------------------------------------------------
# objects DB: VSX + exoplanets
# ---------------------------------------------------------------------------
class VSXCatalogError(RuntimeError):
    """VSX table path/schema/availability failure (fail-loud, like Gaia)."""


class ExoplanetCatalogError(RuntimeError):
    """Exoplanet table path/schema/availability failure (fail-loud when configured)."""


def validate_vsx_schema(objects_db_path: str | Path) -> tuple[bool, str]:
    """Validate ``vsx_data`` (VizieR B/vsx/vsx) in the objects DB."""
    return _validate_table(objects_db_path, "vsx_data", {"oid", "ra_deg", "dec_deg"})


def validate_exoplanet_schema(objects_db_path: str | Path) -> tuple[bool, str]:
    """Validate ``exoplanet_data`` (NASA Exoplanet Archive) in the objects DB."""
    return _validate_table(objects_db_path, "exoplanet_data", {"obj_id", "ra_deg", "dec_deg"})


def count_vsx_rows(objects_db_path: str | Path) -> int:
    """Row count of ``vsx_data`` (0 when the table is missing or unreadable)."""
    p = resolve_catalog_path(objects_db_path)
    if not p.is_file():
        return 0
    try:
        con = connect_ro(p)
    except sqlite3.Error:
        return 0
    try:
        if not _has_table(con, "vsx_data"):
            return 0
        row = con.execute("SELECT COUNT(*) FROM vsx_data;").fetchone()
        return int(row[0] if row is not None else 0)
    except sqlite3.Error:
        return 0
    finally:
        con.close()


_BUILD_HINT = "Build it with catalogs/objects_make.py (python catalogs/objects_make.py --vsx --exoplanets)."


def require_vsx_db(objects_db_path: str | Path | None) -> Path:
    """Resolved objects DB path with a usable ``vsx_data`` table, or VSXCatalogError."""
    raw = str(objects_db_path or "").strip()
    if not raw:
        raise VSXCatalogError(f"Objects database path is not set (config key objects_db_path). {_BUILD_HINT}")
    p = resolve_catalog_path(raw)
    if not p.is_file():
        raise VSXCatalogError(f"Objects database file not found: {p}. {_BUILD_HINT}")
    ok, code = validate_vsx_schema(p)
    if not ok:
        raise VSXCatalogError(f"VSX table invalid ({code}) in {p}. {_BUILD_HINT}")
    if count_vsx_rows(p) <= 0:
        raise VSXCatalogError(f"VSX table vsx_data has zero rows in {p}. {_BUILD_HINT}")
    return p


def require_exoplanet_db(objects_db_path: str | Path | None) -> Path:
    """Resolved objects DB path with a usable ``exoplanet_data`` table, or ExoplanetCatalogError."""
    raw = str(objects_db_path or "").strip()
    if not raw:
        raise ExoplanetCatalogError(f"Objects database path is not set (config key objects_db_path). {_BUILD_HINT}")
    p = resolve_catalog_path(raw)
    if not p.is_file():
        raise ExoplanetCatalogError(f"Objects database file not found: {p}. {_BUILD_HINT}")
    ok, code = validate_exoplanet_schema(p)
    if not ok:
        raise ExoplanetCatalogError(f"Exoplanet table invalid ({code}) in {p}. {_BUILD_HINT}")
    return p


def _query_box(
    db_path: str | Path,
    table: str,
    want: list[str],
    key_col: str,
    *,
    ra_min: float,
    ra_max: float,
    dec_min: float,
    dec_max: float,
    max_rows: int | None,
) -> tuple[list[dict[str, Any]], float, float, int]:
    """Rows of an objects table in an RA/Dec box; RA wrap via split intervals; de-duplicated by
    ``key_col`` when present, else by (ra_deg, dec_deg). With ``max_rows`` the rows nearest to the
    box centre are kept (G05-50: a LIMIT without ORDER BY kept an arbitrary subset, the first RA
    interval first)."""
    p = resolve_catalog_path(db_path)
    de0, de1 = float(dec_min), float(dec_max)
    if de1 < de0:
        de0, de1 = de1, de0
    de0 = max(-90.0, min(90.0, de0))
    de1 = max(-90.0, min(90.0, de1))
    intervals = ra_intervals_deg(float(ra_min), float(ra_max))
    if not p.is_file():
        return [], de0, de1, len(intervals)
    lim = _row_limit(max_rows)
    conn = connect_ro(p)
    conn.row_factory = sqlite3.Row
    try:
        if not _has_table(conn, table):
            return [], de0, de1, len(intervals)
        cols = _table_columns(conn, table)
        if "ra_deg" not in cols or "dec_deg" not in cols:
            return [], de0, de1, len(intervals)
        sel = ", ".join(c for c in want if c in cols)
        q = f"SELECT {sel} FROM {table} WHERE ra_deg >= ? AND ra_deg <= ? AND dec_deg >= ? AND dec_deg <= ?;"
        seen: set[Any] = set()
        rows_out: list[dict[str, Any]] = []
        for rlo, rhi in intervals:
            for r in conn.execute(q, (float(rlo), float(rhi), de0, de1)).fetchall():
                d = dict(r)
                k = d.get(key_col)
                if isinstance(k, str):
                    k = k.strip()
                key: Any = k if k is not None and k != "" else (d.get("ra_deg"), d.get("dec_deg"))
                if key in seen:
                    continue
                seen.add(key)
                rows_out.append(d)
        if lim is not None and len(rows_out) > lim:
            rows_out = _nearest_rows(rows_out, float(ra_min), float(ra_max), de0, de1)[:lim]
        return rows_out, de0, de1, len(intervals)
    finally:
        conn.close()


def _nearest_rows(rows: list[dict[str, Any]], ra_min: float, ra_max: float, de0: float, de1: float) -> list[dict[str, Any]]:
    """``rows`` sorted by great-circle distance to the centre of the box (rows without a finite
    position last)."""
    rc = math.radians((0.5 * (ra_min + ra_max)) % 360.0)
    dc = math.radians(0.5 * (de0 + de1))

    def _key(d: dict[str, Any]) -> float:
        try:
            ra, de = math.radians(float(d["ra_deg"])), math.radians(float(d["dec_deg"]))
        except (TypeError, ValueError, KeyError):
            return math.inf
        c = math.sin(dc) * math.sin(de) + math.cos(dc) * math.cos(de) * math.cos(ra - rc)
        return -c if math.isfinite(c) else math.inf

    return sorted(rows, key=_key)


def query_local_vsx(
    objects_db_path: str | Path,
    *,
    ra_min: float,
    ra_max: float,
    dec_min: float,
    dec_max: float,
    max_rows: int | None = None,
) -> list[dict[str, Any]]:
    """VSX rows (``vsx_data``) in a rectangular RA/Dec window (deg, ICRS); same box as ``query_local_gaia``."""
    want = ["oid", "name", "ra_deg", "dec_deg", "var_type", "period", "mag_max", "mag_min"]
    rows, de0, de1, n_int = _query_box(
        objects_db_path, "vsx_data", want, "oid",
        ra_min=ra_min, ra_max=ra_max, dec_min=dec_min, dec_max=dec_max, max_rows=max_rows,
    )
    log_event(f"VSX SQL: {len(rows)} rows (box Dec=[{de0:.3f},{de1:.3f}], RA intervals={n_int})")
    return rows


def vsx_amplitude_mag(mag_max: Any, mag_min: Any) -> float:
    """Amplitude (mag) of a VSX entry: ``mag_min - mag_max``; VSX writes the amplitude itself in the
    minimum field when only the range is known ("(0.34)"), which the table holds as a value below
    ``mag_max``. NaN when either is missing."""
    try:
        mx, mn = float(mag_max), float(mag_min)
    except (TypeError, ValueError):
        return float("nan")
    if not (math.isfinite(mx) and math.isfinite(mn)):
        return float("nan")
    return mn - mx if mn >= mx else mn


def query_local_exoplanet(
    objects_db_path: str | Path,
    *,
    ra_min: float,
    ra_max: float,
    dec_min: float,
    dec_max: float,
    max_rows: int | None = None,
) -> list[dict[str, Any]]:
    """Exoplanet rows (``exoplanet_data``) in a rectangular RA/Dec window (deg)."""
    want = ["obj_id", "name", "host_name", "ra_deg", "dec_deg", "cat_source", "disposition", "period", "mag", "mag_band"]
    rows, de0, de1, n_int = _query_box(
        objects_db_path, "exoplanet_data", want, "obj_id",
        ra_min=ra_min, ra_max=ra_max, dec_min=dec_min, dec_max=dec_max, max_rows=max_rows,
    )
    log_event(f"EXO SQL: {len(rows)} rows (box Dec=[{de0:.3f},{de1:.3f}], RA intervals={n_int})")
    return rows


# --- moved from pipeline_catalog (LAYER-2026-10) ---


#: float rounding pad of the RA half-width [deg] (the half-width itself is exact)
_RA_HALF_WIDTH_PAD_DEG = 1e-9


def cone_ra_half_width_deg(dec0_deg: float, radius_deg: float) -> float | None:
    """Exact RA half-width [deg] of a great-circle cone of ``radius_deg`` about Dec ``dec0_deg``.

    The cone's RA extremes are its tangent points with meridians: sin(dRA) = sin(r) / cos(dec0)
    (RA-BOX-2026-10, G03-05). ``None`` when the cone reaches a pole (|dec0| + r >= 90): every RA."""
    r = max(0.0, float(radius_deg))
    d0 = float(dec0_deg)
    if abs(d0) + r >= 90.0:
        return None
    s = math.sin(math.radians(r)) / math.cos(math.radians(d0))
    if s >= 1.0:
        return None
    return math.degrees(math.asin(s)) + _RA_HALF_WIDTH_PAD_DEG


def ra_boxes_for_cone(ra0_deg: float, dec0_deg: float, radius_deg: float) -> tuple[list[tuple[float, float]], float, float]:
    """RA intervals inside [0, 360] and the Dec range [deg] of the box that contains the cone.

    The one RA-box rule of VYVAR (RA-BOX-2026-10, Milan 2026-10-29): every catalogue box query
    (Gaia, VSX, exoplanets, the plate-solve cone, the blind index cells) takes its RA range from
    :func:`cone_ra_half_width_deg` and splits the wrap at RA 0 with :func:`ra_intervals_deg`."""
    r = max(0.0, float(radius_deg))
    d0 = float(dec0_deg)
    de_min, de_max = max(-90.0, d0 - r), min(90.0, d0 + r)
    half = cone_ra_half_width_deg(d0, r)
    if half is None or half >= 180.0:
        return [(0.0, 360.0)], de_min, de_max
    ra0 = float(ra0_deg) % 360.0
    return ra_intervals_deg(ra0 - half, ra0 + half), de_min, de_max


def cone_sky_box(center: SkyCoord, radius_deg: float) -> tuple[float, float, float, float]:
    """RA/Dec box [deg] that contains the great-circle cone (CATALOG-FOOTPRINT-2026-10).

    RA half-width from :func:`cone_ra_half_width_deg` (RA-BOX-2026-10). A cone that reaches a pole
    takes every RA. RA may lie outside [0, 360); ``query_local_*`` splits the wrap."""
    ra0 = float(center.icrs.ra.deg)
    de0 = float(center.icrs.dec.deg)
    r = max(0.0, float(radius_deg))
    de_min = max(-90.0, de0 - r)
    de_max = min(90.0, de0 + r)
    half = cone_ra_half_width_deg(de0, r)
    if half is None or half >= 180.0:
        return 0.0, 360.0, de_min, de_max
    return ra0 - half, ra0 + half, de_min, de_max
