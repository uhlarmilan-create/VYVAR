"""objects_make.py - VYVAR objects catalog: AAVSO VSX + NASA Exoplanet Archive in one SQLite file.

Replaces VSX/vsx_make.py and exoplanets/exoplanet_make.py (CATALOG-DB-2026-10, Milan 2026-10-20).
Output: ``<data>/catalogs/vyvar_objects.db`` (config key ``objects_db_path``) with

- ``vsx_data``        VizieR B/vsx/vsx (variable stars), key ``oid``
- ``exoplanet_data``  NASA Exoplanet Archive pscomppars (confirmed) + toi (TESS candidates), key ``obj_id``
- ``catalog_meta``    one row per table: source, build date, parameters, row count, content sha256

Both tables are always created (an empty table is valid), so one file serves VSX and exoplanets.
``--vsx`` / ``--exoplanets`` replace the table as a whole (Milan 2026-10-20, "uplna vymena"): the
download goes into a staging table; only when every strip / archive table arrived and the row count
is plausible (>= half of the old table, ``--force`` overrides) the staging table replaces the old one
in one transaction. A failed download leaves the old table untouched. The rows added, removed and
changed against the old table are printed and stored in ``catalog_meta.params_json``.
``--import-vsx`` / ``--import-exoplanets`` copy the tables of an existing VSX / exoplanet SQLite file
without downloading (rows not yet present are added).

Examples:
  python catalogs/objects_make.py --vsx --exoplanets
  python catalogs/objects_make.py --vsx --vsx-dec-min -30 --vsx-dec-max 90 --vsx-mag-limit 18
  python catalogs/objects_make.py --exoplanets --confirmed-only
  python catalogs/objects_make.py --import-vsx VSX/vyvar_vsx_local_v2.db --import-exoplanets exoplanets/vyvar_exoplanet_local.db
"""
from __future__ import annotations

import argparse
import io
import sqlite3
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _find_src(start: Path) -> tuple[Path, Path]:
    """(install root, module dir holding catalog_db) - src_py of a checkout or of a release bundle."""
    for candidate in (start, *start.parents):
        src = candidate / "src_py"
        if (src / "catalog_db.py").is_file() or any(src.glob("catalog_db*.pyd")) or any(src.glob("catalog_db*.so")):
            return candidate, src
    raise SystemExit("objects_make.py needs src_py/catalog_db from the VYVAR install; run it from the install.")


_ROOT, _SRC = _find_src(_HERE)
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from catalog_db import (  # noqa: E402
    CATALOG_META_SCHEMA,
    OBJECTS_DB_DEFAULT,
    VSX_BUILD_MAG_LIMIT_DEFAULT,
    connect_rw,
    write_catalog_meta,
)

BUILDER = "catalogs/objects_make.py"

VSX_SCHEMA = """
CREATE TABLE IF NOT EXISTS vsx_data (
    oid       INTEGER PRIMARY KEY,   -- VSX OID
    name      TEXT,
    ra_deg    REAL,
    dec_deg   REAL,
    var_type  TEXT,
    period    REAL,                  -- days; NULL when VSX gives none
    mag_max   REAL,
    mag_min   REAL
);
"""
VSX_COLS = ["oid", "name", "ra_deg", "dec_deg", "var_type", "period", "mag_max", "mag_min"]

EXO_SCHEMA = """
CREATE TABLE IF NOT EXISTS exoplanet_data (
    obj_id      TEXT PRIMARY KEY,   -- pl_name (CONFIRMED) / 'TOI-1234.01' (TOI)
    name        TEXT,
    host_name   TEXT,               -- host star (cross-match context)
    ra_deg      REAL,
    dec_deg     REAL,
    cat_source  TEXT,               -- 'CONFIRMED' | 'TOI'
    disposition TEXT,               -- 'CONFIRMED' | TFOPWG: PC/CP/KP/APC/FP/FA
    period      REAL,               -- orbital period [days]
    mag         REAL,               -- TESS mag (preferred), V fallback
    mag_band    TEXT                -- 'TESS' | 'V' | NULL
);
"""
EXO_COLS = ["obj_id", "name", "host_name", "ra_deg", "dec_deg", "cat_source", "disposition", "period", "mag", "mag_band"]

VSX_SOURCE = "VizieR B/vsx/vsx"
EXO_SOURCE = "NASA Exoplanet Archive TAP (pscomppars + toi)"
TAP_BASE = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
HTTP_TIMEOUT_S = 300

#: NASA archive columns per table; required ones missing -> abort, optional ones missing -> NULL
_EXO_SRC = {
    "pscomppars": {"required": ["pl_name", "ra", "dec"], "optional": ["hostname", "pl_orbper", "sy_tmag", "sy_vmag"]},
    "toi": {"required": ["toi", "ra", "dec"], "optional": ["tid", "pl_orbper", "st_tmag", "tfopwg_disp"]},
}


def default_db_path() -> Path:
    """``objects_db_path`` of config.json (Settings), else ``<data>/catalogs/vyvar_objects.db``."""
    from catalog_db import configured_catalog_path  # noqa: PLC0415

    return configured_catalog_path("objects_db_path", OBJECTS_DB_DEFAULT)


SCHEMAS = {"vsx_data": VSX_SCHEMA, "exoplanet_data": EXO_SCHEMA}
KEYS = {"vsx_data": "oid", "exoplanet_data": "obj_id"}
INDEXES = {
    "vsx_data": ["CREATE INDEX IF NOT EXISTS idx_vsx_ra ON vsx_data (ra_deg);",
                 "CREATE INDEX IF NOT EXISTS idx_vsx_dec ON vsx_data (dec_deg);"],
    "exoplanet_data": ["CREATE INDEX IF NOT EXISTS idx_exo_ra ON exoplanet_data (ra_deg);",
                       "CREATE INDEX IF NOT EXISTS idx_exo_dec ON exoplanet_data (dec_deg);",
                       "CREATE INDEX IF NOT EXISTS idx_exo_src ON exoplanet_data (cat_source);"],
}
#: a refreshed table smaller than this fraction of the old one is refused (truncated download)
MIN_KEEP_FRACTION = 0.5


class DownloadError(RuntimeError):
    """A strip / archive table could not be downloaded: the old table stays."""


def ensure_schema(conn: sqlite3.Connection) -> None:
    for table, ddl in SCHEMAS.items():
        conn.execute(ddl)
        for sql in INDEXES[table]:
            conn.execute(sql)
    conn.execute(CATALOG_META_SCHEMA)
    conn.commit()


def refresh_table(conn: sqlite3.Connection, table: str, fill, *, force: bool = False) -> dict:  # noqa: ANN001
    """Replace ``table`` by a complete new download. ``fill(conn, staging_table)`` downloads into
    the staging table and raises :class:`DownloadError` when anything is missing. Returns the
    difference to the old table (added / removed / changed rows by key)."""
    staging = f"{table}_new"
    key = KEYS[table]
    conn.execute(f"DROP TABLE IF EXISTS {staging};")
    conn.execute(SCHEMAS[table].replace(f"EXISTS {table} (", f"EXISTS {staging} ("))
    conn.commit()
    try:
        fill(conn, staging)
    except BaseException:
        conn.execute(f"DROP TABLE IF EXISTS {staging};")
        conn.commit()
        raise
    n_old, n_new = _count(conn, table), _count(conn, staging)
    if n_old and n_new < MIN_KEEP_FRACTION * n_old and not force:
        conn.execute(f"DROP TABLE IF EXISTS {staging};")
        conn.commit()
        raise DownloadError(f"{table}: the download has {n_new} rows, the current table {n_old}; "
                            f"less than {MIN_KEEP_FRACTION:.0%} looks truncated - kept the current table "
                            "(--force replaces it anyway)")
    added = int(conn.execute(f"SELECT COUNT(*) FROM {staging} WHERE {key} NOT IN (SELECT {key} FROM {table});").fetchone()[0])
    removed = int(conn.execute(f"SELECT COUNT(*) FROM {table} WHERE {key} NOT IN (SELECT {key} FROM {staging});").fetchone()[0])
    new_or_changed = int(conn.execute(
        f"SELECT COUNT(*) FROM (SELECT * FROM {staging} EXCEPT SELECT * FROM {table});").fetchone()[0])
    diff = {"previous_rows": n_old, "rows": n_new, "added": added, "removed": removed,
            "changed": new_or_changed - added}
    iso = conn.isolation_level
    conn.isolation_level = None
    try:
        conn.execute("BEGIN IMMEDIATE;")
        conn.execute(f"DROP TABLE {table};")
        conn.execute(f"ALTER TABLE {staging} RENAME TO {table};")
        for sql in INDEXES[table]:
            conn.execute(sql)
        conn.execute("COMMIT;")
    except BaseException:
        conn.execute("ROLLBACK;")
        raise
    finally:
        conn.isolation_level = iso
    print(f"[{table}] replaced: {n_new} rows (before {n_old}); added {added}, removed {removed}, "
          f"changed {diff['changed']}")
    return diff


def _count(conn: sqlite3.Connection, table: str) -> int:
    return int(conn.execute(f"SELECT COUNT(*) FROM {table};").fetchone()[0])


def _insert(conn: sqlite3.Connection, table: str, cols: list[str], rows: list[tuple]) -> int:
    before = _count(conn, table)
    conn.executemany(
        f"INSERT OR IGNORE INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})", rows
    )
    conn.commit()
    return _count(conn, table) - before


# ---------------------------------------------------------------------------
# import from existing files (no download)
# ---------------------------------------------------------------------------
def import_table(conn: sqlite3.Connection, src: Path, table: str, cols: list[str]) -> int:
    """Copy ``table`` of an existing SQLite file (same columns) into the objects DB."""
    if not src.is_file():
        raise SystemExit(f"[ABORT] {src} not found")
    s = sqlite3.connect(f"file:{src.resolve().as_posix()}?mode=ro", uri=True)
    try:
        have = {str(r[1]).lower() for r in s.execute(f"PRAGMA table_info('{table}');")}
        missing = [c for c in cols if c not in have]
        if missing:
            raise SystemExit(f"[ABORT] {src}: table {table} lacks columns {missing}")
        cur = s.execute(f"SELECT {', '.join(cols)} FROM {table};")
        added = 0
        while True:
            rows = cur.fetchmany(200_000)
            if not rows:
                break
            added += _insert(conn, table, cols, rows)
        return added
    finally:
        s.close()


# ---------------------------------------------------------------------------
# VSX download (VizieR B/vsx/vsx in declination strips)
# ---------------------------------------------------------------------------
def build_vsx(conn: sqlite3.Connection, *, dec_min: float, dec_max: float, mag_limit: float, batch_deg: float,
              table: str = "vsx_data") -> int:
    import pandas as pd  # noqa: PLC0415
    from astroquery.vizier import Vizier  # type: ignore[import-not-found]  # noqa: PLC0415

    v = Vizier(columns=["OID", "Name", "RAJ2000", "DEJ2000", "Type", "Period", "max", "min"], row_limit=-1)
    rename = {"OID": "oid", "Name": "name", "RAJ2000": "ra_deg", "DEJ2000": "dec_deg", "Type": "var_type",
              "Period": "period", "max": "mag_max", "min": "mag_min"}
    added_total = 0
    d0 = float(dec_min)
    while d0 < dec_max:
        d1 = min(d0 + batch_deg, dec_max)
        print(f"[VSX] Dec {d0} .. {d1} deg")
        for attempt in range(5):
            try:
                result = v.query_constraints(catalog="B/vsx/vsx", DEJ2000=f"{d0}..{d1}", max=f"<{mag_limit}")
                if not result or len(result) == 0:
                    print("   empty strip")
                    break
                df = result[0].to_pandas().rename(columns=rename)
                for c in VSX_COLS:
                    if c not in df.columns:
                        df[c] = None
                for c in ("oid", "ra_deg", "dec_deg", "period", "mag_max", "mag_min"):
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                df = df.dropna(subset=["oid", "ra_deg", "dec_deg"])
                rows = [
                    (
                        int(r.oid),
                        None if pd.isna(r.name) else str(r.name),
                        float(r.ra_deg),
                        float(r.dec_deg),
                        None if pd.isna(r.var_type) else str(r.var_type),
                        None if pd.isna(r.period) else float(r.period),
                        None if pd.isna(r.mag_max) else float(r.mag_max),
                        None if pd.isna(r.mag_min) else float(r.mag_min),
                    )
                    for r in df[VSX_COLS].itertuples(index=False)
                ]
                added = _insert(conn, table, VSX_COLS, rows)
                added_total += added
                print(f"   {len(rows)} rows ({added} new)")
                break
            except Exception as exc:  # noqa: BLE001 - VizieR timeouts: retry with back-off
                if attempt == 4:
                    raise DownloadError(f"VSX strip Dec {d0} .. {d1} failed 5 times: {exc}") from exc
                wait = (attempt + 1) * 20
                print(f"   attempt {attempt + 1} failed: {exc}; waiting {wait} s")
                time.sleep(wait)
        d0 = d1
    return added_total


# ---------------------------------------------------------------------------
# exoplanet download (NASA Exoplanet Archive TAP)
# ---------------------------------------------------------------------------
def _tap_csv(adql: str) -> str:
    url = TAP_BASE + "?" + urllib.parse.urlencode({"query": adql, "format": "csv"})
    req = urllib.request.Request(url, headers={"User-Agent": "VYVAR-objects-builder/1.0"})
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _select_columns(table: str) -> list[str]:
    """Columns to fetch, checked against the live TAP_SCHEMA (abort when a required one is missing)."""
    import pandas as pd  # noqa: PLC0415

    df = pd.read_csv(io.StringIO(_tap_csv(f"select column_name from TAP_SCHEMA.columns where table_name = '{table}'")))
    col = "column_name" if "column_name" in df.columns else df.columns[0]
    avail = {str(c).strip() for c in df[col].tolist()}
    req, opt = _EXO_SRC[table]["required"], _EXO_SRC[table]["optional"]
    missing_req = [c for c in req if c not in avail]
    if missing_req:
        raise DownloadError(f"NASA table '{table}' lacks required columns {missing_req}")
    missing_opt = [c for c in opt if c not in avail]
    if missing_opt:
        print(f"   WARNING '{table}': optional columns {missing_opt} missing -> NULL")
    return [c for c in req + opt if c in avail]


def _exo_rows(table: str, df) -> list[tuple]:  # noqa: ANN001
    import pandas as pd  # noqa: PLC0415

    def num(name: str):  # noqa: ANN202
        return pd.to_numeric(df[name], errors="coerce") if name in df else pd.Series([float("nan")] * len(df))

    out = []
    if table == "pscomppars":
        tmag, vmag, per = num("sy_tmag"), num("sy_vmag"), num("pl_orbper")
        for i in range(len(df)):
            name = df["pl_name"].iloc[i]
            t, v = tmag.iloc[i], vmag.iloc[i]
            mag, band = (t, "TESS") if pd.notna(t) else ((v, "V") if pd.notna(v) else (None, None))
            out.append((
                None if pd.isna(name) else str(name), None if pd.isna(name) else str(name),
                str(df["hostname"].iloc[i]) if "hostname" in df and pd.notna(df["hostname"].iloc[i]) else None,
                pd.to_numeric(df["ra"].iloc[i], errors="coerce"), pd.to_numeric(df["dec"].iloc[i], errors="coerce"),
                "CONFIRMED", "CONFIRMED",
                None if pd.isna(per.iloc[i]) else float(per.iloc[i]),
                None if mag is None else float(mag), band,
            ))
    else:
        tmag, per, tid = num("st_tmag"), num("pl_orbper"), num("tid")
        for i in range(len(df)):
            try:
                oid = f"TOI-{float(df['toi'].iloc[i]):.2f}"
            except (TypeError, ValueError):
                oid = None
            t = tmag.iloc[i]
            disp = df["tfopwg_disp"].iloc[i] if "tfopwg_disp" in df else None
            out.append((
                oid, oid,
                "TIC " + str(int(tid.iloc[i])) if pd.notna(tid.iloc[i]) else None,
                pd.to_numeric(df["ra"].iloc[i], errors="coerce"), pd.to_numeric(df["dec"].iloc[i], errors="coerce"),
                "TOI", None if disp is None or pd.isna(disp) else str(disp),
                None if pd.isna(per.iloc[i]) else float(per.iloc[i]),
                None if pd.isna(t) else float(t), "TESS" if pd.notna(t) else None,
            ))
    keep = []
    for r in out:
        if r[0] is None or r[3] is None or r[4] is None or pd.isna(r[3]) or pd.isna(r[4]):
            continue
        keep.append(r[:3] + (float(r[3]), float(r[4])) + r[5:])
    return keep


def build_exoplanets(conn: sqlite3.Connection, *, mag_limit: float | None, confirmed_only: bool,
                     table_out: str = "exoplanet_data") -> int:
    import pandas as pd  # noqa: PLC0415

    added_total = 0
    for table in (["pscomppars"] if confirmed_only else ["pscomppars", "toi"]):
        print(f"[EXO] {table}")
        magcol = "sy_tmag" if table == "pscomppars" else "st_tmag"
        for attempt in range(5):
            try:
                # the column check is a request too: same retries (HTTP 526 from the archive, 2026-10-20)
                cols = _select_columns(table)
                where = f" where {magcol} < {float(mag_limit)}" if mag_limit is not None and magcol in cols else ""
                df = pd.read_csv(io.StringIO(_tap_csv(f"select {','.join(cols)} from {table}{where}")))
                rows = _exo_rows(table, df)
                added = _insert(conn, table_out, EXO_COLS, rows)
                added_total += added
                print(f"   {len(rows)} rows ({added} new)")
                break
            except DownloadError:
                raise
            except Exception as exc:  # noqa: BLE001 - archive timeouts / HTTP errors: retry with back-off
                if attempt == 4:
                    raise DownloadError(f"NASA archive table {table} failed 5 times: {exc}") from exc
                wait = (attempt + 1) * 15
                print(f"   attempt {attempt + 1} failed: {exc}; waiting {wait} s")
                time.sleep(wait)
    return added_total


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="", help="objects DB (default: objects_db_path of config.json, else <data>/catalogs/vyvar_objects.db)")
    ap.add_argument("--vsx", action="store_true", help="download VSX (VizieR B/vsx/vsx)")
    ap.add_argument("--vsx-dec-min", type=float, default=-90.0)
    ap.add_argument("--vsx-dec-max", type=float, default=90.0)
    ap.add_argument("--vsx-mag-limit", type=float, default=VSX_BUILD_MAG_LIMIT_DEFAULT, help="cut on VSX max (brightness at maximum)")
    ap.add_argument("--vsx-batch-deg", type=float, default=1.0)
    ap.add_argument("--exoplanets", action="store_true", help="download the NASA Exoplanet Archive")
    ap.add_argument("--exo-mag-limit", type=float, default=None, help="cut on host TESS mag (default: none)")
    ap.add_argument("--confirmed-only", action="store_true", help="confirmed planets only, no TOI candidates")
    ap.add_argument("--import-vsx", default="", help="copy vsx_data of an existing SQLite file (no download)")
    ap.add_argument("--import-exoplanets", default="", help="copy exoplanet_data of an existing SQLite file")
    ap.add_argument("--force", action="store_true", help="replace a table even when the download is much smaller")
    args = ap.parse_args(argv)
    if not (args.vsx or args.exoplanets or args.import_vsx or args.import_exoplanets):
        ap.error("choose at least one of --vsx, --exoplanets, --import-vsx, --import-exoplanets")

    db = Path(args.db).expanduser() if str(args.db).strip() else default_db_path()
    db.parent.mkdir(parents=True, exist_ok=True)
    print(f"objects DB: {db}")
    conn = connect_rw(db)
    try:
        return _run(conn, args)
    finally:
        conn.close()


def _run(conn: sqlite3.Connection, args: argparse.Namespace) -> int:
    ensure_schema(conn)
    if args.import_vsx:
        src = Path(args.import_vsx).expanduser()
        n = import_table(conn, src, "vsx_data", VSX_COLS)
        print(f"[VSX] imported {n} rows from {src}")
        write_catalog_meta(conn, "vsx_data", source=f"{VSX_SOURCE} (imported from {src.name})",
                           params={"imported_from": str(src.resolve())}, builder=BUILDER)
    failed: list[str] = []
    if args.vsx:
        try:
            diff = refresh_table(conn, "vsx_data", lambda c, t: build_vsx(
                c, dec_min=args.vsx_dec_min, dec_max=args.vsx_dec_max, mag_limit=args.vsx_mag_limit,
                batch_deg=args.vsx_batch_deg, table=t), force=args.force)
            write_catalog_meta(conn, "vsx_data", source=VSX_SOURCE, builder=BUILDER, params={
                "dec_min": args.vsx_dec_min, "dec_max": args.vsx_dec_max, "mag_limit_max": args.vsx_mag_limit,
                "update": diff})
        except DownloadError as exc:  # the exoplanet table is independent: still update it
            print(f"ERROR: {exc}\n[vsx_data] unchanged.")
            failed.append("vsx_data")
    if args.import_exoplanets:
        src = Path(args.import_exoplanets).expanduser()
        n = import_table(conn, src, "exoplanet_data", EXO_COLS)
        print(f"[EXO] imported {n} rows from {src}")
        write_catalog_meta(conn, "exoplanet_data", source=f"{EXO_SOURCE} (imported from {src.name})",
                           params={"imported_from": str(src.resolve())}, builder=BUILDER)
    if args.exoplanets:
        try:
            diff = refresh_table(conn, "exoplanet_data", lambda c, t: build_exoplanets(
                c, mag_limit=args.exo_mag_limit, confirmed_only=args.confirmed_only, table_out=t), force=args.force)
            write_catalog_meta(conn, "exoplanet_data", source=EXO_SOURCE, builder=BUILDER, params={
                "mag_limit_tess": args.exo_mag_limit, "confirmed_only": bool(args.confirmed_only), "update": diff})
        except DownloadError as exc:
            print(f"ERROR: {exc}\n[exoplanet_data] unchanged.")
            failed.append("exoplanet_data")
    print("VACUUM ...")
    conn.execute("VACUUM")
    for t in ("vsx_data", "exoplanet_data"):
        print(f"{t}: {_count(conn, t)} rows")
    if failed:
        print(f"FAILED: {', '.join(failed)} not updated (download incomplete); the other tables are current.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
