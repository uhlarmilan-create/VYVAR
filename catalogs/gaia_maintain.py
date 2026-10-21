"""gaia_maintain.py - maintenance of the local Gaia DR3 catalog (CATALOG-DB-2026-10, Milan 2026-10-20).

Standard library only. Close VYVAR before running it (the DB must not be in use).

  --report        indexes, file size, catalog_meta, coverage (read-only)
  --drop-unused   drop the indexes no VYVAR query uses: idx_ra (covered by idx_ra_dec),
                  idx_teff and idx_parallax_snr (both columns are read by source_id only)
  --register      write the catalog_meta row of gaia_dr3 (source, build date, depth, sky, rows)
  --vacuum-into P write a compact copy of the DB to P (VACUUM INTO). The original stays as it is
                  until you delete it; the copy is checked (integrity, row count, tables).

Examples (from C:\\ASTRO\\python\\VYVAR):
  python catalogs\\gaia_maintain.py --report
  python catalogs\\gaia_maintain.py --drop-unused --register
  python catalogs\\gaia_maintain.py --vacuum-into catalogs\\vyvar_gaia_dr3.db
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
#: no VYVAR query uses them (checked with EXPLAIN QUERY PLAN, 2026-10-20)
UNUSED_INDEXES = ("idx_ra", "idx_teff", "idx_parallax_snr")
#: indexes VYVAR needs: cone / box (idx_ra_dec), dec strips of the index build (idx_dec), magnitude (idx_g_mag)
NEEDED_INDEXES = ("idx_ra_dec", "idx_dec", "idx_g_mag")

#: same schema as src_py/catalog_db.CATALOG_META_SCHEMA (test_catalog_db checks it)
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


def default_db() -> Path:
    for p in (ROOT / "catalogs" / "vyvar_gaia_dr3.db", ROOT / "GAIA_DR3" / "vyvar_gaia_dr3.db"):
        if p.is_file():
            return p
    return ROOT / "catalogs" / "vyvar_gaia_dr3.db"


def gb(n: float) -> str:
    return f"{n / 1024 ** 3:.1f} GB"


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def connect(db: Path, *, write: bool) -> sqlite3.Connection:
    if not db.is_file():
        sys.exit(f"ERROR: Gaia DB not found: {db}")
    if write:
        con = sqlite3.connect(str(db), timeout=5)
        try:
            con.execute("BEGIN EXCLUSIVE")
            con.execute("ROLLBACK")
        except sqlite3.OperationalError:
            sys.exit("ERROR: the Gaia DB is in use. Close VYVAR (and any other program using the DB) and run again.")
        return con
    return sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True)


def indexes(con: sqlite3.Connection) -> list[tuple[str, str]]:
    return [(r[0], r[1] or "") for r in con.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='index' AND tbl_name='gaia_dr3' ORDER BY name")]


def has_table(con: sqlite3.Connection, name: str) -> bool:
    return con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def report(db: Path) -> None:
    con = connect(db, write=False)
    try:
        ps = con.execute("PRAGMA page_size").fetchone()[0]
        fl = con.execute("PRAGMA freelist_count").fetchone()[0]
        print(f"DB:            {db}")
        print(f"file size:     {gb(db.stat().st_size)}  (free pages inside: {gb(fl * ps)})")
        print("indexes on gaia_dr3:")
        for name, sql in indexes(con):
            tag = "UNUSED - drop" if name in UNUSED_INDEXES else ("needed" if name in NEEDED_INDEXES else "?")
            print(f"  {name:18s} {tag:15s} {sql}")
        if has_table(con, "gaia_coverage"):
            for r in con.execute("SELECT ra_min, ra_max, dec_min, dec_max, g_max, source, n_stars, created_utc FROM gaia_coverage"):
                print(f"coverage:      RA {r[0]:g}-{r[1]:g}, Dec {r[2]:g}..{r[3]:g}, G <= {r[4]:g}, {r[5]}, {r[6]} stars, {r[7]}")
        if has_table(con, "catalog_meta"):
            for r in con.execute("SELECT table_name, source, built_utc, row_count, params_json FROM catalog_meta"):
                print(f"catalog_meta:  {r}")
        else:
            print("catalog_meta:  none (run --register)")
        free = shutil.disk_usage(db.parent).free
        print(f"free disk:     {gb(free)} on the drive of the DB")
    finally:
        con.close()


def drop_unused(db: Path) -> None:
    con = connect(db, write=True)
    try:
        present = {n for n, _ in indexes(con)}
        missing_needed = [n for n in NEEDED_INDEXES if n not in present]
        if missing_needed:
            sys.exit(f"ERROR: needed indexes missing {missing_needed} - not dropping anything.")
        for name in UNUSED_INDEXES:
            if name not in present:
                log(f"{name}: not present")
                continue
            t0 = time.monotonic()
            log(f"dropping {name} ...")
            con.execute(f"DROP INDEX {name}")
            con.commit()
            log(f"dropped {name} in {time.monotonic() - t0:.0f} s")
        fl = con.execute("PRAGMA freelist_count").fetchone()[0] * con.execute("PRAGMA page_size").fetchone()[0]
        log(f"free pages now {gb(fl)} - the file shrinks only with --vacuum-into")
    finally:
        con.close()


def register(db: Path) -> None:
    con = connect(db, write=True)
    try:
        cov = []
        if has_table(con, "gaia_coverage"):
            cov = con.execute("SELECT ra_min, ra_max, dec_min, dec_max, g_max, source, n_stars, created_utc "
                              "FROM gaia_coverage ORDER BY id").fetchall()
        built = None
        if has_table(con, "strip_progress"):
            built = con.execute("SELECT MAX(finished_at) FROM strip_progress").fetchone()[0]
        if not built and cov:
            built = cov[0][7]
        if not built:
            built = datetime.fromtimestamp(db.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        log("counting rows of gaia_dr3 (once, about a minute) ...")
        n = int(con.execute("SELECT COUNT(*) FROM gaia_dr3").fetchone()[0])
        base = [c for c in cov if c[5] != "gaia_tap_topup"]
        params = {"g_max": base[0][4] if base else None,
                  "ra": [base[0][0], base[0][1]] if base else None,
                  "dec": [base[0][2], base[0][3]] if base else None,
                  "online_topups": sum(1 for c in cov if c[5] == "gaia_tap_topup")}
        # gaia_source_lite has no astrophysical parameters: teff empty in the first rows = lite build
        n_teff = con.execute("SELECT COUNT(teff_gspphot) FROM (SELECT teff_gspphot FROM gaia_dr3 LIMIT 100000)").fetchone()[0]
        table = "gaiadr3.gaia_source" if n_teff else "gaiadr3.gaia_source_lite"
        con.execute(CATALOG_META_SCHEMA)
        con.execute(
            "INSERT OR REPLACE INTO catalog_meta (table_name, source, built_utc, row_count, content_sha256, params_json, builder) "
            "VALUES (?, ?, ?, ?, NULL, ?, ?)",
            ("gaia_dr3", f"ESA Gaia DR3 {table} (TAP)", str(built), n, json.dumps(params, sort_keys=True),
             "build_gaia_catalog.py (registered by gaia_maintain.py)"),
        )
        con.commit()
        log(f"catalog_meta written: {n} rows, built {built}, {params}")
    finally:
        con.close()


def vacuum_into(db: Path, target: Path) -> None:
    target = target if target.is_absolute() else (Path.cwd() / target)
    if target.resolve() == db.resolve():
        sys.exit("ERROR: the target must be a new file (the original stays until you delete it).")
    if target.exists():
        sys.exit(f"ERROR: {target} already exists - delete or rename it first.")
    target.parent.mkdir(parents=True, exist_ok=True)
    con = connect(db, write=True)
    try:
        ps = con.execute("PRAGMA page_size").fetchone()[0]
        used = (con.execute("PRAGMA page_count").fetchone()[0] - con.execute("PRAGMA freelist_count").fetchone()[0]) * ps
        free = shutil.disk_usage(target.parent).free
        log(f"copy needs about {gb(used)}; free on the target drive {gb(free)}")
        if free < used * 1.1:
            sys.exit("ERROR: not enough free disk space for the copy.")
        n_src = int(con.execute("SELECT COUNT(*) FROM gaia_dr3").fetchone()[0])
        log(f"VACUUM INTO {target} ... (hours; do not close this window)")
        t0 = time.monotonic()
        con.execute("VACUUM INTO ?", (str(target),))
        log(f"copy written in {(time.monotonic() - t0) / 60:.0f} min, {gb(target.stat().st_size)}")
    finally:
        con.close()
    new = sqlite3.connect(str(target))
    try:
        log("checking the copy (quick_check, rows, tables) ...")
        qc = new.execute("PRAGMA quick_check").fetchone()[0]
        n_new = int(new.execute("SELECT COUNT(*) FROM gaia_dr3").fetchone()[0])
        tables = sorted(r[0] for r in new.execute("SELECT name FROM sqlite_master WHERE type='table'"))
        idx = sorted(n for n, _ in indexes(new))
    finally:
        new.close()
    ok = qc == "ok" and n_new == n_src
    log(f"quick_check={qc}, rows {n_new} (original {n_src}), tables {tables}, indexes {idx}")
    if not ok:
        sys.exit("ERROR: the copy does not match - keep the original and send me this output.")
    log(f"OK. Original {db} is unchanged; delete it after VYVAR works with the copy.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="", help="Gaia DB (default: catalogs/ or GAIA_DR3/vyvar_gaia_dr3.db)")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--drop-unused", action="store_true")
    ap.add_argument("--register", action="store_true")
    ap.add_argument("--vacuum-into", default="")
    a = ap.parse_args(argv)
    db = Path(a.db) if a.db else default_db()
    if not (a.report or a.drop_unused or a.register or a.vacuum_into):
        a.report = True
    t0 = time.monotonic()
    if a.drop_unused:
        drop_unused(db)
    if a.register:
        register(db)
    if a.vacuum_into:
        vacuum_into(db, Path(a.vacuum_into))
    if a.report:
        report(db)
    log(f"done in {(time.monotonic() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
