#!/usr/bin/env python3
"""One-time DB hygiene swap: copy readable tables to a fresh post-C3 schema DB.

MS-SOURCES-RETIRE Phase 3. Run with the VYVAR app CLOSED.

Usage (from repo root):
    python dev/tools/db_hygiene_swap.py [--db PATH] [--dry-run]
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src_py") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src_py"))

from database import VyvarDatabase  # noqa: E402

COPY_TABLES: tuple[str, ...] = (
    "EQUIPMENTS",
    "TELESCOPE",
    "LOCATION",
    "CALIBRATION_LIBRARY",
    "FITS_HEADER_CACHE",
    "OBS_QC_PROCESSING_RUN",
    "OBS_QC_PROCESSING_FILE",
    "FIELD_REGISTRY",
    "COMP_STAR_LIBRARY",
)

# Legacy mirror; no active read/write path (MS-SOURCES-RETIRE audit).
EXCLUDED_TABLES: tuple[str, ...] = ("LOCATION_OLD", "MASTER_SOURCES")


def _table_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
    return {str(r[0]) for r in rows}


def _row_count(conn: sqlite3.Connection, table: str) -> int | str:
    try:
        return int(conn.execute(f"SELECT COUNT(*) FROM {table};").fetchone()[0])
    except sqlite3.Error as exc:
        return f"ERROR: {exc}"


def build_fresh_db(out_path: Path) -> sqlite3.Connection:
    if out_path.exists():
        out_path.unlink()
    VyvarDatabase(out_path)
    return sqlite3.connect(out_path)


def copy_tables(src: sqlite3.Connection, dst: sqlite3.Connection) -> dict[str, int | str]:
    counts: dict[str, int | str] = {}
    src_tables = _table_names(src)
    for table in COPY_TABLES:
        if table not in src_tables:
            counts[table] = 0
            continue
        cols = [str(r[1]) for r in src.execute(f"PRAGMA table_info('{table}');").fetchall()]
        if not cols:
            counts[table] = 0
            continue
        col_list = ", ".join(cols)
        qmarks = ", ".join(["?"] * len(cols))
        rows = src.execute(f"SELECT {col_list} FROM {table};").fetchall()
        dst.executemany(
            f"INSERT INTO {table} ({col_list}) VALUES ({qmarks});",
            [tuple(r) for r in rows],
        )
        counts[table] = len(rows)
    dst.commit()
    return counts


def integrity_ok(conn: sqlite3.Connection) -> tuple[bool, str]:
    row = conn.execute("PRAGMA integrity_check;").fetchone()
    msg = str(row[0]) if row else ""
    return msg.lower() == "ok", msg


def main() -> int:
    parser = argparse.ArgumentParser(description="VYVAR DB hygiene swap (MS-SOURCES-RETIRE Phase 3)")
    parser.add_argument("--db", type=Path, default=REPO_ROOT / "vyvar.sqlite3", help="Production DB path")
    parser.add_argument("--dry-run", action="store_true", help="Report counts only; do not swap files")
    args = parser.parse_args()

    db_path = Path(args.db).expanduser().resolve()
    if not db_path.is_file():
        print(f"ERROR: DB not found: {db_path}")
        return 1

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    backup_name = f"{db_path.name}.corrupt-{stamp}"
    backup_path = db_path.with_name(backup_name)
    new_path = db_path.with_suffix(".sqlite3.new")

    src = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    src_tables = _table_names(src)
    print("Source tables:", sorted(src_tables))
    print("Excluded:", list(EXCLUDED_TABLES))
    print("\nSource row counts:")
    for table in COPY_TABLES:
        if table in src_tables:
            print(f"  {table}: {_row_count(src, table)}")

    if args.dry_run:
        src.close()
        print("\nDRY RUN - no files changed.")
        return 0

    dst = build_fresh_db(new_path)
    try:
        counts = copy_tables(src, dst)
        ok, msg = integrity_ok(dst)
        if not ok:
            print(f"ERROR: integrity_check failed on new DB: {msg}")
            return 1
        print("\nCopied row counts:")
        for table, n in counts.items():
            print(f"  {table}: {n}")
        print(f"\nintegrity_check: {msg}")
    finally:
        src.close()
        dst.close()

    if db_path.exists():
        shutil.move(str(db_path), str(backup_path))
    shutil.move(str(new_path), str(db_path))
    print(f"\nSwapped: {db_path.name}")
    print(f"Forensic artifact kept: {backup_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
