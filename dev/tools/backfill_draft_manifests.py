#!/usr/bin/env python3
"""Backfill draft_manifest.json v3 (draft core + files[]) from OBS_DRAFT/OBS_FILES (idempotent)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src_py"))

from config import AppConfig
from database import VyvarDatabase
from draft_provenance import backfill_draft_manifest_from_db


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Path to vyvar.sqlite3 (default: config.json database_path)",
    )
    args = ap.parse_args()

    cfg = AppConfig(project_root=REPO)
    db_path = Path(args.db) if args.db is not None else Path(cfg.database_path)
    if not db_path.is_file():
        print(f"FAIL database missing: {db_path}", file=sys.stderr)
        return 1

    db = VyvarDatabase(db_path)
    try:
        rows = db.conn.execute("SELECT ID FROM OBS_DRAFT ORDER BY ID;").fetchall()
        ok = 0
        skipped = 0
        for row in rows:
            did = int(row["ID"])
            path = backfill_draft_manifest_from_db(db, did)
            if path is None:
                skipped += 1
            else:
                ok += 1
        print(f"backfill ok={ok} skipped={skipped} total={len(rows)}")
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
