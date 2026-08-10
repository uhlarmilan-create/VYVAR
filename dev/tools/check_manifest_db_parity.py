#!/usr/bin/env python3
"""Assert draft_manifest.json core fields match OBS_DRAFT (Phase 1 parity gate)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src_py"))

from config import AppConfig
from database import VyvarDatabase
from draft_provenance import manifest_db_parity_errors, record_draft_manifest_core


def check_draft(db: VyvarDatabase, draft_id: int, *, backfill: bool) -> list[str]:
    if backfill:
        record_draft_manifest_core(db, int(draft_id))
    return manifest_db_parity_errors(db, int(draft_id))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, default=None, help="vyvar.sqlite3 path")
    ap.add_argument("--draft-id", type=int, required=True, help="OBS_DRAFT.ID")
    ap.add_argument(
        "--backfill",
        action="store_true",
        help="Write manifest from DB before checking",
    )
    args = ap.parse_args()

    cfg = AppConfig(project_root=REPO)
    db_path = Path(args.db) if args.db is not None else Path(cfg.database_path)
    if not db_path.is_file():
        print(f"FAIL database missing: {db_path}", file=sys.stderr)
        return 1

    db = VyvarDatabase(db_path)
    try:
        errors = check_draft(db, int(args.draft_id), backfill=bool(args.backfill))
    finally:
        db.close()

    if errors:
        print(f"FAIL {len(errors)} mismatch(es)", file=sys.stderr)
        for err in errors:
            print(err, file=sys.stderr)
        return 1
    print(f"PASS draft_id={int(args.draft_id)} manifest matches OBS_DRAFT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
