#!/usr/bin/env python3
"""Phase 2.1: exercise rig-id accessors and report manifest shadow counters."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src_py"))

from config import AppConfig
from database import VyvarDatabase
from draft_provenance import (
    clear_manifest_shadow_load_cache,
    manifest_shadow_counter_snapshot,
    reset_manifest_shadow_counters,
)
from param_resolver import _draft_id_location, _draft_location


def _exercise_draft(db: VyvarDatabase, draft_id: int) -> None:
    db.fetch_obs_draft_by_id(int(draft_id))
    db.fetch_obs_draft_telescope_equipment(int(draft_id))
    _draft_id_location(db, int(draft_id))
    _draft_location(db, int(draft_id))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Path to vyvar.sqlite3 (default: config.json database_path)",
    )
    ap.add_argument(
        "--draft-id",
        type=int,
        action="append",
        dest="draft_ids",
        help="Draft ID to exercise (repeatable; default: all OBS_DRAFT rows)",
    )
    args = ap.parse_args()

    cfg = AppConfig(project_root=REPO)
    db_path = Path(args.db) if args.db is not None else Path(cfg.database_path)
    if not db_path.is_file():
        print(f"FAIL database missing: {db_path}", file=sys.stderr)
        return 1

    reset_manifest_shadow_counters()
    clear_manifest_shadow_load_cache()

    db = VyvarDatabase(db_path)
    try:
        if args.draft_ids:
            draft_ids = [int(d) for d in args.draft_ids]
        else:
            draft_ids = [int(r["ID"]) for r in db.conn.execute("SELECT ID FROM OBS_DRAFT ORDER BY ID;")]
        for did in draft_ids:
            _exercise_draft(db, did)
        snap = manifest_shadow_counter_snapshot()
        print(
            f"drafts={len(draft_ids)} equal={snap['equal']} absent={snap['absent']} "
            f"mismatch={snap['mismatch']}"
        )
        if snap["mismatch"] > 0:
            print("FAIL: manifest rig shadow mismatches detected", file=sys.stderr)
            return 1
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
