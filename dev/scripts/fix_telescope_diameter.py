#!/usr/bin/env python3
"""Maintenance: correct Carl-Zeiss wide-rig TELESCOPE.DIAMETER (72 mm -> 200 mm)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# dev/scripts/<x>.py -> repo root is parents[2]. VYVAR modules live under src_py after the
# reorg (repo root before it); add both so flat imports resolve when run standalone.
_REPO = Path(__file__).resolve().parents[2]
for _p in (_REPO / "src_py", _REPO):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from config import AppConfig  # noqa: E402
from database import VyvarDatabase  # noqa: E402

TARGET_NAME = "Carl-Zeiss"
CORRECT_DIAMETER_MM = 200.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="Write DB update (default: dry-run)")
    args = ap.parse_args()
    cfg = AppConfig()
    db = VyvarDatabase(cfg.database_path)
    rows = db.conn.execute(
        "SELECT ID, TELESCOPENAME, DIAMETER, FOCAL FROM TELESCOPE WHERE TELESCOPENAME LIKE ?;",
        (f"%{TARGET_NAME}%",),
    ).fetchall()
    if not rows:
        print("No matching TELESCOPE rows found.")
        return
    for row in rows:
        rd = dict(row)
        before = float(rd["DIAMETER"])
        print(
            f"ID={rd['ID']} name={rd['TELESCOPENAME']!r} "
            f"DIAMETER before={before} mm FOCAL={rd['FOCAL']} mm"
        )
        if args.apply and abs(before - CORRECT_DIAMETER_MM) > 1e-6:
            db.conn.execute(
                "UPDATE TELESCOPE SET DIAMETER = ? WHERE ID = ?;",
                (CORRECT_DIAMETER_MM, int(rd["ID"])),
            )
            db.conn.commit()
            after = float(
                db.conn.execute(
                    "SELECT DIAMETER FROM TELESCOPE WHERE ID = ?;", (int(rd["ID"]),)
                ).fetchone()[0]
            )
            print(f"  DIAMETER after={after} mm (applied)")
        elif args.apply:
            print("  unchanged (already correct)")
        else:
            print(f"  would set DIAMETER={CORRECT_DIAMETER_MM} mm (dry-run; pass --apply)")


if __name__ == "__main__":
    main()
