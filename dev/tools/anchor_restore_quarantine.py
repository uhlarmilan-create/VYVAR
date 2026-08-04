#!/usr/bin/env python3
"""Quarantine mutated snapshot and write photometry manifest.

One-shot ANCHOR-RESTORE-1 artifact retained for archaeology; do not re-run against
a live tree without editing paths.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

ARCHIVE = Path(r"C:\ASTRO\python\VYVAR\Archive")
LIVE = ARCHIVE / "Drafts" / "draft_000435_snapshot_skysurface_20260716"
QUAR = ARCHIVE / "Drafts" / "_quarantine" / "draft_000435_snapshot_MUTATED_20260804_1147"
MANIFEST = Path(r"c:\ASTRO\python\VYVAR\dev\results\anchor_restore\manifest_mutated_20260804.txt")
PHOT = LIVE / "platesolve" / "NoFilter_60_2" / "photometry"


def main() -> int:
    if not LIVE.is_dir():
        print("ERROR: live tree missing", LIVE, file=sys.stderr)
        return 1
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# manifest: platesolve/NoFilter_60_2/photometry/ only",
        f"# generated {ts}",
        "relative_path\tsize_bytes\tmtime_iso\tsha256",
    ]
    count = 0
    total = 0
    for root, _, files in os.walk(PHOT):
        for fn in files:
            fp = Path(root) / fn
            rel = fp.relative_to(LIVE).as_posix()
            st = fp.stat()
            h = hashlib.sha256()
            with fp.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            mtime = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            lines.append(f"{rel}\t{st.st_size}\t{mtime}\t{h.hexdigest()}")
            count += 1
            total += st.st_size
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="ascii")
    print(f"manifest_files={count}")
    print(f"manifest_bytes={total}")
    print(f"manifest_path={MANIFEST}")

    QUAR.parent.mkdir(parents=True, exist_ok=True)
    if QUAR.exists():
        print("ERROR: quarantine path already exists", QUAR, file=sys.stderr)
        return 1
    shutil.move(str(LIVE), str(QUAR))
    print(f"moved_to={QUAR}")
    print(f"live_exists={LIVE.exists()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
