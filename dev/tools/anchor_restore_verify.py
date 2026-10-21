#!/usr/bin/env python3
"""Verify restored snapshot photometry against zip."""
from __future__ import annotations

import hashlib
import random
import zipfile
from pathlib import Path

ZIP_PATH = Path(r"C:\ASTRO\backups\draft_000435_snapshot_skysurface_20260716.zip")
LIVE = Path(
    r"C:\ASTRO\python\VYVAR\Archive\Drafts\draft_000435_snapshot_skysurface_20260716"
)
PREFIX = "draft_000435_snapshot_skysurface_20260716/platesolve/NoFilter_60_2/photometry/"


def main() -> None:
    phot_entries: list[str] = []
    size_mismatches: list[str] = []
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        for name in z.namelist():
            if name.startswith(PREFIX) and not name.endswith("/"):
                phot_entries.append(name)
        for name in phot_entries:
            info = z.getinfo(name)
            rel = name[len("draft_000435_snapshot_skysurface_20260716/") :]
            local = LIVE / rel.replace("/", "\\")
            if not local.is_file():
                size_mismatches.append(f"MISSING {rel}")
                continue
            if local.stat().st_size != info.file_size:
                size_mismatches.append(
                    f"SIZE {rel} zip={info.file_size} local={local.stat().st_size}"
                )

        sample = random.sample(phot_entries, min(50, len(phot_entries)))
        hash_mismatches: list[str] = []
        for name in sample:
            rel = name[len("draft_000435_snapshot_skysurface_20260716/") :]
            local = LIVE / rel.replace("/", "\\")
            zdata = z.read(name)
            zsha = hashlib.sha256(zdata).hexdigest()
            lsha = hashlib.sha256(local.read_bytes()).hexdigest()
            if zsha != lsha:
                hash_mismatches.append(rel)

    print(f"photometry_zip_entries={len(phot_entries)}")
    print(f"size_mismatches={len(size_mismatches)}")
    for m in size_mismatches[:10]:
        print(" ", m)
    print(f"hash_sample_n={len(sample)}")
    print(f"hash_mismatches={len(hash_mismatches)}")
    for m in hash_mismatches:
        print(" ", m)


if __name__ == "__main__":
    main()
