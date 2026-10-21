#!/usr/bin/env python3
"""Diff anchor snapshot photometry manifest against live tree (Archive-local tripwire)."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_LIVE = REPO / "Archive" / "Drafts" / "draft_000435_snapshot_skysurface_20260716"
DEFAULT_MANIFEST = REPO / "dev" / "results" / "anchor_restore" / "manifest_restored_20260804.txt"
PHOT_REL = Path("platesolve/NoFilter_60_2/photometry")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(*, live_root: Path, manifest_path: Path) -> tuple[int, int]:
    phot = live_root / PHOT_REL
    if not phot.is_dir():
        raise FileNotFoundError(f"photometry dir missing: {phot}")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# manifest: platesolve/NoFilter_60_2/photometry/ only",
        f"# generated {ts}",
        "relative_path\tsize_bytes\tmtime_iso\tsha256",
    ]
    count = 0
    total = 0
    for root, _, files in os.walk(phot):
        for fn in files:
            fp = Path(root) / fn
            rel = fp.relative_to(live_root).as_posix()
            st = fp.stat()
            digest = _sha256_file(fp)
            mtime = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            lines.append(f"{rel}\t{st.st_size}\t{mtime}\t{digest}")
            count += 1
            total += st.st_size
    manifest_path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return count, total


def _parse_manifest(manifest_path: Path) -> list[tuple[str, int, str, str]]:
    rows: list[tuple[str, int, str, str]] = []
    for line in manifest_path.read_text(encoding="ascii").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 4:
            continue
        rel, size_s, mtime, digest = parts
        if rel == "relative_path" or size_s == "size_bytes":
            continue
        try:
            size = int(size_s)
        except ValueError:
            continue
        rows.append((rel, size, mtime, digest))
    return rows


def check_manifest(*, live_root: Path, manifest_path: Path) -> list[str]:
    if not manifest_path.is_file():
        return [f"manifest missing: {manifest_path}"]
    errors: list[str] = []
    manifest_rows = _parse_manifest(manifest_path)
    manifest_rels = {rel for rel, _, _, _ in manifest_rows}

    for rel, size, _mtime, digest in manifest_rows:
        local = live_root / rel.replace("/", os.sep)
        if not local.is_file():
            errors.append(f"MISSING {rel}")
            continue
        st = local.stat()
        if st.st_size != size:
            errors.append(f"SIZE {rel} manifest={size} live={st.st_size}")
        live_digest = _sha256_file(local)
        if live_digest != digest:
            errors.append(f"SHA256 {rel}")

    phot = live_root / PHOT_REL
    if phot.is_dir():
        for root, _, files in os.walk(phot):
            for fn in files:
                fp = Path(root) / fn
                rel = fp.relative_to(live_root).as_posix()
                if rel not in manifest_rels:
                    errors.append(f"EXTRA {rel}")

    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--live",
        type=Path,
        default=DEFAULT_LIVE,
        help="Live snapshot root (default: draft_435 snapshot)",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Manifest file to read or write",
    )
    ap.add_argument(
        "--write",
        action="store_true",
        help="Write manifest from live tree (does not check)",
    )
    args = ap.parse_args()

    if args.write:
        count, total = write_manifest(live_root=args.live, manifest_path=args.manifest)
        print(f"manifest_files={count}")
        print(f"manifest_bytes={total}")
        print(f"manifest_path={args.manifest}")
        return 0

    errors = check_manifest(live_root=args.live, manifest_path=args.manifest)
    if errors:
        print(f"FAIL differences={len(errors)}", file=sys.stderr)
        for err in errors[:50]:
            print(err, file=sys.stderr)
        if len(errors) > 50:
            print(f"... and {len(errors) - 50} more", file=sys.stderr)
        return 1
    print("PASS manifest matches live tree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
