#!/usr/bin/env python3
"""Build anchor checksum manifest JSON for a draft tree."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(*, draft_id: int, root: Path, label: str, git_head: str) -> dict:
    files: dict[str, dict] = {}
    for fp in sorted(root.rglob("*")):
        if not fp.is_file():
            continue
        rel = fp.relative_to(root).as_posix()
        st = fp.stat()
        files[rel] = {
            "sha256": _sha256(fp),
            "size": st.st_size,
            "mtime_utc": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
        }
    return {
        "version": 1,
        "label": label,
        "draft_id": draft_id,
        "git_head": git_head,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": "sha256",
        "root": root.relative_to(REPO).as_posix(),
        "file_count": len(files),
        "files": files,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft-id", type=int, default=510)
    ap.add_argument("--label", required=True)
    ap.add_argument("--git-head", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = REPO / "Archive" / "Drafts" / f"draft_{args.draft_id:06d}"
    if not args.git_head:
        import subprocess

        args.git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    man = build_manifest(draft_id=args.draft_id, root=root, label=args.label, git_head=args.git_head)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(man, indent=2) + "\n", encoding="ascii")
    print(f"wrote {out} files={man['file_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
