#!/usr/bin/env python3
"""Launch anchor pair from a clean git worktree (porcelain empty at run start)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401  (repo layout: src_py + dev on sys.path)
ROOT = _bootstrap.REPO_ROOT
WT = ROOT / "tmp" / "anchor_run_wt"
LOG = ROOT / "tmp" / "anchor_pair_run.log"


def main() -> int:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if WT.exists():
        subprocess.run(["git", "worktree", "remove", "--force", str(WT)], cwd=ROOT, check=False)
    subprocess.run(["git", "worktree", "add", str(WT), head], cwd=ROOT, check=True)
    porcelain = subprocess.check_output(["git", "status", "--porcelain"], cwd=WT, text=True)
    if porcelain.strip():
        print("ERROR: worktree not clean:", porcelain, file=sys.stderr)
        return 1
    cmd = [
        sys.executable,
        str(WT / "scripts" / "anchor_pair_run.py"),
        "--expected-git",
        head,
        "--finalize",
    ]
    print("Launching:", " ".join(cmd))
    print("Log:", LOG)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("w", encoding="utf-8") as logf:
        proc = subprocess.Popen(cmd, cwd=WT, stdout=logf, stderr=subprocess.STDOUT)
    print(f"PID {proc.pid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
