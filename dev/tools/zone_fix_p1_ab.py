# -*- coding: ascii -*-
"""Local P1 headless A/B via fresh subprocess per variant."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src_py"
RUN_ONCE = ROOT / "dev/tools/zone_fix_p1_run_once.py"


def _git_bytes(path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=ROOT)


def _run_variant(label: str, pipeline_bytes: bytes, config_bytes: bytes) -> tuple[str, int]:
    (SRC / "pipeline.py").write_bytes(pipeline_bytes)
    (SRC / "config.py").write_bytes(config_bytes)
    proc = subprocess.run(
        [sys.executable, str(RUN_ONCE)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(f"{label} failed rc={proc.returncode}")
    sha = ""
    nc = 0
    for line in proc.stdout.splitlines():
        if line.startswith("core_sha="):
            sha = line.split("=", 1)[1].strip()
        if line.startswith("core_n="):
            nc = int(line.split("=", 1)[1].strip())
    print(f"{label}: core_sha={sha} core_n={nc}")
    return sha, nc


def main() -> None:
    pre_p = _git_bytes("src_py/pipeline.py")
    pre_c = _git_bytes("src_py/config.py")
    with_p = (SRC / "pipeline.py").read_bytes()
    with_c = (SRC / "config.py").read_bytes()
    if with_p.count(0) or with_c.count(0):
        raise SystemExit("working tree pipeline/config contains null bytes")
    sha_pre, _ = _run_variant("pre_zone_fix_reverted", pre_p, pre_c)
    sha_with, _ = _run_variant("with_zone_fix_legacy_default", with_p, with_c)
    (SRC / "pipeline.py").write_bytes(with_p)
    (SRC / "config.py").write_bytes(with_c)
    print(f"MATCH={sha_pre == sha_with}")
    if sha_pre != sha_with:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
