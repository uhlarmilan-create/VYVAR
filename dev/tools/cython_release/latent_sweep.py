# -*- coding: ascii -*-
"""Latent-bug sweep before release compile (F821 + pyflakes)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from module_list import REPO_ROOT, SRC_PY, module_list

LOG = REPO_ROOT / "tmp" / "cython_release" / "latent_sweep.log"


def _run(cmd: list[str], *, cwd: Path | None = None) -> tuple[int, str]:
    p = subprocess.run(
        cmd,
        cwd=str(cwd or REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    out = (p.stdout or "") + (p.stderr or "")
    return p.returncode, out


def sweep(modules: list[str] | None = None) -> tuple[list[str], list[str]]:
    mods = modules or module_list()
    ruff_hits: list[str] = []
    flake_hits: list[str] = []
    for name in mods:
        path = SRC_PY / f"{name}.py"
        code, out = _run(
            [sys.executable, "-m", "ruff", "check", "--select", "F821", str(path)]
        )
        if code != 0 and out.strip():
            ruff_hits.append(f"=== {name} ===\n{out.strip()}")
        code2, out2 = _run([sys.executable, "-m", "pyflakes", str(path)])
        if code2 != 0 and out2.strip():
            flake_hits.append(f"=== {name} ===\n{out2.strip()}")
    LOG.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"modules={len(mods)}",
        f"ruff_f821_issues={len(ruff_hits)}",
        f"pyflakes_issues={len(flake_hits)}",
        "",
    ]
    if ruff_hits:
        lines.append("--- ruff F821 ---")
        lines.extend(ruff_hits)
    if flake_hits:
        lines.append("--- pyflakes ---")
        lines.extend(flake_hits)
    LOG.write_text("\n".join(lines) + "\n", encoding="ascii")
    return ruff_hits, flake_hits


def main() -> None:
    ruff_hits, flake_hits = sweep()
    print(f"latent sweep: ruff F821={len(ruff_hits)} pyflakes={len(flake_hits)} log={LOG}")
    if ruff_hits or flake_hits:
        sys.exit(1)


if __name__ == "__main__":
    main()
