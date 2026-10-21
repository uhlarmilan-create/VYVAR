# -*- coding: ascii -*-
"""Per-module import smoke in a clean interpreter (compiled .pyd must shadow .py)."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

from module_list import REPO_ROOT, SRC_PY, module_list

LOG = REPO_ROOT / "tmp" / "cython_release" / "smoke_imports.log"


def _is_compiled_path(path: str) -> bool:
    return path.endswith((".pyd", ".so"))


def smoke(modules: list[str] | None = None, *, require_compiled: bool = True) -> list[str]:
    mods = modules or module_list()
    if str(SRC_PY) not in sys.path:
        sys.path.insert(0, str(SRC_PY))
    failures: list[str] = []
    lines: list[str] = []
    for name in mods:
        try:
            if name in sys.modules:
                del sys.modules[name]
            mod = importlib.import_module(name)
            f = str(getattr(mod, "__file__", "") or "")
            compiled = _is_compiled_path(f)
            if require_compiled and not compiled:
                failures.append(f"{name}: not compiled ({f})")
            lines.append(f"OK {name} -> {f} compiled={compiled}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            lines.append(f"FAIL {name}: {exc}")
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("\n".join(lines) + "\n", encoding="ascii")
    return failures


def main() -> None:
    failures = smoke()
    if failures:
        print("smoke FAIL:")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    print(f"smoke PASS {len(module_list())} modules; log {LOG}")


if __name__ == "__main__":
    main()
