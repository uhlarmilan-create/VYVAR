# -*- coding: ascii -*-
"""User-facing bundle selftest (--selftest launcher mode)."""
from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

INSTALL = Path(__file__).resolve().parent
SRC = INSTALL / "src_py"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import resolve_data_root  # noqa: E402

# Sibling bundle_layout import via path hack for dev smoke runs
_BUNDLE = INSTALL / "dev" / "tools" / "cython_release" / "bundle"
if _BUNDLE.is_dir() and str(_BUNDLE.parent) not in sys.path:
    sys.path.insert(0, str(_BUNDLE.parent.parent))
    sys.path.insert(0, str(_BUNDLE))


def _module_list() -> list[str]:
    try:
        from module_list import module_list

        return module_list()
    except ImportError:
        # Fallback when running from unpacked bundle without dev tree
        compiled: list[str] = []
        for p in sorted(SRC.glob("*.pyd")) + sorted(SRC.glob("*.so")):
            stem = p.name.split(".cp", 1)[0] if ".cp" in p.name else p.stem
            compiled.append(stem)
        return sorted(set(compiled))


def main() -> int:
    data_root = resolve_data_root(INSTALL)
    print(f"VYVAR selftest platform={platform.platform()}")
    print(f"python={sys.version.split()[0]} executable={sys.executable}")
    print(f"install_dir={INSTALL}")
    print(f"data_dir={data_root}")
    key_deps = (
        "numpy",
        "astropy",
        "photutils",
        "streamlit",
        "pandas",
        "scipy",
    )
    for dep in key_deps:
        mod = importlib.import_module(dep)
        ver = getattr(mod, "__version__", "?")
        print(f"dep {dep}={ver}")
    failures: list[str] = []
    for name in _module_list():
        try:
            mod = importlib.import_module(name)
            path = str(getattr(mod, "__file__", "") or "")
            if not path.endswith((".pyd", ".so")):
                failures.append(f"{name}: not compiled ({path})")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    if failures:
        print("SELFTEST FAIL:")
        for line in failures:
            print(f"  {line}")
        return 1
    print(f"SELFTEST PASS modules={len(_module_list())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
