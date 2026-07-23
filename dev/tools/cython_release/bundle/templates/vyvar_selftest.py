# -*- coding: ascii -*-
"""User-facing bundle selftest (--selftest launcher mode)."""
from __future__ import annotations

import importlib
import json
import os
import platform
import sys
from pathlib import Path

INSTALL = Path(__file__).resolve().parent
SRC = INSTALL / "src_py"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import resolve_data_root  # noqa: E402

PINNED_DEPS = (
    "numpy",
    "astropy",
    "photutils",
    "streamlit",
    "pandas",
    "scipy",
)


def _module_list() -> list[str]:
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "dev" / "tools" / "cython_release"))
        from module_list import module_list

        return module_list()
    except ImportError:
        compiled: list[str] = []
        for p in sorted(SRC.glob("*.pyd")) + sorted(SRC.glob("*.so")):
            stem = p.name.split(".cp", 1)[0] if ".cp" in p.name else p.stem
            compiled.append(stem)
        return sorted(set(compiled))


def _bundle_python_roots(install: Path) -> list[Path]:
    roots: list[Path] = []
    for rel in ("python", "python/python", "python/Lib/site-packages", "python/python/lib"):
        p = install / rel
        if p.is_dir():
            roots.append(p.resolve())
    return roots


def _load_runtime_pin(install: Path) -> dict:
    pin_path = install / "RUNTIME_PIN.json"
    if not pin_path.is_file():
        raise SystemExit(f"SELFTEST FAIL: missing {pin_path.name}")
    return json.loads(pin_path.read_text(encoding="utf-8"))


def _path_under_bundle(path: Path, install: Path, roots: list[Path]) -> bool:
    resolved = path.resolve()
    install_resolved = install.resolve()
    if str(resolved).startswith(str(install_resolved)):
        return True
    for root in roots:
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _verify_pinned_deps(install: Path, pin: dict) -> None:
    expected: dict[str, str] = pin.get("dep_versions") or {}
    if not expected:
        raise SystemExit("SELFTEST FAIL: RUNTIME_PIN.json missing dep_versions (rebuild bundle)")
    roots = _bundle_python_roots(install)
    for dep in PINNED_DEPS:
        exp_ver = expected.get(dep)
        if not exp_ver:
            print("SELFTEST FAIL:")
            print(f"  {dep}: missing pinned version in RUNTIME_PIN.json")
            raise SystemExit(1)
        try:
            mod = importlib.import_module(dep)
        except Exception as exc:  # noqa: BLE001
            print("SELFTEST FAIL:")
            print(
                f"  environment contamination detected: {dep} import failed ({type(exc).__name__}: {exc}) "
                f"expected {exp_ver} from bundle"
            )
            raise SystemExit(1) from exc
        act_ver = str(getattr(mod, "__version__", "") or "")
        mod_file = getattr(mod, "__file__", None) or ""
        origin = Path(mod_file).resolve() if mod_file else None
        if act_ver != exp_ver:
            print("SELFTEST FAIL:")
            print(
                f"  environment contamination detected: {dep} {act_ver} "
                f"expected {exp_ver} from bundle"
            )
            raise SystemExit(1)
        if origin is None or not _path_under_bundle(origin, install, roots):
            origin_s = str(origin) if origin else "(unknown)"
            print("SELFTEST FAIL:")
            print(
                f"  environment contamination detected: {dep} {act_ver} from {origin_s} "
                f"expected {exp_ver} from bundle"
            )
            raise SystemExit(1)


def _verify_required_files(install: Path, pin: dict) -> None:
    required: list[str] = list(pin.get("required_files") or [])
    if not required:
        raise SystemExit("SELFTEST FAIL: RUNTIME_PIN.json missing required_files (rebuild bundle)")
    missing = [rel for rel in required if not (install / rel).is_file()]
    if missing:
        print("SELFTEST FAIL:")
        for rel in missing:
            print(f"  required runtime file missing: {rel}")
        raise SystemExit(1)


def _verify_runtime_loaders(install: Path) -> None:
    import citations
    import params_registry as pr

    reg = pr.load_registry()
    if not reg:
        print("SELFTEST FAIL:")
        print("  params_registry.load_registry() returned empty registry")
        raise SystemExit(1)
    bib = citations.load_citations_bib()
    logo = install / "img" / "VYVAR_logo.png"
    if not logo.is_file():
        print("SELFTEST FAIL:")
        print(f"  missing PDF logo: {logo}")
        raise SystemExit(1)
    print(f"runtime_data OK registry_keys={len(reg)} citations={len(bib)}")


def _verify_data_dir_skeleton(install: Path) -> None:
    import os
    import tempfile

    from vyvar_runtime import ensure_release_data_dir

    expected = (
        "Archive",
        "Archive/Drafts",
        "CalibrationLibrary",
        "GAIA_DR3",
        "VSX",
        "exoplanets",
        "logs",
    )
    prev = os.environ.get("VYVAR_DATA_DIR")
    with tempfile.TemporaryDirectory(prefix="vyvar_selftest_data_") as tmp:
        os.environ["VYVAR_DATA_DIR"] = tmp
        try:
            ensure_release_data_dir(install)
            for rel in expected:
                p = Path(tmp) / rel
                status = "OK" if p.is_dir() else "MISSING"
                print(f"data_skeleton {rel}: {status}")
                if not p.is_dir():
                    print("SELFTEST FAIL:")
                    print(f"  data dir bootstrap missing: {rel}")
                    raise SystemExit(1)
        finally:
            if prev is None:
                os.environ.pop("VYVAR_DATA_DIR", None)
            else:
                os.environ["VYVAR_DATA_DIR"] = prev


def main() -> int:
    pin = _load_runtime_pin(INSTALL)
    _verify_required_files(INSTALL, pin)
    _verify_pinned_deps(INSTALL, pin)
    _verify_runtime_loaders(INSTALL)
    _verify_data_dir_skeleton(INSTALL)
    data_root = resolve_data_root(INSTALL)
    print(f"VYVAR selftest platform={platform.platform()}")
    print(f"python={sys.version.split()[0]} executable={sys.executable}")
    print(f"install_dir={INSTALL}")
    print(f"data_dir={data_root}")
    print(f"isolated={getattr(sys.flags, 'isolate', None)}")
    for dep in PINNED_DEPS:
        mod = importlib.import_module(dep)
        ver = getattr(mod, "__version__", "?")
        origin = getattr(mod, "__file__", "?")
        print(f"dep {dep}={ver} origin={origin}")
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
