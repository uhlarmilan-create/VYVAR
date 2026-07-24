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


def _verify_data_dir_bootstrap(install: Path) -> None:
    from vyvar_runtime import bootstrap_failures, bootstrap_release_data_dir

    data_root, report = bootstrap_release_data_dir(install)
    failures = bootstrap_failures(report)
    for key in sorted(report):
        print(f"bootstrap {key}: {report[key]}")
    if failures:
        print("SELFTEST FAIL:")
        print(f"  data-dir bootstrap failed for {data_root}")
        for key, status in sorted(failures.items()):
            print(f"  {key}: {status}")
        raise SystemExit(1)

    expected_dirs = (
        "Archive",
        "Archive/Drafts",
        "CalibrationLibrary",
        "GAIA_DR3",
        "VSX",
        "exoplanets",
        "logs",
    )
    for rel in expected_dirs:
        p = data_root / rel
        if not p.is_dir():
            print("SELFTEST FAIL:")
            print(f"  data dir missing on disk after bootstrap: {p}")
            raise SystemExit(1)
    cfg_path = data_root / "config.json"
    if not cfg_path.is_file():
        print("SELFTEST FAIL:")
        print(f"  config.json missing after bootstrap: {cfg_path}")
        raise SystemExit(1)
    db_path = data_root / "vyvar.sqlite3"
    if not db_path.is_file():
        print("SELFTEST FAIL:")
        print(f"  vyvar.sqlite3 missing after bootstrap: {db_path}")
        raise SystemExit(1)


def main() -> int:
    pin = _load_runtime_pin(INSTALL)
    _verify_required_files(INSTALL, pin)
    _verify_pinned_deps(INSTALL, pin)
    _verify_runtime_loaders(INSTALL)
    _verify_data_dir_bootstrap(INSTALL)
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
