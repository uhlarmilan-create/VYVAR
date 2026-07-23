# -*- coding: ascii -*-
"""CYTHON-RELEASE-1 build driver (plain compile, pinned flags S3)."""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from Cython.Build import cythonize
from Cython.Compiler import Options
from setuptools import Extension, setup

# S3: strip docstrings from compiled binaries (must be set before cythonize).
Options.docstrings = False

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from module_list import REPO_ROOT, SRC_PY, module_list

BUILD_DIR = REPO_ROOT / "build" / "_cython_out"
LOG_DIR = REPO_ROOT / "tmp" / "cython_release"
DEFAULT_LOG = LOG_DIR / "build.log"

CYTHON_SPIKE_VERSION = "3.2.8"

# S3: pinned exactly as spike GO verdict.
REQUIRED_COMPILER_DIRECTIVES = {
    "language_level": "3",
    "embedsignature": False,
    "annotation_typing": False,
}
COMPILER_DIRECTIVES = dict(REQUIRED_COMPILER_DIRECTIVES)


def _assert_pinned_flags() -> None:
    if Options.docstrings is not False:
        raise SystemExit(
            "REFUSE: Options.docstrings must be False (release pin S3); "
            f"got {Options.docstrings!r}"
        )
    for key, val in REQUIRED_COMPILER_DIRECTIVES.items():
        if COMPILER_DIRECTIVES.get(key) != val:
            raise SystemExit(
                f"REFUSE: compiler directive {key!r} drifted: "
                f"expected {val!r}, got {COMPILER_DIRECTIVES.get(key)!r}"
            )


def _extensions(modules: list[str]) -> list[Extension]:
    exts: list[Extension] = []
    for name in modules:
        src = SRC_PY / f"{name}.py"
        if not src.is_file():
            print(f"WARNING: missing source {src}", file=sys.stderr)
            continue
        exts.append(Extension(name, [str(src)]))
    return exts


def _relocate_pyd_artifacts(modules: list[str], log: list[str]) -> None:
    """Move cp312-*.pyd from repo root into src_py/ (setuptools inplace quirk on Windows)."""
    for name in modules:
        for pyd in REPO_ROOT.glob(f"{name}.cp*.pyd"):
            dest = SRC_PY / pyd.name
            pyd.replace(dest)
            log.append(f"relocated {pyd.name} -> src_py/")
        for pyd in REPO_ROOT.glob(f"{name}.*.pyd"):
            if pyd.parent == REPO_ROOT:
                dest = SRC_PY / pyd.name
                pyd.replace(dest)
                log.append(f"relocated {pyd.name} -> src_py/")


def run_build(*, modules: list[str] | None = None, log_path: Path | None = None) -> Path:
    _assert_pinned_flags()
    Options.docstrings = False
    mods = sorted(modules if modules is not None else module_list())
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    lp = log_path or DEFAULT_LOG
    lines: list[str] = [
        f"CYTHON-RELEASE build {datetime.now(timezone.utc).isoformat()}",
        f"modules={len(mods)}",
        f"directives={COMPILER_DIRECTIVES}",
        f"Options.docstrings={Options.docstrings}",
        "",
    ]
    exts = _extensions(mods)
    if not exts:
        raise SystemExit("No extensions to build")
    argv = ["build_ext", "--inplace"]
    setup(
        name="vyvar_cython_release",
        ext_modules=cythonize(
            exts,
            compiler_directives=COMPILER_DIRECTIVES,
            build_dir=str(BUILD_DIR),
        ),
        script_args=argv,
    )
    _relocate_pyd_artifacts(mods, lines)
    lp.write_text("\n".join(lines) + "\n", encoding="ascii")
    return lp


def run_clean() -> None:
    import shutil

    removed: list[str] = []
    for pattern in ("*.pyd", "*.so", "*.c"):
        for p in SRC_PY.glob(pattern):
            p.unlink(missing_ok=True)
            removed.append(str(p.relative_to(REPO_ROOT)))
    if BUILD_DIR.is_dir():
        shutil.rmtree(BUILD_DIR, ignore_errors=True)
        removed.append(str(BUILD_DIR.relative_to(REPO_ROOT)))
    log = LOG_DIR / "clean.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log.write_text("\n".join(removed) + "\n", encoding="ascii")
    print(f"cleaned {len(removed)} paths; log {log}")


def main() -> None:
    parser = argparse.ArgumentParser(description="VYVAR Cython release build")
    parser.add_argument("command", choices=("build", "clean"), nargs="?", default="build")
    parser.add_argument(
        "--modules",
        default="",
        help="Comma-separated subset; default full MODULE_LIST",
    )
    parser.add_argument("--log", type=Path, default=None, help="Build log path")
    args = parser.parse_args()
    if args.command == "clean":
        run_clean()
        return
    subset = [m.strip() for m in args.modules.split(",") if m.strip()] or None
    lp = run_build(modules=subset, log_path=args.log)
    print(f"build OK; log {lp}")


if __name__ == "__main__":
    main()
