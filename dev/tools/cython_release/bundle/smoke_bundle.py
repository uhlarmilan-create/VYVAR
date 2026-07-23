# -*- coding: ascii -*-
"""Dev-side bundle smoke: unpack, --selftest, assert layout (A3)."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

BUNDLE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BUNDLE_DIR))

from bundle_layout import DIST_DIR, assert_no_compiled_py_sources, compiled_module_stems  # noqa: E402

SMOKE_DIST = Path(os.environ.get("VYVAR_BUNDLE_SMOKE_DIST", str(DIST_DIR)))


def _unpack(artifact: Path, dest: Path) -> Path:
    if artifact.suffix == ".zip":
        with zipfile.ZipFile(artifact) as zf:
            zf.extractall(dest)
        children = list(dest.iterdir())
        return children[0] if len(children) == 1 else dest
    with tarfile.open(artifact, "r:gz") as tf:
        tf.extractall(dest)
    children = list(dest.iterdir())
    return children[0] if len(children) == 1 else dest


def _find_bundle_root(dest: Path) -> Path:
    for name in ("VYVAR.bat", "vyvar.sh"):
        hits = list(dest.rglob(name))
        if hits:
            return hits[0].parent
    raise SystemExit(f"bundle root not found under {dest}")


def _write_log(text: str) -> Path:
    log = SMOKE_DIST / "smoke_last.log"
    SMOKE_DIST.mkdir(parents=True, exist_ok=True)
    log.write_text(text, encoding="ascii", errors="replace")
    return log


def _run_selftest(bundle_root: Path, *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    if sys.platform == "win32":
        launcher = bundle_root / "VYVAR.bat"
        cmd = ["cmd", "/c", str(launcher), "--selftest"]
    else:
        launcher = bundle_root / "vyvar.sh"
        launcher.chmod(0o755)
        cmd = [str(launcher), "--selftest"]
    return subprocess.run(cmd, cwd=str(bundle_root), capture_output=True, text=True, env=run_env)


def _bundle_python_exe(bundle_root: Path) -> Path:
    if sys.platform == "win32":
        return bundle_root / "python" / "python.exe"
    for candidate in (
        bundle_root / "python" / "python" / "bin" / "python3",
        bundle_root / "python" / "bin" / "python3",
    ):
        if candidate.is_file():
            return candidate
    hits = list(bundle_root.glob("python/**/python3"))
    if not hits:
        raise SystemExit("bundle python3 not found for contamination test")
    return hits[0]


def _contamination_regression(bundle_root: Path) -> None:
    poison_root = bundle_root.parent / "poison_site"
    poison_numpy = poison_root / "numpy"
    if poison_numpy.is_dir():
        import shutil

        shutil.rmtree(poison_root)
    poison_numpy.mkdir(parents=True)
    (poison_numpy / "__init__.py").write_text('__version__ = "0.0.0"\n', encoding="ascii")
    poison_env = {"PYTHONPATH": str(poison_root)}

    isolated = _run_selftest(bundle_root, env=poison_env)
    if isolated.returncode != 0:
        log = _write_log((isolated.stdout or "") + (isolated.stderr or ""))
        raise SystemExit(
            f"contamination regression FAIL: isolated launcher selftest failed log={log}"
        )

    py = _bundle_python_exe(bundle_root)
    selftest = bundle_root / "vyvar_selftest.py"
    inject = (
        "import sys, runpy\n"
        f"sys.path.insert(0, {str(poison_root)!r})\n"
        f"raise SystemExit(runpy.run_path({str(selftest)!r}, run_name='__main__') or 0)\n"
    )
    direct = subprocess.run(
        [str(py), "-c", inject],
        cwd=str(bundle_root),
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    combined = (direct.stdout or "") + (direct.stderr or "")
    if direct.returncode == 0:
        log = _write_log(combined)
        raise SystemExit(
            f"contamination regression FAIL: non-isolated selftest should fail log={log}"
        )
    if "environment contamination detected" not in combined:
        log = _write_log(combined)
        raise SystemExit(
            f"contamination regression FAIL: missing contamination message log={log}"
        )


def smoke(artifact: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="vyvar_bundle_smoke ") as tmp:
        root = Path(tmp)
        _unpack(artifact, root)
        bundle_root = _find_bundle_root(root)
        assert_no_compiled_py_sources(bundle_root / "src_py", compiled_module_stems())
        proc = _run_selftest(bundle_root)
        log = _write_log((proc.stdout or "") + (proc.stderr or ""))
        if proc.returncode != 0:
            raise SystemExit(f"bundle smoke FAIL exit={proc.returncode} log={log}")
        _contamination_regression(bundle_root)
        print(f"bundle smoke PASS log={log}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    smoke(args.artifact.resolve())


if __name__ == "__main__":
    main()
