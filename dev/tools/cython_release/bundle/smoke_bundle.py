# -*- coding: ascii -*-
"""Dev-side bundle smoke: unpack, --selftest, assert layout (A3)."""
from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

BUNDLE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BUNDLE_DIR))

from bundle_layout import DIST_DIR, assert_no_compiled_py_sources, compiled_module_stems  # noqa: E402


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


def smoke(artifact: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="vyvar_bundle_smoke ") as tmp:
        root = Path(tmp)
        _unpack(artifact, root)
        bundle_root = _find_bundle_root(root)
        assert_no_compiled_py_sources(bundle_root / "src_py", compiled_module_stems())
        if sys.platform == "win32":
            launcher = bundle_root / "VYVAR.bat"
            cmd = ["cmd", "/c", str(launcher), "--selftest"]
        else:
            launcher = bundle_root / "vyvar.sh"
            launcher.chmod(0o755)
            cmd = [str(launcher), "--selftest"]
        proc = subprocess.run(cmd, cwd=str(bundle_root), capture_output=True, text=True)
        log = DIST_DIR / "smoke_last.log"
        log.write_text((proc.stdout or "") + (proc.stderr or ""), encoding="ascii")
        if proc.returncode != 0:
            raise SystemExit(f"bundle smoke FAIL exit={proc.returncode} log={log}")
        print(f"bundle smoke PASS log={log}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    smoke(args.artifact.resolve())


if __name__ == "__main__":
    main()
