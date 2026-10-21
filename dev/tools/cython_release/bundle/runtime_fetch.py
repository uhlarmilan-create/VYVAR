# -*- coding: ascii -*-
"""Download and cache embedded Python runtimes with SHA256 verification."""
from __future__ import annotations

import hashlib
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

from runtime_pins import RUNTIME_PINS, RuntimePin

CACHE_DIR = Path(__file__).resolve().parent / "cache"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=600) as resp, tmp.open("wb") as out:
        shutil.copyfileobj(resp, out)
    tmp.replace(dest)


def fetch_runtime_archive(platform_key: str) -> Path:
    pin = RUNTIME_PINS[platform_key]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    name = Path(pin.url).name
    dest = CACHE_DIR / name
    sidecar = dest.with_suffix(dest.suffix + ".sha256")
    if not dest.is_file():
        _download(pin.url, dest)
    digest = _sha256_file(dest)
    expected = pin.sha256 or (sidecar.read_text(encoding="ascii").strip() if sidecar.is_file() else "")
    if expected and digest != expected:
        raise SystemExit(
            f"REFUSE: runtime SHA256 mismatch for {platform_key}: expected {expected}, got {digest}"
        )
    if not expected:
        sidecar.write_text(digest + "\n", encoding="ascii")
    return dest


def extract_runtime(archive: Path, pin: RuntimePin, dest_dir: Path) -> Path:
    """Extract runtime; return path to python home directory."""
    if dest_dir.exists():
        shutil.rmtree(dest_dir, ignore_errors=True)
    dest_dir.mkdir(parents=True, exist_ok=True)
    if pin.archive_kind == "zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest_dir)
        return dest_dir
    # python-build-standalone terminfo symlinks break on WSL drvfs (/mnt/c).
    dest_str = str(dest_dir)
    use_tmp = dest_str.startswith("/mnt/")
    tmp_root: Path | None = None
    extract_dest = dest_dir
    if use_tmp:
        import tempfile

        tmp_root = Path(tempfile.mkdtemp(prefix="vyvar_runtime_"))
        extract_dest = tmp_root
    with tarfile.open(archive, "r:gz") as tf:
        try:
            tf.extractall(extract_dest, filter="data")
        except TypeError:
            tf.extractall(extract_dest)
    subs = [p for p in extract_dest.iterdir() if p.is_dir()]
    py_home = subs[0] if len(subs) == 1 else extract_dest
    if use_tmp and tmp_root is not None:
        if dest_dir.exists():
            shutil.rmtree(dest_dir, ignore_errors=True)
        shutil.copytree(py_home, dest_dir, symlinks=False)
        shutil.rmtree(tmp_root, ignore_errors=True)
        return dest_dir
    return py_home
