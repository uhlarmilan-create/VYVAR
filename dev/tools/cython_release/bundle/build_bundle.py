# -*- coding: ascii -*-
"""Assemble a self-contained VYVAR release bundle (RELEASE-2 / A1)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

BUNDLE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BUNDLE_DIR))

from bundle_layout import (  # noqa: E402
    CATALOG_SCRIPT_SOURCES,
    DIST_DIR,
    REPO_ROOT,
    REQUIRED_RUNTIME_FILES,
    RUNTIME_FILE_SOURCES,
    SRC_PY,
    assert_no_compiled_py_sources,
    bundle_name,
    compiled_module_stems,
    ui_py_names,
)
from runtime_fetch import extract_runtime, fetch_runtime_archive  # noqa: E402
from runtime_pins import BUNDLE_REQUIREMENTS_EXCLUDE, RUNTIME_PINS  # noqa: E402

TEMPLATES = BUNDLE_DIR / "templates"
LOG_DIR = REPO_ROOT / "tmp" / "cython_release" / "bundle"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _bundle_requirements_lines() -> list[str]:
    req = REPO_ROOT / "requirements.txt"
    lines: list[str] = []
    for raw in req.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        pkg = line.split("[", 1)[0].split("=", 1)[0].split("<", 1)[0].split(">", 1)[0].strip()
        if pkg.lower() in BUNDLE_REQUIREMENTS_EXCLUDE:
            continue
        lines.append(line)
    return lines


def _prepare_windows_embed(python_dir: Path) -> None:
    pth_files = list(python_dir.glob("python*._pth"))
    if not pth_files:
        raise SystemExit(f"missing python*._pth in {python_dir}")
    site = python_dir / "Lib" / "site-packages"
    site.mkdir(parents=True, exist_ok=True)
    text = pth_files[0].read_text(encoding="utf-8")
    text = text.replace("#import site", "import site")
    if "import site" not in text:
        text += "\nimport site\n"
    rel = "Lib\\site-packages"
    if rel not in text.replace("/", "\\"):
        text += f"\n{rel}\n"
    pth_files[0].write_text(text, encoding="utf-8")


def _install_site_packages(python_exe: Path, target: Path, req_lines: list[str]) -> None:
    target.mkdir(parents=True, exist_ok=True)
    tmp_req = LOG_DIR / "requirements-bundle.txt"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    tmp_req.write_text("\n".join(req_lines) + "\n", encoding="ascii")
    if sys.platform == "win32" and python_exe.name.lower() == "python.exe":
        # Bootstrap pip into embeddable runtime when needed
        get_pip = LOG_DIR / "get-pip.py"
        if not get_pip.is_file():
            subprocess.run(
                [sys.executable, "-m", "pip", "download", "pip", "-d", str(LOG_DIR / "pip_wheels")],
                check=False,
            )
            import urllib.request

            urllib.request.urlretrieve(
                "https://bootstrap.pypa.io/get-pip.py",
                get_pip,
            )
        subprocess.run([str(python_exe), str(get_pip), "--no-warn-script-location"], check=True)
    subprocess.run(
        [
            str(python_exe),
            "-m",
            "pip",
            "install",
            "--no-warn-script-location",
            "-r",
            str(tmp_req),
            "--target",
            str(target),
        ],
        check=True,
    )


def _pinned_dep_versions(python_exe: Path) -> dict[str, str]:
    code = (
        "import importlib.metadata as m\n"
        "deps = ['numpy','astropy','photutils','streamlit','pandas','scipy']\n"
        "for d in deps:\n"
        "    print(d, m.version(d))\n"
    )
    proc = subprocess.run(
        [str(python_exe), "-I", "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    out: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2:
            out[parts[0]] = parts[1]
    return out


def _write_runtime_pin(staging: Path, platform_key: str, pin: RuntimePin, python_exe: Path) -> None:
    dep_versions = _pinned_dep_versions(python_exe)
    (staging / "RUNTIME_PIN.json").write_text(
        json.dumps(
            {
                "platform": platform_key,
                "python_version": pin.version,
                "url": pin.url,
                "sha256": pin.sha256 or "(see cache sidecar)",
                "dep_versions": dep_versions,
                "required_files": list(REQUIRED_RUNTIME_FILES),
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )


def _ascii_fold(text: str) -> str:
    return text.encode("ascii", "replace").decode("ascii")


def _third_party_notices(site_packages: Path) -> str:
    lines = [
        "VYVAR THIRD_PARTY_NOTICES",
        "Bundled Python packages (see PyPI for full license texts).",
        "",
    ]
    try:
        from importlib.metadata import distributions

        for dist in sorted(distributions(path=[str(site_packages)]), key=lambda d: (d.metadata.get("Name") or "").lower()):
            name = dist.metadata.get("Name") or dist.name
            ver = dist.version
            lic = _ascii_fold(str(dist.metadata.get("License") or dist.metadata.get("License-Expression") or "SEE PYPI"))
            lines.append(f"{name} {ver} - {lic}")
    except Exception as exc:  # noqa: BLE001
        lines.append(f"(metadata scan failed: {exc})")
    lines.append("")
    return "\n".join(lines)


def _copy_science_and_ui(staging: Path, platform_key: str) -> None:
    compiled = compiled_module_stems()
    dest = staging / "src_py"
    dest.mkdir(parents=True)
    ext = ".pyd" if platform_key == "win64" else ".so"
    for name in compiled:
        matches = list(SRC_PY.glob(f"{name}*{ext}"))
        if not matches:
            raise SystemExit(f"missing compiled artifact for {name}{ext} in src_py (run RELEASE-1 build first)")
        shutil.copy2(matches[0], dest / matches[0].name)
    for fname in ui_py_names():
        shutil.copy2(SRC_PY / fname, dest / fname)
    assert_no_compiled_py_sources(dest, compiled)


def _write_root_shim(staging: Path) -> None:
    shutil.copy2(REPO_ROOT / "app.py", staging / "app.py")


def _copy_runtime_data_files(staging: Path) -> None:
    """Ship install-root data files that src_py reads at runtime."""
    for rel, src in RUNTIME_FILE_SOURCES.items():
        if not src.is_file():
            raise SystemExit(f"REFUSE: required runtime file missing in repo: {rel} ({src})")
        dest = staging / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)


def _stage_catalog_scripts(staging: Path) -> None:
    for rel, src in CATALOG_SCRIPT_SOURCES.items():
        if not src.is_file():
            raise SystemExit(f"REFUSE: catalog script missing in repo: {rel} ({src})")
        dest = staging / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)


def _assert_runtime_files(staging: Path) -> None:
    missing = [rel for rel in REQUIRED_RUNTIME_FILES if not (staging / rel).is_file()]
    if missing:
        raise SystemExit(
            "Bundle assertion failed: required runtime files missing from staging:\n  "
            + "\n  ".join(missing)
        )


def _stage_runtime(staging: Path, platform_key: str) -> Path:
    pin = RUNTIME_PINS[platform_key]
    archive = fetch_runtime_archive(platform_key)
    py_home = staging / "python"
    py_home = extract_runtime(archive, pin, py_home)
    if platform_key == "win64":
        _prepare_windows_embed(py_home)
        python_exe = py_home / "python.exe"
        site = py_home / "Lib" / "site-packages"
    else:
        python_exe = py_home / "bin" / "python3"
        if not python_exe.is_file():
            python_exe = next(py_home.rglob("python3"))
        ver = f"{pin.version.split('.')[0]}.{pin.version.split('.')[1]}"
        site = py_home / "lib" / f"python{ver}" / "site-packages"
        site.mkdir(parents=True, exist_ok=True)
    _install_site_packages(python_exe, site, _bundle_requirements_lines())
    return site


def _write_config_template(staging: Path) -> None:
    """Generate reference ``config.template.json`` from the canonical grouped writer at build time.

    Bootstrap no longer copies this file; it is documentation/reference only. When the build
    host lacks full science deps (typical WSL system Python), fall back to the dev checkout
    ``config.json`` which is maintained by the same writer.
    """
    dest = staging / "config.template.json"
    src_py = str(SRC_PY)
    if src_py not in sys.path:
        sys.path.insert(0, src_py)
    try:
        from config import AppConfig, render_config_jsonc  # noqa: WPS433

        cfg = AppConfig(project_root=REPO_ROOT)
        dest.write_text(render_config_jsonc(cfg.to_json()), encoding="utf-8")
        return
    except Exception:  # noqa: BLE001 -- WSL/minimal hosts may lack pandas etc.
        pass
    fallback = REPO_ROOT / "config.json"
    if not fallback.is_file():
        raise SystemExit(
            "config.template.json: cannot materialize from AppConfig and no config.json fallback"
        )
    dest.write_text(fallback.read_text(encoding="utf-8"), encoding="utf-8")


def build_bundle(*, tag: str, platform_key: str, skip_runtime: bool = False) -> Path:
    if platform_key not in RUNTIME_PINS:
        raise SystemExit(f"unknown platform {platform_key}")
    pin = RUNTIME_PINS[platform_key]
    name = bundle_name(tag, platform_key)
    staging = DIST_DIR / "staging" / name
    if staging.is_dir():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    _copy_science_and_ui(staging, platform_key)
    _write_root_shim(staging)
    _copy_runtime_data_files(staging)
    _stage_catalog_scripts(staging)
    for tpl in ("VYVAR.bat", "vyvar.sh", "vyvar_selftest.py"):
        shutil.copy2(TEMPLATES / tpl, staging / tpl)
    _write_config_template(staging)
    shutil.copy2(REPO_ROOT / "LICENSE", staging / "LICENSE")
    _assert_runtime_files(staging)
    if not skip_runtime:
        site = _stage_runtime(staging, platform_key)
        (staging / "THIRD_PARTY_NOTICES.txt").write_text(
            _third_party_notices(site), encoding="ascii"
        )
        if platform_key == "win64":
            python_exe = staging / "python" / "python.exe"
        else:
            py_home = staging / "python"
            subs = [p for p in py_home.iterdir() if p.is_dir()]
            inner = subs[0] if len(subs) == 1 else py_home
            python_exe = inner / "bin" / "python3"
            if not python_exe.is_file():
                python_exe = next(inner.rglob("python3"))
        _write_runtime_pin(staging, platform_key, pin, python_exe)
    else:
        (staging / "THIRD_PARTY_NOTICES.txt").write_text(
            "THIRD_PARTY_NOTICES skipped (--skip-runtime)\n", encoding="ascii"
        )
        (staging / "RUNTIME_PIN.json").write_text(
            json.dumps({"platform": platform_key, "dep_versions": {}}, indent=2) + "\n",
            encoding="ascii",
        )
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    if platform_key == "win64":
        out = DIST_DIR / f"{name}.zip"
        if out.is_file():
            out.unlink()
        with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in staging.rglob("*"):
                if path.is_file():
                    arcname = path.relative_to(staging.parent)
                    zf.write(path, arcname.as_posix())
    else:
        out = DIST_DIR / f"{name}.tar.gz"
        if out.is_file():
            out.unlink()
        with tarfile.open(out, "w:gz") as tf:
            tf.add(staging, arcname=name)
    sums = DIST_DIR / "SHA256SUMS"
    digest = _sha256_file(out)
    line = f"{digest}  {out.name}\n"
    if sums.is_file():
        old = [ln for ln in sums.read_text(encoding="ascii").splitlines() if not ln.endswith(out.name)]
        sums.write_text("\n".join(old + [line.strip()]) + "\n", encoding="ascii")
    else:
        sums.write_text(line, encoding="ascii")
    log = LOG_DIR / f"build_{platform_key}.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log.write_text(f"artifact={out}\nsha256={digest}\n", encoding="ascii")
    print(f"bundle OK: {out} sha256={digest}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build VYVAR release bundle")
    parser.add_argument("--tag", default="preview-20260723")
    parser.add_argument("--platform", choices=sorted(RUNTIME_PINS), required=True)
    parser.add_argument("--skip-runtime", action="store_true", help="Layout-only (no pip/runtime fetch)")
    args = parser.parse_args()
    build_bundle(tag=args.tag, platform_key=args.platform, skip_runtime=args.skip_runtime)


if __name__ == "__main__":
    main()
