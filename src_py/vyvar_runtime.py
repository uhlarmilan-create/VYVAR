# -*- coding: ascii -*-
"""Release bundle first-launch bootstrap and data-dir helpers (RELEASE-2 / B1)."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from config import config_json_path, is_git_dev_checkout, materialize_fresh_config_json, resolve_data_root

BootstrapStatus = Literal["created", "preexisting"] | str

_NEXT_STEPS = """VYVAR - first launch complete
================================

Your data directory was created. Next steps:

1. Build local catalogs (required for science-grade photometry):
   Linux:   ./vyvar.sh --tool build_gaia -- --mag-limit 16.5
            ./vyvar.sh --tool build_blind_index --
            ./vyvar.sh --tool build_vsx --
            ./vyvar.sh --tool build_exoplanets --
   Windows: VYVAR.bat --tool build_gaia -- --mag-limit 16.5
            (same pattern for build_blind_index, build_vsx, build_exoplanets)
   See scripts/catalogs/README.md in the install folder.
   Gaia full-sky G<=16.5 download is ~9-10 GB and may take hours/days.

2. Open VYVAR and use Database Explorer to add Location, Telescope, and
   Equipment (OSC cameras: set BAYERMASK on the equipment record).

3. Import your first night and run the pipeline.

Data directory: {data_dir}
Override with env var VYVAR_DATA_DIR before launching VYVAR.
"""

_DATA_SKELETON_DIRS = (
    "Archive",
    "Archive/Drafts",
    "CalibrationLibrary",
    "GAIA_DR3",
    "VSX",
    "exoplanets",
    "logs",
)



def _ensure_data_skeleton(data_root: Path) -> None:
    for sub in ("Archive", "CalibrationLibrary", "GAIA_DR3", "VSX", "exoplanets", "logs"):
        (data_root / sub).mkdir(parents=True, exist_ok=True)
    (data_root / "Archive" / "Drafts").mkdir(parents=True, exist_ok=True)


def _record_path(
    report: dict[str, BootstrapStatus],
    key: str,
    path: Path,
    *,
    is_dir: bool,
) -> None:
    try:
        existed = path.is_dir() if is_dir else path.is_file()
        if is_dir:
            path.mkdir(parents=True, exist_ok=True)
        report[key] = "preexisting" if existed else "created"
    except OSError as exc:
        report[key] = f"FAILED:{type(exc).__name__}:{exc}"


def bootstrap_release_data_dir(install_root: Path) -> tuple[Path, dict[str, BootstrapStatus]]:
    """Create the bundled user data tree and return ``{item -> created|preexisting|FAILED:...}``."""
    install_root = install_root.resolve()
    data_root = resolve_data_root(install_root)
    report: dict[str, BootstrapStatus] = {}

    if is_git_dev_checkout(install_root):
        report["_bootstrap"] = "skipped:git_dev_checkout"
        return data_root, report

    _record_path(report, "data_root", data_root, is_dir=True)
    for rel in _DATA_SKELETON_DIRS:
        _record_path(report, rel, data_root / rel, is_dir=True)

    cfg_path = config_json_path(data_root)
    if cfg_path.is_file():
        report["config.json"] = "preexisting"
    else:
        try:
            materialize_fresh_config_json(install_root, data_root)
            report["config.json"] = "created"
        except Exception as exc:  # noqa: BLE001
            report["config.json"] = f"FAILED:{type(exc).__name__}:{exc}"

    db_path = data_root / "vyvar.sqlite3"
    if db_path.is_file():
        report["vyvar.sqlite3"] = "preexisting"
    else:
        try:
            from database import VyvarDatabase

            db = VyvarDatabase(str(db_path))
            db.close()
            report["vyvar.sqlite3"] = "created"
        except Exception as exc:  # noqa: BLE001
            report["vyvar.sqlite3"] = f"FAILED:{type(exc).__name__}:{exc}"

    note = data_root / "NEXT_STEPS.txt"
    if note.is_file():
        report["NEXT_STEPS.txt"] = "preexisting"
    else:
        try:
            note.write_text(_NEXT_STEPS.format(data_dir=data_root), encoding="ascii")
            report["NEXT_STEPS.txt"] = "created"
        except OSError as exc:
            report["NEXT_STEPS.txt"] = f"FAILED:{type(exc).__name__}:{exc}"

    return data_root, report


def bootstrap_failures(report: dict[str, BootstrapStatus]) -> dict[str, BootstrapStatus]:
    return {key: status for key, status in report.items() if str(status).startswith("FAILED:")}


def ensure_release_data_dir(install_root: Path, *, template_path: Path | None = None) -> Path:
    """Bundled launch bootstrap; raises when any report item FAILED."""
    del template_path  # legacy kwarg; bootstrap uses canonical writer, not template copy
    data_root, report = bootstrap_release_data_dir(install_root)
    failures = bootstrap_failures(report)
    if failures:
        lines = "\n".join(f"  {key}: {status}" for key, status in sorted(failures.items()))
        raise RuntimeError(f"VYVAR data-dir bootstrap failed:\n{lines}")
    return data_root
