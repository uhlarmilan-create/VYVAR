# -*- coding: ascii -*-
"""Release bundle first-launch bootstrap and data-dir helpers (RELEASE-2 / B1)."""
from __future__ import annotations

from pathlib import Path

from config import config_json_path, is_git_dev_checkout, materialize_fresh_config_json, resolve_data_root

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


def install_root_from_src_py(src_py_file: Path) -> Path:
    return src_py_file.resolve().parent.parent


def _ensure_data_skeleton(data_root: Path) -> None:
    for sub in ("Archive", "CalibrationLibrary", "GAIA_DR3", "VSX", "exoplanets", "logs"):
        (data_root / sub).mkdir(parents=True, exist_ok=True)
    (data_root / "Archive" / "Drafts").mkdir(parents=True, exist_ok=True)


def ensure_release_data_dir(install_root: Path, *, template_path: Path | None = None) -> Path:
    """Create the user data tree on first bundled launch; no-op for git dev checkouts."""
    del template_path  # legacy kwarg; bootstrap uses canonical writer, not template copy
    data_root = resolve_data_root(install_root)
    if is_git_dev_checkout(install_root):
        return data_root
    _ensure_data_skeleton(data_root)
    cfg_path = config_json_path(data_root)
    if cfg_path.is_file():
        return data_root
    materialize_fresh_config_json(install_root, data_root)
    db_path = data_root / "vyvar.sqlite3"
    if not db_path.is_file():
        from database import VyvarDatabase

        db = VyvarDatabase(str(db_path))
        db.close()
    note = data_root / "NEXT_STEPS.txt"
    note.write_text(_NEXT_STEPS.format(data_dir=data_root), encoding="ascii")
    return data_root
