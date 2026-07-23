# -*- coding: ascii -*-
"""Release bundle first-launch bootstrap and data-dir helpers (RELEASE-2 / B1)."""
from __future__ import annotations

import shutil
from pathlib import Path

from config import config_json_path, is_git_dev_checkout, resolve_data_root

_NEXT_STEPS = """VYVAR - first launch complete
================================

Your data directory was created. Next steps:

1. Build local catalogs (required for science-grade photometry):
   - Gaia DR3 DB:     python GAIA_DR3/build_gaia_catalog.py
   - Blind indexes:   python GAIA_DR3/build_blind_index.py
   - VSX local DB:    python VSX/vsx_make.py
   - Exoplanet DB:    python exoplanets/exoplanet_make.py
   Run these from a dev checkout or copy the built files into:
     GAIA_DR3/  VSX/  exoplanets/  under your data directory.

2. Open VYVAR and use Database Explorer to add Location, Telescope, and
   Equipment (OSC cameras: set BAYERMASK on the equipment record).

3. Import your first night and run the pipeline.

Data directory: {data_dir}
Override with env var VYVAR_DATA_DIR before launching VYVAR.
"""


def install_root_from_src_py(src_py_file: Path) -> Path:
    return src_py_file.resolve().parent.parent


def ensure_release_data_dir(install_root: Path, *, template_path: Path | None = None) -> Path:
    """Create the user data tree on first bundled launch; no-op for git dev checkouts."""
    data_root = resolve_data_root(install_root)
    if is_git_dev_checkout(install_root):
        return data_root
    cfg_path = config_json_path(data_root)
    if cfg_path.is_file():
        return data_root
    for sub in ("Archive", "CalibrationLibrary", "GAIA_DR3", "VSX", "exoplanets", "logs"):
        (data_root / sub).mkdir(parents=True, exist_ok=True)
    tpl = template_path or (install_root / "config.template.json")
    if tpl.is_file():
        shutil.copy2(tpl, cfg_path)
    else:
        cfg_path.write_text("{}\n", encoding="utf-8")
    db_path = data_root / "vyvar.sqlite3"
    if not db_path.is_file():
        from database import VyvarDatabase

        VyvarDatabase(str(db_path))
    note = data_root / "NEXT_STEPS.txt"
    note.write_text(_NEXT_STEPS.format(data_dir=data_root), encoding="ascii")
    return data_root
