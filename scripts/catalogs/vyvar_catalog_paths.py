# -*- coding: ascii -*-
"""Default catalog output paths under the VYVAR data directory (release + dev)."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def install_root_from_script_dir(script_dir: Path) -> Path:
    """Map a catalog script directory to the VYVAR install root."""
    here = script_dir.resolve()
    if here.name == "catalogs":
        return here.parent.parent
    if here.name in ("GAIA_DR3", "VSX", "exoplanets"):
        return here.parent
    return here.parent


def data_root_from_script_dir(script_dir: Path) -> Path:
    install = install_root_from_script_dir(script_dir)
    src = install / "src_py"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from config import resolve_data_root

    return resolve_data_root(install)


def default_catalog_file(script_dir: Path, *rel_parts: str) -> Path:
    return data_root_from_script_dir(script_dir).joinpath(*rel_parts)


def load_helper_module(script_dir: Path):
    """Load this module when copied next to a catalog script (import fallback)."""
    here = script_dir.resolve()
    candidates = [here, here.parent / "scripts" / "catalogs"]
    install = install_root_from_script_dir(here)
    candidates.append(install / "scripts" / "catalogs")
    seen: set[Path] = set()
    for base in candidates:
        helper = (base / "vyvar_catalog_paths.py").resolve()
        if helper in seen or not helper.is_file():
            continue
        seen.add(helper)
        import importlib.util

        spec = importlib.util.spec_from_file_location("vyvar_catalog_paths", helper)
        if spec is None or spec.loader is None:
            continue
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    return None


def release_bundle_marked(install: Path) -> bool:
    if os.environ.get("VYVAR_RELEASE_BUNDLE", "").strip().lower() in ("1", "true", "yes", "on"):
        return True
    root = install.resolve()
    return (root / "RUNTIME_PIN.json").is_file()
