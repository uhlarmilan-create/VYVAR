"""Guard for the installer's config path writer (INSTALL-ARC).

Covers the pure ``apply_paths`` logic: chosen paths win, author ``C:\\ASTRO`` paths
are blanked when not chosen, non-author values are left untouched, and the
end-to-end write produces a config that validates and keeps no author paths.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import config

_SCRIPTS = Path(config.__file__).resolve().parents[1] / "dev" / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def _load_helper():
    return _load_script("apply_install_config")


def test_apply_paths_blanks_author_paths_when_not_chosen() -> None:
    mod = _load_helper()
    data = {
        "archive_root": "C:\\ASTRO\\python\\VYVAR\\Archive",
        "gaia_db_path": "C:\\ASTRO\\python\\VYVAR\\GAIA_DR3\\vyvar_gaia_dr3.db",
        "exoplanet_local_db_path": "exoplanets/vyvar_exoplanet_local.db",
    }
    changes = mod.apply_paths(data, {k: None for k, _ in mod._PATH_KEYS})
    changed = {c[0] for c in changes}
    assert "archive_root" in changed and data["archive_root"] == ""
    assert "gaia_db_path" in changed and data["gaia_db_path"] == ""
    # A non-author (relative) value is left untouched.
    assert "exoplanet_local_db_path" not in changed
    assert data["exoplanet_local_db_path"] == "exoplanets/vyvar_exoplanet_local.db"


def test_apply_paths_uses_chosen_value() -> None:
    mod = _load_helper()
    data = {"gaia_db_path": "C:\\ASTRO\\python\\VYVAR\\GAIA_DR3\\vyvar_gaia_dr3.db"}
    chosen = {k: None for k, _ in mod._PATH_KEYS}
    chosen["gaia_db"] = "/data/vyvar/GAIA_DR3/vyvar_gaia_dr3.db"
    mod.apply_paths(data, chosen)
    assert "ASTRO" not in data["gaia_db_path"].upper()
    assert data["gaia_db_path"].endswith("vyvar_gaia_dr3.db")


def test_end_to_end_write_validates_and_drops_author_paths(tmp_path: Path) -> None:
    mod = _load_helper()
    src = Path(config.__file__).resolve().parents[1] / "config.json"
    target = tmp_path / "config.json"
    target.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    rc = mod.main([
        "--config", str(target),
        "--archive-root", str(tmp_path / "Archive"),
        "--gaia-db", str(tmp_path / "gaia.db"),
    ])
    assert rc == 0

    text = target.read_text(encoding="utf-8")
    # No author absolute path may survive in any value line.
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith('"') and ":" in stripped:
            assert "C:\\ASTRO" not in stripped and "C:/ASTRO" not in stripped

    validate_config = _load_script("validate_config")
    problems, data = validate_config.validate_text(text)
    assert data is not None
    assert not [m for sev, m in problems if sev == "ERROR"]
