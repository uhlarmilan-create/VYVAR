"""Write the machine-local paths block into config.json during installation.

Used by ``install_vyvar.ps1`` / ``install_vyvar.sh`` (PATHS phase). It sets only the
file/catalog path keys the installer chose and NEVER keeps the author's absolute
``C:\\ASTRO\\...`` paths: any path key that is not explicitly provided is sanitised --
if it currently holds one of those shipped absolute paths it is blanked so the
project-root-relative default resolves on the new machine.

The write goes through the canonical, comment-preserving writer
(``config.save_config_json`` under ``config.ui_config_persist()``), so every other key
and its grouping/comments are regenerated exactly as a UI save would produce them.

Usage (all path args optional):
    python dev/scripts/apply_install_config.py \
        --archive-root PATH --calibration-root PATH --database-path PATH \
        --gaia-db PATH --vsx-db PATH --exoplanet-db PATH \
        --blind-fine PATH --blind-wide PATH [--config PATH] [--dry-run]

Exit code 0 on success, non-zero on failure.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401

_ROOT = _bootstrap.REPO_ROOT

import config  # noqa: E402

# CLI arg name -> config.json key. Order is the display order in the summary.
_PATH_KEYS: tuple[tuple[str, str], ...] = (
    ("archive_root", "archive_root"),
    ("calibration_root", "calibration_library_root"),
    ("database_path", "database_path"),
    ("gaia_db", "gaia_db_path"),
    ("vsx_db", "vsx_local_db_path"),
    ("exoplanet_db", "exoplanet_local_db_path"),
    ("blind_fine", "blind_index_fine_path"),
    ("blind_wide", "blind_index_wide_path"),
)


def _is_authors_absolute_path(value: object) -> bool:
    """True for the shipped author machine paths that must never survive an install."""
    if not isinstance(value, str):
        return False
    return "c:\\astro" in value.lower() or "c:/astro" in value.lower()


def apply_paths(data: dict, chosen: dict[str, str | None]) -> list[tuple[str, str, str]]:
    """Mutate ``data`` in place. Return a list of (key, old, new) changes."""
    changes: list[tuple[str, str, str]] = []
    for arg_name, cfg_key in _PATH_KEYS:
        provided = chosen.get(arg_name)
        old = str(data.get(cfg_key, ""))
        if provided:
            new = str(Path(provided))  # normalise separators for this OS
        elif _is_authors_absolute_path(data.get(cfg_key)):
            new = ""  # blank -> project-root default resolves on load
        else:
            continue  # leave untouched
        if new != old:
            data[cfg_key] = new
            changes.append((cfg_key, old, new))
    return changes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive-root")
    ap.add_argument("--calibration-root")
    ap.add_argument("--database-path")
    ap.add_argument("--gaia-db")
    ap.add_argument("--vsx-db")
    ap.add_argument("--exoplanet-db")
    ap.add_argument("--blind-fine")
    ap.add_argument("--blind-wide")
    ap.add_argument("--config", default=str(_ROOT / "config.json"))
    ap.add_argument("--dry-run", action="store_true", help="print changes, do not write")
    args = ap.parse_args(argv)

    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        print(f"ERROR: config.json not found: {cfg_path}")
        return 2

    try:
        data = config.parse_config_text(cfg_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: could not parse {cfg_path}: {exc}")
        return 2

    chosen = {
        "archive_root": args.archive_root,
        "calibration_root": args.calibration_root,
        "database_path": args.database_path,
        "gaia_db": args.gaia_db,
        "vsx_db": args.vsx_db,
        "exoplanet_db": args.exoplanet_db,
        "blind_fine": args.blind_fine,
        "blind_wide": args.blind_wide,
    }
    changes = apply_paths(data, chosen)

    if not changes:
        print("config.json paths already local; nothing to change.")
        return 0

    for key, old, new in changes:
        old_disp = old if old else "(unset)"
        new_disp = new if new else "(blanked -> default)"
        print(f"  {key}: {old_disp} -> {new_disp}")

    if args.dry_run:
        print("(dry-run: config.json not modified)")
        return 0

    try:
        # render_config_jsonc is the canonical, comment-preserving writer that
        # save_config_json wraps; call it directly so the exact --config path is
        # honoured (save_config_json always targets <root>/config.json).
        cfg_path.write_text(config.render_config_jsonc(data), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: failed to write {cfg_path}: {exc}")
        return 1
    print(f"OK: wrote {len(changes)} path change(s) to {cfg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
