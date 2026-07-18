"""Standalone validator for editing config.json without the UI (CONFIG-HUMAN-EDIT STEP 4).

Checks a config.json file and reports, with human-friendly output:
  (a) syntax errors (with line/column numbers),
  (b) unknown keys (with the closest registered key as a suggestion),
  (c) values outside the registered/clamp range,
  (d) type mismatches vs the AppConfig field type.

Exit code is non-zero when any problem is found, so it is usable in scripts/CI.

Usage:
    python dev/scripts/validate_config.py                 # validate the repo config.json
    python dev/scripts/validate_config.py path/to/config.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401

_ROOT = _bootstrap.REPO_ROOT

import config  # noqa: E402
import params_registry as pr  # noqa: E402


_HEAD_KIND = {
    "bool": "bool",
    "int": "int",
    "float": "float",
    "str": "str",
    "path": "str",
    "dict": "dict",
    "mapping": "dict",
    "list": "list",
    "tuple": "list",
    "sequence": "list",
}


def _expected_kind(type_str: str) -> tuple[str | None, bool]:
    """Coarse (kind, optional) from an AppConfig annotation string.

    Uses the OUTERMOST type token (so ``list[dict[...]]`` is a list, not a dict) and treats
    ``X | None`` / ``Optional[X]`` as optional. Multi-type unions (e.g. ``str | int``) are
    ambiguous and skip the type check.
    """
    t = (type_str or "").strip().lower()
    if not t:
        return None, False
    parts = [p.strip() for p in t.split("|")]
    optional = "none" in parts or "optional" in t
    non_none = [p for p in parts if p and p != "none"]
    if len(non_none) != 1:
        return None, optional  # union of multiple real types -> skip
    head = non_none[0].split("[")[0].strip()
    return _HEAD_KIND.get(head), optional


def _kind_matches(kind: str, value: object) -> bool:
    if kind == "bool":
        return isinstance(value, bool)
    if kind == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "float":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "str":
        return isinstance(value, str)
    if kind == "dict":
        return isinstance(value, dict)
    if kind == "list":
        return isinstance(value, list)
    return True


def validate_text(text: str) -> tuple[list[tuple[str, str]], dict | None]:
    """Return (problems, parsed). problems is a list of (severity, message)."""
    registry = pr.load_registry()
    field_names = sorted(pr.appconfig_field_names())
    field_types = pr.appconfig_field_types()
    known = set(field_names) | set(config._LEGACY_CONFIG_KEYS)

    try:
        data = config.parse_config_text(text)
    except json.JSONDecodeError as exc:
        return (
            [("ERROR", f"syntax error at line {exc.lineno}, column {exc.colno}: {exc.msg}")],
            None,
        )

    problems: list[tuple[str, str]] = []
    for key in data:
        if key not in known:
            near = _difflib_suggest(key, field_names)
            hint = f" (did you mean '{near}'?)" if near else ""
            problems.append(("ERROR", f"unknown key '{key}'{hint}"))
            continue
        if key in config._LEGACY_CONFIG_KEYS:
            problems.append(
                ("WARN", f"deprecated key '{key}' is still accepted but should be migrated")
            )
            continue
        value = data[key]
        kind, optional = _expected_kind(field_types.get(key, ""))
        if value is None:
            if not optional:
                problems.append(("ERROR", f"key '{key}': null is not allowed for this field"))
            continue
        if kind is not None and not _kind_matches(kind, value):
            problems.append(
                ("ERROR", f"key '{key}': expected {kind}, got {type(value).__name__} ({value!r})")
            )
            continue
        rng = (registry.get(key) or {}).get("range")
        if rng and isinstance(value, (int, float)) and not isinstance(value, bool):
            lo, hi = rng
            if not (lo <= value <= hi):
                problems.append(
                    ("ERROR", f"key '{key}': value {value} outside allowed range [{lo}, {hi}]")
                )
    return problems, data


def _difflib_suggest(key: str, field_names: list[str]) -> str | None:
    import difflib

    near = difflib.get_close_matches(key, field_names, n=1)
    return near[0] if near else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "path",
        nargs="?",
        default=str(_ROOT / "config.json"),
        help="config.json to validate (default: repo config.json)",
    )
    args = ap.parse_args(argv)
    path = Path(args.path)

    if not path.is_file():
        print(f"ERROR: file not found: {path}")
        return 2

    problems, data = validate_text(path.read_text(encoding="utf-8"))
    errors = [m for sev, m in problems if sev == "ERROR"]
    warns = [m for sev, m in problems if sev == "WARN"]

    print(f"Validating {path}")
    if data is not None:
        print(f"  parsed OK: {len(data)} keys")
    for m in errors:
        print(f"  ERROR: {m}")
    for m in warns:
        print(f"  WARN:  {m}")

    if errors:
        print(f"\nFAIL: {len(errors)} error(s), {len(warns)} warning(s).")
        return 1
    print(f"\nOK: no errors, {len(warns)} warning(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
