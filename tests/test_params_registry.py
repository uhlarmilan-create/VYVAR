"""Guard tests for the machine-readable parameter registry (PARAMS-REGISTRY-UI STEP 1).

Turns Config <-> registry parity into a tested property: every public AppConfig field
has exactly one registry entry, every registry entry names a real field, and every
entry validates against the schema (enum fields, range shape, ASCII-only text).
"""
from __future__ import annotations

import params_registry as pr


def _registry() -> dict:
    return pr.load_registry()


def test_registry_covers_every_public_field_exactly_once() -> None:
    reg = _registry()
    fields = pr.appconfig_field_names()
    reg_keys = set(reg.keys())

    missing = sorted(fields - reg_keys)
    extra = sorted(reg_keys - fields)
    assert not missing, f"AppConfig fields with no registry entry: {missing}"
    assert not extra, f"Registry entries naming non-existent AppConfig fields: {extra}"
    # exactly one entry per field: JSON object keys are unique by construction, so
    # coverage + no-extras above already implies a bijection with the field set.
    assert len(reg_keys) == len(fields), (
        f"Registry entry count {len(reg_keys)} != field count {len(fields)}"
    )


def test_every_entry_has_the_full_schema() -> None:
    reg = _registry()
    bad: list[str] = []
    for key, entry in reg.items():
        if not isinstance(entry, dict):
            bad.append(f"{key}: entry is not an object")
            continue
        for req in pr.ENTRY_KEYS:
            if req not in entry:
                bad.append(f"{key}: missing field '{req}'")
    assert not bad, "Registry entries with schema violations:\n" + "\n".join(bad)


def test_enum_fields_are_valid() -> None:
    reg = _registry()
    bad: list[str] = []
    for key, entry in reg.items():
        if entry.get("tier") not in pr.TIERS:
            bad.append(f"{key}: tier={entry.get('tier')!r} not in {pr.TIERS}")
        if entry.get("phase") not in pr.PHASES:
            bad.append(f"{key}: phase={entry.get('phase')!r} not in {pr.PHASES}")
        if entry.get("kind") not in pr.KINDS:
            bad.append(f"{key}: kind={entry.get('kind')!r} not in {pr.KINDS}")
        if entry.get("widget") not in pr.WIDGETS:
            bad.append(f"{key}: widget={entry.get('widget')!r} not in {pr.WIDGETS}")
    assert not bad, "Registry entries with invalid enum values:\n" + "\n".join(bad)


def test_range_is_null_or_ordered_numeric_pair() -> None:
    reg = _registry()
    bad: list[str] = []
    for key, entry in reg.items():
        rng = entry.get("range")
        if rng is None:
            continue
        if (
            not isinstance(rng, list)
            or len(rng) != 2
            or not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in rng)
        ):
            bad.append(f"{key}: range={rng!r} is not [number, number] or null")
            continue
        if not rng[0] < rng[1]:
            bad.append(f"{key}: range={rng!r} requires min < max")
    assert not bad, "Registry entries with invalid range:\n" + "\n".join(bad)


def test_strings_are_ascii_only() -> None:
    reg = _registry()
    bad: list[str] = []
    for key, entry in reg.items():
        if not key.isascii():
            bad.append(f"{key}: key is not ASCII")
        for sfield in ("label", "help", "tier", "phase", "kind", "widget"):
            val = entry.get(sfield)
            if isinstance(val, str) and not val.isascii():
                bad.append(f"{key}: {sfield} contains non-ASCII text")
        unit = entry.get("unit")
        if isinstance(unit, str) and not unit.isascii():
            bad.append(f"{key}: unit contains non-ASCII text")
    assert not bad, "Registry entries with non-ASCII strings:\n" + "\n".join(bad)


def test_basic_tier_keys_are_auto_widgets() -> None:
    # Basic-tier params must be renderable by the generated dashboard.
    reg = _registry()
    bad = [
        key for key, e in reg.items()
        if e.get("tier") == "basic" and e.get("widget") != "auto"
    ]
    assert not bad, f"basic-tier keys must be widget=auto: {sorted(bad)}"
