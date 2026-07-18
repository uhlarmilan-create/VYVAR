"""Guard tests for the machine-readable parameter registry (PARAMS-REGISTRY-UI STEP 1).

Turns Config <-> registry parity into a tested property: every public AppConfig field
has exactly one registry entry, every registry entry names a real field, and every
entry validates against the schema (enum fields, range shape, ASCII-only text).
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import params_registry as pr

_ROOT = Path(__file__).resolve().parent.parent  # dev/ (holds tools/, validation/)
_REPO = Path(__file__).resolve().parents[2]  # repo root (holds docs/)


def _registry() -> dict:
    return pr.load_registry()


def _load_gen_module():
    spec = importlib.util.spec_from_file_location(
        "gen_params_md", _ROOT / "tools" / "gen_params_md.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


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


def test_owner_axis_full_coverage_and_valid_enum() -> None:
    # Ownership axis (PARAM-OWNERSHIP-WAVE-A): every entry declares exactly one owner
    # from the fixed vocabulary. Failures name the offending keys.
    reg = _registry()
    missing = sorted(k for k, e in reg.items() if "owner" not in e)
    assert not missing, f"registry entries with no owner: {missing}"
    bad = sorted(
        f"{k}={e.get('owner')!r}" for k, e in reg.items() if e.get("owner") not in pr.OWNERS
    )
    assert not bad, f"registry entries with owner not in {pr.OWNERS}:\n" + "\n".join(bad)


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


def test_generated_params_md_is_fresh(tmp_path) -> None:
    # Regenerating in a temp dir must reproduce the committed VYVAR_PARAMS.md exactly,
    # excluding only the volatile timestamp/HEAD header line. Doc freshness is a tested
    # property, not a discipline.
    gen = _load_gen_module()
    committed = (_REPO / "docs" / "VYVAR_PARAMS.md").read_text(encoding="utf-8")
    regenerated = gen.build_markdown()

    # exercise the write path into a throwaway dir (faithful "regenerate in a temp dir")
    tmp_out = tmp_path / "VYVAR_PARAMS.md"
    tmp_out.write_text(regenerated, encoding="utf-8")

    got = gen.strip_volatile(tmp_out.read_text(encoding="utf-8"))
    want = gen.strip_volatile(committed)
    if got != want:  # produce a compact, actionable diff head
        import difflib

        diff = "\n".join(
            list(difflib.unified_diff(want.splitlines(), got.splitlines(), lineterm=""))[:40]
        )
        raise AssertionError(
            "docs/VYVAR_PARAMS.md is stale; run `python tools/gen_params_md.py`.\n" + diff
        )


def test_help_is_nonempty_ascii_and_not_placeholder() -> None:
    # CONFIG-HUMAN-EDIT STEP 1: help is the single source of truth for config.json
    # comments + dashboard tooltips, ported from VYVAR_CONFIG_GUIDE_EN.md. It must be a
    # real explanation, never the old mechanical placeholder ("<Label> (<phase> parameter).").
    reg = _registry()
    placeholder = re.compile(r"\(\w+ parameter\)\.\s*$")
    empty: list[str] = []
    non_ascii: list[str] = []
    stale: list[str] = []
    for key, entry in reg.items():
        help_txt = entry.get("help")
        if not isinstance(help_txt, str) or not help_txt.strip():
            empty.append(key)
            continue
        if not help_txt.isascii():
            non_ascii.append(key)
        if placeholder.search(help_txt):
            stale.append(key)
    assert not empty, f"registry entries with empty help: {sorted(empty)}"
    assert not non_ascii, f"registry entries with non-ASCII help: {sorted(non_ascii)}"
    assert not stale, f"registry entries still using placeholder help: {sorted(stale)}"


def test_basic_tier_keys_are_auto_widgets() -> None:
    # Basic-tier params must be renderable by the generated dashboard.
    reg = _registry()
    bad = [
        key for key, e in reg.items()
        if e.get("tier") == "basic" and e.get("widget") != "auto"
    ]
    assert not bad, f"basic-tier keys must be widget=auto: {sorted(bad)}"


def test_hidden_widget_implies_expert_tier() -> None:
    # Cosmetic rule: tier has no UX meaning for a key that never renders, so every
    # widget=hidden entry is pinned to tier=expert to avoid confusion at review time.
    reg = _registry()
    bad = [
        key for key, e in reg.items()
        if e.get("widget") == "hidden" and e.get("tier") != "expert"
    ]
    assert not bad, f"widget=hidden keys must be tier=expert: {sorted(bad)}"
