"""Regenerate docs/VYVAR_PARAMS.md from the parameter registry (PARAMS-REGISTRY-UI STEP 2).

The registry (``validation/params_registry.json``) supplies editorial metadata only;
defaults and types come from ``dataclasses.fields(AppConfig)`` introspection so this
document can never silently drift from ``config.py``.

Usage:
    python tools/gen_params_md.py            # rewrite docs/VYVAR_PARAMS.md
    python tools/gen_params_md.py --check     # print to stdout, do not write

This is documentation plumbing only; it touches no science path.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import params_registry as pr  # noqa: E402

OUT_MD = ROOT / "docs" / "VYVAR_PARAMS.md"

# A single volatile header line carries the timestamp + git HEAD; the freshness test
# strips lines with this prefix before comparing, so regeneration is deterministic.
VOLATILE_PREFIX = "_Generated "


def _git_head() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _num(x: float) -> str:
    f = float(x)
    if f.is_integer() and abs(f) < 1e15:
        return str(int(f))
    return repr(f)


def _range_str(rng: list | None) -> str:
    if not rng:
        return "-"
    return f"{_num(rng[0])} .. {_num(rng[1])}"


def _cell(s: str) -> str:
    return str(s).replace("|", "\\|")


def build_markdown(*, generated_at: str | None = None, git_head: str | None = None) -> str:
    registry = pr.load_registry()
    defaults = pr.appconfig_defaults()
    field_names = pr.appconfig_field_names()
    generated_at = generated_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    git_head = git_head or _git_head()

    tier_c = Counter(e["tier"] for e in registry.values())
    kind_c = Counter(e["kind"] for e in registry.values())
    widget_c = Counter(e["widget"] for e in registry.values())

    lines: list[str] = []
    lines.append("# VYVAR -- Config <-> UI parameter registry")
    lines.append("")
    lines.append("<!-- GENERATED FILE -- DO NOT EDIT BY HAND. -->")
    lines.append(
        "Regenerate with `python tools/gen_params_md.py`. Hand edits will be overwritten."
    )
    lines.append(
        "Source: `validation/params_registry.json` (editorial metadata) + "
        "`dataclasses.fields(AppConfig)` (defaults and types, from code)."
    )
    lines.append("")
    lines.append(f"{VOLATILE_PREFIX}{generated_at} at git HEAD {git_head}._")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Entries: {len(registry)}")
    lines.append(
        "- Tier: "
        + ", ".join(f"{t} {tier_c.get(t, 0)}" for t in pr.TIERS)
    )
    lines.append(
        "- Kind: "
        + ", ".join(f"{k} {kind_c.get(k, 0)}" for k in pr.KINDS)
    )
    lines.append(
        "- Widget: "
        + ", ".join(f"{w} {widget_c.get(w, 0)}" for w in pr.WIDGETS)
    )
    lines.append("")
    lines.append(
        "Columns: key, default, range, tier, kind, widget, label. `kind=resolved` means "
        "the runtime value can be auto-derived/overridden by the pipeline (the configured "
        "value is the base/fallback). `widget=custom` keys keep their hand-built UI; "
        "`widget=hidden` keys are plumbing not surfaced in the generated dashboard."
    )
    lines.append("")

    header = "| key | default | range | tier | kind | widget | label |"
    sep = "|-----|---------|-------|------|------|--------|-------|"

    for phase in pr.PHASES:
        keys = sorted(k for k, e in registry.items() if e["phase"] == phase)
        if not keys:
            continue
        lines.append(f"## {phase}")
        lines.append("")
        lines.append(header)
        lines.append(sep)
        for key in keys:
            e = registry[key]
            if key in defaults:
                default = pr.default_repr(defaults[key])
            elif key in field_names:
                default = "(resolved at runtime)"
            else:
                default = "?"
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{key}`",
                        _cell(default),
                        _range_str(e.get("range")),
                        _cell(e["tier"]),
                        _cell(e["kind"]),
                        _cell(e["widget"]),
                        _cell(e["label"]),
                    ]
                )
                + " |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def strip_volatile(text: str) -> str:
    """Remove the timestamp/HEAD header line so two regenerations compare equal."""
    return "\n".join(
        ln for ln in text.splitlines() if not ln.startswith(VOLATILE_PREFIX)
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="print to stdout, do not write")
    args = ap.parse_args(argv)
    text = build_markdown()
    if args.check:
        sys.stdout.write(text)
        return 0
    OUT_MD.write_text(text, encoding="utf-8")
    print(f"Wrote {OUT_MD} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
