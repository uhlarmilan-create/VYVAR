#!/usr/bin/env python3
"""Reproducible loader/validator for the VYVAR gates inventory.

This helper documents how gates were found and validates
``dev/validation/gates_inventory.json``. It does NOT auto-discover every gate
perfectly; discovery notes below are heuristics for human / future AST work.

Discovery notes (AST / heuristics)
----------------------------------
1. Grep for population filters: ``.loc[~``, ``cand_mask``, ``flux_map.pop``,
   ``quality = \"excluded\"``, ``IS_REJECTED``, ``DAOStarFinder(``.
2. Grep for admission / diag stamps: ``[SAT-DIAG]``, ``[CAL-DIAG]``,
   ``admit_pool_stars``, ``derive_pool_thresholds``, ``zone\" = \"saturated``.
3. AST (optional future): walk ``Compare`` / ``BoolOp`` under functions named
   ``*filter*``, ``*gate*``, ``*admit*``, ``build_global_comp_pool``,
   ``_apply_comp_metric_hard_filters``. Match call sites of
   ``assert_population_nonempty`` for empty-population guards.
4. Rank vs derived_fit rule (CRITICAL):
   - ``rank_statistic`` ONLY when the threshold is a percentile/quantile/order
     statistic OF THE POPULATION BEING FILTERED (admission cut on that quantity).
   - If a percentile estimates a distribution parameter (median/MAD/IQR as
     scatter estimator feeding a physical cut), use ``derived_fit`` and say so
     in ``note``.

Kind taxonomy (inventory convention)
-----------------------------------
- G1: hard membership / admission cut (removes population members).
- G2: quality / stability metric gate on candidates under consideration.
- G3: diagnostic, annotation, or convention gate (flags / provenance).

Usage
-----
  python dev/tools/scan_gates_inventory.py
  python dev/tools/scan_gates_inventory.py --validate
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
INVENTORY_PATH = REPO / "dev" / "validation" / "gates_inventory.json"

REQUIRED_FIELDS: tuple[str, ...] = (
    "gate_id",
    "file",
    "line",
    "function",
    "kind",
    "stage",
    "population",
    "quantity",
    "threshold_source",
    "threshold_value",
    "threshold_unit",
    "null_hypothesis",
    "param_names",
    "instrumented",
    "can_empty_population",
    "has_empty_guard",
    "ambiguous",
    "note",
)

KINDS = frozenset({"G1", "G2", "G3"})
STAGES = frozenset(
    {
        "calibration",
        "best_frame",
        "platesolve",
        "alignment",
        "dao_detection",
        "comp_pool",
        "comp_assignment",
        "photometry",
        "reporting",
    }
)
THRESHOLD_SOURCES = frozenset(
    {
        "literal_constant",
        "config_param",
        "derived_fit",
        "rank_statistic",
        "catalog_flag",
        "none",
    }
)


def load_inventory(path: Path = INVENTORY_PATH) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"gates inventory missing: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("gates_inventory.json root must be a JSON array")
    return raw


def _type_ok(name: str, value: Any) -> str | None:
    if name == "gate_id":
        return None if isinstance(value, str) and value.strip() else "gate_id must be non-empty string"
    if name == "file":
        if not isinstance(value, str) or not value.startswith("src_py/"):
            return "file must be string starting with src_py/"
        return None
    if name == "line":
        return None if isinstance(value, int) and not isinstance(value, bool) and value > 0 else "line must be positive int"
    if name == "function":
        return None if isinstance(value, str) and value.strip() else "function must be non-empty string"
    if name == "kind":
        return None if value in KINDS else f"kind must be one of {sorted(KINDS)}"
    if name == "stage":
        return None if value in STAGES else f"stage must be one of {sorted(STAGES)}"
    if name in ("population", "quantity", "threshold_unit", "null_hypothesis", "note"):
        return None if isinstance(value, str) else f"{name} must be string"
    if name == "threshold_source":
        return None if value in THRESHOLD_SOURCES else f"threshold_source must be one of {sorted(THRESHOLD_SOURCES)}"
    if name == "threshold_value":
        if value is None or isinstance(value, (int, float, str, bool)):
            return None
        return "threshold_value must be null, number, bool, or string expression"
    if name == "param_names":
        if not isinstance(value, list):
            return "param_names must be a list"
        if not all(isinstance(x, str) for x in value):
            return "param_names entries must be strings"
        return None
    if name in ("instrumented", "can_empty_population", "has_empty_guard", "ambiguous"):
        return None if isinstance(value, bool) else f"{name} must be bool"
    return f"unknown field {name}"


def validate_inventory(gates: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    seen_ids: set[str] = set()
    for i, g in enumerate(gates):
        prefix = f"[{i}]"
        if not isinstance(g, dict):
            errors.append(f"{prefix} entry is not an object")
            continue
        missing = [f for f in REQUIRED_FIELDS if f not in g]
        if missing:
            errors.append(f"{prefix} missing fields: {', '.join(missing)}")
        extra = sorted(set(g) - set(REQUIRED_FIELDS))
        if extra:
            errors.append(f"{prefix} unexpected fields: {', '.join(extra)}")
        for field in REQUIRED_FIELDS:
            if field not in g:
                continue
            err = _type_ok(field, g[field])
            if err:
                errors.append(f"{prefix} {g.get('gate_id', '?')}: {err}")
        gid = g.get("gate_id")
        if isinstance(gid, str):
            if gid in seen_ids:
                errors.append(f"{prefix} duplicate gate_id: {gid}")
            seen_ids.add(gid)
        fpath = g.get("file")
        if isinstance(fpath, str):
            abs_path = REPO / fpath
            if not abs_path.is_file():
                errors.append(f"{prefix} {gid}: file not found: {fpath}")
    return errors


def print_counts(gates: list[dict[str, Any]]) -> None:
    by_stage = Counter(str(g.get("stage")) for g in gates)
    by_kind = Counter(str(g.get("kind")) for g in gates)
    by_src = Counter(str(g.get("threshold_source")) for g in gates)
    print(f"gates_inventory: {len(gates)} gates ({INVENTORY_PATH.relative_to(REPO).as_posix()})")
    print("by stage:")
    for k in sorted(by_stage):
        print(f"  {k}: {by_stage[k]}")
    print("by kind:")
    for k in sorted(by_kind):
        print(f"  {k}: {by_kind[k]}")
    print("by threshold_source:")
    for k in sorted(by_src):
        print(f"  {k}: {by_src[k]}")
    n_amb = sum(1 for g in gates if g.get("ambiguous") is True)
    n_empty = sum(1 for g in gates if g.get("can_empty_population") is True)
    n_guard = sum(1 for g in gates if g.get("has_empty_guard") is True)
    print(f"ambiguous={n_amb} can_empty_population={n_empty} has_empty_guard={n_guard}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load/validate VYVAR gates_inventory.json")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Exit non-zero on schema/path errors (still prints counts when loadable).",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=INVENTORY_PATH,
        help="Override inventory JSON path",
    )
    args = parser.parse_args(argv)

    try:
        gates = load_inventory(args.path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: failed to load inventory: {exc}", file=sys.stderr)
        return 2

    errors = validate_inventory(gates)
    print_counts(gates)
    if errors:
        print(f"validation errors ({len(errors)}):", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        if args.validate:
            return 1
        return 0
    if args.validate:
        print("validation: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
