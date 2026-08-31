"""Remove listed top-level functions from one src_py module by AST line span."""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src_py"
OUT = Path(__file__).resolve().parent
spans = json.loads((OUT / "dead_spans.json").read_text(encoding="utf-8"))


def remove_funcs(mod: str, names: list[str]) -> int:
    path = SRC / f"{mod}.py"
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    tree = ast.parse(text)
    cuts: list[tuple[int, int]] = []
    found = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            start = node.lineno
            if node.decorator_list:
                start = min(d.lineno for d in node.decorator_list)
            end = node.end_lineno or start
            cuts.append((start, end))
            found.add(node.name)
    missing = set(names) - found
    if missing:
        raise SystemExit(f"{mod}: missing {sorted(missing)}")
    extra = found - set(names)
    if extra:
        raise SystemExit(f"{mod}: unexpected extra {sorted(extra)}")
    # merge overlapping, drop from bottom
    cuts.sort(reverse=True)
    for start, end in cuts:
        # also drop one following blank line if present
        del_end = end
        if del_end < len(lines) and lines[del_end].strip() == "":
            del_end += 1
        del lines[start - 1 : del_end]
    path.write_text("".join(lines), encoding="utf-8")
    return sum(e - s + 1 for s, e in cuts)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: apply_dead_funcs.py MODULE [name ...]")
        return 2
    mod = argv[1]
    if len(argv) > 2:
        names = argv[2:]
    else:
        block = next(m for m in spans["modules"] if m["module"] == mod)
        names = [f["name"] for f in block["funcs"]]
    n = remove_funcs(mod, names)
    print(f"removed {n} lines from {mod}: {names}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
