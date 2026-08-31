"""Line spans for classified-dead functions; top-level dup names only."""
from __future__ import annotations

import ast
import json
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src_py"
OUT = Path(__file__).resolve().parent
cls = json.loads((OUT / "function_class.json").read_text(encoding="utf-8"))
dead = [x for x in cls["items"] if x["status"] == "dead"]


def spans_for_file(path: Path, names: set[str]) -> list[dict]:
    src = path.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(src)
    out = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            start = node.lineno
            # include decorator lines
            if node.decorator_list:
                start = min(d.lineno for d in node.decorator_list)
            end = node.end_lineno or start
            out.append({"name": node.name, "start": start, "end": end, "nlines": end - start + 1})
    return out


def top_level_dups() -> list[dict]:
    by_name: dict[str, list[dict]] = defaultdict(list)
    for p in sorted(SRC.glob("*.py")):
        src_lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        tree = ast.parse("\n".join(src_lines), filename=str(p))
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            start = node.lineno
            end = node.end_lineno or start
            body = "\n".join(src_lines[start - 1 : end])
            by_name[node.name].append(
                {
                    "file": f"src_py/{p.name}",
                    "line": start,
                    "end": end,
                    "sha": sha256(body.encode("utf-8")).hexdigest()[:16],
                    "nbytes": len(body.encode("utf-8")),
                }
            )
    items = []
    for name, locs in sorted(by_name.items()):
        files = {x["file"] for x in locs}
        if len(files) < 2:
            continue
        shas = {x["sha"] for x in locs}
        items.append(
            {
                "name": name,
                "n_files": len(files),
                "byte_equal": len(shas) == 1,
                "locs": [f"{x['file']}:{x['line']}" for x in locs],
                "locs_full": locs,
            }
        )
    return items


def main() -> None:
    by_mod: dict[str, list] = defaultdict(list)
    total = 0
    for rec in dead:
        by_mod[rec["module"]].append(rec["name"])
    spans_all = []
    for mod, names in sorted(by_mod.items()):
        path = SRC / f"{mod}.py"
        sp = spans_for_file(path, set(names))
        n = sum(x["nlines"] for x in sp)
        total += n
        spans_all.append({"module": mod, "nlines": n, "funcs": sp})
        print(f"{mod:32s} {n:5d} lines  {len(sp)} funcs")
    print("TOTAL dead func lines", total)
    (OUT / "dead_spans.json").write_text(
        json.dumps({"total_lines": total, "modules": spans_all}, indent=2) + "\n",
        encoding="utf-8",
    )
    dups = top_level_dups()
    (OUT / "dup_helpers_toplevel.json").write_text(
        json.dumps({"n": len(dups), "items": dups}, indent=2) + "\n", encoding="utf-8"
    )
    print("top-level dup names", len(dups))


if __name__ == "__main__":
    main()
