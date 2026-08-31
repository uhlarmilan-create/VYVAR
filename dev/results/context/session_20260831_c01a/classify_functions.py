"""Classify vulture unused-function hits by whole-repo grep (src_py + dev)."""
from __future__ import annotations

import ast
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src_py"
DEV = ROOT / "dev"
OUT = Path(__file__).resolve().parent
SKIP_PARTS = {"__pycache__", "session_20260831_c01a"}

vult = json.loads((OUT / "vulture_functions.json").read_text(encoding="utf-8"))
funcs = [r for r in vult if r["kind"] == "function" and r["reachable_module"]]
print("unused-function reachable:", len(funcs))

# also top-level-only dup count
reach = json.loads((OUT / "reachability.json").read_text(encoding="utf-8"))


def load_corpus() -> list[tuple[str, list[str]]]:
    files = []
    for folder in (SRC, DEV):
        for p in folder.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in {".py", ".json"}:
                continue
            if any(x in p.parts for x in SKIP_PARTS):
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            rel = str(p.relative_to(ROOT)).replace("\\", "/")
            files.append((rel, text.splitlines()))
    return files


def exports_and_dynamic(mod_path: Path) -> set[str]:
    kept: set[str] = set()
    text = mod_path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return kept
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "__all__":
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                kept.add(elt.value)
    if re.search(r"st\.(button|sidebar|form_submit_button|fragment)", text):
        pass
    return kept


def main() -> int:
    corpus = load_corpus()
    names = sorted({r["name"] for r in funcs})
    # precompile
    pats = {n: re.compile(r"\b" + re.escape(n) + r"\b") for n in names}
    hits: dict[str, list[str]] = defaultdict(list)
    for rel, lines in corpus:
        joined_need = False
        # skip files that contain none of the names quickly
        blob = "\n".join(lines)
        for n, pat in pats.items():
            if n not in blob:
                continue
            for i, line in enumerate(lines, 1):
                if pat.search(line):
                    hits[n].append(f"{rel}:{i}:{line.strip()[:180]}")

    results = []
    by_module: dict[str, list[dict]] = defaultdict(list)
    for r in funcs:
        name = r["name"]
        mod = r["module"]
        def_file = f"src_py/{mod}.py"
        others = []
        same_def = []
        dynamic = []
        for h in hits.get(name, []):
            rel, rest = h.split(":", 1)
            line_no_s, _, snippet = rest.partition(":")
            try:
                line_no = int(line_no_s)
            except ValueError:
                line_no = -1
            is_def = rel == def_file and (
                snippet.startswith("def " + name) or f"def {name}(" in snippet
            )
            if is_def:
                same_def.append(h)
                continue
            if (
                f'"{name}"' in snippet
                or f"'{name}'" in snippet
                or "getattr(" in snippet
                or "globals(" in snippet
                or "__all__" in snippet
            ):
                dynamic.append(h)
            others.append(h)
        status = "dead"
        if dynamic and not others:
            status = "kept-dynamic"
        elif others:
            # usages besides def
            status = "kept-used"
        # Streamlit / registry heuristics on remaining dead
        if status == "dead":
            src_py_path = SRC / f"{mod}.py"
            if src_py_path.is_file():
                all_exp = exports_and_dynamic(src_py_path)
                if name in all_exp:
                    status = "kept-dynamic"
                    dynamic.append(f"{def_file}:__all__")
        rec = {
            **r,
            "status": status,
            "n_hits": len(hits.get(name, [])),
            "n_other": len(others),
            "other_sample": others[:8],
            "dynamic_sample": dynamic[:6],
        }
        results.append(rec)
        by_module[mod].append(rec)

    dead = [x for x in results if x["status"] == "dead"]
    kept_d = [x for x in results if x["status"] == "kept-dynamic"]
    kept_u = [x for x in results if x["status"] == "kept-used"]
    payload = {
        "n_vulture_functions_reachable": len(funcs),
        "n_dead": len(dead),
        "n_kept_dynamic": len(kept_d),
        "n_kept_used": len(kept_u),
        "dead_by_module": {
            m: [x["name"] for x in xs if x["status"] == "dead"]
            for m, xs in sorted(by_module.items())
            if any(x["status"] == "dead" for x in xs)
        },
        "items": results,
    }
    (OUT / "function_class.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print("dead", len(dead), "kept-dynamic", len(kept_d), "kept-used", len(kept_u))
    print("dead_by_module", json.dumps(payload["dead_by_module"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
