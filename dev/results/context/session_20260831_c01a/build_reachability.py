"""CONSOLIDATE-01A reachability closure from R1..R5. ASCII. Read-only on src_py/dev."""
from __future__ import annotations

import ast
import json
import re
import sys
from collections import defaultdict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src_py"
DEV = ROOT / "dev"
DOCS = ROOT / "docs"
OUT = Path(__file__).resolve().parent

STD_OR_THIRD = {
    "__future__", "abc", "argparse", "ast", "asyncio", "base64", "binascii",
    "collections", "concurrent", "contextlib", "copy", "csv", "ctypes",
    "dataclasses", "datetime", "decimal", "difflib", "enum", "fnmatch",
    "functools", "gc", "glob", "hashlib", "heapq", "html", "http", "importlib",
    "inspect", "io", "itertools", "json", "logging", "math", "mmap", "multiprocessing",
    "numbers", "operator", "os", "pathlib", "pickle", "platform", "pprint",
    "queue", "random", "re", "shutil", "signal", "socket", "sqlite3", "stat",
    "string", "struct", "subprocess", "sys", "tempfile", "textwrap", "threading",
    "time", "traceback", "typing", "types", "unittest", "uuid", "warnings",
    "weakref", "xml", "zipfile",
    "numpy", "np", "pandas", "pd", "astropy", "photutils", "matplotlib",
    "mpl_toolkits", "scipy", "sklearn", "streamlit", "st", "PIL", "cv2",
    "sep", "skyfield", "astroquery", "requests", "yaml", "toml", "pytest",
    "ruff", "numpy.polynomial",
}


def module_name(path: Path) -> str:
    return path.stem


def src_modules() -> dict[str, Path]:
    return {p.stem: p for p in sorted(SRC.glob("*.py"))}


def parse_imports(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
    except SyntaxError:
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level and not node.module:
                continue
            if node.module:
                names.add(node.module.split(".")[0])
    return names


def string_literals(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
    except SyntaxError:
        return []
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            out.append(node.value)
    return out


def has_def_main(path: Path) -> bool:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
    except SyntaxError:
        return False
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            return True
    return False


def has_argparse(path: Path) -> bool:
    text = path.read_text(encoding="utf-8", errors="replace")
    return "argparse" in text


def iter_py(folder: Path):
    for p in folder.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        yield p


def main() -> int:
    mods = src_modules()
    mod_names = set(mods)
    importers: dict[str, list[str]] = defaultdict(list)
    imports_of: dict[str, list[str]] = defaultdict(list)
    string_refs: dict[str, list[str]] = defaultdict(list)

    for name, path in mods.items():
        imported = parse_imports(path) & mod_names
        imported.discard(name)
        imports_of[name] = sorted(imported)
        for dep in imported:
            importers[dep].append(name)
        for lit in string_literals(path):
            for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", lit):
                if token in mod_names and token != name:
                    string_refs[token].append(f"{name}:str")
            if lit.endswith(".py"):
                stem = Path(lit).stem
                if stem in mod_names and stem != name:
                    string_refs[stem].append(f"{name}:strpath")

    roots: dict[str, list[str]] = defaultdict(list)

    # R1 Streamlit app
    roots["app"].append("R1")
    # string refs from app.py to other src_py modules
    app_text = mods["app"].read_text(encoding="utf-8", errors="replace")
    for name in mod_names:
        if name == "app":
            continue
        if re.search(rf"\b{re.escape(name)}\b", app_text):
            if name not in imports_of["app"] and f"app:str" not in string_refs[name]:
                string_refs[name].append("app:text")

    # R2 CLI: def main + argparse
    r2_argparse: list[str] = []
    r2_main_only: list[str] = []
    for name, path in mods.items():
        if not has_def_main(path):
            continue
        if has_argparse(path):
            r2_argparse.append(name)
            if "R2" not in roots[name]:
                roots[name].append("R2")
        else:
            r2_main_only.append(name)

    # R3 tests
    test_importers: dict[str, list[str]] = defaultdict(list)
    for tp in iter_py(DEV / "tests"):
        rel = str(tp.relative_to(ROOT)).replace("\\", "/")
        imported = parse_imports(tp) & mod_names
        text = tp.read_text(encoding="utf-8", errors="replace")
        for name in sorted(imported):
            test_importers[name].append(rel)
            if "R3" not in roots[name]:
                roots[name].append("R3")
        for name in mod_names:
            if name in imported:
                continue
            if re.search(rf"(?:from|import)\s+{re.escape(name)}\b", text):
                test_importers[name].append(rel + ":text")
                if "R3" not in roots[name]:
                    roots[name].append("R3")

    # R4 session_baseline_check.py
    sbc = DEV / "scripts" / "session_baseline_check.py"
    sbc_text = sbc.read_text(encoding="utf-8", errors="replace")
    sbc_imported = parse_imports(sbc) & mod_names
    for name in sorted(sbc_imported):
        if "R4" not in roots[name]:
            roots[name].append("R4")
    for name in mod_names:
        if re.search(rf"\b{re.escape(name)}\b", sbc_text):
            if "R4" not in roots[name]:
                # only if it looks like an import or invoke, not a comment-only
                if re.search(
                    rf"(?:from|import)\s+{re.escape(name)}\b|{re.escape(name)}\.py",
                    sbc_text,
                ):
                    if "R4" not in roots[name]:
                        roots[name].append("R4")

    # R5 commands named in STATE / ROADMAP / PROCESS
    doc_hits: dict[str, list[str]] = defaultdict(list)
    for doc_name in ("VYVAR_STATE.md", "VYVAR_ROADMAP.md", "VYVAR_PROCESS.md"):
        text = (DOCS / doc_name).read_text(encoding="utf-8", errors="replace")
        for name in mod_names:
            patterns = [
                rf"`{re.escape(name)}\.py`",
                rf"`{re.escape(name)}`",
                rf"python(?:3)?\s+{re.escape(name)}\.py",
                rf"src_py/{re.escape(name)}\.py",
                rf"\b{re.escape(name)}\.py\b",
            ]
            if any(re.search(p, text) for p in patterns):
                doc_hits[name].append(doc_name)
                if "R5" not in roots[name]:
                    roots[name].append("R5")

    # BFS closure: edges = AST imports + string module refs
    graph: dict[str, set[str]] = defaultdict(set)
    for name in mod_names:
        for dep in imports_of[name]:
            graph[name].add(dep)
        for ref in string_refs[name]:
            src = ref.split(":")[0]
            if src in mod_names:
                graph[src].add(name)

    # reverse: from importer to imported already in graph[importer]
    # also add test/sbc as synthetic roots already in `roots`

    reachable: set[str] = set()
    why: dict[str, set[str]] = defaultdict(set)
    q: deque[str] = deque()
    for name, rs in roots.items():
        if name in mod_names:
            reachable.add(name)
            for r in rs:
                why[name].add(r)
            q.append(name)

    while q:
        cur = q.popleft()
        for dep in graph[cur]:
            if dep not in reachable:
                reachable.add(dep)
                q.append(dep)
            why[dep].add(f"via:{cur}")

    # importers list for json: src_py importers + test + sbc + docs
    closure: dict[str, dict] = {}
    for name in sorted(mod_names):
        src_imps = sorted(importers.get(name, []))
        all_imps = list(src_imps)
        all_imps.extend(f"test:{p}" for p in test_importers.get(name, []))
        if name in sbc_imported or (name in roots and "R4" in roots[name]):
            all_imps.append("dev/scripts/session_baseline_check.py")
        for d in doc_hits.get(name, []):
            all_imps.append(f"docs/{d}")
        for s in string_refs.get(name, []):
            all_imps.append(f"str:{s}")
        closure[name] = {
            "roots": sorted(roots.get(name, [])),
            "reachable": name in reachable,
            "importers": sorted(set(all_imps)),
            "src_py_importers": src_imps,
            "n_src_py_importers": len(src_imps),
            "via": sorted(why.get(name, [])),
        }

    unreachable = [n for n in sorted(mod_names) if n not in reachable]
    zero_src = [n for n in sorted(mod_names) if not importers.get(n)]

    meta = {
        "head": "d320697",
        "n_modules": len(mod_names),
        "n_reachable": len(reachable),
        "n_unreachable": len(unreachable),
        "unreachable": unreachable,
        "zero_src_py_importers": zero_src,
        "r2_main_argparse": sorted(r2_argparse),
        "r2_main_no_argparse": sorted(r2_main_only),
        "r5_doc_hits": {k: v for k, v in sorted(doc_hits.items())},
        "r1": ["app"],
    }
    payload = {"meta": meta, "modules": closure}
    (OUT / "reachability.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="ascii"
    )
    (OUT / "reachability_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="ascii"
    )
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
