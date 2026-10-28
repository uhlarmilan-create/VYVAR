"""LAYER-2026-10 (code review 2 T13): import layering contracts.

1. Every src_py module imports on its own in a fresh interpreter (12 modules could not before).
2. No import cycle among module-level imports.
3. Cycles through call-time (function-local) imports only within a frozen list that may only shrink.
4. Pipeline modules never import a re-export through a facade (``pipeline``, ``photometry_core``,
   ``photometry``); the facades stay for dev/ scripts (D-FACADE-PERMANENT-01).
5. No streamlit below the UI layer (``app`` and ``ui_*``).
"""

from __future__ import annotations

import ast
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src_py"
MODULES = sorted(p.stem for p in SRC.glob("*.py"))
FACADES = {"pipeline", "photometry_core", "photometry"}
UI = {m for m in MODULES if m == "app" or m.startswith("ui_")}

# Call-time cycles that remain: driver <-> stage modules and dispatchers. Split in T12 (aperture
# driver, comp selection) or by design (night_run <-> run_job entry points). May only shrink.
ALLOWED_CALL_TIME_CYCLES = [
    {"aperture_photometry", "aperture_photometry_state", "aperture_photometry_target"},
    {"database", "observer_location"},
    {"config", "params_registry"},
    {"astro_index", "astrosolve"},
    {"comp_selection_per_target", "photometry_comp"},
    {"night_run", "run_job"},
]


def _is_type_checking(node: ast.AST) -> bool:
    t = getattr(node, "test", None)
    return isinstance(node, ast.If) and (
        (isinstance(t, ast.Name) and t.id == "TYPE_CHECKING")
        or (isinstance(t, ast.Attribute) and t.attr == "TYPE_CHECKING")
    )


def _imports(tree: ast.Module, *, module_level_only: bool) -> list[ast.AST]:
    out = []
    stack: list[tuple[ast.AST, bool]] = [(tree, False)]
    while stack:
        node, in_fn = stack.pop()
        if isinstance(node, (ast.Import, ast.ImportFrom)) and not (module_level_only and in_fn):
            out.append(node)
        if _is_type_checking(node):
            continue
        child_in_fn = in_fn or isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
        stack.extend((c, child_in_fn) for c in ast.iter_child_nodes(node))
    return out


def _targets(node: ast.AST) -> list[str]:
    if isinstance(node, ast.ImportFrom):
        return [node.module.split(".")[0]] if node.module and node.level == 0 else []
    return [a.name.split(".")[0] for a in node.names]


def _graph(module_level_only: bool) -> dict[str, set[str]]:
    g: dict[str, set[str]] = {}
    for m in MODULES:
        tree = ast.parse((SRC / f"{m}.py").read_text(encoding="utf-8"))
        g[m] = {t for n in _imports(tree, module_level_only=module_level_only) for t in _targets(n)}
        g[m] = {t for t in g[m] if t in MODULES and t != m}
    return g


def _sccs(g: dict[str, set[str]]) -> list[set[str]]:
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    on: set[str] = set()
    out: list[set[str]] = []
    counter = [0]
    sys.setrecursionlimit(10000)

    def strong(v: str) -> None:
        index[v] = low[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        on.add(v)
        for w in g[v]:
            if w not in index:
                strong(w)
                low[v] = min(low[v], low[w])
            elif w in on:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp = set()
            while True:
                w = stack.pop()
                on.discard(w)
                comp.add(w)
                if w == v:
                    break
            if len(comp) > 1:
                out.append(comp)

    for v in g:
        if v not in index:
            strong(v)
    return out


def test_every_module_imports_on_its_own() -> None:
    def run(m: str) -> tuple[str, str]:
        r = subprocess.run(
            [sys.executable, "-c", f"import {m}"],
            cwd=str(SRC),
            capture_output=True,
            text=True,
            timeout=300,
        )
        return m, (r.stderr.strip().splitlines() or [""])[-1] if r.returncode else ""

    with ThreadPoolExecutor(max_workers=8) as ex:
        bad = [f"{m}: {err}" for m, err in ex.map(run, MODULES) if err]
    assert bad == []


def test_no_module_level_import_cycle() -> None:
    assert _sccs(_graph(module_level_only=True)) == []


def test_call_time_cycles_only_frozen_ones() -> None:
    found = _sccs(_graph(module_level_only=False))
    extra = [sorted(c) for c in found if not any(c <= a for a in ALLOWED_CALL_TIME_CYCLES)]
    assert extra == [], "new import cycle (move the shared helper to the lower module)"


def test_pipeline_modules_do_not_import_through_facades() -> None:
    defined = {}
    for f in FACADES:
        tree = ast.parse((SRC / f"{f}.py").read_text(encoding="utf-8"))
        defined[f] = {
            n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))
        }
    bad = []
    for m in MODULES:
        if m in FACADES:
            continue
        tree = ast.parse((SRC / f"{m}.py").read_text(encoding="utf-8"))
        for n in _imports(tree, module_level_only=False):
            if isinstance(n, ast.ImportFrom) and n.module in FACADES:
                bad += [f"{m}:{n.lineno} {n.module}.{a.name}" for a in n.names if a.name not in defined[n.module]]
            elif isinstance(n, ast.Import) and any(a.name in FACADES for a in n.names):
                bad.append(f"{m}:{n.lineno} import {[a.name for a in n.names]}")
    assert bad == []


def test_no_streamlit_below_ui() -> None:
    bad = []
    for m in MODULES:
        if m in UI:
            continue
        tree = ast.parse((SRC / f"{m}.py").read_text(encoding="utf-8"))
        bad += [f"{m}:{n.lineno}" for n in _imports(tree, module_level_only=False) if "streamlit" in _targets(n)]
    assert bad == []
