"""Static kwarg compatibility scanner for src_py (PP-KWARG-01).

Detects call sites that pass keyword arguments absent from the resolved target
signature, including ``**dict_var`` splats built from ``dict(...)`` or ``{...}``
literals in the same function scope.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src_py"


@dataclass(frozen=True)
class KwargViolation:
    module: str
    line: int
    callee: str
    bad_kw: str
    snippet: str


def _param_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[frozenset[str], bool]:
    args = node.args
    names: set[str] = set()
    for a in args.posonlyargs + args.args + args.kwonlyargs:
        names.add(a.arg)
    if args.vararg:
        names.add(args.vararg.arg)
    has_varkw = args.kwarg is not None
    if args.kwarg:
        names.add(args.kwarg.arg)
    return frozenset(names), has_varkw


def _dict_keys_from_node(node: ast.AST) -> frozenset[str] | None:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "dict":
        keys: set[str] = set()
        for kw in node.keywords:
            if kw.arg is not None:
                keys.add(kw.arg)
        return frozenset(keys)
    if isinstance(node, ast.Dict):
        keys: set[str] = set()
        for k in node.keys:
            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                keys.add(k.value)
        return frozenset(keys)
    return None


def _collect_signatures() -> dict[str, tuple[frozenset[str], bool]]:
    sigs: dict[str, tuple[frozenset[str], bool]] = {}
    bare_counts: dict[str, int] = {}
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        mod = rel[:-3].replace("/", ".")
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                q = f"{mod}.{node.name}"
                sigs[q] = _param_names(node)
                bare_counts[node.name] = bare_counts.get(node.name, 0) + 1
            elif isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        q = f"{mod}.{node.name}.{item.name}"
                        sigs[q] = _param_names(item)
                        bare_counts[item.name] = bare_counts.get(item.name, 0) + 1
    for name, count in bare_counts.items():
        if count == 1:
            for q in sigs:
                if q.endswith(f".{name}"):
                    sigs[name] = sigs[q]
                    break
    return sigs


def _import_map(tree: ast.AST) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                local = alias.asname or alias.name
                mapping[local] = f"{node.module}.{alias.name}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name
                mapping[local] = alias.name
    return mapping


def _resolve_callee(func: ast.AST, imports: dict[str, str]) -> str | None:
    if isinstance(func, ast.Name):
        if func.id in imports:
            return imports[func.id]
        return func.id
    if isinstance(func, ast.Attribute):
        parts: list[str] = []
        cur: ast.AST = func
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            base = imports.get(cur.id, cur.id)
            parts.append(base)
            return ".".join(reversed(parts))
    return None


class _FunctionScanner(ast.NodeVisitor):
    def __init__(self, *, rel: str, sigs: dict[str, tuple[frozenset[str], bool]], imports: dict[str, str]) -> None:
        self.rel = rel
        self.sigs = sigs
        self.imports = imports
        self.violations: list[KwargViolation] = []
        self._dict_vars: dict[str, frozenset[str]] = {}

    def _lookup_sig(self, callee: str) -> tuple[frozenset[str], bool] | None:
        if callee in self.sigs:
            return self.sigs[callee]
        if "." in callee:
            tail = callee.rsplit(".", 1)[-1]
            if tail in self.sigs:
                return self.sigs[tail]
        return None

    def visit_Assign(self, node: ast.Assign) -> Any:
        keys = _dict_keys_from_node(node.value)
        if keys is not None:
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    self._dict_vars[tgt.id] = keys
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        callee = _resolve_callee(node.func, self.imports)
        if callee is None:
            self.generic_visit(node)
            return

        sig = self._lookup_sig(callee)
        if sig is None:
            self.generic_visit(node)
            return

        params, has_varkw = sig
        if has_varkw:
            self.generic_visit(node)
            return

        kw_names: set[str] = set()
        for kw in node.keywords:
            if kw.arg is not None:
                kw_names.add(kw.arg)
            elif isinstance(kw.value, ast.Name) and kw.value.id in self._dict_vars:
                kw_names.update(self._dict_vars[kw.value.id])

        for bad in sorted(kw_names - params):
            line = getattr(node, "lineno", 0)
            self.violations.append(
                KwargViolation(
                    module=self.rel,
                    line=line,
                    callee=callee,
                    bad_kw=bad,
                    snippet=f"unexpected keyword {bad!r} for {callee}",
                )
            )
        self.generic_visit(node)


def scan_src_py() -> list[KwargViolation]:
    sigs = _collect_signatures()
    hits: list[KwargViolation] = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        hits.extend(_scan_module_text(path.read_text(encoding="utf-8", errors="replace"), rel, sigs))
    return hits


def scan_source_text(text: str, module: str = "<fixture>") -> list[KwargViolation]:
    """Scan arbitrary source (fire-proof fixtures)."""
    return _scan_module_text(text, module, _collect_signatures())


def _scan_module_text(text: str, rel: str, sigs: dict[str, tuple[frozenset[str], bool]]) -> list[KwargViolation]:
    hits: list[KwargViolation] = []
    try:
        tree = ast.parse(text, filename=rel)
    except SyntaxError:
        return hits
    imports = _import_map(tree)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scanner = _FunctionScanner(rel=rel, sigs=sigs, imports=imports)
            scanner.visit(node)
            hits.extend(scanner.violations)
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    scanner = _FunctionScanner(rel=rel, sigs=sigs, imports=imports)
                    scanner.visit(item)
                    hits.extend(scanner.violations)
    return hits


def main() -> int:
    hits = scan_src_py()
    if not hits:
        print("OK: no kwarg mismatches")
        return 0
    for v in hits:
        print(f"{v.module}:{v.line}: {v.callee}() {v.bad_kw} -- {v.snippet}")
    print(f"FAIL: {len(hits)} kwarg mismatch(es)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
