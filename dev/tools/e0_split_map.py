#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CONSOLIDATE-01E0: measure-only split map of pipeline.py and photometry_core.py.

No product code is modified. Deterministic (sorted iteration, LPA seed order fixed).
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src_py"
TARGETS = ("pipeline.py", "photometry_core.py")
PHOTOMETRY_ALIAS = "photometry"
LPA_ITERS = 50
LINE_CAP = 4000
PACK_CAP = 4200
STAGE_ORDER = (
    "import",
    "calibration",
    "astrometry-MASTERSTAR",
    "phase0+1 comp selection",
    "photometry-shared",
    "phase2a photometry",
    "ePSF hooks",
    "exports-reports",
    "UI-only",
    "gate-only",
    "unreachable",
)
SCIENCE_STAGES = (
    "import",
    "calibration",
    "astrometry-MASTERSTAR",
    "phase0+1 comp selection",
    "phase2a photometry",
    "ePSF hooks",
    "exports-reports",
)

# night_run / importer seeds (product flow). Names are top-level in TARGETS.
SEED_STAGES: dict[str, tuple[str, ...]] = {
    # import
    "extract_fits_metadata": ("import",),
    "fits_metadata_from_primary_header": ("import",),
    "log_lights_binning_from_headers_preflight": ("import",),
    "scan_usb_folder": ("import",),
    "generate_observation_hash": ("import",),
    "observation_group_key_from_metadata": ("import",),
    # calibration / QC
    "run_draft_ram_calibration_qc_to_obs_files": ("calibration",),
    "estimate_archive_memory_profile": ("calibration",),
    "scan_calibrated_lights_pointing": ("calibration",),
    "compute_auto_fwhm_limit": ("calibration",),
    "_calibrate_one_light_disk": ("calibration",),
    "_calibrate_one_light_apply_masters_in_ram": ("calibration",),
    "_init_calibrate_batch_worker": ("calibration",),
    "_calibrate_batch_process_one": ("calibration",),
    # astrometry + MASTERSTAR (preprocess is this stage in the night run)
    "build_prefilter_rejected_map": ("astrometry-MASTERSTAR",),
    "calibrated_paths_for_draft_apply_filters": ("astrometry-MASTERSTAR",),
    "qc_enrich_calibrated_lights_in_place": ("astrometry-MASTERSTAR",),
    "_iter_light_fits": ("astrometry-MASTERSTAR",),
    "astrometry_align_and_build_masterstar": ("astrometry-MASTERSTAR",),
    "draft_is_multi_group_obs": ("astrometry-MASTERSTAR",),
    "resolve_preprocess_target_coordinates": ("astrometry-MASTERSTAR",),
    "preprocess_calibrated_to_processed": ("astrometry-MASTERSTAR",),
    "quick_preprocess_last_import": ("astrometry-MASTERSTAR",),
    "SkySurfaceOrderConflictError": ("astrometry-MASTERSTAR",),
    # phase 0+1
    "run_phase0_and_phase1": ("phase0+1 comp selection",),
    "select_active_targets": ("phase0+1 comp selection",),
    "select_comparison_stars_per_target": ("phase0+1 comp selection",),
    "ensure_full_variable_targets_if_presel_stub": ("phase0+1 comp selection",),
    # phase 2a
    "run_phase2a": ("phase2a photometry",),
    "run_full_photometry_pipeline": ("photometry-shared",),
    "run_sysrem_field": ("phase2a photometry",),
    "_select_comps_by_rms_then_color": ("phase0+1 comp selection",),
    "_select_comps_by_color_then_rms": ("phase0+1 comp selection",),
    "_select_comps_tiered": ("phase0+1 comp selection",),
    "calibrate_lights_to_calibrated": ("calibration",),
    "_get_plate_scale_from_cfg": ("photometry-shared",),
    "merge_photometry_pipeline_meta": ("photometry-shared",),
    # ePSF hooks living in the two files
    "_photometry_mode_run_flags": ("ePSF hooks",),
    "_epsf_target_catalog_ids": ("ePSF hooks",),
    "_epsf_lc_catalog_ids": ("ePSF hooks",),
    "_epsf_fit_catalog_ids": ("ePSF hooks",),
    "_fill_psf_catalog_columns": ("ePSF hooks",),
    "_export_catalog_psf_st_fields": ("ePSF hooks",),
    "load_epsf_metrics_for_draft": ("ePSF hooks",),
    # exports / LC artifacts
    "save_lightcurve_csv": ("exports-reports",),
    "save_lightcurve_png": ("exports-reports",),
    "save_field_map_png": ("exports-reports",),
    "save_target_field_map_png": ("exports-reports",),
    "save_cutout_png": ("exports-reports",),
    "apply_reporting_postprocess": ("exports-reports",),
    "auto_export_variability_candidates_csv": ("exports-reports",),
}

NIGHT_FN_STAGE = {
    "_night_run_preprocess": "astrometry-MASTERSTAR",
    "_night_run_platesolve": "astrometry-MASTERSTAR",
    "run_night_photometry": "phase2a photometry",
}

UI_NAME_RE = re.compile(r"^ui_.*\.py$")
PATCH_RE = re.compile(
    r"^(pipeline|photometry_core|photometry)\.([A-Za-z_][A-Za-z0-9_]*)$"
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _parse(path: Path) -> ast.Module:
    return ast.parse(_read(path), filename=str(path))


def _qid(file: str, name: str) -> str:
    return f"{file}:{name}"


def _section_hint(lines: list[str], lineno: int) -> str:
    lo = max(0, lineno - 16)
    comments: list[str] = []
    for i in range(lineno - 2, lo - 1, -1):
        if i < 0:
            break
        raw = lines[i].rstrip()
        s = raw.strip()
        if not s:
            if comments:
                break
            continue
        if s.startswith("#"):
            comments.append(s.lstrip("#").strip())
            continue
        break
    comments.reverse()
    text = " ".join(c for c in comments if c and not set(c) <= set("-=*#"))
    return text[:120]


def collect_symbols(file: str, tree: ast.Module, lines: list[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            end = int(getattr(node, "end_lineno", node.lineno) or node.lineno)
            rec = {
                "file": file,
                "name": node.name,
                "kind": "class" if isinstance(node, ast.ClassDef) else "function",
                "lineno": int(node.lineno),
                "end_lineno": end,
                "size": end - int(node.lineno) + 1,
                "section_hint": _section_hint(lines, int(node.lineno)),
                "methods": [],
            }
            if isinstance(node, ast.ClassDef):
                for ch in node.body:
                    if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        cend = int(getattr(ch, "end_lineno", ch.lineno) or ch.lineno)
                        rec["methods"].append(
                            {
                                "name": ch.name,
                                "lineno": int(ch.lineno),
                                "end_lineno": cend,
                                "size": cend - int(ch.lineno) + 1,
                                "ref_names": sorted(names_referenced(ch)[0] | names_referenced(ch)[1]),
                            }
                        )
            out[node.name] = rec
    return out


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def names_referenced(fn_node: ast.AST) -> tuple[set[str], set[str], set[str]]:
    """Return (call_names, all_load_names, getattr_or_string_names)."""
    calls: set[str] = set()
    loads: set[str] = set()
    dyn: set[str] = set()

    class V(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            n = _call_name(node.func)
            if n:
                calls.add(n)
            if isinstance(node.func, ast.Name) and node.func.id == "getattr" and len(node.args) >= 2:
                a1 = node.args[1]
                if isinstance(a1, ast.Constant) and isinstance(a1.value, str):
                    dyn.add(a1.value)
            self.generic_visit(node)

        def visit_Name(self, node: ast.Name) -> None:
            if isinstance(node.ctx, ast.Load):
                loads.add(node.id)

        def visit_Attribute(self, node: ast.Attribute) -> None:
            if isinstance(node.value, ast.Name) and node.value.id in (
                "pipeline",
                "photometry_core",
                PHOTOMETRY_ALIAS,
            ):
                loads.add(node.attr)
                calls.add(node.attr)
            self.generic_visit(node)

    V().visit(fn_node)
    return calls, loads, dyn


def module_imports(tree: ast.Module) -> dict[str, str]:
    """local_name -> 'module' or 'module.attr'."""
    m: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for a in node.names:
                m[a.asname or a.name.split(".")[0]] = a.name
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for a in node.names:
                if a.name == "*":
                    continue
                m[a.asname or a.name] = f"{mod}.{a.name}" if mod else a.name
    return m


def all_py_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for sub in (root / "src_py", root / "dev" / "tests", root / "dev" / "scripts", root / "dev" / "tools"):
        if not sub.is_dir():
            continue
        for p in sub.rglob("*.py"):
            if "__pycache__" in p.parts:
                continue
            out.append(p)
    return sorted(out)


def file_bucket(path: Path) -> str:
    rel = path.relative_to(REPO).as_posix()
    name = path.name
    if name == "night_run.py":
        return "night_run"
    if name == "importer.py":
        return "import"
    if name in ("app.py",) or UI_NAME_RE.match(name):
        return "UI-only"
    if name in ("export_reports.py", "photometry_report.py"):
        return "exports-reports"
    if name.startswith("epsf_") or name.startswith("psf_"):
        return "ePSF hooks"
    if rel.startswith("dev/tests/") or rel.startswith("dev/scripts/") or rel.startswith("dev/tools/"):
        return "gate-only"
    if rel.startswith("src_py/"):
        return "src_other"
    return "gate-only"


def enclosing_function(tree: ast.Module, lineno: int) -> str | None:
    hit: str | None = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = int(getattr(node, "end_lineno", node.lineno) or node.lineno)
            if int(node.lineno) <= lineno <= end:
                if hit is None:
                    hit = node.name
                else:
                    # innermost
                    hit = node.name
    # prefer innermost: scan again keeping smallest span
    best: tuple[int, str] | None = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = int(getattr(node, "end_lineno", node.lineno) or node.lineno)
            if int(node.lineno) <= lineno <= end:
                span = end - int(node.lineno)
                if best is None or span < best[0]:
                    best = (span, node.name)
    return best[1] if best else None


def night_stage_for_lineno(lineno: int, fn: str | None) -> str | None:
    if fn and fn in NIGHT_FN_STAGE:
        if fn == "run_night_photometry":
            return None  # mixed: photometry + ePSF + reports; use name seeds
        return NIGHT_FN_STAGE[fn]
    if fn == "run_night_pipeline":
        if lineno < 1520:
            return "import"
        if lineno < 1652:
            return "calibration"
        if lineno < 1841:
            return "astrometry-MASTERSTAR"
        return "phase2a photometry"
    return None


def scan_callers(
    symbol_names: dict[str, str],
    py_files: list[Path],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]], dict[str, list[str]]]:
    """callers[name] = list of {file, lineno, how}; dyn_refs; private_test_imports."""
    callers: dict[str, list[dict[str, Any]]] = defaultdict(list)
    dyn_refs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    private_tests: dict[str, list[str]] = defaultdict(list)
    seen: set[tuple[str, str, int, str]] = set()

    def add_caller(name: str, path: Path, lineno: int, how: str) -> None:
        if name not in symbol_names:
            return
        key = (name, path.as_posix(), int(lineno), how)
        if key in seen:
            return
        seen.add(key)
        rec = {
            "file": path.relative_to(REPO).as_posix(),
            "lineno": int(lineno),
            "how": how,
            "bucket": file_bucket(path),
        }
        callers[name].append(rec)
        if how.startswith("dyn") or how in ("getattr", "patch-string", "globals-key"):
            dyn_refs[name].append(rec)

    for path in py_files:
        try:
            src = _read(path)
            tree = ast.parse(src, filename=str(path))
        except (SyntaxError, UnicodeDecodeError):
            continue
        rel = path.relative_to(REPO).as_posix()
        imported: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if mod in ("pipeline", "photometry_core", PHOTOMETRY_ALIAS):
                    for a in node.names:
                        if a.name == "*":
                            continue
                        local = a.asname or a.name
                        imported[local] = a.name
                        add_caller(a.name, path, int(node.lineno), f"import-from:{mod}")
                        if a.name.startswith("_") and rel.startswith("dev/tests/"):
                            private_tests[a.name].append(rel)
            elif isinstance(node, ast.Import):
                for a in node.names:
                    if a.name in ("pipeline", "photometry_core", PHOTOMETRY_ALIAS):
                        imported[a.asname or a.name.split(".")[-1]] = f"mod:{a.name}"

        class CV(ast.NodeVisitor):
            def visit_Call(self, node: ast.Call) -> None:
                if isinstance(node.func, ast.Name) and node.func.id in imported:
                    add_caller(imported[node.func.id], path, int(node.lineno), "call-imported-name")
                if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                    base = node.func.value.id
                    if base in imported and str(imported[base]).startswith("mod:"):
                        add_caller(node.func.attr, path, int(node.lineno), f"call-attr:{base}")
                    if base in ("pipeline", "photometry_core", PHOTOMETRY_ALIAS):
                        add_caller(node.func.attr, path, int(node.lineno), f"call-attr:{base}")
                if isinstance(node.func, ast.Name) and node.func.id == "getattr" and len(node.args) >= 2:
                    a1 = node.args[1]
                    if isinstance(a1, ast.Constant) and isinstance(a1.value, str):
                        add_caller(a1.value, path, int(node.lineno), "getattr")
                self.generic_visit(node)

            def visit_Constant(self, node: ast.Constant) -> None:
                if isinstance(node.value, str):
                    m = PATCH_RE.match(node.value)
                    if m:
                        add_caller(m.group(2), path, int(getattr(node, "lineno", 0) or 0), "patch-string")
                    elif node.value in symbol_names and len(node.value) >= 8:
                        # long exact name as string: likely getattr/patch/__all__
                        add_caller(node.value, path, int(getattr(node, "lineno", 0) or 0), "string-literal")

        CV().visit(tree)
    return callers, dyn_refs, private_tests


def label_propagation(nodes: list[str], undirected: dict[str, Counter[str]]) -> dict[str, str]:
    labels = {n: n for n in nodes}
    order = sorted(nodes)
    for _ in range(LPA_ITERS):
        changed = False
        for n in order:
            neigh = undirected.get(n) or Counter()
            if not neigh:
                continue
            scores: dict[str, int] = defaultdict(int)
            for other, w in neigh.items():
                scores[labels[other]] += int(w)
            # tie-break: highest weight, then lexicographically smallest label
            best = max(scores.items(), key=lambda kv: (kv[1], -len(kv[0]), kv[0]))
            # actually: weight desc, then label asc
            best = min(scores.items(), key=lambda kv: (-kv[1], kv[0]))
            if best[0] != labels[n]:
                labels[n] = best[0]
                changed = True
        if not changed:
            break
    return labels


def merge_weak_clusters(
    labels: dict[str, str],
    undirected: dict[str, Counter[str]],
    sizes: dict[str, int],
) -> dict[str, str]:
    labels = dict(labels)

    def clusters() -> dict[str, list[str]]:
        g: dict[str, list[str]] = defaultdict(list)
        for n, lab in labels.items():
            g[lab].append(n)
        return g

    def edge_stats(members: set[str]) -> tuple[int, int, dict[str, int]]:
        internal = 0
        cross = 0
        to: dict[str, int] = defaultdict(int)
        seen: set[tuple[str, str]] = set()
        for a in members:
            for b, w in (undirected.get(a) or {}).items():
                pair = tuple(sorted((a, b)))
                if pair in seen:
                    continue
                seen.add(pair)
                if b in members:
                    internal += 1
                else:
                    cross += 1
                    to[labels[b]] += 1
        return internal, cross, to

    for _ in range(40):
        cl = clusters()
        weak: list[tuple[str, int, int, str]] = []
        for lab, members in cl.items():
            internal, cross, to = edge_stats(set(members))
            if cross > internal and to:
                target = max(to.items(), key=lambda kv: (kv[1], kv[0]))[0]
                weak.append((lab, internal, cross, target))
        if not weak:
            break
        # merge smallest weak cluster first
        weak.sort(key=lambda t: (sum(sizes[n] for n in cl[t[0]]), t[0]))
        lab, _i, _c, target = weak[0]
        if lab == target:
            break
        for n in cl[lab]:
            labels[n] = target
    return labels


def pipeline_method_stage(name: str) -> str | None:
    n = name.lower()
    if "calibrat" in n:
        return "calibration"
    if "preprocess" in n:
        return "astrometry-MASTERSTAR"
    return None


def primary_stage(stages: list[str]) -> str:
    for s in STAGE_ORDER:
        if s in stages:
            return s
    return "unreachable"


def propose_modules(
    symbols: dict[str, dict[str, Any]],
    qid_of: dict[str, str],
    labels: dict[str, str],
    undirected: dict[str, Counter[str]],
) -> list[dict[str, Any]]:
    """Stage-first packing; LPA only to split buckets over LINE_CAP."""
    by_stage: dict[str, list[str]] = defaultdict(list)
    for qid, rec in symbols.items():
        by_stage[rec["primary_stage"]].append(qid)
    for s in by_stage:
        by_stage[s].sort(key=lambda q: (-symbols[q]["size"], q))

    stage_module_name = {
        "import": "pipeline_import.py",
        "calibration": "pipeline_calibrate.py",
        "astrometry-MASTERSTAR": "pipeline_astrometry.py",
        "phase0+1 comp selection": "photometry_comp.py",
        "photometry-shared": "photometry_shared.py",
        "phase2a photometry": "photometry_phase2a.py",
        "ePSF hooks": "pipeline_epsf_hooks.py",
        "exports-reports": "photometry_exports.py",
        "UI-only": "pipeline_ui_helpers.py",
        "gate-only": "pipeline_gate_helpers.py",
        "unreachable": "pipeline_dead.py",
    }

    modules: list[dict[str, Any]] = []

    def emit(name: str, qids: list[str], stage: str, note: str) -> None:
        lines = sum(symbols[q]["size"] for q in qids)
        files = sorted({symbols[q]["file"] for q in qids})
        marked = ""
        if lines > LINE_CAP:
            marked = f"OVER_CAP {lines} lines; least-bad cut, mark for architect"
        modules.append(
            {
                "module": name,
                "stage": stage,
                "n_defs": len(qids),
                "lines": lines,
                "source_files": files,
                "defs": [symbols[q]["name"] for q in qids],
                "qids": qids,
                "over_cap": lines > LINE_CAP,
                "note": note or marked,
            }
        )

    for stage in STAGE_ORDER:
        qids = by_stage.get(stage) or []
        if not qids:
            continue
        by_file: dict[str, list[str]] = defaultdict(list)
        for q in qids:
            by_file[symbols[q]["file"]].append(q)
        for src_file, fqids in sorted(by_file.items()):
            stem_base = stage_module_name[stage]
            if src_file == "photometry_core.py" and stem_base.startswith("pipeline_"):
                stem_base = "photometry_" + stem_base[len("pipeline_") :]
            elif src_file == "pipeline.py" and stem_base.startswith("photometry_"):
                stem_base = "pipeline_" + stem_base[len("photometry_") :]
            base = stem_base
            GIANT = 800
            giants = [q for q in fqids if symbols[q]["size"] >= GIANT]
            rest = [q for q in fqids if symbols[q]["size"] < GIANT]
            for gq in sorted(giants, key=lambda q: -symbols[q]["size"]):
                gname = symbols[gq]["name"].lstrip("_")
                emit(
                    f"{base[:-3]}__{gname}.py",
                    [gq],
                    stage,
                    f"from {src_file}; single def {symbols[gq]['size']} lines (>=800). Function body split is a later E-task if still over cap.",
                )
            fqids = rest
            if not fqids:
                continue
            total = sum(symbols[q]["size"] for q in fqids)
            if total <= PACK_CAP:
                note = f"from {src_file}"
                emit(base, fqids, stage, note)
                continue
            # pack LPA clusters into bins <= LINE_CAP, largest first
            groups: dict[str, list[str]] = defaultdict(list)
            for q in fqids:
                groups[labels.get(q, q)].append(q)
            cluster_list = sorted(groups.values(), key=lambda g: -sum(symbols[q]["size"] for q in g))
            bins: list[list[str]] = []
            bin_size: list[int] = []
            leftovers: list[str] = []
            for g in cluster_list:
                gsz = sum(symbols[q]["size"] for q in g)
                if gsz > LINE_CAP:
                    acc: list[str] = []
                    acc_n = 0
                    for q in sorted(g, key=lambda x: -symbols[x]["size"]):
                        sz = symbols[q]["size"]
                        if acc and acc_n + sz > LINE_CAP:
                            bins.append(acc)
                            bin_size.append(acc_n)
                            acc, acc_n = [], 0
                        acc.append(q)
                        acc_n += sz
                    if acc:
                        bins.append(acc)
                        bin_size.append(acc_n)
                    continue
                placed = False
                for i, bs in enumerate(bin_size):
                    if bs + gsz <= LINE_CAP:
                        bins[i].extend(g)
                        bin_size[i] += gsz
                        placed = True
                        break
                if not placed:
                    bins.append(list(g))
                    bin_size.append(gsz)
            for i, b in enumerate(bins):
                suffix = "" if i == 0 else f"_{i + 1}"
                stem = base[:-3] + suffix + ".py"
                note = f"from {src_file}; stage over 4000; packed by LPA clusters, product stage kept"
                if sum(symbols[q]["size"] for q in b) > LINE_CAP:
                    note = f"from {src_file}; OVER_CAP after pack; least-bad cut"
                emit(stem, b, stage, note)
            if leftovers:
                emit(base.replace(".py", "_rest.py"), leftovers, stage, f"from {src_file}; overflow")

    return modules


def edge_counts_between(a: set[str], b: set[str], directed: dict[str, Counter[str]]) -> dict[str, Any]:
    ab = 0
    ba = 0
    pairs: list[dict[str, Any]] = []
    for src, dests in directed.items():
        for dst, w in dests.items():
            if src in a and dst in b:
                ab += int(w)
                pairs.append({"from": src, "to": dst, "weight": int(w), "dir": "a->b"})
            elif src in b and dst in a:
                ba += int(w)
                pairs.append({"from": src, "to": dst, "weight": int(w), "dir": "b->a"})
    pairs.sort(key=lambda p: (-p["weight"], p["from"], p["to"]))
    return {"a_to_b": ab, "b_to_a": ba, "total": ab + ba, "top": pairs[:30]}


def write_module_md(path: Path, modules: list[dict[str, Any]], extra: dict[str, Any]) -> None:
    lines = [
        "# CONSOLIDATE-01E0 proposed module map",
        "",
        "Measure only. Facades `pipeline.py` and `photometry_core.py` stay and re-export.",
        "Facade removal is E-final, a separate decision. Stage boundaries win over graph aesthetics.",
        f"Line cap {LINE_CAP}. Base {extra.get('base_sha', '')}.",
        "",
        "## Table",
        "",
        "| module | stage | n_defs | lines | over_cap | note |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for m in modules:
        note = (m.get("note") or "").replace("|", "/")
        lines.append(
            f"| `{m['module']}` | {m['stage']} | {m['n_defs']} | {m['lines']} | "
            f"{'YES' if m['over_cap'] else ''} | {note} |"
        )
    lines += ["", "## Defs per module (names)", ""]
    for m in modules:
        lines.append(f"### `{m['module']}` ({m['n_defs']} defs, {m['lines']} lines)")
        lines.append("")
        lines.append("Source files: " + ", ".join(m["source_files"]))
        lines.append("")
        names = m["defs"]
        chunk = ", ".join(f"`{n}`" for n in names)
        lines.append(chunk)
        lines.append("")
        lines.append("Imports it will need (same-file/cross-file callees outside this module):")
        for imp in m.get("imports_needed", [])[:40]:
            lines.append(f"- `{imp}`")
        if len(m.get("imports_needed", [])) > 40:
            lines.append(f"- ... {len(m['imports_needed']) - 40} more")
        lines.append("")
        lines.append("Who imports it (external files that call these defs today):")
        for w in m.get("who_imports", [])[:30]:
            lines.append(f"- `{w}`")
        lines.append("")
    p2 = extra.get("phase2a_vs_comp") or {}
    lines += [
        "## photometry_core phase2a vs comp-selection (do not auto-merge)",
        "",
        f"- phase0+1 symbols: {p2.get('n_comp')}, lines {p2.get('lines_comp')}",
        f"- phase2a symbols: {p2.get('n_p2a')}, lines {p2.get('lines_p2a')}",
        f"- shared (both stages): {p2.get('n_shared')}, lines {p2.get('lines_shared')}",
        f"- directed call weight comp->phase2a: {p2.get('comp_to_p2a')}",
        f"- directed call weight phase2a->comp: {p2.get('p2a_to_comp')}",
        f"- directed call weight comp->shared: {p2.get('comp_to_shared')} / shared->comp: {p2.get('shared_to_comp')}",
        f"- directed call weight phase2a->shared: {p2.get('p2a_to_shared')} / shared->phase2a: {p2.get('shared_to_p2a')}",
        "",
        "These two stay separate proposed modules even if they call each other.",
        "Shared names go to `photometry_shared.py` in the table if that module exists.",
        "",
        "## Facades",
        "",
        "- `pipeline.py` re-exports every name moved out of it.",
        "- `photometry_core.py` re-exports every name moved out of it (and `photometry.py` star-import stays).",
        "- Spawn MP workers listed in the risk register must remain importable as `pipeline.<name>`.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "dev" / "results" / "context" / "session_20260901_e0")
    args = ap.parse_args()
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    file_trees: dict[str, ast.Module] = {}
    file_lines: dict[str, list[str]] = {}
    file_imps: dict[str, dict[str, str]] = {}
    symbols_by_file: dict[str, dict[str, dict[str, Any]]] = {}
    name_to_file: dict[str, str] = {}
    collisions: list[str] = []

    for fn in TARGETS:
        path = SRC / fn
        src = _read(path)
        file_lines[fn] = src.splitlines()
        tree = ast.parse(src, filename=str(path))
        file_trees[fn] = tree
        file_imps[fn] = module_imports(tree)
        symbols_by_file[fn] = collect_symbols(fn, tree, file_lines[fn])
        for name in symbols_by_file[fn]:
            if name in name_to_file:
                collisions.append(name)
            name_to_file[name] = fn

    # qid index
    symbols: dict[str, dict[str, Any]] = {}
    for fn, sm in symbols_by_file.items():
        for name, rec in sm.items():
            symbols[_qid(fn, name)] = rec

    # callees
    directed: dict[str, Counter[str]] = defaultdict(Counter)
    for fn, tree in file_trees.items():
        sm = symbols_by_file[fn]
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            src_q = _qid(fn, node.name)
            calls, loads, dyn = names_referenced(node)
            refs = calls | {n for n in loads if n in name_to_file}
            same: list[str] = []
            cross: list[str] = []
            for n in sorted(refs):
                if n not in name_to_file:
                    continue
                if n == node.name and name_to_file[n] == fn:
                    continue
                dst_file = name_to_file[n]
                dst_q = _qid(dst_file, n)
                directed[src_q][dst_q] += 1
                if dst_file == fn:
                    same.append(n)
                else:
                    cross.append(n)
            rec = symbols[src_q]
            rec["callees_same_file"] = sorted(set(same))
            rec["callees_cross_file"] = sorted(set(cross))
            rec["dyn_names_in_body"] = sorted(dyn)

    # callers across repo
    py_files = [p for p in all_py_files(REPO) if p.name != "e0_split_map.py"]
    callers, dyn_refs, private_tests = scan_callers(name_to_file, py_files)

    # same-file callers from directed
    same_file_callers: dict[str, list[str]] = defaultdict(list)
    for src, dests in directed.items():
        src_name = src.split(":", 1)[1]
        src_file = src.split(":", 1)[0]
        for dst in dests:
            dst_name = dst.split(":", 1)[1]
            dst_file = dst.split(":", 1)[0]
            if src_file == dst_file:
                same_file_callers[dst_name].append(src_name)

    for fn, sm in symbols_by_file.items():
        for name, rec in sm.items():
            ext = callers.get(name, [])
            rec["callers_external"] = ext
            rec["callers_same_file"] = sorted(set(same_file_callers.get(name, [])))
            rec["n_callers_external"] = len(ext)
            rec["n_callers_same_file"] = len(rec["callers_same_file"])

    # reachability stages
    stages: dict[str, set[str]] = {q: set() for q in symbols}
    for name, st in SEED_STAGES.items():
        if name in name_to_file:
            stages[_qid(name_to_file[name], name)].update(st)

    # night_run / importer extra: any imported name in those files
    for path in py_files:
        if path.name not in ("night_run.py", "importer.py", "export_reports.py", "photometry_report.py") and not (
            path.name.startswith("epsf_") or path.name.startswith("psf_")
        ):
            if not UI_NAME_RE.match(path.name) and path.name != "app.py":
                if not str(path.relative_to(REPO)).startswith("dev/tests/"):
                    continue
        try:
            tree = _parse(path)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if (node.module or "") not in ("pipeline", "photometry_core", PHOTOMETRY_ALIAS):
                continue
            fn = enclosing_function(tree, int(node.lineno))
            st: str | None = None
            if path.name == "night_run.py":
                st = night_stage_for_lineno(int(node.lineno), fn)
            elif path.name == "importer.py":
                st = "import"
            elif path.name in ("export_reports.py", "photometry_report.py"):
                st = "exports-reports"
            elif path.name.startswith("epsf_") or path.name.startswith("psf_"):
                st = "ePSF hooks"
            elif UI_NAME_RE.match(path.name) or path.name == "app.py":
                st = "UI-only"
            elif str(path.relative_to(REPO)).startswith("dev/tests/"):
                st = "gate-only"
            if not st:
                continue
            for a in node.names:
                if a.name in name_to_file:
                    stages[_qid(name_to_file[a.name], a.name)].add(st)

    # Independent same-file BFS per science stage so hubs do not paint the other file
    # or collapse phase2a into phase0+1.
    same_file_edges: dict[str, list[str]] = defaultdict(list)
    for src, dests in directed.items():
        src_file = src.split(":", 1)[0]
        for dst in dests:
            if dst.split(":", 1)[0] == src_file:
                same_file_edges[src].append(dst)

    reach: dict[str, set[str]] = {s: set() for s in SCIENCE_STAGES}
    for name, st_tuple in SEED_STAGES.items():
        if name not in name_to_file:
            continue
        q0 = _qid(name_to_file[name], name)
        for st in st_tuple:
            if st not in reach:
                continue
            stack = [q0]
            seen = reach[st]
            seen.add(q0)
            while stack:
                cur = stack.pop()
                for nxt in same_file_edges.get(cur, []):
                    if nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)

    # AstroPipeline methods (class is a hub; do not BFS the class as one node).
    ap = symbols_by_file.get("pipeline.py", {}).get("AstroPipeline")
    if ap:
        for meth in ap.get("methods") or []:
            st = pipeline_method_stage(str(meth.get("name") or ""))
            if st is None or st not in reach:
                continue
            seen = reach[st]
            for n in meth.get("ref_names") or []:
                if n not in name_to_file or name_to_file[n] != "pipeline.py":
                    continue
                q0 = _qid("pipeline.py", n)
                if q0 not in seen:
                    seen.add(q0)
                    stack = [q0]
                    while stack:
                        cur = stack.pop()
                        for nxt in same_file_edges.get(cur, []):
                            if nxt not in seen:
                                seen.add(nxt)
                                stack.append(nxt)

    FILE_STAGE_PREF = {
        "pipeline.py": ("import", "calibration", "astrometry-MASTERSTAR", "ePSF hooks"),
        "photometry_core.py": (
            "phase0+1 comp selection",
            "phase2a photometry",
            "exports-reports",
            "ePSF hooks",
            "calibration",
        ),
    }

    for qid, rec in symbols.items():
        sci = [s for s in SCIENCE_STAGES if qid in reach[s]]
        extra = {s for s in stages[qid] if s not in SCIENCE_STAGES}
        if "phase0+1 comp selection" in sci and "phase2a photometry" in sci:
            primary = "photometry-shared"
            st_list = ["photometry-shared"] + [s for s in sci if s not in ("phase0+1 comp selection", "phase2a photometry")]
        elif len(sci) == 1:
            primary = sci[0]
            st_list = list(sci)
        elif len(sci) > 1:
            pref = FILE_STAGE_PREF.get(rec["file"], STAGE_ORDER)
            primary = next((s for s in pref if s in sci), sci[0])
            st_list = list(sci)
        else:
            buckets = {c["bucket"] for c in rec.get("callers_external") or []}
            src_py_files = [
                c["file"]
                for c in rec.get("callers_external") or []
                if str(c["file"]).startswith("src_py/")
            ]
            seed_st = SEED_STAGES.get(rec["name"], ())
            if "photometry-shared" in seed_st:
                primary, st_list = "photometry-shared", ["photometry-shared"]
            elif extra:
                primary = primary_stage(
                    sorted(extra, key=lambda s: STAGE_ORDER.index(s) if s in STAGE_ORDER else 99)
                )
                st_list = sorted(extra, key=lambda s: STAGE_ORDER.index(s) if s in STAGE_ORDER else 99)
            elif any("comp_selection" in f or "pinned_ensembles" in f for f in src_py_files):
                primary, st_list = "phase0+1 comp selection", ["phase0+1 comp selection"]
            elif rec["file"] == "photometry_core.py" and any(f.endswith("pipeline.py") for f in src_py_files):
                primary, st_list = "photometry-shared", ["photometry-shared"]
            elif any(f.endswith("night_run.py") for f in src_py_files):
                primary, st_list = "photometry-shared", ["photometry-shared"]
            elif any("/epsf_" in f or "/psf_" in f for f in src_py_files):
                primary, st_list = "ePSF hooks", ["ePSF hooks"]
            elif any(
                "export_reports" in f or "photometry_report" in f or "check_star" in f or "method_lc" in f
                for f in src_py_files
            ):
                primary, st_list = "exports-reports", ["exports-reports"]
            elif any(
                not f.endswith("app.py") and "/ui_" not in f for f in src_py_files
            ):
                primary, st_list = (
                    ("photometry-shared", ["photometry-shared"])
                    if rec["file"] == "photometry_core.py"
                    else ("astrometry-MASTERSTAR", ["astrometry-MASTERSTAR"])
                )
            elif "ePSF hooks" in buckets:
                primary, st_list = "ePSF hooks", ["ePSF hooks"]
            elif "exports-reports" in buckets:
                primary, st_list = "exports-reports", ["exports-reports"]
            elif buckets <= {"UI-only"} and buckets:
                primary, st_list = "UI-only", ["UI-only"]
            elif buckets <= {"gate-only"} and buckets:
                primary, st_list = "gate-only", ["gate-only"]
            else:
                primary, st_list = "unreachable", ["unreachable"]
        rec["stages"] = st_list
        rec["primary_stage"] = primary
        rec["science_stages"] = sci

    ap_q = _qid("pipeline.py", "AstroPipeline")
    if ap_q in symbols:
        symbols[ap_q]["primary_stage"] = "calibration"
        symbols[ap_q]["stages"] = ["calibration", "astrometry-MASTERSTAR"]

    # undirected graph for LPA
    nodes = sorted(symbols)
    undirected: dict[str, Counter[str]] = defaultdict(Counter)
    for src, dests in directed.items():
        for dst, w in dests.items():
            undirected[src][dst] += int(w)
            undirected[dst][src] += int(w)

    sizes = {q: symbols[q]["size"] for q in nodes}
    labels = label_propagation(nodes, undirected)
    labels = merge_weak_clusters(labels, undirected, sizes)

    clusters_raw: dict[str, list[str]] = defaultdict(list)
    for n, lab in labels.items():
        clusters_raw[lab].append(n)

    cluster_rows: list[dict[str, Any]] = []
    for lab, members in clusters_raw.items():
        mset = set(members)
        internal = 0
        cross = 0
        seen_e: set[tuple[str, str]] = set()
        for a in members:
            for b, w in undirected[a].items():
                pair = tuple(sorted((a, b)))
                if pair in seen_e:
                    continue
                seen_e.add(pair)
                if b in mset:
                    internal += 1
                else:
                    cross += 1
        cluster_rows.append(
            {
                "id": lab,
                "n_defs": len(members),
                "lines": sum(sizes[m] for m in members),
                "internal_edges": internal,
                "cross_edges": cross,
                "not_a_module": cross > internal,
                "members": [symbols[m]["name"] for m in sorted(members, key=lambda x: -sizes[x])],
                "qids": sorted(members),
                "primary_stages": sorted({symbols[m]["primary_stage"] for m in members}),
            }
        )
    cluster_rows.sort(key=lambda r: (-r["lines"], r["id"]))
    for i, row in enumerate(cluster_rows, 1):
        row["cluster"] = f"C{i:02d}"

    # top 20 cross-cluster edges
    q_to_c = {}
    for row in cluster_rows:
        for q in row["qids"]:
            q_to_c[q] = row["cluster"]
    cross_edges: list[dict[str, Any]] = []
    seen_ce: set[tuple[str, str]] = set()
    for src, dests in directed.items():
        for dst, w in dests.items():
            c1, c2 = q_to_c[src], q_to_c[dst]
            if c1 == c2:
                continue
            pair = tuple(sorted((src, dst)))
            if pair in seen_ce:
                continue
            seen_ce.add(pair)
            # sum both directions
            w2 = int(directed.get(dst, {}).get(src, 0))
            cross_edges.append(
                {
                    "a": symbols[src]["name"],
                    "a_file": symbols[src]["file"],
                    "a_cluster": c1,
                    "b": symbols[dst]["name"],
                    "b_file": symbols[dst]["file"],
                    "b_cluster": c2,
                    "weight": int(w) + w2,
                    "a_to_b": int(w),
                    "b_to_a": w2,
                }
            )
    cross_edges.sort(key=lambda e: (-e["weight"], e["a"], e["b"]))
    top20 = cross_edges[:20]

    # phase2a vs comp
    comp_q = {q for q, r in symbols.items() if r["primary_stage"] == "phase0+1 comp selection"}
    p2a_q = {q for q, r in symbols.items() if r["primary_stage"] == "phase2a photometry"}
    shared_q = {q for q, r in symbols.items() if r["primary_stage"] == "photometry-shared"}
    both_names = {symbols[q]["name"] for q in shared_q}
    e_cp = edge_counts_between(comp_q, p2a_q, directed)
    e_cs = edge_counts_between(comp_q, shared_q, directed)
    e_ps = edge_counts_between(p2a_q, shared_q, directed)

    modules = propose_modules(symbols, {r["name"]: q for q, r in symbols.items()}, labels, undirected)

    # split shared photometry into photometry_shared if they landed in phase2a or comp
    # (already stage-primary so shared has primary of earlier stage = phase0+1). Flag them.
    for m in modules:
        m["imports_needed"] = []
        m["who_imports"] = []
        mset = set(m["qids"])
        imps: set[str] = set()
        who: set[str] = set()
        for q in m["qids"]:
            rec = symbols[q]
            for c in rec.get("callees_same_file", []) + rec.get("callees_cross_file", []):
                if c not in name_to_file:
                    continue
                cq = _qid(name_to_file[c], c)
                if cq not in mset:
                    imps.add(f"{name_to_file[c]}:{c}")
            for ext in rec.get("callers_external") or []:
                who.add(ext["file"])
            for other in rec.get("callers_same_file") or []:
                if other in name_to_file:
                    oq = _qid(name_to_file[other], other)
                    if oq not in mset:
                        imps.add(f"caller-in-other-module:{other}")
        m["imports_needed"] = sorted(imps)
        m["who_imports"] = sorted(who)

    # __all__
    all_names: list[str] = []
    for fn, tree in file_trees.items():
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id == "__all__":
                        try:
                            all_names.extend(ast.literal_eval(node.value))
                        except (ValueError, TypeError, SyntaxError, MemoryError):
                            pass

    mp_workers = [
        "_init_calibrate_batch_worker",
        "_calibrate_batch_process_one",
        "_init_export_per_frame_worker",
        "_export_per_frame_ram_worker_task",
        "_calibrate_one_light_disk",
    ]

    pickle_hits = []
    for fn in TARGETS:
        text = "\n".join(file_lines[fn])
        if "pickle." in text:
            pickle_hits.append(fn)

    risk = {
        "in__all__": sorted(n for n in all_names if n in name_to_file),
        "string_or_getattr": {
            n: dyn_refs[n]
            for n in sorted(dyn_refs)
            if n in name_to_file
        },
        "private_test_imports": {k: sorted(set(v)) for k, v in sorted(private_tests.items())},
        "mp_spawn_workers": mp_workers,
        "pickle_in_files": pickle_hits,
        "name_collisions_across_two_files": sorted(set(collisions)),
        "photometry_star_import": "src_py/photometry.py re-exports photometry_core via star import",
    }

    # counts
    summary = {
        "pipeline.py": {
            "lines": len(file_lines["pipeline.py"]),
            "top_level_functions": sum(
                1 for r in symbols_by_file["pipeline.py"].values() if r["kind"] == "function"
            ),
            "top_level_classes": sum(
                1 for r in symbols_by_file["pipeline.py"].values() if r["kind"] == "class"
            ),
        },
        "photometry_core.py": {
            "lines": len(file_lines["photometry_core.py"]),
            "top_level_functions": sum(
                1 for r in symbols_by_file["photometry_core.py"].values() if r["kind"] == "function"
            ),
            "top_level_classes": sum(
                1 for r in symbols_by_file["photometry_core.py"].values() if r["kind"] == "class"
            ),
        },
        "name_collisions": sorted(set(collisions)),
        "stage_counts": {
            s: {
                "n": sum(1 for r in symbols.values() if r["primary_stage"] == s),
                "lines": sum(r["size"] for r in symbols.values() if r["primary_stage"] == s),
            }
            for s in STAGE_ORDER
        },
    }

    symbol_out = {
        "summary": summary,
        "symbols": [
            {
                **{k: v for k, v in rec.items() if k != "callers_external"},
                "qid": qid,
                "callers_external_n": rec.get("n_callers_external", 0),
                "callers_external_files": sorted({c["file"] for c in rec.get("callers_external") or []}),
                "callers_external_sample": (rec.get("callers_external") or [])[:15],
            }
            for qid, rec in sorted(symbols.items(), key=lambda kv: (kv[1]["file"], kv[1]["lineno"]))
        ],
    }

    clusters_out = {
        "algorithm": "label-propagation, nodes sorted, neighbor-label weight then lex tie-break, then merge if cross>internal",
        "n_clusters": len(cluster_rows),
        "clusters": cluster_rows,
        "top20_cross_cluster_edges": top20,
        "n_cross_cluster_edges": len(cross_edges),
        "weak_clusters_remaining": [c["cluster"] for c in cluster_rows if c["not_a_module"]],
    }

    p2_extra = {
        "n_comp": len(comp_q),
        "lines_comp": sum(sizes[q] for q in comp_q),
        "n_p2a": len(p2a_q),
        "lines_p2a": sum(sizes[q] for q in p2a_q),
        "n_shared": len(shared_q),
        "lines_shared": sum(sizes[q] for q in shared_q),
        "shared_names": sorted(both_names),
        "comp_to_p2a": e_cp["a_to_b"],
        "p2a_to_comp": e_cp["b_to_a"],
        "comp_to_shared": e_cs["a_to_b"],
        "shared_to_comp": e_cs["b_to_a"],
        "p2a_to_shared": e_ps["a_to_b"],
        "shared_to_p2a": e_ps["b_to_a"],
        "top_edges": e_cp["top"][:20],
    }

    (out_dir / "symbol_map.json").write_text(json.dumps(symbol_out, indent=2) + "\n", encoding="utf-8")
    (out_dir / "clusters.json").write_text(json.dumps(clusters_out, indent=2) + "\n", encoding="utf-8")
    (out_dir / "risk_register.json").write_text(json.dumps(risk, indent=2) + "\n", encoding="utf-8")
    (out_dir / "modules_raw.json").write_text(
        json.dumps({"modules": modules, "phase2a_vs_comp": p2_extra, "summary": summary}, indent=2) + "\n",
        encoding="utf-8",
    )
    write_module_md(
        out_dir / "module_map_proposal.md",
        modules,
        {"phase2a_vs_comp": p2_extra, "base_sha": "5b1068d"},
    )
    print(json.dumps({"out": str(out_dir), "summary": summary, "n_clusters": len(cluster_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
