"""CONSOLIDATE-01A: vulture unused-function candidates + dup-name + annulus inventory."""
from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src_py"
DEV = ROOT / "dev"
OUT = Path(__file__).resolve().parent
reach = json.loads((OUT / "reachability.json").read_text(encoding="utf-8"))
reachable = {k for k, v in reach["modules"].items() if v["reachable"]}


def parse_vulture() -> list[dict]:
    proc = subprocess.run(
        [sys.executable, "-m", "vulture", str(SRC), "--min-confidence", "60"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    rows = []
    # path:line: unused function 'name' (60% confidence)
    pat = re.compile(
        r"^(?P<path>.+):(?P<line>\d+): unused (?P<kind>function|method|property) '(?P<name>[^']+)' \((?P<conf>\d+)% confidence\)"
    )
    for line in (proc.stdout or "").splitlines():
        m = pat.search(line.replace("\\", "/"))
        if not m:
            continue
        p = Path(m.group("path"))
        stem = p.stem
        rows.append(
            {
                "path": str(p).replace("\\", "/"),
                "module": stem,
                "line": int(m.group("line")),
                "kind": m.group("kind"),
                "name": m.group("name"),
                "confidence": int(m.group("conf")),
                "reachable_module": stem in reachable,
                "raw": line.strip(),
            }
        )
    (OUT / "vulture_raw.txt").write_text(proc.stdout or "", encoding="utf-8")
    return rows


IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def grep_name(name: str) -> list[str]:
    hits: list[str] = []
    quoted = re.compile(r"['\"]" + re.escape(name) + r"['\"]")
    getattr_re = re.compile(r"getattr\([^,]+,\s*['\"]" + re.escape(name) + r"['\"]")
    ident_re = re.compile(r"\b" + re.escape(name) + r"\b")
    for folder in (SRC, DEV):
        for p in folder.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in {".py", ".json", ".md", ".txt"}:
                continue
            if "__pycache__" in p.parts or "session_20260831_c01a" in p.parts:
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            rel = str(p.relative_to(ROOT)).replace("\\", "/")
            for i, line in enumerate(text.splitlines(), 1):
                if ident_re.search(line) or quoted.search(line) or getattr_re.search(line):
                    hits.append(f"{rel}:{i}:{line.strip()[:160]}")
                    if len(hits) >= 40:
                        return hits
    return hits


def function_defs() -> dict[str, list[dict]]:
    by_name: dict[str, list[dict]] = defaultdict(list)
    for p in sorted(SRC.glob("*.py")):
        try:
            tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"), filename=str(p))
        except SyntaxError:
            continue
        src_lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            # include nested? spec says helper names - top-level + methods
            start = node.lineno
            end = getattr(node, "end_lineno", start) or start
            body = "\n".join(src_lines[start - 1 : end])
            digest = hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]
            by_name[node.name].append(
                {
                    "file": f"src_py/{p.name}",
                    "line": start,
                    "end": end,
                    "sha": digest,
                    "nbytes": len(body.encode("utf-8")),
                }
            )
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        continue
                    if item.name.startswith("__") and item.name.endswith("__"):
                        continue
                    start = item.lineno
                    end = getattr(item, "end_lineno", start) or start
                    body = "\n".join(src_lines[start - 1 : end])
                    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]
                    by_name[item.name].append(
                        {
                            "file": f"src_py/{p.name}",
                            "line": start,
                            "end": end,
                            "sha": digest,
                            "nbytes": len(body.encode("utf-8")),
                            "class": node.name,
                        }
                    )
    return by_name


def annulus_sites() -> list[dict]:
    sites = []
    pat = re.compile(r"CircularAnnulus|CircularAperture|annulus_inner|annulus_outer|r_in\s*=|r_out\s*=")
    for p in sorted(SRC.glob("*.py")):
        text = p.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        for i, line in enumerate(lines, 1):
            if "CircularAnnulus" in line or (
                "CircularAperture" in line and "import" not in line
            ):
                window = "\n".join(lines[max(0, i - 4) : min(len(lines), i + 8)])
                sites.append(
                    {
                        "file": f"src_py/{p.name}",
                        "line": i,
                        "code": line.strip()[:200],
                        "window": window[:600],
                    }
                )
    return sites


def main() -> int:
    vult = parse_vulture()
    (OUT / "vulture_functions.json").write_text(
        json.dumps(vult, indent=2) + "\n", encoding="utf-8"
    )
    print("vulture unused function/method/property:", len(vult))
    print("in reachable modules:", sum(1 for r in vult if r["reachable_module"]))

    defs = function_defs()
    dups = []
    for name, locs in sorted(defs.items()):
        files = {x["file"] for x in locs}
        if len(files) < 2:
            continue
        shas = {x["sha"] for x in locs}
        dups.append(
            {
                "name": name,
                "n_defs": len(locs),
                "n_files": len(files),
                "byte_equal": len(shas) == 1,
                "locs": locs,
            }
        )
    (OUT / "dup_helpers.json").write_text(
        json.dumps({"n": len(dups), "items": dups}, indent=2) + "\n", encoding="utf-8"
    )
    print("dup helper names (2+ files):", len(dups))

    ann = annulus_sites()
    (OUT / "annulus_sites.json").write_text(
        json.dumps({"n": len(ann), "sites": ann}, indent=2) + "\n", encoding="utf-8"
    )
    print("annulus/aperture construction sites:", len(ann))
    return 0


if __name__ == "__main__":
    sys.exit(main())
