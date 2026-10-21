#!/usr/bin/env python3
# -*- coding: ascii -*-
"""Weekly VYVAR invariants report (ENCODING-POLICY: ASCII; stdlib + repo only).

Sections: registry, cheap guards, ledger, runtime WARN sweep, P1 pointer.
Exit 0 unless a guard subprocess fails (then 1).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY = REPO_ROOT / "docs" / "VYVAR_INVARIANTS.md"
LEDGER = REPO_ROOT / "dev" / "validation" / "VYVAR_VALIDATION_LEDGER.json"
RESULTS = REPO_ROOT / "dev" / "results"

GUARD_TESTS = (
    "dev/tests/test_docs_sync_guard.py",
    "dev/tests/test_ascii_policy.py",
    "dev/tests/test_invariants_p2.py",
)

P1_CMD = (
    "set VYVAR_INVARIANTS_P1=1 && "
    "pytest dev/tests/test_invariants_p1_seed.py "
    "dev/tests/test_invariants_p1_golden.py -q"
)


def _parse_registry(text: str) -> tuple[Counter[str], Counter[str], list[str]]:
    """Return (by_policy, by_enforcement, wired_ids)."""
    by_policy: Counter[str] = Counter()
    by_enf: Counter[str] = Counter()
    wired: list[str] = []
    # Table rows: | INV-XXX **[wired]** | ... | Policy | ...
    row_re = re.compile(
        r"^\|\s*(INV-[A-Z0-9-]+)(?:\s+\*\*\[wired\]\*\*)?\s*\|"
        r"[^|]*\|([^|]*)\|([^|]*)\|",
        re.MULTILINE,
    )
    for m in row_re.finditer(text):
        inv_id = m.group(1).strip()
        enforced = m.group(2).strip().lower()
        policy_cell = m.group(3).strip().upper()
        is_wired = "**[wired]**" in m.group(0) or "[wired]" in m.group(0)
        if "FAIL" in policy_cell and "WARN" in policy_cell:
            pol = "FAIL+WARN"
        elif "FAIL" in policy_cell:
            pol = "FAIL"
        elif "WARN" in policy_cell:
            pol = "WARN"
        else:
            pol = "OTHER"
        by_policy[pol] += 1
        if is_wired:
            by_enf["wired"] += 1
            wired.append(inv_id)
        else:
            by_enf["registry-only"] += 1
    return by_policy, by_enf, wired


def _run_guard(rel: str) -> tuple[str, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", rel],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    detail = (proc.stdout or "").strip().splitlines()
    tail = detail[-1] if detail else f"exit {proc.returncode}"
    status = "PASS" if proc.returncode == 0 else "FAIL"
    return status, tail


def _ledger_summary() -> list[str]:
    lines: list[str] = []
    if not LEDGER.is_file():
        return ["ledger missing"]
    data = json.loads(LEDGER.read_text(encoding="utf-8"))
    items = data.get("items") or []
    active = [
        it
        for it in items
        if it.get("passes") is True
        and str(it.get("status") or "").lower() not in {"superseded_offline", "todo"}
    ]
    lines.append(f"ledger version={data.get('version')} updated={data.get('updated')}")
    lines.append(f"items total={len(items)} active_passes_true~={len(active)}")
    # last auto-stamp: max last_verified among items with commit
    stamped = [
        it
        for it in items
        if it.get("last_verified") and it.get("commit")
    ]
    if stamped:
        stamped.sort(key=lambda it: str(it.get("last_verified")), reverse=True)
        top = stamped[0]
        lines.append(
            f"latest stamp: {top.get('id')} last_verified={top.get('last_verified')} "
            f"commit={top.get('commit')}"
        )
    active_ids = sorted(it["id"] for it in active if it.get("id"))
    lines.append("ACTIVE (passes=true, not superseded): " + ", ".join(active_ids))
    return lines


def _archive_root() -> Path | None:
    try:
        sys.path.insert(0, str(REPO_ROOT / "src_py"))
        from config import AppConfig  # noqa: PLC0415

        root = Path(AppConfig().archive_root)
        if root.is_dir():
            return root
    except Exception:  # noqa: BLE001
        pass
    fallback = REPO_ROOT / "Archive"
    return fallback if fallback.is_dir() else None


def _runtime_warn_sweep(archive: Path | None) -> list[str]:
    if archive is None:
        return ["no drafts reachable (Archive absent or unreadable)"]
    metas = list(archive.glob("Drafts/*/platesolve/*/photometry/pipeline_meta.json"))
    metas += list(archive.glob("Drafts/*/platesolve/*/pipeline_meta.json"))
    # de-dupe
    metas = sorted({p.resolve() for p in metas})
    if not metas:
        return [f"no pipeline_meta.json under {archive.as_posix()} Drafts"]
    by_id: Counter[str] = Counter()
    n_warn = n_fail = 0
    n_files = 0
    for mp in metas:
        try:
            data = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        block = data.get("invariants")
        if not isinstance(block, list):
            continue
        n_files += 1
        for rec in block:
            if not isinstance(rec, dict):
                continue
            inv_id = str(rec.get("id") or rec.get("inv_id") or "?")
            st = str(rec.get("status") or rec.get("policy") or "").upper()
            if st == "WARN":
                n_warn += 1
                by_id[f"{inv_id}:WARN"] += 1
            elif st == "FAIL":
                n_fail += 1
                by_id[f"{inv_id}:FAIL"] += 1
    lines = [
        f"scanned_meta_files={n_files} (of {len(metas)} found)",
        f"WARN events={n_warn} FAIL events={n_fail}",
    ]
    if by_id:
        top = ", ".join(f"{k}={v}" for k, v in by_id.most_common(12))
        lines.append(f"by_id: {top}")
    else:
        lines.append("by_id: (none)")
    return lines


def build_report() -> tuple[str, int]:
    """Return (markdown_body, exit_code)."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = [
        "# VYVAR weekly invariants report",
        "",
        f"Generated: {now}",
        "",
        "## Registry (`docs/VYVAR_INVARIANTS.md`)",
        "",
    ]
    exit_code = 0
    if not REGISTRY.is_file():
        lines.append("registry missing")
    else:
        text = REGISTRY.read_text(encoding="utf-8")
        by_pol, by_enf, wired = _parse_registry(text)
        lines.append(
            f"- by policy: "
            + ", ".join(f"{k}={v}" for k, v in sorted(by_pol.items()))
        )
        lines.append(
            f"- by enforcement: "
            + ", ".join(f"{k}={v}" for k, v in sorted(by_enf.items()))
        )
        lines.append(f"- wired IDs ({len(wired)}): " + ", ".join(wired))

    lines.extend(["", "## Guards (cheap pytest)", ""])
    for rel in GUARD_TESTS:
        status, detail = _run_guard(rel)
        lines.append(f"- `{rel}`: **{status}** ({detail})")
        if status != "PASS":
            exit_code = 1

    lines.extend(["", "## Ledger", ""])
    for row in _ledger_summary():
        lines.append(f"- {row}")

    lines.extend(["", "## Runtime WARN/FAIL sweep (Archive drafts)", ""])
    for row in _runtime_warn_sweep(_archive_root()):
        lines.append(f"- {row}")

    lines.extend(
        [
            "",
            "## P1 golden pointer (opt-in; do not auto-run)",
            "",
            f"```",
            P1_CMD,
            "```",
            "",
            "P1 is ~10 min; run when locking a golden or after P1-touching changes.",
            "",
        ]
    )
    return "\n".join(lines) + "\n", exit_code


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--write",
        action="store_true",
        help="Also write dev/results/INVARIANTS_WEEKLY_<yyyymmdd>.md",
    )
    args = ap.parse_args(argv)
    body, code = build_report()
    sys.stdout.write(body)
    if args.write:
        RESULTS.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
        out = RESULTS / f"INVARIANTS_WEEKLY_{stamp}.md"
        out.write_text(body, encoding="ascii")
        sys.stdout.write(f"\nWrote {out.as_posix()}\n")
    return code


if __name__ == "__main__":
    sys.exit(main())
