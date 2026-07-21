#!/usr/bin/env python3
"""Summarize py-spy speedscope JSON: top-N functions by self and total time."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def _frame_key(frame: dict) -> tuple[str, str, str]:
    name = frame.get("name") or "<unknown>"
    file = frame.get("file") or ""
    line = str(frame.get("line") or "")
    # Strip speedscope path prefix to module-ish name
    mod = file.replace("\\", "/")
    if "/src_py/" in mod:
        mod = mod.split("/src_py/")[-1]
    elif mod.endswith(".py"):
        mod = Path(mod).name
    else:
        mod = Path(file).name if file else ""
    return name, mod, line


def summarize(path: Path, top_n: int = 30) -> tuple[list[tuple], list[tuple]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    profiles = data.get("profiles") or []
    if not profiles:
        raise SystemExit(f"No profiles in {path}")

    profile = profiles[0]
    samples = profile.get("samples") or []
    weights = profile.get("weights") or []
    frames = (data.get("shared") or {}).get("frames") or profile.get("frames") or []

    self_us: dict[tuple[str, str, str], float] = defaultdict(float)
    total_us: dict[tuple[str, str, str], float] = defaultdict(float)

    for i, stack in enumerate(samples):
        weight = float(weights[i] if i < len(weights) else 1.0)
        if not stack:
            continue
        leaf = stack[-1]
        if 0 <= leaf < len(frames):
            key = _frame_key(frames[leaf])
            self_us[key] += weight
        for idx in stack:
            if 0 <= idx < len(frames):
                total_us[_frame_key(frames[idx])] += weight

    total_self = sum(self_us.values()) or 1.0
    total_all = sum(total_us.values()) or 1.0

    by_self = sorted(self_us.items(), key=lambda x: -x[1])[:top_n]
    by_total = sorted(total_us.items(), key=lambda x: -x[1])[:top_n]

    self_rows = [
        (k[0], k[1], 100.0 * v / total_self, 100.0 * total_us.get(k, 0) / total_all)
        for k, v in by_self
    ]
    total_rows = [
        (k[0], k[1], 100.0 * self_us.get(k, 0) / total_self, 100.0 * v / total_all)
        for k, v in by_total
    ]
    return self_rows, total_rows


def _print_table(title: str, rows: list[tuple], file: sys.stdout | None = None) -> None:
    out = file or sys.stdout
    print(f"\n{title}", file=out)
    print(f"{'function':<40} {'module':<28} {'self%':>8} {'total%':>8}", file=out)
    print("-" * 88, file=out)
    for func, mod, self_pct, total_pct in rows:
        print(f"{func[:40]:<40} {mod[:28]:<28} {self_pct:7.2f} {total_pct:7.2f}", file=out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("speedscope_json", type=Path)
    ap.add_argument("-n", "--top", type=int, default=30)
    ap.add_argument("-o", "--output", type=Path, help="Write ASCII table to file")
    args = ap.parse_args()

    self_rows, total_rows = summarize(args.speedscope_json, args.top)
    if args.output:
        import io

        buf = io.StringIO()
        _print_table(f"TOP {args.top} BY SELF TIME", self_rows, buf)
        _print_table(f"TOP {args.top} BY TOTAL TIME", total_rows, buf)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(buf.getvalue(), encoding="ascii", errors="replace")
        print(f"Wrote {args.output}")
    else:
        _print_table(f"TOP {args.top} BY SELF TIME", self_rows)
        _print_table(f"TOP {args.top} BY TOTAL TIME", total_rows)


if __name__ == "__main__":
    main()
