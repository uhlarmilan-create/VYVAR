#!/usr/bin/env python3
"""A/B compare verify_mag_limit=16 vs 14 (accuracy, margin, timing)."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _run_harness(
    *,
    mag: float,
    out_csv: Path,
    fields: str | None,
    mode: str,
) -> dict:
    cmd = [
        sys.executable,
        str(_ROOT / "scripts" / "blind_solve_rate.py"),
        "--mode",
        mode,
        "--verify-mag-limit",
        str(mag),
        "--out-csv",
        str(out_csv),
    ]
    if fields:
        cmd.extend(["--fields", fields])
    print("RUN:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(_ROOT), capture_output=True, text=True)
    print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    summary_path = out_csv.parent / "blind_solve_rate_summary.json"
    if summary_path.is_file():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    return {}


def _load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _num(v, default=0.0):
    try:
        if v in ("", None):
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _compare(rows16: list[dict], rows14: list[dict]) -> list[dict]:
    by16 = {r["id"]: r for r in rows16}
    by14 = {r["id"]: r for r in rows14}
    out = []
    for fid in sorted(set(by16) | set(by14)):
        a, b = by16.get(fid, {}), by14.get(fid, {})
        out.append(
            {
                "id": fid,
                "status_16": a.get("status"),
                "status_14": b.get("status"),
                "sep_16": a.get("sep_deg"),
                "sep_14": b.get("sep_deg"),
                "sep_delta": round(abs(_num(a.get("sep_deg")) - _num(b.get("sep_deg"))), 4)
                if a.get("sep_deg") not in ("", None) and b.get("sep_deg") not in ("", None)
                else "",
                "n_matched_16": a.get("n_matched"),
                "n_matched_14": b.get("n_matched"),
                "truth_near_n_16": a.get("truth_near_n_matched"),
                "truth_near_n_14": b.get("truth_near_n_matched"),
                "max_false_16": a.get("max_false_n_matched"),
                "max_false_14": b.get("max_false_n_matched"),
                "catalog_load_s_16": a.get("catalog_load_s"),
                "catalog_load_s_14": b.get("catalog_load_s"),
                "verify_s_16": a.get("verify_s"),
                "verify_s_14": b.get("verify_s"),
                "total_s_16": a.get("total_s"),
                "total_s_14": b.get("total_s"),
                "early_exit_16": a.get("early_exit_fired"),
                "early_exit_14": b.get("early_exit_fired"),
                "cone_n_cat_16": a.get("cone_n_cat"),
                "cone_n_cat_14": b.get("cone_n_cat"),
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="auto")
    ap.add_argument(
        "--fields",
        default=None,
        help="Comma-separated field ids (default: full battery).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=_ROOT / "validation" / "mag_ab",
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv16 = args.out_dir / f"blind_solve_mag16_{ts}.csv"
    csv14 = args.out_dir / f"blind_solve_mag14_{ts}.csv"
    cmp_csv = args.out_dir / f"mag_ab_compare_{ts}.csv"

    s16 = _run_harness(mag=16.0, out_csv=csv16, fields=args.fields, mode=args.mode)
    s14 = _run_harness(mag=14.0, out_csv=csv14, fields=args.fields, mode=args.mode)
    cmp_rows = _compare(_load_rows(csv16), _load_rows(csv14))
    if cmp_rows:
        keys = list(cmp_rows[0].keys())
        with cmp_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(cmp_rows)

    def _sum_s(rows, key):
        return round(sum(_num(r.get(key)) for r in rows), 2)

    report = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "fields": args.fields or "all",
        "summary_16": s16,
        "summary_14": s14,
        "total_s_sum_16": _sum_s(cmp_rows, "total_s_16"),
        "total_s_sum_14": _sum_s(cmp_rows, "total_s_14"),
        "csv_16": str(csv16),
        "csv_14": str(csv14),
        "compare_csv": str(cmp_csv),
        "compare_rows": cmp_rows,
    }
    report_path = args.out_dir / f"mag_ab_report_{ts}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
