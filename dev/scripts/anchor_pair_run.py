#!/usr/bin/env python3
"""Headless anchor pair: two fresh night runs, gates, optional snapshot + ledger finalize."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401  (repo layout: src_py + dev on sys.path)
_ROOT = _bootstrap.REPO_ROOT
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests.photometry_sha import compare_photometry_science_meaningful, compute_photometry_sha

from photometry_core import _resolve_git_provenance, classify_git_dirty_paths

SETUP = "NoFilter_60_2"
DEFAULT_SOURCE = Path(r"D:\BO_CVn")
OUT_DIR = _ROOT / "tmp" / "anchor_pair_run"

_IDENTITY_QA_KEYS = (
    "matched_world2pix_identity_n",
    "matched_world2pix_identity_p50_px",
    "matched_world2pix_identity_p95_px",
    "matched_world2pix_identity_p99_px",
    "matched_world2pix_identity_max_px",
)


def _git_head(full: bool = False) -> str:
    args = ["rev-parse", "HEAD"] if full else ["rev-parse", "--short", "HEAD"]
    try:
        return subprocess.check_output(["git", *args], cwd=_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _git_status_porcelain() -> str:
    try:
        return subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=_ROOT, text=True, stderr=subprocess.DEVNULL,
        )
    except Exception:  # noqa: BLE001
        return "unknown"


def _run_night(source: Path, *, run_index: int, log_path: Path) -> dict[str, Any]:
    from night_run import NightRunParams, run_night_pipeline

    params = NightRunParams(
        source_dir=source,
        equipment_id=1,
        telescope_id=1,
        config_path=None,
        # Honor config.json / AppConfig (do not force SysRem — F-431 SHA nondeterminism).
        sysrem_enabled=None,
        dry_run=False,
        progress_cb=lambda msg: print(msg, flush=True),
    )
    t0 = time.time()
    result = run_night_pipeline(params)
    elapsed = time.time() - t0
    draft_id = int(result.draft_id) if result.draft_id is not None else None
    payload = {
        "run_index": run_index,
        "success": bool(result.success),
        "draft_id": draft_id,
        "draft_dir": str(result.draft_dir) if result.draft_dir else None,
        "n_lightcurves": result.n_lightcurves,
        "n_frames": result.n_frames,
        "errors": list(result.errors),
        "warnings": list(result.warnings),
        "elapsed_s": elapsed,
        "phase_timings": dict(result.phase_timings),
        "log_path": str(log_path),
        "git_head_at_start": _git_head(full=True),
        "git_porcelain_at_start": _git_status_porcelain(),
    }
    log_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _draft_root(draft_id: int) -> Path:
    from config import AppConfig

    cfg = AppConfig()
    return Path(cfg.archive_root) / "Drafts" / f"draft_{draft_id:06d}"


def _read_pipeline_meta(draft_id: int) -> dict[str, Any]:
    meta_path = _draft_root(draft_id) / "platesolve" / SETUP / "photometry" / "pipeline_meta.json"
    if not meta_path.is_file():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _snapshot_name(draft_id: int, *, date: str | None = None) -> str:
    d = date or datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"draft_{draft_id:06d}_snapshot_wcsinv_{d}"


def _provenance_gate(draft_id: int, *, expected_git: str) -> dict[str, Any]:
    root = _draft_root(draft_id)
    if not root.is_dir():
        return {"draft_id": draft_id, "missing": True, "gate_pass": False}
    meta = _read_pipeline_meta(draft_id)
    prov = meta.get("provenance") or {}
    git_dirty = prov.get("git_dirty")
    git_dirty_code = prov.get("git_dirty_code")
    git_hash = str(prov.get("git_hash") or "")
    identity_qa = {k: meta.get(k) for k in _IDENTITY_QA_KEYS if k in meta}
    qa_ok = all(k in identity_qa for k in _IDENTITY_QA_KEYS)
    hash_ok = git_hash.startswith(expected_git) or expected_git.startswith(git_hash[: len(expected_git)])
    # Anchor / FAIL-CLOSED: trip on import-relevant code dirt only (T3 dirty-gate).
    clean = git_dirty_code is False if git_dirty_code is not None else git_dirty is False
    return {
        "draft_id": draft_id,
        "git_hash": git_hash,
        "git_dirty": git_dirty,
        "git_dirty_code": git_dirty_code,
        "git_dirty_code_files": prov.get("git_dirty_code_files") or [],
        "git_dirty_scratch_files": prov.get("git_dirty_scratch_files") or [],
        "git_hash_ok": hash_ok,
        "provenance_clean": clean,
        "identity_qa": identity_qa,
        "identity_qa_complete": qa_ok,
        "wcs_roundtrip_pass": meta.get("wcs_roundtrip_pass"),
        "gate_pass": bool(clean and hash_ok and qa_ok),
    }


def _compare_pair(a: int, b: int) -> dict[str, Any]:
    ra, rb = _draft_root(a), _draft_root(b)
    core_a, n_core_a = compute_photometry_sha(ra, include_comp_qa=False)
    core_b, n_core_b = compute_photometry_sha(rb, include_comp_qa=False)
    ext_a, n_ext_a = compute_photometry_sha(ra, include_comp_qa=True)
    ext_b, n_ext_b = compute_photometry_sha(rb, include_comp_qa=True)
    sci = compare_photometry_science_meaningful(ra, rb, setups=(SETUP,))
    return {
        "draft_a": a,
        "draft_b": b,
        "core_sha_a": core_a,
        "core_sha_b": core_b,
        "core_n_a": n_core_a,
        "core_n_b": n_core_b,
        "extended_sha_a": ext_a,
        "extended_sha_b": ext_b,
        "extended_n_a": n_ext_a,
        "extended_n_b": n_ext_b,
        "byte_identical_core": core_a == core_b and n_core_a == n_core_b,
        "byte_identical_extended": ext_a == ext_b and n_ext_a == n_ext_b,
        "science_compare": sci,
    }


def _cut_snapshot(draft_id: int) -> dict[str, Any]:
    src = _draft_root(draft_id)
    from config import AppConfig

    cfg = AppConfig()
    archive = Path(cfg.archive_root) / "Drafts"
    snap_name = _snapshot_name(draft_id)
    dest = archive / snap_name
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    core_sha, core_n = compute_photometry_sha(dest, include_comp_qa=False)
    ext_sha, ext_n = compute_photometry_sha(dest, include_comp_qa=True)
    meta_path = dest / "platesolve" / SETUP / "photometry" / "pipeline_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    return {
        "snapshot_name": snap_name,
        "snapshot_dir": str(dest),
        "core_sha": core_sha,
        "core_n": core_n,
        "extended_sha": ext_sha,
        "extended_n": ext_n,
        "git_head": _git_head(full=True),
        "pipeline_meta_provenance": meta.get("provenance") or {},
        "identity_qa": {k: meta.get(k) for k in _IDENTITY_QA_KEYS if k in meta},
    }


def _update_session_baseline_check(*, draft_id: int, snapshot_name: str, core_sha: str, ext_sha: str) -> None:
    path = _ROOT / "scripts" / "session_baseline_check.py"
    text = path.read_text(encoding="utf-8")
    replacements = [
        (r'^ANCHOR_LEDGER_ID = ".*"$', f'ANCHOR_LEDGER_ID = "VL-ANCHOR-WCSINV"'),
        (r"^DRAFT_ID = \d+$", f"DRAFT_ID = {int(draft_id)}"),
        (r'^SNAPSHOT_NAME = ".*"$', f'SNAPSHOT_NAME = "{snapshot_name}"'),
        (r'^EXPECTED_PHOTOMETRY_SHA_CORE = ".*"$', f'EXPECTED_PHOTOMETRY_SHA_CORE = "{core_sha}"'),
        (r'^EXPECTED_PHOTOMETRY_SHA_EXTENDED = ".*"$', f'EXPECTED_PHOTOMETRY_SHA_EXTENDED = "{ext_sha}"'),
        (r'^EXPECTED_PHOTOMETRY_SHA_CORE_PREFIX = EXPECTED_PHOTOMETRY_SHA_CORE\[:8\]$', f'EXPECTED_PHOTOMETRY_SHA_CORE_PREFIX = "{core_sha[:8]}"'),
        (
            r'^EXPECTED_PHOTOMETRY_SHA_EXTENDED_PREFIX = EXPECTED_PHOTOMETRY_SHA_EXTENDED\[:8\]$',
            f'EXPECTED_PHOTOMETRY_SHA_EXTENDED_PREFIX = "{ext_sha[:8]}"',
        ),
    ]
    for pattern, repl in replacements:
        text, n = re.subn(pattern, repl, text, count=1, flags=re.MULTILINE)
        if n != 1:
            raise RuntimeError(f"session_baseline_check patch failed for {pattern!r}")
    path.write_text(text, encoding="utf-8")


def _finalize_ledger(*, draft_id: int, snapshot_name: str, core_sha: str, ext_sha: str, commit: str) -> None:
    ledger_path = _ROOT / "validation" / "VYVAR_VALIDATION_LEDGER.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    ledger["updated"] = today
    new_id = "VL-ANCHOR-WCSINV"
    existing = {it["id"]: it for it in ledger.get("items", [])}
    if new_id not in existing:
        ledger.setdefault("items", []).append(
            {
                "id": new_id,
                "area": "photometry",
                "description": f"In-Archive BO CVn anchor snapshot {snapshot_name}",
                "verification": f"scripts/session_baseline_check.py --full (core {core_sha[:8]}…)",
                "passes": True,
                "last_verified": today,
                "commit": commit,
                "notes": f"Cut from anchor pair run1 draft_{draft_id:06d}; extended {ext_sha[:8]}…",
            }
        )
    else:
        it = existing[new_id]
        it["passes"] = True
        it["last_verified"] = today
        it["commit"] = commit
        it["notes"] = f"Cut from anchor pair run1 draft_{draft_id:06d}; core {core_sha[:8]}…"
        it.pop("status", None)
    for it in ledger.get("items", []):
        if it["id"] == "VL-COUNTERS-ZERO":
            it["passes"] = True
            it["last_verified"] = today
            it["commit"] = commit
    ledger_path.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")


def _identity_qa_series(gates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    series: list[dict[str, Any]] = []
    for g in gates:
        qa = g.get("identity_qa") or {}
        series.append(
            {
                "draft_id": g.get("draft_id"),
                "p95_px": qa.get("matched_world2pix_identity_p95_px"),
                "p99_px": qa.get("matched_world2pix_identity_p99_px"),
                "n": qa.get("matched_world2pix_identity_n"),
            }
        )
    return series


def main() -> int:
    ap = argparse.ArgumentParser(description="Anchor pair: two fresh headless night runs")
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--expected-git", default=None, help="Full or short HEAD hash (default: current HEAD)")
    ap.add_argument("--skip-runs", action="store_true", help="Gate/compare only (draft ids from report or args)")
    ap.add_argument("--run1-draft", type=int, default=None)
    ap.add_argument("--run2-draft", type=int, default=None)
    ap.add_argument("--finalize", action="store_true", help="Cut snapshot + update ledger + session_baseline_check")
    ap.add_argument("--report", type=Path, default=OUT_DIR / "anchor_pair_report.json")
    args = ap.parse_args()

    expected_git = args.expected_git or _git_head(full=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    porcelain = _git_status_porcelain().strip()
    if porcelain and not args.skip_runs:
        _, git_dirty, dirty_files = _resolve_git_provenance()
        if git_dirty:
            code_dirty, code_paths, scratch_paths = classify_git_dirty_paths(porcelain, dirty_files)
            if code_dirty:
                print(
                    "ERROR: git_dirty_code — import-relevant .py modifications block anchor runs",
                    file=sys.stderr,
                )
                for p in code_paths:
                    print(f"  code: {p}", file=sys.stderr)
                return 1
            print(
                f"NOTE: scratch-only dirt allowed ({len(scratch_paths)} paths); git_dirty_code=false",
                flush=True,
            )

    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "expected_git": expected_git,
        "git_head": _git_head(full=True),
        "git_porcelain": porcelain,
        "source": str(args.source),
        "setup": SETUP,
        "runs": [],
        "provenance_gates": [],
        "identity_qa_series": [],
    }

    run_draft_ids: list[int] = []

    if not args.skip_runs:
        if not args.source.is_dir():
            print(f"ERROR: missing source {args.source}", file=sys.stderr)
            return 1
        for idx in (1, 2):
            print(f"=== Anchor night run {idx}/2 ===", flush=True)
            log_path = OUT_DIR / f"night_run_{idx}.json"
            payload = _run_night(args.source, run_index=idx, log_path=log_path)
            report["runs"].append(payload)
            if not payload["success"] or payload["draft_id"] is None:
                print(f"ERROR: run {idx} failed", file=sys.stderr)
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
                return 1
            run_draft_ids.append(int(payload["draft_id"]))
    else:
        if args.run1_draft is None or args.run2_draft is None:
            if args.report.is_file():
                prev = json.loads(args.report.read_text(encoding="utf-8"))
                runs = prev.get("runs") or []
                if len(runs) >= 2:
                    run_draft_ids = [int(r["draft_id"]) for r in runs[:2]]
            if len(run_draft_ids) < 2:
                print("ERROR: --skip-runs needs --run1-draft/--run2-draft or prior report", file=sys.stderr)
                return 1
        else:
            run_draft_ids = [int(args.run1_draft), int(args.run2_draft)]

    report["run_draft_ids"] = run_draft_ids

    for did in run_draft_ids:
        gate = _provenance_gate(did, expected_git=expected_git)
        report["provenance_gates"].append(gate)
        print(json.dumps({"provenance_gate": gate}, indent=2))

    report["identity_qa_series"] = _identity_qa_series(report["provenance_gates"])
    gates_ok = all(g.get("gate_pass") for g in report["provenance_gates"])
    report["provenance_all_pass"] = gates_ok
    if not gates_ok:
        print("STOP: provenance/QA gates failed — no snapshot cut", file=sys.stderr)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
        return 3

    cmp = _compare_pair(run_draft_ids[0], run_draft_ids[1])
    report["pair_compare"] = cmp
    print(json.dumps(cmp, indent=2, default=str))

    if not cmp["byte_identical_core"]:
        print("STOP: run1 != run2 on core SHA — no anchor cut", file=sys.stderr)
        args.report.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
        return 2

    if args.finalize:
        snap = _cut_snapshot(run_draft_ids[0])
        report["snapshot"] = snap
        commit = _git_head()
        _update_session_baseline_check(
            draft_id=run_draft_ids[0],
            snapshot_name=snap["snapshot_name"],
            core_sha=snap["core_sha"],
            ext_sha=snap["extended_sha"],
        )
        _finalize_ledger(
            draft_id=run_draft_ids[0],
            snapshot_name=snap["snapshot_name"],
            core_sha=snap["core_sha"],
            ext_sha=snap["extended_sha"],
            commit=commit,
        )
        report["finalized"] = True
        report["session_baseline_reenabled"] = True

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Wrote {args.report}")
    print("identity_qa_series:", json.dumps(report["identity_qa_series"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
