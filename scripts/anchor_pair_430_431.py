#!/usr/bin/env python3
"""VALIDATE-429 Part D: headless anchor pair (draft_430 + draft_431) from D:\\BO_CVn."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests.photometry_sha import compare_photometry_science_meaningful, compute_photometry_sha

SETUP = "NoFilter_60_2"
DEFAULT_SOURCE = Path(r"D:\BO_CVn")
DRAFT_IDS = (430, 431)
SNAPSHOT_NAME = "draft_000430_snapshot_wcsinv_20260716"
OUT_DIR = _ROOT / "tmp" / "anchor_pair_430_431"

_IDENTITY_QA_KEYS = (
    "matched_world2pix_identity_n",
    "matched_world2pix_identity_p50_px",
    "matched_world2pix_identity_p95_px",
    "matched_world2pix_identity_p99_px",
    "matched_world2pix_identity_max_px",
)


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _run_night(source: Path, *, log_path: Path) -> dict[str, Any]:
    from night_run import NightRunParams, run_night_pipeline

    params = NightRunParams(
        source_dir=source,
        equipment_id=1,
        telescope_id=1,
        config_path=None,
        sysrem_enabled=True,
        sysrem_n_iter=3,
        dry_run=False,
        progress_cb=lambda msg: print(msg, flush=True),
    )
    t0 = time.time()
    result = run_night_pipeline(params)
    elapsed = time.time() - t0
    payload = {
        "success": bool(result.success),
        "draft_id": result.draft_id,
        "draft_dir": str(result.draft_dir) if result.draft_dir else None,
        "n_lightcurves": result.n_lightcurves,
        "n_frames": result.n_frames,
        "errors": list(result.errors),
        "warnings": list(result.warnings),
        "elapsed_s": elapsed,
        "phase_timings": dict(result.phase_timings),
        "log_path": str(log_path),
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


def _provenance_gate(draft_id: int) -> dict[str, Any]:
    root = _draft_root(draft_id)
    if not root.is_dir():
        return {
            "draft_id": draft_id,
            "missing": True,
            "provenance_clean": False,
        }
    meta = _read_pipeline_meta(draft_id)
    prov = meta.get("provenance") or {}
    git_dirty = prov.get("git_dirty")
    git_dirty_code = prov.get("git_dirty_code")
    clean = git_dirty_code is False if git_dirty_code is not None else git_dirty is False
    return {
        "draft_id": draft_id,
        "git_hash": prov.get("git_hash"),
        "git_dirty": git_dirty,
        "git_dirty_code": git_dirty_code,
        "provenance_clean": clean,
        "entry_point": prov.get("entry_point"),
        "identity_qa": {k: meta.get(k) for k in _IDENTITY_QA_KEYS if k in meta},
        "wcs_roundtrip_pass": meta.get("wcs_roundtrip_pass"),
        "wcs_roundtrip_p99_px": meta.get("wcs_roundtrip_p99_px"),
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
    dest = archive / SNAPSHOT_NAME
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    core_sha, core_n = compute_photometry_sha(dest, include_comp_qa=False)
    ext_sha, ext_n = compute_photometry_sha(dest, include_comp_qa=True)
    meta_path = dest / "platesolve" / SETUP / "photometry" / "pipeline_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    return {
        "snapshot_dir": str(dest),
        "core_sha": core_sha,
        "core_n": core_n,
        "extended_sha": ext_sha,
        "extended_n": ext_n,
        "git_head": _git_head(),
        "pipeline_meta_provenance": (meta.get("provenance") or {}),
        "identity_qa": {k: meta.get(k) for k in _IDENTITY_QA_KEYS if k in meta},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Anchor pair 430/431 headless runs")
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--skip-runs", action="store_true", help="Only compare/cut if drafts exist")
    ap.add_argument("--cut-snapshot", action="store_true", help="Cut anchor snapshot from draft_430")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": _git_head(),
        "source": str(args.source),
        "setup": SETUP,
        "runs": [],
        "provenance_gates": [],
    }

    if not args.skip_runs:
        if not args.source.is_dir():
            print(f"ERROR: missing source {args.source}", file=sys.stderr)
            return 1
        for draft_id in DRAFT_IDS:
            print(f"=== Night run -> draft_{draft_id:06d} ===", flush=True)
            log_path = OUT_DIR / f"night_run_{draft_id}.json"
            payload = _run_night(args.source, log_path=log_path)
            report["runs"].append(payload)
            if not payload["success"]:
                print(f"ERROR: draft {draft_id} failed", file=sys.stderr)
                break

    for draft_id in DRAFT_IDS:
        gate = _provenance_gate(draft_id)
        report["provenance_gates"].append(gate)
        print(json.dumps({"provenance_gate": gate}, indent=2))

    prov_ok = all(g.get("provenance_clean") for g in report["provenance_gates"])
    report["provenance_all_clean"] = prov_ok
    if not prov_ok:
        dirty = [g for g in report["provenance_gates"] if not g.get("provenance_clean")]
        print(
            "STOP: pipeline_meta git_dirty_code must be false on BOTH 430 and 431 — no snapshot cut",
            file=sys.stderr,
        )
        print(f"Dirty drafts: {dirty}", file=sys.stderr)
        out_path = OUT_DIR / "anchor_pair_report.json"
        out_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
        return 3

    cmp = _compare_pair(DRAFT_IDS[0], DRAFT_IDS[1])
    report["pair_compare"] = cmp
    print(json.dumps(cmp, indent=2, default=str))

    if not cmp["byte_identical_core"]:
        print("STOP: 430 != 431 on core SHA — no anchor cut", file=sys.stderr)
        out_path = OUT_DIR / "anchor_pair_report.json"
        out_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
        return 2

    if args.cut_snapshot:
        report["snapshot"] = _cut_snapshot(DRAFT_IDS[0])

    out_path = OUT_DIR / "anchor_pair_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
