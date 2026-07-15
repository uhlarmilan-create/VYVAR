#!/usr/bin/env python3
"""Session-start baseline check (--fast default; --full for draft_424 anchor re-verify).

Exit 0 = PASS or SUSPENDED, 1 = FAIL. ASCII output; concise summary table at end.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = REPO_ROOT / "validation" / "VYVAR_VALIDATION_LEDGER.json"
ANCHOR_LEDGER_ID = "VL-ANCHOR-424"

DRAFT_ID = 424
SETUP = "NoFilter_60_2"
SNAPSHOT_NAME = "draft_000424_snapshot_sigma_floor_20260713"
# Content anchors (PROD-SIGMA-FLOOR re-anchor 2026-07-13; c4 + Newton floor; wide un-floored).
EXPECTED_PHOTOMETRY_SHA_CORE = "bf3743a150d788283eab2ab51db7b31f59e6d1c481159208bbe3f573092ec975"
EXPECTED_PHOTOMETRY_SHA_EXTENDED = "dec5c637724e0ca536e97a01194ab8cc06df9471ce4813fcfd26024b9e880fd1"
EXPECTED_PHOTOMETRY_SHA_CORE_PREFIX = EXPECTED_PHOTOMETRY_SHA_CORE[:8]
EXPECTED_PHOTOMETRY_SHA_EXTENDED_PREFIX = EXPECTED_PHOTOMETRY_SHA_EXTENDED[:8]

# Known untracked paths: WARN only (not FAIL). Extend when deliberately added.
KNOWN_UNTRACKED_PREFIXES = (
    ".worktrees/",
    "CURSOR_RESULT",
    "docs/VYVAR_CODE_AUDIT.md",
    "docs/round2_figs/",
    "scripts/dy_peg_night_run_bvr.py",
    "scripts/qatar8_night_run_v.py",
)


@dataclass
class CheckResult:
    name: str
    status: str  # PASS | FAIL | WARN | SKIP
    detail: str = ""


@dataclass
class SessionReport:
    tier: str
    results: list[CheckResult] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str = "") -> None:
        self.results.append(CheckResult(name, status, detail))

    @property
    def ok(self) -> bool:
        return not any(r.status == "FAIL" for r in self.results)

    @property
    def suspended(self) -> bool:
        return any(r.status == "SUSPENDED" for r in self.results)


def _run_git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _git_short_head() -> str:
    return _run_git("rev-parse", "--short", "HEAD") or "unknown"


def _is_known_untracked(path: str) -> bool:
    norm = path.replace("\\", "/")
    return any(norm.startswith(p) or norm == p.rstrip("/") for p in KNOWN_UNTRACKED_PREFIXES)


def check_git_state(report: SessionReport) -> None:
    branch = _run_git("rev-parse", "--abbrev-ref", "HEAD") or "unknown"
    head = _git_short_head()
    report.add("git-branch", "PASS", branch)
    report.add("git-head", "PASS", head)

    porcelain = _run_git("status", "--porcelain")
    if not porcelain:
        report.add("git-tree", "PASS", "clean")
        return

    staged_diff = _run_git("diff", "--cached", "--name-only")
    if staged_diff and staged_diff.strip():
        n_staged = len([ln for ln in staged_diff.splitlines() if ln.strip()])
        report.add("git-staged", "FAIL", f"{n_staged} staged change(s)")
    else:
        report.add("git-staged", "PASS", "none")

    untracked = [ln[3:].strip() for ln in porcelain.splitlines() if ln.startswith("??")]
    unknown = [p for p in untracked if not _is_known_untracked(p)]
    known = [p for p in untracked if _is_known_untracked(p)]
    if known:
        report.add("git-untracked-known", "WARN", f"{len(known)} known untracked")
    if unknown:
        report.add("git-untracked", "WARN", "; ".join(unknown[:5]) + ("..." if len(unknown) > 5 else ""))

    local = _run_git("rev-parse", "HEAD")
    remote = _run_git("rev-parse", "origin/main")
    if local and remote and local != remote:
        report.add("git-origin-main", "WARN", f"differs from origin/main ({remote[:7]}); consider git pull")


def check_config_paths(report: SessionReport) -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from config import AppConfig  # noqa: PLC0415

    cfg = AppConfig()
    checks = [
        ("database_path", Path(cfg.database_path)),
        ("gaia_db_path", Path(cfg.gaia_db_path) if cfg.gaia_db_path else None),
        ("archive_root", Path(cfg.archive_root)),
        ("calibration_library_root", Path(cfg.calibration_library_root)),
    ]
    missing: list[str] = []
    for label, path in checks:
        if path is None or not path.exists():
            missing.append(label)
    if missing:
        report.add("config-paths", "FAIL", "missing: " + ", ".join(missing))
    else:
        report.add("config-paths", "PASS", "all present")


def check_pytest(report: SessionReport) -> None:
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=3600,
        )
    except subprocess.TimeoutExpired:
        report.add("pytest", "FAIL", "timeout")
        return
    out = (proc.stdout or "") + (proc.stderr or "")
    m_pass = re.search(r"(\d+) passed", out)
    m_skip = re.search(r"(\d+) skipped", out)
    counts = []
    if m_pass:
        counts.append(f"{m_pass.group(1)} passed")
    if m_skip:
        counts.append(f"{m_skip.group(1)} skipped")
    detail = ", ".join(counts) if counts else out.strip()[-200:]
    if proc.returncode == 0:
        report.add("pytest", "PASS", detail)
    else:
        report.add("pytest", "FAIL", detail or f"exit {proc.returncode}")


def check_ledger_hint(report: SessionReport) -> None:
    if not LEDGER_PATH.is_file():
        report.add("ledger", "FAIL", f"missing {LEDGER_PATH.relative_to(REPO_ROOT)}")
        return
    ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    report.add("ledger", "PASS", f"v{ledger.get('version')} {len(ledger.get('items', []))} items")
    todo: list[str] = []
    for it in ledger.get("items", []):
        if not it.get("passes"):
            todo.append(it["id"])
        elif not it.get("last_verified"):
            todo.append(f"{it['id']}(never verified)")
    if todo:
        report.add("ledger-todo", "WARN", ", ".join(todo))


def _load_ledger() -> dict[str, Any]:
    return json.loads(LEDGER_PATH.read_text(encoding="utf-8"))


def _ledger_item(ledger_id: str) -> dict[str, Any] | None:
    for it in _load_ledger().get("items", []):
        if it.get("id") == ledger_id:
            return it
    return None


def _full_baseline_suspend_message() -> str | None:
    """Return suspend message when VL-ANCHOR-424 is offline; else None."""
    item = _ledger_item(ANCHOR_LEDGER_ID)
    if not item or item.get("status") != "suspended_offline":
        return None
    backup = item.get("offline_backup") or {}
    path = backup.get("path") or "unknown"
    return (
        "full baseline SUSPENDED pending new anchor "
        f"(Archive cleared 2026-07-15; golden reference offline at {path})"
    )


def provenance_block_hash(phot_dir: Path) -> str:
    meta_path = phot_dir / "pipeline_meta.json"
    if not meta_path.is_file():
        return ""
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    prov = dict(meta.get("provenance") or {})
    prov.pop("stamped_at_utc", None)
    blob = json.dumps(prov, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _update_ledger_on_full_pass(commit: str) -> None:
    ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    for it in ledger.get("items", []):
        if it["id"] in ("VL-ANCHOR-424", "VL-COUNTERS-ZERO"):
            it["passes"] = True
            it["last_verified"] = today
            it["commit"] = commit
    ledger["updated"] = today
    LEDGER_PATH.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")


def run_full_baseline(report: SessionReport) -> None:
    suspend_msg = _full_baseline_suspend_message()
    if suspend_msg:
        report.add("full-baseline", "SUSPENDED", suspend_msg)
        return

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from config import AppConfig  # noqa: PLC0415
    from database import VyvarDatabase  # noqa: PLC0415
    from except_fix_counters import get_except_fix_counters, reset_except_fix_counters  # noqa: PLC0415
    from photometry_core import run_full_photometry_pipeline  # noqa: PLC0415
    from tests.photometry_sha import (  # noqa: PLC0415
        compare_photometry_science_meaningful,
        compute_photometry_sha,
    )

    cfg = AppConfig()
    cfg.k2_mode = "literature"
    cfg.save_lightcurve_png = False

    draft = Path(cfg.archive_root) / "Drafts" / f"draft_{DRAFT_ID:06d}"
    snapshot = Path(cfg.archive_root) / "Drafts" / SNAPSHOT_NAME
    ps = draft / "platesolve" / SETUP
    lights = draft / "detrended_aligned" / "lights" / SETUP

    if not snapshot.is_dir():
        report.add("full-snapshot", "FAIL", f"missing {snapshot}")
        return
    snap_meta_path = snapshot / "platesolve" / SETUP / "photometry" / "pipeline_meta.json"
    if snap_meta_path.is_file():
        from scripts.provenance_guard import parseable_git_hash, provenance_block  # noqa: PLC0415

        snap_meta = json.loads(snap_meta_path.read_text(encoding="utf-8"))
        gh = parseable_git_hash(provenance_block(snap_meta))
        if gh:
            report.add("full-provenance", "PASS", f"anchor git_hash={gh[:12]}...")
        else:
            report.add("full-provenance", "FAIL", "anchor snapshot missing provenance git_hash")
    else:
        report.add("full-provenance", "FAIL", f"missing {snap_meta_path.name}")
    for req, label in [
        (ps / "MASTERSTAR.fits", "MASTERSTAR.fits"),
        (ps / "variable_targets.csv", "variable_targets.csv"),
        (ps / "masterstars_full_match.csv", "masterstars_full_match.csv"),
        (lights, "detrended_aligned/lights"),
    ]:
        if not req.exists():
            report.add("full-inputs", "FAIL", f"missing {label}")
            return

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    work_root = REPO_ROOT / "tmp" / "session_baseline" / ts
    out_phot = work_root / "platesolve" / SETUP / "photometry"
    out_phot.mkdir(parents=True, exist_ok=True)

    reset_except_fix_counters()
    db = VyvarDatabase(cfg.database_path)
    t0 = datetime.now(timezone.utc)
    try:
        run_full_photometry_pipeline(
            masterstar_fits_path=ps / "MASTERSTAR.fits",
            variable_targets_csv=ps / "variable_targets.csv",
            masterstars_csv=ps / "masterstars_full_match.csv",
            per_frame_csv_dir=lights,
            detrended_aligned_dir=lights,
            output_dir=out_phot,
            cfg=cfg,
            db=db,
            draft_id=DRAFT_ID,
        )
    except Exception as exc:  # noqa: BLE001
        report.add("full-pipeline", "FAIL", str(exc)[:200])
        return
    finally:
        try:
            db.conn.close()
        except Exception:  # noqa: BLE001
            pass

    elapsed = (datetime.now(timezone.utc) - t0).total_seconds()
    report.add("full-pipeline", "PASS", f"{elapsed:.0f}s -> {work_root.relative_to(REPO_ROOT)}")

    cmp = compare_photometry_science_meaningful(
        work_root,
        snapshot,
        setups=(SETUP,),
    )
    summary = cmp.get("summary", {})
    if summary.get("benign"):
        report.add(
            "full-science-compare",
            "PASS",
            f"n_lc={summary.get('n_lc_compared')} failures=0",
        )
    else:
        sf = summary.get("science_failure_sample") or []
        detail = json.dumps(sf[:3], default=str)[:300]
        report.add(
            "full-science-compare",
            "FAIL",
            f"science_failures={summary.get('science_failures')} time_failures={summary.get('time_failures')} {detail}",
        )

    prov_hash = provenance_block_hash(out_phot)
    report.add(
        "full-provenance-hash",
        "PASS",
        f"{prov_hash[:16]}... (informational; git-bound, not cross-commit gate)",
    )

    core_sha, core_n = compute_photometry_sha(work_root, include_comp_qa=False)
    ext_sha, ext_n = compute_photometry_sha(work_root, include_comp_qa=True)
    snap_core_sha, snap_core_n = compute_photometry_sha(snapshot, include_comp_qa=False)
    snap_ext_sha, snap_ext_n = compute_photometry_sha(snapshot, include_comp_qa=True)
    if snap_core_sha != EXPECTED_PHOTOMETRY_SHA_CORE:
        report.add(
            "full-snapshot-sha-core",
            "FAIL",
            f"snapshot {snap_core_sha[:16]}... != expected {EXPECTED_PHOTOMETRY_SHA_CORE[:16]}...",
        )
    else:
        report.add("full-snapshot-sha-core", "PASS", f"{snap_core_sha[:16]}... n={snap_core_n}")
    if core_sha == snap_core_sha and core_n == snap_core_n:
        report.add(
            "full-photometry-sha-core",
            "PASS",
            f"{core_sha[:16]}... n={core_n}",
        )
    else:
        report.add(
            "full-photometry-sha-core",
            "FAIL",
            f"run {core_sha[:16]}... n={core_n} vs snap {snap_core_sha[:16]}... n={snap_core_n}",
        )
    if ext_sha == snap_ext_sha and ext_n == snap_ext_n:
        report.add(
            "full-photometry-sha-extended",
            "PASS",
            f"{ext_sha[:16]}... n={ext_n}",
        )
    else:
        report.add(
            "full-photometry-sha-extended",
            "FAIL",
            f"run {ext_sha[:16]}... vs snapshot extended mismatch",
        )

    counters = get_except_fix_counters().snapshot()
    nonzero = {k: v for k, v in counters.items() if v}
    meta_path = out_phot / "pipeline_meta.json"
    meta_summary: dict[str, Any] = {}
    if meta_path.is_file():
        meta_summary = json.loads(meta_path.read_text(encoding="utf-8")).get("except_fix_summary") or {}
    meta_nonzero = {k: v for k, v in meta_summary.items() if v}

    report.add("full-counters-runtime", "PASS" if not nonzero else "FAIL", json.dumps(nonzero))
    report.add("full-counters-meta", "PASS" if not meta_nonzero else "FAIL", json.dumps(meta_nonzero))

    if report.ok:
        _update_ledger_on_full_pass(_git_short_head())


def print_summary(report: SessionReport) -> None:
    print()
    print(f"SESSION BASELINE CHECK ({report.tier})")
    print("-" * 72)
    print(f"{'Check':<28} {'Status':<6} Detail")
    print("-" * 72)
    for r in report.results:
        detail = r.detail.replace("\n", " ")[:80]
        print(f"{r.name:<28} {r.status:<6} {detail}")
    print("-" * 72)
    if report.suspended:
        overall = "SUSPENDED"
    elif report.ok:
        overall = "PASS"
    else:
        overall = "FAIL"
    print(f"OVERALL: {overall}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Explicit alias for default fast tier (pytest + git/config/ledger only)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Deliberate full tier: draft_424 headless run + anchor + counters (~25 min)",
    )
    args = parser.parse_args(argv)

    report = SessionReport(tier="full" if args.full else "fast")
    check_git_state(report)
    check_config_paths(report)
    check_pytest(report)
    check_ledger_hint(report)
    if args.full:
        run_full_baseline(report)

    print_summary(report)
    if report.suspended:
        return 0
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
