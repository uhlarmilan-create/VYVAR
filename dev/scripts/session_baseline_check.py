#!/usr/bin/env python3
"""Session-start baseline check (--fast default; --full for draft_424 anchor re-verify).

Exit 0 = PASS or SUSPENDED, 1 = FAIL. ASCII output; concise summary table at end.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# dev/scripts/session_baseline_check.py -> repo root is parents[2].
REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER_PATH = REPO_ROOT / "dev" / "validation" / "VYVAR_VALIDATION_LEDGER.json"


def _ensure_import_paths() -> None:
    """Put the VYVAR module roots on sys.path for direct (non-pytest) execution.

    src_py holds the flat VYVAR modules (config, photometry_core, ...); dev/ makes the
    tests/scripts namespace packages importable (tests.photometry_sha, scripts.provenance_guard);
    repo root is kept for the Phase-A layout where modules still live at the root.
    """
    for _p in (REPO_ROOT / "src_py", REPO_ROOT / "dev", REPO_ROOT):
        if _p.is_dir() and str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
ANCHOR_LEDGER_ID = "VL-ANCHOR-WCSINV"

DRAFT_ID = 435
SETUP = "NoFilter_60_2"
SNAPSHOT_NAME = "draft_000435_snapshot_skysurface_20260716"
# Content anchors (Anchor #3 / sky-surface; LABBE-DET err-stable dual pass on 10d610c).
EXPECTED_PHOTOMETRY_SHA_CORE = "b7f980c09e238b855c2ee1b9518061777934d8f0a61eaec7431cda4f537aed52"
EXPECTED_PHOTOMETRY_SHA_EXTENDED = "2c43bbbf06921fbef46fb6a4ed1f8afccdabacaa5827b8ec50372de0e3816205"
EXPECTED_PHOTOMETRY_SHA_CORE_PREFIX = "b7f980c0"
EXPECTED_PHOTOMETRY_SHA_EXTENDED_PREFIX = "2c43bbbf"
# Structural empty-comp drops keyed by draft_id only (Anchor #3 / R CVn on 435).
# A future draft with phase2a_empty_comp_drop>0 must FAIL until explicitly listed.
# Value must match exactly - empty_comp_drop=2 on 435 still trips the gate.
EXPECTED_EXCEPT_FIX_COUNTERS_BY_DRAFT: dict[int, dict[str, int]] = {
    435: {"phase2a_empty_comp_drop": 1},  # R CVn 1496795041799526400 (+ siblings w/ 0 comps)
}

# Phase 0 funnel fingerprints: frozen input VT + post-pipeline active_targets.
# Plan-regen (write_photometry_plan_files) is checked separately via EXPECTED_PLAN_REGEN_BY_DRAFT
# and covers the VSX->Gaia matcher; photometry SHA covers the identity join at pipeline time.
EXPECTED_PLAN_REGEN_BY_DRAFT: dict[int, dict[str, Any]] = {
    435: {
        "variable_targets_rows": 875,
        "gaia_match_source_histogram": {
            "gaia_dr3_direct": 512,
            "masterstars": 205,
            "masterstars_exo": 2,
            "no_match": 156,
        },
    },
}

EXPECTED_PHASE0_FUNNEL_BY_DRAFT: dict[int, dict[str, Any]] = {
    435: {
        "variable_targets_rows": 245,
        "gaia_match_source_histogram": {
            "gaia_dr3_direct": 64,
            "masterstars": 178,
            "masterstars_exo": 2,
            "no_match": 1,
        },
        "active_targets_rows": 165,
        "skip_photometry_true": 2,
        "skip_reason_histogram": {"": 162, "no_comps": 1, "zone_flag": 2},
        "zone_flag_histogram": {
            "linear": 110,
            "noisy1": 10,
            "noisy2": 5,
            "noisy3": 38,
            "saturated": 2,
        },
    },
}

# Known untracked paths: WARN only (not FAIL). Extend when deliberately added.
KNOWN_UNTRACKED_PREFIXES = (
    ".worktrees/",
    "dev/results/CURSOR_RESULT",
    "dev/scripts/dy_peg_night_run_bvr.py",
    "dev/scripts/qatar8_night_run_v.py",
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
    _ensure_import_paths()
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


# Packages with explicit major-holding pins in requirements.txt. Outdated
# versions here are informational only (upgrades are gated per DEPS_POLICY.md).
_TRACKED_DEPS = ("numpy", "astropy", "photutils")


def check_deps_outdated(report: SessionReport) -> None:
    """Informational: surface outdated tracked deps. Never FAILs (WARN/PASS/SKIP).

    Upgrades are a deliberate, gated ritual (see docs/DEPS_POLICY.md); this line
    is a nudge, not a blocker. Best-effort: offline or slow index => SKIP.
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--outdated", "--format=json"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        report.add("deps-outdated", "SKIP", "pip list --outdated timed out (offline?)")
        return
    except Exception as exc:  # noqa: BLE001 - informational check must never break session
        report.add("deps-outdated", "SKIP", f"unavailable: {exc}")
        return
    if proc.returncode != 0:
        report.add("deps-outdated", "SKIP", "pip list --outdated unavailable (offline?)")
        return
    try:
        rows = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError:
        report.add("deps-outdated", "SKIP", "could not parse pip output")
        return
    by_name = {str(r.get("name", "")).lower(): r for r in rows if isinstance(r, dict)}
    tracked_hits = []
    for name in _TRACKED_DEPS:
        r = by_name.get(name)
        if r:
            tracked_hits.append(f"{name} {r.get('version')}->{r.get('latest_version')}")
    if tracked_hits:
        extra = len(rows) - len(tracked_hits)
        suffix = f" (+{extra} other)" if extra > 0 else ""
        report.add(
            "deps-outdated",
            "WARN",
            "; ".join(tracked_hits) + suffix + " - gated upgrade, see docs/DEPS_POLICY.md",
        )
    else:
        report.add("deps-outdated", "PASS", f"tracked deps current ({len(rows)} other outdated)")


def _load_ledger() -> dict[str, Any]:
    return json.loads(LEDGER_PATH.read_text(encoding="utf-8"))


def _ledger_item(ledger_id: str) -> dict[str, Any] | None:
    for it in _load_ledger().get("items", []):
        if it.get("id") == ledger_id:
            return it
    return None


def _full_baseline_suspend_message() -> str | None:
    """Return suspend message when no in-Archive anchor is active; else None."""
    wcs = _ledger_item("VL-ANCHOR-WCSINV")
    if wcs and wcs.get("passes") and wcs.get("status") != "suspended_offline":
        return None
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
        if it["id"] in (ANCHOR_LEDGER_ID, "VL-COUNTERS-ZERO"):
            it["passes"] = True
            it["last_verified"] = today
            it["commit"] = commit
            it.pop("status", None)
    ledger["updated"] = today
    LEDGER_PATH.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")


def _check_plan_regen_fingerprint(
    report: SessionReport,
    *,
    cfg: Any,
    ps: Path,
    work_root: Path,
    draft_id: int,
) -> None:
    """Regenerate plan-time variable_targets.csv and compare to recorded anchor fingerprint."""
    from config import AppConfig  # noqa: PLC0415
    from phase0_funnel import compare_phase0_funnel_fingerprints, compute_phase0_funnel_fingerprint  # noqa: PLC0415
    from pipeline import write_photometry_plan_files  # noqa: PLC0415

    expected = EXPECTED_PLAN_REGEN_BY_DRAFT.get(int(draft_id))
    if not expected:
        report.add("full-plan-regen", "SKIP", f"no plan-regen fingerprint for draft {draft_id}")
        return

    regen_dir = work_root / "plan_regen"
    regen_dir.mkdir(parents=True, exist_ok=True)
    ms_fits = ps / "MASTERSTAR.fits"
    ms_csv = ps / "masterstars_full_match.csv"
    if not ms_fits.is_file() or not ms_csv.is_file():
        report.add("full-plan-regen", "FAIL", "missing MASTERSTAR or masterstars for regen")
        return

    _cfg = cfg if isinstance(cfg, AppConfig) else AppConfig()
    try:
        plan_result = write_photometry_plan_files(
            platesolve_dir=regen_dir,
            masterstar_fits=ms_fits,
            masterstars_csv=ms_csv,
            draft_id=int(draft_id),
            database_path=_cfg.database_path,
        )
    except Exception as exc:  # noqa: BLE001
        report.add("full-plan-regen", "FAIL", f"write_photometry_plan_files: {exc!s}"[:200])
        return

    regen_vt = regen_dir / "variable_targets.csv"
    if not regen_vt.is_file():
        report.add("full-plan-regen", "FAIL", "regenerated variable_targets.csv missing")
        return

    regen_fp = compute_phase0_funnel_fingerprint(regen_vt, active_targets_csv=None)
    exp_input = {
        "variable_targets_rows": expected.get("variable_targets_rows"),
        "gaia_match_source_histogram": expected.get("gaia_match_source_histogram"),
    }
    issues = compare_phase0_funnel_fingerprints(
        {
            "variable_targets_rows": regen_fp.get("variable_targets_rows"),
            "gaia_match_source_histogram": regen_fp.get("gaia_match_source_histogram"),
        },
        exp_input,
    )
    plan_json = regen_dir / "photometry_plan.json"
    diag_note = ""
    if plan_json.is_file():
        try:
            diag = json.loads(plan_json.read_text(encoding="utf-8")).get("variable_targets_diagnostics") or {}
            if diag:
                diag_note = f" diag_keys={sorted(diag.keys())[:6]}"
        except Exception:  # noqa: BLE001
            pass
    if plan_result.get("error"):
        report.add(
            "full-plan-regen",
            "FAIL",
            f"plan error: {plan_result.get('error')}{diag_note}"[:200],
        )
        return
    if issues:
        report.add(
            "full-plan-regen",
            "FAIL",
            f"regen mismatch: {'; '.join(issues)[:180]}{diag_note}",
        )
    else:
        report.add(
            "full-plan-regen",
            "PASS",
            json.dumps(
                {
                    "variable_targets_rows": regen_fp.get("variable_targets_rows"),
                    "gaia_match_source_histogram": regen_fp.get("gaia_match_source_histogram"),
                }
            )[:200],
        )


def run_full_baseline(report: SessionReport) -> None:
    """Deliberate full tier: force P1 golden execution when env flag is set."""
    os.environ["VYVAR_P1_FORCE"] = "1"
    suspend_msg = _full_baseline_suspend_message()
    if suspend_msg:
        report.add("full-baseline", "SUSPENDED", suspend_msg)
        return

    _ensure_import_paths()

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

    _check_plan_regen_fingerprint(report, cfg=cfg, ps=ps, work_root=work_root, draft_id=DRAFT_ID)

    reset_except_fix_counters()
    db = VyvarDatabase(cfg.database_path)
    # DB-SEED-SPLIT: fresh DBs are empty; ensure the author reference observatory rows
    # exist so the anchor draft's optics/location FKs resolve. INSERT OR IGNORE => no-op
    # on the author's populated production DB, so the anchor stays byte-identical (and a
    # run over a fresh DB now also proves the seed split preserved the anchor context).
    from tools.reference_seed import seed_reference_observatory  # noqa: PLC0415

    seed_reference_observatory(db)
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
    nonzero = {k: int(v) for k, v in counters.items() if v}
    meta_path = out_phot / "pipeline_meta.json"
    meta_summary: dict[str, Any] = {}
    if meta_path.is_file():
        meta_summary = json.loads(meta_path.read_text(encoding="utf-8")).get("except_fix_summary") or {}
    meta_nonzero = {k: int(v) for k, v in meta_summary.items() if v}
    expected = {
        k: int(v)
        for k, v in (EXPECTED_EXCEPT_FIX_COUNTERS_BY_DRAFT.get(int(DRAFT_ID), {}) or {}).items()
        if int(v) > 0
    }
    counters_ok = nonzero == expected
    meta_ok = meta_nonzero == expected
    report.add(
        "full-counters-runtime",
        "PASS" if counters_ok else "FAIL",
        json.dumps(nonzero) if nonzero else "{}",
    )
    report.add(
        "full-counters-meta",
        "PASS" if meta_ok else "FAIL",
        json.dumps(meta_nonzero) if meta_nonzero else "{}",
    )
    if expected and counters_ok:
        report.add(
            "full-counters-expected",
            "PASS",
            f"allowlisted {json.dumps(expected)} (structural empty-comp drops)",
        )

    from phase0_funnel import (  # noqa: PLC0415
        compare_phase0_funnel_fingerprints,
        compute_phase0_funnel_fingerprint,
    )

    expected_funnel = EXPECTED_PHASE0_FUNNEL_BY_DRAFT.get(int(DRAFT_ID))
    if expected_funnel:
        input_fp = compute_phase0_funnel_fingerprint(
            ps / "variable_targets.csv",
            active_targets_csv=None,
        )
        input_only = {
            "variable_targets_rows": input_fp.get("variable_targets_rows"),
            "gaia_match_source_histogram": input_fp.get("gaia_match_source_histogram"),
        }
        exp_input = {
            "variable_targets_rows": expected_funnel.get("variable_targets_rows"),
            "gaia_match_source_histogram": expected_funnel.get("gaia_match_source_histogram"),
        }
        input_issues = compare_phase0_funnel_fingerprints(input_only, exp_input)
        report.add(
            "full-phase0-input-vt",
            "PASS" if not input_issues else "FAIL",
            json.dumps(input_only)[:200] if not input_issues else "; ".join(input_issues)[:200],
        )

        out_at = out_phot / "active_targets.csv"
        run_fp = compute_phase0_funnel_fingerprint(
            ps / "variable_targets.csv",
            active_targets_csv=out_at,
        )
        run_issues = compare_phase0_funnel_fingerprints(run_fp, expected_funnel)
        report.add(
            "full-phase0-funnel",
            "PASS" if not run_issues else "FAIL",
            json.dumps(
                {
                    "active_targets_rows": run_fp.get("active_targets_rows"),
                    "skip_photometry_true": run_fp.get("skip_photometry_true"),
                    "zone_flag_histogram": run_fp.get("zone_flag_histogram"),
                }
            )[:200]
            if not run_issues
            else "; ".join(run_issues)[:200],
        )

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
        help="Deliberate full tier: draft_435 headless run + anchor + counters (~25 min)",
    )
    args = parser.parse_args(argv)

    report = SessionReport(tier="full" if args.full else "fast")
    check_git_state(report)
    check_config_paths(report)
    check_pytest(report)
    check_ledger_hint(report)
    check_deps_outdated(report)
    if args.full:
        run_full_baseline(report)

    print_summary(report)
    if report.suspended:
        return 0
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
