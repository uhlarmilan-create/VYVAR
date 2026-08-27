#!/usr/bin/env python3
"""Session-start baseline check (--fast default; --full for frozen 516 anchor).

Exit 0 = PASS or SUSPENDED, 1 = FAIL. ASCII output; concise summary table at end.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# dev/scripts/session_baseline_check.py -> repo root is parents[2].
REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER_PATH = REPO_ROOT / "dev" / "validation" / "VYVAR_VALIDATION_LEDGER.json"
DB_QUICK_CHECK_WAIVER_PATH = REPO_ROOT / "dev" / "validation" / "db_quick_check_waiver.json"


def _load_db_quick_check_waiver() -> str | None:
    """Return waiver text if a committed waiver marker is present, else None."""
    if not DB_QUICK_CHECK_WAIVER_PATH.is_file():
        return None
    try:
        payload = json.loads(DB_QUICK_CHECK_WAIVER_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    text = str(payload.get("db_quick_check_waiver", "") or "").strip()
    return text or None


def _ensure_import_paths() -> None:
    """Put the VYVAR module roots on sys.path for direct (non-pytest) execution.

    src_py holds the flat VYVAR modules (config, photometry_core, ...); dev/ makes the
    tests/scripts namespace packages importable (tests.photometry_sha, scripts.provenance_guard);
    repo root is kept for the Phase-A layout where modules still live at the root.
    """
    for _p in (REPO_ROOT / "src_py", REPO_ROOT / "dev", REPO_ROOT):
        if _p.is_dir() and str(_p) not in sys.path:
            sys.path.insert(0, str(_p))


def _full_work_stamp() -> str:
    """UTC stamp safe as a single path component on Windows (no colons)."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


ANCHOR_LEDGER_ID = "VL-ANCHOR-WCSINV"

DRAFT_ID = 516
SETUP = "NoFilter_60_2"
SNAPSHOT_NAME = "draft_000516_snapshot_era04_20260826"
# era04 v1 raw-byte SHA (history; freeze-reproducible only at dffe859).
EXPECTED_PHOTOMETRY_SHA_CORE_V1 = "9367f99848c14b43016321d000ec53651c9b260290bcb37afd2f6bab5035b2d7"
EXPECTED_PHOTOMETRY_SHA_EXTENDED_V1 = "d3cefff3240b4874d9b0ba3f76f7a303a5e3ea8b83f051149202d5b9c65d6863"
# ANCHOR-HASH-01 v2 mixed core (53 aperture + 53 empty PSF + 54 other). History only.
EXPECTED_PHOTOMETRY_SHA_CORE_V2_MIXED = "af218acd32a4892cc4f0030168829852ced5c5140f83575301c1a39869437e66"
EXPECTED_PHOTOMETRY_SHA_EXTENDED_V2_MIXED = "ada5caff61692ff0489631e6278efedd8c92cb9bd26d05fcb67f2fb3729b1676"
# EPSF-CHAIN-01 ANCHOR SPLIT. era04_aperture = 53 aperture LCs (unchanged bytes).
EXPECTED_PHOTOMETRY_SHA_CORE_APERTURE = "d55fcc9d8ad9b55213c5c1813415cb54d54b88c3fc917bc81706065e4d824810"
EXPECTED_PHOTOMETRY_SHA_EXT_APERTURE = "cc8b532ee668b9b339e4170752b9d1054771b1236ecac8163688693586117167"
EXPECTED_PHOTOMETRY_SHA_CORE_APERTURE_N = 53
EXPECTED_PHOTOMETRY_SHA_EXT_APERTURE_N = 157
# core_psf (epsf01) is set after G3 passes on a --full product; empty until then.
EXPECTED_PHOTOMETRY_SHA_CORE_PSF = ""
EXPECTED_PHOTOMETRY_SHA_CORE_PSF_N = 53
EXPECTED_PHOTOMETRY_SHA_CORE = EXPECTED_PHOTOMETRY_SHA_CORE_V2_MIXED
EXPECTED_PHOTOMETRY_SHA_EXTENDED = EXPECTED_PHOTOMETRY_SHA_EXTENDED_V2_MIXED
EXPECTED_PHOTOMETRY_SHA_CORE_PREFIX = "af218acd"
EXPECTED_PHOTOMETRY_SHA_EXTENDED_PREFIX = "ada5caff"
EXPECTED_PHOTOMETRY_SHA_CORE_N = 160
EXPECTED_PHOTOMETRY_SHA_EXTENDED_N = 210
ANCHOR_MANIFEST_PATH = REPO_ROOT / "dev" / "validation" / "anchor_manifest.json"
G3_BO_ID = "1498613634033133184"
G3_FW_ID = "1497343732462852864"
G3_N_FULL = 134
G3_BO_REF_MMAG = 145.917
G3_FW_REF_MMAG = 14.557
G3_TOL_MMAG = 0.001
# Structural empty-comp drops keyed by draft_id only.
# 516 era04: three POOL-STARVE pin n_survivors<3 (phase2a_empty_comp_drop=3).
EXPECTED_EXCEPT_FIX_COUNTERS_BY_DRAFT: dict[int, dict[str, int]] = {
    516: {"phase2a_empty_comp_drop": 3},
}

# Phase 0 funnel fingerprints: frozen input VT + post-pipeline active_targets.
# Plan-regen (write_photometry_plan_files) is checked separately via EXPECTED_PLAN_REGEN_BY_DRAFT
# and covers the VSX->Gaia matcher; photometry SHA covers the identity join at pipeline time.
EXPECTED_PLAN_REGEN_BY_DRAFT: dict[int, dict[str, Any]] = {
    516: {
        "variable_targets_rows": 873,
        "gaia_match_source_histogram": {
            "gaia_dr3_direct": 433,
            "masterstars": 282,
            "masterstars_exo": 2,
            "no_match": 156,
        },
    },
}

EXPECTED_PHASE0_FUNNEL_BY_DRAFT: dict[int, dict[str, Any]] = {
    516: {
        "variable_targets_rows": 873,
        "gaia_match_source_histogram": {
            "gaia_dr3_direct": 433,
            "masterstars": 282,
            "masterstars_exo": 2,
            "no_match": 156,
        },
        "active_targets_rows": 253,
        "skip_photometry_true": 197,
        "skip_reason_histogram": {
            "": 53,
            "no_comps": 3,
            "per_frame_saturation": 1,
            "vsx_type_out_of_scope": 182,
            "zone_noise": 14,
        },
        "zone_flag_histogram": {
            "linear": 157,
            "noise": 14,
            "saturated": 1,
            "unknown": 81,
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


def _pytest_fail_nodes(out: str) -> list[str]:
    """Node ids from pytest -q FAILED/ERROR summary lines (NET-TEST-01)."""
    nodes: list[str] = []
    for m in re.finditer(r"^(?:FAILED|ERROR)\s+(\S+)", out, flags=re.M):
        node = m.group(1)
        if node not in nodes:
            nodes.append(node)
    return nodes


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
    if proc.returncode != 0:
        nodes = _pytest_fail_nodes(out)
        if nodes:
            shown = nodes[:8]
            counts.append("fail=" + ",".join(shown))
            if len(nodes) > 8:
                counts.append(f"+{len(nodes) - 8} more")
        else:
            counts.append(f"exit {proc.returncode} (no FAILED/ERROR node parsed)")
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


def check_anchor_manifest_db_parity(report: SessionReport) -> None:
    """Anchor draft: ``draft_manifest.json`` exists and is readable."""
    _ensure_import_paths()
    from config import AppConfig
    from database import VyvarDatabase
    from draft_provenance import backfill_draft_manifest_from_db, manifest_db_parity_errors

    cfg = AppConfig(project_root=REPO_ROOT)
    db_path = Path(cfg.database_path)
    if not db_path.is_file():
        report.add("manifest-db-parity", "WARN", "vyvar.sqlite3 missing")
        return

    anchor_dir = REPO_ROOT / "Archive" / "Drafts" / SNAPSHOT_NAME
    if not anchor_dir.is_dir():
        report.add("manifest-db-parity", "WARN", f"anchor dir missing ({SNAPSHOT_NAME})")
        return

    db = VyvarDatabase(db_path)
    try:
        did = int(DRAFT_ID)
        backfill_draft_manifest_from_db(db, did)
        errors = manifest_db_parity_errors(db, did)
        if errors:
            report.add("manifest-db-parity", "FAIL", errors[0][:80])
        else:
            report.add("manifest-db-parity", "PASS", f"draft_id={did}")
    except Exception as exc:  # noqa: BLE001
        report.add("manifest-db-parity", "FAIL", str(exc)[:80])
    finally:
        db.close()


def check_db_quick_check(report: SessionReport) -> None:
    """Fail fast if vyvar.sqlite3 PRAGMA quick_check is not ok (~1.2 s on production DB).

    When ``dev/validation/db_quick_check_waiver.json`` is present with
    ``db_quick_check_waiver`` text, a failing check reports WARN (not FAIL) so
    ``--fast`` can pass while corruption status remains visible on every run.
    """
    _ensure_import_paths()
    from config import AppConfig

    cfg = AppConfig(project_root=REPO_ROOT)
    db_path = Path(cfg.database_path).expanduser()
    if not db_path.is_file():
        report.add("db-quick-check", "WARN", "vyvar.sqlite3 missing")
        return
    waiver = _load_db_quick_check_waiver()
    try:
        conn = sqlite3.connect(f"file:{db_path.resolve().as_posix()}?mode=ro", uri=True)
        try:
            row = conn.execute("PRAGMA quick_check;").fetchone()
        finally:
            conn.close()
        msg = str(row[0]) if row else ""
        if msg.lower() == "ok":
            report.add("db-quick-check", "PASS", "ok")
        elif waiver:
            detail = f"WAIVED ({waiver[:48]}...): {msg[:40]}"
            report.add("db-quick-check", "WARN", detail)
        else:
            report.add("db-quick-check", "FAIL", msg[:80])
    except Exception as exc:  # noqa: BLE001
        if waiver:
            report.add("db-quick-check", "WARN", f"WAIVED ({waiver[:48]}...): {exc}"[:80])
        else:
            report.add("db-quick-check", "FAIL", str(exc)[:80])


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


def _load_anchor_manifest() -> dict[str, Any]:
    if not ANCHOR_MANIFEST_PATH.is_file():
        return {}
    return json.loads(ANCHOR_MANIFEST_PATH.read_text(encoding="utf-8"))


def _run_git_provenance() -> tuple[str, Any, list[str]]:
    _ensure_import_paths()
    from photometry_core import _resolve_git_provenance  # noqa: PLC0415

    git_hash, git_dirty, dirty_files = _resolve_git_provenance()
    files = [str(x) for x in (dirty_files or [])]
    return str(git_hash or ""), git_dirty, files


def _write_run_provenance(path: Path, *, git_hash: str, git_dirty: Any, files: list[str]) -> dict[str, Any]:
    payload = {
        "git_hash": git_hash,
        "git_dirty": git_dirty,
        "files": list(files),
    }
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="ascii")
    return payload


def _compare_provenance_to_anchor(
    report: SessionReport,
    run_prov: dict[str, Any],
    *,
    check_name: str = "full-provenance-drift",
) -> None:
    """PASS if git_hash+files match the freeze manifest; DRIFT otherwise (never FAIL)."""
    man = _load_anchor_manifest()
    want_hash = str(man.get("git_hash") or "")
    want_files = [str(x) for x in (man.get("files") or [])]
    got_hash = str(run_prov.get("git_hash") or "")
    got_files = [str(x) for x in (run_prov.get("files") or [])]
    files_ok = got_files == want_files
    if got_hash == want_hash and files_ok:
        report.add(check_name, "PASS", f"git_hash={got_hash[:12]}... files={len(got_files)}")
        return
    bits = []
    if got_hash != want_hash:
        bits.append(f"git_hash run={got_hash[:12]}... freeze={want_hash[:12]}...")
    if not files_ok:
        bits.append(f"files n={len(got_files)} vs freeze n={len(want_files)}")
    report.add(check_name, "DRIFT", "; ".join(bits)[:220])


def _run_epsf_stage_on_work(
    ps: Path,
    lights: Path,
    cfg: Any,
    *,
    db: Any,
    draft_id: int,
) -> dict[str, Any]:
    from epsf_stage import EpsfStagePaths, run_epsf_stage  # noqa: PLC0415

    return run_epsf_stage(
        params=None,
        paths=EpsfStagePaths(
            platesolve_dir=ps,
            frames_root=lights,
            masterstar_fits=ps / "MASTERSTAR.fits",
            masterstars_csv=ps / "masterstars_full_match.csv",
            photometry_dir=ps / "photometry",
        ),
        cfg=cfg,
        db=db,
        draft_id=int(draft_id),
    )


def _report_v1_raw_identity(
    report: SessionReport,
    work_root: Path,
    snapshot: Path,
) -> None:
    """One-shot v1 raw-byte record: expect 53 PSF LCs differ, 107 identical."""
    from tests.photometry_sha import photometry_file_hash_map  # noqa: PLC0415

    snap_map = photometry_file_hash_map(snapshot, include_comp_qa=False, strip_provenance=False)
    run_map = photometry_file_hash_map(work_root, include_comp_qa=False, strip_provenance=False)
    common = sorted(set(snap_map) & set(run_map))
    differ = [k for k in common if snap_map[k] != run_map[k]]
    identical = len(common) - len(differ)
    psf_differ = [k for k in differ if Path(k).name.endswith("_psf.csv")]
    other_differ = [k for k in differ if k not in psf_differ]
    if other_differ:
        report.add(
            "full-sha-v1-identity",
            "FAIL",
            f"STOP: non-PSF v1 diffs n={len(other_differ)} sample={other_differ[:3]}",
        )
        return
    report.add(
        "full-sha-v1-identity",
        "PASS",
        f"v1 raw: {identical} identical, {len(psf_differ)} PSF differ (expected 107/53)",
    )


def _g3_zp_ok_meters(work_root: Path) -> dict[str, Any]:
    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    lc_dir = work_root / "platesolve" / SETUP / "photometry" / "lightcurves"

    def _one(tid: str) -> dict[str, Any]:
        path = lc_dir / f"lightcurve_{tid}_psf.csv"
        if not path.is_file():
            return {"path": str(path), "missing": True}
        df = pd.read_csv(path, comment="#", low_memory=False)
        if "psf_delta_mag" not in df.columns:
            return {"path": str(path), "missing_col": True}
        x = pd.to_numeric(df["psf_delta_mag"], errors="coerce").to_numpy(dtype=float)
        fin = x[np.isfinite(x)]
        n_fin = int(fin.size)
        n = int(x.size)
        if n_fin == 0:
            dem = float("nan")
        else:
            med = float(np.median(fin))
            dem = float(np.sqrt(np.mean((fin - med) ** 2))) * 1000.0
        return {
            "catalog_id": tid,
            "n_finite": n_fin,
            "n_rows": n,
            "demeaned_rms_mmag": dem,
            "missing": False,
        }

    return {"bo": _one(G3_BO_ID), "fw": _one(G3_FW_ID)}


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


def _check_catalog_provenance(
    report: SessionReport,
    *,
    cfg: Any,
    snap_meta: dict[str, Any],
    run_meta: dict[str, Any],
) -> bool:
    from catalog_provenance import build_catalog_provenance_block, summarize_catalog_delta  # noqa: PLC0415

    snap_prov = snap_meta.get("provenance") if isinstance(snap_meta.get("provenance"), dict) else {}
    exp = snap_prov.get("catalog_databases")
    if not exp:
        exp = build_catalog_provenance_block(cfg)
    run_prov = run_meta.get("provenance") if isinstance(run_meta.get("provenance"), dict) else {}
    act = run_prov.get("catalog_databases")
    issues = summarize_catalog_delta(exp, act)
    if issues:
        report.add("full-catalog-provenance", "FAIL", "; ".join(issues)[:400])
        return False
    gaia = (act or {}).get("gaia_dr3") or {}
    vsx = (act or {}).get("vsx_local") or {}
    report.add(
        "full-catalog-provenance",
        "PASS",
        f"gaia rows={gaia.get('row_count')} g<={gaia.get('max_g_mag')} vsx rows={vsx.get('row_count')}",
    )
    return True


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


def _copy_frozen_anchor_inputs(snapshot: Path, work_root: Path) -> tuple[Path, Path]:
    """Copy snapshot catalogs + aligned lights into tmp so --full cannot mutate the freeze.

    The 435 lesson: photometering a live draft under evolving config is not a
    reproducibility gate. Inputs are snapshotted; the pipeline writes only under
    work_root.
    """
    ps_src = snapshot / "platesolve" / SETUP
    lights_src = snapshot / "detrended_aligned" / "lights" / SETUP
    ps_dst = work_root / "platesolve" / SETUP
    lights_dst = work_root / "detrended_aligned" / "lights" / SETUP
    if ps_dst.exists():
        shutil.rmtree(ps_dst)
    shutil.copytree(
        ps_src,
        ps_dst,
        ignore=shutil.ignore_patterns("photometry", "_hrd_cache", "*.pdf"),
    )
    if lights_dst.exists():
        shutil.rmtree(lights_dst)
    shutil.copytree(lights_src, lights_dst)
    # APERTURE-01d: night QC FWHM stamps fwhm_night_median_px on LCs (mode a).
    qc_src = snapshot / "calibrated" / "lights" / "qc_metrics.csv"
    if qc_src.is_file():
        qc_dst = work_root / "calibrated" / "lights"
        qc_dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(qc_src, qc_dst / "qc_metrics.csv")
    for name in ("cal_diag.json", "draft_manifest.json", "sat_diag.json"):
        src = snapshot / name
        if src.is_file():
            shutil.copy2(src, work_root / name)
            shutil.copy2(src, ps_dst / name)
    out_phot = ps_dst / "photometry"
    if out_phot.exists():
        shutil.rmtree(out_phot)
    out_phot.mkdir(parents=True, exist_ok=True)
    return ps_dst, lights_dst


def run_full_baseline(report: SessionReport) -> None:
    """Deliberate full tier: photometer frozen 516 snapshot inputs into tmp."""
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
        compute_photometry_sha_split,
    )

    cfg = AppConfig()
    cfg.k2_mode = "literature"
    cfg.save_lightcurve_png = False
    cfg.per_frame_saturation_enabled = True
    cfg.export_err_mode = "calibrated"

    snapshot = Path(cfg.archive_root) / "Drafts" / SNAPSHOT_NAME

    if not snapshot.is_dir():
        report.add("full-snapshot", "FAIL", f"missing {snapshot}")
        return
    snap_meta_path = snapshot / "platesolve" / SETUP / "photometry" / "pipeline_meta.json"
    snap_meta: dict[str, Any] = {}
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
    snap_ps = snapshot / "platesolve" / SETUP
    snap_lights = snapshot / "detrended_aligned" / "lights" / SETUP
    for req, label in [
        (snap_ps / "MASTERSTAR.fits", "MASTERSTAR.fits"),
        (snap_ps / "variable_targets.csv", "variable_targets.csv"),
        (snap_ps / "masterstars_full_match.csv", "masterstars_full_match.csv"),
        (snap_lights, "detrended_aligned/lights"),
    ]:
        if not req.exists():
            report.add("full-inputs", "FAIL", f"missing {label}")
            return

    ts = _full_work_stamp()
    work_root = REPO_ROOT / "tmp" / "session_baseline" / ts
    work_root.mkdir(parents=True, exist_ok=True)
    ps, lights = _copy_frozen_anchor_inputs(snapshot, work_root)
    out_phot = ps / "photometry"
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
        elapsed = (datetime.now(timezone.utc) - t0).total_seconds()
        report.add("full-pipeline", "PASS", f"{elapsed:.0f}s -> {work_root.relative_to(REPO_ROOT)}")
        t_psf = datetime.now(timezone.utc)
        epsf_out = _run_epsf_stage_on_work(ps, lights, cfg, db=db, draft_id=DRAFT_ID)
        n_psf = int((epsf_out.get("lc") or {}).get("n_written") or 0)
        psf_s = (datetime.now(timezone.utc) - t_psf).total_seconds()
        report.add(
            "full-epsf-stage",
            "PASS",
            f"n_stars={epsf_out.get('n_stars')} wrote {n_psf} PSF LCs in {psf_s:.0f}s",
        )
    except Exception as exc:  # noqa: BLE001
        if not any(r.name == "full-pipeline" for r in report.results):
            report.add("full-pipeline", "FAIL", str(exc)[:200])
        else:
            report.add("full-epsf-stage", "FAIL", str(exc)[:200])
        return
    finally:
        try:
            db.conn.close()
        except Exception:  # noqa: BLE001
            pass

    run_meta_path = out_phot / "pipeline_meta.json"
    run_meta: dict[str, Any] = {}
    if run_meta_path.is_file():
        run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
    catalog_ok = _check_catalog_provenance(report, cfg=cfg, snap_meta=snap_meta, run_meta=run_meta)

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
    from tests.photometry_sha import photometry_sha_files  # noqa: PLC0415

    sha_files = [p.relative_to(work_root).as_posix() for p in photometry_sha_files(work_root)]
    git_hash, git_dirty, _dirty = _run_git_provenance()
    run_prov = _write_run_provenance(
        work_root / "provenance.json",
        git_hash=git_hash,
        git_dirty=git_dirty,
        files=sha_files,
    )
    _write_run_provenance(
        out_phot / "provenance.json",
        git_hash=git_hash,
        git_dirty=git_dirty,
        files=sha_files,
    )
    _compare_provenance_to_anchor(report, run_prov)

    core_ap, n_ap = compute_photometry_sha_split(work_root, "core_aperture")
    core_psf, n_psf = compute_photometry_sha_split(work_root, "core_psf")
    ext_ap, n_ext_ap = compute_photometry_sha_split(work_root, "ext_aperture")
    snap_ap, snap_n_ap = compute_photometry_sha_split(snapshot, "core_aperture")
    snap_ext_ap, snap_n_ext_ap = compute_photometry_sha_split(snapshot, "ext_aperture")
    _report_v1_raw_identity(report, work_root, snapshot)
    if snap_ap != EXPECTED_PHOTOMETRY_SHA_CORE_APERTURE or snap_n_ap != EXPECTED_PHOTOMETRY_SHA_CORE_APERTURE_N:
        report.add(
            "full-snapshot-sha-core-aperture",
            "FAIL",
            f"snapshot {snap_ap[:16]}... n={snap_n_ap} != era04_aperture",
        )
    else:
        report.add(
            "full-snapshot-sha-core-aperture",
            "PASS",
            f"era04_aperture {snap_ap[:16]}... n={snap_n_ap}",
        )
    if not catalog_ok:
        report.add(
            "full-photometry-sha-core-aperture",
            "FAIL",
            "input catalogue changed (see full-catalog-provenance)",
        )
    elif core_ap == snap_ap and n_ap == snap_n_ap:
        report.add(
            "full-photometry-sha-core-aperture",
            "PASS",
            f"era04_aperture {core_ap[:16]}... n={n_ap}",
        )
    else:
        report.add(
            "full-photometry-sha-core-aperture",
            "FAIL",
            f"run {core_ap[:16]}... n={n_ap} vs snap {snap_ap[:16]}... n={snap_n_ap}",
        )
    if ext_ap == snap_ext_ap and n_ext_ap == snap_n_ext_ap:
        report.add(
            "full-photometry-sha-ext-aperture",
            "PASS",
            f"era04_aperture ext {ext_ap[:16]}... n={n_ext_ap}",
        )
    else:
        report.add(
            "full-photometry-sha-ext-aperture",
            "FAIL",
            f"run ext_aperture {ext_ap[:16]}... n={n_ext_ap} vs snap mismatch",
        )
    report.add(
        "full-photometry-sha-core-psf",
        "PASS" if n_psf == EXPECTED_PHOTOMETRY_SHA_CORE_PSF_N else "FAIL",
        f"epsf01 candidate {core_psf[:16]}... n={n_psf} (gate after G3)",
    )
    (work_root / "sha_split.json").write_text(
        json.dumps(
            {
                "core_aperture": core_ap,
                "n_core_aperture": n_ap,
                "core_psf": core_psf,
                "n_core_psf": n_psf,
                "ext_aperture": ext_ap,
                "n_ext_aperture": n_ext_ap,
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )

    g3 = _g3_zp_ok_meters(work_root)
    (work_root / "g3_zp_ok_meters.json").write_text(
        json.dumps(g3, indent=2) + "\n", encoding="ascii"
    )
    bo = g3.get("bo") or {}
    fw = g3.get("fw") or {}
    if bo.get("missing") or fw.get("missing") or bo.get("missing_col") or fw.get("missing_col"):
        report.add("full-g3-zp-ok", "FAIL", "missing BO/FW PSF LC or psf_delta_mag")
    else:
        bo_n = int(bo.get("n_finite") or 0)
        fw_n = int(fw.get("n_finite") or 0)
        bo_rms = float(bo.get("demeaned_rms_mmag"))
        fw_rms = float(fw.get("demeaned_rms_mmag"))
        bo_ok = bo_n == G3_N_FULL and abs(bo_rms - G3_BO_REF_MMAG) <= G3_TOL_MMAG
        fw_ok = fw_n == G3_N_FULL and abs(fw_rms - G3_FW_REF_MMAG) <= G3_TOL_MMAG
        detail = (
            f"BO {bo_n}/{bo.get('n_rows')} {bo_rms:.3f} mmag (ref {G3_BO_REF_MMAG}); "
            f"FW {fw_n}/{fw.get('n_rows')} {fw_rms:.3f} mmag (ref {G3_FW_REF_MMAG})"
        )
        if bo_n != G3_N_FULL or fw_n != G3_N_FULL:
            report.add("full-g3-zp-ok", "FAIL", f"n_full must be {G3_N_FULL}; {detail}")
        elif not (bo_ok and fw_ok):
            report.add("full-g3-zp-ok", "FAIL", detail)
        else:
            report.add("full-g3-zp-ok", "PASS", detail)

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


def check_clean_tree(report: SessionReport) -> None:
    """S7: tracked-only git worktree; pytest/ruff/pyflakes subset. Folder name must not matter."""
    import tempfile
    import uuid

    try:
        top = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(REPO_ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception as exc:  # noqa: BLE001
        report.add("clean-tree", "FAIL", f"git toplevel: {exc}")
        return
    wt_name = "b1b_clean_" + uuid.uuid4().hex[:8]
    parent = Path(tempfile.mkdtemp(prefix="vyvar_clean_"))
    wt = parent / wt_name
    add = subprocess.run(
        ["git", "worktree", "add", "--detach", str(wt), "HEAD"],
        cwd=top,
        capture_output=True,
        text=True,
    )
    if add.returncode != 0:
        shutil.rmtree(parent, ignore_errors=True)
        report.add("clean-tree", "FAIL", "worktree add: " + (add.stderr or add.stdout or "")[:180])
        return
    details: list[str] = [f"worktree={wt_name}"]
    ok = True
    try:
        pytest_cmd = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "dev/tests/test_params_registry.py",
            "dev/tests/test_s3_match_radius_d1.py",
            "dev/tests/test_s4_optimizer_d2.py",
            "dev/tests/test_s5_d3_candidacy.py",
            "dev/tests/test_s7_clean_tree_name.py",
        ]
        pt = subprocess.run(pytest_cmd, cwd=str(wt), capture_output=True, text=True, timeout=600)
        out = (pt.stdout or "") + (pt.stderr or "")
        m_pass = re.search(r"(\d+) passed", out)
        details.append("pytest " + (m_pass.group(0) if m_pass else f"exit {pt.returncode}"))
        if pt.returncode != 0:
            ok = False
            nodes = _pytest_fail_nodes(out)
            if nodes:
                details.append("fail=" + ",".join(nodes[:6]))
        ruff = subprocess.run(
            [sys.executable, "-m", "ruff", "check", "src_py", "dev/tests/test_params_registry.py"],
            cwd=str(wt),
            capture_output=True,
            text=True,
            timeout=120,
        )
        if ruff.returncode != 0:
            ok = False
            details.append("ruff FAIL")
        else:
            details.append("ruff PASS")
        flake = subprocess.run(
            [
                sys.executable,
                "-m",
                "pyflakes",
                "src_py/params_registry.py",
                "src_py/d3_comparison_candidacy.py",
            ],
            cwd=str(wt),
            capture_output=True,
            text=True,
            timeout=120,
        )
        if flake.returncode == 2 and "No module named" in (flake.stderr or ""):
            details.append("pyflakes SKIP (not installed)")
        elif flake.returncode != 0:
            ok = False
            details.append("pyflakes FAIL")
        else:
            details.append("pyflakes PASS")
    except subprocess.TimeoutExpired:
        ok = False
        details.append("timeout")
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(wt)],
            cwd=top,
            capture_output=True,
            text=True,
        )
        shutil.rmtree(parent, ignore_errors=True)
    report.add("clean-tree", "PASS" if ok else "FAIL", "; ".join(details)[:220])


def _stamp_params_dump(phot: dict[str, Any]) -> dict[str, Any]:
    return {
        "cfg_source": phot.get("cfg_source"),
        "cfg_changed_keys": list(phot.get("cfg_changed_keys") or []),
        "photometry_context": dict(phot.get("photometry_context") or {}),
        "errors": list(phot.get("errors") or []),
        "n_lightcurves": phot.get("n_lightcurves"),
        "n_frames": phot.get("n_frames"),
    }


def _reset_parity_outputs(snapshot: Path, ps: Path, lights: Path) -> None:
    """Wipe photometry/ and restore snapshot sidecars so a second sequential run is fair.

    One freeze copy: W2 hashes, then this reset, then W1. Do not copy lights twice
    (WinError 112 on a dual copy). Restore proc_*.csv and comparison_stars so a
    W2 sidecar mutation cannot seed W1.
    """
    out_phot = ps / "photometry"
    if out_phot.exists():
        shutil.rmtree(out_phot)
    out_phot.mkdir(parents=True, exist_ok=True)
    ps_src = snapshot / "platesolve" / SETUP
    dst_comp = ps / "comparison_stars_per_target.csv"
    src_comp = ps_src / "comparison_stars_per_target.csv"
    if src_comp.is_file():
        shutil.copy2(src_comp, dst_comp)
    elif dst_comp.is_file():
        dst_comp.unlink()
    lights_src = snapshot / "detrended_aligned" / "lights" / SETUP
    for p in lights_src.glob("proc_*.csv"):
        shutil.copy2(p, lights / p.name)


def run_parity_baseline(report: SessionReport) -> None:
    """G7: era04 snapshot through W1-as-wrapper and W2; core+ext hashes must match.

    Sequential one copy (W2 then wipe photometry/ then W1). Dual copy of the freeze
    exhausted the disk (WinError 112).
    """
    os.environ["VYVAR_P1_FORCE"] = "1"
    _ensure_import_paths()

    from config import AppConfig  # noqa: PLC0415
    from night_run import run_night_photometry, run_ui_night_photometry  # noqa: PLC0415
    from pipeline import AstroPipeline  # noqa: PLC0415
    from tests.photometry_sha import compute_photometry_sha_split  # noqa: PLC0415

    cfg = AppConfig()
    cfg.k2_mode = "literature"
    cfg.save_lightcurve_png = False
    cfg.per_frame_saturation_enabled = True
    cfg.export_err_mode = "calibrated"

    snapshot = Path(cfg.archive_root) / "Drafts" / SNAPSHOT_NAME
    if not snapshot.is_dir():
        report.add("parity-snapshot", "FAIL", f"missing {snapshot}")
        return
    report.add("parity-snapshot", "PASS", SNAPSHOT_NAME)

    ts = _full_work_stamp()
    work_root = REPO_ROOT / "tmp" / "session_parity" / ts
    work_root.mkdir(parents=True, exist_ok=True)
    ps, lights = _copy_frozen_anchor_inputs(snapshot, work_root)

    pipeline = AstroPipeline(cfg)

    def _run(label: str, fn: Any) -> dict[str, Any]:
        t0 = datetime.now(timezone.utc)
        phot = fn(
            cfg=cfg,
            pipeline=pipeline,
            draft_id=DRAFT_ID,
            draft_dir_override=work_root,
            write_pdfs=False,
            existing_draft=True,
        )
        elapsed = (datetime.now(timezone.utc) - t0).total_seconds()
        errs = list(phot.get("errors") or [])
        if errs:
            report.add(f"parity-{label}-run", "FAIL", f"{elapsed:.0f}s {errs[0]}"[:200])
        else:
            report.add(f"parity-{label}-run", "PASS", f"{elapsed:.0f}s")
        dump_path = work_root / f"parity_stamped_params_{label}.json"
        dump_path.write_text(
            json.dumps(_stamp_params_dump(phot), indent=2, default=str) + "\n",
            encoding="ascii",
        )
        stage = phot.get("epsf_stage") or {}
        report.add(
            f"parity-{label}-epsf",
            "PASS" if not errs else "FAIL",
            f"setups={list(stage.keys())}"[:200],
        )
        return phot

    phot_w2 = _run("w2", run_night_photometry)
    ap_w2, n_ap_w2 = compute_photometry_sha_split(work_root, "core_aperture")
    psf_w2, n_psf_w2 = compute_photometry_sha_split(work_root, "core_psf")
    sha_w2 = {
        "core_aperture": ap_w2,
        "n_core_aperture": n_ap_w2,
        "core_psf": psf_w2,
        "n_core_psf": n_psf_w2,
    }
    (work_root / "parity_sha_w2.json").write_text(
        json.dumps(sha_w2, indent=2) + "\n", encoding="ascii"
    )

    _reset_parity_outputs(snapshot, ps, lights)

    phot_w1 = _run("w1", run_ui_night_photometry)
    ap_w1, n_ap_w1 = compute_photometry_sha_split(work_root, "core_aperture")
    psf_w1, n_psf_w1 = compute_photometry_sha_split(work_root, "core_psf")
    sha_w1 = {
        "core_aperture": ap_w1,
        "n_core_aperture": n_ap_w1,
        "core_psf": psf_w1,
        "n_core_psf": n_psf_w1,
    }
    (work_root / "parity_sha_w1.json").write_text(
        json.dumps(sha_w1, indent=2) + "\n", encoding="ascii"
    )

    ap_ok = (
        ap_w1 == ap_w2 == EXPECTED_PHOTOMETRY_SHA_CORE_APERTURE
        and n_ap_w1 == n_ap_w2 == EXPECTED_PHOTOMETRY_SHA_CORE_APERTURE_N
    )
    psf_ok = ap_ok and psf_w1 == psf_w2 and n_psf_w1 == n_psf_w2 == EXPECTED_PHOTOMETRY_SHA_CORE_PSF_N
    if ap_ok and psf_ok:
        report.add(
            "parity-sha",
            "PASS",
            f"core_aperture={ap_w1[:16]}... n={n_ap_w1} core_psf={psf_w1[:16]}... n={n_psf_w1}",
        )
    else:
        dump_w1 = _stamp_params_dump(phot_w1)
        dump_w2 = _stamp_params_dump(phot_w2)
        report.add(
            "parity-sha",
            "FAIL",
            (
                f"w1 ap={ap_w1[:16]} n={n_ap_w1} psf={psf_w1[:16]} n={n_psf_w1} "
                f"w2 ap={ap_w2[:16]} n={n_ap_w2} psf={psf_w2[:16]} n={n_psf_w2} "
                f"want ap={EXPECTED_PHOTOMETRY_SHA_CORE_APERTURE[:16]} n=53 "
                f"params_w1={json.dumps(dump_w1, default=str)[:120]} "
                f"params_w2={json.dumps(dump_w2, default=str)[:120]}"
            )[:400],
        )


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
        help="Deliberate full tier: frozen 516 snapshot headless run + anchor + counters",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Also run the tracked-only worktree gate (pytest/ruff/pyflakes subset)",
    )
    parser.add_argument(
        "--parity",
        action="store_true",
        help="G7 EXPORT-PARITY: era04 snapshot through W1-as-wrapper and W2; core+ext hashes must match",
    )
    args = parser.parse_args(argv)

    try:
        sys.path.insert(0, str(REPO_ROOT / "dev" / "scripts"))
        from push_guard import install_hook  # noqa: PLC0415

        install_hook(REPO_ROOT)
    except Exception:  # noqa: BLE001
        pass

    report = SessionReport(
        tier="parity" if args.parity and not args.full else ("full" if args.full else "fast")
    )
    check_git_state(report)
    check_config_paths(report)
    # Dedicated --parity is a W1/W2 hash gate. Pytest stays on --fast/--full so a
    # flaky sqlite threading test cannot false-fail G7 (G1 already covers pytest).
    if not (args.parity and not args.full and not args.fast):
        check_pytest(report)
        check_anchor_manifest_db_parity(report)
        check_db_quick_check(report)
        check_ledger_hint(report)
        check_deps_outdated(report)
    if args.clean:
        check_clean_tree(report)
    if args.full:
        run_full_baseline(report)
    if args.parity:
        run_parity_baseline(report)

    print_summary(report)
    if report.suspended:
        return 0
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
