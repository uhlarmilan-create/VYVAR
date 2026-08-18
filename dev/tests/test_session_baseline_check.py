"""Tests for scripts/session_baseline_check.py suspended-full path."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = REPO_ROOT / "validation" / "VYVAR_VALIDATION_LEDGER.json"


@pytest.fixture(name="ledger_path")
def fixture_ledger_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    import scripts.session_baseline_check as sbc

    path = tmp_path / "ledger.json"
    path.write_text(LEDGER_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(sbc, "LEDGER_PATH", path)
    return path


def test_suspend_message_when_offline(ledger_path: Path) -> None:
    import scripts.session_baseline_check as sbc

    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    # Remove active WCSINV so suspend falls through to offline anchor id.
    ledger["items"] = [it for it in ledger["items"] if it["id"] != "VL-ANCHOR-WCSINV"]
    for it in ledger["items"]:
        if it["id"] == "VL-ANCHOR-424":
            it["id"] = "VL-ANCHOR-WCSINV"
            it["status"] = "suspended_offline"
            it["passes"] = False
            it["offline_backup"] = {"path": r"C:\ASTRO\backups\test_anchor.zip"}
    ledger_path.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")

    msg = sbc._full_baseline_suspend_message()
    assert msg is not None
    assert "SUSPENDED pending new anchor" in msg
    assert r"C:\ASTRO\backups\test_anchor.zip" in msg


def test_suspend_message_none_when_active(ledger_path: Path) -> None:
    import scripts.session_baseline_check as sbc

    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    found = False
    for it in ledger["items"]:
        if it["id"] == "VL-ANCHOR-WCSINV":
            it["passes"] = True
            it.pop("status", None)
            found = True
    if not found:
        ledger["items"].append(
            {
                "id": "VL-ANCHOR-WCSINV",
                "area": "photometry",
                "description": "test",
                "verification": "test",
                "passes": True,
                "last_verified": "2026-07-16",
                "commit": None,
                "notes": "",
            }
        )
    ledger_path.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")

    assert sbc._full_baseline_suspend_message() is None


def test_run_full_baseline_suspended_short_circuit(ledger_path: Path) -> None:
    import scripts.session_baseline_check as sbc

    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["items"] = [it for it in ledger["items"] if it["id"] != "VL-ANCHOR-WCSINV"]
    for it in ledger["items"]:
        if it["id"] == "VL-ANCHOR-424":
            it["id"] = "VL-ANCHOR-WCSINV"
            it["status"] = "suspended_offline"
            it["passes"] = False
            it["offline_backup"] = {"path": r"C:\ASTRO\backups\test_anchor.zip"}
    ledger_path.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")

    report = sbc.SessionReport(tier="full")
    sbc.run_full_baseline(report)
    assert report.suspended
    assert report.ok
    assert len(report.results) == 1
    assert report.results[0].name == "full-baseline"
    assert report.results[0].status == "SUSPENDED"


def test_full_work_stamp_is_windows_path_safe() -> None:
    import scripts.session_baseline_check as sbc

    ts = sbc._full_work_stamp()
    assert ":" not in ts
    assert "T" in ts and ts.endswith("Z")
    probe = REPO_ROOT / "tmp" / f"_stamp_probe_{ts}"
    probe.mkdir(parents=True, exist_ok=True)
    probe.rmdir()


def test_except_fix_allowlist_is_draft_scoped() -> None:
    """B3: empty_comp_drop allowlist must be draft-keyed, not a global exemption."""
    import scripts.session_baseline_check as sbc

    by_draft = sbc.EXPECTED_EXCEPT_FIX_COUNTERS_BY_DRAFT
    assert 516 in by_draft
    assert by_draft[516] == {}
    # Other drafts have no allowlist - a nonzero counter would fail the gate.
    assert by_draft.get(999, {}) == {}
    assert by_draft.get(424, {}) == {}


def test_phase0_funnel_compare_detects_mismatch() -> None:
    from phase0_funnel import compare_phase0_funnel_fingerprints

    issues = compare_phase0_funnel_fingerprints(
        {"active_targets_rows": 322, "variable_targets_rows": 873},
        {"active_targets_rows": 169, "variable_targets_rows": 245},
    )
    assert any("active_targets_rows" in i for i in issues)
    assert any("variable_targets_rows" in i for i in issues)


def test_phase0_funnel_fingerprint_from_csv(tmp_path: Path) -> None:
    from phase0_funnel import compute_phase0_funnel_fingerprint

    vt = tmp_path / "variable_targets.csv"
    pd.DataFrame(
        [{"catalog_id": "1", "gaia_match_source": "masterstars", "vsx_name": "A"}]
    ).to_csv(vt, index=False)
    fp = compute_phase0_funnel_fingerprint(vt)
    assert fp["variable_targets_rows"] == 1
    assert fp["gaia_match_source_histogram"].get("masterstars") == 1


def test_main_full_exit_zero_when_suspended(monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.session_baseline_check as sbc

    monkeypatch.setattr(sbc, "check_git_state", lambda report: None)
    monkeypatch.setattr(sbc, "check_config_paths", lambda report: None)
    monkeypatch.setattr(sbc, "check_pytest", lambda report: None)
    monkeypatch.setattr(sbc, "check_ledger_hint", lambda report: None)
    monkeypatch.setattr(
        sbc,
        "_full_baseline_suspend_message",
        lambda: "full baseline SUSPENDED pending new anchor (test)",
    )

    code = sbc.main(["--full"])
    assert code == 0
