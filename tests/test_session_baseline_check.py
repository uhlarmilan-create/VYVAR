"""Tests for scripts/session_baseline_check.py suspended-full path."""

from __future__ import annotations

import json
from pathlib import Path

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
    for it in ledger["items"]:
        if it["id"] == "VL-ANCHOR-424":
            it["status"] = "suspended_offline"
            it["offline_backup"] = {"path": r"C:\ASTRO\backups\test_anchor.zip"}
    ledger_path.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")

    msg = sbc._full_baseline_suspend_message()
    assert msg is not None
    assert "SUSPENDED pending new anchor" in msg
    assert r"C:\ASTRO\backups\test_anchor.zip" in msg


def test_suspend_message_none_when_active(ledger_path: Path) -> None:
    import scripts.session_baseline_check as sbc

    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    for it in ledger["items"]:
        if it["id"] == "VL-ANCHOR-424":
            it.pop("status", None)
    ledger_path.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")

    assert sbc._full_baseline_suspend_message() is None


def test_run_full_baseline_suspended_short_circuit(ledger_path: Path) -> None:
    import scripts.session_baseline_check as sbc

    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    for it in ledger["items"]:
        if it["id"] == "VL-ANCHOR-424":
            it["status"] = "suspended_offline"
            it["offline_backup"] = {"path": r"C:\ASTRO\backups\test_anchor.zip"}
    ledger_path.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")

    report = sbc.SessionReport(tier="full")
    sbc.run_full_baseline(report)
    assert report.suspended
    assert report.ok
    assert len(report.results) == 1
    assert report.results[0].name == "full-baseline"
    assert report.results[0].status == "SUSPENDED"


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
