# -*- coding: ascii -*-
"""VYVAR v12c (analysis of Milan's v12 runs, 2026-11-06).

XMATCH-HOLD-2026-11: an online catalogue that did not answer is not asked again for every candidate.
STATUS-REPLACE-2026-11: the run status file survives a Windows sharing violation (WinError 5).
CHECK-KMAG-PREF-2026-11: the check star is taken within 2 mag of the target when such a candidate exists."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import catalog_crossmatch as cx
import net_call
import run_job
from config import AppConfig


# ------------------------------------------------------------------ XMATCH-HOLD-2026-11
def test_unreachable_catalogue_is_skipped_for_the_next_candidates(monkeypatch) -> None:
    net_call.clear_unavailable()
    calls: list[str] = []

    def _down(coord, r):
        calls.append("SIMBAD")
        raise ConnectionError("SIMBAD: connection timed out")  # requests / socket errors are OSError

    def _ok(coord, r):
        calls.append("VSX")
        return []

    monkeypatch.setattr(cx, "_CATALOG_WORKERS", [("SIMBAD", _down), ("VSX", _ok)])
    r1 = cx.check_candidate_in_catalogs(10.0, 20.0)
    assert "SIMBAD" in r1.errors and calls.count("SIMBAD") == 1
    r2 = cx.check_candidate_in_catalogs(10.1, 20.1)
    assert calls.count("SIMBAD") == 1 and calls.count("VSX") == 2, "SIMBAD not asked again"
    assert "not reachable a moment ago" in r2.errors["SIMBAD"]
    net_call.clear_unavailable()


def test_a_query_error_that_is_not_a_connection_problem_does_not_hold(monkeypatch) -> None:
    net_call.clear_unavailable()
    calls: list[str] = []

    def _bad(coord, r):
        calls.append("ZTF")
        raise ValueError("unexpected table layout")

    monkeypatch.setattr(cx, "_CATALOG_WORKERS", [("ZTF", _bad)])
    cx.check_candidate_in_catalogs(10.0, 20.0)
    cx.check_candidate_in_catalogs(10.1, 20.1)
    assert calls == ["ZTF", "ZTF"]
    net_call.clear_unavailable()


# ------------------------------------------------------------------ STATUS-REPLACE-2026-11
def test_status_replace_retries_a_sharing_violation(monkeypatch, tmp_path: Path) -> None:
    real = run_job.os.replace
    n = {"fail": 3}

    def _flaky(src, dst):
        if n["fail"]:
            n["fail"] -= 1
            raise PermissionError(13, "Access is denied")
        return real(src, dst)

    monkeypatch.setattr(run_job.os, "replace", _flaky)
    monkeypatch.setattr(run_job, "_REPLACE_WAIT_S", 0.0)
    p = tmp_path / "run_status.json"
    run_job._write_json_atomic(p, {"state": "running"})
    assert json.loads(p.read_text())["state"] == "running" and not list(tmp_path.glob("*.tmp"))


def test_status_replace_gives_up_loudly(monkeypatch, tmp_path: Path) -> None:
    def _locked(src, dst):
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr(run_job.os, "replace", _locked)
    monkeypatch.setattr(run_job, "_REPLACE_WAIT_S", 0.0)
    with pytest.raises(PermissionError):
        run_job._write_json_atomic(tmp_path / "run_status.json", {"state": "running"})
    assert not list(tmp_path.glob("*.tmp"))


# ------------------------------------------------------------------ CHECK-KMAG-PREF-2026-11
def _pool(rows):
    return pd.DataFrame([{"status": "good", "comp_tier": 1, "bp_rp": 1.0, **r} for r in rows])


def test_check_star_within_two_mag_beats_a_quieter_much_brighter_one() -> None:
    from check_star_kmag import select_check_star

    pool = _pool([{"catalog_id": "1", "comp_rms": 0.003, "catalog_mag": 13.0},   # 4.5 mag brighter
                  {"catalog_id": "2", "comp_rms": 0.008, "catalog_mag": 16.5}])  # 1.0 mag brighter
    row = select_check_star(pool, ensemble_ids=set(), cfg=AppConfig(), target_bp_rp=1.0, target_mag=17.5)
    assert str(row["catalog_id"]) == "2"


def test_without_a_candidate_in_the_window_the_quietest_is_taken() -> None:
    from check_star_kmag import select_check_star

    pool = _pool([{"catalog_id": "1", "comp_rms": 0.003, "catalog_mag": 13.0},
                  {"catalog_id": "2", "comp_rms": 0.004, "catalog_mag": 14.0}])
    row = select_check_star(pool, ensemble_ids=set(), cfg=AppConfig(), target_bp_rp=1.0, target_mag=17.5)
    assert str(row["catalog_id"]) == "1"
