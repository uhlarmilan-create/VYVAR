# -*- coding: ascii -*-
"""INFOLOG-ONE-2026-10: one infolog file per run, every line written once, no UI tab."""
from __future__ import annotations

import logging
from pathlib import Path

from infolog import (
    end_infolog_session,
    ensure_infolog_logging,
    log_event,
    log_milestone,
    log_phase_boundary,
    start_infolog_session,
)

SRC = Path(__file__).resolve().parents[2] / "src_py"


def test_one_file_each_line_once(tmp_path: Path) -> None:
    draft = tmp_path / "draft"
    ensure_infolog_logging()
    log_milestone("[SITE] observer location id=2 source=config")  # before the draft log opens
    start_infolog_session(draft)
    log_phase_boundary("calibration", status="start")
    log_milestone("INV-PREP-01 Preprocess gradient guard: ok ratio=1.0")
    log_event("plain event")
    logging.getLogger("pipeline").info("logger line")
    end_infolog_session()
    log_event("after close")  # outside a run: nowhere
    files = list(draft.glob("infolog_*.txt"))
    assert len(files) == 1
    text = files[0].read_text(encoding="utf-8")
    for needle in ("[SITE] observer location", "[PHASE] calibration start", "INV-PREP-01", "plain event", "logger line"):
        assert text.count(needle) == 1, needle
    assert "after close" not in text


def test_next_run_does_not_repeat_old_milestones(tmp_path: Path) -> None:
    log_milestone("early milestone")
    start_infolog_session(tmp_path / "a")
    end_infolog_session()
    start_infolog_session(tmp_path / "b")
    end_infolog_session()
    assert "early milestone" in next((tmp_path / "a").glob("infolog_*.txt")).read_text(encoding="utf-8")
    assert "early milestone" not in next((tmp_path / "b").glob("infolog_*.txt")).read_text(encoding="utf-8")


def test_app_has_no_infolog_tab_or_second_writer() -> None:
    app = (SRC / "app.py").read_text(encoding="utf-8")
    assert '"Infolog"' not in app and "render_infolog" not in app
    assert "from infolog import" not in app
