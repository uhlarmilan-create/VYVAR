# -*- coding: ascii -*-
"""POST-453 fixes Part 2: single infolog implementation."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src_py"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from infolog import (  # noqa: E402
    clear_log,
    get_milestones,
    log_milestone,
    log_phase_boundary,
    start_infolog_session,
    write_run_infolog,
)


def test_write_run_infolog_is_single_save_entrypoint(tmp_path: Path) -> None:
    clear_log()
    draft = tmp_path / "draft"
    draft.mkdir()
    start_infolog_session(draft)
    log_milestone("[SITE] observer location id=2 source=config")
    log_phase_boundary("calibration", status="start")
    log_milestone("INV-PREP-01 Preprocess gradient guard: ok ratio=1.0")
    log_milestone("INV-MS-01 MASTERSTAR count guard: ok")
    path = write_run_infolog(draft)
    assert path is not None
    text = Path(path).read_text(encoding="utf-8")
    assert "[SITE]" in text
    assert "INV-PREP-01" in text
    assert "INV-MS-01" in text
    assert "[PHASE] calibration start" in text
    assert "authoritative: durable session log" in text


def test_ring_buffer_export_labeled_partial_when_no_session(tmp_path: Path) -> None:
    clear_log()
    draft = tmp_path / "draft"
    draft.mkdir()
    log_milestone("INV-PREP-01 Preprocess gradient guard: ok ratio=1.0")
    path = write_run_infolog(draft)
    assert path is not None
    text = Path(path).read_text(encoding="utf-8")
    assert "partial: ring-buffer tail only" in text
    assert "INV-PREP-01" in text


def test_ui_and_headless_use_same_write_function() -> None:
    from infolog import write_run_infolog as ui_fn
    from infolog import write_run_infolog as headless_fn

    assert ui_fn is headless_fn
