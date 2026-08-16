"""RUN-HARDEN-01: exit log + active-RUN gating fire proofs."""
from __future__ import annotations

import pytest

from run_lifecycle import (
    format_run_exit_line,
    is_vyvar_run_active,
    run_callable_with_exit_log,
)


def test_is_vyvar_run_active_true_false() -> None:
    assert is_vyvar_run_active({"running": True}) is True
    assert is_vyvar_run_active({"running": False}) is False
    assert is_vyvar_run_active({}) is False
    assert is_vyvar_run_active(None) is False
    assert is_vyvar_run_active("nope") is False


def test_format_run_exit_ok_and_false() -> None:
    assert format_run_exit_line(ok=True) == "[RUN] finished OK"
    assert format_run_exit_line(ok=False) == "[RUN] aborted: returned False"


def test_run_callable_logs_finished_ok() -> None:
    lines: list[str] = []
    assert run_callable_with_exit_log(lambda: True, lines.append) is True
    assert lines == ["[RUN] finished OK"]


def test_run_callable_logs_aborted_on_exception() -> None:
    lines: list[str] = []

    def _boom() -> bool:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        run_callable_with_exit_log(_boom, lines.append)
    assert len(lines) == 1
    assert lines[0].startswith("[RUN] aborted: RuntimeError: boom")


def test_run_callable_logs_interrupted_on_streamlit_rerun() -> None:
    from streamlit.runtime.scriptrunner_utils.exceptions import RerunData, RerunException

    lines: list[str] = []

    def _rerun() -> bool:
        raise RerunException(RerunData())

    with pytest.raises(RerunException):
        run_callable_with_exit_log(_rerun, lines.append)
    assert lines == ["[RUN] interrupted by script rerun"]
