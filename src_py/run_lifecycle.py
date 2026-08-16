"""RUN lifecycle helpers (exit visibility + active-RUN gating).

ASCII-only. Used by the Streamlit UI RUN path and unit tests without requiring
a live Streamlit script context for the pure predicates.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any


def is_vyvar_run_active(footer_state: object | None) -> bool:
    """True when the shared footer reports an in-progress RUN/job."""
    if not isinstance(footer_state, dict):
        return False
    return bool(footer_state.get("running"))


def _streamlit_control_flow_exceptions() -> tuple[type[BaseException], ...]:
    try:
        from streamlit.runtime.scriptrunner_utils.exceptions import (  # noqa: PLC0415
            RerunException,
            StopException,
        )

        return (RerunException, StopException)
    except Exception:
        try:
            from streamlit.runtime.scriptrunner.script_runner import (  # noqa: PLC0415
                RerunException,
                StopException,
            )

            return (RerunException, StopException)
        except Exception:
            return ()


def format_run_exit_line(*, ok: bool | None = None, exc: BaseException | None = None) -> str:
    """Build the single durable [RUN] exit line (no I/O)."""
    if exc is not None:
        ctrl = _streamlit_control_flow_exceptions()
        if ctrl and isinstance(exc, ctrl):
            return "[RUN] interrupted by script rerun"
        return f"[RUN] aborted: {type(exc).__name__}: {exc}"
    if ok is True:
        return "[RUN] finished OK"
    if ok is False:
        return "[RUN] aborted: returned False"
    return "[RUN] aborted: unknown"


def run_callable_with_exit_log(
    fn: Callable[[], Any],
    log_fn: Callable[[str], None],
) -> Any:
    """Run ``fn`` and always emit one [RUN] finished/aborted/interrupted line.

    Streamlit script interruption surfaces as RerunException / StopException;
    those are re-raised after logging so Streamlit control flow is preserved.
    """
    exit_line = "[RUN] aborted: unknown"
    try:
        result = fn()
        exit_line = format_run_exit_line(ok=bool(result))
        return result
    except BaseException as exc:
        exit_line = format_run_exit_line(exc=exc)
        raise
    finally:
        try:
            log_fn(exit_line)
        except Exception:
            pass
