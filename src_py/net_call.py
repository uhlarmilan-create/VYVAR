"""Calls to web services: a hard timeout and a per-process "unavailable" hold (G10-36).

A hung web query must not hold a run, and a service that just failed should not make every later
setup wait again. ``call_with_timeout`` runs the call in a daemon thread nobody joins;
``unavailable_reason`` / ``mark_unavailable`` keep one registry of services that failed recently.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

#: Seconds a failed service keeps later calls in this process from waiting again
#: (one wait per run instead of one per setup; the GUI process recovers later).
UNAVAILABLE_HOLD_S = 600.0

_UNAVAILABLE: dict[str, tuple[float, str]] = {}


def call_with_timeout(fn: Callable[[], Any], timeout_s: float, what: str) -> Any:
    """Run ``fn`` in a daemon thread and give up after ``timeout_s`` (TimeoutError).

    The thread is a daemon and nobody waits for it (ThreadPoolExecutor's context manager and
    interpreter exit both join their threads, which made an earlier timeout ineffective).
    """
    box: dict[str, Any] = {}

    def _target() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised in the caller's thread below
            box["error"] = exc

    th = threading.Thread(target=_target, name=f"vyvar-{what}", daemon=True)
    th.start()
    th.join(max(1.0, float(timeout_s)))
    if th.is_alive():
        raise TimeoutError(f"{what} timed out after {timeout_s}s")
    if "error" in box:
        raise box["error"]
    return box.get("value")


def unavailable_reason(service: str) -> str | None:
    """The failure text when ``service`` failed less than UNAVAILABLE_HOLD_S ago, else None."""
    down = _UNAVAILABLE.get(service)
    if down and time.monotonic() - down[0] < UNAVAILABLE_HOLD_S:
        return down[1]
    return None


def mark_unavailable(service: str, exc: BaseException) -> None:
    _UNAVAILABLE[service] = (time.monotonic(), str(exc)[:200])


def clear_unavailable() -> None:
    """Forget every recorded failure (tests; a new GUI session)."""
    _UNAVAILABLE.clear()
