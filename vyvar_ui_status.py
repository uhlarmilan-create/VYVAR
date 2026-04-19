"""Spoločná aktualizácia spodného stavového riadku (Streamlit session + rerender)."""

from __future__ import annotations


def vyvar_footer_running(
    process: str,
    status_detail: str,
    *,
    pct: int | None = None,
    current_file: str = "",
    step: str = "",
) -> None:
    import streamlit as st

    prev = st.session_state.get("vyvar_footer_state")
    base: dict = prev if isinstance(prev, dict) else {}
    st.session_state["vyvar_footer_state"] = {
        "running": True,
        "process": str(process)[:200],
        "status_detail": str(status_detail)[:800],
        "pct": pct if pct is not None else base.get("pct"),
        "current_file": (current_file or str(base.get("current_file") or ""))[:500],
        "step": (step or str(base.get("step") or ""))[:200],
    }
    _fn = st.session_state.get("vyvar_ui_rerender_footer")
    if callable(_fn):
        _fn()


def vyvar_footer_idle(
    *,
    process: str = "VYVAR",
    status_detail: str = "Pripravený — spusti úlohu na záložke VAR-STREM.",
) -> None:
    import streamlit as st

    st.session_state["vyvar_footer_state"] = {
        "running": False,
        "process": str(process)[:200],
        "status_detail": str(status_detail)[:800],
        "pct": None,
        "current_file": "",
        "step": "",
    }
    _fn = st.session_state.get("vyvar_ui_rerender_footer")
    if callable(_fn):
        _fn()
