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


# --- Phase 0+1 color UI (BP-RP primary vs legacy B-V) ---------------------------------

_BV_RELATED_UI_COL_EXACT = frozenset(
    {
        "b_v",
        "bv_source",
        "b_v_src",
        "b_v_source",
        "target_b_v",
        "target_bv_source",
        "delta_bv_abs",
        "src",
    }
)


def is_bv_related_phase01_ui_column(column_name: object) -> bool:
    """True if this dataframe column is B–V / Johnson legacy metadata (hide when BP-RP is primary)."""
    low = str(column_name).strip().lower()
    if "bp_rp" in low:
        return False
    if low in _BV_RELATED_UI_COL_EXACT:
        return True
    if low.startswith("delta_bv"):
        return True
    if "b-v" in low:
        return True
    return False


def log_if_ui_hiding_bv_for_bprp_primary(*, bprp_primary_ui_active: bool) -> None:
    """Log once when UI actually hides B–V columns (avoids spam on every Streamlit rerun)."""
    import streamlit as st

    from infolog import log_event

    prev = st.session_state.get("_vyvar_ui_bprp_primary_for_bv_log")
    if bprp_primary_ui_active and prev is not True:
        log_event("[UI] B-V stĺpce skryté — BP-RP je primárny filter")
    st.session_state["_vyvar_ui_bprp_primary_for_bv_log"] = bool(bprp_primary_ui_active)
