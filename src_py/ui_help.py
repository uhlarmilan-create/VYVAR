"""Shared Streamlit expander help used by Settings / DAO-STARS / photometry UI."""

from __future__ import annotations

import streamlit as st


def _detail_help(title: str, *, phase: str, used_in: str, compute: str | None = None) -> None:
    with st.expander(f"? {title}", expanded=False):
        st.markdown(f"**Phase / process:** {phase}")
        st.markdown(f"**Where and how it is used:** {used_in}")
        if compute:
            st.markdown(f"**Derivation / computation:** {compute}")
