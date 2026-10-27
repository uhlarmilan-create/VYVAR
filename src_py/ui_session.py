"""Streamlit session state of the GUI (CORE-NO-UI-2026-10).

The pipeline modules never import Streamlit: they log, report progress through ``progress_cb``
and return their results. What the GUI remembers between pages - the selected camera and
telescope, the processed field of the last job - is read and written only here and in the
``ui_*`` / ``app`` modules.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from optics_selection import VyvarOpticsSelection

SESSION_EQUIPMENT_KEY = "vyvar_active_equipment_id"
SESSION_TELESCOPE_KEY = "vyvar_active_telescope_id"
# Legacy aliases (same values, kept for older session_state readers)
LEGACY_IMPORT_EQUIPMENT_KEY = "vyvar_last_import_equipment_id"
LEGACY_IMPORT_TELESCOPE_KEY = "vyvar_last_import_telescope_id"


def sync_optics_session(selection: VyvarOpticsSelection) -> None:
    """Persist UI optics to Streamlit session (import + platesolve + MASTERSTAR)."""
    st.session_state[SESSION_EQUIPMENT_KEY] = int(selection.equipment_id)
    st.session_state[SESSION_TELESCOPE_KEY] = int(selection.telescope_id)
    st.session_state[LEGACY_IMPORT_EQUIPMENT_KEY] = int(selection.equipment_id)
    st.session_state[LEGACY_IMPORT_TELESCOPE_KEY] = int(selection.telescope_id)


def optics_from_session() -> VyvarOpticsSelection | None:
    """The camera + telescope last selected in the GUI, None when nothing is selected."""
    eq = st.session_state.get(SESSION_EQUIPMENT_KEY)
    if eq is None:
        eq = st.session_state.get(LEGACY_IMPORT_EQUIPMENT_KEY)
    tel = st.session_state.get(SESSION_TELESCOPE_KEY)
    if tel is None:
        tel = st.session_state.get(LEGACY_IMPORT_TELESCOPE_KEY)
    if eq is None or tel is None:
        return None
    try:
        return VyvarOpticsSelection(int(eq), int(tel))
    except (TypeError, ValueError):
        return None


def resolve_draft_dir(
    draft_dir_override: str | Path | None = None,
    draft_id: int | None = None,
    archive_root: str | Path | None = None,
    *,
    drafts_before_session: bool = False,
) -> str | None:
    """
    Draft directory of a GUI page (the pipeline itself uses ``draft_provenance.resolve_draft_dir``).

    Priority chain (default, ``drafts_before_session=False``):
    1. explicit ``draft_dir_override`` (only if existing directory)
    2. ``vyvar_last_job_output["archive_path"]`` from session state
    3. ``vyvar_post_cal_archive_path`` from session state
    4. the folder of processed field ``draft_id`` under ``archive_root/Drafts`` (any name, by manifest)
    5. None

    When ``drafts_before_session=True`` (MASTERSTAR QA): steps 2-3 move after step 4
    (Drafts folder is tried before session archive paths).
    """
    if draft_dir_override is not None:
        p = Path(draft_dir_override).expanduser()
        if p.is_dir():
            return str(p.resolve())

    def _drafts_path() -> str | None:
        if draft_id is None or not archive_root:
            return None
        try:
            from draft_provenance import draft_dir_for_id  # noqa: PLC0415
            return str(draft_dir_for_id(archive_root, int(draft_id)).resolve())
        except (TypeError, ValueError):
            return None

    def _session_archive_paths() -> str | None:
        last_job = st.session_state.get("vyvar_last_job_output") or {}
        if isinstance(last_job, dict):
            jap = last_job.get("archive_path")
            if jap:
                s = str(jap).strip()
                if s:
                    return s
        post_cal = st.session_state.get("vyvar_post_cal_archive_path")
        if post_cal:
            s = str(post_cal).strip()
            if s:
                return s
        return None

    if drafts_before_session:
        p = _drafts_path()
        if p:
            return p
        return _session_archive_paths()

    s = _session_archive_paths()
    if s:
        return s
    return _drafts_path()


def resolve_draft_dir_path(
    draft_dir_override: str | Path | None = None,
    draft_id: int | None = None,
    archive_root: str | Path | None = None,
    *,
    drafts_before_session: bool = False,
) -> Path | None:
    """``resolve_draft_dir`` returning a ``Path`` (or ``None``)."""
    s = resolve_draft_dir(
        draft_dir_override,
        draft_id,
        archive_root,
        drafts_before_session=drafts_before_session,
    )
    if not s:
        return None
    return Path(s).expanduser().resolve()
