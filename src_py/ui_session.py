"""Streamlit session state of the GUI (CORE-NO-UI-2026-10).

The pipeline modules never import Streamlit: they log, report progress through ``progress_cb``
and return their results. What the GUI remembers between pages - the selected camera and
telescope, the processed field of the last job - is read and written only here and in the
``ui_*`` / ``app`` modules.
"""

from __future__ import annotations

from dataclasses import dataclass
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


DRAFT_OVERRIDE_KEY = "vyvar_draft_dir_override"
DRAFT_ID_KEY = "vyvar_last_draft_id"


@dataclass(frozen=True)
class ActiveDraft:
    """The processed field every Pipeline view shows (G14-12: one resolver, computed once in ``main``)."""

    draft_id: int | None
    draft_dir: Path | None


def set_active_draft(draft_dir: Path | str | None, draft_id: int | None) -> None:
    """Load a processed field (Load button, finished RUN): folder and id are set together."""
    if draft_dir is None:
        st.session_state.pop(DRAFT_OVERRIDE_KEY, None)
    else:
        st.session_state[DRAFT_OVERRIDE_KEY] = str(Path(draft_dir).expanduser().resolve())
    if draft_id is None:
        st.session_state.pop(DRAFT_ID_KEY, None)
    else:
        st.session_state[DRAFT_ID_KEY] = int(draft_id)


def clear_active_draft() -> None:
    """Load with an empty field: no processed field (folder and id both cleared)."""
    set_active_draft(None, None)


def active_draft(archive_root: str | Path) -> ActiveDraft:
    """The loaded processed field: the loaded folder (with its manifest id), else ``Drafts/<id>``.

    The GUI's job outputs (last job, post-calibration path, import result) are never used: a finished
    RUN loads its field with :func:`set_active_draft`. An id whose folder is missing gives
    ``draft_dir=None`` (the views say so)."""
    from draft_provenance import draft_dir_for_id  # noqa: PLC0415
    from processed_field import processed_field_id  # noqa: PLC0415

    raw = st.session_state.get(DRAFT_OVERRIDE_KEY)
    if raw:
        p = Path(str(raw)).expanduser()
        if p.is_dir():
            return ActiveDraft(processed_field_id(p), p.resolve())
    did = st.session_state.get(DRAFT_ID_KEY)
    if did is None:
        return ActiveDraft(None, None)
    d = draft_dir_for_id(archive_root, int(did))
    return ActiveDraft(int(did), d.resolve() if d.is_dir() else None)


ACTIVE_JOB_KEY = "vyvar_active_job"


def any_job_running(data_root: str | Path) -> bool:
    """G14-37: a job of this data directory runs (any browser session); every start button is off."""
    from run_job import running_job  # noqa: PLC0415

    return running_job(data_root) is not None


def start_task_job(
    data_root: str | Path,
    task: str,
    args: dict,
    *,
    label: str,
    on_done_pop: tuple[str, ...] = (),
) -> None:
    """Start a ``task_jobs`` task as a job process and follow it with the app's job monitor (G14-04 /
    G14-14). ``on_done_pop``: session keys dropped when the job ends. Shows the error when another
    job of the data directory runs."""
    from run_job import JobBusyError  # noqa: PLC0415
    from task_jobs import launch_task  # noqa: PLC0415

    try:
        jd = launch_task(task, args, data_root, label=label)
    except JobBusyError as exc:
        st.error(str(exc))
        return
    st.session_state[ACTIVE_JOB_KEY] = {
        "job_dir": str(jd), "label": label, "kind": task, "applied": False, "on_done_pop": list(on_done_pop),
    }
    st.rerun()
