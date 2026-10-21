"""Reusable Streamlit UI components."""

from __future__ import annotations

import logging


_log = logging.getLogger(__name__)

DRAFT_CENTER_RA_STATE_KEY = "cur_draft_ra"
DRAFT_CENTER_DE_STATE_KEY = "cur_draft_de"

# NOTE: render_masterstar_selection_qc / render_photometric_grid_qa are not wired into app.py (inactive).
# persist_draft_center_on_change and draft center state keys remain active (app.py).
# Last audit: 2026-05-18


