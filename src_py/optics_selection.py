"""Active camera + telescope selection - one source of truth for VYVAR.

Priority when resolving optics for platesolve / calibration / masters:
1. ``draft manifest`` row (frozen at import) when ``draft_id`` is set
2. Explicit UI combobox ids passed by the caller
3. Streamlit session keys (synced on every Varstream page render)

Never silently fall back to hardcoded equipment id=1 (QHY294MM).
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from typing import Any

from infolog import log_event

_log = logging.getLogger(__name__)

SESSION_EQUIPMENT_KEY = "vyvar_active_equipment_id"
SESSION_TELESCOPE_KEY = "vyvar_active_telescope_id"
# Legacy aliases (same values, kept for older session_state readers)
LEGACY_IMPORT_EQUIPMENT_KEY = "vyvar_last_import_equipment_id"
LEGACY_IMPORT_TELESCOPE_KEY = "vyvar_last_import_telescope_id"


@dataclass(frozen=True, slots=True)
class VyvarOpticsSelection:
    equipment_id: int
    telescope_id: int
    equipment_label: str = ""
    telescope_label: str = ""


def _first_db_optics_ids(db: Any) -> tuple[int | None, int | None]:
    """First row with positive PIXELSIZE / FOCAL - not assumed id=1."""
    if db is None:
        return None, None
    eq_id: int | None = None
    tel_id: int | None = None
    try:
        row = db.conn.execute(
            """
            SELECT ID FROM EQUIPMENTS
            WHERE PIXELSIZE IS NOT NULL AND PIXELSIZE > 0
            ORDER BY ID
            LIMIT 1;
            """
        ).fetchone()
        if row is not None:
            eq_id = int(row["ID"])
    except (sqlite3.Error, AttributeError, TypeError, ValueError, KeyError) as exc:
        _log.warning(
            "EQUIPMENTS optics lookup failed (%s); equipment_id resolution fell through to None",
            exc,
        )
    try:
        row = db.conn.execute(
            """
            SELECT ID FROM TELESCOPE
            WHERE FOCAL IS NOT NULL AND FOCAL > 0
            ORDER BY ID
            LIMIT 1;
            """
        ).fetchone()
        if row is not None:
            tel_id = int(row["ID"])
    except (sqlite3.Error, AttributeError, TypeError, ValueError, KeyError) as exc:
        _log.warning(
            "TELESCOPE optics lookup failed (%s); telescope_id resolution fell through to None",
            exc,
        )
    return eq_id, tel_id


def parse_ui_optics_from_labels(
    *,
    equipment_label: str,
    telescope_label: str,
    equipment_options: dict[str, int],
    telescope_options: dict[str, int],
    eq_labels: list[str],
    tel_labels: list[str],
    db: Any = None,
) -> VyvarOpticsSelection:
    """Map Session Upload combobox labels -> ids. Avoids silent ``else 1``."""
    if eq_labels and equipment_label in equipment_options:
        eq_id = int(equipment_options[equipment_label])
        eq_lbl = str(equipment_label)
    else:
        eq_id, _ = _first_db_optics_ids(db)
        if eq_id is None:
            raise ValueError(
                "Vyberte platnu kameru v Equipment (library) - v DB nie je ziadna kamera s PIXELSIZE."
            )
        eq_lbl = f"(DB fallback id={eq_id})"

    if tel_labels and telescope_label in telescope_options:
        tel_id = int(telescope_options[telescope_label])
        tel_lbl = str(telescope_label)
    else:
        _, tel_id = _first_db_optics_ids(db)
        if tel_id is None:
            raise ValueError(
                "Vyberte platny dalekohlad v Telescope (library) - v DB nie je ziadny s FOCAL."
            )
        tel_lbl = f"(DB fallback id={tel_id})"

    return VyvarOpticsSelection(
        equipment_id=eq_id,
        telescope_id=tel_id,
        equipment_label=eq_lbl,
        telescope_label=tel_lbl,
    )


def sync_optics_session(selection: VyvarOpticsSelection) -> None:
    """Persist UI optics to Streamlit session (import + platesolve + MASTERSTAR)."""
    try:
        import streamlit as st
    except ImportError:
        return
    st.session_state[SESSION_EQUIPMENT_KEY] = int(selection.equipment_id)
    st.session_state[SESSION_TELESCOPE_KEY] = int(selection.telescope_id)
    st.session_state[LEGACY_IMPORT_EQUIPMENT_KEY] = int(selection.equipment_id)
    st.session_state[LEGACY_IMPORT_TELESCOPE_KEY] = int(selection.telescope_id)


def optics_from_session() -> VyvarOpticsSelection | None:
    try:
        import streamlit as st
    except ImportError:
        return None
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


def resolve_optics_ids_for_platesolve(
    db: Any,
    draft_id: int | None,
    *,
    equipment_id: int | None = None,
    telescope_id: int | None = None,
) -> tuple[int | None, int | None]:
    """``draft manifest`` optics override caller ids when ``draft_id`` is set."""
    eq = int(equipment_id) if equipment_id is not None else None
    tel = int(telescope_id) if telescope_id is not None else None
    if db is None or draft_id is None:
        return eq, tel
    try:
        dr = db.fetch_obs_draft_by_id(int(draft_id))
    except Exception as exc:  # noqa: BLE001
        from except_fix_counters import get_except_fix_counters

        get_except_fix_counters().optics_draft_override_read_fail += 1
        _log.error(
            "[OPTICS] draft manifest override read failed for draft_id=%s: %s",
            draft_id,
            exc,
        )
        return eq, tel
    if dr is None:
        return eq, tel
    d_eq = dr.get("ID_EQUIPMENTS")
    d_tel = dr.get("ID_TELESCOPE")
    if d_eq is not None:
        d_eq_i = int(d_eq)
        if eq is not None and eq != d_eq_i:
            log_event(
                f"OPTICS: equipment_id={eq} -> draft {int(draft_id)} ID_EQUIPMENTS={d_eq_i} "
                "(draft z importu ma prednost pred session/UI)."
            )
        eq = d_eq_i
    if d_tel is not None:
        d_tel_i = int(d_tel)
        if tel is not None and tel != d_tel_i:
            log_event(
                f"OPTICS: telescope_id={tel} -> draft {int(draft_id)} ID_TELESCOPE={d_tel_i}."
            )
        tel = d_tel_i
    return eq, tel


def resolve_working_optics(
    db: Any,
    *,
    draft_id: int | None = None,
    ui: VyvarOpticsSelection | None = None,
    context: str = "VYVAR",
) -> VyvarOpticsSelection:
    """Resolve ids for pipeline work: UI/session base, draft overlay when present."""
    base = ui or optics_from_session()
    if base is None:
        eq, tel = _first_db_optics_ids(db)
        if eq is None or tel is None:
            raise ValueError(
                f"{context}: chyba vyber kamery/dalekohladu - nastavte Equipment a Telescope v UI."
            )
        base = VyvarOpticsSelection(equipment_id=int(eq), telescope_id=int(tel))
    eq, tel = resolve_optics_ids_for_platesolve(
        db,
        draft_id,
        equipment_id=base.equipment_id,
        telescope_id=base.telescope_id,
    )
    if eq is None or tel is None:
        raise ValueError(f"{context}: nepodarilo sa urcit ID kamery alebo dalekohladu.")
    return VyvarOpticsSelection(
        equipment_id=int(eq),
        telescope_id=int(tel),
        equipment_label=base.equipment_label,
        telescope_label=base.telescope_label,
    )


def log_active_optics(
    db: Any,
    selection: VyvarOpticsSelection,
    *,
    draft_id: int | None = None,
    context: str = "OPTICS",
) -> None:
    """Infolog: active camera pixel + focal for verification."""
    pix = focal = None
    if db is not None:
        try:
            pix = db.get_equipment_pixel_size_um(int(selection.equipment_id))
        except Exception:  # noqa: BLE001
            pix = None
        try:
            focal = db.get_telescope_focal_mm(int(selection.telescope_id))
        except Exception:  # noqa: BLE001
            focal = None
    draft_note = f", draft_id={int(draft_id)}" if draft_id is not None else ""
    log_event(
        f"{context}: kamera id={selection.equipment_id} ({selection.equipment_label or '?'}) "
        f"PIXELSIZE={pix if pix is not None else '?'} um | "
        f"dalekohlad id={selection.telescope_id} ({selection.telescope_label or '?'}) "
        f"FOCAL={focal if focal is not None else '?'} mm{draft_note}"
    )
