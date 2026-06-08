from unittest.mock import MagicMock

from optics_selection import (
    parse_ui_optics_from_labels,
    resolve_optics_ids_for_platesolve,
)


def test_draft_overrides_stale_caller_equipment_id():
    """OBS_DRAFT optics from import override stale session/UI ids (equipment 1 → 2)."""
    db = MagicMock()
    db.fetch_obs_draft_by_id.return_value = {
        "ID_EQUIPMENTS": 2,
        "ID_TELESCOPE": 2,
    }
    eq, tel = resolve_optics_ids_for_platesolve(
        db, 332, equipment_id=1, telescope_id=1
    )
    assert eq == 2
    assert tel == 2
    db.fetch_obs_draft_by_id.assert_called_once_with(332)


def test_parse_ui_optics_maps_labels():
    opts = {"2: C3-26000 (Camera2)": 2}
    tel = {"2: DDT 300/1200 (Newton)": 2}
    sel = parse_ui_optics_from_labels(
        equipment_label="2: C3-26000 (Camera2)",
        telescope_label="2: DDT 300/1200 (Newton)",
        equipment_options=opts,
        telescope_options=tel,
        eq_labels=list(opts),
        tel_labels=list(tel),
    )
    assert sel.equipment_id == 2
    assert sel.telescope_id == 2
