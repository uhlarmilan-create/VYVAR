"""WAVE-B guard tests: parameters that left the config.json persistence surface.

STEP 3 internalized ``frame_width_px`` / ``frame_height_px``; CODE-REVIEW step 7 removed
them (the frame size always comes from FITS NAXIS). This test pins that the save payload
never carries them.
"""
from __future__ import annotations

from config import AppConfig

_INTERNALIZED_KEYS = ("frame_width_px", "frame_height_px")


def test_internalized_frame_dims_absent_from_save_payload() -> None:
    cfg = AppConfig()
    payload = cfg.to_json()  # exact dict save_config_json persists
    for key in _INTERNALIZED_KEYS:
        assert key not in payload, f"{key} must not be persisted to config.json (WAVE-B STEP 3)"
    # to_dict is the alias used by report/snapshot callers; same guarantee.
    assert all(k not in cfg.to_dict() for k in _INTERNALIZED_KEYS)


# WAVE-B STEP 5 (DELETE-DB-DUP): DB-authoritative site keys leave config.json persistence but
# remain AppConfig fields (run-time hydrated mirrors). gain / read_noise were removed in
# CODE-REVIEW step 7B (header / EQUIPMENTS only).
_DB_DUP_KEYS = (
    "observer_lat",
    "observer_lon",
    "observer_alt_m",
    "observer_location_name",
)


def test_db_dup_keys_absent_from_save_payload() -> None:
    cfg = AppConfig()
    payload = cfg.to_json()
    for key in _DB_DUP_KEYS:
        assert key not in payload, f"{key} must not be persisted to config.json (WAVE-B STEP 5)"


# CODE-REVIEW-2026-10 step 2: the plate scale comes only from the WCS (no AppConfig mirror).
_REMOVED_PLATE_SCALE_KEYS = ("plate_scale_arcsec_per_px", "phase01_plate_scale_arcsec_per_px")


def test_plate_scale_keys_removed() -> None:
    cfg = AppConfig()
    for key in _REMOVED_PLATE_SCALE_KEYS:
        assert not hasattr(cfg, key) and key not in cfg.to_json()


def test_db_dup_keys_still_appconfig_fields() -> None:
    cfg = AppConfig()
    for key in _DB_DUP_KEYS:
        assert hasattr(cfg, key), f"{key} must remain an AppConfig field (hydrated mirror)"
