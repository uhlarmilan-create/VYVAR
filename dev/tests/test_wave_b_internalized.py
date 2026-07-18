"""WAVE-B guard tests: parameters that left the config.json persistence surface.

STEP 3 internalized ``frame_width_px`` / ``frame_height_px`` (they remain AppConfig
fields resolved from FITS NAXIS at run time, but are neither saved to nor loaded from
config.json). This test pins that the save payload never carries them.
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


def test_internalized_frame_dims_still_appconfig_fields() -> None:
    # They stay as fields (5 real read sites) with their fallback defaults intact.
    cfg = AppConfig()
    assert cfg.frame_width_px == 2082
    assert cfg.frame_height_px == 1397


# WAVE-B STEP 5 (DELETE-DB-DUP): 9 DB/FITS-authoritative keys leave config.json persistence
# but remain AppConfig fields (run-time hydrated mirrors).
_DB_DUP_KEYS = (
    "gain",
    "read_noise",
    "plate_scale_arcsec_per_px",
    "phase01_plate_scale_arcsec_per_px",
    "observer_lat",
    "observer_lon",
    "observer_alt_m",
    "observer_location_name",
    "export_arcsec_per_px",
)


def test_db_dup_keys_absent_from_save_payload() -> None:
    cfg = AppConfig()
    payload = cfg.to_json()
    for key in _DB_DUP_KEYS:
        assert key not in payload, f"{key} must not be persisted to config.json (WAVE-B STEP 5)"


def test_db_dup_keys_still_appconfig_fields() -> None:
    cfg = AppConfig()
    for key in _DB_DUP_KEYS:
        assert hasattr(cfg, key), f"{key} must remain an AppConfig field (hydrated mirror)"
