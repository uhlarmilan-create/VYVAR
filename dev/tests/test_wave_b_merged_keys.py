"""WAVE-B STEP 4 guard tests: 14 scalar tier/aperture keys merged into 3 structured keys.

Covers: structured defaults, accessor helpers, save payload uses only the new form, and the
config.json loader back-compat that maps legacy scalar keys into the structured keys.
"""
from __future__ import annotations

import json
from pathlib import Path

from config import AppConfig

_OLD_SCALARS = (
    "comp_tier1_bprp_limit", "comp_tier2_bprp_limit", "comp_tier3_bprp_limit", "comp_tier4_bprp_limit",
    "comp_tier1_weight", "comp_tier2_weight", "comp_tier3_weight", "comp_tier4_weight",
    "phase01_tier1_mag", "phase01_tier2_mag", "phase01_tier3_mag", "phase01_tier4_mag",
    "aperture_fwhm_factor_small", "aperture_fwhm_factor_large",
)
_NEW_KEYS = ("comp_color_tiers", "phase01_tiers", "aperture_snr_sizing")


def test_structured_defaults_and_accessors() -> None:
    cfg = AppConfig()
    assert cfg.comp_tier_bprp_limits() == [0.15, 0.30, 0.55, 1.10]
    assert cfg.comp_tier_weights() == [1.00, 0.85, 0.50, 0.25]
    assert cfg.phase01_tier_mags() == [0.50, 1.00, 1.50, 2.00]
    assert cfg.aperture_snr_sizing == {"small": 1.5, "large": 4.0}


def test_save_payload_uses_only_new_form() -> None:
    payload = AppConfig().to_json()
    for key in _NEW_KEYS:
        assert key in payload, f"{key} must be persisted"
    for old in _OLD_SCALARS:
        assert old not in payload, f"{old} must no longer be persisted (WAVE-B STEP 4)"


def test_legacy_scalar_keys_map_into_structured(tmp_path: Path) -> None:
    """A config.json carrying the OLD scalar keys loads them into the structured keys
    (one-transition back-compat), producing identical effective values."""
    legacy = {
        "comp_tier1_bprp_limit": 0.20,
        "comp_tier2_bprp_limit": 0.40,
        "comp_tier3_bprp_limit": 0.60,
        "comp_tier4_bprp_limit": 1.20,
        "comp_tier1_weight": 0.90,
        "comp_tier2_weight": 0.80,
        "comp_tier3_weight": 0.40,
        "comp_tier4_weight": 0.20,
        "aperture_fwhm_factor_small": 1.8,
        "aperture_fwhm_factor_large": 3.6,
    }
    (tmp_path / "config.json").write_text(json.dumps(legacy), encoding="utf-8")
    cfg = AppConfig(project_root=tmp_path)
    assert cfg.comp_tier_bprp_limits() == [0.20, 0.40, 0.60, 1.20]
    assert cfg.comp_tier_weights() == [0.90, 0.80, 0.40, 0.20]
    assert cfg.aperture_snr_sizing == {"small": 1.8, "large": 3.6}
    # save now emits only the new structured form
    payload = cfg.to_json()
    assert all(old not in payload for old in _OLD_SCALARS)


def test_new_structured_keys_win_over_legacy(tmp_path: Path) -> None:
    data = {
        "comp_color_tiers": [
            {"bprp": 0.11, "w": 0.99},
            {"bprp": 0.22, "w": 0.77},
            {"bprp": 0.33, "w": 0.44},
            {"bprp": 0.99, "w": 0.22},
        ],
        "aperture_snr_sizing": {"small": 2.0, "large": 5.0},
        "phase01_tiers": [0.6, 1.2, 1.8, 2.4],
        # legacy keys present but ignored because the structured form is authoritative
        "comp_tier1_bprp_limit": 9.9,
        "aperture_fwhm_factor_small": 9.9,
    }
    (tmp_path / "config.json").write_text(json.dumps(data), encoding="utf-8")
    cfg = AppConfig(project_root=tmp_path)
    assert cfg.comp_tier_bprp_limits() == [0.11, 0.22, 0.33, 0.99]
    assert cfg.phase01_tier_mags() == [0.6, 1.2, 1.8, 2.4]
    assert cfg.aperture_snr_sizing == {"small": 2.0, "large": 5.0}
