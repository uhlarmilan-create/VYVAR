"""``comp_color_tiers``: comparison-star colour classes 1-3 (CFG-REVIEW-2026-10).

The next rung is ``comp_max_delta_bprp``; the 4th row old files carry only mirrored it, and the per-row
weight ``w`` never reached the ensemble (Broeg weights). The legacy scalar keys
(``comp_tier{1..4}_bprp_limit`` / ``_weight``) and the other WAVE-B merged keys (``phase01_tiers``,
``aperture_snr_sizing``) are KNOWN_REMOVED_KEYS.
"""
from __future__ import annotations

import json
from pathlib import Path

from config import KNOWN_REMOVED_KEYS, AppConfig

_OLD_SCALARS = (
    "comp_tier1_bprp_limit", "comp_tier2_bprp_limit", "comp_tier3_bprp_limit", "comp_tier4_bprp_limit",
    "comp_tier1_weight", "comp_tier2_weight", "comp_tier3_weight", "comp_tier4_weight",
)
_REMOVED = (
    "phase01_tiers", "aperture_snr_sizing",
    "phase01_tier1_mag", "phase01_tier2_mag", "phase01_tier3_mag", "phase01_tier4_mag",
    "aperture_fwhm_factor_small", "aperture_fwhm_factor_large",
)


def test_defaults_three_colour_classes() -> None:
    cfg = AppConfig()
    assert cfg.comp_tier_bprp_limits() == [0.15, 0.30, 0.55]
    assert cfg.to_json()["comp_color_tiers"] == [{"bprp": 0.15}, {"bprp": 0.3}, {"bprp": 0.55}]


def test_removed_keys_are_known_removed_and_not_saved() -> None:
    cfg = AppConfig()
    payload = cfg.to_json()
    for key in _REMOVED + _OLD_SCALARS:
        assert key in KNOWN_REMOVED_KEYS, key
        assert not hasattr(cfg, key), key
        assert key not in payload, key


def test_old_four_row_table_with_weights_loads_three_classes(tmp_path: Path) -> None:
    data = {"comp_max_delta_bprp": 0.65, "comp_color_tiers": [
        {"bprp": 0.15, "w": 1.0}, {"bprp": 0.3, "w": 0.85}, {"bprp": 0.55, "w": 0.5}, {"bprp": 1.1, "w": 0.25}]}
    (tmp_path / "config.json").write_text(json.dumps(data), encoding="utf-8")
    cfg = AppConfig(project_root=tmp_path)
    assert cfg.comp_tier_bprp_limits() == [0.15, 0.3, 0.55]
    assert cfg.comp_max_delta_bprp == 0.65
    assert cfg.to_json()["comp_color_tiers"] == [{"bprp": 0.15}, {"bprp": 0.3}, {"bprp": 0.55}]


def test_short_table_keeps_defaults(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"comp_color_tiers": [{"bprp": 0.2}]}), encoding="utf-8")
    assert AppConfig(project_root=tmp_path).comp_tier_bprp_limits() == [0.15, 0.30, 0.55]
