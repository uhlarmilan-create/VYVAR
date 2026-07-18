"""CONFIG-HUMAN-EDIT STEP 3: canonical grouped + commented config.json writer.

The writer emits a deterministic JSONC-lite document (file header, sections in pipeline
order, per-key help comments) whose VALUES round-trip exactly: save -> load -> save is a
fixed point, and parsing the rendered text reproduces the input dict.
"""
from __future__ import annotations

import json

import config
import params_registry as pr


def _sample_payload() -> dict:
    # A spread of phases/tiers + a structured (merged) key + a nested dict value.
    return {
        "sips_dao_fwhm_px": 3.1,
        "qc_min_stars": 7,
        "aperture_fwhm_factor": 1.9,
        "aperture_snr_sizing": {"small": 1.5, "large": 4.0},
        "comp_color_tiers": [
            {"bprp": 0.15, "w": 1.0},
            {"bprp": 0.3, "w": 0.85},
        ],
        "phase01_tiers": [0.5, 1.0, 1.5, 2.0],
        "sysrem_enabled": True,
        "observer_name": "Test Observer",
        "k2_mode": "literature",
    }


def test_render_roundtrips_values_exactly() -> None:
    payload = _sample_payload()
    text = config.render_config_jsonc(payload)
    assert config.parse_config_text(text) == payload


def test_save_load_save_is_fixed_point(tmp_path) -> None:
    payload = _sample_payload()
    with config.ui_config_persist():
        config.save_config_json(tmp_path, payload)
    text1 = (tmp_path / "config.json").read_text(encoding="utf-8")
    reloaded = config.load_config_json(tmp_path)
    assert reloaded == payload
    with config.ui_config_persist():
        config.save_config_json(tmp_path, reloaded)
    text2 = (tmp_path / "config.json").read_text(encoding="utf-8")
    assert text1 == text2


def test_output_has_header_section_and_key_comments() -> None:
    text = config.render_config_jsonc(_sample_payload())
    assert "VYVAR config.json" in text
    assert "validate_config.py" in text
    assert "Observatory" in text
    assert "Resolved Facts" in text
    assert "VYVAR_CONFIG_GUIDE_EN.md" in text
    # a section header and its phase help
    assert "// === Photometry ===" in text
    assert "// === Comparison-star selection ===" in text
    # a per-key help line (ported from the guide) precedes its key
    lines = text.splitlines()
    key_line = next(i for i, ln in enumerate(lines) if ln.strip().startswith('"sips_dao_fwhm_px"'))
    assert lines[key_line - 1].strip().startswith("//"), "key must be preceded by a help comment"


def test_tier_ordering_within_section() -> None:
    # Within a section, basic keys come before advanced before expert.
    reg = pr.load_registry()
    # pick photometry keys of differing tiers that persist
    payload = {
        "aperture_fwhm_factor": 1.9,  # basic
        "err_background_mode": "empirical",  # advanced/expert
        "psf_chi2_threshold": 5.0,
    }
    # keep only keys that exist in registry & their tiers
    payload = {k: v for k, v in payload.items() if k in reg}
    text = config.render_config_jsonc(payload)
    order = [
        ln.split(":")[0].strip().strip('"')
        for ln in text.splitlines()
        if ln.strip().startswith('"')
    ]
    tiers = [reg[k]["tier"] for k in order]
    rank = {"basic": 0, "advanced": 1, "expert": 2}
    ranks = [rank[t] for t in tiers]
    assert ranks == sorted(ranks), f"keys not tier-ordered: {list(zip(order, tiers))}"


def test_unknown_key_goes_to_other_section_and_survives() -> None:
    text = config.render_config_jsonc({"totally_made_up_key": 5})
    assert "// === Other ===" in text
    assert config.parse_config_text(text) == {"totally_made_up_key": 5}


def test_empty_payload_is_valid_json() -> None:
    text = config.render_config_jsonc({})
    assert config.parse_config_text(text) == {}
