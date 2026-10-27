"""CODE-REVIEW-2026-10 step 7: table-driven config.json loader.

Every numeric key has ONE range (config.PARAM_RANGES): the loader clamps to it with a warning,
a value that is not a number keeps the default with a warning; on / off keys accept JSON
booleans (and 0/1, "true"/"false"). The Settings page and the validator read the same range
through params_registry.
"""
from __future__ import annotations

import pytest
import dataclasses
import json
import logging

import config
import params_registry as pr

pytestmark = pytest.mark.usefixtures("tmp_path_is_data_root")  # TEST-ISOLATION-2026-10


def _load(tmp_path, payload: dict) -> config.AppConfig:
    (tmp_path / "config.json").write_text(json.dumps(payload), encoding="utf-8")
    return config.AppConfig(project_root=tmp_path)


def test_every_numeric_config_key_has_a_range() -> None:
    cfg = config.AppConfig()
    numeric = {
        k for k, v in cfg.to_json().items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    assert numeric == set(config.PARAM_RANGES)
    bools = {k for k, v in cfg.to_json().items() if isinstance(v, bool)}
    assert bools == set(config.BOOL_KEYS)


def test_defaults_lie_inside_their_range() -> None:
    defaults = {f.name: f.default for f in dataclasses.fields(config.AppConfig)}
    for key, (lo, hi) in config.PARAM_RANGES.items():
        assert lo <= defaults[key] <= hi, key


def test_out_of_range_is_clamped_with_a_warning(tmp_path, caplog) -> None:
    with caplog.at_level(logging.WARNING):
        cfg = _load(tmp_path, {"snr_min": 1.0, "alignment_max_stars": 99999})
    assert cfg.snr_min == 3.0
    assert cfg.alignment_max_stars == 5000
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "snr_min" in msgs and "alignment_max_stars" in msgs


def test_not_a_number_keeps_the_default_with_a_warning(tmp_path, caplog) -> None:
    with caplog.at_level(logging.WARNING):
        cfg = _load(tmp_path, {"aperture_fwhm_factor": "wide", "masterstar_best_of_n": None})
    assert cfg.aperture_fwhm_factor == 1.35
    assert cfg.masterstar_best_of_n == 10
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "aperture_fwhm_factor" in msgs and "masterstar_best_of_n" in msgs


def test_types_follow_the_default(tmp_path) -> None:
    cfg = _load(tmp_path, {"masterstar_best_of_n": 7.0, "snr_min": 12})
    assert cfg.masterstar_best_of_n == 7 and isinstance(cfg.masterstar_best_of_n, int)
    assert cfg.snr_min == 12.0 and isinstance(cfg.snr_min, float)


def test_bool_words(tmp_path, caplog) -> None:
    cfg = _load(tmp_path, {"tess_enabled": "false", "epsf_auto_run": 1, "save_lightcurve_png": True})
    assert cfg.tess_enabled is False and cfg.epsf_auto_run is True and cfg.save_lightcurve_png is True
    with caplog.at_level(logging.WARNING):
        cfg = _load(tmp_path, {"tess_enabled": "maybe"})
    assert cfg.tess_enabled is False
    assert "tess_enabled" in " ".join(r.getMessage() for r in caplog.records)


def test_rules_between_keys(tmp_path) -> None:
    cfg = _load(tmp_path, {"annulus_inner_fwhm": 6.0, "annulus_outer_fwhm": 5.0,
                           "phase01_comparison_n_comp_min": 6, "phase01_comparison_n_comp_max": 4,
                           "comp_trust_min_comps": 9})
    assert cfg.annulus_outer_fwhm == 7.0
    assert cfg.phase01_comparison_n_comp_max == 6
    assert cfg.comp_trust_min_comps == 6


def test_removed_keys_load_silently_with_info(tmp_path, caplog) -> None:
    with caplog.at_level(logging.INFO):
        cfg = _load(tmp_path, {"lc_outlier_n_sigma": 7.0, "hrd_simbad_enrich_enabled": False})
    assert not hasattr(cfg, "lc_outlier_n_sigma")
    warn = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert not warn, [r.getMessage() for r in warn]
    assert "step 7" in " ".join(r.getMessage() for r in caplog.records)


def test_registry_range_is_the_code_range() -> None:
    reg = pr.load_registry()
    for key, entry in reg.items():
        rng = config.PARAM_RANGES.get(key)
        assert entry["range"] == (list(rng) if rng else None), key
    raw = json.loads(pr.REGISTRY_PATH.read_text(encoding="utf-8"))
    assert not any("range" in v for k, v in raw.items() if not k.startswith("__"))
