"""SETTINGS-2026-10: one Settings page for every config.json key (replaces test_ui_params_dashboard)."""
from __future__ import annotations

import collections
import json

import pytest

import config
import params_registry as pr
import settings_model as sm


@pytest.fixture(scope="module")
def reg() -> dict:
    return pr.load_registry()


@pytest.fixture(scope="module")
def cfg_json() -> dict:
    return config.AppConfig().to_json()


def test_every_config_key_is_on_the_page_once(reg, cfg_json) -> None:
    groups = sm.settings_keys(reg, list(cfg_json))
    flat = [k for keys in groups.values() for k in keys]
    assert sorted(flat) == sorted(cfg_json), "a config.json key is missing on the Settings page or shown twice"
    assert list(groups) == sm.group_order()
    assert set(groups) == set(config._CONFIG_SECTION_ORDER)


def test_every_config_key_has_user_text(reg, cfg_json) -> None:
    for k in cfg_json:
        e = reg[k]
        assert e["widget"] == "auto", k
        assert e["label"] and e["label"] != k and "_" not in e["label"], k
        assert len(e["help"]) >= 20, k
        assert all(ord(c) < 128 for c in e["label"] + e["help"]), k
        for ticket in ("-01", "-02", "-03", "2026-", "CFG-", "STEP"):
            assert ticket not in e["help"], f"{k}: ticket code {ticket!r} in help"


def test_defaults_inside_ranges(reg) -> None:
    defaults = pr.appconfig_defaults()
    for k, e in reg.items():
        r = e.get("range")
        v = defaults.get(k)
        if r and isinstance(v, (int, float)) and not isinstance(v, bool):
            assert r[0] <= v <= r[1], (k, v, r)


def test_widget_kinds(reg, cfg_json) -> None:
    kinds = {k: sm.widget_kind(k, cfg_json[k], reg[k]) for k in cfg_json}
    assert kinds["comp_color_tiers"] == "table"
    assert kinds["aavso_filter_map"] == "table"
    assert kinds["vsx_out_of_scope_types"] == "tags"
    assert kinds["k2_mode"] == "select"
    assert kinds["epsf_auto_run"] == "checkbox"
    assert kinds["masterstar_best_of_n"] == "int"
    assert kinds["aperture_fwhm_factor"] == "float"
    assert set(kinds.values()) <= {"checkbox", "select", "int", "float", "text", "tags", "table"}


@pytest.mark.parametrize("key", ["comp_color_tiers", "aavso_filter_map", "k2_defaults_bprp"])
def test_table_round_trip(key) -> None:
    value = {"comp_color_tiers": [{"bprp": 0.15}, {"bprp": 0.3}, {"bprp": 0.55}],
             "aavso_filter_map": {"NOFILTER": "CV", "G": "TG"},
             "k2_defaults_bprp": {"V": -0.03}}[key]
    back, err = sm.table_value(key, sm.table_rows(key, value))
    assert err is None and back == value


def test_colour_classes_must_increase() -> None:
    rows = sm.table_rows("comp_color_tiers", [{"bprp": 0.3}, {"bprp": 0.15}, {"bprp": 0.55}])
    assert sm.table_value("comp_color_tiers", rows)[1]


def test_cross_checks(cfg_json) -> None:
    assert sm.cross_check(cfg_json) == []
    bad = dict(cfg_json, annulus_outer_fwhm=2.0)
    assert any("outer radius" in e for e in sm.cross_check(bad))
    bad = dict(cfg_json, phase01_comparison_n_comp_max=2, phase01_comparison_n_comp_min=3)
    assert sm.cross_check(bad)


def test_paths_are_never_marked_changed(cfg_json) -> None:
    defaults = pr.appconfig_defaults()
    for k in sm.PATH_KEYS:
        assert not sm.differs_from_default(k, "C:/somewhere/else", defaults)


def test_owner_distribution(reg) -> None:
    dist = collections.Counter(e["owner"] for e in reg.values())
    # SETTINGS-2026-10: observer_name / observer_code / aavso_filter_map are config settings
    # (db_static 8 -> 5); observer_location_id is run time only (hidden, db_static).
    # DAO-FALSE-2026-10: masterstar_prematch_peak_sigma_floor removed (config_runtime 108 -> 107).
    # COMP-CATOFF-2026-10: comp_catalog_offset_max_mag added (107 -> 108).
    # CODE-REVIEW step 2: comp_pool_derived_admission, photon_transfer_ci_max_width_factor,
    # trust_flag_enabled removed (108 -> 105); both plate scales removed (fits_dynamic 5 -> 3).
    # CODE-REVIEW step 7: 58 fixed constants / removed keys (config_runtime 105 -> 47),
    # plate_solve_fov_deg removed (fits_dynamic 3 -> 2), frame_width/height_px removed (internal 9 -> 7).
    # CODE-REVIEW step 7B: isolation radius, chip margin, sips_dao_fwhm_px removed (47 -> 44),
    # gain / read_noise removed (fits_dynamic 2 -> 0).
    assert dict(dist) == {"db_static": 5, "config_runtime": 44, "internal": 7}, dist


def test_site_is_not_stored_in_config(cfg_json) -> None:
    assert "observer_location_id" not in cfg_json
    assert "observer_location_id" in config.KNOWN_REMOVED_KEYS


def test_masterstar_suggestions() -> None:
    s = sm.masterstar_suggestions(2.0, 900)
    assert s == {"aperture_fwhm_factor": 1.5, "detection_threshold_sigma": 2.0}


def test_render_config_jsonc_round_trip(cfg_json) -> None:
    txt = config.render_config_jsonc(cfg_json)
    back = config.parse_config_text(txt)
    assert json.dumps(back, sort_keys=True, default=str) == json.dumps(cfg_json, sort_keys=True, default=str)
    assert "// === Other ===" not in txt
