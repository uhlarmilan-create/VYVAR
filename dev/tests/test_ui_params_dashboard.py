"""Smoke tests for the tiered Parameters dashboard (PARAMS-REGISTRY-UI STEP 3).

Exercises the registry -> widget resolution for every ``widget=auto`` key without a
Streamlit runtime: the planning helper is a pure function that must cover every auto
key and never raise.
"""
from __future__ import annotations

import params_registry as pr
import ui_params_dashboard as upd

_VALID_KINDS = {"checkbox", "number", "select", "text"}


def test_plan_covers_every_auto_key_without_exceptions() -> None:
    reg = pr.load_registry()
    auto_keys = {k for k, e in reg.items() if e.get("widget") == "auto"}

    plan = upd.plan_auto_widgets(reg)

    assert set(plan.keys()) == auto_keys, (
        "widget plan does not cover exactly the widget=auto keys; "
        f"missing={sorted(auto_keys - set(plan))} extra={sorted(set(plan) - auto_keys)}"
    )
    bad = {k: v for k, v in plan.items() if v not in _VALID_KINDS}
    assert not bad, f"widget plan produced invalid kinds: {bad}"


def test_every_auto_key_resolves_individually() -> None:
    reg = pr.load_registry()
    types = pr.appconfig_field_types()
    errors: list[str] = []
    for key, entry in reg.items():
        if entry.get("widget") != "auto":
            continue
        try:
            kind = pr.infer_widget_kind(key, types.get(key, ""), entry)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{key}: raised {exc!r}")
            continue
        if kind not in _VALID_KINDS:
            errors.append(f"{key}: invalid kind {kind!r}")
    assert not errors, "widget resolution errors:\n" + "\n".join(errors)


def test_count_modified_returns_report() -> None:
    import config

    cfg = config.AppConfig()
    n, dev = upd.count_modified(cfg)
    assert isinstance(n, int) and n >= 0
    assert set(dev.keys()) == {"modified", "unknown"}
    assert len(dev["modified"]) == n


def test_owner_groups_partition_every_key() -> None:
    # Ownership grouping (PARAM-OWNERSHIP-WAVE-A STEP 3) must cover every registry key exactly
    # once across the four owners.
    reg = pr.load_registry()
    groups = upd.group_keys_by_owner(reg)
    assert set(groups.keys()) == set(pr.OWNERS)
    flat = [k for keys in groups.values() for k in keys]
    assert sorted(flat) == sorted(reg.keys())
    assert len(flat) == len(set(flat)), "a key landed in more than one owner group"
    # locked distribution seeded from the audit CSV; WAVE-B STEP 2 removed 4 dead
    # config_runtime keys (aperture_fwhm_factor_medium, masterstar_log_astroalign,
    # phase01_comparison_proximity_tiebreak, phase01_comparison_rms_bin_mag): 277 -> 273.
    dist = {o: len(groups[o]) for o in pr.OWNERS}
    assert dist == {"db_static": 9, "config_runtime": 273, "fits_dynamic": 7, "internal": 11}, dist


def test_editable_keys_are_config_runtime_auto_only() -> None:
    reg = pr.load_registry()
    editable = set(upd.editable_config_keys(reg))
    assert editable, "expected at least one editable config key"
    for key in editable:
        assert reg[key]["owner"] == "config_runtime"
        assert reg[key]["widget"] == "auto"
    # editable keys never overlap the read-only / hidden groups
    groups = upd.group_keys_by_owner(reg)
    for owner in ("db_static", "fits_dynamic", "internal"):
        assert not (editable & set(groups[owner])), f"editable set overlaps {owner}"


def test_modified_counter_counts_config_runtime_only() -> None:
    # A db_static deviation (e.g. observer_location_name) must NOT inflate the editable counter.
    import config

    cfg = config.AppConfig()
    cfg.observer_location_name = "SomeSiteThatIsNotDefault"
    n_cfg, dev_cfg = upd.count_modified(cfg, owners=("config_runtime",))
    assert len(dev_cfg["modified"]) == n_cfg
    assert all(
        pr.load_registry()[m["key"]]["owner"] == "config_runtime" for m in dev_cfg["modified"]
    )
    assert "observer_location_name" not in {m["key"] for m in dev_cfg["modified"]}
