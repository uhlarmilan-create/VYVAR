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
