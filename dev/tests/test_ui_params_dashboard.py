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
    # config_runtime keys (277 -> 273); STEP 3 internalized frame_width_px/frame_height_px
    # (fits_dynamic 7 -> 5, internal 11 -> 13); STEP 4 merged 14 config_runtime scalars into
    # 3 structured config_runtime keys (comp_color_tiers, phase01_tiers, aperture_snr_sizing):
    # 273 - 14 + 3 = 262. STEP 5 (DELETE-DB-DUP) moved export_arcsec_per_px config_runtime ->
    # fits_dynamic (262 -> 261, fits_dynamic 5 -> 6); the other 8 dup keys were already
    # db_static/fits_dynamic. STEP 6 (HARDCODE) removed 20 config_runtime solver-internal keys
    # (261 -> 241; +2 PER-FRAME-SAT -> 243). PHASE0-IDENTITY-GATE removed phase01_match_radius_arcsec
    # (243 -> 242). batch D added preprocess_sky_surface_force_reapply (242 -> 243);
    # BATCH-E-PARAMS-REGISTRY added 6 config_runtime keys (243 -> 249); total 277.
    # Task D1 added 10 hidden config_runtime normalised companion fields (277 -> 287).
    # ZONE-SIMPLIFY removed 4 config_runtime keys (291 -> 287; config_runtime 263 -> 259).
    # CAL-DIAG removal dropped 5 config_runtime keys (287 -> 282; config_runtime 259 -> 254).
    # L.A.Cosmic removal dropped 3 config_runtime keys (282 -> 279; config_runtime 254 -> 251).
    # Zero-clipping policy dropped 7 clip/CR gate keys (279 -> 272; config_runtime 251 -> 244).
    # COMP-POOL-01 added comp_pool_derived_admission + comparison_stars_pool_n (244 -> 246).
    # COMP-ADMIT-03 added 3 weight coeff keys (246 -> 249).
    # FORCED-PHOT-01 / COMP-WEIGHT-COEFF-01: +4 keys (249 -> 253).
    # IMPL-01: +3 keys (color_level_k, stderr, snr_cog_isolation) (253 -> 256).
    # IMPL-03: +4 aperture scatter selection keys (256 -> 260).
    # ERA-03: +14 MASTERSTAR DAO/pin registry keys (263 -> 277).
    # EPSF-AC-02: +1 config_runtime key psf_ac_policy (277 -> 278).
    dist = {o: len(groups[o]) for o in pr.OWNERS}
    assert dist == {"db_static": 9, "config_runtime": 278, "fits_dynamic": 6, "internal": 13}, dist


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


def test_structured_config_keys_use_text_not_number_widget() -> None:
    """list[float] / dict[str, float] types must not match the scalar float heuristic."""
    reg = pr.load_registry()
    types = pr.appconfig_field_types()
    structured = (
        "comp_color_tiers",
        "phase01_tiers",
        "aperture_snr_sizing",
        "sigma_sys_mag",
    )
    for key in structured:
        kind = pr.infer_widget_kind(key, types.get(key, ""), reg[key])
        assert kind == "text", f"{key} with type {types.get(key)!r} resolved to {kind!r}"


def test_none_scalar_number_uses_text_fallback() -> None:
    """Optional float fields with None must not coerce through float() (field bug #4)."""
    import config

    cfg = config.AppConfig()
    cfg.qc_max_background_rms = None
    reg = pr.load_registry()
    types = pr.appconfig_field_types()
    defaults = pr.appconfig_defaults()
    kind, display, none_fallback = upd.resolve_auto_widget_display(
        cfg, "qc_max_background_rms", reg["qc_max_background_rms"], types, defaults
    )
    assert kind == "text"
    assert display == ""
    assert none_fallback is True


def test_fresh_bootstrap_config_widget_sweep(tmp_path, monkeypatch) -> None:
    """First-run bootstrap must materialize every persisted key and pass widget resolution."""
    import json

    import config
    from vyvar_runtime import _ensure_data_skeleton

    install = tmp_path / "install"
    install.mkdir()
    (install / "RUNTIME_PIN.json").write_text("{}", encoding="ascii")
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv("VYVAR_DATA_DIR", str(data))

    _ensure_data_skeleton(data)
    config.materialize_fresh_config_json(install, data)

    cfg_path = data / "config.json"
    assert cfg_path.is_file(), "bootstrap must write config.json"
    payload = json.loads(config.strip_jsonc_comments(cfg_path.read_text(encoding="utf-8")))
    cfg = config.AppConfig(project_root=install)
    expected = set(cfg.to_json().keys())
    missing = expected - set(payload.keys())
    assert not missing, f"config.json missing persisted keys: {sorted(missing)[:12]}"

    reg = pr.load_registry()
    types = pr.appconfig_field_types()
    defaults = pr.appconfig_defaults()
    plan = upd.plan_auto_widgets(reg)
    errors: list[str] = []
    for key, entry in reg.items():
        if entry.get("widget") != "auto":
            continue
        try:
            kind, _display, none_fallback = upd.resolve_auto_widget_display(
                cfg, key, entry, types, defaults
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{key}: raised {exc!r}")
            continue
        if entry.get("kind") in ("resolved", "derived"):
            if kind != "resolved":
                errors.append(f"{key}: expected resolved, got {kind!r}")
            continue
        planned = plan[key]
        if planned == "number" and kind == "text" and none_fallback:
            continue
        if kind != planned:
            errors.append(f"{key}: plan={planned!r} resolved={kind!r}")
    assert not errors, "fresh-config widget sweep errors:\n" + "\n".join(errors)
