"""CONFIG-HUMAN-EDIT STEP 2: comment-tolerant loader + unknown-key warnings.

The loader must strip ``//`` line comments that sit OUTSIDE string literals (so a strict
JSON file and the same file annotated with comments produce an identical AppConfig), and
must WARN (never raise) on unknown keys with a difflib suggestion -- while staying silent
on migrated legacy aliases.
"""
from __future__ import annotations

import json
import logging

import config


def test_strip_preserves_slashes_inside_strings() -> None:
    src = '{"url": "http://example.com/x", "path": "a//b"}'
    assert config.strip_jsonc_comments(src) == src
    assert config.parse_config_text(src) == {"url": "http://example.com/x", "path": "a//b"}


def test_strip_removes_line_and_trailing_comments() -> None:
    src = (
        "{\n"
        "  // full-line comment\n"
        '  "a": 1, // trailing comment with // and "quotes"\n'
        '  "b": "http://keep"\n'
        "}\n"
    )
    parsed = config.parse_config_text(src)
    assert parsed == {"a": 1, "b": "http://keep"}


def test_strip_preserves_line_count() -> None:
    src = '{\n  "a": 1 // c\n}\n'
    stripped = config.strip_jsonc_comments(src)
    assert stripped.count("\n") == src.count("\n")


def test_strict_and_commented_files_yield_identical_appconfig(tmp_path) -> None:
    payload = {
        "sips_dao_fwhm_px": 3.1,
        "qc_min_stars": 7,
        "sysrem_enabled": True,
        "aperture_fwhm_factor": 2.2,
    }
    cfg_path = tmp_path / "config.json"
    # Same project_root for both so path-derived fields are identical; only the on-disk
    # config.json format (strict vs commented) differs between the two loads.
    cfg_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    strict_json = config.AppConfig(project_root=tmp_path).to_json()

    commented_text = (
        "{\n"
        "  // detection FWHM used for DAO star finding\n"
        '  "sips_dao_fwhm_px": 3.1,\n'
        '  "qc_min_stars": 7, // minimum stars to accept a frame\n'
        '  "sysrem_enabled": true,\n'
        '  "aperture_fwhm_factor": 2.2 // base aperture radius\n'
        "}\n"
    )
    cfg_path.write_text(commented_text, encoding="utf-8")
    commented_json = config.AppConfig(project_root=tmp_path).to_json()

    assert strict_json == commented_json


def test_unknown_key_warns_with_suggestion(tmp_path, caplog) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"qc_min_star": 5}), encoding="utf-8"  # typo: missing trailing 's'
    )
    with caplog.at_level(logging.WARNING):
        data = config.load_config_json(tmp_path)
    assert "qc_min_star" in data  # not dropped, just warned
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "unknown key 'qc_min_star'" in msgs
    assert "qc_min_stars" in msgs  # difflib suggestion


def test_legacy_alias_keys_do_not_warn(tmp_path, caplog) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"comp_tier1_bprp_limit": 0.15, "GAIA_DB_PATH": "x.sqlite"}),
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING):
        config.load_config_json(tmp_path)
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "unknown key" not in msgs


def test_malformed_json_returns_empty_and_warns(tmp_path, caplog) -> None:
    (tmp_path / "config.json").write_text('{"a": 1,,}', encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        data = config.load_config_json(tmp_path)
    assert data == {}
    assert "could not be parsed" in " ".join(r.getMessage() for r in caplog.records)
