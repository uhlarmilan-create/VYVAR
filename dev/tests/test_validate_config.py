"""CONFIG-HUMAN-EDIT STEP 4: standalone config.json validator tests."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]


def _load_validator():
    spec = importlib.util.spec_from_file_location(
        "validate_config", _REPO / "dev" / "scripts" / "validate_config.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


VC = _load_validator()


def _errors(text: str) -> list[str]:
    problems, _ = VC.validate_text(text)
    return [m for sev, m in problems if sev == "ERROR"]


def test_valid_config_has_no_errors() -> None:
    assert _errors('{"qc_min_stars": 7}') == []


def test_comment_tolerant() -> None:
    assert _errors('{\n  // a comment\n  "qc_min_stars": 7\n}') == []


def test_syntax_error_reports_line() -> None:
    problems, data = VC.validate_text('{\n  "a": 1,,\n}')
    assert data is None
    assert any("syntax error at line" in m for _, m in problems)


def test_unknown_key_suggests_closest() -> None:
    errs = _errors('{"qc_min_star": 5}')
    assert any("unknown key 'qc_min_star'" in e and "qc_min_stars" in e for e in errs)


def test_out_of_range_value_flagged() -> None:
    errs = _errors('{"alignment_max_stars": 99999}')
    assert any("outside allowed range" in e for e in errs)


def test_type_mismatch_flagged() -> None:
    errs = _errors('{"qc_min_stars": "seven"}')
    assert any("expected int" in e for e in errs)


def test_optional_null_accepted() -> None:
    assert _errors('{"qc_max_background_rms": null}') == []


def test_main_returns_nonzero_on_broken_fixture(tmp_path, capsys) -> None:
    broken = tmp_path / "config.json"
    broken.write_text(
        "{\n"
        '  "qc_min_star": 5,\n'          # unknown (typo)
        '  "alignment_max_stars": 99999,\n'  # out of range
        '  "qc_min_stars": "seven"\n'     # type mismatch
        "}\n",
        encoding="utf-8",
    )
    rc = VC.main([str(broken)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "FAIL" in out
    assert "unknown key" in out
    assert "outside allowed range" in out
    assert "expected int" in out


def test_main_returns_zero_on_valid_fixture(tmp_path) -> None:
    good = tmp_path / "config.json"
    good.write_text('{\n  // fine\n  "qc_min_stars": 7\n}\n', encoding="utf-8")
    assert VC.main([str(good)]) == 0
