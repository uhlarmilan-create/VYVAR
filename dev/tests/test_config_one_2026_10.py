"""CONFIG-ONE-2026-10 (code review 2 T06): one run config, no second defaults.

Static contracts (E3, E4) and the behaviour fixes of the package.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import logging
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src_py"
sys.path.insert(0, str(SRC))

import config  # noqa: E402
from config import AppConfig  # noqa: E402

# task_jobs: G14-04 job child; vyvar_runtime: the entry points' data-dir bootstrap (G12-29: it reads the
# configured database_path)
ENTRY_MODULES = {"app", "night_run", "run_job", "task_jobs", "config", "vyvar_runtime"}


def _defaults() -> dict[str, object]:
    out = {}
    for f in dataclasses.fields(AppConfig):
        if f.name.startswith("_"):
            continue
        if f.default is not dataclasses.MISSING:
            out[f.name] = f.default
    return out


def test_appconfig_built_only_in_entry_points() -> None:
    """E3: library code takes the run's cfg; only the entry points (and the UI layer) build one."""
    bad = []
    for p in sorted(SRC.glob("*.py")):
        m = p.stem
        if m in ENTRY_MODULES or m.startswith("ui_"):
            continue
        for n in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if isinstance(n, ast.Call):
                f = n.func
                name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""
                if name == "AppConfig":
                    bad.append(f"{p.name}:{n.lineno}")
    assert bad == []


def test_numeric_fallbacks_equal_the_default() -> None:
    """E4: a fallback / keyword default named like a config key is the AppConfig default or None."""
    defaults = {k: v for k, v in _defaults().items() if isinstance(v, (int, float)) and not isinstance(v, bool)}

    def differs(k: str, node: ast.AST) -> bool:
        try:
            v = ast.literal_eval(node)
        except (ValueError, SyntaxError, TypeError):
            return False
        if v is None or isinstance(v, bool) or not isinstance(v, (int, float)):
            return False
        d = float(defaults[k])
        return not (math.isnan(d) and math.isnan(float(v))) and float(v) != d

    bad = []
    for p in sorted(SRC.glob("*.py")):
        if p.name == "config.py":
            continue
        for n in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "getattr" and len(n.args) == 3:
                k = n.args[1].value if isinstance(n.args[1], ast.Constant) else None
                if k in defaults and differs(k, n.args[2]):
                    bad.append(f"{p.name}:{n.lineno} getattr({k})")
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "get" and len(n.args) == 2:
                k = n.args[0].value if isinstance(n.args[0], ast.Constant) else None
                if k in defaults and differs(k, n.args[1]):
                    bad.append(f"{p.name}:{n.lineno} .get({k})")
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                a = n.args
                pos = a.posonlyargs + a.args
                pairs = list(zip(pos[len(pos) - len(a.defaults):], a.defaults)) + [
                    (x, d) for x, d in zip(a.kwonlyargs, a.kw_defaults) if d is not None
                ]
                bad += [f"{p.name}:{n.lineno} {n.name}({x.arg})" for x, d in pairs if x.arg in defaults and differs(x.arg, d)]
    assert bad == []


def test_config_file_loads_exactly_that_file(tmp_path: Path) -> None:
    """G12-03: ``--config my.json`` used to load config.json next to it (snr_min stayed 10)."""
    f = tmp_path / "my.json"
    f.write_text('// comment\n{"snr_min": 42}\n', encoding="utf-8")
    assert AppConfig.from_file(f).snr_min == 42.0
    with pytest.raises(FileNotFoundError):
        AppConfig.from_file(tmp_path / "missing.json")


def test_night_run_config_uses_the_file(tmp_path: Path) -> None:
    from night_run import _load_app_config

    f = tmp_path / "other.json"
    f.write_text(json.dumps({"snr_min": 33}), encoding="utf-8")
    assert _load_app_config(f).snr_min == 33.0


def test_gaia_and_objects_default_to_one_file(tmp_path: Path) -> None:
    """G05-15 / G14-13 (Milan 2026-10-27): empty key = the file the builder writes."""
    from pipeline_constants import GAIA_DB_DEFAULT, OBJECTS_DB_DEFAULT

    (tmp_path / "config.json").write_text('{"gaia_db_path": "", "objects_db_path": ""}', encoding="utf-8")
    cfg = AppConfig.from_file(tmp_path / "config.json")
    assert Path(cfg.gaia_db_path) == (Path(cfg.data_root) / GAIA_DB_DEFAULT).resolve()
    assert Path(cfg.objects_db_path) == (Path(cfg.data_root) / OBJECTS_DB_DEFAULT).resolve()


def test_rules_between_keys_one_list(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """G13-08: Settings, loader and validator share config_rule_problems; the loader warns."""
    import settings_model as sm

    vals = {"annulus_inner_fwhm": 1.0, "annulus_outer_fwhm": 5.2, "aperture_fwhm_factor": 1.35}
    assert sm.cross_check(vals) == config.config_rule_problems(vals) != []
    (tmp_path / "config.json").write_text(json.dumps(vals), encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        AppConfig.from_file(tmp_path / "config.json")
    assert any("inner radius must be larger than the aperture" in r.getMessage() for r in caplog.records)


def test_loader_warns_on_typos_and_truncation(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """G13-35 / G13-36: no silent correction."""
    (tmp_path / "config.json").write_text(
        json.dumps({"k2_mode": "litrature", "apply_color_term": "atuo", "masterstar_best_of_n": 10.7,
                    "phase01_comparison_n_comp_min": 6, "phase01_comparison_n_comp_max": 4}),
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING):
        cfg = AppConfig.from_file(tmp_path / "config.json")
    msgs = " ".join(r.getMessage() for r in caplog.records)
    for key in ("k2_mode", "apply_color_term", "masterstar_best_of_n", "phase01_comparison_n_comp_max"):
        assert key in msgs, key
    assert cfg.phase01_comparison_n_comp_max == 6


def test_values_differ_nan_equal_nan() -> None:
    """G13-25."""
    import params_registry as pr

    assert not pr.values_differ(float("nan"), float("nan"))
    assert pr.values_differ(1.0, 2.0)


def test_flat_offset_without_database_raises(tmp_path: Path) -> None:
    """G01-34: a worker without the run database never opens the database of a freshly read config."""
    from astropy.io import fits

    from calibration import _flat_offset_at_use

    with pytest.raises(ValueError, match="no database"):
        _flat_offset_at_use(tmp_path / "flat.fits", fits.Header(), None)


def test_frame_size_has_no_rig_default(tmp_path: Path) -> None:
    """G07-21: 2082 x 1397 px was one camera; an unknown frame size is an error."""
    from photometry_comp import _resolve_frame_hw_px_from_masterstar

    with pytest.raises(RuntimeError, match="frame size unknown"):
        _resolve_frame_hw_px_from_masterstar(tmp_path / "missing.fits")


def test_trust_thresholds_from_the_run_config() -> None:
    """G13-03: no 3 / 12 fallback; GREEN minimum and comparison maximum are the config values."""
    from trust_flag_core import CompTrustThresholds, comp_thresholds_from_config

    cfg = AppConfig()
    expected = CompTrustThresholds.from_bounds(cfg.comp_trust_min_comps, cfg.phase01_comparison_n_comp_max)
    assert comp_thresholds_from_config(cfg) == expected
    assert expected != CompTrustThresholds.from_bounds(3, 12)


def test_database_for_config_uses_the_run_archive(tmp_path: Path) -> None:
    """G13-11: the archive root comes from the run config (config.json is JSONC; json.loads failed)."""
    from database import VyvarDatabase

    cfg = AppConfig()
    cfg.database_path = tmp_path / "vyvar.sqlite3"
    cfg.archive_root = tmp_path / "elsewhere" / "Archive"
    db = VyvarDatabase.for_config(cfg)
    try:
        assert db.resolve_archive_root() == cfg.archive_root.resolve()
    finally:
        db.conn.close()


def test_archive_root_fallback_reads_jsonc(tmp_path: Path) -> None:
    from database import VyvarDatabase

    (tmp_path / "config.json").write_text('// VYVAR\n{"archive_root": "/elsewhere/Archive"}\n', encoding="utf-8")
    db = VyvarDatabase(tmp_path / "vyvar.sqlite3")
    try:
        assert db.resolve_archive_root() == Path("/elsewhere/Archive").resolve()
    finally:
        db.conn.close()
