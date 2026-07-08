"""G7-F003c: PDF report uses pipeline_meta provenance config_snapshot."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import cm, mm

from photometry_report import (
    _PhotometryReportBuilder,
    resolve_report_config,
)


def _patch_appconfig_root(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    def _factory(**kwargs: object):
        from config import AppConfig

        return AppConfig(project_root=root)

    monkeypatch.setattr("config.AppConfig", _factory)
    monkeypatch.setattr("photometry_report.AppConfig", _factory, raising=False)


def _minimal_builder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _PhotometryReportBuilder:
    obs_group = "NoFilter_60_2"
    phot = tmp_path / "platesolve" / obs_group / "photometry"
    phot.mkdir(parents=True)
    pd.DataFrame({"catalog_id": ["486430957815961344"], "lc_rms": [0.05]}).to_csv(
        phot / "photometry_summary.csv", index=False
    )
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    _patch_appconfig_root(monkeypatch, tmp_path)
    return _PhotometryReportBuilder(
        draft_dir=tmp_path,
        obs_group=obs_group,
        output_pdf=tmp_path / "report.pdf",
        var_results=None,
        candidates=None,
        crossmatch_bullets=None,
        accepted_periods=None,
        variability_timestamp=None,
        report_draft_label=None,
        tess_results=None,
        report_title="Test report",
        font_reg="Helvetica",
        font_bold="Helvetica-Bold",
        font_obl="Helvetica-Oblique",
        colors_mod=colors,
        cm_mod=cm,
        mm_mod=mm,
        landscape_fn=landscape,
        a4_size=A4,
        canvas_mod=MagicMock(),
        image_reader_mod=MagicMock(),
        table_mod=MagicMock(),
        table_style_mod=MagicMock(),
        paragraph_mod=MagicMock(),
        paragraph_style_mod=MagicMock(),
        ta_left_mod=MagicMock(),
    )


def test_resolve_report_config_uses_snapshot_not_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"phase01_use_bprp_primary": True, "aperture_comp_factor": 9.9}),
        encoding="utf-8",
    )
    _patch_appconfig_root(monkeypatch, tmp_path)
    meta = {
        "provenance": {
            "config_snapshot": {
                "phase01_use_bprp_primary": False,
                "aperture_comp_factor": 1.25,
            }
        }
    }
    cfg, label = resolve_report_config(meta)
    assert label == "run snapshot"
    assert bool(getattr(cfg, "phase01_use_bprp_primary")) is False
    assert float(getattr(cfg, "aperture_comp_factor")) == 1.25


def test_builder_uses_snapshot_decoupled_from_live_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    obs_group = "NoFilter_60_2"
    phot = tmp_path / "platesolve" / obs_group / "photometry"
    phot.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"catalog_id": ["486430957815961344"], "lc_rms": [0.05]}).to_csv(
        phot / "photometry_summary.csv", index=False
    )
    (phot / "pipeline_meta.json").write_text(
        json.dumps(
            {
                "provenance": {
                    "config_snapshot": {"phase01_use_bprp_primary": False},
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "config.json").write_text(
        json.dumps({"phase01_use_bprp_primary": True}),
        encoding="utf-8",
    )
    _patch_appconfig_root(monkeypatch, tmp_path)
    builder = _PhotometryReportBuilder(
        draft_dir=tmp_path,
        obs_group=obs_group,
        output_pdf=tmp_path / "report.pdf",
        var_results=None,
        candidates=None,
        crossmatch_bullets=None,
        accepted_periods=None,
        variability_timestamp=None,
        report_draft_label=None,
        tess_results=None,
        report_title="Test report",
        font_reg="Helvetica",
        font_bold="Helvetica-Bold",
        font_obl="Helvetica-Oblique",
        colors_mod=colors,
        cm_mod=cm,
        mm_mod=mm,
        landscape_fn=landscape,
        a4_size=A4,
        canvas_mod=MagicMock(),
        image_reader_mod=MagicMock(),
        table_mod=MagicMock(),
        table_style_mod=MagicMock(),
        paragraph_mod=MagicMock(),
        paragraph_style_mod=MagicMock(),
        ta_left_mod=MagicMock(),
    )
    assert builder._cfg_source_label == "run snapshot"
    assert builder._use_bprp_primary is False


def test_builder_without_snapshot_labels_live_footer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _minimal_builder(tmp_path, monkeypatch)
    assert builder._cfg_source_label == "live (no run snapshot)"
    assert builder._use_bprp_primary is True
