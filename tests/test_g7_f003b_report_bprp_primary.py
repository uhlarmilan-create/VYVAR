"""G7-F003b: photometry_report reads cfg.phase01_use_bprp_primary (not hardcoded True)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import cm, mm

from config import AppConfig
from photometry_report import _PhotometryReportBuilder, _norm_cid

_BPRP_PRIMARY_NOTE_TXT = (
    "Gaia BP-RP colour | COMP weights: w = 1/sigma^2 - Broeg et al., Astron. Nachr. 326, 134 (2005)"
)
_LEGACY_BV_NOTE_TXT = (
    "B-V from Gaia BP-RP (Riello et al. 2021) | "
    "COMP weights: w = 1/sigma^2 - Broeg et al., Astron. Nachr. 326, 134 (2005)"
)


def _patch_appconfig_root(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    def _factory(**kwargs: object) -> AppConfig:
        return AppConfig(project_root=root)

    monkeypatch.setattr("config.AppConfig", _factory)


def _minimal_report_builder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    bprp_primary: bool,
    comp_df: pd.DataFrame | None = None,
) -> _PhotometryReportBuilder:
    obs_group = "V_test"
    phot = tmp_path / "platesolve" / obs_group / "photometry"
    phot.mkdir(parents=True)
    pd.DataFrame({"catalog_id": ["486430957815961344"], "lc_rms": [0.05]}).to_csv(
        phot / "photometry_summary.csv", index=False
    )
    if comp_df is not None:
        comp_df.to_csv(phot / "comparison_stars_per_target.csv", index=False)

    (tmp_path / "config.json").write_text(
        json.dumps({"phase01_use_bprp_primary": bprp_primary}),
        encoding="utf-8",
    )
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


def test_report_use_bprp_primary_true_matches_legacy_wording(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _minimal_report_builder(tmp_path, monkeypatch, bprp_primary=True)
    assert builder._use_bprp_primary is True
    assert builder.NOTE_TXT == _BPRP_PRIMARY_NOTE_TXT


def test_report_use_bprp_primary_false_uses_legacy_bv_wording(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _minimal_report_builder(tmp_path, monkeypatch, bprp_primary=False)
    assert builder._use_bprp_primary is False
    assert builder.NOTE_TXT == _LEGACY_BV_NOTE_TXT


def test_comp_rows_false_includes_bv_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = "486430957815961344"
    comp_df = pd.DataFrame(
        {
            "target_catalog_id": [target],
            "catalog_id": ["1498613634033133184"],
            "mag": [12.0],
            "b_v": [0.55],
            "bv_source": ["gaia_bprp"],
            "bp_rp": [0.8],
            "delta_bprp_abs": [0.1],
            "dist_deg": [0.01],
            "comp_n_frames": [10],
            "comp_rms": [0.02],
            "comp_tier": [1],
        }
    )
    builder = _minimal_report_builder(
        tmp_path, monkeypatch, bprp_primary=False, comp_df=comp_df
    )
    rows, _ = builder._comp_rows_for_target(_norm_cid(target))
    assert len(rows) == 1
    assert rows[0][3] == "0.550"
    assert rows[0][4] == "G-bp"


def test_comp_rows_true_skips_bv_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = "486430957815961344"
    comp_df = pd.DataFrame(
        {
            "target_catalog_id": [target],
            "catalog_id": ["1498613634033133184"],
            "mag": [12.0],
            "b_v": [0.55],
            "bv_source": ["gaia_bprp"],
            "bp_rp": [0.8],
            "delta_bprp_abs": [0.1],
            "dist_deg": [0.01],
            "comp_n_frames": [10],
            "comp_rms": [0.02],
            "comp_tier": [1],
        }
    )
    builder = _minimal_report_builder(
        tmp_path, monkeypatch, bprp_primary=True, comp_df=comp_df
    )
    rows, _ = builder._comp_rows_for_target(_norm_cid(target))
    assert len(rows) == 1
    assert rows[0][3] == "0.800"
