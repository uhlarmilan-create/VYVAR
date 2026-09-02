"""Phase 0 identity gate tests (PHASE0-IDENTITY-GATE)."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from config import AppConfig
from photometry_core import select_active_targets


def _base_masterstars(tmp_path: Path, extra_rows: list[dict] | None = None) -> Path:
    rows = [
        {
            "name": "1111111111111111111",
            "catalog_id": "1111111111111111111",
            "ra_deg": 150.01,
            "dec_deg": 45.01,
            "x": 300.0,
            "y": 300.0,
            "mag": 11.0,
            "zone": "linear",
            "is_usable": True,
            "is_saturated": False,
            "is_noisy": False,
            "snr50_ok": True,
        },
        {
            "name": "2222222222222222222",
            "catalog_id": "2222222222222222222",
            "ra_deg": 150.02,
            "dec_deg": 45.02,
            "x": 310.0,
            "y": 310.0,
            "mag": 8.0,
            "zone": "linear",
            "is_usable": True,
            "is_saturated": False,
            "is_noisy": False,
            "snr50_ok": True,
        },
    ]
    if extra_rows:
        rows.extend(extra_rows)
    ms = pd.DataFrame(rows)
    p = tmp_path / "masterstars_full_match.csv"
    ms.to_csv(p, index=False)
    return p


def _write_vt(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "variable_targets.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def test_identity_join_excludes_missing_masterstar(tmp_path: Path) -> None:
    ms_csv = _base_masterstars(tmp_path)
    vt_csv = _write_vt(
        tmp_path,
        [
            {
                "name": "VSX_A",
                "vsx_name": "VSX_A",
                "vsx_type": "EA",
                "catalog": "VSX",
                "catalog_id": "9999999999999999999",
                "gaia_match_source": "masterstars",
                "ra_deg": 150.0,
                "dec_deg": 45.0,
                "x": 256.0,
                "y": 256.0,
                "mag": 12.0,
            }
        ],
    )
    out = select_active_targets(vt_csv, ms_csv, frame_w_px=512, frame_h_px=512, edge_margin_px=50)
    assert len(out) == 0


def test_misassociation_guard_near_bright_neighbor(tmp_path: Path) -> None:
    """VSX planner id absent from masterstars; bright neighbor nearby must not promote."""
    ms_csv = _base_masterstars(tmp_path)
    vt_csv = _write_vt(
        tmp_path,
        [
            {
                "name": "VSX_FAINT",
                "vsx_name": "VSX_FAINT",
                "vsx_type": "ROT",
                "catalog": "VSX",
                "catalog_id": "3333333333333333333",
                "gaia_match_source": "masterstars",
                "ra_deg": 150.019,
                "dec_deg": 45.019,
                "x": 309.0,
                "y": 309.0,
                "mag": 17.0,
            }
        ],
    )
    out = select_active_targets(vt_csv, ms_csv, frame_w_px=512, frame_h_px=512, edge_margin_px=50)
    assert len(out) == 0


def test_gaia_dr3_direct_never_active(tmp_path: Path) -> None:
    ms_csv = _base_masterstars(tmp_path)
    vt_csv = _write_vt(
        tmp_path,
        [
            {
                "name": "VSX_FB",
                "vsx_name": "VSX_FB",
                "vsx_type": "EA",
                "catalog": "VSX",
                "catalog_id": "1111111111111111111",
                "gaia_match_source": "gaia_dr3_direct",
                "ra_deg": 150.01,
                "dec_deg": 45.01,
                "x": 300.0,
                "y": 300.0,
                "mag": 11.0,
            }
        ],
    )
    out = select_active_targets(vt_csv, ms_csv, frame_w_px=512, frame_h_px=512, edge_margin_px=50)
    assert len(out) == 0


def test_no_match_source_never_active(tmp_path: Path) -> None:
    ms_csv = _base_masterstars(tmp_path)
    vt_csv = _write_vt(
        tmp_path,
        [
            {
                "name": "VSX_NM",
                "vsx_name": "VSX_NM",
                "vsx_type": "EA",
                "catalog": "VSX",
                "catalog_id": "",
                "gaia_match_source": "no_match",
                "ra_deg": 150.01,
                "dec_deg": 45.01,
                "x": 300.0,
                "y": 300.0,
                "mag": 11.0,
            }
        ],
    )
    out = select_active_targets(vt_csv, ms_csv, frame_w_px=512, frame_h_px=512, edge_margin_px=50)
    assert len(out) == 0


def test_masterstars_identity_promotes_matching_planner_id(tmp_path: Path) -> None:
    ms_csv = _base_masterstars(tmp_path)
    vt_csv = _write_vt(
        tmp_path,
        [
            {
                "name": "VSX_OK",
                "vsx_name": "VSX_OK",
                "vsx_type": "DSCT",
                "catalog": "VSX",
                "catalog_id": "1111111111111111111",
                "gaia_match_source": "masterstars",
                "ra_deg": 150.01,
                "dec_deg": 45.01,
                "x": 300.0,
                "y": 300.0,
                "mag": 11.0,
            }
        ],
    )
    out = select_active_targets(vt_csv, ms_csv, frame_w_px=512, frame_h_px=512, edge_margin_px=50)
    assert len(out) == 1
    assert str(out.iloc[0]["catalog_id"]) == "1111111111111111111"


def test_vsx_out_of_scope_types_masks_when_configured(tmp_path: Path) -> None:
    ms_csv = _base_masterstars(tmp_path)
    vt_csv = _write_vt(
        tmp_path,
        [
            {
                "name": "VSX_ROT",
                "vsx_name": "VSX_ROT",
                "vsx_type": "ROT",
                "catalog": "VSX",
                "catalog_id": "1111111111111111111",
                "gaia_match_source": "masterstars",
                "ra_deg": 150.01,
                "dec_deg": 45.01,
                "x": 300.0,
                "y": 300.0,
                "mag": 11.0,
            }
        ],
    )
    cfg = AppConfig()
    cfg.vsx_out_of_scope_types = ["ROT"]
    out = select_active_targets(
        vt_csv, ms_csv, frame_w_px=512, frame_h_px=512, edge_margin_px=50, cfg=cfg
    )
    assert len(out) == 1
    assert bool(out.iloc[0]["skip_photometry"]) is True
    assert str(out.iloc[0]["skip_reason"]) == "vsx_type_out_of_scope"


def test_no_fixed_radius_in_select_active_targets_source() -> None:
    src = (Path(__file__).resolve().parents[2] / "src_py" / "photometry_comp.py").read_text(
        encoding="utf-8"
    )
    fn_start = src.index("def select_active_targets(")
    fn_end = src.index("\ndef _batch_enrich_targets_bp_rp_from_gaia_db")
    body = src[fn_start:fn_end]
    assert "match_radius_arcsec" not in body
    assert "phase01_match_radius_arcsec" not in body
    assert "5.0 *" not in body or "plate_nominal * 5" not in body.replace(" ", "")


def test_no_fixed_radius_in_vsx_gaia_crossmatch_module() -> None:
    src = (Path(__file__).resolve().parents[2] / "src_py" / "vsx_gaia_crossmatch.py").read_text(
        encoding="utf-8"
    )
    assert "max_sep" not in src
    assert not re.search(r"\d+\.0\s*\*\s*u\.arcsec", src)
