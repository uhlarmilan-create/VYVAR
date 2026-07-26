# -*- coding: ascii -*-
"""Tests for vsx_out_of_scope_types token match + Phase-0 mask-first skip."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from config import AppConfig
from photometry_core import select_active_targets
from vsx_type_scope import (
    is_vsx_auto_selected_target,
    tokenize_vsx_type,
    vsx_type_is_out_of_scope,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("ROT", ["ROT"]),
        ("LB:", ["LB"]),
        ("RRC:", ["RRC"]),
        ("DSCT|GDOR|SXPHE", ["DSCT", "GDOR", "SXPHE"]),
        ("RRAB/BL", ["RRAB", "BL"]),
        ("ACV|roAm|roAp|SXARI", ["ACV", "ROAM", "ROAP", "SXARI"]),
        ("VAR", ["VAR"]),
        ("", []),
        ("  ", []),
    ],
)
def test_tokenize_vsx_type_real_field_values(raw: str, expected: list[str]) -> None:
    assert tokenize_vsx_type(raw) == expected


def test_out_of_scope_examples_from_spec() -> None:
    assert vsx_type_is_out_of_scope("ROT", ["ROT"])
    assert vsx_type_is_out_of_scope("ROT:", ["ROT"])
    assert not vsx_type_is_out_of_scope("DSCT|GDOR|SXPHE", ["ROT"])
    assert vsx_type_is_out_of_scope("DSCT|GDOR|SXPHE", ["SXPHE"])
    assert vsx_type_is_out_of_scope("RRAB", ["RRAB"])
    assert vsx_type_is_out_of_scope("RRAB/BL", ["RRAB"])
    assert vsx_type_is_out_of_scope("M", ["M"])
    assert not vsx_type_is_out_of_scope("SRM", ["M"])
    assert not vsx_type_is_out_of_scope("EA/M?", ["M"])  # token is M? -> stripped? trailing : only
    # Exact token only: "M?" after upper is "M?" not "M"
    assert tokenize_vsx_type("EA/M?") == ["EA", "M?"]
    assert not vsx_type_is_out_of_scope("EA/M?", ["M"])
    assert not vsx_type_is_out_of_scope("", ["ROT"])


def test_manual_targets_never_auto() -> None:
    assert is_vsx_auto_selected_target({"catalog": "VSX", "vsx_name": "X"})
    assert not is_vsx_auto_selected_target({"catalog": "MANUAL", "vsx_name": "X", "vsx_type": "ROT"})
    assert not is_vsx_auto_selected_target({"catalog": "USER", "vsx_type": "ROT"})


def _write_phase0(tmp_path: Path) -> tuple[Path, Path]:
    vt = pd.DataFrame(
        [
            {
                "name": "ROT_STAR",
                "vsx_name": "ROT_STAR",
                "vsx_type": "ROT",
                "vsx_period": "",
                "priority": 1,
                "catalog": "VSX",
                "ra_deg": 150.0,
                "dec_deg": 45.0,
                "x": 200.0,
                "y": 200.0,
                "catalog_id": "1111111111111111111",
                "gaia_match_source": "masterstars",
                "mag": 12.0,
            },
            {
                "name": "DSCT_STAR",
                "vsx_name": "DSCT_STAR",
                "vsx_type": "DSCT|GDOR|SXPHE",
                "vsx_period": "",
                "priority": 1,
                "catalog": "VSX",
                "ra_deg": 150.02,
                "dec_deg": 45.02,
                "x": 300.0,
                "y": 300.0,
                "catalog_id": "2222222222222222222",
                "gaia_match_source": "masterstars",
                "mag": 11.5,
            },
            {
                "name": "MANUAL_ROT",
                "vsx_name": "MANUAL_ROT",
                "vsx_type": "ROT",
                "vsx_period": "",
                "priority": 0,
                "catalog": "MANUAL",
                "ra_deg": 150.04,
                "dec_deg": 45.04,
                "x": 350.0,
                "y": 350.0,
                "catalog_id": "3333333333333333333",
                "mag": 11.0,
            },
        ]
    )
    ms = pd.DataFrame(
        [
            {
                "name": cid,
                "catalog_id": cid,
                "ra_deg": ra,
                "dec_deg": de,
                "x": x,
                "y": y,
                "mag": 12.0,
                "zone": "linear",
                "is_usable": True,
                "is_saturated": False,
                "is_noisy": False,
                "snr50_ok": True,
            }
            for cid, ra, de, x, y in [
                ("1111111111111111111", 150.0, 45.0, 200.0, 200.0),
                ("2222222222222222222", 150.02, 45.02, 300.0, 300.0),
                ("3333333333333333333", 150.04, 45.04, 350.0, 350.0),
            ]
        ]
    )
    vt_p = tmp_path / "variable_targets.csv"
    ms_p = tmp_path / "masterstars_full_match.csv"
    vt.to_csv(vt_p, index=False)
    ms.to_csv(ms_p, index=False)
    return vt_p, ms_p


def test_select_active_out_of_scope_mask_and_manual_immune(tmp_path: Path) -> None:
    vt_p, ms_p = _write_phase0(tmp_path)
    cfg = AppConfig()
    cfg.vsx_out_of_scope_types = ["ROT"]
    out = select_active_targets(
        vt_p,
        ms_p,
        frame_w_px=512,
        frame_h_px=512,
        edge_margin_px=10,
        cfg=cfg,
    )
    assert len(out) == 3
    by = {str(r["vsx_name"]): r for _, r in out.iterrows()}
    assert bool(by["ROT_STAR"]["skip_photometry"]) is True
    assert str(by["ROT_STAR"]["skip_reason"]) == "vsx_type_out_of_scope"
    assert bool(by["DSCT_STAR"]["skip_photometry"]) is False
    assert bool(by["MANUAL_ROT"]["skip_photometry"]) is False


def test_empty_list_noop_equivalence(tmp_path: Path) -> None:
    vt_p, ms_p = _write_phase0(tmp_path)
    cfg_off = AppConfig()
    cfg_off.vsx_out_of_scope_types = []
    cfg_default = AppConfig()
    cfg_default.vsx_out_of_scope_types = []
    a = select_active_targets(
        vt_p, ms_p, frame_w_px=512, frame_h_px=512, edge_margin_px=10,
        cfg=cfg_off,
    )
    b = select_active_targets(
        vt_p, ms_p, frame_w_px=512, frame_h_px=512, edge_margin_px=10,
        cfg=cfg_default,
    )
    assert list(a["skip_photometry"]) == list(b["skip_photometry"])
    assert not any(a["skip_photometry"])
    assert "vsx_type_out_of_scope" not in set(a.get("skip_reason", pd.Series(dtype=str)).astype(str))


def test_out_of_scope_still_in_comp_exclude_set(tmp_path: Path) -> None:
    """Out-of-scope ROT stays in active_targets -> still in Phase-2A exclude set."""
    vt_p, ms_p = _write_phase0(tmp_path)
    cfg = AppConfig()
    cfg.vsx_out_of_scope_types = ["ROT"]
    out = select_active_targets(
        vt_p, ms_p, frame_w_px=512, frame_h_px=512, edge_margin_px=10,
        cfg=cfg,
    )
    cids = {
        str(c)
        for c in out["catalog_id"].tolist()
        if str(c).strip()
    }
    assert "1111111111111111111" in cids  # ROT skipped but still present
    # Simulate Phase-2A exclude construction (all active_targets ids)
    exclude = frozenset(cids)
    assert "1111111111111111111" in exclude
