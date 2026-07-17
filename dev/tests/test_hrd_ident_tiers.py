"""HRD identification/confirmation tier tests (TODO-12e)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from config import AppConfig
from hrd_analysis import (
    _LABEL_RG,
    _LABEL_RSG,
    _LABEL_VERY_COOL,
    _LABEL_WD,
    _effective_logg,
    _finalize_ident,
    _match_literature_confirmed,
    _stage2_labels,
    get_top_interesting_stars,
    hrd_dsc_confirm_prob_from_cfg,
)
from hrd_enrich import CACHE_VERSION, _load_cache, enrich_candidates


def _row(**kwargs) -> pd.Series:
    base = {
        "catalog_id": "4035720806645181440",
        "bp_rp": 0.8,
        "abs_mag_g": 5.0,
        "hrd_reliable": True,
        "teff_gspphot": None,
        "logg_gspphot": None,
        "non_single_star": 0,
        "phot_g_mean_mag": 12.0,
        "simbad_otype": "",
        "simbad_sp_type": "",
        "classprob_dsc_combmod_whitedwarf": None,
    }
    base.update(kwargs)
    return pd.Series(base)


@pytest.mark.parametrize(
    ("base", "otype", "sp_type", "expect_detail"),
    [
        (_LABEL_WD, "WD*", "DA2.1", "DA2.1, SIMBAD"),
        (_LABEL_WD, "WD?", None, None),
        (_LABEL_WD, None, "DB3", "DB3, SIMBAD"),
        (_LABEL_RSG, "s*r", "M4Iab", "M4Iab, SIMBAD"),
        (_LABEL_RG, "RG*", "K3III", "K3III, SIMBAD"),
        (_LABEL_VERY_COOL, "LP*", "M7", "M7, SIMBAD"),
        (_LABEL_WD, None, None, None),
    ],
)
def test_literature_confirmed_param(base: str, otype: str | None, sp_type: str | None, expect_detail: str | None) -> None:
    got = _match_literature_confirmed(base, otype, sp_type)
    assert got == expect_detail


def test_dsc_likely_wd_tier() -> None:
    row = _row(classprob_dsc_combmod_whitedwarf=0.99, simbad_otype="")
    tier, display, detail, _ = _finalize_ident(
        row, _LABEL_WD, dsc_threshold=0.90, enrichment_active=True
    )
    assert tier == "likely"
    assert "likely, DSC p=0.99" in display
    assert "DSC p=0.99" in detail


def test_candidate_otype_question_mark() -> None:
    row = _row(simbad_otype="WD?", classprob_dsc_combmod_whitedwarf=0.5)
    tier, display, detail, _ = _finalize_ident(
        row, _LABEL_WD, dsc_threshold=0.90, enrichment_active=True
    )
    assert tier == "candidate"
    assert display == _LABEL_WD
    assert "uncertain" in detail


def test_offline_all_candidate() -> None:
    row = _row(simbad_otype="WD*", simbad_sp_type="DA2")
    tier, display, detail, logg_src = _finalize_ident(
        row, _LABEL_WD, dsc_threshold=0.90, enrichment_active=False
    )
    assert tier == "candidate"
    assert display == _LABEL_WD
    assert detail == ""
    assert logg_src in ("n/a", "gaia")


def test_lumclass_supergiant_when_logg_na() -> None:
    row = _row(
        bp_rp=3.43,
        abs_mag_g=-5.5,
        hrd_reliable=True,
        logg_gspphot=None,
        simbad_sp_type="M4Iab",
        teff_gspphot=None,
    )
    logg, src = _effective_logg(row)
    assert src == "simbad_lumclass"
    assert logg is not None and logg < 1.5
    labels = _stage2_labels(row)
    assert _LABEL_RSG in labels
    assert _LABEL_VERY_COOL not in labels


def test_lumclass_dwarf_blocks_giant() -> None:
    row = _row(
        bp_rp=2.0,
        abs_mag_g=1.0,
        hrd_reliable=True,
        logg_gspphot=None,
        simbad_sp_type="K2V",
    )
    logg, src = _effective_logg(row)
    assert src == "simbad_lumclass"
    assert logg is not None and logg > 3.5
    labels = _stage2_labels(row)
    assert _LABEL_RG not in labels
    assert _LABEL_RSG not in labels


def test_gaia_logg_wins_over_simbad_lumclass() -> None:
    row = _row(logg_gspphot=4.2, simbad_sp_type="M4Iab", bp_rp=3.5, abs_mag_g=-4.0, hrd_reliable=True)
    logg, src = _effective_logg(row)
    assert src == "gaia"
    assert logg == 4.2
    labels = _stage2_labels(row)
    assert _LABEL_RSG not in labels


def test_cache_version_mismatch_discarded(tmp_path: Path) -> None:
    cache = tmp_path / "hrd_enrich.json"
    cache.write_text(
        json.dumps(
            {
                "4035720806645181440": {
                    "teff_gspphot": 5000.0,
                    "logg_gspphot": 4.0,
                    "simbad_otype": "WD*",
                }
            }
        ),
        encoding="utf-8",
    )
    assert _load_cache(cache) == {}


def test_cache_v1_refetch_on_enrich(tmp_path: Path) -> None:
    cache = tmp_path / "hrd_enrich.json"
    sid = "4035720806645181440"
    cache.write_text(
        json.dumps(
            {
                "cache_version": 1,
                "entries": {
                    sid: {
                        "teff_gspphot": 5000.0,
                        "logg_gspphot": 4.0,
                        "simbad_otype": "WD*",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    cand = pd.DataFrame([{"catalog_id": sid}])
    gaia_payload = {
        sid: {
            "teff_gspphot": 9000.0,
            "logg_gspphot": 8.0,
            "classprob_dsc_combmod_whitedwarf": 0.01,
            "classprob_dsc_combmod_binarystar": 0.0,
            "spectraltype_esphs": None,
            "fetched_at_utc": "2026-07-10T00:00:00+00:00",
            "enrich_source": "gaia_tap",
        }
    }
    with patch("hrd_enrich._fetch_gaia_tap", return_value=gaia_payload) as mock_gaia:
        out = enrich_candidates(cand, cache, enabled=True, simbad_enabled=False)
        mock_gaia.assert_called_once()
    assert float(out.iloc[0]["teff_gspphot"]) == 9000.0
    saved = json.loads(cache.read_text(encoding="utf-8"))
    assert saved["cache_version"] == CACHE_VERSION


def test_get_top_interesting_offline_ident_candidate() -> None:
    df = pd.DataFrame(
        [
            {
                "catalog_id": "9001",
                "bp_rp": 0.1,
                "abs_mag_g": 11.5,
                "hrd_reliable": True,
                "teff_gspphot": None,
                "logg_gspphot": None,
                "non_single_star": 0,
                "phot_g_mean_mag": 15.0,
            }
        ]
    )
    cfg = AppConfig()
    cfg.hrd_online_enrich_enabled = False
    cfg.hrd_simbad_enrich_enabled = False
    top = get_top_interesting_stars(df, cfg=cfg, cache_path=None)
    assert top.iloc[0]["ident"] == "candidate"
    assert "White dwarf" in str(top.iloc[0]["category"])


def test_lumclass_embedded_in_mk_type() -> None:
    row = _row(
        bp_rp=3.43,
        abs_mag_g=-5.5,
        hrd_reliable=True,
        logg_gspphot=None,
        simbad_sp_type="M3.5IabFe-1",
        simbad_otype="s*r",
        teff_gspphot=None,
    )
    logg, src = _effective_logg(row)
    assert src == "simbad_lumclass"
    labels = _stage2_labels(row)
    assert _LABEL_RSG in labels
    assert _LABEL_VERY_COOL not in labels


def test_rs_per_style_finalize_rsg() -> None:
    row = _row(
        bp_rp=3.43,
        abs_mag_g=-5.5,
        hrd_reliable=True,
        logg_gspphot=None,
        simbad_sp_type="M3.5IabFe-1",
        simbad_otype="s*r",
        simbad_main_id="V* RS Per",
    )
    base = _LABEL_RSG if _LABEL_RSG in _stage2_labels(row) else _LABEL_VERY_COOL
    tier, display, _, logg_src = _finalize_ident(
        row, base, dsc_threshold=0.90, enrichment_active=True
    )
    assert _LABEL_RSG in _stage2_labels(row)
    assert "Red supergiant" in display
    assert tier == "confirmed"
    assert logg_src == "simbad_lumclass"


def test_hrd_dsc_confirm_prob_clamp() -> None:
    cfg = AppConfig()
    cfg.hrd_dsc_confirm_prob = 0.1
    assert hrd_dsc_confirm_prob_from_cfg(cfg) == 0.5
    cfg.hrd_dsc_confirm_prob = 5.0
    assert hrd_dsc_confirm_prob_from_cfg(cfg) == 1.0
