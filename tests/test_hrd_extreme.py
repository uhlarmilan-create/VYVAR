"""HRD extreme-object selection, classification, and online enrichment tests."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from config import AppConfig
from hrd_analysis import (
    HRD_EMPTY_FIELD_MSG,
    _LABEL_BINARY,
    _pick_label,
    _select_stage1_candidates,
    _stage1_candidate_mask,
    _stage2_labels,
    get_top_interesting_stars,
    hrd_parallax_params_from_cfg,
    is_hrd_parallax_reliable,
)
from hrd_enrich import enrich_candidates


def _row(**kwargs) -> pd.Series:
    base = {
        "catalog_id": "4035720806645181440",
        "bp_rp": 0.8,
        "abs_mag_g": 5.0,
        "hrd_reliable": False,
        "teff_gspphot": None,
        "logg_gspphot": None,
        "non_single_star": 0,
        "phot_g_mean_mag": 12.0,
    }
    base.update(kwargs)
    return pd.Series(base)


def test_stage1_hits_each_criterion_and_respects_cap() -> None:
    df = pd.DataFrame(
        [
            {"catalog_id": "1", "bp_rp": -0.2, "abs_mag_g": 4.0, "hrd_reliable": False, "non_single_star": 0},
            {"catalog_id": "2", "bp_rp": 2.6, "abs_mag_g": 3.0, "hrd_reliable": False, "non_single_star": 0},
            {"catalog_id": "3", "bp_rp": 0.2, "abs_mag_g": 10.5, "hrd_reliable": True, "non_single_star": 0},
            {"catalog_id": "4", "bp_rp": 0.5, "abs_mag_g": -2.5, "hrd_reliable": True, "non_single_star": 0},
            {"catalog_id": "5", "bp_rp": 0.9, "abs_mag_g": 6.0, "hrd_reliable": True, "non_single_star": 1},
            {"catalog_id": "6", "bp_rp": 0.7, "abs_mag_g": 5.0, "hrd_reliable": True, "non_single_star": 0},
        ]
    )
    mask = _stage1_candidate_mask(df)
    assert mask.sum() == 4
    assert not bool(mask.iloc[4])
    capped = _select_stage1_candidates(df, 3, min_per_net=0)
    assert len(capped) == 3


def test_stage2_teff_preferred_and_bp_rp_fallback() -> None:
    hot_teff = _stage2_labels(_row(teff_gspphot=30000, bp_rp=0.5))
    assert any("Very hot" in x for x in hot_teff)

    hot_bp = _stage2_labels(_row(teff_gspphot=None, bp_rp=-0.35))
    assert any("Very hot" in x for x in hot_bp)

    cool_teff = _stage2_labels(_row(teff_gspphot=2800, bp_rp=1.0))
    assert any("Very cool" in x for x in cool_teff)

    cool_bp = _stage2_labels(_row(teff_gspphot=None, bp_rp=3.2))
    assert any("Very cool" in x for x in cool_bp)


def test_stage2_supergiant_beats_very_cool() -> None:
    labels = _stage2_labels(
        _row(bp_rp=3.2, logg_gspphot=1.0, abs_mag_g=-5.0, hrd_reliable=True)
    )
    assert "Red supergiant" in labels
    assert not any("Very cool" in x for x in labels)
    picked = _pick_label(labels, "s*r")
    assert picked.startswith("Red supergiant")
    assert "Very cool" not in picked


def test_stage1_luminous_net_reserved_slots() -> None:
    red_rows = [
        {
            "catalog_id": f"r{i}",
            "bp_rp": 3.0 + i * 0.01,
            "abs_mag_g": 2.0,
            "hrd_reliable": True,
            "non_single_star": 0,
        }
        for i in range(15)
    ]
    lum_rows = [
        {
            "catalog_id": f"l{i}",
            "bp_rp": 0.5,
            "abs_mag_g": -3.0 - i * 0.2,
            "hrd_reliable": True,
            "non_single_star": 0,
        }
        for i in range(3)
    ]
    df = pd.DataFrame(red_rows + lum_rows)
    picked = _select_stage1_candidates(df, 10, min_per_net=2)
    ids = set(picked["catalog_id"].astype(str))
    assert any(cid.startswith("l") for cid in ids)


def test_shrink_net_reservations_round_robin() -> None:
    from hrd_analysis import _shrink_net_reservations

    alloc = {"blue": 4, "red": 4, "wd": 4, "luminous": 4, "nss": 4}
    out = _shrink_net_reservations(alloc, budget=6)
    assert sum(out.values()) == 6
    assert out["nss"] <= 4


def test_stage2_label_priority_and_simbad_suffix() -> None:
    labels = _stage2_labels(
        _row(
            bp_rp=0.1,
            abs_mag_g=11.0,
            hrd_reliable=True,
            teff_gspphot=32000,
            non_single_star=1,
        )
    )
    picked = _pick_label(labels, "WR*")
    assert picked.startswith("White dwarf candidate")
    assert "SIMBAD: WR*" in picked


def test_empty_field_returns_explicit_marker() -> None:
    df = pd.DataFrame(
        [
            {
                "catalog_id": "100",
                "bp_rp": 0.9,
                "abs_mag_g": 4.0,
                "hrd_reliable": True,
                "non_single_star": 0,
                "phot_g_mean_mag": 11.0,
            }
        ]
    )
    cfg = AppConfig()
    cfg.hrd_online_enrich_enabled = False
    out = get_top_interesting_stars(df, cfg=cfg, cache_path=None)
    assert len(out) == 1
    assert bool(out.iloc[0]["_empty_field"])
    assert HRD_EMPTY_FIELD_MSG in str(out.iloc[0]["category"])


def test_enrichment_merge_and_cache_roundtrip(tmp_path: Path) -> None:
    cache = tmp_path / "hrd_enrich.json"
    cand = pd.DataFrame([{"catalog_id": "4035720806645181440", "teff_gspphot": None, "logg_gspphot": None}])

    gaia_payload = {
        "4035720806645181440": {
            "teff_gspphot": 5500.0,
            "logg_gspphot": 4.2,
            "fetched_at_utc": "2026-07-10T00:00:00+00:00",
            "enrich_source": "gaia_tap",
        }
    }
    sim_payload = {
        "4035720806645181440": {
            "simbad_main_id": "HD 123",
            "simbad_otype": "EB*",
            "fetched_at_utc": "2026-07-10T00:00:00+00:00",
            "enrich_source": "simbad",
        }
    }

    with patch("hrd_enrich._fetch_gaia_tap", return_value=gaia_payload):
        with patch("hrd_enrich._fetch_simbad", return_value=sim_payload):
            out = enrich_candidates(cand, cache, enabled=True, simbad_enabled=True)
    assert float(out.iloc[0]["teff_gspphot"]) == 5500.0
    assert out.iloc[0]["simbad_otype"] == "EB*"
    assert cache.is_file()

    cached = json.loads(cache.read_text(encoding="utf-8"))
    assert "4035720806645181440" in cached

    cand2 = pd.DataFrame([{"catalog_id": "4035720806645181440"}])
    with patch("hrd_enrich._fetch_gaia_tap") as mock_gaia:
        out2 = enrich_candidates(cand2, cache, enabled=True, simbad_enabled=True)
        mock_gaia.assert_not_called()
    assert float(out2.iloc[0]["teff_gspphot"]) == 5500.0


def test_enrichment_negative_result_cached(tmp_path: Path) -> None:
    cache = tmp_path / "hrd_enrich.json"
    sid = "1111111111111111111"
    cand = pd.DataFrame([{"catalog_id": sid}])
    with patch("hrd_enrich._fetch_gaia_tap", return_value={}):
        enrich_candidates(cand, cache, enabled=True, simbad_enabled=False)
    data = json.loads(cache.read_text(encoding="utf-8"))
    assert sid in data
    assert data[sid].get("teff_gspphot") is None


def test_enrichment_fail_open_on_exception(tmp_path: Path) -> None:
    cache = tmp_path / "hrd_enrich.json"
    cand = pd.DataFrame([{"catalog_id": "2222222222222222222"}])
    with patch("hrd_enrich._fetch_gaia_tap", side_effect=RuntimeError("network down")):
        out = enrich_candidates(cand, cache, enabled=True, simbad_enabled=False)
    assert out.iloc[0]["enrich_source"] == "n/a"


def test_enrichment_disabled_skips_network() -> None:
    cand = pd.DataFrame([{"catalog_id": "3333333333333333333"}])
    with patch("hrd_enrich._fetch_gaia_tap") as mock_gaia:
        out = enrich_candidates(cand, None, enabled=False, simbad_enabled=True)
        mock_gaia.assert_not_called()
    assert out.iloc[0]["enrich_source"] == "local"


def test_hrd_config_defaults() -> None:
    cfg = AppConfig()
    assert cfg.hrd_online_enrich_enabled is True
    assert cfg.hrd_simbad_enrich_enabled is True
    assert 1 <= cfg.hrd_enrich_max_candidates <= 100
    assert cfg.hrd_parallax_min_mas == 0.15
    assert cfg.hrd_parallax_snr_min == 5.0
    assert cfg.hrd_max_per_category == 3
    assert cfg.hrd_min_per_net == 4
    assert cfg.hrd_nss_category_enabled is False


def test_nss_default_off_no_binary_rows() -> None:
    nss_rows = [
        {
            "catalog_id": f"n{i}",
            "bp_rp": 0.9,
            "abs_mag_g": 5.0,
            "hrd_reliable": True,
            "logg_gspphot": 4.0,
            "non_single_star": 1,
            "phot_g_mean_mag": 12.0,
            "x": 100.0 + i * 10,
            "y": 200.0,
        }
        for i in range(5)
    ]
    df = pd.DataFrame(nss_rows)
    cfg = AppConfig()
    cfg.hrd_online_enrich_enabled = False
    cfg.hrd_simbad_enrich_enabled = False
    assert cfg.hrd_nss_category_enabled is False
    top = get_top_interesting_stars(df, cfg=cfg, cache_path=None)
    assert len(top) == 1
    assert bool(top.iloc[0]["_empty_field"])
    assert not any(_LABEL_BINARY in str(c) for c in top.get("category", []))


def test_nss_enabled_stage1_includes_binary_net() -> None:
    df = pd.DataFrame(
        [
            {
                "catalog_id": "5",
                "bp_rp": 0.9,
                "abs_mag_g": 6.0,
                "hrd_reliable": True,
                "non_single_star": 1,
            }
        ]
    )
    assert not bool(_stage1_candidate_mask(df).iloc[0])
    assert bool(_stage1_candidate_mask(df, nss_enabled=True).iloc[0])


def test_parallax_gate_default_and_clamps() -> None:
    assert is_hrd_parallax_reliable(0.4, 10.0)
    assert not is_hrd_parallax_reliable(0.4, 3.0)
    assert not is_hrd_parallax_reliable(0.05, 10.0)

    cfg = AppConfig()
    cfg.hrd_parallax_min_mas = 99.0
    cfg.hrd_parallax_snr_min = 99.0
    pmin, psnr = hrd_parallax_params_from_cfg(cfg)
    assert pmin == 10.0
    assert psnr == 20.0


def test_category_cap_and_nss_deprioritized() -> None:
    giants = [
        {
            "catalog_id": f"g{i}",
            "bp_rp": 2.6 + i * 0.01,
            "abs_mag_g": 0.0,
            "hrd_reliable": True,
            "logg_gspphot": 2.0,
            "non_single_star": 0,
            "phot_g_mean_mag": 10.0,
        }
        for i in range(2)
    ]
    nss_rows = [
        {
            "catalog_id": f"n{i}",
            "bp_rp": 0.9,
            "abs_mag_g": 5.0,
            "hrd_reliable": True,
            "logg_gspphot": 4.0,
            "non_single_star": 1,
            "phot_g_mean_mag": 12.0,
        }
        for i in range(10)
    ]
    df = pd.DataFrame(giants + nss_rows)
    cfg = AppConfig()
    cfg.hrd_online_enrich_enabled = False
    cfg.hrd_simbad_enrich_enabled = False
    cfg.hrd_enrich_max_candidates = 6
    cfg.hrd_max_per_category = 3
    cfg.hrd_nss_category_enabled = True
    top = get_top_interesting_stars(df, cfg=cfg, cache_path=None)
    cats = top["category"].tolist()
    assert sum("Red giant" in c for c in cats) == 2
    assert sum(_LABEL_BINARY in c for c in cats) <= 3
    assert len(top) <= 6


def test_get_top_interesting_stars_offline_with_classification() -> None:
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
    assert not top.empty
    assert "White dwarf" in str(top.iloc[0]["category"])
    assert top.iloc[0]["teff"] == "N/A"
