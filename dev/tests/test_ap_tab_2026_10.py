"""AP-TAB-2026-10: Aperture Photometry tab helpers and the shared target-page data."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from target_page import TargetPages, star_fields
from ui_aperture_photometry import _comp_table_html, _measured_cids, _tier_text, _ut_ticks


def _pages(tmp_path):
    lc = tmp_path / "photometry" / "lightcurves"
    lc.mkdir(parents=True)
    pd.DataFrame({"bjd": [2461154.30, 2461154.35, 2461154.40], "jd": [2461154.30, 2461154.35, 2461154.40],
                  "mag_calib": [12.1, 12.2, 12.3], "err": [0.01, 0.02, 0.03],
                  "flag": ["normal", "normal", "low_snr"]}).to_csv(lc / "lightcurve_11.csv", index=False)
    (lc / "comp_quality_11.json").write_text(json.dumps({"broeg_weights": {"21": 2.0, "22": 1.0}}))
    summary = pd.DataFrame({"catalog_id": ["11", "12"], "_cid": ["11", "12"], "n_frames": [3, 0],
                            "lc_rms": [0.02, np.nan], "vsx_name": ["X Cnv", "Y CVn"], "vsx_type": ["EA", "EW"],
                            # an absolute path from another computer must not matter
                            "lc_csv": ["C:\\\\ASTRO\\\\lightcurve_11.csv", ""]})
    comp = pd.DataFrame({"target_catalog_id": ["11", "11"], "_tcid": ["11", "11"], "catalog_id": ["21", "22"],
                         "mag": [10.0, 10.5], "bp_rp": [0.8, 0.9], "comp_rms": [0.011, 0.012],
                         "comp_tier": [1.0, 2.0]})
    return TargetPages(tmp_path, summary_df=summary, at_df=pd.DataFrame(), comp_df=comp, cfg=None)


def test_lc_stats_from_canonical_light_curve(tmp_path) -> None:
    p = _pages(tmp_path)
    st = p.lc_stats_by_cid["11"]
    assert st["n_points"] == 2  # flagged point excluded
    assert abs(st["merr_med"] - 1.0857362 * 0.015) < 1e-6


def test_measured_targets_and_page(tmp_path) -> None:
    p = _pages(tmp_path)
    assert _measured_cids(p) == ["11"]
    pg = p.page(star_fields(p.summary_row("11"), p.lc_stats_by_cid["11"]), obs_group="NoFilter_60_2")
    assert pg["name"] == "X Cnv" and len(pg["comp_rows"]) == 2
    html = _comp_table_html(pg)
    assert "I/355/gaiadr3&amp;Source=21" in html and "<th style='padding:6px 8px;text-align:left'>tier</th>" in html


def test_tier_text_and_ut_ticks() -> None:
    assert _tier_text("1.0") == "1" and _tier_text("TIER3_x") == "3" and _tier_text("") == "-"
    jd = np.array([2461154.30, 2461154.45])
    vals, labs = _ut_ticks(jd - 2461154.0, 2461154.0)  # G14-17: offset from star_page_m3.lc_plot_data
    assert labs[0] == "20:00" and abs(vals[0] - (2461154.0 + 20 / 24.0 - 0.5 - 2461154.0 + 0.0)) < 1e-9
