"""REPORT-AUDIT-2026-10 (Milan 2026-10-21): exports / TRUST / report values."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config import AppConfig
from trust_flag_core import (
    CHECK_DELTA_HARD_MAG,
    CHECK_DELTA_SOFT_MAG,
    ZP_SIGMA_HARD_MAG,
    ZP_SIGMA_SOFT_MAG,
    CompTrustThresholds,
    check_delta_and_zp_sigma,
    classify_warnings,
    trust_level,
)

TH = CompTrustThresholds(min_comps=3, max_comps=8, strong=5)


def _cls(**kw):
    base = dict(n_clean=8, check_scatter=0.01, lc_quality="good", thresholds=TH, n_check=40, n_frames=40)
    base.update(kw)
    hard, soft = classify_warnings(**base)
    return hard, soft, trust_level(8, hard, soft, TH)


def test_trust_thresholds_are_the_agreed_values() -> None:
    assert (CHECK_DELTA_SOFT_MAG, CHECK_DELTA_HARD_MAG) == (0.10, 0.25)
    assert (ZP_SIGMA_SOFT_MAG, ZP_SIGMA_HARD_MAG) == (0.050, 0.150)


@pytest.mark.parametrize(
    "delta, level",
    [(0.05, "GREEN"), (-0.12, "YELLOW"), (0.30, "RED"), (float("nan"), "GREEN")],
)
def test_check_star_delta_levels(delta: float, level: str) -> None:
    assert _cls(check_delta_mag=delta)[2] == level


@pytest.mark.parametrize("sig, level", [(0.02, "GREEN"), (0.08, "YELLOW"), (0.20, "RED")])
def test_zero_point_sigma_levels(sig: float, level: str) -> None:
    assert _cls(zp_sigma_mag=sig)[2] == level


def test_t3_comps_counted_once() -> None:
    _, soft, _ = _cls(n_tier12=5)
    assert sum("T3+" in s for s in soft) == 1


def test_check_delta_and_zp_sigma_from_files(tmp_path: Path) -> None:
    lc = tmp_path / "lightcurves"
    lc.mkdir()
    (lc / "comp_quality_1.json").write_text(json.dumps({
        "band_system": "V",
        "zero_point": {"sigma_zp_data_mag": 0.03, "sigma_zp_model_mag": 0.01},
        "comp_band_mag": {"77": 11.0},
        "check_star": {"catalog_id": "77"},
    }), encoding="utf-8")
    pd.DataFrame({"bjd": [1.0, 2.0, 3.0], "source_file": ["a", "b", "c"], "kmag": [11.10, 11.12, 11.14],
                  "kmag_err": [0.01] * 3}).to_csv(lc / "check_kmag_1.csv", index=False)
    # G11-11: median over the target's exported epochs (its LC rows)
    pd.DataFrame({"source_file": ["a", "b", "c"], "jd": [1.0, 2.0, 3.0], "bjd": [1.0, 2.0, 3.0],
                  "mag_calib": [12.0] * 3, "err": [0.01] * 3, "flag": ["normal"] * 3}).to_csv(
        lc / "lightcurve_1.csv", index=False)
    d, z = check_delta_and_zp_sigma(tmp_path, "1")
    assert d == pytest.approx(0.12) and z == pytest.approx(0.03)
    # band NONE -> not judged
    (lc / "comp_quality_1.json").write_text(json.dumps({"band_system": "NONE"}), encoding="utf-8")
    assert all(np.isnan(v) for v in check_delta_and_zp_sigma(tmp_path, "1"))


def test_header_site_without_elevation_takes_config_altitude() -> None:
    from astropy.io import fits

    from param_resolver import resolve_site

    h = fits.Header()
    h["SITELAT"], h["SITELONG"] = 50.11214, 14.69828
    cfg = AppConfig()
    cfg.observer_lat, cfg.observer_lon, cfg.observer_alt_m = 50.112, 14.698, 310.0
    r = resolve_site(h, cfg=cfg)
    assert r.source == "header" and r.elev == pytest.approx(310.0)
    cfg.observer_lat = 49.0  # another place: the header stays as it is
    assert resolve_site(h, cfg=cfg).elev == pytest.approx(0.0)


def test_target_airmass_differs_from_field_centre() -> None:
    from photometry_lightcurve import target_airmass

    cfg = AppConfig()
    jd = np.array([2461161.40, 2461161.45])
    frame = np.array([1.10, 1.08])
    am = target_airmass(jd, 244.9, 50.8, cfg, site=(50.11, 14.70, 300.0), fallback=frame)
    assert np.all(np.isfinite(am)) and np.all(am >= 1.0)
    assert not np.allclose(am, frame)
    # no site -> frame value kept
    nul = target_airmass(jd, 244.9, 50.8, cfg, site=(0.0, 0.0, 0.0), fallback=frame)
    assert np.allclose(nul, frame)


def test_blend_band_is_gaia_g_for_clear_filters() -> None:
    from gaia_bands import filter_band

    assert filter_band("NoFilter").blend_band == "G"
    assert filter_band("g").blend_band == "g"


def test_report_frame_stem_matches_raw_proc_and_csv_names() -> None:
    from photometry_report import _PhotometryReportBuilder as B

    a = B._frame_stem("C:\\ASTRO\\Raw\\lights\\NoFilter_60_2\\V842_Her_Light_134.fits")
    assert a == B._frame_stem("proc_V842_Her_Light_134.fits") == B._frame_stem("proc_V842_Her_Light_134.csv")


def test_clear_export_dirs_includes_dontsubmit(tmp_path: Path) -> None:
    from export_reports import DONTSUBMIT_DIRNAME
    from run_outputs import clear_export_dirs

    d = tmp_path / "lightcurves_reports" / "aavso" / DONTSUBMIT_DIRNAME
    d.mkdir(parents=True)
    (d / "x.txt").write_text("x")
    (d.parent / "y.txt").write_text("y")
    assert clear_export_dirs(tmp_path) == 2
