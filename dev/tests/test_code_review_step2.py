"""CODE-REVIEW-2026-10 step 2: result-changing bug fixes (Milan 2026-10-22)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from astropy.io import fits

import pipeline  # noqa: F401  app import order
from config import KNOWN_REMOVED_KEYS, AppConfig


def test_removed_parameters_are_gone() -> None:
    cfg = AppConfig()
    for k in ("comp_pool_derived_admission", "plate_scale_arcsec_per_px", "phase01_plate_scale_arcsec_per_px",
              "photon_transfer_ci_max_width_factor", "trust_flag_enabled"):
        assert not hasattr(cfg, k), k
        assert k in KNOWN_REMOVED_KEYS, k


def test_photometric_gain_uses_equipment_and_container_scale(tmp_path: Path) -> None:
    import sqlite3

    from rig_photometry import photometric_gain_read_noise

    db = tmp_path / "v.sqlite3"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE EQUIPMENTS (ID INTEGER, ADC_BITS INTEGER, SIGMA_SYS_MAG REAL, GAIN_ADU REAL, READNOISE_E REAL)")
    con.execute("INSERT INTO EQUIPMENTS VALUES (7, 14, NULL, 3.2, 1.6)")
    con.commit()
    con.close()
    cfg = AppConfig()
    cfg.database_path = str(db)
    pg = photometric_gain_read_noise(fits.Header(), cfg, 7)
    assert pg.gain_native == pytest.approx(3.2) and pg.container_scale == 4.0
    assert pg.gain == pytest.approx(0.8) and pg.read_noise == pytest.approx(1.6)
    # header e-/ADU card wins (session truth)
    h = fits.Header()
    h["EGAIN"] = (2.0, "electrons per ADU")
    assert photometric_gain_read_noise(h, cfg, 7).gain_native == pytest.approx(2.0)


def test_load_photometric_gain(tmp_path: Path) -> None:
    from aperture_pertarget import load_photometric_gain

    (tmp_path / "aperture_per_target.json").write_text(json.dumps({"gain": 0.79, "read_noise": 1.6}))
    assert load_photometric_gain(tmp_path) == (pytest.approx(0.79), pytest.approx(1.6))
    g, rn = load_photometric_gain(tmp_path / "missing")
    assert g != g and rn != rn  # NaN


def test_forced_rows_copy_masterstar_mag() -> None:
    from forced_photometry import _MASTERSTAR_COPY_COLS

    assert "mag" in _MASTERSTAR_COPY_COLS and "catalog_mag" not in _MASTERSTAR_COPY_COLS
    assert "gaia_dr3_variable_catalog" in _MASTERSTAR_COPY_COLS


def test_selection_requires_plate_scale(tmp_path: Path) -> None:
    import inspect

    import target_comp_selection

    src = inspect.getsource(target_comp_selection.run_target_and_comp_selection)
    assert "or 1.3" not in src and "plate scale not derivable" in src


def test_export_fails_closed_without_trust(tmp_path: Path) -> None:
    from export_reports import DONTSUBMIT_DIRNAME, export_lightcurve_reports

    lc = pd.DataFrame({"jd": [2459000.499, 2459000.599], "bjd": [2459000.5, 2459000.6], "mag_calib": [12.0, 12.01],
                       "err": [0.02, 0.02], "flag": ["normal", "normal"], "airmass": [1.2, 1.3],
                       "aperture_r_px": [5.0, 5.2], "time_base": ["BJD_TDB", "BJD_TDB"]})
    paths = export_lightcurve_reports(
        tmp_path / "reports", pd.Series({"vsx_name": "T STAR", "catalog_id": "123", "vsx_type": "EA"}), lc,
        pd.DataFrame(), pd.Series({"band_system": "V"}), cfg=AppConfig(), obs_group="V_60_2",
    )
    assert paths["aavso"].parent.name == DONTSUBMIT_DIRNAME
    assert "TRUST=RED" in paths["aavso"].read_text()


def test_comp_pool_known_variable_gate() -> None:
    import inspect

    import photometry_comp

    src = inspect.getsource(photometry_comp.build_global_comp_pool)
    assert "COMP_KNOWN_VARIABLE" in src and "analyze_draft_comp_pool" not in src
