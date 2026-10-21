"""Batch E re-cut #2: E.1 pairing, E.4 N_equiv, E.5 admission gate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT / "src_py"))
sys.path.insert(0, str(ROOT / "dev" / "scripts"))

from config import AppConfig
from pipeline import _dao_detection_threshold_adu, _apply_dao_centroid_wcs_guard


def test_part0c_pairs_on_source_file_not_position() -> None:
    from audit_stage3_part0c_cohort_delta import _delta_table

    adf = pd.DataFrame(
        {
            "source_file": ["a.csv", "b.csv"],
            "mag_calib_final": [10.0, 11.0],
            "err": [0.01, 0.02],
            "n_good_comp": [3, 3],
            "trust_flag": ["GREEN", "GREEN"],
        }
    )
    rdf = pd.DataFrame(
        {
            "source_file": ["b.csv", "a.csv"],
            "mag_calib_final": [11.01, 10.01],
            "err": [0.021, 0.011],
            "n_good_comp": [3, 3],
            "trust_flag": ["GREEN", "GREEN"],
        }
    )
    tmp = Path(__file__).parent / "_tmp_batch_e_lc"
    an = tmp / "anchor"
    rb = tmp / "rebuild"
    an.mkdir(parents=True, exist_ok=True)
    rb.mkdir(parents=True, exist_ok=True)
    adf.to_csv(an / "lightcurve_t1.csv", index=False)
    rdf.to_csv(rb / "lightcurve_t1.csv", index=False)
    out = _delta_table(rb, an, "t1")
    assert out is not None
    assert len(out) == 2
    by_sf = out.set_index("source_file")
    assert abs(float(by_sf.loc["a.csv", "delta_mag"]) - 0.01) < 1e-9
    assert abs(float(by_sf.loc["b.csv", "delta_mag"]) - 0.01) < 1e-9


def test_dao_n_equiv_threshold_uses_measured_n() -> None:
    cfg = AppConfig()
    cfg.detection_threshold_sigma = 3.78
    thr, n_eff = _dao_detection_threshold_adu(55.63, cfg=cfg, dao_threshold_sigma=3.8)
    assert abs(n_eff - 3.78) < 1e-9
    assert abs(thr - 3.78 * 55.63) < 0.01


def test_dao_centroid_wcs_guard_replaces_large_shift() -> None:
    x = np.array([100.0, 49.0])
    y = np.array([100.0, 50.0])
    matched = np.array([True, True])
    safe = np.array([0, 1])
    master = pd.DataFrame({"x": [100.0, 52.0], "y": [100.0, 50.0]})
    xo, yo, nfb = _apply_dao_centroid_wcs_guard(
        x, y, matched=matched, safe=safe, master_df=master, fwhm_px=2.0, max_shift_fwhm=1.0,
    )
    assert nfb == 1
    assert abs(float(xo[0]) - 100.0) < 1e-9
    assert abs(float(xo[1]) - 52.0) < 1e-9


def test_saturation_fraction_fields_replaced_by_linearity_frac() -> None:
    """SAT-LIN-2026-10: admission/zone saturation fractions replaced by LINEARITY_FRAC."""
    from pipeline_constants import LINEARITY_FRAC

    cfg = AppConfig()
    assert not hasattr(cfg, "admission_sat_peak_frac")
    assert not hasattr(cfg, "saturate_limit_fraction")
    assert abs(float(LINEARITY_FRAC) - 0.90) < 1e-12


def test_lacosmic_config_removed() -> None:
    """L.A.Cosmic removed 2026-08-12 (ate undersampled star cores on wide-field rig)."""
    cfg = AppConfig()
    assert not hasattr(cfg, "enable_lacosmic")
    assert not hasattr(cfg, "lacosmic_sigclip")
    assert not hasattr(cfg, "lacosmic_objlim")
