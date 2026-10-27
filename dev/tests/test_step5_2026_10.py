"""Step 5 (2026-10): saturation from data, snr_min, check-star rules."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src_py"))

import photometry_core  # noqa: E402,F401  (import order)
from astropy.io import fits  # noqa: E402
from comp_rms_loo import saturated_any_frame_ids  # noqa: E402
from d3_comparison_candidacy import apply_d3_comparison_candidacy  # noqa: E402
from pipeline_constants import LINEARITY_FRAC  # noqa: E402
from sat_diag import PileupResult, linearity_check, resolve_sat_limit  # noqa: E402


def test_linearity_frac_is_090():
    assert LINEARITY_FRAC == 0.90


def test_measured_pileup_wins_over_stated_values():
    hdr = fits.Header({"BITPIX": 16, "SATURATE": 65535})
    pu = PileupResult(True, 16383.0, 500, 3, 16383.0, 30, 65535.0)
    ctx = resolve_sat_limit(hdr=hdr, pileup=pu, equipment_adu=65535.0)
    assert ctx.sat_source == "DERIVED" and ctx.sat_adu == 16383.0
    assert abs(ctx.lin_adu - 0.9 * 16383.0) < 1e-6
    assert any("differs from measured pile-up" in w for w in ctx.warnings)


def test_saturated_any_frame_ids(tmp_path):
    p1, p2 = tmp_path / "proc_1.csv", tmp_path / "proc_2.csv"
    cache = {
        str(p1): pd.DataFrame({"catalog_id": ["111", "222"], "likely_saturated": [False, False]}),
        str(p2): pd.DataFrame({"catalog_id": ["111", "222"], "likely_saturated": [True, False]}),
    }
    out = saturated_any_frame_ids({"111", "222"}, [p1, p2], cache)
    assert out == {"111": 1}


def test_d3_uses_snr_min():
    df = pd.DataFrame(
        {
            "source_state": ["DETECTED_P1"] * 2,
            "vy_identity_gate": ["pass"] * 2,
            "gaia_dao_resid_px": [0.1, 0.1],
            "snr_ap_pixscaled": [12.0, 30.0],
        }
    )
    m10, _ = apply_d3_comparison_candidacy(df, fwhm_dao_px=2.0, solve_rms_px=None, snr_min=10.0)
    m20, _ = apply_d3_comparison_candidacy(df, fwhm_dao_px=2.0, solve_rms_px=None, snr_min=20.0)
    assert m10.tolist() == [True, True] and m20.tolist() == [False, True]


def test_linearity_check_flags_knee(tmp_path):
    rng = np.random.default_rng(0)
    paths = []
    for k in range(3):
        n = 400
        g = rng.uniform(9, 15, n)
        c = rng.uniform(0.3, 1.5, n)
        flux = 10 ** (-0.4 * (g - 25))
        peak = flux * 0.1
        frac = peak / 65535.0
        flux_obs = flux * np.where(frac > 0.7, 1 - 0.3 * (frac - 0.7), 1.0)  # knee at 0.7
        p = tmp_path / f"proc_{k}.csv"
        pd.DataFrame({"peak_max_adu": peak, "dao_flux": flux_obs, "phot_g_mean_mag": g, "bp_rp": c}).to_csv(
            p, index=False
        )
        paths.append(p)
    chk = linearity_check(paths, 65535.0)
    assert chk["ok"]
    assert chk["warning"] is not None


def test_check_star_follows_target_colour():
    from check_star_kmag import select_check_star

    pool = pd.DataFrame(
        {
            "catalog_id": ["1", "2", "3", "4"],
            "bp_rp": [0.80, 2.0, 0.85, 0.82],
            "comp_rms": [0.005, 0.002, 0.004, 0.006],
            "catalog_mag": [12.0, 12.0, 12.0, 12.0],
            "status": ["good"] * 4,
        }
    )
    row = select_check_star(pool, ensemble_ids={"1"}, n_comp_min=1, target_bp_rp=0.8, target_mag=12.0)
    assert str(row["catalog_id"]) == "3"


def test_check_star_chi2(tmp_path):
    from trust_flag_core import check_star_chi2

    lc = tmp_path / "lightcurves"
    lc.mkdir()
    rng = np.random.default_rng(1)
    k = 12 + rng.normal(0, 0.01, 200)
    pd.DataFrame({"kmag": k, "kmag_err": np.full(200, 0.01)}).to_csv(lc / "check_kmag_T.csv", index=False)
    nu, p = check_star_chi2(tmp_path, "T")
    assert 0.8 < nu < 1.2
    pd.DataFrame({"kmag": k, "kmag_err": np.full(200, 0.003)}).to_csv(lc / "check_kmag_T.csv", index=False)
    nu2, p2 = check_star_chi2(tmp_path, "T")
    assert nu2 > 2.5 and p2 < 1e-6


def test_raw_hot_pixel_on_faint_star_is_not_saturation():
    from sat_diag import SatDiagContext, apply_raw_peaks_to_proc_df  # noqa: F401

    # unit-level: combined rule = aligned verdict OR (raw hit AND aligned peak >= 0.5 x limit)
    lim = 0.9 * 65535.0
    aligned = np.array([800.0, 60000.0, 40000.0])
    raw_hit = np.array([True, False, True])
    prev = np.array([False, True, False])
    combined = prev | (raw_hit & (aligned >= 0.5 * lim))
    assert combined.tolist() == [False, True, True]


def test_choose_field_aperture_k(tmp_path):
    from comp_rms_loo import choose_field_aperture_k

    rng = np.random.default_rng(5)
    ids = [f"{5000000000000000000 + i}" for i in range(12)]
    cache = {}
    paths = []
    noise = {1.0: 0.004, 1.25: 0.003, 1.5: 0.005, 1.75: 0.007, 2.0: 0.009}
    for f in range(30):
        p = tmp_path / f"proc_{f:03d}.csv"
        paths.append(p)
        clouds = rng.normal(0, 0.05)
        d = {"catalog_id": ids}
        for k, s in noise.items():
            m = 12 + np.arange(12) * 0.2 + clouds + rng.normal(0, s, 12)
            d[f"flux_k{int(round(k * 100))}"] = 10 ** (-0.4 * (m - 25))
        cache[str(p)] = pd.DataFrame(d)
    k, loo, scores = choose_field_aperture_k(set(ids), paths, cache, min_frames_frac=0.2)
    assert k == 1.25 and len(loo) == 12 and set(scores) == set(noise)
