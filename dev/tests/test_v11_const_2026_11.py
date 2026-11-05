"""VYVAR item 26a+26b, tag const (Milan approved 2026-11-04): CONST-ONE-2026-11.

G12-50: one MAD -> sigma factor, ``stats_core.MAD_TO_SIGMA = 1.4826`` (the former / 0.6745 sites now x 1.4826).
G08-31: one 2.5 / ln 10, ``mag_constants.MAG_ERR_SCALE`` (the short 1.0857 literals are gone).
G07-32: every robust-sigma helper computes its MAD through ``stats_core.median_mad`` / ``mad`` / ``robust_sigma``
        and keeps its own, named degenerate policy; ``dilution`` uses ``sky_separation.angular_distance_deg``.

The helper table below holds the outputs recorded on the code BEFORE this change (v10, 04801fb) on fixed inputs;
the new code must reproduce them bit for bit, except the two former ``/ 0.6745`` helpers (lc_outlier.mad_sigma,
photometry_gate_helpers._robust_scatter_mad), whose rows hold the new x 1.4826 value (+1.37e-5 relative).
"""
from __future__ import annotations

import io
import math
import sys
import tokenize
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src_py"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

#: the only files allowed to spell the factors as numbers
_CONST_HOMES = {"stats_core.py", "mag_constants.py"}
_FORBIDDEN_PREFIXES = ("1.4826", "0.6745", ".6745", "1.0857")


# ------------------------------------------------------------------------------------------------ constants
def test_mad_to_sigma_value_is_1_4826_and_close_to_exact():
    from scipy.stats import norm

    from stats_core import MAD_TO_SIGMA

    assert MAD_TO_SIGMA == 1.4826
    assert abs(MAD_TO_SIGMA - 1.0 / norm.ppf(0.75)) < 3e-5


def test_mag_err_scale_is_exact_2p5_over_ln10():
    from mag_constants import MAG_ERR_SCALE

    assert MAG_ERR_SCALE == 2.5 / math.log(10.0)


def _numeric_literal_hits() -> list[str]:
    hits: list[str] = []
    for p in sorted(_SRC.rglob("*.py")):
        if p.name in _CONST_HOMES or "__pycache__" in p.parts:
            continue
        text = p.read_bytes()
        for tok in tokenize.tokenize(io.BytesIO(text).readline):
            if tok.type == tokenize.NUMBER and tok.string.startswith(_FORBIDDEN_PREFIXES):
                hits.append(f"{p.relative_to(_ROOT)}:{tok.start[0]}: {tok.string}")
    return hits


def test_guard_no_mad_or_mag_err_factor_literals_outside_their_homes():
    """Numbers only (tokenize NUMBER): text in strings, docstrings and comments may describe the factor."""
    hits = _numeric_literal_hits()
    assert not hits, "use stats_core.MAD_TO_SIGMA / mag_constants.MAG_ERR_SCALE:\n" + "\n".join(hits)


def test_guard_scanner_sees_numbers_not_text():
    src = b'x = 1.4826 * y  # 1.4826\ns = "1.0857"\n'
    toks = [t.string for t in tokenize.tokenize(io.BytesIO(src).readline) if t.type == tokenize.NUMBER]
    assert toks == ["1.4826"]


def test_removed_constant_names_are_gone():
    import calib_library
    import cal_diag
    import comp_frame_normalize
    import comp_rms_loo
    import dao_reconcile
    import lc_outlier
    import pipeline_constants

    gone = [
        (pipeline_constants, "MAD_CONSISTENCY"), (pipeline_constants, "_N_STARS_DIAG_MAD_TO_SIGMA"),
        (calib_library, "MAD_SIGMA"), (cal_diag, "CAL_MAD_SIGMA"), (comp_frame_normalize, "_MAD_SIGMA_SCALE"),
        (lc_outlier, "_MAD_CONSISTENCY"), (lc_outlier, "_MAD_TO_SIGMA"), (comp_rms_loo, "LN10_OVER_2P5"),
        (dao_reconcile, "SIGMA_G_SNR_TERM"),
    ]
    left = [f"{m.__name__}.{n}" for m, n in gone if hasattr(m, n)]
    assert not left, left


def test_modules_share_the_one_constant_object():
    import frame_metrics
    import frame_qc
    import gain_ptc
    import psf_chi2_locus
    import stats_core

    for m in (frame_metrics, frame_qc, gain_ptc, psf_chi2_locus):
        assert m.MAD_TO_SIGMA is stats_core.MAD_TO_SIGMA, m.__name__


# ------------------------------------------------------------------------------------------------ MAD core
def test_median_mad_core():
    from stats_core import mad, median_mad

    assert median_mad([1.0, 2.0, 4.0, 8.0, 16.0]) == (4.0, 3.0)
    assert median_mad([1.0, np.nan, 2.0, np.inf, 4.0]) == (2.0, 1.0)
    md, m = median_mad([])
    assert math.isnan(md) and math.isnan(m)
    assert mad([5, 5, 5, 5, 9]) == 0.0
    # float32 input is computed in float32 (calibration._robust_sigma relies on it)
    x32 = np.random.default_rng(1).normal(1.0, 0.01, 1001).astype(np.float32)
    want = float(np.median(np.abs(x32 - np.median(x32))))
    assert mad(x32) == want


# ------------------------------------------------------------------------------------------------ helpers (G07-32)
_RNG_SEED = 20261104
_INPUTS = {
    "const5_9": [5, 5, 5, 5, 9],
    "n2": [1, 3],
    "geo": [1, 2, 4, 8, 16],
    "empty": [],
    "nan": [1.0, float("nan"), 2.0, 4.0, 8.0, float("nan"), 16.0],
    "rand1000": None,  # rng normal(3, 0.7, 1000)
}


def _inp(name: str) -> list[float]:
    if name == "rand1000":
        return np.random.default_rng(_RNG_SEED).normal(3.0, 0.7, 1000).tolist()
    return list(_INPUTS[name])


def _helpers() -> dict:
    import cal_diag as cd
    import calibration as cal
    import comp_frame_normalize as cfn
    import comp_rms_loo as crl
    import epsf_lc_compare as elc
    import gain_ptc as gp
    import lc_outlier as lo
    import photometry_gate_helpers as pgh
    import photometry_lightcurve as plc
    import plain_stats as ps
    import sigma_floor_core as sfc
    import stats_core as sc

    return {
        "cfn.mad_sigma_or_std": lambda v: cfn.mad_sigma_or_std(np.asarray(v, float)),
        "crl.mad_sigma_scale_or_zero": lambda v: crl.mad_sigma_scale_or_zero(np.asarray(v, float)),
        "lc_outlier.mad_sigma": lambda v: lo.mad_sigma(np.asarray(v, float)),
        "gain_ptc._mad_sigma": lambda v: gp._mad_sigma(np.asarray(v, float)),
        "cal_diag._mad_sigma_r": lambda v: cd._mad_sigma_r(np.asarray(v, float)),
        "calibration._robust_sigma": lambda v: cal._robust_sigma(np.asarray(v, float)),
        "calibration._robust_sigma_f32": lambda v: cal._robust_sigma(np.asarray(v, np.float32)),
        "plain_stats.sky_mad_sigma_adu": lambda v: list(ps.sky_mad_sigma_adu(np.asarray(v, float))),
        "pgh._robust_scatter_mad": lambda v: pgh._robust_scatter_mad(np.asarray(v, float)),
        "sigma_floor_core._robust_pull_scale": lambda v: sfc._robust_pull_scale(v),
        "plc.compute_lc_precision_p2p": lambda v: plc.compute_lc_precision_p2p(np.asarray(v, float), ["normal"] * len(v)),
        "stats_core.robust_sigma": lambda v: sc.robust_sigma(v),
        "stats_core.robust_p2p_scatter3": lambda v: sc.robust_p2p_scatter(v, min_points=3),
        "epsf_lc_compare.p2p_scatter": lambda v: elc.p2p_scatter(v),
    }


#: float.hex of each helper's output; recorded on v10 (04801fb) except the two former / 0.6745 helpers
_EXPECTED = {
    'cal_diag._mad_sigma_r|const5_9': '0x0.0p+0',
    'cal_diag._mad_sigma_r|empty': '0x0.0p+0',
    'cal_diag._mad_sigma_r|geo': '0x1.1ca8c154c985fp+2',
    'cal_diag._mad_sigma_r|n2': '0x1.7b8bac710cb29p+0',
    'cal_diag._mad_sigma_r|nan': '0x1.1ca8c154c985fp+2',
    'cal_diag._mad_sigma_r|rand1000': '0x1.71ad181a5c17fp-1',
    'calibration._robust_sigma_f32|const5_9': 'nan',
    'calibration._robust_sigma_f32|empty': 'nan',
    'calibration._robust_sigma_f32|geo': '0x1.1ca8c154c985fp+2',
    'calibration._robust_sigma_f32|n2': '0x1.7b8bac710cb29p+0',
    'calibration._robust_sigma_f32|nan': '0x1.1ca8c154c985fp+2',
    'calibration._robust_sigma_f32|rand1000': '0x1.71ad176f69446p-1',
    'calibration._robust_sigma|const5_9': 'nan',
    'calibration._robust_sigma|empty': 'nan',
    'calibration._robust_sigma|geo': '0x1.1ca8c154c985fp+2',
    'calibration._robust_sigma|n2': '0x1.7b8bac710cb29p+0',
    'calibration._robust_sigma|nan': '0x1.1ca8c154c985fp+2',
    'calibration._robust_sigma|rand1000': '0x1.71ad181a5c17fp-1',
    'cfn.mad_sigma_or_std|const5_9': '0x1.999999999999ap+0',
    'cfn.mad_sigma_or_std|empty': 'inf',
    'cfn.mad_sigma_or_std|geo': '0x1.1ca8c154c985fp+2',
    'cfn.mad_sigma_or_std|n2': '0x1.0000000000000p+0',
    'cfn.mad_sigma_or_std|nan': '0x1.1ca8c154c985fp+2',
    'cfn.mad_sigma_or_std|rand1000': '0x1.71ad181a5c17fp-1',
    'crl.mad_sigma_scale_or_zero|const5_9': '0x0.0p+0',
    'crl.mad_sigma_scale_or_zero|empty': 'nan',
    'crl.mad_sigma_scale_or_zero|geo': '0x1.1ca8c154c985fp+2',
    'crl.mad_sigma_scale_or_zero|n2': 'nan',
    'crl.mad_sigma_scale_or_zero|nan': '0x1.1ca8c154c985fp+2',
    'crl.mad_sigma_scale_or_zero|rand1000': '0x1.71ad181a5c17fp-1',
    'epsf_lc_compare.p2p_scatter|const5_9': '0x0.0p+0',
    'epsf_lc_compare.p2p_scatter|empty': 'nan',
    'epsf_lc_compare.p2p_scatter|geo': '0x1.9291a383e5908p+0',
    'epsf_lc_compare.p2p_scatter|n2': 'nan',
    'epsf_lc_compare.p2p_scatter|nan': '0x1.9291a383e5908p+0',
    'epsf_lc_compare.p2p_scatter|rand1000': '0x1.7da7fa9aec203p-1',
    'gain_ptc._mad_sigma|const5_9': '0x0.0p+0',
    'gain_ptc._mad_sigma|empty': 'nan',
    'gain_ptc._mad_sigma|geo': '0x1.1ca8c154c985fp+2',
    'gain_ptc._mad_sigma|n2': '0x1.7b8bac710cb29p+0',
    'gain_ptc._mad_sigma|nan': '0x1.1ca8c154c985fp+2',
    'gain_ptc._mad_sigma|rand1000': '0x1.71ad181a5c17fp-1',
    'lc_outlier.mad_sigma|const5_9': '0x1.19799812dea11p-40',
    'lc_outlier.mad_sigma|empty': '0x1.19799812dea11p-40',
    'lc_outlier.mad_sigma|geo': '0x1.1ca8c154c985fp+2',
    'lc_outlier.mad_sigma|n2': '0x1.7b8bac710cb29p+0',
    'lc_outlier.mad_sigma|nan': '0x1.1ca8c154c985fp+2',
    'lc_outlier.mad_sigma|rand1000': '0x1.71ad181a5c17fp-1',
    'pgh._robust_scatter_mad|const5_9': '0x0.0p+0',
    'pgh._robust_scatter_mad|empty': 'nan',
    'pgh._robust_scatter_mad|geo': '0x1.1ca8c154c985fp+2',
    'pgh._robust_scatter_mad|n2': 'nan',
    'pgh._robust_scatter_mad|nan': '0x1.1ca8c154c985fp+2',
    'pgh._robust_scatter_mad|rand1000': '0x1.71ad181a5c17fp-1',
    'plain_stats.sky_mad_sigma_adu|const5_9': ['0x1.4000000000000p+2', '0x0.0p+0'],
    'plain_stats.sky_mad_sigma_adu|empty': ['nan', 'nan'],
    'plain_stats.sky_mad_sigma_adu|geo': ['0x1.0000000000000p+2', '0x1.1ca8c154c985fp+2'],
    'plain_stats.sky_mad_sigma_adu|n2': ['0x1.0000000000000p+1', '0x1.7b8bac710cb29p+0'],
    'plain_stats.sky_mad_sigma_adu|nan': ['0x1.0000000000000p+2', '0x1.1ca8c154c985fp+2'],
    'plain_stats.sky_mad_sigma_adu|rand1000': ['0x1.851e94bbdf748p+1', '0x1.71ad181a5c17fp-1'],
    'plc.compute_lc_precision_p2p|const5_9': '0x0.0p+0',
    'plc.compute_lc_precision_p2p|empty': 'nan',
    'plc.compute_lc_precision_p2p|geo': '0x1.9291a383e5908p+0',
    'plc.compute_lc_precision_p2p|n2': 'nan',
    'plc.compute_lc_precision_p2p|nan': '0x1.9291a383e5908p+0',
    'plc.compute_lc_precision_p2p|rand1000': '0x1.7da7fa9aec203p-1',
    'sigma_floor_core._robust_pull_scale|const5_9': '0x0.0p+0',
    'sigma_floor_core._robust_pull_scale|empty': 'nan',
    'sigma_floor_core._robust_pull_scale|geo': '0x1.1ca8c154c985fp+2',
    'sigma_floor_core._robust_pull_scale|n2': 'nan',
    'sigma_floor_core._robust_pull_scale|nan': '0x1.1ca8c154c985fp+2',
    'sigma_floor_core._robust_pull_scale|rand1000': '0x1.71ad181a5c17fp-1',
    'stats_core.robust_p2p_scatter3|const5_9': '0x0.0p+0',
    'stats_core.robust_p2p_scatter3|empty': 'nan',
    'stats_core.robust_p2p_scatter3|geo': '0x1.9291a383e5908p+0',
    'stats_core.robust_p2p_scatter3|n2': 'nan',
    'stats_core.robust_p2p_scatter3|nan': '0x1.9291a383e5908p+0',
    'stats_core.robust_p2p_scatter3|rand1000': '0x1.7da7fa9aec203p-1',
    'stats_core.robust_sigma|const5_9': '0x0.0p+0',
    'stats_core.robust_sigma|empty': 'nan',
    'stats_core.robust_sigma|geo': '0x1.1ca8c154c985fp+2',
    'stats_core.robust_sigma|n2': '0x1.7b8bac710cb29p+0',
    'stats_core.robust_sigma|nan': '0x1.1ca8c154c985fp+2',
    'stats_core.robust_sigma|rand1000': '0x1.71ad181a5c17fp-1',
}


def _hex(r):
    return [float(x).hex() for x in r] if isinstance(r, list) else float(r).hex()


@pytest.mark.parametrize("key", sorted(_EXPECTED))
def test_helper_outputs_identical_to_recorded(key):
    helper, inp = key.split("|")
    assert _hex(_helpers()[helper](_inp(inp))) == _EXPECTED[key]


@pytest.mark.parametrize("helper", ["lc_outlier.mad_sigma", "pgh._robust_scatter_mad"])
def test_former_06745_helpers_now_multiply_by_mad_to_sigma(helper):
    """Fails on v10: these divided the MAD by 0.6745 (1.4825797 instead of 1.4826)."""
    from stats_core import MAD_TO_SIGMA

    v = [1.0, 2.0, 4.0, 8.0, 16.0]  # MAD = 3
    got = _helpers()[helper](v)
    assert got == MAD_TO_SIGMA * 3.0
    assert got != 3.0 / 0.6745
    assert abs(got / (3.0 / 0.6745) - 1.0) == pytest.approx(1.37e-5, rel=1e-3)


def test_degenerate_policies_stay_per_caller():
    import cal_diag as cd
    import comp_frame_normalize as cfn
    import lc_outlier as lo

    assert cfn.mad_sigma_or_std(np.array([5.0, 5.0, 5.0, 5.0, 9.0])) == pytest.approx(1.6)  # std fallback
    assert math.isinf(cfn.mad_sigma_or_std(np.array([])))
    assert lo.mad_sigma(np.array([5.0, 5.0, 5.0, 5.0, 9.0])) == 1e-12  # floor
    assert lo.mad_sigma(np.array([1.0])) == 1e-12
    assert cd._mad_sigma_r(np.array([np.nan])) == 0.0


# ------------------------------------------------------------------------------------------------ 1.0857 (G08-31)
def test_limit_magnitudes_error_uses_exact_2p5_over_ln10():
    """Fails on v10 (1.0857 x sig / flux)."""
    from ccd_noise import limit_magnitudes
    from mag_constants import MAG_ERR_SCALE

    _m, e, below = limit_magnitudes(np.array([1000.0]), np.array([100.0]))
    assert e[0] == MAG_ERR_SCALE * 10.0 / 1000.0
    assert not below[0]


def test_sigma_g_row_uses_mag_err_scale():
    """Fails on v10 (1.0857362 / SNR)."""
    from dao_reconcile import sigma_g_row
    from mag_constants import MAG_ERR_SCALE

    assert sigma_g_row(zp_residual_rms=0.0, snr=7.0) == MAG_ERR_SCALE / 7.0


# ------------------------------------------------------------------------------------------------ haversine (G07-32)
#: (ra1, dec1, ra2, dec2, separation arcsec of the former dilution._haversine_arcsec, v10)
_HAV = [
    (0.0,0.0,180.0,0.0, 648000.0),
    (0.0,89.9999,180.0,89.9999, 0.7200000000325434),
    (10.0,20.0,10.001,20.001, 4.940029887832363),
    (123.4,-89.9998,303.4,-89.9998, 1.4400000000398265),
    (15.0,-30.0,15.0003,-29.9996, 1.7170915892829948),
    (200.0,45.0,200.0,45.0, 0.0),
    (359.9995,0.0,0.0005,0.0, 3.599999999995336),
    (359.99999,-10.0,1e-05,-10.00001, 0.07952158898021602),
]


def test_dilution_has_no_private_haversine():
    import dilution

    assert not hasattr(dilution, "_haversine_arcsec")


@pytest.mark.parametrize("ra1,dec1,ra2,dec2,want", _HAV)
def test_sky_separation_matches_former_dilution_haversine(ra1, dec1, ra2, dec2, want):
    """RA wrap at 0/360 and pairs near both poles included."""
    from sky_separation import angular_distance_deg

    assert abs(angular_distance_deg(ra1, dec1, ra2, dec2) * 3600.0 - want) < 1e-9
