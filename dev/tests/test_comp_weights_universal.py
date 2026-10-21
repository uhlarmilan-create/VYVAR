"""COMP-ADMIT-03 universality tests for continuous comparison weights."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src_py"))

from comp_pool_noise import (  # noqa: E402
    DerivedPoolThresholds,
    NoiseCurveFit,
    admit_pool_stars,
)


def _thr(**kwargs) -> DerivedPoolThresholds:
    base = dict(
        faint_limit_g=12.0,
        faint_limit_snr_approx=80.0,
        bright_limit_g=8.0,
        bright_upturn_visible=True,
        default_lin_frac=0.85,
        detect_frac_min=1.0,
        detect_frac_rule="test",
        dilution_threshold=0.99,
        dilution_rule="test",
        stability_excess_mad=1.5,
        stability_excess_iqr=1.5,
        stability_excess_inv_eta=0.8,
        stability_rule="p84",
        nonparametric_min_bin_n=8,
        nonparametric_usable_above_g=None,
    )
    base.update(kwargs)
    return DerivedPoolThresholds(**base)


def _fit() -> NoiseCurveFit:
    return NoiseCurveFit(
        n_stars=10,
        gain_e_per_adu=3.17,
        read_noise_e=7.6,
        sky_adu_median=1500.0,
        aperture_area_px_median=36.0,
        zp_inst=22.0,
        sigma_sys_mag=0.01,
        sigma_sys_mag_err=0.001,
        chi2_red=1.0,
        n_fit=10,
        scint_mag_predicted=0.002,
        scint_rel_predicted=0.002,
        scint_airmass_used=1.1,
    )


def _uniform_good_stars(n: int = 20) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append(
            {
                "catalog_id": str(1000 + i),
                "mag_g": 10.0,
                "detect_frac": 1.0,
                "scatter_mad": 0.01,
                "scatter_iqr": 0.01,
                "inv_eta": 0.5,
                "dilution_factor": 1.0,
                "flux_median": 1.0e5,
                "sky_median": 1500.0,
                "aperture_r_median": 3.4,
                "vsx_known_variable": False,
                "gaia_variable_flag": False,
            }
        )
    return pd.DataFrame(rows)


def _old_admit_rank_cuts(stars: pd.DataFrame, fit: NoiseCurveFit, thr: DerivedPoolThresholds) -> pd.DataFrame:
    """Replica of pre-COMP-ADMIT-03 admit_pool_stars rank cuts (fire proof)."""
    rows = []
    for _, st in stars.iterrows():
        reasons = []
        if bool(st.get("vsx_known_variable")):
            reasons.append("vsx")
        if bool(st.get("gaia_variable_flag")):
            reasons.append("gaia")
        dfrac = float(st.get("detect_frac", float("nan")))
        if math.isfinite(thr.detect_frac_min) and (
            not math.isfinite(dfrac) or dfrac < float(thr.detect_frac_min)
        ):
            reasons.append("detect")
        mg = float(st.get("mag_g", float("nan")))
        if thr.faint_limit_g is not None and math.isfinite(mg) and mg > float(thr.faint_limit_g):
            reasons.append("faint")
        sc_mad = float(st.get("scatter_mad", float("nan")))
        sp = 0.005
        stot = math.sqrt(sp * sp + fit.sigma_sys_mag * fit.sigma_sys_mag)
        ratio_mad = sc_mad / stot if stot > 0 else float("nan")
        if thr.stability_excess_mad is not None and math.isfinite(ratio_mad):
            if ratio_mad > float(thr.stability_excess_mad):
                reasons.append("mad")
        inv = float(st.get("inv_eta", float("nan")))
        if thr.stability_excess_inv_eta is not None and math.isfinite(inv):
            if inv > float(thr.stability_excess_inv_eta):
                reasons.append("inv")
        rows.append({"catalog_id": st["catalog_id"], "admit": len(reasons) == 0, "reasons": reasons})
    return pd.DataFrame(rows)


def test_uniformly_good_population_no_exclusion_new_rule():
    stars = _uniform_good_stars(25)
    # Force rank-cut fire on OLD rule: set detect_frac_min=1 and p84-like thr below all ratios.
    thr = _thr(detect_frac_min=1.0, faint_limit_g=9.0, stability_excess_mad=0.5, stability_excess_inv_eta=0.1)
    fit = _fit()
    old = _old_admit_rank_cuts(stars.assign(mag_g=11.0), fit, thr)
    assert int((~old["admit"]).sum()) > 0, "fire proof: old rank/faint cuts must reject"

    new = admit_pool_stars(stars.assign(mag_g=11.0))
    assert int(new["admit"].sum()) == len(new)
    assert int((~new["admit"]).sum()) == 0


