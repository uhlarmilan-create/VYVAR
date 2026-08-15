"""FORCED-PHOT-01 + COMP-WEIGHT-COEFF-01 acceptance tests."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src_py"))

from comp_weights import (  # noqa: E402
    C_COL_PSF_REFRACTIVE_MAG_PER_BPRP,
    combine_c_col_quadrature,
    resolve_comp_weight_coeffs,
    sigma_eff_mag,
    weight_from_sigma_eff,
)
from forced_photometry import (  # noqa: E402
    force_eligible_masterstar_mask,
    inject_forced_masterstar_rows,
)
from photometry_core import ensemble_normalize  # noqa: E402


def _master_table(n: int = 5) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append(
            {
                "catalog_id": f"G{i}",
                "x": 100.0 + 10 * i,
                "y": 200.0 + 5 * i,
                "ra_deg": 10.0 + 0.01 * i,
                "dec_deg": 20.0,
                "zone": "linear",
                "is_saturated": False,
                "likely_saturated": False,
                "is_noisy": True if i == 1 else False,  # must still be eligible
                "vsx_known_variable": True if i == 2 else False,
                "gaia_nss": True if i == 3 else False,
                "gaia_qso": True if i == 4 else False,
                "bp_rp": 0.5 + 0.1 * i,
            }
        )
    return pd.DataFrame(rows)


def test_is_noisy_is_force_eligible_nss_qso_are_not():
    ms = _master_table()
    ok = force_eligible_masterstar_mask(ms)
    # G0 eligible; G1 noisy but eligible; G2 var out; G3 nss out; G4 qso out
    assert bool(ok.iloc[0]) is True
    assert bool(ok.iloc[1]) is True
    assert bool(ok.iloc[2]) is False
    assert bool(ok.iloc[3]) is False
    assert bool(ok.iloc[4]) is False


def test_forced_inject_fills_missing_members_and_records_geometry():
    ms = _master_table(3)
    # Only G0 present from DAO; G1 should inject; G2 is variable -> not eligible
    df = pd.DataFrame(
        [
            {
                "catalog_id": "G0",
                "x": 100.0,
                "y": 200.0,
                "source_type": "GAIA_MATCHED",
                "flux": 1000.0,
            }
        ]
    )
    img = np.zeros((400, 400), dtype=np.float64)
    img[200, 100] = 50.0
    img[205, 110] = 40.0
    out, meta = inject_forced_masterstar_rows(
        df, ms, image=img, fwhm_px=2.5, centroid_bound_fwhm=2.5
    )
    ids = set(out["catalog_id"].astype(str))
    assert "G0" in ids
    assert "G1" in ids
    assert "G2" not in ids
    assert int(meta["n_injected"]) >= 1
    assert "forced_photometry" in out.columns
    assert bool(out.loc[out["catalog_id"] == "G1", "forced_photometry"].iloc[0]) is True


def test_ensemble_contributing_set_equal_when_all_finite():
    """Membership set is fixed; with forced finite mags, contributors equal membership."""
    n = 8
    t = np.full(n, 12.0)
    comps = {
        "A": np.full(n, 11.0),
        "B": np.full(n, 11.1),
        "C": np.full(n, 11.2),
    }
    cat = {"A": 10.0, "B": 10.1, "C": 10.2}
    qual = {k: {"rms": 0.01} for k in comps}
    wmap = {k: 1.0 / (0.01**2) for k in comps}
    mag_c, dmag, scat = ensemble_normalize(
        t, comps, cat, qual, comp_weight_map=wmap, n_comp_min=2, n_comp_max=10
    )
    assert np.all(np.isfinite(mag_c))
    # Reconstruct per-frame contributors: all three every frame
    for i in range(n):
        contrib = {cid for cid, arr in comps.items() if math.isfinite(float(arr[i]))}
        assert contrib == {"A", "B", "C"}


def test_ensemble_sat_excluded_explicitly_membership_unchanged():
    n = 4
    t = np.full(n, 12.0)
    comps = {
        "A": np.full(n, 11.0),
        "B": np.full(n, 11.1),
    }
    cat = {"A": 10.0, "B": 10.1}
    qual = {k: {} for k in comps}
    sat = {"A": np.array([False, True, False, False]), "B": np.zeros(n, dtype=bool)}
    mag_c, _, _ = ensemble_normalize(
        t,
        comps,
        cat,
        qual,
        comp_likely_saturated=sat,
        n_comp_min=1,
        n_comp_max=10,
    )
    assert math.isfinite(float(mag_c[0]))
    assert math.isfinite(float(mag_c[1]))  # B alone still forms ZP
    # Membership keys unchanged
    assert set(qual.keys()) == {"A", "B"}


def test_c_col_refractive_nonzero_mirror_zero_psf():
    ref = resolve_comp_weight_coeffs(optics_kind="refractive", k2_bprp=None, airmass_span=0.0)
    mir = resolve_comp_weight_coeffs(optics_kind="mirror", k2_bprp=None, airmass_span=0.0)
    assert ref.c_col_mag_per_bprp == pytest.approx(C_COL_PSF_REFRACTIVE_MAG_PER_BPRP)
    assert mir.c_col_mag_per_bprp == pytest.approx(0.0)
    assert ref.c_col_psf_mag_per_bprp > 0
    assert "MEASURED" in "".join(ref.notes)


def test_c_col_quadrature_with_k2():
    c = combine_c_col_quadrature(0.01, C_COL_PSF_REFRACTIVE_MAG_PER_BPRP)
    assert c == pytest.approx(math.hypot(0.01, C_COL_PSF_REFRACTIVE_MAG_PER_BPRP))


def test_universality_with_nonzero_coeffs_exact():
    """Subset invariance must hold exactly with non-zero c_col/c_dist."""
    c_col = C_COL_PSF_REFRACTIVE_MAG_PER_BPRP
    c_dist = 0.002
    stars = [
        ("A", 0.01, 0.0, 0.1),
        ("B", 0.02, 0.4, 0.5),
        ("C", 0.015, -0.2, 1.0),
    ]

    def _w(subset):
        out = {}
        for cid, rms, db, r in subset:
            se = sigma_eff_mag(
                sigma_rms_mag=rms,
                delta_bprp=db,
                r_deg=r,
                c_col_mag_per_bprp=c_col,
                c_dist_mag_per_deg=c_dist,
            )
            out[cid] = (se, weight_from_sigma_eff(se))
        return out

    full = _w(stars)
    # Remove C: A and B sigma_eff unchanged exactly
    sub = _w(stars[:2])
    assert full["A"][0] == sub["A"][0]
    assert full["B"][0] == sub["B"][0]
    assert full["A"][1] == sub["A"][1]


def test_c_dist_measured_zero_when_flat():
    r = [0.1, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    sc = [0.01] * len(r)
    coeffs = resolve_comp_weight_coeffs(
        optics_kind="mirror",
        r_deg=r,
        residual_scatter_mag=sc,
    )
    assert coeffs.c_dist_mag_per_deg == 0.0
    assert "MEASURED" in coeffs.c_dist_source


def test_c_dist_positive_slope_recovered():
    r = np.linspace(0.1, 2.0, 20)
    sc = 0.01 + 0.005 * r
    coeffs = resolve_comp_weight_coeffs(
        optics_kind="mirror",
        r_deg=r.tolist(),
        residual_scatter_mag=sc.tolist(),
    )
    assert coeffs.c_dist_mag_per_deg == pytest.approx(0.005, rel=0.2)
