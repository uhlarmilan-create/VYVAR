"""Tests for WCS round-trip invertibility gate (F-428-WCS-INV FIX 1/2)."""
from __future__ import annotations

import numpy as np
import pytest
from astropy.wcs import WCS

from wcs_invertibility import (
    apply_post_match_identity_gate_df,
    evaluate_wcs_roundtrip,
    finalize_masterstar_sky_coords,
)


def _tan_wcs(ra: float = 120.0, dec: float = 30.0, scale: float = 2.6) -> WCS:
    w = WCS(naxis=2)
    w.wcs.crpix = [256.0, 256.0]
    w.wcs.crval = [ra, dec]
    w.wcs.cd = np.array([[-scale / 3600.0, 0.0], [0.0, scale / 3600.0]])
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w


def test_roundtrip_pass_consistent_tan():
    w = _tan_wcs()
    gate = evaluate_wcs_roundtrip(w, naxis1=512, naxis2=512, grid=9)
    assert gate["pass"] is True
    assert float(gate["wcs_roundtrip_p99_px"]) < 0.2


def test_roundtrip_fail_high_residuals(monkeypatch):
    def _fake(*_a, **_k):
        return np.full(81, 1.0, dtype=np.float64)

    monkeypatch.setattr("wcs_invertibility.wcs_roundtrip_grid_residuals", _fake)
    gate = evaluate_wcs_roundtrip(_tan_wcs(), naxis1=512, naxis2=512, grid=9)
    assert gate["pass"] is False
    assert float(gate["wcs_roundtrip_p99_px"]) == 1.0


def test_finalize_masterstar_sky_coords_matched_vs_unmatched():
    import pandas as pd

    w = _tan_wcs(ra=209.5, dec=41.0)
    df = pd.DataFrame(
        {
            "x": [100.0, 200.0],
            "y": [150.0, 250.0],
            "catalog_id": ["", "123"],
        }
    )
    out = finalize_masterstar_sky_coords(
        df,
        w,
        gaia_db_path=None,
    )
    assert list(out["coord_source"]) == ["final_wcs", "final_wcs"]
    assert out.loc[0, "ra_deg"] == pytest.approx(float(w.all_pix2world(100, 150, 0)[0]), abs=1e-6)


def test_post_match_identity_gate_drops_bad_assignment():
    import pandas as pd

    w = _tan_wcs()
    x0, y0 = 256.0, 256.0
    ra, de = w.all_pix2world(x0, y0, 0)
    df = pd.DataFrame(
        {
            "x": [x0, x0 + 50.0],
            "y": [y0, y0 + 50.0],
            "catalog_id": ["1110000000000000001", "1110000000000000002"],
            "catalog": ["GAIA_DR3", "GAIA_DR3"],
            "match_sep_arcsec": [0.1, 0.1],
        }
    )
    gmap = {"1110000000000000001": (float(ra), float(de)), "1110000000000000002": (float(ra), float(de))}
    out, counts = apply_post_match_identity_gate_df(df, w, gaia_ra_dec_by_cid=gmap, fwhm_px=3.0)
    assert counts["ok"] >= 1
    assert counts["fail"] >= 1
    assert str(out.loc[1, "catalog_id"]).strip() == ""


def test_evaluate_matched_world2pix_identity_px_near_zero():
    import pandas as pd

    from wcs_invertibility import evaluate_matched_world2pix_identity_px

    w = _tan_wcs()
    x0, y0 = 256.0, 256.0
    ra, de = w.all_pix2world(x0, y0, 0)
    df = pd.DataFrame({"x": [x0], "y": [y0], "catalog_id": ["1110000000000000001"]})
    stats = evaluate_matched_world2pix_identity_px(
        df,
        w,
        gaia_ra_dec_by_cid={"1110000000000000001": (float(ra), float(de))},
    )
    assert stats["matched_world2pix_identity_n"] == 1
    assert stats["matched_world2pix_identity_p99_px"] < 0.01
