# -*- coding: ascii -*-
"""INV-MATCH-IDENTITY-01: one identity, one gate, no name rehydration."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from gaia_catalog_id import catalog_id_series_for_masterstars_export
from invariants_runtime import InvariantViolation, assert_inv_match_identity_01
from masterstar_gaia_accounting import lock_existing_and_leftover_assign
from wcs_invertibility import (
    accumulate_identity_gate,
    apply_post_match_identity_gate_df,
    empty_identity_gate_acc,
)


def _tan_wcs(ra: float = 120.0, dec: float = 30.0, scale: float = 2.6) -> WCS:
    w = WCS(naxis=2)
    w.wcs.crpix = [256.0, 256.0]
    w.wcs.crval = [ra, dec]
    w.wcs.cd = np.array([[-scale / 3600.0, 0.0], [0.0, scale / 3600.0]])
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w


def test_export_does_not_copy_name_onto_empty_catalog_id() -> None:
    cid = "1112112413285008896"
    df = pd.DataFrame({"catalog_id": [""], "name": [cid]})
    out = catalog_id_series_for_masterstars_export(df)
    assert str(out.iloc[0]).strip() == ""


def test_gate_and_csv_roundtrip_clears_name_and_catalog_id(tmp_path: Path) -> None:
    from pipeline import _vyvar_df_to_csv

    w = _tan_wcs()
    x0, y0 = 256.0, 256.0
    ra, de = w.all_pix2world(x0, y0, 0)
    cid = "1112112413285008896"
    df = pd.DataFrame(
        {
            "name": [cid],
            "catalog_id": [cid],
            "catalog": ["GAIA_DR3"],
            "x": [x0 + 60.0],
            "y": [y0],
            "mag": [12.0],
            "b_v": [0.5],
            "bp_rp": [0.6],
            "catalog_mag": [12.0],
            "phot_g_mean_mag": [12.0],
            "match_sep_arcsec": [80.0],
            "gaia_nss": [0.0],
            "gaia_qso": [0.0],
            "gaia_gal": [0.0],
        }
    )
    gmap = {cid: (float(ra), float(de))}
    out, counts = apply_post_match_identity_gate_df(
        df, w, gaia_ra_dec_by_cid=gmap, fwhm_px=1.25
    )
    assert counts["fail"] == 1
    assert str(out.loc[0, "catalog_id"]).strip() == ""
    assert str(out.loc[0, "name"]).startswith("DET_")
    assert str(out.loc[0, "vy_identity_gate"]) == "fail"
    assert math.isfinite(float(out.loc[0, "gaia_dao_resid_px"]))
    assert float(out.loc[0, "gaia_dao_resid_px"]) > 3.0 * 1.25

    p = tmp_path / "masterstars_full_match.csv"
    _vyvar_df_to_csv(out, p)
    back = pd.read_csv(p, dtype={"catalog_id": str, "name": str})
    cid_back = str(back.loc[0, "catalog_id"]).strip()
    assert cid_back in ("", "nan")
    assert str(back.loc[0, "name"]).startswith("DET_")


def test_identity_gate_acc_three_passes() -> None:
    acc = empty_identity_gate_acc()
    for n_out in (10, 8, 6):
        acc = accumulate_identity_gate(acc, {"ok": 2, "warn": 1, "fail": 3}, n_out)
    assert acc["passes"] == 3
    assert acc["ok"] == 6
    assert acc["warn"] == 3
    assert acc["fail"] == 9
    assert acc["n_matched_out"] == 6


def test_optimizer_entry_stale_ids_raise(tmp_path: Path) -> None:
    from astrometry_optimizer import optimize_masterstar_matches

    w = _tan_wcs()
    hdr = w.to_header()
    hdu = fits.PrimaryHDU(np.zeros((32, 32), dtype=np.float32), header=hdr)
    fp = tmp_path / "MASTERSTAR.fits"
    hdu.writeto(fp)
    ids = [str(1_000_000_000_000_000_000 + i) for i in range(10)]
    csv = tmp_path / "masterstars.csv"
    pd.DataFrame(
        {
            "x": np.arange(10, dtype=np.float64) + 1.0,
            "y": np.arange(10, dtype=np.float64) + 1.0,
            "catalog_id": ids,
            "name": ids,
            "ra_deg": np.full(10, 120.0),
            "dec_deg": np.full(10, 30.0),
        }
    ).to_csv(csv, index=False)
    with pytest.raises(InvariantViolation, match="INV-MATCH-IDENTITY-01"):
        optimize_masterstar_matches(
            masterstars_csv=csv,
            masterstar_fits=fp,
            gaia_db_path=tmp_path / "missing.db",
            output_csv=tmp_path / "out.csv",
            identity_gate_n_out=2,
            fwhm_dao_px=1.25,
        )


def test_assert_inv_match_identity_01_limit() -> None:
    assert_inv_match_identity_01(n_in=2, n_out_of_gate=2)
    with pytest.raises(InvariantViolation, match="INV-MATCH-IDENTITY-01"):
        assert_inv_match_identity_01(n_in=10, n_out_of_gate=2)


def test_d4_lock_reject_uses_identity_fail_not_lock_tol() -> None:
    """D4: 2.6 px at FWHM 3.3 stays locked; 12 px is rejected."""
    cid_keep = "1112112413285008896"
    cid_rej = "1112115024625070720"
    fwhm = 3.3
    identity_fail = 3.0 * fwhm
    lock_tol = 2.5
    gaia = pd.DataFrame(
        {
            "catalog_id": [cid_keep, cid_rej],
            "x_gaia": [100.0, 200.0],
            "y_gaia": [100.0, 200.0],
        }
    )
    det_x = np.array([102.6, 212.0], dtype=np.float64)
    det_y = np.array([100.0, 200.0], dtype=np.float64)
    locked = {cid_keep: (102.6, 100.0), cid_rej: (212.0, 200.0)}
    det_cids = np.array([cid_keep, cid_rej], dtype=object)
    dtg, _own, modes, rej = lock_existing_and_leftover_assign(
        det_x,
        det_y,
        gaia,
        locked_pairs=locked,
        leftover_radius_px=0.0,
        lock_tol_px=lock_tol,
        identity_fail_px=identity_fail,
        det_catalog_ids=det_cids,
    )
    assert float(identity_fail) == pytest.approx(9.9)
    assert str(modes[0]) == "locked"
    assert int(dtg[0]) == 0
    assert 0 not in {int(i) for i in rej}
    assert 1 in {int(i) for i in rej}
    assert str(modes[1]) == ""
    assert int(dtg[1]) < 0


def test_born_owned_lock_geometry_reject() -> None:
    det_x = np.array([100.0], dtype=np.float64)
    det_y = np.array([100.0], dtype=np.float64)
    gaia = pd.DataFrame(
        {
            "catalog_id": ["1112112413285008896"],
            "x_gaia": [120.0],
            "y_gaia": [100.0],
        }
    )
    locked = {"1112112413285008896": (100.0, 100.0)}
    det_cids = np.array(["1112112413285008896"], dtype=object)
    _dtg, _own, modes, rej = lock_existing_and_leftover_assign(
        det_x,
        det_y,
        gaia,
        locked_pairs=locked,
        leftover_radius_px=3.0,
        lock_tol_px=3.0,
        det_catalog_ids=det_cids,
    )
    assert len(rej) == 1
    assert int(rej[0]) == 0
    assert str(modes[0]) == ""
    assert int(_dtg[0]) < 0
