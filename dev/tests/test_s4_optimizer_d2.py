# -*- coding: ascii -*-
"""D2: optimizer refit guard + entry WCS backup."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS

from astrometry_optimizer import (
    _FITS_KEY_WCS_BACKUP,
    _WCS_BACKUP_PREFIX,
    backup_entry_wcs,
    d2_refit_should_accept,
    evaluate_d2_refit,
    entry_wcs_sidecar_path,
)


def _tan_header(*, crval1: float = 100.0, crval2: float = 40.0, scale_deg: float = 0.0002) -> fits.Header:
    h = fits.Header()
    h["NAXIS"] = 2
    h["NAXIS1"] = 200
    h["NAXIS2"] = 200
    h["CTYPE1"] = "RA---TAN"
    h["CTYPE2"] = "DEC--TAN"
    h["CRPIX1"] = 100.0
    h["CRPIX2"] = 100.0
    h["CRVAL1"] = float(crval1)
    h["CRVAL2"] = float(crval2)
    h["CDELT1"] = -float(scale_deg)
    h["CDELT2"] = float(scale_deg)
    h["CUNIT1"] = "deg"
    h["CUNIT2"] = "deg"
    h["RADESYS"] = "ICRS"
    return h


def _write_fits(path: Path, hdr: fits.Header) -> None:
    data = np.zeros((200, 200), dtype=np.float32)
    fits.PrimaryHDU(data=data, header=hdr).writeto(path, overwrite=True)


def _honest_table(*, n: int, w: WCS, resid_px: float) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    x = rng.uniform(20.0, 180.0, size=n)
    y = rng.uniform(20.0, 180.0, size=n)
    ra, dec = w.all_pix2world(x, y, 0)
    # Offset detections so residual is ~resid_px on this WCS.
    x_det = x + float(resid_px)
    return pd.DataFrame(
        {
            "x": x_det,
            "y": y,
            "ra_deg": ra,
            "dec_deg": dec,
            "vy_identity_gate": ["ok"] * n,
            "catalog_id": [str(i) for i in range(n)],
        }
    )


def test_d2_backup_cards_and_sidecar(tmp_path: Path) -> None:
    fp = tmp_path / "MASTERSTAR.fits"
    _write_fits(fp, _tan_header())
    st = backup_entry_wcs(fp)
    assert st["backup_written"] is True
    hdr = fits.getheader(fp)
    assert bool(hdr.get(_FITS_KEY_WCS_BACKUP))
    assert any(str(k).replace("HIERARCH ", "").startswith(_WCS_BACKUP_PREFIX) for k in hdr.keys())
    side = entry_wcs_sidecar_path(fp)
    assert side.is_file()
    assert "CTYPE1" in side.read_text(encoding="ascii")
    st2 = backup_entry_wcs(fp)
    assert st2["already_present"] is True
    assert st2["backup_written"] is False


def test_d2_347_pair_80px_rejected_header_unchanged(tmp_path: Path) -> None:
    fp = tmp_path / "MASTERSTAR.fits"
    hdr0 = _tan_header()
    _write_fits(fp, hdr0)
    backup_entry_wcs(fp)
    before = Path(fp).read_bytes()
    w = WCS(fits.getheader(fp))
    df = _honest_table(n=347, w=w, resid_px=80.0)
    d2 = evaluate_d2_refit(w_entry=w, w_candidate=w, df=df, fwhm_dao_px=1.25)
    assert d2["rejected"] is True
    assert d2["n"] == 347
    assert float(d2["rms_sip"]) > 70.0
    after = Path(fp).read_bytes()
    assert after == before
    hdr = fits.getheader(fp)
    assert bool(hdr.get(_FITS_KEY_WCS_BACKUP))
    assert float(hdr["CRVAL1"]) == float(hdr0["CRVAL1"])


def test_d2_2618_pair_085px_accepted() -> None:
    w = WCS(_tan_header())
    df = _honest_table(n=2618, w=w, resid_px=0.85)
    d2 = evaluate_d2_refit(w_entry=w, w_candidate=w, df=df, fwhm_dao_px=1.25)
    assert d2["rejected"] is False
    assert d2["n"] == 2618
    assert float(d2["rms_sip"]) < 1.2


def test_d2_worsening_p95_rejected() -> None:
    ok, reason = d2_refit_should_accept(
        rms_sip=0.85,
        n_honest=2618,
        p95_entry=0.40,
        p95_candidate=0.90,
        fwhm_dao_px=1.25,
    )
    assert ok is False
    assert "p95" in reason
    w_e = WCS(_tan_header(crval1=100.0))
    w_c = WCS(_tan_header(crval1=100.02))  # ~72" ~ 100 px at 0.72"/px scale_deg=0.0002
    df = _honest_table(n=80, w=w_e, resid_px=0.4)
    d2 = evaluate_d2_refit(w_entry=w_e, w_candidate=w_c, df=df, fwhm_dao_px=1.25)
    assert d2["rejected"] is True
    assert float(d2["p95_candidate"]) > float(d2["p95_entry"])
