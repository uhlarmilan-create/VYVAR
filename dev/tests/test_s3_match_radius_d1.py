# -*- coding: ascii -*-
"""D1: catalog match radius is one-pass, not match-rate driven."""

from __future__ import annotations

from pathlib import Path

from astropy.io import fits

from dao_gaia_calibration import (
    catalog_match_radius_d1_arcsec,
    solve_rms_px_from_fits_header,
)


def test_d1_520_like_radius_is_floor_12() -> None:
    used, inp = catalog_match_radius_d1_arcsec(
        solve_rms_px=1.44,
        fwhm_dao_px=1.25,
        plate_scale_arcsec_per_px=0.5618,
    )
    assert used == 12.0
    assert float(inp["formula_arcsec"]) < 12.0


def test_d1_516_like_radius_reported() -> None:
    used, inp = catalog_match_radius_d1_arcsec(
        solve_rms_px=1.36,
        fwhm_dao_px=2.5,
        plate_scale_arcsec_per_px=9.774,
    )
    expected = max(12.0, 3.0 * max(1.36, 2.5) * 9.774)
    assert used == expected
    assert abs(float(inp["formula_arcsec"]) - expected) < 1e-9


def test_d1_widening_loops_removed_from_pipeline_source() -> None:
    text = Path("src_py/pipeline.py").read_text(encoding="utf-8")
    assert "0.95 widen iter" not in text
    assert "zhoda %.0f%% < 70" not in text
    assert "cur_thr * 1.12" not in text


def test_solve_rms_from_history_lin() -> None:
    hdr = fits.Header()
    hdr.add_history("VYVAR: SIP rejected by RMS guard (lin=1.442 sip=2.144 ratio=1.486)")
    assert solve_rms_px_from_fits_header(hdr) == 1.442
