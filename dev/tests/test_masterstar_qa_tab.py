"""MSQA-2026-10: MASTERSTAR QA tab facts (plate solve, DAO <-> Gaia match) and overlay."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS

from masterstar_qa_plot import load_masterstar_qa, render_masterstar_overlay


def _bundle(tmp_path):
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crpix = [50.5, 40.5]
    w.wcs.crval = [209.5, 41.0]
    w.wcs.cdelt = [-10.0 / 3600.0, 10.0 / 3600.0]
    hdr = w.to_header()
    hdr.update({"VY_SOLVR": "astrosolve", "VY_SOLMD": "hinted", "VY_SOLLO": 120.4, "VY_SOLNM": 30,
                "VY_SOLRM": 0.2, "XBINNING": 2, "YBINNING": 2, "OBJECT": "TEST", "DATE-OBS": "2026-04-23T23:00:00"})
    rng = np.random.default_rng(1)
    fits.PrimaryHDU(rng.normal(1000, 10, (80, 100)).astype(np.float32), header=hdr).writeto(tmp_path / "MASTERSTAR.fits")
    rows = [dict(x=10 + i, y=20, source_state="DETECTED_P1", gaia_dao_resid_px=0.1 * (i + 1), phot_g_mean_mag=12 + i)
            for i in range(4)]
    rows += [dict(x=60, y=60, source_state="DAO_ONLY", gaia_dao_resid_px=np.nan, phot_g_mean_mag=np.nan),
             dict(x=np.nan, y=np.nan, source_state="catalog_membership", gaia_dao_resid_px=np.nan, phot_g_mean_mag=14.0)]
    pd.DataFrame(rows).to_csv(tmp_path / "masterstars_full_match.csv", index=False)
    ra, dec = w.wcs_pix2world([[30, 30], [40, 50], [5, 5]], 0).T
    pd.DataFrame({"ra_deg": ra, "dec_deg": dec, "mag": [12.0, 14.5, 18.0]}).to_csv(
        tmp_path / "field_catalog_cone.csv", index=False)
    return tmp_path


def test_load_masterstar_qa_facts(tmp_path) -> None:
    qa = load_masterstar_qa(_bundle(tmp_path))
    fr, so, ma = qa["frame"], qa["solve"], qa["match"]
    assert (fr["width_px"], fr["height_px"], fr["binning"]) == (100, 80, "2\u00d72")
    assert abs(fr["scale_arcsec_px"] - 10.0) < 1e-6
    assert abs(fr["fov_w_deg"] - 1000.0 / 3600.0) < 1e-9
    assert so["solver"] == "astrosolve" and so["n_fit"] == 30 and so["rms_px"] == 0.2
    assert (ma["n_detected"], ma["n_matched"], ma["n_unmatched"]) == (5, 4, 1)
    assert abs(ma["matched_pct"] - 80.0) < 1e-9
    assert abs(ma["resid_p50_px"] - 0.25) < 1e-9
    # Gaia layer stops at the matching depth: faintest matched G 15 (no frame_depth in meta).
    assert qa["gaia_depth_g"] == 15.0
    assert qa["gaia_xy"].shape == (2, 2)
    assert qa["matched_xy"].shape == (4, 2) and qa["unmatched_xy"].shape == (1, 2)


def test_gaia_depth_from_frame_depth_meta(tmp_path) -> None:
    b = _bundle(tmp_path)
    (b / "photometry").mkdir()
    (b / "photometry" / "pipeline_meta.json").write_text(json.dumps({"frame_depth": {"g_lim": 18.5}}))
    assert load_masterstar_qa(b)["gaia_xy"].shape == (3, 2)


def test_overlay_layers_change_the_image(tmp_path) -> None:
    qa = load_masterstar_qa(_bundle(tmp_path))
    kw = dict(matched_xy=qa["matched_xy"], unmatched_xy=qa["unmatched_xy"], gaia_xy=qa["gaia_xy"])
    bare, cap = render_masterstar_overlay(qa["image"], show_gaia=False, show_matches=False, **kw)
    full, _ = render_masterstar_overlay(qa["image"], show_gaia=True, show_matches=True, **kw)
    det, cap_d = render_masterstar_overlay(qa["image"], show_gaia=True, show_matches=True, detail=True, **kw)
    assert bare != full and bare[:2] == b"\xff\xd8"
    assert cap.startswith("Whole frame") and cap_d.startswith("Detail")
