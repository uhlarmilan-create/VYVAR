"""Quick platesolve test for V842 Her draft_343 pre-run."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs.utils import pixel_to_skycoord

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import AppConfig
from importer import observation_group_folder_name, observation_group_key
from vyvar_platesolver import solve_wcs_with_local_gaia

LIGHT = Path(r"D:\V842_Her\Light\V842_Her_Light_001.fits")
VSX_RA = 241.509
VSX_DEC = 50.187


def _wcs_center_deg(hdr) -> tuple[float, float] | None:
    try:
        ny, nx = int(hdr["NAXIS2"]), int(hdr["NAXIS1"])
        sc = pixel_to_skycoord((nx - 1) / 2.0, (ny - 1) / 2.0, hdr, origin=0)
        return float(sc.ra.deg), float(sc.dec.deg)
    except Exception:
        return None


def main() -> None:
    cfg = AppConfig()
    gaia = Path(cfg.gaia_db_path)
    if not LIGHT.is_file():
        raise SystemExit(f"Missing: {LIGHT}")

    with fits.open(LIGHT, memmap=False) as hdul:
        h0 = hdul[0].header
        exp = float(h0.get("EXPTIME", 0) or 0)
        xb = int(h0.get("XBINNING", h0.get("BINNING", 1)) or 1)
        filt = str(h0.get("FILTER", "") or "NoFilter").strip() or "NoFilter"
        obj = h0.get("OBJECT", "")
        gk = observation_group_key(filt, exp, xb)
        setup = observation_group_folder_name(gk)
        print("OBJECT:", obj)
        print("EXPTIME:", exp, "XBINNING:", xb, "FILTER:", repr(filt))
        print("group_key:", gk, "setup_folder:", setup)
        before = _wcs_center_deg(h0)

    print("WCS before solve:", before)
    hdr_ra = float(h0.get("RA", VSX_RA) or VSX_RA)
    hdr_dec = float(h0.get("DEC", VSX_DEC) or VSX_DEC)
    crval_sep = None
    if "CRVAL1" in h0 and "CRVAL2" in h0:
        vsx = SkyCoord(ra=VSX_RA * u.deg, dec=VSX_DEC * u.deg, frame="icrs")
        cr = SkyCoord(ra=float(h0["CRVAL1"]) * u.deg, dec=float(h0["CRVAL2"]) * u.deg, frame="icrs")
        crval_sep = float(cr.separation(vsx).arcsec)
        print(f"Existing SIPS CRVAL vs VSX: {crval_sep:.1f} arcsec")

    print("Running solve_wcs_with_local_gaia (header hints) ...")
    res = solve_wcs_with_local_gaia(
        LIGHT,
        hint_ra_deg=hdr_ra,
        hint_dec_deg=hdr_dec,
        fov_diameter_deg=2.5,
        gaia_db_path=gaia,
        expected_plate_scale_arcsec_per_px=float(cfg.phase01_plate_scale_arcsec_per_px),
        enable_sip=True,
        sip_max_order=3,
    )
    print("solve result keys:", {k: res.get(k) for k in ("solved", "reason", "rms_px", "n_matched") if k in res})

    with fits.open(LIGHT, memmap=False) as hdul:
        after = _wcs_center_deg(hdul[0].header)

    print("WCS after solve:", after)
    vsx = SkyCoord(ra=VSX_RA * u.deg, dec=VSX_DEC * u.deg, frame="icrs")
    if after:
        cen = SkyCoord(ra=after[0] * u.deg, dec=after[1] * u.deg, frame="icrs")
        sep = cen.separation(vsx).arcsec
        print(f"Separation WCS center vs VSX V0842 Her: {sep:.2f} arcsec")
    else:
        print("No WCS center after solve")


if __name__ == "__main__":
    main()
