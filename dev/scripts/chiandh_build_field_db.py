#!/usr/bin/env python3
"""Build chiandh field DB via ADQL (faster than cone_search in Galactic plane)."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401  (repo layout: src_py + dev on sys.path)
_ROOT = _bootstrap.REPO_ROOT
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.build_field_db import build_field_db  # noqa: E402


def main() -> int:
    out = _ROOT / "GAIA_DR3" / "vyvar_gaia_dr3_chiandh_field.db"
    result = _ROOT / "GAIA_DR3" / "vyvar_gaia_dr3_chiandh_field_build.json"

    sys.path.insert(0, str(_ROOT / "scripts"))
    import pilot_palomar7_deep_gaia_ab as pal7  # noqa: E402

    pal7.PAL_RA = 35.15
    pal7.PAL_DEC = 57.13
    pal7.CONE_RADIUS_DEG = 0.75
    pal7.FIELD_DB = out
    pal7.MAG_LIMIT_INITIAL = 19.5
    pal7.MAG_LIMIT_CAP = 19.5

    orig_part_a2 = pal7.part_a2_astroquery_cone

    def _adql_first(*args, **kwargs):
        from astropy.coordinates import SkyCoord  # noqa: PLC0415
        import astropy.units as u  # noqa: PLC0415
        from astroquery.gaia import Gaia  # noqa: PLC0415
        import pandas as pd  # noqa: PLC0415

        center = SkyCoord(35.15 * u.deg, 57.13 * u.deg, frame="icrs")
        mag_lim = 19.5
        adql = f"""
        SELECT source_id, ra, dec, phot_g_mean_mag AS g_mag, bp_rp,
               phot_bp_mean_mag, phot_rp_mean_mag, parallax, pmra, pmdec
        FROM gaiadr3.gaia_source
        WHERE 1=CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', 35.15, 57.13, 0.75)
        )
        AND phot_g_mean_mag <= {mag_lim}
        """
        job = Gaia.launch_job_async(adql)
        df = job.get_results().to_pandas()
        meta = {
            "method": "adql_minimal",
            "mag_limit_used": mag_lim,
            "row_count": int(len(df)),
            "capped": False,
        }
        return df, meta

    pal7.part_a2_astroquery_cone = _adql_first
    try:
        report = build_field_db(
            center_ra=35.15,
            center_dec=57.13,
            radius_deg=0.75,
            out_path=out,
            mag_limit_initial=19.5,
            mag_limit_cap=19.5,
            result_path=result,
        )
    finally:
        pal7.part_a2_astroquery_cone = orig_part_a2

    import json

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
