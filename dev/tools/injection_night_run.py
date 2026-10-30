# -*- coding: ascii -*-
"""Night run with synthetic catalogue stars (injection-recovery step B, dev only).

usage: python dev/tools/injection_night_run.py --inject-catalog inject_catalog.csv <night_run arguments>

VYVAR measures catalogue stars only (identity join on source_id). The injection-recovery tool
(dev/tools/injection_recovery.py, plan-new / inject-new) puts synthetic stars into raw frames where Gaia
has none and lists them in ``inject_catalog.csv`` (ids above the Gaia range). This launcher adds those
rows to the local Gaia cone of the run and then runs the normal night pipeline (night_run.main).

G05-39 (Milan 2026-10-29): the hook lived in production code (``$VYVAR_INJECT_CATALOG`` in
pipeline_catalog); VYVAR itself never uses it, so it moved here. The cone is queried in the main
process (MASTERSTAR catalogue match); the launcher fails loudly if the injected rows were never added.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src_py"))

LOGGER = logging.getLogger("injection_night_run")
_USED = {"n_calls": 0, "n_rows": 0}


def append_injected(df: pd.DataFrame, extra: pd.DataFrame, center, radius_deg: float, max_mag) -> pd.DataFrame:
    """Rows of ``extra`` inside the cone (and G <= max_mag) appended to the Gaia cone ``df``."""
    import astropy.units as u
    import numpy as np
    from astropy.coordinates import SkyCoord

    if extra.empty:
        return df
    coo = SkyCoord(ra=extra["ra_deg"].astype(float).to_numpy() * u.deg,
                   dec=extra["dec_deg"].astype(float).to_numpy() * u.deg, frame="icrs")
    ex = extra.loc[center.separation(coo).deg <= float(radius_deg) + 1e-9].copy()
    if max_mag is not None:
        ex = ex.loc[pd.to_numeric(ex["mag"], errors="coerce") <= float(max_mag)]
    ex["catalog"] = "GAIA_DR3"
    if "b_v" in df.columns and "b_v" not in ex.columns:
        ex["b_v"] = np.nan
    if "bp_rp" in ex.columns:
        ex["bp_rp"] = pd.to_numeric(ex["bp_rp"], errors="coerce")
    _USED["n_calls"] += 1
    _USED["n_rows"] += int(len(ex))
    LOGGER.warning("[INJECT] %d synthetic catalogue stars added to the Gaia cone (test run)", len(ex))
    if df.empty:
        return ex.reset_index(drop=True)
    return pd.concat([df, ex[[c for c in ex.columns if c in df.columns]]], ignore_index=True)


def install(inject_csv: Path) -> None:
    """Wrap the local Gaia cone query of pipeline_catalog (and its import in catalog_match)."""
    import catalog_match
    import pipeline_catalog

    extra = pd.read_csv(inject_csv, dtype={"catalog_id": str})
    orig = pipeline_catalog._query_gaia_local

    def _query_gaia_local(*, center, radius_deg, gaia_db_path, max_mag=None):
        df = orig(center=center, radius_deg=radius_deg, gaia_db_path=gaia_db_path, max_mag=max_mag)
        return append_injected(df, extra, center, radius_deg, max_mag)

    pipeline_catalog._query_gaia_local = _query_gaia_local
    catalog_match._query_gaia_local = _query_gaia_local


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--inject-catalog", required=True, type=Path)
    args, rest = ap.parse_known_args(argv)
    install(args.inject_catalog)
    import night_run

    rc = int(night_run.main(rest))
    if _USED["n_calls"] == 0:
        print("ERROR: the injected catalogue was never added (Gaia cone not queried in this process)", file=sys.stderr)
        return 3
    print(f"[INJECT] cone queries {_USED['n_calls']}, synthetic rows added {_USED['n_rows']}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
