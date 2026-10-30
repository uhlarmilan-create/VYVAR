"""G05-39: the synthetic-catalogue hook lives in the dev launcher, not in production code."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import astropy.units as u
import pandas as pd
from astropy.coordinates import SkyCoord

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src_py"))


def _launcher():
    spec = importlib.util.spec_from_file_location("injection_night_run", ROOT / "dev/tools/injection_night_run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_production_has_no_inject_hook() -> None:
    for p in (ROOT / "src_py").glob("*.py"):
        assert "VYVAR_INJECT_CATALOG" not in p.read_text(encoding="utf-8"), p.name


def test_launcher_patches_the_cone_used_by_catalog_match(tmp_path: Path) -> None:
    import catalog_match
    import pipeline_catalog

    mod = _launcher()
    csv = tmp_path / "inj.csv"
    pd.DataFrame({"catalog_id": ["9000000000000000001", "9000000000000000002"], "ra_deg": [10.0, 30.0],
                  "dec_deg": [20.0, 20.0], "mag": [14.0, 14.0], "bp_rp": [1.0, 1.0]}).to_csv(csv, index=False)
    orig_pc, orig_cm = pipeline_catalog._query_gaia_local, catalog_match._query_gaia_local
    pipeline_catalog._query_gaia_local = lambda **kw: pd.DataFrame(
        {"catalog_id": ["1"], "ra_deg": [10.01], "dec_deg": [20.0], "mag": [12.0], "bp_rp": [0.8], "catalog": ["GAIA_DR3"]})
    try:
        mod.install(csv)
        assert catalog_match._query_gaia_local is pipeline_catalog._query_gaia_local
        out = catalog_match._query_gaia_local(center=SkyCoord(10 * u.deg, 20 * u.deg), radius_deg=0.5,
                                              gaia_db_path=None, max_mag=15.0)
        assert list(out["catalog_id"]) == ["1", "9000000000000000001"]  # only the star inside the cone
        assert mod._USED["n_calls"] == 1
    finally:
        pipeline_catalog._query_gaia_local, catalog_match._query_gaia_local = orig_pc, orig_cm
