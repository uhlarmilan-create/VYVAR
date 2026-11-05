# -*- coding: ascii -*-
"""VYVAR v12 (Milan 2026-11-05): the Gaia top-up query is plain ADQL, a rejected query is not an
outage, and the top-up fetches the field cone only (GAIA-TOPUP-ADQL-2026-11).

Milan's PC (2026-11-05, 4h Cyg N1/N2): every top-up failed with "Error 200: Cannot parse query" -
the SELECT list used NULLIF, which is not ADQL; the failure was reported as "offline", the archive was
then marked unavailable for the rest of the run, and the box was the field +-1 deg (~6 deg2)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import astropy.units as u
import pandas as pd
import pytest
from astropy.coordinates import SkyCoord

import gaia_coverage as gc
import gaia_tap_job as gtj
import net_call

ROOT = Path(__file__).resolve().parents[2]

#: Functions of ADQL 2.0 (IVOA REC 2008, sec. 2.3) and the ADQL 2.1 additions (IVOA REC 2023:
#: LOWER, UPPER, ILIKE operator, IN_UNIT, COALESCE is optional) - the only calls a TAP service must parse.
ADQL_FUNCTIONS = frozenset({
    "ABS", "CEILING", "DEGREES", "EXP", "FLOOR", "LOG", "LOG10", "MOD", "PI", "POWER", "RADIANS",
    "SQRT", "RAND", "ROUND", "TRUNCATE", "ACOS", "ASIN", "ATAN", "ATAN2", "COS", "COT", "SIN", "TAN",
    "COUNT", "AVG", "MAX", "MIN", "SUM", "AREA", "BOX", "CENTROID", "CIRCLE", "CONTAINS", "COORD1",
    "COORD2", "COORDSYS", "DISTANCE", "INTERSECTS", "POINT", "POLYGON", "REGION", "LOWER", "UPPER",
    "IN_UNIT",
})
_KEYWORDS = frozenset({"SELECT", "FROM", "WHERE", "AND", "OR", "NOT", "AS", "TOP", "IN", "BETWEEN"})


def _function_calls(adql: str) -> set[str]:
    return {m.upper() for m in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", adql)} - _KEYWORDS


def _sample_queries() -> dict[str, str]:
    sys.path.insert(0, str(ROOT / "catalogs"))
    import build_gaia_catalog as bgc  # noqa: PLC0415

    box = (317.97, 322.75, 50.34, 53.29)
    return {
        "topup": gc.build_topup_adql(box, g_from=17.5, g_to=20.6),
        "builder_full": bgc.build_adql(0.0, 10.0, 0.0, 10.0, last_ra=False, last_dec=False, mag_limit=17.5,
                                       use_full_source=True),
        "builder_lite": bgc.build_adql(0.0, 10.0, 0.0, 10.0, last_ra=False, last_dec=False, mag_limit=17.5,
                                       use_full_source=False),
        "count": gc.region_count_adql(gc._topup_region(box), table="gaiadr3.gaia_source",
                                      mag_where=gc._topup_mag_where(17.5, 20.6)),
    }


# ------------------------------------------------------------------ 1. plain ADQL
@pytest.mark.parametrize("name", ["topup", "builder_full", "builder_lite", "count"])
def test_archive_queries_call_only_adql_functions(name: str) -> None:
    q = _sample_queries()[name]
    bad = _function_calls(q) - ADQL_FUNCTIONS
    assert not bad, f"{name}: functions the Gaia archive ADQL parser rejects: {sorted(bad)}"


def test_topup_query_parses_with_an_adql_grammar() -> None:
    """Independent check with the ADQL grammar of queryparser (AIP); skipped when not installed."""
    adql = pytest.importorskip("queryparser.adql")
    q = _sample_queries()["topup"]
    adql.ADQLQueryTranslator(q).to_postgresql()  # raises QuerySyntaxError on non-ADQL (e.g. NULLIF)


def test_relative_flux_error_column_is_kept() -> None:
    assert "(phot_g_mean_flux_error / phot_g_mean_flux) AS g_flux_error_rel" in gc.GAIA_SOURCE_SELECT


# ------------------------------------------------------------------ 2. rejected query != offline
class _Job:
    jobid = "1762300000000O"

    def __init__(self, error: str) -> None:
        self._error = error

    def get_phase(self, *, update: bool = False) -> str:
        return "ERROR"

    def get_error(self) -> str:
        return self._error


_ARCHIVE_PARSE_ERROR = (
    "Error 200: \nCannot parse query 'SELECT \n            source_id,\n            ra,\n"
    "            (phot_g_mean_flux_error / NULLIF(phot_g_mean_flux, 0.0)) AS g_flux_error_rel\n"
    "        FROM gaiadr3.gaia_source' for ADQL 2.0 ! Cause: Encountered \"NULLIF\" at line 9, column 38."
)


def _run_job(error: str):
    return gtj.run_tap_job("SELECT 1", what="T", launch=lambda q: _Job(error), clock=lambda: 0.0,
                           sleep=lambda s: None, call=lambda fn, _t, _w: fn(), report=lambda m: None)


def test_error_phase_raises_rejected_with_the_reason_not_the_query_echo() -> None:
    with pytest.raises(gtj.TapJobRejected) as ei:
        _run_job(_ARCHIVE_PARSE_ERROR)
    err = ei.value.archive_error
    assert "'<query>'" in err and "source_id" not in err
    assert 'Encountered "NULLIF" at line 9, column 38' in err, "the parser's reason survives"
    assert isinstance(ei.value, gtj.TapJobFailed), "callers catching TapJobFailed keep working"


def test_rejected_topup_is_query_rejected_and_does_not_mark_the_archive_down(monkeypatch, tmp_path) -> None:
    net_call.clear_unavailable()

    def _rejecting(_adql: str) -> pd.DataFrame:
        raise gtj.TapJobRejected("T: job ERROR", archive_error="Cannot parse query '<query>' ...")

    monkeypatch.setattr(gc, "_fetch_tap", _rejecting)
    monkeypatch.setattr(gc, "_coverage_table_stored", lambda _p: True)
    out = gc.top_up_field(tmp_path / "gaia.db", box=(10.0, 10.5, 20.0, 20.5), g_from=17.5, g_to=19.0)
    assert out["status"] == "query_rejected"
    assert "Cannot parse query" in out["error"]
    assert net_call.unavailable_reason("gaia_tap_topup") is None, "a rejected query is not an outage"
    net_call.clear_unavailable()


# ------------------------------------------------------------------ 3. the field cone only
def test_topup_box_is_the_field_cone_without_a_degree_margin(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def _top_up(db_path, *, box, g_from, g_to, fetch=None):
        seen["box"] = box
        return {"status": "ok", "n_fetched": 0, "n_new": 0, "error": None}

    monkeypatch.setattr(gc, "top_up_field", _top_up)
    monkeypatch.setattr(gc, "covered_depth_g", lambda _db, _box: 17.5)
    monkeypatch.setattr(gc, "_warn_unverified_strips", lambda _db, _box: None)
    center = SkyCoord(320.36 * u.deg, 51.82 * u.deg)  # 4h Cyg, cone r 0.476 deg
    gc.ensure_field_depth("unused.db", center=center, radius_deg=0.476, g_needed=19.5)
    ra0, ra1, de0, de1 = seen["box"]
    assert de0 == pytest.approx(51.82 - 0.476, abs=1e-3) and de1 == pytest.approx(51.82 + 0.476, abs=1e-3)
    assert seen["box"] == gc.field_box(center, 0.476)
    assert not hasattr(gc, "TOPUP_MARGIN_DEG")
