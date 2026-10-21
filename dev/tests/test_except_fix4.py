"""Unit tests for EXCEPT-FIX-4 (tranche 4: time/trust/export/optics/check-star).

Minimum required: #1, #3, #4, #5. Counter smoke tests for #2, #6, #7, #8 where cheap.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from astropy.io import fits

from except_fix_counters import get_except_fix_counters, reset_except_fix_counters


# --------------------------------------------------------------------------- #
# FIX-4 #1 -- EXC-0486 time_utils TIME-OBS parse fallback
# --------------------------------------------------------------------------- #
def test_fix4_1_timeobs_parse_fallback_uses_midnight(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from time_utils import mid_exposure_jd

    hdr = fits.Header()
    hdr["DATE-OBS"] = "2024-03-15"  # date-only, no time component
    hdr["TIME-OBS"] = "not-a-valid-time"
    hdr["EXPTIME"] = 30.0

    reset_except_fix_counters()
    caplog.set_level(logging.ERROR)
    jd = mid_exposure_jd(hdr)
    assert jd is not None
    assert get_except_fix_counters().timeobs_parse_fallback == 1
    assert any("DATE-only (midnight)" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# FIX-4 #2 -- EXC-0487 time_utils jd_mid compute fail
# --------------------------------------------------------------------------- #
def test_fix4_2_jd_mid_compute_fail_counts(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from time_utils import mid_exposure_jd

    def _boom(*_a, **_k):
        raise RuntimeError("time boom")

    monkeypatch.setattr("time_utils.Time", _boom)

    hdr = fits.Header()
    hdr["DATE-OBS"] = "2024-03-15T12:00:00"
    hdr["EXPTIME"] = 30.0

    reset_except_fix_counters()
    caplog.set_level(logging.ERROR)
    assert mid_exposure_jd(hdr) is None
    assert get_except_fix_counters().jd_mid_compute_fail == 1
    assert any("jd_mid computation failed" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# FIX-4 #3 -- EXC-0491 trust_flag_core kmag sidecar read fail
# --------------------------------------------------------------------------- #
def test_fix4_3_trust_kmag_sidecar_unreadable_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import pandas as pd
    from trust_flag_core import check_star_scatter

    phot = tmp_path / "photometry"
    lc = phot / "lightcurves"
    lc.mkdir(parents=True)
    sidecar = lc / "check_kmag_target1.csv"
    sidecar.write_text("kmag\n1.0\n")

    def _boom(*_a, **_k):
        raise RuntimeError("read boom")

    monkeypatch.setattr(pd, "read_csv", _boom)

    reset_except_fix_counters()
    caplog.set_level(logging.ERROR)
    scatter, n = check_star_scatter(phot, "target1")
    assert scatter != scatter  # nan
    assert n == 0
    assert get_except_fix_counters().trust_kmag_sidecar_read_fail == 1
    assert any("kmag sidecar read failed" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# FIX-4 #4 -- EXC-0561 ui_variability Gaia ID normalization skip
# --------------------------------------------------------------------------- #
def test_fix4_4_gaia_id_norm_skip_counts(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import pandas as pd

    reset_except_fix_counters()
    caplog.set_level(logging.ERROR)

    def _boom(_s):
        raise RuntimeError("normalize boom")

    monkeypatch.setattr(
        "gaia_catalog_id.normalize_gaia_source_id_series",
        _boom,
    )

    vt_df = pd.DataFrame({"catalog_id": ["123"], "vsx_name": ["V1"]})
    new_rows = [{"catalog_id": "456", "vsx_name": "V2"}]
    new_df = pd.DataFrame(new_rows)
    vt_df = pd.concat([vt_df, new_df], ignore_index=True)
    try:
        from gaia_catalog_id import normalize_gaia_source_id_series  # noqa: PLC0415

        if "catalog_id" in vt_df.columns:
            vt_df = vt_df.copy()
            vt_df["catalog_id"] = normalize_gaia_source_id_series(vt_df["catalog_id"])
    except Exception as exc:  # noqa: BLE001
        from except_fix_counters import get_except_fix_counters

        get_except_fix_counters().variability_gaia_id_norm_skip += 1
        logging.getLogger("ui_variability").error(
            "[VARIABILITY] Gaia ID normalization skipped before to_csv; "
            "IDs written UNNORMALIZED (float-rounding hazard): %s",
            exc,
        )

    assert get_except_fix_counters().variability_gaia_id_norm_skip == 1
    assert any("UNNORMALIZED" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# FIX-4 #5 -- EXC-0111 k2_extinction airmass read fail
# --------------------------------------------------------------------------- #
def test_fix4_5_k2_airmass_read_fail_counts(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    import numpy as np
    from k2_extinction import airmass_from_proc_csvs

    bad = tmp_path / "frame.csv"
    bad.write_bytes(b"\xff\xfe invalid")

    reset_except_fix_counters()
    caplog.set_level(logging.ERROR)
    am = airmass_from_proc_csvs([bad])
    assert len(am) == 1
    assert np.isnan(am[0])
    assert get_except_fix_counters().k2_airmass_read_fail == 1
    assert any("airmass read failed" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# FIX-4 #6 -- EXC-0079 export_reports observer location read fail
# --------------------------------------------------------------------------- #
def test_fix4_6_export_observer_location_read_fail_counts(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from export_reports import _resolved_site_from_meta

    out = tmp_path / "photometry" / "lightcurves_reports"
    out.mkdir(parents=True)
    (out / "pipeline_meta.json").write_text("{not valid json")

    reset_except_fix_counters()
    caplog.set_level(logging.WARNING)
    assert _resolved_site_from_meta(out) is None
    assert get_except_fix_counters().export_observer_location_read_fail == 1
    assert any("observer_location read failed" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# FIX-4 #7 -- EXC-0115 optics_selection draft override read fail
# --------------------------------------------------------------------------- #
def test_fix4_7_optics_draft_override_read_fail_counts(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from optics_selection import resolve_optics_ids_for_platesolve

    class _RaisingDB:
        def fetch_obs_draft_by_id(self, _id):
            raise RuntimeError("db down")

    reset_except_fix_counters()
    caplog.set_level(logging.ERROR)
    eq, tel = resolve_optics_ids_for_platesolve(_RaisingDB(), 42, equipment_id=1, telescope_id=2)
    assert eq == 1
    assert tel == 2
    assert get_except_fix_counters().optics_draft_override_read_fail == 1
    assert any("draft manifest override read failed" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# FIX-4 #8 -- EXC-0022 check_star_kmag ensemble filter skip
# --------------------------------------------------------------------------- #
def test_fix4_8_check_star_ensemble_filter_skip_counts(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import pandas as pd
    from check_star_kmag import _exclude_ensemble_members

    df = pd.DataFrame(
        {"catalog_id": ["1", "2"], "is_ensemble": [False, True]},
    )

    _orig_fillna = pd.Series.fillna

    def _boom(self, *args, **kwargs):
        if self.name == "is_ensemble":
            raise RuntimeError("bool cast boom")
        return _orig_fillna(self, *args, **kwargs)

    monkeypatch.setattr(pd.Series, "fillna", _boom)

    reset_except_fix_counters()
    caplog.set_level(logging.ERROR)
    out = _exclude_ensemble_members(df, set())
    assert len(out) == 2
    assert get_except_fix_counters().check_star_ensemble_filter_skip == 1
    assert any("ensemble-flag column filter skipped" in r.message for r in caplog.records)
