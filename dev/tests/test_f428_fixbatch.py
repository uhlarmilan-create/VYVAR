"""F-428 fix batch unit tests (VSX path, repair flood, HRD TAP retry, excluded targets)."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest


def test_resolve_variable_targets_csv_draft428_layout(tmp_path: Path) -> None:
    from photometry_core import resolve_variable_targets_csv

    setup = tmp_path / "platesolve" / "NoFilter_60_2"
    phot = setup / "photometry"
    phot.mkdir(parents=True)
    vt = setup / "variable_targets.csv"
    vt.write_text("catalog_id,vsx_name,vsx_type\n123,BO CVn,EA\n", encoding="utf-8")
    comp = phot / "comparison_stars_per_target.csv"
    comp.write_text("catalog_id\n999\n", encoding="utf-8")

    resolved = resolve_variable_targets_csv(comparison_stars_csv=comp)
    assert resolved == vt


def test_compute_rms_vsx_match_from_parent_grandparent(tmp_path: Path) -> None:
    from variability_detector import compute_rms_variability

    setup = tmp_path / "NoFilter_60_2"
    phot = setup / "photometry"
    phot.mkdir(parents=True)
    vsx_id = "1400549806859236864"
    (setup / "variable_targets.csv").write_text(
        f"catalog_id,vsx_name,vsx_type\n{vsx_id},BO CVn,EA\n",
        encoding="utf-8",
    )

    meta = pd.DataFrame(
        {
            "mag": [11.0],
            "ra_deg": [202.0],
            "dec_deg": [42.0],
            "x": [100.0],
            "y": [100.0],
                "is_usable": [True],
                "gaia_dr3_variable_catalog": [False],
                "vsx_known_variable": [False],
            },
        index=[vsx_id],
    )
    meta.index.name = "catalog_id"
    flux = pd.DataFrame({j: [1000.0 + (j % 3)] for j in range(25)}, index=[vsx_id])

    out = compute_rms_variability(
        flux,
        meta,
        [],
        sigma_threshold=2.3,
        vsx_targets_csv=setup / "variable_targets.csv",
        config={"variability_sigma_threshold": 2.3, "variability_mag_limit": 13.0, "variability_min_points_rms": 5},
    )
    row = out.iloc[0]
    assert bool(row["vsx_match"]) is True
    assert bool(row["vsx_known_variable"]) is True
    assert str(row["vsx_name"]) == "BO CVn"


def test_auto_export_excludes_known_vsx_candidate(tmp_path: Path, monkeypatch) -> None:
    from photometry_core import auto_export_variability_candidates_csv

    vsx_id = "1400549806859236864"
    setup = tmp_path / "setup"
    phot = setup / "photometry"
    phot.mkdir(parents=True)
    (setup / "variable_targets.csv").write_text(
        f"catalog_id,vsx_name,vsx_type\n{vsx_id},BO CVn,EA\n",
        encoding="utf-8",
    )
    comp = phot / "comparison_stars_per_target.csv"
    comp.write_text("catalog_id\n999\n", encoding="utf-8")

    rms_df = pd.DataFrame(
        {
            "catalog_id": [vsx_id],
            "x": [100.0],
            "y": [100.0],
            "ra_deg": [202.0],
            "dec_deg": [42.0],
            "mag": [11.0],
            "rms_pct": [5.0],
            "is_variable_candidate": [True],
            "vsx_known_variable": [True],
            "vsx_match": [True],
            "vsx_name": ["BO CVn"],
            "vsx_type": ["EA"],
        }
    )

    import variability_detector

    monkeypatch.setattr(
        variability_detector,
        "load_field_flux_matrix",
        lambda *a, **k: (pd.DataFrame(), pd.DataFrame(), []),
    )
    monkeypatch.setattr(variability_detector, "compute_rms_variability", lambda *a, **k: rms_df)
    monkeypatch.setattr(
        variability_detector,
        "compute_vdi",
        lambda *a, **k: pd.DataFrame(
            columns=["catalog_id", "vdi_score", "vdi_z_score", "is_variable_candidate"]
        ),
    )
    monkeypatch.setattr(
        "photometry_core._edge_ok_from_masterstar_pipeline",
        lambda *a, **k: (pd.Series([True], index=rms_df.index), False),
    )

    class _Cfg:
        variability_sigma_threshold = 2.3
        variability_mag_limit = 14.5
        gaia_db_path = ""

        def to_dict(self):
            return {}

    ms = setup / "MASTERSTAR.fits"
    ms.write_bytes(b"")
    out = auto_export_variability_candidates_csv(
        masterstar_fits_path=ms,
        comparison_stars_csv=comp,
        per_frame_csv_dir=tmp_path / "frames",
        output_dir=phot,
        cfg=_Cfg(),
        platesolve_dir=setup,
    )
    assert out is not None
    exported = pd.read_csv(out, dtype={"catalog_id": str})
    assert exported.empty


def test_repair_skips_det_placeholders_and_summarizes(tmp_path: Path) -> None:
    from repair_catalog_ids import repair_csv_catalog_ids_from_gaia_db

    csv_path = tmp_path / "masterstars_full_match.csv"
    csv_path.write_text(
        "catalog_id,ra_deg,dec_deg,name\n"
        "DET_0001,180.0,45.0,det1\n"
        "DET_0042,180.001,45.001,det2\n"
        ",180.0,45.0,empty\n"
        "1234567890123456789,180.0,45.0,real\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "gaia.db"
    con = sqlite3.connect(str(db_path))
    con.execute("CREATE TABLE gaia_dr3 (source_id INTEGER PRIMARY KEY, ra REAL, dec REAL)")
    con.commit()
    con.close()

    logs: list[str] = []

    res = repair_csv_catalog_ids_from_gaia_db(
        csv_path=csv_path,
        gaia_db_path=db_path,
        skip_unmatched_placeholders=True,
        log_fn=logs.append,
    )
    assert res["kept_placeholder"] == 3
    assert res["checked"] == 1
    assert any("REPAIR summary:" in line for line in logs)
    assert not any("boxe +/-0.001" in line for line in logs)


def test_hrd_enrich_tap_retry_then_success(tmp_path: Path) -> None:
    from hrd_enrich import enrich_candidates

    cand = pd.DataFrame({"catalog_id": ["1234567890123456789"]})
    cache = tmp_path / "hrd_enrich.json"
    payload = {
        "1234567890123456789": {
            "teff_gspphot": 6000.0,
            "logg_gspphot": 4.0,
            "classprob_dsc_combmod_whitedwarf": 0.01,
            "classprob_dsc_combmod_binarystar": 0.02,
            "spectraltype_esphs": "G2V",
            "enrich_source": "gaia_tap",
        }
    }
    calls = {"n": 0}

    def _flaky(*_a, **_k):
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError("simulated TAP timeout")
        return payload

    with patch("hrd_enrich._fetch_gaia_tap", side_effect=_flaky):
        with patch("hrd_enrich.time.sleep"):
            out = enrich_candidates(cand, cache, enabled=True, simbad_enabled=False, timeout_s=5.0)

    assert calls["n"] == 3
    assert out.loc[0, "teff_gspphot"] == 6000.0
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert "enrich_skip_reason" not in summary
    assert summary.get("enrich_attempts") == 3


def test_hrd_enrich_tap_fail_records_skip_reason(tmp_path: Path) -> None:
    from hrd_enrich import enrich_candidates

    cand = pd.DataFrame({"catalog_id": ["1234567890123456789"]})
    cache = tmp_path / "hrd_enrich.json"

    with patch("hrd_enrich._fetch_gaia_tap", side_effect=RuntimeError("network down")):
        with patch("hrd_enrich.time.sleep"):
            out = enrich_candidates(cand, cache, enabled=True, simbad_enabled=False, timeout_s=5.0)

    assert str(out.loc[0, "enrich_source"]) == "n/a"
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert "Gaia TAP" in str(summary.get("enrich_skip_reason", ""))
    assert summary.get("enrich_attempts") == 3


def test_excluded_targets_sidecar_on_no_match(tmp_path: Path, monkeypatch) -> None:
    import photometry_core as pc

    vt = pd.DataFrame(
        {
            "name": ["BO CVn"],
            "vsx_name": ["BO CVn"],
            "vsx_type": ["EA"],
            "ra_deg": [202.0],
            "dec_deg": [42.0],
            "mag": [10.5],
            "catalog_id": ["1400549806859236864"],
            "x": [500.0],
            "y": [500.0],
        }
    )
    ms = pd.DataFrame(
        {
            "name": ["DET_0001"],
            "ra_deg": [200.0],
            "dec_deg": [40.0],
            "x": [10.0],
            "y": [10.0],
            "catalog_id": ["9999999999999999999"],
            "mag": [12.0],
            "flux": [1000.0],
        }
    )
    ms_csv = tmp_path / "masterstars_full_match.csv"
    ms.to_csv(ms_csv, index=False)
    vt_csv = tmp_path / "variable_targets.csv"
    vt.to_csv(vt_csv, index=False)

    monkeypatch.setattr(
        "photometry_core._enrich_active_targets_bp_rp",
        lambda df, **k: df,
    )
    monkeypatch.setattr(
        "photometry_core._ensure_active_target_display_names",
        lambda df: df,
    )

    active = pc.select_active_targets(
        variable_targets_csv=vt_csv,
        masterstars_csv=ms_csv,
        frame_w_px=2000,
        frame_h_px=2000,
    )
    assert active.empty
    excluded = pc.LAST_EXCLUDED_TARGETS
    assert not excluded.empty
    assert excluded.iloc[0]["reason"] == "no_dao_gaia_match"
    assert excluded.iloc[0]["vsx_name"] == "BO CVn"

    sidecar = tmp_path / "excluded_targets.csv"
    excluded.to_csv(sidecar, index=False)
    loaded = pd.read_csv(sidecar)
    assert len(loaded) == 1
    assert loaded.iloc[0]["reason"] == "no_dao_gaia_match"


def test_infolog_formatter_uses_utc() -> None:
    import logging
    import time

    from infolog import ensure_infolog_logging

    ensure_infolog_logging()
    lg = logging.getLogger("pipeline")
    handler = next(h for h in lg.handlers if getattr(h, "formatter", None) is not None)
    assert handler.formatter.converter is time.gmtime
