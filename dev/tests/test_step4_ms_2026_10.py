# -*- coding: ascii -*-
"""Code review 2, step 4 (Milan 2026-10-29): MASTERSTAR Gaia accounting.

G04-05: catalogue-membership rows are never detections, never lock their own Gaia star and the
census counts DETECTED_* only from DAO detections (FORCED_SEED mechanism removed).
G04-08: detection counts / raw completeness count detections only, against on-chip Gaia to the
matching depth.
G04-25: the MASTERSTAR build keeps each row's own vy_dao_pass / ambiguous_owner (no CSV
round-trip remap by catalog_id).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src_py"))

import masterstar_gaia_accounting as mga  # noqa: E402
from dao_reconcile import gaia_detection_counts  # noqa: E402

W, H = 200, 200


def _gaia() -> pd.DataFrame:
    # 1001/1002 detected (pass 1 / pass 2), 1003/1004 membership only, 1005 too faint.
    return pd.DataFrame(
        {
            "catalog_id": ["1001", "1002", "1003", "1004", "1005"],
            "x_gaia": [50.0, 100.0, 150.0, 60.0, 120.0],
            "y_gaia": [50.0, 100.0, 150.0, 140.0, 40.0],
            "g_mag": [12.0, 14.0, 13.0, 14.5, 18.0],
        }
    )


def _ms() -> pd.DataFrame:
    """Detections (two Gaia, two DAO_ONLY) + membership rows as the M1 expansion makes them."""
    return pd.DataFrame(
        {
            "x": [50.3, 100.2, 30.0, 170.0, 150.0, 60.0],
            "y": [50.1, 99.8, 170.0, 30.0, 150.0, 140.0],
            "catalog_id": ["1001", "1002", "", "", "1003", "1004"],
            "peak_dao": [500.0, 40.0, 30.0, 25.0, np.nan, np.nan],
            "vy_dao_pass": [1, 2, 1, 2, np.nan, np.nan],
            "vy_match_mode": ["", "", "", "", "catalog_membership", "catalog_membership"],
            "forced_photometry": [False] * 6,
            "ambiguous_owner": [False, True, False, True, np.nan, np.nan],
        }
    )


def _enrich(df: pd.DataFrame):
    return mga.enrich_masterstar_gaia_complete(
        df,
        gaia_on_chip=_gaia(),
        wpx=W,
        h=H,
        fwhm_px=3.5,
        target_depth_g=16.0,
        catalog_derived_membership=True,
    )


def test_census_detected_count_equals_dao_detection_rows() -> None:
    df = _ms()
    out, census, meta = _enrich(df)
    n_det_census = int(census["source_state"].astype(str).str.startswith("DETECTED").sum())
    vp = pd.to_numeric(df["vy_dao_pass"], errors="coerce")
    n_identified_dao = int((vp.isin([1, 2]) & df["catalog_id"].ne("")).sum())
    assert n_det_census == n_identified_dao == 2
    st = dict(zip(census["catalog_id"], census["source_state"]))
    assert st["1001"] == mga.SOURCE_DETECTED_P1
    assert st["1002"] == mga.SOURCE_DETECTED_P2
    assert st["1003"] == mga.SOURCE_CATALOG_ONLY
    assert st["1004"] == mga.SOURCE_CATALOG_ONLY
    assert st["1005"] == mga.SOURCE_TOO_FAINT
    assert meta["census_by_state"].get(mga.SOURCE_DETECTED_P1, 0) == 1
    assert "n_forced_seed" not in meta and "n_seed_rejected" not in meta
    assert len(out) == len(df)  # the accounting never adds rows


def test_membership_row_never_detected_and_never_locks() -> None:
    df = _ms()
    meta: dict = {"n_leftover_promotions": 0}
    work = df.copy()
    gaia_owner, _gdf, _ilo, det_pass, _n_g = mga._acc_identity_locks(
        catalog_derived_membership=True,
        fwhm_px=3.5,
        gaia_on_chip=_gaia(),
        identity_lock_only=False,
        locked_pairs=None,
        meta=meta,
        out=work,
        tolerance_overrides=None,
    )
    assert list(det_pass) == [1, 2, 1, 2, 0, 0]
    # Gaia 1001 -> row 0, 1002 -> row 1; membership stars 1003 / 1004 have no owner.
    assert list(gaia_owner) == [0, 1, -1, -1, -1]
    out, _census, _meta = _enrich(df)
    st = out["source_state"].astype(str).tolist()
    assert st[4] == st[5] == mga.SOURCE_CATALOG_MEMBERSHIP
    assert st[2] == st[3] == "DAO_ONLY"
    assert out["vy_match_mode"].tolist()[4:] == ["catalog_membership", "catalog_membership"]


def test_membership_row_near_detection_is_blended_not_detected() -> None:
    gaia = _gaia()
    gaia.loc[3, ["x_gaia", "y_gaia"]] = [52.0, 51.0]  # 1004 within one FWHM of detected 1001
    df = _ms()
    df.loc[5, ["x", "y"]] = [52.0, 51.0]
    _out, census, _meta = mga.enrich_masterstar_gaia_complete(
        df, gaia_on_chip=gaia, wpx=W, h=H, fwhm_px=3.5, target_depth_g=16.0, catalog_derived_membership=True
    )
    st = dict(zip(census["catalog_id"], census["source_state"]))
    assert st["1001"] == mga.SOURCE_DETECTED_P1
    assert st["1004"] == mga.SOURCE_BLENDED


def test_removed_states_and_seed_api_are_gone() -> None:
    assert "FORCED_SEED" not in mga.ALL_SOURCE_STATES
    assert "SEED_REJECTED" not in mga.ALL_SOURCE_STATES
    for name in ("forced_seed_measure_at_position", "forced_seed_accept", "ForcedSeedAcceptParams"):
        assert not hasattr(mga, name)
    import pipeline_constants

    assert not any(k.startswith("MS_FORCED_SEED") for k in dir(pipeline_constants))


def test_detection_counts_exclude_membership_and_stay_below_100_pct() -> None:
    out, _census, _meta = _enrich(_ms())
    c = gaia_detection_counts(out, _gaia(), depth_g=16.0)
    vp = pd.to_numeric(out["vy_dao_pass"], errors="coerce")
    n_dao = int(vp.isin([1, 2]).sum())
    assert c["n_detection_rows"] == n_dao == 4
    assert c["n_gaia_detection_rows"] == 2
    assert c["n_gaia_reference"] == 4  # on-chip to G <= 16 (1005 is fainter)
    assert c["n_gaia_detected"] == 2 <= n_dao
    assert c["n_gaia_undetected"] == 2
    assert c["completeness_raw_pct"] == 50.0
    assert 0.0 <= c["completeness_raw_pct"] <= 100.0
    # without a reference only the row counts
    c0 = gaia_detection_counts(out, None, depth_g=None)
    assert c0["completeness_raw_pct"] is None and c0["n_gaia_detection_rows"] == 2


def test_build_keeps_own_vy_dao_pass_of_dao_only_rows(tmp_path, monkeypatch) -> None:
    import catalog_match
    import masterstar_build
    import rig_photometry

    df_det = pd.DataFrame(
        {
            "name": ["1001", "DET_00002", "DET_00003", "DET_00004"],
            "catalog_id": ["1001", "", "", ""],
            "x": [10.0, 20.0, 30.0, 40.0],
            "y": [10.0, 20.0, 30.0, 40.0],
            "mag": [12.0, np.nan, np.nan, np.nan],
            "peak_dao": [100.0, 20.0, 30.0, 25.0],
            "vy_dao_pass": np.asarray([1, 2, 1, 2], dtype=np.int16),
            "ambiguous_owner": np.asarray([False, True, False, False], dtype=bool),
        }
    )
    det_meta = {"n_detected": 4, "n_matched": 1, "catalog_rows": 50, "frame_depth": {}}

    class _Gain:
        gain = 1.0

    monkeypatch.setattr(catalog_match, "detect_stars_and_match_catalog", lambda *a, **k: (df_det.copy(), dict(det_meta)))
    monkeypatch.setattr(rig_photometry, "photometric_gain_read_noise", lambda *a, **k: _Gain())
    monkeypatch.setattr(masterstar_build, "_update_masterstar_obs_file_status", lambda **k: None)
    _rows, csv_path, _meta, df_final = masterstar_build._ms_detect_and_match(
        _cfg_ms=None,
        _dao_fwhm_bypass_hdr=False,
        _eq_ms=None,
        _full_db=tmp_path / "gaia.db",
        _ms_measured_sat=None,
        _selected_ref_path=None,
        dao_fwhm_px_for_ms=3.5,
        data=np.zeros((50, 50), dtype=np.float32),
        draft_id=None,
        equipment_saturate_adu=None,
        hdr={},
        masterstars_csv_basename=None,
        platesolve_dir=tmp_path,
    )
    assert pd.to_numeric(df_final["vy_dao_pass"]).tolist() == [1, 2, 1, 2]
    assert df_final["ambiguous_owner"].astype(bool).tolist() == [False, True, False, False]
    assert not Path(csv_path).exists()  # written once, at the end of the build
