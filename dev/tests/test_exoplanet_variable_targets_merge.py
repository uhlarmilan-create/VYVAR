"""VSX + exoplanet variable_targets merge and dedup."""
from __future__ import annotations

import pandas as pd

from pipeline import _merge_vsx_exoplanet_variable_targets


def test_merge_same_catalog_id_keeps_vsx_and_adds_exo_labels() -> None:
    shared_cid = "1625373404725030528"
    vsx = pd.DataFrame(
        [
            {
                "name": "VSX Star",
                "catalog_id": shared_cid,
                "catalog": "VSX",
                "vsx_name": "VSX Star",
                "vsx_type": "RR",
                "vsx_period": 0.5,
                "ra_deg": 248.43,
                "dec_deg": 61.72,
            }
        ]
    )
    exo = pd.DataFrame(
        [
            {
                "name": "TIC 198213332",
                "catalog_id": shared_cid,
                "catalog": "EXOPLANET",
                "vsx_name": "",
                "vsx_type": "",
                "exo_host_obj_id": "TOI-1131.01",
                "exo_host_name": "TIC 198213332",
                "exo_cat_source": "TOI",
                "exo_disposition": "PC",
                "exo_match_sep_arcsec": 0.01,
                "target_origin": "EXOPLANET",
            }
        ]
    )
    merged = _merge_vsx_exoplanet_variable_targets(vsx, exo)
    assert len(merged) == 1
    row = merged.iloc[0]
    assert str(row["vsx_name"]) == "VSX Star"
    assert str(row["vsx_type"]) == "RR"
    assert str(row["exo_host_obj_id"]) == "TOI-1131.01"
    assert str(row["exo_disposition"]) == "PC"


def test_merge_exo_only_appended() -> None:
    vsx = pd.DataFrame(
        [
            {
                "name": "VSX Only",
                "catalog_id": "1111111111111111111",
                "catalog": "VSX",
                "vsx_name": "VSX Only",
                "vsx_type": "DSCT",
            }
        ]
    )
    exo = pd.DataFrame(
        [
            {
                "name": "exo host",
                "catalog_id": "2222222222222222222",
                "catalog": "EXOPLANET",
                "exo_host_obj_id": "K2-100.01",
                "exo_disposition": "CP",
                "target_origin": "EXOPLANET",
            }
        ]
    )
    merged = _merge_vsx_exoplanet_variable_targets(vsx, exo)
    assert len(merged) == 2
    cids = set(merged["catalog_id"].astype(str))
    assert "1111111111111111111" in cids
    assert "2222222222222222222" in cids


def test_merge_does_not_dedup_float_corrupted_id_against_true_gaia() -> None:
    """Float-rounded ...0400 must not collapse a distinct true ...0528 exo promotion."""
    true_cid = "1625373404725030528"
    wrong_cid = "1625373404725030400"
    vsx = pd.DataFrame(
        [
            {
                "name": "VSX Star",
                "catalog_id": wrong_cid,
                "catalog": "VSX",
                "vsx_name": "VSX Star",
                "vsx_type": "RR",
            }
        ]
    )
    exo = pd.DataFrame(
        [
            {
                "name": "TIC 198213332",
                "catalog_id": true_cid,
                "catalog": "EXOPLANET",
                "exo_host_obj_id": "TOI-1131.01",
                "exo_disposition": "PC",
                "target_origin": "EXOPLANET",
            }
        ]
    )
    merged = _merge_vsx_exoplanet_variable_targets(vsx, exo)
    assert len(merged) == 2
    cids = set(merged["catalog_id"].astype(str))
    assert true_cid in cids
    assert wrong_cid in cids
