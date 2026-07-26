"""VSX targets without masterstar (DAO+Gaia) match are excluded from active_targets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from photometry_core import select_active_targets


def _write_fixtures(tmp_path: Path) -> tuple[Path, Path]:
    vt_csv = tmp_path / "variable_targets.csv"
    ms_csv = tmp_path / "masterstars_full_match.csv"

    vt = pd.DataFrame(
        [
            {
                "name": "VSX_UNMATCHED",
                "vsx_name": "VSX_UNMATCHED",
                "vsx_type": "EA",
                "vsx_period": "",
                "priority": 1,
                "ra_deg": 150.0,
                "dec_deg": 45.0,
                "x": 256.0,
                "y": 256.0,
                "catalog_id": "9999999999999999999",
                "catalog": "VSX",
                "gaia_match_source": "masterstars",
                "mag": 12.5,
            },
            {
                "name": "VSX_MATCHED",
                "vsx_name": "VSX_MATCHED",
                "vsx_type": "DSCT",
                "vsx_period": "",
                "priority": 1,
                "ra_deg": 150.01,
                "dec_deg": 45.01,
                "x": 300.0,
                "y": 300.0,
                "catalog_id": "1111111111111111111",
                "catalog": "VSX",
                "gaia_match_source": "masterstars",
                "mag": 11.0,
            },
        ]
    )
    vt.to_csv(vt_csv, index=False)

    ms = pd.DataFrame(
        [
            {
                "name": "1111111111111111111",
                "catalog_id": "1111111111111111111",
                "ra_deg": 150.01,
                "dec_deg": 45.01,
                "x": 300.0,
                "y": 300.0,
                "mag": 11.0,
                "zone": "linear",
                "is_usable": True,
                "is_saturated": False,
                "is_noisy": False,
                "snr50_ok": True,
            }
        ]
    )
    ms.to_csv(ms_csv, index=False)
    return vt_csv, ms_csv


def test_select_active_targets_excludes_unmatched_vsx(tmp_path: Path) -> None:
    vt_csv, ms_csv = _write_fixtures(tmp_path)
    out = select_active_targets(
        vt_csv,
        ms_csv,
        frame_w_px=512,
        frame_h_px=512,
        edge_margin_px=50,
    )
    assert len(out) == 1
    assert str(out.iloc[0]["catalog_id"]) == "1111111111111111111"
    assert "9999999999999999999" not in out["catalog_id"].astype(str).tolist()
