"""Saturated targets still skip Phase 2A photometry (skip_photometry unchanged)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd

from config import AppConfig
from photometry_core import (
    _normalize_gaia_id,
    _phase2a_process_one_target,
    _Phase2AState,
)


def _cfg_phase2a_minimal() -> AppConfig:
    cfg = AppConfig()
    cfg.psf_photometry_enabled = False
    cfg.gs11_dilution_enabled = False
    cfg.aperture_correction_enabled = False
    cfg.democratic_detrend_enabled = False
    cfg.savgol_detrend_enabled = False
    cfg.temporal_binning_enabled = False
    return cfg


def _comp_table(target_cid: str, comp_cids: list[str]) -> pd.DataFrame:
    rows = []
    for i, cid in enumerate(comp_cids):
        rows.append(
            {
                "catalog_id": cid,
                "mag": 12.0 + i * 0.05,
                "comp_tier": 1 if i == 0 else 2,
                "comp_rms": 0.01,
                "x": 50.0 + i * 10.0,
                "y": 50.0 + i * 10.0,
            }
        )
    return pd.DataFrame(rows)


def _build_state(tmp_path: Path, target_cid: str, comp_cids: list[str]) -> _Phase2AState:
    cfg = _cfg_phase2a_minimal()
    comps = _comp_table(target_cid, comp_cids)
    comp_index = {target_cid: comps}

    star_xy: dict[str, tuple[float, float]] = {target_cid: (100.0, 100.0)}
    for i, cid in enumerate(comp_cids):
        star_xy[cid] = (50.0 + i * 10.0, 50.0 + i * 10.0)

    csv_path = tmp_path / "proc_frame1.csv"
    csv_path.write_text("stub", encoding="utf-8")
    apertures = {target_cid: 5.0}
    for cid in comp_cids:
        apertures[cid] = 5.0

    return _Phase2AState(
        at_df=pd.DataFrame(),
        comp_df=comps,
        _comp_index=comp_index,
        target_bp_rp_by_cid={},
        csv_files=[csv_path],
        n_frames=1,
        _phase2a_csv_cache={
            str(csv_path): pd.DataFrame({"jd_mid": [2459000.5], "bjd_tdb_mid": [2459000.5]}),
        },
        _phase2a_lookup_cache={},
        frame_time_lookup={
            "proc_frame1": {"jd": 2459000.5, "bjd": 2459000.5, "airmass": 1.2},
        },
        fwhm_px=3.0,
        apertures_px=apertures,
        star_xy=star_xy,
        chip_fw=512,
        chip_fh=512,
        _ms_header=None,
        _ms_data=None,
        _flux_matrix=pd.DataFrame(),
        _all_lc_ids_list=[],
        field_map_path=tmp_path / "field_map.png",
        obs_group="V",
        _gain_phot=1.0,
        _rn_phot=10.0,
        sat_limit_resolved=60000.0,
        _aligned_dir_2a=tmp_path / "aligned",
        _cfg=cfg,
        _nt=1,
        masterstars_df=pd.DataFrame(),
    )


def _target_row(target_cid: str, *, zone_flag: str) -> dict:
    return {
        "catalog_id": target_cid,
        "zone_flag": zone_flag,
        "skip_photometry": zone_flag == "saturated",
        "ra_deg": 100.0,
        "dec_deg": 20.0,
        "x": 100.0,
        "y": 100.0,
        "mag": 13.0,
        "vsx_name": "TEST_TARGET",
    }


TARGET_CID = "1234567890123456789"
COMP_CIDS = [
    "1111111111111111111",
    "2222222222222222222",
    "3333333333333333333",
]


def test_saturated_target_skips_with_zero_frames(tmp_path: Path) -> None:
    state = _build_state(tmp_path, TARGET_CID, COMP_CIDS)
    row = _target_row(TARGET_CID, zone_flag="saturated")
    lc_dir = tmp_path / "lc"
    out_dir = tmp_path / "out"
    lc_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    fits_stub = tmp_path / "masterstar.fits"
    fits_stub.write_bytes(b"")

    with patch("phase2a_target.read_flux_from_csv") as mock_read:
        summary, n_lc = _phase2a_process_one_target(
            row,
            ti=1,
            state=state,
            summary_rows=[],
            n_lc=0,
            lc_dir=lc_dir,
            output_dir=out_dir,
            progress_cb=None,
            masterstar_fits_path=fits_stub,
            annulus_inner_fwhm=2.0,
            annulus_outer_fwhm=3.0,
            outlier_sigma=3.0,
            stability_sigma=3.0,
            _apt_fw=1.5,
            _save_png=False,
            ac_sign_logged=[False],
        )
        mock_read.assert_not_called()

    assert len(summary) == 1
    assert summary[0]["n_frames"] == 0
    assert _normalize_gaia_id(summary[0]["catalog_id"]) == _normalize_gaia_id(TARGET_CID)
    assert n_lc == 0
