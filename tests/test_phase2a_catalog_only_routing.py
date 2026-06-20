"""G2-F001: catalog_only must route to forced-aperture, not early skip."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from config import AppConfig
from photometry_core import (
    _catalog_only_fixed_aperture_flux,
    _normalize_gaia_id,
    _phase2a_process_one_target,
    _Phase2AState,
    _target_row_is_catalog_only,
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


def _flux_frame_rows(
    cids: list[str],
    mag_vals: list[float],
    *,
    source_file: str = "proc_frame1.csv",
    bjd: float = 2459000.5,
) -> pd.DataFrame:
    rows: list[dict] = []
    for cid, mag in zip(cids, mag_vals, strict=True):
        rows.append(
            {
                "catalog_id": cid,
                "source_file": source_file,
                "mag_inst": mag,
                "err": 0.01,
                "bjd": bjd,
                "hjd": bjd,
                "jd": bjd,
                "x": 80.0,
                "y": 80.0,
                "aperture_r_px": 5.0,
                "airmass": 1.2,
                "flag": "ok",
                "is_flipped": False,
                "sky_annulus_r_out_px": 30.0,
            }
        )
    return pd.DataFrame(rows)


def _merge_inject_target_mag(df_frame: pd.DataFrame, **kwargs) -> pd.DataFrame:
    out = df_frame.copy()
    cid_n = _normalize_gaia_id(kwargs["target_catalog_id"])
    m = out["catalog_id"].astype(str).map(_normalize_gaia_id).eq(cid_n)
    if bool(m.any()):
        out.loc[m, "mag_inst"] = 13.0
        out.loc[m, "err"] = 0.01
        out.loc[m, "flag"] = "ok"
    return out


def _build_state(
    tmp_path: Path,
    target_cid: str,
    comp_cids: list[str],
    *,
    flux_matrix: pd.DataFrame | None = None,
) -> tuple[_Phase2AState, Path]:
    cfg = _cfg_phase2a_minimal()
    comps = _comp_table(target_cid, comp_cids)
    comp_df = comps.copy()
    comp_index = {target_cid: comps}

    star_xy: dict[str, tuple[float, float]] = {target_cid: (100.0, 100.0)}
    for i, cid in enumerate(comp_cids):
        star_xy[cid] = (50.0 + i * 10.0, 50.0 + i * 10.0)

    csv_path = tmp_path / "proc_frame1.csv"
    csv_path.write_text("stub", encoding="utf-8")
    apertures = {target_cid: 5.0}
    for cid in comp_cids:
        apertures[cid] = 5.0

    fm = flux_matrix if flux_matrix is not None else pd.DataFrame()

    state = _Phase2AState(
        at_df=pd.DataFrame(),
        comp_df=comp_df,
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
        _flux_matrix=fm,
        _all_lc_ids_list=[],
        field_map_path=tmp_path / "field_map.png",
        obs_group="V",
        _gain_phot=1.0,
        _rn_phot=10.0,
        sat_limit_resolved=60000.0,
        _aligned_dir_2a=tmp_path / "aligned",
        _cfg=cfg,
        _nt=1,
        _n_catalog_only=1,
        masterstars_df=pd.DataFrame(),
    )
    (tmp_path / "aligned").mkdir(parents=True, exist_ok=True)
    return state, csv_path


def _target_row(
    target_cid: str,
    *,
    zone_flag: str,
    skip_photometry: bool = False,
) -> dict:
    return {
        "catalog_id": target_cid,
        "zone_flag": zone_flag,
        "skip_photometry": skip_photometry,
        "ra_deg": 100.0,
        "dec_deg": 20.0,
        "x": 100.0,
        "y": 100.0,
        "mag": 13.0,
        "vsx_name": "TEST_TARGET",
    }


def _run_one(
    tmp_path: Path,
    target_row: dict,
    state: _Phase2AState,
    *,
    read_flux_side_effect=None,
    merge_side_effect=None,
) -> tuple[list, int]:
    lc_dir = tmp_path / "lc"
    out_dir = tmp_path / "out"
    lc_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    fits_stub = tmp_path / "masterstar.fits"
    fits_stub.write_bytes(b"")

    patches = []
    if read_flux_side_effect is not None:
        patches.append(
            patch(
                "photometry_core.read_flux_from_csv",
                side_effect=read_flux_side_effect,
            )
        )
    if merge_side_effect is not None:
        patches.append(
            patch(
                "photometry_core._catalog_only_merge_frame_flux",
                side_effect=merge_side_effect,
            )
        )

    for p in patches:
        p.start()

    try:
        return _phase2a_process_one_target(
            target_row,
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
    finally:
        for p in reversed(patches):
            p.stop()


TARGET_CID = "1234567890123456789"
COMP_CIDS = [
    "1111111111111111111",
    "2222222222222222222",
    "3333333333333333333",
]


def test_target_row_is_catalog_only_from_zone_flag() -> None:
    row = _target_row(TARGET_CID, zone_flag="catalog_only")
    assert _target_row_is_catalog_only(row)


def test_catalog_only_fixed_aperture_flux_finite_on_synthetic_stamp() -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(100.0, 5.0, (128, 128))
    data[60:68, 60:68] += 500.0
    flux, sky, peak = _catalog_only_fixed_aperture_flux(data, 64.0, 64.0, 4.0, 6.0, 12.0)
    assert np.isfinite(flux) and flux > 0
    assert np.isfinite(sky)
    assert np.isfinite(peak)


def test_saturated_target_skips_with_zero_frames(tmp_path: Path) -> None:
    state, _ = _build_state(tmp_path, TARGET_CID, COMP_CIDS)
    row = _target_row(TARGET_CID, zone_flag="saturated")
    summary, n_lc = _run_one(tmp_path, row, state)
    assert len(summary) == 1
    assert summary[0]["n_frames"] == 0
    assert summary[0]["lc_source"] == "dao_matched"
    assert n_lc == 0


def test_catalog_only_calls_forced_aperture_merge_and_produces_lc(tmp_path: Path) -> None:
    state, csv_path = _build_state(tmp_path, TARGET_CID, COMP_CIDS)
    row = _target_row(TARGET_CID, zone_flag="catalog_only")
    all_ids = [TARGET_CID] + COMP_CIDS
    comp_mags = [12.0, 12.05, 12.1]

    def _read_flux(_csv, ids, *args, **kwargs):
        ordered = [c for c in all_ids if c in ids]
        mags = [
            13.0 if c == TARGET_CID else comp_mags[COMP_CIDS.index(c)]
            for c in ordered
        ]
        return _flux_frame_rows(ordered, mags, source_file=csv_path.name)

    merge_calls: list = []

    def _merge(df_frame, **kwargs):
        merge_calls.append(kwargs.get("target_catalog_id"))
        return _merge_inject_target_mag(df_frame, **kwargs)

    summary, n_lc = _run_one(
        tmp_path,
        row,
        state,
        read_flux_side_effect=_read_flux,
        merge_side_effect=_merge,
    )
    assert merge_calls, "catalog_only must invoke _catalog_only_merge_frame_flux"
    assert len(summary) == 1
    assert summary[0]["n_frames"] > 0
    assert summary[0]["lc_source"] == "forced_aperture"
    assert n_lc == 1


def test_dao_matched_uses_flux_matrix_without_catalog_merge(tmp_path: Path) -> None:
    all_ids = [TARGET_CID] + COMP_CIDS
    mag_vals = [13.0, 12.0, 12.05, 12.1]
    flux_matrix = _flux_frame_rows(all_ids, mag_vals)
    state, _ = _build_state(tmp_path, TARGET_CID, COMP_CIDS, flux_matrix=flux_matrix)
    row = _target_row(TARGET_CID, zone_flag="dao_matched")

    with patch("photometry_core._catalog_only_merge_frame_flux") as mock_merge:
        summary, n_lc = _run_one(tmp_path, row, state)
        mock_merge.assert_not_called()

    assert len(summary) == 1
    assert summary[0]["n_frames"] > 0
    assert summary[0]["lc_source"] == "dao_matched"
    assert n_lc == 1
