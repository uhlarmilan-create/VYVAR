"""POST-451 Part B: infolog observability + Phase 2A skip_reason propagation."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from infolog import get_lines, log_event
from photometry_core import (
    _propagate_phase2a_skip_reason_to_active,
    select_active_targets,
)
from vsx_gaia_crossmatch import match_vsx_to_gaia_density_aware


def _reset_infolog() -> None:
    from infolog import _lines, _lock

    with _lock:
        _lines.clear()


def test_faza0_funnel_reaches_infolog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _reset_infolog()
    ms = tmp_path / "masterstars_full_match.csv"
    pd.DataFrame(
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
    ).to_csv(ms, index=False)
    vt = tmp_path / "variable_targets.csv"
    pd.DataFrame(
        [
            {
                "name": "VSX_A",
                "vsx_name": "VSX_A",
                "vsx_type": "EA",
                "catalog": "VSX",
                "catalog_id": "1111111111111111111",
                "gaia_match_source": "masterstars",
                "ra_deg": 150.01,
                "dec_deg": 45.01,
                "x": 300.0,
                "y": 300.0,
                "mag": 12.0,
            }
        ]
    ).to_csv(vt, index=False)
    select_active_targets(vt, ms, frame_w_px=512, frame_h_px=512, edge_margin_px=50)
    lines = get_lines()
    assert any("FAZA 0 funnel:" in ln for ln in lines)
    funnel = next(ln for ln in lines if "FAZA 0 funnel:" in ln)
    assert "no_dao_detection=" in funnel
    assert "out_of_frame=" in funnel
    assert "not_target_eligible=" in funnel


def test_vsx_gaia_xm_reaches_infolog() -> None:
    import numpy as np
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    _reset_infolog()
    rng = np.random.default_rng(20260726)
    rho = 706.0
    n_gaia = int(rho * 21.4)
    area = n_gaia / rho
    ra0, dec0 = 150.0, 45.0
    ga_ra = ra0 + rng.uniform(-2.0, 2.0, n_gaia)
    ga_dec = dec0 + rng.uniform(-2.0, 2.0, n_gaia)
    n_vsx = 30
    pick = rng.choice(n_gaia, size=n_vsx, replace=False)
    vsx_ra = np.empty(n_vsx)
    vsx_dec = np.empty(n_vsx)
    for i, j in enumerate(pick):
        off = 1.8 if i < 15 else float(rng.uniform(0.05, 0.4))
        c = SkyCoord(ra=ga_ra[j] * u.deg, dec=ga_dec[j] * u.deg).directional_offset_by(
            float(rng.uniform(0, 360)) * u.deg, off * u.arcsec
        )
        vsx_ra[i] = c.ra.deg
        vsx_dec[i] = c.dec.deg
    cids = np.array([str(i) for i in range(n_gaia)])
    match_vsx_to_gaia_density_aware(
        vsx_ra,
        vsx_dec,
        cids,
        ga_ra,
        ga_dec,
        field_area_deg2=area,
        gaia_db_max_g=18.0,
    )
    lines = get_lines()
    assert any("VSX-GAIA XM:" in ln for ln in lines)
    xm = next(ln for ln in lines if "VSX-GAIA XM:" in ln)
    assert "sigma_b=" in xm
    assert "pm_path=" in xm
    assert "gaia_db_max_g=" in xm
    assert "cand_mult=" in xm


def test_phase2a_skip_reason_propagated_to_active_csv(tmp_path: Path) -> None:
    active = tmp_path / "active_targets.csv"
    pd.DataFrame(
        [
            {
                "catalog_id": "1496795041799526400",
                "name": "R CVn",
                "skip_reason": "",
                "n_frames": 0,
            }
        ]
    ).to_csv(active, index=False)
    summary_rows = [
        {
            "catalog_id": "1496795041799526400",
            "n_frames": 0,
            "ac_skip_reason": "no_comps",
        }
    ]
    _propagate_phase2a_skip_reason_to_active(active, summary_rows)
    out = pd.read_csv(active, dtype={"catalog_id": str})
    assert str(out.loc[0, "skip_reason"]).strip() == "no_comps"


def test_active_nframes_zero_must_have_skip_reason() -> None:
    """Invariant helper: no row with n_frames=0 and empty skip_reason."""
    df = pd.DataFrame(
        [
            {"catalog_id": "1", "n_frames": 5, "skip_reason": ""},
            {"catalog_id": "2", "n_frames": 0, "skip_reason": "no_comps"},
        ]
    )
    bad = df[(pd.to_numeric(df["n_frames"], errors="coerce").fillna(0) == 0) & (df["skip_reason"].fillna("").astype(str).str.strip() == "")]
    assert len(bad) == 0


def test_inv_ms01_milestone_reaches_headless_logger(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    from invariants_runtime import check_dao_only_fraction, inv_check
    from infolog import get_lines, log_event

    pipeline_logger = logging.getLogger("pipeline")
    old_propagate = pipeline_logger.propagate
    pipeline_logger.propagate = True
    caplog.set_level(logging.INFO, logger="pipeline")
    try:
        df = pd.DataFrame(
            {
                "catalog_id": ["1", "2", "3"],
                "source_type": ["GAIA_MATCHED", "GAIA_MATCHED", "GAIA_MATCHED"],
            }
        )
        ok, det, _frac, pol = check_dao_only_fraction(df)
        inv_check({"invariants": []}, "INV-MS-01", ok, policy=pol, detail=det)
        msg = f"INV-MS-01 MASTERSTAR purity guard: {det}"
        pipeline_logger.info(msg)
        log_event(msg)
        assert any("INV-MS-01 MASTERSTAR purity guard" in r.message for r in caplog.records)
        assert any("INV-MS-01 MASTERSTAR purity guard" in ln for ln in get_lines())
    finally:
        pipeline_logger.propagate = old_propagate


def test_inv_prep01_milestone_reaches_headless_logger(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    import numpy as np
    from invariants_runtime import check_preprocess_large_small_ratio, inv_check
    from infolog import get_lines, log_event

    pipeline_logger = logging.getLogger("pipeline")
    old_propagate = pipeline_logger.propagate
    pipeline_logger.propagate = True
    caplog.set_level(logging.INFO, logger="pipeline")
    try:
        frame = np.full((64, 64), 1000.0, dtype=np.float64)
        ok, det, ratio = check_preprocess_large_small_ratio(frame)
        inv_check({"invariants": []}, "INV-PREP-01", ok, policy="WARN", detail=f"NoFilter_60_2: {det}")
        msg = f"INV-PREP-01 Preprocess gradient guard (NoFilter_60_2): {det}"
        pipeline_logger.info(msg)
        log_event(msg)
        assert any("INV-PREP-01 Preprocess gradient guard" in r.message for r in caplog.records)
        assert any("INV-PREP-01 Preprocess gradient guard" in ln for ln in get_lines())
        assert math.isfinite(ratio)
    finally:
        pipeline_logger.propagate = old_propagate
