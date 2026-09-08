# -*- coding: ascii -*-
"""RED-TARGET-T4-01: T4_FALLBACK rung (D-RED-TARGET-T4-01)."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from photometry_core import (
    _bprp_tier_ladder_for_selection,
    _select_comps_by_rms_then_color,
    select_comparison_stars_per_target,
)


def _cfg() -> SimpleNamespace:
    def limits() -> list[float]:
        return [0.15, 0.30, 0.55, 1.10]

    return SimpleNamespace(
        comp_tier_bprp_limits=limits,
        comp_max_delta_bprp=0.79,
        phase01_comparison_max_comp_rms=0.08,
        snr_cog_isolation_fwhm=3.0,
        comp_select_rms_floor=1e-6,
        comp_rms_loo_photon_k=5.0,
        gaia_db_path="",
        vsx_local_db_path="",
    )


def _row(cid: str, bp_rp: float, rms: float, dist: float = 0.2) -> dict:
    return {
        "catalog_id": cid,
        "bp_rp": bp_rp,
        "comp_rms": rms,
        "_dist_deg": dist,
        "_nn_dist_fwhm": 5.0,
        "snr_ap_pixscaled": 80.0,
    }


def test_a_cap_empties_finite_bprp_fallback_admits_min_delta() -> None:
    """(a) cap n < n_min + finite BP-RP -> min-|dBP-RP| from quality set."""
    tgt = 5.675
    rows = [
        _row("FAR_LOWRMS", 0.20, 0.010, 0.10),
        _row("NEAR_HIGHRMS", 3.40, 0.040, 0.40),
        _row("MID", 2.90, 0.025, 0.20),
        _row("NEAR2", 3.10, 0.030, 0.30),
    ]
    df = pd.DataFrame(rows)
    out = _select_comps_by_rms_then_color(
        df, target_bprp=tgt, n_comp_min=3, n_comp_max=8, max_delta_bprp=0.79, cfg=_cfg()
    )
    assert not out.empty
    assert bool(out.attrs.get("color_fallback")) is True
    ids = out["catalog_id"].astype(str).tolist()
    assert ids[0] == "NEAR_HIGHRMS"
    assert ids[1] == "NEAR2"
    assert ids[2] == "MID"
    assert float(out.attrs.get("max_delta_bprp_used")) >= 2.2


def test_b_cap_suffices_byte_identical_to_rms_order() -> None:
    """(b) cap has n >= n_min -> same RMS-then-color selection as pre-fallback."""
    rows = [
        _row("A", 1.00, 0.020, 0.50),
        _row("B", 1.02, 0.015, 0.80),
        _row("C", 1.04, 0.018, 0.10),
        _row("D", 1.10, 0.030, 0.20),
        _row("E", 1.08, 0.022, 0.15),
    ]
    df = pd.DataFrame(rows)
    out = _select_comps_by_rms_then_color(
        df, target_bprp=1.0, n_comp_min=3, n_comp_max=8, max_delta_bprp=0.79, cfg=_cfg()
    )
    assert bool(out.attrs.get("color_fallback")) is False
    ids = out["catalog_id"].astype(str).tolist()
    assert ids[0] == "B"
    assert set(ids) == {"A", "B", "C", "D", "E"}
    assert float(out.iloc[0]["comp_rms"]) <= float(out.iloc[1]["comp_rms"])


def test_c_nan_bprp_legacy_delta_zero_bypass() -> None:
    """(c) NaN target BP-RP: _delta_bprp_abs=0.0; no T4_FALLBACK."""
    rows = [_row(f"S{i}", 0.2 + i, 0.02 + 0.001 * i) for i in range(5)]
    df = pd.DataFrame(rows)
    out = _select_comps_by_rms_then_color(
        df, target_bprp=float("nan"), n_comp_min=3, n_comp_max=8, max_delta_bprp=0.79, cfg=_cfg()
    )
    assert not out.empty
    assert bool(out.attrs.get("color_fallback")) is False
    assert (pd.to_numeric(out["_delta_bprp_abs"], errors="coerce") == 0.0).all()


def test_d_fallback_still_below_nmin_returns_empty() -> None:
    """(d) quality set still < n_comp_min -> empty (no_comps)."""
    tgt = 5.675
    rows = [
        _row("ONLY1", 3.4, 0.02),
        _row("ONLY2", 3.2, 0.03),
    ]
    df = pd.DataFrame(rows)
    out = _select_comps_by_rms_then_color(
        df, target_bprp=tgt, n_comp_min=3, n_comp_max=8, max_delta_bprp=0.79, cfg=_cfg()
    )
    assert out.empty
    assert bool(out.attrs.get("color_fallback")) is True


def test_e_pinned_target_never_invokes_color_ladder() -> None:
    """(e) pin overlay early-return: _select_comps_by_rms_then_color not called."""
    target = pd.Series(
        {
            "catalog_id": "1496795041799526400",
            "name": "R CVn",
            "ra_deg": 207.24,
            "dec_deg": 39.54,
            "mag": 7.12,
            "phot_g_mean_mag": 7.12,
            "bp_rp": 5.675,
            "x": 100.0,
            "y": 100.0,
        }
    )
    ms = pd.DataFrame([target])
    sentinel = pd.DataFrame({"catalog_id": ["PINNED_COMP"]})

    def _fake_members(cid: str):
        assert str(cid) == "1496795041799526400"
        return [object()]

    def _fake_pin(*_a, **_k):
        return sentinel

    def _boom(*_a, **_k):
        raise AssertionError("color ladder must not run for pinned targets")

    with (
        patch("pinned_ensembles.get_pinned_members_for_target", _fake_members),
        patch("pinned_ensembles.select_pinned_comparison_stars_for_target", _fake_pin),
        patch("photometry_comp._select_comps_by_rms_then_color", _boom),
    ):
        out = select_comparison_stars_per_target(
            target,
            ms,
            per_frame_csv_paths=[],
            cfg=_cfg(),
        )
    assert list(out["catalog_id"].astype(str)) == ["PINNED_COMP"]


def test_ladder_last_rung_is_cap_not_t4_110() -> None:
    """T4 1.10 is not on the selection ladder; last rung is cap 0.79."""
    ladder = _bprp_tier_ladder_for_selection(_cfg(), 0.79)
    assert ladder == [0.15, 0.30, 0.55, 0.79]
