#!/usr/bin/env python3
"""Unit tests for WIDE-ERR-03 gain photon transfer + sigma_sys log."""
from __future__ import annotations

from pathlib import Path

import pytest

from gain_photon_transfer import PhotonTransferGain, resolve_photometric_gain


def test_authority_prefers_g_pt():
    pt = PhotonTransferGain(
        g_pt=0.637,
        g_pt_ci_lo=0.44,
        g_pt_ci_hi=1.09,
        n_frames=134,
        aperture_r_px=4.0,
        slope=1.57,
        intercept=0.0,
        ok=True,
    )
    auth = resolve_photometric_gain(g_pt_result=pt, g_db_native=3.17, container_scale=4.0)
    assert auth.source == "g_pt"
    assert abs(auth.value_e_per_adu_container - 0.637) < 1e-9


def test_authority_falls_back_to_db_div_scale():
    pt = PhotonTransferGain(
        g_pt=float("nan"),
        g_pt_ci_lo=float("nan"),
        g_pt_ci_hi=float("nan"),
        n_frames=0,
        aperture_r_px=4.0,
        slope=float("nan"),
        intercept=float("nan"),
        ok=False,
    )
    auth = resolve_photometric_gain(g_pt_result=pt, g_db_native=3.17, container_scale=4.0)
    assert auth.source == "db_div_container_scale"
    assert abs(auth.value_e_per_adu_container - 3.17 / 4.0) < 1e-9


def test_sigma_sys_explicit_zero_for_equipment_1(caplog):
    class Cfg:
        database_path = ""  # no EQUIPMENTS row -> unset

    import logging

    import sigma_floor_core as sfc

    # Once-per-process log; earlier suite tests may have already consumed the key.
    sfc._LOGGED_UNFLOORED.discard("1:default0")
    with caplog.at_level(logging.INFO):
        v = sfc.resolve_sigma_sys_mag(1, Cfg(), rig_label="test")
    assert v == 0.0
    assert any("explicit default 0.0" in r.message for r in caplog.records)


@pytest.mark.skipif(
    not Path("Archive/Drafts/draft_000516/detrended_aligned/lights/NoFilter_60_2").is_dir(),
    reason="draft 516 proc CSVs not present",
)
def test_s2d_photon_transfer_on_draft_516():
    from gain_photon_transfer import estimate_photon_transfer_gain_from_proc_dir

    proc = Path("Archive/Drafts/draft_000516/detrended_aligned/lights/NoFilter_60_2")
    pt = estimate_photon_transfer_gain_from_proc_dir(proc, aperture_r_px=3.999)
    assert pt.ok
    assert 0.44 <= pt.g_pt <= 1.09
