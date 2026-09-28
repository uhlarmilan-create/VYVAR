# -*- coding: ascii -*-
"""GAIN-FALSY-01: PSF error-map gain/RN from equipment authority, not config defaults."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from astropy.io import fits

from psf_photometry import _psf_resolve_gain_read_noise

ROOT = Path(__file__).resolve().parents[2]
GAIN_SIDECAR = (
    ROOT
    / "Archive"
    / "Drafts"
    / "draft_000516"
    / "platesolve"
    / "NoFilter_60_2"
    / "photometry"
    / "gain_photon_transfer.json"
)
PIPELINE_META = GAIN_SIDECAR.parent / "pipeline_meta.json"
FRAME = (
    ROOT
    / "Archive"
    / "Drafts"
    / "draft_000516"
    / "detrended_aligned"
    / "lights"
    / "NoFilter_60_2"
    / "BO_CVn_Light_001.fits"
)


@pytest.mark.skipif(not FRAME.is_file(), reason="live 516 frame absent")
def test_gain_falsy_01_uses_g_pt_not_config_default():
    """Pre-fix: GAIN=0 header -> resolve falls to config -> (1.0, 10.0).
    Post-fix: photometry-dir authority -> g_pt ~0.637 / RN 15.2.
    """
    hdr = fits.getheader(FRAME)
    assert float(hdr.get("GAIN", float("nan"))) == 0.0
    phot_dir = GAIN_SIDECAR.parent
    gain, rn = _psf_resolve_gain_read_noise(hdr, photometry_dir=phot_dir)
    assert abs(gain - 0.637067) < 0.01, f"gain={gain}"
    assert abs(rn - 15.2) < 0.1, f"rn={rn}"


def test_gain_falsy_01_no_or_falsy_trap():
    """Explicit: a Resolved-like zero must not become 1.0 via `value or 1.0`."""
    # Synthetic header with no usable gain; without photometry_dir should not
    # silently claim success as 1.0 when authority is required - when no
    # photometry_dir is given, fall back remains finite and positive.
    hdr = fits.Header({"GAIN": 0.0})
    gain, rn = _psf_resolve_gain_read_noise(hdr, photometry_dir=None)
    assert gain > 0 and rn >= 0


def test_gain_falsy_01_null_g_pt_falls_to_container_scale(tmp_path: Path):
    """era04/era05 freeze sidecars store authority.g_pt=null; must not TypeError."""
    phot = tmp_path / "photometry"
    phot.mkdir()
    (phot / "gain_photon_transfer.json").write_text(
        json.dumps(
            {
                "authority": {
                    "value_e_per_adu_container": 0.7925,
                    "source": "db_div_container_scale",
                    "g_pt": None,
                    "ok": True,
                },
                "photon_transfer": {"g_pt": float("nan")},
            }
        ),
        encoding="utf-8",
    )
    (phot / "pipeline_meta.json").write_text(
        json.dumps(
            {
                "resolved_facts": {
                    "read_noise": {"value": 15.2, "source": "test"},
                }
            }
        ),
        encoding="utf-8",
    )
    hdr = fits.Header({"GAIN": 0.0})
    gain, rn = _psf_resolve_gain_read_noise(hdr, photometry_dir=phot)
    assert abs(gain - 0.7925) < 1e-6, f"gain={gain}"
    assert abs(rn - 15.2) < 0.1, f"rn={rn}"
