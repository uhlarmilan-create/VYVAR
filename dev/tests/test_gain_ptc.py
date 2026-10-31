"""CAMERA-PROFILE-2026-10: photon-transfer gain / read noise on synthetic raw flats and darks."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from gain_ptc import measure_from_files, native_gain

SHAPE = (400, 600)


def _write(path: Path, data: np.ndarray, exptime: float, kind: str, gain_setting: int = 0, binning: int = 1) -> Path:
    h = fits.Header()
    h["IMAGETYP"] = kind
    h["EXPTIME"] = exptime
    h["GAIN"] = (gain_setting, "Gain")
    h["XBINNING"] = binning
    fits.writeto(path, data.astype(np.uint16), h, overwrite=True)
    return path


def _night(tmp_path: Path, *, g_native: float, rn_e: float, adc_bits: int, levels_e: list[float], seed: int = 1):
    rng = np.random.default_rng(seed)
    scale = 2 ** (16 - adc_bits)
    yy, xx = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
    vignet = 1.0 - 0.25 * (((xx - 300) / 300) ** 2 + ((yy - 200) / 200) ** 2)
    prnu = 1.0 + 0.01 * rng.standard_normal(SHAPE)
    pattern = vignet * prnu
    bias_native = 120.0

    def frame(mean_e: float) -> np.ndarray:
        e = rng.poisson(np.clip(mean_e * pattern, 0, None)) + rng.normal(0, rn_e, SHAPE)
        native = np.round(e / g_native + bias_native)
        return np.clip(native, 0, 2 ** adc_bits - 1) * scale

    flats, darks = [], []
    for i, lev in enumerate(levels_e):
        exp = 0.5 * (i + 1)
        for k in range(2):
            flats.append(_write(tmp_path / f"flat_{i}_{k}.fits", frame(lev), exp, "Flat Field"))
        for k in range(2):
            darks.append(_write(tmp_path / f"dark_{i}_{k}.fits", frame(0.0), exp, "Dark Frame"))
    return flats, darks


def test_ptc_recovers_gain_and_read_noise_16bit(tmp_path: Path) -> None:
    flats, darks = _night(tmp_path, g_native=0.78, rn_e=3.5, adc_bits=16, levels_e=[2e3, 6e3, 12e3, 20e3, 30e3])
    r = measure_from_files(flats, darks, saturation_adu=65535.0)
    assert r.ok and r.n_levels == 5
    assert r.gain == pytest.approx(0.78, rel=0.03)
    assert r.read_noise_e == pytest.approx(3.5, rel=0.08)
    assert r.r2 > 0.99


def test_ptc_14bit_camera_in_16bit_files(tmp_path: Path) -> None:
    flats, darks = _night(tmp_path, g_native=3.17, rn_e=7.6, adc_bits=14, levels_e=[5e3, 1.5e4, 3e4])
    r = measure_from_files(flats, darks, saturation_adu=65535.0)
    assert r.ok
    assert r.gain == pytest.approx(3.17 / 4, rel=0.03)  # gain of the files (container ADU)
    assert native_gain(r.gain, 14) == pytest.approx(3.17, rel=0.03)
    assert r.read_noise_e == pytest.approx(7.6, rel=0.08)


def test_ptc_rejects_saturated_and_mixed_settings(tmp_path: Path) -> None:
    flats, darks = _night(tmp_path, g_native=0.78, rn_e=3.5, adc_bits=16, levels_e=[5e3, 1e4, 4.5e4])
    r = measure_from_files(flats, darks, saturation_adu=65535.0)
    assert r.ok and r.n_levels == 2
    assert any("saturation" in p.note for p in r.points if not p.used)
    other = _write(tmp_path / "flat_x.fits", np.full(SHAPE, 1000), 0.5, "Flat Field", gain_setting=100)
    r = measure_from_files(flats + [other], darks, saturation_adu=65535.0)
    assert not r.ok and "gain setting" in r.messages[0]


def test_ptc_needs_two_flats(tmp_path: Path) -> None:
    flats, darks = _night(tmp_path, g_native=0.78, rn_e=3.5, adc_bits=16, levels_e=[5e3])
    r = measure_from_files(flats[:1], darks, saturation_adu=65535.0)
    assert not r.ok and "two raw flats" in r.messages[0]
