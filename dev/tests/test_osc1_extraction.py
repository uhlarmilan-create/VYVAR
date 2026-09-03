# -*- coding: ascii -*-
"""OSC-1: Bayer extraction, noise model, cross-check guards."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from osc_extract import (
    OSC_CHANNELS,
    average_bin_2d,
    bayer_planes_from_mosaic,
    channel_obs_group_folder,
    checkerboard_column_delta,
    derive_channel_planes,
    effective_gain_rn,
    extract_one_light_to_channels,
    validate_bayer_crosscheck,
    valid_bayer_pattern_4,
)
from photometry_core import _howell_variance_adu2


def _synthetic_mosaic(pattern: str, *, h: int = 8, w: int = 8) -> np.ndarray:
    """Build mosaic with known plane values for R/G1/G2/B."""
    pat = valid_bayer_pattern_4(pattern)
    assert pat is not None
    d = np.zeros((h, w), dtype=np.float32)
    plane_val = {"R": 10.0, "G1": 20.0, "G2": 22.0, "B": 40.0}
    g_n = 0
    coords = ((0, 0, pat[0]), (0, 1, pat[1]), (1, 0, pat[2]), (1, 1, pat[3]))
    for row, col, ch in coords:
        if ch == "G":
            key = "G1" if g_n == 0 else "G2"
            g_n += 1
        else:
            key = ch
        d[row::2, col::2] = plane_val[key]
    return d


@pytest.mark.parametrize("pattern", ["RGGB", "BGGR", "GBRG", "GRBG"])
def test_bayer_planes_per_mask(pattern: str) -> None:
    d = _synthetic_mosaic(pattern)
    r, g1, g2, b = bayer_planes_from_mosaic(d, pattern)
    assert float(np.mean(r)) == pytest.approx(10.0)
    assert float(np.mean(g1)) == pytest.approx(20.0)
    assert float(np.mean(g2)) == pytest.approx(22.0)
    assert float(np.mean(b)) == pytest.approx(40.0)
    ch = derive_channel_planes(r, g1, g2, b)
    assert float(np.mean(ch["R"])) == pytest.approx(10.0)
    assert float(np.mean(ch["G"])) == pytest.approx(21.0)
    assert float(np.mean(ch["B"])) == pytest.approx(40.0)
    assert float(np.mean(ch["oneRGGB"])) == pytest.approx((10 + 20 + 22 + 40) / 4.0)


def _low_signal_mosaic(*, h: int = 40, w: int = 40, base_adu: float = 1.5) -> np.ndarray:
    mosaic = np.full((h, w), base_adu, dtype=np.float32)
    mosaic[0::2, 0::2] += 0.4
    mosaic[0::2, 1::2] += 0.2
    mosaic[1::2, 0::2] += 0.2
    mosaic[1::2, 1::2] += 0.8
    return mosaic


def _monte_carlo_channel_variance(
    *,
    channel: str,
    osc_bin: int,
    gain: float,
    rn: float,
    mosaic: np.ndarray,
    pattern: str,
    n_trials: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    var_emp: list[float] = []
    electrons_scale = np.maximum(mosaic.astype(np.float64) * gain, 0.0)
    for _ in range(n_trials):
        electrons = rng.poisson(electrons_scale) + rng.normal(0.0, rn, mosaic.shape)
        adu = electrons / gain
        r, g1, g2, b = bayer_planes_from_mosaic(adu, pattern)
        ch = derive_channel_planes(r, g1, g2, b)[channel]
        ch_b = average_bin_2d(ch, osc_bin)
        var_emp.append(float(np.var(ch_b)))
    var_mean = float(np.mean(var_emp))
    r0, g1s, g2s, b0 = bayer_planes_from_mosaic(mosaic, pattern)
    ch_plane = derive_channel_planes(r0, g1s, g2s, b0)[channel]
    mean_adu = float(np.mean(average_bin_2d(ch_plane, osc_bin)))
    g_eff, rn_eff = effective_gain_rn(gain, rn, channel, osc_bin)
    var_pred = _howell_variance_adu2(mean_adu, 0.0, 1.0, gain=g_eff, read_noise=rn_eff)
    return var_mean, float(var_pred)


def test_effective_gain_rn_formula() -> None:
    g0, rn0 = 2.0, 4.0
    ge, rne = effective_gain_rn(g0, rn0, "G", 2)
    assert ge == pytest.approx(2.0 * 2 * 4)
    assert rne == pytest.approx(4.0 * math.sqrt(2 * 4))


def test_monte_carlo_noise_model() -> None:
    rng = np.random.default_rng(42)
    gain = 1.7
    rn = 3.2
    pattern = "RGGB"
    mosaic = np.full((40, 40), 500.0, dtype=np.float32)
    mosaic[0::2, 0::2] += 5.0
    mosaic[0::2, 1::2] += 2.0
    mosaic[1::2, 0::2] += 2.0
    mosaic[1::2, 1::2] += 8.0
    var_mean, var_pred = _monte_carlo_channel_variance(
        channel="G",
        osc_bin=2,
        gain=gain,
        rn=rn,
        mosaic=mosaic,
        pattern=pattern,
        n_trials=400,
        rng=rng,
    )
    assert var_mean == pytest.approx(var_pred, rel=0.25)


@pytest.mark.parametrize("channel", ["R", "G", "oneRGGB"])
@pytest.mark.parametrize("osc_bin", [1, 2])
def test_monte_carlo_noise_model_rn_dominated(channel: str, osc_bin: int) -> None:
    """RN-dominated regime catches rn_eff pairing errors (signal case cannot)."""
    rng = np.random.default_rng(100 + hash((channel, osc_bin)) % 10_000)
    gain = 1.8
    rn = 8.0
    mosaic = _low_signal_mosaic()
    var_mean, var_pred = _monte_carlo_channel_variance(
        channel=channel,
        osc_bin=osc_bin,
        gain=gain,
        rn=rn,
        mosaic=mosaic,
        pattern="RGGB",
        n_trials=1200,
        rng=rng,
    )
    assert var_pred > 0.0
    assert var_mean == pytest.approx(var_pred, rel=0.05)


def test_validate_bayer_crosscheck_fail_and_warn() -> None:
    v, m = validate_bayer_crosscheck(fits_bayerpat="RGGB", equipment_bayermask=None)
    assert v == "fail"
    assert m
    v2, _ = validate_bayer_crosscheck(fits_bayerpat="RGGB", equipment_bayermask="BGGR")
    assert v2 == "warn"
    v3, _ = validate_bayer_crosscheck(fits_bayerpat="RGGB", equipment_bayermask="RGGB")
    assert v3 == "ok"


def test_extract_writes_headers_and_reduces_checkerboard(tmp_path: Path) -> None:
    d = _synthetic_mosaic("RGGB", h=32, w=32)
    for c in range(d.shape[1]):
        d[:, c] += 50.0 if c % 2 else 0.0
    cb_before = checkerboard_column_delta(d)
    src = tmp_path / "light.fits"
    hdr = fits.Header()
    hdr["BAYERPAT"] = "RGGB"
    hdr["EXPTIME"] = 15.0
    hdr["DATE-OBS"] = "2026-07-22T20:00:00"
    fits.writeto(src, d, hdr, overwrite=True)
    out_dirs = {ch: tmp_path / channel_obs_group_folder("NoFilter_15_1", ch) for ch in OSC_CHANNELS}
    written = extract_one_light_to_channels(
        src,
        out_dirs=out_dirs,
        bayermask="RGGB",
        osc_bin=2,
        gain_e_per_adu=1.5,
        read_noise_e=3.0,
    )
    assert set(written) == set(OSC_CHANNELS)
    with fits.open(written["G"], memmap=False) as hdul:
        ch_hdr = hdul[0].header
        ch_data = hdul[0].data
        assert ch_hdr["VY_CHANNEL"] == "G"
        assert float(ch_hdr["VY_EGAIN"]) > 1.5
        cb_after = checkerboard_column_delta(ch_data)
    assert cb_before > 10.0
    assert cb_after == pytest.approx(0.0, abs=1.0)


def test_sky_surface_not_on_mosaic_regression(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import pipeline as pl
    import pipeline_calibrate as pc

    calls: list[str] = []

    def _track(data, order=2, **kw):
        calls.append(f"order={order}")
        return data, {"sky_surface_applied": False, "sky_surface_order": order}

    monkeypatch.setattr(pc, "_fit_subtract_preprocess_sky_surface", _track)
    mosaic = tmp_path / "mosaic.fits"
    hdr = fits.Header()
    hdr["BAYERPAT"] = "RGGB"
    fits.writeto(mosaic, np.ones((8, 8), dtype=np.float32), hdr, overwrite=True)
    pl._qc_enrich_calibrated_in_place(tmp_path, app_config=pl.AppConfig())
    assert calls == []
    mono_dir = tmp_path / "mono_only"
    mono_dir.mkdir()
    mono = mono_dir / "mono.fits"
    fits.writeto(mono, np.ones((8, 8), dtype=np.float32), overwrite=True)
    pl._qc_enrich_calibrated_in_place(mono_dir, app_config=pl.AppConfig())
    assert len(calls) == 1
    ch_dir = tmp_path / "ch_only"
    ch_dir.mkdir()
    ch = ch_dir / "ch.fits"
    ch_hdr = fits.Header()
    ch_hdr["VY_CHANNEL"] = "R"
    fits.writeto(ch, np.ones((8, 8), dtype=np.float32), ch_hdr, overwrite=True)
    pl._qc_enrich_calibrated_in_place(ch_dir, app_config=pl.AppConfig())
    assert len(calls) == 2


def test_osc01_invariant_blocks_mosaic(tmp_path: Path) -> None:
    from invariants_runtime import InvariantViolation, check_osc01_channel_extraction_required

    fp = tmp_path / "mosaic.fits"
    hdr = fits.Header()
    hdr["BAYERPAT"] = "RGGB"
    fits.writeto(fp, np.ones((4, 4), dtype=np.float32), hdr, overwrite=True)
    with pytest.raises(InvariantViolation):
        check_osc01_channel_extraction_required([fp], equipment_bayermask="RGGB")
