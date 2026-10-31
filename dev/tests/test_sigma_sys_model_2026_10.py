"""SIGMA-SYS-MODEL-2026-10 (D3): the night floor is what is left above photon + scintillation +
reference noise. Simulated comps with a known floor give it back; with no floor the floor is ~0."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sigma_floor_core import night_sigma_sys_from_comps


def _night(sys_mag: float, seed: int = 3):
    rng = np.random.default_rng(seed)
    n_c, n_f, g, sky, fw, k = 12, 300, 1.0, 400.0, 3.0, 1.25
    area = math.pi * (k * fw) ** 2
    n_sky = math.pi * ((6 * fw) ** 2 - (4 * fw) ** 2)
    scint = np.full(n_f, 0.004)
    flux, skyd = {}, {}
    for i in range(n_c):
        f0 = 10 ** (-0.4 * (9.0 + 0.25 * i - 25.0))
        var = f0 / g + area * (1 + (math.pi / 2) * area / n_sky) * (sky / g)  # CCD-ONE-2026-10: median sky
        e_ph = 1.0857 * math.sqrt(var) / f0
        dm = rng.normal(0, e_ph, n_f) + rng.normal(0, 0.004, n_f) + rng.normal(0, sys_mag, n_f)
        flux[str(i)] = f0 * 10 ** (-0.4 * dm)
        skyd[str(i)] = np.full(n_f, sky)
    nsk = {c: np.full(n_f, n_sky) for c in flux}
    return night_sigma_sys_from_comps(flux, skyd, np.full(n_f, fw), k=k, gain=g, read_noise=0.0,
                                      n_sky_by_cid=nsk, scint_mag_by_frame=scint)


def test_no_floor_gives_no_floor():
    assert _night(0.0)["sigma_sys_mag"] < 0.0015


@pytest.mark.parametrize("sys_mag", [0.003, 0.006])
def test_known_floor_is_recovered(sys_mag):
    v = _night(sys_mag)["sigma_sys_mag"]
    assert abs(v - sys_mag) < 0.0015
