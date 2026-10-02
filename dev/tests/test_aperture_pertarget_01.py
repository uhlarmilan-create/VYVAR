"""APERTURE-DYNAMIC-02 / APERTURE-PERTARGET tests (Howell S/N f*)."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src_py"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aperture_pertarget import (  # noqa: E402
    DEFAULT_APERTURE_F_GRID,
    FALLBACK_F_STAR,
    ApertureGridNight,
    abbe_p2p_scatter,
    apply_grid_fluxes_to_frames,
    choose_f_for_target,
    f_star_is_grid_edge,
    howell_snr,
    measure_flux_on_f_grid,
    normalize_f_grid,
    pick_f_star,
    pick_f_star_snr,
)
from aperture_policy import fwhm_for_radius, normalize_aperture_policy_mode  # noqa: E402
from config import AppConfig  # noqa: E402


def test_default_mode_is_per_target() -> None:
    """APERTURE-DYNAMIC-02: default config is per_target with fine grid."""
    cfg = AppConfig()
    assert cfg.aperture_policy_mode == "per_target"
    assert normalize_aperture_policy_mode(cfg.aperture_policy_mode) == "per_target"
    assert abs(DEFAULT_APERTURE_F_GRID[0] - 0.4) < 1e-12
    assert abs(DEFAULT_APERTURE_F_GRID[-1] - 3.0) < 1e-12
    # step <= 0.05
    diffs = np.diff(DEFAULT_APERTURE_F_GRID)
    assert float(np.max(diffs)) <= 0.05 + 1e-12
    assert normalize_f_grid(None) == list(DEFAULT_APERTURE_F_GRID)
    assert abs(cfg.aperture_f_grid[0] - 0.4) < 1e-12
    assert abs(FALLBACK_F_STAR - 0.70) < 1e-12


def test_per_frame_r_scales_with_frame_fwhm() -> None:
    """r = f* x FWHM_frame for target and comps (same f*)."""
    f_star = 1.0
    night = ApertureGridNight(f_grid=[0.5, 1.0, 1.35], fwhm_night_px=5.0)
    night.fwhm_by_frame = {"f001": 4.0, "f002": 6.0}
    night.frames = {
        "f001": {"T": {1.0: 1000.0}, "C1": {1.0: 900.0}},
        "f002": {"T": {1.0: 1100.0}, "C1": {1.0: 950.0}},
    }
    df = pd.DataFrame(
        [
            {"catalog_id": "T", "source_file": "proc_f001.csv", "mag_inst": 99.0},
            {"catalog_id": "C1", "source_file": "proc_f001.csv", "mag_inst": 99.0},
            {"catalog_id": "T", "source_file": "proc_f002.csv", "mag_inst": 99.0},
            {"catalog_id": "C1", "source_file": "proc_f002.csv", "mag_inst": 99.0},
        ]
    )
    out = apply_grid_fluxes_to_frames(df, night, catalog_ids=["T", "C1"], f_star=f_star)
    r1 = out.loc[out["source_file"] == "proc_f001.csv", "aperture_r_px"].to_numpy()
    r2 = out.loc[out["source_file"] == "proc_f002.csv", "aperture_r_px"].to_numpy()
    assert np.allclose(r1, 4.0)
    assert np.allclose(r2, 6.0)
    assert abs(float(r1[0]) - float(r1[1])) < 1e-12
    assert abs(float(r2[0]) - float(r2[1])) < 1e-12
    assert fwhm_for_radius("per_target", fwhm_frame_px=4.2, fwhm_night_median_px=5.5) == 4.2


def test_target_and_comps_share_f_star() -> None:
    fwhm = 5.5
    f_grid = [0.6, 0.8, 1.0, 1.2, 1.5]
    night = ApertureGridNight(f_grid=f_grid, fwhm_night_px=fwhm)
    # Enclosed-energy growth ~ erf-like; sky high so S/N peaks mid-grid.
    enc = {0.6: 0.55, 0.8: 0.72, 1.0: 0.85, 1.2: 0.92, 1.5: 0.96}
    F0 = 5000.0
    sky = 800.0
    for stem in ("f001", "f002", "f003", "f004", "f005", "f006", "f007", "f008"):
        night.frames[stem] = {}
        night.sky_pp[stem] = {}
        night.fwhm_by_frame[stem] = fwhm
        for cid in ("T", "C1", "C2"):
            night.frames[stem][cid] = {f: F0 * enc[f] for f in f_grid}
            night.sky_pp[stem][cid] = {f: sky for f in f_grid}
    ch = choose_f_for_target(
        night,
        target_cid="T",
        comp_ids=["C1", "C2"],
        frame_order=sorted(night.frames.keys()),
        gain=1.0,
        read_noise=10.0,
    )
    assert ch.f_star in night.f_grid
    assert abs(ch.r_ap_px - ch.f_star * fwhm) < 1e-9
    assert ch.reason.startswith("snr_")
    assert ch.snr_by_f


def test_pick_f_star_snr_flat_top_largest() -> None:
    """Bright-star flat top -> largest f within 1% of max."""
    # Peak at 1.0; 1.2 and 1.5 within 1% -> pick 1.5
    snr = {0.6: 90.0, 0.8: 99.0, 1.0: 100.0, 1.2: 99.5, 1.5: 99.2, 2.0: 95.0}
    assert pick_f_star_snr(snr) == 1.5
    assert pick_f_star_snr({0.5: 10.0, 0.7: 50.0, 1.0: 40.0}) == 0.7
    assert pick_f_star_snr({0.5: float("nan"), 1.0: float("nan")}) is None


def test_grid_edge_flag() -> None:
    assert f_star_is_grid_edge(0.4, DEFAULT_APERTURE_F_GRID) is True
    assert f_star_is_grid_edge(3.0, DEFAULT_APERTURE_F_GRID) is True
    assert f_star_is_grid_edge(1.35, DEFAULT_APERTURE_F_GRID) is False
    assert pick_f_star({0.5: 0.01, 0.6: 0.02, 1.0: 0.03, 2.5: 0.04}) == 0.5


def test_fallback_not_grid_midpoint_135() -> None:
    """No finite S/N -> documented FALLBACK_F_STAR=0.70, not grid midpoint 1.35."""
    night = ApertureGridNight(f_grid=list(DEFAULT_APERTURE_F_GRID), fwhm_night_px=5.0)
    for stem in ("a", "b", "c"):
        night.frames[stem] = {"T": {f: float("nan") for f in night.f_grid}}
        night.sky_pp[stem] = {"T": {f: float("nan") for f in night.f_grid}}
        night.fwhm_by_frame[stem] = 5.0
    ch = choose_f_for_target(
        night, target_cid="T", comp_ids=[], frame_order=sorted(night.frames.keys())
    )
    assert abs(ch.f_star - FALLBACK_F_STAR) < 1e-12
    assert "fallback" in ch.reason
    assert abs(ch.f_star - 1.35) > 0.1


def test_config_ui_parity_keys() -> None:
    cfg = AppConfig()
    raw = cfg.to_json()
    assert raw["aperture_policy_mode"] == "per_target"
    assert abs(raw["aperture_f_grid"][0] - 0.4) < 1e-12
    reg = json.loads((ROOT / "dev" / "validation" / "params_registry.json").read_text(encoding="utf-8"))
    assert "aperture_f_grid" in reg
    assert "aperture_policy_mode" in reg
    assert "Howell" in reg["aperture_policy_mode"]["help"]


def test_measure_flux_grid_smoke() -> None:
    img = np.zeros((80, 80), dtype=float)
    yy, xx = np.mgrid[0:80, 0:80]
    img += 200.0 * np.exp(-0.5 * (((xx - 40) / 2.0) ** 2 + ((yy - 40) / 2.0) ** 2))
    img += 50.0
    xy = np.array([[40.0, 40.0]])
    out_f, out_s = measure_flux_on_f_grid(
        img,
        xy,
        f_grid=[1.0, 1.35],
        fwhm_px=4.0,
        annulus_inner_fwhm=2.7,
        annulus_outer_fwhm=5.2,
    )
    assert set(out_f.keys()) == {1.0, 1.35}
    assert out_f[1.35][0] > out_f[1.0][0] > 0
    assert math.isfinite(float(out_s[1.0][0]))


def _gaussian_growth_curve(
    f_grid: list[float],
    *,
    fwhm_px: float,
    total_flux: float,
) -> dict[float, float]:
    """Exact enclosed flux for 2D Gaussian: 1 - exp(-r^2 / (2 sigma^2))."""
    sigma = float(fwhm_px) / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    out = {}
    for f in f_grid:
        r = float(f) * float(fwhm_px)
        enc = 1.0 - math.exp(-(r * r) / (2.0 * sigma * sigma))
        out[float(f)] = float(total_flux) * enc
    return out


def test_synthetic_sky_limited_f_star_band() -> None:
    """Sky-limited Gaussian star -> f* in ~0.6-0.75 FWHM; not on grid edge."""
    fwhm = 5.0
    f_grid = list(DEFAULT_APERTURE_F_GRID)
    enc = _gaussian_growth_curve(f_grid, fwhm_px=fwhm, total_flux=800.0)
    sky = 2000.0  # high sky -> sky-limited
    night = ApertureGridNight(f_grid=f_grid, fwhm_night_px=fwhm)
    for stem in [f"f{i:03d}" for i in range(12)]:
        night.frames[stem] = {"T": dict(enc), "C1": {f: enc[f] * 0.9 for f in enc}}
        night.sky_pp[stem] = {
            "T": {f: sky for f in f_grid},
            "C1": {f: sky for f in f_grid},
        }
        night.fwhm_by_frame[stem] = fwhm
    ch = choose_f_for_target(
        night,
        target_cid="T",
        comp_ids=["C1"],
        frame_order=sorted(night.frames.keys()),
        gain=1.0,
        read_noise=10.0,
    )
    assert 0.55 <= ch.f_star <= 0.85, f"sky-limited f*={ch.f_star}"
    assert ch.f_edge is False


def test_synthetic_photon_limited_f_star_larger() -> None:
    """Photon-limited (bright, low sky) -> f* larger than sky-limited; interior."""
    fwhm = 5.0
    f_grid = list(DEFAULT_APERTURE_F_GRID)
    enc_bright = _gaussian_growth_curve(f_grid, fwhm_px=fwhm, total_flux=1.0e5)
    enc_faint = _gaussian_growth_curve(f_grid, fwhm_px=fwhm, total_flux=800.0)
    # Modest sky: S/N peaks interior but at larger f than the faint/sky-limited case.
    sky_bright = 200.0
    sky_faint = 2000.0

    def _run(enc, sky):
        night = ApertureGridNight(f_grid=f_grid, fwhm_night_px=fwhm)
        for stem in [f"f{i:03d}" for i in range(12)]:
            night.frames[stem] = {"T": dict(enc)}
            night.sky_pp[stem] = {"T": {f: sky for f in f_grid}}
            night.fwhm_by_frame[stem] = fwhm
        return choose_f_for_target(
            night,
            target_cid="T",
            comp_ids=[],
            frame_order=sorted(night.frames.keys()),
            gain=1.0,
            read_noise=5.0,
        )

    ch_sky = _run(enc_faint, sky_faint)
    ch_ph = _run(enc_bright, sky_bright)
    assert ch_ph.f_star > ch_sky.f_star + 0.05, (
        f"photon f*={ch_ph.f_star} should exceed sky f*={ch_sky.f_star}"
    )
    assert ch_ph.f_edge is False, f"photon f* on edge: {ch_ph.f_star}"
    assert ch_sky.f_edge is False, f"sky f* on edge: {ch_sky.f_star}"


def test_howell_snr_monotone_in_flux() -> None:
    s1 = howell_snr(1000.0, 100.0, 50.0, gain=1.0, read_noise=10.0)
    s2 = howell_snr(2000.0, 100.0, 50.0, gain=1.0, read_noise=10.0)
    assert s2 > s1 > 0
    assert math.isfinite(abbe_p2p_scatter(np.full(40, 12.0) + 0.01 * np.random.default_rng(1).normal(size=40)))
