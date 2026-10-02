"""APERTURE-DYNAMIC-01 / APERTURE-PERTARGET-01 tests."""
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
    ApertureGridNight,
    abbe_p2p_scatter,
    apply_grid_fluxes_to_frames,
    choose_f_for_target,
    f_star_is_grid_edge,
    measure_flux_on_f_grid,
    normalize_f_grid,
    pick_f_star,
)
from aperture_policy import fwhm_for_radius, normalize_aperture_policy_mode  # noqa: E402
from config import AppConfig  # noqa: E402


def test_default_mode_is_per_target() -> None:
    """APERTURE-DYNAMIC-01: default config is per_target."""
    cfg = AppConfig()
    assert cfg.aperture_policy_mode == "per_target"
    assert normalize_aperture_policy_mode(cfg.aperture_policy_mode) == "per_target"
    assert 0.5 in DEFAULT_APERTURE_F_GRID and 0.6 in DEFAULT_APERTURE_F_GRID
    assert normalize_f_grid(None) == list(DEFAULT_APERTURE_F_GRID)
    assert cfg.aperture_f_grid[0] <= 0.5 + 1e-12


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
    night = ApertureGridNight(f_grid=[1.0, 1.35, 2.0], fwhm_night_px=fwhm)
    rng = np.random.default_rng(0)
    for stem in ("f001", "f002", "f003", "f004", "f005", "f006", "f007", "f008"):
        night.frames[stem] = {}
        night.fwhm_by_frame[stem] = fwhm
        for cid, base in (("T", 10000.0), ("C1", 9000.0), ("C2", 11000.0)):
            night.frames[stem][cid] = {}
            for f in night.f_grid:
                noise = 0.02 if abs(f - 1.35) < 1e-9 else 0.08
                night.frames[stem][cid][f] = base * (0.7 + 0.15 * f) * (
                    1.0 + noise * rng.normal()
                )
    ch = choose_f_for_target(
        night, target_cid="T", comp_ids=["C1", "C2"], frame_order=sorted(night.frames.keys())
    )
    assert ch.f_star in night.f_grid
    assert abs(ch.r_ap_px - ch.f_star * fwhm) < 1e-9


def test_grid_edge_flag() -> None:
    assert f_star_is_grid_edge(0.5, DEFAULT_APERTURE_F_GRID) is True
    assert f_star_is_grid_edge(2.5, DEFAULT_APERTURE_F_GRID) is True
    assert f_star_is_grid_edge(1.35, DEFAULT_APERTURE_F_GRID) is False
    assert pick_f_star({0.5: 0.01, 0.6: 0.02, 1.0: 0.03, 2.5: 0.04}) == 0.5
    night = ApertureGridNight(f_grid=[0.5, 1.0, 2.5], fwhm_night_px=5.0)
    for stem in ("a", "b", "c", "d", "e", "f"):
        night.frames[stem] = {
            "T": {0.5: 1000.0, 1.0: 1000.0 * (1.05), 2.5: 1000.0 * (1.15)},
            "C1": {0.5: 900.0, 1.0: 900.0 * (1.05), 2.5: 900.0 * (1.15)},
        }
        # Extra noise at larger f so p2p rises with f
        rng = np.random.default_rng(abs(hash(stem)) % (2**31))
        night.frames[stem]["T"][1.0] *= 1.0 + 0.08 * rng.normal()
        night.frames[stem]["T"][2.5] *= 1.0 + 0.15 * rng.normal()
        night.frames[stem]["C1"][1.0] *= 1.0 + 0.08 * rng.normal()
        night.frames[stem]["C1"][2.5] *= 1.0 + 0.15 * rng.normal()
    ch = choose_f_for_target(
        night, target_cid="T", comp_ids=["C1"], frame_order=sorted(night.frames.keys())
    )
    assert ch.f_star == 0.5
    assert ch.f_edge is True


def test_config_ui_parity_keys() -> None:
    cfg = AppConfig()
    raw = cfg.to_json()
    assert raw["aperture_policy_mode"] == "per_target"
    assert 0.5 in raw["aperture_f_grid"]
    reg = json.loads((ROOT / "dev" / "validation" / "params_registry.json").read_text(encoding="utf-8"))
    assert "aperture_f_grid" in reg
    assert "aperture_policy_mode" in reg


def test_measure_flux_grid_smoke() -> None:
    img = np.zeros((80, 80), dtype=float)
    yy, xx = np.mgrid[0:80, 0:80]
    img += 200.0 * np.exp(-0.5 * (((xx - 40) / 2.0) ** 2 + ((yy - 40) / 2.0) ** 2))
    img += 50.0
    xy = np.array([[40.0, 40.0]])
    out = measure_flux_on_f_grid(
        img,
        xy,
        f_grid=[1.0, 1.35],
        fwhm_px=4.0,
        annulus_inner_fwhm=2.7,
        annulus_outer_fwhm=5.2,
    )
    assert set(out.keys()) == {1.0, 1.35}
    assert out[1.35][0] > out[1.0][0] > 0


def test_synthetic_bright_faint_pick() -> None:
    assert pick_f_star({0.5: 0.009, 0.75: 0.010, 1.0: 0.012, 1.35: 0.018, 2.5: 0.040}) == 0.5
    assert pick_f_star({0.5: 0.020, 1.0: 0.015, 1.35: 0.011, 2.0: 0.011, 2.5: 0.014}) == 2.0
    n = 60
    base = np.full(n, 12.0)
    noise = 0.01 * np.random.default_rng(3).normal(size=n)
    assert math.isfinite(abbe_p2p_scatter(base + noise))
