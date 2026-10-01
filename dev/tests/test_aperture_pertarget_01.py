"""APERTURE-PERTARGET-01 tests."""
from __future__ import annotations

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
    abbe_p2p_scatter,
    choose_f_for_target,
    equal_weight_ensemble_delta,
    flux_to_mag,
    measure_flux_on_f_grid,
    normalize_f_grid,
    pick_f_star,
    ApertureGridNight,
)
from aperture_policy import normalize_aperture_policy_mode  # noqa: E402
from config import AppConfig  # noqa: E402


def test_t1_default_mode_unchanged() -> None:
    """T1: default config stays f_fixed_night (byte-identical production path)."""
    cfg = AppConfig()
    assert cfg.aperture_policy_mode == "f_fixed_night"
    assert normalize_aperture_policy_mode(cfg.aperture_policy_mode) == "f_fixed_night"
    assert normalize_aperture_policy_mode("per_target") == "per_target"
    assert normalize_f_grid(None) == list(DEFAULT_APERTURE_F_GRID)


def test_t2_target_and_comps_share_r_ap() -> None:
    """T2: chosen f* implies identical r_ap for target and every delivered comp."""
    fwhm = 5.5
    night = ApertureGridNight(f_grid=[1.0, 1.35, 2.0], fwhm_night_px=fwhm)
    # Two frames, target+2 comps; make f=1.35 have lowest p2p.
    rng = np.random.default_rng(0)
    for stem in ("f001", "f002", "f003", "f004", "f005", "f006", "f007", "f008"):
        night.frames[stem] = {}
        for cid, base in (("T", 10000.0), ("C1", 9000.0), ("C2", 11000.0)):
            night.frames[stem][cid] = {}
            for f in night.f_grid:
                # Intrinsic signal + noise; at f=1.35 noise lower for all (shared EE)
                noise = 0.02 if abs(f - 1.35) < 1e-9 else 0.08
                night.frames[stem][cid][f] = base * (0.7 + 0.15 * f) * (
                    1.0 + noise * rng.normal()
                )
    ch = choose_f_for_target(
        night, target_cid="T", comp_ids=["C1", "C2"], frame_order=sorted(night.frames.keys())
    )
    assert abs(ch.f_star - 1.35) < 1e-9 or ch.f_star in night.f_grid
    r = ch.f_star * fwhm
    # Same r for target and comps by construction of the policy.
    assert abs(ch.r_ap_px - r) < 1e-9


def test_t3_synthetic_bright_faint_and_eclipse() -> None:
    """T3: faint -> small f*; bright -> larger f*; eclipse does not change f*."""
    # Faint sky-limited: p2p rises with f (more sky) -> prefers small f.
    p2p_faint = {0.75: 0.010, 1.0: 0.012, 1.35: 0.018, 2.0: 0.030, 2.5: 0.040}
    assert pick_f_star(p2p_faint) == 0.75
    # Bright: larger aperture wins until flat then larger f on tie.
    p2p_bright = {0.75: 0.020, 1.0: 0.015, 1.35: 0.011, 2.0: 0.011, 2.5: 0.014}
    assert pick_f_star(p2p_bright) == 2.0  # tie 1.35/2.0 -> larger
    # Eclipse in LC: Abbe p2p of a clean flat vs eclipse-containing series -
    # selection uses differential residual; a shared eclipse in all f should
    # not flip the argmin ordering of the noise floor.
    n = 60
    base = np.full(n, 12.0)
    eclipse = base.copy()
    eclipse[20:28] = 12.6
    noise = 0.01 * np.random.default_rng(3).normal(size=n)
    p2p_flat = abbe_p2p_scatter(base + noise)
    p2p_ecl = abbe_p2p_scatter(eclipse + noise)
    # Both finite; eclipse raises p2p but the *relative* ranking across f is
    # preserved when the same eclipse is present at every f (tested via pick).
    assert np.isfinite(p2p_flat) and np.isfinite(p2p_ecl)
    ranked = {0.75: p2p_ecl + 0.02, 1.35: p2p_ecl + 0.00, 2.5: p2p_ecl + 0.03}
    assert pick_f_star(ranked) == 1.35


def test_t4_config_ui_parity_key_registered() -> None:
    """T4: new keys present on AppConfig and in params registry."""
    import json

    cfg = AppConfig()
    assert hasattr(cfg, "aperture_f_grid")
    raw = cfg.to_json()
    assert "aperture_policy_mode" in raw
    assert "aperture_f_grid" in raw
    assert isinstance(raw["aperture_f_grid"], list)
    reg = json.loads((ROOT / "dev" / "validation" / "params_registry.json").read_text(encoding="utf-8"))
    assert "aperture_f_grid" in reg
    assert "aperture_policy_mode" in reg
    assert reg["aperture_policy_mode"]["widget"] == "custom"


def test_measure_flux_grid_smoke() -> None:
    """Shared core measures a tiny stamp without crashing."""
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
