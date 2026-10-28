"""APERTURE-K-2026-10 tests: k grid 1.0-2.0 x FWHM, k* by comparison-star leave-one-out scatter."""
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
    FALLBACK_F_STAR,
    ApertureGridNight,
    apply_grid_fluxes_to_frames,
    BrightnessK,
    apply_brightness_k,
    field_k_by_brightness,
    PerTargetChoice,
    choose_f_for_target,
    f_star_is_grid_edge,
    howell_snr,
    measure_flux_on_f_grid,
    pick_f_star_snr,
)
from photometry_defs import DEFAULT_APERTURE_F_GRID  # noqa: E402
from aperture_policy import fwhm_for_radius  # noqa: E402
from config import AppConfig  # noqa: E402


def test_fixed_k_grid() -> None:
    cfg = AppConfig()
    assert not hasattr(cfg, "aperture_policy_mode")  # per_target is the only policy (CFG-REVIEW-2026-10)
    assert tuple(DEFAULT_APERTURE_F_GRID) == (0.6, 0.75, 0.875, 1.0, 1.25, 1.5, 1.75, 2.0)  # AP-SMALL-K-2026-10
    assert not hasattr(cfg, "aperture_f_grid")


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
    out = apply_grid_fluxes_to_frames(df, night, catalog_ids=["T", "C1"], f_star=f_star, gain=None, read_noise=None, annulus_inner_fwhm=None, annulus_outer_fwhm=None)
    r1 = out.loc[out["source_file"] == "proc_f001.csv", "aperture_r_px"].to_numpy()
    r2 = out.loc[out["source_file"] == "proc_f002.csv", "aperture_r_px"].to_numpy()
    assert np.allclose(r1, 4.0)
    assert np.allclose(r2, 6.0)
    assert abs(float(r1[0]) - float(r1[1])) < 1e-12
    assert abs(float(r2[0]) - float(r2[1])) < 1e-12
    assert fwhm_for_radius(fwhm_frame_px=4.2, fwhm_night_median_px=5.5) == 4.2


def test_target_and_comps_share_f_star() -> None:
    """Noise-free identical comps -> equal (zero) scatter at every k -> tie -> largest k."""
    fwhm = 5.5
    f_grid = [1.0, 1.25, 1.5]
    night = ApertureGridNight(f_grid=f_grid, fwhm_night_px=fwhm)
    enc = {1.0: 0.94, 1.25: 0.99, 1.5: 0.999}
    for i in range(8):
        stem = f"f{i:03d}"
        night.frames[stem] = {c: {f: 5000.0 * enc[f] for f in f_grid} for c in ("T", "C1", "C2")}
        night.sky_pp[stem] = {c: {f: 800.0 for f in f_grid} for c in ("T", "C1", "C2")}
        night.fwhm_by_frame[stem] = fwhm
    ch = choose_f_for_target(night, target_cid="T", comp_ids=["C1", "C2"],
                             frame_order=sorted(night.frames.keys()), gain=1.0, read_noise=10.0)
    assert ch.f_star == 1.5
    assert abs(ch.r_ap_px - ch.f_star * fwhm) < 1e-9
    assert ch.reason == "comp_loo_scatter_min"
    assert ch.snr_by_f


def test_pick_f_star_snr_flat_top_largest() -> None:
    """Bright-star flat top -> largest f within 1% of max."""
    # Peak at 1.0; 1.2 and 1.5 within 1% -> pick 1.5
    snr = {0.6: 90.0, 0.8: 99.0, 1.0: 100.0, 1.2: 99.5, 1.5: 99.2, 2.0: 95.0}
    assert pick_f_star_snr(snr) == 1.5
    assert pick_f_star_snr({0.5: 10.0, 0.7: 50.0, 1.0: 40.0}) == 0.7
    assert pick_f_star_snr({0.5: float("nan"), 1.0: float("nan")}) is None


def test_grid_edge_flag() -> None:
    assert f_star_is_grid_edge(0.6, DEFAULT_APERTURE_F_GRID) is True
    assert f_star_is_grid_edge(1.0, DEFAULT_APERTURE_F_GRID) is False
    assert f_star_is_grid_edge(2.0, DEFAULT_APERTURE_F_GRID) is True
    assert f_star_is_grid_edge(1.5, DEFAULT_APERTURE_F_GRID) is False


def test_fallback_not_grid_midpoint_135() -> None:
    """No finite S/N -> documented FALLBACK_F_STAR=0.70, not grid midpoint 1.35."""
    night = ApertureGridNight(f_grid=list(DEFAULT_APERTURE_F_GRID), fwhm_night_px=5.0)
    for stem in ("a", "b", "c"):
        night.frames[stem] = {"T": {f: float("nan") for f in night.f_grid}}
        night.sky_pp[stem] = {"T": {f: float("nan") for f in night.f_grid}}
        night.fwhm_by_frame[stem] = 5.0
    ch = choose_f_for_target(
        night, target_cid="T", comp_ids=[], frame_order=sorted(night.frames.keys()),
        gain=1.0, read_noise=10.0,
    )
    assert abs(ch.f_star - FALLBACK_F_STAR) < 1e-12
    assert "fallback" in ch.reason
    assert abs(ch.f_star - 1.35) > 0.1


def test_config_ui_parity_keys() -> None:
    cfg = AppConfig()
    raw = cfg.to_json()
    assert "aperture_policy_mode" not in raw
    assert "aperture_f_grid" not in raw  # retired: the k grid is fixed (APERTURE-K-2026-10)
    reg = json.loads((ROOT / "dev" / "validation" / "params_registry.json").read_text(encoding="utf-8"))
    assert "aperture_f_grid" not in reg
    assert "aperture_policy_mode" not in reg


def test_measure_flux_grid_smoke() -> None:
    img = np.zeros((80, 80), dtype=float)
    yy, xx = np.mgrid[0:80, 0:80]
    img += 200.0 * np.exp(-0.5 * (((xx - 40) / 2.0) ** 2 + ((yy - 40) / 2.0) ** 2))
    img += 50.0
    xy = np.array([[40.0, 40.0]])
    out_f, out_s, _edge = measure_flux_on_f_grid(
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


def _noisy_night(total_flux: float, sky: float, *, n_frames: int = 300, n_comps: int = 6, seed: int = 7):
    """Comps measured with CCD-equation noise at each k (Gaussian PSF, gain 1, RN 5)."""
    fwhm = 5.0
    f_grid = list(DEFAULT_APERTURE_F_GRID)
    enc = _gaussian_growth_curve(f_grid, fwhm_px=fwhm, total_flux=total_flux)
    rng = np.random.default_rng(seed)
    night = ApertureGridNight(f_grid=f_grid, fwhm_night_px=fwhm)
    ids = [f"C{j}" for j in range(n_comps)]
    for i in range(n_frames):
        stem = f"f{i:04d}"
        night.frames[stem], night.sky_pp[stem] = {}, {}
        night.fwhm_by_frame[stem] = fwhm
        for c in ["T", *ids]:
            u = rng.normal(size=1)[0]  # same random draw scaled per k (nested apertures)
            fl, sk = {}, {}
            for f in f_grid:
                area = math.pi * (f * fwhm) ** 2
                sd = math.sqrt(enc[f] + area * (sky + 25.0))
                fl[f] = enc[f] + sd * (u + 0.6 * rng.normal())
                sk[f] = sky
            night.frames[stem][c] = fl
            night.sky_pp[stem][c] = sk
    return night, ids


def test_sky_limited_comps_choose_small_k() -> None:
    night, ids = _noisy_night(total_flux=3000.0, sky=2000.0)
    ch = choose_f_for_target(night, target_cid="T", comp_ids=ids,
                             frame_order=sorted(night.frames.keys()), gain=1.0, read_noise=5.0)
    assert ch.f_star <= 0.75, ch.p2p_by_f  # AP-SMALL-K-2026-10: below the former grid edge 1.0


def test_photon_limited_comps_choose_larger_k_than_sky_limited() -> None:
    faint, ids = _noisy_night(total_flux=3000.0, sky=2000.0)
    bright, _ = _noisy_night(total_flux=3.0e6, sky=20.0)
    f_faint = choose_f_for_target(faint, target_cid="T", comp_ids=ids,
                                  frame_order=sorted(faint.frames.keys()), gain=1.0, read_noise=10.0).f_star
    f_bright = choose_f_for_target(bright, target_cid="T", comp_ids=ids,
                                   frame_order=sorted(bright.frames.keys()), gain=1.0, read_noise=10.0).f_star
    assert f_bright > f_faint


def test_missing_grid_flux_gives_nan_not_other_aperture() -> None:
    night = ApertureGridNight(f_grid=[1.0, 1.5], fwhm_night_px=5.0)
    night.fwhm_by_frame = {"a": 5.0}
    night.frames = {"a": {"T": {1.0: 1000.0}, "C1": {}}}
    df = pd.DataFrame([
        {"catalog_id": "T", "source_file": "proc_a.csv", "mag_inst": 1.0},
        {"catalog_id": "C1", "source_file": "proc_a.csv", "mag_inst": 2.0},
    ])
    out = apply_grid_fluxes_to_frames(df, night, catalog_ids=["T", "C1"], f_star=1.0, gain=None, read_noise=None, annulus_inner_fwhm=None, annulus_outer_fwhm=None)
    assert math.isnan(float(out.loc[out.catalog_id == "C1", "mag_inst"].iloc[0]))


def test_err_from_ccd_equation_for_chosen_aperture() -> None:
    night = ApertureGridNight(f_grid=[1.0], fwhm_night_px=4.0)
    night.fwhm_by_frame = {"a": 4.0}
    night.frames = {"a": {"T": {1.0: 10000.0}}}
    night.sky_pp = {"a": {"T": {1.0: 100.0}}}
    df = pd.DataFrame([{"catalog_id": "T", "source_file": "proc_a.csv", "mag_inst": 0.0, "err": 9.9}])
    out = apply_grid_fluxes_to_frames(df, night, catalog_ids=["T"], f_star=1.0, gain=2.0, read_noise=4.0,
                                      annulus_inner_fwhm=3.0, annulus_outer_fwhm=5.0)
    A = math.pi * 16.0
    nsky = math.pi * (400.0 - 144.0)
    var = 10000.0 / 2.0 + A * (1.0 + A / nsky) * (100.0 / 2.0 + 4.0)
    assert abs(float(out["err"].iloc[0]) - math.sqrt(var) / 10000.0) < 1e-12


def test_howell_snr_monotone_in_flux() -> None:
    s1 = howell_snr(1000.0, 100.0, 50.0, gain=1.0, read_noise=10.0)
    s2 = howell_snr(2000.0, 100.0, 50.0, gain=1.0, read_noise=10.0)
    assert s2 > s1 > 0


def _ch(cid: str, curve: dict[float, float], allowed=None) -> "PerTargetChoice":
    best = min(curve.values())
    return PerTargetChoice(catalog_id=cid, f_star=max(k for k, v in curve.items() if v <= best * 1.01),
                           r_ap_px=0.0, p2p_by_f={f"{k:.4g}": v for k, v in curve.items()}, f_allowed=allowed)


def _synthetic_procs(tmp: Path, *, n_frames: int = 40, seed: int = 0) -> Path:
    """Proc CSVs: bright stars whose scatter falls with k (seeing / centring drifts), faint stars whose
    scatter grows with k (sky noise), plus a variable and a saturated star that must be ignored."""
    rng = np.random.default_rng(seed)
    g = list(DEFAULT_APERTURE_F_GRID)
    stars = [(f"{1000000000000000000 + i}", 2.0e5, "bright") for i in range(80)]
    stars += [(f"{2000000000000000000 + i}", 2.0e2, "faint") for i in range(120)]
    stars += [("3000000000000000001", 2.0e3, "var"), ("3000000000000000002", 5.0e5, "sat")]  # Gaia-like ids
    d = tmp / "frames"
    d.mkdir()
    for f in range(n_frames):
        rows = []
        for cid, flux, kind in stars:
            r = {"catalog_id": cid, "vsx_known_variable": kind == "var", "is_saturated": kind == "sat" and f == 3}
            for k in g:
                sig = 0.004 / k if kind in ("bright", "sat") else 0.02 * k  # magnitude scatter at this k
                m = -2.5 * np.log10(flux) + rng.normal(0, sig) + (0.3 * math.sin(f) if kind == "var" else 0.0)
                r[f"flux_k{int(round(k * 100)):03d}"] = 10 ** (-0.4 * m)
            rows.append(r)
        pd.DataFrame(rows).to_csv(d / f"proc_f{f:03d}.csv", index=False)
    return d


def test_brightness_k_from_constant_field_stars(tmp_path: Path) -> None:
    """AP-BRIGHT-K-2026-10: bright stars take a large k, faint stars the smallest; variables and stars
    saturated on any frame are not used."""
    d = _synthetic_procs(tmp_path)
    bk = field_k_by_brightness(sorted(d.glob("proc_*.csv")), DEFAULT_APERTURE_F_GRID)
    assert bk is not None and bk.k[0] == 2.0 and set(bk.k) == {0.6, 2.0}
    assert bk.k_for(-2.5 * math.log10(2.0e5)) == 2.0
    assert bk.k_for(-2.5 * math.log10(2.0e2)) == 0.6
    assert sum(bk.n_stars) == 200  # VAR and SAT excluded
    assert bk.k_for(-30.0) == 2.0 and bk.k_for(0.0) == 0.6  # outside the range: end groups
    assert bk.k_for(float("nan")) is None


def test_brightness_k_assignment_blend_limit_and_no_magnitude() -> None:
    g = list(DEFAULT_APERTURE_F_GRID)
    bk = BrightnessK(mag_lo=[float("-inf"), -10.0], mag_hi=[-10.0, float("inf")], k=[1.5, 1.0], n_stars=[50, 50],
                     curves=[{}, {}], se=[0.0, 0.0])
    ch = {
        "BR": PerTargetChoice(catalog_id="BR", f_star=1.0, r_ap_px=5.0),
        "FA": PerTargetChoice(catalog_id="FA", f_star=1.0, r_ap_px=5.0),
        "BL": PerTargetChoice(catalog_id="BL", f_star=1.0, r_ap_px=5.0, f_allowed=[1.0, 1.25]),
        "NM": PerTargetChoice(catalog_id="NM", f_star=1.25, r_ap_px=6.25),
    }
    info = apply_brightness_k(ch, g, {"BR": -12.0, "FA": -7.0, "BL": -12.0, "NM": float("nan")}, bk)
    assert ch["BR"].f_star == 1.5 and abs(ch["BR"].r_ap_px - 7.5) < 1e-9 and ch["BR"].reason == "brightness_k"
    assert ch["FA"].f_star == 1.0
    assert ch["BL"].f_star == 1.25 and "blend limit" in ch["BL"].reason
    assert ch["NM"].f_star == 1.0 and "no brightness group" in ch["NM"].reason
    assert info["k_by_target"]["BR"] == {"inst_mag": -12.0, "k": 1.5}


def test_brightness_k_does_not_depend_on_the_target_comps() -> None:
    """The k comes from the field, not from the target's own comp curve (stable when a comp changes)."""
    g = list(DEFAULT_APERTURE_F_GRID)
    bk = BrightnessK(mag_lo=[float("-inf")], mag_hi=[float("inf")], k=[1.25], n_stars=[50], curves=[{}], se=[0.0])
    a = {"T": PerTargetChoice(catalog_id="T", f_star=1.0, r_ap_px=5.0, p2p_by_f={"1": 1.0, "1.25": 2.0})}
    b = {"T": PerTargetChoice(catalog_id="T", f_star=2.0, r_ap_px=10.0, p2p_by_f={"1": 2.0, "2": 1.0})}
    apply_brightness_k(a, g, {"T": -9.0}, bk)
    apply_brightness_k(b, g, {"T": -9.0}, bk)
    assert a["T"].f_star == b["T"].f_star == 1.25


def test_brightness_k_without_field_curve_gives_smallest_k() -> None:
    ch = {"N": PerTargetChoice(catalog_id="N", f_star=1.5, r_ap_px=7.5)}
    info = apply_brightness_k(ch, DEFAULT_APERTURE_F_GRID, {"N": -9.0}, None)
    assert info["brightness_k"] is None and ch["N"].f_star == 1.0
