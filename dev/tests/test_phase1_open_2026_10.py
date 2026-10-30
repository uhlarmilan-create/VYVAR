"""Phase-1 open items, Milan's decisions 2026-10-28: AP-SMALL-K, BLEND-SEEING, SIGMA-SYS-TOTAL / CHECK-SCINT, SKY-ONE."""

from __future__ import annotations

import ast
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src_py"
sys.path.insert(0, str(SRC))


def test_grid_reaches_down_to_0p6_and_columns_follow_it() -> None:
    """AP-SMALL-K-2026-10: one grid definition; proc / comp-table columns are generated from it."""
    from photometry_defs import BLEND_K_COLUMNS, DEFAULT_APERTURE_F_GRID, FLUX_K_COLUMNS, PHASE_USECOLS_PERFRAME
    from proc_frame_store import PROC_ROUND_SIG_COLS

    assert min(DEFAULT_APERTURE_F_GRID) == 0.6 and max(DEFAULT_APERTURE_F_GRID) == 2.0
    assert FLUX_K_COLUMNS[0] == "flux_k060" and FLUX_K_COLUMNS[3] == "flux_k100"
    assert BLEND_K_COLUMNS[0] == "blend_k060"
    assert set(FLUX_K_COLUMNS) <= set(PHASE_USECOLS_PERFRAME)
    assert set(FLUX_K_COLUMNS) <= set(PROC_ROUND_SIG_COLS)
    for p in SRC.glob("*.py"):
        if p.name == "photometry_defs.py":
            continue
        assert '"flux_k100",\n    "flux_k125"' not in p.read_text(encoding="utf-8"), p.name


def test_no_brightness_group_keeps_k_1() -> None:
    """Without a measured field curve a target does not get the small apertures."""
    from aperture_pertarget import BRIGHT_K_NO_GROUP, PerTargetChoice, apply_brightness_k
    from photometry_defs import DEFAULT_APERTURE_F_GRID

    ch = {"N": PerTargetChoice(catalog_id="N", f_star=1.5, r_ap_px=7.5, p2p_by_f={}, snr_by_f={}, n_comps=0,
                               n_frames=0, f_edge=False, reason="", snr_max=float("nan"))}
    apply_brightness_k(ch, DEFAULT_APERTURE_F_GRID, {"N": float("nan")}, None)
    assert ch["N"].f_star == BRIGHT_K_NO_GROUP == 1.0


def test_blend_judged_at_bad_seeing() -> None:
    """BLEND-SEEING-2026-10: a neighbour clean at the median FWHM is a blend at the p90 FWHM."""
    from comp_blend import BLEND_FLUX_FRAC_MAX, BLEND_FWHM_QUANTILE, blend_fwhm_px, contamination_fractions

    fw = [4.0] * 50 + [5.0] * 30 + [8.5] * 20
    fw90, src = blend_fwhm_px(fw, 4.5)
    assert math.isclose(fw90, float(np.quantile(fw, BLEND_FWHM_QUANTILE))) and src.startswith("p90")
    assert blend_fwhm_px([], 4.5) == (4.5, "night median (no per-frame FWHM)")
    star, nb = np.array([[100.0, 100.0]]), np.array([[110.0, 100.0]])  # neighbour 10 px away, 2.5 mag fainter

    def c(f):
        return float(contamination_fractions(star, np.array([1.0]), nb, np.array([0.1]), r_ap_px=1.0 * f,
                                             fwhm_px=f, beta=2.7, annulus_px=(2.7 * f, 5.2 * f))[0])

    assert abs(c(4.5)) <= BLEND_FLUX_FRAC_MAX < abs(c(fw90))


def test_blend_callers_use_the_bad_seeing_fwhm() -> None:
    """Every blend evaluation passes the bad-seeing FWHM and measures beta at the median FWHM."""
    for name in ("photometry_comp.py", "aperture_photometry.py", "variability_field.py"):
        t = ast.parse((SRC / name).read_text(encoding="utf-8"))
        calls = [n for n in ast.walk(t) if isinstance(n, ast.Call)
                 and getattr(n.func, "id", getattr(n.func, "attr", "")) == "blend_contamination_for_pool"]
        assert calls, name
        for cl in calls:
            kw = {k.arg for k in cl.keywords}
            assert {"fwhm_px", "profile_fwhm_px"} <= kw, name
        assert "blend_fwhm_px(" in (SRC / name).read_text(encoding="utf-8"), name


def test_sigma_sys_holds_slow_systematics() -> None:
    """SIGMA-SYS-TOTAL-2026-10: a slow drift of the comps (not seen by p2p) enters the night floor."""
    from sigma_floor_core import night_sigma_sys_from_comps

    rng = np.random.default_rng(3)
    n_f, n_c = 120, 8
    t = np.linspace(0.0, 1.0, n_f)
    flux, sky = {}, {}
    for i in range(n_c):
        drift = 0.006 * np.sin(2 * np.pi * (t + i / n_c))  # 6 mmag slow drift, different phase per comp
        m = -2.5 * np.log10(2.0e6) + drift + rng.normal(0, 0.001, n_f)
        flux[f"c{i}"] = 10 ** (-0.4 * m)
        sky[f"c{i}"] = np.full(n_f, 10.0)
    out = night_sigma_sys_from_comps(flux, sky, np.full(n_f, 4.0), k=1.0, gain=1.0e6, read_noise=0.0,
                                     annulus_inner_fwhm=2.7, annulus_outer_fwhm=5.2)
    assert "total scatter" in out["method"]
    assert out["sigma_sys_mag"] > 0.003  # p2p would see only the 1 mmag white noise


def test_check_star_error_includes_scintillation() -> None:
    """CHECK-SCINT-2026-10 (D10 / G08-10): the check K error has the target's scintillation term."""
    src = (SRC / "aperture_photometry_target.py").read_text(encoding="utf-8")
    # G06-39: the check K error is built by the target's own error helper (scintillation included)
    i = src.index("_kerr_rel, _ = _combine_err_with_ensemble_scatter_keyed(")
    assert "sigma_scint_mag=_scint_mag_arr" in src[i:i + 500]


def test_empty_apertures_use_the_science_sky() -> None:
    """SKY-ONE-2026-10: sources masked in the sky image do not bias the empty-aperture noise."""
    from photometry_gate_helpers import measure_empty_aperture_sigma_bkg

    rng = np.random.default_rng(7)
    data = rng.normal(100.0, 5.0, (220, 220))
    yy, xx = np.mgrid[0:220, 0:220]
    src = np.zeros_like(data, dtype=bool)
    for x0, y0 in rng.uniform(20, 200, (40, 2)):
        r2 = (xx - x0) ** 2 + (yy - y0) ** 2
        data += 400.0 * np.exp(-r2 / (2 * 1.5**2))
        src |= r2 < 6.0**2
    sky = np.where(src, np.nan, data)
    clean = rng.normal(100.0, 5.0, (220, 220))
    a, na, ra = measure_empty_aperture_sigma_bkg(data, np.array([]), np.array([]), 3.0, 6.0, 10.0,
                                                 n_apertures=48, min_valid=12, seed=5, sky=sky, covered=None)
    b, nb, rb = measure_empty_aperture_sigma_bkg(clean, np.array([]), np.array([]), 3.0, 6.0, 10.0,
                                                 n_apertures=48, min_valid=12, seed=5, sky=clean, covered=None)
    assert ra == rb == "" and math.isfinite(a) and math.isfinite(b)
    import pytest

    with pytest.raises(ValueError, match="sky image shape"):
        measure_empty_aperture_sigma_bkg(data, np.array([]), np.array([]), 3.0, 6.0, 10.0, seed=5, sky=sky[:10], covered=None)
