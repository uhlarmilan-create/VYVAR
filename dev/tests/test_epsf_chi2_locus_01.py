# -*- coding: ascii -*-
"""EPSF-CHI2-LOCUS-01 / CHI2-FLUX-REF-2026-10: chi2 reference for psf_fit_ok.

chi2 = noise term + PSF-model-mismatch term ~ flux^2: curved in log-log. The reference is a night curve of plain
medians of log10(chi2) in log10(flux) bins, shifted per frame by a plain median offset; no clipping.
Pre-fail (straight-line MAD-clipped locus, 88d6368): bright stars on the curved locus were rejected.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from psf_chi2_locus import (
    MIN_N_BIN,
    MIN_N_LOCUS_FIT,
    Chi2Locus,
    chi2_ok_from_locus,
    fit_chi2_locus,
    locus_residual_sigma,
    reapply_night_locus_to_proc_dir,
)


def _apply(tmp_path: Path, rows: list[dict], night: Chi2Locus | None, n_sigma: float = 5.0):
    """Run the production rule (``reapply_night_locus_to_proc_dir``) on one proc sidecar holding ``rows``."""
    proc = tmp_path / "proc"
    proc.mkdir(exist_ok=True)
    fp = proc / "proc_f0001.csv"
    pd.DataFrame(rows).to_csv(fp, index=False)
    summary = reapply_night_locus_to_proc_dir(proc, n_sigma=n_sigma, night_locus=night)
    out = pd.read_csv(fp).to_dict("records")
    for r in out:
        r["psf_fit_ok"] = bool(r["psf_fit_ok"])
    fr = (summary.get("frames") or [None])[0]
    return out, fr, summary


def _curved_chi2(logf: np.ndarray, *, n_pix: float = 50.0, eps: float = 0.01, sky_var: float = 400.0) -> np.ndarray:
    """chi2 of a PSF fit: n_pix (noise) + (eps * F)^2 / (variance) (model mismatch grows with flux^2)."""
    f = 10**logf
    var = sky_var * n_pix + f
    return n_pix + (eps * f) ** 2 / var * n_pix


def _frame(n: int, seed: int, *, offset_dex: float = 0.0, scatter: float = 0.08, n_out: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    logf = np.concatenate([rng.uniform(2.5, 4.5, size=int(n * 0.85)), rng.uniform(4.5, 6.0, size=n - int(n * 0.85))])
    logc = np.log10(_curved_chi2(logf)) + offset_dex + rng.normal(0.0, scatter, size=n)
    rows = [
        {"catalog_id": f"{1000000000000000000 + i}", "psf_flux": float(10 ** logf[i]), "psf_chi2": float(10 ** logc[i]),
         "psf_converged": True, "psf_fit_ok": False}
        for i in range(n)
    ]
    for j in range(n_out):
        lf = float(rng.uniform(3.0, 5.5))
        rows.append({"catalog_id": f"{9000000000000000000 + j}", "psf_flux": 10**lf,
                     "psf_chi2": float(10 ** (math.log10(_curved_chi2(np.array([lf]))[0]) + offset_dex + 10 * scatter)),
                     "psf_converged": True, "psf_fit_ok": False})
    return rows


def _night(rows: list[dict]) -> Chi2Locus:
    night = fit_chi2_locus([r["psf_flux"] for r in rows], [r["psf_chi2"] for r in rows],
                           converged=[True] * len(rows), source="night")
    assert night is not None
    return night


def _straight_line_ok(rows: list[dict], n_sigma: float = 5.0) -> dict[str, bool]:
    """Pre-fix rule: one straight line log chi2 = a + k log flux (3 x 3-sigma MAD clipping), 5 sigma."""
    x = np.log10([r["psf_flux"] for r in rows]); y = np.log10([r["psf_chi2"] for r in rows])
    keep = np.ones(len(x), bool)
    for _ in range(3):
        k, a = np.polyfit(x[keep], y[keep], 1); res = y - (a + k * x)
        med = np.median(res[keep]); mad = np.median(np.abs(res[keep] - med)); keep = np.abs(res - med) <= 3 * mad
    k, a = np.polyfit(x[keep], y[keep], 1); res = y - (a + k * x)
    sig = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep])))
    return {r["catalog_id"]: bool(res[i] / sig <= n_sigma) for i, r in enumerate(rows)}


def test_t1_bright_stars_on_curved_locus_pass(tmp_path: Path):
    rows = _frame(800, seed=1)
    old = _straight_line_ok(rows)
    bright = [r for r in rows if r["psf_flux"] > 10**5.3]
    assert bright
    assert sum(not old[r["catalog_id"]] for r in bright) > 0.3 * len(bright)  # pre-fix: many rejected
    out, fr, _ = _apply(tmp_path, rows, _night(rows))
    assert fr["source"] == "frame"
    by_id = {str(r["catalog_id"]): r for r in out}
    n_ok = sum(by_id[r["catalog_id"]]["psf_fit_ok"] for r in bright)
    assert n_ok >= 0.97 * len(bright)


def test_t2_outliers_fail_at_every_flux(tmp_path: Path):
    rows = _frame(800, seed=2, n_out=12)
    night = _night([r for r in rows if not r["catalog_id"].startswith("9")])
    out, _, _ = _apply(tmp_path, rows, night)
    for r in out:
        if str(r["catalog_id"]).startswith("9"):
            assert r["psf_fit_ok"] is False
            assert r["psf_chi2_locus_resid_sigma"] > 5.0


def test_t3_nonfinite_chi2_fails(tmp_path: Path):
    assert chi2_ok_from_locus(converged=True, chi2=float("nan"), resid_sigma=0.0) is False
    assert chi2_ok_from_locus(converged=True, chi2=float("inf"), resid_sigma=0.0) is False
    assert chi2_ok_from_locus(converged=False, chi2=5.0, resid_sigma=0.0) is False
    base = _frame(400, seed=3)
    night = _night(base)
    rows = base[:40] + [{"catalog_id": "8000000000000000001", "psf_flux": 1.0e4, "psf_chi2": float("nan"), "psf_converged": True,
                          "psf_fit_ok": False}]
    out, _, _ = _apply(tmp_path, rows, night)
    by_id = {str(r["catalog_id"]): r for r in out}
    assert by_id["8000000000000000001"]["psf_fit_ok"] is False


def test_t4a_frame_offset_absorbs_global_chi2_shift(tmp_path: Path):
    night = _night(_frame(1500, seed=10))
    frame = _frame(300, seed=11, offset_dex=0.4)  # seeing-like global shift, 5 x the scatter
    out, fr, _ = _apply(tmp_path, frame, night)
    assert fr["source"] == "frame"
    assert abs(fr["offset"] - 0.4) < 0.05
    assert sum(r["psf_fit_ok"] for r in out) >= 0.97 * len(frame)


def test_t4b_small_frame_uses_night_curve_and_no_night_curve_admits_nothing(tmp_path: Path):
    night = _night(_frame(1500, seed=12))
    tiny = _frame(MIN_N_LOCUS_FIT - 2, seed=13)
    _, fr, _ = _apply(tmp_path, tiny, night)
    assert fr["source"] == "night" and fr["offset"] == 0.0
    (tmp_path / "b").mkdir()
    out2, fr2, meta2 = _apply(tmp_path / "b", tiny, None)  # fewer than MIN_N_BIN stars in the night
    assert len(tiny) < MIN_N_BIN
    assert meta2["error"] == "night_locus_unavailable"
    assert fr2 is None
    assert all(r["psf_fit_ok"] is False for r in out2)


def test_t4c_no_clipping_and_no_night_constants():
    src = (Path(__file__).resolve().parents[1].parent / "src_py" / "psf_chi2_locus.py").read_text(encoding="ascii")
    for tok in ("_CLIP_ROUNDS", "_CLIP_K", "sigma_clip", "0.002668", "0.336561", "30076", "3343"):
        assert tok not in src, tok


def test_t5_residual_uses_scatter_of_the_star_flux_bin():
    rng = np.random.default_rng(5)
    lf = rng.uniform(3.0, 6.0, 6000)
    sc = np.where(lf < 4.5, 0.05, 0.3)  # bright bins scatter more
    lc = np.log10(_curved_chi2(lf)) + rng.normal(0, 1, lf.size) * sc
    night = fit_chi2_locus(10**lf, 10**lc, source="night")
    assert night is not None
    r_faint = locus_residual_sigma(10**4.0, 10 ** (np.log10(_curved_chi2(np.array([4.0]))[0]) + 0.3), night)
    r_bright = locus_residual_sigma(10**5.7, 10 ** (np.log10(_curved_chi2(np.array([5.7]))[0]) + 0.3), night)
    assert r_faint > 4.0 and r_bright < 2.0


def test_finalize_night_locus_refreshes_n_ok_for_invariant(tmp_path):
    """Night finalize must run before INV-PSF-FRAME-01 (merge leaves n_ok=0)."""
    from psf_chi2_locus import finalize_night_locus_for_inv_psf_frame_01
    import pandas as pd

    proc = tmp_path / "proc_BO_CVn_Light_001.csv"
    rows = []
    for i in range(40):
        f = 10 ** (4.0 + i * 0.005)
        c = 10 ** (-3.9 + 1.06 * math.log10(f))
        rows.append(
            {
                "catalog_id": str(i),
                "psf_flux": f,
                "psf_chi2": c,
                "psf_converged": True,
                "psf_fit_ok": False,
                "psf_chi2_locus_source": "none_need_night",
            }
        )
    pd.DataFrame(rows).to_csv(proc, index=False)
    recs = [{"frame_name": "BO_CVn_Light_001.fits", "n_fit": 40, "n_ok": 0}]
    summary = finalize_night_locus_for_inv_psf_frame_01(tmp_path, recs, n_sigma=5.0)
    assert summary.get("n_kept_frame", 0) >= 1 or summary.get("n_rewritten", 0) >= 1
    assert recs[0]["n_ok"] == 40
    assert summary.get("n_ok_refreshed_records") == 1


def test_d2_filter_never_fitted_comps_from_ensemble():
    """Pre-fail: never-fitted pin kills cov; post-pass: structural drop restores."""
    from psf_internal_lc import (
        comps_with_psf_measurement_on_night,
        filter_ensemble_to_psf_measured,
    )
    import pandas as pd

    stack = pd.DataFrame(
        {
            "catalog_id": ["111", "111", "222", "222", "333", "333"],
            "psf_flux": [100.0, 110.0, 200.0, 210.0, float("nan"), float("nan")],
            "psf_chi2": [1.0, 1.1, 2.0, 2.1, float("nan"), float("nan")],
            "psf_fit_ok": [True, True, True, True, False, False],
        }
    )
    measured = comps_with_psf_measurement_on_night(stack)
    assert measured == {"111", "222"}
    assert "333" not in measured
    # Pre-fix style: INV-PIN would drop every epoch because C never has flux.
    pins = ["111", "222", "333"]
    kept, dropped = filter_ensemble_to_psf_measured(pins, measured)
    assert kept == ["111", "222"]
    assert dropped == ["333"]
